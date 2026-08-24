"""Lifecycle and cleanup.

**``gc`` defaults to ``--dry-run``.** It lists what it would remove and how much
that would free; actually removing anything requires ``--apply``. A garbage
collector that deletes by default *is* the data-loss class it is supposed to
prevent.

Retention, by object:

| Object | Policy |
|---|---|
| Audit records | **Never** removed automatically |
| Adapters (`.rbs`) | Never removed automatically |
| Shadow copies | Explicit confirmation, unless the job already spent one |
| Reports | Candidate after 90 days |
| Sample cache | Candidate after 30 days |
| `.tmp.*` leftovers | Candidate after 24 hours (crashed processes) |
| `.bak` files | Newest three kept |

The asymmetry is what makes this simple: **everything critical is small.** The
only large object is the shadow copy, and it is temporary. Being generous with
the small things costs almost nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rebasis.observability import Events, get_logger
from rebasis.storage.layout import CACHE_DIR, REPORTS_DIR, SHADOW_DIR

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["GCCandidate", "GCPlan", "plan_gc"]

log = get_logger(__name__)

DAY = 86_400.0

#: Retention, in days, per category.
RETENTION = {
    "reports": 90.0,
    "cache": 30.0,
    "tmp": 1.0,
}

#: Backup generations kept per file.
KEEP_BACKUPS = 3

#: Categories that are never removed without an explicit, per-object request.
PROTECTED = frozenset({"audit", "adapters", "shadow"})

#: Items listed per category before the listing is summarised.
_PREVIEW = 3


@dataclass(slots=True)
class GCCandidate:
    """One thing that could be removed."""

    path: Path
    category: str
    size_bytes: int
    age_days: float
    reason: str
    requires_confirmation: bool = False


@dataclass(slots=True)
class GCPlan:
    """What ``gc`` would do."""

    candidates: list[GCCandidate] = field(default_factory=list)
    protected: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        """How much removing everything listed would free."""
        return sum(c.size_bytes for c in self.candidates)

    def render(self) -> str:
        """The dry-run listing."""
        if not self.candidates:
            return "Nothing to collect."

        lines = ["Garbage collection plan (dry run — nothing has been removed)", ""]
        by_category: dict[str, list[GCCandidate]] = {}
        for candidate in self.candidates:
            by_category.setdefault(candidate.category, []).append(candidate)

        for category, items in sorted(by_category.items()):
            size = sum(i.size_bytes for i in items)
            # Shadows can now be a mix: one named on the command line, one left
            # by a rollback. Reading the flag off the first item alone labelled
            # the whole category by whichever happened to sort first.
            confirm = (
                " [needs --i-understand]" if any(i.requires_confirmation for i in items) else ""
            )
            lines.append(f"  {category:<12} {len(items):>4} items  {_human(size):>10}{confirm}")
            lines.extend(f"      {item.path.name}  ({item.reason})" for item in items[:_PREVIEW])
            if len(items) > _PREVIEW:
                lines.append(f"      ... and {len(items) - _PREVIEW} more")

        lines.extend(["", f"  Would free {_human(self.total_bytes)}"])
        if self.protected:
            lines.extend(
                [
                    "",
                    "  Never collected automatically: " + ", ".join(sorted(set(self.protected))),
                ]
            )
        lines.extend(["", "  Run with --apply to remove."])
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, for the audit record."""
        return {
            "count": len(self.candidates),
            "total_bytes": self.total_bytes,
            "categories": sorted({c.category for c in self.candidates}),
        }


def plan_gc(
    state_dir: Path | str,
    *,
    include_shadows: Iterable[str] = (),
    spent_shadows: Iterable[str] = (),
    now: float | None = None,
) -> GCPlan:
    """Work out what could be removed.

    Args:
        state_dir: The ``.rebasis/`` directory.
        include_shadows: Job ids whose shadow copies the user explicitly asked to
            remove. Shadows are never listed without being named, because
            removing one silently destroys the ability to roll that job back.
        spent_shadows: Job ids whose rollback has already happened. Their shadow
            is the largest thing rebasis leaves on disk and it can never be used
            again — ``rolled_back`` is terminal — so it is listed unprompted and
            without the confirmation the others need. There is nothing left to
            protect. Which jobs those are is a question for the manifest, and
            ``storage`` sits below it in the layer contract; the caller looks it
            up and passes the ids down.
        now: Current time, for testing.
    """
    root = Path(state_dir)
    plan = GCPlan(protected=["audit records", "adapters (.rbs)"])
    if not root.exists():
        return plan

    current = now if now is not None else time.time()

    plan.candidates.extend(
        _scan(root / REPORTS_DIR, "reports", RETENTION["reports"], current, "older than 90 days")
    )
    plan.candidates.extend(
        _scan(root / CACHE_DIR, "cache", RETENTION["cache"], current, "older than 30 days")
    )
    plan.candidates.extend(
        _scan_pattern(root, "*.tmp*", "tmp", RETENTION["tmp"], current, "left by a crashed process")
    )
    plan.candidates.extend(_old_backups(root, current))

    named = list(dict.fromkeys(include_shadows))
    spent = [j for j in dict.fromkeys(spent_shadows) if j not in named]

    for job_id in named:
        candidate = _shadow_candidate(
            root, job_id, current, f"explicitly requested for job {job_id}", confirm=True
        )
        if candidate is not None:
            plan.candidates.append(candidate)

    for job_id in spent:
        candidate = _shadow_candidate(
            root,
            job_id,
            current,
            "job already rolled back — the copy cannot be used again",
            confirm=False,
        )
        if candidate is not None:
            plan.candidates.append(candidate)

    return plan


def _shadow_candidate(
    root: Path, job_id: str, now: float, reason: str, *, confirm: bool
) -> GCCandidate | None:
    directory = root / SHADOW_DIR / job_id
    if not directory.exists():
        return None
    return GCCandidate(
        path=directory,
        category="shadow",
        size_bytes=sum(f.stat().st_size for f in directory.rglob("*") if f.is_file()),
        age_days=(now - directory.stat().st_mtime) / DAY,
        reason=reason,
        requires_confirmation=confirm,
    )


def apply_gc(plan: GCPlan, *, confirmed: bool = False) -> int:
    """Remove what the plan lists, and return the bytes freed.

    Args:
        plan: What :func:`plan_gc` produced.
        confirmed: Required for anything marked ``requires_confirmation`` —
            shadow copies, whose removal makes a job permanently irreversible.
    """
    import shutil

    freed = 0
    for candidate in plan.candidates:
        if candidate.requires_confirmation and not confirmed:
            continue
        try:
            if candidate.path.is_dir():
                shutil.rmtree(candidate.path)
            else:
                candidate.path.unlink()
        except OSError:
            continue
        freed += candidate.size_bytes

    log.info(Events.STORAGE_GC_COMPLETED, count=len(plan.candidates))
    return freed


def _scan(
    directory: Path, category: str, max_age_days: float, now: float, reason: str
) -> list[GCCandidate]:
    if not directory.exists():
        return []
    return [
        GCCandidate(
            path=path,
            category=category,
            size_bytes=path.stat().st_size,
            age_days=(now - path.stat().st_mtime) / DAY,
            reason=reason,
        )
        for path in directory.rglob("*")
        if path.is_file() and (now - path.stat().st_mtime) / DAY > max_age_days
    ]


def _scan_pattern(  # noqa: PLR0913, PLR0917 - see _scan
    root: Path, pattern: str, category: str, max_age_days: float, now: float, reason: str
) -> list[GCCandidate]:
    return [
        GCCandidate(
            path=path,
            category=category,
            size_bytes=path.stat().st_size,
            age_days=(now - path.stat().st_mtime) / DAY,
            reason=reason,
        )
        for path in root.rglob(pattern)
        if path.is_file() and (now - path.stat().st_mtime) / DAY > max_age_days
    ]


def _old_backups(root: Path, now: float) -> list[GCCandidate]:
    """Backups beyond the newest :data:`KEEP_BACKUPS` generations."""
    groups: dict[str, list[Path]] = {}
    for path in root.rglob("*.bak.*"):
        if path.is_file():
            groups.setdefault(path.name.rsplit(".bak.", 1)[0], []).append(path)

    return [
        GCCandidate(
            path=path,
            category="backups",
            size_bytes=path.stat().st_size,
            age_days=(now - path.stat().st_mtime) / DAY,
            reason=f"beyond the newest {KEEP_BACKUPS} generations",
        )
        for paths in groups.values()
        for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[KEEP_BACKUPS:]
    ]


_KIB = 1024


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < _KIB or unit == "GB":
            return f"{size:.0f} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= _KIB
    return f"{size:.0f} B"
