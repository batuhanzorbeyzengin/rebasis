"""Generate the reference catalogues.

``docs/reference/events.md``, ``docs/reference/errors.md`` and
``docs/reference/profiles.md`` are **generated, never hand-written.** A
hand-maintained catalogue drifts from the code within one release, and a
catalogue that lies is worse than none: users grep these pages to find out what
an event carries, what an error code means, or which prefix a model wants.

It lives in ``report`` rather than in ``observability`` because it reads from
three layers — errors, the event catalogue and the profile table — and a module
that reaches across the whole package belongs at the top of it, not at the
bottom. Putting it under ``observability`` inverted the layer contract the
moment it learned about encoding profiles.

Run with ``just docs-gen`` or::

    python -m rebasis.report.catalog --out docs/reference
"""

from __future__ import annotations

import argparse
import inspect
import logging
import sys
from pathlib import Path

from rebasis import errors as errors_module
from rebasis.embed import known_profiles
from rebasis.observability.events import EVENT_CATALOG, Events

__all__ = ["render_errors_markdown", "render_events_markdown", "render_profiles_markdown"]

_HEADER = "<!-- GENERATED FILE — edit the source module, not this page. -->\n"


def render_events_markdown() -> str:
    """Render the event catalogue as Markdown."""
    lines = [
        _HEADER,
        "# Event catalogue",
        "",
        "Stable event names emitted by rebasis. The message text of an",
        "event may change between releases; **the event name may not**. Users grep",
        "for these and dashboards group by them.",
        "",
        "Naming: `<module>.<object>.<verb-past-tense>`, dot-separated, lowercase.",
        "",
        "`Audited` marks the small subset that also produces an audit record.",
        "Log and audit trail are different things: a log may be lost, sampled",
        "and reformatted; an audit record may not.",
        "",
        "| Event | Level | Audited | Fields | Description |",
        "|---|---|---|---|---|",
    ]
    for event in sorted(Events, key=str):
        spec = EVENT_CATALOG[event]
        fields = ", ".join(f"`{f}`" for f in spec.fields) or "—"
        level = logging.getLevelName(spec.level)
        audited = "**yes**" if spec.audited else "no"
        description = spec.description.replace("\n", " ").strip()
        lines.append(f"| `{event}` | {level} | {audited} | {fields} | {description} |")
    lines.append("")
    return "\n".join(lines)


def render_errors_markdown() -> str:
    """Render the error-code catalogue as Markdown."""
    families: dict[str, list[type[errors_module.RebasisError]]] = {}
    for name in errors_module.__all__:
        obj = getattr(errors_module, name)
        if not (inspect.isclass(obj) and issubclass(obj, errors_module.RebasisError)):
            continue
        family = obj.code[:5]  # RB-E1, RB-E2, ...
        families.setdefault(family, []).append(obj)

    lines = [
        _HEADER,
        "# Error codes",
        "",
        "Every rebasis error carries a code, a cause and a next step.",
        "",
        "**Codes are stable.** A code is never removed or repurposed — only marked",
        "deprecated — because user scripts and dashboards depend on them.",
        "",
        "`Transient` errors are retried with exponential backoff, at most three",
        "times. Permanent errors are never retried: a silently retried permanent",
        "failure is the most common cause of undiagnosable slowness.",
        "",
        "Exit codes: `0` success · `1` unexpected · `2` usage or configuration ·",
        "`3` domain error · `130` interrupted.",
        "",
    ]
    for family in sorted(families):
        members = sorted(families[family], key=lambda c: c.code)
        lines.extend(
            [
                f"## {family}xxx",
                "",
                "| Code | Class | Transient | Exit | What it means |",
                "|---|---|---|---|---|",
            ]
        )
        for cls in members:
            doc = (cls.__doc__ or "").strip().split("\n")[0]
            transient = "**yes**" if cls.transient else "no"
            lines.append(
                f"| `{cls.code}` | `{cls.__name__}` | {transient} | {cls.exit_code} | {doc} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_profiles_markdown() -> str:
    """Render the encoding-profile table as Markdown.

    Generated from the shipped table for the same reason as the other two: a
    hand-copied prefix is exactly the mistake this table exists to prevent.
    """
    lines = [
        _HEADER,
        "# Encoding profiles",
        "",
        "The models rebasis knows without being told. Values come from",
        "each model card's official retrieval instructions, not from guesswork.",
        "",
        "**Why this table exists.** Many retrieval models encode a query",
        "differently from a document — usually with a prefix. Getting that wrong",
        "produces no error and no warning; it only lowers quality, which is the",
        "hardest kind of failure to attribute. So the prefix lives in the",
        "profile, the profile is fingerprinted, and the fingerprint blocks an",
        "adapter from loading against an index it was not built for.",
        "",
        "**A model not listed here still works.** Pass `--dim`, and",
        "`--query-prefix` / `--document-prefix` if it is asymmetric. rebasis will",
        "not guess a prefix: being asked once is cheaper than debugging a quality",
        "drop.",
        "",
        "`Asymmetric` marks the models where the two prefixes differ. For those,",
        "`auto` measures both a shared adapter and a query-specific one and keeps",
        "whichever scores better on the held-out set — M0 measured the mean",
        "difference at -0.003, with the sign varying by model pair.",
        "",
        "| Model | Dim | Asymmetric | Query prefix | Document prefix | Pooling |",
        "|---|---|---|---|---|---|",
    ]
    for model_id, profile in sorted(known_profiles().items()):
        query = f"`{profile.query_prefix}`" if profile.query_prefix else "—"
        document = f"`{profile.document_prefix}`" if profile.document_prefix else "—"
        asymmetric = "no" if profile.symmetric else "**yes**"
        lines.append(
            f"| `{model_id}` | {profile.dim} | {asymmetric} | {query} | "
            f"{document} | {profile.pooling} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Write all three catalogues into the given directory."""
    parser = argparse.ArgumentParser(prog="rebasis-catalog")
    parser.add_argument("--out", default="docs/reference", help="output directory")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, render in (
        ("events.md", render_events_markdown),
        ("errors.md", render_errors_markdown),
        ("profiles.md", render_profiles_markdown),
    ):
        path = out_dir / name
        path.write_text(render(), encoding="utf-8")
        sys.stdout.write(f"wrote {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
