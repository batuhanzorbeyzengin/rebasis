"""Every file write under ``src/`` goes through ``storage/atomic.py``.

`rebasis.storage.atomic`'s own docstring has claimed since it was written that
"direct ``open(path, "w")`` elsewhere under ``src/`` is rejected by a test",
and gives the reason: ENOSPC while overwriting in place truncates the old
content before the new content can be written, and the user's data is simply
gone. `pyproject.toml`'s `banned-api` section repeats the ban in a comment.

**Neither of them enforced it.** The lint rule bans `os.rename` and nothing
else, and no test scanned the source tree. The rule was real, the reasoning was
right, and the only thing holding it up was that nobody had written a direct
`open(..., "w")` yet. This is the test both of those texts were describing.

Scanned with `ast` rather than a regular expression, because a grep for
`open(` matches the word in a docstring explaining why not to use it — which
this file is full of, and which would make the check fail on its own
documentation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "src" / "rebasis"

#: Modes that destroy what is already at the path, or append to it.
#:
#: ``+`` is in here for ``r+``: it opens for reading *and* writing without
#: truncating, which is the one shape that looks harmless and is not — a partial
#: overwrite of a live file is the failure this whole discipline exists to
#: prevent, and it leaves no temporary file behind to notice afterwards.
DESTRUCTIVE = frozenset("wax+")

#: Methods that write a whole file in one call, no mode argument to inspect.
WHOLE_FILE_WRITES = frozenset({"write_text", "write_bytes"})

#: Where a direct write is allowed, keyed by ``path::enclosing function``.
#:
#: Keyed on the function rather than the file so that a *second* write appearing
#: in an already-listed module is still caught, and on the function rather than
#: the line so that the key survives ordinary editing. Every entry carries the
#: argument for it; an exemption whose reason is not written down is how a rule
#: like this stops meaning anything.
ALLOWED: dict[str, str] = {
    "storage/locks.py::state_lock": (
        "The lock-info sidecar, written after the lock is already held. It is "
        "advisory metadata about a running process, not user data: a truncated "
        "one makes `rebasis status` unable to say who holds the lock, which is "
        "recoverable by looking at the lock itself. Routing it through "
        "atomic_write would also mean taking a lock to describe a lock."
    ),
    "report/catalog.py::main": (
        "`just docs-gen` regenerating `docs/reference/` in this repository. A "
        "developer tool that writes into the source tree it was run from and "
        "never touches an index, a manifest or anything a user owns."
    ),
    "storage/shadow.py::append": (
        "The shadow copy's append path, and the one place where atomic write "
        "would be the wrong tool rather than an inconvenience. A shadow grows "
        "batch by batch through a migration; writing it atomically would mean "
        "copying the whole shadow to a temporary file and replacing it on every "
        "batch, so the cost of protecting the copy would grow with the copy. "
        "The failure atomic write defends against is handled here instead and "
        "more precisely: the file is truncated to the record count the manifest "
        "records before anything is appended, so a partial append left by a "
        "crash is discarded rather than built upon, and the manifest is what "
        "decides how much of the file is real. `open` is reached through a "
        "conditional mode (`r+b` where the file exists, `wb` where it does "
        "not), which is why this shows as a computed mode rather than a "
        "literal one."
    ),
    "observability/logging.py::configure_logging": (
        "The optional log file, opened for append. The module already states "
        "the argument at the call site: a log is explicitly allowed to be "
        "lossy, and durability guarantees belong to the audit trail rather "
        "than to logs. An append also cannot truncate what is already there, "
        "so the ENOSPC failure this discipline exists to prevent does not "
        "apply — the worst case is a short final line."
    ),
}


def _enclosing_functions(tree: ast.Module) -> dict[int, str]:
    """Map every line in a module to the innermost function that contains it."""
    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            end = node.end_lineno or node.lineno
            # Innermost wins: an outer function is walked first only by
            # accident of traversal order, so a nested definition overwrites
            # the lines it actually owns.
            for line in range(node.lineno, end + 1):
                if line not in owner or node.lineno > tree.body[0].lineno:
                    owner[line] = node.name
    return owner


#: Sentinel for "a mode was passed and this cannot read it".
#:
#: Distinct from `"r"`, which is what *no* mode argument means. Conflating the
#: two is the bug the first version of this file shipped with: it flagged every
#: `path.open("rb")` in the codebase as a possible write, because it looked for
#: the mode at the builtin's argument index and found nothing there.
UNREADABLE = "<computed>"


def _mode_of(call: ast.Call, *, builtin: bool) -> str:
    """The mode a call to `open` was given.

    The two spellings put it in different places, and that is the whole reason
    this takes a flag: the builtin is ``open(path, mode)`` and `Path.open` is
    ``path.open(mode)``, so the positional index differs by one. Both default to
    ``"r"`` when it is absent, which is the safe answer and not an unknown one.
    """
    index = 1 if builtin else 0
    if len(call.args) > index:
        argument = call.args[index]
        return str(argument.value) if isinstance(argument, ast.Constant) else UNREADABLE
    for keyword in call.keywords:
        if keyword.arg == "mode":
            value = keyword.value
            return str(value.value) if isinstance(value, ast.Constant) else UNREADABLE
    return "r"


def _direct_writes(path: Path) -> list[tuple[int, str, str]]:
    """Every direct write in one module, as (line, function, what it was)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = _enclosing_functions(tree)
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        builtin = isinstance(node.func, ast.Name)
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name == "open":
            mode = _mode_of(node, builtin=builtin)
            # A mode this cannot read is a mode it cannot clear, so a computed
            # one is flagged: the escape hatch has to be deliberate rather than
            # a side effect of indirection.
            if mode == UNREADABLE or set(mode) & DESTRUCTIVE:
                spelling = "open" if builtin else ".open"
                where = owner.get(node.lineno, "<module>")
                found.append((node.lineno, where, f"{spelling}({mode!r})"))
        elif name in WHOLE_FILE_WRITES:
            found.append((node.lineno, owner.get(node.lineno, "<module>"), f".{name}()"))
    return found


@pytest.mark.unit
def test_no_module_writes_a_file_without_going_through_atomic() -> None:
    """The claim `storage/atomic.py` makes about itself, made true."""
    # atomic.py is the implementation. It opens a temporary file and replaces
    # the target; excluding it is the point rather than an exception to it.
    modules = [p for p in sorted(SOURCE.rglob("*.py")) if p.name != "atomic.py"]
    assert modules, "found no source to scan, which means the path above is wrong"

    unexplained: list[str] = []
    for path in modules:
        relative = path.relative_to(SOURCE).as_posix()
        for line, function, what in _direct_writes(path):
            key = f"{relative}::{function}"
            if key not in ALLOWED:
                unexplained.append(f"{relative}:{line} in {function}() — {what}")

    assert not unexplained, (
        "These write a file without going through rebasis.storage.atomic:\n  "
        + "\n  ".join(unexplained)
        + "\n\nUse atomic_write_bytes/atomic_write_text. If the write genuinely "
        "cannot be atomic — it is not user data, or it is a developer tool — add "
        "it to ALLOWED in this file with the argument for it."
    )


@pytest.mark.unit
def test_every_exemption_still_exists() -> None:
    """An allowlist nobody prunes becomes a list of places to write freely."""
    live = {
        f"{path.relative_to(SOURCE).as_posix()}::{function}"
        for path in sorted(SOURCE.rglob("*.py"))
        if path.name != "atomic.py"
        for _, function, _ in _direct_writes(path)
    }
    stale = sorted(set(ALLOWED) - live)
    assert not stale, (
        "These exemptions no longer describe a write that exists; remove them "
        f"from ALLOWED: {stale}"
    )
