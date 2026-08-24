"""Nothing internal reaches a terminal.

Two things must not appear in output a user sees.

**Section references.** Comments and docstrings under `src/` cite the project's
private design notes freely — they are its reasoning, and a maintainer should be
able to follow them. But those notes are not in the repository, so a section
reference in `--help` points a user at something they cannot read.

**Markup.** A docstring is rendered as help text verbatim; RST or Markdown that
was written for a docs generator arrives as punctuation.

And one thing must not *dis*appear. `rich` reads `[otel]` as a style tag, so an
install hint printed through it silently loses the extra it exists to name --
`pip install "rebasis[otel]"` reaches the user as `pip install "rebasis"`, which
tells them to install what they already have.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from rebasis.cli import app

pytestmark = pytest.mark.unit

runner = CliRunner()
WIDE = {"COLUMNS": "200"}


def every_command() -> list[list[str]]:
    """Every invocation that renders help, including subcommand groups."""
    root = get_command(app)
    invocations = [["--help"]]
    for name, command in root.commands.items():
        invocations.append([name, "--help"])
        invocations.extend([name, sub, "--help"] for sub in getattr(command, "commands", {}))
    return invocations


def as_id(args: list[str]) -> str:
    return " ".join(args)


@pytest.mark.parametrize("args", every_command(), ids=as_id)
def test_help_carries_no_section_references(args: list[str]) -> None:
    output = runner.invoke(app, args, env=WIDE).output

    offenders = [line.strip() for line in output.splitlines() if "§" in line]
    assert not offenders, f"`rebasis {' '.join(args)}` shows: {offenders[:2]}"


@pytest.mark.parametrize("args", every_command(), ids=as_id)
def test_help_carries_no_markup(args: list[str]) -> None:
    """``:func:`x``` and ``**bold**`` are for a docs generator, not a terminal."""
    output = runner.invoke(app, args, env=WIDE).output

    patterns = (r":[a-z]+:`", r"``", r"\*\*\w")
    offenders = [
        line.strip() for line in output.splitlines() if any(re.search(p, line) for p in patterns)
    ]
    assert not offenders, f"`rebasis {' '.join(args)}` shows markup: {offenders[:2]}"


def test_doctor_carries_no_section_references() -> None:
    """`doctor` is what a confused user runs first, so its output is the last
    place an unreadable reference belongs."""
    output = runner.invoke(app, ["doctor"], env=WIDE).output

    assert "§" not in output


def test_the_doctor_telemetry_hint_keeps_its_extra() -> None:
    """Rendered directly rather than by running `doctor`.

    Whether `doctor` prints this line at all depends on whether the otel extra
    happens to be installed, and a test that only runs on machines without it
    is a test that never runs in CI.
    """
    from rich.console import Console

    from rebasis.cli.doctor import _telemetry_row

    console = Console(file=io.StringIO(), width=200, force_terminal=False)
    console.print(_telemetry_row({"available": False, "enabled": False}))

    assert "rebasis[otel]" in console.file.getvalue()  # type: ignore[attr-defined]


def test_an_error_hint_keeps_its_extra() -> None:
    """The most useful sentence rebasis prints is an install hint, and it goes
    through rich by the error panel.

    `MissingDependency` is raised the moment someone points rebasis at a store
    whose client they have not installed, and its hint names the extra to
    install, in brackets.
    """
    from rebasis.cli._common import err_console
    from rebasis.errors import MissingDependency

    error = MissingDependency(
        "The Chroma backend needs chromadb.",
        hint='Install it with `pip install "rebasis[chroma]"`.',
        context={"store_backend": "chroma"},
    )

    with err_console.capture() as captured:
        from rebasis.cli._common import _render

        _render(error)

    assert "rebasis[chroma]" in captured.get()


def test_an_error_does_not_lose_brackets_from_user_data() -> None:
    """A path, an id or a URI with a bracket in it is data, not markup."""
    from rebasis.cli._common import _render, err_console
    from rebasis.errors import CollectionNotFound

    error = CollectionNotFound(
        "No collection named 'notes[2026]'.",
        hint="Check the fragment in the store URI.",
        context={"path": "~/vaults/my [odd] vault/db"},
    )

    with err_console.capture() as captured:
        _render(error)

    output = captured.get()
    assert "notes[2026]" in output
    assert "[odd]" in output


def test_a_deliberate_exit_is_not_reported_as_a_crash() -> None:
    """`typer.Exit` subclasses `RuntimeError`.

    So the top-level `except Exception` caught every intentional non-zero exit
    and rendered it as an internal bug, with a pre-filled issue link and the
    wrong exit code. Answering "no" to the `migrate` confirmation told the user
    rebasis had crashed.
    """
    import typer

    from rebasis.cli._common import handle_errors

    @handle_errors
    def declines() -> None:
        raise typer.Exit(code=2)

    with pytest.raises(typer.Exit) as caught:
        declines()

    assert caught.value.exit_code == 2


def test_no_printed_line_carries_a_section_reference() -> None:
    """The ban covers everything the CLI prints, not only `--help`.

    Checked in the source because most of these lines only appear part-way
    through a real run against a real store. `rollback` printed a section
    reference in its preview for exactly this reason: the tests looked at help
    text and at `doctor`, and that line is neither.
    """
    offenders = [
        f"{path.name}:{number}: {line.strip()}"
        for path in (Path(__file__).parents[2] / "src" / "rebasis" / "cli").glob("*.py")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if "§" in line and ("console.print" in line or "err_console.print" in line)
    ]

    assert not offenders, f"a section reference reaches the terminal: {offenders}"


def test_every_command_has_a_description() -> None:
    """A command with no help is a command nobody finds."""
    root = get_command(app)

    for name, command in root.commands.items():
        assert command.help or command.short_help, f"{name} has no description"
