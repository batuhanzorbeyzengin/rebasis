"""Shared CLI plumbing.

One ``@handle_errors`` decorator turns every :class:`RebasisError` into a
rendered panel and the right exit code. Exit codes are a **contract** for script
users::

    0    success
    1    unexpected
    2    usage or configuration
    3    domain error
    130  interrupted

An unexpected error prints a pre-filled GitHub issue link carrying the version,
Python, OS and command — and **no PII**.
"""

from __future__ import annotations

import contextlib
import functools
import platform
import sys
import urllib.parse
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from rebasis.errors import EXIT_UNEXPECTED, ConfigError, RebasisError, UserAbort

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

__all__ = [
    "confirm",
    "console",
    "count_progress",
    "err_console",
    "handle_errors",
    "interactive",
    "step_progress",
    "verbosity_to_level",
]

#: Reports go to stdout so they can be piped; diagnostics and progress go to
#: stderr so they do not corrupt that pipe. Progress belongs on stderr for the
#: same reason `--json` exists: `rebasis probe --json | jq` has to receive JSON
#: and nothing else, while the person watching still sees the run move.
console = Console()
err_console = Console(stderr=True)

_ISSUE_URL = "https://github.com/batuhanzorbeyzengin/rebasis/issues/new"


def interactive() -> bool:
    """Whether a person is on the other end of stdin.

    Every prompt in this CLI is gated on this. A prompt that fires in a script
    or a CI job cannot be answered, and the failure it produces is unreadable:
    `typer.confirm` raises `Abort`, which is not `typer.Exit`, so it reached the
    unexpected-error boundary and told the user their normal, correct
    invocation was a bug in rebasis.
    """
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):  # detached or closed stdin
        return False


def confirm(question: str, *, assume_yes: bool, no_input: bool) -> bool:
    """Ask before something irreversible, or explain why it cannot be asked.

    Three outcomes, and the third is the one that was missing: `--yes` proceeds,
    an interactive terminal is prompted, and anything else is a configuration
    error naming the flag that fixes it.
    """
    if assume_yes:
        return True
    if no_input or not interactive():
        raise ConfigError(
            f"{question} needs an answer, and stdin is not a terminal.",
            hint="Pass --yes to proceed without asking. Nothing has been written.",
        )
    return bool(typer.confirm(question, default=False))


class _Steps:
    """Named stages, announced as each begins.

    A single spinner that says one thing for five minutes cannot distinguish
    working from wedged. Naming the stage it is on can: the text changing is the
    signal, and on a pipe the timestamps in the log are.
    """

    def __init__(self, status: Any) -> None:
        self._status = status

    def stage(self, text: str) -> None:
        """Announce the stage now starting."""
        if self._status is not None:
            self._status.update(f"{text}…")
        else:
            err_console.print(f"[dim]{escape(text)}…[/dim]")


@contextlib.contextmanager
def step_progress(first: str) -> Iterator[_Steps]:
    """Stage-by-stage progress for a pipeline whose total is not countable.

    Animated on a terminal, one line per stage anywhere else — a spinner
    redirected to a file is thousands of control characters and no information.
    """
    if err_console.is_terminal:
        with err_console.status(f"{first}…") as status:
            yield _Steps(status)
    else:
        err_console.print(f"[dim]{escape(first)}…[/dim]")
        yield _Steps(None)


class _Counter:
    """``X of Y``, the pattern for work whose size is known up front."""

    def __init__(self, progress: Any, task: Any, total: int, label: str) -> None:
        self._progress = progress
        self._task = task
        self._total = total
        self._label = label
        self._done = 0
        self._announced = 0

    def advance(self, count: int) -> None:
        """Record ``count`` more units of finished work."""
        self._done += count
        if self._progress is not None:
            self._progress.update(self._task, completed=self._done)
            return
        # Off a terminal: a line per decile. Enough to see it moving in a CI
        # log, few enough not to bury everything else in it.
        decile = (self._done * 10) // max(self._total, 1)
        if decile > self._announced:
            self._announced = decile
            err_console.print(f"[dim]{escape(self._label)} {self._done:,}/{self._total:,}[/dim]")


@contextlib.contextmanager
def count_progress(total: int, label: str) -> Iterator[_Counter]:
    """A progress bar over a known total, or deciles when there is no terminal."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    if not err_console.is_terminal:
        err_console.print(f"[dim]{escape(label)} 0/{total:,}[/dim]")
        yield _Counter(None, None, total, label)
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=err_console,
        transient=True,
    ) as progress:
        task = progress.add_task(label, total=total)
        yield _Counter(progress, task, total, label)


def verbosity_to_level(verbose: int, quiet: bool) -> int | None:
    """Translate ``-v`` / ``-vv`` / ``-q`` into a log level.

    Returns ``None`` when the user said nothing, so the environment default
    stays in charge.
    """
    import logging

    if quiet:
        return logging.ERROR
    if verbose >= 2:  # noqa: PLR2004 - -vv is a documented flag, not a magic number
        return logging.DEBUG
    if verbose == 1:
        return logging.INFO
    return None


def handle_errors[T](func: Callable[..., T]) -> Callable[..., T]:
    """Render errors and exit with the documented code."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except UserAbort:
            err_console.print("[yellow]Interrupted.[/yellow]")
            raise typer.Exit(code=UserAbort.exit_code) from None
        except KeyboardInterrupt:
            err_console.print("[yellow]Interrupted.[/yellow]")
            raise typer.Exit(code=UserAbort.exit_code) from None
        except RebasisError as exc:
            _render(exc)
            raise typer.Exit(code=exc.exit_code) from None
        except typer.Exit:
            # `typer.Exit` subclasses `RuntimeError`, so without this it is
            # caught below and rendered as a crash. Every deliberate non-zero
            # exit went out under "This is a bug in rebasis", with a pre-filled
            # issue link and the wrong exit code — including answering "no" to
            # the `migrate` confirmation.
            raise
        except Exception as exc:  # noqa: BLE001 - the top-level boundary
            _render_unexpected(exc, func.__name__)
            raise typer.Exit(code=EXIT_UNEXPECTED) from None

    return wrapper


#: Where an error code is explained. Rendered as the panel's subtitle, so it
#: has to be somewhere a user of the installed package can actually go.
ERROR_DOCS = "https://batuhanzorbeyzengin.github.io/rebasis/reference/errors/"


def error_docs_url(code: str) -> str:
    """The published explanation of one error code, anchor included.

    A function rather than an f-string at the call site because the fragment is
    half of a contract: `report.catalog` has to generate a matching anchor into
    `docs/reference/errors.md`, and for a while it did not — the page carried
    only its ten family headings, so `#rb-e3004` scrolled nowhere and the
    subtitle was a link that looked precise and was not.
    `tests/unit/test_errors.py` holds the two ends together.
    """
    return f"{ERROR_DOCS}#{code.lower()}"


def _render(exc: RebasisError) -> None:
    """A known error: what happened, the detail, the next step.

    Everything taken from the exception is escaped before it is put inside
    markup. Square brackets are how rich writes a style tag, so an unescaped
    hint loses whatever they contain -- `pip install "rebasis[chroma]"`, the
    single most useful sentence rebasis prints, reached the user as
    `pip install "rebasis"`. The same applies to a message or a context value
    carrying a filename, an id or a URI that happens to contain a bracket.
    """
    body = [f"[bold]{escape(exc.message)}[/bold]"]
    if exc.context:
        body.append("")
        body.extend(
            f"  [dim]{escape(str(k))}[/dim] {escape(str(v))}"
            for k, v in sorted(exc.context.items())
        )
    if exc.hint:
        body.extend(["", f"[cyan]{escape(exc.hint)}[/cyan]"])
    err_console.print(
        Panel(
            "\n".join(body),
            title=f"[red]{exc.code}[/red]",
            # A URL, not a repository path: `pip install rebasis` ships no
            # `docs/` directory, so the old subtitle named a file the reader
            # could not open and a fragment the docs site does not define.
            subtitle=f"[dim]{error_docs_url(exc.code)}[/dim]",
            border_style="red",
        )
    )


def _render_unexpected(exc: Exception, command: str) -> None:
    """An unexpected error: say so plainly and make reporting it one click.

    The pre-filled link carries version, Python, OS and command name only. No
    paths, no arguments, no corpus content.
    """
    from rebasis.__about__ import __version__

    err_console.print(
        Panel(
            f"[bold]{escape(f'{type(exc).__name__}: {exc}')}[/bold]\n\n"
            "[dim]This is a bug in rebasis, not a problem with your data.[/dim]",
            title="[red]Unexpected error[/red]",
            border_style="red",
        )
    )
    query = urllib.parse.urlencode(
        {
            "title": f"Unexpected {type(exc).__name__} in `rebasis {command}`",
            "body": (
                f"rebasis: {__version__}\n"
                f"python: {sys.version.split()[0]}\n"
                f"platform: {platform.platform()}\n"
                f"command: {command}\n\n"
                f"error: {type(exc).__name__}: {exc}\n\n"
                "<!-- Please add what you were doing. Do not paste corpus content. -->"
            ),
        }
    )
    err_console.print(f"[dim]Report it: {_ISSUE_URL}?{query}[/dim]")
