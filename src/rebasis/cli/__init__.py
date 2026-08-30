"""Command-line interface.

Logging is configured **here and nowhere else**: importing rebasis as a library
must never touch the host application's logging configuration.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer

from rebasis.cli._common import console, handle_errors, verbosity_to_level
from rebasis.cli.adapter import adapter_app
from rebasis.cli.audit import audit_app
from rebasis.cli.compare import compare_command
from rebasis.cli.doctor import doctor_command
from rebasis.cli.eval import eval_command
from rebasis.cli.expose import expose_command
from rebasis.cli.fit import fit_command
from rebasis.cli.migrate import (
    gc_command,
    migrate_command,
    pause_command,
    resume_command,
    rollback_command,
    status_command,
)
from rebasis.cli.probe import probe_command

__all__ = ["app"]

app = typer.Typer(
    name="rebasis",
    help=(
        "Measure whether an embedding upgrade is worth it, bridge it without "
        "reindexing when it is, and migrate safely when you are ready."
    ),
    no_args_is_help=True,
    # Completion is on. The arguments this CLI takes are the kind nobody types
    # correctly twice — store URIs like `chroma:///long/path#collection` and
    # model ids like `sentence-transformers/all-MiniLM-L6-v2` — and typer gives
    # `--install-completion` for them at no cost.
    add_completion=True,
)


def _version_callback(value: bool) -> None:
    """`--version` alongside the `version` subcommand.

    The subcommand came first, but `--version` is what a person types, and a CLI
    that answers `No such option: --version` looks broken before it has done
    anything. Eager, so it prints without the callback configuring logging.
    Read from `__about__` rather than the package root: importing `rebasis`
    pulls in `Bridge`, which is a lot of work to print a string.
    """
    if not value:
        return
    from rebasis.__about__ import __version__

    console.print(__version__)
    raise typer.Exit


@app.callback()
def main(  # noqa: PLR0913, PLR0917 - each option is a documented global flag
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Print the version and exit.",
        ),
    ] = False,
    verbose: Annotated[
        int, typer.Option("-v", "--verbose", count=True, help="-v for INFO, -vv for DEBUG")
    ] = 0,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="Errors only")] = False,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="debug|info|warning|error — overrides -v and -q"),
    ] = None,
    log_format: Annotated[
        str | None, typer.Option("--log-format", help="auto|console|json")
    ] = None,
    log_file: Annotated[str | None, typer.Option("--log-file", help="Write logs here")] = None,
    unsafe_log_content: Annotated[
        bool,
        typer.Option(
            "--unsafe-log-content",
            help="Disable redaction. Debugging only; recorded in the audit trail.",
        ),
    ] = False,
) -> None:
    """Configure logging and telemetry once, at the entry point."""
    del version  # handled eagerly by _version_callback, before this runs
    import atexit

    from rebasis.observability import configure_logging, setup_telemetry, shutdown_telemetry

    if unsafe_log_content:
        # Disabling redaction cannot be hidden. The warning is on every run;
        # the record is what survives the terminal being closed, and it was the
        # half that was missing.
        console.print(
            "[red bold]Redaction is disabled.[/red bold] Document text, queries "
            "and vectors may appear in the logs. This is recorded in the audit "
            "trail."
        )
        _record_unsafe_logging()

    settings = configure_logging(
        # The explicit flag outranks the counted ones: someone who writes
        # `--log-level debug` in a script means it, and should not have to know
        # how many `v`s that is.
        cli_level=_named_level(log_level) or verbosity_to_level(verbose, quiet),
        cli_format=log_format,
        cli_log_file=log_file,
        redact=not unsafe_log_content,
    )

    # Off unless REBASIS_OTEL_ENABLED says otherwise, and pointed at the user's
    # own collector — rebasis sends nothing anywhere. Registered at exit because
    # a short CLI run would otherwise end before the batch processor's first
    # export, and the user would conclude it does not work.
    if setup_telemetry():
        atexit.register(shutdown_telemetry)

    _announce_runtime(settings)


def _named_level(name: str | None) -> int | None:
    """Turn ``--log-level warning`` into a level, or refuse it clearly."""
    if name is None:
        return None
    import logging

    level = logging.getLevelNamesMapping().get(name.strip().upper())
    if level is None:
        from rebasis.errors import ConfigError

        raise ConfigError(
            f"{name!r} is not a log level.",
            hint="One of: debug, info, warning, error, critical.",
            context={"log_level": name},
        )
    return level


def _announce_runtime(settings: Any) -> None:
    """Say which environment was detected and what it resolved to.

    The first event of every run, so that "why is it logging at this level" is
    answerable from the log itself rather than by re-deriving the precedence
    order. At INFO, so a run at WARNING does not pay for it.
    """
    import logging

    from rebasis.compute import blas_info
    from rebasis.config import settings as config_settings
    from rebasis.observability import Events, get_logger

    get_logger(__name__).info(
        Events.RUNTIME_DETECTED,
        environment=str(settings.runtime),
        log_level=logging.getLevelName(settings.level),
        log_format=str(settings.fmt),
        blas_threads=blas_info()[1],
        device=config_settings().device,
    )


def _record_unsafe_logging() -> None:
    """Write the audit record for `--unsafe-log-content`.

    Best-effort by design: this runs before any command has said where its state
    lives, so it lands in the default state directory. A run that cannot write
    a record still gets the terminal warning — failing the command because the
    *warning* could not be filed would be the wrong way round.
    """
    from contextlib import suppress

    with suppress(Exception):
        from rebasis.cli._pipeline import audit_writer_for
        from rebasis.observability import Events

        audit_writer_for(None).write(
            Events.UNSAFE_LOGGING_ENABLED,
            inputs={"redaction": "disabled"},
            outputs={"redaction": "disabled"},
            subject="config",
        )


@app.command("version")
@handle_errors
def version_command() -> None:
    """Print the version."""
    from rebasis import __version__

    console.print(__version__)


app.command("probe")(probe_command)
app.command("compare")(compare_command)
app.command("expose")(expose_command)
app.command("fit")(fit_command)
app.command("eval")(eval_command)
app.command("migrate")(migrate_command)
app.command("pause")(pause_command)
app.command("resume")(resume_command)
app.command("status")(status_command)
app.command("rollback")(rollback_command)
app.command("gc")(gc_command)
app.command("doctor")(doctor_command)
app.add_typer(audit_app, name="audit")
app.add_typer(adapter_app, name="adapter")
