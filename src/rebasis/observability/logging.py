"""The single logging configuration point.

All log setup lives in :func:`configure_logging`. Modules never call
``logging.basicConfig``, never add handlers and never change levels. They only
ask for a logger::

    from rebasis.observability import get_logger

    log = get_logger(__name__)

This is enforced by a lint rule: ``logging.basicConfig``, ``logging.getLogger``
and bare ``print(`` are rejected under ``src/`` outside ``observability/`` and
``cli/``. Centralised logging has exactly one failure mode — somebody attaching
their own handler off to the side — and that is not left to human discipline.

**Library / application split.** rebasis is both a CLI and a library. Embedded as
a library it **never touches the root logger**: :func:`configure_logging` is
triggered only from the CLI entry point or an explicit
``Bridge.configure_logging()`` call. Otherwise rebasis would clobber the logging
configuration of the application importing it.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from rebasis.observability.env import LogSettings, Runtime, resolve_log_settings
from rebasis.observability.events import is_known_event
from rebasis.observability.otel import span_context_fields
from rebasis.observability.redaction import make_redactor

if TYPE_CHECKING:
    from collections.abc import MutableMapping

__all__ = [
    "InvalidEventName",
    "configure_logging",
    "current_settings",
    "get_logger",
    "is_configured",
    "reset_logging",
]

_settings: LogSettings | None = None
_LOGGER_NAMESPACE = "rebasis"


class InvalidEventName(Exception):
    """A log call used free text instead of a registered event.

    Raised only under the test runtime. In production a stray event name is
    tagged and passed through: refusing to log is worse than logging something
    unregistered. Tests are where this must be caught.
    """


def is_configured() -> bool:
    """Whether :func:`configure_logging` has run in this process."""
    return _settings is not None


def current_settings() -> LogSettings | None:
    """The resolved settings, or ``None`` in library mode."""
    return _settings


def get_logger(name: str = _LOGGER_NAMESPACE) -> Any:
    """Return a bound structlog logger.

    Safe to call before :func:`configure_logging`: until then the ``rebasis``
    namespace carries a ``NullHandler`` and emits nothing.
    """
    return structlog.get_logger(name)


def add_event_namespace(
    _logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Validate that ``event`` is a registered event name.

    Message text may change between releases; event names may not. Users grep
    for them and dashboards group by them.
    """
    event = event_dict.get("event")
    if isinstance(event, str) and not is_known_event(event):
        if _settings is not None and _settings.runtime is Runtime.TEST:
            msg = (
                f"{event!r} is not a registered event. Add it to "
                f"rebasis.observability.events.Events and EVENT_CATALOG, or use "
                f"an existing name."
            )
            raise InvalidEventName(msg)
        event_dict["unregistered_event"] = True
    return event_dict


def add_otel_context(
    _logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Attach ``trace_id`` / ``span_id`` when a span is active."""
    event_dict.update(span_context_fields())
    return event_dict


def _drop_none(
    _logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Remove keys whose value is ``None``.

    ``trace_id``/``span_id`` are ``None`` whenever telemetry is off, which is the
    default. Emitting them as nulls on every line would double the noise for no
    information.
    """
    for key in [k for k, v in event_dict.items() if v is None]:
        del event_dict[key]
    return event_dict


def _build_processors(settings: LogSettings) -> list[Any]:
    """Assemble the processor chain.

    Order matters and each stage has exactly one job. Two placements are
    deliberate:

    * **Redaction sits immediately before the renderer**, not at the end of the
      chain, so no later processor can add a field after redaction has run.
    * **Caller info is only added at DEBUG**, because stack inspection is
      expensive and produces measurable slowdown in hot loops such as batch
      embedding.
    """
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        add_event_namespace,
        structlog.processors.TimeStamper(
            fmt="iso" if settings.utc_timestamps else "%H:%M:%S",
            utc=settings.utc_timestamps,
        ),
        add_otel_context,
    ]

    if settings.caller_info:
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameter.MODULE,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                }
            )
        )

    processors.extend(
        [
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _drop_none,
            make_redactor(enabled=settings.redact),
            _renderer(settings),
        ]
    )
    return processors


def _renderer(settings: LogSettings) -> Any:
    """Pick the renderer for this environment."""
    if settings.fmt == "json":
        return structlog.processors.JSONRenderer(sort_keys=True)
    return structlog.dev.ConsoleRenderer(colors=settings.colour)


def configure_logging(
    *,
    settings: LogSettings | None = None,
    force: bool = False,
    **resolve_kwargs: Any,
) -> LogSettings:
    """Configure logging for the ``rebasis`` namespace.

    Called from the CLI entry point or explicitly by an embedding application.
    Importing rebasis as a library never triggers it.

    Args:
        settings: Pre-resolved settings. When omitted they are resolved from
            flags, environment and the level matrix.
        force: Reconfigure even if already configured. Without this a second
            call is a no-op, so that a library consumer calling it twice does
            not tear down its own handlers.
        **resolve_kwargs: Forwarded to
            :func:`~rebasis.observability.env.resolve_log_settings`.

    Returns:
        The settings actually applied.
    """
    global _settings  # noqa: PLW0603 - the single configuration point owns this

    if _settings is not None and not force:
        return _settings

    resolved = settings or resolve_log_settings(**resolve_kwargs)

    # Only the rebasis namespace is touched. The root logger belongs to the
    # host application.
    namespace = logging.getLogger(_LOGGER_NAMESPACE)
    for handler in list(namespace.handlers):
        namespace.removeHandler(handler)
        # Reconfiguring must not leak the previous run's log file descriptor.
        prev = getattr(handler, "stream", None)
        if prev is not None and prev not in (sys.stderr, sys.stdout):
            handler.close()

    stream = sys.stderr
    if resolved.log_file:
        # storage.atomic is not used here: a log file is append-only and
        # explicitly allowed to be lossy. Durability guarantees belong to the
        # audit trail, not to logs.
        stream = Path(resolved.log_file).open("a", encoding="utf-8")  # noqa: SIM115

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    namespace.addHandler(handler)
    namespace.setLevel(resolved.level)
    # Do not propagate to root: that is how a library ends up printing twice in
    # an application that configured its own handlers.
    namespace.propagate = False

    for module, level in resolved.module_levels.items():
        logging.getLogger(module).setLevel(level)

    structlog.configure(
        processors=_build_processors(resolved),
        wrapper_class=structlog.make_filtering_bound_logger(resolved.level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _settings = resolved
    return resolved


def reset_logging() -> None:
    """Return to library mode: no handlers, no configuration.

    Used by test fixtures. Without it, the first test that configures logging
    would leak its settings into every subsequent test — and with
    ``pytest-randomly`` shuffling order, that failure would be irreproducible.
    """
    global _settings  # noqa: PLW0603 - see configure_logging

    namespace = logging.getLogger(_LOGGER_NAMESPACE)
    for handler in list(namespace.handlers):
        namespace.removeHandler(handler)
        # Close only streams we opened ourselves. NullHandler has no `stream` at
        # all, and closing sys.stderr would break the interpreter for everyone
        # else — a reset must be safe to call twice, and safe on a namespace
        # that was never configured.
        stream = getattr(handler, "stream", None)
        if stream is not None and stream not in (sys.stderr, sys.stdout):
            handler.close()
    namespace.addHandler(logging.NullHandler())
    namespace.setLevel(logging.NOTSET)
    structlog.reset_defaults()
    _settings = None


# Library default: attach a NullHandler so that importing rebasis emits nothing
# and never warns about missing handlers.
logging.getLogger(_LOGGER_NAMESPACE).addHandler(logging.NullHandler())
