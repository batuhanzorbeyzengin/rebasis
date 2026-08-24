"""Retrying the errors that are worth retrying.

One decorator, and one rule about what it may be applied to: **only errors that
declare themselves transient**. `RebasisError.transient` is a class attribute
precisely so this decision is made once, where the error is defined, rather than
guessed at each call site.

Three errors carry it today, and each is a real network or filesystem hiccup
rather than a wrong answer: `BackendUnavailable` (the embedding service is not
answering), `ProfileMismatch`'s sibling `StoreWriteFailed` (the store refused a
write that may take next time) and `TelemetryExportFailed`.

**Nothing that changes the answer is retried.** A dimension mismatch, a
capability the store does not have, an adapter built for another model — those
fail the same way every time, and retrying them turns a clear error into a slow
one. That is why the predicate is the flag and not the exception type.

Every attempt after the first is logged. A run that took four seconds because
the embedding service was slow to wake should say so; the alternative is a
mystery pause with nothing in the record.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from rebasis.errors import RebasisError
from rebasis.observability.events import Events
from rebasis.observability.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["MAX_ATTEMPTS", "retry_transient"]

log = get_logger(__name__)

#: Total attempts, the first one included.
MAX_ATTEMPTS = 3

#: Seconds. Exponential from the first, with full jitter so that a fleet of
#: workers hitting one rate-limited endpoint does not retry in lockstep.
INITIAL_BACKOFF = 0.5
MAX_BACKOFF = 8.0


def _is_transient(exc: BaseException) -> bool:
    """Whether this error declared itself worth another attempt."""
    return isinstance(exc, RebasisError) and exc.transient


def retry_transient[T](func: Callable[..., T]) -> Callable[..., T]:
    """Retry a call that failed with a transient error.

    Wraps the function rather than the call site so that a backend author gets
    the behaviour by decorating one method, and so that the retry policy lives
    in one place instead of being re-tuned per backend.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        def _log(state: RetryCallState) -> None:
            outcome = state.outcome
            if outcome is None or not outcome.failed:
                return
            exc = outcome.exception()
            log.warning(
                Events.RETRY_ATTEMPTED,
                attempt=state.attempt_number,
                duration_ms=round((state.seconds_since_start or 0.0) * 1000, 1),
                error_code=getattr(exc, "code", type(exc).__name__),
            )

        for attempt in Retrying(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(MAX_ATTEMPTS),
            wait=wait_exponential_jitter(initial=INITIAL_BACKOFF, max=MAX_BACKOFF),
            before_sleep=_log,
            reraise=True,
        ):
            with attempt:
                return func(*args, **kwargs)
        raise AssertionError("unreachable: Retrying always returns or raises")

    return wrapper
