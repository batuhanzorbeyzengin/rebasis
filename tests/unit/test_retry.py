"""Retrying the errors that are worth retrying.

The rule the tests exist to hold: **only errors that declare themselves
transient**. Retrying a wrong answer turns a clear failure into a slow one, and
the decision about which is which belongs where the error is defined, not at the
call site.
"""

from __future__ import annotations

import pytest

from rebasis.errors import BackendUnavailable, EmbeddingDimensionMismatch, StoreWriteFailed
from rebasis.observability.retry import MAX_ATTEMPTS, retry_transient

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backoff is real seconds; the behaviour under test is not the sleeping."""
    monkeypatch.setattr("time.sleep", lambda _seconds: None)


class TestWhatIsRetried:
    def test_a_transient_error_is_retried_until_it_succeeds(self) -> None:
        attempts = {"n": 0}

        @retry_transient
        def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] < MAX_ATTEMPTS:
                raise BackendUnavailable("the service is waking up")
            return "ok"

        assert flaky() == "ok"
        assert attempts["n"] == MAX_ATTEMPTS

    def test_it_gives_up_after_the_documented_number_of_attempts(self) -> None:
        attempts = {"n": 0}

        @retry_transient
        def always_down() -> None:
            attempts["n"] += 1
            raise StoreWriteFailed("the store is not answering")

        with pytest.raises(StoreWriteFailed):
            always_down()

        assert attempts["n"] == MAX_ATTEMPTS

    def test_the_original_error_is_what_reaches_the_caller(self) -> None:
        """`reraise=True`: a user should see the store's problem, not tenacity's
        wrapper around it, and the error code has to survive to the panel."""

        @retry_transient
        def always_down() -> None:
            raise StoreWriteFailed("the store is not answering", hint="check the disk")

        with pytest.raises(StoreWriteFailed) as caught:
            always_down()

        assert caught.value.hint == "check the disk"


class TestWhatIsNot:
    def test_an_error_that_will_fail_the_same_way_is_not_retried(self) -> None:
        """A dimension mismatch is a wrong answer, not a hiccup. Retrying it
        three times produces the same error three times more slowly."""
        attempts = {"n": 0}

        @retry_transient
        def wrong_shape() -> None:
            attempts["n"] += 1
            raise EmbeddingDimensionMismatch("384 against 768")

        with pytest.raises(EmbeddingDimensionMismatch):
            wrong_shape()

        assert attempts["n"] == 1

    def test_a_plain_exception_is_not_retried(self) -> None:
        """Only `RebasisError` carries the flag. Anything else is a bug, and a
        bug retried is a bug hidden."""
        attempts = {"n": 0}

        @retry_transient
        def broken() -> None:
            attempts["n"] += 1
            raise ValueError("this is a bug")

        with pytest.raises(ValueError, match="this is a bug"):
            broken()

        assert attempts["n"] == 1

    def test_success_costs_nothing(self) -> None:
        @retry_transient
        def fine() -> int:
            return 7

        assert fine() == 7


def test_every_retry_is_logged() -> None:
    """A run that took four seconds because the service was slow should say so;
    the alternative is a pause with nothing in the record.

    Captured through structlog rather than `caplog`: the project logs through
    structlog, and the stdlib capture sees a rendered line rather than the
    event name and its fields.
    """
    from structlog.testing import capture_logs

    attempts = {"n": 0}

    @retry_transient
    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise BackendUnavailable("waking up")
        return "ok"

    with capture_logs() as entries:
        flaky()

    retries = [e for e in entries if e.get("event") == "retry.attempted"]
    assert len(retries) == 1
    assert retries[0]["attempt"] == 1
    assert retries[0]["error_code"] == "RB-E2001"
