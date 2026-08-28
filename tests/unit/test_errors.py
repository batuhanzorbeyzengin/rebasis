"""Error contract.

Error codes are a **public interface**: users google them, scripts branch on
exit codes and dashboards group by `error_code`. That makes uniqueness and
stability testable properties, not conventions.
"""

from __future__ import annotations

import inspect
import re

import pytest

from rebasis import errors as errors_module
from rebasis.cli._common import error_docs_url
from rebasis.errors import (
    EXIT_ABORTED,
    EXIT_DOMAIN,
    EXIT_USAGE,
    ConfigError,
    DeviceOutOfMemory,
    EmbeddingDimensionMismatch,
    LeakageDetected,
    RebasisError,
    UserAbort,
)
from rebasis.report.catalog import render_errors_markdown

CODE_PATTERN = re.compile(r"^RB-E\d{4}$")


def _error_classes() -> list[type[RebasisError]]:
    out = []
    for name in errors_module.__all__:
        obj = getattr(errors_module, name)
        if inspect.isclass(obj) and issubclass(obj, RebasisError):
            out.append(obj)
    return out


def _defined_error_classes() -> list[type[RebasisError]]:
    """Every error defined in the module, whether it is exported or not."""
    return [
        obj
        for _, obj in inspect.getmembers(errors_module, inspect.isclass)
        if issubclass(obj, RebasisError) and obj.__module__ == errors_module.__name__
    ]


ERROR_CLASSES = _error_classes()


@pytest.mark.unit
def test_every_error_class_is_exported() -> None:
    """Direction two: module → `__all__`.

    `__all__` is the only list the generated catalogue and every parametrised
    test in this file read, so a class missing from it is invisible to all of
    them at once — not one check goes red. That is how `MalformedQueryLog`
    (RB-E1004) and `WritesDidNotSurvive` (RB-E6005) came to be raised on live
    paths, `cli/_pipeline.py` and `migrate/engine.py`, while the reference page
    the CLI links to listed neither. The check they needed is this one: the
    previous test asserted `len(ERROR_CLASSES) > 20`, which is the list counted
    against itself.
    """
    exported = {cls.__name__ for cls in ERROR_CLASSES}
    missing = sorted(
        cls.__name__ for cls in _defined_error_classes() if cls.__name__ not in exported
    )
    assert not missing, f"error classes absent from __all__: {missing}"


@pytest.mark.unit
@pytest.mark.parametrize("cls", ERROR_CLASSES, ids=lambda c: c.__name__)
def test_code_format(cls: type[RebasisError]) -> None:
    """Codes follow `RB-Ennnn`."""
    assert CODE_PATTERN.match(cls.code), f"{cls.__name__} has a malformed code: {cls.code}"


@pytest.mark.unit
def test_codes_are_unique() -> None:
    """Two errors sharing a code would make the documentation and metrics ambiguous."""
    seen: dict[str, str] = {}
    duplicates = []
    for cls in ERROR_CLASSES:
        if cls.code in seen:
            duplicates.append((cls.code, seen[cls.code], cls.__name__))
        seen[cls.code] = cls.__name__
    assert not duplicates, f"duplicate error codes: {duplicates}"


@pytest.mark.unit
@pytest.mark.parametrize("cls", ERROR_CLASSES, ids=lambda c: c.__name__)
def test_every_error_is_documented(cls: type[RebasisError]) -> None:
    """A docstring is what the generated reference page renders."""
    assert (cls.__doc__ or "").strip(), f"{cls.__name__} has no docstring"


@pytest.mark.unit
@pytest.mark.parametrize("cls", ERROR_CLASSES, ids=lambda c: c.__name__)
def test_exit_codes_are_from_the_documented_set(cls: type[RebasisError]) -> None:
    """Exit codes are a contract for script users."""
    assert cls.exit_code in {1, EXIT_USAGE, EXIT_DOMAIN, EXIT_ABORTED}


@pytest.mark.unit
def test_configuration_errors_exit_with_usage_code() -> None:
    """A configuration mistake means "you invoked it wrongly" (2), not "the domain failed" (3)."""
    assert ConfigError.exit_code == EXIT_USAGE


@pytest.mark.unit
def test_user_abort_uses_the_conventional_code() -> None:
    """Ctrl-C exits 130, as every shell expects."""
    assert UserAbort.exit_code == EXIT_ABORTED


@pytest.mark.unit
def test_only_recoverable_errors_are_transient() -> None:
    """Retry eligibility is deliberate, never accidental.

    A permanent error retried silently is the most common cause of
    undiagnosable slowness, so the transient set is asserted explicitly rather
    than merely counted.
    """
    transient = {cls.__name__ for cls in ERROR_CLASSES if cls.transient}
    assert transient == {
        "BackendUnavailable",
        "StoreWriteFailed",
        "DeviceOutOfMemory",
    }, f"the transient set changed: {sorted(transient)}"


@pytest.mark.unit
def test_oom_is_transient_so_a_job_survives_it() -> None:
    """OOM must never kill a job mid-migration."""
    assert DeviceOutOfMemory.transient


@pytest.mark.unit
def test_leakage_is_not_transient() -> None:
    """Leakage invalidates every ARR number; retrying it would be nonsense."""
    assert not LeakageDetected.transient


@pytest.mark.unit
def test_render_includes_code_context_and_hint() -> None:
    """Every error carries what happened, the detail, and the next step."""
    error = EmbeddingDimensionMismatch(
        "Collection has 384 dimensions, query has 1024.",
        hint="Run `rebasis doctor`.",
        context={"collection_dim": 384, "query_dim": 1024},
    )
    rendered = error.render()
    assert "RB-E2004" in rendered
    assert "384" in rendered
    assert "rebasis doctor" in rendered


@pytest.mark.unit
def test_render_omits_the_sections_it_has_nothing_for() -> None:
    """A bare error renders as one line, not as empty headings.

    `render` is what a user sees when a command fails, and the two optional
    sections are optional in practice: plenty of call sites raise with a message
    and nothing else. Printing "context:" with nothing after it, or a blank
    "hint:", reads as a bug in the tool at the exact moment the user is already
    dealing with one.
    """
    rendered = EmbeddingDimensionMismatch("Collection has 384 dimensions.").render()

    assert rendered == "[RB-E2004] Collection has 384 dimensions."
    assert "context:" not in rendered
    assert "hint:" not in rendered


@pytest.mark.unit
def test_cause_is_preserved() -> None:
    """Third-party exceptions are converted, not discarded.

    The original is kept as `__cause__` so the traceback still shows where it
    really came from.
    """
    original = ValueError("underlying library failure")
    error = EmbeddingDimensionMismatch("boundary translation", cause=original)
    assert error.__cause__ is original


@pytest.mark.unit
def test_generated_catalog_covers_every_code() -> None:
    """The generated reference page lists every error code."""
    rendered = render_errors_markdown()
    missing = [cls.code for cls in ERROR_CLASSES if f"`{cls.code}`" not in rendered]
    assert not missing, f"codes absent from the generated catalogue: {missing}"


@pytest.mark.unit
def test_every_code_the_cli_links_to_is_a_link_target() -> None:
    """The fragment the error panel prints has to exist on the page it names.

    The subtitle under every error is a URL ending in `#rb-exxxx`. The page
    carried anchors for its ten family headings and none for the rows, so all
    forty-three links scrolled to the same place: the top. Asserted against
    `error_docs_url` rather than a literal, so the two halves cannot drift.
    """
    rendered = render_errors_markdown()
    missing = [
        cls.code
        for cls in ERROR_CLASSES
        if f"{{ #{error_docs_url(cls.code).rsplit('#', 1)[1]} }}" not in rendered
    ]
    assert not missing, f"codes with no anchor on the reference page: {missing}"
