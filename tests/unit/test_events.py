"""Event catalogue contract.

The catalogue is checked **in both directions**: every event in the enum must
have an entry, and every entry must correspond to an event. A one-way check
lets the catalogue rot in whichever direction is not tested.
"""

from __future__ import annotations

import logging
import re

import pytest

from rebasis.observability.events import EVENT_CATALOG, Events, is_known_event
from rebasis.observability.redaction import ALLOWED_FIELDS
from rebasis.report.catalog import render_events_markdown

#: `<object>.<verb>` or `<module>.<object>.<verb>` — both spellings are in use.
EVENT_NAME_PATTERN = re.compile(r"^[a-z][a-z_]*(\.[a-z][a-z_]*){1,2}$")


@pytest.mark.unit
def test_every_event_has_a_catalog_entry() -> None:
    """Direction one: enum → catalogue."""
    missing = [str(e) for e in Events if e not in EVENT_CATALOG]
    assert not missing, f"events with no EventSpec: {missing}"


@pytest.mark.unit
def test_every_catalog_entry_has_an_event() -> None:
    """Direction two: catalogue → enum."""
    known = set(Events)
    orphans = [str(e) for e in EVENT_CATALOG if e not in known]
    assert not orphans, f"catalogue entries with no event: {orphans}"


@pytest.mark.unit
@pytest.mark.parametrize("event", list(Events), ids=str)
def test_event_name_follows_convention(event: Events) -> None:
    """Names are dot-separated lowercase, two or three segments."""
    assert EVENT_NAME_PATTERN.match(str(event)), f"{event} breaks the naming convention"


@pytest.mark.unit
@pytest.mark.parametrize("event", list(Events), ids=str)
def test_event_spec_is_well_formed(event: Events) -> None:
    """Every spec carries a real level, declared fields and a description."""
    spec = EVENT_CATALOG[event]
    assert spec.level in {
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    }
    assert isinstance(spec.fields, tuple)
    assert spec.description.strip(), f"{event} has no description"


@pytest.mark.unit
def test_declared_fields_are_all_loggable() -> None:
    """Every field an event declares must be in the redaction allowlist.

    Without this invariant an event could declare a field that redaction then
    strips at render time — the log would be silently useless, and the
    `observability.redaction.triggered` counter would fire on a legitimate code
    path, destroying its value as a bug signal.
    """
    declared = {field for spec in EVENT_CATALOG.values() for field in spec.fields}
    missing = sorted(declared - ALLOWED_FIELDS)
    assert not missing, f"fields declared by events but not in ALLOWED_FIELDS: {missing}"


@pytest.mark.unit
def test_audited_events_are_a_small_subset() -> None:
    """Audit records are the exception, not the rule.

    If most events were audited, the audit trail would become a second log —
    and the whole point of the distinction is that a log may be lost while an
    audit record may not.
    """
    audited = [e for e, spec in EVENT_CATALOG.items() if spec.audited]
    assert audited, "no audited events at all — the decision record would be empty"
    assert len(audited) < len(EVENT_CATALOG) / 2, (
        f"{len(audited)} of {len(EVENT_CATALOG)} events are audited; the audit "
        "trail is turning into a log"
    )


@pytest.mark.unit
def test_decision_event_is_audited() -> None:
    """`probe.decision.made` must be audited — it is the tool's actual output."""
    assert EVENT_CATALOG[Events.PROBE_DECISION_MADE].audited


@pytest.mark.unit
def test_store_write_is_audited() -> None:
    """Any change to user data must be reconstructable."""
    assert EVENT_CATALOG[Events.STORE_WRITE_PERFORMED].audited


@pytest.mark.unit
def test_is_known_event_rejects_free_text() -> None:
    """Free-text messages must not pass as event names."""
    assert is_known_event("probe.decision.made")
    assert not is_known_event("something happened")
    assert not is_known_event("")


@pytest.mark.unit
def test_generated_catalog_covers_every_event() -> None:
    """The generated reference page lists every event.

    The page is generated rather than written, so this asserts the generator is
    complete rather than that a human remembered to update a table.
    """
    rendered = render_events_markdown()
    missing = [str(e) for e in Events if f"`{e}`" not in rendered]
    assert not missing, f"events absent from the generated catalogue: {missing}"
