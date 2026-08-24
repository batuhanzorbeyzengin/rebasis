"""Space budgeting and garbage collection.

Two properties carry the weight here:

* **The pre-check must be roughly right.** The estimate has to land within the
  20% safety margin of what is actually used. A wildly wrong estimate is worse
  than none, because the user stops believing the next one.
* **``gc`` must never remove something irreplaceable.** Audit records, adapters
  and un-named shadow copies are off limits, and the property-based test throws
  arbitrary state directories at it to check that holds however the tree looks.
"""

from __future__ import annotations

import os
import time
from pathlib import Path  # noqa: TC003 - pytest resolves fixture annotations at runtime

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rebasis.core import l2_normalize
from rebasis.storage import apply_gc, plan_gc
from rebasis.storage.budget import SAFETY_MARGIN, estimate_budget
from rebasis.storage.shadow import ShadowStore

FAST = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@pytest.mark.unit
@pytest.mark.parametrize(("records", "dim"), [(1000, 64), (5000, 128), (20_000, 384)])
def test_shadow_estimate_matches_what_is_written(
    tmp_path: Path, rng: np.random.Generator, records: int, dim: int
) -> None:
    """The estimate must land inside the safety margin.

    Checked against a real write rather than against the same arithmetic, so this
    catches a formula that has drifted away from the format.
    """
    estimated = ShadowStore.estimate_bytes(records, dim)
    shadow = ShadowStore(tmp_path / "shadow", "job-estimate")
    shadow.append(
        [f"doc-{i}" for i in range(records)],
        l2_normalize(rng.standard_normal((records, dim)).astype(np.float32)),
    )

    actual = shadow.size_bytes()
    assert abs(actual - estimated) / estimated <= SAFETY_MARGIN


@pytest.mark.unit
def test_budget_reports_the_shadow_and_the_margin(tmp_path: Path) -> None:
    """The plan a user sees before a migration starts.

    Checked against a worked example: 487k records at 1024 dimensions in
    float32 is about 1.87 GB.
    """
    budget = estimate_budget(record_count=487_312, dim=1024, state_dir=tmp_path)

    assert 1.8 <= budget.shadow_bytes / 1024**3 <= 1.95
    assert budget.required_bytes > budget.shadow_bytes
    rendered = budget.render()
    assert "Shadow copy" in rendered
    assert "Free space needed" in rendered


@pytest.mark.unit
def test_disabling_the_shadow_removes_its_cost_and_warns(tmp_path: Path) -> None:
    """The trade is real, and it is stated rather than implied."""
    budget = estimate_budget(record_count=100_000, dim=768, state_dir=tmp_path, keep_original=False)
    assert budget.shadow_bytes == 0
    assert any("Rollback is disabled" in w for w in budget.warnings)


@pytest.mark.unit
def test_float16_shadow_halves_the_space_and_says_what_it_costs(tmp_path: Path) -> None:
    """Half the space, an approximate rollback rather than an exact one.

    A half-guarantee can be more dangerous than none, so the trade is spelled
    out where the user makes the choice.
    """
    full = estimate_budget(record_count=100_000, dim=768, state_dir=tmp_path)
    half = estimate_budget(
        record_count=100_000, dim=768, state_dir=tmp_path, shadow_precision="float16"
    )

    assert half.shadow_bytes == full.shadow_bytes // 2
    assert any("bit-identical" in w for w in half.warnings)


@pytest.mark.unit
def test_a_store_that_rebuilds_is_budgeted_for(tmp_path: Path) -> None:
    """A backend that cannot update in place needs both copies for a while.

    That is the store's cost, but preventing the surprise is ours.
    """
    budget = estimate_budget(
        record_count=10_000, dim=768, state_dir=tmp_path, store_growth_bytes=5 * 1024**3
    )
    assert budget.required_bytes > 5 * 1024**3
    assert any("cannot update in place" in w for w in budget.warnings)


@pytest.mark.unit
def test_gc_defaults_to_a_dry_run(tmp_path: Path) -> None:
    """Listing is not removing."""
    state = tmp_path / ".rebasis"
    (state / "reports").mkdir(parents=True)
    stale = state / "reports" / "old.md"
    stale.write_text("report")
    ancient = time.time() - 200 * 86_400
    os.utime(stale, (ancient, ancient))

    plan = plan_gc(state)
    assert plan.candidates
    assert stale.exists(), "planning must not remove anything"
    assert "dry run" in plan.render()


@pytest.mark.unit
def test_gc_never_lists_adapters_or_audit_records(tmp_path: Path) -> None:
    """Neither is ever removed automatically.

    An adapter is reproducible but expensive; an audit record is not reproducible
    at all.
    """
    state = tmp_path / ".rebasis"
    (state / "adapters").mkdir(parents=True)
    adapter = state / "adapters" / "bridge.rbs"
    adapter.write_text("weights")
    manifest = state / "manifest.db"
    manifest.write_text("db")
    ancient = time.time() - 3650 * 86_400
    for path in (adapter, manifest):
        os.utime(path, (ancient, ancient))

    listed = {c.path for c in plan_gc(state).candidates}
    assert adapter not in listed
    assert manifest not in listed


@pytest.mark.unit
def test_a_shadow_is_only_listed_when_named(tmp_path: Path) -> None:
    """Removing a shadow makes a migration permanently irreversible.

    It is never swept up incidentally, and even when named it needs confirmation.
    """
    state = tmp_path / ".rebasis"
    shadow = state / "shadow" / "job-1"
    shadow.mkdir(parents=True)
    (shadow / "vectors.bin").write_bytes(b"x" * 4096)

    assert not [c for c in plan_gc(state).candidates if c.category == "shadow"]

    named = [
        c for c in plan_gc(state, include_shadows=["job-1"]).candidates if c.category == "shadow"
    ]
    assert len(named) == 1
    assert named[0].requires_confirmation


@pytest.mark.unit
def test_apply_skips_unconfirmed_shadows(tmp_path: Path) -> None:
    """Confirmation is enforced at apply time, not only in the message."""
    state = tmp_path / ".rebasis"
    shadow = state / "shadow" / "job-1"
    shadow.mkdir(parents=True)
    vectors = shadow / "vectors.bin"
    vectors.write_bytes(b"x" * 4096)

    apply_gc(plan_gc(state, include_shadows=["job-1"]), confirmed=False)
    assert vectors.exists()

    apply_gc(plan_gc(state, include_shadows=["job-1"]), confirmed=True)
    assert not vectors.exists()


@pytest.mark.unit
def test_gc_keeps_the_newest_backups(tmp_path: Path) -> None:
    """Three generations kept, older ones collected."""
    state = tmp_path / ".rebasis"
    state.mkdir(parents=True)
    for index in range(1, 7):
        path = state / f"manifest.db.bak.{index}"
        path.write_bytes(b"b" * 128)
        stamp = time.time() - index * 3600
        os.utime(path, (stamp, stamp))

    collected = {c.path.name for c in plan_gc(state).candidates if c.category == "backups"}
    assert collected == {"manifest.db.bak.4", "manifest.db.bak.5", "manifest.db.bak.6"}


@pytest.mark.property
@FAST
@given(
    subdirs=st.lists(
        st.sampled_from(["reports", "cache", "adapters", "shadow/job-a", "shadow/job-b"]),
        min_size=1,
        max_size=5,
        unique=True,
    ),
    file_count=st.integers(1, 4),
)
def test_gc_never_touches_protected_paths(
    tmp_path_factory: pytest.TempPathFactory, subdirs: list[str], file_count: int
) -> None:
    """Whatever the state tree looks like, the protected things survive.

    Property-based because a fixed layout only proves the rule for that layout,
    and the guarantee has to hold for directories rebasis has never seen.
    """
    state = tmp_path_factory.mktemp("state") / ".rebasis"
    ancient = time.time() - 3650 * 86_400

    protected: list[Path] = []
    for subdir in subdirs:
        directory = state / subdir
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(file_count):
            path = directory / f"file-{index}.bin"
            path.write_bytes(b"x" * 256)
            os.utime(path, (ancient, ancient))
            if subdir.startswith(("adapters", "shadow")):
                protected.append(path)

    manifest = state / "manifest.db"
    manifest.write_bytes(b"db")
    os.utime(manifest, (ancient, ancient))
    protected.append(manifest)

    apply_gc(plan_gc(state), confirmed=False)

    for path in protected:
        assert path.exists(), f"gc removed a protected path: {path}"
