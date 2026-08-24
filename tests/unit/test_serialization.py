"""The ``.rbs`` format.

The property that matters most here is negative: an adapter built for a
different model **must not load**. That is structural — applying the wrong
adapter to the wrong index should be impossible, not merely discouraged.
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - pytest resolves fixture annotations at runtime

import numpy as np
import pytest

from rebasis.core import (
    CenteredProcrustesAdapter,
    IdentityAdapter,
    LinearAdapter,
    LowRankAffineAdapter,
    ProcrustesAdapter,
    ScaledAdapter,
    ScoreCalibrator,
    l2_normalize,
    load_adapter,
    save_adapter,
)
from rebasis.core.serialization import CURRENT_SCHEMA
from rebasis.errors import AdapterSchemaVersion, IncompatibleAdapter, SerializationError
from rebasis.types import EncodingProfile

OLD_PROFILE = EncodingProfile(model_id="sentence-transformers/all-MiniLM-L6-v2", dim=32)
NEW_PROFILE = EncodingProfile(
    model_id="BAAI/bge-small-en-v1.5",
    dim=32,
    query_prefix="Represent this sentence for searching relevant passages: ",
)


@pytest.fixture
def pairs(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A matched pair set large enough to fit any adapter."""
    src = l2_normalize(rng.standard_normal((400, 32)).astype(np.float32))
    dst = l2_normalize(rng.standard_normal((400, 32)).astype(np.float32))
    return src, dst


def _save(adapter: object, tmp_path: Path) -> Path:
    return save_adapter(
        adapter,  # type: ignore[arg-type]
        tmp_path / "adapter.rbs",
        direction="query_to_old",
        old_profile=OLD_PROFILE,
        new_profile=NEW_PROFILE,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "builder",
    [
        ProcrustesAdapter.fit,
        CenteredProcrustesAdapter.fit,
        LinearAdapter.fit,
        LowRankAffineAdapter.fit,
        IdentityAdapter.fit,
    ],
    ids=lambda f: f.__qualname__.split(".")[0],
)
def test_round_trip_is_bit_identical(
    builder: object, pairs: tuple[np.ndarray, np.ndarray], tmp_path: Path
) -> None:
    """Saving and loading must not change a single output bit.

    A serialisation that "works" while perturbing the weights degrades quality
    silently.
    """
    src, dst = pairs
    adapter = builder(src, dst)  # type: ignore[operator]
    path = _save(adapter, tmp_path)

    loaded, manifest, calibrator = load_adapter(path, verify=True)
    assert manifest.schema == CURRENT_SCHEMA
    assert manifest.adapter_type == adapter.type_name
    assert calibrator is None
    assert np.array_equal(adapter.apply(src), loaded.apply(src))


@pytest.mark.unit
def test_composite_adapter_round_trips(
    pairs: tuple[np.ndarray, np.ndarray], tmp_path: Path
) -> None:
    """A DSM-wrapped adapter must rebuild its base as well as its diagonal."""
    src, dst = pairs
    adapter = ScaledAdapter.fit(CenteredProcrustesAdapter.fit(src, dst), src, dst)
    loaded, manifest, _ = load_adapter(_save(adapter, tmp_path), verify=True)

    assert manifest.adapter_type == "procrustes_centered+dsm"
    assert np.array_equal(adapter.apply(src), loaded.apply(src))


@pytest.mark.unit
def test_wrong_old_profile_is_refused(pairs: tuple[np.ndarray, np.ndarray], tmp_path: Path) -> None:
    """A fingerprint mismatch blocks the load.

    This is the structural guarantee. Loading anyway would degrade retrieval
    with no error at all — the failure mode the format exists to prevent.
    """
    src, dst = pairs
    path = _save(CenteredProcrustesAdapter.fit(src, dst), tmp_path)

    other = EncodingProfile(model_id="some/other-model", dim=32)
    with pytest.raises(IncompatibleAdapter) as excinfo:
        load_adapter(path, expected_old=other)
    assert excinfo.value.code == "RB-E4002"


@pytest.mark.unit
def test_a_changed_prefix_alone_is_enough_to_refuse(
    pairs: tuple[np.ndarray, np.ndarray], tmp_path: Path
) -> None:
    """The prefix trap, caught by the fingerprint.

    Same model, different prefix, is a *different encoding*, and applying an
    adapter across that boundary is a silent train/serve mismatch. The
    fingerprint covers the prefix precisely so this cannot pass.
    """
    src, dst = pairs
    path = _save(CenteredProcrustesAdapter.fit(src, dst), tmp_path)

    same_model_no_prefix = EncodingProfile(model_id=NEW_PROFILE.model_id, dim=32)
    with pytest.raises(IncompatibleAdapter):
        load_adapter(path, expected_new=same_model_no_prefix)


@pytest.mark.unit
def test_matching_profiles_load_cleanly(
    pairs: tuple[np.ndarray, np.ndarray], tmp_path: Path
) -> None:
    """The positive case: correct profiles load without complaint."""
    src, dst = pairs
    path = _save(CenteredProcrustesAdapter.fit(src, dst), tmp_path)
    adapter, _, _ = load_adapter(path, expected_old=OLD_PROFILE, expected_new=NEW_PROFILE)
    assert adapter.apply(src).shape == (400, 32)


@pytest.mark.unit
def test_calibrator_survives_the_round_trip(
    pairs: tuple[np.ndarray, np.ndarray], tmp_path: Path, rng: np.random.Generator
) -> None:
    """The calibrator lives in the file so thresholds keep working months later."""
    src, dst = pairs
    calibrator = ScoreCalibrator.fit(
        rng.random(300).astype(np.float32), rng.random(300).astype(np.float32)
    )
    path = save_adapter(
        CenteredProcrustesAdapter.fit(src, dst),
        tmp_path / "adapter.rbs",
        direction="query_to_old",
        old_profile=OLD_PROFILE,
        new_profile=NEW_PROFILE,
        calibrator=calibrator,
        evaluation={"arr_r10": 0.96},
    )

    _, _, loaded = load_adapter(path)
    assert loaded is not None
    probe = np.linspace(0.0, 1.0, 50, dtype=np.float32)
    assert np.allclose(calibrator.transform(probe), loaded.transform(probe), atol=1e-6)


@pytest.mark.unit
def test_corrupt_weights_are_detected(pairs: tuple[np.ndarray, np.ndarray], tmp_path: Path) -> None:
    """A flipped byte must fail loudly under `--verify`.

    Silent acceptance would mean quality degrading with no signal at all.
    """
    src, dst = pairs
    path = _save(CenteredProcrustesAdapter.fit(src, dst), tmp_path)

    weights = path / "weights.safetensors"
    raw = bytearray(weights.read_bytes())
    raw[-8] ^= 0xFF
    weights.write_bytes(bytes(raw))

    with pytest.raises(SerializationError):
        load_adapter(path, verify=True)


@pytest.mark.unit
def test_missing_files_are_reported_clearly(tmp_path: Path) -> None:
    """An incomplete directory fails with a message that says what is missing."""
    empty = tmp_path / "empty.rbs"
    empty.mkdir()
    with pytest.raises(SerializationError, match=r"not a valid \.rbs"):
        load_adapter(empty)


@pytest.mark.unit
def test_future_schema_is_refused(pairs: tuple[np.ndarray, np.ndarray], tmp_path: Path) -> None:
    """A newer schema raises RB-E4003 rather than being read best-effort.

    "I will do what I can" is exactly what the versioning policy forbids,
    because its failure is silent.
    """
    src, dst = pairs
    path = _save(CenteredProcrustesAdapter.fit(src, dst), tmp_path)

    manifest_path = path / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["schema"] = CURRENT_SCHEMA + 5
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(AdapterSchemaVersion) as excinfo:
        load_adapter(path)
    assert excinfo.value.code == "RB-E4003"
    assert "upgrade" in (excinfo.value.hint or "").lower()


@pytest.mark.unit
def test_manifest_records_what_is_needed_to_reproduce(
    pairs: tuple[np.ndarray, np.ndarray], tmp_path: Path
) -> None:
    """The manifest must identify both models, the direction and the dimensions."""
    src, dst = pairs
    path = _save(CenteredProcrustesAdapter.fit(src, dst), tmp_path)
    payload = json.loads((path / "manifest.json").read_text())

    assert payload["direction"] == "query_to_old"
    assert payload["old_model_id"] == OLD_PROFILE.model_id
    assert payload["new_model_id"] == NEW_PROFILE.model_id
    assert payload["input_dim"] == 32
    assert payload["output_dim"] == 32
    assert payload["symmetric"] is False, "one side is prefixed, so this is asymmetric"
    assert payload["tensor_hashes"], "integrity hashes are required"
