"""The ``.rbs`` adapter format.

Layout::

    adapter.rbs/
    ├── manifest.json         schema version, direction, type, dims, fingerprints
    ├── weights.safetensors   the tensors
    ├── calibration.json      isotonic calibrator, when present
    └── eval.json             the metrics the adapter was selected on

Three decisions worth stating:

**A directory, not an archive.** safetensors is memory-mappable, and wrapping it
in a zip would throw that away for no benefit.

**safetensors, not pickle.** It executes no code on load — which matters for a
tool whose files travel between machines — and the header-plus-contiguous-block
layout maps straight into memory.

**Fingerprints are load-blocking.** The encoding profiles of both models are
recorded. On mismatch the adapter refuses to load rather than degrading quietly,
which makes "applied the wrong adapter to the wrong index" a structural
impossibility rather than a documented hazard.

Schema versioning is independent of the package version: a `.rbs` file on disk
outlives the release that wrote it. A release producing schema *N* reads *N*
and *N-1*; anything older raises RB-E4003 and points at
``rebasis adapter upgrade``. Silent best-effort loading is forbidden.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rebasis.core.calibration import ScoreCalibrator
from rebasis.core.registry import build_adapter
from rebasis.errors import (
    AdapterSchemaVersion,
    IncompatibleAdapter,
    IntegrityCheckFailed,
    SerializationError,
)
from rebasis.observability import Events, get_logger
from rebasis.storage.atomic import atomic_write_bytes, atomic_write_json
from rebasis.storage.integrity import hash_state_dict, verify_state_dict
from rebasis.types import AdapterDirection, EncodingProfile, as_float32

if TYPE_CHECKING:
    from rebasis.core.base import BaseAdapter

log = get_logger(__name__)

__all__ = [
    "CURRENT_SCHEMA",
    "MIN_READABLE_SCHEMA",
    "AdapterManifest",
    "load_adapter",
    "save_adapter",
]

#: Current ``.rbs`` schema version.
CURRENT_SCHEMA = 1

#: Oldest schema this release can read. The promise is N and N-1.
MIN_READABLE_SCHEMA = 1

_MANIFEST = "manifest.json"
_WEIGHTS = "weights.safetensors"
_CALIBRATION = "calibration.json"
_EVAL = "eval.json"


@dataclass(slots=True)
class AdapterManifest:
    """Everything needed to validate and rebuild an adapter."""

    schema: int
    adapter_type: str
    direction: AdapterDirection
    input_dim: int
    output_dim: int
    old_profile_fingerprint: str
    new_profile_fingerprint: str
    old_model_id: str
    new_model_id: str
    rebasis_version: str
    config: dict[str, Any] = field(default_factory=dict)
    tensor_hashes: dict[str, str] = field(default_factory=dict)
    symmetric: bool = True
    created_utc: str = ""

    def to_json(self) -> dict[str, Any]:
        """Serialisable form."""
        return asdict(self)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> AdapterManifest:
        """Parse a manifest, rejecting unreadable schema versions.

        Raises:
            AdapterSchemaVersion: When the file is too old or too new.
        """
        schema = int(payload.get("schema", 0))
        if schema > CURRENT_SCHEMA:
            raise AdapterSchemaVersion(
                f"This adapter uses schema {schema}, but this release reads at most "
                f"{CURRENT_SCHEMA}.",
                hint="Upgrade rebasis: `pip install --upgrade rebasis`.",
                context={"count": schema},
            )
        if schema < MIN_READABLE_SCHEMA:
            raise AdapterSchemaVersion(
                f"This adapter uses schema {schema}, which this release no longer reads "
                f"(minimum {MIN_READABLE_SCHEMA}).",
                hint=(
                    "Convert it with `rebasis adapter upgrade <file>`; the "
                    "original file is never modified."
                ),
                context={"count": schema},
            )
        known = set(AdapterManifest.__slots__)
        return cls(**{k: v for k, v in payload.items() if k in known})


def save_adapter(  # noqa: PLR0913 - one argument per element of the .rbs format
    adapter: BaseAdapter,
    path: Path,
    *,
    direction: AdapterDirection,
    old_profile: EncodingProfile,
    new_profile: EncodingProfile,
    calibrator: ScoreCalibrator | None = None,
    evaluation: dict[str, Any] | None = None,
) -> Path:
    """Write an adapter to a ``.rbs`` directory.

    Every file goes through the atomic writer, so an interrupted save leaves the
    previous version intact rather than a half-written one.
    """
    from safetensors.numpy import save as safetensors_save

    from rebasis.__about__ import __version__

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    state = {k: np.ascontiguousarray(as_float32(v)) for k, v in adapter.state_dict().items()}

    manifest = AdapterManifest(
        schema=CURRENT_SCHEMA,
        adapter_type=adapter.type_name,
        direction=direction,
        input_dim=adapter.input_dim,
        output_dim=adapter.output_dim,
        old_profile_fingerprint=old_profile.fingerprint(),
        new_profile_fingerprint=new_profile.fingerprint(),
        old_model_id=old_profile.model_id,
        new_model_id=new_profile.model_id,
        rebasis_version=__version__,
        config={
            **adapter.config,
            "input_dim": adapter.input_dim,
            "output_dim": adapter.output_dim,
        },
        tensor_hashes=hash_state_dict(state),
        symmetric=old_profile.symmetric and new_profile.symmetric,
        created_utc=_utc_now(),
    )

    # safetensors rejects an empty tensor set, and the identity adapter has no
    # weights at all — a marker keeps the format uniform rather than making
    # every reader special-case it.
    payload = state or {"__empty__": np.zeros(1, dtype=np.float32)}
    atomic_write_bytes(path / _WEIGHTS, safetensors_save(payload))
    atomic_write_json(path / _MANIFEST, manifest.to_json())

    if calibrator is not None:
        atomic_write_json(
            path / _CALIBRATION,
            {
                "x": calibrator.x_thresholds.tolist(),
                "y": calibrator.y_thresholds.tolist(),
                "summary": calibrator.summary(),
            },
        )
    if evaluation is not None:
        atomic_write_json(path / _EVAL, evaluation)

    return path


def load_adapter(
    path: Path,
    *,
    expected_old: EncodingProfile | None = None,
    expected_new: EncodingProfile | None = None,
    verify: bool = False,
) -> tuple[BaseAdapter, AdapterManifest, ScoreCalibrator | None]:
    """Load an adapter, validating it before returning.

    Args:
        path: The ``.rbs`` directory.
        expected_old: Profile of the model the index was built with. When given,
            a fingerprint mismatch refuses the load.
        expected_new: Profile of the model being adopted.
        verify: Recompute every tensor hash. Off by default because the fast
            path costs microseconds and the full path milliseconds; ``adapter
            upgrade`` always turns it on.

    Raises:
        SerializationError: When the directory is malformed or a hash fails.
        IncompatibleAdapter: On a profile fingerprint mismatch.
        AdapterSchemaVersion: When the schema is unreadable.
    """
    from safetensors.numpy import load as safetensors_load

    path = Path(path)
    manifest_path = path / _MANIFEST
    weights_path = path / _WEIGHTS

    if not manifest_path.exists() or not weights_path.exists():
        raise SerializationError(
            f"{path.name} is not a valid .rbs directory.",
            hint=f"Expected {_MANIFEST} and {_WEIGHTS} inside it.",
        )

    try:
        manifest = AdapterManifest.from_json(json.loads(manifest_path.read_text()))
    except (ValueError, TypeError) as exc:
        raise SerializationError(
            f"The manifest of {path.name} could not be parsed.",
            hint="The file is corrupt. Restore from a backup, or refit the adapter.",
            cause=exc,
        ) from exc

    try:
        raw = safetensors_load(weights_path.read_bytes())
    except Exception as exc:
        raise SerializationError(
            f"The weights of {path.name} could not be read.",
            hint="The file is corrupt. Restore from a backup, or refit the adapter.",
            cause=exc,
        ) from exc

    state = {k: as_float32(v) for k, v in raw.items() if k != "__empty__"}

    _check_fingerprints(manifest, expected_old, expected_new)
    if verify and manifest.tensor_hashes:
        try:
            verify_state_dict(state, manifest.tensor_hashes)
        except IntegrityCheckFailed as exc:
            # Boundary translation: storage reports a generic integrity failure,
            # but what the user has in front of them is a corrupt adapter.
            # RB-E4004 carries an adapter-specific next step, and "refit" is
            # advice that only makes sense here.
            raise SerializationError(
                f"{path.name} failed its integrity check: the stored weights are corrupt.",
                hint=(
                    "Restore it from a backup, or refit with `rebasis fit` — "
                    "fitting is cheap relative to a reindex."
                ),
                context={"adapter_type": manifest.adapter_type},
                cause=exc,
            ) from exc

    adapter = build_adapter(manifest.adapter_type, state, manifest.config)

    calibrator = None
    calibration_path = path / _CALIBRATION
    if calibration_path.exists():
        payload = json.loads(calibration_path.read_text())
        calibrator = ScoreCalibrator(as_float32(payload["x"]), as_float32(payload["y"]))

    return adapter, manifest, calibrator


def _check_fingerprints(
    manifest: AdapterManifest,
    expected_old: EncodingProfile | None,
    expected_new: EncodingProfile | None,
) -> None:
    """Refuse to load an adapter built for different models."""
    for label, expected, recorded in (
        ("old", expected_old, manifest.old_profile_fingerprint),
        ("new", expected_new, manifest.new_profile_fingerprint),
    ):
        if expected is None:
            continue
        if expected.fingerprint() != recorded:
            # `adapter.rejected` is audited because it is security-relevant: it
            # means an adapter was pointed at an index it was not built for. It
            # was catalogued and never emitted, so the one event whose value is
            # that it appears in a record appeared nowhere.
            log.error(
                Events.ADAPTER_REJECTED,
                error_code=IncompatibleAdapter.code,
                adapter_type=manifest.adapter_type,
            )
            raise IncompatibleAdapter(
                f"This adapter was built for a different {label} model or encoding profile.",
                hint=(
                    "Applying it would silently degrade results, so the load is "
                    "refused. Refit with `rebasis fit`, or check that --old and "
                    "--new match the ones the adapter was created with."
                ),
                context={
                    "adapter_type": manifest.adapter_type,
                    "direction": manifest.direction,
                },
            )


def _utc_now() -> str:
    import datetime

    return datetime.datetime.now(tz=datetime.UTC).isoformat()
