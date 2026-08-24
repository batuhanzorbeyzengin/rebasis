"""OpenTelemetry attribute names, isolated in one file.

Standard fields use OTel semantic conventions; we do not invent our own names.
rebasis-specific attributes live under the ``rebasis.*`` namespace.

**Why one file:** the ``gen_ai.*`` namespace is still development/experimental
and attribute names change between spec versions. Using it is the right call —
conforming to a standard beats inventing one — but the blast radius of a spec
change is contained here.

Content is never placed in a span attribute. The GenAI convention says the same:
prompts and content belong in span *events*, not attributes, because attributes
are always indexed and size-limited. rebasis records no content at all, so the
problem does not arise — but conforming keeps the right answer ready for the day
someone proposes "let's add the chunk text too".
"""

from __future__ import annotations

import os
from typing import Final

__all__ = [
    "DB_SYSTEM_NAME",
    "GEN_AI_EMBEDDINGS_DIMENSION_COUNT",
    "GEN_AI_OPERATION_NAME",
    "GEN_AI_REQUEST_MODEL",
    "OPERATION_EMBEDDINGS",
    "REBASIS_ADAPTER_DIRECTION",
    "REBASIS_ADAPTER_TYPE",
    "REBASIS_MIGRATE_STATE",
    "REBASIS_PROBE_ARR_R10",
    "REBASIS_PROBE_DECISION",
    "SERVICE_NAME",
    "SERVICE_VERSION",
    "semconv_opt_in",
]

# Resource attributes
SERVICE_NAME: Final = "service.name"
SERVICE_VERSION: Final = "service.version"

# GenAI semantic conventions — EXPERIMENTAL, may change between spec versions
GEN_AI_OPERATION_NAME: Final = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL: Final = "gen_ai.request.model"
GEN_AI_EMBEDDINGS_DIMENSION_COUNT: Final = "gen_ai.embeddings.dimension.count"
OPERATION_EMBEDDINGS: Final = "embeddings"

# Database semantic conventions
DB_SYSTEM_NAME: Final = "db.system.name"

# rebasis-specific attributes
REBASIS_ADAPTER_TYPE: Final = "rebasis.adapter.type"
REBASIS_ADAPTER_DIRECTION: Final = "rebasis.adapter.direction"
REBASIS_PROBE_ARR_R10: Final = "rebasis.probe.arr_r10"
REBASIS_PROBE_DECISION: Final = "rebasis.probe.decision"
REBASIS_MIGRATE_STATE: Final = "rebasis.migrate.state"


def semconv_opt_in() -> frozenset[str]:
    """Values of ``OTEL_SEMCONV_STABILITY_OPT_IN``.

    The standard mechanism for opting into newer convention versions while a
    namespace stabilises.
    """
    raw = os.environ.get("OTEL_SEMCONV_STABILITY_OPT_IN", "")
    return frozenset(part.strip() for part in raw.split(",") if part.strip())
