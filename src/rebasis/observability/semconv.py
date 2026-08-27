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

**Two names that are not here, and why.**

``gen_ai.provider.name`` is a current core attribute alongside the operation and
the model, and it is absent because its value is not reachable. The honest value
is the embedding backend — ``sentence-transformers``, ``ollama``, an
OpenAI-compatible endpoint — and the ``Embedder`` protocol does not carry the
entry-point name it was opened under. Adding one would change the protocol and
every backend implementing it, which is a real change to make for a real reason
rather than for conformance to a namespace that is still development-status. It
is listed as absent rather than emitted as a guess.

``OTEL_SEMCONV_STABILITY_OPT_IN`` was read here and nothing branched on it. The
variable exists so a library can emit an old attribute name and a new one during
a migration period; this project has never emitted an old one — ``db.system.name``
is the stable spelling and is the only spelling it has ever used — so there is
nothing for an opt-in to select between. A reader is entitled to assume a
function that parses a standard environment variable does something with it, and
this one could not.
"""

from __future__ import annotations

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
]

# Resource attributes
SERVICE_NAME: Final = "service.name"
SERVICE_VERSION: Final = "service.version"

# GenAI semantic conventions — EXPERIMENTAL, may change between spec versions
GEN_AI_OPERATION_NAME: Final = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL: Final = "gen_ai.request.model"
GEN_AI_EMBEDDINGS_DIMENSION_COUNT: Final = "gen_ai.embeddings.dimension.count"
OPERATION_EMBEDDINGS: Final = "embeddings"

# Database semantic conventions. STABLE, unlike the block above — a collector
# already knows what to do with this one. There is no vector-database convention
# to conform to (the upstream issue asking for one is open and unassigned), so
# the backend's declared name goes in the standard field and rebasis invents no
# `db.vector.*` namespace of its own.
DB_SYSTEM_NAME: Final = "db.system.name"

# rebasis-specific attributes
REBASIS_ADAPTER_TYPE: Final = "rebasis.adapter.type"
REBASIS_ADAPTER_DIRECTION: Final = "rebasis.adapter.direction"
REBASIS_PROBE_ARR_R10: Final = "rebasis.probe.arr_r10"
REBASIS_PROBE_DECISION: Final = "rebasis.probe.decision"
REBASIS_MIGRATE_STATE: Final = "rebasis.migrate.state"
