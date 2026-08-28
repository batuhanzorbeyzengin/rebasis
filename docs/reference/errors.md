<!-- GENERATED FILE — edit the source module, not this page. -->

# Error codes

Every rebasis error carries a code, a cause and a next step.

**Codes are stable.** A code is never removed or repurposed — only marked
deprecated — because user scripts and dashboards depend on them.

`Transient` errors are retried with exponential backoff, at most three
times. Permanent errors are never retried: a silently retried permanent
failure is the most common cause of undiagnosable slowness.

Exit codes: `0` success · `1` unexpected · `2` usage or configuration ·
`3` domain error · `130` interrupted.

## RB-E0xxx

| Code | Class | Transient | Exit | What it means |
|---|---|---|---|---|
| `RB-E0000`{ #rb-e0000 } | `RebasisError` | no | 3 | Base class for every error rebasis raises. |

## RB-E1xxx

| Code | Class | Transient | Exit | What it means |
|---|---|---|---|---|
| `RB-E1000`{ #rb-e1000 } | `ConfigError` | no | 2 | Invalid configuration or invocation. |
| `RB-E1001`{ #rb-e1001 } | `InvalidStoreURI` | no | 2 | The store URI could not be parsed or names an unknown backend. |
| `RB-E1002`{ #rb-e1002 } | `MissingDependency` | no | 2 | An optional dependency is required for the requested operation. |
| `RB-E1003`{ #rb-e1003 } | `BadThreshold` | no | 2 | A decision threshold is outside its valid range. |
| `RB-E1004`{ #rb-e1004 } | `MalformedQueryLog` | no | 2 | A query log file could not be read as JSONL. |

## RB-E2xxx

| Code | Class | Transient | Exit | What it means |
|---|---|---|---|---|
| `RB-E2000`{ #rb-e2000 } | `EmbedError` | no | 3 | Embedding backend failure. |
| `RB-E2001`{ #rb-e2001 } | `BackendUnavailable` | **yes** | 3 | The embedding backend is temporarily unreachable. |
| `RB-E2002`{ #rb-e2002 } | `ProfileMismatch` | no | 3 | The encoding profile does not match the one the index was built with. |
| `RB-E2003`{ #rb-e2003 } | `UnknownModelProfile` | no | 3 | No encoding profile is known for this model and none was supplied. |
| `RB-E2004`{ #rb-e2004 } | `EmbeddingDimensionMismatch` | no | 3 | Vector dimensionality does not match what the collection expects. |

## RB-E3xxx

| Code | Class | Transient | Exit | What it means |
|---|---|---|---|---|
| `RB-E3000`{ #rb-e3000 } | `StoreError` | no | 3 | Vector store failure. |
| `RB-E3001`{ #rb-e3001 } | `StoreUnsupported` | no | 3 | No backend is registered for this store type. |
| `RB-E3002`{ #rb-e3002 } | `CapabilityMissing` | no | 3 | The store does not support an operation this command requires. |
| `RB-E3003`{ #rb-e3003 } | `CollectionNotFound` | no | 3 | The named collection does not exist in the store. |
| `RB-E3004`{ #rb-e3004 } | `StoreWriteFailed` | **yes** | 3 | A write to the store failed and may succeed on retry. |
| `RB-E3005`{ #rb-e3005 } | `StoreDimensionMismatch` | no | 3 | The vectors to write do not match the collection's dimension. |

## RB-E4xxx

| Code | Class | Transient | Exit | What it means |
|---|---|---|---|---|
| `RB-E4000`{ #rb-e4000 } | `AdapterError` | no | 3 | Adapter fitting, loading or serialisation failure. |
| `RB-E4001`{ #rb-e4001 } | `FitFailed` | no | 3 | The adapter could not be fitted (singular system, degenerate input). |
| `RB-E4002`{ #rb-e4002 } | `IncompatibleAdapter` | no | 3 | Adapter fingerprint does not match this index or these models. |
| `RB-E4003`{ #rb-e4003 } | `AdapterSchemaVersion` | no | 3 | The ``.rbs`` schema version cannot be read by this release. |
| `RB-E4004`{ #rb-e4004 } | `SerializationError` | no | 3 | The adapter file is corrupt — integrity hash mismatch. |

## RB-E5xxx

| Code | Class | Transient | Exit | What it means |
|---|---|---|---|---|
| `RB-E5000`{ #rb-e5000 } | `ProbeError` | no | 3 | Diagnosis failure. |
| `RB-E5001`{ #rb-e5001 } | `InsufficientSamples` | no | 3 | Too few samples to produce a trustworthy measurement. |
| `RB-E5002`{ #rb-e5002 } | `GroundTruthUnavailable` | no | 3 | Ground truth could not be established from held-out chunks, a query log or synthesis. |
| `RB-E5003`{ #rb-e5003 } | `LeakageDetected` | no | 3 | The fit set and the query set overlap. |

## RB-E6xxx

| Code | Class | Transient | Exit | What it means |
|---|---|---|---|---|
| `RB-E6000`{ #rb-e6000 } | `MigrateError` | no | 3 | Migration engine failure. |
| `RB-E6001`{ #rb-e6001 } | `MigrationInterrupted` | no | 3 | The migration stopped before completion; resume from the checkpoint. |
| `RB-E6002`{ #rb-e6002 } | `ItemFailed` | no | 3 | A single record failed to migrate. |
| `RB-E6003`{ #rb-e6003 } | `ManifestLocked` | no | 3 | Another rebasis process holds the manifest lock. |
| `RB-E6004`{ #rb-e6004 } | `InsufficientDiskSpace` | no | 3 | Not enough free space to complete the migration safely. |
| `RB-E6005`{ #rb-e6005 } | `WritesDidNotSurvive` | no | 3 | The store accepted every write and a fresh connection cannot see them. |

## RB-E7xxx

| Code | Class | Transient | Exit | What it means |
|---|---|---|---|---|
| `RB-E7000`{ #rb-e7000 } | `StorageError` | no | 3 | Durability or filesystem failure. |
| `RB-E7001`{ #rb-e7001 } | `AtomicWriteFailed` | no | 3 | An atomic write could not be completed; the target is untouched. |
| `RB-E7002`{ #rb-e7002 } | `IntegrityCheckFailed` | no | 3 | A stored artefact failed its integrity hash check. |
| `RB-E7003`{ #rb-e7003 } | `ShadowMissing` | no | 3 | The shadow copy needed to roll back this job is absent. |
| `RB-E7004`{ #rb-e7004 } | `LockHeld` | no | 3 | A file lock is held by another process. |

## RB-E8xxx

| Code | Class | Transient | Exit | What it means |
|---|---|---|---|---|
| `RB-E8000`{ #rb-e8000 } | `ComputeError` | no | 3 | Compute backend or device failure. |
| `RB-E8001`{ #rb-e8001 } | `DeviceUnavailable` | no | 3 | The requested device is not available on this machine. |
| `RB-E8002`{ #rb-e8002 } | `DeviceOutOfMemory` | **yes** | 3 | The device ran out of memory; a smaller batch may succeed. |
| `RB-E8003`{ #rb-e8003 } | `UnsupportedDtype` | no | 3 | The device does not support the requested dtype. |
| `RB-E8004`{ #rb-e8004 } | `BackendMismatch` | no | 3 | Two compute backends disagree beyond the tolerance the device-parity tests allow. |

## RB-E9xxx

| Code | Class | Transient | Exit | What it means |
|---|---|---|---|---|
| `RB-E9000`{ #rb-e9000 } | `UserAbort` | no | 130 | The user interrupted the operation (Ctrl-C). |
