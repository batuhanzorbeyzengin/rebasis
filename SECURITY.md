# Security policy

## Why this file exists

rebasis reads personal corpora — notes, code, archives, agent memory. That makes
two categories of bug security-relevant rather than merely annoying:

1. **Content leaking into logs, reports or telemetry.**
2. **Damage to a user's index**, which they may have accumulated over months.

## Reporting a vulnerability

Please report privately through
[GitHub Security Advisories](https://github.com/batuhanzorbeyzengin/rebasis/security/advisories/new),
not as a public issue.

Include what you were doing, what happened, and the version. **Do not include
corpus content or vectors** — a description is enough, and the same reasoning
that keeps them out of logs applies to reports.

## What rebasis guarantees

- **No outbound telemetry.** rebasis sends no data anywhere. There is no
  phone-home and no usage counter. The optional OpenTelemetry support sends data
  to *your own* Collector, only when you explicitly enable it, and is off by
  default.
- **Content never reaches a log.** Logging uses an allowlist, not a denylist: a
  field that is not explicitly permitted is redacted. Document text, vectors,
  query text, filesystem paths and credentials are excluded categorically.
- **Vectors are treated as sensitive.** Text can be reconstructed from
  embeddings, so a vector in a log file is as exposed as plaintext. This is a
  requirement, not a preference.
- **`probe` and `fit` never write to your index.** The only write path is
  `migrate`, it only upserts, and it never deletes.
- **`migrate` keeps a shadow copy by default**, so a migration can be rolled
  back. Disabling it prints a warning and is recorded in the audit trail.
- **Adapters refuse to load against the wrong index.** The encoding profiles of
  both models are fingerprinted into the `.rbs` file; on mismatch the load fails
  rather than silently degrading retrieval.
- **`--unsafe-log-content` cannot be used quietly.** It disables redaction, warns
  in red on every run, and writes `config.unsafe_logging_enabled` to the audit
  trail.

## What rebasis does not claim

The audit trail is **tamper-evident, not tamper-proof.** It is a local file with
a hash chain, and its owner can regenerate it. The chain exists to catch
accidental corruption and silent data loss, not to defend against a determined
local attacker. Claiming otherwise would be false.

## Supported versions

While the project is 0.x, only the latest release receives fixes.
