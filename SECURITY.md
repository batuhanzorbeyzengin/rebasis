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

### What to expect, and what is not promised

There is one maintainer. What follows is an intention, stated so that silence
can be read correctly; it is not a service-level agreement and nothing here is
contractual.

| | |
|---|---|
| Acknowledgement | Within **3 working days**. If you have not heard back in a week, the report did not arrive — open a public issue saying only that you are waiting on a security response, with no details. |
| Assessment | Within **10 working days**: whether it is a vulnerability, and how severe. |
| Fix | No date is promised. Severity and how much of a user's data is at risk decide the order. |
| Disclosure | Coordinated. An advisory is published when a fix is released, and it credits you unless you ask otherwise. |

GitHub is a CVE Numbering Authority, so an accepted report can be issued a
**CVE** through the same advisory, and GitHub publishes it to the **GitHub
Advisory Database** and from there to **OSV**. That is what makes it reachable by
`pip-audit`, `osv-scanner` and Dependabot, rather than only by someone reading
this repository.

### Verifying what you installed

Releases are published from GitHub Actions with PyPI Trusted Publishing, so no
long-lived API token exists that could be stolen and used to publish in this
project's name. Since `pypa/gh-action-pypi-publish` v1.11.0 each published file
also carries a **PEP 740 attestation** binding it to the workflow that built it;
PyPI shows it on the file listing.

Every release also attaches a **CycloneDX SBOM** covering the tree `uv.lock`
resolves with all extras installed. It is on the GitHub release, not on PyPI,
which accepts distributions only.

### What is scanned, and how often

| | |
|---|---|
| Every pull request | `gitleaks` for secrets, and `actions/dependency-review-action` for dependencies the change introduces — failing on a high-severity advisory or a copyleft licence |
| Weekly | `pip-audit` over the resolved tree with every extra, and an OpenSSF Scorecard run |
| Continuously | Dependabot, grouped by blast radius |

**One Scorecard check this project cannot pass, stated rather than worked
around.** *Code-Review* counts pull requests reviewed by somebody other than
their author; there is one maintainer, so it is low by construction. It does not
improve by configuring something — only by the project having more people.

*Branch Protection* used to be listed here beside it, and no longer belongs
there. It was not a check that could not pass; it was one blocked by a
mechanism. The release workflow pushes the changelog commit and its tag to
`main`, and it pushed them as `github-actions[bot]` — an identity **no ruleset
can exempt**, because bypass is granted to repository roles, deploy keys and
installed GitHub Apps, and a first-party integration is none of those. Any rule
strong enough to be worth having would have broken every release.

That push is now made by a GitHub App installed on this repository, scoped to
`contents: write` and minting a token that expires with the run, and an App is
something a ruleset's bypass list can name. Protecting `main` is therefore a
repository setting rather than a code change: a ruleset requiring a pull request
with the CI checks green, with the App bypassing that one push and nothing else.
[How the App is set up](https://batuhanzorbeyzengin.github.io/rebasis/development/release/#the-push-to-main-is-made-by-a-github-app).

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

## Your vectors are alignable, and `rebasis expose` says how much

The premise the rest of this file rests on — that a vector is as sensitive as
the text it came from — has a second consequence that is newer than the first.

[vec2vec](https://arxiv.org/abs/2505.12540) showed that embeddings can be
translated between models with no paired data at all, preserving their geometry
well enough to infer things about the underlying documents; its own framing is
that *vector databases reveal (almost) as much as their inputs*.
[mini-vec2vec](https://arxiv.org/abs/2510.02348) then reduced the cost of that
translation from a day of adversarial training to an orthogonal solve and an
assignment. So "somebody took the vectors but not the text" is a smaller
mitigation than it reads as.

`rebasis expose` measures how far that goes on **your** index, and returns a
single number:

```bash
rebasis expose --store <uri> --json
```

Five things about it, and each of them is a limit rather than a feature:

- **It returns no translation.** No aligned vectors, no reconstructed text, no
  inversion — asserted by a test rather than promised in prose. A number grants
  nobody a capability; mini-vec2vec is published and installable already.
- **It is an upper bound.** The measurement draws its reference half from your
  own corpus, which is a better position than any adversary has.
- **On some indexes it is not one number.** The method is stochastic, so the
  command runs three alignments. On one BEIR corpus those agreed to within
  0.054; on another they returned 0.087, 0.624 and 0.034 for the same index. The
  command says so when they disagree, and that sentence is the result on such an
  index — a single low figure from a single run is not evidence of safety.
- **It carries no band.** low/medium/high would be a policy threshold with no
  labelled harm to calibrate against, and this project does not ship those.
- **The reference model must be local.** A hosted endpoint is refused, because
  measuring exposure by creating some is worse than not measuring it.

[How alignable is your index?](https://batuhanzorbeyzengin.github.io/rebasis/exposure/)
is the full account, including the list of what the number does not say. It is
written to be read before the number rather than after it.

## What rebasis does not claim

The audit trail is **tamper-evident, not tamper-proof.** It is a local file with
a hash chain, and its owner can regenerate it. The chain exists to catch
accidental corruption and silent data loss, not to defend against a determined
local attacker. Claiming otherwise would be false.

## Supported versions

While the project is 0.x, only the latest release receives fixes.

## Known advisories in the dependency tree

Two packages in the resolved tree carry open advisories with **no fixed version
upstream**, so neither is closed by an upgrade. Both were assessed against how
rebasis actually uses them rather than dismissed, and the reasoning is here so a
reviewer does not have to repeat it.

**Identifiers are given as GHSA first**, because that is what a Dependabot alert
shows and matching an alert to a row here should not require a lookup. The PYSEC
alias follows where one exists, since that is the spelling `pip-audit` prints for
the same advisory.

**chromadb — four advisories, none reachable through rebasis.**

| Advisory | Also known as | What it is |
|---|---|---|
| [GHSA-f4j7-r4q5-qw2c](https://osv.dev/GHSA-f4j7-r4q5-qw2c) | PYSEC-2026-311, CVE-2026-45829 | Pre-authentication code injection via the `/api/v2/tenants/{tenant}/databases/{db}/collections` endpoint |
| [GHSA-36p7-vc44-83pf](https://osv.dev/GHSA-36p7-vc44-83pf) | CVE-2026-45833 | Authenticated code injection via the collection-update endpoint |
| [GHSA-2wm9-hf6c-p5cr](https://osv.dev/GHSA-2wm9-hf6c-p5cr) | CVE-2026-45830 | Missing authorisation validation across tenants |
| [GHSA-xph7-9rjv-w5fr](https://osv.dev/GHSA-xph7-9rjv-w5fr) | CVE-2026-45831 | `SimpleRBACAuthorizationProvider` ignores which tenant a permission applies to |

All four are properties of the **Chroma server** — its HTTP API, its
authentication and its authorisation providers. rebasis opens
`chromadb.PersistentClient(path=...)` and nothing else: there is no `HttpClient`
in the backend and the Chroma URI carries no host, so rebasis cannot connect to a
Chroma server at all, with or without these bugs.

That is a statement about rebasis, not about your deployment. **If you run a
Chroma server, these advisories apply to it** and rebasis is not what protects
you from them.

**diskcache — [GHSA-w8v5-vhqr-4h9v](https://osv.dev/GHSA-w8v5-vhqr-4h9v)
(PYSEC-2026-2447, CVE-2025-69872), unsafe pickle
deserialization.** It arrives through `llama-cpp-python`, an optional extra
deliberately left out of `rebasis[all]` because it compiles from source. The
attack requires write access to the cache directory, which is already local
compromise: an attacker who can write there can write to the rest of your
environment too. Not installing that extra removes the package.

Re-checked weekly by the `Audit` workflow, which runs `pip-audit` over the tree
`uv.lock` resolves with every extra. When a fix is published, the upgrade is a
lockfile change.

## Regulatory scope

Written down because a compliance reviewer would otherwise have to work it out,
and because the answers are short. **None of this is legal advice.**

**EU Cyber Resilience Act (Regulation (EU) 2024/2847).** rebasis appears to fall
outside it. The European Commission's own guidance states the CRA "does not apply
to developers who contribute with source code to free and open-source software
that are not under their responsibility", and that providing FOSS which its
maintainers do not monetise is not a commercial activity. The lighter
"open-source software steward" regime under Article 24 applies to a *legal
person* — a foundation or a company — and not to an individual. rebasis is
maintained by one person and is not monetised, so it is neither a manufacturer
nor a steward. If that changes — a company shipping it commercially, or a
foundation taking it on — the analysis changes with it and deserves an actual
legal read.

**Frameworks that certify organisations, not software.** SOC 2 examines a service
organisation's controls over a period; ISO/IEC 27001 and ISO/IEC 42001 certify an
organisation's management system; the EU AI Act places obligations on providers
and deployers of AI systems. rebasis is a library you run on your own machine
against your own index. It has no service for an auditor to observe, and claiming
any of those certifications *for the tool* would be a category error.

What it does offer a compliance programme is evidence. The audit trail records
who ran what, when, with which parameters — hash-chained, replayable, and
carrying no document content or vectors by construction. That is something a
deploying organisation can feed into its own SOC 2 or ISO 27001 controls. The
certification belongs to the organisation deploying it, not to the tool.
