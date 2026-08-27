`SECURITY.md` says what the open advisories in the dependency tree are, and why neither is reachable through rebasis.

Five, in two packages, and none of them has a fixed version upstream — so neither is closed by an upgrade, and a reviewer who finds them needs the assessment rather than a promise to bump something.

**chromadb carries four**, all of them properties of the Chroma *server*: pre-authentication code injection through its collections endpoint, an authenticated variant of the same, missing authorisation validation across tenants, and an RBAC provider that never checks which tenant a permission applies to. rebasis opens `chromadb.PersistentClient(path=...)` and nothing else — there is no `HttpClient` in the backend and the Chroma URI carries no host — so it cannot reach a Chroma server at all. That is a statement about rebasis and not about your deployment: if you run a Chroma server, those advisories apply to it.

**diskcache carries one**, unsafe pickle deserialization, and arrives through `llama-cpp-python` — an optional extra deliberately outside `rebasis[all]` because it compiles from source. The attack needs write access to the cache directory, which is already local compromise.

Both are re-checked weekly by the `Audit` workflow, over the tree `uv.lock` actually resolves with every extra installed.
