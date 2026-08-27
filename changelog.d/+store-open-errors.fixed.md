Four of the five store backends leaked their client library's exception when the store could not be opened.

`errors.py` states the rule in its own module docstring: *"Third-party exceptions never cross a module boundary. Each backend catches its own library's exception, converts it to a `RebasisError` subclass, and keeps the original as `__cause__`. Contract tests enforce this."* They did not enforce this half of it.

The half that was covered is a store that opened and then refused something — a dimension mismatch, a missing collection. The half that was not is the store that does not open at all, which is the more common thing to get wrong on a first run: a path typed with a missing directory, a database owned by another user, a volume that is not mounted yet.

Measured against a path that exists as a parent and refuses everything under it. Only FAISS converted. Chroma raised `chromadb.errors.InternalError`, LanceDB and Qdrant raised `FileNotFoundError`, sqlite-vec raised `sqlite3.OperationalError` — each reaching the caller with no `RB-Exxxx` code, no hint and nothing to look up.

All four now raise `StoreError` (`RB-E3000`) naming the path, with the original kept as `__cause__`. The code is deliberately not `RB-E3003`: a database that cannot be opened is a different problem from one that opened and does not hold the collection you named, and a user told the wrong one looks in the wrong place.

Found by running `rebasis doctor --store` against a bad path and reading what it printed. The tool diagnosed itself — beside the leaked exception it printed its own note that *"a backend is meant to convert its client library's exceptions into a rebasis error, so this one is a bug"*. It was.

A contract test now covers it on every backend. Its own first version derived the import name from the backend name, asked for `chroma` and `qdrant`, and skipped both while they were installed — which is the failure `ci.yml` greps for, and the reason it does.
