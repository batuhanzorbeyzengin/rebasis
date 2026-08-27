CI runs on macOS again, on the one job that can carry it.

The macOS leg was removed from the full suite because `faiss-cpu` and `torch` each link their own OpenMP runtime there, and a process holding both aborts before either library does any work. What went with it was coverage of the **storage layer** — directory fsync, `os.replace`, a system sqlite3 that cannot load extensions — which is exactly where the two platforms differ and exactly where a bug costs data rather than accuracy.

A core-only install has neither library, so the conflict cannot arise. The `no torch, no otel` job was already installing nothing optional for its own unrelated reason, so a second operating system costs it one matrix entry and no new risk. It now runs the unit, property and contract layers on Ubuntu and on macOS.

This does not restore the full macOS suite: the backends, which is where `faiss-cpu` enters, still run on Linux only.
