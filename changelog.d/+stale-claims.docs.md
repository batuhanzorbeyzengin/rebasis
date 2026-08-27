Three documents said things that had stopped being true.

`README.md`'s decision-rule paragraph ended mid-sentence — "Above 1.0 bridging beats leaving things alone. It has" — and then jumped straight into the withdrawal of the count that sentence used to carry. The break is closed, and the headline "12 of those 62" now says what it is: how often the tool declined to recommend the adapter, which is not the same as how often it was right. The accuracy reading of that count is the identity that was withdrawn.

`ROADMAP.md` said "Linux and macOS run today" under what 1.0 needs. Only Linux runs. The macOS leg was added and then removed over the `faiss-cpu`/`torch` OpenMP conflict, which is the wrong thing to have lost — the storage layer is where the platforms actually differ and where a bug costs data. The entry now says so, and names the narrower leg that would clear the conflict.

`pyproject.toml` pointed at `tests/unit/test_coverage_floors.py` for the per-module coverage targets; that check is `tools/check_coverage_floors.py`, and it is a script rather than a test for a reason worth keeping written down. A second comment described an instruction-count benchmark as what the pull-request gate is built on, left behind when the CodSpeed dependency was removed.
