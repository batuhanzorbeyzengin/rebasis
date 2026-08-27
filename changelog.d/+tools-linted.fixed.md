CI lints `tools/`. The pre-commit hook always did and CI did not, which is the same divergence `ci.yml` already documents one directory over — and a fork's pull request never runs the hook.

It is not a quiet directory: `tools/check_citations.py` runs as a step in that very job, and `tools/bridge_band.py` is what every published measurement comes out of. The release workflow's paths are updated to match, because they diverging once is how the gap in `examples/` survived.
