`Release` takes a `mode` instead of a `dry_run` boolean: `dry-run`, `rehearse`, or `release`.

`dry-run` prints the version the commits imply and the changelog that would be written, and stops. `release` is what it was. `rehearse` is new, and it is the one worth having before a first release: it assembles the changelog, commits it on the runner's throwaway checkout, tags locally, builds at that tag and uploads to TestPyPI — everything the real run does except the two irreversible parts, since nothing is pushed and nothing reaches PyPI. The job is not granted `contents: write`, which makes "nothing is pushed" a property rather than a promise.

What a rehearsal proves is what a dry run cannot: that towncrier assembles, that `hatch-vcs` reads the tag, and that Trusted Publishing works end to end for this repository. It needs a pending publisher registered on TestPyPI against the `testpypi` environment.

It commits rather than building a dirty tree because `hatch-vcs` appends a local version segment to a build made from uncommitted changes, and PyPI rejects local version identifiers — so a rehearsal from a dirty tree fails at the upload for a reason unrelated to the release. Both publishing jobs now check the built filenames for that segment and name it, rather than letting it surface as a rejected upload.

The pre-release suite also drops the `perf` layer, matching `ci.yml`. It was running the wall-clock assertions CI had already removed as unmeasurable on a shared runner, including the one that had gone red twice on noise — a release blocked by a 2.6% timing wobble is a release blocked by nothing. Those numbers are gated on the host before a release instead.
