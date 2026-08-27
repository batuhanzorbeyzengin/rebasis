The release tooling is in `uv.lock` instead of being resolved fresh on every run.

`release.yml` called `uv run --with python-semantic-release`, which bypasses the lock entirely and takes whatever is newest at that moment. GitPython 3.1.60 removed `Actor.name_email_regex`; python-semantic-release 10.6.1 reads it to parse `commit_author`. So the release workflow broke on a repository where nothing had changed, with `type object 'Actor' has no attribute 'name_email_regex'` and no version printed.

**A release path that can break from somebody else's upload is not a release path.** The tooling is now a `release` dependency group, pinned by the lock like everything else, with `gitpython<3.1.60` and the reason recorded next to it. Dependabot proposes the upgrade as a pull request rather than applying it silently at the worst moment.

`ci.yml` gains the other half: a ten-second step on every pull request that asks the tooling to work out a version and fails if it cannot. The breakage was invisible until somebody wanted a release; now it is visible on the change that introduces it.

Found by `mode: dry-run`, which is what that mode is for — it printed the error and stopped, having touched nothing.
