The release workflow assembles the changelog. It did not, and a release cut before this would have shipped an unchanged `CHANGELOG.md` with every news fragment still sitting in `changelog.d/`.

`semantic-release` was called with `--no-changelog` and a comment explaining that towncrier owns the file — which is right, and the reason two writers on one file is the wrong design. Nothing then called towncrier. Neither `release.yml` nor `docs/development/release.md` had the step; the `justfile` had only `--draft`, which previews without writing.

The assembly is now the release commit, so the tag lands on a tree that contains the changelog rather than one commit behind it. `verify` also refuses to release at all when `changelog.d/` is empty — an empty directory means either nothing user-visible changed, or somebody forgot the fragment, and the second is worth catching.

The version and the tag are read from `semantic-release version --print` and `--print-tag`, both of which exit before touching anything, and the tag is then placed by `git`. The commit messages still choose the version and `tag_format` still decides the tag's shape; what changed is that the tagging step no longer depends on how the release tool behaves when there is no version file to rewrite — this project has none, since `hatch-vcs` derives the version from the tag.
