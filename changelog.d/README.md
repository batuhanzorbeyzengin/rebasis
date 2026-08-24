# News fragments

One file per user-visible change, added in the same PR as the change.

```
changelog.d/<issue-or-pr>.<type>.md
```

Types, in the order they appear in the changelog:

| Type | For |
|---|---|
| `behaviour` | A changed decision threshold, metric definition, or default. **Always include the reasoning and the measurement.** |
| `removed` | Something that no longer exists |
| `added` | New capability |
| `fixed` | A defect |
| `performance` | A measured speed or memory change |
| `docs` | Documentation worth mentioning in a release |

Write for someone upgrading who did not follow the work:

```markdown
<!-- changelog.d/142.behaviour.md -->
`probe` now compares bridging against keeping the current model, not only
against a full reindex. Measured on BEIR/scifact: bridging recovered 0.903 of a
reindex while keeping the current model gave 0.944 — so the old rule recommended
migrating to something measurably worse.
```

Not "fixed the decision rule". The subject line of the commit already says that;
the fragment is the part that is still useful in six months.

`chore`, `ci`, `style`, `refactor` and `test` changes need no fragment — if a
change has nothing to say to a user, saying nothing is correct.

Preview the assembled changelog without writing it:

```bash
uv run towncrier build --draft
```
