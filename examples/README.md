# Examples

Three worked situations, each a real one rather than a toy.

| Example | The situation |
|---|---|
| [`obsidian_vault/`](obsidian_vault/) | A personal note vault indexed in Chroma. A better model came out. Re-embedding 40,000 notes is hours you do not want to spend. |
| [`codebase_rag/`](codebase_rag/) | Code chunks in LanceDB, where a code-specialised model would genuinely help — and the question is whether it helps *enough*. |
| [`otel/`](otel/) | Sending rebasis traces to your own collector. Off unless you turn it on; nothing leaves the machine otherwise. |

Each directory has a `README.md` explaining the situation and a runnable script.
None of them needs a GPU or a network connection beyond downloading the model
you choose.
