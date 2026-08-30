`probe --truncate` and `--quantize` — what a cheaper representation of the index you already have would cost.

```bash
rebasis probe --store <uri> --queries queries.jsonl \
  --truncate 1024,512,256,128 --quantize float32,float16,int8,binary --floor 0.95
```

The most common index transformation in the field is not a model change. It is a cut in dimension and precision, and it raises the question `probe` already answers: what do I lose, on **my** corpus rather than on a benchmark average? No `--old`, no `--new` and no adapter — the reference is the index's own full-width, float32 state and every cell is the same vectors held more cheaply.

Three properties are worth knowing before reading a grid.

**A whole grid costs what one probe costs.** The model runs once; truncating and quantizing what it produced is free. Sixteen cells are sixteen searches over arrays already in memory.

**Both sides are cut, always.** Truncating documents while leaving queries at full width compares coordinates that no longer correspond — not a cheaper index but a broken one — and every truncation renormalises, because cutting a unit vector's tail leaves it shorter by an amount that differs per document.

**The precision axis is a simulation and says so in the report.** rebasis produces float32 and each store narrows it in its own way; `sqlite-vec`'s `int8`, pgvector's `halfvec` and Qdrant's `datatype` are three different narrowings. A cell measures what the arithmetic costs, which is a lower bound on what a particular codec costs. The dimension axis is not simulated: truncating a vector is the whole operation.

Every cell carries **two** retentions and a confidence interval. The second is what the cell returns when the full-precision vectors reorder its top candidates — the cascade's shape on a different axis, and unlike the cascade it costs no embedding at all, because those vectors are the ones the index already holds. On the binary row the two are far apart, which is the finding: measured on BEIR/scifact with bge-base, binary retains **0.905** on its own at full width and **0.999** rescored, and **0.695 against 0.964** at 256 dimensions.

**And Matryoshka training bought nothing measurable.** `mxbai-embed-large-v1`, whose card documents MRL support, was measured against `bge-base` over the same sixteen corpora: the MRL model leads at three of four depths by at most 0.007, against a corpus-to-corpus spread of 0.037 to 0.276 at those same depths, and the sign of that lead flips if you take medians instead of means. Which corpus you run on moves the answer between five and forty times further than which of the two models you run. That is [Takeshita et al.](https://arxiv.org/abs/2605.16608)'s headline reproduced on a different corpus family. It bought no steadiness either — the spread is 0.062 against 0.065 at 256 and 0.276 against 0.249 at 64. `embed/data/profiles.json` records MRL support from the model's own card and from nowhere else; a card that is silent gets no entry and the flag measures the model rather than assuming.

`--floor 0.95` names the cheapest cell that clears a quality floor — a Pareto choice rather than a break-even, because quality and cost are two axes and which matters more is not the tool's decision. Where the chosen cell's interval spans the floor, the run says it cannot settle it.

Writing the result back is deliberately **out of scope**: going from `vector(1024)` to `vector(256)` means recreating the column, and `migrate` changes vectors rather than schemas. That is the line that keeps rebasis from becoming a vector database.
