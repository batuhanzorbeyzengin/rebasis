`rebasis compare` — rank several candidate models on your own corpus, without rebuilding the index.

```bash
rebasis compare --store chroma:///path/db#docs --old <the index's model> \
  --candidates BAAI/bge-small-en-v1.5,BAAI/bge-base-en-v1.5 --queries queries.jsonl
```

`probe` already answers "is this model better on my corpus" from a sample. Asking it of N candidates is a different question in one respect that decides the whole design: what comes back is an **ordering**, and an ordering is the shape this instrument was measured to be good at. [Section 9](https://batuhanzorbeyzengin.github.io/rebasis/bridge-band/#9-what-the-counting-is-worth) demolished the estimate as a threshold — the count that said otherwise was an identity — and what survived was a rank correlation.

**One sample, one split, one reference.** Every candidate is scored on the same drawn sample, the same fit/held-out split and the same queries; only the embedding pass and everything after it is per candidate. A redraw per candidate would introduce a shift larger than some of the gaps being compared, and a comparison cannot survive rows that are not comparable. The index's own model is the **reference**, not a row: its vectors are already there.

It prints what it will cost before it runs, measured on your machine on 64 of your own documents, because N candidates is N embedding passes and a static 8M model and a 300M transformer differ by two orders of magnitude. A candidate on a hosted endpoint is named first, since a comparison sends the sample off the machine once per such candidate. `--tiered` scores everything on a small sample and carries through only what that round could not separate, on the same ±0.025 band the decision rule already reports its own borderline cases at.

**And the measurement behind it lost.** Over 16 corpora with three candidates each against human judgements, the null was "pick whatever scores highest on the published MTEB table". That null gets top-1 right 14 times out of 16. `compare` gets it right 9. Both numbers are in [which model, on your corpus](https://batuhanzorbeyzengin.github.io/rebasis/model-selection/), along with what the ordering *does* carry and how it moves with sample size — the command reports a table and a caveat rather than a winner, and that is why.
