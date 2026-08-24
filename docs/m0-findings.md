# M0 Spike — Measurement Findings

**Testing the technical design's assumptions on real hardware**

| | |
|---|---|
| Phase | M0 |
| Date | 2026-08-22 |
| Code | `spikes/m0_spike.py`, `spikes/device_compare.py`, `spikes/knn_threshold.py` |
| Raw output | `reports/m0-*.json`, on the host that ran it |
| Scope | 4 corpora × 3 model pairs × 7 adapters = **84 configurations** |

> **What this document is not:** a design proposal or a decision. It is a record
> of what was measured and of which clauses in the technical design those
> measurements contradict. The decisions themselves are taken before M1, in light
> of these numbers.

---

## 0. Executive summary

M0 had three tasks. All three are answered, and five unplanned findings came out
alongside them.

| # | Open question | Answer | Affects |
|---|---|---|---|
| 1 | Is the T0 ground-truth proxy valid? | **Partly.** Strong as a ranking signal (Pearson 0.91), weak as an absolute value (±0.095). The problem is not the proxy but the ground-truth definition. | T0 ground truth, decision bands, borderline band |
| 2 | Does CSLS contribute to ARR? | **Conditionally.** +0.103 on a weak adapter, **−0.045** on a strong one. Not free — a trade. | CSLS |
| 4 | Is isotonic calibration enough for threshold filters? | **Yes, partly.** KS 0.924 → 0.094 and ranking preserved 100% of the time. | Calibration, `score_shift` |
| 9 | Are the performance budgets realistic? | **No, in both directions.** Fit budgets are 30–600× too loose; the hot-path budget cannot be met. | Performance budgets, device projections |
| 14 | Where is the GPU threshold for kNN? | **There is none.** Even at 2,000 documents the GPU is 22× faster, transfer included. | GPU policy |

**Unplanned findings:**

| # | Finding | Affects |
|---|---|---|
| A | **Removing the mean before Procrustes raises ARR by +0.26 on average** (+0.75 in the best case). This step was not planned at all. | Preprocessing, adapter table |
| B | Fixed-threshold score filters break for **every** adapter (100% exceed KS 0.1). The "warn when score_shift > 0.1" rule therefore discriminates nothing. | `score_shift`, decision rule |
| C | The "two adapters for asymmetric models" rule is **neutral on average** (−0.003); the effect changes sign between model pairs, ranging over ±0.10. | Asymmetric models |
| D | The Residual MLP memory figure (1.6 MB) is inconsistent with its own formula; the linear term is not counted. Actual: 3.76 MB at d=768. | Adapter table |
| E | On the test corpora, **doing nothing at all yields a mean ARR of 0.983**. The decision rule has no branch for that. | Decision rule |

---

## 1. Measurement environment

All measurements were taken **on the server**. The only local numbers are one
hot-path comparison, labelled as such.

| | Server (primary) | MacBook (comparison only) |
|---|---|---|
| Hardware | AWS `g5.2xlarge`, eu-central-1b | Apple Silicon, arm64 |
| CPU | 8 vCPU | 10 vCPU |
| RAM | 31 GB | 32 GB |
| GPU | NVIDIA A10G, 23 GB, driver 595.71.05 | MPS (unified memory) |
| OS / kernel | Ubuntu 24.04, 7.0.0-1011-aws | macOS 26.3 |
| Python | 3.12.3 | 3.14.2 |
| torch | 2.11.0+cu128 (CUDA 12.8) | 2.13.0 |
| numpy / scipy / sklearn | 2.5.2 / 1.18.1 / 1.9.0 | same |
| sentence-transformers | 6.0.0 | 6.0.0 |

**The Python versions differ (3.12.3 / 3.14.2).** This did not affect the M0
results, but it should be aligned in M1 for reproducibility — the determinism
fixture cannot paper over a version gap.

**TF32.** Explicitly disabled at the start of every measurement run. The
reason for disabling it was also measured: on the A10G, a 4096³ matmul with TF32
enabled deviates from CPU by a maximum absolute error of **0.106**, versus
**0.001** with it disabled — roughly 100× the deviation in exchange for 30% more
throughput (23.4 → 30.5 TFLOP/s). For a decision metric like ARR that trade is
unacceptable. Note that `torch.backends.cudnn.allow_tf32` defaults to **on** in
PyTorch; embedding models do not take the convolution path so it is currently
inert, but it belongs on the disable list regardless.

### 1.1 Corpora

Four BEIR corpora were used so that nothing is generalised from a single one.
All four have **real queries and human relevance judgements (qrels)** — the only
way to measure T0 against T1.

| Corpus | Documents | Queries | qrels | Fit pairs | T0 queries |
|---|---|---|---|---|---|
| `beir/scifact/test` | 5,183 | 300 | 339 | 4,183 | 1,000 |
| `beir/nfcorpus/test` | 3,633 | 323 | 12,334 | 2,907 | 726 |
| `beir/scidocs` | 25,657 | 1,000 | 4,928 | 24,657 | 1,000 |
| `beir/arguana` | 8,674 | 1,406 | 1,406 | 7,674 | 1,000 |

Data was fetched through `ir_datasets`. **Pulling `BeIR/*` through `datasets` is
wrong here:** it returns 1,109 queries (all splits) while the qrels cover only
the test split, so queries without judgements silently corrupt the recall
denominator. `ir_datasets` returns the 300 queries belonging to the split.

### 1.2 Model pairs

| Old → New | Dimension | Symmetry | Why chosen |
|---|---|---|---|
| `all-MiniLM-L6-v2` → `bge-small-en-v1.5` | 384 → 384 | symmetric → asymmetric | The "trap" scenario: collapses if prefix handling breaks |
| `all-MiniLM-L6-v2` → `e5-small-v2` | 384 → 384 | symmetric → asymmetric | A different prefix scheme (`query: `/`passage: `) |
| `e5-small-v2` → `bge-small-en-v1.5` | 384 → 384 | asymmetric → asymmetric | Both sides prefixed, schemes differ |

**Dimension change was not tested in M0.** All three pairs are 384→384.
The `d_new ≠ d_old` path exists in the code (zero-pad plus truncate) and is
verified by the synthetic test, but it was never measured on real models — that
is M1's job.

---

## 2. Method

### 2.1 Adapter direction and contract

Direction is `query→old`, the default: `g: new space → old space`. Every fit call
is `fit(src, dst)` with `apply(src) ≈ dst`, where `src = f_new(d)` and
`dst = f_old(d)`. This matches the contract the property tests assert.

Preprocessing: ℓ2 normalisation before fitting, renormalisation after applying.

### 2.2 Leakage control

The fit set and the query set are **strictly disjoint**. Disjointness is
verified at runtime on every run; an intersection aborts the run with an error.
In T0 the query chunk sits inside the index, so it is masked out of its own
results — without the mask every query retrieves itself and ARR inflates.

### 2.3 Two different definitions of ARR — and why they must not be conflated

This is M0's most important methodological point.

**T0 (the proxy):** ground truth is the new model's exhaustive in-sample
top-k neighbours — literally "what would I get after a full reindex". Oracle
recall is 1.0 by construction, so `ARR_T0 = recall@k(adapted)`.

**T1 (real queries):** ground truth is human judgement (qrels). The oracle
is no longer 1.0 — even a full reindex does not achieve perfect recall. So
`ARR_T1 = recall@k(adapted) / recall@k(oracle)`, a genuine ratio.

**These are not the same quantity.** T0 says "reproduce the new model's neighbour
SET"; T1 says "be as good as the new model at finding the relevant document".
The second is markedly easier. Section 3 measures how large that difference is.

### 2.4 Memory and compute discipline

Preserved in M0 because it affects the measurement itself:

- the float32 contract — nothing is promoted to float64 (except
  `LinearAdapter`'s normal equation, where the condition number is squared, so
  it is solved in float64 and returned to float32 at the boundary)
- the full distance matrix is never materialised; chunked matmul plus
  `argpartition`
- `topk_search` is verified to match a reference brute-force `argsort` exactly

---

## 3. Finding 1 — validity of the T0 ground-truth proxy

> Open question 1: *"The correlation of the document-as-query proxy with the
> real query distribution must be measured in M0. **The backbone of the project;
> it should be measured before even the adapter validation.**"*

### 3.1 Raw result

T0 and T1 were measured side by side across 84 configurations.

| | mean \|ARR_T0 − ARR_T1\| | worst | signed error | **decision agreement** | Pearson r | Spearman ρ |
|---|---|---|---|---|---|---|
| **T0 strict GT** (full top-10 kNN) | 0.2613 | 0.5823 | −0.2607 | **41.7%** | 0.888 | 0.764 |
| **T0 relaxed GT** (nearest neighbour only) | **0.0954** | 0.2750 | **+0.0036** | **53.6%** | **0.907** | 0.735 |

### 3.2 Decomposition: is the proxy at fault, or the metric?

The 0.26 gap between strict T0 and T1 had two candidate sources:

- **(a) the proxy** — standing documents in for queries, the caveat T0 carried
- **(b) the metric** — T0 asks for the whole kNN set to be reproduced, while T1
  asks only that the relevant document (≈1.1 per query under sparse qrels) lands
  in the top k

To separate them, T0 was re-measured with T1's metric shape: only the **nearest
single neighbour** of the ground truth counts as relevant, the structural
equivalent of qrel sparsity. The result is in the table: the error drops from
**0.2613 to 0.0954**, a **5.5× reduction**.

> **Conclusion: the document-as-query proxy is sound. The problem is the
> ground-truth definition.**

This shows that T0's own caveat — "T0 assumes the query distribution
resembles the document distribution" — is **not the dominant error source** in
the M0 data. The caveat remains true and belongs in the report, but what actually
needs fixing is the strictness of the ground truth.

### 3.3 No systematic bias — but real noise

Looking at a single corpus (scifact), T0 appeared systematically pessimistic
(signed error −0.061, with all 8 disagreements pointing the same way). **Across
four corpora that disappears:** the signed error is **+0.0036**. The pessimism
was a property of that corpus, not of T0.

This is a concrete example of how working from a single corpus leads to the wrong
design decision: had we looked only at scifact and added a fixed +0.06 offset to
T0, the other three corpora would have got worse.

**Unbiased but noisy:** the mean error is near zero, the mean absolute error is
±0.095.

### 3.4 The critical measurement — the uncertainty of the reference itself

A 53.6% decision agreement looks like T0's failure. But T1 is itself computed
from a finite sample of queries. Bootstrap (2,000 resamples, paired at the query
level — ARR is a ratio, so numerator and denominator must be resampled from the
same queries; resampling them independently would inflate the interval):

| Measurement | median 95% CI half-width | worst |
|---|---|---|
| ARR_T0 | ±0.0241 | ±0.0351 |
| **ARR_T1** (the "ground truth" being compared against) | **±0.0423** | **±0.1061** |
| Decision band width | 0.10 | — |

Combined measurement noise ≈ √(0.024² + 0.042²) ≈ **0.048**. The observed mean
error is 0.095 — so roughly **half of the disagreement is measurement noise**,
not T0's error.

### 3.5 Design implications

| Area | What was assumed | What the measurement says |
|---|---|---|
| T0 definition | "GT = the new model's exhaustive in-sample kNN" | The full kNN set is too strict. A sparsity-matched GT cuts the error 5.5×. |
| Decision bands | 0.95 / 0.85 / 0.70, width 0.10 | Measurement uncertainty (±0.024–0.042) is the same order as the band width. Decisions near a threshold are settled by sampling noise. |
| Borderline band | ±0.005 | **An order of magnitude too narrow.** A realistic figure is at least ±0.025 (T0), ±0.045 with real queries. |
| ARR as "the main decision metric" | Compared against absolute thresholds | T0 is strong as a **ranking** signal (Pearson 0.91), so `auto`'s adapter selection is well founded. It is not sufficient alone for an absolute threshold call. |

**The most defensible reading:** T0 is reliable at ordering adapters relative to
one another, and not at placing a single adapter on one side of a fixed
threshold. `auto` picking the best adapter is supported by the M0 data;
`probe` declaring "bridging is enough" versus "reindex" requires either a wider
uncertainty band or an explicit "the measurement cannot make this distinction"
output.

---

## 4. Finding A — mean centering before Procrustes *(absent from the design)*

M0's unplanned finding with the largest effect.

### 4.1 How it surfaced

In the first run, `procrustes` landed far below `linear` (T0 ARR 0.089 vs 0.464
on MiniLM→e5) — the reverse of the reference work's ordering, which put OP at
0.95–0.97.

The structural difference: `LinearAdapter` subtracts the mean of both sides
before fitting (carrying the translation separately as `b`), while
`ProcrustesAdapter` does not.

### 4.2 Hypothesis

Plain OP searches for a rotation **through the origin**. Embedding spaces are not
isotropic — they carry a large shared mean component (anisotropy). When the two
spaces' means point in different directions, no rotation pinned to the origin can
close that gap. The cross-lingual alignment literature (MUSE, vecmap)
already applies centering before Procrustes as a standard step.

### 4.3 Measurement

`g(x) = (x − μ_src)·R + μ_dst` with `RᵀR = I`. Orthogonality is preserved (still
a rigid rotation in the centred space); the extra cost is two vectors (2d
parameters).

| Corpus | Model pair | ΔT0 | ΔT1 |
|---|---|---|---|
| scifact | MiniLM→bge | +0.1955 | +0.1617 |
| scifact | MiniLM→e5 | +0.2900 | **+0.5834** |
| scifact | e5→bge | −0.0024 | +0.0000 |
| nfcorpus | MiniLM→bge | +0.1972 | +0.1783 |
| nfcorpus | MiniLM→e5 | +0.2351 | +0.2533 |
| nfcorpus | e5→bge | +0.0019 | −0.0019 |
| scidocs | MiniLM→bge | +0.2636 | +0.4159 |
| scidocs | MiniLM→e5 | +0.2860 | **+0.6754** |
| scidocs | e5→bge | −0.0021 | −0.0178 |
| arguana | MiniLM→bge | +0.1505 | +0.1400 |
| arguana | MiniLM→e5 | **+0.3764** | **+0.7478** |
| arguana | e5→bge | −0.0063 | −0.0168 |
| **mean** | | **+0.1655** | **+0.2599** |
| median | | +0.1964 | +0.1700 |

Cases where it hurt (< −0.005): 1 of 12 at T0, 2 of 12 at T1 — all on the
`e5→bge` pair, all with magnitude ≤ 0.018, i.e. within measurement noise.

**The pattern:** the gain is large on `MiniLM→*` pairs and zero on `e5→bge`. A
plausible explanation is that e5 and bge's prefixed encoding schemes produce
similar anisotropy, so their means are already aligned, whereas MiniLM's
prefix-free space sits somewhere else.

### 4.4 Effect on the adapter ranking

Mean over 4 corpora × 3 pairs (T1 ARR):

| Adapter | T1 ARR mean | median | min | max | T0ᵍᵗ¹ mean | Memory (d=384) |
|---|---|---|---|---|---|---|
| `linear` | 0.8390 | 0.8433 | 0.6433 | 1.1410 | 0.8427 | 0.56 MB |
| `residual_mlp` | 0.8371 | 0.8497 | 0.6442 | 1.1366 | 0.8427 | 1.32 MB |
| `procrustes_centered+dsm` | 0.8365 | 0.8249 | 0.5912 | 1.0253 | 0.8414 | 0.57 MB |
| **`procrustes_centered`** | **0.8343** | 0.8613 | 0.5270 | 0.9824 | 0.8351 | **0.56 MB** |
| `procrustes` (uncentred) | 0.5744 | 0.6742 | 0.1126 | 0.9514 | 0.5735 | 0.56 MB |
| `low_rank_affine` (r=64) | 0.4584 | 0.5091 | 0.0282 | 0.8932 | 0.4739 | 0.19 MB |
| `identity` (no adaptation) | 0.2741 | 0.2625 | 0.0833 | 0.4845 | 0.2701 | 0 |

The spread across the top four (0.8343–0.8390) is far below the measurement
uncertainty (±0.042) — they are **statistically indistinguishable**. When that is
the case the choice falls to cost, and centred Procrustes wins: half the MLP's
memory, 2.7× lower latency (section 7), orthogonality's resistance to
overfitting, and a closed form (no iteration, no seed, determinism for free).

`low_rank_affine` (rank 64) lands far below expectation — LA was projected at
0.97–0.98, the measurement gives 0.4584, below even plain Procrustes. At
d=384, rank 64 retains only 17% of the dimensions; here truncation is information
loss rather than regularisation. The rank may need to scale with the dimension —
to be tested in M1.

### 4.5 Design implications

- **Preprocessing** was ℓ2 normalisation only. Centering should be added; the
  measured gain is +0.26 on average (T1) and +0.75 in the best case.
- **The adapter table** has no centred variant of `procrustes`. The measurement
  shows centred OP delivering MLP-equivalent quality far more cheaply.
- **CSLS** was already taken from MUSE/vecmap; the centering step from the same
  literature should be taken too.

---

## 5. Finding 2 — the CSLS hubness correction

> Open question 2: *"How much does CSLS contribute? The reference work never
> tried it. If it is meaningful it becomes rebasis's first original
> contribution."*
> And the expectation it carried: *"Applying CSLS at the adapter output could
> raise ARR for free."*

### 5.1 Implementation note — the factor of 2

`CSLS(q,d) = 2·cos(q,d) − r_T(d) − r_S(q)`. For ranking purposes `r_S(q)` is
constant per query and drops out. Ranking by the remaining `2·cos − r_T(d)` is
**identical** to ranking by `cos − r_T(d)/2` — so the per-document bias is
`−r_T(d)/2`, not `−r_T(d)`. Dropping the factor of 2 doubles the penalty and
makes CSLS look worse than it is.

Cost: computed once per document, zero per query. The hot path is untouched.

### 5.2 Measurement

| Tier | mean Δ | median Δ | range | fraction improved |
|---|---|---|---|---|
| T0 | +0.0527 | +0.0369 | [−0.0718, +0.2403] | 85% |
| T1 | +0.0186 | +0.0038 | [−0.1828, +0.4111] | 51% |

Reading "it helps" off the mean would be misleading. Split by adapter quality,
the picture inverts:

| Adapter quality | mean Δ | n |
|---|---|---|
| Weak (ARR < 0.5) | **+0.1028** | 71 |
| Strong (ARR ≥ 0.8) | **−0.0452** | 36 |

The **Spearman correlation between Δ and ARR is −0.704**.

### 5.3 Interpretation

Hubness forms when the adapter maps poorly: a weak transform crowds documents
into a narrow region, a handful become everyone's neighbour, and CSLS corrects
that. When the adapter maps well there is no crowding to correct, and CSLS's
penalty distorts genuine signal instead.

> **CSLS is not a free improvement but a trade: a safety net for a weak adapter,
> a liability for a strong one.**

### 5.4 Design implications

- The phrasing "could raise ARR for free" is not supported by the measurement
  and should be made conditional.
- CSLS should **not be always-on**. The right place is what `auto` already does:
  evaluate the CSLS and non-CSLS variants on the held-out set and keep the better
  one. No new mechanism is needed, only one more variant in the existing
  selection loop.
- The expectation of "rebasis's first original contribution to the literature" is
  not met in this form. The stronger candidate turned out to be **mean centering**
  (section 4).

---

## 6. Findings 4 and B — score shift and isotonic calibration

> The design: *"`score_shift` is rarely discussed and critical: many RAG pipelines
> use a fixed threshold such as `similarity > 0.7`"* and *"when
> `score_shift > 0.1`, warn independently of the decision band"*.
> Open question 4: *"Is isotonic calibration enough for threshold-based
> filters?"*

### 6.1 The size of the problem — the threshold discriminates nothing

Without calibration, the KS distance between adapted and oracle scores:

| Adapter | median KS | worst | fraction with KS > 0.1 |
|---|---|---|---|
| `linear` | 0.923 | 1.000 | **100%** |
| `low_rank_affine` | 0.993 | 1.000 | **100%** |
| `procrustes` | 0.920 | 1.000 | **100%** |
| `procrustes_centered` | 0.928 | 1.000 | **100%** |
| `procrustes_centered+dsm` | 0.925 | 1.000 | **100%** |
| `residual_mlp` | 0.921 | 1.000 | **100%** |

**Every adapter, on every corpus, for every model pair exceeds that warning
threshold.** A warning that always fires carries no information.

The cause is structural: the adapter moves the vector into the old space, and the
cosine similarity distribution there is entirely different from the one in the
new space. That is not a bug but a natural consequence of the transform.

**The product consequence is serious:** every user with a fixed similarity
threshold has their filter broken by bridging, even when ranking is preserved
perfectly.

### 6.2 Isotonic calibration measurement

The proposed calibrator was implemented: it is fitted on **one third** of the
held-out queries and measured on the **other two thirds**. (Fitting and measuring
on the same scores would flatter the result.)

| | median KS | fraction under threshold | ranking preserved |
|---|---|---|---|
| **Before** calibration | 0.924 | 0% (i.e. 100% exceed it) | — |
| **After** calibration | **0.094** | **54%** | **100%** |

n = 72 (adapter × model pair × corpus).

### 6.3 Interpretation

- Calibration reduces the score shift by **an order of magnitude** (0.924 →
  0.094).
- The monotonicity claim is confirmed: ranking is preserved **100%** of the time,
  so calibration does not change ARR at all — it only moves the score scale.
- It is not sufficient, however: even after calibration **46%** of cases remain
  above the 0.1 threshold.

> **Calibration is correct and necessary, not optional. The claim that it is
> sufficient is not supported by the M0 data.**

### 6.4 Design implications

- The `score_shift > 0.1` warning must be evaluated **after**
  calibration. Measured before it, it is constantly true and therefore
  uninformative.
- Storing the calibrator inside the `.rbs` file is right. The fallback
  path — "if there is no calibrator, use RRF" — becomes more important still,
  because without a calibrator absolute scores are unusable.
- The report needs an explicit warning that is independent of ARR and affects
  every user: *"If you use a fixed similarity threshold, you will need to retune
  it after bridging."*

---

## 7. Findings 9 and D — the hot-path budget

> The budget: *"Adapter apply — single query, d=768: **under 15 µs**; batch of 256:
> **under 1.5 ms**. Logging, dict copying and validation are forbidden on this
> path."*
> The adapter table: OP ~3 µs, MLP ~8 µs (d=768, from the reference work)

Measured as the median of 7 repetitions after warm-up, including `apply` plus
renormalisation — what the hot path actually does.

### 7.1 d=768 — the dimension the budget is defined for (server CPU)

| Adapter | Memory | Single query | Budget 15 µs | Batch 256 | Budget 1.5 ms |
|---|---|---|---|---|---|
| `low_rank_affine` | 0.38 MB | 23.87 µs | ✗ (1.6×) | 0.478 ms | ✓ |
| `procrustes` | 2.25 MB | 33.85 µs | ✗ (2.3×) | 1.073 ms | ✓ |
| `linear` | 2.25 MB | 34.30 µs | ✗ (2.3×) | 1.133 ms | ✓ |
| `procrustes_centered` | 2.26 MB | 41.65 µs | ✗ (2.8×) | 1.190 ms | ✓ |
| `procrustes_centered+dsm` | 2.26 MB | 44.02 µs | ✗ (2.9×) | 1.245 ms | ✓ |
| `residual_mlp` | 3.76 MB | 91.11 µs | ✗ (6.1×) | 2.816 ms | ✗ (1.9×) |

**No adapter meets the 15 µs budget at d=768.**

### 7.2 d=384 — hardware comparison

| Adapter | MacBook (Apple Silicon) | Server (8 vCPU x86) | ratio |
|---|---|---|---|
| `procrustes` | 7.12 µs ✓ | 21.80 µs ✗ | 3.1× |
| `linear` | 7.81 µs ✓ | 24.64 µs ✗ | 3.2× |
| `low_rank_affine` | 7.74 µs ✓ | 17.40 µs ✗ | 2.2× |
| `residual_mlp` | 19.52 µs ✗ | 53.68 µs ✗ | 2.8× |

Apple Silicon is roughly **3× faster** than a cloud vCPU on these small
operations. The budget holds on the MacBook at d=384 and fails on the server.

### 7.3 The bottleneck is call overhead, not FLOPs

Micro-breakdown of the MLP `apply` (MacBook, d=384):

| Operation | time |
|---|---|
| `x @ W (384×384) + b` | 3.47 µs |
| `x @ W1 (384×256) + b1` | 2.78 µs |
| GELU — `scipy.special.erf` | 3.62 µs |
| GELU — tanh approximation | **5.26 µs** |
| `h @ W2 (256×384) + b2` | 2.82 µs |
| `l2_normalize(1×384)` | 3.45 µs |
| **total `mlp.apply()`** | **14.95 µs** |
| fused variant (single matmul + tanh GELU) | 13.95 µs |

**Every numpy operation on a 1×384 array costs ~2.5–3.5 µs regardless of the
actual FLOPs.** Procrustes performs one operation, the MLP five. The budget
overrun is entirely Python/numpy call overhead. Two observations confirm this:

- **the tanh approximation is slower than `erf`** (5.26 vs 3.62 µs) — fewer
  FLOPs, more numpy operations. The intuition that "an approximation is faster"
  does not hold at this scale.
- **fusing does not help**: computing `W` and `W1` in one matmul takes 14.95 →
  13.95 µs, i.e. 1 µs. Eliminating one of five operations does not save a fifth,
  because the slicing needed to split the result eats the gain.

**Batching removes the problem entirely:** at batch 256 the MLP costs 4.3 µs per
query (1.092 ms ÷ 256, MacBook), far inside the budget. The issue is not compute
but per-call fixed cost on single items.

### 7.4 An inconsistency in the memory table

| Adapter | Projected (d=768) | Measured (d=768) | Status |
|---|---|---|---|
| Orthogonal Procrustes | ~2.4 MB | 2.25 MB | ✓ |
| Low-Rank Affine (r=64) | ~0.4 MB | 0.38 MB | ✓ |
| Residual MLP (256 hidden) | ~1.6 MB | **3.76 MB** | ✗ |

1.6 MB is exactly the size of `W₁ + W₂` (2 × 768 × 256 × 4 B = 1.57 MB). In other
words, the linear `Wx` term from the formula (`Wx + W₂·GELU(W₁x+b₁)+b₂`)
**was not counted in the memory figure.** That term adds 2.25 MB at d=768.

### 7.5 Design implications

- The 15 µs budget is not defined for any single hardware class. The measured
  range runs from 7 µs (Apple Silicon, d=384) to 91 µs (cloud vCPU, d=768). It
  should either be split by hardware class or redefined as a **regression gate**,
  which is what the instruction-count benchmark already does, rather than an
  absolute threshold.
- The "hot path is categorically CPU" decision is **confirmed**, and
  its rationale strengthened: single-query cost is already dominated by call
  overhead, and adding a GPU transfer can only make it worse.
- If the MLP is used on the hot path it must be **batched**. Where single-query
  latency matters, centred Procrustes is the right choice: the same quality
  (section 4.4) at 2.7× lower latency.
- The memory column needs correcting.

---

## 8. Finding 14 — the GPU threshold for kNN

> The design: *"Ground truth kNN — GPU gain is **borderline**. Threshold-based: GPU
> above N > 50k, CPU below."*
> Open question 14: *"Faiss's observation that 'a few thousand is faster on
> CPU, hundreds of thousands on GPU' puts our 10–20k sample right on the line."*

Measured with 10,000 queries against a varying number of documents, d=384, k=10,
chunked so the full score matrix is never materialised. **Two scenarios measured
separately:** vectors already resident on the GPU (as they are when embeddings
were produced there), and vectors on the CPU requiring transfer. The first
measurement left transfer outside the timer; since the "borderline" verdict
is precisely about transfer cost, both are reported.

| Documents | CPU (numpy) | GPU (resident) | GPU (+transfer) | speedup (resident) | **speedup (+transfer)** |
|---|---|---|---|---|---|
| 2,000 | 0.192 s | 0.003 s | 0.009 s | 76.1× | **22.0×** |
| 5,000 | 0.423 s | 0.005 s | 0.011 s | 83.5× | **39.1×** |
| 10,000 *(default sample)* | 0.855 s | 0.009 s | 0.017 s | 92.9× | **49.2×** |
| 20,000 | 1.602 s | 0.017 s | 0.028 s | 92.5× | **56.5×** |
| 50,000 *(the proposed threshold)* | 3.566 s | 0.042 s | 0.062 s | 84.1× | **57.3×** |
| 100,000 | 6.819 s | 0.082 s | 0.119 s | 83.1× | **57.3×** |
| 200,000 | 13.420 s | 0.167 s | 0.231 s | 80.6× | **58.2×** |

### 8.1 Interpretation

**There is no threshold.** Even at the smallest scale tested (2,000 documents)
and with transfer included, the GPU is 22× faster. The "N > 50k" threshold does
not apply to this workload.

The apparent conflict with the cited Faiss observation dissolves on inspection:
that observation concerns **ANN index search** with small query batches, where
per-query fixed cost dominates. Our workload is a **batched exhaustive kNN over
10,000 queries** — a single large dense matmul, exactly what a GPU is built for.
The design document applied a sound source to the wrong workload shape.

### 8.2 Design implications

- The kNN row should read "use the GPU when available" rather than "borderline /
  threshold-based".
- **Open question 14** can be closed. There is no need to hold a kNN threshold in
  `compute/thresholds.py`, and `doctor`'s local calibration is unnecessary for
  this operation.
- The budget of "ground truth kNN 10k×10k under 30 s" is very loose: even
  on CPU it takes 0.855 s, 35× inside the budget.

---

## 9. Device breakdown

Taken on the same host, with the same corpus, in the same run — measuring on
separate machines and comparing would be misleading.

### 9.1 Embedding generation — where the GPU actually pays

| Model | CPU (8 vCPU) | CUDA (A10G) | MPS (Apple Silicon) | CUDA/CPU |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 29 docs/s | **736 docs/s** | 62 docs/s | **25.4×** |
| `bge-small-en-v1.5` | 10 docs/s | **399 docs/s** | 29 docs/s | **40.0×** |

The projection was "CUDA 5-10× CPU, MPS 2-3× CPU" for `migrate` throughput.
Measured: **CUDA 25-40×**, MPS ~2-3× relative to CPU. The CUDA projection is
markedly conservative.

The decision — "embedding generation, GPU gain very high, always use the GPU
when available" — is strongly confirmed.

### 9.2 Adapter fitting (20,000 pairs)

| Job | d | CPU | CUDA | ratio | Budget | status |
|---|---|---|---|---|---|---|
| Orthogonal Procrustes | 384 | 0.15 s | — | — | < 20 s | **133× inside** |
| Orthogonal Procrustes | 768 | 0.29 s | — | — | < 20 s | **69× inside** |
| Linear (ridge) | 384 | 0.15 s | — | — | < 90 s (LA) | **600× inside** |
| Linear (ridge) | 768 | 0.32 s | — | — | < 90 s (LA) | **281× inside** |
| Residual MLP (30 epochs) | 384 | 2.52 s | 0.77 s | 3.3× | < 3 min | **71× inside** |
| Residual MLP (30 epochs) | 768 | 5.86 s | 0.99 s | **5.9×** | < 3 min | **31× inside** |

The "Orthogonal Procrustes fit — no GPU gain, always CPU" call is confirmed: at
0.15–0.29 s the transfer cost would exceed the computation itself.

**Every one of the fit budgets is 30–600× too loose.** Converted into CI gates as
planned, they would catch no regression whatsoever.

### 9.3 Overall conclusion

The closing claim — *"the GPU pays off in embedding generation;
everything else is either already fast or not worth moving"* — is **partly
confirmed and partly in need of correction:**

- ✅ Embedding: confirmed, and stronger than projected (25-40×)
- ✅ OP / Linear fitting: confirmed, stays on CPU
- ✅ Hot path: confirmed, stays on CPU
- ❌ **kNN: "not worth moving" is wrong.** It pays 22-58× at every scale.

---

## 10. Sample-size curve

> The design: *"0.97+ ARR at 5,000 pairs; saturating at 16,000."*

`beir/scidocs` (25,657 documents), MiniLM→bge, 1,000 held-out queries:

| Matched pairs | T0 ARR | T1 ARR | gain over previous step (T1) |
|---|---|---|---|
| 250 | 0.2913 | 0.4670 | — |
| 500 | 0.3794 | 0.5890 | +0.122 |
| 1,000 | 0.4349 | 0.7014 | +0.112 |
| 2,000 | 0.4684 | 0.7805 | +0.079 |
| **4,000** | 0.4826 | **0.8161** | +0.036 |
| 8,000 | 0.4951 | 0.8177 | **+0.002** |
| 16,000 | 0.5002 | 0.8155 | −0.002 |
| 24,000 | 0.5018 | 0.8174 | +0.002 |

### 10.1 Interpretation

**The shape of the curve matches; the saturation point arrives earlier.**
Adding 20,000 further pairs beyond 4,000 gains +0.001 in total — nothing.

The plateau (0.82) sits below the reference's 0.97, which is expected: different
model pair, different dimension (384 vs 768), different corpus. What matters is
**where it flattens**, and that is earlier than the document assumes (16,000).

### 10.2 Design implications

- A sensible default for `rebasis fit --pairs` is around **4,000**, consistent
  with the projected "0.97+ at 5,000 pairs".
- This strengthens the tool's value proposition: **for a 500,000-chunk vault,
  embedding 4,000 chunks with the new model suffices** — below even the projected
  10,000 chunks.
- Measured on a single model pair and a single corpus; M1 should widen it.

---

## 11. Finding C — two adapters for asymmetric profiles

> The design: *"The most likely source of silent errors. Fit an adapter on document
> pairs, apply it to query vectors, and you get a **train/serve mismatch**. For
> asymmetric profiles use **two adapters**: `g_doc` and `g_query`."*

All three model pairs involve an asymmetric side, giving 72 comparisons:
`g_query` (fitted on query-prefixed pairs) versus `g_doc` (fitted on document
pairs), both applied to query vectors, compared on T0 ARR.

| | value |
|---|---|
| mean cost (two adapters − one adapter) | **−0.0025** |
| median | −0.0020 |
| largest positive (two adapters better) | +0.0975 |
| largest negative (one adapter better) | −0.1001 |
| fraction where two adapters win | 43% |

### 11.1 The pattern — mean zero, but the effect is real

Reading "no difference" off the mean would be wrong. The effect **changes sign**
with the model pair:

- **When the old model is symmetric** (`MiniLM→e5`): two adapters win, +0.04…+0.10
- **When the old model is asymmetric** (`e5→bge`): one adapter wins, −0.03…−0.10
- **`MiniLM→bge`**: close to zero

A plausible explanation: the target space is always the old model's **document**
space, because those are the vectors sitting in the index. When the old model is
symmetric, `g_query`'s target is that same document space — the right target.
When it is asymmetric, `g_query` maps into the query space while the search
happens in the document space, introducing an extra mismatch.

### 11.2 Design implications

- The "two adapters" rule should **not be applied unconditionally.** The
  measurement shows which strategy wins depends on the model pair.
- The right mechanism is again `auto`: measure both strategies on the held-out
  set and keep the better one. No additional infrastructure is required.
- The core warning — **that prefixes must be applied correctly** — stands.
  Prefixes were applied per profile in every configuration measured; the
  `identity` baseline sitting at 0.27 (below) shows the prefix handling really is
  working.
- `EncodingProfile.fingerprint()` is part of the cache key, so a prefix change
  automatically misses the cache. Without that, the prefix bug could be
  reintroduced through a stale cache and become undiagnosable.

---

## 12. Finding E — the option missing from the decision rule

> The decision rule defines four bands: bridging is enough / bridge and migrate /
> caution / full reindex. All four **assume the upgrade is going to happen.**

Baselines from the measurement:

| Baseline | mean | range |
|---|---|---|
| Oracle (full reindex) recall@10 | 0.4943 | [0.1596, 0.8515] |
| **"Do not upgrade at all" ARR** (old model + old index) | **0.9833** | **[0.7611, 1.2378]** |
| Unadapted ARR (raw new vector into old index) | 0.2741 | [0.0833, 0.4845] |

### 12.1 Interpretation

On these corpora, **staying on the old model delivers on average 98% of the new
model** — and the upper bound of 1.24 means that on some corpora the old model is
actually **better** than the new one.

This is specific to these model pairs and these corpora; it is not a general
claim that upgrading embedding models is pointless. But it points at a clear
product gap: the first thing a user should learn from `probe` is **whether the
new model is actually better on their corpus.** If it is not, neither bridging
nor reindexing is worth doing.

The information is already computed — the T1 tier measures oracle and old-model
recall side by side. It only needs to reach the report and the decision rule.

Note that without a real query log (in T0) this comparison cannot be made at all,
because "which model is better" is a question about task success, and T0's ground
truth *is* the new model — under T0 the new model is perfect by definition. That
is the second, undocumented reason why a real query log is "always preferred where
available".

### 12.2 Design implications

- A candidate **fifth output** for the decision rule: *"the new model is not
  better on this corpus; no upgrade needed."* It belongs before the ARR bands in the decision
  tree.
- A metric to add: `upgrade_gain` = oracle recall / old-model recall.
  At or below 1.0 it renders every other metric moot.
- Measurable only at the T1 tier; on T0-only runs the report must say explicitly
  that it could not be measured.

---

## 13. Infrastructure findings

Outside the code but worth recording for reproducibility. All of these actually
happened during these runs.

### 13.1 Two traps in the NVIDIA driver installation

The host bootstrap was hardened against both.

1. **`ubuntu-drivers install --gpgpu` does not install `nvidia-smi`.** It
   installs only `nvidia-headless-no-dkms-*`; `nvidia-utils-*`, which provides
   the `nvidia-smi` binary, does not come along. The result is a loaded kernel
   module, a working GPU, and `nvidia-smi: command not found` — a misleading
   symptom that reads as "there is no GPU".

2. **The kernel module is built for the newest kernel, not the running one.**
   The `linux-modules-nvidia-*-aws` metapackage tracks the current
   `linux-image-aws`. On this machine the running kernel was `6.17.0-1017-aws`
   while the module was installed for `7.0.0-1011-aws`, pulling that kernel in
   alongside it. It worked after reboot because GRUB picked the newer kernel —
   but that was **luck**; a different GRUB choice would have hidden the GPU.
   Bootstrap now installs the module for the running kernel explicitly as well.

### 13.2 `pipefail` + `head` → a SIGPIPE false negative

With `set -o pipefail` active, `nvidia-smi | head -12` sends SIGPIPE to
`nvidia-smi` when `head` closes the pipe early; the pipeline is reported as
failed and the `||` branch fires. The script printed "driver not active yet"
while the GPU was working perfectly. Fixed by replacing `head` with
`--query-gpu ... --format=csv,noheader`, which returns a single line.

### 13.3 `idle_stop.sh` shut down a running job

**During these runs the server was shut down in the middle of a 12-configuration
experiment.** The root cause was two bugs in the idle-stop script's busy
detection:

1. `pgrep -x '[p]ython3'` — the `[p]ython3` bracket trick belongs to `ps | grep`;
   `pgrep` never lists itself. Combined with `-x` (exact name match) it searches
   for a process literally named `"[p]ython3"`, which never matches. On top of
   that, a venv process appears as `python`, not `python3`.
2. `who` — non-interactive `ssh host 'cmd'` calls create **no utmp entry**. On a
   host driven by automation, `who` is always empty.

The fix has three layers:
- busy detection now matches on the **command line** rather than the process
  name (`pgrep -f`), and additionally checks `sshd` children, tmux and
  `nvidia-smi --query-compute-apps`
- a `/var/tmp/rebasis-keep-alive` file disables shutdown entirely
- the background-run subcommand takes that lock **automatically** before a long
  run and releases it afterwards

The design rationale is recorded explicitly: *leaving a GPU host up by mistake
costs a few dollars an hour; killing a running job costs hours of work. The
asymmetry is clear and the default should follow it.*

Following this incident the auto-shutdown cron is **not installed by default**
(it is opt-in via `REBASIS_INSTALL_IDLE_STOP=1`), and the remote wrapper
deliberately has no `stop` command. Idle-stop had been specified as "not
optional, the default"; that is defensible in principle, but the script should
not have been made the default before it was tested.

### 13.4 EBS growth does not propagate automatically

When the root disk was grown from 50 to 150 GB in the console, neither the
partition nor the filesystem followed; `growpart` plus `resize2fs` were required.
Added to bootstrap. Before: `df` showed 48 G. After: 145 G.

### 13.5 The public DNS changes when the instance stops

Without an Elastic IP, a stop/start changed the DNS (`ec2-<old-ip>...` →
`ec2-<new-ip>...`) and `~/.ssh/config` had to be edited by hand. The standard
remedy is an Elastic IP.

Solved in code instead: the remote wrapper's `up` subcommand starts the instance
and `dns` resolves the current DNS from AWS and writes it into `~/.ssh/config`.
The instance id lives in `.env.local` (gitignored). This gives the same result
without a monthly Elastic IP charge.

### 13.6 Instance store is wiped on stop

`/mnt/scratch` (419 GB NVMe) came back empty after a stop/start — the documented
behaviour, confirmed. Everything on EBS survived: the HF model cache
(357 MB), `ir_datasets` (403 MB), the venv and the repo. The decision to keep
model and data caches on EBS is correct; on instance store they would be
re-downloaded after every stop.

The fstab entry uses `nofail` — without it, a boot after the device disappears
hangs waiting for it.

---

## 14. Proposed corrections to the design document

Measurement-backed, gathered into one table. **These are proposals; the decisions
are taken before M1.**

| Area | Assumption | Measurement | Proposal |
|---|---|---|---|
| **Preprocessing** | ℓ2 normalisation only | Centering raises ARR by +0.26 (mean) | Add a mean-removal step before fitting |
| **Adapter table** | The `procrustes` row is uncentred | Centred OP matches MLP quality at half the memory and 2.7× the speed | Make the centred variant first-class |
| **Adapter table** | Residual MLP memory ~1.6 MB | Measured 3.76 MB (d=768) | Count the linear `Wx` term |
| **Adapter table** | LA (r=64) expected 0.97–0.98 | Measured 0.4584 (d=384) | Scale rank with dimension; a fixed 64 is too aggressive at d=384 |
| **Adapter table** | LA latency ~5 µs vs OP ~3 µs | Confirmed, and the *reason* is now known | Low-rank halves the FLOPs but **doubles the numpy calls**; on a cloud vCPU the two cancel exactly. Its benefit is memory, not latency |
| **Asymmetric models** | Two adapters unconditionally | Mean effect −0.003; sign varies by model pair | Let `auto` measure both strategies and choose |
| **T0 ground truth** | GT = full top-k kNN | A sparsity-matched GT cuts error 5.5× | Relax the GT definition |
| **Metrics** | `score_shift` measured raw | 100% exceed the threshold before calibration | Measure it **after** calibration |
| **Metrics** | No `upgrade_gain` metric | "Do nothing" averages ARR 0.983 | Add the metric and a report line |
| **Decision rule** | 4 bands, all assuming an upgrade | The new model is not always better | Add a fifth output: "no upgrade needed" |
| **Decision rule** | `score_shift > 0.1` warning | Fires 100% of the time | Move it after calibration |
| **Calibration** | Isotonic calibration proposed | KS 0.924→0.094, ranking preserved 100% | **Confirmed.** Document it as required, not optional |
| **CSLS** | "Could raise ARR for free" | +0.103 weak, **−0.045** strong | Make it conditional; let `auto` choose |
| **Performance budgets** | Hot path under 15 µs (d=768) | Measured 23.9–91.1 µs on a cloud vCPU | **Closed by [ADR 11](adr/0011-the-hot-path-budget-is-per-dimension.md):** the 768×768 matvec costs 15.8 µs by itself, so the budget is per dimension |
| **Performance budgets** | OP fit < 20 s, LA < 90 s, MLP < 3 min | Measured 0.15 s / 0.15 s / 2.5 s | 30–600× too loose; useless as CI gates |
| **Performance budgets** | Ground truth kNN < 30 s | Measured 0.855 s (CPU) | Same problem |
| **GPU policy** | kNN "borderline", threshold N > 50k | 22× at 2,000 documents including transfer | "Use the GPU when available" |
| **Borderline band** | ±0.005 | Measurement uncertainty ±0.024 (T0), ±0.042 (T1) | Widen to at least ±0.025 |
| **Device projections** | CUDA `migrate` 5-10× CPU | Measured embedding 25–40× | Update |
| **Sample size** | Saturation at 16,000 pairs | Saturates at 4,000 | Default `--pairs` to 4,000 |
| **Idle stop** | Installed by default | Its first version killed a running job | Keep-alive lock added; do not default a script before testing it |
| **Open question 14** | GPU threshold for kNN is open | There is no threshold | The question can be closed |

---

## 15. Limits — what these measurements do **not** establish

Stated explicitly so the numbers are not read more broadly than they support.

1. **One dimension.** All three model pairs are 384→384. The dimension-change
   path — in particular the claim about bypassing Chroma's dimension lock — was
   never exercised on real models.
2. **Small models.** All are ~33M parameters at 384 dimensions. Behaviour may
   differ for `bge-m3` (1024) or 768-dimensional families, especially the
   centering gain and LA's rank selection.
3. **One language, one domain.** All four corpora are English scientific or
   argumentative text. The Turkish-language angle was not measured at all.
4. **Sparse qrels.** Three of the four corpora have ~1 relevant document per
   query. Part of T1's "looseness" comes from that; on a densely judged corpus
   (e.g. TREC) the T0/T1 gap could look different.
5. **The sample curve used one pair.** The saturation point (4,000) was
   measured on a single model pair and a single corpus.
6. **MLP training stops early and aggressively.** `patience=8`, initialised from
   the linear solution. The MLP failing to clearly beat the linear solution may
   be an artefact of that setup; longer training could differ.
7. **The `migrate` path was never measured.** M0 is read-only; the job engine,
   checkpointing, shadow copies and rollback belong to M3.
8. **Determinism was not measured.** The claims about it (same seed → same
   result, cross-device equivalence) were not exercised in M0. The device-parity
   test is M2's job.
9. **One run each.** Every configuration ran with a single seed (0). The bootstrap
   intervals capture query-sampling uncertainty, not fit/split seed uncertainty.

---

## 16. How the measurements were taken

Every run went to the GPU host through the maintainer's remote wrapper, which is
local to that machine. The spike code it ran is in the repository, under
`spikes/`:

- `m0 synthetic` — synthetic validation, no model download, seconds.
- `m0-bg pairs --device auto --dataset "beir/scifact/test,beir/nfcorpus/test,beir/scidocs,beir/arguana"`
  — the main experiment, 4 corpora × 3 model pairs, about a minute with a warm
  embedding cache. Progress and the JSON come back over the same wrapper.
- `m0 latency --latency-dims 384,768` — hot-path latency. The budget is defined
  for d=768.
- `m0 curve --dataset beir/scidocs --curve-sizes "250,500,1000,2000,4000,8000,16000,24000"`
  — the sample-size curve.
- `spikes/device_compare.py` — the device breakdown, CPU against CUDA on one host.
- `spikes/knn_threshold.py` — the kNN GPU threshold.
- `m0 pairs --device cpu --no-mlp` — verifies the torch-free path.

Long runs take the keep-alive lock so the idle-stop cron cannot shut the host
down mid-experiment; the background subcommand takes it automatically.

Raw output lands in `reports/m0-*.json`, which is gitignored and stays on the
host that produced it. Every report carries its own hardware
fingerprint (the `host` field), its arguments and all intermediate timings, so
the conditions a number was produced under are readable from the file itself.

---

## 17. Open questions for M1

What M0 could not answer, to be decided in M1:

1. **On what basis are the decision thresholds retuned?** T0 is unbiased but carries
   ±0.095 error. Widen the bands, calibrate T0, or have the tool say "I cannot
   make this distinction"? All three are defensible; the data does not pick one.
2. **Should centering be the default, or a separate adapter type?** It did no
   harm in 12 of 12 cases (worst −0.018, within noise). Is there any remaining
   reason to keep plain OP?
3. **How many variants should `auto` try?** Adding centering, CSLS and the
   two-adapter strategy multiplies the combinations. There is to be no premature
   parallelisation, but `auto` is the largest item in the budget (open question
   11).
4. **How is `upgrade_gain` estimated at T0?** Without real queries, "is the new
   model better" cannot be measured. If it cannot be measured, how should the
   report say so?
5. **How should LA's rank be chosen?** A fixed 64 fails at d=384. A proportional
   rule such as `d/4`, or selection on the held-out set?
