# Token selection (Mixing Laws Dataset 10B × OLMo-2 370M)

**Question.** Under a matched one-epoch Mixing Laws Dataset budget, can selecting a subset of tokens per sequence beat full-token CE on macro task-loss?

**Answer.** No. Every selection arm finished **worse** than the full-CE control. RHO-1 (excess loss against a frozen Instruct reference) came closest, 0.0146 bpb behind full CE (95% CI [0.0061, 0.0260], two-sided p < 0.0001), and it is indistinguishable from a random-60% keep-rate control (+0.0012 bpb, 95% CI [−0.0073, 0.0097], two-sided p = 0.784). BLADE, Perplexity, Attention and REL-EMA are all significantly worse than both controls.

---

## Setup


| Knob                         | Value                                                                                           |
| ---------------------------- | ----------------------------------------------------------------------------------------------- |
| Architecture                 | OLMo-2 370M (full attention): d=1024, 16L/16H, FFN 4096, RoPE \theta=5\times10^5, vocab 100,352 |
| Train corpus                 | `pretrain/regmix-10b` **v1** — realized **10,004,807,041** tokens, flat shuffle; identical dataset for all eight training runs |
| Global batch / seq / LR      | 4,194,304 / 2048 / 4\times10^{-4} cosine (warmup 24, \alpha_f=0.1)                              |
| Steps                        | 2360 = 9,898,557,440 tokens for every arm — 98.94% of one epoch (10,004,807,041 total), no wrap |
| FLOPs / arm                  | **analytic**, 26.03-56.11\times10^{18} depending on arm - see [Cost and FLOPs](#cost-and-flops). The logged W&B throughput counter (2.63\times10^{19} for every arm) **excludes the scoring forward passes** and so understates every selection arm. |
| Keep rate (where applicable) | top / middle **60%** of valid target tokens per sequence, including BLADE (its selection and reference K-update masks are per-row too, the same unit as every other arm); **realized 0.599609** (1228 kept of 2047 valid target positions, logged over all 2048 positions) — see [Realized keep rate](#realized-keep-rate) |
| Primary metric               | Macro mean CE bits-per-byte over the **20 (task, split) labels** of the OLMo ladder, covering 10 OLMES benchmarks. MMLU supplies 8 of the 20 (**40% of the macro weight**), and 7 of the 20 are **test** splits, so "validation macro bpb" is a misnomer. |


Arm-by-arm contract table: [arms/README.md](arms/README.md). Contamination audit (paper
Section 2.3): [contamination/](contamination/). Code that built the training corpus and
the reference corpus from Hugging Face sources (Appendix A): [`../../datasets/`](../../datasets/).
`python fit_and_plot.py` reproduces Table 1, every paired difference below, and Figures 1
and 2 ([`figures/`](figures/)) from the committed curve cache; `python flops.py` reproduces
Table 2's FLOPs.

**Data availability.** We do not redistribute these datasets or our derived subsets. Each
corpus can be rebuilt byte for byte from its public Hugging Face sources with the document
manifests in [`../../datasets/manifests/`](../../datasets/manifests/): `regmix-10b-v1` (the
10B training corpus) and `refhq-new-v1` (the Instruct reference). (`olmo-127b-v1` there is
the Domain weighting paper's 127B reservoir, not used here.)

Shared contract: same architecture, batch, LR, and step budget. The manipulation is **which tokens receive gradient** on each step (full CE vs a scored subset). Sequences still come from the same shuffled corpus; selection is per-token inside the batch.

### Frozen reference training data

RHO-1 and Perplexity score tokens against one **frozen** same-architecture CE model, the
`instruct-reference` arm's final (step 940) checkpoint.

**Reference — instruct mix (~3.9B).** Plain CE on a one-pass **English-filtered instruct** corpus (no upsampling; realized size is whatever the filtered pass yields, about 3.9B tokens). Sources are instruction / chat / math / code SFT-style collections with Dolci's and Tulu-3's safety subsets and Dolci's tool-use and precise-IF subsets dropped, then the Dolma English filter on all sources, keeping document score ≥ 0.5. Token counts are the paper's Appendix A reference-dataset table:


| Source family  | Tokens | Kept (summary)                                                                                                                                                                   |
| -------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SmolTalk       | 1.607B | the aggregate `all` config plus the 11 individual configs other than `apigen-80k` and `smol-constraints`, so those two subsets appear once and the other 11 twice |
| Dolci          | 0.901B | drop `domain` Safety or Precise IF; drop rows whose `source_dataset` + `id` contains coconot, aya, wildguard, wildjailbreak or "tool use"; drop rows with non-empty `function_calls` / `functions` |
| Tulu-3         | 0.537B | drop rows whose `source` + `id` contains wildguardmix, wildjailbreak, coconot or aya; the Persona instruction-following subset is retained (its exclusion needle matched nothing) |
| OpenHermes-2.5 | 0.342B | drop only rows whose `language` is set and not en / eng / english; null or missing language is kept |
| Hermes-3       | 0.342B | all                                                                                                                                                                              |
| Tulu-v2        | 0.208B | all                                                                                                                                                                              |


Domains labeled general / math / code / science / chat from metadata. dolma2 tokenizer. Training FLOPs \approx 1.04\times10^{19} (940 steps). Used as the frozen scorer for RHO-1 and Perplexity; BLADE's dynamic reference also trains on this corpus.

### What each method manipulates

**RHO-1 (excess loss).** For each target token, score L_{\mathrm{curr}} - L_{\mathrm{ref}} under the live student vs the **frozen** Instruct reference. Keep the top 60% (largest excess loss — tokens where the student is worst relative to the reference). Selection is active from step 0. Reference losses come from one offline pass (see Perplexity). Method from [RHO-1](https://arxiv.org/abs/2404.07965).

**Attention top-k.** No external reference. Score each token by the **causal attention it receives** on the last transformer layer, aligned to the target token whose loss it gates, then z-score it against the mean and standard deviation of tokens at the same position from the model's own immediately preceding training step (trained attention is recency-biased, so a uniform-attention normalizer would leave the score position-confounded). Keep the top 60%. Active from step 0. Manipulation: prefer tokens that the model currently "attends to" as context for later predictions. Method from [ssToken](https://arxiv.org/abs/2510.18250) (attention-based score).

**Perplexity (middle-PPL).** Score tokens by token CE under the frozen Instruct reference's step-940 checkpoint, the same checkpoint RHO-1 scores against. Scored **offline**: the whole corpus is scored once, on a single GPU, and training looks up each instance's row by its global index instead of keeping a second copy of the reference resident (see `token_selection_370m/reference_scores.py`); this is exact, not an approximation, since selection depends only on the reference-loss row and instance content never depends on world size or microbatch. Keep the **middle 60%** (20th to 80th percentile) of the per-sequence score distribution (drop both easiest and hardest tails). Manipulation: train on "medium difficulty" tokens under that frozen scorer, excluding extremes. Method from [Marion et al., Investigating Data Pruning for Pretraining LLMs at Scale](https://arxiv.org/abs/2309.04564).

**BLADE.** Bi-level setup with a **proxy** (trained student) and a **dynamic reference** that is periodically reset from the proxy, from step 0. At sync steps 0, 400, 800, 1200, 1600, 2000 (\tau=400): score the outgoing reference's K=75 upcoming update batches against the proxy before overwriting it, copy proxy → reference, then run the K reference updates (each sums the unmasked mean CE of one Instruct-reference batch and the selection-weighted mean CE of one training-corpus batch, AdamW at the proxy's own scheduled LR, floored at its post-warmup value so the step-0 sync doesn't train at LR 0). Keep proxy tokens with largest L_{\mathrm{proxy}}-L_{\mathrm{ref}} at keep-rate \gamma=0.6, per row like every other arm, from step 0. Manipulation: select tokens where the fast proxy outruns a lagged copy of itself. Method from [BLADE](https://arxiv.org/abs/2606.18650).

**REL-EMA (exponential).** Online relative loss vs an **EMA of the student** (bias-corrected from zero; no external seed). EMA rate \alpha(t)=1-e^{-t/300}; the resulting reference lags the student by about 25 steps at step 1000 and about 570 steps at the end of training. Score \mathrm{REL}=L_{\mathrm{curr}}-L_{\mathrm{hist}}; keep top 60%. Active from step 0. Manipulation: prefer tokens where the live model is worse than its own exponential history. Method from [ssToken](https://arxiv.org/abs/2510.18250) (retrospective excess loss / REL).

### Runs reported

Every run is in W&B project `eduLLM/token-selection`, trained under the vendored code in
[`olmo_core_token_selection/`](olmo_core_token_selection/) (fork revision `64c28145`; the
reference recorded `765ae838`, see [PROVENANCE](olmo_core_token_selection/PROVENANCE.md)).

| Arm | W&B run | Init / data seed |
| --- | --- | --- |
| Full-loss control | `2ba31b3f5bb006897451d9f68fbcf93b` | 6198 / 42 |
| Random control, seed 42 | `928cc12b8ed5e9652b2a9a9f5cbbc6bb` | 6198 / 42 |
| Random control, seed 69 | `6b741657880c37f422020c594a3a2bfc` | 12345 / 69 |
| RHO-1 | `bad4d901c4579b8c0319ebcec43ba765` | 6198 / 42 |
| BLADE | `59e6d62ee97123b755812e9e2d822267` | 6198 / 42 |
| Perplexity | `a6a187590e4cbf16c7d279cb3315d009` | 6198 / 42 |
| Attention | `194a3c2a720db4fa11942a246f9a51b3` | 6198 / 42 |
| REL-EMA | `1de7041a160ce9a1d2d5f2be9fc1a940` | 6198 / 42 |
| Instruct reference (940 steps, not an arm) | `f0460e5c099079f314cbed0e9c50fb62` | 6198 / 42 |

Every arm starts from the same initialization (step-0 macro **4.4662** bpb) except the
second random-control run, whose different init seed is deliberate (step 0: **4.4610**).
The random-60% selection control (keep-rate 60% chosen uniformly at random per row, no
scoring signal) was run at **two seeds**; the data seed sets both the data order and the
random token selection, drawn per instance so the two runs never share a mask. It is
reported as a single two-run fit; see Results below.

---

## Evaluation and uncertainty

### Fitting and bootstrap procedure

Every fitted-final number and every CI in this README comes from the following
procedure (`fit_and_plot.py`).

1. **Model.** `y = a + b * step^(-alpha)` fitted to the macro task-loss curve.
2. **Fit window.** Only steps **>= 1000** are used. Not because of warmup, which is 24
   steps, but because the early curve is far noisier: the two random-control seeds differ
   by 0.0251 bpb on average over steps 125-875 against 0.0106 bpb inside the window.
   Every arm is evaluated every 125 steps on `{0, 125, ..., 2125, 2250, 2360}`; step 2250
   is kept because it is 110 steps from the final step. So each run has 12 fit points
   (1000, 1125, ..., 2250, 2360). `fit_and_plot.py` warns for any arm whose fit window is
   not exactly this grid.
3. **alpha search.** `alpha` is chosen by grid search over
   `np.linspace(0.05, 6.0, 1192)`, with `a` and `b` solved in closed form by least
   squares at each `alpha`. Six of the seven reported arms have an interior optimum
   (from 0.999 for the full-loss control to 3.027 for Attention). **Perplexity's fitted
   `alpha` sits on the grid's lower bound (0.05)**, as do about half of its bootstrap
   draws, so its exponent is clipped rather than data-determined and its interval is
   probably too narrow.
4. **Bootstrap.** 10,000 i.i.d. residual bootstrap draws: residuals from the point fit
   are rescaled by `sqrt(n/(n-p))` with `p=3` (1.155 at `n=12`, 1.069 at `n=24`),
   then resampled with replacement and added back to the fitted curve. OLS residuals
   are shrunk relative to the true errors by that factor on average, so resampling
   them raw understates the spread. Without the rescaling the mean interval width over
   the seven arms would be 0.01422 bpb rather than 0.01620.
5. **alpha re-estimated on every draw.** Each bootstrap replicate re-runs the full
   `alpha` grid search rather than holding `alpha` at its point estimate. This was
   chosen deliberately **because it yields the wider intervals** (mean width 0.01620 vs
   0.01213 bpb alpha-fixed) -- it propagates curvature uncertainty instead of
   conditioning it away.
6. **Interval.** 95% CI = the 2.5 / 97.5 percentiles of the bootstrap distribution of
   the fitted final.
7. **Fitted final** is evaluated at **each arm's own final logged step** (2360 for all
   seven arms).
8. **RNG seed 0**, so the numbers are reproducible.

**The random control is fitted across two runs.** It was run at seeds 42 and 69. Rather
than picking one, it is reported as a single **two-run fit**: one power law fitted by
least squares to the union of both runs' fit-window points (12 each, 24 total), with
`alpha` profiled on the same grid and the same 10,000-draw residual bootstrap taken over
the **pooled** residuals. Because the two runs are offset from one another, that offset
enters the residual pool, so this arm's interval carries seed-to-seed spread as well as
within-run noise. Every comparison involving the random control uses that fit.

**What these intervals are not.** For the other six arms they are **curve-fit intervals
for a SINGLE run**, quantifying only how well a power law pins down the endpoint of one
observed trajectory, with **no seed-to-seed variance**. The one direct measurement we
have of that variance is the random control's two seeds: seed 69 finishes **0.0076 bpb**
above seed 42 (95% CI [−0.0022, 0.0169], two-sided p = 0.1214), not separated from
within-run noise. Their per-checkpoint difference varies from −0.008 to +0.030 bpb inside
the fit window and shrinks over training. Treat any single-seed gap of that order as
unresolved.

**Pairwise significance.** Paired bootstrap on the fitted finals: both arms' 10k draws
are generated with `default_rng(0)`, so the resample index matrices are identical and the
draws are **paired by index**; `delta_stats` then forms the empirical distribution of
`fitted_final(A) - fitted_final(B)`. The **one-sided** p-value in the tables below is the
bootstrap mass on the side opposite the point estimate. The paper reports **two-sided**
p-values, twice the one-sided value, because the direction of each comparison was not
pre-specified. "< 0.0001" means no draw fell on the other side of zero.

---

## Results

### Fitted final macro task-loss (bpb)

Matched-protocol Table 1. Lower is better.

| Arm                     | Fitted final | Observed | 95% CI           |
| ----------------------- | ------------ | -------- | ---------------- |
| Full-loss control       | **1.6789**   | 1.6793   | [1.6695, 1.6877] |
| Random control seed 42  | 1.6885       | 1.6897   | [1.6825, 1.6931] |
| Random control seed 69  | 1.6961       | 1.6934   | [1.6887, 1.7028] |
| **Random control (two-run fit)** | **1.6923** | 1.6915 | **[1.6851, 1.6991]** |
| RHO-1                   | 1.6935       | 1.6943   | [1.6879, 1.6981] |
| BLADE                   | 1.7487       | 1.7537   | [1.7429, 1.7544] |
| Perplexity              | 1.8492       | 1.8499   | [1.8429, 1.8581] |
| Attention top-k         | 1.9163       | 1.9171   | [1.8939, 1.9328] |
| REL-EMA (exponential)   | 1.9435       | 1.9449   | [1.9408, 1.9463] |

The only overlapping pairs at 95% are full-loss control↔random control and random
control↔RHO-1. No selection arm's interval overlaps the full-loss control's.

### Paired differences

Against the full-loss control:

| Arm | Delta | 95% CI | One-sided p |
| --- | --- | --- | --- |
| Random control (two-run) | +0.0134 | [+0.0022, +0.0249] | 0.0082 |
| RHO-1 | +0.0146 | [+0.0061, +0.0260] | < 0.0001 |
| BLADE | +0.0698 | [+0.0592, +0.0822] | < 0.0001 |
| Perplexity | +0.1716 | [+0.1638, +0.1792] | < 0.0001 |
| Attention top-k | +0.2368 | [+0.2213, +0.2505] | < 0.0001 |
| REL-EMA | +0.2647 | [+0.2559, +0.2738] | < 0.0001 |

Against the two-run random control:

| Arm | Delta | 95% CI | One-sided p |
| --- | --- | --- | --- |
| Full-loss control | −0.0134 | [−0.0249, −0.0022] | 0.0082 |
| **RHO-1** | **+0.0012** | **[−0.0073, +0.0097]** | **0.3920** |
| BLADE | +0.0564 | [+0.0475, +0.0656] | < 0.0001 |
| Perplexity | +0.1582 | [+0.1479, +0.1684] | < 0.0001 |
| Attention top-k | +0.2234 | [+0.2007, +0.2420] | < 0.0001 |
| REL-EMA | +0.2513 | [+0.2440, +0.2589] | < 0.0001 |

**RHO-1 does not beat random token masking** (two-sided p = 0.784). It sits between the
two random-control runs: 0.0026 bpb better than seed 69 (95% CI [−0.0037, 0.0092],
two-sided p = 0.4468) and 0.0050 bpb worse than seed 42 (95% CI [−0.0035, 0.0136],
two-sided p = 0.2532). Random masking itself costs 0.0134 bpb against full CE
(two-sided p = 0.0164), the expected price of dropping 40% of the gradient signal.

### Realized keep rate

Every selection arm (RHO-1, BLADE, Perplexity, Attention, REL-EMA and both random-control
runs) realized a keep rate of exactly **0.599609** -- **1228 of 2047** target positions
per 2048-token sequence -- **at every logged step**. This is the W&B metric
`train/selected token fraction` (the full-loss control logs 2047/2048 = 0.999512). The
value is `round(2047 * 0.6) / 2048`: the nominal 0.6 is applied to the 2047 valid
next-token targets in a 2048-token sequence and rounded to an integer count per row
(1228), while the logged metric divides by all 2048 positions. Against the 2047 valid
targets the realized rate is 1228/2047 = 0.599902. Because the count is fixed per row
rather than thresholded on the score, the keep rate carries no information about the
scorer -- the arms differ only in *which* 1228 tokens they keep.

### Cost and FLOPs

The **logged throughput FLOPs counter** is incomplete, because it **excludes the
scoring forward passes** and so reports the same number for a full-CE run as for a run
that additionally evaluates a reference model on every token. We therefore report
analytic FLOPs throughout.

**Model.** Forward cost per token = `2N + 4*L*T*d`, with `N = 371,195,904` matmul
parameters (**including the untied output head**), `L = 16` layers, `T = 2048` sequence
length, `d = 1024` model width. The `4*L*T*d` term is the attention score/value
matmuls, which scale with sequence length and are not captured by a parameter count.
Forward+backward = **3x** forward. For the full-loss control this gives 2.603e19 FLOPs,
within 1% of the logged W&B counter (2.63e19).

| Arm | In-run FLOPs (x10^18) | Total, with reference training and offline scoring (x10^18) | Total relative to control | GPU-hours |
| --- | --- | --- | --- | --- |
| **Full-loss control** | **26.03** | **26.03** | **1.00x** | **67.3** |
| Random control | 26.03 | 26.03 | 1.00x | 67.3 |
| RHO-1 | 26.03 | 45.17 | 1.73x | 126.8 |
| BLADE | 46.19 | 56.11 | 2.16x | 177.6 |
| Perplexity | 26.03 | 45.17 | 1.73x | 127.5 |
| Attention top-k | 26.11 | 26.11 | 1.00x | 73.9 |
| REL-EMA (exponential) | 34.71 | 34.71 | 1.33x | 85.5 |

`flops.py` reproduces every FLOPs number here (`python flops.py`). It counts in units of
one no-grad forward over one 4,194,304-token global batch. GPU-hours are L40S hours from
the Slurm accounting of every job that contributed to the reported run: GPUs × wall-clock,
excluding the work after the last checkpoint a later job resumed from. RHO-1 and
Perplexity also include the Instruct reference's training and the offline scoring pass
(both 1 GPU), each charged in full although the two arms share them. GPU-hours track the
analytic FLOPs closely (2.46 to 2.83 GPU-hours per 10^18 FLOPs, against the control's
2.59) except for BLADE at 3.17, 22% above the control: holding two models and two
optimizers forced it onto an 8,192-token microbatch, and its syncs run outside the
normal training loop.

Reading the table: Attention top-k is nearly free (+0.3%) -- its only extra cost is a
forward hook on the last transformer block that captures the Q/K inputs, recomputes Q
and K, and forms QK^T for that one layer. REL-EMA pays a live extra forward pass on its
EMA copy every step (1.33x). RHO-1 and Perplexity do not score during training: both
read per-token reference losses from one table computed once, offline, by a single
forward pass over the whole RegMix corpus (4,885,156 instances, 8.77e18 FLOPs) with the
frozen Instruct step-940 reference, which itself cost 10.37e18 FLOPs to train (940 steps
of forward+backward). The two arms share that pass and that reference but are each
charged the full cost here (1.73x total, 1.00x in-run). BLADE scores every training
batch against both the proxy and its dynamic reference (2 forwards per step for all 2360
steps, 17.35e18), scores each reference-update batch at the five later syncs (2.76e18)
and runs a parity check at every sync (0.04e18), for **46.19** in-run; its 6 syncs x 75
reference updates then add the itemized cost below, for a total of **56.11**.
**The two most expensive arms are BLADE (2.16x) and RHO-1/Perplexity (1.73x), against
Attention's 1.00x.**

**BLADE's K-update overhead, itemized.** 6 syncs (steps 0, 400, ..., 2000) x 75 K-steps
x 2 batches per step (one training-corpus batch and one Instruct-corpus batch) = **900
full batches** of 4,194,304 tokens = **3,774,873,600 tokens** of forward+backward, on top
of the 2360 training steps. At 2.6298e9 FLOPs/token that is 9.93e18, which accounts for
the **56.11 vs 46.19** total-versus-in-run difference. The in-run overhead (46.19 vs
26.03) is the per-step and per-sync scoring passes: 2 x 2360 forwards per step, plus the
pre-scoring and parity forwards above (20.16e18).

### Takeaways

1. **Full-token CE wins.** Dropping 40% of tokens never improved macro task-loss at this
   budget. RHO-1, the closest selection arm, still trails full CE by 0.0146 bpb
   (two-sided p < 0.0001).
2. **Ranking.** Full-loss control < Random control ≈ RHO-1 < BLADE << Perplexity <
   Attention < REL-EMA. Excess loss against a frozen Instruct reference is the least
   damaging scorer; BLADE is clearly worse, and Perplexity, Attention and REL-EMA form a
   far-worse tier, 0.17 to 0.26 bpb above full CE.
3. **RHO-1 is indistinguishable from random masking** (+0.0012 bpb, two-sided
   p = 0.784), and it falls between the two random-control seeds. The scoring rule adds
   nothing measurable over a random mask of the same rate.
4. **BLADE's extra machinery did not pay off** -- worse than both RHO-1 and random
   masking despite 3,774,873,600 extra tokens of K-update forward+backward (2.16x
   control FLOPs and 2.6x its GPU-hours).
5. **Cost.** Token selection was the most expensive lever and the weakest scientific
   return: up to **2.16x** the control's analytic FLOPs for a strictly worse result.
6. **Seed variance.** The random control's two seeds differ by 0.0076 bpb in fitted
   final, not separated from within-run noise (p = 0.1214). Only the random control has
   a seed replicate; the other arms are single runs, so their intervals bound within-run
   noise only. The gaps from BLADE, Perplexity, Attention and REL-EMA to either control
   are 7 to 35 times the measured seed difference; the gaps among the full-loss control,
   the random control and RHO-1 are of the same order as it.

---

## Conclusions

Under the P1 Mixing Laws Dataset × 370M one-epoch contract, **token selection is a negative result**: every tested scorer underperforms full CE. The best scorer (RHO-1) is statistically indistinguishable from a random-60% keep-rate control (delta +0.0012, 95% CI [−0.0073, 0.0097], **two-sided p = 0.784**), and it loses to full CE by 0.0146 bpb (two-sided p < 0.0001), so selection does not pay for itself here. Prefer mixture optimization (MixLaw) over token masking for this setup.

Two caveats a reader should carry out of this page. First, only the random control has a
seed replicate; the other arms are single runs whose intervals contain **no seed-to-seed
variance**, and the one seed contrast we can measure is 0.0076 bpb. Second, Perplexity's
fitted exponent sits on the lower bound of the search grid, so its interval is probably
understated, and Attention's interval is the widest by far (0.039 bpb). The direction of
the headline result is robust (the selection arms lose to full CE by 0.015 to 0.26 bpb).

Benchmark contamination for this corpus and the Instruct reference corpus is audited in
[`contamination/`](contamination/).
