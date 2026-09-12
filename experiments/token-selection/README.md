# Token selection (Mixing Laws Dataset 10B × OLMo-2 370M)

**Question.** Under a matched one-epoch Mixing Laws Dataset budget, can selecting a subset of tokens per sequence beat full-token CE on macro task-loss?

**Answer.** No. Every selection arm finished **worse** than the full-CE baseline. Excess-loss ρ-1 was the least harmful, but it isn't significantly better than a random-60% keep-rate control (one-sided bootstrap **p = 0.182**); middle-perplexity and relative-EMA selection collapsed performance.

---

## Setup


| Knob                         | Value                                                                                           |
| ---------------------------- | ----------------------------------------------------------------------------------------------- |
| Architecture                 | OLMo-2 370M (full attention): d=1024, 16L/16H, FFN 4096, RoPE \theta=5\times10^5, vocab 100,352 |
| Train corpus                 | `pretrain/regmix-10b` **v1** — realized **10,004,807,041** tokens, flat shuffle; identical dataset for all seven arms |
| Global batch / seq / LR      | 4,194,304 / 2048 / 4\times10^{-4} cosine (warmup 24, \alpha_f=0.1)                              |
| Steps                        | 2360 = 9,898,557,440 tokens — one epoch, no wrap (the full-loss control ran 2384 = 9,999,220,736, also no wrap) |
| FLOPs / arm                  | **analytic**, 26.03-49.21\times10^{18} depending on arm - see [Cost and FLOPs](#cost-and-flops). The logged W&B throughput counter (2.63\times10^{19} for every arm) **excludes the scoring forward passes** and so understates every selection arm. |
| Keep rate (where applicable) | top / middle **60%** of valid target tokens per sequence; **realized 0.599609** (1228 of 2047 target positions per sequence) — see [Realized keep rate](#realized-keep-rate) |
| Primary metric               | Macro mean CE bits-per-byte over the **20 (task, split) labels** of the OLMo ladder — 10 OLMES benchmarks × val/test. MMLU supplies 8 of the 20 (**40% of the macro weight**), and 7 of the 20 are **test** splits, so "validation macro bpb" is a misnomer. |


Arm-by-arm contract table: [ARMS.md](ARMS.md). Contamination audit (paper Section 4):
[contamination/](contamination/).

Shared contract: same architecture, batch, LR, and step budget. The manipulation is **which tokens receive gradient** on each step (full CE vs a scored subset). Sequences still come from the same shuffled corpus; selection is per-token inside the batch.

### Frozen reference training data

Several arms score tokens against a **frozen** same-architecture CE model. Two reference checkpoints were used; they differ in the corpus they were trained on (not in student architecture).

**Reference A — HQ web mix (~5.5B).** Plain CE on an HQ-filtered ~5.509B-token corpus in Mixing Laws 7-domain proportions (scaled to that budget):


| Domain          | Approx. target tokens | Source character                                  |
| --------------- | --------------------- | ------------------------------------------------- |
| dclm            | ~2.07B                | DataDecide DCLM-Baseline QC (FineWeb-2 7% recipe) |
| arxiv           | ~1.38B                | arXiv                                             |
| starcoder       | ~0.78B                | StarCoder + Dolma code-HQ filter                  |
| pes2o           | ~0.52B                | peS2o                                             |
| open-web-math   | ~0.35B                | OpenWebMath (HQ; pool binds)                      |
| algebraic-stack | ~0.34B                | AlgebraicStack                                    |
| wiki            | ~0.09B                | Wikipedia                                         |


dolma2 tokenizer. Training FLOPs \approx 1.45\times10^{19}. Used as the frozen scorer for Middle-PPL late-average scores (mean of late checkpoints).

**Reference B — instruct mix.** Plain CE on a one-pass **English-filtered instruct** corpus (no upsampling to a fixed 5.5B cap; realized size is whatever the filtered unique pass yields). Sources are instruction / chat / math / code SFT-style collections with safety / tool / IFEval-oriented subsets dropped, then Dolma English document-score ≥ 0.5:


| Source family  | Kept (summary)                                                                                                                                                                   |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tulu-v2        | all                                                                                                                                                                              |
| OpenHermes-2.5 | all (optional non-English metadata drop)                                                                                                                                         |
| Tulu-3         | FLAN, WildChat, math/code/science personas, Numina-TIR, Evol CodeAlpaca, SciRIFF, TableGPT, No Robots, OASST, etc.; drop WildGuard / WildJailbreak / CoCoNot / Aya / IF personas |
| Hermes-3       | all                                                                                                                                                                              |
| SmolTalk       | all configs except API-gen / constraint suites                                                                                                                                   |
| Dolci          | keep non-safety / non-tool / non-IF / non-Aya rows                                                                                                                               |


Domains labeled general / math / code / science / chat from metadata. dolma2 tokenizer. Training FLOPs \approx 1.04\times10^{19}. Used as the frozen scorer for ρ-1.

### What each method manipulates

**ρ-1 (excess loss).** For each target token, score L_{\mathrm{curr}} - L_{\mathrm{ref}} under the live student vs a **frozen** reference model of the same architecture. Keep the top 60% (largest excess loss — tokens where the student is worst relative to the reference). Selection is active from step 0. Method from [RHO-1](https://arxiv.org/abs/2404.07965).

**Attention top-k.** No external reference. Score each token by **causal attention received** on the last transformer layer (mean over heads of attention mass from later positions). Keep the top 60%. Active from step 0. Manipulation: prefer tokens that the model currently “attends to” as context for later predictions. Method from [ssToken](https://arxiv.org/abs/2510.18250) (attention-based score).

**Middle-PPL.** Score tokens by **late-checkpoint average** token CE under **Reference A**. Keep the **middle 60%** of the per-sequence score distribution (drop both easiest and hardest tails). Masks can be precomputed. Manipulation: train on “medium difficulty” tokens under that frozen scorer, excluding extremes. Method from [Marion et al., Investigating Data Pruning for Pretraining LLMs at Scale](https://arxiv.org/abs/2309.04564).

**BLADE.** Bi-level setup with a **proxy** (trained student) and a **dynamic reference** that is periodically reset from the proxy. Steps 0–499: full CE on the proxy only (no selection). At sync steps 500, 875, 1250, 1625, 2000: copy proxy → reference, run K=75 reference updates, then keep proxy tokens with largest L_{\mathrm{proxy}}-L_{\mathrm{ref}} at keep-rate \gamma=0.6 (\tau=375). After the last sync, hold that reference to the end. Manipulation: select tokens where the fast proxy outruns a lagged copy of itself. Method from [BLADE](https://arxiv.org/abs/2606.18650).

**REL-EMA (exponential).** Online relative loss vs an **EMA of the student** (bias-corrected from zero; no external seed). EMA rate \alpha(t)=1-e^{-t/300}. Score \mathrm{REL}=L_{\mathrm{curr}}-L_{\mathrm{hist}}; keep top 60%. Active from step 0. Manipulation: prefer tokens where the live model is worse than its own exponential history. Method from [ssToken](https://arxiv.org/abs/2510.18250) (retrospective excess loss / REL). Corrected 2026-09-05: the score was previously computed as $L_{\mathrm{hist}}-L_{\mathrm{curr}}$ (inverted relative to rho\_excess/blade's convention); re-run with the fixed polarity.

### Arms actually run

> **Compute is reported as analytic FLOPs only.** Wall-clock hours are not comparable
> across these runs: five arms ran on **8xA100-80GB** (rho-1, BLADE, Attention,
> Middle-PPL, full-loss control) while the **Random control and REL-EMA ran on 4xL40S**.
> The logged W&B throughput counter is also incomplete: it **excludes every scoring
> forward pass**, so it reports the same 2.63e19 for a full-CE run as for an arm that
> additionally evaluates a frozen reference on every token. See
> [Cost and FLOPs](#cost-and-flops) for the model and the in-run/with-reference split.

| Arm | Analytic FLOPs (x10^18) | Relative to control |
| --- | --- | --- |
| Random control | 26.03 | 0.99x |
| Attention top-k | 26.11 | 0.99x |
| **Control (full-CE)** | **26.30** | **1.00x** |
| REL-EMA (exponential) | 34.71 | 1.32x |
| rho-1 | 45.08 | 1.71x |
| BLADE | 47.98 | 1.83x |
| Middle-PPL | 49.21 | 1.87x |

The rho-1, BLADE and Middle-PPL totals include pretraining their reference models.
Middle-PPL's figure also covers precomputing its masks with one forward pass over the
whole corpus, which is why it is the most expensive arm despite adding no scoring cost
during training itself.

**BLADE's W&B record does not contain its own curve.** Run `005xjces` has a **7-second
runtime** and logs **no throughput and no keep-rate metrics**, so its loss curve was
backfilled or resumed into that record rather than produced by it. The BLADE numbers
are usable but their provenance is one hop removed from the run ID they are attributed
to.


Control is `eduLLM/token-selection/hh19uatg` (`full-loss-control-regmix10b-v2`), created 2026-09-05, `eval/macro_bpb` final 1.6487. This replaces the earlier control, which was reused from the `mixlaw-1` project. Per this run's own logged config, it is itself cloned from `eduLLM/hpo-ladder` (group `hpo-ladder-batch-ablation`, run `library-4mi`), **not** from the curriculum project. Its config also carries this note verbatim: *"actual nested model.init_seed logged by the source run's config was 0 (init_seed did not propagate into TransformerConfig) rather than 6199 or the standard 6198 used by the other token-selection arms."* So the weight-init-seed confound described in the paper's Section 3 is unchanged by this re-run — only the source project changed, not the underlying seed/step mismatch. Data-loader seed remains 6199 (vs. the other six arms' 42), and step count remains 2384 (vs. 2360). Its eval grid is also ~119 steps rather than the 125-step permanent ladder the selection arms use, and the run is tagged `cloned`.

**A matched rerun is in flight.** `full-loss-control-regmix10b-v3`
is in flight (FarmShare job **1719708**, 4xL40S, **2360** steps, **125**-step ladder,
`method="full"` with `keep_fraction=1.0` routed to the stock train module). Once it
lands, the control is seed-, step- and grid-matched to the selection arms and the
Section 3 confound is resolved. See `ARMS.md` for the full field-by-field comparison.

A random-60% selection control (`random-control-regmix10b-v1`: keep-rate 60% chosen uniformly at random per row, no scoring signal) was run separately on 4×L40S (FarmShare) rather than A100, using the same `pretrain/regmix-10b` v1 corpus; see Results below. Task-loss curves below report the ρ-1 arm.

---

## Evaluation and uncertainty

### Fitting and bootstrap procedure

Every fitted-final number and every CI in this README comes from the following
procedure. It is the matched protocol used for every number in this file.

1. **Model.** `y = a + b * step^(-alpha)` fitted to the macro task-loss curve.
2. **Fit window.** Only steps **>= 1000** are used; earlier points are dominated by the
   LR warmup transient and bias `alpha`.
3. **alpha search.** `alpha` is chosen by grid search over
   `np.linspace(0.05, 6.0, 1192)`, with `a` and `b` solved in closed form by least
   squares at each `alpha`. The grid was **widened from the previous `[0.05, 3.0]`**,
   which **boundary-pinned REL-EMA at exactly `alpha = 3.0`** -- that pin is the whole
   reason REL-EMA's fitted final moves **1.9175 -> 1.9199**. No other arm's fitted
   `alpha` sat at a boundary under either grid.
4. **Bootstrap.** 10,000 i.i.d. residual bootstrap draws: residuals from the point fit
   are resampled with replacement and added back to the fitted curve.
5. **alpha re-estimated on every draw.** Each bootstrap replicate re-runs the full
   `alpha` grid search rather than holding `alpha` at its point estimate. This was
   chosen deliberately **because it yields the wider intervals** -- it propagates
   curvature uncertainty instead of conditioning it away.
6. **Interval.** 95% CI = the 2.5 / 97.5 percentiles of the bootstrap distribution of
   the fitted final.
7. **Fitted final** is evaluated at **each arm's own final logged step** (2384 for the
   full-loss control, 2360 for the other six arms), not at a common step.
8. **RNG seed 0**, so the numbers are reproducible.

**What these intervals are not.** They are **curve-fit intervals for a SINGLE run per
arm**. They quantify only how well a power law pins down the endpoint of one observed
loss trajectory. They contain **no seed-to-seed variance** whatsoever: no arm was run
at more than one seed, so nothing here bounds run-to-run spread. Treat any between-arm
gap smaller than a plausible seed effect as unresolved.

**A quantitative seed floor.** Companion 370M runs in this repo
(`experiments/curriculum`, seeds 42/69, same OLMo2-370M architecture, same macro-bpb
metric, same saturating-power-law fitted-final estimator at step 2384) measured a
**pooled per-run SD of 0.0199 bpb** across two two-seed pairs. Taking that as the
per-run SD, a single-seed difference between two arms carries a standard error of
about `0.0199 * sqrt(2) = 0.028` bpb. Against that floor:

| Gap vs full-CE control | Delta | Multiples of 0.028 | Survives seed floor? |
| ---------------------- | ----- | ------------------ | -------------------- |
| REL-EMA                | +0.2727 | 9.7x | yes |
| Middle-PPL             | +0.2574 | 9.2x | yes |
| BLADE                  | +0.0617 | 2.2x | marginal |
| Attention top-k        | +0.0518 | 1.8x | no |
| Random control         | +0.0384 | 1.4x | no |
| rho-1                  | +0.0351 | 1.3x | no |

So the *ordering* is robust for Middle-PPL and REL-EMA, and the four near-control arms
are separated from the control by less than single-seed noise. That does not rescue
them -- none is *better* than full CE either -- but the honest reading of rho-1,
random, Attention, and BLADE is "no improvement detected," not "worse by this much."

p-values against the full-CE control are omitted because **no selection arm beat
control** -- there is no positive claim to test.

**Pairwise significance (rho-1 vs random-control).** Since rho-1 was the only selection
arm to land below random-control's point estimate, we tested whether that edge is real
with the same machinery extended to two independent arms: fit each arm's own power law,
bootstrap 10k resamples of its fitted final from its own residuals (independent RNG
stream per arm, since the two runs are independent), then form the empirical
distribution of `fitted_final(rho-1) - fitted_final(random-control)` from the two
10k-draw arrays. The reported **one-sided** p-value is the fraction of that difference
distribution that is `>= 0`, i.e. the fraction in which rho-1 is *not* actually better.

---

## Results

### Fitted final macro task-loss (bpb)

Matched-protocol Table 1. Lower is better.

| Arm                   | Fitted final | Observed | 95% CI           |
| --------------------- | ------------ | -------- | ---------------- |
| Control (full-CE)     | **1.6473**   | 1.6487   | [1.6433, 1.6515] |
| rho-1                 | 1.6824       | 1.6843   | [1.6771, 1.6857] |
| Random control        | 1.6857       | 1.6870   | [1.6775, 1.6920] |
| Attention top-k       | 1.6991       | 1.7005   | [1.6935, 1.7037] |
| BLADE                 | 1.7090       | 1.7115   | [1.7028, 1.7182] |
| Middle-PPL            | 1.9047       | 1.9001   | [1.9007, 1.9093] |
| REL-EMA (exponential) | 1.9199       | 1.9207   | [1.9180, 1.9217] |

All selection CIs sit above the control's. Note that the rho-1 and random-control
intervals **overlap heavily**, which is the substance of the test below.

### rho-1 vs random-control significance test

| Comparison              | Delta (rho-1 - random-control) | 95% CI of delta        | One-sided p (H1: rho-1 < random-control) |
| ----------------------- | ------------------------------ | ---------------------- | ---------------------------------------- |
| rho-1 vs random-control | **-0.0033**                    | **[-0.0102, +0.0042]** | **0.182**                                |

Only the one-sided p-value is reported, matching the paper.

The 95% CI of the difference straddles zero and the one-sided p does not clear 0.05:
roughly **one in five and a half** bootstrap resamples still favor random-control over
rho-1. **rho-1's apparent edge over random token masking is not statistically
significant** at this budget. And recall that this interval carries no seed variance
(above), so the true uncertainty on the -0.0033 gap is strictly larger than shown.

### Realized keep rate

The four online-scoring arms all realized a keep rate of exactly **0.599609** --
**1228 of 2047** target positions per 2048-token sequence -- and it was **constant at
every logged step**. This is the W&B metric `train/selected token fraction`, and it is
identical for `rho-1`, `attention`, `rel_ema` and `random_control`. The value is
`round(2047 * 0.6) / 2047`: the nominal 0.6 applied to the 2047 valid next-token
targets in a 2048-token sequence, rounded to an integer count per row. Because the
count is fixed per row rather than thresholded on the score, the keep rate carries no
information about the scorer -- the arms differ only in *which* 1228 tokens they keep.

**Middle-PPL used precomputed masks** (offline scoring against Reference A), so it never
logs `train/selected token fraction`; the absence of that metric for `middle-ppl-token`
is expected and is not evidence of a different keep rate.

### Cost and FLOPs

Wall-clock hours are not a usable cost measure here: five arms (rho-1, BLADE,
Attention, Middle-PPL, and the full-loss control) ran on **8xA100-80GB** while the
**Random control and REL-EMA ran on 4xL40S**, and hours on those two platforms are not
interchangeable. The **logged throughput FLOPs counter** is also incomplete, because it
**excludes the scoring forward passes** and so reports the same 2.63e19 for a full-CE
run as for a run that additionally evaluates a frozen reference on every token, an
understatement of up to ~1.9x. We therefore report analytic FLOPs throughout.

**Model.** Forward cost per token = `2N + 4*L*T*d`, with `N = 371,195,904` matmul
parameters (**including the untied output head**), `L = 16` layers, `T = 2048` sequence
length, `d = 1024` model width. The `4*L*T*d` term is the attention score/value
matmuls, which scale with sequence length and are not captured by a parameter count.
Forward+backward = **3x** forward. This model reproduces the logged W&B throughput
counter to within **0.05%**, which is what licenses using it for the arms whose counter
is incomplete.

| Arm | In-run FLOPs (x10^18) | With reference pretraining (x10^18) | Relative to control |
| --- | --- | --- | --- |
| Random control | 26.03 | 26.03 | 0.99x |
| Attention top-k | 26.11 | 26.11 | 0.99x |
| **Control (full-CE)** | **26.30** | **26.30** | **1.00x** |
| REL-EMA (exponential) | 34.71 | 34.71 | 1.32x |
| rho-1 | 34.71 | 45.08 | 1.71x |
| BLADE | 39.71 | 47.98 | 1.83x |
| Perplexity (Middle-PPL) | 34.71 | 49.21 | 1.87x |

Reading the table: Attention top-k is essentially free -- the score is read off
attention already computed in the forward pass, and dropping 40% of tokens from the
loss makes it marginally *cheaper* than full CE. Every arm that needs a second model's
forward pass -- REL-EMA's EMA copy, rho-1's and Middle-PPL's frozen reference, BLADE's
lagged reference -- pays ~1.32x in-run, and the arms that also had to *pretrain* that
reference pay up to **1.87x** end to end. **The three most expensive arms are three of
the worst-performing ones** (Perplexity 1.87x, BLADE 1.83x, rho-1 1.71x, against
Attention's 0.99x), so token selection bought negative return on a large compute
premium.

**BLADE's K-update overhead, itemized.** 5 syncs x 75 K-steps x 2 streams (proxy and
reference) = **750 full batches** of 4,194,304 tokens = **3,145,728,000 tokens** of
forward+backward, on top of the 2360 training steps. That accounts for the 39.71 vs
34.71 in-run difference.

### Takeaways

1. **Full-token CE wins.** Dropping ~40% of tokens never improved macro task-loss at
   this budget.
2. **Ranking among failures.** rho-1 ~ Random control < Attention top-k < BLADE <<
   Middle-PPL ~ REL-EMA. Excess loss vs a strong frozen reference is least damaging;
   middle-percentile perplexity and EMA-relative selection (once corrected to the same
   current-minus-history polarity as rho-1/BLADE) land in the same worst tier, close to
   each other (~+0.25 bpb over full-CE) rather than REL-EMA being a catastrophic
   outlier.
3. **rho-1's edge over random masking isn't real.** Random-control (no scoring signal,
   just a random 60% keep-rate) lands at 1.6857, essentially tied with rho-1's 1.6824.
   The bootstrap test on the difference gives **one-sided p = 0.182**, so rho-1's
   informative scoring rule isn't demonstrably better than picking tokens at random at
   this budget. Every *other* selection arm (Attention, BLADE, Middle-PPL, REL-EMA) is
   clearly worse than random-control too.
4. **BLADE's extra machinery did not pay off** -- worse than simple rho-1 despite
   3,145,728,000 extra tokens of K-update forward+backward (1.83x control FLOPs).
5. **Cost.** Token selection was the most expensive lever and the weakest scientific
   return: up to **1.87x** the control's analytic FLOPs for a strictly worse result.
6. **The control is not yet seed-matched.** The reported full-loss control differs from
   the selection arms in dataloader seed (6199 vs 42), init seed (logged 0), step count
   (2384 vs 2360) and eval grid (~119 vs 125). A matched rerun
   (`full-loss-control-regmix10b-v3`) is in flight; see `ARMS.md`.

---

## Conclusions

Under the P1 Mixing Laws Dataset × 370M one-epoch contract, **token selection is a negative result**: every tested scorer underperforms full CE, and the best scorer (ρ-1) is statistically indistinguishable from a random-60% keep-rate control (delta -0.0033, 95% CI [-0.0102, +0.0042], **one-sided p = 0.182**). Prefer mixture optimization (MixLaw) or, secondarily, difficulty curricula over token masking for this setup.

Two caveats a reader should carry out of this page: the intervals above are single-run
curve-fit intervals and contain **no seed-to-seed variance**, and the full-loss control
is **not yet seed/step/grid-matched** to the selection arms (matched rerun
`full-loss-control-regmix10b-v3` pending). The direction of the result is robust -- the
selection arms lose by 0.035 to 0.27 bpb -- but the *size* of the control's margin is
not yet pinned down.

Benchmark contamination for this corpus and both reference corpora is audited in
[`contamination/`](contamination/).
