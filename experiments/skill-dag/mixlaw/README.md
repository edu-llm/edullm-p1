# MixLaw (Mixing Laws Dataset / OLMoHQ × OLMo-2 370M)

**Question.** Can a mixing law fitted on cheap short runs predict a domain mixture that beats OLMo Mix 1124 under a matched one-epoch training budget?

**Answer.** Yes. Both fitted mixtures beat the OLMo Mix 1124 control on macro task-loss bits-per-byte (bpb), with non-overlapping 95% confidence intervals and two-sided residual-bootstrap $p < 10^{-4}$. The MixLaw optimum is slightly better than the LightGBM optimum.

---

## 370M validation setup

| Knob | Value |
|------|-------|
| Architecture | OLMo-2 370M (full attention) |
| Train stream | Domain-stratified sampling over 7 OLMoHQ domains at fixed recipe weights |
| Domains | dclm, arxiv, starcoder, pes2o, open-web-math, algebraic-stack, wiki |
| Data | The ~127B-token reservoir `pretrain/olmo-127b` v1, a subset of Olmo-mix-1124; rebuild it byte for byte from the public Olmo-mix-1124 files with [`datasets/manifests/olmo-127b-v1/`](../../../datasets/manifests/olmo-127b-v1/README.md) (`rebuild.py`). No source documents are redistributed here. |
| Global batch / seq / LR | 4,194,304 tokens / 2048 / $4\times10^{-4}$ cosine ($T_{\max}=2360$ after 24 warmup steps; 2384 steps total, $\alpha_f=0.1$) |
| Full-run budget | 2384 steps ≈ one epoch (~10B tokens); cosine horizon $T_{\max}=2360$ after 24 warmup steps |
| FLOPs / full arm | $2.63\times10^{19}$ ($6ND$ plus the PaLM attention term at $N_{\text{non-emb}} = 371{,}262{,}464$, as for the pilots below; measured from W&B) |
| Primary metric | Macro mean CE bits-per-byte over 20 OLMES-style labels (task-loss) |
| Seeds (as launched) | `--seed 12536` for all four static arms — data-stream and mixture-sampling seed; `seed_all(stream_seed + rank)` also seeds the process RNGs. Model weights are initialized from `TransformerConfig`'s own `init_seed`, and the `model.init_seed = 0` visible in the W&B config is an **unset default, not a choice**: the trainer never assigns `init_seed` (the same defect is recorded in `experiments/token-selection/ARMS.md`). The realized initialization therefore follows the parallel layout, not the stream seed: every 8×A100 arm starts at step-0 task loss 4.4261 bpb, and every 4×L40S run (the seed-12345 control and both Skill-It arms, at stream seeds 12345 and 42) at 4.4728 bpb. |

**Shared recipe across arms:** same architecture, tokenizer, batch, LR schedule, and one-epoch step budget. The four arms below differ only in **domain mixture weights**; the second Olmo-mix-1124 control additionally differs in seed, hardware and trainer (see the note below).

> **Read the seed row, not the code default.** `train_mixlaw_validation_370m.py`
> falls back to `stream_seed = recipe_seed + mix_id` when `--seed` is not passed,
> which would give the four arms *different* seeds (6198 / 6199 / 6223 / 6225).
> That is not what ran: every one of the four was launched with an explicit
> `--seed 12536`, so all four share one data order and one model initialization
> and the only thing that differs between them is the domain mixture. The W&B
> configs are the record (`eduLLM/mixlaw-1`:
> `dataset.source_mixture_config.seed = 12536`, `data_loader.seed = 12536`,
> `model.init_seed = 0` on all four). The second control (`olmo-mix-seed12345`)
> trains the same Olmo-mix-1124 mixture but differs from the first in three ways:
> dataloader seed 12345 instead of 12536; hardware, FarmShare 4×L40S (`oat-04`)
> rather than 8×A100, which also changed the realized model initialization
> (step-0 task loss 4.4728 vs 4.4261 bpb); and training code, since it ran this
> directory's `train_mixlaw_validation_370m.py` rather than the OLMo-core branch
> entrypoint the four arms used (see Training-code provenance). Despite
> its name, the first control's stream seed is 12536, not 6198 — 6198 is the
> recipe seed used to build the pools and to train the 60M pilots.

### Arms actually run (370M)

| Arm | Role | A100-h | FLOPs |
|-----|------|-------:|------:|
| OLMo Mix 1124 | Control — natural OLMo Mix 1124 domain proportions (~95% DCLM) | 47.46 | $2.63\times10^{19}$ |
| Data Mixing Laws paper | Fixed proportions from the Data Mixing Laws paper (Pilot 01) | 44.11 | $2.63\times10^{19}$ |
| LightGBM | 1%-floor LightGBM optimum | 48.39 | $2.63\times10^{19}$ |
| MixLaw | Unconstrained MixLaw optimum | 48.78 | $2.63\times10^{19}$ |
| **Total** | | **188.74** | **$1.05\times10^{20}$** |

A100-hours are the no-waste totals measured for these runs.

Exact domain weights for these four arms are in Validated mixture weights below.

---

## Training-code provenance

Where each reported 370M run ran, as recorded in its W&B run metadata
(`eduLLM/mixlaw-1`):

| Arm | W&B run | Platform | Code |
|-----|---------|----------|------|
| OLMo Mix 1124 (seed 12536) | `f8adf30b7d9d5754221bf27cd86aed2f` | RunPod, 8×A100-80GB | OLMo-core branch `edullm/mixlaw-validation-370m` at `1aec878` |
| Data Mixing Laws paper | `1e9df6cccc294c6d0f1bce328497f6a6` | RunPod, 8×A100-80GB | same branch, at `1aec878` |
| LightGBM | `78a3a85b7a5304a426f71629de27b198` | RunPod, 8×A100-80GB | same branch, at `1aec878` |
| MixLaw | `83772a7c8c5da65a988ea26f8665673e` | RunPod, 8×A100-80GB | same branch; W&B records `855af63`, and the RunPod adapter it ran was committed in the next commit, `1aec878` |
| OLMo Mix 1124 (seed 12345) | `fz8z82lh` | FarmShare, 4×L40S | this directory's [`train_mixlaw_validation_370m.py`](train_mixlaw_validation_370m.py) |

The four RunPod arms ran OLMo-core's `.edullm/runpod/entrypoint.py`, which
calls `.edullm/mixlaw_entrypoint.py` on that branch; they did not use the
trainer in this directory. The `eduLLM/skillit` W&B project holds copies of two
of them for comparison against the Skill-It arms: `ugzjxsda` is a clone of the
Data Mixing Laws paper run, and `2m00hdwr` replays the LightGBM run's eval
metrics.

## Proxy pilot (DataDecide-60M)

24 designed mixtures over the same 7 domains, each trained with a **DataDecide-60M** proxy and scored on OLMo-ladder task-loss (bits-per-byte). Surrogates are fit on **Chinchilla-extrapolated** family losses (step 5806, tokens/param = 20).

### Proxy architecture

- Hidden size 384, 16 layers, 12 heads, MLP ratio 8, sequence length 2048
- Global batch 96 sequences; learning rate $5.8\times10^{-3}$
- Tokenizer: dolma2 (100,352 embedding rows); untied LM head
- Body params 37.8M; **non-embedding params 76.3M** (76,296,576, including the untied LM head); total ~114.8M with dolma2 vocab
- **Budget:** tokens/param = 5 → **285M tokens / 1451 steps** per mixture, sized against DataDecide's original non-embedding count, 57.1M (57,078,144), as is the tpp = 20 Chinchilla target
- **Pilot FLOPs.** We use the 6ND training estimate of Kaplan et al. (2020) plus the
  attention term of Chowdhery et al. (2023, PaLM), $C \approx 6 N_{\text{non-emb}} D + 12\,n_{\text{layers}}\,s\,d_{\text{model}}\,D$,
  evaluated at the **trained** model's $N_{\text{non-emb}} = 76{,}296{,}576$ (dolma2 vocab):
  $\approx 1.74\times10^{17}$ per mix → **$\approx 4.17\times10^{18}$** for 24, about 16% of
  one $2.63\times10^{19}$ validation run.

### Evaluation and Chinchilla targets

In-run curves use six **ARC + MMLU** val families, evaluated every 120 of the 1451 steps on a fixed four-batch subsample of each family's validation split. Step-laws fit **in-run curve points only** (steps 120–1440); a post-hoc full eval at step 1451 is kept for reporting but **not** used in the step law. Curves are extrapolated to Chinchilla step **5806** (tpp = 20). Mixing-law / LightGBM targets are those six extrapolated family losses.

Observed Chinchilla-target range across the 24 pilots:

| family | min | max | std |
|--------|----:|----:|----:|
| arc_challenge | 1.5043 | 1.7831 | 0.0790 |
| arc_easy | 1.8108 | 2.1452 | 0.0878 |
| mmlu_humanities | 1.6810 | 1.9250 | 0.0678 |
| mmlu_other | 2.2812 | 2.7868 | 0.1331 |
| mmlu_social_sciences | 1.2668 | 1.4775 | 0.0608 |
| mmlu_stem | 2.1722 | 2.4410 | 0.0805 |

---

## Mixture sampling

Mixtures follow **Algorithm 2** (double-diminishing grid) from [Ye et al., Data Mixing Laws](https://arxiv.org/abs/2403.16952):

1. Start from the **Data Mixing Laws paper** mixture (Pilot 01).
2. Compute per-domain **$r_{\max}$** from the planned per-domain token counts at a 30B target corpus.
3. Sample a double-diminishing grid with step **$\delta = 0.05$** and **seed 42**.
4. Apply Algorithm 2 feasibility constraints from Ye et al. (availability-aware simplex sampling; wiki-ablation tags may zero wiki).
5. Inject three mid–high DCLM points at 50% / 55% / 60% DCLM that the coarse grid cannot reach with all domains positive.

Result: **24 designed probe points** (not uniform random simplex samples).

| Tag | Meaning |
|-----|---------|
| base | Data Mixing Laws paper reference mixture (Pilot 01) |
| C0-wiki0 | Wiki ablation (wiki = 0) |
| C0-dclm0 | DCLM ablation (dclm = 0) |
| C1-dclm50 / 55 / 60 | Injected high-DCLM points |
| C1 | Standard grid points |

### $r_{\max}$ at 30B target

| domain | $r_{\max}$ |
|--------|----------:|
| dclm | 0.9533 |
| arxiv | 0.6933 |
| starcoder | 0.6767 |
| pes2o | 0.8767 |
| open-web-math | 0.4067 |
| algebraic-stack | 0.3933 |
| wiki | 0.1220 |

Wikipedia is the binding domain: all of Wikipedia is only ~3.75B tokens, so its
pilot weight is capped at 12.2%, the share of a 30B-token corpus that the planned
3.66B Wikipedia tokens can supply. The optimized mixtures use a separate 30%
Wikipedia cap (see the constraint settings below).

### Pilot mixture domain weights

Weights sum to 1.

| mix | tag | dclm | arxiv | starcoder | pes2o | open-web-math | alg-stack | wiki |
|-----|-----|-----:|------:|----------:|------:|--------------:|----------:|-----:|
| Pilot 01 | base | 0.375 | 0.250 | 0.141 | 0.094 | 0.064 | 0.061 | 0.016 |
| Pilot 02 | C0-wiki0 | 0.059 | 0.041 | 0.650 | 0.106 | 0.100 | 0.044 | 0.000 |
| Pilot 03 | C0-wiki0 | 0.059 | 0.650 | 0.041 | 0.106 | 0.100 | 0.044 | 0.000 |
| Pilot 04 | C0-dclm0 | 0.000 | 0.041 | 0.041 | 0.425 | 0.400 | 0.044 | 0.050 |
| Pilot 05 | C1-dclm50 | 0.500 | 0.200 | 0.113 | 0.075 | 0.051 | 0.049 | 0.013 |
| Pilot 06 | C1-dclm55 | 0.550 | 0.180 | 0.101 | 0.068 | 0.046 | 0.044 | 0.011 |
| Pilot 07 | C1-dclm60 | 0.600 | 0.160 | 0.090 | 0.060 | 0.041 | 0.039 | 0.010 |
| Pilot 08 | C1 | 0.119 | 0.041 | 0.041 | 0.027 | 0.400 | 0.350 | 0.023 |
| Pilot 09 | C1 | 0.030 | 0.163 | 0.325 | 0.027 | 0.050 | 0.350 | 0.056 |
| Pilot 10 | C1 | 0.059 | 0.041 | 0.041 | 0.425 | 0.050 | 0.350 | 0.034 |
| Pilot 11 | C1 | 0.030 | 0.163 | 0.325 | 0.027 | 0.400 | 0.044 | 0.013 |
| Pilot 12 | C1 | 0.030 | 0.325 | 0.041 | 0.425 | 0.050 | 0.044 | 0.086 |
| Pilot 13 | C1 | 0.059 | 0.041 | 0.325 | 0.425 | 0.100 | 0.044 | 0.006 |
| Pilot 14 | C1 | 0.059 | 0.325 | 0.081 | 0.053 | 0.200 | 0.175 | 0.106 |
| Pilot 15 | C1 | 0.030 | 0.325 | 0.325 | 0.212 | 0.050 | 0.044 | 0.014 |
| Pilot 16 | C1 | 0.237 | 0.163 | 0.041 | 0.106 | 0.400 | 0.044 | 0.009 |
| Pilot 17 | C1 | 0.475 | 0.041 | 0.041 | 0.027 | 0.050 | 0.350 | 0.017 |
| Pilot 18 | C1 | 0.237 | 0.041 | 0.041 | 0.425 | 0.100 | 0.044 | 0.113 |
| Pilot 19 | C1 | 0.237 | 0.041 | 0.325 | 0.106 | 0.100 | 0.087 | 0.103 |
| Pilot 20 | C1 | 0.059 | 0.081 | 0.163 | 0.212 | 0.200 | 0.175 | 0.109 |
| Pilot 21 | C1 | 0.237 | 0.041 | 0.163 | 0.053 | 0.050 | 0.350 | 0.106 |
| Pilot 22 | C1 | 0.237 | 0.163 | 0.041 | 0.212 | 0.050 | 0.175 | 0.122 |
| Pilot 23 | C1 | 0.475 | 0.041 | 0.325 | 0.053 | 0.050 | 0.044 | 0.013 |
| Pilot 24 | C1 | 0.119 | 0.325 | 0.041 | 0.106 | 0.050 | 0.350 | 0.009 |

---

## Surrogate fits

Two surrogates map 7-domain weights → six Chinchilla-extrapolated losses.

| | Mixing law | LightGBM |
|---|---|---|
| Model | Regularized [Ye et al.](https://arxiv.org/abs/2403.16952) law | One gradient-boosted tree per family ([RegMix](https://arxiv.org/abs/2407.01492)-style) |
| Pilot runs | 24 | 24 |
| Chinchilla step | 5806 | 5806 |

### Parametric mixing law

$$
L_i(r) = c_i + k_i \exp\big(\mathrm{clip}(\sum_j t_{ij} r_j,\,-60,\,60)\big)
$$

More negative $t_{ij}$ means increasing domain $j$ lowers family $i$ loss. Among multi-start solutions within 1.35× best RMSE, pick the most parsimonious under:

| Parameter | Value |
|-----------|-------|
| t_soft | 4.0 |
| t_hard | 8.0 |
| k_ratio_soft | 20.0 |
| lambda_t | 0.02 |
| lambda_k | 0.05 |
| rmse_slack | 1.35 |
| n_starts | 128 |

#### Fitted $c_i$, $k_i$

| family | $c_i$ | $k_i$ | $k$/std | max$\|t\|$ | in-sample RMSE |
|--------|------:|------:|--------:|----------:|---------------:|
| arc_challenge | 1.5043 | 0.0790 | 1.00 | 3.72 | 0.0471 |
| arc_easy | 1.8108 | 0.0878 | 1.00 | 2.16 | 0.0887 |
| mmlu_humanities | 1.6022 | 0.0678 | 1.00 | 1.69 | 0.0163 |
| mmlu_other | 2.1336 | 0.1331 | 1.00 | 1.77 | 0.0951 |
| mmlu_social_sciences | 1.2571 | 0.0608 | 1.00 | 2.77 | 0.0198 |
| mmlu_stem | 2.1427 | 0.0805 | 1.00 | 1.88 | 0.0433 |

#### Skill / transfer matrix $t_{ij}$ (rows = families, columns = domains)

| family | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|--------|---:|---:|---:|---:|---:|---:|---:|
| arc_challenge | -3.72 | 2.07 | 1.20 | -0.07 | 0.49 | 1.03 | -1.50 |
| arc_easy | 1.31 | 0.69 | 0.84 | 0.72 | -2.16 | 1.48 | -1.92 |
| mmlu_humanities | -0.67 | 1.63 | 1.69 | 0.72 | 1.63 | 1.61 | -1.27 |
| mmlu_other | 0.19 | 1.77 | 1.22 | 0.70 | 1.41 | 1.58 | -1.23 |
| mmlu_social_sciences | -2.77 | 1.69 | 1.34 | 0.17 | 1.50 | 1.82 | -1.64 |
| mmlu_stem | -0.07 | 1.88 | 1.43 | -1.66 | 1.18 | 1.59 | -0.49 |

Only within-row comparisons of $t_{ij}$ are meaningful: on the simplex, adding the
same constant to a family's whole row while rescaling its $k_i$ leaves the fitted
law unchanged, so each row is pinned only by the fit's regularization. $c_i$ is
constrained at or below the lowest extrapolated loss across the 24 pilots; that
bound is active for ARC Challenge and ARC Easy.

### LightGBM

Features = 7 mixture weights; target = per-family Chinchilla loss; predicted macro = mean over families. Hyperparameters chosen by a **144-config LOO grid** minimizing macro LOO RMSE. Optima: **50k random simplex samples + SLSQP polish**.

| Parameter | Value |
|-----------|-------|
| objective | regression |
| metric | rmse |
| verbosity | -1 |
| feature_fraction | 1.0 |
| bagging_fraction | 1.0 |
| seed | 0 |
| num_leaves | 7 |
| max_depth | 3 |
| min_data_in_leaf | 2 |
| learning_rate | 0.05 |
| lambda_l2 | 0.1 |
| num_boost_round | 100 |

LOO grid: hand-picked default macro LOO RMSE 0.0439 → selected 0.0366
(num_leaves 7, max_depth 3, min_data_in_leaf 2, lr 0.05, λ₂ 0.1, rounds 100).

#### LightGBM feature importance (gain)

| family | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|--------|---:|---:|---:|---:|---:|---:|---:|
| arc_challenge | 0.75 | 0.06 | 0.02 | 0.15 | 0.32 | 0.05 | 0.18 |
| arc_easy | 0.27 | 0.32 | 0.26 | 0.08 | 0.53 | 0.19 | 0.19 |
| mmlu_humanities | 0.85 | 0.00 | 0.05 | 0.06 | 0.04 | 0.01 | 0.12 |
| mmlu_other | 1.76 | 0.25 | 0.59 | 0.96 | 0.30 | 0.06 | 0.40 |
| mmlu_social_sciences | 0.67 | 0.02 | 0.03 | 0.14 | 0.01 | 0.02 | 0.01 |
| mmlu_stem | 0.22 | 0.06 | 0.24 | 0.71 | 0.02 | 0.01 | 0.31 |

### Leave-one-out cross-validation

| Metric | Mixing law | LightGBM |
|--------|-----------:|---------:|
| Mean LOO RMSE | 0.0766 | 0.0751 |
| Mean LOO RMSE / std | 84.4% | 84.1% |
| Macro LOO RMSE | 0.0443 | 0.0366 |

| family | ML LOO | LGB LOO | ML in-sample | LGB in-sample |
|--------|-------:|--------:|-------------:|--------------:|
| arc_challenge | 0.0774 | 0.0757 | 0.0471 | 0.0079 |
| arc_easy | 0.1269 | 0.1163 | 0.0887 | 0.0159 |
| mmlu_humanities | 0.0271 | 0.0288 | 0.0163 | 0.0028 |
| mmlu_other | 0.1432 | 0.1306 | 0.0951 | 0.0111 |
| mmlu_social_sciences | 0.0269 | 0.0317 | 0.0198 | 0.0037 |
| mmlu_stem | 0.0581 | 0.0674 | 0.0433 | 0.0090 |

### Mixture optima and near-optimal candidates

Surrogate optima plus nearby mixtures (within +0.04 bpb of that model’s optimum and ≥ 8 pp ($L_\infty$) from the optimum). None exactly match a pilot point.

#### Table I: optima under the two constraint settings

Each surrogate was minimized under two settings, both keeping the 30% Wikipedia
cap: unconstrained, and with a 1% per-domain floor. **Bold** rows were trained at
370M. The predicted macro is each surrogate's own Chinchilla-extrapolated 60M
prediction, so it compares settings within a surrogate, not across the two.

| Mixture | pred. macro | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Olmo-mix-1124** (`olmo-mix-1124`) | — | 0.951 | 0.005 | 0.021 | 0.015 | 0.003 | 0.003 | 0.001 |
| **Data Mixing Laws paper** (`mix01`) | — | 0.375 | 0.250 | 0.141 | 0.094 | 0.064 | 0.061 | 0.016 |
| **MixLaw unconstrained** (`ML-pilot_caps`) | 1.7965 | 0.568 | 0.000 | 0.000 | 0.097 | 0.035 | 0.000 | 0.300 |
| MixLaw 1%-floor | 1.7984 | 0.556 | 0.010 | 0.010 | 0.092 | 0.023 | 0.010 | 0.300 |
| LightGBM unconstrained | 1.8316 | 0.354 | 0.083 | 0.061 | 0.441 | 0.030 | 0.022 | 0.009 |
| **LightGBM 1%-floor** (`LGB-min1pct`) | 1.8335 | 0.553 | 0.212 | 0.087 | 0.082 | 0.042 | 0.014 | 0.011 |

**Which optimum was trained.** The 1% floor was the default, to keep every
domain in the mixture, but the unconstrained optimum was trained instead
wherever the floor did little more than lift zero weights. For MixLaw the floor
only lifts its three zero weights (arXiv, StarCoder, Algebraic Stack) to 1% and
shifts the others minimally, so the unconstrained MixLaw optimum was trained.
For LightGBM the floor moves the optimum to a different location on the
simplex, raising DCLM from 35% to 55% and lowering pes2o from 44% to 8%, so the
1%-floor LightGBM optimum was trained. Both choices were made after the pilots,
before any validation runs. In both surrogates the floor raises the predicted
macro loss by 0.0019 bpb. The unconstrained MixLaw optimum pushes Wikipedia to
the 30% cap, beyond the 12.2% maximum across the 24 pilots, so on that domain it
extrapolates beyond the pilot grid.

In code, `MIXTURE_OPT_CONSTRAINTS` (`mixlaw_common.py`) implements the two
settings as `uncapped` and `min1pct`. A third setting, `pilot_caps` (dclm ≤ 0.6,
other domains ≤ 0.7, 0.005 ≤ wiki ≤ 0.30), returns the same optimum as
`uncapped` for both surrogates, which is why the trained MixLaw arm is named
`ML-pilot_caps`. The row labelled "optimum" in each table below is the mixture
that was trained.

That a 1% floor moves the LightGBM argmin by ~40 pp of mass while changing the
predicted macro by 0.0019 bpb — against a macro LOO RMSE of 0.0366 — says the
tree surrogate's objective is a broad plateau over the simplex and its argmin is
weakly identified. Treat the specific LightGBM weight vector accordingly.

**Mixing law**

| candidate | pred macro | max_w | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|-----------|------------:|-------:|---:|---:|---:|---:|---:|---:|---:|
| optimum | 1.7965 | 0.568 | 0.568 | — | — | 0.097 | 0.035 | — | 0.300 |
| near-opt 1 | 1.7995 | 0.440 | 0.440 | 0.011 | 0.011 | 0.207 | 0.013 | 0.018 | 0.300 |
| near-opt 2 | 1.8016 | 0.435 | 0.435 | 0.018 | 0.011 | 0.134 | 0.069 | 0.032 | 0.300 |
| near-opt 3 | 1.8022 | 0.354 | 0.354 | 0.020 | 0.012 | 0.258 | 0.027 | 0.029 | 0.300 |
| near-opt 4 | 1.8024 | 0.440 | 0.440 | 0.012 | 0.012 | 0.043 | 0.181 | 0.012 | 0.300 |
| near-opt 5 | 1.8026 | 0.346 | 0.346 | 0.012 | 0.012 | 0.180 | 0.137 | 0.014 | 0.300 |
| near-opt 6 | 1.8032 | 0.309 | 0.274 | 0.011 | 0.021 | 0.309 | 0.073 | 0.011 | 0.300 |
| near-opt 7 | 1.8036 | 0.397 | 0.223 | 0.010 | 0.010 | 0.397 | 0.051 | 0.010 | 0.300 |
| near-opt 8 | 1.8041 | 0.638 | 0.638 | 0.024 | 0.036 | 0.010 | 0.016 | 0.010 | 0.267 |

**LightGBM**

| candidate | pred macro | max_w | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|-----------|------------:|-------:|---:|---:|---:|---:|---:|---:|---:|
| optimum | 1.8335 | 0.553 | 0.553 | 0.212 | 0.087 | 0.082 | 0.042 | 0.014 | 0.011 |
| near-opt 1 | 1.8335 | 0.558 | 0.558 | 0.199 | 0.073 | 0.083 | 0.033 | 0.042 | 0.012 |
| near-opt 2 | 1.8380 | 0.491 | 0.491 | 0.167 | 0.071 | 0.087 | 0.046 | 0.073 | 0.064 |
| near-opt 3 | 1.8383 | 0.495 | 0.495 | 0.167 | 0.072 | 0.179 | 0.015 | 0.055 | 0.017 |
| near-opt 4 | 1.8391 | 0.502 | 0.502 | 0.172 | 0.095 | 0.128 | 0.048 | 0.022 | 0.033 |
| near-opt 5 | 1.8397 | 0.505 | 0.505 | 0.190 | 0.125 | 0.080 | 0.021 | 0.024 | 0.055 |
| near-opt 6 | 1.8398 | 0.520 | 0.520 | 0.188 | 0.083 | 0.078 | 0.049 | 0.044 | 0.037 |
| near-opt 7 | 1.8398 | 0.583 | 0.583 | 0.171 | 0.082 | 0.069 | 0.029 | 0.048 | 0.018 |
| near-opt 8 | 1.8410 | 0.377 | 0.182 | 0.182 | 0.076 | 0.377 | 0.020 | 0.037 | 0.127 |

### Random-simplex plausibility

1000 mixtures ~ Dirichlet(1,…,1) on the 7-simplex (seed 42). Both surrogates predict the Chinchilla-extrapolated loss, so they are compared against the pilots' Chinchilla-extrapolated macro range, **1.8369 – 2.0561 bpb** (the raw observed pilot range, 2.0444 – 2.2888 bpb, is on a different basis).

| Metric | Mixing law | LightGBM |
|--------|-----------:|---------:|
| Macro min | 1.7838 | 1.8408 |
| Macro p50 | 1.8895 | 1.9297 |
| Macro p95 | 1.9973 | 2.0022 |
| Macro p99 | 2.0460 | 2.0217 |
| Macro max | 2.1064 | 2.0339 |
| Macro mean ± std | 1.8968 ± 0.0542 | 1.9393 ± 0.0441 |
| % inside pilots' extrapolated macro range | 87.6% (117 below, 7 above) | 100.0% |
| Mixtures with macro > 3 bpb | 0 | 0 |
| Mixtures with macro > 5 bpb | 0 | 0 |

Off-hull predictions stay bounded; claimed gains of the surrogate optima over the best measured pilot are **not distinguishable from LOO error** at 60M scale — hence the 370M validation.

### Pilot mixtures ranked by predicted macro (top 12)

| ML rank | LGB rank | mix | tag | ML pred | LGB pred | measured curve-6 |
|--------:|---------:|-----|-----|--------:|---------:|-----------------:|
| 1 | 4 | Pilot 18 | C1 | 1.8368 | 1.8700 | 2.1089 |
| 2 | 2 | Pilot 07 | C1-dclm60 | 1.8618 | 1.8451 | 2.0444 |
| 3 | 7 | Pilot 22 | C1 | 1.8666 | 1.8810 | 2.0899 |
| 4 | 1 | Pilot 06 | C1-dclm55 | 1.8668 | 1.8370 | 2.0963 |
| 5 | 5 | Pilot 05 | C1-dclm50 | 1.8726 | 1.8714 | 2.0830 |
| 6 | 14 | Pilot 23 | C1 | 1.8733 | 1.9107 | 2.0981 |
| 7 | 17 | Pilot 19 | C1 | 1.8752 | 1.9628 | 2.1456 |
| 8 | 13 | Pilot 17 | C1 | 1.8897 | 1.9093 | 2.0835 |
| 9 | 10 | Pilot 01 | base | 1.8913 | 1.8951 | 2.0727 |
| 10 | 8 | Pilot 04 | C0-dclm0 | 1.8964 | 1.8888 | 2.1174 |
| 11 | 6 | Pilot 21 | C1 | 1.8969 | 1.8764 | 2.1364 |
| 12 | 3 | Pilot 16 | C1 | 1.8992 | 1.8527 | 2.1125 |

---

## Validated mixture weights (370M)

Recipe domain weights for the four 370M arms that were trained:

| arm | source | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|-----|--------|---:|---:|---:|---:|---:|---:|---:|
| OLMo Mix 1124 | reference | 0.951 | 0.005 | 0.021 | 0.015 | 0.003 | 0.003 | 0.001 |
| Data Mixing Laws paper | pilot | 0.375 | 0.250 | 0.141 | 0.094 | 0.064 | 0.061 | 0.016 |
| MixLaw | mixing-law | 0.568 | 0.000 | 0.000 | 0.097 | 0.035 | 0.000 | 0.300 |
| LightGBM | lightgbm | 0.553 | 0.212 | 0.087 | 0.082 | 0.042 | 0.014 | 0.011 |

**What each validated arm manipulates**

- **OLMo Mix 1124 (control)** — hold mix fixed at the natural OLMo Mix 1124 corpus proportions (very DCLM-heavy).
- **Data Mixing Laws paper** — hold domain mix fixed at the Data Mixing Laws paper proportions (Pilot 01). That paper's published optimal split (CommonCrawl 0.1250, C4 0.2500, GitHub 0.1406, arXiv 0.2500, Books 0.0938, StackExchange 0.1250, Wikipedia 0.0156) is mapped onto our domains by semantic category: CommonCrawl + C4 → DCLM, GitHub → StarCoder, Books → pes2o, StackExchange → OpenWebMath (0.0635) + Algebraic Stack (0.0615), arXiv and Wikipedia directly.
- **MixLaw** — train the unconstrained MixLaw optimum. Method from [Ye et al., Data Mixing Laws](https://arxiv.org/abs/2403.16952).
- **LightGBM** — train the 1%-floor LightGBM optimum. Method from [Liu et al., RegMix](https://arxiv.org/abs/2407.01492) (regression / tree surrogate over mixture probes).

---

## 370M evaluation and uncertainty

Task-loss is evaluated on the shared 20-label suite every 125 of the 2384 steps; the step-2375 eval is excluded for its proximity to the final step. Final performance uses a **power-law residual bootstrap**:

1. Fit $y = a + b / \mathrm{step}^{\alpha}$ on all eval points with step ≥ 1000 ($\alpha$ on a grid in $[0.05, 3]$).
2. Take the fitted value at the final step as the point estimate.
3. Alpha-free residual bootstrap (200,000 draws, $\alpha$ re-selected on every draw) for a 95% CI on that fitted final.
4. Pairwise $\Delta = \mathrm{arm} - \mathrm{control}$ from independent bootstraps (control = the two-seed Olmo-mix-1124 average); two-sided $p$ reported only when the arm beats control.

---

## 370M results

### Fitted final macro task-loss (bpb)

Regenerate with [`fit_and_bootstrap_370m.py`](fit_and_bootstrap_370m.py) from the
committed curves in [`skill_dag_370m_wandb_curves.json`](skill_dag_370m_wandb_curves.json);
full output in [`skill_dag_370m_bootstrap_results.json`](skill_dag_370m_bootstrap_results.json).
The control is the **average of the two Olmo-mix-1124 dataloader seeds** (12536, 12345),
whose bootstrap distribution is the element-by-element mean of the two seeds' own
alpha-free bootstrap distributions. Every arm is resampled from its own
independent random stream, so between-arm intervals are not coupled through a
shared generator.

| Arm | Fitted final | Observed | 95% CI | vs control |
|-----|-------------:|---------:|--------|------------|
| **MixLaw** | **1.6057** | 1.6048 | [1.6015, 1.6099] | $p < 10^{-4}$ |
| **LightGBM** | **1.6080** | 1.6077 | [1.6049, 1.6106] | $p < 10^{-4}$ |
| Olmo-mix-1124 seed 12536 | 1.6313 | 1.6370 | [1.6261, 1.6357] | — |
| Olmo-mix-1124 seed 12345 | 1.6269 | 1.6285 | [1.6195, 1.6343] | — |
| Olmo-mix-1124 average (control) | 1.6291 | 1.6327 | [1.6246, 1.6335] | — |
| Data Mixing Laws paper | 1.6515 | 1.6518 | [1.6473, 1.6556] | worse by 0.0224 [0.0163, 0.0285] |

Lower is better. Both fitted mixtures beat the Olmo-mix-1124 control. The Data Mixing
Laws paper mixture finishes **worse** than the control, by 0.0224 bpb (95% CI
[0.0163, 0.0285]).

**Seed-variance reference.** The two control seeds differ by 0.0044 bpb
(95% CI [-0.0046, 0.0132], $p = 0.34$). The two runs differ in dataloader seed
(12536 vs 12345), in hardware (8×A100 vs 4×L40S), which also changed the realized
model initialization (step-0 task loss 4.4261 vs 4.4728 bpb), and in training code
(the second control ran a separate trainer; see Training-code provenance), so this
is not a pure data-order contrast. Any effect smaller than ~0.004 bpb is inside
that noise floor.

### Targeted vs never-targeted labels

The six fitted skills supply 12 of the 20 labels (validation and test splits of
ARC Easy, ARC Challenge and the four MMLU groups); the other 8 (BoolQ, CSQA,
HellaSwag, OpenBookQA validation and test, PIQA, SocialIQA, WinoGrande) were never
optimized. Same fit and bootstrap as above, on each subset's per-step mean
([`heldout_label_bootstrap.py`](heldout_label_bootstrap.py) from
[`heldout_label_curves.json`](heldout_label_curves.json)); improvement over the
control, positive = better:

| Labels | MixLaw | LightGBM |
|---|---|---|
| 12 targeted | 0.0351 [0.0275, 0.0425], $p < 10^{-4}$ | 0.0397 [0.0331, 0.0461], $p < 10^{-4}$ |
| 8 never-targeted | 0.0055 [-0.0009, 0.0126], $p = 0.10$ | -0.0067 [-0.0128, -0.0003], $p = 0.039$ |

The gain is concentrated on the targeted labels. Per label
([`heldout_perlabel_bootstrap.py`](heldout_perlabel_bootstrap.py)), HellaSwag and
PIQA are worse under both mixtures (MixLaw +0.0315 and +0.0303 bpb, LightGBM
+0.0227 and +0.0219, all $p < 10^{-4}$) against seed spreads of 0.0024 and 0.0034
bpb. BoolQ moves in opposite directions, 0.0931 bpb better under MixLaw and 0.0990
worse under LightGBM, against a 0.0472 bpb seed spread.

### Takeaways

1. **Mixing-law search works at this scale.** Short-run fits produced mixtures that clearly dominate the OLMo Mix 1124 control under a matched one-epoch budget.
2. **MixLaw ≈ LightGBM**, with a small edge to the MixLaw optimum (0.0023 bpb, inside the 0.0044 bpb seed noise floor).
3. **Data Mixing Laws paper proportions underperform the control.** That fixed paper mix finishes worse than OLMo Mix 1124.
4. **Cost.** Four full arms 188.74 A100-hours and $\approx 1.05\times10^{20}$ FLOPs, plus $\approx 4.17\times10^{18}$ FLOPs for the 60M pilot grid.
5. **Compute savings.** The MixLaw mixture reaches the two-seed control's fitted
   final in 21.7% fewer steps (95% CI [18.4%, 24.8%]); the control would take
   1.53× as long (95% CI [1.27×, 2.66×]) to reach the MixLaw mixture's final,
   assuming its fitted power law holds past the end of its LR schedule. The 24
   pilots cost about 16% of one validation run, so the net compute saving is
   5.9% (95% CI [2.6%, 9.0%]); equivalently, the control would consume 32% more
   FLOPs (95% CI [10%, 129%]) than the optimized mixture plus its pilots. The point
   estimates are reproduced by
   [`plot_figure_iii_compute_savings.py`](plot_figure_iii_compute_savings.py)
   (Figure III).

---

## Conclusions

Under the P1 one-epoch 370M contract, **data mixture is a high-leverage lever**: mixing-law optimization yields 0.021–0.023 bpb absolute macro task-loss improvement over the OLMo Mix 1124 control, with tight CIs and decisive bootstrap significance. Among levers in this campaign, MixLaw is the clearest positive result.

Benchmark contamination between the shared 127B-token reservoir and the
evaluation suite, and its projection onto each arm's mixture, is audited in
[contamination/](contamination/).
