# MixLaw (Mixing Laws Dataset / OLMoHQ × OLMo-2 370M)

**Question.** Can a mixing law fitted on cheap short runs predict a domain mixture that beats OLMo Mix 1124 under a matched one-epoch training budget?

**Answer.** No. All three optimized static mixtures finish behind the Olmo-mix-1124 control (average of data seeds 42 and 69, fitted final macro task-loss 1.6148 bits-per-byte (bpb)): the Data Mixing Laws paper mixture by 0.0264 bpb (95% CI [+0.0200, +0.0325]), the MixLaw 1%-floor optimum by 0.0286 ([+0.0234, +0.0338]), both with two-sided residual-bootstrap $p < 5\times10^{-6}$, and the LightGBM 1%-floor optimum by 0.0091 ([+0.0026, +0.0158], $p = 0.0048$). The two control seeds differ by 0.0105 bpb, comparable to the LightGBM gap and well below the other two. All arms are single runs, so the bootstrap intervals cover power-law-fit uncertainty within a run only, and the two control seeds are the only run-to-run estimate.

---

## 370M validation setup

| Knob | Value |
|------|-------|
| Architecture | OLMo-2 370M (full attention) |
| Train stream | Domain-stratified sampling over 7 OLMoHQ domains at fixed recipe weights |
| Domains | dclm, arxiv, starcoder, pes2o, open-web-math, algebraic-stack, wiki |
| Data | The ~127B-token reservoir `pretrain/olmo-127b` v1, a subset of Olmo-mix-1124; rebuild it byte for byte from the public Olmo-mix-1124 files with [`datasets/manifests/olmo-127b-v1/`](../../../datasets/manifests/olmo-127b-v1/README.md) (`rebuild.py`). No source documents are redistributed here. |
| Global batch / seq / LR | 4,194,304 tokens / 2048 / $4\times10^{-4}$ cosine ($T_{\max}=2360$ after 24 warmup steps; 2384 steps total, $\alpha_f=0.1$) |
| Full-run budget | 2384 steps ≈ one epoch (~10B tokens) |
| Hardware / trainer | FarmShare 4×L40S, the vendored Skill-It trainer ([`../skillit/olmo_core_skillit/`](../skillit/olmo_core_skillit/)) with dynamic reweighting disabled for the static arms; the same for every run reported here |
| Data seed | 42 for every arm; the second Olmo-mix-1124 control uses 69. The seed sets the data-stream and mixture-sampling order; model initialization is the same in every run (step-0 macro task loss 4.4728 bpb in all of them) |
| FLOPs / full arm | $2.63\times10^{19}$ ($6ND$ plus the PaLM attention term at $N_{\text{non-emb}} = 371{,}262{,}464$, as for the pilots below; measured from W&B) |
| Primary metric | Macro mean CE bits-per-byte over 20 OLMES-style labels (task-loss) |

**Shared recipe across arms:** same architecture, tokenizer, batch, LR schedule, hardware, trainer, initialization and one-epoch step budget. The four static arms below differ only in **domain mixture weights**; the second Olmo-mix-1124 control additionally differs in data seed.

### Arms actually run (370M)

| Arm | Role | Weights | Data seed | FLOPs |
|-----|------|---------|----------:|------:|
| Olmo-mix-1124 | Control — natural Olmo-mix-1124 domain proportions (~95% DCLM) | `olmo-mix-1124` | 42 and 69 | $2.63\times10^{19}$ per run |
| Data Mixing Laws paper | Fixed proportions from the Data Mixing Laws paper (Pilot 01) | `mix01` | 42 | $2.63\times10^{19}$ |
| LightGBM | 1%-floor LightGBM optimum | `LGB-min1pct` | 42 | $2.63\times10^{19}$ |
| MixLaw | 1%-floor MixLaw optimum | `ML-min1pct` | 42 | $2.63\times10^{19}$ |
| **Total** | 5 runs | | | **$1.32\times10^{20}$** |

The control is reported as the average of its two data seeds. Exact domain weights for these four mixtures are in Validated mixture weights below. The two Skill-It (dynamic reweighting) arms from the same setup are reported in [`../skillit/README.md`](../skillit/README.md#results).

---

## Training-code provenance

Where each reported 370M run ran, as recorded in its W&B run metadata
(`eduLLM/mixlaw-new`, `eduLLM/skillit`); all on FarmShare 4×L40S:

| Run | Slurm job | W&B run | Data seed | Step-0 macro bpb |
|-----|----------:|---------|----------:|-----------------:|
| Olmo-mix-1124 control | 1745704 | `eduLLM/mixlaw-new/i9z1vtbt` | 42 | 4.4728 |
| Olmo-mix-1124 control | 1760339 | `eduLLM/mixlaw-new/3vbmxzmg` | 69 | 4.4728 |
| Data Mixing Laws paper | 1745708 | `eduLLM/mixlaw-new/z0alta8r` | 42 | 4.4728 |
| MixLaw | 1745706 | `eduLLM/mixlaw-new/nm6i8hxy` | 42 | 4.4728 |
| LightGBM | 1744338 | `eduLLM/mixlaw-new/zgmte13g` | 42 | 4.4728 |
| Skill-It offline probe (matrix rebuilt against the LightGBM-start probe) | 1771665 | `eduLLM/skillit/iy441nc7` | 42 | 4.4728 |
| Skill-It online derivative | 1728144 | `eduLLM/skillit/c0844ce36f24d6773c7f45cb31d810f4` | 42 | 4.4728 |

The step-0 value is 4.47281289100647 in all seven runs. All seven ran the vendored
Skill-It trainer in [`../skillit/olmo_core_skillit/`](../skillit/olmo_core_skillit/);
its [`PROVENANCE.md`](../skillit/olmo_core_skillit/PROVENANCE.md) records where the code
came from, how it differs from the nearest upstream commit, and what was left out.
The two Skill-It runs are described in [`../skillit/README.md`](../skillit/README.md#training-code-provenance).

The static runs hold the domain weights fixed for the whole run (a static arm
of the same entrypoint, no Skill-It updates). The four runs in
`eduLLM/mixlaw-new` were produced by
[`patch_static_validation_arms.py`](../skillit/olmo_core_skillit/farmshare/patch_static_validation_arms.py),
which adds the static arms (arm 3 and 4, the two controls; arm 5, Data Mixing
Laws paper; arm 6, MixLaw) to a copy of the vendored bundle with their weights and data seeds,
and launched with
[`farmshare_static_validation_l40s.sbatch`](../skillit/olmo_core_skillit/farmshare/farmshare_static_validation_l40s.sbatch),
one job per arm. The LightGBM static run (arm 2) was added to its own copy of the bundle by
[`patch_legacy_static_lgbm_arm.py`](../skillit/olmo_core_skillit/farmshare/patch_legacy_static_lgbm_arm.py)
and launched with
[`farmshare_static_lgbm_l40s.sbatch`](../skillit/olmo_core_skillit/farmshare/farmshare_static_lgbm_l40s.sbatch);
`patch_static_validation_arms.py` folds in the same arm-2 patch. W&B marks that run failed only
because OLMo-core's W&B callback called `wandb.finish(quiet=...)`, which the installed wandb rejects,
after all 2384 steps had trained and the step-2384 eval had been logged.

**Reproducing the arm weights.** `patch_static_validation_arms.py` reads the MixLaw arm's weights from
`mixlaw_fit_chinchilla.json` → `optimization.min1pct`. That file now holds the refit of the rerun pilots
(see [Proxy pilot](#proxy-pilot-datadecide-60m)). The arms that ran were chosen from the earlier fit, which is in git history at commit
`c3d9906` (check it out to rerun the patch script); the weights actually trained are also recorded in `validation_mixtures_10b.json`.

## Proxy pilot (DataDecide-60M)

24 designed mixtures over the same 7 domains, each trained as a **DataDecide-60M** proxy and scored on OLMo-ladder task-loss (bits-per-byte). The pilots and the 7 one-hot probes of the Skill-It matrix come from one trainer and one launcher, [`../domain_probes/`](../domain_probes/README.md). Surrogates are fit on **Chinchilla-extrapolated** family losses (step 5806, tokens/param = 20).

**This section describes the rerun of the pilots.** The 370M arms above were trained at the optima of an earlier fit, whose pilots ran 1451 steps on different hardware and an older data path and scored only the first 128 items of each validation split. That prefix is not representative of the split, so all 24 pilots (and the 7 probes) were rerun with every item of the test split ([why, and the setup](../domain_probes/README.md)). The 370M runs were not repeated; [Earlier fit versus refit](#earlier-fit-versus-refit) compares the two sets of optima.

### Proxy architecture

- Hidden size 384, 16 layers, 12 heads, MLP ratio 8, sequence length 2048
- Global batch 96 sequences; learning rate $5.8\times10^{-3}$
- Tokenizer: dolma2 (100,352 embedding rows); untied LM head
- Body params 37.8M; **non-embedding params 76.3M** (76,296,576, including the untied LM head); total ~114.8M with dolma2 vocab
- **Budget:** 1440 steps of 96 sequences = **283M tokens** per mixture (tokens/param = 4.96), sized against DataDecide's published non-embedding count, 57.1M (57,078,144), as is the tpp = 20 Chinchilla target. The learning rate warms up over 144 steps, holds, and decays by cosine to 10% of peak over the last 144.
- **Pilot FLOPs.** We use the 6ND training estimate of Kaplan et al. (2020) plus the
  attention term of Chowdhery et al. (2023, PaLM), $C \approx 6 N_{\text{non-emb}} D + 12\,n_{\text{layers}}\,s\,d_{\text{model}}\,D$,
  evaluated at the **trained** model's $N_{\text{non-emb}} = 76{,}296{,}576$ (dolma2 vocab):
  $\approx 1.72\times10^{17}$ per mix → **$\approx 4.14\times10^{18}$** for 24, about 16% of
  one $2.63\times10^{19}$ validation run (the 7 probes add $1.21\times10^{18}$). `flops.py` computes all of these from the
  two architectures and the trained token counts.

### Evaluation and Chinchilla targets

In-run curves use six **ARC + MMLU** test families, evaluated every 120 of the 1440 steps on **every item** of each family's test split (1172, 2376, 3018, 4705, 3077 and 3242 items for ARC Challenge, ARC Easy and the MMLU STEM, humanities, social-sciences and other groups). Step-laws fit all 12 in-run points (steps 120–1440); the last point is the run's final loss, so there is no separate post-hoc eval. Curves are extrapolated to Chinchilla step **5806** (tpp = 20). Mixing-law / LightGBM targets are those six extrapolated family losses.

Observed Chinchilla-target range across the 24 pilots:

| family | min | max | std |
|--------|----:|----:|----:|
| arc_challenge | 1.6395 | 2.0027 | 0.0960 |
| arc_easy | 1.6368 | 1.9285 | 0.0916 |
| mmlu_humanities | 1.2417 | 1.4282 | 0.0559 |
| mmlu_other | 1.9674 | 2.2725 | 0.0737 |
| mmlu_social_sciences | 1.4839 | 1.6823 | 0.0585 |
| mmlu_stem | 2.4705 | 2.8377 | 0.1011 |

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
| arc_challenge | 0.8077 | 0.0960 | 1.00 | 2.59 | 0.0530 |
| arc_easy | 0.0500 | 0.0916 | 1.00 | 3.12 | 0.0812 |
| mmlu_humanities | 1.0524 | 0.0559 | 1.00 | 2.08 | 0.0200 |
| mmlu_other | 1.9674 | 0.0737 | 1.00 | 1.73 | 0.0326 |
| mmlu_social_sciences | 0.0500 | 0.0585 | 1.00 | 3.35 | 0.0213 |
| mmlu_stem | 2.4035 | 0.1011 | 1.00 | 1.89 | 0.0764 |

#### Skill / transfer matrix $t_{ij}$ (rows = families, columns = domains)

| family | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|--------|---:|---:|---:|---:|---:|---:|---:|
| arc_challenge | 2.06 | 2.37 | 2.59 | 2.27 | 2.42 | 2.26 | 2.39 |
| arc_easy | 2.87 | 2.92 | 2.86 | 2.87 | 3.12 | 2.98 | 2.97 |
| mmlu_humanities | 0.82 | 2.00 | 1.59 | 2.08 | 1.85 | 1.90 | 0.56 |
| mmlu_other | -1.45 | 1.20 | 1.61 | -0.13 | 1.73 | 1.17 | 0.24 |
| mmlu_social_sciences | 3.13 | 3.30 | 3.34 | 3.28 | 3.35 | 3.26 | 3.22 |
| mmlu_stem | -0.27 | -0.17 | 1.89 | 1.32 | 1.04 | 1.65 | -1.06 |

Only within-row comparisons of $t_{ij}$ are meaningful: on the simplex, adding the
same constant to a family's whole row while rescaling its $k_i$ leaves the fitted
law unchanged, so each row is pinned only by the fit's regularization. $c_i$ is
constrained at or below the lowest extrapolated loss across the 24 pilots; that
bound is active for MMLU Other. The ARC Easy and MMLU Social Sciences fits sit at the
0.05 floor of $c_i$, an intercept far below any observed loss, and their rows of $t_{ij}$ are all near 3,
so those two fits carry little beyond the overall level of the family and are weakly identified.

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
| num_leaves | 5 |
| max_depth | 2 |
| min_data_in_leaf | 4 |
| learning_rate | 0.1 |
| lambda_l2 | 1.0 |
| num_boost_round | 100 |

LOO grid (144 configs): hand-picked default macro LOO RMSE 0.0466 → selected 0.0395 (num_leaves 5, max_depth 2, min_data_in_leaf 4, lr 0.1, λ₂ 1.0, rounds 100).

#### LightGBM feature importance (gain)

| family | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|--------|---:|---:|---:|---:|---:|---:|---:|
| arc_challenge | 0.66 | 0.02 | 0.21 | 0.03 | 0.14 | 0.01 | 0.03 |
| arc_easy | 0.06 | 0.16 | 0.03 | 0.20 | 0.27 | 0.05 | 0.14 |
| mmlu_humanities | 0.32 | 0.01 | 0.01 | 0.02 | 0.00 | 0.02 | 0.00 |
| mmlu_other | 0.44 | 0.02 | 0.02 | 0.10 | 0.07 | 0.00 | 0.02 |
| mmlu_social_sciences | 0.27 | 0.01 | 0.02 | 0.03 | 0.05 | 0.01 | 0.03 |
| mmlu_stem | 0.19 | 0.24 | 0.38 | 0.06 | 0.20 | 0.02 | 0.11 |

### Leave-one-out cross-validation

| Metric | Mixing law | LightGBM |
|--------|-----------:|---------:|
| Mean LOO RMSE | 0.0732 | 0.0657 |
| Mean LOO RMSE / std | 84.9% | 77.2% |
| Macro LOO RMSE | 0.0422 | 0.0395 |

| family | ML LOO | LGB LOO | ML in-sample | LGB in-sample |
|--------|-------:|--------:|-------------:|--------------:|
| arc_challenge | 0.0870 | 0.0744 | 0.0530 | 0.0185 |
| arc_easy | 0.0898 | 0.1069 | 0.0812 | 0.0348 |
| mmlu_humanities | 0.0286 | 0.0238 | 0.0200 | 0.0056 |
| mmlu_other | 0.0524 | 0.0384 | 0.0326 | 0.0096 |
| mmlu_social_sciences | 0.0264 | 0.0351 | 0.0213 | 0.0073 |
| mmlu_stem | 0.1552 | 0.1154 | 0.0764 | 0.0258 |

### Mixture optima and near-optimal candidates

Surrogate optima plus nearby mixtures (within +0.04 bpb of that model’s optimum and ≥ 8 pp ($L_\infty$) from the optimum). None exactly match a pilot point.

#### Table I: the four 370M mixtures

The 370M arms were trained at the optima of the earlier fit. Table I scores them with the refit surrogates and adds the refit optima. The predicted
macro is each surrogate's Chinchilla-extrapolated 60M prediction, so the two columns are on different scales. The surrogates disagree on
the natural mixture: the mixing law predicts it better than every pilot and than its own 1%-floor optimum below (the natural mixture
puts less than 1% on wiki, which the floor forbids), while LightGBM predicts it behind its own optimum.

| Mixture | ML pred. macro | LGB pred. macro | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Olmo-mix-1124 (`olmo-mix-1124`) | 1.7179 | 1.8062 | 0.951 | 0.005 | 0.021 | 0.015 | 0.003 | 0.003 | 0.001 |
| Data Mixing Laws paper (`mix01`) | 1.8189 | 1.8491 | 0.375 | 0.250 | 0.141 | 0.094 | 0.064 | 0.061 | 0.016 |
| MixLaw (`ML-min1pct`) | 1.7557 | 1.8201 | 0.556 | 0.010 | 0.010 | 0.092 | 0.023 | 0.010 | 0.300 |
| LightGBM (`LGB-min1pct`) | 1.7810 | 1.7784 | 0.553 | 0.212 | 0.087 | 0.082 | 0.042 | 0.014 | 0.011 |
| refit MixLaw optimum | 1.7191 | 1.8027 | 0.940 | 0.010 | 0.010 | 0.010 | 0.010 | 0.010 | 0.010 |
| refit LightGBM optimum | 1.7863 | 1.7690 | 0.551 | 0.179 | 0.159 | 0.019 | 0.033 | 0.050 | 0.010 |

**Selection rule.** Both surrogates are minimized with a 1% per-domain floor,
so every domain stays in the mixture, and the 30% Wikipedia cap
(`MIXTURE_OPT_CONSTRAINTS` in `mixlaw_common.py`). The row labelled "optimum"
in each table below is that minimizer. The refit MixLaw
optimum puts 94% of the mixture in DCLM and every other domain at the 1% floor, beyond the 60% maximum
DCLM weight across the 24 pilots, so it extrapolates far beyond the pilot grid. (The earlier fit's MixLaw optimum, the one trained at 370M, held arXiv,
StarCoder and Algebraic Stack at the floor and put Wikipedia at the 30% cap, beyond the 12.2% maximum across the pilots.)

The LightGBM surrogate is piecewise constant, so its minimum is a region, not a
point: near-opt 1 below predicts 1.7683 (the optimum: 1.7690) while moving up to
7.0 pp of weight per domain, and all eight near-optimal mixtures lie within
0.0050 bpb of the optimum, against a macro LOO RMSE of 0.0395. Its argmin is weakly
identified. Treat the specific LightGBM weight vector accordingly.

**Mixing law**

| candidate | pred macro | max_w | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|-----------|------------:|-------:|---:|---:|---:|---:|---:|---:|---:|
| optimum | 1.7191 | 0.940 | 0.940 | 0.010 | 0.010 | 0.010 | 0.010 | 0.010 | 0.010 |
| near-opt 1 | 1.7334 | 0.825 | 0.825 | 0.010 | 0.010 | 0.109 | 0.011 | 0.025 | 0.010 |
| near-opt 2 | 1.7361 | 0.811 | 0.811 | 0.010 | 0.024 | 0.034 | 0.039 | 0.018 | 0.064 |
| near-opt 3 | 1.7390 | 0.791 | 0.791 | 0.090 | 0.010 | 0.053 | 0.018 | 0.019 | 0.018 |
| near-opt 4 | 1.7390 | 0.739 | 0.739 | 0.048 | 0.010 | 0.059 | 0.010 | 0.010 | 0.124 |
| near-opt 5 | 1.7442 | 0.748 | 0.748 | 0.010 | 0.010 | 0.018 | 0.017 | 0.130 | 0.068 |
| near-opt 6 | 1.7458 | 0.731 | 0.731 | 0.065 | 0.031 | 0.122 | 0.010 | 0.010 | 0.031 |
| near-opt 7 | 1.7460 | 0.737 | 0.737 | 0.164 | 0.010 | 0.056 | 0.010 | 0.010 | 0.012 |
| near-opt 8 | 1.7462 | 0.746 | 0.746 | 0.014 | 0.119 | 0.011 | 0.010 | 0.017 | 0.082 |

**LightGBM**

| candidate | pred macro | max_w | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|-----------|------------:|-------:|---:|---:|---:|---:|---:|---:|---:|
| optimum | 1.7690 | 0.551 | 0.551 | 0.179 | 0.159 | 0.019 | 0.033 | 0.050 | 0.010 |
| near-opt 1 | 1.7683 | 0.598 | 0.598 | 0.178 | 0.088 | 0.048 | 0.027 | 0.051 | 0.010 |
| near-opt 2 | 1.7700 | 0.517 | 0.517 | 0.168 | 0.067 | 0.180 | 0.010 | 0.048 | 0.010 |
| near-opt 3 | 1.7700 | 0.378 | 0.378 | 0.220 | 0.068 | 0.255 | 0.010 | 0.058 | 0.010 |
| near-opt 4 | 1.7718 | 0.343 | 0.343 | 0.185 | 0.074 | 0.056 | 0.030 | 0.302 | 0.010 |
| near-opt 5 | 1.7725 | 0.432 | 0.432 | 0.174 | 0.088 | 0.176 | 0.010 | 0.109 | 0.010 |
| near-opt 6 | 1.7725 | 0.358 | 0.358 | 0.167 | 0.096 | 0.167 | 0.010 | 0.192 | 0.010 |
| near-opt 7 | 1.7731 | 0.368 | 0.368 | 0.210 | 0.148 | 0.077 | 0.017 | 0.171 | 0.010 |
| near-opt 8 | 1.7733 | 0.479 | 0.479 | 0.200 | 0.039 | 0.010 | 0.021 | 0.240 | 0.010 |

### Random-simplex plausibility

1000 mixtures ~ Dirichlet(1,…,1) on the 7-simplex (seed 42). Both surrogates predict the Chinchilla-extrapolated loss, so they are compared against the pilots' Chinchilla-extrapolated macro range, **1.7567 - 1.9930 bpb** (the raw observed pilot range, 1.9768 - 2.1857 bpb, is on a different basis).

| Metric | Mixing law | LightGBM |
|--------|-----------:|---------:|
| Macro min | 1.7509 | 1.7793 |
| Macro p50 | 1.8702 | 1.9013 |
| Macro p95 | 1.9328 | 1.9598 |
| Macro p99 | 1.9521 | 1.9786 |
| Macro max | 1.9883 | 2.0002 |
| Macro mean ± std | 1.8699 ± 0.0386 | 1.8956 ± 0.0427 |
| % inside pilots' extrapolated macro range | 99.9% (1 below, 0 above) | 99.8% (0 below, 2 above) |
| Mixtures with macro > 3 bpb | 0 | 0 |
| Mixtures with macro > 5 bpb | 0 | 0 |

Off-hull predictions stay bounded. The best measured pilot (`mix06`) has an extrapolated macro of 1.7567. The mixing law's optimum is predicted 0.0376 bpb below it, less than the mixing law's macro LOO RMSE (0.0422); LightGBM's optimum is predicted at 1.7690, not below that pilot. A claimed gain of a surrogate optimum over the best measured pilot is **not distinguishable from LOO error** at 60M scale, hence the 370M validation.

### Pilot mixtures ranked by predicted macro (top 12)

| ML rank | LGB rank | mix | tag | ML pred | LGB pred | measured curve-6 |
|--------:|---------:|-----|-----|--------:|---------:|-----------------:|
| 1 | 5 | Pilot 07 | C1-dclm60 | 1.7743 | 1.8214 | 2.0385 |
| 2 | 1 | Pilot 06 | C1-dclm55 | 1.7835 | 1.7698 | 1.9768 |
| 3 | 2 | Pilot 05 | C1-dclm50 | 1.7931 | 1.7945 | 2.0143 |
| 4 | 4 | Pilot 17 | C1 | 1.8024 | 1.8095 | 2.0518 |
| 5 | 6 | Pilot 23 | C1 | 1.8126 | 1.8288 | 2.0445 |
| 6 | 9 | Pilot 01 | base | 1.8189 | 1.8491 | 2.0633 |
| 7 | 3 | Pilot 22 | C1 | 1.8286 | 1.8064 | 2.0298 |
| 8 | 8 | Pilot 18 | C1 | 1.8289 | 1.8461 | 2.0260 |
| 9 | 7 | Pilot 21 | C1 | 1.8491 | 1.8334 | 2.0495 |
| 10 | 17 | Pilot 19 | C1 | 1.8578 | 1.9045 | 2.0862 |
| 11 | 12 | Pilot 12 | C1 | 1.8708 | 1.8877 | 2.0868 |
| 12 | 13 | Pilot 24 | C1 | 1.8770 | 1.8904 | 2.0947 |

### Earlier fit versus refit

The earlier fit used pilots of 1451 steps scored on the first 128 items of each validation split; the refit uses the 1440-step reruns scored on every test item. Both fit the same two surrogates to the same 24 mixtures. Optima under the same constraints (1% floor, 30% Wikipedia cap):

| Surrogate | Fit | pred. macro | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MixLaw | earlier | 1.7984 | 0.556 | 0.010 | 0.010 | 0.092 | 0.023 | 0.010 | 0.300 |
| MixLaw | refit | 1.7191 | 0.940 | 0.010 | 0.010 | 0.010 | 0.010 | 0.010 | 0.010 |
| LightGBM | earlier | 1.8335 | 0.553 | 0.212 | 0.087 | 0.082 | 0.042 | 0.014 | 0.011 |
| LightGBM | refit | 1.7690 | 0.551 | 0.179 | 0.159 | 0.019 | 0.033 | 0.050 | 0.010 |

- The **MixLaw optimum moved by 0.77** in L1 distance (from a Wikipedia-capped mixture to 94% DCLM); the **LightGBM optimum moved by 0.21**, with DCLM at 0.55 in both.
- Leave-one-out error is similar: mixing-law macro LOO RMSE 0.0443 → 0.0422, LightGBM 0.0366 → 0.0395.
- The pilots' ranking changed: the Spearman correlation between the earlier and refit rankings of the 24 pilots is 0.78 by extrapolated macro and 0.84 by the measured macro at the end of the run. The best measured pilot changed from `mix07` (60% DCLM) to `mix06` (55% DCLM). Absolute losses are not comparable between the two fits: the eval items, split and run length differ.
- The 370M arms and their results above are unchanged; they remain the product of the earlier fit.

---

## Validated mixture weights (370M)

Recipe domain weights for the four 370M mixtures that were trained (`validation_mixtures_10b.json`):

| arm | source | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|-----|--------|---:|---:|---:|---:|---:|---:|---:|
| Olmo-mix-1124 | reference | 0.951 | 0.005 | 0.021 | 0.015 | 0.003 | 0.003 | 0.001 |
| Data Mixing Laws paper | pilot | 0.375 | 0.250 | 0.141 | 0.094 | 0.064 | 0.061 | 0.016 |
| MixLaw | mixing-law | 0.556 | 0.010 | 0.010 | 0.092 | 0.023 | 0.010 | 0.300 |
| LightGBM | lightgbm | 0.553 | 0.212 | 0.087 | 0.082 | 0.042 | 0.014 | 0.011 |

**What each validated arm manipulates**

- **Olmo-mix-1124 (control)** — hold mix fixed at the natural Olmo-mix-1124 corpus proportions (very DCLM-heavy); run at data seeds 42 and 69 and reported as their average.
- **Data Mixing Laws paper** — hold domain mix fixed at the Data Mixing Laws paper proportions (Pilot 01). That paper's published optimal split (CommonCrawl 0.1250, C4 0.2500, GitHub 0.1406, arXiv 0.2500, Books 0.0938, StackExchange 0.1250, Wikipedia 0.0156) is mapped onto our domains by semantic category: CommonCrawl + C4 → DCLM, GitHub → StarCoder, Books → pes2o, StackExchange → OpenWebMath (0.0635) + Algebraic Stack (0.0615), arXiv and Wikipedia directly.
- **MixLaw** — train the 1%-floor MixLaw optimum. Method from [Ye et al., Data Mixing Laws](https://arxiv.org/abs/2403.16952).
- **LightGBM** — train the 1%-floor LightGBM optimum. Method from [Liu et al., RegMix](https://arxiv.org/abs/2407.01492) (regression / tree surrogate over mixture probes).

---

## 370M evaluation and uncertainty

Task-loss is evaluated on the shared 20-label suite every 125 of the 2384 steps (steps 0, 125, …, 2250) plus the final step 2384, 20 points per run. Final performance uses a **power-law residual bootstrap**:

1. Fit $y = a + b / \mathrm{step}^{\alpha}$ on all eval points with step ≥ 1000 ($\alpha$ on a grid in $[0.05, 3]$).
2. Take the fitted value at the final step as the point estimate.
3. Alpha-free residual bootstrap (200,000 draws, $\alpha$ re-selected on every draw) for a 95% CI on that fitted final.
4. Pairwise $\Delta = \mathrm{arm} - \mathrm{control}$ from independent bootstraps (control = the two-seed Olmo-mix-1124 average); two-sided $p$ is floored at the bootstrap resolution, $5\times10^{-6}$. Positive $\Delta$ means the arm finishes worse than the control. The results JSON also carries a one-sided $p$ (`p_one_sided`: the fraction of draws of $\Delta$ that are $\ge 0$, half the two-sided $p$ when the arm is better), which is the convention the paper quotes for "arm beats reference", with an em dash for an arm that did not beat it; the seed-variance estimate stays two-sided. The $p$ values in the tables below are two-sided unless marked one-sided.

Each arm is a single run, so these intervals cover power-law-fit uncertainty within a run only. The two control seeds are the only run-to-run estimate.

---

## 370M results

### Fitted final macro task-loss (bpb)

Regenerate with [`fit_and_bootstrap_370m.py`](fit_and_bootstrap_370m.py) from the
committed curves in [`skill_dag_370m_wandb_curves.json`](skill_dag_370m_wandb_curves.json)
(written from W&B by [`pull_370m_curves.py`](pull_370m_curves.py));
full output in [`skill_dag_370m_bootstrap_results.json`](skill_dag_370m_bootstrap_results.json).
The control is the **average of the two Olmo-mix-1124 data seeds** (42, 69),
whose bootstrap distribution is the element-by-element mean of the two seeds' own
alpha-free bootstrap distributions. Every arm is resampled from its own
independent random stream, so between-arm intervals are not coupled through a
shared generator.

| Arm | Fitted final | Observed | 95% CI | Δ vs control average | $p$ |
|-----|-------------:|---------:|--------|---------------------|----:|
| Olmo-mix-1124 seed 42 | 1.6200 | 1.6223 | [1.6133, 1.6270] | — | — |
| Olmo-mix-1124 seed 69 | 1.6095 | 1.6029 | [1.6040, 1.6139] | — | — |
| **Olmo-mix-1124 average (control)** | **1.6148** | 1.6126 | [1.6104, 1.6190] | — | — |
| LightGBM | 1.6238 | 1.6248 | [1.6190, 1.6291] | +0.0091 [+0.0026, +0.0158] | 0.0048 |
| Data Mixing Laws paper | 1.6411 | 1.6409 | [1.6363, 1.6454] | +0.0264 [+0.0200, +0.0325] | $< 5\times10^{-6}$ |
| MixLaw | 1.6433 | 1.6434 | [1.6404, 1.6463] | +0.0286 [+0.0234, +0.0338] | $< 5\times10^{-6}$ |

Lower is better. All three optimized static mixtures finish **behind** the
Olmo-mix-1124 control. The two fitted-law arms were both minimized under a 1% per-domain floor and the 30%
Wikipedia cap (one rule for every surrogate), and neither beats the control.

**Seed-variance reference.** The two control seeds differ by 0.0105 bpb
(seed 42 minus seed 69, 95% CI [+0.0023, +0.0193], $p = 0.0095$). They differ
only in data seed, but even identically configured runs on this stack are not
bit-reproducible, so the difference reflects data order plus GPU nondeterminism and
is the only run-to-run estimate available. The LightGBM gap (0.0091 bpb, 0.9× the
seed difference) is of that size; the Data Mixing Laws and MixLaw gaps are 2.5× and 2.7× it.
With one run per arm, the LightGBM shortfall in particular cannot be separated from
run-to-run variation with confidence.

### Targeted vs never-targeted labels

The six fitted skills supply 12 of the 20 labels (validation and test splits of
ARC Easy, ARC Challenge and the four MMLU groups); the other 8 (BoolQ, CSQA,
HellaSwag, SocialIQA, PIQA and WinoGrande validation, and OpenBookQA validation and
test) were never optimized. Same fit and bootstrap as above, on each subset's
per-step mean ([`heldout_label_bootstrap.py`](heldout_label_bootstrap.py) from
[`heldout_label_curves.json`](heldout_label_curves.json)); difference from the
control average, positive = worse:

| Labels | MixLaw | LightGBM | Control seed spread (69 − 42) |
|---|---|---|---|
| 12 targeted | +0.0325 [+0.0263, +0.0389], $p < 5\times10^{-6}$ | +0.0117 [+0.0041, +0.0194], $p = 0.0013$ | +0.0103 [+0.0005, +0.0189], $p = 0.040$ |
| 8 never-targeted | +0.0218 [+0.0141, +0.0288], $p < 5\times10^{-6}$ | +0.0043 [-0.0040, +0.0127], $p = 0.31$ | -0.0438 [-0.0574, -0.0317], $p < 5\times10^{-6}$ |

Neither mixture improves the labels it was fitted to: MixLaw and LightGBM both finish
behind the control on the 12 targeted labels, so the overall shortfall is not a
trade of targeted gains for never-targeted losses. On the 8 never-targeted labels
MixLaw is behind and LightGBM is within noise of the control, but those labels are
where the two control seeds disagree most (0.0438 bpb on the subset average).
Per label ([`heldout_perlabel_bootstrap.py`](heldout_perlabel_bootstrap.py)),
MixLaw is significantly worse than the control on 7 of the 8 never-targeted labels and
LightGBM on 6 (all but WinoGrande and BoolQ). BoolQ is the only label where either
is significantly better (MixLaw -0.2253 bpb, LightGBM -0.1580), against a 0.3578 bpb
seed spread on that label, so that difference cannot be separated from run-to-run
variation. (The same subsets and per-label comparison for the two Skill-It arms, against both the control and the LightGBM mixture they start from, are in [`dynamic_arm_labels.py`](dynamic_arm_labels.py); [`mmlu_stem_combined.py`](mmlu_stem_combined.py) averages the MMLU STEM validation and test labels for the static arms.) HellaSwag is worse under both mixtures
(MixLaw +0.0274, LightGBM +0.0280, both $p < 5\times10^{-6}$) against a seed spread
of 0.0053 bpb; PIQA is worse under both (+0.0382 and +0.0415, $p < 5\times10^{-6}$)
against a seed spread of 0.0282.

### Compute

[`plot_figure_iii_compute_savings.py`](plot_figure_iii_compute_savings.py) draws
Figure III, which compares the Skill-It probe arm against the LightGBM static run
(both Skill-It arms' numbers are in [`../skillit/README.md`](../skillit/README.md#compute-savings-figure-iii)).
It also asks how many steps the MixLaw mixture and the control need to reach each
other's final loss, using the same fitted power laws and paired bootstrap draws
(results under `mixlaw_vs_control` in
[`compute_savings_results.json`](compute_savings_results.json)). The MixLaw mixture
**never reaches the control's final loss** (1.6148): its fitted asymptote is 1.6275,
above that target, and none of the 200,000 bootstrap draws crosses it within the
run. Conversely, the control's fitted curve reaches the MixLaw mixture's final loss
(1.6433) at step 1755 (95% CI [1699, 1813]), 629 steps before the end of training.
This assumes each fitted power law holds up to the end of its LR schedule. The
pilot cost does not depend on the arm: the earlier 24 pilots (1451 steps, the ones behind this figure and
`compute_savings_results.json`) took $4.17\times10^{18}$
FLOPs ($1.74\times10^{17}$ per 60M run), about 15.8% of one $2.63\times10^{19}$-FLOP
370M arm, and the 1440-step reruns take $4.14\times10^{18}$ ($1.72\times10^{17}$ each, 15.7%); with no training-compute saving to set against it, it is overhead here.

### Takeaways

1. **No optimized static mixture beats the control.** The Data Mixing Laws paper
   mixture, the MixLaw 1%-floor optimum and the LightGBM 1%-floor optimum finish
   0.0264, 0.0286 and 0.0091 bpb behind the Olmo-mix-1124 average, respectively.
2. **The size of the gaps relative to run-to-run noise differs.** The two control
   seeds differ by 0.0105 bpb. The LightGBM gap is of that size ($p = 0.0048$); the
   Data Mixing Laws and MixLaw gaps ($p < 5\times10^{-6}$) are well above it. These are
   single runs, and the bootstrap intervals do not include run-to-run variation.
3. **The shortfall is not a targeted-vs-held-out trade.** Neither fitted mixture
   improves the 12 labels it was fitted to, and MixLaw is also behind on the 8
   never-targeted labels.
4. **No compute saving.** MixLaw never reaches the control's final loss within the
   run; the control reaches MixLaw's final loss 629 steps before the end.
5. **Cost.** Five full static arms (the control at two seeds) $\approx 1.32\times10^{20}$
   FLOPs on 4×L40S, plus $\approx 4.2\times10^{18}$ FLOPs for the 60M pilot grid ($4.17\times10^{18}$ for the earlier grid that picked these arms, $4.14\times10^{18}$ for the rerun).

---

## Conclusions

At this scale and budget, **neither mixing-law nor LightGBM optimization over the
60M-proxy pilots produced a mixture that beats the natural Olmo-mix-1124 proportions**
under the matched one-epoch 370M budget: the MixLaw optimum finishes clearly behind
the two-seed control, as does the Data Mixing Laws paper mixture, and the LightGBM
optimum finishes behind it by a margin comparable to the control-seed difference.
Two features of the design may matter and were not isolated here: the surrogates'
predicted gains over the best measured pilot are not distinguishable from their
leave-one-out error at 60M scale (see Random-simplex plausibility; the refit gives the same picture), and the MixLaw
optimum trained at 370M, from the earlier fit, puts Wikipedia at the 30% cap, beyond the 12.2% maximum across the pilots (the refit MixLaw optimum instead sits at 94% DCLM).
With one run per arm and a two-seed run-to-run estimate, small differences should be read with
that uncertainty in mind. Dynamic reweighting starting from the LightGBM mixture is
reported in [`../skillit/README.md`](../skillit/README.md#results).

Benchmark contamination between the shared 127B-token reservoir and the
evaluation suite, and its projection onto each arm's mixture, is audited in
[contamination/](contamination/). The MixLaw mixture has 2.11× the control's
contaminated exposure, LightGBM 0.79× and the Data Mixing Laws mixture 0.69×; the
measured losses do not follow that ordering.
