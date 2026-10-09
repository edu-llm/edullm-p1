# Skill-It (Mixing Laws Dataset / OLMoHQ × OLMo-2 370M)

**Question.** Can Skill-It domain reweighting — driven by an offline probe adjacency or by online mixing-law derivatives — improve macro task-loss over the static mixture both arms start from (the 1%-floor LightGBM optimum) under a matched one-epoch budget?

**Answer.** No. Neither dynamic arm beats the Olmo-mix-1124 control
(average of data seeds 42 and 69): the offline probe ends 0.0099 bpb worse
(95% CI [+0.0039, +0.0161], two-sided $p = 0.0010$) and the online derivative
0.0019 bpb worse (CI [-0.0050, +0.0099], $p = 0.656$). The static LightGBM
mixture they start from itself finishes 0.0091 bpb behind the control average
($p = 0.0048$). Against that starting mixture the offline probe arm is
indistinguishable (0.0009 bpb worse, CI [-0.0059, +0.0075], $p = 0.79$); the
online derivative arm finishes 0.0072 bpb better (CI [-0.0148, +0.0012],
two-sided $p = 0.087$, one-sided $p = 0.044$), inside the 0.0105 bpb difference
between the two control seeds. All arms are single runs, so the bootstrap
intervals cover power-law-fit uncertainty within a run only, and the two control
seeds are the only run-to-run estimate.

---

## Setup

| Knob | Value |
|------|-------|
| Architecture | OLMo-2 370M (full attention), shared 370M contract |
| Train stream | Domain-stratified sampling over 7 OLMoHQ domains with **time-varying** mixture weights |
| Domains | dclm, arxiv, starcoder, pes2o, open-web-math, algebraic-stack, wiki |
| Data | The ~127B-token reservoir `pretrain/olmo-127b` v1, shared with MixLaw; rebuild it byte for byte from the public Olmo-mix-1124 files with [`datasets/manifests/olmo-127b-v1/`](../../../datasets/manifests/olmo-127b-v1/README.md) (`rebuild.py`) |
| Global batch / seq / LR | 4,194,304 / 2048 / \(4\times10^{-4}\) cosine (warmup 24, \(\alpha_f=0.1\)) |
| Full-run budget | ~2384 steps ≈ one epoch |
| Data seed | 42 for both Skill-It arms and the LightGBM static run; the Olmo-mix-1124 control uses 42 and 69 |
| Hardware / trainer | FarmShare 4×L40S, the vendored Skill-It trainer ([`olmo_core_skillit/`](olmo_core_skillit/)), for every run compared below; runs differ only in mixture (and, for the second control, data seed) |
| FLOPs / full arm | \(2.63\times10^{19}\) (\(6ND\) plus the PaLM attention term, as for the probes below; measured from W&B) |
| Skill-It update | \(\eta=0.2\), \(w=1\); five mid-run updates |
| Update schedule | steps 500, 875, 1250, 1625, 2000 |
| Primary metric | Macro mean CE bits-per-byte over 20 OLMES-style labels, every 125 steps, plus the final step 2384 |

Unlike MixLaw (fixed weights for the whole run), Skill-It **reweights domains mid-training**. Between updates the sampler holds the current mixture fixed; at each update step it updates the domain probabilities from an adjacency \(A\) and the current per-family losses \(L\).

### Skill-It update (shared by all arms)

\[
p_i(t{+}1) \;\propto\; p_i(t)\,\exp\!\Big(\eta\, w \sum_j A_{ij} L_j\Big),\qquad \eta=0.2,\; w=1
\]

then renormalize \(p\) onto the simplex. This is the **multiplicative-weights**
rule of Chen et al. (their Eq. 4): the update *rescales the current mixture*, it
does not rebuild it from \(A L\) alone. A domain whose \(A\) row is all zeros
therefore keeps its existing share (scaled by the common normalizer) rather than
collapsing to parity with every other zero-row domain — which is exactly what the
logged trajectories below show. Intuition: domains that the adjacency says “help”
high-loss task families get more mass. The two arms differ only in **how \(A\) is
built**; both start from the same mixture, the 1%-floor LightGBM optimum.

### Offline probe matrix

1. Train **8** DataDecide-60M probes: the 7 one-hot probes (100% of each domain in turn) and one probe at the LightGBM starting mixture (5 tokens/param → 1451 steps / ~285M tokens, sized against DataDecide's original \(N=57.1\mathrm{M}\) non-embedding count; data seed 6198, as for the 24 pilots). All eight use the same training setup (micro-batch 4 × 24 accumulation, global batch 196,608) and are submitted together by [`submit_skillit_probes.sh`](submit_skillit_probes.sh).
2. Fit Chinchilla step-laws on in-run curves (evals every 120 of the 1451 steps on the first 128 items of each family's validation split, as 16 batches of 8: the same 128 items the 24 pilots are scored on, so every probe is read on the same subsample); extrapolate to tpp=20 (step 5806). The curves are rebuilt from each probe's trainer log by [`import_probe_logs.py`](import_probe_logs.py).
3. Build
   \[
   A_{ij} = \max\!\big(0,\; L_j(r_{\mathrm{LGB}}) - L_j(i)\big)
   \]
   where \(L_j(i)\) is family \(j\)’s extrapolated loss after training on 100% domain \(i\), and \(L_j(r_{\mathrm{LGB}})\) is the extrapolated loss of the probe trained on the LightGBM starting mixture \(r_{\mathrm{LGB}}\) (`LGB-min1pct`), also at the Chinchilla-style budget ([`build_adjacency.py --reference-run probe_lgb_start`](build_adjacency.py)). Positive \(A_{ij}\) means domain \(i\) alone beat the starting mixture on family \(j\).

**Probe FLOPs.** The 6ND estimate of Kaplan et al. (2020) plus the attention term of Chowdhery et al. (2023, PaLM),
\(C \approx 6 N_{\text{non-emb}} D + 12\,n_{\text{layers}}\,s\,d_{\text{model}}\,D\),
at the trained model's \(N_{\text{non-emb}} = 76{,}296{,}576\):
\(\approx 1.74\times10^{17}\) per probe → **\(\approx 1.39\times10^{18}\)** for all 8.

**Offline \(A\) used by the Offline probe arm** (rows = domains, columns = task families; Chinchilla step 5806):

| domain \\ family | arc_challenge | arc_easy | mmlu_humanities | mmlu_other | mmlu_social_sciences | mmlu_stem |
|------------------|--------------:|---------:|----------------:|-----------:|---------------------:|----------:|
| dclm | 0.000 | 0.077 | 0.000 | 0.000 | 0.000 | 0.000 |
| arxiv | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| starcoder | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| pes2o | 0.000 | 0.161 | 0.000 | 0.000 | 0.000 | 0.065 |
| open-web-math | 0.000 | 0.240 | 0.000 | 0.009 | 0.000 | 0.212 |
| algebraic-stack | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| wiki | 0.000 | 0.046 | 0.072 | 0.000 | 0.000 | 0.000 |

The rows for arXiv, StarCoder and Algebraic Stack are all zeros, and so are the columns for ARC Challenge and MMLU Social Sciences: no single domain beats the probe trained on the LightGBM starting mixture on those skills at Chinchilla scale. OpenWebMath and pes2o carry most of the edges. Density: 8 of 42 cells (19%) are nonzero.

### Online mixing-law derivative

Reuse the parametric MixLaw surrogate per family \(j\):

\[
L_j(r) = c_j + k_j \exp\!\Big(\sum_i t_{ij} r_i\Big)
\]

Skill-It-compatible adjacency at the **current** weights \(r\):

\[
A_{ij} = \max\!\big(0,\; -(dL_j/dr_i)\big) = \max\!\big(0,\; -t_{ij}(L_j(r)-c_j)\big)
\]

So \(A\) **changes every update** as \(r\) and predicted \(L(r)\) move. Fitted \(t_{ij}\) / \(c_j\) / \(k_j\) are those from the MixLaw parametric fit.

**This adjacency depends on the fit's gauge.** On the simplex \(\sum_i r_i = 1\), so
replacing \(t_{ij} \to t_{ij} + q_j\) for every domain \(i\) and \(k_j \to k_j e^{-q_j}\)
leaves \(L_j(r)\) unchanged for every mixture but moves every \(A_{ij}\), including
which ones are zero. The fitted \(t\) values are pinned only by the fit's
regularization, so the derivative arm's edges inherit that choice. The gauge-invariant
alternative is the benefit of moving mass from \(r\) toward domain \(i\),
\(\max\!\big(0,\;(\sum_q r_q t_{qj} - t_{ij})(L_j(r)-c_j)\big)\), available as
`online_A_from_fit(..., gauge_invariant=True)` in [`skillit_math.py`](skillit_math.py).
The reported run used the default (historical) form, `derivative_a` in the vendored
[`olmo_core_skillit/skillit_math.py`](olmo_core_skillit/skillit_math.py).

#### Which \(r\) the derivative matrix is evaluated at

Because \(A\) depends on \(r\), a single printed derivative matrix is a snapshot and
the reference point has to be stated. Both matrices here refer to the same mixture,
`LGB-min1pct`, the LightGBM-optimized mixture both arms start from: the offline
matrix compares each one-hot probe with the probe trained on that mixture, and the
derivative matrix is the one the derivative arm used at its first update (step 500).
Run [`compare_offline_online_A.py`](compare_offline_online_A.py) to regenerate it.

| Derivative matrix, evaluated at `LGB-min1pct` | Density | Pearson \(r\) vs offline probe \(A\) | Edge-presence disagreements |
|---|---:|---:|---:|
| vs the offline matrix above | 13/42 (31%) | **0.454** | 13/42 |

The two matrices have an edge in the same cell in only 4 cells (pes2o → MMLU STEM,
OpenWebMath → ARC Easy, Wikipedia → ARC Easy and MMLU Humanities). Both have
all-zero rows for StarCoder, arXiv and Algebraic Stack. The derivative matrix gives
Wikipedia an edge on all six skills and DCLM on four, where the probe matrix gives
Wikipedia two and DCLM one.

Figure IV is generated by
[`plot_adjacency_comparison.py`](plot_adjacency_comparison.py); the committed output is
[`figures/adjacency_comparison.png`](figures/adjacency_comparison.png).

#### Robustness check (Appendix E)

[`appendix_e_robustness.py`](appendix_e_robustness.py) reproduces the paper's
robustness check on the derivative construction and writes
[`artifacts/appendix_e_robustness.json`](artifacts/appendix_e_robustness.json):

- **Derivative along the simplex.** The gauge-invariant form above, the derivative
  along \(r(\varepsilon) = (1-\varepsilon)r + \varepsilon e_i\)
  (`online_A_from_fit(..., gauge_invariant=True)`), has \(r = 0.981\)–\(0.985\) to
  the form the arm used at the domain weights of each of its five updates, i.e.
  \(r \ge 0.98\) at every update (`simplex_derivative.per_update`).

### Arms actually run

Skill-It reweighting follows [Chen et al., Skill-It!](https://arxiv.org/abs/2307.14430). Online derivative \(A\) additionally uses the MixLaw parametric form from [Ye et al., Data Mixing Laws](https://arxiv.org/abs/2403.16952).

Two arms were trained, both starting from the **LightGBM-optimized mixture**
(`LGB-min1pct`, id 27 in
[`../mixlaw/validation_mixtures_10b.json`](../mixlaw/validation_mixtures_10b.json))
— confirmed by each run's own step-0 logged weights
(`skillit_updates.jsonl`), which match that published weight vector to full
float precision.

| Arm | Manipulation | FLOPs |
|-----|--------------|------:|
| Offline probe | Start at the LightGBM-optimized mixture; at each update apply Skill-It with the **fixed** offline \(A\) above and current task losses | \(2.63\times10^{19}\) |
| Online derivative | Start at the LightGBM-optimized mixture; at each update **recompute** \(A(r)\) from MixLaw derivatives, then Skill-It-update | \(2.63\times10^{19}\) |
| **Total** | | **\(5.26\times10^{19}\)** |

Comparisons use the **Olmo-mix-1124 seed average** as the control (the corpus's
natural weighting), and additionally report each arm against the **LightGBM
static** mixture it starts from: a single run of the same trainer with dynamic
reweighting disabled. All of these references are fixed-weight full runs, not
extra Skill-It trains. Every run, Skill-It arms and references alike, ran on
FarmShare 4×L40S (see Training-code provenance below) with the same trainer,
initialization (step-0 task loss 4.4728 bpb in all of them), batch, schedule and
data seed 42; the only other difference is the second control, which uses data
seed 69.

### Domain weights after each update

Logged \(p\) (post-update domain mixture) read directly from each run's own
`skillit_updates.jsonl` progress log at step 0 and each Skill-It update.
Weights sum to 1. Step 0 for both arms is `LGB-min1pct`, the
LightGBM-optimized mixture (see Arms actually run above).

**Offline probe**

| step | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|-----:|-----:|------:|----------:|------:|--------------:|----------------:|-----:|
| 0 | 0.553 | 0.212 | 0.087 | 0.082 | 0.042 | 0.014 | 0.011 |
| 500 | 0.550 | 0.204 | 0.084 | 0.087 | 0.050 | 0.013 | 0.011 |
| 875 | 0.546 | 0.198 | 0.081 | 0.092 | 0.059 | 0.013 | 0.011 |
| 1250 | 0.540 | 0.191 | 0.079 | 0.097 | 0.070 | 0.012 | 0.011 |
| 1625 | 0.534 | 0.184 | 0.076 | 0.102 | 0.081 | 0.012 | 0.011 |
| 2000 | 0.526 | 0.177 | 0.073 | 0.107 | 0.095 | 0.011 | 0.011 |

**Online derivative**

| step | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|-----:|-----:|------:|----------:|------:|--------------:|----------------:|-----:|
| 0 | 0.553 | 0.212 | 0.087 | 0.082 | 0.042 | 0.014 | 0.011 |
| 500 | 0.556 | 0.200 | 0.082 | 0.086 | 0.047 | 0.013 | 0.016 |
| 875 | 0.557 | 0.189 | 0.078 | 0.091 | 0.052 | 0.012 | 0.021 |
| 1250 | 0.556 | 0.178 | 0.073 | 0.095 | 0.057 | 0.011 | 0.028 |
| 1625 | 0.553 | 0.168 | 0.069 | 0.099 | 0.062 | 0.011 | 0.037 |
| 2000 | 0.549 | 0.159 | 0.065 | 0.103 | 0.067 | 0.010 | 0.047 |

Offline probe's sparse adjacency moves the mixture only slowly: `dclm` drifts down from
0.553 to 0.526 and `arxiv`/`starcoder` fall, while `open-web-math` (0.042 to 0.095) and
`pes2o` (0.082 to 0.107) rise and `wiki` stays at 0.011. Online derivative's recomputed
adjacency instead redistributes weight from `arxiv`/`starcoder` onto `wiki` (about 4×),
`open-web-math` and `pes2o` while leaving `dclm` close to its starting share. See
[`contamination/README.md`](contamination/README.md#reading-these-together)
for how this drives each arm's contaminated exposure.

---

## Training-code provenance

Where each run compared in this README ran, as recorded in its W&B run metadata
(`eduLLM/skillit`, `eduLLM/mixlaw-new`); all on FarmShare 4×L40S:

| Run | Slurm job | W&B run | Data seed |
|-----|----------:|---------|----------:|
| Offline probe (matrix rebuilt against the LightGBM-start probe) | 1771665 | `eduLLM/skillit/iy441nc7` | 42 |
| Online derivative | 1728144 | `eduLLM/skillit/c0844ce36f24d6773c7f45cb31d810f4` | 42 |
| LightGBM static | 1744338 | `eduLLM/mixlaw-new/zgmte13g` | 42 |
| Olmo-mix-1124 control | 1745704 | `eduLLM/mixlaw-new/i9z1vtbt` | 42 |
| Olmo-mix-1124 control | 1760339 | `eduLLM/mixlaw-new/3vbmxzmg` | 69 |

The online derivative arm ran the vendored `entrypoint.py` from an uncommitted copy of
OLMo-core's Skill-It `.edullm/` code on FarmShare scratch, launched with
`train_no_aws.sbatch`. The offline probe arm ran the same entrypoint from a copy of the
bundle patched for the static arms (below), with only the offline matrix and its two
checksum pins changed by
[`patch_probe_matrix.py`](olmo_core_skillit/farmshare/patch_probe_matrix.py);
[`stage_probe_rerun.sh`](olmo_core_skillit/farmshare/stage_probe_rerun.sh),
[`farmshare_probe_rerun_l40s.sbatch`](olmo_core_skillit/farmshare/farmshare_probe_rerun_l40s.sbatch)
and
[`farmshare_preflight_submit_probe_rerun.sh`](olmo_core_skillit/farmshare/farmshare_preflight_submit_probe_rerun.sh)
stage and submit it (one 4×L40S node, `--arm-index 0`). The Skill-It code is vendored in
[`olmo_core_skillit/`](olmo_core_skillit/); its
[`PROVENANCE.md`](olmo_core_skillit/PROVENANCE.md) records where it came from,
how it differs from the nearest upstream commit (`f2ded0b6` on
`edullm/skillit-370m`), and what was left out.

The LightGBM static run (Slurm job 1744338, run name
`static-lgbm-min1pct-farmshare-1744338`) ran the same entrypoint from a
separate copy of that code with a third arm added: arm index 2, `a_mode`
`static`, which holds the domain weights at `LGB-min1pct` for the whole run, so
W&B logs only the step-0 Skill-It snapshot. Its step-0 macro bpb, 4.4728, is
identical to both Skill-It arms'. W&B marks the run failed, but only because,
after all 2384 steps had trained and the step-2384 eval had been logged,
OLMo-core's W&B callback called `wandb.finish(exit_code=..., quiet=True)`, which
the installed wandb rejects. The three files that set it up and launched it were added to
[`olmo_core_skillit/farmshare/`](olmo_core_skillit/farmshare/) after vendoring,
and the vendored files were left unmodified (see
[`PROVENANCE.md`](olmo_core_skillit/PROVENANCE.md#added-after-vendoring)):

| File | Role |
|------|------|
| `patch_legacy_static_lgbm_arm.py` | Adds the static arm to a copy of the vendored `.edullm/` bundle (recipe, `skillit_math.py`, `skillit_controller.py`, `train_skillit_370m.py`), refusing to patch if any target text differs |
| `farmshare_static_lgbm_l40s.sbatch` | The Slurm job: one 4×L40S node, reads only the pre-staged local data manifest, runs the vendored `entrypoint.py --arm-index 2` |
| `farmshare_preflight_submit_static_lgbm.sh` | Checks the staged run folder, venv and W&B/HF session files, then submits the job |

The two Olmo-mix-1124 control runs use the same entrypoint on a separate copy of
the bundle with static arms added by
[`patch_static_validation_arms.py`](olmo_core_skillit/farmshare/patch_static_validation_arms.py)
and launched with
[`farmshare_static_validation_l40s.sbatch`](olmo_core_skillit/farmshare/farmshare_static_validation_l40s.sbatch);
the other MixLaw static runs are described in
[`../mixlaw/README.md`](../mixlaw/README.md#training-code-provenance).

## Evaluation and uncertainty

Same as MixLaw: evals every 125 of the 2384 steps (plus the final step); power law \(y = a + b/\mathrm{step}^{\alpha}\) on steps ≥ 1000;
fitted final as center; **alpha-free** residual bootstrap (200,000 draws, \(\alpha\)
re-selected on every draw) for the 95% CI. Reproduce with
[`../mixlaw/fit_and_bootstrap_370m.py`](../mixlaw/fit_and_bootstrap_370m.py) from
the committed curves in
[`../mixlaw/skill_dag_370m_wandb_curves.json`](../mixlaw/skill_dag_370m_wandb_curves.json).

---

## Results

Numbers below are for the two Skill-It runs (both starting from the LightGBM
optimum), the LightGBM static run they start from, and the two Olmo-mix-1124
control runs (data seeds 42 and 69) with their average. All are single runs on
the same FarmShare 4×L40S setup.

### Fitted final macro task-loss (bpb)

| Arm | Fitted final | Observed | 95% CI | vs control average |
|-----|-------------:|---------:|--------|--------------------|
| Olmo-mix-1124 control, seed 42 | 1.6200 | 1.6223 | [1.6133, 1.6270] | — |
| Olmo-mix-1124 control, seed 69 | 1.6095 | 1.6029 | [1.6040, 1.6139] | — |
| Olmo-mix-1124 average (control) | 1.6148 | 1.6126 | [1.6104, 1.6190] | — |
| LightGBM static | 1.6238 | 1.6248 | [1.6190, 1.6291] | \(p = 0.0048\) (worse) |
| Offline probe | 1.6247 | 1.6246 | [1.6204, 1.6292] | \(p = 0.0010\) (worse) |
| Online derivative | 1.6166 | 1.6216 | [1.6114, 1.6235] | \(p = 0.656\) |

Lower is better. The probe arm is 0.0099 bpb worse than the control average (95% CI
[+0.0039, +0.0161]); the derivative arm is 0.0019 bpb worse (95% CI [-0.0050, +0.0099])
and indistinguishable from it. The LightGBM static mixture they start from is 0.0091 bpb
worse than the control average (95% CI [+0.0026, +0.0158], \(p = 0.0048\)). Against
that static run (Δ = first minus second, so negative means the first is better; \(p\)
two-sided, with the one-sided \(p\) for "beats the reference" in the last column):

| Comparison | Δ bpb | 95% CI | \(p\) (two-sided) | \(p\) (one-sided) |
|------------|------:|--------|------:|------:|
| Offline probe − LightGBM static | +0.0009 | [-0.0059, +0.0075] | 0.79 | — (worse) |
| Online derivative − LightGBM static | -0.0072 | [-0.0148, +0.0012] | 0.087 | 0.044 |

**Run-to-run variation.** The two Olmo-mix-1124 controls differ only in data
seed (42 vs 69) and land 0.0105 bpb apart (95% CI [+0.0023, +0.0193],
\(p = 0.0095\)). Even identically configured runs on this stack are not
bit-reproducible: the probe and LightGBM static runs, which share a
data seed and every setting up to the first update at step 500, report identical gradient
norms through step 20 and have already diverged by step 30 (0.7326 vs 0.7325; 0.5282 vs
0.5061 at step 50), before any Skill-It update. The seed difference therefore reflects
data order plus GPU nondeterminism, and is the only run-to-run estimate
available. Each arm is a single run, and the bootstrap intervals above capture
only power-law-fit uncertainty within a run, not run-to-run variation. The
probe's +0.0009 bpb difference from the LightGBM static run is 0.09× the
control-seed difference; the derivative's 0.0072 bpb advantage is 0.7× it.

### Compute savings (Figure III)

[`../mixlaw/plot_figure_iii_compute_savings.py`](../mixlaw/plot_figure_iii_compute_savings.py)
asks how many steps each Skill-It arm needs to reach the LightGBM static run's
final fitted loss (1.6238), using the same fitted power laws and paired bootstrap
draws (results under `dynamic_vs_lightgbm` in
[`../mixlaw/compute_savings_results.json`](../mixlaw/compute_savings_results.json)).
Training FLOPs are linear in steps, so the step saving is also the FLOP saving as
a fraction of one \(2.63\times10^{19}\)-FLOP arm. The net saving charges each arm
for the 60M runs it depends on (\(1.74\times10^{17}\) FLOPs each): 8 for the probe
arm (the seven one-hot probes and the probe at the LightGBM starting mixture, 5.28%
of one arm) and 24 for the derivative arm (the MixLaw pilot grid its derivatives come
from, 15.85%).

| Arm | Reaches 1.6238 at step | Step saving | Net of overhead | P(net > 0) |
|-----|-----------------------:|------------:|----------------:|-----------:|
| Offline probe | 2420 [2187, 2879] (past the end of the run) | -1.5% [-20.8, 8.2] | -6.8% [-26.1, +3.0] | 0.099 |
| Online derivative | 2216 [2087, 2421] | 7.0% [-1.5, 12.5] | -8.8% [-17.4, -3.4] | 0.0003 |

Step intervals are the bootstrap 95% intervals around the point crossing; 39.7% of the
probe's draws and 95.6% of the derivative's cross within the run. Conversely,
LightGBM's fitted curve would reach the probe arm's final loss at 0.98× the run
(95% CI [0.88×, 1.24×]; 0.1% of draws never do) and the derivative's at 1.20×
([0.98×, 2.26×]). As in the MixLaw analysis, this reads each run's fitted curve at an
intermediate step, so it assumes the power law holds up to the end of its LR schedule.

### Takeaways

1. **No evidence that dynamic reweighting beats the control.** Against the
   Olmo-mix-1124 average the probe is 0.0099 bpb worse (two-sided \(p = 0.0010\))
   and the derivative 0.0019 bpb worse (\(p = 0.656\)).
2. **The static starting mixture finishes behind the control.** LightGBM static
   is 0.0091 bpb worse than the control average (95% CI [+0.0026, +0.0158],
   \(p = 0.0048\)).
3. **Neither arm clearly improves on its starting mixture.** The probe arm
   finishes 0.0009 bpb behind LightGBM static (95% CI [-0.0059, +0.0075]); the
   derivative arm 0.0072 bpb ahead (two-sided \(p = 0.087\), one-sided
   \(p = 0.044\)), 0.7× the 0.0105 bpb control-seed difference. Both
   comparisons are single runs against a single run.
4. **Neither arm saves compute against its starting mixture.** The probe arm does
   not reach the LightGBM static final loss within the run (-6.8% net of its 8
   probes); the derivative arm's 7.0% step saving does not cover its 24 pilots
   (-8.8% net).
5. **Run-to-run variation is of the same size as these gaps.** The control
   seeds differ by 0.0105 bpb (95% CI [+0.0023, +0.0193], \(p = 0.0095\)), and
   matched runs on this stack are not bit-reproducible.
6. **Cost.** Two Skill-It trains \(\approx 5.26\times10^{19}\) FLOPs on 4×L40S,
   plus \(2.63\times10^{19}\) for the LightGBM static run and
   \(\approx 1.39\times10^{18}\) FLOPs for the 8 60M probes.

---

## Conclusions

At this scale and budget, **there is no evidence that the Skill-It rule at
\(\eta = 0.2\), with either the offline probe or the mixing-law derivative
adjacency, improves on the Olmo-mix-1124 control.** The offline probe arm finishes
behind the control (+0.0099 bpb, two-sided \(p = 0.0010\)) and indistinguishable from the
LightGBM mixture it starts from; the derivative arm is within noise of the control and
0.0072 bpb ahead of its starting mixture, less than the control-seed difference. The
LightGBM mixture itself finishes behind the control, so neither form of reweighting
demonstrably recovers its deficit. With single runs and a two-seed run-to-run estimate,
these differences cannot be separated from run-to-run variation with confidence. Note
the design limit: both arms start *at* an optimized mixture and move away from it, so
this tests whether Skill-It can improve on a good mix, not whether it can rescue a bad
one.

Benchmark contamination between the shared 127B-token reservoir and the
evaluation suite is audited in [contamination/](contamination/), including
each arm's time-weighted exposure over its realized, time-varying mixture.
