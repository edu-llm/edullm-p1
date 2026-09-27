# Skill-It (Mixing Laws Dataset / OLMoHQ × OLMo-2 370M)

**Question.** Can Skill-It domain reweighting — driven by an offline probe adjacency or by online mixing-law derivatives — improve macro task-loss over the static mixture both arms start from (the 1%-floor LightGBM optimum) under a matched one-epoch budget?

**Answer.** There is **no reliable evidence** that it does. Both Skill-It arms
beat the Olmo-mix-1124 control (probe $p < 10^{-4}$, derivative $p = 0.004$).
Against the static LightGBM mixture they start from, the sign depends on which
run of that mixture is the reference. Against the original 8×A100 run, the
offline probe arm ends 0.0032 bpb worse ($p = 0.14$) and the online derivative
arm 0.0086 bpb worse ($p = 0.002$), but that comparison also carries a hardware
and initialization difference. Against a 4×L40S rerun matched to the dynamic
arms (same hardware, initialization, data seed 42 and training code, with
reweighting disabled), the probe arm ends 0.0126 bpb better ($p < 10^{-4}$) and
the derivative arm 0.0072 bpb better ($p = 0.087$). The two static runs differ
from each other by 0.0158 bpb ($p < 10^{-4}$), 3.6× the 0.0044 bpb difference
between the two control seeds and larger than either dynamic arm's advantage,
so the effect of dynamic reweighting cannot be separated from run-to-run
variation.

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
| Data seed | 42 (both arms, and the matched LightGBM static rerun) |
| FLOPs / full arm | \(2.63\times10^{19}\) (\(6ND\) plus the PaLM attention term, as for the probes below; measured from W&B) |
| Skill-It update | \(\eta=0.2\), \(w=1\); five mid-run updates |
| Update schedule | steps 500, 875, 1250, 1625, 2000 |
| Primary metric | Macro mean CE bits-per-byte over 20 OLMES-style labels, every 125 steps (step 2375 excluded) |

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

1. Train **7 one-hot** DataDecide-60M probes (100% of each domain in turn; 5 tokens/param → 1451 steps / ~285M tokens, sized against DataDecide's original \(N=57.1\mathrm{M}\) non-embedding count; data seed 6198, as for the 24 pilots).
2. Fit Chinchilla step-laws on in-run curves (evals every 120 of the 1451 steps on a fixed four-batch subsample of each family's validation split); extrapolate to tpp=20 (step 5806).
3. Build
   \[
   A_{ij} = \max\!\big(0,\; L_j(r_{\mathrm{DML}}) - L_j(i)\big)
   \]
   where \(L_j(i)\) is family \(j\)’s extrapolated loss after training on 100% domain \(i\), and \(L_j(r_{\mathrm{DML}})\) is the MixLaw fit's prediction for family \(j\) at the Data Mixing Laws paper mixture \(r_{\mathrm{DML}}\), also at the Chinchilla-style budget. Positive \(A_{ij}\) means domain \(i\) alone beat that prediction on family \(j\).

**Probe FLOPs.** The 6ND estimate of Kaplan et al. (2020) plus the attention term of Chowdhery et al. (2023, PaLM),
\(C \approx 6 N_{\text{non-emb}} D + 12\,n_{\text{layers}}\,s\,d_{\text{model}}\,D\),
at the trained model's \(N_{\text{non-emb}} = 76{,}296{,}576\):
\(\approx 1.74\times10^{17}\) per probe → **\(\approx 1.22\times10^{18}\)** for all 7.

**Offline \(A\) used by the Offline probe arm** (rows = domains, columns = task families; Chinchilla step 5806):

| domain \\ family | arc_challenge | arc_easy | mmlu_humanities | mmlu_other | mmlu_social_sciences | mmlu_stem |
|------------------|--------------:|---------:|----------------:|-----------:|---------------------:|----------:|
| dclm | 0.340 | 0.149 | 0.368 | 0.000 | 0.079 | 0.486 |
| arxiv | 0.006 | 0.000 | 0.000 | 0.000 | 0.000 | 0.340 |
| starcoder | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| pes2o | 0.252 | 0.084 | 0.000 | 0.000 | 0.000 | 0.450 |
| open-web-math | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.232 |
| algebraic-stack | 0.017 | 0.000 | 0.000 | 0.000 | 0.000 | 0.242 |
| wiki | 0.227 | 0.000 | 0.470 | 0.013 | 0.000 | 0.352 |

Starcoder’s row is all zeros (never beats the fit's prediction for the Data Mixing Laws paper mix on these families at Chinchilla scale). DCLM and wiki dominate many columns. Density: 17 of 42 cells (40%) are nonzero.

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

#### Which \(r\) a published derivative matrix is evaluated at

Because \(A\) depends on \(r\), a single printed derivative matrix is a snapshot and
the reference point has to be stated. Two are in play and they are not the same
matrix. Run
[`compare_offline_online_A.py`](compare_offline_online_A.py) to regenerate both.

| Evaluated at | Density | Pearson \(r\) vs offline probe \(A\) | Edge-presence disagreements |
|---|---:|---:|---:|
| \(r_{\mathrm{DML}}\) (`mix01`) | 13/42 (31%) | **0.171** | 10/42 |
| `LGB-min1pct` (the arm's own start) | 13/42 (31%) | **0.065** | 10/42 |

\(r_{\mathrm{DML}}\) is the like-for-like point, since the offline probe matrix is
also referenced to \(r_{\mathrm{DML}}\); `LGB-min1pct` is the matrix the derivative
arm actually used at its first update. At \(r_{\mathrm{DML}}\) the entries are
roughly twice as large (dclm → arc_challenge 0.154 vs 0.065). Density and the
10-of-42 edge-presence disagreement are the same at both points; only the
correlation and the magnitudes move.

Figure IV is generated by
[`plot_adjacency_comparison.py`](plot_adjacency_comparison.py) and its derivative
panel is evaluated at **`LGB-min1pct`**, which is the point the paper's body
describes and the point its \(r = 0.07\) is computed at; the committed output is
[`figures/adjacency_comparison.png`](figures/adjacency_comparison.png).

The derivative matrix's qualitative pattern is the same at either point:
StarCoder, Algebraic Stack and arXiv all-zero, OpenWebMath helping only
ARC Easy, pes2o trivial on both ARC skills, Wikipedia and DCLM broadly helpful,
31% density, 10 of 42 edge-presence disagreements with the probe matrix.

#### Robustness checks (Appendix E)

[`appendix_e_robustness.py`](appendix_e_robustness.py) reproduces the paper's two
robustness checks on these constructions and writes
[`artifacts/appendix_e_robustness.json`](artifacts/appendix_e_robustness.json):

- **Probe reference.** Rebuilding the probe matrix against the extrapolated losses
  of the trained Data Mixing Laws pilot (`mix01`), instead of the MixLaw fit's
  prediction for that mixture, gives Pearson \(r = 0.987\) to the matrix used, with
  2 of 42 cells differing in edge presence (`probe_reference.mix01_vs_committed`).
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
(`skillit-370m-probe-rerun-20260918-011120`,
`skillit-370m-deriv-20260916-124719`), which match that published weight vector
to full float precision.

| Arm | Manipulation | FLOPs |
|-----|--------------|------:|
| Offline probe | Start at the LightGBM-optimized mixture; at each update apply Skill-It with the **fixed** offline \(A\) above and current task losses | \(2.63\times10^{19}\) |
| Online derivative | Start at the LightGBM-optimized mixture; at each update **recompute** \(A(r)\) from MixLaw derivatives, then Skill-It-update | \(2.63\times10^{19}\) |
| **Total** | | **\(5.26\times10^{19}\)** |

Comparisons use the **Olmo-mix-1124 seed average** as the control (the corpus's
natural weighting), and additionally report each arm against two runs of the
**LightGBM static** mixture it starts from: the original 8×A100 MixLaw arm, and
a 4×L40S rerun with dynamic reweighting disabled that matches the Skill-It arms
in hardware, initialization, data seed (42) and training code. All of these
references are fixed-weight full runs, not extra Skill-It trains. Both Skill-It
arms and the rerun ran on FarmShare 4×L40S (see Training-code provenance below).

### Domain weights after each update

Logged \(p\) (post-update domain mixture) read directly from each run's own
`skillit_updates.jsonl` progress log at step 0 and each Skill-It update.
Weights sum to 1. Step 0 for both arms is `LGB-min1pct`, the
LightGBM-optimized mixture (see Arms actually run above).

**Offline probe**

| step | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|-----:|-----:|------:|----------:|------:|--------------:|----------------:|-----:|
| 0 | 0.553 | 0.212 | 0.087 | 0.082 | 0.042 | 0.014 | 0.011 |
| 500 | 0.646 | 0.168 | 0.057 | 0.077 | 0.031 | 0.010 | 0.011 |
| 875 | 0.720 | 0.131 | 0.037 | 0.071 | 0.023 | 0.008 | 0.011 |
| 1250 | 0.779 | 0.101 | 0.023 | 0.064 | 0.017 | 0.006 | 0.010 |
| 1625 | 0.827 | 0.077 | 0.015 | 0.057 | 0.012 | 0.004 | 0.009 |
| 2000 | 0.864 | 0.057 | 0.009 | 0.050 | 0.008 | 0.003 | 0.008 |

**Online derivative**

| step | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|-----:|-----:|------:|----------:|------:|--------------:|----------------:|-----:|
| 0 | 0.553 | 0.212 | 0.087 | 0.082 | 0.042 | 0.014 | 0.011 |
| 500 | 0.556 | 0.200 | 0.082 | 0.086 | 0.047 | 0.013 | 0.016 |
| 875 | 0.557 | 0.189 | 0.078 | 0.091 | 0.052 | 0.012 | 0.021 |
| 1250 | 0.556 | 0.178 | 0.073 | 0.095 | 0.057 | 0.011 | 0.028 |
| 1625 | 0.553 | 0.168 | 0.069 | 0.099 | 0.062 | 0.011 | 0.037 |
| 2000 | 0.549 | 0.159 | 0.065 | 0.103 | 0.067 | 0.010 | 0.047 |

Offline probe's fixed adjacency drives weight toward `dclm` monotonically at
every update; online derivative's recomputed adjacency instead redistributes
weight from `arxiv`/`starcoder` onto `wiki` (about 4×), `open-web-math` and
`pes2o` while leaving `dclm` close to its starting share. See
[`contamination/README.md`](contamination/README.md#reading-these-together)
for how this drives each arm's contaminated exposure.

---

## Training-code provenance

Where each Skill-It 370M run ran, as recorded in its W&B run metadata
(`eduLLM/skillit`):

| Arm | W&B run | Platform |
|-----|---------|----------|
| Offline probe | `87ad0201c4b5781a3df50d7bb394776c` | FarmShare, 4×L40S |
| Online derivative | `c0844ce36f24d6773c7f45cb31d810f4` | FarmShare, 4×L40S |
| LightGBM static rerun | `zgmte13g` | FarmShare, 4×L40S |

Both Skill-It arms ran `.edullm/runpod/entrypoint.py` from an uncommitted copy of OLMo-core's
Skill-It `.edullm/` code on FarmShare scratch, launched with
`train_no_aws.sbatch`. That code is vendored in
[`olmo_core_skillit/`](olmo_core_skillit/); its
[`PROVENANCE.md`](olmo_core_skillit/PROVENANCE.md) records where it came from,
how it differs from the nearest upstream commit (`f2ded0b6` on
`edullm/skillit-370m`), and what was left out.

The LightGBM static rerun (Slurm job 1744338, run name
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
| `farmshare_static_lgbm_l40s.sbatch` | The Slurm job: one 4×L40S node, reads only the pre-staged local data manifest, runs `runpod/entrypoint.py --arm-index 2` |
| `farmshare_preflight_submit_static_lgbm.sh` | Checks the staged run folder, venv and W&B/HF session files, then submits the job |

The other controls the arms are compared against are the MixLaw runs; see
[`../mixlaw/README.md`](../mixlaw/README.md#training-code-provenance).

## Evaluation and uncertainty

Same as MixLaw: evals every 125 of the 2384 steps, with step 2375 excluded; power law \(y = a + b/\mathrm{step}^{\alpha}\) on steps ≥ 1000;
fitted final as center; **alpha-free** residual bootstrap (200,000 draws, \(\alpha\)
re-selected on every draw) for the 95% CI. Reproduce with
[`../mixlaw/fit_and_bootstrap_370m.py`](../mixlaw/fit_and_bootstrap_370m.py) from
the committed curves in
[`../mixlaw/skill_dag_370m_wandb_curves.json`](../mixlaw/skill_dag_370m_wandb_curves.json).

---

## Results

Numbers below are for the two Skill-It runs
(`skillit-370m-probe-rerun-20260918-011120`,
`skillit-370m-deriv-20260916-124719`), both starting from the LightGBM optimum,
and for the two runs of that static mixture they are compared against: the
original 8×A100 MixLaw arm and the matched 4×L40S rerun
(`static-lgbm-min1pct-farmshare-1744338`).

### Fitted final macro task-loss (bpb)

| Arm | Fitted final | Observed | 95% CI | vs Olmo control |
|-----|-------------:|---------:|--------|-----------------|
| Olmo-mix-1124 average (control) | 1.6291 | 1.6327 | [1.6246, 1.6335] | — |
| LightGBM static (8×A100) | 1.6080 | 1.6077 | [1.6049, 1.6106] | \(p < 10^{-4}\) |
| LightGBM static (4×L40S) | 1.6238 | 1.6248 | [1.6190, 1.6291] | \(p = 0.13\) |
| Offline probe | 1.6112 | 1.6124 | [1.6078, 1.6141] | \(p < 10^{-4}\) |
| Online derivative | 1.6166 | 1.6216 | [1.6114, 1.6235] | \(p = 0.004\) |

Lower is better. Both Skill-It arms beat the Olmo-mix-1124 control (probe
\(p < 10^{-4}\), derivative \(p = 0.004\)). The matched 4×L40S static rerun is
not distinguishable from it: 0.0053 bpb better (95% CI [-0.0016, 0.0118],
\(p = 0.13\)). Against the two runs of the LightGBM static mixture the arms
start from (Δ = first minus second, so negative means the first is better):

| Comparison | Δ bpb | 95% CI | \(p\) |
|------------|------:|--------|------:|
| Offline probe − LightGBM static (8×A100) | +0.0032 | [-0.0011, +0.0075] | 0.14 |
| Online derivative − LightGBM static (8×A100) | +0.0086 | [+0.0026, +0.0161] | 0.002 |
| Offline probe − LightGBM static (4×L40S) | -0.0126 | [-0.0188, -0.0068] | \(< 10^{-4}\) |
| Online derivative − LightGBM static (4×L40S) | -0.0072 | [-0.0148, +0.0012] | 0.087 |
| LightGBM static (4×L40S) − LightGBM static (8×A100) | +0.0158 | [+0.0102, +0.0218] | \(< 10^{-4}\) |

The comparisons against the 8×A100 run carry a hardware and initialization
difference (step-0 task loss 4.4261 vs 4.4728 bpb); the 4×L40S rerun shares the
dynamic arms' hardware, initialization, data seed and training code.

**Run-to-run variation.** Two Olmo-mix-1124 runs differing in dataloader seed (12536 vs 12345), hardware
(8×A100 vs 4×L40S, which also changed the realized initialization) and training code (the second control ran a separate trainer) land 0.0044 bpb apart (95% CI [-0.0046, 0.0132], \(p = 0.34\)). The two
LightGBM static runs, which differ in the same three ways (data seed 12536 vs
42), land 0.0158 bpb apart: 3.6× the control-seed difference, and larger than
either dynamic arm's advantage over the matched rerun. The effect of dynamic
reweighting therefore cannot be separated from run-to-run variation.

### Takeaways

1. **No reliable evidence that Skill-It helps** under this one-epoch 370M
   contract. Against the matched 4×L40S static rerun both arms come out ahead
   (probe by 0.0126 bpb, \(p < 10^{-4}\); derivative by 0.0072 bpb,
   \(p = 0.087\)); against the 8×A100 static run both come out behind (probe by
   0.0032 bpb, \(p = 0.14\); derivative by 0.0086 bpb, \(p = 0.002\)).
2. **Run-to-run variation swamps the effect.** The two static LightGBM runs
   differ by 0.0158 bpb (95% CI [0.0102, 0.0218], \(p < 10^{-4}\)), 3.6× the
   0.0044 bpb control-seed difference and larger than either arm's advantage
   over the matched rerun.
3. **The matched rerun is not distinguishable from the control.** It is 0.0053
   bpb better than the Olmo-mix-1124 average (95% CI [-0.0016, 0.0118],
   \(p = 0.13\)), against 0.021 bpb for the 8×A100 run of the same mixture.
4. **Both Skill-It arms beat the natural corpus weighting** by 0.013–0.018 bpb
   (probe \(p < 10^{-4}\), derivative \(p = 0.004\)).
5. **Cost.** Two Skill-It trains \(\approx 5.26\times10^{19}\) FLOPs on 4×L40S,
   plus \(2.63\times10^{19}\) for the matched static rerun and
   \(\approx 1.22\times10^{18}\) FLOPs for the 60M probes.

---

## Conclusions

At this scale and budget, **there is no reliable evidence that the Skill-It
rule at \(\eta = 0.2\), with either the offline probe or the mixing-law
derivative adjacency, improves training efficiency.** Whether the arms beat the
LightGBM mixture they start from depends on which run of that mixture is the
reference: they trail the 8×A100 run and lead the matched 4×L40S rerun, and the
two static runs differ by more than any of those gaps. The effect of dynamic
reweighting cannot be separated from run-to-run variation. Note the design
limit: both arms start *at* an optimized mixture and move away from it, so this
tests whether Skill-It can improve on a good mix, not whether it can rescue a
bad one.

Benchmark contamination between the shared 127B-token reservoir and the
evaluation suite is audited in [contamination/](contamination/), including
each arm's time-weighted exposure over its realized, time-varying mixture.
