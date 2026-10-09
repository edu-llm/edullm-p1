# Skill-DAG experiments

Domain-mixture levers under a shared Mixing Laws Dataset / OLMoHQ × OLMo-2 370M one-epoch contract.

| Experiment | Question | Outcome |
|------------|----------|---------|
| MixLaw | Can a mixing law from short runs beat fixed mixtures? | **No** — none of the three optimized static mixtures beats the Olmo-mix-1124 control average (fitted final 1.6148 bpb): LightGBM 1.6238 (\(+0.0091\), \(p = 0.0048\)), Data Mixing Laws 1.6411 (\(+0.0264\), \(p < 5\times10^{-6}\)) and MixLaw 1.6433 (\(+0.0286\), \(p < 5\times10^{-6}\)) all finish behind it |
| Skill-It | Can mid-run Skill-It reweighting beat the fixed mixture it starts from? | **No** — against the control average, the probe arm is worse (\(+0.0099\) bpb, \(p = 0.0010\) two-sided) and the derivative arm indistinguishable (\(+0.0019\), \(p = 0.66\)). Against its own static starting mixture (LightGBM) the probe arm is indistinguishable (\(+0.0009\), \(p = 0.79\)) and the derivative arm finishes \(0.0072\) ahead (two-sided \(p = 0.087\), one-sided \(0.044\)), less than the 0.0105 bpb difference between the two control seeds |

All seven reported 370M runs (the Olmo-mix-1124 control at data seeds 42 and 69,
Data Mixing Laws, MixLaw, LightGBM, and the two Skill-It arms) ran the vendored
Skill-It trainer on FarmShare 4×L40S, matched in everything except mixture (and,
for the second control, the data seed).

Shared evaluation: macro task-loss CE bits-per-byte over 20 OLMES-style labels;
power-law **alpha-free** residual-bootstrap CIs on fitted finals (steps ≥ 1000),
one independent resampling stream per arm, reproducible offline via
[`mixlaw/fit_and_bootstrap_370m.py`](mixlaw/fit_and_bootstrap_370m.py) from the
single curve file `mixlaw/skill_dag_370m_wandb_curves.json`, which
[`mixlaw/pull_370m_curves.py`](mixlaw/pull_370m_curves.py) writes from W&B.
FLOPs: \(2.63\times10^{19}\) per 370M arm (W&B); MixLaw 60M pilot grid
\(\approx 4.17\times10^{18}\); Skill-It 60M probes \(\approx 1.22\times10^{18}\).

**Data.** Every arm draws from the ~127B-token reservoir `pretrain/olmo-127b` v1,
a subset of Olmo-mix-1124 (DCLM 29.691B, arXiv 22.148B, pes2o 26.379B, StarCoder
18.541B, OpenWebMath 13.238B, Algebraic Stack 12.902B, Wikipedia 3.752B tokens).
Olmo-mix-1124 is licensed ODC-By v1.0, and its DCLM component is also subject to
the Common Crawl terms of use. No source documents are redistributed here; rebuild
the reservoir byte for byte from the public Olmo-mix-1124 files with
[`datasets/manifests/olmo-127b-v1/`](../../datasets/manifests/olmo-127b-v1/README.md)
(`rebuild.py`).

**Seed noise floor.** Two Olmo-mix-1124 runs differing only in data seed (42 vs 69)
land 0.0105 bpb apart (95% CI [+0.0023, +0.0193], \(p = 0.0095\)); treat
single-run differences of about 0.01 bpb or less as indistinguishable from
run-to-run variation. Even identically configured runs on this stack are not
bit-reproducible (the Skill-It probe, derivative and LightGBM runs diverge by
step ~15, before any Skill-It update), so this is a floor on data order plus
GPU nondeterminism, not on data order alone; see
[`skillit/README.md`](skillit/README.md#results).

## Bootstrap convention (and how it differs from the token-selection paper)

Both papers report 95% intervals from an alpha-free residual bootstrap on the
fit window, but the two are **not** computed identically. The difference is
worth knowing before comparing their seed-noise floors (0.0105 bpb here,
0.0082 bpb there).

| | this paper (`fit_and_bootstrap_370m.py`) | token selection (`fit_and_plot.py`) |
|---|---|---|
| alpha grid | `linspace(0.05, 3.0, 400)` | `linspace(0.05, 6.0, 1192)` |
| draws | 200,000 | 10,000 |
| small-sample residual rescaling | **not applied** | `sqrt(n/(n-p))`, `p=3` |

The rescaling corrects for OLS residuals being shrunk relative to the true
errors; without it, intervals are roughly 15% narrower. The intervals behind
Tables II and III are therefore narrower than the token-selection convention
would produce, not wider.

`skill_dag_370m_bootstrap_results.json` does not record the per-arm fitted
`alpha`, so unlike the token-selection artifact it cannot be inspected for
boundary-pinning at the 3.0 grid edge.
