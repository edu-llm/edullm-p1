# Skill-DAG experiments

Domain-mixture levers under a shared Mixing Laws Dataset / OLMoHQ × OLMo-2 370M one-epoch contract.

| Experiment | Question | Outcome |
|------------|----------|---------|
| MixLaw | Can a mixing law from short runs beat fixed mixtures? | **Yes** — fitted mixtures beat the baseline (\(p < 10^{-4}\)) |
| Skill-It | Can mid-run Skill-It reweighting beat the fixed mixture it starts from? | **No reliable evidence** — both arms beat the Olmo-mix-1124 control (probe \(p < 10^{-4}\), derivative \(p = 0.004\)), but they trail the 8×A100 run of their LightGBM starting mixture and lead a matched 4×L40S rerun of it; the two static runs differ by 0.0158 bpb, more than any arm-vs-static gap, so the effect cannot be separated from run-to-run variation |

Shared evaluation: macro task-loss CE bits-per-byte over 20 OLMES-style labels;
power-law **alpha-free** residual-bootstrap CIs on fitted finals (steps ≥ 1000),
one independent resampling stream per arm, reproducible offline via
[`mixlaw/fit_and_bootstrap_370m.py`](mixlaw/fit_and_bootstrap_370m.py).
A100-hours: MixLaw 188.74 (its four 8×A100 arms; both Skill-It arms and the
LightGBM static rerun ran on 4×L40S). FLOPs: \(2.63\times10^{19}\) per 370M
arm (W&B); MixLaw 60M pilot grid \(\approx 4.17\times10^{18}\); Skill-It 60M
probes \(\approx 1.22\times10^{18}\).

**Data.** Every arm draws from the ~127B-token reservoir `pretrain/olmo-127b` v1,
a subset of Olmo-mix-1124 (DCLM 29.691B, arXiv 22.148B, pes2o 26.379B, StarCoder
18.541B, OpenWebMath 13.238B, Algebraic Stack 12.902B, Wikipedia 3.752B tokens).
Olmo-mix-1124 is licensed ODC-By v1.0, and its DCLM component is also subject to
the Common Crawl terms of use. No source documents are redistributed here; rebuild
the reservoir byte for byte from the public Olmo-mix-1124 files with
[`datasets/manifests/olmo-127b-v1/`](../../datasets/manifests/olmo-127b-v1/README.md)
(`rebuild.py`).

**Seed noise floor.** Two Olmo-mix-1124 runs differing in dataloader seed (12536 vs
12345), hardware (8×A100 vs 4×L40S, which also changed the realized initialization)
and training code (the second control ran a separate trainer)
land 0.0044 bpb apart (\(p = 0.34\)); treat differences below ~0.004 bpb as
indistinguishable. Two runs of the LightGBM mixture that differ in the same three
ways land 0.0158 bpb apart (\(p < 10^{-4}\)), so run-to-run variation across
hardware and training code can be several times larger; see
[`skillit/README.md`](skillit/README.md#results).

## Bootstrap convention (and how it differs from the token-selection paper)

Both papers report 95% intervals from an alpha-free residual bootstrap on the
fit window, but the two are **not** computed identically. The difference is
worth knowing before comparing their seed-noise floors (0.0044 bpb here,
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
