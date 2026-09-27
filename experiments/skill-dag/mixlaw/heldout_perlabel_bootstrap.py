"""Per-label bootstrap over the 8 never-targeted labels.

Same method as the paper (imported from fit_and_bootstrap_370m): power-law fit
on steps >= 1000, alpha-free residual bootstrap, n_boot = 200000, seed = 0, one
stream per arm, step 2375 excluded, control = element-wise mean of the two
control seeds' draws.
"""
import json
import pathlib
import sys

import numpy as np

MIXLAW = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(MIXLAW))
from fit_and_bootstrap_370m import ci, diff_p, fit_and_bootstrap, fmt_p  # noqa: E402

data = json.loads((MIXLAW / "heldout_perlabel_curves.json").read_text(encoding="utf-8"))

LABELS = [
    "boolq_val_rc_5shot_bpb", "csqa_val_rc_5shot_bpb",
    "hellaswag_val_rc_5shot_bpb", "openbookqa_val_rc_5shot_bpb",
    "openbookqa_test_rc_5shot_bpb", "piqa_val_rc_5shot_bpb",
    "socialiqa_val_rc_5shot_bpb", "winogrande_val_rc_5shot_bpb",
]
SHORT = {l: l.replace("_rc_5shot_bpb", "") for l in LABELS}
ARMS = ["olmo_s42", "olmo_s69", "mixlaw", "lightgbm"]
EXCLUDE = {2375}
FINAL, NB = 2384, 200_000

def series(arm, label):
    st, v = [], []
    for s, rec in sorted(data[arm].items(), key=lambda kv: int(kv[0])):
        if int(s) in EXCLUDE:
            continue
        st.append(int(s)); v.append(rec[label])
    return st, v

print(f"{'label':26s} {'ctrl':>7s} {'MixLaw d':>9s} {'95% CI':>20s} {'p':>9s}   "
      f"{'LGB d':>8s} {'95% CI':>20s} {'p':>9s}   "
      f"{'seed d':>8s} {'95% CI':>20s} {'p':>9s}")
print("-" * 155)
rows = []
for label in LABELS:
    streams = np.random.SeedSequence(0).spawn(len(ARMS))
    finals, fitted = {}, {}
    for arm, stream in zip(ARMS, streams):
        st, v = series(arm, label)
        f, dist = fit_and_bootstrap(st, v, final_step=FINAL, n_boot=NB, seed=stream)
        finals[arm], fitted[arm] = dist, f
    finals["control"] = 0.5 * (finals["olmo_s42"] + finals["olmo_s69"])
    fitted["control"] = 0.5 * (fitted["olmo_s42"] + fitted["olmo_s69"])

    cells = []
    for arm in ("mixlaw", "lightgbm"):
        d = finals[arm] - finals["control"]
        lo, hi = ci(d)
        p = diff_p(finals[arm], finals["control"])
        cells.append((d.mean(), lo, hi, p))
    seed_d = finals["olmo_s69"] - finals["olmo_s42"]
    seed_lo, seed_hi = ci(seed_d)
    seed_p = diff_p(finals["olmo_s69"], finals["olmo_s42"])
    seed_cell = (seed_d.mean(), seed_lo, seed_hi, seed_p)
    rows.append((label, fitted["control"], cells, seed_cell))
    (dm, dlo, dhi, dp), (lm, llo, lhi, lp) = cells
    sm, slo, shi, sp = seed_cell
    print(f"{SHORT[label]:26s} {fitted['control']:7.4f} {dm:+9.4f} "
          f"[{dlo:+.4f},{dhi:+.4f}] {fmt_p(dp, NB):>9s}   "
          f"{lm:+8.4f} [{llo:+.4f},{lhi:+.4f}] {fmt_p(lp, NB):>9s}   "
          f"{sm:+8.4f} [{slo:+.4f},{shi:+.4f}] {fmt_p(sp, NB):>9s}")

print("\nsign summary (negative = better than control):")
for arm_i, arm in enumerate(("mixlaw", "lightgbm")):
    better = [SHORT[l] for l, _, c, _ in rows if c[arm_i][0] < 0]
    worse = [SHORT[l] for l, _, c, _ in rows if c[arm_i][0] > 0]
    sig_w = [SHORT[l] for l, _, c, _ in rows if c[arm_i][1] > 0]
    sig_b = [SHORT[l] for l, _, c, _ in rows if c[arm_i][2] < 0]
    print(f"  {arm}: better {len(better)}/8, worse {len(worse)}/8")
    print(f"    significantly BETTER (CI excludes 0): {sig_b or 'none'}")
    print(f"    significantly WORSE  (CI excludes 0): {sig_w or 'none'}")

sig_seed = [SHORT[l] for l, _, _, sc in rows if sc[1] > 0 or sc[2] < 0]
print(f"\n  seed spread (s69 - s42) significant (CI excludes 0): {sig_seed or 'none'}")
