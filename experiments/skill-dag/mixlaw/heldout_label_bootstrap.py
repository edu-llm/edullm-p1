"""Run the paper's own bootstrap on the targeted-12 vs held-out-8 label subsets.

Imports `fit_and_bootstrap` / `ci` / `diff_p` from the committed
`fit_and_bootstrap_370m.py` rather than reimplementing them, so the CIs and
p-values are produced by exactly the method Tables II and III use: power-law
fit y = a + b/step**alpha on steps >= 1000, alpha-free residual bootstrap,
n_boot = 200000, seed = 0, one independent stream per arm, control = the
element-by-element average of the two control seeds' draws.

Input is heldout_label_curves.json, the per-step means of the two label subsets
pulled from W&B (12 targeted = val+test of the six optimized skills; 8
never-targeted = boolq, csqa, hellaswag, openbookqa val+test, piqa, socialiqa,
winogrande).
"""
import json
import pathlib
import sys

import numpy as np

MIXLAW = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(MIXLAW))
from fit_and_bootstrap_370m import ci, diff_p, fit_and_bootstrap, fmt_p  # noqa: E402

curves = json.loads((MIXLAW / "heldout_label_curves.json").read_text(encoding="utf-8"))

# All-L40S arm set; aliases match pull_370m_curves.py's key_alias mapping.
# No arm here has a previously reported targeted/held-out number to keep
# bit-identical (that breakdown is new for every arm), so a plain positional
# spawn is fine, unlike fit_and_bootstrap_370m.py's Table II/III streams.
ORDER = ["olmo_s42", "olmo_s69", "dml_paper", "mixlaw", "lightgbm", "probe", "derivative"]
EXCLUDE = {2375}          # the paper excludes the off-cadence eval
FINAL_STEP = 2384
N_BOOT = 200_000

def series(arm, field):
    rows = curves[arm]
    steps, vals = [], []
    for s, rec in sorted(rows.items(), key=lambda kv: int(kv[0])):
        st = int(s)
        if st in EXCLUDE:
            continue
        steps.append(st)
        vals.append(rec[field])
    return steps, vals

for field, pretty, n_lab in (("targeted", "12 targeted labels", 12),
                             ("held_out", "8 never-targeted labels", 8)):
    print(f"\n{'='*74}\n{pretty}  (n = {n_lab})\n{'='*74}")
    streams = np.random.SeedSequence(0).spawn(len(ORDER))
    fitted, finals, obs = {}, {}, {}
    for arm, stream in zip(ORDER, streams):
        st, v = series(arm, field)
        f, dist = fit_and_bootstrap(st, v, final_step=FINAL_STEP,
                                    n_boot=N_BOOT, seed=stream)
        fitted[arm], finals[arm], obs[arm] = f, dist, float(v[-1])

    finals["control"] = 0.5 * (finals["olmo_s42"] + finals["olmo_s69"])
    fitted["control"] = 0.5 * (fitted["olmo_s42"] + fitted["olmo_s69"])
    obs["control"] = 0.5 * (obs["olmo_s42"] + obs["olmo_s69"])

    print(f"{'arm':12s} {'fitted':>8s} {'observed':>9s}   95% CI")
    for arm in ORDER + ["control"]:
        lo, hi = ci(finals[arm])
        print(f"{arm:12s} {fitted[arm]:8.4f} {obs[arm]:9.4f}   [{lo:.4f}, {hi:.4f}]")

    print(f"\n{'vs control':28s} {'delta':>8s}   95% CI{'':14s} p")
    for arm in ("mixlaw", "lightgbm", "probe", "derivative"):
        d = finals[arm] - finals["control"]
        lo, hi = ci(d)
        p = diff_p(finals[arm], finals["control"])
        sign = "better" if d.mean() < 0 else "WORSE"
        print(f"{arm + ' - control':28s} {d.mean():+8.4f}   "
              f"[{lo:+.4f}, {hi:+.4f}]   {fmt_p(p, N_BOOT)}  ({sign})")
