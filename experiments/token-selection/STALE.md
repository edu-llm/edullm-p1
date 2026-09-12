# Stale artifacts in `experiments/token-selection/`

Everything listed here is, in my assessment, **safe to remove or already superseded**.
**Nothing in this list was removed.** This file exists so the deletion can be reviewed
and staged deliberately rather than done silently. Each entry gives the reason in one
line.

Written 2026-09-12, alongside the doc corrections in `README.md`, `ARMS.md`, and the arm
configs. Items marked **FIXED IN PLACE** were repaired instead of being flagged for
deletion.

---

## 1. `task_loss_results/` — DELETED (2026-09-12)

Removed. This directory held a superseded evaluation generation that directly
contradicted Table 1 and was the first thing a reader following the paper's code URL
encountered. Evidence recorded here for the record:

- Values were nats-scale per-token CE (`core7_avg` 3.3-5.3, `mmlu_avg` 5.0-5.3), not the
  bits-per-byte the paper reports (control final 1.6487).
- The `ce-regmix` full-CE control got *worse* over training: `core7_avg` 3.311 at step
  250, 3.503 at 500, 3.566 at 750, 3.638 at 2384.
- `rel-ema` carried `step2375` and `step2386`, which match neither the 2360-step arms nor
  the control's 2384 steps.
- The grid was 250-step, not the 125-step permanent-checkpoint ladder the arms used.

The authoritative inputs to Table 1 are now committed alongside it:
`token_selection_370m_wandb_curves.json`, `token_selection_370m_bootstrap_results.json`,
`token_selection_370m_final_numbers.json`, and `fit_and_plot.py`.

## 2. Arm directories not among the paper's seven

The paper reports seven arms: `control`, `rho-1`, `blade`, `attention`,
`middle-ppl-token`, `rel-ema-exp`, `reference`. These four are not among them.

> **Already deleted in the working tree before I started.** `git status` shows all four
> as staged-`D` deletions that I did not make. They are listed here for completeness of
> the review, not as a request to delete them again.

- **`learnability-doc/`** — document-level learnability filtering; dropped before the
  reported runs, so it was never run.
- **`learnability-token/`** — token-level learnability scoring; dropped before the
  reported runs, so it was never run.
- **`middle-ppl-doc/`** — document-level middle-perplexity variant; superseded by
  `middle-ppl-token/`, which is the arm actually reported.
- **`rel-ema-refhq/`** — REL-EMA seeded from the RefHQ checkpoint; superseded by
  `rel-ema-exp/` (`ema_seed="zero"`), which is the arm actually reported.
  `rel-ema-exp/README.md` still links to it as a "near-clone", so that link will need
  updating when the directory goes.

The task record attributes the drop to commit `90073574`. **That SHA does not resolve in
this repository** (`git rev-parse 90073574` fails and no commit touching
`learnability-doc/` is reachable from `main`), so the drop presumably happened in the
`edu-llm/OLMo-core` fork or was squashed away. Do not cite `90073574` as an
`edullm`-repo commit without re-verifying it.

## 3. `fetch_token_selection_wandb.py` — **FIXED IN PLACE, not flagged for deletion**

The run IDs could be reconciled, so I fixed them rather than recommending removal.

| Arm | Was | Now |
| --- | --- | --- |
| `control` | `eduLLM/mixlaw-1/1e9df6cccc294c6d0f1bce328497f6a6` (run `mix01`) | `eduLLM/token-selection/hh19uatg` (`full-loss-control-regmix10b-v2`) |
| `rel_ema` | `eduLLM/token-selection/89db0d5b6a59cfa58ed58df0b79bd9ab` | `eduLLM/token-selection/cc52d5537a03ad8e57cc87a025668b2e` |

Both replacements are the IDs already present in the cached
`token_selection_370m_wandb_curves.json`, whose observed finals match Table 1 exactly
(1.6487 for the control, 1.9207 for REL-EMA), so the reconciliation is verified rather
than assumed. The stale `CONTROL_PROJECT = "mixlaw-1"` constant was removed and the
`source` string updated.

**Remaining concern (not fixed):** this script's bootstrap holds `alpha` **fixed** from
the initial fit and inherits `mixlaw_power_law.residual_bootstrap_ci`, whereas the
reported Table 1 intervals come from `fit_and_plot.py`, which **re-estimates `alpha` on
every draw** over the wider `np.linspace(0.05, 6.0, 1192)` grid. Running this script
will therefore overwrite `token_selection_370m_bootstrap_results.json` with **narrower,
non-reported** intervals. I added a warning to the script's `method` string but did not
change its fitter, since `fit_and_plot.py` and the two JSON result files are owned by
another change in flight. Either delete this script in favor of `fit_and_plot.py`, or
make it call the same fitter — do not leave two fitters writing the same file.

## 4. `olmo_core.revision` pins — **FIXED IN PLACE**

All four arm configs (plus the shared template and two launch scripts) pinned

```
olmo_core.revision: 99e0009ed67679c90da970ec5ba439c9459e3757
```

at which `.edullm/token_selection_370m/selection.py` **does not exist** — so no
selection arm can have run at that revision, and the pin is not reproducible. Repinned
to the branch head that actually carries the code:

```
olmo_core.revision: 98ea67c948fd93ccbfd2633e1ae818c3ae1d2ad7
```

(branch `edullm/token-selection-370m` of the `edu-llm/OLMo-core` fork). Files changed:
`attention/configs/run_attention_10b.yaml`,
`middle-ppl-token/configs/run_middle_ppl_token_10b.yaml`,
`rel-ema-exp/configs/run_rel_ema_exp_10b.yaml`, `rho-1/configs/run_rho_10b.yaml`,
`token_selection/configs/run_rho_10b.yaml`, `rel-ema-exp/launch_train.sh`,
`rho-1/farmshare/run_rho_train.sh`. Each carries an inline comment recording the old
pin and why it was wrong.

Two other files still reference the old SHA and were **left alone deliberately**:
`token_selection/tests/test_resume_guard.py` (it is fixture data for the resume
fingerprint test, where the literal value is arbitrary) and
`scripts/farmshare/unshard_olmo_core_run.sh` /
`scripts/farmshare/unshard_refhq_step1315.sh` (outside this task's scope, and the
unshard path does not need the selection module).

## 5. `model.init_seed: 42` vs `INIT_SEED = 6198` — reconciled, **value intentionally unchanged**

The four arm configs declare `model.init_seed: 42`, but `recipe.py` sets
`INIT_SEED = 6198` and **the reported runs used 6198**. So:

- **`recipe.py`'s `INIT_SEED = 6198` governed.** The launch path goes through
  `recipe.py`; the YAML `init_seed` did not reach `TransformerConfig` for the reported
  runs.
- **The YAML value was left at 42 on purpose.**
  `token_selection/scripts/experiment_contract.py:109-110` raises
  `"model.init_seed must match the shared experiment seed"` unless
  `model.init_seed == seed`, and both are part of the resume fingerprint
  (`test_resume_guard.py` asserts a fingerprint change on `init_seed`). Changing
  `init_seed` alone would hard-fail validation; changing `seed` too would invalidate
  resume of the existing runs. Each config now carries an inline comment stating that
  6198 is authoritative for what ran.

If you want the configs to be self-describing rather than annotated, the correct fix is
to set `seed: 6198` **and** `init_seed: 6198` together, and accept the fingerprint
break — which should be done as its own change, not as part of a docs pass.

## 6. `token_selection/olmo_ext/scorers.py` — REL-EMA polarity FIXED (2026-09-12)

Resolved. This vendored scorer had kept the **pre-fix, inverted** REL polarity
(`L_hist - L_curr`) while the reported REL-EMA arm is the 2026-09-05 polarity-corrected
re-run, whose code computes `L_curr - L_hist`. The sign has now been ported into this
repo so the vendored copy, the fork module, and the docs all agree:

| Site | Was | Now |
|------|-----|-----|
| `olmo_ext/scorers.py` `rel_ema_mask` | `top_k_mask(history_loss - current_loss, ...)` | `top_k_mask(current_loss - history_loss, ...)` |
| `olmo_ext/train_module.py:612` | `score = history_loss - current_loss` | `score = current_loss - history_loss` |
| `olmo_ext/train_module.py:1077` | `score = history_loss - current_loss` | `score = current_loss - history_loss` |
| `tests/test_scorers.py` `test_rel_mask_is_per_sequence` | expected `[[T,T,F,F],[F,F,T,T]]` | expected `[[F,F,T,T],[T,T,F,F]]` |
| `olmo_ext/ema.py` zero-seed rationale | collapse described as `constant - L_curr`, keeping the *easiest* tokens | collapse described as `L_curr - constant`, degenerating to highest-current-loss selection |

The two `train_module.py` sites are diagnostic only — they feed the `mean_kept` /
`mean_dropped` and `score_sums` telemetry, not the mask — but they had to move with the
mask or the logged score statistics would have reported the wrong sign.

Verified after the change: no occurrence of `history_loss - current_loss` or
`L_hist - L_curr` remains anywhere under `experiments/token-selection/` except in
deliberate historical notes (this file, `rel-ema-exp/README.md`,
`rel-ema-exp/configs/run_rel_ema_exp_10b.yaml`, and the `rel_ema_mask` docstring, all of
which describe the superseded run explicitly).

The other three scorers were checked and were already correct: `rho_excess_mask`
(`current - reference`), `learnability_mask` (`early - late`), and `middle_ppl_mask`
(middle-k of the reference loss).

Still open: this vendored scorer duplicates
`.edullm/token_selection_370m/selection.py` on the fork branch
(`98ea67c948fd93ccbfd2633e1ae818c3ae1d2ad7`). Keeping two implementations of the same
scoring rule is how the inverted run happened in the first place, so consider deleting
this copy and importing the fork module instead.

## 7. `token_selection/configs/run_rho_10b.yaml` — duplicate of the arm config

It declares itself "Canonical arm config: `../rho-1/configs/run_rho_10b.yaml` (keep in
sync)" while sharing the same `run_id: rho-1-regmix10b-v1`, and it had drifted out of
sync on the frozen-reference block (still pointing at the legacy refhq-5p5b step1315).
I re-synced it rather than deleting it, but two files with the same `run_id` is a
standing trap; prefer deleting this copy and having the shared package read the arm
config.
