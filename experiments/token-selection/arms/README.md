# Arms

Every arm below runs through the same code path: one entrypoint
(`token_selection_entrypoint.py`), one train module (`TokenWeightedTrainModule`), and
one hardware contract (FarmShare, 4×L40S). There is no RunPod path and no AWS/S3 code
anywhere in this tree; corpora are read from local FarmShare paths staged by
`../olmo_core_token_selection/farmshare/stage_local.py`, bound by the per-file sha256
recorded in this repository's `../../datasets/manifests/<corpus>/outputs.json`. See
[`../olmo_core_token_selection/PROVENANCE.md`](../olmo_core_token_selection/PROVENANCE.md)
for the exact commit every run logged, and
[`../olmo_core_token_selection/token_selection_370m/arms.py`](../olmo_core_token_selection/token_selection_370m/arms.py)
for the source of truth these YAMLs describe.

This directory replaces the previous per-arm directories (`attention/`, `rho-1/`,
`middle-ppl-token/`, `rel-ema-exp/`, `reference/`, `ARMS.md`), each of which was a
README plus a config file describing a superseded, partly RunPod/8×A100 setup that
produced none of the numbers reported after the unification. That history is kept in
git, not here.

## Shared contract

- **Architecture:** `TransformerConfig.olmo2_370M`: `d_model=1024`, 16 layers/heads,
  reordered norm, gated-SiLU 4096 MLP, full attention, QK-RMSNorm, RoPE theta 500,000,
  Dolma2 vocabulary 100,352, sequence length 2048.
- **Optimization:** SkipStepAdamW, peak LR `4e-4`, betas `(0.9, 0.95)`, weight decay
  `0.1` except embeddings at `0`, 24-step cosine warmup, `alpha_f=0.1`, Z-loss `1e-5`,
  grad norm clip `1.0`, HSDP bf16 parameters / fp32 reductions, compilation enabled.
- **Global batch:** 4,194,304 tokens (`GLOBAL_BATCH_TOKENS` in `recipe.py`).
  `total_steps(max_tokens) = max_tokens // GLOBAL_BATCH_TOKENS`.
- **Hardware:** one FarmShare node, 4×L40S (`PRODUCTION_WORLD_SIZE = 4`), for every
  *reported* arm. `instruct-reference` is the one exception -- see its row below.
- **Rank microbatch:** 16,384 tokens (`ArmSpec.rank_microbatch_tokens` default) for
  every arm except BLADE. Arms that hold a second model in memory during training
  (BLADE, RHO-1, Perplexity) may fall back to 8,192 or 4,096 if the smoke test needs
  it; BLADE's 1-GPU smoke test OOM'd a 44 GiB L40S at 16,384 (both the proxy and the
  dynamic reference, plus both optimizers, are resident during its K-update sync), so
  it now uses 8,192. RHO-1 and Perplexity haven't needed to fall back yet.
- **Permanent checkpoints:** step 0, every 125 steps, and the true final step,
  omitting the last 125-grid point when it falls within 100 steps of the final step. The
  2360-step arms therefore keep step 2250 (110 steps from the end); the 940-step
  reference omits step 875 (65 steps from the end).
- **Evaluator:** the 20-label `*_rc_5shot_bpb` OLMES suite
  (`../olmo_core_token_selection/eval_task_loss_olmo_core.py`), run on every permanent
  checkpoint.
- **W&B project:** `token-selection` for every arm.

## Arms and references

| id | method | corpus | keep | seeds (init/data) | reference | notes |
|---|---|---:|---:|---|---|---|
| [`instruct-reference`](instruct-reference.yaml) | `full` | `pretrain/refhq-instruct` | 100% | 6198 / 42 | — | trains the frozen Instruct reference `rho-1` and `perplexity` score against offline, and BLADE's L_val stream; 940 steps, one whole-stream epoch of the train split; frozen at step 940; trained on FarmShare's 1-GPU `qos=normal` lane (not the 4×L40S contract above) while the 4-GPU lane was occupied by an unrelated study -- a hardware choice only, since its checkpoint identity is pinned by weight sha256, not world size (see `../olmo_core_token_selection/README-token-selection.md`) |
| [`full-loss-control`](full-loss-control.yaml) | `full` | `pretrain/regmix-10b` | 100% | 6198 / 42 | — | 2360 steps |
| [`random-control`](random-control.yaml) | `random` | `pretrain/regmix-10b` | 60% | 6198 / 42 | — | 2360 steps; mask drawn per corpus instance, independent of world size/rank/microbatch |
| [`random-control-seed69`](random-control-seed69.yaml) | `random` | `pretrain/regmix-10b` | 60% | 12345 / 69 | — | the seed-variance replicate; independent draw from `random-control`, never overlapping it |
| [`rho-1`](rho-1.yaml) | `rho_excess` (top `L_curr − L_ref`) | `pretrain/regmix-10b` | 60% | 6198 / 42 | frozen `instruct-reference` step 940, scored offline once into a per-instance table | 2360 steps |
| [`rel-ema-exp`](rel-ema-exp.yaml) | `rel_ema` (top `L_curr − L_hist`) | `pretrain/regmix-10b` | 60% | 6198 / 42 | zero-seeded bias-corrected EMA, `alpha(t) = 1 - exp(-t/300)` | 2360 steps |
| [`perplexity`](perplexity.yaml) | `middle_ppl` (middle 60% by frozen loss) | `pretrain/regmix-10b` | 60% | 6198 / 42 | frozen `instruct-reference` step 940, scored offline once into the same per-instance table `rho-1` reads | 2360 steps |
| [`attention`](attention.yaml) | `attention_topk` (top target-aligned attention received, z-scored per position against the model's own immediately preceding step) | `pretrain/regmix-10b` | 60% | 6198 / 42 | — | 2360 steps |
| [`blade`](blade.yaml) | `blade` (top `L_proxy - L_ref`, selection-weighted reference update) | `pretrain/regmix-10b` | 60% | 6198 / 42 | dynamic, synced from the proxy at steps 0/400/800/1200/1600/2000 (`tau=400`, `K=75`); second stream `pretrain/refhq-instruct` | 2360 steps |

Every arm above except the Instruct reference is trained under the one commit pinned in
`../olmo_core_token_selection/PROVENANCE.md`. The Instruct reference recorded the earlier
commit `765ae838`; its method (full loss) and its checkpoint ladder are unchanged in every
later commit, which differ only in the Attention scoring, the entrypoint's dataset-version
check, and the 2360-step ladder. Nothing is kept from a pre-unification commit.

## Methodology fixes since the confounded runs

- **Attention.** The raw causal column-mass score favored early positions and was one
  position off from the target it was meant to gate; the token-count alignment fix
  landed first. A second, larger problem surfaced later, from a pre-production
  diagnostic (`farmshare/attention_diagnostic.py`) run against a real trained
  checkpoint: normalizing by the *theoretical* expectation under uniform causal
  attention still left the score almost entirely position-confounded (keep rate
  ranged from 1.9% at the start of a row to 100% at the end), because real trained
  attention is recency-biased, not uniform -- the normalizer assumed a ~117x drop in
  attention mass from the first to the last position, while the real drop is only
  ~5.4x. The fix replaces that theoretical normalizer with an *empirical* one
  (`AttentionPositionBaseline`): each token is z-scored against the mean/std of
  tokens at its own position from the model's own immediately preceding training
  step (no smoothing/decay constant -- an offline sensitivity check found smoothing
  over more steps cut responsiveness to real drift for a barely-measurable noise
  benefit at this batch size), falling back to the old uniform-attention prior only
  on a fresh run's first step, before any real history exists. Re-running the same
  diagnostic against the same checkpoint under the new score: every bin lands
  between 56.8% and 63.6% keep rate.
- **BLADE.** The reference's own training term is now selection-weighted (Wang et al.
  2026, Sec. 2.2), scored against the *outgoing* reference before it is overwritten by
  the sync. The schedule starts at step 0 (previously step 500), syncing at
  0/400/800/1200/1600/2000 instead of 500/875/1250/1625/2000, so BLADE selects at the
  same 60% budget as every other arm from step 0 rather than running full-loss (68.5%
  effective keep) through step 500.

## Running the tests

```powershell
cd experiments\token-selection\olmo_core_token_selection
$env:PYTHONPATH="$PWD\src;$PWD"
py -3 -m pytest tests -q
```

`../token_selection/tests/test_arms_yaml.py` checks every YAML in this directory
against `token_selection_370m.arms.ARM_SPECS`.
