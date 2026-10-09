# Arms

Every arm below runs through the same code path: one entrypoint
(`token_selection_entrypoint.py`), one train module (`TokenWeightedTrainModule`), and
one hardware contract (FarmShare, 4×L40S). Corpora are read from local FarmShare paths
staged by `../olmo_core_token_selection/farmshare/stage_local.py`, bound by the per-file
sha256 recorded in this repository's `../../datasets/manifests/<corpus>/outputs.json`. See
[`../olmo_core_token_selection/PROVENANCE.md`](../olmo_core_token_selection/PROVENANCE.md)
for the exact commit every run logged, and
[`../olmo_core_token_selection/token_selection_370m/arms.py`](../olmo_core_token_selection/token_selection_370m/arms.py)
for the source of truth these YAMLs describe. Each YAML's `wandb_run` is the W&B run the
paper reports for that arm.

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
  every arm except BLADE, which uses 8,192: both the proxy and the dynamic reference,
  plus both optimizers, are resident during its K-update sync, and 16,384 OOMs a
  44 GiB L40S. RHO-1 and Perplexity read reference losses from the offline table, so
  they hold no second model.
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
check, and the 2360-step ladder.

## Method notes

- **Attention.** Each token's score is the causal attention it receives on the last
  block, aligned to the target token whose loss it gates. Trained attention is
  recency-biased, so a theoretical normalizer (the expectation under uniform causal
  attention) leaves the score position-confounded. The score is instead z-scored
  against the mean and standard deviation of tokens at its own position from the
  model's own immediately preceding training step (`AttentionPositionBaseline`, no
  smoothing), falling back to the uniform-attention prior only on a run's first step.
  The pre-production diagnostic `farmshare/attention_diagnostic.py`, run against a
  trained checkpoint, puts every position bin between 56.8% and 63.6% keep rate under
  this score.
- **BLADE.** The reference's own training term is selection-weighted (Wang et al.
  2026, Sec. 2.2), scored against the *outgoing* reference before it is overwritten by
  the sync. Syncs run at steps 0/400/800/1200/1600/2000, so BLADE selects at the same
  60% budget as every other arm from step 0.

## Running the tests

```powershell
cd experiments\token-selection\olmo_core_token_selection
$env:PYTHONPATH="$PWD\src;$PWD"
py -3 -m pytest tests -q
```

`../token_selection/tests/test_arms_yaml.py` checks every YAML in this directory
against `token_selection_370m.arms.ARM_SPECS`.
