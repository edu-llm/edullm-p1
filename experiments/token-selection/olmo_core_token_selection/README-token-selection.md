# Token-selection 370M branch

This branch contains the reported token-selection arms outside `src/olmo_core`.
Every arm uses the same `TransformerConfig.olmo2_370M` recipe: `d_model=1024`,
16 layers/heads, reordered norm, gated-SiLU 4096 MLP, full attention,
QK-RMSNorm, RoPE theta 500,000, Dolma2 vocabulary 100,352, sequence 2048,
global batch 4,194,304 tokens. Optimization is SkipStepAdamW at peak LR
`4e-4`, betas `(0.9, 0.95)`, weight decay `0.1` except embeddings at `0`,
24-step cosine warmup, `alpha_f=0.1`, Z-loss `1e-5`, grad norm 1.0, HSDP bf16
parameters/fp32 reductions, and compilation enabled.

Every run trains on FarmShare, through `token_selection_entrypoint.py` (see
`.edullm/farmshare/`). There is no RunPod path, no AWS/S3 code, and no
separate submission platform anywhere in this tree: corpora are pinned
local FarmShare directories, and the reference checkpoint
(`instruct-reference`) is trained by this branch's own arms, not
downloaded from anywhere.

Every *reported* arm runs on one 4×L40S node (`PRODUCTION_WORLD_SIZE = 4`).
`instruct-reference` is the one exception: it ran on FarmShare's separate
1-GPU `qos=normal` lane (`EDULLM_LOCAL=1`, `TRAIN_GPUS=1`), while the 4-GPU
`qos=gpu` lane was occupied by an unrelated study sharing the same
per-user quota. This is a training-hardware choice only, not a scientific
one: `instruct-reference` isn't a reported comparison arm, its own
checkpoint identity is pinned by weight sha256
(`scientific_identity`/`reference_sha256`) rather than by anything
world-size-dependent, and `GLOBAL_BATCH_TOKENS` (hence every optimizer
step's data volume) is fixed independent of GPU count -- the data loader
runs proportionally more gradient-accumulation microbatches per rank
instead. The real synchronous task-loss eval and real (non-offline) W&B
logging both still ran; only the world-size-4 topology assertion and the
strict-online-upload fail-closed check were relaxed for this run, the same
`EDULLM_LOCAL` path already used by this branch's own 1-GPU smoke tests.

## Arms

| arm | method | keep | notes |
|---|---|---:|---|
| `instruct-reference` | `full` | 100% | trains the frozen Instruct reference `rho-1` and `perplexity` both score against, and BLADE's L_val stream |
| `full-loss-control` | `full` | 100% | |
| `rho-1` | `rho_excess` (top `L_curr - L_ref`) | 60% | frozen `instruct-reference` step 940 |
| `rel-ema-exp` | `rel_ema` (top `L_curr - L_hist`) | 60% | zero-seeded bias-corrected EMA, `alpha(t) = 1 - exp(-t/300)` |
| `perplexity` | `middle_ppl` (middle 60% by frozen loss) | 60% | frozen `instruct-reference` step 940 |
| `attention` | `attention_topk` (top target-aligned log attention received, z-scored per position against the model's own recent attention) | 60% | per-position baseline: the immediately preceding step's own global-batch statistics (unblended, no smoothing constant), uniform-attention prior before the first step; checkpointed |
| `blade` | `blade` (top `L_proxy - L_ref`, selection-weighted reference update) | 60% | syncs at steps 0/400/800/1200/1600/2000, `tau=400`, `K=75`, `gamma=0.6`, `lambda=1.0`; second stream `pretrain/refhq-instruct` |
| `random-control` | `random` | 60% | data seed 42, init seed 6198 |
| `random-control-seed69` | `random` | 60% | data seed 69, init seed 12345 (the seed-variance replicate) |

Every arm routes to W&B project `token-selection`. `random-control` and
`random-control-seed69` are non-scientific baselines: they mask a uniformly
random 60% of tokens per row from the loss
(`selection_weights(method="random", ...)` in
`token_selection_370m/selection.py`) instead of selecting by any score, using
the same keep rate, model, hyperparameters, dataset, and checkpoint/eval
contract as every other arm.

`token_selection_370m/arms.py` is the source of truth for run IDs, dataset
IDs, and reference contracts.

## Inputs and checkpoint identity

Corpora are resolved from a local manifest (`farmshare/stage_local.py`
builds it from the pinned directories in `farmshare/stage_local.py`'s
`CORPUS_SOURCES`, verified against this repository's paper counterpart's
`datasets/manifests/*/outputs.json`). Checkpoint identity binds the resolved
version, dtype, row count, and SHA-256 of the ordered object-path list; BLADE
binds its `refhq-instruct` stream independently. Reference checkpoints are
materialized flat `.pt` files, resolved via `resolve_reference_path()` from
the local manifest and bound by their own SHA-256 into the run identity:
`rho-1` and `perplexity` both read `instruct-reference`'s step 940. Those two
arms also read a second, offline artifact -- a per-instance reference-loss
table computed once by `farmshare/score_reference.py` (see
`token_selection_370m/reference_scores.py`) -- resolved via
`resolve_reference_scores()` and bound the same way.

Outputs stay on FarmShare scratch and synchronously upload to W&B; only the
true final checkpoint uploads as a model artifact. A fresh run refuses a
non-empty checkpoint directory. `--resume` restores a local checkpoint, or a
completed run's final checkpoint through `WANDB_RESUME_ARTIFACT`, and
requires its schema-v3 fingerprint to match. Resume rejects a changed
dataset object order, dtype, row count, reference hash, BLADE secondary
stream, arm, or hyperparameter.

`sync_repo.sh` refuses to sync a dirty local tree and stamps the synced code
with the exact commit (`GIT_COMMIT`), which every run logs into its W&B
config and its `run_identity.json`.

## Checkpoints, evaluator, resume, and artifacts

The permanent ladder is step 0, every 125 steps, and the true final step,
omitting the last grid point when it is less than 125 steps before final.
The branch-local evaluator is self-contained: `.edullm/eval_task_loss_olmo_core.py`
carries the exact 20 `*_rc_5shot_bpb` labels. Training pauses until the
complete suite and awaited W&B uploads succeed. Partial evals and upload
failures do not advance `last_durable_step.json`.

Checkpoints, progress, metrics, selection state, BLADE state, and task-loss
JSON stay on FarmShare scratch. Token-selection EMA/history state is
callback-checkpointed. BLADE additionally checkpoints the dynamic reference
and its optimizer, both secondary stream cursors, completed step, and last
sync, so resume never re-runs, skips, or reorders a sync.

## Running the tests

```powershell
$env:PYTHONPATH="$PWD\src;$PWD\.edullm"
py -3 -m pytest .edullm\tests -q
```
