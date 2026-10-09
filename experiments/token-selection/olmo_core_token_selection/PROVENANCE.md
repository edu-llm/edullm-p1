# Provenance of this directory

This is the code that produced every run reported in the paper. It is a
**copy**, vendored here so that the paper's "all code is available" claim is
true from this repository alone.

## Where it came from

| Field | Value |
| --- | --- |
| Upstream repo | `https://github.com/edu-llm/OLMo-core` |
| Branch | `edullm/token-selection-370m-unified` |
| Commit | `64c28145` |
| Source path | `.edullm/` |

Apart from this file, the directory is byte-identical to `.edullm/` at that
commit. No vendored file was edited here.

## Which commit each run recorded

Every run logs `GIT_COMMIT` into its W&B config and `run_identity.json`;
`farmshare/sync_repo.sh` refuses to sync a dirty tree and stamps the synced code
with its exact commit.

- The eight reported arms (`full-loss-control`, `random-control`,
  `random-control-seed69`, `rho-1`, `perplexity`, `attention`, `blade`,
  `rel-ema-exp`) all record `64c28145`.
- `instruct-reference` records `765ae838`, the commit the shared run directory
  held when it started. Its method (full loss) is untouched by every later
  commit, and its checkpoint ladder (step 0, every 125 steps, final step 940,
  omitting 875) is the same under `64c28145`.

## What the code does

Every arm runs through one entrypoint (`token_selection_entrypoint.py`), one
train module (`TokenWeightedTrainModule`, the mean CE over kept or weighted
tokens) and one hardware contract (FarmShare, 4×L40S). Corpora are resolved
from a local manifest (`farmshare/stage_local.py`) bound to the pinned
FarmShare directories verified against this repository's
`datasets/manifests/*/outputs.json`.

- **Reference model.** `instruct-reference` is trained in this study. `rho-1`
  and `perplexity` both read their reference losses from its final checkpoint
  (step 940, no averaging).
- **Offline reference scoring** (`token_selection_370m/reference_scores.py`,
  `farmshare/score_reference.py`). The whole RegMix-10B corpus is scored once
  against that checkpoint, on the 1-GPU `qos=normal` lane, into a
  per-instance reference-CE table bound to the checkpoint's sha256 and the
  exact corpus by a manifest. `rho-1` and `perplexity` look up rows by
  `batch["index"]` instead of keeping a reference model resident.
- **Per-instance random draws and tie-breaking** (`selection.py`).
  `random-control`'s mask, and every score-based method's tie-break
  (including BLADE's), are drawn per corpus instance
  (`SeedSequence([tag, data_seed, index])`), independent of world size, rank
  and microbatch composition, so seeds 42 and 69 never share a draw.
- **Attention** (`selection.py`). Each token's score is the causal attention it
  receives on the last block, aligned to the token whose loss it gates
  (`get_labels` shifts labels left by one), and z-scored against the mean and
  standard deviation of tokens at its own position from the model's own
  immediately preceding training step (`AttentionPositionBaseline`, no extra
  forward pass, no smoothing), with the uniform-attention prior used only on a
  run's first step. Trained attention is recency-biased, so a uniform-attention
  normalizer alone leaves the score position-confounded; the pre-production
  diagnostic `farmshare/attention_diagnostic.py`, run against a trained
  checkpoint, puts every position bin between 56.8% and 63.6% keep rate under
  this score.
- **BLADE** (`blade.py`). Syncs at steps 0/400/800/1200/1600/2000 (`tau=400`,
  `K=75`), so it selects from step 0 at the same 60% budget as every other arm.
  The reference's own training term is selection-weighted (Wang et al. 2026,
  Sec. 2.2), scored against the *outgoing* reference before it is overwritten.
  Each sync gets a fresh optimizer at the proxy's own scheduled LR, floored at
  its post-warmup value so the step-0 sync does not train at LR 0. Selection is
  per row (`round(0.6*count)`), like every other arm. The reference is scored
  under the same bf16-cast parameters the proxy trains under (via
  `torch.func.functional_call`), with a parity check logged after every sync.
  The K-update backward goes through `output.loss` (OLMo-core detaches
  `output.ce_loss`; the two are numerically identical here because no z-loss is
  set). BLADE's rank microbatch is 8,192 tokens, since it holds the proxy, the
  dynamic reference and both optimizers during a sync and 16,384 OOMs a 44 GiB
  L40S.
- **Exact step count** (`recipe.py`). `Duration.steps(steps)`, so every run
  stops at exactly 2360 (or 940) steps.
- **All-token CE for the spike detector.** `SkipStepOptimizer` watches the mean
  CE over all valid tokens, so a discontinuity in the kept set alone (a BLADE
  resync, REL-EMA's growing alpha) cannot look like a loss spike.
- **Checkpoints and eval** (`production_contract/`). Permanent checkpoints at
  step 0, every 125 steps and the final step, omitting the last 125-grid point
  only when it is within 100 steps of the final step (so the 2360-step arms keep
  step 2250). Every checkpoint is kept with its optimizer state. The 20-label
  task-loss eval (`eval_task_loss_olmo_core.py`, `strict=True` loading) fires on
  each permanent save, after every rank flushes its buffered train metrics to
  W&B.
- **Environment recording.** torch/CUDA/driver/GPU and a pip-freeze hash are
  written to `run_identity.json` and the W&B config, outside the
  resume-blocking scientific identity.
- **1-GPU lane.** `farmshare/{config.env,launch.sh,submit_from_laptop.sh}` let
  a run override `TRAIN_GPUS`, `TRAIN_PARTITION`/`TRAIN_QOS`, `EDULLM_LOCAL` and
  `WANDB_MODE` to use FarmShare's 1-GPU `qos=normal` cap (used for the
  reference model and the offline scoring pass); a single-rank eval uses `gloo`
  with a file-based rendezvous.

## What is included, and why

Only code that produced a reported result:

| Path | Role |
| --- | --- |
| `token_selection_entrypoint.py` | The one entrypoint every arm runs through |
| `token_selection_370m/arms.py` | Arm definitions: run IDs, seeds, keep rates, reference contracts |
| `token_selection_370m/selection.py` | The scoring and masking rules for every method |
| `token_selection_370m/recipe.py` | Model, optimizer, data recipe |
| `token_selection_370m/train_module.py` | The one train module (mean over kept/weighted tokens) |
| `token_selection_370m/blade.py` | BLADE's dynamic-reference sync, selection-weighted K-updates, and resume |
| `token_selection_370m/reference_scores.py` | Schema and lookup for the offline per-instance reference-CE table |
| `production_contract/checkpoint.py` | Permanent-checkpoint ladder and resume durability |
| `production_contract/task_loss.py` | Task-loss eval callback fired on each permanent save |
| `production_contract/wandb_artifacts.py` | W&B artifact upload |
| `eval_task_loss_olmo_core.py` | The 20-label OLMES evaluator behind every reported bpb number |
| `farmshare/*` | The FarmShare staging and Slurm launch path every run used |
| `requirements-token-selection-eval.txt` | Evaluator runtime pins |
| `tests/` | The fork's tests for this code |

The OLMo-core library itself (`src/`) is not copied; pin it from the upstream
branch and commit above.
