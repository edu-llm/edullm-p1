# Provenance of this directory

This is the code that produced every run reported in the paper. It is a
**copy**, vendored here so that the paper's "all code is available" claim is
true from this repository alone.

## Where it came from

| Field | Value |
| --- | --- |
| Upstream repo | `https://github.com/edu-llm/OLMo-core` |
| Branch | `edullm/token-selection-370m-unified` |
| Commit | `cc86ab96` (see below) |
| Source path | `.edullm/` |
| Copied on | 2026-09-28 |

This directory is byte-identical to `.edullm/` at that commit. No file was
edited here after vendoring.

`cc86ab96` unifies every arm onto one code path: one entrypoint
(`token_selection_entrypoint.py`), one train module
(`TokenWeightedTrainModule`), and one hardware contract (FarmShare, 4×L40S).
It replaces `53daffdf`, the commit the previous version of this directory
was vendored from, which still had four arms running through a separate
RunPod path on 8×A100.

## What changed, and why

The full history is in the branch's commits, but the paper-relevant changes
are:

- **No RunPod path, no AWS/S3 code.** `runpod/`, `train_on_corpus.py`,
  `precomputed.py` and `Dockerfile` are deleted. Corpora are resolved from a
  local manifest (`farmshare/stage_local.py`) bound to the pinned FarmShare
  directories verified against this repository's
  `datasets/manifests/*/outputs.json`.
- **Two new arms, trained in this study rather than read from elsewhere:**
  `hq-reference` and `instruct-reference`. `rho-1` and `perplexity` read
  their frozen references from these arms' own checkpoints.
- **`random-control-seed69`** is now an ArmSpec (it previously ran from an
  uncommitted hand edit; its seeds are confirmed against its W&B
  `run_identity.json` artifact).
- **Attention** (`selection.py`): the raw causal column-mass score is
  normalized by its expectation under uniform attention (removing the bias
  toward early positions) and aligned to the token whose loss it gates
  (`get_labels` shifts labels left by one). See
  `aligned_normalized_attention_scores`.
- **BLADE** (`blade.py`): the reference's own training term is now
  selection-weighted (Wang et al. 2026, Sec. 2.2), scored against the
  *outgoing* reference before it is overwritten. The schedule moved to syncs
  at steps 0/400/800/1200/1600/2000 (`tau=400`, `K=75`), so every arm selects
  from step 0 at the same 60% budget. Each sync gets a fresh optimizer at the
  proxy's own scheduled LR.
- Every arm, including `full-loss-control` and `blade`, now runs through the
  same `TokenWeightedTrainModule`; there is no separate stock-module branch.
- `sync_repo.sh` refuses to sync a dirty tree and stamps the synced code with
  its exact commit (`GIT_COMMIT`), which every run logs into its W&B config
  and `run_identity.json`, closing the previous version's `git_commit=None`
  gap.
- `farmshare/{config.env,launch.sh,submit_from_laptop.sh}`: a smoke test can
  now override `TRAIN_GPUS`, `TRAIN_PARTITION`/`TRAIN_QOS`, `EDULLM_LOCAL`
  and `WANDB_MODE` to run one arm on the separate `qos=normal` 1-GPU cap with
  offline W&B, instead of the production `qos=gpu` 4-GPU cap. Every
  production launch leaves all of these unset and gets the same behavior as
  before.

## Kept runs: what code they ran

Three runs are kept from before this unification and were **not** rerun:
the two random-control seeds (`random-control-regmix10b-v1`,
`random-control-regmix10b-seed69-v1`) and REL-EMA
(`rel-ema-exp-10b-scratch-v1`). All three ran `53daffdf` (the previous
commit), on FarmShare 4×L40S, through the RunPod-wrapper entrypoint that
`token_selection_entrypoint.py` has since absorbed. The random-selection and
REL-EMA code paths are unchanged in `cc86ab96`: no rank term was added to the
mask seed, and the EMA update is untouched. This is protected by a golden
test (`tests/test_token_selection_370m.py`) that checks the per-microbatch
weight derivation, `EMAHistory`'s update sequence, and `ema_alpha` against
hashes recorded from `53daffdf`, plus a GPU replay of each kept run's first
few steps against its own W&B history before any rerun began.

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
| `production_contract/checkpoint.py` | Permanent-checkpoint ladder and resume durability |
| `production_contract/task_loss.py` | Task-loss eval callback fired on each permanent save |
| `production_contract/wandb_artifacts.py` | W&B artifact upload |
| `eval_task_loss_olmo_core.py` | The 20-label OLMES evaluator behind every reported bpb number |
| `farmshare/*` | The FarmShare staging and Slurm launch path every run used |
| `requirements-token-selection-eval.txt` | Evaluator runtime pins |

## What is deliberately excluded

- `tests/` (from the fork; the copy here is included for completeness, not
  because it ran to produce a number) — kept as the record that the unified
  code was tested before the reruns launched.
- The OLMo-core library itself (`src/`). Pin it from the upstream branch and
  commit above.

## Caveat

The correspondence between this code and the reported runs rests on
`GIT_COMMIT` (logged in every run's W&B config, stamped by `sync_repo.sh` at
sync time from the exact commit that was synced), not on a commit hash
recorded by some earlier, unrelated mechanism.
