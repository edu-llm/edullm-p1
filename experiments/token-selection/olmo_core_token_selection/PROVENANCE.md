# Provenance of this directory

This is the code that produced every run reported in the paper. It is a
**copy**, vendored here so that the paper's "all code is available" claim is
true from this repository alone.

## Where it came from

| Field | Value |
| --- | --- |
| Upstream repo | `https://github.com/edu-llm/OLMo-core` |
| Branch | `edullm/token-selection-370m-unified` |
| Commit | `3ffb5175` (see below) |
| Source path | `.edullm/` |
| Copied on | 2026-09-28 |

This directory is byte-identical to `.edullm/` at that commit. No file was
edited here after vendoring.

`d294e419` unifies every arm onto one code path: one entrypoint
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
- **One new arm, trained in this study rather than read from elsewhere:**
  `instruct-reference`. `rho-1` and `perplexity` both read their reference
  losses from it: the final checkpoint (step 940, no averaging), scored
  offline once against the whole corpus (see A1 below). There is no HQ
  reference arm or checkpoint anywhere in this history.
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
- `eval_task_loss_olmo_core.py`: the single-rank (`world_size=1`) task-loss
  eval process group now uses `gloo` with a file-based rendezvous, not
  `nccl` with `env://`. Confirmed on FarmShare across two fixes: the 1-GPU
  smoke test first hung at `dist.init_process_group(backend="nccl")` --
  NCCL still probes IB/network topology on init even at `world_size=1`.
  Switching to `gloo` didn't fix it either: `env://` makes rank 0 bind a
  TCPStore server on a port that was only *probed* free a moment earlier
  (bound, read, released) in `_default_single_rank_env()`, and FarmShare
  doesn't let that bind-probe-release-rebind sequence proceed cleanly. A
  file-based store has no port to race on and no other rank to wait for.
  Production's real multi-rank eval (`world_size>1`, `task_loss_nproc=4`,
  `env://` under torchrun) is unaffected by either change.
- `token_selection_370m/arms.py`: BLADE's `rank_microbatch_tokens` drops
  from the 16,384 default to 8,192. Confirmed on FarmShare: BLADE's step-0
  pre_train sync OOM'd a 44 GiB L40S (39.47 GiB already in use, short by
  6.12 GiB) -- it holds the proxy, the dynamic reference, and both their
  optimizers resident during the K-update sync, exactly the case the
  original design called out as needing a fallback.
- `token_selection_370m/blade.py`: the K-update backward
  (`_mean_ce_with_weight`) now backprops through `output.loss`, not
  `output.ce_loss`. OLMo-core's LM head documents `ce_loss` as
  logging-only and unconditionally `.detach()`-es it; once the OOM above
  was fixed, the very first real K-update backward (this schedule's first
  sync is at step 0, so nothing before this fix had ever exercised the
  path end to end) failed with "element 0 of tensors does not require
  grad and does not have a grad_fn". `output.loss` is the live tensor and
  is numerically identical to `ce_loss` here, since this call never sets
  `z_loss_multiplier` (it defaults to `None` upstream) -- the fix changes
  nothing about what BLADE optimizes, only restores the gradient.

`3ffb5175` (on top of `90c2eb66`) reruns every arm under this one commit --
nothing is kept from `53daffdf` -- and makes these further changes:

- **Offline reference scoring (`token_selection_370m/reference_scores.py`,
  `farmshare/score_reference.py`).** The whole RegMix-10B corpus is scored
  once against the frozen `instruct-reference` checkpoint (step 940), on the
  1-GPU `qos=normal` lane, into a per-instance reference-CE table bound to
  that checkpoint's sha256 and the exact corpus by a manifest. `rho-1` and
  `perplexity` look up rows by `batch["index"]` instead of keeping a second
  reference model resident during training.
- **Per-instance random derivation and tie-breaking (`selection.py`).**
  `random-control`'s mask, and every score-based method's tie-break
  (including BLADE's), are drawn per corpus instance
  (`SeedSequence([tag, data_seed, index])`) -- independent of world size,
  rank, and microbatch composition, so seeds 42 and 69 never share a draw
  and a 1-GPU smoke run draws the same masks production would.
- **BLADE (`blade.py`):** the step-0 sync's reference LR is floored at its
  post-warmup value (it was training at LR 0, since the proxy's own
  scheduler is still in warmup at step 0); selection is now per row
  (`round(0.6*count)`, matching every other arm) instead of a per-rank batch
  threshold; the reference is scored under the same bf16-cast parameters the
  proxy trains under (via `torch.func.functional_call`, without touching the
  reference's own fp32 AdamW state), with a post-sync parity check logged
  after every sync.
- **Exact step count (`recipe.py`).** `Duration.steps(steps)`, not
  `Duration.tokens(max_tokens)` -- the latter rounds up, running one step
  past every ladder/eval/FLOP computation that assumes the floor.
- **All-token CE.** `SkipStepOptimizer`'s spike detector now watches the
  mean CE over all valid tokens, not the kept-token CE, so a discontinuity
  in the kept set alone (a BLADE resync, REL-EMA's growing alpha) can't look
  like a loss spike.
- **W&B flush before eval (`task_loss.py`).** Every rank flushes buffered
  train metrics before the eval logs at a fixed step; otherwise W&B silently
  drops the few train points just before every eval.
- **Strict eval loading (`eval_task_loss_olmo_core.py`).** `strict=True`,
  not a 5%-missing-keys tolerance -- both sides build the same class, so a
  missing or unexpected key is a real bug.
- **Keep every checkpoint (`checkpoint.py`, `task_loss.py`).** Pruning is
  disabled for these runs (`keep_all_checkpoints=True`), so every ladder
  checkpoint and its optimizer state stays on disk for later re-evaluation.
- **Environment recording.** torch/CUDA/driver/GPU and a pip-freeze hash are
  written to `run_identity.json` and the W&B config, kept out of the
  resume-blocking scientific identity so a driver bump can't refuse a
  resume.

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
