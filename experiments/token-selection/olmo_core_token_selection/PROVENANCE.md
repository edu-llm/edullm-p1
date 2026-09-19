# Provenance of this directory

This is the code that actually produced the runs reported in the paper. It is a
**copy**, vendored here so that the paper's "all code is available" claim is true
from this repository alone.

## Where it came from

| Field | Value |
| --- | --- |
| Upstream repo | `https://github.com/edu-llm/OLMo-core` |
| Branch | `edullm/token-selection-370m` |
| Base commit | `98ea67c948fd93ccbfd2633e1ae818c3ae1d2ad7` |
| Source path | `.edullm/` |
| State copied | the **uncommitted working tree**, not the base commit |
| Copied on | 2026-09-19 |

## Why the base commit is not enough

The arm YAMLs in the sibling directories pin `revision: 98ea67c9…`, but that
commit **cannot** have produced four of the seven reported arms. Diffing the
working tree against the branch tip shows:

- `token_selection_370m/arms.py` at `98ea67c9` contains no `full-loss-control`
  and no `random-control` ArmSpec at all. Both arms exist only in the working
  tree. The reported W&B runs `full-loss-control-regmix10b-v3` and
  `random-control-regmix10b-v1` carry exactly the working-tree run ids.
- `token_selection_370m/selection.py` at `98ea67c9` computes REL-EMA as
  `per_row_topk(history - current, …)` — the **inverted** polarity. The working
  tree computes `per_row_topk(current - history, …)`, which is the
  current-minus-history convention the paper describes and which the
  `rel-ema-exp/README.md` records as corrected on 2026-09-05.
- `arms.py` at `98ea67c9` sets `REFHQ = "pretrain/refhq-regmix-5p5b"` (the HQ
  corpus). The working tree sets `REFHQ = "pretrain/refhq-instruct"`, which is
  what the paper states BLADE's dynamic reference was trained against, and the
  reported run id is `blade-regmix10b-refhq-instruct-v3-v1` — again the
  working-tree value.

So the vendored copy here, not the pinned revision, is the auditable record.
The upstream branch should still be committed and pushed; until it is, this
directory is the only published form of that code.

## What is included, and why

Only code that ran to produce reported results:

| Path | Role |
| --- | --- |
| `token_selection_entrypoint.py` | Entrypoint for the FarmShare runs (full-loss control v3, random control seeds 42/69, REL-EMA) |
| `token_selection_370m/arms.py` | Arm definitions: run ids, keep rates, reference checkpoints |
| `token_selection_370m/selection.py` | The scoring and masking rules for every method |
| `token_selection_370m/recipe.py` | Model, optimizer, data recipe; `INIT_SEED = 6198`, `DATA_SEED = 42` |
| `token_selection_370m/train_module.py` | Token-weighted train module (mean over kept tokens) |
| `token_selection_370m/blade.py` | BLADE dynamic-reference sync and K-update loop |
| `token_selection_370m/precomputed.py` | Precomputed-mask path used by the Perplexity arm |
| `production_contract/checkpoint.py` | Permanent-checkpoint ladder and resume durability |
| `production_contract/task_loss.py` | Task-loss eval callback fired on each permanent save |
| `production_contract/wandb_artifacts.py` | W&B artifact upload |
| `eval_task_loss_olmo_core.py` | The 20-label OLMES evaluator behind every reported bpb number |
| `runpod/entrypoint.py`, `runpod/stage_inputs.py` | RunPod path used for the August arms (RHO-1, Attention, Perplexity, BLADE) |
| `runpod/precompute_middle_ppl_masks.py` | Offline mask precompute for the Perplexity arm |
| `runpod/bootstrap.sh`, `runpod/launch.sh` | RunPod launch scaffolding |
| `farmshare/*` | FarmShare staging and Slurm launch path for the September runs |
| `Dockerfile`, `requirements-token-selection-eval.txt` | Runtime the RunPod arms were built against |

## What is deliberately excluded

- `train_on_corpus.py` — the general-purpose corpus trainer. The reported runs
  do **not** go through it; they go through `token_selection_entrypoint.py`,
  which has different checkpoint/resume behavior.
- `tests/` — unit tests for the selection and contract logic. Useful, but they
  did not produce any reported number.
- `platform/`, `fixtures/` — eduLLM submission adapters and fixtures. The
  reported runs were launched directly on RunPod and FarmShare, not through the
  platform submission path.
- `runpod/probe_rho_fsdp.py`, `rehearsal.md` — debugging and scratch material.
- The OLMo-core library itself (`src/`). Pin it from the upstream branch above.

## Caveat

Because this is a copy of an uncommitted working tree, it has no upstream commit
SHA of its own, and the reported W&B runs logged `git_commit=None`. The
correspondence between this code and those runs rests on the run ids, arm
configuration, and reference paths lining up, as set out above — not on a
recorded commit hash.
