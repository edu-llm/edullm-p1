# Provenance of this directory

This is the code that actually produced the runs reported in the paper. It is a
**copy**, vendored here so that the paper's "all code is available" claim is true
from this repository alone.

## Where it came from

| Field | Value |
| --- | --- |
| Upstream repo | `https://github.com/edu-llm/OLMo-core` |
| Branch | `edullm/token-selection-370m` |
| Commit | `53daffdf66d07f617e14b17beff3be89cafdc95d` |
| Source path | `.edullm/` |
| Copied on | 2026-09-19 |

This directory is byte-identical to `.edullm/` at that commit, for every file
listed below.

At the time the study ran, this code existed only as an uncommitted working tree
on one laptop. It was committed upstream as `bf087c8f` ("Commit the
token-selection tree that produced the reported runs") and merged with the two
commits that had landed on the branch meanwhile, giving `53daffdf`. Cite
`53daffdf` for reproduction.

## Why the previously pinned commit is not enough

The arm YAMLs in the sibling directories pin `revision: 98ea67c9…`, but that
commit **cannot** have produced four of the seven reported arms. Diffing the
code that ran against that older tip shows:

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

So `53daffdf`, not the `98ea67c9` revision the YAMLs pin, is the auditable
record. The YAML `revision:` fields are stale and should be read as historical.

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

The code now has an upstream commit, but it was committed *after* the study ran,
and the reported W&B runs logged `git_commit=None`. So the correspondence
between this code and those runs rests on the run ids, arm configuration, and
reference paths lining up, as set out above — not on a commit hash recorded at
training time.

One asymmetry worth knowing: `runpod/launch.sh` defaults
`OLMO_CORE_CHECKPOINT_SKIP_FDATASYNC=1`, so the four RunPod arms (RHO-1,
Attention, Perplexity, BLADE) wrote checkpoints without a per-file `fdatasync`.
The FarmShare arms (full-loss control, both random-control seeds, REL-EMA) do
not set that flag and synced normally. This affects crash durability, not the
contents of any checkpoint that was successfully read back and evaluated.
