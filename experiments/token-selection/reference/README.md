# Reference arm — RefHQ RegMix 5.5B CE

Plain CE on the 5.5B-token HQ reference corpus (`pretrain/refhq-regmix-5p5b`, in the
project's private `s3://edullm-data/` store) with the RefHQ-matched OLMo-2 370M stack.
Produces the reference ladder consumed by downstream token-selection arms (via DistCP →
`export_refhq_reference.py`); the Perplexity arm scores against the weight average of its
steps 1000, 1125 and 1315 checkpoints.

The corpus has the training corpus's seven domains, but only arXiv comes from
OLMo-mix-1124; the other six come from their original public releases, each under its own
terms (StarCoderData is gated under The Stack's terms of use; Wikipedia is CC-BY-SA 3.0 and
GFDL). It is not redistributed; rebuild it byte for byte with
[`../../../datasets/manifests/refhq-regmix-5p5b-v1/`](../../../datasets/manifests/refhq-regmix-5p5b-v1/).

| Knob | Value |
|------|-------|
| Architecture | `TransformerConfig.olmo2_370M` (full attn) |
| GBS / seq / LR | `4_194_304` / 2048 / `4e-4` (warmup 24, `alpha_f=0.1`) |
| Token budget | the reported run's final step is **1315** (the last checkpoint the Perplexity arm averages). This script's default budget, the 5,509,020,202 published train tokens, gives 1313 steps (`5_509_020_202 // 4_194_304`), so it does not by itself reproduce that step count |
| Dataset | `pretrain/refhq-regmix-5p5b` (`s3://edullm-data/`, latest validated) |
| Artifact durability | Runtime scratch + W&B (production online fail-closed) |

## Files

| File | Role |
|------|------|
| [`prepare_refhq_data.py`](prepare_refhq_data.py) | Resolve + stage edullm-data shards → path list |
| [`train_olmo3_370m_refhq.py`](train_olmo3_370m_refhq.py) | Trainer (1..N GPU via `torchrun`) |
| [`launch_train.sh`](launch_train.sh) | Stage inputs and launch training |
| [`export_refhq_reference.py`](export_refhq_reference.py) | DistCP → flat `model.pt` for other arms |

## Export flat reference weights

Example read-only bootstrap DistCP URI:

`s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-5p5b/checkpoints/step1315/`

```bash
python experiments/token-selection/reference/export_refhq_reference.py \
  --s3-uri s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-5p5b/checkpoints/step1315/ \
  --work-dir /scratch/refhq-export \
  --output /scratch/refhq_step1315.pt
```

## Ephemeral runtime

Scratch starts empty and may be wiped when the job ends.

- **Allowed:** stage shards from `s3://edullm-data` into scratch for the job.
- **Forbidden:** assuming FarmShare/laptop corpora, legacy `s3://edullm-datasets/`,
  or persistent local checkpoints already on disk.
- **Artifacts:** permanent DistCP steps and progress remain on scratch and
  upload to W&B. Production online checkpoint uploads fail closed.

Resume with `WANDB_RESUME_ARTIFACT` into a fresh `SAVE_FOLDER`, or pass
`--load-path` to a local DistCP dir.

## Prepare (optional; trainer can stage itself)

```bash
python experiments/token-selection/reference/prepare_refhq_data.py \
  --work /scratch/refhq
# → tokenized/paths_train.txt, length_tokens.txt, refhq_data_summary.json
```

## Launch

```bash
# Clean ephemeral node: stage inputs + train + W&B artifacts
STAGE_DIR=/scratch/staged \
SAVE_FOLDER=/scratch/ckpts/refhq-regmix-5p5b-v1 \
PROGRESS_DIR=/scratch/progress/refhq-regmix-5p5b-v1 \
NPROC=4 \
bash experiments/token-selection/reference/launch_train.sh

# Or via prepare:
REFHQ_WORK=/scratch/refhq \
SAVE_FOLDER=/scratch/ckpts/refhq-regmix-5p5b-v1 \
PROGRESS_DIR=/scratch/progress/refhq-regmix-5p5b-v1 \
bash experiments/token-selection/reference/launch_train.sh

# Train from s3:// URIs (no local stage):
STAGE_DIR= \
SAVE_FOLDER=/scratch/ckpts/refhq-regmix-5p5b-v1 \
PROGRESS_DIR=/scratch/progress/refhq-regmix-5p5b-v1 \
bash experiments/token-selection/reference/launch_train.sh
```

Requires the `edullm-data` package and AWS read access to `s3://edullm-data`.
No S3 artifact-write permission is needed.
