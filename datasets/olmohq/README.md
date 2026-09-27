# OLMo HQ ~100B+ upsample pool

**Entry:** `submit_olmo_mix_upsample.sh` → `submit_olmo_pool_tokenize.sh` (in `../olmo/`)

Reuses DCLM shards from the completed 30B run; upsamples other domains from `allenai/olmo-mix-1124`.

## Availability note (plan vs measured)

`plan_olmohq_topup.py`'s `PLANNED_AVAILABLE` is the **planned** inventory. Dolma2-measured
totals initially under-shot that plan for **starcoder** and **pes2o**. Use:

```bash
SRC_RUN_DIR=<tokenized run dir> bash datasets/olmohq/submit_olmohq_topup.sh
```

to append additional HF shards, then (if the empirical rate overshoots) trim the
**active** manifests with `trim_olmohq_topup_manifest.py` until
`|plan − measured| / measured ≤ 10%` for every domain. This is **append-only** on
local scratch and **never writes** `regmix/regmix-10b/`.

As of 2026-07-29 all seven domains satisfy the 10% gate
(`plan/availability_after_topup.json`). Active measured totals: starcoder **18.541B**,
pes2o **26.379B**. Excess topup shards may remain on local scratch but are
excluded from `plan/tokenized_manifest.json`.

## Published reservoir

`publish_olmohq_edullm_data.py` laid out this pool as the ~127B-token reservoir
`pretrain/olmo-127b` v1 that the domain-weighting paper trains on. Its source
documents are not redistributed; rebuild it byte for byte from the public
`allenai/olmo-mix-1124` files with
[`../manifests/olmo-127b-v1/`](../manifests/olmo-127b-v1/README.md) (`rebuild.py`).
