# RegMix 10B

**S3 (private project store, not a public download):** `s3://edullm-datasets/regmix/regmix-10b/`

Despite the `regmix` name, the domain weights are the Data Mixing Laws weights of the token
selection paper's Table 4 (see `plan_regmix_mix.py`), applied to a subsample of
`allenai/olmo-mix-1124` (ODC-By v1.0; DCLM also subject to the Common Crawl terms of use).
We do not redistribute the corpus; rebuild it byte for byte with
[`../manifests/regmix-10b-v1/`](../manifests/regmix-10b-v1/).

**Entry:** `submit_regmix_mix.sh` (source: olmohq pool on S3 or local mirror)

**AWS data prep:** `prepare_regmix_data.py`

**Difficulty labels (FarmShare):** `label_regmix_shard.sbatch` (one array task per
manifest row, from `build_regmix_label_manifest.py`) — compression ratio, Flesch reading ease, and MTLD on the seven trimmed domain shards under `trim/<domain>/`. Writes `RUN_DIR/labels/` (`READY`, `docs/`, `metrics/`, `metrics_index.jsonl.gz`).

**Label upload to S3** (upload scripts since removed):

- `labels/` → `s3://edullm-datasets/regmix/regmix-10b/labels/`
- Receipt: `RUN_DIR/labels_upload_manifest.json` (also copied to the corpus prefix)
