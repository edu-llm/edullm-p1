# RegMix 10B

**S3 (private project store, not a public download):** `s3://edullm-datasets/regmix/regmix-10b/`

Despite the `regmix` name, the domain weights are the Data Mixing Laws weights of paper
Table 4 (see `plan_regmix_mix.py`), applied to a subsample of `allenai/olmo-mix-1124`
(ODC-By v1.0; DCLM also subject to the Common Crawl terms of use). We do not redistribute
the corpus; rebuild it byte for byte with
[`datasets/manifests/regmix-10b-v1/`](../../../../datasets/manifests/regmix-10b-v1/).

**Entry:** `submit_regmix_mix.sh` (source: olmohq pool on S3 or local mirror)

**AWS data prep:** `prepare_regmix_data.py`

**Publish:** `publish_regmix_edullm_data.py` (called from `submit_publish_regmix_edullm_data.sh`) stages the trimmed, tokenized shards and calls `edullm_data.publish()` under `pretrain/regmix-10b`.

The source repo's `regmix/` also holds a second, unrelated job graph — per-document difficulty labels (compression ratio, Flesch, MTLD) — for a different paper. That graph reads the finished `pretrain/regmix-10b` corpus but never writes it, so it isn't part of this pipeline; see `../README.md` for why it's excluded here.
