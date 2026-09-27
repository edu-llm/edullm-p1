# RegMix 10B

Despite the `regmix` name, the domain weights are the Data Mixing Laws weights of the token
selection paper's Table 4 (see `plan_regmix_mix.py`), applied to a subsample of
`allenai/olmo-mix-1124` (ODC-By v1.0; DCLM also subject to the Common Crawl terms of use).
We do not redistribute the corpus; rebuild it byte for byte with
[`../manifests/regmix-10b-v1/`](../manifests/regmix-10b-v1/).

**Entry:** `submit_regmix_mix.sh` (source: a local olmohq run's pool, via `SRC_RUN_DIR`)

**Difficulty labels and their upload scripts have since been removed** (stale, not used by
any current corpus); they are kept in git history.
