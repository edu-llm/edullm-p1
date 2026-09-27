# OLMo-mix ~30B

**Entry:** `submit_olmo_mix_sample.sh` → `submit_olmo_pool_tokenize.sh`

**HF source:** `allenai/olmo-mix-1124` (seed 42, stratified sample)

Only this sample's DCLM shards are reused downstream (by `../olmohq/`); the other six
domains are resampled at `olmohq/`'s larger scale. See [`../README.md`](../README.md).
