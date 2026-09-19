# RefHQ 5.5B

**S3:** `s3://edullm-datasets/refhq/refhq-regmix-5p5b-v1/`

**Entry:** `scripts/submit_refhq_regmix_5p5.sh` → `scripts/submit_refhq_tokenize.sh`

HQ-filtered domain pulls from Hugging Face (see `scripts/hq_reference_sources.py`). Python package: `refhq` (`import refhq`; add this vendored `datasets/` to `PYTHONPATH`).

(The original `tests/` for this package are not vendored here; see `../README.md` for what was left out and why.)
