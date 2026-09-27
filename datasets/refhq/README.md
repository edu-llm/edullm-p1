# RefHQ 5.5B

**S3 (private project store, not a public download):** `s3://edullm-datasets/refhq/refhq-regmix-5p5b-v1/`

**Entry:** `scripts/submit_refhq_regmix_5p5.sh` → `scripts/submit_refhq_tokenize.sh`

**Rebuild:** we do not redistribute this corpus; rebuild it byte for byte from public data
with [`../manifests/refhq-regmix-5p5b-v1/`](../manifests/refhq-regmix-5p5b-v1/).

HQ-filtered domain pulls from Hugging Face (see `scripts/hq_reference_sources.py`): the
training corpus's seven domains, but only arXiv comes from `allenai/olmo-mix-1124`; the
other six come from their original public releases, each under its own terms
(StarCoderData is gated under The Stack's terms of use; Wikipedia is CC-BY-SA 3.0 and
GFDL). Stricter filters apply to DCLM, StarCoder, OpenWebMath and Algebraic Stack (paper
Table 5). Python package: `refhq` (`import refhq`; add repo `datasets/` to `PYTHONPATH`).

**Tests:** `pytest datasets/refhq/tests/`
