# Dataset pipelines

Build scripts for the project's corpora. The S3 buckets below are the project's private
stores, not public download locations.

| Directory | S3 bucket / prefix | Description |
|-----------|-------------------|-------------|
| [`olmo/`](olmo/) | `edullm-datasets/olmo30b/olmo-mix-1124-30b/` | ~30B trimmed sample from `allenai/olmo-mix-1124` |
| [`olmohq/`](olmohq/) | `edullm-datasets/olmo100b/olmo-mix-1124-30b/` | ~100B upsampled / rebalanced pool (feeds `regmix/`) |
| [`regmix/`](regmix/) | `edullm-datasets/regmix/regmix-10b/` | 10B mix from olmohq with Data Mixing Laws domain weights (token-selection Appendix A Tables 3-4; the `regmix` name is historical) |
| [`refhq/`](refhq/) | `edullm-datasets/refhq/refhq-regmix-5p5b-v1/` | 5.5B HQ-filtered reference corpus: the same seven domains, arXiv from `allenai/olmo-mix-1124` and the other six from their original HF releases (token-selection Appendix A Table 5) |
| [`refhq_new/`](refhq_new/) | `edullm-data pretrain/refhq-instruct` | ~3.9B instruction-sourced reference corpus (token-selection Appendix A Table 6) |

## Data availability: rebuilding the corpora

We do not redistribute these datasets or our derived subsets. Every corpus can be rebuilt
byte for byte from its public Hugging Face sources with the document manifests in
[`manifests/`](manifests/), each with a `rebuild.py` that checks every output's sha256:

| Manifest | Corpus |
|----------|--------|
| [`manifests/regmix-10b-v1/`](manifests/regmix-10b-v1/) | the token selection paper's 10B training corpus (`pretrain/regmix-10b` v1) |
| [`manifests/refhq-regmix-5p5b-v1/`](manifests/refhq-regmix-5p5b-v1/) | its 5.5B HQ reference corpus (StarCoder needs a Hugging Face token: `bigcode/starcoderdata` is gated under The Stack's terms of use) |
| [`manifests/refhq-new-v1/`](manifests/refhq-new-v1/) | its ~3.9B Instruct reference corpus |
| [`manifests/olmo-127b-v1/`](manifests/olmo-127b-v1/) | the Domain weighting paper's 127B reservoir (`pretrain/olmo-127b` v1) |

Each source keeps its own license: OLMo-mix-1124 is ODC-By v1.0 (DCLM also subject to the
Common Crawl terms of use), Wikipedia is CC-BY-SA 3.0 and GFDL, and the instruction
mixtures' terms are listed in [`refhq_new/DATASET-DESIGN.md`](refhq_new/DATASET-DESIGN.md).

## Shared dataset utilities (this directory)

Scripts used by more than one corpus pipeline live here (not under `olmo/`, `regmix/`, etc.):

| File | Purpose |
|------|---------|
| `olmo_shard_utils.py` | OLMo-mix shard I/O, domain token totals, doc materialization |
| `trim_olmo_overshoot.py` | Trim one overshot OLMo domain to a token budget |
| `trim_and_tokenize_regmix.py` | Trim one domain and emit dolma2 uint32 `.npy` memmaps |

The AWS-credential helpers these pipelines once sourced and the pure S3 transfer workers
have been removed, along with the calls to the credential helpers. The scripts are kept as a
record of how each corpus was built; their S3 reads and writes now use the standard AWS
credential chain.
