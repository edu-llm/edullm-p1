# Dataset pipelines

Build scripts for the project's corpora, kept as the record of how each corpus was
filtered and selected. Every corpus is rebuilt byte for byte from public sources with
the manifests in [`manifests/`](manifests/) below — these pipeline scripts read only
local FarmShare scratch and public Hugging Face sources; nothing here reads or writes a
private object store.

| Directory | Description |
|-----------|-------------|
| [`olmo/`](olmo/) | Samples a ~30B mix across all seven domains from `allenai/olmo-mix-1124`. Only its DCLM shards are reused downstream (by `olmohq/`, and from there by `regmix/` and the Domain-weighting paper's `olmo-127b-v1` reservoir); the other six domains are resampled at `olmohq/`'s larger scale. |
| [`olmohq/`](olmohq/) | The ~100B–127B reservoir: reuses `olmo/`'s DCLM shards, upsamples/rebalances the other six domains, and tops up starcoder/pes2o. Feeds `regmix/`, and is itself published as the Domain-weighting paper's `pretrain/olmo-127b` reservoir. |
| [`regmix/`](regmix/) | 10B mix sampled from `olmohq/`'s pool with Data Mixing Laws domain weights (token-selection Appendix A Tables 3-4; the `regmix` name is historical) |
| [`refhq/`](refhq/) | 5.5B HQ-filtered reference corpus: the same seven domains, arXiv from `allenai/olmo-mix-1124` and the other six from their original HF releases (token-selection Appendix A Table 5) |
| [`refhq_new/`](refhq_new/) | ~3.9B instruction-sourced reference corpus, published as `pretrain/refhq-instruct` (token-selection Appendix A Table 6) |

## Data availability: rebuilding the corpora

We do not redistribute these datasets or our derived subsets. Every corpus can be rebuilt
byte for byte from its public Hugging Face sources with the document manifests in
[`manifests/`](manifests/), each with a `rebuild.py` that checks every output's sha256 —
this, not the pipeline scripts above, is the reproducible rebuild route:

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
| `trim_and_tokenize_regmix.py` | Trim one domain and emit dolma2 uint32 `.npy` memmaps |
