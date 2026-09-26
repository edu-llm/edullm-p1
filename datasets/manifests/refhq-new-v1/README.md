# refhq-new-v1 document manifest

Corpus id: `refhq-new-v1` — the ~3.9B-token Instruct reference corpus of the
token-selection paper, published internally as `pretrain/refhq-instruct`
(profile `pretrain-tokens/v1` + `vendored/v1`; working store
`s3://edullm-datasets/refhq/refhq-new/`, promoted to
`s3://edullm-data/pretrain/refhq-instruct/v3/`). The Instruct reference model
trained on it; its checkpoints scored the RHO-1 arm, and BLADE's reference
updates also drew from this corpus.

Built by `experiments/token-selection/datasets/refhq_new/scripts/` (see
`DATASET-DESIGN.md` there) and run on FarmShare 2026-08-03/04 under
`/scratch/users/nzhao2/refhq-new-v1/` (read-only build dir; raw/normalized
text was deleted after tokenization — only `manifests/*.json`,
`tokenized/<source>/<domain>/{train,val}.parts/*.npy(+.json)` and merged
`{train,val}.npy` survive; ~2,690 log files also survive).

## What trained on it

Six public HF sources, metadata-filtered, Dolma English-filtered (score
>= 0.5), 0.15%-per-`(source,domain)` document holdout (seed 42, before
tokenize), tokenized with `allenai/dolma2-tokenizer` (EOS id 100257), one
example = one document (chat turns flattened to `"Role: content"` lines):

| source | HF repo @ pinned revision | stream tokens (train+val, incl. EOS) |
|---|---|---|
| tulu-v2 | `allenai/tulu-v2-sft-mixture` @ `6248b175d2ccb5ec7c4aeb22e6d8ee3b21b2c752` | 208,767,746 |
| openhermes-25 | `teknium/OpenHermes-2.5` @ `b82037821055c377bed0d495e72e46de3bc72e84` | 343,948,449 |
| tulu-3 | `allenai/tulu-3-sft-mixture` @ `b14afda60f1bbebe55d5d2fa1e4df5042f97f8be` | 538,071,884 |
| hermes-3 | `NousResearch/Hermes-3-Dataset` @ `b1fddbdcae4e6714889365d1e6ce266a45289cc9` | 343,331,093 |
| smoltalk | `HuggingFaceTB/smoltalk` @ `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa` | 1,611,239,483 |
| dolci | `allenai/Dolci-Instruct-SFT` @ `bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221` | 903,445,414 |

Total 3,948,804,069 stream tokens (with EOS), matching
`manifests/tokenized_manifest.json`'s `total_stream_tokens_with_eos` on the
build dir. Tokenizer: `allenai/dolma2-tokenizer` @
`5292e5d6c0f40b67cc765fe41bec991cf4345b5c`.

46 trained-on files: `tokenized/<source>/<domain>/{train,val}.npy`, headerless
little-endian `uint32` token-id streams, EOS(100257) after every document, no
dedup, no upsampling. Domains: `general`, `math`, `code`, `science`, `chat`
(SmolTalk `all` config and Tulu-3 `personahub_*` rows map to `general`).
Published (v3) object layout: each `<split>.npy` is byte-split into 1 GiB
(`1_073_741_824`-byte) shards named `tokens/<source>/<domain>/<split>-NNNNN.u32le.bin`
(`publish_refhq_new.py: split_npy_to_shards`); see `outputs.json`'s
`v3_objects` per output (derived directly from the verified `.npy` bytes, not
re-fetched from S3 -- the rebuild only depends on public HF data).

## As-built quirks (recorded, not "fixed")

- **SmolTalk double-counts 11 configs.** `normalize_filter_source.py` iterates
  `sorted(get_dataset_config_names(...))` and skips only `apigen-80k` and
  `smol-constraints`; it does **not** skip the aggregate `all` config. `all`
  is itself a superset of the 11 domain configs, so those 11 configs'
  documents are candidates twice: once via `all` (1,043,917 kept after the
  metadata filter) and once via their own config name (926,349 kept) =
  1,970,266 kept candidates. `documents-smoltalk.tsv.gz` records each
  surviving copy against its own (file, row) — i.e. duplicate content,
  distinct provenance, exactly as trained.
- **Tulu-3 Persona-IF exclusion matches nothing.** `exclusion_rules.yaml`
  lists `tulu-3-sft-personas-instruction-following` as a drop substring
  against the `source` column, but no row's `source` contains that string
  (the real column value uses a different separator/casing), so all 29,980
  Persona-IF rows are retained. The 211,103 rows Tulu-3 *does* drop are
  CoCoNot, WildJailbreak, WildGuardMix, Aya, plus 120 rows whose row `id`
  happens to contain the substring `aya`.
- **`hermes-3`'s `"gated": true` in `plan.json` is stale.** `NousResearch/Hermes-3-Dataset`
  downloaded fine with no `HF_TOKEN` at the pinned revision on 2026-09-25 (see
  "Contradictions found" below).

## Rebuild

```
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python3 rebuild.py --out OUT_DIR --cache CACHE_DIR
```

`--only smoltalk/math/train` builds one output; `--only <source>` builds every
output for that source (used for the FarmShare verification: 6 jobs, one per
source, each with `--workers 8` for internal tokenizer multiprocessing, rather
than one Slurm task per part or per output — grouping matters because a single
`match`/`splitparts`-style task only takes seconds, but a `--only <source>`
build does real work). `--list-tasks`/`--task-file`/`--task-index` are also
available for building one part at a time if finer-grained parallelism is ever
needed. Needs network access to Hugging Face; no `HF_TOKEN` is needed for any
of the six sources (confirmed 2026-09-25 — `NousResearch/Hermes-3-Dataset`
downloads unauthenticated despite `plan.json` marking it gated). No fastText,
no filters, no random seeds: every trained document's source row is already
recorded in `documents-<source>.tsv.gz`.

## Verification (FarmShare)

Run 2026-09-25 on FarmShare, `/scratch/users/nzhao2/agent-runs/repro-manifests-20260925/refhq-new-v1/`:
fresh `/usr/bin/python3.12 -m venv` + `pip install -r requirements.txt` (versions
confirmed identical to the original build venv), empty HF cache, no `HF_TOKEN`
needed for any of the six sources. Slurm: `--partition=normal`, no `--nice`,
`--exclude=oat-01..06,barley-01..04` (barley-01 has a broken `/scratch` and
`/home` NFS mount; barley-02..04 were mid-reboot/completing at run time).

**Result: all 46/46 trained-on files are sha256-identical to the rebuild, including every derived v3 shard object.**

| output | expected sha256 (12) | rebuilt sha256 (12) | match |
|---|---|---|---|
| dolci/chat/train | de8309666f2e | de8309666f2e | yes |
| dolci/chat/val | c574ba462dac | c574ba462dac | yes |
| dolci/code/train | db28e3561eb6 | db28e3561eb6 | yes |
| dolci/code/val | e3d8848967a3 | e3d8848967a3 | yes |
| dolci/general/train | 9688a5422dcf | 9688a5422dcf | yes |
| dolci/general/val | c5ba333c4c6a | c5ba333c4c6a | yes |
| dolci/math/train | c8f3800894e7 | c8f3800894e7 | yes |
| dolci/math/val | ef793a7d4048 | ef793a7d4048 | yes |
| dolci/science/train | a4bb606f0e46 | a4bb606f0e46 | yes |
| dolci/science/val | 6ebafbd7e87c | 6ebafbd7e87c | yes |
| hermes-3/general/train | 955ee6c476e3 | 955ee6c476e3 | yes |
| hermes-3/general/val | 260c01a38110 | 260c01a38110 | yes |
| openhermes-25/chat/train | aeee938a5b01 | aeee938a5b01 | yes |
| openhermes-25/chat/val | 13ffe0c2daba | 13ffe0c2daba | yes |
| openhermes-25/code/train | 2ac5574e9724 | 2ac5574e9724 | yes |
| openhermes-25/code/val | cc65dd89a83b | cc65dd89a83b | yes |
| openhermes-25/general/train | b5a1ec311cb6 | b5a1ec311cb6 | yes |
| openhermes-25/general/val | 75be68ac308c | 75be68ac308c | yes |
| openhermes-25/math/train | f44b762e3098 | f44b762e3098 | yes |
| openhermes-25/math/val | 005699b6097d | 005699b6097d | yes |
| smoltalk/chat/train | a903b93ec7a7 | a903b93ec7a7 | yes |
| smoltalk/chat/val | 3bfa75e549e6 | 3bfa75e549e6 | yes |
| smoltalk/code/train | c3b8a4445147 | c3b8a4445147 | yes |
| smoltalk/code/val | 3dcf27e3e011 | 3dcf27e3e011 | yes |
| smoltalk/general/train | 2c346ce3ae26 | 2c346ce3ae26 | yes |
| smoltalk/general/val | b0aaba5f236c | b0aaba5f236c | yes |
| smoltalk/math/train | 5299eda27ba8 | 5299eda27ba8 | yes |
| smoltalk/math/val | ae5547425665 | ae5547425665 | yes |
| tulu-3/chat/train | 33f2567de39b | 33f2567de39b | yes |
| tulu-3/chat/val | 1aa47ddc113c | 1aa47ddc113c | yes |
| tulu-3/code/train | 878bfe454860 | 878bfe454860 | yes |
| tulu-3/code/val | 624d2d729579 | 624d2d729579 | yes |
| tulu-3/general/train | 3f37307e798d | 3f37307e798d | yes |
| tulu-3/general/val | 7699b2d7a085 | 7699b2d7a085 | yes |
| tulu-3/math/train | e1fcfcb80a30 | e1fcfcb80a30 | yes |
| tulu-3/math/val | 87fd801f5165 | 87fd801f5165 | yes |
| tulu-3/science/train | 2d0c3416989d | 2d0c3416989d | yes |
| tulu-3/science/val | 4c22c15b253b | 4c22c15b253b | yes |
| tulu-v2/chat/train | e7337da1fde4 | e7337da1fde4 | yes |
| tulu-v2/chat/val | b022dc2720a4 | b022dc2720a4 | yes |
| tulu-v2/code/train | 7ed1c2926324 | 7ed1c2926324 | yes |
| tulu-v2/code/val | 5eca2c6009f5 | 5eca2c6009f5 | yes |
| tulu-v2/general/train | f8b399cf2ac7 | f8b399cf2ac7 | yes |
| tulu-v2/general/val | 4b3e1fe3189a | 4b3e1fe3189a | yes |
| tulu-v2/science/train | 2e67142088db | 2e67142088db | yes |
| tulu-v2/science/val | e22e4bf5ecc4 | e22e4bf5ecc4 | yes |

Manifest-derivation jobs (`build_manifest.py`, on the read-only build dir's own
venv): candidates 1742509 (51-task array, 4 CPU/task), splitparts 1742696
(46-task array, 2 CPU/task), match 1742697+1743083+1743098 (partial, hit a
tuple-unpack bug, fixed) then 1743138 (16-pair sequential job, 2 CPU, 72s),
reduce 1743967 (2 CPU, 4m06s). Every one of the 23 `(source,domain)` pairs
matched with `n_unmatched: 0`, `holdout_counts_match: true`,
`n_out_of_order_duplicate_matches: 0` (see `match_stats.json`).

Rebuild-verification jobs (fresh venv, empty cache): download 1743981+1744038
(51-task array, 2 CPU/task, ~380s total CPU time — one task hit a benign
NFS-race `FileExistsError` in `rebuild.py`'s marker-dir creation, fixed,
resubmitted), then 6 grouped per-source jobs (not one task per part, per
reviewer direction once per-domain match tasks turned out to take only
seconds each): `sbatch_rebuild_verify_source.sh --only <source> --workers 8`.

| source | job | elapsed | outputs |
|---|---|---|---|
| tulu-v2 | 1744039 | 2m45s | 6/6 |
| openhermes-25 | 1744040 | 3m26s | 8/8 |
| tulu-3 | 1744041 | 4m26s | 10/10 |
| hermes-3 | 1744042 | 3m02s | 2/2 |
| smoltalk | 1744043 | 13m41s | 8/8 |
| dolci | 1744044 | 7m21s | 10/10 |

Approximate compute: candidates+splitparts ≈5.1 CPU-hours, match+reduce
≈0.18 CPU-hours, download ≈0.21 CPU-hours, verify ≈4.6 CPU-hours (8 CPU ×
34m41s total across the 6 jobs) — roughly 10 CPU-hours end to end. Scratch
disk used under the work dir: 37 GB (10 GB re-downloaded HF cache for the
manifest derivation, 10 GB empty-start HF cache for the verification rebuild,
17 GB rebuilt output tree, ~1 GB everything else). Wall-clock time across the
whole session was substantially longer than the sum of job "Elapsed" times
because of Slurm fairshare queueing shared with three other concurrently
running agents (each doing comparable FarmShare work under the same Slurm
user) — none of the delay was compute time.

## Files

| File | Content |
|---|---|
| `sources.json` | Every public source file (repo, pinned revision, path, size, sha256) + tokenizer. |
| `documents-<source>.tsv.gz` | One row per output document, in output order, per source: `domain`, `split`, `src` (index into `sources.json.files`), `row` (0-based, native per-file order), `ntok` (content tokens, excludes EOS). |
| `outputs.json` | The 46 trained files: bytes/sha256/doc count/parts (with `doc_start`/`docs` for slicing `documents-<source>.tsv.gz`), plus derived v3 `tokens/.../*.u32le.bin` shard objects. |
| `rebuild.py` | Standalone rebuild + verify. |
| `requirements.txt` | Exact pins, Python 3.12.3. |
| `build_manifest.py` | The FarmShare script that derived this manifest (provenance; not needed to rebuild). |
| `match_stats.json` | Per-`(source,domain)` matching diagnostics (holdout-seed replay check, unmatched counts, duplicate-match notes). |

## Manifest recovery method

The build dir kept the trained token files but deleted the intermediate
normalized/English-filtered/held-out text. `build_manifest.py` recovers
per-document provenance by **content**, not by re-deriving selection:

1. Re-download all six sources at the pinned revisions; recompute the
   metadata filter (`keep_row`) and domain map (`map_domain`) — cheap,
   deterministic, no fastText — to get, per `(source, domain)`, the ordered
   candidate rows that would have gone into the (now-deleted) Dolma
   English-filter step.
2. Render (`flatten_conversation`) and tokenize every candidate; hash the
   exact bytes (`uint32`-LE ids + EOS) that would land in a trained file.
3. Split every trained `.npy` part into documents on EOS(100257), cross-checked
   against that part's own `.json` doc count; where a literal
   `<|endoftext|>` tokenized to EOS mid-document (over-split), adjacent
   pieces are re-merged until they hash-match a candidate.
4. Replay the holdout RNG (`_pair_seed`/`holdout_counts`, copied verbatim)
   with the recovered `n_docs` to reconstruct the single pre-holdout,
   post-English-filter order from the two independently-ordered train/val
   sequences (each individually preserves relative order; only membership
   was randomized).
5. Walk that reconstructed order against the candidates in one left-to-right
   pass, consuming the earliest not-yet-consumed candidate with a matching
   hash per document (this is what "preserving order among duplicates"
   means); a candidate with no consumer is an English-filter drop, not an
   error.

See the module docstring in `build_manifest.py` for the full algorithm and
`match_stats.json` for per-pair diagnostics (`n_candidates`, `n_matched`,
`n_unmatched`, `holdout_counts_match`, `n_out_of_order_duplicate_matches`).

## Known limits

- **None found that affect byte-identity.** All 46 outputs rebuilt
  sha256-identical from public HF data with no filters, no fastText, no
  random seeds — only the committed manifest, `rebuild.py`, and
  `requirements.txt`.
- The manifest attributes documents among **true content duplicates** (e.g.
  SmolTalk rows that appear both via the `all` config and their own named
  config) by consuming the earliest not-yet-consumed candidate with a
  matching token-hash, in reconstructed pre-holdout order. Because the
  underlying content is byte-identical for these duplicates, this ordering
  choice cannot be verified against the (deleted) intermediate text — it
  affects only which specific `(src, row)` a duplicate-content document is
  attributed to, never the rebuilt bytes. 0 pairs showed the
  `n_out_of_order_duplicate_matches` counter fire above 0, i.e. no case
  needed this tie-break to resolve an actual ambiguity between train and val.
- `documents-<source>.tsv.gz` is one file per **source** (not per source+domain);
  all six are well under the 50MB guideline (largest, smoltalk, is 8.8MB), so
  no split was needed.
- The rebuild needs network access to Hugging Face; if any of the six repos'
  pinned revision is ever garbage-collected (unlikely for public dataset
  revisions) the rebuild would fail at the `hf_hub_download` step with a
  clear error, not silently produce wrong output (sha256 of the downloaded
  file is checked against `sources.json` before any tokenization happens).

## Contradictions found (report only)

- `plan.json`'s `sources.hermes-3.gated: true` does not reflect current
  reality: `NousResearch/Hermes-3-Dataset` at the pinned revision downloaded
  without any `HF_TOKEN` on 2026-09-25 (both `list_repo_files` and
  `hf_hub_download` succeeded unauthenticated). The `.hf_token` files under
  `/scratch/users/nzhao2/refhq-new-v1/` and
  `/scratch/users/nzhao2/refhq-regmix-5p5b-v1/` are both currently invalid
  (`401 Invalid user token` / expired OAuth claim) — harmless here since the
  repo isn't actually gated right now, but worth flagging since other
  in-flight agent work may assume that token is live.
