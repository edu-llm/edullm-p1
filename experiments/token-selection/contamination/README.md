# Benchmark contamination audit (verbatim 13-gram)

This directory vendors the code and the reduced results behind **Section 4** of the
paper. It is the authoritative copy: the working directory on FarmShare was
`/scratch/users/nzhao2/agent-runs/ngram13-20260911T161822/`, and everything here
was copied down from that run.

Two large/derived artifacts are deliberately **not** vendored, because they are
regenerable and would dominate the repo: the eval-side index `eval_index.pkl`
(13 MB), and the raw per-shard hit directories `results/`, `strict_hits/`,
`res_hits/`, `hits_*`, and `logs/`. Re-running `build_eval_index.py` plus the
`*.sbatch` array jobs reproduces them.

> **Hash stability is required.** Every sbatch wrapper exports
> `PYTHONHASHSEED=0` (see `strict.sbatch`, `res.sbatch`). Without it, CPython's
> built-in `hash()` is salted per process, n-gram hashes are not stable across
> array tasks, and hits fail to reduce. The strict eval index
> `eval_ngrams_strict.pkl` and the scanner `contam_scan.py` are reused unchanged
> so these numbers stay directly comparable to the published 0.89%.

## Methodology

**Unit of overlap: the verbatim 13-gram.** `n = 13`, matching the prior
published protocol. A hit means a corpus document contains a word 13-gram that
also occurs in an evaluation item's prompt stem.

**Normalization** (recorded machine-readably in `fields.json`):

- lowercase
- every non-alphanumeric character is treated as a **separator**, not as a
  character (so punctuation cannot be part of a token)
- tokens are the resulting whitespace-delimited runs

n-grams are hashed (FNV-1a 64-bit in `build_eval_index.py`; CPython `hash()`
with `PYTHONHASHSEED=0` in the strict/reservoir path) and looked up in a
`gram -> {eval item id}` dictionary. Collisions are possible in principle at
64 bits but negligible at this scale, and they can only *inflate* the reported
rate, so the numbers are conservative upper bounds.

**Question text only.** For every benchmark we index only the prompt stem the
model is actually shown — never the answer options and never the gold label.
Indexing the options would make almost any mention of a common noun a "hit".
The exact field per benchmark, from `fields.json`:

| Benchmark | Field matched |
| --- | --- |
| `arc_challenge` | `question` |
| `arc_easy` | `question` |
| `boolq` | `passage` + `question` (both appear in the RC prompt) |
| `csqa` | `question` |
| `hellaswag` | `ctx` (activity label + context, as shown in the prompt) |
| `mmlu_*` | `question` |
| `openbookqa` | `question_stem` |
| `piqa` | `goal` |
| `socialiqa` | `context` + `question` |
| `winogrande` | `sentence` |

**Denominator: 40,087 evaluation items.** These are all items across the **20
(task, split) labels** of the OLMo-ladder RC suite — the same 20 labels the
training runs evaluate `task_loss_bpb` on. Per-item counts:
`arc_challenge_val` 299, `arc_challenge_test` 1,171, `arc_easy_val` 570,
`arc_easy_test` 2,374, `boolq_val` 3,270, `csqa_val` 1,220, `hellaswag_val`
10,042, `openbookqa_val` 463, `openbookqa_test` 462, `piqa_val` 1,560,
`socialiqa_val` 1,954, `winogrande_val` 1,267, and MMLU 1,519 (val) + 13,916
(test) = 15,435. Sum = 40,087.

MMLU supplies **8 of the 20** labels (`stem`, `humanities`, `social_sciences`,
`other` x `val`, `test`); in `strict_summary.json` and
`reservoir_summary.json` those 8 are reported *pooled* as `mmlu_val` and
`mmlu_test`, so the per-task tables print 14 rows rather than 20. The item
denominator is unaffected.

**"Self-contained question stems."** MMLU, HellaSwag and BoolQ stems are not
self-contained: a HellaSwag `ctx` is a sentence fragment, a BoolQ stem carries
a whole Wikipedia passage, and MMLU questions frequently quote textbook or
statute text that legitimately appears in pretraining data. A 13-gram hit on
those is usually *source* overlap, not test leakage. The self-contained subset
therefore restricts to the 8 benchmarks whose stem is a standalone question or
sentence — `arc_challenge`, `arc_easy`, `csqa`, `openbookqa`, `piqa`,
`winogrande`, `socialiqa`, `boolq` — and is the number to read as an estimate
of actual leakage. (The subset list is `SELFC` in `reduce_strict.py`; `boolq`
is included there for continuity with the prior protocol even though its stem
embeds a passage, and it contributes zero hits in every corpus, so it does not
move the rate.)

## Results — the paper's three corpora

Denominator 40,087 eval items. From `strict_summary.json` -> `by_corpus`.

| Corpus | Role | Docs scanned | 13-grams | Items matched | Item rate | Self-contained |
| --- | --- | --- | --- | --- | --- | --- |
| `dml_train_10b` | 10B training corpus (`pretrain/regmix-10b` v1) | 4,748,990 | 5,822,742,335 | 358 | **0.8931%** | **0.1574%** |
| `refhq_hq` | HQ reference corpus | 3,265,570 | 3,175,606,016 | 438 | **1.0926%** | **0.1437%** |
| `refhq_instruct` | Instruct reference corpus | 6,193,748 | 2,663,753,719 | 1,472 | **3.6720%** | **3.6277%** |

These match the paper's reported 0.89% / 0.16%, 1.09% / 0.14%, and
3.67% / 3.63%.

Read the contrast this way: for the training corpus and the HQ reference, the
overall rate (~1%) is an order of magnitude above the self-contained rate
(~0.15%), i.e. essentially all of it is MMLU/HellaSwag source overlap. For the
Instruct reference the two rates *coincide* (3.67% vs 3.63%) — the overlap is
in genuine standalone question stems, which is what instruction-tuning mixtures
are made of.

### `refhq_instruct` per task

From `strict_summary.json` -> `by_corpus.refhq_instruct.per_task`
(matched / n / rate).

| Label | Matched | n | Rate |
| --- | --- | --- | --- |
| `csqa_val` | 394 | 1,220 | **32.2951%** |
| `mmlu_val` | 415 | 1,519 | **27.3206%** |
| `arc_challenge_test` | 35 | 1,171 | 2.9889% |
| `mmlu_test` | 379 | 13,916 | 2.7235% |
| `arc_easy_test` | 51 | 2,374 | 2.1483% |
| `arc_challenge_val` | 6 | 299 | 2.0067% |
| `winogrande_val` | 25 | 1,267 | 1.9732% |
| `openbookqa_val` | 8 | 463 | 1.7279% |
| `hellaswag_val` | 148 | 10,042 | 1.4738% |
| `arc_easy_val` | 7 | 570 | 1.2281% |
| `openbookqa_test` | 1 | 462 | 0.2165% |
| `piqa_val` | 3 | 1,560 | 0.1923% |
| `boolq_val` | 0 | 3,270 | 0.0000% |
| `socialiqa_val` | 0 | 1,954 | 0.0000% |

The paper's headline per-task figures are `csqa_val` **32.3%** and `mmlu_val`
**27.3%**. CSQA and the MMLU dev/validation split are both widely redistributed
inside public instruction mixtures, which is exactly what this measures.

Per-source attribution for `refhq_instruct` (`by_domain`): `hermes-3` 2.60%,
`openhermes-25` 1.52%, `tulu-3` 1.40%, `dolci` 1.29%, `tulu-v2` 1.16%,
`smoltalk` 0.61%. No single source accounts for it; the mixtures overlap each
other.

For the 10B training corpus, `by_domain` localizes the overlap to `dclm`
(337 of 358 matched items, 0.8407%); `arxiv` 0.0449%, `pes2o` 0.0499%,
`starcoder` 0.0324%, `wiki` 0.0299%, and `open-web-math` and
`algebraic-stack` contribute exactly zero.

## Other result files

- `strict_summary.json` — the strict-protocol scan. Also covers three corpora
  not in the paper's Section 4 table: `olmomix_stock_30b` (0.1297% / 0.0205%),
  `olmoe_synthetic_10b` (0.9928% / 0.1027%), `lgbm_opt` (1.1350% / 0.1027%).
- `reservoir_summary.json` — the full 127B tokenized reservoir, scanned from
  raw `uint32` Dolma2 shards by decoding tokens back to text (`scan_tok.py`).
  62,057,627 docs, 74,305,588,695 13-grams, 7.4713% overall / 1.9165%
  self-contained. Per domain the overall rate ranges from `starcoder` 0.1222%
  to `algebraic-stack` 3.4799% and `dclm` 3.2380%.
- `arm_exposure.json` — per-domain rates re-weighted into each candidate data
  mixture, at a common per-domain token budget (`budget_tokens` 3,745,444,630,
  set by the smallest domain, `wiki`). Arm exposure: Olmo-mix-1124 natural
  0.7645%, Data Mixing Laws paper weights 0.4000%, MixLaw fit 0.6184%,
  LightGBM fit 0.5051%.
- `arm_union.json` — the same arms evaluated as an actual 10B-token union draw
  rather than a re-weighting: Olmo-mix natural 1.5491%, DML paper 1.0702%,
  MixLaw fit 1.2248%, LightGBM fit — see file.
- `mine_summary.json` — an independent pass over the two reference corpora
  using the **new** `build_eval_index.py` index (26,123 items, 15 labels, MMLU
  kept split into its four categories, short items dropped). Useful as a
  cross-check that the strict numbers are not an artifact of the older index;
  it is *not* the source of any paper number, and its denominator differs, so
  do not mix its rates with the table above.
- `fields.json` — the machine-readable protocol record: `n`, per-benchmark
  field, normalization string, index size.

## Code map

Index construction

- `build_eval_index.py` — builds the n=13 eval-side index from HuggingFace
  datasets; writes `eval_index.pkl` and `fields.json`. Submitted by
  `build_index.sbatch`.

Scanning

- `scan_shard.py` — scan one JSON/JSONL(.gz) corpus shard against the index
  (`scan.sbatch`). Handles plain-text documents and flattened chat records
  (`messages` / `conversations`).
- `scan_big.py` — multi-core variant for the largest shards
  (`scan_big.sbatch`).
- `scan_tok.py` — scan a raw-`uint32` tokenized Dolma2 shard by decoding to
  text, then applying exactly the strict normalization. Token-range sharded
  (`res.sbatch`).
- `strict.sbatch` — runs the *prior* scanner `contam_scan.py` (from the Sep-10
  directory) unchanged against `eval_ngrams_strict.pkl`. This is the path that
  produces `strict_summary.json`, i.e. the paper's numbers.

Manifests

- `mkstrict.py` — builds `strict_shards.tsv`, one row per (file, line-shard),
  ~700 MB gz per array task, capped at 20 shards per file.
- `mkres.py` — builds `res_shards.tsv` for the 127B reservoir, sharded by
  token range (120M tokens per task), topping up `pes2o` and `starcoder` from
  the topup tree to hit the paper's per-domain totals.

Reduction and analysis

- `reduce.py` / `reduce_strict.py` — aggregate per-shard hit files into
  per-corpus and per-domain tables with item rates and the self-contained
  subset.
- `ref_numbers.py` — recompute the previously published numbers straight from
  the Sep-10 artifacts, as an exact reference point for the new scans.
- `arm_exposure.py` / `arm_union.py` — project per-domain rates onto the
  candidate data mixtures.
- `tokcount.py` — per-domain token totals from `.npy` headers (no data read).
- `tokinv.py` — inventory every tokenized shard tree as `filesize / 4` (raw
  `uint32`), and diff against the paper's reservoir targets.

## Reproducing

```bash
RUN=/scratch/users/nzhao2/agent-runs/ngram13-<stamp>
mkdir -p "$RUN"/{logs,results,strict_hits,res_hits}
cp *.py *.sbatch "$RUN"/

sbatch build_index.sbatch "$RUN"                      # eval_index.pkl
python mkstrict.py "$RUN"                             # strict_shards.tsv
sbatch --array=0-$(( $(wc -l < "$RUN"/strict_shards.tsv) - 1 )) strict.sbatch "$RUN"
python reduce_strict.py "$RUN"/strict_hits "$RUN"/strict_summary.json
```

Every sbatch wrapper pins `PYTHONHASHSEED=0`. Dropping it silently invalidates
the reduction — that is the Sep-10 failure mode.
