# refhq-regmix-5p5b-v1

The 5.5B-token HQ (quality-filtered) reference corpus for the token-selection paper's
Perplexity arm: the HQ reference model was trained on it, and its checkpoints scored that
arm. It mixes the training corpus's seven domains -- dclm, arxiv, starcoder, pes2o,
open-web-math, algebraic-stack, wiki -- each independently sampled/filtered from a public
Hugging Face source, tokenized with `allenai/dolma2-tokenizer`, and published internally as
24 train + 7 val objects (one EOS token, id 100257, appended after every document's content
tokens). Only arxiv comes from `allenai/olmo-mix-1124`; the other six domains come from their
original public releases (`sources.json`), each under its own terms: StarCoderData is gated
under The Stack's terms of use, and Wikipedia is CC-BY-SA 3.0 and GFDL. We do not
redistribute the corpus; this manifest is how to rebuild it.

This directory is a **document-level manifest**: for every document that ended up in the
trained-on token files, it records exactly which public-HF row it came from (or, for dclm,
which token-memmap window) and every transformation applied, so that anyone can rebuild the
trained-on files from public data and check them byte-for-byte, without our filters,
classifiers, random seeds, or the original (now-deleted) build venvs.

## Trained-on objects

31 objects under `tokens/<domain>/{train-NNNNN,val-00000}.u32le.bin` (headerless uint32
little-endian, dolma2 vocabulary, EOS id 100257): 24 train (5,509,020,202 tokens) + 7 val
(8,275,942 tokens) = 5,517,296,144 tokens with EOS (5,514,030,574 content tokens,
3,265,570 documents). Every shard's sha256, and the whole-domain `tokenized/<domain>/<domain>.npy`
sha256 it was cut from, is in `outputs.json`, computed directly from the surviving
`tokenized/*.npy` files on FarmShare (read-only) -- see Verification below.

Layout rule (`scripts/publish_refhq_edullm_data.py`): split each domain's `.npy` into
contiguous 1 GiB (1,073,741,824 byte) shards, then carve the last 0.15% of the domain's
byte stream off the tail of the final shard into `val-00000.u32le.bin` (same fraction for
every domain, so val mix weights match train). `outputs.json`'s object list (24 train + 7
val, with the same total token counts quoted above) matches this rule exactly.

## Repo layout of this manifest

`build_manifest.py`, the FarmShare script that derived this manifest from the surviving build artifacts, has been removed (it referenced FarmShare-only paths and is not needed to rebuild); it is kept in git history.

| File | Content |
|---|---|
| `sources.json` | Every public HF file this manifest's documents actually reference: repo id/type/revision (full commit SHA), path, size, sha256. |
| `documents-<domain>.tsv.gz` | One row per output document, in output order: `idx`, `file` (path into `sources.json`'s file list for that domain), `row`, `ntok`; dclm has `start`,`length`,`wrapped`,`trimmed_to` instead of `row`; starcoder additionally has `id`, `spans` (JSON list of `[start,end)` character ranges deleted from the raw row, `""` if none), and `templated` (`"1"` for the 42 documents with no source row at all -- see "StarCoder placeholder documents" below). |
| `outputs.json` | Every trained-on object (name, bytes, sha256, tokens) plus the whole-domain `.npy` hashes and the layout rule, computed read-only from the surviving `tokenized/*.npy`. |
| `rebuild.py` | `--out DIR --cache DIR [--only DOMAIN]`: downloads sources, replays the manifest, tokenizes, shards, and checks every sha256 against `outputs.json`. |
| `requirements.txt` | Exact `==` pins (Python 3.12.3) -- this is the environment the whole rebuild was validated against on FarmShare. |

## How each domain was selected, and how it maps back to public data

All seven domains were built by `scripts/build_hq_reference_domain.py`: for six of them, a
Python `random.Random(seed)` reservoir shuffle (window 10,000) reorders each domain's raw
stream, an optional filter runs per document, and passers are tokenized and accumulated
until a token budget is hit (with the tail trimmed back under budget). **None of that
selection needs to be replayed** -- each kept document's `id`/`metadata` field already
pins it to a specific public row, so the manifest records that mapping directly instead of
re-deriving it:

- **wiki** (73,824 docs): `id` is the row's own Wikipedia page id. All fall in
  `20231101.en/train-00000-of-00041.parquet` (row-scanned once for the needed ids).
  No transform. Verified 300/300 sampled documents byte-identical against a fresh
  download.
- **algebraic-stack** (99,445 docs): the algebraic-stack parquet schema has neither `id`
  nor `url`, so the build fell back to `typeof/algebraic-stack-<i>`, the 0-based row index
  in the dataset's native file order (`train/agda0000.parquet`, `train/c0000.parquet`, ...,
  sorted, matching how `datasets` streams a plain multi-file parquet split). `<i>` is
  resolved to (file, local row) by cumulative per-file row counts (footer-only reads, no
  data download needed for the mapping itself). No transform. Verified 300/300.
- **arxiv** (72,894 docs, all from a single `data/arxiv/train/arxiv-train-0016.json.gz`
  -- of the domain's 20 files, that one happened to sort first under
  `random.Random(45).shuffle()` and alone was large enough to fill the reservoir buffer and
  the whole token budget): `id` is `proofpile-arXiv_<tag>-<k>`, an identifier from the
  upstream proofpile packaging. **`<k>` is only sequential within a single upstream
  proofpile shard, not the physical line number** -- a repacked `arxiv-train-*.json.gz` can
  concatenate several such shards, so the manifest resolves each id by an exact-match scan
  of the file rather than trusting the numeric suffix as a position (an earlier pass that
  assumed suffix == line number was wrong for 72,766 of 72,894 documents; fixed before this
  manifest was finalized). No transform. Verified 150/150 sampled documents byte-identical.
- **pes2o** (1,879,376 docs): `id` is the row's own peS2o paper id; peS2o's own
  loading script (`data/v2/train-{i:05d}-of-00020.json.gz`, config `v2`, read in file order
  0..19) is what the build streamed, so `id` is resolved by scanning those files in order
  until every needed id is found. No transform. Verified 150/150.
- **open-web-math** (125,554 docs, from a full pass over all 6,315,233 rows): `id` is the
  page's own `url`; resolved by scanning the `url` column only (no full-text read) across
  all 114 `data/train-*.parquet` files. No transform (the `math_score`/native-signal HQ cut
  in `math_quality.keep_openwebmath_hq_record` decides which rows are kept upstream of this
  manifest; the manifest only records which rows survived, per the "record selection, don't
  recompute filters" rule). Verified N/N (see Verification table).
- **starcoder** (762,037 docs): initially blocked -- the pinned `.hf_token` on FarmShare
  was expired ("Invalid user token" / HTTP 401 on every call, including plain listing,
  while every other, public repo in this manifest worked fine anonymously). The user
  supplied a fresh token partway through this pass; once verified (`whoami()` succeeded,
  listing `bigcode/starcoderdata` at its pinned revision worked), starcoder was mapped like
  the manifest's other `hf_dataset`-kind domains, with two starcoder-specific wrinkles:
  - `id` is the row's own `id` column, but starcoderdata resets it to 0 at the start of
    each top-level language directory (not globally, and not per file within a directory),
    so the same id string is shared by many unrelated rows across different languages;
    Dolma's tag/mix round-trip also overwrote the build's per-document `source` field to
    a generic `"starcoder-hq"` label, discarding which file a document came from. Neither
    is enough alone to place a document, so the manifest resolves each one among every
    same-id candidate in the head-of-stream prefix by checking whether that candidate's
    raw `content` can be turned into the (possibly copyright/comment-stripped) output text
    by deleting some character spans -- recording the winning span list, verified by
    reconstructing the output text from it before accepting.
  - **The head-of-stream prefix must be sorted by full file path, not by directory name.**
    `datasets`' actual streaming order sorts complete relative paths as single strings, so
    `c-sharp/train-00000-of-00045.parquet` sorts *before* `c/train-00000-of-00053.parquet`
    (`-` is `0x2D`, `/` is `0x2F`, and they first differ at that position) even though the
    bare directory names sort the other way (`"c"` is a prefix of `"c-sharp"`, so `"c" <
    "c-sharp"`). An initial version of this manifest grouped by directory name and sorted
    directories separately, silently reproducing the wrong order; a live streaming trace
    (`load_dataset("bigcode/starcoderdata", streaming=True)`, no `data_dir`) confirmed the
    true order empirically -- the first `.cs` (C#) file starts at exactly row 570,394,
    immediately after `bluespec`, matching the full-path sort exactly. Under the wrong
    (directory-name) order, the assumed prefix never reached "c-sharp" at all within the
    980,000-row boundary the build actually scanned, so every genuine C# document in that
    range (roughly half of starcoder's kept documents, by row count) had no candidate and
    came back "unresolved." Fixed by sorting the full 865-file listing as one list of path
    strings; the corrected 13-file prefix (`ada`, `agda`, `alloy`, `antlr`, `applescript`,
    `assembly`×2, `augeas`, `awk`, `batchfile`, `bluespec`, `c-sharp`×2) resolves all but 42
    of 762,037 documents (see below), and 300/300 sampled documents reconstruct
    byte-identically.

  ### StarCoder placeholder documents (data finding, not just a manifest bug)

  **42 of starcoder's 762,037 documents (0.0055%, 475 of 775,268,890 tokens, 0.00006%) are
  not code at all.** Their entire text is `f"{id}\n{repo_id}\n"` -- e.g. `"486\n
  bigcode/starcoderdata\n"` -- literally just the row's own id and the source repo name.
  Code path (`datasets/refhq/scripts/build_hq_reference_domain.py:265-269`
  together with `olmo_shard_utils.py:126-131,142`, both in the code that ran on FarmShare):
  after Dolma's `code_copyright_comments_v1` span-replacement strips a *whole* document's
  `text` field to `""` (a `copyright_notice` or `comment_block` span covering the entire
  file, score >= 0.3), `iter_docs()`'s guard (`olmo_shard_utils.py:142`,
  `if isinstance(obj, dict) and doc_text(obj): yield obj`) still yields the object, because
  `doc_text()` (`olmo_shard_utils.py:126-131`) falls through its `("text","content","code",
  "body")` checks -- all absent or empty -- to its last-resort branch,
  `"\n".join(v for v in obj.values() if isinstance(v, str))`, which joins the *surviving*
  string-typed fields (`id` and `source`) into a non-empty, non-code string. Truthy, so it
  passes the guard; `build_hq_reference_domain.py:269` then keeps it as an ordinary
  document (`{"id": ..., "text": <that fallback string>, "source": "starcoder-hq"}`) with
  no further check that the text is actual code. **This means up to 42 documents (a
  negligible token share) that the HQ reference model trained on are not source code, but
  an artifact of the copyright-stripping pipeline silently keeping a fully-redacted
  document instead of dropping it.** These 42 have no source row to fetch at all -- their
  text is fully determined by two fields already in `out/starcoder` -- so the manifest
  records them as `templated` (`documents-starcoder.tsv.gz`'s `templated="1"` rows carry
  just `id`; `rebuild.py` regenerates the exact text from `id` + `sources.json`'s `repo_id`,
  no download needed). Every one of starcoder's originally-"unresolved" documents is
  accounted for this way -- there are no documents that are neither resolved nor templated.
- **dclm** (252,440 docs): not a row-level filter at all -- `sample_datadecide_dclm.py`
  samples 8192-token windows (crossing document boundaries) from 3 of
  `allenai/DataDecide-data-recipes`'s tokenized shards, decoded with
  `allenai/gpt-neox-olmo-dolma-v1_5` and re-tokenized with dolma2. Which 3 files, and every
  window's (file, start, length) in source-token space, is fully determined by two seeded
  shuffles that this manifest replays exactly (see below) -- so `documents-dclm.tsv.gz`
  records concrete `(file, start, length, wrapped, trimmed_to)` windows rather than any
  row/id.

### dclm's two shuffles (both replayed, both confirmed against the known answer)

1. `scripts/download_hq_source.py` lists every `.npy` file under
   `allenai/DataDecide-data-recipes`'s `preprocessed/dclm/v0_rep32_ft7percentile_fw2/
   gpt-neox-olmo-dolma-v1_5/` prefix at revision `3baf34ba...` (500 files), applies
   `random.Random(42).shuffle()`, and keeps the first 52 (`REGMIX_5P5_DCLM_MAX_FILES`).
2. `scripts/sample_datadecide_dclm.py` sorts those 52 filenames, applies
   `numpy.random.default_rng(42).shuffle(range(52))`, and walks the files in that order,
   drawing one more `rng.integers(0, file_len)` start offset per file and consuming the
   whole file circularly from there (chunks of 8192 source tokens, decode -> re-encode)
   until the 2,067,750,000-token budget is hit.

Replaying both shuffles (without downloading anything -- shuffling only needs the sorted
52-name list and its length) reproduces **exactly** the three files the task record names:
`part-17-00011`, `part-06-00000`, `part-34-00008`, in that order. Only those 3 files (not
all 52) needed to be downloaded to finish the replay.

`transformers` >= 5 auto-detects that `allenai/gpt-neox-olmo-dolma-v1_5` is a BPE tokenizer
and **ignores** a `clean_up_tokenization_spaces=True` default for `.decode()` (it prints a
warning saying so) -- which is exactly the behavior the original build needed (a
`transformers` 4.x install would apply that destructive cleanup and corrupt the decoded
text). `rebuild.py` therefore does not pass `clean_up_tokenization_spaces` explicitly; it
relies on `requirements.txt`'s pinned `transformers==5.17.0` doing the right thing by
default. See the Verification table for the sampled byte-equality check.

## Rebuilding

StarCoder comes from `bigcode/starcoderdata`, which is gated under The Stack's terms of use.
Before rebuilding it, accept the dataset's terms on its Hugging Face page, then make a read
token available to the script: set `HF_TOKEN`, run `huggingface-cli login`, or pass `--hf-token`. The other six domains
download anonymously.

```
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python rebuild.py --out /path/to/out --cache /path/to/cache
# or one domain at a time while iterating:
python rebuild.py --out /path/to/out --cache /path/to/cache --only wiki
```

### Running domains in parallel / chunked (e.g. under Slurm)

Each domain is independent, so `--only DOMAIN` jobs can simply run concurrently sharing
one `--cache` (safe: downloads are content-addressed and idempotent). For a domain whose
tokenize step is itself slow (dclm's per-window decode/re-encode; any domain with many
documents), `rebuild.py` can also split that ONE domain's document list into contiguous,
equal-size chunks and tokenize them in parallel, since concatenating the chunks' token
streams in order is byte-identical to tokenizing the whole list at once:

```
# one call per chunk index 0..N-1 (e.g. one Slurm array task each), same --cache:
python rebuild.py --only dclm --chunk-index $i --chunk-count $N --part-dir PARTS --cache CACHE --out SCRATCH_OUT
# then once all N parts exist under PARTS/dclm/part-00000-of-000N0.u32le.bin ..:
python rebuild.py --only dclm --assemble --chunk-count $N --part-dir PARTS --out OUT --cache CACHE
```
`--out` in chunk mode is only used for scratch space and is not the final output; point it
at node-local disk (`$TMPDIR` or `/tmp`) and copy each finished `--part-dir` file to shared
scratch yourself once it's done, to avoid many parallel writers hitting the same
network filesystem path at once (this is what this repo's own verification run did).

## Verification (FarmShare, 2026-09-25)

Step 1 (sha256 every trained-on file, read-only) was run first and is unconditional on
everything below: Slurm jobs 1741746 (venv setup) and 1741747 (array 0-6, one task per
domain) streamed every surviving `tokenized/<domain>/<domain>.npy` once, hashing the whole
file and every shard/val-slice the layout rule implies. Their independent totals (24 train
objects, 7 val objects, 5,509,020,202 train tokens, 8,275,942 val tokens) match the task's
recorded totals exactly, which is a strong check that the assumed layout rule is the real
one (confirmed directly against `scripts/publish_refhq_edullm_data.py`'s source).

Document-to-source mapping + a random-sample byte-equality re-download check (Slurm jobs
1742207 array 0-4, 1742266, 1742664, 1742734, 1743091; `tools2/resolve.py`,
`tools2/dclm_resolve.py`; three bugs were caught and fixed mid-run -- see Known limits --
and rerun clean before being trusted):

| Domain | Docs mapped | Missing | Sampled re-download check | Result |
|---|---:|---:|---:|---|
| wiki | 73,824 / 73,824 | 0 | 300 | 300/300 byte-identical |
| algebraic-stack | 99,445 / 99,445 | 0 | 300 | 300/300 byte-identical |
| arxiv | 72,894 / 72,894 | 0 | 150 | 150/150 byte-identical |
| pes2o | 1,879,376 / 1,879,376 | 0 | 150 | 150/150 byte-identical |
| open-web-math | 125,554 / 125,554 | 0 | 300 | 300/300 byte-identical |
| dclm | 252,440 / 252,440 | 0 | 300 | 300/300 byte-identical under both `clean_up_tokenization_spaces=True` and `=False` (transformers 5.17 ignores the argument for this BPE tokenizer either way) |
| starcoder | 761,995 resolved + 42 templated = 762,037 / 762,037 | 0 | 300 | 300/300 byte-identical |

dclm's replay also independently reproduced the corpus's own recorded totals exactly:
252,440 documents, 2,067,750,000 realized (target) tokens, 2,067,985,387 source tokens
consumed from the same 3 files (`part-17-00011`, `part-06-00000`, `part-34-00008`) named in
the task record.

### Full rebuild-from-scratch sha256 verification -- ALL 7 DOMAINS PASS

One clean pass of the exact committed `rebuild.py` (sha256 `2fd028e59add611f70596c17
dcb76ff55fd51c1b8358c0c3ef70c391f9a803e6` -- verify with `sha256sum rebuild.py`), covering
every domain including starcoder, chunked and run fully in parallel (Slurm array tasks per
domain, `--chunk-index`/`--chunk-count`/`--part-dir`/`--assemble` as documented above; node-
local `/tmp` for each chunk's part file, copied to shared scratch once finished, then a
dependent `--assemble` job per domain that concatenates, shards, carves val, and checks
every sha256 against `outputs.json`):

| Domain | Chunks | Chunk array job | Assemble job | Result |
|---|---:|---|---|---|
| dclm | 24 | 1744942 | 1744966 | **OK** -- npy + all 8 train shards + val |
| pes2o | 12 | 1744967 | 1744968 | **OK** -- npy + both train shards + val |
| arxiv | 8 | 1744969 | 1744970 | **OK** -- npy + all 6 train shards + val |
| starcoder | 8 | 1744989 | 1744993 | **OK** -- npy + all 3 train shards + val |
| open-web-math | 4 | 1744994 | 1744995 | **OK** -- npy + both train shards + val |
| algebraic-stack | 2 | 1744996 | 1744997 | **OK** -- npy + both train shards + val |
| wiki | 2 | 1744998 | 1744999 | **OK** -- npy + train shard + val |

Every one of the 7 whole-domain `tokenized/<domain>/<domain>.npy` sha256 values, and every
one of the 31 publish-object (`tokens/<domain>/{train-*,val-00000}.u32le.bin`) sha256
values, matched `outputs.json` exactly -- **`rebuild.py` reproduces the trained-on corpus
byte-for-byte from public data alone, for all seven domains.**

This run reused `--cache` (`official-cache/`) that earlier per-domain passes had already
populated and sha256-verified file-by-file inside `rebuild.py`'s own
`download_and_verify()` (every `hf_hub_download` call is immediately checked against
`sources.json`'s recorded sha256 before use) -- re-downloading into a literally empty
directory would only re-exercise the same verified-download code path at the cost of
network time, not add confidence. `official-cache/` and this run's `final-out/` (the
assembled, sharded, sha256-confirmed output tree) are left on FarmShare scratch under this
work dir.

An earlier attempt to verify all 7 domains as plain (non-chunked) `--only DOMAIN` jobs hit
a 45-minute wall-clock limit on arxiv (its 72,894 documents average ~19,000 tokens each,
1.378B content tokens total -- far more per-document tokenizer work than its document
count suggests); re-run chunked (8-way) instead of raising the time limit, both to fit
comfortably inside a short job and to demonstrate the parallel path for every domain.

## Known limits

- **starcoder's 42 templated documents are not source code** -- see "StarCoder placeholder
  documents" above; this is a finding about the trained-on data itself (0.0055% of
  documents, 0.00006% of tokens), not a manifest gap. Every other document in the corpus
  (100% of tokens across all 7 domains) is fully mapped, byte-verified, and reproduced by
  `rebuild.py` end to end.
- The publish-stage log that would have recorded per-object sha256 at publish time
  (`hq-tok-up-1660899.*`) was deleted; `outputs.json`'s hashes are computed fresh from the
  surviving `tokenized/*.npy` using the layout rule recovered from
  `scripts/publish_refhq_edullm_data.py`'s source, not copied from any surviving
  publish-time record.
- `documents-<domain>.tsv.gz` stores the source file as a path string (looked up in
  `sources.json`) rather than a numeric index into it, for readability; this is a
  reformatting of the same information the spec's `src` column asks for, not a different
  fact.
- `rebuild.py`'s `DocStream` gathers every referenced row of a file into memory in one pass
  (documents are shuffled relative to a file's native row order, so a naive per-document
  seek would re-read a large file once per document). For a domain concentrated in one big
  file (pes2o: ~1.9M documents from one file) this holds that domain's whole text stream in
  memory at once (a few GB); it does not hold multiple domains at once.
- Five bugs were caught and fixed during this pass, three by the sampled-verification
  step, one by the full fresh-venv rebuild, and one by the starcoder placeholder
  investigation, all before being counted in the tables above:
  1. arxiv's id numeric suffix is not the physical line number in general (only sequential
     within whichever upstream proofpile shard a file was repacked from) -- an initial
     version assumed it was and got 72,766/72,894 wrong; fixed to resolve by exact id match
     (job 1742664).
  2. the first pes2o resolver re-streamed its ~1.9M-line source file once per sampled
     document during verification (job 1742383, cancelled); fixed to gather the whole
     sample in one pass (job 1742734). `rebuild.py`'s `DocStream` uses the same one-pass
     approach for every document (not just the sample), for the same reason.
  3. open-web-math re-crawls the same URL at different dates with different content (e.g.
     StackExchange "hot"/"new" tag-listing pages): 125,554 out documents have only 118,521
     distinct urls, so `url` alone is not a unique key. An initial version mapped by url
     only and got 18/300 sampled documents wrong (the wrong same-url row); fixed to key on
     `(url, metadata)` -- metadata is passed through verbatim from the source row and
     differs across re-crawls -- giving 125,554 unique keys and 0 missing (job 1743091).
  4. the pes2o resolver wrote `documents-pes2o.tsv.gz`'s `file` column as just the
     filename (`train-00000-of-00020.json.gz`) instead of the full path under the repo
     (`data/v2/train-00000-of-00020.json.gz`); `sha_for()` silently recorded `null`
     size/sha256 in `sources.json` for that entry instead of erroring, and the sampled
     verification didn't catch it because it looked files up by the internal integer index
     it already had, not by re-reading the written path string. The first full
     fresh-venv/empty-cache rebuild (job 1743976) caught it immediately: `rebuild.py`
     downloads strictly from `sources.json`'s recorded paths, and this one 404'd. Fixed in
     `resolve.py`/`build_manifest.py`, rerun (job 1744650), and confirmed against the full
     rebuild's second pass.
  5. starcoder's head-of-stream prefix was computed by grouping files by directory name
     and sorting directories separately, instead of sorting the full file-path strings as
     `datasets` actually does -- see "StarCoder placeholder documents" above for the full
     story. This made 396,846/762,037 (52%) of documents come back "unresolved" (their true
     source directory, `c-sharp`, was never in the assumed prefix at all) even though only
     42 are genuinely unresolvable. Fixed by sorting the global file listing as one list of
     path strings; confirmed against a live streaming trace before trusting it.
- FarmShare compute nodes `barley-01` through `barley-04` have broken `/scratch` mounts
  (confirmed by another agent working the same task on a different corpus): jobs on them
  failed instantly (`CANCELLED`/`OUT_OF_MEMORY` within seconds, no output) for reasons
  unrelated to this manifest's code. All Slurm submissions after discovering this exclude
  `barley-01,barley-02,barley-03,barley-04` in addition to the oat-* GPU nodes.
