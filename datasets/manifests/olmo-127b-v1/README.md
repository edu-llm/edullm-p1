# olmo-127b-v1

## What this corpus is

`pretrain/olmo-127b` v1 is the ~127B-token dolma2 reservoir published internally
for the domain-weighting paper (edullm-data, published 2026-07-31). It mirrors
seven sources of `allenai/olmo-mix-1124` (dclm, arxiv, starcoder, pes2o,
open-web-math, algebraic-stack, wiki): every row of 512 whole source files,
dolma2-tokenized with one EOS token appended per document, concatenated per
source, cut into <=1 GiB `train-NNNNN.u32le.bin` shards, with the last 0.15%
of each source's token stream carved into `val-00000.u32le.bin` so validation
mix weights match the full mix.

Trained on by:
- The four RunPod 370M static-mixture arms, which drew from it via
  OLMo-core's source-mixture sampler (seed 12536) over leading runs of the v1
  train shards, per domain.
- The two Skill-It 370M arms, which read all v1 train shards from the
  FarmShare publish-stage copy directly.

## Trained-on objects

`tokens/<source>/{train-NNNNN,val-00000}.u32le.bin`, uint32 little-endian
token ids, dolma2 vocabulary, EOS id 100257 (`allenai/dolma2-tokenizer`
@ `5292e5d6c0f40b67cc765fe41bec991cf4345b5c`). 481 objects total (7 sources x
(N train shards + 1 val shard)); see `outputs.json` for the exact name/bytes/
sha256/token count of every object, and `outputs.json:file_streams` for the
sha256 of each of the 512 source files' own token stream (content + one EOS
per document, in row order) as sliced out of the published concatenated
stream for that source.

Sources, in the join order used to build each source's stream (ascending
`join_key`, which is the string-sorted key the original publish step itself
sorted shards by): see `sources.json:files`. 512 files total:

| source | files | of which top-up |
|---|---|---|
| algebraic-stack | 16 | 0 |
| arxiv | 20 | 0 |
| dclm | 212 | 0 |
| open-web-math | 13 | 0 |
| pes2o | 13 | 8 |
| starcoder | 236 | 187 |
| wiki | 2 | 0 |

All 512 files are unmodified files of `allenai/olmo-mix-1124` at revision
`99ee6aaace88779d1ef099d36251b91101c1679b` (dataset repo). "Top-up" files were
added by a later pass that measured a shortfall against the planned mix and
pulled additional files for the same source to close it (pes2o started with
5 files pre-top-up, starcoder with 49; both needed more files, not fewer, to
reach their planned token share). Top-up files sort after all non-top-up
files of the same source because their `join_key` starts with `topup_...`,
which is lexicographically greater than any 5-digit numeric prefix.

## Manifest files

- `sources.json` -- every one of the 512 public input files (`repo_id`,
  `repo_type`, `revision`, `path`, `size`, `sha256` = the HF LFS oid), plus
  the tokenizer repo/revision/files.
- `documents_per_file.tsv.gz` -- stands in for `documents*.tsv.gz`. One row
  **per source file**, not per document: `src`, `hf_path`, `row_start`,
  `row_end` (0 and the file's row count -- every row of every file is kept,
  so a per-document table would just repeat "kept" 126M+ times), `skipped_rows`
  (row:reason for any blank/invalid-JSON/no-"text" row -- these are dropped,
  same as the original `tokenize_olmo_shard.py`), `ndocs`, `ntok` (excluding
  EOS), `eos_in_text` (documents whose raw text itself already contained the
  literal EOS id, counted for information; dolma2's BPE cannot merge across a
  raw id 100257 the way a string token would, so these pass through unchanged
  and are not treated as errors).
- `doc_windows.tsv.gz` -- per file, sha256 of consecutive spans of 65,536
  documents (`src`, `start_token`, `ntokens`, `sha256_16` = first 16 hex
  chars). Diagnostic only: narrows a mismatch to within 65,536 documents of a
  38B-token file before `rebuild.py --reference` does the token-exact
  document diagnosis.
- `outputs.json` -- every one of the 481 published objects (name, bytes,
  sha256, token count, source, position in that source's stream) plus the
  512 per-file stream hashes described above, plus the format/layout rule.
- `rebuild.py` -- standalone rebuild + verifier (see below).
- `build_manifest.py` -- the one-off script that produced
  `documents_per_file.tsv.gz`, `doc_windows.tsv.gz` and `outputs.json` (not
  needed to rebuild; documents provenance). References FarmShare paths.
- `verify_layout.py` -- cheap, network-free check that the layout rule in
  `outputs.json["layout"]` applied to `outputs.json["file_streams"]`'s
  per-file token counts reproduces every published object's name, byte
  length and byte offset (see Verification, check 3).
- `requirements.txt` -- exact pins, Python 3.12.3.

## Rebuild

```
python rebuild.py --out OUT --cache CACHE                       # all 481 objects
python rebuild.py --out OUT --cache CACHE --only wiki            # one source
python rebuild.py --out OUT --cache CACHE --only src:341 --hash-only   # one input file
```

`--only` accepts object names (`tokens/wiki/val-00000.u32le.bin`), source
names (`wiki`), or input files (`src:341` or the HF path). Exit status is
non-zero on any mismatch; the first mismatching input file and (with
`--reference <published tokens/ dir>`) the exact document are printed.

## Verification

Ran on FarmShare under
`/scratch/users/nzhao2/agent-runs/repro-manifests-20260925/olmo-127b-v1/`,
2026-09-25/26, fresh venv from `requirements.txt` (Python 3.12.3), empty HF
cache at the start of the run (later reused across steps -- every fetch is
sha256-checked against `sources.json` regardless of cache state, so reuse
never weakens any check). All three checks below are complete, with **zero
mismatches found anywhere**.

**1. Manifest construction + per-file check** (`build_manifest.py`): one
Slurm array task per one of the 512 source files downloaded each file from
`allenai/olmo-mix-1124` at the pinned revision (sha256-checked), tokenized it
exactly as `datasets/olmo/tokenize_olmo_shard.py` did, and recorded its own
token-stream sha256 (jobs 1742437 "starcoder", 236/236 `COMPLETED`; 1742474
"rest" -- the other six sources, 276/276 `COMPLETED`; 512/512 total, 0
failures). A second pass made one sequential read of each source's real
published `tokens/<source>/*.u32le.bin` bytes on FarmShare (job 1743169 for
starcoder + job 1743969 for the other six sources, 7/7 `COMPLETED`, 0
failures), hashing the exact byte range each file occupies in that
concatenated stream *and* every published object in the same pass. `assemble`
(job 1744065) then compared the two independently-computed per-file hashes
for **every one of the 512 files** and found **zero mismatches** -- including
the one edge case worth calling out: starcoder's last train shard was
truncated by the val carve (1,038,619,244 of 1,073,741,824 bytes) and still
matched exactly. This also produced the 481 object hashes in `outputs.json`,
each computed directly from the real published bytes on disk. This is what
settles the mid-`dclm`/mid-`pes2o` resume question from the Known limits
section: byte-identical, independent of how the original publish job
resumed.

**2. Standalone `rebuild.py` per-file check, all 512 files**: same fresh
venv, same warm download cache. A 3-file smoke test
(`--only src:341,src:476,src:369`) matched `outputs.json` first. The first
full-512 attempt (job 1744069, `--jobs 4` grouped array, 32 tasks x 16 files)
hit a real bug: combining `TOKENIZERS_PARALLELISM=true` with `rebuild.py`'s
Python-level `ProcessPoolExecutor` (`--jobs 4`) means every forked worker
also tries to run the `tokenizers` Rust thread pool; after the first
per-source batch this reliably wedged (36/512 files done, then no progress at
all for ~10h until `TIMEOUT`). `--jobs` is documented above as parallelism
via *processes*; it should not be combined with `TOKENIZERS_PARALLELISM`.
Fixed by resubmitting with `--jobs 1` (job 1744614: no multiprocessing,
parallelism only from the `tokenizers` library's own thread pool -- the same
pattern that worked for `build_manifest.py tokenize`). To maximize
parallelism against the account's 512-CPU cap (only ~176 of 512 CPUs were in
use), 1744614 was cancelled once its already-completed files (185/512) were
safely recorded -- `rebuild.py` does not skip already-verified files, so
in-flight-but-not-yet-written work was simply discarded, not double-counted
-- and the remaining 327 files were redistributed into a finer array
(1744664: 60 tasks x 3 CPU, ~5-6 files each) for roughly 3x throughput. Final
result: **512/512 files verified, 0 mismatches**, combining 1744614's
185 + 1744664's 327.

**3. Standalone `rebuild.py` object-level check, via `ObjectRouter`**:
running `--only <source>` for all 7 sources would have re-tokenized the
entire ~127B-token corpus a *third* time (~115 CPU-h estimated, with `dclm`
alone exceeding a 12h single-task budget since one task per source has no
further parallelism) purely to re-derive object hashes that step 1 already
computed from the real published bytes. To avoid that duplicate compute, this
was narrowed to two checks:
  - *Code-path check*: ran `ObjectRouter` end-to-end on the two smallest
    sources, `wiki` (job 1744662 task 0, 2 files, 15 objects) and
    `algebraic-stack` (task 1, 16 files, 49 objects) -- **64/64 objects
    matched exactly** (all of both sources' objects). This exercises the same
    routing code (multi-file objects, join order, sequential feed) that the
    other 5 sources also use, just not the other 5 sources' own bytes.
  - *Layout-rule check, all 7 sources, network-free* (`verify_layout.py`):
    recomputes every object's name/byte-length/byte-offset from
    `outputs.json`'s per-file token counts and the stated layout rule (1 GiB
    shards in path order, then `floor(total_bytes * 0.0015)` bytes -- rounded
    down to a multiple of 4 -- carved off the tail into `val-00000.u32le.bin`)
    with no download, no tokenizing, no FarmShare needed, and compares
    against `outputs.json["objects"]`. Result: **7/7 sources, 481/481
    objects, 0 mismatches** (`python verify_layout.py`).
  Between (a) code-path correctness on 2 real sources and (b) the layout
  arithmetic matching all 7 sources' real recorded object boundaries exactly,
  plus step 1 having already hashed all 481 objects against the real
  published bytes independently, object-level equality for all 7 sources is
  established without re-tokenizing `dclm`/`arxiv`/`pes2o`/`open-web-math`/
  `starcoder` a third time.

| output | expected sha256 (12 hex) | rebuilt sha256 (12 hex) | match |
|---|---|---|---|
| all 512 file streams, independently re-tokenized and byte-range-matched against real published objects (check 1) | `outputs.json:file_streams` | recomputed from public HF data | **512/512, 0 mismatches** |
| all 481 published objects, hashed directly from disk (check 1's domain scan) | `outputs.json:objects` | recomputed by streaming each source's real files once | **481/481** |
| all 512 files, standalone `rebuild.py --hash-only` (check 2; jobs 1744614+1744664) | `outputs.json:file_streams` | recomputed independently, fresh venv | **512/512, 0 mismatches** |
| `wiki` + `algebraic-stack` objects, standalone `rebuild.py`'s `ObjectRouter` (check 3a; job 1744662) | `outputs.json:objects` | recomputed independently, fresh venv | **64/64, 0 mismatches** |
| all 7 sources' 481 objects, layout-rule arithmetic (check 3b, `verify_layout.py`) | `outputs.json:objects` (name/bytes/offset) | recomputed from `outputs.json:file_streams` + the stated layout rule | **7/7 sources, 481/481 objects** |

Totals: 512 files, 481 objects, 126,650,724,924 tokens, 506,602,899,696 bytes
(matches the publish job's own accounting and the task's stated split:
189,976,084 val / 126,460,748,840 train tokens).

Scratch disk used under the work dir: 167 GB in the HF download cache
(compressed source files; smaller than the ~500 GB decompressed corpus) plus
a few MB of stats/JSON, stable across every rerun (nothing re-downloaded).
No second full copy of the decompressed token streams was kept anywhere
(`--hash-only` throughout).

Total compute: ~325 CPU-hours across every job below, of which ~33 CPU-h was
spent on the two superseded/cancelled attempts (1744069's `TIMEOUT`, and an
initial full 7-source `ObjectRouter` job cancelled per the scope change in
check 3).

Slurm job IDs (partition `normal`; `--exclude` always `oat-01..06` and, from
job 1742437 on, `barley-01..04` -- see Known limits):
venv setup 1742214 &middot;
tokenize `starcoder` 1742437 (236 tasks, `COMPLETED`, 13.7 CPU-h) &middot;
tokenize other 6 sources 1742474 (276 tasks, `COMPLETED`, 99.7 CPU-h) &middot;
domain scan `starcoder` 1743169 (`COMPLETED`, 0.1 CPU-h) &middot;
domain scan other 6 sources 1743969 (6 tasks, `COMPLETED`, 1.2 CPU-h) &middot;
`assemble` 1744065 (`COMPLETED`) &middot;
standalone smoke test (interactive, 3 files) &middot;
standalone full-512 attempt 1 1744069 (32 tasks, `TIMEOUT` after 10h due to
the `--jobs 4`/`TOKENIZERS_PARALLELISM` bug, 36/512 done, 32.1 CPU-h) &middot;
standalone full-512 attempt 2 1744614 (32 tasks, `--jobs 1`; cancelled by us
after 185/512 to free capacity for more parallelism, 71.2 CPU-h) &middot;
standalone remaining-327 fine array 1744664 (60 tasks x 3 CPU, `COMPLETED`,
88.4 CPU-h) &middot;
object check attempt 1 1744653 (7 tasks, full re-tokenize, cancelled per the
check-3 scope change, 1.3 CPU-h) &middot;
object check `wiki`+`algebraic-stack` 1744662 (2 tasks, `COMPLETED`,
16.9 CPU-h) &middot;
layout-rule check: local, then confirmed on FarmShare against the live
`outputs.json`/`sources.json` (negligible compute, no Slurm job needed).

## Known limits / things worth knowing

- **The FarmShare copy of `publish_olmohq_edullm_data.py` (the script that
  ran the publish job) no longer matches the repo's checked-in
  `datasets/olmohq/publish_olmohq_edullm_data.py`, and the FarmShare copy is
  currently broken**: `ShardWriter.__init__` and `stage_publish_layout` both
  read `out_dir = out_di` (undefined name; `NameError` on any call). Job
  timestamps make clear this is not what ran: the four staging jobs that
  actually built `publish-stage/tokens/` (1669864, 1670455, 1670482, 1670526)
  and the one that finished staging + carved val (1670701) all completed
  before the script's current mtime (2026-07-31 05:39:56); the working
  scripts they used are gone. The final job (1670744) used
  `publish_olmohq_skip_stage.sbatch --skip-stage`, which never calls
  `ShardWriter`/`stage_publish_layout` at all -- it only reads the
  already-staged directory and calls `publish()` -- so the broken code path
  was never exercised and the currently-published bytes are unaffected. The
  rest of the deployed script (shard/val-cut math, `sorted(domains)` join
  order, 1 GiB shard cap, 0.15% val fraction) is otherwise identical to the
  repo's version; the only other differences are AWS STS session-refresh
  plumbing (`RefreshingS3`/`RefreshingBoto3S3`, needed because the ~255 GB
  publish outlasts a 1-hour STS token) that this rebuild has no use for
  (rebuild.py never touches AWS/S3). This verification does not depend on
  the deployed script at all -- it checks the actual published bytes -- so
  it settles the question either way regardless of this discrepancy.
- `datasets/olmo/tokenize_olmo_shard.py` differs from the FarmShare copy that
  ran (`fs_tokenize_olmo_shard.py`, pulled during the earlier audit) only in
  line endings (sha256 differs, diff shows no content change); the tokenize
  logic itself is identical.
- The publish run resumed mid-`dclm` and mid-`pes2o` (S3 `ExpiredToken` /
  `HeadObject` 400 errors across jobs 1669864 -> 1670701, restarting the AWS
  session each time). Resume was designed to be byte-exact but was
  unverified before this check; since this rebuild re-derives every source
  file's tokens independently from public HF data and compares against the
  actual published bytes (not against the publish job's own bookkeeping),
  a clean pass on `dclm` and `pes2o` settles it.
- The library versions that ran are unverifiable with certainty (no
  `pip freeze` survives); `requirements.txt` pins the versions recorded in
  the venv of `olmohq-topup-20260728-185642`, the last (and working) topup
  job, which matches the versions in every other surviving tokenize venv
  (`edullm-dataset-olmo*-dolma2-tok-*`) except `huggingface_hub`
  (1.24.0 -> 1.25.1) and `tqdm` (4.69.1 -> 4.70.0); neither affects
  tokenization output.
- A `publish-stage/text/` directory also exists next to `publish-stage/tokens/`
  (populated 2026-07-31 23:52, after `tokens/` finished at 03:55). It is not
  part of `pretrain/olmo-127b` v1's trained-on objects (only `tokens/` is)
  and is out of scope for this manifest.
