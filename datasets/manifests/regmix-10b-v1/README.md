# regmix-10b-v1

Document-level manifest and byte-for-byte rebuild recipe for `pretrain/regmix-10b`
v1, the 10B-token corpus that all seven 370M-parameter token-selection arms in the
P1 paper trained on. It is a subsample of `allenai/olmo-mix-1124` (ODC-By v1.0; DCLM also
subject to the Common Crawl terms of use) with Data Mixing Laws domain weights (the paper's
Tables 3 and 4). We do not redistribute the corpus; this manifest is how to rebuild it.

## What this corpus is

- **Domains (7):** `dclm`, `arxiv`, `starcoder`, `pes2o`, `open-web-math`,
  `algebraic-stack`, `wiki`. Each domain is a prefix of a recorded, ordered list of
  source files from `allenai/olmo-mix-1124` (revision
  `99ee6aaace88779d1ef099d36251b91101c1679b`, 54 files total, unmodified), trimmed to
  a per-domain token budget by a seed-42 shuffle over that domain's documents (see
  "Selection rule" below).
- **Tokenizer:** `allenai/dolma2-tokenizer` @ `5292e5d6c0f40b67cc765fe41bec991cf4345b5c`,
  fast tokenizer, `add_special_tokens=False`, EOS id `100257` appended after every
  document (no BOS).
- **Trained-on objects (55 total):**
  - 7 headerless `tokenized/<domain>/<domain>.npy` memmaps (uint32 little-endian),
    each domain's kept documents concatenated in output order.
  - 48 `tokens/<domain>/{train-NNNNN,val-00000}.u32le.bin` objects: the same 7
    per-domain streams cut into 1 GiB train shards plus a validation split carved
    from the tail (`0.15%` of the domain's tokens, rounded down to a multiple of 4
    bytes). This is the `pretrain/regmix-10b` v1 publish layout the runs actually
    read from.
  - Realized totals: 4,748,990 docs; 10,000,058,051 content tokens;
    10,004,807,041 tokens with EOS; 9,989,799,834 train tokens.

## Files in this directory

| File | Content |
|---|---|
| `sources.json` | The 54 `allenai/olmo-mix-1124` files (repo/revision/path/size/sha256), plus the tokenizer repo+revision+files. |
| `documents/<domain>.tsv.gz` | One row per kept document, in output order: `src` (index into `sources.json["files"]`), `row` (0-based line number in that file's decompressed JSONL stream), `ntok` (content tokens, excluding EOS), `crc32` (of the document's tokens + its own EOS, as written to the trained-on stream). |
| `outputs.json` | Every trained-on object: name, bytes, sha256, token/doc counts, format (dtype/endianness/EOS id), and the tokenized/publish layout rules. Also carries, per domain, the file materialization order, per-file document counts, and (for 5 domains) the embedded-EOS finding below. |
| `rebuild.py` | Standalone rebuild + verifier. `python rebuild.py --out DIR --cache DIR [--only NAME]`. |
| `requirements.txt` | Exact pins for everything `rebuild.py` imports. Python 3.12.3. |
| `build_manifest.py` | The one-off script run on FarmShare to derive the files above from the surviving build artifacts. Not needed to rebuild; kept for provenance. Also has the `retok` stage used to fix the embedded-EOS finding below. |

## Selection rule (recorded, not recomputed)

For each domain: take the files in `outputs.json["domains"][d]["selection"]["files_in_materialization_order"]`
in that order; every JSON line in each file whose extracted text field
(`text`/`content`/`code`/`body`, first non-empty) is non-empty is one candidate
document, indexed by its 0-based line number in the *decompressed* stream (blank
lines and lines with no usable text field are skipped and do not get a row number).
Concatenate all candidates across the domain's files in that file order into one
list of `docs_before` documents. Then:

```
order = list(range(docs_before))
random.Random(42).shuffle(order)
keep  = order[:docs_after]     # kept, IN THIS ORDER == the domain's output order
```

No document is truncated: the original trim stopped after the first whole document
that brought the domain's content-token count to at least its target; the
0.98/1.02 tolerance-skip rule in the original trim script never fired for this
corpus (`docs_scanned == docs_after` in every domain). StarCoder's file list
reflects a manual top-up of 10 pool shards followed by a retrim (job `1660091`);
the other six domains used the original plan-rank file order.

## A genuine artifact: EOS-valued tokens embedded inside documents

While deriving the manifest, `build_manifest.py`'s `hash` stage found more raw
occurrences of the EOS token id in five of the seven tokenized `.npy` files than
there are documents in that domain (compared against `plan/trim_results.json`,
cross-checked against the build's own per-domain stats file and
`plan/summary_final.json`, which agree with each other exactly):

| domain | documents | raw EOS occurrences | delta |
|---|---:|---:|---:|
| dclm | 2,800,640 | 2,800,640 | 0 |
| arxiv | 130,087 | 130,088 | 1 |
| starcoder | 985,925 | 985,928 | 3 |
| pes2o | 150,710 | 150,711 | 1 |
| open-web-math | 97,937 | 97,957 | 20 |
| algebraic-stack | 223,705 | 223,744 | 39 |
| wiki | 359,986 | 359,986 | 0 |

**Root cause:** the pinned `dolma2-tokenizer` has the string `<|endoftext|>`
registered as special token id `100257` (the same id used as this corpus's EOS
separator). Hugging Face "fast" tokenizers still recognize a special token's
literal text *inside raw input* even when called with `add_special_tokens=False`
-- that flag only suppresses tokens the tokenizer would otherwise *add itself*
(e.g. an automatic BOS/EOS), not ones already spelled out in the document text.
A small number of documents in these five domains contain that literal substring
in their own text (visible, unrelated to any document-boundary marker), so the
tokenizer emits an extra `100257` in the middle of the token stream for that
document. Naively scanning the tokenized stream for EOS occurrences (cheap,
doesn't require re-downloading or re-tokenizing anything) therefore overcounts
document boundaries wherever this occurs.

**This does not block a byte-identical rebuild.** `rebuild.py` tokenizes each
document independently from its own text and appends its own EOS, so it
reproduces the embedded `100257` as a normal, deterministic side effect of
tokenizing that document's actual text -- exactly as the original build did. The
only thing that needed fixing was *our* post-hoc manifest derivation: for the five
affected domains, `build_manifest.py --stage retok` retokenizes every document in
the domain directly (instead of scanning the archived `.npy` for EOS positions) to
get each document's true `ntok`/`crc32`. `dclm` and `wiki` keep using the cheap
EOS-scan (delta 0, already exact). Affected documents and a short decoded context
window around the embedded token are listed under
`outputs.json["domains"][d]["embedded_eos_examples"]` for the five affected
domains.

## Rebuild

```
python -m venv .venv && . .venv/bin/activate   # /usr/bin/python3.12 on FarmShare
pip install -r requirements.txt
python rebuild.py --out OUT --cache CACHE
```

`--only NAME[,NAME...]` restricts to one or more domains (e.g. `wiki`), a
`tokenized/...` name, or a `tokens/...` publish-object name; the whole domain is
rebuilt (fetch/tokenize/assemble/npy are all domain-scoped). On a Slurm cluster,
run the stages separately over `--task 0..count-1` (`--stage plan` prints the
counts per stage): `fetch`, `tokenize`, `assemble`, `npy`, then a single `verify`
pass. `rebuild.py` checks every source file's sha256 against `sources.json` before
tokenizing it, and every output's sha256 against `outputs.json` at the end,
printing the first mismatching document (by expected ntok/crc32) for anything that
doesn't match.

## Verification result

**Acceptance pass (authoritative): the single committed `rebuild.py`
(sha256 `ccba6c75346b6728243608de02526ea9cfb666f18629658c93a08f26f11a52a3`,
verified on FarmShare before and after this run) rebuilt all 7 domains into a
brand-new, empty `--out` directory in one clean end-to-end pass: 55/55 outputs
sha256-identical, 0 documents differ from the manifest.** Fresh
`/usr/bin/python3.12 -m venv`, `pip install -r requirements.txt` (no other
packages), and the committed `sources.json`/`outputs.json`/`documents/*.tsv.gz`
only for input. `--cache` pointed at this task's already-downloaded, already
sha256-verified Hugging Face cache from the iterative pass below (`rebuild.py`
independently re-checks every source file's sha256 against `sources.json` the
first time it touches a fresh `--out` tree regardless of cache reuse, so this
does not skip that check -- it only skips re-downloading bytes already proven
correct). A genuinely from-scratch run needs only an empty `--cache`, which the
iterative pass below exercised for all 55 outputs.

Fanned out per domain, 3 chained Slurm stages each
(`tokenize`→`assemble`→`npy`, `--dependency=afterok`, local-disk writes as
described below, `barley-01..04`/`oat-*` excluded): `dclm` tok/asm/npy
1744854/1744855/1744856 (30+15+1 tasks); `arxiv` 1744857/1744858/1744859
(14+11+1); `starcoder` 1744860/1744861/1744862 (16+7+1); `pes2o`
1744863/1744864/1744865 (5+5+1); `open-web-math` 1744866/1744867/1744868
(4+4+1); `algebraic-stack` 1744869/1744870/1744871 (4+4+1); `wiki`
1744872/1744873/1744874 (1+2+1). Final consolidated `--stage verify` (no
`--only`, all 7 domains, reading this run's own outputs) at
2026-09-26T11:07:48-0700: `all_match: true`.

### Iterative pass (history / where the from-scratch, empty-cache proof lives)

The acceptance pass above reused a download cache rather than starting from an
empty one; the run below is where every one of the 55 outputs was proven
byte-identical starting from a genuinely empty `--cache` (downloading and
sha256-checking all 54 source files + the tokenizer from scratch), while
`rebuild.py` was still being hardened against the NFS issue described further
below. Different domains here ran under different (successively more robust)
revisions of `rebuild.py` as that hardening landed; the acceptance pass above
is the one to cite for "the committed script reproduces the corpus."

`wiki` (small-scale proof) 1743980; `algebraic-stack` 1744059;
`open-web-math` 1744064; `pes2o` 1744165; `starcoder` 1744647; `dclm` 1744660
(run alone, see below); `arxiv` as a 3-stage array 1744778 (tokenize, 14 tasks)
→ 1744779 (assemble, 11 tasks) → 1744780 (npy); final consolidated `--stage
verify` (no `--only`, all 7 domains) on 2026-09-26T10:42:41-0700.

The table below is the same in both passes -- tokenization is a pure function
of pinned tokenizer + source bytes + manifest, so the acceptance pass's outputs
hashed identically to this one, which is itself a second confirmation of
determinism across two independent output trees.

| output | expected sha256 (12 hex) | rebuilt sha256 (12 hex) | match |
|---|---|---|---|
| tokenized/dclm/dclm.npy | 131cb8aae5b7 | 131cb8aae5b7 | yes |
| tokenized/arxiv/arxiv.npy | 9a76d5aed5b0 | 9a76d5aed5b0 | yes |
| tokenized/starcoder/starcoder.npy | 6d12a1c93102 | 6d12a1c93102 | yes |
| tokenized/pes2o/pes2o.npy | ab42d8586c7f | ab42d8586c7f | yes |
| tokenized/open-web-math/open-web-math.npy | aac692663221 | aac692663221 | yes |
| tokenized/algebraic-stack/algebraic-stack.npy | 279119d804f5 | 279119d804f5 | yes |
| tokenized/wiki/wiki.npy | c1382e6c201c | c1382e6c201c | yes |
| tokens/algebraic-stack/train-00000.u32le.bin | 4308659988f1 | 4308659988f1 | yes |
| tokens/algebraic-stack/train-00001.u32le.bin | daced2447d35 | daced2447d35 | yes |
| tokens/algebraic-stack/train-00002.u32le.bin | 3a940953a323 | 3a940953a323 | yes |
| tokens/algebraic-stack/val-00000.u32le.bin | 997509a9916e | 997509a9916e | yes |
| tokens/arxiv/train-00000.u32le.bin | 42051ad044b5 | 42051ad044b5 | yes |
| tokens/arxiv/train-00001.u32le.bin | 3ad8e5a63ca7 | 3ad8e5a63ca7 | yes |
| tokens/arxiv/train-00002.u32le.bin | 8eae99ede67b | 8eae99ede67b | yes |
| tokens/arxiv/train-00003.u32le.bin | 2aef3fc2d3e9 | 2aef3fc2d3e9 | yes |
| tokens/arxiv/train-00004.u32le.bin | d8a07dd6cfa2 | d8a07dd6cfa2 | yes |
| tokens/arxiv/train-00005.u32le.bin | f02a78b68f05 | f02a78b68f05 | yes |
| tokens/arxiv/train-00006.u32le.bin | 85f871a7d02b | 85f871a7d02b | yes |
| tokens/arxiv/train-00007.u32le.bin | 5eedec879dbc | 5eedec879dbc | yes |
| tokens/arxiv/train-00008.u32le.bin | 3325af57d75c | 3325af57d75c | yes |
| tokens/arxiv/train-00009.u32le.bin | d8822b65f3ac | d8822b65f3ac | yes |
| tokens/arxiv/val-00000.u32le.bin | 684844c335a0 | 684844c335a0 | yes |
| tokens/dclm/train-00000.u32le.bin | b5703b543ab7 | b5703b543ab7 | yes |
| tokens/dclm/train-00001.u32le.bin | 0a4fe27930c5 | 0a4fe27930c5 | yes |
| tokens/dclm/train-00002.u32le.bin | 1db1d6a24937 | 1db1d6a24937 | yes |
| tokens/dclm/train-00003.u32le.bin | 0e97407dcc91 | 0e97407dcc91 | yes |
| tokens/dclm/train-00004.u32le.bin | bc8569cbeece | bc8569cbeece | yes |
| tokens/dclm/train-00005.u32le.bin | 908fb755d81f | 908fb755d81f | yes |
| tokens/dclm/train-00006.u32le.bin | 23c8f1576ce1 | 23c8f1576ce1 | yes |
| tokens/dclm/train-00007.u32le.bin | caa78e0ae1c1 | caa78e0ae1c1 | yes |
| tokens/dclm/train-00008.u32le.bin | 4c7d2bb12545 | 4c7d2bb12545 | yes |
| tokens/dclm/train-00009.u32le.bin | d9728514c65f | d9728514c65f | yes |
| tokens/dclm/train-00010.u32le.bin | da3ee691e143 | da3ee691e143 | yes |
| tokens/dclm/train-00011.u32le.bin | b9cf260df1db | b9cf260df1db | yes |
| tokens/dclm/train-00012.u32le.bin | 8d63293c1131 | 8d63293c1131 | yes |
| tokens/dclm/train-00013.u32le.bin | 1c257d8d3602 | 1c257d8d3602 | yes |
| tokens/dclm/val-00000.u32le.bin | 288d6fbed854 | 288d6fbed854 | yes |
| tokens/open-web-math/train-00000.u32le.bin | 5ff06c06edde | 5ff06c06edde | yes |
| tokens/open-web-math/train-00001.u32le.bin | 3be7e03022f8 | 3be7e03022f8 | yes |
| tokens/open-web-math/train-00002.u32le.bin | ebca7f5ef2c6 | ebca7f5ef2c6 | yes |
| tokens/open-web-math/val-00000.u32le.bin | 10057fb04c8d | 10057fb04c8d | yes |
| tokens/pes2o/train-00000.u32le.bin | 7f30d2b95105 | 7f30d2b95105 | yes |
| tokens/pes2o/train-00001.u32le.bin | 381a933b99d9 | 381a933b99d9 | yes |
| tokens/pes2o/train-00002.u32le.bin | 73ca3e7c0ce2 | 73ca3e7c0ce2 | yes |
| tokens/pes2o/train-00003.u32le.bin | f576bee83fcf | f576bee83fcf | yes |
| tokens/pes2o/val-00000.u32le.bin | 8ec4139a51a0 | 8ec4139a51a0 | yes |
| tokens/starcoder/train-00000.u32le.bin | 402d94e6c37f | 402d94e6c37f | yes |
| tokens/starcoder/train-00001.u32le.bin | e4d47a871460 | e4d47a871460 | yes |
| tokens/starcoder/train-00002.u32le.bin | 891ea1f16bed | 891ea1f16bed | yes |
| tokens/starcoder/train-00003.u32le.bin | 6e949293b27a | 6e949293b27a | yes |
| tokens/starcoder/train-00004.u32le.bin | 6741ca715b7f | 6741ca715b7f | yes |
| tokens/starcoder/train-00005.u32le.bin | 79bfbcd837e4 | 79bfbcd837e4 | yes |
| tokens/starcoder/val-00000.u32le.bin | b9de04633d89 | b9de04633d89 | yes |
| tokens/wiki/train-00000.u32le.bin | 8d5feebbad1d | 8d5feebbad1d | yes |
| tokens/wiki/val-00000.u32le.bin | 83d8c753c898 | 83d8c753c898 | yes |

55/55 match; 0 documents differ from the manifest anywhere.

### NFS write contention under heavy parallel load (operational, not a data issue)

The first full-parallelism attempt (6-7 domains at once, each with an 4-16-worker
`ProcessPoolExecutor` writing directly to preallocated files on FarmShare's
NFS-backed `/scratch`) repeatedly hit `OSError: [Errno 116] Stale file handle`
and, once, a reopen that itself transiently failed with `FileNotFoundError`.
Evidence ruled out a logic bug: unrelated domains running the identical code path
at the same time succeeded, and the failures always came in bursts across many
different files/workers at once -- consistent with a brief, shared-mount-wide
hiccup (it also coincided with two unrelated drops of this task's SSH control
socket), not a per-file race. Fixed in two layers, both now permanent parts of
`rebuild.py`:

1. Recognized that once an NFS file handle goes stale, retrying the same write on
   the same file descriptor can never succeed -- only closing and reopening the
   path gets a fresh handle -- and made the reopen itself retry with backoff too
   (it can transiently fail the same way).
2. More fundamentally, every write stage (`tokenize`, `assemble`, `npy`) now does
   its actual I/O -- `ftruncate`, `pwrite`, `fsync` -- on node-local disk
   (`$SLURM_TMPDIR`/`$TMPDIR`/`/tmp`), then copies the one finished file to its
   NFS destination in a single retried copy+atomic-rename. This trades many
   small concurrent network writes (what triggers staleness under load) for one
   big local write plus one bulk copy per output, and was what let `dclm` and
   `arxiv` (the two domains that kept failing under contention) finish cleanly:
   `dclm` run alone, `arxiv` split into a 3-stage Slurm array
   (tokenize→assemble→npy, 14+11+1 tasks chained with `--dependency=afterok`) so
   no single task held the shared mount under sustained multi-worker load.

Total across every job for this corpus, both passes, including the retries
above: **~82 CPU-hours**, ~157 GB of FarmShare scratch (22 GB
manifest-derivation workspace `mb/`; 75 GB iterative-pass workspace `verify/`
-- empty-cache downloads of all 54 source files + tokenizer, plus that pass's
rebuilt outputs; 60 GB acceptance-pass workspace `verify/out2/` -- rebuilt
outputs only, reusing the already-downloaded cache).

## Known limits / operational notes

- **Node `barley-01` on the `normal` partition does not mount `/scratch` or
  `/home` for this user** (confirmed via `scontrol show node` + a minimal `srun`
  probe: `/software` is visible, `/scratch/users/nzhao2` and
  `/home/users/nzhao2` are not). Every array task that landed there failed in 8s
  with signal 53 and no log file (Slurm couldn't even open the output file).
  Other agents working on sibling corpora independently found the same fault on
  `barley-02..04`. All Slurm submissions for this corpus exclude
  `barley-01..04` in addition to the `oat-*` GPU nodes. This is reported for the
  cluster owner; it was not fixed (out of scope).
- Publish shard/val boundaries fall at **token** boundaries, not document
  boundaries; `outputs.json`'s publish entries carry `docs_touched` (documents
  with at least one token in that shard) rather than a clean document count for
  this reason.
- `documents/<domain>.tsv.gz` lists every kept document individually (not
  ranges): the seed-42 shuffle means a domain's kept documents are not a
  contiguous run within any single source file.
