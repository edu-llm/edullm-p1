#!/usr/bin/env python3
"""One-off script actually run on FarmShare (2026-09-25) to derive this manifest from the
surviving refhq-regmix-5p5b-v1 build artifacts. References FarmShare-only paths (the
read-only build dir /scratch/users/nzhao2/refhq-regmix-5p5b-v1 and this agent's work dir
under /scratch/users/nzhao2/agent-runs/repro-manifests-20260925/refhq-regmix-5p5b-v1). Not
needed to rebuild -- that's rebuild.py, which only touches public HF data plus this
manifest's own files. Kept close to what was actually run, for provenance; run via:

    python build_manifest.py sources                # tools2/sources_survey.py
    python build_manifest.py resolve <domain>        # tools2/resolve.py <domain>
    python build_manifest.py dclm                    # tools2/dclm_resolve.py
    python build_manifest.py starcoder               # tools2/starcoder_resolve.py

domain in {algebraic-stack, wiki, open-web-math, pes2o, arxiv}. `starcoder` needs a valid
HF token at /scratch/users/nzhao2/refhq-regmix-5p5b-v1/.hf_token (bigcode/starcoderdata is
gated) -- the token pinned there was expired for most of this pass (HTTP 401 on every
call, including plain listing) until the user supplied a fresh one; see README.md's
"StarCoder placeholder documents" section for what running this step found.

Output, per step:
  meta/sources_survey.json                    -- file listing + sha256 for every pinned
                                                   public repo (anonymous; starcoder tried
                                                   with the pinned token, recorded as failed)
  meta/manifest/<domain>.docs.jsonl.gz         -- one row per out/<domain> document, in
                                                   output order, resolved to (file, row)
  meta/manifest/<domain>.verify.json           -- random-sample re-download byte-equality
                                                   check against the resolved (file, row)
  meta/manifest/dclm.replay_stats.json         -- dclm double-shuffle replay stats
  meta/manifest/dclm.verify_cleanup_{T,F}.json -- dclm sample check under both
                                                   clean_up_tokenization_spaces settings
"""
from __future__ import annotations

import gzip
import json
import random
import sys
import time
from pathlib import Path

BD = Path("/scratch/users/nzhao2/refhq-regmix-5p5b-v1")  # read-only build dir
W = Path("/scratch/users/nzhao2/agent-runs/repro-manifests-20260925/refhq-regmix-5p5b-v1")  # this agent's work dir
RAW = W / "raw"
RAW.mkdir(parents=True, exist_ok=True)

REV = {
    "wiki": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    "algebraic-stack": "260231ddd1c6382ad1454112b1e756d1ec33ed3e",
    "open-web-math": "fde8ef8de2300f5e778f56261843dab89f230815",
    "pes2o": "636a503e44a3ca1b58e01fb61eab0825cd574de0",
    "arxiv": "99ee6aaace88779d1ef099d36251b91101c1679b",
}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ============================================================ step: sources
def step_sources():
    from huggingface_hub import HfApi

    OUT = W / "meta"
    OUT.mkdir(parents=True, exist_ok=True)

    REPOS = {
        "allenai/DataDecide-data-recipes": ("3baf34baf5b636f0943401b5c6a2ccb7e5cf3bb9", "preprocessed/dclm/v0_rep32_ft7percentile_fw2/gpt-neox-olmo-dolma-v1_5", "dataset", False),
        "allenai/olmo-mix-1124": ("99ee6aaace88779d1ef099d36251b91101c1679b", "data/arxiv/train", "dataset", False),
        "allenai/peS2o": ("636a503e44a3ca1b58e01fb61eab0825cd574de0", "data/v2", "dataset", False),
        "open-web-math/open-web-math": ("fde8ef8de2300f5e778f56261843dab89f230815", "data", "dataset", False),
        "typeof/algebraic-stack": ("260231ddd1c6382ad1454112b1e756d1ec33ed3e", "train", "dataset", False),
        "wikimedia/wikipedia": ("b04c8d1ceb2f5cd4588862100d08de323dccfbaa", "20231101.en", "dataset", False),
        "allenai/dolma2-tokenizer": ("5292e5d6c0f40b67cc765fe41bec991cf4345b5c", None, "model", False),
        "allenai/gpt-neox-olmo-dolma-v1_5": ("8571ec72989ee67572049df907dec46cebe92335", None, "model", False),
        "bigcode/starcoderdata": ("9fc30b578cedaec69e47302df72cf00feed7c8c4", None, "dataset", True),
    }

    tok_path = BD / ".hf_token"
    TOKEN = tok_path.read_text().strip() if tok_path.exists() else None

    def listing(api, repo, rev, sub, repo_type):
        items = api.list_repo_tree(repo, path_in_repo=sub, recursive=True, revision=rev, repo_type=repo_type, expand=False)
        out = []
        for it in items:
            if not hasattr(it, "size"):
                continue
            lfs = getattr(it, "lfs", None)
            sha = None
            if lfs is not None:
                sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            out.append({"path": it.path, "size": it.size, "sha256": sha})
        out.sort(key=lambda x: x["path"])
        return out

    res = {}
    api_anon = HfApi(token=None)
    api_tok = HfApi(token=TOKEN)
    for repo, (rev, sub, rtype, gated) in REPOS.items():
        api = api_tok if gated else api_anon
        try:
            files = listing(api, repo, rev, sub, rtype)
            res[repo] = {"revision": rev, "repo_type": rtype, "gated": gated, "ok": True, "n_files": len(files), "files": files}
            log(repo, "OK", len(files), "files")
        except Exception as exc:  # noqa: BLE001
            res[repo] = {"revision": rev, "repo_type": rtype, "gated": gated, "ok": False, "error": repr(exc)[:500]}
            log(repo, "FAILED", repr(exc)[:200])
    (OUT / "sources_survey.json").write_text(json.dumps(res, indent=1))
    log("wrote", OUT / "sources_survey.json")


# ============================================================ step: resolve (5 domains)
def _read_out_docs(domain: str) -> list[dict]:
    shards = sorted((BD / "out" / domain).glob("documents-*.json.gz"))
    if not shards:
        raise SystemExit(f"no out shards for {domain}")
    docs = []
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
    return docs


def _sample_check(records, get_text_fn, k=300, seed=0):
    rng = random.Random(seed)
    idxs = list(range(len(records)))
    rng.shuffle(idxs)
    idxs = idxs[:k]
    checked = matched = 0
    mismatches = []
    for i in idxs:
        rec = records[i]
        try:
            actual = get_text_fn(rec["lookup"])
        except Exception as exc:  # noqa: BLE001
            mismatches.append({"idx": rec["idx"], "id": rec["id"], "error": repr(exc)[:300]})
            checked += 1
            continue
        checked += 1
        if actual == rec["text"]:
            matched += 1
        else:
            mismatches.append({"idx": rec["idx"], "id": rec["id"], "expected_head": rec["text"][:80],
                                "actual_head": (actual or "")[:80]})
    return {"checked": checked, "matched": matched, "mismatches": mismatches[:20]}


def step_resolve(domain: str):
    from huggingface_hub import HfFileSystem, hf_hub_download
    import pyarrow.parquet as pq

    OUT = W / "meta" / "manifest"
    OUT.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem(token=None)

    if domain == "algebraic-stack":
        docs = _read_out_docs(domain)
        log(domain, "out docs:", len(docs))
        files = sorted(fs.ls(f"datasets/typeof/algebraic-stack@{REV[domain]}/train", detail=False),
                       key=lambda p: p.rsplit("/", 1)[-1])
        counts = []
        for p in files:
            with fs.open(p, "rb") as f:
                counts.append(pq.ParquetFile(f).metadata.num_rows)
        cum = [0]
        for c in counts:
            cum.append(cum[-1] + c)

        def locate(gidx):
            lo, hi = 0, len(files)
            while lo < hi:
                mid = (lo + hi) // 2
                if cum[mid + 1] <= gidx:
                    lo = mid + 1
                else:
                    hi = mid
            return lo, gidx - cum[lo]

        out_rows = []
        for idx, d in enumerate(docs):
            gidx = int(d["id"].rsplit("-", 1)[-1])
            fi, row = locate(gidx)
            out_rows.append({"idx": idx, "id": d["id"], "ntok": d["n_tokens"],
                              "file": files[fi].split(f"@{REV[domain]}/", 1)[-1], "row": row})
        with gzip.open(OUT / f"{domain}.docs.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")

        cache: dict = {}

        def get_text(lookup):
            fi, row = lookup
            p = files[fi]
            if p not in cache:
                cache[p] = pq.ParquetFile(fs.open(p, "rb"))
            pf = cache[p]
            rg, base = 0, 0
            while rg < pf.metadata.num_row_groups:
                n = pf.metadata.row_group(rg).num_rows
                if base + n > row:
                    break
                base += n
                rg += 1
            return pf.read_row_group(rg, columns=["text"]).column("text")[row - base].as_py()

        records = [{"idx": idx, "id": d["id"], "text": d["text"], "lookup": locate(int(d["id"].rsplit("-", 1)[-1]))}
                   for idx, d in enumerate(docs)]
        result = _sample_check(records, get_text, k=300)
        (OUT / f"{domain}.verify.json").write_text(json.dumps(result, indent=1))
        log(domain, "verify:", result["checked"], result["matched"], "mismatches:", len(result["mismatches"]))
        return

    if domain in ("wiki", "open-web-math"):
        docs = _read_out_docs(domain)
        log(domain, "out docs:", len(docs))
        # open-web-math has re-crawled the same URL at different dates with different
        # content (e.g. StackExchange "hot"/"new" tag-listing pages), so `url` alone is not
        # a unique key (125,554 out docs have only 118,521 distinct urls). Key on
        # (url, metadata) instead -- metadata is passed through verbatim from the source row
        # and differs across re-crawls (it embeds warc_path/date), so it disambiguates.
        # wiki's own page id has no such duplicates; a single-field key is fine there.
        composite = domain == "open-web-math"
        key_field = "id" if domain == "wiki" else "url"
        needed = {(d["id"], d["metadata"]) for d in docs} if composite else {d["id"] for d in docs}
        prefix = "20231101.en" if domain == "wiki" else "data"
        repo = "wikimedia/wikipedia" if domain == "wiki" else "open-web-math/open-web-math"
        files = sorted(f for f in fs.ls(f"datasets/{repo}@{REV[domain]}/{prefix}", detail=False) if f.endswith(".parquet"))
        key_to_loc: dict = {}
        cols = [key_field, "metadata"] if composite else [key_field]
        for fi, p in enumerate(files):
            if len(key_to_loc) >= len(needed):
                break
            with fs.open(p, "rb") as fobj:
                pf = pq.ParquetFile(fobj)
                base = 0
                for rg in range(pf.metadata.num_row_groups):
                    tbl = pf.read_row_group(rg, columns=cols)
                    vals = tbl.column(key_field).to_pylist()
                    metas = tbl.column("metadata").to_pylist() if composite else None
                    for local_row, v in enumerate(vals):
                        key = (v, metas[local_row]) if composite else v
                        if key in needed and key not in key_to_loc:
                            key_to_loc[key] = (fi, base + local_row)
                    base += len(vals)
                    if len(key_to_loc) >= len(needed):
                        break
            if fi % 10 == 0 or len(key_to_loc) >= len(needed):
                log(domain, "scanned file", fi, "found so far", len(key_to_loc))

        def key_of(d):
            return (d["id"], d["metadata"]) if composite else d["id"]

        out_rows, missing = [], 0
        for idx, d in enumerate(docs):
            loc = key_to_loc.get(key_of(d))
            if loc is None:
                missing += 1
                out_rows.append({"idx": idx, "id": d["id"], "ntok": d["n_tokens"], "file": None, "row": None})
                continue
            fi, row = loc
            out_rows.append({"idx": idx, "id": d["id"], "ntok": d["n_tokens"],
                              "file": files[fi].split(f"@{REV[domain]}/", 1)[-1], "row": row})
        log(domain, "missing:", missing)
        with gzip.open(OUT / f"{domain}.docs.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")

        cache: dict = {}

        def get_text(lookup):
            fi, row = lookup
            p = files[fi]
            if p not in cache:
                cache[p] = pq.ParquetFile(fs.open(p, "rb"))
            pf = cache[p]
            rg, base = 0, 0
            while rg < pf.metadata.num_row_groups:
                n = pf.metadata.row_group(rg).num_rows
                if base + n > row:
                    break
                base += n
                rg += 1
            return pf.read_row_group(rg, columns=["text"]).column("text")[row - base].as_py()

        records = [{"idx": idx, "id": d["id"], "text": d["text"], "lookup": key_to_loc[key_of(d)]}
                   for idx, d in enumerate(docs) if key_of(d) in key_to_loc]
        result = _sample_check(records, get_text, k=300)
        result["missing_ids"] = missing
        (OUT / f"{domain}.verify.json").write_text(json.dumps(result, indent=1))
        log(domain, "verify:", result["checked"], result["matched"], "mismatches:", len(result["mismatches"]))
        return

    if domain == "pes2o":
        docs = _read_out_docs(domain)
        needed = {d["id"] for d in docs}
        rel_files = [f"data/v2/train-{i:05d}-of-00020.json.gz" for i in range(20)]
        id_to_loc: dict[str, tuple[int, int]] = {}
        local_paths: dict[int, str] = {}
        for fi, rel in enumerate(rel_files):
            if len(id_to_loc) >= len(needed):
                break
            local = hf_hub_download(repo_id="allenai/peS2o", filename=rel, repo_type="dataset",
                                     revision=REV[domain], local_dir=str(RAW / domain))
            local_paths[fi] = local
            with gzip.open(local, "rt", encoding="utf-8") as gz:
                for local_row, line in enumerate(gz):
                    if not line.strip():
                        continue
                    rid = json.loads(line).get("id")
                    if rid in needed and rid not in id_to_loc:
                        id_to_loc[rid] = (fi, local_row)
                    if len(id_to_loc) >= len(needed):
                        break
            log(domain, "scanned file", fi, "found so far", len(id_to_loc))

        out_rows, missing = [], 0
        for idx, d in enumerate(docs):
            loc = id_to_loc.get(d["id"])
            if loc is None:
                missing += 1
                out_rows.append({"idx": idx, "id": d["id"], "ntok": d["n_tokens"], "file": None, "row": None})
                continue
            fi, row = loc
            out_rows.append({"idx": idx, "id": d["id"], "ntok": d["n_tokens"], "file": rel_files[fi], "row": row})
        with gzip.open(OUT / f"{domain}.docs.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")

        # Bulk-fetch the sample in ONE sequential pass per referenced file (per-sample
        # re-streaming is O(k * file_len) and was too slow for pes2o's ~1.9M-line file).
        rng = random.Random(0)
        sample_idxs = [i for i in range(len(docs)) if docs[i]["id"] in id_to_loc]
        rng.shuffle(sample_idxs)
        sample_idxs = sample_idxs[:150]
        by_file_rows: dict[int, set[int]] = {}
        for i in sample_idxs:
            fi, row = id_to_loc[docs[i]["id"]]
            by_file_rows.setdefault(fi, set()).add(row)
        text_by_loc: dict[tuple[int, int], str] = {}
        for fi, rows in by_file_rows.items():
            local = local_paths.get(fi) or hf_hub_download(repo_id="allenai/peS2o", filename=rel_files[fi],
                                                             repo_type="dataset", revision=REV[domain], local_dir=str(RAW / domain))
            max_row = max(rows)
            with gzip.open(local, "rt", encoding="utf-8") as gz:
                for local_row, line in enumerate(gz):
                    if local_row in rows:
                        text_by_loc[(fi, local_row)] = json.loads(line).get("text")
                    if local_row >= max_row:
                        break

        def get_text(lookup):
            return text_by_loc.get(lookup)

        records = [{"idx": idx, "id": docs[idx]["id"], "text": docs[idx]["text"], "lookup": id_to_loc[docs[idx]["id"]]}
                   for idx in sample_idxs]
        result = _sample_check(records, get_text, k=150)
        result["missing_ids"] = missing
        (OUT / f"{domain}.verify.json").write_text(json.dumps(result, indent=1))
        log(domain, "verify:", result["checked"], result["matched"], "mismatches:", len(result["mismatches"]))
        return

    if domain == "arxiv":
        docs = _read_out_docs(domain)
        by_file: dict[str, list[int]] = {}
        for idx, d in enumerate(docs):
            by_file.setdefault(d["source"].rsplit("/", 1)[-1], []).append(idx)

        out_rows = [None] * len(docs)
        missing = 0
        local_by_file: dict[str, str] = {}
        for srcfile, idxs in by_file.items():
            rel = f"data/arxiv/train/{srcfile}"
            local = hf_hub_download(repo_id="allenai/olmo-mix-1124", filename=rel, repo_type="dataset",
                                     revision=REV[domain], local_dir=str(RAW / domain))
            local_by_file[srcfile] = local
            # ids are only *locally* sequential within whatever upstream proofpile shard a
            # file was repacked from (a single arxiv-train-*.json.gz can concatenate several
            # such shards), so the numeric id suffix is NOT the physical line index in
            # general -- search by id equality instead, same as pes2o/wiki.
            needed = {docs[i]["id"] for i in idxs}
            found: dict[str, int] = {}
            with gzip.open(local, "rt", encoding="utf-8") as gz:
                for local_row, line in enumerate(gz):
                    rid = json.loads(line).get("id")
                    if rid in needed and rid not in found:
                        found[rid] = local_row
                    if len(found) >= len(needed):
                        break
            for i in idxs:
                rid = docs[i]["id"]
                row = found.get(rid)
                if row is None:
                    missing += 1
                out_rows[i] = {"idx": i, "id": rid, "ntok": docs[i]["n_tokens"],
                               "file": f"data/arxiv/train/{srcfile}", "row": row}
        log(domain, "missing:", missing)
        with gzip.open(OUT / f"{domain}.docs.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")

        rng = random.Random(1)
        sample_idxs = [i for i in range(len(docs)) if out_rows[i]["row"] is not None]
        rng.shuffle(sample_idxs)
        sample_idxs = sample_idxs[:150]
        by_file_rows: dict[str, set[int]] = {}
        for i in sample_idxs:
            by_file_rows.setdefault(out_rows[i]["file"].rsplit("/", 1)[-1], set()).add(out_rows[i]["row"])
        text_by_loc: dict[tuple[str, int], str] = {}
        for srcfile, rows in by_file_rows.items():
            local = local_by_file[srcfile]
            max_row = max(rows)
            with gzip.open(local, "rt", encoding="utf-8") as gz:
                for local_row, line in enumerate(gz):
                    if local_row in rows:
                        text_by_loc[(srcfile, local_row)] = json.loads(line).get("text")
                    if local_row >= max_row:
                        break

        def get_text(lookup):
            return text_by_loc.get(lookup)

        records = []
        for idx in sample_idxs[:150]:
            d = docs[idx]
            records.append({"idx": idx, "id": d["id"], "text": d["text"],
                             "lookup": (out_rows[idx]["file"].rsplit("/", 1)[-1], out_rows[idx]["row"])})
        result = _sample_check(records, get_text, k=150)
        result["missing_ids"] = missing
        (OUT / f"{domain}.verify.json").write_text(json.dumps(result, indent=1))
        log(domain, "verify:", result["checked"], result["matched"], "mismatches:", len(result["mismatches"]))
        return

    raise SystemExit(f"unknown domain {domain}")


# ============================================================ step: dclm
def step_dclm():
    import numpy as np
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    OUT = W / "meta" / "manifest"
    OUT.mkdir(parents=True, exist_ok=True)
    RAWDIR = RAW / "dclm"
    RAWDIR.mkdir(parents=True, exist_ok=True)

    REPO = "allenai/DataDecide-data-recipes"
    REVD = "3baf34baf5b636f0943401b5c6a2ccb7e5cf3bb9"
    PREFIX = "preprocessed/dclm/v0_rep32_ft7percentile_fw2/gpt-neox-olmo-dolma-v1_5/"
    SEED, TARGET_TOKENS, CHUNK_TOKENS = 42, 2_067_750_000, 8192

    def load_token_memmap(path: Path) -> np.ndarray:
        size = path.stat().st_size
        if size % 2 == 0:
            try:
                mm = np.memmap(path, dtype=np.uint16, mode="r")
                sample = np.asarray(mm[: min(4096, mm.size)])
                if sample.size and int(sample.max()) < 100_000:
                    return mm
            except Exception:
                pass
        if size % 4 == 0:
            return np.memmap(path, dtype=np.uint32, mode="r")
        return np.load(path, mmap_mode="r")

    api = HfApi(token=None)
    all_files = [f for f in api.list_repo_files(REPO, repo_type="dataset", revision=REVD)
                 if f.startswith(PREFIX) and not f.endswith("/")]
    rng1 = random.Random(SEED)
    stage1 = list(all_files)
    rng1.shuffle(stage1)
    chosen52_sorted = sorted(stage1[:52])
    rng2 = np.random.default_rng(SEED)
    order = list(range(len(chosen52_sorted)))
    rng2.shuffle(order)
    ordered_files = [chosen52_sorted[i] for i in order]
    log("dclm processing order (first 3):", ordered_files[:3])

    src_tok = AutoTokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5",
                                            revision="8571ec72989ee67572049df907dec46cebe92335", use_fast=True)
    try:
        tgt_tok = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer",
                                                revision="5292e5d6c0f40b67cc765fe41bec991cf4345b5c", use_fast=True)
    except Exception:  # noqa: BLE001
        tgt_tok = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer",
                                                revision="5292e5d6c0f40b67cc765fe41bec991cf4345b5c", use_fast=False)

    manifest_rows, doc_idx, realized, source_seen, files_used = [], 0, 0, 0, []
    for rel in ordered_files:
        if realized >= TARGET_TOKENS:
            break
        local = hf_hub_download(repo_id=REPO, filename=rel, repo_type="dataset", revision=REVD, local_dir=str(RAWDIR))
        files_used.append(rel)
        flat = np.asarray(load_token_memmap(Path(local))).reshape(-1)
        log(rel, "n_source_tokens=", flat.size)
        if flat.size == 0:
            continue
        start = int(rng2.integers(0, max(1, flat.size)))
        remaining, pos = int(flat.size), start
        while remaining > 0 and realized < TARGET_TOKENS:
            take = min(CHUNK_TOKENS, remaining, max(CHUNK_TOKENS, TARGET_TOKENS - realized + CHUNK_TOKENS))
            end = pos + take
            wrapped = end > flat.size
            if not wrapped:
                chunk = np.asarray(flat[pos:end], dtype=np.int64)
            else:
                chunk = np.concatenate([np.asarray(flat[pos:], dtype=np.int64),
                                         np.asarray(flat[: end - flat.size], dtype=np.int64)])
            chunk_start, chunk_len = pos, int(chunk.size)
            pos = end % flat.size
            remaining -= int(chunk.size)
            source_seen += int(chunk.size)
            text = src_tok.decode(chunk.tolist(), skip_special_tokens=True)
            if not text.strip():
                continue
            tgt_ids = tgt_tok.encode(text, add_special_tokens=False)
            n = len(tgt_ids)
            if n == 0:
                continue
            trimmed_to = None
            if realized + n > TARGET_TOKENS:
                keep = TARGET_TOKENS - realized
                if keep <= 0:
                    break
                n, trimmed_to = keep, keep
            manifest_rows.append({"idx": doc_idx, "id": f"datadecide-qc7p-fw2-{doc_idx:08d}", "file": rel,
                                   "start": chunk_start, "length": chunk_len, "wrapped": wrapped,
                                   "ntok": n, "trimmed_to": trimmed_to})
            doc_idx += 1
            realized += n
            if doc_idx % 20000 == 0:
                log("progress doc_idx=", doc_idx, "realized=", realized)

    log("DONE replay: doc_idx=", doc_idx, "realized=", realized, "files_used=", files_used)
    with gzip.open(OUT / "dclm.docs.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in manifest_rows:
            f.write(json.dumps(r) + "\n")
    stats = {"doc_idx": doc_idx, "realized_tokens": realized, "source_tokens_seen": source_seen,
              "files_used": files_used, "expected_doc_count": 252440, "expected_realized": 2067750000,
              "match_doc_count": doc_idx == 252440, "match_realized": realized == 2067750000}
    (OUT / "dclm.replay_stats.json").write_text(json.dumps(stats, indent=1))
    log("stats:", stats)

    real_docs = []
    for shard in sorted((BD / "out" / "dclm").glob("documents-*.json.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            real_docs.extend(json.loads(line) for line in f if line.strip())

    rngv = random.Random(7)
    sample_idxs = list(range(min(len(real_docs), doc_idx)))
    rngv.shuffle(sample_idxs)
    sample_idxs = sample_idxs[:300]
    file_arrs: dict[str, np.ndarray] = {}

    def get_arr(rel):
        if rel not in file_arrs:
            file_arrs[rel] = np.asarray(load_token_memmap(RAWDIR / rel)).reshape(-1)
        return file_arrs[rel]

    for cleanup in (True, False):
        matched = checked = 0
        mismatches = []
        for i in sample_idxs:
            row, real = manifest_rows[i], real_docs[i]
            flat = get_arr(row["file"])
            s, l = row["start"], row["length"]
            chunk = (np.asarray(flat[s: s + l], dtype=np.int64) if not row["wrapped"] else
                     np.concatenate([np.asarray(flat[s:], dtype=np.int64),
                                     np.asarray(flat[: l - (flat.size - s)], dtype=np.int64)]))
            try:
                text = src_tok.decode(chunk.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=cleanup)
            except TypeError:
                text = src_tok.decode(chunk.tolist(), skip_special_tokens=True)
            if row["trimmed_to"] is not None:
                text = tgt_tok.decode(tgt_tok.encode(text, add_special_tokens=False)[: row["trimmed_to"]],
                                       skip_special_tokens=True)
            checked += 1
            if text == real["text"]:
                matched += 1
            else:
                mismatches.append({"idx": i, "id": row["id"], "expected_head": real["text"][:60], "actual_head": text[:60]})
        log(f"cleanup={cleanup} checked={checked} matched={matched} mismatches={len(mismatches)}")
        (OUT / f"dclm.verify_cleanup_{cleanup}.json").write_text(
            json.dumps({"checked": checked, "matched": matched, "mismatches": mismatches[:15]}, indent=1))


# ============================================================ step: starcoder
def _diff_spans(raw: str, out: str):
    """[start,end) spans deleted from `raw` to produce `out`, or None if impossible."""
    if raw == out:
        return []
    spans, i, j, RESYNC = [], 0, 0, 24
    while i < len(raw) and j < len(out):
        if raw[i] == out[j]:
            i += 1
            j += 1
            continue
        window = out[j:j + RESYNC]
        if not window:
            break
        i2 = raw.find(window, i)
        if i2 == -1:
            return None
        spans.append((i, i2))
        i = i2
    if j < len(out):
        return None
    if i < len(raw):
        spans.append((i, len(raw)))
    rebuilt, prev = [], 0
    for s, e in spans:
        rebuilt.append(raw[prev:s])
        prev = e
    rebuilt.append(raw[prev:])
    return spans if "".join(rebuilt) == out else None


def step_starcoder():
    """Requires a valid HF token at /scratch/users/nzhao2/refhq-regmix-5p5b-v1/.hf_token
    (bigcode/starcoderdata is gated). Never prints/copies/logs the token."""
    import random as _random

    from huggingface_hub import HfFileSystem, hf_hub_download
    import pyarrow.parquet as pq

    OUT = W / "meta" / "manifest"
    OUT.mkdir(parents=True, exist_ok=True)
    RAWDIR = W / "raw" / "starcoder"
    RAWDIR.mkdir(parents=True, exist_ok=True)
    REV = "9fc30b578cedaec69e47302df72cf00feed7c8c4"
    tok_path = BD / ".hf_token"
    TOKEN = tok_path.read_text().strip() if tok_path.exists() else None
    if not TOKEN:
        raise SystemExit("no HF token found; starcoder is gated")
    fs = HfFileSystem(token=TOKEN)

    docs = []
    for shard in sorted((BD / "out" / "starcoder").glob("documents-*.json.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            docs.extend(json.loads(line) for line in f if line.strip())
    log("starcoder out docs:", len(docs))
    needed_ids = {d["id"] for d in docs}

    # Per-file row counts (footer-only reads over ALL files), cached across reruns.
    rc_path = W / "meta" / "starcoder_file_rowcounts.json"
    if rc_path.exists():
        rowcounts = json.loads(rc_path.read_text())
    else:
        from huggingface_hub import HfApi

        api = HfApi(token=TOKEN)
        files = sorted(f for f in api.list_repo_files("bigcode/starcoderdata", repo_type="dataset", revision=REV)
                       if f.endswith(".parquet"))
        rowcounts = []
        for f in files:
            with fs.open(f"datasets/bigcode/starcoderdata@{REV}/{f}", "rb") as fobj:
                rowcounts.append({"path": f, "num_rows": pq.ParquetFile(fobj).metadata.num_rows})
        rc_path.parent.mkdir(parents=True, exist_ok=True)
        rc_path.write_text(json.dumps(rowcounts, indent=1))

    # IMPORTANT: sort by the FULL relative file path as one string -- this is what
    # `datasets`' actual streaming order does. "c-sharp/train-..." sorts BEFORE
    # "c/train-..." as full paths ('-' 0x2D < '/' 0x2F at the first differing position)
    # even though the bare directory names sort the other way ("c" < "c-sharp", since "c"
    # is a prefix of "c-sharp"). Grouping by directory name and sorting directories
    # separately (an earlier version of this script did that) reproduces the wrong order
    # for exactly this pair of language names, and made 396,846/762,037 (52%) of documents
    # come back "unresolved" (the true prefix never reaches "c-sharp" at all) even though
    # only 42 are genuinely unresolvable. Confirmed empirically: a live
    # `load_dataset("bigcode/starcoderdata", streaming=True)` trace hits the first .cs
    # file at exactly row 570394, immediately after "bluespec/...".
    files_sorted = sorted(rowcounts, key=lambda r: r["path"])
    prefix_files, cum = [], 0
    for r in files_sorted:
        if cum >= 980_000:
            break
        prefix_files.append(r["path"])
        cum += r["num_rows"]
    log("head-of-stream prefix (full-path sort):", len(prefix_files), "cumulative rows:", cum)

    id_candidates: dict[str, list[tuple[str, int]]] = {}
    for fpath in prefix_files:
        with fs.open(f"datasets/bigcode/starcoderdata@{REV}/{fpath}", "rb") as fobj:
            pf = pq.ParquetFile(fobj)
            base = 0
            for rg in range(pf.metadata.num_row_groups):
                ids = pf.read_row_group(rg, columns=["id"]).column("id").to_pylist()
                for local_row, rid in enumerate(ids):
                    if rid in needed_ids:
                        id_candidates.setdefault(rid, []).append((fpath, base + local_row))
                base += len(ids)
        log("scanned", fpath, "candidates so far", sum(len(v) for v in id_candidates.values()))

    files_with_candidates = sorted({fp for v in id_candidates.values() for fp, _ in v})
    local_paths = {fp: hf_hub_download(repo_id="bigcode/starcoderdata", filename=fp, repo_type="dataset",
                                        revision=REV, token=TOKEN, local_dir=str(RAWDIR))
                   for fp in files_with_candidates}

    # Read each prefix file's full `content` column into memory ONCE (not per row-group,
    # never per document -- an earlier version called pq.read_row_group per document and
    # was thousands of times too slow).
    content_cache: dict[str, list] = {}

    def get_content(fp: str, row: int) -> str:
        col = content_cache.get(fp)
        if col is None:
            col = pq.ParquetFile(local_paths[fp]).read(columns=["content"]).column("content").to_pylist()
            content_cache[fp] = col
        return col[row]

    out_rows, unresolved, ambiguous = [], 0, 0
    for idx, d in enumerate(docs):
        cands = id_candidates.get(d["id"], [])
        matches = []
        for fp, row in cands:
            spans = _diff_spans(get_content(fp, row), d["text"])
            if spans is not None:
                matches.append((fp, row, spans))
        if not matches:
            unresolved += 1
            out_rows.append({"idx": idx, "id": d["id"], "ntok": d["n_tokens"], "file": None, "row": None, "spans": None})
            continue
        if len(matches) > 1:
            ambiguous += 1
            matches.sort(key=lambda m: (len(m[2]), sum(e - s for s, e in m[2])))
        fp, row, spans = matches[0]
        out_rows.append({"idx": idx, "id": d["id"], "ntok": d["n_tokens"], "file": fp, "row": row,
                          "spans": json.dumps(spans) if spans else ""})
        if idx % 50_000 == 0:
            log("resolved", idx, "/", len(docs))
    log("unresolved:", unresolved, "ambiguous:", ambiguous)

    with gzip.open(OUT / "starcoder.docs.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    rng = _random.Random(0)
    sample_idxs = [i for i in range(len(docs)) if out_rows[i]["file"]]
    rng.shuffle(sample_idxs)
    sample_idxs = sample_idxs[:300]
    checked = matched = 0
    mismatches = []
    for i in sample_idxs:
        row = out_rows[i]
        raw = get_content(row["file"], row["row"])
        spans = json.loads(row["spans"]) if row["spans"] else []
        rebuilt, prev = [], 0
        for s, e in spans:
            rebuilt.append(raw[prev:s])
            prev = e
        rebuilt.append(raw[prev:])
        checked += 1
        if "".join(rebuilt) == docs[i]["text"]:
            matched += 1
        else:
            mismatches.append({"idx": i, "id": row["id"]})
    result = {"checked": checked, "matched": matched, "mismatches": mismatches[:20],
              "unresolved": unresolved, "ambiguous": ambiguous, "n_prefix_files": len(prefix_files)}
    (OUT / "starcoder.verify.json").write_text(json.dumps(result, indent=1))
    log("verify:", result)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    step = sys.argv[1]
    if step == "sources":
        step_sources()
    elif step == "resolve":
        step_resolve(sys.argv[2])
    elif step == "dclm":
        step_dclm()
    elif step == "starcoder":
        step_starcoder()
    else:
        raise SystemExit(f"unknown step {step!r}")
