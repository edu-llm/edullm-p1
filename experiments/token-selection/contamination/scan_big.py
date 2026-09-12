#!/usr/bin/env python3
"""Scan one large single-file corpus domain for 13-gram overlap with the OLMES
eval index, parallelised inside the task: pigz decompresses, a worker pool
hashes.

usage: scan_big.py EVAL_INDEX.pkl PATH OUT.json CORPUS SOURCE NWORKERS
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict

N = 13
_M = 0xFFFFFFFFFFFFFFFF
_FNV0 = 0xcbf29ce484222325
_FNVP = 0x100000001b3

IDX_PATH = None
GRAMS: dict = {}
ITEMS: list = []


def _init(idx_path):
    global GRAMS, ITEMS
    with open(idx_path, 'rb') as f:
        d = pickle.load(f)
    GRAMS = d['grams']
    ITEMS = d['items']


def norm_tokens(text: str) -> list[str]:
    out, cur = [], []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            out.append(''.join(cur))
            cur = []
    if cur:
        out.append(''.join(cur))
    return out


def doc_text(rec: dict) -> str:
    t = rec.get('text')
    if isinstance(t, str) and t:
        return t
    for k in ('content', 'raw', 'body'):
        v = rec.get(k)
        if isinstance(v, str) and v:
            return v
    msgs = rec.get('messages') or rec.get('conversations')
    if isinstance(msgs, list):
        parts = []
        for m in msgs:
            if isinstance(m, dict):
                c = m.get('content') or m.get('value')
                if isinstance(c, str):
                    parts.append(c)
            elif isinstance(m, str):
                parts.append(m)
        return '\n'.join(parts)
    return ''


def work(batch: list[str]):
    memo: dict[str, int] = {}
    grams = GRAMS
    items = ITEMS
    matched: set[int] = set()
    hits: dict[str, int] = defaultdict(int)
    dhit: dict[str, int] = defaultdict(int)
    ndocs = nhit = ntok = 0
    for line in batch:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        text = doc_text(rec)
        if not text:
            continue
        ndocs += 1
        toks = norm_tokens(text)
        ntok += len(toks)
        if len(toks) < N:
            continue
        hs = []
        for t in toks:
            h = memo.get(t)
            if h is None:
                h = _FNV0
                for b in t.encode('utf-8'):
                    h = ((h ^ b) * _FNVP) & _M
                memo[t] = h
            hs.append(h)
        doc_tasks: set[str] = set()
        for i in range(len(hs) - N + 1):
            g = _FNV0
            for j in range(i, i + N):
                g = ((g ^ hs[j]) * _FNVP) & _M
            ids = grams.get(g)
            if ids is not None:
                for iid in ids:
                    matched.add(iid)
                    task = items[iid][0]
                    hits[task] += 1
                    doc_tasks.add(task)
        if doc_tasks:
            nhit += 1
            for task in doc_tasks:
                dhit[task] += 1
    return ndocs, nhit, ntok, matched, dict(hits), dict(dhit)


def batches(path, size=2000):
    dec = ['pigz', '-dc', '-p', '3'] if path.endswith('.gz') else ['cat']
    p = subprocess.Popen(dec + [path], stdout=subprocess.PIPE, bufsize=1 << 22)
    buf = []
    for raw in p.stdout:
        if not raw.strip():
            continue
        buf.append(raw.decode('utf-8', 'replace'))
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
    p.stdout.close()
    p.wait()


def main():
    idx_path, path, out_path, corpus, source = sys.argv[1:6]
    nw = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    t0 = time.time()
    matched: set[int] = set()
    hits: dict[str, int] = defaultdict(int)
    dhit: dict[str, int] = defaultdict(int)
    ndocs = nhit = ntok = 0
    with mp.Pool(nw, initializer=_init, initargs=(idx_path,)) as pool:
        for d, h, tk, mset, hh, dh in pool.imap_unordered(
                work, batches(path), chunksize=1):
            ndocs += d
            nhit += h
            ntok += tk
            matched |= mset
            for k, v in hh.items():
                hits[k] += v
            for k, v in dh.items():
                dhit[k] += v
            if ndocs and ndocs % 2_000_000 < 2000:
                print(f'  .. {ndocs:,} docs {ntok:,} tok '
                      f'{time.time() - t0:.0f}s', flush=True)
    res = {
        'corpus': corpus, 'source': source, 'shard': path,
        'n_docs': ndocs, 'n_docs_with_hit': nhit, 'n_tokens': ntok,
        'elapsed_s': round(time.time() - t0, 1),
        'matched_item_ids': sorted(matched),
        'gram_hits_per_task': dict(hits), 'docs_hit_per_task': dict(dhit),
        'file_bytes': os.path.getsize(path),
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(res, f)
    print(f'{corpus}/{source}: docs={ndocs:,} tokens={ntok:,} '
          f'docs_hit={nhit:,} matched_items={len(matched):,} '
          f'{res["elapsed_s"]}s', flush=True)


if __name__ == '__main__':
    main()
