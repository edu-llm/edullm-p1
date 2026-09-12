#!/usr/bin/env python3
"""Scan one corpus shard for verbatim 13-gram overlap with the OLMES eval index.

usage: scan_shard.py EVAL_INDEX.pkl SHARD_PATH OUT.json CORPUS SOURCE
"""
from __future__ import annotations

import gzip
import json
import pickle
import sys
import time
from collections import defaultdict

N = 13
_M = 0xFFFFFFFFFFFFFFFF
_FNV0 = 0xcbf29ce484222325
_FNVP = 0x100000001b3

idx_path, shard, out_path, corpus, source = sys.argv[1:6]
with open(idx_path, 'rb') as f:
    IDX = pickle.load(f)
GRAMS: dict = IDX['grams']
ITEMS = IDX['items']

_memo: dict[str, int] = {}


def tok_hash(t: str) -> int:
    h = _memo.get(t)
    if h is None:
        h = _FNV0
        for b in t.encode('utf-8'):
            h = ((h ^ b) * _FNVP) & _M
        _memo[t] = h
    return h


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
    """Raw text for an OLMo-mix document or a flattened chat record."""
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


matched_items: set[int] = set()
hits_per_task: dict[str, int] = defaultdict(int)   # 13-gram hit events
docs_hit_per_task: dict[str, int] = defaultdict(int)
ndocs = 0
ndocs_hit = 0
ntokens = 0
t0 = time.time()

opener = gzip.open if shard.endswith('.gz') else open
with opener(shard, 'rt', encoding='utf-8', errors='replace') as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        text = doc_text(rec)
        if not text:
            continue
        ndocs += 1
        toks = norm_tokens(text)
        ntokens += len(toks)
        if len(toks) < N:
            continue
        hs = [tok_hash(t) for t in toks]
        doc_tasks: set[str] = set()
        for i in range(len(hs) - N + 1):
            g = _FNV0
            for j in range(i, i + N):
                g = ((g ^ hs[j]) * _FNVP) & _M
            ids = GRAMS.get(g)
            if ids is not None:
                for iid in ids:
                    matched_items.add(iid)
                    task = ITEMS[iid][0]
                    hits_per_task[task] += 1
                    doc_tasks.add(task)
        if doc_tasks:
            ndocs_hit += 1
            for task in doc_tasks:
                docs_hit_per_task[task] += 1

res = {
    'corpus': corpus,
    'source': source,
    'shard': shard,
    'n_docs': ndocs,
    'n_docs_with_hit': ndocs_hit,
    'n_tokens': ntokens,
    'elapsed_s': round(time.time() - t0, 1),
    'matched_item_ids': sorted(matched_items),
    'gram_hits_per_task': dict(hits_per_task),
    'docs_hit_per_task': dict(docs_hit_per_task),
}
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(res, f)
print(f'{source} {shard.rsplit("/", 1)[-1]}: docs={ndocs} tokens={ntokens} '
      f'docs_hit={ndocs_hit} matched_items={len(matched_items)} '
      f'{res["elapsed_s"]}s', flush=True)
