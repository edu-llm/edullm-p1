#!/usr/bin/env python3
"""Scan a raw-uint32 Dolma2 token shard for eval 13-gram overlap.

Decodes tokens back to text, then applies the EXACT normalisation and hashing
of contam_scan.py so results are comparable to the published 0.89%.
Run with PYTHONHASHSEED=0.

usage: scan_tok.py EVAL_PKL SHARD OUT_JSON CORPUS SOURCE START END
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
import unicodedata

import numpy as np

EOS = 100257
WINDOW = 4_000_000          # tokens per read window
N = 13
_ws = re.compile(r'\s+')
_punct = re.compile(r'[^\w\s]')

EVAL_PKL, SHARD, OUT, CORPUS, SOURCE = sys.argv[1:6]
START = int(sys.argv[6])
END = int(sys.argv[7])

with open(EVAL_PKL, 'rb') as f:
    E = pickle.load(f)
INDEX, ITEMS = E['index'], E['items']

os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoTokenizer  # noqa: E402

TOK = AutoTokenizer.from_pretrained(os.environ['NG13_TOKENIZER'])

n_tok_total = os.path.getsize(SHARD) // 4
limit = min(END, n_tok_total)
arr = np.memmap(SHARD, dtype=np.uint32, mode='r', shape=(n_tok_total,))
probe = np.asarray(arr[:10000])
if probe.size and int(probe.max()) > 100352:
    raise SystemExit(f'{SHARD}: max token {int(probe.max())} exceeds vocab; '
                     'file is probably not raw uint32')

hits: dict[int, dict] = {}
docs = ngrams = ntok = 0
t0 = time.time()
carry: list[int] = []
pos = min(START, n_tok_total)

while pos < limit:
    end = min(pos + WINDOW, limit)
    win = np.asarray(arr[pos:end])
    pos = end
    eos_at = np.flatnonzero(win == EOS)
    pieces: list[list[int]] = []
    prev = 0
    for e in eos_at:
        seg = win[prev:e].tolist()
        if carry:
            seg = carry + seg
            carry = []
        pieces.append(seg)
        prev = e + 1
    tail = win[prev:].tolist()
    if carry:
        carry = carry + tail
    else:
        carry = tail
    if pos >= limit and carry:          # flush final partial document
        pieces.append(carry)
        carry = []

    for seg in pieces:
        if len(seg) < 8:
            continue
        docs += 1
        ntok += len(seg)
        try:
            txt = TOK.decode(seg, skip_special_tokens=True)
        except Exception:
            continue
        t = unicodedata.normalize('NFKC', txt).lower()
        t = _punct.sub(' ', t)
        w = _ws.sub(' ', t).strip().split()
        L = len(w)
        if L < N:
            continue
        ngrams += L - N + 1
        for i in range(L - N + 1):
            h = hash(' '.join(w[i:i + N]))
            got = INDEX.get(h)
            if got is not None:
                for gid in got:
                    e2 = hits.get(gid)
                    if e2 is None:
                        hits[gid] = {'n': 1, 'ex': None, 'domain': SOURCE}
                    else:
                        e2['n'] += 1
    if docs and docs % 400_000 < 2000:
        print(f'  .. docs={docs:,} tok={ntok:,} hits={len(hits)} '
              f'{time.time() - t0:.0f}s', flush=True)

json.dump({'shard': SHARD, 'start': START, 'end': limit, 'nsh': 1, 'sid': 0, 'docs': docs,
           'ngrams': ngrams, 'n_tokens': ntok,
           'elapsed_s': round(time.time() - t0, 1),
           'hits': {str(k): v for k, v in hits.items()}},
          open(OUT, 'w', encoding='utf-8'))
print(f'{CORPUS}/{SOURCE} {os.path.basename(SHARD)}: docs={docs:,} '
      f'tokens={ntok:,} ngrams={ngrams:,} items={len(hits)} '
      f'{time.time() - t0:.0f}s', flush=True)
