#!/usr/bin/env python3
"""Manifest for the 127B reservoir scan, token-range sharded.

Columns: corpus, domain, path, start_tok, end_tok
Base tree already matches the paper's reservoir exactly for dclm / arxiv /
open-web-math / algebraic-stack / wiki; pes2o and starcoder are topped up from
the topup tree until the paper's per-domain totals are reached.
"""
from __future__ import annotations

import glob
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
RUN = sys.argv[1]
AR = '/scratch/users/nzhao2/agent-runs'
BASE = f'{AR}/edullm-dataset-olmohq-dolma2-tok-v2-20260726-075247/tokenized/shards'
TOPUP = f'{AR}/olmohq-topup-20260728-185841/tokenized/shards'
CHUNK = 120_000_000        # tokens per array task
TARGET = {'dclm': 29_691_000_000, 'arxiv': 22_148_000_000,
          'pes2o': 26_379_000_000, 'starcoder': 18_541_000_000,
          'open-web-math': 13_238_000_000, 'algebraic-stack': 12_902_000_000,
          'wiki': 3_752_000_000}


def files_by_dom(root):
    out = defaultdict(list)
    for fp in sorted(glob.glob(f'{root}/*.npy')):
        if not os.path.isfile(fp):
            continue
        parts = os.path.basename(fp).split('__')
        if len(parts) < 3:
            continue
        out[parts[1]].append((fp, os.path.getsize(fp) // 4))
    return out


base, topup = files_by_dom(BASE), files_by_dom(TOPUP)
rows = []
used = defaultdict(int)
for dom, target in TARGET.items():
    for src in (base.get(dom, []), topup.get(dom, [])):
        for fp, ntok in src:
            if used[dom] >= target:
                break
            take = min(ntok, target - used[dom])
            s = 0
            while s < take:
                e = min(s + CHUNK, take)
                rows.append(('reservoir_127b', dom, fp, s, e))
                s = e
            used[dom] += take

with open(f'{RUN}/res_shards.tsv', 'w', encoding='utf-8') as fh:
    for r in rows:
        fh.write('\t'.join(map(str, r)) + '\n')

print(f'tasks: {len(rows)}\n')
print(f'{"domain":18s}{"tokens used":>16s}{"target":>16s}{"tasks":>8s}')
print('-' * 58)
for dom in TARGET:
    nt = sum(1 for r in rows if r[1] == dom)
    print(f'{dom:18s}{used[dom] / 1e9:>14.3f}B{TARGET[dom] / 1e9:>15.3f}B{nt:>8d}')
print(f'\ntotal {sum(used.values()) / 1e9:.2f}B tokens in {len(rows)} tasks')
print(f'wrote {RUN}/res_shards.tsv')
