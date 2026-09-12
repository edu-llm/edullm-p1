#!/usr/bin/env python3
"""Inventory every tokenized shard tree: tokens = filesize/4 (raw uint32)."""
from __future__ import annotations

import glob
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
ROOT = '/scratch/users/nzhao2'
PAPER = {'dclm': 29.691, 'arxiv': 22.148, 'pes2o': 26.379, 'starcoder': 18.541,
         'open-web-math': 13.238, 'algebraic-stack': 12.902, 'wiki': 3.752}

trees = sorted(set(
    os.path.dirname(p) for p in
    glob.glob(f'{ROOT}/**/tokenized/shards/*', recursive=True)))
grand = defaultdict(int)
rows = []
for t in trees:
    per = defaultdict(int)
    n = 0
    for fp in glob.glob(f'{t}/*'):
        if not os.path.isfile(fp):
            continue
        sz = os.path.getsize(fp)
        base = os.path.basename(fp)
        parts = base.split('__')
        dom = parts[1] if len(parts) > 2 else 'unknown'
        per[dom] += sz // 4
        n += 1
    tot = sum(per.values())
    run = t.replace(f'{ROOT}/', '').replace('/tokenized/shards', '')
    rows.append((tot, run, n, dict(per)))
    for d, v in per.items():
        grand[d] += v

rows.sort(reverse=True)
for tot, run, n, per in rows:
    print(f'\n{run}\n  {n} shards, {tot / 1e9:.2f}B tokens')
    for d in sorted(per, key=lambda k: -per[k]):
        print(f'    {d:20s} {per[d] / 1e9:8.3f}B')

print('\n' + '=' * 66)
print(f'{"domain":20s}{"on disk (all trees)":>22s}{"paper reservoir":>18s}')
print('-' * 66)
for d in sorted(set(grand) | set(PAPER), key=lambda k: -grand.get(k, 0)):
    p = PAPER.get(d)
    print(f'{d:20s}{grand.get(d, 0) / 1e9:>20.3f}B'
          f'{(f"{p:.3f}B" if p else "-"):>18s}')
print(f'\ntotal on disk: {sum(grand.values()) / 1e9:.2f}B   '
      f'paper reservoir: {sum(PAPER.values()):.2f}B')
