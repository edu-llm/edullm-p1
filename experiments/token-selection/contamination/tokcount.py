#!/usr/bin/env python3
"""Sum token counts per domain from .npy headers (no data read)."""
from __future__ import annotations

import glob
import os
import sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
AR = '/scratch/users/nzhao2/agent-runs'
RUNS = [
    'edullm-dataset-olmohq-dolma2-tok-v2-20260726-075247',
    'olmohq-topup-20260728-185841',
    'edullm-dataset-olmo-dolma2-tok-v2-20260726-075138',
]
grand = defaultdict(int)
for r in RUNS:
    files = glob.glob(f'{AR}/{r}/tokenized/shards/*.npy')
    per = defaultdict(int)
    dts = set()
    for fp in files:
        try:
            with open(fp, 'rb') as fh:
                ver = np.lib.format.read_magic(fh)
                shape, _fo, dt = np.lib.format._read_array_header(fh, ver)
        except Exception:
            continue
        name = os.path.basename(fp)
        parts = name.split('__')
        dom = parts[1] if len(parts) > 2 else 'unknown'
        n = int(np.prod(shape))
        per[dom] += n
        dts.add(str(dt))
    tot = sum(per.values())
    print(f'\n===== {r}')
    print(f'  shards={len(files)}  dtype={sorted(dts)}  total={tot:,} tokens '
          f'({tot / 1e9:.2f}B)')
    for d in sorted(per, key=lambda k: -per[k]):
        print(f'    {d:20s} {per[d] / 1e9:8.3f}B')
        grand[d] += per[d]

print('\n' + '=' * 60)
print(f'GRAND TOTAL across the three runs: {sum(grand.values()) / 1e9:.2f}B tokens')
for d in sorted(grand, key=lambda k: -grand[k]):
    print(f'  {d:20s} {grand[d] / 1e9:8.3f}B')
