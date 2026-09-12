#!/usr/bin/env python3
"""Build the strict-protocol scan manifest: one row per (file, line-shard).

Reuses the previous agent's contam_scan.py unchanged, so every number is
directly comparable to the curriculum paper's published 0.89%.
Columns: corpus, source, path, NSHARDS, SHARD_ID
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
RUN = Path(sys.argv[1])
AR = Path('/scratch/users/nzhao2/agent-runs')
TARGET = 700 * 1024 * 1024   # gz bytes per task
CAP = 20

TRIM = {
    'dml_train_10b':       AR / 'regmix-10b-20260725-124810' / 'trim',
    'olmomix_stock_30b':   AR / 'olmo-mix-30b-20260722' / 'trim',
    'lgbm_opt':            AR / 'lgbm-opt-15b-20260807T223744Z' / 'trim',
    'olmoe_synthetic_10b': AR / 'opt-with-synthetic-10b-20260808T074749Z' / 'trim',
}
MULTI = {
    'refhq_hq': (Path('/scratch/users/nzhao2/refhq-regmix-5p5b-v1/out'),
                 'documents-*.json.gz', False),
    'refhq_instruct': (Path('/scratch/users/nzhao2/refhq-new-v1/out'),
                       'documents-*.jsonl.gz', True),
}

rows: list[tuple[str, str, str, int, int]] = []
summary: dict[str, list] = {}

for corpus, trim in TRIM.items():
    if not trim.is_dir():
        print(f'  !! missing {trim}')
        continue
    for dom_dir in sorted(trim.iterdir()):
        if not dom_dir.is_dir():
            continue
        for f in sorted(dom_dir.glob('*-trimmed.json.gz')):
            sz = f.stat().st_size
            nsh = max(1, min(CAP, round(sz / TARGET)))
            for sid in range(nsh):
                rows.append((corpus, dom_dir.name, str(f), nsh, sid))
            summary.setdefault(corpus, []).append((dom_dir.name, sz, nsh))

for corpus, (root, pat, needs_documents) in MULTI.items():
    if not root.is_dir():
        print(f'  !! missing {root}')
        continue
    n = 0
    tot = 0
    for f in sorted(root.rglob(pat)):
        if needs_documents and 'documents' not in f.parts:
            continue
        if 'attributes' in f.parts:
            continue
        src = f.relative_to(root).parts[0]
        rows.append((corpus, src, str(f), 1, 0))
        n += 1
        tot += f.stat().st_size
    summary[corpus] = [(f'{n} shards', tot, 1)]

out = RUN / 'strict_shards.tsv'
with open(out, 'w', encoding='utf-8') as fh:
    for r in rows:
        fh.write('\t'.join(map(str, r)) + '\n')

print(f'tasks: {len(rows)}\n')
for corpus, items in summary.items():
    tb = sum(s for _, s, _ in items)
    nt = sum(1 for r in rows if r[0] == corpus)
    print(f'{corpus:22s} {tb / 2**30:7.2f} GB gz   {nt:4d} tasks')
    for dom, sz, nsh in items:
        print(f'    {dom:20s} {sz / 2**30:7.2f} GB  x{nsh}')
print(f'\nwrote {out}')
