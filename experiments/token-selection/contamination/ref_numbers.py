#!/usr/bin/env python3
"""Recompute the published curriculum-paper contamination numbers from the
previous agent's artifacts, so the new scans have an exact reference."""
import json
import pickle
from collections import defaultdict

C = '/scratch/users/nzhao2/agent-runs/contam-20260910-060952'
E = pickle.load(open(f'{C}/eval_ngrams_strict.pkl', 'rb'))
ITEMS = E['items']
S = json.load(open(f'{C}/contam_summary_strict.json'))
per_item = {int(k): v for k, v in S['per_item'].items()}

ntask = defaultdict(int)
for lbl, _i, _t in ITEMS:
    ntask[lbl] += 1
hit = defaultdict(int)
for gid in per_item:
    hit[ITEMS[gid][0]] += 1

tot = len(ITEMS)
nhit = len(per_item)
print(f"docs scanned   : {S['docs']:,}")
print(f"13-grams       : {S['ngrams']:,}")
print(f'eval items     : {tot:,}')
print(f'items matched  : {nhit:,}  -> {100.0 * nhit / tot:.4f}%')
print()
print(f'{"task":22s}{"matched":>9s}{"n":>8s}{"rate":>10s}')
selfcontained = ('arc_challenge', 'arc_easy', 'csqa', 'openbookqa', 'piqa',
                 'winogrande', 'socialiqa', 'boolq')
sc_h = sc_n = 0
for t in sorted(ntask):
    h, n = hit.get(t, 0), ntask[t]
    print(f'{t:22s}{h:>9d}{n:>8d}{100.0 * h / n:>9.4f}%')
    if t.rsplit('_', 1)[0] in selfcontained:
        sc_h += h
        sc_n += n
print()
print(f'self-contained-stem subset: {sc_h}/{sc_n} = {100.0 * sc_h / sc_n:.4f}%')
print()
print('loaded splits:', E.get('loaded'))
print('failed:', E.get('failed'))
