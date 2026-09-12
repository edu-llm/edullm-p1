#!/usr/bin/env python3
"""Correct per-arm contamination: for each arm, draw weight_d * BUDGET tokens
from each reservoir domain and take the UNION of matched eval items.

The earlier 'expected exposure' number was a weighted MEAN of per-domain rates,
which is bounded by the largest single-domain rate and therefore cannot be an
estimate of an arm's actual rate. This is the union, at the arm's real token
allocation per domain.

usage: arm_union.py RES_HITS_DIR [BUDGET_TOKENS] [OUT.json]
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
HITS = sys.argv[1]
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 10e9
OUT = sys.argv[3] if len(sys.argv) > 3 else None

C = '/scratch/users/nzhao2/agent-runs/contam-20260910-060952'
E = pickle.load(open(f'{C}/eval_ngrams_strict.pkl', 'rb'))
ITEMS = E['items']
NTASK: dict[str, int] = defaultdict(int)
for lbl, _i, _t in ITEMS:
    NTASK[lbl] += 1
TOT = len(ITEMS)
SELFC = ('arc_challenge', 'arc_easy', 'csqa', 'openbookqa', 'piqa',
         'winogrande', 'socialiqa', 'boolq')

ARMS = {
    'Olmo-mix-1124 (natural)': {
        'dclm': .951, 'arxiv': .005, 'starcoder': .021, 'pes2o': .015,
        'open-web-math': .003, 'algebraic-stack': .003, 'wiki': .001},
    'Data Mixing Laws paper': {
        'dclm': .375, 'arxiv': .250, 'starcoder': .141, 'pes2o': .094,
        'open-web-math': .064, 'algebraic-stack': .061, 'wiki': .016},
    'MixLaw fit': {
        'dclm': .568, 'arxiv': .000, 'starcoder': .000, 'pes2o': .097,
        'open-web-math': .035, 'algebraic-stack': .000, 'wiki': .300},
    'LightGBM fit': {
        'dclm': .553, 'arxiv': .212, 'starcoder': .087, 'pes2o': .082,
        'open-web-math': .042, 'algebraic-stack': .014, 'wiki': .011},
}

per_dom: dict[str, list] = defaultdict(list)
for fp in sorted(glob.glob(f'{HITS}/*.json')):
    dom = os.path.basename(fp).split('__')[1]
    d = json.load(open(fp, encoding='utf-8'))
    per_dom[dom].append((d.get('n_tokens', 0), set(int(k) for k in d['hits'])))
avail = {d: sum(t for t, _s in v) for d, v in per_dom.items()}


def take(dom, ntok):
    """Union of matched items in the first `ntok` tokens of `dom`."""
    acc = 0
    ids: set[int] = set()
    for t, s in per_dom.get(dom, []):
        if acc >= ntok:
            break
        acc += t
        ids |= s
    return acc, ids


def rate(ids):
    per = defaultdict(set)
    for gid in ids:
        per[ITEMS[gid][0]].add(gid)
    sh = sn = 0
    for t in NTASK:
        if t.rsplit('_', 1)[0] in SELFC:
            sh += len(per.get(t, ()))
            sn += NTASK[t]
    return 100.0 * len(ids) / TOT, (100.0 * sh / sn if sn else 0.0), per


print(f'arm budget = {BUDGET / 1e9:.1f}B tokens (matched to the validation runs)\n')
print(f'{"arm":26s}{"tokens":>9s}{"items":>8s}{"overall":>10s}'
      f'{"self-cont":>12s}{"short?":>9s}')
print('-' * 76)
out = {}
for name, w in ARMS.items():
    ids: set[int] = set()
    got = 0
    short = []
    for dom, wt in w.items():
        want = wt * BUDGET
        if want <= 0:
            continue
        a, s = take(dom, want)
        got += a
        ids |= s
        if a < want * 0.98:
            short.append(f'{dom} {a / 1e9:.2f}/{want / 1e9:.2f}B')
    r, sc, per = rate(ids)
    out[name] = {'tokens': got, 'items_matched': len(ids),
                 'overall_pct': round(r, 4), 'self_contained_pct': round(sc, 4),
                 'short_domains': short,
                 'per_task': {t: [len(per.get(t, ())), NTASK[t]]
                              for t in sorted(NTASK)}}
    print(f'{name:26s}{got / 1e9:>8.2f}B{len(ids):>8d}{r:>9.4f}%{sc:>11.4f}%'
          f'{(";".join(short) or "-"):>9s}')

base = out['Olmo-mix-1124 (natural)']['overall_pct']
dml = out['Data Mixing Laws paper']['overall_pct']
print(f'\n{"arm":26s}{"vs natural":>12s}{"vs DML":>9s}')
print('-' * 76)
for name in ARMS:
    o = out[name]['overall_pct']
    print(f'{name:26s}{o / base:>11.2f}x{o / dml:>8.2f}x')

print('\nreservoir availability (tokens scanned per domain):')
for d in sorted(avail, key=lambda k: -avail[k]):
    print(f'  {d:18s}{avail[d] / 1e9:>8.2f}B')

if OUT:
    json.dump({'budget_tokens': BUDGET, 'arms': out, 'arm_weights': ARMS},
              open(OUT, 'w', encoding='utf-8'), indent=1)
    print(f'\nwrote {OUT}')
