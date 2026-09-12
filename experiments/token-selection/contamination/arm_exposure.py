#!/usr/bin/env python3
"""Two things the domain-weighting paper needs:

(1) per-domain contamination at a COMMON token budget, so the seven rates are
    comparable to each other rather than confounded by how much was scanned;
(2) each validated arm's expected exposure = sum(weight_d * rate_d), using the
    arm weights of the paper's Table I.

usage: arm_exposure.py RES_HITS_DIR [OUT.json]
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
OUT = sys.argv[2] if len(sys.argv) > 2 else None
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
    per_dom[dom].append((d.get('n_tokens', 0), d['ngrams'],
                         set(int(k) for k in d['hits'])))

totals = {d: sum(t for t, _g, _s in v) for d, v in per_dom.items()}
BUDGET = min(totals.values())
smallest = min(totals, key=totals.get)
print(f'common per-domain budget = {BUDGET / 1e9:.3f}B tokens '
      f'(set by the smallest domain, {smallest})\n')


def rate(ids):
    per = defaultdict(set)
    for gid in ids:
        per[ITEMS[gid][0]].add(gid)
    sh = sn = 0
    for t in NTASK:
        if t.rsplit('_', 1)[0] in SELFC:
            sh += len(per.get(t, ()))
            sn += NTASK[t]
    return 100.0 * len(ids) / TOT, (100.0 * sh / sn if sn else 0.0)


print(f'{"domain":18s}{"tokens":>12s}{"full rate":>11s}'
      f'{"capped rate":>13s}{"capped self-c":>15s}{"items/Bgram":>13s}')
print('-' * 84)
capped: dict[str, float] = {}
capped_sc: dict[str, float] = {}
rows_out = {}
for dom in sorted(per_dom, key=lambda d: -totals[d]):
    rows = per_dom[dom]
    full_ids: set[int] = set()
    for _t, _g, s in rows:
        full_ids |= s
    fr, fs = rate(full_ids)
    acc = 0
    cap_ids: set[int] = set()
    for t, _g, s in rows:
        if acc >= BUDGET:
            break
        acc += t
        cap_ids |= s
    cr, cs = rate(cap_ids)
    capped[dom] = cr
    capped_sc[dom] = cs
    grams = sum(g for _t, g, _s in rows)
    ipg = len(full_ids) / (grams / 1e9)
    rows_out[dom] = {'tokens': totals[dom], 'grams': grams,
                     'full_rate_pct': round(fr, 4),
                     'full_self_contained_pct': round(fs, 4),
                     'capped_tokens': acc, 'capped_rate_pct': round(cr, 4),
                     'capped_self_contained_pct': round(cs, 4),
                     'items_per_Bgram': round(ipg, 1)}
    print(f'{dom:18s}{totals[dom] / 1e9:>10.3f}B{fr:>10.4f}%{cr:>12.4f}%'
          f'{cs:>14.4f}%{ipg:>13.1f}')

print('\n' + '=' * 84)
print('EXPECTED ARM EXPOSURE  =  sum over domains of (weight x capped rate)')
print(f'{"arm":26s}{"overall":>10s}{"self-cont":>12s}{"vs DML":>9s}{"vs natural":>12s}')
print('-' * 84)
ex_all, ex_sc = {}, {}
for name, w in ARMS.items():
    ex_all[name] = sum(w.get(d, 0.0) * capped.get(d, 0.0) for d in capped)
    ex_sc[name] = sum(w.get(d, 0.0) * capped_sc.get(d, 0.0) for d in capped)
base = ex_all['Data Mixing Laws paper']
nat = ex_all['Olmo-mix-1124 (natural)']
for name in ARMS:
    print(f'{name:26s}{ex_all[name]:>9.4f}%{ex_sc[name]:>11.4f}%'
          f'{ex_all[name] / base:>8.2f}x{ex_all[name] / nat:>11.2f}x')

if OUT:
    json.dump({'budget_tokens': BUDGET, 'per_domain': rows_out,
               'arm_exposure_pct': {k: round(v, 4) for k, v in ex_all.items()},
               'arm_exposure_self_contained_pct':
                   {k: round(v, 4) for k, v in ex_sc.items()},
               'arm_weights': ARMS},
              open(OUT, 'w', encoding='utf-8'), indent=1)
    print(f'\nwrote {OUT}')
