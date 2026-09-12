#!/usr/bin/env python3
"""Aggregate strict-protocol hits into per-corpus and per-domain tables.

usage: reduce_strict.py HITS_DIR OUT.json
Filenames: <corpus>__<source>__<taskid>.json
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
HITS, OUT = sys.argv[1], sys.argv[2]
C = '/scratch/users/nzhao2/agent-runs/contam-20260910-060952'
E = pickle.load(open(f'{C}/eval_ngrams_strict.pkl', 'rb'))
ITEMS = E['items']

NTASK: dict[str, int] = defaultdict(int)
for lbl, _i, _t in ITEMS:
    NTASK[lbl] += 1
TOT = len(ITEMS)
SELFC = ('arc_challenge', 'arc_easy', 'csqa', 'openbookqa', 'piqa',
         'winogrande', 'socialiqa', 'boolq')

by_cs: dict[tuple[str, str], dict] = {}
files = sorted(glob.glob(f'{HITS}/*.json'))
print(f'{len(files)} hit files')
for fp in files:
    base = os.path.basename(fp)[:-5]
    corpus, source, _ = base.split('__', 2)
    try:
        d = json.load(open(fp, encoding='utf-8'))
    except Exception as e:
        print(f'  !! unreadable {base}: {e}')
        continue
    a = by_cs.setdefault((corpus, source), {'docs': 0, 'ngrams': 0,
                                            'items': set(), 'gram_hits': 0,
                                            'tasks': 0})
    a['tasks'] += 1
    a['docs'] += d['docs']
    a['ngrams'] += d['ngrams']
    for gid, rec in d['hits'].items():
        a['items'].add(int(gid))
        a['gram_hits'] += rec['n']


def rates(ids):
    per = defaultdict(set)
    for gid in ids:
        per[ITEMS[gid][0]].add(gid)
    tab = {t: [len(per.get(t, ())), NTASK[t],
               round(100.0 * len(per.get(t, ())) / NTASK[t], 4)]
           for t in sorted(NTASK)}
    sh = sn = 0
    for t, (h, n, _) in tab.items():
        if t.rsplit('_', 1)[0] in SELFC:
            sh += h
            sn += n
    return tab, sh, sn


out: dict = {'eval_items_total': TOT, 'by_domain': {}, 'by_corpus': {}}
corpus_items: dict[str, set] = defaultdict(set)
corpus_tot: dict[str, dict] = defaultdict(
    lambda: {'docs': 0, 'ngrams': 0, 'tasks': 0, 'gram_hits': 0})

print(f'\n{"corpus":22s}{"domain":18s}{"docs":>13s}{"13-grams":>17s}'
      f'{"items":>8s}{"rate":>9s}')
print('-' * 90)
for (corpus, source), a in sorted(by_cs.items()):
    tab, sh, sn = rates(a['items'])
    n = len(a['items'])
    out['by_domain'][f'{corpus}/{source}'] = {
        'tasks': a['tasks'], 'docs': a['docs'], 'ngrams': a['ngrams'],
        'gram_hits': a['gram_hits'], 'items_matched': n,
        'item_rate_pct': round(100.0 * n / TOT, 4),
        'self_contained_pct': round(100.0 * sh / sn, 4) if sn else None,
        'per_task': tab,
    }
    print(f'{corpus:22s}{source:18s}{a["docs"]:>13,d}{a["ngrams"]:>17,d}'
          f'{n:>8d}{100.0 * n / TOT:>8.4f}%')
    corpus_items[corpus] |= a['items']
    c = corpus_tot[corpus]
    for k in ('docs', 'ngrams', 'tasks', 'gram_hits'):
        c[k] += a[k]

print('\n' + '=' * 90)
print('PER-CORPUS TOTALS (union of matched eval items)')
print(f'{"corpus":22s}{"docs":>13s}{"13-grams":>17s}{"items":>8s}'
      f'{"overall":>10s}{"self-cont":>11s}')
print('-' * 90)
for corpus, c in sorted(corpus_tot.items()):
    ids = corpus_items[corpus]
    tab, sh, sn = rates(ids)
    out['by_corpus'][corpus] = {
        **c, 'items_matched': len(ids),
        'item_rate_pct': round(100.0 * len(ids) / TOT, 4),
        'self_contained_pct': round(100.0 * sh / sn, 4) if sn else None,
        'per_task': tab,
    }
    print(f'{corpus:22s}{c["docs"]:>13,d}{c["ngrams"]:>17,d}{len(ids):>8d}'
          f'{100.0 * len(ids) / TOT:>9.4f}%{100.0 * sh / sn:>10.4f}%')

print('\nPER-TASK, by corpus')
for corpus in sorted(out['by_corpus']):
    print(f'\n  {corpus}')
    for t, (h, n, pct) in out['by_corpus'][corpus]['per_task'].items():
        print(f'    {t:22s}{h:>7d}/{n:<7d}{pct:>8.4f}%')

json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1)
print(f'\nwrote {OUT}')
