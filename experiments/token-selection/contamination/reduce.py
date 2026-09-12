#!/usr/bin/env python3
"""Merge per-shard scan results into per-corpus / per-source / per-task rates.

usage: reduce.py EVAL_INDEX.pkl RESULTS_DIR OUT.json
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
idx_path, res_dir, out_path = sys.argv[1:4]
with open(idx_path, 'rb') as f:
    IDX = pickle.load(f)
ITEMS = IDX['items']
NTASK = IDX['ntask']

files = sorted(glob.glob(f'{res_dir}/*.json'))
print(f'{len(files)} shard results')

agg: dict[tuple[str, str], dict] = {}
for fp in files:
    with open(fp, encoding='utf-8') as fh:
        r = json.load(fh)
    key = (r['corpus'], r['source'])
    a = agg.setdefault(key, {'n_docs': 0, 'n_docs_with_hit': 0, 'n_tokens': 0,
                             'items': set(), 'docs_hit_per_task': defaultdict(int),
                             'gram_hits_per_task': defaultdict(int), 'shards': 0})
    a['shards'] += 1
    a['n_docs'] += r['n_docs']
    a['n_docs_with_hit'] += r['n_docs_with_hit']
    a['n_tokens'] += r['n_tokens']
    a['items'].update(r['matched_item_ids'])
    for k, v in r['docs_hit_per_task'].items():
        a['docs_hit_per_task'][k] += v
    for k, v in r['gram_hits_per_task'].items():
        a['gram_hits_per_task'][k] += v


def task_rates(item_ids):
    per = defaultdict(set)
    for iid in item_ids:
        per[ITEMS[iid][0]].add(iid)
    return {t: (len(per.get(t, ())), NTASK[t], 100.0 * len(per.get(t, ())) / NTASK[t])
            for t in sorted(NTASK)}


out = {'corpora': {}, 'by_source': {}}
corpus_items: dict[str, set] = defaultdict(set)
corpus_tot: dict[str, dict] = defaultdict(lambda: {'n_docs': 0, 'n_docs_with_hit': 0,
                                                   'n_tokens': 0, 'shards': 0})

print(f'\n{"corpus / source":34s}{"shards":>7s}{"docs":>12s}{"tokens":>15s}'
      f'{"docs w/ hit":>13s}{"eval items matched":>21s}')
print('-' * 102)
for (corpus, source), a in sorted(agg.items()):
    tr = task_rates(a['items'])
    n_match = len(a['items'])
    n_total = sum(NTASK.values())
    out['by_source'][f'{corpus}/{source}'] = {
        'shards': a['shards'], 'n_docs': a['n_docs'],
        'n_docs_with_hit': a['n_docs_with_hit'], 'n_tokens': a['n_tokens'],
        'doc_hit_rate_pct': 100.0 * a['n_docs_with_hit'] / max(a['n_docs'], 1),
        'eval_items_matched': n_match,
        'eval_item_rate_pct': 100.0 * n_match / n_total,
        'per_task': tr,
        'docs_hit_per_task': dict(a['docs_hit_per_task']),
    }
    corpus_items[corpus].update(a['items'])
    c = corpus_tot[corpus]
    c['n_docs'] += a['n_docs']
    c['n_docs_with_hit'] += a['n_docs_with_hit']
    c['n_tokens'] += a['n_tokens']
    c['shards'] += a['shards']
    print(f'{corpus + " / " + source:34s}{a["shards"]:>7d}{a["n_docs"]:>12,d}'
          f'{a["n_tokens"]:>15,d}{a["n_docs_with_hit"]:>13,d}'
          f'{n_match:>10,d} / {n_total:,d} ({100.0 * n_match / n_total:.3f}%)')

n_total = sum(NTASK.values())
print('\n' + '=' * 102)
print('PER-CORPUS TOTALS')
for corpus, c in sorted(corpus_tot.items()):
    ids = corpus_items[corpus]
    out['corpora'][corpus] = {
        **c,
        'doc_hit_rate_pct': 100.0 * c['n_docs_with_hit'] / max(c['n_docs'], 1),
        'eval_items_matched': len(ids),
        'eval_item_rate_pct': 100.0 * len(ids) / n_total,
        'per_task': task_rates(ids),
    }
    print(f'\n{corpus}: {c["n_docs"]:,} docs, {c["n_tokens"]:,} tokens, '
          f'{c["n_docs_with_hit"]:,} docs with >=1 hit '
          f'({100.0 * c["n_docs_with_hit"] / max(c["n_docs"], 1):.4f}%)')
    print(f'  eval items matched: {len(ids):,} / {n_total:,} '
          f'({100.0 * len(ids) / n_total:.3f}%)')
    print(f'  {"task":34s}{"matched":>9s}{"n":>8s}{"rate":>9s}')
    for t, (m, n, pct) in out['corpora'][corpus]['per_task'].items():
        print(f'  {t:34s}{m:>9d}{n:>8d}{pct:>8.3f}%')

out['eval_items_total'] = n_total
out['n_tasks'] = len(NTASK)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1, default=list)
print(f'\nwrote {out_path}')
