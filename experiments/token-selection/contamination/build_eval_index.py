#!/usr/bin/env python3
"""Build the eval-side 13-gram index for the OLMES 5-shot RC suite.

Emits a pickle with:
  grams : dict[int, array of item-ids]   -- 64-bit stable FNV-1a rolling hash
  items : list[(task, item_key)]         -- item-id -> which eval item
  ntask : dict[task, n_items]

"Question text alone": for every task we take only the prompt stem the model is
shown, never the answer options. Field choices are recorded in `fields.json`
next to the index so the protocol is auditable.
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
OUT.mkdir(parents=True, exist_ok=True)
N = 13

MMLU_CATS = {
    'stem': """abstract_algebra anatomy astronomy college_biology college_chemistry
        college_computer_science college_mathematics college_physics computer_security
        conceptual_physics electrical_engineering elementary_mathematics high_school_biology
        high_school_chemistry high_school_computer_science high_school_mathematics
        high_school_physics high_school_statistics machine_learning""".split(),
    'humanities': """formal_logic high_school_european_history high_school_us_history
        high_school_world_history international_law jurisprudence logical_fallacies
        moral_disputes moral_scenarios philosophy prehistory professional_law
        world_religions""".split(),
    'social_sciences': """econometrics high_school_geography
        high_school_government_and_politics high_school_macroeconomics
        high_school_microeconomics high_school_psychology human_sexuality
        professional_psychology public_relations security_studies sociology
        us_foreign_policy""".split(),
    'other': """business_ethics clinical_knowledge college_medicine global_facts
        human_aging management marketing medical_genetics miscellaneous nutrition
        professional_accounting professional_medicine virology""".split(),
}
SUBJ2CAT = {s: c for c, subs in MMLU_CATS.items() for s in subs}

FIELDS = {
    'arc_challenge': 'question',
    'arc_easy': 'question',
    'boolq': 'passage + question (both shown in the RC prompt)',
    'csqa': 'question',
    'hellaswag': 'ctx (activity_label + context shown as the prompt)',
    'mmlu_*': 'question',
    'openbookqa': 'question_stem',
    'piqa': 'goal',
    'socialiqa': 'context + question',
    'winogrande': 'sentence',
}


def norm_tokens(text: str) -> list[str]:
    out, cur = [], []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            out.append(''.join(cur))
            cur = []
    if cur:
        out.append(''.join(cur))
    return out


_M = 0xFFFFFFFFFFFFFFFF
_memo: dict[str, int] = {}


def tok_hash(t: str) -> int:
    h = _memo.get(t)
    if h is None:
        h = 0xcbf29ce484222325
        for b in t.encode('utf-8'):
            h = ((h ^ b) * 0x100000001b3) & _M
        _memo[t] = h
    return h


def grams_of(text: str):
    toks = norm_tokens(text)
    if len(toks) < N:
        return
    hs = [tok_hash(t) for t in toks]
    for i in range(len(hs) - N + 1):
        g = 0xcbf29ce484222325
        for j in range(i, i + N):
            g = ((g ^ hs[j]) * 0x100000001b3) & _M
        yield g


def load(name, *a, **kw):
    try:
        return load_dataset(name, *a, **kw)
    except Exception as e:
        print(f'  !! {name} {a}: {type(e).__name__}: {e}', flush=True)
        return None


def add(task, split, rows, textfn, bucket):
    if rows is None:
        return
    for i, r in enumerate(rows):
        try:
            txt = textfn(r)
        except Exception:
            continue
        if not txt:
            continue
        bucket.append((f'{task}_{split}', f'{task}_{split}#{i}', txt))


records: list[tuple[str, str, str]] = []

print('ARC', flush=True)
for cfg, task in (('ARC-Challenge', 'arc_challenge'), ('ARC-Easy', 'arc_easy')):
    d = load('allenai/ai2_arc', cfg)
    if d:
        for sp, lbl in (('validation', 'val'), ('test', 'test')):
            add(task, lbl, d.get(sp), lambda r: r['question'], records)

print('BoolQ', flush=True)
d = load('google/boolq')
if d:
    add('boolq', 'val', d.get('validation'),
        lambda r: f"{r['passage']} {r['question']}", records)

print('CSQA', flush=True)
d = load('tau/commonsense_qa')
if d:
    add('csqa', 'val', d.get('validation'), lambda r: r['question'], records)

print('HellaSwag', flush=True)
d = load('Rowan/hellaswag')
if d:
    add('hellaswag', 'val', d.get('validation'),
        lambda r: r.get('ctx') or r.get('ctx_a') or '', records)

print('MMLU', flush=True)
d = load('cais/mmlu', 'all')
if d:
    for sp, lbl in (('validation', 'val'), ('test', 'test')):
        rows = d.get(sp)
        if rows is None:
            continue
        for i, r in enumerate(rows):
            cat = SUBJ2CAT.get(r.get('subject', ''))
            if not cat or not r.get('question'):
                continue
            records.append((f'mmlu_{cat}_{lbl}', f'mmlu_{cat}_{lbl}#{i}', r['question']))

print('OpenBookQA', flush=True)
d = load('allenai/openbookqa', 'main')
if d:
    for sp, lbl in (('validation', 'val'), ('test', 'test')):
        add('openbookqa', lbl, d.get(sp), lambda r: r['question_stem'], records)

print('PIQA', flush=True)
d = load('ybisk/piqa', trust_remote_code=True)
if d:
    add('piqa', 'val', d.get('validation'), lambda r: r['goal'], records)

print('SocialIQA', flush=True)
d = load('allenai/social_i_qa', trust_remote_code=True)
if d:
    add('socialiqa', 'val', d.get('validation'),
        lambda r: f"{r['context']} {r['question']}", records)

print('WinoGrande', flush=True)
d = load('allenai/winogrande', 'winogrande_xl', trust_remote_code=True)
if d:
    add('winogrande', 'val', d.get('validation'), lambda r: r['sentence'], records)

# ------------------------------------------------------------------ index ---
items: list[tuple[str, str]] = []
grams: dict[int, list[int]] = defaultdict(list)
ntask: dict[str, int] = defaultdict(int)
short: dict[str, int] = defaultdict(int)

for task, key, txt in records:
    iid = len(items)
    items.append((task, key))
    ntask[task] += 1
    got = False
    for g in grams_of(txt):
        grams[g].append(iid)
        got = True
    if not got:
        short[task] += 1

grams = {k: tuple(sorted(set(v))) for k, v in grams.items()}
print(f'\nitems: {len(items)}   distinct 13-grams: {len(grams)}')
print(f'tasks: {len(ntask)}')
for t in sorted(ntask):
    print(f'  {t:34s} n={ntask[t]:6d}  too-short={short.get(t, 0)}')

with open(OUT / 'eval_index.pkl', 'wb') as f:
    pickle.dump({'grams': grams, 'items': items, 'ntask': dict(ntask),
                 'too_short': dict(short), 'n': N}, f, protocol=5)
(OUT / 'fields.json').write_text(json.dumps(
    {'n': N, 'fields': FIELDS,
     'normalization': 'lowercase; non-alphanumeric -> separator; whitespace tokens',
     'items': len(items), 'grams': len(grams)}, indent=1), encoding='utf-8')
print(f"\nwrote {OUT / 'eval_index.pkl'}")
