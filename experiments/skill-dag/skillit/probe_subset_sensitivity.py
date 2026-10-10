#!/usr/bin/env python3
"""How much does the probe adjacency matrix depend on which eval items are scored?

Input: per-item losses of the eight finished probe checkpoints (step 1451),
written by ``eval_item_losses.py``. Everything below is computed on the measured
step-1451 losses; the reported A_ij additionally extrapolates each probe's loss
curve to the Chinchilla step, which a single final checkpoint cannot reproduce, so
these are sensitivity numbers for the matrix's inputs, not the reported matrix.

  A_ij = max(0, L_j(reference) - L_j(one-hot i)),  reference = LightGBM-start probe

``L_j`` is the mean bits-per-byte of the gold continuation over a set of items of
task family ``j`` (the six curve labels are one-to-one with the six families). The
same items are scored for every probe, so every comparison is paired.

Subsets compared: the items the in-run evals scored (first 16 batches of 8), all
items, first-N prefixes, and random N-item subsets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SKILLIT = Path(__file__).resolve().parent
DOMAINS = ["dclm", "arxiv", "starcoder", "pes2o", "open-web-math", "algebraic-stack", "wiki"]
REF = "lgb_start"
LABELS = [
    "arc_challenge_val_rc_5shot_bpb",
    "arc_easy_val_rc_5shot_bpb",
    "mmlu_humanities_val_rc_5shot_bpb",
    "mmlu_other_val_rc_5shot_bpb",
    "mmlu_social_sciences_val_rc_5shot_bpb",
    "mmlu_stem_val_rc_5shot_bpb",
]
FAMILIES = [lb.replace("_val_rc_5shot_bpb", "") for lb in LABELS]
INRUN_BATCHES = 16  # in-run evals: 16 batches of 8 sequences per label


def load_probe(path: Path) -> dict[str, dict]:
    """label -> {doc_ids: [..] in loader order, bpb: gold bpb per doc, inrun: bool mask}."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for label in LABELS:
        rows = payload["labels"][label]
        by_doc: dict[int, dict[int, tuple[float, int]]] = {}
        order: list[int] = []
        for doc, cont, bpb, batch in rows:
            if doc not in by_doc:
                by_doc[doc] = {}
                order.append(doc)
            by_doc[doc].setdefault(cont, (bpb, batch))
        docs, vals, inrun = [], [], []
        for doc in order:
            conts = by_doc[doc]
            if 0 not in conts:
                continue
            docs.append(doc)
            vals.append(conts[0][0])
            # replicate ICLMetric.compute over only the first INRUN_BATCHES batches: the
            # doc counts if its continuations seen so far are 0..k-1 (else it is skipped).
            seen = sorted(c for c, (_, b) in conts.items() if b < INRUN_BATCHES)
            inrun.append(bool(seen) and seen == list(range(len(seen))) and conts[0][1] < INRUN_BATCHES)
        out[label] = {"docs": np.array(docs), "bpb": np.array(vals), "inrun": np.array(inrun)}
    return out


def matrix_from_losses(L: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """L: (7 probes, 6 families); ref: (6,). Returns A."""
    return np.maximum(0.0, ref[None, :] - L)


def losses(probes: dict[str, dict], select: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-family mean loss for the seven one-hots and the reference over item index sets."""
    L = np.zeros((len(DOMAINS), len(LABELS)))
    ref = np.zeros(len(LABELS))
    for j, label in enumerate(LABELS):
        idx = select[label]
        for i, d in enumerate(DOMAINS):
            L[i, j] = probes[d][label]["bpb"][idx].mean()
        ref[j] = probes[REF][label]["bpb"][idx].mean()
    return L, ref


def compare(A: np.ndarray, A_ref: np.ndarray) -> dict[str, float]:
    nz, nz_ref = A > 0, A_ref > 0
    r = float(np.corrcoef(A.ravel(), A_ref.ravel())[0, 1]) if A.std() > 0 and A_ref.std() > 0 else float("nan")
    return {
        "nonzero": int(nz.sum()),
        "edges_disagree": int((nz != nz_ref).sum()),
        "corr": r,
        "mean_abs_diff": float(np.abs(A - A_ref).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items-dir", type=Path, default=SKILLIT / "artifacts/probe_subset_sensitivity/items")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=SKILLIT / "artifacts/probe_subset_sensitivity/results.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    probes = {name: load_probe(args.items_dir / f"probe_{name}.json") for name in DOMAINS + [REF]}
    n_docs = {lb: len(probes[REF][lb]["docs"]) for lb in LABELS}
    for name in DOMAINS:
        for lb in LABELS:  # the same items, in the same order, for every probe
            assert np.array_equal(probes[name][lb]["docs"], probes[REF][lb]["docs"]), (name, lb)
    inrun_n = {lb: int(probes[REF][lb]["inrun"].sum()) for lb in LABELS}
    print("items per label:", n_docs)
    print("items the in-run eval scored:", inrun_n)

    # is the in-run subset (the first items of each label, in the loader's order) typical?
    print("\nreference probe: mean loss on the in-run items vs 5000 random subsets of the same size")
    zs = {}
    for lb in LABELS:
        b = probes[REF][lb]["bpb"]
        k = inrun_n[lb]
        draws = np.array([b[rng.choice(len(b), k, replace=False)].mean() for _ in range(5000)])
        first = float(b[probes[REF][lb]["inrun"]].mean())
        zs[lb] = {"all_items": float(b.mean()), "inrun": first, "random_sd": float(draws.std()),
                  "z": float((first - draws.mean()) / draws.std())}
        print(f"  {lb[:-14]:<24} all {b.mean():.3f}  in-run {first:.3f}  random-subset sd {draws.std():.3f}  z={zs[lb]['z']:+.1f}")

    full = {lb: np.arange(n_docs[lb]) for lb in LABELS}
    inrun = {lb: np.flatnonzero(probes[REF][lb]["inrun"]) for lb in LABELS}
    L_full, ref_full = losses(probes, full)
    L_in, ref_in = losses(probes, inrun)
    A_full = matrix_from_losses(L_full, ref_full)
    A_in = matrix_from_losses(L_in, ref_in)
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    print("\nL_j(reference) full / in-run:\n", ref_full, "\n", ref_in)
    print("\nA on ALL items:\n", A_full, "\n nonzero", int((A_full > 0).sum()))
    print("\nA on the items the in-run evals scored:\n", A_in, "\n nonzero", int((A_in > 0).sum()))
    print("\nin-run vs all items:", compare(A_in, A_full))

    result: dict[str, object] = {
        "items_per_label": n_docs,
        "inrun_items_per_label": inrun_n,
        "A_all_items": A_full.tolist(),
        "A_inrun_items": A_in.tolist(),
        "inrun_vs_all": compare(A_in, A_full),
        "reference_loss_inrun_vs_all": zs,
    }

    # first-N prefixes of each label's item list (the loader's order)
    print("\nfirst-N prefixes vs all items")
    prefixes = {}
    for n in (16, 32, 64, 128, 256, 512, 10**9):
        sel = {lb: np.arange(min(n, n_docs[lb])) for lb in LABELS}
        A = matrix_from_losses(*losses(probes, sel))
        c = compare(A, A_full)
        name = "all" if n == 10**9 else str(n)
        prefixes[name] = {**c, "A": A.tolist()}
        print(f"  first {name:>4}: nonzero {c['nonzero']:2d}  edge disagreements {c['edges_disagree']:2d}  "
              f"corr {c['corr']:.2f}  mean|dA| {c['mean_abs_diff']:.4f}")
    result["first_n"] = prefixes

    # random subsets of N items per label, the same items for every probe in a draw
    print("\nrandom N-item subsets per label (paired across probes), %d draws" % args.draws)
    rand = {}
    for n in (64, 128, 256):
        As, stats = [], []
        for _ in range(args.draws):
            sel = {lb: np.sort(rng.choice(n_docs[lb], size=min(n, n_docs[lb]), replace=False)) for lb in LABELS}
            A = matrix_from_losses(*losses(probes, sel))
            As.append(A)
            stats.append(compare(A, A_full))
        As = np.array(As)
        freq = (As > 0).mean(0)
        nz = np.array([s["nonzero"] for s in stats])
        dis = np.array([s["edges_disagree"] for s in stats])
        cor = np.array([s["corr"] for s in stats])
        mad = np.array([s["mean_abs_diff"] for s in stats])
        inrun_c = compare(A_in, A_full)
        rand[str(n)] = {
            "nonzero_mean": float(nz.mean()), "nonzero_p5_p95": [float(np.percentile(nz, 5)), float(np.percentile(nz, 95))],
            "edges_disagree_mean": float(dis.mean()), "edges_disagree_p5_p95": [float(np.percentile(dis, 5)), float(np.percentile(dis, 95))],
            "corr_median": float(np.nanmedian(cor)), "corr_p5_p95": [float(np.nanpercentile(cor, 5)), float(np.nanpercentile(cor, 95))],
            "mean_abs_diff_mean": float(mad.mean()),
            "edge_frequency": freq.tolist(), "A_mean": As.mean(0).tolist(), "A_sd": As.std(0).tolist(),
            "inrun_edges_disagree_percentile": float((dis < inrun_c["edges_disagree"]).mean() * 100),
            "inrun_mean_abs_diff_percentile": float((mad < inrun_c["mean_abs_diff"]).mean() * 100),
        }
        r = rand[str(n)]
        print(f"  N={n:>3}: nonzero {r['nonzero_mean']:.1f} (5-95%: {r['nonzero_p5_p95'][0]:.0f}-{r['nonzero_p5_p95'][1]:.0f}), "
              f"edge disagreements vs all-items {r['edges_disagree_mean']:.1f} ({r['edges_disagree_p5_p95'][0]:.0f}-{r['edges_disagree_p5_p95'][1]:.0f}), "
              f"corr median {r['corr_median']:.2f}, mean|dA| {r['mean_abs_diff_mean']:.4f}")
        print(f"         in-run subset's mean|dA| is at the {r['inrun_mean_abs_diff_percentile']:.0f}th percentile of random subsets; "
              f"edge disagreements at the {r['inrun_edges_disagree_percentile']:.0f}th")
        if n == 128:
            print("  edge frequency over random-128 draws (rows: domains):")
            for d, row in zip(DOMAINS, freq):
                print(f"    {d:<16}", " ".join(f"{v:5.2f}" for v in row))
            print("  A sd over random-128 draws:")
            for d, row in zip(DOMAINS, As.std(0)):
                print(f"    {d:<16}", " ".join(f"{v:5.3f}" for v in row))
    result["random_subsets"] = rand

    # sampling uncertainty of the all-items matrix itself: resample items with replacement
    boots = []
    for _ in range(args.draws):
        sel = {lb: rng.integers(0, n_docs[lb], size=n_docs[lb]) for lb in LABELS}
        boots.append(matrix_from_losses(*losses(probes, sel)))
    boots = np.array(boots)
    result["bootstrap_all_items"] = {"edge_frequency": (boots > 0).mean(0).tolist(), "A_sd": boots.std(0).tolist()}
    print("\nedge frequency when resampling ALL items with replacement (item-sampling noise of the full-set matrix):")
    for d, row in zip(DOMAINS, (boots > 0).mean(0)):
        print(f"    {d:<16}", " ".join(f"{v:5.2f}" for v in row))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
