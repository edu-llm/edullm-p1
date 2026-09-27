#!/usr/bin/env python3
"""Network-free check: does the layout rule reproduce every published object?

Given only outputs.json's per-file token counts (file_streams) and the
layout rule stated in outputs.json["layout"] (1 GiB shards in path/join
order, then the last floor(total_bytes * 0.0015) bytes -- rounded down to a
multiple of 4 -- carved off the tail into val-00000.u32le.bin), this
recomputes the name/byte-length/byte-offset of every object from arithmetic
alone and compares against outputs.json["objects"]. No download, no
tokenizing, no FarmShare: this is the cheap complement to the expensive
per-file/per-object hash checks (which prove the *bytes* are right; this
proves the *layout rule* -- i.e. that outputs.json's own bookkeeping is
self-consistent, not just that a hash happens to match).

Usage: python verify_layout.py [--manifest-dir DIR]
Exit status is non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GIB = 1073741824


def simulate(total_tokens: int, shard_bytes: int, val_fraction: float) -> list[tuple[str, int, int]]:
    """Return [(name, stream_start_byte, bytes), ...] train shards + val, in order."""
    total_bytes = total_tokens * 4
    val_bytes = int(total_bytes * val_fraction)
    val_bytes -= val_bytes % 4
    if total_bytes == 0:
        return [("val-00000.u32le.bin", 0, 0)]
    n_shards = (total_bytes + shard_bytes - 1) // shard_bytes
    sizes = [shard_bytes] * n_shards
    sizes[-1] = total_bytes - (n_shards - 1) * shard_bytes
    remaining = val_bytes
    i = len(sizes) - 1
    while remaining > 0 and i >= 0:
        if sizes[i] <= remaining:
            remaining -= sizes[i]
            sizes[i] = 0
            i -= 1
        else:
            sizes[i] -= remaining
            remaining = 0
    if remaining != 0:
        raise SystemExit(f"could not carve {val_bytes} bytes of val (short by {remaining})")
    objects, pos = [], 0
    for idx, size in enumerate(s for s in sizes if s > 0):
        objects.append((f"train-{idx:05d}.u32le.bin", pos, size))
        pos += size
    objects.append(("val-00000.u32le.bin", pos, val_bytes))
    return objects


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-dir", type=Path, default=HERE)
    args = ap.parse_args()

    outputs = json.loads((args.manifest_dir / "outputs.json").read_text(encoding="utf-8"))
    sources = json.loads((args.manifest_dir / "sources.json").read_text(encoding="utf-8"))
    files_by_src = {f["src"]: f for f in sources["files"]}

    layout = outputs["layout"]
    shard_bytes = int(layout["shard_bytes"])
    val_fraction = float(layout["val_fraction"])

    totals: dict[str, int] = {}
    for fs in outputs["file_streams"]:
        s = files_by_src[fs["src"]]["source"]
        totals[s] = totals.get(s, 0) + fs["tokens_with_eos"]

    mismatches = 0
    for source in sorted(totals):
        predicted = simulate(totals[source], shard_bytes, val_fraction)
        actual = sorted(
            (o for o in outputs["objects"] if o["source"] == source),
            key=lambda o: (o["name"].endswith("val-00000.u32le.bin"), o["name"]),
        )
        ok = len(predicted) == len(actual)
        for (name, start, size), o in zip(predicted, actual):
            act_name = o["name"].rsplit("/", 1)[-1]
            act_start = o["stream_start_token"] * 4
            if name != act_name or start != act_start or size != o["bytes"]:
                ok = False
                mismatches += 1
                print(f"MISMATCH {source}/{name}: predicted start={start} bytes={size}; "
                      f"actual name={act_name} start={act_start} bytes={o['bytes']}", flush=True)
        print(f"{'OK      ' if ok else 'MISMATCH'} {source:16s} tokens={totals[source]:>14,d} "
              f"objects predicted={len(predicted)} actual={len(actual)}", flush=True)

    print(f"summary: {len(totals) - (mismatches > 0)}/{len(totals)} sources ok, "
          f"{mismatches} object-level mismatch(es)", flush=True)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
