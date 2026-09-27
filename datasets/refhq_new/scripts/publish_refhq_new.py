#!/usr/bin/env python3
"""Build the local tokens/<source>/<domain>/ layout split for refhq-new as pretrain/refhq-instruct.

Note: the dataset id is ``pretrain/refhq-instruct``, not ``refhq-new`` (kept for the
FarmShare scratch layout only).

Layout (holdout already done before tokenize):
  tokens/<source>/<domain>/{train,val}-NNNNN.u32le.bin

Reuses split_npy_to_shards pattern from datasets/refhq/scripts/publish_refhq_edullm_data.py
but does NOT carve val (doc-level 0.15% holdout already produced train/val memmaps).

Tokenizer dep: tokenizer/dolma2-bpe.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATASET_ID = "pretrain/refhq-instruct"
DEFAULT_TOKENIZER = "tokenizer/dolma2-bpe"
DEFAULT_PURPOSE = (
    "One-pass filtered instruct mix (Tulu-v2/OH-2.5, Tulu-3/Hermes-3, SmolTalk/Dolci) "
    "for OLMo-2 370M CE reference / rho-1; tool/safety/IF/Aya removed; Dolma English; "
    "tuned for 20-label OLMES BPB"
)
DEFAULT_SHARD_BYTES = 1_073_741_824
SPLITS = ("train", "val")


def _align_shard_bytes(n: int) -> int:
    n -= n % 4
    if n < 4:
        raise ValueError("shard_bytes must be at least 4")
    return n


def split_npy_to_shards(
    npy_path: Path,
    out_dir: Path,
    *,
    shard_bytes: int,
    split: str,
) -> list[Path]:
    """Split a headerless uint32 memmap into {split}-NNNNN.u32le.bin shards."""
    shard_bytes = _align_shard_bytes(shard_bytes)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = npy_path.stat().st_size
    if total % 4 != 0:
        raise ValueError(f"{npy_path}: size {total} is not a multiple of 4 (uint32)")

    written: list[Path] = []
    shard_idx = 0
    with npy_path.open("rb") as src:
        while True:
            chunk = src.read(shard_bytes)
            if not chunk:
                break
            if len(chunk) % 4 != 0:
                raise ValueError(f"{npy_path}: trailing partial token in shard {shard_idx}")
            out = out_dir / f"{split}-{shard_idx:05d}.u32le.bin"
            out.write_bytes(chunk)
            written.append(out)
            shard_idx += 1
    if not written:
        raise ValueError(f"{npy_path}: empty input")
    return written


def resolve_tok_root(tokenized_root: Path | None, scratch_root: Path) -> Path:
    if tokenized_root is not None and tokenized_root.is_dir():
        return tokenized_root
    for name in ("tokenized", "tok"):
        cand = scratch_root / name
        if cand.is_dir():
            return cand
    raise SystemExit(f"missing tokenized/ or tok/ under {scratch_root}")

def iter_source_domain_dirs(tok_root: Path) -> list[tuple[str, str, Path]]:
    pairs: list[tuple[str, str, Path]] = []
    for source_dir in sorted(p for p in tok_root.iterdir() if p.is_dir()):
        for domain_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
            pairs.append((source_dir.name, domain_dir.name, domain_dir))
    return pairs


def stage_publish_layout(
    *,
    tok_root: Path,
    out_root: Path,
    shard_bytes: int,
    force: bool,
) -> dict[str, list[str]]:
    """Build tokens/<source>/<domain>/{train,val}-*.u32le.bin — no val carve."""
    if out_root.exists():
        if not force:
            raise SystemExit(f"staging dir exists: {out_root} (pass --force to replace)")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    staged: dict[str, list[str]] = {}
    for source, domain, domain_dir in iter_source_domain_dirs(tok_root):
        key = f"{source}/{domain}"
        rel_dir = Path("tokens") / source / domain
        dest = out_root / rel_dir
        dest.mkdir(parents=True, exist_ok=True)
        staged[key] = []
        found_train = False
        for split in SPLITS:
            npy = domain_dir / f"{split}.npy"
            existing = sorted(domain_dir.glob(f"{split}-*.u32le.bin"))
            if npy.is_file():
                shards = split_npy_to_shards(
                    npy, dest, shard_bytes=shard_bytes, split=split
                )
                staged[key].extend(str(rel_dir / p.name) for p in shards)
                print(
                    f"staged {key} {split}: {len(shards)} shard(s) from {npy.name}",
                    flush=True,
                )
                if split == "train":
                    found_train = True
            elif existing:
                for src in existing:
                    dst = dest / src.name
                    shutil.copy2(src, dst)
                    staged[key].append(str(rel_dir / src.name))
                print(
                    f"staged {key} {split}: copied {len(existing)} existing u32le shard(s)",
                    flush=True,
                )
                if split == "train":
                    found_train = True
            elif split == "val":
                print(f"note: {key}: no val split (ok for tiny smoke)", flush=True)
        if not found_train:
            raise SystemExit(f"{key}: missing train.npy or train-*.u32le.bin under {domain_dir}")
    if not staged:
        raise SystemExit(f"no source/domain pairs under {tok_root}")
    return staged


def load_or_build_manifest(manifest_path: Path | None, tok_root: Path) -> dict:
    if manifest_path is not None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("accepted", True):
            raise SystemExit(f"tokenized manifest not accepted: {manifest_path}")
        return manifest

    # Build on the fly from tok/ (same shape as finalize_upload).
    by_source: dict[str, int] = {}
    pairs: dict[str, dict] = {}
    total = 0
    for source, domain, domain_dir in iter_source_domain_dirs(tok_root):
        key = f"{source}/{domain}"
        splits: dict[str, dict] = {}
        pair_tokens = 0
        for split in SPLITS:
            npy = domain_dir / f"{split}.npy"
            shards = sorted(domain_dir.glob(f"{split}-*.u32le.bin"))
            meta_path = domain_dir / f"{split}.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            if npy.is_file():
                nbytes = npy.stat().st_size
                tokens = int(
                    meta.get("stream_tokens_with_eos")
                    or meta.get("tokens_with_eos")
                    or (nbytes // 4)
                )
            elif shards:
                nbytes = sum(p.stat().st_size for p in shards)
                tokens = int(
                    meta.get("stream_tokens_with_eos")
                    or meta.get("tokens_with_eos")
                    or (nbytes // 4)
                )
            else:
                continue
            splits[split] = {"bytes": nbytes, "stream_tokens_with_eos": tokens}
            pair_tokens += tokens
        if "train" not in splits:
            continue
        pairs[key] = {
            "source": source,
            "domain": domain,
            "splits": splits,
            "stream_tokens_with_eos": pair_tokens,
        }
        by_source[source] = by_source.get(source, 0) + pair_tokens
        total += pair_tokens
    return {
        "accepted": True,
        "total_stream_tokens_with_eos": total,
        "by_source": by_source,
        "pairs": pairs,
    }


def build_sources(manifest: dict) -> list[dict]:
    total = int(manifest.get("total_stream_tokens_with_eos") or 0)
    by_source = manifest.get("by_source") or {}
    if not by_source and manifest.get("pairs"):
        for meta in manifest["pairs"].values():
            src = meta["source"]
            by_source[src] = by_source.get(src, 0) + int(meta["stream_tokens_with_eos"])
    sources: list[dict] = []
    for name in sorted(by_source):
        tokens = int(by_source[name])
        row: dict = {
            "name": name,
            "tokens": tokens,
            "scope": "measured-in-this-dataset",
        }
        if total:
            row["share"] = round(tokens / total, 6)
        sources.append(row)
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenized-root",
        type=Path,
        default=None,
        help="tokenized/ root (default: <scratch>/tokenized or …/tok)",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=None,
        help="run scratch root containing tok/, out/, holdout/, manifests/",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="tokenized_manifest.json from finalize_upload (optional)",
    )
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--purpose", default=DEFAULT_PURPOSE)
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument("--skip-stage", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    scratch = args.scratch_root
    if scratch is None:
        if args.tokenized_root is not None:
            scratch = args.tokenized_root.parent
        elif args.manifest is not None:
            scratch = args.manifest.parent.parent
        else:
            raise SystemExit("pass --scratch-root or --tokenized-root or --manifest")

    tok_root = resolve_tok_root(args.tokenized_root, scratch)
    manifest_path = args.manifest or (scratch / "manifests" / "tokenized_manifest.json")
    manifest = load_or_build_manifest(
        manifest_path if manifest_path.is_file() else None,
        tok_root,
    )

    if not args.skip_stage:
        stage_publish_layout(
            tok_root=tok_root,
            out_root=args.stage_dir,
            shard_bytes=args.shard_bytes,
            force=args.force,
        )
    elif not args.stage_dir.is_dir():
        raise SystemExit(f"--skip-stage but stage-dir missing: {args.stage_dir}")

    about = (
        "One-pass filtered instruct mix for OLMo-2 370M CE reference / rho-1: "
        "Tulu-v2, OpenHermes-2.5, Tulu-3, Hermes-3, SmolTalk, Dolci. Metadata drops "
        "for tools/safety/IF/Aya; Dolma English lang-id; dolma2-tokenizer (EOS 100257). "
        "Shards nested tokens/<source>/<domain>/ with doc-level 0.15% val holdout "
        "before tokenize (no token carve). No dedup; no upsampling."
    )
    notes = (
        "Validation: 0.15% of documents per (source, domain) reserved before tokenize "
        "(seed 42). No dedup. No upsampling. Realized size is one filtered pass."
    )
    limitations = [
        {
            "kind": "license",
            "detail": "Tulu ODC-BY with some NC subsets; research use",
        }
    ]

    dataset_manifest = {
        "dataset_id": args.dataset_id,
        "tokenizer": args.tokenizer,
        "purpose": args.purpose,
        "about": about,
        "notes": notes,
        "license": {"id": "ODC-By-1.0", "basis": "declared"},
        "limitations": limitations,
        "sources": build_sources(manifest),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.stage_dir / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "stage_dir": str(args.stage_dir),
                "dataset_id": args.dataset_id,
                "sources": len(dataset_manifest["sources"]),
                "total_tokens": manifest.get("total_stream_tokens_with_eos"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
