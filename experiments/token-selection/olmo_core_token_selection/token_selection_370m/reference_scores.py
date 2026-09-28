"""Shared schema for the offline per-instance reference-loss table.

RHO-1 and Perplexity both score against the same frozen reference
(``instruct-reference`` at step 940). Rather than keep a second copy of that
model resident during training, ``farmshare/score_reference.py`` scores the
whole RegMix corpus once, on the 1-GPU ``qos=normal`` lane, and writes one
row of per-token CE per corpus instance. Training then looks up rows by
``batch["index"]`` instead of running a second forward pass every step.

This works because selection is per 2048-token instance, and instance content
depends only on corpus paths/dtype/sequence-length -- never on world size,
rank, or microbatch composition (see ``token_selection_370m/selection.py``).
So one table indexed by instance id is valid at any world size.

Layout, at some root directory:
  - ``scores.f32``: a flat float32 binary, row-major, shape ``(rows, sequence_length)``.
    Row i is the reference's per-token CE for corpus instance i, exactly as
    OLMo's ``LMOutputWithLoss.ce_loss`` returns it (0.0 at ignored positions).
  - ``manifest.json``: binds the table to the exact reference checkpoint and
    corpus it was computed from, plus a per-chunk sha256 so a partial/corrupt
    write is detectable without re-scoring.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import Tensor

MANIFEST_SCHEMA_VERSION = 1
ALGORITHM = "reference-token-ce-v1"
# Row-write granularity for progress checkpointing and chunk hashing. Arbitrary
# but fixed: changing it invalidates previously recorded chunk_sha256 values.
CHUNK_ROWS = 65_536


def table_file(root: Path) -> Path:
    return Path(root) / "scores.f32"


def manifest_file(root: Path) -> Path:
    return Path(root) / "manifest.json"


def progress_file(root: Path) -> Path:
    return Path(root) / "progress.json"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_manifest(root: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_json(manifest_file(root), payload)


def write_progress(root: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_json(progress_file(root), payload)


def read_progress(root: Path) -> Optional[dict[str, Any]]:
    path = progress_file(root)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_manifest(root: Path) -> dict[str, Any]:
    payload = json.loads(manifest_file(root).read_text(encoding="utf-8"))
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported reference-score manifest schema at {root}")
    if payload.get("algorithm") != ALGORITHM:
        raise ValueError(f"unexpected reference-score algorithm at {root}: {payload.get('algorithm')!r}")
    return payload


def chunk_sha256(array: np.ndarray) -> str:
    """Hash one chunk's raw bytes, exactly as written to ``scores.f32``."""
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def open_table(root: Path, *, rows: int, sequence_length: int) -> np.memmap:
    return np.memmap(
        table_file(root), mode="r", dtype=np.float32, shape=(int(rows), int(sequence_length))
    )


class ReferenceScoreTable:
    """Read-only lookup into a materialized, manifest-verified score table.

    Validation happens once, at construction: the manifest must be
    ``complete``, must have been scored against the exact reference checkpoint
    the caller expects (by sha256), and must have been scored against the
    exact corpus the caller expects (by ``immutable_corpus_binding``). A
    mismatch on any of these fails closed rather than silently reading a
    table that doesn't correspond to this run.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        expected_reference_sha256: Optional[str],
        expected_corpus_binding: Mapping[str, Any],
    ) -> None:
        root = Path(root)
        manifest = read_manifest(root)
        if not manifest.get("complete"):
            raise ValueError(f"reference-score table at {root} is not complete")
        if manifest.get("reference_sha256") != expected_reference_sha256:
            raise ValueError(
                f"reference-score table at {root} was scored against a different reference "
                f"checkpoint (table={manifest.get('reference_sha256')!r}, "
                f"expected={expected_reference_sha256!r})"
            )
        bound = manifest.get("corpus_binding")
        if bound != dict(expected_corpus_binding):
            raise ValueError(f"reference-score table at {root} was scored against a different corpus")
        self.root = root
        self.manifest = manifest
        self._array = open_table(
            root, rows=int(manifest["rows"]), sequence_length=int(manifest["sequence_length"])
        )

    def gather(self, index: Tensor, *, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
        """Return a ``(len(index), sequence_length)`` tensor of reference CE rows."""
        idx = index.detach().to("cpu", dtype=torch.int64).numpy()
        rows = np.ascontiguousarray(self._array[idx])
        return torch.from_numpy(rows).to(device=device, dtype=dtype)
