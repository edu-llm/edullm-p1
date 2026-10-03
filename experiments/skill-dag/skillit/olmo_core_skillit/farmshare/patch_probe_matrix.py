#!/usr/bin/env python3
"""Swap the offline matrix the Skill-It probe arm (arm 0) reads in a copied bundle.

Run it on a *new copy* of a bundle (for the reported rerun: the bundle that
``patch_static_validation_arms.py`` patched to arms 0-6). It replaces ``skillit.offline_a`` in the copy's recipe with the
``A`` of a rebuilt ``A_offline.json`` (``build_adjacency.py --reference-run
probe_lgb_start``), re-pins that file's hash as the recipe's ``offline_a_source_sha256``,
and re-pins the recipe's own hash in ``skillit_math.py`` so the trainer's checksum guards
accept it. Nothing else changes: the derivative arm, the static arms, the trainer and the
controller are untouched, and arm 0 still starts from the LightGBM mixture.

The script is fail-closed. The copy's recipe must hash to the ``RECIPE_SHA256`` its own
``skillit_math.py`` pins, arms 0 and 1 must be the probe and derivative arms, and every
constant it rewrites must occur exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

DOMAINS = (
    "dclm",
    "arxiv",
    "starcoder",
    "pes2o",
    "open-web-math",
    "algebraic-stack",
    "wiki",
)
FAMILIES = (
    "arc_challenge",
    "arc_easy",
    "mmlu_humanities",
    "mmlu_other",
    "mmlu_social_sciences",
    "mmlu_stem",
)
REFERENCE_NOTE = (
    "offline A rebuilt with the 60M probe trained on the LightGBM starting mixture "
    "(probe_lgb_start) as the reference; all eight probes read on the same 128 eval items"
)


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def pinned(text: str, name: str) -> str:
    found = re.findall(rf'^{name} = "([0-9a-f]{{64}})"$', text, flags=re.MULTILINE)
    if len(found) != 1:
        raise SystemExit(f"skillit_math.py: expected one {name} pin, found {len(found)}")
    return found[0]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one patch target, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def read_matrix(a_json: Path) -> list[list[float]]:
    payload = json.loads(a_json.read_text(encoding="utf-8"))
    if tuple(payload.get("domain_order") or ()) != DOMAINS:
        raise SystemExit(f"{a_json}: domain order differs from the recipe")
    if tuple(payload.get("family_order") or ()) != FAMILIES:
        raise SystemExit(f"{a_json}: family order differs from the recipe")
    matrix = payload.get("A")
    if (
        not isinstance(matrix, list)
        or len(matrix) != len(DOMAINS)
        or any(not isinstance(row, list) or len(row) != len(FAMILIES) for row in matrix)
    ):
        raise SystemExit(f"{a_json}: A is not {len(DOMAINS)}x{len(FAMILIES)}")
    out = []
    for row in matrix:
        values = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SystemExit(f"{a_json}: non-numeric entry {value!r}")
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise SystemExit(f"{a_json}: entry {value!r} is not a finite non-negative number")
            values.append(value)
        out.append(values)
    if not any(v > 0 for row in out for v in row):
        raise SystemExit(f"{a_json}: the matrix is all zeros")
    return out


def patch_probe_matrix(edullm_dir: Path, a_json: Path) -> tuple[str, str]:
    recipe_path = edullm_dir / "skillit_recipe.json"
    math_path = edullm_dir / "skillit_math.py"
    for path in (recipe_path, math_path):
        if not path.is_file():
            raise SystemExit(f"missing bundle file: {path}")

    math_text = math_path.read_text(encoding="utf-8")
    old_recipe_sha = pinned(math_text, "RECIPE_SHA256")
    old_a_sha = pinned(math_text, "OFFLINE_A_SOURCE_SHA256")
    if normalized_sha256(recipe_path) != old_recipe_sha:
        raise SystemExit("recipe does not hash to the RECIPE_SHA256 its skillit_math.py pins")

    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    arms = recipe.get("arms") or []
    if [item.get("a_mode") for item in arms[:2]] != ["probe", "derivative"]:
        raise SystemExit("arms 0 and 1 are not the probe and derivative arms")
    if recipe["methodology"].get("offline_a_source_sha256") != old_a_sha:
        raise SystemExit("recipe and skillit_math.py disagree on the offline A hash")

    matrix = read_matrix(a_json)
    new_a_sha = normalized_sha256(a_json)
    recipe["skillit"]["offline_a"] = matrix
    recipe["methodology"]["offline_a_source_sha256"] = new_a_sha
    recipe["methodology"]["offline_a_reference"] = REFERENCE_NOTE
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8", newline="\n")
    new_recipe_sha = normalized_sha256(recipe_path)

    replace_once(math_path, f'RECIPE_SHA256 = "{old_recipe_sha}"', f'RECIPE_SHA256 = "{new_recipe_sha}"')
    replace_once(
        math_path,
        f'OFFLINE_A_SOURCE_SHA256 = "{old_a_sha}"',
        f'OFFLINE_A_SOURCE_SHA256 = "{new_a_sha}"',
    )
    return new_recipe_sha, new_a_sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edullm-dir", type=Path, required=True, help="copied OLMo-core/.edullm to patch")
    parser.add_argument("--a-offline-json", type=Path, required=True, help="rebuilt A_offline.json")
    args = parser.parse_args()
    recipe_sha, a_sha = patch_probe_matrix(args.edullm_dir, args.a_offline_json)
    print(f"patched offline A in {args.edullm_dir}")
    print(f"OFFLINE_A_SOURCE_SHA256 = {a_sha}")
    print(f"RECIPE_SHA256 = {recipe_sha}")


if __name__ == "__main__":
    main()
