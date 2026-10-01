#!/usr/bin/env python3
"""Add the fixed LightGBM static arm to a copied legacy Skill-It bundle.

It patches a copy of the FarmShare Skill-It bundle vendored in this repository
as ``olmo_core_skillit/``. This script modifies only a *new copy* of such
a bundle before it is submitted. It refuses unexpected source text so it never
silently patches a different training implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


STATIC_ARM = {
    "arm_index": 2,
    "arm_id": "static-lgbm-min1pct",
    "a_mode": "static",
    "wandb_project": "skillit",
}


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one patch target, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def patch_bundle(edullm_dir: Path) -> None:
    recipe_path = edullm_dir / "skillit_recipe.json"
    math_path = edullm_dir / "skillit_math.py"
    controller_path = edullm_dir / "skillit_controller.py"
    trainer_path = edullm_dir / "train_skillit_370m.py"
    for path in (recipe_path, math_path, controller_path, trainer_path):
        if not path.is_file():
            raise SystemExit(f"missing legacy Skill-It source file: {path}")

    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if [item.get("arm_index") for item in recipe.get("arms", [])] != [0, 1]:
        raise SystemExit("legacy recipe does not contain exactly the validated probe/deriv arms")
    if recipe.get("training", {}).get("seed") != 42:
        raise SystemExit("legacy recipe seed is not 42")
    if recipe.get("initial_weights") != [
        0.5528505096102141,
        0.21178482308348515,
        0.08723935826031831,
        0.08163337756774411,
        0.041786239404329344,
        0.013571059505826969,
        0.01113463256808213,
    ]:
        raise SystemExit("legacy recipe initial weights are not LGB-min1pct")
    recipe["arms"].append(STATIC_ARM)
    recipe.setdefault("methodology", {})["static_rerun"] = {
        "arm_id": STATIC_ARM["arm_id"],
        "stream_seed": 42,
        "updates": "disabled",
        "description": "fixed LightGBM 1%-floor mixture; no dynamic reweighting",
    }
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8", newline="\n")

    replace_once(
        math_path,
        'RECIPE_SHA256 = "28506e7c3e15814c1dd0082c1c3c52dd228083fec83556616019504583b48f91"',
        f'RECIPE_SHA256 = "{normalized_sha256(recipe_path)}"',
    )
    replace_once(
        math_path,
        "if tuple(arm.index for arm in ARMS) != (0, 1):\n"
        '    raise SkillItContractError("arm indexes must be exactly 0 and 1")',
        "if tuple(arm.index for arm in ARMS) != (0, 1, 2):\n"
        '    raise SkillItContractError("arm indexes must be exactly 0, 1, and 2")',
    )
    replace_once(
        math_path,
        '    ("deriv", "derivative", "skillit"),\n'
        "):\n"
        '    raise SkillItContractError("Skill-It arm definitions changed")',
        '    ("deriv", "derivative", "skillit"),\n'
        '    ("static-lgbm-min1pct", "static", "skillit"),\n'
        "):\n"
        '    raise SkillItContractError("Skill-It arm definitions changed")',
    )
    replace_once(
        math_path,
        "if ARMS[0].initial_weights is not None or ARMS[1].initial_weights is not None:\n"
        '    raise SkillItContractError("original Skill-It arms must use the baseline initial weights")',
        "if any(arm.initial_weights is not None for arm in ARMS):\n"
        '    raise SkillItContractError("Skill-It arms must use the baseline initial weights")',
    )
    replace_once(
        math_path,
        '    if a_mode == "derivative":\n'
        "        return derivative_a(weights)\n"
        '    raise ValueError(f"unknown a_mode={a_mode!r}")',
        '    if a_mode == "derivative":\n'
        "        return derivative_a(weights)\n"
        '    if a_mode == "static":\n'
        "        return np.zeros((len(DOMAINS), len(FAMILIES)), dtype=np.float64)\n"
        '    raise ValueError(f"unknown a_mode={a_mode!r}")',
    )
    replace_once(
        math_path,
        '        raise SkillItContractError("arm index must be 0 or 1") from exc',
        '        raise SkillItContractError("arm index must be 0, 1, or 2") from exc',
    )
    replace_once(
        math_path,
        '    raise SkillItContractError("arm index must be 0 (probe) or 1 (deriv)")',
        '    raise SkillItContractError("arm index must be 0 (probe), 1 (deriv), or 2 (static)")',
    )
    replace_once(
        controller_path,
        "    _applied_steps: set[int] = field(default_factory=set)\n\n"
        "    @property\n"
        "    def loader",
        "    _applied_steps: set[int] = field(default_factory=set)\n\n"
        "    def __post_init__(self) -> None:\n"
        '        if self.a_mode == "static":\n'
        "            self.update_steps = ()\n"
        '        elif self.a_mode not in ("probe", "derivative"):\n'
        '            raise SkillItControllerError(f"unknown a_mode={self.a_mode!r}")\n\n'
        "    @property\n"
        "    def loader",
    )
    replace_once(
        controller_path,
        '                        note="arm-specific starting domain weights; no Skill-It update",',
        '                        note=(\n'
        '                            "fixed LightGBM 1%-floor mixture; dynamic reweighting disabled"\n'
        '                            if self.a_mode == "static"\n'
        '                            else "arm-specific starting domain weights; no Skill-It update"\n'
        "                        ),",
    )
    replace_once(
        trainer_path,
        'parser.add_argument("--arm-index", type=int, choices=(0, 1), required=True)',
        'parser.add_argument("--arm-index", type=int, choices=(0, 1, 2), required=True)',
    )
    replace_once(
        trainer_path,
        '    if args.arm_index == 1 and "probe" in args.run_name.lower():\n'
        '        parser.error("arm-index 1 is deriv but run name says probe")\n'
        "    return args",
        '    if args.arm_index == 1 and "probe" in args.run_name.lower():\n'
        '        parser.error("arm-index 1 is deriv but run name says probe")\n'
        '    if args.arm_index == 2 and "static" not in args.run_name.lower():\n'
        '        parser.error("arm-index 2 is static but run name omits static")\n'
        "    return args",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edullm-dir",
        type=Path,
        required=True,
        help="Copied OLMo-core/.edullm directory to patch in place",
    )
    args = parser.parse_args()
    patch_bundle(args.edullm_dir)
    print(f"patched static LGB-min1pct arm in {args.edullm_dir}")


if __name__ == "__main__":
    main()
