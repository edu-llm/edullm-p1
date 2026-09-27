#!/usr/bin/env python3
"""Add the fixed-mixture static arms 2-6 to a copied Skill-It bundle.

It patches a *new copy* of the FarmShare Skill-It bundle vendored in this
repository as ``olmo_core_skillit/`` (the copy's ``OLMo-core/.edullm``). The
vendored files themselves are never edited.

This supersedes ``patch_legacy_static_lgbm_arm.py`` for new runs. That script
stays unchanged as the provenance of the static LightGBM rerun (Slurm job
1744338, W&B ``eduLLM/skillit/zgmte13g``). This one folds in that arm-2 patch,
so it runs once against a pristine copy and produces all of arms 0-6:

- arms 0 and 1 (probe, deriv) and arm 2 (``static-lgbm-min1pct``) exactly as
  they ran, with the default data seed 42 and W&B project ``skillit``;
- arms 3-6, static (no Skill-It updates), each with its own fixed domain
  weights, its own data seed and W&B project ``mixlaw-new``.

The arm 3-6 weight vectors are read at patch time from the repository's own
mixture files (``--mixlaw-dir``): ``validation_mixtures_10b.json`` for
olmo-mix-1124 and mix01 (the Data Mixing Laws mixture), and
``mixlaw_fit_chinchilla.json`` -> ``optimization.min1pct`` for the MixLaw arm.
They are then written into ``skillit_math.py`` as literal expected values, so
the trainer checks them without re-reading any file.

The script is fail-closed. It refuses a copy whose four target files do not
hash to the vendored originals, and every source string it replaces must occur
exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
DEFAULT_DATA_SEED = 42

# LF-normalized SHA-256 of the four target files as vendored (PROVENANCE.md).
PRISTINE_SHA256 = {
    "skillit_recipe.json": "28506e7c3e15814c1dd0082c1c3c52dd228083fec83556616019504583b48f91",
    "skillit_math.py": "f8d4c1b8792fa56bbbe0733d1f78942179fbd903dbfdd74cb34ba9e6ce94a186",
    "skillit_controller.py": "eb5c41b0dd41ecabaa18e3743240eb30d6440babd5c30f1abf9ca7bfde421590",
    "train_skillit_370m.py": "e112c2e42c57be4cc027d3552015b0071da84a69c86359a4e57c9d1617d1bb92",
}

# Identical to patch_legacy_static_lgbm_arm.STATIC_ARM (the zgmte13g rerun).
LGBM_STATIC_ARM = {
    "arm_index": 2,
    "arm_id": "static-lgbm-min1pct",
    "a_mode": "static",
    "wandb_project": "skillit",
}
LGBM_BASELINE_WEIGHTS = [
    0.5528505096102141,
    0.21178482308348515,
    0.08723935826031831,
    0.08163337756774411,
    0.041786239404329344,
    0.013571059505826969,
    0.01113463256808213,
]

VALIDATION_MIXTURES = "validation_mixtures_10b.json"
MIXLAW_FIT = "mixlaw_fit_chinchilla.json"
REPO_MIXLAW_DIR = "experiments/skill-dag/mixlaw"

# (arm_index, arm_id, data_seed, source file, selector, description)
VALIDATION_ARMS = (
    (3, "static-olmo-mix-1124-s42", 42, VALIDATION_MIXTURES, "olmo-mix-1124",
     "natural Olmo-mix-1124 mixture (control, data seed 42)"),
    (4, "static-olmo-mix-1124-s69", 69, VALIDATION_MIXTURES, "olmo-mix-1124",
     "natural Olmo-mix-1124 mixture (second control; only the data seed differs)"),
    (5, "static-mix01-s42", 42, VALIDATION_MIXTURES, "mix01",
     "Data Mixing Laws mixture"),
    (6, "static-ml-min1pct-s42", 42, MIXLAW_FIT, "min1pct",
     "MixLaw optimum under the 1% per-domain floor and 30% Wikipedia cap"),
)
WANDB_PROJECT = "mixlaw-new"
# The plan's table, rounded to 4 decimals: a guard against reading the wrong key.
MIXLAW_MIN1PCT_ROUNDED = (0.5557, 0.01, 0.01, 0.0916, 0.0227, 0.01, 0.30)


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one patch target, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def check_vector(label: str, weights: object) -> list[float]:
    if not isinstance(weights, list) or len(weights) != len(DOMAINS):
        raise SystemExit(f"{label}: expected {len(DOMAINS)} weights, got {weights!r}")
    vector = []
    for value in weights:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SystemExit(f"{label}: non-numeric weight {value!r}")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise SystemExit(f"{label}: invalid weight {value!r}")
        vector.append(value)
    if abs(math.fsum(vector) - 1.0) > 1e-9:
        raise SystemExit(f"{label}: weights sum to {math.fsum(vector)!r}, not 1")
    return vector


def read_weight_sources(mixlaw_dir: Path, derivative_fit_sha256: str) -> dict[int, dict]:
    mixtures_path = mixlaw_dir / VALIDATION_MIXTURES
    fit_path = mixlaw_dir / MIXLAW_FIT
    for path in (mixtures_path, fit_path):
        if not path.is_file():
            raise SystemExit(f"missing mixture source file: {path}")
    hashes = {
        VALIDATION_MIXTURES: normalized_sha256(mixtures_path),
        MIXLAW_FIT: normalized_sha256(fit_path),
    }
    # The MixLaw fit is the same file the recipe already pins for the derivative
    # arm. That pin was taken over a CRLF (Windows) checkout, so compare the CRLF
    # form; the recipe records the LF form, like PROVENANCE.md.
    fit_lf = fit_path.read_bytes().replace(b"\r\n", b"\n")
    fit_crlf_sha256 = hashlib.sha256(fit_lf.replace(b"\n", b"\r\n")).hexdigest()
    if fit_crlf_sha256 != derivative_fit_sha256:
        raise SystemExit(
            f"{fit_path}: CRLF sha256 {fit_crlf_sha256} is not the recipe's pinned "
            f"derivative_fit_source_sha256 {derivative_fit_sha256}"
        )

    mixtures = json.loads(mixtures_path.read_text(encoding="utf-8"))
    if tuple(mixtures.get("domain_order") or ()) != DOMAINS:
        raise SystemExit(f"{mixtures_path}: domain order differs from the recipe")
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    if tuple(fit.get("domain_order") or ()) != DOMAINS:
        raise SystemExit(f"{fit_path}: domain order differs from the recipe")

    out: dict[int, dict] = {}
    for arm_index, arm_id, seed, source, selector, description in VALIDATION_ARMS:
        if source == VALIDATION_MIXTURES:
            matches = [m for m in mixtures.get("mixtures", []) if m.get("run_name") == selector]
            if len(matches) != 1:
                raise SystemExit(f"{mixtures_path}: expected one mixture {selector!r}, got {len(matches)}")
            weights = check_vector(f"{source}:{selector}", matches[0].get("weights"))
            key = f"mixtures[run_name={selector}].weights"
        else:
            named = ((fit.get("optimization") or {}).get(selector) or {}).get("weights")
            if not isinstance(named, dict) or tuple(named) != DOMAINS:
                raise SystemExit(f"{fit_path}: optimization.{selector}.weights is not keyed by DOMAINS")
            weights = check_vector(f"{source}:{selector}", [named[d] for d in DOMAINS])
            if tuple(round(w, 4) for w in weights) != MIXLAW_MIN1PCT_ROUNDED:
                raise SystemExit(f"{fit_path}: optimization.{selector} is not the planned 1%-floor optimum")
            key = f"optimization.{selector}.weights"
        out[arm_index] = {
            "arm_id": arm_id,
            "data_seed": seed,
            "weights": weights,
            "file": f"{REPO_MIXLAW_DIR}/{source}",
            "key": key,
            "sha256": hashes[source],
            "description": description,
        }
    return out


def float_tuple_literal(values: list[float]) -> str:
    # repr() round-trips every float64 exactly.
    return "(" + ", ".join(repr(float(v)) for v in values) + ")"


def patch_recipe(recipe_path: Path, sources: dict[int, dict]) -> dict:
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if [item.get("arm_index") for item in recipe.get("arms", [])] != [0, 1]:
        raise SystemExit("legacy recipe does not contain exactly the validated probe/deriv arms")
    if recipe.get("training", {}).get("seed") != 42:
        raise SystemExit("legacy recipe seed is not 42")
    if recipe.get("initial_weights") != LGBM_BASELINE_WEIGHTS:
        raise SystemExit("legacy recipe initial weights are not LGB-min1pct")
    if "static_rerun" in recipe.get("methodology", {}) or "static_validation" in recipe.get(
        "methodology", {}
    ):
        raise SystemExit("recipe already carries a static-arm block")

    # Arm 2 exactly as patch_legacy_static_lgbm_arm.py added it.
    recipe["arms"].append(dict(LGBM_STATIC_ARM))
    recipe.setdefault("methodology", {})["static_rerun"] = {
        "arm_id": LGBM_STATIC_ARM["arm_id"],
        "stream_seed": 42,
        "updates": "disabled",
        "description": "fixed LightGBM 1%-floor mixture; no dynamic reweighting",
    }
    for arm_index in sorted(sources):
        src = sources[arm_index]
        recipe["arms"].append(
            {
                "arm_index": arm_index,
                "arm_id": src["arm_id"],
                "a_mode": "static",
                "wandb_project": WANDB_PROJECT,
                "initial_weights": src["weights"],
                "initial_weights_source": f"{src['file']}: {src['key']} (LF sha256 {src['sha256']})",
                "data_seed": src["data_seed"],
            }
        )
    recipe["methodology"]["static_validation"] = {
        "arm_ids": [sources[i]["arm_id"] for i in sorted(sources)],
        "updates": "disabled",
        "wandb_project": WANDB_PROJECT,
        "default_data_seed": DEFAULT_DATA_SEED,
        "data_seed_use": (
            "WeightedDomainDataLoader(seed=data_seed): batch b draws from "
            "numpy.random.default_rng(data_seed + 1_000_003 * b); model init is unchanged "
            "(seed_all(42), init_seed 0)"
        ),
        "weight_sources": {
            sources[i]["arm_id"]: {
                "file": sources[i]["file"],
                "key": sources[i]["key"],
                "sha256_lf": sources[i]["sha256"],
                "description": sources[i]["description"],
            }
            for i in sorted(sources)
        },
        "description": "fixed-mixture static arms; no dynamic reweighting",
    }
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8", newline="\n")
    return recipe


def patch_math(math_path: Path, recipe_sha256: str, sources: dict[int, dict]) -> None:
    replace_once(
        math_path,
        'RECIPE_SHA256 = "28506e7c3e15814c1dd0082c1c3c52dd228083fec83556616019504583b48f91"',
        f'RECIPE_SHA256 = "{recipe_sha256}"',
    )
    replace_once(
        math_path,
        "    wandb_project: str\n"
        "    initial_weights: tuple[float, ...] | None = None\n",
        "    wandb_project: str\n"
        "    initial_weights: tuple[float, ...] | None = None\n"
        "    data_seed: int | None = None\n",
    )
    replace_once(
        math_path,
        '            if "initial_weights" in item\n'
        "            else None\n"
        "        ),\n"
        "    )\n"
        '    for item in RECIPE["arms"]\n'
        ")\n",
        '            if "initial_weights" in item\n'
        "            else None\n"
        "        ),\n"
        '        data_seed=int(item["data_seed"]) if "data_seed" in item else None,\n'
        "    )\n"
        '    for item in RECIPE["arms"]\n'
        ")\n",
    )
    replace_once(
        math_path,
        "if tuple(arm.index for arm in ARMS) != (0, 1):\n"
        '    raise SkillItContractError("arm indexes must be exactly 0 and 1")',
        "if tuple(arm.index for arm in ARMS) != (0, 1, 2, 3, 4, 5, 6):\n"
        '    raise SkillItContractError("arm indexes must be exactly 0 through 6")',
    )
    definitions = "".join(
        f'    ("{sources[i]["arm_id"]}", "static", "{WANDB_PROJECT}"),\n' for i in sorted(sources)
    )
    replace_once(
        math_path,
        '    ("deriv", "derivative", "skillit"),\n'
        "):\n"
        '    raise SkillItContractError("Skill-It arm definitions changed")',
        '    ("deriv", "derivative", "skillit"),\n'
        '    ("static-lgbm-min1pct", "static", "skillit"),\n'
        f"{definitions}"
        "):\n"
        '    raise SkillItContractError("Skill-It arm definitions changed")',
    )
    expected = "".join(
        f"    {i}: (\n"
        f"        {float_tuple_literal(sources[i]['weights'])},\n"
        f"        {int(sources[i]['data_seed'])},\n"
        "    ),\n"
        for i in sorted(sources)
    )
    replace_once(
        math_path,
        "if ARMS[0].initial_weights is not None or ARMS[1].initial_weights is not None:\n"
        '    raise SkillItContractError("original Skill-It arms must use the baseline initial weights")\n',
        "if ARMS[0].initial_weights is not None or ARMS[1].initial_weights is not None:\n"
        '    raise SkillItContractError("original Skill-It arms must use the baseline initial weights")\n'
        "if ARMS[2].initial_weights is not None:\n"
        '    raise SkillItContractError("static LightGBM arm must use the baseline initial weights")\n'
        "if any(arm.data_seed is not None for arm in ARMS[:3]):\n"
        '    raise SkillItContractError("arms 0-2 must use the default data seed")\n'
        "# Static validation arms: exact weights and data seeds, written by\n"
        "# farmshare/patch_static_validation_arms.py from the repository's mixture files.\n"
        "STATIC_VALIDATION_ARMS = {\n"
        f"{expected}"
        "}\n"
        "for _index, (_weights, _seed) in STATIC_VALIDATION_ARMS.items():\n"
        "    _raw_seed = RECIPE[\"arms\"][_index].get(\"data_seed\")\n"
        "    if ARMS[_index].initial_weights != _weights:\n"
        '        raise SkillItContractError(f"arm {_index}: initial weights changed")\n'
        "    if type(_raw_seed) is not int or ARMS[_index].data_seed != _seed:\n"
        '        raise SkillItContractError(f"arm {_index}: data seed changed")\n'
        "    if abs(math.fsum(_weights) - 1.0) > 1e-9 or min(_weights) < 0:\n"
        '        raise SkillItContractError(f"arm {_index}: weights are not a distribution")\n',
    )
    replace_once(
        math_path,
        "    return weights / weights.sum()\n"
        "\n"
        "\n"
        "def offline_a() -> np.ndarray:\n",
        "    return weights / weights.sum()\n"
        "\n"
        "\n"
        f"DEFAULT_DATA_SEED = {DEFAULT_DATA_SEED}\n"
        "\n"
        "\n"
        "def data_seed(arm_index: int) -> int:\n"
        '    """The loader stream seed for one arm; arms without their own use 42."""\n'
        "    seed = arm_by_index(arm_index).data_seed\n"
        "    return DEFAULT_DATA_SEED if seed is None else int(seed)\n"
        "\n"
        "\n"
        "def offline_a() -> np.ndarray:\n",
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
        '        raise SkillItContractError("arm index must be an integer from 0 to 6") from exc',
    )
    replace_once(
        math_path,
        '    raise SkillItContractError("arm index must be 0 (probe) or 1 (deriv)")',
        '    raise SkillItContractError(\n'
        '        "arm index must be 0 (probe), 1 (deriv), 2 (static LightGBM) or 3-6 (static validation)"\n'
        "    )",
    )
    replace_once(
        math_path,
        '    "arm_by_index",\n'
        '    "derivative_a",\n',
        '    "arm_by_index",\n'
        '    "data_seed",\n'
        '    "derivative_a",\n',
    )


def patch_controller(controller_path: Path) -> None:
    # The static a_mode needs no update steps; the check is generic per a_mode.
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
    # Arm 2 keeps its rerun note; arms 3-6 must not claim to be LightGBM.
    replace_once(
        controller_path,
        '                        note="arm-specific starting domain weights; no Skill-It update",',
        '                        note=(\n'
        '                            "fixed LightGBM 1%-floor mixture; dynamic reweighting disabled"\n'
        '                            if self.arm_id == "static-lgbm-min1pct"\n'
        '                            else "fixed static mixture; dynamic reweighting disabled"\n'
        '                            if self.a_mode == "static"\n'
        '                            else "arm-specific starting domain weights; no Skill-It update"\n'
        "                        ),",
    )


def patch_trainer(trainer_path: Path) -> None:
    replace_once(
        trainer_path,
        "import argparse\n"
        "from pathlib import Path\n"
        "from typing import Any, Optional, Sequence, cast\n"
        "\n"
        "import torch.distributed as dist\n",
        "import argparse\n"
        "import importlib\n"
        "import inspect\n"
        "from pathlib import Path\n"
        "from typing import Any, Optional, Sequence, cast\n"
        "\n"
        "import numpy as np\n"
        "import torch.distributed as dist\n",
    )
    replace_once(
        trainer_path,
        "from skillit_math import RECIPE, arm_by_index, initial_weights\n"
        "\n"
        "RANK_MICROBATCH_TOKENS = 32_768\n"
        "CHECKPOINT_INTERVAL = 125\n",
        "from skillit_math import RECIPE, arm_by_index, data_seed, initial_weights\n"
        "\n"
        "RANK_MICROBATCH_TOKENS = 32_768\n"
        "CHECKPOINT_INTERVAL = 125\n"
        "\n"
        "\n"
        "def _install_wandb_finish_shim() -> None:\n"
        '    """Call ``wandb.finish(exit_code=...)`` without the ``quiet`` keyword.\n'
        "\n"
        "    OLMo-core's ``WandBCallback.finalize`` passes ``quiet=True``, which the\n"
        "    installed wandb rejects with a TypeError after the last eval. That error\n"
        "    marked the derivative run and the static LightGBM rerun failed. The\n"
        "    replacement is otherwise identical and runs after training ends.\n"
        '    """\n'
        '    module = importlib.import_module("olmo_core.train.callbacks.wandb")\n'
        "    if module.WandBCallback is not WandBCallback:\n"
        '        raise RuntimeError("unexpected WandBCallback class")\n'
        "    original = WandBCallback.finalize\n"
        '    if getattr(original, "_edullm_finish_shim", False):\n'
        "        return\n"
        '    if "self.wandb.finish(exit_code=exit_code, quiet=True)" not in inspect.getsource(original):\n'
        '        raise RuntimeError("unexpected WandBCallback.finalize; refusing to replace it")\n'
        "\n"
        "    def finalize(self: WandBCallback, exit_code: int = 0) -> None:\n"
        "        if not self.finalized:\n"
        "            if exit_code > 0:\n"
        '                module.log.warning("Finalizing failed W&B run...")\n'
        "            else:\n"
        '                module.log.info("Finalizing successful W&B run...")\n'
        "            self.wandb.finish(exit_code=exit_code)\n"
        "            self._finalized = True\n"
        "\n"
        "    finalize._edullm_finish_shim = True  # type: ignore[attr-defined]\n"
        "    WandBCallback.finalize = finalize  # type: ignore[method-assign]\n"
        "\n"
        "\n"
        "_install_wandb_finish_shim()\n",
    )
    replace_once(
        trainer_path,
        '                "initial_weights": initial_weights(arm.index).tolist(),\n'
        '                "methodology": RECIPE,\n',
        '                "initial_weights": initial_weights(arm.index).tolist(),\n'
        '                "data_seed": data_seed(arm.index),\n'
        '                "methodology": RECIPE,\n',
    )
    replace_once(
        trainer_path,
        '        "initial_weights": initial_weights(arm.index).tolist(),\n'
        '        "recipe": RECIPE,\n',
        '        "initial_weights": initial_weights(arm.index).tolist(),\n'
        '        "data_seed": data_seed(arm.index),\n'
        '        "recipe": RECIPE,\n',
    )
    replace_once(
        trainer_path,
        "        weights=initial_weights(arm.index),\n"
        "        dp_world_size=get_world_size(train_module.dp_process_group),\n"
        "        dp_rank=get_rank(train_module.dp_process_group),\n"
        "        fs_local_rank=get_fs_local_rank(),\n"
        "    )\n",
        "        weights=initial_weights(arm.index),\n"
        "        seed=data_seed(arm.index),\n"
        "        dp_world_size=get_world_size(train_module.dp_process_group),\n"
        "        dp_rank=get_rank(train_module.dp_process_group),\n"
        "        fs_local_rank=get_fs_local_rank(),\n"
        "    )\n"
        "    if loader.seed != data_seed(arm.index):\n"
        '        raise RuntimeError(f"loader seed {loader.seed} is not the recipe data seed")\n'
        "    if arm.index <= 2 and loader.seed != SEED:\n"
        '        raise RuntimeError("arms 0-2 must keep the original loader seed")\n'
        "    if np.max(np.abs(loader.weights - initial_weights(arm.index))) > 1e-15:\n"
        '        raise RuntimeError("loader weights differ from the recipe initial weights")\n',
    )
    replace_once(
        trainer_path,
        'parser.add_argument("--arm-index", type=int, choices=(0, 1), required=True)',
        'parser.add_argument("--arm-index", type=int, choices=(0, 1, 2, 3, 4, 5, 6), required=True)',
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
        "    if args.arm_index >= 3:\n"
        "        arm_id = arm_by_index(args.arm_index).arm_id\n"
        '        if "static" not in args.run_name.lower() or arm_id not in args.run_name.lower():\n'
        "            parser.error(\n"
        '                f"arm-index {args.arm_index} is {arm_id} but run name omits static or {arm_id}"\n'
        "            )\n"
        "    return args",
    )


def patch_bundle(edullm_dir: Path, mixlaw_dir: Path) -> None:
    paths = {name: edullm_dir / name for name in PRISTINE_SHA256}
    for name, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"missing legacy Skill-It source file: {path}")
        got = normalized_sha256(path)
        if got != PRISTINE_SHA256[name]:
            raise SystemExit(f"{path}: sha256 {got} is not the vendored original; refusing to patch")

    pristine_recipe = json.loads(paths["skillit_recipe.json"].read_text(encoding="utf-8"))
    derivative_fit_sha256 = pristine_recipe["methodology"]["derivative_fit_source_sha256"]
    sources = read_weight_sources(mixlaw_dir, derivative_fit_sha256)

    recipe = patch_recipe(paths["skillit_recipe.json"], sources)
    recipe_sha256 = normalized_sha256(paths["skillit_recipe.json"])
    patch_math(paths["skillit_math.py"], recipe_sha256, sources)
    patch_controller(paths["skillit_controller.py"])
    patch_trainer(paths["train_skillit_370m.py"])

    print(f"patched static arms 2-6 in {edullm_dir}")
    print(f"RECIPE_SHA256 = {recipe_sha256}")
    for item in recipe["arms"]:
        weights = item.get("initial_weights")
        print(
            f"arm {item['arm_index']}: id={item['arm_id']} a_mode={item['a_mode']} "
            f"wandb_project={item['wandb_project']} "
            f"data_seed={item.get('data_seed', DEFAULT_DATA_SEED)} "
            f"weights={'recipe baseline (LGB-min1pct)' if weights is None else weights}"
        )
        if "initial_weights_source" in item:
            print(f"    source: {item['initial_weights_source']}")
    for name, path in paths.items():
        print(f"{normalized_sha256(path)}  {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edullm-dir",
        type=Path,
        required=True,
        help="Copied OLMo-core/.edullm directory to patch in place",
    )
    parser.add_argument(
        "--mixlaw-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "mixlaw",
        help="Directory holding validation_mixtures_10b.json and mixlaw_fit_chinchilla.json "
        "(default: this repository's experiments/skill-dag/mixlaw)",
    )
    args = parser.parse_args()
    patch_bundle(args.edullm_dir, args.mixlaw_dir)


if __name__ == "__main__":
    main()
