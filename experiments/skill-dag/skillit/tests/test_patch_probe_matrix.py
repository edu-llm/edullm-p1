"""patch_probe_matrix.py swaps arm 0's offline matrix in a bundle copy and nothing else."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SKILLIT = Path(__file__).resolve().parents[1]
VENDORED = _SKILLIT / "olmo_core_skillit"
sys.path.insert(0, str(VENDORED / "farmshare"))

import patch_probe_matrix as ppm  # noqa: E402


def _bundle(tmp_path: Path) -> Path:
    """An LF copy of the vendored bundle files the patch touches."""
    edullm = tmp_path / ".edullm"
    edullm.mkdir()
    for name in ("skillit_recipe.json", "skillit_math.py"):
        (edullm / name).write_bytes((VENDORED / name).read_bytes().replace(b"\r\n", b"\n"))
    return edullm


def _a_json(tmp_path: Path) -> tuple[Path, np.ndarray]:
    A = np.zeros((7, 6))
    A[0, 1] = 0.25
    A[6, 5] = 0.5
    path = tmp_path / "A_offline.json"
    path.write_text(
        json.dumps({"domain_order": list(ppm.DOMAINS), "family_order": list(ppm.FAMILIES), "A": A.tolist()}),
        encoding="utf-8",
    )
    return path, A


def _import_math(edullm: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, edullm / "skillit_math.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # the module's dataclasses resolve annotations through it
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_matrix_is_swapped_and_the_trainer_guards_accept_it(tmp_path):
    edullm = _bundle(tmp_path)
    before = json.loads((edullm / "skillit_recipe.json").read_text(encoding="utf-8"))
    a_json, A = _a_json(tmp_path)

    recipe_sha, a_sha = ppm.patch_probe_matrix(edullm, a_json)

    math = _import_math(edullm, "skillit_math_patched_probe_test")  # runs load_recipe()'s checks
    assert np.array_equal(math.offline_a(), A)
    assert math.RECIPE_SHA256 == recipe_sha and math.OFFLINE_A_SOURCE_SHA256 == a_sha
    after = json.loads((edullm / "skillit_recipe.json").read_text(encoding="utf-8"))
    changed = {k for k in after["skillit"] if after["skillit"][k] != before["skillit"][k]}
    assert changed == {"offline_a"}
    assert after["arms"] == before["arms"]
    assert after["initial_weights"] == before["initial_weights"]
    assert after["methodology"]["offline_a_source_sha256"] == a_sha


def test_refuses_a_recipe_that_does_not_match_its_pin(tmp_path):
    edullm = _bundle(tmp_path)
    recipe = edullm / "skillit_recipe.json"
    recipe.write_text(recipe.read_text(encoding="utf-8") + " ", encoding="utf-8")
    a_json, _ = _a_json(tmp_path)
    with pytest.raises(SystemExit, match="RECIPE_SHA256"):
        ppm.patch_probe_matrix(edullm, a_json)


def test_refuses_a_bundle_without_the_probe_and_derivative_arms(tmp_path):
    edullm = _bundle(tmp_path)
    recipe_path = edullm / "skillit_recipe.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe["arms"][0]["a_mode"] = "derivative"
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8", newline="\n")
    # Re-pin the edited recipe so only the arm check can refuse it.
    math_path = edullm / "skillit_math.py"
    text = math_path.read_text(encoding="utf-8")
    math_path.write_text(
        text.replace(ppm.pinned(text, "RECIPE_SHA256"), ppm.normalized_sha256(recipe_path)),
        encoding="utf-8",
        newline="\n",
    )
    a_json, _ = _a_json(tmp_path)
    with pytest.raises(SystemExit, match="probe and derivative"):
        ppm.patch_probe_matrix(edullm, a_json)


def test_refuses_a_degenerate_matrix(tmp_path):
    edullm = _bundle(tmp_path)
    zero = tmp_path / "zero.json"
    zero.write_text(
        json.dumps(
            {"domain_order": list(ppm.DOMAINS), "family_order": list(ppm.FAMILIES), "A": np.zeros((7, 6)).tolist()}
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="all zeros"):
        ppm.patch_probe_matrix(edullm, zero)
