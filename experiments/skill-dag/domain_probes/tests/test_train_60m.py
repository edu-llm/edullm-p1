"""Unit tests for the pure parts of train_60m.py (no GPU, no OLMo install needed)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_60m as t  # noqa: E402


def test_registry_has_31_runs_one_hots_first():
    reg = t.load_registry()
    names = [r["name"] for r in reg["runs"]]
    assert len(names) == 31
    assert names[:7] == [f"onehot_{d}" for d in reg["domains"]]
    assert names[7:] == [f"mix{i:02d}" for i in range(1, 25)]
    assert not any("lgb" in n for n in names)
    for r in reg["runs"][:7]:
        assert sorted(r["weights"]) == [0.0] * 6 + [1.0]
    st = reg["settings"]
    assert st["steps"] == 1440 and st["eval_interval"] == 120
    assert all(lb.endswith("_test_rc_5shot_bpb") for lb in st["eval_labels"])
    assert set(st["expected_eval_items"]) == set(st["eval_labels"])


def test_pilot_weights_match_mixlaw_mixtures():
    reg = t.load_registry()
    mix = json.loads((Path(__file__).resolve().parents[2] / "mixlaw" / "mixtures.json").read_text())
    pilots = [r for r in reg["runs"] if r["kind"] == "pilot"]
    assert [p["weights"] for p in pilots] == [m["weights"] for m in mix["mixtures"]]


def test_schedule_is_10_80_10():
    assert t.schedule_steps(1440) == (144, 144)
    peak, floor = t.LEARNING_RATE, t.LEARNING_RATE * t.ALPHA_F
    assert t.lr_at(0, 1440) == 0.0
    assert t.lr_at(72, 1440) == pytest.approx(peak / 2)
    assert t.lr_at(144, 1440) == peak
    assert t.lr_at(1295, 1440) == peak
    assert t.lr_at(1296, 1440) == peak  # decay starts at 1296, cosine progress 0
    assert t.lr_at(1368, 1440) == pytest.approx((peak + floor) / 2)
    assert t.lr_at(1440, 1440) == pytest.approx(floor)
    lrs = [t.lr_at(s, 1440) for s in range(1296, 1441)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


def test_token_budget_is_exactly_1440_steps():
    assert 1440 * t.TOKENS_PER_STEP == 283_115_520
    assert 1440 * t.GLOBAL_BATCH_SEQS == 138_240


@pytest.mark.parametrize("weights", [[1, 0, 0, 0, 0, 0, 0], [0.375, 0.25, 0.1406, 0.0938, 0.0635, 0.0615, 0.0156]])
def test_allocation_sums_and_keeps_zeros(weights):
    counts = t.allocate_sequences(weights, 138_240)
    assert sum(counts) == 138_240
    for c, w in zip(counts, weights):
        assert (c == 0) == (w == 0)
        assert abs(c / 138_240 - w / sum(weights)) < 1e-4


def test_plan_has_no_repeats_and_matches_weights():
    weights = [0.5, 0.3, 0.2, 0.0]
    chunks = [10_000, 6_000, 4_000, 5_000]
    plan = t.build_plan(chunks, weights, 5_000, seed=6198)
    assert plan.shape == (5_000, 2)
    assert len({(int(d), int(c)) for d, c in plan}) == 5_000  # no sequence repeats in a run
    assert np.bincount(plan[:, 0], minlength=4).tolist() == [2500, 1500, 1000, 0]
    for di, n in enumerate(chunks[:3]):
        assert plan[plan[:, 0] == di][:, 1].max() < n
    # the draw spans each domain's whole stream rather than a prefix
    d0 = plan[plan[:, 0] == 0][:, 1]
    assert d0.max() > 0.9 * chunks[0] and d0.min() < 0.1 * chunks[0]
    assert np.array_equal(plan, t.build_plan(chunks, weights, 5_000, seed=6198))
    assert not np.array_equal(plan, t.build_plan(chunks, weights, 5_000, seed=1))


def test_plan_refuses_oversubscribed_domain():
    with pytest.raises(SystemExit):
        t.build_plan([100, 100], [1.0, 0.0], 500, seed=0)


def test_locate_and_read_across_shards(tmp_path):
    torch = pytest.importorskip("torch")
    shards = {}
    base = 0
    for i, n_chunks in enumerate((3, 2)):  # two shards of 3 and 2 chunks
        arr = np.arange(base, base + n_chunks * t.SEQ_LEN, dtype="<u4")
        p = tmp_path / f"train-0000{i}.u32le.bin"
        arr.tofile(p)
        shards.setdefault("a", []).append((str(p), n_chunks * t.SEQ_LEN))
        base += n_chunks * t.SEQ_LEN
    plan = np.array([[0, c] for c in (0, 2, 3, 4)], dtype=np.int64)
    ds = t.PlannedDataset(plan, ["a"], shards)
    assert len(ds) == 4
    for i, chunk in enumerate((0, 2, 3, 4)):
        ids = ds[i]["input_ids"]
        assert ids.dtype == torch.int64 and ids.shape == (t.SEQ_LEN,)
        assert int(ids[0]) == chunk * t.SEQ_LEN  # token values are their global positions


def test_shard_table_rejects_truncated_shard(tmp_path):
    manifest = {"objects": [{"name": "tokens/a/train-00000.u32le.bin", "source": "a",
                             "bytes": 4 * t.SEQ_LEN * 2, "tokens": t.SEQ_LEN * 2}]}
    (tmp_path / "a").mkdir()
    shard = tmp_path / "a" / "train-00000.u32le.bin"
    shard.write_bytes(b"\0" * 8)
    with pytest.raises(SystemExit):
        t.shard_table(tmp_path, manifest, ["a"])
    shard.write_bytes(b"\0" * (4 * t.SEQ_LEN * 2))
    assert t.shard_table(tmp_path, manifest, ["a"])["a"] == [(str(shard), t.SEQ_LEN * 2)]


LABELS = {"arc_easy_test_rc_5shot_bpb", "mmlu_stem_test_rc_5shot_bpb"}


def test_log_parser_ai2_olmo_060_split_records():
    step, loss, task = t.parse_trainer_log_record("[step=240/1440,epoch=1]", LABELS, None)
    assert step == 240 and not task
    step, _, task = t.parse_trainer_log_record(
        "eval/downstream_bpb/arc_easy_test_rc_5shot_bpb_bpb=2.0123", LABELS, 240)
    assert step == 240 and task == {"arc_easy_test_rc_5shot_bpb": 2.0123}
    _, _, task = t.parse_trainer_log_record("eval/downstream_bpb/boolq_val_rc_5shot_bpb_bpb=0.9", LABELS, 240)
    assert task == {}
    step, loss, task = t.parse_trainer_log_record(
        "[step=120/1440,epoch=1] train/CrossEntropyLoss=3.5 eval/x/mmlu_stem_test_rc_5shot_bpb_bpb=3.1", LABELS)
    assert (step, loss, task) == (120, 3.5, {"mmlu_stem_test_rc_5shot_bpb": 3.1})
