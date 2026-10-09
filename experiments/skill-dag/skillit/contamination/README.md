# Benchmark contamination audit -- Skill-It's time-varying exposure

Skill-It reweights domains mid-training (see `../README.md`'s "Skill-It
update"), so unlike the fixed-mixture arms in `../../mixlaw/contamination/`, a
single static weight vector does not describe either of its two arms'
exposure to contaminated eval text over a full run. This directory computes
that exposure honestly: per-segment, then as a token-weighted average over
the run.

**This directory does not re-scan the reservoir.** Both Skill-It arms train
on domain-stratified samples of the same `pretrain/olmo-127b` reservoir the
MixLaw arms do, against the same 40,582-item eval suite, so the per-domain
contamination rates are identical; only the mixture weights differ, and only
because they change over time here. `../../mixlaw/contamination/` holds the
scan, the eval-item index pipeline, and `results_olmo127b-reservoir.json`;
this directory reads that file directly rather than duplicating a ~5 MB
eval-item dump and a 172 MB index for a scan that would produce the same
numbers.

## The two arms

| Arm | Starting mixture | Adjacency |
| --- | --- | --- |
| Offline probe | LightGBM optimum (`LGB-min1pct`) | fixed offline `A` |
| Online derivative | LightGBM optimum (`LGB-min1pct`) | recomputed `A(r)` from MixLaw derivatives at each update |

Both arms start from the same mixture -- the LightGBM-optimized mixture --
and differ only in how the adjacency that drives each update is built.
Comparisons here are reported against two reference points:
`olmo-mix-1124` (the shared baseline `../../mixlaw/contamination/exposure_by_arm.json`
already uses) and `LGB-min1pct` itself, since that is the mixture both arms
actually start from and is also one of `exposure_by_arm.json`'s four rows.

## Method

Each arm's realized mixture is piecewise-constant: it holds fixed between
Skill-It's five update steps (500, 875, 1250, 1625, 2000), jumping to a new
mixture at each one, then holds again until the run ends at step 2384. Six
segments in total for a full run. For each segment,

    exposure(segment) = sum over domains of  weight(segment, d) * rate(d)

exactly as for a fixed mixture, using the domain-weighted rates from
`../../mixlaw/contamination/results_olmo127b-reservoir.json`. Segments are then
combined into a single run-level number by weighting each segment's exposure
by its length in steps -- token-weighted, since every step draws the same
number of tokens (a run's contaminated-word exposure in aggregate, not just
at any one point in it):

    exposure(run) = sum over segments of  length(segment) * exposure(segment)  /  total_steps

`trajectory_exposure.py` computes this from each arm's own
`skillit_updates_<arm>.jsonl` -- one row per Skill-It update step, `step`
and the realized post-update mixture `p_after`.

**Provenance of these six-row files.** Each is `step` and `p_after` read
directly off the raw per-step Skill-It update log (`skillit_updates.jsonl`)
written by that arm's own FarmShare training run --
`probe-lgbref-20261003` for offline probe,
`skillit-370m-deriv-20260916-124719` for online derivative -- not
transcribed from a table. Both logs' step-0 `p_after` matches
`LGB-min1pct`'s published weight vector in
`../../mixlaw/validation_mixtures_10b.json` to full float precision,
confirming both arms actually start from the LightGBM-optimized mixture.

## Results

All figures below are from `exposure_offline-probe.json` and
`exposure_online-derivative.json`. Both arms start at span rate 1.494e-05,
doc rate 6.545e-04 -- `LGB-min1pct`'s own exposure
(`../../mixlaw/contamination/README.md`'s per-arm table).

### Offline probe

| Step | Length (steps) | `weight[dclm]` | `weight[wiki]` | Span rate | Doc rate |
| --- | --- | --- | --- | --- | --- |
| 0 | 500 | 0.553 | 0.011 | 1.494e-05 | 6.545e-04 |
| 500 | 375 | 0.550 | 0.011 | 1.489e-05 | 6.530e-04 |
| 875 | 375 | 0.546 | 0.011 | 1.484e-05 | 6.515e-04 |
| 1250 | 375 | 0.540 | 0.011 | 1.476e-05 | 6.500e-04 |
| 1625 | 375 | 0.534 | 0.011 | 1.468e-05 | 6.485e-04 |
| 2000 | 384 | 0.526 | 0.011 | 1.459e-05 | 6.470e-04 |

**Time-weighted average: span rate 1.479e-05 (0.78x the `olmo-mix-1124`
baseline; 0.99x the LightGBM-optimized mixture it actually starts from),
doc rate 6.509e-04 (0.91x baseline; 0.99x its own start).**

### Online derivative

| Step | Length (steps) | `weight[dclm]` | `weight[wiki]` | Span rate | Doc rate |
| --- | --- | --- | --- | --- | --- |
| 0 | 500 | 0.553 | 0.011 | 1.494e-05 | 6.545e-04 |
| 500 | 375 | 0.556 | 0.016 | 1.536e-05 | 6.554e-04 |
| 875 | 375 | 0.557 | 0.021 | 1.583e-05 | 6.564e-04 |
| 1250 | 375 | 0.556 | 0.028 | 1.642e-05 | 6.579e-04 |
| 1625 | 375 | 0.553 | 0.037 | 1.712e-05 | 6.597e-04 |
| 2000 | 384 | 0.549 | 0.047 | 1.795e-05 | 6.621e-04 |

**Time-weighted average: span rate 1.621e-05 (0.86x baseline; 1.08x its own
start), doc rate 6.575e-04 (0.92x baseline; 1.00x its own start).**

### Reading these together

Both arms start at the identical LightGBM-optimized mixture but move
differently. Offline probe's sparse adjacency (8 nonzero cells) moves the mixture
only slowly: `dclm` weight drifts down from 0.553 to 0.526 by step 2000, `arxiv`
and `starcoder` fall, `open-web-math` (0.042 to 0.095) and `pes2o` (0.082 to
0.107) rise, and `wiki` stays at 0.011. Online derivative's recomputed adjacency
instead more than quadruples the `wiki` share (0.011 to 0.047) while leaving `dclm`
close to where it started (0.553 to 0.549).

The two arms end up on opposite sides of their own starting exposure. Offline
probe's exposure falls slightly (0.99x its start in both rates): `dclm`'s own
span rate (1.965e-05) sits above the LightGBM blend's average (1.494e-05), so
the small shift away from `dclm` lowers exposure. Online derivative's rise comes
from the opposite mechanism: `wiki`'s span rate (9.504e-05) is 4.8x `dclm`'s, so
even a modest reallocation onto it (1.1% to 4.7% of the mixture) is enough to
lift the time-weighted average while `dclm` barely moves.

Span rate and document rate agree in direction for the offline probe (both
fall slightly relative to its start), but for the online derivative the span
rate rises (1.08x its start) while the document rate is unchanged (1.00x), as in
the mixlaw arms in `../../mixlaw/contamination/README.md`, where a large
realized shift onto `wiki` specifically is what pulls the two metrics apart.
Neither arm here moves enough onto (or away from) `wiki` on its own to produce
a large split at the level of the time-weighted average.

This directory reports exposure only -- it does not compare either arm's
task-loss outcome, since a same-scale contamination comparison depends on
knowing what each arm's own results were, which belongs in `../README.md`
alongside this run's other metrics.

## Code map

- `trajectory_exposure.py` -- computes the per-segment and time-weighted
  average exposure described above. Stdlib only. Reads
  `../../mixlaw/contamination/results_olmo127b-reservoir.json` for per-domain
  rates; needs no other file from `../../mixlaw/`.
- `skillit_updates_offline-probe.jsonl`, `skillit_updates_online-derivative.jsonl`
  -- each arm's realized per-update-step mixture, `step` and `p_after` read
  directly off that arm's own FarmShare `skillit_updates.jsonl` run log (see
  Provenance above). Six rows each (steps 0, 500, 875, 1250, 1625, 2000).
- `exposure_offline-probe.json`, `exposure_online-derivative.json` --
  `trajectory_exposure.py`'s output for each arm; the source of every number
  in Results above.

For the underlying per-domain scan (the eval-item dump, index, scanner and
aggregator, and their own dependencies), see
[`../../mixlaw/contamination/README.md`](../../mixlaw/contamination/README.md).

## Reproducing

```bash
CONTAM=path/to/this/directory       # experiments/skill-dag/skillit/contamination
MIXLAW=path/to/mixlaw/contamination # experiments/skill-dag/mixlaw/contamination

# The per-domain scan itself is reproduced from $MIXLAW; see that
# directory's own README. Given results_olmo127b-reservoir.json there:

python "$CONTAM/trajectory_exposure.py" \
  --per-domain "$MIXLAW/results_olmo127b-reservoir.json" \
  --updates "$CONTAM/skillit_updates_offline-probe.jsonl" \
  --arm offline-probe --final-step 2384 \
  --out "$CONTAM/exposure_offline-probe.json"

python "$CONTAM/trajectory_exposure.py" \
  --per-domain "$MIXLAW/results_olmo127b-reservoir.json" \
  --updates "$CONTAM/skillit_updates_online-derivative.jsonl" \
  --arm online-derivative --final-step 2384 \
  --out "$CONTAM/exposure_online-derivative.json"

# The "vs. baseline" and "vs. its own start" ratios quoted in Results, from
# the two files above and $MIXLAW/exposure_by_arm.json.
python "$CONTAM/exposure_ratios.py"
```

To reproduce the `skillit_updates_<arm>.jsonl` files themselves from source,
pull the raw `skillit_updates.jsonl` progress log written by each arm's own
FarmShare run (`probe-lgbref-20261003` for offline probe,
`skillit-370m-deriv-20260916-124719` for online derivative) and extract each
row's `step` and `p_after` fields; the rest of that log (`A`, `losses`,
`p_before`, `r`, ...) is not needed here.
