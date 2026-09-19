# Benchmark contamination audit -- Skill-It's time-varying exposure

Skill-It reweights domains mid-training (see the top-level `README.md`'s
"Skill-It update"), so unlike the fixed-mixture arms in
`../mixlaw/contamination/`, a single static weight vector does not describe
either arm's exposure to contaminated eval text over a full run. This
directory computes that exposure honestly: per-segment, then as a
token-weighted average over the run.

**This directory does not re-scan the reservoir.** Both Skill-It arms train
on domain-stratified samples of the same `pretrain/olmo-127b` reservoir the
MixLaw arms do, against the same 40,582-item eval suite, so the per-domain
contamination rates are identical; only the mixture weights differ, and only
because they change over time here. `../mixlaw/contamination/` holds the
scan, the eval-item index pipeline, and `results_olmo127b-reservoir.json`;
this directory reads that file directly rather than duplicating a ~10 MB
eval-item dump and a 172 MB index for a scan that would produce the same
numbers.

## Method

Each arm's realized mixture is piecewise-constant: it holds fixed between
Skill-It's five update steps (500, 875, 1250, 1625, 2000), jumping to a new
mixture at each one, then holds again until the run ends at step 2384. Six
segments in total for a full run. For each segment,

    exposure(segment) = sum over domains of  weight(segment, d) * rate(d)

exactly as for a fixed mixture, using the domain-weighted rates from
`../mixlaw/contamination/results_olmo127b-reservoir.json`. Segments are then
combined into a single run-level number by weighting each segment's exposure
by its length in steps -- token-weighted, since every step draws the same
number of tokens (a run's contaminated-word exposure in aggregate, not just
at any one point in it):

    exposure(run) = sum over segments of  length(segment) * exposure(segment)  /  total_steps

`trajectory_exposure.py` computes this from each arm's own
`skillit_updates_<arm>.jsonl` -- the realized per-step mixture Skill-It's
training loop logged for that specific run (schema shared with
`../plot_weights.py`), not a value re-derived from `skillit_train_recipe.json`.
That distinction matters here: both arms' realized step-0 mixture is the
`LGB-min1pct` weights (see `../mixlaw/validation_mixtures_10b.json`), not the
`initial_weights` field in `../skillit_train_recipe.json`, which records a
different, generic starting point than what these specific runs were
actually launched from.

## Results

Both arms start at step 0 from identical weights (`LGB-min1pct`,
matched-span word rate 1.494e-05 -- see the shared per-arm table in
`../mixlaw/contamination/README.md`) and diverge under their own update
rule, for two different reasons. In `probe`, `weight[dclm]` rises
monotonically and substantially (0.553 -> 0.864) while `weight[wiki]` stays
roughly flat; since `dclm`'s own matched-span word rate (1.965e-05) sits
above the run's starting exposure, `probe`'s span rate climbs mainly because
it concentrates onto `dclm`. In `deriv`, `weight[dclm]` is not monotonic --
it rises slightly then falls back close to its starting value -- while
`weight[wiki]` rises steadily and by a large relative factor (0.011 ->
0.047); since `wiki` carries by far the highest per-domain rate
(9.504e-05), `deriv`'s span rate climbs for a different reason than
`probe`'s: growing (if still small) exposure to `wiki` rather than
concentration onto `dclm`.

### `skillit-deriv` (online mixing-law A(r))

From `exposure_deriv.json`.

| Step | Length (steps) | `weight[dclm]` | `weight[wiki]` | Span rate | Doc rate |
| --- | --- | --- | --- | --- | --- |
| 0 | 500 | 0.553 | 0.011 | 1.494e-05 | 6.545e-04 |
| 500 | 375 | 0.556 | 0.016 | 1.536e-05 | 6.554e-04 |
| 875 | 375 | 0.557 | 0.021 | 1.583e-05 | 6.564e-04 |
| 1250 | 375 | 0.556 | 0.028 | 1.642e-05 | 6.579e-04 |
| 1625 | 375 | 0.553 | 0.037 | 1.712e-05 | 6.597e-04 |
| 2000 | 384 | 0.549 | 0.047 | 1.795e-05 | 6.621e-04 |

**Time-weighted average: span rate 1.621e-05 (0.855x the `olmo-mix-1124`
baseline), doc rate 6.575e-04 (0.921x baseline).**

### `skillit-probe` (fixed offline A)

From `exposure_probe.json`.

| Step | Length (steps) | `weight[dclm]` | `weight[wiki]` | Span rate | Doc rate |
| --- | --- | --- | --- | --- | --- |
| 0 | 500 | 0.553 | 0.011 | 1.494e-05 | 6.545e-04 |
| 500 | 375 | 0.646 | 0.011 | 1.610e-05 | 6.748e-04 |
| 875 | 375 | 0.720 | 0.011 | 1.698e-05 | 6.889e-04 |
| 1250 | 375 | 0.779 | 0.010 | 1.765e-05 | 6.991e-04 |
| 1625 | 375 | 0.827 | 0.009 | 1.817e-05 | 7.067e-04 |
| 2000 | 384 | 0.864 | 0.008 | 1.856e-05 | 7.123e-04 |

**Time-weighted average: span rate 1.696e-05 (0.895x the `olmo-mix-1124`
baseline), doc rate 6.877e-04 (0.963x baseline).**

The `probe` arm's fixed adjacency routes weight toward `dclm` far more
aggressively than `deriv`'s online one (`weight[dclm]` reaches 0.864 by step
2000 vs. `deriv`'s 0.549), and `wiki` correspondingly shrinks rather than
grows -- so despite ending up more `dclm`-concentrated, `probe`'s exposure
stays close to `deriv`'s and to the shared starting point, because `dclm`'s
own rate is unremarkable next to `wiki`'s. Both time-weighted averages fall
within the range the four validated MixLaw-experiment arms span in
`../mixlaw/contamination/README.md` (1.303e-05 to 4.016e-05), below the
natural baseline and closest to `LGB-min1pct` (LightGBM) -- the exact
mixture both Skill-It arms are seeded from.

## Code map

- `trajectory_exposure.py` -- computes the per-segment and time-weighted
  average exposure described above. Stdlib only. Reads
  `../mixlaw/contamination/results_olmo127b-reservoir.json` for per-domain
  rates; needs no other file from `../mixlaw/`.
- `skillit_updates_deriv.jsonl`, `skillit_updates_probe.jsonl` -- each arm's
  realized per-update-step mixture, as logged by Skill-It's training loop.
  Copied unmodified from that run's own progress log; six rows each
  (steps 0, 500, 875, 1250, 1625, 2000), the update schedule in
  `../skillit_train_recipe.json` plus the run's starting weights at step 0.
- `exposure_deriv.json`, `exposure_probe.json` -- `trajectory_exposure.py`'s
  output for each arm; the source of every number in Results above.

For the underlying per-domain scan (the eval-item dump, index, scanner and
aggregator, and their own dependencies), see
[`../mixlaw/contamination/README.md`](../mixlaw/contamination/README.md).

## Reproducing

```bash
CONTAM=path/to/this/directory       # experiments/skill-dag/skillit/contamination
MIXLAW=path/to/mixlaw/contamination # experiments/skill-dag/mixlaw/contamination

# The per-domain scan itself is reproduced from $MIXLAW; see that
# directory's own README. Given results_olmo127b-reservoir.json there:

python "$CONTAM/trajectory_exposure.py" \
  --per-domain "$MIXLAW/results_olmo127b-reservoir.json" \
  --updates "$CONTAM/skillit_updates_deriv.jsonl" \
  --arm skillit-deriv --final-step 2384 \
  --out "$CONTAM/exposure_deriv.json"

python "$CONTAM/trajectory_exposure.py" \
  --per-domain "$MIXLAW/results_olmo127b-reservoir.json" \
  --updates "$CONTAM/skillit_updates_probe.jsonl" \
  --arm skillit-probe --final-step 2384 \
  --out "$CONTAM/exposure_probe.json"
```

`skillit_updates_<arm>.jsonl` is written by the training loop itself (see
`../train_skillit_370m.py` and `../wandb_logging.py`) at each Skill-It update
step; it is not something this script regenerates from config, since it
records what a specific completed run actually did, including which
mixture it was launched from.
