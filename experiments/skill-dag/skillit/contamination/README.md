# Benchmark contamination audit -- Skill-It's time-varying exposure

Skill-It reweights domains mid-training (see `../README.md`'s "Skill-It
update"), so unlike the fixed-mixture arms in `../../mixlaw/contamination/`, a
single static weight vector does not describe any of its three arms'
exposure to contaminated eval text over a full run. This directory computes
that exposure honestly: per-segment, then as a token-weighted average over
the run.

**This directory does not re-scan the reservoir.** All three Skill-It arms
train on domain-stratified samples of the same `pretrain/olmo-127b`
reservoir the MixLaw arms do, against the same 40,582-item eval suite, so
the per-domain contamination rates are identical; only the mixture weights
differ, and only because they change over time here. `../../mixlaw/contamination/`
holds the scan, the eval-item index pipeline, and
`results_olmo127b-reservoir.json`; this directory reads that file directly
rather than duplicating a ~10 MB eval-item dump and a 172 MB index for a
scan that would produce the same numbers.

## The three arms

`../README.md`'s "Arms actually run" table names three Skill-It arms, not
two:

| Arm | Starting mixture | Adjacency |
| --- | --- | --- |
| Offline probe | Data Mixing Laws paper mix (`mix01`) | fixed offline `A` |
| Online derivative | Data Mixing Laws paper mix (`mix01`) | recomputed `A(r)` from MixLaw derivatives at each update |
| Offline (MixLaw start) | MixLaw optimum (`ML-pilot_caps`) | same fixed offline `A` as Offline probe |

Comparisons in `../README.md` use the **Data Mixing Laws paper mixture**
as control (a full fixed-weight run, not an extra Skill-It train) -- not
`olmo-mix-1124`. This directory reports exposure relative to both, since
`../../mixlaw/contamination/exposure_by_arm.json` already has `olmo-mix-1124`
as its baseline and the Data Mixing Laws paper mixture's own exposure is one
of its four rows.

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

**Provenance of these six-row files.** They are transcribed from
`../README.md`'s own "Domain weights after each update" table, which that
paper sourced from W&B (`skillit/weight/*`) for the three arms it actually
reports results for. This directory does **not** use the raw
`skillit_updates.jsonl` progress logs found on FarmShare scratch for
similarly-named runs (`skillit-370m-deriv-*`, `skillit-370m-probe-rerun-*`):
those runs' logged starting mixture is `LGB-min1pct`, and their weights
move in the *opposite* direction from what `../README.md` reports (their
`dclm` weight rises toward 0.55-0.86 over training; the published runs'
`dclm` weight *falls* from 0.375 toward ~0.15-0.19) -- they are a different,
unpublished pair of runs, not the ones behind this paper's numbers. Using
the paper's own published table, sourced from the same W&B metric the
paper cites, is the measurement that matches what was actually reported,
even though it costs one significant figure of precision against a raw
per-step log.

## Results

All three arms start from a mixture already measured in
`../../mixlaw/contamination/README.md`: Offline probe and Online derivative
both start at the Data Mixing Laws paper mixture (span rate 1.303e-05,
0.687x the `olmo-mix-1124` baseline); Offline (MixLaw start) starts at the
MixLaw optimum (4.016e-05, 2.119x baseline). All three then move weight
toward `wiki` -- by far the highest-rate domain (9.504e-05) -- and away from
`dclm`, so exposure *rises* over the run in every arm, ending well above
where each one started.

### Offline probe

From `exposure_offline-probe.json`.

| Step | Length (steps) | `weight[dclm]` | `weight[wiki]` | Span rate | Doc rate |
| --- | --- | --- | --- | --- | --- |
| 0 | 500 | 0.374 | 0.016 | 1.305e-05 | 6.029e-04 |
| 500 | 375 | 0.194 | 0.165 | 2.361e-05 | 6.003e-04 |
| 875 | 375 | 0.190 | 0.163 | 2.337e-05 | 5.979e-04 |
| 1250 | 375 | 0.188 | 0.163 | 2.333e-05 | 5.974e-04 |
| 1625 | 375 | 0.187 | 0.162 | 2.324e-05 | 5.963e-04 |
| 2000 | 384 | 0.186 | 0.162 | 2.324e-05 | 5.966e-04 |

**Time-weighted average: span rate 2.120e-05 (1.12x the `olmo-mix-1124`
baseline; 1.63x the Data Mixing Laws paper control it is actually compared
against), doc rate 5.988e-04 (0.84x baseline; 0.99x control).**

### Online derivative

From `exposure_online-derivative.json`.

| Step | Length (steps) | `weight[dclm]` | `weight[wiki]` | Span rate | Doc rate |
| --- | --- | --- | --- | --- | --- |
| 0 | 500 | 0.374 | 0.016 | 1.305e-05 | 6.029e-04 |
| 500 | 375 | 0.141 | 0.199 | 2.594e-05 | 6.009e-04 |
| 875 | 375 | 0.151 | 0.180 | 2.438e-05 | 5.918e-04 |
| 1250 | 375 | 0.151 | 0.180 | 2.439e-05 | 5.922e-04 |
| 1625 | 375 | 0.151 | 0.180 | 2.437e-05 | 5.919e-04 |
| 2000 | 384 | 0.151 | 0.179 | 2.430e-05 | 5.915e-04 |

**Time-weighted average: span rate 2.223e-05 (1.17x baseline; 1.71x
control), doc rate 5.956e-04 (0.83x baseline; 0.99x control).**

### Offline (MixLaw start)

From `exposure_offline-mixlaw-start.json`.

| Step | Length (steps) | `weight[dclm]` | `weight[wiki]` | Span rate | Doc rate |
| --- | --- | --- | --- | --- | --- |
| 0 | 500 | 0.568 | 0.300 | 4.016e-05 | 7.575e-04 |
| 500 | 375 | 0.194 | 0.166 | 2.369e-05 | 6.005e-04 |
| 875 | 375 | 0.190 | 0.163 | 2.339e-05 | 5.980e-04 |
| 1250 | 375 | 0.188 | 0.162 | 2.328e-05 | 5.968e-04 |
| 1625 | 375 | 0.187 | 0.162 | 2.325e-05 | 5.966e-04 |
| 2000 | 384 | 0.186 | 0.162 | 2.324e-05 | 5.966e-04 |

**Time-weighted average: span rate 2.689e-05 (1.42x baseline; 2.06x
control), doc rate 6.312e-04 (0.88x baseline; 1.05x control).**

### Reading these together

Offline probe and Offline (MixLaw start) share the same fixed adjacency and
converge to nearly identical weights by step 2000 (`dclm` 0.186 either way)
despite starting almost as far apart as two mixtures in this experiment can
-- so their exposure gap is driven almost entirely by the first 500-step
segment, where MixLaw-start's much higher starting exposure (4.016e-05 vs
1.305e-05) has not yet been diluted by four updates toward the shared fixed
point. Online derivative's own online-recomputed adjacency settles on a
different fixed point (`dclm` 0.151, `wiki` 0.179) with a higher `wiki`
share than either offline arm, which is why its time-weighted average span
rate (2.223e-05) sits between the two offline arms' despite starting equal
to Offline probe.

Span rate and document rate disagree in direction for all three arms: every
arm's span rate finishes *above* the `olmo-mix-1124` baseline while its
document rate finishes *below* it. Moving weight from `dclm` (huge document
count, comparatively low per-word density) onto `wiki` (far fewer documents,
much higher per-word density) raises the fraction of *words* sitting in a
matched span while lowering the fraction of *documents* that contain one at
all -- the same length-bias `../../mixlaw/contamination/README.md` flags for
why both rates are reported rather than a single "contamination score."

All three arms end the run **more** exposed than the Data Mixing Laws paper
mixture they are compared against (1.63x-2.06x its span rate) -- yet
`../README.md`'s own results show none of the three beating that control
(Data Mixing Laws paper 1.6518 fitted-final bpb; Offline probe 1.6544;
Online derivative 1.6690; Offline (MixLaw start) 1.6747). If contaminated
exposure conferred a memorization advantage on eval-adjacent text, the
arms most exposed relative to their control should have had the easiest
path to matching or beating it; instead every arm not only failed to beat
a less-exposed control, the ranking runs the other way -- Offline probe,
the least additionally exposed of the three (1.63x control), comes closest
to matching it, and Offline (MixLaw start), the most exposed (2.06x
control), finishes furthest behind.

## Code map

- `trajectory_exposure.py` -- computes the per-segment and time-weighted
  average exposure described above. Stdlib only. Reads
  `../../mixlaw/contamination/results_olmo127b-reservoir.json` for per-domain
  rates; needs no other file from `../../mixlaw/`.
- `skillit_updates_offline-probe.jsonl`, `skillit_updates_online-derivative.jsonl`,
  `skillit_updates_offline-mixlaw-start.jsonl` -- each arm's realized
  per-update-step mixture, transcribed from `../README.md`'s own published
  table (see Provenance above). Six rows each (steps 0, 500, 875, 1250,
  1625, 2000).
- `exposure_offline-probe.json`, `exposure_online-derivative.json`,
  `exposure_offline-mixlaw-start.json` -- `trajectory_exposure.py`'s output
  for each arm; the source of every number in Results above.

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

python "$CONTAM/trajectory_exposure.py" \
  --per-domain "$MIXLAW/results_olmo127b-reservoir.json" \
  --updates "$CONTAM/skillit_updates_offline-mixlaw-start.jsonl" \
  --arm offline-mixlaw-start --final-step 2384 \
  --out "$CONTAM/exposure_offline-mixlaw-start.json"
```

To reproduce the `skillit_updates_<arm>.jsonl` files themselves from source
rather than from `../README.md`'s table, pull the `skillit/weight/*` metric
history for each arm's actual W&B run (see `../wandb_logging.py` for the
metric names) -- there is no committed raw per-step log for these three
specific runs in this repository.
