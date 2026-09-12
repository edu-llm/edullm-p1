# RunPod Cost Analysis (P1 + CoLMLM)

Measured cost analysis for RunPod work Aug 1–5, 2026: **P1 experiments** (token selection, curriculum learning, SkillIt, MixLaw) plus **CoLMLM / SmolLM2-135M** fact-masked training and annotation.

**Analysis date:** 2026-08-05  
**Billing window:** 2026-08-01 through 2026-08-05

### Account overview (RunPod billing)


| Category         | RunPod $    | Section                    |
| ---------------- | ----------- | -------------------------- |
| P1 experiments   | $1,858      | §1–§4                      |
| CoLMLM / SmolLM2 | ~$263       | §5                         |
| **Total**        | **~$2,121** | Billing API: **$2,136.77** |


P1 and CoLMLM line items below sum to the account total. Hours are reported by **hardware type** (A100-hours vs L40S-hours), not a generic GPU-hour blend.

---



## Methodology


| Source                                                  | Role                                                          | Limitation                                                                                                         |
| ------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **RunPod billing** (`get-billing`, Aug 1–5 2026)        | **Ground truth for actual spend**                             | Includes idle time, staging, crashes, reruns                                                                       |
| **W&B** `_runtime` (all 32 `eduLLM/`* projects scanned) | **Ground truth for per-run wall time** where runs still exist | **Incomplete** — failed runs were deleted; smoke/debug runs lived in `mixlaw`, `token-selection-rel-ema-exp`, etc. |
| **Chat logs**                                           | Fill gaps where W&B is gone                                   | Used for BLADE completion, middle-ppl v1 context, and the deleted RefHQ ρ-1 run                                    |


**Rates (measured):**


| Platform                               | Pod rate          | Per-device rate    |
| -------------------------------------- | ----------------- | ------------------ |
| RunPod secure 8×A100                   | **$11.92/hr**     | **$1.49/A100-hr**  |
| RunPod secure 8×L40S                   | **~$8.60/hr**     | **~$1.08/L40S-hr** |
| AWS `p4d.24xlarge` on-demand us-east-1 | **$21.957642/hr** | **$2.745/A100-hr** |


AWS pricing from AWS Pricing API (2026-08-04). RunPod A100 rate from secure 8×A100 launches; some early P1 pods used community pricing at **$11.12/hr** ($1.39/A100-hr). L40S rate imputed from pod `y83pcj0g00wijz` billing ÷ W&B wall time on the 8×L40S training run.

---



# P1 experiments

P1 activity concentrated Aug 2–5. All P1 training used **8×A100 SXM 80GB** pods unless noted.

## 1. Efficient work (optimized code, one clean pass per arm)

These are **W&B** `_runtime` **× 8 GPUs** for finished canonical runs, plus imputed values only where W&B records were deleted. Arms with deleted W&B are marked.

**Arm count:** 18 production training arms (16 from the original P1 grid plus **two ρ-1 reference variants**), plus bootstrap/precompute lines below.

### MixLaw — 4 arms (`mixlaw-1`)


| Arm           | A100-hours | W&B      |
| ------------- | ---------- | -------- |
| ML-pilot_caps | 48.78      | measured |
| LGB-min1pct   | 48.39      | measured |
| mix01         | 44.11      | measured |
| olmo-mix-1124 | 47.46      | measured |
| **Subtotal**  | **188.74** |          |




### SkillIt — 3 arms


| Arm                   | A100-hours | W&B                                                                                                                  |
| --------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------- |
| probe                 | 47.26      | measured                                                                                                             |
| deriv (online)        | **53.8**   | repriced — W&B wall 70.51 A100-h (`e9ac2617…`, host `f3d49045bd7f`); efficient at steady-state throughput (see note) |
| offline-ml-pilot-caps | 47.14      | measured                                                                                                             |
| **Subtotal**          | **148.2**  |                                                                                                                      |


**SkillIt online derivative note:** Steady-state reference = `skillit-probe` median `throughput/device/TPS (actual avg)` (~60,050 tok/s/device). On the defective pod, steps ~626–1519 hit data-loading up to **94%** and TPS as low as **22k**. Repricing those slow steps to steady throughput yields **53.8 A100-h efficient**; the remaining **~16.8 A100-h** of wall time is I/O waste (§2).

### Curriculum — 5 arms


| Arm               | A100-hours | W&B                                                                                                                                |
| ----------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| linear10-flesch   | 50.81      | measured                                                                                                                           |
| linear10-mtld     | 43.97      | measured                                                                                                                           |
| warmup-flesch     | 43.84      | measured                                                                                                                           |
| interleave-flesch | 47.74      | measured                                                                                                                           |
| linear10-learn    | **43.84**  | measured — successful run to step 2384 (`83b3001e`, W&B deleted). W&B **replay** for chart backfill has `_runtime≈0` (negligible). |
| **Subtotal**      | **230.20** |                                                                                                                                    |




### Token selection — 6 training arms + bootstrap/precompute


| Arm / step                    | A100-hours | W&B / provenance                                                                                                                                                                                                |
| ----------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| rel-ema-exp                   | 59.40      | measured (`token-selection`)                                                                                                                                                                                    |
| attention-topk                | 58.55      | measured                                                                                                                                                                                                        |
| middle-ppl v2 (training)      | 42.99      | measured                                                                                                                                                                                                        |
| middle-ppl mask precompute    | **~2.0**   | measured 15-min 8-GPU benchmark (chat); not a separate W&B run                                                                                                                                                  |
| **ρ-1 (RefHQ step1315)**      | **56.56**  | **W&B deleted** (`d889eb3…`, pod `xw8xu5i7nim10j`). Full optimized run to step 2361 on `edullm-370M-refhq-5p5b` reference. A100-hours imputed from measured instruct-v3 sibling (same recipe, same step count). |
| **ρ-1 (instruct-v3 step940)** | **56.56**  | measured — `[rho-1-regmix10b-v1](https://wandb.ai/eduLLM/token-selection/runs/ebf1fa33048b3459f768cd471c2a8917)`, pod `ksdd38tuvye23h`, finished step 2361                                                      |
| refhq-instruct bootstrap      | 19.19      | measured (reference CE train for instruct-v3; used as the frozen reference by rho-1, and as BLADE's K-update corpus)                                                                                                                                       |
| BLADE                         | **~70.5**  | **primary run deleted** (`5766dcf8…`); training finished step 2360/2361 per chat. Imputed from skillit-deriv (similar sync/eval overhead).                                                                      |
| **Subtotal**                  | **~365.8** |                                                                                                                                                                                                                 |


**ρ-1 note:** These are intentionally counted as **two separate essential arms** — one frozen-reference experiment with legacy RefHQ (`s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-5p5b/checkpoints/step1315/`) and one with the new instruct reference (`s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-instruct-v3/checkpoints/step940/`). The pre-optimization ρ-1 attempt on the slow weight-shadow path (~40 steps, W&B `6bfbddbd…`, also deleted) is **not** counted as efficient work. **Only the instruct-v3 arm is the reported RHO-1 run** (`rho-1-regmix10b-v1`, W&B `ebf1fa33048b3459f768cd471c2a8917`); `arms.py` pins `RHO_REFERENCE_CHECKPOINT` to the instruct-v3 step940 path. The legacy refhq-5p5b step1315 row is sunk cost on a deleted run and must not be cited as the paper's RHO-1 reference.

> **The reported rel-ema-exp arm uses NO reference model.** `arms.py` configures it with
> `ema_seed="zero"` and no `reference_contract` -- the "history" model is a
> bias-corrected EMA of the student itself. Any statement that rel-ema consumes the
> instruct-v3 reference is wrong.

### Token selection: A100-hours are not comparable across arms

**Do not compare the A100-hour column across token-selection arms.** Five arms ran on
**8xA100-80GB** (rho-1, BLADE, Attention, Middle-PPL, and the full-loss control); the
**Random control and REL-EMA ran on 4xL40S**. Hours on those two platforms are not
interchangeable, and the L40S entries in an A100-hours column are conversions rather
than measurements. Compounding this, the **logged throughput FLOPs counter excludes the
scoring forward passes**, so it reports an identical 2.63e19 for a full-CE run and for a
run that additionally evaluates a frozen reference on every token.

Use these analytic FLOPs for any cross-arm comparison.

**Model.** Forward cost per token = `2N + 4*L*T*d`, with `N = 371,195,904` matmul
parameters (**including the untied output head**), `L = 16` layers, `T = 2048` sequence
length, `d = 1024` model width. Forward+backward = **3x** forward. This reproduces the
logged W&B throughput counter to within **0.05%**, which is what licenses using it for
the arms whose counter is incomplete.

| Arm | In-run (x10^18) | With reference pretraining (x10^18) | Relative to full-loss control |
| --- | --- | --- | --- |
| Random control | 26.03 | 26.03 | 0.99x |
| Attention | 26.12 | 26.12 | 0.99x |
| **Control (full-loss)** | **26.30** | **26.30** | **1.00x** |
| REL-EMA | 34.71 | 34.71 | 1.32x |
| RHO-1 | 34.71 | 45.08 | 1.71x |
| BLADE | 39.71 | 47.98 | 1.83x |
| Perplexity (Middle-PPL) | 34.71 | 49.21 | 1.87x |

**BLADE K-update overhead, itemized.** 5 syncs x 75 K-steps x 2 streams (proxy and
reference) = **750 full batches** of 4,194,304 tokens = **3,145,728,000 tokens** of
forward+backward, on top of the 2360 training steps. That is the 39.71 vs 34.71 in-run
difference.

### Token selection: two provenance defects in the cost rows above

**BLADE `005xjces` did not produce the curve attributed to it.** That W&B run has a
**7-second runtime** and logs **no throughput and no keep-rate metrics**. Its loss curve
was therefore backfilled or resumed into that record, not generated by it. The BLADE
cost figure (~70.5 A100-h) is already flagged as imputed; this is a separate point
about the *metrics* record.

**The 59.40 A100-h REL-EMA figure is the wrong run.** It corresponds to the **OLD
inverted-polarity A100 run `89db0d5b...`**, which is superseded. The REL-EMA run used in
Table 1 is the polarity-corrected `cc52d5537a03ad8e57cc87a025668b2e`, which ran
**21.39 h on 4xL40S**. The 59.40 figure should be read as sunk cost on a discarded
polarity, not as the cost of the reported arm.


### Efficient totals


| Lever           | A100-hours      | RunPod @ $1.49/A100-hr | AWS @ $2.745/A100-hr |
| --------------- | --------------- | ---------------------- | -------------------- |
| MixLaw          | 188.7           | $281                   | $518                 |
| SkillIt         | 148.2           | $221                   | $407                 |
| Curriculum      | 230.2           | $343                   | $632                 |
| Token selection | 365.8           | $545                   | $1,004               |
| **Total**       | **~933 A100-h** | **~$1,390**            | **~$2,561**          |


---



## 2. Total spend (efficient + waste)

**Totals are summed from line items** — §1 efficient A100-hours plus the waste rows below. RunPod billing and W&B are used to attribute individual arms, not to derive the aggregate.


| Category                                    | A100-h     | RunPod @ $1.49/A100-hr |
| ------------------------------------------- | ---------- | ---------------------- |
| Efficient required work (§1)                | ~933       | ~$1,390                |
| Debugging / crashes / reruns / idle (below) | ~314       | ~$468                  |
| **Total**                                   | **~1,247** | **~$1,858**            |


**You spent ~1.34× the efficient minimum** (~34% overhead).

Only ~58 A100-h of non-canonical work still appears in surviving W&B; the rest is deleted runs, early aborts, and serial pod reuse (`vu6arqkxs0gv9h` billed **$534** across middle-ppl, curriculum, and BLADE).

### Waste by experiment and arm

Allocated to arms using **measured** W&B non-canonical runtime where available, plus **billing/chat attribution** for deleted runs. Dollars at **$1.49/A100-hr**. *Imputed* = no surviving W&B record.

#### MixLaw — ~91 A100-h waste (~$136)


| Arm           | Waste A100-h | Waste $   | What happened                                                                                                                            |
| ------------- | ------------ | --------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **mix01**     | **52.2**     | **$78**   | Measured W&B: duplicate finished run in project `mixlaw` (43.7 A100-h, Aug 2) + crashed attempt at step 370 (7.8 A100-h) + launch smokes |
| ML-pilot_caps | 12           | 18        | Bootstrap/staging and failed launches before canonical Aug 3 run                                                                         |
| LGB-min1pct   | 12           | 18        | Same                                                                                                                                     |
| olmo-mix-1124 | 14.5         | 22        | Measured W&B smokes (1.5 A100-h) + pre-canonical staging                                                                                 |
| **Subtotal**  | **~91**      | **~$136** |                                                                                                                                          |




#### SkillIt — ~52 A100-h waste (~$77)


| Arm                           | Waste A100-h | Waste $  | What happened                                                                                                                                                              |
| ----------------------------- | ------------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| skillit-probe                 | 10           | 15       | Serial re-bootstrap / idle between back-to-back arms on shared pod                                                                                                         |
| **skillit-deriv (online)**    | **29**       | **$43**  | **~16.8 A100-h** defective-pod I/O on host `f3d49045bd7f` (steps ~626–1519: data-loading up to 94%, TPS down to 22k vs steady ~60k) + 12 A100-h shared-pod serial overhead |
| skillit-offline-ml-pilot-caps | 13           | 19       | Final arm on same shared pod                                                                                                                                               |
| **Subtotal**                  | **~52**      | **~$77** |                                                                                                                                                                            |


No failed SkillIt training runs in surviving W&B.

#### Curriculum — ~36 A100-h waste (~$54)


| Arm                    | Waste A100-h | Waste $  | What happened                                     |
| ---------------------- | ------------ | -------- | ------------------------------------------------- |
| linear10-flesch        | 5            | 7        | Staging/idle on shared curriculum pod             |
| linear10-mtld          | 5            | 7        | Same                                              |
| warmup-flesch          | 5            | 7        | Same                                              |
| interleave-flesch      | 5            | 7        | Same                                              |
| shared serial pod idle | 16           | 24       | Idle on `vu6arqkxs0gv9h` between curriculum swaps |
| **Subtotal**           | **~36**      | **~$54** |                                                   |


**Not waste:** `linear10-learn` — training finished step 2384 on pod `vu6arqkxs0gv9h` (`83b3001e`, W&B deleted after disk-full eval at step 2000). Those **~44 A100-h are efficient** (already in §1). Only a negligible W&B replay was rerun for chart backfill (`_runtime≈0`).

#### Token selection — ~135 A100-h waste (~$201)


| Arm                                            | Waste A100-h | Waste $   | What happened                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------------------------------------------- | ------------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **BLADE v2 (old RefHQ, abandoned)**            | **~69**      | **~$103** | Pod `e9futclc4va3s7` billed **$103** Aug 3–4 (~8.7h wall × 8 GPU). Run `blade-regmix10b-v2` on `pretrain/refhq-regmix-5p5b/v2`; W&B deleted (`df99ee09…`). Failed at step 125 (volume EIO), then **repeated step-500 sync failures** — first a rank-0-only state-dict bug (process wedged at step 500), then OOM on post-sync selection after the fix — across **~15 resume attempts**; max durable step **375**. Abandoned for instruct-v3 relaunch on `vu6arqkxs0gv9h`. |
| **middle-ppl (v1 online scoring)**             | **22**       | **$33**   | Suboptimal path stopped at step 12; W&B deleted (`6e69fcf…`). ~2.8h wall × 8 before kill + failed staging pod `sy1u16xc8736ag` (~3.5 A100-h). *Not* the ~20h launch projection                                                                                                                                                                                                                                                                                            |
| **BLADE instruct-v3 (canonical run overhead)** | **18**       | **$27**   | Successful run W&B deleted (`5766dcf8…`); step-875 OOM recovery on `vu6arqkxs0gv9h` beyond ~70.5 A100-h efficient run                                                                                                                                                                                                                                                                                                                                                     |
| **ρ-1 (RefHQ step1315)**                       | **6**        | **$9**    | Pre-optimization slow path (~40 steps, W&B `6bfbddbd…` deleted) before optimized restart; pod `xw8xu5i7nim10j` billed $87 vs 56.6 A100-h efficient                                                                                                                                                                                                                                                                                                                        |
| ρ-1 (instruct-v3 step940)                      | 4            | 6         | Staging on pod `ksdd38tuvye23h` ($84 billed vs 56.6 A100-h efficient)                                                                                                                                                                                                                                                                                                                                                                                                     |
| rel-ema-exp                                    | 2            | 3         | Measured W&B crash at step 9 (1.9 A100-h) before successful run                                                                                                                                                                                                                                                                                                                                                                                                           |
| attention-topk                                 | 7            | 10        | Recovery after initial hang; pod `km1t4srez712c7` ~66 A100-h billed vs 58.6 efficient                                                                                                                                                                                                                                                                                                                                                                                     |
| refhq-instruct bootstrap                       | 3            | 4         | Staging retries; efficient train is 19.2 A100-h                                                                                                                                                                                                                                                                                                                                                                                                                           |
| middle-ppl v2 (training)                       | 4            | 6         | `vu6arqkxs0gv9h` swap overhead between v1 abort → precompute → v2                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Subtotal**                                   | **~135**     | **~$201** |                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |


middle-ppl mask precompute (~2 A100-h) is efficient work in §1, not waste.

#### Summary — waste by lever


| Lever           | Waste A100-h | Waste $   | Largest bucket                                                                                          |
| --------------- | ------------ | --------- | ------------------------------------------------------------------------------------------------------- |
| MixLaw          | ~91          | ~$136     | mix01 smokes/rerun (52 A100-h)                                                                          |
| SkillIt         | ~52          | ~$77      | **skillit-deriv defective-pod I/O** (~17 A100-h)                                                        |
| Curriculum      | ~36          | ~$54      | Shared-pod serial overhead on `vu6arqkxs0gv9h`                                                          |
| Token selection | ~135         | ~$201     | **BLADE v2 abandoned old-RefHQ run** (~69 A100-h on `e9futclc4va3s7`) + middle-ppl v1 abort (22 A100-h) |
| **Total**       | **~314**     | **~$468** |                                                                                                         |


**Not in this table:** clean single-pass canonical runs; post-completion idle on `9g7tjgi3jrhn80` after refhq bootstrap (~55 A100-h not attributed to any training arm).

---



## 3. AWS equivalent (same experiments, 8×A100)

Using the **same measured A100-hours** on `p4d.24xlarge`:


|                              | RunPod  | AWS on-demand |
| ---------------------------- | ------- | ------------- |
| Efficient work (~933 A100-h) | ~$1,390 | **~$2,561**   |
| Total spent (~1,247 A100-h)  | ~$1,858 | **~$3,423**   |


RunPod was **~1.84× cheaper** than AWS on-demand for the same A100-hours.

---



## 4. Per-lever actual spend (efficient + waste)


| Lever           | Efficient A100-h | Waste A100-h | Actual A100-h | RunPod $    |
| --------------- | ---------------- | ------------ | ------------- | ----------- |
| MixLaw          | 188.7            | ~91          | ~280          | ~$417       |
| SkillIt         | 148.2            | ~52          | ~200          | ~$298       |
| Curriculum      | 230.2            | ~36          | ~266          | ~$397       |
| Token selection | 365.8            | ~135         | ~501          | ~$746       |
| **Total**       | **~933**         | **~314**     | **~1,247**    | **~$1,858** |


---



## Key takeaways

1. **~34% overhead** on top of efficient work — mostly reruns, deleted runs, defective-pod I/O, and serial pod reuse rather than idle-only billing.
2. **MixLaw mix01** is the single largest waste bucket (~52 A100-h / ~$78) from duplicate finished run + crash at step 370.
3. **middle-ppl v1** waste is **~22 A100-h**, not the ~160 A100-h projected at launch when the suboptimal online-scoring path was killed at step 12.
4. **Shared pod** `vu6arqkxs0gv9h` drove cross-experiment waste (middle-ppl, curriculum, BLADE) — serial swaps and recovery time add up.
5. **Both ρ-1 arms are essential** (~57 A100-h each); only the RefHQ slow-path pre-restart (~6 A100-h) counts as waste.
6. **BLADE had two runs:** the abandoned `blade-regmix10b-v2` on old `refhq-regmix-5p5b` (~69 A100-h / ~$103, step-500 sync death loop) and the canonical `refhq-instruct-v3` run (~70.5 A100-h efficient + ~18 A100-h recovery overhead).
7. `linear10-learn` **is efficient, not waste** — successful training to step 2384 (~44 A100-h in §1); only a negligible W&B replay was rerun for chart backfill.
8. `skillit-deriv` **I/O is waste, not efficient** — defective pod `f3d49045bd7f` throttled steps ~626–1519; **53.8 A100-h efficient** at steady-state throughput vs **70.5 A100-h** W&B wall.
9. **CoLMLM annotation was ~19 A100-h, not ~64** — 19 parallel 1×A100 workers (~1h wall); prior doc wrongly used `es73` 8×A100 billing.

---



# CoLMLM / SmolLM2-135M (§5)

CoLMLM work ran **Aug 1–2** (annotation + throughput tuning) before the P1 grid. Hardware mix: **19×1×A100** for ModernBERT annotation (parallel fleet), **4×/8×L40S** for SmolLM2 fact-masked training. Line items sum to **~$263** (see account note below).

**Billing pods (measured):** 19 single-A100 workers in `scripts/runpod/colmlm_annotate/fleet_2026-08-01.json` (~$27 GPU total), `y83pcj0g00wijz` (8×L40S training, $118 GPU), `s7zhyh3yopmq8l` (staging companion, $29 GPU), plus **~48 short Aug 1 annotate/smoke pods** (~$40 GPU). Pod `es73etvp8x7zrq` was **not** the annotate fleet — it billed **$112** as an **8×A100** pod used for **P1 MixLaw** Aug 2+, not CoLMLM annotation.

**Account note:** CoLMLM line items total **~$263** ($261 GPU + ~$2 disk). With P1 at **$1,858**, that is **~$2,121** of the **$2,136.77** RunPod bill; the **~$15** remainder is mostly `es73` Aug 1 GPU time before the MixLaw handoff (not counted in either line-item table).

## 5.1 Efficient work


| Work stream                             | Hardware        | Hours                         | W&B / provenance                                                                                                                                                                                                                           | RunPod $  |
| --------------------------------------- | --------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------- |
| ModernBERT annotation (19 shards → S3)  | **19×1×A100**   | **~19 A100-h**                | `create_annotate_fleet.js` launched **19 parallel 1×A100 workers** (`fleet_2026-08-01.json`, started 2026-08-01T20:42Z); **~1h wall** across the fleet → **19 A100-h**. Output at `s3://edullm-checkpoints/runpod/colmlm-annotate/output/` | ~$28      |
| 8×L40S CoLMLM training (W&B resume)     | 8×L40S          | **~110 L40S-h**               | `smollm2-colmlm-8xnvidia-l40s-wandb-resume` (`eduLLM/edullm-smollm2-colmlm`), pod `y83pcj0g00wijz`, finished step 61,020                                                                                                                   | ~$88      |
| 2×L40S fresh training (completed run)   | 2×L40S          | **~59 L40S-h**                | `smollm2-135m-750m-27ep-fresh` (`eduLLM/edullm-smollm2`), finished step 305,176                                                                                                                                                            | ~$52      |
| Throughput baselines (pre-optimization) | 4×A100 + 4×L40S | **~0.1 A100-h + ~1.2 L40S-h** | Measured stable windows: A100 153k tok/s (`4jbi0wgc`), L40S 131k tok/s (`v5lmbvdl`), optimized L40S 371k tok/s (`j4iyod7p`) per `scripts/runpod/smollm2_colmlm/README.md`                                                                  | ~$2       |
| **Subtotal**                            |                 | **~19 A100-h + ~170 L40S-h**  |                                                                                                                                                                                                                                            | **~$170** |


**Annotation note:** The earlier **~64 A100-h** figure was wrong — it divided `es73etvp8x7zrq` **8×A100** billing by $1.49/A100-hr. That pod was not the annotate fleet. The real annotate job was **19 single-A100 pods** running in parallel for **~1 wall-hour** each.

## 5.2 Waste


| Work stream                                | Hardware         | Hours               | What happened                                                                               | RunPod $ |
| ------------------------------------------ | ---------------- | ------------------- | ------------------------------------------------------------------------------------------- | -------- |
| Aug 1 annotate smoke / pre-fleet pods      | mixed small GPUs | **~28 small-GPU-h** | ~48 short-lived pods (~$1.44/hr each) before the 19-worker fleet launched                   | ~$40     |
| `smollm2-135m-750m-27ep-fresh-BROKEN-1gpu` | 2×L40S           | **~21 L40S-h**      | Crashed mid-run (step 86,420) before the successful fresh relaunch                          | ~$23     |
| 4×L40S crash + batch-size probes           | 4×L40S           | **~36 L40S-h**      | `…032940` crashed at step 14,460; failed `b32`/`b56`/`003206`/`014718` smoke/probe launches | ~$39     |
| Staging companion                          | —                | —                   | `s7zhyh3yopmq8l` disk staging / idle between annotate → train handoff                       | ~$29     |
| **Subtotal**                               |                  | **~57 L40S-h**      |                                                                                             | **~$91** |


Disk attributed to CoLMLM pods: **~$2**.

## 5.3 CoLMLM totals


| Category         | A100-h  | L40S-h   | RunPod $  |
| ---------------- | ------- | -------- | --------- |
| Efficient (§5.1) | ~19     | ~170     | ~$170     |
| Waste (§5.2)     | —       | ~57      | ~$91      |
| Disk             | —       | —        | ~$2       |
| **Total**        | **~19** | **~227** | **~$263** |


**~35% overhead** on CoLMLM efficient work — dominated by the Aug 1 smoke pods ($40) and failed L40S training launches before the optimized 8×L40S resume path.

---



## Data sources

- RunPod billing via MCP `get-billing` (Aug 1–5, 2026)
- W&B API scan across `eduLLM/*` projects (`_runtime`, run state, project tags), including `edullm-smollm2` and `edullm-smollm2-colmlm` for CoLMLM
- `scripts/runpod/smollm2_colmlm/README.md` (measured L40S/A100 throughput baselines)
- Chat: [ρ-1 RefHQ launch](893f100a), [ρ-1 instruct relaunch](ae01830b), [middle-ppl v1 abort](41fd065e)

