# Token-selection experiments

Shared package: [`token_selection/`](token_selection/) (`PYTHONPATH=experiments/token-selection`).
Reference architecture source of truth: [`reference/`](reference/) (RefHQ CE, leave as-is).

| Arm | Directory | Selection | Status |
|-----|-----------|-----------|--------|
| Full-loss control | [`control/`](control/) | none (full CE on every valid target token) | `full-loss-control-regmix10b-v2` — W&B [`eduLLM/token-selection/hh19uatg`](https://wandb.ai/eduLLM/token-selection/runs/hh19uatg), tagged `cloned`; **being superseded**, see below |
| Control (random 60%) | [`control/`](control/) | uniform random keep 60% | Standalone trainer; `random-control-regmix10b-v1` (W&B `fa841187ff07e9164da282efd353c217`) |
| BLADE | [`blade/`](blade/) | top-60% `L_proxy − L_ref` | RegMix proxy/penalty stream + pinned `pretrain/refhq-instruct/v3` HQ updates; syncs 500/875/1250/1625/2000; K=75, τ=375, γ=0.6, λ=1.0; blade_start=500; pre/post-sync checkpoints |
| RHO-1 | [`rho-1/`](rho-1/) | top-60% `L_curr − L_ref` | Frozen refhq-instruct v3 step940; `t0=0`; YAML spine |
| REL exp-α | [`rel-ema-exp/`](rel-ema-exp/) | top-60% `L_curr − L_hist` | Bias-corrected EMA from zero; `α(t)=1−e^(−t/300)`; `t0=0` |
| Middle PPL (token) | [`middle-ppl-token/`](middle-ppl-token/) | middle-60% by frozen RefHQ `L_ref` | Online scorer; `t0=0` |
| Attention | [`attention/`](attention/) | top-60% attn-received | `attention_topk`; FA-safe hook+recompute; `t0=0` |
| Reference (RefHQ) | [`reference/`](reference/) | — | Frozen HQ 5.5B; arch / refs only |

### Shared contracts

- **Architecture:** `TransformerConfig.olmo2_370M` (full attn, no SWA), GBS `4_194_304`, seq 2048, peak LR `4e-4`, warmup 24, `alpha_f=0.1` (match RefHQ).
- **Permanent checkpoints:** step 0, every 125 (skip last grid point if within 125 of final), plus final — e.g. `{0,125,…,2125,2360}` (omit 2250). Helper: `token_selection.olmo_ext.checkpoint_ladder`.
- **Token budget:** one epoch of published `pretrain/regmix-10b` **v1**, whose realized size is
  **10,004,807,041 tokens**. YAML/standalone defaults use `9900000000` → **2360** steps at GBS
  `4_194_304`. All seven arms (full-loss control, random control, RHO-1, BLADE, Attention,
  Middle-PPL, REL-EMA) train on this same published dataset. Neither 2360 steps
  (9,898,557,440 tokens) nor the full-loss control's 2384 steps (9,999,220,736 tokens) wraps into
  a second epoch — both stay under 10,004,807,041.

  Realized per-domain token counts (uint32 `.npy` bytes / 4, verified on FarmShare):

  | Domain | Tokens |
  | --- | --- |
  | dclm | 3,752,801,841 |
  | arxiv | 2,500,162,905 |
  | starcoder | 1,406,986,385 |
  | pes2o | 938,157,310 |
  | open-web-math | 635,098,778 |
  | algebraic-stack | 615,239,017 |
  | wiki | 156,360,805 |
  | **total** | **10,004,807,041** |

- **Online selection warmup:** `t0_steps=0` / `t0_frac=0` for all online scorers. BLADE keeps its separate 500-step proxy warmup.
- **Task loss:** full 20-label OLMo-ladder `task_loss_bpb` (RC 5-shot) via `task_loss_hook` / `TaskLossEvalCallback` on each permanent save. Evaluator: `scripts/farmshare/task_loss/eval_task_loss_olmo_core.py`.
  The **20 labels are (task, split) pairs over 10 OLMES benchmarks**, not 20 distinct benchmarks:
  ARC-Challenge, ARC-Easy, BoolQ, CSQA, HellaSwag, MMLU, OpenBookQA, PIQA, SocialIQA, WinoGrande.
  Two consequences worth stating explicitly when reading the macro number:
  - **MMLU contributes 8 of the 20 labels** (`stem` / `humanities` / `social_sciences` / `other`
    × `val` / `test`), i.e. **40% of the macro weight** sits on a single benchmark.
  - **7 of the 20 labels are `test` splits** (ARC-Challenge, ARC-Easy, OpenBookQA, and the four
    MMLU categories), so calling the aggregate "validation macro bpb" is a **misnomer**; it is a
    mixed val/test macro.
- **Hardware:** discover world size from `torchrun` / env; no hardcoded GPU count, device pins, or host paths as required defaults.

### Full-loss control (reported)

The full-loss control is **not** a matched sibling of the six selection arms, and the mismatch is
load-bearing for how Table 1 is read.

| Field | Value |
| --- | --- |
| Run | `full-loss-control-regmix10b-v2` |
| W&B | [`eduLLM/token-selection/hh19uatg`](https://wandb.ai/eduLLM/token-selection/runs/hh19uatg) |
| Tags | `cloned` |
| Cloned from | `eduLLM/hpo-ladder`, group `hpo-ladder-batch-ablation`, run `library-4mi` |
| Dataloader seed | **6199** (the other arms use 42) |
| Init seed | **0** as actually logged — `init_seed` did not propagate into `TransformerConfig`, so the intended 6199/6198 never reached weight init |
| Steps | **2384** (the other arms run 2360) |
| Eval grid | ~**119**-step spacing (the other arms use the 125-step permanent ladder) |

**Being superseded.** A matched rerun `full-loss-control-regmix10b-v3` is in flight: FarmShare job
**1719708**, 4×L40S, **2360** steps, **125**-step checkpoint ladder, `method="full"` with
`keep_fraction=1.0` routed through the stock train module (not the selection path). Once it lands,
the control is seed-, step- and grid-matched to the selection arms and the confound described in
the paper's Section 3 goes away. Until then, treat the control's margin over the selection arms as
carrying an unquantified seed/step/grid component.

---

See each arm’s `README.md` for launch commands. Do not submit AWS workloads unless explicitly authorized.

---

Full experiment plan: [README.md](README.md).  
Contamination audit (paper Section 4): [contamination/](contamination/).  
Artifacts believed stale but deliberately NOT removed: [STALE.md](STALE.md).