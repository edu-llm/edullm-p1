# Token selection 370M on FarmShare (4 × L40S)

Every arm below runs on one 4×L40S node, through
`token_selection_entrypoint.py`. There is no RunPod path and no AWS/S3 code in
this tree; corpora are pinned local FarmShare directories, verified once and
bound into `ready.json` by `stage_local.py`, and the reference checkpoint is
trained by the `instruct-reference` arm below, not downloaded.

Arms via `ARM`: `instruct-reference`, `full-loss-control`,
`rho-1`, `rel-ema-exp`, `perplexity`, `attention`, `blade`, `random-control`,
`random-control-seed69`.

```bash
cd /mnt/c/alpha_ai/OLMo-core-token-selection-370m
LOCAL_REPO="$(pwd)" ARM=attention bash .edullm/farmshare/submit_from_laptop.sh
```

## Staging order

1. `STAGE_MODE=corpora` (the default) verifies the two pinned corpus
   directories and writes `ready.json`. Run once; every arm shares it.
2. Train `instruct-reference` first. `rho-1` and `perplexity` both need its
   materialized checkpoint.
3. `STAGE_MODE=references` materializes that checkpoint into `ready.json`'s
   `references` map, keyed by the symbolic contract name in
   `token_selection_370m/arms.py`:

   ```bash
   STAGE_MODE=references SKIP_TRAIN=1 bash .edullm/farmshare/submit_from_laptop.sh
   ```

4. `STAGE_MODE=reference-scores` scores the whole RegMix corpus once against
   that checkpoint, on the 1-GPU `qos=normal` lane (`score_reference.py`; see
   `token_selection_370m/reference_scores.py`), and writes the result into
   `ready.json`'s `reference_scores` map:

   ```bash
   STAGE_MODE=reference-scores SKIP_TRAIN=1 bash .edullm/farmshare/submit_from_laptop.sh
   ```

   `rho-1` and `perplexity` both need this table, not the raw checkpoint --
   this replaces keeping a second copy of the reference resident in GPU
   memory during their training. Takes roughly 15-25 h. This can run while
   `attention`/`blade`/etc. use the 4-GPU `qos=gpu` lane, since it's a
   separate quota.

5. Then launch `rho-1`, `perplexity`, `full-loss-control`, `attention`,
   `blade`, and the two random-control seeds. **Not in parallel:** `qos=gpu`
   caps each user at 4 GPUs *in total*, and every one of these arms needs a
   full 4-GPU node, so only one of these training jobs (yours, or a
   concurrent job of anyone else sharing this quota) can actually run at a
   time; the rest queue behind it. Budget accordingly -- `blade` alone is
   about 36 h, so set `TRAIN_TIME=2-00:00:00` for it (the 24 h default is
   too short).

Restage before switching arms only if `ready.json` doesn't already cover the
arm you're launching (every training arm shares the same one).

## Smoke tests

Set `EDULLM_LOCAL=1` before `launch.sh` to bypass the 4-GPU production
topology assertion for a smoke run at a different GPU count (also set
`TASK_LOSS_NPROC` if you need eval there too). Production launches never set
this: every reported run uses the full 4-GPU contract.
