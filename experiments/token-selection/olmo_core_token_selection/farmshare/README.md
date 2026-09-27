# Token selection 370M on FarmShare (4 × L40S)

Every arm below runs on one 4×L40S node, through
`token_selection_entrypoint.py`. There is no RunPod path and no AWS/S3 code in
this tree; corpora are pinned local FarmShare directories, verified once and
bound into `ready.json` by `stage_local.py`, and reference checkpoints are
trained by the `hq-reference`/`instruct-reference` arms below, not downloaded.

Arms via `ARM`: `hq-reference`, `instruct-reference`, `full-loss-control`,
`rho-1`, `rel-ema-exp`, `perplexity`, `attention`, `blade`, `random-control`,
`random-control-seed69`.

```bash
cd /mnt/c/alpha_ai/OLMo-core-token-selection-370m
LOCAL_REPO="$(pwd)" ARM=attention bash .edullm/farmshare/submit_from_laptop.sh
```

## Staging order

1. `STAGE_MODE=corpora` (the default) verifies the three pinned corpus
   directories and writes `ready.json`. Run once; every arm shares it.
2. Train `hq-reference` and `instruct-reference` first. `rho-1` and
   `perplexity` need their materialized checkpoints.
3. `STAGE_MODE=references` materializes and averages those checkpoints into
   `ready.json`'s `references` map, keyed by the symbolic contract names in
   `token_selection_370m/arms.py`:

   ```bash
   STAGE_MODE=references SKIP_TRAIN=1 bash .edullm/farmshare/submit_from_laptop.sh
   ```

4. Then launch `rho-1`, `perplexity`, `full-loss-control`, `attention`,
   `blade`, and the two random-control seeds; they can all run in parallel.

Restage before switching arms only if `ready.json` doesn't already cover the
arm you're launching (every training arm shares the same one).

## Smoke tests

Set `EDULLM_LOCAL=1` before `launch.sh` to bypass the 4-GPU production
topology assertion for a smoke run at a different GPU count (also set
`TASK_LOSS_NPROC` if you need eval there too). Production launches never set
this: every reported run uses the full 4-GPU contract.
