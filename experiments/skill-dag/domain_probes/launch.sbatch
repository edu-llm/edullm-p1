#!/usr/bin/env bash
# Launcher for the 31 domain-probe runs on one FarmShare L40S each.
#
#   MODE=smoke  sbatch --job-name=domsmoke ... launch.sbatch   # 60-step check, picks the microbatch
#   MODE=array  sbatch --job-name=domprobe --array=0-30%1 ... launch.sbatch
#
# Required env (set by the submit wrapper): RUN_ROOT (holds code/, runs/, hf-cache/),
# DATA_ROOT (<domain>/train-*.u32le.bin), VENV. Optional: WANDB_ENV (file with WANDB_API_KEY).
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
set -uo pipefail

: "${RUN_ROOT:?}" "${DATA_ROOT:?}" "${VENV:?}"
MODE="${MODE:-array}"
CODE="$RUN_ROOT/code/experiments/skill-dag/domain_probes"
OUT="$RUN_ROOT/runs"
mkdir -p "$OUT" "$RUN_ROOT/logs"

source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load python/3.12.3 2>/dev/null || true
module load cuda/12.9.0 2>/dev/null || module load cuda/12.4.0 2>/dev/null || true
source "$VENV/bin/activate"
export HF_HOME="$RUN_ROOT/hf-cache"
export OLMO_FLASH_ATTENTION=0 OLMO_FUSED_LOSS=0 OLMO_ACTIVATION_CHECKPOINTING=0
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CODE_COMMIT="$(cat "$RUN_ROOT/code/CODE_COMMIT" 2>/dev/null || echo unknown)"
export MASTER_ADDR=127.0.0.1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
export MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 2000))
if [[ -n "${WANDB_ENV:-}" && -f "$WANDB_ENV" ]]; then set -a; source "$WANDB_ENV"; set +a; fi
echo "[$(date -Is)] mode=$MODE host=$(hostname) job=${SLURM_JOB_ID:-} task=${SLURM_ARRAY_TASK_ID:-} commit=$CODE_COMMIT"
nvidia-smi -L

run_name_for_index() {
  python - "$1" <<'PY'
import json, sys
print(json.load(open("runs.json"))["runs"][int(sys.argv[1])]["name"])
PY
}

if [[ "$MODE" == "smoke" ]]; then
  # 60 steps with an eval at step 60 (all test items), once per microbatch. The array reads the
  # chosen microbatch from $RUN_ROOT/microbatch.txt; any failed check fails the job, and the
  # array is submitted with --dependency=afterok so it never starts on a broken setup.
  cd "$CODE"
  SMOKE_OUT="$RUN_ROOT/smoke"; rm -rf "$SMOKE_OUT"; mkdir -p "$SMOKE_OUT"
  best_mb=""; best_s=999999
  for mb in 4 8 16; do
    d="$SMOKE_OUT/mb$mb"
    ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SMOKE_OUT/mem_mb$mb.txt"; sleep 5; done ) &
    sampler=$!
    t0=$(date +%s)
    python train_60m.py --run mix01 --data-root "$DATA_ROOT" --out-dir "$d" --microbatch "$mb" \
      --steps 60 --eval-interval 60 --wandb-mode disabled > "$SMOKE_OUT/mb$mb.log" 2>&1
    rc=$?; t1=$(date +%s); kill $sampler 2>/dev/null
    mem=$(sort -n "$SMOKE_OUT/mem_mb$mb.txt" | tail -1)
    echo "microbatch=$mb rc=$rc seconds=$((t1 - t0)) peak_mem_mib=$mem"
    if [[ $rc -ne 0 ]]; then
      if [[ "$mb" == "4" ]]; then echo "FAILED at the default microbatch"; tail -30 "$SMOKE_OUT/mb$mb.log"; exit 1; fi
      continue  # a larger microbatch may OOM; that only rules it out
    fi
    python - "$d/mix01/task_loss.jsonl" <<'PY' || exit 1
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
assert rows and rows[-1]["step"] == 60 and len(rows[-1]["task_loss_bpb"]) == 6, rows
print("step-60 eval ok:", {k[:14]: round(v, 3) for k, v in rows[-1]["task_loss_bpb"].items()})
PY
    if [[ -n "$mem" && "$mem" -lt 40000 && $((t1 - t0)) -lt $best_s ]]; then best_s=$((t1 - t0)); best_mb=$mb; fi
    rm -rf "$d/mix01/checkpoints"
  done
  [[ -n "$best_mb" ]] || { echo "no microbatch fit in memory"; exit 1; }
  echo "$best_mb" > "$RUN_ROOT/microbatch.txt"
  echo "chosen microbatch: $best_mb"
  exit 0
fi

# array mode: one run per task, idempotent
cd "$CODE"
name="$(run_name_for_index "${SLURM_ARRAY_TASK_ID:?}")"
final="$OUT/$name/task_loss.jsonl"
if [[ -f "$final" ]] && python - "$final" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
sys.exit(0 if rows and rows[-1]["step"] == 1440 and len(rows[-1]["task_loss_bpb"]) == 6 else 1)
PY
then echo "[$name] already complete; skipping"; exit 0; fi
mb="$(cat "$RUN_ROOT/microbatch.txt" 2>/dev/null || echo 4)"
echo "[$name] microbatch=$mb"
python train_60m.py --run "$name" --data-root "$DATA_ROOT" --out-dir "$OUT" --microbatch "$mb" \
  --wandb-mode "${WANDB_MODE:-online}"
rc=$?
echo "[$(date -Is)] [$name] exit $rc"
exit $rc
