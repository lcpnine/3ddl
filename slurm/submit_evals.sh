#!/bin/bash
# Queue-polling submitter for the post-training eval phase.
#
# After all trainings in slurm/retrain_manifest.txt have completed, submit
# one slurm/job_eval.sh per EXP/seed. Respects q_m1x16 2-submitted cap.
#
# Usage:
#   bash slurm/submit_evals.sh
#
# Output:
#   logs/eval_submit.log
#   slurm/eval_job_ids.txt  (JOB_ID EXP seed)

set -euo pipefail

cd "$HOME/3ddl"
mkdir -p logs

MANIFEST="${MANIFEST:-slurm/retrain_manifest.txt}"
JOB_IDS_FILE="slurm/eval_job_ids.txt"
LOG="logs/eval_submit.log"
MAX_JOBS="${MAX_JOBS:-2}"

[ -f "$MANIFEST" ] || { echo "ABORT: $MANIFEST not found"; exit 1; }

{
    echo
    echo "=============================================="
    echo "submit_evals.sh starting at $(date)"
    echo "=============================================="
} | tee -a "$LOG"

while IFS=$'\t' read -r EXP_ID SEED OVERRIDES; do
    [ -z "$EXP_ID" ] && continue
    [[ "$EXP_ID" =~ ^# ]] && continue

    EXP_DIR="experiments/${EXP_ID}/seed${SEED}"
    if [ ! -f "$EXP_DIR/checkpoints/latest.pt" ] && [ ! -f "$EXP_DIR/checkpoints/best.pt" ]; then
        echo "[$(date +%H:%M:%S)] SKIP: $EXP_DIR has no checkpoint (training not complete?)" | tee -a "$LOG"
        continue
    fi

    # Older train.py did not save train_shapes.json. Reconstruct it using the
    # same sorted+shuffle logic so evaluate.py doesn't fall back to mismatched
    # sorted order. Uses dataset.py's default seed=42 (the only behavior old
    # train.py could trigger — it never passed seed= to SDFDataset).
    if [ ! -f "$EXP_DIR/train_shapes.json" ]; then
        echo "[$(date +%H:%M:%S)] Reconstructing train_shapes.json for $EXP_DIR" | tee -a "$LOG"
        python -c "
import glob, os, random, json, yaml
cfg = yaml.safe_load(open('$EXP_DIR/config.yaml'))
ratio = cfg.get('supervision_ratio', 1.0)
ratio_str = f'ratio_{ratio:.2f}'.replace('.', 'p')
all_files = sorted(glob.glob(f'data/processed_shapenet/{ratio_str}/*.npz'))
# dataset.py SDFDataset default seed=42; old train.py never passed seed=
rng = random.Random(42)
rng.shuffle(all_files)
names = [os.path.splitext(os.path.basename(f))[0] for f in all_files]
with open('$EXP_DIR/train_shapes.json', 'w') as f:
    json.dump(names, f)
print(f'  wrote {len(names)} shapes')
"
    fi

    while true; do
        count=$(squeue -u yutaek001 -h 2>/dev/null | wc -l | awk '{print $1}')
        if [ "$count" -lt "$MAX_JOBS" ]; then break; fi
        echo "[$(date +%H:%M:%S)] Queue full ($count/$MAX_JOBS). Waiting 60s before eval $EXP_ID/seed$SEED..." | tee -a "$LOG"
        sleep 60
    done

    echo "[$(date +%H:%M:%S)] Submitting EVAL $EXP_DIR (MC_RES=${MC_RES:-96})" | tee -a "$LOG"
    eval_job=$(EXP_DIR="$EXP_DIR" \
        DATA_DIR="data/processed_shapenet" \
        MC_RES="${MC_RES:-96}" \
        sbatch --parsable \
            --job-name="${EXP_ID}_s${SEED}_eval" \
            slurm/job_eval.sh)
    echo "  eval_job=$eval_job" | tee -a "$LOG"
    echo "$eval_job $EXP_ID $SEED" >> "$JOB_IDS_FILE"
done < "$MANIFEST"

echo "[$(date +%H:%M:%S)] All evals submitted." | tee -a "$LOG"
