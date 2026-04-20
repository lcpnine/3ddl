#!/bin/bash
# Queue-polling submitter for the full post-fix retrain matrix.
#
# Reads slurm/retrain_manifest.txt (one row per EXP/seed with OVERRIDES),
# submits chained train+eval via slurm/submit.sh, sleeps while the queue is
# full (q_m1x16: 2 jobs max submitted, 2 concurrent running).
#
# Usage:
#   bash slurm/submit_retrains.sh             # submit all rows
#   SKIP_LINES=1 bash slurm/submit_retrains.sh  # skip the first N rows (already submitted)
#
# Output:
#   logs/retrain_submit.log        — submission activity log
#   slurm/retrain_job_ids.txt      — TRAIN_JOB EVAL_JOB EXP seed

set -euo pipefail

cd "$HOME/3ddl"
mkdir -p logs

MANIFEST="${MANIFEST:-slurm/retrain_manifest.txt}"
JOB_IDS_FILE="slurm/retrain_job_ids.txt"
LOG="logs/retrain_submit.log"
MAX_JOBS="${MAX_JOBS:-2}"
SKIP_LINES="${SKIP_LINES:-0}"

[ -f "$MANIFEST" ] || { echo "ABORT: $MANIFEST not found"; exit 1; }

{
    echo
    echo "=============================================="
    echo "submit_retrains.sh starting at $(date)"
    echo "Manifest: $MANIFEST"
    echo "MAX_JOBS=$MAX_JOBS  SKIP_LINES=$SKIP_LINES"
    echo "=============================================="
} | tee -a "$LOG"

tail -n +$((SKIP_LINES + 1)) "$MANIFEST" | while IFS=$'\t' read -r EXP_ID SEED OVERRIDES; do
    [ -z "$EXP_ID" ] && continue
    [[ "$EXP_ID" =~ ^# ]] && continue

    while true; do
        count=$(squeue -u yutaek001 -h 2>/dev/null | wc -l | awk '{print $1}')
        if [ "$count" -lt "$MAX_JOBS" ]; then
            break
        fi
        echo "[$(date +%H:%M:%S)] Queue full ($count/$MAX_JOBS). Waiting 60s before submitting $EXP_ID/seed$SEED..." | tee -a "$LOG"
        sleep 60
    done

    echo "[$(date +%H:%M:%S)] Submitting TRAIN $EXP_ID seed=$SEED  OVERRIDES=\"$OVERRIDES\"" | tee -a "$LOG"

    ALL_OVERRIDES="exp_name=$EXP_ID seed=$SEED $OVERRIDES"
    train_job=$(OVERRIDES="$ALL_OVERRIDES" \
        sbatch --parsable \
            --job-name="${EXP_ID}_s${SEED}_train" \
            slurm/job_train.sh)
    echo "  train_job=$train_job" | tee -a "$LOG"
    echo "$train_job $EXP_ID $SEED" >> "$JOB_IDS_FILE"
done

echo "[$(date +%H:%M:%S)] All rows submitted." | tee -a "$LOG"
