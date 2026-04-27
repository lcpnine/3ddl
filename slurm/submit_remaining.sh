#!/bin/bash
# Submit remaining manifest entries (skip already-submitted rows).
# Polls squeue and waits for a free slot before each submission.
# Run in tmux: tmux new-session -d -s reruns "bash ~/3ddl/slurm/submit_remaining.sh"

set -uo pipefail
cd "$HOME/3ddl"

MANIFEST=slurm/rerun_manifest.txt
JOB_IDS_FILE=slurm/rerun_job_ids.txt
SKIP_LINES=3   # EXP-01/02/03 already done
MAX_JOBS=2     # QoS limit

log() { echo "[$(date +%H:%M:%S)] $*"; }

tail -n +$((SKIP_LINES + 1)) "$MANIFEST" | while read -r checkpoint_mode expected_n_total result_path; do
    exp_dir=$(dirname "$result_path")
    suffix=$(basename "$result_path")
    name=$(echo "$exp_dir" | sed 's|experiments/||; s|/|_|g')

    # Wait until job count drops below MAX_JOBS
    while true; do
        count=$(squeue -u yutaek001 -h | wc -l)
        if [ "$count" -lt "$MAX_JOBS" ]; then
            break
        fi
        log "Queue full ($count jobs). Waiting 60s before submitting $name..."
        sleep 60
    done

    job_id=$(CHECKPOINT_MODE=$checkpoint_mode OUTPUT_SUFFIX=$suffix EXP_DIR=$exp_dir \
        sbatch --parsable slurm/job_eval.sh)
    log "Submitted $job_id $name ($checkpoint_mode, n=$expected_n_total)"
    echo "$job_id $name $exp_dir $checkpoint_mode $expected_n_total" >> "$JOB_IDS_FILE"
done

log "All remaining jobs submitted."
