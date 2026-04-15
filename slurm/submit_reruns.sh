#!/bin/bash
set -euo pipefail

cd "$HOME/3ddl"

MANIFEST=slurm/rerun_manifest.txt
JOB_IDS_FILE=slurm/rerun_job_ids.txt
FAIL_LOG=slurm/rerun_failed_jobs.txt

[ -f "$MANIFEST" ] || { echo "ABORT: $MANIFEST not found"; exit 1; }
[ "$(wc -l < "$MANIFEST")" = "16" ] || { echo "ABORT: expected 16 manifest lines"; exit 1; }
awk 'NF != 3 { exit 1 }' "$MANIFEST" || { echo "ABORT: malformed manifest"; exit 1; }

> "$JOB_IDS_FILE"
> "$FAIL_LOG"

while read -r checkpoint_mode expected_n_total result_path; do
    exp_dir=$(dirname "$result_path")
    suffix=$(basename "$result_path")
    name=$(echo "$exp_dir" | sed 's|experiments/||; s|/|_|g')
    job_id=$(CHECKPOINT_MODE=$checkpoint_mode OUTPUT_SUFFIX=$suffix EXP_DIR=$exp_dir \
        sbatch --parsable slurm/job_eval.sh)
    echo "$job_id $name $exp_dir $checkpoint_mode $expected_n_total" | tee -a "$JOB_IDS_FILE"
done < "$MANIFEST"
