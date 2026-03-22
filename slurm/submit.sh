#!/bin/bash
#===============================================================
# Submit train + eval as dependency chain on TC2.
#
# Usage:
#   ./slurm/submit.sh EXP-01 42
#   ./slurm/submit.sh EXP-04 42 "supervision_ratio=0.1 use_eikonal=true"
#===============================================================

set -euo pipefail

EXP_ID="${1:?Usage: submit.sh <EXP_ID> <SEED> [OVERRIDES]}"
SEED="${2:?Usage: submit.sh <EXP_ID> <SEED> [OVERRIDES]}"
OVERRIDES="${3:-}"

# Construct config override string
BASE_OVERRIDES="exp_name=${EXP_ID} seed=${SEED}"
if [ -n "$OVERRIDES" ]; then
    ALL_OVERRIDES="$BASE_OVERRIDES $OVERRIDES"
else
    ALL_OVERRIDES="$BASE_OVERRIDES"
fi

EXP_DIR="experiments/${EXP_ID}/seed${SEED}"

echo "============================================"
echo "Submitting: ${EXP_ID} seed=${SEED}"
echo "Overrides:  ${ALL_OVERRIDES}"
echo "Exp dir:    ${EXP_DIR}"
echo "============================================"

# Ensure logs directory exists
mkdir -p logs

# Submit training job
TRAIN_JOB=$(
    OVERRIDES="$ALL_OVERRIDES" \
    sbatch --parsable \
        --job-name="${EXP_ID}_s${SEED}_train" \
        slurm/job_train.sh
)
echo "Training job submitted: $TRAIN_JOB"

# Submit evaluation job (chained after training)
EVAL_JOB=$(
    EXP_DIR="$EXP_DIR" \
    sbatch --parsable \
        --dependency=afterok:${TRAIN_JOB} \
        --job-name="${EXP_ID}_s${SEED}_eval" \
        slurm/job_eval.sh
)
echo "Evaluation job submitted: $EVAL_JOB (depends on $TRAIN_JOB)"

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Train log:    logs/output_${EXP_ID}_s${SEED}_train_${TRAIN_JOB}.out"
echo "Eval log:     logs/output_${EXP_ID}_s${SEED}_eval_${EVAL_JOB}.out"
