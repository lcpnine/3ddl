#!/bin/bash
#===============================================================
# DeepSDF Evaluation Job — CCDS GPU Cluster TC2
#
# CPU-focused: no GPU requested.
# Chain after training via:
#   sbatch --dependency=afterok:$TRAIN_JOBID slurm/job_eval.sh
#
# Usage:
#   EXP_DIR=experiments/EXP-01/seed42 sbatch slurm/job_eval.sh
#===============================================================
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=yt3dl_eval
#SBATCH --output=/home/msai/yutaek001/3ddl/logs/output_%x_%j.out
#SBATCH --error=/home/msai/yutaek001/3ddl/logs/error_%x_%j.err

# --- Environment Setup (TC2 Guide) ---
module load anaconda
eval "$(conda shell.bash hook)"
conda activate yt3dl

# --- Config ---
EXP_DIR="${EXP_DIR:-experiments/debug/seed42}"
DATA_DIR="${DATA_DIR:-data/processed}"
VOXEL_RES="${VOXEL_RES:-128 256}"

echo "============================================"
echo "DeepSDF Evaluation Job"
echo "============================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Exp dir:    $EXP_DIR"
echo "Data dir:   $DATA_DIR"
echo "Voxel res:  $VOXEL_RES"
echo "Start time: $(date)"
echo "============================================"

# --- Run Evaluation ---
cd $HOME/3ddl
export PYTHONPATH=src:$PYTHONPATH
export PYTHONUNBUFFERED=1

SKIP_IOU="${SKIP_IOU:-}"

python src/evaluate.py \
    --exp_dir "$EXP_DIR" \
    --data_dir "$DATA_DIR" \
    --output "$EXP_DIR/results.json" \
    --voxel_res $VOXEL_RES \
    ${SKIP_IOU:+--skip_iou}

EXIT_CODE=$?
echo "============================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
