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
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=yt3dl_eval
#SBATCH --output=/home/msai/yutaek001/3ddl/logs/output_%x_%j.out
#SBATCH --error=/home/msai/yutaek001/3ddl/logs/error_%x_%j.err

# --- Environment Setup (TC2 Guide) ---
module load anaconda
eval "$(conda shell.bash hook)"
conda activate yt3dl

# --- Config ---
EXP_DIR="${EXP_DIR:-experiments/debug/seed42}"
DATA_DIR="${DATA_DIR:-data/processed_shapenet}"
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

SKIP_IOU="${SKIP_IOU:-1}"
MC_RES="${MC_RES:-256}"
EVAL_SPLIT="${EVAL_SPLIT:-train}"
SPHERE_CLIP="${SPHERE_CLIP:-1}"
TTO_N_ITERS="${TTO_N_ITERS:-800}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-auto}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-results.json}"

SKIP_IOU_ARG=""
if [ "$SKIP_IOU" = "1" ]; then
    SKIP_IOU_ARG="--skip_iou"
fi

SPHERE_CLIP_ARG="--sphere_clip"
if [ "$SPHERE_CLIP" = "0" ]; then
    SPHERE_CLIP_ARG="--no_sphere_clip"
fi

python src/evaluate.py \
    --exp_dir "$EXP_DIR" \
    --data_dir "$DATA_DIR" \
    --output "$EXP_DIR/$OUTPUT_SUFFIX" \
    --voxel_res $VOXEL_RES \
    --mc_resolution $MC_RES \
    --eval_split "$EVAL_SPLIT" \
    --checkpoint_mode "$CHECKPOINT_MODE" \
    --tto_n_iters $TTO_N_ITERS \
    $SPHERE_CLIP_ARG \
    $SKIP_IOU_ARG

EXIT_CODE=$?
echo "============================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
