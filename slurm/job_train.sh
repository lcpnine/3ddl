#!/bin/bash
#===============================================================
# DeepSDF Training Job — CCDS GPU Cluster TC2
#
# Usage:
#   sbatch slurm/job_train.sh
#   CONFIG_FILE=exp01_seed42.yaml sbatch slurm/job_train.sh
#   EXP_ID=EXP-01 SEED=42 sbatch slurm/job_train.sh
#===============================================================
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=25G
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --job-name=yt3dl_train
#SBATCH --output=/home/msai/yutaek001/3ddl/logs/output_%x_%j.out
#SBATCH --error=/home/msai/yutaek001/3ddl/logs/error_%x_%j.err

# --- Environment Setup (TC2 Guide) ---
module load anaconda
module load cuda/12.8.0
eval "$(conda shell.bash hook)"
conda activate yt3dl

# --- Config ---
CONFIG_FILE="${CONFIG_FILE:-config.yaml}"
OVERRIDES="${OVERRIDES:-}"

echo "============================================"
echo "DeepSDF Training Job"
echo "============================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Config:     $CONFIG_FILE"
echo "Overrides:  $OVERRIDES"
echo "Start time: $(date)"
echo "============================================"

# --- Run Training ---
cd $HOME/3ddl
export PYTHONPATH=src:$PYTHONPATH
export PYTHONUNBUFFERED=1

if [ -n "$OVERRIDES" ]; then
    python src/train.py --config "configs/$CONFIG_FILE" --config_override $OVERRIDES
else
    python src/train.py --config "configs/$CONFIG_FILE"
fi

EXIT_CODE=$?
echo "============================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
