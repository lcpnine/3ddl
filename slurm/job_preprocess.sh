#!/bin/bash
#===============================================================
# SDF Preprocessing Job — CCDS GPU Cluster TC2
#===============================================================
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=25G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=preprocess
#SBATCH --output=/home/msai/yutaek001/3ddl/logs/output_%x_%j.out
#SBATCH --error=/home/msai/yutaek001/3ddl/logs/error_%x_%j.err

module load anaconda
eval "$(conda shell.bash hook)"
conda activate yt3dl

cd $HOME/3ddl
export PYTHONUNBUFFERED=1

N_SUP="${N_SUP:-250000}"
N_UNSUP="${N_UNSUP:-250000}"

echo "============================================"
echo "SDF Preprocessing Job"
echo "============================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Sup points: $N_SUP"
echo "Unsup pts:  $N_UNSUP"
echo "Start time: $(date)"
echo "============================================"

python scripts/preprocess.py \
    --mesh_dir data/raw \
    --output_dir data/processed \
    --n_sup_points $N_SUP \
    --n_unsup_points $N_UNSUP

EXIT_CODE=$?
echo "============================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"
