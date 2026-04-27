#!/bin/bash
#SBATCH --job-name=mesh_extract_all
#SBATCH --output=slurm/logs/mesh_extract_all_%j.out
#SBATCH --error=slurm/logs/mesh_extract_all_%j.err
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=q_m1x16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=04:00:00

set -eo pipefail

cd /home/msai/yutaek001/3ddl

module load anaconda
module load cuda/12.8.0
eval "$(conda shell.bash hook)"
conda activate yt3dl

EXPERIMENTS=(
  "EXP-01/seed42"
  "EXP-02/seed42"
  "EXP-03/seed42"
  "EXP-04/seed42"
  "EXP-05/seed42"
  "EXP-06/seed42"
  "EXP-11/seed42"
)

for exp in "${EXPERIMENTS[@]}"; do
  echo "================================================================"
  echo "[$(date)] Extracting all meshes from: $exp"
  echo "================================================================"
  python3 scripts/extract_sample_meshes.py \
    --exp_dir "experiments/$exp" \
    --all_shapes \
    --out_subdir all_reconstructions \
    --mc_resolution 96
done

echo "[$(date)] DONE"
