#!/bin/bash
#SBATCH --job-name=mesh_extract
#SBATCH --output=slurm/logs/mesh_extract_%j.out
#SBATCH --error=slurm/logs/mesh_extract_%j.err
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=q_m1x16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=00:30:00

set -eo pipefail

cd /home/msai/yutaek001/3ddl

module load anaconda
module load cuda/12.8.0
eval "$(conda shell.bash hook)"
conda activate yt3dl

SHAPES="airplane_0003 chair_0003 table_0003"
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
  echo "Extracting sample meshes from: $exp"
  echo "================================================================"
  python3 scripts/extract_sample_meshes.py \
    --exp_dir "experiments/$exp" \
    --shapes $SHAPES \
    --mc_resolution 128
done

echo "DONE"
