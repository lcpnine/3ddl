# Experiment Pipeline Plan

**Created**: 2026-03-26
**QoS Deadline**: 2026-04-02 (q_m1x16 expires)
**Epochs**: 3000 per experiment (~5.7hr/job at 6.9s/epoch)
**Concurrency**: 2 jobs max on TC2 (includes pending jobs — submit.sh uses both slots per experiment)

**Strategy**: Submit training jobs directly via sbatch (not submit.sh) to run 2 trains concurrently. Run evals separately after training completes. Train-only command:
```
OVERRIDES="exp_name=EXP-XX seed=42 <overrides>" sbatch --job-name=EXP-XX_s42_train slurm/job_train.sh
```
Eval command (after training):
```
EXP_DIR=experiments/EXP-XX/seed42 sbatch --job-name=EXP-XX_s42_eval slurm/job_eval.sh
```

---

## Step 0: Config Fixes (before any experiments)

- [x] **0.1** Update `slurm/job_train.sh`: QoS `normal` → `q_m1x16`, wall time `06:00:00` → `16:00:00`
- [x] **0.2** Update `slurm/job_eval.sh`: QoS `normal` → `q_m1x16`, DATA_DIR default → `data/processed_shapenet`
- [x] **0.3** Update `configs/config.yaml`: data_dir → `data/processed_shapenet`, gt_mesh_dir → `data/processed_shapenet/gt_meshes`, epochs → `3000`
- [x] **0.4** Update `slurm/submit.sh`: pass `DATA_DIR="data/processed_shapenet"` to eval job
- [x] **0.5** Update `src/evaluate.py`: argparse default data_dir → `data/processed_shapenet`
- [x] **0.6** Commit changes locally (5692e88)
- [x] **0.7** Sync updated files to TC2 via scp
- [x] **0.8** Verify on TC2: QoS=q_m1x16, data_dir=processed_shapenet, epochs=3000, gt_meshes exist
- [x] **0.9** Delete QoS demo artifacts on TC2 (~1.6GB reclaimed)

## Step 0.10: Fix Model Collapse (CRITICAL)

**Diagnosis**: Both EXP-01 and EXP-02 collapsed to constant output (-0.00008 everywhere). Two root causes:

1. **`src/model.py:124`** — Last layer weight init `std=0.0001` too small → output ignores input, gradients starved
   - Fix: change to `std=1/sqrt(hidden_dim)` which is ~0.044 for hidden_dim=512

2. **`src/train.py:336-345`** — optimizer.zero_grad()/step() inside per-shape loop → 225 conflicting weight updates per epoch
   - Fix: accumulate gradients across all shapes, step once per epoch

- [x] **0.10a** Fix model.py last layer init: `std=0.0001` → `std=1.0/math.sqrt(hidden_dim)`
- [x] **0.10b** Fix train.py: move zero_grad before shape loop, move step/clip after shape loop, average losses
- [x] **0.10c** Commit (cb2ee58), sync to TC2
- [x] **0.10d** Cleaned experiment dirs, resubmitted EXP-01 (15491) + EXP-02 (15492). Early results: L_sdf 0.096→0.054 in 2 epochs (model is learning!)

## Step 1: Phase 1 — Baselines (1 GPU at a time, QoS limit)

**Note**: QoS allows 1 GPU, 10 CPUs, 2 submitted jobs. Only 1 job runs at a time (train=4CPU+1GPU, eval=8CPU — can't overlap due to CPU limit). Keep 2 jobs queued so they chain automatically.
**Actual epoch time**: ~2-6s/epoch (varies), 3000ep ≈ 4hr. Eval ≈ 2hr. Total per experiment ≈ 6hr.

- [x] **1.1** Submit EXP-01 seed 42 train (job 15347, running on TC2N01)
- [x] **1.2** Submit EXP-02 seed 42 train (job 15348, pending — auto-starts after EXP-01)
- [x] **1.3** Verified: 225 train shapes, L_sdf=0.0325 at ep10, 2.1s/epoch
- [x] **1.4** EXP-01 done: CD=0.0593, NC=0.5522 (300/300 shapes, 0 failures). Logged.
- [x] **1.5** EXP-02 done: CD=0.0543, NC=0.5920 (261/300 shapes, 39 failures). Eikonal improves CD 8.4%, NC 7.2% vs baseline.

## Step 2: Phase 2 — Label Reduction (2 at a time)

- [x] **2.1** Submit EXP-03 seed 42 (job 15572, completed on TC2N01)
- [x] **2.2** Submit EXP-04 seed 42 (job 15573, training complete; eval job 15662 submitted)
- [x] **2.3** EXP-03 done: CD=0.0534, NC=0.5924 (259/300 shapes). 50% labels matches full supervision with Eikonal.
- [x] **2.4** EXP-04 done: CD=0.0609, NC=0.5805 (263/300 shapes). 10% labels ≈ baseline CD, retains NC gain from Eikonal.
- [x] **2.5** Submit EXP-05 seed 42 (job 15687, training on TC2)
- [x] **2.6** EXP-05 done: CD=0.0509, NC=0.5766 (295/300 shapes). Best CD so far! 5% labels + Eikonal outperforms all higher-label experiments.

## Step 3: Phase 3 — Positional Encoding (2 at a time)

- [x] **3.1** Submit EXP-06 seed 42 (job 15966, queued behind EXP-05 eval)
- [x] **3.2** Submit EXP-07 seed 42 (job 16056, queued behind EXP-06 eval)
- [x] **3.3** EXP-06 done: CD=0.1515, NC=0.5059 (240/300 shapes, partial — disk quota). PE severely degrades quality at 10% supervision.
- [x] **3.4** EXP-07 done: CD=0.1448, NC=0.5074 (0/300 success). PE catastrophic at 5% supervision, similar to EXP-06.
- [x] **3.5** EXP-09 submitted and completed (job 16391 train, 16536 eval)
- [x] **3.6** EXP-09 done: CD=0.1450, NC=0.5031 (0/300 success). **PE L=6 catastrophic even at 100% labels.** All PE experiments (EXP-06/07/09) produce CD ~0.145 regardless of supervision.

## Step 4: Phase 4 — Advanced Regularization

- [x] **4.1** Submit EXP-08 seed 42: `./slurm/submit.sh EXP-08 42 "supervision_ratio=0.1 use_eikonal=true use_pe=true lambda_2nd=0.1 batch_size=8192"` (completed; see `experiments/experiment_log.md`)
- [x] **4.2** When EXP-08 completes: run `/log-experiment EXP-08` (completed; EXP-08 logged with CD `0.1443`, NC `0.5053`)
- [x] **4.3** Run `/disk-check` (completed on 2026-04-06; see `experiments/disk_usage_log.md`)

## Step 5: Multi-Seed Runs

- [x] **5.1** Check EXP-04 CD coefficient of variation. If CV > 0.2, plan 5 seeds; otherwise 3. Result: `CV(CD)=0.197 < 0.2`
- [x] **5.2** Submit EXP-04 seed 123: `./slurm/submit.sh EXP-04 123 "supervision_ratio=0.1 use_eikonal=true use_pe=false"` (completed)
- [x] **5.3** Submit EXP-04 seed 456: `./slurm/submit.sh EXP-04 456 "supervision_ratio=0.1 use_eikonal=true use_pe=false"` (completed)
- [x] **5.4** Check EXP-06 CD coefficient of variation. If CV > 0.2, plan 5 seeds; otherwise 3. Result: `CV(CD)=0.056 < 0.2`
- [x] **5.5** Submit EXP-06 seed 123: `./slurm/submit.sh EXP-06 123 "supervision_ratio=0.1 use_eikonal=true use_pe=true"` (completed)
- [x] **5.6** Submit EXP-06 seed 456: `./slurm/submit.sh EXP-06 456 "supervision_ratio=0.1 use_eikonal=true use_pe=true"` (completed)
- [x] **5.7** Log all multi-seed results
- [x] **5.8** If CV > 0.2 for either: submit seeds 789 and 101 as well. Not needed; no expansion triggered.
- [x] **5.9** Run `/disk-check` (completed on 2026-04-06; see `experiments/disk_usage_log.md`)

## Step 6: Wrap-Up

- [x] **6.1** Generate figures for label efficiency curve and ablation charts (completed on 2026-04-06; outputs in `experiments/figures/`, generated via `scripts/generate_figures.py`)
- [x] **6.2** Review all results in `experiments/experiment_log.md` (completed 2026-04-06; evaluator bug fixed, PE eval reruns submitted)
- [ ] **6.3** Clean up `data/shapenet_raw/` (~26GB) if disk needed

## Step 7: PE L=4 Follow-Up Sweep

- [x] **7.1** Submit EXP-10 seed 42 train: `OVERRIDES="exp_name=EXP-10 seed=42 supervision_ratio=1.0 use_eikonal=true use_pe=true pe_levels=4" sbatch --job-name=EXP-10_s42_train slurm/job_train.sh` (job 17755)
- [x] **7.2** Submit EXP-11 seed 42 train: `OVERRIDES="exp_name=EXP-11 seed=42 supervision_ratio=0.1 use_eikonal=true use_pe=true pe_levels=4" sbatch --job-name=EXP-11_s42_train slurm/job_train.sh` (job 17756)
- [x] **7.3** Submit EXP-12 seed 42 train: `OVERRIDES="exp_name=EXP-12 seed=42 supervision_ratio=0.05 use_eikonal=true use_pe=true pe_levels=4" sbatch --job-name=EXP-12_s42_train slurm/job_train.sh` (job 17929)
- [x] **7.4** EXP-10 train complete; user submitted eval job `18123`
- [x] **7.5** EXP-11 train complete; eval submitted as job `18180` on 2026-04-05 and completed
- [x] **7.6** EXP-10 eval finished (job `18123`); logged from `results.json` using per-shape metrics because the aggregate block is empty when all shapes are marked failed
- [x] **7.7** Submit EXP-11 eval after a queue slot frees: `EXP_DIR=experiments/EXP-11/seed42 sbatch --job-name=EXP-11_s42_eval slurm/job_eval.sh` (job `18180`)
- [x] **7.8** EXP-12 train finished; eval submitted as job `18181` on 2026-04-05 and completed
- [x] **7.9** Updated `experiments/experiment_log.md` with final EXP-10/11/12 metrics. Conclusion: PE L=4 remains catastrophic at 100%, 10%, and 5% supervision; lowering frequency from L=6 to L=4 does not rescue PE in this setup
- [x] **7.10** Fixed evaluator bug (`src/evaluate.py`): separated metric computation from mesh export so decimation failures no longer mark valid metrics as failed. Reverted QoS to `normal` in SLURM scripts. Cancelled eval rerun (job 18532) due to maintenance.

## Step 8: TC2 Maintenance Backup & Recovery (2026-04-06 → 04-07)

**Context**: TC2 scheduled downtime 2026-04-07 9:00–10:00am. All files backed up locally.

### 8.1 Backup (2026-04-06, before maintenance)
- [x] **8.1a** Back up `experiments/` (7.4 GB) → `tc2_backup/experiments/`
- [x] **8.1b** Back up `data/processed_shapenet/` (5.0 GB) → `tc2_backup/data_processed_shapenet/`
- [x] **8.1c** Back up `logs/` → `tc2_backup/logs/`
- [x] **8.1d** Back up `configs/` → `tc2_backup/configs/`
- [x] **8.1e** Back up `data/raw_shapenet/` (420 MB) → `tc2_backup/data_raw_shapenet/`
- [x] **8.1f** Skip `data/shapenet_raw/` (26 GB) — re-downloadable from ShapeNet
- [x] **8.1g** Source code already on local machine via git

### 8.2 Post-Maintenance Recovery (2026-04-07, after 10:00am)
- [x] **8.2a** Verify TC2 is back online: `ssh tc2 'hostname && squeue -u yutaek001'`
- [x] **8.2b** Verify project files intact: `ssh tc2 'ls -la /home/msai/yutaek001/3ddl/experiments/EXP-10/seed42/checkpoints/'`
- [x] **8.2c** Files intact after maintenance — no restore needed
- [x] **8.2d** Verify fixed `src/evaluate.py` and `slurm/job_eval.sh` on TC2; also updated QoS to `q_m1x16` (extended to 2026-04-09)
- [x] **8.2e** Submit eval reruns — 9 total (original 7 + EXP-08 + EXP-06/s42):
  - Batch 1: EXP-10/s42 (18670), EXP-11/s42 (18671)
  - Batch 2: EXP-12/s42 (18784), EXP-07/s42 (18785)
  - Batch 3: EXP-09/s42 (18855), EXP-06/s123 (18856)
  - Batch 4: EXP-06/s456 (19030), EXP-08/s42 (19031)
  - Batch 5: EXP-06/s42 (19118)
- [x] **8.2f** Collect all 9 results.json locally
- [x] **8.2g** Verified `n_shapes_evaluated=300` (0 failures) in all 9 results.json
- [x] **8.2h** Updated `experiments/experiment_log.md` — results table and detailed sections with corrected shape counts, CD/NC values, and 3-seed summary
- [ ] **8.2i** Commit all changes

---

## File Change Reference (Step 0)

| File | Line | Change |
|------|------|--------|
| `slurm/job_train.sh` | 11 | `--qos=normal` → `--qos=q_m1x16` |
| `slurm/job_train.sh` | 16 | `--time=06:00:00` → `--time=16:00:00` |
| `slurm/job_eval.sh` | 13 | `--qos=normal` → `--qos=q_m1x16` |
| `slurm/job_eval.sh` | 29 | default `data/processed` → `data/processed_shapenet` |
| `configs/config.yaml` | 6 | `data_dir: "data/processed"` → `"data/processed_shapenet"` |
| `configs/config.yaml` | 7 | `gt_mesh_dir: "data/processed/gt_meshes"` → `"data/processed_shapenet/gt_meshes"` |
| `configs/config.yaml` | 34 | `epochs: 1000` → `epochs: 3000` |
| `slurm/submit.sh` | ~46 | add `DATA_DIR="data/processed_shapenet"` to eval sbatch |
| `src/evaluate.py` | 414 | `default="data/processed"` → `default="data/processed_shapenet"` |
