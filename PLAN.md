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
- [ ] **3.2** Submit EXP-07 seed 42: `./slurm/submit.sh EXP-07 42 "supervision_ratio=0.05 use_eikonal=true use_pe=true"`
- [ ] **3.3** When EXP-06 completes: run `/log-experiment EXP-06`
- [ ] **3.4** When EXP-07 completes: run `/log-experiment EXP-07`
- [ ] **3.5** Submit EXP-09 seed 42: `./slurm/submit.sh EXP-09 42 "supervision_ratio=1.0 use_eikonal=true use_pe=true"`
- [ ] **3.6** When EXP-09 completes: run `/log-experiment EXP-09`

## Step 4: Phase 4 — Advanced Regularization

- [ ] **4.1** Submit EXP-08 seed 42: `./slurm/submit.sh EXP-08 42 "supervision_ratio=0.1 use_eikonal=true use_pe=true lambda_2nd=0.1 batch_size=8192"`
- [ ] **4.2** When EXP-08 completes: run `/log-experiment EXP-08`
- [ ] **4.3** Run `/disk-check`

## Step 5: Multi-Seed Runs

- [ ] **5.1** Check EXP-04 CD coefficient of variation. If CV > 0.2, plan 5 seeds; otherwise 3.
- [ ] **5.2** Submit EXP-04 seed 123: `./slurm/submit.sh EXP-04 123 "supervision_ratio=0.1 use_eikonal=true use_pe=false"`
- [ ] **5.3** Submit EXP-04 seed 456: `./slurm/submit.sh EXP-04 456 "supervision_ratio=0.1 use_eikonal=true use_pe=false"`
- [ ] **5.4** Check EXP-06 CD coefficient of variation. If CV > 0.2, plan 5 seeds; otherwise 3.
- [ ] **5.5** Submit EXP-06 seed 123: `./slurm/submit.sh EXP-06 123 "supervision_ratio=0.1 use_eikonal=true use_pe=true"`
- [ ] **5.6** Submit EXP-06 seed 456: `./slurm/submit.sh EXP-06 456 "supervision_ratio=0.1 use_eikonal=true use_pe=true"`
- [ ] **5.7** Log all multi-seed results
- [ ] **5.8** If CV > 0.2 for either: submit seeds 789 and 101 as well
- [ ] **5.9** Run `/disk-check`

## Step 6: Wrap-Up

- [ ] **6.1** Run `/generate-figures` for label efficiency curve and ablation charts
- [ ] **6.2** Review all results in `experiments/experiment_log.md`
- [ ] **6.3** Clean up `data/shapenet_raw/` (~26GB) if disk needed

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
