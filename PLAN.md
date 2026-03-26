# Experiment Pipeline Plan

**Created**: 2026-03-26
**QoS Deadline**: 2026-04-02 (q_m1x16 expires)
**Epochs**: 3000 per experiment (~5.7hr/job at 6.9s/epoch)
**Concurrency**: 2 jobs max on TC2

---

## Step 0: Config Fixes (before any experiments)

- [x] **0.1** Update `slurm/job_train.sh`: QoS `normal` → `q_m1x16`, wall time `06:00:00` → `16:00:00`
- [x] **0.2** Update `slurm/job_eval.sh`: QoS `normal` → `q_m1x16`, DATA_DIR default → `data/processed_shapenet`
- [x] **0.3** Update `configs/config.yaml`: data_dir → `data/processed_shapenet`, gt_mesh_dir → `data/processed_shapenet/gt_meshes`, epochs → `3000`
- [x] **0.4** Update `slurm/submit.sh`: pass `DATA_DIR="data/processed_shapenet"` to eval job
- [x] **0.5** Update `src/evaluate.py`: argparse default data_dir → `data/processed_shapenet`
- [ ] **0.6** Commit changes locally
- [ ] **0.7** Sync updated files to TC2 via scp
- [ ] **0.8** Verify on TC2: grep QoS, grep data paths, check data exists
- [ ] **0.9** Delete QoS demo artifacts on TC2 (~1.6GB): `processed_qos_demo/`, `raw_qos_demo/`, `QOS-DEMO/`, `QOS-DEMO-SHAPENET/`

## Step 1: Phase 1 — Baselines (2 jobs concurrent)

- [ ] **1.1** Submit EXP-01 seed 42: `./slurm/submit.sh EXP-01 42 "supervision_ratio=1.0 use_eikonal=false use_pe=false"`
- [ ] **1.2** Submit EXP-02 seed 42: `./slurm/submit.sh EXP-02 42 "supervision_ratio=1.0 use_eikonal=true use_pe=false"`
- [ ] **1.3** Monitor: check logs after ~30 min for first 10 epochs, verify data loaded correctly
- [ ] **1.4** When EXP-01 completes: run `/log-experiment EXP-01`
- [ ] **1.5** When EXP-02 completes: run `/log-experiment EXP-02`

## Step 2: Phase 2 — Label Reduction (2 at a time)

- [ ] **2.1** Submit EXP-03 seed 42: `./slurm/submit.sh EXP-03 42 "supervision_ratio=0.5 use_eikonal=true use_pe=false"`
- [ ] **2.2** Submit EXP-04 seed 42: `./slurm/submit.sh EXP-04 42 "supervision_ratio=0.1 use_eikonal=true use_pe=false"`
- [ ] **2.3** When EXP-03 completes: run `/log-experiment EXP-03`
- [ ] **2.4** When EXP-04 completes: run `/log-experiment EXP-04`
- [ ] **2.5** Submit EXP-05 seed 42: `./slurm/submit.sh EXP-05 42 "supervision_ratio=0.05 use_eikonal=true use_pe=false"`
- [ ] **2.6** When EXP-05 completes: run `/log-experiment EXP-05`

## Step 3: Phase 3 — Positional Encoding (2 at a time)

- [ ] **3.1** Submit EXP-06 seed 42: `./slurm/submit.sh EXP-06 42 "supervision_ratio=0.1 use_eikonal=true use_pe=true"`
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
