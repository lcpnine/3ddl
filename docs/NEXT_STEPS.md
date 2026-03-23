# Next Steps — Semi-Supervised DeepSDF Project

**Status as of 2026-03-23**: Pipeline validated with EXP-01 on parametric meshes (10K points). ShapeNet access granted. Next: watertight scan, ShapeNet preprocessing, and proper baseline run.

---

## Immediate Actions (before running more experiments)

### 1. Watertight mesh scan on ShapeNet categories
Run `trimesh.is_watertight` on target categories (chair `04256520`, airplane `02691156`, table `04379243`). Log counts to determine usable meshes per category. Target: 100 shapes/category (60 train / 20 val / 20 test).

### 2. Preprocess ShapeNet data (250K points)
```bash
python scripts/preprocess.py --mesh_dir /path/to/ShapeNetCore/<category_id> --output_dir data/processed --n_sup 250000 --n_unsup 250000
```
Upload processed data to TC2:
```bash
scp -r data/processed/ tc2:~/3ddl/data/processed/
```

### 3. Re-upload updated scripts to TC2
Several local files were updated (env names, PYTHONPATH, etc.) after the first upload:
```bash
scp -r src/ scripts/ configs/ slurm/ tc2:~/3ddl/
```

### 4. Send QoS extension email
Send the drafted email to `ccdsgpu-tc@ntu.edu.sg` requesting Extended QoS (24hr wall time). Fill in your name and student ID. Reference Job ID 14823 as evidence.

### 5. Re-run EXP-01 with ShapeNet data (proper baseline)
```bash
ssh tc2 "cd ~/3ddl && ./slurm/submit.sh EXP-01 42"
```
This establishes the true baseline with production-quality ShapeNet data. Compare against the parametric mesh results:
- Parametric (10K pts): CD=0.064, NC=0.705
- ShapeNet (250K pts): expected significant improvement (real geometry + better surface coverage)

---

## Experiment Execution Plan

Run experiments in this order. Each experiment needs 3 seeds (42, 123, 456) for statistical confidence.

### Phase 1: Baselines
| ID | Command | Purpose |
|----|---------|---------|
| EXP-01 | `./slurm/submit.sh EXP-01 42` | Fully-supervised, no Eikonal (re-run with 250K data) |
| EXP-02 | `./slurm/submit.sh EXP-02 42 "use_eikonal=true"` | Full supervision + Eikonal |

**Key question**: Does Eikonal help even with full supervision? Compare EXP-02 vs EXP-01.

### Phase 2: Label Reduction
| ID | Command | Purpose |
|----|---------|---------|
| EXP-03 | `./slurm/submit.sh EXP-03 42 "supervision_ratio=0.5 use_eikonal=true"` | 50% labels + Eikonal |
| EXP-04 | `./slurm/submit.sh EXP-04 42 "supervision_ratio=0.1 use_eikonal=true"` | 10% labels + Eikonal (KEY) |
| EXP-05 | `./slurm/submit.sh EXP-05 42 "supervision_ratio=0.05 use_eikonal=true"` | 5% labels + Eikonal |

**Key question**: How far can we reduce labels before quality collapses? EXP-04 is the main data point.
**Note**: If EXP-04 CD coefficient of variation > 0.2 across 3 seeds, expand to 5 seeds (add 789, 101).

### Phase 3: Positional Encoding
| ID | Command | Purpose |
|----|---------|---------|
| EXP-06 | `./slurm/submit.sh EXP-06 42 "supervision_ratio=0.1 use_eikonal=true use_pe=true pe_levels=6"` | 10% + Eikonal + PE (KEY) |
| EXP-07 | `./slurm/submit.sh EXP-07 42 "supervision_ratio=0.05 use_eikonal=true use_pe=true pe_levels=6"` | 5% + Eikonal + PE |
| EXP-09 | `./slurm/submit.sh EXP-09 42 "supervision_ratio=1.0 use_eikonal=true use_pe=true pe_levels=6"` | Full + Eikonal + PE (ceiling) |

**Key question**: Does PE improve label efficiency? Compare EXP-06 vs EXP-04, EXP-07 vs EXP-05.
**Note**: If EXP-06 CD CV > 0.2, expand to 5 seeds.

### Phase 4: Advanced Regularization
| ID | Command | Purpose |
|----|---------|---------|
| EXP-08 | `./slurm/submit.sh EXP-08 42 "supervision_ratio=0.1 use_eikonal=true use_pe=true pe_levels=6 lambda_2nd=0.01 batch_size=8192"` | + second-order loss |

**Key question**: Does L_2nd provide additional benefit on top of Eikonal + PE?

---

## After Each Experiment

1. **Check training output**: `ssh tc2 "tail -20 ~/3ddl/logs/output_<JOB_NAME>_<JOB_ID>.out"`
2. **Log results**: Run `/log-experiment EXP-XX`
3. **Disk check**: Run `/disk-check` after every 3 experiments (100GB quota)
4. **Diagnose if needed**: If loss looks abnormal, run `/diagnose-training EXP-XX seed`

## After All Experiments

1. **Generate figures**: Run `/generate-figures label-efficiency` for the label efficiency curve
2. **Ablation chart**: Bar chart comparing EXP-04 vs EXP-06 (Eikonal only vs Eikonal+PE)
3. **Write report**: Run `/write-report-section discussion`

---

## Known Issues & Workarounds

| Issue | Workaround |
|-------|-----------|
| ~~Thingi10K API down~~ | Resolved: ShapeNet access granted (2026-03-23). Run watertight scan + preprocess ShapeNet data. |
| IoU extremely slow on CPU | Use `--skip_iou` flag or `SKIP_IOU=1` env var; compute IoU only for final results |
| MC resolution 256 slow on CPU | Use `mc_resolution: 64` for dev, `128` for production |
| Python output buffered in SLURM | Fixed: `PYTHONUNBUFFERED=1` in all job scripts |
| Login node OOM for preprocessing | Use `slurm/job_preprocess.sh` instead of running on login node |
| Divergence false positives | Threshold updated from 0.5 → 0.95 |

---

## TC2 Cluster Quick Reference

```bash
# Connect
ssh tc2

# Check jobs
squeue -u yutaek001
sacct -j <JOB_ID> --format=State,Elapsed,MaxRSS

# Job efficiency
seff <JOB_ID>

# Check disk usage
du -sh ~/3ddl/

# Cancel a job
scancel <JOB_ID>

# View logs
tail -f ~/3ddl/logs/output_<JOB_NAME>_<JOB_ID>.out
```
