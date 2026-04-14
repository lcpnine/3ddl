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
- [x] **8.2i** Commit all changes (8a41399)

## Step 9: PE Failure Root Cause Analysis (2026-04-15)

**Why**: All 7 PE experiments (EXP-06/07/08/09/10/11/12) produced CD ~0.14 regardless of
supervision level (5%–100%) or frequency (L=4, L=6). The mechanism was unknown. Diagnosing
the exact cause was needed to (a) explain the failure in the conference report and (b) determine
whether PE can be rescued for future work.

**4 hypotheses tested**:
1. Coordinate scaling wrong
2. PE frequency too high
3. Training/inference distribution mismatch (near-surface training vs full [-1,1]³ eval)
4. π factor in PE formula

- [x] **9.1** Check A — Training point coverage: confirmed unsupervised points have max radius = 1.000,
  zero coverage at r > 1.0. Cube corners (r = √3 ≈ 1.73) never sampled during training.
  Code: `scripts/preprocess.py:125-143` (unit sphere rejection sampling). Run locally on
  `data/processed/` parametric data (80 shapes).

- [x] **9.2** Check D — PE feature distance: `dist(surface, cube corner) = 2.83` out of max 8.49 for L=6;
  14 oscillation cycles at highest frequency (2⁵·π ≈ 100 rad/unit). Even near-surface
  movement of 0.05 units yields feature distance 2.62 — confirms PE is highly sensitive to
  coordinate position. Run locally using `src/model.py:FourierPositionalEncoding`.

- [x] **9.3** Wrote diagnostic scripts: `scripts/check_b_clipped_eval.py` and `scripts/check_c_diagonal.py`.
  Copied to TC2 via scp.

- [x] **9.4** Check C — SDF diagonal cross-section (TC2, EXP-09/seed42 vs EXP-02/seed42):
  PE model shows **14 sign changes** along diagonal toward cube corner (1,1,1) outside r>1.0;
  no-PE model shows 0 sign changes, monotone extrapolation. Oscillation period ≈ 0.063 units
  matches 2⁵·π ≈ 100 rad/unit exactly. These 14 zero-crossings are what marching cubes
  picks up as phantom surfaces → all shapes fail mesh extraction.

- [x] **9.5** Check B — Sphere-clipped evaluation (TC2, EXP-09/seed42, 5 airplane shapes, res=128):
  Masking OOD corners (set SDF=+1.0 for r>1.0 before marching cubes, no retraining):
  CD drops from **0.1926 → 0.0974 (−49%)**. NC unchanged (0.4953 → 0.4970).
  This single change — with the same trained weights — closes half the gap to EXP-02's CD=0.054.

- [x] **9.6** Documented in `experiments/experiment_log.md` (new section: "PE Failure Root Cause Analysis").

- [x] **9.7** Copied `experiments/figures/check_c_sdf_crosssection.png` from TC2 to local.

**Conclusion**: Root cause is **hypothesis #3 — training/inference distribution mismatch**.
The unsupervised sampler only covers the unit sphere; eval grid queries the full cube.
PE amplifies extrapolation errors at OOD corners into 14 sign-flipping oscillations per axis.
No-PE extrapolates incorrectly but monotonically → valid mesh. PE extrapolates with oscillations
→ phantom surfaces everywhere → mesh extraction fails.

**Frequency (hypothesis #2) is secondary**: Check B shows clipping alone halves CD with L=6
unchanged. The gap, not the frequency, is the root cause.

**Future fix options** (not pursued in this project):
- Extend unsupervised sampling to full [-1,1]³ cube (Eikonal covers corners, OOD gap closed)
- Clip eval grid to inscribed sphere (no retraining, cosmetic fix)
- Replace Fourier PE with hash encoding (localized, no OOD extrapolation by design)

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

## Step 5: Evaluation Bug Fixes (2026-04-15)

Two bugs identified in `src/evaluate.py` that corrupt all reported metrics. Applied fixes to existing checkpoints — no retraining needed.

### Bug 1 — Unoptimized latent codes for val shapes (critical)

**Root cause**: `LatentCodes` allocates `n_train + n_val` codes, but the training loop only iterates `range(n_train)`, so val-shape codes (indices `n_train..N-1`) are never updated. Evaluation used all shape indices 0..N-1, so val shapes were reconstructed from near-random latent vectors. All CD/NC numbers for val shapes (EXP-01 through EXP-09) are noise.

**Fix** (`src/evaluate.py` after line 294): Restrict `shape_names` to first `n_train = int(len(shape_names) * train_split)` shapes before evaluation. This reports honest reconstruction quality on train shapes only.

### Bug 2 — Marching cubes spacing off by one (minor)

**Root cause**: Grid uses `np.linspace(-1, 1, resolution)` (spacing = `2/(resolution-1)`), but `marching_cubes` was called with `spacing=(2.0/resolution,)*3` — ~0.4% too small at resolution=256, shrinking all reconstructed geometry uniformly.

**Fix** (`src/evaluate.py` line 112): `spacing=(2.0 / (resolution - 1),) * 3`

### Next step

- [ ] **5.1** Re-run evaluation on existing checkpoints (EXP-01 through EXP-09) on TC2 — **ask user before proceeding**
- [ ] **5.2** Compare new CD/NC (train-only, corrected spacing) vs previously reported numbers
- [ ] **5.3** Re-examine PE failure hypothesis with clean metrics

### Step 5.1: Additional Evaluation Fixes (2026-04-15)

Follow-up fixes from second review pass. All changes committed alongside Step 5 fixes.

#### Bug 3 — Category-skewed train/val split

**Root cause**: `dataset.py` used `sorted(glob.glob(...))` → alphabetical order → with 300 shapes (100 airplanes, 100 chairs, 100 tables), train = 100 airplanes + 100 chairs + 25 tables, val = 75 tables only. Val set was a single category.

**Fix** (`src/dataset.py`): Added `seed` parameter; shuffle `all_files` with `random.Random(seed)` before the split. Categories now distributed proportionally. `seed` passed from `train()` in `src/train.py`.

**Note**: Existing checkpoints (EXP-01–EXP-09) were trained with the old skewed split. New training runs will use the balanced split.

#### Bug 4 — Checkpoint selection via unoptimized val latents

**Root cause**: `evaluate_val()` (train.py:439) fetches val-shape latent codes at indices `n_train..n_train+n_val-1`. Those codes are allocated but never updated in training. `best.pt` was selected by this meaningless loss.

**Fix** (`src/train.py`): Added `best_train_sdf` tracker. `best.pt` is now saved when `epoch_losses["L_sdf"]` (train reconstruction loss) improves. `evaluate_val()` is retained for monitoring only.

#### Bug 5 — No fixed metric seed

**Root cause**: `trimesh.sample.sample_surface()` in `chamfer_distance()` and `normal_consistency()` used system random state — CD/NC varied between re-evaluations of the same checkpoint.

**Fix** (`src/evaluate.py`): Sets `np.random.seed(metric_seed + shape_idx)` before each shape's metric computation. `metric_seed` comes from `config["seed"]`.

#### Bug 6 — MC_RES default 128 vs config 256

**Root cause**: `slurm/job_eval.sh` defaulted `MC_RES=128`; `config.yaml` specifies `mc_resolution: 256`. All past evaluations ran at the lower resolution.

**Fix** (`slurm/job_eval.sh`): Changed default to `MC_RES="${MC_RES:-256}"`.

#### Enhancement — n_ok/n_total in aggregate

**Fix** (`src/evaluate.py`): Aggregate now includes `n_ok`, `n_total`, `success_rate` fields alongside per-metric stats. Makes failure rate explicit in `results.json`.

### Pending

- [ ] **5.2** Re-evaluate existing checkpoints (EXP-01–EXP-09) on TC2 with corrected evaluate.py — **ask user before proceeding**
- [ ] **5.3** Re-run training experiments with balanced split + train-loss checkpoint selection
- [ ] **5.4** Compare new vs old CD/NC; re-examine PE hypothesis with clean metrics

### Step 5.2: Shape-to-Latent Index Fix (2026-04-15)

Third review pass identified that the shuffle fix in Step 5.1 introduced a new index-mismatch bug.

#### Bug 7 — Shape→latent index mismatch after shuffle fix

**Root cause**: `dataset.py` now shuffles `all_files` before splitting, so latent code index `i` corresponds to `shuffled_files[i]`, not `sorted_files[i]`. But `evaluate.py` still rebuilt `shape_names` from `sorted(os.listdir(...))` and assumed index `i` → sorted shape `i`. For any checkpoint trained after the shuffle fix, every shape was being reconstructed with the wrong latent code.

**Fix**:
- `src/train.py`: After creating `train_dataset`, saves `train_dataset.shape_names` (in shuffled latent-index order) to `{exp_dir}/train_shapes.json`
- `src/evaluate.py`: Loads `train_shapes.json` to reconstruct exact shape→index mapping. Falls back to sorted alphabetical order with a WARNING for legacy checkpoints that predate this fix

#### Bug 8 — Metric seed tied to training seed

**Root cause**: `metric_seed = config.get("seed", 42)` meant that multi-seed experiments sampled different surface points for the same shape, adding evaluation noise on top of training variation.

**Fix** (`src/evaluate.py`): Changed to `metric_seed = 0` — fixed constant independent of training seed.

### Remaining known limitations (not code bugs)

- **Reconstruction-only evaluation**: No test-time latent optimization. Reported CD/NC measure train-set reconstruction quality, not generalization. Disclose in paper.
- **Legacy checkpoints (EXP-01–EXP-09)**: Trained with sorted split + val-loss checkpoint selection. `train_shapes.json` absent → legacy fallback (sorted order, consistent with how they were trained). Metrics are internally consistent for those runs but use the skewed category split.
- **Sphere-vs-cube query mismatch**: Training samples concentrated near surface + unit sphere; marching cubes queries full cube. Can produce unstable geometry in outer shell, especially with PE.
- **SKIP_IOU=1 default**: IoU is skipped in all SLURM eval jobs. Narrower protocol than config intends.

### Pending

- [ ] **5.3** Sync fixed code to TC2
- [ ] **5.4** Ask user before re-evaluating existing checkpoints or retraining

### Step 5.3: Protocol Disclosure (pending)

Reviewer confirmed legacy checkpoint concern is withdrawn (sorted fallback matches sorted training order).
Recommendation: do not implement test-time latent optimization. Keep current protocol with explicit disclosure.

- [ ] **5.5** Update `report/conference_101719.tex` methodology section with three disclosure statements:
  1. Metrics are training-shape reconstruction quality, not held-out generalization
  2. Protocol is not standard DeepSDF evaluation — no test-time latent optimization is performed
  3. EXP-01–EXP-09 best.pt was selected using validation loss computed on unoptimized val latents; results for those runs should be interpreted accordingly
- [ ] **5.6** When re-evaluating legacy checkpoints (EXP-01–EXP-09), evaluate with `latest.pt` instead of `best.pt` to bypass biased checkpoint selection

### Step 5.4: Bug Fixes (Bug 9 + Bug 10)

**Bug 9 — Summary print crash** (`src/evaluate.py:443–445`)
- **Problem**: The post-evaluation summary loop iterated `aggregate.items()` and called `stats['mean']` / `stats['std']` on every entry. The aggregate dict also contains scalar entries (`n_total`, `n_ok`, `success_rate`), causing `TypeError` on those keys. The crash happened after `results.json` was already written, so metrics were saved but the job exited non-zero and the summary was never printed.
- **Fix**: Added `isinstance(stats, dict)` guard — dict entries print `mean ± std`, scalar entries print the raw value.
- [x] **5.4.1** Fixed in `src/evaluate.py` lines 443–447

**Bug 10 — Misleading shuffle comment** (`src/dataset.py:64`)
- **Problem**: Comment said "Shuffle before split so all categories are represented in both train and val" — overstates what the code does. The shuffle is global (not stratified), so category distribution can still skew, especially for small datasets.
- **Fix**: Replaced with accurate description: "Globally shuffle before split (not stratified — category distribution may vary)".
- [x] **5.4.2** Fixed in `src/dataset.py` line 64

**Reviewer items already handled (no code change needed)**
- `train_shapes.json` index mismatch — fixed in earlier step (train.py saves mapping, evaluate.py loads it with legacy fallback)
- Marching cubes spacing `2/(res-1)` correction — already applied
- `best.pt` checkpoint selection bias — disclosed in Step 5.3; legacy runs use sorted fallback consistently
