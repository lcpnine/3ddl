# Semi-Supervised DeepSDF with Geometric Regularization + Positional Encoding

## Project Execution Plan (v6)

**AI6131 3D Deep Learning Course Project**

---

# Part 1: Project Overview

## 1.1 Project Objective

Implement a semi-supervised learning framework that reduces SDF supervision cost in DeepSDF while maintaining reconstruction quality. Combine Eikonal regularization with Fourier positional encoding to investigate two questions: (1) how far can supervision be reduced with geometric regularization, and (2) does positional encoding further improve label efficiency in the semi-supervised setting.

**Scope:** This study focuses on per-shape SDF fitting (auto-decoder), not generalizable SDF inference across unseen shapes. The supervision ratio refers to the fraction of SDF sample points per shape, not the fraction of shapes with labels. This per-shape scope is standard in DeepSDF-family work and sufficient to isolate the effect of supervision reduction on reconstruction quality.

## 1.2 Core Research Questions

1. What percentage of SDF labels is sufficient to achieve reconstruction quality equivalent to fully-supervised DeepSDF?
2. Does Fourier positional encoding improve label efficiency under semi-supervised training, and if so, by how much?
3. At what supervision ratio threshold does PE transition from beneficial to harmful, and what explains this transition?
4. Can we quantitatively measure the change in SDF gradient field quality as the supervision ratio varies?

## 1.3 Novel Contribution

Prior work on Eikonal regularization (IGR, SAL, StEik) focused on fully unsupervised or fully supervised settings and on stabilizing Eikonal optimization itself. GenSDF explored semi-supervised SDF via meta-learning, but required a two-stage pipeline. The interaction between positional encoding and supervision ratio in SDF learning has not been systematically studied.

**Primary contribution:** We provide the first systematic characterization of the supervision ratio threshold at which Fourier PE transitions from beneficial to harmful in single-stage semi-supervised SDF learning, offering practical guidelines for PE usage in data-scarce settings. This goes beyond combining two known techniques: we quantitatively map the PE-supervision interaction landscape and identify the critical ratio below which PE degrades rather than improves reconstruction.

**Relationship to arXiv 2401.01391:** That work analyzes PE's sampling rate requirements in fully-supervised SDF fitting, identifying noisy artifacts from insufficient sampling. Our work extends this analysis along a different axis: we investigate PE's effect under reduced supervision, where the network must extrapolate SDF values from sparse labels. If PE amplifies noise at low supervision ratios, this finding complements their sampling rate analysis; if PE remains robust, it demonstrates PE's utility beyond the fully-supervised regime they studied.

**Positioning relative to StEik:** StEik addresses structural instability of Eikonal optimization itself; we instead explore reducing supervision requirements along a different axis while using standard Eikonal regularization as a tool.

If PE degrades quality at low supervision ratios (due to overfitting to high-frequency noise), that is equally reportable and directly informs practical guidelines. If PE shows no clear effect, the label efficiency analysis with Eikonal alone remains a concrete contribution with the supervision threshold characterization as the primary finding.

## 1.4 Expected Deliverables

| Deliverable | Description | Deadline |
|---|---|---|
| Code Repository | DeepSDF-based semi-supervised framework with PE, reproducible scripts + SLURM job scripts | Week 12 |
| Experimental Results | Quantitative comparison tables + visualizations across supervision ratios | Week 13 |
| Presentation Materials | Slides for Week 13 in-person presentation + demo video | Week 13 |
| Final Report | Conference-style report (Abstract through Future Work) | Week 15 |

---

# Part 2: Execution Structure

The project is organized into 3 sequential stages, each with a clear checklist. Each stage is executed as a task list in a single Claude session.

## 2.1 Three-Stage Overview

| Stage | Tasks | Output | Estimated Duration |
|---|---|---|---|
| Stage 1: Setup | Literature review, metric definition, data preprocessing, code implementation | Working codebase + preprocessed data + SLURM scripts | Week 7-9 |
| Stage 2: Experiments | Run core experiments on CCDS TC2, evaluate results, tune hyperparameters | Experiment logs + result tables | Week 9-12 |
| Stage 3: Analysis & Reporting | Visualizations, result interpretation, presentation, final report | Figures + slides + report | Week 11-15 |

Note: Stage 3 overlaps with Stage 2 intentionally. Start drafting key figures and narrative by Week 11, not after experiments are complete.

## 2.2 Stage 1 Checklist: Setup

- [ ] **Literature review:** Survey 9 key papers (see Part 3.1), extract loss function designs and hyperparameter settings into literature_review.md
- [ ] **Metric definition:** Define success criteria (see Part 3.2), write evaluate.py auto-evaluation script
- [ ] **Watertight mesh scan:** Run trimesh.is_watertight on all ShapeNet meshes for target categories, log counts per category to determine feasible dataset size (see Part 3.3)
- [ ] **Data preprocessing:** Build SDF sampling pipeline from ShapeNet meshes (see Part 3.3), generate supervised/unsupervised splits at all ratios
- [ ] **Code implementation:** Implement DeepSDF baseline + Eikonal loss + PE on PyTorch 2.x (see Part 3.4), including ratio-dependent Eikonal warmup schedule
- [ ] **SLURM job scripts:** Create template job.sh for CCDS TC2 cluster (see Part 4.2), with separate evaluation job using --dependency
- [ ] **Validation run:** Train baseline (EXP-01) on a small subset (10 shapes) on TC2 to verify the full pipeline works end-to-end
- [ ] **Disk cleanup:** Delete raw ShapeNet data immediately after preprocessing is complete; verify total usage < 30GB before starting Stage 2

## 2.3 Stage 2 Checklist: Experiments

- [ ] **Run core experiments** EXP-01 through EXP-07 (see Part 3.5) with 3 random seeds each, submitting parallel jobs (2 at a time) on TC2
- [ ] **Auto-evaluate** each experiment via SLURM dependency chain (eval_job.sh), save results to experiments/{EXP-ID}/results.json
- [ ] **Quick diagnosis** after first 3 experiments: check gradient norms, training curves, L_eik/L_sdf ratio, identify obvious issues
- [ ] **Hyperparameter tuning** if needed: adjust lambda_eik, learning rate, warmup_epochs based on initial results and Diagnosis Guide
- [ ] **PE fallback check:** If EXP-06/07 show PE degrading CD vs EXP-04/05, run L=4 variants before concluding PE is harmful (see Diagnosis Guide)
- [ ] **Conditional seed expansion:** If EXP-04 or EXP-06 show CD coefficient of variation (std/mean) > 0.2 across 3 seeds, expand those experiments to 5 seeds (add seeds 789, 101) for statistical reliability
- [ ] **Run EXP-08, EXP-09** (second-order regularization + full PE isolation) if core experiments are clean by Week 11

## 2.4 Stage 3 Checklist: Analysis & Reporting

- [ ] **Read arXiv 2401.01391** (PE + sampling rate analysis) for discussion section material on PE behavior
- [ ] **Generate core figures** (Week 11): label efficiency curve, ablation bar chart, mesh comparison grid
- [ ] **Draft narrative arc** (Week 11): what story do the results tell? Focus on the PE-supervision threshold finding
- [ ] **Build presentation** (Week 12): slides with key figures, demo video of mesh reconstructions
- [ ] **Deliver presentation** (Week 13): collect professor feedback
- [ ] **Write final report** (Week 13-15): incorporate feedback, conference-style format

---

# Part 3: Detailed Task Specifications

## 3.1 Literature Review

### Required Paper List

| Paper | Key Content to Extract | Priority |
|---|---|---|
| DeepSDF (Park et al., CVPR 2019) | Baseline architecture, loss function, latent code regularization | Required |
| SAL (Atzmon & Lipman, CVPR 2020) | Sign-agnostic learning, unsigned distance learning method | Required |
| IGR (Gropp et al., ICML 2020) | Eikonal regularization loss design, gradient penalty implementation, warmup strategy | Required |
| StEik (NeurIPS 2023) | Eikonal optimization instability analysis, over-smoothing problem, stabilization techniques | Required |
| NeRF (Mildenhall et al., ECCV 2020) | Fourier positional encoding design, frequency selection | Required |
| Fourier Features (Tancik et al., NeurIPS 2020) | Theoretical analysis of PE for coordinate networks, spectral bias | Required |
| GenSDF (Chung et al., 2023) | 2-stage semi-supervised meta-learning for SDF; closest prior work structurally | Required |
| On Optimal Sampling for Learning SDF Using MLPs with PE (arXiv 2401.01391) | PE-induced noisy artifacts, sampling rate requirements for PE networks | Required |
| DiGS (Ben-Shabat et al., CVPR 2022) | Second-order regularization (divergence), unsupervised SDF learning | Reference |
| Neural-Pull (Ma et al., ICML 2021) | Pulling operation from query points to surface | Reference |

**Output:** literature_review.md containing per-paper (1) problem definition, (2) core loss function equations, (3) hyperparameter settings, (4) elements applicable to our project. Special attention to: how PE interacts with SDF learning (including failure modes from arXiv 2401.01391), how StEik's stability analysis relates to our warmup strategy, and how our single-stage approach differs from GenSDF's two-stage meta-learning.

### Related Work Positioning Notes

When writing the final report's Related Work section, structure the Eikonal regularization narrative as: IGR (introduced Eikonal for SDF) -> SAL (sign-agnostic extension) -> StEik (stability analysis of Eikonal optimization). Position our work as: "StEik addresses structural instability of Eikonal optimization; we instead explore reducing supervision requirements along a different axis."

For PE-related positioning, include: "arXiv 2401.01391 analyzes PE's sampling rate requirements in fully-supervised SDF fitting but does not investigate PE's effect on label efficiency in semi-supervised settings. Our work fills this gap by characterizing the supervision ratio threshold at which PE transitions from beneficial to harmful."

Additionally, briefly note the following alternatives in one paragraph:
- **SIREN (Sitzmann et al., NeurIPS 2020):** Periodic activations as an alternative to PE for implicit representations. We fix the activation function (ReLU) and vary input encoding to isolate PE's contribution.
- **SAPE (NeurIPS 2021):** Spatially-adaptive progressive encoding as a learnable PE schedule. We use fixed-frequency PE; adaptive scheduling is a natural future work direction.

MetaSDF is not required; GenSDF already covers the meta-learning lineage sufficiently.

---

## 3.2 Metric Definition

### Success Criteria

| Metric Category | Metric Name | Target Threshold | Measurement Method |
|---|---|---|---|
| Reconstruction Quality | Chamfer Distance (CD) | < 0.005 (normalized) | GT mesh vs reconstructed mesh, 30,000 points sampled per mesh |
| Reconstruction Quality | IoU (128^3) | > 0.80 | Voxelization at 128^3 resolution + occupancy comparison |
| Reconstruction Quality | IoU (256^3) | > 0.80 | Voxelization at 256^3 resolution (captures thin structures) |
| Reconstruction Quality | Normal Consistency | > 0.85 | Co-primary metric. Nearest-neighbor correspondence: sample GT points, find nearest on reconstructed mesh, compute normal dot product |
| Label Efficiency | Critical Ratio | CD < 0.008 at 10% labels | Derived from CD curve across supervision ratios |
| PE Threshold | PE Transition Point | Identify ratio where PE flips from helpful to harmful | Compare PE vs no-PE CD across all ratios |
| SDF Quality | Eikonal Deviation | mean |gradient norm - 1| < 0.05 | Gradients at random points in learned SDF |
| PE Effect | PE Improvement | CD improvement > 10% vs no-PE at same ratio | Compare EXP-06 vs EXP-04 |

### Metric Hierarchy

**Co-primary metrics:** Chamfer Distance and Normal Consistency. CD measures global geometric accuracy; Normal Consistency captures surface detail quality especially for thin structures (chair legs, airplane wings) where CD may be insensitive. Report both prominently in all result tables.

**Secondary metrics:** IoU at 128^3 and 256^3. The dual-resolution IoU serves as a sanity check. If 128^3 IoU and 256^3 IoU diverge significantly for a shape category, this indicates thin-structure sensitivity and should be noted in the report.

### Evaluation Protocol (Fixed for Reproducibility)

- **Chamfer Distance:** Sample exactly 30,000 points from each mesh (GT and reconstructed) using trimesh.sample.sample_surface(). Report mean and std across shapes.
- **IoU:** Voxelize both GT and reconstructed meshes at both 128^3 and 256^3 resolution. Report intersection/union of occupied voxels at both resolutions.
- **Normal Consistency:** Sample 30,000 points from the GT mesh with normals. For each GT point, find its nearest neighbor on the reconstructed mesh surface via KD-tree lookup. Compute mean of abs(dot(n_gt, n_pred)) over all pairs.

### Divergence Handling (Seed Outliers)

A run is marked as "diverged" if L_sdf at epoch 500 has not decreased below 50% of its value at epoch 10 (after warmup effects stabilize). Diverged runs are excluded from mean/std computation and reported separately in the results table (e.g., "2/3 seeds converged, 1 diverged"). This is expected primarily at 5% supervision ratio.

### Statistical Reliability Check

After computing 3-seed results for EXP-04 and EXP-06, check the coefficient of variation (CV = std/mean) of CD. If CV > 0.2, those experiments are expanded to 5 seeds (add seeds 789, 101). This ensures the core PE comparison has sufficient statistical power.

**Deliverable:** evaluate.py that takes an experiment directory, computes all metrics with the above fixed parameters (including both IoU resolutions), outputs JSON report.

---

## 3.3 Data Preprocessing

### Step 0: Watertight Mesh Scan (Must Do First)

ShapeNet contains many non-manifold and self-intersecting meshes. The mesh-to-sdf library and direct surface sampling both require watertight meshes for reliable SDF computation. Before building the full pipeline:

1. Run `trimesh.is_watertight` on all meshes in each target category
2. Log the count of watertight meshes per category
3. If any category has fewer than 100 watertight meshes, either:
   - Add a 4th category (e.g., car: 02958343, lamp: 03636649) to compensate
   - Apply mesh repair via `trimesh.repair.fix_normals()` + `trimesh.repair.fill_holes()` (may not always work)
4. Record final counts in a data_audit.md file for the report

**Fallback SDF sampling (if mesh-to-sdf fails on repaired meshes):**
Use `trimesh.sample.sample_surface()` to get surface points + normals, then compute near-surface SDF using multi-scale offsets (see Key Parameters below). This avoids mesh-to-sdf's strict watertight requirement.

### Data Source: ShapeNet

ShapeNet access granted (2026-03-23). Pipeline was validated on parametric meshes; all production experiments use ShapeNet meshes. The preprocessing pipeline (`scripts/preprocess.py`) requires only changing `--mesh_dir` to point at ShapeNet category directories. Before preprocessing, run the watertight mesh scan (Step 0 above) to determine usable mesh counts per category.

### Pipeline

| Step | Task | Tool/Method | Output |
|---|---|---|---|
| 0. Watertight Scan | Scan all meshes, log watertight counts | trimesh.is_watertight | data_audit.md |
| 1. Load Mesh | Load and normalize ShapeNet .obj/.off files | trimesh | Normalized meshes (unit sphere) |
| 2. Supervised Samples | Surface points + multi-scale offset points | Multi-scale strategy (see below) | near_surface_sdf.npz |
| 3. Unsupervised Samples | Uniform random sampling in unit sphere (no GT SDF) | numpy | unsupervised_points.npz |
| 4. Ratio Split | Split supervised samples: 100%, 50%, 10%, 5% | Random subsampling | Per-ratio directories |
| 5. Validation | Visualize SDF values for quality check | matplotlib 3D scatter | Validation images |

### Key Parameters

| Parameter | Value | Description |
|---|---|---|
| Near-surface offset (epsilon) | {0.005, 0.01, 0.05} | Multi-scale offsets for robust SDF coverage |
| Supervised points per shape | 250,000 (at 100%) | Total SDF samples for fully-supervised case |
| Unsupervised points per shape | 250,000 | Points with Eikonal-only supervision |
| Offset directions (j) | -2, -1, 1, 2 | Bidirectional multiple offsets per epsilon scale |
| Categories | chair (04256520), airplane (02691156), table (04379243) | ShapeNet categories |
| Shapes per category | Train 60 / Val 20 / Test 20 (target) | Adjust based on watertight scan results |

**Multi-scale sampling strategy:** For each surface point, generate offset samples at all three epsilon scales {0.005, 0.01, 0.05}. This ensures the network receives supervision both very close to the surface (capturing fine detail) and further away (stabilizing the broader SDF field). The 250,000 supervised points are distributed approximately equally across the three scales.

**Total dataset size:** 3 categories x 100 shapes = 300 shapes (target). Final count depends on watertight scan.

---

## 3.4 Code Implementation

### Architecture

The network is DeepSDF's 8-layer MLP with skip connection at layer 4. Two variants:

1. **Baseline (no PE):** Input = [latent_code z; x, y, z] (dim = latent_dim + 3)
2. **With PE:** Input = [latent_code z; gamma(x), gamma(y), gamma(z)] where gamma is Fourier PE

Positional encoding: gamma(p) = [sin(2^0 * pi * p), cos(2^0 * pi * p), ..., sin(2^(L-1) * pi * p), cos(2^(L-1) * pi * p)]

Default L = 6 (each coordinate: 3 -> 12 dims, total xyz: 3 -> 36 dims). Search range: L in {4, 6, 8}.

### Loss Function

```
L_total = L_sdf + lambda_eik(t) * L_eik + lambda_z * L_z

Where:
  L_sdf = mean(|f(z, x) - SDF_gt|)           # supervised points only
  L_eik = mean(||grad_x f(z, x)|| - 1)^2     # all points (supervised + unsupervised)
  L_z   = mean(||z||^2)                        # latent code regularization

Eikonal warmup schedule (ratio-dependent):
  lambda_eik(t) = lambda_eik * min(1.0, t / warmup_epochs)
  Default: lambda_eik = 0.1
  
  Warmup epochs by supervision ratio:
    100% / 50% supervision: warmup_epochs = 100
    10% supervision:        warmup_epochs = 150
    5% supervision:         warmup_epochs = 200
  
  Rationale: At low supervision ratios, L_sdf is computed over fewer points and
  thus has smaller magnitude. A longer warmup prevents L_eik from dominating early
  training before the network has learned basic shape geometry from L_sdf.

Gradient clipping: max_norm = 1.0 (applied globally per step)
```

Optional (feasible on A40 48GB):
```
  L_2nd = mean(|div(grad_x f(z, x))|)         # second-order regularization
  Requires create_graph=True for second-order autodiff
  Memory note: ~2x graph size. On A40 48GB, feasible with batch_size=16384 at 100 shapes/category.
```

### Implementation Modules

| Module | File | Description |
|---|---|---|
| Network | model.py | DeepSDF MLP + optional PE layer |
| Loss | losses.py | L_sdf, L_eik (with ratio-dependent warmup), L_2nd, L_z as separate functions |
| Data | dataset.py | Mixed supervised/unsupervised batch loader with ratio control |
| Training | train.py | Training loop with auto-diff for Eikonal, gradient clipping (max_norm=1.0), checkpoint save/resume, CUDA compatibility |
| Evaluation | evaluate.py | Marching cubes extraction + metric computation (CD@30k points, IoU@128^3 and 256^3, Normal Consistency) |
| Config | config.yaml | All hyperparameters in one file, per-experiment override |
| SLURM (train) | job_template.sh | Template SLURM script for TC2 training submission |
| SLURM (eval) | eval_job.sh | Separate CPU-focused evaluation job, chained via --dependency |

### Base Code Approach

Reference the official DeepSDF (github.com/facebookresearch/DeepSDF) for architecture details, but reimplement cleanly on PyTorch 2.x. The official repo has Python 2 remnants and outdated dependencies that would waste time fixing.

---

## 3.5 Experiment Design

### Core Experiment Matrix (Must Complete)

| ID | Supervision Ratio | Eikonal | PE | Purpose |
|---|---|---|---|---|
| EXP-01 | 100% | X | X | Fully-supervised DeepSDF baseline |
| EXP-02 | 100% | O | X | Full supervision + Eikonal effect |
| EXP-03 | 50% | O | X | Half labels, Eikonal only |
| EXP-04 | 10% | O | X | Low label, Eikonal only (key data point) |
| EXP-05 | 5% | O | X | Minimal label limit |
| EXP-06 | 10% | O | O (L=6) | Low label + PE (key comparison vs EXP-04) |
| EXP-07 | 5% | O | O (L=6) | Minimal label + PE (key comparison vs EXP-05) |

**Total core runs:** 7 experiments x 3 seeds = 21 runs (potentially 23-25 if EXP-04/06 need seed expansion). With TC2 parallel submission (2 jobs at a time), this takes ~11-13 submission rounds.

### Extended Experiments (Feasible on A40)

| ID | Supervision Ratio | Eikonal | PE | L_2nd | Purpose |
|---|---|---|---|---|---|
| EXP-08 | 10% | O | O (L=6) | O | PE + second-order combined effect |
| EXP-09 | 100% | O | O (L=6) | X | Full supervision + PE (isolate PE effect on quality ceiling) |

### PE Fallback Experiments (Triggered Conditionally)

| ID | Supervision Ratio | Eikonal | PE | Trigger Condition |
|---|---|---|---|---|
| EXP-06b | 10% | O | O (L=4) | Run if EXP-06 CD > EXP-04 CD (PE hurts at L=6) |
| EXP-07b | 5% | O | O (L=4) | Run if EXP-07 CD > EXP-05 CD (PE hurts at L=6) |

These are not scheduled upfront. They exist as a contingency if high-frequency PE causes overfitting at low supervision ratios.

### Hyperparameter Defaults

| Parameter | Default | Notes |
|---|---|---|
| lambda_eik | 0.1 | Tune in [0.01, 0.05, 0.1, 0.5] if gradient norms deviate |
| warmup_epochs | Ratio-dependent (see Loss Function section) | 100/50%: 100, 10%: 150, 5%: 200 |
| lambda_z | 1e-4 | Standard DeepSDF setting |
| Latent dimension | 256 | Standard DeepSDF setting |
| PE frequency levels (L) | 6 | Fallback to L=4 if overfitting observed (see Diagnosis Guide) |
| Learning rate | 5e-4 | Reduce to 1e-4 if training diverges |
| Gradient clipping | max_norm = 1.0 | Applied globally per optimization step |
| Epochs | 1000 | Extend to 2000 if not converged (request extended QoS) |
| Batch size (points/shape) | 16384 | Can go up to 32768 on A40 48GB |
| Random seeds | 42, 123, 456 | For mean/std reporting. Expand to 5 seeds (add 789, 101) for EXP-04/06 if CV > 0.2 |

### Execution Rules

1. All experiments use identical random seeds and report mean/std.
2. Save config.yaml, training logs, checkpoints, results.json per experiment in experiments/{EXP-ID}/.
3. Submit 2 jobs in parallel on TC2 to maximize throughput.
4. Run evaluate.py as a separate SLURM job chained via `sbatch --dependency=afterok:$TRAIN_JOBID eval_job.sh`. This prevents evaluation from exceeding training job's wall time.
5. Checkpoint every 100 epochs for job resumption after wall time limit.
6. **Checkpoint saving strategy:** Latest checkpoint = full state (model weights + optimizer state + scheduler, for resume). Best checkpoint = model weights only (for evaluation). This roughly halves checkpoint disk usage.
7. Monitor disk usage with `ncdu`. Target: stay under 80GB at all times. Delete intermediate checkpoints immediately.
8. Delete raw ShapeNet data immediately after Stage 1 preprocessing is complete.

### Diagnosis Guide

After running initial experiments, check these patterns before proceeding:

| Symptom | Likely Cause | Action |
|---|---|---|
| Gradient norm mean far from 1 (deviation > 0.1) | lambda_eik too low or warmup too short | Increase lambda_eik or extend warmup_epochs by 50 |
| Eikonal deviation > 0.05 but training stable | Gradient collapse (cf. StEik) | Verify gradient clipping is active (max_norm=1.0); try lambda_eik=0.5 |
| L_eik > 10x L_sdf in early training (before warmup completes) | Warmup insufficient for supervision ratio | Extend warmup_epochs: 5% -> 250, 10% -> 200. If already extended, also reduce lambda_eik to 0.05 |
| CD degrades sharply below 25% supervision | Insufficient regularization | Increase unsupervised points or lambda_eik |
| Loss diverges in early epochs | Learning rate too high despite warmup | Reduce LR to 1e-4, verify warmup is active |
| One category much worse than others | Category-specific difficulty or mesh quality | Check watertight quality for that category; note in report as finding |
| PE worsens CD at 10% or 5% vs no-PE | High-frequency overfitting at low supervision | Run EXP-06b/07b with L=4; if still worse, report as negative finding with analysis referencing arXiv 2401.01391 |
| PE shows no improvement (CD roughly equal) | L too low or too high | Try L=4 and L=8 to bracket the effect |
| OOM on A40 during 2nd-order experiments | Computation graph too large | Reduce batch to 8192 for EXP-08 only |
| High variance across 3 seeds (CV > 0.2) | Insufficient statistical power | Expand to 5 seeds for affected experiments |

---

## 3.6 Required Visualizations

| Visualization | Axes | Purpose |
|---|---|---|
| Label Efficiency Curve | X: Supervision Ratio (%), Y: Chamfer Distance | Core result: two lines (with PE, without PE) showing PE's effect on efficiency. Annotate the threshold point where PE flips from helpful to harmful. |
| Ablation Bar Chart | X: Experiment conditions, Y: CD / Normal Consistency | Compare all experiment conditions side by side (co-primary metrics) |
| Gradient Norm Histogram | X: ||grad f|| value, Y: Frequency | SDF quality verification |
| Mesh Comparison Grid | Per-category GT vs reconstructions at different ratios | Qualitative evaluation for presentation/report |
| Training Curves | X: Epoch, Y: Loss components (including warmup ramp visible) | Convergence behavior, stability analysis. Include L_eik/L_sdf ratio plot. |
| IoU Resolution Comparison | X: Experiment, Y: IoU at 128^3 vs 256^3 | Document thin-structure sensitivity if divergence is observed |
| PE Frequency Ablation (if EXP-06b/07b run) | X: PE level L, Y: CD at 10% supervision | Justify PE frequency choice or document overfitting behavior |

---

# Part 4: Prerequisites (Must Complete Before Starting)

## 4.1 Data Acquisition

| Item | Method | Status |
|---|---|---|
| ShapeNet Core v2 | Applied at shapenet.org | Applied (awaiting approval) |
| Thingi10K (development fallback) | Public download, no application needed | Available immediately for code development |
| Category selection | chair (04256520), airplane (02691156), table (04379243) | Ready after ShapeNet download |
| Watertight scan | Run trimesh.is_watertight on all meshes per category | First task after download |

**Development strategy:** All code (model, loss, training loop, evaluation pipeline) is developed and debugged using Thingi10K meshes. When ShapeNet is approved, only the data preprocessing step is re-run. The pipeline is designed so data source swap requires changing only the input mesh directory path.

## 4.2 Computing Environment: CCDS GPU Cluster TC2

### Cluster Specifications

| Resource | Value |
|---|---|
| GPU | NVIDIA A40 (48GB VRAM) per compute node |
| Nodes | 6 compute nodes (TC2N01-TC2N06), 4-8 GPUs each |
| Access | SSH to head node 10.96.189.12 (NTU network / VPN required) |
| QoS "normal" | 1 GPU, 10 CPU cores, 30GB RAM, max 6hr wall time, 2 concurrent jobs |
| Storage | 100GB home directory quota |
| Extended QoS | Apply via ccdsgpu-tc@ntu.edu.sg for up to 48hr wall time |

**Action: Request extended QoS (12hr or 24hr) preemptively before Stage 2 begins.**

### Disk Budget (100GB Quota)

| Item | Estimated Size | Notes |
|---|---|---|
| Preprocessed NPZ files | ~5-10GB | 300 shapes x multi-scale samples |
| Conda environment | ~8-10GB | PyTorch + dependencies |
| Checkpoints (latest full + best weights-only) | ~20-35GB | 63 runs x (1 full ~400MB + 1 weights-only ~100MB) |
| Code + configs + logs | ~1GB | Lightweight |
| **Total** | **~45-70GB** | Buffer of 30-55GB |

**Critical:** Delete raw ShapeNet immediately after preprocessing (saves ~10-30GB depending on categories). Run `ncdu` after each batch of experiments completes.

### SLURM Job Script Template (Training)

```bash
#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=25G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --job-name=EXP-01_seed42
#SBATCH --output=logs/output_%x_%j.out
#SBATCH --error=logs/error_%x_%j.err

module load anaconda
module load cuda/12.8.0
eval "$(conda shell.bash hook)"
conda activate yt3dl

python train.py --config configs/exp01_seed42.yaml
```

### SLURM Job Script Template (Evaluation, chained)

```bash
#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=EVAL-EXP-01_seed42
#SBATCH --output=logs/output_%x_%j.out
#SBATCH --error=logs/error_%x_%j.err

module load anaconda
eval "$(conda shell.bash hook)"
conda activate yt3dl

python evaluate.py --exp_dir experiments/EXP-01/seed42 \
    --output experiments/EXP-01/seed42/results.json \
    --voxel_res 128 256
```

**Submission pattern:**
```bash
TRAIN_JOB=$(sbatch --parsable job_train.sh)
sbatch --dependency=afterok:$TRAIN_JOB eval_job.sh
```

### Key Workflow Notes

- **No code execution on head node.** All training must go through `sbatch`.
- **File transfer:** Use WinSCP/FileZilla (SFTP) or `scp` to upload code/data and download results.
- **Conda setup:** Create environment on first SSH login, install packages via `conda install` (prefer over pip).
- **Disk management:** Run `ncdu` after every batch of experiments. Delete intermediate checkpoints immediately. Latest checkpoint = full state (for resume), best checkpoint = model weights only (for eval). Delete raw ShapeNet after preprocessing.
- **Parallel submission:** Submit 2 jobs simultaneously (QoS allows 2 concurrent). E.g., EXP-01_seed42 and EXP-01_seed123 at the same time.
- **Extended QoS:** Request preemptively before Stage 2. Email ccdsgpu-tc@ntu.edu.sg with estimated training time needs.

### Estimated Training Time per Experiment (A40)

| Configuration | Estimated Time | Fits in 6hr QoS? |
|---|---|---|
| 100 shapes, 1000 epochs, batch 16384, no L_2nd | ~3-4 hours | Yes |
| 100 shapes, 1000 epochs, batch 16384, with L_2nd | ~5-6 hours | Tight, may need extended QoS |
| 100 shapes, 2000 epochs, batch 16384 | ~6-8 hours | Needs 8hr or 12hr QoS |

## 4.3 Software Environment

Setup on TC2 via conda:

```bash
module load anaconda
conda create -n deepsdf python=3.10
conda activate yt3dl
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge trimesh matplotlib scikit-image pyyaml
pip install mesh-to-sdf  # only if not available via conda
```

| Package | Version | Purpose |
|---|---|---|
| Python | 3.10 | Base |
| PyTorch | 2.0+ with CUDA 12.x | Training |
| trimesh | latest | Mesh I/O, watertight check, evaluation |
| mesh-to-sdf | latest | GT SDF computation (fallback: manual multi-scale surface sampling) |
| scikit-image | latest | Marching cubes |
| matplotlib | latest | Visualization |
| PyYAML | latest | Config management |

## 4.4 Base Code Preparation

1. Clone DeepSDF official repo for reference: github.com/facebookresearch/DeepSDF
2. Review architecture: 8-layer MLP, skip at layer 4, in deep_sdf/networks/
3. Review SDF sampling: deep_sdf/data.py
4. Initialize project repo: semi-supervised-deepsdf/ with the module structure from Part 3.4
5. Upload code + data to TC2 via SFTP

---

# Part 5: Weekly Execution Plan

| Week | Milestone | Tasks | Notes |
|---|---|---|---|
| Week 7-8 | Proposal + setup | Literature review (including StEik, arXiv 2401.01391), metric definition, code development on Thingi10K, ShapeNet download + watertight scan (when approved), data pipeline with multi-scale epsilon, SLURM templates (train + eval), conda env on TC2, proposal submission, request extended QoS | Code dev proceeds on Thingi10K regardless of ShapeNet status |
| Week 8-9 | Baseline working | Implement DeepSDF + Eikonal (with ratio-dependent warmup + gradient clipping), run EXP-01 and EXP-02 on TC2, verify pipeline. Delete raw ShapeNet after preprocessing. | Critical: if baseline doesn't work, nothing else matters |
| Week 9-10 | Semi-supervised experiments | Run EXP-03, 04, 05 (varying supervision ratios without PE), 2 jobs parallel. Monitor L_eik/L_sdf ratio. | Consult professor with initial results at Friday office hours |
| Week 10-11 | PE experiments + drafting | Implement PE, run EXP-06, 07. If PE hurts, run EXP-06b/07b (L=4). Check CV for EXP-04/06, expand seeds if needed. Start drafting key figures and narrative. Run EXP-08/09 if time allows. Read arXiv 2401.01391 for discussion material. | Begin Stage 3 in parallel |
| Week 11-12 | Analysis + presentation prep | Generate all visualizations (including IoU resolution comparison), tune hyperparameters if needed, build slides | Request extended QoS if needed for remaining runs |
| Week 13 | Presentation | Deliver in-person presentation, collect feedback | |
| Week 13-15 | Final report | Write conference-style report incorporating professor feedback. Frame PE threshold finding as primary contribution. | |

---

# Part 6: Claude Prompt Template

Provide this entire document to Claude along with the following instruction. Adjust the bracketed sections.

> You are helping me execute a 3D deep learning course project. This document is the full project plan. Work through the Stage [1/2/3] checklist items sequentially. My data is at [ShapeNet path or "not yet available, using Thingi10K for development"]. My compute environment is CCDS GPU Cluster TC2 (NVIDIA A40, SLURM scheduler, 6hr wall time with extended QoS requested, 100GB quota). Current progress: [describe what's done so far].
>
> For each checklist item, produce the specified output (code file, config, SLURM script, markdown, etc.) and confirm completion before moving to the next item. If you encounter an issue or need a decision from me, ask before proceeding.
>
> Priority rules:
> - Working code over perfect code. Get the pipeline running first, optimize later.
> - Core experiments (EXP-01 through EXP-07) before extended experiments.
> - If any experiment fails, diagnose using the Diagnosis Guide before running more experiments.
> - Checkpoint every 100 epochs for SLURM job resumption. Latest checkpoint = full state, best checkpoint = model weights only.
> - Include ratio-dependent Eikonal warmup and gradient clipping (max_norm=1.0) in all Eikonal experiments.
> - Monitor disk usage: keep under 80GB, delete intermediate checkpoints and raw ShapeNet after preprocessing.
> - Mark a run as "diverged" if L_sdf at epoch 500 has not decreased below 50% of its epoch 10 value. Exclude diverged runs from mean/std.
> - If EXP-04 or EXP-06 show CD coefficient of variation > 0.2 across 3 seeds, expand to 5 seeds.

---

# Part 7: Risks and Mitigation

| Risk | Impact | Mitigation | Priority |
|---|---|---|---|
| ShapeNet approval delayed | Cannot start data pipeline | Use Thingi10K for all code development and pipeline validation; swap to ShapeNet data when approved (directory path change only) | High |
| Too few watertight meshes per category | Dataset smaller than planned | Add 4th category (car/lamp), or use fallback multi-scale surface sampling instead of mesh-to-sdf | High |
| TC2 6hr wall time insufficient | Cannot complete 1000 epochs in one job | Checkpoint + resume across jobs; request extended QoS preemptively (up to 48hr) from ccdsgpu-tc@ntu.edu.sg | Medium |
| TC2 queue congestion | Jobs stuck in PENDING | Submit during off-peak hours; check `squeue` before submitting; defer if many PENDING jobs | Medium |
| Eikonal training instability | Divergent loss early in training | Ratio-dependent warmup schedule + gradient clipping (max_norm=1.0); reduce LR to 1e-4 if needed | Medium |
| L_eik dominates L_sdf at low supervision | Network learns Eikonal constraint but not shape | Extend warmup_epochs further; monitor L_eik/L_sdf ratio in training logs | Medium |
| OOM during second-order experiments | EXP-08 fails on A40 | Reduce batch to 8192 for L_2nd experiments only; still feasible on A40 48GB | Low |
| PE causes overfitting at low supervision | CD worsens with PE at 5-10% labels | Run L=4 fallback (EXP-06b/07b); report as negative finding with analysis referencing arXiv 2401.01391 and frame as threshold characterization | Medium |
| PE shows no improvement | Weaker novel contribution | Reportable as negative finding; frame as "PE's benefit does not extend to semi-supervised regime" with threshold analysis. Label efficiency curve with Eikonal alone remains a concrete contribution. | Medium |
| Quality collapse at 5% labels | One less data point in curve | Expected and reportable; 10% result is the key data point | Low |
| Disk quota exceeded (100GB) | Cannot save checkpoints | Delete raw ShapeNet immediately after preprocessing; best ckpt = weights only; run `ncdu` after each experiment batch | High |
| Time shortage | Incomplete experiments | EXP-01, 04, 06 are absolute minimum (baseline, low-label no-PE, low-label with-PE = the core story) | High |
| High variance in key experiments | Weak statistical claims | Conditional seed expansion (3 -> 5) for EXP-04/06 if CV > 0.2 | Medium |
| Thin structures lost in 128^3 IoU | Misleading quality metrics | Dual-resolution IoU (128^3 + 256^3) + Normal Consistency as co-primary metric | Low |

---

# Appendix: Changelog

## v5 -> v6 Changes

1. **Novelty framing (Part 1.3):** Rewritten to emphasize "first systematic characterization of PE-supervision threshold" rather than simple technique combination. Added explicit positioning against arXiv 2401.01391.
2. **Scope clarification (Part 1.1):** Added explicit per-shape fitting scope statement to prevent misinterpretation of supervision ratio.
3. **Research question added (Part 1.2):** Q3 now asks about PE transition threshold.
4. **Metric hierarchy (Part 3.2):** Normal Consistency elevated to co-primary metric. IoU measured at both 128^3 and 256^3. Added statistical reliability check (CV > 0.2 triggers seed expansion).
5. **Evaluation job separation (Part 4.2):** evaluate.py runs as separate SLURM job via --dependency chain instead of appending to training job.
6. **Ratio-dependent warmup (Part 3.4):** warmup_epochs now scales with supervision ratio (100: 100ep, 10%: 150ep, 5%: 200ep). Added L_eik/L_sdf dominance diagnosis.
7. **Conditional seed expansion (Part 2.3, 3.5):** EXP-04/06 expand from 3 to 5 seeds if CV > 0.2.
8. **Data fallback strategy (Part 3.3, 4.1):** Thingi10K development strategy made explicit with directory-swap design.
9. **Extended QoS (Part 4.2):** Changed from reactive to preemptive request.
10. **Visualization additions (Part 3.6):** Added IoU resolution comparison plot and L_eik/L_sdf ratio in training curves.
