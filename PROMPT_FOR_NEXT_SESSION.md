# Report Writing Prompt — Semi-Supervised DeepSDF Project

## Your Task

Write the final project report for an AI6131 (Deep Learning) course project at NTU. The report should be a scientific paper in standard ML conference format (Introduction, Related Work, Method, Experiments, Results, Discussion, Conclusion).

## Read These Files First

Read these in order to fully understand the project:

1. `CLAUDE.md` — Project rules and cluster setup
2. `PLAN.md` — Full experiment pipeline (Steps 0–8, all completed)
3. `experiments/experiment_log.md` — **Single source of truth** for all results (12 experiments, multi-seed runs, detailed analysis, final conclusions)
4. `src/model.py` — DeepSDF model architecture (MLP with optional Fourier positional encoding)
5. `src/train.py` — Training loop (semi-supervised SDF loss + Eikonal + optional L_2nd)
6. `src/losses.py` — Loss functions (L_sdf, L_eik, L_2nd, L_z)
7. `src/dataset.py` — Data loading (supervised/unsupervised point sampling)
8. `src/evaluate.py` — Evaluation pipeline (Chamfer Distance, Normal Consistency, IoU, marching cubes mesh extraction)
9. `scripts/generate_figures.py` — Figure generation code
10. `configs/config.yaml` — Default hyperparameters
11. `Description.pdf, Slides.pdf` — Documentation for the project

## What This Project Is About

### Research Question
Can geometric regularization (Eikonal constraint) reduce the need for signed distance supervision in DeepSDF, and does Fourier positional encoding help or hurt in this semi-supervised setting?

### Background
- **DeepSDF** (Park et al., 2019) learns a continuous signed distance function (SDF) per 3D shape using an auto-decoder with learned latent codes. It requires dense signed distance supervision for every training point.
- **Eikonal regularization** enforces the physical constraint |∇f(x)| = 1 on the SDF, providing a training signal from unsupervised points (no SDF labels needed). This could reduce the need for expensive signed distance annotations.
- **Fourier Positional Encoding** (Mildenhall et al., 2020 / Tancik et al., 2020) maps low-dimensional coordinates to high-dimensional features via sinusoidal functions, helping MLPs learn high-frequency details. It's standard in NeRF but less tested in SDF learning.

### Method
We train DeepSDF with varying supervision ratios (5%, 10%, 50%, 100%) and combinations of:
- **Eikonal loss** (L_eik): |∇_x f(x)|² - 1)² on unsupervised points
- **Fourier PE**: γ(x) = [sin(2^0 πx), cos(2^0 πx), ..., sin(2^{L-1} πx), cos(2^{L-1} πx)] at L=4 and L=6
- **Second-order regularization** (L_2nd): penalizes Hessian magnitude to smooth the SDF

### Data
- 300 ShapeNet shapes (airplane, chair, table categories)
- 250K supervised + 250K unsupervised points per shape
- Preprocessed with watertight mesh conversion

### Architecture
- 8-layer MLP, hidden_dim=512, skip connection at layer 4
- Latent code dim=256, learned per shape (auto-decoder)
- 3000 epochs, batch_size=16384, Adam optimizer

### Evaluation
- **Chamfer Distance (CD)** @ 30K points (lower is better)
- **Normal Consistency (NC)** @ 30K points (higher is better, max 1.0)
- Mesh extraction via marching cubes at resolution 128
- IoU was skipped for computational reasons

## Experiment Design (12 Experiments)

| ID | Supervision | Eikonal | PE | Extra | Purpose |
|----|------------|---------|-----|-------|---------|
| EXP-01 | 100% | No | No | — | Vanilla DeepSDF baseline |
| EXP-02 | 100% | Yes | No | — | Eikonal effect at full supervision |
| EXP-03 | 50% | Yes | No | — | Half labels + Eikonal |
| EXP-04 | 10% | Yes | No | — | Key: low-label + Eikonal (3 seeds) |
| EXP-05 | 5% | Yes | No | — | Minimal labels + Eikonal |
| EXP-06 | 10% | Yes | L=6 | — | Key: PE effect at low labels (3 seeds) |
| EXP-07 | 5% | Yes | L=6 | — | PE at minimal labels |
| EXP-08 | 10% | Yes | L=6 | L_2nd | Can second-order reg save PE? |
| EXP-09 | 100% | Yes | L=6 | — | PE quality ceiling (full supervision) |
| EXP-10 | 100% | Yes | L=4 | — | Lower PE frequency follow-up |
| EXP-11 | 10% | Yes | L=4 | — | Lower PE freq at low labels |
| EXP-12 | 5% | Yes | L=4 | — | Lower PE freq at minimal labels |

## Key Results

### Finding 1: Eikonal enables 90% label reduction

| Experiment | Ratio | CD | NC | vs Baseline |
|-----------|-------|------|------|------------|
| EXP-01 (baseline) | 100%, no eik | 0.0593 | 0.5522 | — |
| EXP-02 | 100% + eik | 0.0543 | 0.5920 | CD -8.4% |
| EXP-03 | 50% + eik | 0.0534 | 0.5924 | CD -9.9% |
| EXP-04 (3-seed) | 10% + eik | 0.0496 | 0.5987 | **CD -16.4%** |
| EXP-05 | 5% + eik | 0.0509 | 0.5766 | CD -14.2% |

10% labels + Eikonal **beats** the fully-supervised baseline. The Eikonal constraint provides such strong geometric regularization that reducing labels actually improves generalization (less overfitting to noisy SDF labels).

### Finding 2: Fourier PE is catastrophic

| Experiment | PE | Ratio | CD | NC |
|-----------|-----|-------|------|------|
| EXP-06 (3-seed) | L=6 | 10% | 0.1399 | 0.5078 |
| EXP-07 | L=6 | 5% | 0.1448 | 0.5070 |
| EXP-08 | L=6+L_2nd | 10% | 0.1443 | 0.5045 |
| EXP-09 | L=6 | 100% | 0.1450 | 0.5027 |
| EXP-10 | L=4 | 100% | 0.1401 | 0.5078 |
| EXP-11 | L=4 | 10% | 0.1427 | 0.5067 |
| EXP-12 | L=4 | 5% | 0.1399 | 0.5078 |

ALL PE experiments collapse to CD ~0.14 (2.4–2.8x worse than baseline) regardless of:
- Supervision level (5%–100%)
- PE frequency (L=4 vs L=6)
- Additional regularization (L_2nd)

PE failure is highly reproducible (3-seed CV=0.022). NC std is extremely tight (~0.01) across all PE experiments, confirming PE collapses normal diversity.

**Root cause**: PE amplifies high-frequency SDF oscillations. Training points are near the surface, but marching cubes evaluates on the full [-1,1]³ grid — PE extrapolates wildly at cube corners far from training data.

### Finding 3: Second-order regularization cannot save PE
EXP-08 (PE + L_2nd) produces CD=0.1443, virtually identical to EXP-06 (PE alone, CD=0.1399). The failure is fundamental to PE's interaction with SDF learning, not a regularization gap.

## Multi-Seed Validation
- **EXP-04** (10% + Eikonal): 3 seeds, CD=0.0496±0.0098, CV=0.197
- **EXP-06** (10% + Eikonal + PE): 3 seeds, CD=0.1399±0.0031, CV=0.022

## Technical Issues Encountered (worth mentioning in report)
1. **Model collapse** (Step 0.10): Initial weight init too small + per-shape optimizer stepping caused collapse. Fixed with proper init and gradient accumulation.
2. **Evaluator bug** (Step 7.10): Mesh export and metric computation shared a try/except, causing PE experiments to report 0/300 shapes despite valid metrics. Fixed by separating the two.

## Report Structure Suggestion

1. **Abstract** (~150 words)
2. **Introduction** — Motivation for semi-supervised SDF learning, why label efficiency matters for 3D reconstruction
3. **Related Work** — DeepSDF, Eikonal/geometric regularization in neural implicit functions, positional encoding (NeRF, SIREN, etc.)
4. **Method** — Architecture, loss functions (L_sdf, L_eik, L_2nd, L_z), semi-supervised training procedure, positional encoding formulation
5. **Experiments** — Dataset, evaluation metrics, experiment matrix, training details
6. **Results** — Label efficiency curve, PE ablation, multi-seed validation, per-category analysis
7. **Discussion** — Why Eikonal works (geometric prior as free supervision), why PE fails (spectral bias amplification in extrapolation regions), limitations
8. **Conclusion** — Practical recommendation (10% labels + Eikonal), future work (hash grids, learned features instead of Fourier PE)

## Figures to Include
- Label efficiency curve (CD vs supervision ratio, with/without Eikonal)
- PE ablation bar chart (CD across all PE experiments)
- Qualitative mesh comparisons (good reconstruction vs PE failure)
- Existing figures may be in `experiments/figures/` (generated by `scripts/generate_figures.py`)

## Constraints
- Course project report, not a full conference paper — keep it concise (8–10 pages)
- Focus on empirical findings, not theoretical proofs
- Be honest about limitations (single dataset, limited categories, no IoU)
