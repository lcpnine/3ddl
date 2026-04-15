# Presentation Slide Outline

---

## Slide 1 — Title

- **Title**: Semi-Supervised 3D Shape Reconstruction with DeepSDF
- Course code, name, date
- One-line framing: *"Reducing annotation requirements for implicit neural shape learning"*

---

## Slide 2 — Research Question

- **Question**: Can we train a DeepSDF auto-decoder with fewer labelled SDF samples while maintaining reconstruction quality?
- **Motivation**: SDF ground truth is expensive to compute; real-world scans may only have surface points
- **Variables**: supervision ratio (100% → 5%), Eikonal regularization, positional encoding

---

## Slide 3 — Background: DeepSDF Auto-Decoder

- One-paragraph intuition: each shape gets a latent code `z_i`; decoder `f_θ(z_i, x)` maps (latent, 3D point) → SDF value
- **Figure**: architecture diagram — latent code + point → MLP → SDF value; training optimizes both `θ` and all `z_i` jointly
- Key point: there is no encoder; unseen shapes require test-time latent optimization (standard protocol); this project evaluates train-set reconstruction only (explain why at Slide 11)

---

## Slide 4 — Semi-Supervised Extension

- **L_sdf**: supervised SDF regression on labelled points
- **L_eik**: Eikonal term `|∇f| = 1` applied to all points (labelled + unlabelled)
- Total loss: `L = L_sdf + λ · L_eik`
- **Figure**: diagram showing supervised points (near-surface offsets, approximate signed SDF labels) vs. unsupervised points (uniform unit-sphere probes, no SDF label, contribute only to L_eik)
- Supervision ratios: 1.0, 0.5, 0.1, 0.05

---

## Slide 5 — Experiment Design

- **Table**: one row per experiment

| Exp | Supervision | Eikonal | Positional Enc. | Seeds |
|-----|------------|---------|-----------------|-------|
| EXP-01 | 100% | No | No | 1 |
| EXP-02 | 100% | Yes | No | 1 |
| EXP-03 | 50% | Yes | No | 1 |
| EXP-04 | 10% | Yes | No | 3 |
| EXP-05 | 5% | Yes | No | 1 |
| EXP-06 | 10% | Yes | Yes (L=6) | 3 |
| EXP-07 | 5% | Yes | Yes (L=6) | 1 |
| EXP-08 | 10% | Yes | Yes (L=6) + `L_2nd` | 1 |
| EXP-09 | 100% | Yes | Yes (L=6) | 1 |
| EXP-10 | 100% | Yes | Yes (L=4) | 1 |
| EXP-11 | 10% | Yes | Yes (L=4) | 1 |
| EXP-12 | 5% | Yes | Yes (L=4) | 1 |

- Dataset: ShapeNet (3 categories, 300 shapes), ~6h per training run on TC2

---

## Slide 6 — Preliminary Results

- **Table or bar chart**: CD and NC per experiment (from `experiments/experiment_log.md`)
- **Prominent label on the slide**: *"Preliminary — evaluation bugs discovered after these runs (see next slides)"*
- Do not interpret yet; let the numbers sit while pivoting to the bug story

---

## Slide 7 — The Evaluation Review

- Transition: *"Before drawing conclusions, I audited the evaluation pipeline."*
- Short message:
  1. The main issue was incorrect latent-code usage during evaluation
  2. A few additional secondary evaluation issues were also identified
  3. So the current numbers should be treated as preliminary

---

## Slide 8 — Main Evaluation Issue: Latent-Code Mapping

- **Core explanation**:
  - DeepSDF is an auto-decoder, so each training shape is associated with its own optimized latent code
  - Reconstruction metrics are meaningful only if evaluation uses the correct trained latent for each shape
  - If the shape-to-latent mapping is wrong, CD/NC no longer reflect actual reconstruction quality
- **Fix**:
  - Evaluation is now restricted to training shapes with the stored latent-index mapping
  - These results should be described as train-set reconstruction metrics, not held-out generalization

---

## Slide 9 — Current Status and Next Step

- **What is already fixed**:
  - The 12-experiment design is fixed
  - Training runs and checkpoints already exist for the planned matrix
  - The evaluation code has been corrected

- **What I cannot claim yet**:
  - The reported quantitative comparisons are not yet final
  - Final conclusions should wait for re-validation with the corrected evaluation protocol

- **What happens next**:
  - Re-evaluate completed checkpoints
  - Update final tables and qualitative figures
  - Finish the remaining verification runs for the report
