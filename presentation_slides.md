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
- **Figure**: diagram showing supervised points (with SDF labels) vs. unsupervised points (surface only, contribute only to L_eik)
- Supervision ratios: 1.0, 0.5, 0.1, 0.05

---

## Slide 5 — Experiment Design

- **Table**: one row per experiment group

| Exp | Supervision | Eikonal | Positional Enc. | Seeds |
|-----|------------|---------|-----------------|-------|
| EXP-01 | 100% | No | No | 1 |
| EXP-02 | 100% | Yes | No | 1 |
| EXP-03 | 50% | Yes | No | 1 |
| EXP-04 | 10% | Yes | No | 3 |
| EXP-05 | 5% | Yes | No | 1 |
| EXP-06 | 10% | No | No | 3 |
| EXP-07 | 10% | Yes | Yes | 1 |
| EXP-08 | 10% | Yes | No | 1 (large batch) |
| EXP-09 | 10% | Yes | Yes (4 levels) | 1 |

- Dataset: ShapeNet (3 categories, 300 shapes), ~6h per training run on TC2

---

## Slide 6 — Preliminary Results

- **Table or bar chart**: CD and NC per experiment (from `experiments/experiment_log.md`)
- **Prominent label on the slide**: *"Preliminary — evaluation bugs discovered after these runs (see next slides)"*
- Do not interpret yet; let the numbers sit while pivoting to the bug story

---

## Slide 7 — The Evaluation Review

- Transition: *"Before drawing conclusions, we audited the evaluation pipeline — and found serious problems"*
- **List** (high-level, one line each):
  1. Val-set shapes evaluated with untrained latent codes
  2. Checkpoint selected using a meaningless validation signal
  3. Train/val split was category-skewed (sorted alphabetical order)
  4. Marching cubes spacing off-by-one
  5. Default MC resolution (128) mismatched config (256)
  6. Non-reproducible metric sampling
  7. Shape → latent index mismatch after shuffle fix
- Point: these compound — the reported numbers reflect several independent sources of noise

---

## Slide 8 — Critical Bug: Unoptimized Latent Codes

- **Figure** (simple diagram, two columns):
  - Left: latent array indices 0…n_train-1 (trained), n_train…N-1 (never updated — random)
  - Right: old evaluator iterated all N shapes; val shapes were reconstructed from random latents
- Impact: all CD/NC numbers for val-set shapes are noise, not reconstruction quality
- Fix: evaluation now restricted to training shapes using their stored latent indices

---

## Slide 9 — Other Key Bugs

Three bugs that affect every experiment:

| Bug | Impact | Fix |
|-----|--------|-----|
| Category-skewed split (alphabetical sort) | Val set = tables only; training set biased | Shuffle with fixed seed before split |
| Checkpoint (best.pt) selected by val loss on random latents | Best model ≠ best reconstructor | Select by training L_sdf |
| MC resolution 128 vs config 256 | Lower resolution geometry in all past evaluations | Changed SLURM default to 256 |

---

## Slide 10 — Fixes Applied

- **Summary**: 8 bugs identified across 4 review passes; all fixed in source
- Files changed: `src/evaluate.py`, `src/train.py`, `src/dataset.py`, `slurm/job_eval.sh`
- Key addition: `train_shapes.json` saved during training to guarantee correct shape→latent mapping at eval time
- All fixes committed; code ready to sync to TC2

---

## Slide 11 — What This Means for the Results

- Old numbers (EXP-01–EXP-09): not directly usable — wrong latent mapping, biased checkpoint, wrong MC resolution
- **What is still valid**:
  - Training loss curves show the model is learning (EXP-04/06 PE failure is real signal)
  - Pipeline end-to-end is working (meshes generated, no crashes)
  - Experimental structure (which variables were changed) is sound
- **Protocol note**: this project reports train-set reconstruction quality, not held-out generalization — disclosed intentionally (no test-time latent optimization; out of scope for course timeline)

---

## Slide 12 — Plan Forward

- [ ] Sync corrected code to TC2
- [ ] Re-evaluate EXP-01–EXP-09 checkpoints with fixed evaluator (using `latest.pt` for legacy runs)
- [ ] Re-train key experiments (EXP-04, EXP-06, EXP-07) with corrected split + checkpoint selection
- [ ] Re-examine PE failure hypothesis with clean metrics
- [ ] Report due: 2 weeks — corrected results will be included with explicit protocol disclosures

---

## Slide 13 — Q&A

- Keep experiment table (Slide 5) and bug table (Slide 9) ready as backup slides
