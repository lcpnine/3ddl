# Presentation Script — Semi-Supervised 3D Shape Reconstruction with DeepSDF

AI6131 · NTU MSAI · April 2026

---

## Slide 1 — Title (Opening)

Good afternoon. Today I'm presenting a midterm progress update on my project for AI6131. The work is still ongoing, so what I'll share today is a combination of completed design, preliminary results, and an honest account of an evaluation issue I identified, corrected in the pipeline, and am now using for re-validation. I'll walk through the research question, the method, the experiment design, and where things currently stand.

The project is on semi-supervised 3D shape reconstruction using DeepSDF. The central question is whether we can reduce the amount of labeled SDF data required during training while maintaining reconstruction quality. I'll explain what that means concretely in the next few slides.

---

## Slide 2 — Research Question

The core motivation is this: generating SDF training labels requires dense sampling around a shape's surface, which is computationally expensive per shape. If we can reduce the amount of labeled data and supplement training with unlabeled point samples that constrain the field through the Eikonal term, then the method becomes more scalable to larger shape collections. The main question is how far we can push that reduction.

---

## Slide 3 — Research Variables

To investigate this, I'm varying three things: the supervision ratio — from fully labeled down to 5% — the presence or absence of Eikonal regularization, and whether Fourier positional encoding is applied to the 3D input coordinates. These three dimensions define the experiment matrix I'll describe shortly.

---

## Slide 4 — Background: DeepSDF Auto-Decoder

DeepSDF uses what's called an auto-decoder framework. There is no encoder. Instead, each training shape gets its own learned latent code, and a shared 8-layer MLP maps the concatenation of that code with a 3D query point to a signed distance value. The latent codes and the network weights are optimized jointly. One important detail for this project: we are evaluating train-set reconstruction only — test-time latent optimization, which would be needed for generalization to unseen shapes, is out of scope here.

---

## Slide 5 — Semi-Supervised Extension: Loss Formulation

The task-specific loss has two main components here: the supervised SDF term and the Eikonal term. In all runs I also keep the standard DeepSDF latent regularization. The SDF loss is a mean absolute error between the predicted and approximate SDF labels, applied only to the supervised points. The Eikonal loss penalizes deviations of the gradient norm from 1, and it applies to all points including the unlabeled ones. The supervised points are sampled near the surface using normal-offset sampling, and they contribute to both the SDF loss and the Eikonal loss. The unsupervised points are sampled uniformly inside the unit sphere and contribute only to the Eikonal term. One thing to note: the SDF labels here are approximate signed offsets from normal-offset sampling, not ray-cast ground truth.

---

## Slide 6 — Semi-Supervised Extension: Eikonal Warmup

At this stage, I also introduce a linear warmup for the Eikonal weight. For 100% and 50% supervision, I keep the warmup at 100 epochs, since there is enough supervised signal from the start. So for 10% supervision I extend the warmup to 150 epochs, and for 5% to 200, to prevent L_eik dominating sparse labels.

---

## Slide 7 — Experiment Design (EXP-01 to EXP-06)

This first half of the matrix covers EXP-01 through EXP-06, including the first positional-encoding condition in EXP-06. The dataset is ShapeNet, 300 shapes across three categories, with 75% used for training. Each shape is associated with 250K unsupervised points and up to 250K supervised points depending on the supervision ratio, and each run takes roughly 6 hours on the cluster. I'll summarize the primary comparisons after the next slide.

---

## Slide 8 — Experiment Design (EXP-07 to EXP-12)

This second half expands the PE experiments, adding the 5% setting, the L_2nd variant, and the L=4 comparisons. EXP-09 and EXP-10 are the 100% supervision PE settings in this matrix. Across the full matrix there are 12 experiment settings, with multiple seeds for some conditions. The primary comparisons are: EXP-01 versus EXP-02 to isolate the Eikonal effect; EXP-02, EXP-04, and EXP-05 to track how performance changes as supervision drops; and EXP-04 versus EXP-06 to test whether positional encoding helps at low supervision.

---

## Slide 9 — Preliminary Results: No Positional Encoding

These non-PE numbers are preliminary and were evaluated with the original evaluator — evaluation bugs were discovered after these runs. There was also variable mesh extraction success in some non-PE runs; exact counts are in the footnote of the next slide. Note that the warning box at the top covers both groups: cross-group CD/NC comparison between non-PE and PE results is not valid. I'll explain the evaluator issue and fix in the slides that follow.

---

## Slide 10 — Preliminary Results: With Positional Encoding

These PE results are still preliminary, but they were rerun with the corrected evaluator, unlike the non-PE table. So the main point here is within-group reporting only, not direct comparison against the non-PE results. The footnote also shows per-seed CD means for EXP-04 and EXP-06, so I'll keep seed-to-seed variation in mind in the final analysis.

---

## Slide 11 — Evaluation Pipeline Review

Before going further, I want to explain what went wrong with the evaluation. The main issue was incorrect shape-to-latent-code assignment. DeepSDF learns one latent code per training shape, indexed by training order. The evaluator was iterating shapes in sorted alphabetical order, which does not necessarily match training order. As a result, shapes were being decoded with mismatched latent codes, and the CD and NC values did not reflect actual reconstruction quality. Several secondary issues were also identified and corrected — including checkpoint selection and train/validation split ordering. Under the corrected pipeline, evaluation is restricted to training shapes with the correct stored latent indices. I re-evaluated the PE runs under the corrected evaluator.

---

## Slide 12 — Key Evaluation Issue: Latent Code Assignment

To make the issue concrete: the fixed evaluator now loads a `train_shapes.json` file that records the exact order in which shapes were seen during training. The evaluator now reconstructs the exact training-order mapping from that file, so each training shape is decoded with its intended latent code. Validation shapes do not have stored latent codes, so evaluating them would require test-time latent optimization, which is outside the current scope. The PE metrics already reflect this corrected mapping. I'll return to the remaining re-validation status on the final slide.

---

## Slide 13 — Current Status and Next Step

To summarize where things stand: the 12-experiment design itself remains unchanged, all checkpoints already exist, and the evaluation pipeline has been corrected. Quantitative comparisons across experiments are still being re-validated, so I'm not claiming a comparative conclusion yet. The next step is to re-validate the existing checkpoints under the corrected evaluation and then report the quantitative comparison across experiments. That concludes the midterm update. Thank you — happy to take questions.
