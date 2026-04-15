# Presentation Script — Semi-Supervised 3D Shape Reconstruction with DeepSDF

AI6131 · NTU MSAI · April 2026

---

## Opening

Good afternoon. Today I'm presenting a midterm progress update on my project for AI6131. The work is still ongoing, so what I'll share today is a combination of completed design, preliminary results, and an honest account of an evaluation issue I identified and corrected. I'll walk through the research question, the method, the experiment design, and where things currently stand.

---

## Slide 1 — Title

The project is on semi-supervised 3D shape reconstruction using DeepSDF. The central question is whether we can reduce the amount of labeled SDF data required during training without significantly degrading reconstruction quality. I'll explain what that means concretely in the next few slides.

---

## Slide 2 — Research Question

The core motivation is this: generating SDF training labels requires dense sampling around a shape's surface, which is computationally expensive per shape. If we can replace some of that labeled data with unlabeled point samples — which need no annotation — and still maintain reconstruction quality by using the Eikonal constraint to regularize the field, then the method becomes more scalable to larger shape collections. The main question is how far we can push that reduction.

---

## Slide 3 — Research Variables

To investigate this, I'm varying three things: the supervision ratio — from fully labeled down to 5% — the presence or absence of Eikonal regularization, and whether positional encoding is applied to the 3D input coordinates. These three dimensions define the experiment matrix I'll describe shortly.

---

## Slide 4 — Background: DeepSDF Auto-Decoder

DeepSDF uses what's called an auto-decoder framework. There is no encoder. Instead, each training shape gets its own learned latent code, and a shared 8-layer MLP maps the concatenation of that code with a 3D query point to a signed distance value. The latent codes and the network weights are optimized jointly. One important detail for this project: we are evaluating train-set reconstruction only — test-time latent optimization, which would be needed for generalization to unseen shapes, is out of scope here.

---

## Slide 5 — Semi-Supervised Extension: Loss Formulation

The task-specific loss has two main components here: the supervised SDF term and the Eikonal term. In all runs I also keep the standard DeepSDF latent regularization. The SDF loss is a mean absolute error between the predicted and approximate SDF labels, applied only to the supervised points. The Eikonal loss penalizes deviations of the gradient norm from 1, and it applies to all points including the unlabeled ones. The supervised points are sampled near the surface using normal-offset sampling. The unsupervised points are sampled uniformly inside the unit sphere and contribute only to the Eikonal term. One thing to note: the SDF labels here are approximate signed offsets, not ray-cast ground truth, which is a deliberate simplification to keep sampling tractable.

---

## Slide 6 — Semi-Supervised Extension: Eikonal Warmup

At this stage, I also introduce a linear warmup for the Eikonal weight. The rationale is that with very few supervised points, a strong Eikonal penalty early in training can dominate and prevent the network from fitting the shape signal. So for 10% supervision I extend the warmup to 150 epochs, and for 5% I extend it to 200. This is a heuristic, but it follows a reasonable principle: the sparser the supervision, the more carefully you need to ramp up the unsupervised regularizer.

---

## Slide 7 — Experiment Design (EXP-01 to EXP-06)

The experiment matrix covers 12 runs in total. The dataset is ShapeNet, 300 shapes across three categories, with 75% used for training. Each shape sees up to 250K supervised and 250K unsupervised points per epoch, and each run takes roughly 6 hours on the cluster. The primary comparisons are: EXP-01 versus EXP-02 to isolate the Eikonal effect; EXP-02, EXP-04, and EXP-05 to track what happens as we reduce supervision; and EXP-04 versus EXP-06 to test whether positional encoding helps at low supervision.

---

## Slide 8 — Experiment Design (EXP-07 to EXP-12)

The second half of the matrix extends into the positional encoding regime, testing different encoding frequencies and one additional regularization variant — EXP-08, which adds a second-order smoothness term. EXP-09 and EXP-10 serve as fully-supervised baselines with PE, providing reference points for the low-supervision PE runs. All 12 training runs have produced checkpoints — the remaining work is corrected evaluation, which I'll explain.

---

## Slide 9 — Preliminary Results: No Positional Encoding

These are the preliminary numbers from the non-PE group, evaluated using the original evaluator. The original non-PE numbers show an unexpected pattern, but because they were produced before the evaluator fix, I'm treating them only as provisional and not interpreting them. I'll explain the fix in the next two slides.

---

## Slide 10 — Preliminary Results: With Positional Encoding

The PE group reflects the corrected pipeline. One limitation is that the non-PE and PE groups cannot be directly compared numerically — they were produced under different evaluator versions. Within the PE group, the results are fairly close across supervision levels, with CD values clustering around 0.14. Cross-experiment conclusions will be drawn once the non-PE group is also re-evaluated.

---

## Slide 11 — Evaluation Pipeline Review

Before going further, I want to explain what went wrong with the evaluation. The main issue was incorrect shape-to-latent-code assignment. DeepSDF learns one latent code per training shape, indexed by training order. The evaluator was iterating shapes in sorted alphabetical order, which does not necessarily match training order. As a result, each shape was being decoded with the wrong latent code, and the CD and NC values did not reflect actual reconstruction quality. Several secondary issues were also identified and corrected — including checkpoint selection and train/validation split ordering. This bug invalidated the earlier metrics, so I corrected the evaluator before making any cross-experiment claim.

---

## Slide 12 — Key Evaluation Issue: Latent Code Assignment

To make the issue concrete: the fixed evaluator now loads a `train_shapes.json` file that records the exact order in which shapes were seen during training. This guarantees that each shape is decoded with the correct latent code. For the non-PE group, re-evaluation using this corrected pipeline is the immediate next task. The metrics reported for the PE group already reflect this fix. It's also worth noting that all reported metrics measure train-set reconstruction — held-out generalization is not part of the current scope.

---

## Slide 13 — Current Status and Next Step

To summarize where things stand: the full experiment matrix is in place, all checkpoints exist, and the evaluation pipeline has been corrected. What is not yet claimable is any quantitative conclusion across experiments — the non-PE group still needs to go through the fixed evaluator. Once that is done, I'll be in a position to report a valid cross-experiment comparison. That's where the project stands. Happy to take any questions.

---

## Closing

That concludes the midterm update. Thank you, and I'm happy to take questions.
