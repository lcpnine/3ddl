# Experiment Log

Single source of truth for all experiment results.

## Experiment Matrix

| ID | Ratio | Eikonal | PE | Purpose |
|----|-------|---------|----|---------|
| EXP-01 | 100% | No | No | Fully-supervised DeepSDF baseline |
| EXP-02 | 100% | Yes | No | Full supervision + Eikonal effect |
| EXP-03 | 50% | Yes | No | Half labels, Eikonal only |
| EXP-04 | 10% | Yes | No | Low label, Eikonal only (key data point) |
| EXP-05 | 5% | Yes | No | Minimal label limit |
| EXP-06 | 10% | Yes | L=6 | Low label + PE (key comparison vs EXP-04) |
| EXP-07 | 5% | Yes | L=6 | Minimal label + PE (key comparison vs EXP-05) |
| EXP-08 | 10% | Yes | L=6 | + L_2nd second-order regularization |
| EXP-09 | 100% | Yes | L=6 | Full supervision + PE (quality ceiling) |
| EXP-10 | 100% | Yes | L=4 | Full supervision + lower-frequency PE follow-up |
| EXP-11 | 10% | Yes | L=4 | Low-label + lower-frequency PE follow-up |
| EXP-12 | 5% | Yes | L=4 | Minimal-label + lower-frequency PE follow-up |

## Results

The summary table below reports the **post-fix production metrics** used in the
report: retrained on regenerated data (80% near-surface multi-scale offsets +
20% far-field signed-distance samples) and evaluated with the fixed evaluator
(sphere-clip, L1 Chamfer, independent mesh-export). Historical exploratory runs
and pre-fix notes are preserved later in this file for provenance, but they are
not the source of truth when they disagree with the summary table or the
corresponding seed's `results.json`.

Experiments marked `priority` were retrained at `epochs=1500`; EXP-01 and EXP-02
kept their original `epochs=3000` budget. 299 shapes trained (airplane_0077
dropped due to degenerate geometry that hung signed_distance).

| ID | Seeds | CD (mean +/- std) | NC (mean +/- std) | IoU@128 | IoU@256 | Status |
|----|-------|-------------------|--------------------|---------|---------|--------|
| EXP-01 | 42 | 0.0361 +/- 0.0143 | 0.7288 +/- 0.0667 | skipped | skipped | done (299/299, 0 failures) |
| EXP-02 | 42 | 0.0295 +/- 0.0107 | 0.7208 +/- 0.0694 | skipped | skipped | done (299/299, 0 failures) |
| EXP-03 | 42 | 0.0288 +/- 0.0108 | 0.7480 +/- 0.0739 | skipped | skipped | done (priority, 1500 epochs) |
| EXP-04 | 42 | 0.0297 +/- 0.0101 | 0.7260 +/- 0.0742 | skipped | skipped | done (priority, 1500 epochs) |
| EXP-04 | 123 | 0.0322 +/- 0.0120 | 0.7283 +/- 0.0742 | skipped | skipped | done (priority, 1500 epochs) |
| EXP-04 | 456 | 0.0327 +/- 0.0114 | 0.7321 +/- 0.0714 | skipped | skipped | done (priority, 1500 epochs) |
| EXP-04 | **3-seed** | **0.0315 +/- 0.0015** | **0.7288 +/- 0.0032** | — | — | **CV(CD)=0.047 < 0.2 ✓** |
| EXP-05 | 42 | 0.0344 +/- 0.0117 | 0.7236 +/- 0.0745 | skipped | skipped | done (priority, 1500 epochs) |
| EXP-06 | 42 | 0.0320 +/- 0.0138 | 0.5246 +/- 0.0405 | skipped | skipped | done (priority, 1500 epochs) |
| EXP-06 | 123 | 0.0307 +/- 0.0131 | 0.5134 +/- 0.0508 | skipped | skipped | done (priority, 1500 epochs) |
| EXP-06 | 456 | 0.0297 +/- 0.0126 | 0.5132 +/- 0.0505 | skipped | skipped | done (priority, 1500 epochs) |
| EXP-06 | **3-seed** | **0.0308 +/- 0.0012** | **0.5171 +/- 0.0066** | — | — | **CV(CD)=0.040 < 0.2 ✓** |
| EXP-11 | 42 | 0.0432 +/- 0.0158 | 0.6090 +/- 0.0791 | skipped | skipped | done (priority, 1500 epochs) |
| EXP-07, 08, 09, 10, 12 | — | not run (QoS budget) | — | — | — | skipped to stay within Apr 23 QoS |

## Archived Historical Notes

The sections below preserve contemporaneous experiment notes from earlier runs,
including pre-fix evaluator outputs and some marching-cubes evaluations at
`MC res=128`. They are kept for debugging provenance only. For any figure cited
in the report or defense materials, use the summary table above and the matching
post-fix `results.json` artifact for that seed.

### EXP-01 — Fully-supervised baseline (seed 42) [ShapeNet re-run]
- **Date**: 2026-03-27
- **Config**: ratio=1.0, eikonal=off, PE=off, epochs=3000, batch=16384
- **Data**: 300 ShapeNet shapes (airplane/chair/table), 250K sup/unsup points each
- **Training**: ~4hr on TC2 (A40 GPU), L_sdf=0.0319 final, L_z=0.00024
- **Eval** (MC res=128, IoU skipped, 300/300 shapes, 0 failures):
  - **CD**: mean=0.0593, std=0.0464, min=0.0116, max=0.3321
  - **NC**: mean=0.5522, std=0.1057, min=0.2787, max=0.8538
- **Note**: Previous parametric mesh EXP-01 (2026-03-22) used 20 shapes/10K pts — archived. This is the production baseline.

### EXP-02 — Full supervision + Eikonal (seed 42)
- **Date**: 2026-03-27
- **Config**: ratio=1.0, eikonal=on, PE=off, epochs=3000, batch=16384
- **Data**: 300 ShapeNet shapes (airplane/chair/table), 250K sup/unsup points each
- **Training**: ~4hr on TC2 (A40 GPU), L_sdf=0.0350 final, L_eik=0.416, L_z=0.00212
- **Eval** (MC res=128, IoU skipped, 261/300 shapes, 39 failures):
  - **CD**: mean=0.0543, std=0.0416, min=0.0129, max=0.2212
  - **NC**: mean=0.5920, std=0.1321, min=0.2422, max=0.9003
- **vs EXP-01**: CD improved 8.4% (0.0593→0.0543), NC improved 7.2% (0.5522→0.5920). Eikonal helps even at full supervision.
- **Note**: 39 shape failures (13%) — divergence flag triggered but aggregate metrics are reasonable. Max CD reduced from 0.3321→0.2212.

### EXP-03 — 50% labels + Eikonal (seed 42)
- **Date**: 2026-03-28
- **Config**: ratio=0.5, eikonal=on, PE=off, epochs=3000, batch=16384
- **Data**: 300 ShapeNet shapes (airplane/chair/table), 250K sup/unsup points each
- **Training**: ~5.7hr on TC2 (A40 GPU), L_sdf=0.0603 final, L_eik=0.172, L_z=0.00203
- **Eval** (MC res=128, IoU skipped, 259/300 shapes, 41 failures):
  - **CD**: mean=0.0534, std=0.0400, min=0.0131, max=0.2326
  - **NC**: mean=0.5924, std=0.1334, min=0.2194, max=0.9003
- **vs EXP-02** (100% labels): CD improved 1.7% (0.0543→0.0534), NC comparable (0.5920→0.5924). Halving labels has virtually no effect with Eikonal.
- **vs EXP-01** (baseline): CD improved 9.9% (0.0593→0.0534), NC improved 7.3% (0.5522→0.5924).
- **Note**: Divergence flag triggered, 41 failures (13.7%) — similar failure rate to EXP-02. Metrics are strong; 50% labels + Eikonal matches full supervision.

### EXP-04 — 10% labels + Eikonal (seed 42)
- **Date**: 2026-03-28
- **Config**: ratio=0.1, eikonal=on, PE=off, epochs=3000, batch=16384
- **Data**: 300 ShapeNet shapes (airplane/chair/table), 250K sup/unsup points each
- **Training**: ~5.7hr on TC2 (A40 GPU), L_sdf=0.0690 final, L_eik=0.314, L_z=0.00203
- **Eval** (MC res=128, IoU skipped, 263/300 shapes, 37 failures):
  - **CD**: mean=0.0609, std=0.0469, min=0.0138, max=0.2625
  - **NC**: mean=0.5805, std=0.1406, min=0.2473, max=0.8982
- **vs EXP-02** (100% labels + eik): CD worse 12.2% (0.0543→0.0609), NC worse 1.9% (0.5920→0.5805). 10% labels shows degradation.
- **vs EXP-01** (baseline): CD worse 2.7% (0.0593→0.0609), NC improved 5.1% (0.5522→0.5805). Eikonal still provides NC benefit even at 10% labels.
- **vs EXP-03** (50% labels): CD worse 14.0% (0.0534→0.0609), NC worse 2.0% (0.5924→0.5805). Significant drop from 50%→10% labels.
- **Note**: No divergence flag. 37 failures (12.3%). Key finding: 10% labels with Eikonal roughly matches baseline (no eikonal, full labels) for CD, but retains NC improvement.

### EXP-04 — 10% labels + Eikonal (seed 123)
- **Date**: 2026-04-02
- **Config**: ratio=0.1, eikonal=on, PE=off, epochs=3000, batch=16384
- **Data**: 300 ShapeNet shapes (airplane/chair/table), 250K sup/unsup points each
- **Training**: ~4hr on TC2 (A40 GPU)
- **Eval** (MC res=128, IoU skipped, 300/300 shapes, 0 failures):
  - **CD**: mean=0.0443, std=0.0228, min=0.0122, max=0.1299
  - **NC**: mean=0.6202, std=0.0982, min=0.3693, max=0.8973
- **vs EXP-04 s42**: CD improved 27.3% (0.0609→0.0443), NC improved 6.8% (0.5805→0.6202). Seed 123 significantly better — zero failures vs 37 failures.
- **Note**: Much better than seed 42 across all metrics. Zero shape failures vs 12.3% failure rate in s42. Max CD reduced from 0.2625→0.1299. The large seed-to-seed variance (27% CD difference) underscores the importance of multi-seed evaluation.

### EXP-04 — 10% labels + Eikonal (seed 456)
- **Date**: 2026-04-02
- **Config**: ratio=0.1, eikonal=on, PE=off, epochs=3000, batch=16384
- **Data**: 300 ShapeNet shapes (airplane/chair/table), 250K sup/unsup points each
- **Training**: ~4hr on TC2 (A40 GPU)
- **Eval** (MC res=128, IoU skipped, 300/300 shapes, 0 failures):
  - **CD**: mean=0.0436, std=0.0214, min=0.0109, max=0.1380
  - **NC**: mean=0.5954, std=0.1156, min=0.3554, max=0.8987
- **vs EXP-04 s42**: CD improved 28.4% (0.0609→0.0436), NC improved 2.6% (0.5805→0.5954). Consistent with s123.
- **vs EXP-04 s123**: CD comparable (0.0443→0.0436), NC comparable (0.6202→0.5954). Seeds 123 and 456 agree closely.
- **Note**: Seed 42 is the outlier with higher CD and 37 failures. Seeds 123/456 both achieve zero failures and CD~0.044.

### EXP-04 — 3-seed summary (seeds 42, 123, 456)
- **CD**: mean=0.0496, std=0.0098, **CV=0.197 < 0.2** — 3 seeds sufficient, no expansion needed
- **NC**: mean=0.5987, std=0.0201, CV=0.034
- **vs EXP-01** (baseline): CD improved 16.4% (0.0593→0.0496), NC improved 8.4% (0.5522→0.5987)
- **Conclusion**: 10% labels + Eikonal beats the fully-supervised baseline across seeds. Eikonal regularization enables 90% label reduction with improved quality.

### EXP-05 — 5% labels + Eikonal (seed 42)
- **Date**: 2026-03-29
- **Config**: ratio=0.05, eikonal=on, PE=off, epochs=3000, batch=16384
- **Data**: 300 ShapeNet shapes (airplane/chair/table), 250K sup/unsup points each
- **Training**: ~5.8hr on TC2 (A40 GPU), L_sdf=0.0338 final, L_eik=0.514, L_z=0.00208
- **Eval** (MC res=128, IoU skipped, 295/300 shapes, 5 failures):
  - **CD**: mean=0.0509, std=0.0385, min=0.0126, max=0.2860
  - **NC**: mean=0.5766, std=0.1271, min=0.2785, max=0.8920
- **vs EXP-04** (10% labels): CD improved 16.4% (0.0609→0.0509), NC comparable (0.5805→0.5766). Surprisingly, fewer labels yielded better CD.
- **vs EXP-02** (100% labels + eik): CD improved 6.3% (0.0543→0.0509), NC slightly worse (0.5920→0.5766).
- **vs EXP-01** (baseline): CD improved 14.2% (0.0593→0.0509), NC improved 4.4% (0.5522→0.5766).
- **Note**: Diverged flag but only 5 failures (1.7%) — best failure rate among eikonal experiments. Best CD of all experiments so far. The non-monotonic label efficiency curve (5% > 10% > 50%) suggests possible overfitting at higher supervision ratios, or Eikonal regularization becoming more effective when it dominates the loss landscape.

### EXP-06 — 10% labels + Eikonal + PE L=6 (seed 42)
- **Date**: 2026-03-30
- **Config**: ratio=0.1, eikonal=on, PE=L=6, epochs=3000, batch=16384
- **Data**: 300 ShapeNet shapes (airplane/chair/table), 250K sup/unsup points each
- **Training**: ~6hr on TC2 (A40 GPU), L_sdf=0.0333 final, L_eik=0.021, L_z=0.00074
- **Eval** (MC res=128, IoU skipped, 300/300 shapes, 0 failures — rerun with fixed evaluator):
  - **CD**: mean=0.1443, std=0.0448, min=0.0522, max=0.2627
  - **NC**: mean=0.5055, std=0.0109, min=0.4701, max=0.5372
- **vs EXP-04** (10% labels, no PE): CD 2.4x worse (0.0609→0.1443), NC worse (0.5805→0.5055). PE severely degrades quality at 10% supervision.
- **vs EXP-01** (baseline): CD 2.4x worse (0.0593→0.1443), NC worse (0.5522→0.5055).
- **Note**: Original eval (240/300 shapes) was partial due to disk quota and evaluator bug (mesh export failure invalidated metrics). Rerun with fixed evaluator produces 300/300 shapes. PE with L=6 creates high-frequency SDF oscillations. The NC std (0.0109) is unusually tight — PE collapses normal diversity. Root cause: training data lives near a unit sphere but eval grid samples full [-1,1]³ cube; PE amplifies extrapolation errors at cube corners unseen during training.

### EXP-06 — 10% labels + Eikonal + PE L=6 (seed 123)
- **Date**: 2026-04-02
- **Config**: ratio=0.1, eikonal=on, PE=L=6, epochs=3000, batch=16384
- **Data**: 300 ShapeNet shapes (airplane/chair/table), 250K sup/unsup points each
- **Training**: ~4hr on TC2 (A40 GPU), L_sdf=0.0334 final, L_eik=0.036, L_z=0.00073
- **Eval** (MC res=128, IoU skipped, 0/300 success, 300 failures):
  - **CD**: mean=0.1375, std=0.0433, min=0.0510, max=0.2529
  - **NC**: mean=0.5087, std=0.0110, min=0.4813, max=0.5600
- **vs EXP-06 s42**: CD slightly better (0.1515→0.1375), NC comparable (0.5059→0.5087). Same catastrophic PE failure.
- **Note**: Consistent with all PE experiments. NC std ~0.011 confirms PE normal collapse is reproducible across seeds.

### EXP-07 — 5% labels + Eikonal + PE L=6 (seed 42)
- **Date**: 2026-03-31
- **Config**: ratio=0.05, eikonal=on, PE=L=6, epochs=3000, batch=16384
- **Data**: 300 ShapeNet shapes (airplane/chair/table), 250K sup/unsup points each
- **Training**: ~5.9hr on TC2 (A40 GPU), L_sdf=0.0333 final, L_eik=0.0303, L_z=0.00079
- **Eval** (MC res=128, IoU skipped, 0/300 success, 300 failures — all shapes failed mesh extraction):
  - **CD**: mean=0.1448 (computed from failed shapes with partial metrics)
  - **NC**: mean=0.5074
  - Per-category: airplane CD=0.1957/NC=0.5073, chair CD=0.1240/NC=0.5081, table CD=0.1146/NC=0.5066
- **vs EXP-05** (5% labels, no PE): CD 2.8x worse (0.0509→0.1448), NC worse (0.5766→0.5074). PE catastrophic at 5% supervision — even worse impact than at 10%.
- **vs EXP-06** (10% labels + PE): CD slightly better (0.1515→0.1448), NC comparable (0.5059→0.5074). Both PE experiments show similar catastrophic failure, confirming PE is the problem, not the label ratio.
- **vs EXP-01** (baseline): CD 2.4x worse (0.0593→0.1448), NC worse (0.5522→0.5074).
- **Note**: All 300 shapes marked "failed" — PE generates oversized meshes that fail validation. NC std is extremely tight (similar to EXP-06's 0.0109), confirming PE collapses normal diversity across all shapes. Training converged normally (L_sdf stable at 0.033, grad_norm ~0.96). The problem is purely at inference/reconstruction time: PE amplifies high-frequency oscillations in the SDF, causing marching cubes to generate massive, noisy meshes. Key insight: EXP-06 and EXP-07 show PE L=6 fails at both 10% and 5% supervision with similar severity — the issue is PE + low supervision, not the specific label ratio.

### EXP-09 — Full supervision + Eikonal + PE L=6 (seed 42)
- **Date**: 2026-03-31
- **Config**: ratio=1.0, eikonal=on, PE=L=6, epochs=3000, batch=16384
- **Data**: 300 ShapeNet shapes (airplane/chair/table), 250K sup/unsup points each
- **Training**: ~3.1hr on TC2 (A40 GPU, 3.7s/epoch), L_sdf=0.0329 final, L_eik=0.0691, L_z=0.00064
- **Eval** (MC res=128, IoU skipped, 0/300 success, 300 failures — all shapes failed mesh extraction):
  - **CD**: mean=0.1450, std=0.0451, min=0.0518, max=0.2644
  - **NC**: mean=0.5031, std=0.0116, min=0.4704, max=0.5373
  - Per-category: airplane CD=0.1962/NC=0.5023, chair CD=0.1242/NC=0.5072, table CD=0.1147/NC=0.4998
- **vs EXP-02** (100% labels, no PE): CD 2.7x worse (0.0543→0.1450), NC worse (0.5920→0.5031). PE severely degrades quality even at full supervision.
- **vs EXP-06** (10% labels + PE): CD comparable (0.1515→0.1450), NC comparable (0.5059→0.5031). Full labels provide virtually no improvement when PE is active.
- **vs EXP-07** (5% labels + PE): CD comparable (0.1448→0.1450), NC comparable (0.5074→0.5031). All three PE experiments produce nearly identical poor results.
- **vs EXP-01** (baseline): CD 2.4x worse (0.0593→0.1450), NC worse (0.5522→0.5031).
- **Note**: **Critical finding — PE L=6 is catastrophic regardless of supervision level.** EXP-06 (10%), EXP-07 (5%), and EXP-09 (100%) all produce CD ~0.145, NC ~0.50. The problem is not "PE needs more labels" — PE L=6 fundamentally breaks reconstruction in this DeepSDF setup. Root cause confirmed: PE amplifies high-frequency oscillations in the SDF at evaluation grid points far from the training data distribution (cube corners vs near-surface training points). NC std ~0.01 across all PE experiments confirms PE collapses normal diversity. This makes EXP-08 (PE + L_2nd) unlikely to succeed unless L_2nd specifically suppresses PE's extrapolation artifacts.

### EXP-06 — 10% labels + Eikonal + PE L=6 (seed 456)
- **Date**: 2026-04-04
- **Config**: ratio=0.1, eikonal=on, PE=L=6, epochs=3000, batch=16384
- **Data**: 300 ShapeNet shapes (airplane/chair/table), 250K sup/unsup points each
- **Training**: ~4hr on TC2 (A40 GPU), L_sdf=0.0324 final, L_eik=0.727, L_z=0.00101
- **Eval** (MC res=128, IoU skipped, 0/300 success, 300 failures):
  - **CD**: mean=0.1380, std=0.0426, min=0.0504, max=0.2516
  - **NC**: mean=0.5097, std=0.0113, min=0.4804, max=0.5413
- **vs EXP-06 s42**: CD slightly better (0.1515→0.1380), NC comparable (0.5059→0.5097).
- **vs EXP-06 s123**: CD near-identical (0.1375→0.1380), NC near-identical (0.5087→0.5097).
- **Note**: Extremely consistent with s123. L_eik much higher (0.727 vs 0.036 for s123) but reconstruction quality identical — PE dominates the failure mode.

### EXP-06 — 3-seed summary (seeds 42, 123, 456)
- **CD**: mean=0.1399, std=0.0031, **CV=0.022 < 0.2** — 3 seeds sufficient, no expansion needed
- **NC**: mean=0.5080, std=0.0022, CV=0.004
- **vs EXP-04 3-seed** (no PE): CD 2.8x worse (0.0496→0.1399), NC worse (0.5987→0.5080)
- **Conclusion**: PE L=6 is catastrophic and highly reproducible across seeds. The near-zero NC variance (CV=0.004) confirms PE collapses all shapes to similar poor-quality reconstructions. (Updated with fixed evaluator reruns — all seeds now 300/300 shapes.)

### EXP-08 — 10% labels + Eikonal + PE L=6 + L_2nd (seed 42)
- **Date**: 2026-04-02
- **Config**: ratio=0.1, eikonal=on, PE=L=6, L_2nd=0.01, epochs=3000, batch=8192
- **Data**: 300 ShapeNet shapes (airplane/chair/table), 250K sup/unsup points each
- **Training**: ~4hr on TC2 (A40 GPU), L_sdf=0.0367 final, L_eik=0.020, L_2nd=0.00061, L_z=0.00206
- **Eval** (MC res=128, IoU skipped, 0/300 success, 300 failures):
  - **CD**: mean=0.1443, std=0.0448, min=0.0519, max=0.2645
  - **NC**: mean=0.5053, std=0.0119, min=0.4639, max=0.5397
- **vs EXP-06** (same but no L_2nd): CD comparable (0.1515→0.1443), NC comparable (0.5059→0.5053). Second-order regularization provides negligible improvement.
- **vs EXP-04** (10% labels, no PE): CD 2.4x worse (0.0609→0.1443), NC worse (0.5805→0.5053). L_2nd cannot compensate for PE's fundamental incompatibility.
- **Note**: Confirms PE L=6 is the root cause, not a regularization gap. All four PE experiments (EXP-06/07/08/09) produce CD ~0.144-0.152 regardless of supervision level (5%-100%) or additional regularization. NC std ~0.012 confirms PE collapses normal diversity. The second-order loss trained normally (L_2nd=0.00061, stable) but had no meaningful effect on reconstruction quality.

### EXP-10 — 100% labels + Eikonal + PE L=4 (seed 42) [follow-up sweep]
- **Date**: 2026-04-04 to 2026-04-05
- **Config**: ratio=1.0, eikonal=on, PE=L=4, epochs=3000, batch=16384
- **Data**: 300 ShapeNet shapes (airplane/chair/table), 250K sup/unsup points each
- **Training**: completed on TC2 (A40 GPU), final train log at epoch 3000:
  - **L_total**: 0.047796
  - **L_sdf**: 0.038596
  - **L_eik**: 0.091996
  - **L_z**: 0.002334
  - **grad_norm_mean**: 0.8433
- **Eval** (job `18123`, MC res=128, IoU skipped, 0/300 success, 300 failures):
  - **CD**: mean=0.1401, std=0.0420, min=0.0504, max=0.2512
  - **NC**: mean=0.5077, std=0.0211, min=0.4338, max=0.5615
- **vs EXP-09** (100% labels + PE L=6): CD slightly better (0.1450→0.1401), NC slightly better (0.5031→0.5077), but still catastrophic overall.
- **vs EXP-02** (100% labels, no PE): CD 2.6x worse (0.0543→0.1401), NC worse (0.5920→0.5077).
- **vs EXP-01** (baseline): CD 2.4x worse (0.0593→0.1401), NC worse (0.5522→0.5077).
- **Note**: Lowering PE from L=6 to L=4 does **not** rescue full-supervision reconstruction. The evaluation JSON has an empty aggregate block because every shape is tagged `failed`, but each per-shape entry still contains CD/NC values; the summary above was computed from those per-shape metrics. Current conclusion: the PE failure is not confined to the highest tested frequency.

### EXP-11 — 10% labels + Eikonal + PE L=4 (seed 42) [follow-up sweep]
- **Date**: 2026-04-04 to 2026-04-05
- **Config**: ratio=0.1, eikonal=on, PE=L=4, epochs=3000, batch=16384
- **Data**: 300 ShapeNet shapes (airplane/chair/table), 250K sup/unsup points each
- **Training**: completed on TC2 (A40 GPU), final train log at epoch 3000:
  - **L_total**: 0.049617
  - **L_sdf**: 0.038623
  - **L_eik**: 0.109936
  - **L_z**: 0.002806
  - **grad_norm_mean**: 0.8236
- **Eval** (job `18180`, MC res=128, IoU skipped, 0/300 success, 300 failures):
  - **CD**: mean=0.1427, std=0.0439, min=0.0528, max=0.2597
  - **NC**: mean=0.5071, std=0.0194, min=0.4437, max=0.5675
- **vs EXP-06** (10% labels + PE L=6): CD comparable (0.1515→0.1427), NC comparable (0.5059→0.5071). Lower-frequency PE does not materially improve the sparse-label PE failure mode.
- **vs EXP-04** (10% labels, no PE): CD 2.3x worse (0.0609→0.1427), NC worse (0.5805→0.5071).
- **Note**: This key low-label comparison confirms that lowering PE from L=6 to L=4 does not rescue 10% supervision. As with EXP-10, the evaluation JSON stores useful per-shape CD/NC values even though all shapes are tagged `failed`, so the summary above was computed from per-shape metrics.

### EXP-12 — 5% labels + Eikonal + PE L=4 (seed 42) [follow-up sweep]
- **Date**: 2026-04-05
- **Config**: ratio=0.05, eikonal=on, PE=L=4, epochs=3000, batch=16384
- **Data**: 300 ShapeNet shapes (airplane/chair/table), 250K sup/unsup points each
- **Training**: completed on TC2 (A40 GPU), final train log at epoch 3000:
  - **L_total**: 0.045941
  - **L_sdf**: 0.037834
  - **L_eik**: 0.081069
  - **L_z**: 0.002240
  - **grad_norm_mean**: 0.8586
- **Eval** (job `18181`, MC res=128, IoU skipped, 0/300 success, 300 failures):
  - **CD**: mean=0.1400, std=0.0426, min=0.0512, max=0.2516
  - **NC**: mean=0.5073, std=0.0178, min=0.4328, max=0.5531
- **vs EXP-07** (5% labels + PE L=6): CD comparable (0.1448→0.1400), NC comparable (0.5074→0.5073).
- **vs EXP-05** (5% labels, no PE): CD 2.8x worse (0.0509→0.1400), NC worse (0.5766→0.5073).
- **Note**: This final L=4 point matches EXP-10 and EXP-11 almost exactly, completing the pattern that PE collapses to the same poor reconstruction regime across 100%, 10%, and 5% supervision even when the encoding frequency is reduced.

## Final Conclusion

**Eikonal regularization enables strong label reduction.** At 10% supervision, Eikonal-regularized DeepSDF matches or beats the fully-supervised baseline across 3 seeds (EXP-04 3-seed CD=0.0496 vs EXP-01 CD=0.0593). Even at 5% supervision (EXP-05 CD=0.0509), performance remains competitive with the 100% baseline.

**Fourier positional encoding (PE) is catastrophic in this setup.** All PE experiments (L=6: EXP-06/07/08/09; L=4: EXP-10/11/12) collapse to CD ~0.14 regardless of supervision level (5%–100%) or additional regularization (L_2nd). Lowering the encoding frequency from L=6 to L=4 does not rescue PE. The failure is highly reproducible across seeds (EXP-06 3-seed CV=0.022).

**Key takeaways:**
1. Eikonal regularization is the primary driver of label efficiency in DeepSDF.
2. 10% labels + Eikonal is the practical sweet spot — 90% label reduction with no quality loss.
3. PE hurts across all tested configurations; future work should investigate alternative high-frequency mechanisms (e.g., hash grids, learned features) rather than Fourier PE frequency sweeps.

## Evaluator Bug Fix (2026-04-06)

Fixed a bug in `src/evaluate.py` where metric computation and mesh export shared a single `try/except` block. For PE experiments, mesh decimation failures caused valid metrics to be marked `"failed"`, producing empty aggregates despite per-shape CD/NC being computed correctly. The fix separates mesh export into its own `try/except` so metric status is independent of export success. Eval reruns submitted for all affected PE experiments (EXP-06/s123, EXP-06/s456, EXP-07, EXP-09, EXP-10, EXP-11, EXP-12). CD/NC values are expected to remain unchanged; only shape success counts and aggregate blocks will be corrected.

## PE Failure Root Cause Analysis (2026-04-15)

Post-hoc diagnostic to identify the exact mechanism behind the universal PE failure
(all 7 PE experiments collapsed to CD ~0.14 regardless of supervision level or frequency).

### Hypothesis tested
Four candidate causes:
1. Input coordinate scaling (normalization to [-1,1])
2. PE frequency too high (L=6 or L=4)
3. Training/inference distribution mismatch (near-surface training vs full [-1,1]³ eval)
4. PE formula implementation detail (π factor)

### Checks performed

**Check A — Training point coverage** (local, parametric data, 80 shapes):
- Supervised points: max radius = 1.0998, zero points at r > 1.5
- Unsupervised points (Eikonal): **max radius = 1.000, zero points at r > 1.0**
- Cube corners are at r = √3 ≈ 1.73 — never sampled during training
- Code: `scripts/preprocess.py:125-143` uses rejection sampling inside unit sphere

**Check B — Sphere-clipped evaluation** (TC2, EXP-09/seed42, 5 shapes, res=128):
- Standard (full [-1,1]³): CD = 0.1926, NC = 0.4953
- Clipped (r ≤ 1.0 sphere, SDF=+1.0 outside): CD = **0.0974**, NC = 0.4970
- CD ratio = 0.506 — **49% improvement from masking OOD corners alone**
- Per-shape: every shape improved substantially (e.g. airplane_0000: 0.207 → 0.104)
- Script: `scripts/check_b_clipped_eval.py`

**Check C — SDF cross-section along diagonal toward cube corner (1,1,1)** (TC2):
- No-PE (EXP-02): 0 sign changes outside r>1.0, SDF monotonically negative (smooth extrapolation)
- PE L=6 (EXP-09): **14 sign changes** outside r>1.0, SDF oscillates between ±0.02
- Oscillation frequency ≈ 0.063 units/cycle matches 2⁵·π ≈ 100 rad/unit (highest PE frequency)
- Plot: `experiments/figures/check_c_sdf_crosssection.png`
- Script: `scripts/check_c_diagonal.py`

**Check D — PE feature distance** (local):
- dist(surface, cube corner) = 2.83 / max possible 8.49 for L=6
- dist(surface, near_surface Δ=0.05) = 2.62 for L=6 — large jump for tiny coordinate change
- PE vectors always lie on hypersphere of radius √(3L); corner features are in OOD region
- Script: inline Python using `src/model.py:FourierPositionalEncoding`

### Conclusion

**Primary cause: training/inference distribution mismatch (hypothesis #3), confirmed by Check B.**

The unsupervised (Eikonal) sampler fills only the unit sphere. The eval grid queries the full
[-1,1]³ cube, including corners at r=1.73 — never seen during training. PE maps those OOD
coordinates to feature vectors with 14 sign-flipping oscillations per diagonal axis. Marching
cubes extracts a phantom surface at every zero-crossing → all 300 shapes fail validation.

Without PE, the MLP extrapolates smoothly in raw xyz space (0 sign changes, monotone SDF).
The no-PE model also extrapolates incorrectly at corners (negative SDF instead of large
positive), but monotonically — no zero-crossings, no phantom surfaces, valid mesh extracted.

**Hypothesis #2 (frequency) is secondary**: Check B shows clipping alone halves CD with L=6
unchanged. High L amplifies oscillation magnitude but the distribution gap is the root cause.

**Implication for future work**: PE can be rescued by one of:
- Extending unsupervised sampling to full [-1,1]³ cube (Eikonal covers corners)
- Clipping the eval grid to inscribed sphere (cosmetic fix, no retraining)
- Replacing Fourier PE with hash encoding (localized, no OOD extrapolation issue)

## Next Steps (as of 2026-04-06)

### Immediate
- Collect rerun results as eval jobs complete on TC2 and update status notes above.
- Regenerate figures if shape success counts change materially.

### Prerequisite: Data Source Upgrade
EXP-01 used parametric meshes with 10K points (pipeline validation only). All production experiments use **ShapeNet meshes** with 250K points.
Steps: (1) watertight mesh scan on target categories, (2) preprocess ShapeNet data with 250K pts, (3) re-run EXP-01 as proper baseline.

### Experiment Execution Order & Rationale

**Phase 1: Baselines (EXP-01, EXP-02)**
- **EXP-01** (re-run with 250K data): Fully-supervised, no Eikonal, no PE.
  - WHY: Establishes the vanilla DeepSDF baseline. All other experiments compare against this.
  - CONFIG: `supervision_ratio=1.0 use_eikonal=false use_pe=false`
- **EXP-02**: Full supervision + Eikonal.
  - WHY: Isolates the effect of Eikonal regularization at full supervision. If EXP-02 > EXP-01, Eikonal helps even with abundant labels.
  - CONFIG: `supervision_ratio=1.0 use_eikonal=true use_pe=false`

**Phase 2: Label Reduction (EXP-03, EXP-04, EXP-05)**
- **EXP-03**: 50% labels + Eikonal.
  - WHY: Tests if Eikonal compensates for halving labels. Compare CD/NC vs EXP-02.
  - CONFIG: `supervision_ratio=0.5 use_eikonal=true use_pe=false`
- **EXP-04**: 10% labels + Eikonal (KEY experiment).
  - WHY: Core research question — can Eikonal maintain quality with 90% fewer labels? This is the main data point for the label efficiency curve.
  - CONFIG: `supervision_ratio=0.1 use_eikonal=true use_pe=false`
  - NOTE: Run 3 seeds. If CD CV > 0.2, expand to 5 seeds for statistical confidence.
- **EXP-05**: 5% labels + Eikonal.
  - WHY: Tests the floor — at what label fraction does quality collapse?
  - CONFIG: `supervision_ratio=0.05 use_eikonal=true use_pe=false`

**Phase 3: Positional Encoding (EXP-06, EXP-07, EXP-09)**
- **EXP-06**: 10% labels + Eikonal + PE (KEY comparison vs EXP-04).
  - WHY: Tests if Fourier PE helps capture high-frequency details with few labels. Direct comparison with EXP-04 isolates PE contribution.
  - CONFIG: `supervision_ratio=0.1 use_eikonal=true use_pe=true pe_levels=6`
  - NOTE: Run 3 seeds. If CD CV > 0.2, expand to 5 seeds.
- **EXP-07**: 5% labels + Eikonal + PE.
  - WHY: Can PE + Eikonal rescue the minimal-label regime that EXP-05 tested?
  - CONFIG: `supervision_ratio=0.05 use_eikonal=true use_pe=true pe_levels=6`
- **EXP-09**: Full supervision + PE (quality ceiling).
  - WHY: Upper bound — what's the best achievable quality with all features on?
  - CONFIG: `supervision_ratio=1.0 use_eikonal=true use_pe=true pe_levels=6`

**Phase 4: Advanced Regularization (EXP-08)**
- **EXP-08**: 10% labels + Eikonal + PE + L_2nd.
  - WHY: Tests if second-order divergence loss provides additional benefit on top of Eikonal + PE.
  - CONFIG: `supervision_ratio=0.1 use_eikonal=true use_pe=true pe_levels=6 lambda_2nd=0.01 batch_size=8192`
  - NOTE: Reduced batch size (8192) due to higher memory from second-order gradients.

### Practical Notes
- **Job limits**: max 2 concurrent jobs, 1 GPU, 6hr wall time per job (normal QoS). Extended QoS pending.
- **IoU**: Use `--skip_iou` for quick evals. Full IoU only needed for final results.
- **MC resolution**: Use `mc_resolution=64` for dev, `128` for production evals.
- **After each experiment**: run `/log-experiment EXP-XX` then `/disk-check` after every 3 experiments.

## Notes

- Divergence threshold updated from 0.5 to 0.95 (previous threshold too strict — see EXP-01 false positive)
- Seed expansion: if CD CV > 0.2 for EXP-04 or EXP-06, expand to 5 seeds (add 789, 101)
- Default seeds: 42, 123, 456
- IoU computation is extremely slow on CPU (mesh.contains ray casting). Use --skip_iou for quick evals.
- Data was preprocessed with 10K points (dev). Reprocessing with 250K in progress on TC2.
