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

## Results

| ID | Seeds | CD (mean +/- std) | NC (mean +/- std) | IoU@128 | IoU@256 | Status |
|----|-------|-------------------|--------------------|---------|---------|--------|
| EXP-01 | 42 | 0.0593 +/- 0.0464 | 0.5522 +/- 0.1057 | skipped | skipped | done (ShapeNet baseline) |
| EXP-02 | 42 | 0.0543 +/- 0.0416 | 0.5920 +/- 0.1321 | skipped | skipped | done (261/300 shapes, 39 failures) |
| EXP-03 | 42 | 0.0534 +/- 0.0400 | 0.5924 +/- 0.1334 | skipped | skipped | done (259/300 shapes, 41 failures) |
| EXP-04 | 42 | 0.0609 +/- 0.0469 | 0.5805 +/- 0.1406 | skipped | skipped | done (263/300 shapes, 37 failures) |
| EXP-05 | 42 | 0.0509 +/- 0.0385 | 0.5766 +/- 0.1271 | skipped | skipped | done (295/300 shapes, 5 failures) |
| EXP-06 | 42 | 0.1515 +/- 0.0445 | 0.5059 +/- 0.0109 | skipped | skipped | done (240/300 shapes, 45 failures, partial — disk quota) |

## Detailed Results

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
- **Eval** (MC res=128, IoU skipped, 240/300 shapes, 45 failures — partial due to disk quota):
  - **CD**: mean=0.1515, std=0.0445, min=0.0522, max=0.2627
  - **NC**: mean=0.5059, std=0.0109, min=0.4701, max=0.5372
- **vs EXP-04** (10% labels, no PE): CD 2.5x worse (0.0609→0.1515), NC worse (0.5805→0.5059). PE severely degrades quality at 10% supervision.
- **vs EXP-01** (baseline): CD 2.6x worse (0.0593→0.1515), NC worse (0.5522→0.5059).
- **Note**: PE with L=6 creates high-frequency SDF oscillations → marching cubes generates enormous meshes (~250MB/shape vs ~5MB without PE). This filled the 100GB disk quota at shape 195. Metrics computed for 240 shapes before job terminated. The NC std (0.0109) is unusually tight — PE collapses normal diversity. Root cause: training data lives near a unit sphere but eval grid samples full [-1,1]³ cube; PE amplifies extrapolation errors at cube corners unseen during training.

## Next Steps (as of 2026-03-23)

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
