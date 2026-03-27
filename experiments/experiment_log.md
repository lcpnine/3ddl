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
