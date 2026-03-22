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
| | | | | | | |

## Notes

- Divergence: L_sdf at epoch 500 must be < 50% of epoch 10 value
- Seed expansion: if CD CV > 0.2 for EXP-04 or EXP-06, expand to 5 seeds (add 789, 101)
- Default seeds: 42, 123, 456
