---
name: generate-figures
description: Generate project figures from experiment results. Use when experiments are complete or user says "generate figures". Creates label efficiency curve, ablation bar chart, gradient histograms.
argument-hint: [figure-type] e.g. "label-efficiency" or "all"
allowed-tools: Bash, Read, Write
---

Generate figures for $ARGUMENTS.

## Steps

Based on $ARGUMENTS (or "all" if not specified), generate:

### label-efficiency
Run `python scripts/plot_label_efficiency.py` using results from EXP-01,02,03,04,05,06,07.
- X: Supervision Ratio (%), Y: Chamfer Distance
- Two lines: with PE (EXP-06,07) vs without PE (EXP-04,05)
- Annotate PE transition threshold if visible
- Save to `figures/label_efficiency_curve.pdf` and `.png`

### ablation-bar
Run `python scripts/plot_ablation.py`
- X: Experiment conditions, Y: CD and Normal Consistency (dual axis)
- Save to `figures/ablation_bar_chart.pdf`

### training-curves
Read `experiments/EXP-XX/seed42/train_log.csv`, plot:
- L_total, L_sdf, L_eik per epoch
- L_eik/L_sdf ratio subplot
- Warmup ramp visible
- Save per-experiment to `figures/training_curves/`

### iou-resolution
Compare IoU@128³ vs IoU@256³ across experiments
- Save to `figures/iou_resolution_comparison.pdf`

### mesh-grid
Run marching cubes on best checkpoint, render GT vs reconstructed per category
- Save to `figures/mesh_comparison_grid.png`

After generation, log to `experiments/experiment_log.md`:
```markdown
## Figures Generated | $(date)
- [x] Label efficiency curve
- [x] Ablation bar chart
[etc.]
```
