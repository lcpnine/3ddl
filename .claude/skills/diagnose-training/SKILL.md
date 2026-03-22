---
name: diagnose-training
description: Diagnose training issues from logs. Use when training looks unstable, loss diverges, or metrics are unexpectedly bad. Reads training logs and applies Diagnosis Guide from project plan.
argument-hint: <EXP-ID> [seed]
allowed-tools: Read, Bash, Grep
---

Diagnose training for $ARGUMENTS.

## Steps

1. Read `experiments/$ARGUMENTS[0]/seed$ARGUMENTS[1]/train.log`
2. Extract: gradient norms per epoch, L_eik values, L_sdf values, L_eik/L_sdf ratio
3. Check against Diagnosis Guide thresholds:
   - Gradient norm deviation > 0.1 → recommend lambda_eik increase
   - Eikonal deviation > 0.05 → check gradient clipping active
   - L_eik > 10x L_sdf before warmup completes → extend warmup_epochs
   - Loss diverges in early epochs → reduce LR to 1e-4
   - CD degrades sharply below 25% → increase lambda_eik or unsupervised points
4. If PE experiment (EXP-06/07): compare CD to corresponding no-PE experiment
   - If PE CD > no-PE CD → recommend EXP-06b/07b with L=4
5. Output structured diagnosis report to `experiments/$ARGUMENTS[0]/diagnosis.md`
6. Suggest specific config changes with exact values

## Output format
```markdown
# Diagnosis: $ARGUMENTS
**Issue detected**: [description]
**Root cause**: [from Diagnosis Guide]
**Recommended action**: [exact config change]
**Expected outcome**: [what should change]
```
