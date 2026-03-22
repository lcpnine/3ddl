---
name: log-experiment
description: Log experiment results, what changed, what was expected, and what actually happened. Use after results.json is available, or when the user says "log results for EXP-XX".
argument-hint: <EXP-ID>
disable-model-invocation: false
allowed-tools: Read, Write, Bash
---

Log results for experiment $ARGUMENTS.

## Steps

1. **Read results**: Load `experiments/$ARGUMENTS/seed*/results.json` for all seeds.

2. **Compute statistics**: Calculate mean ± std for:
   - Chamfer Distance (CD)
   - Normal Consistency (NC)
   - IoU@128³ and IoU@256³
   - Eikonal Deviation
   - Check if CV > 0.2 (flag for seed expansion)

3. **Check divergence**: Mark run as DIVERGED if L_sdf at epoch 500 > 50% of epoch 10 value.

4. **Append to `experiments/experiment_log.md`**:

```markdown
### Results: $ARGUMENTS | $(date)

**Status**: COMPLETE / DIVERGED (N/3 seeds)

#### Quantitative Results
| Metric         | Mean    | Std     | Notes          |
|----------------|---------|---------|----------------|
| CD             | X.XXXX  | X.XXXX  |                |
| Normal Cons.   | X.XXX   | X.XXX   |                |
| IoU@128³       | X.XXX   | X.XXX   |                |
| IoU@256³       | X.XXX   | X.XXX   |                |
| Eikonal Dev.   | X.XXXX  | X.XXXX  |                |

**CV Check**: CD CV = X.XX → [OK / ⚠️ EXPAND TO 5 SEEDS]

#### What We Expected
[Fill: based on hypothesis from pre-run entry]

#### What Actually Happened
[Fill: compare to expectation — better/worse/same, why?]

#### Key Observations
- Training stability: [stable / unstable, gradient norms]
- L_eik/L_sdf ratio at convergence: X.X
- Category breakdown: chair=X, airplane=X, table=X

#### Diagnosis Triggered?
- [ ] Gradient norm deviation > 0.1 → action taken
- [ ] L_eik > 10x L_sdf → action taken
- [ ] PE degradation → EXP-06b/07b needed
- [ ] Seed expansion needed

#### Action Items for Next Experiment
1. [specific change to make, e.g. "increase warmup to 200"]
2. [hyperparameter to tune]
```

5. **Update `experiments/progress_table.md`**: Append one row to the master results table.

6. **Auto-commit results**:

After writing to experiment_log.md and progress_table.md:

```bash
git add experiments/experiment_log.md \
        experiments/progress_table.md \
        experiments/$EXP_ID/seed*/results.json \
        configs/$EXP_ID_seed*.yaml
git commit -m "results($EXP_ID): CD=$(mean_cd)±$(std_cd) NC=$(mean_nc) | $(short_summary)"
```

Commit message format example:
`results(EXP-04): CD=0.0043±0.0003 NC=0.87 | 10% supervision Eikonal-only`

7. **Context management**: If this is the 3rd or more experiment logged in this session, run `/compact` with instructions to preserve the current experiment state and action items before proceeding to the next experiment.
