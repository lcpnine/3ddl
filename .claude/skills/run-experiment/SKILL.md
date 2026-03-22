---
name: run-experiment
description: Run a DeepSDF experiment (EXP-01 through EXP-09). Use when the user says "run EXP-XX". Generates config, submits SLURM job, and chains evaluation job.
argument-hint: <EXP-ID> <seed>
disable-model-invocation: false
allowed-tools: Bash, Read, Write
---

Run experiment $ARGUMENTS[0] with seed $ARGUMENTS[1].

## Steps

1. **Config generation**: Read `configs/template.yaml`, override fields for this EXP-ID:
   - supervision_ratio, use_eikonal, use_pe, pe_levels, warmup_epochs (ratio-dependent)
   - seed = $ARGUMENTS[1]
   - Save to `configs/$ARGUMENTS[0]_seed$ARGUMENTS[1].yaml`

2. **Commit config**:
   ```bash
   git add configs/$ARGUMENTS[0]_seed$ARGUMENTS[1].yaml
   git commit -m "config($ARGUMENTS[0]/seed$ARGUMENTS[1]): supervision=$(ratio) pe=$(use_pe) warmup=$(warmup_epochs)"
   ```

3. **Pre-run log entry**: Before submitting, append to `experiments/experiment_log.md`:
   ```
   ## $ARGUMENTS[0] | seed $ARGUMENTS[1] | $(date)
   **Status**: SUBMITTED
   **Config**: [paste key params]
   **Hypothesis**: [what we expect based on EXP design]
   **Expected CD range**: [based on prior experiments if available]
   ```

4. **SLURM submission**:
   ```bash
   TRAIN_JOB=$(sbatch --parsable job_train.sh --config configs/$ARGUMENTS[0]_seed$ARGUMENTS[1].yaml)
   sbatch --dependency=afterok:$TRAIN_JOB eval_job.sh --exp_dir experiments/$ARGUMENTS[0]/seed$ARGUMENTS[1]
   echo "Train job: $TRAIN_JOB submitted"
   ```

5. **Confirm**: Print job ID and expected wall time to console.
