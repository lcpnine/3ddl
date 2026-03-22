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

4. **Sync config to TC2**:
   ```bash
   scp configs/$ARGUMENTS[0]_seed$ARGUMENTS[1].yaml tc2:/home/msai/yutaek001/3ddl/configs/
   ```

5. **SLURM submission via SSH** (all sbatch commands run on TC2, not locally):
   ```bash
   TRAIN_JOB=$(ssh tc2 'cd ~/3ddl && sbatch --parsable job_train.sh --config configs/$ARGUMENTS[0]_seed$ARGUMENTS[1].yaml')
   ssh tc2 "cd ~/3ddl && sbatch --dependency=afterok:$TRAIN_JOB eval_job.sh --exp_dir experiments/$ARGUMENTS[0]/seed$ARGUMENTS[1]"
   echo "Train job: $TRAIN_JOB submitted"
   ```
   Note: TC2 limits — partition=MGPU-TC2, qos=normal, max 2 concurrent jobs, 1 GPU, 30GB mem, 6hr wall time.
   Job scripts must include: `module load anaconda && eval "$(conda shell.bash hook)" && conda activate yt3dl`

6. **Confirm**: Print job ID. Max wall time is 6 hours — flag if experiment is expected to exceed this.
