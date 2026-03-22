---
name: submit-slurm
description: Submit multiple SLURM jobs in parallel (max 2 concurrent per QoS). Use when user says "submit jobs". Manages job queue and dependency chains.
argument-hint: <EXP-ID-list> e.g. "EXP-03 EXP-04 seeds 42 123 456"
disable-model-invocation: true
allowed-tools: Bash, Read, Write
---

Submit SLURM jobs for $ARGUMENTS.

All commands run on TC2 via SSH (`ssh tc2 '...'`). Never run SLURM commands locally.

## TC2 Constraints
- Partition: MGPU-TC2, QoS: normal
- Max 2 concurrent jobs per user
- 1 GPU, 10 CPU, 30GB mem, 6hr wall time per job
- A train+eval dependency chain counts as 2 jobs — so only 1 chain can run at a time

## Steps

1. Parse experiment IDs and seeds from $ARGUMENTS
2. Check current queue: `ssh tc2 'squeue -u yutaek001 --format="%j %T" | grep -c RUNNING'`
3. Submit in batches respecting the 2-job limit:
   - If 2 jobs already running, wait until a slot opens
   - For each experiment: submit train → chain eval via --dependency
   ```bash
   TRAIN_JOB=$(ssh tc2 'cd ~/3ddl && sbatch --parsable job_train.sh --config configs/EXP-XX_seedYY.yaml')
   ssh tc2 "cd ~/3ddl && sbatch --dependency=afterok:$TRAIN_JOB eval_job.sh --exp_dir experiments/EXP-XX/seedYY"
   ```
4. Log all job IDs to `experiments/job_registry.json`:
   ```json
   { "EXP-03_seed42": { "train_job": 12345, "eval_job": 12346, "status": "RUNNING" } }
   ```
5. Print submission summary with estimated completion times
6. Monitor: `ssh tc2 'squeue -u yutaek001 -la'`
