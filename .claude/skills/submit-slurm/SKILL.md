---
name: submit-slurm
description: Submit multiple SLURM jobs in parallel (max 2 concurrent per QoS). Use when user says "submit jobs". Manages job queue and dependency chains.
argument-hint: <EXP-ID-list> e.g. "EXP-03 EXP-04 seeds 42 123 456"
disable-model-invocation: true
allowed-tools: Bash, Read, Write
---

Submit SLURM jobs for $ARGUMENTS.

## Steps

1. Parse experiment IDs and seeds from $ARGUMENTS
2. Check current queue: `squeue -u $USER --format="%j %T" | grep -c RUNNING`
3. Submit in batches of 2 (QoS limit):
   - If 2 jobs already running, wait or queue locally
   - For each job pair: submit train → chain eval via --dependency
4. Log all job IDs to `experiments/job_registry.json`:
   ```json
   { "EXP-03_seed42": { "train_job": 12345, "eval_job": 12346, "status": "RUNNING" } }
   ```
5. Print submission summary with estimated completion times
