# TC2 GPU Cluster — Command Reference

Quick reference for submitting, monitoring, and retrieving results on the CCDS TC2 cluster.

**Connection:** `ssh tc2` (alias configured in ~/.ssh/config)
**Home directory:** `/home/msai/yutaek001`
**Project directory:** `~/3ddl`

---

## 1. Job Submission

### Submit a single experiment (train + eval chain)
```bash
# On TC2:
cd ~/3ddl
./slurm/submit.sh EXP-01 42

# With config overrides:
./slurm/submit.sh EXP-04 42 "supervision_ratio=0.1 use_eikonal=true"

# With PE enabled:
./slurm/submit.sh EXP-06 42 "supervision_ratio=0.1 use_eikonal=true use_pe=true pe_levels=6"
```

### Submit manually (without helper script)
```bash
# Training job
OVERRIDES="exp_name=EXP-01 seed=42" sbatch --job-name=EXP-01_s42_train slurm/job_train.sh

# Eval job chained after training (replace 12345 with actual train job ID)
EXP_DIR=experiments/EXP-01/seed42 sbatch --dependency=afterok:12345 --job-name=EXP-01_s42_eval slurm/job_eval.sh
```

### Submit two experiments in parallel (max 2 concurrent jobs)
```bash
./slurm/submit.sh EXP-01 42
./slurm/submit.sh EXP-01 123
```

---

## 2. Monitoring Running Jobs

### Check your job queue
```bash
ssh tc2 "squeue -u yutaek001"
```

### Detailed job queue (shows time limit, node, reason)
```bash
ssh tc2 "squeue -u yutaek001 -la"
```

### Watch training output in real-time
```bash
# First, find the exact log filename
ssh tc2 "ls -lt ~/3ddl/logs/ | head -10"

# Then tail it (replace with actual filename)
ssh tc2 "tail -f ~/3ddl/logs/output_EXP-01_s42_train_12345.out"
```

### Check training progress from CSV log
```bash
# Last 5 lines of training log
ssh tc2 "tail -5 ~/3ddl/experiments/EXP-01/seed42/train.log"

# Check L_sdf trend (epoch, L_sdf columns)
ssh tc2 "awk -F',' 'NR==1 || NR%100==0' ~/3ddl/experiments/EXP-01/seed42/train.log"
```

### Check what GPU card your job is using
```bash
# Replace <jobid> with your SLURM job ID
ssh tc2 "scontrol show -d jobid <jobid>" | grep JOB_GRES
```

---

## 3. Checking Completed Jobs

### Job history (today's jobs)
```bash
ssh tc2 "sacct -u yutaek001 --format=jobid,jobname%+20,state,elapsed,exitcode,reason%+20"
```

### Job history (all time, exclude batch steps)
```bash
ssh tc2 "sacct -u yutaek001 --format=jobid,jobname%+20,qos,alloctres%+50,nodelist,start,elapsed,state,reason%+20 | grep -wv batch"
```

### Job efficiency report (CPU/memory/GPU utilization)
```bash
ssh tc2 "seff <jobid>"
```

### Custom job history script (if available on TC2)
```bash
ssh tc2 "myjobhistory"
```

---

## 4. Retrieving Results

### Check if results.json exists
```bash
ssh tc2 "ls -la ~/3ddl/experiments/EXP-01/seed42/results.json"
```

### View results summary
```bash
ssh tc2 "python3 -m json.tool ~/3ddl/experiments/EXP-01/seed42/results.json | head -30"
```

### Download results to local machine
```bash
# Single experiment
scp tc2:~/3ddl/experiments/EXP-01/seed42/results.json experiments/EXP-01/seed42/

# All results at once
scp -r tc2:~/3ddl/experiments/ experiments/

# Training log only
scp tc2:~/3ddl/experiments/EXP-01/seed42/train.log experiments/EXP-01/seed42/
```

### Download reconstructed meshes
```bash
scp -r tc2:~/3ddl/experiments/EXP-01/seed42/reconstructions/ experiments/EXP-01/seed42/reconstructions/
```

---

## 5. Checking SLURM Errors

### View error log
```bash
# Find error files
ssh tc2 "ls -lt ~/3ddl/logs/error_* | head -10"

# View specific error log (replace with actual filename)
ssh tc2 "cat ~/3ddl/logs/error_EXP-01_s42_train_12345.err"
```

### Common SLURM errors and causes
| Error in REASON column | Cause | Fix |
|------------------------|-------|-----|
| QOSMaxJobsPerUserLimit | Already 2 jobs running | Wait for a job to finish |
| QOSMaxCpuPerUserLimit | CPU request exceeds 10 total | Reduce --cpus-per-task |
| QOSMaxMemoryPerUser | Memory request exceeds 30G total | Reduce --mem |
| QOSMaxGRESPerUser | GPU request exceeds 1 | Only 1 GPU job at a time |
| Priority | Other jobs have priority | Wait |
| Resources | No nodes available | Wait |

---

## 6. Disk Management

### Check disk usage
```bash
ssh tc2 "du -sh ~/3ddl/"
ssh tc2 "du -sh ~/3ddl/experiments/ ~/3ddl/data/ ~/.conda/"
```

### Interactive disk explorer
```bash
# ncdu is interactive — run directly on TC2 (not via ssh command)
# SSH in first, then run:
ncdu ~/
```

### Delete intermediate checkpoints (keep latest + best only)
```bash
ssh tc2 "find ~/3ddl/experiments/ -name 'epoch_*.pt' -delete"
```

### Check total quota usage
```bash
ssh tc2 "df -h /home/msai/yutaek001 2>/dev/null || du -sh ~/"
```

---

## 7. Cancelling Jobs

### Cancel a specific job
```bash
ssh tc2 "scancel <jobid>"
```

### Cancel all your jobs
```bash
ssh tc2 "scancel -u yutaek001"
```

### Cancel only pending jobs (keep running ones)
```bash
ssh tc2 "scancel -u yutaek001 -t PENDING"
```

---

## 8. Cluster Status

### Check node availability
```bash
ssh tc2 "sinfo -N -l"
```

### Check specific nodes in MGPU-TC2 partition
```bash
ssh tc2 "scontrol show node TC2N0[1-6]" | grep -E "NodeName|State|AllocTRES|FreeMem"
```

### Find idle nodes
```bash
ssh tc2 "sinfo -N -p MGPU-TC2 | grep idle"
```

---

## 9. Conda Environment

### First-time setup on TC2
```bash
ssh tc2
cd ~/3ddl
module load anaconda
eval "\$(conda shell.bash hook)"
conda env create -f environment.yml
```

### Activate environment (in SSH session)
```bash
module load anaconda
eval "$(conda shell.bash hook)"
conda activate deepsdf
```

### Install additional packages
```bash
conda install -c conda-forge <package>   # prefer conda
pip install <package>                     # only if not in conda
```

### Check environment size
```bash
ssh tc2 "du -sh ~/.conda/envs/deepsdf/"
```

---

## 10. File Transfer (Local ↔ TC2)

### Upload code to TC2
```bash
# Entire project (excluding data and git)
rsync -avz --exclude='data/' --exclude='.git/' --exclude='experiments/' \
    /Users/lcpnine/3ddl/ tc2:~/3ddl/
```

### Upload specific files
```bash
scp src/train.py tc2:~/3ddl/src/
scp configs/config.yaml tc2:~/3ddl/configs/
```

### Download everything
```bash
rsync -avz tc2:~/3ddl/experiments/ experiments/
```

---

## Quick Workflows

### Full experiment cycle
```bash
# 1. Upload latest code
rsync -avz --exclude='data/' --exclude='.git/' --exclude='experiments/' \
    /Users/lcpnine/3ddl/ tc2:~/3ddl/

# 2. Submit (on TC2)
ssh tc2 "cd ~/3ddl && ./slurm/submit.sh EXP-01 42"

# 3. Monitor
ssh tc2 "squeue -u yutaek001"

# 4. Check results when done
ssh tc2 "python3 -m json.tool ~/3ddl/experiments/EXP-01/seed42/results.json | head -30"

# 5. Download results
scp tc2:~/3ddl/experiments/EXP-01/seed42/results.json experiments/EXP-01/seed42/
```

### Quick status check (all-in-one)
```bash
ssh tc2 "echo '=== JOBS ===' && squeue -u yutaek001 2>/dev/null; echo '=== DISK ===' && du -sh ~/3ddl/experiments/ ~/3ddl/data/ 2>/dev/null; echo '=== LATEST RESULTS ===' && ls -lt ~/3ddl/experiments/*/seed*/results.json 2>/dev/null | head -5"
```
