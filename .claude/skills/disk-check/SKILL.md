---
name: disk-check
description: Check disk usage and clean up intermediate checkpoints. Auto-invoked after each experiment batch. Warns if usage > 80GB.
disable-model-invocation: false
allowed-tools: Bash, Write
---

All commands run on TC2 via SSH (`ssh tc2 '...'`). Home directory: `/home/msai/yutaek001`.

## Steps

1. Check disk usage on TC2:
   ```bash
   ssh tc2 'du -sh ~/3ddl/experiments ~/3ddl/data ~/*.conda 2>/dev/null; df -h /home/msai/yutaek001'
   ```
2. Check total usage vs 100GB quota
3. If > 80GB: identify and delete intermediate checkpoints on TC2
   - Keep: latest full checkpoint (for resume) + best weights-only checkpoint (for eval)
   - Delete: all epoch-100, epoch-200, ... checkpoints except latest
   ```bash
   ssh tc2 'find ~/3ddl/experiments -name "checkpoint_epoch_*.pt" | sort | head -n -1 | xargs rm -f'
   ```
4. Log to local `experiments/disk_usage_log.md`:
   ```
   $(date) | Total: X GB | Action: [deleted X GB of intermediate ckpts from EXP-XX]
   ```
5. If > 90GB: STOP and alert user before submitting new jobs
6. Remind: TC2 has NO backup service. Consider `scp` critical results to local machine.
