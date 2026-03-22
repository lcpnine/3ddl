---
name: disk-check
description: Check disk usage and clean up intermediate checkpoints. Auto-invoked after each experiment batch. Warns if usage > 80GB.
disable-model-invocation: false
allowed-tools: Bash, Write
---

## Steps

1. Run `ncdu --export - ~ | head -50` or `du -sh ~/experiments ~/data ~/envs`
2. Check total usage vs 100GB quota
3. If > 80GB: identify and delete intermediate checkpoints
   - Keep: latest full checkpoint (for resume) + best weights-only checkpoint (for eval)
   - Delete: all epoch-100, epoch-200, ... checkpoints except latest
4. Log to `experiments/disk_usage_log.md`:
   ```
   $(date) | Total: X GB | Action: [deleted X GB of intermediate ckpts from EXP-XX]
   ```
5. If > 90GB: STOP and alert user before submitting new jobs
