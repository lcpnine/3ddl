---
name: checkpoint-stage
description: Mark completion of a project stage with a git tag and summary. Use when Stage 1/2/3 is complete or a major milestone is reached.
argument-hint: <stage> e.g. "stage1" or "baseline-working" or "pe-experiments-done"
allowed-tools: Bash, Read, Write
---

Create stage checkpoint: $ARGUMENTS

## Steps

1. Run final disk check
2. Ensure all experiment results are committed
3. Update CLAUDE.md "Current Stage" field
4. Commit the CLAUDE.md update:
```bash
git add CLAUDE.md
git commit -m "milestone: $ARGUMENTS complete"
git tag -a $ARGUMENTS -m "$(date): $ARGUMENTS — [one-line summary of what's in this stage]"
```
5. Print summary: what experiments are done, what's next
