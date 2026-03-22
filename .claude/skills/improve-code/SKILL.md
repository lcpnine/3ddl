---
name: improve-code
description: Apply a code improvement based on diagnosis results or experiment findings. Use when user says "fix [issue]" or after diagnose-training triggers an action. Always commits the change with full context of what/why/expected.
argument-hint: <reason> e.g. "extend-warmup EXP-04" or "fix-gradient-clip" or "add-pe-L4"
allowed-tools: Read, Write, Bash
---

Apply code improvement: $ARGUMENTS

## Steps

1. **Read current state**:
   - Read the relevant source file (model.py / losses.py / train.py / config)
   - Read `experiments/experiment_log.md` to understand what triggered this change

2. **Make the change**: Edit the file(s) with the specific improvement

3. **Log the change BEFORE committing** — append to `experiments/experiment_log.md`:

```markdown
## Code Change | $(date)
**File(s) modified**: [list files]
**What changed**: [exact change, e.g. "warmup_epochs: 150 → 200 for 5% supervision"]
**Why**: [diagnosis finding that triggered this]
**Expected effect**: [what metric should improve and by how much]
**Commit**: [will be filled after commit]
```

4. **Commit**:
```bash
git add [modified files] experiments/experiment_log.md
git commit -m "improve: $ARGUMENTS

What: [one-line description of change]
Why: [diagnosis finding]
Expected: [expected metric improvement]
Triggered by: [EXP-ID that surfaced the issue]"
```

5. **Update log** with the commit hash:
```bash
HASH=$(git rev-parse --short HEAD)
# Append hash to the log entry just written
```

## Commit Message Examples
- `improve: extend warmup 150→200 for 5% supervision (EXP-05 L_eik dominated)`
- `improve: reduce PE levels L=6→L=4 (EXP-06 CD worse than EXP-04)`
- `improve: add gradient norm logging per epoch (missing from train.py)`
- `improve: fix CD sampling 10k→30k points (evaluate.py protocol violation)`

6. **Context management**: After committing the improvement, if context feels heavy (many experiments processed this session), run `/compact` preserving: current EXP-ID, the change just made, and next steps.
