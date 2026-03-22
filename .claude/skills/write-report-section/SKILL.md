---
name: write-report-section
description: Draft a section of the final report based on experiment results. Use when user says "draft [section name]". Reads experiment_log.md and generates conference-style prose.
argument-hint: <section> e.g. "related-work" "experiments" "discussion"
disable-model-invocation: true
allowed-tools: Read, Write
---

Draft report section: $ARGUMENTS

## Steps

1. Read `experiments/experiment_log.md` for all completed results
2. Read `literature_review.md` for citations
3. Draft the specified section following the positioning in the project plan:
   - **related-work**: IGR→SAL→StEik narrative + PE positioning + GenSDF comparison
   - **experiments**: Table of all EXP results, describe setup
   - **discussion**: PE-supervision threshold finding, reference arXiv 2401.01391
   - **conclusion**: Primary contribution framing

4. Save draft to `report/sections/$ARGUMENTS_draft.md` with timestamp
5. Append summary to `experiments/experiment_log.md`:
   ```
   ## Report Section Draft: $ARGUMENTS | $(date)
   Key claims made: [list]
   Figures referenced: [list]
   TODO before submission: [list]
   ```
