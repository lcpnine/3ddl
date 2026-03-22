# Semi-Supervised DeepSDF Project

## Critical Rules
- All training via sbatch ONLY — never run training on head node
- After each experiment: always run /log-experiment before starting next
- Disk: run /disk-check after every batch of 3+ experiments
- Divergence check: L_sdf at epoch 500 must be < 50% of epoch 10 value
- Seed expansion: if CD CV > 0.2 for EXP-04 or EXP-06, expand to 5 seeds

## Experiment Log Location
`experiments/experiment_log.md` — single source of truth for all results

## Current Stage
[Update this manually: Stage 1 / Stage 2 / Stage 3]

## Key Thresholds
- lambda_eik default: 0.1
- Warmup: 100% → 100ep, 10% → 150ep, 5% → 200ep
- Gradient clip: max_norm=1.0
- Batch size: 16384 (reduce to 8192 for EXP-08 only)

## TC2 GPU Cluster
- **SSH**: `ssh yutaek001@10.96.189.12` (requires NTU VPN off-campus)
- **Home**: `/home/msai/yutaek001`
- **Conda**: `module load anaconda && conda activate yt3dl`
- **CUDA**: `module load cuda/12.8.0`
- **Partition**: `MGPU-TC2`, QoS: `normal`
- **Limits**: max 2 concurrent jobs, 1 GPU, 10 CPU, 30GB mem, 6hr wall time
- **Disk quota**: 100GB — run /disk-check regularly

## Compact Instructions
When compacting, always preserve:
- Current experiment ID and stage (EXP-XX, Stage N)
- Last experiment results (CD, NC values)
- Any pending action items from diagnosis
- The current improve→run→log cycle position
- File paths of recently modified source files
Discard: full log contents (readable from experiments/experiment_log.md), old diagnosis details, verbose SLURM output

## Skills Usage

| Situation | Skill to Run |
|-----------|-------------|
| Before starting experiment | `/run-experiment EXP-04 42` |
| When results are available | `/log-experiment EXP-04` → auto-records expected vs actual |
| When training looks abnormal | `/diagnose-training EXP-04 42` |
| Multiple experiments at once | `/submit-slurm EXP-03 EXP-04 seeds 42 123 456` |
| After experiments complete | `/disk-check` (can be auto-triggered) |
| Week 11 onwards | `/generate-figures label-efficiency` |
| Weeks 13-15 | `/write-report-section discussion` |
