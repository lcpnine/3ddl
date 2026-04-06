# Disk Usage Log

| Date | Total | Experiments | Data | Conda | Action |
|------|-------|-------------|------|-------|--------|
| 2026-03-28 | ~51 GB | 3.8 GB | 31 GB | 16 GB | No cleanup needed. 5 experiments (EXP-01 to EXP-05), 30MB checkpoints each. |
| 2026-03-30 | ~52 GB | 4.5 GB | 31 GB | 16 GB | Deleted 49GB EXP-06 reconstructions (PE meshes ~250MB/shape). Added mesh decimation to evaluate.py (200K face cap). EXP-07 training in progress. |
| 2026-04-06 | partial check | 7.4 GB | confirmed at least 5.4 GB (`processed_shapenet` 5.0 GB, `raw_shapenet` 420 MB, dev dirs negligible; `shapenet_raw` still present and is the likely remaining bulk) | 16 GB | Disk check run after EXP-08 and multi-seed/L=4 sweep. No urgent cleanup required to continue documentation/figure work. If space is needed later, Step 6.3 should target `data/shapenet_raw/` first. |
