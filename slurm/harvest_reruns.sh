#!/bin/bash
# Verify all 16 rerun result files exist on TC2 and bundle them for scp.
# Run on TC2 after all rerun jobs complete.
# Usage: bash slurm/harvest_reruns.sh
#
# Outputs: ~/3ddl/rerun_results.tar.gz

set -euo pipefail
cd "$HOME/3ddl"

MANIFEST=slurm/rerun_manifest.txt
BUNDLE=rerun_results.tar.gz
MISSING=0

echo "=== Verifying manifest targets ==="
while read -r checkpoint_mode expected_n_total result_path; do
    if [ -f "$result_path" ]; then
        echo "OK  $result_path"
    else
        echo "MISSING  $result_path"
        MISSING=$((MISSING + 1))
    fi
done < "$MANIFEST"

if [ "$MISSING" -gt 0 ]; then
    echo "ABORT: $MISSING files missing — do not harvest until all jobs complete"
    exit 1
fi

echo ""
echo "=== All 16 result files present. Building bundle ==="
# Collect all result paths from manifest
RESULT_FILES=$(awk '{print $3}' "$MANIFEST")
tar czf "$BUNDLE" $RESULT_FILES slurm/rerun_job_ids.txt
echo "Bundle written: $BUNDLE ($(du -h $BUNDLE | cut -f1))"
echo "Run locally: scp -i ~/.ssh/tc2_key yutaek001@10.96.189.12:/home/msai/yutaek001/3ddl/rerun_results.tar.gz ."
