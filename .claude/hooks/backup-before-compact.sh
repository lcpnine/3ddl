#!/bin/bash
INPUT=$(cat)
TRIGGER=$(echo "$INPUT" | jq -r '.trigger // "manual"')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/Users/lcpnine/3ddl/experiments/.compact-backups"
mkdir -p "$BACKUP_DIR"

# Save the transcript path if available
TRANSCRIPT=$(echo "$INPUT" | jq -r '.transcript_path // empty')
if [ -n "$TRANSCRIPT" ] && [ -f "$TRANSCRIPT" ]; then
  cp "$TRANSCRIPT" "$BACKUP_DIR/transcript_${TIMESTAMP}.jsonl"
fi

# Keep only last 5 backups
ls -t "$BACKUP_DIR"/transcript_*.jsonl 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null

exit 0
