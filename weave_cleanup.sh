#!/bin/bash
# weave_cleanup.sh — LoomAI cleanup script.
#
# Honors KEEP_SLICE: with KEEP_SLICE=true (default) the slice is NOT deleted, so
# the demo node and its installed stack survive between sessions. Set
# KEEP_SLICE=false (or run ACTION=stop ./weave.sh) to actually delete it.
set -uo pipefail

SLICE_NAME="${SLICE_NAME:-${1:-cryoem-demo}}"
SLICE_NAME=$(echo "$SLICE_NAME" | sed 's/[^a-zA-Z0-9-]/-/g; s/--*/-/g; s/^-//; s/-$//')
KEEP_SLICE="${KEEP_SLICE:-true}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="python3"
[ -x "$SCRIPT_DIR/.venv/bin/python" ] && PY="$SCRIPT_DIR/.venv/bin/python"

if [ "$KEEP_SLICE" = "true" ]; then
    echo "### PROGRESS: KEEP_SLICE=true — keeping slice '$SLICE_NAME'."
    echo "### PROGRESS: To delete it: KEEP_SLICE=false ./weave_cleanup.sh $SLICE_NAME"
else
    cd "$SCRIPT_DIR" && "$PY" "$SCRIPT_DIR/provision.py" stop "$SLICE_NAME" 2>&1 || true
fi
echo "### PROGRESS: Cleanup complete."
