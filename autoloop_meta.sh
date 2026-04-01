#!/usr/bin/env bash
# autoloop_meta.sh — Literature scan + program_rl.md self-update
# ==============================================================
# Run this BETWEEN experiment batches (e.g. every 5 experiments,
# or via cron nightly). It:
#   1. Scans arxiv + Semantic Scholar for new RLVR papers
#   2. Uses ollama API to update program_rl.md with new experiments
#   3. Optionally patches train_trl.py with new reward function stubs
#
# Prerequisites:
#   source env.sh                     (sets AUTORL_HOME, VIRTUAL_ENV, etc.)
#   ollama serve running              (meta_update.py calls Ollama locally)
#   AUTORL_META_MODEL env var         (default: qwen3-coder-next)
#   OLLAMA_HOST env var               (default: http://localhost:11434)
#
# Usage:
#   bash autoloop_meta.sh              # dry run (no writes)
#   bash autoloop_meta.sh --apply      # Write program_rl.md updates
#   bash autoloop_meta.sh --apply --patch-train  # also patch train_trl.py
#   bash autoloop_meta.sh --since 14   # scan last 14 days

set -euo pipefail

# --------------------------
# Step 0: Determine SCRIPT_DIR
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------
# Step 1: Source env.sh if not already done
# --------------------------
if [[ -f "$SCRIPT_DIR/env.sh" ]]; then
    source "$SCRIPT_DIR/env.sh"
else
    echo "ERROR: env.sh not found in $SCRIPT_DIR"
    exit 1
fi

# --------------------------
# Step 2: Ensure log directory exists
# --------------------------
AUTORL_LOG="${AUTORL_LOG:-$AUTORL_HOME/log}"
mkdir -p "$AUTORL_LOG"

# --------------------------
# Step 3: Determine Python interpreter
# --------------------------
PYTHON="${VIRTUAL_ENV:-}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python3"
fi

# --------------------------
# Step 4: Parse command-line args
# --------------------------
SINCE=7
APPLY_FLAG=""
PATCH_FLAG=""

for arg in "$@"; do
    case $arg in
        --apply)       APPLY_FLAG="--apply" ;;
        --patch-train) PATCH_FLAG="--patch-train" ;;
        --since=*)     SINCE="${arg#*=}" ;;
        --since)       shift; SINCE="$1" ;;
    esac
done

# --------------------------
# Step 5: Validate environment
# --------------------------
if [[ -z "${AUTORL_HOME:-}" ]]; then
    echo "ERROR: AUTORL_HOME is not set. Did you run 'source env.sh'?"
    exit 1
fi

if [[ -z "${AUTORL_WORKSPACE:-}" ]]; then
    echo "ERROR: AUTORL_WORKSPACE is not set. Did you run 'source env.sh'?"
    exit 1
fi

# --------------------------
# Step 6: Start logging
# --------------------------
LOG_FILE="$AUTORL_LOG/autoloop_meta_$(date +%Y%m%d_%H%M%S).log"
# Use exec to redirect stdout to log, but let stderr go to both terminal and log

# Use exec to redirect stdout to log, but let stderr go to both terminal and log
exec > >(tee -a "$LOG_FILE") 2> >(tee -a "$LOG_FILE" >&2)

echo "========================================================"
echo "AutoRL-LRM Meta Loop — Started at $(date)"
echo "  AUTORL_HOME     : $AUTORL_HOME"
echo "  AUTORL_WORKSPACE: $AUTORL_WORKSPACE"
echo "  Python          : $PYTHON"
echo "  Scan since      : $SINCE days"
echo "  Apply           : ${APPLY_FLAG:-dry run}"
echo "  Patch train     : ${PATCH_FLAG:-no}"
echo "========================================================"

# --------------------------
# Step 7: Scan papers
# --------------------------
echo ""
echo "[1/2] Scanning literature ..."

# Capture the output of scan_papers.py
SCAN_OUTPUT=$("$PYTHON" "$AUTORL_WORKSPACE/scan_papers.py" --since "$SINCE" 2>&1)
SCAN_EXIT_CODE=$?

# Print output
echo "$SCAN_OUTPUT"

# Check for errors
if [ $SCAN_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ scan_papers.py failed (exit code $SCAN_EXIT_CODE). See above for error."
    exit 1
fi

# ✅ if new papers found, ⚠️ if no new relevant papers
if echo "$SCAN_OUTPUT" | grep -q "\[scan_papers\] 0 total → 0 above threshold"; then
    echo "⚠️ [scan_papers] No new relevant papers found."
else
    echo "✅ [scan_papers] New papers found!"
fi

# --------------------------
# Step 8: Run meta update
# --------------------------
echo ""
echo "[2/2] Running meta-update ..."

# Capture output
META_OUTPUT=$("$PYTHON" "$AUTORL_WORKSPACE/meta_update.py" 2>&1)
META_EXIT_CODE=$?

# Print output
echo "$META_OUTPUT"

# Check for errors
if [ $META_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ meta_update.py failed (exit code $META_EXIT_CODE). See above for error."
    exit 1
fi

# ✅ if new content added, ⚠️ if no new content
if echo "$META_OUTPUT" | grep -q "\[meta_update\] No new content to add"; then
    echo "⚠️ [meta_update] No new content to add to program_rl.md"
else
    echo "✅ [meta_update] New content added!"
fi

# --------------------------
# Step 9: Loop complete
# --------------------------
echo ""
echo "========================================================"
echo " ✅ Meta loop complete at $(date)"
if [[ -z "$APPLY_FLAG" ]]; then
    echo "  → Dry run. Re-run with --apply to commit changes."
fi
echo "========================================================"
