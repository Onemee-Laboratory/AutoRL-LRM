#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# AutoRL-LRM "The Great Loop" — Full Research Pipeline
# ---------------------------------------------------------------------------
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
#   bash autoloop_meta.sh --apply --patch-train  # Also patch train_trl.py
#   bash autoloop_meta.sh --since 14   # scan last 14 days

set -euo pipefail

# --------------------------
# Step 0: Define default variable
# --------------------------
SINCE="${SINCE:-7}"
OLD_DAYS="${OLD_DAYS:-365}"
SHORT_SLEEP_HOURS="${SHORT_SLEEP_HOURS:-1}"
LONG_SLEEP_HOURS="${LONG_SLEEP_HOURS:-6}"
MAX_LOGS="${MAX_LOGS:-30}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-50}"

GREEN="\033[1;32m"
YELLOW="\033[1;33m"
BLUE="\033[1;34m"
CYAN="\033[1;36m"
MAGENTA="\033[1;35m"
RESET="\033[0m"

# --------------------------
# Step 1: Determine SCRIPT_DIR
# --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------
# Step 2: Source env.sh if not already done
# --------------------------
if [[ -f "$SCRIPT_DIR/env.sh" ]]; then
    source "$SCRIPT_DIR/env.sh"
else
    echo "ERROR: env.sh not found in $SCRIPT_DIR"
    exit 1
fi

# --------------------------
# Step 3: Ensure log directory exists
# --------------------------
AUTORL_LOG="${AUTORL_LOG:-$AUTORL_HOME/log}"
AUTORL_DATA="${AUTORL_DATA:-$AUTORL_HOME/data}"
WATCHLIST_DIR="${AUTORL_WATCHLIST:-$AUTORL_HOME/watchlist}"

mkdir -p "$AUTORL_LOG"
mkdir -p "$AUTORL_DATA"
mkdir -p "$WATCHLIST_DIR"

# --------------------------
# Step 4: Determine Python interpreter
# --------------------------
PYTHON="${VIRTUAL_ENV:-}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python3"
fi

# --------------------------
# Step 5: Parse command-line args
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
# Step 6: Validate environment
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
# Step 7: Start logging
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

# ---------------------------------------------------------------------------
# Rotate old logs
# ---------------------------------------------------------------------------
rotate_logs() {
    local logs=("$AUTORL_LOG"/autoloop_meta_*.log)
    if (( ${#logs[@]} > MAX_LOGS )); then
        ls -1t "$AUTORL_LOG"/autoloop_meta_*.log | tail -n +$((MAX_LOGS + 1)) | xargs -r rm -f
    fi
}

# --------------------------
# Step 8: Scan papers
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

# ---------------------------------------------------------------------------
# Step 8: Scan paper and save output
# ---------------------------------------------------------------------------
run_scan() {
    local TIER="$1" SINCE_DAYS="$2" OUTPUT_FILE="$3" SYMBOL="$4"
    local LOG_FILE="$AUTORL_LOG/${TIER}_$(date +%Y%m%d_%H%M%S).log"
    echo "[meta_loop] Running $TIER scan — $(date)" | tee -a "$LOG_FILE"

    echo ""
    echo "[1/2] Scanning literature ..."

    # Capture the output of scan_papers.py
    local SCAN_OUTPUT
    SCAN_OUTPUT=$("$PYTHON" "$AUTORL_WORKSPACE/scan_papers.py" --since "$SINCE" 2>&1 | tee -a "$LOG_FILE")
    #local OUTPUT=$(cat "$LOG_FILE")
    SCAN_EXIT_CODE=$?

    #Print output
    echo "$SCAN_OUTPUT"

    # Check for errors
    if [ $SCAN_EXIT_CODE -ne 0 ]; then
	echo ""
        echo "❌ scan_papers.py failed (exit code $SCAN_EXIT_CODE). See above for error."
        exit 1
    fi

    # ✅ if new papers found, ⚠️ if no new relevant papers
    if echo "$SCAN_OUTPUT" | grep -q "\[scan_papers\] 0 total → 0 above threshold"; then
        echo "⚠️ [scan_papers]  ${YELLOW}⚠️ No new papers for $TIER${RESET}" | tee -a "$LOG_FILE"
        return 1
    else
	echo "$SCAN_OUTPUT" > "$OUTPUT_FILE"
        echo -e "[scan_papers] ${SYMBOL} $TIER papers updated" | tee -a "$LOG_FILE"
	echo "✅ [scan_papers] New papers found!"
        return 0
    fi
}

# --------------------------
# Step 9: Run meta update
# --------------------------
run_meta_update() {
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
}

# --------------------------
# Step 9: Loop complete
# --------------------------
#echo ""
#echo "========================================================"
#echo " ✅ Meta loop complete at $(date)"
#if [[ -z "$APPLY_FLAG" ]]; then
#    echo "  → Dry run. Re-run with --apply to commit changes."
#fi
#echo "========================================================"


# ---------------------------------------------------------------------------
# Selective Git commit/push
# ---------------------------------------------------------------------------
git_commit_push() {
    cd "$AUTORL_WORKSPACE" || return
    FILES=("scan_papers.py" "meta_update.py" "autoloop_meta.sh" "train_trl.py" "results_rl.tsv")
    local CHANGED=0
    for f in "${FILES[@]}"; do
        [[ -f "$f" && -n $(git status --porcelain "$f") ]] && git add "$f" && CHANGED=1
    done

    if (( CHANGED == 1 )); then
        git commit -m "AutoRL update: papers refreshed — $(date '+%Y-%m-%d %H:%M:%S')"
        echo "[meta_loop] Git commit done locally. Push manually from MacOS."
    else
        echo "[meta_loop] No changes to commit"
    fi
}

# ---------------------------------------------------------------------------
# Ensure watchlist file exists
# ---------------------------------------------------------------------------
ensure_watchlist_file() {
    local FILENAMES="$1"
    local FULL_PATH="$WATCHLIST_DIR/$FILENAMES"
    if [[ ! -f "$FULL_PATH" ]]; then
        echo -e "[meta_loop] ${CYAN}🆕 Creating new watchlist file: $FILENAMES${RESET}"
        touch "$FULL_PATH"
    fi
}

# ---------------------------------------------------------------------------
# Run

# ---------------------------------------------------------------------------
# Step 9: Forever loop
# ---------------------------------------------------------------------------
trap "echo '[meta_loop] Stopped by user'; exit 0" SIGINT SIGTERM

while true; do
    rotate_logs
    echo "========================================================"
    echo "[meta_loop] Iteration — $(date)"

    # Always ensure core watchlist files
    for file in news_papers.md classic_papers.md high_index_papers.md old_papers.md; do
	ensure_watchlist_file "$file"
    done

    if run_scan "new" "$SINCE" "$WATCHLIST_DIR/news_papers.md" "$GREEN✅"; then
        run_scan "classic" "$OLD_DAYS" "$WATCHLIST_DIR/classic_papers.md" "$CYAN📚"
        run_scan "high-index" "$OLD_DAYS" "$WATCHLIST_DIR/high_index_papers.md" "$BLUE🔹"
        run_scan "old" "$OLD_DAYS" "$WATCHLIST_DIR/old_papers.md" "$YELLOW⚠️"

        # Example: dynamic watchlist by famous scientist
        FAMOUS_SCIENTISTS=("Geoffrey_Hinton" "Yoshua_Bengio" "Demis_Hassabis")
        for SCIENTIST in "${FAMOUS_SCIENTISTS[@]}"; do
            ensure_watchlist_file "${SCIENTIST}.md"
            run_scan "scientist_$SCIENTIST" "$OLD_DAYS" "$WATCHLIST_DIR/${SCIENTIST}.md" "$MAGENTA📌"
        done

        #update_watchlist
        SLEEP_HOURS="$SHORT_SLEEP_HOURS"
    else
        SLEEP_HOURS="$LONG_SLEEP_HOURS"
    fi

    run_meta_update
    git_commit_push

    echo " ✅ [meta_loop] Next run in ${SLEEP_HOURS}h — $(date)"
    echo "========================================================"
    sleep $((SLEEP_HOURS*3600))
done
