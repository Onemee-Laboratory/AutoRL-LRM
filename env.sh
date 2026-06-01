#!/usr/bin/env bash
# env.sh — AutoRL-LRM environment setup
# ======================================
# Usage: source env.sh          ← must be sourced, not executed
#
# Sets all path variables for the AutoRL-LRM project.
# AUTORL_WORKSPACE is derived from this file's location — no hardcode.
# AUTORL_HOME is prompted if not already set in the shell.

# ---------------------------------------------------------------------------
# 0. Guard: must be sourced, not executed directly
# ---------------------------------------------------------------------------

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "⚠️  WARNING: env.sh must be sourced, not executed directly."
    echo "   Run:  source env.sh"
    echo "   The venv activation and all exports will not persist otherwise."
    # Continue anyway — useful for CI / path-verification-only runs
fi

# ---------------------------------------------------------------------------
# 1. AUTORL_WORKSPACE — always the directory containing this file
# ---------------------------------------------------------------------------

export AUTORL_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# ---------------------------------------------------------------------------
# 2. AUTORL_HOME — runtime dir (data, checkpoints, logs, venv)
#    Prompted if not already exported in the shell.
# ---------------------------------------------------------------------------

if [[ -z "${AUTORL_HOME:-}" ]]; then
    read -rp "Enter AUTORL_HOME (runtime dir for data/checkpoints/logs/venv): " AUTORL_HOME
    if [[ -z "$AUTORL_HOME" ]]; then
        echo "ERROR: AUTORL_HOME cannot be empty."
        return 1 2>/dev/null || exit 1
    fi
fi
export AUTORL_HOME

# ---------------------------------------------------------------------------
# 3. Derived runtime paths and files
# ---------------------------------------------------------------------------

export AUTORL_DATA="$AUTORL_HOME/data"
export AUTORL_CHECKPOINTS="$AUTORL_HOME/checkpoints"
export AUTORL_LOG="$AUTORL_HOME/log"
export AUTORL_WATCHLIST="$AUTORL_HOME/watchlist"
export AUTORL_PAPERS_DB="$AUTORL_WATCHLIST/papers_db.json"
export AUTORL_BASELINES="$AUTORL_HOME/baselines"
export AUTORL_WATCHLIST_NEW_PAPERS="new_papers.md"
export AUTORL_WATCHLIST_CLASSIC_PAPERS="classic_papers.md"
export AUTORL_WATCHLIST_OLD_PAPERS="old_papers.md"
export AUTORL_WATCHLIST_HIGH_INDEX_PAPERS="high_index_papers.md"
export AUTORL_WATCHLIST_FILE_DEFAULT="misc_papers.md"
export AUTORL_WATCHLIST_YEAR_LIMIT=2025
export AUTORL_WATCHLIST_CITE_LIMIT=100
export AUTORL_PROGRAM_RL="$AUTORL_WORKSPACE/program_rl.md"
export AUTORL_THEORY="$AUTORL_WORKSPACE/theory.md"
export AUTORL_TRAIN_TRL="$AUTORL_WORKSPACE/train_trl.py"
export AUTORL_BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"   # training model
export AUTORL_META_MODEL="/home/oz/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/main"
export VLLM_PYTHON="$AUTORL_HOME/venv/vllm-env/bin/python"
export VLLM_HOST="http://localhost:8000"
export OLLAMA_HOST="http://localhost:11434"

# ---------------------------------------------------------------------------
# 3a. Creātiō dīrectōriōrum (Runtime)
# ---------------------------------------------------------------------------

for _dir in "$AUTORL_DATA" "$AUTORL_CHECKPOINTS" "$AUTORL_LOG" \
            "$AUTORL_WATCHLIST" "$AUTORL_BASELINES"; do
    [[ -z "$_dir" ]] && { echo "  ⚠️  Skipping empty dir variable"; continue; }
    if [[ ! -d "$_dir" ]]; then
        mkdir -p "$_dir" \
            && echo "  📁 Created: $_dir" \
            || echo "  ❌ FAILED to create: $_dir"
    fi
done

# ---------------------------------------------------------------------------
# 3b. Creātiō fichāriōrum watchlist
# ---------------------------------------------------------------------------

for _file in "$AUTORL_WATCHLIST_NEW_PAPERS" \
             "$AUTORL_WATCHLIST_CLASSIC_PAPERS" \
             "$AUTORL_WATCHLIST_OLD_PAPERS" \
             "$AUTORL_WATCHLIST_HIGH_INDEX_PAPERS" \
             "$AUTORL_WATCHLIST_FILE_DEFAULT"; do
    [[ -z "$_file" ]] && { echo "  ⚠️  Skipping empty file variable"; continue; }
    _full_path="$AUTORL_WATCHLIST/$_file"
    if [[ ! -f "$_full_path" ]]; then
        touch "$_full_path" \
            && echo "  📄 Created: $_full_path" \
            || echo "  ❌ FAILED to touch: $_full_path"
    fi
done

# ---------------------------------------------------------------------------
# 3c. Creātiō Database (JSON)
# ---------------------------------------------------------------------------

if [[ ! -f "$AUTORL_PAPERS_DB" ]]; then
    echo "{}" > "$AUTORL_PAPERS_DB" \
        && echo "  🗄️  Created Database: $AUTORL_PAPERS_DB" \
        || echo "  ❌ FAILED to create: $AUTORL_PAPERS_DB"
fi

# ---------------------------------------------------------------------------
# 4. Virtual environment
# ---------------------------------------------------------------------------

export UV_PROJECT_ENVIRONMENT="$AUTORL_HOME/venv/.venv"
export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
export PATH="$VIRTUAL_ENV/bin:$PATH"

if [[ -f "$VIRTUAL_ENV/bin/activate" ]]; then
    source "$VIRTUAL_ENV/bin/activate"
    export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"   # re-pin after activate
fi

# ---------------------------------------------------------------------------
# 5. CUDA — detect from nvidia-smi (driver-matched), with fallback chain
# ---------------------------------------------------------------------------

if [[ -z "${CUDA_HOME:-}" ]]; then
    _cuda_ver="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[\d\.]+' | cut -d. -f1,2)"
    if [[ -n "$_cuda_ver" && -d "/usr/local/cuda-$_cuda_ver" ]]; then
        CUDA_HOME="/usr/local/cuda-$_cuda_ver"
    elif command -v nvcc &>/dev/null; then
        CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
    elif [[ -d /usr/local/cuda ]]; then
        CUDA_HOME="$(realpath /usr/local/cuda)"
    fi
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
    _cuda_warning="WARNING: could not detect — set manually or install cuda-toolkit"
else
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    _cuda_warning=""
fi

# ---------------------------------------------------------------------------
# 6. Verify — critical paths and files exist
# ---------------------------------------------------------------------------

echo ""
echo "========================================================================"
echo "              AutoRL-LRM Verify critical paths and files"
echo "========================================================================"
echo ""
_ok=true
for _var in AUTORL_HOME AUTORL_DATA AUTORL_WATCHLIST AUTORL_BASELINES \
            AUTORL_CHECKPOINTS AUTORL_LOG VIRTUAL_ENV; do
    _path="${!_var}"
    if [[ -z "$_path" ]]; then
        echo "  ⚠️  $_var is unset"
        _ok=false
    elif [[ ! -d "$_path" ]]; then
        echo "  ⚠️  $_var does not exist: $_path"
        _ok=false
    fi
done

if [[ ! -f "$AUTORL_PAPERS_DB" ]]; then
    echo "  ⚠️  AUTORL_PAPERS_DB does not exist: $AUTORL_PAPERS_DB"
    _ok=false
fi

# Verify watchlist .md files
for _file in "$AUTORL_WATCHLIST_NEW_PAPERS" \
             "$AUTORL_WATCHLIST_CLASSIC_PAPERS" \
             "$AUTORL_WATCHLIST_OLD_PAPERS" \
             "$AUTORL_WATCHLIST_HIGH_INDEX_PAPERS" \
             "$AUTORL_WATCHLIST_FILE_DEFAULT"; do
    [[ -z "$_file" ]] && continue
    if [[ ! -f "$AUTORL_WATCHLIST/$_file" ]]; then
        echo "  ⚠️  Watchlist file missing: $AUTORL_WATCHLIST/$_file"
        _ok=false
    fi
done

echo ""
[[ "$_ok" == true ]] \
    && echo "  ✅ All paths verified." \
    || echo "  ❌ Some paths are missing — see warnings above."
echo ""

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------

echo ""
echo "========================================================================"
echo "                              AutoRL-LRM Environment"
echo "========================================================================"
printf "  %-22s= %s\n" "AUTORL_WORKSPACE"                   "$AUTORL_WORKSPACE"
printf "  %-22s= %s\n" "AUTORL_HOME"                        "$AUTORL_HOME"
printf "  %-22s= %s\n" "AUTORL_DATA"                        "$AUTORL_DATA"
printf "  %-22s= %s\n" "AUTORL_WATCHLIST"                   "$AUTORL_WATCHLIST"
printf "  %-22s= %s\n" "AUTORL_PAPERS_DB"                   "$AUTORL_PAPERS_DB"
printf "  %-22s= %s\n" "AUTORL_BASELINES"                   "$AUTORL_BASELINES"
printf "  %-22s= %s\n" "AUTORL_CHECKPOINTS"                 "$AUTORL_CHECKPOINTS"
printf "  %-22s= %s\n" "AUTORL_LOG"                         "$AUTORL_LOG"
printf "  %-22s= %s\n" "AUTORL_PROGRAM_RL"                  "$AUTORL_PROGRAM_RL"
printf "  %-22s= %s\n" "AUTORL_THEORY"                      "$AUTORL_THEORY"
printf "  %-22s= %s\n" "AUTORL_TRAIN_TRL"                   "$AUTORL_TRAIN_TRL"
printf "  %-22s= %s\n" "AUTORL_BASE_MODEL"                  "$AUTORL_BASE_MODEL"
printf "  %-22s= %s\n" "AUTORL_META_MODEL"                  "$AUTORL_META_MODEL"
printf "  %-22s= %s\n" "VLLM_PYTHON"                        "$VLLM_PYTHON"
printf "  %-22s= %s\n" "VLLM_HOST"                          "$VLLM_HOST"
printf "  %-22s= %s\n" "OLLAMA_HOST"                        "$OLLAMA_HOST"
printf "  %-22s= %s\n" "VIRTUAL_ENV"                        "$VIRTUAL_ENV"
printf "  %-22s= %s\n" "UV_PROJECT_ENVIRONMENT"             "$UV_PROJECT_ENVIRONMENT"
printf "  %-22s= %s\n" "CUDA_HOME"                          "${CUDA_HOME:-${_cuda_warning}}"
printf "  %-36s= %s\n" "AUTORL_WATCHLIST_NEW_PAPERS"        "$AUTORL_WATCHLIST_NEW_PAPERS"
printf "  %-36s= %s\n" "AUTORL_WATCHLIST_CLASSIC_PAPERS"    "$AUTORL_WATCHLIST_CLASSIC_PAPERS"
printf "  %-36s= %s\n" "AUTORL_WATCHLIST_OLD_PAPERS"        "$AUTORL_WATCHLIST_OLD_PAPERS"
printf "  %-36s= %s\n" "AUTORL_WATCHLIST_HIGH_INDEX_PAPERS" "$AUTORL_WATCHLIST_HIGH_INDEX_PAPERS"
printf "  %-36s= %s\n" "AUTORL_WATCHLIST_YEAR_LIMIT"        "$AUTORL_WATCHLIST_YEAR_LIMIT"
printf "  %-36s= %s\n" "AUTORL_WATCHLIST_CITE_LIMIT"        "$AUTORL_WATCHLIST_CITE_LIMIT"
printf "  %-36s= %s\n" "AUTORL_WATCHLIST_FILE_DEFAULT"      "$AUTORL_WATCHLIST_FILE_DEFAULT"

echo "========================================================================"
