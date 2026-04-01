#!/usr/bin/env bash
# env.sh — AutoRL-LRM environment setup
# ======================================
# Usage: source env.sh
#
# Sets all path variables for the AutoRL-LRM project.
# AUTORL_WORKSPACE is derived from this file's location — no hardcode.
# AUTORL_HOME is prompted if not already set in the shell.

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
# 3. Derived runtime paths
# ---------------------------------------------------------------------------

export AUTORL_DATA="$AUTORL_HOME/data"
export AUTORL_CHECKPOINTS="$AUTORL_HOME/checkpoints"
export AUTORL_LOG="$AUTORL_HOME/log"

# ---------------------------------------------------------------------------
# 4. Virtual environment
# ---------------------------------------------------------------------------

export UV_PROJECT_ENVIRONMENT="$AUTORL_HOME/venv/.venv"
export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
export PATH="$VIRTUAL_ENV/bin:$PATH"

if [[ -f "$VIRTUAL_ENV/bin/activate" ]]; then
    source "$VIRTUAL_ENV/bin/activate"
    # re-export after activate — activate may overwrite with stale original path
    export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
fi

# ---------------------------------------------------------------------------
# 5. CUDA — detect from nvidia-smi (driver-matched), with fallback chain
# ---------------------------------------------------------------------------

if [[ -z "${CUDA_HOME:-}" ]]; then
    # 1. nvidia-smi: driver-reported CUDA version → matching toolkit
    _cuda_ver="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[\d\.]+' | cut -d. -f1,2)"
    if [[ -n "$_cuda_ver" && -d "/usr/local/cuda-$_cuda_ver" ]]; then
        CUDA_HOME="/usr/local/cuda-$_cuda_ver"

    # 2. nvcc already in PATH
    elif command -v nvcc &>/dev/null; then
        CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"

    # 3. /usr/local/cuda symlink (distro default)
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
# 6. Summary — all resolved paths printed for human and agent verification
# ---------------------------------------------------------------------------

echo ""
echo "========================================================"
echo "  AutoRL-LRM Environment"
echo "========================================================"
echo "  AUTORL_WORKSPACE      = $AUTORL_WORKSPACE"
echo "  AUTORL_HOME           = $AUTORL_HOME"
echo "  AUTORL_DATA           = $AUTORL_DATA"
echo "  AUTORL_CHECKPOINTS    = $AUTORL_CHECKPOINTS"
echo "  AUTORL_LOG            = $AUTORL_LOG"
echo "  VIRTUAL_ENV           = $VIRTUAL_ENV"
echo "  UV_PROJECT_ENVIRONMENT= $UV_PROJECT_ENVIRONMENT"
echo "  CUDA_HOME             = ${CUDA_HOME:-${_cuda_warning}}"
echo "========================================================"

# ---------------------------------------------------------------------------
# 7. Verify — critical paths exist
# ---------------------------------------------------------------------------
# 
_ok=true
for _var in AUTORL_HOME AUTORL_DATA AUTORL_CHECKPOINTS AUTORL_LOG VIRTUAL_ENV; do
    _path="${!_var}"
    if [[ ! -d "$_path" ]]; then
        echo "  ⚠️  $_var does not exist: $_path"
        _ok=false
    fi
done
[[ "$_ok" == true ]] && echo "  ✅ All paths verified." || echo "  ℹ️  Run 'mkdir -p' for missing dirs."
echo ""
