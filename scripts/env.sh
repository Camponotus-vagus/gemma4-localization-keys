# Source this file to keep every cache and config inside the project folder.
# Usage: source scripts/env.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/.." && pwd)"
export PROJECT_ROOT

export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/.cache/uv-python"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache/xdg"
export KAGGLE_CONFIG_DIR="$PROJECT_ROOT/.secrets/kaggle"
# Kaggle SDK accepts a file path here; the token value never enters the environment
export KAGGLE_API_TOKEN="$KAGGLE_CONFIG_DIR/access_token"
export DATA_DIR="$PROJECT_ROOT/data"

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME" "$PIP_CACHE_DIR" \
  "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$KAGGLE_CONFIG_DIR" "$DATA_DIR"
