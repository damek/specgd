#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN="$ENV_DIR/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[specgd] Missing virtual environment. Run $SCRIPT_DIR/install.sh first." >&2
  exit 1
fi

OUTPUT_PATH="$SCRIPT_DIR/logs/swiglu_stable_ranks.png"
mkdir -p "$(dirname "$OUTPUT_PATH")"

"$PYTHON_BIN" "$SCRIPT_DIR/swiglu_stable_rank_sweep.py" \
  --output "$OUTPUT_PATH" \
  --device cpu

echo "[specgd] SwiGLU stable-rank sweep saved to $OUTPUT_PATH"

