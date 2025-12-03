#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN="$ENV_DIR/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[specgd] Missing virtual environment. Run $SCRIPT_DIR/install.sh first." >&2
  exit 1
fi

OUTPUT_DIR="${RF_RANK_OUTPUT_DIR:-$SCRIPT_DIR/logs/rf_rank_plots}"
mkdir -p "$OUTPUT_DIR"

if [ -n "${RF_RANK_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${RF_RANK_EXTRA_ARGS})
else
  EXTRA_ARGS=()
fi

echo "[specgd] Running rf_nuclear_rank_plots.py ..."
"$PYTHON_BIN" "$SCRIPT_DIR/rf_nuclear_rank_plots.py" \
  --dims 64 128 256 512 \
  --steps 1024 \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"

echo "[specgd] Nuclear-rank plots saved to $OUTPUT_DIR"

