#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN="$ENV_DIR/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[specgd] Missing virtual environment. Run $SCRIPT_DIR/install.sh first." >&2
  exit 1
fi

OUTPUT_DIR="$SCRIPT_DIR/logs_weighted"
mkdir -p "$OUTPUT_DIR"

if [ -n "${WEIGHTED_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${WEIGHTED_EXTRA_ARGS})
else
  EXTRA_ARGS=()
fi

plot_latest_weighted() {
  local latest_json
  latest_json=$(ls -t "$OUTPUT_DIR"/*.json 2>/dev/null | head -n 1 || true)
  if [ -z "$latest_json" ]; then
    echo "[specgd] No weighted_spectral_vs_gd JSON found to plot." >&2
    return
  fi
  echo "[specgd] Plotting weighted_spectral_vs_gd results from $latest_json"
  "$PYTHON_BIN" "$SCRIPT_DIR/plot_spectral_vs_gd.py" \
    --input "$latest_json" \
    --output-dir "$OUTPUT_DIR"
}

base_args=(
  --plot
  --progress
  --steps 400
  --output-dir "$OUTPUT_DIR"
)

echo "[specgd] Running weighted_spectral_vs_gd.py (ReLU baseline)..."
"$PYTHON_BIN" "$SCRIPT_DIR/weighted_spectral_vs_gd.py" \
  "${base_args[@]}" \
  "${EXTRA_ARGS[@]}" \
  --activation relu_square \
  --method-lr gradient_descent=.081 \
  --method-lr specgd=0.1
plot_latest_weighted

echo "[specgd] Running weighted_spectral_vs_gd.py (SwiGLU, batch 8196)..."
"$PYTHON_BIN" "$SCRIPT_DIR/weighted_spectral_vs_gd.py" \
  "${base_args[@]}" \
  "${EXTRA_ARGS[@]}" \
  --activation swiglu \
  --batch-size 8196 \
  --method-lr gradient_descent=.0035 \
  --method-lr specgd=0.0015
plot_latest_weighted

echo "[specgd] Running weighted_spectral_vs_gd.py (SwiGLU, batch 512)..."
"$PYTHON_BIN" "$SCRIPT_DIR/weighted_spectral_vs_gd.py" \
  "${base_args[@]}" \
  "${EXTRA_ARGS[@]}" \
  --activation swiglu \
  --batch-size 512 \
  --method-lr gradient_descent=.01 \
  --method-lr specgd=0.025
plot_latest_weighted

echo "[specgd] Weighted spectral vs GD runs complete. Outputs stored in $OUTPUT_DIR"

