#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN="$ENV_DIR/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[specgd] Missing virtual environment. Run $SCRIPT_DIR/install.sh first." >&2
  exit 1
fi

OUTPUT_DIR="$SCRIPT_DIR/logs_rf"
mkdir -p "$OUTPUT_DIR"

if [ -n "${RF_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${RF_EXTRA_ARGS})
else
  EXTRA_ARGS=()
fi

plot_latest_run() {
  local latest_json
  latest_json=$(ls -t "$OUTPUT_DIR"/*.json 2>/dev/null | head -n 1 || true)
  if [ -z "$latest_json" ]; then
    echo "[specgd] No JSON log found to plot in $OUTPUT_DIR" >&2
    return
  fi
  echo "[specgd] Plotting random-feature results from $latest_json"
  "$PYTHON_BIN" "$SCRIPT_DIR/plot_random_feature_specgd.py" \
    --input "$latest_json" \
    --output-dir "$OUTPUT_DIR"
}

common_args=(
  --plot
  --output-dir "$OUTPUT_DIR"
  --progress
  --specgd-restart-frequency 100
  --specgd-from-peak
  --num-samples 400
  --steps 1000
)

echo "[specgd] Running random_feature_specgd.py (ReLU)..."
rf_relu_cmd=(
  "$PYTHON_BIN" "$SCRIPT_DIR/random_feature_specgd.py"
  "${common_args[@]}"
)
if ((${#EXTRA_ARGS[@]})); then
  rf_relu_cmd+=("${EXTRA_ARGS[@]}")
fi
rf_relu_cmd+=(--activation relu)
"${rf_relu_cmd[@]}"
plot_latest_run

echo "[specgd] Running random_feature_specgd.py (SwiGLU, 400-step variant)..."
rf_swiglu_cmd=(
  "$PYTHON_BIN" "$SCRIPT_DIR/random_feature_specgd.py"
  --plot
  --output-dir "$OUTPUT_DIR"
  --progress
  --activation swiglu
  --specgd-restart-frequency 100
  --specgd-from-peak
  --num-samples 400
  --steps 400
)
if ((${#EXTRA_ARGS[@]})); then
  rf_swiglu_cmd+=("${EXTRA_ARGS[@]}")
fi
"${rf_swiglu_cmd[@]}"
plot_latest_run

echo "[specgd] Random-feature experiments complete. Logs saved under $OUTPUT_DIR"

