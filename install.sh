#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$SCRIPT_DIR/.venv"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

# Ensure uv is available.
if ! command -v uv >/dev/null 2>&1; then
  echo "[specgd] uv not found; installing via official installer..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

UV_BIN="${UV_BIN:-$(command -v uv)}"

if [ ! -x "$UV_BIN" ]; then
  echo "[specgd] Unable to locate uv even after installation attempt." >&2
  exit 1
fi

echo "[specgd] Creating virtual environment at $ENV_DIR"
"$UV_BIN" venv "$ENV_DIR"

echo "[specgd] Installing Python dependencies from $REQUIREMENTS_FILE"
source "$ENV_DIR/bin/activate"
"$UV_BIN" pip install -r "$REQUIREMENTS_FILE"

echo "[specgd] Environment ready. Activate it with: source \"$ENV_DIR/bin/activate\""

