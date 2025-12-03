#!/usr/bin/env bash
set -euo pipefail

# Always make sure user-level scripts (including uv) are visible
export PATH="$HOME/.local/bin:$PATH"

########################################
# 0. System deps for Triton / Inductor #
########################################

if command -v apt-get >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    echo "[install] Installing Python headers and build tools (requires sudo)..."
    sudo apt-get update
    # Try the exact version first, fall back to the generic one
    sudo apt-get install -y python3.10-dev build-essential \
      || sudo apt-get install -y python3-dev build-essential
  else
    echo "[warn] apt-get found but sudo is not available."
    echo "       Please ask an admin to install: python3-dev build-essential"
    # we continue anyway; flex_attention will fail to compile without Python.h
  fi
else
  echo "[warn] apt-get not found; skipping system package install."
  echo "       Make sure Python dev headers and a compiler toolchain are installed."
fi

########################################
# 1. Install uv if it isn't available  #
########################################

if ! command -v uv >/dev/null 2>&1; then
  echo "[install] uv not found, installing to \$HOME/.local/bin ..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "Error: need curl or wget to install uv." >&2
    exit 1
  fi
  # refresh shell's command cache so `uv` is seen
  hash -r
fi

########################################
# 2. Create / reuse project venv       #
########################################

uv venv

# Activate the virtualenv so python/torchrun use it
# shellcheck disable=SC1091
source .venv/bin/activate

########################################
# 3. Install Python dependencies       #
########################################

# Keep requirements.txt exactly as-is
uv pip install -r requirements.txt

# Replace whatever torch that pulled in with a stable CUDA 12.6 build
uv pip install torch --index-url https://download.pytorch.org/whl/cu126 --upgrade

########################################
# 4. Prepare cached FineWeb data       #
########################################

python data/cached_fineweb10B.py 8
