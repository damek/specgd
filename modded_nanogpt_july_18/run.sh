#!/usr/bin/env bash
set -euo pipefail

# Activate project virtualenv if it exists
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# ensure user-level pip scripts (e.g., torchrun) are on PATH
PATH="$HOME/.local/bin:$PATH"

if command -v torchrun >/dev/null 2>&1; then
    exec torchrun --standalone --nproc_per_node=8 train_gpt_july18_tracked_stable_rank.py
elif [ -x "$HOME/.local/bin/torchrun" ]; then
    exec "$HOME/.local/bin/torchrun" --standalone --nproc_per_node=8 train_gpt_july18_tracked_stable_rank.py
else
    # fallback to module invocation if the console script isn't available on PATH
    exec python3 -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt_july18_tracked_stable_rank.py
fi
