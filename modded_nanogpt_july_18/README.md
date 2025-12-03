# modded_nanogpt_july_18 quickstart

This folder is the July 18, 2025 snapshot of the speedrun experiment from [the modded-nanogpt repo](https://github.com/KellerJordan/modded-nanogpt). The only edits are logging hooks inside `train_gpt_july18_tracked_stable_rank.py`; the training path is otherwise identical to upstream.

1. Run the installer to set up the venv and deps:
   ```
   ./install.sh
   ```
2. Launch the training run:
   ```
   ./run.sh
   ```
3. After a run finishes, plot the stable-rank figures:
   ```
   python plot_paper_figures.py --input logs/<run_id>_stable_rank.json
   ```

## stable-rank logging cheatsheet

- tracker fires every `TRACK_METRIC_INTERVAL` (100 steps by default) and dumps activation stable ranks, RMSNorm stats, q/k/v column norms, gradients (mlp/attn/embed/value embeds), weight stable ranks, and token-frequency imbalance.
- outputs land in `logs/<run_id>_stable_rank.json` plus an auto-rendered `<run_id>_stable_rank.png`. the JSON is what `plot_paper_figures.py` expects.
- if you need fresh figures later, rerun `python plot_paper_figures.py --input logs/<run_id>_stable_rank.json` or point the script at any other stable-rank JSON dropped by the tracker.

