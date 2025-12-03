"""
Sweep batch sizes for a SwigLU FourLayerNet at initialization and plot post-activation stable ranks.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
import torch
import torch.nn as nn
import torch.nn.functional as F

from weighted_spectral_vs_gd import ExperimentConfig, FourLayerNet, stable_rank_exact


@torch.no_grad()
def compute_layer_stable_ranks(
    model: FourLayerNet,
    inputs: torch.Tensor,
) -> Dict[str, float]:
    outputs, activations = model.forward_with_activations(inputs)
    ranks = {}
    for idx, act in enumerate(activations, start=1):
        sr, _ = stable_rank_exact(act)
        ranks[f"layer{idx}"] = sr
    return ranks


def sweep_batch_sizes(
    dim: int,
    batch_sizes: List[int],
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[int, Dict[str, float]]:
    torch.manual_seed(seed)
    config = ExperimentConfig(dim=dim, batch_size=batch_sizes[0], activation="swiglu", dtype=dtype)
    model = FourLayerNet(config, "swiglu").to(device=device, dtype=dtype)
    results: Dict[int, Dict[str, float]] = {}
    for batch in batch_sizes:
        inputs = torch.randn(batch, dim, device=device, dtype=dtype)
        ranks = compute_layer_stable_ranks(model, inputs)
        results[batch] = ranks
    return results


def plot_stable_ranks(results: Dict[int, Dict[str, float]], output: Path, dpi: int) -> None:
    if not results:
        raise ValueError("No results to plot")
    batch_sizes = sorted(results.keys())
    layers = sorted(next(iter(results.values())).keys())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for layer in layers:
        values = [results[bs][layer] for bs in batch_sizes]
        label = layer.replace("layer", "Layer ")
        ax.semilogx(batch_sizes, values, marker="o", label=label)

    ax.set_xlabel("Batch size", fontsize=16)
    ax.set_ylabel("Stable rank", fontsize=16)
    ax.set_title("SwigLU post-activation stable ranks at initialization", fontsize=18)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="best", fontsize=14)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SwigLU stable-rank sweep")
    parser.add_argument("--dim", type=int, default=128, help="Model dimension d")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="List of batch sizes to sweep",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("logs") / "swiglu_stable_ranks.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    results = sweep_batch_sizes(args.dim, args.batch_sizes, args.seed, dtype, device)
    plot_stable_ranks(results, args.output, args.dpi)


if __name__ == "__main__":
    main()


