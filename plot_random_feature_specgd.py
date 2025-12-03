"""Plotting utilities for the random-feature GD vs SpecGD experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"


METHOD_STYLES = {
    "gradient_descent": {"label": "GD", "color": "#1f77b4", "linestyle": "-", "linewidth": 2.2},
    "specgd": {"label": "SpecGD", "color": "#ff7f0e", "linestyle": "-", "linewidth": 2.2},
    "specgd_from_peak": {"label": "SpecGD from peak", "color": "#8b0000", "linestyle": "-", "linewidth": 2.6},
}


def _plot_objective_gap(ax: plt.Axes, methods: Dict[str, Dict[str, Dict[str, List[float]]]], title_suffix: str) -> None:
    legend_handles = []
    legend_labels = []

    restart_legend_added = False

    for name, payload in methods.items():
        series = payload.get("objective_gap")
        if not series or not series["steps"]:
            continue
        color = METHOD_STYLES.get(name, {}).get("color", "#555555")
        linewidth = METHOD_STYLES.get(name, {}).get("linewidth", 1.8)
        if name.startswith("specgd_from_iter_"):
            linestyle = (0, (4, 2))
            if not restart_legend_added:
                label = r"SpecGD from iter $t$"
                restart_legend_added = True
            else:
                label = "_nolegend_"
        else:
            linestyle = METHOD_STYLES.get(name, {}).get("linestyle", "-")
            label = METHOD_STYLES.get(name, {}).get("label", name)
        handle = ax.semilogy(
            series["steps"],
            series["values"],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        if label != "_nolegend_":
            legend_handles.append(handle[0])
            legend_labels.append(label)

    ax.set_xlabel("Training iteration", fontsize=16)
    ax.set_ylabel("MSE", fontsize=16)
    ax.set_title("Training loss", fontsize=18)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="upper right", frameon=True, framealpha=0.9, fontsize=14)
    ax.tick_params(axis="both", labelsize=14)


def _plot_nuclear_rank(ax: plt.Axes, methods: Dict[str, Dict[str, Dict[str, List[float]]]], st_a: float | None) -> None:
    curve_handles = []
    curve_labels = []
    for name in ("gradient_descent", "specgd", "specgd_from_peak"):
        payload = methods.get(name)
        if payload is None:
            continue
        series = payload.get("nuclear_rank")
        if not series or not series["steps"]:
            continue
        style = METHOD_STYLES.get(name, {})
        line, = ax.plot(
            series["steps"],
            series["values"],
            label=style.get("label", name),
            color=style.get("color", "#1f77b4"),
            linestyle=style.get("linestyle", "-"),
            linewidth=style.get("linewidth", 2.2),
        )
        curve_handles.append(line)
        curve_labels.append(style.get("label", name))

    st_handle = None
    if st_a is not None:
        st_handle = ax.axhline(
            st_a,
            color="#333333",
            linestyle=(0, (4, 2)),
            linewidth=2.0,
            label=r"$\mathrm{st}(A)$",
        )

    ax.set_xlabel("Training iteration", fontsize=16)
    ax.set_ylabel(r"$\mathrm{nr}(\nabla L(W))$", fontsize=16)
    ax.set_title("Nuclear rank of gradient", fontsize=18)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.tick_params(axis="both", labelsize=14)
    legend_handles = curve_handles.copy()
    legend_labels = curve_labels.copy()
    if st_handle is not None:
        legend_handles.append(st_handle)
        legend_labels.append(r"$\mathrm{st}(A)$")
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="upper right", fontsize=14)


def plot_random_feature_summary(payload, base_path: Path, dpi: int) -> None:
    methods = payload.get("methods", {})
    dataset = payload.get("dataset", {})
    st_a = dataset.get("stable_rank_A")
    activation = payload.get("config", {}).get("activation", "unknown")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    _plot_objective_gap(axes[0], methods, activation)
    _plot_nuclear_rank(axes[1], methods, st_a if isinstance(st_a, (int, float)) else None)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    activation_label = activation
    if activation.lower() == "swiglu":
        activation_label = "SwiGLU"
    elif activation.lower() == "relu":
        activation_label = "ReLU"
    fig.text(0.5, 0.02, f"{activation_label} random-feature experiment", ha="center", fontsize=16)

    output_path = base_path.parent / f"{base_path.name}_{activation}_rf_summary.pdf"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    resolved = output_path.resolve()
    print(f"Saved {output_path}")
    print(f"Figure: file://{resolved}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot random-feature GD vs SpecGD results")
    parser.add_argument("--input", type=Path, required=True, help="JSON log produced by random_feature_specgd.py")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    output_dir = args.output_dir if args.output_dir is not None else args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / args.input.stem
    plot_random_feature_summary(payload, base_path, args.dpi)


if __name__ == "__main__":
    main()


