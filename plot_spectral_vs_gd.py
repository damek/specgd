"""Plot helper for spectral-vs-GD experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors
from matplotlib import ticker as mticker


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14


METHOD_STYLES = {
    "gradient_descent": {"label": "GD", "color": "#1f77b4", "linestyle": "-"},
    "spectral_mixed": {"label": "Mixed Spectral", "color": "#d62728", "linestyle": (0, (4, 2))},
    "specgd": {"label": "SpecGD", "color": "#d62728", "linestyle": (0, (4, 2))},
    "block_specgd": {"label": "SpecGD", "color": "#ff7f0e", "linestyle": (0, (2, 2))},
    "hybrid": {"label": "Hybrid", "color": "#ff7f0e", "linestyle": (0, (2, 2))},
    "preconditioned": {"label": "Preconditioned", "color": "#2ca02c", "linestyle": "-."},
    "spectral_descent": {"label": "Spectral", "color": "#9467bd", "linestyle": ":"},
    "preconditioned_sgd": {"label": "PrecondSGD", "color": "#17becf", "linestyle": "-."},
    "rkfac": {"label": "RKFAC", "color": "#8c564b", "linestyle": "-"},
}


def _add_top_legend(ax: plt.Axes, legend_ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    legend_ax.axis("off")
    legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=len(handles),
        fontsize="medium",
        frameon=False,
        handlelength=2.5,
    )


def _make_figure() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    fig = plt.figure(figsize=(6.0, 5.5))
    grid = fig.add_gridspec(2, 1, height_ratios=[0.35, 1.0], hspace=0.25)
    legend_ax = fig.add_subplot(grid[0])
    data_ax = fig.add_subplot(grid[1])
    return fig, data_ax, legend_ax


def _configure_ratio_axis(ax: plt.Axes, values: List[float], *, pad_frac: float = 0.1) -> None:
    if not values:
        return
    vmax = max(values)
    if vmax <= 0:
        vmax = 1.0
    upper = vmax * (1.0 + pad_frac)
    upper = max(upper, 1e-6)
    ax.set_ylim(0.0, upper)
    locator = mticker.MaxNLocator(min_n_ticks=6)
    ax.yaxis.set_major_locator(locator)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.3g}"))


def _plot_metric(
    methods: Dict[str, Dict[str, Dict[str, list]]],
    accessor,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int,
    *,
    logy: bool = False,
    ylim_bottom: float | None = None,
    ratio_axis: bool = False,
    separate_legend: bool = True,
    title_fontsize: int | None = None,
    label_fontsize: int | None = None,
) -> None:
    if separate_legend:
        fig, data_ax, legend_ax = _make_figure()
    else:
        fig, data_ax = plt.subplots(figsize=(7.5, 5.0))
        legend_ax = None
    all_values: List[float] = []
    for method_name, payload in methods.items():
        if method_name not in METHOD_STYLES:
            continue
        series = accessor(payload)
        if not series["steps"]:
            continue
        style = METHOD_STYLES[method_name]
        if not logy and not ratio_axis:
            all_values.extend(series["values"])
        elif ratio_axis:
            all_values.extend(series["values"])
        if logy:
            data_ax.semilogy(
                series["steps"],
                series["values"],
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.8,
            )
        else:
            data_ax.plot(
                series["steps"],
                series["values"],
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.8,
            )

    if label_fontsize is not None:
        data_ax.set_xlabel("Training iteration", fontsize=label_fontsize)
        data_ax.set_ylabel(ylabel, fontsize=label_fontsize)
    else:
        data_ax.set_xlabel("Training iteration")
        data_ax.set_ylabel(ylabel)
    if title_fontsize is not None:
        data_ax.set_title(title, fontsize=title_fontsize)
    else:
        data_ax.set_title(title)
    data_ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    if ratio_axis:
        _configure_ratio_axis(data_ax, all_values)
    elif not logy and ylim_bottom is not None:
        data_ax.set_ylim(bottom=ylim_bottom)
    if separate_legend:
        _add_top_legend(data_ax, legend_ax)
    else:
        data_ax.legend(loc="upper right", fontsize=(label_fontsize or 14), frameon=True, framealpha=0.9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_all(methods: Dict[str, Dict[str, Dict[str, list]]], base_path: Path, dpi: int) -> None:
    # Loss
    loss_path = base_path.parent / f"{base_path.name}_loss.pdf"
    _plot_metric(
        methods,
        accessor=lambda payload: payload["loss"],
        ylabel="MSE",
        title="Training loss",
        output_path=loss_path,
        dpi=dpi,
        logy=True,
        separate_legend=False,
        title_fontsize=24,
        label_fontsize=18,
    )

    # Combined activation plot (layers 1-3)
    activation_path = base_path.parent / f"{base_path.name}_activations_layers123.pdf"
    fig = plt.figure(figsize=(8.5, 5.5))
    grid = fig.add_gridspec(2, 1, height_ratios=[0.28, 1.0], hspace=0.3)
    legend_ax = fig.add_subplot(grid[0])
    data_ax = fig.add_subplot(grid[1])
    legend_ax.axis("off")

    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(20)
    color_map = {
        ("gradient_descent", "layer1"): cmap(0),   # tab20 color 0
        ("gradient_descent", "layer2"): cmap(2),   # tab20 color 2
        ("gradient_descent", "layer3"): cmap(4),   # tab20 color 4
        ("block_specgd", "layer1"): cmap(6),       # tab20 color 6
        ("block_specgd", "layer2"): cmap(8),       # tab20 color 8
        ("block_specgd", "layer3"): cmap(10),      # tab20 color 10
    }

    layer_styles = {
        "layer1": {"linestyle": "-", "label": "Layer 1"},
        "layer2": {"linestyle": (0, (5, 2)), "label": "Layer 2"},
        "layer3": {"linestyle": (0, (1, 1)), "label": "Layer 3"},
    }

    method_order = ["gradient_descent", "block_specgd"]
    
    # Collect handles per method
    gd_handles = []
    bsgd_handles = []
    
    for method_name in method_order:
        if method_name not in methods:
            continue
        payload = methods[method_name]
        style = METHOD_STYLES.get(method_name)
        if style is None:
            continue
        method_handles = []
        for layer, layer_info in layer_styles.items():
            series = payload["activations"].get(layer, {"steps": [], "values": []})
            if not series["steps"]:
                continue
            color = color_map.get((method_name, layer), "#000000")
            line = data_ax.plot(
                series["steps"],
                series["values"],
                color=color,
                linestyle=layer_info["linestyle"],
                linewidth=2.5,
                label=f"{style['label']} • {layer_info['label']}",
            )
            method_handles.append(line[0])
        if method_name == "gradient_descent":
            gd_handles = method_handles
        else:
            bsgd_handles = method_handles
    
    # Interleave for column-first layout: [GD-L1, BSGD-L1, GD-L2, BSGD-L2, GD-L3, BSGD-L3]
    all_handles = []
    for i in range(3):
        if i < len(gd_handles):
            all_handles.append(gd_handles[i])
        if i < len(bsgd_handles):
            all_handles.append(bsgd_handles[i])
    all_labels = [h.get_label() for h in all_handles]

    data_ax.set_xlabel("Training iteration")
    data_ax.set_ylabel("Stable rank")
    data_ax.set_title("Post-activation stable rank (layers 1–3)")
    data_ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    data_ax.set_ylim(bottom=0)

    legend_ax.legend(
        all_handles,
        all_labels,
        loc="center",
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.5,
        labelspacing=0.8,
    )

    fig.tight_layout()
    activation_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(activation_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {activation_path}")

    # Combined gradient-ratio plot (weights 1-3)
    grad_ratio_path = base_path.parent / f"{base_path.name}_grad_ratios_layers123.pdf"
    fig = plt.figure(figsize=(8.5, 5.5))
    grid = fig.add_gridspec(2, 1, height_ratios=[0.28, 1.0], hspace=0.3)
    legend_ax = fig.add_subplot(grid[0])
    data_ax = fig.add_subplot(grid[1])
    legend_ax.axis("off")

    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(20)
    grad_color_map = {
        ("gradient_descent", "W1"): cmap(12),
        ("gradient_descent", "W2"): cmap(14),
        ("gradient_descent", "W3"): cmap(16),
        ("block_specgd", "W1"): cmap(1),
        ("block_specgd", "W2"): cmap(3),
        ("block_specgd", "W3"): cmap(5),
    }

    weight_styles = {
        "W1": {"linestyle": "-", "label": "Layer 1"},
        "W2": {"linestyle": (0, (5, 2)), "label": "Layer 2"},
        "W3": {"linestyle": (0, (1, 1)), "label": "Layer 3"},
    }

    method_order = ["gradient_descent", "block_specgd"]

    gd_handles = []
    bsgd_handles = []

    ratio_values: List[float] = []

    for method_name in method_order:
        if method_name not in methods:
            continue
        payload = methods[method_name]
        style = METHOD_STYLES.get(method_name)
        if style is None:
            continue
        method_handles = []
        for weight_name, weight_info in weight_styles.items():
            series = payload.get("grad_ratio", {}).get(weight_name, {"steps": [], "values": []})
            if not series["steps"]:
                continue
            ratio_values.extend(series["values"])
            color = grad_color_map.get((method_name, weight_name), "#000000")
            line = data_ax.plot(
                series["steps"],
                series["values"],
                color=color,
                linestyle=weight_info["linestyle"],
                linewidth=2.5,
                label=f"{style['label']} • {weight_info['label']}",
            )
            method_handles.append(line[0])
        if method_name == "gradient_descent":
            gd_handles = method_handles
        else:
            bsgd_handles = method_handles

    all_handles = []
    for i in range(3):
        if i < len(gd_handles):
            all_handles.append(gd_handles[i])
        if i < len(bsgd_handles):
            all_handles.append(bsgd_handles[i])
    all_labels = [h.get_label() for h in all_handles]

    data_ax.set_xlabel("Training iteration")
    data_ax.set_ylabel("Nuclear rank")
    data_ax.set_title(r"$\mathrm{nr}(\nabla_{W_\ell} \mathcal{L}(W))$ for layers 1–3")
    data_ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    _configure_ratio_axis(data_ax, ratio_values)

    legend_ax.legend(
        all_handles,
        all_labels,
        loc="center",
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.5,
        labelspacing=0.8,
    )

    fig.tight_layout()
    grad_ratio_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(grad_ratio_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {grad_ratio_path}")

    # Side-by-side friendly gradient-ratio plot with legend on the right
    grad_ratio_side_path = base_path.parent / f"{base_path.name}_grad_ratios_layers123_side.pdf"
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ratio_values_side: List[float] = []
    handles_side: List[Line2D] = []
    labels_side: List[str] = []

    label_human = {
        "gradient_descent": "GD",
        "block_specgd": "SpecGD",
    }

    for method_name in method_order:
        if method_name not in methods:
            continue
        payload = methods[method_name]
        style = METHOD_STYLES.get(method_name)
        if style is None:
            continue
        method_label = label_human.get(method_name, style["label"])
        for weight_name, weight_info in weight_styles.items():
            series = payload.get("grad_ratio", {}).get(weight_name, {"steps": [], "values": []})
            if not series["steps"]:
                continue
            ratio_values_side.extend(series["values"])
            color = grad_color_map.get((method_name, weight_name), "#000000")
            line = ax.plot(
                series["steps"],
                series["values"],
                color=color,
                linestyle=weight_info["linestyle"],
                linewidth=2.5,
                label=f"{method_label} • {weight_info['label']}",
            )
            handles_side.append(line[0])
            labels_side.append(line[0].get_label())

    _configure_ratio_axis(ax, ratio_values_side)
    ax.set_xlabel("Training iteration", fontsize=18)
    ax.set_ylabel("Nuclear rank", fontsize=18)
    ax.set_title(r"$\mathrm{nr}(\nabla_{W_\ell} \mathcal{L}(W))$ for layers 1–3", fontsize=24)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)

    legend = ax.legend(
        handles_side,
        labels_side,
        loc="upper right",
        borderaxespad=0.8,
        frameon=True,
        framealpha=0.9,
        handlelength=2.8,
        labelspacing=0.5,
    )
    for text in legend.get_texts():
        text.set_fontsize(16)

    fig.tight_layout()
    grad_ratio_side_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(grad_ratio_side_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {grad_ratio_side_path}")

    # Gradient ratios per weight matrix (layers 1-3 only)
    for weight in ("W1", "W2", "W3"):
        if not any(weight in payload.get("grad_ratio", {}) for payload in methods.values()):
            continue
        grad_path = base_path.parent / f"{base_path.name}_grad_ratio_{weight}.pdf"
        subscript = weight[1:]
        ratio_expr = rf"$\|\nabla_{{W_{subscript}}} L\|_*^2 / \|\nabla_{{W_{subscript}}} L\|_F^2$"
        title = f"Nuclear rank {weight}: {ratio_expr}"
        _plot_metric(
            methods,
            accessor=lambda payload, weight=weight: payload.get("grad_ratio", {}).get(weight, {"steps": [], "values": []}),
            ylabel="Nuclear rank",
            title=title,
            output_path=grad_path,
            dpi=dpi,
            ratio_axis=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot spectral-vs-GD experiment metrics")
    parser.add_argument("--input", type=Path, required=True, help="Experiment JSON log")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    methods = payload.get("methods", {})
    if not methods:
        raise ValueError("No methods found in JSON file")

    output_dir = args.output_dir if args.output_dir is not None else args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / args.input.stem

    plot_all(methods, base_path, args.dpi)


if __name__ == "__main__":
    main()

