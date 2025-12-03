#!/usr/bin/env python3
"""Generate publication-quality plots from stable-rank metrics for LaTeX documents.

Usage:
    python plot_paper_figures.py --input dataV1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# Set Times New Roman font for LaTeX compatibility
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # STIX fonts match Times


def _format_title(base: str, param_tag: str) -> tuple[str, str]:
    tagline = "Modded NanoGPT"
    if param_tag:
        tagline = f"{tagline} {param_tag}"
    return tagline, base


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stable-rank metrics for publication")
    parser.add_argument("--input", type=Path, required=True, help="Path to metrics JSON file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to same dir as input)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output resolution")
    parser.add_argument("--param-tag", type=str, default="", help="Parameter tag to append to titles, e.g. 124M or 1B")
    return parser.parse_args()


def _parse_block_index(key: str) -> int | None:
    if key.startswith("block"):
        rest = key[5:]
        try:
            return int(rest.split(".", 1)[0])
        except ValueError:
            return None
    head = key.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def _activation_component(key: str) -> str | None:
    if key.endswith("c_fc"):
        return "mlp.layer1"
    if key.endswith("c_proj"):
        return "mlp.layer2"
    return None


def _qkv_column_component(key: str) -> str | None:
    parts = key.split(".")
    if len(parts) >= 3 and parts[2] in {"q", "k", "v"}:
        return parts[2]
    return None


def _embed_logits_component(key: str) -> str | None:
    if key == "embedding":
        return "embedding"
    if key == "token_inv_pmax":
        return "token_inv_pmax"
    if key == "logits":
        return "logits"
    if key.startswith("value_embed"):
        return key.split(".")[0]
    return None


def _embed_gradient_component(key: str) -> str | None:
    if key == "embedding":
        return "embedding"
    if key == "lm_head":
        return "lm_head"
    if key.startswith("value_embed"):
        return key.split(".")[0]
    return None


def _rms_component(key: str) -> str | None:
    if key.endswith("pre_attn"):
        return "pre_attn"
    if key.endswith("pre_mlp"):
        return "pre_mlp"
    if key == "lm_head.pre":
        return "lm_head_pre"
    return None


def _compute_x_limits(history: Dict[str, List[tuple[int, float]]]) -> tuple[int, int] | None:
    min_step: int | None = None
    max_step: int | None = None
    for series in history.values():
        if not series:
            continue
        steps = [s for s, _ in series]
        local_min = min(steps)
        local_max = max(steps)
        min_step = local_min if min_step is None else min(min_step, local_min)
        max_step = local_max if max_step is None else max(max_step, local_max)
    if min_step is None or max_step is None:
        return None
    if max_step == min_step:
        max_step += 1
    return min_step, max_step


def _apply_shared_x_limits(axes: List[plt.Axes], limits: List[tuple[int, int] | None]) -> None:
    valid = [lim for lim in limits if lim is not None]
    if not valid:
        return
    global_min = min(lim[0] for lim in valid)
    global_max = max(lim[1] for lim in valid)
    margin = max(1, int(0.01 * (global_max - global_min)))
    for ax in axes:
        ax.set_xlim(global_min - margin, global_max + margin)


def _gradient_component(component_group: str, key: str) -> str | None:
    if component_group == "mlp":
        if key.endswith("c_fc.weight") or key.endswith("c_fc"):
            return "mlp.layer1"
        if key.endswith("c_proj.weight") or key.endswith("c_proj"):
            return "mlp.layer2"
    elif component_group == "attention":
        if key.endswith("attn.c_proj.weight") or key.endswith(".attn.c_proj"):
            return "attn.c_proj"
        for suffix in (".attn.q", ".attn.k", ".attn.v"):
            if key.endswith(suffix):
                return f"attn.{suffix[-1]}"
    return None


def _serialize_history(section: Dict[str, Dict[str, List[float]]]) -> Dict[str, List[tuple[int, float]]]:
    out: Dict[str, List[tuple[int, float]]] = {}
    for key, payload in section.items():
        out[key] = list(zip(payload["steps"], payload["values"]))
    return out


def _gather_block_colors(histories: List[Dict[str, Any]]) -> Dict[int, Any]:
    block_ids: set[int] = set()
    for history in histories:
        for key in history.keys():
            idx = _parse_block_index(key)
            if idx is not None:
                block_ids.add(idx)
    block_ids = sorted(block_ids)
    cmap = plt.cm.get_cmap("tab20", max(len(block_ids), 1))
    return {idx: cmap(i) for i, idx in enumerate(block_ids)}


def _render_panel(
    ax: plt.Axes,
    legend_ax: plt.Axes,
    title: tuple[str, str] | str,
    history: Dict[str, List[tuple[int, float]]],
    component_fn,
    style_lookup: Dict[str, tuple[Any, ...]],
    block_colors: Dict[int, Any],
) -> str:
    default_color = plt.cm.get_cmap("tab20")(0)
    block_usage: Dict[int, Any] = {}
    component_usage: set[str] = set()
    component_styles: Dict[str, tuple[Any, str, Any | None]] = {}

    for key in sorted(history.keys()):
        data = history[key]
        if not data:
            continue
        steps = [s for s, _ in data]
        values = [v for _, v in data]
        block_idx = _parse_block_index(key)
        color = block_colors.get(block_idx, default_color)
        component = component_fn(key)
        linestyle, label, color_override = "-", component or key, None
        if component in style_lookup:
            style_info = style_lookup[component]
            if len(style_info) >= 3:
                linestyle, label, color_override = style_info[0], style_info[1], style_info[2]
            else:
                linestyle, label = style_info[0], style_info[1]
                color_override = None
            component_usage.add(component)
            component_styles[component] = (linestyle, label, color_override)
        plot_color = color_override if color_override is not None else color
        ax.plot(steps, values, color=plot_color, linestyle=linestyle, linewidth=1.6)
        if block_idx is not None:
            block_usage.setdefault(block_idx, color)

    tagline = ""
    base_title = title
    if isinstance(title, tuple):
        tagline, base_title = title
    elif isinstance(title, str) and "\n" in title:
        tagline, base_title = [part.strip() for part in title.split("\n", 1)]
    ax.set_title(base_title, fontsize=14)
    ax.set_ylabel("Stable rank", fontsize=12)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.tick_params(labelsize=11)

    # Block colors legend
    color_handles = [
        Line2D([0], [0], color=color, linestyle="-", linewidth=2.5, label=f"block{idx}")
        for idx, color in sorted(block_usage.items())
    ]
    color_legend = None
    color_rows = 1
    if color_handles:
        color_ncol = min(len(color_handles), 6)
        color_rows = max(1, math.ceil(len(color_handles) / color_ncol))
        color_anchor_y = 1.0 + 0.18 * (color_rows - 1)
        color_legend = legend_ax.legend(
            color_handles,
            [h.get_label() for h in color_handles],
            loc="upper center",
            ncol=color_ncol,
            fontsize=10,
            frameon=False,
            bbox_to_anchor=(0.5, color_anchor_y),
            handlelength=2.2,
            handletextpad=0.8,
        )
    
    # Component style legend
    ordered_components = [c for c in style_lookup if c in component_usage]
    if not ordered_components:
        ordered_components = list(style_lookup.keys())
    style_handles: List[Line2D] = []
    for c in ordered_components:
        base = component_styles.get(c, (style_lookup[c][0], style_lookup[c][1], None))
        label = base[1]
        if not label:
            continue
        style_handles.append(
            Line2D(
                [0],
                [0],
                color=(base[2] or "k"),
                linestyle=base[0],
                linewidth=2.5,
                label=label,
            )
        )
    if style_handles:
        style_anchor_y = -0.06 - 0.12 * (color_rows - 1)
        style_legend = legend_ax.legend(
            style_handles,
            [h.get_label() for h in style_handles],
            loc="lower center",
            ncol=len(style_handles),
            fontsize=10,
            frameon=False,
            bbox_to_anchor=(0.5, style_anchor_y),
            handlelength=3.0,
            handletextpad=0.8,
        )
        if color_legend is not None:
            legend_ax.add_artist(color_legend)
    elif color_legend is not None:
        legend_ax.add_artist(color_legend)

    return tagline


def plot_activation_figure(data: Dict[str, Any], output: Path, dpi: int, param_tag: str) -> None:
    """Plot activation upper bounds: MLP, QKV, Embedding & Logits."""
    activation_history = _serialize_history(data.get("activation", {}))
    activation_history = {k: v for k, v in activation_history.items() if _activation_component(k) == "mlp.layer1"}
    qkv_column_history = _serialize_history(data.get("qkv_columns", {}))
    embed_logits_history = _serialize_history(data.get("embed_lmhead", {}))
    rms_history = _serialize_history(data.get("rms", {}))
    token_frequency_payload = data.get("token_frequency") or {}
    token_frequency_history = {}
    if "steps" in token_frequency_payload and "values" in token_frequency_payload:
        steps = token_frequency_payload.get("steps") or []
        values = token_frequency_payload.get("values") or []
        token_frequency_history["token_inv_pmax"] = list(zip(steps, values))
 
    activation_styles = {
        "mlp.layer1": ("-", ""),
    }
    qkv_column_styles = {
        "q": ("-", r"$Q$"),
        "k": ((0, (4, 2)), r"$K$"),  # medium dash
        "v": ((0, (1, 1)), r"$V$"),  # dotted
    }
    rms_styles = {
        "pre_attn": ("-", "RMSNorm before attention"),
        "pre_mlp": ((0, (2, 2)), "RMSNorm before MLP"),
        "lm_head_pre": ((0, (1, 1)), "RMSNorm before LM head", "#ff7f0e"),
    }
    embed_logits_styles = {
        "embedding": ("-", "Embedding", "#1f77b4"),
        "token_inv_pmax": ((0, (6, 2)), "Token-Indicator", "#d62728"),
    }
    combined_embed_history = {}
    if "embedding" in embed_logits_history:
        combined_embed_history["embedding"] = embed_logits_history["embedding"]
    if token_frequency_history.get("token_inv_pmax"):
        combined_embed_history["token_inv_pmax"] = token_frequency_history["token_inv_pmax"]

    panels: List[tuple[tuple[str, str], Dict[str, List[tuple[int, float]]], Any, Dict[str, tuple[Any, ...]], str]] = []
    if activation_history:
        panels.append((_format_title("MLP post-activation stable rank", param_tag), activation_history, _activation_component, activation_styles, "mlp"))
    if rms_history:
        panels.append((_format_title("RMSNorm activation stable rank", param_tag), rms_history, _rms_component, rms_styles, "rms"))
    if qkv_column_history:
        panels.append((_format_title("Attention post-activation stable rank", param_tag), qkv_column_history, _qkv_column_component, qkv_column_styles, "attention"))
    if combined_embed_history:
        panels.append((_format_title("Embedding post-activation and Token-Indicator stable rank", param_tag), combined_embed_history, _embed_logits_component, embed_logits_styles, "embedding"))

    if not panels:
        print("No activation data to plot")
        return

    block_colors = _gather_block_colors([panel[1] for panel in panels])
    for title, history, component_fn, styles, suffix in panels:
        fig = plt.figure(figsize=(10, 3.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[0.35, 1.0], hspace=0.3)
        legend_ax = fig.add_subplot(gs[0])
        legend_ax.axis("off")
        data_ax = fig.add_subplot(gs[1])
        tagline = _render_panel(data_ax, legend_ax, title, history, component_fn, styles, block_colors)
        limits = _compute_x_limits(history)
        if limits is not None:
            min_step, max_step = limits
            margin = max(1, int(0.01 * (max_step - min_step)))
            data_ax.set_xlim(min_step - margin, max_step + margin)
        xlabel = "Training iteration" if not tagline else f"Training iteration\n{tagline}"
        data_ax.set_xlabel(xlabel, fontsize=12)
        fig.tight_layout()
        output_path = output.parent / f"{output.stem}_activation_{suffix}.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved activation plot to {output_path}")


def _render_gradient_panel(
    ax: plt.Axes,
    legend_ax: plt.Axes,
    title: tuple[str, str] | str,
    history: Dict[str, List[tuple[int, float]]],
    component_fn,
    style_lookup: Dict[str, tuple[Any, ...]],
    block_colors: Dict[int, Any],
) -> str:
    """Specialized render for gradient panels with custom y-label."""
    default_color = plt.cm.get_cmap("tab20")(0)
    block_usage: Dict[int, Any] = {}
    component_usage: set[str] = set()
    component_styles: Dict[str, tuple[Any, str, Any | None]] = {}

    for key in sorted(history.keys()):
        data = history[key]
        if not data:
            continue
        steps = [s for s, _ in data]
        values = [v for _, v in data]
        block_idx = _parse_block_index(key)
        color = block_colors.get(block_idx, default_color)
        component = component_fn(key)
        linestyle, label, color_override = "-", component or key, None
        if component in style_lookup:
            style_info = style_lookup[component]
            if len(style_info) >= 3:
                linestyle, label, color_override = style_info[0], style_info[1], style_info[2]
            else:
                linestyle, label = style_info[0], style_info[1]
                color_override = None
            component_usage.add(component)
            component_styles[component] = (linestyle, label, color_override)
        plot_color = color_override if color_override is not None else color
        ax.plot(steps, values, color=plot_color, linestyle=linestyle, linewidth=1.6)
        if block_idx is not None:
            block_usage.setdefault(block_idx, color)

    tagline = ""
    base_title = title
    if isinstance(title, tuple):
        tagline, base_title = title
    elif isinstance(title, str) and "\n" in title:
        tagline, base_title = [part.strip() for part in title.split("\n", 1)]
    ax.set_title(base_title, fontsize=14)
    ax.set_ylabel("Nuclear rank", fontsize=12)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.tick_params(labelsize=11)

    # Block colors legend
    color_handles = [
        Line2D([0], [0], color=color, linestyle="-", linewidth=2.5, label=f"block{idx}")
        for idx, color in sorted(block_usage.items())
    ]
    color_legend = None
    color_rows = 1
    if color_handles:
        color_ncol = min(len(color_handles), 6)
        color_rows = max(1, math.ceil(len(color_handles) / color_ncol))
        color_anchor_y = 1.0 + 0.18 * (color_rows - 1)
        color_legend = legend_ax.legend(
            color_handles,
            [h.get_label() for h in color_handles],
            loc="upper center",
            ncol=color_ncol,
            fontsize=10,
            frameon=False,
            bbox_to_anchor=(0.5, color_anchor_y),
            handlelength=2.2,
            handletextpad=0.8,
        )
    
    # Component style legend
    ordered_components = [c for c in style_lookup if c in component_usage]
    if not ordered_components:
        ordered_components = list(style_lookup.keys())
    style_handles = []
    for c in ordered_components:
        base = component_styles.get(c, (style_lookup[c][0], style_lookup[c][1], None))
        label = base[1]
        if not label:
            continue
        style_handles.append(
            Line2D(
                [0],
                [0],
                color=(base[2] or "k"),
                linestyle=base[0],
                linewidth=2.5,
                label=label,
            )
        )
    if style_handles:
        style_anchor_y = -0.06 - 0.12 * (color_rows - 1)
        style_legend = legend_ax.legend(
            style_handles,
            [h.get_label() for h in style_handles],
            loc="lower center",
            ncol=len(style_handles),
            fontsize=10,
            frameon=False,
            bbox_to_anchor=(0.5, style_anchor_y),
            handlelength=3.0,
            handletextpad=0.8,
        )
        if color_legend is not None:
            legend_ax.add_artist(color_legend)
    elif color_legend is not None:
        legend_ax.add_artist(color_legend)

    return tagline


def plot_gradient_figure(data: Dict[str, Any], output: Path, dpi: int, param_tag: str) -> None:
    """Plot gradient sr_nuc: MLP, Attention, and Embedding/LM head."""
    grad_mlp_history = _serialize_history(data.get("gradient", {}).get("mlp", {}))
    grad_attn_history = _serialize_history(data.get("gradient", {}).get("attention", {}))
    grad_embed_history = _serialize_history(data.get("gradient", {}).get("embed_lmhead", {}))
    grad_value_history = _serialize_history(data.get("gradient", {}).get("value_embed", {}))

    mlp_gradient_styles = {
        "mlp.layer1": ("-", "MLP layer 1"),
        "mlp.layer2": ((0, (4, 4)), "MLP layer 2"),
    }
    attn_gradient_styles = {
        "attn.c_proj": ("-", r"$W_O$"),
        "attn.q": ("-", r"$W_Q$"),
        "attn.k": ((0, (4, 2)), r"$W_K$"),
        "attn.v": ("-", r"$W_V$"),
    }
    embed_gradient_styles = {
        "embedding": ("-", "Embedding", "#1f77b4"),
        "lm_head": ((0, (2, 2)), "LM head", "#d62728"),
        "value_embed0": ((0, (3, 2)), "Value embed 0", "#2ca02c"),
        "value_embed1": ((0, (1, 3)), "Value embed 1", "#9467bd"),
        "value_embed2": ((0, (5, 2)), "Value embed 2", "#8c564b"),
    }
 
    combined_grad_embed_history = {**grad_embed_history, **grad_value_history}

    panels: List[tuple[tuple[str, str], Dict[str, List[tuple[int, float]]], Any, Dict[str, tuple[Any, ...]], str]] = []
    if grad_mlp_history:
        panels.append((_format_title(r"MLP nuclear rank of gradient: $\mathrm{nr}(\nabla_W L(W))$", param_tag), grad_mlp_history, lambda key: _gradient_component("mlp", key), mlp_gradient_styles, "mlp"))
    def _filter_attn(history: Dict[str, List[tuple[int, float]]], allowed: set[str]) -> Dict[str, List[tuple[int, float]]]:
        filtered: Dict[str, List[tuple[int, float]]] = {}
        for key, data in history.items():
            component = _gradient_component("attention", key)
            if component in allowed:
                filtered[key] = data
        return filtered

    attn_qk_history = _filter_attn(grad_attn_history, {"attn.q", "attn.k"})
    attn_v_history = _filter_attn(grad_attn_history, {"attn.v"})
    attn_o_history = _filter_attn(grad_attn_history, {"attn.c_proj"})

    if attn_qk_history:
        qk_styles = {comp: attn_gradient_styles[comp] for comp in ("attn.q", "attn.k")}
        panels.append(
            (
                _format_title(r"Attention nuclear rank ($W_Q$ / $W_K$): $\mathrm{nr}(\nabla_W L(W))$", param_tag),
                attn_qk_history,
                lambda key: _gradient_component("attention", key),
                qk_styles,
                "attention_qk",
            )
        )
    if attn_v_history:
        v_styles = {"attn.v": attn_gradient_styles["attn.v"]}
        panels.append(
            (
                _format_title(r"Attention nuclear rank ($W_V$): $\mathrm{nr}(\nabla_W L(W))$", param_tag),
                attn_v_history,
                lambda key: _gradient_component("attention", key),
                v_styles,
                "attention_v",
            )
        )
    if attn_o_history:
        o_styles = {"attn.c_proj": attn_gradient_styles["attn.c_proj"]}
        panels.append(
            (
                _format_title(r"Attention nuclear rank ($W_O$): $\mathrm{nr}(\nabla_W L(W))$", param_tag),
                attn_o_history,
                lambda key: _gradient_component("attention", key),
                o_styles,
                "attention_o",
            )
        )
    if combined_grad_embed_history:
        panels.append((_format_title(r"Embedding/LM head/value embedding nuclear rank of gradient: $\mathrm{nr}(\nabla_W L(W))$", param_tag), combined_grad_embed_history, _embed_gradient_component, embed_gradient_styles, "embedding"))

    if not panels:
        print("No gradient data to plot")
        return

    block_colors = _gather_block_colors([panel[1] for panel in panels])
    for title, history, component_fn, styles, suffix in panels:
        fig = plt.figure(figsize=(10, 3.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[0.35, 1.0], hspace=0.3)
        legend_ax = fig.add_subplot(gs[0])
        legend_ax.axis("off")
        data_ax = fig.add_subplot(gs[1])
        tagline = _render_gradient_panel(data_ax, legend_ax, title, history, component_fn, styles, block_colors)
        limits = _compute_x_limits(history)
        if limits is not None:
            min_step, max_step = limits
            margin = max(1, int(0.01 * (max_step - min_step)))
            data_ax.set_xlim(min_step - margin, max_step + margin)
        xlabel = "Training iteration" if not tagline else f"Training iteration\n{tagline}"
        data_ax.set_xlabel(xlabel, fontsize=12)
        fig.tight_layout()
        output_path = output.parent / f"{output.stem}_gradient_{suffix}.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved gradient plot to {output_path}")


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.input.parent

    with args.input.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    stem = args.input.stem
    base_path = output_dir / stem

    plot_activation_figure(metrics, base_path, args.dpi, args.param_tag)
    plot_gradient_figure(metrics, base_path, args.dpi, args.param_tag)


if __name__ == "__main__":
    main()

