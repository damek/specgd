"""
Render spectrum-tracker GIFs for weighted_spectral_vs_gd outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


LAYER_DISPLAY = {
    "layer1": "Layer 1",
    "layer2": "Layer 2",
    "layer3": "Layer 3",
    "layer4": "Layer 4",
}

DEFAULT_COLORS = {
    "loss": "#1f77b4",
    "layer2": "#ff7f0e",
    "layer3": "#2ca02c",
    "layer4": "#9467bd",
}

METHOD_LABELS = {
    "gradient_descent": r"$\mathtt{GD}$",
    "specgd": r"$\mathtt{SpecGD}$",
}


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _map_stable_rank_series(method_payload: Dict[str, Dict], layers: List[str]) -> Dict[str, Dict[str, List[float]]]:
    sr_map = {}
    for idx, layer in enumerate(layers, start=1):
        source_key = f"layer{idx}"
        if source_key in method_payload["activations"]:
            sr_map[layer] = method_payload["activations"][source_key]
    return sr_map


def _map_grad_rank_series(method_payload: Dict[str, Dict], weights: List[str]) -> Dict[str, Dict[str, List[float]]]:
    grad_map = method_payload.get("grad_ratio", {})
    series = {}
    for weight in weights:
        entry = grad_map.get(weight)
        series[weight] = entry if entry else {"steps": [], "values": []}
    return series


def _global_hist_limits(frames: List[Dict], layers: List[str]) -> Dict[str, float]:
    limits = {layer: 1.0 for layer in layers}
    for layer in layers:
        max_val = 0.0
        for frame in frames:
            values = frame["spectra"].get(layer, [])
            if values:
                max_val = max(max_val, max(values))
        limits[layer] = max(max_val, 1.0)
    return limits


def _render_frame(
    *,
    frame_step: int,
    output_path: Path,
    loss_series_map: Dict[str, Dict[str, List[float]]],
    sr_series_map: Dict[str, Dict[str, Dict[str, List[float]]]],
    layers: List[str],
    hist_limits_map: Dict[str, Dict[str, float]],
    spectra_map: Dict[str, List[Dict]],
    dpi: int,
) -> None:
    method_names = list(spectra_map.keys())
    extra_rows = max(0, len(method_names) - 1)
    fig = plt.figure(figsize=(13, 6 + 2 * extra_rows))
    gs = fig.add_gridspec(2 + extra_rows, 6, height_ratios=[1.1, 1] + [1] * extra_rows)
    ax_loss = fig.add_subplot(gs[0, 0:3])
    ax_sr = fig.add_subplot(gs[0, 3:6])

    # Training loss overlay
    for method_name, loss_series in loss_series_map.items():
        ax_loss.semilogy(
            loss_series["steps"],
            loss_series["values"],
            label=METHOD_LABELS.get(method_name, method_name),
            linewidth=2.0,
        )
    ax_loss.axvline(frame_step, color="#d62728", linestyle="--", linewidth=2.0)
    ax_loss.set_title("Training loss", fontsize=18)
    ax_loss.set_xlabel("Iteration", fontsize=16)
    ax_loss.set_ylabel("MSE", fontsize=16)
    ax_loss.grid(True, linestyle="--", alpha=0.4)
    ax_loss.tick_params(axis="both", labelsize=14)
    ax_loss.legend(fontsize=14)

    # Stable rank overlay
    for method_name, sr_series in sr_series_map.items():
        for layer in layers:
            series = sr_series.get(layer)
            if not series:
                continue
            ax_sr.plot(
                series["steps"],
                series["values"],
                label=rf"{METHOD_LABELS.get(method_name, method_name)} {LAYER_DISPLAY.get(layer, layer)}",
                linewidth=1.8,
            )
    ax_sr.axvline(frame_step, color="#d62728", linestyle="--", linewidth=2.0)
    ax_sr.set_title("Post-activation stable ranks", fontsize=18)
    ax_sr.set_xlabel("Iteration", fontsize=16)
    ax_sr.set_ylabel("Stable rank", fontsize=16)
    ax_sr.grid(True, linestyle="--", alpha=0.4)
    ax_sr.tick_params(axis="both", labelsize=14)
    ax_sr.legend(fontsize=12, ncol=2, columnspacing=1.5)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for row_idx, method_name in enumerate(method_names):
        row_axes = [
            fig.add_subplot(gs[row_idx + 1, 0:2]),
            fig.add_subplot(gs[row_idx + 1, 2:4]),
            fig.add_subplot(gs[row_idx + 1, 4:6]),
        ]
        hist_limits = hist_limits_map[method_name]
        frame = next((f for f in spectra_map[method_name] if f["step"] == frame_step), None)
        for col_idx, layer in enumerate(layers):
            ax = row_axes[col_idx]
            if frame is None:
                ax.axis("off")
                continue
            singulars = frame["spectra"].get(layer, [])
            if not singulars:
                ax.axis("off")
                continue
            singulars_np = np.array(singulars, dtype=np.float64)
            bins = min(90, max(20, singulars_np.size // 60))
            ax.hist(
                singulars_np,
                bins=bins,
                density=True,
                log=True,
                color=colors[col_idx % len(colors)],
                alpha=0.85,
            )
            ax.set_xlim(left=0.0, right=hist_limits[layer] * 1.05)
            ax.set_title(
                rf"{LAYER_DISPLAY.get(layer, layer)} ({METHOD_LABELS.get(method_name, method_name)})",
                fontsize=16,
            )
            ax.set_xlabel("Singular value", fontsize=14)
            ax.set_ylabel("Density (log)", fontsize=14)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
            ax.tick_params(axis="both", labelsize=12)

    fig.suptitle(f"Iteration {frame_step}", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=1.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _render_frame_with_grad(
    *,
    frame_step: int,
    output_path: Path,
    loss_series_map: Dict[str, Dict[str, List[float]]],
    sr_series_map: Dict[str, Dict[str, Dict[str, List[float]]]],
    grad_series_map: Dict[str, Dict[str, List[float]]],
    layers: List[str],
    hist_limits_map: Dict[str, Dict[str, float]],
    spectra_map: Dict[str, List[Dict]],
    dpi: int,
) -> None:
    method_names = list(spectra_map.keys())
    extra_rows = max(0, len(method_names) - 1)
    fig = plt.figure(figsize=(14, 7 + 2 * extra_rows))
    gs = fig.add_gridspec(2 + extra_rows, 9, height_ratios=[1.1, 1] + [1] * extra_rows)
    ax_loss = fig.add_subplot(gs[0, 0:3])
    ax_sr = fig.add_subplot(gs[0, 3:6])
    ax_grad = fig.add_subplot(gs[0, 6:9])

    for method_name, loss_series in loss_series_map.items():
        ax_loss.semilogy(
            loss_series["steps"],
            loss_series["values"],
            label=METHOD_LABELS.get(method_name, method_name),
            linewidth=2.0,
        )
    ax_loss.axvline(frame_step, color="#d62728", linestyle="--", linewidth=2.0)
    ax_loss.set_title("Training loss", fontsize=18)
    ax_loss.set_xlabel("Iteration", fontsize=16)
    ax_loss.set_ylabel("MSE", fontsize=16)
    ax_loss.grid(True, linestyle="--", alpha=0.4)
    ax_loss.tick_params(axis="both", labelsize=14)
    ax_loss.legend(fontsize=14)

    for method_name, sr_series in sr_series_map.items():
        for layer in layers:
            series = sr_series.get(layer)
            if not series:
                continue
            ax_sr.plot(
                series["steps"],
                series["values"],
                label=rf"{METHOD_LABELS.get(method_name, method_name)} {LAYER_DISPLAY.get(layer, layer)}",
                linewidth=1.8,
            )
    ax_sr.axvline(frame_step, color="#d62728", linestyle="--", linewidth=2.0)
    ax_sr.set_title("Post-activation stable ranks", fontsize=18)
    ax_sr.set_xlabel("Iteration", fontsize=16)
    ax_sr.set_ylabel("Stable rank", fontsize=16)
    ax_sr.grid(True, linestyle="--", alpha=0.4)
    ax_sr.tick_params(axis="both", labelsize=14)
    ax_sr.legend(fontsize=12, ncol=2, columnspacing=1.5)

    for method_name, layer_series in grad_series_map.items():
        for weight_name, grad_series in layer_series.items():
            if not grad_series["steps"]:
                continue
            ax_grad.plot(
                grad_series["steps"],
                grad_series["values"],
                label=rf"{METHOD_LABELS.get(method_name, method_name)} {weight_name}",
                linewidth=1.8,
            )
    ax_grad.axvline(frame_step, color="#d62728", linestyle="--", linewidth=2.0)
    ax_grad.set_title("Gradient nuclear ranks", fontsize=18)
    ax_grad.set_xlabel("Iteration", fontsize=16)
    ax_grad.set_ylabel("Nuclear rank", fontsize=16)
    ax_grad.grid(True, linestyle="--", alpha=0.4)
    ax_grad.tick_params(axis="both", labelsize=14)
    ax_grad.legend(fontsize=12, ncol=2, columnspacing=1.2)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for row_idx, method_name in enumerate(method_names):
        row_axes = [
            fig.add_subplot(gs[row_idx + 1, 0:3]),
            fig.add_subplot(gs[row_idx + 1, 3:6]),
            fig.add_subplot(gs[row_idx + 1, 6:9]),
        ]
        hist_limits = hist_limits_map[method_name]
        frame = next((f for f in spectra_map[method_name] if f["step"] == frame_step), None)
        for col_idx, layer in enumerate(layers):
            ax = row_axes[col_idx]
            if frame is None:
                ax.axis("off")
                continue
            singulars = frame["spectra"].get(layer, [])
            if not singulars:
                ax.axis("off")
                continue
            singulars_np = np.array(singulars, dtype=np.float64)
            bins = min(90, max(20, singulars_np.size // 60))
            ax.hist(
                singulars_np,
                bins=bins,
                density=True,
                log=True,
                color=colors[col_idx % len(colors)],
                alpha=0.85,
            )
            ax.set_xlim(left=0.0, right=hist_limits[layer] * 1.05)
            ax.set_title(
                rf"{LAYER_DISPLAY.get(layer, layer)} ({METHOD_LABELS.get(method_name, method_name)})",
                fontsize=16,
            )
            ax.set_xlabel("Singular value", fontsize=14)
            ax.set_ylabel("Density (log)", fontsize=14)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
            ax.tick_params(axis="both", labelsize=12)

    fig.suptitle(f"Iteration {frame_step}", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=1.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_weighted_spec_spectrum_gif(
    *,
    log_path: Path,
    methods: List[str] | None,
    output_dir: Path,
    gif_name: str,
    fps: float,
    dpi: int,
    keep_frames: bool,
) -> None:
    data = _load_json(log_path)
    spectrum_map = data.get("spectrum_gif")
    if not isinstance(spectrum_map, dict):
        print(f"[spectrum-gif] No spectrum data in {log_path}")
        return

    frames_per_method: Dict[str, List[Dict]] = {}
    loss_series_map: Dict[str, Dict[str, List[float]]] = {}
    sr_series_map: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    grad_series_map: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    hist_limits_map: Dict[str, Dict[str, float]] = {}
    layers = None

    method_names = methods or list(spectrum_map.keys())
    for method in method_names:
        payload = spectrum_map.get(method)
        if payload is None:
            print(f"[spectrum-gif] Method '{method}' missing spectrum data; skipping.")
            continue
        frames = payload.get("frames", [])
        if not frames:
            continue
        frames_per_method[method] = frames
        if layers is None:
            layers = payload.get("layers", ["layer2", "layer3", "layer4"])
        method_payload = data["methods"].get(method)
        if method_payload is None:
            continue
        loss_series_map[method] = method_payload["loss"]
        sr_series_map[method] = _map_stable_rank_series(method_payload, layers)
        grad_series_map[method] = _map_grad_rank_series(method_payload, ["W1", "W2", "W3"])
        hist_limits_map[method] = _global_hist_limits(frames, layers)

    if not frames_per_method:
        print(f"[spectrum-gif] No spectrum frames to plot in {log_path}")
        return

    reference_method = next(iter(frames_per_method))
    frame_steps = [frame["step"] for frame in frames_per_method[reference_method]]
    if not frame_steps or frame_steps[0] != 0:
        frame_steps = [0] + frame_steps
    max_step = max(frame_steps)
    total_steps = spectrum_map[reference_method].get("total_steps")
    if total_steps is not None and total_steps not in frame_steps:
        frame_steps.append(total_steps)

    frames_dir = output_dir / "spectrum_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: List[Path] = []
    for idx, step in enumerate(frame_steps):
        frame_path = frames_dir / f"frame_{idx:03d}.png"
        _render_frame(
            frame_step=step,
            output_path=frame_path,
            loss_series_map=loss_series_map,
            sr_series_map=sr_series_map,
            layers=layers or ["layer2", "layer3", "layer4"],
            hist_limits_map=hist_limits_map,
            spectra_map=frames_per_method,
            dpi=dpi,
        )
        frame_paths.append(frame_path)

    gif_path = output_dir / (gif_name if gif_name.endswith(".gif") else f"{gif_name}.gif")
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    duration_ms = max(int(1000 / fps), 1) if fps > 0 else 200
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"[spectrum-gif] Saved GIF to {gif_path} ({len(frame_paths)} frames, fps={fps})")

    if not keep_frames:
        for path in frame_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            frames_dir.rmdir()
        except OSError:
            pass

    grad_frames_dir = output_dir / "spectrum_frames_grad"
    grad_frames_dir.mkdir(parents=True, exist_ok=True)
    grad_frame_paths: List[Path] = []
    for idx, step in enumerate(frame_steps):
        frame_path = grad_frames_dir / f"frame_{idx:03d}.png"
        _render_frame_with_grad(
            frame_step=step,
            output_path=frame_path,
            loss_series_map=loss_series_map,
            sr_series_map=sr_series_map,
            grad_series_map=grad_series_map,
            layers=layers or ["layer2", "layer3", "layer4"],
            hist_limits_map=hist_limits_map,
            spectra_map=frames_per_method,
            dpi=dpi,
        )
        grad_frame_paths.append(frame_path)

    grad_gif_name = gif_name if gif_name.endswith(".gif") else f"{gif_name}.gif"
    grad_gif_path = output_dir / grad_gif_name.replace(".gif", "_toprow.gif")
    grad_images = [Image.open(path).convert("RGB") for path in grad_frame_paths]
    grad_images[0].save(
        grad_gif_path,
        save_all=True,
        append_images=grad_images[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"[spectrum-gif] Saved top-row GIF to {grad_gif_path} ({len(grad_frame_paths)} frames, fps={fps})")

    if not keep_frames:
        for path in grad_frame_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            grad_frames_dir.rmdir()
        except OSError:
            pass


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render spectrum GIF from weighted_spectral_vs_gd log")
    parser.add_argument("--input", type=Path, required=True, help="JSON log produced by weighted_spectral_vs_gd.py")
    parser.add_argument("--methods", type=str, nargs="*", default=None, help="Method names to visualize (default: all)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for GIF (default: log directory)")
    parser.add_argument("--gif-name", type=str, default="spectrum.gif")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_weighted_spec_spectrum_gif(
        log_path=args.input,
        methods=args.methods,
        output_dir=output_dir,
        gif_name=args.gif_name,
        fps=args.fps,
        dpi=args.dpi,
        keep_frames=args.keep_frames,
    )


if __name__ == "__main__":
    main()

