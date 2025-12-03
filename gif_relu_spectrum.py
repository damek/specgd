"""
ReLU(WX) spectrum animation for increasing dimensions.

For each requested dimension d, we sample Gaussian matrices
    W ∈ ℝ^{2d×d},  X ∈ ℝ^{d×8d},
compute A = ReLU(WX), and visualize the distribution of singular values
along with the dominant spike and the stable rank st(A).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
EPS = 1e-12


def relu_activation_spectrum(
    dim: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, float, float]:
    torch.manual_seed(seed)
    W = torch.randn(2 * dim, dim, device=device, dtype=dtype) / torch.sqrt(torch.tensor(dim, dtype=dtype, device=device))
    X = torch.randn(dim, 8 * dim, device=device, dtype=dtype)
    activations = F.relu(W @ X)
    singulars = torch.linalg.svdvals(activations.to(torch.float64))
    singulars_np = singulars.cpu().numpy()
    fro_sq = float(torch.sum(singulars * singulars))
    spike = float(torch.max(singulars))
    stable_rank = fro_sq / max(spike * spike, EPS)
    return singulars_np, stable_rank, spike


def render_histogram_frame(
    *,
    dim: int,
    singulars: np.ndarray,
    stable_rank: float,
    spike: float,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = min(90, max(20, singulars.size // 60))
    ax.hist(
        singulars,
        bins=bins,
        density=True,
        log=True,
        color="#1f77b4",
        alpha=0.8,
        label=r"Singular values of $\mathrm{ReLU}(WX)$",
    )
    ylim = ax.get_ylim()
    ymax = ylim[1]
    ax.axvline(
        spike,
        color="#d62728",
        linestyle=(0, (4, 2)),
        linewidth=2.2,
        label=f"Largest singular value ≈ {spike:.2f}",
    )
    ax.set_title(
        rf"$\mathrm{{ReLU}}(WX)$ spectrum  ($d={dim}$, $\mathrm{{st}}(A)={stable_rank:.2f}$)",
        fontsize=20,
    )
    ax.set_xscale("linear")
    ax.set_xlabel("Singular value", fontsize=18)
    ax.set_ylabel("Density (log scale)", fontsize=18)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=18, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_gif(frame_paths: Sequence[Path], gif_path: Path, fps: float) -> None:
    if not frame_paths:
        raise RuntimeError("No frames generated; cannot create GIF")
    gif_path = gif_path if gif_path.suffix else gif_path.with_name(gif_path.name + ".gif")
    frames = [Image.open(path).convert("RGB") for path in frame_paths]
    duration_ms = max(int(1000 / fps), 1) if fps > 0 else 200
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"[relu-spectrum] Saved GIF to {gif_path} ({len(frame_paths)} frames, fps={fps})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GIF of ReLU(WX) spectrum across dimensions")
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096],
        help="Dimensions d to sweep (W ∈ ℝ^{2d×d}, X ∈ ℝ^{d×8d})",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--device", default="cpu", help="Computation device, e.g., cpu or cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "gifs")
    parser.add_argument("--gif-name", type=str, default="relu_spectrum.gif")
    parser.add_argument("--fps", type=float, default=1.2, help="GIF playback rate")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for intermediate PNG frames")
    parser.add_argument("--keep-frames", action="store_true", help="Retain frame PNGs after GIF is written")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    output_dir = args.output_dir
    frames_dir = output_dir / "frames_relu_spectrum"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: List[Path] = []

    for idx, dim in enumerate(args.dims):
        singulars, stable_rank, spike = relu_activation_spectrum(
            dim=dim,
            seed=args.seed + idx,
            device=device,
            dtype=dtype,
        )
        frame_path = frames_dir / f"frame_{idx:03d}.png"
        render_histogram_frame(
            dim=dim,
            singulars=singulars,
            stable_rank=stable_rank,
            spike=spike,
            output_path=frame_path,
            dpi=args.dpi,
        )
        frame_paths.append(frame_path)
        print(
            f"[relu-spectrum] d={dim} st(A)={stable_rank:.2f} "
            f"max singular ≈ {spike:.2f} -> {frame_path.name}"
        )

    gif_path = output_dir / args.gif_name
    save_gif(frame_paths, gif_path, args.fps)

    if not args.keep_frames:
        for frame in frame_paths:
            try:
                frame.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            frames_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()


