"""
Lambda sweep for the toy random-feature GD vs SpecGD problem.

This script generates a sequence of figures (objective gap + nuclear-rank panels)
for matrices of the form A = I + λ · 11ᵀ, then stitches them into an animated GIF
so you can watch how the dynamics change as the activation stable rank varies.

No other modules are imported from this repo; the implementation is self-contained.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib as mpl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"

from PIL import Image
import torch


EPS = 1e-12
METHOD_STYLES = {
    "gradient_descent": {"label": "GD", "color": "#1f77b4", "linestyle": "-", "linewidth": 2.2},
    "specgd": {"label": "SpecGD", "color": "#ff7f0e", "linestyle": "-", "linewidth": 2.2},
    "specgd_from_peak": {"label": "SpecGD from peak", "color": "#8b0000", "linestyle": "-", "linewidth": 2.4},
}


def stable_rank_exact(matrix: torch.Tensor) -> Tuple[float, float, float]:
    mat64 = matrix.to(torch.float64)
    fro_sq = torch.sum(mat64 * mat64)
    fro_val = fro_sq.item()
    op_norm = torch.linalg.matrix_norm(mat64, ord=2).item()
    sr = (fro_val / (op_norm * op_norm)) if op_norm > 0 else float("inf")
    return sr, op_norm, math.sqrt(fro_val) if fro_val > 0 else 0.0


def spectral_polar_and_nuclear(grad: torch.Tensor) -> Tuple[torch.Tensor, float]:
    grad_matrix = grad if grad.ndim == 2 else grad.reshape(grad.shape[0], -1)
    if grad_matrix.numel() == 0:
        return torch.zeros_like(grad), 0.0
    g64 = grad_matrix.to(torch.float64)
    gram = g64 @ g64.T
    gram = 0.5 * (gram + gram.T)
    evals, evecs = torch.linalg.eigh(gram)
    evals = torch.clamp(evals, min=0.0)
    sing = torch.sqrt(evals)
    nuclear = sing.sum().item()
    mask = sing > 0
    if mask.any():
        vectors = evecs[:, mask]
        inv_root = vectors / sing[mask]
        sqrt_inv = inv_root @ vectors.T
        polar = sqrt_inv @ g64
    else:
        polar = torch.zeros_like(g64)
    return polar.to(dtype=grad_matrix.dtype).view_as(grad), nuclear


def build_activation_matrix(dim: int, lam: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    identity = torch.eye(dim, dtype=dtype, device=device)
    ones = torch.ones(dim, dim, dtype=dtype, device=device)
    return identity + lam * ones


def _init_history() -> Dict[str, List[float]]:
    return {"loss": [], "nuclear_rank": [], "steps": []}


def _record_history(history: Dict[str, List[float]], loss: float, nuclear_rank: float, step: int):
    history["loss"].append(loss)
    history["nuclear_rank"].append(nuclear_rank)
    history["steps"].append(step)


def run_methods(
    A: torch.Tensor,
    Y: torch.Tensor,
    num_steps: int,
    lr_gd: float,
    lr_specgd: float,
) -> Tuple[Dict[str, Dict[str, List[float]]], List[torch.Tensor]]:
    dim = A.size(0)
    n = A.size(1)
    methods = {
        "gradient_descent": {
            "weight": torch.zeros(dim, dim, dtype=A.dtype, device=A.device),
            "loss": [],
            "nuclear_rank": [],
            "steps": [],
        },
        "specgd": {
            "weight": torch.zeros(dim, dim, dtype=A.dtype, device=A.device),
            "loss": [],
            "nuclear_rank": [],
            "steps": [],
        },
    }
    gd_snapshots: List[torch.Tensor] = []

    for step in range(num_steps):
        gd_snapshots.append(methods["gradient_descent"]["weight"].detach().clone())
        for name, payload in methods.items():
            W = payload["weight"]
            diff = W @ A - Y
            loss = 0.5 * torch.linalg.norm(diff, ord="fro") ** 2 / n
            grad = (diff @ A.T) / n
            fro_sq = torch.sum(grad * grad).item()
            polar, nuclear = spectral_polar_and_nuclear(grad)
            if fro_sq == 0.0:
                nuclear_rank = float("inf") if nuclear > 0 else 0.0
            else:
                nuclear_rank = (nuclear * nuclear) / fro_sq

            payload["loss"].append(loss.item())
            payload["nuclear_rank"].append(nuclear_rank)
            payload["steps"].append(step)

            with torch.no_grad():
                if name == "specgd":
                    if nuclear > 0.0:
                        W.add_(polar, alpha=-lr_specgd * nuclear)
                else:
                    W.add_(grad, alpha=-lr_gd)

    for payload in methods.values():
        payload.pop("weight", None)

    return methods, gd_snapshots


def run_specgd_restart(
    A: torch.Tensor,
    Y: torch.Tensor,
    num_steps: int,
    start_iter: int,
    init_weight: torch.Tensor,
    lr_specgd: float,
) -> Dict[str, List[float]]:
    if start_iter >= num_steps:
        raise ValueError("Start iteration must be < num_steps")
    dim = A.size(0)
    model_weight = init_weight.clone().to(A.dtype)
    history = _init_history()
    n = A.size(1)
    for step in range(start_iter, num_steps):
        diff = (model_weight @ A) - Y
        loss = 0.5 * torch.linalg.norm(diff, ord="fro") ** 2 / n
        grad = (diff @ A.T) / n
        polar, nuclear = spectral_polar_and_nuclear(grad)
        fro_sq = torch.sum(grad * grad).item()
        if fro_sq == 0.0:
            nuclear_rank = float("inf") if nuclear > 0 else 0.0
        else:
            nuclear_rank = (nuclear * nuclear) / fro_sq
        _record_history(history, loss.item(), nuclear_rank, step)
        if nuclear > 0.0:
            model_weight.add_(polar, alpha=-lr_specgd * nuclear)

    return history


def get_style(name: str) -> Dict[str, object]:
    if name.startswith("specgd_from_iter_"):
        iter_idx = name.rsplit("_", 1)[-1]
        return {
            "label": f"SpecGD from iter {iter_idx}",
            "color": "#8b0000",
            "linestyle": (0, (4, 2)),
            "linewidth": 2.0,
        }
    return METHOD_STYLES.get(
        name, {"label": name, "color": "#555555", "linestyle": "-", "linewidth": 2.0}
    )


def render_frame(
    lambda_value: float,
    stable_rank: float,
    methods: Dict[str, Dict[str, List[float]]],
    output_path: Path,
    dpi: int,
    loss_ylim: Tuple[float, float],
    nuclear_ylim: Tuple[float, float],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax0 = axes[0]
    for name in methods.keys():
        series = methods[name]["loss"]
        steps = methods[name].get("steps") or list(range(len(series)))
        style = get_style(name)
        ax0.semilogy(
            steps,
            series,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
    ax0.set_title(r"Training loss  $\mathcal{L}(W) = \frac{1}{2}\|(W - W_\sharp)A\|_F^2$", fontsize=18)
    ax0.set_xlabel("Training iteration", fontsize=16)
    ax0.set_ylabel("MSE", fontsize=16)
    ax0.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ax0.legend(fontsize=16)
    ax0.set_ylim(loss_ylim)

    ax1 = axes[1]
    for name in methods.keys():
        series = methods[name]["nuclear_rank"]
        steps = methods[name].get("steps") or list(range(len(series)))
        style = get_style(name)
        ax1.plot(
            steps,
            series,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
    ax1.axhline(
        stable_rank,
        color="#333333",
        linestyle=(0, (4, 2)),
        linewidth=2.0,
        label=fr"st(A) = {stable_rank:.2f}",
    )
    ax1.set_title(r"Nuclear rank of gradient  $\mathrm{nr}(\nabla \mathcal{L}(W))$", fontsize=18)
    ax1.set_xlabel("Training iteration", fontsize=16)
    ax1.set_ylabel("Nuclear rank", fontsize=16)
    ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ax1.legend(fontsize=16)
    ax1.set_ylim(nuclear_ylim)

    fig.tight_layout(rect=(0, 0, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_gif(frame_paths: List[Path], gif_path: Path, fps: float) -> None:
    if not frame_paths:
        raise RuntimeError("No frames generated; cannot build GIF")
    frames = [Image.open(path).convert("RGB") for path in frame_paths]
    duration_ms = max(int(1000 / fps), 1) if fps > 0 else 200
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved GIF to {gif_path} ({len(frame_paths)} frames, fps={fps})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lambda sweep GIF for GD vs SpecGD")
    parser.add_argument("--dim", type=int, default=40, help="Matrix dimension (W ∈ ℝ^{d×d})")
    parser.add_argument("--num-steps", type=int, default=400, help="Iterations per method")
    parser.add_argument("--lambdas", type=float, nargs="*", default=None, help="Explicit λ list to sweep")
    parser.add_argument("--lambda-min", type=float, default=0.0, help="Min λ (inclusive) if --lambdas omitted")
    parser.add_argument("--lambda-max", type=float, default=0.12, help="Max λ (inclusive) if --lambdas omitted")
    parser.add_argument("--lambda-count", type=int, default=8, help="# of λ values between min/max")
    parser.add_argument(
        "--gd-lr",
        type=float,
        default=None,
        help="Optional override for GD step size; default uses n / ‖A‖₂² per random_feature_specgd.py",
    )
    parser.add_argument(
        "--specgd-lr",
        type=float,
        default=None,
        help="Optional override for SpecGD step size; default uses n / ‖A‖_F²",
    )
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for W♯")
    parser.add_argument("--output-dir", type=Path, default=Path("lambda_gif_logs"))
    parser.add_argument("--gif-name", type=str, default="stable_rank_sweep.gif")
    parser.add_argument("--fps", type=float, default=1.5, help="GIF playback rate (frames per second)")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI for intermediate frames")
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Retain per-lambda PNG frames instead of deleting them after the GIF is written",
    )
    parser.add_argument(
        "--specgd-from-peak",
        action="store_true",
        help="Plot an additional SpecGD trajectory initialized from the GD iter where nuclear rank peaks",
    )
    parser.add_argument(
        "--specgd-from-iter",
        type=int,
        nargs="*",
        default=None,
        help="Specific GD iteration indices to restart SpecGD from (space separated)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dtype = torch.float64
    device = torch.device("cpu")

    if args.lambdas is not None and len(args.lambdas) > 0:
        lambda_values = args.lambdas
    else:
        if args.lambda_count <= 1:
            lambda_values = [args.lambda_min]
        else:
            step = (args.lambda_max - args.lambda_min) / (args.lambda_count - 1)
            lambda_values = [args.lambda_min + i * step for i in range(args.lambda_count)]

    output_dir = args.output_dir
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: List[Path] = []

    cases = []
    loss_min, loss_max = float("inf"), 0.0
    nuclear_min, nuclear_max = float("inf"), 0.0

    for lam in lambda_values:
        A = build_activation_matrix(args.dim, lam, dtype, device)
        W_sharp = torch.randn(args.dim, args.dim, dtype=dtype, device=device) / math.sqrt(args.dim)
        Y = W_sharp @ A
        stable_rank, op_norm, fro_norm = stable_rank_exact(A)
        n = A.size(1)
        gd_lr = args.gd_lr if args.gd_lr is not None else n / max(op_norm * op_norm, EPS)
        spec_lr = (
            args.specgd_lr if args.specgd_lr is not None else n / max(fro_norm * fro_norm, EPS)
        )
        raw_methods, gd_snapshots = run_methods(A, Y, args.num_steps, gd_lr, spec_lr)

        for payload in raw_methods.values():
            payload["loss"] = [max(val, EPS) for val in payload["loss"]]
            payload["nuclear_rank"] = [max(val, EPS) for val in payload["nuclear_rank"]]
            if payload["loss"]:
                loss_min = min(loss_min, min(payload["loss"]))
                loss_max = max(loss_max, max(payload["loss"]))
            if payload["nuclear_rank"]:
                nuclear_min = min(nuclear_min, min(payload["nuclear_rank"]))
                nuclear_max = max(nuclear_max, max(payload["nuclear_rank"]))

        nuclear_min = min(nuclear_min, stable_rank)
        nuclear_max = max(nuclear_max, stable_rank)

        restarts_added: Dict[str, Dict[str, List[float]]] = {}
        gd_nuclear = raw_methods["gradient_descent"]["nuclear_rank"]
        gd_steps = raw_methods["gradient_descent"]["steps"]
        available_iters = {step: idx for idx, step in enumerate(gd_steps)}

        if args.specgd_from_peak and gd_nuclear:
            peak_idx = max(range(len(gd_nuclear)), key=lambda i: gd_nuclear[i])
            start_iter = gd_steps[peak_idx]
            if start_iter < args.num_steps:
                history = run_specgd_restart(
                    A, Y, args.num_steps, start_iter, gd_snapshots[peak_idx], spec_lr
                )
                history["loss"] = [max(val, EPS) for val in history["loss"]]
                history["nuclear_rank"] = [max(val, EPS) for val in history["nuclear_rank"]]
                restarts_added["specgd_from_peak"] = history

        if args.specgd_from_iter:
            unique_iters = sorted(set(i for i in args.specgd_from_iter if i is not None and i >= 0))
            for iter_idx in unique_iters:
                if iter_idx >= args.num_steps:
                    continue
                lookup = available_iters.get(iter_idx)
                if lookup is None:
                    continue
                history = run_specgd_restart(
                    A, Y, args.num_steps, iter_idx, gd_snapshots[lookup], spec_lr
                )
                key = f"specgd_from_iter_{iter_idx}"
                history["loss"] = [max(val, EPS) for val in history["loss"]]
                history["nuclear_rank"] = [max(val, EPS) for val in history["nuclear_rank"]]
                restarts_added[key] = history

        for key, history in restarts_added.items():
            raw_methods[key] = history
            if history["loss"]:
                loss_min = min(loss_min, min(history["loss"]))
                loss_max = max(loss_max, max(history["loss"]))
            if history["nuclear_rank"]:
                nuclear_min = min(nuclear_min, min(history["nuclear_rank"]))
                nuclear_max = max(nuclear_max, max(history["nuclear_rank"]))

        cases.append(
            {
                "lambda": lam,
                "stable_rank": stable_rank,
                "methods": raw_methods,
            }
        )

    if not math.isfinite(loss_min):
        loss_min = EPS
    if loss_max <= loss_min:
        loss_max = loss_min * 10
    loss_ylim = (max(loss_min * 0.9, EPS), loss_max * 1.05)

    if not math.isfinite(nuclear_min):
        nuclear_min = 1.0
    if not math.isfinite(nuclear_max):
        nuclear_max = nuclear_min * 10
    if nuclear_max <= nuclear_min + EPS:
        nuclear_max = nuclear_min * 10
    nuclear_ylim = (max(nuclear_min * 0.9, EPS), nuclear_max * 1.05)

    for idx, case in enumerate(cases):
        frame_path = frames_dir / f"frame_{idx:03d}.png"
        render_frame(
            lambda_value=case["lambda"],
            stable_rank=case["stable_rank"],
            methods=case["methods"],
            output_path=frame_path,
            dpi=args.dpi,
            loss_ylim=loss_ylim,
            nuclear_ylim=nuclear_ylim,
        )
        frame_paths.append(frame_path)
        print(f"[λ={case['lambda']:.4f}] frame {frame_path.name}")

    gif_path = output_dir / args.gif_name
    if gif_path.suffix == "":
        gif_path = gif_path.with_name(gif_path.name + ".gif")
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


