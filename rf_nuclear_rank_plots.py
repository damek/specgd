"""
Nuclear-rank sweeps for the random-feature experiments.

This script reuses the random_feature_specgd utilities to run multiple
gradient-descent trajectories across feature dimensions and seeds, then
plots the average nuclear rank of the gradient for two models:

1. Realizable random feature regression (matching Model 1 in the paper).
2. Teacher–student random feature regression (Model 2).

Four plots are generated:
    - Model 1: nuclear rank vs step.
    - Model 1: initialization/first-step nuclear rank vs feature dimension.
    - Model 2: nuclear rank vs step.
    - Model 2: initialization/first-step nuclear rank vs feature dimension.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from random_feature_specgd import (
    ACTIVATIONS,
    ExperimentConfig,
    DEFAULT_DEVICE,
    build_random_feature_data,
    run_gradient_descent,
    stable_rank_exact,
)


mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"


def build_teacher_student_data(
    config: ExperimentConfig,
    activation_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor | float]:
    """Construct teacher–student dataset with the same structure as build_random_feature_data."""
    dim = config.dim
    hidden = config.feature_dim
    num_samples = config.num_samples
    dtype = config.dtype

    X = torch.randn(hidden, num_samples, device=device, dtype=dtype)
    V_student = torch.randn(dim, hidden, device=device, dtype=dtype) / torch.sqrt(torch.tensor(hidden, dtype=dtype))
    V_teacher = torch.randn(dim, hidden, device=device, dtype=dtype) / torch.sqrt(torch.tensor(hidden, dtype=dtype))

    pre_student = V_student @ X
    pre_teacher = V_teacher @ X
    A_student = activation_fn(pre_student)
    A_teacher = activation_fn(pre_teacher)

    W_teacher = torch.randn(dim, dim, device=device, dtype=dtype) / torch.sqrt(torch.tensor(dim, dtype=dtype))
    Y = W_teacher @ A_teacher

    sr_A, op_norm_A = stable_rank_exact(A_student)
    fro_norm_A = torch.linalg.matrix_norm(A_student, ord="fro").item()

    return {
        "A": A_student,
        "Y": Y,
        "W_sharp": W_teacher,
        "stable_rank_A": sr_A,
        "activation_op_norm": op_norm_A,
        "activation_fro_norm": fro_norm_A,
    }


def collect_nuclear_rank_curves(
    dims: Iterable[int],
    seeds: Iterable[int],
    steps: int,
    model_builder: Callable[[ExperimentConfig, Callable[[torch.Tensor], torch.Tensor], torch.device], Dict[str, torch.Tensor | float]],
    config_template: ExperimentConfig,
    device: torch.device,
    feature_dim_fn: Callable[[int], int],
    sample_count_fn: Callable[[int], int],
) -> Tuple[List[int], Dict[int, List[float]], Dict[int, float], Dict[int, float]]:
    """Run GD for each dimension/seed and average the nuclear rank curves."""
    activation_fn = ACTIVATIONS[config_template.activation]
    step_indices: List[int] = []
    avg_curves: Dict[int, List[float]] = {}
    nr0: Dict[int, float] = {}
    nr1: Dict[int, float] = {}

    for k in dims:
        sum_curve = None
        seeds_count = 0
        for seed in seeds:
            config = ExperimentConfig(
                dim=k,
                feature_dim=max(1, feature_dim_fn(k)),
                num_samples=max(1, sample_count_fn(k)),
                num_steps=steps,
                base_lr=config_template.base_lr,
                activation=config_template.activation,
                seed=seed,
                dtype=config_template.dtype,
            )
            torch.manual_seed(seed)
            data = model_builder(config, activation_fn, device)
            act_op = float(data["activation_op_norm"])
            lr = config.num_samples / (act_op * act_op) if act_op > 0 else config.base_lr
            gd_results, _ = run_gradient_descent(
                config=config,
                data=data,
                lr=lr,
                progress=False,
                checkpoint_iters=[],
            )
            curve = gd_results["grad_ratio"]["W1"]["values"]
            if not step_indices:
                step_indices = gd_results["grad_ratio"]["W1"]["steps"]
            tensor_curve = torch.tensor(curve, dtype=torch.float64)
            sum_curve = tensor_curve if sum_curve is None else sum_curve + tensor_curve
            seeds_count += 1
        if seeds_count == 0 or sum_curve is None:
            continue
        avg = (sum_curve / seeds_count).tolist()
        avg_curves[k] = avg
        nr0[k] = avg[0]
        nr1[k] = avg[1] if len(avg) > 1 else avg[0]

    return step_indices, avg_curves, nr0, nr1


def plot_steps(
    steps: List[int],
    curves: Dict[int, List[float]],
    dims: List[int],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in dims:
        if k not in curves:
            continue
        ax.plot(steps, curves[k], label=fr"$k={k}$", linewidth=2.0)
    ax.set_xlabel("Training iteration", fontsize=16)
    ax.set_ylabel(r"$\mathrm{nr}(\nabla L(W))$", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="best", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_first_step(
    dims: List[int],
    nr0: Dict[int, float],
    nr1: Dict[int, float],
    title: str,
    output_path: Path,
) -> None:
    xs = [k for k in dims if k in nr0]
    if not xs:
        return
    y0 = [nr0[k] for k in xs]
    y1 = [nr1[k] for k in xs]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, y0, marker="o", label="Initialization", linewidth=2.0)
    ax.plot(xs, y1, marker="s", label="After one step", linewidth=2.0)
    ax.set_xlabel("Feature dimension k", fontsize=16)
    ax.set_ylabel(r"$\mathrm{nr}(\nabla L(W))$", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="best", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nuclear rank plots for random-feature models")
    parser.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256, 512], help="Feature dimensions k to sweep")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds")
    parser.add_argument("--steps", type=int, default=200, help="Number of GD iterations")
    parser.add_argument("--num-samples", type=int, default=None, help="Override number of samples (default 2*k)")
    parser.add_argument("--feature-dim", type=int, default=None, help="Override input dimension d (default k/2)")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("logs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    dims = sorted(args.dims)
    seeds = sorted(args.seeds)

    def feature_dim_fn(k: int) -> int:
        if args.feature_dim is not None:
            return max(1, args.feature_dim)
        return max(1, k // 2)

    def sample_count_fn(k: int) -> int:
        if args.num_samples is not None:
            return max(1, args.num_samples)
        return max(1, 2 * k)

    base_config = ExperimentConfig(
        dim=dims[0],
        feature_dim=feature_dim_fn(dims[0]),
        num_samples=sample_count_fn(dims[0]),
        num_steps=args.steps,
        activation="relu",
        seed=0,
        dtype=dtype,
    )

    # Model 1: realizable random feature regression
    steps_m1, curves_m1, nr0_m1, nr1_m1 = collect_nuclear_rank_curves(
        dims=dims,
        seeds=seeds,
        steps=args.steps,
        model_builder=build_random_feature_data,
        config_template=base_config,
        device=device,
        feature_dim_fn=feature_dim_fn,
        sample_count_fn=sample_count_fn,
    )
    plot_steps(
        steps=steps_m1,
        curves=curves_m1,
        dims=dims,
        title="Model 1 (realizable) nuclear rank vs step",
        output_path=args.output_dir / "rf_model1_nuclear_rank_vs_step.pdf",
    )
    plot_first_step(
        dims=dims,
        nr0=nr0_m1,
        nr1=nr1_m1,
        title="Model 1 (realizable) first-step scaling",
        output_path=args.output_dir / "rf_model1_first_step_vs_dimension.pdf",
    )

    # Model 2: teacher–student
    steps_m2, curves_m2, nr0_m2, nr1_m2 = collect_nuclear_rank_curves(
        dims=dims,
        seeds=seeds,
        steps=args.steps,
        model_builder=build_teacher_student_data,
        config_template=base_config,
        device=device,
        feature_dim_fn=feature_dim_fn,
        sample_count_fn=sample_count_fn,
    )
    plot_steps(
        steps=steps_m2,
        curves=curves_m2,
        dims=dims,
        title="Model 2 (teacher–student) nuclear rank vs step",
        output_path=args.output_dir / "rf_model2_nuclear_rank_vs_step.pdf",
    )
    plot_first_step(
        dims=dims,
        nr0=nr0_m2,
        nr1=nr1_m2,
        title="Model 2 (teacher–student) first-step scaling",
        output_path=args.output_dir / "rf_model2_first_step_vs_dimension.pdf",
    )


if __name__ == "__main__":
    main()


