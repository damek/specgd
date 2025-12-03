"""
Random-feature comparison of gradient descent (GD) and spectral gradient descent (SpecGD).

This reproduces the setting from the provided figure:

    min_W L(W) = (1 / 2n) || W A - Y ||_F^2

with W ∈ ℝ^{d×d}, A = σ(VX) generated from frozen Gaussian features, and Y = W♯ A for a
Gaussian ground-truth matrix W♯. We log the objective gap, nuclear rank of the gradient,
and the stable rank of the activation matrix in order to visualize the GD vs SpecGD
trajectories (including SpecGD restarts from intermediate GD iterates).
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_DEVICE = torch.device("cpu")


@dataclass
class ExperimentConfig:
    dim: int = 100
    feature_dim: int = 50
    num_samples: int = 2*512
    num_steps: int = 1000
    base_lr: float = 1e-2
    activation: str = "relu"
    seed: int = 0
    dtype: torch.dtype = field(default=torch.float64, repr=False)


def relu_square(x: torch.Tensor) -> torch.Tensor:
    y = F.relu(x)
    return y * y


ACTIVATIONS: Dict[str, torch.nn.Module] = {
    "relu_square": relu_square,
    "relu": F.relu,
    "gelu": F.gelu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
}

ACTIVATION_CHOICES = list(ACTIVATIONS.keys()) + ["swiglu"]


class RandomFeatureModel(nn.Module):
    def __init__(self, dim: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        weight = torch.zeros(dim, dim, dtype=dtype, device=device)
        self.weight = nn.Parameter(weight)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return self.weight @ activations


def stable_rank_exact(matrix: torch.Tensor) -> Tuple[float, float]:
    mat64 = matrix.to(torch.float64)
    fro_sq = torch.sum(mat64 * mat64)
    op_norm = torch.linalg.matrix_norm(mat64, ord=2)
    sr = (fro_sq / (op_norm * op_norm)).item() if op_norm > 0 else float("inf")
    return sr, op_norm.item()


def spectral_polar_and_nuclear(grad: torch.Tensor) -> Tuple[torch.Tensor, float]:
    grad_matrix = grad if grad.ndim == 2 else grad.view(grad.shape[0], -1)
    if grad_matrix.numel() == 0:
        return torch.zeros_like(grad), 0.0

    g64 = grad_matrix.to(torch.float64)
    gram = g64 @ g64.T
    gram = 0.5 * (gram + gram.T)
    evals, evecs = torch.linalg.eigh(gram)
    evals = torch.clamp(evals, min=0.0)
    singulars = torch.sqrt(evals)
    nuclear = singulars.sum().item()
    mask = singulars > 0
    if mask.any():
        vectors = evecs[:, mask]
        inv_root = vectors / singulars[mask]
        sqrt_inv = inv_root @ vectors.T
        polar = sqrt_inv @ g64
    else:
        polar = torch.zeros_like(g64)

    polar = polar.to(dtype=grad_matrix.dtype)
    return polar, nuclear


def build_random_feature_data(
    config: ExperimentConfig,
    activation_fn,
    device: torch.device,
) -> Dict[str, torch.Tensor | float]:
    dim = config.dim
    hidden = config.feature_dim
    num_samples = config.num_samples
    dtype = config.dtype

    X = torch.randn(hidden, num_samples, device=device, dtype=dtype)
    if config.activation == "swiglu":
        V1 = torch.randn(dim, hidden, device=device, dtype=dtype) / math.sqrt(hidden)
        V2 = torch.randn(dim, hidden, device=device, dtype=dtype) / math.sqrt(hidden)
        pre1 = V1 @ X
        pre2 = V2 @ X
        A = F.silu(pre1) * pre2
    else:
        V = torch.randn(dim, hidden, device=device, dtype=dtype) / math.sqrt(hidden)
        pre_activation = V @ X
        A = activation_fn(pre_activation)

    W_sharp = torch.randn(dim, dim, device=device, dtype=dtype)/math.sqrt(dim)
    Y = W_sharp @ A

    sr_A, op_norm_A = stable_rank_exact(A)
    fro_norm_A = torch.linalg.matrix_norm(A, ord="fro").item()

    return {
        "A": A,
        "Y": Y,
        "W_sharp": W_sharp,
        "stable_rank_A": sr_A,
        "activation_op_norm": op_norm_A,
        "activation_fro_norm": fro_norm_A,
    }


def _init_results() -> Dict[str, Any]:
    return {
        "loss": {"steps": [], "values": []},
        "objective_gap": {"steps": [], "values": []},
        "grad_ratio": {"W1": {"steps": [], "values": []}},
        "grad_norm": {"steps": [], "values": []},
        "nuclear_norm": {"steps": [], "values": []},
        "nuclear_rank": {"steps": [], "values": []},
        "activations": {
            "layer1": {"steps": [], "values": []},
            "layer2": {"steps": [], "values": []},
            "layer3": {"steps": [], "values": []},
        },
        "activation_op_norm": {
            "A0": {"steps": [], "values": []},
            "A1": {"steps": [], "values": []},
            "A2": {"steps": [], "values": []},
            "A3": {"steps": [], "values": []},
        },
        "weight_op_norm": {"W1": {"steps": [], "values": []}},
        "metadata": {},
    }


def _append_activation_metrics(
    results: Dict[str, Any],
    step: int,
    stable_rank_A: float,
    op_norm_A: float,
) -> None:
    results["activations"]["layer1"]["steps"].append(step)
    results["activations"]["layer1"]["values"].append(stable_rank_A)
    results["activation_op_norm"]["A1"]["steps"].append(step)
    results["activation_op_norm"]["A1"]["values"].append(op_norm_A)


def _fmt(value: float) -> str:
    if isinstance(value, float) and math.isfinite(value):
        return f"{value:.4g}"
    return str(value)


def train_method(
    method_name: str,
    update_kind: str,
    model: RandomFeatureModel,
    activations: torch.Tensor,
    targets: torch.Tensor,
    lr: float,
    iteration_range: Iterable[int],
    stable_rank_A: float,
    activation_op_norm: float,
    progress: bool,
    checkpoint_hook: Optional[Callable[[int, torch.Tensor], None]] = None,
) -> Dict[str, Any]:
    results = _init_results()
    results["metadata"]["update_kind"] = update_kind
    results["metadata"]["lr"] = lr
    range_list = list(iteration_range)
    if not range_list:
        return results
    results["metadata"]["start_iter"] = int(range_list[0])
    results["metadata"]["stop_iter"] = int(range_list[-1])

    n = activations.shape[1]

    for step in range_list:
        if checkpoint_hook is not None:
            checkpoint_hook(step, model.weight.detach())
        outputs = model(activations)
        diff = outputs - targets
        loss = (.5/n)* torch.linalg.norm(diff, ord='fro')**2
        loss.backward()

        grad = model.weight.grad
        if grad is None:
            raise RuntimeError("Gradient missing for model weight")
        grad_detached = grad.detach()
        fro_sq = torch.sum(grad_detached * grad_detached).item()
        polar, nuclear = spectral_polar_and_nuclear(grad_detached)
        if fro_sq == 0.0:
            nuclear_rank = float("inf") if nuclear > 0 else 0.0
        else:
            nuclear_rank = (nuclear * nuclear) / fro_sq

        results["loss"]["steps"].append(step)
        results["loss"]["values"].append(loss.item())
        results["objective_gap"]["steps"].append(step)
        results["objective_gap"]["values"].append(loss.item())
        results["grad_ratio"]["W1"]["steps"].append(step)
        results["grad_ratio"]["W1"]["values"].append(nuclear_rank)
        results["grad_norm"]["steps"].append(step)
        results["grad_norm"]["values"].append(math.sqrt(fro_sq))
        results["nuclear_norm"]["steps"].append(step)
        results["nuclear_norm"]["values"].append(nuclear)
        results["nuclear_rank"]["steps"].append(step)
        results["nuclear_rank"]["values"].append(nuclear_rank)
        results["weight_op_norm"]["W1"]["steps"].append(step)
        weight_op = torch.linalg.matrix_norm(model.weight.detach(), ord=2).item()
        results["weight_op_norm"]["W1"]["values"].append(weight_op)

        _append_activation_metrics(results, step, stable_rank_A, activation_op_norm)

        with torch.no_grad():
            if update_kind == "specgd":
                if nuclear > 0.0:
                    direction = polar.view_as(model.weight)
                    model.weight.add_(direction, alpha=-lr * nuclear)
            else:
                model.weight.add_(grad_detached, alpha=-lr)

        model.zero_grad(set_to_none=True)

        if progress and step % 50 == 0:
            print(
                f"[{method_name}] step {step} "
                f"loss={_fmt(loss.item())} "
                f"nr={_fmt(nuclear_rank)} "
                f"||grad||_*={_fmt(nuclear)} "
                f"||grad||_F={_fmt(math.sqrt(fro_sq))} "
                f"||W||_op={_fmt(weight_op)}"
            )

    return results


def run_gradient_descent(
    config: ExperimentConfig,
    data: Dict[str, torch.Tensor | float],
    lr: float,
    progress: bool,
    checkpoint_iters: Iterable[int],
) -> Tuple[Dict[str, Any], Dict[int, torch.Tensor]]:
    model = RandomFeatureModel(config.dim, config.dtype, DEFAULT_DEVICE)
    checkpoints: Dict[int, torch.Tensor] = {}
    checkpoint_set = set(checkpoint_iters)

    def hook(step: int, weight: torch.Tensor) -> None:
        if step in checkpoint_set and step not in checkpoints:
            checkpoints[step] = weight.detach().clone()

    iteration_range = range(0, config.num_steps)
    results = train_method(
        method_name="gradient_descent",
        update_kind="gd",
        model=model,
        activations=data["A"],
        targets=data["Y"],
        lr=lr,
        iteration_range=iteration_range,
        stable_rank_A=float(data["stable_rank_A"]),
        activation_op_norm=float(data["activation_op_norm"]),
        progress=progress,
        checkpoint_hook=hook if checkpoint_set else None,
    )

    return results, checkpoints


def compute_gd_checkpoint_at_iter(
    config: ExperimentConfig,
    data: Dict[str, torch.Tensor | float],
    lr: float,
    target_iter: int,
) -> torch.Tensor:
    """Recompute the GD weight at the start of `target_iter`."""
    model = RandomFeatureModel(config.dim, config.dtype, DEFAULT_DEVICE)
    n = data["A"].shape[1]
    for step in range(target_iter):
        outputs = model(data["A"])
        diff = outputs - data["Y"]
        loss = (.5/n)* torch.linalg.norm(diff, ord='fro')**2
        loss.backward()
        grad = model.weight.grad
        assert grad is not None
        with torch.no_grad():
            model.weight.add_(grad.detach(), alpha=-lr)
        model.zero_grad(set_to_none=True)
    return model.weight.detach().clone()


def run_specgd_family(
    config: ExperimentConfig,
    data: Dict[str, torch.Tensor | float],
    base_lr: float,
    progress: bool,
    checkpoints: Dict[int, torch.Tensor],
    restart_iters: Iterable[int],
) -> Dict[str, Dict[str, Any]]:
    methods: Dict[str, Dict[str, Any]] = {}

    # Baseline SpecGD from the zero initialization
    model = RandomFeatureModel(config.dim, config.dtype, DEFAULT_DEVICE)
    spec_results = train_method(
        method_name="specgd",
        update_kind="specgd",
        model=model,
        activations=data["A"],
        targets=data["Y"],
        lr=base_lr,
        iteration_range=range(0, config.num_steps),
        stable_rank_A=float(data["stable_rank_A"]),
        activation_op_norm=float(data["activation_op_norm"]),
        progress=progress,
    )
    methods["specgd"] = spec_results

    for restart_iter in restart_iters:
        if restart_iter <= 0:
            continue
        if restart_iter >= config.num_steps:
            continue
        if restart_iter not in checkpoints:
            print(f"[specgd] warning: no GD checkpoint for iter {restart_iter}, skipping restart")
            continue
        model_restart = RandomFeatureModel(config.dim, config.dtype, DEFAULT_DEVICE)
        with torch.no_grad():
            model_restart.weight.copy_(checkpoints[restart_iter])
        results = train_method(
            method_name=f"specgd_from_iter_{restart_iter}",
            update_kind="specgd",
            model=model_restart,
            activations=data["A"],
            targets=data["Y"],
            lr=base_lr,
            iteration_range=range(restart_iter, config.num_steps),
            stable_rank_A=float(data["stable_rank_A"]),
            activation_op_norm=float(data["activation_op_norm"]),
            progress=progress,
        )
        results["metadata"]["start_iter"] = restart_iter
        methods[f"specgd_from_iter_{restart_iter}"] = results

    return methods


def main() -> None:
    parser = argparse.ArgumentParser(description="Random feature GD vs SpecGD experiment")
    parser.add_argument("--dim", type=int, default=100, help="Matrix dimension for W and the activations")
    parser.add_argument("--feature-dim", type=int, default=50, help="Hidden dimension used to form σ(VX)")
    parser.add_argument("--num-samples", type=int, default=512, help="Number of random feature columns (n)")
    parser.add_argument("--activation", default="relu", choices=ACTIVATION_CHOICES)
    parser.add_argument("--output-dir", type=Path, default=Path("logs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None, help="Override number of training iterations")
    parser.add_argument("--progress", action="store_true", help="Print diagnostics every 50 iterations")
    parser.add_argument("--plot", action="store_true", help="Generate the summary figure after the run")
    parser.add_argument("--plot-dpi", type=int, default=300)
    parser.add_argument(
        "--specgd-restart-iters",
        type=int,
        nargs="*",
        default=None,
        help="Explicit GD iteration indices used to initialize additional SpecGD trajectories",
    )
    parser.add_argument(
        "--specgd-restart-frequency",
        type=int,
        default=100,
        help="If --specgd-restart-iters is omitted, spawn SpecGD restarts every this many GD steps",
    )
    parser.add_argument(
        "--method-lr",
        action="append",
        default=[],
        help="Override learning rate per method, e.g., --method-lr gradient_descent=0.001",
    )
    parser.add_argument(
        "--specgd-from-peak",
        action="store_true",
        help="Start a SpecGD run from the GD iteration where the nuclear rank peaks",
    )
    args = parser.parse_args()

    config = ExperimentConfig(
        dim=args.dim,
        feature_dim=args.feature_dim,
        num_samples=args.num_samples,
        activation=args.activation,
        seed=args.seed,
    )
    if args.steps is not None:
        config.num_steps = args.steps

    torch.manual_seed(config.seed)

    activation_fn = ACTIVATIONS.get(config.activation)
    data = build_random_feature_data(config, activation_fn, DEFAULT_DEVICE)

    methods: Dict[str, Dict[str, Any]] = {}

    act_op = float(data["activation_op_norm"])
    act_fro = float(data["activation_fro_norm"])
    n = config.num_samples
    default_gd_lr = n / (act_op * act_op) if act_op > 0 else config.base_lr
    default_spec_lr = n / (act_fro * act_fro) if act_fro > 0 else config.base_lr
    # default_spec_lr = 1/(1+ 1/default_spec_lr)
    # default_gd_lr = 1/(1+ 1/default_gd_lr)


    method_lrs: Dict[str, float] = {
        "gradient_descent": default_gd_lr,
        "specgd": default_spec_lr,
    }
    for override in args.method_lr:
        if "=" not in override:
            raise ValueError(f"Invalid --method-lr value '{override}'; expected format method=lr")
        method_key, value = override.split("=", 1)
        try:
            method_lrs[method_key] = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid learning rate '{value}' for method '{method_key}'") from exc

    if args.specgd_restart_iters:
        restart_iters = sorted(set(i for i in args.specgd_restart_iters if i >= 0))
    else:
        freq = max(1, args.specgd_restart_frequency)
        restart_iters = list(range(freq, config.num_steps, freq))

    gd_results, gd_checkpoints = run_gradient_descent(
        config=config,
        data=data,
        lr=method_lrs["gradient_descent"],
        progress=args.progress,
        checkpoint_iters=restart_iters,
    )
    methods["gradient_descent"] = gd_results

    spec_methods = run_specgd_family(
        config=config,
        data=data,
        base_lr=method_lrs["specgd"],
        progress=args.progress,
        checkpoints=gd_checkpoints,
        restart_iters=restart_iters,
    )
    methods.update(spec_methods)

    if args.specgd_from_peak:
        nuclear_series = gd_results.get("nuclear_rank")
        if nuclear_series and nuclear_series["values"]:
            values = nuclear_series["values"]
            steps = nuclear_series["steps"]
            peak_idx = max(range(len(values)), key=lambda i: values[i])
            peak_iter = int(steps[peak_idx])
            if peak_iter >= config.num_steps:
                peak_iter = config.num_steps - 1
            if peak_iter < 0:
                peak_iter = 0
            print(f"[specgd_from_peak] peak nuclear rank at iter {peak_iter} (value={values[peak_idx]:.4g})")
            if peak_iter not in gd_checkpoints:
                checkpoint = compute_gd_checkpoint_at_iter(config, data, method_lrs["gradient_descent"], peak_iter)
                gd_checkpoints[peak_iter] = checkpoint
            muon_model = RandomFeatureModel(config.dim, config.dtype, DEFAULT_DEVICE)
            with torch.no_grad():
                muon_model.weight.copy_(gd_checkpoints[peak_iter])
            muon_results = train_method(
                method_name="specgd_from_peak",
                update_kind="specgd",
                model=muon_model,
                activations=data["A"],
                targets=data["Y"],
                lr=method_lrs["specgd"],
                iteration_range=range(peak_iter, config.num_steps),
                stable_rank_A=float(data["stable_rank_A"]),
                activation_op_norm=float(data["activation_op_norm"]),
                progress=args.progress,
            )
            muon_results["metadata"]["start_iter"] = peak_iter
            muon_results["metadata"]["source"] = "nuclear_rank_peak"
            methods["specgd_from_peak"] = muon_results


    run_id = str(uuid.uuid4())
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = asdict(config)
    config_payload["dtype"] = str(config.dtype)

    payload = {
        "run_id": run_id,
        "config": config_payload,
        "dataset": {
            "stable_rank_A": data["stable_rank_A"],
            "activation_op_norm": data["activation_op_norm"],
            "activation_fro_norm": data["activation_fro_norm"],
        },
        "methods": methods,
    }

    output_path = output_dir / f"{run_id}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved results to {output_path}")

    if args.plot:
        from plot_random_feature_specgd import plot_random_feature_summary

        base_path = output_path.parent / output_path.stem
        plot_random_feature_summary(payload, base_path, args.plot_dpi)


if __name__ == "__main__":
    main()


