"""Compare gradient descent and weighted spectral descent on a simple polynomial task.

To reproduce the ReLU-style plots from the paper, run:

    python experiments/weighted_spectral_vs_gd.py \\
        --plot \\
        --method-lr gradient_descent=.081 \\
        --method-lr block_specgd=0.1 \\
        --progress \\
        --steps 400 \\
        --activation relu_square

For SwigLU with a larger batch size (8196), one useful command is:

    python experiments/weighted_spectral_vs_gd.py \\
        --plot \\
        --method-lr gradient_descent=.0035 \\
        --method-lr block_specgd=0.0025 \\
        --progress \\
        --steps 400 \\
        --activation swiglu \\
        --batch-size 8196

Another SwigLU configuration (batch size 512) to try:

    python experiments/weighted_spectral_vs_gd.py \\
        --plot \\
        --method-lr gradient_descent=.01 \\
        --method-lr block_specgd=0.025 \\
        --progress \\
        --steps 400 \\
        --activation swiglu \\
        --batch-size 512

This script trains a four-layer, bias-free neural network on the task of
regressing the cubic monomial y = x_1 x_2 x_3 over Gaussian data. Two update
rules run by default:

1. Pure gradient descent (all layers updated with vanilla GD).
2. SpecGD (layer 1 GD, layers 2-3 spectral, layer 4 GD).

Optionally, SpecGD (layers 1-3 spectral, final layer GD) can be enabled via
CLI.

Metrics collected per iteration:

* Training MSE.
* Stable rank of intermediate activations A_i = σ(W_i A_{i-1}).
* Gradient nuclear/Frobenius norm ratios for each weight matrix.

Results are serialized to JSON (logs/<uuid>.json) for downstream plotting.
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
import math
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_DEVICE = torch.device("cpu")


@dataclass
class ExperimentConfig:
    mult = 1
    dim: int = mult*128
    batch_size: int = mult*512
    num_steps: int = 2000
    base_lr: float = 1e-2
    activation: str = "relu_square"
    seed: int = 42
    dtype: torch.dtype = field(default=torch.float64, repr=False)


def relu_square(x: torch.Tensor) -> torch.Tensor:
    y = F.relu(x)
    return y * y


ACTIVATIONS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu_square": relu_square,
    "relu": F.relu,
    "gelu": F.gelu,
    "sigmoid": F.sigmoid,
    "tanh": F.tanh,
}

ACTIVATION_CHOICES = list(ACTIVATIONS.keys()) + ["swiglu"]


class FourLayerNet(torch.nn.Module):
    def __init__(self, config: ExperimentConfig, activation_name: str):
        super().__init__()
        self.activation_name = activation_name
        self.activation = ACTIVATIONS.get(activation_name)
        self.use_swiglu = activation_name == "swiglu"
        d = config.dim
        # h = 4 * d
        dtype = config.dtype
        hidden = 4 * d
        mult = 2 if self.use_swiglu else 1
        self.layer1 = nn.Linear(d, hidden * mult, bias=False, dtype=dtype)      # d -> 4d
        self.layer2 = nn.Linear(hidden, hidden * mult, bias=False, dtype=dtype)      # 4d -> 4d
        self.layer3 = nn.Linear(hidden, hidden * mult, bias=False, dtype=dtype)      # 4d -> 4d
        self.layer4 = nn.Linear(hidden, 1, bias=False, dtype=dtype)      # 4d -> 1


    def _apply_activation(self, linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        z = linear(x)
        if self.use_swiglu:
            z1, z2 = torch.chunk(z, 2, dim=1)
            return F.silu(z1) * z2
        if self.activation is None:
            raise ValueError(f"Activation '{self.activation_name}' has no callable.")
        return self.activation(z)

    def forward_with_activations(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        a0 = x
        a1 = self._apply_activation(self.layer1, a0)
        a2 = self._apply_activation(self.layer2, a1)
        a3 = self._apply_activation(self.layer3, a2)
        out = self.layer4(a3)
        return out, [a1, a2, a3]

    def weight_map(self) -> Dict[str, torch.nn.Parameter]:
        return {
            "W1": self.layer1.weight,
            "W2": self.layer2.weight,
            "W3": self.layer3.weight,
            "W4": self.layer4.weight,
        }


def stable_rank_upper_bound(matrix: torch.Tensor) -> float:
    cols = matrix.transpose(0, 1)
    col_norm_sq = torch.sum(cols * cols, dim=1)
    numerator = torch.mean(col_norm_sq)
    mean_column = torch.mean(cols, dim=0)
    denominator = torch.sum(mean_column * mean_column)
    return (numerator / denominator).item()


def stable_rank_exact(matrix: torch.Tensor) -> tuple[float, float]:
    mat64 = matrix.detach().to(torch.float64)
    fro_sq = torch.sum(mat64 * mat64)
    op_norm = torch.linalg.norm(mat64, ord=2)
    sr = (fro_sq / (op_norm * op_norm)).item()
    return sr, op_norm.item()


def spectral_polar_and_nuclear(grad: torch.Tensor) -> tuple[torch.Tensor, float]:
    flattened = False
    if grad.ndim == 1:
        grad_matrix = grad.unsqueeze(0)
        flattened = True
    else:
        grad_matrix = grad

    if grad_matrix.numel() == 0:
        zero = torch.zeros_like(grad_matrix)
        return (zero.squeeze(0) if flattened else zero, 0.0)

    g64 = grad_matrix.to(torch.float64)
    m, n = g64.shape
    if m <= n:
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
    else:
        gram = g64.T @ g64
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
            polar = g64 @ sqrt_inv
        else:
            polar = torch.zeros_like(g64)

    polar = polar.to(dtype=grad_matrix.dtype)
    if flattened:
        polar = polar.squeeze(0)
    return polar, nuclear


def clone_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def apply_state_dict(module: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            param.copy_(state_dict[name])


def build_dataset(config: ExperimentConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(config.batch_size, config.dim, device=device, dtype=config.dtype)
    y_scalar = x[:, 0] * x[:, 1] * x[:, 2]
    y = y_scalar.unsqueeze(1)
    return x, y


def _fmt(value: float) -> str:
    if isinstance(value, float) and math.isfinite(value):
        return f"{value:.4g}"
    return str(value)


def _operator_norm(matrix: torch.Tensor) -> float:
    mat = matrix.detach()
    if mat.ndim == 1:
        return torch.linalg.vector_norm(mat, ord=2).item()
    return torch.linalg.matrix_norm(mat, ord=2).item()


def _fro_norm(matrix: torch.Tensor) -> float:
    return torch.linalg.vector_norm(matrix.detach().reshape(-1), ord=2).item()


class SpectrumRecorder:
    def __init__(self, interval: int):
        self.interval = max(1, interval)
        self.layers = ["layer1", "layer2", "layer3"]
        self.frames: List[Dict[str, Any]] = []

    def maybe_record(
        self,
        step: int,
        loss: float,
        sr_snapshot: Dict[str, float],
        activations: List[torch.Tensor],
    ) -> None:
        if step % self.interval != 0:
            return
        stable_rank = {}
        for idx, layer_name in enumerate(self.layers, start=1):
            src_key = f"layer{idx}"
            if src_key in sr_snapshot:
                stable_rank[layer_name] = float(sr_snapshot[src_key])
        spectra = {}
        for idx, layer_name in enumerate(self.layers):
            if idx >= len(activations):
                break
            act = activations[idx].detach()
            try:
                singulars = torch.linalg.svdvals(act.to(torch.float32))
            except RuntimeError:
                singulars = torch.linalg.svdvals(act.to(torch.float32).cpu())
            spectra[layer_name] = singulars.cpu().tolist()
        self.frames.append(
            {
                "step": int(step),
                "loss": float(loss),
                "stable_rank": stable_rank,
                "spectra": spectra,
            }
        )


def train_method(
    method_name: str,
    model: FourLayerNet,
    config: ExperimentConfig,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    spectral_layers: Tuple[str, ...],
    progress: bool,
    lr: float,
    exact_sr: bool,
    spectrum_recorder: SpectrumRecorder | None = None,
) -> Dict[str, Any]:
    results = {
        "loss": {"steps": [], "values": []},
        "activations": {
            "layer1": {"steps": [], "values": []},
            "layer2": {"steps": [], "values": []},
            "layer3": {"steps": [], "values": []},
        },
        "grad_ratio": {
            "W1": {"steps": [], "values": []},
            "W2": {"steps": [], "values": []},
            "W3": {"steps": [], "values": []},
        },
        "grad_scale": {
            "W1": {"steps": [], "values": []},
            "W2": {"steps": [], "values": []},
            "W3": {"steps": [], "values": []},
            "W4": {"steps": [], "values": []},
        },
        "activation_op_norm": {
            "A0": {"steps": [], "values": []},
            "A1": {"steps": [], "values": []},
            "A2": {"steps": [], "values": []},
            "A3": {"steps": [], "values": []},
        },
        "weight_op_norm": {
            "W1": {"steps": [], "values": []},
            "W2": {"steps": [], "values": []},
            "W3": {"steps": [], "values": []},
            "W4": {"steps": [], "values": []},
        },
        "spectral_layers": list(spectral_layers),
        "lr": lr,
    }

    param_dict = model.weight_map()
    layer_order = ["W1", "W2", "W3", "W4"]

    # Initial diagnostics before any updates
    model.zero_grad(set_to_none=True)
    init_outputs, init_activations = model.forward_with_activations(inputs)
    init_loss = F.mse_loss(init_outputs, targets)

    op_norm_cache_init: Dict[str, float] = {}
    sr_snapshot_init: Dict[str, float] = {}
    if exact_sr:
        for idx, act in enumerate(init_activations, start=1):
            sr, op_norm = stable_rank_exact(act)
            op_norm_cache_init[f"A{idx}"] = op_norm
            sr_snapshot_init[f"layer{idx}"] = sr
    else:
        for idx, act in enumerate(init_activations, start=1):
            sr_snapshot_init[f"layer{idx}"] = stable_rank_upper_bound(act.detach())

    activation_mats_init = [inputs] + init_activations
    activation_norms_init: Dict[str, float] = {}
    for idx, mat in enumerate(activation_mats_init):
        layer_name = layer_order[idx] if idx < len(layer_order) else None
        use_fro = False
        if layer_name is not None:
            if method_name == "specgd" and layer_name in spectral_layers:
                use_fro = True
            elif method_name == "block_specgd" and layer_name in spectral_layers:
                use_fro = True
        if exact_sr and f"A{idx}" in op_norm_cache_init and not use_fro:
            activation_norms_init[f"A{idx}"] = op_norm_cache_init[f"A{idx}"]
        elif use_fro:
            activation_norms_init[f"A{idx}"] = _fro_norm(mat)
        else:
            activation_norms_init[f"A{idx}"] = _operator_norm(mat)

    grad_scales_init: Dict[str, float] = {}
    for idx, name in enumerate(layer_order):
        act_norm = activation_norms_init[f"A{idx}"]
        base_scale = 1.0 / act_norm
        grad_scales_init[name] = config.batch_size * base_scale * base_scale

    init_loss.backward()

    grad_snapshot_init: Dict[str, float] = {}
    weight_norms_init: Dict[str, float] = {}
    with torch.no_grad():
        for name, param in param_dict.items():
            grad = param.grad
            if grad is None:
                continue
            grad_detached = grad.detach()
            fro_sq = torch.sum(grad_detached * grad_detached)
            _, nuclear = spectral_polar_and_nuclear(grad_detached)
            if name != "W4":
                fro_val = fro_sq.item()
                if fro_val == 0.0:
                    ratio = float("inf") if nuclear > 0.0 else 0.0
                else:
                    ratio = (nuclear * nuclear) / fro_val
                grad_snapshot_init[name] = ratio
            else:
                grad_snapshot_init[name] = 1.0
            weight_norms_init[name] = torch.linalg.matrix_norm(param, ord=2).item()

    model.zero_grad()

    if spectrum_recorder is not None:
        spectrum_recorder.maybe_record(
            step=0,
            loss=init_loss.item(),
            sr_snapshot=sr_snapshot_init,
            activations=init_activations,
        )

    if progress:
        activ_str_init = ", ".join(
            f"layer{i}:{_fmt(sr_snapshot_init.get(f'layer{i}', float('nan')))}" for i in range(1, 4)
        )
        grad_str_init = ", ".join(
            f"{name}:{_fmt(grad_snapshot_init.get(name, float('nan')))}" for name in ("W1", "W2", "W3", "W4")
        )
        scale_str_init = ", ".join(
            f"{name}:{_fmt(grad_scales_init.get(name, float('nan')))}" for name in ("W1", "W2", "W3", "W4")
        )
        actnorm_str_init = ", ".join(
            f"A{idx}:{_fmt(activation_norms_init.get(f'A{idx}', float('nan')))}" for idx in range(4)
        )
        weight_str_init = ", ".join(
            f"{name}:{_fmt(weight_norms_init.get(name, float('nan')))}" for name in ("W1", "W2", "W3", "W4")
        )
        print(
            f"[{method_name}] init lr={_fmt(lr)} "
            f"loss={_fmt(init_loss.item())} activations=[{activ_str_init}] gradients=[{grad_str_init}] "
            f"scales=[{scale_str_init}] act_norms=[{actnorm_str_init}] weights=[{weight_str_init}]"
        )

    for step in range(config.num_steps):
        outputs, activations = model.forward_with_activations(inputs)
        loss = F.mse_loss(outputs, targets)
        results["loss"]["steps"].append(step)
        results["loss"]["values"].append(loss.item())

        sr_snapshot: Dict[str, float] = {}
        grad_scales: Dict[str, float] = {}
        with torch.no_grad():
            op_norm_cache: Dict[str, float] = {}
            for idx, act in enumerate(activations, start=1):
                if exact_sr:
                    sr, op_norm = stable_rank_exact(act)
                    op_norm_cache[f"A{idx}"] = op_norm
                else:
                    sr = stable_rank_upper_bound(act.detach())
                activ_key = f"layer{idx}"
                results["activations"][activ_key]["steps"].append(step)
                results["activations"][activ_key]["values"].append(sr)
                sr_snapshot[activ_key] = sr

            # compute activation and weight operator norms for scaling
            activation_mats = [inputs] + activations
            activation_norms: Dict[str, float] = {}
            for idx, mat in enumerate(activation_mats):
                layer_name = layer_order[idx] if idx < len(layer_order) else None
                use_fro = False
                if layer_name is not None:
                    if method_name == "specgd" and layer_name in spectral_layers:
                        use_fro = True
                    elif method_name == "block_specgd" and layer_name in spectral_layers:
                        use_fro = True
                if exact_sr and f"A{idx}" in op_norm_cache and not use_fro:
                    activation_norms[f"A{idx}"] = op_norm_cache[f"A{idx}"]
                elif use_fro:
                    activation_norms[f"A{idx}"] = _fro_norm(mat)
                else:
                    activation_norms[f"A{idx}"] = _operator_norm(mat)
                key = f"A{idx}"
                results["activation_op_norm"][key]["steps"].append(step)
                results["activation_op_norm"][key]["values"].append(activation_norms[key])
            for idx, name in enumerate(layer_order):
                act_norm = activation_norms[f"A{idx}"]
                base_scale = 1.0 / act_norm
                grad_scales[name] = config.batch_size * base_scale * base_scale

        loss.backward()

        grad_snapshot: Dict[str, float] = {}
        with torch.no_grad():
            for name, param in param_dict.items():
                grad = param.grad
                if grad is None:
                    continue
                grad_detached = grad.detach()
                scale = grad_scales[name]
                results["grad_scale"][name]["steps"].append(step)
                results["grad_scale"][name]["values"].append(scale)
                fro_sq = torch.sum(grad_detached * grad_detached)
                direction, nuclear = spectral_polar_and_nuclear(grad_detached)
                if name != "W4":
                    fro_val = fro_sq.item()
                    if fro_val == 0.0:
                        ratio = float("inf") if nuclear > 0.0 else 0.0
                    else:
                        ratio = (nuclear * nuclear) / fro_val
                    results["grad_ratio"][name]["steps"].append(step)
                    results["grad_ratio"][name]["values"].append(ratio)
                    grad_snapshot[name] = ratio
                else:
                    grad_snapshot[name] = 1.0

                if name in spectral_layers and nuclear > 0.0:
                    direction = direction.view_as(grad)
                    param.add_(direction, alpha=-lr * nuclear * scale)
                else:
                    param.add_(grad_detached, alpha=-lr * scale)

                weight_norm = torch.linalg.matrix_norm(param, ord=2).item()
                results["weight_op_norm"][name]["steps"].append(step)
                results["weight_op_norm"][name]["values"].append(weight_norm)

        model.zero_grad()

        if spectrum_recorder is not None:
            spectrum_recorder.maybe_record(step + 1, loss.item(), sr_snapshot, activations)

        if progress and step % 10 == 0:
            activ_str = ", ".join(
                f"{layer}:{_fmt(sr_snapshot.get(layer, float('nan')))}"
                for layer in ("layer1", "layer2", "layer3")
            )
            grad_str = ", ".join(
                f"{name}:{_fmt(grad_snapshot.get(name, float('nan')))}"
                for name in ("W1", "W2", "W3", "W4")
            )
            scale_str = ", ".join(
                f"{name}:{_fmt(grad_scales[name])}" for name in ("W1", "W2", "W3", "W4")
            )
            actnorm_str = ", ".join(
                f"A{idx}:{_fmt(results['activation_op_norm'][f'A{idx}']['values'][-1])}" for idx in range(4)
            )
            weight_str = ", ".join(
                f"{name}:{_fmt(results['weight_op_norm'][name]['values'][-1])}"
                for name in ("W1", "W2", "W3", "W4")
            )
            print(
                f"[{method_name}] step {step}/{config.num_steps} lr={_fmt(lr)} "
                f"loss={_fmt(loss.item())} activations=[{activ_str}] gradients=[{grad_str}] scales=[{scale_str}] act_norms=[{actnorm_str}] weights=[{weight_str}]"
            )
    return results


def plot_summary(
    methods: Dict[str, Dict[str, Dict[str, List[float]]]],
    base_path: Path,
    dpi: int,
    method_styles: Dict[str, Dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    specs = [
        ("Training loss", "MSE", lambda payload: payload["loss"]),
        ("Post-activation stable rank layer1", "Stable rank", lambda payload: payload["activations"]["layer1"]),
        ("Post-activation stable rank layer2", "Stable rank", lambda payload: payload["activations"]["layer2"]),
        ("Post-activation stable rank layer3", "Stable rank", lambda payload: payload["activations"]["layer3"]),
        ("Gradient ratio W1", r"$\|\nabla L\|_*^2 / \|\nabla L\|_F^2$", lambda payload: payload["grad_ratio"]["W1"]),
        ("Gradient ratio W2", r"$\|\nabla L\|_*^2 / \|\nabla L\|_F^2$", lambda payload: payload["grad_ratio"]["W2"]),
        ("Gradient ratio W3", r"$\|\nabla L\|_*^2 / \|\nabla L\|_F^2$", lambda payload: payload["grad_ratio"]["W3"]),
        ("Activation op norm A0", r"$\|A_0\|_{op}$", lambda payload: payload.get("activation_op_norm", {}).get("A0", {"steps": [], "values": []})),
        ("Activation op norm A1", r"$\|A_1\|_{op}$", lambda payload: payload.get("activation_op_norm", {}).get("A1", {"steps": [], "values": []})),
        ("Activation op norm A2", r"$\|A_2\|_{op}$", lambda payload: payload.get("activation_op_norm", {}).get("A2", {"steps": [], "values": []})),
        ("Activation op norm A3", r"$\|A_3\|_{op}$", lambda payload: payload.get("activation_op_norm", {}).get("A3", {"steps": [], "values": []})),
        ("Weight op norm W1", r"$\|W_1\|_{op}$", lambda payload: payload.get("weight_op_norm", {}).get("W1", {"steps": [], "values": []})),
        ("Weight op norm W2", r"$\|W_2\|_{op}$", lambda payload: payload.get("weight_op_norm", {}).get("W2", {"steps": [], "values": []})),
        ("Weight op norm W3", r"$\|W_3\|_{op}$", lambda payload: payload.get("weight_op_norm", {}).get("W3", {"steps": [], "values": []})),
        ("Weight op norm W4", r"$\|W_4\|_{op}$", lambda payload: payload.get("weight_op_norm", {}).get("W4", {"steps": [], "values": []})),
    ]

    cols = 3
    rows = math.ceil(len(specs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.0))
    axes_flat = axes.flatten()

    for idx, (title, ylabel, accessor) in enumerate(specs):
        ax = axes_flat[idx]
        for method_name, payload in methods.items():
            style = method_styles.get(method_name)
            if style is None:
                continue
            series = accessor(payload)
            if not series["steps"]:
                continue
            if idx == 0:
                ax.semilogy(
                    series["steps"],
                    series["values"],
                    label=style["label"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.6,
                )
            else:
                ax.plot(
                    series["steps"],
                    series["values"],
                    label=style["label"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.6,
                )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)

    for idx in range(len(specs), len(axes_flat)):
        axes_flat[idx].axis("off")

    for ax in axes_flat[-cols:]:
        ax.set_xlabel("Training iteration")

    legend_handles = [
        Line2D([0], [0], color=style["color"], linestyle=style["linestyle"], linewidth=1.8, label=style["label"])
        for name, style in method_styles.items()
        if name in methods
    ]
    if legend_handles:
        fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=len(legend_handles))

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    summary_path = base_path.parent / f"{base_path.name}_summary.pdf"
    fig.savefig(summary_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary plot to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GD and spectral descent on a toy problem")
    parser.add_argument("--activation", default="relu_square", choices=ACTIVATION_CHOICES)
    parser.add_argument("--output-dir", type=Path, default=Path("logs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size (default mult*512)")
    parser.add_argument("--plot", action="store_true", help="Generate plots after the run")
    parser.add_argument("--plot-dpi", type=int, default=300, help="Resolution for generated plots")
    parser.add_argument("--progress", action="store_true", help="Print training progress every 100 steps")
    parser.add_argument(
        "--spectrum-gif",
        type=int,
        default=None,
        metavar="K",
        help="Capture activation spectra every K iterations and build a GIF",
    )
    parser.add_argument(
        "--spectrum-gif-output",
        type=Path,
        default=None,
        help="Directory for spectrum GIF assets (default: same as --output-dir)",
    )
    parser.add_argument("--spectrum-gif-name", type=str, default=None, help="Filename for the spectrum GIF")
    parser.add_argument("--spectrum-gif-fps", type=float, default=1.0, help="GIF playback rate")
    parser.add_argument("--spectrum-gif-dpi", type=int, default=200, help="DPI for spectrum GIF frames")
    parser.add_argument("--spectrum-gif-keep-frames", action="store_true", help="Retain spectrum GIF PNG frames")
    parser.add_argument(
        "--method-lr",
        action="append",
        default=[],
        help="Override learning rate per method, e.g., --method-lr gradient_descent=0.001",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override number of training iterations")
    parser.add_argument(
        "--no-exact-stable-rank",
        action="store_false",
        dest="exact_stable_rank",
        help="Disable exact stable-rank computation and fall back to the upper-bound estimator",
    )
    parser.set_defaults(exact_stable_rank=True)
    args = parser.parse_args()

    config = ExperimentConfig(activation=args.activation, seed=args.seed)
    if args.steps is not None:
        config.num_steps = args.steps
    torch.manual_seed(config.seed)

    device = DEFAULT_DEVICE

    if args.batch_size is not None:
        config.batch_size = args.batch_size
    dataset_inputs, dataset_targets = build_dataset(config, device)

    base_model = FourLayerNet(config, config.activation).to(device=device, dtype=config.dtype)
    initial_state = clone_state_dict(base_model)

    methods: Dict[str, Tuple[str, ...]] = {
        "gradient_descent": tuple(),
        "specgd": ("W2", "W3"),
    }

    config_payload = asdict(config)
    config_payload["dtype"] = str(config.dtype)
    all_results = {
        "run_id": str(uuid.uuid4()),
        "config": config_payload,
        "methods": {},
    }

    method_lrs: Dict[str, float] = {name: config.base_lr for name in methods}
    for override in args.method_lr:
        if "=" not in override:
            raise ValueError(f"Invalid --method-lr value '{override}'; expected format method=lr")
        method_key, value = override.split("=", 1)
        if method_key not in methods:
            raise ValueError(f"Unknown method '{method_key}' for --method-lr override")
        try:
            method_lrs[method_key] = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid learning rate '{value}' for method '{method_key}'") from exc

    spectrum_payloads: Dict[str, Dict[str, Any]] = {}

    for method_name, spectral_layers in methods.items():
        recorder: SpectrumRecorder | None = None
        if args.spectrum_gif:
            recorder = SpectrumRecorder(args.spectrum_gif)
        model = FourLayerNet(config, config.activation).to(device=device, dtype=config.dtype)
        apply_state_dict(model, initial_state)
        method_results = train_method(
            method_name,
            model,
            config,
            dataset_inputs,
            dataset_targets,
            spectral_layers,
            args.progress,
            method_lrs[method_name],
            args.exact_stable_rank,
            spectrum_recorder=recorder,
        )
        if recorder is not None:
            spectrum_payload = {
                "method": method_name,
                "interval": recorder.interval,
                "layers": recorder.layers,
                "frames": recorder.frames,
                "total_steps": config.num_steps,
            }
            method_results["spectrum_gif"] = spectrum_payload
            spectrum_payloads[method_name] = spectrum_payload
            all_results.setdefault("spectrum_gif", {})[method_name] = spectrum_payload
        all_results["methods"][method_name] = method_results

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{all_results['run_id']}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved results to {output_path}")

    if args.plot:
        from plot_spectral_vs_gd import METHOD_STYLES
        base_path = output_path.parent / output_path.stem
        plot_summary(all_results["methods"], base_path, args.plot_dpi, METHOD_STYLES)

    if args.spectrum_gif and spectrum_payloads:
        from plot_weighted_spec_spectrum import generate_weighted_spec_spectrum_gif

        gif_output_dir = args.spectrum_gif_output or args.output_dir
        gif_output_dir.mkdir(parents=True, exist_ok=True)
        gif_name = args.spectrum_gif_name or f"{output_path.stem}_spectrum.gif"
        generate_weighted_spec_spectrum_gif(
            log_path=output_path,
            methods=list(spectrum_payloads.keys()),
            output_dir=gif_output_dir,
            gif_name=gif_name,
            fps=args.spectrum_gif_fps,
            dpi=args.spectrum_gif_dpi,
            keep_frames=args.spectrum_gif_keep_frames,
        )


if __name__ == "__main__":
    main()

