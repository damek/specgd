import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
import json
import math
from collections import defaultdict
from typing import Any, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Instrumented copy of train_gpt.py from commit 0d20074260664590e2a1686b9371936944a3ddff (2025-07-18).
# Training logic matches upstream; this variant adds stable-rank tracking and logging only.

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
#torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)


mm_op.register_autograd(backward, setup_context=setup_context)

TRACK_METRIC_INTERVAL = 100
stable_rank_tracker = None

class StableRankTracker:
    def __init__(
        self,
        interval: int,
        device: torch.device,
        master_process: bool,
        run_id: uuid.UUID | None,
        save_dir: str = "logs",
        vocab_size: int | None = None,
    ):
        self.interval = interval
        self.device = device
        self.master_process = master_process
        self.run_id = run_id
        self.save_dir = Path(save_dir)
        self.save_path = self.save_dir / (f"{run_id}_stable_rank.json" if run_id is not None else "stable_rank.json")
        self.vocab_size = vocab_size if vocab_size is not None else 50304
        self.active = False
        self.current_step: int | None = None
        self.activation_accumulators: dict[str, dict[str, Tensor]] = {}
        self.activation_history: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
        self.grad_history_mlp: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
        self.grad_history_attn: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
        self.grad_history_embed: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
        self.grad_history_value_embed: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
        self.weight_history_mlp: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
        self.weight_history_attn: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
        self.qkv_column_history: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
        self.qkv_column_accumulators: dict[str, dict[str, Tensor]] = {}
        self.embed_lmhead_history: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
        self.embed_lmhead_accumulators: dict[str, dict[str, Tensor]] = {}
        self.value_embed_history: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
        self.value_embed_accumulators: dict[str, dict[str, Tensor]] = {}
        self.rms_history: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
        self.rms_accumulators: dict[str, dict[str, Tensor]] = {}
        self.token_frequency_history: list[tuple[int, float]] = []
        self.token_count_accumulator: Tensor | None = None
        self.activation_chunk_rows = 4096
        self.accum_device = torch.device("cpu")
        self.data_dirty = False
        self._announced_data_path = False

    @staticmethod
    def _parse_block_index(key: str) -> int | None:
        if key.startswith("block"):
            rest = key[5:]
            try:
                return int(rest.split(".", 1)[0])
            except ValueError:
                return None
        head = key.split(".", 1)[0]
        return int(head) if head.isdigit() else None

    @staticmethod
    def _activation_component(key: str) -> str | None:
        if key.endswith("c_fc"):
            return "mlp.layer1"
        if key.endswith("c_proj"):
            return "mlp.layer2"
        return None

    @staticmethod
    def _gradient_component(key: str) -> str | None:
        parts = key.split(".")
        if len(parts) < 2:
            return None
        if parts[1] == "mlp" and len(parts) >= 3:
            if parts[2].startswith("c_fc"):
                return "mlp.layer1"
            if parts[2].startswith("c_proj"):
                return "mlp.layer2"
        if parts[1] == "attn" and len(parts) >= 3:
            if parts[2].startswith("c_proj"):
                return "attn.c_proj"
            if parts[2] in {"q", "k", "v"}:
                return f"attn.{parts[2]}"
        return None

    @staticmethod
    def _gradient_group(key: str) -> str | None:
        parts = key.split(".")
        if len(parts) < 2:
            return None
        if parts[1] == "mlp":
            return "mlp"
        if parts[1] == "attn":
            return "attn"
        return None

    def should_track(self, step: int) -> bool:
        return step > 1 and (step % self.interval == 0 or step == 2)

    def start_step(self, step: int):
        if self.should_track(step):
            self.active = True
            self.current_step = step
            self.activation_accumulators = {}
            self.qkv_column_accumulators = {}
            self.embed_lmhead_accumulators = {}
            self.value_embed_accumulators = {}
            self.rms_accumulators = {}
            self.token_count_accumulator = torch.zeros(
                self.vocab_size, device=self.device, dtype=torch.float64
            )
        else:
            self.active = False
            self.current_step = None
            self.activation_accumulators = {}
            self.qkv_column_accumulators = {}
            self.embed_lmhead_accumulators = {}
            self.value_embed_accumulators = {}
            self.rms_accumulators = {}
            self.token_count_accumulator = None

    def record_activation(self, key: str | None, tensor: Tensor):
        if not self.active or key is None:
            return
        with torch.no_grad():
            if tensor.ndim < 2:
                return
            flat = tensor.detach().reshape(-1, tensor.shape[-1])
            rows = flat.shape[0]
            dims = flat.shape[-1]
            chunk_rows = self.activation_chunk_rows
            store = self.activation_accumulators.setdefault(
                key,
                {
                    "gram": torch.zeros((dims, dims), device=self.device, dtype=torch.float64),
                    "frob_sq": torch.zeros((), device=self.device, dtype=torch.float64),
                },
            )
            gram_acc = store["gram"]
            frob_acc = store["frob_sq"]
            for start in range(0, rows, chunk_rows):
                chunk = flat[start:start + chunk_rows].to(dtype=torch.float32)
                if chunk.numel() == 0:
                    continue
                gram_chunk = chunk.transpose(0, 1) @ chunk
                gram_acc.add_(gram_chunk.to(dtype=torch.float64))
                frob_acc.add_(torch.sum(chunk * chunk, dtype=torch.float64))

    def record_tokens(self, tokens: Tensor):
        if not self.active or self.token_count_accumulator is None:
            return
        with torch.no_grad():
            flat = tokens.detach().reshape(-1).to(device=self.device, dtype=torch.int64)
            if flat.numel() == 0:
                return
            counts = torch.bincount(flat, minlength=self.vocab_size).to(dtype=torch.float64)
            self.token_count_accumulator.add_(counts)

    def _accumulate_rms(self, key: str, tensor: Tensor):
        if not self.active:
            return
        with torch.no_grad():
            if tensor.ndim < 2:
                return
            flat = tensor.detach().reshape(-1, tensor.shape[-1])
            rows = flat.shape[0]
            dims = flat.shape[-1]
            chunk_rows = self.activation_chunk_rows
            store = self.rms_accumulators.setdefault(
                key,
                {
                    "gram": torch.zeros((dims, dims), device=self.device, dtype=torch.float64),
                    "frob_sq": torch.zeros((), device=self.device, dtype=torch.float64),
                },
            )
            gram_acc = store["gram"]
            frob_acc = store["frob_sq"]
            for start in range(0, rows, chunk_rows):
                chunk = flat[start:start + chunk_rows].to(dtype=torch.float32)
                if chunk.numel() == 0:
                    continue
                gram_chunk = chunk.transpose(0, 1) @ chunk
                gram_acc.add_(gram_chunk.to(dtype=torch.float64))
                frob_acc.add_(torch.sum(chunk * chunk, dtype=torch.float64))

    def record_rms(self, block_id: int, label: str, tensor: Tensor):
        key = f"block{block_id}.rms.{label}"
        self._accumulate_rms(key, tensor)

    def record_lm_head_rms(self, label: str, tensor: Tensor):
        key = f"lm_head.{label}"
        self._accumulate_rms(key, tensor)

    def _finalize_activation_stats(self) -> dict[str, float]:
        if not self.active or self.current_step is None:
            self.activation_accumulators = {}
            return {}
        results: dict[str, float] = {}
        eps = 1e-12
        for key, stats in self.activation_accumulators.items():
            gram = stats["gram"].clone()
            frob_sq = stats["frob_sq"].clone()
            dist.all_reduce(gram, op=dist.ReduceOp.SUM)
            dist.all_reduce(frob_sq, op=dist.ReduceOp.SUM)
            if not self.master_process:
                continue
            frob_val = frob_sq.item()
            if frob_val <= eps:
                results[key] = 0.0
                continue
            gram_cpu = gram.cpu()
            gram_sym = 0.5 * (gram_cpu + gram_cpu.mT)
            try:
                evals = torch.linalg.eigvalsh(gram_sym)
            except RuntimeError:
                evals = torch.linalg.eigvalsh(gram_sym.to(dtype=torch.float32)).to(dtype=torch.float64)
            spec_sq = float(torch.clamp(evals.max(), min=0.0))
            if spec_sq <= eps:
                results[key] = math.inf if frob_val > eps else 0.0
            else:
                results[key] = frob_val / spec_sq
        self.activation_accumulators = {}
        return results

    def _finalize_rms_stats(self) -> dict[str, float]:
        if not self.active or self.current_step is None:
            self.rms_accumulators = {}
            return {}
        results: dict[str, float] = {}
        eps = 1e-12
        for key, stats in self.rms_accumulators.items():
            gram = stats["gram"].clone()
            frob_sq = stats["frob_sq"].clone()
            dist.all_reduce(gram, op=dist.ReduceOp.SUM)
            dist.all_reduce(frob_sq, op=dist.ReduceOp.SUM)
            if not self.master_process:
                continue
            frob_val = frob_sq.item()
            if frob_val <= eps:
                results[key] = 0.0
                continue
            gram_cpu = gram.cpu()
            gram_sym = 0.5 * (gram_cpu + gram_cpu.mT)
            try:
                evals = torch.linalg.eigvalsh(gram_sym)
            except RuntimeError:
                evals = torch.linalg.eigvalsh(gram_sym.to(dtype=torch.float32)).to(dtype=torch.float64)
            spec_sq = float(torch.clamp(evals.max(), min=0.0))
            if spec_sq <= eps:
                results[key] = math.inf if frob_val > eps else 0.0
            else:
                results[key] = frob_val / spec_sq
        self.rms_accumulators = {}
        return results

    def _finalize_token_frequency(self) -> float | None:
        if not self.active or self.token_count_accumulator is None:
            self.token_count_accumulator = None
            return None
        counts = self.token_count_accumulator.clone()
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        total = counts.sum()
        max_count = counts.max()
        self.token_count_accumulator = None
        if not self.master_process:
            return None
        total_val = float(total.item()) if total.numel() > 0 else 0.0
        max_val = float(max_count.item()) if max_count.numel() > 0 else 0.0
        if total_val <= 0 or max_val <= 0:
            return 0.0
        return total_val / max_val

    def record_qkv_columns(self, block_id: int, q: Tensor, k: Tensor, v: Tensor):
        """Record QKV column statistics (aggregated across all heads).
        q, k, v have shape (B, T, num_heads, head_dim) after normalization/rotary.
        We compute the upper bound mean(||col_i||^2) / ||mean(col_i)||^2 treating the full
        (num_heads*head_dim)-dimensional vectors per token as samples.
        """
        if not self.active:
            return
        with torch.no_grad():
            for qkv_tensor, suffix in [(q, "q"), (k, "k"), (v, "v")]:
                # (B, T, num_heads, head_dim) -> (B*T, num_heads*head_dim)
                flat = qkv_tensor.detach().reshape(-1, qkv_tensor.shape[-2] * qkv_tensor.shape[-1])
                rows, dims = flat.shape
                chunk_rows = self.activation_chunk_rows
                key = f"block{block_id}.attn.{suffix}"
                store = self.qkv_column_accumulators.setdefault(
                    key,
                    {
                        "gram": torch.zeros((dims, dims), device=self.device, dtype=torch.float64),
                        "frob_sq": torch.zeros((), device=self.device, dtype=torch.float64),
                    },
                )
                gram_acc = store["gram"]
                frob_acc = store["frob_sq"]
                for start in range(0, rows, chunk_rows):
                    chunk = flat[start:start + chunk_rows].to(dtype=torch.float32)
                    if chunk.numel() == 0:
                        continue
                    gram_chunk = chunk.transpose(0, 1) @ chunk
                    gram_acc.add_(gram_chunk.to(dtype=torch.float64))
                    frob_acc.add_(torch.sum(chunk * chunk, dtype=torch.float64))

    def _finalize_qkv_column_stats(self) -> dict[str, float]:
        if not self.active or self.current_step is None:
            self.qkv_column_accumulators = {}
            return {}
        results: dict[str, float] = {}
        eps = 1e-12
        for key, stats in self.qkv_column_accumulators.items():
            gram = stats["gram"].clone()
            frob_sq = stats["frob_sq"].clone()
            dist.all_reduce(gram, op=dist.ReduceOp.SUM)
            dist.all_reduce(frob_sq, op=dist.ReduceOp.SUM)
            if not self.master_process:
                continue
            frob_val = frob_sq.item()
            if frob_val <= eps:
                results[key] = 0.0
                continue
            gram_cpu = gram.cpu()
            gram_sym = 0.5 * (gram_cpu + gram_cpu.mT)
            try:
                evals = torch.linalg.eigvalsh(gram_sym)
            except RuntimeError:
                evals = torch.linalg.eigvalsh(gram_sym.to(dtype=torch.float32)).to(dtype=torch.float64)
            spec_sq = float(torch.clamp(evals.max(), min=0.0))
            if spec_sq <= eps:
                results[key] = math.inf if frob_val > eps else 0.0
            else:
                results[key] = frob_val / spec_sq
        self.qkv_column_accumulators = {}
        return results

    def _accumulate_exact_sr(self, accumulator: dict[str, dict[str, Tensor]], key: str, tensor: Tensor | None):
        if tensor is None or tensor.ndim < 2:
            return
        flat = tensor.detach().reshape(-1, tensor.shape[-1])
        rows = flat.shape[0]
        if rows == 0:
            return
        dims = flat.shape[-1]
        chunk_rows = self.activation_chunk_rows
        store = accumulator.setdefault(
            key,
            {
                "gram": torch.zeros((dims, dims), device=self.device, dtype=torch.float64),
                "frob_sq": torch.zeros((), device=self.device, dtype=torch.float64),
            },
        )
        gram_acc = store["gram"]
        frob_acc = store["frob_sq"]
        for start in range(0, rows, chunk_rows):
            chunk = flat[start:start + chunk_rows].to(dtype=torch.float32, device=self.device)
            if chunk.numel() == 0:
                continue
            gram_chunk = chunk.transpose(0, 1) @ chunk
            gram_acc.add_(gram_chunk.to(dtype=torch.float64))
            frob_acc.add_(torch.sum(chunk * chunk, dtype=torch.float64))

    def record_embed_logits(self, embed_output: Tensor | None, logits: Tensor | None):
        """Record embedding output stable ranks via Gram accumulation."""
        if not self.active:
            return
        with torch.no_grad():
            if embed_output is None:
                return
            self._accumulate_exact_sr(self.embed_lmhead_accumulators, "embedding", embed_output)

    def record_value_embeddings(self, buffers: list[Tensor]):
        """Record stable-rank upper bound for value embedding lookup matrices."""
        if not self.active or not buffers:
            return
        with torch.no_grad():
            for idx, tensor in enumerate(buffers):
                key = f"value_embed{idx}"
                self._accumulate_exact_sr(self.value_embed_accumulators, key, tensor)

    def _finalize_embed_lmhead_stats(self) -> dict[str, float]:
        if not self.active or self.current_step is None:
            self.embed_lmhead_accumulators = {}
            return {}
        results: dict[str, float] = {}
        eps = 1e-12
        for key, stats in self.embed_lmhead_accumulators.items():
            gram = stats["gram"].clone()
            frob_sq = stats["frob_sq"].clone()
            dist.all_reduce(gram, op=dist.ReduceOp.SUM)
            dist.all_reduce(frob_sq, op=dist.ReduceOp.SUM)
            if not self.master_process:
                continue
            frob_val = frob_sq.item()
            if frob_val <= eps:
                results[key] = 0.0
                continue
            gram_cpu = gram.cpu()
            gram_sym = 0.5 * (gram_cpu + gram_cpu.mT)
            try:
                evals = torch.linalg.eigvalsh(gram_sym)
            except RuntimeError:
                evals = torch.linalg.eigvalsh(gram_sym.to(dtype=torch.float32)).to(dtype=torch.float64)
            spec_sq = float(torch.clamp(evals.max(), min=0.0))
            if spec_sq <= eps:
                results[key] = math.inf if frob_val > eps else 0.0
            else:
                results[key] = frob_val / spec_sq
        self.embed_lmhead_accumulators = {}
        return results

    def _finalize_value_embed_stats(self) -> dict[str, float]:
        if not self.active or self.current_step is None:
            self.value_embed_accumulators = {}
            return {}
        results: dict[str, float] = {}
        eps = 1e-12
        for key, stats in self.value_embed_accumulators.items():
            gram = stats["gram"].clone()
            frob_sq = stats["frob_sq"].clone()
            dist.all_reduce(gram, op=dist.ReduceOp.SUM)
            dist.all_reduce(frob_sq, op=dist.ReduceOp.SUM)
            if not self.master_process:
                continue
            frob_val = frob_sq.item()
            if frob_val <= eps:
                results[key] = 0.0
                continue
            gram_cpu = gram.cpu()
            gram_sym = 0.5 * (gram_cpu + gram_cpu.mT)
            try:
                evals = torch.linalg.eigvalsh(gram_sym)
            except RuntimeError:
                evals = torch.linalg.eigvalsh(gram_sym.to(dtype=torch.float32)).to(dtype=torch.float64)
            spec_sq = float(torch.clamp(evals.max(), min=0.0))
            if spec_sq <= eps:
                results[key] = math.inf if frob_val > eps else 0.0
            else:
                results[key] = frob_val / spec_sq
        self.value_embed_accumulators = {}
        return results

    def _record_gradients(self, named_params: list[tuple[str, nn.Parameter]]) -> tuple[dict[str, float], dict[str, float]]:
        mlp_results: dict[str, float] = {}
        attn_results: dict[str, float] = {}
        eps = 1e-12
        for name, param in named_params:
            grad = param.grad
            if grad is None or grad.is_sparse:
                continue
            with torch.no_grad():
                grad_array = grad.detach().clone().to(dtype=torch.float32)
                dist.all_reduce(grad_array, op=dist.ReduceOp.AVG)
                if grad_array.ndim < 2:
                    continue
                if grad_array.ndim == 3 and grad_array.shape[0] == 4 and name.endswith("attn.qkvo_w"):
                    base_prefix = name.rsplit(".", 1)[0]  # e.g., "0.attn"
                    for idx, suffix in enumerate(("q", "k", "v", "c_proj")):
                        slice_matrix = grad_array[idx]
                        svals = torch.linalg.svdvals(slice_matrix)
                        nuc = torch.sum(svals, dtype=torch.float64)
                        frob_sq = torch.sum(svals * svals, dtype=torch.float64)
                        if torch.isnan(frob_sq) or frob_sq.item() <= eps:
                            continue
                        num = (nuc * nuc).to(dtype=torch.float64)
                        den = frob_sq.to(dtype=torch.float64)
                        denom = den.item()
                        if denom <= eps:
                            continue
                        key = f"{base_prefix}.{suffix}"
                        attn_results[key] = num.item() / denom
                    continue
                grad_matrix = grad_array
                if grad_matrix.ndim > 2:
                    grad_matrix = grad_matrix.flatten(0, -2)
                group = self._gradient_group(name)
                svals = torch.linalg.svdvals(grad_matrix)
                nuc = torch.sum(svals, dtype=torch.float64)
                frob_sq = torch.sum(svals * svals, dtype=torch.float64)
                if torch.isnan(frob_sq) or frob_sq.item() <= eps:
                    continue
                num = (nuc * nuc).to(dtype=torch.float64)
                den = frob_sq.to(dtype=torch.float64)
                denom = den.item()
                if denom <= eps:
                    continue
                if group == "mlp":
                    mlp_results[name] = num.item() / denom
                elif group == "attn":
                    attn_results[name] = num.item() / denom
        return mlp_results, attn_results

    def _record_value_embed_gradients(self, value_params: Sequence[nn.Parameter]) -> dict[str, float]:
        results: dict[str, float] = {}
        eps = 1e-12
        for idx, param in enumerate(value_params):
            if param is None:
                continue
            grad = param.grad
            if grad is None or grad.is_sparse:
                continue
            with torch.no_grad():
                grad_matrix = grad.detach().clone().to(dtype=torch.float32)
                dist.all_reduce(grad_matrix, op=dist.ReduceOp.AVG)
                if grad_matrix.ndim < 2:
                    continue
                if grad_matrix.ndim > 2:
                    grad_matrix = grad_matrix.flatten(0, -2)
                rows, cols = grad_matrix.shape
                if rows <= cols:
                    gram = grad_matrix @ grad_matrix.t()
                else:
                    gram = grad_matrix.t() @ grad_matrix
                gram = gram.to(dtype=torch.float64)
                try:
                    evals = torch.linalg.eigvalsh(gram)
                except RuntimeError:
                    evals = torch.linalg.eigvalsh(gram.to(dtype=torch.float32)).to(dtype=torch.float64)
                evals = torch.clamp(evals, min=0.0)
                frob_sq = evals.sum()
                if torch.isnan(frob_sq) or frob_sq.item() <= eps:
                    continue
                nuclear = torch.sum(torch.sqrt(evals), dtype=torch.float64)
                num = (nuclear * nuclear).to(dtype=torch.float64)
                den = frob_sq
                denom = den.item()
                if denom <= eps:
                    continue
                key = f"value_embed{idx}.weight"
                results[key] = num.item() / denom
        return results

    def _record_embed_gradients(self, embed_param: nn.Parameter | None, lm_head_param: nn.Parameter | None) -> dict[str, float]:
        results: dict[str, float] = {}
        eps = 1e-12
        for key, param in (("embedding", embed_param), ("lm_head", lm_head_param)):
            if param is None:
                continue
            grad = param.grad
            if grad is None or grad.is_sparse:
                continue
            with torch.no_grad():
                grad_matrix = grad.detach().clone().to(dtype=torch.float32)
                dist.all_reduce(grad_matrix, op=dist.ReduceOp.AVG)
                if grad_matrix.ndim < 2:
                    continue
                if grad_matrix.ndim > 2:
                    grad_matrix = grad_matrix.flatten(0, -2)
                rows, cols = grad_matrix.shape
                if rows <= cols:
                    gram = grad_matrix @ grad_matrix.t()
                else:
                    gram = grad_matrix.t() @ grad_matrix
                gram = gram.to(dtype=torch.float64)
                try:
                    evals = torch.linalg.eigvalsh(gram)
                except RuntimeError:
                    # fall back to float32 if numerical issues arise
                    evals = torch.linalg.eigvalsh(gram.to(dtype=torch.float32)).to(dtype=torch.float64)
                evals = torch.clamp(evals, min=0.0)
                frob_sq = evals.sum()
                if torch.isnan(frob_sq) or frob_sq.item() <= eps:
                    continue
                nuclear = torch.sum(torch.sqrt(evals), dtype=torch.float64)
                num = (nuclear * nuclear).to(dtype=torch.float64)
                den = frob_sq
                denom = den.item()
                if denom <= eps:
                    continue
                results[key] = num.item() / denom
        return results

    def _record_weight_stable_ranks(self, named_params: list[tuple[str, nn.Parameter]]) -> tuple[dict[str, float], dict[str, float]]:
        mlp_results: dict[str, float] = {}
        attn_results: dict[str, float] = {}
        eps = 1e-12
        for name, param in named_params:
            if param is None:
                continue
            with torch.no_grad():
                weight = param.detach().to(dtype=torch.float32)
                if weight.ndim < 2:
                    continue
                if weight.ndim == 3 and weight.shape[0] == 3 and name.endswith("attn.qkv_w"):
                    base_prefix = name.rsplit(".", 1)[0]
                    for idx, suffix in enumerate(("q", "k", "v")):
                        slice_matrix = weight[idx]
                        frob_sq = torch.linalg.norm(slice_matrix, ord="fro") ** 2
                        if torch.isnan(frob_sq) or frob_sq.item() <= eps:
                            continue
                        spec = torch.linalg.norm(slice_matrix, ord=2)
                        spec_sq = spec * spec
                        if torch.isnan(spec_sq) or spec_sq.item() <= eps:
                            continue
                        key = f"{base_prefix}.{suffix}"
                        attn_results[key] = frob_sq.item() / spec_sq.item()
                    continue
                weight_matrix = weight
                if weight_matrix.ndim > 2:
                    weight_matrix = weight_matrix.flatten(0, -2)
                group = self._gradient_group(name)
                frob_sq = torch.linalg.norm(weight_matrix, ord="fro") ** 2
                if torch.isnan(frob_sq) or frob_sq.item() <= eps:
                    continue
                spec = torch.linalg.norm(weight_matrix, ord=2)
                spec_sq = spec * spec
                if torch.isnan(spec_sq) or spec_sq.item() <= eps:
                    continue
                if group == "mlp":
                    mlp_results[name] = frob_sq.item() / spec_sq.item()
                elif group == "attn":
                    attn_results[name] = frob_sq.item() / spec_sq.item()
        return mlp_results, attn_results

    def finalize_step(
        self,
        step: int,
        named_params: list[tuple[str, nn.Parameter]],
        embed_param: nn.Parameter | None = None,
        lm_head_param: nn.Parameter | None = None,
        value_params: Sequence[nn.Parameter] | None = None,
    ):
        if not self.should_track(step):
            self.active = False
            self.current_step = None
            self.activation_accumulators = {}
            return
        mlp_grad_results, attn_grad_results = self._record_gradients(named_params)
        embed_grad_results = self._record_embed_gradients(embed_param, lm_head_param)
        value_grad_results = self._record_value_embed_gradients(list(value_params)) if value_params is not None else {}
        act_results = self._finalize_activation_stats()
        rms_results = self._finalize_rms_stats()
        mlp_weight_results, attn_weight_results = self._record_weight_stable_ranks(named_params)
        qkv_col_results = self._finalize_qkv_column_stats()
        embed_lmhead_results = self._finalize_embed_lmhead_stats()
        value_embed_results = self._finalize_value_embed_stats()
        token_inv_pmax = self._finalize_token_frequency()
        if self.master_process:
            for key, value in act_results.items():
                self.activation_history[key].append((step, value))
            for key, value in mlp_grad_results.items():
                self.grad_history_mlp[key].append((step, value))
            for key, value in attn_grad_results.items():
                self.grad_history_attn[key].append((step, value))
            for key, value in embed_grad_results.items():
                self.grad_history_embed[key].append((step, value))
            for key, value in value_grad_results.items():
                self.grad_history_value_embed[key].append((step, value))
            for key, value in mlp_weight_results.items():
                self.weight_history_mlp[key].append((step, value))
            for key, value in attn_weight_results.items():
                self.weight_history_attn[key].append((step, value))
            for key, value in qkv_col_results.items():
                self.qkv_column_history[key].append((step, value))
            for key, value in rms_results.items():
                self.rms_history[key].append((step, value))
            for key, value in embed_lmhead_results.items():
                self.embed_lmhead_history[key].append((step, value))
            for key, value in value_embed_results.items():
                self.value_embed_history[key].append((step, value))
            if token_inv_pmax is not None:
                self.token_frequency_history.append((step, token_inv_pmax))
            if (
                act_results
                or mlp_grad_results
                or attn_grad_results
                or embed_grad_results
                or value_grad_results
                or rms_results
                or mlp_weight_results
                or attn_weight_results
                or qkv_col_results
                or embed_lmhead_results
                or value_embed_results
                or token_inv_pmax is not None
            ):
                self.data_dirty = True
                self.save_data()
        self.active = False
        self.current_step = None

    def metrics_snapshot(self) -> dict[str, Any]:
        def serialize(history: dict[str, list[tuple[int, float]]]) -> dict[str, dict[str, list[float]]]:
            return {
                key: {
                    "steps": [int(s) for s, _ in values],
                    "values": [float(v) for _, v in values],
                }
                for key, values in history.items()
                if values
            }

        return {
            "interval": self.interval,
            "activation": serialize(self.activation_history),
            "gradient": {
                "mlp": serialize(self.grad_history_mlp),
                "attention": serialize(self.grad_history_attn),
                "embed_lmhead": serialize(self.grad_history_embed),
                "value_embed": serialize(self.grad_history_value_embed),
            },
            "weight": {
                "mlp": serialize(self.weight_history_mlp),
                "attention": serialize(self.weight_history_attn),
            },
            "qkv_columns": serialize(self.qkv_column_history),
            "embed_lmhead": serialize(self.embed_lmhead_history),
            "value_embed": serialize(self.value_embed_history),
            "rms": serialize(self.rms_history),
            "token_frequency": {
                "steps": [int(s) for s, _ in self.token_frequency_history],
                "values": [float(v) for _, v in self.token_frequency_history],
            },
        }

    def save_data(self):
        if not (self.master_process and self.data_dirty):
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        data = self.metrics_snapshot()
        with self.save_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.data_dirty = False
        if 'print0' in globals() and not self._announced_data_path:
            print0(f"stable-rank metrics data saved to {self.save_path}", console=True)
            self._announced_data_path = True

    @staticmethod
    def _qkv_column_component(key: str) -> str | None:
        # key format: "block0.attn.q"
        parts = key.split(".")
        if len(parts) >= 3 and parts[2] in {"q", "k", "v"}:
            return parts[2]
        return None

    @staticmethod
    def _embed_logits_component(key: str) -> str | None:
        if key == "embedding":
            return "embedding"
        if key == "logits":
            return "logits"
        if key.startswith("value_embed"):
            return key.split(".")[0]
        return None

    @staticmethod
    def _embed_gradient_component(key: str) -> str | None:
        if key == "embedding":
            return "embedding"
        if key == "lm_head":
            return "lm_head"
        if key.startswith("value_embed"):
            return key.split(".")[0]
        return None

    @staticmethod
    def _rms_component(key: str) -> str | None:
        if key.endswith("pre_attn"):
            return "pre_attn"
        if key.endswith("pre_mlp"):
            return "pre_mlp"
        if key == "lm_head.pre":
            return "lm_head_pre"
        return None

    def _panel_definitions(self):
        combined_embed_history = {**{k: v for k, v in self.embed_lmhead_history.items()}, **{k: v for k, v in self.value_embed_history.items()}}
        return (
            ("MLP post-activation stable rank", self.activation_history, self._activation_component),
            ("RMSNorm activation stable rank", self.rms_history, self._rms_component),
            ("Attention post-activation stable rank", self.qkv_column_history, self._qkv_column_component),
            ("Embedding/logits/value embedding $\\mathrm{NSR}$", combined_embed_history, self._embed_logits_component),
            ("MLP gradient sr_nuc", self.grad_history_mlp, self._gradient_component),
            ("Attention gradient sr_nuc", self.grad_history_attn, self._gradient_component),
            ("Embedding/LM head/value embedding gradient sr_nuc", {**self.grad_history_embed, **self.grad_history_value_embed}, self._embed_gradient_component),
        )

    def save_plot(self, run_id: uuid.UUID | None):
        if not self.master_process:
            return
        has_activation = any(self.activation_history.values())
        has_grad_mlp = any(self.grad_history_mlp.values())
        has_grad_attn = any(self.grad_history_attn.values())
        has_grad_embed = any(self.grad_history_embed.values())
        has_weight_mlp = any(self.weight_history_mlp.values())
        has_weight_attn = any(self.weight_history_attn.values())
        has_qkv_col = any(self.qkv_column_history.values())
        has_embed_lmhead = any(self.embed_lmhead_history.values())
        has_value_embed = any(self.value_embed_history.values())
        has_rms = any(self.rms_history.values())
        if not (
            has_activation
            or has_grad_mlp
            or has_grad_attn
            or has_grad_embed
            or has_weight_mlp
            or has_weight_attn
            or has_qkv_col
            or has_embed_lmhead
            or has_value_embed
            or has_rms
        ):
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import MaxNLocator

        panels = [panel for panel in self._panel_definitions() if any(panel[1].values())]
        if not panels:
            return

        histories_for_blocks = [panel[1] for panel in panels]
        block_ids: set[int] = set()
        for history in histories_for_blocks:
            for key in history.keys():
                idx = self._parse_block_index(key)
                if idx is not None:
                    block_ids.add(idx)
        block_ids = sorted(block_ids)
        num_blocks = max(len(block_ids), 1)
        cmap = matplotlib.colormaps.get_cmap("tab20").resampled(num_blocks)
        block_colors = {idx: cmap(i) for i, idx in enumerate(block_ids)}
        default_color = cmap(0)

        activation_styles: dict[str, tuple[any, str]] = {
            "mlp.layer1": ("-", ""),
        }
        rms_styles: dict[str, tuple[any, str]] = {
            "pre_attn": ("-", "RMSNorm before attention"),
            "pre_mlp": ((0, (2, 2)), "RMSNorm before MLP"),
        }
        mlp_gradient_styles: dict[str, tuple[any, str]] = {
            "mlp.layer1": ("-", "MLP layer 1"),
            "mlp.layer2": ("--", "MLP layer 2"),
        }
        attn_gradient_styles: dict[str, tuple[any, str]] = {
            "attn.c_proj": ("-.", "Attn c_proj"),
            "attn.q": ((0, (3, 1)), "W_Q"),
            "attn.k": ((0, (1, 1)), "W_K"),
            "attn.v": ((0, (5, 2)), "W_V"),
        }
        embed_gradient_styles: dict[str, tuple[any, ...]] = {
            "embedding": ("-", "Embedding", "#1f77b4"),
            "lm_head": ((0, (4, 2)), "LM head", "#d62728"),
            "value_embed0.weight": ((0, (3, 2)), "Value embed 0", "#2ca02c"),
            "value_embed1.weight": ((0, (1, 3)), "Value embed 1", "#9467bd"),
            "value_embed2.weight": ((0, (5, 2)), "Value embed 2", "#8c564b"),
        }
        value_embed_gradient_styles: dict[str, tuple[any, str]] = {
            "value_embed0.weight": ("-", "Value embed 0"),
            "value_embed1.weight": ((0, (3, 2)), "Value embed 1"),
            "value_embed2.weight": ((0, (1, 3)), "Value embed 2"),
        }
        qkv_column_styles: dict[str, tuple[any, str]] = {
            "q": ("-", "Q"),
            "k": ("--", "K"),
            "v": ("-.", "V"),
        }
        embed_logits_styles: dict[str, tuple[any, ...]] = {
            "embedding": ("-", "Embedding", "#1f77b4"),
            "logits": ("--", "Logits", "#d62728"),
            "value_embed0.weight": ((0, (3, 2)), "Value embed 0", "#2ca02c"),
            "value_embed1.weight": ((0, (1, 3)), "Value embed 1", "#9467bd"),
            "value_embed2.weight": ((0, (5, 2)), "Value embed 2", "#8c564b"),
        }
        value_embed_styles: dict[str, tuple[any, str]] = {
            "value_embed0.weight": ("-", "Value embed 0"),
            "value_embed1.weight": ((0, (3, 2)), "Value embed 1"),
            "value_embed2.weight": ((0, (1, 3)), "Value embed 2"),
        }

        num_panels = len(panels)
        fig = plt.figure(figsize=(12, 4.25 * num_panels))
        gs = fig.add_gridspec(2 * num_panels, 1, height_ratios=[0.4, 1.0] * num_panels, hspace=0.3)
        axes = []
        legend_axes = []
        for idx in range(num_panels):
            legend_ax = fig.add_subplot(gs[2 * idx])
            legend_ax.axis("off")
            data_ax = fig.add_subplot(gs[2 * idx + 1], sharex=axes[0] if axes else None)
            axes.append(data_ax)
            legend_axes.append(legend_ax)

        panel_block_usage: list[dict[int, tuple[float, ...]]] = []
        panel_component_usage: list[set[str]] = []
        panel_component_styles: list[dict[str, tuple[Any, Any, Any | None]]] = []

        for ax, (title, history, component_fn) in zip(axes, panels):
            block_usage: dict[int, tuple[float, ...]] = {}
            component_usage: set[str] = set()
            component_styles: dict[str, tuple[Any, Any, Any | None]] = {}
            if title.startswith("MLP post-activation"):
                style_lookup = activation_styles
            elif title.startswith("RMSNorm"):
                style_lookup = rms_styles
            elif title.startswith("Attention post-activation"):
                style_lookup = qkv_column_styles
            elif title.startswith("MLP gradient"):
                style_lookup = mlp_gradient_styles
            elif title.startswith("Attention gradient"):
                style_lookup = attn_gradient_styles
            elif "Embedding" in title and "gradient" in title:
                style_lookup = embed_gradient_styles
            elif title.startswith("Embedding"):
                style_lookup = embed_logits_styles
            else:
                style_lookup = activation_styles
            for key in sorted(history.keys()):
                data = history[key]
                if not data:
                    continue
                steps = [s for s, _ in data]
                values = [v for _, v in data]
                block_idx = self._parse_block_index(key)
                color = block_colors.get(block_idx, default_color)
                component = component_fn(key)
                style_info = style_lookup.get(component)
                linestyle = style_info[0] if style_info else "-"
                label = component or key
                color_override = None
                if style_info:
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
            ax.set_title(title)
            ax.set_ylabel("value")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
            panel_block_usage.append(block_usage)
            panel_component_usage.append(component_usage)
            panel_component_styles.append(component_styles)

        axes[-1].set_xlabel("step")

        for legend_ax, block_usage, component_usage, component_styles, (title, _, _) in zip(
            legend_axes, panel_block_usage, panel_component_usage, panel_component_styles, panels
        ):
            if title.startswith("MLP post-activation"):
                style_lookup = activation_styles
            elif title.startswith("MLP gradient"):
                style_lookup = mlp_gradient_styles
            elif title.startswith("Attention gradient"):
                style_lookup = attn_gradient_styles
            elif "Embedding" in title and "gradient" in title:
                style_lookup = embed_gradient_styles
            elif title.startswith("MLP weight"):
                style_lookup = mlp_gradient_styles
            elif title.startswith("Attention weight"):
                style_lookup = attn_gradient_styles
            elif title.startswith("Attention post-activation"):
                style_lookup = qkv_column_styles
            elif title.startswith("Embedding/logits/value embedding $"):
                style_lookup = embed_logits_styles
            else:
                style_lookup = activation_styles

            color_handles = [
                Line2D([0], [0], color=color, linestyle="-", linewidth=2.5, label=f"block{idx}")
                for idx, color in sorted(block_usage.items())
            ]
            color_legend = None
            if color_handles:
                color_legend = legend_ax.legend(
                    color_handles,
                    [h.get_label() for h in color_handles],
                    loc="upper center",
                    ncol=min(len(color_handles), 6),
                    fontsize="small",
                    frameon=False,
                )

            ordered_components = [c for c in style_lookup.keys() if c in component_usage]
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
                style_legend = legend_ax.legend(
                    style_handles,
                    [h.get_label() for h in style_handles],
                    loc="lower center",
                    ncol=len(style_handles),
                    fontsize="small",
                    frameon=False,
                )
                if color_legend is not None:
                    legend_ax.add_artist(color_legend)
            elif color_legend is not None:
                legend_ax.add_artist(color_legend)

        fig.tight_layout()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        filename = self.save_dir / (f"{run_id}_stable_rank.png" if run_id is not None else "stable_rank.png")
        fig.savefig(filename, dpi=200)
        plt.close(fig)
        if 'print0' in globals():
            print0(f"stable-rank metrics plot saved to {filename}", console=True)


def load_stable_rank_metrics(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)


# -----------------------------------------------------------------------------
# Triton kernel for symmetric matrix multiplication by @byronxu99

def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]

@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ns_line_1_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def ns_line_1(A: torch.Tensor, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ns_line_1_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
    )
    return out

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ns_line_2_kernel(
    A_ptr, C_ptr,
    M,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    # This is mostly duplicated from ns_line_1_kernel, but also loads and adds a block of A
    # Performance is slightly slower than ns_line_1_kernel, so we use two separate kernels
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A to add (corresponds to the current block of C)
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    # Apply alpha and beta
    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def ns_line_2(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = alpha * A @ A.T + beta * A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K, "Input matrix must be square"
    assert out.size(-2) == M
    assert out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ns_line_2_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
    )
    return out

@torch.compile(dynamic=False, fullgraph=True) # Must use dynamic=False or else it's much slower
def newton_schulz_triton(G: torch.Tensor):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Allocate buffers
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    ns_line_3 = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Perform the NS iterations
    for _ in range(5):
        ns_line_1(X, out=A)  # A = X @ X.mT
        ns_line_2(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A
        ns_line_3(X, B, X, beta=a, out=C)  # C = a * X + B @ X
        X, C = C, X  # Swap references to avoid unnecessary copies

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

# -----------------------------------------------------------------------------
# Muon optimizer

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        # Efficient systems-wise implementation of step developed by @YouJiacheng,
        # @KonstantinWilleke, @alexrgilbert, @adricarda, @tuttyfrutyee, @vdlad,
        # @ryanyang0, and @vagrawal.
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_gather_futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            grad_pad = [param.grad for param in params] + [torch.zeros_like(params[-1])] * world_size
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    grad = params[base_i + rank].grad
                # This gives strange dynamo warnings
                reduce_scatter_futures.append(dist.reduce_scatter(grad, grad_pad[base_i:base_i + world_size], op=dist.ReduceOp.AVG, async_op=True).get_future())

        idx = 0
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * world_size
            momentum = group["momentum"]
            for base_i in range(0, len(params), world_size):
                reduce_scatter_futures[idx].wait()
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    grad = p.grad
                    eff_lr = group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5 * getattr(p, "lr_mul", 1.0)
                    eff_weight_decay = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    momentum_buffer = state["momentum_buffer"]
                    p.mul_(1 - eff_weight_decay)
                    momentum_buffer.lerp_(grad, 1 - momentum)
                    grad = grad.lerp_(momentum_buffer, momentum)
                    v = newton_schulz_triton(grad)
                    p.add_(other=v, alpha=-eff_lr)
                idx += 1
                all_gather_futures.append(dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank], async_op=True).get_future())
        torch.futures.collect_all(all_gather_futures).wait()

class DistAdam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)
        # DistributedAdam implementation by @vagrawal

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_gather_futures: list[torch.Future] = []
        grad_slices = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]
                # State init
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1
                all_gather_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        torch.futures.collect_all(all_gather_futures).wait()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"model_dim ({dim}) must be divisible by num_heads ({num_heads})")
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkvo_w = nn.Parameter(torch.empty(4, hdim, dim))
        with torch.no_grad():
            self.qkvo_w[:3].uniform_(-bound, bound) # init QKV weights
            self.qkvo_w[3].zero_() # init output weights to zero
        self.rotary = Rotary(head_dim, max_seq_len)
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, lambdas: Tensor, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkvo_w[:3].flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        tracker = stable_rank_tracker
        if tracker is not None and tracker.active and hasattr(self, "stable_rank_block_id"):
            tracker.record_qkv_columns(self.stable_rank_block_id, q, k, v)
        if ve is not None:
            v = lambdas[0] * v + lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = lambdas[0] * v
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, self.qkvo_w[3].type_as(y))
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        # make both matrices have the same shape because optimizer sorts params by shape
        # 2 matrices x 12 layers = 24 total, which is divisible by 8 GPU world size
        self.c_fc = nn.Parameter(torch.empty(dim, hdim))
        self.c_proj = nn.Parameter(torch.empty(dim, hdim))
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        prefix = getattr(self, "stable_rank_prefix", None)
        x = F.linear(x, self.c_fc.T.type_as(x))
        tracker = stable_rank_tracker
        if tracker is not None and tracker.active:
            tracker.record_activation(f"{prefix}.c_fc" if prefix is not None else None, x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = F.linear(x, self.c_proj.type_as(x))
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, lambdas: Tensor, sa_lambdas: Tensor, block_mask: BlockMask):
        x = lambdas[0] * x + lambdas[1] * x0
        tracker = stable_rank_tracker
        rms_pre_attn = norm(x)
        if tracker is not None and tracker.active:
            tracker.record_rms(self.layer_idx, "pre_attn", rms_pre_attn)
        if self.attn is not None:
            x = x + self.attn(rms_pre_attn, ve, sa_lambdas, block_mask)
        rms_pre_mlp = norm(x)
        if tracker is not None and tracker.active:
            tracker.record_rms(self.layer_idx, "pre_mlp", rms_pre_mlp)
        x = x + self.mlp(rms_pre_mlp)
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        for i, block in enumerate(self.blocks):
            if block.attn is not None:
                block.attn.stable_rank_block_id = i
            block.mlp.stable_rank_prefix = f"block{i}.mlp"
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        use_fp8 = not os.environ.get("DISABLE_FP8", False)
        self.lm_head = CastedLinear(model_dim, vocab_size, use_fp8=use_fp8, x_s=(model_dim**0.5)/448, w_s=24/448, grad_s=1/448)
        self.lm_head.weight.detach().zero_() # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        pad = (-num_layers * 5) % dist.get_world_size()
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers), # skip_weights
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)], # block lambdas
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)], # SA lambdas
            torch.ones(pad),
        ]))
        # set learning rates
        for param in self.embed.parameters():
            param.lr_mul = 75.
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.
        self.lm_head.weight.lr_mul = 27.5
        self.scalars.lr_mul = 5.0

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        tracker = stable_rank_tracker
        value_vectors = [value_embed(input_seq) for value_embed in self.value_embeds]
        if tracker is not None and tracker.active:
            tracker.record_value_embeddings(value_vectors)
        ve = [value_vectors[0], value_vectors[1], value_vectors[2]] + [None] * (len(self.blocks) - 6) + [value_vectors[0], value_vectors[1], value_vectors[2]]
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        push_pattern = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm]
        pop_pattern = list(reversed(push_pattern))
        half = len(self.blocks) // 2
        block_masks = [push_pattern[i % len(push_pattern)] for i in range(half)] + [pop_pattern[i % len(pop_pattern)] for i in range(half)]
        assert len(block_masks) == len(self.blocks)

        embed_output = self.embed(input_seq)[None]
        if tracker is not None and tracker.active:
            tracker.record_embed_logits(embed_output, None)
        x = x0 = norm(embed_output) # use of norm here by @Grad62304977

        # U-net design by @brendanh0gan
        skip_connections = []
        skip_weights = self.scalars[:(len(self.blocks) // 2)]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)

        n = len(self.blocks) // 2

        for i in range(len(self.blocks)):
            if i >= n:
                x = x + skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        if tracker is not None and tracker.active:
            tracker.record_lm_head_rms("pre", x)
        raw_logits = self.lm_head(x).float()
        if tracker is not None and tracker.active:
            tracker.record_embed_logits(None, raw_logits)
        logits = 30 * torch.sigmoid(raw_logits / (7.5 * x.size(-1)**0.5))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction="sum" if self.training else "mean")
        return loss

# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

# find world_size starting indicies, such that each begins with token 50256 and local_batches don't overlap
def find_batch_starts(tokens: Tensor, pos: int, seq_len: int, token_window: int):
    boundary_mask = tokens[pos : pos + token_window] == 50256
    boundary_positions = torch.nonzero(boundary_mask, as_tuple=False).squeeze(-1) + pos
    start = boundary_positions[0].item()
    starts = []
    for i in range(1, len(boundary_positions)):
        end = boundary_positions[i].item() 
        if end - start >= seq_len:
            starts.append(start) # append start once end pos is confirmed
            if len(starts) == dist.get_world_size():
                return starts, end - pos
            start = end
    assert False # increase token_window if necessary

def distributed_data_generator(filename_pattern: str, seq_len: int, grad_accum_steps: int, align_to_bos: bool):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    batch_size = seq_len * world_size
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        token_window = grad_accum_steps * (2 * batch_size if align_to_bos else batch_size) # provide buffer to handle samples up to length seq_len
        if pos + token_window + 1 >= len(tokens):
            tokens = _load_data_shard(next(file_iter))
            pos = 0
        for _ in range(grad_accum_steps):
            if align_to_bos:
                batch_starts, tokens_consumed = find_batch_starts(tokens, pos, seq_len, token_window)
                start_idx = batch_starts[rank]
            else:
                tokens_consumed = batch_size
                start_idx = pos + rank * seq_len
            buf = tokens[start_idx:][:seq_len + 1]
            inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
            targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
            pos += tokens_consumed
            token_window -= tokens_consumed
            yield inputs, targets

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len = 48*1024 # FlexAttention sequence length
    val_seq_len = 4*64*1024 # FlexAttention sequence length for validation
    # model selection
    model_variant = "124M"  # options: 124M, 3B
    model_dim = 768
    num_layers = 12
    num_heads = 6
    # optimization
    num_iterations = 1750 # number of iterations to run
    cooldown_frac = 0.45 # fraction of training spent cooling down the learning rate
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = False

    def __post_init__(self):
        if self.model_variant == "124M":
            self.model_dim = 768
            self.num_layers = 12
            self.num_heads = 6
        elif self.model_variant == "3B":
            self.model_dim = 2560
            self.num_layers = 36
            # keep head width reasonable; ensure divisibility
            self.num_heads = 20
        else:
            raise ValueError(f"Unknown model_variant: {self.model_variant}")
args = Hyperparameters()

data_path = os.environ.get("DATA_PATH", ".")
args.train_files = os.path.join(data_path, args.train_files)
args.val_files = os.path.join(data_path, args.val_files)

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert 8 % world_size == 0, "world_size must be a divisor of 8"
grad_accum_steps = 8 // world_size
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0(f"Running Triton version {triton.__version__}")
def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

model: nn.Module = GPT(
    vocab_size=50257,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    model_dim=args.model_dim,
    max_seq_len=max(args.train_seq_len, args.val_seq_len),
).cuda()
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_named_params = [(n, p) for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
hidden_matrix_params = [p for _, p in hidden_matrix_named_params]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

# init the optimizer(s)
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = DistAdam(scalar_params + head_params + embed_params, lr=0.008, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0)
optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, weight_decay=0.0)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1

# attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
def get_window_size_blocks(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)

model: nn.Module = torch.compile(model, dynamic=False, fullgraph=True)

stable_rank_tracker = StableRankTracker(
    interval=TRACK_METRIC_INTERVAL,
    device=device,
    master_process=master_process,
    run_id=run_id if master_process else None,
    vocab_size=model.embed.num_embeddings,
)

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
train_loader = distributed_data_generator(args.train_files, args.train_seq_len, grad_accum_steps, align_to_bos=True)
for _ in range(warmup_steps):
    inputs, targets = next(train_loader)
    model(inputs, targets, get_window_size_blocks(1)).backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del train_loader, initial_state

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, args.train_seq_len, grad_accum_steps, align_to_bos=True)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, args.val_seq_len, grad_accum_steps, align_to_bos=False)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss += model(inputs, targets, get_window_size_blocks(step))
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    if stable_rank_tracker is not None:
        stable_rank_tracker.start_step(step)
    for _ in range(grad_accum_steps):
        inputs, targets = next(train_loader)
        if stable_rank_tracker is not None and stable_rank_tracker.active:
            stable_rank_tracker.record_tokens(inputs)
        model(inputs, targets, get_window_size_blocks(step)).backward()
    if stable_rank_tracker is not None:
        stable_rank_tracker.finalize_step(
            step,
            hidden_matrix_named_params,
            embed_param=model.embed.weight,
            lm_head_param=model.lm_head.weight,
            value_params=[value_embed.weight for value_embed in model.value_embeds],
        )
    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1) # momentum warmup for muon
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers
    for opt in optimizers:
        opt.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

if stable_rank_tracker is not None:
    stable_rank_tracker.save_data()
    if master_process:
        stable_rank_tracker.save_plot(run_id)
print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()
