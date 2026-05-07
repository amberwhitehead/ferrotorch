#!/usr/bin/env python3
"""Regenerate PyTorch reference fixtures for ferrotorch-gpu C8.2.

Tracking issue: #825 (parent: #806).

Output:
    ferrotorch-gpu/tests/conformance/fixtures/gpu_kernels.json

Coverage (4 source modules):

* `kernels.rs` — PTX kernel launchers (elementwise, reductions, normalisation,
  optimiser step, GRU, pooling).  One representative shape per kernel group.
  f32 primary; f64 where the launcher exists.

* `flash_attention.rs` — `gpu_flash_attention_f32` / `gpu_flash_attention_f64`.
  Reference is `torch.nn.functional.scaled_dot_product_attention` with
  `scale=1/sqrt(d)`.  Shape: (B=1, H=1, N_q=8, N_k=8, d=16, d_v=16).

* `sparse.rs` — cuSPARSE wrappers.
  Reference is PyTorch sparse CSR `torch.sparse.mm`.
  Shapes: (m=4, k=4, n=4) for SpMM.
  to_dense / from_dense round-trips for 4x4 CSR matrices.
  CSR<->CSC and COO<->CSR conversions (ferrotorch produces host-side results;
  reference is PyTorch's `.to_sparse_csr()` / `.to_sparse()` indicesConvert).

* `conv.rs` — `gpu_conv2d_f32` (im2col + cuBLAS).
  Reference: `torch.nn.functional.conv2d`.
  Shape: input (1,1,5,5), weight (1,1,3,3), no bias, stride=1, pad=0.

RNG seed: 0xBADCAFE (matches other phases).

Usage:
    python3 scripts/regenerate_gpu_kernels_fixtures.py

The script exits 0 on success and 1 on any error.  All reference values are
serialised as double-precision (float64) for accuracy; the Rust tests choose
their own tolerance per op category.
"""
from __future__ import annotations

import datetime
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = (
    REPO_ROOT
    / "ferrotorch-gpu"
    / "tests"
    / "conformance"
    / "fixtures"
    / "gpu_kernels.json"
)

RNG_SEED: int = 0xBADCAFE
torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RNG_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def to_listf(t: torch.Tensor) -> list[Any]:
    """Materialise a tensor to a Python list with NaN/Inf sentinels."""
    raw = t.detach().to("cpu").to(torch.float64).reshape(-1).tolist()
    out: list[Any] = []
    for v in raw:
        if math.isnan(v):
            out.append("NaN")
        elif math.isinf(v):
            out.append("Infinity" if v > 0 else "-Infinity")
        else:
            out.append(v)
    return out


def rng(shape: list[int], *, low: float = -1.0, high: float = 1.0, seed: int | None = None) -> torch.Tensor:
    """Generate a reproducible uniform tensor on CPU."""
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
        return torch.empty(shape, dtype=torch.float32).uniform_(low, high, generator=gen)
    return torch.empty(shape, dtype=torch.float32).uniform_(low, high)


def fixture_metadata() -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rng_seed": RNG_SEED,
    }


# ---------------------------------------------------------------------------
# Module 1: kernels.rs — elementwise, reductions, normalisation
# ---------------------------------------------------------------------------


def make_kernels_fixtures() -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    torch.manual_seed(RNG_SEED)

    # --- elementwise binary ops -----------------------------------------------
    for op_name, torch_op in [
        ("add", torch.add),
        ("sub", torch.sub),
        ("mul", torch.mul),
        ("div", torch.div),
    ]:
        a = rng([16], seed=RNG_SEED ^ 0x01)
        b = rng([16], low=0.5, high=2.0, seed=RNG_SEED ^ 0x02)
        out = torch_op(a, b)
        fixtures.append({
            "module": "kernels",
            "op": op_name,
            "dtype": "float32",
            "input_a": to_listf(a),
            "input_b": to_listf(b),
            "shape": [16],
            "expected": to_listf(out),
        })

    # f64 variants
    for op_name, torch_op in [
        ("add_f64", torch.add),
        ("sub_f64", torch.sub),
        ("mul_f64", torch.mul),
        ("div_f64", torch.div),
    ]:
        a = rng([16], seed=RNG_SEED ^ 0x03).to(torch.float64)
        b = rng([16], low=0.5, high=2.0, seed=RNG_SEED ^ 0x04).to(torch.float64)
        out = torch_op(a, b)
        fixtures.append({
            "module": "kernels",
            "op": op_name,
            "dtype": "float64",
            "input_a": to_listf(a),
            "input_b": to_listf(b),
            "shape": [16],
            "expected": to_listf(out),
        })

    # --- broadcast ops -----------------------------------------------------------
    for op_name, torch_op in [
        ("broadcast_add", torch.add),
        ("broadcast_mul", torch.mul),
    ]:
        a = rng([4, 8], seed=RNG_SEED ^ 0x10)
        b = rng([8], seed=RNG_SEED ^ 0x11)
        # broadcast: a[i, j] op b[j]
        out = torch_op(a, b)
        fixtures.append({
            "module": "kernels",
            "op": op_name,
            "dtype": "float32",
            "input_a": to_listf(a),
            "input_b": to_listf(b),
            "shape_a": [4, 8],
            "shape_b": [8],
            "shape_out": [4, 8],
            "expected": to_listf(out),
        })

    # --- unary ops ---------------------------------------------------------------
    for op_name, torch_op in [
        ("neg", torch.neg),
        ("relu", F.relu),
        ("abs", torch.abs),
        ("exp", torch.exp),
        ("log", lambda t: torch.log(torch.abs(t) + 0.1)),
        ("sqrt", lambda t: torch.sqrt(torch.abs(t))),
        ("sigmoid", torch.sigmoid),
        ("tanh", torch.tanh),
        ("gelu", lambda t: F.gelu(t, approximate="none")),
        ("gelu_tanh", lambda t: F.gelu(t, approximate="tanh")),
        ("silu", F.silu),
        ("mish", F.mish),
    ]:
        a = rng([32], seed=RNG_SEED ^ 0x20)
        out = torch_op(a)
        fixtures.append({
            "module": "kernels",
            "op": op_name,
            "dtype": "float32",
            "input_a": to_listf(a),
            "shape": [32],
            "expected": to_listf(out),
        })

    # --- elu (has alpha param) ---------------------------------------------------
    alpha = 1.0
    a = rng([32], seed=RNG_SEED ^ 0x21)
    out = F.elu(a, alpha=alpha)
    fixtures.append({
        "module": "kernels",
        "op": "elu",
        "dtype": "float32",
        "alpha": alpha,
        "input_a": to_listf(a),
        "shape": [32],
        "expected": to_listf(out),
    })

    # --- clamp -------------------------------------------------------------------
    a = rng([32], seed=RNG_SEED ^ 0x22)
    out = torch.clamp(a, min=-0.5, max=0.5)
    fixtures.append({
        "module": "kernels",
        "op": "clamp",
        "dtype": "float32",
        "min": -0.5,
        "max": 0.5,
        "input_a": to_listf(a),
        "shape": [32],
        "expected": to_listf(out),
    })

    # --- scale -------------------------------------------------------------------
    a = rng([32], seed=RNG_SEED ^ 0x23)
    scale = 2.5
    out = a * scale
    fixtures.append({
        "module": "kernels",
        "op": "scale",
        "dtype": "float32",
        "scalar": scale,
        "input_a": to_listf(a),
        "shape": [32],
        "expected": to_listf(out),
    })

    # --- transpose 2D -----------------------------------------------------------
    a = rng([6, 8], seed=RNG_SEED ^ 0x30)
    out = a.t().contiguous()
    fixtures.append({
        "module": "kernels",
        "op": "transpose_2d",
        "dtype": "float32",
        "input_a": to_listf(a),
        "shape_in": [6, 8],
        "shape_out": [8, 6],
        "expected": to_listf(out),
    })

    # --- permute_0213 (4D) -------------------------------------------------------
    # shape [B, H, S, D] -> [B, S, H, D]
    a = rng([2, 4, 8, 16], seed=RNG_SEED ^ 0x31)
    out = a.permute(0, 2, 1, 3).contiguous()
    fixtures.append({
        "module": "kernels",
        "op": "permute_0213",
        "dtype": "float32",
        "input_a": to_listf(a),
        "shape_in": [2, 4, 8, 16],
        "shape_out": [2, 8, 4, 16],
        "expected": to_listf(out),
    })

    # --- reductions: sum, prod, min, max ----------------------------------------
    for op_name, torch_op in [
        ("reduce_sum", torch.sum),
        ("reduce_min", torch.min),
        ("reduce_max", torch.max),
    ]:
        a = rng([64], seed=RNG_SEED ^ 0x40)
        out_val = torch_op(a)
        fixtures.append({
            "module": "kernels",
            "op": op_name,
            "dtype": "float32",
            "input_a": to_listf(a),
            "shape": [64],
            "expected": [float(out_val)],
        })

    # reduce_prod on small input (avoid overflow)
    a = torch.empty([16]).uniform_(0.5, 1.5, generator=torch.Generator().manual_seed(RNG_SEED ^ 0x41))
    out_val = torch.prod(a)
    fixtures.append({
        "module": "kernels",
        "op": "reduce_prod",
        "dtype": "float32",
        "input_a": to_listf(a),
        "shape": [16],
        "expected": [float(out_val)],
    })

    # --- sum_axis ----------------------------------------------------------------
    a = rng([8, 12], seed=RNG_SEED ^ 0x50)
    # sum along axis=1 (reduce columns -> shape [8])
    out = a.sum(dim=1)
    fixtures.append({
        "module": "kernels",
        "op": "sum_axis",
        "dtype": "float32",
        "axis": 1,
        "input_a": to_listf(a),
        "shape_in": [8, 12],
        "shape_out": [8],
        "expected": to_listf(out),
    })

    # --- softmax -----------------------------------------------------------------
    a = rng([4, 16], seed=RNG_SEED ^ 0x60)
    out = F.softmax(a, dim=1)
    fixtures.append({
        "module": "kernels",
        "op": "softmax",
        "dtype": "float32",
        "dim": 1,
        "input_a": to_listf(a),
        "shape": [4, 16],
        "expected": to_listf(out),
    })

    # --- log_softmax -------------------------------------------------------------
    a = rng([4, 16], seed=RNG_SEED ^ 0x61)
    out = F.log_softmax(a, dim=1)
    fixtures.append({
        "module": "kernels",
        "op": "log_softmax",
        "dtype": "float32",
        "dim": 1,
        "input_a": to_listf(a),
        "shape": [4, 16],
        "expected": to_listf(out),
    })

    # --- cumsum, cumprod, cummax, cummin, logcumsumexp ---------------------------
    a = rng([32], seed=RNG_SEED ^ 0x70)
    fixtures.append({
        "module": "kernels",
        "op": "cumsum",
        "dtype": "float32",
        "input_a": to_listf(a),
        "shape": [32],
        "expected": to_listf(torch.cumsum(a, dim=0)),
    })

    a_prod = torch.empty([16]).uniform_(0.8, 1.2, generator=torch.Generator().manual_seed(RNG_SEED ^ 0x71))
    fixtures.append({
        "module": "kernels",
        "op": "cumprod",
        "dtype": "float32",
        "input_a": to_listf(a_prod),
        "shape": [16],
        "expected": to_listf(torch.cumprod(a_prod, dim=0)),
    })

    a = rng([32], seed=RNG_SEED ^ 0x72)
    fixtures.append({
        "module": "kernels",
        "op": "cummax",
        "dtype": "float32",
        "input_a": to_listf(a),
        "shape": [32],
        "expected": to_listf(torch.cummax(a, dim=0).values),
    })

    fixtures.append({
        "module": "kernels",
        "op": "cummin",
        "dtype": "float32",
        "input_a": to_listf(a),
        "shape": [32],
        "expected": to_listf(torch.cummin(a, dim=0).values),
    })

    # logcumsumexp: reference via torch.logcumsumexp
    a = rng([32], seed=RNG_SEED ^ 0x73)
    fixtures.append({
        "module": "kernels",
        "op": "logcumsumexp",
        "dtype": "float32",
        "input_a": to_listf(a),
        "shape": [32],
        "expected": to_listf(torch.logcumsumexp(a, dim=0)),
    })

    # --- layernorm ---------------------------------------------------------------
    rows, cols = 4, 16
    x = rng([rows * cols], seed=RNG_SEED ^ 0x80)
    w = rng([cols], low=0.8, high=1.2, seed=RNG_SEED ^ 0x81)
    bias = rng([cols], low=-0.1, high=0.1, seed=RNG_SEED ^ 0x82)
    x_2d = x.view(rows, cols)
    ln = torch.nn.LayerNorm(cols, elementwise_affine=True)
    ln.weight.data.copy_(w)
    ln.bias.data.copy_(bias)
    out = ln(x_2d).reshape(-1)
    fixtures.append({
        "module": "kernels",
        "op": "layernorm",
        "dtype": "float32",
        "rows": rows,
        "cols": cols,
        "eps": 1e-5,
        "input": to_listf(x),
        "weight": to_listf(w),
        "bias": to_listf(bias),
        "expected": to_listf(out),
    })

    # --- rmsnorm -----------------------------------------------------------------
    # PyTorch doesn't have a built-in RMSNorm until torch>=2.4; compute manually.
    x = rng([rows * cols], seed=RNG_SEED ^ 0x83)
    w = rng([cols], low=0.8, high=1.2, seed=RNG_SEED ^ 0x84)
    x_2d = x.view(rows, cols)
    eps = 1e-5
    rms = torch.sqrt(torch.mean(x_2d ** 2, dim=1, keepdim=True) + eps)
    out = (x_2d / rms * w).reshape(-1)
    fixtures.append({
        "module": "kernels",
        "op": "rmsnorm",
        "dtype": "float32",
        "rows": rows,
        "cols": cols,
        "eps": eps,
        "input": to_listf(x),
        "weight": to_listf(w),
        "expected": to_listf(out),
    })

    # --- embed_lookup ------------------------------------------------------------
    n_tokens, vocab_size, d = 8, 32, 16
    token_ids = torch.randint(0, vocab_size, [n_tokens], generator=torch.Generator().manual_seed(RNG_SEED ^ 0x90))
    weight = rng([vocab_size * d], seed=RNG_SEED ^ 0x91)
    weight_2d = weight.view(vocab_size, d)
    out = weight_2d[token_ids].reshape(-1)
    fixtures.append({
        "module": "kernels",
        "op": "embed_lookup",
        "dtype": "float32",
        "vocab_size": vocab_size,
        "d": d,
        "n_tokens": n_tokens,
        "token_ids": token_ids.tolist(),
        "weight": to_listf(weight),
        "expected": to_listf(out),
    })

    # --- index_select_1d --------------------------------------------------------
    n_total, d_feat = 32, 8
    src = rng([n_total * d_feat], seed=RNG_SEED ^ 0xA0)
    idx = torch.randint(0, n_total, [12], generator=torch.Generator().manual_seed(RNG_SEED ^ 0xA1))
    src_2d = src.view(n_total, d_feat)
    out = src_2d[idx].reshape(-1)
    fixtures.append({
        "module": "kernels",
        "op": "index_select_1d",
        "dtype": "float32",
        "n_total": n_total,
        "d": d_feat,
        "n_select": 12,
        "indices": idx.tolist(),
        "input": to_listf(src),
        "expected": to_listf(out),
    })

    # --- strided split ----------------------------------------------------------
    # Split [rows, 2*cols] into two [rows, cols] halves along last dim.
    rows, cols = 4, 8
    a = rng([rows, 2 * cols], seed=RNG_SEED ^ 0xB0)
    left, right = a[:, :cols].contiguous(), a[:, cols:].contiguous()
    fixtures.append({
        "module": "kernels",
        "op": "strided_split",
        "dtype": "float32",
        "rows": rows,
        "cols": cols,
        "input": to_listf(a),
        "expected_left": to_listf(left),
        "expected_right": to_listf(right),
    })

    # --- strided cat ------------------------------------------------------------
    # Cat two [rows, cols] tensors along last dim -> [rows, 2*cols]
    a_left = rng([rows, cols], seed=RNG_SEED ^ 0xC0)
    a_right = rng([rows, cols], seed=RNG_SEED ^ 0xC1)
    out = torch.cat([a_left, a_right], dim=1).contiguous()
    fixtures.append({
        "module": "kernels",
        "op": "strided_cat",
        "dtype": "float32",
        "rows": rows,
        "cols": cols,
        "input_left": to_listf(a_left),
        "input_right": to_listf(a_right),
        "shape_out": [rows, 2 * cols],
        "expected": to_listf(out),
    })

    # --- dropout: just verify output shape + seed stability (stochastic) --------
    a = rng([64], seed=RNG_SEED ^ 0xD0)
    fixtures.append({
        "module": "kernels",
        "op": "dropout",
        "dtype": "float32",
        "p": 0.5,
        "input": to_listf(a),
        "shape": [64],
        "note": "stochastic: test verifies output shape and that p% of elements zero",
    })

    # --- fused adam (single step reference) -------------------------------------
    n = 32
    param = rng([n], seed=RNG_SEED ^ 0xE0)
    grad  = rng([n], seed=RNG_SEED ^ 0xE1)
    exp_avg = torch.zeros(n)
    exp_avg_sq = torch.zeros(n)
    beta1, beta2, lr, eps, wd = 0.9, 0.999, 1e-3, 1e-8, 0.0
    # bc1 = 1 - beta1^t  for t=1 -> bc1 = 0.1
    # bc2 = 1 - beta2^t  for t=1 -> bc2 = 0.001
    bc1, bc2 = 1 - beta1, 1 - beta2
    new_ea  = beta1 * exp_avg + (1 - beta1) * grad
    new_eas = beta2 * exp_avg_sq + (1 - beta2) * grad ** 2
    denom = torch.sqrt(new_eas / bc2) + eps
    step = new_ea / bc1 / denom
    new_param = param - lr * step
    fixtures.append({
        "module": "kernels",
        "op": "fused_adam",
        "dtype": "float32",
        "n": n,
        "beta1": beta1, "beta2": beta2, "lr": lr, "eps": eps,
        "bc1": bc1, "bc2": bc2, "weight_decay": wd,
        "param": to_listf(param),
        "grad": to_listf(grad),
        "exp_avg": to_listf(exp_avg),
        "exp_avg_sq": to_listf(exp_avg_sq),
        "expected_param": to_listf(new_param),
        "expected_exp_avg": to_listf(new_ea),
        "expected_exp_avg_sq": to_listf(new_eas),
    })

    # --- fused GRU forward (single-step, batch=2) --------------------------------
    hsz, batch = 8, 2
    # GRU equations:
    #   r = sigmoid(input_r + hidden_r + b_r)
    #   z = sigmoid(input_z + hidden_z + b_z)
    #   n = tanh(input_n + r * (hidden_n + b_hn))
    #   h' = (1 - z) * n + z * hx
    # Input gates: [batch, 3*hsz] = [r_gate, z_gate, n_gate]
    ig = rng([batch * 3 * hsz], seed=RNG_SEED ^ 0xF0)
    hg = rng([batch * 3 * hsz], seed=RNG_SEED ^ 0xF1)
    bih = rng([3 * hsz], seed=RNG_SEED ^ 0xF2)
    bhh = rng([3 * hsz], seed=RNG_SEED ^ 0xF3)
    hx = rng([batch * hsz], seed=RNG_SEED ^ 0xF4)

    ig_t = ig.view(batch, 3, hsz)
    hg_t = hg.view(batch, 3, hsz)
    bih_t = bih.view(3, hsz)
    bhh_t = bhh.view(3, hsz)
    hx_t = hx.view(batch, hsz)

    rg = torch.sigmoid(ig_t[:, 0] + hg_t[:, 0] + bih_t[0] + bhh_t[0])
    zg = torch.sigmoid(ig_t[:, 1] + hg_t[:, 1] + bih_t[1] + bhh_t[1])
    ng = torch.tanh(ig_t[:, 2] + bih_t[2] + rg * (hg_t[:, 2] + bhh_t[2]))
    hy_ref = (1 - zg) * ng + zg * hx_t

    fixtures.append({
        "module": "kernels",
        "op": "fused_gru_forward",
        "dtype": "float32",
        "batch": batch,
        "hsz": hsz,
        "input_gates": to_listf(ig),
        "hidden_gates": to_listf(hg),
        "bias_ih": to_listf(bih),
        "bias_hh": to_listf(bhh),
        "hx": to_listf(hx),
        "expected_hy": to_listf(hy_ref.reshape(-1)),
    })

    # --- maxpool2d ---------------------------------------------------------------
    inp = rng([1, 1, 8, 8], seed=RNG_SEED ^ 0x100)
    out_mp = F.max_pool2d(inp, kernel_size=2, stride=2)
    fixtures.append({
        "module": "kernels",
        "op": "maxpool2d",
        "dtype": "float32",
        "input": to_listf(inp),
        "input_shape": [1, 1, 8, 8],
        "kernel_h": 2, "kernel_w": 2,
        "stride_h": 2, "stride_w": 2,
        "output_shape": list(out_mp.shape),
        "expected": to_listf(out_mp),
    })

    # --- avgpool2d ---------------------------------------------------------------
    out_ap = F.avg_pool2d(inp, kernel_size=2, stride=2)
    fixtures.append({
        "module": "kernels",
        "op": "avgpool2d",
        "dtype": "float32",
        "input": to_listf(inp),
        "input_shape": [1, 1, 8, 8],
        "kernel_h": 2, "kernel_w": 2,
        "stride_h": 2, "stride_w": 2,
        "output_shape": list(out_ap.shape),
        "expected": to_listf(out_ap),
    })

    # --- has_inf_nan -------------------------------------------------------------
    a_clean = rng([32], seed=RNG_SEED ^ 0x110)
    a_inf   = a_clean.clone(); a_inf[5] = float("inf")
    a_nan   = a_clean.clone(); a_nan[10] = float("nan")
    fixtures.append({
        "module": "kernels",
        "op": "has_inf_nan",
        "dtype": "float32",
        "input_clean": to_listf(a_clean),
        "input_inf": to_listf(a_inf),
        "input_nan": to_listf(a_nan),
        "expected_clean": False,
        "expected_inf": True,
        "expected_nan": True,
    })

    # --- fill_f32 / fill_f64 ----------------------------------------------------
    fixtures.append({
        "module": "kernels",
        "op": "fill_f32",
        "dtype": "float32",
        "n": 16,
        "scalar": 3.14159,
        "expected": [3.14159] * 16,
    })

    # --- strided_copy ------------------------------------------------------------
    # Non-contiguous input: [4, 4] transposed -> [4, 4] contiguous
    a = rng([4, 4], seed=RNG_SEED ^ 0x120)
    out = a.t().contiguous()
    fixtures.append({
        "module": "kernels",
        "op": "strided_copy",
        "dtype": "float32",
        "input": to_listf(a),
        "in_shape": [4, 4],
        "in_strides": [1, 4],  # transposed strides
        "out_shape": [4, 4],
        "expected": to_listf(out),
    })

    # --- pow --------------------------------------------------------------------
    a = torch.abs(rng([32], seed=RNG_SEED ^ 0x130)) + 0.1
    exp = 2.5
    out = torch.pow(a, exp)
    fixtures.append({
        "module": "kernels",
        "op": "pow",
        "dtype": "float32",
        "exponent": exp,
        "input_a": to_listf(a),
        "shape": [32],
        "expected": to_listf(out),
    })

    return fixtures


# ---------------------------------------------------------------------------
# Module 2: flash_attention.rs
# ---------------------------------------------------------------------------


def make_flash_attention_fixtures() -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    torch.manual_seed(RNG_SEED)

    for dtype_name, torch_dtype in [("float32", torch.float32), ("float64", torch.float64)]:
        for causal in [False, True]:
            B, H, N_q, N_k, d, d_v = 1, 2, 8, 8, 16, 16
            batch_heads = B * H

            q = torch.randn(batch_heads, N_q, d, dtype=torch_dtype,
                            generator=torch.Generator().manual_seed(RNG_SEED ^ 0x200))
            k = torch.randn(batch_heads, N_k, d, dtype=torch_dtype,
                            generator=torch.Generator().manual_seed(RNG_SEED ^ 0x201))
            v = torch.randn(batch_heads, N_k, d_v, dtype=torch_dtype,
                            generator=torch.Generator().manual_seed(RNG_SEED ^ 0x202))

            scale = 1.0 / math.sqrt(d)

            # Manual reference: avoid torch.nn.functional.scaled_dot_product_attention
            # so we control the causal mask precisely.
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [BH, N_q, N_k]
            if causal:
                # Lower triangular causal mask
                mask = torch.triu(torch.ones(N_q, N_k, dtype=torch.bool), diagonal=1)
                attn_scores = attn_scores.masked_fill(mask, float("-inf"))
            attn_weights = torch.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn_weights, v)  # [BH, N_q, d_v]

            fixtures.append({
                "module": "flash_attention",
                "op": "gpu_flash_attention",
                "dtype": dtype_name,
                "causal": causal,
                "batch_heads": batch_heads,
                "n_q": N_q,
                "n_k": N_k,
                "d": d,
                "d_v": d_v,
                "scale": scale,
                "query": to_listf(q.reshape(-1)),
                "key": to_listf(k.reshape(-1)),
                "value": to_listf(v.reshape(-1)),
                "expected": to_listf(out.reshape(-1)),
            })

    return fixtures


# ---------------------------------------------------------------------------
# Module 3: sparse.rs — cuSPARSE wrappers
# ---------------------------------------------------------------------------


def make_sparse_fixtures() -> list[dict[str, Any]]:
    """
    Reference values for the cuSPARSE wrapper functions.

    We generate CSR triples on the host (Python) and use PyTorch's sparse
    matmul as the oracle. For structural ops (to_dense, from_dense, format
    conversions) we construct the expected output manually.
    """
    fixtures: list[dict[str, Any]] = []
    torch.manual_seed(RNG_SEED ^ 0x300)

    # --- spmm_csr_f32 -----------------------------------------------------------
    # A: 4x4 sparse CSR, B: 4x4 dense -> C: 4x4 dense
    m, k, n = 4, 4, 4
    density = 0.5
    A_dense = torch.zeros(m, k)
    for r in range(m):
        for c in range(k):
            if torch.rand(1).item() < density:
                A_dense[r, c] = torch.randn(1).item()
    B_dense = torch.randn(k, n, generator=torch.Generator().manual_seed(RNG_SEED ^ 0x301))

    C_dense = A_dense @ B_dense

    # Build CSR from A_dense
    A_sp = A_dense.to_sparse_csr()
    crow = A_sp.crow_indices().tolist()
    col  = A_sp.col_indices().tolist()
    vals = A_sp.values().tolist()

    fixtures.append({
        "module": "sparse",
        "op": "spmm_csr_f32",
        "dtype": "float32",
        "m": m, "k": k, "n": n,
        "crow_indices": [int(x) for x in crow],
        "col_indices":  [int(x) for x in col],
        "values":       [float(x) for x in vals],
        "dense":        to_listf(B_dense.reshape(-1)),
        "expected":     to_listf(C_dense.reshape(-1)),
    })

    # spmm_csr_f64 variant
    A_dense64 = A_dense.to(torch.float64)
    B_dense64 = B_dense.to(torch.float64)
    C_dense64 = A_dense64 @ B_dense64
    fixtures.append({
        "module": "sparse",
        "op": "spmm_csr_f64",
        "dtype": "float64",
        "m": m, "k": k, "n": n,
        "crow_indices": [int(x) for x in crow],
        "col_indices":  [int(x) for x in col],
        "values":       [float(x) for x in A_sp.values().to(torch.float64).tolist()],
        "dense":        to_listf(B_dense64.reshape(-1)),
        "expected":     to_listf(C_dense64.reshape(-1)),
    })

    # --- sparse_to_dense_csr_f32 ------------------------------------------------
    # Upload a CSR matrix to GPU and convert to dense.
    fixtures.append({
        "module": "sparse",
        "op": "sparse_to_dense_csr_f32",
        "dtype": "float32",
        "m": m, "n": k,
        "crow_indices": [int(x) for x in crow],
        "col_indices":  [int(x) for x in col],
        "values":       [float(x) for x in vals],
        "expected":     to_listf(A_dense.reshape(-1)),
    })

    # --- dense_to_sparse_csr_f32 ------------------------------------------------
    # Upload dense to GPU and extract CSR structure.
    fixtures.append({
        "module": "sparse",
        "op": "dense_to_sparse_csr_f32",
        "dtype": "float32",
        "m": m, "n": k,
        "dense": to_listf(A_dense.reshape(-1)),
        "expected_crow": [int(x) for x in crow],
        "expected_col":  [int(x) for x in col],
        "expected_vals": [float(x) for x in vals],
    })

    # --- csr_to_csc_f32 ---------------------------------------------------------
    # CSR to CSC conversion. Reference: transpose-then-transpose trick.
    # CSC of A = CSR of A^T
    A_csc_ref = A_dense.t().to_sparse_csr()
    csc_crow = A_csc_ref.crow_indices().tolist()
    csc_col  = A_csc_ref.col_indices().tolist()
    csc_vals = A_csc_ref.values().tolist()
    fixtures.append({
        "module": "sparse",
        "op": "csr_to_csc_f32",
        "dtype": "float32",
        "m": m, "n": k,
        "crow_indices": [int(x) for x in crow],
        "col_indices":  [int(x) for x in col],
        "values":       [float(x) for x in vals],
        # expected CSC expressed as (col_ptr, row_indices, values)
        # ferrotorch's CscTensor stores: col_ptr = crow of A^T, row_idx = col of A^T
        "expected_col_ptr":  [int(x) for x in csc_crow],
        "expected_row_idx":  [int(x) for x in csc_col],
        "expected_csc_vals": [float(x) for x in csc_vals],
    })

    # --- coo_to_csr_f32 ---------------------------------------------------------
    # Build a COO representation from the same matrix.
    nnz = len(vals)
    row_indices: list[int] = []
    for r_i in range(m):
        row_start = int(crow[r_i])
        row_end   = int(crow[r_i + 1])
        row_indices.extend([r_i] * (row_end - row_start))
    fixtures.append({
        "module": "sparse",
        "op": "coo_to_csr_f32",
        "dtype": "float32",
        "m": m, "n": k,
        "row_indices": row_indices,
        "col_indices": [int(x) for x in col],
        "values":      [float(x) for x in vals],
        "expected_crow": [int(x) for x in crow],
        "expected_col":  [int(x) for x in col],
        "expected_vals": [float(x) for x in vals],
    })

    # --- csc_to_dense_f32 -------------------------------------------------------
    # Reconstruct dense from CSC.
    fixtures.append({
        "module": "sparse",
        "op": "csc_to_dense_f32",
        "dtype": "float32",
        "m": m, "n": k,
        "col_ptr":   [int(x) for x in csc_crow],
        "row_idx":   [int(x) for x in csc_col],
        "csc_vals":  [float(x) for x in csc_vals],
        "expected":  to_listf(A_dense.reshape(-1)),
    })

    # --- csr_to_coo_f32 ---------------------------------------------------------
    fixtures.append({
        "module": "sparse",
        "op": "csr_to_coo_f32",
        "dtype": "float32",
        "m": m, "n": k,
        "crow_indices": [int(x) for x in crow],
        "col_indices":  [int(x) for x in col],
        "values":       [float(x) for x in vals],
        "expected_row": row_indices,
        "expected_col": [int(x) for x in col],
        "expected_vals": [float(x) for x in vals],
    })

    return fixtures


# ---------------------------------------------------------------------------
# Module 4: conv.rs — gpu_conv2d_f32
# ---------------------------------------------------------------------------


def make_conv_fixtures() -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    torch.manual_seed(RNG_SEED ^ 0x400)

    # --- conv2d no-bias ----------------------------------------------------------
    B, C_in, H, W = 1, 1, 5, 5
    C_out, kH, kW = 1, 3, 3
    stride, pad = (1, 1), (0, 0)

    inp   = torch.randn(B, C_in, H, W, generator=torch.Generator().manual_seed(RNG_SEED ^ 0x401))
    wt    = torch.randn(C_out, C_in, kH, kW, generator=torch.Generator().manual_seed(RNG_SEED ^ 0x402))
    out   = F.conv2d(inp, wt, bias=None, stride=stride, padding=pad)

    H_out = (H + 2 * pad[0] - kH) // stride[0] + 1
    W_out = (W + 2 * pad[1] - kW) // stride[1] + 1

    fixtures.append({
        "module": "conv",
        "op": "conv2d_no_bias",
        "dtype": "float32",
        "input_shape": [B, C_in, H, W],
        "weight_shape": [C_out, C_in, kH, kW],
        "stride": list(stride),
        "padding": list(pad),
        "has_bias": False,
        "input":  to_listf(inp.reshape(-1)),
        "weight": to_listf(wt.reshape(-1)),
        "output_shape": [B, C_out, H_out, W_out],
        "expected": to_listf(out.reshape(-1)),
    })

    # --- conv2d with bias --------------------------------------------------------
    bias_data = torch.randn(C_out, generator=torch.Generator().manual_seed(RNG_SEED ^ 0x403))
    out_bias  = F.conv2d(inp, wt, bias=bias_data, stride=stride, padding=pad)
    fixtures.append({
        "module": "conv",
        "op": "conv2d_with_bias",
        "dtype": "float32",
        "input_shape": [B, C_in, H, W],
        "weight_shape": [C_out, C_in, kH, kW],
        "stride": list(stride),
        "padding": list(pad),
        "has_bias": True,
        "input":  to_listf(inp.reshape(-1)),
        "weight": to_listf(wt.reshape(-1)),
        "bias":   to_listf(bias_data),
        "output_shape": [B, C_out, H_out, W_out],
        "expected": to_listf(out_bias.reshape(-1)),
    })

    # --- conv2d with padding -----------------------------------------------------
    pad2 = (1, 1)
    H_out2 = (H + 2 * pad2[0] - kH) // stride[0] + 1
    W_out2 = (W + 2 * pad2[1] - kW) // stride[1] + 1
    out_p   = F.conv2d(inp, wt, bias=None, stride=stride, padding=pad2)
    fixtures.append({
        "module": "conv",
        "op": "conv2d_padded",
        "dtype": "float32",
        "input_shape": [B, C_in, H, W],
        "weight_shape": [C_out, C_in, kH, kW],
        "stride": list(stride),
        "padding": list(pad2),
        "has_bias": False,
        "input":  to_listf(inp.reshape(-1)),
        "weight": to_listf(wt.reshape(-1)),
        "output_shape": [B, C_out, H_out2, W_out2],
        "expected": to_listf(out_p.reshape(-1)),
    })

    # --- multi-channel conv2d (C_in=3, C_out=2) ----------------------------------
    B2, C_in2, H2, W2 = 2, 3, 7, 7
    C_out2, kH2, kW2  = 2, 3, 3
    inp2  = torch.randn(B2, C_in2, H2, W2, generator=torch.Generator().manual_seed(RNG_SEED ^ 0x410))
    wt2   = torch.randn(C_out2, C_in2, kH2, kW2, generator=torch.Generator().manual_seed(RNG_SEED ^ 0x411))
    out2  = F.conv2d(inp2, wt2, bias=None, stride=(1, 1), padding=(0, 0))
    H_out3 = (H2 + 2 * 0 - kH2) // 1 + 1
    W_out3 = (W2 + 2 * 0 - kW2) // 1 + 1
    fixtures.append({
        "module": "conv",
        "op": "conv2d_multichannel",
        "dtype": "float32",
        "input_shape": [B2, C_in2, H2, W2],
        "weight_shape": [C_out2, C_in2, kH2, kW2],
        "stride": [1, 1],
        "padding": [0, 0],
        "has_bias": False,
        "input":  to_listf(inp2.reshape(-1)),
        "weight": to_listf(wt2.reshape(-1)),
        "output_shape": [B2, C_out2, H_out3, W_out3],
        "expected": to_listf(out2.reshape(-1)),
    })

    return fixtures


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"[regenerate_gpu_kernels_fixtures] torch {torch.__version__}, "
          f"CUDA: {torch.cuda.is_available()}", flush=True)

    all_fixtures: list[dict[str, Any]] = []
    all_fixtures.extend(make_kernels_fixtures())
    all_fixtures.extend(make_flash_attention_fixtures())
    all_fixtures.extend(make_sparse_fixtures())
    all_fixtures.extend(make_conv_fixtures())

    output = {
        "metadata": fixture_metadata(),
        "fixtures": all_fixtures,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    n = len(all_fixtures)
    print(f"[regenerate_gpu_kernels_fixtures] wrote {n} fixtures to {FIXTURE_PATH}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
