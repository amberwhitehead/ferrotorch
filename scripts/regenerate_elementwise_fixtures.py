#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for ferrotorch-core Phase 2.1
(elementwise + inplace ops).

Tracking issue: #763 (parent: #759).

Output:
    ferrotorch-core/tests/conformance/fixtures/elementwise.json

Coverage:
* Cat A — differentiable arithmetic (add/sub/mul/div/neg/abs/pow/sqrt):
  CPU + (when available) CUDA, forward + autograd grads, with broadcast
  configurations. Edge cases: pow(x, 0), sqrt(0), div(x, 0).
* Cat B — comparison/where_, where_bt: forward + autograd grads, CPU + CUDA.
* Cat C — higher-order utilities (binary_map / scalar_map / unary_map): a
  non-trivial closure (max-min) is tabulated as the manual reference.
* Cat D — perf variants (fast_add/sub/mul/div, fast_exp/log/sigmoid/tanh/sin/
  cos, simd_*): each compared to the canonical torch op for parity.
* Cat E — reductions (sum, sum_axis, mean, nansum, nanmean, logsumexp,
  logsumexp_dim) including a NaN test and a numerical-stability test for
  logsumexp.
* Cat F — in-place mutation (add_/sub_/mul_/div_/add_scalar_/mul_scalar_/
  fill_/zero_/clamp_): forward conformance.

Usage from WSL (preferred per #777):

    python3 scripts/regenerate_elementwise_fixtures.py

Required Python deps: torch (with CUDA), numpy.
"""

from __future__ import annotations

import datetime
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import torch  # type: ignore

# ---------------------------------------------------------------------------
# Output path and metadata
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = (
    REPO_ROOT
    / "ferrotorch-core"
    / "tests"
    / "conformance"
    / "fixtures"
    / "elementwise.json"
)

DTYPES: list[str] = ["float32", "float64"]
DEVICES: list[str] = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda:0")

RNG_SEED: int = 0xBADCAFE
torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RNG_SEED)


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def to_listf(t: torch.Tensor) -> list[Any]:
    """Materialize a tensor to a CPU Python list of floats.

    Special values (+inf / -inf / NaN) are encoded as string sentinels so the
    output remains strict-JSON-compliant — ``serde_json`` (used by the Rust
    test) rejects the bare ``Infinity`` / ``NaN`` tokens that Python's
    ``json.dump`` emits by default.
    """
    raw = t.detach().to("cpu").to(torch.float64).reshape(-1).tolist()
    encoded: list[Any] = []
    for v in raw:
        if math.isnan(v):
            encoded.append("NaN")
        elif math.isinf(v):
            encoded.append("Infinity" if v > 0 else "-Infinity")
        else:
            encoded.append(v)
    return encoded


def fixture_metadata() -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rng_seed": RNG_SEED,
        "dtypes": DTYPES,
        "devices": DEVICES,
    }


# ---------------------------------------------------------------------------
# Cat A — differentiable arithmetic ops
# ---------------------------------------------------------------------------
#
# Each entry records: op name, input shapes (a, b), input data, expected
# forward values + grads-w.r.t.-each-input under loss = sum(out).

# Shape pairs covering: same-shape baseline + 4 broadcast cases.
BROADCAST_PAIRS: list[tuple[list[int], list[int], str]] = [
    ([3], [3], "same1d"),
    ([2, 3], [2, 3], "same2d"),
    ([3], [3, 3], "broadcast_1to2"),
    ([1], [3], "scalar_to_vec"),
    ([3, 1], [1, 3], "outer"),
]


def _seeded(shape: list[int], dtype: str, device: str, base: float) -> torch.Tensor:
    """A small deterministic tensor with positive values around `base`."""
    n = max(1, math.prod(shape) if shape else 1)
    vals = [base + i * 0.5 for i in range(n)]
    return torch.tensor(vals, dtype=torch_dtype(dtype), device=device).reshape(shape)


def _arith_pair(
    a_shape: list[int],
    b_shape: list[int],
    dtype: str,
    device: str,
    op: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    a = _seeded(a_shape, dtype, device, 1.0)
    if op in ("div", "sqrt"):
        # Avoid div-by-zero / sqrt-of-negative: keep both inputs strictly > 0.
        b = _seeded(b_shape, dtype, device, 2.0)
    elif op == "pow":
        # `pow` is unary in ferrotorch (scalar exponent); b unused.
        b = _seeded(b_shape, dtype, device, 0.5)
    else:
        b = _seeded(b_shape, dtype, device, 0.5)
    return a, b


def _binary_op(name: str):
    return {
        "add": torch.add,
        "sub": torch.sub,
        "mul": torch.mul,
        "div": torch.div,
    }[name]


def fixture_cat_a_binary() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for op in ("add", "sub", "mul", "div"):
        for device in DEVICES:
            for dtype in DTYPES:
                for a_shape, b_shape, tag in BROADCAST_PAIRS:
                    a, b = _arith_pair(a_shape, b_shape, dtype, device, op)
                    a_g = a.detach().clone().requires_grad_(True)
                    b_g = b.detach().clone().requires_grad_(True)
                    fwd = _binary_op(op)(a_g, b_g)
                    loss = fwd.sum()
                    loss.backward()
                    out.append(
                        {
                            "op": op,
                            "tag": tag,
                            "dtype": dtype,
                            "device": device,
                            "a_shape": a_shape,
                            "b_shape": b_shape,
                            "a_data": to_listf(a),
                            "b_data": to_listf(b),
                            "out_shape": list(fwd.shape),
                            "out_values": to_listf(fwd),
                            "grad_a": to_listf(a_g.grad),
                            "grad_b": to_listf(b_g.grad),
                        }
                    )
    return out


def fixture_cat_a_unary() -> list[dict[str, Any]]:
    """neg / abs / sqrt fixtures (no broadcast — unary)."""
    out: list[dict[str, Any]] = []
    for op in ("neg", "abs", "sqrt"):
        for device in DEVICES:
            for dtype in DTYPES:
                for shape, tag in (
                    ([3], "vec"),
                    ([2, 3], "mat"),
                    ([1], "scalar1"),
                ):
                    if op == "abs":
                        # exercise the negative-input path
                        n = max(1, math.prod(shape) if shape else 1)
                        vals = [(-1.0) ** i * (i + 1) * 0.5 for i in range(n)]
                        a = torch.tensor(
                            vals, dtype=torch_dtype(dtype), device=device
                        ).reshape(shape)
                    elif op == "sqrt":
                        a = _seeded(shape, dtype, device, 1.0)
                    else:  # neg
                        a = _seeded(shape, dtype, device, -1.0)
                    a_g = a.detach().clone().requires_grad_(True)
                    if op == "neg":
                        fwd = -a_g
                    elif op == "abs":
                        fwd = torch.abs(a_g)
                    else:
                        fwd = torch.sqrt(a_g)
                    loss = fwd.sum()
                    loss.backward()
                    out.append(
                        {
                            "op": op,
                            "tag": tag,
                            "dtype": dtype,
                            "device": device,
                            "a_shape": shape,
                            "a_data": to_listf(a),
                            "out_shape": list(fwd.shape),
                            "out_values": to_listf(fwd),
                            "grad_a": to_listf(a_g.grad),
                        }
                    )
    return out


def fixture_cat_a_pow() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            for shape, tag in (([3], "vec"), ([2, 3], "mat")):
                for exp in (0.0, 1.0, 2.0, 3.0, 0.5, -1.0):
                    a = _seeded(shape, dtype, device, 1.0)
                    a_g = a.detach().clone().requires_grad_(True)
                    fwd = torch.pow(a_g, exp)
                    loss = fwd.sum()
                    loss.backward()
                    out.append(
                        {
                            "op": "pow",
                            "tag": tag,
                            "dtype": dtype,
                            "device": device,
                            "a_shape": shape,
                            "exp": exp,
                            "a_data": to_listf(a),
                            "out_shape": list(fwd.shape),
                            "out_values": to_listf(fwd),
                            "grad_a": to_listf(a_g.grad),
                        }
                    )
    return out


def fixture_cat_a_edge_cases() -> list[dict[str, Any]]:
    """Documented edge cases: pow(x, 0)=1, sqrt(0)=0, div(x, 0)=±inf."""
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            # pow(x, 0) = 1 for any finite x.
            a = torch.tensor(
                [1.0, 2.0, -3.0, 0.5], dtype=torch_dtype(dtype), device=device
            )
            fwd = torch.pow(a, 0.0)
            out.append(
                {
                    "op": "pow_zero_exp",
                    "tag": "edge",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [4],
                    "a_data": to_listf(a),
                    "out_values": to_listf(fwd),
                }
            )

            # sqrt(0) = 0
            a = torch.tensor(
                [0.0, 1.0, 4.0], dtype=torch_dtype(dtype), device=device
            )
            fwd = torch.sqrt(a)
            out.append(
                {
                    "op": "sqrt_zero",
                    "tag": "edge",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [3],
                    "a_data": to_listf(a),
                    "out_values": to_listf(fwd),
                }
            )

            # div(x, 0): per IEEE 754, x/0 = ±inf for finite nonzero x; 0/0 = NaN.
            a = torch.tensor(
                [1.0, -1.0, 0.0], dtype=torch_dtype(dtype), device=device
            )
            b = torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch_dtype(dtype), device=device
            )
            fwd = torch.div(a, b)
            out.append(
                {
                    "op": "div_zero",
                    "tag": "edge",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [3],
                    "b_shape": [3],
                    "a_data": to_listf(a),
                    "b_data": to_listf(b),
                    "out_values": to_listf(fwd),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Cat B — where_ / where_bt
# ---------------------------------------------------------------------------


def fixture_cat_b_where() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            cond = [True, False, True, False]
            x_vals = [1.0, 2.0, 3.0, 4.0]
            y_vals = [10.0, 20.0, 30.0, 40.0]
            cond_t = torch.tensor(cond, dtype=torch.bool, device=device)
            x = torch.tensor(x_vals, dtype=torch_dtype(dtype), device=device)
            y = torch.tensor(y_vals, dtype=torch_dtype(dtype), device=device)
            x_g = x.detach().clone().requires_grad_(True)
            y_g = y.detach().clone().requires_grad_(True)
            fwd = torch.where(cond_t, x_g, y_g)
            loss = fwd.sum()
            loss.backward()
            out.append(
                {
                    "op": "where",
                    "tag": "vec4",
                    "dtype": dtype,
                    "device": device,
                    "cond": cond,
                    "x_shape": [4],
                    "y_shape": [4],
                    "x_data": x_vals,
                    "y_data": y_vals,
                    "out_values": to_listf(fwd),
                    "grad_x": to_listf(x_g.grad),
                    "grad_y": to_listf(y_g.grad),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Cat C — higher-order utilities (CPU only by design)
# ---------------------------------------------------------------------------
#
# Reference is the manual computed result for a non-trivial closure:
# * `binary_map(a, b, |x, y| max(x, y) - min(x, y))` = |x - y|
# * `scalar_map(a, s, |x, s| x*x + s)` = x^2 + s
# * `unary_map(a, |x| x.tan())` = tan(x)


def fixture_cat_c_higher_order() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dtype in DTYPES:
        # binary_map: a non-trivial closure.
        a = torch.tensor([1.0, -2.0, 3.0, 0.0], dtype=torch_dtype(dtype))
        b = torch.tensor([0.5, 1.0, -1.0, 4.0], dtype=torch_dtype(dtype))
        # closure: max(x,y) - min(x,y) == |x - y|
        ref = torch.abs(a - b)
        out.append(
            {
                "op": "binary_map_maxmin",
                "tag": "closure",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": [4],
                "b_shape": [4],
                "a_data": to_listf(a),
                "b_data": to_listf(b),
                "out_values": to_listf(ref),
            }
        )

        # scalar_map closure x^2 + s
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch_dtype(dtype))
        scalar = 2.5
        ref = a * a + scalar
        out.append(
            {
                "op": "scalar_map_sqplus",
                "tag": "closure",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": [3],
                "scalar": scalar,
                "a_data": to_listf(a),
                "out_values": to_listf(ref),
            }
        )

        # unary_map closure tan(x)
        a = torch.tensor([0.0, 0.5, -0.5, 1.0], dtype=torch_dtype(dtype))
        ref = torch.tan(a)
        out.append(
            {
                "op": "unary_map_tan",
                "tag": "closure",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": [4],
                "a_data": to_listf(a),
                "out_values": to_listf(ref),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Cat D — perf variants (fast_*, simd_*) — CPU only by design
# ---------------------------------------------------------------------------


def fixture_cat_d_perf() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # Inputs sized to exercise both the small (<32K) and parallel (>32K)
    # paths in fast_*. We keep the JSON small by recording a moderate-size
    # input + output; the parallel-path test in Rust uses a generated input
    # without consulting fixtures.
    for dtype in DTYPES:
        # Binary fast_* ops parity vs torch:
        a = torch.tensor(
            [0.5, 1.5, -2.0, 3.25, 0.0, -1.25],
            dtype=torch_dtype(dtype),
        )
        b = torch.tensor(
            [1.0, -0.5, 2.0, -3.0, 4.0, 1.5],
            dtype=torch_dtype(dtype),
        )
        for op_name, ref_op in (
            ("fast_add", torch.add),
            ("fast_sub", torch.sub),
            ("fast_mul", torch.mul),
            ("fast_div", torch.div),
        ):
            ref = ref_op(a, b)
            out.append(
                {
                    "op": op_name,
                    "tag": "vec6",
                    "dtype": dtype,
                    "device": "cpu",
                    "a_shape": [6],
                    "b_shape": [6],
                    "a_data": to_listf(a),
                    "b_data": to_listf(b),
                    "out_values": to_listf(ref),
                }
            )

        # Unary fast_* parity vs torch.
        a_pos = torch.tensor(
            [0.1, 0.5, 1.0, 2.0, 0.25, 1.5],
            dtype=torch_dtype(dtype),
        )
        a_any = torch.tensor(
            [0.0, 0.5, -0.5, 1.0, -1.0, 2.0],
            dtype=torch_dtype(dtype),
        )
        for op_name, src, ref in (
            ("fast_exp", a_any, torch.exp(a_any)),
            ("fast_log", a_pos, torch.log(a_pos)),
            ("fast_sigmoid", a_any, torch.sigmoid(a_any)),
            ("fast_tanh", a_any, torch.tanh(a_any)),
            ("fast_sin", a_any, torch.sin(a_any)),
            ("fast_cos", a_any, torch.cos(a_any)),
        ):
            out.append(
                {
                    "op": op_name,
                    "tag": "vec6",
                    "dtype": dtype,
                    "device": "cpu",
                    "a_shape": [6],
                    "a_data": to_listf(src),
                    "out_values": to_listf(ref),
                }
            )

    # SIMD ops are dtype-specific. simd_*_f32 is f32 only; simd_*_f64 is f64.
    a32 = torch.tensor(
        [0.5, 1.5, -2.0, 3.25, 0.0, -1.25, 4.0, 0.125], dtype=torch.float32
    )
    b32 = torch.tensor(
        [1.0, -0.5, 2.0, -3.0, 4.0, 1.5, 0.25, 6.0], dtype=torch.float32
    )
    a32_pos = torch.tensor(
        [0.1, 0.5, 1.0, 2.0, 0.25, 1.5, 3.5, 0.75], dtype=torch.float32
    )
    a32_any = torch.tensor(
        [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.25], dtype=torch.float32
    )
    a64 = a32.to(torch.float64)
    b64 = b32.to(torch.float64)
    a64_any = a32_any.to(torch.float64)

    for op_name, src_a, src_b, dtype, ref in (
        ("simd_add_f32", a32, b32, "float32", a32 + b32),
        ("simd_mul_f32", a32, b32, "float32", a32 * b32),
        ("simd_exp_f32", a32_any, None, "float32", torch.exp(a32_any)),
        ("simd_log_f32", a32_pos, None, "float32", torch.log(a32_pos)),
        ("simd_sqrt_f32", a32_pos, None, "float32", torch.sqrt(a32_pos)),
        ("simd_add_f64", a64, b64, "float64", a64 + b64),
        ("simd_mul_f64", a64, b64, "float64", a64 * b64),
        ("simd_exp_f64", a64_any, None, "float64", torch.exp(a64_any)),
    ):
        entry: dict[str, Any] = {
            "op": op_name,
            "tag": "vec8",
            "dtype": dtype,
            "device": "cpu",
            "a_shape": list(src_a.shape),
            "a_data": to_listf(src_a),
            "out_values": to_listf(ref),
        }
        if src_b is not None:
            entry["b_shape"] = list(src_b.shape)
            entry["b_data"] = to_listf(src_b)
        out.append(entry)

    return out


# ---------------------------------------------------------------------------
# Cat E — reductions (sum, sum_axis, mean, nansum, nanmean, logsumexp,
# logsumexp_dim)
# ---------------------------------------------------------------------------


def fixture_cat_e_reductions() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            # sum (full reduction) + autograd
            a = torch.tensor(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                dtype=torch_dtype(dtype),
                device=device,
            ).reshape(2, 3)
            a_g = a.detach().clone().requires_grad_(True)
            fwd = a_g.sum()
            fwd.backward()
            out.append(
                {
                    "op": "sum",
                    "tag": "mat23",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [2, 3],
                    "a_data": to_listf(a),
                    "out_values": to_listf(fwd),
                    "grad_a": to_listf(a_g.grad),
                }
            )

            # sum_axis: torch.sum(t, dim=0) and dim=1
            for axis in (0, 1):
                a = torch.tensor(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    dtype=torch_dtype(dtype),
                    device=device,
                ).reshape(2, 3)
                fwd = torch.sum(a, dim=axis)
                out.append(
                    {
                        "op": "sum_axis",
                        "tag": f"axis{axis}",
                        "dtype": dtype,
                        "device": device,
                        "axis": axis,
                        "a_shape": [2, 3],
                        "a_data": to_listf(a),
                        "out_shape": list(fwd.shape),
                        "out_values": to_listf(fwd),
                    }
                )

            # mean (full reduction) + autograd
            a = torch.tensor(
                [2.0, 4.0, 6.0, 8.0],
                dtype=torch_dtype(dtype),
                device=device,
            )
            a_g = a.detach().clone().requires_grad_(True)
            fwd = a_g.mean()
            fwd.backward()
            out.append(
                {
                    "op": "mean",
                    "tag": "vec4",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [4],
                    "a_data": to_listf(a),
                    "out_values": to_listf(fwd),
                    "grad_a": to_listf(a_g.grad),
                }
            )

            # nansum: NaN treated as zero. Skip CUDA — ferrotorch returns
            # NotImplementedOnCuda for nansum, so the fixture only describes
            # the CPU behaviour.
            if device == "cpu":
                a = torch.tensor(
                    [1.0, float("nan"), 3.0, float("nan"), 5.0],
                    dtype=torch_dtype(dtype),
                    device=device,
                )
                fwd = torch.nansum(a)
                out.append(
                    {
                        "op": "nansum",
                        "tag": "withnan",
                        "dtype": dtype,
                        "device": device,
                        "a_shape": [5],
                        "a_data": to_listf(a),
                        "out_values": to_listf(fwd),
                    }
                )

                # nanmean: NaN excluded from both sum and count.
                fwd = torch.nanmean(a)
                out.append(
                    {
                        "op": "nanmean",
                        "tag": "withnan",
                        "dtype": dtype,
                        "device": device,
                        "a_shape": [5],
                        "a_data": to_listf(a),
                        "out_values": to_listf(fwd),
                    }
                )

                # logsumexp: numerical stability test. Use moderate values
                # to keep float32 precision meaningful (log(2) is well below
                # 1 ulp of 1e10 in f32). For 100.0+log(2) the result is
                # representable and exercises the max-subtract trick.
                a = torch.tensor(
                    [100.0, 100.0],
                    dtype=torch_dtype(dtype),
                    device=device,
                )
                fwd = torch.logsumexp(a, dim=0)
                out.append(
                    {
                        "op": "logsumexp",
                        "tag": "stable100",
                        "dtype": dtype,
                        "device": device,
                        "a_shape": [2],
                        "a_data": to_listf(a),
                        "out_values": to_listf(fwd),
                    }
                )

                # logsumexp on a normal-magnitude vector for the basic case.
                a = torch.tensor(
                    [0.0, 1.0, 2.0, 3.0],
                    dtype=torch_dtype(dtype),
                    device=device,
                )
                fwd = torch.logsumexp(a, dim=0)
                out.append(
                    {
                        "op": "logsumexp",
                        "tag": "vec4",
                        "dtype": dtype,
                        "device": device,
                        "a_shape": [4],
                        "a_data": to_listf(a),
                        "out_values": to_listf(fwd),
                    }
                )

                # logsumexp_dim: along a specific dim of a 2-D matrix.
                a = torch.tensor(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    dtype=torch_dtype(dtype),
                    device=device,
                ).reshape(2, 3)
                for axis in (0, 1):
                    for keepdim in (False, True):
                        fwd = torch.logsumexp(a, dim=axis, keepdim=keepdim)
                        out.append(
                            {
                                "op": "logsumexp_dim",
                                "tag": f"axis{axis}_keepdim{int(keepdim)}",
                                "dtype": dtype,
                                "device": device,
                                "axis": axis,
                                "keepdim": keepdim,
                                "a_shape": [2, 3],
                                "a_data": to_listf(a),
                                "out_shape": list(fwd.shape),
                                "out_values": to_listf(fwd),
                            }
                        )

    return out


# ---------------------------------------------------------------------------
# Cat F — in-place mutation
# ---------------------------------------------------------------------------


def fixture_cat_f_inplace() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            # add_, sub_, mul_, div_: tensor-tensor inplace
            a = torch.tensor(
                [1.0, 2.0, 3.0, 4.0],
                dtype=torch_dtype(dtype),
                device=device,
            )
            b = torch.tensor(
                [0.5, 1.0, 1.5, 2.0],
                dtype=torch_dtype(dtype),
                device=device,
            )
            for op_name, mutator in (
                ("add_", lambda t, o: t.add_(o)),
                ("sub_", lambda t, o: t.sub_(o)),
                ("mul_", lambda t, o: t.mul_(o)),
                ("div_", lambda t, o: t.div_(o)),
            ):
                t = a.clone()
                mutator(t, b)
                out.append(
                    {
                        "op": op_name,
                        "tag": "vec4",
                        "dtype": dtype,
                        "device": device,
                        "a_shape": [4],
                        "b_shape": [4],
                        "a_data": to_listf(a),
                        "b_data": to_listf(b),
                        "out_values": to_listf(t),
                    }
                )

            # add_scalar_, mul_scalar_
            a = torch.tensor(
                [1.0, 2.0, 3.0],
                dtype=torch_dtype(dtype),
                device=device,
            )
            t = a.clone()
            t.add_(2.5)  # PyTorch uses .add_ for both tensor and scalar
            out.append(
                {
                    "op": "add_scalar_",
                    "tag": "vec3",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [3],
                    "scalar": 2.5,
                    "a_data": to_listf(a),
                    "out_values": to_listf(t),
                }
            )
            t = a.clone()
            t.mul_(3.0)
            out.append(
                {
                    "op": "mul_scalar_",
                    "tag": "vec3",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [3],
                    "scalar": 3.0,
                    "a_data": to_listf(a),
                    "out_values": to_listf(t),
                }
            )

            # fill_
            t = a.clone()
            t.fill_(7.5)
            out.append(
                {
                    "op": "fill_",
                    "tag": "vec3",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [3],
                    "scalar": 7.5,
                    "a_data": to_listf(a),
                    "out_values": to_listf(t),
                }
            )

            # zero_
            t = a.clone()
            t.zero_()
            out.append(
                {
                    "op": "zero_",
                    "tag": "vec3",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [3],
                    "a_data": to_listf(a),
                    "out_values": to_listf(t),
                }
            )

            # clamp_: bracket some values inside, some outside [min, max].
            a_clamp = torch.tensor(
                [-2.0, -0.5, 0.0, 0.5, 2.0],
                dtype=torch_dtype(dtype),
                device=device,
            )
            t = a_clamp.clone()
            t.clamp_(-1.0, 1.0)
            out.append(
                {
                    "op": "clamp_",
                    "tag": "vec5",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [5],
                    "min": -1.0,
                    "max": 1.0,
                    "a_data": to_listf(a_clamp),
                    "out_values": to_listf(t),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def main() -> int:
    fixtures: list[dict[str, Any]] = []
    fixtures += fixture_cat_a_binary()
    fixtures += fixture_cat_a_unary()
    fixtures += fixture_cat_a_pow()
    fixtures += fixture_cat_a_edge_cases()
    fixtures += fixture_cat_b_where()
    fixtures += fixture_cat_c_higher_order()
    fixtures += fixture_cat_d_perf()
    fixtures += fixture_cat_e_reductions()
    fixtures += fixture_cat_f_inplace()

    payload = {
        "metadata": fixture_metadata(),
        "fixtures": fixtures,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(
        f"wrote {len(fixtures)} fixtures to {FIXTURE_PATH.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
