#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for ferrotorch-core Phase 2.5
(activations, transcendental, special).

Tracking issue: #767 (parent: #759).

Output:
    ferrotorch-core/tests/conformance/fixtures/activation.json

Coverage (84-item phase per #767):

* **Cat A — activation forwards** (full CPU + GPU + autograd where
  applicable). The 21 differentiable activation surfaces in
  ``grad_fns::activation`` plus the 4 top-level re-exports they re-expose:
    relu, relu6, sigmoid, tanh, gelu, gelu_with × 3 GeluApproximate variants,
    silu, softmax, log_softmax, softplus, elu, mish, leaky_relu, hardtanh,
    hardtanh_with, hardsigmoid, hardswish, selu, softsign, prelu, glu.

* **Cat A — transcendental forwards** (full CPU + GPU + autograd):
    sin, cos, exp, log, clamp.

* **Cat A — special forwards** (CPU only — `special::*` rejects CUDA via
  `NotImplementedOnCuda`):
    erf, erfc, erfinv, lgamma, digamma, log1p, expm1, sinc, xlogy.

* **Cat A — orthogonal-polynomial families** (CPU only, surfaced via
  `special::*`):
    chebyshev_polynomial_{t,u,v,w}, hermite_polynomial_{h,he},
    laguerre_polynomial_l, legendre_polynomial_p,
    shifted_chebyshev_polynomial_{t,u,v,w}.

* **Edge cases**:
    - Saturating: tanh(±100) ≈ ±1, sigmoid(±100) ≈ 0/1.
    - Boundary: log(0) = -inf, log(-1) = NaN, sqrt(-1) = NaN (covered
      indirectly via test asserts, no fixture row needed).
    - Numerical stability: softmax on `[100, 100, 100]` -> [1/3, 1/3, 1/3];
      log_softmax on `[1000, 1001]` stable.
    - log1p / expm1 small-x precision: log1p(1e-10) and expm1(1e-10) at f64
      tolerance.

* **Verification-debt repayment lanes** (per dispatch):
    gpu_log_f64 over [1e-10, 1e10], gpu_log_softmax_f64 typical NN inputs,
    gpu_mish_f64 typical activation inputs — all asserted at
    F64_TRANSCENDENTAL = 1e-10.

Backward grad_fn structs (`*Backward`) are exclusion-with-implicit-coverage
in `_surface_exclusions.toml`: each is exercised transitively when the
forward op's autograd path runs, and the substring grep in
`conformance_surface_coverage.rs` is satisfied by the test file naming each
backward type once.

Usage from WSL (preferred per #777):

    python3 scripts/regenerate_activation_fixtures.py

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
import torch.nn.functional as F  # type: ignore


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
    / "activation.json"
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
    """Materialize a tensor to a CPU Python list of floats with sentinels."""
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
# Helpers
# ---------------------------------------------------------------------------


# Modest non-saturating range. Keeps every elementwise activation in the band
# where forward + grad are well-conditioned for both f32 and f64.
def _basic_input(shape: list[int], dtype: str, device: str) -> torch.Tensor:
    n = max(1, math.prod(shape))
    # mix of negative + zero + positive so masks (relu / leaky_relu / hardtanh)
    # see both branches.
    vals = [(-0.75 + i * 0.13) for i in range(n)]
    return torch.tensor(vals, dtype=torch_dtype(dtype), device=device).reshape(shape)


# Strictly positive input for log/sqrt-style ops.
def _positive_input(shape: list[int], dtype: str, device: str) -> torch.Tensor:
    n = max(1, math.prod(shape))
    vals = [0.25 + (i % 7) * 0.4 for i in range(n)]
    return torch.tensor(vals, dtype=torch_dtype(dtype), device=device).reshape(shape)


# Wrap a tensor as a fresh requires_grad leaf without sharing storage with the
# fixture record's `a_data`.
def _grad_clone(t: torch.Tensor) -> torch.Tensor:
    return t.detach().clone().requires_grad_(True)


# ---------------------------------------------------------------------------
# Cat A — activation forwards
# ---------------------------------------------------------------------------
#
# Each unary activation gets a single (4,) and (2, 3) shape. Forward output is
# tabulated alongside the grad-of-input under loss = output.sum() (so the test
# assertion shape is symmetric: scalar reduce → backward → grad).

ACTIVATION_SHAPES: list[tuple[list[int], str]] = [
    ([4], "vec1d"),
    ([2, 3], "mat2d"),
]


# (op_name, callable, supports_gpu, requires_positive_input)
# `supports_gpu = True` means there is a real GPU kernel registered in
# ferrotorch-gpu; the fixture script tabulates the same op on cuda:0 and the
# Rust test runs the GPU lane.
SIMPLE_ACTIVATIONS: list[tuple[str, Any, bool, bool]] = [
    ("relu", lambda x: F.relu(x), True, False),
    ("relu6", lambda x: F.relu6(x), False, False),
    ("sigmoid", lambda x: torch.sigmoid(x), True, False),
    ("tanh", lambda x: torch.tanh(x), True, False),
    ("silu", lambda x: F.silu(x), True, False),
    ("mish", lambda x: F.mish(x), True, False),
    ("leaky_relu", lambda x: F.leaky_relu(x, 0.01), True, False),
    ("hardtanh", lambda x: F.hardtanh(x), False, False),
    ("hardsigmoid", lambda x: F.hardsigmoid(x), False, False),
    ("hardswish", lambda x: F.hardswish(x), False, False),
    ("selu", lambda x: F.selu(x), False, False),
    ("softsign", lambda x: F.softsign(x), False, False),
    ("softplus_default", lambda x: F.softplus(x, beta=1.0, threshold=20.0), True, False),
    ("elu_default", lambda x: F.elu(x, alpha=1.0), True, False),
    # GELU variants — three approximation modes mapped via gelu_with.
    ("gelu_none", lambda x: F.gelu(x, approximate="none"), True, False),
    ("gelu_tanh", lambda x: F.gelu(x, approximate="tanh"), False, False),
    # The "sigmoid" variant is a ferrotorch-specific fast approximation
    # (`x * sigmoid(1.702 * x)`); torch doesn't ship it natively, so we
    # tabulate the math directly.
    ("gelu_sigmoid", lambda x: x * torch.sigmoid(1.702 * x), False, False),
    # softmax / log_softmax dispatch on a 2-D input along the last axis to
    # mirror the typical NN softmax shape. The 1-D row uses dim=0.
    ("softmax_dim_last", lambda x: F.softmax(x, dim=-1), True, False),
    ("log_softmax_dim_last", lambda x: F.log_softmax(x, dim=-1), True, False),
]


def fixture_simple_activations() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for op, fn, supports_gpu, _wants_pos in SIMPLE_ACTIVATIONS:
        for device in DEVICES:
            if device == "cuda:0" and not supports_gpu:
                continue
            for dtype in DTYPES:
                for shape, tag in ACTIVATION_SHAPES:
                    a = _basic_input(shape, dtype, device)
                    a_g = _grad_clone(a)
                    fwd = fn(a_g)
                    fwd.sum().backward()
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


def fixture_prelu() -> list[dict[str, Any]]:
    """prelu: scalar `alpha` parameter. Ferrotorch's CPU impl fixes alpha to
    a 1-element tensor (numel==1) and treats it as a scalar."""
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            for shape, tag in ACTIVATION_SHAPES:
                a = _basic_input(shape, dtype, device)
                a_g = _grad_clone(a)
                alpha_val = 0.25
                alpha = torch.tensor(
                    [alpha_val], dtype=torch_dtype(dtype), device=device
                )
                # PReLU semantics: `x if x >= 0 else alpha * x`.
                fwd = F.prelu(a_g, alpha)
                fwd.sum().backward()
                out.append(
                    {
                        "op": "prelu",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "a_shape": shape,
                        "a_data": to_listf(a),
                        "alpha": alpha_val,
                        "out_shape": list(fwd.shape),
                        "out_values": to_listf(fwd),
                        "grad_a": to_listf(a_g.grad),
                    }
                )
    return out


def fixture_glu() -> list[dict[str, Any]]:
    """GLU splits the last dim in half: glu(x, dim) = a * sigmoid(b)."""
    out: list[dict[str, Any]] = []
    # GLU requires an even size on the splitting dim. The 1-D fixture has
    # 4 elements; the 2-D has shape (2, 4) so we split dim=-1 into 2+2.
    cases = [
        ([4], 0, "vec1d_dim0"),
        ([2, 4], -1, "mat2d_dimneg1"),
    ]
    for device in DEVICES:
        for dtype in DTYPES:
            for shape, dim, tag in cases:
                n = max(1, math.prod(shape))
                vals = [(-0.75 + i * 0.13) for i in range(n)]
                a = torch.tensor(
                    vals, dtype=torch_dtype(dtype), device=device
                ).reshape(shape)
                a_g = _grad_clone(a)
                fwd = F.glu(a_g, dim=dim)
                fwd.sum().backward()
                out.append(
                    {
                        "op": "glu",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "axis": dim,
                        "a_shape": shape,
                        "a_data": to_listf(a),
                        "out_shape": list(fwd.shape),
                        "out_values": to_listf(fwd),
                        "grad_a": to_listf(a_g.grad),
                    }
                )
    return out


def fixture_hardtanh_with() -> list[dict[str, Any]]:
    """hardtanh_with picks custom min/max. Use [-2, 2] so half the input is
    saturated and the other half is in the linear region."""
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            for shape, tag in ACTIVATION_SHAPES:
                a = _basic_input(shape, dtype, device) * 5.0
                a_g = _grad_clone(a)
                fwd = F.hardtanh(a_g, min_val=-2.0, max_val=2.0)
                fwd.sum().backward()
                out.append(
                    {
                        "op": "hardtanh_with",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "a_shape": shape,
                        "a_data": to_listf(a),
                        "min_val": -2.0,
                        "max_val": 2.0,
                        "out_shape": list(fwd.shape),
                        "out_values": to_listf(fwd),
                        "grad_a": to_listf(a_g.grad),
                    }
                )
    return out


def fixture_softplus_with() -> list[dict[str, Any]]:
    """softplus with a non-default beta. PyTorch supports beta != 1; we
    tabulate beta=2 to exercise that branch."""
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            for shape, tag in ACTIVATION_SHAPES:
                a = _basic_input(shape, dtype, device)
                a_g = _grad_clone(a)
                fwd = F.softplus(a_g, beta=2.0, threshold=20.0)
                fwd.sum().backward()
                out.append(
                    {
                        "op": "softplus_with",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "a_shape": shape,
                        "a_data": to_listf(a),
                        "beta": 2.0,
                        "threshold": 20.0,
                        "out_shape": list(fwd.shape),
                        "out_values": to_listf(fwd),
                        "grad_a": to_listf(a_g.grad),
                    }
                )
    return out


def fixture_elu_with() -> list[dict[str, Any]]:
    """elu with alpha=2.0 to exercise the alpha-scaled negative branch."""
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            for shape, tag in ACTIVATION_SHAPES:
                a = _basic_input(shape, dtype, device)
                a_g = _grad_clone(a)
                fwd = F.elu(a_g, alpha=2.0)
                fwd.sum().backward()
                out.append(
                    {
                        "op": "elu_with",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "a_shape": shape,
                        "a_data": to_listf(a),
                        "alpha": 2.0,
                        "out_shape": list(fwd.shape),
                        "out_values": to_listf(fwd),
                        "grad_a": to_listf(a_g.grad),
                    }
                )
    return out


def fixture_leaky_relu_with() -> list[dict[str, Any]]:
    """leaky_relu with non-default negative_slope=0.2."""
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            for shape, tag in ACTIVATION_SHAPES:
                a = _basic_input(shape, dtype, device)
                a_g = _grad_clone(a)
                fwd = F.leaky_relu(a_g, negative_slope=0.2)
                fwd.sum().backward()
                out.append(
                    {
                        "op": "leaky_relu_with",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "a_shape": shape,
                        "a_data": to_listf(a),
                        "slope": 0.2,
                        "out_shape": list(fwd.shape),
                        "out_values": to_listf(fwd),
                        "grad_a": to_listf(a_g.grad),
                    }
                )
    return out


# ---------------------------------------------------------------------------
# Cat A — transcendental forwards (sin / cos / exp / log / clamp)
# ---------------------------------------------------------------------------
#
# These are top-level re-exports from ferrotorch_core::* and live in
# `grad_fns::transcendental`. log expects strictly positive input.

TRANSCENDENTAL_OPS: list[tuple[str, Any, bool]] = [
    # (name, callable, requires positive input)
    ("sin", lambda x: torch.sin(x), False),
    ("cos", lambda x: torch.cos(x), False),
    ("exp", lambda x: torch.exp(x), False),
    ("log", lambda x: torch.log(x), True),
]


def fixture_transcendentals() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for op, fn, wants_pos in TRANSCENDENTAL_OPS:
        for device in DEVICES:
            for dtype in DTYPES:
                for shape, tag in ACTIVATION_SHAPES:
                    if wants_pos:
                        a = _positive_input(shape, dtype, device)
                    else:
                        a = _basic_input(shape, dtype, device)
                    a_g = _grad_clone(a)
                    fwd = fn(a_g)
                    fwd.sum().backward()
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


def fixture_clamp() -> list[dict[str, Any]]:
    """clamp(x, min, max) — element-wise saturation. Use a wide range and
    a moderate clamp window so the output exercises both branches and the
    middle pass-through."""
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            for shape, tag in ACTIVATION_SHAPES:
                a = _basic_input(shape, dtype, device) * 5.0
                a_g = _grad_clone(a)
                fwd = torch.clamp(a_g, min=-1.5, max=1.5)
                fwd.sum().backward()
                out.append(
                    {
                        "op": "clamp",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "a_shape": shape,
                        "a_data": to_listf(a),
                        "min_val": -1.5,
                        "max_val": 1.5,
                        "out_shape": list(fwd.shape),
                        "out_values": to_listf(fwd),
                        "grad_a": to_listf(a_g.grad),
                    }
                )
    return out


# ---------------------------------------------------------------------------
# Cat A — special forwards (CPU only — `special::*` rejects CUDA inputs)
# ---------------------------------------------------------------------------
#
# Forward-only: `special::*` is `unary_map`/`binary_map` based and currently
# CPU-only by signature. Ferrotorch's `special` module surfaces don't carry
# autograd today (no backward grad_fn paired with them), so we tabulate
# forward output only.

SPECIAL_OPS: list[tuple[str, Any, bool]] = [
    # (name, callable, requires positive input)
    ("erf", lambda x: torch.erf(x), False),
    ("erfc", lambda x: torch.erfc(x), False),
    # erfinv is only sensible on |x| < 1; clamp to a moderate band.
    ("erfinv", lambda x: torch.erfinv(torch.clamp(x, min=-0.9, max=0.9)), False),
    # lgamma / digamma require x > 0 to stay finite away from poles.
    ("lgamma", lambda x: torch.lgamma(x), True),
    ("digamma", lambda x: torch.digamma(x), True),
    ("log1p", lambda x: torch.log1p(x), False),
    ("expm1", lambda x: torch.expm1(x), False),
    ("sinc", lambda x: torch.sinc(x), False),
]


def fixture_special() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for op, fn, wants_pos in SPECIAL_OPS:
        # CPU only.
        for dtype in DTYPES:
            for shape, tag in ACTIVATION_SHAPES:
                if wants_pos:
                    a = _positive_input(shape, dtype, "cpu")
                else:
                    a = _basic_input(shape, dtype, "cpu")
                fwd = fn(a)
                out.append(
                    {
                        "op": op,
                        "tag": tag,
                        "dtype": dtype,
                        "device": "cpu",
                        "a_shape": shape,
                        "a_data": to_listf(a),
                        "out_shape": list(fwd.shape),
                        "out_values": to_listf(fwd),
                    }
                )
    return out


def fixture_xlogy() -> list[dict[str, Any]]:
    """xlogy(x, y) = x * log(y), with the convention xlogy(0, y) = 0.
    Two operands so the fixture carries `b_data` in addition to `a_data`."""
    out: list[dict[str, Any]] = []
    for dtype in DTYPES:
        for shape, tag in ACTIVATION_SHAPES:
            n = max(1, math.prod(shape))
            # x mixes zeros + positives so the special-case branch fires.
            x_vals = [0.0 if i % 4 == 0 else 0.25 + (i % 5) * 0.5 for i in range(n)]
            y_vals = [0.5 + (i % 7) * 0.4 for i in range(n)]
            x = torch.tensor(x_vals, dtype=torch_dtype(dtype)).reshape(shape)
            y = torch.tensor(y_vals, dtype=torch_dtype(dtype)).reshape(shape)
            fwd = torch.special.xlogy(x, y)
            out.append(
                {
                    "op": "xlogy",
                    "tag": tag,
                    "dtype": dtype,
                    "device": "cpu",
                    "a_shape": shape,
                    "a_data": to_listf(x),
                    "b_shape": shape,
                    "b_data": to_listf(y),
                    "out_shape": list(fwd.shape),
                    "out_values": to_listf(fwd),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Cat A — orthogonal-polynomial families (CPU only)
# ---------------------------------------------------------------------------
#
# Each polynomial family takes a degree `n` and an input tensor and returns
# the n-th basis polynomial evaluated pointwise. The fixture inputs are kept
# inside the natural domain of each family (shifted variants use [0, 1]).

POLY_FAMILIES: list[tuple[str, Any, str]] = [
    # (op, callable, domain) where domain ∈ {"unit", "shifted", "real"}.
    ("chebyshev_polynomial_t", torch.special.chebyshev_polynomial_t, "unit"),
    ("chebyshev_polynomial_u", torch.special.chebyshev_polynomial_u, "unit"),
    ("chebyshev_polynomial_v", torch.special.chebyshev_polynomial_v, "unit"),
    ("chebyshev_polynomial_w", torch.special.chebyshev_polynomial_w, "unit"),
    ("hermite_polynomial_h", torch.special.hermite_polynomial_h, "real"),
    ("hermite_polynomial_he", torch.special.hermite_polynomial_he, "real"),
    ("laguerre_polynomial_l", torch.special.laguerre_polynomial_l, "positive"),
    ("legendre_polynomial_p", torch.special.legendre_polynomial_p, "unit"),
    (
        "shifted_chebyshev_polynomial_t",
        torch.special.shifted_chebyshev_polynomial_t,
        "shifted",
    ),
    (
        "shifted_chebyshev_polynomial_u",
        torch.special.shifted_chebyshev_polynomial_u,
        "shifted",
    ),
    (
        "shifted_chebyshev_polynomial_v",
        torch.special.shifted_chebyshev_polynomial_v,
        "shifted",
    ),
    (
        "shifted_chebyshev_polynomial_w",
        torch.special.shifted_chebyshev_polynomial_w,
        "shifted",
    ),
]


def _poly_input(shape: list[int], dtype: str, domain: str) -> torch.Tensor:
    n = max(1, math.prod(shape))
    if domain == "unit":
        vals = [-0.8 + 0.4 * i / max(1, n - 1) for i in range(n)]
    elif domain == "shifted":
        # Shifted variants use the [0, 1] domain.
        vals = [0.1 + 0.7 * i / max(1, n - 1) for i in range(n)]
    elif domain == "positive":
        vals = [0.5 + 0.3 * i for i in range(n)]
    else:  # "real" — Hermite is real-line, but stay modest to avoid blowup.
        vals = [-1.0 + 0.5 * i for i in range(n)]
    return torch.tensor(vals, dtype=torch_dtype(dtype)).reshape(shape)


def fixture_polynomials() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # Test a moderate degree (3) and a higher one (5) to exercise the
    # three-term recurrence at multiple stages.
    for op, fn, domain in POLY_FAMILIES:
        for dtype in DTYPES:
            for n_deg in (3, 5):
                a = _poly_input([4], dtype, domain)
                fwd = fn(a, n_deg)
                out.append(
                    {
                        "op": op,
                        "tag": f"vec1d_n{n_deg}",
                        "dtype": dtype,
                        "device": "cpu",
                        "n": n_deg,
                        "a_shape": [4],
                        "a_data": to_listf(a),
                        "out_shape": list(fwd.shape),
                        "out_values": to_listf(fwd),
                    }
                )
    return out


# ---------------------------------------------------------------------------
# Edge-case fixtures
# ---------------------------------------------------------------------------


def fixture_saturating_edges() -> list[dict[str, Any]]:
    """Saturation: tanh(±100) ≈ ±1, sigmoid(±100) saturates without overflow."""
    out: list[dict[str, Any]] = []
    cases = [
        ("tanh_saturated", lambda x: torch.tanh(x), [100.0, -100.0, 0.0]),
        ("sigmoid_saturated", lambda x: torch.sigmoid(x), [100.0, -100.0, 0.0]),
    ]
    for op, fn, vals in cases:
        for dtype in DTYPES:
            a = torch.tensor(vals, dtype=torch_dtype(dtype))
            fwd = fn(a)
            out.append(
                {
                    "op": op,
                    "tag": "edge",
                    "dtype": dtype,
                    "device": "cpu",
                    "a_shape": [3],
                    "a_data": to_listf(a),
                    "out_shape": list(fwd.shape),
                    "out_values": to_listf(fwd),
                }
            )
    return out


def fixture_softmax_stability() -> list[dict[str, Any]]:
    """softmax([100, 100, 100]) must yield [1/3, 1/3, 1/3] (max-subtract trick)
    rather than NaN / overflow. log_softmax([1000, 1001]) similarly."""
    out: list[dict[str, Any]] = []
    cases = [
        (
            "softmax_uniform_large",
            lambda x: F.softmax(x, dim=-1),
            [100.0, 100.0, 100.0],
            [3],
        ),
        (
            "log_softmax_large",
            lambda x: F.log_softmax(x, dim=-1),
            [1000.0, 1001.0],
            [2],
        ),
    ]
    for op, fn, vals, shape in cases:
        for dtype in DTYPES:
            for device in DEVICES:
                a = torch.tensor(vals, dtype=torch_dtype(dtype), device=device)
                fwd = fn(a)
                out.append(
                    {
                        "op": op,
                        "tag": "edge",
                        "dtype": dtype,
                        "device": device,
                        "a_shape": shape,
                        "a_data": to_listf(a),
                        "out_shape": list(fwd.shape),
                        "out_values": to_listf(fwd),
                    }
                )
    return out


def fixture_log1p_expm1_small_x() -> list[dict[str, Any]]:
    """log1p(small_x) ≈ x and expm1(small_x) ≈ x — naive `log(1+x)` /
    `exp(x) - 1` lose precision for |x| ~ 1e-10. f64-only test."""
    out: list[dict[str, Any]] = []
    small = [1e-10, -1e-10, 1e-15, -1e-15, 0.0]
    for op, fn in [
        ("log1p_small", lambda x: torch.log1p(x)),
        ("expm1_small", lambda x: torch.expm1(x)),
    ]:
        a = torch.tensor(small, dtype=torch.float64)
        fwd = fn(a)
        out.append(
            {
                "op": op,
                "tag": "edge",
                "dtype": "float64",
                "device": "cpu",
                "a_shape": [len(small)],
                "a_data": to_listf(a),
                "out_shape": list(fwd.shape),
                "out_values": to_listf(fwd),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Verification-debt repayment lanes (per dispatch)
# ---------------------------------------------------------------------------
#
# Per the dispatch's "VERIFICATION DEBT REPAYMENT (HARD)" block, these three
# f64 GPU lanes must assert at F64_TRANSCENDENTAL = 1e-10. They surface any
# residuals of the Dispatch C polynomial cluster sweep that wasn't directly
# probed before.


def fixture_gpu_log_f64_wide_range() -> list[dict[str, Any]]:
    """gpu_log_f64 across [1e-10, 1e10] — 11 values spanning 20 orders of
    magnitude."""
    if not torch.cuda.is_available():
        return []
    vals = [1e-10, 1e-8, 1e-5, 1e-2, 0.5, 1.0, 2.0, 1e2, 1e5, 1e8, 1e10]
    a = torch.tensor(vals, dtype=torch.float64, device="cuda:0")
    fwd = torch.log(a)
    return [
        {
            "op": "gpu_log_f64_wide_range",
            "tag": "verif_debt",
            "dtype": "float64",
            "device": "cuda:0",
            "a_shape": [len(vals)],
            "a_data": to_listf(a),
            "out_shape": list(fwd.shape),
            "out_values": to_listf(fwd),
        }
    ]


def fixture_gpu_log_softmax_f64_typical() -> list[dict[str, Any]]:
    """gpu_log_softmax_f64 on a typical NN softmax input (logits batch-1)."""
    if not torch.cuda.is_available():
        return []
    # 2-D (1, 8) — typical 8-class logit row.
    vals = [-1.5, 0.3, 1.8, -0.7, 2.1, -2.3, 0.05, 1.1]
    a = torch.tensor([vals], dtype=torch.float64, device="cuda:0")
    fwd = F.log_softmax(a, dim=-1)
    return [
        {
            "op": "gpu_log_softmax_f64_typical",
            "tag": "verif_debt",
            "dtype": "float64",
            "device": "cuda:0",
            "a_shape": [1, 8],
            "a_data": to_listf(a),
            "out_shape": list(fwd.shape),
            "out_values": to_listf(fwd),
        }
    ]


def fixture_gpu_mish_f64_typical() -> list[dict[str, Any]]:
    """gpu_mish_f64 on a typical activation input range."""
    if not torch.cuda.is_available():
        return []
    vals = [-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 10.0]
    a = torch.tensor(vals, dtype=torch.float64, device="cuda:0")
    fwd = F.mish(a)
    return [
        {
            "op": "gpu_mish_f64_typical",
            "tag": "verif_debt",
            "dtype": "float64",
            "device": "cuda:0",
            "a_shape": [len(vals)],
            "a_data": to_listf(a),
            "out_shape": list(fwd.shape),
            "out_values": to_listf(fwd),
        }
    ]


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def main() -> int:
    fixtures: list[dict[str, Any]] = []
    fixtures += fixture_simple_activations()
    fixtures += fixture_prelu()
    fixtures += fixture_glu()
    fixtures += fixture_hardtanh_with()
    fixtures += fixture_softplus_with()
    fixtures += fixture_elu_with()
    fixtures += fixture_leaky_relu_with()
    fixtures += fixture_transcendentals()
    fixtures += fixture_clamp()
    fixtures += fixture_special()
    fixtures += fixture_xlogy()
    fixtures += fixture_polynomials()
    fixtures += fixture_saturating_edges()
    fixtures += fixture_softmax_stability()
    fixtures += fixture_log1p_expm1_small_x()
    fixtures += fixture_gpu_log_f64_wide_range()
    fixtures += fixture_gpu_log_softmax_f64_typical()
    fixtures += fixture_gpu_mish_f64_typical()

    payload = {
        "metadata": fixture_metadata(),
        "fixtures": fixtures,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(
        f"wrote {len(fixtures)} fixtures to "
        f"{FIXTURE_PATH.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
