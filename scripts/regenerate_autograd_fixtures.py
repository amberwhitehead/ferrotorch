#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for ferrotorch-core Phase 2.10 (autograd
internals).

Tracking issue: #772 (parent: #759).

Output:
    ferrotorch-core/tests/conformance/fixtures/autograd.json

Coverage: 193 surface items in `_surface_exclusions.toml` filtered by
``tracking_issue = "#772"``, spanning the autograd machinery —

  * forward-mode AD: `DualTensor`, `dual_*` ops, `jvp_exact`, `jacfwd`
  * reverse-mode higher-order: `grad`, `jacobian`, `hessian`
  * jvp / vjp (finite-diff and graph-based)
  * gradient penalty (WGAN-GP) and grad_norm
  * checkpoint / checkpoint_multi (recomputation)
  * vmap / vmap2 / vmap3 / vmap_many / vmap_multi_output / per_sample_grad
  * hooks (register_hook + post_accumulate_grad_hook)
  * AnomalyMode + ForwardBacktrace + check_gradient_anomaly
  * autocast / autocast_ops policy (state, snapshot, with_autocast_state,
    category, guard, log)
  * no_grad / enable_grad / inference_mode / set_grad_enabled
  * saved_tensors_hooks / pack / unpack
  * fixed_point (deep equilibrium)
  * gradcheck
  * cond / scan / validate_cond_branches (control flow)

# Numerical parity
#
# The autograd machinery is mostly *control flow*, not tensor numerics: most
# tests verify behavioural contracts (state toggles, hook fires, anomaly
# detection). Fixtures for numerics target the ops that have a stable closed-
# form derivative computable in PyTorch (torch.func.{jvp, vjp, jacrev, jacfwd,
# hessian}). All numerics are pinned to torch 2.11.0+cu130 — earlier minor
# torch releases shifted the func API surface.

# torch.func status
#
# `torch.func` (formerly `functorch`) is stable as of torch 2.0+. We pin the
# torch version in metadata so future API drift is caught by the conformance
# suite rather than silently absorbed.

Usage from WSL (preferred per #777):

    python3 scripts/regenerate_autograd_fixtures.py

Required Python deps: torch.
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
    / "autograd.json"
)

DTYPES: list[str] = ["float32", "float64"]
DEVICES: list[str] = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda:0")

RNG_SEED: int = 0xC0FFEE
torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RNG_SEED)


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def to_listf(values: Any) -> list[Any]:
    """Encode an iterable of floats with NaN/Inf sentinels."""
    encoded: list[Any] = []
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list):
        values = [values]
    for v in values:
        if isinstance(v, list):
            encoded.append(to_listf(v))
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            encoded.append(v)
            continue
        if math.isnan(fv):
            encoded.append("NaN")
        elif math.isinf(fv):
            encoded.append("Infinity" if fv > 0 else "-Infinity")
        else:
            encoded.append(fv)
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
        # Pin the torch.func generation we observed so future drift is caught
        # explicitly here, not silently in the test harness.
        "torch_func_status": "stable since torch 2.0; pinned to 2.11.0+cu130",
    }


# ---------------------------------------------------------------------------
# Forward-mode AD reference values via torch.func
# ---------------------------------------------------------------------------
#
# The Rust `dual_*` ops implement the same forward rules as torch.func.jvp.
# We seed jvp with v=ones(shape) so the tangent equals the column-sum of the
# Jacobian — easier to interpret in failure messages than a randomized seed.


def _t(values: list[float], dtype: str) -> torch.Tensor:
    return torch.tensor(values, dtype=torch_dtype(dtype))


def fixture_dual_unary() -> list[dict[str, Any]]:
    """JVP of unary ops: relu, sigmoid, tanh, exp, log, sin, cos, neg."""
    out: list[dict[str, Any]] = []
    cases = [
        # name, callable, primal-input, tangent
        ("relu", torch.relu, [-1.0, 0.5, 2.0, -3.0], [1.0, 1.0, 1.0, 1.0]),
        ("sigmoid", torch.sigmoid, [-2.0, 0.0, 1.5], [1.0, 1.0, 1.0]),
        ("tanh", torch.tanh, [-1.5, 0.0, 0.7], [1.0, 1.0, 1.0]),
        ("exp", torch.exp, [-1.0, 0.0, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]),
        ("log", torch.log, [0.5, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0]),
        ("sin", torch.sin, [0.0, math.pi / 4, math.pi / 2, math.pi], [1.0, 1.0, 1.0, 1.0]),
        ("cos", torch.cos, [0.0, math.pi / 4, math.pi / 2, math.pi], [1.0, 1.0, 1.0, 1.0]),
        ("neg", torch.neg, [1.0, -2.0, 3.5], [0.5, 0.25, 1.0]),
    ]
    for dtype in DTYPES:
        for name, fn, x_data, v_data in cases:
            x = _t(x_data, dtype)
            v = _t(v_data, dtype)
            primal, tangent = torch.func.jvp(fn, (x,), (v,))
            out.append(
                {
                    "op": "dual_unary",
                    "tag": name,
                    "dtype": dtype,
                    "device": "cpu",
                    "a_shape": list(x.shape),
                    "a_data": to_listf(x),
                    "v_data": to_listf(v),
                    "out_primal": to_listf(primal),
                    "out_tangent": to_listf(tangent),
                }
            )
    return out


def fixture_dual_binary() -> list[dict[str, Any]]:
    """JVP of binary ops: add, sub, mul, div."""
    out: list[dict[str, Any]] = []
    cases = [
        ("add", torch.add, [1.0, 2.0, 3.0], [0.5, 0.3, 0.1], [4.0, 5.0, 6.0], [0.1, 0.2, 0.3]),
        ("sub", torch.sub, [5.0, 3.0, 1.0], [1.0, 0.5, 0.2], [2.0, 1.0, 0.5], [0.3, 0.1, 0.05]),
        ("mul", torch.mul, [2.0, 3.0, 4.0], [0.5, 0.1, 0.2], [3.0, 2.0, 5.0], [0.1, 0.4, 0.3]),
        ("div", torch.div, [6.0, 8.0, 9.0], [1.0, 0.5, 0.3], [3.0, 4.0, 3.0], [0.5, 0.1, 0.2]),
    ]
    for dtype in DTYPES:
        for name, fn, a_data, da_data, b_data, db_data in cases:
            a = _t(a_data, dtype)
            da = _t(da_data, dtype)
            b = _t(b_data, dtype)
            db = _t(db_data, dtype)
            primal, tangent = torch.func.jvp(fn, (a, b), (da, db))
            out.append(
                {
                    "op": "dual_binary",
                    "tag": name,
                    "dtype": dtype,
                    "device": "cpu",
                    "a_shape": list(a.shape),
                    "a_data": to_listf(a),
                    "da_data": to_listf(da),
                    "b_data": to_listf(b),
                    "db_data": to_listf(db),
                    "out_primal": to_listf(primal),
                    "out_tangent": to_listf(tangent),
                }
            )
    return out


def fixture_dual_matmul() -> list[dict[str, Any]]:
    """JVP of 2-D matmul: d(A @ B) = dA @ B + A @ dB."""
    out: list[dict[str, Any]] = []
    for dtype in DTYPES:
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch_dtype(dtype))
        da = torch.tensor([[0.1, 0.0], [0.0, 0.1]], dtype=torch_dtype(dtype))
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch_dtype(dtype))
        db = torch.zeros_like(b)
        primal, tangent = torch.func.jvp(torch.matmul, (a, b), (da, db))
        out.append(
            {
                "op": "dual_matmul",
                "tag": "2x2_2x2",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": list(a.shape),
                "b_shape": list(b.shape),
                "a_data": to_listf(a.flatten()),
                "da_data": to_listf(da.flatten()),
                "b_data": to_listf(b.flatten()),
                "db_data": to_listf(db.flatten()),
                "out_shape": list(primal.shape),
                "out_primal": to_listf(primal.flatten()),
                "out_tangent": to_listf(tangent.flatten()),
            }
        )
    return out


# ---------------------------------------------------------------------------
# jvp_exact + chain rule (composition)
# ---------------------------------------------------------------------------


def fixture_jvp_exact_chain() -> list[dict[str, Any]]:
    """JVP through a chain of ops — exercises chain-rule composition."""
    out: list[dict[str, Any]] = []
    for dtype in DTYPES:
        # f(x) = exp(x * x); f'(x) = 2x * exp(x^2)
        x = _t([1.0, 0.5, 0.25], dtype)
        v = _t([1.0, 1.0, 1.0], dtype)

        def f(t: torch.Tensor) -> torch.Tensor:
            return torch.exp(t * t)

        primal, tangent = torch.func.jvp(f, (x,), (v,))
        out.append(
            {
                "op": "jvp_chain",
                "tag": "exp_x_squared",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": list(x.shape),
                "a_data": to_listf(x),
                "v_data": to_listf(v),
                "out_primal": to_listf(primal),
                "out_tangent": to_listf(tangent),
            }
        )

        # f(x) = sin^2(x) + cos^2(x) = 1, f'(x) = 0
        x = _t([1.5, 2.5, -0.5], dtype)
        v = _t([1.0, 1.0, 1.0], dtype)

        def g(t: torch.Tensor) -> torch.Tensor:
            return torch.sin(t) * torch.sin(t) + torch.cos(t) * torch.cos(t)

        primal, tangent = torch.func.jvp(g, (x,), (v,))
        out.append(
            {
                "op": "jvp_chain",
                "tag": "sin2_plus_cos2",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": list(x.shape),
                "a_data": to_listf(x),
                "v_data": to_listf(v),
                "out_primal": to_listf(primal),
                "out_tangent": to_listf(tangent),
            }
        )
    return out


# ---------------------------------------------------------------------------
# jacfwd — full Jacobian via vmap(jvp)
# ---------------------------------------------------------------------------


def fixture_jacfwd() -> list[dict[str, Any]]:
    """torch.func.jacfwd reference Jacobians for elementwise functions."""
    out: list[dict[str, Any]] = []
    for dtype in DTYPES:
        # f(x) = 2x → J = 2 I
        x = _t([1.0, 2.0, 3.0], dtype)
        jac = torch.func.jacfwd(lambda t: t * 2.0)(x)
        out.append(
            {
                "op": "jacfwd",
                "tag": "linear_2x",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": list(x.shape),
                "a_data": to_listf(x),
                "out_shape": list(jac.shape),
                "out_values": to_listf(jac.flatten()),
            }
        )
        # f(x) = x*x → J = diag(2x)
        x = _t([1.0, 2.0, 3.0], dtype)
        jac = torch.func.jacfwd(lambda t: t * t)(x)
        out.append(
            {
                "op": "jacfwd",
                "tag": "quadratic",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": list(x.shape),
                "a_data": to_listf(x),
                "out_shape": list(jac.shape),
                "out_values": to_listf(jac.flatten()),
            }
        )
        # f(x) = sin(x) → J = diag(cos(x))
        x = _t([0.0, math.pi / 2, math.pi], dtype)
        jac = torch.func.jacfwd(torch.sin)(x)
        out.append(
            {
                "op": "jacfwd",
                "tag": "sin",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": list(x.shape),
                "a_data": to_listf(x),
                "out_shape": list(jac.shape),
                "out_values": to_listf(jac.flatten()),
            }
        )
    return out


# ---------------------------------------------------------------------------
# jacobian (reverse-mode) — torch.func.jacrev
# ---------------------------------------------------------------------------


def fixture_jacobian() -> list[dict[str, Any]]:
    """jacobian via reverse-mode (torch.func.jacrev) — same shape as jacfwd."""
    out: list[dict[str, Any]] = []
    for dtype in DTYPES:
        # f(x) = sum(x*x) → J = 2x
        x = _t([2.0, 3.0, 4.0], dtype)
        jac = torch.func.jacrev(lambda t: torch.sum(t * t))(x)
        # Broadcast to [1, n] shape so it matches `jacobian()`'s output shape.
        out.append(
            {
                "op": "jacobian_scalar_out",
                "tag": "sum_square",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": list(x.shape),
                "a_data": to_listf(x),
                "out_shape": [1, x.numel()],
                "out_values": to_listf(jac.flatten()),
            }
        )
    return out


# ---------------------------------------------------------------------------
# hessian — torch.func.hessian
# ---------------------------------------------------------------------------


def fixture_hessian() -> list[dict[str, Any]]:
    """Hessian H[i,j] = d^2 f / dx_i dx_j for a scalar function f: R^n -> R."""
    out: list[dict[str, Any]] = []
    for dtype in DTYPES:
        # f(x) = sum(x^2) → H = 2 I
        x = _t([3.0, 4.0], dtype)
        H = torch.func.hessian(lambda t: torch.sum(t * t))(x)
        out.append(
            {
                "op": "hessian",
                "tag": "sum_square",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": list(x.shape),
                "a_data": to_listf(x),
                "out_shape": list(H.shape),
                "out_values": to_listf(H.flatten()),
            }
        )
        # f(x, y) = x*y → H = [[0, 1], [1, 0]]
        x = _t([2.0, 3.0], dtype)
        H = torch.func.hessian(lambda t: t[0] * t[1])(x)
        out.append(
            {
                "op": "hessian",
                "tag": "xy",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": list(x.shape),
                "a_data": to_listf(x),
                "out_shape": list(H.shape),
                "out_values": to_listf(H.flatten()),
            }
        )
        # f(x) = x^3 → f''(x) = 6x → at x=2, H = [[12]]
        x = _t([2.0], dtype)
        H = torch.func.hessian(lambda t: torch.sum(t ** 3))(x)
        out.append(
            {
                "op": "hessian",
                "tag": "cubic",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": list(x.shape),
                "a_data": to_listf(x),
                "out_shape": list(H.shape),
                "out_values": to_listf(H.flatten()),
            }
        )
    return out


# ---------------------------------------------------------------------------
# vjp / jvp (graph-based reverse / finite-diff forward)
# ---------------------------------------------------------------------------


def fixture_vjp() -> list[dict[str, Any]]:
    """Reverse-mode VJP: v^T @ J. Reference via torch.func.vjp."""
    out: list[dict[str, Any]] = []
    for dtype in DTYPES:
        # f(x) = 2x → v^T J = 2v
        x = _t([1.0, 2.0, 3.0], dtype)
        v = _t([4.0, 5.0, 6.0], dtype)
        _, vjp_fn = torch.func.vjp(lambda t: t * 2.0, x)
        (g,) = vjp_fn(v)
        out.append(
            {
                "op": "vjp",
                "tag": "linear_2x",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": list(x.shape),
                "a_data": to_listf(x),
                "v_data": to_listf(v),
                "out_shape": list(g.shape),
                "out_values": to_listf(g),
            }
        )
        # f(x) = x*x → J = diag(2x), v^T J = 2x*v
        x = _t([1.0, 2.0, 3.0], dtype)
        v = _t([1.0, 1.0, 1.0], dtype)
        _, vjp_fn = torch.func.vjp(lambda t: t * t, x)
        (g,) = vjp_fn(v)
        out.append(
            {
                "op": "vjp",
                "tag": "square",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": list(x.shape),
                "a_data": to_listf(x),
                "v_data": to_listf(v),
                "out_shape": list(g.shape),
                "out_values": to_listf(g),
            }
        )
    return out


def fixture_jvp_finite_diff() -> list[dict[str, Any]]:
    """Forward-mode JVP: J @ v. Reference via torch.func.jvp."""
    out: list[dict[str, Any]] = []
    for dtype in DTYPES:
        # f(x) = x*x; J=diag(2x); J v = 2x * v
        x = _t([3.0, 4.0], dtype)
        v = _t([1.0, 1.0], dtype)
        _, tangent = torch.func.jvp(lambda t: t * t, (x,), (v,))
        out.append(
            {
                "op": "jvp_finite",
                "tag": "square",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": list(x.shape),
                "a_data": to_listf(x),
                "v_data": to_listf(v),
                "out_shape": list(tangent.shape),
                "out_values": to_listf(tangent),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Higher-order grad: dy/dx and d2y/dx2 for power and product
# ---------------------------------------------------------------------------


def fixture_higher_order_grad() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dtype in DTYPES:
        # f(x) = x^3 → f'(x) = 3 x^2 → f''(x) = 6 x
        x_val = 2.0
        out.append(
            {
                "op": "higher_order_grad",
                "tag": "x_cubed_at_2",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": [1],
                "a_data": [x_val],
                "first_deriv": [3.0 * x_val * x_val],
                "second_deriv": [6.0 * x_val],
            }
        )
        # f(x) = x^2 at x=5
        x_val = 5.0
        out.append(
            {
                "op": "higher_order_grad",
                "tag": "x_squared_at_5",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": [1],
                "a_data": [x_val],
                "first_deriv": [2.0 * x_val],
                "second_deriv": [2.0],
            }
        )
    return out


# ---------------------------------------------------------------------------
# vmap reference: per-batch matmul should equal bmm
# ---------------------------------------------------------------------------


def fixture_vmap() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dtype in DTYPES:
        # vmap of matmul over leading batch should equal bmm
        a = torch.arange(24, dtype=torch_dtype(dtype)).reshape(2, 3, 4)
        b = (torch.arange(16, dtype=torch_dtype(dtype)) * 0.1).reshape(2, 4, 2)
        c = torch.bmm(a, b)
        out.append(
            {
                "op": "vmap_matmul",
                "tag": "bmm_equiv",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": list(a.shape),
                "a_data": to_listf(a.flatten()),
                "b_shape": list(b.shape),
                "b_data": to_listf(b.flatten()),
                "out_shape": list(c.shape),
                "out_values": to_listf(c.flatten()),
            }
        )
        # Per-row sum: sum(x[i, :]) for each i → vmapped-sum
        x = torch.arange(12, dtype=torch_dtype(dtype)).reshape(3, 4) + 1.0
        per_row_sum = x.sum(dim=1)
        out.append(
            {
                "op": "vmap_sum",
                "tag": "per_row_sum",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": list(x.shape),
                "a_data": to_listf(x.flatten()),
                "out_shape": list(per_row_sum.shape),
                "out_values": to_listf(per_row_sum),
            }
        )
    return out


# ---------------------------------------------------------------------------
# WGAN-GP gradient penalty: linear discriminator → ||grad||_2 = sqrt(n)
# ---------------------------------------------------------------------------


def fixture_grad_penalty() -> list[dict[str, Any]]:
    """For D(x) = sum(x), grad = ones(n), ||grad||_2 = sqrt(n).
    penalty = lambda * (sqrt(n) - 1)^2 — independent of inputs."""
    out: list[dict[str, Any]] = []
    for dtype in DTYPES:
        for n in (1, 4, 16):
            real = [float(i + 1) for i in range(n)]
            fake = [float(i) * 0.5 for i in range(n)]
            lam = 10.0
            sqrt_n = math.sqrt(n)
            penalty_val = lam * (sqrt_n - 1.0) * (sqrt_n - 1.0)
            out.append(
                {
                    "op": "gradient_penalty",
                    "tag": f"linear_n{n}",
                    "dtype": dtype,
                    "device": "cpu",
                    "a_shape": [n],
                    "real": real,
                    "fake": fake,
                    "lambda": lam,
                    "out_values": [penalty_val],
                }
            )
    return out


# ---------------------------------------------------------------------------
# Fixed point reference: x = a*x has fixed point 0 when |a| < 1
# ---------------------------------------------------------------------------


def fixture_fixed_point() -> list[dict[str, Any]]:
    """Closed-form: f(x, a) = a * x. Fixed point: 0 (when |a| < 1).
    dx*/da = 0 (the fixed point is constant w.r.t. a)."""
    out: list[dict[str, Any]] = []
    for dtype in DTYPES:
        out.append(
            {
                "op": "fixed_point",
                "tag": "linear_contraction",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": [1],
                "x0": [10.0],
                "param": [0.5],
                "out_values": [0.0],
                "max_iter": 200,
                "tol": 1e-8,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def main() -> int:
    fixtures: list[dict[str, Any]] = []

    # Forward-mode AD reference values
    fixtures += fixture_dual_unary()
    fixtures += fixture_dual_binary()
    fixtures += fixture_dual_matmul()
    fixtures += fixture_jvp_exact_chain()

    # Reverse / forward mode Jacobians and Hessians
    fixtures += fixture_jacfwd()
    fixtures += fixture_jacobian()
    fixtures += fixture_hessian()

    # vjp / jvp
    fixtures += fixture_vjp()
    fixtures += fixture_jvp_finite_diff()

    # Higher-order
    fixtures += fixture_higher_order_grad()

    # vmap
    fixtures += fixture_vmap()

    # gradient penalty
    fixtures += fixture_grad_penalty()

    # fixed point
    fixtures += fixture_fixed_point()

    payload = {"metadata": fixture_metadata(), "fixtures": fixtures}
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"wrote {len(fixtures)} fixtures to {FIXTURE_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
