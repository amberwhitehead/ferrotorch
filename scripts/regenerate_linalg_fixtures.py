#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for ferrotorch-core Phase 2.4 (linalg).

Tracking issue: #766 (parent: #759).

Output:
    ferrotorch-core/tests/conformance/fixtures/linalg.json

Coverage (63 surface items per `_surface_exclusions.toml`, partitioned by
category; see `conformance_linalg.rs` for the test runner):

* Cat A — matmul forwards (CPU + GPU + autograd, all device-supported):
    mm, matmul, bmm, dot, mv, transpose, mm_bt (linear forward shape),
    plus the differentiable wrappers and `linear_fused`.
* Cat B — factorizations (CPU + GPU forward; reconstruction asserts):
    qr, svd, cholesky, eigh, eigvalsh, lu, lu_factor, svdvals.
* Cat C — solvers (CPU + GPU forward where supported):
    solve, solve_ex, lstsq_solve, lstsq, solve_triangular, ldl_factor,
    ldl_solve, tensorsolve, tensorinv.
* Cat D — det / norm / power (CPU; det/inv/slogdet are CPU-only):
    det, slogdet, inv, inv_ex, cholesky_ex, matrix_power, matrix_norm,
    vector_norm, matrix_rank, cond, pinv.
* Cat E — misc (CPU): cross, multi_dot, diagonal, householder_product,
    matrix_exp, eig, eigvals, permute_0213.

Edge cases per the dispatch:
  * Non-square matmul (e.g. [3,4] @ [4,5]).
  * Singular matrix path for inv/solve (PyTorch RuntimeError; ferrotorch
    must return Err).
  * Batched bmm (batch dim).
  * Empty matrix paths.
  * Scalar/1×1 degenerate factorizations.
  * Complex eig output (returned as [n, 2] real-imag tensor).

For non-unique factorizations (qr, svd, eigh, lu) the fixture stores the
INPUT and (for forward sanity) the singular values / eigenvalues only.
Reconstruction tolerance is enforced in Rust by computing
`(Q @ R - A).norm() / A.norm() < tol` rather than comparing Q raw — Q is
not unique up to column-sign flips, the conformance test must not pin it.

Usage from WSL (preferred per #777):

    python3 scripts/regenerate_linalg_fixtures.py

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
    / "linalg.json"
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


def _spd(n: int, dtype: str, device: str, jitter: float = 1.0) -> torch.Tensor:
    """Symmetric positive-definite n×n matrix: `M @ M^T + jitter * I`."""
    g = torch.Generator(device=device)
    g.manual_seed(RNG_SEED + n)
    m = torch.randn(n, n, dtype=torch_dtype(dtype), device=device, generator=g)
    a = m @ m.T + jitter * torch.eye(n, dtype=torch_dtype(dtype), device=device)
    return a


def _gen_matrix(
    rows: int, cols: int, dtype: str, device: str, seed_offset: int
) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(RNG_SEED + seed_offset)
    return torch.randn(rows, cols, dtype=torch_dtype(dtype), device=device, generator=g)


# ---------------------------------------------------------------------------
# Cat A — matmul forwards (mm / mv / dot / matmul / bmm / transpose /
#                          mm_bt / linear_fused)
# ---------------------------------------------------------------------------
#
# `ops::linalg::{mm, mv, bmm, dot, transpose}` are CPU-only (return
# Err(GpuTensorNotAccessible) on a CUDA tensor — that is the documented
# behaviour and we don't go through them on GPU).
#
# `grad_fns::linalg::{mm_differentiable, mv_differentiable, bmm_differentiable,
# dot_differentiable, matmul_differentiable, mm_bt_differentiable, linear_fused,
# bmm}` dispatch to GPU when the input is CUDA-resident.
#
# Tags:
#   * `mm_2x2`, `mm_3x4_4x5` (non-square)
#   * `mv_3x4_4`
#   * `dot_5`
#   * `bmm_2_3_4_2_4_3` (batch=2)
#   * `bmm_4_8_8_4_8_8` (square batch)
#   * `matmul_3d_3d` (broadcast over batch)
#   * `matmul_4d_2d` (broadcast)
#   * `transpose_3x4`


def fixture_matmul_forwards() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            # mm: square 2x2 and non-square 3x4 @ 4x5
            for tag, m, k, n in [("mm_2x2", 2, 2, 2), ("mm_3x4_4x5", 3, 4, 5)]:
                a = _gen_matrix(m, k, dtype, device, 11)
                b = _gen_matrix(k, n, dtype, device, 13)
                a_g = a.detach().clone().requires_grad_(True)
                b_g = b.detach().clone().requires_grad_(True)
                c = a_g @ b_g
                c.sum().backward()
                out.append(
                    {
                        "op": "mm",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "a_shape": [m, k],
                        "b_shape": [k, n],
                        "a_data": to_listf(a),
                        "b_data": to_listf(b),
                        "out_shape": [m, n],
                        "out_values": to_listf(c.detach()),
                        "grad_a": to_listf(a_g.grad),
                        "grad_b": to_listf(b_g.grad),
                    }
                )

            # mv: (3, 4) @ (4,) -> (3,)
            a = _gen_matrix(3, 4, dtype, device, 21)
            b = torch.arange(1.0, 5.0, dtype=torch_dtype(dtype), device=device)
            a_g = a.detach().clone().requires_grad_(True)
            b_g = b.detach().clone().requires_grad_(True)
            c = a_g @ b_g
            c.sum().backward()
            out.append(
                {
                    "op": "mv",
                    "tag": "mv_3x4_4",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [3, 4],
                    "b_shape": [4],
                    "a_data": to_listf(a),
                    "b_data": to_listf(b),
                    "out_shape": [3],
                    "out_values": to_listf(c.detach()),
                    "grad_a": to_listf(a_g.grad),
                    "grad_b": to_listf(b_g.grad),
                }
            )

            # dot: 1-D length-5
            a = torch.tensor(
                [1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch_dtype(dtype), device=device
            )
            b = torch.tensor(
                [0.5, -0.25, 1.5, -1.0, 2.0], dtype=torch_dtype(dtype), device=device
            )
            a_g = a.detach().clone().requires_grad_(True)
            b_g = b.detach().clone().requires_grad_(True)
            c = torch.dot(a_g, b_g)
            c.backward()
            out.append(
                {
                    "op": "dot",
                    "tag": "dot_5",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [5],
                    "b_shape": [5],
                    "a_data": to_listf(a),
                    "b_data": to_listf(b),
                    "out_shape": [],
                    "out_values": to_listf(c.detach()),
                    "grad_a": to_listf(a_g.grad),
                    "grad_b": to_listf(b_g.grad),
                }
            )

            # bmm: batch=2, 3x4 @ 4x5
            for tag, batch, m, k, n in [
                ("bmm_2_3_4_4_5", 2, 3, 4, 5),
                ("bmm_4_8_8_8_8", 4, 8, 8, 8),
            ]:
                g = torch.Generator(device=device)
                g.manual_seed(RNG_SEED + 31 + batch + m + k + n)
                a = torch.randn(
                    batch, m, k, dtype=torch_dtype(dtype), device=device, generator=g
                )
                b = torch.randn(
                    batch, k, n, dtype=torch_dtype(dtype), device=device, generator=g
                )
                a_g = a.detach().clone().requires_grad_(True)
                b_g = b.detach().clone().requires_grad_(True)
                c = torch.bmm(a_g, b_g)
                c.sum().backward()
                out.append(
                    {
                        "op": "bmm",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "a_shape": [batch, m, k],
                        "b_shape": [batch, k, n],
                        "a_data": to_listf(a),
                        "b_data": to_listf(b),
                        "out_shape": [batch, m, n],
                        "out_values": to_listf(c.detach()),
                        "grad_a": to_listf(a_g.grad),
                        "grad_b": to_listf(b_g.grad),
                    }
                )

            # matmul: 3D x 3D (batched), 4D x 2D (broadcast), 1D x 1D, 2D x 1D, 1D x 2D
            for tag, a_shape, b_shape in [
                ("matmul_3d_3d", [2, 3, 4], [2, 4, 5]),
                ("matmul_2d_2d", [4, 5], [5, 6]),
                ("matmul_1d_1d", [4], [4]),
                ("matmul_2d_1d", [3, 4], [4]),
                ("matmul_1d_2d", [4], [4, 5]),
                # broadcast: leading dim broadcasts (a has [1, M, K], b has [B, K, N])
                ("matmul_broadcast", [1, 3, 4], [2, 4, 5]),
            ]:
                g = torch.Generator(device=device)
                g.manual_seed(RNG_SEED + 51 + sum(a_shape) + sum(b_shape))
                a = torch.randn(
                    *a_shape, dtype=torch_dtype(dtype), device=device, generator=g
                )
                b = torch.randn(
                    *b_shape, dtype=torch_dtype(dtype), device=device, generator=g
                )
                a_g = a.detach().clone().requires_grad_(True)
                b_g = b.detach().clone().requires_grad_(True)
                c = torch.matmul(a_g, b_g)
                # matmul forward output may be scalar (1D x 1D) — backward ok.
                c.sum().backward()
                out.append(
                    {
                        "op": "matmul",
                        "tag": tag,
                        "dtype": dtype,
                        "device": device,
                        "a_shape": list(a_shape),
                        "b_shape": list(b_shape),
                        "a_data": to_listf(a),
                        "b_data": to_listf(b),
                        "out_shape": list(c.shape),
                        "out_values": to_listf(c.detach()),
                        "grad_a": to_listf(a_g.grad),
                        "grad_b": to_listf(b_g.grad),
                    }
                )

            # transpose: 2-D 3x4 -> 4x3 (CPU-only, no autograd via this op)
            a = _gen_matrix(3, 4, dtype, device, 71)
            out.append(
                {
                    "op": "transpose",
                    "tag": "transpose_3x4",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [3, 4],
                    "a_data": to_listf(a),
                    "out_shape": [4, 3],
                    "out_values": to_listf(a.T.contiguous()),
                }
            )

            # mm_bt: A (M, K) @ B^T where B is (N, K) — fused linear forward
            # shape. Use a rectangular case to catch transpose bugs.
            a = _gen_matrix(3, 4, dtype, device, 81)
            b = _gen_matrix(5, 4, dtype, device, 83)  # (N=5, K=4)
            a_g = a.detach().clone().requires_grad_(True)
            b_g = b.detach().clone().requires_grad_(True)
            c = a_g @ b_g.T
            c.sum().backward()
            out.append(
                {
                    "op": "mm_bt",
                    "tag": "mm_bt_3x4_5x4",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [3, 4],
                    "b_shape": [5, 4],
                    "a_data": to_listf(a),
                    "b_data": to_listf(b),
                    "out_shape": [3, 5],
                    "out_values": to_listf(c.detach()),
                    "grad_a": to_listf(a_g.grad),
                    "grad_b": to_listf(b_g.grad),
                }
            )

            # linear_fused: input (M, K) @ weight^T (N, K) + bias (N,) -> (M, N)
            inp = _gen_matrix(3, 4, dtype, device, 91)
            w = _gen_matrix(5, 4, dtype, device, 93)
            bias = torch.arange(0.1, 0.6, 0.1, dtype=torch_dtype(dtype), device=device)
            inp_g = inp.detach().clone().requires_grad_(True)
            w_g = w.detach().clone().requires_grad_(True)
            bias_g = bias.detach().clone().requires_grad_(True)
            cf = inp_g @ w_g.T + bias_g
            cf.sum().backward()
            out.append(
                {
                    "op": "linear_fused",
                    "tag": "linear_3x4_5x4_5",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [3, 4],
                    "b_shape": [5, 4],
                    "bias_shape": [5],
                    "a_data": to_listf(inp),
                    "b_data": to_listf(w),
                    "bias_data": to_listf(bias),
                    "out_shape": [3, 5],
                    "out_values": to_listf(cf.detach()),
                    "grad_a": to_listf(inp_g.grad),
                    "grad_b": to_listf(w_g.grad),
                    "grad_bias": to_listf(bias_g.grad),
                }
            )

            # permute_0213: 4-D [d0, d1, d2, d3] -> [d0, d2, d1, d3].
            # CPU-only by ferrotorch's permute_0213 surface; device handled below.
            d0, d1, d2, d3 = 2, 3, 4, 5
            x = _gen_matrix(d0 * d1, d2 * d3, dtype, device, 101).reshape(d0, d1, d2, d3)
            out.append(
                {
                    "op": "permute_0213",
                    "tag": f"permute_{d0}_{d1}_{d2}_{d3}",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [d0, d1, d2, d3],
                    "a_data": to_listf(x),
                    "out_shape": [d0, d2, d1, d3],
                    "out_values": to_listf(x.permute(0, 2, 1, 3).contiguous()),
                }
            )

    return out


# ---------------------------------------------------------------------------
# Cat B — factorizations (qr, svd, cholesky, eigh, eigvalsh, lu, lu_factor,
#                         svdvals)
# ---------------------------------------------------------------------------
#
# For non-unique factors, store only the INPUT (for reconstruction asserts in
# Rust) plus, where available, the canonical scalars (singular values,
# eigenvalues — these ARE unique).


def fixture_factorizations() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            # qr: 4x3 (tall, k=3) — reduced QR shapes are Q (4,3), R (3,3).
            a = _gen_matrix(4, 3, dtype, device, 211)
            qq, rr = torch.linalg.qr(a, mode="reduced")
            out.append(
                {
                    "op": "qr",
                    "tag": "qr_4x3",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [4, 3],
                    "a_data": to_listf(a),
                    "q_shape": list(qq.shape),
                    "r_shape": list(rr.shape),
                    # Recorded but the Rust test does NOT pin Q/R element-by-element
                    # (sign-flipped Q is also a valid QR factorization).
                }
            )

            # qr: square 3x3 (degenerate full-rank case)
            a = _gen_matrix(3, 3, dtype, device, 213)
            out.append(
                {
                    "op": "qr",
                    "tag": "qr_3x3",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [3, 3],
                    "a_data": to_listf(a),
                    "q_shape": [3, 3],
                    "r_shape": [3, 3],
                }
            )

            # qr: 1x1 degenerate — Q = ±1, R = ±a; reconstruction MUST hold.
            a = torch.tensor([[2.5]], dtype=torch_dtype(dtype), device=device)
            out.append(
                {
                    "op": "qr",
                    "tag": "qr_1x1",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [1, 1],
                    "a_data": to_listf(a),
                    "q_shape": [1, 1],
                    "r_shape": [1, 1],
                }
            )

            # svd: 3x2 (tall) — reduced SVD: U (3,2), S (2,), Vh (2,2).
            a = _gen_matrix(3, 2, dtype, device, 221)
            _u, s, _vh = torch.linalg.svd(a, full_matrices=False)
            out.append(
                {
                    "op": "svd",
                    "tag": "svd_3x2",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [3, 2],
                    "a_data": to_listf(a),
                    "u_shape": [3, 2],
                    "s_shape": [2],
                    "vh_shape": [2, 2],
                    "s_values": to_listf(s),
                }
            )

            # svd: 4x4 square
            a = _gen_matrix(4, 4, dtype, device, 223)
            _u, s, _vh = torch.linalg.svd(a, full_matrices=False)
            out.append(
                {
                    "op": "svd",
                    "tag": "svd_4x4",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [4, 4],
                    "a_data": to_listf(a),
                    "u_shape": [4, 4],
                    "s_shape": [4],
                    "vh_shape": [4, 4],
                    "s_values": to_listf(s),
                }
            )

            # svdvals (CPU only in ferrotorch): 3x4
            a = _gen_matrix(3, 4, dtype, device, 225)
            s = torch.linalg.svdvals(a)
            out.append(
                {
                    "op": "svdvals",
                    "tag": "svdvals_3x4",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [3, 4],
                    "a_data": to_listf(a),
                    "out_shape": list(s.shape),
                    "out_values": to_listf(s),
                }
            )

            # cholesky: 3x3 SPD
            a = _spd(3, dtype, device, jitter=1.0)
            _l = torch.linalg.cholesky(a)
            out.append(
                {
                    "op": "cholesky",
                    "tag": "chol_3x3",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [3, 3],
                    "a_data": to_listf(a),
                    "l_shape": [3, 3],
                }
            )
            # cholesky: 1x1 (degenerate, valid)
            a = torch.tensor([[4.0]], dtype=torch_dtype(dtype), device=device)
            out.append(
                {
                    "op": "cholesky",
                    "tag": "chol_1x1",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [1, 1],
                    "a_data": to_listf(a),
                    "l_shape": [1, 1],
                }
            )

            # cholesky_ex: 3x3 SPD (info=0) — CPU only; structured info return.
            a = _spd(3, dtype, "cpu", jitter=1.0)
            out.append(
                {
                    "op": "cholesky_ex",
                    "tag": "chol_ex_3x3",
                    "dtype": dtype,
                    "device": "cpu",
                    "a_shape": [3, 3],
                    "a_data": to_listf(a),
                    "info_expected": 0,
                }
            )

            # eigh: 3x3 SPD (eigenvalues real, eigenvectors orthogonal)
            a = _spd(3, dtype, device, jitter=1.0)
            w, _v = torch.linalg.eigh(a)
            out.append(
                {
                    "op": "eigh",
                    "tag": "eigh_3x3",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [3, 3],
                    "a_data": to_listf(a),
                    "w_shape": [3],
                    "v_shape": [3, 3],
                    "w_values": to_listf(w),
                }
            )
            # eigh: 1x1 trivial
            a = torch.tensor([[7.0]], dtype=torch_dtype(dtype), device=device)
            w, _v = torch.linalg.eigh(a)
            out.append(
                {
                    "op": "eigh",
                    "tag": "eigh_1x1",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [1, 1],
                    "a_data": to_listf(a),
                    "w_shape": [1],
                    "v_shape": [1, 1],
                    "w_values": to_listf(w),
                }
            )

            # eigvalsh: 3x3 SPD
            a = _spd(3, dtype, device, jitter=1.0)
            w = torch.linalg.eigvalsh(a)
            out.append(
                {
                    "op": "eigvalsh",
                    "tag": "eigvalsh_3x3",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [3, 3],
                    "a_data": to_listf(a),
                    "out_shape": [3],
                    "out_values": to_listf(w),
                }
            )

    return out


def fixture_factorizations_cpu_only() -> list[dict[str, Any]]:
    """Factorization-shaped ops that ferrotorch only supports on CPU
    (`require_cpu` guard): lu, eig, eigvals."""
    out: list[dict[str, Any]] = []
    device = "cpu"
    for dtype in DTYPES:
        # lu: 3x3
        a = _gen_matrix(3, 3, dtype, device, 311)
        out.append(
            {
                "op": "lu",
                "tag": "lu_3x3",
                "dtype": dtype,
                "device": device,
                "a_shape": [3, 3],
                "a_data": to_listf(a),
            }
        )

        # eig: 3x3 — complex eigenvalues encoded as [n, 2] real-imag.
        a = _gen_matrix(3, 3, dtype, device, 321)
        # PyTorch returns complex tensors; just pass through to torch.linalg.eig.
        # The reference values are device-resident complex numbers; ferrotorch
        # encodes them as real-imag pairs of length 2 along a trailing dim.
        w_complex, _v_complex = torch.linalg.eig(a)
        w_real_imag = torch.stack([w_complex.real, w_complex.imag], dim=-1).to(
            torch_dtype(dtype)
        )
        out.append(
            {
                "op": "eig",
                "tag": "eig_3x3",
                "dtype": dtype,
                "device": device,
                "a_shape": [3, 3],
                "a_data": to_listf(a),
                "w_shape": [3, 2],
                "v_shape": [3, 3, 2],
                # Sort eigenvalues by real part for sign-stable comparison.
                "w_values_sorted_re": sorted(to_listf(w_real_imag[..., 0])),
            }
        )
        # eigvals (only eigenvalues; complex)
        out.append(
            {
                "op": "eigvals",
                "tag": "eigvals_3x3",
                "dtype": dtype,
                "device": device,
                "a_shape": [3, 3],
                "a_data": to_listf(a),
                "out_shape": [3, 2],
                "w_values_sorted_re": sorted(to_listf(w_real_imag[..., 0])),
            }
        )

        # lu_factor: 3x3
        a = _gen_matrix(3, 3, dtype, device, 331)
        out.append(
            {
                "op": "lu_factor",
                "tag": "lu_factor_3x3",
                "dtype": dtype,
                "device": device,
                "a_shape": [3, 3],
                "a_data": to_listf(a),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Cat C — solvers
# ---------------------------------------------------------------------------


def fixture_solvers() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            # solve: 3x3 SPD, b = (3,)
            a = _spd(3, dtype, device, jitter=1.0)
            b = torch.tensor(
                [1.0, 2.0, 3.0], dtype=torch_dtype(dtype), device=device
            )
            x = torch.linalg.solve(a, b)
            out.append(
                {
                    "op": "solve",
                    "tag": "solve_3x3_3",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [3, 3],
                    "b_shape": [3],
                    "a_data": to_listf(a),
                    "b_data": to_listf(b),
                    "out_shape": [3],
                    "out_values": to_listf(x),
                }
            )
            # solve with multiple RHS: A (3,3), b (3,2)
            b2 = _gen_matrix(3, 2, dtype, device, 411)
            x2 = torch.linalg.solve(a, b2)
            out.append(
                {
                    "op": "solve",
                    "tag": "solve_3x3_3x2",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [3, 3],
                    "b_shape": [3, 2],
                    "a_data": to_listf(a),
                    "b_data": to_listf(b2),
                    "out_shape": [3, 2],
                    "out_values": to_listf(x2),
                }
            )

            # solve_ex: same shape, succeeds (info=0).
            out.append(
                {
                    "op": "solve_ex",
                    "tag": "solve_ex_ok",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [3, 3],
                    "b_shape": [3],
                    "a_data": to_listf(a),
                    "b_data": to_listf(b),
                    "out_shape": [3],
                    "out_values": to_listf(x),
                    "info_expected": 0,
                }
            )

            # lstsq_solve: m=4, n=3 (overdetermined), nrhs=1
            a_tall = _gen_matrix(4, 3, dtype, device, 421)
            b_tall = torch.tensor(
                [1.0, 2.0, 3.0, 4.0], dtype=torch_dtype(dtype), device=device
            )
            sol = torch.linalg.lstsq(a_tall, b_tall).solution
            out.append(
                {
                    "op": "lstsq_solve",
                    "tag": "lstsq_solve_4x3_4",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [4, 3],
                    "b_shape": [4],
                    "a_data": to_listf(a_tall),
                    "b_data": to_listf(b_tall),
                    "out_shape": [3],
                    "out_values": to_listf(sol),
                }
            )

    # CPU-only solvers
    device = "cpu"
    for dtype in DTYPES:
        # solve_triangular: 3x3 lower tri, b (3,)
        a = torch.tensor(
            [[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.5, -1.0, 4.0]],
            dtype=torch_dtype(dtype),
            device=device,
        )
        b = torch.tensor(
            [4.0, 7.0, 9.0], dtype=torch_dtype(dtype), device=device
        )
        x = torch.linalg.solve_triangular(a, b.unsqueeze(-1), upper=False).squeeze(-1)
        out.append(
            {
                "op": "solve_triangular",
                "tag": "solve_tri_lower_3x3_3",
                "dtype": dtype,
                "device": device,
                "a_shape": [3, 3],
                "b_shape": [3],
                "a_data": to_listf(a),
                "b_data": to_listf(b),
                "out_shape": [3],
                "out_values": to_listf(x),
                "upper": False,
                "transpose": False,
                "unit_diagonal": False,
            }
        )

        # ldl: 3x3 SPD
        a = _spd(3, dtype, device, jitter=1.0)
        b = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch_dtype(dtype), device=device
        )
        x = torch.linalg.solve(a, b)
        out.append(
            {
                "op": "ldl_factor",
                "tag": "ldl_3x3",
                "dtype": dtype,
                "device": device,
                "a_shape": [3, 3],
                "a_data": to_listf(a),
            }
        )
        out.append(
            {
                "op": "ldl_solve",
                "tag": "ldl_solve_3x3_3",
                "dtype": dtype,
                "device": device,
                "a_shape": [3, 3],
                "b_shape": [3],
                "a_data": to_listf(a),
                "b_data": to_listf(b),
                "out_shape": [3],
                "out_values": to_listf(x),
            }
        )

        # lstsq (full output): m=4, n=3
        a_tall = _gen_matrix(4, 3, dtype, device, 451)
        b_tall = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch_dtype(dtype), device=device
        )
        result = torch.linalg.lstsq(a_tall, b_tall)
        out.append(
            {
                "op": "lstsq",
                "tag": "lstsq_4x3_4",
                "dtype": dtype,
                "device": device,
                "a_shape": [4, 3],
                "b_shape": [4],
                "a_data": to_listf(a_tall),
                "b_data": to_listf(b_tall),
                "sol_shape": list(result.solution.shape),
                "sol_values": to_listf(result.solution),
            }
        )

        # tensorsolve: a in (2, 3, 6), b in (2, 3) — torch's tensorsolve
        # interprets the trailing axes of a as matching b's shape.
        # Use a simple identity-like tensor where the answer is known.
        # Construct A as a 6x6 invertible matrix reshaped to (2, 3, 6),
        # B as a length-6 vector reshaped to (2, 3); torch.tensorsolve
        # then returns x of shape (6,).
        m6 = _gen_matrix(6, 6, dtype, device, 461)
        b_vec = torch.arange(1.0, 7.0, dtype=torch_dtype(dtype), device=device)
        # tensorsolve expects A.shape = (*B.shape, *X.shape); here B=(2,3),
        # X=(6,) so A=(2, 3, 6). Reshape m6 (6,6) -> (2,3,6).
        a_t = m6.reshape(2, 3, 6)
        b_t = b_vec.reshape(2, 3)
        x_t = torch.linalg.tensorsolve(a_t, b_t)
        out.append(
            {
                "op": "tensorsolve",
                "tag": "tensorsolve_2x3x6_2x3",
                "dtype": dtype,
                "device": device,
                "a_shape": [2, 3, 6],
                "b_shape": [2, 3],
                "a_data": to_listf(a_t),
                "b_data": to_listf(b_t),
                "out_shape": list(x_t.shape),
                "out_values": to_listf(x_t),
            }
        )

        # tensorinv: A square reshaped, ind=2.
        # A has shape (2, 3, 2, 3); flatten to 6x6 should be invertible.
        m6 = _gen_matrix(6, 6, dtype, device, 471)
        a_t = m6.reshape(2, 3, 2, 3)
        # tensorinv with ind=2 inverts treating dims [0:2] as "row" and
        # [2:] as "col"; result has shape (2, 3, 2, 3).
        inv_t = torch.linalg.tensorinv(a_t, ind=2)
        out.append(
            {
                "op": "tensorinv",
                "tag": "tensorinv_2x3x2x3",
                "dtype": dtype,
                "device": device,
                "a_shape": [2, 3, 2, 3],
                "a_data": to_listf(a_t),
                "out_shape": list(inv_t.shape),
                "out_values": to_listf(inv_t),
                "ind": 2,
            }
        )

    return out


# ---------------------------------------------------------------------------
# Cat D — det / norm / inverse / power
# ---------------------------------------------------------------------------


def fixture_det_norm_inv() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    device = "cpu"
    for dtype in DTYPES:
        # det: 3x3 well-conditioned
        a = _gen_matrix(3, 3, dtype, device, 511)
        d = torch.linalg.det(a)
        out.append(
            {
                "op": "det",
                "tag": "det_3x3",
                "dtype": dtype,
                "device": device,
                "a_shape": [3, 3],
                "a_data": to_listf(a),
                "out_shape": [],
                "out_values": to_listf(d),
            }
        )
        # det: 1x1
        a = torch.tensor([[3.0]], dtype=torch_dtype(dtype), device=device)
        out.append(
            {
                "op": "det",
                "tag": "det_1x1",
                "dtype": dtype,
                "device": device,
                "a_shape": [1, 1],
                "a_data": to_listf(a),
                "out_shape": [],
                "out_values": to_listf(torch.linalg.det(a)),
            }
        )

        # slogdet: 3x3
        a = _gen_matrix(3, 3, dtype, device, 521)
        s, ld = torch.linalg.slogdet(a)
        out.append(
            {
                "op": "slogdet",
                "tag": "slogdet_3x3",
                "dtype": dtype,
                "device": device,
                "a_shape": [3, 3],
                "a_data": to_listf(a),
                "sign_value": to_listf(s),
                "logabsdet_value": to_listf(ld),
            }
        )

        # inv: 3x3
        a = _gen_matrix(3, 3, dtype, device, 531)
        inv = torch.linalg.inv(a)
        out.append(
            {
                "op": "inv",
                "tag": "inv_3x3",
                "dtype": dtype,
                "device": device,
                "a_shape": [3, 3],
                "a_data": to_listf(a),
                "out_shape": [3, 3],
                "out_values": to_listf(inv),
            }
        )
        # inv_ex: same input, info=0 (success)
        out.append(
            {
                "op": "inv_ex",
                "tag": "inv_ex_ok",
                "dtype": dtype,
                "device": device,
                "a_shape": [3, 3],
                "a_data": to_listf(a),
                "out_shape": [3, 3],
                "out_values": to_listf(inv),
                "info_expected": 0,
            }
        )

        # matrix_power: 3x3, n=3
        a = _gen_matrix(3, 3, dtype, device, 541)
        out.append(
            {
                "op": "matrix_power",
                "tag": "matrix_power_3x3_3",
                "dtype": dtype,
                "device": device,
                "a_shape": [3, 3],
                "a_data": to_listf(a),
                "n": 3,
                "out_shape": [3, 3],
                "out_values": to_listf(torch.linalg.matrix_power(a, 3)),
            }
        )

        # matrix_norm: Frobenius (default)
        a = _gen_matrix(3, 4, dtype, device, 551)
        out.append(
            {
                "op": "matrix_norm",
                "tag": "matrix_norm_fro_3x4",
                "dtype": dtype,
                "device": device,
                "a_shape": [3, 4],
                "a_data": to_listf(a),
                "out_shape": [],
                "out_values": to_listf(torch.linalg.matrix_norm(a)),
            }
        )

        # vector_norm: ord=2 over a 4-element vector.
        v = torch.tensor(
            [3.0, 4.0, 0.0, 0.0], dtype=torch_dtype(dtype), device=device
        )
        out.append(
            {
                "op": "vector_norm",
                "tag": "vector_norm_l2_4",
                "dtype": dtype,
                "device": device,
                "a_shape": [4],
                "a_data": to_listf(v),
                "ord": 2.0,
                "out_shape": [],
                "out_values": to_listf(torch.linalg.vector_norm(v, ord=2)),
            }
        )

        # matrix_rank: 3x4 known full rank = 3
        a = _gen_matrix(3, 4, dtype, device, 561)
        r = torch.linalg.matrix_rank(a)
        out.append(
            {
                "op": "matrix_rank",
                "tag": "rank_3x4",
                "dtype": dtype,
                "device": device,
                "a_shape": [3, 4],
                "a_data": to_listf(a),
                "rank_expected": int(r.item()),
            }
        )

        # cond: 3x3 (p=2)
        a = _spd(3, dtype, device, jitter=1.0)
        c = torch.linalg.cond(a, p=2)
        out.append(
            {
                "op": "cond",
                "tag": "cond_3x3_p2",
                "dtype": dtype,
                "device": device,
                "a_shape": [3, 3],
                "a_data": to_listf(a),
                "p": 2.0,
                "out_shape": [],
                "out_values": to_listf(c),
            }
        )

        # pinv: 3x4 (CPU only)
        a = _gen_matrix(3, 4, dtype, device, 571)
        p = torch.linalg.pinv(a)
        out.append(
            {
                "op": "pinv",
                "tag": "pinv_3x4",
                "dtype": dtype,
                "device": device,
                "a_shape": [3, 4],
                "a_data": to_listf(a),
                "out_shape": list(p.shape),
                "out_values": to_listf(p),
            }
        )

    return out


# ---------------------------------------------------------------------------
# Cat E — misc (cross / multi_dot / diagonal / householder / matrix_exp)
# ---------------------------------------------------------------------------


def fixture_misc() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    device = "cpu"
    for dtype in DTYPES:
        # cross: length-3 vectors
        a = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch_dtype(dtype), device=device
        )
        b = torch.tensor(
            [4.0, 5.0, 6.0], dtype=torch_dtype(dtype), device=device
        )
        out.append(
            {
                "op": "cross",
                "tag": "cross_3",
                "dtype": dtype,
                "device": device,
                "a_shape": [3],
                "b_shape": [3],
                "a_data": to_listf(a),
                "b_data": to_listf(b),
                "axis": -1,
                "out_shape": [3],
                "out_values": to_listf(torch.linalg.cross(a, b)),
            }
        )

        # multi_dot: [3x4] @ [4x5] @ [5x6] -> [3x6]
        m1 = _gen_matrix(3, 4, dtype, device, 611)
        m2 = _gen_matrix(4, 5, dtype, device, 613)
        m3 = _gen_matrix(5, 6, dtype, device, 615)
        out.append(
            {
                "op": "multi_dot",
                "tag": "multi_dot_3_4_5_6",
                "dtype": dtype,
                "device": device,
                "shapes": [[3, 4], [4, 5], [5, 6]],
                "data": [to_listf(m1), to_listf(m2), to_listf(m3)],
                "out_shape": [3, 6],
                "out_values": to_listf(torch.linalg.multi_dot([m1, m2, m3])),
            }
        )

        # diagonal: 3x4, offset=0 and offset=1
        a = _gen_matrix(3, 4, dtype, device, 621)
        out.append(
            {
                "op": "diagonal",
                "tag": "diag_3x4_off0",
                "dtype": dtype,
                "device": device,
                "a_shape": [3, 4],
                "a_data": to_listf(a),
                "offset": 0,
                "out_shape": [3],
                "out_values": to_listf(torch.diagonal(a, offset=0)),
            }
        )
        out.append(
            {
                "op": "diagonal",
                "tag": "diag_3x4_off1",
                "dtype": dtype,
                "device": device,
                "a_shape": [3, 4],
                "a_data": to_listf(a),
                "offset": 1,
                "out_shape": [3],
                "out_values": to_listf(torch.diagonal(a, offset=1)),
            }
        )

        # householder_product: V (4, 2), tau (2,) — round-trip from QR.
        a = _gen_matrix(4, 2, dtype, device, 631)
        # Get the implicit Householder representation via geqrf.
        v_tau = torch.geqrf(a)
        v = v_tau.a
        tau = v_tau.tau
        # PyTorch's householder_product expects the V matrix where the
        # below-diagonal of V (cols < min(m,n)) holds the reflectors and
        # the diagonal/above is irrelevant — we pass v directly.
        q = torch.linalg.householder_product(v, tau)
        out.append(
            {
                "op": "householder_product",
                "tag": "hh_4x2",
                "dtype": dtype,
                "device": device,
                "v_shape": [4, 2],
                "tau_shape": [2],
                "v_data": to_listf(v),
                "tau_data": to_listf(tau),
                "out_shape": list(q.shape),
                "out_values": to_listf(q),
            }
        )

        # matrix_exp: 3x3 small scaled matrix (avoid overflow)
        a = _gen_matrix(3, 3, dtype, device, 641) * 0.5
        out.append(
            {
                "op": "matrix_exp",
                "tag": "expm_3x3_small",
                "dtype": dtype,
                "device": device,
                "a_shape": [3, 3],
                "a_data": to_listf(a),
                "out_shape": [3, 3],
                "out_values": to_listf(torch.linalg.matrix_exp(a)),
            }
        )

    return out


# ---------------------------------------------------------------------------
# Edge cases — singular matrix paths, empty
# ---------------------------------------------------------------------------


def fixture_edge_cases() -> list[dict[str, Any]]:
    """Singular-matrix paths (PyTorch raises RuntimeError; ferrotorch must
    return Err) and empty paths.

    For these we don't store a "correct answer" — the test asserts the call
    returns Err. We still record the input so the test is data-driven."""
    out: list[dict[str, Any]] = []
    for dtype in DTYPES:
        # Singular 3x3 (rank=2, last row = 2 * first).
        sing = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [2.0, 4.0, 6.0]],
            dtype=torch_dtype(dtype),
            device="cpu",
        )
        out.append(
            {
                "op": "inv_singular",
                "tag": "edge",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": [3, 3],
                "a_data": to_listf(sing),
                "expect_err": True,
            }
        )
        b = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch_dtype(dtype), device="cpu"
        )
        out.append(
            {
                "op": "solve_singular",
                "tag": "edge",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": [3, 3],
                "b_shape": [3],
                "a_data": to_listf(sing),
                "b_data": to_listf(b),
                "expect_err": True,
            }
        )

        # Non-SPD cholesky failure — the input is symmetric but indefinite.
        # Eigenvalues will straddle zero. cholesky must Err.
        non_spd = torch.tensor(
            [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=torch_dtype(dtype),
            device="cpu",
        )
        out.append(
            {
                "op": "cholesky_singular",
                "tag": "edge",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": [3, 3],
                "a_data": to_listf(non_spd),
                "expect_err": True,
            }
        )

    return out


# ---------------------------------------------------------------------------
# Stress lanes — rank deficiency, repeated singular values, 64x64
# (CORE-199 / #1893)
# ---------------------------------------------------------------------------
#
# Comparison contracts (documented per case, enforced by the existing
# gauge-aware handlers in conformance_linalg.rs):
#   * svd:      singular values pinned element-wise (unique up to sort) +
#               A ≈ U @ diag(S) @ Vh reconstruction. U/V columns are gauge-
#               free (sign, and arbitrary rotation inside a repeated-σ
#               subspace) and are NEVER compared element-wise.
#   * qr:       A ≈ Q @ R reconstruction only (Q sign columns are gauge-free;
#               for rank-deficient A the factorization is non-unique but the
#               product contract still holds).
#   * cholesky: A ≈ L @ L^T reconstruction (L unique for SPD, but the
#               reconstruction contract is what survives rounding).
#   * solve:    x pinned element-wise against torch (unique solution — the
#               stress matrices are deliberately well-conditioned SPD+shift
#               so the comparison is meaningful at f32).
#   * det:      scalar pinned against torch. The 64x64 input is built with
#               log-balanced eigenvalues so |det| stays O(1) in f32.
#   * rank-deficient solve / cholesky of a PSD-singular matrix: expect_err
#               (torch raises RuntimeError; ferrotorch must Err).


def _orthogonal(n: int, dtype: str, device: str, seed_offset: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(RNG_SEED + seed_offset)
    m = torch.randn(n, n, dtype=torch_dtype(dtype), generator=g)
    q, _ = torch.linalg.qr(m)
    return q.to(device)


def _rank_deficient(
    rows: int, cols: int, rank: int, dtype: str, device: str, seed_offset: int
) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(RNG_SEED + seed_offset)
    b = torch.randn(rows, rank, dtype=torch_dtype(dtype), generator=g)
    c = torch.randn(rank, cols, dtype=torch_dtype(dtype), generator=g)
    return (b @ c).to(device)


def fixture_stress() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            td = torch_dtype(dtype)

            # --- svd: rank-deficient 6x4 (rank 2) ------------------------
            a = _rank_deficient(6, 4, 2, dtype, device, 611)
            _u, s, _vh = torch.linalg.svd(a, full_matrices=False)
            out.append(
                {
                    "op": "svd",
                    "tag": "svd_rankdef_6x4r2",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [6, 4],
                    "a_data": to_listf(a),
                    "u_shape": [6, 4],
                    "s_shape": [4],
                    "vh_shape": [4, 4],
                    "s_values": to_listf(s),
                }
            )

            # --- svd: repeated singular values (σ = [3, 3, 3, 0.5]) ------
            # A = Q1 @ diag(σ) @ Q2^T. The repeated-σ subspace makes U/V
            # non-unique BEYOND sign — only S + reconstruction are valid
            # comparisons (the handlers already do exactly that).
            q1 = _orthogonal(4, dtype, device, 613)
            q2 = _orthogonal(4, dtype, device, 617)
            sigma = torch.tensor([3.0, 3.0, 3.0, 0.5], dtype=td, device=device)
            a = q1 @ torch.diag(sigma) @ q2.mT
            _u, s, _vh = torch.linalg.svd(a, full_matrices=False)
            out.append(
                {
                    "op": "svd",
                    "tag": "svd_repeated_sigma_4x4",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [4, 4],
                    "a_data": to_listf(a),
                    "u_shape": [4, 4],
                    "s_shape": [4],
                    "vh_shape": [4, 4],
                    "s_values": to_listf(s),
                }
            )

            # --- svd: 64x64 ----------------------------------------------
            a = _gen_matrix(64, 64, dtype, "cpu", 619).to(device)
            _u, s, _vh = torch.linalg.svd(a, full_matrices=False)
            out.append(
                {
                    "op": "svd",
                    "tag": "svd_64x64",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [64, 64],
                    "a_data": to_listf(a),
                    "u_shape": [64, 64],
                    "s_shape": [64],
                    "vh_shape": [64, 64],
                    "s_values": to_listf(s),
                }
            )

            # --- qr: rank-deficient 4x3 (rank 2) and 64x64 ----------------
            a = _rank_deficient(4, 3, 2, dtype, device, 631)
            qq, rr = torch.linalg.qr(a, mode="reduced")
            out.append(
                {
                    "op": "qr",
                    "tag": "qr_rankdef_4x3r2",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [4, 3],
                    "a_data": to_listf(a),
                    "q_shape": list(qq.shape),
                    "r_shape": list(rr.shape),
                }
            )
            a = _gen_matrix(64, 64, dtype, "cpu", 633).to(device)
            qq, rr = torch.linalg.qr(a, mode="reduced")
            out.append(
                {
                    "op": "qr",
                    "tag": "qr_64x64",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [64, 64],
                    "a_data": to_listf(a),
                    "q_shape": list(qq.shape),
                    "r_shape": list(rr.shape),
                }
            )

            # --- cholesky: 64x64 SPD (G @ G^T / n + I, κ small) ------------
            g = torch.Generator(device="cpu")
            g.manual_seed(RNG_SEED + 641)
            gm = torch.randn(64, 64, dtype=td, generator=g)
            a = (gm @ gm.mT / 64.0 + torch.eye(64, dtype=td)).to(device)
            torch.linalg.cholesky(a)  # sanity: SPD
            out.append(
                {
                    "op": "cholesky",
                    "tag": "chol_64x64",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [64, 64],
                    "a_data": to_listf(a),
                    "l_shape": [64, 64],
                }
            )

            # --- solve: 64x64 SPD+shift (κ ~ few; unique, f32-comparable) --
            b = torch.tensor(
                [((i * 7) % 23) * 0.25 - 2.0 for i in range(64)],
                dtype=td,
                device=device,
            )
            x = torch.linalg.solve(a, b)
            out.append(
                {
                    "op": "solve",
                    "tag": "solve_64x64",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [64, 64],
                    "b_shape": [64],
                    "a_data": to_listf(a),
                    "b_data": to_listf(b),
                    "out_shape": [64],
                    "out_values": to_listf(x),
                }
            )

    # CPU-only stress rows (det is CPU-only in ferrotorch; the *_singular
    # error contracts are CPU-only like the existing edge fixtures).
    for dtype in DTYPES:
        td = torch_dtype(dtype)

        # --- det: rank-deficient 4x4 (det = 0 analytically) ----------------
        a = _rank_deficient(4, 4, 2, dtype, "cpu", 651)
        out.append(
            {
                "op": "det",
                "tag": "det_rankdef_4x4r2",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": [4, 4],
                "a_data": to_listf(a),
                "out_shape": [],
                "out_values": to_listf(torch.linalg.det(a)),
            }
        )

        # --- det: 64x64 with log-balanced eigenvalues ----------------------
        # A = Q D Q^T with D pairing λ and 1/λ so |det| = O(1) and the value
        # is comparable in f32 without overflow.
        q = _orthogonal(64, dtype, "cpu", 653)
        lams = []
        for i in range(32):
            lam = 1.0 + (i % 8) * 0.25
            lams += [lam, 1.0 / lam]
        d = torch.tensor(lams, dtype=td)
        a = q @ torch.diag(d) @ q.mT
        out.append(
            {
                "op": "det",
                "tag": "det_64x64_balanced",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": [64, 64],
                "a_data": to_listf(a),
                "out_shape": [],
                "out_values": to_listf(torch.linalg.det(a)),
            }
        )

        # --- solve on a rank-deficient A: torch raises; ferrotorch must Err.
        sing = _rank_deficient(4, 4, 2, dtype, "cpu", 655)
        b4 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=td)
        out.append(
            {
                "op": "solve_singular",
                "tag": "stress_rankdef_4x4r2",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": [4, 4],
                "b_shape": [4],
                "a_data": to_listf(sing),
                "b_data": to_listf(b4),
                "expect_err": True,
            }
        )

        # --- cholesky on a PSD-singular matrix: torch raises; must Err. ----
        psd_sing = sing @ sing.mT  # rank 2, PSD, not PD
        out.append(
            {
                "op": "cholesky_singular",
                "tag": "stress_psd_rankdef_4x4",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": [4, 4],
                "a_data": to_listf(psd_sing),
                "expect_err": True,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def main() -> int:
    fixtures: list[dict[str, Any]] = []
    fixtures += fixture_matmul_forwards()
    fixtures += fixture_factorizations()
    fixtures += fixture_factorizations_cpu_only()
    fixtures += fixture_solvers()
    fixtures += fixture_det_norm_inv()
    fixtures += fixture_misc()
    fixtures += fixture_edge_cases()
    fixtures += fixture_stress()

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
