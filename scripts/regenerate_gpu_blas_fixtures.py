#!/usr/bin/env python3
"""Regenerate reference fixtures for ferrotorch-gpu C8.3 BLAS-family conformance.

Tracking: C8.3 (BLAS family FFI binding layer + Rust wrapper correctness).

Output:
    ferrotorch-gpu/tests/conformance/fixtures/gpu_blas_family.json

Coverage (5 modules, Layer-2 fixtures for FFI lifecycle + binding correctness):

* blas.rs  — cuBLAS handle init/destroy; SGEMM + DGEMM 4×4; batched SGEMM 2×4×4.
* cufft.rs — cuFFT C2C plan lifecycle; 4-point forward+inverse round-trip (f32 + f64);
             R2C / C2R 4-point round-trip (f32 + f64).
* cusolver.rs — cuSOLVER DnHandle lifecycle; SVD of 4×4 (f32 + f64); Cholesky of 4×4 SPD.
* cusparselt.rs — cuSPARSELt handle init/destroy; 2:4 SpMM round-trip on 8×8 f32
                  (minimum aligned dimension for cuSPARSELt).
* bf16.rs  — bf16 mul / add / silu / relu elementwise round-trip for N=16.

All inputs are small synthetic matrices so the fixtures are compact and deterministic.
PyTorch / numpy is used purely as the ground-truth oracle; we do NOT exercise CUDA in
this Python script — the fixtures capture CPU-resident expected values that the Rust
conformance tests compare against after moving tensors to GPU and reading back.

Usage:
    pip install torch numpy
    python3 scripts/regenerate_gpu_blas_fixtures.py
"""

from __future__ import annotations

import datetime
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = (
    REPO_ROOT
    / "ferrotorch-gpu"
    / "tests"
    / "conformance"
    / "fixtures"
    / "gpu_blas_family.json"
)

TORCH_VERSION = torch.__version__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fl(t: torch.Tensor) -> list[float]:
    """Flatten a tensor to a list of Python floats."""
    return t.detach().cpu().float().flatten().tolist()


def f64l(t: torch.Tensor) -> list[float]:
    return t.detach().cpu().double().flatten().tolist()


def f32_to_bf16_bits(x: float) -> int:
    """Convert a Python float to bf16 via round-to-nearest-even; return the raw u16 bits."""
    f32_bits = struct.unpack("<I", struct.pack("<f", x))[0]
    lsb = (f32_bits >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    rounded = (f32_bits + rounding_bias) >> 16
    return rounded & 0xFFFF


def bf16_bits_to_float(bits: int) -> float:
    """Convert raw bf16 u16 bits back to a Python float."""
    f32_bits = bits << 16
    return struct.unpack("<f", struct.pack("<I", f32_bits))[0]


def f32_vec_to_bf16_bits(xs: list[float]) -> list[int]:
    return [f32_to_bf16_bits(x) for x in xs]


def bf16_bits_vec_to_float(bits: list[int]) -> list[float]:
    return [bf16_bits_to_float(b) for b in bits]


# ---------------------------------------------------------------------------
# Module: blas.rs fixtures
# ---------------------------------------------------------------------------

def make_blas_fixtures() -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []

    # 4×4 SGEMM: C = A @ B, row-major.
    torch.manual_seed(42)
    A = torch.randn(4, 4, dtype=torch.float32)
    B = torch.randn(4, 4, dtype=torch.float32)
    C = A @ B
    fixtures.append({
        "module": "blas",
        "op": "gpu_matmul_f32",
        "tag": "4x4_f32",
        "m": 4, "k": 4, "n": 4,
        "a_data": fl(A),
        "b_data": fl(B),
        "expected": fl(C),
    })

    # 4×4 DGEMM.
    Ad = A.double()
    Bd = B.double()
    Cd = Ad @ Bd
    fixtures.append({
        "module": "blas",
        "op": "gpu_matmul_f64",
        "tag": "4x4_f64",
        "m": 4, "k": 4, "n": 4,
        "a_data": f64l(Ad),
        "b_data": f64l(Bd),
        "expected": f64l(Cd),
    })

    # Non-square matmul [3,4] @ [4,5] = [3,5]
    torch.manual_seed(7)
    An = torch.randn(3, 4, dtype=torch.float32)
    Bn = torch.randn(4, 5, dtype=torch.float32)
    Cn = An @ Bn
    fixtures.append({
        "module": "blas",
        "op": "gpu_matmul_f32",
        "tag": "3x4_x_4x5_f32",
        "m": 3, "k": 4, "n": 5,
        "a_data": fl(An),
        "b_data": fl(Bn),
        "expected": fl(Cn),
    })

    # Batched SGEMM [2, 4, 4] x [2, 4, 4] = [2, 4, 4].
    torch.manual_seed(13)
    Ab = torch.randn(2, 4, 4, dtype=torch.float32)
    Bb = torch.randn(2, 4, 4, dtype=torch.float32)
    Cb = torch.bmm(Ab, Bb)
    fixtures.append({
        "module": "blas",
        "op": "gpu_bmm_f32",
        "tag": "batch2_4x4_f32",
        "batch": 2, "m": 4, "k": 4, "n": 4,
        "a_data": fl(Ab),
        "b_data": fl(Bb),
        "expected": fl(Cb),
    })

    return fixtures


# ---------------------------------------------------------------------------
# Module: cufft.rs fixtures
# ---------------------------------------------------------------------------

def make_cufft_fixtures() -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []

    # 4-point C2C forward, f32.  Input: [1, 2, 3, 4] real.
    # Layout: interleaved [re, im, re, im, ...].
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    xc = torch.complex(x, torch.zeros(4))
    Xc = torch.fft.fft(xc)
    # Flatten to interleaved re/im pairs.
    interleaved_out = []
    for c in Xc:
        interleaved_out.append(c.real.item())
        interleaved_out.append(c.imag.item())
    interleaved_in = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]
    fixtures.append({
        "module": "cufft",
        "op": "gpu_fft_c2c_f32",
        "tag": "4pt_forward_f32",
        "batch": 1, "n": 4,
        "inverse": False,
        "input": interleaved_in,
        "expected": interleaved_out,
    })

    # 4-point C2C inverse round-trip: ifft(fft(x)) ≈ x.
    # We store the FFT output as input and the original as expected (after IFFT).
    ifft_out = torch.fft.ifft(Xc)
    interleaved_ifft = []
    for c in ifft_out:
        interleaved_ifft.append(c.real.item())
        interleaved_ifft.append(c.imag.item())
    fixtures.append({
        "module": "cufft",
        "op": "gpu_fft_c2c_f32",
        "tag": "4pt_inverse_f32",
        "batch": 1, "n": 4,
        "inverse": True,
        "input": interleaved_out,
        "expected": interleaved_ifft,
    })

    # 4-point C2C forward, f64.
    xd = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    xcd = torch.complex(xd, torch.zeros(4, dtype=torch.float64))
    Xcd = torch.fft.fft(xcd)
    interleaved_in_d = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]
    interleaved_out_d = []
    for c in Xcd:
        interleaved_out_d.append(c.real.item())
        interleaved_out_d.append(c.imag.item())
    fixtures.append({
        "module": "cufft",
        "op": "gpu_fft_c2c_f64",
        "tag": "4pt_forward_f64",
        "batch": 1, "n": 4,
        "inverse": False,
        "input": interleaved_in_d,
        "expected": interleaved_out_d,
    })

    # 4-point R2C (rfft), f32.
    xr = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    Xr = torch.fft.rfft(xr)  # 4/2+1 = 3 complex bins.
    in_r2c = [1.0, 2.0, 3.0, 4.0]
    out_r2c = []
    for c in Xr:
        out_r2c.append(c.real.item())
        out_r2c.append(c.imag.item())
    fixtures.append({
        "module": "cufft",
        "op": "gpu_rfft_r2c_f32",
        "tag": "4pt_r2c_f32",
        "batch": 1, "n": 4,
        "input": in_r2c,
        "expected": out_r2c,
    })

    # 4-point C2R (irfft), f32: irfft(rfft(x), n=4) round-trip.
    xrr = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    Xrr = torch.fft.rfft(xrr)
    in_c2r = []
    for c in Xrr:
        in_c2r.append(c.real.item())
        in_c2r.append(c.imag.item())
    out_c2r = torch.fft.irfft(Xrr, n=4).tolist()
    fixtures.append({
        "module": "cufft",
        "op": "gpu_irfft_c2r_f32",
        "tag": "4pt_c2r_f32",
        "batch": 1, "n_out": 4,
        "input": in_c2r,
        "expected": out_c2r,
    })

    return fixtures


# ---------------------------------------------------------------------------
# Module: cusolver.rs fixtures
# ---------------------------------------------------------------------------

def make_cusolver_fixtures() -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []

    # SVD of a 4×4 matrix, f32.  We only store S (singular values) as ground
    # truth since U and V are non-unique; the test validates reconstruction
    # via ||U @ diag(S) @ Vh - A||_F < tol.
    torch.manual_seed(17)
    M = torch.randn(4, 4, dtype=torch.float32)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    fixtures.append({
        "module": "cusolver",
        "op": "gpu_svd_f32",
        "tag": "4x4_f32",
        "m": 4, "n": 4,
        "input": fl(M),
        "expected_s": fl(S),
        # Also store A so the test can do the reconstruction check.
        "a_data": fl(M),
    })

    # SVD of a 4×4 matrix, f64.
    Md = M.double()
    Ud, Sd, Vhd = torch.linalg.svd(Md, full_matrices=False)
    fixtures.append({
        "module": "cusolver",
        "op": "gpu_svd_f64",
        "tag": "4x4_f64",
        "m": 4, "n": 4,
        "input": f64l(Md),
        "expected_s": f64l(Sd),
        "a_data": f64l(Md),
    })

    # Cholesky of a 4×4 SPD matrix, f32.
    # A = L @ L^T where L is lower triangular.
    torch.manual_seed(31)
    tmp = torch.randn(4, 4, dtype=torch.float32)
    SPD = tmp @ tmp.T + 4.0 * torch.eye(4, dtype=torch.float32)
    L_ref = torch.linalg.cholesky(SPD, upper=False)
    # Validation: L @ L^T must equal SPD within tolerance.
    recon = L_ref @ L_ref.T
    fixtures.append({
        "module": "cusolver",
        "op": "gpu_cholesky_f32",
        "tag": "4x4_spd_f32",
        "n": 4,
        "input": fl(SPD),
        "expected_l": fl(L_ref),
        # Store the reconstruction for double-checking; test verifies ||L@L^T - A|| < tol.
        "spd_data": fl(SPD),
    })

    # Cholesky of a 4×4 SPD matrix, f64.
    SPDd = SPD.double()
    Ld_ref = torch.linalg.cholesky(SPDd, upper=False)
    fixtures.append({
        "module": "cusolver",
        "op": "gpu_cholesky_f64",
        "tag": "4x4_spd_f64",
        "n": 4,
        "input": f64l(SPDd),
        "expected_l": f64l(Ld_ref),
        "spd_data": f64l(SPDd),
    })

    return fixtures


# ---------------------------------------------------------------------------
# Module: cusparselt.rs fixtures
# ---------------------------------------------------------------------------

def make_cusparselt_fixtures() -> list[dict[str, Any]]:
    """
    cuSPARSELt requires dimensions that are multiples of 8 for FP16/BF16
    (or 4 for FP32) and operates with 2:4 structured sparsity on the B
    operand.  For the fixture we use an 8×8 f32 matrix where B has exactly
    the 2:4 pattern (2 non-zeros in every group of 4 consecutive values).

    The fixture stores:
    - a_data: dense [8, 8] row-major A.
    - b_decompressed: dense [8, 8] row-major B with zeros at masked positions.
    - expected: dense [8, 8] row-major C = A @ B_decompressed.

    The Rust test can validate that gpu_sparse_matmul_24(A, B_decompressed)
    produces C within tolerance — the cuSPARSELt runtime compresses B
    internally.

    Note: we use the *decompressed* B (values present at 2:4 positions, zeros
    elsewhere) as both input AND as the RHS for the reference matmul, because
    that matches ferrotorch's contract: `b_dense_decompressed` is the
    zero-padded form.
    """
    fixtures: list[dict[str, Any]] = []

    torch.manual_seed(55)
    # 8×8 dense A.
    A = torch.randn(8, 8, dtype=torch.float32)

    # Build 2:4-sparse B: for every group of 4 consecutive elements in each row,
    # keep exactly 2 and zero the other 2.
    B_dense = torch.randn(8, 8, dtype=torch.float32)
    B_24 = B_dense.clone()
    for row in range(8):
        for group_start in range(0, 8, 4):
            group = B_dense[row, group_start : group_start + 4]
            # Zero out the two smallest-magnitude elements.
            indices = torch.argsort(group.abs())
            B_24[row, group_start + indices[0]] = 0.0
            B_24[row, group_start + indices[1]] = 0.0

    # Reference: C = A @ B_24 (using the decompressed/zero-padded B).
    C_ref = A @ B_24
    fixtures.append({
        "module": "cusparselt",
        "op": "gpu_sparse_matmul_24",
        "tag": "8x8_f32_24sparse",
        "m": 8, "k": 8, "n": 8,
        "a_data": fl(A),
        "b_decompressed": fl(B_24),
        "expected": fl(C_ref),
        "note": (
            "B has 2:4 sparsity (2 non-zeros per group of 4). "
            "Reference C = A @ B_decompressed (zero-padded B). "
            "cuSPARSELt compresses B internally; tol=5e-3 for TF32 mode."
        ),
    })

    return fixtures


# ---------------------------------------------------------------------------
# Module: bf16.rs fixtures
# ---------------------------------------------------------------------------

def make_bf16_fixtures() -> list[dict[str, Any]]:
    """
    bf16 ops use PTX kernels that convert u16 ↔ f32 per element.  We store
    inputs as raw bf16 bit patterns (u16 integers) and expected outputs as
    raw bf16 bit patterns, matching the ferrotorch buffer representation.

    N=16 for all ops.
    """
    fixtures: list[dict[str, Any]] = []
    N = 16
    torch.manual_seed(99)

    # Source values as f32, then convert to bf16 for the fixtures.
    a_f32 = torch.randn(N, dtype=torch.float32)
    b_f32 = torch.randn(N, dtype=torch.float32)

    # Clamp to bf16-representable range to avoid inf.
    a_f32 = torch.clamp(a_f32, -4.0, 4.0)
    b_f32 = torch.clamp(b_f32, -4.0, 4.0)

    def to_bf16_bits(t: torch.Tensor) -> list[int]:
        bf = t.to(torch.bfloat16)
        # View as int16, then & 0xFFFF to get u16 semantics.
        raw = bf.view(torch.int16).tolist()
        return [r & 0xFFFF for r in raw]

    def from_bf16_bits(bits: list[int]) -> torch.Tensor:
        # bits are unsigned u16 values; cast through int32 to avoid overflow,
        # then view as int16 (same bit pattern) before reinterpreting as bfloat16.
        int32 = torch.tensor(bits, dtype=torch.int32)
        int16 = int32.to(torch.int16)
        return int16.view(torch.bfloat16).float()

    a_bits = to_bf16_bits(a_f32)
    b_bits = to_bf16_bits(b_f32)

    # Re-decode to get the bf16-rounded input values for reference computation.
    a_bf16 = from_bf16_bits(a_bits)
    b_bf16 = from_bf16_bits(b_bits)

    # mul: out = a * b
    mul_out = a_bf16 * b_bf16
    mul_bits = to_bf16_bits(mul_out)
    fixtures.append({
        "module": "bf16",
        "op": "gpu_mul_bf16",
        "tag": "n16_mul",
        "n": N,
        "a_bits": a_bits,
        "b_bits": b_bits,
        "expected_bits": mul_bits,
    })

    # add: out = a + b
    add_out = a_bf16 + b_bf16
    add_bits = to_bf16_bits(add_out)
    fixtures.append({
        "module": "bf16",
        "op": "gpu_add_bf16",
        "tag": "n16_add",
        "n": N,
        "a_bits": a_bits,
        "b_bits": b_bits,
        "expected_bits": add_bits,
    })

    # silu: out = x * sigmoid(x)
    silu_in = torch.clamp(a_bf16, -3.0, 3.0)
    silu_bits_in = to_bf16_bits(silu_in)
    silu_out = silu_in * torch.sigmoid(silu_in)
    silu_bits_out = to_bf16_bits(silu_out)
    fixtures.append({
        "module": "bf16",
        "op": "gpu_silu_bf16",
        "tag": "n16_silu",
        "n": N,
        "a_bits": silu_bits_in,
        "expected_bits": silu_bits_out,
    })

    # relu: out = max(0, x)
    relu_out = torch.relu(a_bf16)
    relu_bits = to_bf16_bits(relu_out)
    fixtures.append({
        "module": "bf16",
        "op": "gpu_relu_bf16",
        "tag": "n16_relu",
        "n": N,
        "a_bits": a_bits,
        "expected_bits": relu_bits,
    })

    return fixtures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"torch {TORCH_VERSION} — generating GPU BLAS-family fixtures …")

    fixtures: list[dict[str, Any]] = []
    fixtures.extend(make_blas_fixtures())
    fixtures.extend(make_cufft_fixtures())
    fixtures.extend(make_cusolver_fixtures())
    fixtures.extend(make_cusparselt_fixtures())
    fixtures.extend(make_bf16_fixtures())

    document = {
        "version": "1",
        "generated": datetime.datetime.utcnow().isoformat() + "Z",
        "torch_version": TORCH_VERSION,
        "python_version": sys.version,
        "description": (
            "C8.3 BLAS-family conformance fixtures. "
            "Covers ferrotorch-gpu: blas, cufft, cusolver, cusparselt, bf16."
        ),
        "fixtures": fixtures,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FIXTURE_PATH, "w") as f:
        json.dump(document, f, indent=2)

    print(f"Wrote {len(fixtures)} fixtures to {FIXTURE_PATH}")
    by_module: dict[str, int] = {}
    for fix in fixtures:
        by_module[fix["module"]] = by_module.get(fix["module"], 0) + 1
    for mod, cnt in sorted(by_module.items()):
        print(f"  {mod}: {cnt} fixture(s)")


if __name__ == "__main__":
    main()
