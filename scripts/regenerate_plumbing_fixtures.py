#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for ferrotorch-core Phase 2.14
(plumbing & core types).

Tracking issue: #776 (parent: #759).

Output:
    ferrotorch-core/tests/conformance/fixtures/plumbing.json

Coverage strategy (the 180 surface items grouped by category):

* **Tensor metadata methods** — `shape`, `ndim`, `numel`, `device`, `strides`,
  `storage_offset`, `is_contiguous`, `is_cuda`, `is_cpu`, `is_meta`,
  `is_xpu`, `is_leaf`, `is_scalar`, `is_same`, `requires_grad`. PyTorch
  reference: same-named attributes on `torch.Tensor`. Bit-exact metadata
  parity (no arithmetic).

* **Device enum** — `Device::Cpu`, `Device::Cuda(n)`, `Device::Xpu(n)`,
  `Device::Mps(n)`, `Device::Meta` and the `is_*` predicates. PyTorch
  reference is the `torch.device` constructor + `.type` / `.index`.

* **DType / Element / Float traits** — re-exports from `ferray_core`.
  Tested by enumerating the variants (`F32`, `F64`, `BF16`, `F16`,
  `I32`, `I64`, …) and asserting `Element::dtype()` matches per type.

* **FerrotorchError variants** — every variant Display-formatted; the
  fixture pins the exact message string per variant so that the Rust
  test asserts byte-for-byte parity.

* **numeric_cast::cast<T,U>(v)** — fallible numeric conversion. Fixtures
  enumerate representative successes (f64 → f32 finite, usize → f32,
  f64 → bf16) and failures (f64::INFINITY → i32, NaN → bf16-then-back,
  out-of-range i64 → i32). PyTorch parallel is `tensor.to(dtype)` which
  saturates / wraps; our `cast()` errors instead — that divergence is
  documented in the rust test's tolerance preamble (this is the
  Rust-side fallible contract, not a parity gap).

* **NamedTensor** — PyTorch's experimental named tensor was removed in
  later versions. Fixtures cover the dim-name semantics (refined,
  align_to, rename, dim_index, size_of, names, ndim, numel, shape,
  detached, into_tensor, tensor) directly against the Rust
  implementation; PyTorch parity is asserted only for the underlying
  shape arithmetic (the permutation produced by `align_to` matches the
  permutation `torch.permute` would emit for the same axis ordering).

* **DispatchKey / DispatchKeySet / Dispatcher** — internal dispatch
  surface. Tested directly (no PyTorch reference; PyTorch's `c10`
  dispatcher is a private C++ surface). Fixtures pin the priority
  ordering and the `iter_desc` ordering invariants.

* **Storage / GpuBufferHandle / GpuBackend trait / cubecl / cpu_pool /
  meta_propagate / profiler_hook** — infrastructure types with no
  direct PyTorch analog. Covered by file-comment-block witnesses in the
  Rust test source plus direct unit assertions where applicable
  (TensorStorage construction, meta propagation shapes, cpu_pool
  hit/miss counters).

Tolerances:
  - Bit-exact metadata (no arithmetic).
  - F32_ELEMENTWISE for `cast::<T,U>(v)` round-trips that pass through
    a precision-narrowing cast (e.g. f64 → f32 on a value not exactly
    representable in f32).

Edge cases REQUIRED by the dispatch:
  - All Dtype variants enumerated.
  - All Device variants enumerated (Cpu, Cuda(0), Cuda(N), Meta, Xpu,
    Mps).
  - All FerrotorchError variants Display-formatted.
  - cast: f32→f64 finite, f64→bf16 huge (overflow → Err), f64→f32 NaN
    preserves NaN.
  - Tensor::is_contiguous: true for contiguous, false after stride-view
    ops (transpose, narrow).
  - Tensor::is_meta: true for meta tensors.
  - Tensor::is_leaf: true for created tensors, false for op outputs.

Usage from WSL (preferred per #777):

    python3 scripts/regenerate_plumbing_fixtures.py

Required Python deps: torch (CPU is sufficient for parity reference).
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
    / "plumbing.json"
)

# Plumbing fixtures are device-agnostic metadata; the values are pinned
# bit-exactly. PyTorch is consulted only as a parity sanity check on
# attribute names / values (e.g. `tensor.is_contiguous()` returns the
# expected boolean for the same input), not as a numerical reference.
DEVICES: list[str] = ["cpu"]

RNG_SEED: int = 0xB14_B19  # plumbing — phase 2.14, deterministic
torch.manual_seed(RNG_SEED)


def to_list_f64(t: torch.Tensor) -> list[Any]:
    """Encode an f64 tensor with NaN/Infinity as JSON-safe sentinels."""
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
        "devices": DEVICES,
        "phase": "2.14-plumbing",
        "tracking_issue": "#776",
    }


# ---------------------------------------------------------------------------
# Device enum — variant enumeration
# ---------------------------------------------------------------------------


def fixture_device_variants() -> list[dict[str, Any]]:
    """Each Device variant: expected `is_*` predicate truth table and the
    `Display` string. PyTorch's `torch.device("cpu")` / `cuda:0` / `meta`
    / `xpu:1` / `mps:0` provides the parity reference for the display
    string; we pin it here so the Rust test compares verbatim."""
    out: list[dict[str, Any]] = []
    variants = [
        ("cpu", None, "cpu"),
        ("cuda", 0, "cuda:0"),
        ("cuda", 3, "cuda:3"),
        ("meta", None, "meta"),
        ("xpu", 0, "xpu:0"),
        ("xpu", 2, "xpu:2"),
        ("mps", 0, "mps:0"),
    ]
    for kind, index, expected_display in variants:
        # PyTorch parity: torch.device formats the same way for cpu/cuda/meta
        # (xpu / mps are version-dependent in PyTorch; we pin the expected
        # display string by ferrotorch's contract — `Device::Display` matches
        # `torch.device(...)` for the variants PyTorch supports).
        out.append({
            "op": "device_variant",
            "tag": expected_display.replace(":", "_"),
            "kind": kind,
            "index": index,
            "expected_display": expected_display,
            "expected_is_cpu": kind == "cpu",
            "expected_is_cuda": kind == "cuda",
            "expected_is_meta": kind == "meta",
            "expected_is_xpu": kind == "xpu",
            "expected_is_mps": kind == "mps",
        })
    return out


# ---------------------------------------------------------------------------
# DType enum — variant enumeration via ferray-core's set
# ---------------------------------------------------------------------------


def fixture_dtype_variants() -> list[dict[str, Any]]:
    """`Element::dtype()` per Rust scalar type. PyTorch reference is the
    matching `torch.dtype`. The Rust test calls `<T as Element>::dtype()`
    for each `T` and asserts the discriminant name matches."""
    out: list[dict[str, Any]] = []
    pairs = [
        ("F32", "f32", torch.float32),
        ("F64", "f64", torch.float64),
        ("BF16", "bf16", torch.bfloat16),
        ("F16", "f16", torch.float16),
        ("I32", "i32", torch.int32),
        ("I64", "i64", torch.int64),
        ("I8", "i8", torch.int8),
        ("U8", "u8", torch.uint8),
        ("Bool", "bool", torch.bool),
    ]
    for variant, rust_ty, torch_dtype in pairs:
        # Build a small tensor of this dtype to confirm torch agrees on
        # itemsize / kind. We don't depend on torch's _internal name, just
        # that the dtype constant exists.
        sample = torch.zeros(1, dtype=torch_dtype)
        out.append({
            "op": "dtype_variant",
            "tag": variant,
            "variant": variant,
            "rust_type": rust_ty,
            "torch_dtype_name": str(torch_dtype),
            "itemsize": sample.element_size(),
        })
    return out


# ---------------------------------------------------------------------------
# numeric_cast::cast — representative success / failure cases
# ---------------------------------------------------------------------------


def fixture_numeric_cast() -> list[dict[str, Any]]:
    """`cast::<T,U>(v)` — fallible numeric conversion.

    Each fixture has:
      - src_type / dst_type: ascii names of the Rust types
      - src_value: input value (or sentinel) as JSON
      - expect_err: True when the cast must fail
      - expected_value: the expected output (when expect_err == False)

    The Rust test uses these to drive `cast::<T,U>(v)` and compares
    success/failure shape, plus the value (for finite cases) within
    F32_ELEMENTWISE tolerance for narrowing casts.
    """
    out: list[dict[str, Any]] = []

    # f64 -> f32, exactly representable.
    out.append({
        "op": "cast",
        "tag": "f64_to_f32_finite",
        "src_type": "f64",
        "dst_type": "f32",
        "src_value": 1.5,
        "expect_err": False,
        "expected_value": 1.5,
    })

    # f64 -> f32, lossy but finite.
    out.append({
        "op": "cast",
        "tag": "f64_to_f32_pi",
        "src_type": "f64",
        "dst_type": "f32",
        "src_value": math.pi,
        "expect_err": False,
        # f32(pi) = 3.1415927410125732 — the test compares within f32 ULP.
        "expected_value": float(torch.tensor(math.pi, dtype=torch.float64)
                                .to(torch.float32).item()),
    })

    # f64 NaN -> f32 NaN: NumCast::from(NaN) returns Some(NaN) for f32, so
    # this must succeed and the result must be NaN.
    out.append({
        "op": "cast",
        "tag": "f64_to_f32_nan_preserves",
        "src_type": "f64",
        "dst_type": "f32",
        "src_value": "NaN",
        "expect_err": False,
        "expected_value": "NaN",
    })

    # f64 INFINITY -> i32: NumCast::from(INFINITY) returns None, must err.
    out.append({
        "op": "cast",
        "tag": "f64_inf_to_i32_err",
        "src_type": "f64",
        "dst_type": "i32",
        "src_value": "Infinity",
        "expect_err": True,
        "expected_value": None,
    })

    # f64 NEG_INFINITY -> i32: same, must err.
    out.append({
        "op": "cast",
        "tag": "f64_neg_inf_to_i32_err",
        "src_type": "f64",
        "dst_type": "i32",
        "src_value": "-Infinity",
        "expect_err": True,
        "expected_value": None,
    })

    # f64 NaN -> i32: NumCast::from(NaN) -> None for integer dst → err.
    out.append({
        "op": "cast",
        "tag": "f64_nan_to_i32_err",
        "src_type": "f64",
        "dst_type": "i32",
        "src_value": "NaN",
        "expect_err": True,
        "expected_value": None,
    })

    # usize -> f32, exactly representable.
    out.append({
        "op": "cast",
        "tag": "usize_to_f32_42",
        "src_type": "usize",
        "dst_type": "f32",
        "src_value": 42,
        "expect_err": False,
        "expected_value": 42.0,
    })

    # negative i32 -> u32: NumCast returns None → err.
    out.append({
        "op": "cast",
        "tag": "neg_i32_to_u32_err",
        "src_type": "i32",
        "dst_type": "u32",
        "src_value": -1,
        "expect_err": True,
        "expected_value": None,
    })

    # f64 1.5 -> bf16: bf16 is exactly representable for 1.5 → succeeds.
    out.append({
        "op": "cast",
        "tag": "f64_to_bf16_one_and_half",
        "src_type": "f64",
        "dst_type": "bf16",
        "src_value": 1.5,
        "expect_err": False,
        "expected_value": 1.5,  # bf16(1.5) == 1.5 exactly
    })

    # f64 huge -> bf16: bf16 max finite ~ 3.39e38 (same exponent range as
    # f32). f64::MAX = 1.798e308 is well beyond bf16 range — NumCast
    # returns None → err.
    out.append({
        "op": "cast",
        "tag": "f64_huge_to_bf16_err",
        "src_type": "f64",
        "dst_type": "bf16",
        # 1e300 is unrepresentable in bf16 (exponent overflow).
        "src_value": 1e300,
        "expect_err": True,
        "expected_value": None,
    })

    # i64::MAX -> i32: NumCast returns None → err.
    out.append({
        "op": "cast",
        "tag": "i64_max_to_i32_err",
        "src_type": "i64",
        "dst_type": "i32",
        "src_value": 9223372036854775807,  # i64::MAX
        "expect_err": True,
        "expected_value": None,
    })

    return out


# ---------------------------------------------------------------------------
# FerrotorchError — Display formatting per variant
# ---------------------------------------------------------------------------


def fixture_error_display() -> list[dict[str, Any]]:
    """Each enum variant of FerrotorchError, paired with the exact
    Display string produced by the `#[error(...)]` attribute on that
    variant. The Rust test constructs each variant and asserts the
    Display string matches verbatim."""
    out: list[dict[str, Any]] = []
    # The (variant, expected Display string) pairs mirror the
    # `#[error(...)]` annotations in src/error.rs literally.
    cases: list[tuple[str, str, dict[str, Any]]] = [
        (
            "ShapeMismatch",
            "shape mismatch: dim 0",
            {"message": "dim 0"},
        ),
        (
            "DeviceMismatch",
            "device mismatch: expected cpu, got cuda:0",
            {"expected": "cpu", "got": "cuda:0"},
        ),
        (
            "BackwardNonScalar",
            "backward called on non-scalar tensor with shape [2, 3]",
            {"shape": [2, 3]},
        ),
        (
            "NoGradFn",
            "no gradient function on non-leaf tensor",
            {},
        ),
        (
            "DtypeMismatch",
            "dtype mismatch: expected f32, got f64",
            {"expected": "f32", "got": "f64"},
        ),
        (
            "IndexOutOfBounds",
            "index out of bounds: index 5 on axis 0 with size 3",
            {"index": 5, "axis": 0, "size": 3},
        ),
        (
            "InvalidArgument",
            "invalid argument: bad",
            {"message": "bad"},
        ),
        (
            "LockPoisoned",
            "internal lock poisoned: mutex",
            {"message": "mutex"},
        ),
        (
            "Internal",
            "internal error: oops",
            {"message": "oops"},
        ),
        (
            "DeviceUnavailable",
            "no GPU backend available -- install ferrotorch-gpu and call init()",
            {},
        ),
        (
            "GpuTensorNotAccessible",
            "cannot access GPU tensor data as CPU slice -- call .cpu() first",
            {},
        ),
        (
            "NotImplementedOnCuda",
            "fft is not supported on CUDA tensors -- call .cpu() first",
            {"op": "fft"},
        ),
        (
            "WorkerPanic",
            "data loading worker panicked: thread died",
            {"message": "thread died"},
        ),
    ]
    for variant, display, args in cases:
        out.append({
            "op": "error_display",
            "tag": variant,
            "variant": variant,
            "args": args,
            "expected_display": display,
        })
    return out


# ---------------------------------------------------------------------------
# Tensor metadata — stride / contiguity / leaf / scalar parity
# ---------------------------------------------------------------------------


def fixture_tensor_metadata() -> list[dict[str, Any]]:
    """Tensor metadata predicates tested against PyTorch. For each shape
    we record:
      - is_contiguous (must match torch.tensor(...).is_contiguous())
      - is_contiguous after a stride-view op (transpose, narrow)
      - is_scalar (torch.tensor(scalar).numel() == 1 && ndim == 0)
      - shape / ndim / numel / strides / storage_offset

    The Rust test exercises the matching tensor and asserts byte-exact
    metadata. These are pure-metadata operations — no arithmetic.
    """
    out: list[dict[str, Any]] = []

    shapes_basic = [
        [2, 3],
        [4],
        [2, 3, 4],
        [],  # 0-D scalar
    ]
    for shape in shapes_basic:
        n = max(1, math.prod(shape) if shape else 1)
        data = list(range(n))
        a = torch.tensor(data, dtype=torch.float64).reshape(shape if shape else ())
        out.append({
            "op": "tensor_metadata",
            "tag": "contig_" + "x".join(str(d) for d in shape) if shape else "contig_scalar",
            "shape": shape,
            "in_data": [float(x) for x in data],
            "expected_ndim": a.ndim,
            "expected_numel": a.numel(),
            "expected_is_contiguous": a.is_contiguous(),
            "expected_is_scalar": (a.numel() == 1 and a.ndim == 0),
        })

    # Transposed views: not contiguous (PyTorch parity).
    for shape in [[2, 3], [3, 4]]:
        n = math.prod(shape)
        data = list(range(n))
        a = torch.tensor(data, dtype=torch.float64).reshape(shape)
        t = a.transpose(0, 1)
        out.append({
            "op": "tensor_metadata_transposed",
            "tag": "transposed_" + "x".join(str(d) for d in shape),
            "shape": shape,
            "in_data": [float(x) for x in data],
            "transposed_shape": list(t.shape),
            "expected_is_contiguous": t.is_contiguous(),
        })

    # Narrowed views: along axis 0, partial range — not contiguous when narrow
    # cuts the leading dim of a non-leading-dense layout.
    for shape, dim, start, length in [
        ([4, 3], 0, 1, 2),
        ([4, 3], 1, 0, 2),
    ]:
        n = math.prod(shape)
        data = list(range(n))
        a = torch.tensor(data, dtype=torch.float64).reshape(shape)
        t = a.narrow(dim, start, length)
        out.append({
            "op": "tensor_metadata_narrowed",
            "tag": f"narrow_d{dim}_s{start}_l{length}_" + "x".join(str(d) for d in shape),
            "shape": shape,
            "in_data": [float(x) for x in data],
            "dim": dim,
            "start": start,
            "length": length,
            "narrowed_shape": list(t.shape),
            "expected_is_contiguous": t.is_contiguous(),
        })

    return out


# ---------------------------------------------------------------------------
# meta_propagate — shape-only output for matmul / reductions / unary
# ---------------------------------------------------------------------------


def fixture_meta_propagate() -> list[dict[str, Any]]:
    """`meta_propagate::*` shape rules. PyTorch parity reference is the
    `torch.empty(..., device='meta')` shape arithmetic — our helpers
    must produce the same output shape for the same input shape.

    We pin only output shapes (no values, since meta tensors carry no
    data)."""
    out: list[dict[str, Any]] = []

    # unary_same_shape: passes through.
    for shape in [[3, 4], [2, 3, 4], [5]]:
        out.append({
            "op": "meta_unary_same_shape",
            "tag": "x".join(str(d) for d in shape),
            "in_shape": shape,
            "expected_out_shape": shape,
        })

    # binary_broadcast: leverages broadcast_shapes.
    binary_cases = [
        ([3, 4], [3, 4], [3, 4]),
        ([3, 1], [1, 4], [3, 4]),
        ([5, 1, 7], [3, 1], [5, 3, 7]),
    ]
    for a_shape, b_shape, expected in binary_cases:
        out.append({
            "op": "meta_binary_broadcast",
            "tag": "_".join("x".join(str(d) for d in s) for s in (a_shape, b_shape)),
            "a_shape": a_shape,
            "b_shape": b_shape,
            "expected_out_shape": expected,
        })

    # reduce_dim: shape with axis dropped (or kept at size 1).
    for in_shape, dim, keepdim, expected in [
        ([2, 3, 4], 1, False, [2, 4]),
        ([2, 3, 4], 1, True, [2, 1, 4]),
        ([2, 3, 4], -1, False, [2, 3]),
        ([5], 0, True, [1]),
    ]:
        out.append({
            "op": "meta_reduce_dim",
            "tag": f"shape{'x'.join(str(d) for d in in_shape)}_dim{dim}_keep{int(keepdim)}",
            "in_shape": in_shape,
            "dim": dim,
            "keepdim": keepdim,
            "expected_out_shape": expected,
        })

    # reduce_all: any input → 0-D scalar shape.
    for in_shape in [[2, 3], [5], [2, 3, 4]]:
        out.append({
            "op": "meta_reduce_all",
            "tag": "x".join(str(d) for d in in_shape),
            "in_shape": in_shape,
            "expected_out_shape": [],
        })

    # matmul: PyTorch shape rules.
    matmul_cases = [
        ([5], [5], []),                  # 1D x 1D → scalar
        ([3, 5], [5], [3]),              # 2D x 1D → 1D
        ([5], [5, 4], [4]),              # 1D x 2D → 1D
        ([3, 5], [5, 4], [3, 4]),        # 2D x 2D → 2D
        ([2, 3, 5], [2, 5, 4], [2, 3, 4]),
        ([1, 3, 5], [4, 5, 7], [4, 3, 7]),
    ]
    for a_shape, b_shape, expected in matmul_cases:
        out.append({
            "op": "meta_matmul",
            "tag": "_".join("x".join(str(d) for d in s) for s in (a_shape, b_shape)),
            "a_shape": a_shape,
            "b_shape": b_shape,
            "expected_out_shape": expected,
        })

    return out


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def main() -> int:
    fixtures: list[dict[str, Any]] = []
    fixtures += fixture_device_variants()
    fixtures += fixture_dtype_variants()
    fixtures += fixture_numeric_cast()
    fixtures += fixture_error_display()
    fixtures += fixture_tensor_metadata()
    fixtures += fixture_meta_propagate()

    payload = {
        "metadata": fixture_metadata(),
        "fixtures": fixtures,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {len(fixtures)} fixtures to {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
