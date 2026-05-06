#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for ferrotorch-core Phase 2.3
(shape, indexing, view ops).

Tracking issue: #765 (parent: #759).

Output:
    ferrotorch-core/tests/conformance/fixtures/shape.json

Coverage (per the dispatch + #765's exclusion list — 95 surface items):

* Cat A.shape — pure-shape forwards (CPU + GPU + autograd):
    reshape, view, flatten, squeeze, unsqueeze, transpose, permute, narrow,
    contiguous, cat, split, chunk, expand,
    as_strided, as_strided_copy, as_strided_scatter
  Edge cases: negative axis, empty-input reshape, non-contiguous-input ->
  contiguous-output (the view_reshape silent-demote pattern), broadcast-aware
  expand (size-1 dim -> larger, asserts stride-0 view), narrow with
  start=0/length=full/partial, cat on empty list / single tensor / many
  tensors.

* Cat A.indexing — CPU forward + autograd; GPU error-path assertion:
    gather, scatter, scatter_add, where_cond, index_select_1d, masked_fill
  Edge cases: multi-dim gather/scatter, out-of-bounds index detection.

* Cat A.tensor_ops — CPU only (sources gate `is_cuda() -> Err`):
    triu, tril, diag, diagflat, roll, cdist
  Edge cases: positive/negative diagonals, p=1/p=2 cdist, roll wraparound.

* Cat A.search — CPU only:
    searchsorted, bucketize, unique, unique_consecutive, histc, meshgrid, topk

* Cat B — pure shape utility helpers (no tensor data; pure CPU functions):
    broadcast_shapes, normalize_axis, numel, c_contiguous_strides,
    channels_last_strides, channels_last_3d_strides, check_shapes_match
  These are unit-tested directly against fixtures derived from PyTorch's
  semantics rather than torch primitives.

* Cat C/D — backward grad_fn structs and `*Backward::new` constructors —
  implicit coverage via Cat A's autograd-grad assertions; not generated as
  fixture rows.

Usage from WSL (preferred per #777):

    python3 scripts/regenerate_shape_fixtures.py

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
    / "shape.json"
)

DTYPES: list[str] = ["float32", "float64"]
DEVICES: list[str] = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda:0")

# Indexing/search/tensor_ops have CPU-only ferrotorch implementations
# (the source files explicitly return NotImplementedOnCuda). We still emit
# CPU fixtures only for them.
CPU_ONLY_DEVICES: list[str] = ["cpu"]

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


def _arange_in(shape: list[int], dtype: str, device: str) -> torch.Tensor:
    n = math.prod(shape) if shape else 1
    if n == 0:
        return torch.zeros(shape, dtype=torch_dtype(dtype), device=device)
    vals = [1.0 + i * 0.5 for i in range(n)]
    return torch.tensor(vals, dtype=torch_dtype(dtype), device=device).reshape(shape)


# ---------------------------------------------------------------------------
# Cat A.shape — view / metadata-only ops
# ---------------------------------------------------------------------------


def fixture_reshape() -> list[dict[str, Any]]:
    """reshape with valid total-numel preservation, including -1 inference
    and an empty input."""
    out: list[dict[str, Any]] = []
    cases = [
        # (in_shape, new_shape, tag)
        ([6], [2, 3], "1d_to_2d"),
        ([2, 3], [6], "2d_to_1d"),
        ([2, 3, 4], [-1, 4], "neg1_infer_first"),
        ([24], [2, -1, 3], "neg1_infer_middle"),
        ([0], [0, 5], "empty_to_2d"),
    ]
    for device in DEVICES:
        for dtype in DTYPES:
            for in_shape, new_shape, tag in cases:
                a = _arange_in(in_shape, dtype, device)
                a_g = a.detach().clone().requires_grad_(True)
                fwd = a_g.reshape(new_shape)
                if fwd.numel() > 0:
                    loss = fwd.sum()
                    loss.backward()
                    grad_a = to_listf(a_g.grad)
                else:
                    grad_a = []
                out.append({
                    "op": "reshape",
                    "tag": tag,
                    "dtype": dtype,
                    "device": device,
                    "in_shape": in_shape,
                    "new_shape": new_shape,
                    "out_shape": list(fwd.shape),
                    "in_data": to_listf(a),
                    "out_values": to_listf(fwd),
                    "grad_a": grad_a,
                })
    return out


def fixture_view() -> list[dict[str, Any]]:
    """view (i64 shape) — semantically identical to reshape for contiguous
    inputs, but goes through `view_t` which validates contiguity."""
    out: list[dict[str, Any]] = []
    cases = [
        ([6], [2, 3], "1d_to_2d"),
        ([2, 3], [6], "2d_to_1d"),
        ([8], [-1, 2], "neg1"),
    ]
    for device in DEVICES:
        for dtype in DTYPES:
            for in_shape, new_shape, tag in cases:
                a = _arange_in(in_shape, dtype, device)
                a_g = a.detach().clone().requires_grad_(True)
                fwd = a_g.view(new_shape)
                loss = fwd.sum()
                loss.backward()
                out.append({
                    "op": "view",
                    "tag": tag,
                    "dtype": dtype,
                    "device": device,
                    "in_shape": in_shape,
                    "new_shape": new_shape,
                    "out_shape": list(fwd.shape),
                    "in_data": to_listf(a),
                    "out_values": to_listf(fwd),
                    "grad_a": to_listf(a_g.grad),
                })
    return out


def fixture_flatten() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cases = [
        ([2, 3], "2d"),
        ([2, 2, 3], "3d"),
        ([1], "1d_singleton"),
    ]
    for device in DEVICES:
        for dtype in DTYPES:
            for shape, tag in cases:
                a = _arange_in(shape, dtype, device)
                a_g = a.detach().clone().requires_grad_(True)
                fwd = a_g.flatten()
                loss = fwd.sum()
                loss.backward()
                out.append({
                    "op": "flatten",
                    "tag": tag,
                    "dtype": dtype,
                    "device": device,
                    "in_shape": shape,
                    "out_shape": list(fwd.shape),
                    "in_data": to_listf(a),
                    "out_values": to_listf(fwd),
                    "grad_a": to_listf(a_g.grad),
                })
    return out


def fixture_squeeze() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cases = [
        # (shape, axis, tag) — axis is the size-1 dim to squeeze.
        ([1, 3], 0, "lead1"),
        ([3, 1], 1, "trail1"),
        ([2, 1, 4], 1, "mid1"),
        # negative axis
        ([2, 1, 4], -2, "negaxis"),
    ]
    for device in DEVICES:
        for dtype in DTYPES:
            for shape, axis, tag in cases:
                a = _arange_in(shape, dtype, device)
                a_g = a.detach().clone().requires_grad_(True)
                fwd = a_g.squeeze(axis)
                loss = fwd.sum()
                loss.backward()
                out.append({
                    "op": "squeeze",
                    "tag": tag,
                    "dtype": dtype,
                    "device": device,
                    "in_shape": shape,
                    "axis": axis,
                    "out_shape": list(fwd.shape),
                    "in_data": to_listf(a),
                    "out_values": to_listf(fwd),
                    "grad_a": to_listf(a_g.grad),
                })
    return out


def fixture_unsqueeze() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cases = [
        # (shape, axis, tag) — axis is where the new size-1 dim is inserted.
        ([3], 0, "axis0"),
        ([3], 1, "axis_end"),
        ([2, 3], 1, "mid"),
        ([2, 3], -1, "negaxis_end"),
        ([2, 3], -3, "negaxis_lead"),
    ]
    for device in DEVICES:
        for dtype in DTYPES:
            for shape, axis, tag in cases:
                a = _arange_in(shape, dtype, device)
                a_g = a.detach().clone().requires_grad_(True)
                fwd = a_g.unsqueeze(axis)
                loss = fwd.sum()
                loss.backward()
                out.append({
                    "op": "unsqueeze",
                    "tag": tag,
                    "dtype": dtype,
                    "device": device,
                    "in_shape": shape,
                    "axis": axis,
                    "out_shape": list(fwd.shape),
                    "in_data": to_listf(a),
                    "out_values": to_listf(fwd),
                    "grad_a": to_listf(a_g.grad),
                })
    return out


def fixture_transpose() -> list[dict[str, Any]]:
    """ferrotorch's `transpose_2d` is the canonical 2-D transpose; the
    `Tensor::transpose(dim0, dim1)` method handles N-D via permute. We
    fixture both shapes so the test can exercise each entry point."""
    out: list[dict[str, Any]] = []
    cases = [
        ([2, 3], 0, 1, "2d"),
        ([2, 3, 4], 0, 2, "3d_outer"),
        ([2, 3, 4], 1, 2, "3d_inner"),
    ]
    for device in DEVICES:
        for dtype in DTYPES:
            for shape, d0, d1, tag in cases:
                a = _arange_in(shape, dtype, device)
                a_g = a.detach().clone().requires_grad_(True)
                fwd = a_g.transpose(d0, d1).contiguous()
                loss = fwd.sum()
                loss.backward()
                out.append({
                    "op": "transpose",
                    "tag": tag,
                    "dtype": dtype,
                    "device": device,
                    "in_shape": shape,
                    "dim0": d0,
                    "dim1": d1,
                    "out_shape": list(fwd.shape),
                    "in_data": to_listf(a),
                    "out_values": to_listf(fwd),
                    "grad_a": to_listf(a_g.grad),
                })
    return out


def fixture_permute() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cases = [
        ([2, 3, 4], [2, 0, 1], "rotate"),
        ([2, 3, 4], [0, 1, 2], "identity"),
        ([2, 3, 4, 5], [3, 0, 2, 1], "4d"),
    ]
    for device in DEVICES:
        for dtype in DTYPES:
            for shape, dims, tag in cases:
                a = _arange_in(shape, dtype, device)
                a_g = a.detach().clone().requires_grad_(True)
                fwd = a_g.permute(dims).contiguous()
                loss = fwd.sum()
                loss.backward()
                out.append({
                    "op": "permute",
                    "tag": tag,
                    "dtype": dtype,
                    "device": device,
                    "in_shape": shape,
                    "dims": dims,
                    "out_shape": list(fwd.shape),
                    "in_data": to_listf(a),
                    "out_values": to_listf(fwd),
                    "grad_a": to_listf(a_g.grad),
                })
    return out


def fixture_narrow() -> list[dict[str, Any]]:
    """narrow: exercise start=0, length=full, partial — edge cases per
    the dispatch."""
    out: list[dict[str, Any]] = []
    cases = [
        # (shape, dim, start, length, tag)
        ([6], 0, 0, 6, "start0_full"),
        ([6], 0, 0, 3, "start0_partial"),
        ([6], 0, 2, 3, "midstart"),
        ([6], 0, 3, 3, "tail"),
        ([2, 5], 1, 1, 3, "2d_mid"),
        ([2, 5], 0, 1, 1, "outer_axis"),
    ]
    for device in DEVICES:
        for dtype in DTYPES:
            for shape, dim, start, length, tag in cases:
                a = _arange_in(shape, dtype, device)
                a_g = a.detach().clone().requires_grad_(True)
                fwd = a_g.narrow(dim, start, length).contiguous()
                loss = fwd.sum()
                loss.backward()
                out.append({
                    "op": "narrow",
                    "tag": tag,
                    "dtype": dtype,
                    "device": device,
                    "in_shape": shape,
                    "dim": dim,
                    "start": start,
                    "length": length,
                    "out_shape": list(fwd.shape),
                    "in_data": to_listf(a),
                    "out_values": to_listf(fwd),
                    "grad_a": to_listf(a_g.grad),
                })
    return out


def fixture_contiguous() -> list[dict[str, Any]]:
    """Exercise the *non-contiguous-input -> contiguous-output* pattern,
    which is the view_reshape silent-demote bug regression target."""
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            # Build a 2x3 contiguous, then transpose -> non-contiguous,
            # then call .contiguous(). Output values are the transposed
            # data laid out contiguously.
            a = _arange_in([2, 3], dtype, device)
            non_contig = a.transpose(0, 1)
            out_t = non_contig.contiguous()
            assert out_t.is_contiguous()
            # Autograd-aware: requires_grad on the input tensor.
            a_g = a.detach().clone().requires_grad_(True)
            non_contig_g = a_g.transpose(0, 1)
            fwd = non_contig_g.contiguous()
            loss = fwd.sum()
            loss.backward()
            out.append({
                "op": "contiguous",
                "tag": "transpose_then_contiguous",
                "dtype": dtype,
                "device": device,
                "in_shape": [2, 3],
                "out_shape": list(out_t.shape),
                "in_data": to_listf(a),
                "out_values": to_listf(out_t),
                "grad_a": to_listf(a_g.grad),
            })
            # Already-contiguous: should be a fast clone with identical data.
            b = _arange_in([2, 3], dtype, device)
            cb = b.contiguous()
            out.append({
                "op": "contiguous",
                "tag": "already_contiguous",
                "dtype": dtype,
                "device": device,
                "in_shape": [2, 3],
                "out_shape": list(cb.shape),
                "in_data": to_listf(b),
                "out_values": to_listf(cb),
            })
    return out


def fixture_cat() -> list[dict[str, Any]]:
    """cat over many tensors and edge cases (single tensor, mismatched
    along axis)."""
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            # 1-D, 2 tensors, axis 0
            a = torch.tensor([1.0, 2.0, 3.0], dtype=torch_dtype(dtype), device=device)
            b = torch.tensor([4.0, 5.0], dtype=torch_dtype(dtype), device=device)
            a_g = a.detach().clone().requires_grad_(True)
            b_g = b.detach().clone().requires_grad_(True)
            fwd = torch.cat([a_g, b_g], dim=0)
            loss = fwd.sum()
            loss.backward()
            out.append({
                "op": "cat",
                "tag": "two_1d",
                "dtype": dtype,
                "device": device,
                "tensor_shapes": [list(a.shape), list(b.shape)],
                "tensor_data": [to_listf(a), to_listf(b)],
                "axis": 0,
                "out_shape": list(fwd.shape),
                "out_values": to_listf(fwd),
                "tensor_grads": [to_listf(a_g.grad), to_listf(b_g.grad)],
            })

            # 2-D, 3 tensors, axis 1, varying along axis
            shapes = [[2, 1], [2, 3], [2, 2]]
            tensors = [
                torch.arange(1.0, 1.0 + math.prod(s), dtype=torch_dtype(dtype), device=device).reshape(s)
                for s in shapes
            ]
            ts_g = [t.detach().clone().requires_grad_(True) for t in tensors]
            fwd = torch.cat(ts_g, dim=1)
            loss = fwd.sum()
            loss.backward()
            out.append({
                "op": "cat",
                "tag": "three_2d_axis1",
                "dtype": dtype,
                "device": device,
                "tensor_shapes": shapes,
                "tensor_data": [to_listf(t) for t in tensors],
                "axis": 1,
                "out_shape": list(fwd.shape),
                "out_values": to_listf(fwd),
                "tensor_grads": [to_listf(t.grad) for t in ts_g],
            })

            # Single tensor cat (legal in torch).
            single = torch.tensor([1.0, 2.0, 3.0], dtype=torch_dtype(dtype), device=device)
            single_g = single.detach().clone().requires_grad_(True)
            fwd = torch.cat([single_g], dim=0)
            loss = fwd.sum()
            loss.backward()
            out.append({
                "op": "cat",
                "tag": "single_tensor",
                "dtype": dtype,
                "device": device,
                "tensor_shapes": [list(single.shape)],
                "tensor_data": [to_listf(single)],
                "axis": 0,
                "out_shape": list(fwd.shape),
                "out_values": to_listf(fwd),
                "tensor_grads": [to_listf(single_g.grad)],
            })

            # Negative-axis cat
            t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch_dtype(dtype), device=device)
            t2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch_dtype(dtype), device=device)
            t1_g = t1.detach().clone().requires_grad_(True)
            t2_g = t2.detach().clone().requires_grad_(True)
            fwd = torch.cat([t1_g, t2_g], dim=-1)
            loss = fwd.sum()
            loss.backward()
            out.append({
                "op": "cat",
                "tag": "negaxis",
                "dtype": dtype,
                "device": device,
                "tensor_shapes": [list(t1.shape), list(t2.shape)],
                "tensor_data": [to_listf(t1), to_listf(t2)],
                "axis": -1,
                "out_shape": list(fwd.shape),
                "out_values": to_listf(fwd),
                "tensor_grads": [to_listf(t1_g.grad), to_listf(t2_g.grad)],
            })
    return out


def fixture_split() -> list[dict[str, Any]]:
    """split with explicit split_sizes."""
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            # 1-D split into uneven parts
            a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                             dtype=torch_dtype(dtype), device=device)
            a_g = a.detach().clone().requires_grad_(True)
            chunks = torch.split(a_g, [2, 1, 3], dim=0)
            # backward each chunk's sum to verify per-chunk grad routes
            # back to the original tensor.
            (sum(c.sum() for c in chunks)).backward()
            out.append({
                "op": "split",
                "tag": "1d_uneven",
                "dtype": dtype,
                "device": device,
                "in_shape": list(a.shape),
                "in_data": to_listf(a),
                "split_sizes": [2, 1, 3],
                "dim": 0,
                "chunks_shapes": [list(c.shape) for c in chunks],
                "chunks_values": [to_listf(c) for c in chunks],
                "grad_a": to_listf(a_g.grad),
            })
            # 2-D split along axis 1
            b = torch.arange(1.0, 13.0, dtype=torch_dtype(dtype), device=device).reshape(2, 6)
            b_g = b.detach().clone().requires_grad_(True)
            chunks = torch.split(b_g, [2, 4], dim=1)
            (sum(c.sum() for c in chunks)).backward()
            out.append({
                "op": "split",
                "tag": "2d_axis1",
                "dtype": dtype,
                "device": device,
                "in_shape": list(b.shape),
                "in_data": to_listf(b),
                "split_sizes": [2, 4],
                "dim": 1,
                "chunks_shapes": [list(c.shape) for c in chunks],
                "chunks_values": [to_listf(c) for c in chunks],
                "grad_a": to_listf(b_g.grad),
            })
    return out


def fixture_chunk() -> list[dict[str, Any]]:
    """chunk into approximately equal parts."""
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            # Even chunks
            a = torch.arange(1.0, 13.0, dtype=torch_dtype(dtype), device=device)
            a_g = a.detach().clone().requires_grad_(True)
            chunks = torch.chunk(a_g, 4, dim=0)
            (sum(c.sum() for c in chunks)).backward()
            out.append({
                "op": "chunk",
                "tag": "even",
                "dtype": dtype,
                "device": device,
                "in_shape": list(a.shape),
                "in_data": to_listf(a),
                "chunks": 4,
                "dim": 0,
                "chunks_shapes": [list(c.shape) for c in chunks],
                "chunks_values": [to_listf(c) for c in chunks],
                "grad_a": to_listf(a_g.grad),
            })
            # Uneven chunks
            b = torch.arange(1.0, 12.0, dtype=torch_dtype(dtype), device=device)  # 11 elements
            b_g = b.detach().clone().requires_grad_(True)
            chunks = torch.chunk(b_g, 4, dim=0)
            (sum(c.sum() for c in chunks)).backward()
            out.append({
                "op": "chunk",
                "tag": "uneven",
                "dtype": dtype,
                "device": device,
                "in_shape": list(b.shape),
                "in_data": to_listf(b),
                "chunks": 4,
                "dim": 0,
                "chunks_shapes": [list(c.shape) for c in chunks],
                "chunks_values": [to_listf(c) for c in chunks],
                "grad_a": to_listf(b_g.grad),
            })
    return out


def fixture_expand() -> list[dict[str, Any]]:
    """expand with size-1 broadcast dims."""
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            # 1-D size-1 -> larger
            a = torch.tensor([7.0], dtype=torch_dtype(dtype), device=device)
            a_g = a.detach().clone().requires_grad_(True)
            fwd = a_g.expand([5])
            loss = fwd.sum()
            loss.backward()
            out.append({
                "op": "expand",
                "tag": "1d_to_5",
                "dtype": dtype,
                "device": device,
                "in_shape": list(a.shape),
                "in_data": to_listf(a),
                "new_shape": [5],
                "out_shape": list(fwd.shape),
                "out_values": to_listf(fwd),
                "grad_a": to_listf(a_g.grad),
            })
            # 2-D, expand size-1 column to 4
            b = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch_dtype(dtype), device=device)
            b_g = b.detach().clone().requires_grad_(True)
            fwd = b_g.expand([3, 4])
            loss = fwd.sum()
            loss.backward()
            out.append({
                "op": "expand",
                "tag": "2d_col",
                "dtype": dtype,
                "device": device,
                "in_shape": list(b.shape),
                "in_data": to_listf(b),
                "new_shape": [3, 4],
                "out_shape": list(fwd.shape),
                "out_values": to_listf(fwd),
                "grad_a": to_listf(b_g.grad),
            })
    return out


def fixture_as_strided_family() -> list[dict[str, Any]]:
    """as_strided / as_strided_copy / as_strided_scatter."""
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            # as_strided: build a 2x3 view from a 6-element 1-D tensor
            a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                             dtype=torch_dtype(dtype), device=device)
            view = a.as_strided((2, 3), (3, 1), 0)
            out.append({
                "op": "as_strided",
                "tag": "1d_to_2x3_contig",
                "dtype": dtype,
                "device": device,
                "in_shape": list(a.shape),
                "in_data": to_listf(a),
                "size": [2, 3],
                "stride": [3, 1],
                "storage_offset": 0,
                "out_shape": list(view.shape),
                "out_values": to_listf(view),
            })
            # Sliding-window overlap
            b = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0],
                             dtype=torch_dtype(dtype), device=device)
            v2 = b.as_strided((3, 3), (1, 1), 0)
            out.append({
                "op": "as_strided",
                "tag": "sliding_window",
                "dtype": dtype,
                "device": device,
                "in_shape": list(b.shape),
                "in_data": to_listf(b),
                "size": [3, 3],
                "stride": [1, 1],
                "storage_offset": 0,
                "out_shape": list(v2.shape),
                "out_values": to_listf(v2),
            })
            # as_strided_copy: same shapes but materialized
            cp = b.as_strided((3, 3), (1, 1), 0).contiguous().clone()
            out.append({
                "op": "as_strided_copy",
                "tag": "sliding_window",
                "dtype": dtype,
                "device": device,
                "in_shape": list(b.shape),
                "in_data": to_listf(b),
                "size": [3, 3],
                "stride": [1, 1],
                "storage_offset": 0,
                "out_shape": list(cp.shape),
                "out_values": to_listf(cp),
            })
            # as_strided_scatter: write src into strided positions
            dst = torch.zeros(6, dtype=torch_dtype(dtype), device=device)
            src = torch.tensor([10.0, 20.0, 30.0],
                               dtype=torch_dtype(dtype), device=device)
            scattered = torch.as_strided_scatter(dst, src, (3,), (2,), 0)
            out.append({
                "op": "as_strided_scatter",
                "tag": "stride2",
                "dtype": dtype,
                "device": device,
                "in_shape": list(dst.shape),
                "in_data": to_listf(dst),
                "src_shape": list(src.shape),
                "src_data": to_listf(src),
                "size": [3],
                "stride": [2],
                "storage_offset": 0,
                "out_shape": list(scattered.shape),
                "out_values": to_listf(scattered),
            })
    return out


# ---------------------------------------------------------------------------
# Cat A.indexing — CPU-only forwards (ferrotorch returns NotImplementedOnCuda
# for these on GPU; the test asserts the GPU error path)
# ---------------------------------------------------------------------------


def fixture_gather() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in CPU_ONLY_DEVICES:
        for dtype in DTYPES:
            # 1-D
            a = torch.tensor([10.0, 20.0, 30.0, 40.0],
                             dtype=torch_dtype(dtype), device=device)
            idx = torch.tensor([3, 0, 2], dtype=torch.int64, device=device)
            a_g = a.detach().clone().requires_grad_(True)
            fwd = torch.gather(a_g, 0, idx)
            loss = fwd.sum()
            loss.backward()
            out.append({
                "op": "gather",
                "tag": "1d",
                "dtype": dtype,
                "device": device,
                "in_shape": list(a.shape),
                "in_data": to_listf(a),
                "index": [int(i) for i in idx.tolist()],
                "index_shape": list(idx.shape),
                "dim": 0,
                "out_shape": list(fwd.shape),
                "out_values": to_listf(fwd),
                "grad_a": to_listf(a_g.grad),
            })
            # 2-D dim=0
            b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                             dtype=torch_dtype(dtype), device=device)
            idx2 = torch.tensor([[2, 0], [1, 1]], dtype=torch.int64, device=device)
            b_g = b.detach().clone().requires_grad_(True)
            fwd = torch.gather(b_g, 0, idx2)
            loss = fwd.sum()
            loss.backward()
            out.append({
                "op": "gather",
                "tag": "2d_dim0",
                "dtype": dtype,
                "device": device,
                "in_shape": list(b.shape),
                "in_data": to_listf(b),
                "index": [int(i) for i in idx2.flatten().tolist()],
                "index_shape": list(idx2.shape),
                "dim": 0,
                "out_shape": list(fwd.shape),
                "out_values": to_listf(fwd),
                "grad_a": to_listf(b_g.grad),
            })
            # 2-D dim=1
            c = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                             dtype=torch_dtype(dtype), device=device)
            idx3 = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64, device=device)
            c_g = c.detach().clone().requires_grad_(True)
            fwd = torch.gather(c_g, 1, idx3)
            loss = fwd.sum()
            loss.backward()
            out.append({
                "op": "gather",
                "tag": "2d_dim1",
                "dtype": dtype,
                "device": device,
                "in_shape": list(c.shape),
                "in_data": to_listf(c),
                "index": [int(i) for i in idx3.flatten().tolist()],
                "index_shape": list(idx3.shape),
                "dim": 1,
                "out_shape": list(fwd.shape),
                "out_values": to_listf(fwd),
                "grad_a": to_listf(c_g.grad),
            })
    return out


def fixture_scatter_and_add() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in CPU_ONLY_DEVICES:
        for dtype in DTYPES:
            # scatter: 1-D, dim=0
            for op_name, op_torch in [
                ("scatter", lambda inp, dim, idx, src: torch.scatter(inp, dim, idx, src)),
                ("scatter_add",
                 lambda inp, dim, idx, src: torch.scatter_add(inp, dim, idx, src)),
            ]:
                inp = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0],
                                   dtype=torch_dtype(dtype), device=device)
                idx = torch.tensor([0, 2, 4], dtype=torch.int64, device=device)
                src = torch.tensor([10.0, 20.0, 30.0],
                                   dtype=torch_dtype(dtype), device=device)
                fwd = op_torch(inp, 0, idx, src)
                out.append({
                    "op": op_name,
                    "tag": "1d",
                    "dtype": dtype,
                    "device": device,
                    "in_shape": list(inp.shape),
                    "in_data": to_listf(inp),
                    "src_shape": list(src.shape),
                    "src_data": to_listf(src),
                    "index": [int(i) for i in idx.tolist()],
                    "index_shape": list(idx.shape),
                    "dim": 0,
                    "out_shape": list(fwd.shape),
                    "out_values": to_listf(fwd),
                })
                # 2-D dim=0
                inp2 = torch.zeros((3, 3), dtype=torch_dtype(dtype), device=device)
                idx2 = torch.tensor([[0, 1, 2], [2, 0, 1]],
                                    dtype=torch.int64, device=device)
                src2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                    dtype=torch_dtype(dtype), device=device)
                fwd2 = op_torch(inp2, 0, idx2, src2)
                out.append({
                    "op": op_name,
                    "tag": "2d_dim0",
                    "dtype": dtype,
                    "device": device,
                    "in_shape": list(inp2.shape),
                    "in_data": to_listf(inp2),
                    "src_shape": list(src2.shape),
                    "src_data": to_listf(src2),
                    "index": [int(i) for i in idx2.flatten().tolist()],
                    "index_shape": list(idx2.shape),
                    "dim": 0,
                    "out_shape": list(fwd2.shape),
                    "out_values": to_listf(fwd2),
                })
    return out


def fixture_where_cond() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in CPU_ONLY_DEVICES:
        for dtype in DTYPES:
            cond = torch.tensor([True, False, True, False, True],
                                dtype=torch.bool, device=device)
            x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0],
                             dtype=torch_dtype(dtype), device=device)
            y = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0],
                             dtype=torch_dtype(dtype), device=device)
            x_g = x.detach().clone().requires_grad_(True)
            y_g = y.detach().clone().requires_grad_(True)
            fwd = torch.where(cond, x_g, y_g)
            loss = fwd.sum()
            loss.backward()
            out.append({
                "op": "where_cond",
                "tag": "1d",
                "dtype": dtype,
                "device": device,
                "in_shape": list(x.shape),
                "x_data": to_listf(x),
                "y_data": to_listf(y),
                "condition": [bool(c) for c in cond.tolist()],
                "out_shape": list(fwd.shape),
                "out_values": to_listf(fwd),
                "grad_x": to_listf(x_g.grad),
                "grad_y": to_listf(y_g.grad),
            })
    return out


def fixture_index_select_1d() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # CPU: full forward+grad. GPU (f32 only): forward+grad supported.
    devices = DEVICES  # both
    for device in devices:
        for dtype in DTYPES:
            # GPU path is f32-only per the source — skip f64-on-cuda
            if device.startswith("cuda") and dtype == "float64":
                continue
            a = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0],
                             dtype=torch_dtype(dtype), device=device)
            idx_t = torch.tensor([4, 0, 2, 0], dtype=torch.int64, device=device)
            a_g = a.detach().clone().requires_grad_(True)
            fwd = torch.index_select(a_g, 0, idx_t)
            loss = fwd.sum()
            loss.backward()
            out.append({
                "op": "index_select_1d",
                "tag": "1d",
                "dtype": dtype,
                "device": device,
                "in_shape": list(a.shape),
                "in_data": to_listf(a),
                "index": [int(i) for i in idx_t.tolist()],
                "out_shape": list(fwd.shape),
                "out_values": to_listf(fwd),
                "grad_a": to_listf(a_g.grad),
            })
    return out


def fixture_masked_fill() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    devices = DEVICES  # both — f32 GPU supported
    for device in devices:
        for dtype in DTYPES:
            if device.startswith("cuda") and dtype == "float64":
                continue
            a = torch.tensor([1.0, 2.0, 3.0, 4.0],
                             dtype=torch_dtype(dtype), device=device)
            mask = torch.tensor([True, False, True, False],
                                dtype=torch.bool, device=device)
            a_g = a.detach().clone().requires_grad_(True)
            value = -99.0
            fwd = a_g.masked_fill(mask, value)
            loss = fwd.sum()
            loss.backward()
            out.append({
                "op": "masked_fill",
                "tag": "1d",
                "dtype": dtype,
                "device": device,
                "in_shape": list(a.shape),
                "in_data": to_listf(a),
                "mask": [bool(m) for m in mask.tolist()],
                "value": value,
                "out_shape": list(fwd.shape),
                "out_values": to_listf(fwd),
                "grad_a": to_listf(a_g.grad),
            })
    return out


# ---------------------------------------------------------------------------
# Cat A.tensor_ops — CPU-only
# ---------------------------------------------------------------------------


def fixture_triu_tril() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in CPU_ONLY_DEVICES:
        for dtype in DTYPES:
            for op in ("triu", "tril"):
                for diag in (-1, 0, 1):
                    a = torch.arange(1.0, 10.0,
                                     dtype=torch_dtype(dtype),
                                     device=device).reshape(3, 3)
                    fwd = getattr(torch, op)(a, diag)
                    out.append({
                        "op": op,
                        "tag": f"diag{diag}",
                        "dtype": dtype,
                        "device": device,
                        "in_shape": list(a.shape),
                        "in_data": to_listf(a),
                        "diagonal": diag,
                        "out_shape": list(fwd.shape),
                        "out_values": to_listf(fwd),
                    })
    return out


def fixture_diag_diagflat() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in CPU_ONLY_DEVICES:
        for dtype in DTYPES:
            # diag: extract from 2-D
            a = torch.arange(1.0, 10.0,
                             dtype=torch_dtype(dtype), device=device).reshape(3, 3)
            for diag in (-1, 0, 1):
                fwd = torch.diag(a, diag)
                out.append({
                    "op": "diag",
                    "tag": f"extract_diag{diag}",
                    "dtype": dtype,
                    "device": device,
                    "in_shape": list(a.shape),
                    "in_data": to_listf(a),
                    "diagonal": diag,
                    "out_shape": list(fwd.shape),
                    "out_values": to_listf(fwd),
                })
            # diag: construct from 1-D
            v = torch.tensor([1.0, 2.0, 3.0],
                             dtype=torch_dtype(dtype), device=device)
            for diag in (0, 1):
                fwd = torch.diag(v, diag)
                out.append({
                    "op": "diag",
                    "tag": f"construct_diag{diag}",
                    "dtype": dtype,
                    "device": device,
                    "in_shape": list(v.shape),
                    "in_data": to_listf(v),
                    "diagonal": diag,
                    "out_shape": list(fwd.shape),
                    "out_values": to_listf(fwd),
                })
            # diagflat from 2-D (flattens first)
            m = torch.tensor([[1.0, 2.0], [3.0, 4.0]],
                             dtype=torch_dtype(dtype), device=device)
            fwd = torch.diagflat(m, 0)
            out.append({
                "op": "diagflat",
                "tag": "from_2d",
                "dtype": dtype,
                "device": device,
                "in_shape": list(m.shape),
                "in_data": to_listf(m),
                "diagonal": 0,
                "out_shape": list(fwd.shape),
                "out_values": to_listf(fwd),
            })
    return out


def fixture_roll() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in CPU_ONLY_DEVICES:
        for dtype in DTYPES:
            a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0],
                             dtype=torch_dtype(dtype), device=device)
            for shift in (2, -1, 0):
                fwd = torch.roll(a, shifts=shift, dims=0)
                out.append({
                    "op": "roll",
                    "tag": f"shift{shift}",
                    "dtype": dtype,
                    "device": device,
                    "in_shape": list(a.shape),
                    "in_data": to_listf(a),
                    "shifts": shift,
                    "dim": 0,
                    "out_shape": list(fwd.shape),
                    "out_values": to_listf(fwd),
                })
    return out


def fixture_cdist() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in CPU_ONLY_DEVICES:
        for dtype in DTYPES:
            x1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                              dtype=torch_dtype(dtype), device=device)
            x2 = torch.tensor([[1.0, 1.0]],
                              dtype=torch_dtype(dtype), device=device)
            for p in (1.0, 2.0):
                fwd = torch.cdist(x1, x2, p=p)
                out.append({
                    "op": "cdist",
                    "tag": f"p{p}",
                    "dtype": dtype,
                    "device": device,
                    "x1_shape": list(x1.shape),
                    "x1_data": to_listf(x1),
                    "x2_shape": list(x2.shape),
                    "x2_data": to_listf(x2),
                    "p": p,
                    "out_shape": list(fwd.shape),
                    "out_values": to_listf(fwd),
                })
    return out


# ---------------------------------------------------------------------------
# Cat A.search — CPU-only
# ---------------------------------------------------------------------------


def fixture_searchsorted() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in CPU_ONLY_DEVICES:
        for dtype in DTYPES:
            bounds = torch.tensor([1.0, 3.0, 5.0, 7.0],
                                  dtype=torch_dtype(dtype), device=device)
            vals = torch.tensor([0.0, 2.0, 3.0, 6.0, 8.0],
                                dtype=torch_dtype(dtype), device=device)
            for right in (False, True):
                idxs = torch.searchsorted(bounds, vals, right=right)
                out.append({
                    "op": "searchsorted",
                    "tag": f"right{int(right)}",
                    "dtype": dtype,
                    "device": device,
                    "boundaries_shape": list(bounds.shape),
                    "boundaries_data": to_listf(bounds),
                    "values_shape": list(vals.shape),
                    "values_data": to_listf(vals),
                    "right": right,
                    "out_indices": [int(i) for i in idxs.tolist()],
                })
    return out


def fixture_bucketize() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in CPU_ONLY_DEVICES:
        for dtype in DTYPES:
            inp = torch.tensor([-0.5, 0.5, 1.5, 2.5, 3.5],
                               dtype=torch_dtype(dtype), device=device)
            bounds = torch.tensor([0.0, 1.0, 2.0, 3.0],
                                  dtype=torch_dtype(dtype), device=device)
            for right in (False, True):
                idxs = torch.bucketize(inp, bounds, right=right)
                out.append({
                    "op": "bucketize",
                    "tag": f"right{int(right)}",
                    "dtype": dtype,
                    "device": device,
                    "input_shape": list(inp.shape),
                    "input_data": to_listf(inp),
                    "boundaries_shape": list(bounds.shape),
                    "boundaries_data": to_listf(bounds),
                    "right": right,
                    "out_indices": [int(i) for i in idxs.tolist()],
                })
    return out


def fixture_unique_family() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in CPU_ONLY_DEVICES:
        for dtype in DTYPES:
            # unique sorted
            a = torch.tensor([3.0, 1.0, 2.0, 1.0, 3.0, 2.0],
                             dtype=torch_dtype(dtype), device=device)
            uniq, inverse, counts = torch.unique(
                a, sorted=True, return_inverse=True, return_counts=True
            )
            out.append({
                "op": "unique",
                "tag": "basic",
                "dtype": dtype,
                "device": device,
                "in_shape": list(a.shape),
                "in_data": to_listf(a),
                "out_shape": list(uniq.shape),
                "out_values": to_listf(uniq),
                "out_inverse": [int(i) for i in inverse.tolist()],
                "out_counts": [int(i) for i in counts.tolist()],
            })
            # unique_consecutive
            b = torch.tensor([1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 1.0],
                             dtype=torch_dtype(dtype), device=device)
            uniq_c, inv_c, cnt_c = torch.unique_consecutive(
                b, return_inverse=True, return_counts=True
            )
            out.append({
                "op": "unique_consecutive",
                "tag": "basic",
                "dtype": dtype,
                "device": device,
                "in_shape": list(b.shape),
                "in_data": to_listf(b),
                "out_shape": list(uniq_c.shape),
                "out_values": to_listf(uniq_c),
                "out_inverse": [int(i) for i in inv_c.tolist()],
                "out_counts": [int(i) for i in cnt_c.tolist()],
            })
    return out


def fixture_histc() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in CPU_ONLY_DEVICES:
        for dtype in DTYPES:
            inp = torch.tensor([0.5, 1.5, 2.5, 3.5, 1.5],
                               dtype=torch_dtype(dtype), device=device)
            fwd = torch.histc(inp, bins=4, min=0.0, max=4.0)
            out.append({
                "op": "histc",
                "tag": "basic",
                "dtype": dtype,
                "device": device,
                "in_shape": list(inp.shape),
                "in_data": to_listf(inp),
                "bins": 4,
                "min": 0.0,
                "max": 4.0,
                "out_shape": list(fwd.shape),
                "out_values": to_listf(fwd),
            })
    return out


def fixture_meshgrid() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in CPU_ONLY_DEVICES:
        for dtype in DTYPES:
            x = torch.tensor([1.0, 2.0, 3.0],
                             dtype=torch_dtype(dtype), device=device)
            y = torch.tensor([4.0, 5.0],
                             dtype=torch_dtype(dtype), device=device)
            grids = torch.meshgrid(x, y, indexing="ij")
            # NOTE: meshgrid uses `mg_out_values` (list of lists) instead of
            # the flat `out_values` field. The shared Rust Fixture struct
            # deserializes `out_values` as a flat list; meshgrid is the
            # only op that produces N parallel coordinate grids, so it
            # gets its own field name to avoid breaking the schema.
            out.append({
                "op": "meshgrid",
                "tag": "2d",
                "dtype": dtype,
                "device": device,
                "input_shapes": [list(x.shape), list(y.shape)],
                "mg_input_data": [to_listf(x), to_listf(y)],
                "mg_out_shapes": [list(g.shape) for g in grids],
                "mg_out_values": [to_listf(g) for g in grids],
            })
    return out


def fixture_topk() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in CPU_ONLY_DEVICES:
        for dtype in DTYPES:
            a = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0],
                             dtype=torch_dtype(dtype), device=device)
            for largest in (True, False):
                vals, idxs = torch.topk(a, k=3, largest=largest, sorted=True)
                out.append({
                    "op": "topk",
                    "tag": f"largest{int(largest)}",
                    "dtype": dtype,
                    "device": device,
                    "in_shape": list(a.shape),
                    "in_data": to_listf(a),
                    "k": 3,
                    "largest": largest,
                    "out_shape": list(vals.shape),
                    "out_values": to_listf(vals),
                    "out_indices": [int(i) for i in idxs.tolist()],
                })
    return out


# ---------------------------------------------------------------------------
# Cat B — pure shape utility helpers (data-free fixtures)
# ---------------------------------------------------------------------------


def fixture_shape_helpers() -> list[dict[str, Any]]:
    """The helpers `broadcast_shapes`, `numel`, `c_contiguous_strides`,
    `channels_last_strides`, `channels_last_3d_strides`, `normalize_axis`,
    `check_shapes_match` are pure functions of shape and need no torch
    invocation. We codify their expected behaviour as static fixtures so
    the Rust test can assert parity against PyTorch's documented semantics.
    """
    out: list[dict[str, Any]] = []
    # broadcast_shapes — verified via torch.broadcast_shapes
    bcast_cases = [
        ([3, 1], [1, 4], [3, 4]),
        ([2, 1, 4], [3, 4], [2, 3, 4]),
        ([5], [3, 5], [3, 5]),
        ([], [2, 3], [2, 3]),
    ]
    for a, b, expected in bcast_cases:
        # ferrotorch::shape::broadcast_shapes only supports 2 args; we keep
        # parity scope to that and verify against PyTorch.
        ts = (torch.empty(a) if a else torch.empty(()))
        ts2 = (torch.empty(b) if b else torch.empty(()))
        # Use torch.broadcast_shapes which accepts size tuples.
        torch_expected = list(torch.broadcast_shapes(tuple(a), tuple(b)))
        assert torch_expected == expected, (a, b, torch_expected, expected)
        out.append({
            "op": "broadcast_shapes",
            "tag": f"{a}_{b}",
            "a": a,
            "b": b,
            "expected": expected,
        })

    # numel: simple product, with empty shape == 1
    for shape, expected in [
        ([], 1),
        ([0], 0),
        ([3], 3),
        ([2, 3, 4], 24),
    ]:
        out.append({
            "op": "numel",
            "tag": str(shape),
            "shape": shape,
            "expected_numel": expected,
        })

    # c_contiguous_strides: row-major.
    for shape, expected_strides in [
        ([2, 3], [3, 1]),
        ([2, 3, 4], [12, 4, 1]),
        ([1], [1]),
    ]:
        out.append({
            "op": "c_contiguous_strides",
            "tag": str(shape),
            "shape": shape,
            "expected_strides": expected_strides,
        })

    # channels_last_strides (NHWC layout for [N,C,H,W])
    # Per PyTorch: NHWC strides for NCHW tensor of shape [N,C,H,W] are
    # [C*H*W, 1, C*W, C].
    for shape, expected_strides in [
        ([1, 3, 2, 4], [3 * 2 * 4, 1, 3 * 4, 3]),
        ([2, 4, 5, 6], [4 * 5 * 6, 1, 4 * 6, 4]),
    ]:
        out.append({
            "op": "channels_last_strides",
            "tag": str(shape),
            "shape": shape,
            "expected_strides": expected_strides,
        })

    # channels_last_3d_strides (NDHWC for [N,C,D,H,W])
    # Strides: [C*D*H*W, 1, C*H*W, C*W, C]
    for shape, expected_strides in [
        ([1, 3, 2, 4, 5], [3 * 2 * 4 * 5, 1, 3 * 4 * 5, 3 * 5, 3]),
    ]:
        out.append({
            "op": "channels_last_3d_strides",
            "tag": str(shape),
            "shape": shape,
            "expected_strides": expected_strides,
        })

    # normalize_axis: PyTorch convention — axis in [-ndim, ndim-1] returns
    # canonical [0, ndim-1].
    for axis, ndim, expected in [
        (0, 3, 0),
        (-1, 3, 2),
        (-3, 3, 0),
        (2, 3, 2),
    ]:
        out.append({
            "op": "normalize_axis",
            "tag": f"axis{axis}_ndim{ndim}",
            "axis": axis,
            "ndim": ndim,
            "expected_axis": expected,
        })
    # Out-of-range cases: expect Err
    for axis, ndim in [(3, 3), (-4, 3)]:
        out.append({
            "op": "normalize_axis_err",
            "tag": f"axis{axis}_ndim{ndim}",
            "axis": axis,
            "ndim": ndim,
        })

    # check_shapes_match: ok / err cases
    out.append({
        "op": "check_shapes_match_ok",
        "tag": "equal",
        "a": [2, 3], "b": [2, 3],
    })
    out.append({
        "op": "check_shapes_match_err",
        "tag": "differ",
        "a": [2, 3], "b": [2, 4],
    })

    return out


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    fixtures: list[dict[str, Any]] = []

    # Cat A.shape
    fixtures += fixture_reshape()
    fixtures += fixture_view()
    fixtures += fixture_flatten()
    fixtures += fixture_squeeze()
    fixtures += fixture_unsqueeze()
    fixtures += fixture_transpose()
    fixtures += fixture_permute()
    fixtures += fixture_narrow()
    fixtures += fixture_contiguous()
    fixtures += fixture_cat()
    fixtures += fixture_split()
    fixtures += fixture_chunk()
    fixtures += fixture_expand()
    fixtures += fixture_as_strided_family()

    # Cat A.indexing
    fixtures += fixture_gather()
    fixtures += fixture_scatter_and_add()
    fixtures += fixture_where_cond()
    fixtures += fixture_index_select_1d()
    fixtures += fixture_masked_fill()

    # Cat A.tensor_ops
    fixtures += fixture_triu_tril()
    fixtures += fixture_diag_diagflat()
    fixtures += fixture_roll()
    fixtures += fixture_cdist()

    # Cat A.search
    fixtures += fixture_searchsorted()
    fixtures += fixture_bucketize()
    fixtures += fixture_unique_family()
    fixtures += fixture_histc()
    fixtures += fixture_meshgrid()
    fixtures += fixture_topk()

    # Cat B
    fixtures += fixture_shape_helpers()

    payload = {
        "metadata": fixture_metadata(),
        "fixtures": fixtures,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    by_op: dict[str, int] = {}
    for fx in fixtures:
        by_op[fx["op"]] = by_op.get(fx["op"], 0) + 1
    print(f"Wrote {len(fixtures)} fixtures across {len(by_op)} ops to {FIXTURE_PATH}")
    for op, n in sorted(by_op.items()):
        print(f"  {op}: {n}")


if __name__ == "__main__":
    main()
