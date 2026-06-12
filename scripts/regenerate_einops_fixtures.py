#!/usr/bin/env python3
"""Regenerate PyTorch reference fixtures for ferrotorch-core Phase 2.6
(einops + einsum).

Tracking issue: #768 (parent: #759).

Output:
    ferrotorch-core/tests/conformance/fixtures/einops.json

Coverage (7 canonical-path surface items):

* `ferrotorch_core::einops::EinopsReduction` — exercised transitively through
  every `reduce` fixture (Sum / Mean / Max / Min variants are tagged so each
  reduction enum discriminator is covered).
* `ferrotorch_core::einops::rearrange` — pattern-matrix: identity, transpose
  (NHWC->NCHW), pure flatten, axis-merge, axis-split (left-side group).
* `ferrotorch_core::einops::rearrange_with` — split with `axes_lengths` (the
  feature that distinguishes it from `rearrange`).
* `ferrotorch_core::einops::reduce` — sum / mean / max / min, both
  axis-aligned-fast-path and reorder-fallback shapes.
* `ferrotorch_core::einops::repeat` — single-new-axis broadcast, multi-new-axis
  expansion, tile-existing-axis.
* `ferrotorch_core::einsum::einsum` — every contraction shape from the dispatch
  edge-case list: ij->ji (transpose), i-> (sum), ij,jk->ik (matmul),
  bij,bjk->bik (batched matmul), i,i-> (dot), i,i (implicit dot), i,j->ij
  (outer), ii-> (trace), ii->i (diagonal), ij,ij->ij (Hadamard),
  ij,jk,kl->il *NOT included — ferrotorch's einsum errors on >2 inputs by
  design; the test pins that error*.
* `ferrotorch_core::einsum::einsum_differentiable` — autograd on the matmul
  case (the only differentiable shape ferrotorch handles fully end-to-end on
  the device the input lives on, modulo the CPU compute detour).

Tolerances follow the dispatch table (matmul-like contractions use
F32_MATMUL_GPU = 1e-3; pure transposes are bit-exact; reductions use
F32_REDUCTION; rearrange / repeat are bit-exact).

Usage from WSL (preferred per #777):

    python3 scripts/regenerate_einops_fixtures.py

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
    / "einops.json"
)

DTYPES: list[str] = ["float32", "float64"]
DEVICES: list[str] = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda:0")

RNG_SEED: int = 0xE1A055
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
# rearrange — pattern matrix
# ---------------------------------------------------------------------------
#
# Each entry: (tag, pattern, input_shape). For `rearrange_with`, the test
# also passes axes_lengths; we encode that in the `axes_lengths` field as a
# list of [name, size] pairs.
#
# We compute PyTorch's reference via `einops.rearrange` if available, else
# native torch ops (transpose / reshape).


def _rearrange_ref(t: torch.Tensor, pattern: str, axes_lengths: dict[str, int]) -> torch.Tensor:
    """Reference rearrange via the einops package if installed; else
    fall back to a hand-coded dispatch covering only the patterns we exercise.

    We require einops here so the reference matches the upstream definition
    exactly. The script bails loudly if einops is unavailable.
    """
    try:
        from einops import rearrange  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "einops package is required to regenerate this fixture. "
            "Install with: pip install einops"
        ) from exc
    return rearrange(t, pattern, **axes_lengths)


REARRANGE_CASES: list[tuple[str, str, list[int], dict[str, int]]] = [
    # (tag, pattern, input_shape, axes_lengths)
    ("identity_4d", "b c h w -> b c h w", [2, 3, 2, 2], {}),
    ("transpose_nhwc_to_nchw", "b h w c -> b c h w", [1, 2, 2, 3], {}),
    ("transpose_2d", "i j -> j i", [2, 3], {}),
    ("flatten_trailing", "b c h w -> b (c h w)", [2, 3, 2, 2], {}),
    ("merge_hw", "b h w c -> b (h w) c", [1, 2, 3, 4], {}),
    # rearrange_with — axis split (requires axes_lengths)
    ("split_with_c", "b (c h) w -> b c h w", [2, 6, 4], {"c": 3}),
    # rearrange_with — split-then-permute (split a dim AND reorder kept dims)
    ("split_and_permute", "b (h2 w2) c -> b c h2 w2", [1, 6, 4], {"h2": 2}),
]


def fixture_rearrange() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for op_name in ("rearrange", "rearrange_with"):
        for tag, pattern, shape, axes_lengths in REARRANGE_CASES:
            # `rearrange` (no axes_lengths) only exercises the cases that
            # don't need them; `rearrange_with` exercises all of them.
            if op_name == "rearrange" and axes_lengths:
                continue
            for device in DEVICES:
                for dtype in DTYPES:
                    n = max(1, math.prod(shape))
                    a = torch.arange(
                        n, dtype=torch_dtype(dtype), device=device
                    ).reshape(shape)
                    fwd = _rearrange_ref(a, pattern, axes_lengths)
                    out.append(
                        {
                            "op": op_name,
                            "tag": tag,
                            "pattern": pattern,
                            "axes_lengths": [
                                [k, v] for k, v in axes_lengths.items()
                            ],
                            "dtype": dtype,
                            "device": device,
                            "a_shape": shape,
                            "a_data": to_listf(a),
                            "out_shape": list(fwd.shape),
                            "out_values": to_listf(fwd),
                        }
                    )
    return out


# ---------------------------------------------------------------------------
# repeat — covers the 5th surface item
# ---------------------------------------------------------------------------


REPEAT_CASES: list[tuple[str, str, list[int], dict[str, int]]] = [
    # (tag, pattern, input_shape, axes_lengths)
    ("new_batch_dim", "h w -> b h w", [2, 2], {"b": 3}),
    ("tile_trailing", "c -> c n", [3], {"n": 2}),
    ("multi_new_axes", "c -> b c n", [3], {"b": 2, "n": 2}),
    # CORE-062 / #1756 — kept-axis reorder combined with new axes. Pre-fix,
    # ferrotorch collected source coordinates in right-pattern order but
    # flattened them with the left shape (wrong elements / OOB reads).
    ("reorder_new_trailing", "a b -> b a c", [2, 3], {"c": 2}),
    ("reorder_new_leading", "a b -> c b a", [2, 3], {"c": 2}),
    ("reorder_new_in_merge", "a b -> b (a c)", [2, 3], {"c": 2}),
    ("reorder_merged_kept_new", "a b -> (b c) a", [2, 3], {"c": 3}),
    ("split_reorder_new", "(a b) -> b c a", [4], {"b": 2, "c": 2}),
    ("pure_reorder", "a b -> b a", [2, 3], {}),
]


def fixture_repeat() -> list[dict[str, Any]]:
    try:
        from einops import repeat  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "einops package is required to regenerate this fixture."
        ) from exc

    out: list[dict[str, Any]] = []
    for tag, pattern, shape, axes_lengths in REPEAT_CASES:
        for device in DEVICES:
            for dtype in DTYPES:
                n = max(1, math.prod(shape))
                a = torch.arange(
                    1, n + 1, dtype=torch_dtype(dtype), device=device
                ).reshape(shape)
                fwd = repeat(a, pattern, **axes_lengths)
                out.append(
                    {
                        "op": "repeat",
                        "tag": tag,
                        "pattern": pattern,
                        "axes_lengths": [[k, v] for k, v in axes_lengths.items()],
                        "dtype": dtype,
                        "device": device,
                        "a_shape": shape,
                        "a_data": to_listf(a),
                        "out_shape": list(fwd.shape),
                        "out_values": to_listf(fwd),
                    }
                )
    return out


# ---------------------------------------------------------------------------
# reduce — Sum/Mean/Max/Min
# ---------------------------------------------------------------------------


REDUCE_CASES: list[tuple[str, str, list[int]]] = [
    # (tag, pattern, input_shape)
    ("global_avg_pool", "b c h w -> b c", [1, 2, 2, 2]),
    ("sum_batch", "b c -> c", [3, 2]),
    ("trailing_full_pool", "b c h w -> b", [2, 2, 2, 3]),
    # CORE-062 / #1756 — kept-axis reorder (forces the non-fast-path mapping).
    # Pre-fix, ferrotorch collected kept coordinates in left order but
    # flattened them with the right-order output shape (wrong positions /
    # OOB accumulator writes — kept_reorder_wide panicked).
    ("kept_reorder", "a b c -> c a", [2, 3, 4]),
    ("kept_reorder_wide", "a b c -> c a", [5, 3, 2]),
    ("kept_reorder_trailing_reduced", "a b c -> b a", [2, 3, 4]),
    ("kept_reorder_merged", "a b c -> (c a)", [2, 3, 4]),
]


def _reduce_ref(t: torch.Tensor, pattern: str, op: str) -> torch.Tensor:
    from einops import reduce  # type: ignore

    return reduce(t, pattern, op)


def fixture_reduce() -> list[dict[str, Any]]:
    try:
        from einops import reduce  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "einops package is required to regenerate this fixture."
        ) from exc

    out: list[dict[str, Any]] = []
    for op in ("sum", "mean", "max", "min"):
        for tag, pattern, shape in REDUCE_CASES:
            for device in DEVICES:
                for dtype in DTYPES:
                    n = max(1, math.prod(shape))
                    # Use a non-monotonic pattern so max != last and min != first.
                    raw_vals = [(((i * 7) % 13) + 1) * 0.5 for i in range(n)]
                    a = torch.tensor(
                        raw_vals, dtype=torch_dtype(dtype), device=device
                    ).reshape(shape)
                    fwd = _reduce_ref(a, pattern, op)
                    out.append(
                        {
                            "op": "reduce",
                            "reduction": op,
                            "tag": tag,
                            "pattern": pattern,
                            "dtype": dtype,
                            "device": device,
                            "a_shape": shape,
                            "a_data": to_listf(a),
                            "out_shape": list(fwd.shape),
                            "out_values": to_listf(fwd),
                        }
                    )
    return out


# ---------------------------------------------------------------------------
# einsum — single-input + two-input contractions
# ---------------------------------------------------------------------------
#
# Coverage matrix (per #768 dispatch edge-case list):
#  * "ij->ji"     transpose         (single-input)
#  * "i->"        sum               (single-input)
#  * "ij,jk->ik"  matmul            (two-input)
#  * "bij,bjk->bik" batched matmul  (two-input)
#  * "i,i->"      dot product       (two-input)
#  * "i,i"        implicit dot      (two-input, no arrow)
#  * "i,j->ij"    outer product     (two-input)
#  * "ii->"       trace             (single-input, repeated index)
#  * "ii->i"      diagonal          (single-input, repeated index)
#  * "ij,ij->ij"  Hadamard          (two-input, elementwise)
#  * "ij->"       full sum          (single-input)
#  * "ij->i"      axis sum          (single-input)
#
# Triple contraction "ij,jk,kl->il" is excluded — ferrotorch's einsum
# explicitly errors on >2 inputs (see einsum::einsum body); we add a
# negative-shape test in Rust that exercises this error path directly,
# no fixture needed.


EINSUM_CASES_SINGLE: list[tuple[str, str, list[int]]] = [
    # (tag, equation, input_shape)
    ("transpose_2d", "ij->ji", [2, 3]),
    ("sum_1d", "i->", [4]),
    ("trace_2d", "ii->", [3, 3]),
    ("diagonal_2d", "ii->i", [3, 3]),
    ("full_sum_2d", "ij->", [2, 3]),
    ("axis_sum_2d", "ij->i", [2, 3]),
]

EINSUM_CASES_TWO: list[tuple[str, str, list[int], list[int]]] = [
    # (tag, equation, a_shape, b_shape)
    ("matmul_2x2", "ij,jk->ik", [2, 2], [2, 2]),
    ("matmul_nonsquare", "ij,jk->ik", [2, 3], [3, 4]),
    ("bmm", "bij,bjk->bik", [2, 2, 3], [2, 3, 4]),
    ("dot", "i,i->", [4], [4]),
    ("dot_implicit", "i,i", [4], [4]),
    ("outer", "i,j->ij", [3], [2]),
    ("hadamard", "ij,ij->ij", [2, 3], [2, 3]),
]


def fixture_einsum() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # Single-input — forward only (autograd on permutations like "ij->ji"
    # is straightforward; we cover that via einsum_differentiable below).
    for tag, eq, shape in EINSUM_CASES_SINGLE:
        for device in DEVICES:
            for dtype in DTYPES:
                n = max(1, math.prod(shape))
                a = torch.arange(
                    1, n + 1, dtype=torch_dtype(dtype), device=device
                ).reshape(shape)
                fwd = torch.einsum(eq, a)
                out.append(
                    {
                        "op": "einsum",
                        "tag": tag,
                        "equation": eq,
                        "dtype": dtype,
                        "device": device,
                        "a_shape": shape,
                        "a_data": to_listf(a),
                        "out_shape": list(fwd.shape),
                        "out_values": to_listf(fwd),
                    }
                )

    # Two-input — forward only.
    for tag, eq, a_shape, b_shape in EINSUM_CASES_TWO:
        for device in DEVICES:
            for dtype in DTYPES:
                an = max(1, math.prod(a_shape))
                bn = max(1, math.prod(b_shape))
                a = torch.arange(
                    1, an + 1, dtype=torch_dtype(dtype), device=device
                ).reshape(a_shape)
                b = torch.arange(
                    1, bn + 1, dtype=torch_dtype(dtype), device=device
                ).reshape(b_shape)
                fwd = torch.einsum(eq, a, b)
                out.append(
                    {
                        "op": "einsum",
                        "tag": tag,
                        "equation": eq,
                        "dtype": dtype,
                        "device": device,
                        "a_shape": a_shape,
                        "b_shape": b_shape,
                        "a_data": to_listf(a),
                        "b_data": to_listf(b),
                        "out_shape": list(fwd.shape),
                        "out_values": to_listf(fwd),
                    }
                )

    return out


# ---------------------------------------------------------------------------
# einsum_differentiable — forward + backward on matmul + transpose
# ---------------------------------------------------------------------------


def fixture_einsum_differentiable() -> list[dict[str, Any]]:
    """Forward + backward grads for a small set of differentiable einsum
    shapes. Loss is the sum-reduction of the output (the canonical scalar
    loss used throughout this conformance suite).
    """
    out: list[dict[str, Any]] = []

    cases: list[tuple[str, str, list[int], list[int] | None]] = [
        ("matmul_2x2", "ij,jk->ik", [2, 2], [2, 2]),
        ("matmul_nonsquare", "ij,jk->ik", [2, 3], [3, 4]),
        ("bmm", "bij,bjk->bik", [2, 2, 3], [2, 3, 4]),
        ("dot", "i,i->", [4], [4]),
        ("outer", "i,j->ij", [3], [2]),
        ("hadamard", "ij,ij->ij", [2, 3], [2, 3]),
        # Single-input differentiable cases (forward only — the grad is a
        # plain reverse-permutation, also tested by EinsumBackwardSingle).
        ("transpose", "ij->ji", [2, 3], None),
        ("axis_sum", "ij->i", [2, 3], None),
    ]

    for tag, eq, a_shape, b_shape in cases:
        for device in DEVICES:
            for dtype in DTYPES:
                an = max(1, math.prod(a_shape))
                a = torch.arange(
                    1, an + 1, dtype=torch_dtype(dtype), device=device
                ).reshape(a_shape)
                a_g = a.detach().clone().requires_grad_(True)
                if b_shape is None:
                    fwd = torch.einsum(eq, a_g)
                    loss = fwd.sum()
                    loss.backward()
                    rec = {
                        "op": "einsum_differentiable",
                        "tag": tag,
                        "equation": eq,
                        "dtype": dtype,
                        "device": device,
                        "a_shape": a_shape,
                        "a_data": to_listf(a),
                        "out_shape": list(fwd.shape),
                        "out_values": to_listf(fwd),
                        "grad_a": to_listf(a_g.grad),
                    }
                else:
                    bn = max(1, math.prod(b_shape))
                    b = torch.arange(
                        1, bn + 1, dtype=torch_dtype(dtype), device=device
                    ).reshape(b_shape)
                    b_g = b.detach().clone().requires_grad_(True)
                    fwd = torch.einsum(eq, a_g, b_g)
                    loss = fwd.sum()
                    loss.backward()
                    rec = {
                        "op": "einsum_differentiable",
                        "tag": tag,
                        "equation": eq,
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
                out.append(rec)

    return out


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def main() -> int:
    fixtures: list[dict[str, Any]] = []
    fixtures.extend(fixture_rearrange())
    fixtures.extend(fixture_repeat())
    fixtures.extend(fixture_reduce())
    fixtures.extend(fixture_einsum())
    fixtures.extend(fixture_einsum_differentiable())

    payload = {
        "metadata": fixture_metadata(),
        "fixtures": fixtures,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"wrote {len(fixtures)} fixtures to {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
