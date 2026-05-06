#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for the ferrotorch-core creation module.

Tracking issue: #759 (conformance phase 2.0 — ferrotorch-core::creation).

Output: ``ferrotorch-core/tests/conformance/fixtures/creation.json``.

The fixture file pairs every public function in ``ferrotorch-core/src/creation.rs``
with the values PyTorch produces for the same arguments. The Rust-side
conformance test (``tests/conformance_creation.rs``) loads this JSON and
compares its op outputs against these values, on both CPU and CUDA when
``--features gpu`` is enabled.

How fixtures are generated per-op:

* **Deterministic ops** (``zeros``, ``ones``, ``full``, ``from_slice``,
  ``from_vec``, ``tensor``, ``scalar``, ``eye``, ``arange``, ``linspace``):
  exact values are recorded.

* **Random ops** (``rand``, ``randn``, ``rand_like``, ``randn_like``):
  10K-element samples are drawn and the *distribution moments* (mean,
  variance, min, max, kurtosis, count outside 4σ) are recorded. The
  Rust test asserts moments match within statistical tolerance, NOT raw
  values — ferrotorch's RNG and PyTorch's RNG use different algorithms.

* **Meta ops** (``zeros_meta``, ``ones_meta``, ``full_meta``, ``meta_like``):
  PyTorch's analog is ``torch.empty(*, device='meta')``. Meta tensors carry
  shape but no value, so the fixture records shape + dtype + device only.

* **``_like`` variants**: PyTorch's ``torch.zeros_like(other)`` returns a
  tensor with the same shape/dtype/device as ``other``. Fixture records
  the input shape and the expected matching shape.

* **``requires_grad`` paths**: PyTorch's ``torch.zeros(..., requires_grad=True)``
  on a float-typed tensor produces a leaf tensor with ``grad_fn=None`` and
  ``is_leaf=True``. The fixture records both flags so the Rust test can
  assert the same.

Usage from WSL (preferred — Linux-native after #777):

    python3 scripts/regenerate_core_creation_fixtures.py

Required Python deps (installed in WSL via ``pip install --user`` per #777):

    torch>=2.5  (with CUDA support to populate the cuda paths)
    numpy

Fallback via the Windows host Python (only if WSL install is unavailable;
this was the original Path-2 workflow before #777):

    /mnt/c/Users/texas/AppData/Local/Programs/Python/Python312/python.exe \\
        scripts/regenerate_core_creation_fixtures.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import platform
from pathlib import Path
from typing import Any

import torch  # type: ignore  # provided by the user's Windows-side site-packages

# ----------------------------------------------------------------------------
# Output paths and metadata
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "ferrotorch-core" / "tests" / "conformance" / "fixtures" / "creation.json"

# Shapes covered for non-RNG ops. Includes scalar `[]`, 1-D, 2-D, 3-D, 4-D
# (the broadcasting edge), and `[0,3]` (the zero-size-dim edge). Very large
# shapes are out of scope for fixture data — they would bloat the JSON file
# without changing the conformance contract.
NON_RNG_SHAPES: list[list[int]] = [
    [],          # scalar
    [3],         # 1-D
    [2, 3],      # 2-D
    [4, 5, 6],   # 3-D
    [1, 1, 1, 1],# 4-D broadcasting edge
    [0, 3],      # zero-size dim
]
DTYPES: list[str] = ["float32", "float64"]
DEVICES: list[str] = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda:0")

# RNG ops use a single (large) sample to make moment comparisons meaningful.
RNG_SHAPE: list[int] = [10_000]
RNG_SEED: int = 0xC0FFEE  # arbitrary fixed seed; does not need parity with ferrotorch.


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def to_listf(t: torch.Tensor) -> list[float]:
    """Materialize a tensor to a CPU Python list of floats (NaN-safe)."""
    return t.detach().to("cpu").to(torch.float64).reshape(-1).tolist()


def fixture_metadata() -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rng_seed": RNG_SEED,
        "non_rng_shapes": NON_RNG_SHAPES,
        "dtypes": DTYPES,
        "devices": DEVICES,
    }


# ----------------------------------------------------------------------------
# Per-op fixture builders
# ----------------------------------------------------------------------------

def fixture_const(name: str, ctor) -> list[dict[str, Any]]:
    """Fixture builder for ops that take only a shape: zeros, ones."""
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            for shape in NON_RNG_SHAPES:
                t = ctor(shape, dtype=torch_dtype(dtype), device=device)
                out.append({
                    "op": name,
                    "shape": shape,
                    "dtype": dtype,
                    "device": device,
                    "values": to_listf(t),
                })
    return out


def fixture_full() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    fill_values = [0.0, 1.0, -1.5, 7.5]
    for device in DEVICES:
        for dtype in DTYPES:
            for shape in NON_RNG_SHAPES:
                for fv in fill_values:
                    t = torch.full(shape, fv, dtype=torch_dtype(dtype), device=device)
                    out.append({
                        "op": "full",
                        "shape": shape,
                        "dtype": dtype,
                        "device": device,
                        "fill_value": fv,
                        "values": to_listf(t),
                    })
    return out


def fixture_eye() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            for n in (1, 3, 5, 8):
                t = torch.eye(n, dtype=torch_dtype(dtype), device=device)
                out.append({
                    "op": "eye",
                    "n": n,
                    "dtype": dtype,
                    "device": device,
                    "values": to_listf(t),
                })
    return out


def fixture_arange() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cases = [
        (0.0, 5.0, 1.0),
        (1.0, 4.0, 0.5),
        (5.0, 0.0, -1.0),
        (-2.0, 3.0, 0.25),
        (0.0, 0.0, 1.0),  # empty
    ]
    for device in DEVICES:
        for dtype in DTYPES:
            for (start, end, step) in cases:
                t = torch.arange(start, end, step, dtype=torch_dtype(dtype), device=device)
                out.append({
                    "op": "arange",
                    "start": start,
                    "end": end,
                    "step": step,
                    "dtype": dtype,
                    "device": device,
                    "values": to_listf(t),
                })
    return out


def fixture_linspace() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cases = [
        (0.0, 1.0, 5),
        (-1.0, 1.0, 11),
        (3.0, 3.0, 1),  # degenerate single-point
        (0.0, 1.0, 0),  # empty
        (0.0, 10.0, 100),
    ]
    for device in DEVICES:
        for dtype in DTYPES:
            for (start, end, num) in cases:
                t = torch.linspace(start, end, num, dtype=torch_dtype(dtype), device=device)
                out.append({
                    "op": "linspace",
                    "start": start,
                    "end": end,
                    "num": num,
                    "dtype": dtype,
                    "device": device,
                    "values": to_listf(t),
                })
    return out


def fixture_from_slice() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cases = [
        ([1.0, 2.0, 3.0, 4.0], [2, 2]),
        ([0.5, -0.25, 1.75], [3]),
        ([float(i) for i in range(24)], [2, 3, 4]),
        ([], [0]),
        ([7.5], []),  # scalar from 1-element slice
    ]
    for device in DEVICES:
        for dtype in DTYPES:
            for (data, shape) in cases:
                t = torch.tensor(data, dtype=torch_dtype(dtype), device=device).reshape(shape)
                out.append({
                    "op": "from_slice",
                    "data": data,
                    "shape": shape,
                    "dtype": dtype,
                    "device": device,
                    "values": to_listf(t),
                })
    return out


def fixture_from_vec() -> list[dict[str, Any]]:
    # Same generation as from_slice; ferrotorch's `from_vec` differs only in
    # ownership semantics. The fixture exercises the same cases so the test
    # can call the appropriate ferrotorch entry point.
    out = fixture_from_slice()
    for entry in out:
        entry["op"] = "from_vec"
    return out


def fixture_tensor() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cases = [
        [1.0, 2.0, 3.0],
        [0.5, -0.25, 1.75, 9.0],
        [float(i) * 0.1 for i in range(20)],
        [],
    ]
    for device in DEVICES:
        for dtype in DTYPES:
            for data in cases:
                t = torch.tensor(data, dtype=torch_dtype(dtype), device=device)
                out.append({
                    "op": "tensor",
                    "data": data,
                    "shape": [len(data)],
                    "dtype": dtype,
                    "device": device,
                    "values": to_listf(t),
                })
    return out


def fixture_scalar() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cases = [0.0, 1.0, -1.0, 42.0, 3.14]
    for device in DEVICES:
        for dtype in DTYPES:
            for v in cases:
                t = torch.tensor(v, dtype=torch_dtype(dtype), device=device)
                out.append({
                    "op": "scalar",
                    "value": v,
                    "dtype": dtype,
                    "device": device,
                    "values": to_listf(t),
                })
    return out


def fixture_meta(name: str) -> list[dict[str, Any]]:
    """Meta tensors carry shape + dtype only — no values."""
    out: list[dict[str, Any]] = []
    for dtype in DTYPES:
        for shape in NON_RNG_SHAPES:
            t = torch.empty(shape, dtype=torch_dtype(dtype), device="meta")
            entry: dict[str, Any] = {
                "op": name,
                "shape": shape,
                "dtype": dtype,
                "device": "meta",
                "numel": t.numel(),
            }
            if name == "full_meta":
                entry["fill_value"] = 0.0  # meta carries no value; arbitrary
            out.append(entry)
    return out


def fixture_meta_like() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dtype in DTYPES:
        for shape in NON_RNG_SHAPES:
            base = torch.zeros(shape, dtype=torch_dtype(dtype), device="cpu")
            # meta_like(other) always lives on the meta device irrespective
            # of `other`'s device (matches ferrotorch's docstring).
            t = torch.empty_like(base, device="meta")
            out.append({
                "op": "meta_like",
                "shape": shape,
                "dtype": dtype,
                "input_device": "cpu",
                "device": "meta",
                "numel": t.numel(),
            })
    return out


def fixture_like(name: str, ctor) -> list[dict[str, Any]]:
    """zeros_like / ones_like / full_like."""
    out: list[dict[str, Any]] = []
    fill = 7.5 if name == "full_like" else None
    for device in DEVICES:
        for dtype in DTYPES:
            for shape in NON_RNG_SHAPES:
                base = torch.zeros(shape, dtype=torch_dtype(dtype), device=device)
                if name == "full_like":
                    t = torch.full_like(base, fill)
                elif name == "zeros_like":
                    t = torch.zeros_like(base)
                else:  # ones_like
                    t = torch.ones_like(base)
                entry: dict[str, Any] = {
                    "op": name,
                    "shape": shape,
                    "dtype": dtype,
                    "device": device,
                    "values": to_listf(t),
                }
                if fill is not None:
                    entry["fill_value"] = fill
                out.append(entry)
    return out


def moments(samples: list[float]) -> dict[str, float]:
    """Distribution moments for RNG conformance.

    `kurtosis` is excess kurtosis (subtract 3); a Gaussian has 0, a uniform
    has -6/5 = -1.2. `outside_4sigma` is the count of samples whose magnitude
    after standardization exceeds 4. These two extra fields catch RNG bugs
    that pass mean/variance but generate the wrong tail shape (e.g. a
    truncated normal vs. a true normal).
    """
    n = len(samples)
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / n
    std = math.sqrt(var) if var > 0 else 1.0
    m4 = sum((x - mean) ** 4 for x in samples) / n
    excess_kurt = m4 / (var ** 2) - 3 if var > 0 else 0.0
    outside_4sigma = sum(1 for x in samples if abs(x - mean) > 4 * std)
    return {
        "n": n,
        "mean": mean,
        "var": var,
        "std": std,
        "min": min(samples),
        "max": max(samples),
        "excess_kurtosis": excess_kurt,
        "outside_4sigma": outside_4sigma,
    }


def fixture_rng(name: str) -> list[dict[str, Any]]:
    """rand / randn / rand_like / randn_like."""
    out: list[dict[str, Any]] = []
    expected_dist = {
        "rand": "uniform_0_1",
        "randn": "standard_normal",
        "rand_like": "uniform_0_1",
        "randn_like": "standard_normal",
    }[name]
    for device in DEVICES:
        for dtype in DTYPES:
            torch.manual_seed(RNG_SEED)
            if device.startswith("cuda"):
                torch.cuda.manual_seed_all(RNG_SEED)
            shape = RNG_SHAPE
            if name == "rand":
                t = torch.rand(shape, dtype=torch_dtype(dtype), device=device)
            elif name == "randn":
                t = torch.randn(shape, dtype=torch_dtype(dtype), device=device)
            elif name == "rand_like":
                base = torch.zeros(shape, dtype=torch_dtype(dtype), device=device)
                t = torch.rand_like(base)
            else:  # randn_like
                base = torch.zeros(shape, dtype=torch_dtype(dtype), device=device)
                t = torch.randn_like(base)
            samples = to_listf(t)
            out.append({
                "op": name,
                "shape": shape,
                "dtype": dtype,
                "device": device,
                "expected_distribution": expected_dist,
                "moments": moments(samples),
            })
    return out


def fixture_requires_grad() -> list[dict[str, Any]]:
    """`requires_grad=True` parity. PyTorch's leaf creation produces a tensor
    with `requires_grad=True`, `is_leaf=True`, and `grad_fn=None`. After a
    downstream op (e.g. `t * 2.0`) the result has a non-None grad_fn.
    """
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            shape = [2, 3]
            t = torch.zeros(shape, dtype=torch_dtype(dtype), device=device, requires_grad=True)
            out.append({
                "op": "requires_grad_leaf",
                "factory": "zeros",
                "shape": shape,
                "dtype": dtype,
                "device": device,
                "requires_grad": True,
                "is_leaf": bool(t.is_leaf),
                "grad_fn_is_none": (t.grad_fn is None),
            })
            # Downstream op: t.sum() should have a grad_fn (`SumBackward0`
            # in PyTorch). Record only the *presence* — the symbolic name
            # of ferrotorch's grad_fn does not need to match PyTorch's.
            s = t.sum()
            out.append({
                "op": "requires_grad_after_sum",
                "factory": "zeros",
                "shape": shape,
                "dtype": dtype,
                "device": device,
                "requires_grad": True,
                "is_leaf": bool(s.is_leaf),
                "grad_fn_is_none": (s.grad_fn is None),
                "grad_fn_name": type(s.grad_fn).__name__ if s.grad_fn is not None else None,
            })
    return out


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(FIXTURE_PATH), help="Output JSON path")
    p.add_argument("--check-only", action="store_true", help="Print summary only; do not write")
    args = p.parse_args()

    print(f"torch        = {torch.__version__}")
    print(f"cuda_version = {torch.version.cuda}")
    print(f"cuda_avail   = {torch.cuda.is_available()}")
    print(f"devices      = {DEVICES}")

    fixture_data: dict[str, Any] = {
        "metadata": fixture_metadata(),
        "fixtures": [],
    }
    fixtures: list[dict[str, Any]] = fixture_data["fixtures"]

    # Deterministic ops
    fixtures += fixture_const("zeros", torch.zeros)
    fixtures += fixture_const("ones", torch.ones)
    fixtures += fixture_full()
    fixtures += fixture_eye()
    fixtures += fixture_arange()
    fixtures += fixture_linspace()
    fixtures += fixture_from_slice()
    fixtures += fixture_from_vec()
    fixtures += fixture_tensor()
    fixtures += fixture_scalar()

    # `_like` ops
    fixtures += fixture_like("zeros_like", torch.zeros_like)
    fixtures += fixture_like("ones_like", torch.ones_like)
    fixtures += fixture_like("full_like", torch.full_like)

    # Meta ops (PyTorch analog: torch.empty(*, device='meta'))
    fixtures += fixture_meta("zeros_meta")
    fixtures += fixture_meta("ones_meta")
    fixtures += fixture_meta("full_meta")
    fixtures += fixture_meta_like()

    # RNG ops — distribution moments only
    fixtures += fixture_rng("rand")
    fixtures += fixture_rng("randn")
    fixtures += fixture_rng("rand_like")
    fixtures += fixture_rng("randn_like")

    # requires_grad parity
    fixtures += fixture_requires_grad()

    if args.check_only:
        print(f"Would emit {len(fixtures)} fixture entries to {args.out}")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(fixture_data, f, indent=2, sort_keys=False)
    size_kb = os.path.getsize(out) / 1024.0
    print(f"Wrote {len(fixtures)} fixture entries to {out} ({size_kb:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
