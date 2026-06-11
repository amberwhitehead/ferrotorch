#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for ferrotorch-core Phase 2.12 (masked
tensors).

Tracking issue: #774 (parent: #759).

Output:
    ferrotorch-core/tests/conformance/fixtures/masked.json

Coverage (31 surface items in `_surface_exclusions.toml` filtered by
tracking_issue = "#774", spanning the `MaskedTensor` type + 8 free
functions + their top-level re-exports):

* Cat A — Construction:
    MaskedTensor::new, MaskedTensor::from_data, MaskedTensor::with_fill_value,
    masked_where, masked_invalid, masked_equal.
  Edge cases: matching all-true mask, partial mask, all-false mask, finite +
  NaN + ±Inf inputs (for `masked_invalid`), zero-vs-nonzero scrubbing
  (`masked_equal`).

* Cat A — Operations:
    masked_sum, masked_mean, masked_min, masked_max, masked_count,
    plus the consume-side accessors (filled / to_tensor / count_valid /
    count_masked / numel / shape / data / mask / fill_value).
  Edge cases per the dispatch:
    * all-true mask -> equivalent to the unmasked op
    * all-false mask -> sum=0, mean=NaN (both match torch.masked);
      min/max=NaN is a TRACKED DIVERGENCE (#1924, split out of CORE-197
      #1891): torch.masked amax/amin return a fully-masked 0-d MaskedTensor
      with the +/-inf identity payload on all-masked non-empty input, and
      RAISE IndexError on empty input, while ferrotorch returns a 0-d NaN
      tensor in both cases. The pinned rows carry a ``divergence`` note
      probed from live torch at generation time, and the Rust suite asserts
      the NaN pin explicitly (no NaN-tolerant laundering).
    * partial mask
    * empty masked tensor (len 0)

# torch.masked status

`torch.masked.MaskedTensor` is a *prototype* API (PyTorch ≥ 1.13). The
reduction semantics are stable, but the API surface moves. We compute the
reference values via `numpy.ma` (numpy.ma.masked_array) which is the
upstream definition torch.masked mirrors. Both produce identical numerics
for the ops we exercise (sum / mean / max / min / count). The fixture
metadata pins the torch + numpy versions for reproducibility.

# Mask convention bridge

ferrotorch (and torch.masked) use ``mask=True`` => VALID. NumPy uses
``mask=True`` => INVALID. We compute references in numpy (so we mirror
``mask=True`` => INVALID internally), then emit fixture data using
ferrotorch's "mask=true => valid" convention.

Usage from WSL (preferred per #777):

    python3 scripts/regenerate_masked_fixtures.py

Required Python deps: torch, numpy.
"""

from __future__ import annotations

import datetime
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
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
    / "masked.json"
)

DTYPES: list[str] = ["float32", "float64"]
DEVICES: list[str] = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda:0")

RNG_SEED: int = 0xBADCAFE
torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RNG_SEED)
np.random.seed(RNG_SEED & 0xFFFFFFFF)


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def numpy_dtype(name: str) -> np.dtype:
    return {"float32": np.float32, "float64": np.float64}[name]


def to_listf(values: Any) -> list[Any]:
    """Encode a 1-D iterable of floats with NaN/Inf sentinels."""
    encoded: list[Any] = []
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    for v in arr.tolist():
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
        "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rng_seed": RNG_SEED,
        "dtypes": DTYPES,
        "devices": DEVICES,
        # torch.masked is prototype; pin the API generation we observed so
        # divergences from a future torch release are caught here, not
        # silently absorbed by the conformance suite.
        "torch_masked_status": "prototype (≥1.13); reductions confirmed against numpy.ma",
    }


# ---------------------------------------------------------------------------
# Reference computation via numpy.ma
#
# ferrotorch convention: mask[i] == True => valid (use it).
# numpy.ma  convention: mask[i] == True => invalid (ignore it).
# Functions here take ferrotorch-convention masks and translate at the
# boundary so internal reference code mirrors numpy.ma directly.
# ---------------------------------------------------------------------------


def _to_ma(data_np: np.ndarray, valid_mask: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_array(data_np, mask=~valid_mask)


def ref_masked_sum(data_np: np.ndarray, valid_mask: np.ndarray) -> float:
    """Sum of valid entries. All-masked => 0 (matches torch.masked)."""
    if not valid_mask.any():
        return 0.0
    arr = _to_ma(data_np, valid_mask)
    return float(arr.sum())


def ref_masked_mean(data_np: np.ndarray, valid_mask: np.ndarray) -> float:
    """Mean of valid entries. All-masked => NaN (matches torch.masked)."""
    if not valid_mask.any():
        return float("nan")
    arr = _to_ma(data_np, valid_mask)
    return float(arr.mean())


def ref_masked_extreme(
    data_np: np.ndarray, valid_mask: np.ndarray, *, pick_min: bool
) -> float:
    """min/max of valid entries. All-masked => NaN (ferrotorch convention).

    TRACKED DIVERGENCE (#1924, split out of CORE-197 #1891): the all-masked
    and empty rows encode ferrotorch's pinned NaN sentinel, NOT a torch
    oracle value. Live torch 2.11.0 `MaskedTensor.amax/amin` on all-masked
    non-empty input return a fully-masked 0-d MaskedTensor whose payload is
    the reduction identity (-inf for amax, +inf for amin; the free functions
    `torch.masked.amax/amin` return the bare identity tensor), and on EMPTY
    input they RAISE
    `IndexError: amax(): Expected reduction dim 0 to have non-zero size.`
    (amin analogous). ferrotorch's `masked_min` / `masked_max` return a 0-d
    NaN tensor in both cases (see `masked_extremum_cpu` in src/masked.rs).

    The pinned fixture rows carry a ``divergence`` note probed from live
    torch at generation time (see `all_masked_extremum_note`), and the Rust
    suite asserts the NaN pin explicitly with the issue number — retire both
    when #1924 is fixed.
    """
    if not valid_mask.any():
        return float("nan")
    arr = _to_ma(data_np, valid_mask)
    return float(arr.min()) if pick_min else float(arr.max())


# Tracking issue for the all-masked extremum divergence pin (CORE-197).
DIVERGENCE_ISSUE: str = "#1924"

_ALL_MASKED_EXTREMUM_NOTE: str | None = None


def all_masked_extremum_note() -> str:
    """Probe live torch.masked for its all-masked / empty amax+amin contract.

    The resulting note is embedded in the pinned fixture rows (``divergence``
    field) so the divergence vs. ferrotorch's NaN sentinel is recorded next
    to the expectation instead of being laundered through a NaN-tolerant
    compare. Regenerating against a future torch refreshes the recorded
    upstream behavior automatically; the Rust suite hard-fails if the note
    drops the tracking-issue reference.
    """
    global _ALL_MASKED_EXTREMUM_NOTE
    if _ALL_MASKED_EXTREMUM_NOTE is not None:
        return _ALL_MASKED_EXTREMUM_NOTE

    import warnings

    from torch.masked import masked_tensor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch.masked prototype warnings
        all_masked = masked_tensor(
            torch.tensor([1.0, 2.0]), torch.tensor([False, False])
        )
        amax_payload = float(all_masked.amax().get_data())
        amin_payload = float(all_masked.amin().get_data())
        empty = masked_tensor(torch.tensor([]), torch.tensor([], dtype=torch.bool))
        try:
            empty.amax()
            empty_msg = (
                "no exception (torch behavior changed -- re-audit "
                f"{DIVERGENCE_ISSUE})"
            )
        except Exception as exc:  # noqa: BLE001 -- recording the raise verbatim
            empty_msg = f"{type(exc).__name__}: {exc}"

    _ALL_MASKED_EXTREMUM_NOTE = (
        f"TRACKED DIVERGENCE {DIVERGENCE_ISSUE}: expected value is ferrotorch's "
        "pinned NaN sentinel, NOT a torch oracle. "
        f"torch {torch.__version__} MaskedTensor.amax/.amin on all-masked "
        "non-empty input return a fully-masked 0-d MaskedTensor whose payload "
        f"is the reduction identity (amax: {amax_payload}, amin: "
        f"{amin_payload}); on EMPTY input they raise \"{empty_msg}\" (amin "
        "analogous). ferrotorch returns a 0-d NaN tensor in both cases. "
        f"Retire this pin and regenerate when {DIVERGENCE_ISSUE} is fixed."
    )
    return _ALL_MASKED_EXTREMUM_NOTE


def ref_masked_count(valid_mask: np.ndarray) -> float:
    """Number of valid entries, returned as a float scalar (0-d tensor)."""
    return float(int(valid_mask.sum()))


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
#
# Each builder is called once per (op, device, dtype) and emits one or more
# fixture rows. Keeping the rows shape-tagged makes failure messages tell us
# precisely which fixture diverged.


SHAPES: list[tuple[list[int], str]] = [
    ([5], "vec1d"),
    ([2, 3], "mat2d"),
    ([2, 2, 3], "ten3d"),
]


def _seeded_data(shape: list[int], dtype: str) -> np.ndarray:
    """Deterministic input. Avoids zeros to keep masked_equal's mask non-empty."""
    n = max(1, math.prod(shape))
    vals = [0.5 + i * 0.25 for i in range(n)]
    return np.asarray(vals, dtype=numpy_dtype(dtype)).reshape(shape)


def _alternating_mask(numel: int) -> np.ndarray:
    """Half-true / half-false (start with True). Length = numel."""
    out = np.zeros(numel, dtype=bool)
    out[::2] = True
    return out


def _all_true(numel: int) -> np.ndarray:
    return np.ones(numel, dtype=bool)


def _all_false(numel: int) -> np.ndarray:
    return np.zeros(numel, dtype=bool)


# --------- Operation fixtures ----------------------------------------------


def fixture_op(op: str) -> list[dict[str, Any]]:
    """Run one masked reduction across CPU/GPU x dtype x shape x mask-pattern.

    `op` ∈ {"masked_sum", "masked_mean", "masked_min", "masked_max",
            "masked_count"}.

    Mask patterns:
      * "all_true":  all entries valid → equivalent to the unmasked op.
      * "partial":   alternating mask → tests the masked compute path.
      * "all_false": no entries valid → sum=0, mean/min/max=NaN, count=0.
    """
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            for shape, shape_tag in SHAPES:
                data = _seeded_data(shape, dtype)
                numel = data.size
                for mask_kind in ("all_true", "partial", "all_false"):
                    if mask_kind == "all_true":
                        mask = _all_true(numel)
                    elif mask_kind == "partial":
                        mask = _alternating_mask(numel)
                    else:
                        mask = _all_false(numel)

                    if op == "masked_sum":
                        scalar = ref_masked_sum(data, mask)
                    elif op == "masked_mean":
                        scalar = ref_masked_mean(data, mask)
                    elif op == "masked_min":
                        scalar = ref_masked_extreme(data, mask, pick_min=True)
                    elif op == "masked_max":
                        scalar = ref_masked_extreme(data, mask, pick_min=False)
                    elif op == "masked_count":
                        scalar = ref_masked_count(mask)
                    else:
                        raise ValueError(op)

                    row: dict[str, Any] = {
                        "op": op,
                        "tag": f"{shape_tag}_{mask_kind}",
                        "dtype": dtype,
                        "device": device,
                        "a_shape": shape,
                        "a_data": to_listf(data.reshape(-1).tolist()),
                        "mask": [bool(b) for b in mask.tolist()],
                        "out_shape": [],
                        "out_values": to_listf([scalar]),
                    }
                    # All-masked min/max is a pinned divergence (#1924): the
                    # expected NaN is ferrotorch's contract, not torch's.
                    # Record the live-probed torch behavior on the row.
                    if mask_kind == "all_false" and op in (
                        "masked_min",
                        "masked_max",
                    ):
                        row["divergence"] = all_masked_extremum_note()
                    out.append(row)
    return out


# --------- Empty-tensor fixtures -------------------------------------------


def fixture_empty() -> list[dict[str, Any]]:
    """Empty 1-D masked tensor. PyTorch contract: sum=0, mean=NaN, count=0
    (ferrotorch matches these exactly). min/max=NaN is the pinned #1924
    divergence — torch.masked amax/amin RAISE IndexError on empty input;
    the rows carry the live-probed ``divergence`` note."""
    out: list[dict[str, Any]] = []
    for device in DEVICES:
        for dtype in DTYPES:
            empty_data = np.asarray([], dtype=numpy_dtype(dtype))
            empty_mask = np.asarray([], dtype=bool)
            cases = [
                ("masked_sum", ref_masked_sum(empty_data, empty_mask)),
                ("masked_mean", ref_masked_mean(empty_data, empty_mask)),
                ("masked_min", ref_masked_extreme(empty_data, empty_mask, pick_min=True)),
                ("masked_max", ref_masked_extreme(empty_data, empty_mask, pick_min=False)),
                ("masked_count", ref_masked_count(empty_mask)),
            ]
            for op_name, scalar in cases:
                row: dict[str, Any] = {
                    "op": f"{op_name}_empty",
                    "tag": "empty",
                    "dtype": dtype,
                    "device": device,
                    "a_shape": [0],
                    "a_data": [],
                    "mask": [],
                    "out_shape": [],
                    "out_values": to_listf([scalar]),
                }
                if op_name in ("masked_min", "masked_max"):
                    row["divergence"] = all_masked_extremum_note()
                out.append(row)
    return out


# --------- Constructor fixtures --------------------------------------------


def fixture_constructors() -> list[dict[str, Any]]:
    """Construction parity:

    * `from_data`: all entries valid, count_valid == numel, count_masked == 0.
    * `masked_where(data, condition)`: mask = !condition (numpy convention
      flipped to torch convention by the constructor).
    * `masked_invalid(data)`: NaN/+Inf/-Inf entries are masked OUT (i.e.,
      ferrotorch `mask=false`); finite entries stay valid.
    * `masked_equal(data, value)`: entries == value are masked OUT.
    * `with_fill_value(v).filled()`: substitutes `v` at masked positions.
    """
    out: list[dict[str, Any]] = []

    # Spread across both dtypes so f32 + f64 both exercise the path.
    for dtype in DTYPES:
        # ----- from_data: all-valid mask --------------------------------
        d = np.asarray([1.0, 2.0, 3.0], dtype=numpy_dtype(dtype))
        out.append(
            {
                "op": "from_data",
                "tag": "vec3",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": [3],
                "a_data": to_listf(d.tolist()),
                # expected mask after from_data: all true
                "expected_mask": [True, True, True],
                # expected count_valid / count_masked / numel
                "expected_count_valid": 3,
                "expected_count_masked": 0,
                "expected_numel": 3,
            }
        )

        # ----- masked_where: mask = !condition --------------------------
        d = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=numpy_dtype(dtype))
        condition = [False, True, False, True]
        # ferrotorch mask = NOT condition (torch convention: true = valid)
        expected_mask = [not c for c in condition]
        out.append(
            {
                "op": "masked_where",
                "tag": "vec4",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": [4],
                "a_data": to_listf(d.tolist()),
                "condition": condition,
                "expected_mask": expected_mask,
                "expected_count_valid": int(sum(expected_mask)),
                "expected_count_masked": int(len(expected_mask) - sum(expected_mask)),
                "expected_numel": 4,
            }
        )

        # ----- masked_invalid: finite=valid, NaN/Inf=masked -------------
        # NaN, +Inf and -Inf must all be masked OUT (ferrotorch
        # mask=false). Finite entries stay valid.
        d = np.asarray(
            [1.0, float("nan"), 3.0, float("inf"), -float("inf"), 5.0],
            dtype=numpy_dtype(dtype),
        )
        expected_mask = [True, False, True, False, False, True]
        out.append(
            {
                "op": "masked_invalid",
                "tag": "vec6",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": [6],
                "a_data": to_listf(d.tolist()),
                "expected_mask": expected_mask,
                "expected_count_valid": int(sum(expected_mask)),
                "expected_count_masked": int(len(expected_mask) - sum(expected_mask)),
                "expected_numel": 6,
            }
        )

        # ----- masked_equal: scalar match masks OUT ---------------------
        d = np.asarray([1.0, 5.0, 5.0, 2.0], dtype=numpy_dtype(dtype))
        target = 5.0
        # mask=true means VALID. So entries != 5 stay valid; == 5 are masked.
        expected_mask = [bool(v != target) for v in d.tolist()]
        out.append(
            {
                "op": "masked_equal",
                "tag": "vec4",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": [4],
                "a_data": to_listf(d.tolist()),
                "value": target,
                "expected_mask": expected_mask,
                "expected_count_valid": int(sum(expected_mask)),
                "expected_count_masked": int(len(expected_mask) - sum(expected_mask)),
                "expected_numel": 4,
            }
        )

        # ----- filled / with_fill_value ---------------------------------
        # Default fill_value is 0; with_fill_value(-99) overrides it.
        d = np.asarray([1.0, 2.0, 3.0], dtype=numpy_dtype(dtype))
        mask = [True, False, True]
        # Default fill (0) — masked entries become 0.
        filled_default = [v if m else 0.0 for v, m in zip(d.tolist(), mask)]
        # Override fill — masked entries become -99.
        filled_override = [v if m else -99.0 for v, m in zip(d.tolist(), mask)]
        out.append(
            {
                "op": "filled_default",
                "tag": "vec3",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": [3],
                "a_data": to_listf(d.tolist()),
                "mask": mask,
                "fill_value": 0.0,
                "out_shape": [3],
                "out_values": to_listf(filled_default),
            }
        )
        out.append(
            {
                "op": "filled_override",
                "tag": "vec3",
                "dtype": dtype,
                "device": "cpu",
                "a_shape": [3],
                "a_data": to_listf(d.tolist()),
                "mask": mask,
                "fill_value": -99.0,
                "out_shape": [3],
                "out_values": to_listf(filled_override),
            }
        )

    return out


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def main() -> int:
    fixtures: list[dict[str, Any]] = []
    # Reductions, including all-true / partial / all-false mask patterns.
    for op in ("masked_sum", "masked_mean", "masked_min", "masked_max", "masked_count"):
        fixtures += fixture_op(op)
    # Empty-tensor edge cases.
    fixtures += fixture_empty()
    # Constructor + filled() coverage (CPU only — masked_invalid/equal both
    # reject GPU input; from_data + masked_where are device-transparent but
    # exercising them on CPU keeps the fixture deterministic).
    fixtures += fixture_constructors()

    payload = {"metadata": fixture_metadata(), "fixtures": fixtures}
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"wrote {len(fixtures)} fixtures to {FIXTURE_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
