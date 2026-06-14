#!/usr/bin/env python3
"""
Regenerate PyTorch reference fixtures for ferrotorch-core Phase 2.9
(quantization + pruning).

Tracking issue: #771 (parent: #759). Oracle rewrite: CORE-194 -> #1888.

Output:
    ferrotorch-core/tests/conformance/fixtures/quantize_prune.json

ORACLE POLICY (goal-audit-fix.md R-ORACLE-2): every expected value in this
file is computed by a REAL PyTorch API. No function in this script
re-implements a ferrotorch algorithm. The torch APIs used, per op family:

* quantize_per_tensor / roundtrip:
    - qparams:  `torch.ao.quantization.observer.MinMaxObserver` with
      `qscheme=torch.per_tensor_affine` (`calculate_qparams()`),
      INT4 via `quant_min=-8, quant_max=7`.
    - codes/dequant (int8/uint8): `torch.quantize_per_tensor(...)`,
      `.int_repr()`, `.dequantize()`.
    - codes/dequant (int4): `torch.ops.quantized_decomposed.
      quantize_per_tensor` / `dequantize_per_tensor` (registered by
      `torch.ao.quantization.fx._decomposed`), which accept explicit
      quant_min/quant_max.
* quantize_per_channel:
    - qparams: `torch.ao.quantization.observer.PerChannelMinMaxObserver`
      with `qscheme=torch.per_channel_affine`.
    - codes/dequant: `torch.quantize_per_channel(...)` (int8/uint8) or
      `torch.ops.quantized_decomposed.quantize_per_channel` (int4).
* qparams_symmetric / qparams_asymmetric: `MinMaxObserver` with
  `qscheme=torch.per_tensor_symmetric` / `torch.per_tensor_affine`.
* fake_quantize_differentiable: `torch.fake_quantize_per_tensor_affine`
  forward + autograd backward (unchanged — this was already a real oracle).
* quantized_matmul: real-valued reference is `torch.matmul` (`a @ b`);
  the output (scale, zp) requantization is internal to ferrotorch, so the
  suite compares the dequantized output against the float matmul within a
  quantization-step bound.
* magnitude_prune: `torch.nn.utils.prune.l1_unstructured`
  (torch/nn/utils/prune.py, `L1Unstructured`), reading the module's
  pruned `weight` after the call.
* apply_2_4_mask: `torch.ao.pruning.WeightNormSparsifier(
  sparsity_level=1.0, sparse_block_shape=(1, 4), zeros_per_block=2)` —
  the documented PyTorch 2:4 semi-structured pruning path (see the
  PyTorch tutorial "Accelerating BERT with semi-structured (2:4)
  sparsity"). Shapes the sparsifier REJECTS (rows not a multiple of 4)
  are recorded with `torch_error` instead of a `masked` expectation.
* sparsity_ratio: `(t == 0).float().mean()` evaluated by torch.

Tie-magnitude cases are deliberately INCLUDED (they were previously
avoided): ties at the prune threshold (CORE-083 -> #1777), 2:4 in-group
magnitude ties, and a half-integer prune-count case. The fixture always
records the torch-side truth.

Usage:

    python3 scripts/regenerate_quantize_prune_fixtures.py

Required Python deps: torch (CUDA optional).
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
import torch.ao.quantization as taoq  # type: ignore

# Registers the `torch.ops.quantized_decomposed.*` ops used for INT4
# (explicit quant_min/quant_max) quantize/dequantize.
import torch.ao.quantization.fx._decomposed  # noqa: F401  # type: ignore

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
    / "quantize_prune.json"
)

RNG_SEED: int = 0x_771_BADCAFE
torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RNG_SEED)


def to_listf(t: torch.Tensor) -> list[Any]:
    """Materialize a tensor to a CPU Python list of floats with sentinels.

    Negative zero is preserved (JSON serializes it as `-0.0`), because
    torch's mask-multiply pruning genuinely produces `-0.0` for pruned
    negative elements and the suite asserts bit patterns.
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


def to_list_int(t: torch.Tensor) -> list[int]:
    return [int(v) for v in t.detach().to("cpu").reshape(-1).tolist()]


GENERATION_COMMAND = "python3 scripts/regenerate_quantize_prune_fixtures.py"


def fixture_metadata() -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "python_executable": sys.executable,
        "python_platform": platform.platform(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rng_seed": RNG_SEED,
        "phase": "2.9-quantize-prune",
        "tracking_issue": "#771; oracle rewrite CORE-194 -> #1888",
        "generation_command": GENERATION_COMMAND,
        "oracle_policy": "torch APIs only (goal-audit-fix.md R-ORACLE-2)",
    }


# ---------------------------------------------------------------------------
# Torch oracles — quantization
# ---------------------------------------------------------------------------

# ferrotorch QuantDtype -> (torch observer kwargs, torch quantized dtype or
# None when the `quantized_decomposed` explicit-range ops are required).
QDTYPE_TORCH = {
    "int8": (dict(dtype=torch.qint8), torch.qint8, (-128, 127)),
    "uint8": (dict(dtype=torch.quint8), torch.quint8, (0, 255)),
    # No native sub-byte affine qdtype with this range; use qint8 observers
    # with explicit quant_min/quant_max and the decomposed quantize ops.
    "int4": (dict(dtype=torch.qint8, quant_min=-8, quant_max=7), None, (-8, 7)),
}


def torch_qparams_per_tensor(x: torch.Tensor, qdtype: str) -> tuple[float, int]:
    """Oracle: `torch.ao.quantization.observer.MinMaxObserver` with
    `qscheme=torch.per_tensor_affine`; returns `calculate_qparams()`."""
    obs_kwargs, _, _ = QDTYPE_TORCH[qdtype]
    obs = taoq.MinMaxObserver(qscheme=torch.per_tensor_affine, **obs_kwargs)
    obs(x)
    scale, zp = obs.calculate_qparams()
    return float(scale.item()), int(zp.item())


def torch_quantize_per_tensor(
    x: torch.Tensor, scale: float, zp: int, qdtype: str
) -> tuple[list[int], list[Any]]:
    """Oracle: `torch.quantize_per_tensor(...).int_repr()` / `.dequantize()`
    for int8/uint8; `torch.ops.quantized_decomposed.quantize_per_tensor` /
    `dequantize_per_tensor` for int4 (explicit quant_min/quant_max).

    Returns (i32-domain integer codes, dequantized values)."""
    _, tdtype, (qmin, qmax) = QDTYPE_TORCH[qdtype]
    if tdtype is not None:
        q = torch.quantize_per_tensor(x, scale, zp, tdtype)
        return to_list_int(q.int_repr()), to_listf(q.dequantize())
    codes = torch.ops.quantized_decomposed.quantize_per_tensor(
        x, scale, zp, qmin, qmax, torch.int8
    )
    deq = torch.ops.quantized_decomposed.dequantize_per_tensor(
        codes, scale, zp, qmin, qmax, torch.int8
    )
    return to_list_int(codes), to_listf(deq)


def torch_qparams_per_channel(
    x: torch.Tensor, axis: int, qdtype: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Oracle: `torch.ao.quantization.observer.PerChannelMinMaxObserver`
    with `qscheme=torch.per_channel_affine`; returns `calculate_qparams()`."""
    obs_kwargs, _, _ = QDTYPE_TORCH[qdtype]
    obs = taoq.PerChannelMinMaxObserver(
        ch_axis=axis, qscheme=torch.per_channel_affine, **obs_kwargs
    )
    obs(x)
    return obs.calculate_qparams()


def torch_quantize_per_channel(
    x: torch.Tensor,
    scales: torch.Tensor,
    zps: torch.Tensor,
    axis: int,
    qdtype: str,
) -> tuple[list[int], list[Any]]:
    """Oracle: `torch.quantize_per_channel(...).int_repr()` / `.dequantize()`
    for int8/uint8; `torch.ops.quantized_decomposed.quantize_per_channel` /
    `dequantize_per_channel` for int4."""
    _, tdtype, (qmin, qmax) = QDTYPE_TORCH[qdtype]
    if tdtype is not None:
        q = torch.quantize_per_channel(x, scales, zps, axis, tdtype)
        return to_list_int(q.int_repr()), to_listf(q.dequantize())
    codes = torch.ops.quantized_decomposed.quantize_per_channel(
        x, scales, zps, axis, qmin, qmax, torch.int8
    )
    deq = torch.ops.quantized_decomposed.dequantize_per_channel(
        codes, scales, zps, axis, qmin, qmax, torch.int8
    )
    return to_list_int(codes), to_listf(deq)


# ---------------------------------------------------------------------------
# Torch oracles — pruning
# ---------------------------------------------------------------------------


def torch_l1_unstructured(
    data: list[float], shape: list[int], sparsity: float
) -> torch.Tensor:
    """Oracle: `torch.nn.utils.prune.l1_unstructured` (torch/nn/utils/
    prune.py `L1Unstructured`). The pruned tensor is the module's `weight`
    after the call (`weight_orig * mask`); note torch's mask-multiply
    yields `-0.0` for pruned negative weights, which is preserved here."""
    import torch.nn as nn
    import torch.nn.utils.prune as prune

    m = nn.Linear(1, 1)
    m.weight = nn.Parameter(torch.tensor(data, dtype=torch.float32).reshape(shape))
    prune.l1_unstructured(m, "weight", sparsity)
    return m.weight.detach()


def torch_2_4_mask(data: list[float], shape: list[int]) -> torch.Tensor:
    """Oracle: `torch.ao.pruning.WeightNormSparsifier(sparsity_level=1.0,
    sparse_block_shape=(1, 4), zeros_per_block=2)` — PyTorch's documented
    2:4 semi-structured pruning configuration (see the PyTorch tutorial
    "Accelerating BERT with semi-structured (2:4) sparsity").

    1-D inputs are presented to the sparsifier as a single row `[1, n]`
    (the 2:4 pattern groups along the innermost dimension, so this is the
    identity mapping for 1-D data). Raises whatever the sparsifier raises
    for shapes it rejects (rows not a multiple of 4 wide)."""
    import torch.nn as nn
    from torch.ao.pruning import WeightNormSparsifier

    t = torch.tensor(data, dtype=torch.float32).reshape(shape)
    t2d = t.reshape(1, -1) if t.dim() == 1 else t
    m = nn.Linear(t2d.shape[-1], t2d.shape[0])
    m.weight = nn.Parameter(t2d.clone())
    sparsifier = WeightNormSparsifier(
        sparsity_level=1.0, sparse_block_shape=(1, 4), zeros_per_block=2
    )
    sparsifier.prepare(m, [{"tensor_fqn": "weight"}])
    sparsifier.step()
    sparsifier.squash_mask()
    return m.weight.detach().reshape(shape)


def torch_zero_count(t: torch.Tensor) -> int:
    """Oracle: `(t == 0).sum()` evaluated by torch (counts ±0.0)."""
    return int((t == 0).sum().item())


# ---------------------------------------------------------------------------
# Quantize fixtures (per-tensor, per-channel; INT8/UINT8/INT4)
# ---------------------------------------------------------------------------


def fixture_quantize_per_tensor() -> list[dict[str, Any]]:
    """Per-tensor quantize: torch-observer qparams + torch quantize codes
    and dequantized round-trip."""
    out: list[dict[str, Any]] = []

    cases = [
        # (tag, qdtype, shape, data)
        ("int8_signed_range", "int8", [8], [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
        ("int8_pos_range", "int8", [6], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ("uint8_pos_range", "uint8", [6], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ("uint8_signed_range", "uint8", [8], [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]),
        ("int4_signed_range", "int4", [16], [float(v) for v in range(-8, 8)]),
        ("int4_pos_range", "int4", [4], [0.0, 1.0, 2.0, 3.0]),
        # Degenerate: all-equal — range still includes zero so it is finite.
        ("int8_constant", "int8", [4], [5.0, 5.0, 5.0, 5.0]),
        # All-zero: degenerate range; torch floors the SCALE at f32 eps
        # (`MinMaxObserver` -> `determine_qparams`), giving
        # scale = 1.1920929e-7.
        ("int8_all_zero", "int8", [4], [0.0, 0.0, 0.0, 0.0]),
        # Single element.
        ("int8_single", "int8", [1], [42.0]),
        # 2-D shape so we exercise the multi-dim path.
        ("int8_2d", "int8", [2, 3], [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
    ]

    for tag, qdtype, shape, data in cases:
        x = torch.tensor(data, dtype=torch.float32).reshape(shape)
        scale, zp = torch_qparams_per_tensor(x, qdtype)
        codes, dequant = torch_quantize_per_tensor(x, scale, zp, qdtype)
        out.append({
            "op": "quantize_per_tensor",
            "tag": tag,
            "qdtype": qdtype,
            "shape": shape,
            "x_data": to_listf(x),
            "scale": scale,
            "zero_point": zp,
            "codes": codes,
            "dequant": dequant,
            "oracle": "MinMaxObserver.calculate_qparams + torch.quantize_per_tensor"
            if qdtype != "int4"
            else "MinMaxObserver(quant_min=-8,quant_max=7) + "
            "torch.ops.quantized_decomposed.quantize_per_tensor",
        })
    return out


def fixture_quantize_per_channel() -> list[dict[str, Any]]:
    """Per-channel quantize: torch `PerChannelMinMaxObserver` qparams +
    `torch.quantize_per_channel` codes (decomposed ops for int4)."""
    out: list[dict[str, Any]] = []

    # Shape [3, 4]: 3 channels along axis 0, distinct ranges.
    data_3x4 = [
        0.0, 1.0, 2.0, 3.0,
        -10.0, -5.0, 5.0, 10.0,
        100.0, 130.0, 170.0, 200.0,
    ]
    cases = [
        ("int8_axis0", "int8", [3, 4], 0, data_3x4),
        ("uint8_axis0", "uint8", [3, 4], 0, data_3x4),
        ("int4_axis0", "int4", [3, 4], 0, data_3x4),
        # axis=1 path: shape [2, 3], 3 channels along last axis.
        ("int8_axis1", "int8", [2, 3], 1,
         [-1.0, 5.0, 10.0, -2.0, 4.0, 8.0]),
    ]

    for tag, qdtype, shape, axis, data in cases:
        x = torch.tensor(data, dtype=torch.float32).reshape(shape)
        scales_t, zps_t = torch_qparams_per_channel(x, axis, qdtype)
        codes, dequant = torch_quantize_per_channel(x, scales_t, zps_t, axis, qdtype)
        out.append({
            "op": "quantize_per_channel",
            "tag": tag,
            "qdtype": qdtype,
            "shape": shape,
            "axis": axis,
            "x_data": to_listf(x),
            "scales": [float(s) for s in scales_t],
            "zero_points": [int(z) for z in zps_t],
            "codes": codes,
            "dequant": dequant,
            "oracle": "PerChannelMinMaxObserver.calculate_qparams + "
            "torch.quantize_per_channel"
            if qdtype != "int4"
            else "PerChannelMinMaxObserver(quant_min=-8,quant_max=7) + "
            "torch.ops.quantized_decomposed.quantize_per_channel",
        })
    return out


# ---------------------------------------------------------------------------
# QParams symmetric / asymmetric reference values
# ---------------------------------------------------------------------------


def fixture_qparams() -> list[dict[str, Any]]:
    """Symmetric and asymmetric QParams reference values for boundary cases.

    Oracle: `torch.ao.quantization.observer.MinMaxObserver` with
    `qscheme=torch.per_tensor_symmetric` (fed `[-max_abs, max_abs]`) or
    `torch.per_tensor_affine` (fed `[min, max]`); `calculate_qparams()`.
    Note torch's symmetric scale is `max_abs / ((qmax - qmin) / 2)`
    (= /127.5 for int8/uint8, /7.5 for int4-range)."""
    out: list[dict[str, Any]] = []

    for max_abs in (5.0, 1.0, 100.0):
        for qdtype in ("int8", "uint8", "int4"):
            obs_kwargs, _, _ = QDTYPE_TORCH[qdtype]
            obs = taoq.MinMaxObserver(
                qscheme=torch.per_tensor_symmetric, **obs_kwargs
            )
            obs(torch.tensor([-max_abs, max_abs], dtype=torch.float32))
            scale_t, zp_t = obs.calculate_qparams()
            out.append({
                "op": "qparams_symmetric",
                "tag": f"{qdtype}_maxabs{max_abs}",
                "qdtype": qdtype,
                "max_abs": max_abs,
                "scale": float(scale_t.item()),
                "zero_point": int(zp_t.item()),
                "oracle": "MinMaxObserver(qscheme=per_tensor_symmetric)"
                ".calculate_qparams",
            })

    # Asymmetric covering boundary zp values.
    asym_cases = [
        # (tag, qdtype, min, max)
        ("int8_signed", "int8", -3.0, 3.0),     # zp ≈ 0
        ("int8_all_pos", "int8", 0.0, 1.0),     # zp = -128 boundary
        ("int8_all_neg", "int8", -1.0, 0.0),    # zp = 127 (range expanded to include 0)
        ("uint8_signed", "uint8", -1.0, 1.0),   # zp ≈ 128
        ("uint8_all_pos", "uint8", 0.0, 1.0),   # zp = 0 boundary
        ("int4_signed", "int4", -1.0, 1.0),     # zp ≈ 0
    ]
    for tag, qdtype, mn, mx in asym_cases:
        obs_kwargs, _, _ = QDTYPE_TORCH[qdtype]
        obs = taoq.MinMaxObserver(qscheme=torch.per_tensor_affine, **obs_kwargs)
        obs(torch.tensor([mn, mx], dtype=torch.float32))
        scale_t, zp_t = obs.calculate_qparams()
        out.append({
            "op": "qparams_asymmetric",
            "tag": tag,
            "qdtype": qdtype,
            "min_val": mn,
            "max_val": mx,
            "scale": float(scale_t.item()),
            "zero_point": int(zp_t.item()),
            "oracle": "MinMaxObserver(qscheme=per_tensor_affine)"
            ".calculate_qparams",
        })
    return out


# ---------------------------------------------------------------------------
# quantized_matmul fixtures
# ---------------------------------------------------------------------------


def fixture_quantized_matmul() -> list[dict[str, Any]]:
    """quantized_matmul on small INT8 inputs.

    Oracle: `torch.matmul` (`a @ b`) for the real-valued output. The
    suite compares ferrotorch's *dequantized* output to this float
    reference within a quantization-step tolerance — the output's
    scale/zp requantization is internal to ferrotorch and not part of
    PyTorch's surface API."""
    out: list[dict[str, Any]] = []

    cases = [
        # (tag, A_shape, B_shape, A_data, B_data)
        ("identity_2x2", [2, 2], [2, 2],
         [1.0, 2.0, 3.0, 4.0],
         [1.0, 0.0, 0.0, 1.0]),
        ("rect_2x3_3x2", [2, 3], [3, 2],
         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
         [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        ("rect_3x2_2x3", [3, 2], [2, 3],
         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ]

    for tag, a_shape, b_shape, a_data, b_data in cases:
        a = torch.tensor(a_data, dtype=torch.float32).reshape(a_shape)
        b = torch.tensor(b_data, dtype=torch.float32).reshape(b_shape)
        c = a @ b
        out.append({
            "op": "quantized_matmul",
            "tag": tag,
            "qdtype": "int8",
            "a_shape": a_shape,
            "b_shape": b_shape,
            "a_data": to_listf(a),
            "b_data": to_listf(b),
            "c_shape": list(c.shape),
            "c_data": to_listf(c),
            "oracle": "torch.matmul",
        })

    # CORE-201 -> #1895: a k=64 case so the suite's analytic matmul error
    # bound (which scales linearly with k) is exercised on a long i32
    # accumulation, not just the 2x2/3x3 toys above. Small magnitudes
    # (|x| <= 0.25) keep the quantization step — and therefore the bound —
    # tight (~3e-2 absolute), so the case genuinely discriminates sub-0.5
    # errors (the old unjustified tolerance floor). A local torch.Generator
    # makes the data independent of module-level RNG consumption order.
    g = torch.Generator().manual_seed(RNG_SEED)
    a64 = (torch.rand(4, 64, generator=g, dtype=torch.float32) - 0.5) * 0.5
    b64 = (torch.rand(64, 3, generator=g, dtype=torch.float32) - 0.5) * 0.5
    c64 = a64 @ b64
    out.append({
        "op": "quantized_matmul",
        "tag": "long_k64_4x64_64x3",
        "qdtype": "int8",
        "a_shape": [4, 64],
        "b_shape": [64, 3],
        "a_data": to_listf(a64),
        "b_data": to_listf(b64),
        "c_shape": list(c64.shape),
        "c_data": to_listf(c64),
        "oracle": "torch.matmul",
    })
    return out


# ---------------------------------------------------------------------------
# fake_quantize_differentiable fixtures (CPU; PyTorch parity).
# ---------------------------------------------------------------------------


def fixture_fake_quantize() -> list[dict[str, Any]]:
    """fake_quantize_differentiable: forward + STE backward.

    Oracle: `torch.fake_quantize_per_tensor_affine` forward and its
    autograd backward (`y.sum().backward()`), which zeroes the gradient
    for out-of-range inputs (STE)."""
    out: list[dict[str, Any]] = []

    cases = [
        # (tag, x_data, scale, zp, qmin, qmax)
        ("int8_in_range", [-1.0, 0.0, 0.5, 1.0], 0.05, 0, -128, 127),
        ("int8_out_of_range", [-200.0, -1.0, 0.0, 1.0, 200.0], 0.05, 0, -128, 127),
        ("uint8_zp128", [-1.0, 0.0, 0.5, 1.0], 0.01, 128, 0, 255),
        ("int4_small_range", [-2.0, -0.5, 0.5, 2.0], 0.5, 0, -8, 7),
    ]
    for tag, x_data, scale, zp, qmin, qmax in cases:
        x = torch.tensor(x_data, dtype=torch.float32, requires_grad=True)
        y = torch.fake_quantize_per_tensor_affine(x, scale, zp, qmin, qmax)
        y.sum().backward()
        out.append({
            "op": "fake_quantize_differentiable",
            "tag": tag,
            "x_data": to_listf(x),
            "scale": scale,
            "zero_point": zp,
            "qmin": qmin,
            "qmax": qmax,
            "y_data": to_listf(y),
            "grad_x": to_listf(x.grad),
            "oracle": "torch.fake_quantize_per_tensor_affine + autograd",
        })
    return out


# ---------------------------------------------------------------------------
# Pruning fixtures (CPU; bit-exact mask × original).
# ---------------------------------------------------------------------------


def fixture_magnitude_prune() -> list[dict[str, Any]]:
    """Oracle: `torch.nn.utils.prune.l1_unstructured` (see
    `torch_l1_unstructured`). Tie-magnitude cases are included on purpose
    to guard the torch topk selection order and prune-count rounding."""
    out: list[dict[str, Any]] = []
    cases = [
        # (tag, sparsity, shape, data)
        ("vec_50pct", 0.5, [4], [1.0, -4.0, 2.0, -3.0]),
        ("vec_25pct", 0.25, [8], [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]),
        ("vec_75pct", 0.75, [8], [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]),
        ("zero_sparsity", 0.0, [4], [1.0, 2.0, 3.0, 4.0]),
        # Boundary: 90% — should keep at least one element.
        ("vec_high_sparsity", 0.9, [10], [float(i + 1) for i in range(10)]),
        # 2-D tensor preserves shape.
        ("mat2d_50pct", 0.5, [2, 4], [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]),
        # --- tie-magnitude cases (CORE-083 -> #1777 territory) ---
        # All four magnitudes equal; torch prunes exactly 2 via topk.
        ("tie_all_equal", 0.5, [4], [1.0, 1.0, 1.0, 1.0]),
        # Tie AT the threshold: |2.0| appears twice, only one may go.
        ("tie_threshold_partial", 0.5, [4], [1.0, 2.0, 2.0, 3.0]),
        # Four-way tie at the threshold, three to prune.
        ("tie_multiway", 0.5, [6], [2.0, -2.0, 2.0, 5.0, -2.0, 7.0]),
        # Half-integer prune count: 0.125 * 4 = 0.5; torch (Python
        # round-half-to-even) prunes 0.
        ("count_round_half", 0.125, [4], [1.0, 2.0, 3.0, 4.0]),
    ]
    for tag, sparsity, shape, data in cases:
        pruned = torch_l1_unstructured(data, shape, sparsity)
        out.append({
            "op": "magnitude_prune",
            "tag": tag,
            "sparsity": sparsity,
            "shape": shape,
            "x_data": data,
            "pruned": to_listf(pruned),
            "n_zeros": torch_zero_count(pruned),
            "oracle": "torch.nn.utils.prune.l1_unstructured",
        })
    return out


def fixture_apply_2_4_mask() -> list[dict[str, Any]]:
    """Oracle: `torch.ao.pruning.WeightNormSparsifier` 2:4 configuration
    (see `torch_2_4_mask`). Shapes the sparsifier rejects (rows not a
    multiple of 4 wide) carry `torch_error` instead of `masked`.
    In-group tie-magnitude cases are included on purpose."""
    out: list[dict[str, Any]] = []
    cases = [
        # (tag, shape, data)
        ("group1", [4], [1.0, -4.0, 2.0, -3.0]),
        ("group2", [8], [1.0, -4.0, 2.0, -3.0, 0.5, 0.1, 0.9, 0.8]),
        # Trailing < 4 elements: torch's 2:4 sparsifier REJECTS this shape.
        ("trailing", [6], [1.0, -4.0, 2.0, -3.0, 0.5, 0.1]),
        # 2-D shape, 8 elements (preserves shape).
        ("mat2d", [2, 4], [1.0, -4.0, 2.0, -3.0, 0.5, 0.1, 0.9, 0.8]),
        # Rows 6 wide: torch's 2:4 sparsifier REJECTS this shape.
        ("rows_cross_2x6", [2, 6],
         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
        # --- in-group tie-magnitude cases ---
        ("tie_all_equal", [4], [2.0, 2.0, 2.0, 2.0]),
        ("tie_three_equal", [4], [1.0, 3.0, 3.0, 3.0]),
        ("tie_neg_pair", [4], [-2.0, 2.0, -2.0, 2.0]),
    ]
    for tag, shape, data in cases:
        row: dict[str, Any] = {
            "op": "apply_2_4_mask",
            "tag": tag,
            "shape": shape,
            "x_data": data,
            "oracle": "torch.ao.pruning.WeightNormSparsifier("
            "sparsity_level=1.0, sparse_block_shape=(1,4), zeros_per_block=2)",
        }
        try:
            masked = torch_2_4_mask(data, shape)
        except Exception as exc:  # sparsifier rejects the shape
            row["torch_error"] = f"{type(exc).__name__}: {exc}"
        else:
            row["masked"] = to_listf(masked)
            row["n_zeros"] = torch_zero_count(masked)
        out.append(row)
    return out


def fixture_sparsity_ratio() -> list[dict[str, Any]]:
    """Oracle: `(t == 0).float().mean()` evaluated by torch."""
    cases = [
        ("half", [4], [0.0, 1.0, 0.0, 2.0]),
        ("none", [4], [1.0, 2.0, 3.0, 4.0]),
        ("all", [4], [0.0, 0.0, 0.0, 0.0]),
        ("75pct", [4], [0.0, 0.0, 0.0, 1.0]),
    ]
    out = []
    for tag, shape, data in cases:
        t = torch.tensor(data, dtype=torch.float32).reshape(shape)
        ratio = float((t == 0).to(torch.float64).mean().item())
        out.append({
            "op": "sparsity_ratio",
            "tag": tag,
            "shape": shape,
            "x_data": data,
            "ratio": ratio,
            "oracle": "(t == 0).float().mean()",
        })
    return out


# ---------------------------------------------------------------------------
# Round-trip parity (dequantize(quantize(x)) ≈ x within step)
# ---------------------------------------------------------------------------


def fixture_roundtrip() -> list[dict[str, Any]]:
    """Round-trip dequant∘quant: error <= scale (one quantization step).

    Oracle: torch observer qparams + `torch.quantize_per_tensor(...)
    .dequantize()` (decomposed ops for int4)."""
    out: list[dict[str, Any]] = []

    cases = [
        ("rt_int8", "int8", [11], [-5.0 + 0.5 * i for i in range(11)]),
        ("rt_uint8", "uint8", [11], [0.0 + 0.2 * i for i in range(11)]),
        ("rt_int4", "int4", [16], [float(v) for v in range(-8, 8)]),
    ]
    for tag, qdtype, shape, data in cases:
        x = torch.tensor(data, dtype=torch.float32).reshape(shape)
        scale, zp = torch_qparams_per_tensor(x, qdtype)
        _codes, recovered = torch_quantize_per_tensor(x, scale, zp, qdtype)
        out.append({
            "op": "roundtrip",
            "tag": tag,
            "qdtype": qdtype,
            "shape": shape,
            "x_data": to_listf(x),
            "scale": scale,
            "zero_point": zp,
            "recovered": recovered,
            "oracle": "MinMaxObserver.calculate_qparams + "
            "torch.quantize_per_tensor(...).dequantize()",
        })
    return out


# ---------------------------------------------------------------------------
# Build & write
# ---------------------------------------------------------------------------


def all_fixtures() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out.extend(fixture_quantize_per_tensor())
    out.extend(fixture_quantize_per_channel())
    out.extend(fixture_qparams())
    out.extend(fixture_quantized_matmul())
    out.extend(fixture_fake_quantize())
    out.extend(fixture_magnitude_prune())
    out.extend(fixture_apply_2_4_mask())
    out.extend(fixture_sparsity_ratio())
    out.extend(fixture_roundtrip())
    return out


def main() -> None:
    payload = {
        "metadata": fixture_metadata(),
        "fixtures": all_fixtures(),
    }
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {FIXTURE_PATH} ({len(payload['fixtures'])} fixtures)")


if __name__ == "__main__":
    main()
