#!/usr/bin/env python3
"""
Persistent torch oracle for the ferrotorch parity-sweep harness.

Speaks newline-delimited JSON over stdio. One request per line, one response per
line. Tensors are passed as raw little-endian bytes, base64-encoded, with a tiny
JSON header (shape + dtype). No precision loss across the IPC boundary.

Protocol (request -> response):

  {"cmd":"ready"}                            -> {"ok":true,"torch":"2.5.0","ops":N}
  {"cmd":"list_ops"}                         -> {"ok":true,"ops":["add","mul",...]}
  {"cmd":"sample","op":"add","seed":42,"i":0}
      -> {"ok":true,"args":[<tensor>...],"kwargs":{...},"output":<tensor>}
  {"cmd":"execute","op":"add","args":[...],"kwargs":{...}}
      -> {"ok":true,"output":<tensor>}
  {"cmd":"shutdown"}                         -> {"ok":true}

  Tensor = {"shape":[...], "dtype":"float32|float64|float16|bfloat16|int32|int64|bool",
            "data_b64":"<base64 of raw native-endian bytes>"}

  Errors: {"ok":false,"err":"<message>"}

Designed for one persistent child per runner. Each `sample` and `execute` resets
torch RNG + clears autograd state so calls do not pollute each other.
"""

from __future__ import annotations

import base64
import json
import sys
import traceback
from typing import Any

import torch
from torch.testing._internal.common_methods_invocations import op_db


# ---------------------------------------------------------------------------
# Tensor <-> wire format
# ---------------------------------------------------------------------------

_DTYPE_TO_NAME = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_NAME_TO_DTYPE = {v: k for k, v in _DTYPE_TO_NAME.items()}


def tensor_to_wire(t: torch.Tensor) -> dict[str, Any]:
    """Serialize a torch.Tensor to the wire format."""
    if t.dtype not in _DTYPE_TO_NAME:
        raise ValueError(f"unsupported dtype: {t.dtype}")
    # bf16 has no numpy equivalent — view as i16 to extract raw bytes.
    if t.dtype == torch.bfloat16:
        raw = t.contiguous().view(torch.int16).cpu().numpy().tobytes()
    else:
        raw = t.contiguous().cpu().numpy().tobytes()
    return {
        "shape": list(t.shape),
        "dtype": _DTYPE_TO_NAME[t.dtype],
        "data_b64": base64.b64encode(raw).decode("ascii"),
    }


def wire_to_tensor(w: dict[str, Any]) -> torch.Tensor:
    """Deserialize a wire-format tensor back to torch.Tensor."""
    import numpy as np
    dtype = _NAME_TO_DTYPE[w["dtype"]]
    raw = base64.b64decode(w["data_b64"])
    if dtype == torch.bfloat16:
        arr = np.frombuffer(raw, dtype=np.int16).reshape(w["shape"]).copy()
        return torch.from_numpy(arr).view(torch.bfloat16)
    np_dtype = {
        "float32": np.float32, "float64": np.float64, "float16": np.float16,
        "int8": np.int8, "int16": np.int16, "int32": np.int32, "int64": np.int64,
        "uint8": np.uint8, "bool": np.bool_,
    }[w["dtype"]]
    arr = np.frombuffer(raw, dtype=np_dtype).reshape(w["shape"]).copy()
    return torch.from_numpy(arr)


def encode_arg(a: Any) -> Any:
    """Encode an arg/kwarg value for the wire. Tensors get the tensor envelope;
    everything else is passed as plain JSON (we only allow JSON-safe scalars)."""
    if isinstance(a, torch.Tensor):
        return {"__tensor__": tensor_to_wire(a)}
    if isinstance(a, (list, tuple)):
        return [encode_arg(x) for x in a]
    if isinstance(a, dict):
        return {k: encode_arg(v) for k, v in a.items()}
    if isinstance(a, (int, float, bool, str)) or a is None:
        return a
    # torch.dtype, torch.device, torch.memory_format, etc — encode by name so
    # they round-trip readably. The runner only needs to skip them for now;
    # the discriminator/re-corrector will care once dispatch covers ops that
    # actually accept these args.
    if isinstance(a, torch.dtype):
        return {"__dtype__": _DTYPE_TO_NAME.get(a, str(a))}
    if isinstance(a, torch.memory_format):
        return {"__memory_format__": str(a).rsplit(".", 1)[-1]}
    if isinstance(a, torch.device):
        return {"__device__": str(a)}
    if isinstance(a, torch.layout):
        return {"__layout__": str(a).rsplit(".", 1)[-1]}
    raise ValueError(f"cannot encode arg of type {type(a).__name__}: {a!r}")


def decode_arg(a: Any) -> Any:
    if isinstance(a, dict):
        if "__tensor__" in a:
            return wire_to_tensor(a["__tensor__"])
        if "__dtype__" in a:
            return _NAME_TO_DTYPE[a["__dtype__"]]
        if "__memory_format__" in a:
            return getattr(torch, a["__memory_format__"])
        if "__device__" in a:
            return torch.device(a["__device__"])
        if "__layout__" in a:
            return getattr(torch, a["__layout__"])
        return {k: decode_arg(v) for k, v in a.items()}
    if isinstance(a, list):
        return [decode_arg(x) for x in a]
    return a


# ---------------------------------------------------------------------------
# op_db index
# ---------------------------------------------------------------------------

# Build a name -> OpInfo map. Some ops appear multiple times (variants); take
# the first registration (the canonical aten op) for the sweep entrypoint.
_OPS: dict[str, Any] = {}
for op_info in op_db:
    name = op_info.name
    if name not in _OPS:
        _OPS[name] = op_info


# ---------------------------------------------------------------------------
# Custom-op registry
# ---------------------------------------------------------------------------
#
# Some torch ops we want parity-sweep coverage for are NOT in
# `torch.testing._internal.common_methods_invocations.op_db` because op_db is
# the unit-test op set (quantization ops live under a separate
# `torch.testing._internal.quantization` harness — see
# `aten/src/ATen/native/quantized/FakeQuantPerTensorAffine.cpp:31-51` for the
# user-facing entry points and `torch/_torch_docs.py` for their documented
# signatures). For each such op we hand-craft sample inputs covering shape /
# dtype / scale / zero_point / quant_min / quant_max variations, then invoke
# the top-level `torch.<op_name>` callable directly.
#
# A registry entry is a dict with:
#   - "callable": the torch function (e.g. `torch.fake_quantize_per_tensor_affine`)
#   - "samples":  list[(args_tuple, kwargs_dict)] — handed to `callable` to
#                 produce the expected output.
#
# Custom ops are listed alongside op_db ops in `list_ops()` so the parity-sweep
# runner can discover them. They take precedence over op_db lookups when names
# collide (defensive — no current collisions).

def _fake_quantize_per_tensor_affine_samples() -> list[tuple[tuple, dict]]:
    """Hand-crafted samples for `torch.fake_quantize_per_tensor_affine`.

    Covers:
      - 1-D / 2-D / 3-D input shapes
      - scale: tiny (0.01), unit (1.0), large (100.0), default-ish (0.1)
      - zero_point: 0 (symmetric int8), 127 / -128 (asymmetric edges),
        64 (mid-range), 128 (uint8 mid)
      - quant_min / quant_max: -128/127 (int8 signed), 0/255 (uint8)
      - Edge inputs: zeros, exact-multiple-of-scale values that round-trip,
        out-of-range values that clamp.

    Each sample is `((input_tensor, scale, zp, qmin, qmax), {})`.
    """
    # Reuse a small RNG for determinism.
    g = torch.Generator().manual_seed(0)
    samples: list[tuple[tuple, dict]] = []

    # Sample 0: 1-D, int8 symmetric, scale=0.1, random-ish small floats.
    s0 = torch.randn(8, generator=g, dtype=torch.float32) * 0.5
    samples.append(((s0, 0.1, 0, -128, 127), {}))

    # Sample 1: 1-D, int8 symmetric, scale=1.0, includes out-of-range to test
    # clamp at both ends.
    s1 = torch.tensor(
        [-200.0, -128.0, -1.0, 0.0, 1.0, 127.0, 200.0],
        dtype=torch.float32,
    )
    samples.append(((s1, 1.0, 0, -128, 127), {}))

    # Sample 2: 2-D, uint8 asymmetric, scale=0.01, zp=128 (mid uint8).
    s2 = torch.randn(3, 4, generator=g, dtype=torch.float32) * 0.5
    samples.append(((s2, 0.01, 128, 0, 255), {}))

    # Sample 3: 2-D, uint8 zp=0 (asymmetric at lower edge), scale=100.0
    # (coarse quantization — exercises round behavior on integer-valued
    # inputs).
    s3 = torch.tensor(
        [[0.0, 100.0, 200.0, 300.0], [-50.0, 50.0, 150.0, 25500.0]],
        dtype=torch.float32,
    )
    samples.append(((s3, 100.0, 0, 0, 255), {}))

    # Sample 4: 3-D, int8 with non-zero zero_point, scale=0.05.
    s4 = torch.randn(2, 3, 4, generator=g, dtype=torch.float32) * 0.3
    samples.append(((s4, 0.05, 64, -128, 127), {}))

    # Sample 5: scalar-like (1-element 1-D), tests degenerate-size path.
    s5 = torch.tensor([3.14159], dtype=torch.float32)
    samples.append(((s5, 0.01, 0, -128, 127), {}))

    # Sample 6: includes +Inf and -Inf — torch's behavior is to clamp Inf to
    # the (dequantized) quant_max / quant_min boundary via the round-then-
    # clamp path. NaN coverage moved to samples 9/10 below — the prior
    # "NaN excluded because f32::clamp debug-asserts" rationale went stale
    # with commits 77781d844 (#1238: f64 min/max NaN-safe semantics) and
    # 0258ffb0c (#1259: kept the NaN-safe property when reverting f64 ->
    # f32 arithmetic to match upstream byte-for-byte). The unit test
    # `fake_quantize_nan_input_does_not_panic` at
    # `ferrotorch-core/src/grad_fns/quantize_grad.rs:850-862` locks the
    # NaN contract as "no panic, finite output (clamps to quant_min)".
    s6 = torch.tensor(
        [float("inf"), -float("inf"), 0.0, 1.0],
        dtype=torch.float32,
    )
    samples.append(((s6, 1.0, 0, -128, 127), {}))

    # Sample 7: zeros (verifies zp=0 special-case and zero-times-anything).
    s7 = torch.zeros(5, dtype=torch.float32)
    samples.append(((s7, 0.1, 0, -128, 127), {}))

    # Sample 8: int8 with zp at upper bound (zp=127), exercises the
    # `zero_point >= quant_min && zero_point <= quant_max` upstream check
    # at FakeQuantPerTensorAffine.cpp:79-81.
    s8 = torch.randn(6, generator=g, dtype=torch.float32) * 0.5
    samples.append(((s8, 0.1, 127, -128, 127), {}))

    # Sample 9: NaN propagation through the per-tensor kernel.
    # Live torch returns `tensor([-128., 1., 2.])` because NaN propagates
    # through `inv_scale * x` then `nearbyint(NaN) == NaN`, then upstream's
    # `std::fmin(std::fmax(NaN, quant_min), quant_max)` collapses NaN to
    # `quant_min` (IEEE-754-2019 minimum/maximum semantics: NaN argument
    # is dropped, the finite operand wins; per upstream
    # `aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp:
    # 2683-2684`). ferrotorch's NaN-safe `f32::min` / `f32::max` chain at
    # `grad_fns/quantize_grad.rs:262-263` reproduces this. The unit test
    # `fake_quantize_nan_input_does_not_panic` only checks "no panic +
    # finite output" — the parity sweep now also checks the value
    # equals torch's `-128.0` byte-for-byte.
    s9 = torch.tensor([float("nan"), 1.0, 2.0], dtype=torch.float32)
    samples.append(((s9, 1.0, 0, -128, 127), {}))

    # Sample 10: combined non-finite coverage — NaN alongside +Inf / -Inf.
    # Expected per live torch:
    #   `tensor([-128., 127., -128.])` (NaN -> quant_min, +Inf -> quant_max,
    #   -Inf -> quant_min). Mirrors the audit probe at
    #   `tools/parity-sweep/runs/fake_quantize_per_tensor_affine/
    #   discriminator_audit_probes_post_1259.jsonl` id
    #   `nan_input_now_safe_per_1259`.
    s10 = torch.tensor(
        [float("nan"), float("inf"), -float("inf")], dtype=torch.float32
    )
    samples.append(((s10, 1.0, 0, -128, 127), {}))

    return samples


def _fake_quantize_per_channel_affine_samples() -> list[tuple[tuple, dict]]:
    """Hand-crafted samples for `torch.fake_quantize_per_channel_affine`.

    Covers:
      - 2-D / 3-D / 4-D input shapes (with the channel axis on each)
      - per-channel scale / zero_point 1-D tensors matching `input.size(axis)`
      - axis on first, middle, last dimension
      - int8 (-128/127) and uint8 (0/255) ranges
      - varied per-channel scales and zero_points

    Each sample is `((input, scale_1d, zp_1d, axis, qmin, qmax), {})` where
    `axis`, `qmin`, `qmax` are passed positionally to match the upstream
    signature `(input, scale, zero_point, axis, quant_min, quant_max)`
    from `aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp:32-42`.
    """
    g = torch.Generator().manual_seed(0)
    samples: list[tuple[tuple, dict]] = []

    # Sample 0: 2-D, axis=1 (channel-last for 2-D), int8 symmetric.
    s0 = torch.randn(3, 4, generator=g, dtype=torch.float32) * 0.5
    scale0 = torch.tensor([0.1, 0.05, 0.2, 0.01], dtype=torch.float32)
    zp0 = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
    samples.append(((s0, scale0, zp0, 1, -128, 127), {}))

    # Sample 1: 2-D, axis=0, uint8 asymmetric with non-zero zero_points.
    s1 = torch.randn(3, 4, generator=g, dtype=torch.float32) * 0.5
    scale1 = torch.tensor([0.01, 0.02, 0.05], dtype=torch.float32)
    zp1 = torch.tensor([128, 64, 200], dtype=torch.int32)
    samples.append(((s1, scale1, zp1, 0, 0, 255), {}))

    # Sample 2: 3-D, axis=0, int8 with varied zps.
    s2 = torch.randn(2, 3, 4, generator=g, dtype=torch.float32) * 0.3
    scale2 = torch.tensor([0.1, 0.05], dtype=torch.float32)
    zp2 = torch.tensor([0, 64], dtype=torch.int32)
    samples.append(((s2, scale2, zp2, 0, -128, 127), {}))

    # Sample 3: 3-D, axis=2 (last dim), int8.
    s3 = torch.randn(2, 3, 4, generator=g, dtype=torch.float32) * 0.5
    scale3 = torch.tensor([0.1, 0.05, 0.2, 0.01], dtype=torch.float32)
    zp3 = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
    samples.append(((s3, scale3, zp3, 2, -128, 127), {}))

    # Sample 4: 4-D, axis=1 (channel-first conv-weight layout), int8.
    # Typical Conv2d weight layout [out_channels, in_channels, H, W];
    # per-channel quantization on out_channels.
    s4 = torch.randn(4, 2, 3, 3, generator=g, dtype=torch.float32) * 0.3
    scale4 = torch.tensor([0.05, 0.1, 0.02, 0.2], dtype=torch.float32)
    zp4 = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
    samples.append(((s4, scale4, zp4, 0, -128, 127), {}))

    # Sample 5: includes out-of-range values that clamp on a per-channel
    # basis (each channel has a different effective dequantized range).
    s5 = torch.tensor(
        [[-200.0, 200.0, 0.0, 1.0], [10.0, -10.0, 1000.0, -1000.0]],
        dtype=torch.float32,
    )
    scale5 = torch.tensor([1.0, 10.0], dtype=torch.float32)
    zp5 = torch.tensor([0, 0], dtype=torch.int32)
    samples.append(((s5, scale5, zp5, 0, -128, 127), {}))

    # Sample 6: includes +Inf / -Inf to exercise clamp on per-channel.
    s6 = torch.tensor(
        [[float("inf"), -float("inf"), 0.0, 1.0]],
        dtype=torch.float32,
    )
    scale6 = torch.tensor([1.0], dtype=torch.float32)
    zp6 = torch.tensor([0], dtype=torch.int32)
    samples.append(((s6, scale6, zp6, 0, -128, 127), {}))

    # Sample 7: 2-D, axis=1, scale at extremes (very small and very large).
    s7 = torch.randn(2, 3, generator=g, dtype=torch.float32)
    scale7 = torch.tensor([0.001, 1.0, 100.0], dtype=torch.float32)
    zp7 = torch.tensor([0, 0, 0], dtype=torch.int32)
    samples.append(((s7, scale7, zp7, 1, -128, 127), {}))

    # Sample 8: zeros input — verifies degenerate-input path on per-channel.
    # All zps must lie in [quant_min, quant_max] per upstream's
    # FakeQuantPerTensorAffine.cpp:79-81 check (`zero_point >= quant_min
    # && zero_point <= quant_max`), which the per-channel path inherits.
    s8 = torch.zeros(2, 3, dtype=torch.float32)
    scale8 = torch.tensor([0.1, 0.2, 0.05], dtype=torch.float32)
    zp8 = torch.tensor([-128, 0, 127], dtype=torch.int32)
    samples.append(((s8, scale8, zp8, 1, -128, 127), {}))

    # Sample 9: NaN propagation through the per-channel kernel.
    # The per-channel kernel uses the cast-to-int64-first ordering from
    # upstream `aten/src/ATen/native/quantized/cpu/kernels/
    # QuantizedOpKernels.cpp:2836-2848`, where `static_cast<int64_t>(NaN)`
    # on x86-64 via `cvttsd2si` snaps to INT64_MIN (invalid-op result)
    # which then `std::fmax(INT64_MIN, quant_min)` clamps to `quant_min`.
    # ferrotorch replicates this with an explicit non-finite check at
    # `grad_fns/quantize_grad.rs:319,346-350` (NaN, +Inf, -Inf all
    # resolve to INT64_MIN before clamping). Live torch: input
    # `[[NaN, 1.0, -1.0]]`, scale=[0.1], zp=[0], axis=0, qmin=-128,
    # qmax=127 yields `[[-12.8, 1.0, -1.0]]` — the NaN slot clamps to
    # `quant_min * scale = -128 * 0.1 = -12.8` (per-channel dequant
    # multiplies by the per-channel scale; per-tensor's NaN-> -128 was
    # scale=1.0). Mirrors audit probe `per_channel_nan_input` at
    # `tools/parity-sweep/runs/fake_quantize_per_tensor_affine/
    # discriminator_audit_probes_post_1259.jsonl`.
    s9 = torch.tensor([[float("nan"), 1.0, -1.0]], dtype=torch.float32)
    scale9 = torch.tensor([0.1], dtype=torch.float32)
    zp9 = torch.tensor([0], dtype=torch.int32)
    samples.append(((s9, scale9, zp9, 0, -128, 127), {}))

    return samples


_CUSTOM_OPS: dict[str, dict[str, Any]] = {
    "fake_quantize_per_tensor_affine": {
        "callable": torch.fake_quantize_per_tensor_affine,
        "samples": _fake_quantize_per_tensor_affine_samples(),
    },
    "fake_quantize_per_channel_affine": {
        "callable": torch.fake_quantize_per_channel_affine,
        "samples": _fake_quantize_per_channel_affine_samples(),
    },
}


def list_ops() -> list[str]:
    return sorted(set(_OPS.keys()) | set(_CUSTOM_OPS.keys()))


def _reset_torch_state(seed: int) -> None:
    """Reset torch global state between calls so they don't pollute each other."""
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    if torch.is_autocast_enabled():
        torch.clear_autocast_cache()
    # default dtype is float32 unless user changed it; force it back.
    torch.set_default_dtype(torch.float32)


def sample(op_name: str, seed: int, index: int) -> dict[str, Any]:
    """Generate sample input #index for op `op_name`, then execute torch on
    it so the response carries (args, kwargs, expected_output).

    For ops in op_db, samples come from `OpInfo.sample_inputs`. For ops in
    `_CUSTOM_OPS` (ops not in op_db — see registry definition above), samples
    come from the hand-crafted list and the op is invoked via the registered
    top-level torch callable.
    """
    # Custom-op path takes precedence (no current collisions, but defensive).
    if op_name in _CUSTOM_OPS:
        entry = _CUSTOM_OPS[op_name]
        _reset_torch_state(seed)
        samples = entry["samples"]
        if index >= len(samples):
            return {
                "ok": False,
                "err": f"index {index} >= {len(samples)} samples for {op_name}",
            }
        args, kwargs = samples[index]
        try:
            output = entry["callable"](*args, **kwargs)
        except Exception as e:
            return {"ok": False, "err": f"torch raised on sample {index}: {e!r}"}
        return {
            "ok": True,
            "args": [encode_arg(a) for a in args],
            "kwargs": {k: encode_arg(v) for k, v in kwargs.items()},
            "output": encode_arg(output),
        }

    if op_name not in _OPS:
        return {"ok": False, "err": f"unknown op: {op_name}"}
    op_info = _OPS[op_name]
    _reset_torch_state(seed)

    samples = list(op_info.sample_inputs(
        device="cpu",
        dtype=torch.float32,
        requires_grad=False,
    ))
    if index >= len(samples):
        return {"ok": False, "err": f"index {index} >= {len(samples)} samples for {op_name}"}

    s = samples[index]
    # SampleInput has .input (positional 0), .args (rest of positional), .kwargs.
    args = (s.input, *s.args)
    kwargs = s.kwargs

    try:
        op_callable = op_info.op
        output = op_callable(*args, **kwargs)
    except Exception as e:
        return {"ok": False, "err": f"torch raised on sample {index}: {e!r}"}

    return {
        "ok": True,
        "args": [encode_arg(a) for a in args],
        "kwargs": {k: encode_arg(v) for k, v in kwargs.items()},
        "output": encode_arg(output),
    }


def execute(op_name: str, args: list, kwargs: dict) -> dict[str, Any]:
    """Execute torch's `op_name` on the given args/kwargs. Used by the
    discriminator when probing adversarial inputs not in op_db's defaults.

    Routes through `_CUSTOM_OPS` for ops not in op_db.
    """
    if op_name in _CUSTOM_OPS:
        entry = _CUSTOM_OPS[op_name]
        _reset_torch_state(0)
        try:
            a = [decode_arg(x) for x in args]
            k = {k: decode_arg(v) for k, v in kwargs.items()}
            output = entry["callable"](*a, **k)
            return {"ok": True, "output": encode_arg(output)}
        except Exception as e:
            return {"ok": False, "err": f"torch raised: {e!r}"}

    if op_name not in _OPS:
        return {"ok": False, "err": f"unknown op: {op_name}"}
    op_info = _OPS[op_name]
    _reset_torch_state(0)
    try:
        a = [decode_arg(x) for x in args]
        k = {k: decode_arg(v) for k, v in kwargs.items()}
        output = op_info.op(*a, **k)
        return {"ok": True, "output": encode_arg(output)}
    except Exception as e:
        return {"ok": False, "err": f"torch raised: {e!r}"}


# ---------------------------------------------------------------------------
# Adversarial probe (discriminator pass) — materializes tensors from a JSON
# spec language understood by both this oracle and the Rust runner so that
# the same input shape/dtype/data/transform is constructed on both sides.
# ---------------------------------------------------------------------------

# Special float tokens accepted in `data` / `fill` fields. JSON has no NaN/Inf
# literals so the spec language uses these sentinels (matched in the Rust
# runner verbatim).
import struct as _struct

# Compute f32::MAX exactly (round-trip via 4-byte float pack/unpack) so the
# value survives JSON encoding without crossing the f32::MAX threshold and
# tripping torch's "overflow on cast" guard.
_F32_MAX_BYTES = b"\xff\xff\x7f\x7f"   # IEEE 754 finite max for float32 (0x7F7FFFFF LE)
_F32_MAX = _struct.unpack("<f", _F32_MAX_BYTES)[0]
_NEG_F32_MAX = -_F32_MAX

_FLOAT_TOKENS = {
    "NaN": float("nan"),
    "+Inf": float("inf"),
    "-Inf": float("-inf"),
    "+0": 0.0,
    "-0": -0.0,
    # f32::MIN_POSITIVE / 2 — a true denormal (subnormal) in float32.
    "DENORM": 1.1754943508222875e-38 / 2.0,
    # f32::MAX as the exact f32 value (round-trip-safe across JSON).
    "F32_MAX": _F32_MAX,
    "-F32_MAX": _NEG_F32_MAX,
}


def _resolve_scalar(v: Any) -> float:
    if isinstance(v, str):
        if v not in _FLOAT_TOKENS:
            raise ValueError(f"unknown float token: {v!r}")
        return _FLOAT_TOKENS[v]
    return float(v)


def _torch_dtype(name: str) -> torch.dtype:
    return _NAME_TO_DTYPE[name]


def _materialize_tensor(spec: dict[str, Any]) -> torch.Tensor:
    """Build a torch.Tensor from a discriminator probe spec.

    spec keys: shape, dtype, data | fill, transform, transform_args.
    transform ∈ {"none", "transpose", "expand", "slice_step"}.
    """
    shape = list(spec["shape"])
    dtype_name = spec.get("dtype", "float32")
    dtype = _torch_dtype(dtype_name)
    data = spec.get("data")
    fill = spec.get("fill")

    numel = 1
    for d in shape:
        numel *= d

    if data is not None:
        if numel != len(data) and numel != 0:
            raise ValueError(
                f"tensor spec: data len {len(data)} != shape numel {numel} ({shape})"
            )
        # For bool dtype, accept native bool. For int dtypes, cast through Python int.
        if dtype_name in ("bool",):
            vals = [bool(x) for x in data]
            t = torch.tensor(vals, dtype=dtype).reshape(shape) if numel > 0 else \
                torch.empty(shape, dtype=dtype)
        elif dtype_name in ("int8", "int16", "int32", "int64", "uint8"):
            vals = [int(x) for x in data]
            t = torch.tensor(vals, dtype=dtype).reshape(shape) if numel > 0 else \
                torch.empty(shape, dtype=dtype)
        else:
            vals = [_resolve_scalar(x) for x in data]
            t = torch.tensor(vals, dtype=dtype).reshape(shape) if numel > 0 else \
                torch.empty(shape, dtype=dtype)
    elif fill is not None:
        v = _resolve_scalar(fill) if dtype_name not in ("bool", "int8", "int16",
                                                       "int32", "int64", "uint8") \
            else (bool(fill) if dtype_name == "bool" else int(fill))
        t = torch.full(shape, v, dtype=dtype)
    else:
        t = torch.zeros(shape, dtype=dtype)

    transform = spec.get("transform", "none")
    targs = spec.get("transform_args", {}) or {}
    if transform == "none":
        # No-op transform: tensor is used as constructed.
        t = t
    elif transform == "transpose":
        t = t.transpose(int(targs["dim0"]), int(targs["dim1"]))
    elif transform == "expand":
        t = t.expand(list(targs["shape"]))
    elif transform == "slice_step":
        start = int(targs.get("start", 0))
        stop = int(targs.get("stop", t.shape[0]))
        step = int(targs.get("step", 1))
        t = t[start:stop:step]
    else:
        raise ValueError(f"unknown transform: {transform}")
    return t


def _materialize_arg(spec: Any, *, _alias_a: torch.Tensor | None = None) -> Any:
    if isinstance(spec, str) and spec == "ALIAS_A":
        if _alias_a is None:
            raise ValueError("ALIAS_A referenced before first tensor materialized")
        return _alias_a
    if isinstance(spec, dict):
        kind = spec.get("kind")
        if kind == "tensor":
            return _materialize_tensor(spec)
        if kind == "scalar":
            return spec["value"]
    return spec


def _resolve_alpha(kwargs: dict[str, Any]) -> Any:
    a = kwargs.get("alpha", None)
    if a is None:
        return None
    if isinstance(a, str):
        return _FLOAT_TOKENS.get(a, a)
    return a


def probe(op_name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Execute one adversarial probe in torch and return the output (and grads
    if requested). Mirrors the materialization the Rust runner performs."""
    if op_name not in _OPS:
        return {"ok": False, "err": f"unknown op: {op_name}"}
    _reset_torch_state(0)
    args_spec = spec.get("args_spec", [])
    kwargs = spec.get("kwargs", {}) or {}
    autograd_check = bool(spec.get("autograd_check", False))

    # Materialize args; first tensor available as ALIAS_A for "self-add" probes.
    materialized: list[Any] = []
    first_tensor: torch.Tensor | None = None
    for s in args_spec:
        m = _materialize_arg(s, _alias_a=first_tensor)
        materialized.append(m)
        if first_tensor is None and isinstance(m, torch.Tensor):
            first_tensor = m

    # Apply requires_grad if requested.
    rg = kwargs.get("requires_grad")
    if rg is not None:
        torch.set_grad_enabled(True)
        for i, want in enumerate(rg):
            if want and isinstance(materialized[i], torch.Tensor) \
                    and materialized[i].is_floating_point():
                materialized[i].requires_grad_(True)

    alpha = _resolve_alpha(kwargs)
    inplace = bool(kwargs.get("inplace", False))
    out_spec = kwargs.get("out_spec")

    torch_call_kwargs: dict[str, Any] = {}
    if alpha is not None:
        torch_call_kwargs["alpha"] = alpha
    if out_spec is not None:
        torch_call_kwargs["out"] = _materialize_tensor(out_spec)

    try:
        if inplace:
            # torch.Tensor.add_ — mutating call.
            a, b = materialized[0], materialized[1]
            ip_kwargs = {}
            if alpha is not None:
                ip_kwargs["alpha"] = alpha
            output = a.add_(b, **ip_kwargs)
        else:
            output = torch.add(materialized[0], materialized[1], **torch_call_kwargs)

        result: dict[str, Any] = {
            "ok": True,
            "output": encode_arg(output.detach() if output.requires_grad else output),
        }

        if autograd_check and rg is not None and any(rg):
            # Reduce to scalar then backprop ones. Compare grads on each
            # requires_grad input.
            (output.sum()).backward()
            grads: list[Any] = []
            for i, want in enumerate(rg):
                if want and isinstance(materialized[i], torch.Tensor):
                    g = materialized[i].grad
                    grads.append(encode_arg(g) if g is not None else None)
                else:
                    grads.append(None)
            result["grads"] = grads
        return result
    except Exception as e:
        return {"ok": False, "err": f"torch raised: {type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def respond(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            respond({"ok": False, "err": f"bad json: {e}"})
            continue

        cmd = req.get("cmd")
        try:
            if cmd == "ready":
                respond({
                    "ok": True,
                    "torch": torch.__version__,
                    "ops": len(_OPS) + len(_CUSTOM_OPS),
                })
            elif cmd == "list_ops":
                respond({"ok": True, "ops": list_ops()})
            elif cmd == "sample":
                respond(sample(req["op"], int(req.get("seed", 0)), int(req.get("i", 0))))
            elif cmd == "execute":
                respond(execute(req["op"], req.get("args", []), req.get("kwargs", {})))
            elif cmd == "probe":
                respond(probe(req["op"], req.get("spec", {})))
            elif cmd == "shutdown":
                respond({"ok": True})
                return 0
            else:
                respond({"ok": False, "err": f"unknown cmd: {cmd!r}"})
        except Exception as e:
            respond({"ok": False, "err": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()})

    return 0


if __name__ == "__main__":
    sys.exit(main())
