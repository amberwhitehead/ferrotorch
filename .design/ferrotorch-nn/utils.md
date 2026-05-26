# ferrotorch-nn — `clip_grad_norm_` / `clip_grad_value_`

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/utils/clip_grad.py
-->

## Summary

`ferrotorch-nn/src/utils.rs` defines the two canonical
gradient-clipping helpers: `clip_grad_norm_` (rescale gradients so
the total norm doesn't exceed `max_norm`) and `clip_grad_value_`
(clamp each gradient element into `[-clip_value, clip_value]`).
Both operate in-place on the gradients of a slice of `&Parameter<T>`
references. Mirrors `torch.nn.utils.clip_grad_norm_` /
`clip_grad_value_` from `torch/nn/utils/clip_grad.py:184-298`. The
file ships a documented per-device dispatch policy: CPU runs on
host; CUDA f32/f64 runs on GPU via real kernel launches
(`gpu_reduce_sum`, `gpu_scale`, `gpu_clamp`); mixed-device errors;
non-L2 norms on CUDA explicitly error rather than silently falling
back to CPU.

## Requirements

- REQ-1: `pub fn clip_grad_norm_<T: Float>(params: &[&Parameter<T>],
  max_norm: f64, norm_type: f64) -> FerrotorchResult<f64>` — clips
  the **joint** L_norm_type norm across all parameters and returns
  the **pre-clip total norm**. Parameters with `grad() == None` are
  silently skipped (matches upstream's `if p.grad is not None`).
  Mirrors `torch.nn.utils.clip_grad_norm_(parameters, max_norm,
  norm_type)` from `clip_grad.py:184-232`.

- REQ-2: Device-classification preamble: collect all gradient tensors
  (skipping None); if `grads.is_empty()`, return 0.0; determine
  common device by inspecting the first grad's device and verifying
  every subsequent grad matches. Mismatch → `DeviceMismatch`
  error. Mirrors upstream's per-device grouping
  (`_group_tensors_by_device_and_dtype` from `clip_grad.py:84`),
  but ferrotorch enforces single-device rather than grouping.

- REQ-3: CPU path `clip_grad_norm_cpu`:
  - If `norm_type == f64::INFINITY`: max absolute value across
    all elements.
  - Else: `accum += |v|^norm_type` over all elements, then
    `total_norm = accum^(1/norm_type)`.
  - Uses the workspace `cast::<T, f64>` helper rather than
    `T::to_f64().unwrap()` so bf16 / f16 overflow surfaces as
    `Err(InvalidArgument)` instead of panicking.
  - If `total_norm > max_norm`: compute `clip_coef = max_norm /
    total_norm`, cast to `T`, scale each gradient in place via
    `Vec<T>` map + `Tensor::from_storage(TensorStorage::cpu(scaled), ...)`.

- REQ-4: CUDA path `clip_grad_norm_cuda` — L2 only, f32 or f64 only:
  - Phase 1: per-tensor `gpu_reduce_sum(g * g)` on GPU. The
    1-element `sum_handle` is transferred to host (4 or 8 bytes
    per tensor — the documented "per-tensor scalar boundary").
    `total_sq` accumulates on the host. Single `total_norm.sqrt()`
    on host.
  - Phase 2: if clipping needed, per-tensor `backend.scale_f32` /
    `scale_f64` (one kernel per parameter); the scaled GPU
    handle replaces the gradient via
    `Tensor::from_storage(TensorStorage::gpu(...))`.
  - Boundary documented at the module head: N scalars
    transferred, not N full tensors. Matches PyTorch's own
    implementation which also accumulates per-parameter norm on
    host (cf. `_get_total_norm` in `clip_grad.py:48-117`).

- REQ-5: `pub fn clip_grad_value_<T: Float>(params: &[&Parameter<T>],
  clip_value: f64) -> FerrotorchResult<()>` — clamps each gradient
  element into `[-clip_value, clip_value]`. None-grads silently
  skipped. Mirrors `torch.nn.utils.clip_grad_value_` from
  `clip_grad.py:256-298`.

- REQ-6: CPU path `clip_grad_value_cpu` — straightforward Vec map.
  Uses the workspace `cast::<f64, T>` for `-clip_value` / `+clip_value`
  so bf16 over/underflow surfaces as `Err`.

- REQ-7: CUDA path `clip_grad_value_cuda` — invokes
  `backend.clamp_f32(g_handle, -clip_value as f32, clip_value as
  f32)` (or `clamp_f64`). The clamped GPU handle replaces the
  gradient via `TensorStorage::gpu(...)`.

- REQ-8: Device-error contract:
  - Mixed-device grads (some CPU, some CUDA, or different CUDA
    ordinals) → `DeviceMismatch`.
  - Single non-CUDA non-CPU device (MPS, XPU, …) →
    `DeviceUnavailable` (no implementation).
  - CUDA with non-f32/f64 `T` → `NotImplementedOnCuda` (matches
    PyTorch's GPU kernel constraint).
  - CUDA + `clip_grad_norm_` with `norm_type != 2.0` →
    `NotImplementedOnCuda` (PyTorch also restricts the GPU L2
    fast path; other norms must use `.cpu()` explicitly — R-DEV-1
    numerical-contract match).

- REQ-9: `readback_scalar_f32` / `readback_scalar_f64` private
  helpers that materialize a single `f32` / `f64` from a
  1-element `GpuBufferHandle`. Used only by the CUDA path's
  per-tensor scalar-boundary readback. Each validates the
  buffer's byte length and surfaces `Err(InvalidArgument)` if it's
  shorter than expected (defensive; the upstream
  `gpu_reduce_sum` always produces the right size).

## Acceptance Criteria

- [x] AC-1: `pub fn clip_grad_norm_` signature matches REQ-1.
- [x] AC-2: Returns 0.0 on empty params (or all-None grads).
- [x] AC-3: CPU path supports `f64::INFINITY` norm_type (max-abs).
- [x] AC-4: CPU path supports arbitrary positive norm_type.
- [x] AC-5: Returns the pre-clip total norm.
- [x] AC-6: When `total_norm > max_norm`, gradients scaled in
  place; new total norm ≤ max_norm.
- [x] AC-7: CUDA L2 path runs per-tensor reduce-sum on GPU + one
  scalar transfer per tensor.
- [x] AC-8: Mixed-device error.
- [x] AC-9: Non-L2 norm on CUDA → `NotImplementedOnCuda`.
- [x] AC-10: `pub fn clip_grad_value_` signature matches REQ-5.
- [x] AC-11: CPU path clamps elements (in-place); within-range
  values unchanged.
- [x] AC-12: CUDA path runs `backend.clamp_f32` /
  `clamp_f64`.
- [x] AC-13: GPU vs CPU numerical correctness (within 1e-5 for f32).
- [x] AC-14: After clipping, gradient device is preserved (still
  on CUDA).

## Architecture

### Device dispatch (REQ-2, REQ-8)

```rust
pub fn clip_grad_norm_<T: Float>(params, max_norm, norm_type) -> FerrotorchResult<f64> {
    let grads: Vec<Tensor<T>> = params.iter()
        .filter_map(|p| p.grad().ok().flatten())
        .collect();
    if grads.is_empty() { return Ok(0.0); }

    let common_device = grads[0].device();
    for g in &grads[1..] {
        if g.device() != common_device {
            return Err(FerrotorchError::DeviceMismatch { ... });
        }
    }

    match common_device {
        Device::Cpu => clip_grad_norm_cpu(params, &grads, max_norm, norm_type),
        Device::Cuda(_) => { /* dtype + norm_type guards, then cuda path */ }
        _ => Err(FerrotorchError::DeviceUnavailable),
    }
}
```

Single-device enforcement is deliberate. Upstream PyTorch groups
gradients by device and handles each group independently; ferrotorch
errors instead. The error surfaces a real mismatch (model split
across devices without the user's awareness) earlier — matching
the user expectation that gradient clipping is one of the last
steps before optimizer.step() and the model should already be
device-coherent.

### CPU path (REQ-3, REQ-6)

`clip_grad_norm_cpu` computes the norm in two passes:

1. Norm accumulation:
   - `norm_type == INFINITY`: `max_val = max(|v|)` over all
     elements of all grads.
   - else: `accum += |v|^norm_type` then `^(1/norm_type)`.

2. Scaling (only if `total_norm > max_norm`):
   - `clip_coef = max_norm / total_norm` (cast to `T`).
   - Each gradient: `data.iter().map(|&v| v * clip_t).collect::<Vec<T>>()`.
   - New tensor: `Tensor::from_storage(TensorStorage::cpu(scaled), shape, false)`.
   - Replace via `param.set_grad(Some(new_grad))`.

The cast helper (`cast::<T, f64>` and `cast::<f64, T>`) is used
in place of `T::to_f64().unwrap()` so dtype-range violations
(rare for bf16) surface as `Err(InvalidArgument)` rather than
panicking in user code.

`clip_grad_value_cpu` is simpler — just `Vec<T>::iter().map(|&v|
if v < lo { lo } else if v > hi { hi } else { v }).collect()`.

### CUDA path (REQ-4, REQ-7)

`clip_grad_norm_cuda` is L2-only:

```rust
let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
let mut total_sq: f64 = 0.0;
for g in grads {
    let g_handle = g.gpu_handle()?;
    let numel = g.numel();
    let per_tensor_sq: f64 = if is_f32::<T>() {
        let sq_handle = backend.mul_f32(g_handle, g_handle)?;
        let sum_handle = backend.sum_f32(&sq_handle, numel)?;
        readback_scalar_f32(&sum_handle, backend)? as f64
    } else {
        let sq_handle = backend.mul_f64(g_handle, g_handle)?;
        let sum_handle = backend.sum_f64(&sq_handle, numel)?;
        readback_scalar_f64(&sum_handle, backend)?
    };
    total_sq += per_tensor_sq;
}
let total_norm = total_sq.sqrt();

if total_norm > max_norm {
    let clip_coef = max_norm / total_norm;
    for param in params {
        if let Some(g) = param.grad()? {
            let g_handle = g.gpu_handle()?;
            let scaled_handle = if is_f32::<T>() {
                backend.scale_f32(g_handle, clip_coef as f32)?
            } else {
                backend.scale_f64(g_handle, clip_coef)?
            };
            let new_grad = Tensor::from_storage(TensorStorage::gpu(scaled_handle), g.shape().to_vec(), false)?;
            param.set_grad(Some(new_grad))?;
        }
    }
}
```

Two GPU ops per parameter (`mul`, `sum`); one scalar readback (4 or 8
bytes); one scale op per parameter if clipping fires. The data
never leaves the GPU as a full tensor — that's the "per-tensor
scalar boundary" the module's top-level doc-comment documents.
Matches PyTorch's `_get_total_norm` shape (compute per-parameter
norm, accumulate, single sqrt on host).

`clip_grad_value_cuda` calls `backend.clamp_f32` /
`clamp_f64`; clamped GPU handle replaces the gradient.

### Helper functions

`is_f32::<T>()` / `is_f64::<T>()` — `TypeId`-based comparisons.
Matches the pattern in `ferrotorch-nn/src/embedding.rs` for the
same dtype-dispatch decision (CPU vs GPU vs bf16/f16 fallback).

`readback_scalar_f32` / `readback_scalar_f64` — defensive helpers
that materialize a single scalar from a `GpuBufferHandle`. Both
validate `bytes.len() >= 4` (or 8) before decoding via
`from_le_bytes`. Used only by the CUDA path's phase-1 reduction.

### Non-test production consumers

- `pub use utils::{clip_grad_norm_, clip_grad_value_}` in
  `lib.rs:257`; prelude re-export at `lib.rs:292`.
- `ferrotorch-train/src/grad_utils.rs:23` —
  `pub use ferrotorch_nn::utils::{clip_grad_norm_, clip_grad_value_}`.
  `ferrotorch-train` is the canonical production consumer; it
  re-exports the helpers under its own name so training drivers
  can write `use ferrotorch_train::{clip_grad_norm_, clip_grad_value_}`.
- Test `ferrotorch-train/src/grad_utils.rs:285-303` pins that
  `ferrotorch_nn::clip_grad_norm_` and the re-exported name
  resolve to the same function (compile-time pointer equality
  check).
- Downstream training loops in tutorial / example scripts call
  `clip_grad_norm_(&params, 1.0, 2.0)?` after `backward()` and
  before `optimizer.step()`.

## Parity contract

`parity_ops = []`. The helpers are utility functions with explicit
numerical behaviour. Edge cases:

- **All gradients None / empty params slice**:
  `clip_grad_norm_` returns `Ok(0.0)`; `clip_grad_value_` returns
  `Ok(())`.
- **`norm_type == INFINITY`**: CPU path takes max absolute value.
- **`norm_type == 2.0` on CUDA**: GPU fast path.
- **`norm_type != 2.0` on CUDA**: `NotImplementedOnCuda` (no
  silent CPU fallback — R-CODE-4).
- **bf16 / f16 on CUDA**: `NotImplementedOnCuda` (the GPU helpers
  are f32/f64 only — PyTorch has the same constraint).
- **`total_norm <= max_norm`**: no scaling; gradients unchanged;
  function returns the unclipped norm.
- **`total_norm > max_norm`**: scaling factor `max_norm / total_norm
  ∈ (0, 1)` applied to every gradient. Cast back to `T` always
  succeeds for values in that range (the workspace `cast` helper
  still surfaces `Err` for hypothetical future dtypes that can't
  represent the range).
- **Mixed-device error**: returns `DeviceMismatch { expected, got }`
  with the conflicting device pair surfaced for the caller's
  diagnostic.
- **`clip_grad_value_` value clamping**: each gradient element is
  clamped independently into `[-clip_value, clip_value]`. Within
  range → unchanged. NaN: passes through unchanged (matches
  upstream's `clamp_min_` / `clamp_max_` behavior — NaN is
  neither `< lo` nor `> hi`).

## Verification

Tests in `mod tests in utils.rs` (~14 tests):

- CPU `clip_grad_norm_`: `test_clip_grad_norm_reduces_norm`,
  `_no_clip_when_below`, `_multiple_params`,
  `_returns_total_norm`, `_skips_none_grads`.
- CPU `clip_grad_value_`: `test_clip_grad_value_clamps_elements`,
  `_skips_none_grads`, `_preserves_within_range`.
- GPU paths (gated on `feature = "cuda"`):
  `test_gpu_clip_grad_norm_l2_f32`,
  `test_gpu_clip_grad_norm_l2_f64`,
  `test_gpu_clip_grad_value_f32`,
  `test_mixed_device_returns_device_mismatch`,
  `test_non_l2_cuda_returns_error`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-nn --lib utils:: 2>&1 | tail -3
```

Expected: all CPU tests pass; CUDA tests pass when the feature is enabled.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn clip_grad_norm_<T: Float>(params: &[&Parameter<T>], max_norm: f64, norm_type: f64) -> FerrotorchResult<f64>` in `utils.rs` mirroring `torch/nn/utils/clip_grad.py:184-232`; non-test consumer: `ferrotorch-train/src/grad_utils.rs:23` `pub use ferrotorch_nn::utils::{clip_grad_norm_, clip_grad_value_}` — the canonical production re-export, used by every training driver. |
| REQ-2 | SHIPPED | impl: device-classification preamble in `clip_grad_norm_` body (collect grads, check empty, determine common device, error on mismatch) in `utils.rs`; non-test consumer: every external invocation hits this preamble first — the production training-driver path. |
| REQ-3 | SHIPPED | impl: `fn clip_grad_norm_cpu` with `INFINITY` max-abs branch + general `^norm_type` accumulator + post-clip scaling in `utils.rs`, using the workspace `cast::<T, f64>` helper to avoid panicking on bf16/f16 range issues; non-test consumer: every CPU-resident-gradient training loop in downstream code. |
| REQ-4 | SHIPPED | impl: `fn clip_grad_norm_cuda` with `backend.mul_f32`/`mul_f64` + `sum_f32`/`sum_f64` + per-tensor scalar readback + `scale_f32`/`scale_f64` in `utils.rs`; non-test consumer: every CUDA-resident-gradient training loop — invoked via the same public `clip_grad_norm_` entry point. The per-tensor scalar boundary (N scalars transferred, not N tensors) matches PyTorch's `_get_total_norm` shape from `torch/nn/utils/clip_grad.py:48-117`. |
| REQ-5 | SHIPPED | impl: `pub fn clip_grad_value_<T: Float>(params: &[&Parameter<T>], clip_value: f64) -> FerrotorchResult<()>` in `utils.rs` mirroring `torch/nn/utils/clip_grad.py:256-298`; non-test consumer: `ferrotorch-train/src/grad_utils.rs:23` re-exports it; downstream training drivers consume it after backward. |
| REQ-6 | SHIPPED | impl: `fn clip_grad_value_cpu` with element-wise clamp via Vec map + `cast::<f64, T>` for the lo/hi bounds in `utils.rs`; non-test consumer: every CPU-resident-gradient clamp call. |
| REQ-7 | SHIPPED | impl: `fn clip_grad_value_cuda` invoking `backend.clamp_f32(g_handle, -clip_value, clip_value)` / `clamp_f64` and replacing the gradient with the clamped GPU handle in `utils.rs`; non-test consumer: every CUDA-resident-gradient clamp call — invoked via the same public `clip_grad_value_` entry point. |
| REQ-8 | SHIPPED | impl: layered error returns in `utils.rs`: `DeviceMismatch` on heterogeneous grads, `DeviceUnavailable` on unsupported single backends (MPS, XPU), `NotImplementedOnCuda` for non-f32/f64 dtypes and for non-L2 norms on CUDA; non-test consumer: every external invocation that hits one of these conditions surfaces the appropriate error to the caller — matches R-CODE-4 (no silent CPU↔GPU round-trips) and the R-DEV-1 numerical-contract-match with upstream's GPU-kernel constraint. |
| REQ-9 | SHIPPED | impl: `fn readback_scalar_f32(handle, backend) -> FerrotorchResult<f32>` and `fn readback_scalar_f64` private helpers with byte-length validation in `utils.rs`; non-test consumer: invoked inside `clip_grad_norm_cuda`'s per-tensor reduction loop — the production path through the CUDA fast-path for every CUDA-resident gradient. |
