# GPU GroupNorm forward kernel (f32)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/group_norm_kernel.cu
-->

## Summary

`ferrotorch-gpu/src/group_norm.rs` implements the GPU forward kernel
for `torch.nn.functional.group_norm` on `[B, C, H, W]` f32 tensors.
A single hand-written PTX kernel runs as a `(groups, batch, 1)` grid
of 256-thread blocks; each block reduces over `(C/G) * H*W` elements
to compute per-`(batch, group)` mean and variance, then applies the
per-channel affine `gamma`/`beta`. Mirrors the
`GroupNormKernelImpl` family in
`aten/src/ATen/native/cuda/group_norm_kernel.cu`.

## Requirements

- REQ-1: Public f32 forward entry point `pub fn gpu_group_norm_f32`
  taking `(input, weight, bias, batch, channels, groups, hw, eps,
  device)` and returning a `CudaBuffer<f32>` of length `batch *
  channels * hw`. Mirrors PyTorch's `[B, C, H, W]` layout.
- REQ-2: Hand-written PTX kernel `group_norm_kernel`
  (`GROUP_NORM_PTX`) with the documented ABI `(in_ptr, out_ptr,
  w_ptr, b_ptr, batch, channels, groups, hw, eps)`. The kernel
  performs three passes: mean, variance, normalize+affine.
- REQ-3: Shape validation: `channels % groups == 0`, `input.len() ==
  batch * channels * hw`, `weight.len() == bias.len() == channels`,
  all on the same device. Violations return
  `GpuError::ShapeMismatch` / `DeviceMismatch`.
- REQ-4: Degenerate handling: when `n == 0` or `channels == 0` or
  `hw == 0`, return a zero-filled buffer of the correct length
  without launching.
- REQ-5: Non-test production consumer wiring through the
  `CudaBackendImpl` GroupNorm forward trait method (consumed by
  ferrotorch-nn's `GroupNorm` layer for SD VAE / UNet stacks).
- REQ-6: BatchNorm GPU forward (`gpu_batch_norm_f32`) — per-channel
  normalize over `(batch, hw)` (train) / running stats (eval), wired to
  `ferrotorch-nn::BatchNorm{1,2,3}d::forward`. (#1449)
- REQ-7: BatchNorm GPU backward (`gpu_batch_norm_backward_f32`) —
  on-device `(grad_input, grad_weight, grad_bias)` mirroring
  `Normalization.cuh:388 batch_norm_backward_kernel`, wired to
  `BatchNorm{1,2,3}dBackward` + `InstanceNormBackward`. (#1449)
- REQ-8: LocalResponseNorm GPU forward + backward
  (`gpu_local_response_norm_f32` / `gpu_local_response_norm_backward_f32`) —
  cross-channel-window normalization per `torch/nn/functional.py:3032-3046`,
  wired to `LocalResponseNorm::forward` + `LocalResponseNormBackward`. (#1449)

## Acceptance Criteria

- [x] AC-1: `pub fn gpu_group_norm_f32` exists with the documented
  signature.
- [x] AC-2: `pub(crate) const GROUP_NORM_PTX` carries the three-pass
  reduction-then-affine PTX matching upstream's `GroupNorm1dForward`
  / `GroupNormKernelImplInternal`.
- [x] AC-3: Validation paths emit `ShapeMismatch` for the four error
  shapes (channels-groups divisibility, input length, weight length,
  bias length) and `DeviceMismatch` for cross-device inputs.
- [x] AC-4: Three unit tests in `mod tests` exercise small-shape
  parity vs CPU reference, the SD-VAE `[1, 128, 4, 4]` shape with
  G=32, and the divisibility-rejection error path.
- [x] AC-5: Non-test consumer wired through `CudaBackendImpl` —
  `CudaBackendImpl::group_norm_f32` in `backend_impl.rs` calls
  `crate::group_norm::gpu_group_norm_f32`, and ferrotorch-core's
  `GpuBackend::group_norm_f32` trait method (gpu_dispatch.rs) is
  consumed by `ferrotorch-nn::GroupNorm::forward`'s GPU fast path
  (#1356/#1357 landed).

## Architecture

`pub fn gpu_group_norm_f32 in group_norm.rs` does:

1. Validates `groups != 0` and `channels % groups == 0`.
2. Validates `input.len() == batch * channels * hw`,
   `weight.len() == channels`, `bias.len() == channels`.
3. Validates all buffers share the same device ordinal.
4. Short-circuits the degenerate (`n == 0` / `channels == 0` /
   `hw == 0`) case by returning a zero-allocation buffer.
5. Resolves `group_norm_kernel` via `crate::module_cache::get_or_compile`.
6. Allocates the f32 output via `alloc_zeros_f32(n, device)`.
7. Launches with `grid_dim = (groups, batch, 1)`, `block_dim = (256, 1, 1)`,
   shared_mem = 256 * 4 bytes for the per-block reduction scratch.
8. Returns the `CudaBuffer<f32>` output.

The PTX kernel (`pub(crate) const GROUP_NORM_PTX in group_norm.rs`)
runs three passes per `(b, g)` block:

- **Pass 1 (mean)**: strided thread loop accumulates `Σ x[..]` for the
  group, stores to shared, runs a tree reduction to thread 0, divides
  by `n_elem = (C/G) * hw` to get the mean.
- **Pass 2 (variance)**: strided loop with `fma.rn.f32` accumulates
  `Σ (x - mean)^2`, tree-reduces, divides by `n_elem`, adds `eps`,
  takes `sqrt.approx.f32` then `rcp.approx.f32` to get `inv_std`.
- **Pass 3 (normalize + affine)**: per-element strided loop computes
  `c = c_start + i/hw`, loads `gamma[c]`/`beta[c]`, writes
  `out = gamma * (x - mean) * inv_std + beta` via `fma.rn.f32`.

Non-test consumer (REQ-5): `CudaBackendImpl::group_norm_f32` in
`ferrotorch-gpu/src/backend_impl.rs` unwraps the three f32 handles and
calls `crate::group_norm::gpu_group_norm_f32`. This overrides the
`GpuBackend::group_norm_f32` default (`InvalidArgument`) declared in
`ferrotorch-core/src/gpu_dispatch.rs`, which is invoked by
`ferrotorch-nn::GroupNorm::forward`'s GPU fast path when
`input.is_cuda()` and a backend is registered (`ferrotorch-nn/src/norm.rs`).
The `pub use group_norm::gpu_group_norm_f32` re-export at
`ferrotorch-gpu/src/lib.rs` remains the ergonomic symbol export.

## Parity contract

`parity_ops = []` for this route. GroupNorm forward parity is enforced
at the ferrotorch-nn layer (the `GroupNorm` module's forward test),
which in turn relies on the CPU and GPU forward implementations
agreeing. The unit tests in this file pin the GPU ↔ CPU agreement
within `1e-4` absolute tolerance.

Edge cases preserved:

- **Empty tensor** (`n == 0` or `channels == 0` or `hw == 0`):
  returns a length-`n` zero buffer without launching.
- **`groups == 0`**: rejected with `ShapeMismatch`.
- **Channels not divisible by groups**: rejected with `ShapeMismatch`
  (verified by `group_norm_validates_groups_divisibility`).
- **Numerical stability**: `eps` is added to the variance before
  `sqrt`, matching upstream's `T_ACC eps` handling.
- **Per-channel affine**: `gamma` and `beta` are read at
  `c = c_start + i/hw`, exactly mirroring upstream's
  `gamma[c] * normed + beta[c]`.

The PTX uses `div.approx.f32`, `sqrt.approx.f32`, `rcp.approx.f32`
for the mean / inv_std computation, which is bit-equivalent to what
upstream uses for f32 (the variance reduction itself is
single-precision; for `T_ACC = float` upstream behaves the same way).

## Softmax2d co-resident kernel (#1451)

`group_norm.rs` also hosts the channel-axis softmax (`Softmax2d`)
forward kernel, since it is a norm-adjacent reduction over the channel
axis and the manifest for that build excluded `kernels.rs`:

- `pub(crate) const SOFTMAX2D_PTX` — `softmax2d_kernel` ABI
  `(in_ptr, out_ptr, total, channels, hw)`; one thread per `(n, p)`
  spatial position, three sequential per-position passes
  (max-find, `exp` sum via `ex2.approx.f32(v * log2(e))`, normalize)
  over `c` channel values strided `hw` apart. Mirrors
  `torch.nn.Softmax2d` (softmax over `dim=1`).
- `pub fn gpu_softmax2d_f32(input, n, c, hw, device)` — validates
  `input.len() == n*c*hw`, device match, short-circuits the
  degenerate shape, launches `softmax2d_kernel`.
- Non-test consumer: `CudaBackendImpl::softmax2d_f32 in backend_impl.rs`
  → `GpuBackend::softmax2d_f32` (gpu_dispatch.rs) →
  `ferrotorch-nn::Softmax2d::forward` GPU fast path.

## Verification

Unit tests in `ferrotorch-gpu/src/group_norm.rs` `mod tests`:

- `group_norm_matches_cpu_small` — `[2, 16, 5]` with G=4, eps=1e-6.
- `group_norm_sd_vae_shape` — `[1, 128, 16]` with G=32 (SD VAE shape).
- `group_norm_validates_groups_divisibility` — error-path: C=10, G=3.
- `softmax2d_matches_cpu_small` — `[2, 5, 12]` channel-axis softmax
  vs CPU reference.
- `softmax2d_columns_sum_to_one` — every `(n, p)` channel column sums
  to 1.0 within 1e-4.
- `softmax2d_validates_length` — error-path: buffer length disagrees
  with `n*c*hw`.

Each test uses the `match GpuDevice::new(0)` graceful-skip pattern.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda group_norm:: 2>&1 | tail -3
```

Expected: ≥ 1 `test result: ok` line (or graceful skip on hosts
without CUDA).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn gpu_group_norm_f32 in ferrotorch-gpu/src/group_norm.rs` mirrors upstream `GroupNormKernelImpl` at `aten/src/ATen/native/cuda/group_norm_kernel.cu:649`. Non-test production consumer: `CudaBackendImpl::group_norm_f32 in ferrotorch-gpu/src/backend_impl.rs` calls it; reached from `ferrotorch-nn::GroupNorm::forward` (`ferrotorch-nn/src/norm.rs`) GPU fast path → `GpuBackend::group_norm_f32` (`ferrotorch-core/src/gpu_dispatch.rs`). (#1356) |
| REQ-2 | SHIPPED | impl: `pub(crate) const GROUP_NORM_PTX in group_norm.rs` (line 56) carries the three-pass mean / variance / affine PTX; ABI matches the launch site at `group_norm in group_norm.rs`. (Implementation-detail REQ; verified by the unit tests being numerically correct.) |
| REQ-3 | SHIPPED | impl: validation checks at `group_norm.rs` lines 291-328 (groups divisibility, input length, weight length, bias length, device ordinal). |
| REQ-4 | SHIPPED | impl: degenerate short-circuit at `group_norm in group_norm.rs` returns `alloc_zeros_f32(n, device)` for `n == 0 || channels == 0 || hw == 0`. |
| REQ-5 | SHIPPED | impl: `fn group_norm_f32` on the `GpuBackend` trait in `ferrotorch-core/src/gpu_dispatch.rs` (default `InvalidArgument`), overridden by `CudaBackendImpl::group_norm_f32 in ferrotorch-gpu/src/backend_impl.rs` calling `crate::group_norm::gpu_group_norm_f32`. Non-test production consumer: `ferrotorch-nn::GroupNorm::forward` GPU fast path in `ferrotorch-nn/src/norm.rs` dispatches to `backend.group_norm_f32(...)` for CUDA-resident input (replacing the prior `NotImplementedOnCuda` reject). (#1357) |
| REQ-6 | SHIPPED | impl: `pub fn gpu_batch_norm_f32 in ferrotorch-gpu/src/group_norm.rs` + `pub(crate) const BATCH_NORM_PTX` (`batch_norm_kernel`, one block per channel, reduces over `(batch, hw)` in train mode / reads running stats in eval mode, then per-channel affine), mirroring `aten/src/ATen/native/Normalization.cpp::batch_norm_cpu_transform_input_template`. Trait slot `fn batch_norm_f32` on `GpuBackend` in `ferrotorch-core/src/gpu_dispatch.rs` (default `InvalidArgument`) overridden by `CudaBackendImpl::batch_norm_f32 in ferrotorch-gpu/src/backend_impl.rs`. Non-test production consumer: `ferrotorch-nn::BatchNorm{1,2,3}d::forward` GPU fast path (`batch_norm_gpu_forward` helper in `ferrotorch-nn/src/norm.rs`) dispatches to `backend.batch_norm_f32(...)` for f32 CUDA-resident input. Live GPU↔CPU parity pinned by `batch_norm_training_matches_cpu` / `batch_norm_eval_uses_running_stats` / `batch_norm_validates_lengths`. (#1449) |
| REQ-7 | SHIPPED | impl: `pub fn gpu_batch_norm_backward_f32 in ferrotorch-gpu/src/group_norm.rs` + `pub(crate) const BATCH_NORM_BACKWARD_PTX` (`batch_norm_backward_kernel`, one block per channel: reduces `grad_output_sum` + `dot_p = sum (x-mean)*go`; `grad_input = train ? (go - (x-mean)*proj_scale - grad_mean)*grad_scale : go*grad_scale`; `grad_weight = dot_p*invstd`; `grad_bias = grad_output_sum`), mirroring `aten/src/ATen/native/cuda/Normalization.cuh:388 batch_norm_backward_kernel`. Trait slot `fn batch_norm_backward_f32` on `GpuBackend` in `ferrotorch-core/src/gpu_dispatch.rs` (default `InvalidArgument`) overridden by `CudaBackendImpl::batch_norm_backward_f32 in ferrotorch-gpu/src/backend_impl.rs`. Non-test production consumer: `ferrotorch-nn::BatchNorm{1,2,3}dBackward::backward` (`batch_norm_gpu_backward` helper) + `InstanceNormBackward::backward` (`instance_norm_gpu_backward` helper via the `[1,B*C,S]` reshape) in `ferrotorch-nn/src/norm.rs`, on-device with NO `.cpu()` round trip. Live-vs-torch-autograd grad parity (<1e-3) pinned by `divergence_batchnorm2d_gpu_{train,eval}_backward_vs_torch` + `divergence_instancenorm2d_gpu_backward_vs_torch`. (#1449) |
| REQ-8 | SHIPPED | impl: `pub fn gpu_local_response_norm_f32` (PTX `LRN_FORWARD_PTX`) + `pub fn gpu_local_response_norm_backward_f32` (PTX `LRN_BACKWARD_PTX`) in `ferrotorch-gpu/src/group_norm.rs`, one thread per element with the cross-channel window, mirroring `torch/nn/functional.py:3032-3046 local_response_norm` (square → windowed channel sum → `*alpha+k` → pow(beta) → divide; forward saves `denom` for the backward VJP). Trait slots `fn local_response_norm_f32` / `fn local_response_norm_backward_f32` on `GpuBackend` in `ferrotorch-core/src/gpu_dispatch.rs` (default `InvalidArgument`) overridden by `CudaBackendImpl in ferrotorch-gpu/src/backend_impl.rs`. Non-test production consumer: `ferrotorch-nn::LocalResponseNorm::forward` + `LocalResponseNormBackward::backward` in `ferrotorch-nn/src/norm.rs`, on-device (GPU-resident saved `denom`), NO `.cpu()` round trip. Live-vs-torch fwd+grad parity (<1e-3) pinned by `divergence_local_response_norm_gpu_fwd_bwd_vs_torch`. (#1449) |
