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
- [ ] AC-5: Non-test consumer wired through `CudaBackendImpl` —
  currently no `group_norm_*` trait method consumes this kernel in
  `backend_impl.rs`; the kernel is exported but not dispatched to
  from ferrotorch-core's gpu-dispatch path. See REQ-5 blocker.

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

Non-test consumer (REQ-5): no production consumer currently routes to
this kernel — the `pub use group_norm::gpu_group_norm_f32` re-export
at `ferrotorch-gpu/src/lib.rs:224` exposes the symbol, but
`CudaBackendImpl` in `backend_impl.rs` does not yet implement a
`group_norm_*` trait method, and ferrotorch-core's gpu-dispatch
surface does not list one. The autocast op-list at
`ferrotorch-core/src/autograd/autocast_ops.rs:39` mentions
`"group_norm"`, but the dispatch wiring through GPU is missing.

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

## Verification

Unit tests in `ferrotorch-gpu/src/group_norm.rs` `mod tests`:

- `group_norm_matches_cpu_small` — `[2, 16, 5]` with G=4, eps=1e-6.
- `group_norm_sd_vae_shape` — `[1, 128, 16]` with G=32 (SD VAE shape).
- `group_norm_validates_groups_divisibility` — error-path: C=10, G=3.

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
| REQ-1 | NOT-STARTED | impl: `pub fn gpu_group_norm_f32 in ferrotorch-gpu/src/group_norm.rs` (line 280) mirrors upstream `GroupNormKernelImpl` at `aten/src/ATen/native/cuda/group_norm_kernel.cu:649`, BUT no non-test production consumer invokes it — only the three `#[cfg(test)] mod tests` callers and the `pub use` re-export at `lib.rs:224` reference it. Open prereq blocker #1356: needs a `group_norm_f32` trait method on `GpuBackend` + `CudaBackendImpl` consumer wiring. |
| REQ-2 | SHIPPED | impl: `pub(crate) const GROUP_NORM_PTX in group_norm.rs` (line 56) carries the three-pass mean / variance / affine PTX; ABI matches the launch site at `group_norm.rs:388`. (Implementation-detail REQ; verified by the unit tests being numerically correct.) |
| REQ-3 | SHIPPED | impl: validation checks at `group_norm.rs` lines 291-328 (groups divisibility, input length, weight length, bias length, device ordinal). |
| REQ-4 | SHIPPED | impl: degenerate short-circuit at `group_norm.rs:331` returns `alloc_zeros_f32(n, device)` for `n == 0 || channels == 0 || hw == 0`. |
| REQ-5 | NOT-STARTED | no `group_norm_*` trait method exists on `GpuBackend` in `ferrotorch-core/src/gpu_dispatch.rs`, and no `fn group_norm_*` exists in `backend_impl.rs`. Open prereq blocker #1357: add `group_norm_f32` to the `GpuBackend` trait + wire `CudaBackendImpl` to call `crate::group_norm::gpu_group_norm_f32`. ferrotorch-nn's `GroupNorm` module currently falls back to a CPU path or per-element kernels rather than this single optimised kernel. |
