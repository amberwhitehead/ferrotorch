# cuFFT-backed GPU FFT primitives

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/SpectralOps.cpp
  - aten/src/ATen/native/cuda/SpectralOps.cu
  - aten/src/ATen/native/cuda/CuFFTPlanCache.h
  - aten/src/ATen/native/cuda/CuFFTUtils.h
  - aten/src/ATen/native/SpectralOps.cpp
-->

## Summary

`ferrotorch-gpu/src/cufft.rs` wraps NVIDIA cuFFT for ferrotorch's
spectral ops. The module mirrors PyTorch's `_fft_c2c_cufft` /
`_fft_r2c_cufft` / `_fft_c2r_cufft` family from
`aten/src/ATen/native/cuda/SpectralOps.cpp`. Functions cover 1-D,
2-D, 3-D, N-D, and axis-selected variants in both single (f32) and
double (f64) precision, in C2C (complex-to-complex), R2C (real-to-
complex), and C2R (complex-to-real) directions, plus the Hermitian
half-FFT pair `hfft` / `ihfft`.

Complex buffers use interleaved `[re, im, re, im, ...]` layout in a
real `CudaBuffer<f32>` / `CudaBuffer<f64>`. This is byte-equal with
`cufftComplex` / `cufftDoubleComplex`, so no reformat step is needed.

Plans are recreated per call rather than cached — cuFFT's plan
creation is light and the lifetime story for cached plans across
streams/devices is fiddly. Performance-critical paths can lift
planning out themselves. Inverse transforms apply `1/n` normalisation
in a follow-up kernel launch, matching torch/numpy convention (cuFFT
itself does not normalise).

## Requirements

- REQ-1: 1-D C2C FFT — `pub fn gpu_fft_c2c_f32` /
  `gpu_fft_c2c_f64`. Takes interleaved `[B, 2N]` real buffer, returns
  same shape. `inverse: bool` selects forward / inverse direction.
  Mirrors `_fft_c2c_cufft` at `aten/src/ATen/native/cuda/SpectralOps.cpp:445`.
- REQ-2: 2-D C2C FFT — `pub fn gpu_fft2_c2c_f32` /
  `gpu_fft2_c2c_f64`. Takes `[H, W, 2]` interleaved buffer; 2-D
  transform across both spatial dims.
- REQ-3: N-D C2C FFT (3-D) — `pub fn gpu_fftn3d_c2c_f32` /
  `gpu_fftn3d_c2c_f64`. 3-D transform across `[D, H, W]`.
- REQ-4: N-D C2C FFT (general 2-D / N-D) — `pub fn
  gpu_fftn2d_c2c_f32` / `gpu_fftn2d_c2c_f64`, plus
  `gpu_fftn_axes_c2c_f32` / `gpu_fftn_axes_c2c_f64` for arbitrary
  axis lists.
- REQ-5: R2C (real-to-complex) FFT — `pub fn gpu_rfft_r2c_f32` /
  `gpu_rfft_r2c_f64`. Takes `[B, N]` real buffer, returns
  `[B, 2 * (N/2 + 1)]` interleaved Hermitian output. Mirrors
  `_fft_r2c_cufft` at `aten/src/ATen/native/cuda/SpectralOps.cpp:318`.
- REQ-6: C2R (complex-to-real) inverse FFT — `pub fn gpu_irfft_c2r_f32`
  / `gpu_irfft_c2r_f64`. Inverse R2C — Hermitian-half complex input,
  real output of length `n_out`. Mirrors `_fft_c2r_cufft` at
  `aten/src/ATen/native/cuda/SpectralOps.cpp:405`.
- REQ-7: Hermitian FFT pair — `pub fn gpu_hfft_f32` / `gpu_hfft_f64`
  (Hermitian-input forward, real output of `n_out`) and `pub fn
  gpu_ihfft_f32` / `gpu_ihfft_f64` (real input, Hermitian-half
  output). These are the `torch.fft.hfft` / `torch.fft.ihfft`
  variants — implemented as C2R + conjugation and R2C + conjugation
  per the `torch.fft` documentation.
- REQ-8: Inverse normalisation — every inverse path multiplies the
  result by `1/n` via a follow-up kernel launch, matching numpy /
  torch convention. cuFFT itself does not normalise; ferrotorch
  adds the post-step launch. (`aten/src/ATen/native/cuda/SpectralOps.cpp:298`
  is the upstream `_fft_apply_normalization_out` mirror.)
- REQ-9: No-CUDA stubs — `cfg(not(feature = "cuda"))` stubs return
  `GpuError::NoCudaFeature`.

## Acceptance Criteria

- [x] AC-1: All FFT functions return errors via `GpuError::Fft(...)`
  on cuFFT failure (no `unwrap` / `expect` in production code).
- [x] AC-2: Round-trip `inverse(forward(x)) == x` modulo 1/n
  normalisation — verified by op-level parity-sweep tests against
  PyTorch oracle.
- [x] AC-3: All entry points have non-test consumers in
  `backend_impl.rs` (cuda backend's `_fft_*` dispatch arms).
- [x] AC-4: No-CUDA stub coverage — `cargo build -p ferrotorch-gpu
  --no-default-features` succeeds.

## Architecture

### 1-D / 2-D / 3-D C2C (REQ-1, REQ-2, REQ-3)

`pub fn gpu_fft_c2c_f32 in cufft.rs` (1-D):
1. Build a `cufftHandle` plan via `cufftPlan1d` (or `cufftPlanMany`
   for batched).
2. Bind plan to the device's default stream via `cufftSetStream`.
3. Allocate `out_buf` on-device.
4. Call `cufftExecC2C(plan, in, out, direction)` where
   `direction = CUFFT_FORWARD` or `CUFFT_INVERSE`.
5. If `inverse`, launch the `1/n` normalisation kernel.
6. Destroy plan.
7. Return `out_buf`.

The 2-D `gpu_fft2_c2c_f32 in cufft.rs` and 3-D
`gpu_fftn3d_c2c_f32 in cufft.rs` follow the same pattern with
`cufftPlan2d` / `cufftPlan3d`.

Non-test consumers in `ferrotorch-gpu/src/backend_impl.rs`:
- `gpu_fft_c2c_f32` at line 4443
- `gpu_fft_c2c_f64` at line 4457
- `gpu_fft2_c2c_f32` at line 4500
- `gpu_fft2_c2c_f64` at line 4514
- `gpu_fftn3d_c2c_f32` at line 4664
- `gpu_fftn3d_c2c_f64` at line 4679
- `gpu_fftn2d_c2c_f32` at line 4695

### General N-D axis-selected (REQ-4)

`pub fn gpu_fftn_axes_c2c_f32 in cufft.rs` accepts an `axes: &[usize]`
list and applies the FFT iteratively along each axis. Uses
`cufftPlanMany` with stride / batch parameters computed from the
input shape and the chosen axis. For multi-axis transforms it is
applied per-axis in sequence; this matches torch's behaviour
documented at `aten/src/ATen/native/SpectralOps.cpp::fft_fft`. The
upstream consumer at `ferrotorch-core/src/fft.rs:745` notes the
`s != None` and dim-list-permutation fallback path.

### R2C / C2R (REQ-5, REQ-6)

`pub fn gpu_rfft_r2c_f32 in cufft.rs`:
1. Allocate output of size `2 * (N/2 + 1)` floats (Hermitian half).
2. `cufftPlan1d` with `CUFFT_R2C` type.
3. `cufftExecR2C(plan, in, out)`.
4. Return.

`pub fn gpu_irfft_c2r_f32 in cufft.rs`:
1. Plan with `CUFFT_C2R` and `n_out`.
2. `cufftExecC2R(plan, in, out)`.
3. Apply `1/n_out` normalisation kernel.
4. Return.

Consumers at `backend_impl.rs` (R2C f32), `backend_impl.rs` (R2C f64),
`backend_impl.rs` (C2R f32), `backend_impl.rs` (C2R f64).

### Hermitian FFT pair (REQ-7)

`pub fn gpu_hfft_f32 in cufft.rs`: takes Hermitian-half complex
input, computes real output of length `n_out`. Implemented as a
sign-conjugation kernel + C2R (since `hfft(x) = irfft(conj(x))`
modulo normalisation; see torch.fft docs).

`pub fn gpu_ihfft_f32 in cufft.rs`: real → Hermitian-half. Computed
as R2C + sign-conjugation kernel.

Consumers at `backend_impl.rs` (hfft f32), `backend_impl.rs` (hfft f64),
`backend_impl.rs` (ihfft f32), `backend_impl.rs` (ihfft f64). The `ferrotorch-core/src/fft.rs,1046`
doc-comments explicitly call out the dispatch as "via cuFFT C2R + conj PTX".

### Inverse normalisation (REQ-8)

Every inverse path issues an additional kernel launch (from
`crate::kernels` family) that multiplies each output element by
`1/n`. Mirrors PyTorch's `_fft_apply_normalization_out` at
`aten/src/ATen/native/cuda/SpectralOps.cpp:298`. The doc-comment at
the top of `cufft.rs` calls out: "Inverse transforms apply `1/n`
normalization in a follow-up kernel launch."

### No-CUDA stubs (REQ-9)

Each FFT function pair has a `#[cfg(not(feature = "cuda"))]` stub
that returns `Err(GpuError::NoCudaFeature)`. The crate compiles in
both modes.

## Parity contract

`parity_ops = []` for this module. Reason: FFT ops are op-level
entries in `ferrotorch-core`'s parity surface (`fft`, `rfft`,
`irfft`, `hfft`, `ihfft`, `fft2`, `fftn`); the cuFFT dispatchers are
reached transitively. The op-level parity-sweep coverage in
`ferrotorch-core` exercises these dispatchers.

Edge cases mirrored from upstream:

- **`n == 0`**: empty input → empty output. cuFFT plan creation
  fails for `n = 0`; the wrapper short-circuits and returns an
  empty buffer matching torch's empty-input behaviour.
- **`n == 1`**: identity transform (no work). Plan creation succeeds
  and `cufftExecC2C` is a no-op.
- **NaN / Inf in input**: propagated by cuFFT into the output; both
  ferrotorch and torch preserve NaN/Inf identically.
- **Non-power-of-2 `n`**: cuFFT supports arbitrary `n` (with
  efficiency degraded for prime factors > 7); the wrapper does NOT
  pad to the next power-of-2 — that responsibility is the caller's,
  matching torch.
- **R2C output layout**: Hermitian-half is `(N/2 + 1)` complex bins
  (interleaved as `(N/2 + 1) * 2` reals). The wrapper documents
  this in the function-level rustdoc.

## Verification

Tests are NOT present in-file (the FFT module relies on op-level
tests in `ferrotorch-core/src/fft.rs` for end-to-end coverage). The
0 in-file test count reflects this — the dispatcher is a thin
cuFFT wrapper, and the round-trip / accuracy checks are at the
core layer.

Smoke commands:

```bash
cargo test -p ferrotorch-core --features cuda fft:: 2>&1 | tail -3
cargo build -p ferrotorch-gpu --no-default-features 2>&1 | tail -3
```

Expected: core-side FFT tests pass; no-cuda compile succeeds.
`parity_ops = []` — no per-op parity-sweep smoke applies at this
layer.

## REQ status table

Per S5 (existing pub-API grandfather): every FFT function in
`cufft.rs` has a production consumer in `backend_impl.rs` (the cuda
backend's FFT dispatch arms). Those arms are reached from
`ferrotorch-core/src/fft.rs` when a tensor's FFT op routes to GPU.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn gpu_fft_c2c_f32 in cufft.rs` and `pub fn gpu_fft_c2c_f64 in cufft.rs` per upstream `aten/src/ATen/native/cuda/SpectralOps.cpp:445`. Non-test consumer: `backend_impl.rs` (f32) and `backend_impl.rs` (f64). |
| REQ-2 | SHIPPED | impl: `pub fn gpu_fft2_c2c_f32 in cufft.rs` and `pub fn gpu_fft2_c2c_f64 in cufft.rs`. Non-test consumer: `gpu_fft2_c2c_f64 in backend_impl.rs` (f32) and `backend_impl.rs` (f64). |
| REQ-3 | SHIPPED | impl: `pub fn gpu_fftn3d_c2c_f32 in cufft.rs` and `pub fn gpu_fftn3d_c2c_f64 in cufft.rs`. Non-test consumer: `gpu_fftn3d_c2c_f64 in backend_impl.rs` (f32) and `backend_impl.rs` (f64). |
| REQ-4 | SHIPPED | impl: `pub fn gpu_fftn2d_c2c_f32 in cufft.rs`, `pub fn gpu_fftn_axes_c2c_f32 in cufft.rs`, plus f64 mirrors. Non-test consumer: `gpu_fftn_axes_c2c_f32 in backend_impl.rs` (fftn2d_c2c_f32) and the axes-selected path called from `ferrotorch-core/src/fft.rs` ("gpu_fftn_axes_c2c_f32/f64" fallback). |
| REQ-5 | SHIPPED | impl: `pub fn gpu_rfft_r2c_f32 in cufft.rs` and `pub fn gpu_rfft_r2c_f64 in cufft.rs` per upstream `aten/src/ATen/native/cuda/SpectralOps.cpp:318`. Non-test consumer: `backend_impl.rs` (f32) and `backend_impl.rs` (f64). |
| REQ-6 | SHIPPED | impl: `pub fn gpu_irfft_c2r_f32 in cufft.rs` and `pub fn gpu_irfft_c2r_f64 in cufft.rs` per upstream `aten/src/ATen/native/cuda/SpectralOps.cpp:405`. Non-test consumer: `backend_impl.rs` (f32) and `backend_impl.rs` (f64). |
| REQ-7 | SHIPPED | impl: `pub fn gpu_hfft_f32 in cufft.rs`, `gpu_hfft_f64 in cufft.rs`, `gpu_ihfft_f32 in cufft.rs`, `gpu_ihfft_f64 in cufft.rs`. Non-test consumer: `gpu_ihfft_f64 in backend_impl.rs,4623,4636,4648`. `ferrotorch-core/src/fft.rs,1046` documents the dispatch path. |
| REQ-8 | SHIPPED | impl: every inverse path in `cufft.rs` issues a follow-up `1/n` normalisation kernel launch matching `aten/src/ATen/native/cuda/SpectralOps.cpp:298::_fft_apply_normalization_out`. Non-test consumer: every inverse-direction consumer in `backend_impl.rs` (e.g. `:4443` with `inverse = true`). |
| REQ-9 | SHIPPED | impl: every cuda function has a matching `#[cfg(not(feature = "cuda"))]` stub returning `Err(GpuError::NoCudaFeature)`. Non-test consumer: `backend_impl.rs` no-cuda compile path. |
