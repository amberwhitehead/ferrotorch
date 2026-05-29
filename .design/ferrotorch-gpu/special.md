# GPU orthogonal-polynomial special-function kernels (f32 / f64)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/Math.h
  - aten/src/ATen/native/cuda/chebyshev_polynomial_t.cu
  - aten/src/ATen/native/cuda/chebyshev_polynomial_u.cu
  - aten/src/ATen/native/cuda/chebyshev_polynomial_v.cu
  - aten/src/ATen/native/cuda/chebyshev_polynomial_w.cu
  - aten/src/ATen/native/cuda/hermite_polynomial_h.cu
  - aten/src/ATen/native/cuda/hermite_polynomial_he.cu
  - aten/src/ATen/native/cuda/laguerre_polynomial_l.cu
  - aten/src/ATen/native/cuda/legendre_polynomial_p.cu
  - aten/src/ATen/native/cuda/shifted_chebyshev_polynomial_t.cu
-->

## Summary

`ferrotorch-gpu/src/special.rs` implements the GPU forward path for the
orthogonal-polynomial families exposed by `ferrotorch_core::special`
(Chebyshev T/U/V/W and their shifted variants, physicist's and
probabilist's Hermite, Laguerre, Legendre). Each is a hand-written PTX
kernel loaded through `module_cache::get_or_compile`; one CUDA thread
evaluates the three-term recurrence per input element entirely in
registers, with NO host round-trip (R-CODE-4).

The kernels reproduce the **ferrotorch CPU recurrences** (the
`chebyshev_t` / `chebyshev_u` / `chebyshev_v` / `chebyshev_w` /
`hermite_h` / `hermite_he` / `laguerre_l` / `legendre_p` scalar
evaluators in `ferrotorch_core::special`) so the GPU result is
bit-for-relevant-tolerance identical to the CPU result. The upstream
recurrence body matches `aten/src/ATen/native/Math.h`
`chebyshev_polynomial_t_forward` (line 2861-2869) et al. — but the GPU
kernels deliberately omit torch's CUDA edge-case shortcuts (`|x| == 1`
closed forms, `cos(n*acos(x))` / `sin((n+1)*acos(x))` high-`n`
shortcuts, the `n < 0 -> 0` guard) because reproducing
them here would make the GPU path disagree with the ferrotorch CPU path.
That ferrotorch-CPU vs. torch-CUDA edge-case gap is a pre-existing
CPU-side divergence tracked separately (it is symmetric across both
backends today); the GPU kernel's contract is CPU/GPU agreement.

**Exception — the Hermite high-`n` NaN guard IS reproduced (#1641).** The
physicist's/probabilist's Hermite kernels (`HERMITE_H_F32_PTX` /
`HERMITE_H_F64_PTX` / `HERMITE_HE_F32_PTX` / `HERMITE_HE_F64_PTX`) replicate
torch's `getHermitianLimit<T>()` guard (`Math.h:3044-3052`,
`float -> 128`, `double -> 512`): for `n` above the limit they store
`quiet_NaN()` (`Math.h:3068-3070` / `:3109-3111`) instead of running the
overflowing recurrence. This is the one case where reproducing torch's NaN
short-circuit MAKES the backends agree rather than disagree, because the
ferrotorch CPU path now applies the same guard
(`hermitian_limit` in `ferrotorch-core/src/special.rs`); above the limit
CPU == GPU == torch == NaN.

## Requirements

- REQ-1 (chebyshev T/U/V/W + shifted, f32/f64): `pub fn gpu_chebyshev_poly_f32`
  / `gpu_chebyshev_poly_f64` taking `(input, n, seed_a, seed_b, shift,
  device)`. The kind is selected by the `q1` seed `seed_a*xx + seed_b`
  (T: 1,0; U: 2,0; V: 2,-1; W: 2,1) and `shift` chooses the shifted
  domain `xx = 2x - 1`. Recurrence `r = 2*xx*q - p`.
- REQ-2 (hermite H/He, f32/f64): `pub fn gpu_hermite_h_poly_f32`/`_f64`
  (`r = 2x*q - 2k*p`) and `gpu_hermite_he_poly_f32`/`_f64`
  (`r = x*q - k*p`), ABI `(input, n, device)`.
- REQ-3 (laguerre, f32/f64): `pub fn gpu_laguerre_poly_f32`/`_f64`
  (`r = ((2k+1-x)*q - k*p)/(k+1)`, seed `q1 = 1 - x`).
- REQ-4 (legendre, f32/f64): `pub fn gpu_legendre_poly_f32`/`_f64`
  (`r = ((2k+1)*x*q - k*p)/(k+1)`, seed `q1 = x`).
- REQ-5 (re-export + consumer wiring): the kernels are re-exported off
  the `ferrotorch-gpu` crate root; ferrotorch-core dispatches GPU
  polynomial calls through new `CudaBackendImpl` trait methods (one per
  family/dtype) which call into these kernels. The non-test production
  consumer is `ferrotorch_core::special::chebyshev_polynomial_t` (and
  siblings), which branch to the GPU backend when the input tensor is
  CUDA-resident.

## Acceptance Criteria

- [x] AC-1: 10 public launch fns exist with the documented signatures
  (chebyshev f32/f64; hermite_h f32/f64; hermite_he f32/f64; laguerre
  f32/f64; legendre f32/f64).
- [x] AC-2: 10 `pub(crate) const *_PTX` strings carry the recurrence ABIs.
- [x] AC-3: GPU-gated unit tests in `mod tests` assert the on-device
  result equals the ferrotorch CPU recurrence reference (copied verbatim
  into the test module) across `n in 0..=14` and a spread of `x`
  including the shifted domain. The result is a `CudaBuffer` (is_cuda by
  type) and the device ordinal is asserted, proving no CPU round-trip.
- [x] AC-4: Non-test consumers exist as `CudaBackendImpl` trait methods
  in `backend_impl.rs` and the GPU branch in `ferrotorch_core::special`.
- [x] AC-5: Each `unsafe { ... .launch(cfg) }` block carries a SAFETY
  comment documenting the ABI match, buffer residency/length, grid
  bounds guard, and u32 range check.

## Architecture

The chebyshev kernel folds all four kinds and the shifted variants into
one PTX entry per dtype via `(seed_a, seed_b, shift)` params, so the
core dispatcher passes `(1.0, 0.0, false)` for T, `(2.0, 0.0, false)`
for U, `(2.0, -1.0, false)` for V, `(2.0, 1.0, false)` for W, and the
same seeds with `shift = true` for the shifted variants. Hermite,
laguerre, and legendre each have their own `(in, out, n, total)` PTX
entry because their recurrence bodies differ structurally.

Dispatch path (the production consumer):
`ferrotorch_core::special::chebyshev_polynomial_t<T>` checks
`input.is_cuda()`; when CUDA-resident and `T` is f32/f64 it resolves the
global `GpuBackend` and calls `backend.chebyshev_polynomial_t_f32(handle,
n)` (or `_f64`), wrapping the returned handle into a CUDA-resident output
`Tensor<T>`. The CPU path (the `elementwise_f64` recurrence) is unchanged
for non-CUDA tensors. `require_cpu_poly`'s previous `NotImplementedOnCuda`
rejection is replaced by the GPU dispatch.

The `GpuBackend` trait gains 20 default-`Err` methods (10 families ×
{f32, f64}); the CUDA backend overrides all 20 to call into
`crate::special::*`. Other backends inherit the `InvalidArgument`
default and compile unchanged.

## Parity contract

`parity_ops = []` for this route (the polynomial parity ops are owned by
the `ferrotorch-core/src/special.rs` route at the core layer). These
kernels are device primitives the core ops consume; the GPU/CPU
agreement is enforced by the unit tests in this module.

Edge cases preserved (matching the ferrotorch CPU recurrence, NOT torch
CUDA shortcuts):

- `n == 0` → 1.0 for every family.
- `n == 1` → the family's `q1` seed.
- Empty buffer (`total == 0`) → length-0 buffer, no launch.
- `|x| == 1`, high `n`: evaluated by the plain recurrence (no closed-form
  shortcut), matching the ferrotorch CPU path bit-for-relevant-tolerance.
- u32 index overflow: `total > u32::MAX` rejected with `ShapeMismatch`.

## Verification

GPU-gated unit tests in `ferrotorch-gpu/src/special.rs` `mod tests`:

- `chebyshev_t_on_device_matches_cpu`
- `chebyshev_uvw_seeds_match_cpu`
- `shifted_chebyshev_t_matches_cpu`
- `hermite_h_on_device_matches_cpu`
- `hermite_he_on_device_matches_cpu`
- `laguerre_on_device_matches_cpu`
- `legendre_on_device_matches_cpu`
- `legendre_f64_matches_cpu_tight`

Each uses the `let Some(device) = dev() else { return }` graceful-skip
pattern. Smoke:

```bash
cargo test -p ferrotorch-gpu --features cuda special:: 2>&1 | tail -3
```

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn gpu_chebyshev_poly_f32` / `gpu_chebyshev_poly_f64 in ferrotorch-gpu/src/special.rs` mirroring the recurrence at upstream `aten/src/ATen/native/Math.h:2861`; non-test consumer: `CudaBackendImpl::chebyshev_polynomial_t_f32` … `shifted_chebyshev_polynomial_w_f64 in backend_impl.rs`, dispatched from the GPU branch of `chebyshev_polynomial_t in ferrotorch-core/src/special.rs`. |
| REQ-2 | SHIPPED | impl: `pub fn gpu_hermite_h_poly_f32`/`_f64`, `gpu_hermite_he_poly_f32`/`_f64 in special.rs` per upstream `Math.h:3072` / `Math.h:3113`; non-test consumer: `CudaBackendImpl::hermite_polynomial_h_f32`/`_f64`, `hermite_polynomial_he_f32`/`_f64 in backend_impl.rs` dispatched from `hermite_polynomial_h`/`hermite_polynomial_he in special.rs`. |
| REQ-3 | SHIPPED | impl: `pub fn gpu_laguerre_poly_f32`/`_f64 in special.rs` per upstream `Math.h:3149`; non-test consumer: `CudaBackendImpl::laguerre_polynomial_l_f32`/`_f64 in backend_impl.rs` dispatched from `laguerre_polynomial_l in special.rs`. |
| REQ-4 | SHIPPED | impl: `pub fn gpu_legendre_poly_f32`/`_f64 in special.rs` per upstream `Math.h:3189`; non-test consumer: `CudaBackendImpl::legendre_polynomial_p_f32`/`_f64 in backend_impl.rs` dispatched from `legendre_polynomial_p in special.rs`. |
| REQ-5 | SHIPPED | impl: `pub use special::{ gpu_chebyshev_poly_f32, … } in ferrotorch-gpu/src/lib.rs`; non-test consumer: the GPU branches of `chebyshev_polynomial_t` / `hermite_polynomial_h` / `laguerre_polynomial_l` / `legendre_polynomial_p` (and the shifted/UVW/he siblings) in `ferrotorch-core/src/special.rs`, dispatching through the `CudaBackendImpl` trait methods registered via `init_cuda_backend`. |
| REQ-6 (modified-Bessel-K family GPU f32: k0 / scaled-k0 / k1 / scaled-k1, #1651 batch 3b) | SHIPPED | impl: `pub fn gpu_modified_bessel_k0_f32` / `gpu_scaled_modified_bessel_k0_f32` / `gpu_modified_bessel_k1_f32` / `gpu_scaled_modified_bessel_k1_f32 in ferrotorch-gpu/src/special.rs` (carrying `K0_F32_PTX` / `SCALED_K0_F32_PTX` / `K1_F32_PTX` / `SCALED_K1_F32_PTX`), porting `modified_bessel_k0_forward` / `_k1_forward` + the scaled variants at upstream `aten/src/ATen/native/cuda/Math.cuh:2503-2577, 2582-2656, 2661-2736, 2740-2815`; non-test consumer: `CudaBackendImpl::modified_bessel_k0_f32` / `scaled_modified_bessel_k0_f32` / `modified_bessel_k1_f32` / `scaled_modified_bessel_k1_f32 in backend_impl.rs`, dispatched from the GPU branch (`special_gpu_simple`) of `modified_bessel_k0` / `scaled_modified_bessel_k0` / `modified_bessel_k1` / `scaled_modified_bessel_k1 in ferrotorch-core/src/special.rs`. f64 / bf16 / f16 CUDA cleanly reject `NotImplementedOnCuda`. Live-GPU verified by `verify_*_gpu_f32_on_device_matches_torch in ferrotorch-gpu/tests/divergence_modified_bessel_k_gpu_f32.rs` (RTX 3090, torch 2.11.0+cu130). |
