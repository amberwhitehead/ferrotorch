# FFT Operations

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/SpectralOps.cpp
  - torch/fft/__init__.py
-->

## Summary

`ferrotorch-core/src/fft.rs` implements the `torch.fft.*` family
declared in `torch/fft/__init__.py` and dispatched in
`aten/src/ATen/native/SpectralOps.cpp`. The module ships 1-D / 2-D /
N-D forward and inverse complex-to-complex, real-to-complex, and
Hermitian transforms (`fft`, `ifft`, `fft2`, `ifft2`, `fftn`,
`ifftn`, `rfft`, `irfft`, `rfftn`, `irfftn`, `hfft`, `ihfft`) plus
the frequency helpers (`fftfreq`, `rfftfreq`) and the spectrum
shifters (`fftshift`, `ifftshift`). The CPU path for every transform
delegates to [ferray_fft](https://crates.io/crates/ferray-fft), which
carries numpy's direction-dependent `norm` scaling and arbitrary-axis
transforms. The CPU transform path accepts **f32/f64 only** — `f16`/`bf16`
inputs are rejected to match `torch.fft.*`'s dtype contract (`promote_type_fft`
in `SpectralOps.cpp:82-90` does
`TORCH_CHECK(type == kFloat || type == kDouble, "Unsupported dtype ", type)`
on non-CUDA devices, so `torch.fft.fft(x.half())`/`.bfloat16()` on CPU raise
`RuntimeError: Unsupported dtype`; verified live vs torch 2.11). `half` FFT is
CUDA-only as a native complex-half transform (`torch/fft/__init__.py:49`), never
a CPU upcast to f32. The non-transform shifters (`fftshift`/`ifftshift`) stay
dtype-permissive, matching `torch.fft.fftshift`'s acceptance of half/bfloat16. As of #1294 every transform accepts `norm`
(`backward`/`forward`/`ortho` via the re-exported `FftNorm`) and `dim` / `s`
through the `*_norm` sibling of each public fn (e.g. `fft_norm(input, n, dim,
norm)`); the historical `fft(input, n)` / `fft2(input)` / `fftn(input, s,
axes)` signatures remain as default-arg wrappers. Complex values follow
PyTorch's "trailing dim of size 2" interleaved-real representation. GPU paths
exist for f32/f64 via cuFFT on the default last-axis / `norm="backward"` case
(#579 / #605 / #634 / #636).

## Requirements

- REQ-1: `fft(input, n)` — 1-D complex-to-complex FFT along the last
  dim. Input has shape `[..., n, 2]`. Mirrors `torch.fft.fft`
  (`torch/fft/__init__.py:fft`).
- REQ-2: `ifft(input, n)` — 1-D inverse complex-to-complex FFT.
  Mirrors `torch.fft.ifft`.
- REQ-3: `rfft(input, n)` / `irfft(input, n)` — real-to-complex and
  inverse. Output of `rfft` has shape `[..., n/2+1, 2]` (Hermitian
  redundancy stripped). Mirrors `torch.fft.rfft` / `torch.fft.irfft`.
- REQ-4: `fft2(input)` / `ifft2(input)` — 2-D C2C transforms over
  the last two non-complex dims. Mirrors `torch.fft.fft2` /
  `torch.fft.ifft2`.
- REQ-5: `fftn(input, s)` / `ifftn(input, s)` / `rfftn(input, s)` /
  `irfftn(input, s)` — N-D transforms with optional per-axis size
  spec. Delegate to `ferray_fft`. Mirrors `torch.fft.fftn` etc.
- REQ-6: `hfft(input, n)` / `ihfft(input, n)` — Hermitian FFT pair.
  Delegate to `ferray_fft`. Mirrors `torch.fft.hfft` /
  `torch.fft.ihfft`.
- REQ-7: Frequency helpers — `fftfreq(n, d)` returns the standard
  bin frequencies; `rfftfreq(n, d)` returns the non-redundant half.
  Both produce `f64` 1-D tensors. Mirrors `torch.fft.fftfreq` /
  `torch.fft.rfftfreq`.
- REQ-8: Spectrum shifters — `fftshift(input, dims)` / `ifftshift(input,
  dims)` cycle each axis by `dim_size/2` (floor) / `(dim_size+1)/2`
  respectively. Delegate to `ferray_fft`. Mirrors `torch.fft.fftshift`
  / `torch.fft.ifftshift`.
- REQ-9: GPU dispatch — `fft`/`ifft` route to cuFFT for f32/f64 on
  CUDA (`backend.fft_c2c_*` and `pad_truncate_complex_*` from #579 /
  #605) on the default last-axis / `norm="backward"` case; explicit
  `dim`/`norm`/`s` fall through to the ferray_fft CPU path. `f16`/`bf16`
  inputs are **rejected** on the CPU transform path (`reject_half_cpu_fft`),
  mirroring torch's `Unsupported dtype` `TORCH_CHECK`
  (`SpectralOps.cpp:88-90`); native CUDA complex-half lowering
  (`torch/fft/__init__.py:49`) remains an open prereq blocker (#1545).
- REQ-10: `norm` / `dim` / `s` parameters (#1294) — every transform exposes a
  `*_norm` sibling honouring `torch.fft.*`'s `norm`
  (`backward`/`forward`/`ortho`), `dim`, and (for 2-D/N-D) `s` kwargs. The
  norm string maps 1:1 onto the re-exported `FftNorm` (numpy's
  direction-dependent scaling, matching `SpectralOps.cpp:116-130` +
  `SpectralOpsUtils.h:15-19`); `dim`/`s` thread through ferray_fft's
  axis-aware transforms. Mirrors the Python signatures
  `fft(input, n, dim, norm)` / `fft2(input, s, dim, norm)` /
  `fftn(input, s, dim, norm)` (`torch/fft/__init__.py:36,132,246`).

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib fft` passes
  (the `tests` mod covers 1-D round-trips, rfft/irfft consistency,
  fft2, ifftn, hfft, shift helpers, and frequency arrays).
- [x] AC-2: `ifft(fft(x))` ≈ `x` within numerical tolerance for
  random `[..., n, 2]` inputs.
- [x] AC-3: `rfft` of an even-length real signal produces shape
  `[..., n/2+1, 2]`.
- [x] AC-4: CUDA f32/f64 `fft` paths route through `backend.fft_c2c_*`
  on the default last-axis / `norm="backward"` case (the cuFFT branch in
  `fft_norm` / `ifft_norm`, cuFFT, no host bounce).
- [x] AC-5: `f16`/`bf16` CPU dtype contract — the CPU transforms reject
  half-precision inputs with an `Unsupported dtype` error mirroring
  `SpectralOps.cpp:88-90` (verified live vs torch 2.11, which raises
  `RuntimeError: Unsupported dtype Half|BFloat16` on CPU), instead of the
  prior silent f64 upcast (#1545/#1536). Pinned by `fft.rs`
  `fft_f16_cpu_rejects_matching_torch_unsupported_dtype`,
  `fft_bf16_cpu_rejects_matching_torch_unsupported_dtype`,
  `rfft_f16_and_bf16_cpu_reject`, `nd_and_hermitian_transforms_reject_half`,
  and `fftshift_stays_dtype_permissive_for_half`.
- [ ] AC-5b: native CUDA complex-half (`chalf`) fft lowering — NOT-STARTED,
  blocked on #1545 (cuFFT complex-half kernels not yet wired; the CPU path
  rejects half by design, matching torch's CPU contract).
- [x] AC-6: `norm` / `dim` / `s` (#1294) — `fft_norm`/`ifft_norm`/… honour
  `ortho`/`forward` norm and arbitrary `dim`/`s`; verified by the host-side
  tests (`fft_ortho_norm_scales_dc_by_sqrt_n`, `fft_dim_transforms_named_axis`,
  `rfft_dim_transforms_named_axis`, `fftn_s_resizes_named_axes`) and the
  parity sweep (`grad_fns/fft.md` ACs, all 18 ops `0 skipped, 0 failed`).

## Architecture

Complex values are flat `[re, im, re, im, ...]` in the tensor buffer per
PyTorch's convention. The CPU path delegates to `ferray_fft` for every
transform; the bridge (`tensor_to_complex_array` / `tensor_to_real_array` /
`complex_array_to_tensor` / `real_array_to_tensor`) moves data between
ferrotorch's interleaved-`[..., 2]` layout and ferray's
`Array<Complex<f64>, IxDyn>` (stripping / appending the trailing complex pair),
computing the butterfly in f64.

Each public fn is a thin default-arg wrapper over a `*_norm` sibling threading
`norm` ([`FftNorm`]) and `dim` / `s`:

- `fft` → `fft_norm(input, n, dim, norm)`; `ifft` → `ifft_norm`. cuFFT
  (`backend.fft_c2c_{f32,f64}`, with `pad_truncate_complex_*` for `n !=
  input_n` per #605) handles the default last-axis / `norm="backward"` case;
  everything else goes through `ferray_fft::fft` / `ifft` (which take
  `axis: Option<isize>` + `norm: FftNorm`).
- `rfft` → `rfft_norm` / `irfft` → `irfft_norm` (real-to-complex pair;
  `ferray_fft::rfft` / `irfft` thread `axis` + `norm`). `rfft` output has
  shape `[..., n/2+1, 2]`.
- `fft2` → `fft2_norm` / `ifft2` → `ifft2_norm` (`ferray_fft::fft2` / `ifft2`
  with `s` / `axes` / `norm`; arbitrary-length `dim` lists are honoured since
  ferray's `fft2` accepts any `axes`).
- `fftn`/`ifftn`/`rfftn`/`irfftn` → their `*_norm` siblings delegating to
  `ferray_fft::fftn` / `ifftn` / `rfftn` / `irfftn`.
- `hfft` → `hfft_norm` / `ihfft` → `ihfft_norm`, and the 2-D / N-D Hermitian
  ops `hfft2`/`ihfft2`/`hfftn`/`ihfftn` → their `*_norm` siblings, all
  delegating to the matching `ferray_fft::h*` / `ih*` entry points.

The norm string→`FftNorm` mapping is `fft_norm_from_str` (`backward`→`Backward`,
`forward`→`Forward`, `ortho`→`Ortho`, unknown→`InvalidArgument`, mirroring
upstream `norm_from_string`'s `TORCH_CHECK`).

`fftfreq` / `rfftfreq` build `f64` 1-D tensors via `crate::creation::from_vec`.
`fftshift` / `ifftshift` delegate to `ferray_fft::fftshift` / `ifftshift`.

**Non-test consumer**: `crate::complex_tensor::ComplexTensor` (`ComplexTensor::
fft`/`ifft`/`fft2`/`ifft2`) delegates to `crate::fft::fft` / `ifft` / `fft2` /
`ifft2` (the default-arg wrappers, which in turn consume the `*_norm` path).
Re-exported in `lib.rs` as the top-level `ferrotorch_core::{fft, fft_norm,
ifft, …, FftNorm}` (and the `*_differentiable` / `*_differentiable_norm`
autograd wrappers from `grad_fns::fft`).

## Parity contract

`parity_ops = []` for *this* (forward-kernel) route. The 18 `fft.*`
parity-sweep ops are declared on the autograd-wrapper route — see
`.design/ferrotorch-core/grad_fns/fft.md` `## Parity contract`. As of
#1294 those 18 ops are wired end-to-end (oracle complex-dtype round-trip
+ runner `dispatch_fft` arm) and each verifies `K >= 1` passed / `0
failed` at `--seeds 8`; several forward kernels in *this* file (`hfft`,
`ihfft`, `rfft2`, `irfft2`, `hfft2`, `ihfft2`, `hfftn`, `ihfftn`) are the
production consumers the runner invokes directly (no autograd wrapper
exists for the six exotic ops yet). The numeric contract remains
byte-for-byte parity with `torch.fft.*` for f32/f64 inputs; additionally
cross-checked against the rustfft and ferray-fft reference
implementations in the unit tests.

## Verification

`cargo test -p ferrotorch-core --lib fft` exercises round-trip
identities (`ifft(fft(x)) ≈ x`), rfft/irfft consistency, fft2
consistency, hfft round-trip, fftshift cycle, and frequency-array
correctness.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `fft` at `fft.rs:108` mirrors `torch.fft.fft`; non-test consumer: `ComplexTensor::fft` at `complex_tensor.rs:329` invokes `crate::fft::fft(&interleaved, n)` |
| REQ-2 | SHIPPED | impl: `ifft` at `fft.rs:200`; non-test consumer: `ComplexTensor::ifft` at `complex_tensor.rs:336` invokes `crate::fft::ifft` |
| REQ-3 | SHIPPED | impl: `rfft`/`irfft` at `fft.rs:289,370`; non-test consumer: re-exported as `ferrotorch_core::rfft`/`irfft` at `lib.rs:154-155`, used in `ferrotorch_core::grad_fns::fft::rfft_differentiable` (the differentiable wrapper) at `grad_fns/fft.rs` |
| REQ-4 | SHIPPED | impl: `fft2`/`ifft2` at `fft.rs:472,516`; non-test consumer: `ComplexTensor::fft2` at `complex_tensor.rs:343`, `ComplexTensor::ifft2` at `complex_tensor.rs:350` |
| REQ-5 | SHIPPED | impl: `fftn`/`ifftn`/`rfftn`/`irfftn` at `fft.rs:1103,1234,1351,1382`; non-test consumer: re-exported at `lib.rs:153-155` as `ferrotorch_core::fftn`/`ifftn`/`rfftn`/`irfftn`. The N-D variants are the public surface |
| REQ-6 | SHIPPED | impl: `hfft`/`ihfft` at `fft.rs:1000,1048`; non-test consumer: re-exported as `ferrotorch_core::hfft`/`ihfft` at `lib.rs:154` |
| REQ-7 | SHIPPED | impl: `fftfreq`/`rfftfreq` at `fft.rs:1093,1105`; non-test consumer: re-exported as `ferrotorch_core::fftfreq`/`rfftfreq` at `lib.rs:153-155` |
| REQ-8 | SHIPPED | impl: `fftshift`/`ifftshift` at `fft.rs:1122,1141`; non-test consumer: re-exported as `ferrotorch_core::fftshift`/`ifftshift` at `lib.rs:153-155` |
| REQ-9 | SHIPPED | impl: cuFFT dispatch in `fft_norm`/`ifft_norm` + the CPU half-rejection guard `reject_half_cpu_fft` in `fft.rs` (mirrors `SpectralOps.cpp:88-90`); non-test consumer: `ComplexTensor::fft` in `complex_tensor.rs` (the f32/f64 cuFFT path) and the 18 `*_norm` transform entry points calling `reject_half_cpu_fft` (the CPU f16/bf16 rejection path). NB: native CUDA complex-half (`chalf`) lowering remains blocked on #1545; the CPU contract is now SHIPPED (rejects half like torch) |
