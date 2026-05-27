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
shifters (`fftshift`, `ifftshift`). The 1-D / 2-D paths run on
[rustfft](https://crates.io/crates/rustfft) directly; the N-D /
Hermitian / shift helpers delegate to
[ferray_fft](https://crates.io/crates/ferray-fft). Complex values
follow PyTorch's "trailing dim of size 2" interleaved-real
representation. GPU paths exist for f32/f64 via cuFFT (#579 / #605).

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
  #605). bf16/f16 take the CPU path via f64 round-trip (open prereq
  blocker #1545). N-D / Hermitian / shift helpers are CPU-only;
  CUDA inputs receive `Err(NotImplementedOnCuda)` from the underlying
  `ferray_fft` calls.

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib fft` passes
  (the `tests` mod covers 1-D round-trips, rfft/irfft consistency,
  fft2, ifftn, hfft, shift helpers, and frequency arrays).
- [x] AC-2: `ifft(fft(x))` ≈ `x` within numerical tolerance for
  random `[..., n, 2]` inputs.
- [x] AC-3: `rfft` of an even-length real signal produces shape
  `[..., n/2+1, 2]`.
- [x] AC-4: CUDA f32/f64 `fft` paths route through `backend.fft_c2c_*`
  per the dispatch at `fft.rs:139-169` (cuFFT, no host bounce).
- [ ] AC-5: bf16/f16 GPU fft path — NOT-STARTED, blocked on #1545
  (cuFFT only supports f32/f64; the implementation currently
  upcasts via `data_vec()` → CPU round-trip).

## Architecture

The module imports `rustfft::FftPlanner` (`fft.rs:25-26`) for the
1-D / 2-D paths and `ferray_fft::FftNorm` (`:24`) for the higher-
dimensional delegate path. Complex values are flat
`[re, im, re, im, ...]` in the tensor buffer per PyTorch's
convention.

`fft_1d_last_axis` at `fft.rs:66-94` plans the FFT once via
`FftPlanner::plan_fft_forward(n)` / `plan_fft_inverse(n)` and walks
each batch slice `data[b*n..(b+1)*n]` calling `fft.process` in place.
The inverse path applies `1/n` normalisation after the transform.

`fft` at `:108` validates trailing-dim-2 + ndim ≥ 2, computes the
batch size, and either dispatches to cuFFT (`backend.fft_c2c_{f32,f64}`
at `:160-164`, with optional `pad_truncate_complex_*` at `:151-158`
for n != input_n per #605) or builds a host-side
`Vec<Complex<f64>>`, calls `fft_1d_last_axis`, and writes back
`[re, im]` pairs cast via `complex_to_pairs::<T>` at `:51`.

`ifft` at `:200` is symmetric to `fft` — the inverse-transform plan
selector and the same cuFFT-vs-rustfft branching.

`rfft` at `:289` / `irfft` at `:370` handle the real-to-complex
path. The output of `rfft` has shape `[..., n/2+1, 2]`; the inverse
`irfft` accepts that shape and reproduces the real signal.

`fft2` / `ifft2` at `:472` / `:516` are 2-D variants — they run two
sequential 1-D FFTs (transpose + fft + transpose).

`fftn` / `ifftn` / `rfftn` / `irfftn` at `:735` / `:848` / `:948` /
`:967` delegate to `ferray_fft` via the bridge:
`ferray_fft::fftn(ferray_array, ...)` etc. The bridge constructs a
`ferray_core::Array<Complex<f64>, IxDyn>` from the tensor's host
buffer and casts the result back via `complex_to_pairs`. **GPU
inputs**: this delegate path requires CPU data, so CUDA tensors
take the host-bounce-and-back route — but only for the N-D /
Hermitian helpers, not the 1-D / 2-D core which have direct cuFFT
support.

`hfft` / `ihfft` at `:1000` / `:1048` cover the Hermitian-symmetric
FFT pair through `ferray_fft::hfft` / `ihfft`.

`fftfreq` / `rfftfreq` at `:1093` / `:1105` build `f64` 1-D tensors
via `crate::creation::from_vec`. `fftshift` / `ifftshift` at `:1122`
/ `:1141` delegate to `ferray_fft::fftshift` / `ifftshift`.

**Non-test consumer**: `crate::complex_tensor::ComplexTensor`
methods at `complex_tensor.rs:324-352` (`fft`/`ifft`/`fft2`/`ifft2`)
delegate to `crate::fft::fft` / `ifft` / `fft2` / `ifft2`
respectively. Re-exported at `lib.rs:153-156` as the top-level
`ferrotorch_core::{fft, fft2, ifft, ifft2, fftn, ifftn, rfft, irfft,
rfftn, irfftn, hfft, ihfft, fftfreq, rfftfreq, fftshift, ifftshift}`.

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
| REQ-5 | SHIPPED | impl: `fftn`/`ifftn`/`rfftn`/`irfftn` at `fft.rs:735,848,948,967`; non-test consumer: re-exported at `lib.rs:153-155` as `ferrotorch_core::fftn`/`ifftn`/`rfftn`/`irfftn`. The N-D variants are the public surface |
| REQ-6 | SHIPPED | impl: `hfft`/`ihfft` at `fft.rs:1000,1048`; non-test consumer: re-exported as `ferrotorch_core::hfft`/`ihfft` at `lib.rs:154` |
| REQ-7 | SHIPPED | impl: `fftfreq`/`rfftfreq` at `fft.rs:1093,1105`; non-test consumer: re-exported as `ferrotorch_core::fftfreq`/`rfftfreq` at `lib.rs:153-155` |
| REQ-8 | SHIPPED | impl: `fftshift`/`ifftshift` at `fft.rs:1122,1141`; non-test consumer: re-exported as `ferrotorch_core::fftshift`/`ifftshift` at `lib.rs:153-155` |
| REQ-9 | SHIPPED | impl: cuFFT dispatch at `fft.rs:139-169` (`fft`) and `fft.rs:229-258` (`ifft`); non-test consumer: `ComplexTensor::fft` at `complex_tensor.rs:329`. NB: bf16/f16 GPU path is blocked on #1545 — the f32/f64 SHIPPED claim does NOT cover those dtypes |
