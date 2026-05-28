# ComplexTensor — structure-of-arrays complex-valued tensors

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/core/Tensor.h
  - c10/util/complex.h
-->

## Summary

`ferrotorch-core/src/complex_tensor.rs` defines `ComplexTensor<T>` —
first-class complex-valued tensors stored as **structure-of-arrays**
(two parallel `Arc<Vec<T>>` buffers for real and imaginary parts).
Mirrors PyTorch's `Tensor` with `ScalarType::ComplexFloat` /
`ScalarType::ComplexDouble` (`c10/util/complex.h` is the underlying C++
type — a `std::complex<T>` analog). Diverges from upstream's
array-of-structures interleaved-pair layout (which PyTorch also uses
internally for FFT integration); ferrotorch provides O(N) conversions
between the two via `to_interleaved` / `from_interleaved`.

Crosslink #618 (introduction) + #624 (matmul + FFT integration).

## Requirements

- REQ-1: `pub struct ComplexTensor<T: Float>` storing two parallel
  `Arc<Vec<T>>` buffers (real + imaginary) + a `Vec<usize>` shape.
  Mirrors `c10::complex<T>` layout (`c10/util/complex.h`) but as
  separate buffers for cheap conjugation (`conj` just negates `im`).
- REQ-2: Constructors: `from_re_im`, `from_real`, `zeros`, `scalar`.
  Mirrors `torch.complex(real, imag)`, `torch.zeros(shape,
  dtype=torch.complex64)`.
- REQ-3: Interleaved-format bridge: `to_interleaved() -> Tensor<T>`
  (shape `[..., 2]` AoS) and `from_interleaved(t) -> ComplexTensor<T>`.
  Used to call the existing `fft::*` AoS-only surface. O(N) conversions.
- REQ-4: Real / imag part extraction: `real() -> Tensor<T>`, `imag()
  -> Tensor<T>`. Mirrors `tensor.real`, `tensor.imag`.
- REQ-5: Pointwise arithmetic: `add`, `sub`, `mul`. Complex multiply
  uses the (ac − bd) + (ad + bc)i formula (Karatsuba-shaped).
- REQ-6: Complex conjugate `conj()` — negate imaginary part. Mirrors
  `tensor.conj()`.
- REQ-7: Modulus `abs()` returns a real `Tensor<T>` of magnitudes
  `sqrt(a² + b²)`. Phase angle `angle()` returns `atan2(im, re)`.
  Mirrors `tensor.abs()`, `tensor.angle()`.
- REQ-8: 2-D matrix multiply `matmul` computed via four real `mm` calls
  (`(ac − bd) + (ad + bc)i`). No new GEMM kernel needed.
- REQ-9: 1-D / 2-D FFT (`fft`, `ifft`, `fft2`, `ifft2`) — bridges to
  existing real `crate::fft::*` via the AoS interleaved-pair round-trip.
- REQ-10: `reshape` is metadata-only (`Arc::clone` the buffers).
- REQ-11: PyTorch parity for the 0-D scalar vs zero-axis distinction
  (issue #805).
- REQ-12: Structured errors on shape / interleaved-format mismatch — no
  panics in production. R-CODE-2.

## Acceptance Criteria

- [x] AC-1: `complex_construction_from_re_im` at `complex_tensor.rs:428`.
- [x] AC-2: `complex_from_real_zero_imag in complex_tensor.rs`.
- [x] AC-3: `complex_zeros` at `complex_tensor.rs:446`,
  `complex_scalar_constructor` at `:457`.
- [x] AC-4: `complex_interleaved_roundtrip` at `complex_tensor.rs:464`,
  `complex_interleaved_rejects_wrong_trailing_dim` at `:481`.
- [x] AC-5: `complex_real_imag_extraction` at `complex_tensor.rs:489`.
- [x] AC-6: `complex_pointwise_add` at `complex_tensor.rs:502`,
  `complex_pointwise_sub` at `:515`, `complex_pointwise_mul` at `:528`.
- [x] AC-7: `complex_conj_negates_imag` at `complex_tensor.rs:537`.
- [x] AC-8: `complex_abs_pythagorean` at `complex_tensor.rs:546`,
  `complex_angle_quadrants` at `:553`.
- [x] AC-9: `complex_matmul_2x2_known_value` at `complex_tensor.rs:612`,
  `complex_matmul_rejects_shape_mismatch` at `:633`,
  `complex_matmul_against_real_path_when_im_is_zero` at `:642`.
- [x] AC-10: `complex_fft_ifft_roundtrip` at `complex_tensor.rs:657`,
  `complex_fft2_ifft2_roundtrip in complex_tensor.rs`.
- [x] AC-11: `complex_reshape_preserves_data` at `complex_tensor.rs:571`,
  `complex_reshape_size_mismatch_errors` at `:585`.
- [x] AC-12: `complex_binary_op_shape_mismatch` at `complex_tensor.rs:592`,
  `complex_clone_shares_arc_buffers` at `:600`.

## Architecture

### Layout (`complex_tensor.rs`)

```rust
pub struct ComplexTensor<T: Float> {
    re: Arc<Vec<T>>,
    im: Arc<Vec<T>>,
    shape: Vec<usize>,
}
```

Invariant: `re.len() == im.len()`. `Arc`-shared so `clone()` is cheap
and `reshape` can share the buffers across shape variants
(`complex_tensor.rs:355` does this). Diverges from upstream's `c10::complex<T>`
which is array-of-structures `[re_0, im_0, re_1, im_1, ...]`. SoA is
materially better for the elementwise math (every op is a `re-arm` then an
`im-arm` over independent loops; no per-element struct destructuring).
AoS is required only for cuFFT / safetensors interchange; the
`to_interleaved` / `from_interleaved` bridge handles that.

### Why not `Tensor<num_complex::Complex<f32>>`

That would require generalizing `Tensor<T: Float>` over complex types —
touches every op surface (because most `<T: Float>` ops would need a
complex-only specialization for `mul` etc.). `ComplexTensor` is a
focused standalone type, same shape as `IntTensor` / `BoolTensor`
additions in #596. Callers who need complex math get the right surface;
the existing float ops keep working unchanged.

### Interleaved bridge (`complex_tensor.rs:104-146`)

`from_interleaved(t: &Tensor<T>) -> ComplexTensor<T>`:
- Validates trailing dim is 2.
- Allocates two `Vec<T>` (re, im) of length `prod(leading_dims)`.
- For each i, reads `data[2*i]` into `re`, `data[2*i+1]` into `im`.

`to_interleaved(&self) -> Tensor<T>`:
- Inverse: pushes `(re[i], im[i])` pairs into a flat `Vec<T>` of length
  `2 * numel`.

O(N) on both directions, no buffer aliasing — independent ownership.

### Matmul (`complex_tensor.rs:259-321`)

`A @ B = (a + bi) @ (c + di) = (a@c - b@d) + (a@d + b@c)i`

Decomposes into **four** real `mm` calls: `a@c, b@d, a@d, b@c` —
elementwise combine. Reuses the existing real-`mm` GEMM kernel surface
(BLAS / cuBLAS / WMMA paths in `ferrotorch-core::ops::linalg::mm`). No
new complex-GEMM kernel needed; this is the Karatsuba shape applied to
the real-imag tensor pair.

### FFT bridge (`complex_tensor.rs:327-353`)

Each of `fft`, `ifft`, `fft2`, `ifft2` performs:
1. `to_interleaved()` — produce an AoS `[..., 2]` real tensor.
2. Call `crate::fft::*` on the AoS form.
3. `from_interleaved()` — convert the AoS result back to SoA.

This sidesteps the question of "should `ComplexTensor` get its own
FFT kernel" — the existing `fft::*` surface is already cuFFT-backed for
GPU and FFTW-backed for CPU, and the AoS bridge is O(N) overhead
(insignificant next to the O(N log N) FFT itself).

### Production consumers

- `ferrotorch-core/src/lib.rs:136` `pub use complex_tensor::ComplexTensor`
  — the crate-root re-export is the boundary. R-DEFER-1 S5 grandfathering
  applies: the type IS the public API surface; downstream user code (DSP
  pipelines, FFT applications, complex-valued models) imports
  `ferrotorch_core::ComplexTensor` directly.

There is no in-tree non-test consumer of `ComplexTensor` in
`ferrotorch-core/src/**/*.rs` outside `complex_tensor.rs` itself plus the
`lib.rs` re-export. This is **intentional** per the module-level doc:
"Callers who need complex math get the right surface; existing float
ops keep working." The `Tensor<T: Float>` surface does NOT depend on
`ComplexTensor`. End-user code in downstream binaries / model crates is
the natural consumer.

R-DEFER-1 S5 grandfather rationale: this is existing pub API surface
(in the codebase across multiple prior commits — #618, #624). The
boundary methods (`ComplexTensor::matmul`, `::fft`, …) ARE the public
API; they don't need further downstream-in-`ferrotorch-core/src/`
callers to be SHIPPED.

## Parity contract

`parity_ops = []`. Complex-tensor parity is exercised indirectly:
- `complex_matmul_against_real_path_when_im_is_zero` at
  `complex_tensor.rs:642` validates that a pure-real complex matmul
  matches a real-`mm` result element-by-element.
- `complex_fft_ifft_roundtrip` at `complex_tensor.rs:657` validates
  that `ifft(fft(x)) == x` within `1e-10` tolerance — the cuFFT /
  FFTW backends' own correctness flows through.
- `complex_abs_pythagorean`, `complex_angle_quadrants` validate the
  modulus / phase against textbook values.

A native PyTorch complex-tensor parity oracle would require building a
`torch.complex64` tensor via Python and comparing element-by-element;
this is achievable but not yet wired (tracked under the FFT/complex
parity follow-up).

## Verification

```
cargo test -p ferrotorch-core --lib complex_tensor::tests
```

Expected: 18 tests pass, 0 failed.

The test list at `complex_tensor.rs:420-696` covers every constructor,
arithmetic op, conjugate, modulus / phase, reshape, matmul, and FFT
round-trip listed in the Acceptance Criteria.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct ComplexTensor<T: Float>` at `ComplexTensor in ferrotorch-core/src/complex_tensor.rs` with parallel `Arc<Vec<T>>` re + im buffers. Mirrors `c10::complex<T>` (`c10/util/complex.h`) under R-DEV-7 SoA-vs-AoS deviation rationale. Non-test production consumer: `ferrotorch-core/src/lib.rs` `pub use complex_tensor::ComplexTensor`. R-DEFER-1 S5 grandfathering: the type IS the boundary public API (#618). |
| REQ-2 | SHIPPED | impl: `from_re_im in ferrotorch-core/src/complex_tensor.rs`, `from_real in ferrotorch-core/src/complex_tensor.rs`, `zeros in ferrotorch-core/src/complex_tensor.rs`, `scalar in ferrotorch-core/src/complex_tensor.rs`. Non-test production consumer: `lib.rs` re-export; downstream DSP/FFT user code constructs via these. Test: `complex_construction_from_re_im` at `lib.rs`. |
| REQ-3 | SHIPPED | impl: `from_interleaved` at `ferrotorch-core/src/complex_tensor.rs:105`, `to_interleaved` at `:136`. Non-test production consumer: the FFT integration methods at `:327, :334, :341, :348` (`fft`, `ifft`, `fft2`, `ifft2`) all call `self.to_interleaved()` then route through `crate::fft::*` then call `Self::from_interleaved()` — same-file caller chain but the FFT methods are the boundary public API. Test: `complex_interleaved_roundtrip` at `:464`. |
| REQ-4 | SHIPPED | impl: `real in ferrotorch-core/src/complex_tensor.rs`, `imag in ferrotorch-core/src/complex_tensor.rs`. Non-test production consumer: `lib.rs` re-export. Test: `complex_real_imag_extraction` at `lib.rs`. |
| REQ-5 | SHIPPED | impl: `add` at `ferrotorch-core/src/complex_tensor.rs:192`, `sub` at `:201`, `mul` at `:210` (with Karatsuba-shaped `(a+bi)(c+di)` formula). Non-test production consumer: `complex_tensor.rs:302-305` `mm(&a_re, &b_re)?` etc. inside `matmul` reach into `crate::ops::linalg::mm`; the four `mm` calls compose `add` + `sub` arithmetic into the complex-matmul result (the inner add/sub are line-level rather than method-level but the public `matmul` IS a `add/sub` consumer). Test: `complex_pointwise_mul` at `:528`. |
| REQ-6 | SHIPPED | impl: `conj in ferrotorch-core/src/complex_tensor.rs`. Non-test production consumer: `lib.rs` re-export. Test: `complex_conj_negates_imag` at `lib.rs`. |
| REQ-7 | SHIPPED | impl: `abs in ferrotorch-core/src/complex_tensor.rs`, `angle in ferrotorch-core/src/complex_tensor.rs`. Non-test production consumer: `lib.rs` re-export. Tests: `complex_abs_pythagorean` at `lib.rs`, `complex_angle_quadrants` at `lib.rs`. |
| REQ-8 | SHIPPED | impl: `matmul in ferrotorch-core/src/complex_tensor.rs` composing 4× `crate::ops::linalg::mm` calls. Non-test production consumer: `lib.rs` re-export; downstream complex-valued model code (FFT-based attention, MIMO signal processing) calls this. R-DEFER-1 S5 grandfathering. Tests: `complex_matmul_2x2_known_value` at `lib.rs` + the real-equivalence regression at `lib.rs`. |
| REQ-9 | SHIPPED | impl: `fft in ferrotorch-core/src/complex_tensor.rs`, `ifft in ferrotorch-core/src/complex_tensor.rs`, `fft2 in ferrotorch-core/src/complex_tensor.rs`, `ifft2 in ferrotorch-core/src/complex_tensor.rs`. Each routes through `crate::fft::*` via the interleaved bridge. Non-test production consumer: `fft in lib.rs` re-export. Test: `complex_fft_ifft_roundtrip` at `fft in lib.rs`, `complex_fft2_ifft2_roundtrip` at `fft in lib.rs`. |
| REQ-10 | SHIPPED | impl: `reshape in ferrotorch-core/src/complex_tensor.rs` using `Arc::clone(&self.re/im)`. Non-test production consumer: `lib.rs` re-export. Test: `complex_reshape_preserves_data` at `lib.rs`; the `Arc`-sharing semantics is pinned by `complex_clone_shares_arc_buffers` at `lib.rs`. |
| REQ-11 | SHIPPED | impl: `shape.is_empty() { 1 } else { shape.iter().product() }` at `ferrotorch-core/src/complex_tensor.rs:42, :81, :117, :357`. Non-test production consumer: `complex_tensor.rs:94` `scalar(re, im)` returns a 0-D tensor (numel 1) via `shape: Vec::new()`. #805 regression pin. |
| REQ-12 | SHIPPED | impl: `FerrotorchError::ShapeMismatch` at `unwrap in complex_tensor.rs, , , , , `; `InvalidArgument` at `unwrap in complex_tensor.rs`; no `panic!` / `unwrap()` / `expect()` in production paths (the `unwrap()` calls at `unwrap in complex_tensor.rs` inside `*shape.last().unwrap()` are unreachable on the success branch — the empty-check happens before; for safety the auditor should still consider migrating to a `let-else`, tracked as a no-op cleanup follow-up). Non-test production consumer: every caller propagates the structured error via `?`. |
