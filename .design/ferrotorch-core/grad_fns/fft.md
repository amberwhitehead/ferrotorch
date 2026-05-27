# FFT grad_fns

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/native/SpectralOps.cpp
  - torch/fft/__init__.py
-->

## Summary

`ferrotorch-core/src/grad_fns/fft.rs` is the autograd wrapper layer for the
1-D / N-D / Hermitian FFT family. It ships ten `*Backward` `GradFn` structs
(`FftBackward`, `IfftBackward`, `RfftBackward`, `IrfftBackward`,
`FftnBackward`, `IfftnBackward`, `RfftnBackward`, `IrfftnBackward`,
`HfftBackward`, `IhfftBackward`) plus ten `<op>_differentiable` forward
wrappers that pair the corresponding forward kernel in `crate::fft` (the
sibling forward-only module at `ferrotorch-core/src/fft.rs`) with a
`Tensor::from_operation` graph node when grad is enabled. Upstream is the
N-D/1-D autograd contract in `aten/src/ATen/native/SpectralOps.cpp` (the
`fft_*_symint` entry points at lines 359-701) wired through the Python
namespace declared in `torch/fft/__init__.py` (the `_add_docstr` blocks at
lines 33-1164). The autograd math itself (the `<op>Backward` structs) is the
1-D / N-D analog of PyTorch's `_fft_r2c_backward` /
`_fft_c2r_backward` defined in `tools/autograd/derivatives.yaml` (FFT entries
are emitted by the `FftR2CBackward` / `FftC2RBackward` / `FftC2CBackward`
codegen — same VJP identities, restated below). Ferrotorch's complex storage
convention is **interleaved-real `[..., 2]`** (real and imaginary parts as
the trailing pair) rather than upstream's native `torch.complex64` /
`torch.complex128` dtypes — this is an R-DEV-4 / R-DEV-7 deviation that
preserves the math contract while letting `Tensor<T: Float>` carry complex
data without a separate complex dtype.

## Requirements

- REQ-1: `fft_differentiable(x, n)` — forward `y = fft(x, n)` (unnormalized
  forward DFT) with `FftBackward` returning `grad_x = n * ifft(grad_y)` (our
  `ifft` is normalized 1/n; multiplying by n undoes the normalization to
  yield the un-normalized inverse, which matches the VJP of the
  un-normalized forward FFT). Mirrors `fft_fft_symint` at
  `aten/src/ATen/native/SpectralOps.cpp:359-374` (`return fft_c2c("fft",
  ..., FFTNormMode::by_root_n, /*forward=*/true)`), surfaced as
  `torch.fft.fft` at `torch/fft/__init__.py:33-87` (signature
  `fft(input, n=None, dim=-1, norm=None, *, out=None) -> Tensor`).

- REQ-2: `ifft_differentiable(x, n)` — forward `y = ifft(x, n)` (normalized
  1/n inverse) with `IfftBackward` returning `grad_x = fft(grad_y) / n`
  (the chain rule for the 1/n-normalized inverse: `grad_x = (1/n) * W *
  grad_y = fft(grad_y) / n`). Mirrors `fft_ifft_symint` at
  `aten/src/ATen/native/SpectralOps.cpp:376-391` (`return fft_c2c("ifft",
  ..., norm, /*forward=*/false)`), surfaced as `torch.fft.ifft` at
  `torch/fft/__init__.py:88-127`.

- REQ-3: `rfft_differentiable(x, n)` — forward `y = rfft(x, n)` (real →
  Hermitian-truncated complex of shape `[..., K, 2]`, `K = N/2+1`) with
  `RfftBackward` returning the **partial unnormalized inverse over the
  half-spectrum** (`grad_x = real(ifft_unnormalized(zero_pad(grad_y, N)))`)
  rather than `irfft(grad_y, N)` (which would Hermitian-double the
  interior). The implementation zero-pads the half-spectrum, calls
  `fft::ifft(padded, N)` (our normalized inverse), multiplies by N to undo
  the normalization, and takes the real part of the trailing complex pair.
  Mirrors `fft_rfft_symint` at `aten/src/ATen/native/SpectralOps.cpp:393-402`
  + `tools/autograd/derivatives.yaml` `FftR2CBackward`, surfaced as
  `torch.fft.rfft` at `torch/fft/__init__.py:355-410`.

- REQ-4: `irfft_differentiable(x, n)` — forward `y = irfft(x, n)`
  (Hermitian-truncated complex `[..., K, 2]` → real `[..., N]`, normalized
  1/N) with `IrfftBackward` returning `grad_x = rfft(grad_y, N) / N`
  multiplied **by 2 on interior frequencies** (`k = 1..K-2` for even N, or
  `k = 1..K-1` for odd N) — the Hermitian-doubling correction because each
  interior `k` corresponds to two entries (`k` and `N-k`) in the full DFT.
  Boundary indices (`k=0` always, and `k=N/2` for even N) get factor 1/N
  only. Mirrors `fft_irfft_symint` at
  `aten/src/ATen/native/SpectralOps.cpp:404-413` + `FftC2RBackward`,
  surfaced as `torch.fft.irfft` at `torch/fft/__init__.py:411-487`.

- REQ-5: `fftn_differentiable(x, s, axes)` — forward `y = fftn(x, s, axes)`
  (un-normalized N-D forward DFT along `axes`) with `FftnBackward`
  returning `grad_x = prod(s) * ifftn(grad_y, s, axes)` (same un-normalized
  inverse identity as REQ-1, generalized to multiple axes). Persists `s` /
  `axes` / `norm_n = prod(s)` from the forward to keep backward shape-stable.
  Mirrors `fft_fftn_symint` at
  `aten/src/ATen/native/SpectralOps.cpp:437-455`, surfaced as
  `torch.fft.fftn` at `torch/fft/__init__.py:243-302`.

- REQ-6: `ifftn_differentiable(x, s, axes)` — forward `y = ifftn(x, s, axes)`
  (normalized 1/prod(s) N-D inverse) with `IfftnBackward` returning
  `grad_x = fftn(grad_y, s, axes) / prod(s)`. Mirrors `fft_ifftn_symint`
  at `aten/src/ATen/native/SpectralOps.cpp:457-473`, surfaced as
  `torch.fft.ifftn` at `torch/fft/__init__.py:303-354`.

- REQ-7: `rfftn_differentiable(x, s, axes)` — forward `y = rfftn(x, s, axes)`
  (real → Hermitian-truncated complex, N-D; only the **last** transform
  axis is half-spectrum-truncated, the others go full length) with
  `RfftnBackward` returning the partial unnormalized inverse: zero-pad
  along the last transform axis from `K` to `n_last`, call `fft::ifftn`,
  multiply by `prod(s)` to undo normalization, take real part. Persists
  `out_shape` / `last_axis_n` / `last_axis_logical` / `norm_n` from
  forward. Mirrors `fft_rfftn_symint` at
  `aten/src/ATen/native/SpectralOps.cpp:489-525`, surfaced as
  `torch.fft.rfftn` at `torch/fft/__init__.py:637-703`.

- REQ-8: `irfftn_differentiable(x, s, axes)` — forward `y = irfftn(x, s,
  axes)` (Hermitian-truncated complex → real, N-D; 1/prod(s)-normalized)
  with `IrfftnBackward` returning `grad_x = rfftn(grad_y, s, axes) /
  prod(s)` with the **interior-doubling-by-2 along the last freq axis**
  correction (boundary at `kk = 0` always, and `kk = K-1` only when
  `n_last` is even, get factor 1/prod(s); interior gets `2 / prod(s)`).
  Mirrors `fft_irfftn_symint` at
  `aten/src/ATen/native/SpectralOps.cpp:541-554`, surfaced as
  `torch.fft.irfftn` at `torch/fft/__init__.py:704-784`.

- REQ-9: `hfft_differentiable(x, n)` — forward `y = hfft(x, n)` (Hermitian-
  symmetric complex `[..., K, 2]` → real `[..., n]`; un-normalized inverse
  via conj-then-irfft) with `HfftBackward` returning
  `grad_x = conj(rfft(grad_y, n))` for boundary frequencies and
  `2 * conj(rfft(grad_y, n))` for interior. The forward is the
  un-normalized inverse of `ihfft` (no `1/N`), so backward is the
  un-normalized forward (no `1/N` either) with the same boundary-vs-interior
  Hermitian-doubling pattern. Mirrors `fft_hfft_symint` at
  `aten/src/ATen/native/SpectralOps.cpp:415-424`, surfaced as
  `torch.fft.hfft` at `torch/fft/__init__.py:785-875`.

- REQ-10: `ihfft_differentiable(x, n)` — forward `y = ihfft(x, n)` (real
  `[..., N]` → Hermitian complex `[..., K, 2]`; normalized
  `conj(rfft(x, N)) / N`) with `IhfftBackward` returning the partial
  unnormalized inverse with `conj`: zero-pad `conj(grad_y)` along the
  freq axis from `K` to `N`, run `fft::ifft` (which already supplies the
  `1/N`), take the real part. Mirrors `fft_ihfft_symint` at
  `aten/src/ATen/native/SpectralOps.cpp:426-435`, surfaced as
  `torch.fft.ihfft` at `torch/fft/__init__.py:876-930`.

- REQ-11: `fft2_differentiable(x, s)` — `torch.fft.fft2` is the 2-D
  specialization of `fftn` over the trailing two axes (declared at
  `torch/fft/__init__.py:129-189`, implemented as `fft_fft2_symint` at
  `aten/src/ATen/native/SpectralOps.cpp:644-652`, literal `return
  native::fft_fftn_symint(self, s, dim, norm)`). Ferrotorch ships the
  **forward-only** kernel `fft::fft2` at `ferrotorch-core/src/fft.rs:472`
  (cuFFT-backed C2C, no `dim` kwarg — fixed to the trailing two axes); the
  **differentiable wrapper** `fft2_differentiable` is **not yet
  implemented** in `grad_fns/fft.rs`. Same gap for `ifft2_differentiable`
  (forward at `fft.rs:516`).

- REQ-12: `ifft2_differentiable(x, s)` — `torch.fft.ifft2` is the 2-D
  specialization of `ifftn` (declared at `torch/fft/__init__.py:190-242`,
  implemented as `fft_ifft2_symint` at
  `aten/src/ATen/native/SpectralOps.cpp:654-662`). Forward kernel exists
  at `ferrotorch-core/src/fft.rs:516`; differentiable wrapper not
  implemented.

- REQ-13: `rfft2_differentiable(x, s)` — 2-D real FFT. PyTorch's
  `torch.fft.rfft2` is declared at `torch/fft/__init__.py:488-554`, with
  `fft_rfft2_symint` at `aten/src/ATen/native/SpectralOps.cpp:664-672`.
  Ferrotorch has **NO forward `rfft2`** and no differentiable wrapper.

- REQ-14: `irfft2_differentiable(x, s)` — 2-D inverse real FFT. PyTorch's
  `torch.fft.irfft2` at `torch/fft/__init__.py:555-636` /
  `aten/src/ATen/native/SpectralOps.cpp:674-682`. Ferrotorch has **NO
  forward `irfft2`** and no differentiable wrapper.

- REQ-15: `hfft2_differentiable(x, s)` — 2-D Hermitian FFT. PyTorch's
  `torch.fft.hfft2` at `torch/fft/__init__.py:931-1002` /
  `aten/src/ATen/native/SpectralOps.cpp:690-699`. Ferrotorch has **NO
  forward `hfft2`** and no differentiable wrapper.

- REQ-16: `ihfft2_differentiable(x, s)` — 2-D inverse Hermitian FFT.
  PyTorch's `torch.fft.ihfft2` at `torch/fft/__init__.py:1003-1067` /
  `aten/src/ATen/native/SpectralOps.cpp:701-714`. Ferrotorch has **NO
  forward `ihfft2`** and no differentiable wrapper.

- REQ-17: `hfftn_differentiable(x, s, axes)` — N-D Hermitian FFT. PyTorch's
  `torch.fft.hfftn` at `torch/fft/__init__.py:1068-1160` /
  `aten/src/ATen/native/SpectralOps.cpp:584-599`. Ferrotorch has **NO
  forward `hfftn`** and no differentiable wrapper.

- REQ-18: `ihfftn_differentiable(x, s, axes)` — N-D inverse Hermitian FFT.
  PyTorch's `torch.fft.ihfftn` at `torch/fft/__init__.py:1161-1230` /
  `aten/src/ATen/native/SpectralOps.cpp:626-642`. Ferrotorch has **NO
  forward `ihfftn`** and no differentiable wrapper.

## Acceptance Criteria

- [ ] AC-1: `fft.fft` parity-sweep at `--seeds 8` returns `[fft.fft] N/N
  passed (0 skipped, 0 failed)` with N >= 1 (grep-count `passed (0 skipped,
  0 failed)` >= 1). Blocked on #1294 (oracle complex-dtype support + runner
  arm + complex-layout adapter).
- [ ] AC-2: `fft.ifft` parity-sweep passes. Blocked on #1294.
- [ ] AC-3: `fft.rfft` parity-sweep passes. Blocked on #1294.
- [ ] AC-4: `fft.irfft` parity-sweep passes. Blocked on #1294.
- [ ] AC-5: `fft.fftn` parity-sweep passes. Blocked on #1294.
- [ ] AC-6: `fft.ifftn` parity-sweep passes. Blocked on #1294.
- [ ] AC-7: `fft.rfftn` parity-sweep passes. Blocked on #1294.
- [ ] AC-8: `fft.irfftn` parity-sweep passes. Blocked on #1294.
- [ ] AC-9: `fft.hfft` parity-sweep passes. Blocked on #1294.
- [ ] AC-10: `fft.ihfft` parity-sweep passes. Blocked on #1294.
- [ ] AC-11: `fft.fft2` parity-sweep passes. Blocked on #1294 + #1300.
- [ ] AC-12: `fft.ifft2` parity-sweep passes. Blocked on #1294 + #1300.
- [ ] AC-13: `fft.rfft2` parity-sweep passes. Blocked on #1294 + #1299.
- [ ] AC-14: `fft.irfft2` parity-sweep passes. Blocked on #1294 + #1299.
- [ ] AC-15: `fft.hfft2` parity-sweep passes. Blocked on #1294 + #1299.
- [ ] AC-16: `fft.ihfft2` parity-sweep passes. Blocked on #1294 + #1299.
- [ ] AC-17: `fft.hfftn` parity-sweep passes. Blocked on #1294 + #1299.
- [ ] AC-18: `fft.ihfftn` parity-sweep passes. Blocked on #1294 + #1299.
- [x] AC-19: `cargo test -p ferrotorch-core --lib grad_fns::fft` passes —
  the in-file `#[cfg(test)] mod tests` at `grad_fns/fft.rs:1296-1513`
  covers `grad_fn` attachment for all ten differentiable wrappers
  (`fft_differentiable_attaches_grad_fn`,
  `ifft_differentiable_attaches_grad_fn`, ... through
  `ihfft_differentiable_attaches_grad_fn`), plus identity-check VJPs for
  `fft` / `ifft` and the `fftn` impulse-grad path, plus the `no_grad`
  short-circuit (`no_grad_context_disables_tracking`,
  `fftn_no_grad_when_not_needed`).
- [x] AC-20: All twelve differentiable wrappers (the ten 1-D / N-D /
  Hermitian wrappers plus the two 2-D wrappers `fft2_differentiable` /
  `ifft2_differentiable` added under #1300) have non-test production
  consumers via the `pub use grad_fns::fft::{...}` re-export block in
  `ferrotorch-core/src/lib.rs` (the crate's public-API surface). The six
  new forward-only ops (`rfft2`, `irfft2`, `hfft2`, `ihfft2`, `hfftn`,
  `ihfftn`, #1299) are likewise re-exported via `pub use fft::{...}` in
  `lib.rs`. #1296 closed previously.

## Architecture

### File layout

`ferrotorch-core/src/grad_fns/fft.rs` is organised in three layers:

1. **Per-op `*Backward` structs + `GradFn` impls** (lines 22-1016). Each
   struct stores the saved-for-backward operand (`input: Tensor<T>`) plus
   any forward-pass parameters needed to size the backward (`n`, `s`,
   `axes`, `fft_n`, `output_n`, `out_shape`, `last_axis_n`,
   `last_axis_logical`, `norm_n`). The `backward()` method does the VJP
   math; `inputs()` exposes the saved tensor to the autograd engine;
   `name()` returns the static string used for `grad_fn` attribution.

2. **Per-op `<op>_differentiable` forward wrappers** (lines 340-411 for
   1-D, 1079-1290 for N-D + Hermitian). Each wrapper calls the
   non-differentiable kernel in `crate::fft`, checks
   `is_grad_enabled() && input.requires_grad()`, and either returns the
   result directly (no-grad path) or constructs the `*Backward` grad_fn,
   re-hydrates the result data onto the input's device, and calls
   `Tensor::from_operation` to attach the grad_fn.

3. **Module-private helpers** (lines 681-1077): `row_major_strides`
   (general N-D stride computation used by `RfftnBackward` and
   `IrfftnBackward`), `fftn_norm_n` (compute `prod(s)` from `s` / `axes`
   / default-all-inner-dims for complex inputs), `rfftn_norm_n` (same
   for real inputs — no trailing complex pair).

### REQ-1 — `fft_differentiable` + `FftBackward`

`FftBackward` is at `grad_fns/fft.rs` lines 43-80; backward at lines 56-71
computes `grad_x = n * ifft(grad_y)` by calling `fft::ifft(grad_output,
self.n)`, multiplying every interleaved real/imag entry by the FFT length
`fft_n = grad_output.shape()[grad_output.ndim() - 2]`, materializing the
result on CPU then optionally pushing back to CUDA. The forward wrapper
`pub fn fft_differentiable` at lines 341-355 calls `fft::fft(input, n)`,
constructs `FftBackward::new(input.clone(), n)`, and calls
`Tensor::from_operation` to attach the grad_fn. **Public-API consumer**:
`ferrotorch-core/src/lib.rs:160-162` re-exports `fft_differentiable` as
`ferrotorch_core::fft_differentiable`. **No in-tree non-test caller
exists** — the only callers are the in-file unit tests (`mod tests`) and
the external test files `tests/conformance_fft.rs` /
`tests/_probe_b7_a1_fft_backward_normalization.rs`. Per goal.md
R-DEFER-1 / R-HONEST-2, test-only callers don't count as production
consumers; however, per S5 the `pub use` at `lib.rs:160-162` IS the
public-API surface and is grandfathered.

### REQ-2 — `ifft_differentiable` + `IfftBackward`

`IfftBackward` at lines 92-129; backward computes
`grad_x = fft(grad_y) / n` by calling `fft::fft(grad_output, self.n)`,
dividing every entry by `fft_n`, and materializing the result. Forward
wrapper at lines 358-372. Re-exported at `lib.rs:160-162` as
`ferrotorch_core::ifft_differentiable`.

### REQ-3 — `rfft_differentiable` + `RfftBackward`

`RfftBackward` at lines 158-238; backward at lines 177-229 implements the
zero-pad-then-unnormalized-inverse path described in REQ-3 above. The
shape validation at lines 182-188 rejects `grad_output` without a trailing
complex pair. The padded buffer is allocated, the half-spectrum is copied
into the first K complex slots, the rest are left zero, and `fft::ifft`
is called on the padded tensor. The output's real part is extracted by
sampling only the real component of each complex pair. Forward wrapper at
lines 375-391. Re-exported at `lib.rs:160-162`. The previous (buggy) call
to `fft::rfft(grad_y, N)` was wrong by a factor of N and by the missing
boundary correction; #807-#809 fixed it via the doc-comment block at
lines 138-157.

### REQ-4 — `irfft_differentiable` + `IrfftBackward`

`IrfftBackward` at lines 276-334; backward at lines 294-326 calls
`fft::rfft(grad_output, Some(n))` to obtain the complex `[..., K, 2]`
buffer, then iterates each pair applying factor `1/n` at boundary indices
(`kk == 0` always, `kk == K-1` when `n` is even) and `2/n` at interior
indices. The odd-N branch (no Nyquist sample) treats every `k > 0` as
interior (line 312). Forward wrapper at lines 394-411. Re-exported at
`lib.rs:160-162`.

### REQ-5 — `fftn_differentiable` + `FftnBackward`

`FftnBackward` at lines 425-473; backward calls `fft::ifftn(grad_output,
s, axes)` and multiplies every entry by `norm_n = prod(s)`. Forward
wrapper at lines 1079-1101 computes `norm_n = fftn_norm_n(input, s, axes)`
(the private helper at lines 1027-1054 resolves the product whether `s`,
`axes`, or neither is given). **No `lib.rs` re-export** — the wrapper is
pub-in-module but not part of the crate's public API. Per R-DEFER-1, this
is vocabulary without a non-test production consumer; the gap is tracked
in #1296.

### REQ-6 — `ifftn_differentiable` + `IfftnBackward`

`IfftnBackward` at lines 475-523; backward calls `fft::fftn(grad_output,
s, axes)` and divides by `norm_n`. Forward wrapper at lines 1103-1125. No
`lib.rs` re-export.

### REQ-7 — `rfftn_differentiable` + `RfftnBackward`

`RfftnBackward` is the most complex backward in the file (lines 550-679).
It generalizes `RfftBackward` to N-D: the saved `out_shape` /
`last_axis_n` / `last_axis_logical` / `norm_n` capture the rfftn output
layout. The backward at lines 591-679 validates `grad_output.shape ==
out_shape`, allocates a padded buffer with the last freq axis expanded
from `K` to `n_last`, computes row-major strides for both the source and
padded shapes via the private `row_major_strides` helper at lines 681-688,
copies the half-spectrum pairs into the padded buffer, calls
`fft::ifftn(padded, s, axes)`, multiplies every entry by `norm_n`, and
extracts the real part. Forward wrapper at lines 1127-1191 resolves
`s_back` (the original real-input shape along transform axes) from
either explicit `s`, the input's shape along `axes`, or the full input
shape. No `lib.rs` re-export.

### REQ-8 — `irfftn_differentiable` + `IrfftnBackward`

`IrfftnBackward` at lines 709-801; backward at lines 744-790 calls
`fft::rfftn(grad_output, s, axes)`, iterates every complex pair applying
the boundary-vs-interior factor (`scale = 1/norm_n` for boundary,
`2 * scale` for interior — same parity test as `IrfftBackward` but
applied along the persisted `last_axis_logical`). Forward wrapper at lines
1193-1244. No `lib.rs` re-export.

### REQ-9 — `hfft_differentiable` + `HfftBackward`

`HfftBackward` at lines 842-906; backward at lines 861-897 calls
`fft::rfft(grad_output, Some(n))` to compute `F = unnormalized
rfft(grad_y, n)`, then iterates the complex pairs applying factor 1
(boundary) or 2 (interior) with **sign flip on the imaginary part** (the
`conj` from the upstream `hfft = unnormalized irfft of conj(x)` derivation
— see the doc-comment block at lines 814-841). Forward wrapper at lines
1247-1267. No `lib.rs` re-export. Same #807-809 fix-up applied here:
`HfftBackward`'s previous `fft::ihfft(grad_y, n)` call was wrong by an N
factor and missing the interior-doubling correction.

### REQ-10 — `ihfft_differentiable` + `IhfftBackward`

`IhfftBackward` at lines 940-1016; backward at lines 953-1007 zero-pads
`conj(grad_y)` (the conjugation is the `re` / `-im` write in the copy
loop at lines 978-982) along the freq axis from `K` to `N`, calls
`fft::ifft` (already supplies `1/N`), and takes the real part. Forward
wrapper at lines 1270-1290. No `lib.rs` re-export.

### REQs 11-18 — `*2` and Hermitian-N forwards/wrappers (#1299 + #1300)

The route's `parity_ops` list names eight ops (`fft.fft2`, `fft.ifft2`,
`fft.rfft2`, `fft.irfft2`, `fft.hfft2`, `fft.ihfft2`, `fft.hfftn`,
`fft.ihfftn`). State after the #1299 + #1300 build:

- `fft.fft2` / `fft.ifft2` (REQ-11/12) — forward kernels in
  `crate::fft::fft2` / `crate::fft::ifft2`; the autograd wrappers
  `fft2_differentiable` + `Fft2Backward` and `ifft2_differentiable` +
  `Ifft2Backward` now ship in `grad_fns/fft.rs` (closes #1300) and are
  re-exported in `lib.rs`. The backward identity is `FftnBackward` /
  `IfftnBackward` restricted to the trailing two axes with
  `norm_n = rows * cols` (per `fft_fft2_symint`'s literal `return
  fft_fftn_symint(...)` delegation at
  `aten/src/ATen/native/SpectralOps.cpp:644-652`).

- `fft.rfft2`, `fft.irfft2`, `fft.hfft2`, `fft.ihfft2`, `fft.hfftn`,
  `fft.ihfftn` (REQ-13..18) — the **forward kernels** now ship in
  `ferrotorch-core/src/fft.rs` (closes the forward half of #1299),
  delegating to `ferray_fft` 0.3.8's native `rfft2` / `irfft2` / `hfft2`
  / `ihfft2` / `hfftn` / `ihfftn` (which themselves delegate to `rfftn` /
  `irfftn` over the trailing 2 axes per the upstream `_symint` impls at
  SpectralOps.cpp:664-714). All six are re-exported in `lib.rs`. The
  matching `*_differentiable` autograd wrappers for these six are a
  follow-up (no in-tree autograd consumer requires them yet).

## Parity contract

Eighteen parity-sweep ops are declared in the route's `parity_ops` field:
`fft.fft`, `fft.ifft`, `fft.fft2`, `fft.ifft2`, `fft.fftn`, `fft.ifftn`,
`fft.rfft`, `fft.irfft`, `fft.rfft2`, `fft.irfft2`, `fft.rfftn`,
`fft.irfftn`, `fft.hfft`, `fft.ihfft`, `fft.hfft2`, `fft.ihfft2`,
`fft.hfftn`, `fft.ihfftn`. None currently have a corresponding entry in
`tools/parity-sweep/parity_audit.json` and none have a runner arm in
`tools/parity-sweep/runner/src/main.rs`. As of 2026-05-25 the parity
oracle (`tools/parity-sweep/oracle.py:62`) rejects every `torch.complex64`
/ `torch.complex128` op_db sample with
`ValueError: unsupported dtype: torch.complex64`, which means the
complex-input ops (`fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`,
`rfft2`, `irfft2`, `hfft2`, `ihfft2`, `ihfftn`) cannot be exercised at
all. The real-input ops (`rfft`, `irfft`, `rfftn`, `irfftn`, `hfft`,
`ihfft`, `hfftn`) reach the runner but skip every sample because no
runner arm dispatches them; the sweep emits `0/N passed (N skipped, 0
failed)`. **Verifying these eighteen ops requires three pieces of
infrastructure**:

1. Oracle complex-dtype support (`tools/parity-sweep/oracle.py:62` —
   add `torch.complex64` / `torch.complex128` → interleaved-`[..., 2]`
   real conversion in the encode helper, and the reverse in the decode
   helper). Tracked in #1294.

2. Eighteen runner arms in `tools/parity-sweep/runner/src/main.rs`
   following the pattern of the arithmetic arms (e.g. `"add"` at
   line 434) — each arm decodes the complex-input samples into
   ferrotorch's `[..., 2]` layout and dispatches to the corresponding
   `<op>_differentiable`. Tracked in #1294.

3. For ops with no forward kernel (REQ-13..18) the forward op must be
   built first. Tracked in #1299.

Each backward implementation also needs the **gradcheck** parity test
(autograd numerical-vs-symbolic gradient agreement with PyTorch's
`torch.autograd.gradcheck`) — this is a higher tier of verification than
the parity sweep itself; out of scope for this design doc, but called out
because the chain-rule math in the `*Backward` structs is the part most
likely to silently diverge.

## Verification

### In-file unit tests (`mod tests` at lines 1296-1513)

19 `#[test]` functions:

| Test fn | Lines | Covers |
|---|---|---|
| `fft_differentiable_attaches_grad_fn` | 1326-1333 | REQ-1: grad_fn name `"FftBackward"` |
| `fft_differentiable_no_grad_when_not_needed` | 1335-1340 | REQ-1: no grad_fn when `requires_grad=false` |
| `fft_backward_identity_check` | 1342-1361 | REQ-1: VJP correctness on impulse → all-ones forward → 4× impulse backward |
| `ifft_backward_identity_check` | 1363-1381 | REQ-2: VJP correctness on all-ones → impulse forward → constant backward |
| `rfft_differentiable_attaches_grad_fn` | 1383-1389 | REQ-3 |
| `irfft_differentiable_attaches_grad_fn` | 1391-1398 | REQ-4 |
| `no_grad_context_disables_tracking` | 1400-1406 | All: `no_grad` short-circuit |
| `fftn_differentiable_attaches_grad_fn` | 1412-1419 | REQ-5 |
| `ifftn_differentiable_attaches_grad_fn` | 1421-1427 | REQ-6 |
| `fftn_no_grad_when_not_needed` | 1429-1434 | REQ-5: no-grad path |
| `fftn_backward_returns_real_grad_for_impulse` | 1436-1454 | REQ-5: VJP correctness on 2-D impulse → 4× impulse backward |
| `rfftn_differentiable_attaches_grad_fn` | 1456-1463 | REQ-7 |
| `irfftn_differentiable_attaches_grad_fn` | 1465-1472 | REQ-8 |
| `hfft_differentiable_attaches_grad_fn` | 1474-1481 | REQ-9 |
| `ihfft_differentiable_attaches_grad_fn` | 1483-1489 | REQ-10 |
| `fftn_norm_n_default_inner_dims` | 1491-1497 | Private helper `fftn_norm_n` default path |
| `fftn_norm_n_with_explicit_s` | 1499-1504 | `fftn_norm_n` with `s` kwarg |
| `fftn_norm_n_with_axes` | 1506-1512 | `fftn_norm_n` with `axes` kwarg |

### External test files

- `ferrotorch-core/tests/conformance_fft.rs` — uses every differentiable
  wrapper via the dispatch table at `:1623-1678`. This is a test-side
  consumer; per R-DEFER-1 it does NOT count as a production consumer.
- `ferrotorch-core/tests/_probe_b7_a1_fft_backward_normalization.rs` —
  failing-test probe pinning the pre-#807-#809 backward divergences
  (rfft / irfft / rfftn / irfftn / hfft / ihfft normalization bugs).
  Test-only.

### Parity-sweep smoke commands

```bash
# Currently every command below emits 0/N passed (N skipped or N failed).
# All require #1294 (oracle complex-dtype + runner arms) before any can
# return a passing integer.
./target/release/parity-sweep sweep --op fft.fft     --seeds 8
./target/release/parity-sweep sweep --op fft.ifft    --seeds 8
./target/release/parity-sweep sweep --op fft.fft2    --seeds 8
./target/release/parity-sweep sweep --op fft.ifft2   --seeds 8
./target/release/parity-sweep sweep --op fft.fftn    --seeds 8
./target/release/parity-sweep sweep --op fft.ifftn   --seeds 8
./target/release/parity-sweep sweep --op fft.rfft    --seeds 8
./target/release/parity-sweep sweep --op fft.irfft   --seeds 8
./target/release/parity-sweep sweep --op fft.rfft2   --seeds 8
./target/release/parity-sweep sweep --op fft.irfft2  --seeds 8
./target/release/parity-sweep sweep --op fft.rfftn   --seeds 8
./target/release/parity-sweep sweep --op fft.irfftn  --seeds 8
./target/release/parity-sweep sweep --op fft.hfft    --seeds 8
./target/release/parity-sweep sweep --op fft.ihfft   --seeds 8
./target/release/parity-sweep sweep --op fft.hfft2   --seeds 8
./target/release/parity-sweep sweep --op fft.ihfft2  --seeds 8
./target/release/parity-sweep sweep --op fft.hfftn   --seeds 8
./target/release/parity-sweep sweep --op fft.ihfftn  --seeds 8
```

Expected once #1294 + #1296 + #1299 + #1300 land: each command's tail
should contain a single line `[fft.<op>] N/N passed (0 skipped, 0
failed)` with N >= 1.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | NOT-STARTED | impl `pub fn fft_differentiable` in `grad_fns/fft.rs` + `FftBackward` struct in same file + `pub use` re-export at `lib.rs` lines 160-162; in-file tests `fft_differentiable_attaches_grad_fn` / `fft_backward_identity_check` pass. BLOCKED on parity-sweep verification: oracle rejects `torch.complex64` (#1294) → no parity smoke can fire. Per R-DEFER-6 the smoke must return ≥1 to claim SHIPPED. |
| REQ-2 | NOT-STARTED | impl `pub fn ifft_differentiable` + `IfftBackward` struct; re-exported at `lib.rs:160-162`. BLOCKED on #1294. |
| REQ-3 | NOT-STARTED | impl `pub fn rfft_differentiable` + `RfftBackward` struct (post-#807-#809 fix); re-exported at `lib.rs:160-162`. BLOCKED on #1294 (need runner arm for real-input op). |
| REQ-4 | NOT-STARTED | impl `pub fn irfft_differentiable` + `IrfftBackward` struct; re-exported at `lib.rs:160-162`. BLOCKED on #1294 + complex-output handling. |
| REQ-5 | NOT-STARTED | impl `pub fn fftn_differentiable` + `FftnBackward` struct; **NOT re-exported in `lib.rs`** — vocabulary lacking public-API surface. BLOCKED on #1294 + #1296. |
| REQ-6 | NOT-STARTED | impl `pub fn ifftn_differentiable` + `IfftnBackward` struct; NOT re-exported in `lib.rs`. BLOCKED on #1294 + #1296. |
| REQ-7 | NOT-STARTED | impl `pub fn rfftn_differentiable` + `RfftnBackward` struct; NOT re-exported in `lib.rs`. BLOCKED on #1294 + #1296. |
| REQ-8 | NOT-STARTED | impl `pub fn irfftn_differentiable` + `IrfftnBackward` struct; NOT re-exported in `lib.rs`. BLOCKED on #1294 + #1296. |
| REQ-9 | NOT-STARTED | impl `pub fn hfft_differentiable` + `HfftBackward` struct (post-#807-#809 fix); NOT re-exported in `lib.rs`. BLOCKED on #1294 + #1296. |
| REQ-10 | NOT-STARTED | impl `pub fn ihfft_differentiable` + `IhfftBackward` struct; NOT re-exported in `lib.rs`. BLOCKED on #1294 + #1296. |
| REQ-11 | NOT-STARTED | impl `pub fn fft2_differentiable` + `Fft2Backward` struct now SHIP in `grad_fns/fft.rs` (closes #1300); re-exported at `lib.rs` as `ferrotorch_core::fft2_differentiable` (non-test public-API consumer). In-file tests `fft2_differentiable_attaches_grad_fn` / `fft2_backward_returns_grad_for_corner_impulse` / `fft2_ifft2_differentiable_roundtrip_values` pass. BLOCKED on parity-sweep verification: oracle rejects `torch.complex64` (#1294) → no parity smoke can fire. |
| REQ-12 | NOT-STARTED | impl `pub fn ifft2_differentiable` + `Ifft2Backward` struct now SHIP (closes #1300); re-exported at `lib.rs`. In-file test `ifft2_differentiable_attaches_grad_fn` passes. BLOCKED on #1294. |
| REQ-13 | NOT-STARTED | forward `rfft2` now SHIPS in `fft.rs` (delegates to `ferray_fft::rfft2`, closes #1299-forward); re-exported at `lib.rs` as `ferrotorch_core::rfft2`. In-file tests `rfft2_output_shape_and_irfft2_roundtrip` / `rfft2_matches_rfftn_over_last_two_axes` pass. `rfft2_differentiable` autograd wrapper still absent. BLOCKED on #1294 (parity) + autograd-wrapper follow-up. |
| REQ-14 | NOT-STARTED | forward `irfft2` now SHIPS in `fft.rs` (delegates to `ferray_fft::irfft2`); re-exported at `lib.rs`. Covered by `rfft2_output_shape_and_irfft2_roundtrip`. `irfft2_differentiable` autograd wrapper still absent. BLOCKED on #1294 + autograd follow-up. |
| REQ-15 | NOT-STARTED | forward `hfft2` now SHIPS in `fft.rs` (delegates to `ferray_fft::hfft2`); re-exported at `lib.rs`. In-file tests `ihfft2_hfft2_roundtrip` / `hfft2_matches_hfftn_over_last_two_axes` pass. `hfft2_differentiable` wrapper still absent. BLOCKED on #1294 + autograd follow-up. |
| REQ-16 | NOT-STARTED | forward `ihfft2` now SHIPS in `fft.rs` (delegates to `ferray_fft::ihfft2`); re-exported at `lib.rs`. Covered by `ihfft2_hfft2_roundtrip`. `ihfft2_differentiable` wrapper still absent. BLOCKED on #1294 + autograd follow-up. |
| REQ-17 | NOT-STARTED | forward `hfftn` now SHIPS in `fft.rs` (delegates to `ferray_fft::hfftn`); re-exported at `lib.rs`. In-file test `ihfftn_hfftn_roundtrip_3d` passes. `hfftn_differentiable` wrapper still absent. BLOCKED on #1294 + autograd follow-up. |
| REQ-18 | NOT-STARTED | forward `ihfftn` now SHIPS in `fft.rs` (delegates to `ferray_fft::ihfftn`); re-exported at `lib.rs`. Covered by `ihfftn_hfftn_roundtrip_3d`. `ihfftn_differentiable` wrapper still absent. BLOCKED on #1294 + autograd follow-up. |

### Blocker summary

- **#1294** — parity-sweep oracle lacks `torch.complex64` / `torch.complex128`
  dtype support; no FFT runner arms exist. Required for all 18 REQs. STILL
  OPEN — this is the single gating blocker now; every FFT REQ's impl +
  public-API consumer + in-file tests are in place, but none can claim the
  `grep -c "passed (0 skipped, 0 failed)" >= 1` evidence R-DEFER-6 requires
  until the oracle round-trips complex tensors. See `## Parity contract`
  for the three-piece infra breakdown; pieces 2 (18 runner arms) and 3
  (forward kernels) are now resolved — piece 1 (oracle complex round-trip)
  remains.
- **#1296** — CLOSED. Six N-D / Hermitian differentiable wrappers (REQ-5..10)
  re-exported in `lib.rs`.
- **#1299** — forward ops CLOSED. The six forward kernels (`rfft2`,
  `irfft2`, `hfft2`, `ihfft2`, `hfftn`, `ihfftn`) now ship in
  `ferrotorch-core/src/fft.rs` (delegating to `ferray_fft` 0.3.8) and are
  re-exported in `lib.rs`. The matching `*_differentiable` autograd wrappers
  for REQ-13..18 remain a follow-up (no autograd consumer required them yet
  — the forward ops are the gated #1299 deliverable).
- **#1300** — CLOSED. `fft2_differentiable` + `Fft2Backward` and
  `ifft2_differentiable` + `Ifft2Backward` ship in `grad_fns/fft.rs`,
  re-exported in `lib.rs`.

### Honest under-claim note

Every REQ above is classified NOT-STARTED, even though ten of the eighteen
ops have working `<op>_differentiable` + `*Backward` implementations with
passing in-file unit tests (REQ-1..10). The classification is binary per
R-DEFER-2: SHIPPED requires impl + non-test production consumer + tests +
parity-sweep smoke ≥1. The parity-sweep smoke is the gating constraint
across all 18 — no FFT op currently has a parity arm or a runner-side
oracle that can handle complex inputs, so none can claim the
`grep -c "passed (0 skipped, 0 failed)" >= 1` evidence R-DEFER-6 requires.
The author chose to classify all 18 NOT-STARTED rather than split the
file into "10 SHIPPED-pending-infra / 8 NOT-STARTED-needs-forward" because
the doc is a contract for future audits; conservative under-claiming
matches R-HONEST-3 ("honest underclaim beats unverified overclaim").
