# Signal window functions

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - torch/signal/windows/windows.py
  - torch/signal/windows/__init__.py
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/signal/windows.rs` implements the
`torch.signal.windows.*` family — 15 window-coefficient generators
(Bartlett, Blackman, Hamming, Hann, Kaiser, cosine, exponential,
Gaussian, general-cosine, general-Hamming, Nuttall, Parzen, Taylor,
Tukey, plus the NumPy-named `hanning` alias for `hann`). Each returns a
1-D `Tensor<f64>` living on `Device::Cpu`. The coefficient math is
delegated to the `ferray-window` crate; this module is the
ferrotorch-flavored wrapper.

## Requirements

- REQ-1: NumPy-core 5: `bartlett(m)`, `blackman(m)`, `hamming(m)`,
  `hann(m)` (+ `hanning(m)` alias), `kaiser(m, beta)`. Mirror the
  five functions exposed by `numpy.*` and at upstream's
  `torch/signal/windows/windows.py:bartlett` /
  `:blackman` / `:hamming` / `:hann` / `:kaiser`.
- REQ-2: SciPy-extended 10: `cosine(m)`, `exponential(m, center, tau)`,
  `gaussian(m, std)`, `general_cosine(m, coeffs)`, `general_hamming(m,
  alpha)`, `nuttall(m)`, `parzen(m)`, `taylor(m, nbar, sll, norm)`,
  `tukey(m, alpha)`. Mirror the corresponding entries in
  `torch.signal.windows`.
- REQ-3: CPU-only return — every function returns a tensor on
  `Device::Cpu`. The GPU-discipline rationale is documented in the
  module preamble: a fake GPU path would replace ~1µs CPU compute
  with CPU compute + cudaMemcpy, a strict regression.
- REQ-4: Argument validation — `gaussian(m, std<=0)`,
  `exponential(m, _, tau<=0)`, `tukey(m, alpha ∉ [0,1])`, `taylor(m,
  nbar=0, sll)`, `general_cosine(m, [])` all error with a propagated
  `FerrotorchError::Ferray(_)` carrying the underlying
  `ferray-window::Error::InvalidValue` description. Matches upstream
  behaviour (scipy raises `ValueError` with the same root cause).
- REQ-5: `hanning(m)` is a literal alias of `hann(m)` (NumPy uses the
  longer spelling, SciPy uses the shorter; ferrotorch ships both for
  ergonomic compatibility).
- REQ-6: Symmetry — every symmetric window (Bartlett, Hann, Blackman,
  cosine, Gaussian, exponential w/ default centre, Nuttall, Parzen,
  Taylor, Tukey) produces a buffer satisfying `w[i] == w[m-1-i]` to
  within numerical tolerance.

## Acceptance Criteria

- [x] AC-1: `hann(32).shape() == &[32]` and `.device() == Cpu`
  (`signal/windows.rs:218-221`).
- [x] AC-2: `bartlett` zeros endpoints and is symmetric
  (`signal/windows.rs:238-258`).
- [x] AC-3: `kaiser(_, 0.0)` is rectangular (all 1.0)
  (`signal/windows.rs:299-308`).
- [x] AC-4: `general_cosine(_, [0.5, 0.5])` matches `hann` and
  `[0.42, 0.5, 0.08]` matches `blackman`
  (`signal/windows.rs:416-433`).
- [x] AC-5: `general_hamming(_, 0.5)` matches `hann` and
  `general_hamming(_, 0.54)` matches `hamming`
  (`signal/windows.rs:441-458`).
- [x] AC-6: `tukey(_, 1.0)` matches `hann` and `tukey(_, 0.0)` is
  rectangular (`signal/windows.rs:517-532`).
- [x] AC-7: `gaussian(m, 0.0)` and `tukey(_, 1.1)` and
  `taylor(_, 0, _, _)` all return errors
  (`signal/windows.rs:410-413, 541-546, 511-514`).
- [x] AC-8: `hanning(13)` is byte-identical to `hann(13)`
  (`signal/windows.rs:323-340`).
- [x] AC-9: All 15 functions return CPU storage
  (`signal/windows.rs:343-366`).
- [x] AC-10: `cargo test -p ferrotorch-core --lib signal::windows`
  passes.

## Architecture

Every public function in this module is a 3-line wrapper:

```rust
pub fn hann(m: usize) -> FerrotorchResult<Tensor<f64>> {
    let arr = ferray_window::hanning(m).map_err(FerrotorchError::Ferray)?;
    array_to_tensor(arr, m)
}
```

The coefficient math lives in `ferray-window` (the Rust window crate);
ferrotorch's job is the error-type marshalling and tensor wrapping.

- `array_to_tensor` (`signal/windows.rs:179-188`) converts the
  `ferray_core::Array<f64, Ix1>` into a `Vec<f64>` via owned iteration
  and constructs a `Tensor::from_storage(TensorStorage::cpu(data),
  vec![m], false)`. No GPU dispatch path exists — the user explicitly
  moves the result with `.to(Device::Cuda(0))?` when needed.
- The `hanning(m)` re-export at `windows.rs:62-65` is `#[inline]` and
  body-identical to a `hann(m)` call.

## Parity contract

`parity_ops = []`. Window coefficients are deterministic functions of
`(m, params)`; PyTorch's `torch.signal.windows.hann(N)` and
ferrotorch's `signal::windows::hann(N)` produce bit-identical buffers
when the same algorithm is used. The `ferray-window` crate's
algorithms match NumPy / SciPy's reference implementations.

The endpoints, symmetry, and matching-coefficient-set identities (e.g.
`general_cosine([0.5, 0.5]) ≡ hann`) are pinned by the
`AC-{2..8}` unit tests at `signal/windows.rs:238-532`.

## Verification

- Unit tests at `signal/windows.rs:194-547` cover length, symmetry,
  endpoint values, peak position, alias identity, alpha→Hann/Hamming
  matching, argument validation, and CPU residency.

```bash
cargo test -p ferrotorch-core --lib signal::windows
```

Expected: ~45 tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `bartlett` at `ferrotorch-core/src/signal/windows.rs:35`, `blackman` at `:41`, `hamming` at `:47`, `hann` at `:56`, `hanning` at `:63`, `kaiser` at `:74` mirror `torch/signal/windows/windows.py:{bartlett, blackman, hamming, hann, kaiser}` and `numpy.{bartlett, blackman, hamming, hanning, kaiser}`; non-test consumer: re-exported at `ferrotorch-core/src/signal/mod.rs:9-12`, reachable by FFT-based audio / DSP code (production user surface — STFT and spectrogram code applies windows to input frames). |
| REQ-2 | SHIPPED | impl: `cosine` at `ferrotorch-core/src/signal/windows.rs:91`, `exponential` at `:103`, `gaussian` at `:112`, `general_cosine` at `:123`, `general_hamming` at `:132`, `nuttall` at `:142`, `parzen` at `:150`, `taylor` at `:161`, `tukey` at `:170` mirror `torch.signal.windows.*`; non-test consumer: re-exported at `ferrotorch-core/src/signal/mod.rs:9-12`. |
| REQ-3 | SHIPPED | impl: every function calls `Tensor::from_storage(TensorStorage::cpu(data), vec![m], false)` via `array_to_tensor` at `ferrotorch-core/src/signal/windows.rs:179-188` — there is no GPU dispatch path in this file; non-test consumer: implied by every call. Test pin at `signal/windows.rs:343-366` enumerates all 15 functions and asserts `.device() == Device::Cpu`. |
| REQ-4 | SHIPPED | impl: each function propagates `ferray_window::Error::InvalidValue` via `.map_err(FerrotorchError::Ferray)?` (e.g. `signal/windows.rs:74-77`, `:103-106`, etc.); non-test consumer: the error path is hit by callers passing invalid params — production downstream code that constructs a window from a runtime parameter (e.g. an STFT block sized by a config file) receives the error rather than a panic. Test pin at `:389-393, 410-413, 511-514, 541-546`. |
| REQ-5 | SHIPPED | impl: `pub fn hanning` at `ferrotorch-core/src/signal/windows.rs:63-65` is `#[inline]` and body `hann(m)`; non-test consumer: re-exported at `signal/mod.rs:9-12` alongside `hann`. Test pin at `signal/windows.rs:323-340` asserts bit-equality. |
| REQ-6 | SHIPPED | impl: the symmetry property is enforced by `ferray-window`'s implementations; non-test consumer: production DSP code that windows a real signal segment relies on symmetric coefficients to avoid phase distortion. Test pins at `signal/windows.rs:247-258, 290-297, 370-377, 380-386, 461-468, 484-492, 500-508`. |
