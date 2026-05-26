# ferrotorch-vision — `RandomGaussianBlur` transform

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (torchvision v0.26.0 site-packages)
upstream-paths:
  - torchvision/transforms/v2/_misc.py
  - torchvision/transforms/v2/functional/_misc.py
-->

## Summary

`ferrotorch-vision/src/transforms/random_gaussian_blur.rs` provides
`RandomGaussianBlur<T: Float>`, which applies a separable
(rows-then-columns) Gaussian blur with a random sigma to each channel
of a `[C, H, W]` tensor. Mirrors `torchvision.transforms.v2.GaussianBlur`
at `_misc.py:177-216` (note: torchvision's class name is `GaussianBlur`;
ferrotorch's `RandomGaussianBlur` name emphasizes the random-sigma
sampling — upstream's class is also random in sigma when given a
`(lo, hi)` tuple).

## Requirements

- REQ-1: `pub struct RandomGaussianBlur<T: Float>` storing
  `kernel_size: usize`, `sigma_lo: f64`, `sigma_hi: f64`, and
  `PhantomData<T>`. Mirrors `_misc.py:177` `class GaussianBlur`.

- REQ-2: `pub fn RandomGaussianBlur::new(kernel_size: usize, sigma:
  (f64, f64)) -> FerrotorchResult<Self>` constructor validating:
  `kernel_size >= 1 && kernel_size % 2 == 1` (odd-and-positive, like
  upstream), AND `sigma.0 > 0 && sigma.0 <= sigma.1`. Mirrors
  upstream's `Kernel size value should be an odd and positive number`
  ValueError and the sigma min/max check.

- REQ-3: `fn gaussian_kernel_1d(size, sigma) -> Vec<f64>` private
  helper — computes a 1-D Gaussian filter centered at the kernel
  midpoint, normalized to sum to 1. Used by the separable convolution
  path.

- REQ-4: `fn blur_rows(data, h, w, kernel) -> Vec<f64>` and
  `fn blur_cols(...) -> Vec<f64>` — 1-D zero-padded convolutions
  along rows / columns. Together they form the separable 2-D blur:
  `blur_cols(blur_rows(image))`.

- REQ-5: `impl<T: Float> Transform<T> for RandomGaussianBlur<T>` —
  `apply` rejects non-3-D input, samples
  `sigma = lo + random_f64() * (hi - lo)`, computes the 1-D kernel,
  and applies separable blur per channel. The per-channel data is
  promoted to `f64` for the convolution math (numerical headroom for
  the kernel-weighted sums), then cast back to `T` per element.

- REQ-6: SHIPPED — `fn reflect_index` reflects integer convolution
  indices against `[0, size)` matching PyTorch's `padding_mode='reflect'`
  semantics; `blur_rows` and `blur_cols` use it so border pixels are
  no longer dimmed.

## Acceptance Criteria

- [x] AC-1: `RandomGaussianBlur::new(3, (0.1, 2.0))` constructs.
- [x] AC-2: Even `kernel_size` returns `Err`.
- [x] AC-3: `sigma.0 <= 0` or `sigma.0 > sigma.1` returns `Err`.
- [x] AC-4: Output shape equals input shape (verified by
  `test_gaussian_blur_output_shape` at `random_gaussian_blur.rs:165`).
- [x] AC-5: Uniform-value image is unchanged in the interior after
  blur (border pixels see zero-padding effect) (verified by
  `test_gaussian_blur_uniform_image` at `random_gaussian_blur.rs:174`).
- [x] AC-6: An impulse pixel spreads to its neighbors but stays
  brightest at center (verified by `test_gaussian_blur_smooths_impulse`
  at `random_gaussian_blur.rs:198`).
- [x] AC-7: `gaussian_kernel_1d` sums to 1 within `1e-10` (verified
  at `random_gaussian_blur.rs:217`).
- [x] AC-8: `gaussian_kernel_1d` is symmetric (verified at
  `random_gaussian_blur.rs:227`).
- [x] AC-9: Non-3-D input returns `Err` (verified at
  `random_gaussian_blur.rs:242`).
- [x] AC-10: f32 works (verified at `random_gaussian_blur.rs:250`).
- [x] AC-11: reflection padding (verified by
  `test_reflect_index_canonical_cases` and
  `test_gaussian_blur_border_pixels_not_dimmed` in
  `random_gaussian_blur.rs`).

## Architecture

### Struct + constructor (REQ-1, REQ-2)

```rust
pub struct RandomGaussianBlur<T: Float> {
    kernel_size: usize,
    sigma_lo: f64,
    sigma_hi: f64,
    _marker: std::marker::PhantomData<T>,
}
```

at `random_gaussian_blur.rs:15-20`. Constructor at
`random_gaussian_blur.rs:31-55` enforces:
- `kernel_size >= 1 && kernel_size % 2 == 1` (single combined check).
- `sigma.0 > 0.0 && sigma.0 <= sigma.1`.

### Gaussian kernel (REQ-3)

`fn gaussian_kernel_1d(size, sigma) -> Vec<f64>` at
`random_gaussian_blur.rs:59-75`:

```rust
let half = (size / 2) as i64;
for i in 0..size {
    let x = (i as i64 - half) as f64;
    let val = (-0.5 * (x / sigma).powi(2)).exp();
    kernel.push(val);
    sum += val;
}
for v in kernel.iter_mut() {
    *v /= sum;
}
```

Standard 1-D Gaussian formula. Normalizes by sum so the blur preserves
the per-pixel intensity scale (no DC gain shift).

### Separable convolution (REQ-4)

`fn blur_rows` at `random_gaussian_blur.rs:78-96` and `fn blur_cols` at
`random_gaussian_blur.rs:99-116`. Both walk the source index
`col + (ki - half)` (or row equivalent), reading only positions
inside `[0, w)` (or `[0, h)`). Out-of-bounds indices contribute 0 to
the accumulator — zero-padding semantics.

### Transform impl (REQ-5)

`fn apply` at `random_gaussian_blur.rs:118-159`:

1. 3-D check.
2. Sample sigma uniformly in `[lo, hi]`.
3. Compute kernel.
4. Per channel:
   a. Promote slice to `Vec<f64>` (precision headroom).
   b. `blurred = blur_cols(blur_rows(ch_data))`.
   c. Cast each f64 result back to `T` via `cast::<f64, T>(v)?`.

The two-pass separable blur is `O(C·H·W·K)` instead of the `O(C·H·W·K²)`
naive 2-D convolution.

### NOT-STARTED gap (REQ-6)

Upstream uses reflection padding by default (`torch.nn.functional.pad(...,
mode='reflect')`) which avoids the dark-border artefact ferrotorch's
zero-padding introduces. The fix is local — replace the `src_col >= 0
&& (src_col as usize) < w` guard with a reflection-index helper —
but is worth its own blocker. Blocker #1519.

### Non-test production consumers

- `pub use random_gaussian_blur::RandomGaussianBlur;` at
  `ferrotorch-vision/src/transforms/mod.rs:26` AND `RandomGaussianBlur`
  in the crate-root re-export at `ferrotorch-vision/src/lib.rs:114`.

## Parity contract

`parity_ops = []`.

- **Uniform-value input** in interior pixels: unchanged (kernel sums
  to 1).
- **Border pixels**: dimmed by the missing kernel mass at the image
  edge (zero-padding effect; reflection-padding would avoid this).
- **Impulse input**: spreads symmetrically around center, attenuated
  by the kernel central weight.
- **`sigma_lo == sigma_hi`**: deterministic sigma — useful for tests.
- **Non-3-D input**: `InvalidArgument`.

## Verification

Tests in `mod tests in random_gaussian_blur.rs` (7 tests):

- `test_gaussian_blur_output_shape` at `random_gaussian_blur.rs:165`
- `test_gaussian_blur_uniform_image` at `random_gaussian_blur.rs:174`
- `test_gaussian_blur_smooths_impulse` at `random_gaussian_blur.rs:198`
- `test_gaussian_kernel_1d_sums_to_one` at `random_gaussian_blur.rs:217`
- `test_gaussian_kernel_1d_symmetry` at `random_gaussian_blur.rs:227`
- `test_gaussian_blur_rejects_non_3d` at `random_gaussian_blur.rs:242`
- `test_gaussian_blur_f32` at `random_gaussian_blur.rs:250`

Smoke:

```bash
cargo test -p ferrotorch-vision --lib transforms::random_gaussian_blur:: 2>&1 | tail -3
```

Expected: `7 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct RandomGaussianBlur<T: Float>` with `kernel_size, sigma_lo, sigma_hi, _marker` at `ferrotorch-vision/src/transforms/random_gaussian_blur.rs:15-20`, mirroring `torchvision/transforms/v2/_misc.py:177` `class GaussianBlur`; non-test consumer: `pub use random_gaussian_blur::RandomGaussianBlur;` at `mod.rs:26` AND `RandomGaussianBlur` in the crate-root re-export at `ferrotorch-vision/src/lib.rs:114`. |
| REQ-2 | SHIPPED | impl: `pub fn RandomGaussianBlur::new(kernel_size: usize, sigma: (f64, f64)) -> FerrotorchResult<Self>` with odd-positive kernel and sigma-ordering checks at `random_gaussian_blur.rs:31-55`; non-test consumer: registered as public surface (the conformance inventory listing for `RandomGaussianBlur::new` is at `ferrotorch-vision/tests/conformance/_surface_inventory.toml` — `RandomGaussianBlur` block); reachable via the crate-root re-export at `lib.rs:114`. |
| REQ-3 | SHIPPED | impl: `fn gaussian_kernel_1d(size: usize, sigma: f64) -> Vec<f64>` at `random_gaussian_blur.rs:59-75`; non-test consumer: `fn apply` in this same file calls `gaussian_kernel_1d(self.kernel_size, sigma)` at `random_gaussian_blur.rs:136`. |
| REQ-4 | SHIPPED | impl: `fn blur_rows(data, h, w, kernel) -> Vec<f64>` at `random_gaussian_blur.rs:78-96` and `fn blur_cols(...)` at `random_gaussian_blur.rs:99-116`; non-test consumer: `fn apply` chains `blur_cols(blur_rows(...))` at `random_gaussian_blur.rs:148-149`. |
| REQ-5 | SHIPPED | impl: `impl<T: Float> Transform<T> for RandomGaussianBlur<T>` at `random_gaussian_blur.rs:118-159`; non-test consumer: any `Box<dyn Transform<T>>` slot accepts this — composes into augmentation `Compose` pipelines. The crate-root re-export at `lib.rs:114` is the production-facing handle. |
| REQ-6 | SHIPPED | impl: `fn reflect_index` + reflection-padded `blur_rows` / `blur_cols` in `ferrotorch-vision/src/transforms/random_gaussian_blur.rs:88-127`; non-test consumer: `pub use random_gaussian_blur::RandomGaussianBlur;` at `mod.rs:34` — the `impl<T: Float> Transform<T>` body in the same file calls `blur_cols(blur_rows(...))` per channel. |
