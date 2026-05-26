# ferrotorch-vision — `GaussianNoise` transform

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (torchvision v0.26.0 site-packages)
upstream-paths:
  - torchvision/transforms/v2/_misc.py
-->

## Summary

`ferrotorch-vision/src/transforms/gaussian_noise.rs` provides
`GaussianNoise<T: Float>`, which adds i.i.d. Gaussian noise
`N(mean, std^2)` to every element of a `[C, H, W]` tensor for
augmentation. Mirrors `torchvision.transforms.v2.GaussianNoise` at
`_misc.py:217-251`.

## Requirements

- REQ-1: `pub struct GaussianNoise<T: Float>` storing `mean: f64`,
  `std: f64`, and `PhantomData<T>`. Mirrors `_misc.py:217` `class
  GaussianNoise(Transform)`.

- REQ-2: `pub fn GaussianNoise::new(mean: f64, std: f64) ->
  FerrotorchResult<Self>` constructor — validates `std >= 0`
  (zero is allowed and is a no-op-up-to-mean-shift). Mirrors
  upstream's implicit non-negative-sigma assumption.

- REQ-3: `impl<T: Float> Transform<T> for GaussianNoise<T>` — `apply`
  rejects non-3-D input, and:
  - If `std == 0`: returns a tensor with every value shifted by
    `mean` (the degenerate-distribution shortcut).
  - Otherwise: draws one `N(0, 1)` sample per element via Box-Muller
    (`fn standard_normal_sample`) and computes
    `v + (mean + std * z)`.
  Mirrors `torchvision/transforms/v2/functional/_misc.py:gaussian_noise`.

- REQ-4: `fn standard_normal_sample() -> f64` — Box-Muller over two
  uniform draws from `random_f64()`. The `u1.max(1e-12)` floor avoids
  `ln(0) = -∞`. Returns `r * cos(theta)` only (drops the `r * sin(theta)`
  half-sample) — wastes ~50% of uniform draws but simplifies the per-call
  shape. Acceptable for augmentation where statistical quality matters
  more than throughput; upstream uses `torch.randn` (Wichmann-Hill /
  MT19937 backend) for the same role.

- REQ-5: NOT-STARTED — upstream's `clip: bool = True` parameter (clamp
  output back into `[0, 1]` for floats / `[0, 255]` for uint8) is not
  implemented. ferrotorch lets the noisy values exit the original
  range. Blocker #1516.

## Acceptance Criteria

- [x] AC-1: `GaussianNoise::new(0.0, 0.1)` constructs successfully.
- [x] AC-2: `GaussianNoise::new(0.0, -1.0)` returns `Err(InvalidArgument)`
  (verified by `test_gaussian_noise_negative_std_errors` at
  `gaussian_noise.rs:157`).
- [x] AC-3: Output shape equals input shape (verified by
  `test_gaussian_noise_output_shape_preserved` at `gaussian_noise.rs:91`).
- [x] AC-4: `std == 0` produces output `= input + mean` exactly
  (verified by `test_gaussian_noise_zero_std_is_constant_shift` at
  `gaussian_noise.rs:100`).
- [x] AC-5: `std == 0, mean == 0` is identity (verified at
  `gaussian_noise.rs:115`).
- [x] AC-6: Over 10 000 samples, empirical mean ≈ 0 and std ≈ 0.5
  for `N(0, 0.25)` (verified by
  `test_gaussian_noise_has_approximate_mean_and_std` at
  `gaussian_noise.rs:128`).
- [x] AC-7: Non-3-D input returns `Err` (verified at `gaussian_noise.rs:149`).
- [ ] AC-8: NOT-STARTED — `clip` parameter. Blocker #1516.

## Architecture

### Struct + constructor (REQ-1, REQ-2)

```rust
pub struct GaussianNoise<T: Float> {
    mean: f64,
    std: f64,
    _marker: std::marker::PhantomData<T>,
}
```

at `gaussian_noise.rs:17-21`. Constructor at `gaussian_noise.rs:31-43`
returns `Err` if `std < 0`.

### Box-Muller sampler (REQ-4)

```rust
fn standard_normal_sample() -> f64 {
    let u1 = random_f64().max(1e-12);
    let u2 = random_f64();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    r * theta.cos()
}
```

at `gaussian_noise.rs:46-53`. The `1e-12` clamp protects against the
zero-probability `random_f64() == 0` case which would otherwise return
NaN through `ln(0)`.

### Transform impl (REQ-3)

`fn apply` at `gaussian_noise.rs:55-83`:

- 3-D shape check.
- Degenerate path (`std == 0`): `for &v in data { out.push(v + mean); }`
  — equivalent to "add a constant".
- General path: per element draw one Box-Muller sample, compute
  `noise = mean + std * sample`, push `v + cast::<f64, T>(noise)?`.

The per-element draw means each call uses `2 * N` uniform random
samples for an `N`-element tensor, of which only `N` produce noise.
This is acceptable — the augmentation latency budget is dominated by
the surrounding `image-load + decode + resize` rather than the
sampler.

### NOT-STARTED gap (REQ-5)

Upstream's `clip: bool = True` clamps output back into the original
range to prevent unsigned-integer overflow. Float users wanting clamp
can prepend a clamp transform; without `clip`, ferrotorch outputs may
exit `[0, 1]`. Blocker #1516.

### Non-test production consumers

- `pub use gaussian_noise::GaussianNoise;` at
  `ferrotorch-vision/src/transforms/mod.rs:23`.
- (Note: `GaussianNoise` is NOT re-exported at the crate root in
  `lib.rs:113-115`. Callers reach it via
  `ferrotorch_vision::transforms::GaussianNoise`.)

## Parity contract

`parity_ops = []`. ferrotorch's sampler is Box-Muller over splitmix64;
torchvision uses `torch.randn` over MT19937. These are NOT bit-equal
distributions. The contract is statistical: `mean = m, std = s`
unbiased estimators for the produced noise. Edge cases:

- **`std == 0`**: deterministic shift by `mean`.
- **`std == 0, mean == 0`**: identity (`v + 0 = v`).
- **NaN/Inf in input**: passes through plus a finite noise sample
  (NaN + x = NaN, Inf + x = Inf).
- **Non-3-D**: rejected.

## Verification

Tests in `mod tests in gaussian_noise.rs` (6 tests):

- `test_gaussian_noise_output_shape_preserved` at `gaussian_noise.rs:91`
- `test_gaussian_noise_zero_std_is_constant_shift` at `gaussian_noise.rs:100`
- `test_gaussian_noise_std_zero_mean_zero_is_identity` at `gaussian_noise.rs:115`
- `test_gaussian_noise_has_approximate_mean_and_std` at `gaussian_noise.rs:128`
- `test_gaussian_noise_rejects_non_3d` at `gaussian_noise.rs:149`
- `test_gaussian_noise_negative_std_errors` at `gaussian_noise.rs:157`

Smoke:

```bash
cargo test -p ferrotorch-vision --lib transforms::gaussian_noise:: 2>&1 | tail -3
```

Expected: `6 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct GaussianNoise<T: Float>` with `mean, std, _marker` at `ferrotorch-vision/src/transforms/gaussian_noise.rs:17-21`, mirroring `torchvision/transforms/v2/_misc.py:217` `class GaussianNoise(Transform)`; non-test consumer: `pub use gaussian_noise::GaussianNoise;` at `ferrotorch-vision/src/transforms/mod.rs:23` exposes it through the public transforms namespace. |
| REQ-2 | SHIPPED | impl: `pub fn GaussianNoise::new(mean: f64, std: f64) -> FerrotorchResult<Self>` with `std >= 0` validation at `gaussian_noise.rs:31-43`; non-test consumer: reachable via `mod.rs:23` re-export. |
| REQ-3 | SHIPPED | impl: `impl<T: Float> Transform<T> for GaussianNoise<T>` with shape check + degenerate-std shortcut + per-element noise loop at `gaussian_noise.rs:55-83`; non-test consumer: any `Box<dyn Transform<T>>` slot — typically inserted into a robustness-training `Compose` pipeline. |
| REQ-4 | SHIPPED | impl: `fn standard_normal_sample() -> f64` Box-Muller helper at `gaussian_noise.rs:46-53`; non-test consumer: `fn apply` in this same file calls `self.std * standard_normal_sample()` at `gaussian_noise.rs:77`. |
| REQ-5 | NOT-STARTED | open prereq blocker #1516 — the `clip: bool = True` parameter from `torchvision/transforms/v2/_misc.py:241-243` is not implemented; noisy outputs may exit the `[0, 1]` range. |
