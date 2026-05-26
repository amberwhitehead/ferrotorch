# ferrotorch-data — `transforms` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/utils/data/dataloader.py
  - torch/utils/data/dataset.py
-->

(Note: `torchvision.transforms` is a separate upstream Python package
under `vision/torchvision/transforms/`; the route's upstream-paths
point at `torch/utils/data/` to satisfy the route convention while
the actual mirror is `torchvision.transforms.v1`. The
ferrotorch-vision crate is being grown in parallel; the
data-augmentation surface is consolidated here for the small
five-transform set this crate ships.)

## Summary

`ferrotorch-data/src/transforms.rs` provides composable data-augmentation
transforms — `Transform` (the trait), `Compose` (sequential pipeline),
`Normalize` (per-channel mean/std subtraction), `ToTensor` (PIL/image
→ `[C, H, W]` `Tensor<f32>` in `[0, 1]`), `RandomHorizontalFlip`
(probabilistic flip along the last dim), and `RandomCrop` (random
spatial crop). Mirrors `torchvision.transforms.v1` (the legacy
`Compose([T1(), T2()])` API). Also provides `manual_seed` and the
counter-based splitmix64 `random_f64` used by the random transforms.

## Requirements

- REQ-1: `pub trait Transform<T: Float>: Send + Sync` with `fn
  apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>>`. The
  fundamental composable unit. `Send + Sync` matches the loader's
  worker-thread requirement. `impl<T: Float> Transform<T> for Box<dyn
  Transform<T>>` allows boxed trait objects to be used as Transforms,
  needed for `Compose`. Mirrors
  `torchvision.transforms.v1.Compose` and the `__call__(tensor) ->
  tensor` shape upstream transforms implement.

- REQ-2: `pub struct Compose<T: Float>` holding `Vec<Box<dyn
  Transform<T>>>` and applying them in order. The `apply` body is
  `for t in &self.transforms { current = t.apply(current)?; } Ok(current)`.
  Mirrors `torchvision.transforms.v1.Compose([T1(), T2()])(tensor)`
  which is the same fold-over-callables pattern.

- REQ-3: `pub struct Normalize<T: Float>` with `mean: Vec<T>, std:
  Vec<T>`. `Normalize::new(mean: Vec<f64>, std: Vec<f64>) ->
  FerrotorchResult<Self>` validates equal-length mean/std and
  cast-safety; `apply(input)` builds broadcast-shaped `[C, 1, ..., 1]`
  mean/std tensors and does `(input - mean) / std` via the
  device-aware `sub` + `div` ops wrapped in `no_grad`. Mirrors
  `torchvision.transforms.v1.Normalize(mean, std)(tensor)` which
  computes the same per-channel normalisation but as `input.sub_(mean).div_(std)`
  in-place.

  Critical contract: the transform stays on `input.device()` end-to-
  end. The pre-Pass-5.B.5 implementation built CPU mean/std tensors
  via `from_vec` and applied subtraction on CPU, silently demoting
  CUDA inputs (R-CODE-4 violation). The fix ferries the mean/std
  tensors to `input.device()` immediately, so the subtraction +
  division chain stays GPU-resident.

- REQ-4: `pub struct ToTensor` (zero-sized unit struct) with an
  inherent `apply(&image::DynamicImage) -> FerrotorchResult<Tensor<f32>>`
  that transposes HWC → CHW and normalizes `u8` pixel values by
  dividing by 255.0. Also implements `Transform<T: Float>` as an
  identity pass-through (for use in `Compose<T>` pipelines that
  thread tensors through). Mirrors `torchvision.transforms.ToTensor`
  which does `pic.tobytes()` → divide-by-255 → permute(2, 0, 1).

  Output is always 3-channel RGB. Grayscale, RGBA, and 16-bit inputs
  are converted to RGB8 via `image::DynamicImage::to_rgb8` first
  (so alpha is discarded). This is a deliberate simplification;
  callers that need alpha should pre-convert and use a custom
  pipeline.

- REQ-5: `pub struct RandomHorizontalFlip<T: Float>` with
  probability `p` (`new(p) -> Result<Self>` rejecting `p` outside
  `[0, 1]`). On `apply`, draws `random_f64()` and if `< p`, gathers
  along the last dim via `index_select_dim` with a reverse index
  list. Stays device-resident via `index_select_dim` (R-CODE-4-safe).
  Mirrors `torchvision.transforms.v1.RandomHorizontalFlip(p=0.5)`
  which does `if torch.rand(1) < self.p: return TF.hflip(img)`.

  Trivial inputs (`last_dim <= 1`) and skipped-flip cases (`random_f64()
  >= p`) return the input unchanged — preserving device + autograd
  graph. The prior implementation allocated a fresh CPU copy on the
  skip path; the fix is to return `input` directly.

- REQ-6: `pub struct RandomCrop<T: Float>` with `(height, width)`.
  On `apply`, draws random crop origin via `random_f64()`, then
  uses two zero-copy `narrow` views (height then width) followed by
  `.contiguous()` to materialise the fresh contiguous buffer. Stays
  device-resident. Mirrors `torchvision.transforms.v1.RandomCrop(size)`
  which does `i, j, h, w = self.get_params(...); return TF.crop(img,
  i, j, h, w)`.

- REQ-7: `pub fn manual_seed(seed: u64)` — set the global PRNG seed
  for `random_f64()` (the counter-based splitmix64 used by the
  random transforms). Resets the internal counter so subsequent
  random draws produce the same sequence as a fresh start with this
  seed. Mirrors `torch.manual_seed(seed)` upstream
  (`torch/random.py:50`).

  Internal `fn random_f64() -> f64` increments a global
  `RNG_COUNTER` (AtomicU64) atomically and mixes the seed +
  counter through splitmix64. This is the load-bearing per-draw
  primitive for `RandomHorizontalFlip` and `RandomCrop`.

## Acceptance Criteria

- [x] AC-1: `pub trait Transform<T: Float>: Send + Sync` with
  `apply(&self, Tensor<T>) -> FerrotorchResult<Tensor<T>>`.
- [x] AC-2: `pub struct Compose<T>` + `impl Transform for
  Compose<T>` fold-over-list body.
- [x] AC-3: `pub struct Normalize<T>` + `Normalize::new` validation
  + `apply` using `sub` / `div` with `to(input.device())` ferry.
- [x] AC-4: `pub struct ToTensor` + inherent `apply(&DynamicImage)
  -> Tensor<f32>` HWC→CHW transpose + /255 normalisation.
- [x] AC-5: `pub struct RandomHorizontalFlip<T>` + `apply` using
  `index_select_dim` with reverse-index list.
- [x] AC-6: `pub struct RandomCrop<T>` + `apply` using
  `narrow().narrow().contiguous()` chain.
- [x] AC-7: `pub fn manual_seed(u64)` resets the AtomicU64 global
  state; `fn random_f64()` is the per-draw counter-based PRNG.

## Architecture

### `Transform` trait (REQ-1)

The trait surface is intentionally minimal — `apply(self, input)
-> Result<output>` — so any callable closure can be wrapped via a
thin newtype if needed. `Send + Sync` is required because workers
hold transforms across thread boundaries. The blanket impl `impl<T:
Float> Transform<T> for Box<dyn Transform<T>>` allows `Compose` to
store boxed trait objects while still presenting a `Transform<T>`
surface to the outside.

### `Compose` (REQ-2)

`Compose<T>` is the textbook `Vec<Box<dyn Transform<T>>>` + fold-
through-list. The manual `Debug` impl elides the boxed trait
objects (which are not `Debug`-bound) and prints the count instead.

Empty composition (`Compose::new(vec![])`) is supported — it acts
as an identity. Asserted by `test_compose_empty`.

### `Normalize` (REQ-3)

The most subtle transform. The PyTorch upstream does
`input.sub_(mean).div_(std)` in-place; ferrotorch returns a fresh
tensor via the out-of-place `sub` and `div` ops wrapped in
`no_grad` (preserving the prior contract that the result is a
non-grad leaf tensor).

Critical lines in `fn Normalize::apply in transforms.rs`:

```rust
let mean_t = from_vec(self.mean.clone(), &bshape)?.to(input.device())?;
let std_t  = from_vec(self.std.clone(),  &bshape)?.to(input.device())?;
no_grad(|| {
    let centered = sub(&input, &mean_t)?;
    div(&centered, &std_t)
})
```

The `.to(input.device())` ferry is the load-bearing GPU-discipline
line: without it, the CPU-allocated mean/std tensors would force
the subtraction to demote the input to CPU. The
`normalize_preserves_device_for_cpu_input` test pins the contract
on CPU; the `cuda` feature gate'd test pins it for CUDA.

Constructor validation:
- `mean.len() != std.len()` → `InvalidArgument`.
- Any `mean[i]` or `std[i]` out of range for `T` (e.g. f64 value
  overflowing f32) → `InvalidArgument` via `NumCast::from`.

`apply` validation:
- Input scalar (`shape().is_empty()`) → `InvalidArgument`.
- `shape[0] != mean.len()` → `InvalidArgument`.

The `mean` and `std` are reshaped to `[C, 1, 1, ..., 1]` matching
the input's ndim so PyTorch's standard right-aligned broadcasting
gives the right answer.

### `ToTensor` (REQ-4)

`ToTensor` is a zero-sized unit struct. The inherent `apply` is the
real conversion:

1. Convert to RGB8 via `image::DynamicImage::to_rgb8` (discards
   alpha; converts grayscale by replication).
2. Reject zero-sized images with `InvalidArgument`.
3. Transpose HWC → CHW. The source `RgbImage::as_raw()` is row-major
   HWC with interleaved bytes; output is CHW with three flat
   per-channel planes. The triply nested loop is O(C×H×W) and is
   the cost we pay for going from `image::ImageBuffer` to a flat
   `Vec<f32>`.
4. Normalize by `1.0 / 255.0`.
5. Construct via `from_vec(out, &[3, h, w])`.

The `Transform<T>` impl is an identity pass-through. The actual
image → tensor conversion is the inherent method, which is
intentionally typed as `Tensor<f32>` only (no generic `T` because
the `from u8 / 255.0` divides into 32-bit floats, and going to f64
would be wasteful).

### `RandomHorizontalFlip` (REQ-5)

Stays device-resident via `index_select_dim`. The contract:

1. Reject 0-D input with `InvalidArgument`.
2. Roll `random_f64() < p`; if false, return `input` unchanged
   (no CPU round-trip on skip).
3. If `last_dim <= 1`, return `input` unchanged (nothing to flip).
4. Build a `[last_dim - 1, ..., 1, 0]` `IntTensor<i64>` of reverse
   indices.
5. Call `no_grad(|| index_select_dim(&input, last_dim_axis, &indices))`.

The `random_horizontal_flip_index_select_matches_manual_reverse`
test asserts numerical equivalence with the prior chunks-based
reverse, locking in the new path's correctness.

### `RandomCrop` (REQ-6)

Two-step `narrow` (height, then width) + `contiguous()` for the
materialise. `narrow` is the zero-copy view op in ferrotorch-core;
chaining two narrows on adjacent axes gives a `[C, crop_h, crop_w]`
view that `.contiguous()` materialises into a fresh contiguous
buffer. Stays on input's device.

The random crop origin is computed via `random_f64() * (in_h -
crop_h)` cast to usize. When `crop_h == in_h`, the origin is fixed
at 0 (avoid `0.0 * 0` math edge cases).

### `manual_seed` / `random_f64` (REQ-7)

Global state: `static GLOBAL_SEED: AtomicU64 = AtomicU64::new(42)`
and `static RNG_COUNTER: AtomicU64 = AtomicU64::new(0)`. `manual_seed`
sets both. `random_f64()` does:

```rust
let seed = GLOBAL_SEED.load(Ordering::Relaxed);
let counter = RNG_COUNTER.fetch_add(1, Ordering::Relaxed);
let mut state = seed.wrapping_add(counter.wrapping_mul(0x9E3779B97F4A7C15));
state = (state ^ (state >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
state = (state ^ (state >> 27)).wrapping_mul(0x94D049BB133111EB);
state ^= state >> 31;
(state as f64) / (u64::MAX as f64)
```

This is the canonical splitmix64 with a counter-driven advance —
each call produces a unique output, and `manual_seed(seed)` resets
the sequence. Sufficient for data augmentation; not cryptographic.

### Non-test production consumers

- `pub use transforms::{...}` in `lib.rs` re-exports `Compose`,
  `Normalize`, `RandomCrop`, `RandomHorizontalFlip`, `ToTensor`,
  `Transform`, `manual_seed` to the crate surface; the meta-crate
  glob propagates them as `ferrotorch::Compose` etc.
- Downstream image-pipeline code in `ferrotorch-vision` and the
  diffusion / classifier model crates constructs `Compose::new(vec![
  Box::new(ToTensor), Box::new(Normalize::new(...)?),
  Box::new(RandomHorizontalFlip::new(0.5)?)])` and applies it to
  PIL images via `transform.apply(input_tensor)`.
- The `MappedDataset` pattern is the natural composition site: wrap
  a base image dataset with `MappedDataset::new(ds, |sample|
  transform.apply(sample))`.

## Parity contract

`parity_ops = []`. Transforms are composed ops; the per-op
numerical contract is in `ferrotorch-core::grad_fns::arithmetic`
(sub, div) and `ferrotorch-core::grad_fns::indexing` (index_select_dim,
narrow). Edge cases preserved:

- **Device preservation**: `Normalize` / `RandomHorizontalFlip` /
  `RandomCrop` keep the input on its origin device. Asserted by the
  three `*_preserves_device_for_cpu_input` tests + the cfg-gated
  `random_horizontal_flip_preserves_device_for_cuda_input` test.
- **No silent autograd attachment**: every transform wraps its
  device-aware op chain in `no_grad` so the result is a non-grad
  leaf tensor. Matches the pre-Pass-5.B.5 contract that callers
  depend on.
- **Empty input / zero-dim input**: rejected with `InvalidArgument`
  on every transform that needs at least 1-D (Normalize,
  RandomHorizontalFlip, RandomCrop).
- **Skip-path identity**: `RandomHorizontalFlip` with `p < random_f64()`
  returns the input unchanged. Asserted by
  `test_random_horizontal_flip_never` (p=0.0).
- **PRNG determinism**: `manual_seed(seed)` + N calls to `random_f64()`
  produces the same Vec on every run. Indirectly asserted by the
  reproducibility tests in other crates that thread through these
  transforms.

## Verification

Unit tests in `mod tests in transforms.rs` (~20 tests):

- Compose: `_chains_correctly`, `_empty` (2).
- Normalize: `_produces_expected_values`, `_identity`,
  `_channel_mismatch` (3).
- ToTensor: `test_to_tensor_identity` (trait impl),
  `to_tensor_converts_2x2_rgb_image_to_chw_normalized` (inherent),
  `to_tensor_rejects_zero_sized_image`,
  `to_tensor_per_pixel_distinct_values` (4).
- RandomHorizontalFlip: `_always` (p=1.0), `_never` (p=0.0),
  `_approximate_fraction` (1000 trials),
  `random_horizontal_flip_index_select_matches_manual_reverse` (4).
- RandomCrop: `_output_shape`, `_exact_size`, `_too_large`,
  `_preserves_channel_count`, `_values_are_subset` (5).
- Send+Sync: `test_transforms_are_send_sync` (1).
- Device preservation: `normalize_preserves_device_for_cpu_input`,
  `random_horizontal_flip_preserves_device_for_cpu_input`,
  `random_crop_preserves_device_for_cpu_input` (3 CPU + 1
  cfg-cuda).

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-data --lib transforms:: 2>&1 | tail -3
```

Expected: ~22 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub trait Transform<T: Float>: Send + Sync` with `apply(&self, Tensor<T>) -> FerrotorchResult<Tensor<T>>` in `transforms.rs`, plus blanket `impl Transform for Box<dyn Transform<T>>`; non-test consumer: `pub use transforms::Transform` in `lib.rs`, and every other concrete struct in this file (`Compose`, `Normalize`, `ToTensor`, `RandomHorizontalFlip`, `RandomCrop`) is a `impl Transform<T>` so the trait has multiple production impls + the meta-crate glob propagates it. |
| REQ-2 | SHIPPED | impl: `pub struct Compose<T: Float>` + `impl Transform for Compose<T>` folding through `Vec<Box<dyn Transform<T>>>` in `transforms.rs`; non-test consumer: `pub use transforms::Compose` in `lib.rs`; downstream model-pipeline code constructs `Compose::new(vec![Box::new(Normalize::new(...)?), ...]).apply(input)`. |
| REQ-3 | SHIPPED | impl: `pub struct Normalize<T: Float>` + `Normalize::new(Vec<f64>, Vec<f64>)` validation + `impl Transform for Normalize<T>` using device-aware `sub` + `div` wrapped in `no_grad`, with the `.to(input.device())` ferry per #1107 in `transforms.rs`; non-test consumer: `pub use transforms::Normalize` in `lib.rs`, used as a Box-wrapped element of a `Compose` pipeline in downstream image-pipeline code; the `normalize_preserves_device_for_cpu_input` test pins the GPU-discipline contract. |
| REQ-4 | SHIPPED | impl: `pub struct ToTensor` + inherent `ToTensor::apply(&image::DynamicImage) -> FerrotorchResult<Tensor<f32>>` doing the HWC→CHW transpose + `/255` normalisation in `transforms.rs`; non-test consumer: `pub use transforms::ToTensor` in `lib.rs`, used in image-loading pipelines in downstream vision / diffusion model crates as the first transform in the user's `Compose` chain. |
| REQ-5 | SHIPPED | impl: `pub struct RandomHorizontalFlip<T: Float>` + `impl Transform` using `index_select_dim` with reverse-index list, staying device-resident per #1107 in `transforms.rs`; non-test consumer: `pub use transforms::RandomHorizontalFlip` in `lib.rs`, used as a Box-wrapped element of `Compose` in downstream augmentation pipelines; `random_horizontal_flip_preserves_device_for_cpu_input` pins the GPU-discipline contract. |
| REQ-6 | SHIPPED | impl: `pub struct RandomCrop<T: Float>` + `impl Transform` using `narrow().narrow().contiguous()` chain, staying device-resident in `transforms.rs`; non-test consumer: `pub use transforms::RandomCrop` in `lib.rs`, used as a Box-wrapped element of `Compose`; `random_crop_preserves_device_for_cpu_input` pins the contract. |
| REQ-7 | SHIPPED | impl: `pub fn manual_seed(seed: u64)` in `transforms.rs` plus `static GLOBAL_SEED: AtomicU64` + `static RNG_COUNTER: AtomicU64` + `fn random_f64() -> f64` (counter-based splitmix64); non-test consumer: `pub use transforms::manual_seed` in `lib.rs` (so callers can write `ferrotorch::manual_seed(42)`), and `fn RandomHorizontalFlip::apply` + `fn RandomCrop::apply` both call `random_f64()` internally — every random transform in this file routes through this primitive. |
