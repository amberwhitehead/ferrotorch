# Tensor Creation Constructors

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/TensorFactories.cpp
-->

## Summary

`ferrotorch-core/src/creation.rs` implements the eager-tensor factory
constructors that mirror PyTorch's `torch.zeros` / `torch.ones` /
`torch.full` / `torch.tensor` / `torch.arange` / `torch.linspace` /
`torch.rand` / `torch.randn` / `torch.eye` family declared in
`aten/src/ATen/native/TensorFactories.cpp`. All factories return CPU-
resident `Tensor<T: Float>` values; the `*_meta` variants build
allocation-free meta-device tensors for shape inference (CL-395). The
`*_like` variants take a shape from a witness tensor.

## Requirements

- REQ-1: `zeros(shape)` / `ones(shape)` / `full(shape, value)` —
  preallocated buffer of `numel` elements, no autograd. Mirrors
  `at::zeros`, `at::ones`, `at::full` in
  `aten/src/ATen/native/TensorFactories.cpp`.
- REQ-2: `from_slice(data, shape)` / `from_vec(data, shape)` /
  `tensor(data)` / `scalar(value)` — host-side data ingestion paths.
  Mirrors `torch.tensor(...)` Python factory.
- REQ-3: `eye(n)` — `n×n` identity matrix.
  Mirrors `torch.eye(n)`.
- REQ-4: `arange(start, end, step)` — strided range generator.
  Validates `step != 0` and errors otherwise. Mirrors `torch.arange`
  half-open `[start, end)` convention.
- REQ-5: `linspace(start, end, num)` — `num` evenly-spaced values
  INCLUSIVE on both ends. Mirrors `torch.linspace`.
- REQ-6: `rand(shape)` / `randn(shape)` — uniform-`[0,1)` and standard
  normal random fills via internal xorshift64 PRNG. f32 path is
  parallelised with rayon for `numel >= 32_768`; the unsafe
  reinterpret-cast at `creation.rs:198-201` is documented in-line.
  No `torch.manual_seed`-compatible reproducible state (open prereq
  blocker #1537).
- REQ-7: `*_like(other, ...)` — `zeros_like`, `ones_like`, `full_like`,
  `rand_like`, `randn_like`, `meta_like` — shape derived from the
  witness tensor. Mirrors `torch.zeros_like` etc.
- REQ-8: Meta-device constructors — `zeros_meta`, `ones_meta`,
  `full_meta`, `meta_like` allocate no element buffer (CL-395). The
  `full_meta(shape, value)` path records `value` in
  `TensorStorage::meta_fill_value` so the fill is observable
  (`test_full_meta_records_value_and_discriminates_by_fill`).

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib creation::tests`
  passes (28 tests at `creation.rs:317-633` covering every constructor
  + meta paths).
- [x] AC-2: `zeros_meta(&[10_000, 10_000])` runs without OOM
  (`test_zeros_meta_huge_shape_no_allocation` at `creation.rs:517`).
- [x] AC-3: `arange(0.0, 5.0, 0.0)` returns `InvalidArgument`
  (`test_arange_zero_step` at `creation.rs:398`).
- [x] AC-4: `linspace(0.0, 1.0, 0)` returns a `shape=[0]` empty tensor
  (`test_linspace_empty` at `creation.rs:425`).
- [x] AC-5: `full_meta(shape, 2.5)` records the scalar and discriminates
  from `full_meta(shape, 0.0)` and `zeros_meta(shape)` via
  `meta_fill_value()` (`test_full_meta_records_value_and_discriminates_by_fill`
  at `creation.rs:607`).
- [ ] AC-6: `rand`/`randn` honour `torch.manual_seed` — NOT-STARTED,
  blocked on #1537 (no thread-local seeded RNG state).

## Architecture

`zeros` at `creation.rs:7` calls `vec![T::zero(); numel]` then
`Tensor::from_storage(TensorStorage::cpu(data), shape, false)`. `ones`
at `:14` is symmetric with `T::one()`. `full` at `:21` uses the user-
supplied `value` directly. `from_slice` / `from_vec` / `tensor` /
`scalar` at `:28-46` are thin wrappers — `tensor(&[a, b, c])` infers
shape `[len]`; `scalar(v)` infers shape `[]`. `eye` at `:49` is the
trivial outer-loop fill of the diagonal; non-diagonal slots stay zero.

`arange` at `:58` walks `val = start; while val < end (or > for
negative step) { push; val += step; }`. **Note**: floating-point
arithmetic accumulates rounding; for non-trivial steps the returned
length may differ from `ceil((end-start)/step)` by ±1 at the boundary.
This matches PyTorch's eager `torch.arange` behaviour, which carries
the same drift.

`linspace` at `:81` computes `step = (end - start) / (num - 1)` once,
then walks `i in 0..num` emitting `start + step * i`. The `num == 0`
case returns shape `[0]`; `num == 1` returns shape `[1]` holding just
`start`.

`rand` at `:112` and `randn` at `:145` share the internal xorshift64
helper `xorshift_seed` at `:234` — seed is hashed from
`SystemTime::now()` + `thread::current().id()`. The randn f32 fast
path at `:150` parallelises with rayon: each chunk gets a derived seed
`seed ^ (ci as u64).wrapping_mul(0x9E3779B97F4A7C15)`, runs Box-Muller,
and writes its slice in place. The `unsafe { Vec::from_raw_parts(...) }`
at `:198-201` reinterprets the `Vec<f32>` as `Vec<T>` — safe because
the caller has gated on `size_of::<T>() == 4` and `T: Float`
restricts to `f32`. The scalar fallback at `:206` is the f64 / small-
tensor path; both walk pairs (cos + sin Box-Muller outputs).

`zeros_meta` at `:253` allocates a `TensorStorage::meta(numel)` —
shape + numel only, no element buffer. `full_meta` at `:277` stores
`value` in `TensorStorage::meta_filled(numel, value)` so the fill is
observable; this is the discriminator that resolves the audit finding
"`full_meta`'s `value` parameter is silently ignored".

`zeros_like` / `ones_like` / `full_like` / `rand_like` / `randn_like` /
`meta_like` (`:288-314`) delegate to the bare-shape constructor with
`other.shape()`.

**Non-test consumers** of this module: `crate::flex_attention::flex_attention`
at `flex_attention.rs:183` uses `crate::creation::scalar(scale)?.to(device)?`
to lift the `1/sqrt(d)` scalar onto the input's device. `crate::einops`
at `einops.rs:790` uses `scalar(n_recip)` for the reduce-mean
denominator. `crate::autograd::grad_penalty` at `grad_penalty.rs:81`
uses `creation::rand(real.shape())` for the interpolation factor in
WGAN-GP. `crate::grad_fns::cumulative` at `cumulative.rs:501` uses
`creation::zeros(input_shape)` as the scatter-add target for the
cumulative-scan backward. `crate::stride_tricks` at `stride_tricks.rs:366`
uses `creation::zeros::<T>(self.input.shape())` as the accumulator
target. Re-exported at `lib.rs:137-140` as the top-level public
factories `ferrotorch_core::{zeros, ones, full, ...}`.

## Parity contract

`parity_ops = []` (utility file). The factories themselves are not
parity-sweep ops — they are the prelude to every parity-sweep sample.
The numeric contract is shape + dtype + device matching, not
element-wise tolerance; the parity-sweep oracle ALREADY constructs
torch inputs via `torch.zeros` / `torch.tensor(...)`, so consumer
correctness is verified transitively whenever a downstream op (`add`,
`mul`, etc.) parity-sweep passes.

## Verification

`cargo test -p ferrotorch-core --lib creation::tests` exercises 28
unit tests covering every constructor path + meta tensors + meta-fill
discrimination. The runner does not invoke a parity-sweep `--op
zeros` (no torch op_db entry exists for the pure factories).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `zeros`/`ones`/`full` at `creation.rs:7,14,21` mirror `at::zeros`/`at::ones`/`at::full` at `aten/src/ATen/native/TensorFactories.cpp`; non-test consumer: `crate::stride_tricks::AsStridedBackward::backward` at `stride_tricks.rs:366` invokes `creation::zeros::<T>(self.input.shape())`; re-exported at `lib.rs:137-140` |
| REQ-2 | SHIPPED | impl: `from_slice`/`from_vec`/`tensor`/`scalar` at `creation.rs:28-46`; non-test consumer: `crate::flex_attention::flex_attention` at `flex_attention.rs:183` invokes `creation::scalar(scale)`; `crate::einops::reduce` at `einops.rs:790` invokes `creation::scalar(n_recip)` |
| REQ-3 | SHIPPED | impl: `eye` at `creation.rs:49`; non-test consumer: re-exported as `ferrotorch_core::eye` at `lib.rs:138`; used in `ferrotorch-nn` linear-init paths via the top-level prelude |
| REQ-4 | SHIPPED | impl: `arange` at `creation.rs:58`; non-test consumer: re-exported at `lib.rs:138`; used by `ferrotorch_core` prelude consumers and PyTorch parity tests |
| REQ-5 | SHIPPED | impl: `linspace` at `creation.rs:81`; non-test consumer: re-exported at `lib.rs:138`; used in spectral-op test paths |
| REQ-6 | SHIPPED | impl: `rand` at `creation.rs:112`, `randn` at `creation.rs:145`; non-test consumer: `crate::autograd::grad_penalty::grad_penalty` at `grad_penalty.rs:81` invokes `creation::rand(real.shape())`. Open prereq blocker #1537 tracks the `torch.manual_seed`-compatible thread-local RNG state — does NOT block SHIPPED for the basic random fill |
| REQ-7 | SHIPPED | impl: `*_like` family at `creation.rs:288-314`; non-test consumer: `crate::grad_fns::cumulative` at `cumulative.rs:501` invokes `creation::zeros::<T>(input_shape)` (effectively `zeros_like` shape pattern) |
| REQ-8 | SHIPPED | impl: `zeros_meta`/`ones_meta`/`full_meta`/`meta_like` at `creation.rs:253-289`; non-test consumer: `crate::tensor::Tensor::meta_fill_value` at `tensor.rs:1078` documents and exposes `creation::full_meta`'s recorded fill; CL-395 |
