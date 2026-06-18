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
  normal random fills sourced from the process-global
  `ferrotorch_core::rng::Generator` (MT19937 + Box-Muller). Calling
  `ferrotorch_core::manual_seed(s)` makes both functions deterministic
  and produces byte-exact agreement with `torch.manual_seed(s);
  torch.rand(...)` for f32 (uniform). `randn` deterministic
  reproducibility is guaranteed; byte-exact randn parity is documented
  as a separate divergence (torch uses different algorithms for `size <
  16` cpu_serial vs `>= 16` `normal_fill` SIMD blocks). #1537 closed.
- REQ-7: `*_like(other, ...)` — `zeros_like`, `ones_like`, `full_like`,
  `rand_like`, `randn_like`, `meta_like` — shape derived from the
  witness tensor. Mirrors `torch.zeros_like` etc.
- REQ-8: Meta-device constructors — `zeros_meta`, `ones_meta`,
  `full_meta`, `meta_like` allocate no element buffer (CL-395). The
  `full_meta(shape, value)` path records `value` in
  `TensorStorage::meta_fill_value` so the fill is observable
  (`test_full_meta_records_value_and_discriminates_by_fill`).
- REQ-9: `rand_on_device(shape, device)` / `randn_on_device(shape,
  device)` — device-aware random fills (#1682). For `Device::Cuda` +
  f32/f64/f16/bf16 the values are generated through dtype-specific
  GPU backend RNG slots and returned as CUDA-resident tensors (no CPU
  generate-then-upload), mirroring `torch.rand(size, device='cuda')`
  = `at::empty(size, options).uniform_(0, 1)`
  (`TensorFactories.cpp:1075-1076`) and `torch.randn(..., device='cuda')`
  = `at::empty(...).normal_(0, 1)` (`TensorFactories.cpp:1379`).
  CPU and Meta paths keep the CPU generator behaviour. Reproducible
  after `manual_seed` (which now seeds the GPU generator too).

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib creation::tests`
  passes (28 tests at `creation.rs` covering every constructor
  + meta paths).
- [x] AC-2: `zeros_meta(&[10_000, 10_000])` runs without OOM
  (`test_zeros_meta_huge_shape_no_allocation in creation.rs`).
- [x] AC-3: `arange(0.0, 5.0, 0.0)` returns `InvalidArgument`
  (`test_arange_zero_step in creation.rs`).
- [x] AC-4: `linspace(0.0, 1.0, 0)` returns a `shape=[0]` empty tensor
  (`test_linspace_empty in creation.rs`).
- [x] AC-5: `full_meta(shape, 2.5)` records the scalar and discriminates
  from `full_meta(shape, 0.0)` and `zeros_meta(shape)` via
  `meta_fill_value()` (`test_full_meta_records_value_and_discriminates_by_fill`
  at `test_full_meta_records_value_and_discriminates_by_fill in creation.rs`).
- [x] AC-6: `rand`/`randn` honour `torch.manual_seed` — SHIPPED via
  `ferrotorch_core::manual_seed` (#1537 closed). `rand` is byte-exact
  vs torch for f32; `randn` is deterministic under seed (pinned by
  `ferrotorch-core/tests/divergence_manual_seed_parity.rs`).

## Architecture

`zeros in creation.rs` calls `vec![T::zero(); numel]` then
`Tensor::from_storage(TensorStorage::cpu(data), shape, false)`. `ones`
at `one in creation.rs` is symmetric with `T::one()`. `full in creation.rs` uses the user-
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

`rand` at `creation.rs:283` and `randn` at `creation.rs:312` draw through
`crate::rng::with_thread_rng`, which serializes one process-global MT19937
default generator behind a mutex. `rand` uses the f32 uniform transform for
f32 tensors and the f64 transform for wider floating tensors. `randn` uses the
same default generator plus the Box-Muller normal helper in `Generator`; it is
deterministic under `manual_seed`, while bit-exact `torch.randn` parity remains
documented separately because PyTorch switches CPU normal algorithms by output
size and SIMD availability.

`zeros_meta` at `:253` allocates a `TensorStorage::meta(numel)` —
shape + numel only, no element buffer. `full_meta` at `:277` stores
`value` in `TensorStorage::meta_filled(numel, value)` so the fill is
observable; this is the discriminator that resolves the audit finding
"`full_meta`'s `value` parameter is silently ignored".

`zeros_like` / `ones_like` / `full_like` / `rand_like` / `randn_like` /
`meta_like` (`:288-314`) delegate to the bare-shape constructor with
`other.shape()`.

**Non-test consumers** of this module: `crate::flex_attention::flex_attention`
at `flex_attention in flex_attention.rs` uses `crate::creation::scalar(scale)?.to(device)?`
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
| REQ-1 | SHIPPED | impl: `zeros`/`ones`/`full in creation.rs,14,21` mirror `at::zeros`/`at::ones`/`at::full` at `aten/src/ATen/native/TensorFactories.cpp`; non-test consumer: `crate::stride_tricks::AsStridedBackward::backward` at `backward in stride_tricks.rs` invokes `creation::zeros::<T>(self.input.shape())`; re-exported at `stride_tricks in lib.rs` |
| REQ-2 | SHIPPED | impl: `from_slice`/`from_vec`/`tensor`/`scalar in creation.rs`; non-test consumer: `crate::flex_attention::flex_attention` at `flex_attention in flex_attention.rs` invokes `creation::scalar(scale)`; `crate::einops::reduce` at `reduce in einops.rs` invokes `creation::scalar(n_recip)` |
| REQ-3 | SHIPPED | impl: `eye` at `creation.rs:49`; non-test consumer: re-exported as `ferrotorch_core::eye` at `lib.rs:138`; used in `ferrotorch-nn` linear-init paths via the top-level prelude |
| REQ-4 | SHIPPED | impl: `arange` at `creation.rs:58`; non-test consumer: re-exported at `lib.rs:138`; used by `ferrotorch_core` prelude consumers and PyTorch parity tests |
| REQ-5 | SHIPPED | impl: `linspace` at `creation.rs:81`; non-test consumer: re-exported at `lib.rs:138`; used in spectral-op test paths |
| REQ-6 | SHIPPED | impl: `rand` at `creation.rs:283`, `randn` at `creation.rs:312` source from `crate::rng::with_thread_rng`; non-test consumer: `crate::autograd::grad_penalty::grad_penalty` at `autograd/grad_penalty.rs:94` invokes `creation::rand(real.shape())`. Process-global MT19937 + fallible all-device `manual_seed` ship at `rng.rs` (#1537, #1788, #1789); byte-exact vs `torch.manual_seed(42); torch.rand(10)` for f32 (`divergence_manual_seed_parity.rs:manual_seed_42_rand_byte_exact_vs_torch_f32`). |
| REQ-9 | SHIPPED | Device-aware on-device RNG (#1682). impl: `pub fn rand_on_device` / `pub fn randn_on_device in creation.rs` — for `Device::Cuda` + f32/f64/f16/bf16 they call the dtype-specific `GpuBackend::rand_uniform_*` / `randn_normal_*` slots and wrap the result `TensorStorage::gpu(handle)` with NO host round trip (mirrors `torch.rand(size, device='cuda')` = `at::empty(...).uniform_(0,1)` at `aten/src/ATen/native/TensorFactories.cpp:1075-1076`). Non-test consumer: `ferrotorch/examples/ferrotorch_bench.rs` GPU-rand bench calls `rand_on_device::<f32>(.., Device::Cuda(0))`; also re-exported at `pub use creation in lib.rs`. Reproducibility coupled to `crate::manual_seed` via the GPU seed path; verified on RTX 3090 by `ferrotorch-gpu/tests/on_device_rng.rs`. |
| REQ-7 | SHIPPED | impl: `*_like` family at `creation.rs:288-314`; non-test consumer: `crate::grad_fns::cumulative` at `cumulative.rs:501` invokes `creation::zeros::<T>(input_shape)` (effectively `zeros_like` shape pattern) |
| REQ-8 | SHIPPED | impl: `zeros_meta`/`ones_meta`/`full_meta`/`meta_like in creation.rs`; non-test consumer: `crate::tensor::Tensor::meta_fill_value` at `meta_fill_value in tensor.rs` documents and exposes `creation::full_meta`'s recorded fill; CL-395 |
