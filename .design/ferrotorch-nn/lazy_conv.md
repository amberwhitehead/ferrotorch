# ferrotorch-nn — `lazy_conv` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/conv.py
  - torch/nn/modules/lazy.py
-->

## Summary

`ferrotorch-nn/src/lazy_conv.rs` implements `LazyConv1d<T>`,
`LazyConv2d<T>`, and `LazyConv3d<T>` — the deferred-`in_channels`-
discovery variants of the convolution layers mirroring
`torch.nn.LazyConv{1,2,3}d` (exported from
`torch/nn/modules/conv.py:18-22` via the `LazyModuleMixin`
protocol at `torch/nn/modules/lazy.py`). Each lazy layer carries
the full kernel geometry (`out_channels`, `kernel_size`, `stride`,
`padding`, `bias`) up front and discovers `in_channels` from
`input.shape()[1]` on the first forward call, then constructs a
regular `ConvNd<T>` and delegates all subsequent forwards to it.
Thread-safe via `std::sync::OnceLock`.

## Requirements

- REQ-1: `pub struct LazyConv1d<T: Float>` carrying `out_channels`,
  `kernel_size: usize`, `stride: usize`, `padding: usize`,
  `bias_enabled: bool`, `inner: OnceLock<Conv1d<T>>`, and `training:
  AtomicBool`. Mirror `torch.nn.LazyConv1d` shape conventions.
- REQ-2: `pub struct LazyConv2d<T: Float>` analogous, with
  `kernel_size: (usize, usize)`, `stride: (usize, usize)`, `padding:
  (usize, usize)`, `inner: OnceLock<Conv2d<T>>`. Mirror
  `torch.nn.LazyConv2d`.
- REQ-3: `pub struct LazyConv3d<T: Float>` analogous, with
  `(usize, usize, usize)`-typed geometry fields and `inner:
  OnceLock<Conv3d<T>>`. Mirror `torch.nn.LazyConv3d`.
- REQ-4: Constructors validate `out_channels > 0`, `kernel_size > 0`
  in every dimension, `stride > 0` in every dimension. Mirror the
  same validation `Conv*d::new` does.
- REQ-5: `materialize(in_channels)` — eagerly construct the inner
  `ConvNd` with the given `in_channels` and the constructor-supplied
  geometry. Idempotent: first call wins.
- REQ-6: `<LazyConvNd as Module>::forward` validates `input.ndim()`
  matches the expected rank (3 for 1d, 4 for 2d, 5 for 3d). On
  first call, reads `in_channels = input.shape()[1]` and
  materializes. Delegates the actual convolution to the inner
  `ConvNd::forward`.
- REQ-7: `Module<T>` trait — `parameters()` / `parameters_mut()` /
  `named_parameters()` forward through `inner.get()` / `get_mut()`
  and return empty before materialization. `train` and `eval`
  cascade into `inner` once materialized.
- REQ-8: `is_initialized()` accessor for callers that need to know
  whether the parameters are ready.

## Acceptance Criteria

- [x] AC-1: Pre-init, `is_initialized() == false` and
  `parameters().len() == 0`.
- [x] AC-2: First forward materializes the inner conv;
  `is_initialized() == true` and `parameters().len() == 2` (or 1
  without bias).
- [x] AC-3: Wrong input rank rejected (`LazyConv1d` requires 3-D,
  `LazyConv2d` 4-D, `LazyConv3d` 5-D).
- [x] AC-4: `materialize(in_channels)` works without a forward.
- [x] AC-5: Subsequent forwards reuse the inner conv (verified via
  weight pointer equality).
- [x] AC-6: Zero `out_channels` rejected at construction.
- [x] AC-7: Train/eval propagates to inner.

## Architecture

### The structs (REQ-1, REQ-2, REQ-3)

Three structs in `lazy_conv.rs`. Each has the convention of
storing the geometry as scalar (1d) or fixed-size tuple (2d / 3d).
`OnceLock<Conv1d<T>>` / `OnceLock<Conv2d<T>>` /
`OnceLock<Conv3d<T>>` enable race-free first-init.

### Constructors (REQ-4)

`LazyConv1d::new` / `LazyConv2d::new` / `LazyConv3d::new` in
`lazy_conv.rs` validate `out_channels > 0`, every kernel-size
component > 0, every stride component > 0. The structure is
identical for all three with the dimension count being the only
difference.

### Materialization (REQ-5)

`LazyConvNd::materialize(in_channels)` in `lazy_conv.rs` validates
`in_channels > 0`, then if `inner.get().is_none()` constructs
`ConvNd::new(in_channels, out_channels, kernel_size, stride,
padding, bias_enabled)` and `set`s it into the OnceLock.
`Conv*d::new` does its own validation (kernel-size, stride,
groups=1 by default — note: `LazyConv*` cannot specify `groups`
because the constructor signature mirrors the simpler `Conv*::new`,
not `Conv*::new_full`).

### Forward (REQ-6)

`<LazyConvNd<T> as Module<T>>::forward` in `lazy_conv.rs`:
1. Validate `input.ndim() == expected_ndim`.
2. If inner is not yet materialized, read
   `in_channels = input.shape()[1]` and call `materialize`.
3. Delegate to `inner.get().expect(...).forward(input)`.

### Trait surface (REQ-7, REQ-8)

`impl<T: Float> Module<T> for LazyConvNd<T>` in `lazy_conv.rs`.
`parameters()` returns the inner's parameter list if materialized,
empty otherwise. `train` / `eval` toggle `training: AtomicBool` and
cascade into the inner conv if materialized.

### Non-test production consumers

- `pub use lazy_conv::{LazyConv1d, LazyConv2d, LazyConv3d}` at
  `ferrotorch-nn/src/lib.rs` exposes the types.
- Dynamic-shape vision pipelines (where the input channel count is
  determined by a preceding preprocessing pipeline rather than at
  model construction) use `LazyConv2d::new(out_channels,
  kernel_size, stride, padding, bias)` to defer the in-channels
  decision.

## Parity contract

`parity_ops = []`. `LazyConv*d` is not a parity-tested kernel; its
correctness is inherited from `Conv*d` (verified in
`.design/ferrotorch-nn/conv.md` — REQ-1..REQ-11) and the lazy-init
plumbing is verified by lib tests in this file.

Edge cases:
- **Wrong input rank** — both upstream and ferrotorch reject. The
  exact error type may differ (RuntimeError vs `InvalidArgument`),
  but the rejection occurs at the same shape-check.
- **Materialize with `in_channels == 0`** — both reject.
- **First-wins on `materialize`** — re-calling with a different
  `in_channels` is a no-op. Diverges from a naive "always
  re-initialize" implementation; the behavior matches
  `LazyLinear::materialize`.

## Verification

Tests in `mod tests` of `lazy_conv.rs` (~15 tests):
- LazyConv1d: `test_lazy_conv1d_uninitialized_until_first_forward`,
  `test_lazy_conv1d_materializes_on_first_forward`,
  `test_lazy_conv1d_rejects_wrong_input_ndim`,
  `test_lazy_conv1d_explicit_materialize`,
  `test_lazy_conv1d_zero_out_channels_errors`.
- LazyConv2d: `test_lazy_conv2d_uninitialized_until_first_forward`,
  `test_lazy_conv2d_materializes_on_first_forward`,
  `test_lazy_conv2d_no_bias`,
  `test_lazy_conv2d_subsequent_forward_reuses_inner`,
  `test_lazy_conv2d_rejects_wrong_ndim`,
  `test_lazy_conv2d_train_eval_propagates_to_inner`.
- LazyConv3d: `test_lazy_conv3d_uninitialized_until_first_forward`,
  `test_lazy_conv3d_materializes_on_first_forward`,
  `test_lazy_conv3d_rejects_wrong_ndim`,
  `test_lazy_conv3d_zero_kernel_errors`.

Smoke command:

```bash
cargo test -p ferrotorch-nn --lib lazy_conv:: 2>&1 | tail -3
```

Expected: 15 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LazyConv1d<T: Float>` in `lazy_conv.rs` with the deferred-init field layout; non-test consumer: `pub use lazy_conv::LazyConv1d` in `lib.rs`. |
| REQ-2 | SHIPPED | impl: `pub struct LazyConv2d<T: Float>` in `lazy_conv.rs`; non-test consumer: `pub use lazy_conv::LazyConv2d` in `lib.rs`. |
| REQ-3 | SHIPPED | impl: `pub struct LazyConv3d<T: Float>` in `lazy_conv.rs`; non-test consumer: `pub use lazy_conv::LazyConv3d` in `lib.rs`. |
| REQ-4 | SHIPPED | impl: `LazyConvNd::new` bodies in `lazy_conv.rs` validating `out_channels > 0`, kernel/stride > 0; non-test consumer: dynamic-shape pipeline construction in downstream vision code. |
| REQ-5 | SHIPPED | impl: `LazyConvNd::materialize(in_channels)` body constructing the inner `ConvNd::new(...)` in `lazy_conv.rs`; non-test consumer: dynamic-shape pipelines call `materialize(known_in_channels)` to populate parameters before constructing the optimizer. |
| REQ-6 | SHIPPED | impl: `<LazyConvNd as Module>::forward` body in `lazy_conv.rs` (ndim check + first-call materialize + delegate to inner); non-test consumer: any model containing a `LazyConv2d` runs this on every training forward. |
| REQ-7 | SHIPPED | impl: `Module<T>` impl forwarding `parameters` / `parameters_mut` / `named_parameters` through `inner.get()` in `lazy_conv.rs`; non-test consumer: `ferrotorch_optim::Optimizer` walks `model.parameters_mut()` and sees the inner Conv's params after the first forward materializes. |
| REQ-8 | SHIPPED | impl: `LazyConvNd::is_initialized` accessor in `lazy_conv.rs`; non-test consumer: training-loop setup code that queries `is_initialized` to decide whether to call `materialize` explicitly. |
