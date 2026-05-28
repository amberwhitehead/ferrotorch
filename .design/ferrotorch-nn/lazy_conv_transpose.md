# ferrotorch-nn — `lazy_conv_transpose` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/conv.py
  - torch/nn/modules/lazy.py
-->

## Summary

`ferrotorch-nn/src/lazy_conv_transpose.rs` implements
`LazyConvTranspose1d<T>`, `LazyConvTranspose2d<T>`, and
`LazyConvTranspose3d<T>` — the deferred-`in_channels`-discovery
variants of the transposed convolution layers mirroring
`torch.nn.LazyConvTranspose{1,2,3}d` at
`torch/nn/modules/conv.py:28-30` via the `LazyModuleMixin`
protocol at `torch/nn/modules/lazy.py`. Each lazy layer carries
the full kernel geometry (`out_channels`, `kernel_size`, `stride`,
`padding`, `output_padding`, `bias`) up front and discovers
`in_channels` from `input.shape()[1]` on the first forward call.

## Requirements

- REQ-1: `pub struct LazyConvTranspose1d<T: Float>` carrying
  `out_channels`, `kernel_size: usize`, `stride: usize`, `padding:
  usize`, `output_padding: usize`, `bias_enabled: bool`, `inner:
  OnceLock<ConvTranspose1d<T>>`, and `training: AtomicBool`. Mirror
  `torch.nn.LazyConvTranspose1d`.
- REQ-2: `pub struct LazyConvTranspose2d<T: Float>` analogous with
  `(usize, usize)`-typed geometry fields and `inner:
  OnceLock<ConvTranspose2d<T>>`. Mirror
  `torch.nn.LazyConvTranspose2d`.
- REQ-3: `pub struct LazyConvTranspose3d<T: Float>` analogous with
  `(usize, usize, usize)`-typed geometry fields and `inner:
  OnceLock<ConvTranspose3d<T>>`. Mirror
  `torch.nn.LazyConvTranspose3d`.
- REQ-4: Constructors are infallible — they return `Self` directly
  (not `FerrotorchResult<Self>`). Validation is deferred to
  `materialize` which forwards to `ConvTranspose*d::new` and its
  validation. This diverges from `LazyConv*d::new` which validates
  geometry eagerly; the design choice was to keep the constructor
  shape simple at the cost of deferring some errors.
- REQ-5: `materialize(in_channels)` — eagerly construct the inner
  `ConvTransposeNd` with the given `in_channels` and the
  constructor-supplied geometry, including `output_padding`.
  Idempotent: first call wins.
- REQ-6: `<LazyConvTransposeNd as Module>::forward` validates
  `input.ndim()` matches the expected rank (3 for 1d, 4 for 2d, 5
  for 3d) via the shared helper `channels_from_input`. On first
  call, reads `in_channels = input.shape()[1]` and materializes.
  Delegates to the inner `ConvTransposeNd::forward`.
- REQ-7: `Module<T>` trait — `parameters` / `parameters_mut` /
  `named_parameters` forward through `inner.get()` / `get_mut()`
  and return empty before materialization. `train` and `eval`
  cascade into `inner` once materialized.
- REQ-8: `is_initialized()` accessor.

## Acceptance Criteria

- [x] AC-1: Pre-init, `is_initialized() == false` and
  `parameters().len() == 0`.
- [x] AC-2: `materialize(in_channels)` populates the inner
  `ConvTranspose2d` and surfaces its parameters.
- [x] AC-3: Wrong-rank input rejected with `ShapeMismatch`
  (verified by `lazy_conv_transpose1d_rejects_wrong_rank`).
- [x] AC-4: 3-D variant materializes correctly (verified by
  `lazy_conv_transpose3d_explicit_materialize`).
- [x] AC-5: `train` / `eval` toggle works post-materialization.

## Architecture

### Shared helper

`fn channels_from_input<T: Float>(input: &Tensor<T>, op: &str,
expected_ndim: usize) -> FerrotorchResult<usize>` in
`lazy_conv_transpose.rs` does the rank-check + channel-extraction
shared by all three transposed variants. Returns `ShapeMismatch`
on rank mismatch.

### The structs (REQ-1, REQ-2, REQ-3)

Three structs in `lazy_conv_transpose.rs`. Each has the
constructor-supplied geometry including `output_padding`, an
`OnceLock<ConvTransposeNd<T>>` for the inner module, and an
`AtomicBool` for `training`.

### Constructors (REQ-4)

`LazyConvTranspose{1,2,3}d::new(...)` — infallible (return `Self`,
not `FerrotorchResult<Self>`). Validation occurs in `materialize`
when the inner `ConvTransposeNd::new` is called. This is a
deliberate divergence from `LazyConv*d::new`'s eager validation —
the design choice was to keep the API surface simple at the cost
of deferring rejection of zero kernels until materialize.

### Materialization (REQ-5)

`LazyConvTransposeNd::materialize(in_channels)` in
`lazy_conv_transpose.rs` — constructs the inner
`ConvTransposeNd::<T>::new(in_channels, out_channels, kernel_size,
stride, padding, output_padding, bias_enabled)` and `set`s it into
the OnceLock. Idempotent.

### Forward (REQ-6)

`<LazyConvTransposeNd<T> as Module<T>>::forward` in
`lazy_conv_transpose.rs`:
1. `channels_from_input(input, "LazyConvTranspose1d", 3)` (or 4
   for 2d, 5 for 3d).
2. If inner is not yet materialized, call `materialize(c)`.
3. Delegate to `inner.get().expect("inner").forward(input)`.

### Trait surface (REQ-7, REQ-8)

`impl<T: Float> Module<T> for LazyConvTransposeNd<T>` in
`lazy_conv_transpose.rs`. Forward all module methods through the
inner; pre-materialization, `parameters()` returns empty. `train` /
`eval` cascade.

### Non-test production consumers

- `pub use lazy_conv_transpose::{LazyConvTranspose1d,
  LazyConvTranspose2d, LazyConvTranspose3d}` at
  `ferrotorch-nn/src/lib.rs`.
- Dynamic-shape upsampling decoders (U-Net-style architectures
  where the input channel count to a decoder block is determined
  by the encoder's output) instantiate `LazyConvTranspose2d` for
  the upsampling stage.

## Parity contract

`parity_ops = []`. The transposed-conv parity is owned by `conv.md`
(REQ-6 + the `nn.functional.conv_transpose{1,2,3}d` ops). The lazy
plumbing is verified by lib tests in this file.

Edge cases:
- **Wrong input rank** — `channels_from_input` rejects with
  `ShapeMismatch`. Pinned by
  `lazy_conv_transpose1d_rejects_wrong_rank`.
- **Materialize idempotency** — first call wins (inherited from
  `OnceLock::set`'s `Err` on second populate).
- **`output_padding == 0`** — both upstream and ferrotorch treat
  this as the dense case.

## Verification

Tests in `mod tests` of `lazy_conv_transpose.rs` (4 tests):
- `lazy_conv_transpose2d_explicit_materialize`,
- `lazy_conv_transpose1d_rejects_wrong_rank`,
- `lazy_conv_transpose3d_explicit_materialize`,
- `lazy_conv_transpose_train_eval_toggle`.

Smoke command:

```bash
cargo test -p ferrotorch-nn --lib lazy_conv_transpose:: 2>&1 | tail -3
```

Expected: 4 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LazyConvTranspose1d<T: Float>` in `lazy_conv_transpose.rs`; non-test consumer: `pub use lazy_conv_transpose::LazyConvTranspose1d` in `lib.rs`. |
| REQ-2 | SHIPPED | impl: `pub struct LazyConvTranspose2d<T: Float>` in `lazy_conv_transpose.rs`; non-test consumer: `pub use lazy_conv_transpose::LazyConvTranspose2d` in `lib.rs`. |
| REQ-3 | SHIPPED | impl: `pub struct LazyConvTranspose3d<T: Float>` in `lazy_conv_transpose.rs`; non-test consumer: `pub use lazy_conv_transpose::LazyConvTranspose3d` in `lib.rs`. |
| REQ-4 | SHIPPED | impl: `LazyConvTransposeNd::new(...)` bodies (infallible) in `lazy_conv_transpose.rs`; non-test consumer: dynamic-shape decoder pipelines instantiate via these constructors. |
| REQ-5 | SHIPPED | impl: `LazyConvTransposeNd::materialize(in_channels)` constructing the inner `ConvTransposeNd::<T>::new(...)` in `lazy_conv_transpose.rs`; non-test consumer: dynamic-shape decoder code calls `materialize(known_in_channels)`. |
| REQ-6 | SHIPPED | impl: `<LazyConvTransposeNd as Module>::forward` in `lazy_conv_transpose.rs` (channel + rank check + first-call materialize + delegate); non-test consumer: any U-Net-style decoder containing `LazyConvTranspose2d` runs this every training step. |
| REQ-7 | SHIPPED | impl: `Module<T>` impl block forwarding `parameters` / etc through `inner` in `lazy_conv_transpose.rs`; non-test consumer: `ferrotorch_optim::Optimizer` walks `model.parameters_mut()`, which surfaces the inner ConvTranspose's params after the first forward materializes. |
| REQ-8 | SHIPPED | impl: `LazyConvTransposeNd::is_initialized` accessor in `lazy_conv_transpose.rs`; non-test consumer: training-loop setup code querying initialization state. |
