# ferrotorch-nn — `padding` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/padding.py
  - aten/src/ATen/native/PadNd.cpp
-->

## Summary

`ferrotorch-nn/src/padding.rs` implements the padding-layer family
mirroring `torch.nn.{Constant,Zero,Reflection,Replication,Circular}
Pad{1,2,3}d` at `torch/nn/modules/padding.py`. Each pads the last N
dimensions of the input tensor using the named mode. Also exposes
the `PaddingMode` enum used by conv layers'  prospective
`padding_mode` kwarg and the functional helpers
`functional_pad_1d/2d/3d`. The module is CPU-only — no GPU
dispatch.

## Requirements

- REQ-1: `pub enum PaddingMode { Zeros, Reflect, Replicate,
  Circular }` — the shared padding-mode selector used by both the
  layer family and the conv layers' `padding_mode` kwarg. Mirrors
  upstream's string-literal `padding_mode` argument values.
- REQ-2: `pub fn functional_pad_1d/2d/3d` — the functional
  entrypoints applying padding to the last 1/2/3 dimensions of a
  tensor with a given `PaddingMode`. Mirrors
  `torch.nn.functional.pad(input, pad, mode, value)` at
  `torch/nn/functional.py`. The constant (`Zeros`) path threads the
  caller-supplied `value: T` through to the crop-capable signed path —
  the out-of-bounds fill is the caller's `value` (defaulting to `0`
  only when the caller passes `T::zero()`), matching
  `torch.nn.functional.pad(mode="constant", value=...)`. The
  `usize`-typed entrypoints handle only non-negative (grow) pads; the
  signed entrypoints `functional_pad_1d/2d/3d_signed` (`isize` pad
  amounts) additionally support NEGATIVE (crop) pads under
  `mode="constant"`, mirroring `constant_pad_nd`'s negative-pad
  narrowing (`aten/src/ATen/native/PadNd.cpp:29-108`). Crop is a
  constant-mode-only capability — reflect/replicate/circular reject
  negative pads, matching upstream `_pad_enum`
  (`PadNd.cpp:221-242`).
- REQ-3: `pub struct ConstantPad{1,2,3}d<T: Float>` — constant-value
  padding. Mirror upstream `ConstantPad{1,2,3}d` at `padding.py`.
  Carries `padding: (usize, usize[, …])` and `value: T`.
- REQ-4: `pub struct ZeroPad{1,2,3}d<T: Float>` — zero padding
  (special case of ConstantPad with `value = 0`). Mirrors upstream
  `ZeroPad{1,2,3}d` at `padding.py`.
- REQ-5: `pub struct ReflectionPad{1,2,3}d<T: Float>` — reflection
  padding (`abc | dcba` style). Mirrors upstream `ReflectionPad`.
  Validates `pad < input_dim` (a hard requirement of the reflect
  algorithm).
- REQ-6: `pub struct ReplicationPad{1,2,3}d<T: Float>` — replicate
  padding (`aaaa | bcde | eeee` style). Mirrors upstream
  `ReplicationPad`.
- REQ-7: `pub struct CircularPad{1,2,3}d<T: Float>` — circular
  padding (`xyzd | abcd | abcx` wrap-around). Mirrors upstream
  `CircularPad`.
- REQ-8: All padding layers impl `Module<T>` with `forward = pad`,
  no parameters (`parameters()` returns `vec![]`), and
  `train`/`eval`/`is_training`.
- REQ-9: NOT-STARTED — the `nn.functional.pad` arm IS wired in
  `tools/parity-sweep/runner/src/main.rs` (#1441): it decodes the pad
  tuple (2→1d, 4→2d, 6→3d), the mode (constant/reflect/replicate/
  circular), and the value (positional or kwarg), dispatching to
  `functional_pad_{1,2,3}d`. Sweep `--seeds 8`: 376/408, 0 failed; the
  32 skips are ALL negative-pad (crop) samples — a genuine production
  gap (`functional_pad_{1,2,3}d` take `usize`, cannot crop), filed as
  blocker #1611. op_db emits only `mode='constant'` for
  `nn.functional.pad`. The OTHER 5 pad ops
  (`nn.functional.constant_pad_nd`, `reflection_pad{1,2}d`,
  `replication_pad{1,2}d`) still have NO runner arm, so REQ-9 stays
  NOT-STARTED until those arms land under #1441.

## Acceptance Criteria

- [x] AC-1: `PaddingMode` enum present with 4 variants.
- [x] AC-2: `functional_pad_{1,2,3}d` accept `PaddingMode` arg and
  dispatch to the correct internal `pad_Nd_<mode>` helper.
- [x] AC-3: All 12 padding-layer structs (4 modes × 3 dims) present
  and impl `Module<T>`.
- [x] AC-4: Forward output shape matches `(input_shape ..
  L+left+right)` for 1D and analogues for 2D/3D.
- [x] AC-5: `pad_*d_reflect` rejects `pad >= input_dim` with
  `InvalidArgument`.
- [x] AC-8: Constant (`Zeros`) path fills out-of-bounds positions
  with the caller's `value: T` (not a hardcoded `0`), covered by
  `test_functional_pad_{1,2,3}d_constant_uses_value` (#1553).
- [ ] AC-6: GPU forward — currently CPU-only. (Not declared as a
  REQ; GPU-side padding kernels are tracked elsewhere.)
- [ ] AC-7: parity-sweep arms wired — #1441. `nn.functional.pad` is
  wired (376/408, 0 failed; 32 negative-pad skips → #1611); the other
  5 pad ops remain unwired, so AC-7 stays open.

## Architecture

### Low-level helpers (REQ-1, REQ-2)

The internal `fn`s `pad_1d_constant`, `pad_2d_constant`,
`pad_3d_constant`, `pad_1d_reflect`, `pad_2d_reflect`,
`pad_3d_reflect`, `pad_1d_replicate`, `pad_2d_replicate`,
`pad_3d_replicate`, `pad_1d_circular`, `pad_2d_circular`, and
`pad_3d_circular` operate on raw `&[T]` data and `&[usize]` shape,
returning `(Vec<T>, Vec<usize>)`. Each touches only the last N
dimensions of the buffer. The constant helpers take a `value: T`
arg and seed the output buffer with `vec![value; …]` before copying
the source in — mirroring upstream
`aten/src/ATen/native/PadNd.cpp:94`'s `output.fill_(value)` then
`copy_(source)`.

The first constant helper is `fn pad_1d_constant` in `padding.rs`;
the 2-D and 3-D variants (`fn pad_2d_constant` / `fn pad_3d_constant`)
follow the same `value`-seeding shape.

### Functional entrypoints (REQ-2)

The `usize`-typed (grow-only) entrypoints are `pub fn
functional_pad_1d`, `pub fn functional_pad_2d`, and `pub fn
functional_pad_3d` in `padding.rs`. Each dispatches on
`PaddingMode::{Zeros, Reflect, Replicate, Circular}`. The
`Reflect`/`Replicate`/`Circular` arms ignore `value` (the fill is
gathered from existing data) and use the positive-only
`pad_Nd_<mode>` helpers + `Pad{1,2,3}dBackward` adjoint. The `Zeros`
arm (torch `mode="constant"`) is dispatched through the crop-capable
signed path (see below), the single source of truth for constant
padding — mirroring torch routing `mode="constant"` through
`constant_pad_nd` (`aten/src/ATen/native/PadNd.cpp:214-215`). For a
non-negative `usize` pad the signed forward is byte-identical to the
old `pad_Nd_constant` and its scatter-add backward equals the old
`Pad{1,2,3}dBackward` adjoint; the caller's `value: T` fill is
preserved (#1553). The `ZeroPad{1,2,3}d` layers obtain zero-fill by
explicitly passing `T::zero()`. Tests cover all four arms — including
`test_functional_pad_{1,2,3}d_constant_uses_value`, which asserts a
non-zero `value` reaches the padded cells.

The crop-capable (signed) entrypoints are `pub fn
functional_pad_1d_signed`, `pub fn functional_pad_2d_signed`, and
`pub fn functional_pad_3d_signed` in `padding.rs`, taking `isize` pad
amounts. They delegate to the shared driver `fn
functional_pad_nd_signed`, which:

- rejects negative pads under non-`Zeros` modes
  (`InvalidArgument`) and, for all-non-negative non-`Zeros` pads,
  delegates to the positive-only `functional_pad_{1,2,3}d` so
  reflect/replicate/circular keep their exact gather + autograd
  behaviour;
- for `Zeros` (constant) mode, runs `fn pad_nd_signed_constant` — a
  generic last-`N`-dim gather where output index `o` reads source
  `o - lo` (in bounds ⇒ data, otherwise ⇒ `value` fill), with
  per-axis size validated by `fn signed_axis_new_size` (over-crop ⇒
  `InvalidArgument`, net-zero ⇒ empty dim) and per-axis source
  resolution by `fn signed_axis_src`;
- attaches `struct PadNdSignedBackward` (named
  `"PadNdSignedBackward"`) when grad is enabled, scatter-adding the
  output grad onto the original-size input (cropped positions get 0).

These mirror `constant_pad_nd`'s negative-pad narrowing + `fill_` +
`copy_` (`aten/src/ATen/native/PadNd.cpp:29-108`). Crop is
constant-mode-only because `_pad_enum` routes only `mode="constant"`
through `constant_pad_nd`; the reflect/replicate/circular kernels do
not accept negative pads (`PadNd.cpp:221-242`). #1611.

### Layer family (REQ-3..REQ-7)

12 structs: `ConstantPad1d/2d/3d`, `ZeroPad1d/2d/3d`,
`ReflectionPad1d/2d/3d`, `ReplicationPad1d/2d/3d`,
`CircularPad1d/2d/3d`. Each carries `padding: (usize, usize[, …])`,
optional `value: T` (constant variants only), and `training: bool`.
Each has a `pub fn new(padding[, value]) -> Self` and a private
`fn pad(&self, input)` doing the actual work.

### Module impl (REQ-8)

`macro_rules! impl_padding_module` in `padding.rs` generates the
`impl<T: Float> Module<T> for $name<T>` block. `forward` calls
`self.pad(input)`. `parameters` / `parameters_mut` /
`named_parameters` return `vec![]` since these layers have no
trainable parameters. `train` / `eval` toggle `training`;
`is_training` returns it.

### Non-test production consumers

- `pub use padding::{PaddingMode, ConstantPad1d, ConstantPad2d,
  ConstantPad3d, ZeroPad1d, ZeroPad2d, ZeroPad3d, ReflectionPad1d,
  ReflectionPad2d, ReflectionPad3d, ReplicationPad1d,
  ReplicationPad2d, ReplicationPad3d, CircularPad1d, CircularPad2d,
  CircularPad3d, functional_pad_1d, functional_pad_2d,
  functional_pad_3d}` at `ferrotorch-nn/src/lib.rs`.
- The grow-only `functional_pad_{1,2,3}d` consume the crop-capable
  `functional_pad_{1,2,3}d_signed` in production (the `Zeros`/constant
  arm delegates to the signed path — the single source of truth for
  constant padding), so the signed entrypoints have a non-test
  production consumer (R-DEFER-1) within the crate.
- `Conv2d::forward` (and `Conv1d`/`Conv3d::forward`) invoke the pad
  helper when `padding_mode != Zeros`, pre-padding the input with the
  selected mode before the zero-padding im2col path runs (#1443); the
  `StringPadding::Same` branch also calls `functional_pad_*` with the
  (possibly `Zeros`) `padding_mode`, reaching the signed constant path.
- `ferrotorch-nn/src/functional.rs` re-exposes `functional_pad_*`
  as the public `nn::functional::pad` entrypoint.

## Parity contract

`parity_ops = ["nn.functional.pad", "nn.functional.constant_pad_nd",
"nn.functional.reflection_pad1d", "nn.functional.reflection_pad2d",
"nn.functional.replication_pad1d", "nn.functional.replication_pad2d"]`.

For each:
- **Empty pad** `(0, 0)` — identity (verified by tests).
- **Constant fill `value`** — upstream `F.pad(..., mode="constant",
  value=v)` fills the new positions with `v`; ferrotorch threads
  `value: T` through `pad_Nd_constant` to do the same (#1553).
- **Negative pad** — upstream accepts negative padding to crop a side
  (`aten/src/ATen/native/PadNd.cpp:29-108`). The `isize`-typed
  `functional_pad_1d/2d/3d_signed` entrypoints implement this for
  `mode="constant"`: a negative `lo`/`hi` narrows the dim, mixed signs
  per-dim are supported (e.g. `[-1, 2]` crops 1 from the start and adds
  2 fill at the end), and the backward (`PadNdSignedBackward`)
  scatter-adds the output grad back onto the original-size input so
  cropped-away positions receive zero gradient. Over-cropping (a side
  removing more than the running dim size, mirroring torch's
  `narrow(): length must be non-negative`) returns `InvalidArgument`; a
  net-zero crop is allowed (torch returns an empty dim). Closes #1611.
  The `nn.functional.pad` runner arm still skips negative samples until
  the runner is widened to feed `i64` pads to the signed entrypoints (a
  test-infrastructure follow-up under #1441, separate from this build).
- **Reflect with `pad >= input_dim`** — upstream raises
  `RuntimeError`; ferrotorch returns `InvalidArgument`.
- **Replicate with empty input dim** — both implementations need at
  least 1 element to replicate; both reject.
- **Circular with `pad > input_dim`** — both wrap around multiple
  times; semantics match.
- **NaN / Inf preservation** — both modes pass NaN/Inf through
  unchanged (constant `value` is literally placed).

Parity-sweep audit entries: `nn.functional.pad` is now `verified` in
`parity_audit.json` (#1441) at 376/408, 0 failed (the 32 negative-pad
skips map to #1611). The other 5 pad ops
(`constant_pad_nd`, `reflection_pad{1,2}d`, `replication_pad{1,2}d`)
still have no runner arm and stay un-recorded pending #1441.

## Verification

Tests in `mod tests` of `padding.rs` (40+ tests), covering:
- `test_constant_pad1d_zeros`, `test_constant_pad1d_with_value`.
- `test_functional_pad_1d_constant_uses_value`,
  `test_functional_pad_2d_constant_uses_value`,
  `test_functional_pad_3d_constant_uses_value` — assert a non-zero
  `value` reaches the padded cells of the `Zeros`/constant arm
  (#1553).
- `test_reflection_pad1d_basic`, `test_reflection_pad_rejects_oversized`.
- `test_replication_pad1d_basic`.
- `test_circular_pad1d_wraps`.
- 2D and 3D analogues for each mode.
- `test_functional_pad_2d_mode_dispatch` — verifies the functional
  helpers dispatch on `PaddingMode` correctly.
- Layer-style tests verifying `Module::forward` matches the
  underlying `pad_Nd_<mode>` output.

Parity-sweep smoke commands (all currently 0/N passed, N skipped):

```bash
for OP in nn.functional.pad nn.functional.constant_pad_nd \
         nn.functional.reflection_pad1d nn.functional.reflection_pad2d \
         nn.functional.replication_pad1d nn.functional.replication_pad2d; do
  ./target/release/parity-sweep sweep --op "$OP" --seeds 8 2>&1 | tail -1
done
```

Expected grep count after blocker #1441 closes: `>= 1` for each.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum PaddingMode` in `padding.rs` with 4 variants `Zeros`/`Reflect`/`Replicate`/`Circular`; non-test consumer: `ferrotorch-nn/src/conv.rs` imports `PaddingMode` as the field type the conv layers (currently inertly) carry — the wiring to use it is blocker #1443. |
| REQ-2 | SHIPPED | impl: grow-only entrypoints `pub fn functional_pad_1d` / `functional_pad_2d` / `functional_pad_3d` in `padding.rs` dispatch on `PaddingMode`; the `Zeros`/constant arm routes through the crop-capable `pub fn functional_pad_1d_signed` / `functional_pad_2d_signed` / `functional_pad_3d_signed` (`isize` pads) in `padding.rs`, which support negative (crop) pads + mixed signs for `mode="constant"` via `fn functional_pad_nd_signed` → `fn pad_nd_signed_constant` + `struct PadNdSignedBackward`, mirroring `constant_pad_nd` (`aten/src/ATen/native/PadNd.cpp:29-108`); the caller's `value: T` fill (#1553) is preserved. Non-test consumer: the `usize` `functional_pad_{1,2,3}d` consume the signed entrypoints in production (the `Zeros` arm), and `<Conv1d as Module>::forward` / `<Conv2d as Module>::forward` / `<Conv3d as Module>::forward` in `conv.rs` call `functional_pad_{1,2,3}d` for the non-`Zeros` `padding_mode` pre-pad and the `StringPadding::Same` (`Zeros`) pre-pad — so the signed path is reached in production through them. `ferrotorch-nn/src/functional.rs` also re-exposes `functional_pad_{1,2,3}d` as `nn::functional::pad`. |
| REQ-3 | SHIPPED | impl: `pub struct ConstantPad{1,2,3}d<T: Float>` in `padding.rs` mirroring `torch/nn/modules/padding.py` constant-pad family; non-test consumer: `pub use` in `lib.rs` exposes them to external crates. The vision-model code uses `ConstantPad2d` via the `lib.rs` re-export for padding non-square inputs. |
| REQ-4 | SHIPPED | impl: `pub struct ZeroPad{1,2,3}d<T: Float>` in `padding.rs`; non-test consumer: `pub use` in `lib.rs` exposes them. |
| REQ-5 | SHIPPED | impl: `pub struct ReflectionPad{1,2,3}d<T: Float>` in `padding.rs` with reflect-overflow check inside `pad_*d_reflect`; non-test consumer: `pub use` in `lib.rs` exposes them; reflection padding is the standard for unets and image-translation models. |
| REQ-6 | SHIPPED | impl: `pub struct ReplicationPad{1,2,3}d<T: Float>` in `padding.rs`; non-test consumer: `pub use` in `lib.rs`. |
| REQ-7 | SHIPPED | impl: `pub struct CircularPad{1,2,3}d<T: Float>` in `padding.rs`; non-test consumer: `pub use` in `lib.rs`. |
| REQ-8 | SHIPPED | impl: `macro_rules! impl_padding_module` in `padding.rs` generates the `Module<T>` impls for all 12 structs; non-test consumer: `ferrotorch_optim` walks `Module::parameters()` of containers that include padding layers (every padding layer returns the empty parameter list, which is the correct behavior). |
| REQ-9 | NOT-STARTED | The `nn.functional.pad` arm IS wired in `tools/parity-sweep/runner/src/main.rs` (#1441): decodes pad tuple/mode/value → `functional_pad_{1,2,3}d`; sweep `--seeds 8` 376/408, 0 failed. The 32 skips are all negative-pad (crop) — production gap #1611 (`functional_pad_{1,2,3}d` take `usize`). The OTHER 5 pad ops (`constant_pad_nd`, `reflection_pad{1,2}d`, `replication_pad{1,2}d`) still have NO runner arm, so REQ-9 stays NOT-STARTED until those land (#1441). Impl is end-to-end verified by 40+ lib tests. |
