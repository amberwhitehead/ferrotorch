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
  caller-supplied `value: T` through to `pad_Nd_constant` — the
  out-of-bounds fill is the caller's `value` (defaulting to `0` only
  when the caller passes `T::zero()`), matching
  `torch.nn.functional.pad(mode="constant", value=...)`.
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

The first constant helper `pad_1d_constant` is declared at
`padding.rs:59`; the 2-D and 3-D variants follow the same
`value`-seeding shape.

### Functional entrypoints (REQ-2)

The 1-D entrypoint `functional_pad_1d` is at `padding.rs:615`.

The 2-D entrypoint `functional_pad_2d` is at `padding.rs:779`.

The 3-D entrypoint `functional_pad_3d` is at `padding.rs:829`.

Each matches against `PaddingMode::{Zeros, Reflect, Replicate,
Circular}` and dispatches. The `Zeros` arm (torch
`mode="constant"`) threads the caller's `value: T` through to the
corresponding `pad_Nd_constant` helper — the out-of-bounds fill is
the caller-supplied `value`, not a hardcoded `T::zero()`, so an
arbitrary constant fill works (#1553, commit 276f740bd). The
`ZeroPad{1,2,3}d` layers obtain zero-fill by explicitly passing
`T::zero()`; the `Reflect`/`Replicate`/`Circular` arms ignore
`value` (the fill is gathered from existing data), and tests cover
all four arms — including
`test_functional_pad_{1,2,3}d_constant_uses_value`, which asserts a
non-zero `value` reaches the padded cells.

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
- `Conv2d::forward` invokes the 2-D pad helper at `conv.rs:577`
  when `padding_mode != Zeros`, pre-padding the input with the
  selected mode before the zero-padding im2col path runs (#1443).
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
- **Negative pad** — upstream accepts negative padding to crop;
  ferrotorch's `usize`-typed padding values reject this at the type
  level. This is a genuine production gap, filed as blocker #1611;
  the `nn.functional.pad` runner arm returns `Ok(None)` for
  negative-pad samples (the 32 skips at `--seeds 8`). Cropping must be
  done via slice ops until #1611 lands an `i64`-typed pad path.
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
| REQ-2 | SHIPPED | impl: the 1-D entrypoint `functional_pad_1d` at `padding.rs:615` dispatches on `PaddingMode`; the 2-D entrypoint `functional_pad_2d` at `padding.rs:779` and the 3-D entrypoint `functional_pad_3d` at `padding.rs:829` follow the same shape. The `Zeros`/constant arm threads the caller's `value: T` into `pad_Nd_constant` (the fill is the caller-supplied value, default `0` only when `T::zero()` is passed), matching `torch.nn.functional.pad(mode="constant", value=...)` — fixed in #1553 (commit 276f740bd), where the path previously hardcoded `T::zero()`. Non-test consumer: `Conv2d::forward` calls the 2-D pad helper at `conv.rs:577` for non-`Zeros` `padding_mode`, and `ferrotorch-nn/src/functional.rs` re-exposes it as `nn::functional::pad`. |
| REQ-3 | SHIPPED | impl: `pub struct ConstantPad{1,2,3}d<T: Float>` in `padding.rs` mirroring `torch/nn/modules/padding.py` constant-pad family; non-test consumer: `pub use` in `lib.rs` exposes them to external crates. The vision-model code uses `ConstantPad2d` via the `lib.rs` re-export for padding non-square inputs. |
| REQ-4 | SHIPPED | impl: `pub struct ZeroPad{1,2,3}d<T: Float>` in `padding.rs`; non-test consumer: `pub use` in `lib.rs` exposes them. |
| REQ-5 | SHIPPED | impl: `pub struct ReflectionPad{1,2,3}d<T: Float>` in `padding.rs` with reflect-overflow check inside `pad_*d_reflect`; non-test consumer: `pub use` in `lib.rs` exposes them; reflection padding is the standard for unets and image-translation models. |
| REQ-6 | SHIPPED | impl: `pub struct ReplicationPad{1,2,3}d<T: Float>` in `padding.rs`; non-test consumer: `pub use` in `lib.rs`. |
| REQ-7 | SHIPPED | impl: `pub struct CircularPad{1,2,3}d<T: Float>` in `padding.rs`; non-test consumer: `pub use` in `lib.rs`. |
| REQ-8 | SHIPPED | impl: `macro_rules! impl_padding_module` in `padding.rs` generates the `Module<T>` impls for all 12 structs; non-test consumer: `ferrotorch_optim` walks `Module::parameters()` of containers that include padding layers (every padding layer returns the empty parameter list, which is the correct behavior). |
| REQ-9 | NOT-STARTED | The `nn.functional.pad` arm IS wired in `tools/parity-sweep/runner/src/main.rs` (#1441): decodes pad tuple/mode/value → `functional_pad_{1,2,3}d`; sweep `--seeds 8` 376/408, 0 failed. The 32 skips are all negative-pad (crop) — production gap #1611 (`functional_pad_{1,2,3}d` take `usize`). The OTHER 5 pad ops (`constant_pad_nd`, `reflection_pad{1,2}d`, `replication_pad{1,2}d`) still have NO runner arm, so REQ-9 stays NOT-STARTED until those land (#1441). Impl is end-to-end verified by 40+ lib tests. |
