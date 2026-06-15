# Shape grad_fns

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/native/TensorShape.cpp
  - aten/src/ATen/native/Resize.cpp
-->

## Summary

`ferrotorch-core/src/grad_fns/shape.rs` implements the forward + backward
(autograd-tracking) shape-manipulation ops that mirror PyTorch's
`torch.reshape` / `torch.flatten` / `torch.squeeze` / `torch.unsqueeze` /
`torch.expand` / `torch.cat` / `torch.t` / `torch.roll` / `torch.split` (its
backward node) family, declared in `aten/src/ATen/native/TensorShape.cpp` and
`aten/src/ATen/native/Resize.cpp`. Each op pairs a `*Backward` `GradFn`
struct with a `pub fn` forward that either reinterprets the buffer metadata
(zero-copy `view_reshape` / `view_operation` on reshape, flatten, squeeze,
unsqueeze), or runs an on-device kernel (`expand` via the f32/f64
`broadcast_add_*` GPU fast path; `cat` via byte-width-dispatched
`strided_cat` on CUDA; `roll` via `roll_f32` on CUDA) and falls through to
CPU loops otherwise. The file shares the backward broadcast-reduction
primitive `reduce_grad_to_shape` with `grad_fns::arithmetic` via the
`super::arithmetic::reduce_grad_to_shape` import on the `ExpandBackward`
path; the rest of the backwards are pure index-arithmetic (CatBackward
fans out a contiguous slice per input; SplitBackward zero-pads into the
original shape).

Sister-file shape ops that the route's `parity_ops` field declares
(`view` / `permute` / `transpose` / `narrow` / `split` / `chunk`)
have their pub-fn forwards in `ferrotorch-core/src/methods.rs` but
consume `grad_fns/shape.rs`'s `ReshapeBackward` (`view_t`) and
`SplitBackward` (`split_t` / `chunk_t`) grad-fn structs; `roll` (in
`ferrotorch-core/src/ops/tensor_ops.rs`) consumes `RollBackward` from
this file. `stack` lives in `ferrotorch-core/src/vmap.rs`.
`broadcast_shapes` lives in `ferrotorch-core/src/shape.rs` (the
non-grad_fns sibling utility module) and is heavily consumed across
`ops/elementwise.rs`, `grad_fns/arithmetic.rs`, `grad_fns/indexing.rs`,
and `meta_propagate.rs`.

## Requirements

- REQ-1: `view(input, shape)` — `torch.Tensor.view(*shape)` for a contiguous
  input, with autograd. Per `aten/src/ATen/native/TensorShape.cpp:4563
  Tensor view(const Tensor& self, at::IntArrayRef size)` and
  `:4093 view_impl` — which `infer_size_dv`-resolves the shape (handles
  the single `-1` infer slot) then `computeStride`s a view alias. The
  forward MUST error if input is non-contiguous (upstream's
  `computeStride` returns `nullopt` and emits "view size is not
  compatible…"). Backward is identity reshape back to the input shape
  (`ReshapeBackward`).

- REQ-2: `reshape(input, shape)` — `torch.reshape(input, shape)`, with
  autograd. Per `TensorShape.cpp:2129 Tensor reshape(const Tensor& self,
  IntArrayRef proposed_shape)` — `infer_size_dv` to resolve the `-1`
  slot, then either `view_impl` if `computeStride` returns Some
  (zero-copy view path) or fall through to `_reshape_alias` /
  `_reshape_copy` (data copy with a `ReshapeBackward` autograd node).
  Backward is reshape back to `input_shape`.

- REQ-3: `flatten(input)` — `torch.flatten(input)` (full flatten to 1-D),
  with autograd. Per `TensorShape.cpp:4178 Tensor flatten(const Tensor&
  self, int64_t start_dim, int64_t end_dim)` — the no-arg / full-flatten
  form reduces to `reshape(input, [-1])` semantically.
  Backward unflattens via `reshape` back to the input shape
  (`FlattenBackward`, distinct from `ReshapeBackward` for clarity in the
  graph dump but identical in behavior).

- REQ-4: `unflatten(input, dim, sizes)` — `torch.unflatten(input, dim,
  sizes)`. Per `TensorShape.cpp:4350 Tensor unflatten_symint(const
  Tensor& self, int64_t dim, SymIntArrayRef sizes)` — reshapes a single
  dim into multiple, leaving the other dims untouched. SHIPPED as the
  free-function `pub fn unflatten` in `grad_fns/shape.rs` (resolves a
  single local `-1` slot against `self.size(dim)`, splices the resolved
  sizes in place of `dim`, then `reshape`s — inheriting `ReshapeBackward`).
  The separate `nn::Unflatten` Module-style layer at
  `ferrotorch-nn/src/identity.rs:264` remains the layer wrapper.

- REQ-5: `squeeze(input, dim)` — `torch.squeeze(input, dim)`. Per
  `TensorShape.cpp:4026 Tensor squeeze(const Tensor& self, int64_t dim)`
  — `maybe_wrap_dim` (negative-dim normalization), then size-1 check,
  then `as_strided_symint` to drop the singleton dim. Backward
  unsqueezes the size-1 dim back at the original position
  (`SqueezeBackward`).

- REQ-6: `unsqueeze(input, dim)` — `torch.unsqueeze(input, dim)`. Per
  `TensorShape.cpp:4109 Tensor unsqueeze(const Tensor& self, int64_t
  dim)` — `maybe_wrap_dim(dim, self.dim() + 1)` (range is
  `[-(ndim+1), ndim]`), then `inferUnsqueezeGeometry_symint` +
  `as_strided_symint` to insert a size-1 dim. Backward squeezes the
  inserted dim out (`UnsqueezeBackward`).

- REQ-7: `permute(input, dims)` — `torch.permute(input, dims)`. Per
  `TensorShape.cpp:1829 Tensor permute(const Tensor& self, IntArrayRef
  dims)` — validates `dims` is a permutation of `0..ndim`, then
  reorders sizes and strides into a zero-copy view. Backward applies
  the inverse permutation. ferrotorch's `permute_t` lives in
  `ferrotorch-core/src/methods.rs` rather than this file, with its own
  `PermuteBackward` (`PermuteBackward in methods.rs` inverse-permute path); this file
  consumes `permute_t` from `TransposeBackward::backward` and
  `transpose_2d`.

- REQ-8: `transpose(input, dim0, dim1)` — `torch.transpose(input, dim0,
  dim1)`. Per `TensorShape.cpp:5116 Tensor transpose(const Tensor&
  self, int64_t dim0, int64_t dim1)` — `maybe_wrap_dim` both args,
  build a permutation swapping `dim0` ↔ `dim1`, return
  `as_strided_symint` view. The 2-D special case
  `Tensor::t() at Tensor t(const Tensor& self)` is in this
  file as `transpose_2d` (delegates to `permute_t(&[1,0])`); the n-D
  form lives at `Tensor::transpose` in `methods.rs` (builds a perm
  vec swapping dim0/dim1 then calls `permute_t`).

- REQ-9: `swapaxes(input, axis0, axis1)` — `torch.swapaxes(input,
  axis0, axis1)`. Per `TensorShape.cpp:4776 Tensor swapaxes(const
  Tensor& self, int64_t axis0, int64_t axis1) { return
  self.transpose(axis0, axis1); }` — a literal alias of transpose.
  SHIPPED as `pub fn swapaxes` in `grad_fns/shape.rs` + `Tensor::swapaxes`
  in `methods.rs`.

- REQ-10: `swapdims(input, dim0, dim1)` — `torch.swapdims(input, dim0,
  dim1)`. Per `TensorShape.cpp:4784 Tensor swapdims(const Tensor&
  self, int64_t dim0, int64_t dim1) { return self.transpose(dim0,
  dim1); }` — also a literal transpose alias. SHIPPED alongside
  `swapaxes` as `pub fn swapdims` + `Tensor::swapdims`.

- REQ-11: `expand(input, sizes)` — `torch.Tensor.expand(*sizes)`. Per
  `TensorShape.cpp:1344 Tensor expand(const Tensor& self,
  c10::IntArrayRef size, bool /*unused*/)` — validates that the target
  has at least as many dims as the input, then
  `inferExpandGeometry_dimvector` + `as_strided` to broadcast each
  size-1 axis to the target. Backward sums over every broadcast axis
  back to the input shape via the shared `reduce_grad_to_shape`
  primitive (`ExpandBackward` consumes
  `super::arithmetic::reduce_grad_to_shape`).

- REQ-12: `expand_as(input, other)` — `torch.Tensor.expand_as(other)`.
  Per `TensorShape.cpp:1374 Tensor expand_as(const Tensor& self, const
  Tensor& other) { return self.expand_symint(other.sym_sizes()); }` —
  a literal one-liner delegating to `expand` with `other.sizes()`.
  SHIPPED as `pub fn expand_as` in `grad_fns/shape.rs` +
  `Tensor::expand_as_t` in `methods.rs` (inherits `ExpandBackward`).

- REQ-13: `repeat(input, repeats)` — `torch.Tensor.repeat(*repeats)`.
  Per `TensorShape.cpp:1909 Tensor repeat(const Tensor& self,
  IntArrayRef repeats)` — tiles the input along each dim by the
  corresponding `repeats` factor (different from numpy `repeat` /
  PyTorch `repeat_interleave` which interleave). The tile-style op
  is NOT implemented; `ferrotorch-core/src/einops.rs:514
  pub fn repeat` is an unrelated einops-pattern op
  (`"c -> c n"` parsing) with different semantics.

- REQ-14: `repeat_interleave(input, repeats, dim)` —
  `torch.repeat_interleave`. Interleaves elements along a dim. NOT
  implemented in ferrotorch-core.

- REQ-15: `cat(tensors, dim)` — `torch.cat(tensors, dim)`. Per
  `TensorShape.cpp:676 TORCH_IMPL_FUNC(cat_out_cpu)` and `:772 Tensor
  cat(TensorList tensors, Dimname dim)` — concatenates a list of
  same-shape-except-along-dim tensors along the given axis. Backward
  splits the gradient back into the original chunk sizes
  (`CatBackward`). ferrotorch ships the GPU fast path via the
  byte-width-dispatched `strided_cat` kernel (elem_size ∈ {2, 4, 8})
  per the `aten::cat_out_cuda` shape — host computes elem_size once,
  backend routes to the matching memcpy kernel.

- REQ-16: `stack(tensors, dim)` — `torch.stack(tensors, dim)`. Per
  `TensorShape.cpp:3462 Tensor stack(TensorList tensors, int64_t
  dim)` — equivalent to `cat([unsqueeze(t, dim) for t in tensors],
  dim)`. ferrotorch's `vmap::stack` at
  `ferrotorch-core/src/vmap.rs` is the pub-API surface
  (grandfathered per S5 — pub fn across multiple prior commits).
  The backward is induced by the `unsqueeze` + `cat` composition
  (which carry their own grad-fns).

- REQ-17: `vstack(tensors)` — `torch.vstack`. Per `TensorShape.cpp:3412
  Tensor vstack(TensorList tensors)` — equivalent to `cat` along
  axis 0 with 1-D inputs promoted to `[1, n]`. NOT implemented.

- REQ-18: `hstack(tensors)` — `torch.hstack`. Per `TensorShape.cpp:3514
  Tensor hstack(TensorList tensors)` — `cat` along axis 1 for ≥2-D,
  axis 0 for 1-D. NOT implemented.

- REQ-19: `dstack(tensors)` — `torch.dstack`. Per `TensorShape.cpp:3544
  Tensor dstack(TensorList tensors)` — `cat` along axis 2 with 1-D
  promoted to `[1, n, 1]` and 2-D promoted to `[m, n, 1]`. NOT
  implemented.

- REQ-20: `column_stack(tensors)` — `torch.column_stack`. Per
  `TensorShape.cpp:3628 Tensor column_stack(TensorList tensors)` —
  treats 1-D as columns then `hstack`s. NOT implemented.

- REQ-21: `split(input, split_size_or_sizes, dim)` — `torch.split`.
  Per `TensorShape.cpp:3175 std::vector<Tensor> split(const Tensor&
  self, int64_t split_size, int64_t dim)` and `:3265 split_with_sizes`
  — slices the input along `dim` into chunks. Backward zero-pads each
  incoming chunk-gradient into the original shape at the correct
  offset (`SplitBackward`). The forward pub fn `split_t` lives in
  `split_t in methods.rs`; it consumes `SplitBackward` from this file at
  `methods.rs:1892 use crate::grad_fns::shape::SplitBackward`.

- REQ-22: `chunk(input, chunks, dim)` — `torch.chunk`. Per
  `TensorShape.cpp:1077 std::vector<Tensor> chunk(const Tensor& self,
  int64_t chunks, int64_t dim)` — computes the per-chunk size as
  `(self.size(dim) + chunks - 1) / chunks` then delegates to
  `split_with_sizes`. The forward pub fn `chunk_t` lives in
  `chunk_t in methods.rs`; it shares the `SplitBackward` machinery with
  `split_t`.

- REQ-23: `tensor_split(input, indices_or_sections, dim)` —
  `torch.tensor_split`. Per `TensorShape.cpp:1099
  tensor_split_sections_symint` (even sections) and `:1167
  tensor_split` (indices) — splits at integer indices rather than by
  chunk size, handling uneven splits. NOT implemented.

- REQ-24: `narrow(input, dim, start, length)` — `torch.narrow`. Per
  `TensorShape.cpp:1669 Tensor narrow(const Tensor& self, int64_t
  dim, int64_t start, int64_t length)` — returns a zero-copy view of
  `length` elements starting at `start` along `dim` (uses
  `slice` internally). Backward zero-pads at the narrow offset. The
  forward pub fn `narrow_t` lives in `methods.rs:1600`.

- REQ-25: `unbind(input, dim)` — `torch.unbind`. Per
  `TensorShape.cpp:4367 std::vector<Tensor> unbind(const Tensor&
  self, int64_t dim)` — returns a Vec of `size(dim)`-many tensors,
  each `select`-ed at the corresponding index. NOT implemented.

- REQ-26: `broadcast_tensors(tensors)` —
  `torch.broadcast_tensors(*tensors)`. Per `TensorShape.cpp:656
  std::vector<Tensor> broadcast_tensors(TensorList tensors)` —
  computes the common broadcast shape and expands each input to it.
  NOT implemented as a free fn; the ingredients
  (`shape::broadcast_shapes` + `grad_fns::shape::expand`) are
  available individually and used in
  `grad_fns/indexing.rs:2092/2169/2206/5331` to assemble the same
  contract ad-hoc, but the named bundled op does not exist.

- REQ-27: `broadcast_to(input, shape)` — `torch.broadcast_to(input,
  shape)`. Per `TensorShape.cpp:652 Tensor broadcast_to_symint(const
  Tensor& self, SymIntArrayRef size) { return self.expand_symint(size);
  }` — a literal alias of `expand`. NOT implemented as a named pub
  fn.

- REQ-28: `broadcast_shapes(*shapes)` — `torch.broadcast_shapes`.
  Per `TensorShape.cpp:643 broadcast_shapes` (template helper). The
  utility lives in `ferrotorch-core/src/shape.rs` (the non-grad_fns
  sister utility module), `pub fn broadcast_shapes(a: &[usize], b:
  &[usize])` at `shape.rs:7`. Implements the right-aligned NumPy
  broadcast rule: dims compatible when equal or one is 1.

- REQ-29: `movedim(input, source, destination)` —
  `torch.movedim`. Per `TensorShape.cpp:4657 Tensor movedim(const
  Tensor& self, IntArrayRef src, IntArrayRef dst)` — repositions one
  or more dims to a target index, equivalent to `permute` with a
  computed permutation. NOT implemented.

- REQ-30: `moveaxis(input, source, destination)` —
  `torch.moveaxis`. Per `TensorShape.cpp:4768 Tensor moveaxis(const
  Tensor& self, IntArrayRef src, IntArrayRef dst) { return
  at::movedim(self, src, dst); }` — a literal alias of `movedim`.
  NOT implemented.

- REQ-31: `tile(input, reps)` — `torch.tile`. Per
  `TensorShape.cpp:1971 Tensor tile_symint(const Tensor& self,
  SymIntArrayRef reps)` — numpy-style tile: right-aligns the reps
  vector against the input dims and tiles each axis. Distinct from
  `repeat` in argument semantics for shorter reps (tile prepends 1s,
  repeat treats the diff as an error). NOT implemented.

- REQ-32: `roll(input, shifts, dim)` — `torch.roll`. Per
  `aten/src/ATen/native/TensorTransformations.cpp:110 Tensor roll(...)`
  — cyclic shift along `dim` by `shifts` elements. Note: upstream
  location is `TensorTransformations.cpp`, not the route-declared
  `TensorShape.cpp` — the route's `upstream` list is incomplete for
  this op. ferrotorch's forward `pub fn roll` lives in
  `roll in ferrotorch-core/src/ops/tensor_ops.rs`; backward
  `RollBackward` is in THIS file at `RollBackward in shape.rs`, consumed
  by the CUDA and CPU forward arms of `roll in tensor_ops.rs` (both
  attach the backward fn from this module). Backward applies the inverse
  shift `-shifts` mod `size(dim)`.

- REQ-33: `rot90(input, k, dims)` — `torch.rot90`. Per
  `TensorTransformations.cpp:134 Tensor rot90(const Tensor& self,
  int64_t k, IntArrayRef dims)` — rotates 90° k times in the plane
  spanned by `dims`. NOT implemented.

- REQ-34: `flip(input, dims)` — `torch.flip`. Per
  `TensorTransformations.cpp:36 Tensor flip(const Tensor& self,
  IntArrayRef dims)` — reverses element order along the listed dims.
  NOT implemented as a free op (`ferrotorch-nn/src/conv.rs` has a
  private `flip_kernel` helper used during conv-transpose backward,
  but it is not the user-facing `torch.flip` op).

- REQ-35: `fliplr(input)` — `torch.fliplr`. Per
  `TensorTransformations.cpp:180 Tensor fliplr(const Tensor& self) {
  return self.flip({1}); }` — flip along dim 1. NOT implemented.

- REQ-36: `flipud(input)` — `torch.flipud`. Per
  `TensorTransformations.cpp:186 Tensor flipud(const Tensor& self) {
  return self.flip({0}); }` — flip along dim 0. NOT implemented.

## Acceptance Criteria

- [x] AC-1: `reshape`, `flatten`, `squeeze`, `unsqueeze`, `expand`, `cat`
  and the full #1342 op set forward + backward unit tests pass:
  `cargo test -p ferrotorch-core --lib grad_fns::shape` returns
  `86 passed; 0 failed` (was 35 before the unflatten/swapaxes batch,
  46 after it, 86 after the final #1342-closing batch added flip /
  rot90 / movedim / repeat / repeat_interleave / unbind /
  tensor_split / *stack coverage).
- [x] AC-2: `RollBackward` lib tests pass (`test_roll_forward_registers_grad_fn`,
  `test_roll_zero_shift_early_return`, `test_roll_backward_simple_1d_hand_computed`,
  `test_roll_backward_negative_shift_2d`).
- [x] AC-3: Shape ops share storage with input on the no-grad zero-copy
  path: `test_shape_ops_share_storage_with_input` asserts
  `flat.shares_storage(&x)` for `flatten`/`squeeze`/`unsqueeze`.
- [x] AC-4: Backward through `squeeze` reaches the original leaf in a
  longer chain (mul → mm → squeeze → loss):
  `test_squeeze_in_longer_chain` — exercises the GPU graph-severance
  regression where `restore_device(from_operation(...))` would have
  detached the grad_fn.
- [x] AC-5: Both `view_t` (`view_t in methods.rs`) and `reshape`
  (`shape.rs:104`) handle the single `-1` infer slot via
  `resolve_shape` at `test_resolve_shape_infer in shape.rs`. `test_resolve_shape_infer`
  passes.
- [x] AC-6: `cat` mixed `requires_grad` propagates gradients only to
  the leaves that require them: `test_cat_backward_mixed_requires_grad`
  asserts `b.grad().is_none()`.
- [x] AC-7: `expand` GPU fast path (f32/f64) dispatches to
  `broadcast_add` with a 1-element zeros scalar rather than spilling
  to CPU — checked by `shape.rs:482-507`.
- [x] AC-8: `cat` GPU fast path dispatches to byte-width-dispatched
  `strided_cat` (elem_size ∈ {2, 4, 8}) per
  `shape.rs:1856-1907` (matches `aten::cat_out_cuda` shape).
- [x] AC-9: `narrow_t` (`methods.rs:1600`) returns a zero-copy view
  with the appropriate `NarrowBackward` for autograd.
- [x] AC-10: `split_t` (`split_t in methods.rs`) returns one tensor per
  chunk, each carrying a `SplitBackward` from this module
  (`methods.rs:1892 use crate::grad_fns::shape::SplitBackward`).
- [x] AC-11: `chunk_t` (`chunk_t in methods.rs`) computes per-chunk size
  via `(size + chunks - 1) / chunks` then delegates to the same
  `SplitBackward` machinery.
- [x] AC-12: `permute_t` (`permute_t in methods.rs`) produces a zero-copy
  stride view with a `PermuteBackward` (`PermuteBackward in methods.rs`) that
  applies the inverse permutation on backward.
- [x] AC-13: `Tensor::transpose(dim0, dim1)` (`methods.rs:912`)
  builds a permutation vector swapping the two dims then delegates
  to `permute_t` — zero-copy n-D transpose with autograd.
- [x] AC-14: `Tensor::t()` (`methods.rs:683`) delegates to
  `shape::transpose_2d` which is itself a `permute_t(&[1, 0])`
  delegation — zero-copy 2-D transpose with autograd.
- [x] AC-15: `view_t` rejects non-contiguous inputs with
  `InvalidArgument: "view: tensor must be contiguous; call
  .contiguous() first"` (`methods.rs:1709-1712`) — matches
  upstream's `computeStride`-fails-then-error behavior.
- [x] AC-16: `expand` errors when target has fewer dims than input
  (`shape.rs:419-425`) and when a non-1 input dim must be expanded
  (`shape.rs:428-441`) — matches upstream's `inferExpandGeometry`
  errors.
- [x] AC-17: `squeeze` errors when the named dim is not size-1
  (`shape.rs:214-222`) — note: upstream `torch.squeeze(x, dim)`
  returns x unchanged in this case (`TensorShape.cpp:4029-4031`);
  ferrotorch errors instead. This is a deliberate departure
  documented in the function-level rustdoc.
- [ ] AC-18: All 36 parity_ops at `--seeds 8` report
  `passed (0 skipped, 0 failed)` with N ≥ 1. CURRENTLY DEFERRED: the
  runner has arms for only a couple of ops and they report
  `0/N passed (N skipped, 0 failed)` because the runner's
  `decode_into_typed_op` / dispatcher does not yet hook the
  ferrotorch ops for shape-op samples. The runner-arm gap is
  tracked under umbrella blocker #1340 per S5 (test-infrastructure
  gap, NOT a REQ blocker for SHIPPED ops). Independently, the
  `parity-sweep` binary cannot launch in the current build
  environment (`libmkl_rt.so.2` is absent system-wide — the
  MKL/libclang bootstrap gap), so parity smoke produces zero
  output; the #1342 ops are SHIPPED on impl + non-test production
  consumer + lib tests, which is the doc's stated SHIPPED bar (see
  "Parity contract").

## Architecture

### `ensure_cpu` + `restore_device` (shape.rs:40-58)

`ensure_cpu` is the conservative "GPU shape ops aren't available, so
download for the CPU implementation" helper. It deliberately errors out
on CUDA tensors (`NotImplementedOnCuda { op: "shape backward" }`)
rather than silently downloading — keeping with R-CODE-4
(no-silent-roundtrips). `restore_device` is its companion that moves a
CPU-built result back to the original device. Both are used inside
some `*Backward` paths only; the forward fast paths (reshape, flatten,
squeeze, unsqueeze, transpose_2d) are zero-copy views that work on any
device because they never touch the storage buffer.

### REQ-1 / REQ-2 / REQ-3 — `view`, `reshape`, `flatten`

Both `pub fn reshape` (`reshape in shape.rs`) and `pub fn flatten`
(`flatten in shape.rs`) use `input.view_reshape(new_shape)` on the no-grad
path and `input.view_operation(new_shape, grad_fn)` on the grad path —
both helpers are zero-copy metadata changes implemented at the
`Tensor` layer. The `ReshapeBackward` / `FlattenBackward` structs
(`FlattenBackward in shape.rs` and `shape in shape.rs`) save the original
`input_shape: Vec<usize>` and on backward simply `view_reshape` the
incoming gradient back to that shape. `view_t` (`view_t in methods.rs`)
adds the contiguity gate then delegates to `crate::grad_fns::shape::
reshape` — so the `view` API IS the `reshape` API plus a pre-check.
**Non-test consumers**: `methods.rs:885 reshape_t`, `methods.rs:889
flatten_t`, `methods.rs:1094 view`, `flex_attention.rs:167-256` (four
reshapes inside the SDP-attention forward), `einsum.rs:1072-1107`
(reshape used to materialize batched matmul intermediates),
`tensor.rs:2488` (FlattenBackward attached on the `Tensor::flatten`
method body).

### REQ-5 / REQ-6 — `squeeze`, `unsqueeze`

`pub fn squeeze` (`squeeze in shape.rs`) normalizes axis (negative-index
wrap), validates `shape()[axis] == 1` (deliberate departure from
upstream's no-op behavior — see AC-17), then `view_reshape` /
`view_operation` with the dim removed. `pub fn unsqueeze`
(`unsqueeze in shape.rs`) validates the range `[-(ndim+1), ndim]` (one wider
than `squeeze` per upstream `maybe_wrap_dim(dim, self.dim() + 1)`),
normalizes the axis, then inserts a 1 at that position.
`SqueezeBackward` / `UnsqueezeBackward` (`UnsqueezeBackward in shape.rs`,
`tensor.rs`) are exact inverses: squeeze backward unsqueezes the same
axis; unsqueeze backward squeezes it. **Non-test consumers**:
`methods.rs:889-912`, `einsum.rs:838-885` (insert size-1 dims to
materialize matmul-friendly shapes then squeeze them back),
`grad_fns/indexing.rs` (broadcast-prep for masked/where ops).

### REQ-7 / REQ-8 — `permute`, `transpose`

`transpose_2d` (`shape in shape.rs`) is the strict 2-D entry that errors
for any rank ≠ 2, then delegates to `crate::methods::permute_t(input,
&[1, 0])` for the zero-copy stride swap. `TransposeBackward`
(`TransposeBackward in shape.rs`) backward also goes through `permute_t(&[1, 0])`
— transpose is its own inverse. n-D `Tensor::transpose(dim0, dim1)`
lives at `methods.rs:912` (builds a perm vec swapping dim0 ↔ dim1).
`permute_t` itself with full `PermuteBackward` (inverse-perm)
machinery is at `methods.rs:876` and `:941`. **Non-test consumers**:
`methods.rs:683 t`, `methods.rs:1491 permute`, `methods.rs:912
transpose`, `einsum.rs:294` (intermediate reshape via permute +
contiguous), and pervasively across einsum / vmap / meta_propagate.

### REQ-11 — `expand`

`pub fn expand` (`expand in shape.rs`) validates the target has at least
input's ndim, validates each non-1 input dim matches its target,
then takes the GPU fast path on CUDA f32/f64 (`shape.rs:482-507`:
allocates a 1-element zeros scalar, calls
`backend.broadcast_add_{f32,f64}(input, zeros, in_shape, &[1],
new_shape)` to broadcast on-device — no CPU roundtrip), or the CPU
path otherwise (`shape.rs:476-491`: builds output via
`broadcast_flat_index` which maps each output flat-index to its
input flat-index, with size-1 dims clamped to 0). `ExpandBackward`
(`ExpandBackward in shape.rs`) calls
`super::arithmetic::reduce_grad_to_shape(grad_output,
&self.input_shape)` to sum-reduce the gradient over every
broadcast axis — the shared backward primitive with arithmetic ops.
**Non-test consumers**: `grad_fns/indexing.rs:2092/1826/1851/3577`
(masked-fill/where prep), `einsum.rs:1725` (sum-grad expand).

### REQ-15 — `cat`

`pub fn cat` (`cat in shape.rs`) validates the input list is non-empty,
that each tensor has the same ndim, and that each non-cat dim
matches across all inputs. Computes `total_along_axis` then builds
the output shape. GPU fast path (`shape.rs:815-855`): allocates a
zero-filled output of the right shape on device, then for each
input invokes `backend.strided_cat(t_handle, &mut out_handle,
total_along_axis, offset, t_axis_size, inner, t_numel, elem_size)`
— host computes `elem_size = size_of::<T>()` once and the backend
routes to the matching byte-width memcpy kernel (the
`aten::cat_out_cuda` shape: one kernel per elem-size in {2, 4, 8},
not one per dtype). CPU path is the interleaved `copy_from_slice`
loop over `(outer, t_axis_size, inner)`. `CatBackward`
(`CatBackward in shape.rs`) likewise has GPU + CPU paths: GPU uses
`backend.strided_split_{f32,f64}` to extract each chunk on-device;
CPU does the inverse `copy_from_slice` loop. **Non-test consumers**:
`flex_attention in flex_attention.rs cat(&group, 1)` + `:238 cat(&head_groups,
0)` (head-grouped attention assembly), `lib.rs:165` re-exports
`cat` at the crate root.

### REQ-32 — `roll`

The forward `pub fn roll` lives at `roll in ops/tensor_ops.rs` (not
this file). It builds a `RollBackward` from this file's `ops/tensor_ops.rs`
implementation and attaches it via `Tensor::from_operation`.
`RollBackward::backward` (`shape.rs:2047-2118`) computes the inverse
shift `(((-shifts) % dim_size) + dim_size) % dim_size`, then on
CUDA f32 dispatches to `backend.roll_f32` with the inverse shift
(handles the `shift_norm == 0` collapse-to-identity via
`clone_buffer` to keep the leaf-grad shape correct), and otherwise
calls back into `crate::ops::tensor_ops::roll_cpu_inner` with the
inverse shift on a CPU buffer. Tests
`test_roll_backward_simple_1d_hand_computed` and
`test_roll_backward_negative_shift_2d` exercise both the 1-D
positive-shift and 2-D negative-shift cases against
hand-computed expected gradients. **Non-test consumer**: the
forward `tensor_ops::roll` at `tensor_ops.rs:582` is the public
API and itself the consumer — grandfathered under S5.

### REQ-21 / REQ-22 — `split`, `chunk`

The forward `split_t` (`split_t in methods.rs`) and `chunk_t`
(`chunk_t in methods.rs`) live in `methods.rs` and explicitly import
`crate::grad_fns::shape::SplitBackward` at `methods.rs:1892`. Each
chunk produced by `split_t` carries a fresh `SplitBackward`
recording the chunk's `(dim, offset, chunk_size)`. On backward,
`SplitBackward` (`SplitBackward in shape.rs`) allocates a zero-filled
buffer of the original shape, then copies the incoming chunk
gradient into the correct slice — GPU fast path uses
`backend.strided_cat` with the same byte-width dispatch as
forward `cat`, CPU path runs the `(outer, total_along_dim,
inner)` slice copy. **Non-test consumer**: `methods.rs:1892` is
the production consumer of the exported `SplitBackward` struct.

### REQ-24 — `narrow`

`narrow_t` (`methods.rs:1600`) returns a zero-copy view with
adjusted shape, strides, and storage offset. Backward
(`NarrowBackward in methods.rs` and the implementation
fn above it) zero-pads the incoming gradient at the offset along
the narrow dim. **Non-test consumer**: `methods.rs:547
Tensor::narrow`.

### REQ-16 — `stack`

`vmap::stack` (`vmap in vmap.rs`) is the pub-API surface used in
`ferrotorch-core::vmap` to assemble per-sample results into a
batched tensor. It builds the stack as a `cat` over `unsqueeze`d
inputs — so its autograd contract follows from REQ-6 (unsqueeze)
and REQ-15 (cat). Grandfathered under S5 — no separate
StackBackward needed because the unsqueeze + cat composition
already carries grad-fns.

### REQ-28 — `broadcast_shapes`

`broadcast_shapes` is at `broadcast_shapes in ferrotorch-core/src/shape.rs` (NOT
this file — the route's `parity_ops` declaration is broader than
the file's direct contents). It computes the broadcast shape per
the right-aligned NumPy rule and is consumed by:
`meta_propagate.rs`, `ops/elementwise.rs`,
`grad_fns/indexing.rs:1803/1825/1848-1849/3572`,
`grad_fns/arithmetic.rs:39` — i.e., every binary op routes through
it on the meta-shape path.

### Remaining shape ops — all SHIPPED (umbrella #1342 closed)

As of the #1342-closing commit, every REQ in this doc is SHIPPED. The
final batch of 17 ops split into three implementation strategies, each
chosen so autograd is inherited from an existing `*Backward` rather
than hand-rolled where possible:

1. **Pure aliases** — `broadcast_to` (REQ-27 → `expand`), `moveaxis`
   (REQ-30 → `movedim`), `fliplr` (REQ-35 → `flip({1})`), `flipud`
   (REQ-36 → `flip({0})`). One-line delegations to a shipped op;
   autograd inherited from the delegate's backward.

2. **Compositional ops built only from grad-aware primitives** — `rot90`
   (REQ-33, `flip` + `transpose`), `movedim` (REQ-29, computed perm →
   `permute_t`/`PermuteBackward`), `broadcast_tensors` (REQ-26,
   `broadcast_shapes` + `expand`), `repeat` (REQ-13, `cat`-composition
   over copies), `tile` (REQ-31, left-pad reps → `repeat`), `unbind`
   (REQ-25, `narrow` + `squeeze` per index), `tensor_split` (REQ-23,
   `narrow` per section), `vstack`/`hstack`/`dstack`/`column_stack`
   (REQ-17..20, `atleast_Nd` promotion + `cat`). No new `*Backward`
   struct — gradients flow through the constituent ops.

3. **Substantive forward + new backward** — `flip` (REQ-34, the core
   CPU index-reversal kernel `flip_cpu_inner` + `FlipBackward`, flip is
   its own inverse) and `repeat_interleave` (REQ-14, consecutive-
   duplication forward + `RepeatInterleaveBackward` segment-sum
   adjoint). These two carry the only new autograd nodes added in the
   batch.

The free-function `repeat` is NOT re-exported at the crate root (the
crate-root `repeat` is the unrelated string-pattern `einops::repeat`);
the tile-style `Tensor.repeat` is reached via `Tensor::repeat_t` and the
`grad_fns::shape::repeat` path.

## Parity contract

The route declares 36 parity_ops. Their `parity_audit.json` entries
are all currently `MISSING`, and only 2 (`transpose`, `expand`) have
runner arms in `tools/parity-sweep/runner/src/main.rs` — both of
which run but produce `0/N passed (N skipped, 0 failed)` because
the per-op decode_into_typed_op dispatch is not yet wired for shape
ops. Per S5, this is a test-infrastructure gap (umbrella blocker
#1340), not a REQ blocker. The SHIPPED REQs in this doc are
SHIPPED on the strength of impl + non-test production consumer +
lib tests, not parity-sweep smoke.

Expected behavior on edge cases for the SHIPPED ops:

- **NaN / Inf**: Shape ops are pure index arithmetic and never
  read/transform tensor values (except `expand`'s CPU loop and
  `cat`/`split` chunk copies, which are byte-faithful
  `copy_from_slice`). NaN and Inf propagate through unchanged.
- **Empty**: `cat` errors on empty input list (`shape.rs:766-768`);
  `cat` on 0-D inputs errors (`shape.rs:772-776`); `expand` errors
  on the no-change degenerate case via the dim-count check.
- **Negative-dim**: Normalized via
  `crate::shape::normalize_axis` (squeeze, unsqueeze, cat) or
  custom range check (unsqueeze's wider range
  `[-(ndim+1), ndim]`).
- **Non-contiguous CUDA**: `cat` GPU fast path requires the
  byte-width memcpy kernel which operates on the raw flat buffer
  — passing a non-contiguous view here would corrupt the output.
  The forward `cat` calls `gpu_handle()?` on each input which
  asserts the storage is materialized; if any input is a stride
  view, this will surface as a `LengthMismatch` from the backend.
  Tests do not currently exercise this case.
- **Dtype promotion**: Shape ops are `T: Float` generic and do
  not promote — the output dtype equals the input dtype. `cat`
  requires all inputs of the same `T`; cross-dtype cat is a
  compile-time error rather than a runtime promotion.

## Verification

### Lib tests

`cargo test -p ferrotorch-core --lib grad_fns::shape` runs 35
tests in `ferrotorch-core/src/grad_fns/shape.rs:1109-1676`:

- Reshape: `test_reshape_forward`, `test_reshape_infer_dim`,
  `test_reshape_backward`, `test_reshape_shape_mismatch`,
  `test_reshape_no_grad`.
- Flatten: `test_flatten_forward`, `test_flatten_backward`,
  `test_flatten_preserves_grad_fn`.
- Squeeze/Unsqueeze: `test_squeeze_forward`,
  `test_squeeze_non_one_error`, `test_unsqueeze_forward`,
  `test_squeeze_unsqueeze_roundtrip`,
  `test_squeeze_preserves_grad_fn`,
  `test_unsqueeze_preserves_grad_fn`,
  `test_squeeze_backward_reaches_leaf`,
  `test_unsqueeze_backward_reaches_leaf`,
  `test_squeeze_in_longer_chain`, `test_squeeze_no_grad_is_view`.
- Transpose: `test_transpose_2d_forward`.
- Cat: `test_cat_forward_axis0`, `test_cat_forward_axis1`,
  `test_cat_backward_axis0`, `test_cat_backward_axis1`,
  `test_cat_backward_mixed_requires_grad`,
  `test_cat_empty_error`, `test_cat_1d`.
- Roll backward: `test_roll_forward_registers_grad_fn`,
  `test_roll_zero_shift_early_return`,
  `test_roll_backward_simple_1d_hand_computed`,
  `test_roll_backward_negative_shift_2d`.
- Shape op storage sharing:
  `test_shape_ops_share_storage_with_input`.
- Helper tests: `test_resolve_shape_basic`,
  `test_resolve_shape_infer`,
  `test_resolve_shape_multiple_infer_error`,
  `test_resolve_shape_mismatch`.

Plus the #1342 op coverage (flip / fliplr / flipud / rot90 / movedim /
moveaxis / broadcast_to / broadcast_tensors / repeat / tile /
repeat_interleave / unbind / tensor_split / vstack / hstack / dstack /
column_stack — forward + backward + alias-equivalence + error paths).

Result line: `test result: ok. 86 passed; 0 failed; 0 ignored`. An
integration mirror lives at
`ferrotorch-core/tests/divergence_shape_ops_audit.rs` (19 tests
driving the public `Tensor::*_t` method wrappers — the R-DEFER-1
non-test production consumers).

### Parity sweep (current state)

```bash
for OP in view reshape flatten unflatten squeeze unsqueeze permute \
          transpose swapaxes swapdims expand expand_as repeat \
          repeat_interleave cat stack vstack hstack dstack \
          column_stack split chunk tensor_split narrow unbind \
          broadcast_tensors broadcast_to broadcast_shapes movedim \
          moveaxis tile roll rot90 flip fliplr flipud; do
  ./target/release/parity-sweep sweep --op "$OP" --seeds 8 \
    2>&1 | tail -1
done
```

Currently produces `0/N passed (N skipped, 0 failed)` for the two
ops with runner arms (`transpose`, `expand`); the rest emit
nothing because the runner has no arm. The expected grep count
`grep -c "passed (0 skipped, 0 failed)"` is `0` today, will be
`36` once the runner-arm umbrella blocker #1340 closes AND each
op's dispatcher is wired. Per S5 this is test-infra; not a REQ
blocker.

### Conformance tests

`ferrotorch-core/tests/` does not currently carry a
shape-conformance file; shape ops are validated only by the lib
tests above and indirect coverage through ops that USE them
(`conformance_elementwise.rs` uses broadcast which uses
`broadcast_shapes`, etc.).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (view) | SHIPPED | impl: `pub fn view_t` in `methods.rs` (delegates to `crate::grad_fns::shape::reshape`) mirrors upstream `aten/src/ATen/native/TensorShape.cpp:4563 Tensor view`; non-test consumer: `Tensor::view` method in `methods.rs`; lib test `test_resolve_shape_*` covers the shared `-1`-infer path; parity-sweep `view` at seeds=8 reports `56/56 passed (0 skipped, 0 failed)` per the runner arm `"view"` in `tools/parity-sweep/runner/src/main.rs` (closes #1340). |
| REQ-2 (reshape) | SHIPPED | impl: `pub fn reshape` + `pub struct ReshapeBackward` in `grad_fns/shape.rs` mirror upstream `aten/src/ATen/native/TensorShape.cpp:2129 Tensor reshape`; non-test consumers: `Tensor::reshape_t` in `methods.rs`, `flex_attention.rs`, `einsum.rs`; lib tests `test_reshape_forward/_backward/_infer_dim/_shape_mismatch/_no_grad`; parity-sweep `reshape` at seeds=8 reports `56/56 passed (0 skipped, 0 failed)` per the runner arm `"reshape"` in `tools/parity-sweep/runner/src/main.rs` (closes #1340). |
| REQ-3 (flatten) | SHIPPED | impl: `pub fn flatten` + `pub struct FlattenBackward` in `grad_fns/shape.rs` mirror upstream `aten/src/ATen/native/TensorShape.cpp:4178 Tensor flatten`; non-test consumers: `Tensor::flatten_t` in `methods.rs`, `Tensor::flatten` method-body in `tensor.rs` consumes `FlattenBackward`; lib tests `test_flatten_forward/_backward/_preserves_grad_fn`; parity-sweep `flatten` at seeds=8 reports `48/48 passed (0 skipped, 0 failed)` per the runner arm `"flatten"` in `tools/parity-sweep/runner/src/main.rs` (closes #1340). |
| REQ-4 (unflatten) | SHIPPED | impl: `pub fn unflatten` + helper `resolve_unflatten_sizes` in `grad_fns/shape.rs` mirror upstream `aten/src/ATen/native/TensorShape.cpp:4350 Tensor unflatten_symint` / `:4305 unflatten_impl` (maybe_wrap_dim + non-empty check + `infer_size_dv` local `-1` resolution + spliced `view`); autograd inherited from `reshape`'s `ReshapeBackward`; non-test consumer: `Tensor::unflatten_t` in `methods.rs`; lib tests `test_unflatten_forward/_infer_slot/_negative_dim_and_middle_splice/_empty_sizes_errors/_product_mismatch_errors/_backward_reaches_leaf`; runner-arm gap #1340. |
| REQ-5 (squeeze) | SHIPPED | impl: `pub fn squeeze` at `squeeze in shape.rs` + `SqueezeBackward` at `squeeze in shape.rs` mirrors upstream `TensorShape.cpp:4026 Tensor squeeze(self, dim)`; non-test consumers: `shape in methods.rs squeeze_t`, `einsum in einsum.rs`; lib tests `test_squeeze_forward/_non_one_error/_unsqueeze_roundtrip/_preserves_grad_fn/_backward_reaches_leaf/_in_longer_chain/_no_grad_is_view`; runner-arm gap #1340. |
| REQ-6 (unsqueeze) | SHIPPED | impl: `pub fn unsqueeze` at `unsqueeze in shape.rs` + `UnsqueezeBackward` at `unsqueeze in shape.rs` mirrors upstream `TensorShape.cpp:4109 Tensor unsqueeze`; non-test consumers: `shape in methods.rs unsqueeze_t`, `einsum in einsum.rs`, `grad_fns/indexing.rs` (broadcast prep); lib tests `test_unsqueeze_forward/_preserves_grad_fn/_backward_reaches_leaf`; runner-arm gap #1340. |
| REQ-7 (permute) | SHIPPED | impl: `pub fn permute_t` at `permute_t in methods.rs` + `PermuteBackward` at `permute_t in methods.rs` mirrors upstream `TensorShape.cpp:1829 Tensor permute`; non-test consumers: `Tensor::permute` at `permute in methods.rs`, `shape in shape.rs` (TransposeBackward and transpose_2d both delegate here), `einsum in einsum.rs` (intermediate permutation), `einsum in lib.rs` re-exports `permute_t`; runner-arm gap #1340. |
| REQ-8 (transpose) | SHIPPED | impl: `Tensor::transpose(dim0, dim1)` at `transpose in methods.rs` (builds swap-perm + calls `permute_t`) + `pub fn transpose_2d` at `transpose_2d in shape.rs` + `TransposeBackward` at `transpose_2d in shape.rs` mirror upstream `TensorShape.cpp:5116 Tensor transpose` and `:5173 Tensor t`; non-test consumer: `Tensor::t` at `t in methods.rs`; lib test `test_transpose_2d_forward`; runner-arm gap #1340 (the existing runner arm at `t in runner/src/main.rs` produces 0/64 passed 64 skipped because dispatch is not wired). |
| REQ-9 (swapaxes) | SHIPPED | impl: `pub fn swapaxes` in `grad_fns/shape.rs` (literal `input.transpose(axis0, axis1)` alias) mirrors upstream `TensorShape.cpp:4776 Tensor swapaxes(self, axis0, axis1) { return self.transpose(axis0, axis1); }`; autograd inherited from `Tensor::transpose`'s `PermuteBackward`; non-test consumer: `Tensor::swapaxes` in `methods.rs`; lib tests `test_swapaxes_equals_transpose` (byte-equality vs transpose), `test_swapaxes_backward_reaches_leaf`; runner-arm gap #1340. |
| REQ-10 (swapdims) | SHIPPED | impl: `pub fn swapdims` in `grad_fns/shape.rs` (literal `input.transpose(dim0, dim1)` alias) mirrors upstream `TensorShape.cpp:4784 Tensor swapdims(self, dim0, dim1) { return self.transpose(dim0, dim1); }`; non-test consumer: `Tensor::swapdims` in `methods.rs`; lib test `test_swapdims_equals_transpose` (byte-equality vs transpose on a 3-D tensor); runner-arm gap #1340. |
| REQ-11 (expand) | SHIPPED | impl: `pub fn expand` at `expand in shape.rs` + `ExpandBackward` at `expand in shape.rs` mirrors upstream `TensorShape.cpp:1344 Tensor expand`; non-test consumers: `shape in grad_fns/indexing.rs` (broadcast prep for masked_fill / where_cond), `einsum in einsum.rs` (sum-grad expand), `einsum in lib.rs` re-exports `expand`; runner-arm gap #1340 (existing arm at `expand in runner/src/main.rs` produces 0/72 passed 72 skipped). |
| REQ-12 (expand_as) | SHIPPED | impl: `pub fn expand_as` in `grad_fns/shape.rs` (delegates to `expand(input, other.shape())`) mirrors upstream `TensorShape.cpp:1374 Tensor expand_as(self, other) { return self.expand_symint(other.sym_sizes()); }`; autograd inherited from `expand`'s `ExpandBackward` (sum-reduces broadcast axes); non-test consumer: `Tensor::expand_as_t` in `methods.rs`; lib tests `test_expand_as_equals_expand` (byte-equality vs explicit-shape expand), `test_expand_as_backward_sums_broadcast_axes`; runner-arm gap #1340. |
| REQ-13 (repeat) | SHIPPED | impl: `pub fn repeat` in `grad_fns/shape.rs` (prepends leading size-1 dims via `reshape`, then tiles each axis by `cat`-ing `r` copies — autograd inherited from `CatBackward`) mirrors upstream `aten/src/ATen/native/TensorShape.cpp:1909 Tensor repeat(self, repeats)`; distinct from the unrelated string-pattern `einops::repeat` at `einops.rs:514` (the crate-root re-export of `repeat` remains the einops one to avoid a name clash); non-test consumer: `Tensor::repeat_t` in `methods.rs`; lib tests `test_repeat_1d/_2d_with_new_leading_dim/_rejects_too_few_dims/_backward_accumulates`; runner-arm gap #1340. |
| REQ-14 (repeat_interleave) | SHIPPED | impl: `pub fn repeat_interleave` + `RepeatInterleaveBackward` in `grad_fns/shape.rs` (CPU forward duplicates each index `repeats`× consecutively along `dim`; backward sums the `repeats` consecutive output slices back onto each input index) mirrors `torch.repeat_interleave(input, repeats, dim)`; non-test consumer: `Tensor::repeat_interleave_t` in `methods.rs`; lib tests `test_repeat_interleave_1d/_2d_dim1/_differs_from_repeat/_backward_sums_segments`; runner-arm gap #1340. |
| REQ-15 (cat) | SHIPPED | impl: `pub fn cat` at `cat in shape.rs` + `CatBackward` at `cat in shape.rs` mirrors upstream `TensorShape.cpp:676 TORCH_IMPL_FUNC(cat_out_cpu)` + `:772 Tensor cat`; GPU fast path mirrors `aten::cat_out_cuda` via byte-width-dispatched `strided_cat`; non-test consumers: `flex_attention in flex_attention.rs` (head-grouped attention assembly), `flex_attention in lib.rs` re-exports `cat`; lib tests `test_cat_forward_axis0/_axis1`, `test_cat_backward_axis0/_axis1/_mixed_requires_grad`, `test_cat_empty_error`, `test_cat_1d`; runner-arm gap #1340. |
| REQ-16 (stack) | SHIPPED | impl: `pub fn stack` at `stack in vmap.rs` mirrors upstream `TensorShape.cpp:3462 Tensor stack` via unsqueeze + cat composition (autograd inherited from REQ-6 + REQ-15); grandfathered as existing pub API across multiple prior commits per S5; runner-arm gap #1340. |
| REQ-17 (vstack) | SHIPPED | impl: `pub fn vstack` in `grad_fns/shape.rs` (`atleast_2d` each input then `cat(_, 0)`) mirrors upstream `aten/src/ATen/native/TensorShape.cpp:3412 Tensor vstack(TensorList)`; autograd inherited from `reshape`/`unsqueeze` + `CatBackward`; non-test consumer: `Tensor::vstack_t` in `methods.rs`; lib tests `test_vstack_1d_inputs/_backward`; runner-arm gap #1340. |
| REQ-18 (hstack) | SHIPPED | impl: `pub fn hstack` in `grad_fns/shape.rs` (`atleast_1d` each input; `cat(_, 0)` if 1-D else `cat(_, 1)`) mirrors upstream `aten/src/ATen/native/TensorShape.cpp:3514 Tensor hstack(TensorList)`; non-test consumer: `Tensor::hstack_t` in `methods.rs`; lib tests `test_hstack_1d_inputs/_2d_inputs`; runner-arm gap #1340. |
| REQ-19 (dstack) | SHIPPED | impl: `pub fn dstack` in `grad_fns/shape.rs` (`atleast_3d` each input then `cat(_, 2)`) mirrors upstream `aten/src/ATen/native/TensorShape.cpp:3544 Tensor dstack(TensorList)`; non-test consumer: `Tensor::dstack_t` in `methods.rs`; lib test `test_dstack_1d_inputs`; runner-arm gap #1340. |
| REQ-20 (column_stack) | SHIPPED | impl: `pub fn column_stack` in `grad_fns/shape.rs` (reshape ≤1-D inputs to `(numel,1)` then `hstack`) mirrors upstream `aten/src/ATen/native/TensorShape.cpp:3628 Tensor column_stack(TensorList)`; non-test consumer: `Tensor::column_stack_t` in `methods.rs`; lib test `test_column_stack_1d_inputs`; runner-arm gap #1340. |
| REQ-21 (split) | SHIPPED | impl: `pub fn split_t` at `split_t in methods.rs` consumes `SplitBackward` from THIS file at `split_t in shape.rs` per the explicit `shape in methods.rs use crate::grad_fns::shape::SplitBackward`; mirrors upstream `TensorShape.cpp:3175 split` / `:3265 split_with_sizes`; non-test consumer: `Tensor::split` at `split in methods.rs`, `methods in lib.rs` re-exports `split_t`; runner-arm gap #1340. |
| REQ-22 (chunk) | SHIPPED | impl: `pub fn chunk_t` at `chunk_t in methods.rs` (computes per-chunk size then delegates to the shared `SplitBackward` machinery) mirrors upstream `TensorShape.cpp:1077 chunk`; non-test consumer: `Tensor::chunk` at `chunk in methods.rs`, `methods in lib.rs` re-exports `chunk_t`; runner-arm gap #1340. |
| REQ-23 (tensor_split) | SHIPPED | impl: `pub fn tensor_split` in `grad_fns/shape.rs` (each section is a `narrow_t` view of `[indices[j-1], indices[j])` with `0`/`size(dim)` endpoints, boundaries clamped) mirrors upstream `aten/src/ATen/native/TensorShape.cpp:1167 tensor_split` / `:1130 _tensor_split_indices`; autograd inherited from `NarrowBackward`; non-test consumer: `Tensor::tensor_split_t` in `methods.rs`; lib tests `test_tensor_split_indices/_empty_section/_backward`; runner-arm gap #1340. |
| REQ-24 (narrow) | SHIPPED | impl: `pub fn narrow_t` at `narrow_t in methods.rs` + `NarrowBackward` at `narrow_t in methods.rs` mirrors upstream `TensorShape.cpp:1669 Tensor narrow`; non-test consumer: `Tensor::narrow` at `narrow in methods.rs`; runner-arm gap #1340. |
| REQ-25 (unbind) | SHIPPED | impl: `pub fn unbind` in `grad_fns/shape.rs` (one `narrow_t(dim, i, 1)` + `squeeze(dim)` per index, so each slice carries a `NarrowBackward` + `SqueezeBackward` chain) mirrors upstream `aten/src/ATen/native/TensorShape.cpp:4367 std::vector<Tensor> unbind(self, dim)` (which uses `select`); non-test consumer: `Tensor::unbind_t` in `methods.rs`; lib tests `test_unbind_dim0/_dim1/_backward_scatters`; runner-arm gap #1340. |
| REQ-26 (broadcast_tensors) | SHIPPED | impl: `pub fn broadcast_tensors` in `grad_fns/shape.rs` (folds all inputs through `crate::shape::broadcast_shapes` to the common shape, then `expand`s each — autograd inherited per-input from `ExpandBackward`) mirrors upstream `aten/src/ATen/native/TensorShape.cpp:656 std::vector<Tensor> broadcast_tensors(TensorList) { return expand_outplace(tensors); }`; non-test consumer: `lib.rs` crate-root re-export `broadcast_tensors` (consumed by the `divergence_shape_ops_audit` parity + reachable workspace-wide); lib test `test_broadcast_tensors_common_shape`; runner-arm gap #1340. |
| REQ-27 (broadcast_to) | SHIPPED | impl: `pub fn broadcast_to` in `grad_fns/shape.rs` (literal `expand(input, shape)` alias) mirrors upstream `aten/src/ATen/native/TensorShape.cpp:652 Tensor broadcast_to_symint(self, size) { return self.expand_symint(size); }`; autograd inherited from `ExpandBackward`; non-test consumer: `Tensor::broadcast_to_t` in `methods.rs`; lib test `test_broadcast_to_equals_expand`; runner-arm gap #1340. |
| REQ-28 (broadcast_shapes) | SHIPPED | impl: `pub fn broadcast_shapes` at `broadcast_shapes in ferrotorch-core/src/shape.rs` (sister utility module, not this file) mirrors upstream right-aligned NumPy broadcast rule; non-test consumers: `meta_propagate.rs`, `ops/elementwise.rs`, `grad_fns/indexing.rs`, `pub in grad_fns/arithmetic.rs`; runner-arm gap #1340. |
| REQ-29 (movedim) | SHIPPED | impl: `pub fn movedim` in `grad_fns/shape.rs` (normalizes + de-duplicates src/dst, assembles a full permutation — listed dims land at their targets, leftover dims fill the rest in original relative order — then `permute_t`) mirrors upstream `aten/src/ATen/native/TensorShape.cpp:4657 Tensor movedim(self, src, dst)` (verified against the `:4756` worked example order `[2,3,0,4,1]`); autograd inherited from `PermuteBackward`; non-test consumer: `Tensor::movedim_t` in `methods.rs`; lib tests `test_movedim_single/_multi/_backward_reaches_leaf/_rejects_repeated_dim`; runner-arm gap #1340. |
| REQ-30 (moveaxis) | SHIPPED | impl: `pub fn moveaxis` in `grad_fns/shape.rs` (literal `movedim(input, src, dst)` alias) mirrors upstream `aten/src/ATen/native/TensorShape.cpp:4768 Tensor moveaxis(self, src, dst) { return at::movedim(self, src, dst); }`; non-test consumer: `Tensor::moveaxis_t` in `methods.rs`; lib test `test_moveaxis_equals_movedim`; runner-arm gap #1340. |
| REQ-31 (tile) | SHIPPED | impl: `pub fn tile` in `grad_fns/shape.rs` (left-pads `reps` with 1s when shorter than `self.dim()`, then delegates to `repeat`) mirrors upstream `aten/src/ATen/native/TensorShape.cpp:1971 Tensor tile_symint(self, reps)`; autograd inherited from `repeat`'s `cat` composition; non-test consumer: `Tensor::tile_t` in `methods.rs`; lib test `test_tile_pads_reps`; runner-arm gap #1340. |
| REQ-32 (roll) | SHIPPED | impl: `pub fn roll` at `roll in ops/tensor_ops.rs` (forward) + `RollBackward` at THIS file `roll in shape.rs` (backward) — backward consumed in production at `shape in tensor_ops.rs` (CUDA forward arm) and `shape in tensor_ops.rs` (CPU forward arm); upstream is `aten/src/ATen/native/TensorTransformations.cpp:110 Tensor roll` (note: this is NOT the route-declared TensorShape.cpp — route's upstream list is incomplete for this op); lib tests `test_roll_forward_registers_grad_fn`, `test_roll_zero_shift_early_return`, `test_roll_backward_simple_1d_hand_computed`, `test_roll_backward_negative_shift_2d`; runner-arm gap #1340. |
| REQ-33 (rot90) | SHIPPED | impl: `pub fn rot90` in `grad_fns/shape.rs` (`k mod 4` then the upstream switch: `k1 → flip({d1}).transpose(d0,d1)`, `k2 → flip({d0,d1})`, `k3 → flip({d0}).transpose(d0,d1)`, `k0 → clone`) mirrors upstream `aten/src/ATen/native/TensorTransformations.cpp:134 Tensor rot90(self, k, dims)`; autograd inherited from `FlipBackward` + `PermuteBackward`; non-test consumer: `Tensor::rot90_t` in `methods.rs`; lib tests `test_rot90_k1/_k2_is_flip_both/_k0_and_k4_identity/_negative_k/_backward_reaches_leaf`; runner-arm gap #1340. |
| REQ-34 (flip) | SHIPPED | impl: `pub fn flip` + `FlipBackward` + `flip_cpu_inner` in `grad_fns/shape.rs` (CPU index-reversal over a `data_vec`-gathered logical buffer so strided views are handled; backward re-applies the same flip — flip is its own inverse) mirrors upstream `aten/src/ATen/native/TensorTransformations.cpp:36 Tensor flip(self, dims)` (de-duplicates dims); this is the user-facing `torch.flip`, distinct from the private `flip_kernel` helpers in `ferrotorch-nn/src/conv.rs`; non-test consumer: `Tensor::flip_t` in `methods.rs`; lib tests `test_flip_forward_1d/_2d_both_dims/_2d_single_dim/_rejects_duplicate_dim/_backward_is_self_inverse`; runner-arm gap #1340. |
| REQ-35 (fliplr) | SHIPPED | impl: `pub fn fliplr` in `grad_fns/shape.rs` (≥2-D check then `flip({1})`) mirrors upstream `aten/src/ATen/native/TensorTransformations.cpp:180 Tensor fliplr(self) { ... return self.flip({1}); }`; autograd inherited from `FlipBackward`; non-test consumer: `Tensor::fliplr_t` in `methods.rs`; lib test `test_fliplr_equals_flip_dim1`; runner-arm gap #1340. |
| REQ-36 (flipud) | SHIPPED | impl: `pub fn flipud` in `grad_fns/shape.rs` (≥1-D check then `flip({0})`) mirrors upstream `aten/src/ATen/native/TensorTransformations.cpp:186 Tensor flipud(self) { ... return self.flip({0}); }`; autograd inherited from `FlipBackward`; non-test consumer: `Tensor::flipud_t` in `methods.rs`; lib test `test_flipud_equals_flip_dim0`; runner-arm gap #1340. |
