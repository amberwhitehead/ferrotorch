# Indexing Op Primitives

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/
  - c10/
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/ops/indexing.rs` is the forward-pass layer for
N-D indexing operations: `gather`, `scatter`, `scatter_add`,
`where_cond` (host-`&[bool]` form), `where_cond_bt`
(`BoolTensor`-typed form), and `masked_select`. Each function in
this module is the FORWARD implementation; the backward grad-fns
live in `crate::grad_fns::indexing`. The CUDA paths for
`where_cond_bt` and `masked_select` are GPU-resident through
`backend.where_cond` / stream-compaction primitives (#1185 / #1187);
the other ops are CPU-only with explicit `NotImplementedOnCuda`
returns.

## Requirements

- REQ-1: `gather(input, dim, index, index_shape)` — PyTorch
  `torch.gather` semantics: `output[i,j,k] = input[index[i,j,k], j,
  k]` for `dim=0`, etc. Output has `index_shape`. Mirrors
  `aten::gather` in `aten/src/ATen/native/TensorAdvancedIndexing.cpp`.
- REQ-2: `scatter(input, dim, index, index_shape, src)` —
  `output = input.clone(); output[index[i,j,k], j, k] = src[i,j,k]`
  for `dim=0`. Output has `input.shape`. Mirrors `aten::scatter`.
- REQ-3: `scatter_add(input, dim, index, index_shape, src)` — same
  as scatter but `+=` instead of `=`. Mirrors `aten::scatter_add`.
- REQ-4: `where_cond(condition: &[bool], x, y)` — `out[i] =
  condition[i] ? x[i] : y[i]`. Equal-shape only (no broadcasting).
  Errors with `NotImplementedOnCuda` for CUDA inputs. Mirrors
  `torch.where(condition, x, y)`.
- REQ-5: `where_cond_bt(cond: &BoolTensor, x, y)` — typed-condition
  variant. GPU-resident when all three are on the same CUDA device;
  uses `backend.where_cond` (#1185 Phase 3c). CPU fallback delegates
  to `where_cond`. Mirrors `torch.where` with a tensor condition.
- REQ-6: `masked_select(input, mask)` — return 1-D tensor of
  elements of `input` where `mask` is true, in flat C-order. GPU
  path uses on-device stream compaction (count + compact kernels per
  #1185 Phase 3c); CPU path walks `data + mask` zip. Mirrors
  `torch.masked_select`.
- REQ-7: Autograd for these forwards — when input/src requires grad
  and grad is enabled, the forward attaches the matching grad-fn
  from `crate::grad_fns::indexing` (`GatherBackward`, `ScatterBackward`,
  `ScatterAddBackward`, `WhereCondBackward`, `MaskedSelectBackward`).
  Each grad-fn implements the PyTorch-equivalent VJP.
- REQ-8: Index bounds validation — `validate_gather_shapes` at
  `:66` rejects out-of-bounds index values with
  `IndexOutOfBounds` and rank-mismatched index/input pairs with
  `InvalidArgument`.

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib ops::indexing`
  passes (covers gather/scatter/scatter_add/where_cond round-trips).
- [x] AC-2: `gather` of an out-of-bounds index value errors with
  `IndexOutOfBounds`.
- [x] AC-3: `scatter` then `gather` round-trip recovers the
  original.
- [x] AC-4: `where_cond_bt` with all-CUDA inputs stays GPU-resident
  (no `.cpu()` call on the result).
- [x] AC-5: `masked_select` GPU path emits the compacted size via
  on-device count, then the compaction kernel writes the kept
  elements (no host bounce except for the single integer count).
- [ ] AC-6: GPU paths for `gather` / `scatter` / `scatter_add` —
  NOT-STARTED, blocked on #1535 (these CPU-only paths need
  GPU lowering through `crate::grad_fns::indexing::GatherBackward`'s
  resident path).

## Architecture

`gather` at `ops/indexing.rs:112` walks the output flat index,
decomposes coords from `index_shape`, replaces the `dim`-th coord
with `index[out_flat]`, and reads `input_data[src_flat]`. Validates
the index value is in `[0, input.shape[dim])` via
`validate_gather_shapes`. If `input.requires_grad()`, attaches
`crate::grad_fns::indexing::GatherBackward { input, dim, index,
index_shape }` via `Tensor::from_operation`.

`scatter` at `:183` starts with `output = input.data_vec()?` (a
clone), then walks the index slice writing `output[dst_flat] =
src_data[i]`. `scatter_add` at `:259` is symmetric but with `+=`.
Both check the src has at least `index_numel` elements.

`where_cond` at `:334` requires `x.shape() == y.shape() ==
condition.len()`. Walks zipped `condition + x_data + y_data`. When
either x/y requires grad, wraps in `WhereCondBackward { x, y,
condition: BoolTensor::from_slice(condition, &output_shape) }`. The
backward stores a CPU `BoolTensor` even though the input was
`&[bool]` (no shape info on a raw slice).

`where_cond_bt` at `:397` is the typed-condition variant. GPU-
resident fast path at `:421-453`: all three on same CUDA device →
`backend.where_cond(cond, x, y)` (#1185 Phase 3c). When grad is
needed, stores the GPU cond directly in `WhereCondBackward` (no
`cond.to(Cpu)`; #1187 Phase 3d). CPU fallback at `:458` materialises
the bool slice via `cond.data()?` and delegates to `where_cond`.

`masked_select` at `:478` rejects mask/input numel mismatch. GPU
path at `:492` (if both are CUDA, same device) routes to
`backend.masked_select_count_*` + `backend.masked_select_compact_*`
— the count is a single integer that crosses to the host to size
the data-dependent output (PyTorch parity: a CUDA sync sizes the
output). The data itself stays GPU-resident.

**Non-test consumers**:

- `crate::tensor::Tensor::masked_select` at `tensor.rs:1146`
  invokes `crate::ops::indexing::masked_select(self, mask)` — the
  method-style entry point on `Tensor<T>`. REQ-6 consumer.
- `crate::grad_fns::cumulative::cumsum_backward` at
  `grad_fns/cumulative.rs:503` calls
  `crate::ops::indexing::scatter_add(&zeros, dim as isize, indices,
  input_shape, grad_output)` — the production scatter-add consumer
  in the cumulative-sum backward. REQ-3 consumer.
- `crate::grad_fns::indexing::masked_select_backward` at
  `grad_fns/indexing.rs:1823,1828` calls
  `crate::ops::indexing::masked_select(input, mask)` — recursive
  use in the autograd VJP. REQ-6 consumer.
- `crate::grad_fns::indexing::where_cond_backward` at
  `grad_fns/indexing.rs:1845,1853` calls
  `crate::ops::indexing::where_cond_bt(cond, x, y)` — REQ-5 consumer.
- Re-exported at `lib.rs:174` as
  `ferrotorch_core::{gather, masked_select, scatter, scatter_add,
  where_cond, where_cond_bt}`.

## Parity contract

`parity_ops = []` (no specific parity-sweep op declared in the
route). The indexing operations participate in parity-sweep through
the differentiable wrappers in `grad_fns::indexing` (e.g.
`gather_differentiable`, `where_differentiable`); divergence in
those tests would surface against this module's forward.

## Verification

`cargo test -p ferrotorch-core --lib ops::indexing` covers the
forward semantics + index-bounds validation. Backward correctness is
covered by `cargo test -p ferrotorch-core --lib grad_fns::indexing`.
GPU paths covered by `ferrotorch-core/tests/conformance_indexing.rs`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `gather` at `ops/indexing.rs:112`; non-test consumer: re-exported as `ferrotorch_core::gather` at `lib.rs:174` (boundary public API per goal.md S5) |
| REQ-2 | SHIPPED | impl: `scatter` at `ops/indexing.rs:183`; non-test consumer: re-exported as `ferrotorch_core::scatter` at `lib.rs:174` |
| REQ-3 | SHIPPED | impl: `scatter_add` at `ops/indexing.rs:259`; non-test consumer: `crate::grad_fns::cumulative::cumsum_backward` at `grad_fns/cumulative.rs:503` invokes `crate::ops::indexing::scatter_add` for the cumulative-sum backward — production autograd consumer |
| REQ-4 | SHIPPED | impl: `where_cond` at `ops/indexing.rs:334`; non-test consumer: re-exported as `ferrotorch_core::where_cond` at `lib.rs:174`; called transitively from `where_cond_bt` CPU fallback at `:458` |
| REQ-5 | SHIPPED | impl: `where_cond_bt` at `ops/indexing.rs:397`; non-test consumer: `crate::grad_fns::indexing::where_differentiable` at `grad_fns/indexing.rs:1845,1853` invokes `crate::ops::indexing::where_cond_bt(cond, x, y)` for both the residency-detected and CPU paths |
| REQ-6 | SHIPPED | impl: `masked_select` at `ops/indexing.rs:478`; non-test consumer: `crate::tensor::Tensor::masked_select` at `tensor.rs:1146` invokes `crate::ops::indexing::masked_select(self, mask)`; also `crate::grad_fns::indexing::masked_select_backward` at `grad_fns/indexing.rs:1823,1828` |
| REQ-7 | SHIPPED | impl: grad-fn attachment in each forward path (e.g. `gather` at `ops/indexing.rs:154-164`, `scatter` at `:234-245`, `scatter_add` at `:310-321`, `where_cond` at `:374-386`); non-test consumer: every autograd-tracking caller of these forwards |
| REQ-8 | SHIPPED | impl: `validate_gather_shapes` at `ops/indexing.rs:66`; non-test consumer: invoked from `gather`, `scatter`, and `scatter_add` |
