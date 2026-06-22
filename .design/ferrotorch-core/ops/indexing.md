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
`gather`, `scatter`, `scatter_value`, `scatter_add`, `where_cond`,
`where_cond_bt`, and `masked_select` are GPU-resident through rank-aware
indexing kernels, `backend.where_cond`, and stream-compaction primitives
(#1185 / #1187 / #1545). Host index/mask slices are uploaded to CUDA as
resident buffers; value tensors and results do not round-trip through CPU.

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
  For CUDA `x`/`y`, uploads the host condition once and delegates to
  `where_cond_bt`, keeping value tensors and result resident. Mirrors
  `torch.where(condition, x, y)` for a full flat condition mask.
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
- [x] AC-6: GPU paths for `gather` / `scatter` / `scatter_value` /
  `scatter_add` — SHIPPED (#1545 / sub #1535). Each pub fn has a
  CUDA-resident fast path (f32/f64/f16/bf16) that uploads the host index as a
  resident `i64` buffer and dispatches through
  `GpuBackend::{op}_dim_*` / `GpuBackend::{op}_nd_*` to the PTX kernels in
  `ferrotorch-gpu/src/scatter_gather_kernels.rs`, keeping the result
  GPU-resident. f16/bf16 scatter-add accumulates in f32 and rounds back to
  the original dtype. Live-GPU parity vs torch is pinned by
  `ferrotorch-gpu/tests/divergence_scatter_gather_gpu.rs` and the CUDA
  conformance lane in `ferrotorch-core/tests/conformance_shape.rs`.

## Architecture

`gather` at `ops/indexing.rs:112` walks the output flat index,
decomposes coords from `index_shape`, replaces the `dim`-th coord
with `index[out_flat]`, and reads `input_data[src_flat]`. Validates
the index value is in `[0, input.shape[dim])` via
`validate_gather_shapes`. If `input.requires_grad()`, attaches
`crate::grad_fns::indexing::GatherBackward { input, dim, index,
index_shape }` via `Tensor::from_operation`.

`scatter` / `scatter_value` / `scatter_add` validate the index tensor and
non-dim shape constraints before dispatch. CPU paths clone/walk host data.
CUDA paths materialise non-contiguous value operands on-device, upload the
host index as resident `i64`, and dispatch through rank-aware backend kernels;
the result stays on CUDA.

`where_cond` requires `x.shape() == y.shape() == condition.len()`. CPU walks
zipped `condition + x_data + y_data`. CUDA wraps the host condition as a
`BoolTensor`, uploads it to the operand device, and delegates to
`where_cond_bt`, so value tensors and result stay resident. When either x/y
requires grad, the forward wraps in `WhereCondBackward` with a CPU condition
for CPU execution or resident condition for CUDA execution.

`where_cond_bt` is the typed-condition variant. GPU-resident fast path:
all three on same CUDA device →
`backend.where_cond(cond, x, y)` (#1185 Phase 3c). When grad is
needed, stores the GPU cond directly in `WhereCondBackward` (no
`cond.to(Cpu)`; #1187 Phase 3d). CPU fallback materialises the bool slice via
`cond.data()?` and delegates to `where_cond`.

`masked_select` at `:1165` rejects mask/input numel mismatch. The GPU
path inside `masked_select in ops/indexing.rs` (if both are CUDA, same
device) routes to
`backend.masked_select_count_*` + `backend.masked_select_compact_*`
— the count is a single integer that crosses to the host to size
the data-dependent output (PyTorch parity: a CUDA sync sizes the
output). The data itself stays GPU-resident.

**Non-test consumers**:

- `crate::tensor::Tensor::masked_select` at `tensor.rs:2066`
  invokes `crate::ops::indexing::masked_select(self, mask)` — the
  method-style entry point on `Tensor<T>`. REQ-6 consumer.
- `crate::grad_fns::cumulative::cumsum_backward` at
  `grad_fns/cumulative.rs:571` calls
  `crate::ops::indexing::scatter_add(&zeros, dim as isize, indices,
  input_shape, grad_output)` — the production scatter-add consumer
  in the cumulative-sum backward. REQ-3 consumer.
- `crate::grad_fns::indexing::masked_select_bcast` at
  `grad_fns/indexing.rs:2294,2298` calls
  `crate::ops::indexing::masked_select(input, mask)` — broadcast wrapper
  consumer. REQ-6 consumer.
- `crate::ops::indexing::where_cond_bt` at
  `grad_fns/indexing.rs:3867` is called from scatter-reduce backward for
  the value-aware VJP — REQ-5 consumer.
- Re-exported at `lib.rs:211` as
  `ferrotorch_core::{gather, masked_select, scatter, scatter_add,
  where_cond, where_cond_bt}`.

## Parity contract

`parity_ops = []` (no specific parity-sweep op declared in the
route). The indexing operations participate in parity-sweep through
the differentiable wrappers in `grad_fns::indexing` (e.g.
`where_cond_bcast`, `masked_select_bcast`, and the scatter-reduce
backward helpers); divergence in those tests would surface against this
module's forward.

## Verification

`cargo test -p ferrotorch-core --lib ops::indexing` covers the
forward semantics + index-bounds validation. Backward correctness is
covered by `cargo test -p ferrotorch-core --lib grad_fns::indexing`.
GPU paths are covered by `ferrotorch-core/tests/conformance_indexing.rs`,
`ferrotorch-core/tests/conformance_shape.rs::gpu_indexing_ops_on_cuda`, and
the focused `ferrotorch-gpu` divergence tests.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `gather` at `ops/indexing.rs:112`; non-test consumer: re-exported as `ferrotorch_core::gather` at `lib.rs:211` (boundary public API per goal.md S5) |
| REQ-2 | SHIPPED | impl: `scatter` at `ops/indexing.rs:183`; non-test consumer: re-exported as `ferrotorch_core::scatter` at `lib.rs:211` |
| REQ-3 | SHIPPED | impl: `scatter_add` at `ops/indexing.rs:861`; non-test consumer: `crate::grad_fns::cumulative::cumsum_backward` at `grad_fns/cumulative.rs:571` invokes `crate::ops::indexing::scatter_add` for the cumulative-sum backward — production autograd consumer |
| REQ-4 | SHIPPED | impl: `where_cond in ops/indexing.rs`; non-test consumer: re-exported as `ferrotorch_core::where_cond` at `ops in lib.rs`; called transitively from `where_cond_bt` CPU fallback at `ops in lib.rs` |
| REQ-5 | SHIPPED | impl: `where_cond_bt` at `ops/indexing.rs:1185`; non-test consumer: `crate::ops::indexing::where_cond_bt` at `grad_fns/indexing.rs:3867` is invoked from scatter-reduce backward for the value-aware VJP |
| REQ-6 | SHIPPED | impl: `masked_select` at `ops/indexing.rs:1211`; non-test consumer: `crate::tensor::Tensor::masked_select` at `tensor.rs:2066` invokes `crate::ops::indexing::masked_select(self, mask)`; also `crate::grad_fns::indexing::masked_select_bcast` at `grad_fns/indexing.rs:2294,2298` |
| REQ-7 | SHIPPED | impl: grad-fn attachment in each forward path (e.g. `gather` at `attachment in ops/indexing.rs`, `scatter in ops/indexing.rs`, `scatter_add in ops/indexing.rs`, `where_cond in ops/indexing.rs`); non-test consumer: every autograd-tracking caller of these forwards |
| REQ-8 | SHIPPED | impl: `validate_gather_shapes in ops/indexing.rs` (the SAFETY-bounded i64 widening lives in `upload_index_i64 in ops/indexing.rs`); non-test consumer: invoked from `gather`, `scatter`, and `scatter_add` |
| REQ-9 | SHIPPED | CUDA-resident dim-aware/rank-aware paths for `gather`/`scatter`/`scatter_value`/`scatter_add` (#1545 / sub #1535). impl: the `is_cuda()` f32/f64/f16/bf16 branches in each `ops/indexing.rs` pub fn, plus `upload_index_i64` (host `&[usize]` → resident `i64`); non-test consumer: each branch dispatches through `crate::gpu_dispatch::GpuBackend::{gather,scatter,scatter_value,scatter_add}_{dim,nd}_*`, implemented by `ferrotorch-gpu::CudaBackendImpl` over the PTX kernels in `ferrotorch-gpu/src/scatter_gather_kernels.rs`. The result stays GPU-resident (`TensorStorage::gpu`). The same-module non-CUDA callers (`ferrotorch_core::{gather,scatter,scatter_add}` re-exports at `lib.rs:207`, `Tensor::scatter_value_t`) reach the branch whenever their operands are CUDA-resident. |
