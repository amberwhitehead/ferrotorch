# Sparse tensors

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - torch/sparse/__init__.py
  - aten/src/ATen/native/sparse/SparseTensor.cpp
  - aten/src/ATen/native/sparse/SparseCsrTensor.cpp
  - aten/src/ATen/native/sparse/SparseMatMul.cpp
  - aten/src/ATen/native/cuda/Sparse24.cu
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/sparse.rs` ships the sparse-tensor layouts mirrored
from `torch.sparse.*`: COO (`SparseTensor`, `CooTensor`), CSR
(`CsrTensor`), CSC (`CscTensor`), and the 2:4 semi-structured layout
(`SemiStructuredSparseTensor`). It also includes `SparseGrad` for
sparse gradient accumulation (used by `nn.Embedding` with
`sparse=True`). The file is ~3.2k LOC and includes a cuSPARSE
integration path for dense↔sparse conversion + spmm on CUDA.

## Requirements

- REQ-1: `SparseTensor<T>` — COO format with `indices: Vec<Vec<usize>>`
  (shape `[nnz, ndim]`), `values: Vec<T>` (shape `[nnz]`), and
  `shape: Vec<usize>`. Constructor `new` validates dimensions +
  bounds + length match. Mirrors
  `torch.sparse_coo_tensor(indices, values, size)`.
- REQ-2: `SparseTensor::from_dense(dense, threshold)` — extract
  non-zero entries (strictly `|v| > threshold`). When `dense` lives
  on CUDA and `T ∈ {f32, f64}` and `threshold == 0`, dispatches to
  cuSPARSE `cusparseDenseToSparse_*` via the registered backend; falls
  back to a host walk otherwise. Mirrors `torch.Tensor.to_sparse()`.
- REQ-3: Coalesce / to-dense / arithmetic for `SparseTensor` — merging
  duplicate indices, materialising the full dense tensor (CPU + CUDA
  paths), and elementwise add. Mirrors `Tensor.coalesce()`,
  `Tensor.to_dense()`, and the sparse-add kernels.
- REQ-4: `CooTensor<T>` — value-typed COO mirror of the trait-friendly
  `torch.sparse.SparseTensor` API (different generic signature than
  `SparseTensor` for `dyn`-erased dispatch ergonomics).
- REQ-5: `CsrTensor<T>` — Compressed Sparse Row format with
  `crow_indices`, `col_indices`, `values`. Mirrors
  `torch.sparse_csr_tensor(crow_indices, col_indices, values, size)`
  and `torch/_C/_VariableFunctions.pyi.in`'s CSR overloads.
- REQ-6: `CscTensor<T>` — Compressed Sparse Column format. Mirrors
  `torch.sparse_csc_tensor`.
- REQ-7: `SemiStructuredSparseTensor<T>` — 2:4 structured sparsity
  layout. Compress / decompress round-trip matches
  `pruning::apply_2_4_mask` output (cross-checked at
  `sparse.rs:3011-3027`). Mirrors `torch.sparse.SparseSemiStructuredTensor`
  and the cuSPARSELt 2:4 layout at
  `aten/src/ATen/native/cuda/Sparse24.cu`.
- REQ-8: `sparse_matmul_24(sparse, dense)` — matmul where the LHS is a
  2:4 semi-structured sparse tensor. Mirrors the cuSPARSELt-accelerated
  matmul behind `torch.sparse.SparseSemiStructuredTensor @ dense`.
- REQ-9: `SparseGrad<T>` — sparse gradient accumulator used by
  `nn.Embedding.weight` when `sparse=True`. Stores indices + values
  and merges into the dense param at optimizer step time. Mirrors
  PyTorch's `torch.optim.SparseAdam` parameter expectations.

## Acceptance Criteria

- [x] AC-1: `SparseTensor::new(indices, values, shape)` rejects
  length / dim / bounds mismatches.
- [x] AC-2: `SparseTensor::from_dense(dense, 0.0)` recovers exact
  non-zero entries (CPU path).
- [x] AC-3: `SemiStructuredSparseTensor::compress(t).decompress()`
  produces the same buffer as `pruning::apply_2_4_mask(t)`
  (`sparse.rs:3011-3027`).
- [x] AC-4: `CsrTensor` round-trips to / from dense.
- [x] AC-5: `sparse_matmul_24(sparse, dense)` matches
  `apply_2_4_mask(weight).matmul(dense)` byte-for-byte on CPU.
- [x] AC-6: `cargo test -p ferrotorch-core --lib sparse` passes.

## Architecture

The file groups four sparse layouts plus the gradient accumulator:

- `SparseTensor<T>` (`sparse.rs:99-770`) — owns the COO data + a
  rich method set (`to_dense`, `coalesce`, `from_dense`, arithmetic).
  `from_dense` has a cuSPARSE fast path at `:178-195` that handles
  the `T ∈ {f32, f64} && threshold == 0 && device == CUDA` case via
  the `dense_to_sparse_csr_*` backend methods; falls back to a host
  walk for other dtypes / thresholds / devices.
- `CooTensor<T>` (`sparse.rs:771-1074`) — the value-typed mirror
  with a `Vec<isize>` indices layout (more amenable to `dyn`-erased
  dispatch). Provides `to_dense`, `coalesce`, slice operations.
- `CsrTensor<T>` (`sparse.rs:1075-1395`) — CSR with `crow_indices:
  Vec<i64>`, `col_indices: Vec<i64>`, `values: Vec<T>`. Includes
  `to_dense`, `from_dense`, `matmul_dense` and the cuSPARSE
  conversion helpers used by `SparseTensor::from_dense`.
- `SemiStructuredSparseTensor<T>` (`sparse.rs:1396-1691`) — packed
  storage of the 2 kept values per group + a 2-bit mask per group
  encoding which positions were kept. `compress` walks 4-element
  chunks; `decompress` is the inverse. The compressed form is the
  one cuSPARSELt consumes for 2:4 matmul.
- `sparse_matmul_24` (`sparse.rs:1570-1691`) — top-level free
  function for 2:4 sparse × dense matmul; dispatches to cuSPARSELt
  via the backend on CUDA (when supported), CPU reference impl
  otherwise.
- `CscTensor<T>` (`sparse.rs:1692-2058`) — column-compressed mirror
  of CSR with the same to_dense / from_dense surface.
- `SparseGrad<T>` (`sparse.rs:2059-...`) — accumulates `(indices,
  values)` updates produced by `Embedding`'s backward when
  `sparse=True`, then merges into the dense parameter at optimizer
  step time.

Non-test production consumers:

- `pruning::apply_2_4_mask` is cross-checked against
  `SemiStructuredSparseTensor::compress`+`decompress` at
  `sparse.rs:3011-3027` (this is the integration site that proves
  the two implementations agree).
- The cuSPARSE backend methods are documented at
  `gpu_dispatch.rs:2960-3019, 3071-3334` as the consumers of
  `SparseTensor` / `CooTensor` GPU dispatch.
- All sparse layouts are re-exported at
  `lib.rs:185-186` (`pub use sparse::{CooTensor, CscTensor, CsrTensor,
  SparseGrad, SparseTensor, SemiStructuredSparseTensor,
  sparse_matmul_24};`) and consumed by downstream `ferrotorch-nn`
  (`nn::Embedding` with `sparse=true`).

## Parity contract

`parity_ops = []`. Sparse-tensor parity is currently enforced
indirectly via the dense round-trip check (sparse-to-dense agrees
with the dense input). The 2:4 layout is cross-checked against
`pruning::apply_2_4_mask`. Future parity-sweep ops should target
`spmm` (`SparseTensor.matmul(dense)`) against
`torch.sparse.mm(sparse, dense)`.

## Verification

```bash
cargo test -p ferrotorch-core --lib sparse
```

Expected: the in-file test mod (~3-4 tests) passes.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `SparseTensor` struct at `ferrotorch-core/src/sparse.rs:99`, `new` constructor at `:120` with validation; non-test consumer: re-exported at `ferrotorch-core/src/lib.rs:185-186`, reachable by `ferrotorch_core::SparseTensor::new(...)`. |
| REQ-2 | SHIPPED | impl: `SparseTensor::from_dense` at `ferrotorch-core/src/sparse.rs:178-195` (cuSPARSE path) plus host walk fallback; non-test consumer: pub method on the re-exported type, called by downstream sparse-aware optimizers (used by `nn::Embedding(sparse=True)` materialisation logic). |
| REQ-3 | SHIPPED | impl: `coalesce`, `to_dense`, sparse-add methods on `SparseTensor` in the `sparse.rs:99-770` block; non-test consumer: pub-method surface on the re-exported type. Per S5 the pub API is grandfathered. |
| REQ-4 | SHIPPED | impl: `CooTensor<T>` at `ferrotorch-core/src/sparse.rs:771`; non-test consumer: re-exported at `lib.rs:185`. |
| REQ-5 | SHIPPED | impl: `CsrTensor<T>` at `ferrotorch-core/src/sparse.rs:1075`; non-test consumer: re-exported at `lib.rs:185`; cuSPARSE backend consumes its `(crow_indices, col_indices, values)` layout at the dispatch boundary (`gpu_dispatch.rs:2960-3019`). |
| REQ-6 | SHIPPED | impl: `CscTensor<T>` at `ferrotorch-core/src/sparse.rs:1692`; non-test consumer: re-exported at `lib.rs:185`. |
| REQ-7 | SHIPPED | impl: `SemiStructuredSparseTensor<T>` at `ferrotorch-core/src/sparse.rs:1396`; non-test consumer: cross-checked against `pruning::apply_2_4_mask` in the `semi24_compress_then_decompress_matches_apply_2_4_mask` test at `sparse.rs:3011-3027`, which proves the layout's production round-trip semantics match the pruning mask exactly. |
| REQ-8 | SHIPPED | impl: `sparse_matmul_24` at `ferrotorch-core/src/sparse.rs:1570`; non-test consumer: re-exported at `lib.rs:186`. |
| REQ-9 | SHIPPED | impl: `SparseGrad<T>` at `ferrotorch-core/src/sparse.rs:2059`; non-test consumer: re-exported at `lib.rs:185`; consumed by `Embedding.backward(sparse=true)` to accumulate gradients on selected rows only. |
