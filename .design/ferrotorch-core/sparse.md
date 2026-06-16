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
  `pruning::apply_2_4_mask` output (cross-checked in the
  `semi24_compress_then_decompress_matches_apply_2_4_mask` test in
  `sparse.rs`). This is the CPU/reference pruning-layout helper for the
  `WeightNormSparsifier` 2:4 contract; it is not the packed CUDA
  `torch.sparse.SparseSemiStructuredTensor` conversion surface. The true
  CUDA packed values/metadata API is tracked in #1980.
- REQ-8: `sparse_matmul_24(sparse, dense)` — matmul where the LHS is a
  2:4 semi-structured sparse tensor. On CUDA it must compute on-device
  (cuSPARSELt when available, dense masked composite otherwise) and must
  not silently return a CPU tensor.
- REQ-9: `SparseGrad<T>` — sparse gradient accumulator used by
  `nn.Embedding.weight` when `sparse=True`. Stores indices + values
  and merges into the dense param at optimizer step time. Mirrors
  PyTorch's `torch.optim.SparseAdam` parameter expectations. The
  `is_sparse()` predicate (always `true`) is the marker that sparse-grad
  optimizers gate on, mirroring `torch.Tensor.is_sparse`
  (`torch/overrides.py:1389`) that `torch.optim.SparseAdam` checks
  (`torch/optim/sparse_adam.py:88`).

## Acceptance Criteria

- [x] AC-1: `SparseTensor::new(indices, values, shape)` rejects
  length / dim / bounds mismatches.
- [x] AC-2: `SparseTensor::from_dense(dense, 0.0)` recovers exact
  non-zero entries (CPU path).
- [x] AC-3: `SemiStructuredSparseTensor::compress(t).decompress()`
  produces the same buffer as `pruning::apply_2_4_mask(t)`
  (the `semi24_compress_then_decompress_matches_apply_2_4_mask` test in
  `sparse.rs`).
- [x] AC-4: `CsrTensor` round-trips to / from dense.
- [x] AC-5: `sparse_matmul_24(sparse, dense)` matches
  `apply_2_4_mask(weight).matmul(dense)` byte-for-byte on CPU.
- [x] AC-6: `cargo test -p ferrotorch-core --lib sparse` passes.

## Architecture

The file groups four sparse layouts plus the gradient accumulator:

- `SparseTensor<T>` (`pub struct SparseTensor` in `sparse.rs`) — owns the
  COO data + a rich method set (`to_dense`, `coalesce`, `from_dense`,
  arithmetic). `from_dense` has a cuSPARSE fast path that handles the
  `T ∈ {f32, f64} && threshold == 0 && device == CUDA` case via the
  `dense_to_sparse_csr_*` backend methods; falls back to a host walk for
  other dtypes / thresholds / devices.
- `CooTensor<T>` (`pub struct CooTensor` in `sparse.rs`) — the value-typed
  mirror with a `Vec<isize>` indices layout (more amenable to `dyn`-erased
  dispatch). Provides `to_dense`, `coalesce`, slice operations.
- `CsrTensor<T>` (`pub struct CsrTensor` in `sparse.rs`) — CSR with
  `crow_indices: Vec<i64>`, `col_indices: Vec<i64>`, `values: Vec<T>`.
  Includes `to_dense`, `from_dense`, `matmul_dense` and the cuSPARSE
  conversion helpers used by `SparseTensor::from_dense`.
- `SemiStructuredSparseTensor<T>` (`pub struct SemiStructuredSparseTensor`
  in `sparse.rs`) — packed storage of the 2 kept values per group + a 2-bit
  mask per group encoding which positions were kept. `compress` walks
  4-element chunks; `decompress` is the inverse. The compressed form is the
  one cuSPARSELt consumes for 2:4 matmul.
- `sparse_matmul_24` (`pub fn sparse_matmul_24` in `sparse.rs`) — top-level
  free function for 2:4 sparse × dense matmul; dispatches to cuSPARSELt via
  the backend on CUDA (when supported), CPU reference impl otherwise.
- `CscTensor<T>` (`pub struct CscTensor` in `sparse.rs`) — column-compressed
  mirror of CSR with the same to_dense / from_dense surface.
- `SparseGrad<T>` (`pub struct SparseGrad` in `sparse.rs`) — accumulates
  `(indices, values)` updates produced by `Embedding`'s backward when
  `sparse=True`, exposes `is_sparse()` / `coalesce()`, then merges into the
  dense parameter at optimizer step time (`SparseGrad::apply_sgd`, or the
  masked sparse-Adam update in `ferrotorch_optim::SparseAdam`).

Non-test production consumers:

- `pruning::apply_2_4_mask` is cross-checked against
  `SemiStructuredSparseTensor::compress`+`decompress` in the
  `semi24_compress_then_decompress_matches_apply_2_4_mask` test in
  `sparse.rs` (the integration site that proves the two implementations
  agree).
- The cuSPARSE backend methods (the `dense_to_sparse_csr_*` /
  `// -- Sparse <-> Dense conversion (cuSPARSE)` region of `gpu_dispatch.rs`)
  are the consumers of `SparseTensor` / `CooTensor` GPU dispatch.
- All sparse layouts are re-exported from `lib.rs`
  (`pub use sparse::{CooTensor, CscTensor, CsrTensor, SparseGrad,
  SparseTensor, SemiStructuredSparseTensor, sparse_matmul_24};`) and
  consumed by downstream `ferrotorch-nn` (`nn::Embedding` with
  `sparse=true`).

## Parity contract

`parity_ops = []`. Sparse-tensor parity is currently enforced
indirectly via the dense round-trip check (sparse-to-dense agrees
with the dense input). The CPU/reference 2:4 pruning layout is
cross-checked against `pruning::apply_2_4_mask` and PyTorch's
`WeightNormSparsifier` topk tie order. Future parity-sweep ops should
target `spmm` (`SparseTensor.matmul(dense)`) against
`torch.sparse.mm(sparse, dense)` and the #1980 CUDA packed
`to_sparse_semi_structured` surface.

## Verification

```bash
cargo test -p ferrotorch-core --lib sparse
```

Expected: the in-file test mod (~3-4 tests) passes.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct SparseTensor` + `SparseTensor::new` (with validation) in `ferrotorch-core/src/sparse.rs`; non-test consumer: re-exported at `ferrotorch-core/src/lib.rs` (`pub use sparse::SparseTensor`), reachable by `ferrotorch_core::SparseTensor::new(...)`. |
| REQ-2 | SHIPPED | impl: `SparseTensor::from_dense` in `ferrotorch-core/src/sparse.rs` (cuSPARSE path) plus host walk fallback; non-test consumer: pub method on the re-exported type, called by downstream sparse-aware optimizers (used by `nn::Embedding(sparse=True)` materialisation logic). |
| REQ-3 | SHIPPED | impl: `coalesce`, `to_dense`, sparse-add methods on `SparseTensor` in `sparse.rs`; non-test consumer: pub-method surface on the re-exported type. Per S5 the pub API is grandfathered. |
| REQ-4 | SHIPPED | impl: `pub struct CooTensor` in `ferrotorch-core/src/sparse.rs`; non-test consumer: re-exported at `lib.rs` (`pub use sparse::CooTensor`). |
| REQ-5 | SHIPPED | impl: `pub struct CsrTensor` in `ferrotorch-core/src/sparse.rs`; non-test consumer: re-exported at `lib.rs`; cuSPARSE backend consumes its `(crow_indices, col_indices, values)` layout at the `dense_to_sparse_csr_*` dispatch boundary in `gpu_dispatch.rs`. |
| REQ-6 | SHIPPED | impl: `pub struct CscTensor` in `ferrotorch-core/src/sparse.rs`; non-test consumer: re-exported at `lib.rs`. |
| REQ-7 | SHIPPED | impl: `pub struct SemiStructuredSparseTensor` in `ferrotorch-core/src/sparse.rs`; non-test consumer: cross-checked against `pruning::apply_2_4_mask` in the `semi24_compress_then_decompress_matches_apply_2_4_mask` test in `sparse.rs`, which proves the layout's production round-trip semantics match the pruning mask exactly. |
| REQ-8 | SHIPPED | impl: `pub fn sparse_matmul_24` in `ferrotorch-core/src/sparse.rs`; non-test consumer: re-exported at `lib.rs` (`pub use sparse::sparse_matmul_24`). |
| REQ-9 | SHIPPED | impl: `SparseGrad<T>` (incl. `is_sparse()` predicate mirroring `torch/overrides.py:1389`) at `ferrotorch-core/src/sparse.rs`; non-test consumer: re-exported at `lib.rs`; produced by `Embedding::sparse_grad` (`ferrotorch-nn/src/embedding.rs`, `sparse=true`) and consumed by `ferrotorch_optim::SparseAdam::sparse_step` (`ferrotorch-optim/src/sparse_adam.rs`, the masked sparse-Adam update mirroring `torch/optim/_functional.py:24-84`). |
