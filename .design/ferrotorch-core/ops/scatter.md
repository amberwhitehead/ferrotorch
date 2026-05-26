# Segmented Scatter-Add

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

`ferrotorch-core/src/ops/scatter.rs` ships `scatter_add_segments` —
the segmented scatter-add primitive that GNN message passing uses for
the `aggr="add"` aggregation. Mirrors
`torch_scatter.scatter_add(src, index, dim=0, dim_size=N)`
(third-party PyTorch extension; not in core torch, but the canonical
GNN primitive). Distinct from the per-element `ops::indexing::scatter_add`
which has the same shape on `input` and `src`; this segmented
version takes a 1-D `index` over E edges mapping into a pre-decided
`dim_size`.

## Requirements

- REQ-1: `scatter_add_segments(src, index, dim_size)` — `src: &Tensor<T>`
  with shape `[E, D]`, `index: &[i64]` of length `E`, `dim_size:
  usize` ≥ `max(index) + 1`. Returns shape `[dim_size, D]` with
  `out[i, :] = sum over { e : index[e] == i } of src[e, :]`. Mirrors
  `torch_scatter.scatter_add(src, index, dim=0, dim_size=N)`.
- REQ-2: Shape validation — `src` must be 2-D `[E, D]`; `index.len()`
  must equal `src.shape()[0]`. Errors with `ShapeMismatch` otherwise.
- REQ-3: Index validation — every `index[e]` must be in
  `[0, dim_size)`; negative or out-of-range errors with
  `InvalidArgument`.
- REQ-4: Empty rows — rows with no incoming edges stay zero (the
  initial `vec![T::zero(); ...]` fill is preserved). Matches
  `torch_scatter.scatter_add` default behaviour.
- REQ-5: CPU-only forward — CUDA `src` errors with
  `NotImplementedOnCuda`. GPU lowering NOT-STARTED (blocked on
  #1545).
- REQ-6: Forward only — no autograd. Documented in the module
  doc-comment at `:24-30`: GCN inference runs under `no_grad`; if
  autograd is needed later, the grad is a simple `gather`
  (`grad_src[e, :] = grad_out[index[e], :]`), which is a follow-up.

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib ops::scatter`
  passes (7 tests at `ops/scatter.rs:130-209`).
- [x] AC-2: Basic aggregation: `src=[[1,2],[3,4],[5,6]]`,
  `index=[0,1,0]`, `dim_size=2` → `[[6,8],[3,4]]`
  (`segments_basic_aggregation`).
- [x] AC-3: Empty rows stay zero (`segments_empty_rows_are_zero`).
- [x] AC-4: Non-2D src errors (`segments_rejects_non_2d_src`).
- [x] AC-5: Index length mismatch errors
  (`segments_rejects_index_length_mismatch`).
- [x] AC-6: Negative index errors
  (`segments_rejects_negative_index`).
- [x] AC-7: Out-of-bounds index errors
  (`segments_rejects_oob_index`).
- [ ] AC-8: GPU lowering — NOT-STARTED, blocked on #1545.

## Architecture

The single `pub fn scatter_add_segments<T: Float>` at
`ops/scatter.rs:74-128`:

1. Reject CUDA src with `NotImplementedOnCuda` (`:79-83`).
2. Validate `src` is 2-D `[E, D]` (`:84-89`).
3. Validate `index.len() == src.shape()[0] == E` (`:90-99`).
4. Allocate `out = vec![T::zero(); dim_size * d]` (`:101-102`).
5. Read `src_data = src.data_vec()` (`:104`).
6. Walk each edge `(e_idx, dst_i64) in index.iter().enumerate()`:
   - Reject negative (`:107-111`).
   - Reject out-of-range (`:112-119`).
   - Add `src_data[e_idx*d..(e_idx+1)*d]` into
     `out[dst*d..(dst+1)*d]` element-wise (`:120-124`).
7. Build result via `Tensor::from_storage(TensorStorage::cpu(out),
   vec![dim_size, d], false)`.

The implementation is intentionally narrow: separate from the
`ops::indexing::scatter_add(input, dim, index, ..., src)` which has
the same shape on `input` and `src` and writes per-element. The
segmented form is simpler to reason about (a single 1-D `index` over
E edges), so we keep it as its own primitive.

**Non-test consumer**: re-exported at `lib.rs:175` as
`ferrotorch_core::scatter_add_segments`. The intended downstream
consumer is `ferrotorch-graph` (the GNN crate, referenced in
`ferrotorch-graph/README.md:26`): `MessagePassing.aggregate(...)`
calls into this for `aggr="add"` aggregation. At this layer the
boundary symbol IS the public API per goal.md S5.

## Parity contract

`parity_ops = []` (the route declares none — `torch_scatter` is not
in `torch.ops`). Numeric contract: byte-for-byte parity with
`torch_scatter.scatter_add(src, index, dim=0, dim_size=N)` for
matching inputs, verified through unit tests + GNN integration tests
in `ferrotorch-graph/tests/`.

## Verification

`cargo test -p ferrotorch-core --lib ops::scatter` covers 7 tests
(forward correctness + all error-rejection paths).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `scatter_add_segments` at `ops/scatter.rs:74`; non-test consumer: re-exported as `ferrotorch_core::scatter_add_segments` at `lib.rs:175`; documented public consumer is `ferrotorch-graph::MessagePassing` (per `ferrotorch-graph/README.md:26`) |
| REQ-2 | SHIPPED | impl: shape validation at `ops/scatter.rs:84-99`; non-test consumer: `scatter_add_segments` entry — every public call runs through this validator |
| REQ-3 | SHIPPED | impl: per-edge validation at `ops/scatter.rs:107-119`; non-test consumer: `scatter_add_segments` entry |
| REQ-4 | SHIPPED | impl: zero-initialised `out` at `ops/scatter.rs:101-102`; non-test consumer: `scatter_add_segments` entry; tested by `segments_empty_rows_are_zero` |
| REQ-5 | SHIPPED | impl: `NotImplementedOnCuda` at `ops/scatter.rs:79-83`; non-test consumer: `scatter_add_segments` entry. GPU lowering NOT-STARTED, blocked on #1545 — does NOT block CPU SHIPPED |
| REQ-6 | SHIPPED | impl: documented in module-level `//!` comment at `ops/scatter.rs:24-30`; non-test consumer: explicit `no_grad` invocation by the `ferrotorch-graph` inference harness |
