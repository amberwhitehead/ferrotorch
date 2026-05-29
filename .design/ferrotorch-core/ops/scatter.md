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
- REQ-5: Forward on CPU AND CUDA — the CPU path is the row-loop
  accumulation; the CUDA path (`scatter_add_segments_cuda`) materialises
  `src` contiguous on-device, uploads the host `&[i64]` segment index
  once to a resident `i64` buffer, and runs the atomic segmented
  row-scatter-add GPU kernel (`gpu_scatter_add_segments_f{32,64}` in
  `ferrotorch-gpu`), keeping the result GPU-resident (no host round trip
  for src/out data). f32 AND f64; bf16/f16 CUDA reject with
  `NotImplementedOnCuda`. GPU lowering landed under #1545 / sub #1535.
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
- [x] AC-8: GPU lowering — the `is_cuda()` branch dispatches through
  `GpuBackend::scatter_add_segments_f{32,64}` into the atomic PTX
  kernel; result stays GPU-resident. Live-GPU parity vs
  `torch.zeros(N,D).index_add_(0, index, src)` at
  `ferrotorch-gpu/tests/divergence_scatter_add_segments_gpu.rs`
  (7 tests, RTX 3090): basic f32/f64, duplicate-segment atomic
  (100 rows → exact column sums) f32/f64, empty-row-stays-zero,
  bf16/f16 reject.

## Architecture

The single `pub fn scatter_add_segments<T: Float>` in `ops/scatter.rs`:

1. Validate `src` is 2-D `[E, D]`.
2. Validate `index.len() == src.shape()[0] == E`.
3. Per-edge segment-id validation (shared by CPU and CUDA): reject
   negative / `>= dim_size`. This runs on the HOST before any device
   upload — the CUDA kernel does no device-side bounds check.
4. If `src.is_cuda()`, dispatch to `scatter_add_segments_cuda` (below).
5. CPU path: allocate `out = vec![T::zero(); dim_size * d]`, read
   `src_data = src.data_vec()`, walk each edge and add
   `src_data[e_idx*d..(e_idx+1)*d]` into `out[dst*d..(dst+1)*d]`.
6. Build result via `Tensor::from_storage(TensorStorage::cpu(out),
   vec![dim_size, d], false)`.

**CUDA path** (`scatter_add_segments_cuda`): rejects bf16/f16 with
`NotImplementedOnCuda`; materialises `src` contiguous on-device
(`contiguous()` — no host round trip); uploads the host `&[i64]`
segment index once to a resident `i64` buffer (uploading a
freshly-provided host INPUT, not device data); dispatches
`GpuBackend::scatter_add_segments_f32`/`_f64` (zero-init `[dim_size, d]`
output + atomic row scatter-add); returns `TensorStorage::gpu`.

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
| REQ-5 | SHIPPED | impl: CPU row-loop + CUDA `scatter_add_segments_cuda` (contiguous-on-device, host-`&[i64]` index upload, `backend.scatter_add_segments_f{32,64}`, `TensorStorage::gpu`) in `ferrotorch-core/src/ops/scatter.rs`; GPU primitive `gpu_scatter_add_segments_f{32,64}` + `CudaBackendImpl::scatter_add_segments_f{32,64}` in `ferrotorch-gpu`; non-test consumer: `scatter_add_segments` public entry (the `is_cuda()` branch), re-exported `ferrotorch_core::scatter_add_segments` for `ferrotorch-graph::MessagePassing`. GPU lowering landed #1545 / sub #1535; live-GPU verified at `ferrotorch-gpu/tests/divergence_scatter_add_segments_gpu.rs` |
| REQ-6 | SHIPPED | impl: documented in module-level `//!` comment at `ops/scatter.rs`; non-test consumer: explicit `no_grad` invocation by the `ferrotorch-graph` inference harness |
