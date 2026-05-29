# GPU dim-aware gather / scatter / scatter_value / scatter_add

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/ScatterGatherKernel.cu
  - aten/src/ATen/native/TensorAdvancedIndexing.cpp
  - aten/src/ATen/native/ScatterGatherChecks.h
-->

## Summary

`ferrotorch-gpu/src/scatter_gather_kernels.rs` implements the
dim-parameterised, full-rank-index family that backs PyTorch's
`torch.gather` / `torch.Tensor.scatter_` / `scatter_(dim, index,
value)` / `torch.Tensor.scatter_add_` on CUDA. Unlike the byte-copy
`gather_int.rs` primitives (which serve `index_select` / the 1-D-index
`gather`), this family takes a **full-rank `i64` index** parallel to
the output (gather) or to the source (scatter), and `scatter_add` uses
a hardware ATOMIC add so duplicate index values accumulate.

Eight hand-written PTX entries (4 ops × {f32, f64}) loaded via
`module_cache::get_or_compile`. The `[outer, axis, inner]`
decomposition follows the per-dim stride indexing of upstream
`aten/src/ATen/native/cuda/ScatterGatherKernel.cu`.

## Requirements

- REQ-1: f32 family — `gpu_gather_dim_f32`, `gpu_scatter_dim_f32`,
  `gpu_scatter_value_dim_f32`, `gpu_scatter_add_dim_f32` — each taking
  the `[outer, axis, inner]`-decomposed dims plus a GPU-resident `i64`
  index and returning a fresh resident `CudaBuffer<f32>`.
- REQ-2: f64 family — the `gpu_*_dim_f64` companions, identical
  contract with `f64` value type. `scatter_add` f64 uses
  `atom.global.add.f64` (requires `sm_60+`; the live RTX 3090 is
  `sm_86`).
- REQ-3: Atomic `scatter_add` — the scatter-add PTX writes with
  `atom.global.add.f32` / `atom.global.add.f64`, so multiple threads
  whose index lands on the same output slot accumulate correctly. A
  non-atomic / last-write-wins kernel would FAIL the duplicate-index
  parity test. Mirrors `ReduceAdd` → `fastAtomicAdd` at
  `aten/src/ATen/native/cuda/ScatterGatherKernel.cu:31-48`.
- REQ-4: Dispatch wiring — the four `CudaBackendImpl::*_dim_{f32,f64}`
  overrides forward into these launchers; the production consumer is
  the `is_cuda()` branch of `ferrotorch_core::ops::indexing::gather` /
  `scatter` / `scatter_value` / `scatter_add`, which uploads the host
  `&[usize]` index as a resident `i64` buffer and keeps the result
  GPU-resident (no host round trip). bf16/f16 inputs reject with
  `NotImplementedOnCuda` (no dim-aware kernel for 2-byte dtypes yet).
- REQ-5: Segmented row scatter-add — `gpu_scatter_add_segments_f32` /
  `_f64` for the GNN message-passing primitive
  (`ops::scatter::scatter_add_segments`). `src` is `[E, D]`; `index` is
  a per-ROW `i64` segment id (length `E`, uploaded from the host
  `&[i64]`); output is the ZERO-INITIALISED `[dim_size, D]` with
  `out[index[e], :] += src[e, :]` accumulated atomically
  (`atom.global.add.f{32,64}`, `sm_60+`) over all rows. Distinct from
  the dim-aware `scatter_add` (per-ROW index, not full-rank; flat
  `seg*D + col` addressing; zero-init not clone). Mirrors
  `torch.zeros(dim_size, D).index_add_(0, index, src)` /
  `torch_scatter.scatter_add(src, index, dim=0, dim_size=N)`. bf16/f16
  reject with `NotImplementedOnCuda`.

## Layout contract

For an N-D C-contiguous tensor and axis `dim`, every shape decomposes
into `[outer, axis, inner]` where `outer = prod(shape[..dim])`,
`inner = prod(shape[dim+1..])`. The flat position of element
`(o, a, k)` is `o*axis*inner + a*inner + k`.

- **gather**: input `[outer, in_dim, inner]`; index AND output both
  `[outer, out_dim, inner]`. Thread `t` reads `idx[t]`, then
  `out[t] = in[(o*in_dim + idx[t])*inner + k]`.
- **scatter / scatter_value**: output starts as a device-to-device
  clone of input `[outer, out_dim, inner]`; index/src are
  `[outer, idx_dim, inner]`. Thread `t` writes
  `out[(o*out_dim + idx[t])*inner + k] = src[t]` (scatter) or `= value`
  (scatter_value).
- **scatter_add**: same addressing as scatter, but the write is an
  atomic add into `out[dst]`.

## Acceptance Criteria

- [x] AC-1: 8 `pub fn gpu_*_dim_{f32,f64}` symbols exist and are
  re-exported from `lib.rs`.
- [x] AC-2: The 6 in-module unit tests pass on hardware (gather dim1
  f32, scatter dim1 f32, scatter_value 1D f32, scatter_add dup-index
  f32 AND f64, gather dim0 f64).
- [x] AC-3: `scatter_add` with duplicate indices accumulates (atomic)
  rather than last-write-wins, verified live on the RTX 3090 for both
  f32 and f64.
- [x] AC-4: The four `CudaBackendImpl::*_dim_{f32,f64}` overrides
  forward into the launchers, and `ops::indexing`'s `is_cuda()`
  branches dispatch through them keeping the result resident.
- [x] AC-5: bf16/f16 CUDA inputs reject with `NotImplementedOnCuda`.
- [x] AC-6: Live-GPU divergence parity (`torch.gather` /
  `scatter_` / `scatter_(value)` / `scatter_add_`) at
  `ferrotorch-gpu/tests/divergence_scatter_gather_gpu.rs` (15 tests).
- [x] AC-7: Segmented row scatter-add `gpu_scatter_add_segments_f32` /
  `_f64` exist, are re-exported from `lib.rs`, and are wired through
  `CudaBackendImpl::scatter_add_segments_f{32,64}` into the `is_cuda()`
  branch of `ops::scatter::scatter_add_segments`. Live-GPU parity vs
  `torch.index_add_` at
  `ferrotorch-gpu/tests/divergence_scatter_add_segments_gpu.rs`
  (7 tests): basic, duplicate-segment atomic (100 rows → exact column
  sums) f32 AND f64, empty-row-stays-zero, bf16/f16 reject.

## Architecture

The file is organised around four PTX template macros
(`gather_dim_ptx!`, `scatter_dim_ptx!`, `scatter_value_dim_ptx!`,
`scatter_add_dim_ptx!`) each parameterised by value width-shift
(`2`=f32 / `3`=f64) and the load/store/atomic instruction set. Eight
`const *_PTX: &str` constants instantiate the f32/f64 cells.

The launch wrappers (`launch_gather`, `launch_scatter`,
`launch_scatter_value_f{32,64}`) resolve the PTX via
`module_cache::get_or_compile`, configure `BLOCK_SIZE = 256` with
`grid = ceil(total / 256)`, and launch one thread per output element
(gather) or per index element (scatter family). The eight `pub fn
gpu_*` entries allocate (gather) or device-to-device clone (scatter
family, via `clone_f32`/`clone_f64`) the output buffer before the
launch. All `unsafe` blocks (the kernel launches and the `i64` index
byte-reinterpret in the core consumer) carry `// SAFETY:` comments.

Non-test production consumer: the four
`CudaBackendImpl::{gather,scatter,scatter_value,scatter_add}_dim_*`
trait overrides in `ferrotorch-gpu/src/backend_impl.rs` unwrap the
`GpuBufferHandle`s and forward into the launchers; ferrotorch-core's
`ops::indexing` family dispatches through the
`GpuBackend::*_dim_{f32,f64}` trait methods on the CUDA-resident path.

## Parity contract

`parity_ops = []` for this route — parity for gather/scatter/scatter_add
is enforced at the ferrotorch-core `ops::indexing` layer and by the
live-GPU divergence test in this crate's `tests/`. This file is the
GPU primitive layer.

Edge cases preserved:

- **Duplicate indices (scatter_add)**: atomic accumulation, the
  defining case. Tested live for f32 AND f64.
- **Out-of-range index**: device UB matching upstream CUDA. The core
  validator (`validate_gather_shapes`) rejects OOB before the upload,
  so the resident path only sees in-bounds indices.
- **Empty output** (`total == 0`): the launcher early-returns `Ok`
  without launching.
- **f64 atomic**: `atom.global.add.f64` holds the whole file at
  `.target sm_60`.

## Verification

In-module unit tests (`scatter_gather_kernels::tests`, 9 tests — 6
gather/scatter/scatter_add + 3 scatter_add_segments) and the live-GPU
divergence suites
(`ferrotorch-gpu/tests/divergence_scatter_gather_gpu.rs`, 15 tests;
`ferrotorch-gpu/tests/divergence_scatter_add_segments_gpu.rs`, 7 tests)
all run on the RTX 3090.

Smoke command (no parity ops):

```bash
RUSTFLAGS="-C link-arg=-fuse-ld=lld" \
  cargo test -p ferrotorch-gpu --features cuda --test divergence_scatter_gather_gpu 2>&1 | tail -3
```

Expected: `test result: ok. 15 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `gpu_gather_dim_f32`/`gpu_scatter_dim_f32`/`gpu_scatter_value_dim_f32`/`gpu_scatter_add_dim_f32` in `ferrotorch-gpu/src/scatter_gather_kernels.rs`; non-test consumer: `CudaBackendImpl::{gather,scatter,scatter_value,scatter_add}_dim_f32` overrides in `ferrotorch-gpu/src/backend_impl.rs:7328,7380,7436,7492`, themselves consumed by the `is_cuda()` f32 arm of `ferrotorch_core::ops::indexing::{gather,scatter,scatter_value,scatter_add}` |
| REQ-2 | SHIPPED | impl: `gpu_*_dim_f64` companions in `scatter_gather_kernels.rs`; non-test consumer: `CudaBackendImpl::*_dim_f64` overrides in `backend_impl.rs:7354,7408,7464,7520`, consumed by the f64 arm of `ops::indexing` |
| REQ-3 | SHIPPED | impl: `atom.global.add.f32`/`atom.global.add.f64` in the `scatter_add_dim_ptx!` expansion; verified by `scatter_add_gpu_f32_duplicate_indices_dim0_matches_torch` / `..._f64_...` / `..._dim1_...` in `tests/divergence_scatter_gather_gpu.rs` (3 hits → slot 0, atomic sum 91 vs torch) |
| REQ-4 | SHIPPED | impl: the four `CudaBackendImpl::*_dim_{f32,f64}` overrides in `backend_impl.rs`; non-test consumer: the `is_cuda()` branches of `ferrotorch_core::ops::indexing::gather` (`ops/indexing.rs` gather CUDA arm), `scatter`, `scatter_value`, `scatter_add` — each uploads the host index as resident `i64` via `upload_index_i64` and returns a `TensorStorage::gpu` result |
| REQ-5 | SHIPPED | impl: `gpu_scatter_add_segments_f32`/`_f64` + `launch_scatter_add_segments` + `scatter_add_segments_ptx!` (atom.global.add.f{32,64}, zero-init output) in `ferrotorch-gpu/src/scatter_gather_kernels.rs`; non-test consumer: `CudaBackendImpl::scatter_add_segments_f32`/`_f64` in `ferrotorch-gpu/src/backend_impl.rs`, themselves consumed by the `is_cuda()` branch (`scatter_add_segments_cuda`) of `ferrotorch_core::ops::scatter::scatter_add_segments` — uploads the host `&[i64]` segment index once as resident `i64` and returns a `TensorStorage::gpu` result. Live-GPU verified at `tests/divergence_scatter_add_segments_gpu.rs` (7 tests, RTX 3090) |
