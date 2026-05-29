# GPU Diagonal (diag_embed / diag_extract)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/TensorShape.cpp
  - aten/src/ATen/native/cuda/TriangularOps.cu
-->

## Summary

`ferrotorch-gpu/src/diag.rs` ships on-device PTX kernels for the two halves of
`torch.diag` (crosslink #1545 / sub #1535):

- **diag_embed** — a 1-D `n`-element buffer scattered onto the `k`-th diagonal
  of a fresh zero-initialised `[size, size]` matrix (`size = n + |k|`).
- **diag_extract** — the `k`-th diagonal of a 2-D `[rows, cols]` buffer
  gathered into a 1-D `diag_len`-element vector
  (`diag_len = min(rows - start_r, cols - start_c)`).

`torch.diag` itself dispatches `if (ndim == 1) return at::diag_embed(self,
offset); else return at::diagonal_copy(self, offset);`
(`aten/src/ATen/native/TensorShape.cpp:4610`). Both halves are pure
gather/scatter — every value is copied to/from one slot with no arithmetic — so
the GPU result is **bit-for-bit identical** to the ferrotorch CPU `diag` and to
`torch.diag`; no float tolerance applies.

`k` (the diagonal offset) is signed. For diag_embed, input element `i` lands at
`(r, c) = (i, i + k)` when `k >= 0` and `(i + |k|, i)` when `k < 0` — matching
the ferrotorch CPU `diag` 1-D branch and PyTorch's `diag_embed` offset
convention. For diag_extract, `start = (0, k)` when `k >= 0`, `(|k|, 0)` when
`k < 0`.

## Requirements

- REQ-1: `gpu_diag_embed_f32` — launch the f32 scatter PTX over an `n`-element
  resident `CudaBuffer<f32>`, returning a fresh `[size, size]` resident buffer
  (pre-zeroed via `alloc_zeros_f32`). One thread per input element.
- REQ-2: `gpu_diag_extract_f32` — launch the f32 gather PTX over a
  `[rows, cols]` resident buffer, returning a fresh `diag_len`-element resident
  buffer. One thread per output element.
- REQ-3: `gpu_diag_embed_f64` / `gpu_diag_extract_f64` — f64 counterparts.
- REQ-4: signed offset `k` — the embed kernel selects `(i, i+k)` vs `(i+|k|, i)`
  by `setp.lt.s32 %k`; the extract launcher shifts `start` by `|k|`. All of
  `k < 0`, `k == 0`, `k > 0` match `torch.diag(diagonal=k)`.

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-gpu --features cuda diag` passes LIVE on
  the RTX 3090.
- [x] AC-2: `gpu_diag_embed_f32([1,2,3], k=0)` equals the CPU `diag` 1-D
  reference element-for-element (`[1,0,0, 0,2,0, 0,0,3]`).
- [x] AC-3: `gpu_diag_extract_f32(arange(1,10).reshape(3,3), k=0)` is
  `[1,5,9]`; `k=1` is `[2,6]`; `k=-1` is `[4,8]`.
- [x] AC-4: positive offset `k=1` and negative `k=-1` for diag_embed match the
  CPU reference bit-for-bit.
- [x] AC-5: a CUDA tensor passed to `ops::tensor_ops::diag` / `diagflat`
  returns a tensor whose storage `is_cuda()` (NO `.cpu()` round trip) and whose
  `.cpu()` data equals the CPU reference.

## Architecture

`diag.rs` is `#![cfg(feature = "cuda")]` (mirrors `triangular.rs`). Four PTX
template constants carry one entry each:
`diag_embed_f32_kernel` / `diag_embed_f64_kernel` (5-arg ABI
`(in_ptr, out_ptr, n, size, k)`) and `diag_extract_f32_kernel` /
`diag_extract_f64_kernel` (6-arg ABI
`(in_ptr, out_ptr, diag_len, cols, start_r, start_c)`). `launch_diag_embed<V>`
and `launch_diag_extract<V>` resolve the entry via
`module_cache::get_or_compile`, validate buffer lengths, short-circuit empty
launches, and launch one 1-D grid.

The backend (`backend_impl.rs`) overrides `diag_embed_f32`/`diag_embed_f64`/
`diag_extract_f32`/`diag_extract_f64`, unwrapping the dtype-tagged handle to the
right `CudaBuffer` and re-wrapping. `ops::tensor_ops::diag` gains a CUDA branch
(after the autograd delegation, which re-enters under `no_grad`) that dispatches
embed vs extract by `ndim` and on `is_f32`/`is_f64`, returning a GPU-resident
result; other GPU dtypes keep `NotImplementedOnCuda`. `ops::tensor_ops::diagflat`
flattens via the device-aware `Tensor::view_reshape` (GPU-resident, no host
round-trip) then delegates to `diag`.

**Non-test consumer**: `ferrotorch_core::ops::tensor_ops::diag` and `::diagflat`
(re-exported as `ferrotorch_core::{diag,diagflat}`) call these backend methods
for CUDA-resident inputs.

## Parity contract

`parity_ops = []`. Numeric contract is byte-for-byte parity with `torch.diag` /
`torch.diagflat` and the ferrotorch CPU paths. Verified by the LIVE GPU-vs-CPU
unit tests in `diag.rs` and the `tensor_ops.rs` CUDA dispatch tests.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn gpu_diag_embed_f32 in diag.rs`; non-test consumer: `CudaBackendImpl::diag_embed_f32 in backend_impl.rs` dispatched from `ops::tensor_ops::diag` (1-D CUDA branch) |
| REQ-2 | SHIPPED | impl: `pub fn gpu_diag_extract_f32 in diag.rs`; non-test consumer: `CudaBackendImpl::diag_extract_f32 in backend_impl.rs` dispatched from `ops::tensor_ops::diag` (2-D CUDA branch) |
| REQ-3 | SHIPPED | impl: `pub fn gpu_diag_embed_f64` / `gpu_diag_extract_f64 in diag.rs`; non-test consumer: `CudaBackendImpl::diag_embed_f64`/`diag_extract_f64 in backend_impl.rs` |
| REQ-4 | SHIPPED | impl: `setp.lt.s32 %k` branch in the embed PTX in `diag.rs`; verified by `diag_embed_f32_negative_offset` / `diag_extract_f32_positive_offset` / `diag_extract_f32_negative_offset` unit tests |
