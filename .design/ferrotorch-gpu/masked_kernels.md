# GPU mask-driven compute kernels (masked_fill / where / masked_select / masked_scatter)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/IndexKernel.cu
  - aten/src/ATen/native/TensorAdvancedIndexing.cpp
-->

## Summary

`ferrotorch-gpu/src/masked_kernels.rs` implements four families of
mask-driven GPU operations: `masked_fill`, `where` (ternary select),
`masked_select` (data-dependent compaction), and `masked_scatter`.
Masks are GPU-resident `CudaSlice<u8>` (one byte per element, 0/1),
the same representation produced by the comparison kernels, so a
`compare → mask → masked-op` chain never leaves the device. Mirrors
PyTorch's `masked_fill_kernel` / `masked_scatter_cuda` /
`masked_select` in `aten/src/ATen/native/cuda/IndexKernel.cu` and
the bool-mask `where` selection in upstream's iterator-based
`TensorAdvancedIndexing` path.

## Requirements

- REQ-1: `masked_fill` entry points for six value dtypes:
  `pub fn masked_fill_{f32, f64, f16, bf16, i32, i64}`. Each takes
  `(input, mask, fill_value, device)` (with fill_value typed
  appropriately) and returns a same-dtype output `CudaSlice`.
- REQ-2: `where` entry points: `pub fn where_32<T>`,
  `pub fn where_64<T>` (generic over the 32-/64-bit value types
  `f32/i32` and `f64/i64`), and `pub fn where_16` (concrete `u16`
  for f16/bf16 since the select is a pure 16-bit bit-pattern copy
  — no value decode needed).
- REQ-3: `masked_select` data-dependent compaction:
  `pub fn count_true` (the only host crossing — to size the
  output, exactly mirroring upstream's CUDA sync) plus per-width
  compaction entries `masked_select_32<T>`/`masked_select_64<T>`/
  `masked_select_16`. Returns a 1-D output sized by `count_true`.
- REQ-4: `masked_scatter` inverse-of-masked-select: copies source
  values into positions where the mask is true, in source order.
  Three width-templated entries
  `masked_scatter_{32, 64, 16}` matching the `where` family.
- REQ-5: Hand-written PTX kernels owned by Rust — no CUDA C++, no
  nvrtc, no external toolchain at load time. Loaded through
  `crate::module_cache::get_or_compile` exactly like
  `bool_kernels` / `cast_kernels` / `f16` / `bf16`.
- REQ-6: PyTorch-parity error policy: an unsupported (op, dtype)
  pair returns a structured error upstream
  (`FerrotorchError::NotImplementedOnCuda` / `InvalidArgument`),
  never a silent CPU detour.
- REQ-7: Non-test production consumer wiring through
  `CudaBackendImpl` — the file is imported as
  `use crate::masked_kernels as mk` at four sites in
  `backend_impl.rs` (`6660`, `6735`, `6834`, `6903`) that dispatch
  masked_fill / where / masked_select / masked_scatter calls.

## Acceptance Criteria

- [x] AC-1: Six `pub fn masked_fill_*` entries exist at lines
  731-833.
- [x] AC-2: Three `pub fn where_{32,64,16}` entries exist at lines
  833-877.
- [x] AC-3: One `pub fn count_true` at line 570 and three
  `pub fn masked_select_*` entries at lines 877-936.
- [x] AC-4: Three `pub fn masked_scatter_*` entries at lines
  936-1000.
- [x] AC-5: Every kernel launch loads PTX via
  `module_cache::get_or_compile`; no nvrtc dependency in this file.
- [x] AC-6: Unsupported dtypes are surfaced by the absence of a
  per-dtype `pub fn` (the dispatcher in `backend_impl` returns a
  `NotImplementedOnCuda` error for missing combinations).
- [x] AC-7: Four `use crate::masked_kernels as mk` sites in
  `backend_impl.rs` consume the entries.

## Architecture

The module is organised as a sequence of dtype-specialised sections:

1. **`masked_fill` PTX + entries**: 6 PTX templates
   (`MASKED_FILL_{F32,F64,F16,BF16,I32,I64}_PTX`). For `f16`/`bf16`
   the fill value is passed as `f32` and converted to the half
   bit-pattern in-kernel via `cvt.rn.f16.f32` (or the bf16 truncation
   equivalent). For `f64` the value is passed as `f64`. For
   integer dtypes the value is passed as the native int.
2. **`where` PTX + entries**: 3 PTX templates by byte-width
   (`WHERE_32_PTX`/`WHERE_64_PTX`/`WHERE_16_PTX`). The select is a
   pure bit-pattern copy — `selp.b32` / `selp.b64` / a 16-bit selp
   construction — so the same kernel serves any value type at that
   width (no separate `where_f16` and `where_bf16` PTX, just one
   `where_16`).
3. **`masked_select` compaction**: `count_true` runs an OR-style sum
   reduction kernel over the mask (the only path that reads back
   one i32 to host). The compaction kernel then serially walks the
   input, writing `input[i] -> out[j++]` for each `i` where
   `mask[i] != 0`. A parallel prefix-sum scan is a perf follow-up;
   the serial walk is correct (matches the existing serial
   reductions in `bool_kernels` / `int_kernels`).
4. **`masked_scatter` distribution**: inverse of `masked_select` —
   a serial source-cursor walks `source[..]` and writes into the
   destination position when `mask[i] != 0`.

The single host crossing in this file is the `i32` size returned by
`count_true` — the result *shape* of `masked_select`, not a data
buffer. This exactly mirrors PyTorch's internal CUDA sync to size
the data-dependent output (documented in the upstream
`masked_scatter_size_check` kernel in
`aten/src/ATen/native/cuda/IndexKernel.cu:394`).

Non-test production consumer: `backend_impl.rs` imports this file
as `use crate::masked_kernels as mk` at four locations:

- Line 6660: `masked_fill` dispatch (the `CudaBackendImpl::masked_fill_*`
  trait method body).
- Line 6735: `where` dispatch.
- Line 6834: `masked_select` dispatch.
- Line 6903: `masked_scatter` dispatch.

Each site does a `match dtype` over the value type, then forwards
to the right `mk::masked_fill_<dtype>` / `mk::where_<width>` /
`mk::masked_select_<width>` / `mk::masked_scatter_<width>` entry.

## Parity contract

`parity_ops = []` for this route. Per-op parity is enforced at the
ferrotorch-core layer (the `Tensor::masked_fill` / `where` /
`masked_select` / `masked_scatter` op tests).

Edge cases preserved:

- **`mask[i] == 0` semantics**: byte-zero means "keep input";
  non-zero means "apply fill/source/branch". Matches the
  `mask.to(dtype=bool)` PyTorch contract.
- **Data-dependent output shape**: `masked_select` issues one
  i32 host readback (via `count_true`) to size the output buffer.
  No data round-trip — only the shape integer crosses the boundary.
- **f16/bf16 unification**: `where_16` and `masked_select_16` /
  `masked_scatter_16` share a single PTX per width because the
  value is treated as opaque bits.
- **`masked_fill` value-by-value-passthrough**: the fill value enters
  via the launch parameter buffer (no extra device buffer), exactly
  mirroring upstream's `scalar value` parameter.
- **Empty input / empty mask**: the standard `BLOCK_SIZE = 256`
  launch is sized with `.max(1)` for the grid and the kernel uses
  `setp.ge.u32` to short-circuit out-of-bounds threads.

## Verification

Unit tests in `ferrotorch-gpu/src/masked_kernels.rs` `mod tests` (6
tests) cover the four op families on representative dtypes (f32,
f16, bf16). Each uses the `GpuDevice::new(0)` graceful-skip pattern.

Conformance tests at
`ferrotorch-gpu/tests/conformance_gpu_backend.rs` exercise the
trait-level masked-op surface end-to-end.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda masked_kernels:: 2>&1 | tail -3
```

Expected: ≥ 1 `test result: ok` line.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: six `pub fn masked_fill_*` at `masked_fill_ in ferrotorch-gpu/src/masked_kernels.rs`; non-test consumer: `backend_impl.rs` (`use crate::masked_kernels as mk`) dispatches per-dtype calls. |
| REQ-2 | SHIPPED | impl: `pub fn where_32`/`where_64`/`where_16` at `where_32 in masked_kernels.rs`; non-test consumer: `where_16 in backend_impl.rs`. |
| REQ-3 | SHIPPED | impl: `pub fn count_true` at `count_true in masked_kernels.rs`; `masked_select_32/64/16` at lines 877-936; non-test consumer: `backend_impl.rs`. |
| REQ-4 | SHIPPED | impl: `masked_scatter_32/64/16` at `masked_scatter_32 in masked_kernels.rs`; non-test consumer: `backend_impl.rs`. |
| REQ-5 | SHIPPED | impl: every kernel launch in this file routes through `module_cache::get_or_compile`. The file's `use crate::module_cache::get_or_compile` import at line 44 binds the single PTX load path; the file has no `cudarc::nvrtc` import. |
| REQ-6 | SHIPPED | impl: per-dtype `pub fn` entries mean the (op, dtype) coverage is structurally surfaced — a missing combination is a missing function symbol that the `backend_impl` dispatcher converts to `FerrotorchError::NotImplementedOnCuda` (the policy documented in the module `//!` block at line 36). |
| REQ-7 | SHIPPED | impl: four `use crate::masked_kernels as mk` sites in `backend_impl.rs` at lines 6660, 6735, 6834, 6903 — each is the body of a `CudaBackendImpl` trait method. ferrotorch-core dispatches `Tensor::masked_fill`/etc. through the `GpuBackend` trait when the input is CUDA-resident. |
