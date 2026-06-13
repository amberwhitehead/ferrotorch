# GPU index_select / gather with GPU-resident integer indices

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/Indexing.cu
  - aten/src/ATen/native/cuda/ScatterGatherKernel.cu
  - aten/src/ATen/native/cuda/IndexKernel.cu
-->

## Summary

`ferrotorch-gpu/src/gather_int.rs` implements GPU `index_select(dim)`
and `gather(dim)` kernels driven by GPU-resident **integer** index
buffers (`i32` / `i64`). Element movement is dtype-generic by byte
width (2 / 4 / 8 bytes), so a single 4-byte copy kernel serves both
f32 and i32. 18 hand-written PTX entries cover the 3 byte-widths × 2
index-widths × 3 op/layout matrix: `index_select`, compact gather,
and rank-aware gather for PyTorch-legal index shapes whose non-gather
dimensions are smaller than the input. Mirrors PyTorch's CUDA
`index_select` (in `Indexing.cu`) and `gather` (in
`ScatterGatherKernel.cu`), with the same checked-before-launch index
contract at the core layer.

## Requirements

- REQ-1: Twenty public entry points — `isel_<vty>_<ity>` and
  `gather_<vty>_<ity>` for vty in `{f32, f64, i32, i64, u16}` and
  ity in `{i32, i64}` — each taking
  `(input, idx, outer, in_dim, out_dim, inner, device)` and returning
  a same-vty `CudaSlice` output. (5 value × 2 index × 2 ops = 20.)
- REQ-2: 12 hand-written PTX templates expanded by the
  `index_select_ptx!` / `gather_ptx!` macros: 3 byte-widths
  (`W2`=2B, `W4`=4B, `W8`=8B) × 2 index widths (`I32`/`I64`) × 2 ops.
  Generic byte-copy idiom — values are loaded/stored as raw
  `u16`/`u32`/`u64` and the kernel never decodes them.
- REQ-2b: Six additional `gather_nd_ptx!` templates for rank-aware
  contiguous gather. They take device metadata for C-order input
  strides and index/output shape, decode each output coordinate from
  `index_shape`, replace only `coord[dim]` with `index[t]`, and copy
  the selected value without downloading the value buffer.
- REQ-3: Layout contracts: `index_select(dim)` uses input
  `[outer, in_dim, inner]`, index `[out_dim]`, output
  `[outer, out_dim, inner]`. `gather(dim)` uses input
  `[outer, in_dim, inner]`, index AND output both
  `[outer, out_dim, inner]` (per-output-element index lookup).
- REQ-4: Out-of-range index contract: PyTorch CUDA parity — no
  device-side bounds check, no host round-trip to validate. Out-of-range
  indices are documented undefined behaviour on the device.
- REQ-5: Non-test production consumer at
  `ferrotorch-gpu/src/backend_impl.rs:442-568` — the
  `CudaBackendImpl::gather_or_select` trait method dispatches all
  20 (value, index, op) combinations through the kernels in this
  file.
- REQ-6: Non-test production consumer for rank-aware gather:
  `CudaBackendImpl::gather_intidx_nd` dispatches the 10
  `(value dtype, index dtype)` cells through `gather_nd_*`; the
  CORE-112 branch in `ferrotorch-core/src/ops/phase2c.rs` calls it
  when `index.shape()` is smaller than `input.shape()` on a
  non-gather axis.

## Acceptance Criteria

- [x] AC-1: 20 `pub fn isel_*` / `pub fn gather_*` symbols exist
  (macro-stamped via `select_entry!`).
- [x] AC-2: 12 PTX constants exist (6 select + 6 gather) covering
  the 3 byte-widths × 2 index-widths combinations.
- [x] AC-2b: 6 `GATHER_ND_*` PTX constants exist and are exercised by
  unit tests for smaller non-gather dimensions.
- [x] AC-3: The four unit tests in `mod tests` exercise small-shape
  parity for at least one of each: f32+i32, f32+i64, f64+i64, and
  a half-precision (u16-backed) flow.
- [x] AC-4: `CudaBackendImpl::gather_or_select` dispatches all 20
  (vty, ity, op) cells via the `match src.dtype()` + `run!` macro
  expansion at `backend_impl.rs`.
- [x] AC-5: Out-of-range UB contract documented in the module `//!`
  block at lines 33-39, matching upstream.

## Architecture

`gather_int.rs` organises around two pillars:

1. **PTX template macros** (`index_select_ptx!` and `gather_ptx!`) at
   the top of the file expand 12 hand-written PTX strings sharing
   the same scaffolding — value load/store via raw `ld.global.uXX`
   / `st.global.uXX` instructions, index load via
   `ld.global.s32` / `ld.global.s64`, address math via
   `mul.lo.s64` + `add.s64`. Value width-shift `$wsh` (1/2/3) and
   index width-shift `$ish` (2/3) parameterise the byte-offset
   shifts. No dtype-specific arithmetic — just byte copies.

2. **Launch wrapper `fn launch_select<V: DeviceRepr, I: DeviceRepr>`**
   resolves the named PTX via `module_cache::get_or_compile`,
   allocates the output `CudaSlice<V>`, and launches with
   `BLOCK_SIZE = 256` and `grid = ceil(total / 256)` where
   `total = outer * out_dim * inner`. One thread per output
   element.

The `select_entry!` macro stamps each of the 20 compact-layout
`pub fn` entries by pinning the value type, index type, and
PTX-resolver function (`isel_ptx` or `gathr_ptx`). The
`gather_nd_entry!` macro stamps the 10 rank-aware `gather_nd_*`
entries. Resolver functions pick the correct PTX constant from the
(`ValWidth`, `IdxWidth`) cross-product table.

`fn isel_ptx(vw, iw) -> (&'static str, &'static str)` and
`fn gathr_ptx(vw, iw) -> ...` (lines 449-470) map each
(byte-width, index-width) pair to its `(PTX_CONST, "kernel_name")`
tuple.

Non-test production consumer: `backend_impl.rs` —
`fn gather_or_select` is the unified entry point on
`CudaBackendImpl`. It:

1. Matches on `index.dtype()` (must be I32 or I64) and `src.dtype()`
   (must be f32/f64/i32/i64/f16/bf16).
2. Expands the `run!` macro with the appropriate
   `gi::gather_*` / `gi::isel_*` symbols from this file.
3. Wraps the returned `CudaSlice<V>` back into a `GpuBufferHandle`
   tagged with the correct dtype.

ferrotorch-core's `index_select` / `gather` dispatch through the
`GpuBackend::gather_or_select` trait method (the boundary trait
method is what unifies both ops behind a single dispatch — the
dual nature of the two operations is captured by the `is_gather:
bool` parameter).

## Parity contract

`parity_ops = []` for this route. `index_select` / `gather` parity is
enforced at the ferrotorch-core layer; this file is the dtype-generic
GPU primitive layer.

Edge cases preserved:

- **Out-of-range index**: device UB, matches upstream CUDA. No host
  round-trip to validate (would defeat the no-CPU-detour contract).
- **f16 / bf16**: handled via the `u16` byte-width path — the kernel
  never inspects the value bits.
- **i32 vs i64 index**: separate kernel per width; the resolver picks
  the right one from the (`ValWidth`, `IdxWidth`) table.
- **Empty output** (`outer * out_dim * inner == 0`): grid is sized
  with `.max(1)` so the launch is well-formed, and the predicate
  `setp.ge.u32` ensures no thread writes.
- **gather vs index_select semantic difference**: `gather` reads
  `idx[t]` per-output-element (index is same shape as output);
  `index_select` reads `idx[i]` per row (index is 1-D along dim).
  The two PTX templates implement this difference at the
  index-load site.

## Verification

Unit tests in `ferrotorch-gpu/src/gather_int.rs` `mod tests` (7
tests): each covers a `(value_dtype, index_dtype)` combination with
a `cpu_to_gpu` upload, kernel call, and `gpu_to_cpu` verification
against a hand-computed expected output. Three tests specifically
cover rank-aware gather with smaller non-gather dimensions. Run on
hardware via the `GpuDevice::new(0)` graceful-skip pattern.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda gather_int:: 2>&1 | tail -3
```

Expected: ≥ 1 `test result: ok` line.

The fuller integration tests live at
`ferrotorch-gpu/tests/conformance_gpu_backend.rs` exercising
`gather_or_select` through the trait surface.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: 20 `pub fn isel_*`/`gather_*` entries in `ferrotorch-gpu/src/gather_int.rs` (`select_entry!` invocations at lines 492-653); non-test consumer: `CudaBackendImpl::gather_or_select` body at `ferrotorch-gpu/src/backend_impl.rs:442-568` dispatches all 20 cells through the `run!` macro. |
| REQ-2 | SHIPPED | impl: `index_select_ptx!` and `gather_ptx!` macros at `isel_ptx in gather_int.rs` expand 12 PTX entries (6 select × {W2,W4,W8} × {I32,I64} + 6 gather × ditto), resolved by `isel_ptx`/`gathr_ptx` at lines 449-470. |
| REQ-2b | SHIPPED | impl: `gather_nd_ptx!` expands six rank-aware PTX entries selected by `gather_nd_ptx_for`; tests `gather_nd_dim1_smaller_batch_f32_i64`, `gather_nd_dim0_smaller_column_f32_i64`, and `gather_nd_dim1_smaller_batch_i64_values` exercise the PyTorch-legal smaller non-axis layouts without a value-buffer host round trip. |
| REQ-3 | SHIPPED | impl: layout contract documented at `gather_int.rs` (the module `//!` block) and reflected in the PTX address math; verified by the unit tests' expected-output construction. |
| REQ-4 | SHIPPED | impl: out-of-range UB contract documented at `gather_int.rs`; the PTX templates omit any bounds check on the loaded index, matching upstream `at::native::index_select_cuda` in `aten/src/ATen/native/cuda/Indexing.cu`. |
| REQ-5 | SHIPPED | impl: `CudaBackendImpl::gather_or_select` at `gather_or_select in backend_impl.rs` is the production consumer; ferrotorch-core's `Tensor::index_select` / `Tensor::gather` dispatch through it via the `GpuBackend::gather_or_select` trait method when the source is CUDA-resident. |
| REQ-6 | SHIPPED | impl: `CudaBackendImpl::gather_intidx_nd` dispatches rank-aware gather through the `gather_nd_*` entries; production consumer: `Tensor::gather` and `IntTensor::gather` in `ops/phase2c.rs` call `GpuBackend::gather_intidx_nd` for CORE-112 smaller non-axis index shapes. |
