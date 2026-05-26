# GPU argmax / argmin kernels (multi-dtype, i64 indices)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/ReduceArgMaxKernel.cu
  - aten/src/ATen/native/cuda/ReduceArgMinKernel.cu
-->

## Summary

`ferrotorch-gpu/src/reduce_arg.rs` implements GPU `argmax` / `argmin`
across six value dtypes (f32, f64, i32, i64, f16, bf16) sharing a
single launch path and PTX template per dtype, all returning
GPU-resident `CudaSlice<i64>` indices (PyTorch int64 parity). Tie-break
is the first occurrence (strict `>` / `<`), one thread per output
slice. Mirrors `argmax_kernel_cuda` and `argmin_kernel_cuda` in
`aten/src/ATen/native/cuda/ReduceArgMaxKernel.cu` and
`ReduceArgMinKernel.cu`.

## Requirements

- REQ-1: Twelve public entry points — `gpu_argmax_{f32,f64,i32,i64,f16,bf16}`
  and `gpu_argmin_{f32,f64,i32,i64,f16,bf16}` — each taking
  `(input, outer, dim_size, inner, device)` and returning
  `GpuResult<CudaSlice<i64>>` of length `outer * inner`.
- REQ-2: Six dtype-specialised PTX templates
  (`ARGREDUCE_{F32,F64,I32,I64,F16,BF16}_PTX`) sharing a 7-arg ABI
  `(in_ptr, out_ptr, outer, dim_size, inner, total, op)` with the
  `op` selector at `ARG_MAX=0` / `ARG_MIN=1`. Float dtypes use
  `setp.gt.f{32,64}`; integer dtypes use `setp.gt.s{32,64}`; f16
  / bf16 decode each 16-bit element to f32 and compare in f32.
- REQ-3: First-occurrence tie-break via STRICT comparison
  (`>` for argmax, `<` for argmin): the accumulator only updates when
  a later element is *strictly* better. Equal values never displace
  the earlier index. Matches PyTorch's documented contract.
- REQ-4: Documented NaN divergence: strict-compare argmax skips NaN
  (a NaN is never `> acc`), so a slice of all-NaN reports index 0.
  Upstream CUDA argmax returns the index of any NaN it encounters.
  Documented in the module `//!` doc-comment (lines 40-47), not
  silently hidden.
- REQ-5: Non-test production consumer at
  `ferrotorch-gpu/src/backend_impl.rs:6186-6285` — the
  `CudaBackendImpl::argmax_f32`/`argmin_f32`/... trait methods
  dispatch on `DType` and call the per-dtype entry from this file.

## Acceptance Criteria

- [x] AC-1: Twelve `pub fn gpu_argmax_*` / `pub fn gpu_argmin_*`
  symbols exist (8 are macro-stamped via `arg_entry!`, 4 are
  hand-written for the f16/bf16 `u16`-bitpattern cases).
- [x] AC-2: Six PTX constants `ARGREDUCE_*_PTX` are defined and
  loaded via `crate::module_cache::get_or_compile`.
- [x] AC-3: First-occurrence tie-break verified by
  `argmax_f32_tie_first_index` unit test.
- [x] AC-4: NaN divergence documented in the module `//!` block.
- [x] AC-5: All 12 dtype × {max,min} combinations are dispatched from
  `CudaBackendImpl` argmax/argmin entry points in `backend_impl.rs`.

## Architecture

`reduce_arg.rs` organises around a single launcher
`fn launch_argreduce<V: DeviceRepr>` that:

1. Validates the value-buffer length against `outer * dim_size * inner`.
2. Resolves the PTX function via `crate::module_cache::get_or_compile`.
3. Allocates the i64 output via `device.stream().alloc_zeros::<i64>(outer * inner)`.
4. Launches with the standard `BLOCK_SIZE = 256` 1-D grid sized for
   `total = outer * inner` threads. Each thread handles one output
   slice serially scanning `dim_size` elements at stride `inner`.

The macro `arg_entry!` stamps `gpu_argmax_f32` / `gpu_argmin_f32` /
`gpu_argmax_f64` / `gpu_argmin_f64` / `gpu_argmax_i32` / `gpu_argmin_i32`
/ `gpu_argmax_i64` / `gpu_argmin_i64` (8 entries). The four
`u16`-bitpattern entries `gpu_argmax_f16`/`gpu_argmin_f16`/
`gpu_argmax_bf16`/`gpu_argmin_bf16` are hand-written because their
input type is `CudaSlice<u16>` (the half-precision storage type), not
the parameterised `$ty`.

The PTX kernels share a serial-per-slice template. Each thread:

1. Decomposes its global tid into `(outer_idx, inner_idx)`.
2. Seeds `acc = in[base]` and `best_j = 0`.
3. Loops `j in 1..dim_size`, reading `v = in[base + j*inner]`,
   updating `acc, best_j` when `(op == ARG_MAX && v > acc) ||
   (op == ARG_MIN && v < acc)`.
4. Writes `out[gtid] = best_j as i64`.

For f16 / bf16 the comparison value is decoded to f32 in-register
(`cvt.f32.f16` after a bf16 hi-half splat, or an f16 widening cvt);
the index update logic is identical.

Non-test production consumer: `backend_impl.rs:6186-6285` —
`CudaBackendImpl::argmax_f32`/`argmax_f64`/`argmax_f16`/`argmax_bf16`/
`argmax_i32`/`argmax_i64` and the matching `argmin_*` methods each
dispatch on the input handle's dtype tag and forward to the
corresponding `gpu_argmax_*` / `gpu_argmin_*` function in this file.
ferrotorch-core's `Tensor::argmax` / `Tensor::argmin` dispatch
through the trait when the tensor is CUDA-resident.

## Parity contract

`parity_ops = []` for this route — the `argmax` / `argmin` parity is
enforced at the ferrotorch-core op layer. The kernels here pair with
the CPU reductions to provide matching results within first-occurrence
tie-break semantics.

Edge cases preserved:

- **First-occurrence tie**: strict `>` / `<` keeps the lowest index
  among equal-best values. Matches PyTorch.
- **NaN**: strict-compare-skip — DOCUMENTED divergence from PyTorch
  CUDA argmax. ferrotorch's other float reductions take the same
  pragmatic stance.
- **Empty slice** (`dim_size == 0`): not exercised at the entry level
  — the caller is expected to short-circuit before invoking
  (PyTorch upstream raises in this case too).
- **Global flatten**: `outer = 1, dim_size = numel, inner = 1` →
  single thread folds the entire buffer; documented in the module
  block.
- **Along-dim**: `outer = product(shape[..dim])`,
  `dim_size = shape[dim]`, `inner = product(shape[dim+1..])`.
- **f16 / bf16**: decoded to f32 in-register for comparison; the
  underlying value bit pattern is preserved through `u16` storage.

## Verification

Unit tests in `ferrotorch-gpu/src/reduce_arg.rs` `mod tests` (6 tests):

- `argmax_argmin_f32_global` — round-trip on a known fixture.
- `argmax_f32_tie_first_index` — pins first-occurrence tie-break.
- (4 additional tests covering along-dim, integer dtypes, and
  half-precision dtypes.)

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda reduce_arg:: 2>&1 | tail -3
```

Expected: ≥ 1 `test result: ok` line.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: 12 `pub fn gpu_arg{max,min}_*` symbols in `ferrotorch-gpu/src/reduce_arg.rs` (macro-stamped at lines 686-741, hand-written at lines 744-818); non-test consumer: `ferrotorch-gpu/src/backend_impl.rs:6186-6285` dispatches all 12 dtype × {max,min} combinations through `match dtype` arms. |
| REQ-2 | SHIPPED | impl: six `ARGREDUCE_*_PTX` constants in `reduce_arg.rs` (search for `ARGREDUCE_` to locate; first is `ARGREDUCE_F32_PTX` near line 85) carry the documented 7-arg ABI; loaded via `module_cache::get_or_compile` in `fn launch_argreduce`. |
| REQ-3 | SHIPPED | impl: strict `setp.gt.f32` / `setp.lt.f32` (and dtype counterparts) in the PTX templates; verified by `argmax_f32_tie_first_index` unit test which constructs `[5.0, 1.0, 2.0, 5.0]` and asserts the result is index 0. |
| REQ-4 | SHIPPED | impl: NaN divergence documented at `reduce_arg.rs:40-47` (the module `//!` block). The strict-compare semantics in the PTX naturally produce this behaviour. |
| REQ-5 | SHIPPED | impl: `CudaBackendImpl::argmax_f32` body at `backend_impl.rs:6186` dispatches `match dtype { F32 => gpu_argmax_f32, F64 => gpu_argmax_f64, F16 => gpu_argmax_f16, BF16 => gpu_argmax_bf16, I32 => gpu_argmax_i32, I64 => gpu_argmax_i64 }`; mirror block at line 6249 for argmin. ferrotorch-core dispatches through the `GpuBackend` trait. |
