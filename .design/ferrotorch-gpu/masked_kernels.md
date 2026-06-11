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
- REQ-4: `masked_scatter` two on-device families:
  - BACKWARD (VJP of `masked_select`): scatter the compacted gradient
    into a **zeroed** buffer at the true positions. Three
    width-templated entries `masked_scatter_{32, 64, 16}` matching the
    `where` family.
  - FORWARD (#1662): `out[i] = mask[i] ? source[j++] : input[i]` —
    copies source values (consumed serially in flat order) into a clone
    of `input` at true positions, keeping `input[i]` elsewhere. The
    source-index `j` is the EXCLUSIVE PREFIX-SUM of the mask, matching
    upstream `aten/src/ATen/native/cuda/IndexKernel.cu:416-453`
    (`at::cuda::cub::mask_exclusive_sum` + `source[maskPrefixSum]`
    gather); a single in-order thread realises the same offset without a
    separate scan buffer. Entries `masked_scatter_forward_{32, 64}`
    (f32/f64 — the dtypes torch's all-CUDA masked_scatter forward
    exercises). Distinct from the backward family in the FALSE branch
    (forward passes `input[i]` through; backward leaves the pre-zeroed
    slot).
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
- REQ-8: Predicate-mask entry points for the masked-tensor
  constructors (#1545 / #1534): `pub fn isfinite_mask_{f32,f64}`
  (`out[i] = (v==v) && (|v| != +inf)`, PyTorch parity with
  `aten/src/ATen/native/TensorCompare.cpp:484`) and
  `pub fn ne_scalar_mask_{f32,f64}` (`out[i] = (v != value)` via the
  UNORDERED `setp.neu`, so `NaN != value` is true matching the CPU
  walk). Each reads a value `CudaSlice<T>` and returns a fresh
  `CudaSlice<u8>` 0/1 mask. These let `masked_invalid` / `masked_equal`
  compute their predicate on-device instead of downloading the data
  tensor to host.

## Acceptance Criteria

- [x] AC-1: Six `pub fn masked_fill_*` entries exist at lines
  731-833.
- [x] AC-2: Three `pub fn where_{32,64,16}` entries exist at lines
  833-877.
- [x] AC-3: One `pub fn count_true` at line 570 and three
  `pub fn masked_select_*` entries at lines 877-936.
- [x] AC-4: Three backward `pub fn masked_scatter_{32,64,16}` entries
  plus two forward `pub fn masked_scatter_forward_{32,64}` entries
  (#1662) in `masked_kernels.rs`.
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
4. **`masked_scatter` distribution**:
   - BACKWARD (inverse of `masked_select`): a serial source-cursor
     walks the compacted gradient and writes into the destination
     position when `mask[i] != 0`, leaving a pre-zeroed buffer
     elsewhere.
   - FORWARD (#1662): the same serial source-cursor walk, but the
     destination starts as a copy of `input` — `out[i] = mask[i] ?
     source[j++] : input[i]`. The `out` buffer is fully written (no
     pre-zero needed). Wired by `grad_fns::indexing::masked_scatter`'s
     all-CUDA branch so `Tensor::masked_scatter_t` with input + mask +
     source all on CUDA stays GPU-resident (NO host round trip,
     R-CODE-4) — previously the forward called `mask_b.data()` which
     errors `GpuTensorNotAccessible` on a CUDA bool mask.

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
- **Logical-length launch for `where` (#1660) and `masked_fill`
  (#1661)**: `where_32` / `where_64` / `where_16` (and `launch_where`),
  and the six `masked_fill_*` entries (and `launch_masked_fill`), take an
  explicit LOGICAL element count `n: usize` and validate/launch on it,
  NOT on the raw `CudaSlice::len()`. A `.contiguous()`-materialised
  operand (a packed row-narrowed CUDA view, #1658 storage-offset class) is
  backed by a POOLED buffer rounded up to a multiple of
  `ROUND_ELEMENTS = 256` (`pool.md` REQ-2), while a `clone_htod` operand
  is exact-length; the kernel guard is therefore `cond/x/y.len() >= n`
  (`where`) / `input/mask.len() >= n` (`masked_fill`) and the launch
  reads/writes only `[0, n)`. The dispatch sites
  (`CudaBackendImpl::where_cond` / `masked_fill_dt` in `backend_impl.rs`)
  own the logical operand-shape equality check and thread `n` down.
  (`launch_scatter` still validates raw lens — see bug note in REQ-2.)

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
| REQ-1 | SHIPPED | impl: six `pub fn masked_fill_*` at `masked_fill_ in ferrotorch-gpu/src/masked_kernels.rs`; non-test consumer: `backend_impl.rs` (`use crate::masked_kernels as mk`) dispatches per-dtype calls. `launch_masked_fill` validates+launches on the LOGICAL `n` (#1661), tolerating pooled over-allocated `.contiguous()` inputs (regression test `masked_fill_narrowed_offset_view_gpu_matches_torch` + `masked_fill_normal_offset0_above_round_elements_gpu_matches_torch` in `tests/divergence_masked_fill_storage_offset_gpu.rs`); core-layer `.contiguous()` normalisation in `ferrotorch_core::grad_fns::indexing::masked_fill` + `masked_fill_bt`. |
| REQ-2 | SHIPPED | impl: `pub fn where_32`/`where_64`/`where_16` at `where_32 in masked_kernels.rs`; non-test consumer: `where_16 in backend_impl.rs`. `launch_where` validates+launches on the LOGICAL `n` (#1660), tolerating pooled over-allocated `.contiguous()` operands (regression test `compare_gt_both_narrowed_views_pooled_logical_len_gpu_matches_torch` / `where_cond_bt_narrowed_offset_view_gpu_matches_torch` in `tests/divergence_storage_offset_class_completeness.rs`). KNOWN LATENT (spillover, not fixed here): `launch_scatter` (`mask.len() != out_numel`) still compares RAW lens — same class, but its core-layer `.contiguous()` normalisation + tests need a separate dispatch. (`launch_masked_fill` was fixed in #1661, see REQ-1.) |
| REQ-3 | SHIPPED | impl: `pub fn count_true` at `count_true in masked_kernels.rs`; `masked_select_32/64/16` at lines 877-936; non-test consumer: `backend_impl.rs`. |
| REQ-4 | SHIPPED | BACKWARD impl: `masked_scatter_32/64/16 in ferrotorch-gpu/src/masked_kernels.rs`; non-test consumer: `CudaBackendImpl::masked_scatter in ferrotorch-gpu/src/backend_impl.rs:8300` (the VJP of `masked_select`, used by `MaskedSelectBackward::backward in ferrotorch-core/src/grad_fns/indexing.rs:1111`). FORWARD impl (#1662): `masked_scatter_forward_32`/`masked_scatter_forward_64 in ferrotorch-gpu/src/masked_kernels.rs` (`out[i] = mask[i] ? source[j++] : input[i]`, exclusive-prefix-sum source index per `aten/src/ATen/native/cuda/IndexKernel.cu:416-453`); non-test consumer: `CudaBackendImpl::masked_scatter_forward in ferrotorch-gpu/src/backend_impl.rs:8385` dispatched from `ferrotorch_core::grad_fns::indexing::masked_scatter`'s all-CUDA branch (`ferrotorch-core/src/grad_fns/indexing.rs:3734`), so `Tensor::masked_scatter_t` with input+mask+source all on CUDA keeps the result `is_cuda()` (NO host round trip, R-CODE-4) instead of erroring `GpuTensorNotAccessible` on the resident bool mask. Tests: `masked_scatter_forward_gpu_mask_rejected_divergence` (pinned, #1662) + `masked_scatter_forward_all_cuda_patterns_f32/f64_matches_torch` + `masked_scatter_forward_all_cuda_backward_matches_torch in ferrotorch-gpu/tests/divergence_masked_fill_reaudit.rs`; unit tests `masked_scatter_forward_32_keeps_input_where_false` / `masked_scatter_forward_64_all_false_and_all_true` in this file's `mod tests`. |
| REQ-5 | SHIPPED | impl: every kernel launch in this file routes through `module_cache::get_or_compile`. The file's `use crate::module_cache::get_or_compile` import at line 44 binds the single PTX load path; the file has no `cudarc::nvrtc` import. |
| REQ-6 | SHIPPED | impl: per-dtype `pub fn` entries mean the (op, dtype) coverage is structurally surfaced — a missing combination is a missing function symbol that the `backend_impl` dispatcher converts to `FerrotorchError::NotImplementedOnCuda` (the policy documented in the module `//!` block at line 36). |
| REQ-7 | SHIPPED | impl: four `use crate::masked_kernels as mk` sites in `backend_impl.rs` at lines 6660, 6735, 6834, 6903 — each is the body of a `CudaBackendImpl` trait method. ferrotorch-core dispatches `Tensor::masked_fill`/etc. through the `GpuBackend` trait when the input is CUDA-resident. |
| REQ-8 | SHIPPED | impl: `pub fn isfinite_mask_f32`/`isfinite_mask_f64`/`ne_scalar_mask_f32`/`ne_scalar_mask_f64 in masked_kernels.rs`; non-test consumer: `CudaBackendImpl::isfinite_mask`/`ne_scalar_mask in backend_impl.rs` dispatch to them per-dtype, which `ferrotorch_core::masked_invalid`/`masked_equal` (`masked.rs`) invoke on f32/f64 CUDA inputs. Live-CUDA tests `isfinite_mask_f32_matches_ieee`/`isfinite_mask_f64_matches_ieee`/`ne_scalar_mask_f32_marks_unequal`/`ne_scalar_mask_f64_nan_is_unequal` in this file's `#[cfg(test)] mod tests`. |
