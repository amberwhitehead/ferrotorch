# Boolean / comparison CUDA kernels

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/
  - c10/cuda/
  - aten/src/ATen/cuda/
  - torch/cuda/
-->

## Summary

`ferrotorch-gpu/src/bool_kernels.rs` owns boolean / comparison GPU
compute kernels. Hand-written PTX owned by Rust (no CUDA C++, no
nvrtc, no external toolchain at load time), loaded via
`crate::module_cache::get_or_compile`. Boolean buffers are stored as
native `CudaSlice<u8>` (cudarc `DeviceRepr` for `u8`; a `bool` is one
byte holding 0 or 1). The `crate::backend_impl::CudaBackendImpl` handle
is tagged `DType::Bool` so a u8 bool buffer is never read as an i8/u8
integer. Mirrors PyTorch's `aten/src/ATen/native/cuda/CompareKernels.cu`,
`CompareEQKernel.cu`, and `BinaryLogicalOpsKernels.cu`.

## Requirements

- REQ-1: Comparison kernels — `gpu_cmp_f32`, `gpu_cmp_f64`,
  `gpu_cmp_i32`, `gpu_cmp_i64`, `gpu_cmp_bf16`, `gpu_cmp_f16` — each
  takes two value-dtype buffers and an op-name string
  (`"eq"|"ne"|"lt"|"le"|"gt"|"ge"`), producing a fresh
  `CudaSlice<u8>` of 0/1 bytes. The PTX is built per `(dtype, op)`
  combination at first call and cached.

- REQ-2: Logical binary kernels — `gpu_and_bool`, `gpu_or_bool`,
  `gpu_xor_bool` — each takes two u8 bool buffers, treats inputs as
  "nonzero == true", and produces canonical 0/1 output.

- REQ-3: Logical unary kernel — `gpu_not_bool` — produces `1` where
  input is `0`, else `0`.

- REQ-4: Global reduction kernels — `gpu_any_bool` (OR-reduce),
  `gpu_all_bool` (AND-reduce) — each produces a 1-element u8 buffer.
  Empty-input identity: `any` → `0`, `all` → `1` (matching
  `torch.any(empty)` / `torch.all(empty)`).

- REQ-5: NaN comparison semantics: `eq/lt/le/gt/ge` involving NaN are
  false; `ne` involving NaN is true. PTX `setp.{eq,lt,le,gt,ge}.f32`
  are unordered-false / `setp.ne.f32` is unordered-true, which is
  exactly the IEEE-754 / PyTorch behaviour.

- REQ-6: bf16/f16 comparison correctness — inputs (u16 bit patterns)
  are decoded to f32 first (mirroring `bf16.rs`/`f16.rs` decode
  pattern), then compared in f32. Result dtype is always `bool`
  (canonical u8 0/1), regardless of value dtype.

- REQ-7: Every `unsafe { ... launch(cfg)? }` block carries a SAFETY
  comment per R-CODE-1.

- REQ-8: `n == 0` is an early-return path (no launch); for reductions,
  the empty-identity (0 for `any`, 1 for `all`) is allocated and
  returned without launching.

- REQ-9: On-device bool broadcast — `gpu_broadcast_bool` expands a
  CUDA-resident bool mask by PyTorch right-aligned broadcasting without a
  host round trip.

- REQ-10: Signbit predicates — `gpu_signbit_*` produce bool masks for
  supported value dtypes, including half-precision payload handling.

- REQ-11: Integer broadcast comparison — `gpu_cmp_broadcast_i32` and
  `gpu_cmp_broadcast_i64` compare right-aligned broadcast-compatible integer
  operands on CUDA and produce a CUDA-resident bool buffer. This mirrors
  PyTorch TensorIterator comparison behavior for int32/int64 and must not
  materialize expanded integer operands on the host.

## Acceptance Criteria

- [x] AC-1: `gpu_cmp_{f32,f64,i32,i64,bf16,f16}` exist with the
  documented `(a, b, op_name, device)` → `CudaSlice<u8>` signature.
- [x] AC-2: `gpu_{and,or,xor}_bool` and `gpu_not_bool` exist.
- [x] AC-3: `gpu_any_bool` and `gpu_all_bool` exist and short-circuit
  to the empty-identity for `n == 0`.
- [x] AC-4: Every `unsafe { ... }` block has a SAFETY comment
  immediately above it.
- [x] AC-5: `cargo test -p ferrotorch-gpu --features cuda` exercises
  these kernels through the backend dispatch in
  `tests/conformance_gpu_kernels.rs`.
- [x] AC-6: Broadcasted CUDA integer comparisons are exercised through
  `ferrotorch-core/tests/audit_core090_int_compare_cuda_broadcast.rs` for
  i32/i64 matrix, scalar, high-rank, and zero-size outputs.

## Architecture

### Comparison kernels — PTX-string templating (REQ-1, REQ-5, REQ-6)

Rather than hand-write 36 near-identical comparison kernels (6 dtypes ×
6 operators), the file generates the PTX as owned `String`s at
module-load time (once per `(dtype, op)`, cached by
`get_or_compile_owned` keyed on the kernel name). The body differs
only in: the load type (`f32`/`f64`/`s32`/`s64`), the `setp` form
(`setp.{eq,ne,lt,le,gt,ge}.{f32,f64,s32,s64}`), and the input element
shift (`shl by 2` for f32/i32, `shl by 3` for f64/i64).

`fn cmp_ptx in bool_kernels.rs` builds the standard comparison PTX
prologue (idx, bound-check, offsets) and stamps in the op-specific
`reg_decl` and `setp` lines. `fn cmp_half_ptx in bool_kernels.rs` is
the half-precision variant: the two halves are loaded as `.b16` and
decoded to f32 via the standard bf16 splat (`mov.b32 %u, {%zero16,
%h}`) or f16 native conversion (`cvt.f32.f16`), then compared in f32.

NaN handling (REQ-5) is built into the PTX `setp` semantics:
unordered-quiet by default, so `setp.lt.f32` on NaN inputs sets the
predicate false. `setp.ne.f32` is the only exception — unordered-true
— matching IEEE-754. PyTorch's `aten/src/ATen/native/cuda/CompareKernels.cu`
relies on the same `<` / `==` C++ comparators which the device-side
compiler lowers to the same `setp.*` ordered semantics.

Public entry points `pub fn gpu_cmp_{f32,f64,i32,i64,bf16,f16} in
bool_kernels.rs` thin-wrap `fn launch_cmp` (or `launch_cmp_half`),
selecting the right `setp` form and decode pattern. Non-test consumer:
the comparison ops in `crate::backend_impl::CudaBackendImpl` dispatch
into these wrappers from the bool-result arms of `eq/ne/lt/le/gt/ge`
ops.

### Logical binary kernels (REQ-2)

`fn logic_bin_ptx in bool_kernels.rs` stamps the AND/OR/XOR PTX
template. Each kernel loads `a[i]` and `b[i]` as `.u8`, normalises
each to a predicate (`setp.ne.u16 %pa, %va, 0`), applies the
`and.pred`/`or.pred`/`xor.pred` operator, and writes a canonical 0/1
via `selp.u16 %res, 1, 0, %pr`. Public entry points:
`pub fn gpu_{and,or,xor}_bool in bool_kernels.rs`. Non-test consumer:
`crate::backend_impl::CudaBackendImpl` at
`CudaBackendImpl in backend_impl.rs` (`gpu_and_bool`), `gpu_and_bool in backend_impl.rs` (`gpu_or_bool`),
`gpu_xor_bool in backend_impl.rs` (`gpu_xor_bool`).

### Logical unary kernel (REQ-3)

`const NOT_BOOL_PTX in bool_kernels.rs` is a hand-rolled NOT kernel:
load `va`, `setp.eq.u16 %pa, %va, 0`, `selp.u16 %res, 1, 0, %pa`.
Public entry point: `pub fn gpu_not_bool in bool_kernels.rs`. Non-test
consumer: `gpu_not_bool in backend_impl.rs`.

### Reductions (REQ-4, REQ-8)

`const REDUCE_BOOL_PTX in bool_kernels.rs` is a single-thread serial
reduce. Thread 0 folds all `n` bytes; each input byte is normalised
to 0/1 (nonzero → 1); for `any` (op=0): OR-reduce starting from `a[0]`;
for `all` (op=1): AND-reduce. One launched thread keeps the result
exactly equal to a left-fold over the buffer (matching the CPU
reference bit-for-bit). The host guards `n == 0` before launching and
returns the empty-identity (0 for `any`, 1 for `all`) via a single
`clone_htod`. Public entry points: `pub fn gpu_any_bool` and
`pub fn gpu_all_bool in bool_kernels.rs`. Non-test consumer:
`gpu_all_bool in backend_impl.rs` (`gpu_any_bool`), `gpu_any_bool in backend_impl.rs` (`gpu_all_bool`).

### On-device bool broadcast (REQ-9, #1663)

`const BOOL_BROADCAST_PTX in bool_kernels.rs` is a u8 strided gather (8
dims unrolled, structurally identical to the f32
`crate::kernels::STRIDED_COPY_PTX` but with a 1-byte element, so the byte
offset equals the element index — no `shl.b64`). Thread `i` decodes its
output flat index into per-dim coordinates via the contiguous output
strides `os0..os7`, accumulates an input flat index `Σ_d coord_d *
src_stride[d]`, and copies `in[src] -> out[i]`. Broadcasting reduces to a
gather because the per-dim input element stride is the contiguous input
stride where `in_dim == out_dim`, or `0` where the input dim is size-1 or
absent (the NumPy / torch rule). `pad_bool_broadcast_params` computes the
contiguous output strides and pads unused trailing dims (`os = n+1` so the
coord is 0, `ss = 0` so the contribution is 0). Public entry point:
`pub fn gpu_broadcast_bool in bool_kernels.rs`. Non-test consumer:
`CudaBackendImpl::broadcast_bool in backend_impl.rs` (which derives the
per-dim broadcast strides from `(in_shape, out_shape)`), itself consumed
by `grad_fns::indexing::broadcast_bool_tensor`'s CUDA branch in
`ferrotorch-core`. The result stays a `DType::Bool` CUDA buffer — no host
round trip (R-CODE-4). This is the resident analog of the CPU
`broadcast_bool_tensor` walk and mirrors PyTorch's `expand_outplace(mask,
self)` at `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2406`. Before
#1663 the CUDA path returned `NotImplementedOnCuda`, so a `masked_scatter`
(or any masked op) with a CUDA mask whose shape required broadcasting
diverged from torch.

### Integer broadcast comparison (REQ-11)

`fn cmp_broadcast_ptx in bool_kernels.rs` builds a rank-general comparison
kernel for i32/i64 operands. Each thread decodes its flat output index through
the output shape metadata, accumulates one source flat index per operand using
right-aligned broadcast strides, loads the two integer values, applies the
same signed `setp.{eq,ne,lt,le,gt,ge}.s{32,64}` predicate forms as the
same-shape comparison kernels, and stores one canonical u8 bool byte.

The Rust launcher uploads only compact shape/stride metadata (`u32` vectors)
and never copies operand values to the host. It validates exact logical
operand lengths, metadata rank agreement, `u32` narrowing for `n`, ndim,
shape, and strides, and returns an empty bool buffer without launching when
the broadcasted output numel is zero. Public entry points:
`pub fn gpu_cmp_broadcast_i32` and `pub fn gpu_cmp_broadcast_i64 in
bool_kernels.rs`. Non-test consumer: `CudaBackendImpl::compare_broadcast`,
which is called by `BoolTensor::compare_int` whenever CUDA i32/i64 operands
have broadcast-compatible but different shapes.

### Logical-length launch contract (#1660)

`launch_cmp` (and its forwarders `launch_cmp_half` / the `gpu_cmp_*`
wrappers) take an explicit LOGICAL element count `n: usize` — the
operands' `CudaBuffer::len()` — and validate/launch on that, NOT on the
raw `CudaSlice::len()`. The raw slice may be OVER-ALLOCATED past `n`: a
`.contiguous()`-materialised view (e.g. a row-narrowed CUDA view packed
on-device for the #1658 storage-offset class) is backed by a POOLED
buffer whose raw len is rounded up to a multiple of `ROUND_ELEMENTS = 256`
(see `pool.md` REQ-2), while a `clone_htod` operand is exact-length. The
kernel-level check is therefore a backing-store sufficiency guard
(`a.len() >= n && b.len() >= n`), and the launch reads/writes only
`[0, n)`. Comparing raw lens would spuriously reject pairings such as
`256 vs 6`. The dispatch site (`CudaBackendImpl::compare in
backend_impl.rs`) owns the operand-shape equality check on the logical
`GpuBufferHandle::len()` and threads `n` down. `launch_logic_bin`
consumes only exact-length compare-result bool buffers, so it keeps the
strict raw-len equality guard (logical == raw there).

### SAFETY discipline (REQ-7)

`fn launch_cmp / launch_cmp_broadcast / launch_cmp_half / launch_logic_bin /
launch_not / launch_reduce_bool in bool_kernels.rs` each wrap a single `unsafe {
stream.launch_builder(&f)...launch(cfg)? }` block. Every such block is
preceded by a multi-line SAFETY comment naming: (a) the PTX entry's
parameter signature matching argument push order, (b) the input buffer
backing AT LEAST `n` elements (`*.len() >= n`, tolerating a pooled
over-allocated `.contiguous()` materialisation per #1660), (c) the fresh
`out` allocation, (d) the PTX bound check confining access to `[0, n)`,
(e) the `n as u32` non-truncation. R-CODE-1 grandfathers raw CUDA kernel
launches.

## Parity contract

`parity_ops = []` for this route (S5: INFRASTRUCTURE for the bool
result type). Per-op parity for `eq/ne/lt/le/gt/ge/logical_and/
logical_or/logical_xor/logical_not/any/all` lives in the
ferrotorch-core op crate; the bool kernel file is the dtype-specific
backend that gets dispatched when the result must be `DType::Bool`.

Edge cases preserved:

- **NaN compare**: ordered-quiet `setp` makes `eq/lt/le/gt/ge` false on
  NaN; `setp.ne` is unordered-true → `ne` is true on NaN. Matches
  IEEE-754 and PyTorch's `cmp_kernel_cuda` in
  `aten/src/ATen/native/cuda/CompareKernels.cu`.
- **Nonzero treated as true**: `any`/`all`/`and`/`or`/`xor`/`not`
  normalise via `setp.ne %pa, %va, 0` so any nonzero u8 byte is true.
  Output is always canonical 0/1. Matches PyTorch's
  `BinaryLogicalOpsKernels.cu` which sees `bool` as 0/1 after
  `c10::convert<bool>` normalisation.
- **Empty input**: `cmp` of two empty buffers returns an empty u8
  buffer (no launch). `any(empty) → 0`, `all(empty) → 1` — the host
  short-circuits the launch and returns a 1-element u8 buffer holding
  the empty-identity via `clone_htod`. Matches PyTorch's `torch.any(
  torch.empty(0, dtype=torch.bool))` / `torch.all(...)`.
- **Broadcasted integer comparison**: i32/i64 operands follow PyTorch
  right-aligned broadcasting, including scalar operands, missing leading
  dimensions, broadcast axes with stride zero, and zero-sized outputs. Values
  and bool output stay on CUDA.

## Verification

Integration tests in `ferrotorch-gpu/tests/conformance_gpu_kernels.rs`
exercise the bool path through the backend's comparison and logical
op dispatchers. The `backend_impl::tests` and `conformance_gpu_backend.rs`
suites further exercise the empty-input and NaN-compare edge cases.
Broadcasted integer comparison is pinned through the core-facing CUDA audit
`ferrotorch-core/tests/audit_core090_int_compare_cuda_broadcast.rs`, because
the production call chain starts at `BoolTensor::compare_int` and reaches
`CudaBackendImpl::compare_broadcast`.

Smoke command:

```bash
cargo test -p ferrotorch-gpu --features cuda --lib bool_kernels 2>&1 | tail -3
```

Expected: `test result: ok` on a host with a CUDA device. The
`#![cfg(feature = "cuda")]` gate excludes the module on no-CUDA builds.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn gpu_cmp_{f32,f64,i32,i64,bf16,f16} in bool_kernels.rs` (six wrappers around `fn launch_cmp` / `launch_cmp_half`, each templating the PTX per (dtype, op)). Non-test consumer: the bool-result arms of the comparison ops in `ferrotorch-gpu/src/backend_impl.rs` dispatch into these wrappers from `eq/ne/lt/le/gt/ge` op handlers (the same backend that calls `gpu_and_bool` at line 6545). |
| REQ-2 | SHIPPED | impl: `pub fn gpu_{and,or,xor}_bool in bool_kernels.rs` (each thin-wraps `launch_logic_bin` with the matching PTX). Non-test consumer: `gpu_and_bool in backend_impl.rs` (`gpu_and_bool`), `gpu_and_bool in backend_impl.rs` (`gpu_or_bool`), `gpu_or_bool in backend_impl.rs` (`gpu_xor_bool`). |
| REQ-3 | SHIPPED | impl: `pub fn gpu_not_bool in bool_kernels.rs`. Non-test consumer: `gpu_not_bool in backend_impl.rs`. |
| REQ-4 | SHIPPED | impl: `pub fn gpu_any_bool / gpu_all_bool in bool_kernels.rs`. Non-test consumer: `gpu_any_bool in backend_impl.rs` (`gpu_any_bool`), `gpu_any_bool in backend_impl.rs` (`gpu_all_bool`). |
| REQ-5 | SHIPPED | impl: comparison kernels use PTX `setp.{eq,lt,le,gt,ge}.f32` (unordered-false on NaN) and `setp.ne.f32` (unordered-true on NaN) per the module doc-comment "NaN comparisons follow IEEE: `eq/lt/le/gt/ge` involving NaN are false, `ne` involving NaN is true". Non-test consumer: the bool-comparison ops in `backend_impl.rs` rely on this for IEEE-NaN parity. |
| REQ-6 | SHIPPED | impl: `fn cmp_half_ptx in bool_kernels.rs` decodes bf16 via `mov.b32 %ua, {%zero16, %ha}` (BF16_DECODE constant) and f16 via `cvt.f32.f16 %fa, %ha` (F16_DECODE), then `setp.{op}.f32`. Result is always u8 0/1 (`selp.u16 %res, 1, 0, %c`). Non-test consumer: `pub fn gpu_cmp_{bf16,f16}` invoke `launch_cmp_half` with the right decode, called from the bool-comparison arms of the backend. |
| REQ-7 | SHIPPED | impl: every `unsafe { stream.launch_builder(&f)... }` in `bool_kernels.rs` (in `launch_cmp`, `launch_not`, `launch_reduce_bool`) is preceded by a multi-line SAFETY comment naming entry signature, length binding, alloc, bound check, and `n as u32` non-truncation. Non-test consumer inherits the contract via each public wrapper. |
| REQ-8 | SHIPPED | impl: `launch_cmp` and `launch_not` short-circuit on `n == 0` via `if n == 0 { return Ok(stream.alloc_zeros::<u8>(0)?); }`; `launch_reduce_bool` short-circuits with `let host = [empty_identity]; return Ok(stream.clone_htod(&host)?);`. Non-test consumer relies on the no-launch short circuit via backend dispatch (e.g. `torch.any(empty)` returning a 1-element false). |
| REQ-9 (on-device bool broadcast, #1663) | SHIPPED | impl: `pub fn gpu_broadcast_bool in ferrotorch-gpu/src/bool_kernels.rs` (u8 strided-gather over `BOOL_BROADCAST_PTX`, 8-dim unrolled, `pad_bool_broadcast_params` computes the contiguous output strides + pads unused dims). Non-test consumer: `CudaBackendImpl::broadcast_bool in ferrotorch-gpu/src/backend_impl.rs` computes the per-dim broadcast input strides and invokes `gpu_broadcast_bool`; that backend slot is itself consumed by `grad_fns::indexing::broadcast_bool_tensor`'s CUDA branch in `ferrotorch-core/src/grad_fns/indexing.rs`, which `masked_scatter` / `masked_fill_bcast` / `masked_select_bcast` / `where_cond_bcast` all flow through. Mirrors PyTorch `expand_outplace(mask, self)` at `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2406`. |
| REQ-10 (`signbit`) | SHIPPED | impl: `pub fn gpu_signbit_f32 / gpu_signbit_f64 / gpu_signbit_f16 / gpu_signbit_bf16 in bool_kernels.rs`; non-test consumer: `CudaBackendImpl::signbit_mask` and the CUDA branch of `ferrotorch_core::grad_fns::transcendental::signbit`. |
| REQ-11 (integer broadcast comparison) | SHIPPED | impl: `pub fn gpu_cmp_broadcast_i32 / gpu_cmp_broadcast_i64 in bool_kernels.rs` with rank-general output-shape and per-input broadcast-stride metadata. Non-test consumer: `CudaBackendImpl::compare_broadcast`, reached from `BoolTensor::compare_int` for CUDA broadcast-compatible i32/i64 operands whose shapes differ. Runtime proof: `audit_core090_int_compare_cuda_broadcast` exercises i32/i64 matrix, scalar, high-rank, and empty broadcast cases against PyTorch-observed bool results. |
