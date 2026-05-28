# Integer (i32 / i64) CUDA kernels

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

`ferrotorch-gpu/src/int_kernels.rs` owns the i32 / i64 GPU compute
kernels. Hand-written PTX owned by Rust (no CUDA C++, no nvrtc, no
external toolchain at load time), loaded via
`crate::module_cache::get_or_compile`. Unlike the half-precision
modules, integer buffers are NATIVE `CudaSlice<i32>` / `CudaSlice<i64>`
(cudarc `DeviceRepr`, no u16 bit-pattern transmute): the on-device
element type matches the logical element type. The
`crate::backend_impl::CudaBackendImpl` handle is tagged `DType::I32` /
`DType::I64` to disambiguate from f32. Mirrors PyTorch's
`aten/src/ATen/native/cuda/BinaryMulKernel.cu`,
`BinaryDivFloorKernel.cu`, `BinaryRemainderKernel.cu`,
`BinaryBitwiseOpsKernels.cu`, `BinaryShiftOpsKernels.cu`, and the
integer arms of the reduction kernels.

## Requirements

- REQ-1: Elementwise binary arithmetic (i32 + i64) — `gpu_add_i32/i64`,
  `gpu_sub_i32/i64`, `gpu_mul_i32/i64` — wrapping on overflow (PyTorch
  does NOT upcast integer arithmetic by default; matches C++ signed
  overflow on the CUDA path).

- REQ-2: Integer division semantics (i32 + i64) — `gpu_floor_div_i32/i64`,
  `gpu_remainder_i32/i64`. `floor_divide` floors toward −∞ (NOT C
  truncation); `remainder` takes the sign of the divisor (Python /
  `torch.remainder`). Both correct via post-trunc adjustment when
  remainder is nonzero AND operand signs differ.

- REQ-3: Bitwise binary (i32 + i64) — `gpu_bitand_i32/i64`,
  `gpu_bitor_i32/i64`, `gpu_bitxor_i32/i64` — direct PTX
  `and.b{32,64}` / `or.b{32,64}` / `xor.b{32,64}`.

- REQ-4: Shift operators (i32 + i64) — `gpu_shl_i32/i64` (logical
  left shift) and `gpu_shr_i32/i64` (ARITHMETIC sign-extending right
  shift via PTX `shr.s{32,64}`, matching PyTorch `__rshift__` on
  signed dtypes).

- REQ-5: Elementwise unary (i32 + i64) — `gpu_neg_i32/i64` (negate
  via `sub.s* 0, %va`) and `gpu_bitnot_i32/i64` (PTX `not.b{32,64}`).

- REQ-6: Reductions (i32 + i64) — `gpu_sum_i32/i64`, `gpu_prod_i32/i64`,
  `gpu_min_i32/i64`, `gpu_max_i32/i64` — accumulate in the SAME
  integer width (wrapping on overflow; PyTorch does NOT upcast integer
  `sum` by default). Single-thread serial fold for exact left-fold
  semantics matching the CPU reference bit-for-bit.

- REQ-7: Integer division / remainder by zero is NOT trapped: PTX
  `div.s` / `rem.s` by zero returns an implementation-defined value
  (PyTorch on CUDA likewise does not trap). No host round-trip to
  special-case.

- REQ-8: Every `unsafe { ... launch(cfg)? }` block carries a SAFETY
  comment per R-CODE-1.

- REQ-9: `n == 0` is an early-return path (no launch) for elementwise
  ops; for reductions, the empty-identity (`0` for sum, `1` for prod,
  `T::MAX` for min, `T::MIN` for max) is allocated and returned via
  `clone_htod` without launching.

## Acceptance Criteria

- [x] AC-1: All six elementwise binary arithmetic functions
  (`add/sub/mul × i32/i64`) exist with `(a, b, device)` →
  `CudaSlice<T>` signature.
- [x] AC-2: `floor_div_{i32,i64}` and `remainder_{i32,i64}` exist with
  the documented post-trunc adjustment.
- [x] AC-3: All six bitwise binary + four shift functions exist.
- [x] AC-4: `neg_{i32,i64}` and `bitnot_{i32,i64}` exist.
- [x] AC-5: `sum/prod/min/max × i32/i64` exist as
  one-launched-thread serial reductions.
- [x] AC-6: Every `unsafe { ... }` block has a SAFETY comment
  immediately above it.
- [x] AC-7: `cargo test -p ferrotorch-gpu --features cuda` exercises
  these kernels through the backend dispatch in
  `tests/conformance_gpu_kernels.rs` (the integer-arith conformance
  arm).

## Architecture

### Elementwise binary arithmetic (REQ-1)

`pub fn gpu_add_i32 / gpu_sub_i32 / gpu_mul_i32 in int_kernels.rs`
each thin-wrap `fn launch_binary` with the matching PTX. Each thread
loads `a[i]`, `b[i]` (one 32-bit signed int each), computes
`add.s32 / sub.s32 / mul.lo.s32` (the `.lo` form takes the low 32 bits
of the 64-bit product, i.e. wrapping multiplication), stores `out[i]`.
`off = i << 2` (4 bytes per i32). i64 mirrors with `shl by 3` and
`.s64` / `mul.lo.s64`. Non-test production consumer:
`crate::backend_impl::CudaBackendImpl`'s integer dtype arm at
`backend_impl.rs` (i32 add), `backend_impl.rs` (i64 add), `backend_impl.rs`
(sub), `backend_impl.rs` (mul).

### Integer division semantics (REQ-2)

`fn gpu_floor_div_i32 / gpu_floor_div_i64 in int_kernels.rs` compute
the truncated quotient via PTX `div.s32 / div.s64` (toward zero), then
floor-correct: subtract 1 from `q` when the remainder `r != 0` AND
`(r < 0) != (b < 0)` (i.e. the operand signs differ). This matches
`torch.floor_divide` exactly (the same formula as
`a - (a / b) * b` adjusted to round to −∞).

`fn gpu_remainder_i32 / gpu_remainder_i64 in int_kernels.rs` compute
the truncating remainder `r = rem.s(a, b)` (sign of dividend), then
add `b` when `r != 0` AND signs of `r` and `b` differ. The result has
the sign of the divisor (Python / `torch.remainder`), which is exactly
`a - floor_divide(a, b) * b`. Mirrors PyTorch's
`aten/src/ATen/native/cuda/BinaryRemainderKernel.cu` (`fmod`-then-
adjust pattern for floats; the integer arm in PyTorch is the same
shape, in `c10/util/safe_numerics.h`'s integer remainder helpers).

Non-test production consumer: `backend_impl.rs` (floor_div),
`backend_impl.rs` (remainder).

### Bitwise binary (REQ-3)

`pub fn gpu_bitand_i32/i64 / gpu_bitor_i32/i64 / gpu_bitxor_i32/i64 in
int_kernels.rs` each thin-wrap `launch_binary` with the matching PTX
`and.b{32,64} / or.b{32,64} / xor.b{32,64}`. Non-test consumer:
`backend_impl.rs` (bitand), `backend_impl.rs` (bitor),
`backend_impl.rs` (bitxor).

### Shifts (REQ-4)

`pub fn gpu_shl_i32/i64 in int_kernels.rs` performs logical left shift
via PTX `shl.b{32,64}`. `pub fn gpu_shr_i32/i64` performs ARITHMETIC
right shift via PTX `shr.s{32,64}` — sign-extending, matching PyTorch
`__rshift__` on signed dtypes. Shift count for the i64 variants is
taken from the low 32 bits of the i64 in `b[i]` (PyTorch shift amounts
are small). Non-test consumer: `backend_impl.rs` (shl),
`backend_impl.rs` (shr).

### Unary (REQ-5)

`pub fn gpu_neg_i32/i64 in int_kernels.rs` negates via `mov.s* %zero,
0; sub.s* %vr, %zero, %va`. `pub fn gpu_bitnot_i32/i64` invokes PTX
`not.b{32,64}`. Non-test consumer: `gpu_bitnot_i32 in backend_impl.rs` (neg),
`backend_impl.rs` (bitnot).

### Reductions (REQ-6, REQ-9)

`const REDUCE_I32_PTX / REDUCE_I64_PTX in int_kernels.rs` are
single-thread serial reductions: thread 0 folds all `n` elements with
an integer accumulator (`add.s* / mul.lo.s* / setp.lt.s* + selp.s* /
setp.gt.s* + selp.s*`), all other threads short-circuit via
`setp.ne.u32 %only0, %idx, 0; @%only0 bra DONE`. The `op` parameter
selects sum / prod / min / max. One-thread serial reduce keeps the
result exactly equal to a left-fold over the buffer, matching the CPU
reference bit-for-bit (no parallel-reduction reassociation drift).

Empty-input identity (REQ-9): `fn launch_reduce in int_kernels.rs`
short-circuits `n == 0` via `let host = [empty_identity]; return
Ok(stream.clone_htod(&host)?);` — `0` for sum, `1` for prod, `T::MAX`
for min, `T::MIN` for max. Non-test consumer:
`backend_impl.rs` (sum), `backend_impl.rs` (prod),
`backend_impl.rs` (min), max sibling arms.

### Div/mod by zero (REQ-7)

PTX `div.s` / `rem.s` by zero returns an implementation-defined value
on the device. Neither PyTorch on CUDA nor ferrotorch traps the
zero-divisor case at the kernel level — both rely on the user honoring
the precondition. No host round-trip is taken to special-case it
(round trips would defeat the GPU-resident discipline of R-CODE-4).
Documented at the module doc-comment level so callers don't expect
trapping.

### SAFETY discipline (REQ-8)

`fn launch_binary / launch_unary / launch_reduce in int_kernels.rs`
each wrap a single `unsafe { stream.launch_builder(&f)...launch(cfg)? }`
block preceded by a multi-line SAFETY comment naming: (a) the PTX
entry's parameter signature matching arg push order, (b) the input
buffer length binding to `n` (with `LengthMismatch` enforcement in
binary), (c) the fresh `out` allocation, (d) the PTX bound check, (e)
the `n as u32` non-truncation. R-CODE-1 grandfathers raw CUDA kernel
launches.

### n == 0 (REQ-9)

Every `fn launch_binary / launch_unary` opens with `if n == 0 { return
Ok(stream.alloc_zeros::<T>(0)?); }`. `fn launch_reduce` returns the
empty-identity via `clone_htod` (one byte transferred). Non-test
consumer relies on the no-launch short circuit for empty-tensor
integer ops via backend dispatch.

## Parity contract

`parity_ops = []` for this route (S5: INFRASTRUCTURE for `DType::I32`
and `DType::I64`). Per-op parity for `add/sub/mul/floor_divide/
remainder/bitand/bitor/bitxor/shl/shr/neg/bitnot/sum/prod/amin/amax`
on integer dtypes lives in the ferrotorch-core op crate; this file
is the dtype-specific implementation invoked by the integer arm of
each op's dispatcher.

Edge cases preserved:

- **Wrapping arithmetic**: PTX `add.s32 / sub.s32 / mul.lo.s32`
  wrap modulo 2^32 (`mul.lo` takes the low 32 bits). Matches C++
  signed overflow on CUDA (PyTorch does not promote integer ops).
- **Floor division of mixed signs**: `(-7) // 3 = -3` (not -2 which
  C `/` truncation would yield) via the floor-correct post-step.
  Matches `torch.floor_divide(-7, 3)`.
- **Remainder sign of divisor**: `(-7) % 3 = 2` (positive, sign of 3),
  not `-1` (sign of -7). Matches `torch.remainder(-7, 3)`.
- **Arithmetic right shift**: `(-8) >> 1 = -4` (sign-extending) via
  `shr.s32`, not `0x7FFFFFFC` (logical shift). Matches PyTorch
  `__rshift__` on signed dtypes.
- **Div/rem by zero**: implementation-defined value, no trap (matches
  PyTorch CUDA behavior; documented).
- **Empty reduction**: sum/prod/min/max of empty → identity-bearing
  1-element buffer (`0`/`1`/`T::MAX`/`T::MIN`). Matches
  `torch.sum/prod/amin/amax(torch.empty(0, dtype=torch.int32))`.
- **n == 0**: every elementwise op early-returns without launching.

## Verification

The `conformance_gpu_kernels.rs` integration suite exercises the
integer-arith path through the backend's i32/i64 dispatchers (the
79-test integration suite enumerates the elementwise + reduction
fixtures across dtypes). The `backend_impl::tests` suite and
`conformance_gpu_backend.rs` further cover empty-input,
mixed-sign-floor-div, and remainder-sign edge cases.

Smoke command:

```bash
cargo test -p ferrotorch-gpu --features cuda --lib int_kernels 2>&1 | tail -3
```

Expected: `test result: ok` on a host with a CUDA device. The
`#![cfg(feature = "cuda")]` gate excludes the module on no-CUDA builds.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn gpu_{add,sub,mul}_{i32,i64} in int_kernels.rs` (six wrappers around `fn launch_binary` with the matching `*_I32_PTX` / `*_I64_PTX` consts using `add.s* / sub.s* / mul.lo.s*`). Non-test consumer: `ferrotorch-gpu/src/backend_impl.rs:5800` (gpu_add_i32), `:5807` (gpu_add_i64), `:5826/5833` (sub), `:5852/5859` (mul). |
| REQ-2 | SHIPPED | impl: `pub fn gpu_floor_div_{i32,i64}` (PTX trunc-then-floor-correct in `FLOORDIV_I32_PTX/FLOORDIV_I64_PTX`) and `pub fn gpu_remainder_{i32,i64}` (PTX rem-then-sign-adjust in `REMAINDER_I32_PTX/REMAINDER_I64_PTX`) in `int_kernels.rs`. Non-test consumer: `backend_impl.rs` (floor_div i32), `backend_impl.rs` (floor_div i64), `backend_impl.rs` (remainder i32), `backend_impl.rs` (remainder i64). |
| REQ-3 | SHIPPED | impl: `pub fn gpu_{bitand,bitor,bitxor}_{i32,i64} in int_kernels.rs` (six wrappers, each using PTX `and.b* / or.b* / xor.b*`). Non-test consumer: `backend_impl.rs` (bitand), `backend_impl.rs` (bitor), `backend_impl.rs` (bitxor). |
| REQ-4 | SHIPPED | impl: `pub fn gpu_shl_{i32,i64}` (PTX `shl.b{32,64}`) and `pub fn gpu_shr_{i32,i64}` (PTX `shr.s{32,64}`, arithmetic/sign-extending) in `int_kernels.rs`. Non-test consumer: `shr in backend_impl.rs` (shl), `shr in backend_impl.rs` (shr). |
| REQ-5 | SHIPPED | impl: `pub fn gpu_neg_{i32,i64}` (PTX `sub.s* 0, %va`) and `pub fn gpu_bitnot_{i32,i64}` (PTX `not.b{32,64}`) in `int_kernels.rs`. Non-test consumer: `not in backend_impl.rs` (neg), `not in backend_impl.rs` (bitnot). |
| REQ-6 | SHIPPED | impl: `pub fn gpu_{sum,prod,min,max}_{i32,i64} in int_kernels.rs` (eight wrappers around `fn launch_reduce` with `REDUCE_I32_PTX/REDUCE_I64_PTX` and `REDUCE_SUM/PROD/MIN/MAX` op codes). Single-thread serial fold keeps result equal to left-fold over the buffer. Non-test consumer: `backend_impl.rs` (sum), `backend_impl.rs` (prod), `backend_impl.rs` (min) and sibling max arms. |
| REQ-7 | SHIPPED | impl: the module doc-comment in `int_kernels.rs` states "Integer division / remainder by zero is NOT trapped: PTX `div.s` / `rem.s` by zero returns an implementation-defined value (PyTorch on CUDA likewise does not trap). No host round-trip is taken to special-case it." The `FLOORDIV_*_PTX / REMAINDER_*_PTX` kernels do not include a zero-check branch. Non-test consumer relies on the documented no-trap contract via the backend's integer arm. |
| REQ-8 | SHIPPED | impl: the three `unsafe { stream.launch_builder(&f)...launch(cfg)? }` blocks in `fn launch_binary / launch_unary / launch_reduce` (in `int_kernels.rs`) are each preceded by a multi-line SAFETY comment naming entry signature, length binding, alloc, bound check, and `n as u32` non-truncation. Non-test consumer inherits the SAFETY contract via every `pub fn gpu_*_i{32,64}` wrapper called from backend_impl. |
| REQ-9 | SHIPPED | impl: `fn launch_binary / launch_unary` open with `if n == 0 { return Ok(stream.alloc_zeros::<T>(0)?); }`; `fn launch_reduce` short-circuits `n == 0` with `let host = [empty_identity]; return Ok(stream.clone_htod(&host)?);` using `0`/`1`/`T::MAX`/`T::MIN` for sum/prod/min/max respectively. Non-test consumer relies on the no-launch short circuit via the backend's empty-int handling. |
