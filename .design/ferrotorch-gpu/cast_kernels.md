# Dtype-cast CUDA kernels

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

`ferrotorch-gpu/src/cast_kernels.rs` owns elementwise dtype conversion
GPU kernels. Hand-written PTX, loaded via
`crate::module_cache::get_or_compile`. Elementwise 1-D, one thread per
element, no host round-trip — the result stays resident on the device.
Mirrors PyTorch's `aten/src/ATen/native/cuda/Copy.cu` (the
`gpu_kernel_nocast` cast-on-write path) and the `c10::convert<>`
machinery in `c10/util/`.

## Requirements

- REQ-1: Float → int cast — `cast_f32_to_i32`, `cast_f32_to_i64`,
  `cast_f64_to_i32`, `cast_f64_to_i64`, `cast_f16_to_i32`,
  `cast_f16_to_i64`, `cast_bf16_to_i32`, `cast_bf16_to_i64` — each
  TRUNCATES toward zero (matching PyTorch `.to(torch.int)` /
  `.to(torch.long)`). PTX `cvt.rzi.s{32,64}.f{32,64}` does this
  natively. bf16/f16 inputs are first widened to f32 then `cvt.rzi`.

- REQ-2: Int → float cast — `cast_i32_to_f32`, `cast_i32_to_f64`,
  `cast_i32_to_f16`, `cast_i32_to_bf16`, `cast_i64_to_f32`,
  `cast_i64_to_f64`, `cast_i64_to_f16`, `cast_i64_to_bf16` — PTX
  `cvt.rn.f{32,64}.s{32,64}` (round-to-nearest-even). bf16/f16 outputs
  go via f32 then `cvt.rn.{bf16,f16}.f32`.

- REQ-3: Int → int cast — `cast_i32_to_i64` (sign-extend via
  `cvt.s64.s32`) and `cast_i64_to_i32` (truncate high bits via
  `cvt.s32.s64`, wrapping on overflow — PyTorch CUDA `.to(torch.int)`
  semantics).

- REQ-4: Same-dtype identity copy — `cast_i32_copy`, `cast_i64_copy` —
  bit-for-bit element copy kept GPU-resident. Required because an
  i64→i32→i64 round trip would corrupt values outside the i32 range;
  same-dtype cast must preserve the full value.

- REQ-5: Bool → float cast — `cast_bool_to_f32`, `cast_bool_to_f64`,
  `cast_bool_to_f16`, `cast_bool_to_bf16` — `true → 1.0`, `false →
  0.0` (PyTorch parity for `bool_tensor.float()`). Normalises input
  via `setp.ne %nz, %bv, 0` first so any nonzero u8 maps to true.

- REQ-6: Every `unsafe { ... launch(cfg)? }` block carries a SAFETY
  comment per R-CODE-1.

- REQ-7: `n == 0` is an early-return path (no launch). The `n`
  parameter is the LOGICAL element count supplied by the caller (NOT
  `input.len()`, which may be a pool-rounded over-allocation when the
  input comes from a pooled float op); the kernel reads/writes strictly
  `[0, n)`.

## Acceptance Criteria

- [x] AC-1: All eight float → int cast functions exist with the
  documented `(x, n, device)` signature.
- [x] AC-2: All eight int → float cast functions exist.
- [x] AC-3: `cast_i32_to_i64`, `cast_i64_to_i32`, `cast_i32_copy`,
  `cast_i64_copy` exist.
- [x] AC-4: All four bool → float cast functions exist.
- [x] AC-5: Every `unsafe { ... }` block has a SAFETY comment
  immediately above it.
- [x] AC-6: `cargo test -p ferrotorch-gpu --features cuda --lib
  cast_kernels` passes its inline `#[cfg(test)] mod tests` (7 tests
  in the file's tests module).

## Architecture

### Per-cast PTX `&'static str` (REQ-1, REQ-2, REQ-3, REQ-5)

Each cast has its own hand-written PTX const (e.g.
`const F32_TO_I32_PTX`, `const I32_TO_F32_PTX`, `const I32_TO_I64_PTX`,
`const BOOL_TO_F32_PTX`). Each kernel has the signature
`(in_ptr: u64, out_ptr: u64, n: u32)`. The input element stride and
output element stride are encoded in the PTX body (`shl by 1/2/3` for
2/4/8-byte elements). bf16 outputs require sm_80+ (the `cvt.rn.bf16.f32`
instruction); f16 outputs require sm_53+ (`cvt.rn.f16.f32`); all other
casts target sm_52. The host RTX 3090 is sm_86, well above the floor.

bf16 decode (file: `cast_kernels.rs`): a bf16 is the high 16 bits of an
f32 — splat into the high half of a b32 register via `shl 16` + reinterpret
(`cvt.u32.u16 %bits, %h; shl.b32 %bits, %bits, 16; mov.b32 %v, %bits`).
This mirrors the bf16-decode pattern in `crate::bf16` / `crate::bool_kernels`.

### Float → int truncate-toward-zero (REQ-1)

PTX `cvt.rzi.s{32,64}.f{32,64}` is the round-toward-zero-integer
conversion. Matches PyTorch's `.to(torch.int)` semantics on CUDA
(`c10::static_cast_with_inter_type<int, float>` which the device-side
compiler lowers to the same `cvt.rzi`). NaN: `cvt.rzi` on NaN yields
0 on most NV architectures (matches PyTorch's CUDA-side behavior;
CPU diverges but CUDA-CPU divergence on NaN-to-int is a known PyTorch
caveat).

### Int → float round-to-nearest-even (REQ-2)

PTX `cvt.rn.f{32,64}.s{32,64}` — round-to-nearest-even (the IEEE-754
default). int → f16/bf16 chains: int → f32 (`cvt.rn.f32.s*`) → narrow
to half (`cvt.rn.{f16,bf16}.f32`). Matches PyTorch's
`c10::convert<at::Half, int>` which itself goes through float.

### Int → int widen/narrow (REQ-3)

`cvt.s64.s32` sign-extends; `cvt.s32.s64` truncates the high bits
(wrap-around on overflow, matching PyTorch's CUDA `.to(torch.int)`).
Note: the CPU `IntTensor::cast` in ferrotorch-core errors on
out-of-range narrowing — this GPU path documents the wrapping
divergence at the module doc-comment level (PyTorch CUDA wraps; PyTorch
CPU also wraps for the C++ static_cast path; the ferrotorch CPU
divergence is the one to revisit, not the GPU).

### Same-dtype identity copy (REQ-4)

`cast_i32_copy` / `cast_i64_copy` exist because a `.cast::<I>()` to the
SAME integer dtype must preserve the full value bit-for-bit. A
narrow-then-widen round trip (i64 → i32 → i64) would corrupt values
outside the i32 range. The kernels are plain `ld.global.b{32,64};
st.global.b{32,64}` element copies kept GPU-resident.

### Bool → float (REQ-5)

`const BOOL_TO_F32_PTX` (and friends) reads `a[i]` as `.u8`, normalises
via `setp.ne.u16 %nz, %bv, 0; selp.u32 %iv, 1, 0, %nz` so any nonzero
input maps to `1`, then converts via `cvt.rn.f32.u32`. PyTorch parity:
`bool_tensor.float()` → 1.0 for true, 0.0 for false.

### Single launch harness (REQ-6, REQ-7)

`fn launch_cast<IN, OUT> in cast_kernels.rs` is the single launch
harness shared across all 20+ cast wrappers. It takes the input
buffer, the LOGICAL `n` (not `input.len()`), the PTX `&'static str`,
and the kernel name, then: compiles via `get_or_compile`, allocs
`out: CudaSlice<OUT>` of `n` elements via `stream.alloc_zeros`,
computes `cfg = launch_1d(n)`, casts `n as u32`, and dispatches inside
a single `unsafe { ... launch(cfg)? }` block whose SAFETY comment
names the entry signature, the `n` debug_assert, the alloc, the bound
check, and the non-truncation.

The `debug_assert!(input.len() >= n, "cast input slice shorter than
logical n")` is the key invariant: `n` is the LOGICAL count from the
caller, while `input.len()` may be a pool-rounded over-allocation
when the input came from a pooled float op (cf. the pool-rounded
allocs in `crate::buffer` and `crate::backend_impl`'s float arenas).

Non-test production consumer: `crate::backend_impl::CudaBackendImpl`'s
dtype-cast dispatcher; see `backend_impl.rs:6329` (`use
crate::cast_kernels as ck` followed by float→int casts at `:6334`),
`:6381` (int→float at `:6386`), `:6433` (int↔int at `:6438..:6452`),
`:6616` (bool→float at `:6630`).

## Parity contract

`parity_ops = []` for this route (S5: INFRASTRUCTURE for `to_dtype` /
`.cast()`). Per-op parity is enforced at the ferrotorch-core op crate;
this file is the dtype-cast backend that the op crate dispatches when
the GPU is the target device.

Edge cases preserved:

- **NaN → int**: `cvt.rzi.s*.f32` on NaN typically yields 0 on NVIDIA
  hardware; PyTorch CUDA matches this behavior. (CPU diverges; this is
  a documented PyTorch caveat.)
- **±Inf → int**: `cvt.rzi.s*.f32` on `+Inf` yields the max signed int
  for the destination width; on `-Inf` yields the min. Matches PyTorch
  CUDA.
- **Out-of-range int narrow** (e.g. `i64 → i32` of `2^33`): PTX
  `cvt.s32.s64` truncates the high bits → wrap-around. Matches PyTorch
  CUDA's `.to(torch.int)` (C++ static_cast on int types).
- **Subnormal float → int**: `cvt.rzi` on a subnormal float just
  rounds toward zero, yielding 0 for any subnormal in `(-1, 1)`.
- **Bool nonzero → 1.0**: `setp.ne %nz, %bv, 0; selp 1, 0, %nz` so
  any nonzero u8 maps to `1.0` (not raw `2.0` etc.) — matches
  PyTorch's `bool_tensor.float()` canonical 0/1.
- **n == 0**: early return with `alloc_zeros::<OUT>(0)` — no launch.

## Verification

Inline `#[cfg(test)] mod tests` in
`ferrotorch-gpu/src/cast_kernels.rs` has 7 tests covering: f32→i32
truncation, f32→i64 truncation, i32→f32 + i32→i64 widening,
i64→i32 narrowing + i64→f64, f64→i64 + i32→bf16/f16,
bf16/f16→int truncation, same-dtype copy preserving full i64 value.

Integration tests in `ferrotorch-gpu/tests/conformance_gpu_kernels.rs`
and `conformance_gpu_backend.rs` exercise the cast path through the
backend dispatch.

Smoke command:

```bash
cargo test -p ferrotorch-gpu --features cuda --lib cast_kernels 2>&1 | tail -3
```

Expected: `test result: ok. 7 passed; 0 failed` on a host with a CUDA
device.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn cast_{f32,f64,f16,bf16}_to_{i32,i64} in cast_kernels.rs` (eight wrappers around `fn launch_cast`, each with its own `*_PTX` const using `cvt.rzi`). Non-test consumer: `ferrotorch-gpu/src/backend_impl.rs:6334` (`ck::cast_f32_to_i32`) plus seven sibling float→int dispatch arms in the same `to_dtype` handler. |
| REQ-2 | SHIPPED | impl: `pub fn cast_{i32,i64}_to_{f32,f64,f16,bf16} in cast_kernels.rs` (eight wrappers using `cvt.rn`). Non-test consumer: `backend_impl.rs:6386` (`ck::cast_i32_to_f32`) plus seven sibling int→float dispatch arms. |
| REQ-3 | SHIPPED | impl: `pub fn cast_i32_to_i64` (sign-extend `cvt.s64.s32`) and `pub fn cast_i64_to_i32` (truncate `cvt.s32.s64`, wrapping) in `cast_kernels.rs`. Non-test consumer: `backend_impl.rs:6438` (`ck::cast_i32_to_i64`), `:6443` (`ck::cast_i64_to_i32`). |
| REQ-4 | SHIPPED | impl: `pub fn cast_i32_copy / cast_i64_copy in cast_kernels.rs` (plain `ld.global.b{32,64}; st.global.b{32,64}` element copies). Non-test consumer: `backend_impl.rs:6452` (`ck::cast_i32_copy`) plus the i64-copy sibling in the same handler. |
| REQ-5 | SHIPPED | impl: `pub fn cast_bool_to_{f32,f64,f16,bf16} in cast_kernels.rs` (each normalises via `setp.ne %nz, %bv, 0; selp 1, 0, %nz` then `cvt.rn.f*.u32`). Non-test consumer: `backend_impl.rs:6630` (`ck::cast_bool_to_f32`) plus three sibling bool→float arms in the `to_dtype` handler. |
| REQ-6 | SHIPPED | impl: the single `unsafe { stream.launch_builder(&f)...launch(cfg)? }` block in `fn launch_cast` is preceded by a multi-line SAFETY comment naming entry signature, alloc, bound check, and `n as u32` non-truncation. Non-test consumer inherits the SAFETY contract via every `pub fn cast_*` wrapper called from backend_impl. |
| REQ-7 | SHIPPED | impl: `fn launch_cast in cast_kernels.rs` opens with `if n == 0 { return Ok(stream.alloc_zeros::<OUT>(0)?); }` and a `debug_assert!(input.len() >= n)` for the pool-rounded-input case. Non-test consumer relies on this no-launch short circuit for empty dtype casts via the backend `to_dtype` op. |
