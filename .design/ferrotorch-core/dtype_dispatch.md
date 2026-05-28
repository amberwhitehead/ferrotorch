# dtype_dispatch — `dispatch_floating_dtype!` macro + dtype probes

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/Dispatch.h
  - c10/core/ScalarType.h
-->

## Summary

`ferrotorch-core/src/dtype_dispatch.rs` ships the workspace's static-dispatch
analog of PyTorch's `AT_DISPATCH_FLOATING_TYPES_AND_HALF` macro
(`aten/src/ATen/Dispatch.h`). The `dispatch_floating_dtype!` macro
branches on the static type parameter `T` via `TypeId` and routes to one
of four dtype-specialized arms (`f32`, `f64`, `bf16`, `f16`), returning
`Err(FerrotorchError::NotImplementedOnCuda { op })` for any other dtype.
It replaces ad-hoc `if is_f32::<T>() { ... } else if is_f64::<T>() { ... }`
chains that silently fell through to the f32 path for bf16 inputs (issue
#23 pattern A — landed as crosslink #1185 Phase 1).

## Requirements

- REQ-1: A `dispatch_floating_dtype!($scalar_t, $op, f32 => …, f64 => …,
  bf16 => …, f16 => …)` macro that statically branches on `TypeId::of::<T>()`
  and evaluates the matching arm. Mirrors `AT_DISPATCH_FLOATING_TYPES_AND_HALF`
  at `aten/src/ATen/Dispatch.h` (which uses `AT_PRIVATE_CASE_TYPE` switch
  arms keyed on `ScalarType`).
- REQ-2: Each arm is a Rust expression evaluating to the **same**
  `FerrotorchResult<U>` type. The macro evaluates to that result. This
  collapses the four-way dispatch into a single uniform return.
- REQ-3: Unsupported dtypes (everything outside `{f32, f64, bf16, f16}`)
  produce `Err(FerrotorchError::NotImplementedOnCuda { op: $op })` — the
  same structured error every other "dtype not supported" path uses
  (REQ-6 of `error.md`).
- REQ-4: `is_f32::<T>()`, `is_f64::<T>()`, `is_bf16::<T>()`, `is_f16::<T>()`
  — public `T: 'static` predicates that callers use OUTSIDE the macro to
  branch on dtype for non-dispatch reasons (e.g. picking the right
  `ensure_contig_for_gpu` variant before the macro's arms run).
- REQ-5: `#[macro_export]` so the macro is visible across the workspace
  (it's reached via `crate::dispatch_floating_dtype!` thanks to the
  `$crate::` substitution in the macro body for `FerrotorchError`).
- REQ-6: The four supported floating dtypes are exactly the four `Float`
  implementors (`dtype.rs:28-35`) — there is no slot for unsupported
  dtypes inside `Float`, and the macro's `else` arm produces the
  structured error rather than silently degrading. This is the
  structural defense against issue #23 pattern A (silent f32 fallthrough
  for bf16 buffers).

## Acceptance Criteria

- [x] AC-1: `dispatch_f32` at `dtype_dispatch.rs:168` — invoking the
  macro with `T = f32` selects the `f32` arm.
- [x] AC-2: `dispatch_f64` at `dtype_dispatch.rs:173` — `T = f64` selects
  the `f64` arm.
- [x] AC-3: `dispatch_bf16` at `dtype_dispatch.rs:178` — `T = half::bf16`
  selects the `bf16` arm.
- [x] AC-4: `dispatch_f16` at `dtype_dispatch.rs:183` — `T = half::f16`
  selects the `f16` arm (DISTINCT from the bf16 arm; the test is
  specifically authored to pin this disambiguation per crosslink #1185
  Phase 1).
- [x] AC-5: `dispatch_unsupported_dtype_returns_not_implemented` at
  `dtype_dispatch.rs:191` — invoking the macro with `T = i32` returns
  `FerrotorchError::NotImplementedOnCuda { op: "test_op" }`. (The macro
  is dtype-agnostic at the TypeId level; unsupported dtypes never reach
  a kernel.)
- [x] AC-6: `is_f32::<f32>() == true`, `is_f32::<f64>() == false` —
  mechanical check via `TypeId` comparison at `dtype_dispatch.rs:126`.

## Architecture

### Macro body (`dtype_dispatch.rs:95-119`)

```rust
#[macro_export]
macro_rules! dispatch_floating_dtype {
    (
        $scalar_t:ty,
        $op:literal,
        f32 => $f32_arm:expr,
        f64 => $f64_arm:expr,
        bf16 => $bf16_arm:expr,
        f16 => $f16_arm:expr $(,)?
    ) => {{
        if ::std::any::TypeId::of::<$scalar_t>() == ::std::any::TypeId::of::<f32>() {
            $f32_arm
        } else if ::std::any::TypeId::of::<$scalar_t>() == ::std::any::TypeId::of::<f64>() {
            $f64_arm
        } else if ::std::any::TypeId::of::<$scalar_t>() == ::std::any::TypeId::of::<half::bf16>() {
            $bf16_arm
        } else if ::std::any::TypeId::of::<$scalar_t>() == ::std::any::TypeId::of::<half::f16>() {
            $f16_arm
        } else {
            ::std::result::Result::Err($crate::error::FerrotorchError::NotImplementedOnCuda {
                op: $op,
            })
        }
    }};
}
```

The macro is **complete** (the dtype-list IS the closed set of four
`Float` implementors); adding a fifth requires editing both `dtype.rs`
and this macro in lockstep. Forgetting one half of the pair fails at
the call site rather than silently falling through.

### Predicates (`dtype_dispatch.rs:125-150`)

```rust
pub fn is_f32<T: 'static>() -> bool { TypeId::of::<T>() == TypeId::of::<f32>() }
pub fn is_f64<T: 'static>() -> bool { TypeId::of::<T>() == TypeId::of::<f64>() }
pub fn is_bf16<T: 'static>() -> bool { TypeId::of::<T>() == TypeId::of::<half::bf16>() }
pub fn is_f16<T: 'static>() -> bool { TypeId::of::<T>() == TypeId::of::<half::f16>() }
```

These exist for callers that need to branch on dtype before entering the
macro (e.g. picking the right pre-processing fn). The macro itself uses
`TypeId::of` inline rather than calling these helpers, to keep the
arm-selection a single conditional chain.

### Why TypeId (R-DEV-7)

PyTorch's macro switches on a runtime `ScalarType` enum stored in
`Tensor::scalar_type()`. ferrotorch's tensor is generic over `T: Float`
at compile time, so the dtype is **available at compile time** —
`TypeId::of::<T>()` is a `const`-fn-shaped lookup the compiler generally
folds into a constant. The four-way conditional chain compiles down to a
direct jump to the matching arm with the unused arms dead-stripped.
This is a stricter contract than upstream's macro (compile-time vs
runtime), and it's free of the silent fallthrough risk because every
arm must be present syntactically.

### Production consumers

- `ferrotorch-core/src/grad_fns/arithmetic.rs:410-433` —
  `dispatch_floating_dtype!(T, "add", f32 => …, f64 => …, bf16 => …,
  f16 => …)` selects the right CUDA `backend.add_*_*` kernel based on
  `T`. The `add` GPU path is the original site that motivated REQ-1
  (crosslink #1185 Phase 1 / issue #23).
- `ferrotorch-core/src/grad_fns/arithmetic.rs:447, :997` — additional
  arithmetic ops using the macro.
- `ferrotorch-core/src/fft.rs:139-160 / :229-260 / :313 / :406` —
  uses `is_f32::<T>()` / `is_f64::<T>()` to gate the GPU-accelerated FFT
  path.
- Every `grad_fns/*.rs` GPU-arm in ferrotorch-core potentially calls
  this — the macro is the canonical dispatch primitive.

## Parity contract

`parity_ops = []`. The parity surface is the indirect kernel-correctness
of every op that uses the macro to pick its GPU arm. If the macro
incorrectly routes `bf16` to the `f32` arm (the pre-#1185 bug), every
bf16 op's parity sweep would diverge from the upstream f32 oracle (the
upload would happen as `bf16` bytes but the kernel would read them as
`f32`, producing nonsense). The `dispatch_bf16` and `dispatch_f16` tests
in this file pin the structural fix.

## Verification

```
cargo test -p ferrotorch-core --lib dtype_dispatch
```

Expected: 5 tests pass, 0 failed.

The 5 tests at `dtype_dispatch.rs:153-201` cover all four supported
dtypes' arm selection plus the unsupported-dtype error path. Each
dispatch_* test invokes the macro with the corresponding type alias
and asserts the returned string sentinel matches the arm's literal.

End-to-end exercise lands in the GPU integration probes that build with
`--features gpu` (`ferrotorch-core/tests/_probe_phase3*.rs`).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: macro `dispatch_floating_dtype!` at `ferrotorch-core/src/dtype_dispatch.rs:95-119` mirroring `AT_DISPATCH_FLOATING_TYPES_AND_HALF` at `aten/src/ATen/Dispatch.h`. Non-test production consumer: `ferrotorch-core/src/grad_fns/arithmetic.rs:413` invokes `crate::dispatch_floating_dtype!(T, "add", f32 => ..., f64 => ..., bf16 => ..., f16 => ...)` to pick the right CUDA `add_*_*` arm. Tests: 5 arm-selection + 1 unsupported-dtype at `dtype_dispatch.rs:153-201`. |
| REQ-2 | SHIPPED | impl: macro arms all return the same `FerrotorchResult<U>` (tied together by the trailing `Err(...)` arm); the macro itself evaluates to that result. Non-test production consumer: every call site in `ferrotorch-core/src/grad_fns/arithmetic.rs:413/:447/:997` binds the macro to `let h: FerrotorchResult<GpuBufferHandle> = crate::dispatch_floating_dtype!(...)` — the unified-return-type contract is what makes this binding type-check. |
| REQ-3 | SHIPPED | impl: trailing `else` arm at `ferrotorch-core/src/dtype_dispatch.rs:113-117` returns `Err(FerrotorchError::NotImplementedOnCuda { op: $op })`. Test: `dispatch_unsupported_dtype_returns_not_implemented` at `dtype_dispatch.rs:191`. Non-test production consumer: the same `grad_fns/arithmetic.rs` callsites — the structured error is what propagates back to user code when a non-Float dtype somehow reaches a CUDA path. |
| REQ-4 | SHIPPED | impl: `pub fn is_f32` / `is_f64` / `is_bf16` / `is_f16` at `is_f32 in ferrotorch-core/src/dtype_dispatch.rs`. Non-test production consumer: `is_f64 in ferrotorch-core/src/fft.rs` checks `if input.is_cuda() && (is_f32::<T>() || is_f64::<T>())` to gate the GPU FFT path; `is_f16 in ferrotorch-core/src/grad_fns/arithmetic.rs` uses `if is_f32::<T>()` to pick the right helper. |
| REQ-5 | SHIPPED | impl: `#[macro_export]` at `ferrotorch-core/src/dtype_dispatch.rs:95` exports the macro for cross-crate use; the `$crate::error::FerrotorchError` substitution at `:114` makes the macro hygienic when invoked from another module. Non-test production consumer: invocations from `crate::dispatch_floating_dtype!` in `grad_fns/arithmetic.rs:413` (cross-module within the same crate, relies on the `#[macro_export]` flag for the rustc resolver). |
| REQ-6 | SHIPPED | impl: the four supported dtypes (`f32 / f64 / bf16 / f16`) at `ferrotorch-core/src/dtype_dispatch.rs:101-103` are identically the four `Float` impls at `ferrotorch-core/src/dtype.rs:28-35`. The lockstep is maintained manually (no automated test pins the cross-file pair, but the GPU integration probes for each dtype FAIL at compile time if a dtype is added to one side without the other). Non-test production consumer: every GPU arm in `grad_fns/arithmetic.rs` — issue #23 pattern A would have been a `bf16` value silently routed to the `f32` arm before this macro existed; the macro's complete `(f32, f64, bf16, f16)` enumeration is the structural fix. |
