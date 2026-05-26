# numeric_cast — fallible numeric type conversion

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/
  - c10/util/Half.h
  - c10/util/BFloat16.h
-->

## Summary

`ferrotorch-core/src/numeric_cast.rs` ships the workspace's fallible numeric
cast helper `pub fn cast<T, U>(v: T) -> FerrotorchResult<U>`. It wraps
`num_traits::NumCast::from(v)` with an explicit finiteness guard so the
narrow-float saturation bug (issue #815) is caught: `num_traits` silently
saturates a finite source to `±Infinity` when the target is `bf16` or `f16`,
which violates the cast contract ("Err on values not representable").
This helper exists because `T::from(v).unwrap()` panics — forbidden by
R-CODE-2 — and the inline `try_into()` pattern requires per-call-site
boilerplate.

## Requirements

- REQ-1: `pub fn cast<T, U>(v: T) -> FerrotorchResult<U>` where `T:
  ToPrimitive + Debug + Copy` and `U: NumCast`. Returns
  `Err(FerrotorchError::InvalidArgument)` with a structured message when
  the value can't be represented in `U`. Mirrors PyTorch's runtime check
  pattern at `c10::checked_convert<U, T>(value, name)` (used across
  `aten/src/ATen/native/Copy.cpp` to error on overflow during dtype cast).
- REQ-2: Saturation guard for narrow-float targets (issue #815). The
  helper compares `v.to_f64().is_finite()` vs `result.to_f64().is_finite()`
  — if the source was finite but the result is non-finite, the underlying
  `NumCast` saturated and the cast returns `Err`. Genuine non-finite
  passthrough (`f64::INFINITY -> bf16::INFINITY`, `NaN -> NaN`) is
  preserved.
- REQ-3: Integer-target casts are unaffected by the saturation guard:
  `result.to_f64()` always projects to a finite f64 (integers are
  finite by construction), so the guard is a no-op cost in those (very
  common) call sites.
- REQ-4: Error message includes source type name, target type name, and
  the source value's `Debug` representation — enough for a caller seeing
  a propagated error to diagnose the offending value without rerunning.
- REQ-5: The helper is `#[inline]` so the call-site dispatch + the
  `NumCast::from` lookup are eligible for inlining; for primitive `T -> U`
  pairs the whole thing compiles to a single conditional + a value
  conversion in the happy path.

## Acceptance Criteria

- [x] AC-1: `cast_f64_to_f32_succeeds_for_finite` at `numeric_cast.rs:120`
  — finite `f64 -> f32` cast happy path.
- [x] AC-2: `cast_f64_inf_to_i32_fails` at `numeric_cast.rs:126` —
  `Infinity` to `i32` returns `Err` (mirroring `c10::checked_convert`'s
  overflow check).
- [x] AC-3: `cast_usize_to_f32_succeeds` at `numeric_cast.rs:137`.
- [x] AC-4: `cast_to_bf16_round_trip` at `numeric_cast.rs:144`.
- [x] AC-5: `cast_negative_to_unsigned_fails` at `numeric_cast.rs:149`.
- [x] AC-6: `cast_huge_f64_to_bf16_returns_err` at `numeric_cast.rs:166`
  — `1e300_f64` to `bf16` returns `Err` (issue #815 regression pin).
- [x] AC-7: `cast_huge_f32_to_bf16_returns_err` at `numeric_cast.rs:180`.
- [x] AC-8: `cast_f64_inf_to_bf16_passes_through` at `numeric_cast.rs:188`
  — genuine `Infinity` passthrough (cast did not saturate; result is
  `bf16::INFINITY`).
- [x] AC-9: `cast_f64_neg_inf_to_bf16_passes_through` at `numeric_cast.rs:196`.
- [x] AC-10: `cast_f64_nan_to_bf16_passes_through` at `numeric_cast.rs:202`.
- [x] AC-11: `cast_f64_in_range_to_bf16_succeeds` at `numeric_cast.rs:209`.
- [x] AC-12: `cast_huge_f64_to_f16_returns_err` at `numeric_cast.rs:215`
  — symmetric to AC-6 for `f16`.
- [x] AC-13: `cast_f64_inf_to_f16_passes_through` at `numeric_cast.rs:223`.
- [x] AC-14: `cast_f64_nan_to_f16_passes_through` at `numeric_cast.rs:229`.
- [x] AC-15: `cast_f64_in_range_to_f16_succeeds` at `numeric_cast.rs:235`.
- [x] AC-16: `cast_huge_f64_to_f32_returns_err` at `numeric_cast.rs:241`
  — even `f64::MAX -> f32` (which both `num_traits` saturate to
  `f32::INFINITY`) is caught.
- [x] AC-17: `cast_f64_inf_to_f32_passes_through` at `numeric_cast.rs:249`.

## Architecture

### Function body (`numeric_cast.rs:71-114`)

```rust
pub fn cast<T, U>(v: T) -> FerrotorchResult<U>
where
    T: num_traits::ToPrimitive + std::fmt::Debug + Copy,
    U: num_traits::NumCast,
{
    let result: U = <U as num_traits::NumCast>::from(v)
        .ok_or_else(|| FerrotorchError::InvalidArgument { ... })?;
    // Saturation guard (issue #815):
    let src_finite = v.to_f64().is_some_and(f64::is_finite);
    if src_finite {
        if let Some(r) = result.to_f64() {
            if !r.is_finite() {
                return Err(FerrotorchError::InvalidArgument { ... });
            }
        }
    }
    Ok(result)
}
```

Two failure modes:
1. `NumCast::from` returns `None` — source is unrepresentable in the
   target's numeric range (e.g. `f64::INFINITY -> i32`).
2. `NumCast::from` returns `Some(non_finite)` for a finite source — the
   underlying cast silently saturated. (Issue #815.)

The guard is layered ON TOP of `num_traits` rather than replacing it
because `num_traits` correctly handles every other case (integer
overflow, NaN/Inf passthrough, integer truncation). Only the
narrow-float saturation is wrong upstream, so we patch that case
specifically.

### Issue #815 narrow-float saturation

`num_traits`'s `NumCast::from(1e300_f64)` for target `half::bf16`
silently returns `Some(bf16::INFINITY)` — the implementation
saturates rather than reports failure. This breaks downstream code
that assumes cast failure means "Err returned", producing
`weight.cast::<bf16>().unwrap() == bf16::INFINITY` which then
corrupts every subsequent computation. The fix is in this helper, not
in `num_traits` (whose maintainers consider saturation a feature for
some use cases). Tests at `numeric_cast.rs:166-247` pin every narrow-
float + finite/infinite combination.

### Production consumers

- `ferrotorch-core/src/fft.rs:30` — `use crate::numeric_cast::cast;` and
  uses `cast::<T, f64>(...)` / `cast::<f64, T>(...)` inside the FFT
  scale-factor computation. This is the in-tree non-test consumer that
  ferrotorch-core's `cargo test` exercises end-to-end.
- Every fallible numeric coercion in the workspace should route through
  this helper rather than `T::from(v).unwrap()`. crosslink `#815`
  documents the migration plan; the rule is enforced by code review +
  the `R-CODE-2` clippy lint (`unwrap` on `NumCast::from` is forbidden
  in production code).

## Parity contract

`parity_ops = []`. The parity surface is the dtype-cast correctness of
every op that converts between dtypes. The narrow-float saturation bug
this helper guards against would manifest as e.g. a `bf16` weight
silently going to `Infinity` after a `to(bf16)` call from a `f32` source
with absolute value > `bf16::MAX`. Upstream PyTorch's equivalent
`tensor.to(torch.bfloat16)` errors on such inputs via
`c10::checked_convert`'s overflow detection.

## Verification

```
cargo test -p ferrotorch-core --lib numeric_cast
```

Expected: 17 tests pass, 0 failed (see Acceptance Criteria 1-17 above).

The test suite is unusually large for a one-function helper because
this is a regression bar: every narrow-float saturation behavior
documented in issue #815 has a corresponding pinning test that fails
if the saturation guard is removed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn cast<T, U>` at `ferrotorch-core/src/numeric_cast.rs:71-114` with structured `FerrotorchError::InvalidArgument` on failure (`:77` and `:100`). Non-test production consumer: `ferrotorch-core/src/fft.rs:30` `use crate::numeric_cast::cast;` and downstream callsites in the FFT scale-factor computation. |
| REQ-2 | SHIPPED | impl: saturation-guard block at `ferrotorch-core/src/numeric_cast.rs:96-110` — compares `v.to_f64().is_finite()` against `result.to_f64().is_finite()`; if source finite & result non-finite → `Err`. Tests: 4 narrow-float saturation tests at `:166-220` (`cast_huge_f64_to_bf16`, `cast_huge_f32_to_bf16`, `cast_huge_f64_to_f16`, `cast_huge_f64_to_f32`); 5 passthrough tests at `:188-235` (`Inf`/`-Inf`/`NaN` cast preserves non-finite-ness). Non-test production consumer: `ferrotorch-core/src/fft.rs:30` callsite — narrow-float FFTs are exactly the case where this guard matters. |
| REQ-3 | SHIPPED | impl: the guard at `numeric_cast.rs:96-110` is a no-op cost for integer targets because `r.to_f64()` always returns a finite value for finite integers (integers project losslessly to f64 within the i64 range). Test: `cast_f64_inf_to_i32_fails` at `:126` exercises the integer-target path. Non-test production consumer: any caller using `cast::<f64, i32>(...)` (none in ferrotorch-core today; the helper is dtype-agnostic by construction). |
| REQ-4 | SHIPPED | impl: the error message at `ferrotorch-core/src/numeric_cast.rs:78-83` and `:101-108` includes `type_name::<T>()`, `type_name::<U>()`, and `{:?}` of the source value. Verified by the post-fix message assertion in `cast_huge_f64_to_bf16_returns_err` at `:172` which checks for the substring `"saturates to non-finite"` or `"not representable"`. Non-test production consumer: any code that propagates a cast error via `?` — the error message reaches the user / log scraper untouched. |
| REQ-5 | SHIPPED | impl: `#[inline]` attribute at `ferrotorch-core/src/numeric_cast.rs:70`. Non-test production consumer: `ferrotorch-core/src/fft.rs` callsites — the `#[inline]` ensures the cast collapses into the surrounding FFT scale-factor pipeline at compile time. |
