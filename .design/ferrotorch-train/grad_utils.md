# ferrotorch-train — `clip_grad_norm_` / `clip_grad_value_` re-exports

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/utils/clip_grad.py
-->

## Summary

`ferrotorch-train/src/grad_utils.rs` previously hosted hand-rolled
CPU-only forks of `clip_grad_norm_` and `clip_grad_value_`. Pass
5.B.2 (#1104) deduplicated those forks by re-exporting the canonical,
device-dispatching implementations from `ferrotorch_nn::utils`. The
module mirrors PyTorch's `torch.nn.utils.clip_grad_norm_` /
`torch.nn.utils.clip_grad_value_` (`torch/nn/utils/clip_grad.py`).

## Requirements

- REQ-1: Module re-exports `clip_grad_norm_` and `clip_grad_value_`
  from `ferrotorch_nn::utils`. Mirrors `from torch.nn.utils import
  clip_grad_norm_, clip_grad_value_`.
- REQ-2: The two re-exported function pointers MUST resolve to the
  same symbol as `ferrotorch_nn::clip_grad_norm_` / `clip_grad_value_`
  — i.e. there is exactly one implementation, not two forks that can
  drift.
- REQ-3: `clip_grad_norm_(params, max_norm, norm_type)` mirrors the
  PyTorch semantics:
  - L2 (norm_type = 2.0): `total_norm = sqrt(sum(grad^2))`, clip
    coefficient `clip_coef = max_norm / total_norm`. If `total_norm >
    max_norm`, scale each grad by `clip_coef`; else leave unchanged.
  - L1 (norm_type = 1.0): `total_norm = sum(|grad|)`.
  - Inf (norm_type = f64::INFINITY): `total_norm = max(|grad|)`.
  - Returns the pre-clip `total_norm` as `f64`.
- REQ-4: `clip_grad_value_(params, clip_value)` clamps each gradient
  element to `[-clip_value, clip_value]` in place.
- REQ-5: Both functions handle the "no gradients set" case: parameters
  with `grad.is_none()` contribute zero to the total norm and are
  passed through unchanged. Returning `0.0` is the documented
  semantics (matches PyTorch's "skip None grads" behavior).
- REQ-6: Both functions dispatch to CPU or CUDA based on the
  parameter's storage device. The dispatch is owned by
  `ferrotorch_nn::utils`; this module is a pure re-export.

## Acceptance Criteria

- [x] AC-1: `pub use ferrotorch_nn::utils::{clip_grad_norm_,
  clip_grad_value_};` at the top of the module.
- [x] AC-2: `std::ptr::fn_addr_eq(crate::clip_grad_norm_::<f32>,
  ferrotorch_nn::clip_grad_norm_::<f32>)` is `true` (deduplication
  discriminator).
- [x] AC-3: L2 / L1 / Inf norm clipping all produce the documented
  scale coefficients on the canonical test cases.
- [x] AC-4: `clip_grad_value_` clamps `10.0 → 1.0` and `-10.0 → -1.0`
  with `clip_value = 1.0`.
- [x] AC-5: No-gradient parameter returns `total_norm = 0.0`.
- [x] AC-6: All re-exported function pointers compile with both
  `<f32>` and `<f64>` instantiations.

## Architecture

### Re-exports (REQ-1, REQ-2, REQ-6)

At `ferrotorch-train/src/grad_utils.rs:23` `pub use
ferrotorch_nn::utils::{clip_grad_norm_, clip_grad_value_};`. This is
the entire production surface of the module — there is no extra
wrapping, no extra parameter, no extra error handling. The function
pointer that `ferrotorch_train::clip_grad_norm_::<f32>` resolves to
is byte-identical to `ferrotorch_nn::clip_grad_norm_::<f32>`. The
device-dispatch policy (CPU / CUDA f32 / CUDA f64 / mixed-device
error) lives entirely in `ferrotorch_nn::utils`; see that module's
design doc.

### Deduplication discriminator (REQ-2)

The two tests
`train_clip_grad_norm_is_nn_clip_grad_norm` (line 277) and
`train_clip_grad_value_is_nn_clip_grad_value` (line 295) use
`std::ptr::fn_addr_eq` to assert that the train-crate and nn-crate
function pointers resolve to the same symbol. If a future change
reintroduces a wrapper or fork, these tests fail loudly. This is the
structural guard against the re-emergence of #1104's
duplicate-and-drift bug.

### CPU semantics (REQ-3, REQ-4, REQ-5)

The behavioral tests at lines 57-251 pin the CPU semantics through
the re-exported path. They originally pinned the train-fork CPU
behavior and now continue to pin the canonical impl. Examples:

- **L2 clip when above** (line 58): `grad = [3, 4]`, `max_norm = 2.5`
  → `total_norm = 5.0`, `clip_coef = 0.5`, clipped grad = `[1.5,
  2.0]`.
- **L2 no-clip below** (line 89): `grad = [0.1, 0.2]` →
  `total_norm ~= 0.2236`, unchanged.
- **L1** (line 130): `grad = [3, -4]`, `max_norm = 3.5`, L1 norm =
  `7.0`, clipped to `[1.5, -2.0]`.
- **Inf** (line 148): `grad = [3, -7]`, `max_norm = 3.5`, inf norm =
  `7.0`, clipped to `[1.5, -3.5]`.
- **No gradients** (line 166): `total_norm = 0.0`, parameters
  unchanged.
- **`clip_grad_value_` clamps** (line 195): `grad = [10, -10, 0.5]`,
  `clip_value = 1.0` → `[1.0, -1.0, 0.5]`.

The previous train-fork used `(total_norm + 1e-6)` for the clip
coefficient; the canonical nn impl omits the epsilon. The difference
is < 1e-7 and within the tests' 1e-4 tolerance (the comment at line
73-75 documents this).

### Non-test production consumers

- `ferrotorch-train/src/lib.rs:179` `pub use grad_utils::{clip_grad_norm_,
  clip_grad_value_};` re-exports the names at the crate root, so
  external callers can `use ferrotorch_train::clip_grad_norm_;`.
- No in-tree production caller invokes the clipping helpers today;
  the canonical `ferrotorch_nn::clip_grad_norm_` IS exercised
  elsewhere (in the `ferrotorch_nn` test suite), and the
  `ferrotorch_train` re-export is a thin alias for users translating
  `torch.nn.utils.clip_grad_norm_` ↔ `torch.nn.utils import ...`
  patterns. Open prereq blocker #1503 covers wiring
  `clip_grad_norm_` into a real production training loop (the
  `Learner::fit` body or the multi-epoch dump example).

## Parity contract

`parity_ops = []`. The numerical contract is owned by
`ferrotorch_nn::utils`'s design doc. Edge cases this module owns:

- **Re-export drift**: enforced by the `fn_addr_eq` tests at lines
  277 and 295. Any wrapping reintroduction is caught at compile-test
  time.
- **No-gradient parameters**: returned `total_norm = 0.0` rather
  than erroring. Matches PyTorch's permissive behavior.

## Verification

12 unit tests in `mod tests` (lines 37-307) cover the CPU semantics +
deduplication discriminators.

Smoke command:

```bash
cargo test -p ferrotorch-train --lib grad_utils:: 2>&1 | tail -3
```

Expected: 12 passed, 0 failed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub use ferrotorch_nn::utils::{clip_grad_norm_, clip_grad_value_};` at `ferrotorch-train/src/grad_utils.rs:23`; non-test consumer: `ferrotorch-train/src/lib.rs:179` `pub use grad_utils::{clip_grad_norm_, clip_grad_value_};` re-exports the names at the crate root for external callers — the re-export ladder IS the production consumer of the inner re-export. |
| REQ-2 | SHIPPED | impl: structural — there is no wrapping; the `pub use` IS the deduplication; non-test consumer: the test guards at `ferrotorch-train/src/grad_utils.rs:277, 295` use `std::ptr::fn_addr_eq` to assert the symbol identity; the production usage at `lib.rs:179` consumes the deduplicated re-export. |
| REQ-3 | SHIPPED | impl: behavioral contract is owned by `ferrotorch_nn::utils::clip_grad_norm_`; this module is a re-export. Non-test consumer: same `lib.rs:179` re-export ladder. NOTE: the production-fit-loop consumer (a `Learner::fit` body that calls `clip_grad_norm_` between backward and step) is the open prereq covered by blocker #1503. |
| REQ-4 | SHIPPED | impl: `clip_grad_value_` re-exported from `ferrotorch_nn::utils`; non-test consumer: same `lib.rs:179` ladder; same production-fit-loop gap covered by blocker #1503. |
| REQ-5 | SHIPPED | impl: no-grad handling is owned by `ferrotorch_nn::utils`; non-test consumer: same `lib.rs:179` ladder; behavior pinned by the test at `grad_utils.rs:166`. |
| REQ-6 | SHIPPED | impl: device dispatch is owned by `ferrotorch_nn::utils`; non-test consumer: same `lib.rs:179` ladder. The dispatch policy itself has its own design doc + tests in `ferrotorch_nn`. |
