# ferrotorch-train — `AmpContext` + autocast / scaler re-exports

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/amp/__init__.py
  - torch/amp/autocast_mode.py
  - torch/amp/grad_scaler.py
-->

## Summary

`ferrotorch-train/src/amp.rs` glues `ferrotorch_core::autograd::
autocast` (the autocast context that flips reduced-precision dispatch
on for matmul/conv/linear and keeps softmax/norms/losses in f32) and
`ferrotorch_optim::GradScaler` (the dynamic loss scaler) into a single
`AmpContext<T>` struct that drives the full AMP training step. Mirrors
PyTorch's `torch.amp` namespace: `torch.amp.autocast`,
`torch.amp.GradScaler`, and the recipe documented in
`torch/amp/__init__.py:1-50`.

## Requirements

- REQ-1: Module re-exports the autocast primitives from
  `ferrotorch_core::autograd::autocast` (`AutocastDtype`, `autocast`,
  `autocast_dtype`, `is_autocast_enabled`) and the autocast-op helpers
  from `ferrotorch_core::autograd::autocast_ops`
  (`AutocastCategory`, `autocast_category`, `autocast_guard`,
  `should_cast_to_reduced`, `should_keep_full_precision`). Mirrors
  `from torch.amp import autocast`.
- REQ-2: Module re-exports `GradScaler`, `GradScalerConfig`,
  `GradScalerState` from `ferrotorch_optim`. Mirrors `from torch.amp
  import GradScaler`.
- REQ-3: `pub struct AmpContext<T: Float>` holds an `AutocastDtype`
  and a `GradScaler<T>`. Constructor `new(dtype, scaler_config)`.
- REQ-4: `AmpContext::autocast_forward<F, R>(&self, f: F) -> R where F:
  FnOnce() -> R` runs `f` inside `autocast(self.dtype, f)` so the
  closure observes the reduced-precision dispatch policy.
- REQ-5: `AmpContext::backward_step(&mut self, loss, optimizer) ->
  FerrotorchResult<bool>` runs the full post-forward AMP pipeline:
  `scale(loss)` → `backward` → `step(optimizer)` (returns `false` on
  inf/NaN) → `update()` → `optimizer.zero_grad()`. Mirrors the
  `scaler.scale → scaler.step → scaler.update` recipe documented at
  `torch/amp/__init__.py:1-50`.
- REQ-6: Accessor surface: `scaler()`, `scaler_mut()`, `dtype()`,
  `get_scale()`, `is_enabled()`, `scaler_state_dict()`,
  `load_scaler_state_dict()`. The state-dict round trip lets a user
  checkpoint and restore the scale factor between runs.

## Acceptance Criteria

- [x] AC-1: All 4 autocast names re-export and resolve from the crate
  root via `ferrotorch_train::amp::autocast` etc.
- [x] AC-2: `GradScaler` / `GradScalerConfig` / `GradScalerState`
  re-export.
- [x] AC-3: `AmpContext::new(F16, GradScalerConfig::default())`
  constructs with `dtype() == F16`, `is_enabled() == true`,
  `get_scale() == 65536.0`.
- [x] AC-4: `autocast_forward(is_autocast_enabled)` returns `true`
  inside the closure and the global state is `false` after.
- [x] AC-5: `scaler_state_dict` / `load_scaler_state_dict` round-trip
  the scale factor.
- [x] AC-6: A disabled-config `AmpContext` reports `is_enabled() ==
  false`.

## Architecture

### Re-exports (REQ-1, REQ-2)

At `ferrotorch-train/src/amp.rs:48-55`. The re-exports are
`pub use` lines that surface the autocast primitives + scaler types at
`ferrotorch_train::amp::*`. This keeps `torch.amp`-using PyTorch code
translatable to `use ferrotorch_train::amp::*;` with the same names.

### `AmpContext<T>` (REQ-3, REQ-4, REQ-5, REQ-6)

The struct at lines 75-80 owns:
- `dtype: AutocastDtype` — the reduced-precision dtype (`F16` or
  `BF16`) the autocast region uses.
- `scaler: GradScaler<T>` — the dynamic loss scaler. Always present
  (the disabled-config path just turns the scaler into a passthrough
  no-op).

`new(dtype, scaler_config)` at line 89 constructs the scaler via
`GradScaler::new(scaler_config)`.

`autocast_forward<F, R>` at line 101-106 invokes `autocast(self.dtype,
f)` which enters the autocast context for the closure's duration.
Mirrors `with autocast(dtype=...):` in PyTorch.

`backward_step` at lines 119-134 is the canonical AMP post-forward
recipe:
1. `let scaled_loss = self.scaler.scale(loss)?;`
2. `scaled_loss.backward()?;`
3. `let stepped = self.scaler.step(optimizer)?;` — internally
   unscales and either steps or skips.
4. `self.scaler.update();` — dynamically tune the scale.
5. `optimizer.zero_grad()?;`
6. Return `stepped` so the caller can track skipped-step counts.

The contract matches `torch/amp/__init__.py:1-50` which documents:
```
scaler = torch.amp.GradScaler()
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input)
    loss = loss_fn(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
```
ferrotorch's `AmpContext` consolidates the boilerplate into
`autocast_forward(...)` + `backward_step(...)`.

### Accessors (REQ-6)

`scaler()` / `scaler_mut()` (lines 137-144) for direct field access;
`dtype()` (line 147); `get_scale()` (line 152) and `is_enabled()`
(line 157) forward to the scaler; `scaler_state_dict()` (line 162) /
`load_scaler_state_dict()` (line 167) for checkpoint round-trips.

### Non-test production consumers

- The `AmpContext` is a self-contained convenience wrapper. The
  `GradScaler` re-exports it surfaces are themselves consumed by
  `ferrotorch-train/src/learner.rs:29` (`use ferrotorch_optim::
  grad_scaler::GradScaler;`) and `:122` (`Learner::with_grad_scaler`).
  The autocast re-exports (`AutocastDtype`, `autocast`) are consumed
  by `ferrotorch-train/src/amp.rs:89, 105` themselves — the
  `AmpContext` IS the production consumer of the re-exports it
  declares.
- No out-of-tree production caller constructs an `AmpContext`
  directly today; the `Learner` accepts a `GradScaler` instead of an
  `AmpContext`. Open prereq blocker #1501 covers wiring
  `AmpContext` into the `Learner` (so `Learner::with_amp(ctx)` becomes
  the canonical attachment surface, replacing `with_grad_scaler`).

## Parity contract

`parity_ops = []`. The autocast dispatch rules + the scaler step/skip
behavior are owned by `ferrotorch-core/autograd::autocast` and
`ferrotorch-optim/grad_scaler` respectively; their design docs hold
the per-op parity contract. Edge cases the `AmpContext` itself owns:

- **Disabled scaler** (`scaler_config.enabled = false`): the scaler's
  `scale` / `step` / `update` are passthrough no-ops; `is_enabled()`
  returns `false`. The autocast context is still entered, so the
  reduced-precision dispatch policy still flips on.
- **`backward_step` inf/NaN**: `scaler.step` returns `false`, the
  optimizer is NOT stepped, `zero_grad` is still called. The caller
  observes `false` and bumps a skipped-step counter.
- **`load_scaler_state_dict` round-trip**: writes the `scale_factor`
  + `growth_tracker` from a `GradScalerState` struct back into the
  scaler. Tested by `test_scaler_state_dict_roundtrip` at line 241.

## Verification

Unit tests in `mod tests` (lines 176-275):
- `test_autocast_reexported` / `test_autocast_category_reexported`
  pin the re-export resolutions.
- `test_amp_context_construction` / `test_amp_context_disabled`
  pin the construction + disabled-mode reading.
- `test_autocast_forward_enables_autocast` /
  `test_autocast_forward_sets_dtype` pin the autocast-context entry
  semantics.
- `test_scaler_state_dict_roundtrip` pins the checkpoint round-trip.
- `test_scaler_accessor` / `test_scaler_mut_accessor` pin the
  accessor surface.

Smoke command:

```bash
cargo test -p ferrotorch-train --lib amp:: 2>&1 | tail -3
```

Expected: > 7 passed, 0 failed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub use ferrotorch_core::autograd::autocast::{AutocastDtype, autocast};` etc. at `ferrotorch-train/src/amp.rs:49-54`; non-test consumer: `ferrotorch-train/src/amp.rs:89, 105` constructs `AmpContext` with `AutocastDtype` and invokes `autocast(self.dtype, f)` — same-file production consumer. |
| REQ-2 | SHIPPED | impl: `pub use ferrotorch_optim::{GradScaler, GradScalerConfig, GradScalerState};` at `ferrotorch-train/src/amp.rs:55`; non-test consumer: `ferrotorch-train/src/amp.rs:79, 92, 162, 167` use `GradScaler` / `GradScalerConfig` / `GradScalerState` as struct field type, constructor arg, return type, and parameter type respectively. The same names are independently used by `ferrotorch-train/src/learner.rs:29` (`use ferrotorch_optim::grad_scaler::GradScaler;`) at the `Learner::with_grad_scaler` attachment surface. |
| REQ-3 | NOT-STARTED | open prereq blocker #1501 — `pub struct AmpContext<T: Float>` at `ferrotorch-train/src/amp.rs:75-80` and `new` at `:89-94` are shipped on the public surface, but no production caller constructs an `AmpContext` today; the `Learner` accepts `GradScaler` directly via `with_grad_scaler`. |
| REQ-4 | NOT-STARTED | open prereq blocker #1501 — `autocast_forward` at `ferrotorch-train/src/amp.rs:101-106` is shipped on the public surface but unused outside unit tests; no production caller wraps a forward pass in an `AmpContext` today. |
| REQ-5 | NOT-STARTED | open prereq blocker #1501 — `backward_step` at `ferrotorch-train/src/amp.rs:119-134` is shipped but unconsumed by production code; the equivalent step sequence at `ferrotorch-train/src/learner.rs:272-281` is a parallel implementation that does not invoke `AmpContext::backward_step`. |
| REQ-6 | NOT-STARTED | open prereq blocker #1501 — all 7 accessors at `ferrotorch-train/src/amp.rs:137-169` are shipped but unconsumed by production code; same root cause as REQ-3..5. |

