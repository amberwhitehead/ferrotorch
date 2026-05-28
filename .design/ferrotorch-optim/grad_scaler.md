# ferrotorch-optim — `GradScaler` (automatic mixed-precision loss scaler)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/amp/grad_scaler.py
  - torch/cuda/amp/grad_scaler.py
  - torch/optim/optimizer.py
-->

## Summary

`ferrotorch-optim/src/grad_scaler.rs` implements `GradScaler<T>`, the
dynamic loss-scaling helper for mixed-precision training. The scaler
multiplies the loss by a large constant before `backward()` so that
half-precision gradients survive quantisation, then divides
gradients by the same constant before the optimizer step. If any
gradient element is non-finite (inf / NaN), the optimizer step is
skipped and the scale is reduced; consecutive healthy steps grow the
scale. Mirrors `torch.amp.GradScaler`
(`torch/amp/grad_scaler.py:53`, the modern non-deprecated entry
point; `torch.cuda.amp.GradScaler` at `torch/cuda/amp/grad_scaler.py`
is a deprecated thin wrapper around it).

Three public types are exposed: `GradScalerConfig` (builder-style
hyperparameter struct), `GradScalerState` (checkpoint round-trip
type), and `GradScaler<T>` (the stateful scaler itself).

## Requirements

- REQ-1: `pub struct GradScalerConfig` — the hyperparameter
  bundle. Fields `init_scale: f64` (default `65536.0` =
  `2.0**16`), `growth_factor: f64` (default `2.0`),
  `backoff_factor: f64` (default `0.5`),
  `growth_interval: usize` (default `2000`), `enabled: bool`
  (default `true`). Matches `torch.amp.GradScaler.__init__`
  kwargs (`torch/amp/grad_scaler.py:123-131`).
- REQ-2: `GradScalerConfig` `#[derive(Debug, Clone, Copy)]` and
  `#[non_exhaustive]` so new defaults can be added without a
  breaking change. Five builder-style setters
  (`with_init_scale`, `with_growth_factor`, `with_backoff_factor`,
  `with_growth_interval`, `with_enabled`).
- REQ-3: `pub struct GradScalerState { scale_factor: f64,
  growth_tracker: usize }` — checkpoint-serializable state.
  `#[derive(Debug, Clone, PartialEq)]`. Mirrors PyTorch's
  scale/growth pair stored as scalar tensors
  (`torch/amp/grad_scaler.py:175-179`).
- REQ-4: `pub struct GradScaler<T: Float>` — the stateful
  scaler. Holds `scale_factor`, `growth_tracker`, `config`,
  `found_inf` (set by `unscale_`), and `already_unscaled` (idempotency
  flag).
- REQ-5: `pub fn new(config) -> Self` — builds a scaler with
  `scale_factor = config.init_scale` and `growth_tracker = 0`.
- REQ-6: `pub fn scale(&self, loss) -> FerrotorchResult<Tensor<T>>`
  — multiplies the loss by `scale_factor` via
  `ferrotorch_core::grad_fns::arithmetic::mul` (so the autograd
  graph carries the scaled gradient down through `backward()`).
  When `config.enabled == false`, returns `loss.clone()`
  unchanged. Mirrors `torch.amp.GradScaler.scale`
  (`torch/amp/grad_scaler.py:193-240`).
- REQ-7: `pub fn unscale_(&mut self, optimizer) ->
  FerrotorchResult<()>` — divides every parameter's gradient by
  `scale_factor` and sets `found_inf = true` if any element is
  not finite. Idempotent within a step: a second call before
  `update()` is a no-op. Mirrors
  `torch.amp.GradScaler.unscale_`
  (`torch/amp/grad_scaler.py:280-340`). GPU-f32 fast path uses
  `gpu_backend().scale_f32()` + `sum_f32()` for on-device scaling +
  inf check; other dtype/device combinations fall back to CPU
  data round-trip.
- REQ-8: `pub fn step(&mut self, optimizer) ->
  FerrotorchResult<bool>` — calls `unscale_` if not already
  done, then conditionally calls `optimizer.step()` only if
  `found_inf == false`. Returns `true` when the step was taken,
  `false` when skipped. Mirrors `torch.amp.GradScaler.step`
  (`torch/amp/grad_scaler.py:380-440`).
- REQ-9: `pub fn update(&mut self)` — when `found_inf == true`,
  multiplies `scale_factor` by `backoff_factor` and resets
  `growth_tracker`; otherwise increments `growth_tracker` and,
  once it hits `growth_interval`, multiplies `scale_factor` by
  `growth_factor` and resets. Per-step flags (`found_inf`,
  `already_unscaled`) are cleared at the end. Mirrors
  `torch.amp.GradScaler.update`
  (`torch/amp/grad_scaler.py:540-590`).
- REQ-10: `pub fn get_scale(&self) -> f64`, `pub fn is_enabled(&self)
  -> bool` — diagnostic accessors.
- REQ-11: `pub fn state_dict(&self) -> GradScalerState` /
  `pub fn load_state_dict(&mut self, state)` — checkpoint
  round-trip. Mirrors
  `torch.amp.GradScaler.state_dict` /
  `load_state_dict`.
- REQ-12: When `config.enabled == false`, the scaler is a
  pass-through: `scale` returns the loss unchanged, `unscale_` is
  a no-op, `step` directly calls `optimizer.step()` and returns
  `Ok(true)`, `update` only resets per-step flags. This lets a
  training loop set `enabled = !is_amp_active` once and keep the
  same code path for FP32 + AMP training.

## Acceptance Criteria

- [x] AC-1: `GradScalerConfig::default()` matches PyTorch defaults
  (`init_scale=65536`, `growth_factor=2.0`, `backoff_factor=0.5`,
  `growth_interval=2000`, `enabled=true`).
- [x] AC-2: Five builder setters compile and return `Self`.
- [x] AC-3: `scaler.scale(loss)` with `init_scale=1024` returns
  `loss * 1024`.
- [x] AC-4: `scaler.unscale_` divides gradients by `scale_factor`.
- [x] AC-5: Inf gradient ⇒ `step` returns `Ok(false)` and
  `optimizer.step()` is NOT called.
- [x] AC-6: NaN gradient ⇒ `step` returns `Ok(false)`.
- [x] AC-7: After `growth_interval` consecutive healthy steps,
  `scale_factor *= growth_factor`.
- [x] AC-8: Inf in the middle of a healthy run resets
  `growth_tracker` to 0 (the next growth waits a full interval).
- [x] AC-9: `enabled=false` makes `scale` a no-op, `step` always
  calls `optimizer.step()`.
- [x] AC-10: `state_dict` round-trips `scale_factor` +
  `growth_tracker`.
- [x] AC-11: `unscale_` is idempotent within a step (second call
  does not re-divide).

## Architecture

### `scale` (REQ-6)

```text
scale(loss):
  if !enabled: return loss.clone()
  factor_tensor = scalar(scale_factor as T)
  return mul(loss, factor_tensor)
```

`mul` is the autograd-aware multiplication, so the returned
tensor's autograd graph chains through the constant scale. After
`scaled.backward()`, every model parameter's gradient is the
unscaled gradient multiplied by `scale_factor` — exactly the
"large-multiplier-before-quantization-to-half" idiom AMP requires.

### `unscale_` (REQ-7)

The two paths:

1. **GPU f32 fast path** (`T == f32 && grad.is_cuda() &&
   gpu_backend().is_some()`):
   - `scaled = backend.scale_f32(grad, 1.0/scale_factor)` —
     on-device elementwise multiply.
   - `sum = backend.sum_f32(scaled, numel)` — full reduction.
   - `sum_bytes = gpu_to_cpu(sum)` — 4-byte download.
   - If `!sum.is_finite()`, set `found_inf = true`. (Any inf or
     NaN element makes the sum non-finite — the cheapest GPU
     check that does not need a dedicated reduction kernel.)
   - Write the scaled tensor back as the new `.grad`.

2. **CPU / non-f32 / non-CUDA fallback**:
   - `grad_data = grad.data_vec()` — downloads all values.
   - Per-element multiply by `inv_scale = 1.0 / scale_factor`.
   - Per-element `if !val.is_finite(): found_inf = true`.
   - Build a new tensor via `TensorStorage::on_device(...)`.

The CPU download in the fallback is documented as PyTorch-matching
behaviour — `torch.amp.GradScaler.unscale_` also synchronizes for
the inf check (`torch/amp/grad_scaler.py:340-360` walks the
foreach helpers that aggregate `found_inf` across devices).

**Idempotency** (`already_unscaled` flag): the second call within a
step returns early without re-dividing. `update()` clears the
flag.

**No short-circuit on inf**: when the first parameter group has
inf, the remaining groups are still processed (matches PyTorch's
GradScaler, which always unscales everything before reporting
`found_inf`). This makes the post-step gradient consistent across
groups for diagnostics.

### `step` (REQ-8)

```text
step(optimizer):
  if !enabled: optimizer.step(); return Ok(true)
  if !already_unscaled: unscale_(optimizer)
  if found_inf: return Ok(false)
  optimizer.step()
  return Ok(true)
```

The `Ok(bool)` return tells the caller whether to also call the
LR scheduler (PyTorch convention: don't advance the LR schedule
on a skipped step; ferrotorch's `ferrotorch-train::Learner`
follows the same pattern).

### `update` (REQ-9)

```text
update():
  if !enabled:
    found_inf = false
    already_unscaled = false
    return
  if found_inf:
    scale_factor *= backoff_factor
    growth_tracker = 0
  else:
    growth_tracker += 1
    if growth_tracker >= growth_interval:
      scale_factor *= growth_factor
      growth_tracker = 0
  found_inf = false
  already_unscaled = false
```

Matches `torch.amp.GradScaler.update`'s
backoff/growth/reset semantics
(`torch/amp/grad_scaler.py:540-590`).

### `state_dict` / `load_state_dict` (REQ-11)

```rust
pub struct GradScalerState {
    pub scale_factor: f64,
    pub growth_tracker: usize,
}
```

The two scalars are all the state needed; `config` is rebuilt
fresh at construction, so it's not serialized — the user is
expected to construct a `GradScalerConfig` matching the original
training run before calling `load_state_dict`. This matches
PyTorch's `state_dict()` which stores `_scale` and
`_growth_tracker` (the buffers) but not the
init / growth / backoff / interval / enabled kwargs (those are
constructor args).

### Disabled-mode passthrough (REQ-12)

The `enabled == false` branches in every method make the scaler a
true no-op:
- `scale`: returns `loss.clone()` unchanged.
- `unscale_`: returns `Ok(())` without iterating.
- `step`: calls `optimizer.step()` directly, returns `Ok(true)`.
- `update`: only resets per-step flags.

Same training-loop code can use `GradScalerConfig { enabled:
is_amp_active(), ..Default::default() }` and switch AMP on/off via
the flag.

### Non-test production consumers

- `ferrotorch-train/src/amp.rs` `pub use ferrotorch_optim::{GradScaler, GradScalerConfig, GradScalerState};` — re-exports the AMP surface.
- `ferrotorch-train/src/amp.rs:79` `scaler: GradScaler<T>` field on `AmpContext<T>`; line 92 `GradScaler::new(scaler_config)`; lines 137-142 `scaler()` / `scaler_mut()` accessors; lines 162-167 `scaler_state_dict()` / `load_scaler_state_dict()` delegate.
- `grad_scaler in ferrotorch-train/src/learner.rs` `use ferrotorch_optim::grad_scaler::GradScaler;`; line 66 `grad_scaler: Option<GradScaler<T>>` field; line 122 `with_grad_scaler(scaler)` builder — the central training loop consumes the scaler.

## Parity contract

`parity_ops = []`. Numerical contract:

- **Default `init_scale = 65536.0 (= 2.0**16)`**: matches
  `torch.amp.GradScaler` default
  (`torch/amp/grad_scaler.py:126`).
- **Default `growth_factor = 2.0`**: matches upstream
  (`torch/amp/grad_scaler.py:127`).
- **Default `backoff_factor = 0.5`**: matches upstream
  (`torch/amp/grad_scaler.py:128`).
- **Default `growth_interval = 2000`**: matches upstream
  (`torch/amp/grad_scaler.py:129`).
- **NaN propagation through `scale`**: `mul(loss, factor)` with
  `loss` containing NaN yields a tensor of NaNs (same as
  upstream).
- **inf detection in `unscale_`**: any element being `+inf`,
  `-inf`, or `NaN` flips `found_inf`. Matches the upstream
  `_amp_foreach_non_finite_check_and_unscale_` kernel.
- **Step skip on inf**: `optimizer.step()` is NOT called.
  Matches upstream.
- **Scale halves on inf**: `scale_factor *= 0.5` (with default
  `backoff_factor`). Matches upstream.
- **Scale doubles after 2000 healthy steps**: matches upstream
  (default `growth_interval`).
- **Growth tracker reset on inf**: tracker → 0; subsequent growth
  requires another full `growth_interval` of healthy steps.
  Matches upstream.
- **Idempotent `unscale_`**: second call within a step is a
  no-op. Matches upstream
  (`torch.amp.GradScaler.unscale_` checks the per-optimizer
  `OptState.UNSCALED` stage).
- **GPU f32 sum-based inf check**: any inf or NaN element makes
  the full-reduction sum non-finite. Correct as long as
  inf/NaN elements outnumber zero-cancelling partners in any
  ordering — which is the standard reduction-based check; the
  pathological `+inf` + `-inf` cancellation case has not been
  observed in real models.

## Verification

Eight unit tests in `mod tests` (grad_scaler.rs lines 372-775):

- `test_scale_multiplies_loss` — `loss * 1024` numerical check.
- `test_unscale_divides_gradients` — `grad / 256`.
- `test_inf_skips_step_and_halves_scale` — `+inf` ⇒ skip + halve.
- `test_nan_skips_step` — `NaN` ⇒ skip.
- `test_growth_after_healthy_interval` — 3 healthy ⇒ scale doubles.
- `test_growth_tracker_resets_on_inf` — interleaved healthy+inf+healthy.
- `test_disabled_passthrough` — `enabled=false` is no-op.
- `test_state_dict_roundtrip` — checkpoint round-trip.
- `test_unscale_idempotent` — second `unscale_` is a no-op.

Plus integration test `grad_scaler_*_matches_reference` in
`ferrotorch-optim/tests/conformance_optim_advanced.rs` (around
line 14 — module doctring listing covered consumers) that
exercises the full PyTorch reference comparison.

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib grad_scaler:: 2>&1 | tail -3
```

Expected: `9 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct GradScalerConfig` at `GradScalerConfig in ferrotorch-optim/src/grad_scaler.rs` with the five fields mirroring `torch/amp/grad_scaler.py:123-131` kwargs; non-test consumer: `ferrotorch-train/src/amp.rs` `pub fn new(dtype: AutocastDtype, scaler_config: GradScalerConfig) -> Self` takes the config. |
| REQ-2 | SHIPPED | impl: `#[derive(Debug, Clone, Copy)] #[non_exhaustive]` at `ferrotorch-optim/src/grad_scaler.rs:25-27` and five `with_*` builders on lines 57-89; non-test consumer: `ferrotorch-train/src/amp.rs:207` (`let mut config = GradScalerConfig::default();`) and 242, 259 use the builder pattern. |
| REQ-3 | SHIPPED | impl: `pub struct GradScalerState { scale_factor, growth_tracker }` at `GradScalerState in ferrotorch-optim/src/grad_scaler.rs` mirroring `torch/amp/grad_scaler.py:175-179` (the `_scale` + `_growth_tracker` buffers); non-test consumer: `_scale in ferrotorch-train/src/amp.rs` `pub fn scaler_state_dict(&self) -> GradScalerState` returns the type from `AmpContext`. |
| REQ-4 | SHIPPED | impl: `pub struct GradScaler<T: Float>` at `GradScaler in ferrotorch-optim/src/grad_scaler.rs` with all five fields; non-test consumer: `scaler in ferrotorch-train/src/amp.rs` `scaler: GradScaler<T>` field on `AmpContext<T>`; `scaler in ferrotorch-train/src/learner.rs` `grad_scaler: Option<GradScaler<T>>` field on `Learner<M, T>`. |
| REQ-5 | SHIPPED | impl: `pub fn new` at `new in ferrotorch-optim/src/grad_scaler.rs`; non-test consumer: `new in ferrotorch-train/src/amp.rs` `scaler: GradScaler::new(scaler_config)` inside `AmpContext::new`. |
| REQ-6 | SHIPPED | impl: `pub fn scale` at `scale in ferrotorch-optim/src/grad_scaler.rs` mirroring `torch/amp/grad_scaler.py:193-240`; non-test consumer: `ferrotorch-train/src/amp.rs` exposes `scaler()` (line 137) and `scaler_mut()` (line 142) so the `Learner` can call `scale(loss)` before `backward()`. |
| REQ-7 | SHIPPED | impl: `pub fn unscale_` at `unscale_ in ferrotorch-optim/src/grad_scaler.rs` (with the GPU-f32 fast path on lines 215-242 and CPU fallback on lines 246-263) mirroring `torch/amp/grad_scaler.py:280-340`; non-test consumer: `ferrotorch-train/src/learner.rs` step path invokes `self.grad_scaler.as_mut()?.unscale_(...)?` before clipping (mediated through `step()`'s internal `if !already_unscaled { unscale_ }`). |
| REQ-8 | SHIPPED | impl: `pub fn step` at `step in ferrotorch-optim/src/grad_scaler.rs` returning `FerrotorchResult<bool>` mirroring `torch/amp/grad_scaler.py:380-440`; non-test consumer: `scaler in ferrotorch-train/src/learner.rs` `use ferrotorch_optim::grad_scaler::{GradScaler, GradScalerConfig};` plus the AMP test cases at lines 834-840 — the learner consumes `scaler.step(&mut *optimizer)` via the `grad_scaler` field. |
| REQ-9 | SHIPPED | impl: `pub fn update` at `update in ferrotorch-optim/src/grad_scaler.rs` mirroring `torch/amp/grad_scaler.py:540-590`; non-test consumer: same learner consumer chain — every training step that holds a `GradScaler` invokes `update()` after `step()` (the standard AMP idiom from the doc-comment example). |
| REQ-10 | SHIPPED | impl: `pub fn get_scale` at `get_scale in ferrotorch-optim/src/grad_scaler.rs` and `pub fn is_enabled` at line 349; non-test consumer: the same learner / amp module references — `get_scale` is the canonical observability hook for scale-evolution logging. |
| REQ-11 | SHIPPED | impl: `pub fn state_dict` at `state_dict in ferrotorch-optim/src/grad_scaler.rs` and `pub fn load_state_dict` at line 362; non-test consumer: `load_state_dict in ferrotorch-train/src/amp.rs` `pub fn scaler_state_dict(&self) -> GradScalerState { self.scaler.state_dict() }` and line 167 `pub fn load_scaler_state_dict(&mut self, state: &GradScalerState) { self.scaler.load_state_dict(state); }`. |
| REQ-12 | SHIPPED | impl: `if !self.config.enabled` early-return branches in `scale` (line 181), `unscale_` (line 203), `step` (line 282), `update` (line 314); non-test consumer: `update in ferrotorch-train/src/amp.rs` doc-comment "If the [`GradScalerConfig`] has `enabled = false`, the `AmpContext` still ..." documents the consumer's reliance on the passthrough; pinned by `test_disabled_passthrough` (line 669). |
