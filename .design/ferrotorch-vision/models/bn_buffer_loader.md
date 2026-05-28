# ferrotorch-vision — `models::bn_buffer_loader` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: /home/doll/.local/lib/python3.13/site-packages/torchvision/models/
upstream-paths:
  - /home/doll/pytorch/torch/nn/modules/batchnorm.py
  - /home/doll/pytorch/torch/nn/modules/module.py
-->

## Summary

`ferrotorch-vision/src/models/bn_buffer_loader.rs` ships the BatchNorm
running-statistic (running_mean / running_var / num_batches_tracked)
post-processing pass invoked after `Module::load_state_dict(state, strict=false)`.
PyTorch stores BN buffers in `_buffers` and `Module._load_from_state_dict`
copies them automatically; ferrotorch's BN running statistics live in
`Mutex<Vec<f64>>` (for accumulator stability) and do not participate in
`named_buffers()`, so the generic loader silently drops them — leaving
every pretrained model with zeroed running_mean / unit running_var /
zero num_batches_tracked. This file closes that gap with a separate
typed-setter pass that mirrors PyTorch's
`Module._load_from_state_dict` behavior for BN modules.

## Requirements

- REQ-1: `pub fn apply_bn_buffers_from_state_dict<T: Float + 'static>(model:
  &dyn Module<T>, state_dict: &StateDict<T>) -> FerrotorchResult<()>`
  scans `state_dict.keys()` for keys ending in `.running_mean`,
  `.running_var`, or `.num_batches_tracked`; for each, locates the BN
  module at the parent path via `named_descendants_dyn()` and applies the
  typed setter (`BatchNorm{1,2,3}d::set_running_mean` / `set_running_var`
  / `set_num_batches_tracked`).
- REQ-2: The three BN buffer suffixes are exactly the set PyTorch's
  `nn.modules.batchnorm._BatchNorm` registers as buffers (see
  `torch/nn/modules/batchnorm.py`): `running_mean`, `running_var`,
  `num_batches_tracked`. The helper `split_bn_buffer_key` exposes the
  recognized set so non-BN keys (e.g. `weight`, `bias`, plain `running_*`
  on non-BN modules) are silently skipped.
- REQ-3: `running_var` rounding-noise clamp: torchvision's pretrained
  checkpoints sporadically carry `running_var` entries on the order of
  `-1e-12` (pure f32 rounding noise around zero). torchvision tolerates
  these because its BN forward uses `sqrt(var + eps)` with `eps ~ 1e-5`
  which trivially absorbs the noise. ferrotorch's
  `set_running_var` setter loudly rejects negatives, so this module
  clamps `|v| < 1e-6` negatives to zero before handoff while preserving
  loud failure for genuinely-corrupt negatives.
- REQ-4: Silent-fallback contract for the two known Phase-1A skip paths
  (Phase 1A semantics, tracked under #995 for vision models that haven't
  yet closed the `named_children` gap): a BN parent path absent from
  `named_descendants_dyn()` (model didn't override `named_children`) or a
  module that doesn't opt into `Module::as_any` is silently skipped — the
  BN module's running stats stay at construction defaults rather than
  panicking the entire load. A non-BN module opting into `as_any` and
  matching at a BN buffer path is a hard error (Phase 2 invariant).
- REQ-5: `num_batches_tracked` decoding: the safetensors loader stores
  every leaf tensor as `Tensor<T>` (typed by `T`). The decoder
  (`bn_nbt_from_slice`) requires a 1-element tensor, widens to `f64`, and
  checks for finiteness + non-negativity + integrality so malformed
  fixtures fail loudly instead of writing a nonsense counter.
- REQ-6: BN2d/1d/3d dispatch: the matched downcast walks `BatchNorm2d`
  → `BatchNorm1d` → `BatchNorm3d` in that order (frequency-first for the
  vision domain), routing to `apply_bn_suffix_{1,2,3}d` which translates
  the suffix to the typed setter call.

## Acceptance Criteria

- [x] AC-1: `split_bn_buffer_key("layer1.0.bn1.running_mean")` returns
  `Some(("layer1.0.bn1", "running_mean"))`
  (`split_recognises_three_suffixes`).
- [x] AC-2: `split_bn_buffer_key("layer1.0.bn1.weight")` returns `None`
  (`split_rejects_non_buffer_keys`).
- [x] AC-3: `apply_bn_buffers_from_state_dict` populates a BN's
  `running_mean` AND `running_var` via the typed setter
  (`loader_applies_running_mean_and_var`).
- [x] AC-4: `num_batches_tracked` 1-element tensor is decoded to `usize`
  (`loader_applies_num_batches_tracked`).
- [x] AC-5: Multi-element `num_batches_tracked` fails loudly with a
  message naming the key + the constraint
  (`loader_rejects_length_mismatch_for_nbt`).
- [x] AC-6: Negative `num_batches_tracked` fails loudly
  (`loader_rejects_negative_nbt`).
- [x] AC-7: `running_var` containing a genuinely negative element (e.g.
  `-0.1`, far below `-1e-6`) propagates the setter's rejection
  (`loader_propagates_running_var_negativity`).
- [x] AC-8: An opaque module (no `named_children` override) silently keeps
  BN running stats at construction defaults — the loader returns Ok and
  the BN's mean stays at zero (`loader_silently_skips_unreachable_paths`).
- [x] AC-9: Non-BN state-dict keys (`weight`, `bias`) are silently ignored
  (`loader_skips_non_bn_keys`).

## Architecture

`apply_bn_buffers_from_state_dict` (lines 146-239) is the single public
entry point. The implementation:

1. Build a `path → module` HashMap by walking `model.named_descendants_dyn()`
   exactly once (cost: O(num_modules); reused across all buffer keys).
2. For each key in `state_dict.keys()`:
   - `split_bn_buffer_key` to filter to the three BN-buffer suffixes.
   - Look up the parent path; silently skip if unreachable.
   - `bn_module.as_any()` to opt into the typed downcast; silently skip
     if `None`.
   - Read the value tensor's CPU slice via `Tensor::data`.
   - If suffix == `"running_var"`, run `clamp_running_var_noise` over the
     slice (clamp tiny negatives at `|v| < 1e-6` to zero).
   - Try `downcast_ref::<BatchNorm2d<T>>`, then `BatchNorm1d<T>`, then
     `BatchNorm3d<T>`; route to the matching `apply_bn_suffix_{1,2,3}d`.
   - If none match: hard error (Phase 2 invariant).

The three `apply_bn_suffix_*d` helpers (lines 256-324) are 3-arm matches
that map the suffix string to one of `set_running_mean`,
`set_running_var`, `set_num_batches_tracked` (the last decoded via
`bn_nbt_from_slice`).

`clamp_running_var_noise<T: Float>` (lines 101-119) iterates the slice,
clamps any element strictly less than zero whose absolute value is less
than `1e-6` (`RUNNING_VAR_CLAMP_TOL_F64`) to `T::zero()`, leaves everything
else alone, returns an owned `Vec<T>`. The `T::from(1e-6).unwrap_or(zero)`
fallback means a future `T` that can't represent `1e-6` defaults to no
clamping — preserving loud failure rather than over-clamping.

`bn_nbt_from_slice<T: Float>` (lines 333-360) requires `value.len() == 1`,
widens `value[0].to_f64()`, and checks `is_finite() && v >= 0.0 &&
v.fract() == 0.0` before casting to `usize`.

### Non-test production consumers

- `super::bn_buffer_loader::apply_bn_buffers_from_state_dict(&*model,
  &state_dict)?` invoked inside the `maybe_load_pretrained` closure at
  `ferrotorch-vision/src/models/registry.rs:76`. This call happens
  immediately after `model.load_state_dict(&state_dict, false)` so every
  pretrained vision model that flows through the registry's
  download-load-bind pipeline receives BN-buffer fix-up before being
  inserted into the global `REGISTRY`.
- Every `default_registry()` entry (lines 150-410 in `registry.rs`) that
  consumes pretrained weights — every ResNet variant, every detection
  head, every segmentation model — invokes the loader transitively. The
  vision-suite end-to-end pretrained inference tests are downstream
  validators of this code path's correctness.

## Parity contract

`parity_ops = []`. This file is a loader-side utility, not an op. It is
exercised end-to-end by every pretrained vision-model test that compares
ferrotorch logits against torchvision logits (the BN running statistics
are part of the eval-mode forward path; a regression here turns into a
60-99x activation-magnitude divergence, as #1141 documented).

Edge cases preserved versus PyTorch:

- **Buffer-key set**: `{running_mean, running_var, num_batches_tracked}`
  matches `nn.modules.batchnorm._BatchNorm.__init__`'s `register_buffer`
  calls.
- **`num_batches_tracked` dtype**: PyTorch stores int64; ferrotorch
  decodes via `T → f64 → usize` with finiteness, non-negativity, and
  integrality checks.
- **`strict=false` semantics**: keys in the state dict not matching any
  reachable BN parent path are silently skipped (matching the
  `strict=False` contract of `nn.Module.load_state_dict`).
- **Negative `running_var` clamp**: `|v| < 1e-6` is rounding noise;
  anything substantively negative reaches the setter and rejects.
  Torchvision absorbs the noise implicitly via `sqrt(var + eps)`.

## Verification

Tests in `mod tests` in `bn_buffer_loader.rs`:

- `split_recognises_three_suffixes`,
  `split_rejects_non_buffer_keys`.
- `loader_applies_running_mean_and_var`,
  `loader_applies_num_batches_tracked`.
- `loader_silently_skips_unreachable_paths`,
  `loader_skips_non_bn_keys`.
- `loader_rejects_length_mismatch_for_nbt`,
  `loader_rejects_negative_nbt`,
  `loader_propagates_running_var_negativity`.

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib bn_buffer_loader:: 2>&1 | tail -3
```

Expected: 9 tests pass; no `parity-sweep` ops to run.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn apply_bn_buffers_from_state_dict<T: Float + 'static>` at `apply_bn_buffers_from_state_dict in bn_buffer_loader.rs`; non-test consumer: `super::bn_buffer_loader::apply_bn_buffers_from_state_dict(&*model, &state_dict)?` at `apply_bn_buffers_from_state_dict in ferrotorch-vision/src/models/registry.rs` inside `maybe_load_pretrained`. |
| REQ-2 | SHIPPED | impl: `BN_BUFFER_SUFFIXES: &[&str] = &["running_mean", "running_var", "num_batches_tracked"]` at `BN_BUFFER_SUFFIXES in bn_buffer_loader.rs` + `split_bn_buffer_key in bn_buffer_loader.rs`; non-test consumer: invoked from `apply_bn_buffers_from_state_dict in bn_buffer_loader.rs` which is itself called from `registry.rs`. |
| REQ-3 | SHIPPED | impl: `RUNNING_VAR_CLAMP_TOL_F64: f64 = 1e-6` at `RUNNING_VAR_CLAMP_TOL_F64 in bn_buffer_loader.rs` + `clamp_running_var_noise<T: Float>` at `clamp_running_var_noise in bn_buffer_loader.rs`; non-test consumer: invoked from `apply_bn_buffers_from_state_dict in bn_buffer_loader.rs` for the `running_var` suffix branch, called from `registry.rs`. |
| REQ-4 | SHIPPED | impl: silent-fallback `let Some(bn_module) = path_to_module.get(bn_path).copied() else { continue; }` at `bn_buffer_loader.rs` and `let Some(any) = bn_module.as_any() else { continue; }` at `bn_buffer_loader.rs`; non-BN downcast-mismatch hard-error path at `bn_buffer_loader.rs`; non-test consumer: `registry.rs` invokes the same code path and tolerates the silent fallback for vision models still inside the #995 named_children rollout. |
| REQ-5 | SHIPPED | impl: `bn_nbt_from_slice<T: Float>` at `bn_nbt_from_slice in bn_buffer_loader.rs`; non-test consumer: invoked from each of `apply_bn_suffix_{1,2,3}d` at `apply_bn_suffix_ in bn_buffer_loader.rs, 292, 313` which are dispatched from `apply_bn_buffers_from_state_dict` reached via `registry.rs`. |
| REQ-6 | SHIPPED | impl: BN2d/1d/3d downcast chain at `apply_bn_suffix_ in bn_buffer_loader.rs` + `apply_bn_suffix_{1,2,3}d` helpers at `apply_bn_suffix_ in bn_buffer_loader.rs`; non-test consumer: `apply_bn_buffers_from_state_dict` invoked from `registry.rs` exercises the BN2d-first dispatch on every loaded vision model (BN2d dominates the conv-net vision domain). |
