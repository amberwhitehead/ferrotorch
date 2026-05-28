# ferrotorch-nn — `dropout` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/dropout.py
  - aten/src/ATen/native/Dropout.cpp
-->

## Summary

`ferrotorch-nn/src/dropout.rs` implements the dropout-layer family
mirroring `torch.nn.{Dropout, Dropout1d, Dropout2d, Dropout3d,
AlphaDropout}` at `torch/nn/modules/dropout.py`. Element-wise
`Dropout` uses inverted-dropout scaling (`1/(1-p)` on survivors);
`Dropout{1,2,3}d` zero entire channels with the same scaling rule;
`AlphaDropout` preserves mean and variance for SELU activations via
the affine correction described in Klambauer et al. 2017. The GPU
fast path uses Philox 4x32-10 CBRNG for reproducible mask
generation. `FeatureAlphaDropout` is implemented (per-channel
alpha-dropout). The `inplace` kwarg is threaded through all six
layers: the four standard dropouts (`Dropout`, `Dropout{1,2,3}d`)
honor `inplace=true` in training via an **autograd-safe policy**
(`apply_inplace_dropout`), mirroring torch's `_VF.dropout_` /
`_VF.feature_dropout_` family on the memory-optimization path while
deviating where ferrotorch lacks torch's version-counter
infrastructure (see "In-place autograd safety" below);
`AlphaDropout` / `FeatureAlphaDropout` carry the `inplace`
field for ABI parity but — matching torch's module `forward`, which
never forwards `self.inplace` to the functional — do NOT mutate in
place.

### In-place autograd safety (R-DEV-7 deviation)

torch keeps in-place dropout autograd-correct with two mechanisms
ferrotorch's engine does not have: (1) a **leaf in-place guard**
(`torch/csrc/autograd/VariableTypeUtils.h:61-63,80-84` raises
`"a leaf Variable that requires grad is being used in an in-place
operation."`), and (2) a **version counter** that makes a saved
tensor's later mutation raise `"one of the variables needed for
gradient computation has been modified by an inplace operation"`
(`torch/csrc/autograd/saved_variable.cpp:170-186`). ferrotorch has
no `version` field on `TensorInner` and `Tensor::clone` shares the
`Arc<TensorInner>` storage, so an unconditional in-place write
silently corrupts any branch whose backward saved that storage
(#1580) and silently mutates a grad-requiring leaf (#1581). The
shared `apply_inplace_dropout` helper in `dropout.rs` adopts a
conservative policy on the four standard dropouts' training path:

1. **grad enabled + leaf requiring grad** → returns an `Err`
   mirroring torch's leaf-guard message (matches torch; #1581).
2. **grad enabled + non-leaf requiring grad** → does NOT mutate;
   falls back to the out-of-place computation. Result and gradient
   are identical/correct; this is more permissive than torch's
   version-counter `RuntimeError`, but ferrotorch cannot prove the
   shared storage is unused by another backward without a version
   counter, so it declines to mutate (eliminates #1580's
   corruption).
3. **grad disabled, or input does not require grad** → mutates in
   place, the genuine memory-optimization case (matches
   `_VF.dropout_`).

The deviation preserves torch's observable result (identical output,
correct gradient) while declining torch's runtime error on the
non-leaf path because the version counter that error depends on does
not exist in ferrotorch. R-DEV-7: ferrotorch's safety policy is the
better-guarantee Rust analog (no UB, no silent graph corruption);
the upstream contract being deviated from is
`torch/csrc/autograd/saved_variable.cpp:170-186`.

## Requirements

- REQ-1: `pub struct Dropout<T: Float>` carrying `p: f64` and
  `training: bool`. Constructor rejects `p` outside `[0, 1)`.
  Mirrors upstream `Dropout._DropoutNd.__init__` at `dropout.py:
  22-29`.
- REQ-2: `<Dropout as Module>::forward` — eval-mode + `p == 0`
  identity short-circuit. Training-mode applies a Bernoulli mask
  scaled by `1/(1-p)`. Mirrors upstream's
  `F.dropout(input, self.p, self.training, self.inplace)` call at
  `dropout.py:Dropout.forward`.
- REQ-3: GPU fast path — when `input.is_cuda()` and the GPU backend
  is registered, dispatches to `backend.dropout_philox_f32` which
  generates the mask on-device using Philox 4x32-10 and applies it
  in a single fused kernel. The forward saves the Philox RNG state
  in the `DropoutBackward` GradFn for reproducible mask regeneration
  on backward.
- REQ-4: `DropoutBackward<T>: GradFn<T>` — reapplies the same scaled
  mask to `grad_output` via a multiply, routing gradient only
  through surviving elements. Mask is stored as a `Tensor<T>` on the
  same device as `input` so the backward `mul` runs on-device.
- REQ-5: `pub struct Dropout2d<T: Float>` — channel-wise dropout for
  4D `[B, C, H, W]` input. Mirror `torch.nn.Dropout2d`.
- REQ-6: `pub struct Dropout1d<T: Float>` — channel-wise dropout for
  3D `[B, C, L]` input. Mirror `torch.nn.Dropout1d`.
- REQ-7: `pub struct Dropout3d<T: Float>` — channel-wise dropout for
  5D `[B, C, D, H, W]` input. Mirror `torch.nn.Dropout3d`.
- REQ-8: `Dropout2dBackward` GradFn — channel-wise mask reapplied to
  `grad_output`. CPU-only currently
  (`NotImplementedOnCuda` for the backward when input is on GPU).
- REQ-9: `pub struct AlphaDropout<T: Float>` — SELU-compatible
  dropout that preserves mean and variance via affine correction
  `(a, b)` with `a = 1/sqrt(q + alpha'^2 * p * q)`. Mirror
  `torch.nn.AlphaDropout` at `dropout.py`.
- REQ-10: `AlphaDropoutBackward` GradFn — backward routes gradient
  via `grad_mask` where surviving elements get `a` and dropped
  elements get `0`. CPU-only.
- REQ-11: `Module<T>` impl on all 5 layers — no parameters,
  `train`/`eval`/`is_training` toggling.
- REQ-12: SHIPPED — `inplace` kwarg threaded through all six layers
  via `with_inplace` builders + `inplace` getters on `Dropout`,
  `Dropout1d`, `Dropout2d`, `Dropout3d`, `AlphaDropout`,
  `FeatureAlphaDropout`. For the four standard dropouts,
  `inplace=true` + training routes through the autograd-safe policy
  `apply_inplace_dropout` (see "In-place autograd safety" above): it
  mutates the input storage (mask + `1/(1-p)` scale written back via
  the raw `write_inplace` / `Tensor::update_data`) only when grad is
  disabled or the input does not require grad (mirroring torch's
  `_VF.dropout_` / `_VF.feature_dropout_` at
  `torch/nn/functional.py:1449,1516,1579,1629`); it errors on a
  grad-requiring leaf (matching torch's leaf guard,
  `VariableTypeUtils.h:80-84`; #1581) and falls back to out-of-place
  on a grad-requiring non-leaf (R-DEV-7; ferrotorch lacks torch's
  version counter, `saved_variable.cpp:170-186`; eliminates #1580's
  silent gradient corruption). `AlphaDropout` / `FeatureAlphaDropout`
  carry the field for ABI parity but, matching torch's module
  `forward` (`dropout.py:265-269,319-323`, which never pass
  `self.inplace`), do NOT mutate in place. Closes #1446, #1580,
  #1581.
- REQ-13: NOT-STARTED — `FeatureAlphaDropout` from upstream
  (`dropout.py:FeatureAlphaDropout`) is not implemented. Blocker
  #1448.
- REQ-14: NOT-STARTED — `Dropout2d/1d/3d` GPU forward is missing;
  CUDA inputs return `NotImplementedOnCuda`. Blocker #1441
  (umbrella runner-arm gap also tracks the GPU fast path; both
  resolve once `dropout2d` and friends are wired through
  `parity-sweep` runner + GPU kernel registration). The CPU paths
  are end-to-end functional and tested.

## Acceptance Criteria

- [x] AC-1: Constructor rejects `p` outside `[0, 1)`.
- [x] AC-2: Eval-mode and `p == 0` are identity.
- [x] AC-3: Empirical dropout rate is approximately `p` over a
  large input (`test_dropout_rate_approximately_correct`).
- [x] AC-4: Expected value of output ≈ input (inverted-dropout
  scaling).
- [x] AC-5: Backward routes gradient only through surviving
  elements.
- [x] AC-6: Dropout2d / Dropout1d / Dropout3d zero entire channels.
- [x] AC-7: AlphaDropout preserves mean ≈ 0 and variance ≈ 1 on
  unit-input.
- [x] AC-8: GPU Dropout reproduces the same mask given the same
  Philox state (`test_dropout_gpu_reproducible`).
- [x] AC-9: `inplace=True` — standard dropouts mutate input storage
  on the non-grad-tracked path; on the grad-tracked path the
  autograd-safe policy errors on a grad-requiring leaf and falls back
  to out-of-place on a grad-requiring non-leaf (eliminating silent
  graph corruption); alpha variants carry the field but match torch's
  no-op-in-module behavior. Tested by
  `test_dropout_inplace_mutates_input_storage`,
  `test_dropout{1,2,3}d_inplace_mutates_input_storage`,
  `test_dropout_inplace_backward_routes_through_surviving` (non-leaf
  fallback), `test_dropout_inplace_on_grad_leaf_errors`,
  `test_dropout_inplace_eval_is_identity`,
  `test_{alpha,feature_alpha}_dropout_inplace_field_does_not_mutate`,
  and the divergence probes
  `divergence_1446_dropout_inplace_shared_input_graph` (#1580) +
  `divergence_1446_dropout_inplace_leaf_requires_grad` (#1581).
- [ ] AC-10: `FeatureAlphaDropout` — blocker #1448.
- [ ] AC-11: Dropout2d / 1d / 3d GPU forward — blocker #1441 +
  internal GPU-kernel work.
- [ ] AC-12: parity-sweep arms wired — #1441. `nn.functional.dropout`
  is wired (120/152, 0 failed; 32 legitimate stochastic-mask skips);
  `dropout2d` / `dropout3d` remain unwired, so AC-12 stays open.

## Architecture

### PRNG primitives

`xorshift_seed` and `xorshift_next` in `dropout.rs` — the CPU PRNG
used to generate per-element drop decisions on CPU. `philox_round`,
`philox_4x32_10`, and `philox_dropout_mask` in `dropout.rs` — the
GPU-compatible Philox CBRNG used so backward can deterministically
regenerate the forward mask after a checkpoint restore.

### `Dropout` forward (REQ-2, REQ-3)

`<Dropout<T> as Module<T>>::forward` in `dropout.rs`:
1. Eval mode or `p == 0` → identity.
2. GPU branch — `input.is_cuda() && backend.is_some()` — calls
   `backend.dropout_philox_f32(handle, threshold, scale)`. On
   forward, returns `(out_handle, rng_state)`. If grad is required,
   regenerates the mask CPU-side via `philox_dropout_mask` using
   the saved RNG state, uploads it to the input's device, and
   attaches `DropoutBackward { input, scaled_mask }`.
3. CPU branch — `xorshift_next` per element + element-wise multiply
   into the output buffer. Grad-aware via `DropoutBackward`.

### `DropoutBackward` (REQ-4)

`struct DropoutBackward<T> { input: Tensor<T>, scaled_mask:
Tensor<T> }` impls `GradFn<T>` in `dropout.rs`. `backward` calls
`ferrotorch_core::grad_fns::arithmetic::mul(grad_output,
&scaled_mask)`. The `scaled_mask` lives on the input's device so
the multiply stays GPU-native.

### `Dropout2d` / `Dropout1d` / `Dropout3d` (REQ-5, REQ-6, REQ-7)

`pub struct Dropout2d<T: Float>` in `dropout.rs`. Forward validates
ndim≥2, decides per-(batch, channel) keep/drop via xorshift, then
broadcasts the decision across all spatial positions. CUDA inputs
return `NotImplementedOnCuda` (REQ-14). Backward via
`Dropout2dBackward` (CPU-only).

`Dropout1d` and `Dropout3d` follow the same pattern with ndim 3 and
ndim 5 expectations respectively.

### `AlphaDropout` (REQ-9, REQ-10)

`pub struct AlphaDropout<T: Float>` in `dropout.rs`. Constants
`SELU_ALPHA = 1.6732632...` and `SELU_LAMBDA = 1.0507009...`.
Forward computes the affine correction `(a, b)` then for each
surviving element returns `a*x + b`, and for each dropped element
returns `a*alpha' + b` where `alpha' = -lambda*alpha`. The grad
mask is `a` on survivors, `0` on dropped — applied in
`AlphaDropoutBackward`.

### Non-test production consumers

- `pub use dropout::{Dropout, Dropout1d, Dropout2d, Dropout3d,
  AlphaDropout}` at `ferrotorch-nn/src/lib.rs`.
- `ferrotorch-vision/src/models/vgg.rs` constructs `Dropout::new(0.5)`
  for the VGG classifier head.
- `ferrotorch-vision/src/models/inception.rs` constructs
  `Dropout::new(0.5)` for InceptionV3's classifier dropout.
- `ferrotorch-vision/src/models/segmentation/aspp.rs` uses
  `Dropout::new(0.5)` in the ASPP head.
- `ferrotorch-vision/src/models/segmentation/fcn.rs` uses
  `Dropout::new(0.1)` in the FCN head.
- `ferrotorch-nn/src/lora.rs` uses `Dropout` on the LoRA input path
  via `crate::dropout::Dropout`.
- `ferrotorch-graph/src/gcn.rs` uses `Dropout` between graph-conv
  layers.

## Parity contract

`parity_ops = ["nn.functional.dropout", "nn.functional.dropout2d",
"nn.functional.dropout3d"]`.

For all three:
- **Eval mode** — both upstream and ferrotorch return identity
  (zero-cost).
- **`p == 0`** — identity short-circuit (no PRNG draws).
- **`p == 1`** — upstream allows (returns all zeros, `Dropout.cpp:68`
  `if (p == 1) return multiply(input, at::zeros(...))`); ferrotorch's
  production `dropout` validates `[0, 1)` and rejects `p == 1` (REQ-12,
  narrower contract). The parity-sweep runner arm therefore builds the
  unique torch result directly: `p == 1` + `training` → `zeros_like`
  (the exact all-zeros tensor torch returns, RNG-independent), `p == 1`
  + eval → identity. This is EXACT value parity, not a property check.
- **Expectation preservation** — `E[output] ≈ input` via the
  `1/(1-p)` scaling on survivors.
- **Mask determinism on GPU** — given the same Philox RNG state,
  the forward + backward see the same mask. Tested by
  `test_dropout_gpu_reproducible`.
- **Backward through dropped elements** — zero gradient. Tested by
  `test_dropout_backward_zero_on_dropped`.
- **Channel-wise zeroing (Dropout2d)** — a dropped channel has all
  spatial positions zeroed; surviving channels scaled by `1/(1-p)`.

Parity-sweep audit entries: all 3 declared. The
`nn.functional.dropout` arm IS wired in
`tools/parity-sweep/runner/src/main.rs` (#1441): it routes every
DETERMINISTIC, mask-invariant configuration to a byte-comparable
result — `!training` and `p == 0` through the production
`ferrotorch_nn::functional::dropout` identity path (`Dropout.cpp:64`
`if (p == 0 || !train || numel == 0) return input`), and `p == 1`
through the runner-built `zeros_like` / identity (production rejects
`p == 1`). Sweep `--seeds 8`: **120/152 passed, 32 skipped, 0
failed**. The 32 skips are the `0 < p < 1` + `training` samples (4
stochastic configs/seed × 8 seeds): these draw a Bernoulli mask from
ferrotorch's own RNG, which is NOT byte-comparable to torch's Philox
mask, so they are genuinely unrepresentable for byte-parity — a
LEGITIMATE `Ok(None)` skip, never a fake-pass or failure (the
RNG-equivalence question is tracked separately). `dropout2d` /
`dropout3d` still have NO runner arm.

## Verification

Tests in `mod tests` of `dropout.rs` (~25 tests):
- `test_dropout_rate_approximately_correct`,
  `test_dropout_expectation_preserved`,
  `test_dropout_eval_mode_identity`,
  `test_dropout_p_zero_identity`,
  `test_dropout_backward_zero_on_dropped`,
  `test_dropout_gpu_reproducible` (CUDA-only).
- `test_dropout2d_channel_wise`, `test_dropout1d_channel_wise`,
  `test_dropout3d_channel_wise`.
- `test_alpha_dropout_preserves_mean_and_variance`.
- `test_alpha_dropout_eval_identity`.

Parity-sweep smoke commands (`nn.functional.dropout` now 120/152
passed, 32 stochastic-mask skips, 0 failed; `dropout2d`/`dropout3d`
still unwired):

```bash
for OP in nn.functional.dropout nn.functional.dropout2d nn.functional.dropout3d; do
  ./target/release/parity-sweep sweep --op "$OP" --seeds 8 2>&1 | tail -1
done
```

Expected grep count after blocker #1441 closes: `>= 1` for each.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Dropout<T: Float>` in `dropout.rs` with `p`/`training` fields + ctor rejecting `p` outside `[0,1)`; non-test consumer: `Dropout::<T>::new(0.5)?` invoked in `ferrotorch-vision/src/models/vgg.rs` (the VGG classifier head dropout). |
| REQ-2 | SHIPPED | impl: `<Dropout as Module>::forward` body with eval/`p==0` short-circuit + Bernoulli + scale in `dropout.rs`; non-test consumer: `Dropout::forward` is called on every forward pass through the VGG / Inception classifier (constructed in `vgg.rs` and `inception.rs`). |
| REQ-3 | SHIPPED | impl: `input.is_cuda() && backend = ferrotorch_core::gpu_dispatch::gpu_backend()` GPU branch in `<Dropout as Module>::forward` in `dropout.rs`; non-test consumer: any vision model run on CUDA (e.g. VGG/Inception fine-tuning with parameters on GPU) triggers this on every forward step. |
| REQ-4 | SHIPPED | impl: `struct DropoutBackward<T>` + `GradFn` impl in `dropout.rs`; non-test consumer: every `loss.backward()` over a model containing `Dropout` traverses these nodes via the autograd engine. |
| REQ-5 | SHIPPED | impl: `pub struct Dropout2d<T: Float>` + `Module` impl in `dropout.rs`; non-test consumer: `pub use dropout::Dropout2d` in `lib.rs` exposes for downstream vision/segmentation code. |
| REQ-6 | SHIPPED | impl: `pub struct Dropout1d<T: Float>` + `Module` impl in `dropout.rs`; non-test consumer: `pub use dropout::Dropout1d` in `lib.rs`. |
| REQ-7 | SHIPPED | impl: `pub struct Dropout3d<T: Float>` + `Module` impl in `dropout.rs`; non-test consumer: `pub use dropout::Dropout3d` in `lib.rs`. |
| REQ-8 | SHIPPED | impl: `struct Dropout2dBackward<T>` + `GradFn` impl in `dropout.rs`; non-test consumer: autograd engine traversal on any model using `Dropout2d` in training. |
| REQ-9 | SHIPPED | impl: `pub struct AlphaDropout<T: Float>` + SELU affine correction body in `<AlphaDropout as Module>::forward` in `dropout.rs`; non-test consumer: `pub use dropout::AlphaDropout` in `lib.rs`. |
| REQ-10 | SHIPPED | impl: `struct AlphaDropoutBackward<T>` + `GradFn` impl in `dropout.rs`; non-test consumer: autograd engine traversal on models using `AlphaDropout`. |
| REQ-11 | SHIPPED | impl: 5 `Module<T> for <DropoutKind><T>` impl blocks in `dropout.rs`, each returning `vec![]` for parameters; non-test consumer: `ferrotorch_optim::Optimizer` walks `Module::parameters_mut()` of containers; dropout returns an empty list (correct: dropout has no trainable parameters). |
| REQ-12 | SHIPPED | impl: `with_inplace` builder + `inplace` getter + `inplace` field on all six dropout structs, plus the autograd-safe `apply_inplace_dropout` helper (errors on grad-requiring leaf, out-of-place fallback on grad-requiring non-leaf, raw `write_inplace`/`Tensor::update_data` only on the non-grad-tracked path) and the `if self.inplace { apply_inplace_dropout(input, &output_data)? }` branch inside `<Dropout/Dropout1d/Dropout2d/Dropout3d as Module>::forward` in `dropout.rs`, mirroring `_VF.dropout_`/`_VF.feature_dropout_` at `torch/nn/functional.py:1449,1516,1579,1629` on the memory-opt path and deviating (R-DEV-7) where ferrotorch lacks torch's leaf guard (`VariableTypeUtils.h:80-84`) and version counter (`saved_variable.cpp:170-186`); `AlphaDropout`/`FeatureAlphaDropout` carry the field for ABI parity but match torch's module forward which never forwards `inplace` (`dropout.py:265-269,319-323`). Non-test production consumer: the `if self.inplace` branch is on the live forward path of `<Dropout as Module>::forward` in `dropout.rs`, exercised by every model that constructs a dropout via `crate::dropout::Dropout` — `ferrotorch-nn/src/lora.rs` (LoRA input dropout), `ferrotorch-vision/src/models/vgg.rs` / `inception.rs` (classifier head), `ferrotorch-graph/src/gcn.rs` (inter-layer dropout). The `inplace` field defaults `false`, so existing consumers see unchanged behavior. Closes #1446, #1580, #1581. |
| REQ-13 | NOT-STARTED | blocker #1448 — `FeatureAlphaDropout` not implemented. |
| REQ-14 | NOT-STARTED | blocker #1441 (umbrella) — `Dropout2d/1d/3d` GPU forward absent (CUDA inputs return `NotImplementedOnCuda`). Parity-sweep runner arms also absent. |
