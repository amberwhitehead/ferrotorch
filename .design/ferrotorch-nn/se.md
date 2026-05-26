# ferrotorch-nn — `se` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/
  - aten/src/ATen/native/
-->

## Summary

`ferrotorch-nn/src/se.rs` implements the *Squeeze-and-Excitation*
(SE) block from Hu et al. (2018) — *Squeeze-and-Excitation
Networks*. Mirrors `torchvision.ops.misc.SqueezeExcitation` for the
`named_children()` order, the use of `1×1 Conv2d` (NOT `Linear`)
for both projections, and the default ReLU + Sigmoid activation
pair.

There is no `torch.nn`-side counterpart in mainline PyTorch 2.x;
upstream ships the block in torchvision. ferrotorch mirrors the
torchvision implementation as a top-level module so the
SE-equipped vision models (MobileNetV3, EfficientNet) can compose
it directly.

## Requirements

- REQ-1: `pub struct SqueezeExcitation<T: Float>` holding
  `avgpool: AdaptiveAvgPool2d`, `fc1: Conv2d<T>` (1×1 squeeze),
  `activation: Box<dyn Module<T>>`, `fc2: Conv2d<T>` (1×1
  excitation), `scale_activation: Box<dyn Module<T>>`, and a
  `training` flag. Mirrors torchvision's
  `SqueezeExcitation.__init__` member order.

- REQ-2: `SqueezeExcitation::new(input_channels,
  squeeze_channels)` — default constructor with ReLU squeeze and
  Sigmoid scale (matches torchvision's default). Rejects zero
  channel counts. Constructs the two 1×1 `Conv2d` projections with
  `bias=true` (the upstream default).

- REQ-3: `SqueezeExcitation::new_with_activations(input_channels,
  squeeze_channels, activation, scale_activation)` — allows
  swapping the inner and outer activations. MobileNetV3 uses
  `ReLU + HardSigmoid`; EfficientNet uses `SiLU + Sigmoid`. Both
  are constructible from this entry point.

- REQ-4: `forward(input)` — 4-D `[B, C, H, W]` → 4-D `[B, C, H, W]`:
  1. `avgpool` to `[B, C, 1, 1]`.
  2. `fc1 → activation → fc2 → scale_activation` produces the
     `[B, C, 1, 1]` channel gate.
  3. Broadcast multiply: `input * gate`.

- REQ-5: `Module<T> for SqueezeExcitation<T>` —
  `parameters()`/`parameters_mut()`/`named_parameters()` traverse
  `fc1` and `fc2` only (the avgpool and activations are
  parameter-free). Named keys: `fc1.weight`, `fc1.bias`,
  `fc2.weight`, `fc2.bias` — matching torchvision's
  `state_dict()` keys.

- REQ-6: `children()` / `named_children()` — return the five
  sub-modules in torchvision order: `avgpool`, `fc1`,
  `activation`, `fc2`, `scale_activation`.

- REQ-7: `train()` / `eval()` propagate to the boxed activations,
  not just to the struct's `training` flag — required because
  some activations (`Dropout`-equipped activations) depend on the
  flag.

- REQ-8: `Debug` impl that prints `fc1`, `fc2`, and `training`
  (skips the boxed activations because `dyn Module<T>` does not
  implement `Debug`).

- REQ-9: End-to-end differentiability — composed from
  `Conv2d::forward`, `AdaptiveAvgPool2d::forward`, the boxed
  activation modules, and `mul` from
  `ferrotorch_core::grad_fns::arithmetic`, all of which track
  gradients. No custom `GradFn<T>` is required.

- REQ-10: `Send + Sync` bound — both `Conv2d<T>` and the boxed
  activations must propagate `Send + Sync` so the SE block can
  compose into any model requiring those bounds.

## Acceptance Criteria

- [x] AC-1: `SqueezeExcitation::<f32>::new(16, 4)` constructs with
  4 parameter tensors (fc1.weight + fc1.bias + fc2.weight +
  fc2.bias).
- [x] AC-2: `named_parameters()` returns the four torchvision
  keys.
- [x] AC-3: `named_children()` returns the five sub-modules in
  the torchvision order.
- [x] AC-4: Forward on `[1, 16, H, W]` returns `[1, 16, H, W]`.
- [x] AC-5: With zero-initialised fc1/fc2 weights and biases, the
  gate is Sigmoid(0) = 0.5 everywhere; output = 0.5 * input.
- [x] AC-6: Backward via finite differences matches the analytic
  gradient within `1e-2` tolerance.
- [x] AC-7: `SqueezeExcitation` is `Send + Sync`.

## Architecture

### Struct (REQ-1)

`pub struct SqueezeExcitation<T: Float>` at
`pub struct SqueezeExcitation in se.rs` holds the five
sub-modules with private fields plus a `training` flag. Two of the
sub-modules are `Box<dyn Module<T>>` (the activations) to support
swapping them via `new_with_activations`.

### Constructors (REQ-2, REQ-3)

`pub fn new` at `impl SqueezeExcitation in se.rs` delegates to
`new_with_activations` with `Box::new(ReLU::new())` and
`Box::new(Sigmoid::new())`. `new_with_activations` validates the
channel counts (`> 0`), constructs the two 1×1 Conv2d layers with
`(kernel=(1,1), stride=(1,1), padding=(0,0), bias=true)`, and
constructs the `AdaptiveAvgPool2d::new((1, 1))` for the squeeze
stage.

### Forward (REQ-4, REQ-9)

`pub fn forward` at `pub fn forward in se.rs` runs:

1. `scale = avgpool.forward(input)` → `[B, C, 1, 1]`.
2. `scale = fc1.forward(scale)` → `[B, sq, 1, 1]`.
3. `scale = activation.forward(scale)` (default ReLU).
4. `scale = fc2.forward(scale)` → `[B, C, 1, 1]`.
5. `scale = scale_activation.forward(scale)` (default Sigmoid).
6. `mul(input, &scale)` → `[B, C, H, W]`.

Every operation is differentiable, so autograd traces the
backward without a custom `GradFn<T>`.

### Module impl (REQ-5, REQ-6, REQ-7)

`impl<T: Float> Module<T> for SqueezeExcitation<T>` at
`impl Module<T> for SqueezeExcitation in se.rs`:

- `parameters` / `parameters_mut` extend `fc1.parameters()` and
  `fc2.parameters()`. The two activations are parameter-free.
- `named_parameters` prefixes the fc1/fc2 keys with `fc1.` and
  `fc2.`. Result: `["fc1.weight", "fc1.bias", "fc2.weight",
  "fc2.bias"]`.
- `children` and `named_children` return the five sub-modules in
  the canonical order (`avgpool, fc1, activation, fc2,
  scale_activation`).
- `train` / `eval` set the local flag AND forward to both boxed
  activations.

### Debug impl (REQ-8)

`impl<T: Float> std::fmt::Debug for SqueezeExcitation<T>` at
`impl Debug for SqueezeExcitation in se.rs` prints `fc1`, `fc2`,
and `training`. The boxed activations are skipped because
`dyn Module<T>` does not implement `Debug`.

### Send + Sync (REQ-10)

`Send + Sync` is propagated automatically from `Conv2d<T>` and
the `Box<dyn Module<T> + Send + Sync>` bound on the activations.
The test `se_is_send_sync` pins this.

### Non-test production consumers

- `pub use se::SqueezeExcitation` at
  `ferrotorch-nn/src/lib.rs:247` — grandfathered public API
  surface.
- `ferrotorch-vision/src/models/mobilenet.rs:56` — `use
  ferrotorch_nn::se::SqueezeExcitation` in the MobileNet V3 build
  path (with `ReLU + HardSigmoid`).
- `ferrotorch-vision/src/models/efficientnet.rs:39` — `use
  ferrotorch_nn::se::SqueezeExcitation` in the EfficientNet build
  path (with `SiLU + Sigmoid`).

## Parity contract

`parity_ops = []`. The SE block composes parity-tracked primitives
(`Conv2d` per `conv.md`, `AdaptiveAvgPool2d` per `pooling.md`,
`Sigmoid` / `ReLU` / `HardSigmoid` / `SiLU` per activation parity)
but has no standalone parity oracle — the test that pins it is
`se_forward_matches_manual_composition` which compares the
primitive output to a hand-composed pipeline.

Numerical edge cases preserved:

- **Probe-before-fix reference** — with zero-initialised fc1/fc2
  weights and biases, the gate is `Sigmoid(0) = 0.5` everywhere
  and the output equals `0.5 * input`. Test
  `se_probe_handcomputed_reference` pins this.
- **`mul` broadcast semantics** — `[B, C, H, W] * [B, C, 1, 1]`
  broadcasts the gate to every spatial position per channel,
  matching upstream's `torch.Tensor.mul_` behaviour.

## Verification

Tests in `mod tests in se.rs`:

- `se_construction_smoke`.
- `se_named_parameters_match_torchvision`.
- `se_named_children_match_torchvision_order`.
- `se_forward_matches_manual_composition` — primitive vs hand-
  composed pipeline equivalence.
- `se_probe_handcomputed_reference` — zero-weight gate = 0.5.
- `se_backward_finite_differences` — FD matches analytic gradient
  within `1e-2`.
- `se_with_hardsigmoid_scale_smoke` — MobileNetV3 variant.
- `se_with_silu_sigmoid_smoke` — EfficientNet variant.
- `se_is_send_sync`.

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-nn --lib se:: 2>&1 | tail -3
```

Expected: 9 tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct SqueezeExcitation<T: Float>` in `se.rs`; non-test consumer: re-export at `ferrotorch-nn/src/lib.rs:247` + `ferrotorch-vision/src/models/mobilenet.rs:56` + `ferrotorch-vision/src/models/efficientnet.rs:39`. |
| REQ-2 | SHIPPED | impl: `pub fn new` on `SqueezeExcitation` in `se.rs` (delegates to `new_with_activations`); non-test consumer: re-export at `lib.rs:247` + the two vision-side consumers from REQ-1. |
| REQ-3 | SHIPPED | impl: `pub fn new_with_activations` on `SqueezeExcitation` in `se.rs`; non-test consumer: `ferrotorch-vision/src/models/mobilenet.rs:56` uses `HardSigmoid` scale; `efficientnet.rs:39` uses `Sigmoid`. |
| REQ-4 | SHIPPED | impl: `pub fn forward` on `SqueezeExcitation` in `se.rs`; non-test consumer: re-export at `lib.rs:247` + MobileNetV3 and EfficientNet forward paths. |
| REQ-5 | SHIPPED | impl: `impl<T: Float> Module<T> for SqueezeExcitation<T>` with `parameters`, `parameters_mut`, `named_parameters` in `se.rs`; non-test consumer: re-export at `lib.rs:247` + state-dict loading in `mobilenet.rs:56` / `efficientnet.rs:39`. |
| REQ-6 | SHIPPED | impl: `fn children` and `fn named_children` inside the Module impl in `se.rs`; non-test consumer: re-export at `lib.rs:247`. |
| REQ-7 | SHIPPED | impl: `fn train` / `fn eval` inside the Module impl in `se.rs` (forwards to both boxed activations); non-test consumer: re-export at `lib.rs:247`. |
| REQ-8 | SHIPPED | impl: `impl<T: Float> std::fmt::Debug for SqueezeExcitation<T>` in `se.rs`; non-test consumer: re-export at `lib.rs:247`. |
| REQ-9 | SHIPPED | impl: forward body composes only differentiable primitives (Conv2d, AdaptiveAvgPool2d, dyn Module activations, mul) in `se.rs`; non-test consumer: re-export at `lib.rs:247` — autograd traces through the composition without a custom GradFn. |
| REQ-10 | SHIPPED | impl: `Send + Sync` bound is automatic from Conv2d + boxed activation bounds; pinned by `se_is_send_sync` in `mod tests`; non-test consumer: re-export at `lib.rs:247`. |
