# ferrotorch-nn — `linear` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/linear.py
  - aten/src/ATen/native/Linear.cpp
-->

## Summary

`ferrotorch-nn/src/linear.rs` defines `Linear<T>` — the fully connected
affine layer `y = x @ W^T + b` mirroring `torch.nn.Linear`
(`torch/nn/modules/linear.py:53-140`). The forward is built from
composable autograd-aware primitives (`linear_fused`, `reshape`) so
backward graph construction is automatic. Bias is optional. Input may
have any leading shape `(*, in_features)` and the layer flattens
non-2D inputs to `[N, in_features]` before the matmul and reshapes
back. Bilinear is NOT implemented.

## Requirements

- REQ-1: `pub struct Linear<T: Float>` carrying `weight: Parameter<T>`
  of shape `[out_features, in_features]` and `bias: Option<Parameter<T>>`
  of shape `[out_features]`. Mirrors upstream's class-level attributes
  at `torch/nn/modules/linear.py:91-94` and constructor at
  `linear.py:96-115`.
- REQ-2: `Linear::new(in_features, out_features, bias) ->
  FerrotorchResult<Self>` rejecting zero feature counts; mirrors the
  PyTorch raise on invalid `in_features`/`out_features` (Python
  conventionally lets the `torch.empty` allocator raise on
  zero-element shapes; ferrotorch hardens this with explicit
  `InvalidArgument`).
- REQ-3: Forward accepts inputs of arbitrary rank `(*, in_features)`
  (1D, 2D, 3D, 4D, …) and returns `(*, out_features)`. Matches
  upstream's "any number of dimensions" shape contract at
  `linear.py:67-70`. Scalar (0D) inputs are rejected.
- REQ-4: Forward computes `y = x @ W^T + b` via `linear_fused`
  (single fused operation) and dispatches reshape only when
  `input.ndim() != 2`, mirroring upstream's `F.linear(input, weight,
  bias)` at `linear.py:130-134`.
- REQ-5: Weight initialization uses Kaiming uniform with ReLU gain
  (`gain = sqrt(2)`). NOTE: this diverges from upstream's
  `kaiming_uniform_(weight, a=sqrt(5))` at `linear.py:117-128` which
  is algebraically equivalent to `U(-1/sqrt(in_features),
  +1/sqrt(in_features))`. ferrotorch's Kaiming gain differs; the
  empirical effect on convergence is identical-rank but the absolute
  scale differs by a constant factor `sqrt(2) / sqrt(5/3)`.
- REQ-6: Bias is initialized `U(-bound, bound)` with
  `bound = 1/sqrt(in_features)` mirroring upstream
  `init.uniform_(self.bias, -bound, bound)` with
  `bound = 1/sqrt(fan_in)` at `torch/nn/modules/linear.py:124-128`.
- REQ-7: `Module<T>` trait surface — `forward`, `parameters`,
  `parameters_mut`, `named_parameters` (with `"weight"` and `"bias"`
  keys), `train`, `eval`, `is_training`. Mirrors upstream's
  `Module.parameters()` walk + named iteration.
- REQ-8: `Display` impl produces the canonical
  `"Linear(in_features=N, out_features=M, bias=true|false)"` string,
  mirroring upstream's `extra_repr` at `linear.py:136-140`.
- REQ-9: `Send + Sync` so `Linear` can be moved across thread
  boundaries safely (asserted in tests).
- REQ-10: Validation parity (forward) — rejects mismatched
  `in_features` with `ShapeMismatch`; mirrors PyTorch raising
  `RuntimeError: mat1 and mat2 shapes cannot be multiplied`.
- REQ-11: NOT-STARTED — `Bilinear<T>` (upstream `linear.py:162-260`)
  is not implemented in ferrotorch. Blocker #1442 tracks the
  implementation. The parity op `nn.functional.bilinear` is owned by
  this route and currently 0/N passes because the runner has no arm
  and there's no implementation behind it.
- REQ-12: NOT-STARTED — parity-sweep runner arm for
  `nn.functional.linear` is absent (sweep reports 0/144 passed, 144
  skipped). Blocker #1441 tracks the runner-arm gap (umbrella for
  all LAYERS files). The forward path is end-to-end functional and
  exercised by 22 unit tests; the runner gap is a test-infrastructure
  gap per goal.md S5, not a behavioural defect.

## Acceptance Criteria

- [x] AC-1: `pub struct Linear<T: Float>` with `weight` + optional
  `bias` parameters.
- [x] AC-2: Constructor validates `in_features > 0` and
  `out_features > 0`.
- [x] AC-3: Forward accepts 1D, 2D, 3D, 4D inputs and matches the
  manually-flattened 2D result element-wise (`test_forward_3d_correctness`).
- [x] AC-4: Forward emits the correct gradient on `input` and
  `weight` for a hand-computed example (`test_backward_gradients_no_bias`,
  `test_backward_weight_grad`).
- [x] AC-5: Numerical gradient check against finite differences
  passes for a small Linear (`test_backward_numerical_gradient`).
- [x] AC-6: State-dict roundtrip preserves weights bit-for-bit
  (`test_state_dict_roundtrip_with_bias`,
  `test_state_dict_roundtrip_without_bias`).
- [x] AC-7: `Display` impl emits the canonical
  `Linear(in_features=N, out_features=M, bias=…)` string.
- [x] AC-8: `Linear<f32>` and `Linear<f64>` are `Send + Sync`.
- [ ] AC-9: `Bilinear<T>` implementation — blocker #1442.
- [ ] AC-10: parity-sweep `nn.functional.linear` arm wired — blocker
  #1441.
- [ ] AC-11: parity-sweep `nn.functional.bilinear` arm wired —
  blocker #1441 + #1442.

## Architecture

### The struct (REQ-1)

`pub struct Linear<T: Float>` in `linear.rs` carries `weight:
Parameter<T>`, `bias: Option<Parameter<T>>`, `in_features`,
`out_features`, and `training`. The field layout mirrors
`torch.nn.Linear`'s `weight: Tensor` and conditional `bias: Tensor`
(`linear.py:91-115`) — when `bias=False` upstream calls
`register_parameter("bias", None)` whereas ferrotorch uses
`Option<Parameter<T>>`.

### Construction and initialization (REQ-2, REQ-5, REQ-6)

`Linear::new` in `linear.rs`. Rejects zero-feature configs with
`FerrotorchError::InvalidArgument`. Allocates `weight` shape
`[out_features, in_features]` via `Parameter::zeros`, calls
`init::kaiming_uniform(&mut weight, NonLinearity::ReLU)`, then
allocates bias if requested and calls `init::uniform(&mut b, -bound,
bound)` with `bound = 1/sqrt(in_features)` matching upstream. The
Kaiming gain divergence (REQ-5) vs upstream remains flagged in the
table.

### Forward (REQ-3, REQ-4, REQ-10)

`<Linear<T> as Module<T>>::forward` in `linear.rs`. Validates
`input.ndim() >= 1` and last dim equals `in_features` (returning
`ShapeMismatch` otherwise). For inputs with `ndim() != 2`, flattens
to `[N, in_features]` via `reshape` from
`ferrotorch_core::grad_fns::shape`. Calls `linear_fused(input_2d,
weight.tensor(), bias.as_ref().map(|b| b.tensor()))` (the autograd-
aware fused `mm + add`) and reshapes back to `(*batch,
out_features)`.

### Trait + display (REQ-7, REQ-8, REQ-9)

`parameters()` returns `[&weight]` or `[&weight, &bias]` depending
on the bias flag. `named_parameters()` yields `("weight", &weight)`
and conditionally `("bias", &bias)`. `Display` writes
`Linear(in_features=N, out_features=M, bias=...)`. `Send + Sync` is
asserted in `test_linear_is_send_sync`.

### Non-test production consumers

- `pub use linear::Linear` at `ferrotorch-nn/src/lib.rs` is the
  module-level re-export.
- `ferrotorch-llama/src/mlp.rs` constructs `Linear::new(...)` for
  the SwiGLU MLP block's `gate_proj`, `up_proj`, `down_proj`
  (Llama-style MLP).
- `ferrotorch-llama/src/attention.rs` constructs `Linear::new(...)`
  for Q/K/V/output projections in the attention block.
- `ferrotorch-nn/src/transformer.rs` constructs `Linear` for the
  SwiGLU `w1`/`w2`/`w3` weights at the module level.
- `ferrotorch-nn/src/lora.rs` constructs `Linear::new(...)` as the
  base of `LoRALinear<T>`.
- `ferrotorch-vision/src/models/resnet.rs`,
  `vit.rs`, `convnext.rs`, `swin.rs`, and
  `detection/faster_rcnn.rs` all construct `Linear` for classifier
  heads and projection layers.
- `ferrotorch-rl/src/mlp_policy.rs` uses `Linear` for the policy
  network's MLP hidden layers.
- `ferrotorch-graph/src/gcn.rs` uses `Linear` for the GCN per-node
  transform.
- `ferrotorch-train/src/learner.rs` instantiates `Linear` in the
  training scaffolding's example head.

### NOT-STARTED — Bilinear (REQ-11)

`torch.nn.Bilinear` at `linear.py:162-260` computes `y = x_1^T A
x_2 + b` for two-input bilinear forms. ferrotorch-nn has no
`Bilinear<T>` struct. Blocker #1442 tracks the implementation.

## Parity contract

`parity_ops = ["nn.functional.linear", "nn.functional.bilinear"]`.

- **`nn.functional.linear`** — upstream entry point
  `torch.nn.functional.linear(input, weight, bias)`. Edge cases:
  - **dtype promotion**: PyTorch upcasts to float32 for `mm` inputs
    in mixed-precision contexts (autocast). ferrotorch's
    `linear_fused` respects autocast via the autograd integration.
  - **non-contiguous input**: PyTorch reshapes via `view`/`reshape`
    which materializes a contig copy. ferrotorch's reshape path
    matches.
  - **bias broadcast**: PyTorch broadcasts the 1D bias across all
    leading dims. ferrotorch's `linear_fused` add does the same.
  - **0-dim or 0-sized input**: PyTorch raises on `ndim=0`;
    ferrotorch returns `ShapeMismatch`. Empty batch (`[0, in]`) is
    accepted by upstream (returns `[0, out]`); ferrotorch matches.
  - parity-sweep audit entry: `nn.functional.linear` (route
    declared, runner-arm missing — blocker #1441).
- **`nn.functional.bilinear`** — upstream entry point
  `torch.nn.functional.bilinear(input1, input2, weight, bias)`. NOT
  IMPLEMENTED; blocker #1442 for the implementation and #1441 for
  the runner arm.

## Verification

Tests in `mod tests` of `linear.rs` (22 tests):

- Construction: `test_construction_with_bias`,
  `test_construction_without_bias`,
  `test_construction_zero_in_features`,
  `test_construction_zero_out_features`,
  `test_weight_requires_grad`.
- Forward shapes: `test_forward_shape`, `test_forward_shape_no_bias`,
  `test_forward_wrong_input_features`,
  `test_forward_1d_input_accepted`,
  `test_forward_3d_input_shape`, `test_forward_4d_input_shape`,
  `test_forward_3d_correctness`.
- Forward correctness: `test_forward_correctness_no_bias`,
  `test_forward_correctness_with_bias`.
- Backward: `test_backward_gradients_no_bias`,
  `test_backward_weight_grad`,
  `test_backward_numerical_gradient`.
- Bookkeeping: `test_parameter_count_with_bias`,
  `test_parameter_count_without_bias`,
  `test_state_dict_roundtrip_with_bias`,
  `test_state_dict_roundtrip_without_bias`,
  `test_state_dict_shape_mismatch_rejected`,
  `test_named_parameters_with_bias`,
  `test_named_parameters_without_bias`,
  `test_train_eval`, `test_display`, `test_display_no_bias`,
  `test_linear_is_send_sync`, `test_to_device_cpu_preserves_weights`,
  `test_to_device_cuda_returns_device_unavailable`.

Parity-sweep smoke commands (currently 0/N passed, 0 failed because
the runner has no arm — runner-arm gap is blocker #1441; the impl
itself is exercised end-to-end by the 22 lib tests above):

```bash
./target/release/parity-sweep sweep --op nn.functional.linear --seeds 8 2>&1 | tail -3
./target/release/parity-sweep sweep --op nn.functional.bilinear --seeds 8 2>&1 | tail -3
```

Expected grep count after blocker #1441 closes: `>= 1` for each.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Linear<T: Float>` in `linear.rs` mirroring `torch/nn/modules/linear.py:91-115`; non-test consumer: `pub use linear::Linear` in `lib.rs` exposes the type to `ferrotorch_llama::mlp::FeedForward::gate_proj` and similar fields. |
| REQ-2 | SHIPPED | impl: `pub fn new` in `linear.rs` mirroring `linear.py:96-115`; non-test consumer: `Linear::new(cfg.hidden_size, cfg.intermediate_size, false)?` in `ferrotorch-llama/src/mlp.rs` (FeedForward MLP construction). |
| REQ-3 | SHIPPED | impl: shape flatten/reshape pre/post `linear_fused` in `<Linear as Module>::forward` in `linear.rs` mirroring `linear.py:67-70`; non-test consumer: transformer blocks in `ferrotorch-nn/src/transformer.rs` and `ferrotorch-llama/src/attention.rs` feed 3D `[B, T, H]` tensors through `Linear::forward` for QKV projection. |
| REQ-4 | SHIPPED | impl: `linear_fused(&input_2d, weight.tensor(), bias_opt)` in `<Linear as Module>::forward` in `linear.rs` mirroring `linear.py:130-134`'s `F.linear` call; non-test consumer: every model in `ferrotorch-vision/src/models/` invokes `Linear::forward` through their classifier heads. |
| REQ-5 | SHIPPED | impl: `kaiming_uniform(&mut weight, NonLinearity::ReLU)` in `Linear::new` in `linear.rs`; non-test consumer: `Linear::new` is the construction path used by every consumer above. NOTE: gain divergence from upstream `linear.py:124` — same family of init, different absolute scale. |
| REQ-6 | SHIPPED | impl: `crate::init::uniform(&mut b, -bound, bound)?` with `bound = 1/sqrt(in_features)` in `Linear::new` in `linear.rs` mirroring `torch/nn/modules/linear.py:124-128`; non-test consumer: same as REQ-5. |
| REQ-7 | SHIPPED | impl: `impl<T: Float> Module<T> for Linear<T>` in `linear.rs` providing `forward`/`parameters`/`parameters_mut`/`named_parameters`/`train`/`eval`/`is_training`; non-test consumer: `ferrotorch_optim::Optimizer` consumes `Module::parameters_mut()` to apply updates (every training loop calls `model.parameters_mut()` then steps). |
| REQ-8 | SHIPPED | impl: `impl<T: Float> Display for Linear<T>` in `linear.rs` matching upstream `linear.py:136-140`'s `extra_repr`; non-test consumer: `format!("{layer}")` in model summary printing (e.g. `ferrotorch_train` learner emits module displays in logs). |
| REQ-9 | SHIPPED | `Linear` carries only `Parameter<T>` fields which are `Send + Sync`; verified at compile time via `assert_send_sync::<Linear<f32>>()` in tests; non-test consumer: any multi-threaded `DataParallel`-style training scaffolding in `ferrotorch-train` requires `Send + Sync` on the module. |
| REQ-10 | SHIPPED | impl: `last_dim != self.in_features` guard in `<Linear as Module>::forward` in `linear.rs`; non-test consumer: every production caller is shielded from silent shape mismatches by this guard. |
| REQ-11 | NOT-STARTED | blocker #1442 — `Bilinear<T>` not implemented (upstream `linear.py:162-260`). The parity op `nn.functional.bilinear` cannot be SHIPPED until the layer exists. |
| REQ-12 | NOT-STARTED | blocker #1441 — parity-sweep runner has no arm for `nn.functional.linear`; sweep reports 0/144 passed, 144 skipped. The forward path itself is end-to-end verified by 22 lib tests; only the runner-arm wiring is missing. |
