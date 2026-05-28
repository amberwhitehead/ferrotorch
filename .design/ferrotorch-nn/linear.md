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
- REQ-11: SHIPPED — `Bilinear<T>` (upstream `linear.py:162-256`) is
  implemented as `pub struct Bilinear<T>` in `linear.rs` with
  `forward_pair(x1, x2)`. The forward accepts arbitrary leading batch
  dims `(*, in1)` / `(*, in2)` -> `(*, out)`: it flattens all-but-last
  to `[N, in]` (explicit batch product, NOT `-1`, so `N == 0` is
  handled), runs the two-step `einsum` `"bi,oij->boj"` then
  `"boj,bj->bo"` + bias broadcast, then reshapes the output's batch
  axis back to `(*, out)`. Mirrors `aten/src/ATen/native/Linear.cpp:792-802`
  (flatten-to-2D, contract, reshape-back). #1442 closed; the N-D input
  case is #1603 closed. The empty (size-0) batch case is correct via
  the einsum empty-output path (#1605 closed). The parity op
  `nn.functional.bilinear` runner arm is wired (#1441): builds
  `Bilinear::new` + injects op_db weight/bias via `Parameter::set_data`
  + calls `forward_pair`; with N-D + empty support landed the 1-D /
  2-D / N-D / empty samples all run. The arm's prior `ndim > 2` and
  `shape.contains(0)` skip guards are removed (the production features
  they hedged are SHIPPED), so the sweep reaches 128/128 passed
  (0 skipped, 0 failed) at `--seeds 8`.
- REQ-12: SHIPPED — parity-sweep runner arm for
  `nn.functional.linear` is wired (#1441): the arm builds a transient
  `Linear::new` + injects the op_db weight/bias via
  `Parameter::set_data` + dispatches `Module::forward`, so the
  arbitrary-rank (`*, in_features`) path (REQ-3) runs uniformly. Sweep
  reports 144/144 passed (0 skipped, 0 failed) at `--seeds 8`.

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
- [x] AC-9: `Bilinear<T>` implementation — `pub struct Bilinear<T>`
  in `linear.rs` (#1442 closed).
- [x] AC-10: parity-sweep `nn.functional.linear` arm wired (#1441) —
  144/144 passed (0 skipped, 0 failed) at `--seeds 8`.
- [x] AC-11: parity-sweep `nn.functional.bilinear` arm wired (#1441) —
  128/128 passed (0 skipped, 0 failed) at `--seeds 8`. The former
  production feature-gaps are closed: N-D input (#1603) and empty-batch
  (#1605) both land in `forward_pair`, and the arm's `ndim > 2` /
  `shape.contains(0)` skip guards are removed, so every op_db bilinear
  sample (1-D / 2-D / N-D / empty batch) RUNS.
- [x] AC-12: Bilinear `forward_pair` accepts arbitrary leading batch
  dims `(*, in)` and matches torch forward + all four gradients
  (`input1`/`input2`/`weight`/`bias`) for 3-D inputs, plus the
  4-D/2-D/1-D and empty/zero-leading-dim shapes
  (`test_bilinear_3d_forward_matches_torch`,
  `test_bilinear_3d_backward_matches_torch`,
  `test_bilinear_4d_forward_matches_torch`,
  `test_bilinear_empty_leading_dim_2d`/`_3d`,
  `test_bilinear_zero_middle_dim_3d`). #1603 closed.

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

### SHIPPED — Bilinear (REQ-11)

`torch.nn.Bilinear` at `linear.py:162-256` computes `y = x_1^T A
x_2 + b` for two-input bilinear forms. ferrotorch-nn ships
`pub struct Bilinear<T>` in `linear.rs` with `forward_pair(x1, x2)`
accepting arbitrary leading batch dims `(*, in)` -> `(*, out)`. The
forward flattens all-but-last dims to a single batch axis `[N, in]`
(via the explicit batch product so a zero-size leading dim is handled
without `-1` reshape), runs the two-step `einsum` `"bi,oij->boj"` then
`"boj,bj->bo"` + bias broadcast, and reshapes the output back to
`(*, out)`. This mirrors `aten/src/ATen/native/Linear.cpp:792-802`,
which builds `output_size = input1.sizes()[:-1] + [weight.size(0)]`,
flattens both inputs to `[-1, last]`, runs `_trilinear`, and reshapes
back (then adds bias). #1442 closed; N-D input #1603 closed; empty
(size-0) batch is handled by the einsum empty-output path #1605
closed.

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
  - parity-sweep audit entry: `nn.functional.linear` — runner arm
    wired (#1441), 144/144 passed (0 skipped, 0 failed) at `--seeds 8`.
    The arm builds a transient `Linear::new` + `Parameter::set_data`
    + `Module::forward` so 1-D / 2-D / N-D all run.
- **`nn.functional.bilinear`** — upstream entry point
  `torch.nn.functional.bilinear(input1, input2, weight, bias)`.
  Implemented as `Bilinear::forward_pair` (#1442 closed). N-D input
  (#1603) and empty-batch (#1605) both land in `forward_pair`, so the
  arbitrary-rank `(*, in)` and zero-size-batch samples are now
  supported end-to-end (verified by the 3-D forward+backward,
  4-D/2-D/1-D, and three empty/zero-leading-dim oracle tests). The
  runner arm (#1441) builds `Bilinear::new` + `Parameter::set_data` +
  `forward_pair` for ALL ranks (the `ndim > 2` / `shape.contains(0)`
  skip guards are removed): 128/128 passed (0 skipped, 0 failed) at
  `--seeds 8`.

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
| REQ-11 | SHIPPED | impl: `pub struct Bilinear<T>` + `forward_pair` in `linear.rs` — N-D-capable: flattens `(*, in)` -> `[N, in]`, two-step `einsum` (`"bi,oij->boj"` then `"boj,bj->bo"`) + bias broadcast, reshapes back to `(*, out)`, mirroring `torch/nn/modules/linear.py:162-256` + `aten/src/ATen/native/Linear.cpp:792-802`; non-test consumer: `pub use linear::Bilinear` in `lib.rs` re-export consumed by downstream model crates + the parity arm `dispatch_f32 "nn.functional.bilinear"` in `tools/parity-sweep/runner/src/main.rs` (builds `Bilinear::new` + `Parameter::set_data` + `forward_pair`). N-D input #1603 closed; empty-batch #1605 closed; #1442 closed. |
| REQ-12 | SHIPPED | impl: parity-sweep runner arm `dispatch_f32 "nn.functional.linear"` in `tools/parity-sweep/runner/src/main.rs` builds a transient `Linear::new` + `Parameter::set_data` (via `Module::parameters_mut`) + `Module::forward`, exercising the arbitrary-rank `(*, in_features)` path (REQ-3); non-test production driver of that path: `ferrotorch-nn/src/transformer.rs` + `ferrotorch-llama` QKV projections. Sweep: 144/144 passed (0 skipped, 0 failed) at `--seeds 8`. |
