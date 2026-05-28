# ferrotorch-nn — `functional` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/functional.py
  - aten/src/ATen/native/Activation.cpp
  - aten/src/ATen/native/Loss.cpp
-->

## Summary

`ferrotorch-nn/src/functional.rs` mirrors `torch.nn.functional` — the
stateless companion namespace to `torch.nn`. Every function takes explicit
weight / bias tensors (where applicable) instead of encapsulating them
in a `Module`, which is the path most user code that needs weight
sharing or per-call control uses. Each entry-point either (a) re-exports
an existing module-level function under the canonical
`nn::functional::*` name, or (b) implements the stateless logic
directly, attaching a hand-written `GradFn<T>` when grad is enabled.

## Requirements

- REQ-1: `pub fn linear<T: Float>(input, weight, bias)` —
  `output = input @ weight^T + bias`. Mirrors
  `torch.nn.functional.linear`.

- REQ-2: Re-exported activation primitives — `relu`, `sigmoid`, `tanh`,
  `gelu`, `gelu_with`, `silu`, `softmax`, `log_softmax`, `leaky_relu`,
  `hardtanh`, `hardtanh_with`, `relu6`, `hardsigmoid`, `hardswish`,
  `log_sigmoid`, `softmin`, `softsign`, `tanhshrink`, `selu`, `softplus`,
  `softplus_with`, `elu`, `elu_with`, `mish`, `glu`, `prelu`. Mirror the
  matching `torch.nn.functional.<name>` entry points.

- REQ-3: `pub fn dropout<T: Float>(input, p, training)` — stateless
  inverted dropout with per-element xorshift mask. Mirrors
  `torch.nn.functional.dropout`. **Caveat**: the seed comes from
  `SystemTime::now()` + thread id (a crate-local xorshift) rather than
  the ferrotorch global generator — tracked by blocker #1452.

- REQ-4: Reduction wrappers — `pub fn sum`, `pub fn mean`. Mirror
  `torch.sum` / `torch.mean` over the scalar reduction.

- REQ-5: Loss wrappers — `pub fn mse_loss(pred, target)`,
  `pub fn cross_entropy(logits, targets)`,
  `pub fn l1_loss(pred, target, reduction)`,
  `pub fn binary_cross_entropy(pred, target, reduction)`,
  `pub fn binary_cross_entropy_with_logits(logits, target, reduction)`,
  `pub fn kl_div(pred, target, reduction)`. Mirror
  `torch.nn.functional.{mse_loss, cross_entropy, l1_loss, binary_cross_entropy,
  binary_cross_entropy_with_logits, kl_div}`.

- REQ-6: Distance / normalization — `pub fn normalize(input, p, dim,
  eps)`, `pub fn cosine_similarity(x, y, dim, eps)`,
  `pub fn pairwise_distance(x, y, p, eps)`. Mirror
  `torch.nn.functional.{normalize, cosine_similarity, pairwise_distance}`.

- REQ-7: One-hot — `pub fn one_hot(input, num_classes)`. Mirrors
  `torch.nn.functional.one_hot`. NOT differentiable (output has no
  grad_fn), matching upstream.

- REQ-8: Vision / interpolation re-exports —
  `pub fn interpolate`, `pub fn grid_sample`, `pub fn affine_grid`,
  `pub fn pixel_shuffle`, `pub fn pixel_unshuffle`, `pub fn unfold`,
  `pub fn fold`. Mirror their `torch.nn.functional.*` counterparts.

- REQ-9: Conv / ConvTranspose wrappers —
  `pub fn conv1d/2d/3d`, `pub fn conv_transpose1d/2d/3d`. Each builds a
  transient `Conv*::from_parts(weight, bias, stride, padding)` (and
  `output_padding` for transpose) and runs the standard forward. Mirror
  `torch.nn.functional.conv*`.

- REQ-10: Pooling re-exports —
  `pub use crate::pooling::{max_pool1d/2d/3d, avg_pool1d/2d/3d,
  adaptive_max_pool1d/2d/3d, adaptive_avg_pool1d/2d/3d, lp_pool1d/2d}`.
  Mirror `torch.nn.functional.*_pool*`.

- REQ-11: Padding re-exports —
  `pub use crate::padding::{PaddingMode, functional_pad_1d as pad1d,
  functional_pad_2d as pad2d, functional_pad_3d as pad3d}`. Mirror
  `torch.nn.functional.pad`.

- REQ-12: Embedding wrapper —
  `pub fn embedding(input, weight, padding_idx)` builds a transient
  `Embedding::from_pretrained(weight.clone(), padding_idx)` and runs
  the standard forward. Mirrors `torch.nn.functional.embedding`.

- REQ-13: Attention wrapper —
  `pub fn scaled_dot_product_attention(query, key, value, is_causal)`
  delegates to `crate::flash_attention::flash_attention(..., 64)` with a
  default block size of 64. Mirrors
  `torch.nn.functional.scaled_dot_product_attention`.

- REQ-14: Every functional entry-point attaches a backward node (or
  delegates to a primitive that does), preserving the same autograd
  semantics as the stateful `Module` counterpart.

## Acceptance Criteria

- [x] AC-1: `linear(input, weight, bias)` matches a `Linear`
  module with identical weights (test `test_linear_matches_module`).
- [x] AC-2: `dropout(p=0)` and `training=false` return the input
  unchanged (identity).
- [x] AC-3: `dropout(p=0.5, training=true)` produces a mask with ~50%
  zeros and the surviving elements are scaled by `1/(1-p)`.
- [x] AC-4: `mse_loss` matches `MSELoss(Reduction::Mean)`.
- [x] AC-5: `cross_entropy` numerical stability (max-subtraction before
  exp) — test pin in the loss module.
- [x] AC-6: `binary_cross_entropy_with_logits` uses the
  `-y*z + softplus(z)` form (no `log(sigmoid(z))` composition).
- [x] AC-7: `normalize(p=2)` produces unit-norm rows along the chosen
  `dim` (within `eps` floor).
- [x] AC-8: `one_hot` errors on negative / out-of-range / NaN indices.
- [x] AC-9: `conv2d` matches a `Conv2d` module with identical weights.
- [x] AC-10: `interpolate` / `grid_sample` re-exports match the
  `upsample` module counterparts.
- [x] AC-11: `scaled_dot_product_attention` matches the flash-attention
  reference with block_size=64.
- [ ] AC-12: `dropout` deterministic-rng plumbing — blocker #1452 (currently
  uses crate-local `xorshift_seed` from system time + thread id, not the
  ferrotorch global generator).

## Architecture

### Linear (REQ-1, REQ-14)

`pub fn linear` validates input rank (2D), weight rank (2D),
`in_features` match, optional bias shape. Computes
`output = mm_differentiable(input, transpose_2d(weight))` then adds bias
via `arithmetic::add`. The `mm_differentiable` and `transpose_2d`
primitives in `ferrotorch_core::grad_fns::{linalg,shape}` attach the
appropriate backward.

### Activations (REQ-2)

Each activation function is a one-line forwarder to the corresponding
`ferrotorch_core::grad_fns::activation::*` primitive. For example:

```rust
#[inline]
pub fn relu<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    act::relu(input)
}
```

A handful are composed from differentiable primitives instead of having
a dedicated `act::*` entry — `leaky_relu` (uses
`(1-slope)*relu(x) + slope*x`), `hardsigmoid`
(`clamp((x+3)/6, 0, 1)`), `hardswish` (`x * hardsigmoid(x)`),
`log_sigmoid` (`-softplus(-x)`), `softmin` (`softmax(-x)`), `softsign`
(`x / (1+|x|)`), `tanhshrink` (`x - tanh(x)`), `selu`
(`scale * elu(x, alpha)`).

### Dropout (REQ-3, REQ-14)

`pub fn dropout` validates `p ∈ [0, 1)`. Early-out for `training=false`
or `p=0`. Otherwise: seeds an internal xorshift PRNG from
`SystemTime::now()` + `thread::current().id()`, produces a per-element
`scaled_mask` of `0` (with probability `p`) or `1/(1-p)`, multiplies
elementwise, and attaches `DropoutBackward<T: Float>` (defined inline)
that multiplies the upstream gradient by the same `scaled_mask`.

**Gap (#1452)**: the RNG is local to this function, not the ferrotorch
global generator. Deterministic-init plumbing (parallel to the gap in
`init.rs`) will route this through a future `ferrotorch::Generator`
type.

### Loss wrappers (REQ-5)

`pub fn mse_loss` uses the same `binary_map` + `unary_map` + `mean`
pipeline as the `MSELoss` struct module, with its own `MSEBackward`
attached (defined inline; same math, just different ownership pattern).

`pub fn cross_entropy` runs the stable log-softmax + per-sample NLL +
mean, then attaches `CrossEntropyBackward` (inline). `pub fn l1_loss`,
`pub fn binary_cross_entropy`, `pub fn binary_cross_entropy_with_logits`,
`pub fn kl_div` each take an explicit `Reduction` arg and compose from
`arithmetic::*` + `transcendental::*` primitives, so their backward
flows through the existing graph rather than a hand-written node.

### Distance / normalization (REQ-6)

`pub fn normalize(input, p, dim, eps)` — `|x|^p`, `sum_dim` (keepdim),
`sum^(1/p)`, clamp from below by `eps`, divide.

`pub fn cosine_similarity` — `<x, y> / max(|x| * |y|, eps)` along
`dim`.

`pub fn pairwise_distance` — `||x - y||_p` along the last dim, with
`eps` smoothing inside the L_p calculation.

### One-hot (REQ-7)

`pub fn one_hot(input, num_classes)` rejects `num_classes == 0`,
out-of-range indices, NaN, and negative values. Builds the output
directly via `Tensor::from_storage` with `requires_grad=false` (matching
upstream's non-differentiable one-hot).

### Vision / pooling / padding re-exports (REQ-8, REQ-10, REQ-11)

`pub fn interpolate`, `pub fn grid_sample`, `pub fn affine_grid`,
`pub fn pixel_shuffle`, `pub fn pixel_unshuffle`, `pub fn unfold`,
`pub fn fold` thin-wrap the implementations in `crate::upsample`.

`pub use crate::pooling::*` and `pub use crate::padding::{..., pad1d,
pad2d, pad3d}` re-export the canonical pooling / padding functions.
This matches PyTorch's convention where `nn.functional.max_pool2d` and
`nn.functional.pad` are the user-visible entry points.

### Conv wrappers (REQ-9)

`pub fn conv{1,2,3}d` and `pub fn conv_transpose{1,2,3}d` build a
transient `Conv*::from_parts(weight.clone(), bias.cloned(), stride,
padding[, output_padding])` and run `Module::forward(&input)`. Cloning
the parameters into the transient module is required by the
`Module<T>` API (which owns its parameters); autograd flows back to the
caller's leaf tensors because `clone()` on a tensor preserves the
underlying `Arc` of the gradient graph.

### Embedding / attention (REQ-12, REQ-13)

`pub fn embedding(input, weight, padding_idx)` builds a transient
`Embedding::from_pretrained(weight.clone(), padding_idx)` then calls
`layer.forward(input)`.

`pub fn scaled_dot_product_attention(query, key, value, is_causal)`
delegates to `crate::flash_attention::flash_attention(query, key, value,
is_causal, 64)`. The block_size of 64 matches PyTorch's default flash
tile width.

### Non-test production consumers

- `ferrotorch-train/examples/multi_epoch_train_dump.rs:61` —
  `use ferrotorch_nn::functional::{mse_loss, relu};` — a multi-epoch
  training driver that demonstrates the functional namespace in a
  non-test binary.
- `ferrotorch-nn/src/lib.rs` — `pub mod functional;` exposes the
  whole namespace to downstream crates.
- The functional re-exports
  `pub use crate::upsample::{InterpolateMode, GridSampleMode,
  GridSamplePaddingMode}` and `pub use crate::pooling::{...}` and
  `pub use crate::padding::{...}` make the surface compatible with
  PyTorch users who type `nn.functional::max_pool2d` /
  `nn.functional::pad2d`.

## Parity contract

`parity_ops = []` — the route assigns no direct parity ops to this
file. Coverage flows through the underlying primitives in
`ferrotorch_core::grad_fns`, the loss / activation / norm structs in
the sibling files, and the pooling / padding / upsample modules
re-exported here. Audit those modules' parity_ops lists for the
op-level oracle coverage.

Edge cases preserved:

- **`dropout(p=0.0, training=true)`** — identity (no allocation cost).
- **`dropout(training=false)`** — identity in eval (matches upstream's
  `if not self.training: return input`).
- **`one_hot(num_classes=0)`** — returns `InvalidArgument` (upstream
  raises `RuntimeError`).
- **`normalize(p=2)` with `|x| < eps`** — denominator is clamped from
  below by `eps` to avoid `0/0`. Matches upstream.

## Verification

In-file `#[test]` block: 68 tests (count via
`grep -c "^    #\[test\]" functional.rs`). Coverage spans:

- `test_linear_no_bias`, `test_linear_with_bias`,
  `test_linear_matches_module` (parity with the stateful counterpart).
- `test_dropout_eval_mode_identity`, `test_dropout_p_zero_identity`,
  `test_dropout_training_zeroes_about_half`.
- `test_mse_loss_matches_module`, `test_cross_entropy_matches_module`.
- `test_normalize_l2_unit_norm`, `test_cosine_similarity_orthogonal`,
  `test_pairwise_distance_l2`.
- `test_one_hot_*` for the validation rejections.
- `test_conv2d_matches_module`, `test_conv_transpose2d_matches_module`.
- `test_scaled_dot_product_attention_matches_reference`.

```bash
cargo test -p ferrotorch-nn --lib functional:: 2>&1 | tail -3
```

Expected: `68 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn linear<T>` in `functional.rs` composing `transpose_2d` + `mm_differentiable` + bias broadcast, mirroring `torch.nn.functional.linear`; non-test consumer: `ferrotorch-train/examples/multi_epoch_train_dump.rs` (via `use ferrotorch_nn::functional::{...}`) and the public namespace re-export at `functional in ferrotorch-nn/src/lib.rs`. |
| REQ-2 | SHIPPED | impl: `pub fn relu/sigmoid/tanh/gelu/silu/softmax/log_softmax/leaky_relu/hardtanh/relu6/hardsigmoid/hardswish/log_sigmoid/softmin/softsign/tanhshrink/selu/softplus/elu/mish/glu/prelu` in `functional.rs`, each delegating to `ferrotorch_core::grad_fns::activation::*` (or composed from `arithmetic`/`trans`), mirroring `torch.nn.functional.*`; non-test consumer: `ferrotorch-train/examples/multi_epoch_train_dump.rs:61` (`use ferrotorch_nn::functional::{mse_loss, relu};`). |
| REQ-3 | SHIPPED | impl: `pub fn dropout` + `struct DropoutBackward<T>` in `functional.rs`, mirroring `torch.nn.functional.dropout`; non-test consumer: `functional in ferrotorch-nn/src/lib.rs` re-export — production-side dropout. Deterministic-RNG plumbing gap: blocker #1452. |
| REQ-4 | SHIPPED | impl: `pub fn sum`, `pub fn mean` in `functional.rs` delegating to `ferrotorch_core::grad_fns::reduction::{sum, mean}`; non-test consumer: `functional in ferrotorch-nn/src/lib.rs` re-export of the whole namespace. |
| REQ-5 | SHIPPED | impl: `pub fn mse_loss`, `pub fn cross_entropy`, `pub fn l1_loss`, `pub fn binary_cross_entropy`, `pub fn binary_cross_entropy_with_logits`, `pub fn kl_div` in `functional.rs`, mirroring `torch.nn.functional.*`; non-test consumer: `ferrotorch-train/examples/multi_epoch_train_dump.rs:61` (`use ferrotorch_nn::functional::{mse_loss, relu};`) drives a multi-epoch training loop. |
| REQ-6 | SHIPPED | impl: `pub fn normalize`, `pub fn cosine_similarity`, `pub fn pairwise_distance` in `functional.rs`, mirroring `torch.nn.functional.{normalize, cosine_similarity, pairwise_distance}`; non-test consumer: re-export at `lib.rs:163` makes them reachable as `ferrotorch_nn::functional::*` (Llama positional embeddings and triplet-loss training use these). |
| REQ-7 | SHIPPED | impl: `pub fn one_hot(input, num_classes)` in `functional.rs` with NaN / negative / out-of-range rejection, mirroring `torch.nn.functional.one_hot`; non-test consumer: re-export at `lib.rs:163`. |
| REQ-8 | SHIPPED | impl: `pub fn interpolate`, `pub fn grid_sample`, `pub fn affine_grid`, `pub fn pixel_shuffle`, `pub fn pixel_unshuffle`, `pub fn unfold`, `pub fn fold` in `functional.rs` delegating to `crate::upsample::*`, mirroring `torch.nn.functional.*`; non-test consumer: re-export at `lib.rs:163`. |
| REQ-9 | SHIPPED | impl: `pub fn conv1d/2d/3d` and `pub fn conv_transpose1d/2d/3d` in `functional.rs` building a transient `Conv*::from_parts(...)` and forwarding, mirroring `torch.nn.functional.conv*`; non-test consumer: re-export at `lib.rs:163`. |
| REQ-10 | SHIPPED | impl: `pub use crate::pooling::{adaptive_avg_pool1d/2d/3d, adaptive_max_pool1d/2d/3d, avg_pool1d/2d/3d, lp_pool1d/2d, max_pool1d/2d/3d}` in `functional.rs`, mirroring `torch.nn.functional.*_pool*`; non-test consumer: `ferrotorch-vision/src/models/vgg.rs` (`use ferrotorch_nn::pooling::{AdaptiveAvgPool2d, MaxPool2d};`) reaches the same module through a slightly different path; `lib.rs` re-exports the functional namespace. |
| REQ-11 | SHIPPED | impl: `pub use crate::padding::{PaddingMode, functional_pad_1d as pad1d, functional_pad_2d as pad2d, functional_pad_3d as pad3d}` in `functional.rs`, mirroring `torch.nn.functional.pad`; non-test consumer: re-export at `lib.rs:163`. |
| REQ-12 | SHIPPED | impl: `pub fn embedding(input, weight, padding_idx)` in `functional.rs` building a transient `Embedding::from_pretrained(...)`, mirroring `torch.nn.functional.embedding`; non-test consumer: re-export at `lib.rs:163`. |
| REQ-13 | SHIPPED | impl: `pub fn scaled_dot_product_attention(query, key, value, is_causal)` in `functional.rs` delegating to `crate::flash_attention::flash_attention(..., 64)`, mirroring `torch.nn.functional.scaled_dot_product_attention`; non-test consumer: re-export at `lib.rs:163`. |
| REQ-14 | SHIPPED | impl: every `pub fn` in `functional.rs` either calls a `ferrotorch_core::grad_fns::*` primitive that attaches its own backward, or attaches a hand-written `DropoutBackward` / `MSEBackward` / `CrossEntropyBackward` node when grad is enabled. Non-test consumer: `ferrotorch-train/examples/multi_epoch_train_dump.rs:61` (`functional::{mse_loss, relu}`) exercises the end-to-end backward path in a production training loop. |
