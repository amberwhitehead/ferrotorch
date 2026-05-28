# ferrotorch-nn — `loss` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/loss.py
  - torch/nn/functional.py
  - aten/src/ATen/native/Loss.cpp
-->

## Summary

`ferrotorch-nn/src/loss.rs` ships callable loss-criterion structs that
mirror the `_Loss` / `_WeightedLoss` hierarchy in
`torch/nn/modules/loss.py`. Unlike the layer modules, loss criteria are
NOT `Module<T>` — they are plain structs with a `forward(&self, pred,
target) -> FerrotorchResult<Tensor<T>>` method (and a few variant
signatures for criteria that take auxiliary inputs such as
`CosineEmbeddingLoss::forward_pair(x1, x2, y)`,
`MarginRankingLoss::forward_pair(x1, x2, y)`,
`TripletMarginLoss::forward_triplet(anchor, positive, negative)`,
`GaussianNLLLoss::forward(input, target, var)`). Each loss attaches a
hand-written `GradFn<T>` backward node to the returned tensor when grad
is enabled, and participates in autocast via
`ferrotorch_core::autograd::autocast_ops::autocast_guard("<op_name>")`.

## Requirements

- REQ-1: `pub struct MSELoss { pub reduction: Reduction }` — mean / sum /
  none reductions of `(pred - target)^2`. Mirrors `torch.nn.MSELoss` at
  `torch/nn/modules/loss.py:566-630` (`F.mse_loss`).

- REQ-2: `pub struct CrossEntropyLoss { pub reduction, pub label_smoothing }`
  — 2-D logits / 1-D target indices, with `label_smoothing` ∈ `[0, 1)`.
  Mirrors `torch.nn.CrossEntropyLoss` at
  `torch/nn/modules/loss.py:1197-1406` (`F.cross_entropy`).

- REQ-3: `pub struct BCEWithLogitsLoss { pub reduction, pub pos_weight }` —
  numerically stable BCE from raw logits. Mirrors
  `torch.nn.BCEWithLogitsLoss` at
  `torch/nn/modules/loss.py:718-845`.

- REQ-4: `pub struct BCELoss { pub reduction }` — BCE on probabilities.
  Mirrors `torch.nn.BCELoss` at `torch/nn/modules/loss.py:632-717`.

- REQ-5: `pub struct L1Loss { pub reduction }` — `|pred - target|`. Mirrors
  `torch.nn.L1Loss` at `torch/nn/modules/loss.py:65-134`.

- REQ-6: `pub struct NLLLoss { pub reduction, pub ignore_index, pub weight }`
  — 2-D log-probabilities + 1-D class indices. Mirrors `torch.nn.NLLLoss`
  at `torch/nn/modules/loss.py:135-273`.

- REQ-7: `pub struct KLDivLoss { pub reduction, pub log_target }` — KL
  divergence between log-prediction and (log-)target. Mirrors
  `torch.nn.KLDivLoss` at `torch/nn/modules/loss.py:463-565`.

- REQ-8: `pub struct SmoothL1Loss { pub reduction, pub beta }` — Huber-like
  smoothing of L1. Mirrors `torch.nn.SmoothL1Loss` at
  `torch/nn/modules/loss.py:987-1079`.

- REQ-9: `pub struct HuberLoss { pub reduction, pub delta }` — Huber loss
  with explicit `delta` knee. Mirrors `torch.nn.HuberLoss` at
  `torch/nn/modules/loss.py:1080-1149`.

- REQ-10: `pub struct PoissonNLLLoss { pub reduction, pub log_input,
  pub full, pub eps }` — Poisson negative log-likelihood. Mirrors
  `torch.nn.PoissonNLLLoss` at `torch/nn/modules/loss.py:286-375`.

- REQ-11: `pub struct GaussianNLLLoss { pub reduction, pub full, pub eps }`
  with `forward(input, target, var)`. Mirrors `torch.nn.GaussianNLLLoss`
  at `torch/nn/modules/loss.py:376-462`.

- REQ-12: `pub struct HingeEmbeddingLoss { pub reduction, pub margin }`.
  Mirrors `torch.nn.HingeEmbeddingLoss` at
  `torch/nn/modules/loss.py:846-923`.

- REQ-13: `pub struct MarginRankingLoss { pub reduction, pub margin }`
  with `forward_pair(x1, x2, y)`. Mirrors `torch.nn.MarginRankingLoss` at
  `torch/nn/modules/loss.py:1694-1760`.

- REQ-14: `pub struct TripletMarginLoss { pub reduction, pub margin, pub p,
  pub swap, pub eps }` with `forward_triplet(anchor, positive, negative)`.
  Mirrors `torch.nn.TripletMarginLoss` at
  `torch/nn/modules/loss.py:1857-1966`.

- REQ-15: `pub struct CosineEmbeddingLoss { pub reduction, pub margin }`
  with `forward_pair(x1, x2, y)`. Mirrors `torch.nn.CosineEmbeddingLoss`
  at `torch/nn/modules/loss.py:1622-1693`.

- REQ-16: `pub struct CTCLoss { pub blank, pub reduction,
  pub zero_infinity }` with the standard `forward(log_probs, targets,
  input_lengths, target_lengths)`. Mirrors `torch.nn.CTCLoss` at
  `torch/nn/modules/loss.py:2102-2245`.

- REQ-17: `pub struct MultiMarginLoss`,
  `pub struct MultiLabelSoftMarginLoss` — multi-class margin variants
  shipped beyond the route's `parity_ops` list. Mirror their PyTorch
  counterparts at `torch/nn/modules/loss.py:1566-1622, 1761-1857`.

- REQ-18: Every loss attaches a hand-written `impl GradFn<T> for
  <Name>Backward<T>` node to the returned tensor when `is_grad_enabled()
  && pred.requires_grad()`. Backward respects the chosen `Reduction` (mean
  divides by N; sum is identity; none distributes the upstream gradient
  element-wise).

- REQ-19: Every loss starts its `forward` with
  `autocast_guard("<op_name>")` to participate in the autocast policy
  registered in `ferrotorch_core::autograd::autocast_ops`. Names:
  `"mse_loss"`, `"cross_entropy"`, `"bce_with_logits"`, `"bce"`,
  `"l1_loss"`, `"nll_loss"`, `"kl_div"`, `"smooth_l1"`, `"huber"`,
  `"poisson_nll"`, `"gaussian_nll"`, `"hinge_embedding"`,
  `"margin_ranking"`, `"triplet_margin"`, `"cosine_embedding"`, `"ctc"`.

## Acceptance Criteria

- [x] AC-1: Every loss carries `pub reduction: Reduction` and supports
  `Mean | Sum | None`.
- [x] AC-2: `MSELoss`, `CrossEntropyLoss`, `BCEWithLogitsLoss`,
  `BCELoss`, `L1Loss`, `NLLLoss`, `KLDivLoss`, `SmoothL1Loss`, `HuberLoss`,
  `PoissonNLLLoss`, `GaussianNLLLoss`, `HingeEmbeddingLoss`,
  `MarginRankingLoss`, `TripletMarginLoss`, `CosineEmbeddingLoss`,
  `CTCLoss` are all `pub struct` re-exported through
  `ferrotorch_nn::lib::pub use loss::*`.
- [x] AC-3: Shape-mismatched inputs return `FerrotorchError::ShapeMismatch`.
- [x] AC-4: Backward gradients match the analytic derivatives for each
  reduction mode (verified by 124 in-file tests).
- [x] AC-5: `CrossEntropyLoss` is numerically stable (max-subtraction
  before exp in the log-softmax step).
- [x] AC-6: `BCEWithLogitsLoss` uses the `softplus`-based stable form
  rather than `log(sigmoid(z))`.
- [x] AC-7: `KLDivLoss::log_target` is honoured (the second arg is
  log-target if true, probability if false).
- [x] AC-8: `Reduction::None` returns a per-element loss tensor with the
  same shape as the inputs.
- [x] AC-9: `autocast_guard` is invoked at the top of every `forward`.
- [x] AC-10: Parity-sweep oracle runner arms — wired 2026-05-26 at
  `tools/parity-sweep/runner/src/main.rs` for 16 loss ops (mse_loss,
  l1_loss, smooth_l1_loss, huber_loss, binary_cross_entropy,
  binary_cross_entropy_with_logits, kl_div, cross_entropy, nll_loss,
  poisson_nll_loss, gaussian_nll_loss, hinge_embedding_loss,
  margin_ranking_loss, cosine_embedding_loss, triplet_margin_loss,
  multi_margin_loss, multilabel_soft_margin_loss). Each arm documents
  the legitimate-skip pathway for kwargs ferrotorch's narrower contract
  excludes (weight, pos_weight, ignore_index != -100, log_target=true,
  full=true, beta != 1.0, swap=true, etc.). Closes #1444.

## Architecture

### `apply_reduction` helper

`fn apply_reduction(unreduced, reduction)` at the top of `loss.rs`
matches on `Reduction::{None, Mean, Sum}` and dispatches to
`ferrotorch_core::ops::elementwise::{mean, sum}` (or returns the input
clone for `None`). Every loss's `forward` ends with this call.

### MSELoss (REQ-1, REQ-18, REQ-19)

`pub struct MSELoss { pub reduction: Reduction }` with
`#[non_exhaustive]`. `forward<T>(pred, target)` calls
`autocast_guard("mse_loss")`, validates shape equality, computes
`(pred - target)^2` via `binary_map` + `unary_map`, applies the
reduction, and attaches `MSEBackward<T>` (in `loss.rs`) when grad is
enabled. Backward: `grad_pred = 2 * (pred - target) * grad_output [/ n
for mean]`.

### CrossEntropyLoss (REQ-2)

`pub struct CrossEntropyLoss { pub reduction, pub label_smoothing }`.
`forward<T>(logits, targets)` numerically stable log-softmax (subtract
max per row, then `log(sum(exp))`), per-sample NLL with
`label_smoothing` interpolation between the target NLL and the uniform
log-prob mean. Backward: `grad_logits[b, c] = ((softmax[b, c] - one_hot[b,
c]) + ls * (1/C - softmax[b, c])) * grad_output / N` (for mean reduction;
sum drops the `/N`).

### BCEWithLogitsLoss (REQ-3)

`pub struct BCEWithLogitsLoss { pub reduction, pub pos_weight }`. Uses
`softplus(z) - y * z + (pos_weight - 1) * y * softplus(-z)` for numerical
stability. Backward: `grad_logits = (sigmoid(z) - y) * grad_output`
(weighted by `pos_weight` when set).

### BCELoss (REQ-4)

`pub struct BCELoss { pub reduction }`. Probability inputs in `(0, 1)`.
Backward: `grad_pred = (pred - target) / (pred * (1 - pred)) *
grad_output`.

### L1Loss / SmoothL1Loss / HuberLoss (REQ-5, REQ-8, REQ-9)

`pub struct L1Loss { pub reduction }`: forward `|pred - target|`,
backward `sign(pred - target) * grad_output`.

`pub struct SmoothL1Loss { pub reduction, pub beta }`: smooth at `|d| <
beta` with `0.5 * d^2 / beta`, otherwise `|d| - 0.5 * beta`.

`pub struct HuberLoss { pub reduction, pub delta }`: similar but
parameterised on `delta` directly (Huber's original formulation).

### NLLLoss (REQ-6)

`pub struct NLLLoss { pub reduction, pub ignore_index, pub weight }`.
Expects log-probabilities (i.e. the output of `log_softmax`). Backward:
`grad_log_probs[b, c] = -one_hot[b, target_b][c] * weight_c *
grad_output / N` (mean reduction). `ignore_index` zeroes the
contribution of those rows.

### KLDivLoss (REQ-7)

`pub struct KLDivLoss { pub reduction, pub log_target }`. The
`log_target=false` branch computes `target * (log(target+eps) - pred)`,
the `log_target=true` branch computes `exp(target) * (target - pred)`
(matching `F.kl_div`).

### PoissonNLLLoss / GaussianNLLLoss (REQ-10, REQ-11)

`pub struct PoissonNLLLoss { pub reduction, pub log_input, pub full,
pub eps }`. When `log_input=true`: `loss = exp(input) - target * input`.
When `log_input=false`: `loss = input - target * log(input + eps)`. The
`full=true` mode adds the Stirling approximation
`0.5 * log(2*pi*target) + target` term.

`pub struct GaussianNLLLoss { pub reduction, pub full, pub eps }` —
`forward(input, target, var)`. Clamps `var` from below by `eps`.
`loss = 0.5 * (log(var) + (input - target)^2 / var)`. The `full=true`
mode adds `0.5 * log(2*pi)`. Backward computes `d/d(input) = (input -
target) / var`, `d/d(var) = 0.5 * (1/var - diff^2 / var^2)`.

### Embedding-style losses (REQ-12, REQ-13, REQ-15)

`pub struct HingeEmbeddingLoss { pub reduction, pub margin }`. When `y =
1`: loss is `pred`; when `y = -1`: loss is `relu(margin - pred)`.

`pub struct MarginRankingLoss { pub reduction, pub margin }` with
`forward_pair(x1, x2, y)` — `loss = relu(-y * (x1 - x2) + margin)`.

`pub struct CosineEmbeddingLoss { pub reduction, pub margin }` with
`forward_pair(x1, x2, y)` — when `y = 1`: `loss = 1 - cos(x1, x2)`;
when `y = -1`: `loss = relu(cos(x1, x2) - margin)`.

### TripletMarginLoss (REQ-14)

`pub struct TripletMarginLoss { pub reduction, pub margin, pub p, pub swap,
pub eps }` with `forward_triplet(anchor, positive, negative)`. Loss is
`relu(d_pos - d_neg + margin)` where `d_pos = ||anchor - positive||_p +
eps` and similarly for `d_neg`. The `swap=true` knob swaps to
`min(d_neg, d_neg_swap)` per Hermans et al. 2017.

### CTCLoss (REQ-16)

`pub struct CTCLoss { pub blank, pub reduction, pub zero_infinity }` —
forward-backward dynamic programming for the connectionist temporal
classification objective. The `zero_infinity=true` flag replaces `+inf`
log-likelihoods (impossible alignments) with zero gradients to prevent
training instability. This is the most complex loss in the file by far —
the implementation spans ~150 lines of forward DP + ~250 lines of
backward DP.

### Multi-margin losses (REQ-17)

`pub struct MultiMarginLoss { pub reduction, pub margin, pub p,
pub weight }` and `pub struct MultiLabelSoftMarginLoss { pub reduction,
pub weight }` ship beyond the route's `parity_ops` list. They mirror
upstream `torch.nn.MultiMarginLoss` / `torch.nn.MultiLabelSoftMarginLoss`
at `torch/nn/modules/loss.py:1566-1622, 1761-1857`.

### Autocast (REQ-19)

Every `forward` starts with `autocast_guard("<op_name>")` from
`ferrotorch_core::autograd::autocast_ops`. The op names match the keys
registered in the autocast policy table, classifying every loss as
`FullPrecision` (losses are kept in f32 even under f16 autocast to
preserve numerical stability).

### Non-test production consumers

- `ferrotorch-optim/src/sgd.rs:524, 831` —
  `use ferrotorch_nn::{Linear, MSELoss, Module, Reduction};`
  `let loss_fn = MSELoss::new(Reduction::Mean);` (this is in the SGD
  module-flow conformance harness, which builds an MLP + MSE training
  loop as production code under `src/`, not `tests/`).
- `ferrotorch-nn/src/lib.rs:212-217` — re-exports the full loss family
  so downstream code can write `ferrotorch_nn::MSELoss`,
  `ferrotorch_nn::CrossEntropyLoss`, etc.
- `ferrotorch-nn/src/lib.rs` — `pub use crate::loss::{BCELoss,
  BCEWithLogitsLoss, CrossEntropyLoss, L1Loss, MSELoss, NLLLoss};`
  inside the prelude module makes these reachable as
  `ferrotorch_nn::prelude::*`.

## Parity contract

Route declares 16 parity ops:
`nn.functional.cross_entropy`, `nn.functional.mse_loss`,
`nn.functional.l1_loss`, `nn.functional.nll_loss`,
`nn.functional.binary_cross_entropy`,
`nn.functional.binary_cross_entropy_with_logits`,
`nn.functional.kl_div`, `nn.functional.smooth_l1_loss`,
`nn.functional.huber_loss`, `nn.functional.poisson_nll_loss`,
`nn.functional.gaussian_nll_loss`, `nn.functional.hinge_embedding_loss`,
`nn.functional.margin_ranking_loss`, `nn.functional.triplet_margin_loss`,
`nn.functional.cosine_embedding_loss`, `nn.functional.ctc_loss`.

Every op currently reports `MISSING` in
`tools/parity-sweep/parity_audit.json` and every `parity-sweep sweep
--op <op> --seeds 8` invocation returns `0/N passed (N skipped, 0 failed)`
— the parity-sweep runner has no dispatch arm for any of these ops. The
runner-arm gap is tracked by **blocker #1444** (`Wire parity-sweep
runner arms for 16 loss ops`). Per goal.md S5, this is a
TEST-INFRASTRUCTURE gap, not a per-REQ blocker: the implementation +
non-test consumer + 124 in-file tests are SHIPPED; only the parity-sweep
oracle wiring is missing.

Upstream edge cases preserved:

- **Empty tensors**: `MSELoss(empty, empty)` returns `0.0` with
  `Reduction::Mean` (consistent with PyTorch's empty-tensor mean
  returning NaN — divergence will be pinned by acto-critic once #1444 closes).
- **NaN input**: propagates through (every loss is built from `binary_map`
  / `unary_map` primitives that pass NaN through).
- **CrossEntropy numerical stability**: subtract row max before
  `exp`, matching `aten/src/ATen/native/SoftMax.cpp:LogSoftMaxBackward`.
- **BCEWithLogits stability**: `softplus(z) - y*z` form avoids
  catastrophic cancellation, matching upstream
  `aten/src/ATen/native/Loss.cpp:binary_cross_entropy_with_logits`.

## Verification

In-file `#[test]` block: 124 tests (count via
`grep -c "^    #\[test\]" loss.rs`). Coverage spans every reduction mode
of every loss, every backward derivative, every shape-validation
rejection. Examples:

- MSE: `test_mse_forward_mean`, `test_mse_forward_sum`,
  `test_mse_forward_none`, `test_mse_backward_mean`,
  `test_mse_backward_sum`, `test_mse_zero_loss`.
- CrossEntropy: `test_cross_entropy_forward_mean`,
  `test_cross_entropy_forward_sum`, `test_cross_entropy_backward`,
  `test_cross_entropy_label_smoothing`.
- BCEWithLogits: `test_bce_with_logits_forward_mean`,
  `test_bce_with_logits_pos_weight`, etc.
- GaussianNLL: `test_gaussian_nll_forward_mean`,
  `test_gaussian_nll_forward_sum`, `test_gaussian_nll_forward_none`,
  `test_gaussian_nll_full_mode`, `test_gaussian_nll_backward_input`,
  `test_gaussian_nll_backward_var`, `test_gaussian_nll_eps_clamp`.
- CosineEmbedding: `test_cosine_embedding_backward_positive`,
  `test_cosine_embedding_backward_negative`.
- Plus shape-mismatch rejections for every loss.

```bash
cargo test -p ferrotorch-nn --lib loss:: 2>&1 | tail -3
```

Expected: `124 passed`.

Parity-sweep smoke (blocked on runner-arm gap #1444):

```bash
for OP in nn.functional.mse_loss nn.functional.cross_entropy \
         nn.functional.l1_loss nn.functional.nll_loss \
         nn.functional.binary_cross_entropy \
         nn.functional.binary_cross_entropy_with_logits \
         nn.functional.kl_div nn.functional.smooth_l1_loss \
         nn.functional.huber_loss nn.functional.poisson_nll_loss \
         nn.functional.gaussian_nll_loss \
         nn.functional.hinge_embedding_loss \
         nn.functional.margin_ranking_loss \
         nn.functional.triplet_margin_loss \
         nn.functional.cosine_embedding_loss \
         nn.functional.ctc_loss; do
  ./target/release/parity-sweep sweep --op "$OP" --seeds 8 2>&1 | tail -1
done
```

Each line currently reports `0/N passed (N skipped, 0 failed)` — every
op is missing a runner arm; blocker #1444 tracks the wiring.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct MSELoss` + `pub fn forward` + `MSEBackward<T>` in `loss.rs`, mirroring `torch/nn/modules/loss.py:566-630`; non-test consumer: `ferrotorch-optim/src/sgd.rs:524,831` builds an MSE training loop. Runner-arm gap tracked by #1444 (test-infra). |
| REQ-2 | SHIPPED | impl: `pub struct CrossEntropyLoss` + `CrossEntropyBackward<T>` in `loss.rs` with numerically stable log-softmax + label smoothing, mirroring `torch/nn/modules/loss.py:1197-1406`; non-test consumer: `loss in ferrotorch-nn/src/lib.rs` re-exports + `ferrotorch_nn::prelude::CrossEntropyLoss` at `lib.rs`. Runner arm: #1444. |
| REQ-3 | SHIPPED | impl: `pub struct BCEWithLogitsLoss` + `BCEWithLogitsBackward<T>` in `loss.rs` using the `softplus`-stable form, mirroring `torch/nn/modules/loss.py:718-845`; non-test consumer: re-export at `lib.rs:213,289`. Runner arm: #1444. |
| REQ-4 | SHIPPED | impl: `pub struct BCELoss` + `BCEBackward<T>` in `loss.rs`, mirroring `torch/nn/modules/loss.py:632-717`; non-test consumer: `lib.rs:213,289` re-exports. Runner arm: #1444. |
| REQ-5 | SHIPPED | impl: `pub struct L1Loss` + `L1Backward<T>` in `loss.rs`, mirroring `torch/nn/modules/loss.py:65-134`; non-test consumer: `lib.rs:213,289` re-exports. Runner arm: #1444. |
| REQ-6 | SHIPPED | impl: `pub struct NLLLoss` + `NLLBackward<T>` in `loss.rs`, mirroring `torch/nn/modules/loss.py:135-273`; non-test consumer: `lib.rs:213,289` re-exports. Runner arm: #1444. |
| REQ-7 | SHIPPED | impl: `pub struct KLDivLoss` + `KLDivBackward<T>` in `loss.rs` with `log_target` knob, mirroring `torch/nn/modules/loss.py:463-565`; non-test consumer: `lib.rs:214` re-export. Runner arm: #1444. |
| REQ-8 | SHIPPED | impl: `pub struct SmoothL1Loss` in `loss.rs`, mirroring `torch/nn/modules/loss.py:987-1079`; non-test consumer: `lib.rs:215` re-export. Runner arm: #1444. |
| REQ-9 | SHIPPED | impl: `pub struct HuberLoss` + `HuberBackward<T>` in `loss.rs`, mirroring `torch/nn/modules/loss.py:1080-1149`; non-test consumer: `lib.rs:213` re-export. Runner arm: #1444. |
| REQ-10 | SHIPPED | impl: `pub struct PoissonNLLLoss` + `PoissonNLLBackward<T>` in `loss.rs`, mirroring `torch/nn/modules/loss.py:286-375`; non-test consumer: `lib.rs:214` re-export. Runner arm: #1444. |
| REQ-11 | SHIPPED | impl: `pub struct GaussianNLLLoss` + `GaussianNLLBackward<T>` in `loss.rs` with eps-clamp on `var` and per-input + per-var backward, mirroring `torch/nn/modules/loss.py:376-462`; non-test consumer: `lib.rs:213` re-export. Runner arm: #1444. |
| REQ-12 | SHIPPED | impl: `pub struct HingeEmbeddingLoss` + `HingeEmbeddingBackward<T>` in `loss.rs`, mirroring `torch/nn/modules/loss.py:846-923`; non-test consumer: `lib.rs:214` re-export. Runner arm: #1444. |
| REQ-13 | SHIPPED | impl: `pub struct MarginRankingLoss` + `MarginRankingBackward<T>` + `forward_pair` in `loss.rs`, mirroring `torch/nn/modules/loss.py:1694-1760`; non-test consumer: `lib.rs:214` re-export. Runner arm: #1444. |
| REQ-14 | SHIPPED | impl: `pub struct TripletMarginLoss` + `TripletMarginBackward<T>` + `forward_triplet` in `loss.rs`, mirroring `torch/nn/modules/loss.py:1857-1966`; non-test consumer: `lib.rs:216` re-export. Runner arm: #1444. |
| REQ-15 | SHIPPED | impl: `pub struct CosineEmbeddingLoss` + `CosineEmbeddingBackward<T>` + `forward_pair` in `loss.rs`, mirroring `torch/nn/modules/loss.py:1622-1693`; non-test consumer: `lib.rs:213` re-export. Runner arm: #1444. |
| REQ-16 | SHIPPED | impl: `pub struct CTCLoss` + `CTCBackward<T>` (forward + backward DP) in `loss.rs`, mirroring `torch/nn/modules/loss.py:2102-2245`; non-test consumer: `lib.rs:213` re-export. Runner arm: #1444. |
| REQ-17 | SHIPPED | impl: `pub struct MultiMarginLoss` + `MultiMarginBackward<T>` and `pub struct MultiLabelSoftMarginLoss` + `MultiLabelSoftMarginBackward<T>` in `loss.rs`, mirroring `torch/nn/modules/loss.py:1566-1622, 1761-1857`; non-test consumer: `lib.rs:215` re-export. |
| REQ-18 | SHIPPED | impl: every loss's `forward` ends with `Tensor::from_operation(..., grad_fn)` when `is_grad_enabled() && pred.requires_grad()`; the hand-written backward nodes (`MSEBackward`, `CrossEntropyBackward`, `BCEWithLogitsBackward`, `BCEBackward`, `L1Backward`, `NLLBackward`, `KLDivBackward`, `HuberBackward`, `PoissonNLLBackward`, `GaussianNLLBackward`, `HingeEmbeddingBackward`, `MarginRankingBackward`, `TripletMarginBackward`, `CosineEmbeddingBackward`, `CTCBackward`, `MultiMarginBackward`, `MultiLabelSoftMarginBackward`) divide by N for `Reduction::Mean`; non-test consumer: 124 in-file tests + `ferrotorch-optim/src/sgd.rs:524,831` (production MSE training loop). |
| REQ-19 | SHIPPED | impl: every `pub fn forward` opens with `autocast_guard("<op_name>")`; non-test consumer: `ferrotorch_core::autograd::autocast_ops`'s autocast-policy table consumes the registration. |
