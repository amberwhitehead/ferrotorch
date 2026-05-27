# ferrotorch-distributions — `relaxed_one_hot_categorical` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/relaxed_categorical.py
-->

## Summary

`ferrotorch-distributions/src/relaxed_one_hot_categorical.rs` defines
`RelaxedOneHotCategorical<T: Float>` — the Gumbel-softmax / Concrete
relaxation of `OneHotCategorical` from Maddison et al. 2017 and Jang
et al. 2017, parameterized by `temperature` (scalar `T`) and `probs`
(`Tensor<T>` of shape `[K]`). Samples are points on the open
`(K-1)`-simplex. Mirrors `torch.distributions.RelaxedOneHotCategorical`
(`torch/distributions/relaxed_categorical.py:109-163`). Upstream builds
this as `TransformedDistribution(ExpRelaxedCategorical, ExpTransform)`;
ferrotorch ships a direct Gumbel-softmax sampler and a closed-form
log_prob from Maddison et al. 2017 equation 26.

## Requirements

- REQ-1: `pub struct RelaxedOneHotCategorical<T: Float>` storing
  `temperature: T` (scalar), `probs: Tensor<T>` (the user-supplied
  unnormalized probabilities), `normalized: Vec<T>` (the cached
  normalized probabilities), and `num_categories: usize`. Mirrors
  `relaxed_categorical.py:109-135` `RelaxedOneHotCategorical` whose
  `base_dist: ExpRelaxedCategorical` wraps a `Categorical(probs)`
  with normalized internal probabilities.

- REQ-2: `pub fn RelaxedOneHotCategorical::new(temperature, probs) -> FerrotorchResult<Self>` —
  constructor validating `temperature > 0`, `probs` is 1-D, `K >= 1`,
  every `probs[i] > 0` (strict positive). Normalizes `probs` to sum
  to one, caching the result in `self.normalized`. Mirrors upstream's
  `simplex` constraint on `probs` (`relaxed_categorical.py:130`)
  combined with `Categorical`'s `probs_to_logits` normalization.

- REQ-3: Three accessors: `pub fn temperature(&self) -> T`,
  `pub fn probs(&self) -> &Tensor<T>`,
  `pub fn num_categories(&self) -> usize`. Mirror
  `RelaxedOneHotCategorical.{temperature, probs}` @property delegations
  (`relaxed_categorical.py:153-163`) plus the convenience
  `num_categories` exposing what upstream tracks via
  `_categorical._num_events`.

- REQ-4: `impl<T: Float> Distribution<T> for RelaxedOneHotCategorical<T>`
  provides `sample(shape)` and `rsample(shape)` both invoking the
  internal `relaxed_one_hot_sample` helper. The Gumbel-softmax forward:
  ```text
  g_i ~ Gumbel(0, 1)  for each i in 1..K     (= -log(-log(U_i)))
  l_i = (log(alpha_i) + g_i) / temperature
  z = softmax(l)                              (over the K dim)
  ```
  Mirrors `ExpRelaxedCategorical.rsample`
  (`relaxed_categorical.py:87-94`) then `ExpTransform.forward`
  (element-wise exp) per `TransformedDistribution` composition. The
  output shape is `[...shape, K]`.

- REQ-5: `log_prob(value)` evaluates the Concrete density on the
  simplex via Maddison et al. 2017 equation 26:
  ```text
  log p(z) = log((K-1)!) + (K-1)*log(lambda)
          + sum_k (log(alpha_k) - (lambda+1)*log(z_k))
          - K * log(sum_k alpha_k * z_k^(-lambda))
  ```
  Implemented via log-sum-exp for numerical stability. Validates the
  `value`'s last dim equals `K`. Output shape collapses the K dim:
  input `[N, K] → output [N]`.

- REQ-6: `entropy()` returns `InvalidArgument` because the Concrete
  distribution has no closed-form entropy. Mirrors upstream which
  does NOT override `entropy`.

- REQ-7: NOT-STARTED — `logits` accessor (upstream
  `RelaxedOneHotCategorical.logits`,
  `relaxed_categorical.py:157-159`), `mean`, `mode`, `variance`,
  `support` (`constraints.simplex`, `relaxed_categorical.py:132`),
  `expand` not implemented. Cross-cutting with `lib.md` REQ-5
  (Distribution-trait-surface blocker #1376); RelaxedOneHotCategorical-
  specific surface fill-out tracked in blocker #1422.

- REQ-8: SHIPPED — `pub struct ExpRelaxedCategorical<T>` is the
  standalone log-simplex base distribution
  (`relaxed_categorical.py:17-106`). Its `rsample` returns
  `scores - logsumexp(scores)` (a log-simplex point); its `log_prob`
  evaluates the upstream `lgamma(K) + (K-1)log(temp) + (score - lse).sum`
  form. `RelaxedOneHotCategorical::sample` production-consumes it (the
  `ExpTransform` composition: `exp(ExpRelaxedCategorical::sample)`).
  Closes #1424.

- REQ-9: SHIPPED — both `ExpRelaxedCategorical::rsample` and
  `RelaxedOneHotCategorical::rsample` attach autograd nodes
  (`ExpRelaxedRsampleBackward` / `RelaxedOneHotRsampleBackward`) so
  gradients flow through `probs` (the only `Tensor` parameter). The
  Gumbel-softmax gradient is `dz_i/dp_m = z_i(δ_im - z_m)/(temp·p_m)`.
  Mirrors `relaxed_categorical.py:87-94`. Closes #1425.

- REQ-10: SHIPPED — `new` accepts `probs` of shape `[..., K]`;
  `normalized` is row-normalized per trailing-K block; `sample` /
  `rsample` emit `[...sample_shape, ...batch, K]` (upstream
  `_extended_shape`); `log_prob` cycles each value row against its
  parameter row and collapses the trailing K dim. Closes #1426.

## Acceptance Criteria

- [x] AC-1: `pub struct RelaxedOneHotCategorical<T: Float>` with
  `temperature`, `probs`, `normalized`, `num_categories` fields.
- [x] AC-2: `new` validating `temperature > 0`, `probs.ndim() == 1`,
  `K >= 1`, `probs[i] > 0`; caches normalized probs.
- [x] AC-3: `temperature`, `probs`, `num_categories` accessors.
- [x] AC-4: `impl Distribution::sample` / `rsample` via Gumbel-softmax.
- [x] AC-5: `impl Distribution::log_prob` via Maddison eqn 26 with
  logsumexp; rejects wrong-shape value.
- [x] AC-6: `impl Distribution::entropy` returns `InvalidArgument`.
- [ ] AC-7: `logits`, `mean`, `mode`, `variance`, `support`,
  `expand` — blocker #1422.
- [x] AC-8: `ExpRelaxedCategorical` standalone — #1424.
- [x] AC-9: Differentiable rsample (`ExpRelaxedRsampleBackward` /
  `RelaxedOneHotRsampleBackward`) — #1425.
- [x] AC-10: Batched probs `[..., K]` — #1426.

## Architecture

### The struct (REQ-1, REQ-2, REQ-3)

```rust
pub struct RelaxedOneHotCategorical<T: Float> {
    temperature: T,
    probs: Tensor<T>,
    normalized: Vec<T>,
    num_categories: usize,
}
```

The `normalized: Vec<T>` cache pre-computes
`probs[i] / sum(probs)` at construction time, avoiding a re-norm on
every `sample` / `log_prob` call. `num_categories` is `probs.shape()[0]`
captured once. Both are derivable from `self.probs` but the cached
forms make the per-element loops faster and self-documenting.

The constructor's strict-positive `probs[i] > 0` check (open-half
interval) is stricter than upstream's `constraints.simplex` (which
allows zeros). The R-DEV-7 deviation prevents `log(0)` in the
Gumbel-softmax forward pass.

### The Gumbel-softmax sampler (REQ-4)

```rust
fn relaxed_one_hot_sample<T: Float>(
    temperature: T,
    normalized: &[T],
    k: usize,
    probs: &Tensor<T>,
    shape: &[usize],
    _reparam: bool,
) -> FerrotorchResult<Tensor<T>>
```

For each output position:

```text
For j in 0..K:
    U_j ~ Uniform(0, 1)  (clamped to (eps, 1-eps))
    G_j = -log(-log(U_j))                    (standard Gumbel)
    L_j = (log(alpha_j) + G_j) / temperature  (the Concrete logits)
Stable softmax(L) over K dimension:
    M = max(L)
    z_j = exp(L_j - M) / sum_l exp(L_l - M)
```

The result tensor has shape `[...input_shape, K]` — the K dim is
appended at the end. Test
`test_relaxed_one_hot_sample_shape_and_simplex` verifies that
`sample(&[100])` returns `[100, 3]` and each row sums to 1.0 (within
1e-4) and each entry is in `[0, 1]`.

### Closed-form log_prob (REQ-5)

```text
log p(z; alpha, lambda) = log((K-1)!) + (K-1)*log(lambda)
                       + sum_k (log(alpha_k) - (lambda+1)*log(z_k))
                       - K * log(sum_k alpha_k * z_k^(-lambda))
```

This is Maddison et al. 2017 equation 26. The third term is
`K * logsumexp_k(log(alpha_k) - lambda*log(z_k))` — implemented
with the standard `M = max(...); M + log(sum(exp(... - M)))` form.

Shape validation: the input `value`'s last dim must equal
`num_categories` (`K`). The output collapses the K dim, so
`log_prob(value: [N, K]) → output [N]`, and `log_prob(value: [K]) →
output []` (scalar). Test
`test_relaxed_one_hot_log_prob_wrong_shape_errors` pins the rejection.

Note: upstream's `ExpRelaxedCategorical.log_prob`
(`relaxed_categorical.py:96-106`) operates on the log-simplex
(value is `log(z)`), and the `RelaxedOneHotCategorical` chain wraps
with `ExpTransform` whose `log_abs_det_jacobian = z.log().sum(-1)`.
ferrotorch's direct simplex-space form is mathematically equivalent
to upstream's
`ExpRelaxedCategorical.log_prob(log(z)) - log(z).sum(-1)`.

### Non-test production consumers

- `pub use relaxed_one_hot_categorical::RelaxedOneHotCategorical` in
  `lib.rs:119` — grandfathered public API re-export. Downstream
  Gumbel-softmax VAE / discrete-variable training code constructs
  `RelaxedOneHotCategorical::new(temp, probs)?` directly.
- `RelaxedOneHotCategorical::new` is registered in
  `tests/conformance/_surface_inventory.toml:455`.
- The lib-level docs table in `lib.rs:37` references it.

### Fallback gate

Every `Distribution` method first invokes
`crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "RelaxedOneHotCategorical::<method>")`.

## Parity contract

`parity_ops = []`.

Numerical contracts:

- **Samples lie on the simplex**: each row sums to 1.0
  (within 1e-4 f32 tolerance), each entry in `[0, 1]`. Test
  `test_relaxed_one_hot_sample_shape_and_simplex` pins.
- **Low-temperature mode-collapse**: as `temperature → 0`, samples
  concentrate on the corner with highest `probs`. Test
  `test_relaxed_one_hot_low_temperature_concentrates` verifies that
  with `probs = [0.1, 0.1, 0.8]` and `temperature = 0.05`, at least
  70% of samples have category 2 dominant.
- **Finite `log_prob`**: test `test_relaxed_one_hot_log_prob_finite`.
- **Batch `log_prob` shape**: input `[2, 2]` → output `[2]`. Test
  `test_relaxed_one_hot_log_prob_batch` pins.
- **Wrong-shape input rejected**: input last-dim != K errors out.
  Test `test_relaxed_one_hot_log_prob_wrong_shape_errors` pins.
- **`entropy` errors out**.

## Verification

Tests in `mod tests in relaxed_one_hot_categorical.rs` (8 tests):

- `test_relaxed_one_hot_invalid_temperature`
- `test_relaxed_one_hot_invalid_probs`
- `test_relaxed_one_hot_sample_shape_and_simplex`
- `test_relaxed_one_hot_low_temperature_concentrates`
- `test_relaxed_one_hot_log_prob_finite`
- `test_relaxed_one_hot_log_prob_batch`
- `test_relaxed_one_hot_entropy_errors`
- `test_relaxed_one_hot_log_prob_wrong_shape_errors`

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib relaxed_one_hot:: 2>&1 | tail -3
```

Expected: `8 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct RelaxedOneHotCategorical<T: Float>` with `temperature: T`, `probs: Tensor<T>`, `normalized: Vec<T>`, `num_categories: usize` fields in `relaxed_one_hot_categorical.rs`, mirroring `torch/distributions/relaxed_categorical.py:109-135` (which uses `base_dist: ExpRelaxedCategorical` wrapping `Categorical(probs)`); non-test consumer: `pub use relaxed_one_hot_categorical::RelaxedOneHotCategorical` in `lib.rs:119` — grandfathered public API. |
| REQ-2 | SHIPPED | impl: `pub fn RelaxedOneHotCategorical::new(temperature, probs) -> FerrotorchResult<Self>` with `temperature > 0`, `probs.ndim() == 1`, `K >= 1`, `probs[i] > 0` validation + pre-cached normalization in `relaxed_one_hot_categorical.rs`; non-test consumer: registered in `tests/conformance/_surface_inventory.toml:455`; `pub use RelaxedOneHotCategorical` re-export. |
| REQ-3 | SHIPPED | impl: `pub fn temperature(&self) -> T`, `pub fn probs(&self) -> &Tensor<T>`, `pub fn num_categories(&self) -> usize` accessors in `relaxed_one_hot_categorical.rs`, mirroring `RelaxedOneHotCategorical.temperature` / `RelaxedOneHotCategorical.probs` @property delegations in `relaxed_categorical.py:153-163`; non-test consumer: `pub use RelaxedOneHotCategorical` re-export exposes all three. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Distribution<T> for RelaxedOneHotCategorical<T>` with `sample` / `rsample` invoking `fn relaxed_one_hot_sample` (Gumbel-softmax forward `z = softmax((log_alpha + Gumbel) / temperature)`) in `relaxed_one_hot_categorical.rs`, mirroring `ExpRelaxedCategorical.rsample` (`relaxed_categorical.py:87-94`) + `ExpTransform` composition; non-test consumer: the trait impl is the production dispatch. Test `test_relaxed_one_hot_sample_shape_and_simplex` pins. |
| REQ-5 | SHIPPED | impl: `Distribution::log_prob` in `relaxed_one_hot_categorical.rs` returns Maddison eqn 26 `log((K-1)!) + (K-1)*log(lambda) + sum(log alpha_k - (lambda+1)*log z_k) - K*lse_k(log alpha_k - lambda*log z_k)` via logsumexp; collapses last K dim of input; rejects wrong-shape value; non-test consumer: `pub use RelaxedOneHotCategorical` re-export + impl dispatch; tests `test_relaxed_one_hot_log_prob_{finite, batch, wrong_shape_errors}` pin all three. |
| REQ-6 | SHIPPED | impl: `Distribution::entropy` in `relaxed_one_hot_categorical.rs` returns `InvalidArgument`, mirroring upstream's lack of an `entropy` override; non-test consumer: any caller invoking `.entropy()` hits the error; test `test_relaxed_one_hot_entropy_errors` pins. |
| REQ-7 | NOT-STARTED | blocker #1422 — `logits` accessor (`relaxed_categorical.py:157-159`), `mean`, `mode`, `variance`, `support = constraints.simplex` (`relaxed_categorical.py:132`), `expand` not implemented; cross-cutting with `lib.md` REQ-5. |
| REQ-8 | SHIPPED | impl: `pub struct ExpRelaxedCategorical<T>` with `new` / `sample` / `rsample` / `log_prob` in `relaxed_one_hot_categorical.rs`; `rsample` returns `scores - logsumexp(scores)` (log-simplex) and `log_prob` evaluates `lgamma(K) + (K-1)log(temp) + (score - lse).sum` mirroring `relaxed_categorical.py:17-106`. Re-exported via `pub use relaxed_one_hot_categorical::ExpRelaxedCategorical` in `lib.rs`. Non-test consumer: `RelaxedOneHotCategorical::sample` builds an `ExpRelaxedCategorical` and exponentiates its log-space draw (the upstream `ExpTransform` composition). Tests `test_exp_relaxed_sample_is_log_simplex`, `test_exp_relaxed_log_prob_finite_and_shape`, `test_exp_relaxed_rsample_grad_flows_to_probs` pin. Closes #1424. |
| REQ-9 | SHIPPED | impl: `ExpRelaxedCategorical::rsample` attaches `ExpRelaxedRsampleBackward`; `RelaxedOneHotCategorical::rsample` attaches `RelaxedOneHotRsampleBackward` via `Tensor::from_operation` when `probs` requires grad. Gumbel-softmax backward `dz_i/dp_m = z_i(δ_im - z_m)/(temp·p_m)` in `relaxed_one_hot_categorical.rs`, mirroring `relaxed_categorical.py:87-94`. `temperature` is a scalar `T` (no gradient). Non-test consumer: `impl Distribution::rsample` dispatch via `pub use RelaxedOneHotCategorical` re-export. Tests `test_relaxed_one_hot_rsample_requires_grad_when_probs_grad`, `test_relaxed_one_hot_rsample_grad_flows_to_probs_finite`, `test_relaxed_one_hot_sample_detached` pin grad-flow + detachment. Closes #1425. |
| REQ-10 | SHIPPED | impl: `new` validates + row-normalizes `probs` of shape `[..., K]` via `validate_and_normalize`; `relaxed_one_hot_sample` / `exp_relaxed_sample` emit `[...sample_shape, ...batch, K]`; `log_prob` cycles each value row against `i % num_param_rows` parameter row and collapses K, in `relaxed_one_hot_categorical.rs`, mirroring upstream batched params (`relaxed_categorical.py:56-57`). Non-test consumer: `impl Distribution` dispatch + `pub use` re-export. Tests `test_relaxed_one_hot_batched_sample_shape`, `test_relaxed_one_hot_batched_log_prob_shape`, `test_relaxed_one_hot_batched_rows_differ` pin. Closes #1426. |
