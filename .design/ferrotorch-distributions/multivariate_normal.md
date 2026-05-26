# ferrotorch-distributions — `multivariate_normal` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/multivariate_normal.py
-->

## Summary

`ferrotorch-distributions/src/multivariate_normal.rs` defines
`MultivariateNormal<T>` — a multivariate Gaussian parameterised by
a mean vector `loc` and a lower-triangular Cholesky factor
`scale_tril` such that `Sigma = L * L^T`. Mirrors
`torch.distributions.MultivariateNormal`
(`torch/distributions/multivariate_normal.py:88-274`). Three named
constructors (`from_scale_tril`, `from_covariance`, `from_precision`)
mirror upstream's three mutually-exclusive kwarg modes. The
implementation is device-resident ("Pattern B"): on CUDA every
linear-algebra step routes through cuSOLVER via
`ferrotorch_core::linalg`.

## Requirements

- REQ-1: `pub struct MultivariateNormal<T: Float>` holding
  `loc: Tensor<T>`, `scale_tril: Tensor<T>` (the lower-triangular
  Cholesky factor), and a cached `d: usize` for dimensionality.
  Mirrors upstream's `loc` / `_unbroadcasted_scale_tril` attributes
  at `multivariate_normal.py:135-196`.

- REQ-2: Three constructors:
  - `from_scale_tril(loc, scale_tril)` — direct, most efficient.
  - `from_covariance(loc, covariance_matrix)` — computes
    `L = cholesky(Sigma)` via `linalg::cholesky`.
  - `from_precision(loc, precision_matrix)` — computes
    `Sigma = solve(P, I)` then `L = cholesky(Sigma)`.

  Each enforces `loc.ndim() == 1`, the matrix shape `[d, d]`, and
  same-device parameters. Mirror upstream's three-kwarg dispatch
  at `multivariate_normal.py:135-196`.

- REQ-3: Four accessors — `loc()`, `scale_tril()`, `dim()` — plus
  the `from_*` constructors as the public surface. Mirror
  upstream's `loc` / `scale_tril` / `event_shape` properties.

- REQ-4: `impl<T: Float> Distribution<T> for MultivariateNormal<T>`
  with `sample`, `rsample`, `log_prob`, `mean`, `entropy`. Mirrors
  `multivariate_normal.py:251-274`.

- REQ-5: Sampling via the reparameterization trick —
  `z = loc + eps @ L^T` where `eps ~ N(0, I)`. All composed via
  device-resident `matmul` + broadcast-add, so the autograd graph
  records `scale_tril` and `loc` as upstream nodes. Mirrors
  `multivariate_normal.py:251-254`.

- REQ-6: `log_prob` via the precision-matrix reformulation —
  `log_prob = -0.5 * (d*log(2*pi) + (x-mu)^T * Sigma^{-1} * (x-mu))
              - sum(log(diag(L)))`. The inverse is computed via
  `linalg::solve(Sigma, I)` (cuSOLVER `getrf/getrs` on CUDA).
  Mirrors `multivariate_normal.py:256-264`.

- REQ-7: Device-resident `half_log_det_of_tril` helper — masks the
  diagonal via `eye(d) * L + (1 - eye(d))`, applies `log` and
  sums. This avoids the CPU-only `linalg::diagonal` and stays on
  device. Mirrors `multivariate_normal.py:261-263` (which uses
  `diagonal(dim1=-2, dim2=-1)`).

- REQ-8: `entropy` via the closed form
  `H = 0.5 * d * (1 + ln(2*pi)) + sum(log(diag(L)))`. Mirrors
  `multivariate_normal.py:266-274`.

- REQ-9: `rsample` flows gradients through `loc` and `scale_tril`
  via the standard autograd chain (no hand-rolled backward node).
  This is the "Pattern B" device-resident composition referenced
  in the file's doc comment. Mirrors upstream's autograd-traced
  `loc + _batch_mv(scale_tril, eps)` at `multivariate_normal.py:254`.

- REQ-10: NOT-STARTED — `covariance_matrix` and `precision_matrix`
  lazy properties from `multivariate_normal.py:223-233` are not
  exposed. Only the constructor-mode `scale_tril` is publicly
  reachable. Blocker #1393 tracks the matrix-form accessors.

- REQ-11: NOT-STARTED — `mode` and `variance` properties from
  `multivariate_normal.py:239-249` are not implemented. `mode`
  trivially equals `loc`; `variance` equals
  `scale_tril.pow(2).sum(-1)` (per-coordinate variance). Blocker
  #1394 tracks both.

## Acceptance Criteria

- [x] AC-1: `pub struct MultivariateNormal<T: Float>` with
  `loc`/`scale_tril`/`d`.
- [x] AC-2: Three constructors per REQ-2 with full validation.
- [x] AC-3: `loc()`/`scale_tril()`/`dim()` accessors.
- [x] AC-4: `impl Distribution<T> for MultivariateNormal<T>` with
  5 methods.
- [x] AC-5: `test_mvn_log_prob_standard_at_mean` validates
  `log_prob(0) == -log(2*pi)` for N(0, I_2).
- [x] AC-6: `test_mvn_entropy_standard` validates
  `H == 0.5 * d * (1 + log(2*pi))` for N(0, I_2).
- [x] AC-7: `test_mvn_rsample_backward` validates gradient flow
  through `loc` and `scale_tril`.
- [x] AC-8: `test_mvn_from_covariance` validates the Cholesky
  internal step.
- [x] AC-9: `test_mvn_from_precision` validates the
  invert-then-Cholesky internal step.
- [x] AC-10: `test_mvn_3d` validates higher-dim case.
- [ ] AC-11: `covariance_matrix` / `precision_matrix` accessors —
  blocker #1393.
- [ ] AC-12: `mode` / `variance` properties — blocker #1394.

## Architecture

### The struct (REQ-1, REQ-3)

```rust
pub struct MultivariateNormal<T: Float> {
    loc: Tensor<T>,
    scale_tril: Tensor<T>,
    d: usize,
}
```

Defined at `multivariate_normal.rs:50-56`. Accessors at
`multivariate_normal.rs:163-176`.

### Constructors (REQ-2)

- `from_scale_tril` at `multivariate_normal.rs:63-88` — validates
  shapes, device equality, returns directly.
- `from_covariance` at `multivariate_normal.rs:96-124` — same
  validation, then `scale_tril = no_grad(linalg::cholesky(cov))`.
- `from_precision` at `multivariate_normal.rs:130-161` — same
  validation, then `covariance = no_grad(linalg::solve(P, I))`
  followed by `linalg::cholesky`.

The `no_grad` blocks ensure the constructor's matrix factorisations
don't leak into the autograd graph of subsequent ops.

### half_log_det_of_tril helper (REQ-7)

`multivariate_normal.rs:196-209` builds an `eye(d)` mask on the same
device as `L`, computes `L_safe = L * mask + (1 - mask)` so the
diagonal carries `L_ii` and off-diagonals are 1, then `log` makes
off-diagonals 0 and diagonals `log(L_ii)`, and `sum_all` collapses
to a scalar `sum log(L_ii)`. All ops are grad-aware so backward
through `scale_tril` works.

### The Distribution impl (REQ-4, REQ-5, REQ-6, REQ-8, REQ-9)

`sample` (`multivariate_normal.rs:212-246`) uploads `eps` to the
parameter device, computes `eps @ L^T + loc` in a `no_grad`
context (sampling is detached per trait contract), and reshapes
to `shape ++ [d]`.

`rsample` (`multivariate_normal.rs:248-274`) follows the same
pipeline but WITHOUT `no_grad`, so the autograd engine sees
`scale_tril` and `loc` as upstream nodes through `matmul` and
broadcast-add.

`log_prob` (`multivariate_normal.rs:276-371`) builds `Sigma`,
inverts it via `linalg::solve(Sigma, I)`, computes Mahalanobis as
`sum(diff * (diff @ Sigma^{-1}), dim=-1)`, adds `half_log_det`,
then combines into the final scalar.

`mean` (`multivariate_normal.rs:373-377`) returns `loc.clone()`.

`entropy` (`multivariate_normal.rs:379-394`) computes
`half * d * (1 + log(2*pi)) + half_log_det_of_tril(L, d)` on the
parameter device.

### Non-test production consumers

- `pub use multivariate_normal::MultivariateNormal` at `lib.rs:113`
  — grandfathered public API. Downstream code (Bayesian linear
  regression, VI multivariate posteriors, GP samplers) constructs
  via one of the three `from_*` constructors.
- **`low_rank_multivariate_normal.rs:127`** invokes
  `MultivariateNormal::from_covariance(loc.clone(), cov_t)?` to
  build its inner distribution. This is the only intra-crate
  production consumer.
- `ferrotorch_core::linalg::{cholesky, solve}` are the production
  consumers of `ferrotorch-core`'s linear-algebra surface —
  invoked from `from_covariance`, `from_precision`, and `log_prob`.

## Parity contract

`parity_ops = []`. No direct multivariate parity oracle, but
`cholesky` and `solve` (the heavyweight ops Sigma operates against)
are independently audited under `.design/ferrotorch-core/linalg.md`.
Edge cases preserved:

- **Non-1D `loc`** — constructor errors. Tests
  `test_mvn_shape_mismatch_loc` pin this.
- **Mismatched `scale_tril` shape** — errors. Test
  `test_mvn_shape_mismatch_tril` pins this.
- **Device mismatch** — errors with `DeviceMismatch`. Upstream
  silently broadcasts via `loc.expand` but ferrotorch rejects
  cross-device construction.
- **Standard case** — `log_prob(0) == -log(2*pi)` for N(0, I_2).
- **Backward through `scale_tril`** — every diagonal element
  contributes to `half_log_det`; off-diagonals contribute only via
  the Mahalanobis term. Test `test_mvn_rsample_backward` confirms
  finite gradients on all `L` elements.
- **Three constructor modes** — `from_scale_tril`, `from_covariance`,
  `from_precision` all converge to the same internal `scale_tril`
  representation, so `log_prob` is numerically identical
  modulo the Cholesky/solve roundoff. Tests
  `test_mvn_from_covariance` / `test_mvn_from_precision` pin
  identical `log_prob` values.

## Verification

Tests in `mod tests in multivariate_normal.rs` (12 tests):

- `test_mvn_sample_shape`,
- `test_mvn_sample_2d_shape`,
- `test_mvn_rsample_has_grad`,
- `test_mvn_rsample_no_grad_when_detached`,
- `test_mvn_log_prob_standard_at_mean`,
- `test_mvn_log_prob_batch`,
- `test_mvn_from_covariance`,
- `test_mvn_from_precision`,
- `test_mvn_entropy_standard`,
- `test_mvn_entropy_scaled`,
- `test_mvn_rsample_backward`,
- `test_mvn_shape_mismatch_loc`,
- `test_mvn_shape_mismatch_tril`,
- `test_mvn_f64`,
- `test_mvn_3d`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib multivariate_normal:: 2>&1 | tail -3
```

Expected: `15 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct MultivariateNormal<T: Float>` with `loc`/`scale_tril`/`d` at `multivariate_normal.rs:50-56`, mirroring `torch/distributions/multivariate_normal.py:88-196`; non-test consumer: `pub use multivariate_normal::MultivariateNormal` at `lib.rs:113` exposes it as public API and `low_rank_multivariate_normal.rs:127` constructs an inner instance. |
| REQ-2 | SHIPPED | impl: 3 constructors (`from_scale_tril` at `multivariate_normal.rs:63-88`, `from_covariance` at `multivariate_normal.rs:96-124`, `from_precision` at `multivariate_normal.rs:130-161`) each with shape + device validation, mirroring `multivariate_normal.py:135-196`; non-test consumer: `low_rank_multivariate_normal.rs:127` invokes `from_covariance` to build the inner MVN — direct production consumer of constructor mode 2. |
| REQ-3 | SHIPPED | impl: `loc()`/`scale_tril()`/`dim()` accessors at `multivariate_normal.rs:163-176`; non-test consumer: re-export at `lib.rs:113`. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Distribution<T> for MultivariateNormal<T>` at `multivariate_normal.rs:211-395` with `sample`/`rsample`/`log_prob`/`mean`/`entropy`; non-test consumer: `low_rank_multivariate_normal.rs:166-177` delegates all 4 trait methods (`sample`, `rsample`, `log_prob`, `entropy`) to `self.inner.<method>()` — direct production consumer. |
| REQ-5 | SHIPPED | impl: `eps @ L^T + loc` device-resident composition in `sample` at `multivariate_normal.rs:228-235`, mirroring `multivariate_normal.py:251-254`; non-test consumer: `low_rank_multivariate_normal.rs:167` delegates to `self.inner.sample(shape)`. |
| REQ-6 | SHIPPED | impl: precision-matrix reformulation in `log_prob` at `multivariate_normal.rs:294-350` (`Sigma = L*L^T; Sigma^{-1} = solve(Sigma, I); mahal = sum(diff * (diff @ Sigma^{-1}), dim=-1)`), mirroring `multivariate_normal.py:256-264`; non-test consumer: `low_rank_multivariate_normal.rs:174` delegates `log_prob` to `self.inner.log_prob(value)`. |
| REQ-7 | SHIPPED | impl: `half_log_det_of_tril` helper at `multivariate_normal.rs:196-209` using eye-mask composition; non-test consumer: invoked by `MultivariateNormal::log_prob` at `multivariate_normal.rs:326` and `MultivariateNormal::entropy` at `multivariate_normal.rs:391` — both inside production code paths in the same file. |
| REQ-8 | SHIPPED | impl: `entropy` body at `multivariate_normal.rs:379-394` computing `0.5 * d * (1 + ln(2*pi)) + half_log_det`, mirroring `multivariate_normal.py:266-274`; non-test consumer: `low_rank_multivariate_normal.rs:185-186` delegates entropy to `self.inner.entropy()`. |
| REQ-9 | SHIPPED | impl: `rsample` at `multivariate_normal.rs:248-274` uses raw grad-aware `matmul` + `add` (no `no_grad`), so the autograd engine traces back through `scale_tril` and `loc`; non-test consumer: `low_rank_multivariate_normal.rs:170-172` delegates `rsample` to `self.inner.rsample(shape)`. Test `test_mvn_rsample_backward` pins finite grads. |
| REQ-10 | NOT-STARTED | blocker #1393 — `covariance_matrix` / `precision_matrix` lazy properties at `multivariate_normal.py:223-233` not exposed; only the constructor-mode `scale_tril` accessor at `multivariate_normal.rs:170-172` is public. |
| REQ-11 | NOT-STARTED | blocker #1394 — `mode` (trivially `loc`) and `variance` (`scale_tril.pow(2).sum(-1)`) properties at `multivariate_normal.py:239-249` not implemented; the default trait impls at `lib.rs:216-227` return `InvalidArgument`. |
