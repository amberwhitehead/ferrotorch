# ferrotorch-distributions — `low_rank_multivariate_normal` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/lowrank_multivariate_normal.py
-->

## Summary

`ferrotorch-distributions/src/low_rank_multivariate_normal.rs` defines
`LowRankMultivariateNormal<T>` — a multivariate Gaussian whose
covariance has the structured form `Sigma = W*W^T + diag(D)` where
`W` is `[d, r]` and `D` is `[d]`. Mirrors
`torch.distributions.LowRankMultivariateNormal`
(`torch/distributions/lowrank_multivariate_normal.py:54-252`).
ferrotorch's implementation reifies the dense covariance and
delegates to `MultivariateNormal::from_covariance`, which means
the current per-call cost is `O(d^3)` (Cholesky) rather than the
`O(d * r^2)` Woodbury fast-path upstream provides for `r << d`.

## Requirements

- REQ-1: `pub struct LowRankMultivariateNormal<T: Float>` holding
  `loc: Tensor<T>` (the mean, `[d]`), `cov_factor: Tensor<T>`
  (`W`, `[d, r]`), `cov_diag: Tensor<T>` (`D`, `[d]`), an inner
  `MultivariateNormal<T>` built from the dense Sigma, and cached
  `d` and `r`. Mirrors upstream's `loc`/`cov_factor`/`cov_diag`
  attributes at `lowrank_multivariate_normal.py:97-139`.

- REQ-2: `pub fn LowRankMultivariateNormal::new(loc, cov_factor,
  cov_diag) -> FerrotorchResult<Self>` enforcing:
  - `loc.ndim() == 1` → 1-D mean vector of length `d`.
  - `cov_factor.ndim() == 2` with `shape == [d, r]`.
  - `cov_diag.shape() == [d]`.
  - Every element of `cov_diag > 0`.

  Mirrors upstream validation at
  `lowrank_multivariate_normal.py:104-130`.

- REQ-3: Five accessors — `loc()`, `cov_factor()`, `cov_diag()`,
  `dim()`, `rank()` — for parameter and shape introspection.
  Mirrors upstream property access.

- REQ-4: `impl<T: Float> Distribution<T> for LowRankMultivariateNormal<T>`
  with `sample`, `rsample`, `log_prob`, `entropy`, and `mean` all
  delegating to the inner `MultivariateNormal`. The trait
  surface matches upstream's `LowRankMultivariateNormal` (which
  is itself a `Distribution`).

- REQ-5: Mean property override — returns `self.loc.clone()`
  directly rather than going through the dense-covariance inner.
  Mirrors upstream's `mean` property at
  `lowrank_multivariate_normal.py:157-159`.

- REQ-6: NOT-STARTED — Woodbury fast paths
  (`_batch_capacitance_tril`, `_batch_lowrank_logdet`,
  `_batch_lowrank_mahalanobis`) at
  `lowrank_multivariate_normal.py:16-51` are NOT implemented. The
  current impl reifies the full `[d, d]` covariance, costing
  `O(d^2)` memory and `O(d^3)` per `log_prob`. Upstream achieves
  `O(d * r^2)` per call via the matrix-determinant lemma + Woodbury
  identity. Blocker #1385 tracks the Woodbury fast-path.

- REQ-7: NOT-STARTED — `variance` property is not overridden.
  Upstream computes `cov_factor.pow(2).sum(-1) + cov_diag`
  (`lowrank_multivariate_normal.py:165-169`) as a closed-form
  diagonal-of-Sigma. ferrotorch's inherited trait default returns
  `InvalidArgument`. Blocker #1386 tracks the variance override.

- REQ-8: NOT-STARTED — `scale_tril`, `covariance_matrix`,
  `precision_matrix` lazy properties from
  `lowrank_multivariate_normal.py:171-212` are not exposed. The
  inner `MultivariateNormal` carries the internal `scale_tril` but
  it's not surfaced through a public accessor on
  `LowRankMultivariateNormal`. Blocker #1387 tracks the
  matrix-form accessors.

## Acceptance Criteria

- [x] AC-1: `pub struct LowRankMultivariateNormal<T: Float>` with
  fields per REQ-1.
- [x] AC-2: `pub fn LowRankMultivariateNormal::new` enforces the
  4 shape/positivity preconditions.
- [x] AC-3: 5 accessors per REQ-3.
- [x] AC-4: `impl Distribution<T> for LowRankMultivariateNormal<T>`
  with `sample`/`rsample`/`log_prob`/`entropy`/`mean`.
- [x] AC-5: `test_low_rank_basic_construction` validates the
  `d=3, r=1` happy path.
- [x] AC-6: `test_low_rank_log_prob_at_mean_diagonal_only`
  confirms `Sigma = I_3` case yields
  `log_prob == -1.5 * ln(2*pi)`.
- [x] AC-7: `test_low_rank_sample_shape` validates that
  `sample(&[10])` returns `[10, 3]`.
- [ ] AC-8: Woodbury fast path — blocker #1385.
- [ ] AC-9: `variance` override — blocker #1386.
- [ ] AC-10: `scale_tril` / `covariance_matrix` /
  `precision_matrix` accessors — blocker #1387.

## Architecture

### The struct (REQ-1, REQ-3)

```rust
pub struct LowRankMultivariateNormal<T: Float> {
    loc: Tensor<T>,
    cov_factor: Tensor<T>,
    cov_diag: Tensor<T>,
    inner: MultivariateNormal<T>,
    d: usize,
    r: usize,
}
```

Defined at `low_rank_multivariate_normal.rs:34-43`. The five
accessors live at `low_rank_multivariate_normal.rs:140-162`.

### Constructor (REQ-2)

`low_rank_multivariate_normal.rs:54-137`. After the shape checks
it walks the factor data directly (not via `matmul`) to build
the dense `Sigma = W*W^T + diag(D)`. The result is uploaded to
the same device as `loc` if `loc.is_cuda()`, then handed to
`MultivariateNormal::from_covariance` — which itself runs
`linalg::cholesky` on the dense matrix. The choice of building
covariance manually rather than via tensor ops keeps the
constructor self-contained without pulling in autograd
dependencies.

### The Distribution impl (REQ-4, REQ-5)

`low_rank_multivariate_normal.rs:165-187`. All four methods —
`sample`, `rsample`, `log_prob`, `entropy` — delegate verbatim
to `self.inner.<method>(...)`. The `mean` override returns
`self.loc.clone()` at `low_rank_multivariate_normal.rs:178-182`
to match upstream's `loc` property (this is faster than
delegating to `self.inner.mean()`, which would also return
`loc` but go through an extra clone).

### Non-test production consumers

- `pub use low_rank_multivariate_normal::LowRankMultivariateNormal`
  at `lib.rs:110` — grandfathered public API. Downstream code
  (factor analysis, probabilistic PCA, variational inference with
  low-rank Gaussian posteriors) constructs
  `LowRankMultivariateNormal::new(loc, W, D)?`.
- `MultivariateNormal` is the production consumer of the inner-
  state — `LowRankMultivariateNormal::new` invokes
  `MultivariateNormal::from_covariance` at
  `low_rank_multivariate_normal.rs:127`.

## Parity contract

`parity_ops = []`. No direct parity oracle. The underlying
`linalg::cholesky` and `linalg::solve` ops that the inner
`MultivariateNormal` invokes are independently audited (see
`.design/ferrotorch-core/linalg.md`). Edge cases preserved:

- **`r << d`** — Currently `O(d^3)` (Cholesky) instead of the
  upstream `O(d * r^2)`. Numerically correct, just slow.
  Blocker #1385.
- **`cov_diag[i] <= 0`** — constructor returns
  `InvalidArgument`. Upstream's `arg_constraints` with
  `constraints.positive` would do the same. Test
  `test_low_rank_negative_diag_errors` pins this.
- **Mismatched factor shape** — `cov_factor.shape[0] != d`
  returns `InvalidArgument`. Test
  `test_low_rank_wrong_factor_shape_errors` pins this.
- **Mismatched diag shape** — `cov_diag.shape != [d]` returns
  `InvalidArgument`. Test
  `test_low_rank_wrong_diag_shape_errors` pins this.
- **`W = 0`** — degenerates to a diagonal Gaussian with
  covariance `diag(D)`. Test
  `test_low_rank_log_prob_at_mean_diagonal_only` pins
  `log_prob == -d/2 * log(2*pi)` for the standard case.

## Verification

Tests in `mod tests in low_rank_multivariate_normal.rs` (5 tests):

- `test_low_rank_basic_construction`,
- `test_low_rank_negative_diag_errors`,
- `test_low_rank_wrong_factor_shape_errors`,
- `test_low_rank_wrong_diag_shape_errors`,
- `test_low_rank_log_prob_at_mean_diagonal_only`,
- `test_low_rank_sample_shape`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib low_rank_multivariate_normal:: 2>&1 | tail -3
```

Expected: `6 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LowRankMultivariateNormal<T: Float>` with `loc`/`cov_factor`/`cov_diag`/`inner`/`d`/`r` fields at `low_rank_multivariate_normal.rs:34-43`, mirroring `torch/distributions/lowrank_multivariate_normal.py:54-139`; non-test consumer: `pub use low_rank_multivariate_normal::LowRankMultivariateNormal` at `lib.rs:110`. |
| REQ-2 | SHIPPED | impl: the constructor at `low_rank_multivariate_normal.rs:54-137` validating loc-1D / factor-`[d, r]` / diag-`[d]` / `diag > 0`, mirroring `lowrank_multivariate_normal.py:104-130`; non-test consumer: invoked from `pub use ...::new` via the re-export at `lib.rs:110`. |
| REQ-3 | SHIPPED | impl: `loc()`/`cov_factor()`/`cov_diag()`/`dim()`/`rank()` accessors at `low_rank_multivariate_normal.rs:140-162`; non-test consumer: re-export at `lib.rs:110` exposes them as the introspection surface. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Distribution<T> for LowRankMultivariateNormal<T>` at `low_rank_multivariate_normal.rs:165-187` delegating to inner `MultivariateNormal`, mirroring upstream surface at `lowrank_multivariate_normal.py:214-252`; non-test consumer: re-export means external Distribution-trait calls hit this impl. |
| REQ-5 | SHIPPED | impl: `mean()` override returning `self.loc.clone()` at `low_rank_multivariate_normal.rs:178-182`, mirroring `lowrank_multivariate_normal.py:157-159`; non-test consumer: re-export at `lib.rs:110` exposes the override via the `Distribution` trait. |
| REQ-6 | NOT-STARTED | blocker #1385 — Woodbury / capacitance-tril fast paths at `lowrank_multivariate_normal.py:16-51` not implemented; ferrotorch reifies the dense `[d, d]` covariance at `low_rank_multivariate_normal.rs:107-125` and pays `O(d^3)` per `log_prob`. |
| REQ-7 | NOT-STARTED | blocker #1386 — `variance` override not implemented; the inherited trait default at `lib.rs:223-227` returns `InvalidArgument`. Upstream computes `cov_factor.pow(2).sum(-1) + cov_diag` at `lowrank_multivariate_normal.py:165-169`. |
| REQ-8 | NOT-STARTED | blocker #1387 — `scale_tril` / `covariance_matrix` / `precision_matrix` accessors at `lowrank_multivariate_normal.py:171-212` not exposed; the inner `MultivariateNormal::scale_tril` is private to `LowRankMultivariateNormal`. |
