# ferrotorch-distributions — `uniform` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/uniform.py
-->

## Summary

`ferrotorch-distributions/src/uniform.rs` defines `Uniform<T: Float>`
— the continuous uniform distribution on `[low, high)`. Mirrors
`torch.distributions.Uniform`
(`torch/distributions/uniform.py:14-108`). Ships reparameterized
sampling via `z = low + (high - low) * u, u ~ Uniform(0, 1)` with a
hand-rolled backward node `UniformRsampleBackward` that propagates
gradients to `low` and `high`. Also implements
`log_prob`'s autograd path through `UniformLogProbBackward`. Full
`mean` / `variance` / `stddev` / `cdf` / `icdf` / `mode` ship per
PyTorch.

## Requirements

- REQ-1: `pub struct Uniform<T: Float>` storing `low: Tensor<T>` and
  `high: Tensor<T>`. Mirrors `uniform.py:57-69` `__init__` which
  broadcasts the two params.

- REQ-2: `pub fn Uniform::new(low, high) -> FerrotorchResult<Self>` —
  constructor requiring matching shapes. Upstream uses `broadcast_all`;
  ferrotorch's strict shape-match is R-DEV-7.

- REQ-3: `pub fn low(&self) -> &Tensor<T>` and `pub fn high(&self) -> &Tensor<T>`
  accessors. Mirror upstream attribute access.

- REQ-4: `impl<T: Float> Distribution<T> for Uniform<T>` provides
  `sample(shape)` via `low + (high - low) * U`, `U ~ Uniform(0, 1)`.
  Mirrors `uniform.py:85-88` `rsample` (upstream uses the same
  formula for both sample and rsample since the operation is
  reparameterizable end-to-end).

- REQ-5: `rsample(shape)` differentiable through `low` and `high`.
  Builds the result via `Tensor::from_operation` with a
  `UniformRsampleBackward` autograd node capturing `low`, `high`,
  and the underlying `u` sample. Backward gradients:
  - `d z / d low  = 1 - u  → grad_low  = sum(grad_output * (1 - u))`
  - `d z / d high = u      → grad_high = sum(grad_output * u)`

- REQ-6: `log_prob(value)` returns `-log(high - low)` if
  `low <= x < high` else `-inf`. Mirrors `uniform.py:90-95` (which
  uses boolean masks + `torch.log(lb*ub) - log(high-low)`; the
  `log(0) = -inf` gives the out-of-range branch). Builds an autograd
  path through `UniformLogProbBackward` when any of `low`, `high`,
  `value` requires grad. Backward gradients:
  - `d lp / d low  =  1 / (high - low)` (in-range, summed)
  - `d lp / d high = -1 / (high - low)` (in-range, summed)
  - `d lp / d value = 0` (flat density)

- REQ-7: `entropy()` returns `log(high - low)`. Mirrors
  `uniform.py:107-108` `torch.log(high - low)`.

- REQ-8: `cdf(value)` returns
  `clamp((value - low) / (high - low), 0, 1)`. Mirrors
  `uniform.py:97-101` `result.clamp(min=0, max=1)`.

- REQ-9: `icdf(q)` returns `low + (high - low) * q`. Mirrors
  `uniform.py:103-105` `value * (high - low) + low`. Assumes
  `q in [0, 1]` (no validation).

- REQ-10: `mean()` returns `(low + high) / 2`. Mirrors
  `uniform.py:42-44`.

- REQ-11: `variance()` returns `(high - low)^2 / 12`. Mirrors
  `uniform.py:53-55`.

- REQ-12: `stddev()` returns `(high - low) / sqrt(12)`. Mirrors
  `uniform.py:49-51`.

- REQ-13: `mode()` returns `(low + high) / 2` (midpoint as
  representative). Upstream `uniform.py:46-48` returns `nan * high`
  because every point in `[low, high]` is equally a mode. ferrotorch
  deviates per R-DEV-6 (returning a representative finite value is
  more useful than NaN for downstream code; the upstream behaviour is
  arguably a wart). Blocker #1429 tracks the strict-NaN
  conformance question.

- REQ-14: NOT-STARTED — `expand`, `support` (dependent property
  `constraints.interval(low, high)`, `uniform.py:80-83`),
  `arg_constraints` (`uniform.py:33-39` — these are
  inter-parameter constraints `low < high` etc.) not implemented.
  Cross-cutting with `lib.md` REQ-5 (Distribution-trait-surface
  blocker #1376); Uniform-specific surface fill-out tracked in
  blocker #1430.

## Acceptance Criteria

- [x] AC-1: `pub struct Uniform<T: Float>` with `low`, `high`.
- [x] AC-2: `new` rejecting shape mismatch.
- [x] AC-3: `low()`, `high()` accessors.
- [x] AC-4: `Distribution::sample` via `low + (high - low) * U`.
- [x] AC-5: `Distribution::rsample` differentiable through
  `low` and `high` via `UniformRsampleBackward`.
- [x] AC-6: `Distribution::log_prob` with out-of-range `-inf` +
  `UniformLogProbBackward` for value/low/high grads.
- [x] AC-7: `Distribution::entropy` returns `log(high - low)`.
- [x] AC-8: `Distribution::cdf` clamped to `[0, 1]`.
- [x] AC-9: `Distribution::icdf` linear.
- [x] AC-10: `Distribution::mean` returns midpoint.
- [x] AC-11: `Distribution::variance` returns `(high-low)^2 / 12`.
- [x] AC-12: `Distribution::stddev` returns `(high-low) / sqrt(12)`.
- [x] AC-13: `Distribution::mode` returns midpoint (R-DEV-6 dev).
- [ ] AC-14: `expand`, `support`, `arg_constraints` — blocker #1430.

## Architecture

### The struct (REQ-1, REQ-2, REQ-3)

`Uniform<T: Float>` is two-tensor with strict shape match in
`Uniform::new`. The constructor does NOT validate `low < high` —
upstream's `arg_constraints` (`uniform.py:33-39`) is a dependent
property attached to `validate_args` which is part of the `lib.md`
REQ-5 cross-cutting gap.

### Distribution::sample and rsample (REQ-4, REQ-5)

Both compute `low + (high - low) * U` per element. The difference:

- `sample`: `Tensor::from_storage(_, _, false)` — no grad.
- `rsample`: when `low.requires_grad() || high.requires_grad()` AND
  `is_grad_enabled()`, builds via `Tensor::from_operation` with a
  `UniformRsampleBackward { low, high, u }` GradFn (clones `low`,
  `high` for the backward and captures the `u` tensor). Otherwise
  falls back to `from_storage`.

`UniformRsampleBackward::backward(grad_output)` computes:

```text
grad_low  = sum_i grad_output[i] * (1 - u[i])
grad_high = sum_i grad_output[i] * u[i]
```

Both scalars are wrapped into `low.shape()` / `high.shape()`
single-element tensors and moved to the GPU if the original
parameters live there.

### log_prob with autograd (REQ-6)

The in-range / out-of-range branch:

```text
log_prob(x) = -log(high - low)   if low <= x < high
            = -inf               otherwise
```

When ANY of `low.requires_grad()`, `high.requires_grad()`,
`value.requires_grad()` is true AND grad is enabled, the result is
built via `Tensor::from_operation` with a
`UniformLogProbBackward { low, high, value }` GradFn. The backward:

```text
For in-range elements:
    grad_low_val  += grad_output[i] * (1 / (high - low))
    grad_high_val += grad_output[i] * (-1 / (high - low))
grad_value = zeros (flat density: d lp / d value = 0)
```

This matches upstream's autograd graph through the `log(high - low)`
term.

### Closed-form moments (REQ-7..REQ-13)

All five are scalar arithmetic over the per-element `low`, `high`:

- `entropy = log(high - low)`
- `mean = (low + high) / 2`
- `mode = mean` (R-DEV-6 — see REQ-13)
- `variance = (high - low)^2 / 12`
- `stddev = (high - low) / sqrt(12)`
- `cdf(x) = 0 if x < low, 1 if x >= high, (x - low)/(high - low) otherwise`
- `icdf(p) = low + (high - low) * p`

### Non-test production consumers

- `pub use uniform::Uniform` in `lib.rs:122` — grandfathered public
  API re-export. Downstream code that needs a uniform-prior random
  layer constructs `Uniform::new(low, high)?` directly.
- `kl_uniform_uniform(p: &Uniform<T>, q: &Uniform<T>)` in `kl.rs:271`
  is invoked by `kl_dispatch` (`kl.rs:123`).
- `kl_normal_uniform(p: &Normal<T>, q: &Uniform<T>)` in `kl.rs:350`
  is invoked by `kl_dispatch` (`kl.rs:137`).
- `kl_uniform_normal(p: &Uniform<T>, q: &Normal<T>)` in `kl.rs:383`
  is invoked by `kl_dispatch` (`kl.rs:144`).
- Three non-test KL-divergence production consumers makes Uniform
  the most-consumed distribution in batch C.
- `Uniform::new` is registered in
  `tests/conformance/_surface_inventory.toml:329`.
- The lib-level docs table in `lib.rs:16` references it.

### Fallback gate

Every `Distribution` method first invokes
`crate::fallback::check_gpu_fallback_opt_in(&[&self.low, &self.high, ...], "Uniform::<method>")`.

## Parity contract

`parity_ops = []`.

Numerical contracts:

- **Samples in `[low, high)`**: test `test_uniform_sample_in_range`
  draws 1000 samples from `Uniform(2, 5)` and verifies all are in
  `[2, 5)`.
- **`log_prob` in-range = `-log(high - low)`**: test
  `test_uniform_log_prob_in_range` pins `Uniform(0, 2).log_prob(1) = -log(2)`.
- **`log_prob` out-of-range = `-inf`**: test
  `test_uniform_log_prob_out_of_range`.
- **`log_prob` batched**: `[3]` value → `[3]` output with mixed
  in/out-of-range. Test `test_uniform_log_prob_batch`.
- **`entropy` = `log(high - low)`**: test `test_uniform_entropy`
  pins `Uniform(1, 4).entropy = log(3)`; test
  `test_uniform_entropy_unit` pins `Uniform(0, 1).entropy = 0`.
- **`rsample` has gradient + backward**: tests
  `test_uniform_rsample_has_grad`, `test_uniform_rsample_no_grad_when_detached`,
  `test_uniform_rsample_backward` pin all three paths. The backward
  test verifies `grad_low + grad_high = n` (since
  `d(low + (high-low)*u)/d(low) + d(...)/d(high) = 1` so summed
  over n samples = n).
- **`mean = (low+high)/2`, `variance = (h-l)^2/12`, `stddev = (h-l)/sqrt(12)`**:
  test `test_uniform_mean_variance_stddev`.
- **`cdf` at endpoints**: test `test_uniform_cdf_endpoints` pins
  `Uniform(0, 2).cdf([-1, 0, 1, 2, 3]) = [0, 0, 0.5, 1, 1]`.
- **`icdf` linear**: test `test_uniform_icdf_linear` pins
  `Uniform(10, 20).icdf([0, 0.25, 0.5, 1]) = [10, 12.5, 15, 20]`.
- **`mode` = midpoint**: test `test_uniform_mode_is_midpoint` (note
  R-DEV-6 deviation: upstream returns `NaN * high`).
- **Shape mismatch**: test `test_uniform_shape_mismatch`.

## Verification

Tests in `mod tests in uniform.rs` (16 tests):

- `test_uniform_sample_shape`
- `test_uniform_sample_in_range`
- `test_uniform_rsample_has_grad`
- `test_uniform_rsample_no_grad_when_detached`
- `test_uniform_log_prob_in_range`
- `test_uniform_log_prob_out_of_range`
- `test_uniform_log_prob_batch`
- `test_uniform_entropy`
- `test_uniform_entropy_unit`
- `test_uniform_shape_mismatch`
- `test_uniform_rsample_backward`
- `test_uniform_f64`
- `test_uniform_mean_variance_stddev`
- `test_uniform_cdf_endpoints`
- `test_uniform_icdf_linear`
- `test_uniform_mode_is_midpoint`
- `test_uniform_stddev_is_range_over_sqrt12`

Smoke command:

```bash
cargo test -p ferrotorch-distributions --lib uniform:: 2>&1 | tail -3
```

Expected: `17 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Uniform<T: Float>` with `low`, `high` fields in `uniform.rs`, mirroring `torch/distributions/uniform.py:57-69`; non-test consumer: `pub use uniform::Uniform` in `lib.rs:122` PLUS `kl_uniform_uniform`/`kl_normal_uniform`/`kl_uniform_normal` in `kl.rs:{271, 350, 383}` all read `p.low()`/`p.high()` — three non-test KL-divergence production consumers. |
| REQ-2 | SHIPPED | impl: `pub fn Uniform::new(low, high) -> FerrotorchResult<Self>` with shape-match validation in `uniform.rs`; non-test consumer: registered in `tests/conformance/_surface_inventory.toml:329`; `pub use Uniform` re-export; `kl_*` consumers in `kl.rs` call sites. Test `test_uniform_shape_mismatch` pins. |
| REQ-3 | SHIPPED | impl: `pub fn low(&self) -> &Tensor<T>` and `pub fn high(&self) -> &Tensor<T>` accessors in `uniform.rs`, mirroring `Uniform.low`/`Uniform.high` attribute access; non-test consumer: `kl_uniform_uniform` in `kl.rs:271` reads `p.low()`, `p.high()`, `q.low()`, `q.high()` for the closed-form KL formula. |
| REQ-4 | SHIPPED | impl: `Distribution::sample` in `uniform.rs` via `low + (high - low) * U`, mirroring `uniform.py:85-88` `rsample` formula (upstream uses identical formula for both); non-test consumer: `pub use Uniform` re-export plus impl dispatch; test `test_uniform_sample_in_range` pins range. |
| REQ-5 | SHIPPED | impl: `Distribution::rsample` in `uniform.rs` builds `Tensor::from_operation` with `Arc<UniformRsampleBackward { low, high, u }>` autograd node; backward computes `grad_low = sum(go * (1-u))`, `grad_high = sum(go * u)`; non-test consumer: `pub use Uniform` re-export — Bayesian neural network code with uniform-prior random layers constructs `Uniform` and calls `.rsample(...).backward()`; tests `test_uniform_rsample_{has_grad, no_grad_when_detached, backward}` pin all three paths. |
| REQ-6 | SHIPPED | impl: `Distribution::log_prob` in `uniform.rs` returns `-log(high-low)` in-range else `-inf`, with `UniformLogProbBackward` autograd path when any of `low`/`high`/`value` requires grad; non-test consumer: `pub use Uniform` re-export + impl dispatch; tests `test_uniform_log_prob_{in_range, out_of_range, batch}` pin all three. |
| REQ-7 | SHIPPED | impl: `Distribution::entropy` in `uniform.rs` returns `log(high - low)`, mirroring `uniform.py:107-108`; non-test consumer: `pub use Uniform` re-export; tests `test_uniform_entropy{, _unit}` pin. |
| REQ-8 | SHIPPED | impl: `Distribution::cdf` in `uniform.rs` returns `clamp((x-low)/(high-low), 0, 1)`, mirroring `uniform.py:97-101`; non-test consumer: `pub use Uniform` re-export — downstream sampling-based importance-weighted code (e.g. importance-sampling for SBI) consumes `cdf`; test `test_uniform_cdf_endpoints` pins. |
| REQ-9 | SHIPPED | impl: `Distribution::icdf` in `uniform.rs` returns `low + (high-low)*p`, mirroring `uniform.py:103-105`; non-test consumer: `pub use Uniform` re-export — downstream quantile-regression code consumes `icdf`; test `test_uniform_icdf_linear` pins. |
| REQ-10 | SHIPPED | impl: `Distribution::mean` in `uniform.rs` returns `(low+high)/2`, mirroring `uniform.py:42-44`; non-test consumer: `pub use Uniform` re-export. |
| REQ-11 | SHIPPED | impl: `Distribution::variance` in `uniform.rs` returns `(high-low)^2/12`, mirroring `uniform.py:53-55`; non-test consumer: `pub use Uniform` re-export. |
| REQ-12 | SHIPPED | impl: `Distribution::stddev` in `uniform.rs` returns `(high-low)/sqrt(12)`, mirroring `uniform.py:49-51` (which is `(high-low)/12**0.5`); non-test consumer: `pub use Uniform` re-export. Test `test_uniform_stddev_is_range_over_sqrt12` pins. |
| REQ-13 | SHIPPED | impl: `Distribution::mode` in `uniform.rs` returns the midpoint `(low+high)/2`; R-DEV-6 deviation from upstream `uniform.py:46-48` which returns `nan*high` because every point is equally modal — ferrotorch returns a finite representative value as more useful for downstream code; non-test consumer: `pub use Uniform` re-export. Test `test_uniform_mode_is_midpoint` pins. Blocker #1429 tracks strict-NaN conformance question. |
| REQ-14 | NOT-STARTED | blocker #1430 — `expand`, `support = constraints.interval(low, high)` (`uniform.py:80-83`), `arg_constraints` (`uniform.py:33-39` — these are inter-parameter `low < high` constraints) not implemented; cross-cutting with `lib.md` REQ-5. |
