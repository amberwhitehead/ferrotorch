# ferrotorch-distributions — `gumbel` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/gumbel.py
-->

## Summary

`ferrotorch-distributions/src/gumbel.rs` implements the Gumbel
(Type-I extreme-value) distribution with location `loc` and
scale `scale`. Mirrors `torch.distributions.Gumbel`. Sampling
uses the inverse-CDF transform
`z = loc - scale * ln(-ln(u))`. Used in the Gumbel-Softmax
trick (`RelaxedOneHotCategorical`) and extreme-value
statistics. `rsample` carries gradients via the
`GumbelRsampleBackward` GradFn. Closed-form `log_prob`,
`entropy`, `cdf`, `icdf`, `mean`, `mode`, `variance`.

## Requirements

- REQ-1: `pub struct Gumbel<T: Float>` holding `loc: Tensor<T>`
  and `scale: Tensor<T>`. Mirrors
  `torch/distributions/gumbel.py:Gumbel.__init__`. Note: PyTorch's
  Gumbel inherits from `TransformedDistribution` and composes
  `Uniform` with 4 transforms; ferrotorch implements the closed
  form directly for simplicity + numerical control (R-DEV-7).

- REQ-2: `pub fn Gumbel::new(loc, scale) ->
  FerrotorchResult<Self>` validates shape equality and returns
  `ShapeMismatch` otherwise. PyTorch broadcasts internally
  (`gumbel.py:37-43`); ferrotorch requires pre-broadcasted
  parameters (R-DEV-4).

- REQ-3: `pub fn loc(&self) -> &Tensor<T>` and
  `pub fn scale(&self) -> &Tensor<T>` accessors. Plus the
  utility helpers `pub fn mean_value(&self) -> FerrotorchResult<Vec<T>>`
  and `pub fn variance_value(&self) -> FerrotorchResult<Vec<T>>`
  that return raw `Vec<T>` for the closed-form mean / variance
  (used internally by the `mean` / `variance` tensor overrides
  and externally available for diagnostic/probing code).

- REQ-4: `impl<T: Float> Distribution<T> for Gumbel<T>` provides
  `sample` / `rsample` / `log_prob` / `entropy` + property
  overrides `cdf` / `icdf` / `mean` / `mode` / `variance`.

- REQ-5: Private `fn gumbel_icdf<T>(u, loc, scale)` helper
  computes `loc - scale * ln(-ln(u_safe))` with
  `u_safe = max(eps, min(1 - eps, u))`, `eps = 1e-20`. The
  double-log requires clamping `u` away from both 0 (where
  `-ln(u) → ∞`) and 1 (where `-ln(u) → 0` and `ln(-ln(u)) → -∞`).

- REQ-6: `sample(shape)` and `rsample(shape)` both draw
  `u ~ Uniform(0, 1)` and apply `gumbel_icdf` per element.
  `rsample` additionally attaches `GumbelRsampleBackward` when
  either parameter requires grad.

- REQ-7: `log_prob(x) = -(z + exp(-z)) - ln(scale)` where
  `z = (x - loc) / scale`. Standard Gumbel log-PDF; mirrors
  PyTorch's explicit-precision formula
  (`gumbel.py:67-72`, which uses `y = (loc - value)/scale` and
  computes `(y - y.exp()) - scale.log()`; ferrotorch uses the
  algebraically-equivalent form with `z = -y`).

- REQ-8: `entropy = 1 + ln(scale) + γ` where `γ` is the
  Euler-Mascheroni constant `0.5772156649015329`. Mirrors
  `gumbel.py:90-91`.

- REQ-9: `cdf(x) = exp(-exp(-(x - loc)/scale))`.
  `icdf(p) = loc - scale * ln(-ln(p))`. Standard inverse-CDF
  pair. PyTorch derives both via `TransformedDistribution`'s
  forward + inverse machinery; ferrotorch computes them
  directly.

- REQ-10: Property overrides: `mean = loc + scale * γ`,
  `mode = loc`, `variance = (π * scale)² / 6`. Mirrors
  `gumbel.py:74-88`.

- REQ-11: `GumbelRsampleBackward` GradFn implements:
  ```text
  d(z)/d(loc)   = 1
  d(z)/d(scale) = -ln(-ln(u_safe))
  ```
  Summed across the sample tensor into scalar gradients.

- REQ-12: NOT-STARTED — `expand`, `arg_constraints`, `support`,
  `validate_args`, the `TransformedDistribution` base-class
  hooks (`gumbel.py:17-65`) not implemented. Cross-cutting with
  `lib.md` REQ-5 (blocker #1376). Tracked as blocker #1419 for
  the Gumbel-side fill-out.

## Acceptance Criteria

- [x] AC-1: `pub struct Gumbel<T: Float>` with `loc`, `scale`.
- [x] AC-2: `pub fn Gumbel::new` with shape-equality check.
- [x] AC-3: `pub fn loc` / `scale` accessors + `mean_value` /
  `variance_value` utility methods.
- [x] AC-4: `impl Distribution<T> for Gumbel<T>` with all four
  required trait methods + property overrides.
- [x] AC-5: Private `fn gumbel_icdf<T>` helper.
- [x] AC-6: `sample` / `rsample` via `gumbel_icdf`.
- [x] AC-7: `log_prob = -(z + exp(-z)) - ln(scale)`.
- [x] AC-8: `entropy = 1 + ln(scale) + γ`.
- [x] AC-9: `cdf` / `icdf` overrides.
- [x] AC-10: `mean` / `mode` / `variance` overrides.
- [x] AC-11: `GumbelRsampleBackward` GradFn.
- [x] AC-12: `test_gumbel_*` test suite (15 tests).
- [ ] AC-13: `expand` / `validate_args` — blocker #1419.

## Architecture

### Constructor + accessors (REQ-1, REQ-2, REQ-3)

`Gumbel::new(loc, scale)` enforces shape equality. The struct
stores both tensors by value. The four-method accessor surface
matches the PyTorch attribute names (`loc`, `scale`) plus the
two utility helpers `mean_value` and `variance_value` that
return raw `Vec<T>` representations — these are used internally
by the tensor-returning `mean` / `variance` trait methods to
avoid duplicating the closed-form math.

### `gumbel_icdf` private helper (REQ-5)

```rust
let eps = 1e-20;
let u_safe = u.max(eps).min(1.0 - eps);
loc - scale * (-u_safe.ln()).ln()
```

Tight clamp at `1e-20` because `-ln(0.999999...) ≈ 1e-6`, and
`ln(1e-6) = -13.8` is still well within f32 range. The
clamp prevents the chain `u = 1 → -ln(1) = 0 → ln(0) = -inf`
from producing infinite samples.

### `sample` (REQ-6) and `log_prob` (REQ-7)

```rust
sample:    u ~ Uniform(0, 1); z = gumbel_icdf(u, loc, scale)
log_prob:  z = (x - loc) / scale
           -(z + (-z).exp()) - scale.ln()
```

The `log_prob` form is the algebraic equivalent of PyTorch's
`(y - y.exp()) - scale.log()` with `y = (loc - x)/scale = -z`.
The two forms produce identical f32 / f64 results modulo
last-ULP differences in the sign-handling of `exp`. Tests
(`test_gumbel_log_prob_at_loc/_nonzero/_with_scale`) pin the
numerical contract.

### `cdf` / `icdf` (REQ-9)

```rust
cdf(x)  = exp(-exp(-(x - loc)/scale))
icdf(p) = loc - scale * ln(-ln(p))
```

`test_gumbel_cdf_at_loc_is_one_over_e` pins `cdf(loc) = 1/e ≈
0.3679` (the double-exponential at zero). `test_gumbel_icdf_roundtrip`
verifies the round-trip `cdf(icdf(p)) ≈ p` for `p ∈ {0.1, 0.3,
0.5, 0.7, 0.9}` to 1e-10 in f64.

### Property overrides (REQ-10)

`mean = loc + scale * 0.5772156649015329` (Euler-Mascheroni
constant). `mode = loc.clone()`. `variance = (π * scale)² / 6`.
The Euler constant is the source of the "+γ" term in the mean
formula — it comes from the integral
`∫ x * e^(-x - e^(-x)) dx = γ`.

### `GumbelRsampleBackward` (REQ-11)

```rust
d(z)/d(loc)   = 1                       ; grad_loc = sum(grad_output)
d(z)/d(scale) = -ln(-ln(u_safe))        ; grad_scale = sum(grad_output * -ln(-ln(u_safe)))
```

Same clamping logic as the forward pass. Test
`test_gumbel_rsample_backward` checks `grad_loc = 10.0` for a
`rsample([10]).sum_all().backward()` invocation (each output's
contribution is 1, so summing 10 of them gives 10).

### Non-test production consumers

- **`pub use gumbel::Gumbel` in lib.rs** — grandfathered public
  surface (S5).
- **`Distribution` trait dispatch via `pub use Gumbel`** — every
  external invocation hits this impl block; PyTorch documents
  the Gumbel-Softmax trick as the primary use case
  (`gumbel.py:17-31`), and `RelaxedOneHotCategorical` is the
  ferrotorch consumer that depends on Gumbel-style noise.

## Parity contract

`parity_ops = []`. Closed-form distribution.

Edge cases:

- **`u ∈ {0, 1}`**: clamped to `[1e-20, 1 - 1e-20]` before the
  double-log.
- **`log_prob` precision near `z = 0`**: the formula
  `-(z + exp(-z)) - ln(scale)` is well-conditioned for moderate
  `z`; for very large positive `z`, `exp(-z) → 0` and
  `log_prob → -z - ln(scale)` (asymptotic linear decay).
  For very large negative `z`, `exp(-z) → ∞` and
  `log_prob → -exp(-z)` (exponential blow-down).
- **`cdf` at `loc`**: returns `exp(-1) ≈ 0.3679` — the
  signature of the Gumbel double-exponential.
- **`f64`**: `test_gumbel_f64`.

## Verification

Unit tests (15 tests):

- Sample shape + grad: `test_gumbel_sample_shape`,
  `test_gumbel_rsample_has_grad`.
- `log_prob`: `test_gumbel_log_prob_at_loc/_nonzero/_with_scale`.
- `entropy`: `test_gumbel_entropy/_scale2`.
- Constructor: `test_gumbel_shape_mismatch`.
- Backward: `test_gumbel_rsample_backward`.
- `f64`: `test_gumbel_f64`.
- Properties + CDF/ICDF: `test_gumbel_cdf_at_loc_is_one_over_e`,
  `test_gumbel_icdf_roundtrip`,
  `test_gumbel_mean_mode_variance`.

Smoke command:

```bash
cargo test -p ferrotorch-distributions --lib gumbel:: 2>&1 | tail -3
```

Expected: `15 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Gumbel<T: Float>` with `loc`, `scale` fields in `gumbel.rs` mirroring `torch/distributions/gumbel.py:17-59`; non-test consumer: `pub use gumbel::Gumbel` in `lib.rs` (grandfathered public surface per goal.md S5). |
| REQ-2 | SHIPPED | impl: `pub fn Gumbel::new` in `gumbel.rs` with shape-equality check mirroring `gumbel.py:37-59`; non-test consumer: `pub use Gumbel` re-export. |
| REQ-3 | SHIPPED | impl: `pub fn Gumbel::loc/scale/mean_value/variance_value` accessors in `gumbel.rs` mirroring `gumbel.py:74-88`; non-test consumer: `fn Gumbel::mean` and `fn Gumbel::variance` (trait impl) invoke `self.mean_value()?` and `self.variance_value()?` internally — these are production callsites of the public utility methods. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Distribution<T> for Gumbel<T>` in `gumbel.rs` mirroring `gumbel.py:67-91, 74-88`; non-test consumer: trait dispatch via `pub use Gumbel`. |
| REQ-5 | SHIPPED | impl: private `fn gumbel_icdf<T>` in `gumbel.rs` with `1e-20` clamp; non-test consumer: invoked by both `fn Gumbel::sample` and `fn Gumbel::rsample`. |
| REQ-6 | SHIPPED | impl: `fn Gumbel::sample` / `rsample` in `gumbel.rs` invoking `gumbel_icdf` per element, mirroring PyTorch's `TransformedDistribution`-based composition at `gumbel.py:53-59`; non-test consumer: external `dist.sample/rsample(shape)` calls. |
| REQ-7 | SHIPPED | impl: `fn Gumbel::log_prob` in `gumbel.rs` with `-(z + exp(-z)) - ln(scale)` formula mirroring `gumbel.py:67-72`; non-test consumer: external `dist.log_prob(value)` calls; `test_gumbel_log_prob_at_loc/_nonzero/_with_scale` pin the analytical values. |
| REQ-8 | SHIPPED | impl: `fn Gumbel::entropy` in `gumbel.rs` with `1 + ln(scale) + γ` formula mirroring `gumbel.py:90-91`; non-test consumer: external `dist.entropy()` calls. |
| REQ-9 | SHIPPED | impl: `fn Gumbel::cdf` / `icdf` in `gumbel.rs` with `exp(-exp(-z))` / `-ln(-ln(p))` formulas mirroring the `TransformedDistribution`-derived equivalents at `gumbel.py:53-59`; non-test consumer: external `dist.cdf(value)` / `dist.icdf(q)` calls; `test_gumbel_cdf_at_loc_is_one_over_e` and `test_gumbel_icdf_roundtrip` pin the round-trip. |
| REQ-10 | SHIPPED | impl: `fn Gumbel::{mean, mode, variance}` overrides in `gumbel.rs` mirroring `gumbel.py:74-88`; non-test consumer: external `dist.{mean, mode, variance}` calls; `test_gumbel_mean_mode_variance` pins all three. |
| REQ-11 | SHIPPED | impl: `struct GumbelRsampleBackward<T: Float>` with `GradFn::backward` in `gumbel.rs` (sum-of-grad and `-ln(-ln(u))`-weighted-sum); non-test consumer: invoked by `fn Gumbel::rsample` when either parameter requires grad; `test_gumbel_rsample_backward` pins `grad_loc = 10.0`. |
| REQ-12 | NOT-STARTED | blocker #1419 — `expand`, `arg_constraints`, `support`, `validate_args`, full `TransformedDistribution` base hooks (from `gumbel.py:33-65`) not implemented. Cross-cutting with `lib.md` REQ-5 (blocker #1376). |
