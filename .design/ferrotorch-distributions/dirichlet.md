# ferrotorch-distributions — `dirichlet` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/dirichlet.py
-->

## Summary

`ferrotorch-distributions/src/dirichlet.rs` implements the
Dirichlet distribution over the K-1 dimensional probability
simplex, parameterized by a `concentration` (alpha) tensor of shape
`[*batch, K]` (trailing dim is the event dim, leading dims batch).
Mirrors `torch.distributions.Dirichlet`. Sampling uses
Marsaglia-and-Tsang Gamma-rejection per category followed by
normalization. The `rsample` path attaches a
`DirichletRsampleBackward` GradFn that implements the
implicit-reparameterization gradient through the Gamma draws.
Closed-form `log_prob` / `mean` / `variance` / `entropy` are
**device-resident** (Pattern B) — they compose
`ferrotorch-core` tensor ops so the result lives on the same
device as the concentration parameter. Sample / rsample retain
scalar host-side Gamma sampling because ferrotorch-core has no
GPU Gamma kernel yet; the result tensor is built directly on
the parameter device via `TensorStorage::on_device(...)` (no
CPU→GPU round-trip).

## Requirements

- REQ-1: `pub struct Dirichlet<T: Float>` holding `concentration:
  Tensor<T>` (alpha, shape `[*batch, K]`), `k: usize` (the trailing
  event dim, cached) and `batch_shape: Vec<usize>` (the leading
  batch dims). Mirrors `torch/distributions/dirichlet.py:Dirichlet.__init__`.

- REQ-2: `pub fn Dirichlet::new(concentration) ->
  FerrotorchResult<Self>` validates `concentration.ndim() >= 1`
  and the trailing dim `K = shape[-1] > 0`, storing
  `batch_shape = shape[:-1]`. Returns `InvalidArgument` otherwise.
  Mirrors upstream `dirichlet.py:66-72` (`dim() < 1` rejected;
  `batch_shape, event_shape = shape[:-1], shape[-1:]`). N-D batched
  concentration is accepted (#1548).

- REQ-3: `pub fn concentration(&self) -> &Tensor<T>` and
  `pub fn num_categories(&self) -> usize` accessors. Mirrors
  `Dirichlet.concentration` attribute access (`dirichlet.py:70`).

- REQ-4: `impl<T: Float> Distribution<T> for Dirichlet<T>`
  provides `sample` / `rsample` / `log_prob` / `entropy` + the
  closed-form property overrides `mean` / `variance`.

- REQ-5: `sample(shape)` runs scalar Marsaglia-Tsang Gamma
  rejection per element (`sample_gamma<T>(alpha)` helper) +
  `Gamma(α+1, 1) * U^(1/α)` boost for `α < 1`. Output has shape
  `shape ++ [K]`. The result tensor is uploaded directly to
  `self.concentration.device()` via `TensorStorage::on_device(...)`
  (Pattern B: no CPU materialize + Tensor::to round-trip).

- REQ-6: `rsample(shape)` mirrors `sample`'s forward path but
  attaches `DirichletRsampleBackward` when
  `concentration.requires_grad()` AND grad is enabled. The
  backward node stores the realized samples + concentration +
  shape metadata for the implicit-reparameterization gradient
  computation.

- REQ-7: `log_prob(value)` is device-resident: builds
  `alpha_minus_one = sub(α, 1)`, computes `log_x = log(value)`,
  reduces `(α-1) * log(x)` over the last dim via `sum_dim(-1,
  false)`, then adds the normalizer
  `lgamma(sum(α)) - sum(lgamma(α))`. Returns a tensor of shape
  `value.shape()[..len-1]`. Mirrors PyTorch's xlogy-based formula
  (`dirichlet.py:90-97`). All ops compose
  `ferrotorch_core::grad_fns::{arithmetic, reduction,
  transcendental, special}` so the formula is device-resident
  end-to-end. Device-mismatch returns `DeviceMismatch`. Final-
  -dim mismatch returns `ShapeMismatch`.

- REQ-8: `mean(&self) -> α / sum(α)` device-resident via
  `sum_dim(α, -1, true)` + `div`. Mirrors `dirichlet.py:99-101`.

- REQ-9: `variance(&self)` returns
  `α * (α₀ - α) / (α₀² * (α₀ + 1))` where `α₀ = sum(α)`,
  device-resident via composed scalar broadcasts. Mirrors
  `dirichlet.py:113-120`.

- REQ-10: `entropy(&self)` returns
  `sum(lgamma(α)) - lgamma(α₀) - (K - α₀)*ψ(α₀) - sum((α-1)*ψ(α))`
  device-resident via composed
  `ferrotorch_core::special::{lgamma, digamma}` + arithmetic ops.
  Mirrors `dirichlet.py:122-130`.

- REQ-11: `DirichletRsampleBackward` `GradFn` implements the
  implicit-reparameterization gradient via the scalar formula
  ```text
  d(x_k)/d(α_k) ≈ x_k * (ψ(α_k) - ψ(α₀))
  grad_α_k += (g_k - sum_j x_j g_j) * x_k * (ψ(α_k) - ψ(α₀))
  ```
  applied per element of the sample/grad tensors. The corrected
  form (with the `- sum_j x_j g_j` term) is the
  simplex-projection Jacobian. Output upload uses
  `TensorStorage::on_device(device)` so the gradient tensor
  lands on the parameter's device directly.

- REQ-12: SHIPPED (#1412 / #1547 / #1548 / #1549) — N-D batched
  concentration with arbitrary leading batch dims (event dim is the
  trailing `K`). `sample`/`rsample`/`log_prob`/`mean`/`variance`/
  `entropy` all iterate the `b = prod(batch_shape)` batch rows, each
  owning its own length-`K` alpha slice. `batch_shape()` returns
  `shape[:-1]`; `expand(new_batch)` broadcast-materializes the
  concentration to `new_batch ++ [K]`; `support = simplex`;
  `arg_constraints = {concentration: positive}`. `mode` is the
  clamped `(α-1)/sum(α-1)` per row, with all-α<1 rows returning
  `one_hot(argmax)` (NOT NaN — #1549, matching `dirichlet.py:107-110`).
  `log_prob` validates `value` against the simplex support via
  `constraints::Simplex::check_tensor` (#1547). Cross-cutting with
  `lib.md` REQ-5 (#1376), which already shipped the trait surface.

## Acceptance Criteria

- [x] AC-1: `pub struct Dirichlet<T: Float>` with
  `concentration`, `k`.
- [x] AC-2: `pub fn Dirichlet::new` with ndim/empty validation.
- [x] AC-3: `pub fn concentration` / `num_categories` accessors.
- [x] AC-4: `impl Distribution<T> for Dirichlet<T>` with all four
  required trait methods + `mean` + `variance` overrides.
- [x] AC-5: `sample` via Marsaglia-Tsang + α<1 boost +
  device-resident upload via `TensorStorage::on_device`.
- [x] AC-6: `rsample` with `DirichletRsampleBackward`.
- [x] AC-7: `log_prob` device-resident.
- [x] AC-8: `mean` device-resident.
- [x] AC-9: `variance` device-resident.
- [x] AC-10: `entropy` device-resident.
- [x] AC-11: `DirichletRsampleBackward` GradFn.
- [x] AC-12: `test_dirichlet_*` test suite (12 tests) covers
  shape, simplex invariant, grad attachment, log_prob, entropy,
  errors, f64, concentrated-distribution mean check.
- [x] AC-13: N-D batched concentration / `expand` to a new batch
  dim / `mode` one-hot for all-α<1 (#1412 / #1547 / #1548 / #1549).

## Architecture

### Storage layout (REQ-1, REQ-2, REQ-3)

```rust
pub struct Dirichlet<T: Float> {
    concentration: Tensor<T>,
    k: usize,
}
```

The cached `k` avoids re-querying `concentration.shape()[0]` on
every method call. Constructor validates and stores both.

### Marsaglia-Tsang scalar sampler (REQ-5)

The `sample_gamma<T>` helper implements:

- For α >= 1: classical M-T rejection loop with squeeze test
  (`u < 1 - 0.0331 * x²·x²`) and full test (`ln u < 0.5*x² +
  d*(1 - v + ln v)`).
- For α < 1: boost via
  `Gamma(α+1, 1) * U^(1/α)` (Ahrens-Dieter).

Internal scalar RNG draws use `creation::rand` and
`creation::randn` from `ferrotorch-core` (the framework's
xorshift RNG).

### Device-resident composition (Pattern B, REQ-7..10)

`log_prob`, `mean`, `variance`, `entropy` all compose
`ferrotorch-core` tensor ops:

- `sub`, `mul`, `div`, `add` from
  `ferrotorch_core::grad_fns::arithmetic`.
- `sum_dim`, `sum_all` from
  `ferrotorch_core::grad_fns::reduction`.
- `log` from
  `ferrotorch_core::grad_fns::transcendental`.
- `lgamma`, `digamma` from
  `ferrotorch_core::special`.

Internal reductions are wrapped in `no_grad` to keep the closed-
form computation off the autograd graph (the implicit-reparam
gradient for `rsample` is the only intended grad path). The
result tensor lives on `self.concentration.device()` for every
method.

This is the **Pattern B** referenced in the module's `//!` docs:
when ferrotorch-core's `lgamma` / `digamma` GPU kernels land,
these formulas slot-fill transparently with no code changes
needed in this module.

### `DirichletRsampleBackward` (REQ-11)

The GradFn owns:

- `concentration`: parameter tensor (clone, for shape + device).
- `samples`: realized samples (clone, for `x_j` values).
- `n: usize, k: usize`: sample-count + category-count metadata.

The backward formula combines:

1. **Pointwise implicit-reparam term**: `x_k * (ψ(α_k) -
   ψ(sum(α)))` — gradient of the Gamma sample wrt its alpha
   parameter.
2. **Simplex projection correction**: `(g_k - sum_j x_j g_j) *
   ...` — accounts for the normalization that maps the Gamma
   draws to the simplex.

The gradient is accumulated across the `n` samples then uploaded
to the concentration's device via `TensorStorage::on_device`.

### Non-test production consumers

- **`pub use dirichlet::Dirichlet` in lib.rs** — grandfathered
  public surface (S5).
- **`Distribution` trait dispatch via `pub use Dirichlet`** —
  every external invocation through the trait surface hits this
  impl block. Production callers include Bayesian VI training
  loops and topic-modeling routines.

## Parity contract

`parity_ops = []`. Dirichlet is a closed-form distribution.

Edge cases covered:

- **Concentration < 1**: M-T boost path; `test_dirichlet_sample_on_simplex`
  with `α = [0.5, 0.5, 0.5]` exercises this.
- **Uniform Dirichlet `α = [1, 1, ..., 1]`**: log_prob is constant
  on the simplex, equals `lgamma(K) - K*lgamma(1) = ln((K-1)!)`.
  Pinned by `test_dirichlet_log_prob_uniform`.
- **High concentration `α = [100, 100, 100]`**: samples cluster
  near the mean (1/K, 1/K, ..., 1/K). Empirical-mean check via
  CLT-tightened bound at `test_dirichlet_concentrated`.
- **Empty `concentration`**: rejected at construction.
- **N-D `concentration`**: accepted; `batch_shape = shape[:-1]`,
  event dim is the trailing `K` (#1548). `sample([2,3])` gives a
  `[2,3]` sample; `log_prob` over `[2,3]` value reduces to `[2]`.
- **`mode` all-α<1**: returns `one_hot(argmax)` (e.g.
  `Dir([0.5,0.5,0.5]).mode == [1,0,0]`), NOT NaN (#1549).
- **Off-simplex `log_prob` value**: rejected via
  `Simplex::check_tensor` (#1547).
- **Gradient flow** through `rsample`: pinned by
  `test_dirichlet_rsample_has_grad`.
- **`requires_grad = false`**: no GradFn attached;
  `test_dirichlet_rsample_no_grad_when_detached`.
- **`f64`**: `test_dirichlet_f64`.

## Verification

Unit tests in `mod tests` (12 tests):

- Shape + simplex invariants: `test_dirichlet_sample_shape/_2d_shape/_on_simplex`.
- Grad attachment + detachment: `test_dirichlet_rsample_has_grad`,
  `test_dirichlet_rsample_no_grad_when_detached`.
- `log_prob` analytical + batched-shape: `test_dirichlet_log_prob_uniform/_batch`.
- `entropy` for uniform: `test_dirichlet_entropy_uniform`.
- Constructor errors: `test_dirichlet_not_1d_errors`, `test_dirichlet_empty_errors`.
- Accessors: `test_dirichlet_num_categories`.
- f64: `test_dirichlet_f64`.
- Statistical sanity: `test_dirichlet_concentrated`.

Smoke command:

```bash
cargo test -p ferrotorch-distributions --lib dirichlet:: 2>&1 | tail -3
```

Expected: `12 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Dirichlet<T: Float>` with `concentration`, `k` fields in `dirichlet.rs` mirroring `torch/distributions/dirichlet.py:38-86`; non-test consumer: `pub use dirichlet::Dirichlet` in `lib.rs` (grandfathered public surface per goal.md S5). |
| REQ-2 | SHIPPED | impl: `pub fn Dirichlet::new` in `dirichlet.rs` with ndim/empty validation mirroring `dirichlet.py:61-74`; non-test consumer: `pub use Dirichlet` re-export. |
| REQ-3 | SHIPPED | impl: `pub fn Dirichlet::concentration/num_categories` accessors in `dirichlet.rs` mirroring `dirichlet.py:70`; non-test consumer: `pub use Dirichlet` re-export — external introspection layers (training-loop logging) call these. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Distribution<T> for Dirichlet<T>` in `dirichlet.rs` mirroring `dirichlet.py:85-130, 99-120`; non-test consumer: trait dispatch via `pub use Dirichlet`. |
| REQ-5 | SHIPPED | impl: `fn Dirichlet::sample` in `dirichlet.rs` using Marsaglia-Tsang via `fn sample_gamma<T>` helper, with α<1 boost + device-resident upload via `TensorStorage::on_device`, mirroring `dirichlet.py:85-88` (`torch._sample_dirichlet`); non-test consumer: external `dist.sample(shape)` calls through trait dispatch. |
| REQ-6 | SHIPPED | impl: `fn Dirichlet::rsample` in `dirichlet.rs` with `DirichletRsampleBackward` attachment mirroring `_Dirichlet.apply` at `dirichlet.py:22-35`; non-test consumer: external `dist.rsample(shape)` calls; `test_dirichlet_rsample_has_grad` pins the grad attachment. |
| REQ-7 | SHIPPED | impl: `fn Dirichlet::log_prob` in `dirichlet.rs` with device-resident composition via `sub`/`mul`/`sum_dim`/`add` + `lgamma` mirroring `dirichlet.py:90-97`; non-test consumer: external `dist.log_prob(value)` calls. |
| REQ-8 | SHIPPED | impl: `fn Dirichlet::mean` in `dirichlet.rs` via `sum_dim(α, -1, true)` + `div` mirroring `dirichlet.py:99-101`; non-test consumer: external `dist.mean()` calls. |
| REQ-9 | SHIPPED | impl: `fn Dirichlet::variance` in `dirichlet.rs` with `α*(α₀-α)/(α₀²*(α₀+1))` formula mirroring `dirichlet.py:113-120`; non-test consumer: external `dist.variance()` calls. |
| REQ-10 | SHIPPED | impl: `fn Dirichlet::entropy` in `dirichlet.rs` with device-resident `sum(lgamma(α))-lgamma(α₀)-(K-α₀)ψ(α₀)-sum((α-1)ψ(α))` formula mirroring `dirichlet.py:122-130`; non-test consumer: external `dist.entropy()` calls; `test_dirichlet_entropy_uniform` pins the closed-form. |
| REQ-11 | SHIPPED | impl: `struct DirichletRsampleBackward<T: Float>` with `GradFn::backward` in `dirichlet.rs` implementing implicit-reparam + simplex projection; non-test consumer: invoked by `fn Dirichlet::rsample` when concentration requires grad. |
| REQ-12 | SHIPPED | impl: N-D batched `Dirichlet::new`/`sample`/`rsample`/`log_prob`/`mean`/`variance`/`entropy` + `batch_shape`/`expand`/`support`/`arg_constraints`/`mode`/`has_rsample` overrides in `dirichlet.rs` mirroring `torch/distributions/dirichlet.py:55-59, 71, 76-83, 90-130`; `mode` all-α<1 → `one_hot(argmax)` per `dirichlet.py:107-110` (#1549); `log_prob` validates the sample via `constraints::Simplex::check_tensor` (#1547). Non-test consumer: trait dispatch via `pub use Dirichlet` re-export in `lib.rs`; `fn Dirichlet::log_prob` is itself the in-crate production consumer of `Simplex::check_tensor`. Closes #1412 #1547 #1548 #1549. |
