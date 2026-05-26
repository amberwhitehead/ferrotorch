# ferrotorch-distributions — crate root + `Distribution` trait

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/__init__.py
  - torch/distributions/distribution.py
-->

## Summary

`ferrotorch-distributions/src/lib.rs` is the crate root for the
probability-distribution layer. It defines the `Distribution<T>` trait
that every concrete distribution implements (28+ in this crate),
declares the submodule tree (constraints, transforms, kl, fallback,
special_fns, independent, plus 26 concrete distribution modules), and
re-exports the public API surface (`pub use bernoulli::Bernoulli`,
etc.). Mirrors `torch/distributions/__init__.py` (re-export surface)
and `torch/distributions/distribution.py` (the `class Distribution`
base class).

## Requirements

- REQ-1: `pub trait Distribution<T: Float>: Send + Sync` with four
  required methods — `sample(&self, shape) -> FerrotorchResult<Tensor<T>>`,
  `rsample(&self, shape) -> FerrotorchResult<Tensor<T>>`,
  `log_prob(&self, value) -> FerrotorchResult<Tensor<T>>`,
  `entropy(&self) -> FerrotorchResult<Tensor<T>>`. Mirrors
  `torch/distributions/distribution.py:167-255`
  (`Distribution.sample`, `rsample`, `log_prob`, `entropy`). The
  `Send + Sync` bound is the Rust-side analog of PyTorch's
  thread-safety expectation when distributions are used in
  multi-worker training loops (R-DEV-7).

- REQ-2: Default-implemented `Distribution` property methods —
  `batch_shape() -> Vec<usize>`, `cdf(&self, value)`, `icdf(&self, q)`,
  `mean()`, `mode()`, `variance()`, `stddev()`. Each default returns
  either an empty `Vec` (for `batch_shape`) or a structured
  `InvalidArgument` error (for the others) so concrete distributions
  only override what they can express in closed form. Mirrors
  `torch/distributions/distribution.py:108-165` (`batch_shape`,
  `event_shape`, `arg_constraints`, `support`, `mean`, `mode`,
  `variance`, `stddev` properties) — though ferrotorch ships `stddev`
  with a working default (`sqrt(variance)`) just like upstream
  (`distribution.py:160-165`).

- REQ-3: Crate root declares `pub mod constraints`, `pub mod
  transforms`, `pub mod kl`, `pub(crate) mod fallback`, `pub(crate)
  mod special_fns`, plus private modules for each concrete
  distribution (`bernoulli`, `beta`, etc.) and `pub use` re-exports
  for every public type the crate ships. Mirrors
  `torch/distributions/__init__.py:74-119` (the `from .X import Y`
  block) and the explicit `__all__` (`__init__.py:125-174`).

- REQ-4: `pub trait Distribution<T>` is parameterised by `T: Float`
  (the `ferrotorch_core::dtype::Float` trait — currently `f32` or
  `f64`). This is the Rust-side analog of PyTorch's runtime
  dtype-dispatch (`Distribution` is non-generic in upstream because
  Python tensors carry their dtype at runtime). The single generic
  parameter pushes type-mismatch errors to the call site (R-DEV-5).

- REQ-5: SHIPPED — the full PyTorch `Distribution` surface lands on
  the trait with sensible defaults: `event_shape() -> Vec<usize>`
  (default empty), `has_rsample() -> bool` (default false),
  `has_enumerate_support() -> bool` (default false), `support() ->
  Option<Box<dyn DistConstraint>>` (default None), `arg_constraints()
  -> HashMap<&'static str, Box<dyn DistConstraint>>` (default empty),
  `expand(&self, &[usize]) -> FerrotorchResult<Box<dyn Distribution<T>>>`
  (default InvalidArgument), `enumerate_support(&self, bool)`
  (default InvalidArgument), `perplexity()` (default
  `exp(self.entropy()?)`). The `_validate_sample` / `validate_args`
  pipeline (REQ-future) is deliberately scoped out — those require a
  separate construction-time flag plumbing pass and tracker (R-DEFER:
  the current trait surface gives consumers everything they need to
  introspect a distribution; the validation pipeline is an
  orthogonal concern). The object-safe `DistConstraint` super-trait
  is the load-bearing piece making `support` / `arg_constraints`
  dyn-compatible.

## Acceptance Criteria

- [x] AC-1: `pub trait Distribution<T: Float>: Send + Sync` exists in
  `lib.rs` with the four required methods (`sample`, `rsample`,
  `log_prob`, `entropy`) and the seven default-implemented property
  methods.
- [x] AC-2: The crate re-exports every concrete distribution from
  the type table in the module doc-comment via `pub use`.
- [x] AC-3: `pub mod constraints` / `pub mod transforms` / `pub mod
  kl` are accessible to crate consumers; `fallback` and `special_fns`
  are `pub(crate)` (crate-internal only).
- [x] AC-4: `Distribution::stddev` has a working default body that
  invokes `self.variance()?` and maps `sqrt` element-wise — exercised
  indirectly by any distribution that does not override stddev (e.g.
  `Cauchy` overrides because its variance is undefined; most others
  fall back to the default).
- [x] AC-5: `arg_constraints` / `support` / `expand` /
  `enumerate_support` / `event_shape` / `has_rsample` /
  `has_enumerate_support` / `perplexity` are on the trait with
  defaults; concrete overrides land in Normal/Bernoulli/Exponential
  /Gamma/Uniform/Categorical (#1376 closed).

## Architecture

### `Distribution` trait surface (REQ-1, REQ-4)

`pub trait Distribution<T: Float>: Send + Sync` in `lib.rs` is the
crate-wide trait every concrete distribution implements. The four
required methods (`sample`, `rsample`, `log_prob`, `entropy`) are
exactly the PyTorch ABI surface a user calls in code like
`Normal(0,1).log_prob(x)` — see
`torch/distributions/distribution.py:167` (`sample`),
`distribution.py:175` (`rsample`), `distribution.py:194` (`log_prob`),
`distribution.py:248` (`entropy`).

The trait is generic over `T: Float` rather than over a dtype enum at
runtime: this lets the compiler monomorphise per-dtype, eliminating
the runtime dispatch overhead Python pays (R-DEV-1 — numerical
contract enforced by the type system). The trade-off is that user
code that wants to handle both `f32` and `f64` must be itself generic
or branch at type-construction time, which mirrors the PyTorch-user
experience of choosing dtype at tensor construction.

`Send + Sync` is the load-bearing super-trait: it forbids
`Rc<RefCell<T>>` / non-thread-safe interior mutability inside any
distribution, which is what makes distributions safely usable from
the parallel DataLoader pipeline (`ferrotorch-data::DataLoader`).

### Default-implemented property methods (REQ-2)

Seven property methods are default-implemented in `lib.rs`:

- `batch_shape(&self) -> Vec<usize> { vec![] }` — the shape of
  parameter tensors. Default: empty (scalar batch). Concrete
  distributions with batched parameters (e.g.
  `Normal` with `loc: shape [B]`) override to return `vec![B]`.
- `cdf`, `icdf`, `mean`, `mode`, `variance` — each defaults to
  `Err(FerrotorchError::InvalidArgument { message: "X not implemented
  for this distribution".into() })`. This mirrors PyTorch's
  `raise NotImplementedError` (`distribution.py:140-158`) but with
  structured Rust errors.
- `stddev` — defaults to `sqrt(variance)`:
  `let v = self.variance()?; let data = v.data_vec()?; ...`. Matches
  upstream `distribution.py:160-165` `def stddev(self): return
  self.variance.sqrt()`.

The default-error approach (vs. PyTorch's `NotImplementedError`)
lets concrete distributions skip implementing methods they don't
need, while still surfacing a useful diagnostic. The
`stddev` default propagates any error from `variance`, which is the
correct PyTorch-faithful behaviour.

### Module tree and re-exports (REQ-3)

`lib.rs` declares 33 submodules: 3 public (`constraints`,
`transforms`, `kl`), 2 crate-internal (`fallback`, `special_fns`),
1 wrapper (`independent`), and 27 private concrete distributions.
The `pub use` block at `lib.rs:84-111` re-exports every public type
from those modules: `pub use bernoulli::Bernoulli`,
`pub use transforms::TransformedDistribution`, etc.

Mirrors `torch/distributions/__init__.py:74-119` (the
`from .X import Y` block) and the explicit `__all__` re-export
contract at `__init__.py:125-174`.

### Distribution-surface gap (REQ-5)

PyTorch's `Distribution` class carries surface ferrotorch's trait
does NOT: `arg_constraints` dict, `support` property, `expand`,
`enumerate_support`, `perplexity`, `_validate_sample`, the
`validate_args` boolean path, the `has_rsample` /
`has_enumerate_support` class-level flags, and the explicit
`event_shape` accessor distinct from `batch_shape`. See
`distribution.py:25` (`has_rsample = False`),
`distribution.py:108-119` (`batch_shape` / `event_shape`
properties), `distribution.py:122-138` (`arg_constraints` /
`support`), `distribution.py:86-105` (`expand`),
`distribution.py:224-246` (`enumerate_support`), and
`distribution.py:257-264` (`perplexity`).

Adding these requires (a) wiring concrete distributions to declare
their `arg_constraints` map and `support` constraint object — which
in turn requires the `constraints` module to have production
consumers (`constraints.md` REQ-N), and (b) modifying every concrete
distribution to expose `event_shape` separately from `batch_shape`.
This is cross-cutting and tracked under blocker #1376.

### Non-test production consumers

- The 26 concrete-distribution modules — `bernoulli.rs`, `beta.rs`,
  `categorical.rs`, `cauchy.rs`, `dirichlet.rs`, `exponential.rs`,
  `gamma.rs`, `gumbel.rs`, `half_normal.rs`, `kumaraswamy.rs`,
  `laplace.rs`, `lognormal.rs`, `low_rank_multivariate_normal.rs`,
  `mixture_same_family.rs`, `multinomial.rs`,
  `multivariate_normal.rs`, `normal.rs`, `one_hot_categorical.rs`,
  `pareto.rs`, `poisson.rs`, `relaxed_bernoulli.rs`,
  `relaxed_one_hot_categorical.rs`, `student_t.rs`, `uniform.rs`,
  `von_mises.rs`, `weibull.rs` — each carries `use crate::Distribution;`
  and `impl<T: Float> Distribution<T> for X<T>`. Confirmed via
  `grep -rn "impl Distribution\|use crate::Distribution"
  ferrotorch-distributions/src/`. The Distribution trait IS the API
  the whole crate is structured around.
- `transforms::TransformedDistribution<T>` in `transforms.rs`
  consumes `Box<dyn Distribution<T>>` as its base, making the trait
  object surface its primary external consumer.
- `independent::Independent<T, D>` in `independent.rs` is generic
  over `D: Distribution<T>` and forwards `sample`/`rsample`/
  `log_prob`/`entropy` to the base.
- `kl::kl_divergence<T, P, Q>` in `kl.rs` is bounded by
  `P: Distribution<T> + 'static, Q: Distribution<T> + 'static`,
  using the trait's `'static` requirement as the dispatch entry
  point.

## Parity contract

`parity_ops = []`. This file ships no parity-sweep ops directly;
it's the trait surface every concrete distribution's parity-sweep
suite indirectly verifies. Edge cases the trait contract preserves:

- **`Send + Sync` enforcement**: any concrete distribution holding
  `Rc<RefCell<T>>` or `*mut`-bearing fields fails to compile. The
  anti-pattern hook (`tooling/anti-pattern-gate.py`) flags these on
  write.
- **Error propagation**: `Distribution::stddev`'s default propagates
  any `variance()` error via `?`. A distribution whose variance is
  undefined (e.g. `Cauchy`) returns `InvalidArgument`, which `stddev`
  forwards verbatim — matching PyTorch's `NotImplementedError`
  bubble-up.
- **Default `batch_shape`**: returns `vec![]` (scalar batch). The
  empty-vec semantics differ from PyTorch's `torch.Size([])`
  numerically but match it semantically — both indicate "no
  batching".
- **`rsample` for discrete distributions**: every discrete
  distribution (`Bernoulli`, `Categorical`, `Multinomial`,
  `Poisson`) returns
  `Err(FerrotorchError::InvalidArgument { message: "rsample not
  available for this discrete distribution" })`. Matches upstream's
  `Distribution.rsample` raising `NotImplementedError`
  (`distribution.py:181`).

## Verification

No unit tests live in `lib.rs` directly — every concrete
distribution's `mod tests` exercises the trait surface via
`<Distribution>::sample / rsample / log_prob / entropy`. The 380
library tests cover the trait via specific implementations:

- `bernoulli::tests::*` (~10 tests on the trait surface)
- `normal::tests::*` (~15 tests)
- `transforms::tests::test_transformed_distribution_*` (5 tests on
  the `Box<dyn Distribution>` consumer path)
- `independent::tests::*` (4 tests on the wrapper consumer)
- ... × 26 concrete distributions.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib 2>&1 | tail -3
```

Expected: `380 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub trait Distribution<T: Float>: Send + Sync` with four required methods (`sample`, `rsample`, `log_prob`, `entropy`) in `pub trait Distribution in lib.rs`, mirroring `torch/distributions/distribution.py:167-255`; non-test consumers: `impl<T: Float> Distribution<T> for Normal<T> in normal.rs`, `impl<T: Float> Distribution<T> for Bernoulli<T> in bernoulli.rs`, `impl<T: Float> Distribution<T> for Gamma<T> in gamma.rs` and 23 other concrete `impl Distribution` sites confirmed via grep across the crate. |
| REQ-2 | SHIPPED | impl: seven default property methods (`batch_shape`, `cdf`, `icdf`, `mean`, `mode`, `variance`, `stddev`) in `pub trait Distribution in lib.rs` mirroring `torch/distributions/distribution.py:108-165`; non-test consumer: `fn Independent::batch_shape in independent.rs` overrides the default to forward to the base, exercising the override path in production; `fn TransformedDistribution::entropy in transforms.rs` calls `self.base.mean()?` on the trait object — that is the default-implementation production-consumer path. |
| REQ-3 | SHIPPED | impl: module tree + `pub use` re-export block in `lib.rs` (mod declarations at top, `pub use bernoulli::Bernoulli` through `pub use weibull::Weibull`), mirroring `torch/distributions/__init__.py:74-119`; non-test consumer: `tests/conformance_distributions_continuous.rs` and `tests/conformance_distributions_discrete.rs` import the re-exports as `use ferrotorch_distributions::{Normal, Bernoulli, ...}`; `ferrotorch-vae` / `ferrotorch-bert` (downstream) similarly import from the meta-crate re-export. |
| REQ-4 | SHIPPED | impl: `pub trait Distribution<T: Float>` generic-over-`T` declaration in `lib.rs` with explicit `T: Float` bound on every method signature, mirroring PyTorch's runtime dtype dispatch (R-DEV-7: monomorphise per-dtype instead); non-test consumer: every concrete distribution carries `pub struct Normal<T: Float>` / `pub struct Gamma<T: Float>` etc. with `impl<T: Float> Distribution<T> for ...`, so the generic surface IS the production wiring. f32 and f64 are both exercised by `normal::tests::*_f64` and similar tests per family. |
| REQ-5 | SHIPPED | impl: `event_shape` / `has_rsample` / `has_enumerate_support` / `support` / `arg_constraints` / `expand` / `enumerate_support` / `perplexity` defaults on `pub trait Distribution` in `ferrotorch-distributions/src/lib.rs` mirroring `torch/distributions/distribution.py:25-264`; object-safe `pub trait DistConstraint` super-trait in `lib.rs` with blanket `impl<C: constraints::Constraint + Debug + 'static> DistConstraint for C` makes `Box<dyn DistConstraint>` viable without breaking the existing `constraints::Constraint::check<T>` generic-method surface; non-test consumers: `fn Normal::has_rsample` / `Normal::support` / `Normal::arg_constraints` / `Normal::expand` in `normal.rs`, `fn Bernoulli::{has_rsample, has_enumerate_support, support, arg_constraints, enumerate_support, expand}` in `bernoulli.rs`, `fn Exponential::{has_rsample, support, arg_constraints, expand}` in `exponential.rs`, `fn Gamma::{has_rsample, support, arg_constraints, expand}` in `gamma.rs`, `fn Uniform::{has_rsample, support, arg_constraints, expand}` in `uniform.rs`, `fn Categorical::{has_enumerate_support, support, arg_constraints, enumerate_support, expand}` in `categorical.rs`. Closed: #1376 (umbrella), #1406 (bernoulli expand/enumerate/arg_constraints), #1414 (exponential expand/support/arg_constraints), #1410 (categorical expand/enumerate/arg_constraints partial — `mean`/`mode`/`variance` and N-D batched probs remain), #1416 (gamma expand/support/arg_constraints — `cdf` via incomplete gamma still NOT-STARTED), #1430 (uniform expand/support/arg_constraints). `_validate_sample` / `validate_args` deferred — orthogonal construction-time wiring tracked separately. |
