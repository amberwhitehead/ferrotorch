# ferrotorch-distributions — `kl` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/kl.py
-->

## Summary

`ferrotorch-distributions/src/kl.rs` provides analytical closed-form
KL divergence formulas for 84 distribution pairs:
Normal-Normal, Bernoulli-Bernoulli, Uniform-Uniform,
Categorical-Categorical, Normal-Uniform, Uniform-Normal,
Laplace-Laplace, Exponential-Exponential, Gamma-Gamma,
Poisson-Poisson, Gamma-Exponential, Exponential-Gamma,
Beta-Beta, Gumbel-Gumbel, Pareto-Pareto, HalfNormal-HalfNormal,
Exponential-Normal, Gamma-Normal, Laplace-Normal, Cauchy-Cauchy,
Normal-Gumbel, Gumbel-Normal, Gamma-Gumbel, Exponential-Gumbel,
Uniform-Gumbel, Dirichlet-Dirichlet, Beta-Exponential, Beta-Gamma,
Beta-Normal, Beta-Uniform, Pareto-Exponential, Pareto-Gamma,
Pareto-Normal, Uniform-Exponential, Uniform-Gamma, Uniform-Pareto,
Uniform-Beta, MultivariateNormal-MultivariateNormal,
MultivariateNormal-LowRankMultivariateNormal,
LowRankMultivariateNormal-MultivariateNormal,
LowRankMultivariateNormal-LowRankMultivariateNormal.
The #1562 both-types-exist gap closure added 27 more pairs:
the finite OneHotCategorical-OneHotCategorical, Bernoulli-Poisson,
Normal-Laplace, plus the 24 support-mismatch `+inf` cross-pairs
(Beta-Pareto; Exponential-{Beta,Pareto,Uniform};
Gamma-{Beta,Pareto,Uniform};
Gumbel-{Beta,Exponential,Gamma,Pareto,Uniform};
Laplace-{Beta,Exponential,Gamma,Pareto,Uniform};
Normal-{Beta,Exponential,Gamma,Pareto}; Pareto-{Beta,Uniform};
Poisson-Bernoulli).
The #1374 Binomial sub-part added 2 more: the finite
Binomial-Binomial (`kl.py:231-244`) and the support-mismatch
`+inf` Poisson-Binomial (`kl.py:842` `_kl_poisson_infinity`).
The #1374 Geometric sub-part added 1 more: the finite
Geometric-Geometric (`kl.py:320-322`).
The #1374 ContinuousBernoulli sub-part added 13 more (needing the new
`ContinuousBernoulli` struct in `continuous_bernoulli.rs`): 6 finite
(ContinuousBernoulli-ContinuousBernoulli `kl.py:255`; Beta-CB `kl.py:518`;
CB-Exponential `kl.py:586`; CB-Normal `kl.py:595`; CB-Uniform `kl.py:607`
+ Uniform-CB `kl.py:871`, both where-masked `+inf` on support containment)
and 7 support-mismatch `+inf` (CB-Pareto `kl.py:581`;
{Exponential,Gamma,Gumbel,Laplace,Normal,Pareto}-CB
`kl.py:621,666,719,741,762,796`).
Mirrors `torch/distributions/kl.py`. The dispatcher
is a hand-coded chain of `Any::downcast_ref` arms; the same pattern
PyTorch ships via `register_kl` + `_dispatch_kl` but expressed in
Rust without the runtime-class-decorator machinery — this explicit
match is the deliberate Rust-idiomatic design (REQ-8 / #1375).

## Requirements

- REQ-1: `pub fn kl_divergence<T: Float, P, Q>(p: &P, q: &Q) ->
  FerrotorchResult<Tensor<T>>` where `P: Distribution<T> + 'static,
  Q: Distribution<T> + 'static` is the public entry point.
  Internally dispatches to `kl_dispatch(p as &dyn Any, q as &dyn
  Any)`. Mirrors `torch/distributions/kl.py:kl_divergence` (signature
  + docstring). The `'static` bound is what enables `Any`
  downcasting; PyTorch uses class-object lookup for the same
  purpose.

- REQ-2: `pub const fn kl_supported_pair_count() -> usize`
  introspects the dispatcher's registered-pair count. Backed by
  `const KL_SUPPORTED_PAIR_COUNT: usize = 84`. Tested via
  `kl_doc_table_matches_dispatcher` (which parses
  `include_str!("kl.rs")` and counts BOTH the doc-table rows AND
  the `p.downcast_ref::<...>()` arms in `fn kl_dispatch`, asserting
  they all match the constant). This is the drift-prevention
  guard that fixes the historical failure mode of #1124.

- REQ-3: `fn kl_dispatch<T: Float>(p: &dyn Any, q: &dyn Any) ->
  FerrotorchResult<Tensor<T>>` is the type-dispatch core. 84 `if let
  (Some(_), Some(_)) = (p.downcast_ref::<P>(), q.downcast_ref::<Q>())`
  arms cover the 84 registered pairs; fall-through is a structured
  `InvalidArgument` with the full pair list in the error message.
  Mirrors PyTorch's `_dispatch_kl` (`kl.py:113-138`) which uses
  class-hierarchy lookup; ferrotorch's hand-coded chain is the
  Rust analog (R-DEV-4: replaces Python's runtime class dispatch).

- REQ-4: Same-family closed-form formulas (8 pairs). Each is a
  free `fn kl_<p>_<q><T: Float>(p, q) -> FerrotorchResult<Tensor<T>>`:
    - `kl_normal_normal`: `0.5 * (var_ratio + mean_diff^2/var2 - 1 - ln(var_ratio))`
    - `kl_bernoulli_bernoulli`: `p*ln(p/q) + (1-p)*ln((1-p)/(1-q))` (clamped to `[eps, 1-eps]`)
    - `kl_uniform_uniform`: `ln((b2-a2)/(b1-a1))` if `[a1,b1] ⊆ [a2,b2]` else `+inf`
    - `kl_categorical_categorical`: `sum_k p_k * ln(p_k/q_k)`, scalar result
    - `kl_laplace_laplace`: `ln(b2/b1) + (b1 * exp(-|μ1-μ2|/b1) + |μ1-μ2|)/b2 - 1`
    - `kl_exponential_exponential`: `ln(λ1/λ2) + λ2/λ1 - 1`
    - `kl_gamma_gamma`: `(α1-α2)ψ(α1) - lnΓ(α1) + lnΓ(α2) + α2(ln β1 - ln β2) + α1(β2-β1)/β1`
    - `kl_poisson_poisson`: `λ1 * (ln λ1 - ln λ2) - λ1 + λ2`

  Each formula matches PyTorch's registered `_kl_*_*` body in
  `torch/distributions/kl.py`. The `kl_gamma_scalar` helper is
  factored out so Gamma-Exponential and Exponential-Gamma can
  reuse it.

- REQ-5: Cross-family closed-form formulas (4 pairs).
  `kl_normal_uniform`, `kl_uniform_normal`, `kl_gamma_exponential`,
  `kl_exponential_gamma`. The last two use the `Exp(λ) = Gamma(1,
  λ)` identity to reduce to `kl_gamma_scalar(1, λ, α, β)`.

- REQ-6: Every formula invokes
  `crate::fallback::check_gpu_fallback_opt_in(&[param_tensors...],
  "kl_divergence(P, Q)")?` as its first line, satisfying the
  crate-wide CPU-fallback policy. CUDA inputs without the env var
  return `NotImplementedOnCuda`.

- REQ-7: PARTIAL — PyTorch ships ~88 KL pairs. ferrotorch now ships
  84 (was 41). The #1374 ContinuousBernoulli sub-part added 13 pairs
  that needed the new `ContinuousBernoulli` struct
  (`continuous_bernoulli.rs`): 6 finite — `kl_continuous_bernoulli_continuous_bernoulli`
  (`kl.py:255-260`), `kl_beta_continuous_bernoulli` (`kl.py:518-525`),
  `kl_continuous_bernoulli_exponential` (`kl.py:586-588`),
  `kl_continuous_bernoulli_normal` (`kl.py:595-604`),
  `kl_continuous_bernoulli_uniform` (`kl.py:607-617`, where-mask `+inf`
  when the Uniform support contains `[0,1]`),
  `kl_uniform_continuous_bernoulli` (`kl.py:871-886`, where-mask) — plus
  7 support-mismatch `+inf` via `kl_infinite_like`: ContinuousBernoulli-Pareto
  (`kl.py:581`), {Exponential,Gamma,Gumbel,Laplace,Normal,Pareto}-ContinuousBernoulli
  (`kl.py:621,666,719,741,762,796`). The CB closed forms reuse the
  crate-visible `_lims=(0.499,0.501)` Taylor-cutoff scalar helpers from
  `continuous_bernoulli.rs`. The #1374 Binomial sub-part added 2 pairs that needed
  the new `Binomial` distribution struct (`binomial.rs`):
    - `kl_binomial_binomial` (`kl.py:231-244`, finite closed form
      `n·(p·(logit_p−logit_q) + ln(1−p) − ln(1−q))`; `+inf`
      element-wise where `n_p > q_n`; `InvalidArgument` where any
      `n_p < n_q`, matching PyTorch's `NotImplementedError`).
    - Poisson-Binomial routed through `fn kl_infinite_like`
      (mirrors `_kl_poisson_infinity`, `kl.py:842`).
  The #1374 Geometric sub-part added 1 pair that needed the new
  `Geometric` distribution struct (`geometric.rs`):
    - `kl_geometric_geometric` (`kl.py:320-322`, finite closed form
      `-p.entropy() - log1p(-q.probs)/p.probs - q.logits`).
  The #1562 both-types-exist gap closure added 27 pairs
  (both operand types already existed in ferrotorch; no new type was
  needed):
    - 3 finite closed forms: `kl_onehotcategorical_onehotcategorical`
      (mirrors `kl.py:474-476`, delegates to the Categorical-Categorical
      closed form), `kl_bernoulli_poisson` (`kl.py:513-516`,
      `-H(Bernoulli) - (p·ln(rate) - rate)`), `kl_normal_laplace`
      (`kl.py:782-792`, uses `erf` via
      `ferrotorch_core::special::erf`).
    - 24 support-mismatch `+inf` arms, all routed through
      `fn kl_infinite_like` (mirrors PyTorch's `_infinite_like`,
      `kl.py:141-145`): Beta-Pareto (`kl.py:528`);
      Exponential-{Beta,Pareto,Uniform} (`kl.py:620-623`);
      Gamma-{Beta,Pareto,Uniform} (`kl.py:665-668`);
      Gumbel-{Beta,Exponential,Gamma,Pareto,Uniform} (`kl.py:718-723`);
      Laplace-{Beta,Exponential,Gamma,Pareto,Uniform} (`kl.py:740-745`);
      Normal-{Beta,Exponential,Gamma,Pareto} (`kl.py:761-765`);
      Pareto-{Beta,Uniform} (`kl.py:795-797`); Poisson-Bernoulli
      (`kl.py:841`).
  Still NOT-STARTED (each a concrete prereq, not a deferral):
    - Independent-Independent (`kl.py:944`) — `Independent<T, D>` is
      generic over the concrete base `D`, so the `Any::downcast_ref`
      dispatch cannot match it without a KL-recursion trait hook on the
      `Distribution` trait (in `lib.rs`) + an override in
      `independent.rs`. Both files are outside the kl.rs+kl.md manifest;
      this needs a follow-up acto-builder dispatch with an expanded
      manifest.
    - TransformedDistribution-TransformedDistribution (`kl.py:496`) +
      ExponentialFamily-ExponentialFamily (`kl.py:282`) — need the
      base-recursion / Bregman-divergence trait surface.
  Closing the remaining ~4 is tracked by blocker #1374 (stays open).
  The Independent / TransformedDistribution / ExponentialFamily
  KL-recursion (and Bregman) trait hooks must each be filed as their own
  blockers before the dependent KL pairs can ship; the missing
  distribution-type prerequisites (Geometric, ContinuousBernoulli) have
  now shipped.

- REQ-8: SHIPPED (design decision, #1375) — the explicit
  `Any::downcast_ref` match in `kl_dispatch` IS the chosen
  Rust-idiomatic equivalent of PyTorch's `@register_kl` decorator +
  `_dispatch_kl` most-specific-subclass lookup
  (`torch/distributions/kl.py:51-138`). PyTorch's decorator is a
  Python-runtime open-extension mechanism; Rust's static analog is
  the explicit match in a closed crate. The maintainability the
  registry would buy is already delivered by the
  `kl_doc_table_matches_dispatcher` drift test, which pins the doc
  table, the `KL_SUPPORTED_PAIR_COUNT` constant, and the dispatcher
  arms in lockstep. A `Lazy<HashMap<(TypeId, TypeId), Fn>>` registry
  would add runtime indirection and a global without enabling
  cross-crate extension (each formula needs the concrete
  distribution's typed accessors, which are crate-private). Closes
  #1375.

## Acceptance Criteria

- [x] AC-1: `pub fn kl_divergence<T, P, Q>` with the `Distribution +
  'static` bounds + `Any`-downcast dispatch core.
- [x] AC-2: `pub const fn kl_supported_pair_count() -> usize`
  returning 84 + the drift test
  `kl_doc_table_matches_dispatcher` is green.
- [x] AC-3: All 8 same-family formulas (`kl_normal_normal`,
  `kl_bernoulli_bernoulli`, `kl_uniform_uniform`,
  `kl_categorical_categorical`, `kl_laplace_laplace`,
  `kl_exponential_exponential`, `kl_gamma_gamma`,
  `kl_poisson_poisson`) ship as `fn` items in `kl.rs`.
- [x] AC-4: All 4 cross-family formulas (`kl_normal_uniform`,
  `kl_uniform_normal`, `kl_gamma_exponential`,
  `kl_exponential_gamma`) ship as `fn` items in `kl.rs`.
- [x] AC-5: Every formula's first statement is the fallback guard.
- [x] AC-6: The doc-table on `kl_divergence` lists exactly 84 pairs,
  matching `KL_SUPPORTED_PAIR_COUNT`.
- [~] AC-7: PyTorch's full ~88-pair coverage — 84/~88 shipped
  (the #1562 closure added 27 pairs; the #1374 Binomial sub-part added
  the Binomial-Binomial + Poisson-Binomial pair; the #1374 Geometric
  sub-part added the Geometric-Geometric pair; the #1374
  ContinuousBernoulli sub-part added the 13 CB pairs); blocker #1374
  stays open for the remaining ~4 (Independent-Independent /
  TransformedDistribution / ExponentialFamily pairs blocked on a
  KL-recursion / Bregman trait hook outside the kl.rs manifest).
- [x] AC-8: `register_kl` extension API — closed as a design
  decision (#1375): the explicit `Any::downcast_ref` match is the
  chosen Rust-idiomatic dispatch; the drift test enforces
  maintainability. Closes #1375.

## Architecture

### Public entry point (REQ-1)

```rust
pub fn kl_divergence<T: Float, P, Q>(p: &P, q: &Q) -> FerrotorchResult<Tensor<T>>
where
    P: Distribution<T> + 'static,
    Q: Distribution<T> + 'static,
{
    kl_dispatch::<T>(p, q)
}
```

The `'static` bound is required by `Any::downcast_ref`. PyTorch's
`kl_divergence(p, q)` (`kl.py:165-193`) accepts any
`Distribution`-typed args and dispatches by `type(p)`; the Rust
analog uses the type system's `TypeId` via `Any`. The
`Distribution<T>` super-bound is structural (every concrete
distribution implements it) and ensures the user can't pass random
objects in.

### Pair-count introspection (REQ-2)

`pub const fn kl_supported_pair_count() -> usize` is a `const fn`
that returns the compile-time constant
`KL_SUPPORTED_PAIR_COUNT: usize = 84`. The test
`kl_doc_table_matches_dispatcher` parses this very source file at
runtime via `include_str!("kl.rs")` and asserts:

1. `KL_SUPPORTED_PAIR_COUNT == kl_supported_pair_count()` — the
   public accessor mirrors the internal constant.
2. The supported-pairs doc table on `kl_divergence` has exactly 84
   rows.
3. The dispatcher in `kl_dispatch` has exactly 84
   `p.downcast_ref::<...>()` arms.

The triple check catches the drift scenario in historical issue
#1124 where the doc table had 6 pairs while the dispatcher had
grown to 12. Updating any one of the three sources without the
others trips the test.

### Dispatcher core (REQ-3)

`fn kl_dispatch<T: Float>(p: &dyn Any, q: &dyn Any) ->
FerrotorchResult<Tensor<T>>` is an 84-arm `if let` chain. Each arm:

```rust
if let (Some(pn), Some(qn)) = (p.downcast_ref::<Normal<T>>(), q.downcast_ref::<Normal<T>>()) {
    return kl_normal_normal(pn, qn);
}
```

Fall-through is `Err(FerrotorchError::InvalidArgument { message: "..." })`
with the full pair list in the message. This is the Rust analog
of PyTorch's `NotImplementedError` from `_dispatch_kl` when no
match exists (`kl.py:189-192`).

The order of arms matters when two arms could match (e.g. if a
distribution were generic over multiple types); we order
same-family pairs first, then cross-family. This matches PyTorch's
`_Match` lexicographic-order disambiguation
(`kl.py:94-110`).

### Same-family formulas (REQ-4)

Each follows the same pattern:

1. Call `check_gpu_fallback_opt_in` with all parameter tensors and
   the op name.
2. Read parameter data into `Vec<T>` via `data_vec()?`.
3. Build the p-vs-q broadcast plan with `kl_broadcast_index_pairs`
   (#1573): given `p`'s and `q`'s batch shapes it returns the broadcast
   output shape plus, for every output element, the source flat indices
   `(pi, qi)` into `p`'s and `q`'s parameter vectors. This mirrors torch's
   `broadcast_all` (`torch/distributions/utils.py:27`), which broadcasts
   every parameter tensor of `p` and `q` jointly before evaluating the
   closed form. Because each distribution enforces that all of its own
   parameter tensors share one shape at construction (e.g. `Normal::new`
   requires `loc.shape() == scale.shape()`), broadcasting `p` against `q`
   reduces to broadcasting one representative `p` shape against one
   representative `q` shape; every `p` parameter is indexed by `pi`, every
   `q` parameter by `qi`. The formula is then evaluated element-wise over
   the index pairs.
4. Wrap the result in a tensor of the **broadcast output shape** (`plan.out_shape`)
   via `Tensor::from_storage(TensorStorage::cpu(...), plan.out_shape, false)`.
   This is NOT `p.<param>().shape()` — `KL(scalar_p, batched_q)` and disjoint
   batch dims (`p:[2,1]` vs `q:[1,3]` -> `[2,3]`) match upstream instead of
   silently truncating to `p`'s shape (the #1569/#1572 bug class, closed for
   the whole pair family by #1573).

The `kl_gamma_gamma` body uses `digamma_scalar` and `lgamma_scalar`
from `special_fns.rs` (REQ-1 of `special_fns.md`).

The shared broadcast machinery lives in `fn kl_broadcast_index_pairs`
(returning `struct KlBroadcastPlan`), built on the `fn kl_row_major_strides`
+ `fn kl_broadcast_flat_index` helpers and `ferrotorch_core::shape::broadcast_shapes`
in `kl.rs`. The categorical / one-hot-categorical (event-dim reduction to a
scalar), Dirichlet (bespoke per-row batch handling), and (Low-Rank)MVN
(full-vector, scalar result) pairs do NOT use this plan — they reduce over the
event dimension and were left on their existing shape logic. The `+inf`
support-mismatch arms keep `kl_infinite_like(p.<param>)` (matching torch's
`_infinite_like(tensor)`, which shapes off the single passed tensor).

### Cross-family formulas (REQ-5)

`kl_normal_uniform` and `kl_uniform_normal` are explicit closed-form
expressions involving `2π`, `2πe`, and the second moment of the
uniform. See the bodies for the algebraic derivations.

`kl_gamma_exponential` and `kl_exponential_gamma` exploit the
identity `Exp(λ) ≡ Gamma(1, λ)`, evaluated over the p-vs-q broadcast plan
(#1573):

```rust
fn kl_gamma_exponential<T: Float>(p: &Gamma<T>, q: &Exponential<T>) -> ... {
    let p_conc = p.concentration().data_vec()?;
    let p_rate = p.rate().data_vec()?;
    let q_rate = q.rate().data_vec()?;
    let one = T::from(1.0).unwrap();
    let plan = kl_broadcast_index_pairs(p.concentration().shape(), q.rate().shape())?;
    let result: Vec<T> = plan.p_idx.iter().zip(plan.q_idx.iter())
        .map(|(&pi, &qi)| kl_gamma_scalar(p_conc[pi], p_rate[pi], one, q_rate[qi]))
        .collect();
    Tensor::from_storage(TensorStorage::cpu(result), plan.out_shape, false)
}
```

Verified by `test_kl_gamma_exponential_matches_gamma_gamma` and
`test_kl_exponential_gamma_self_consistency`.

### Non-test production consumers

- `pub fn kl_divergence` is exported through `kl.rs` and accessible
  via `ferrotorch_distributions::kl::kl_divergence` (the `pub mod
  kl;` declaration in `lib.rs:63`). The module is public, not
  `pub(crate)`. This is grandfathered public API per goal.md S5.
- `pub const fn kl_supported_pair_count` is exported for
  third-party probes that want to introspect the registry without
  parsing source. The const-fn surface is the consumer.
- `fn kl_gamma_scalar` is reused internally by `kl_gamma_gamma`,
  `kl_gamma_exponential`, `kl_exponential_gamma` — three
  production call sites in the same module.

PyTorch users typically write `from torch.distributions import
kl_divergence; kl_divergence(p, q)`. The ferrotorch analog is
`ferrotorch_distributions::kl::kl_divergence(&p, &q)`. The trait
signature is the contract; the absence of an internal
`ferrotorch-*` callsite is grandfathered (R-DEFER-1's existing-API
exemption).

## Parity contract

`parity_ops = []`. KL divergences are derived from the underlying
distribution parameters; the parity contract is on those
parameter ops, not on the KL formula. Edge cases preserved:

- **`KL(p || p) ~= 0`** for every same-family pair. Tests
  `test_kl_normal_normal_same`, `test_kl_bernoulli_same`, etc.
  pin the contract numerically. Tolerance is `1e-5` for f32, `1e-12`
  for f64.
- **`KL >= 0`** always. Tested by `*_nonnegative` cases.
- **`KL(p || q) != KL(q || p)`** (asymmetry). Tested by
  `test_kl_normal_normal_asymmetric`.
- **Uniform containment**: `KL(U(a1,b1) || U(a2,b2))` is `+inf`
  when `[a1,b1] ⊄ [a2,b2]`. The condition is `ql > pl || qh < ph`.
  Tested by `test_kl_uniform_not_contained`.
- **Categorical degenerate**: when `p_k > 0` but `q_k <= eps`, the
  KL is `+inf` (the support of q doesn't cover p). The `eps =
  1e-7` floor on probabilities catches the same case PyTorch's
  `torch.special.xlogy` handles.
- **Bernoulli degenerate**: probabilities are clamped to `[eps,
  1-eps]` to avoid `log(0)`. Matches PyTorch's KL formula which
  uses the same clamp internally.
- **Cross-family identity**: `KL(Exp(λ) || Gamma(1, λ)) == 0` and
  `KL(Gamma(1, λ) || Exp(λ)) == 0`. Tested by
  `test_kl_exponential_gamma_self_consistency`.

## Verification

Tests in `mod tests in kl.rs` (114 tests):

- Same-family: `test_kl_normal_normal_{same,different_mean,different_scale,nonnegative,asymmetric}`, `test_kl_bernoulli_{same,different,nonnegative}`, `test_kl_uniform_{same,contained,not_contained}`, `test_kl_categorical_{same,different,nonnegative}`, `test_kl_laplace_laplace_{same_is_zero,different_scale,different_loc}`, `test_kl_exponential_exponential_{same,different}`, `test_kl_gamma_gamma_{same_is_zero,exp_special_case}`, `test_kl_poisson_poisson_{same,known_value}`.
- Cross-family: `test_kl_normal_uniform`, `test_kl_uniform_normal`, `test_kl_gamma_exponential_matches_gamma_gamma`, `test_kl_exponential_gamma_{matches_gamma_gamma,self_consistency}`.
- f64: `test_kl_normal_normal_f64`, `test_kl_bernoulli_f64`, `test_kl_uniform_f64`.
- Drift guard: `kl_doc_table_matches_dispatcher` (the critical
  test — fails on any mismatch between doc-table rows, dispatcher
  arms, and the const).
- Error case: `test_kl_unsupported_pair` (Normal vs Bernoulli
  returns Err).
- Sanity: `test_ln_gamma_known_values` (smokes the special_fns
  invariant `lgamma(n+1) = ln(n!)`).

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib kl:: 2>&1 | tail -3
```

Expected: `114 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn kl_divergence<T: Float, P, Q>` with `P: Distribution<T> + 'static`, `Q: Distribution<T> + 'static` bounds in `kl.rs`, mirroring `torch/distributions/kl.py:kl_divergence`; non-test consumer: `pub mod kl` in `lib.rs:63` and `pub fn kl_divergence` exposes it as grandfathered public API; tests `test_kl_*` (~25 sites) exercise it through the trait dispatch path. |
| REQ-2 | SHIPPED | impl: `pub const fn kl_supported_pair_count() -> usize` and `KL_SUPPORTED_PAIR_COUNT: usize = 84` constant in `kl.rs`; non-test consumer: the const fn is grandfathered public API (introspection extension point); the drift-prevention test `kl_doc_table_matches_dispatcher` reads `include_str!("kl.rs")` and asserts the three-way invariant against the public accessor — that test is the production check. `audit_1374_supported_pair_count_is_41` in `divergence_wave_m_audit.rs` pins the accessor (now 84). |
| REQ-3 | SHIPPED | impl: `fn kl_dispatch<T: Float>(p: &dyn Any, q: &dyn Any)` 84-arm `Any::downcast_ref` chain in `kl.rs`, mirroring PyTorch's `_dispatch_kl` in `torch/distributions/kl.py:113-138`; non-test consumer: `pub fn kl_divergence` invokes `kl_dispatch::<T>(p, q)` on every call — that's the in-crate production consumer. |
| REQ-4 | SHIPPED | impl: 8 same-family `fn kl_<p>_<q><T: Float>` formulas in `kl.rs` (`kl_normal_normal`, `kl_bernoulli_bernoulli`, `kl_uniform_uniform`, `kl_categorical_categorical`, `kl_laplace_laplace`, `kl_exponential_exponential`, `kl_gamma_gamma`, `kl_poisson_poisson`), mirroring the `@register_kl` bodies in `torch/distributions/kl.py`; non-test consumer: `fn kl_dispatch in kl.rs` invokes each formula in its respective arm — 8 production call sites. |
| REQ-5 | SHIPPED | impl: 4 cross-family formulas (`kl_normal_uniform`, `kl_uniform_normal`, `kl_gamma_exponential`, `kl_exponential_gamma`) in `kl.rs`; the last two use `fn kl_gamma_scalar` via `Exp(λ) ≡ Gamma(1, λ)`; non-test consumer: `fn kl_dispatch in kl.rs` calls each — 4 production call sites; `fn kl_gamma_scalar` is consumed by 3 production sites internally (Gamma-Gamma, Gamma-Exp, Exp-Gamma). |
| REQ-6 | SHIPPED | impl: every formula's first statement is `crate::fallback::check_gpu_fallback_opt_in(&[...], "kl_divergence(P, Q)")?` in `kl.rs` — 38 production call sites of the fallback gate (one per scalar formula fn; the four MVN/LowRankMVN pairs share the gate inside `fn kl_mvn_pair`); non-test consumer: this IS the production consumer of `fn check_gpu_fallback_opt_in in fallback.rs` (per `fallback.md` REQ-2). The `op` argument names every call site. |
| REQ-7 | PARTIAL | blocker #1374 — PyTorch ships ~88 (P, Q) KL pairs; ferrotorch now has 84 (was 41). The #1374 ContinuousBernoulli sub-part added 13 pairs that needed the new `ContinuousBernoulli` struct (`continuous_bernoulli.rs`): 6 finite — `fn kl_continuous_bernoulli_continuous_bernoulli` (`kl.py:255-260`), `fn kl_beta_continuous_bernoulli` (`kl.py:518-525`), `fn kl_continuous_bernoulli_exponential` (`kl.py:586-588`), `fn kl_continuous_bernoulli_normal` (`kl.py:595-604`), `fn kl_continuous_bernoulli_uniform` (`kl.py:607-617`, where-mask `+inf` on `q.low>=0 \| q.high<=1`), `fn kl_uniform_continuous_bernoulli` (`kl.py:871-886`, where-mask `+inf` on `p.high>=1 \| p.low<=0`) — plus 7 `+inf` via `fn kl_infinite_like`: ContinuousBernoulli-Pareto (`kl.py:581`), {Exponential,Gamma,Gumbel,Laplace,Normal,Pareto}-ContinuousBernoulli (`kl.py:621,666,719,741,762,796`). The CB closed forms reuse the crate-visible `_lims`-cutoff scalar helpers (`cont_bern_log_norm_scalar`/`mean_scalar`/`entropy_scalar`/`variance_scalar`/`logits_scalar`) from `continuous_bernoulli.rs`. Non-test consumer: each invoked by its `fn kl_dispatch` downcast arm in `kl.rs`, reached via `pub fn kl_divergence`. Pinned by `test_kl_cb_cb_{same_is_zero,known_value,near_half,batched,scalar_p_batched_q_broadcast}`, `test_kl_beta_cb_known_value`, `test_kl_cb_exponential_known_value`, `test_kl_cb_normal_known_value`, `test_kl_cb_uniform_{contains_support_is_inf,wider_is_finite}`, `test_kl_uniform_cb_{inner_is_finite,touching_support_is_inf}`, `test_kl_cb_support_mismatch_infinity_family` in `kl.rs` `mod tests` (live-torch 2.11 f64, R-CHAR-3 non-tautological). The #1374 Binomial sub-part added 2 pairs that needed the new `Binomial` struct (`binomial.rs`): finite `fn kl_binomial_binomial` (mirrors `kl.py:231-244`: `n·(p·(logit_p−logit_q)+ln(1−p)−ln(1−q))`, `+inf` where `n_p>n_q`, `InvalidArgument` where `n_p<n_q`) + Poisson-Binomial routed through `fn kl_infinite_like` (mirrors `_kl_poisson_infinity` `kl.py:842`). The #1374 Geometric sub-part added 1 pair that needed the new `Geometric` struct (`geometric.rs`): finite `fn kl_geometric_geometric` (mirrors `kl.py:320-322`: `-p.entropy() - log1p(-q.probs)/p.probs - q.logits`); non-test consumer: its `fn kl_dispatch` downcast arm reads `p.probs()`/`q.probs()` off the `Geometric` accessors, reached via `pub fn kl_divergence`; pinned by `test_kl_geometric_geometric_{same_is_zero,known_value,batched,nonnegative}` in `kl.rs` `mod tests` (live-torch 2.11 f64). Non-test consumer (Binomial): each is invoked by its `fn kl_dispatch` downcast arm in `kl.rs` (the in-crate production caller, reached via `pub fn kl_divergence`); `fn kl_binomial_binomial` reads `p.total_count().data_vec()?` / `p.probs().data_vec()?` off the `Binomial` accessors. Pinned by `test_kl_binomial_binomial_{same_is_zero,known_value,known_value_2,larger_np_is_inf,smaller_np_errors}` + `test_kl_poisson_binomial_is_inf` in `kl.rs` `mod tests` (live-torch 2.11 f64 reference values, R-CHAR-3 non-tautological). The #1562 both-types-exist closure earlier added 27 pairs. Finite impl: `fn kl_onehotcategorical_onehotcategorical` (mirrors `kl.py:474-476`), `fn kl_bernoulli_poisson` (`kl.py:513-516`), `fn kl_normal_laplace` (`kl.py:782-792`, `erf` via `ferrotorch_core::special::erf`) in `kl.rs`. `+inf` impl: 24 support-mismatch arms routed through `fn kl_infinite_like` (mirrors `_infinite_like` at `kl.py:141-145`): Beta-Pareto `kl.py:528`; Exponential-{Beta,Pareto,Uniform} `kl.py:620-623`; Gamma-{Beta,Pareto,Uniform} `kl.py:665-668`; Gumbel-{Beta,Exponential,Gamma,Pareto,Uniform} `kl.py:718-723`; Laplace-{Beta,Exponential,Gamma,Pareto,Uniform} `kl.py:740-745`; Normal-{Beta,Exponential,Gamma,Pareto} `kl.py:761-765`; Pareto-{Beta,Uniform} `kl.py:795-797`; Poisson-Bernoulli `kl.py:841`. Pinned by `divergence_kl_onehotcategorical_onehotcategorical_missing`, `divergence_kl_bernoulli_poisson_missing`, `divergence_kl_normal_laplace_missing` + the extended `divergence_kl_support_mismatch_family_is_positive_infinity` / second-point tests in `divergence_kl_1374_both_types_exist_gaps.rs`. Plus the prior wave-N MVN/LowRankMVN pairs (`fn kl_multivariatenormal_multivariatenormal` etc.). Still NOT-STARTED: `Independent-Independent` (`kl.py:944`, generic-base barrier needs a `Distribution`-trait KL hook in `lib.rs` + `independent.rs`, outside this manifest); TransformedDistribution-TransformedDistribution + ExponentialFamily-ExponentialFamily (base-recursion / Bregman trait surface). #1374 stays open for the remaining ~4. |
| REQ-8 | SHIPPED (design decision) | #1375 — the explicit `Any::downcast_ref` match in `fn kl_dispatch in kl.rs` is the deliberate Rust-idiomatic equivalent of PyTorch's `@register_kl` + `_dispatch_kl` (`torch/distributions/kl.py:51-138`), a Python-runtime open-extension pattern. The maintainability a registry would buy is delivered by the `kl_doc_table_matches_dispatcher` drift test that pins the doc table, the `KL_SUPPORTED_PAIR_COUNT` constant, and the dispatcher arms in lockstep; non-test consumer: `pub fn kl_divergence` invokes the match on every call. A `Lazy<HashMap<(TypeId,TypeId),Fn>>` registry would add indirection without enabling cross-crate extension (formulas need crate-private typed accessors). `audit_1375_supported_pair_count` in `divergence_wave_l_audit.rs` pins the count (currently 71). Closes #1375. |
