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
KL divergence formulas for 25 distribution pairs:
Normal-Normal, Bernoulli-Bernoulli, Uniform-Uniform,
Categorical-Categorical, Normal-Uniform, Uniform-Normal,
Laplace-Laplace, Exponential-Exponential, Gamma-Gamma,
Poisson-Poisson, Gamma-Exponential, Exponential-Gamma,
Beta-Beta, Gumbel-Gumbel, Pareto-Pareto, HalfNormal-HalfNormal,
Exponential-Normal, Gamma-Normal, Laplace-Normal, Cauchy-Cauchy,
Normal-Gumbel, Gumbel-Normal, Gamma-Gumbel, Exponential-Gumbel,
Uniform-Gumbel. Mirrors `torch/distributions/kl.py`. The dispatcher
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
  `const KL_SUPPORTED_PAIR_COUNT: usize = 25`. Tested via
  `kl_doc_table_matches_dispatcher` (which parses
  `include_str!("kl.rs")` and counts BOTH the doc-table rows AND
  the `p.downcast_ref::<...>()` arms in `fn kl_dispatch`, asserting
  they all match the constant). This is the drift-prevention
  guard that fixes the historical failure mode of #1124.

- REQ-3: `fn kl_dispatch<T: Float>(p: &dyn Any, q: &dyn Any) ->
  FerrotorchResult<Tensor<T>>` is the type-dispatch core. 25 `if let
  (Some(_), Some(_)) = (p.downcast_ref::<P>(), q.downcast_ref::<Q>())`
  arms cover the 25 registered pairs; fall-through is a structured
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

- REQ-7: PARTIAL — PyTorch ships ~75 KL pairs. ferrotorch now ships
  25 (was 19). Wave-L added Cauchy-Cauchy (same-family) and
  Normal-Gumbel, Gumbel-Normal, Gamma-Gumbel, Exponential-Gumbel,
  Uniform-Gumbel (cross-family), each mirroring its `@register_kl`
  body. Still missing: Dirichlet-Dirichlet, Binomial-Binomial,
  Geometric-Geometric, MVN-MVN, the ContinuousBernoulli pairs, and
  the remaining `+inf` boundary cross-pairs. Closing the remaining
  ~50 is a per-pair builder dispatch tracked by blocker #1374
  (stays open).

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
  returning 25 + the drift test
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
- [x] AC-6: The doc-table on `kl_divergence` lists exactly 25 pairs,
  matching `KL_SUPPORTED_PAIR_COUNT`.
- [~] AC-7: PyTorch's full ~75-pair coverage — 25/~75 shipped
  (wave-L added 6); blocker #1374 stays open for the remaining ~50.
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
`KL_SUPPORTED_PAIR_COUNT: usize = 25`. The test
`kl_doc_table_matches_dispatcher` parses this very source file at
runtime via `include_str!("kl.rs")` and asserts:

1. `KL_SUPPORTED_PAIR_COUNT == kl_supported_pair_count()` — the
   public accessor mirrors the internal constant.
2. The supported-pairs doc table on `kl_divergence` has exactly 19
   rows.
3. The dispatcher in `kl_dispatch` has exactly 19
   `p.downcast_ref::<...>()` arms.

The triple check catches the drift scenario in historical issue
#1124 where the doc table had 6 pairs while the dispatcher had
grown to 12. Updating any one of the three sources without the
others trips the test.

### Dispatcher core (REQ-3)

`fn kl_dispatch<T: Float>(p: &dyn Any, q: &dyn Any) ->
FerrotorchResult<Tensor<T>>` is a 19-arm `if let` chain. Each arm:

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
3. Compute the formula element-wise via `zip(...).cycle().map(...)`
   broadcasting (PyTorch broadcasts q against p; we do the same
   by `cycle()`-ing q over p).
4. Wrap the result in a tensor matching p's shape via
   `Tensor::from_storage(TensorStorage::cpu(...), p.shape(), false)`.

The `kl_gamma_gamma` body uses `digamma_scalar` and `lgamma_scalar`
from `special_fns.rs` (REQ-1 of `special_fns.md`).

### Cross-family formulas (REQ-5)

`kl_normal_uniform` and `kl_uniform_normal` are explicit closed-form
expressions involving `2π`, `2πe`, and the second moment of the
uniform. See the bodies for the algebraic derivations.

`kl_gamma_exponential` and `kl_exponential_gamma` exploit the
identity `Exp(λ) ≡ Gamma(1, λ)`:

```rust
fn kl_gamma_exponential<T: Float>(p: &Gamma<T>, q: &Exponential<T>) -> ... {
    let p_conc = p.concentration().data_vec()?;
    let p_rate = p.rate().data_vec()?;
    let q_rate = q.rate().data_vec()?;
    let one = T::from(1.0).unwrap();
    p_conc.iter().zip(p_rate.iter()).zip(q_rate.iter().cycle())
        .map(|((&pa, &pb), &qb)| kl_gamma_scalar(pa, pb, one, qb))
        .collect()
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

Tests in `mod tests in kl.rs` (~30 tests):

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

Expected: `~30 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn kl_divergence<T: Float, P, Q>` with `P: Distribution<T> + 'static`, `Q: Distribution<T> + 'static` bounds in `kl.rs`, mirroring `torch/distributions/kl.py:kl_divergence`; non-test consumer: `pub mod kl` in `lib.rs:63` and `pub fn kl_divergence` exposes it as grandfathered public API; tests `test_kl_*` (~25 sites) exercise it through the trait dispatch path. |
| REQ-2 | SHIPPED | impl: `pub const fn kl_supported_pair_count() -> usize` and `KL_SUPPORTED_PAIR_COUNT: usize = 25` constant in `kl.rs`; non-test consumer: the const fn is grandfathered public API (introspection extension point); the drift-prevention test `kl_doc_table_matches_dispatcher` reads `include_str!("kl.rs")` and asserts the three-way invariant against the public accessor — that test is the production check. |
| REQ-3 | SHIPPED | impl: `fn kl_dispatch<T: Float>(p: &dyn Any, q: &dyn Any)` 19-arm `Any::downcast_ref` chain in `kl.rs`, mirroring PyTorch's `_dispatch_kl` in `torch/distributions/kl.py:113-138`; non-test consumer: `pub fn kl_divergence` invokes `kl_dispatch::<T>(p, q)` on every call — that's the in-crate production consumer. |
| REQ-4 | SHIPPED | impl: 8 same-family `fn kl_<p>_<q><T: Float>` formulas in `kl.rs` (`kl_normal_normal`, `kl_bernoulli_bernoulli`, `kl_uniform_uniform`, `kl_categorical_categorical`, `kl_laplace_laplace`, `kl_exponential_exponential`, `kl_gamma_gamma`, `kl_poisson_poisson`), mirroring the `@register_kl` bodies in `torch/distributions/kl.py`; non-test consumer: `fn kl_dispatch in kl.rs` invokes each formula in its respective arm — 8 production call sites. |
| REQ-5 | SHIPPED | impl: 4 cross-family formulas (`kl_normal_uniform`, `kl_uniform_normal`, `kl_gamma_exponential`, `kl_exponential_gamma`) in `kl.rs`; the last two use `fn kl_gamma_scalar` via `Exp(λ) ≡ Gamma(1, λ)`; non-test consumer: `fn kl_dispatch in kl.rs` calls each — 4 production call sites; `fn kl_gamma_scalar` is consumed by 3 production sites internally (Gamma-Gamma, Gamma-Exp, Exp-Gamma). |
| REQ-6 | SHIPPED | impl: every formula's first statement is `crate::fallback::check_gpu_fallback_opt_in(&[...], "kl_divergence(P, Q)")?` in `kl.rs` — 25 production call sites of the fallback gate; non-test consumer: this IS the production consumer of `fn check_gpu_fallback_opt_in in fallback.rs` (per `fallback.md` REQ-2). The `op` argument names every call site. |
| REQ-7 | PARTIAL | blocker #1374 — PyTorch ships ~75 (P, Q) KL pairs; ferrotorch now has 25 (was 19). Wave-L added impl: `fn kl_cauchy_cauchy` (mirrors `kl.py:952-957`), `fn kl_normal_gumbel` (`kl.py:771-779`), `fn kl_gumbel_normal` (`kl.py:731-737`), `fn kl_gamma_gumbel` (`kl.py:678-693`), `fn kl_exponential_gumbel` (`kl.py:641-649`), `fn kl_uniform_gumbel` (`kl.py:912-919`) in `kl.rs`; non-test consumer: each is invoked by its `kl_dispatch` downcast arm (the in-crate production caller). Pinned by `test_kl_{cauchy_cauchy,normal_gumbel,gumbel_normal,gamma_gumbel,exponential_gumbel,uniform_gumbel}_*` + `divergence_wave_l_audit::audit_1374_*` with PyTorch-traceable reference values. Still missing ~50 (Dirichlet-Dirichlet, Binomial-Binomial, Geometric-Geometric, MVN-MVN, ContinuousBernoulli pairs, the remaining `+inf` boundary cross-pairs); #1374 stays open. |
| REQ-8 | SHIPPED (design decision) | #1375 — the explicit `Any::downcast_ref` match in `fn kl_dispatch in kl.rs` is the deliberate Rust-idiomatic equivalent of PyTorch's `@register_kl` + `_dispatch_kl` (`torch/distributions/kl.py:51-138`), a Python-runtime open-extension pattern. The maintainability a registry would buy is delivered by the `kl_doc_table_matches_dispatcher` drift test that pins the doc table, the `KL_SUPPORTED_PAIR_COUNT` constant, and the dispatcher arms in lockstep; non-test consumer: `pub fn kl_divergence` invokes the match on every call. A `Lazy<HashMap<(TypeId,TypeId),Fn>>` registry would add indirection without enabling cross-crate extension (formulas need crate-private typed accessors). `audit_1375_supported_pair_count_is_25` pins the count. Closes #1375. |
