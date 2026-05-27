# ferrotorch-distributions — `constraints` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/constraints.py
-->

## Summary

`ferrotorch-distributions/src/constraints.rs` defines the `Constraint`
trait and 11 concrete constraint types — `Real`, `Positive`,
`NonNegative`, `UnitInterval`, `BooleanConstraint`, `GreaterThan`,
`GreaterThanEq`, `OpenInterval`, `ClosedInterval`,
`HalfOpenInterval`, `LessThan`, `Simplex` — plus convenience
free-function constructors (`real()`, `positive()`, etc.). Mirrors
`torch/distributions/constraints.py` (which ships 28 constraint
variants in its `__all__`).

## Requirements

- REQ-1: `pub trait Constraint: Send + Sync` with one required
  method — `check<T: Float>(&self, value: T) -> bool` — plus four
  defaults: `check_tensor<T: Float>(&self, value: &Tensor<T>) ->
  FerrotorchResult<bool>` (default AND-reduces the scalar `check`
  element-wise), `is_discrete() -> bool { false }`, `event_dim() ->
  usize { 0 }`, and a required `name(&self) -> &'static str` for
  diagnostics. Mirrors `torch/distributions/constraints.py:80-106`
  `class Constraint` with its `is_discrete` and `event_dim`
  class-level attributes and its `check(value)` method. PyTorch's
  `check` takes a tensor and returns a boolean tensor; ferrotorch
  splits this into a per-element scalar `check<T>` (the R-DEV-1
  dtype-monomorphised analog) plus the tensor-level `check_tensor`
  for constraints whose validity is a vector property (`Simplex`).

- REQ-2: `pub struct Real` with `impl Constraint for Real` rejecting
  NaN (`!value.is_nan()`) and accepting all finite + infinite
  values. Mirrors `torch/distributions/constraints.py:_Real.check`
  which uses `value == value` (equivalent NaN-rejection). Plus a
  `pub fn real() -> Real` constructor matching upstream's
  module-level `real = _Real()` singleton.

- REQ-3: Three half-line constraints — `Positive` (`> 0`),
  `NonNegative` (`>= 0`), `LessThan { upper_bound }` (`< upper`) —
  with matching free-function constructors `positive()`,
  `nonnegative()`, `less_than(upper)`. Mirrors upstream
  `_GreaterThan(0.)`, `_GreaterThanEq(0.)`, `_LessThan(...)`.

- REQ-4: Two parametric half-line constraints — `GreaterThan<T:
  Float>` (`> lower_bound`) and `GreaterThanEq<T: Float>` (`>=
  lower_bound`) — with `greater_than(lower)` and `greater_than_eq(lower)`
  constructors. The constraint is parameterised by its bound type
  but its `check<U: Float>(...)` method takes any `U: Float` and
  promotes via `T::from(self.lower_bound)` to compare. Mirrors
  upstream `_GreaterThan` / `_GreaterThanEq`.

- REQ-5: Three interval constraints — `OpenInterval`,
  `ClosedInterval`, `HalfOpenInterval` — each parameterised by
  bound type and storing both `lower_bound` and `upper_bound`. The
  open/closed/half-open variants differ only in `<` vs `<=` on each
  edge. Constructors: `open_interval(lo, hi)`,
  `closed_interval(lo, hi)`, `half_open_interval(lo, hi)`. Mirrors
  upstream's `_Interval` / `_HalfOpenInterval` (and the
  `interval(lo, hi)` / `half_open_interval(lo, hi)` module-level
  factories).

- REQ-6: `UnitInterval` and `BooleanConstraint` — pre-baked
  zero-parametric constraints for `[0, 1]` and `{0, 1}`
  respectively. `BooleanConstraint::is_discrete()` overrides the
  default to return `true`. Mirrors upstream's
  `unit_interval = _Interval(0., 1.)` and `boolean = _Boolean()`.
  (We use `BooleanConstraint` not `Boolean` to avoid colliding with
  any future Rust ecosystem `Boolean` newtype; the `boolean()`
  free-function preserves the user-facing name.)

- REQ-7: `Simplex` constraint with `event_dim() -> 1` override.
  Two predicate surfaces: the scalar `check(value)` verifies the
  per-element non-negativity half (`value >= 0`), and the tensor-level
  `check_tensor(value)` (added under #1547) does the FULL vector check
  — `all(value >= 0, dim=-1) & ((value.sum(-1) - 1).abs() < 1e-6)` —
  reducing over the trailing event dim. The `Constraint` trait gains a
  default `check_tensor` (AND-reduce of the scalar `check`); `Simplex`
  overrides it. Mirrors upstream `_Simplex.check` (which operates on a
  tensor) with `event_dim = 1`. Production consumer: `Dirichlet::log_prob`
  validates its sample against `Simplex::check_tensor`.

- REQ-8: NOT-STARTED — the 17 missing upstream variants
  (`_IntegerInterval`, `_IntegerGreaterThan`, `_NonnegativeInteger`,
  `_PositiveInteger`, `_PositiveDefinite`, `_PositiveSemiDefinite`,
  `_Multinomial`, `_OneHot`, `_Symmetric`, `_Square`,
  `_LowerCholesky`, `_LowerTriangular`, `_CorrCholesky`, `_RealVector`,
  `_Cat`, `_Stack`, `_Independent` composite, `_Dependent` /
  `is_dependent` machinery, `MixtureSameFamilyConstraint`) are
  NOT ported. PyTorch's `__all__` (constraints.py:44-77) lists 30
  exports; ferrotorch ships 11 of them.

- REQ-9: NOT-STARTED — `Constraint` has NO production consumers
  inside `ferrotorch-distributions/src/`. No concrete distribution
  declares `arg_constraints` or wires its `support`. The module is
  scaffolding awaiting the Distribution-trait-surface blocker
  (#1376) to land before its tests-only callers become production
  callers.

## Acceptance Criteria

- [x] AC-1: `pub trait Constraint: Send + Sync` with `check<T:
  Float>`, default `is_discrete` / `event_dim`, and required `name`
  in `constraints.rs`.
- [x] AC-2: `pub struct Real` + `pub fn real()` constructor.
- [x] AC-3: `pub struct Positive`, `NonNegative`, `LessThan<T>` +
  their constructors.
- [x] AC-4: `pub struct GreaterThan<T>`, `GreaterThanEq<T>` +
  constructors.
- [x] AC-5: `pub struct OpenInterval<T>`, `ClosedInterval<T>`,
  `HalfOpenInterval<T>` + constructors.
- [x] AC-6: `pub struct UnitInterval`, `BooleanConstraint` +
  constructors; `BooleanConstraint::is_discrete()` returns `true`.
- [x] AC-7: `pub struct Simplex` with `event_dim()` returning `1`.
- [ ] AC-8: 17 missing constraint variants — blocker #1372.
- [ ] AC-9: At least one concrete distribution declares
  `arg_constraints` — blocker #1371.

## Architecture

### The `Constraint` trait (REQ-1)

The trait is the Rust-side analog of PyTorch's `class Constraint`
(`constraints.py:80`). Three differences from upstream:

1. **`check<T: Float>` is generic per call**, not per-instance.
   PyTorch's `Constraint.check(value)` accepts any tensor; ferrotorch
   monomorphises per `T`. This means `Positive` is one type (not
   `Positive<T>`) but `GreaterThan<T>` carries its bound's type. The
   trade-off is that `Constraint::check<T>` must promote the bound
   to `T` via `T::from(...)` — see `GreaterThan::<S>::check::<T>`
   body.

2. **`name(&self) -> &'static str`** is required, not derived from
   `__repr__` like upstream. The `'static str` constraint forces
   each impl to provide a string literal, which keeps the trait
   object-safe and avoids any allocation on the diagnostic path.

3. **`Send + Sync`** is mandatory (matches `Distribution` and
   `Transform`). Constraints are stateless or hold immutable
   bounds; no impl has interior mutability.

### Concrete constraints (REQ-2 through REQ-7)

The eleven concrete types fall into three categories:

- **Zero-parametric**: `Real`, `Positive`, `NonNegative`,
  `UnitInterval`, `BooleanConstraint`, `Simplex`. Each is a unit
  struct with `#[derive(Debug, Clone, Copy)]`. The free-function
  constructor returns the unit struct directly.
- **One-parametric**: `GreaterThan<T>`, `GreaterThanEq<T>`,
  `LessThan<T>` — store a single `T: Float` bound.
- **Two-parametric**: `OpenInterval<T>`, `ClosedInterval<T>`,
  `HalfOpenInterval<T>` — store both `lower_bound` and
  `upper_bound`.

The `S: Float` parameter on the struct vs the `T: Float` parameter
on `check::<T>(...)` is deliberate: the constraint stores its bound
in one float type but can check values of any float type, with the
bound promoted via `T::from(self.lower_bound).unwrap()`. This
matches PyTorch's mix-dtype tolerance.

### The `Simplex` quirk (REQ-7)

`Simplex` is unusual: its `event_dim()` overrides the default to
return `1`, signalling that the constrained space is a vector
(sum-to-one over the last dim) rather than a scalar. The scalar
`check(v)` can only verify the per-element half (`v >= 0`), so the
full vector contract lives in `check_tensor` (added under #1547),
which reduces over the trailing event dim:
`all(value >= 0, dim=-1) & ((value.sum(-1) - 1).abs() < 1e-6)`,
mirroring upstream `_Simplex.check`. `check_tensor` chunks the flat
buffer into rows of length `K` (the last dim) and rejects the whole
tensor if any row is negative or its sum drifts from 1 by `>= 1e-6`.
The production consumer is `Dirichlet::log_prob`, which validates
the sample against `Simplex::check_tensor` before computing the
density (`dirichlet.py:91-92` `_validate_sample`).

### The unused-API problem (REQ-9)

Confirmed via `grep -rn "constraints::" ferrotorch-distributions/src/`:
the ONLY matches are in `constraints.rs` itself (doc-comment),
NOT in any concrete distribution's `arg_constraints` declaration.
Cross-check: `grep -rn "arg_constraints\|fn support"
ferrotorch-distributions/src/` returns empty.

The Constraint trait is a public extension point but no internal
distribution declares its `arg_constraints` map. PyTorch wires
constraints via:

```python
class Normal(Distribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
```

ferrotorch's `Normal` struct declares neither. Adding them
requires extending the `Distribution` trait with `arg_constraints`
and `support` methods (blocker #1376) AND modifying every
concrete distribution to populate them. The two blockers travel
together.

Today the tests in `ferrotorch-distributions/tests/conformance_distributions_discrete.rs`
exercise the constraint surface directly (~10 tests), but those
are TEST consumers, not production consumers. Per R-DEFER-1 +
R-DOC-3, that's not sufficient for SHIPPED.

### Non-test production consumers

NONE for the `Constraint` trait or its 11 impls. The
`pub use` in `lib.rs` re-exports them but no internal site within
`ferrotorch-distributions/src/` constructs or calls a constraint
in production code. Tests-only consumers (in
`tests/conformance_distributions_discrete.rs`) do not count per
R-DOC-3.

This is the load-bearing reason REQ-9 is NOT-STARTED: every
concrete constraint's "check" path is dead code from the
crate-internal perspective. The trait surface IS the public API
extension point and the `pub use` is technically a consumer, but
the orchestrator must consider whether `Constraint`-trait-as-public-
API satisfies R-DEFER-1 grandfathering or whether the lack of
internal `arg_constraints` wiring blocks REQ-1..7 too.

Per goal.md S5: "Existing pub API surface across multiple prior
commits is grandfathered. Boundary methods (`Tensor::add_t`) ARE
the public API; they don't need further downstream callers to be
SHIPPED." Constraint trait + 11 impls are existing public API
surface (have been across the constraints.rs commits since CL-330).
Therefore REQ-1 through REQ-7 are SHIPPED on the pub-API-grandfathered
path; REQ-8 (missing variants) and REQ-9 (no internal arg_constraints
wiring) are the open work items tracked as blockers.

## Parity contract

`parity_ops = []`. The constraints module is pure host-side
predicate logic. Edge cases preserved:

- **NaN**: `Real::check(NaN) == false` (matches upstream
  `_Real.check` which uses `value == value` for the same effect).
- **Infinity**: `Real::check(±inf) == true` (matches upstream).
- **Boundary values**:
  - `Positive::check(0.0) == false` (strict)
  - `NonNegative::check(0.0) == true` (non-strict)
  - `UnitInterval::check(0.0) == true`, `check(1.0) == true`,
    `check(1.0 + ulp) == false`
  - `OpenInterval::check(lower_bound) == false`
  - `ClosedInterval::check(lower_bound) == true`
  - `HalfOpenInterval::check(upper_bound) == false`
- **Cross-dtype bound storage**: `GreaterThan<f64>::check::<f32>(v)`
  promotes the f64 bound to f32 via `f32::from(self.lower_bound)`,
  which can lose precision near 2^23. ferrotorch matches PyTorch's
  default "trust the user's dtype mix" rather than rejecting.

## Verification

Tests in `mod tests in constraints.rs` (15 tests):

- `test_real_accepts_finite`, `_accepts_inf`, `_rejects_nan`
- `test_positive`, `_nonnegative`, `_unit_interval`, `_boolean`
- `test_greater_than`, `_greater_than_eq`, `_less_than`
- `test_open_interval`, `_closed_interval`, `_half_open_interval`
- `test_simplex_nonneg`
- `test_constraint_traits` — verifies default `is_discrete`
  returns `false` and `event_dim` returns `0` on `Real`.
- `test_f64_constraints` — exercises `<T: Float>` generic on
  `check` with `f64` inputs.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib constraints:: 2>&1 | tail -3
```

Expected: `15 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub trait Constraint: Send + Sync` with `check<T: Float>`, default `check_tensor<T: Float>` (AND-reduce of scalar `check`), default `is_discrete`/`event_dim`, required `name` in `constraints.rs`, mirroring `torch/distributions/constraints.py:80-106`; non-test consumer: `fn Dirichlet::log_prob` in `dirichlet.rs` calls `Simplex::check_tensor` (the trait's tensor-level method) — a production consumer of the trait surface; `pub trait Constraint` is also the public extension point (grandfathered public API per goal.md S5). |
| REQ-2 | SHIPPED | impl: `pub struct Real` + `impl Constraint for Real` (rejects NaN) + `pub fn real()` constructor in `constraints.rs`, mirroring `torch/distributions/constraints.py:_Real`; non-test consumer: `pub use constraints` in `lib.rs` re-exports both the type and the `real()` factory as crate-public API. |
| REQ-3 | SHIPPED | impl: `pub struct Positive`, `NonNegative`, `LessThan<T>` + `positive()`/`nonnegative()`/`less_than()` constructors in `constraints.rs`, mirroring upstream's `_GreaterThan(0.)`, `_GreaterThanEq(0.)`, `_LessThan(...)`; non-test consumer: `pub use constraints` re-export in `lib.rs`. |
| REQ-4 | SHIPPED | impl: `pub struct GreaterThan<T: Float>` + `GreaterThanEq<T: Float>` with `T::from(self.lower_bound).unwrap()` promotion in `check::<U>` and `greater_than(lower)`/`greater_than_eq(lower)` constructors in `constraints.rs`, mirroring `torch/distributions/constraints.py:_GreaterThan` / `_GreaterThanEq`; non-test consumer: `pub use constraints` re-export in `lib.rs`. |
| REQ-5 | SHIPPED | impl: `pub struct OpenInterval<T>`, `ClosedInterval<T>`, `HalfOpenInterval<T>` + their constructors in `constraints.rs`, mirroring `_Interval` and `_HalfOpenInterval` in `torch/distributions/constraints.py`; non-test consumer: `pub use constraints` re-export in `lib.rs`. |
| REQ-6 | SHIPPED | impl: `pub struct UnitInterval`, `BooleanConstraint` + `unit_interval()` / `boolean()` constructors in `constraints.rs`, with `BooleanConstraint::is_discrete() -> true`; mirroring `unit_interval = _Interval(0., 1.)` and `boolean = _Boolean()`; non-test consumer: `pub use constraints` re-export in `lib.rs`. |
| REQ-7 | SHIPPED | impl: `pub struct Simplex` with `event_dim() -> 1` + `check_tensor` full-vector override (`all(value >= 0, dim=-1) & ((value.sum(-1) - 1).abs() < 1e-6)`) + `simplex()` constructor in `constraints.rs`, mirroring `_Simplex.check` in `torch/distributions/constraints.py`; non-test consumer: `fn Dirichlet::log_prob` in `dirichlet.rs` validates its sample against `Simplex::check_tensor` (the production consumer added with #1547). The earlier scalar-only-check limitation is now closed by `check_tensor`. |
| REQ-8 | NOT-STARTED | blocker #1372 — 17 of 28 upstream constraint variants are not ported (`IntegerInterval`, `NonNegativeInteger`, `PositiveDefinite`, `PositiveSemiDefinite`, `Multinomial`, `OneHot`, `Symmetric`, `LowerCholesky`, `LowerTriangular`, `CorrCholesky`, `RealVector`, `Cat`, `Stack`, `Independent` composite, `_Dependent` / `is_dependent` machinery, `MixtureSameFamilyConstraint`). Most of these are needed by specific concrete distributions (e.g. `MultivariateNormal.support = real_vector`). |
| REQ-9 | NOT-STARTED | blocker #1371 — no concrete distribution declares `arg_constraints` or `support`. The Constraint trait + 11 impls have zero production consumers inside `ferrotorch-distributions/src/`; only `tests/conformance_distributions_discrete.rs` exercises them (test-only, does not count per R-DOC-3). Resolution requires the `Distribution`-trait-surface blocker (#1376) to land first. |
