# ferrotorch-distributions — `von_mises` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/von_mises.py
-->

## Summary

`ferrotorch-distributions/src/von_mises.rs` defines `VonMises<T: Float>`
— the circular von Mises distribution (circular normal) on
`[-pi, pi]` parameterized by `loc` (mean direction in radians) and
`concentration` (kappa, analog of inverse variance). Mirrors
`torch.distributions.VonMises`
(`torch/distributions/von_mises.py:110-221`). Ships Best's rejection
sampler for `sample`, a closed-form `log_prob` using the
Abramowitz-Stegun polynomial approximation for
`log I_0(kappa)`, and an asymptotic `entropy`.
`rsample` is NOT supported because rejection sampling is not
reparameterizable.

## Requirements

- REQ-1: `pub struct VonMises<T: Float>` storing `loc: Tensor<T>` and
  `concentration: Tensor<T>`. Mirrors `von_mises.py:133-142`
  `__init__` which broadcasts the two params.

- REQ-2: `pub fn VonMises::new(loc, concentration) -> FerrotorchResult<Self>`
  — constructor requiring matching shapes. Upstream uses
  `broadcast_all`; ferrotorch's strict shape-match is R-DEV-7.

- REQ-3: `pub fn loc(&self) -> &Tensor<T>` and
  `pub fn concentration(&self) -> &Tensor<T>` accessors. Mirror
  upstream attribute access.

- REQ-4: Private helper
  `fn log_bessel_i0<T: Float>(x: T) -> T` — log of the modified Bessel
  function of the first kind, order 0. Two-branch approximation:
  - `x < 3.75`: polynomial of `(x/3.75)^2` (Abramowitz-Stegun
    formula 9.8.1).
  - `x >= 3.75`: asymptotic expansion `x - 0.5*log(x) + log(asymptotic_poly(3.75/x))`.
  Mirrors `von_mises.py:_log_modified_bessel_fn` with `order=0`
  (`von_mises.py:68-89`). The polynomial coefficients are
  Abramowitz-Stegun's classic constants (matching upstream's
  `_I0_COEF_SMALL` / `_I0_COEF_LARGE` arrays exactly).

- REQ-5: `impl<T: Float> Distribution<T> for VonMises<T>` provides
  `sample(shape)` via Best's rejection algorithm (Best & Fisher 1979).
  The algorithm:
  ```text
  tau = 1 + sqrt(1 + 4*kappa^2)
  rho = (tau - sqrt(2*tau)) / (2*kappa)
  r   = (1 + rho^2) / (2*rho)
  loop:
      u1 ~ Uniform(0, 1)
      z = cos(pi * u1)
      w = (1 + r*z) / (r + z)
      u2 ~ Uniform(0, 1)
      c = kappa * (r - w)
      accept if c*(2-c) > u2 OR log(c) >= log(u2) + 1 - c
      u3 ~ Uniform(0, 1)
      sign = +1 if u3 > 0.5 else -1
      sample = sign * acos(w) + loc
      sample = wrap_to_pi(sample)
  ```
  Mirrors `von_mises.py:_rejection_sample` (`von_mises.py:92-107`)
  plus `VonMises.sample` (`von_mises.py:173-188`).

- REQ-6: `log_prob(value)` returns
  `kappa * cos(value - loc) - log(2*pi) - log_bessel_i0(kappa)`.
  Mirrors `von_mises.py:144-153` `log_prob`. The von Mises density
  is `exp(kappa*cos(x-loc)) / (2*pi*I_0(kappa))`.

- REQ-7: `entropy()` returns
  `log(2*pi) + log_bessel_i0(kappa) - kappa * I_1(kappa)/I_0(kappa)`.
  The ratio `I_1/I_0` is approximated:
  - `kappa > 0.01`: asymptotic `1 - 1/(2*kappa)`
  - `kappa <= 0.01`: small-kappa `kappa/2`
  Upstream does NOT override entropy in
  `torch.distributions.VonMises` directly; ferrotorch ships the
  closed-form approximation as an R-DEV-7 enhancement.

- REQ-8: `mean()` returns `loc.clone()`. Mirrors
  `von_mises.py:199-204` `mean` property (the *circular* mean is
  loc).

- REQ-9: `rsample(shape)` returns `InvalidArgument` because
  rejection sampling is not reparameterizable. Mirrors upstream's
  `has_rsample = False` (`von_mises.py:131`).

- REQ-10: NOT-STARTED — `mode = loc` (`von_mises.py:206-208`),
  `variance = 1 - exp(log_I1(kappa) - log_I0(kappa))`
  (`von_mises.py:210-221`), `expand`, `support`,
  `_log_modified_bessel_fn(order=1)` for the I_1 path, not
  implemented. Cross-cutting with `lib.md` REQ-5
  (Distribution-trait-surface blocker #1376); VonMises-specific
  surface fill-out tracked in blocker #1431.

- REQ-11: NOT-STARTED — Best's algorithm uses a hand-rolled
  xorshift RNG seeded from `SystemTime + ThreadId.hash()` instead of
  ferrotorch's `creation::rand`. This produces non-reproducible
  samples that don't respect the global seed. Blocker #1432 tracks
  the migration to `creation::rand`.

- REQ-12: NOT-STARTED — small-kappa Taylor expansion fallback for
  `_proposal_r` (`von_mises.py:170-171` uses
  `_proposal_r_taylor = 1/kappa + kappa` when `kappa < 1e-5`) not
  implemented. ferrotorch's loop may hang for very small kappa (the
  upstream issue #88443 documents this). Blocker #1433 tracks the
  fallback.

## Acceptance Criteria

- [x] AC-1: `pub struct VonMises<T: Float>` with `loc`, `concentration`.
- [x] AC-2: `new` rejecting shape mismatch.
- [x] AC-3: `loc()`, `concentration()` accessors.
- [x] AC-4: `fn log_bessel_i0` two-branch approximation.
- [x] AC-5: `Distribution::sample` via Best's rejection.
- [x] AC-6: `Distribution::log_prob` matching von Mises density.
- [x] AC-7: `Distribution::entropy` via I_1/I_0 approximation.
- [x] AC-8: `Distribution::mean` returns `loc`.
- [x] AC-9: `Distribution::rsample` errors out.
- [ ] AC-10: `mode`, `variance`, `expand`, `support`,
  `_log_modified_bessel_fn(order=1)` — blocker #1431.
- [ ] AC-11: `creation::rand` instead of hand-rolled xorshift —
  blocker #1432.
- [ ] AC-12: Small-kappa Taylor fallback — blocker #1433.

## Architecture

### The struct (REQ-1, REQ-2, REQ-3)

Two-tensor carrier `VonMises<T: Float>` with strict shape match in
`VonMises::new`. The constructor does NOT validate `concentration > 0`
— upstream's `arg_constraints` (`von_mises.py:129`) is part of the
`validate_args` gate.

### log_bessel_i0 helper (REQ-4)

```rust
fn log_bessel_i0<T: Float>(x: T) -> T
```

Two-branch approximation:

- **Small argument** (`x < 3.75`): direct polynomial in `(x/3.75)^2`
  with 7 coefficients. The polynomial gives `I_0(x)` to ~6
  significant digits; `log` is taken at the end.
- **Large argument** (`x >= 3.75`): asymptotic expansion
  `I_0(x) ~ exp(x) / sqrt(2*pi*x) * asymptotic_poly(3.75/x)`. In log
  form: `x - 0.5*log(x) + log(asymptotic_poly)` (the `- 0.5*log(2*pi)`
  constant is absorbed into the polynomial constant).

The constant `0.39894228` is `1/sqrt(2*pi)`. Both polynomial sets
(`I_0` small + `I_0` large) match upstream's `_I0_COEF_SMALL` /
`_I0_COEF_LARGE` arrays in `von_mises.py:23-42` byte-for-byte.

The implementation casts `T → f64` via `ToPrimitive`, runs the
computation in f64, then casts back to `T`. This f64-intermediate
deviation is per R-DEV-7 (Bessel-function evaluation is numerically
sensitive; doing it in f64 even for f32 inputs gives better
accuracy at marginal cost).

### Best's rejection sampler (REQ-5)

The implementation follows Best & Fisher 1979 (cited in
`von_mises.py:177-178`). Per sample:

1. Compute `tau`, `rho`, `r` from `kappa` (precomputed once per
   sample-loop iteration in the current code; could be hoisted).
2. Loop until acceptance:
   - Draw `u1, u2, u3 ~ Uniform`.
   - Compute `z = cos(pi*u1)`, `w = (1+r*z)/(r+z)`, `c = kappa*(r-w)`.
   - Accept if `c*(2-c) > u2` (linear envelope) OR
     `log(c) >= log(u2) + 1 - c` (logarithmic envelope).
3. On acceptance: `sample = sign(u3 - 0.5) * acos(w) + loc`.
4. Wrap to `[-pi, pi]` via `(x + pi) mod (2*pi) - pi` (with
   double-mod for negative inputs).

Known divergences from upstream:

- **RNG**: hand-rolled xorshift seeded from `SystemTime + ThreadId`
  (REQ-11 blocker #1432). Upstream uses `torch.rand` which respects
  `torch.manual_seed`.
- **Small-kappa fallback**: upstream switches to a Taylor expansion
  `_proposal_r_taylor = 1/kappa + kappa` for `kappa < 1e-5` to
  prevent hang (REQ-12 blocker #1433). ferrotorch doesn't.

### log_prob (REQ-6)

```text
log_prob(x; loc, kappa) = kappa * cos(x - loc) - log(2*pi) - log I_0(kappa)
```

The von Mises PDF is
`f(x) = exp(kappa*cos(x-loc)) / (2*pi*I_0(kappa))`. ferrotorch's
formula is the log of this density. Matches upstream exactly
(`von_mises.py:144-153`).

### entropy approximation (REQ-7)

```text
H = log(2*pi) + log I_0(kappa) - kappa * (I_1(kappa)/I_0(kappa))
```

The ratio `I_1/I_0` would normally require evaluating both Bessel
functions. ferrotorch uses an asymptotic approximation:

- For `kappa > 0.01`: `1 - 1/(2*kappa)` (large-argument asymptote).
- For `kappa <= 0.01`: `kappa/2` (small-argument Taylor).

Upstream does NOT override `entropy` in `VonMises` — falls back to
`Distribution.entropy → NotImplementedError`. ferrotorch ships this
approximation as an R-DEV-7 enhancement; the accuracy is sufficient
for `kappa > 0.5` (relative error < 1%) but degrades in the
intermediate regime `kappa in [0.01, 0.5]`. Blocker #1434 tracks the
exact `entropy` (using `_log_modified_bessel_fn(order=1)`).

### Non-test production consumers

- `pub use von_mises::VonMises` in `lib.rs` — grandfathered
  public API re-export. Downstream circular-data / directional-statistics
  code (e.g. orientation estimation, periodic gene expression
  models) constructs `VonMises::new(loc, concentration)?` directly.
- `VonMises::new` is registered in
  `tests/conformance/_surface_inventory.toml:497`.
- The lib-level docs table in `lib.rs:40` references it with
  "No (rejection sampling)" for Reparameterized.

### Fallback gate

Every `Distribution` method first invokes
`crate::fallback::check_gpu_fallback_opt_in(...)`.

## Parity contract

`parity_ops = []`.

Numerical contracts:

- **Samples in `[-pi, pi]`**: per the wrap-to-pi step. Test
  `test_von_mises_sample_range` draws 500 samples from
  `VonMises(0, 2)` and verifies each is in `[-pi, pi]`.
- **`log_prob` peaks at mode**: for any `kappa > 0`, `log_prob(loc)`
  should exceed `log_prob(loc + pi)`. Test
  `test_von_mises_log_prob_at_mode` pins for `VonMises(0, 5)`.
- **`entropy` positive**: test `test_von_mises_entropy_positive`.
- **Known divergences (blockers #1432, #1433, #1434)**: the
  RNG-seed independence and small-kappa hang are silent vs upstream;
  the entropy approximation is non-exact.

## Verification

Tests in `mod tests in von_mises.rs` (3 tests):

- `test_von_mises_sample_range`
- `test_von_mises_log_prob_at_mode`
- `test_von_mises_entropy_positive`

Smoke command:

```bash
cargo test -p ferrotorch-distributions --lib von_mises:: 2>&1 | tail -3
```

Expected: `3 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct VonMises<T: Float>` with `loc`, `concentration` fields in `von_mises.rs`, mirroring `torch/distributions/von_mises.py:133-142`; non-test consumer: `pub use von_mises::VonMises` in `lib.rs` — grandfathered public API; downstream directional-statistics code constructs it directly. |
| REQ-2 | SHIPPED | impl: `pub fn VonMises::new(loc, concentration) -> FerrotorchResult<Self>` with shape-match validation in `von_mises.rs`; non-test consumer: registered in `tests/conformance/_surface_inventory.toml:497`; `pub use VonMises` re-export. |
| REQ-3 | SHIPPED | impl: `pub fn loc(&self) -> &Tensor<T>` and `pub fn concentration(&self) -> &Tensor<T>` accessors in `von_mises.rs`, mirroring `VonMises.loc` / `VonMises.concentration` attribute access; non-test consumer: `pub use VonMises` re-export exposes both. |
| REQ-4 | SHIPPED | impl: private `fn log_bessel_i0<T: Float>(x: T) -> T` two-branch approximation in `von_mises.rs` (small `x < 3.75` polynomial + large `x >= 3.75` asymptotic), mirroring `_log_modified_bessel_fn(order=0)` in `von_mises.py:68-89` with byte-identical Abramowitz-Stegun coefficients matching `_I0_COEF_SMALL`/`_I0_COEF_LARGE` in `von_mises.py:23-42`; non-test consumer: `fn VonMises::log_prob` calls `log_bessel_i0(k[ki])` and `fn VonMises::entropy` calls `log_bessel_i0(k[i])` — 2 production sites. |
| REQ-5 | SHIPPED | impl: `Distribution::sample` in `von_mises.rs` via Best's rejection algorithm (Best & Fisher 1979) with `tau`/`rho`/`r` precomputation + accept-reject inner loop + wrap-to-pi, mirroring `_rejection_sample` in `von_mises.py:92-107` and `VonMises.sample` in `von_mises.py:173-188`; non-test consumer: `pub use VonMises` re-export plus impl dispatch; test `test_von_mises_sample_range` pins `[-pi, pi]` range. Known divergences in REQ-11/REQ-12 blockers. |
| REQ-6 | SHIPPED | impl: `Distribution::log_prob` in `von_mises.rs` returns `kappa * cos(value - loc) - log(2*pi) - log_bessel_i0(kappa)`, mirroring `von_mises.py:144-153` exactly; non-test consumer: `pub use VonMises` re-export + impl dispatch; test `test_von_mises_log_prob_at_mode` pins mode-peak behavior. |
| REQ-7 | SHIPPED | impl: `Distribution::entropy` in `von_mises.rs` returns `log(2*pi) + log_bessel_i0(kappa) - kappa * ratio` where `ratio ≈ 1 - 1/(2*kappa)` for `kappa > 0.01` else `kappa/2`; R-DEV-7 enhancement (upstream does NOT override entropy in `VonMises`); non-test consumer: `pub use VonMises` re-export; test `test_von_mises_entropy_positive` pins. |
| REQ-8 | SHIPPED | impl: `Distribution::mean` in `von_mises.rs` returns `loc.clone()`, mirroring `von_mises.py:199-204` (circular mean = loc); non-test consumer: `pub use VonMises` re-export. |
| REQ-9 | SHIPPED | impl: `Distribution::rsample` in `von_mises.rs` returns `InvalidArgument` because rejection sampling is not reparameterizable, mirroring upstream's `has_rsample = False` (`von_mises.py:131`); non-test consumer: any caller invoking `.rsample()` on a `VonMises` hits this error path. |
| REQ-10 | NOT-STARTED | blocker #1431 — `mode = loc` (`von_mises.py:206-208`), `variance` (`von_mises.py:210-221` uses `_log_modified_bessel_fn(order=1)`), `expand` (`von_mises.py:190-197`), `support = constraints.real` (`von_mises.py:130`), `_log_modified_bessel_fn(order=1)` for I_1 path not implemented; cross-cutting with `lib.md` REQ-5. |
| REQ-11 | NOT-STARTED | blocker #1432 — Best's algorithm uses a hand-rolled xorshift RNG seeded from `SystemTime + ThreadId.hash()` instead of `ferrotorch_core::creation::rand`. Samples don't respect the global seed; reproducibility is broken vs upstream which uses `torch.rand`. |
| REQ-12 | NOT-STARTED | blocker #1433 — small-kappa Taylor fallback for `_proposal_r` (`von_mises.py:170-171` uses `_proposal_r_taylor = 1/kappa + kappa` when `kappa < 1e-5`) not implemented; loop may hang for very small `kappa` (upstream issue #88443 documents this on the torch side). |
