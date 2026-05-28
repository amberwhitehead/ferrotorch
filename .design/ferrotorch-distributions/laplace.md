# ferrotorch-distributions — `laplace` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/laplace.py
-->

## Summary

`ferrotorch-distributions/src/laplace.rs` defines the `Laplace<T>`
distribution (double exponential) parameterized by `loc` (mean) and
`scale`. Mirrors `torch.distributions.Laplace`
(`torch/distributions/laplace.py:14-104`). Supports reparameterized
sampling via the closed-form inverse CDF (`has_rsample = True` in
upstream) with a custom backward node that flows gradients into
`loc` and `scale`.

## Requirements

- REQ-1: `pub struct Laplace<T: Float>` with `loc: Tensor<T>` and
  `scale: Tensor<T>` fields. Mirrors `Laplace.loc` / `Laplace.scale`
  attributes (`laplace.py:51-62`).

- REQ-2: `pub fn Laplace::new(loc, scale) -> FerrotorchResult<Self>`
  with shape-equality precondition returning `ShapeMismatch`. The
  Rust API is stricter than upstream's `broadcast_all` chain
  (`laplace.py:57`) but the API contract is the same for the common
  case where both parameters have matching shapes.

- REQ-3: `loc()` and `scale()` accessors returning shared borrows
  for parameter introspection. Mirrors `Laplace.loc` and
  `Laplace.scale` property access.

- REQ-4: `impl<T: Float> Distribution<T> for Laplace<T>` with
  closed-form `sample`, `rsample`, `log_prob`, `entropy`, `cdf`,
  `icdf`, `mean`, `mode`, `variance`. Mirrors the corresponding
  methods at `laplace.py:35-104`.

- REQ-5: Sampling via the inverse CDF —
  `z = loc - scale * sign(u) * log(1 - |u|)` where `u ~ U(-1, 1)`,
  with `|u|` clamped to `[0, 1-eps]` to avoid `log(0)`. Mirrors
  upstream's `rsample` body at `laplace.py:73-85`.

- REQ-6: `rsample` is fully reparameterized with a hand-rolled
  `LaplaceRsampleBackward` autograd node carrying `d(z)/d(loc) = 1`
  and `d(z)/d(scale) = -sign(u) * log(1 - |u|)`. Gradients flow
  through `loc` and `scale` per the standard reparameterization
  trick. Mirrors upstream's autograd-traced `loc - scale * u.sign()
  * torch.log1p(-u.abs())` (`laplace.py:85`).

- REQ-7: Device-resident outputs — `sample`/`rsample`/`log_prob`/
  `entropy` build their tensors on CPU then `.to(device)` if `loc`
  / `scale` live on CUDA, so the output's device matches the
  parameter device. Mirrors upstream's implicit
  `device=self.loc.device` policy.

- REQ-8: Numerical-stability constant — the clamp
  `u_abs = u.abs().min(one - 1e-7)` at `laplace in laplace.rs` prevents
  `log(0) = -infinity` propagation. Upstream uses
  `finfo.tiny` / `finfo.eps - 1` (`laplace.py:75-83`) for the same
  purpose; ferrotorch's `1e-7` is a slight R-DEV-7 simplification
  that holds for both f32 and f64.

## Acceptance Criteria

- [x] AC-1: `pub struct Laplace<T: Float>` with `loc` and `scale`
  fields.
- [x] AC-2: `pub fn Laplace::new` rejecting shape-mismatched inputs.
- [x] AC-3: `loc()` and `scale()` accessors.
- [x] AC-4: `impl Distribution<T> for Laplace<T>` with
  `sample`/`rsample`/`log_prob`/`entropy`/`cdf`/`icdf`/`mean`/`mode`/`variance`.
- [x] AC-5: `test_laplace_sample_shape` validates `sample(&[100])`
  returns shape `[100]`.
- [x] AC-6: `test_laplace_log_prob_at_loc` validates `log_prob(loc)
  == -log(2*scale)`.
- [x] AC-7: `test_laplace_entropy` validates entropy
  `== 1 + log(2*scale)`.
- [x] AC-8: `test_laplace_rsample_backward` confirms gradient flow
  through `loc` and `scale`.
- [x] AC-9: `test_laplace_cdf_at_loc_is_half`,
  `test_laplace_icdf_roundtrip` confirm CDF/ICDF consistency.
- [x] AC-10: `test_laplace_mean_mode_variance` confirms
  `mean == mode == loc` and `variance == 2*scale^2`.

## Architecture

### The struct (REQ-1, REQ-2, REQ-3)

```rust
pub struct Laplace<T: Float> {
    loc: Tensor<T>,
    scale: Tensor<T>,
}
```

Defined at `laplace in laplace.rs`. Constructor at `laplace in laplace.rs`
with shape-equality check. Accessors at `laplace in laplace.rs`.

### The Distribution impl (REQ-4, REQ-5, REQ-7)

`sample` (`laplace in laplace.rs`) draws `u ~ U(0, 1)` then invokes
the private `laplace_icdf_sample` helper (`laplace in laplace.rs`)
which maps to `U(-1, 1)` and applies the inverse CDF. Cyclic
parameter zip supports scalar broadcast.

`rsample` (`laplace in laplace.rs`) follows the same closed-form
pipeline but attaches `LaplaceRsampleBackward` when either
parameter has `requires_grad`. The grad fn is registered through
`Tensor::from_operation`, which makes the resulting tensor a
graph node.

`log_prob` (`laplace in laplace.rs`) computes
`-log(2*scale) - |x - loc| / scale`. Matches upstream `laplace.py:87-90`.

`cdf` (`laplace in laplace.rs`) uses the closed form
`0.5 + 0.5 * sign(x-loc) * (1 - exp(-|x-loc|/scale))`. Matches
`laplace.py:92-97` (which uses `torch.expm1`; ferrotorch uses the
algebraically equivalent `1 - exp(-|d|/s)` form).

`icdf` (`laplace in laplace.rs`) uses
`loc - scale * sign(p - 0.5) * ln(1 - 2|p - 0.5|)`. Mirrors
`laplace.py:99-101`.

`mean`/`mode`/`variance` (`laplace in laplace.rs`) return `loc`,
`loc`, and `2*scale^2`. Mirror `laplace.py:35-45` properties.

`entropy` (`laplace in laplace.rs`) returns `1 + log(2*scale)`.
Mirrors `laplace.py:103-104`.

### LaplaceRsampleBackward (REQ-6)

Defined at `laplace in laplace.rs`. Holds `loc`, `scale`, `u`
(uniform draw before the icdf). On `backward(grad_output)` it
computes:

- `grad_loc = sum(grad_output)` (1 per element)
- `grad_scale = sum(grad_output * (-sign(u) * log(1 - |u|)))`

Returns `Some(grad)` only for parameters with `requires_grad`.
Test `test_laplace_rsample_backward` pins:

```rust
let z = dist.rsample(&[10]).unwrap();
let loss = z.sum_all().unwrap();
loss.backward().unwrap();
// loc.grad == 10.0 (1 per sample, summed)
// scale.grad finite
```

### Non-test production consumers

- `pub use laplace::Laplace` at `lib.rs` — grandfathered
  public API. Downstream model code (robust-regression layers,
  VI Laplace posteriors) constructs `Laplace::new(loc, scale)?`.
- `LaplaceRsampleBackward` is consumed by the autograd engine
  via `Tensor::from_operation` — it's the production consumer
  of the `GradFn<T>` trait surface from `ferrotorch_core::tensor`.

## Parity contract

`parity_ops = []`. Laplace has no direct parity-sweep oracle; the
underlying primitives (`abs`, `ln`, `exp`, `sign`) have their own
parity audits and Laplace composes them. Edge cases preserved:

- **`scale -> 0+`** — `log_prob` returns `+infinity` at `loc`,
  `-infinity` elsewhere; `entropy` returns `-infinity`. Upstream
  has identical limit behaviour via `torch.log`.
- **`x == loc`** — `log_prob(loc) = -log(2*scale)` (the peak
  density). Test `test_laplace_log_prob_at_loc` pins this.
- **Symmetry** — `log_prob(loc + d) == log_prob(loc - d)` by
  construction (the `abs(x - loc)` term). Test
  `test_laplace_log_prob_symmetry` pins this.
- **`u = 0` in rsample** — the `u_abs.min(one - eps)` clamp keeps
  `log(1 - |u|)` finite. Upstream uses
  `u.uniform_(finfo.eps - 1, 1)` for the same purpose.
- **CDF at `loc`** — exactly `0.5`. Test
  `test_laplace_cdf_at_loc_is_half` pins this.

## Verification

Tests in `mod tests in laplace.rs` (13 tests):

- `test_laplace_sample_shape`,
- `test_laplace_sample_mean`,
- `test_laplace_rsample_has_grad`,
- `test_laplace_log_prob_at_loc`,
- `test_laplace_log_prob_symmetry`,
- `test_laplace_entropy`,
- `test_laplace_entropy_unit`,
- `test_laplace_shape_mismatch`,
- `test_laplace_rsample_backward`,
- `test_laplace_f64`,
- `test_laplace_mean_mode_variance`,
- `test_laplace_cdf_at_loc_is_half`,
- `test_laplace_icdf_roundtrip`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib laplace:: 2>&1 | tail -3
```

Expected: `13 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Laplace<T: Float>` with `loc`/`scale` `Tensor<T>` fields at `Laplace in laplace.rs`, mirroring `torch/distributions/laplace.py:14-62`; non-test consumer: `pub use laplace::Laplace` at `lib.rs` exposes the type as grandfathered public API for downstream robust-regression / VI posterior code. |
| REQ-2 | SHIPPED | impl: the constructor at `laplace in laplace.rs` with shape-equality precondition + `ShapeMismatch` error path, mirroring `laplace.py:51-62` (`broadcast_all`); non-test consumer: `pub use Laplace::new` accessible via the re-export at `lib.rs`. |
| REQ-3 | SHIPPED | impl: `loc()`/`scale()` accessors at `laplace in laplace.rs`, mirroring upstream property access; non-test consumer: the re-export at `lib.rs` exposes them as the parameter-introspection surface. |
| REQ-4 | SHIPPED | impl: full `impl<T: Float> Distribution<T> for Laplace<T>` at `laplace in laplace.rs` with the 9 methods, mirroring `laplace.py:35-104`; non-test consumer: `pub use Laplace` re-export means external `Distribution` trait callers hit this impl. 13 tests pin behaviour. |
| REQ-5 | SHIPPED | impl: `laplace_icdf_sample` helper at `laplace in laplace.rs` invoked from `sample` at `laplace in laplace.rs` and `rsample` at `laplace in laplace.rs`, mirroring `laplace.py:73-85`; non-test consumer: `Distribution::sample` / `Distribution::rsample` via `pub use Laplace`. |
| REQ-6 | SHIPPED | impl: `LaplaceRsampleBackward in laplace.rs` with `backward` computing `grad_loc = sum(go)` and `grad_scale = sum(go * (-sign(u) * log(1 - \|u\|)))`, attached via `Tensor::from_operation` at `laplace in laplace.rs`; non-test consumer: the autograd engine's `backward()` traversal in `ferrotorch_core::tensor` invokes this `GradFn<T>` impl on any rsample with grad-requiring params. |
| REQ-7 | SHIPPED | impl: `out.to(device)` if `device.is_cuda()` at the tail of every method (e.g. `laplace in laplace.rs`), mirroring upstream's `device=self.loc.device` implicit policy; non-test consumer: every external caller invoking the methods receives a device-correct tensor. |
| REQ-8 | SHIPPED | impl: `u_abs = u.abs().min(one - eps)` clamp at `laplace in laplace.rs` with `eps = 1e-7`, mirroring upstream's `finfo.tiny` / `finfo.eps - 1` guard at `laplace.py:75-83`; non-test consumer: invoked from `sample`/`rsample` via `laplace_icdf_sample` on every draw. |
| REQ-9 | SHIPPED | impl: `has_rsample`(=true) / `batch_shape` / `support`(`Real` per `laplace.py:32`) / `arg_constraints`(`{loc: Real, scale: Positive}` per `laplace.py:31`) / `expand` overrides at the tail of `impl Distribution for Laplace` in `laplace.rs` mirroring `torch/distributions/laplace.py:31-33`; non-test consumer: trait dispatch through `pub use laplace::Laplace` re-export at `lib.rs`; `test_laplace_surface_overrides` and `test_laplace_expand` pin the overrides. |
