# ferrotorch-distributions — `exp_family` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/exp_family.py
  - torch/distributions/kl.py
-->

## Summary

`ferrotorch-distributions/src/exp_family.rs` provides the generic
exponential-family KL divergence — the Bregman-divergence fallback that
mirrors PyTorch's `@register_kl(ExponentialFamily, ExponentialFamily)`
registration (`_kl_expfamily_expfamily`, `torch/distributions/kl.py:282-300`).
It is the lowest-priority KL registration: it fires for two distributions of
the **same** exponential family when no more-specific `(P, Q)` arm matched.

The `ExponentialFamily<T>` trait itself lives in `lib.rs` (it was introduced by
#1404/#1407 as a supertrait surface alongside `Distribution<T>`); this module
extends the design with the `mean_params` method and the generic KL machinery.

An exponential-family density has the canonical form
`p(x; θ) = exp(⟨t(x), η(θ)⟩ − A(η) + k(x))` where η are the natural
parameters, `t(x)` the sufficient statistic, `A` the log-normalizer, and `k`
the carrier measure (`torch/distributions/exp_family.py:11-30`). The Bregman
KL between two members of the same family is

```text
KL(p ‖ q) = A(η_q) − A(η_p) − ⟨η_q − η_p, ∇A(η_p)⟩
```

where `∇A(η_p) = E_p[t(X)]` is the vector of mean parameters (expected
sufficient statistics).

## Deviation from upstream — analytic gradient (R-DEV-7)

Upstream computes `∇A(η_p)` by reverse-mode autograd through the
`_log_normalizer` callable:
`torch.autograd.grad(lg_normal.sum(), p_nparams, create_graph=True)`
(`torch/distributions/kl.py:292`, mirroring `exp_family.py:62` for entropy).

ferrotorch's `_log_normalizer` impls evaluate on host-resident `data_vec()`
and build no autograd graph, so differentiating *through* them is impossible
(this was the open prereq in blocker #1575). Rather than retrofit a
differentiable host path through every distribution's `log_normalizer`, each
`ExponentialFamily` impl supplies `mean_params` — the same gradient
`∇A(η) = E[t(X)]` in **closed form**. This is cleaner, allocation-free, and
exact; the shipped tests verify the resulting Bregman KL equals both the
specific-pair closed-form KL and live PyTorch (via
`torch.distributions.kl_divergence` and `_kl_expfamily_expfamily`) to ~1e-9.
The upstream contract (the API surface and the numeric result) is preserved;
only the gradient mechanism differs.

## Requirements

- REQ-1: `ExponentialFamily<T>` trait exposes `natural_params`,
  `log_normalizer`, **and** `mean_params` (the analytic `∇A`), plus the
  default `mean_carrier_measure = 0`. Mirrors
  `torch/distributions/exp_family.py:32-53` (`_natural_params`,
  `_log_normalizer`, `_mean_carrier_measure`) with `mean_params` standing in
  for upstream's autograd-derived gradient.

- REQ-2: `pub fn kl_expfamily_expfamily<T>(p, q: &dyn ExponentialFamily<T>)`
  computes the Bregman KL above, broadcasting `p`'s batch shape against `q`'s.
  Mirrors `_kl_expfamily_expfamily` (`torch/distributions/kl.py:282-297`). For
  univariate families (`event_shape == []`) the upstream
  `_sum_rightmost(..., 0)` is the identity; the inner product is summed over
  the natural-parameter components element-wise.

- REQ-3: `pub fn try_kl_expfamily<T>(p, q: &dyn Distribution<T>)` is the
  same-family dispatch hook. It downcasts both operands to each registered
  exponential-family concrete type and fires `kl_expfamily_expfamily` only when
  both succeed for the *same* type — the Rust analog of the
  `if type(p) is not type(q): raise NotImplementedError` guard at
  `torch/distributions/kl.py:284`. Returns `Ok(None)` when no registered family
  matches (the caller then raises the no-formula error). Registered families:
  Normal, Poisson, Gamma, Exponential, Beta, Bernoulli.

- REQ-4: `ExponentialFamily` is implemented for the distributions PyTorch
  marks as exponential families that exist in ferrotorch with a closed-form
  `mean_params`: Normal, Poisson, Gamma, Exponential, Beta, Bernoulli. Each
  cites its exp-family parameterization from the matching upstream
  `_natural_params` / `_log_normalizer` block.

## REQ status

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | `pub trait ExponentialFamily<T>` with `natural_params`/`log_normalizer`/`mean_params` in `lib.rs` mirrors `torch/distributions/exp_family.py:32-53`; consumer: `fn kl_expfamily_expfamily` in `exp_family.rs` reads all three on both operands, and `impl ExponentialFamily for {Normal,Poisson,Gamma,Exponential,Beta,Bernoulli}` provide the bodies |
| REQ-2 | SHIPPED | `pub fn kl_expfamily_expfamily` in `exp_family.rs` per `torch/distributions/kl.py:282-297`; consumer: `fn try_kl_expfamily` in `exp_family.rs` calls it |
| REQ-3 | SHIPPED | `pub fn try_kl_expfamily` in `exp_family.rs` per `torch/distributions/kl.py:284`; consumer: `pub fn kl_divergence_dyn` in `kl.rs` invokes it on the no-formula fall-through (every public `kl_divergence` call routes through `kl_divergence_dyn`) |
| REQ-4 | SHIPPED | `impl ExponentialFamily for Normal` (`normal.rs`, per `normal.py:116-122`), `Poisson` (`poisson.rs`, per `poisson.py:81-87`), `Gamma` (`gamma.rs`, per `gamma.py:109-114`), `Exponential` (`exponential.rs`, per `exponential.py:88-94`), `Beta` (`beta.rs`, per `beta.py:112-118`), `Bernoulli` (`bernoulli.rs`, per `bernoulli.py:139-145`); consumer: `fn try_kl_expfamily` downcasts to each |

## Note on shadowing (faithful to upstream)

Every one of the six registered families *also* has a more-specific
same-family KL arm in `kl.rs` (`kl_normal_normal`, `kl_gamma_gamma`, …), so for
all distributions currently built into the crate the generic Bregman fallback
is shadowed — exactly as in PyTorch, where `_dispatch_kl` selects the most
specific registration and the `(ExponentialFamily, ExponentialFamily)` arm is
the registry backstop. The fallback fires through `kl_divergence` only for a
same-family exponential-family pair with no specific arm. The
`exp_family::tests` and `tests/divergence_kl_1575_expfamily_bregman.rs` pin the
Bregman result against both the specific arm and live PyTorch, so the path is
verified correct even where it is shadowed.

## Acceptance criteria

- AC-1: For Normal/Gamma/Exponential/Beta/Poisson/Bernoulli, the Bregman KL
  equals the live-PyTorch `_kl_expfamily_expfamily` value (float64) to 1e-9.
- AC-2: The Bregman KL equals the specific-pair `kl_divergence` value for the
  same pair to 1e-9 (drop-in equivalence).
- AC-3: `try_kl_expfamily` returns `Some` for a same-family exp pair and `None`
  for a cross-family pair (Normal vs Gamma) — mirroring the upstream
  `NotImplementedError` guard at `kl.py:284`.
- AC-4: `kl_expfamily_expfamily` broadcasts `p` against `q` (e.g. `p:[2]` vs
  `q:[1]` -> `[2]`).
