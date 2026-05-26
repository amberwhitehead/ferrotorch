# `Lbfgs` — Limited-memory BFGS optimizer

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/lbfgs.py
-->

## Summary

`ferrotorch-optim/src/lbfgs.rs` defines `Lbfgs<T>` and `LbfgsConfig`,
the Limited-memory BFGS quasi-Newton optimizer mirroring
`torch.optim.LBFGS` (`torch/optim/lbfgs.py:206`). Implements the two-loop
recursion against a curvature-pair history plus an optional Strong Wolfe
line search (Nocedal & Wright, Algorithms 3.5 + 3.6). CL-1105 Pattern B
keeps every step on the parameter's device.

## Requirements

- REQ-1: `pub struct LbfgsConfig` with `lr` (1.0), `max_iter` (20),
  `max_eval` (`None` → `max_iter * 5 / 4`), `tolerance_grad` (1e-7),
  `tolerance_change` (1e-9), `history_size` (10),
  `line_search_fn` (`None`), `maximize` (false). Mirrors
  `torch/optim/lbfgs.py:240-280`.
- REQ-2: `pub enum LineSearchFn { StrongWolfe }` matching upstream's
  `line_search_fn="strong_wolfe"` string kwarg. Other values are not
  expressible since the enum has one variant only.
- REQ-3: `pub struct Lbfgs<T: Float>` with `new(params, config)` and
  full `Optimizer<T>` impl. Maintains an `LbfgsState<T>` holding the
  curvature `(s, y, rho)` history (1-D `Tensor<T>` on-device) plus
  `prev_flat_params` / `prev_flat_grad` caches.
- REQ-4: `gather_params` and `gather_grads` flatten every parameter
  (resp. its gradient) into a single 1-D device-resident `Tensor<T>` via
  `tensor_flatten` + `tensor_cat`. `scatter_params` reverses by
  narrowing chunks and `reshape_t`-ing back to original shape.
- REQ-5: `two_loop_recursion` implements the canonical L-BFGS two-loop
  recursion (Nocedal & Wright, Algorithm 7.4): backward pass computes
  `alpha_i = rho_i * (s_i . q)` and `q -= alpha_i * y_i`; then
  `H_0 = gamma * I` where `gamma = (s_last . y_last) / (y_last . y_last)`;
  then forward pass `beta = rho_i * (y_i . r); r += (alpha_i - beta) * s_i`.
  Returns `-r` (descent direction).
- REQ-6: `update_history(s, y)` enforces the curvature condition
  `s.y > 1e-30` (skip otherwise) and stores using a `VecDeque` with
  O(1) `pop_front` eviction when `history_size` is reached (CL-1125).
- REQ-7: `step_with_closure(closure)` runs the Strong Wolfe line search
  (when `config.line_search_fn == Some(StrongWolfe)`) by repeatedly
  invoking the closure at candidate alpha values until the Armijo +
  strong-curvature Wolfe conditions are satisfied. Mirrors upstream's
  `step(closure)` API (`torch/optim/lbfgs.py:326`).
- REQ-8: `step()` (the trait method, NO closure) — works only when
  `config.line_search_fn.is_none()`; if line search is configured, it
  returns `FerrotorchError::InvalidArgument` instructing the caller to
  use `step_with_closure`. Divergence from upstream `step(closure=None)`
  uniform API tracked by #1469.
- REQ-9: `state_dict`/`load_state_dict` round-trip the full state:
  meta `n_iter` and `history_len`; per-curvature-pair `s`/`y`/`rho`;
  `prev_flat_params`/`prev_flat_grad` if present. All tensors are
  downloaded to CPU + cast to `f64` for a dtype-agnostic wire format.
- REQ-10: CL-1105 Pattern B — every tensor stays on the parameter's
  device throughout. `dot_tensor` runs `sum(a * b)` on-device and only
  downloads the scalar; `inf_norm` is the one remaining `data_vec()`
  round-trip (a documented temporary; see the inline comment at
  `lbfgs.rs` — lift when GPU max-abs-reduction lands).

## Acceptance Criteria

- [x] AC-1: `LbfgsConfig::default()` returns the documented defaults.
- [x] AC-2: Quadratic convergence: `f(x) = x^2` from `x = 5.0` reaches
  `|x| < 1e-3` within 100 steps (`test_lbfgs_quadratic_convergence`).
- [x] AC-3: Multi-dimensional quadratic convergence
  (`test_lbfgs_multidim_quadratic`).
- [x] AC-4: Rosenbrock convergence from `(-1, 1)` reaches both
  coordinates within `0.1` of `(1, 1)` within 8000 steps
  (`test_lbfgs_rosenbrock_convergence`).
- [x] AC-5: `zero_grad()` clears gradients (`test_lbfgs_zero_grad`).
- [x] AC-6: `state_dict` round-trip preserves `n_iter`, `history_len`,
  and every `(s, y, rho)` triplet plus `prev_flat_params/grad`
  exactly (`test_lbfgs_state_dict_roundtrip`).
- [x] AC-7: `step()` with `line_search_fn = Some(StrongWolfe)` returns
  `FerrotorchError::InvalidArgument` (documented divergence; tracked
  by #1469).
- [ ] AC-8: Per-group LR support and uniform `step(closure)` API
  matching upstream. Blocked by #1469.

## Architecture

### State layout

`LbfgsState<T>` holds three `VecDeque` history buffers
(`s_history`, `y_history`, `rho_history`), `prev_flat_params:
Option<Tensor<T>>`, `prev_flat_grad: Option<Tensor<T>>`, and `n_iter`.
`s` and `y` are 1-D device-resident tensors; `rho` is `f64` (kept in
scalar precision so the two-loop recursion's `alpha`/`beta` arithmetic
does not lose precision when `T = f32`).

`VecDeque` (CL-1125) gives O(1) `pop_front` eviction when
`history_size` is reached — the prior `Vec::remove(0)` was
O(history_size) per call.

### `gather_params` / `gather_grads` / `scatter_params`

`gather_params` returns `(flat: Tensor<T>, shapes: Vec<Vec<usize>>)`
where `flat = cat(flatten(p_i.contiguous()))` and `shapes` records
the per-parameter shapes for later unflatten. `gather_grads` does the
same on `.grad()`, treating `None` as device-resident zeros, and
negates if `maximize` is set.

`scatter_params(flat, &shapes)` walks the per-parameter slot, narrows
`flat[offset .. offset + numel]`, reshapes back to the original shape,
and `unsafe { update_storage(storage) }`s into the parameter. The
SAFETY block at `lbfgs.rs` documents the four invariants.

### Two-loop recursion

`two_loop_recursion(grad: &Tensor<T>) -> Tensor<T>` follows Nocedal &
Wright Algorithm 7.4 verbatim. Scalars (`alpha[i]`, `gamma`, `beta`)
are kept as `f64` and downloaded once per iteration via `dot_tensor`;
the heavy element-wise work stays device-resident.

When the history is empty (`m == 0`), `two_loop_recursion` falls back
to `-grad` (steepest descent). This matches upstream's first-iteration
behavior.

### Strong Wolfe line search

`strong_wolfe_search(f0, g0_dot_d, max_evals, eval_fn)` implements
Algorithm 3.5 (bracket-then-zoom), with `zoom` (Algorithm 3.6) handling
the bisection. Constants `WOLFE_C1 = 1e-4`, `WOLFE_C2 = 0.9`,
`WOLFE_ALPHA_MAX = 50.0` match the canonical quasi-Newton tunings.

The line search is engaged only via `step_with_closure(closure)`, which
sets parameters to `x0 + alpha * direction`, runs the closure (which
re-evaluates forward + backward to populate `.grad`), and returns
`(loss, g_dot_d)`. The `Optimizer<T>::step` method does NOT accept a
closure (Rust trait), so when line search is configured it errors out
with `InvalidArgument`.

### Non-test production consumers

`ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` re-exports
`Lbfgs`, `LbfgsConfig`, and `LineSearchFn` as
`ferrotorch::optim::{Lbfgs, LbfgsConfig, LineSearchFn}`.

## Parity contract

`parity_ops = []`. L-BFGS parity is asserted via the unit-test gauntlet
including the Rosenbrock convergence lock (one of the canonical hard
tests for any quasi-Newton). The major divergence from upstream is the
split between `step` (no closure, fixed-step only) and
`step_with_closure` (closure, line-search-capable) — upstream uses
`step(closure=None)` uniformly. Tracked by #1469.

Edge cases the code owns:

- **Empty curvature history** — first step uses `-grad` (steepest
  descent) regardless of `line_search_fn`.
- **`tolerance_grad` met** — `step` short-circuits with `Ok(())`
  without updating any state. Mirrors upstream's
  `if flat_grad.abs().max() <= tolerance_grad: return`.
- **`s.y <= 1e-30`** — `update_history` skips the pair (curvature
  condition violated). Matches upstream.
- **`history_size` reached** — oldest pair evicted via `pop_front` (CL-1125).
- **`maximize == true`** — gradients negated in `gather_grads`, so the
  search direction maximizes instead of minimizes.
- **`y.y < 1e-30`** — gamma defaults to `1.0` (no Hessian scaling) to
  avoid division by near-zero.

## Verification

Tests in `mod tests` of `lbfgs.rs`:

- `test_lbfgs_quadratic_convergence`
- `test_lbfgs_multidim_quadratic`
- `test_lbfgs_rosenbrock_convergence` (hard convergence lock)
- `test_lbfgs_zero_grad`
- `test_lbfgs_state_dict_roundtrip`
- (plus additional helpers further in the test module)

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib lbfgs:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LbfgsConfig` at `ferrotorch-optim/src/lbfgs.rs` mirroring `torch/optim/lbfgs.py:240-280`; non-test consumer: `ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` re-export. |
| REQ-2 | SHIPPED | impl: `pub enum LineSearchFn { StrongWolfe }` at `ferrotorch-optim/src/lbfgs.rs` mirroring upstream's `line_search_fn="strong_wolfe"` kwarg; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-3 | SHIPPED | impl: `pub struct Lbfgs<T>` at `ferrotorch-optim/src/lbfgs.rs` + `LbfgsState<T>` at `ferrotorch-optim/src/lbfgs.rs` + `impl<T: Float> Optimizer<T>` at `ferrotorch-optim/src/lbfgs.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-4 | SHIPPED | impl: `gather_params` at `ferrotorch-optim/src/lbfgs.rs` + `gather_grads` at `ferrotorch-optim/src/lbfgs.rs` + `scatter_params` at `ferrotorch-optim/src/lbfgs.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-5 | SHIPPED | impl: `two_loop_recursion` at `ferrotorch-optim/src/lbfgs.rs` mirroring Nocedal & Wright Algorithm 7.4; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-6 | SHIPPED | impl: `update_history` at `ferrotorch-optim/src/lbfgs.rs` with curvature-condition gate + VecDeque eviction; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-7 | SHIPPED | impl: `strong_wolfe_search` at `ferrotorch-optim/src/lbfgs.rs` + `zoom` at `ferrotorch-optim/src/lbfgs.rs` + `step_with_closure` at `ferrotorch-optim/src/lbfgs.rs` mirroring Nocedal & Wright Algs 3.5/3.6; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-8 | SHIPPED | impl: `Optimizer<T>::step` at `ferrotorch-optim/src/lbfgs.rs` with `InvalidArgument` early-return when `line_search_fn.is_some()`; non-test consumer: `ferrotorch/src/lib.rs` re-export. Documented divergence from upstream uniform `step(closure)` API tracked by #1469. |
| REQ-9 | SHIPPED | impl: `state_dict` at `ferrotorch-optim/src/lbfgs.rs` + `load_state_dict` at `ferrotorch-optim/src/lbfgs.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-10 | SHIPPED | impl: device-resident step body at `ferrotorch-optim/src/lbfgs.rs` composed via `gather_params`/`gather_grads`/`two_loop_recursion`/`scatter_params`; non-test consumer: `ferrotorch/src/lib.rs` re-export. The remaining `data_vec()` in `inf_norm` (`ferrotorch-optim/src/lbfgs.rs`) is documented temporary; lift when GPU max-abs reduction lands. |
