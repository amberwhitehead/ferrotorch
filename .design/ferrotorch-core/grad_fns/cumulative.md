# Cumulative grad_fns

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/native/ReduceOps.cpp
  - torch/_torch_docs.py
  - tools/autograd/derivatives.yaml
-->

## Summary

`ferrotorch-core/src/grad_fns/cumulative.rs` is the autograd-tracking
wrapper layer for the five scan ops PyTorch declares in
`aten/src/ATen/native/ReduceOps.cpp`: `cumsum`, `cumprod`, `cummax`,
`cummin`, and `logcumsumexp`. Each `pub fn` here pairs a forward call into
`ferrotorch_core::ops::cumulative` (the non-autograd kernel layer) with a
`*Backward` `GradFn` struct that records the VJP per
`tools/autograd/derivatives.yaml:521-539`. Negative-`dim` normalization is
delegated to `crate::shape::normalize_axis`. The file is 914 LOC, of which
~342 LOC are production code and ~572 LOC are `#[cfg(test)]` (forward and
backward characterization tests including numerical-gradient checks).

## Requirements

- REQ-1: `cumsum(input, dim)` — forward `out[..., i, ...] = sum(input[...,
  0..=i, ...])` along `dim` with negative-`dim` normalization and autograd.
  Mirrors `TORCH_IMPL_FUNC(cumsum_out)` at
  `aten/src/ATen/native/ReduceOps.cpp:511-517` (which dispatches via
  `cumsum_stub` declared at `:460`). The VJP is `grad_input =
  reverse_cumsum(grad_output, dim)` per
  `tools/autograd/derivatives.yaml:529-531
  - name: cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
    self: cumsum_backward(grad.to(self.scalar_type()), dim)` — and
  `cumsum_backward` in PyTorch is the `flip→cumsum→flip` upper-triangular
  multiplication (`ReduceOps.cpp:527-529
  static Tensor reversed_cumsum(const Tensor& w, int64_t dim) { return
  w.flip(dim).cumsum(dim).flip(dim); }`). ferrotorch implements this as
  `CumsumBackward` at `cumulative.rs:51-64` calling
  `crate::ops::cumulative::reverse_cumsum`. **Diverges from upstream**:
  ferrotorch does NOT accept the `dtype=None` kwarg (PyTorch's
  `cumsum(Tensor self, int dim, *, ScalarType? dtype=None)`); the
  upstream-only dtype-promotion branch at `ReduceOps.cpp:267
  out_dtype = dtype.value_or(...)` is unreachable. The backward path at
  `cumulative.rs:77` calls `reverse_cumsum` on CPU
  logical views; CUDA gradients route through the device-resident
  `cumsum_backward_cuda` path at `cumulative.rs:69-72`. Non-test
  production consumer: `Tensor::cumsum_t(&self, dim: i64)` at
  `cumsum_t in ferrotorch-core/src/methods.rs` (closed by #1232 — chainable
  method-style PyTorch-API surface delegating to `cumsum`).

- REQ-2: `cumprod(input, dim)` — forward `out[..., i, ...] = prod(input[...,
  0..=i, ...])` along `dim` with negative-`dim` normalization and autograd.
  Mirrors `TORCH_IMPL_FUNC(cumprod_out)` at `ReduceOps.cpp:519-525` and the
  `TORCH_META_FUNC(cumprod)` meta at `:276-279`. The VJP per
  `tools/autograd/derivatives.yaml:509-527
  - name: cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
    self: cumprod_backward(grad.to(self.scalar_type()), self, dim, result)`
  routes through `cumprod_backward` at `ReduceOps.cpp:531-790`, which splits
  on whether the input contains zeros:
  * No zeros (fast path, `ReduceOps.cpp:640-642`):
    `grad_input = reverse_cumsum(output * grad, dim) / input`.
  * Zeros present, no second-order grad (`ReduceOps.cpp:654-724`): the
    `cumsum(input == 0)` mask-gymnastics path with three cases by
    `k<z1 / k==z1 / k>z1`.
  * Zeros present with GradMode::is_enabled (`ReduceOps.cpp:725-789`):
    the `O(n^2)` brute-force `prods_until_k * prods_from_k_plus_1`
    fallback that supports second-order grad.

  ferrotorch implements this as `CumprodBackward` at `cumulative.rs:283-388`
  with a same-shape two-path split at `cumulative.rs:290-378` (zeros-vs-no-zeros):
  * Fast path (`cumulative.rs:355-373`, no zeros): exactly matches upstream's
    `reverse_cumsum(go * out, dim) / input` formula, but unrolled into the
    `(outer, dim_size, inner)` triple-loop instead of dispatching the
    tensor-vectorized version.
  * Slow path (`cumulative.rs:328-354`, zeros present): an **O(n^3)** brute-force
    triple-loop (`partial = prod_{kk in 0..=j, kk != i} input[kk]` inner
    loop) that **does not** mirror upstream's two-stage masked-fill
    composite-compliance algorithm. Numerically correct (the
    `test_cumprod_backward_with_zero` characterization test at
    `cumulative.rs:1418` confirms 1/8/0 expected gradients), but slower
    and not second-order-differentiable. CUDA gradients use the
    device-resident `cumprod_backward_cuda` path at `cumulative.rs:300-308`.
    **Diverges from upstream**: does NOT accept the
    `dtype=None` kwarg. Non-test production consumer:
    `Tensor::cumprod_t(&self, dim: i64)` at
    `cumprod_t in ferrotorch-core/src/methods.rs` (closed by #1232).

- REQ-3: `cummax(input, dim)` — forward returns a `(values, indices)` tuple
  where `values[..., i, ...] = max(input[..., 0..=i, ...])` and
  `indices[..., i, ...]` is the position along `dim` where the running max
  was attained. Mirrors `Tensor cummax(const Tensor& self, int64_t dim)` at
  `ReduceOps.cpp:860-865` dispatching via `cummax_out` at `:836-858` →
  `_cummax_helper` → `cummax_helper_cpu` at `:828-834` → the templated
  `cummax_cummin_helper<T1, T2, std::greater_equal<scalar_t>>` at
  `:811-826`. Upstream's tie-break operator is `std::greater_equal` — on
  equal values the LATER index wins. NaN propagation: once a NaN appears
  in the prefix, every subsequent `values[..., j, ...]` is NaN and the
  running `indices[..., j, ...]` pin to the first-NaN position (upstream
  predicate `isnan_(curr_elem) || (!isnan_(out) && op(curr_elem, out))`
  at `:819`). Per `tools/autograd/derivatives.yaml:533-535
  - name: cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)
    self: cummaxmin_backward(grad, self, indices, dim)`, cummax is
  **differentiable** upstream via `cummaxmin_backward` at
  `ReduceOps.cpp:906-918` which does `result.scatter_add_(dim, indices,
  grad)` — gradient flows back to the input positions that won the running
  max.

  ferrotorch implements forward via `cummax` at `cumulative.rs:524`,
  delegating to `crate::ops::cumulative::cummax_forward` (CPU loop at
  `ops/cumulative.rs:245-294` with `>=` tie-break + NaN-poison predicate,
  GPU `cummax_f32`/`cummax_f64` kernels at `ops/cumulative.rs:200-240`).
  Autograd: `CummaxBackward in cumulative.rs` saves both the PyTorch-style
  `indices_tensor: IntTensor<i64>` and a CPU/scalar `indices: Vec<usize>`
  host cache, plus `input_shape` and `dim`. CUDA backward scatters through
  the resident `indices_tensor`; CPU backward uses the host cache and rejects
  cache-length mismatches instead of silently computing with stale/missing
  indices. **Non-test production consumer**:
  `Tensor::cummax_t(&self, dim: i64)` at `cummax_t in
  ferrotorch-core/src/methods.rs` exposes the PyTorch-style method surface
  and delegates directly to this wrapper.

- REQ-4: `cummin(input, dim)` — forward returns a `(values, indices)` tuple
  symmetric to cummax. Mirrors `Tensor cummin(...)` at
  `ReduceOps.cpp:899-904` → `cummin_helper_cpu` at `:867-873` →
  `cummax_cummin_helper<T1, T2, std::less_equal<scalar_t>>`. Upstream's
  tie-break operator is `std::less_equal` — on equal values the LATER
  index wins. NaN propagation: identical to cummax (same templated helper
  at `:811-826`). Backward per `derivatives.yaml:537-539
  - name: cummin(Tensor self, int dim) -> (Tensor values, Tensor indices)
    self: cummaxmin_backward(grad, self, indices, dim)` is the same
  `scatter_add_(dim, indices, grad)` VJP.

  ferrotorch implements forward via `cummin` at `cumulative.rs:556`,
  delegating to `crate::ops::cumulative::cummin_forward` (CPU loop at
  `ops/cumulative.rs:341-389` with `<=` tie-break + NaN-poison predicate,
  GPU kernels at `ops/cumulative.rs:316-348`). Autograd: `CumminBackward`
  at `CumminBackward in cumulative.rs` shares `cummaxmin_backward_impl` with
  `CummaxBackward`, differing only in `name()`. **Non-test production
  consumer**: `Tensor::cummin_t(&self, dim: i64)` at `cummin_t in
  ferrotorch-core/src/methods.rs` exposes the PyTorch-style method surface
  and delegates directly to this wrapper.

- REQ-5: `logcumsumexp(input, dim)` — numerically stable
  `out[..., i, ...] = log(sum(exp(input[..., 0..=i, ...])))` with autograd.
  Mirrors `Tensor logcumsumexp(const Tensor& self, int64_t dim)` at
  `ReduceOps.cpp:475-482` dispatching via `_logcumsumexp_cpu` at
  `:465-468` → `logcumsumexp_stub` at `:471`. The VJP per
  `tools/autograd/derivatives.yaml:521-523
  - name: logcumsumexp(Tensor self, int dim) -> Tensor
    self: logcumsumexp_backward(grad, self, result, dim)` factors as
  `grad_input[i] = exp(input[i]) * reverse_cumsum(grad_output * exp(-output))`
  algebraically, but PyTorch's real implementation splits positive and
  negative upstream gradients in log-space for stability. ferrotorch implements
  forward via `logcumsumexp` at `cumulative.rs:995-1018`, delegating to
  `ops::cumulative::logcumsumexp_forward` (CPU two-pass max-rescaling
  algorithm at `ops/cumulative.rs:378-410`, GPU kernels at
  `ops/cumulative.rs:360-365`). Backward is
  `LogcumsumexpBackward` at `cumulative.rs:882-927`, matching the
  signed log-space `FunctionsManual.cpp::logcumsumexp_backward`
  implementation. The numerical-stability invariant (large
  inputs ~1000.0 stay finite) is covered by
  `test_logcumsumexp_numerical_stability` at `cumulative.rs:1565` and
  the gradient is validated against finite differences in
  `test_logcumsumexp_backward_1d` at `cumulative.rs:1589`.
  CUDA gradients use `logcumsumexp_backward_cuda` from
  `cumulative.rs:905-907`; they do not round-trip through CPU. Non-test
  production consumer: `Tensor::logcumsumexp_t(&self, dim: i64)` at
  `logcumsumexp_t in ferrotorch-core/src/methods.rs` (closed by #1232).

- REQ-6: Shared `dim` normalization — every `pub fn` in the module calls
  `crate::shape::normalize_axis(dim as isize, input.ndim())?` to convert
  negative `dim` to non-negative and to error on out-of-range `dim`.
  Mirrors `maybe_wrap_dim(dim, self.dim())` at `ReduceOps.cpp:506` (in
  `impl_func_cum_ops`) and at `:622, :851, :890`. The error contract
  matches: ferrotorch returns `FerrotorchError::*` while upstream raises a
  Python `IndexError`/`RuntimeError`; the divergence is the Result-vs-raise
  vocabulary substitution permitted by R-DEV-4. Implemented at
  `cumulative.rs:109, :404, :675, :716, :1004` (one call per public op).
  The `dim`-out-of-bounds and `dim`-for-scalar error cases are exercised
  by `test_cumsum_scalar_passthrough`, `test_cumprod_scalar_passthrough`,
  `test_cummax_scalar_passthrough`, `test_cummin_scalar_passthrough`,
  `test_logcumsumexp_scalar_passthrough` at `cumulative.rs:1715`,
  `test_cumsum_scalar_dim_out_of_range` at `cumulative.rs:1727`,
  `test_cummax_scalar_dim_out_of_range` at `cumulative.rs:1734`, and
  `test_cumsum_dim_out_of_bounds` at `cumulative.rs:1814`.

- REQ-7: `reverse_cumsum` helper — `CumsumBackward::backward` calls
  `crate::ops::cumulative::reverse_cumsum` to compute the upper-triangular
  multiplication. Mirrors `static Tensor reversed_cumsum(const Tensor& w,
  int64_t dim)` at `ReduceOps.cpp:527-529`. Implemented at
  `ops/cumulative.rs:109-133` and used at `cumulative.rs:77`. No
  production consumer outside the cumsum autograd path; this is internal
  scaffolding for REQ-1.

## Acceptance Criteria

- [x] AC-1: `cumsum` parity-sweep at `--seeds 8` returns
  `[cumsum] 32/32 passed (0 skipped, 0 failed)` (smoke grep count = 1).
  Post-#1233 the 0-D scalar fast path lands at
  `ferrotorch-core/src/grad_fns/cumulative.rs:105-107` (`pub fn cumsum`
  early-out → `cumulative_scalar_identity`), so the 8 op_db samples
  that pass 0-D inputs now copy the scalar through unchanged, mirroring
  upstream's `impl_func_cum_ops` 0-D branch at `ReduceOps.cpp:501-504`.
- [x] AC-2: `cumprod` parity-sweep at `--seeds 8` returns
  `[cumprod] 80/80 passed (0 skipped, 0 failed)` (smoke grep count = 1).
  Same 0-D fast-path resolution as AC-1, dispatched via
  `cumulative.rs:354-357 pub fn cumprod` → `cumulative_scalar_identity`.
- [x] AC-3: `cummax` parity-sweep at `--seeds 8` returns
  `[cummax] 24/24 passed (0 skipped, 0 failed)` (smoke grep count = 1).
  0-D fast path at `cumulative.rs:524 pub fn cummax` →
  `cumextreme_scalar_identity` returns
  `CumExtremeResult { values: scalar, indices: vec![0] }`. The dispatch
  arm at `tools/parity-sweep/runner/src/main.rs:637` selects only the `values`
  half (Option A from #1230). Post-#1231 the values tensor now carries
  a `CummaxBackward` grad-fn (`CummaxBackward in cumulative.rs`) and the saved indices
  follow upstream `std::greater_equal` tie-break + NaN poisoning at
  `ops/cumulative.rs:251-282`.
- [x] AC-4: `cummin` parity-sweep at `--seeds 8` returns
  `[cummin] 24/24 passed (0 skipped, 0 failed)` (smoke grep count = 1).
  Same 0-D resolution as AC-3 via `cumulative.rs:712 pub fn cummin`
  → `cumextreme_scalar_identity`. Post-#1231 the values tensor carries
  a `CumminBackward` grad-fn at `cumulative.rs:501` sharing the
  scatter-add VJP with cummax via `cummaxmin_backward_impl` at
  `cumulative.rs:540`.
- [x] AC-5: `logcumsumexp` parity-sweep at `--seeds 8` returns
  `[logcumsumexp] 48/48 passed (0 skipped, 0 failed)` (smoke grep
  count = 1). 0-D fast path at `cumulative.rs:995-1018 pub fn
  logcumsumexp` → `cumulative_scalar_identity`. The numerical identity
  is `logcumsumexp(x) = log(exp(x)) = x` on a scalar.
- [x] AC-6: `cargo test -p ferrotorch-core grad_fns::cumulative` passes
  every forward and backward test in `cumulative.rs:753-1569` — covering
  1D / 2D dim=0 / 2D dim=1 / 3D forward shape correctness, negative-dim
  handling, numerical-gradient backward check for cumsum at
  `cumulative.rs:1381` and cumprod at `cumulative.rs:1342`, finite-difference
  backward for logcumsumexp at `cumulative.rs:1146`, zero-input cumprod
  backward at `cumulative.rs:975`, scalar-input passthrough at
  `cumulative.rs:1225-1281`, and dim-out-of-bounds at
  `cumulative in cumulative.rs`.
- [x] AC-7: Negative `dim` produces the same result as the equivalent
  positive `dim` — `test_cumsum_negative_dim in cumulative.rs`
  verifies `cumsum(x, -1) == cumsum(x, 1)` on shape `[2, 3]`.
- [x] AC-8: `requires_grad=false` inputs return a tensor with
  `grad_fn().is_none()` — verified by
  `test_cumsum_no_grad_fn_when_not_requires_grad in cumulative.rs`.
- [x] AC-9: Within a `no_grad` context, the returned tensor has
  `grad_fn().is_none()` even if the input has `requires_grad=true` —
  verified for cumsum/cumprod/logcumsumexp at `cumulative.rs:900` (cumsum),
  `cumulative.rs:1006` (cumprod), and `cumulative.rs:1636` (logcumsumexp).
- [x] AC-10: cummax/cummin backward attaches the appropriate
  `CummaxBackward` / `CumminBackward` grad-fn when `input.requires_grad()`,
  routing grad through the saved indices via the `scatter_add` VJP per
  `derivatives.yaml:533-539`. Implemented at `cumulative.rs:413`
  (CummaxBackward) and `cumulative.rs:447` (CumminBackward) sharing
  `cummaxmin_backward_impl in cumulative.rs`. Tests:
  `test_cummax_backward_monotonic`, `test_cummax_backward_tie`, and
  `test_cummin_backward_tie` verify the scatter-add VJP against
  upstream-traced gradients live-verified 2026-05-25 with torch 2.11.0.
- [x] AC-11: cummax/cummin tie-breaking matches upstream — on equal
  values the LATER index wins (upstream uses `std::greater_equal` /
  `std::less_equal`). Implemented at `ops/cumulative.rs:251-282`
  (cummax, `>=`) and `:315-345` (cummin, `<=`), with the same
  `isnan(curr) || (!isnan(cur) && op(curr, cur))` update predicate as
  upstream `cummax_cummin_helper` at `ReduceOps.cpp:819`. Verified by
  `test_cummin_1d` (indices `[0, 1, 1, 3, 3]` for input
  `[3, 1, 4, 1, 5]`) and `test_cummax_backward_tie`
  (indices `[0, 1, 2, 3]` for input `[1, 2, 2, 3]`).

## Architecture

### Layer split (`ops::cumulative` vs `grad_fns::cumulative`)

The file under design is the autograd layer; the kernel layer lives at
`ferrotorch-core/src/ops/cumulative.rs` (414 LOC: forward kernels for
all five ops plus the `reverse_cumsum` helper plus the
`CumExtremeResult { values, indices }` struct). The split mirrors
PyTorch's `_cummax_helper` / `_logcumsumexp` `_<op>` underscore-prefixed
private dispatchers (`ReduceOps.cpp:465-491, 828-834, 867-873`) vs the
user-facing `cummax` / `logcumsumexp` namespace functions.

### REQ-1 `cumsum` (cumulative.rs lines 53-122)

`CumsumBackward<T>` (`cumulative.rs:53-57`) saves `input: Tensor<T>` and
`dim: usize`. Only the dim is materially used (it's a scalar field saved
to avoid re-normalizing on backward); the `input` field is saved so
`GradFn::inputs(&self)` at `:89-91` returns the right reference for the
autograd-graph topological walk. `backward` at `:59-91` materializes
`grad_output.data()` on CPU (rejecting CUDA at `:64-69`), calls
`reverse_cumsum`, wraps the result back into a tensor with
`requires_grad=false`, and returns `vec![Some(grad_input)]`. `pub fn
cumsum` at `:105-122` normalizes `dim`, calls `cumsum_forward`, and (when
`is_grad_enabled() && input.requires_grad()`) attaches a
`CumsumBackward` node via `Tensor::from_operation`. The non-`grad`
fast-exit at `:119-120` returns `result` unchanged.

### REQ-2 `cumprod` (cumulative.rs lines 283-418)

`CumprodBackward<T>` at `cumulative.rs:283-287` saves `input`, `output`, and `dim`.
Saving the output is the upstream-aligned optimization for the no-zeros
fast path (`output[j] / input[i]` requires both). The backward at
`:291-386` is the two-path split described in REQ-2 above. `pub fn
cumprod` at `:400-418` follows the same forward → optional grad-fn
attach pattern as cumsum.

### REQ-3/4 `cummax` / `cummin` (cumulative.rs lines 460-752)

`CummaxBackward` at `cumulative.rs:460-491` and `CumminBackward` at
`:501-527` share `cummaxmin_backward_impl` at `:540-572`, which implements
upstream's `scatter_add(zeros, dim, indices, grad)` VJP. Public `cummax`
at `:671-702` and `cummin` at `:712-752` attach those grad functions when
tracking is enabled. The non-test consumers are the `EinopsReduction::Max`
and `EinopsReduction::Min` arms in `einops.rs`.

### REQ-5 `logcumsumexp` (cumulative.rs lines 882-1024)

`LogcumsumexpBackward<T>` at `cumulative.rs:882-886` saves `input`, `output`, and
`dim`. Backward at `cumulative.rs:889-927` delegates to the signed
log-space CPU/CUDA implementations rather than the unstable algebraic
`exp(input) * reverse_cumsum(grad * exp(-output))` form.

The formula docstring at `cumulative.rs:866-880` self-corrects mid-comment from a
naive `exp(input - output)` form to the correct
`exp(input) * reverse_cumsum(go * exp(-output))` form — preserved
verbatim because it documents the derivation step the implementer
walked through. `pub fn logcumsumexp` at `cumulative.rs:995-1018` matches the
cumsum/cumprod scaffold.

### REQ-6 dim normalization (call sites only)

The `normalize_axis(dim as isize, ndim)` calls at `:109, :404, :675, :716,
:1004` are the upstream `maybe_wrap_dim` analog (R-DEV-2: API-shape
match). The `isize` cast widens `i64` → `isize` which on every supported
host platform (64-bit) is lossless; on a 32-bit host the cast would
truncate but ferrotorch does not support 32-bit hosts.

### REQ-7 `reverse_cumsum` helper

Implemented in the kernel layer at `ops/cumulative.rs:109-133` and
re-imported at `cumulative.rs:32-35`. Consumers: a `reverse_cumsum`
call from `CumsumBackward::backward` at `cumulative.rs:77`.
The `cumprod` backward's
fast path inlines the equivalent reverse-cumsum-then-divide as a
single-loop `rev_acc` accumulator at `cumulative.rs` rather than calling
`reverse_cumsum` — minor code duplication that is intentional because
the per-element division by `in_data[idx]` interleaves with the
reverse-cumsum accumulation.

## Parity contract

| Op | Upstream entry | Backward formula source | Edge cases mirrored |
|---|---|---|---|
| `cumsum` | `ReduceOps.cpp:511` `TORCH_IMPL_FUNC(cumsum_out)` | `derivatives.yaml:529-531` (`cumsum_backward` = `reversed_cumsum`) | Empty input along dim: ferrotorch returns shape-preserving empty (mirrors `impl_func_cum_ops` at `ReduceOps.cpp:503-504 result.zero_()`). Scalar input: errors (mirrors upstream's `self.dim() == 0` → `result.fill_(self)` branch by erroring instead — ferrotorch's `normalize_axis` errors when `ndim == 0`, so scalars cannot be cumsum'd). NaN / Inf: float arithmetic propagates naturally; no special-case handling. Non-contiguous: forward iterates by computed flat indices `base + i * inner` so stride doesn't matter for the CPU path, but the GPU path uses `gpu_handle()` which requires contiguous storage — non-contiguous CUDA inputs trigger `ops::cumulative` `is_cuda` paths that may need materialization (not yet audited; not blocking here). |
| `cumprod` | `ReduceOps.cpp:519` `TORCH_IMPL_FUNC(cumprod_out)` | `derivatives.yaml:509-527` (`cumprod_backward` = zeros-aware reverse-cumsum-divide) | Zeros in input: ferrotorch slow-path O(n^3) brute force matches upstream's masked-fill composite-compliance path numerically but not algorithmically. Test at `cumulative.rs:577-597` verifies `cumprod([2, 0, 3]).backward() == [1, 8, 0]`. NaN / Inf: propagates naturally; `0 * inf = NaN` will materialize through. Non-contiguous: same caveat as cumsum. Second-order grad (`grad_of_grad`): unsupported — the slow path is O(n^3) and not second-order-differentiable. |
| `cummax` | `ReduceOps.cpp:860` `Tensor cummax(...)` | `derivatives.yaml:533-535` (`cummaxmin_backward` = `scatter_add` through indices) | Returns `CumExtremeResult { values, indices_tensor }` (Rust analog of PyTorch's `(values, indices)` return). Tie-breaking matches upstream `std::greater_equal` (later tie wins). NaN-poisoning matches upstream's `isnan_(curr_elem)` branch at `ReduceOps.cpp:819`. Differentiability matches upstream via `cummaxmin_backward_impl`. |
| `cummin` | `ReduceOps.cpp:899` `Tensor cummin(...)` | `derivatives.yaml:537-539` (same `cummaxmin_backward`) | Symmetric to cummax: returns `CumExtremeResult`, uses later-tie `std::less_equal` semantics, NaN-poisons like upstream, and scatters gradients through saved indices. |
| `logcumsumexp` | `ReduceOps.cpp:475` `Tensor logcumsumexp(...)` | `derivatives.yaml:521-523` (`logcumsumexp_backward` = `exp(input) * reverse_cumsum(grad * exp(-output))`) | Numerical stability: ferrotorch's two-pass running-max algorithm at `exp in ops/cumulative.rs` ensures inputs at scale ~1000 stay finite, verified by `test_logcumsumexp_numerical_stability in cumulative.rs`. NaN / Inf: `(-inf).exp() == 0` and `0.ln() == -inf` give the upstream-aligned `logcumsumexp([-inf, x]) == [-inf, x]` behavior. Empty input: errors (via `normalize_axis` scalar check). |

Parity-sweep audit reference: all five op entries are **MISSING** from
`tools/parity-sweep/parity_audit.json` as of this writeup. Adding them
is part of blocker #1230.

## Verification

### Existing unit tests (all passing)

Located at `ferrotorch-core/src/grad_fns/cumulative.rs:753-1569` (the
`#[cfg(test)] mod tests` block). Key tests:

- `test_cumsum_1d` (`test_cumsum_1d in ferrotorch-core/src/grad_fns/cumulative.rs`), `test_cumsum_2d_dim0` (`test_cumsum_2d_dim0 in ferrotorch-core/src/grad_fns/cumulative.rs`),
  `test_cumsum_2d_dim1` (`test_cumsum_2d_dim1 in ferrotorch-core/src/grad_fns/cumulative.rs`), `test_cumsum_negative_dim`
  (`test_cumsum_negative_dim in ferrotorch-core/src/grad_fns/cumulative.rs`), `test_cumsum_3d` (`test_cumsum_3d in ferrotorch-core/src/grad_fns/cumulative.rs`)
- `test_cumsum_backward_1d` (`test_cumsum_backward_1d in ferrotorch-core/src/grad_fns/cumulative.rs`), `test_cumsum_backward_2d_dim0`
  (`test_cumsum_backward_2d_dim0 in ferrotorch-core/src/grad_fns/cumulative.rs`), `test_cumsum_backward_numerical` (`test_cumsum_backward_numerical in ferrotorch-core/src/grad_fns/cumulative.rs`)
- `test_cumsum_has_grad_fn` (`test_cumsum_has_grad_fn in ferrotorch-core/src/grad_fns/cumulative.rs`),
  `test_cumsum_no_grad_fn_when_not_requires_grad` (`test_cumsum_no_grad_fn_when_not_requires_grad in ferrotorch-core/src/grad_fns/cumulative.rs`),
  `test_cumsum_no_grad_fn_in_no_grad_context` (`test_cumsum_no_grad_fn_in_no_grad_context in ferrotorch-core/src/grad_fns/cumulative.rs`)
- `test_cumprod_1d` (`test_cumprod_1d in ferrotorch-core/src/grad_fns/cumulative.rs`), `test_cumprod_2d_dim0` (`test_cumprod_2d_dim0 in ferrotorch-core/src/grad_fns/cumulative.rs`),
  `test_cumprod_2d_dim1` (`test_cumprod_2d_dim1 in ferrotorch-core/src/grad_fns/cumulative.rs`)
- `test_cumprod_backward_1d` (`test_cumprod_backward_1d in ferrotorch-core/src/grad_fns/cumulative.rs`),
  `test_cumprod_backward_with_zero` (`test_cumprod_backward_with_zero in ferrotorch-core/src/grad_fns/cumulative.rs`),
  `test_cumprod_backward_numerical` (`test_cumprod_backward_numerical in ferrotorch-core/src/grad_fns/cumulative.rs`)
- `test_cummax_1d` (`test_cummax_1d in ferrotorch-core/src/grad_fns/cumulative.rs`), `test_cummax_2d_dim1` (`test_cummax_2d_dim1 in ferrotorch-core/src/grad_fns/cumulative.rs`)
- `test_cummin_1d` (`test_cummin_1d in ferrotorch-core/src/grad_fns/cumulative.rs`), `test_cummin_2d_dim0` (`test_cummin_2d_dim0 in ferrotorch-core/src/grad_fns/cumulative.rs`)
- `test_logcumsumexp_1d` (`test_logcumsumexp_1d in ferrotorch-core/src/grad_fns/cumulative.rs`), `test_logcumsumexp_2d_dim1`
  (`test_logcumsumexp_2d_dim1 in ferrotorch-core/src/grad_fns/cumulative.rs`), `test_logcumsumexp_numerical_stability` (`test_logcumsumexp_numerical_stability in ferrotorch-core/src/grad_fns/cumulative.rs`)
- `test_logcumsumexp_backward_1d` (`test_logcumsumexp_backward_1d in ferrotorch-core/src/grad_fns/cumulative.rs`)
- `test_*_scalar_passthrough` (`:1225-1281`), `test_*_scalar_dim_out_of_range`
  (`:1284-1299`), `test_*_scalar_backward_is_identity` (`:1301-1329`),
  `test_cumsum_dim_out_of_bounds` (`test_cumsum_dim_out_of_bounds in ferrotorch-core/src/grad_fns/cumulative.rs`)

### Parity-sweep status

Post-#1233 (this iteration) the autograd-layer 0-D scalar fast path
lands at the `pub fn cumsum / cumprod / cummax / cummin /
logcumsumexp` entry points in `grad_fns/cumulative.rs`, short-circuiting
before `normalize_axis` rejects `ndim==0`. The forward returns a fresh
0-D tensor with the scalar copied; the backward (where applicable)
returns `vec![Some(grad_output.clone())]` as the identity VJP. Mirrors
PyTorch's `impl_func_cum_ops` 0-D branch at
`aten/src/ATen/native/ReduceOps.cpp:501-504 result.fill_(self)`.

Reproducer (`cd /home/doll/ferrotorch`):

```
./target/release/parity-sweep sweep --op cumsum       --seeds 8
  => [cumsum] 32/32 passed (0 skipped, 0 failed)
./target/release/parity-sweep sweep --op cumprod      --seeds 8
  => [cumprod] 80/80 passed (0 skipped, 0 failed)
./target/release/parity-sweep sweep --op cummax       --seeds 8
  => [cummax] 24/24 passed (0 skipped, 0 failed)
./target/release/parity-sweep sweep --op cummin       --seeds 8
  => [cummin] 24/24 passed (0 skipped, 0 failed)
./target/release/parity-sweep sweep --op logcumsumexp --seeds 8
  => [logcumsumexp] 48/48 passed (0 skipped, 0 failed)
```

Smoke grep count (`grep -c "passed (0 skipped, 0 failed)"`) is `1` for
every op. For cummax/cummin the runner dispatch (`runner/src/main.rs:979
"cummax" =>`, `:511 "cummin" =>`) still implements Option A from #1230:
ferrotorch returns only `result.values` and the sweep loop selects
`output[0]` from the oracle's JSON-array tuple — indices-parity remains
tracked under #1231.

Note on the kernel layer: `ferrotorch-core/src/ops/cumulative.rs:49-56
validate_dim` still rejects `ndim==0` for defense-in-depth, but the
autograd layer short-circuits before reaching it on the 0-D path. Direct
callers of the `*_forward` functions (none exist today outside
`grad_fns/cumulative.rs`) would still see the old rejection — a future
non-blocking cleanup could thread the 0-D fast path into the kernel
layer too, requiring authoring `.design/ferrotorch-core/ops/cumulative.md`
first.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (cumsum) | SHIPPED | impl: `cumsum in ferrotorch-core/src/grad_fns/cumulative.rs` + `CumsumBackward in ferrotorch-core/src/grad_fns/cumulative.rs` mirroring `ReduceOps.cpp:511 TORCH_IMPL_FUNC(cumsum_out)` and `derivatives.yaml:529-531`. 0-D scalar fast path at `cumulative in cumulative.rs` (early-out into `cumulative_scalar_identity`) + `CumsumBackward::backward` 0-D fast path at `backward in cumulative.rs` mirror upstream's `impl_func_cum_ops` 0-D branch at `ReduceOps.cpp:501-504` (`result.fill_(self)`). Post-#1233 parity-sweep: `[cumsum] 32/32 passed (0 skipped, 0 failed)`. Non-test production consumer: `Tensor::cumsum_t(&self, dim: i64)` at `cumsum_t in ferrotorch-core/src/methods.rs` mirroring `torch.Tensor.cumsum(dim, dtype=None)` per `torch/_tensor_docs.py:1500-1506` — the public, chainable method-style surface that closes R-DEFER-1 (closed by #1232). |
| REQ-2 (cumprod) | SHIPPED | impl: `cumprod` at `ferrotorch-core/src/grad_fns/cumulative.rs:400` + `CumprodBackward` at `:283` mirroring `ReduceOps.cpp:519 TORCH_IMPL_FUNC(cumprod_out)` and `derivatives.yaml:509-527`; backward zeros-path is O(n^3) brute-force not upstream's composite-compliance masked-fill. 0-D scalar fast path at `cumulative.rs:401-402` + `CumprodBackward::backward` 0-D fast path at `:291-298`. Post-#1233 parity-sweep: `[cumprod] 80/80 passed (0 skipped, 0 failed)`. Non-test production consumer: `Tensor::cumprod_t(&self, dim: i64)` at `cumprod_t in ferrotorch-core/src/methods.rs` mirroring `torch.Tensor.cumprod(dim, dtype=None)` per `torch/_tensor_docs.py:1482-1488` (closed by #1232). |
| REQ-3 (cummax) | SHIPPED | impl: `cummax in ferrotorch-core/src/grad_fns/cumulative.rs` delegating to `cummax in ops/cumulative.rs cummax_forward` mirroring `aten/src/ATen/native/ReduceOps.cpp:860 Tensor cummax(...)`; CPU kernel at `cummax in ops/cumulative.rs` mirrors `cummax_cummin_helper<...std::greater_equal>` at `ReduceOps.cpp:811-826` (NaN-poison predicate `isnan(curr) || (!isnan(cur) && curr >= cur)` matches `:819`; tie-break `>=` matches `:832 std::greater_equal<scalar_t>`). Backward: `CummaxBackward in cumulative.rs` saves `indices_tensor: IntTensor<i64>` plus a CPU/scalar `indices: Vec<usize>` cache, then implements `scatter_add(zeros, dim, indices, grad)` through `cummaxmin_backward_impl in cumulative.rs` mirroring `cummaxmin_backward` at `ReduceOps.cpp:906-918` per `tools/autograd/derivatives.yaml:533-535 self: cummaxmin_backward(grad, self, indices, dim)`. CUDA backward consumes the resident `indices_tensor`; CPU backward requires a full host cache. **Non-test production consumer**: `Tensor::cummax_t(&self, dim: i64)` at `cummax_t in ferrotorch-core/src/methods.rs` delegates to this wrapper and returns `CumExtremeResult<T>`, closing the method-style PyTorch surface. Post-#1231 parity-sweep: `[cummax] 24/24 passed (0 skipped, 0 failed)`. Backward correctness verified by `test_cummax_backward_monotonic`, `test_cummax_backward_tie`, and `test_method_cummax_cummin_t_values_indices_and_backward_ties`. NaN propagation verified by `test_cummax_forward_nan_propagates`. |
| REQ-4 (cummin) | SHIPPED | impl: `cummin in ferrotorch-core/src/grad_fns/cumulative.rs` delegating to `cummin in ops/cumulative.rs cummin_forward` mirroring `aten/src/ATen/native/ReduceOps.cpp:899 Tensor cummin(...)`; CPU kernel at `cummin in ops/cumulative.rs` mirrors `cummax_cummin_helper<...std::less_equal>` at `ReduceOps.cpp:867-873` + `:811-826` (tie-break `<=` matches `:871 std::less_equal<scalar_t>`). Backward: `CumminBackward in cumulative.rs` shares `cummaxmin_backward_impl in cumulative.rs` with `CummaxBackward`, differing only in `name()` — symmetric to upstream's reuse of the same `cummaxmin_backward` C++ function for both ops per `derivatives.yaml:537-539`. CUDA backward consumes the resident `indices_tensor`; CPU backward uses the host `indices` cache. **Non-test production consumer**: `Tensor::cummin_t(&self, dim: i64)` at `cummin_t in ferrotorch-core/src/methods.rs` delegates to this wrapper and returns `CumExtremeResult<T>`. Post-#1231 parity-sweep: `[cummin] 24/24 passed (0 skipped, 0 failed)`. Backward correctness verified by `test_cummin_backward_tie` and `test_method_cummax_cummin_t_values_indices_and_backward_ties`; tie-break verified by updated `test_cummin_1d` (indices `[0, 1, 1, 3, 3]` for `[3, 1, 4, 1, 5]`). |
| REQ-5 (logcumsumexp) | SHIPPED | impl: `logcumsumexp` at `ferrotorch-core/src/grad_fns/cumulative.rs:995` + `LogcumsumexpBackward` at `:882` mirroring `ReduceOps.cpp:475 Tensor logcumsumexp(...)`, `FunctionsManual.cpp::logcumsumexp_backward`, and `derivatives.yaml:521-523`; backward uses the signed log-space implementation rather than the unstable direct `exp(input) * reverse_cumsum(grad * exp(-output))` expression. 0-D scalar fast path at `cumulative.rs:996-1002` (forward) + `cumulative.rs:895-897` (backward — identity VJP since `log(exp(x)) = x`). Numerical stability covered by `test_logcumsumexp_numerical_stability` at `cumulative.rs:1565`. Post-#1233 parity-sweep: `[logcumsumexp] 48/48 passed (0 skipped, 0 failed)`. Non-test production consumer: `Tensor::logcumsumexp_t(&self, dim: i64)` at `logcumsumexp_t in ferrotorch-core/src/methods.rs` mirroring `torch.Tensor.logcumsumexp(dim)` per `torch/_tensor_docs.py:1455-1462` (closed by #1232). |
| REQ-6 (dim normalization) | SHIPPED | impl: `normalize_axis(dim as isize, input.ndim())?` calls at `cumulative.rs:109, :404, :675, :716, :1004` per `crate::shape::normalize_axis` mirroring `maybe_wrap_dim` at `ReduceOps.cpp:506, :622, :851, :890`; production consumer for the normalized result is each of the five `pub fn` bodies themselves (the normalized `dim` is stored into the `*Backward` struct and threaded into `from_storage`); reachable production callers include the `methods.rs` `*_t` surfaces. Tests cover the negative-dim and out-of-range cases. The fn `test_cumsum_negative_dim` is at `cumulative.rs:1261`; the fn `test_cumsum_dim_out_of_bounds` is at `cumulative.rs:1814`. |
| REQ-7 (reverse_cumsum helper) | SHIPPED | impl: `reverse_cumsum` in `ferrotorch-core/src/ops/cumulative.rs` mirroring `static Tensor reversed_cumsum(const Tensor& w, int64_t dim)` at `ReduceOps.cpp:527-529`; non-test production consumer is `CumsumBackward::backward` at `ferrotorch-core/src/grad_fns/cumulative.rs:77`. The helper itself is internal scaffolding; its end-to-end exercise lands through cumsum backward parity. Forward and backward unit tests verify it numerically through the consumer path. The fn `test_cumsum_backward_1d` is at `cumulative.rs:1291`. |
