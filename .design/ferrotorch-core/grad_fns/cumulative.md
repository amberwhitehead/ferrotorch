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
  `CumsumBackward` at `cumulative.rs:35-64` calling
  `crate::ops::cumulative::reverse_cumsum`. **Diverges from upstream**:
  ferrotorch does NOT accept the `dtype=None` kwarg (PyTorch's
  `cumsum(Tensor self, int dim, *, ScalarType? dtype=None)`); the
  upstream-only dtype-promotion branch at `ReduceOps.cpp:267
  out_dtype = dtype.value_or(...)` is unreachable. **Diverges from
  upstream**: `CumsumBackward::backward` at `cumulative.rs:42-46` errors
  with `FerrotorchError::NotImplementedOnCuda` when `grad_output.is_cuda()`
  — backward forces a CPU round trip the upstream avoids. **No non-test
  production consumer** of `cumsum` exists in `ferrotorch-*/src/`; the
  `lib.rs:159` re-export is reachable but unused outside the test suite
  (blocker #1232).

- REQ-2: `cumprod(input, dim)` — forward `out[..., i, ...] = prod(input[...,
  0..=i, ...])` along `dim` with negative-`dim` normalization and autograd.
  Mirrors `TORCH_IMPL_FUNC(cumprod_out)` at `ReduceOps.cpp:519-525` and the
  `TORCH_META_FUNC(cumprod)` meta at `:276-279`. The VJP per
  `tools/autograd/derivatives.yaml:525-527
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

  ferrotorch implements this as `CumprodBackward` at `cumulative.rs:103-194`
  with a same-shape two-path split at `:131-179` (zeros-vs-no-zeros):
  * Fast path (`:161-178`, no zeros): exactly matches upstream's
    `reverse_cumsum(go * out, dim) / input` formula, but unrolled into the
    `(outer, dim_size, inner)` triple-loop instead of dispatching the
    tensor-vectorized version.
  * Slow path (`:142-160`, zeros present): an **O(n^3)** brute-force
    triple-loop (`partial = prod_{kk in 0..=j, kk != i} input[kk]` inner
    loop) that **does not** mirror upstream's two-stage masked-fill
    composite-compliance algorithm. Numerically correct (the
    `test_cumprod_backward_with_zero` characterization test at
    `cumulative.rs:577-597` confirms 1/8/0 expected gradients), but slower
    and not second-order-differentiable. **Diverges from upstream**:
    ferrotorch's cumprod backward errors on CUDA at `cumulative.rs:111-115`
    just like cumsum. **Diverges from upstream**: does NOT accept the
    `dtype=None` kwarg. **No non-test production consumer** of `cumprod`
    exists in `ferrotorch-*/src/` (blocker #1232).

- REQ-3: `cummax(input, dim)` — forward returns a `(values, indices)` tuple
  where `values[..., i, ...] = max(input[..., 0..=i, ...])` and
  `indices[..., i, ...]` is the position along `dim` where the running max
  was attained. Mirrors `Tensor cummax(const Tensor& self, int64_t dim)` at
  `ReduceOps.cpp:860-865` dispatching via `cummax_out` at `:836-858` →
  `_cummax_helper` → `cummax_helper_cpu` at `:828-834` → the templated
  `cummax_cummin_helper<T1, T2, std::greater_equal<scalar_t>>` at
  `:811-826`. Upstream's tie-break operator is `std::greater_equal` — on
  equal values the LATER index wins. Per
  `tools/autograd/derivatives.yaml:533-535
  - name: cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)
    self: cummaxmin_backward(grad, self, indices, dim)`, cummax is
  **differentiable** upstream via `cummaxmin_backward` at
  `ReduceOps.cpp:906-918` which does `result.scatter_add_(dim, indices,
  grad)` — gradient flows back to the input positions that won the running
  max.

  ferrotorch implements forward via `cummax` at `cumulative.rs:230-232`,
  delegating to `crate::ops::cumulative::cummax_forward` (CPU loop at
  `ops/cumulative.rs:240-255`, GPU `cummax_f32`/`cummax_f64` kernels at
  `ops/cumulative.rs:200-228`). **Diverges from upstream**: ferrotorch's
  CPU tie-break uses **strict `>`** at `ops/cumulative.rs:247
  if in_data[idx] > cur_max` — on equal values the EARLIER index wins,
  the opposite of upstream's `std::greater_equal`. Also **Diverges from
  upstream**: `cumulative.rs:226-229` documents "This operation is **not
  differentiable** — the returned values tensor does not carry a gradient
  function". This contradicts `derivatives.yaml:533-535`. Blocker #1231
  tracks adding `CummaxBackward` saving `indices: Vec<usize>` and
  implementing the scatter-add VJP. **Non-test production consumer**:
  `crate::grad_fns::cumulative::cummax` is invoked at
  `ferrotorch-core/src/einops.rs:796` inside `pub fn reduce<T: Float>`
  (the `EinopsReduction::Max` arm, which `cummax`-then-`narrow` selects
  the global max along a flattened reduction axis).

- REQ-4: `cummin(input, dim)` — forward returns a `(values, indices)` tuple
  symmetric to cummax. Mirrors `Tensor cummin(...)` at
  `ReduceOps.cpp:899-904` → `cummin_helper_cpu` at `:867-873` →
  `cummax_cummin_helper<T1, T2, std::less_equal<scalar_t>>`. Upstream's
  tie-break operator is `std::less_equal` — on equal values the LATER
  index wins. Backward per `derivatives.yaml:537-539
  - name: cummin(Tensor self, int dim) -> (Tensor values, Tensor indices)
    self: cummaxmin_backward(grad, self, indices, dim)` is the same
  `scatter_add_(dim, indices, grad)` VJP.

  ferrotorch implements forward via `cummin` at `cumulative.rs:240-242`
  delegating to `crate::ops::cumulative::cummin_forward` (CPU loop at
  `ops/cumulative.rs:303-321`, GPU kernels at `ops/cumulative.rs:280-302`).
  Same two divergences as REQ-3: (a) strict `<` instead of `<=` at
  `ops/cumulative.rs:312` so the EARLIER index wins on ties; (b) declared
  non-differentiable at `cumulative.rs:236-242` contradicting upstream
  (blocker #1231). **Non-test production consumer**: invoked at
  `ferrotorch-core/src/einops.rs:802` (the `EinopsReduction::Min` arm,
  symmetric to REQ-3).

- REQ-5: `logcumsumexp(input, dim)` — numerically stable
  `out[..., i, ...] = log(sum(exp(input[..., 0..=i, ...])))` with autograd.
  Mirrors `Tensor logcumsumexp(const Tensor& self, int64_t dim)` at
  `ReduceOps.cpp:475-482` dispatching via `_logcumsumexp_cpu` at
  `:465-468` → `logcumsumexp_stub` at `:471`. The VJP per
  `tools/autograd/derivatives.yaml:521-523
  - name: logcumsumexp(Tensor self, int dim) -> Tensor
    self: logcumsumexp_backward(grad, self, result, dim)` factors as
  `grad_input[i] = exp(input[i]) * reverse_cumsum(grad_output * exp(-output))`
  (equivalent to softmax-weighted reverse cumsum). ferrotorch implements
  forward via `logcumsumexp` at `cumulative.rs:322-337`, delegating to
  `ops::cumulative::logcumsumexp_forward` (CPU two-pass max-rescaling
  algorithm at `ops/cumulative.rs:378-410`, GPU kernels at
  `ops/cumulative.rs:360-365`). Backward is
  `LogcumsumexpBackward` at `cumulative.rs:264-312` matching the
  derivatives-yaml formula. The numerical-stability invariant (large
  inputs ~1000.0 stay finite) is covered by
  `test_logcumsumexp_numerical_stability` at `cumulative.rs:719-736` and
  the gradient is validated against finite differences in
  `test_logcumsumexp_backward_1d` at `cumulative.rs:743-779`.
  **Diverges from upstream**: `LogcumsumexpBackward::backward` errors on
  CUDA at `cumulative.rs:272-276` (CPU round trip on backward). **No
  non-test production consumer** of `logcumsumexp` exists in
  `ferrotorch-*/src/` (blocker #1232).

- REQ-6: Shared `dim` normalization — every `pub fn` in the module calls
  `crate::shape::normalize_axis(dim as isize, input.ndim())?` to convert
  negative `dim` to non-negative and to error on out-of-range `dim`.
  Mirrors `maybe_wrap_dim(dim, self.dim())` at `ReduceOps.cpp:506` (in
  `impl_func_cum_ops`) and at `:622, :851, :890`. The error contract
  matches: ferrotorch returns `FerrotorchError::*` while upstream raises a
  Python `IndexError`/`RuntimeError`; the divergence is the Result-vs-raise
  vocabulary substitution permitted by R-DEV-4. Implemented at
  `cumulative.rs:73, :203, :231, :241, :323` (one call per public op).
  The `dim`-out-of-bounds and `dim`-for-scalar error cases are exercised
  by `test_cumsum_scalar_error`, `test_cumprod_scalar_error`,
  `test_cummax_scalar_error`, `test_cummin_scalar_error`,
  `test_logcumsumexp_scalar_error`, and `test_cumsum_dim_out_of_bounds`
  at `cumulative.rs:800-835`.

- REQ-7: `reverse_cumsum` helper — both `CumsumBackward::backward` and
  `LogcumsumexpBackward::backward` call
  `crate::ops::cumulative::reverse_cumsum` to compute the upper-triangular
  multiplication. Mirrors `static Tensor reversed_cumsum(const Tensor& w,
  int64_t dim)` at `ReduceOps.cpp:527-529`. Implemented at
  `ops/cumulative.rs:109-133`. Used at `cumulative.rs:50` (cumsum
  backward) and `cumulative.rs:291` (logcumsumexp backward). No
  production consumer outside the autograd path; this is an internal
  helper to satisfy REQ-1 and REQ-5.

## Acceptance Criteria

- [x] AC-1: `cumsum` parity-sweep at `--seeds 8` returns
  `[cumsum] 32/32 passed (0 skipped, 0 failed)` (smoke grep count = 1).
  Post-#1233 the 0-D scalar fast path lands at
  `ferrotorch-core/src/grad_fns/cumulative.rs:88-91` (`pub fn cumsum`
  early-out → `cumulative_scalar_identity`), so the 8 op_db samples
  that pass 0-D inputs now copy the scalar through unchanged, mirroring
  upstream's `impl_func_cum_ops` 0-D branch at `ReduceOps.cpp:501-504`.
- [x] AC-2: `cumprod` parity-sweep at `--seeds 8` returns
  `[cumprod] 80/80 passed (0 skipped, 0 failed)` (smoke grep count = 1).
  Same 0-D fast-path resolution as AC-1, dispatched via
  `cumulative.rs:337-340 pub fn cumprod` → `cumulative_scalar_identity`.
- [x] AC-3: `cummax` parity-sweep at `--seeds 8` returns
  `[cummax] 24/24 passed (0 skipped, 0 failed)` (smoke grep count = 1).
  0-D fast path at `cumulative.rs:374-377 pub fn cummax` →
  `cumextreme_scalar_identity` returns
  `CumExtremeResult { values: scalar, indices: vec![0] }`. The dispatch
  arm at `runner/src/main.rs:500 "cummax" =>` still selects only the
  `values` half (Option A from #1230). Indices-parity and
  differentiability divergences remain tracked under #1231.
- [x] AC-4: `cummin` parity-sweep at `--seeds 8` returns
  `[cummin] 24/24 passed (0 skipped, 0 failed)` (smoke grep count = 1).
  Same 0-D resolution as AC-3 via `cumulative.rs:391-394 pub fn cummin`
  → `cumextreme_scalar_identity`. Open blocker remains #1231 (indices
  + differentiability).
- [x] AC-5: `logcumsumexp` parity-sweep at `--seeds 8` returns
  `[logcumsumexp] 48/48 passed (0 skipped, 0 failed)` (smoke grep
  count = 1). 0-D fast path at `cumulative.rs:524-532 pub fn
  logcumsumexp` → `cumulative_scalar_identity`. The numerical identity
  is `logcumsumexp(x) = log(exp(x)) = x` on a scalar.
- [x] AC-6: `cargo test -p ferrotorch-core grad_fns::cumulative` passes
  every forward and backward test in `cumulative.rs:355-913` — covering
  1D / 2D dim=0 / 2D dim=1 / 3D forward shape correctness, negative-dim
  handling, numerical-gradient backward check for cumsum at
  `:880-913` and cumprod at `:841-874`, finite-difference backward for
  logcumsumexp at `:743-779`, zero-input cumprod backward at `:577-597`,
  scalar-input errors at `:800-828`, and dim-out-of-bounds at
  `:830-835`.
- [x] AC-7: Negative `dim` produces the same result as the equivalent
  positive `dim` — `test_cumsum_negative_dim` at `cumulative.rs:420-428`
  verifies `cumsum(x, -1) == cumsum(x, 1)` on shape `[2, 3]`.
- [x] AC-8: `requires_grad=false` inputs return a tensor with
  `grad_fn().is_none()` — verified by
  `test_cumsum_no_grad_fn_when_not_requires_grad` at `cumulative.rs:495-499`.
- [x] AC-9: Within a `no_grad` context, the returned tensor has
  `grad_fn().is_none()` even if the input has `requires_grad=true` —
  verified for cumsum/cumprod/logcumsumexp at `cumulative.rs:501-506,
  607-612, 789-794`.
- [ ] AC-10: cummax/cummin backward attaches the appropriate
  `CummaxBackward` / `CumminBackward` grad-fn when `input.requires_grad()`,
  routing grad through the saved indices via the `scatter_add` VJP per
  `derivatives.yaml:533-539`. **Currently fails**: the impls at
  `cumulative.rs:230, :240` are documented "not differentiable" and
  carry no grad-fn (blocker #1231).
- [ ] AC-11: cummax/cummin tie-breaking matches upstream — on equal
  values the LATER index wins (upstream uses `std::greater_equal` /
  `std::less_equal`). **Currently fails**: ferrotorch's CPU loop at
  `ops/cumulative.rs:247, :312` uses strict `>` / `<` so EARLIER index
  wins. Not yet tracked as a separate blocker — folded into #1231 as a
  semantic-correctness ride-along when CummaxBackward / CumminBackward
  ship; without correct tie-breaking the scatter_add VJP attributes
  grad to the wrong input position on ties.

## Architecture

### Layer split (`ops::cumulative` vs `grad_fns::cumulative`)

The file under design is the autograd layer; the kernel layer lives at
`ferrotorch-core/src/ops/cumulative.rs` (414 LOC: forward kernels for
all five ops plus the `reverse_cumsum` helper plus the
`CumExtremeResult { values, indices }` struct). The split mirrors
PyTorch's `_cummax_helper` / `_logcumsumexp` `_<op>` underscore-prefixed
private dispatchers (`ReduceOps.cpp:465-491, 828-834, 867-873`) vs the
user-facing `cummax` / `logcumsumexp` namespace functions.

### REQ-1 `cumsum` (lines 26-86)

`CumsumBackward<T>` (`cumulative.rs:35-39`) saves `input: Tensor<T>` and
`dim: usize`. Only the dim is materially used (it's a scalar field saved
to avoid re-normalizing on backward); the `input` field is saved so
`GradFn::inputs(&self)` at `:57-59` returns the right reference for the
autograd-graph topological walk. `backward` at `:41-55` materializes
`grad_output.data()` on CPU (rejecting CUDA at `:42-46`), calls
`reverse_cumsum`, wraps the result back into a tensor with
`requires_grad=false`, and returns `vec![Some(grad_input)]`. `pub fn
cumsum` at `:72-86` normalizes `dim`, calls `cumsum_forward`, and (when
`is_grad_enabled() && input.requires_grad()`) attaches a
`CumsumBackward` node via `Tensor::from_operation`. The non-`grad`
fast-exit at `:83-85` returns `result` unchanged.

### REQ-2 `cumprod` (lines 88-217)

`CumprodBackward<T>` at `:103-107` saves `input`, `output`, and `dim`.
Saving the output is the upstream-aligned optimization for the no-zeros
fast path (`output[j] / input[i]` requires both). The backward at
`:110-194` is the two-path split described in REQ-2 above. `pub fn
cumprod` at `:202-217` follows the same forward → optional grad-fn
attach pattern as cumsum.

### REQ-3/4 `cummax` / `cummin` (lines 219-242)

Currently thin pass-through wrappers — `cummax`/`cummin` at `:230` /
`:240` are single-line delegations to `cummax_forward` / `cummin_forward`
with no grad-fn attached. This is the divergence call-out: upstream is
differentiable per `derivatives.yaml:533-539` but ferrotorch is not
(blocker #1231). The non-test consumer is
`einops.rs:796 / :802` which uses the `values` field of
`CumExtremeResult` (the `indices` field is ignored at the consumer site,
which is why the missing backward has not yet caused observable failures
— the einops `EinopsReduction::Max` / `EinopsReduction::Min` path is
itself non-differentiable, so no grad needs to flow back through cummax).

### REQ-5 `logcumsumexp` (lines 244-337)

`LogcumsumexpBackward<T>` at `:264-268` saves `input`, `output`, and
`dim`. Backward at `:270-312`:
1. Compute `product[i] = grad_output[i] * exp(-output[i])`.
2. `rev = reverse_cumsum(product, shape, dim)`.
3. `grad_input[i] = exp(input[i]) * rev[i]`.

The formula docstring at `:248-262` self-corrects mid-comment from a
naive `exp(input - output)` form to the correct
`exp(input) * reverse_cumsum(go * exp(-output))` form — preserved
verbatim because it documents the derivation step the implementer
walked through. `pub fn logcumsumexp` at `:322-337` matches the
cumsum/cumprod scaffold.

### REQ-6 dim normalization (call sites only)

The `normalize_axis(dim as isize, ndim)` calls at `:73, :203, :231, :241,
:323` are the upstream `maybe_wrap_dim` analog (R-DEV-2: API-shape
match). The `isize` cast widens `i64` → `isize` which on every supported
host platform (64-bit) is lossless; on a 32-bit host the cast would
truncate but ferrotorch does not support 32-bit hosts.

### REQ-7 `reverse_cumsum` helper

Implemented in the kernel layer at `ops/cumulative.rs:109-133` and
re-imported at `cumulative.rs:16-19`. Consumers: `CumsumBackward`
(`:50`) and `LogcumsumexpBackward` (`:291`). The `cumprod` backward's
fast path inlines the equivalent reverse-cumsum-then-divide as a
single-loop `rev_acc` accumulator at `:172-178` rather than calling
`reverse_cumsum` — minor code duplication that is intentional because
the per-element division by `in_data[idx]` interleaves with the
reverse-cumsum accumulation.

## Parity contract

| Op | Upstream entry | Backward formula source | Edge cases mirrored |
|---|---|---|---|
| `cumsum` | `ReduceOps.cpp:511` `TORCH_IMPL_FUNC(cumsum_out)` | `derivatives.yaml:529-531` (`cumsum_backward` = `reversed_cumsum`) | Empty input along dim: ferrotorch returns shape-preserving empty (mirrors `impl_func_cum_ops` at `ReduceOps.cpp:503-504 result.zero_()`). Scalar input: errors (mirrors upstream's `self.dim() == 0` → `result.fill_(self)` branch by erroring instead — ferrotorch's `normalize_axis` errors when `ndim == 0`, so scalars cannot be cumsum'd). NaN / Inf: float arithmetic propagates naturally; no special-case handling. Non-contiguous: forward iterates by computed flat indices `base + i * inner` so stride doesn't matter for the CPU path, but the GPU path uses `gpu_handle()` which requires contiguous storage — non-contiguous CUDA inputs trigger `ops::cumulative` `is_cuda` paths that may need materialization (not yet audited; not blocking here). |
| `cumprod` | `ReduceOps.cpp:519` `TORCH_IMPL_FUNC(cumprod_out)` | `derivatives.yaml:525-527` (`cumprod_backward` = zeros-aware reverse-cumsum-divide) | Zeros in input: ferrotorch slow-path O(n^3) brute force matches upstream's masked-fill composite-compliance path numerically but not algorithmically. Test at `cumulative.rs:577-597` verifies `cumprod([2, 0, 3]).backward() == [1, 8, 0]`. NaN / Inf: propagates naturally; `0 * inf = NaN` will materialize through. Non-contiguous: same caveat as cumsum. Second-order grad (`grad_of_grad`): unsupported — the slow path is O(n^3) and not second-order-differentiable. |
| `cummax` | `ReduceOps.cpp:860` `Tensor cummax(...)` | `derivatives.yaml:533-535` (`cummaxmin_backward` = `scatter_add` through indices) | Returns `CumExtremeResult { values, indices }` (Rust analog of `std::tuple<Tensor, Tensor>`). Tie-breaking: **DIVERGES** — upstream uses `std::greater_equal` (later wins), ferrotorch uses strict `>` (earlier wins). NaN: **DIVERGES** — upstream's `isnan_(curr_elem)` branch at `ReduceOps.cpp:819` propagates NaN through as the running max (subsequent values stay NaN forever); ferrotorch's strict `cur_max > -inf` after NaN-comparison-with-anything returns false will keep `cur_max` as the prior non-NaN value. Differentiability: **DIVERGES** — upstream is differentiable, ferrotorch declares non-differentiable. (blocker #1231 covers tie-break + differentiability + NaN.) |
| `cummin` | `ReduceOps.cpp:899` `Tensor cummin(...)` | `derivatives.yaml:537-539` (same `cummaxmin_backward`) | Symmetric to cummax with all the same divergences (tie-break, NaN, differentiability). |
| `logcumsumexp` | `ReduceOps.cpp:475` `Tensor logcumsumexp(...)` | `derivatives.yaml:521-523` (`logcumsumexp_backward` = `exp(input) * reverse_cumsum(grad * exp(-output))`) | Numerical stability: ferrotorch's two-pass running-max algorithm at `ops/cumulative.rs:382-410` ensures inputs at scale ~1000 stay finite, verified by `test_logcumsumexp_numerical_stability` at `cumulative.rs:719-736`. NaN / Inf: `(-inf).exp() == 0` and `0.ln() == -inf` give the upstream-aligned `logcumsumexp([-inf, x]) == [-inf, x]` behavior. Empty input: errors (via `normalize_axis` scalar check). |

Parity-sweep audit reference: all five op entries are **MISSING** from
`tools/parity-sweep/parity_audit.json` as of this writeup. Adding them
is part of blocker #1230.

## Verification

### Existing unit tests (all passing)

Located at `ferrotorch-core/src/grad_fns/cumulative.rs:355-913` (the
`#[cfg(test)] mod tests` block). Key tests:

- `test_cumsum_1d` (`:376-386`), `test_cumsum_2d_dim0` (`:388-402`),
  `test_cumsum_2d_dim1` (`:404-417`), `test_cumsum_negative_dim`
  (`:419-428`), `test_cumsum_3d` (`:430-443`)
- `test_cumsum_backward_1d` (`:449-463`), `test_cumsum_backward_2d_dim0`
  (`:465-484`), `test_cumsum_backward_numerical` (`:880-913`)
- `test_cumsum_has_grad_fn` (`:486-492`),
  `test_cumsum_no_grad_fn_when_not_requires_grad` (`:494-499`),
  `test_cumsum_no_grad_fn_in_no_grad_context` (`:501-506`)
- `test_cumprod_1d` (`:512-521`), `test_cumprod_2d_dim0` (`:523-536`),
  `test_cumprod_2d_dim1` (`:538-551`)
- `test_cumprod_backward_1d` (`:557-574`),
  `test_cumprod_backward_with_zero` (`:576-597`),
  `test_cumprod_backward_numerical` (`:841-874`)
- `test_cummax_1d` (`:618-629`), `test_cummax_2d_dim1` (`:631-646`)
- `test_cummin_1d` (`:652-663`), `test_cummin_2d_dim0` (`:665-678`)
- `test_logcumsumexp_1d` (`:684-698`), `test_logcumsumexp_2d_dim1`
  (`:700-716`), `test_logcumsumexp_numerical_stability` (`:718-736`)
- `test_logcumsumexp_backward_1d` (`:742-779`)
- `test_*_scalar_error` (`:800-828`), `test_cumsum_dim_out_of_bounds`
  (`:830-835`)

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
every op. For cummax/cummin the runner dispatch (`runner/src/main.rs:500
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
| REQ-1 (cumsum) | NOT-STARTED | impl: `cumsum` at `ferrotorch-core/src/grad_fns/cumulative.rs:88` + `CumsumBackward` at `:35` mirroring `ReduceOps.cpp:511 TORCH_IMPL_FUNC(cumsum_out)` and `derivatives.yaml:529-531`. 0-D scalar fast path at `cumulative.rs:88-91` (early-out into `cumulative_scalar_identity`) + `CumsumBackward::backward` 0-D fast path at `:49-51` mirror upstream's `impl_func_cum_ops` 0-D branch at `ReduceOps.cpp:501-504` (`result.fill_(self)`). Post-#1233 parity-sweep: `[cumsum] 32/32 passed (0 skipped, 0 failed)`. **Still no non-test production consumer** in `ferrotorch-*/src/` (only `lib.rs:159` re-export, plus the parity-sweep runner dispatch arm at `tools/parity-sweep/runner/src/main.rs:471 "cumsum" =>` which is TEST-SIDE per R-DEFER-1). Open blocker: #1232 (production consumer wiring). |
| REQ-2 (cumprod) | NOT-STARTED | impl: `cumprod` at `cumulative.rs:337` + `CumprodBackward` at `:227` mirroring `ReduceOps.cpp:519 TORCH_IMPL_FUNC(cumprod_out)` and `derivatives.yaml:525-527`; backward zeros-path is O(n^3) brute-force not upstream's composite-compliance masked-fill. 0-D scalar fast path at `cumulative.rs:337-340` + `CumprodBackward::backward` 0-D fast path at `:239-241`. Post-#1233 parity-sweep: `[cumprod] 80/80 passed (0 skipped, 0 failed)`. **No non-test production consumer**. Open blocker: #1232. |
| REQ-3 (cummax) | NOT-STARTED | impl: `cummax` at `cumulative.rs:374` delegating to `ops/cumulative.rs:191 cummax_forward` mirroring `ReduceOps.cpp:860 Tensor cummax(...)`; non-test production consumer at `ferrotorch-core/src/einops.rs:796` inside `pub fn reduce<T: Float>` (uses `cmax.values`). 0-D scalar fast path at `cumulative.rs:374-377` → `cumextreme_scalar_identity` at `:412-432` returns `CumExtremeResult { values: scalar, indices: vec![0] }`. **Diverges**: declared non-differentiable (contradicts `derivatives.yaml:533-535`); tie-break strict `>` (upstream `std::greater_equal`); NaN handling not mirrored. Post-#1233 parity-sweep: `[cummax] 24/24 passed (0 skipped, 0 failed)` via Option A (runner returns `values` only). Open blocker: #1231 (differentiability + tie-break + NaN). |
| REQ-4 (cummin) | NOT-STARTED | impl: `cummin` at `cumulative.rs:391` delegating to `ops/cumulative.rs:272 cummin_forward` mirroring `ReduceOps.cpp:899 Tensor cummin(...)`; non-test production consumer at `ferrotorch-core/src/einops.rs:802`. 0-D scalar fast path at `cumulative.rs:391-394` → `cumextreme_scalar_identity`. Same divergences as REQ-3 (non-diff, strict `<` tie-break, NaN). Post-#1233 parity-sweep: `[cummin] 24/24 passed (0 skipped, 0 failed)` (Option A symmetric to REQ-3). Open blocker: #1231. |
| REQ-5 (logcumsumexp) | NOT-STARTED | impl: `logcumsumexp` at `cumulative.rs:524` + `LogcumsumexpBackward` at `:453` mirroring `ReduceOps.cpp:475 Tensor logcumsumexp(...)` and `derivatives.yaml:521-523`; backward formula matches `exp(input) * reverse_cumsum(grad * exp(-output))`. 0-D scalar fast path at `cumulative.rs:524-532` (forward) + `:466-468` (backward — identity VJP since `log(exp(x)) = x`). Numerical stability covered by `test_logcumsumexp_numerical_stability` (`:929-946`). Post-#1233 parity-sweep: `[logcumsumexp] 48/48 passed (0 skipped, 0 failed)`. **No non-test production consumer**. Open blocker: #1232. |
| REQ-6 (dim normalization) | SHIPPED | impl: `normalize_axis(dim as isize, input.ndim())?` calls at `cumulative.rs:73, :203, :231, :241, :323` per `crate::shape::normalize_axis` mirroring `maybe_wrap_dim` at `ReduceOps.cpp:506, :622, :851, :890`; production consumer for the normalized result is each of the five `pub fn` bodies themselves (the normalized `dim` is stored into the `*Backward` struct and threaded into `from_storage`); reachable production callers: `einops.rs:796 / :802` invoke `cummax(view, 1)` / `cummin(view, 1)` triggering the normalize path. Tests at `cumulative.rs:420-428 test_cumsum_negative_dim` and `:830-835 test_cumsum_dim_out_of_bounds` cover the negative-dim and out-of-range cases. |
| REQ-7 (reverse_cumsum helper) | SHIPPED | impl: `reverse_cumsum` at `ferrotorch-core/src/ops/cumulative.rs:109` mirroring `static Tensor reversed_cumsum(const Tensor& w, int64_t dim)` at `ReduceOps.cpp:527-529`; non-test production consumers at `ferrotorch-core/src/grad_fns/cumulative.rs:50` (CumsumBackward::backward) and `ferrotorch-core/src/grad_fns/cumulative.rs:291` (LogcumsumexpBackward::backward). The helper itself is internal scaffolding; its end-to-end exercise lands when REQ-1 and REQ-5 ship with runner-side parity coverage (blocker #1230). Forward and backward unit tests at `cumulative.rs:449-484` (`test_cumsum_backward_*`) and `:742-779` (`test_logcumsumexp_backward_1d`) verify it numerically through the consumer path. |
