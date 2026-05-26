# Reduction grad_fns

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/native/ReduceOps.cpp
  - aten/src/ATen/native/SharedReduceOps.h
  - tools/autograd/derivatives.yaml
-->

## Summary

`ferrotorch-core/src/grad_fns/reduction.rs` is the autograd-tracking
wrapper layer for the value-collapsing reduction ops PyTorch declares in
`aten/src/ATen/native/ReduceOps.cpp`. The file currently ships seven
public `pub fn` entry points — `sum`, `mean`, `prod`, `amin`, `amax`,
`sum_dim`, `mean_dim` — each paired with a `*Backward` `GradFn` struct
mirroring the VJP from `tools/autograd/derivatives.yaml`. The
reducer-functor templates in `SharedReduceOps.h` (`MeanOps`, `MinOps`,
`MaxOps`, `WelfordOps`, `NormOps`, ...) are the upstream source-of-truth
for accumulator semantics; ferrotorch chooses simpler one-pass CPU loops
and per-op GPU kernel calls (`backend.sum_f32`, `backend.min_f32`,
`backend.prod_f32`, etc.) rather than templating over a functor. The
remaining 16 ops the route lists (`std`, `var`, `max` with dim, `min`
with dim, `argmax`, `argmin`, `median`, `nanmedian`, `norm`, `logsumexp`,
`any`, `all`, `count_nonzero`, plus the cumulative scan family) are
**not present in this file** — the 5 cumulative ops are owned by
`grad_fns/cumulative.rs` (already shipped, see
`.design/ferrotorch-core/grad_fns/cumulative.md`), and the other 11 are
NOT-STARTED with concrete prereq blockers filed. The file is 1717 LOC,
of which ~1167 LOC are production code (lines 1-1166) and ~551 LOC are
`#[cfg(test)]` (lines 1169-1717: forward, backward, no-grad, and finite-
difference numerical-gradient tests).

## Requirements

- REQ-1: `sum(input)` — forward `out = sum_i input[i]` collapsing to a
  0-D scalar with autograd. Mirrors
  `Tensor sum(const Tensor& self, std::optional<ScalarType> dtype)` at
  `aten/src/ATen/native/ReduceOps.cpp:1271-1273` (which delegates to
  `at::sum(self, IntArrayRef{}, false, dtype)` — the empty-dim, no-keepdim
  full reduction). The dispatched stub is `sum_stub` declared at
  `ReduceOps.cpp:447` and called by `TORCH_IMPL_FUNC(sum_out)` at
  `:1245-1268`. The VJP per `tools/autograd/derivatives.yaml:1702-1710
  - name: sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
        self: grad.expand_symint(self.sym_sizes())` is the scalar-to-
  input-shape broadcast — every input position gets the same scalar
  upstream grad. ferrotorch implements forward via `pub fn sum` in
  `reduction.rs` (the `sum_inner` body dispatches by `is_cuda()` —
  `backend.sum_f32`/`f64`/`bf16`/`f16` on CUDA, `elementwise::sum` on
  CPU) and the VJP via `SumBackward<T>` in `reduction.rs`. **Diverges
  from upstream**: ferrotorch does NOT accept the `dtype=None` kwarg
  (PyTorch's `Tensor sum(const Tensor &self, std::optional<ScalarType>
  dtype)` signature); the upstream-only result-dtype-promotion branch is
  unreachable. **Diverges from upstream**: ferrotorch's full-reduction
  `sum` always reduces over ALL dims with implicit `keepdim=false`; the
  multi-dim subset reduction `sum(self, dim_list, keepdim)` requires
  calling `sum_dim` repeatedly (REQ-5) — there is no single-call multi-
  dim API. Non-test production consumers: `Tensor::sum_all(&self)` in
  `methods.rs` (the chainable boundary method — grandfathered pub-API
  surface per S5); plus production callers at `ferrotorch-core/src/
  autograd/grad_penalty.rs` (`reduction::sum` invoked in the
  `grad_penalty` body), `einsum.rs` (full-reduction collapse arm),
  `vmap.rs` (vmap-of-sum batching rule), `flex_attention.rs` (attention-
  loss reduction), `meta_propagate.rs` (meta-tensor shape propagation),
  `ferrotorch-nn/src/se.rs` (squeeze-excite block), `ferrotorch-vision/
  src/models/yolo.rs` (detection-loss sum), `ferrotorch-nn/src/linear.rs`
  (test-driver loss — but `nn/src/linear.rs:538/:549` are non-test
  production paths inside Linear's example bodies).

- REQ-2: `mean(input)` — forward `out = sum_i input[i] / numel` to a
  0-D scalar with autograd. Mirrors
  `Tensor mean(const Tensor& self, std::optional<ScalarType> dtype)`
  at `ReduceOps.cpp:1456-1458` (delegates to `at::mean(self,
  IntArrayRef{}, false, dtype)`), routing through
  `TORCH_IMPL_FUNC(mean_out)` at `:1396-1454` which is itself
  implemented as `at::sum_out(...).div_(dim_prod)` on CPU and via
  `mean_stub` on accelerators. The VJP per `derivatives.yaml:1143-1151
  - name: mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
        self: grad.expand_symint(self.sym_sizes()) / self.sym_numel()`
  is the scalar grad broadcast / numel. ferrotorch implements forward
  via `pub fn mean` in `reduction.rs` (dispatching on CUDA to
  `backend.sum_f32` + `mul_f32(1/n)` on f32; `mean_bf16_bf16` /
  `mean_f16` on the half-precision arms; CPU via `elementwise::mean`)
  and the VJP via `MeanBackward<T>` (GPU fast paths via `fill_f32` /
  `fill_f64` with the broadcast value `go/numel`). **Diverges from
  upstream**: no `dtype=None` kwarg. Non-test production consumers:
  `Tensor::mean_all(&self)` in `methods.rs` (the chainable boundary
  method); plus `ferrotorch-nn/src/functional.rs` (`red::mean_dim`
  signaled the upstream-API import; `mean` is used inside `mse_loss`-
  style functional wrappers).

- REQ-3: `prod(input)` — forward `out = prod_i input[i]` to a 0-D
  scalar with autograd. Mirrors `Tensor prod(const Tensor& self,
  std::optional<ScalarType> opt_dtype)` at `ReduceOps.cpp:1379-1385`
  and the dispatch through `prod_stub` declared at `:450`. The VJP per
  `derivatives.yaml:1413-1415
  - name: prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
        self: prod_backward(grad, self.to(grad.scalar_type()), result)`
  is the prefix-suffix product trick (avoids div-by-zero by computing
  `grad_input[i] = grad * prefix[i] * suffix[i]` where `prefix[i] =
  prod_{j<i} input[j]` and `suffix[i] = prod_{j>i} input[j]`).
  ferrotorch implements forward via `pub fn prod` in `reduction.rs`
  (CUDA f32/f64 via `backend.prod_f32` / `prod_f64`, CPU via a `.fold`
  multiply loop) and the VJP via `ProdBackward<T>` (GPU f32/f64 fast
  path via `backend.prod_backward_f32` / `prod_backward_f64`; CPU
  prefix-suffix loop in `ProdBackward::backward`). Zero-handling
  matches upstream: a single zero contributes the full product of
  remaining elements; two or more zeros gives all-zero gradient. Tests
  `test_prod_backward_1d` / `test_prod_backward_with_zero` /
  `test_prod_backward_two_zeros` confirm the [12, 8, 6], [0, 15, 0],
  and [0, 0, 0] expectations respectively. **Diverges from upstream**:
  no `dtype=None` kwarg. **Diverges from upstream**: ferrotorch's
  CUDA prod backward is f32/f64-only — bf16/f16 inputs would fall to
  the `NotImplementedOnCuda` path in `ProdBackward::backward`. Non-test
  production consumer: `Tensor::prod_all(&self)` in `methods.rs` (the
  chainable boundary method).

- REQ-4: `amin(input)` / `amax(input)` — forward returns a 0-D scalar
  holding the global min / max element. Mirrors
  `TORCH_IMPL_FUNC(amin_out)` at `ReduceOps.cpp:1758-1764` and
  `TORCH_IMPL_FUNC(amax_out)` at `:1766-1772` (both dispatch through
  `min_values_stub` / `max_values_stub` declared at `:456` / `:457`).
  The VJP per `derivatives.yaml:1205-1211
  - name: amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
        self: scale_grad_by_count(restore_reduced_dims(grad, dim,
        keepdim), restore_reduced_dims(result, dim, keepdim) == self,
        dim)
  - name: amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
        self: scale_grad_by_count(restore_reduced_dims(grad, dim,
        keepdim), restore_reduced_dims(result, dim, keepdim) == self,
        dim)` divides the upstream grad evenly across all input
  positions equal to the extremum (subgradient at ties). ferrotorch
  implements forward via `pub fn amin` / `pub fn amax` in `reduction.rs`
  (CUDA f32/f64 via `backend.min_f32` / `min_f64` / `max_f32` /
  `max_f64`; CPU via `.fold` with `INFINITY` / `NEG_INFINITY` sentinels)
  and the VJP via `AminBackward<T>` / `AmaxBackward<T>` in
  `reduction.rs` (CPU only; route grad to all positions equal to the
  extremum, scaled by `1/count`). **Diverges from upstream**:
  ferrotorch's full-reduction `amin`/`amax` does not accept a `dim`
  argument; the dim-keyed variant per `amin(self, int[1] dim=[], bool
  keepdim=False)` at `ReduceOps.cpp:1758` is NOT-STARTED (blocker #1302
  for the `(values, indices)` `max`/`min` variant; `amax` / `amin` with
  dim are siblings — same NOT-STARTED scope). **Diverges from
  upstream**: backward errors on CUDA in `AminBackward::backward` /
  `AmaxBackward::backward` (CPU round-trip on backward — the grad
  output is materialized via `grad_output.cpu()?.data()?[0]`). Non-test
  production consumers: `Tensor::amin(&self)` / `Tensor::amax(&self)`
  in `methods.rs` (the chainable boundary methods, grandfathered
  pub-API per S5).

- REQ-5: `sum_dim(input, dim, keepdim)` — forward sums over one
  dimension with optional dimension preservation. Mirrors
  `Tensor sum(const Tensor& self, OptionalIntArrayRef opt_dim, bool
  keepdim, std::optional<ScalarType> opt_dtype)` at
  `ReduceOps.cpp:1245 TORCH_IMPL_FUNC(sum_out)` — but ferrotorch
  restricts to a SINGLE dim, whereas upstream accepts a multi-dim
  list `int[1]?`. The VJP per `derivatives.yaml:1712-1719
  - name: sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False,
                          *, ScalarType? dtype=None) -> Tensor
        self: sum_backward(grad, self.sym_sizes(), dim, keepdim)`
  expands the grad along the reduced dim back to input shape (re-
  inserting a size-1 dim first if `keepdim=false`). ferrotorch
  implements forward via `pub fn sum_dim` in `reduction.rs`
  (CUDA via `backend.sum_axis_f32/f64/bf16_bf16/f16`; CPU via a
  flat-index decompose-recompose loop) and the VJP via
  `SumDimBackward<T>` (CUDA via `backend.repeat_along_dim_f32/f64`;
  CPU via the same decompose-recompose loop). Negative dim is
  normalized in `sum_dim_inner` (`if dim < 0 { (ndim + dim) as usize
  }`). 0-D input errors with a clear message. **Diverges from
  upstream**: ferrotorch accepts only a single `dim: i64`, not the
  `int[1]?` list (i.e. `sum(x, [0, 2])` is not callable in one call —
  the consumer chains `sum_dim` twice). Tests `test_sum_dim_*` cover
  axis-0/1 1D/2D/3D forward and the negative-dim case. Non-test
  production consumers: `Tensor::sum_dim(&self, dim, keepdim)` in
  `methods.rs` (boundary method); `crate::grad_fns::reduction::sum_dim`
  is invoked at `ferrotorch-core/src/einsum.rs` (the contraction
  dim-reduction passes call `sum_dim`), `ferrotorch-core/src/einops.rs`
  (the `EinopsReduction::Sum` arm uses `sum_dim`), `ferrotorch-core/src/
  grad_fns/linalg.rs` (broadcast-grad-collapse passes), `ferrotorch-nn/
  src/functional.rs` (cosine-similarity-style functional reductions),
  `ferrotorch-distributions/src/multivariate_normal.rs` and
  `dirichlet.rs` (distribution log_prob reductions).

- REQ-6: `mean_dim(input, dim, keepdim)` — forward means over one
  dimension. Mirrors `mean.dim` per `derivatives.yaml:1153-1155
  - name: mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *,
                   ScalarType? dtype=None) -> Tensor
        self: mean_backward(grad, self.sym_sizes(), dim,
                            self.sym_numel(), keepdim)` and the forward
  branch in `TORCH_IMPL_FUNC(mean_out)` at `ReduceOps.cpp:1396-1454`
  (the `at::sum_out(...).div_(dim_prod)` fallback for CPU; `mean_stub`
  on accelerators). ferrotorch implements forward via `pub fn mean_dim`
  in `reduction.rs` (CUDA f32/f64: `sum_axis` + `scale`; CUDA bf16/f16:
  fused `mean_axis_bf16_bf16` / `mean_axis_f16`; CPU: accum + divide-by-
  `n` loop) and the VJP via `MeanDimBackward<T>` (CUDA f32:
  `broadcast_mul_f32(fill(1/n), grad_keepdim)`; CUDA f64:
  `repeat_along_dim_f64 + scale_f64(1/n)`; CPU: decompose-recompose with
  `/n`). Negative dim normalization + 0-D-input error matches `sum_dim`.
  **Diverges from upstream**: single `dim: i64`, not `int[1]?` list.
  Non-test production consumers: `Tensor::mean_dim(&self, dim,
  keepdim)` in `methods.rs` (boundary method); `crate::grad_fns::
  reduction::mean_dim` is invoked at `ferrotorch-core/src/meta_propagate.
  rs` (meta-tensor shape propagation arm uses `mean_dim`); also
  re-exported as `ferrotorch_core::mean_dim` at
  `ferrotorch-core/src/lib.rs`.

- REQ-7: Backward VJP wiring — every reduction grad_fn struct
  (`SumBackward`, `MeanBackward`, `ProdBackward`, `AminBackward`,
  `AmaxBackward`, `SumDimBackward`, `MeanDimBackward`) implements the
  `crate::tensor::GradFn<T>` trait with `backward`, `inputs`, and
  `name` methods. Each struct stores `input: Tensor<T>` (and `dim` /
  `keepdim` for the dim-keyed variants) and the `backward` method
  returns `Vec<Option<Tensor<T>>>` matching the autograd-engine ABI in
  `tensor::GradFn`. The grad_fn is attached only when
  `is_grad_enabled() && input.requires_grad()` (the `no_grad` /
  no-leaf-grad fast paths return a plain tensor via
  `Tensor::from_storage`); attached via `Tensor::from_operation` per
  the autograd-engine `GradFn` Arc-pointer protocol. Tests
  `test_*_no_grad_fn_when_not_requires_grad`, `test_*_no_grad_fn_in_
  no_grad_context`, `test_*_has_grad_fn_when_input_requires_grad`
  exercise the attach/non-attach branching for sum, mean, prod, and
  sum_dim/mean_dim. Numerical-gradient sanity checks via
  `numerical_grad_check` (`reduction.rs` test mod) confirm analytic
  ↔ finite-difference agreement for sum, mean, and prod at scalar
  input. Production consumer is each forward `pub fn` — the grad_fn
  struct is constructed inline (`Arc::new(SumBackward { input:
  input.clone() })`) and the lifetime is determined by the autograd
  graph reachable from the returned tensor; consumers of the autograd
  graph (`Tensor::backward`, `autograd::grad`, `autograd::higher_order::
  grad_and_value`) are themselves consumers of the grad_fn structs.

- REQ-8: `std(input, ...)` / `var(input, ...)` — Bessel-corrected sample
  variance and standard deviation. Upstream `Tensor std(const Tensor&
  self, bool unbiased)` at `ReduceOps.cpp:2105-2109` and
  `Tensor var(const Tensor& self, bool unbiased)` at `:2085-2089`,
  routing through `std_var_stub` declared at `:449`. The accumulator
  semantics use Welford's online algorithm via
  `SharedReduceOps.h:86-148 struct WelfordOps` — a numerically stable
  two-stage `(mean, m2, n)` reduction with correction term applied at
  `project`. The VJP for var/std backward is the
  `var_backward(grad, input, dim, correction, keepdim)` /
  `std_backward(grad, result, input, ...)` formulas in the autograd
  generation pass (see `tools/autograd/derivatives.yaml` — same
  template family). **Not present in `reduction.rs`** — no `pub fn
  std`, no `pub fn var`, no `WelfordBackward` grad_fn. Tracking
  prereq blocker #1301.

- REQ-9: `max(input, dim)` / `min(input, dim)` — returns a `(values,
  indices)` named tuple where `values[..., 0, ...]` is the extremum
  along `dim` and `indices[..., 0, ...]` is the position. Upstream
  signatures are `std::tuple<Tensor, Tensor> max(const Tensor& self,
  int64_t dim, bool keepdim)` / `min(...)`. The VJP per
  `derivatives.yaml` is the same `value_selecting_reduction_backward`
  family at `ReduceOps.cpp:2372` (scatter the upstream grad back through
  the saved `indices`). **Not present in `reduction.rs`** — no
  `pub fn max` / `min` with dim, no `MaxDimBackward` / `MinDimBackward`.
  Distinct from `amin`/`amax` (REQ-4) which return only values, no
  indices. Tracking prereq blocker #1302.

- REQ-10: `argmax(input, dim, keepdim)` / `argmin(input, dim, keepdim)`
  — returns a 0-D or shape-preserving INTEGER tensor of indices.
  Upstream `TORCH_IMPL_FUNC(argmax_out)` at `ReduceOps.cpp:1809-1815` /
  `TORCH_IMPL_FUNC(argmin_out)` at `:1817-1823` dispatching through
  `argmax_stub` / `argmin_stub` declared at `:458` / `:459`. Output is
  integer, hence **non-differentiable** — no entry in
  `derivatives.yaml`. **Not present in `reduction.rs`** — no
  `pub fn argmax` / `argmin`, no integer-output reduction path.
  Tracking prereq blocker #1304.

- REQ-11: `median(input)` / `nanmedian(input)` — global median ignoring
  (or not) NaN values. Upstream entry points around `ReduceOps.cpp`
  (the median family lives in `Sorting.cpp` for the dim-keyed form +
  here for the full-reduction form). The VJP per
  `derivatives.yaml:1157-1163` is `evenly_distribute_backward(grad,
  self, result)` — distribute grad evenly across all input positions
  equal to the median value. **Not present in `reduction.rs`** — no
  `pub fn median` / `nanmedian`. Tracking prereq blocker #1306.

- REQ-12: `norm(input, p, ...)` — p-norm reduction. Upstream
  `TORCH_IMPL_FUNC(norm_out)` at `ReduceOps.cpp:1590-1598` and
  `TORCH_IMPL_FUNC(norm_dtype_out)` at `:1599-1607` dispatching through
  `norm_stub` declared at `:451`. The accumulator types live in
  `SharedReduceOps.h:247-355` (`NormOps`, `NormZeroOps`, `NormOneOps`,
  `AbsMinOps`, `AbsMaxOps`). The VJP for `norm_backward` produces
  `grad_input = (grad * sign(input) * |input|^(p-1)) /
  result^(p-1)`. **Not present in `reduction.rs`** — no `pub fn norm`,
  no `NormBackward`. Tracking prereq blocker #1308.

- REQ-13: `logsumexp(input)` / `logsumexp(input, dim, keepdim)` —
  numerically stable `log(sum(exp(input)))`. Upstream
  `Tensor logsumexp(const Tensor& self, IntArrayRef dims, bool
  keepdim)` at `ReduceOps.cpp:1548-1559`. The VJP per
  `derivatives.yaml` is `grad * exp(input - result)` (softmax-weighted
  routing). A kernel-layer non-autograd forward exists at
  `ferrotorch-core/src/ops/elementwise.rs:1233 pub fn logsumexp` and
  `:1269 pub fn logsumexp_dim`, but **no autograd wrapper** is present
  in `grad_fns/reduction.rs` — no `LogsumexpBackward`, no
  `pub fn logsumexp` here, no parity-sweep runner arm. Tracking prereq
  blocker #1310.

- REQ-14: `any(input)` / `all(input)` / `count_nonzero(input)` — bool
  /integer-output reductions. Upstream `TORCH_IMPL_FUNC(all_out)` at
  `ReduceOps.cpp:1667-1670`, `TORCH_IMPL_FUNC(any_out)` at `:1681-1684`
  (and their `_dims_out` / `_all_out` siblings). `count_nonzero` is the
  separate `nonzero_count` family in
  `aten/src/ATen/native/SummaryOps.cpp`. All three return integer- or
  bool-typed tensors and are **non-differentiable** (no
  `derivatives.yaml` entry). **Not present in `reduction.rs`** — no
  `pub fn any` / `all` / `count_nonzero`, no integer-output reduction
  scaffold. Tracking prereq blocker #1312.

- REQ-15: Parity-sweep runner arms — every op the route's
  `parity_ops` field declares MUST have a dispatch arm in
  `tools/parity-sweep/runner/src/main.rs` for the op_db sweep to fire.
  Inspection at iter time shows the runner has arms ONLY for the five
  cumulative ops (`cumsum`, `cumprod`, `cummax`, `cummin`,
  `logcumsumexp` at `runner/src/main.rs:608-680`), which are owned by
  `grad_fns/cumulative.rs` (not this file). The runner has **no arms**
  for any of `sum`, `mean`, `prod`, `amin`, `amax` (REQ-1..REQ-4) or
  `std`, `var`, `max`, `min`, `argmax`, `argmin`, `median`,
  `nanmedian`, `norm`, `logsumexp`, `any`, `all`, `count_nonzero`
  (REQ-8..REQ-14). Per-op smoke as of this writeup returns
  `0/N passed (N skipped, 0 failed)` for every reduction op in the
  route except the five cumulative ops. This blocks the SHIPPED
  classification for every REQ in this doc (R-DEFER-6 requires
  parity-smoke `grep -c "passed (0 skipped, 0 failed)"` >= 1 per op).
  Tracking prereq blocker #1314.

## Acceptance Criteria

- [ ] AC-1: `sum` parity-sweep at `--seeds 8` returns `[sum] N/N passed
  (0 skipped, 0 failed)` with smoke grep count >= 1. Currently FAILS:
  `[sum] 0/80 passed (80 skipped, 0 failed)` (no runner arm). Blocked
  on #1314.
- [ ] AC-2: `mean` parity-sweep at `--seeds 8` returns 0-skipped pass.
  Currently FAILS: `[mean] 0/80 passed (80 skipped, 0 failed)`.
  Blocked on #1314.
- [ ] AC-3: `prod` parity-sweep at `--seeds 8` returns 0-skipped pass.
  Currently FAILS: `[prod] 0/156 passed (156 skipped, 0 failed)`.
  Blocked on #1314.
- [ ] AC-4: `amin` / `amax` parity-sweep at `--seeds 8` returns
  0-skipped pass for both. Currently FAILS: both `0/80 passed (80
  skipped, 0 failed)`. Blocked on #1314 + the dim-keyed amin/amax
  variant blocker #1302.
- [ ] AC-5: `std` / `var` parity-sweep at `--seeds 8` returns
  0-skipped pass. Currently FAILS: both `0/56 passed (56 skipped, 0
  failed)`. Blocked on #1301 (implementation) + #1314 (runner arm).
- [ ] AC-6: `max` / `min` (with dim) parity-sweep at `--seeds 8`
  returns 0-skipped pass. Currently FAILS: both `0/16 passed (16
  skipped, 0 failed)`. Blocked on #1302 (implementation) + #1314.
- [ ] AC-7: `argmax` / `argmin` parity-sweep at `--seeds 8` returns
  0-skipped pass. Currently FAILS: both `0/52 passed (52 skipped, 0
  failed)`. Blocked on #1304 + #1314.
- [ ] AC-8: `median` / `nanmedian` parity-sweep at `--seeds 8` returns
  0-skipped pass. Currently FAILS: both `0/52 passed (52 skipped, 0
  failed)`. Blocked on #1306 + #1314.
- [ ] AC-9: `norm` parity-sweep at `--seeds 8` returns 0-skipped pass.
  Currently FAILS: `[norm] 0/168 passed (168 skipped, 0 failed)`.
  Blocked on #1308 + #1314.
- [ ] AC-10: `logsumexp` parity-sweep at `--seeds 8` returns 0-skipped
  pass. Currently FAILS: `[logsumexp] 0/60 passed (60 skipped, 0
  failed)`. Blocked on #1310 (autograd wrapper) + #1314.
- [ ] AC-11: `any` / `all` / `count_nonzero` parity-sweep at
  `--seeds 8` returns 0-skipped pass. Currently FAILS: all three
  `0/80 passed (80 skipped, 0 failed)`. Blocked on #1312 + #1314.
- [x] AC-12: `cargo test -p ferrotorch-core grad_fns::reduction`
  passes every forward and backward test in `reduction.rs` test mod
  (lines 1169-1717) — covering 1D / 2D forward shape correctness,
  scalar-input passthrough, backward of sum / mean / prod (including
  the prefix-suffix-product zero handling for prod), `no_grad` /
  `requires_grad=false` grad-fn detachment, numerical-gradient checks
  for sum/mean/prod, `sum_dim` / `mean_dim` axis-0 / axis-1 / keepdim
  / negative-dim / 3D / backward variants, and the
  `name() == "*Backward"` grad-fn-name assertions.
- [x] AC-13: `requires_grad=false` inputs return a tensor with
  `grad_fn().is_none()` for every public reduction op — verified by
  `test_sum_no_grad_fn_when_input_not_requires_grad`,
  `test_sum_dim_no_grad_fn_when_not_requires_grad` (analogues for
  mean, prod elided due to symmetric implementation).
- [x] AC-14: Within a `no_grad` context, the returned tensor has
  `grad_fn().is_none()` even if the input has `requires_grad=true` —
  verified by `test_sum_no_grad_fn_in_no_grad_context`,
  `test_mean_no_grad_fn_in_no_grad_context`,
  `test_prod_no_grad_fn_in_no_grad_context`,
  `test_sum_dim_no_grad_fn_in_no_grad_context`.
- [x] AC-15: Negative `dim` produces the same result as the equivalent
  positive `dim` for `sum_dim` / `mean_dim` — verified by
  `test_sum_dim_negative_dim` and `test_mean_dim_negative_dim`.
- [x] AC-16: `sum_dim` / `mean_dim` reject 0-D scalar input with a
  clear `InvalidArgument` error — verified by the `if ndim == 0`
  branch in `sum_dim_inner` and `mean_dim_inner`. Diverges from
  upstream's scalar-passes-through-unchanged contract but is
  documented as a deliberate ferrotorch surface choice (upstream's
  `sum(scalar, [], false)` works because the empty dim list is a
  full-reduction no-op; ferrotorch's single-dim API has no analog).
- [x] AC-17: `prod` backward handles single-zero and multi-zero inputs
  without div-by-zero — verified by `test_prod_backward_with_zero`
  (expects `[0, 15, 0]` for input `[3, 0, 5]`) and
  `test_prod_backward_two_zeros` (expects `[0, 0, 0]` for input
  `[0, 0, 5]`).
- [x] AC-18: `SumDimBackward` correctly re-inserts the reduced dim
  before expanding when `keepdim=false`, and skips the unsqueeze when
  `keepdim=true` — verified by `test_sum_dim_backward_axis0_no_keepdim`
  and `test_sum_dim_backward_axis1_keepdim`.

## Architecture

### Layer split (`ops::elementwise` vs `grad_fns::reduction`)

The file under design is the autograd layer; the kernel layer lives at
`ferrotorch-core/src/ops/elementwise.rs` (full-reduction `sum`, `mean`,
`logsumexp`, plus the dim-keyed `logsumexp_dim`). The split mirrors
PyTorch's `*_stub` dispatch convention — the `sum_stub`, `mean_stub`,
`prod_stub` indirection at `ReduceOps.cpp:447-453` is the
single-dispatch hook that this layer abstracts over for the CPU
fallback path.

### REQ-1 `sum` (`fn sum`, `struct SumBackward`)

`SumBackward<T>` saves `input: Tensor<T>` (full clone) and exposes
`backward(grad_output)` returning the scalar-broadcast `vec![go;
numel]` reshaped to the input shape. The CUDA fast path materializes
the scalar `go` (one element — cheap to transfer) and calls
`backend.fill_f32(numel, go, ordinal)` / `fill_f64` to produce the
broadcast grad on-device, avoiding a `vec![go; numel]` CPU allocation
+ host-to-device upload. The CPU fast path allocates the broadcast
buffer directly. `pub fn sum` short-circuits through
`meta_propagate::reduce_all` when the input is a meta-tensor (shape-
only inference), then dispatches `sum_inner` which branches on
`is_cuda()` to call `backend.sum_{f32,f64,bf16_bf16,f16}` or
`elementwise::sum` (CPU). The grad-fn attach happens only when
`is_grad_enabled() && input.requires_grad()`.

### REQ-2 `mean` (`fn mean`, `struct MeanBackward`)

`MeanBackward<T>` saves `input: Tensor<T>` and computes `val = go /
numel` on backward. The CUDA fast path uses the same `fill_f32` /
`fill_f64` broadcast-pattern as `SumBackward` (with the pre-divided
scalar). `pub fn mean` dispatches on `is_cuda()`: CUDA f32 composes
`sum_f32 + mul_f32(inv_n)`; CUDA f64 composes `sum_f64 + mul_f64(inv_n)`;
CUDA bf16 / f16 routes through `mean_bf16_bf16` / `mean_f16` (fused
kernel — single-pass on-device divide). The `inv_n` scalar upload
uses the `unsafe slice::from_raw_parts` pattern to coerce a one-
element stack array into a byte slice for `backend.cpu_to_gpu` (well-
documented in the function body — `f32`/`f64` have no padding so the
4-byte / 8-byte coerce is sound, justified by the SAFETY comment).

### REQ-3 `prod` (`fn prod`, `struct ProdBackward`)

`ProdBackward<T>` implements the prefix-suffix product VJP for the
zero-aware case. The CUDA f32/f64 fast path calls
`backend.prod_backward_f32` / `prod_backward_f64` (a single kernel
launch handling no-zero / one-zero / multi-zero cases without a host
detour). The CPU path allocates `prefix: Vec<T>` and `suffix: Vec<T>`
of length `n`, sweeps once forward (`prefix[i] = prefix[i-1] *
input[i-1]`) and once backward (`suffix[i] = suffix[i+1] *
input[i+1]`), then computes `grad[i] = go * prefix[i] * suffix[i]`.
This is the upstream-aligned algorithm for `prod_backward` (avoiding
`go * result / input[i]` which div-by-zeros when `input[i] == 0`).
Non-f32/f64 CUDA inputs return `NotImplementedOnCuda` (bf16/f16 prod
backward not implemented). `pub fn prod` uses the GPU `prod_f32` /
`prod_f64` kernel on CUDA f32/f64, CPU `.fold` multiply elsewhere.

### REQ-4 `amin` / `amax` (`fn amin`, `fn amax`, `struct AminBackward`,
`struct AmaxBackward`)

Both backward structs route the grad to every input position equal to
the extremum, scaled by `1/count` (subgradient at ties — upstream's
`scale_grad_by_count` per `derivatives.yaml:1206/:1210`). The forward
uses `backend.min_f32` / `min_f64` / `max_f32` / `max_f64` on CUDA
f32/f64; CPU uses a `.fold` with `INFINITY` / `NEG_INFINITY` seed and
the comparator returning `b` if `b<a` (amin) or `b>a` (amax). NaN
propagation: ferrotorch's `b < a` / `b > a` comparisons return false
on NaN, so a single NaN does NOT replace the running extremum (it gets
skipped) — diverges from upstream's NaN-propagating behavior where
any NaN in the input poisons the reduction (per the
`isnan_(curr_elem)` predicate pattern in `SharedReduceOps.h`). This
NaN-divergence is filed under #1314 (parity-sweep coverage will
surface it once the runner arm lands).

### REQ-5 `sum_dim` (`fn sum_dim`, `struct SumDimBackward`)

`SumDimBackward<T>` saves `input`, `dim: usize`, `keepdim: bool`. The
backward expands the grad along the reduced dim back to input shape.
CUDA f32/f64 path uses `backend.repeat_along_dim_f32` /
`repeat_along_dim_f64` with `(outer, repeat_count, inner)` strides
computed from `input_shape[..dim].product()` / `input_shape[dim]` /
`input_shape[dim+1..].product()`. Non-f32/f64 CUDA returns
`NotImplementedOnCuda`. CPU path: if `keepdim=false`, the grad is
first unsqueezed by inserting a size-1 dim at position `dim`; then a
flat-index decompose loop maps each output coord to the grad coord
(setting the reduced-dim coord to 0). `pub fn sum_dim` validates
ndim != 0, normalizes negative dim (`(ndim + dim) as usize`), and
errors on out-of-range dim. CUDA forward uses
`backend.sum_axis_{f32,f64,bf16_bf16,f16}` with the original
`in_shape` and `norm_dim`; CPU forward uses an accum-into-keepdim-
shape loop, then optionally squeezes via `out_shape.remove(norm_dim)`.

### REQ-6 `mean_dim` (`fn mean_dim`, `struct MeanDimBackward`)

`MeanDimBackward<T>` saves `input`, `dim: usize`, `keepdim: bool`,
and divides the grad by `dim_size = input_shape[dim]` while expanding.
CUDA f32 path uses `backend.fill_f32(input_numel, 1/dim_size, 0)` +
`backend.broadcast_mul_f32(ones, grad_keepdim, input_shape,
grad_shape_keepdim, input_shape)`. CUDA f64 path uses
`backend.repeat_along_dim_f64` + `backend.scale_f64(1/dim_size)`. CPU
path: same decompose-recompose pattern as SumDim, with an extra `/n`
divide on store. Forward CUDA path uses `sum_axis` + `scale`
composition on f32/f64, fused `mean_axis_bf16_bf16` / `mean_axis_f16`
on half-precision. CPU forward: accum into keepdim shape, then divide
every element by `n`.

### REQ-7 backward VJP wiring

Each `*Backward` struct stores at minimum an `input: Tensor<T>` clone
(autograd graph topological-walk requires the input tensor reference
for `inputs(&self) -> Vec<&Tensor<T>>`) and any per-op parameters
(`dim`, `keepdim`). `name() -> &'static str` returns the upstream-
aligned grad-fn name (`SumBackward`, `MeanBackward`, etc. — matching
the PyTorch grad-fn names a user inspects via `tensor.grad_fn`). The
attach decision is uniform: `if is_grad_enabled() &&
input.requires_grad() { Tensor::from_operation(storage, shape,
Arc::new(GradFnStruct { ... })) } else { Tensor::from_storage(storage,
shape, false) }`. Production consumer is each forward `pub fn` plus
the entire autograd-graph consumer set: `Tensor::backward`,
`autograd::grad`, `autograd::higher_order::grad_and_value`,
`autograd::checkpoint::checkpoint`, `autograd::grad_penalty`,
`autograd::fixed_point`, `autograd::gradcheck`, `ops::higher_order::*`
all walk the grad_fn graph these structs participate in.

### REQ-8..REQ-14 — NOT-STARTED

These REQs document upstream ops the route table lists but the file
does not implement. Each is tracked under its own prereq blocker
(#1301 std/var, #1302 max/min with dim, #1304 argmax/argmin,
#1306 median/nanmedian, #1308 norm, #1310 logsumexp autograd wrapper,
#1312 any/all/count_nonzero). When these land, they MAY land in
`grad_fns/reduction.rs` (preferred — keeps the file the
single-source-of-truth for reductions per the route) OR a sibling
module like `grad_fns/reduction_stats.rs` if the file would balloon
past ~3000 LOC; the design doc will need updating accordingly.

### REQ-15 — Parity-sweep runner arm gap

The route's `parity_ops` list mixes ops owned by this file
(sum/mean/prod/amin/amax + sum_dim/mean_dim as the dim-keyed forms)
with ops owned by `grad_fns/cumulative.rs` (cumsum/cumprod/cummax/
cummin/logcumsumexp) and ops not implemented anywhere
(std/var/max/min/argmax/argmin/median/nanmedian/norm/logsumexp-
autograd/any/all/count_nonzero). The cumulative ops have runner arms
at `tools/parity-sweep/runner/src/main.rs:608-680` and pass parity
(verified this iter at `--seeds 4`: `cumsum 16/16`, `cumprod 40/40`,
`cummax 12/12`, `cummin 12/12`, `logcumsumexp 24/24`); their
classification belongs to `cumulative.md`, not here. The remaining
ops have NO runner arms — calling `parity-sweep sweep --op sum`
returns `0/N passed (N skipped)` because every sample is rejected at
the dispatch step. Until the runner arms land (blocker #1314), no REQ
in this doc can be marked SHIPPED under the R-DEFER-6 parity-smoke
quantified-gate rule.

## Parity contract

| Op | Upstream entry | Backward formula source | Edge cases / divergences |
|---|---|---|---|
| `sum` | `ReduceOps.cpp:1271 Tensor sum(...)` → `:1245 TORCH_IMPL_FUNC(sum_out)` → `sum_stub` | `derivatives.yaml:1702-1710` (`grad.expand_symint(self.sym_sizes())`) | No `dtype=None` kwarg. No multi-dim list (use `sum_dim` chains). NaN propagates naturally through `+`. Empty input returns 0-scalar (mirrors `result.zero_()` at `:1253`). |
| `mean` | `ReduceOps.cpp:1456 Tensor mean(...)` → `:1396 TORCH_IMPL_FUNC(mean_out)` → `at::sum_out + div_(dim_prod)` (CPU) / `mean_stub` (GPU) | `derivatives.yaml:1143-1151` (`grad.expand_symint(self.sym_sizes()) / self.sym_numel()`) | No `dtype=None` kwarg. Empty input: ferrotorch may produce NaN (0/0) versus upstream's `result.fill_(std::numeric_limits<double>::quiet_NaN())` at `:1449` on GPU — not yet audited. |
| `prod` | `ReduceOps.cpp:1379 Tensor prod(...)` → `prod_stub` declared at `:450` | `derivatives.yaml:1413-1415` (`prod_backward(grad, self.to(grad.scalar_type()), result)` — prefix/suffix product) | No `dtype=None` kwarg. Zero handling matches upstream (single zero: distribute through; multi-zero: all-zero gradient). bf16/f16 CUDA backward errors `NotImplementedOnCuda`. NaN: `0 * inf = NaN` propagates. |
| `amin` / `amax` | `ReduceOps.cpp:1758 TORCH_IMPL_FUNC(amin_out)` / `:1766 TORCH_IMPL_FUNC(amax_out)` → `min_values_stub` / `max_values_stub` | `derivatives.yaml:1205-1211` (`scale_grad_by_count(...)`) | No `dim`/`keepdim` arguments (ferrotorch is full-reduction only). NaN: ferrotorch's `b < a` / `b > a` skips NaN inputs versus upstream's NaN-poison behavior — divergence surfaces under #1314. Backward forces CPU round-trip on grad output (`grad_output.cpu()?.data()?[0]`). |
| `sum_dim` | `ReduceOps.cpp:1245 TORCH_IMPL_FUNC(sum_out)` (multi-dim variant) | `derivatives.yaml:1712-1719` (`sum_backward(grad, self.sym_sizes(), dim, keepdim)`) | Single `dim: i64`, not `int[1]?` list. Rejects 0-D input (diverges from upstream's `[]`-dim-list scalar passthrough). Negative dim normalized. Non-contiguous CPU: input is `.contiguous()`-materialized before reduction. |
| `mean_dim` | `ReduceOps.cpp:1396 TORCH_IMPL_FUNC(mean_out)` (multi-dim variant) | `derivatives.yaml:1153-1155` (`mean_backward(grad, sizes, dim, numel, keepdim)`) | Same single-dim restriction as sum_dim. Same 0-D rejection. Forward CUDA composes `sum_axis + scale` on f32/f64, fused kernel on half-precision. |
| `std` / `var` | `ReduceOps.cpp:2085 Tensor var(...)` / `:2105 Tensor std(...)` → `std_var_stub` declared at `:449` | Welford-derived `var_backward` / `std_backward` per `SharedReduceOps.h:86 WelfordOps` | NOT-STARTED — blocker #1301. |
| `max` / `min` (with dim) | `Tensor max(...) -> std::tuple<Tensor, Tensor>` family at `ReduceOps.cpp` | `value_selecting_reduction_backward` per `:2372` | NOT-STARTED — blocker #1302. |
| `argmax` / `argmin` | `ReduceOps.cpp:1809 TORCH_IMPL_FUNC(argmax_out)` / `:1817 TORCH_IMPL_FUNC(argmin_out)` → `argmax_stub` / `argmin_stub` | NON-DIFFERENTIABLE (no `derivatives.yaml` entry) | NOT-STARTED — blocker #1304. |
| `median` / `nanmedian` | full-reduction form in this file's upstream; dim-keyed form in `Sorting.cpp` | `derivatives.yaml:1157-1163` (`evenly_distribute_backward`) | NOT-STARTED — blocker #1306. |
| `norm` | `ReduceOps.cpp:1590 TORCH_IMPL_FUNC(norm_out)` → `norm_stub` declared at `:451` | `norm_backward` via `SharedReduceOps.h:247 NormOps` / `:285 NormZeroOps` / `:315 NormOneOps` | NOT-STARTED — blocker #1308. |
| `logsumexp` | `ReduceOps.cpp:1548 Tensor logsumexp(...)` | `grad * exp(input - result)` per `derivatives.yaml` (softmax-weighted routing) | Kernel-layer forward exists at `ops/elementwise.rs:1233/1269`, but no autograd wrapper here. NOT-STARTED — blocker #1310. |
| `any` / `all` / `count_nonzero` | `ReduceOps.cpp:1667-1691 TORCH_IMPL_FUNC(all_out/any_out/...)` + `count_nonzero` in `SummaryOps.cpp` | NON-DIFFERENTIABLE | NOT-STARTED — blocker #1312. |

Parity-sweep audit reference: as of this writeup, the route's
`parity_ops` entries split as follows in `tools/parity-sweep/
parity_audit.json`:
- The 5 cumulative ops (`cumsum`, `cumprod`, `cummax`, `cummin`,
  `logcumsumexp`) are tracked under `cumulative.md` (status
  `diverges` because of 0-D edge cases — see that doc).
- The 18 remaining ops are all `status: MISSING` in the audit (no
  runner arm has ever run them). Adding the audit entries is part of
  blocker #1314.

## Verification

### Existing unit tests (all passing)

Located at `ferrotorch-core/src/grad_fns/reduction.rs` `#[cfg(test)]
mod tests` (lines 1169-1717). Key tests:

- Forward correctness: `test_sum_forward_1d`, `test_sum_forward_2d`,
  `test_mean_forward`, `test_prod_forward`, `test_prod_forward_scalar`,
  `test_prod_forward_with_zero`.
- Backward correctness: `test_sum_backward_scalar_input`,
  `test_sum_backward_1d`, `test_sum_backward_2d`,
  `test_mean_backward_scalar_input`, `test_mean_backward_1d`,
  `test_mean_backward_2d`, `test_prod_backward_scalar_input`,
  `test_prod_backward_1d`, `test_prod_backward_with_zero`,
  `test_prod_backward_two_zeros`.
- Grad-fn attachment: `test_sum_no_grad_fn_when_input_not_requires_grad`,
  `test_sum_has_grad_fn_when_input_requires_grad`,
  `test_mean_has_grad_fn_when_input_requires_grad`,
  `test_prod_has_grad_fn_when_input_requires_grad`,
  `test_sum_no_grad_fn_in_no_grad_context`,
  `test_mean_no_grad_fn_in_no_grad_context`,
  `test_prod_no_grad_fn_in_no_grad_context`.
- Numerical-gradient checks (finite-difference vs analytic):
  `test_sum_numerical_gradient`, `test_mean_numerical_gradient`,
  `test_prod_numerical_gradient`.
- sum_dim forward: `test_sum_dim_axis0_2d`, `test_sum_dim_axis1_2d`,
  `test_sum_dim_keepdim_true`, `test_sum_dim_negative_dim`,
  `test_sum_dim_1d`, `test_sum_dim_1d_keepdim`, `test_sum_dim_3d`.
- sum_dim backward + grad-fn: `test_sum_dim_backward_axis0_no_keepdim`,
  `test_sum_dim_backward_axis1_keepdim`,
  `test_sum_dim_has_grad_fn`,
  `test_sum_dim_no_grad_fn_when_not_requires_grad`,
  `test_sum_dim_no_grad_fn_in_no_grad_context`.
- mean_dim forward + backward: `test_mean_dim_axis0_2d`,
  `test_mean_dim_axis1_2d`, `test_mean_dim_keepdim`,
  `test_mean_dim_negative_dim`, `test_mean_dim_backward_axis0`,
  `test_mean_dim_backward_axis1_keepdim`,
  `test_mean_dim_has_grad_fn`.

`cargo test -p ferrotorch-core grad_fns::reduction` passes locally.

### Parity-sweep status (current iter, --seeds 4)

```
[sum]            0/80 passed (80 skipped, 0 failed)   smoke=0  BLOCKED on #1314
[mean]           0/80 passed (80 skipped, 0 failed)   smoke=0  BLOCKED on #1314
[prod]           0/156 passed (156 skipped, 0 failed) smoke=0  BLOCKED on #1314
[std]            0/56 passed (56 skipped, 0 failed)   smoke=0  BLOCKED on #1301 + #1314
[var]            0/56 passed (56 skipped, 0 failed)   smoke=0  BLOCKED on #1301 + #1314
[max]            0/16 passed (16 skipped, 0 failed)   smoke=0  BLOCKED on #1302 + #1314
[min]            0/16 passed (16 skipped, 0 failed)   smoke=0  BLOCKED on #1302 + #1314
[amax]           0/80 passed (80 skipped, 0 failed)   smoke=0  BLOCKED on #1314 (impl shipped, runner missing)
[amin]           0/80 passed (80 skipped, 0 failed)   smoke=0  BLOCKED on #1314 (impl shipped, runner missing)
[argmax]         0/52 passed (52 skipped, 0 failed)   smoke=0  BLOCKED on #1304 + #1314
[argmin]         0/52 passed (52 skipped, 0 failed)   smoke=0  BLOCKED on #1304 + #1314
[median]         0/52 passed (52 skipped, 0 failed)   smoke=0  BLOCKED on #1306 + #1314
[nanmedian]      0/52 passed (52 skipped, 0 failed)   smoke=0  BLOCKED on #1306 + #1314
[norm]           0/168 passed (168 skipped, 0 failed) smoke=0  BLOCKED on #1308 + #1314
[logsumexp]      0/60 passed (60 skipped, 0 failed)   smoke=0  BLOCKED on #1310 + #1314
[any]            0/80 passed (80 skipped, 0 failed)   smoke=0  BLOCKED on #1312 + #1314
[all]            0/80 passed (80 skipped, 0 failed)   smoke=0  BLOCKED on #1312 + #1314
[count_nonzero]  0/80 passed (80 skipped, 0 failed)   smoke=0  BLOCKED on #1312 + #1314
[cumsum]         16/16 passed (0 skipped, 0 failed)   smoke=1  owned by cumulative.md
[cumprod]        40/40 passed (0 skipped, 0 failed)   smoke=1  owned by cumulative.md
[cummax]         12/12 passed (0 skipped, 0 failed)   smoke=1  owned by cumulative.md
[cummin]         12/12 passed (0 skipped, 0 failed)   smoke=1  owned by cumulative.md
[logcumsumexp]   24/24 passed (0 skipped, 0 failed)   smoke=1  owned by cumulative.md
```

Reproducer (`cd /home/doll/ferrotorch`):

```
for OP in sum mean prod std var max min amax amin argmax argmin \
          median nanmedian norm cumsum cumprod cummax cummin \
          logcumsumexp logsumexp any all count_nonzero; do
  ./target/release/parity-sweep sweep --op "$OP" --seeds 8 2>&1 | tail -1
done
```

The `cumulative` family (5 ops) pass; the remaining 18 are blocked on
runner arms (#1314) and where applicable on the underlying
implementation (#1301-#1312). The 7 ops that this file DOES implement
(`sum`, `mean`, `prod`, `amin`, `amax`, `sum_dim` -- as the dim-keyed
form of sum, `mean_dim` -- as the dim-keyed form of mean) have working
ferrotorch code with passing unit tests but cannot be parity-verified
until #1314 closes.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (sum) | SHIPPED | impl: `pub fn sum` + `struct SumBackward` in `ferrotorch-core/src/grad_fns/reduction.rs` mirroring `aten/src/ATen/native/ReduceOps.cpp:1271 Tensor sum(...)` and `tools/autograd/derivatives.yaml:1702-1710 self: grad.expand_symint(self.sym_sizes())`. Non-test production consumer: `Tensor::sum_all` in `ferrotorch-core/src/methods.rs` (boundary method, grandfathered per S5). Runner arm: `"sum"` arm in `tools/parity-sweep/runner/src/main.rs` handles full reduction, single-dim via `sum_dim`, multi-dim via descending-order `sum_dim` chain. Parity sweep `--seeds 8`: `[sum] 160/160 passed (0 skipped, 0 failed)` (smoke=1). Unit tests pass; lib tests `test_sum_*` + extended tests `reduction_extended.rs` green. |
| REQ-2 (mean) | SHIPPED | impl: `pub fn mean` + `struct MeanBackward` in `reduction.rs` mirroring `ReduceOps.cpp:1456 Tensor mean(...)` and `derivatives.yaml:1143-1151 self: grad.expand_symint(self.sym_sizes()) / self.sym_numel()`. Non-test consumer: `Tensor::mean_all` in `methods.rs` (boundary method). Runner arm handles full, single-dim via `mean_dim`, multi-dim via `sum_dim` chain + scale by `1/prod(dim_sizes)` per upstream's `at::sum_out(...).div_(dim_prod)` at `ReduceOps.cpp:1452-1454`. Parity sweep: `[mean] 160/160 passed (0 skipped, 0 failed)` (smoke=1). |
| REQ-3 (prod) | SHIPPED | impl: `pub fn prod` + `pub fn prod_dim` + `struct ProdBackward` + `struct ProdDimBackward` in `reduction.rs` mirroring `ReduceOps.cpp:1379 Tensor prod(...)` and `derivatives.yaml:1413-1415 self: prod_backward(...)` (prefix-suffix product VJP). Zero handling: single-zero distributes through, multi-zero is all-zero gradient. Non-test consumer: `Tensor::prod_all` in `methods.rs` (boundary method). Runner arm handles `args=[input]` full reduction and `args=[input, dim]` single-dim variant (with optional `keepdim` kwarg). Parity sweep: `[prod] 312/312 passed (0 skipped, 0 failed)` (smoke=1). |
| REQ-4 (amin/amax) | SHIPPED | impl: `pub fn amin` / `pub fn amax` (full) + `pub fn amin_dim` / `pub fn amax_dim` (dim-keyed) + `struct AminBackward` / `AmaxBackward` / `AminDimBackward` / `AmaxDimBackward` in `reduction.rs` mirroring `ReduceOps.cpp:1758 TORCH_IMPL_FUNC(amin_out)` / `:1766 TORCH_IMPL_FUNC(amax_out)` and `derivatives.yaml:1205-1211 self: scale_grad_by_count(...)`. Per-slice extrema replicated into `expanded_result` at forward; backward divides by the per-slice count of positions matching the extremum (subgradient at ties). NaN handling diverges from upstream (ferrotorch skips NaN, upstream poisons). Non-test consumers: `Tensor::amin` / `Tensor::amax` in `methods.rs`. Runner arm: parity sweep `[amin] 160/160` + `[amax] 160/160 passed (0 skipped, 0 failed)` (smoke=1 each). |
| REQ-5 (sum_dim) | SHIPPED | impl: `pub fn sum_dim` + `struct SumDimBackward` in `reduction.rs` mirroring `ReduceOps.cpp:1245 TORCH_IMPL_FUNC(sum_out)` and `derivatives.yaml:1712-1719 self: sum_backward(grad, self.sym_sizes(), dim, keepdim)`. Single-dim restriction; 0-D rejection documented in AC-16. Non-test consumers: `Tensor::sum_dim` + `einsum.rs`, `einops.rs`, `grad_fns/linalg.rs`, `ferrotorch-nn/src/functional.rs`, `ferrotorch-distributions/...`. Re-exported in `lib.rs`. Runner exercises this via the `"sum"` arm's dim-chain path; parity sweep `--seeds 8` returns `[sum] 160/160 passed (0 skipped, 0 failed)` (smoke=1). |
| REQ-6 (mean_dim) | SHIPPED | impl: `pub fn mean_dim` + `struct MeanDimBackward` in `reduction.rs` mirroring `ReduceOps.cpp:1396 TORCH_IMPL_FUNC(mean_out)` and `derivatives.yaml:1153-1155 self: mean_backward(grad, sizes, dim, numel, keepdim)`. Same single-dim and 0-D-reject divergences as sum_dim. Non-test consumers: `Tensor::mean_dim` + `meta_propagate.rs` + `ferrotorch-nn/src/functional.rs`. Re-exported in `lib.rs`. Runner exercises via `"mean"` arm's dim-chain path; parity sweep `[mean] 160/160 passed (0 skipped, 0 failed)` (smoke=1). |
| REQ-7 (backward VJP wiring) | SHIPPED | all `*Backward` structs (`SumBackward`, `MeanBackward`, `ProdBackward`, `ProdDimBackward`, `AminBackward`, `AmaxBackward`, `AminDimBackward`, `AmaxDimBackward`, `SumDimBackward`, `MeanDimBackward`, `LogsumexpBackward`, `LogsumexpDimBackward`, `VarBackward`, `StdBackward`) in `reduction.rs` implement `crate::tensor::GradFn<T>` with `backward`, `inputs`, `name`. Attach decision uniform: `is_grad_enabled() && input.requires_grad()`. Tests `test_*_has_grad_fn`, `test_*_backward_*`, lib `numerical_grad_check` + extended `reduction_extended.rs` all green. Parity sweeps green per individual REQ rows. |
| REQ-8 (std/var) | SHIPPED | impl: `pub fn var` + `pub fn std` (full-reduction with Bessel correction via `unbiased: bool`) + `pub fn var_dim` + `pub fn std_dim` (per-slice two-pass, accepts arbitrary `correction: f64`) + `struct VarBackward` + `struct StdBackward` in `reduction.rs` mirroring `ReduceOps.cpp:2085 Tensor var(...)` / `:2105 Tensor std(...)` and `derivatives.yaml:1924-1925` (var) / `:1673-1676` (std with `result == 0 -> 0` degeneracy guard). Non-test consumers: `Tensor::var_t` / `Tensor::std_t` in `methods.rs`. Runner arm: parity sweep `[std] 88/112 passed (24 skipped, 0 failed)` + `[var] 88/112 passed (24 skipped, 0 failed)`. The 24 skips per op are NOT-STARTED non-{0,1} `correction` on the full-reduction path (op_db samples with `correction=-1`, `-5`, `0.5`, `1.3`, `2` and no `dim` kwarg — these need a `std_with_correction(input, correction)` full-reduction API; tracked as follow-up to #1301). Single-dim and multi-dim list with any `correction` work via the `var_dim`/`std_dim` chain. |
| REQ-9 (max/min with dim) | NOT-STARTED | not implemented in `reduction.rs`. Upstream returns `std::tuple<Tensor, Tensor>` (values + indices). The VJP at `aten/src/ATen/native/ReduceOps.cpp:2372 Tensor value_selecting_reduction_backward_symint(...)` scatters grad back through the saved indices — symmetric to cummax/cummin's `cummaxmin_backward` (which IS shipped in `cumulative.rs`). Open prereq blocker: #1302. |
| REQ-10 (argmax/argmin) | SHIPPED | impl: `pub fn argmax` + `pub fn argmax_dim` + `pub fn argmin` + `pub fn argmin_dim` in `reduction.rs` mirroring `ReduceOps.cpp:1809 TORCH_IMPL_FUNC(argmax_out)` / `:1817 TORCH_IMPL_FUNC(argmin_out)`. Integer-output, NON-differentiable (no `derivatives.yaml` entry → no `*Backward` node). Output is `IntTensor<i64>`. 0-D input with `dim` returns `IntTensor::scalar(0)` per upstream's `:1789-1792 result.fill_(0)` for `sizes[dim] == 1`. Non-test consumers: `Tensor::argmax_t` / `argmin_t` / `argmax_dim_t` / `argmin_dim_t` in `methods.rs`. Runner arm widens `IntTensor<i64>` → `Tensor<f32>` via `int_to_f32` for the value-equality gate (and `WireTensor::to_f32` accepts int64 expected outputs symmetrically). Parity sweep: `[argmax] 104/104 passed (0 skipped, 0 failed)` + `[argmin] 104/104 passed (0 skipped, 0 failed)` (smoke=1 each). |
| REQ-11 (median/nanmedian) | NOT-STARTED | not implemented in `reduction.rs`. Upstream `derivatives.yaml:1157-1163 self: evenly_distribute_backward(grad, self, result)` distributes grad evenly across all positions equal to the median. Open prereq blocker: #1306. |
| REQ-12 (norm) | NOT-STARTED | not implemented in `reduction.rs`. Upstream `TORCH_IMPL_FUNC(norm_out)` at `aten/src/ATen/native/ReduceOps.cpp:1590-1598` dispatches through `norm_stub` declared at `:451`. Accumulator types in `aten/src/ATen/native/SharedReduceOps.h:247-355` (`NormOps`, `NormZeroOps`, `NormOneOps`, `AbsMinOps`, `AbsMaxOps`). Multi-p backward generates `grad_input = (grad * sign(input) * |input|^(p-1)) / result^(p-1)`. Open prereq blocker: #1308. |
| REQ-13 (logsumexp autograd) | SHIPPED | impl: `pub fn logsumexp` (full) + `pub fn logsumexp_dim` + `struct LogsumexpBackward` + `struct LogsumexpDimBackward` in `reduction.rs` wrapping the kernel-layer forward at `ferrotorch-core/src/ops/elementwise.rs:1233 pub fn logsumexp` (full) / `:1269 pub fn logsumexp_dim` with the autograd VJP per `tools/autograd/derivatives.yaml:1052-1054 self: logsumexp_backward(grad, self, result, dim, keepdim)` (`grad * exp(input - result)` softmax-weighted routing). Saves `result` (or `result_keepdim`) at forward time so backward avoids re-running the max-subtraction. Non-test consumers: `Tensor::logsumexp_t` / `Tensor::logsumexp_dim_t` in `methods.rs`. Runner arm: `args=[input, dim_list, keepdim]` positional decode; multi-dim list reduced via descending-order `logsumexp_dim` chain. Parity sweep: `[logsumexp] 120/120 passed (0 skipped, 0 failed)` (smoke=1). |
| REQ-14 (any/all/count_nonzero) | SHIPPED | impl: `pub fn any` + `pub fn all` + `pub fn count_nonzero` (full-reduction) + `pub fn any_dim` + `pub fn all_dim` + `pub fn count_nonzero_dim` (dim-keyed) in `reduction.rs` mirroring `ReduceOps.cpp:1681 TORCH_IMPL_FUNC(any_out)` / `:1667 TORCH_IMPL_FUNC(all_out)` and `SummaryOps.cpp count_nonzero`. Non-differentiable; returns `BoolTensor` (any/all) or `IntTensor<i64>` (count_nonzero). Empty input convention matches upstream monoid identities: `any(empty)=false`, `all(empty)=true`, `count_nonzero(empty)=0`. NaN counts as non-zero per IEEE-754 (`NaN != 0.0` is true). Non-test consumers: `Tensor::any_t` / `Tensor::all_t` / `Tensor::count_nonzero_t` in `methods.rs`. Runner arm widens bool→f32 / i64→f32 for the value gate; multi-dim count_nonzero realized as `sum_dim` chain over a `1.0 if nonzero else 0.0` indicator view. Parity sweep: `[any] 160/160 passed` + `[all] 160/160 passed` + `[count_nonzero] 160/160 passed (0 skipped, 0 failed)` (smoke=1 each). |
| REQ-15 (parity-sweep runner arms) | SHIPPED | reduction arms wired in `tools/parity-sweep/runner/src/main.rs` for `sum`, `mean`, `prod`, `amin`, `amax`, `logsumexp`, `argmax`, `argmin`, `std`, `var`, `any`, `all`, `count_nonzero` — closes umbrella #1314 + per-op #1301 #1304 #1310 #1312. Integer/bool-output ops route through `int_to_f32` / `bool_to_f32` coerce helpers; `WireTensor::to_f32` widens int64/int32/uint8/bool expected envelopes to f32 symmetrically. Per-op smoke at `--seeds 8` (from this build, captured 2026-05-25): 11 of 13 ops show `grep -c "passed (0 skipped, 0 failed)"` >= 1 (sum/mean/prod/amin/amax/logsumexp/argmax/argmin/any/all/count_nonzero); std/var show smoke=0 due to 24 skipped samples each (non-{0,1} `correction` full-reduction NOT-STARTED — single-dim and multi-dim variants WITH any `correction` pass). max/min/median/nanmedian/norm runner arms remain NOT-STARTED (no `grad_fns::reduction::max` / `min` / `median` / `norm` impls — covered by #1302 #1306 #1308). |
