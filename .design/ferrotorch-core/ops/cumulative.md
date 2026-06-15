# Cumulative (scan) kernel forwards

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/native/ReduceOps.cpp
  - aten/src/ATen/native/cpu/ReduceOpsKernel.cpp
  - c10/core/WrapDimMinimal.h
-->

## Summary

`ferrotorch-core/src/ops/cumulative.rs` is the non-autograd kernel layer for
the five PyTorch scan ops (`cumsum`, `cumprod`, `cummax`, `cummin`,
`logcumsumexp`). Each `*_forward` function pairs a CPU triple-loop
implementation with a GPU `f32/f64` fast path that delegates to
`gpu_dispatch::gpu_backend()`. The autograd wrappers that attach `*Backward`
nodes live in `grad_fns/cumulative.rs` and are the natural production
consumer of every public function this file exports. The layer split
mirrors PyTorch's `cumsum_stub` / `cumprod_stub` / `logcumsumexp_stub`
dispatchers (`ReduceOps.cpp:460-462`) and the `_cummax_helper` /
`_cummin_helper` underscore-prefixed private dispatchers (`ReduceOps.cpp:828-834,
867-873`) vs the user-facing autograd-aware `cumsum`/`cummax`/... namespace
functions.

## Requirements

- REQ-1: `cumsum_forward(input, dim)` — forward `out[..., i, ...] =
  sum(input[..., 0..=i, ...])` along `dim`. Mirrors `cumsum_cpu_kernel` at
  `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:79-96` (the
  `REGISTER_DISPATCH(cumsum_stub, &cumsum_cpu_kernel)` registration at
  `:564` is what `TORCH_IMPL_FUNC(cumsum_out)` at `ReduceOps.cpp:511-517`
  dispatches via `impl_func_cum_ops`). Validates `dim` via `validate_dim`,
  factorises the shape into `(outer, dim_size, inner)`, runs a
  `Float`-typed scalar accumulator (`acc += in_data[idx]`) and writes the
  running sum. GPU fast path for f32/f64 routes through
  `gpu_backend().cumsum_{f32,f64}` returning a GPU-resident tensor without
  CPU round-trip. CUDA + non-{f32,f64} returns
  `FerrotorchError::NotImplementedOnCuda { op: "cumsum" }`.

- REQ-2: `cumprod_forward(input, dim)` — forward `out[..., i, ...] =
  prod(input[..., 0..=i, ...])` along `dim`. Mirrors `cumprod_cpu_kernel`
  at `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:98-115`. Same structure
  as REQ-1 but with `<T as num_traits::One>::one()` initial accumulator and
  `acc = acc * in_data[idx]`. GPU dispatch via `cumprod_{f32,f64}` mirrors
  the cumsum path.

- REQ-3: `cummax_forward(input, dim)` — returns `CumExtremeResult { values,
  indices }` where `values[..., i, ...] = max(input[..., 0..=i, ...])` and
  `indices[..., i, ...]` is the position along `dim` at which each running
  maximum was attained. Mirrors `cummax_helper_cpu` at
  `aten/src/ATen/native/ReduceOps.cpp:828-834` dispatching the templated
  `cummax_cummin_helper<T1, T2, std::greater_equal<scalar_t>>` at `:811-826`.
  Upstream's tie-break operator is `std::greater_equal` — on equal values
  the LATER index wins. NaN propagation: upstream's update predicate
  `isnan_(curr_elem) || (!isnan_(out) && op(curr_elem, out))` at `:819`
  means once a NaN appears in the prefix, every subsequent position is
  NaN with `cur_idx` pinned to the first-NaN position. Implemented in
  `pub fn cummax_forward in ops/cumulative.rs` with `>=` tie-break and
  the matching NaN predicate; seed is `cur = in_data[base]` (the first
  element) mirroring upstream's `T1 out = c10::load(self_data)` at
  `aten/src/ATen/native/ReduceOps.cpp:815`.

- REQ-4: `cummin_forward(input, dim)` — returns `CumExtremeResult { values,
  indices }` symmetric to cummax. Mirrors `cummin_helper_cpu` at
  `aten/src/ATen/native/ReduceOps.cpp:867-873` dispatching
  `cummax_cummin_helper<..., std::less_equal<scalar_t>>`. Same NaN-poison
  predicate as REQ-3. Implemented in `pub fn cummin_forward in
  ops/cumulative.rs` with `<=` tie-break and the matching NaN predicate.

- REQ-5: `logcumsumexp_forward(input, dim)` — numerically stable
  `out[..., i, ...] = log(sum(exp(input[..., 0..=i, ...])))` along `dim`.
  Mirrors `logcumsumexp_cpu_kernel` at
  `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:117-136` (uses
  `_log_add_exp_helper` for stability). ferrotorch ports the helper as
  `fn log_add_exp in ops/cumulative.rs` (`_log_add_exp_helper` at
  `aten/src/ATen/native/cpu/LogAddExp.h:22-33`: min/max with NaN
  propagation, `log1p(exp(min - max)) + max` when `min != max ||
  isfinite(min)`, and pass-through of `x` for equal infinities so the
  scan never computes `inf - inf` — CORE-133 / #1827) and folds each
  element through it sequentially with a `-inf` initial accumulator in
  `pub fn logcumsumexp_forward in ops/cumulative.rs`. GPU dispatch
  via `logcumsumexp_{f32,f64}` (the PTX kernels still lack the
  equal-infinity guard — tracked as #1942).

- REQ-6: `reverse_cumsum(data, shape, dim)` helper — `out[..., i, ...] =
  sum(input[..., i..dim_size, ...])` (the suffix-sum sibling of cumsum).
  Mirrors `static Tensor reversed_cumsum(const Tensor& w, int64_t dim) {
  return w.flip(dim).cumsum(dim).flip(dim); }` at
  `aten/src/ATen/native/ReduceOps.cpp:527-529`. Used by the cumsum
  backward (`grad_input = reverse_cumsum(grad_output, dim)`) and the
  logcumsumexp backward (`grad_input[i] = exp(input[i]) *
  reverse_cumsum(grad_output * exp(-output))[i]`). Operates on raw `&[T]`
  + shape rather than a `Tensor` to keep the autograd backward path
  zero-cost when only the underlying data is needed.

- REQ-7: `validate_dim(ndim, dim, op_name)` shared utility — converts a
  signed `i64` dim (possibly negative for trailing-axis indexing) into a
  non-negative `usize` axis index, erroring on out-of-range or scalar
  (0-D) input. Mirrors `maybe_wrap_dim(dim, self.dim())` at
  `c10/core/WrapDimMinimal.h:34-39` (the helper PyTorch uses universally
  in `impl_func_cum_ops`, `cummax_out`, `cummin_out`, and
  `_logcumsumexp_out_cpu`). **Diverges intentionally from upstream's
  `maybe_wrap_dim`**: ferrotorch rejects `ndim == 0` with an explicit
  `InvalidArgument` error rather than upstream's `wrap_scalar = true` path
  that maps any dim to 0. This is **defense in depth** — the autograd
  layer's 0-D fast paths in `pub fn cumsum`, `pub fn cumprod`, `pub fn
  cummax`, `pub fn cummin`, and `pub fn logcumsumexp` (all in
  `grad_fns/cumulative.rs`) short-circuit 0-D inputs into
  `cumulative_scalar_identity` / `cumextreme_scalar_identity` *before*
  reaching the kernel, mirroring the upstream `if (self.dim() == 0) {
  result.fill_(self); }` branch at
  `aten/src/ATen/native/ReduceOps.cpp:501-504, 847-849, 886-888`. Direct
  callers of the kernel layer (none today outside `grad_fns/cumulative.rs`)
  would see the rejection.

- REQ-8: `CumExtremeResult<T: Float> { values: Tensor<T>,
  indices_tensor: IntTensor<i64>, indices: Vec<usize> }` public struct —
  Rust analog of upstream's `std::tuple<Tensor, Tensor>` return type from
  `cummax`/`cummin` (see `Tensor cummax(const Tensor& self, int64_t dim)`
  at `ReduceOps.cpp:860-865 return std::make_tuple(std::move(values),
  std::move(indices))`). The authoritative indices result is an
  `IntTensor<i64>`, matching upstream's `at::kLong` tensor allocated from
  `self.options()` so it lives on the input device. The legacy
  `indices: Vec<usize>` is a host cache populated for CPU/scalar results
  only; non-scalar CUDA results leave it empty to avoid an implicit D2H
  transfer. Re-exported from the crate root as `pub use
  ops::cumulative::CumExtremeResult in lib.rs`.

## Acceptance Criteria

- [x] AC-1: `cumsum_forward` produces the correct prefix-sum for a 1D
  tensor — exercised indirectly through `pub fn cumsum in
  grad_fns/cumulative.rs`'s forward tests (`fn test_cumsum_1d in
  grad_fns/cumulative.rs`, `test_cumsum_2d_dim0`, `test_cumsum_2d_dim1`,
  `test_cumsum_negative_dim`, `test_cumsum_3d`) and the
  `[cumsum] 32/32 passed (0 skipped, 0 failed)` parity-sweep at
  `--seeds 8`.
- [x] AC-2: `cumprod_forward` produces the correct prefix-product for a
  2D tensor — exercised through `pub fn cumprod in
  grad_fns/cumulative.rs`'s forward tests (`fn test_cumprod_1d in
  grad_fns/cumulative.rs`, `test_cumprod_2d_dim0`, `test_cumprod_2d_dim1`)
  and `[cumprod] 80/80 passed (0 skipped, 0 failed)`.
- [x] AC-3: `cummax_forward` returns the correct running-max values for a
  monotonic sequence — exercised through `pub fn cummax in
  grad_fns/cumulative.rs`'s forward tests (`fn test_cummax_1d in
  grad_fns/cumulative.rs`, `test_cummax_2d_dim1`) and the
  `[cummax] 24/24 passed (0 skipped, 0 failed)` parity-sweep (Option A —
  values-only return per the `"cummax" =>` arm in
  `tools/parity-sweep/runner/src/main.rs`, per blocker #1230).
- [x] AC-4: `cummin_forward` returns the correct running-min values —
  exercised through `pub fn cummin in grad_fns/cumulative.rs`'s forward
  tests (`fn test_cummin_1d in grad_fns/cumulative.rs`,
  `test_cummin_2d_dim0`) and `[cummin] 24/24 passed (0 skipped, 0 failed)`.
- [x] AC-5: `logcumsumexp_forward` is numerically stable for large
  inputs — exercised by `fn test_logcumsumexp_numerical_stability in
  grad_fns/cumulative.rs` (inputs at scale ~1000 stay finite) and
  `[logcumsumexp] 48/48 passed (0 skipped, 0 failed)`.
- [x] AC-6: `reverse_cumsum` produces the upper-triangular reverse-cumsum
  expected by the cumsum backward — exercised by `fn test_cumsum_backward_1d
  in grad_fns/cumulative.rs`, `test_cumsum_backward_2d_dim0`, and
  `test_cumsum_backward_numerical`.
- [x] AC-7: `validate_dim` rejects out-of-range dim with
  `InvalidArgument` — exercised by `fn test_cumsum_dim_out_of_bounds in
  grad_fns/cumulative.rs` (reaches the kernel because the autograd 0-D
  fast path only short-circuits on `ndim == 0`, not on out-of-range
  `dim`).
- [x] AC-8: `validate_dim` rejects scalar (0-D) input — direct kernel
  callers see the error; the autograd layer short-circuits before
  reaching this branch per the
  `cumulative_scalar_identity`/`cumextreme_scalar_identity` early-outs
  introduced by #1233. Defense-in-depth coverage means a direct test on
  the kernel would still see the rejection, but the autograd-layer fast
  path bypasses it for the user-facing surface.
- [x] AC-9: `cummax_forward` tie-breaking matches upstream — on equal
  values the LATER index wins (upstream `std::greater_equal` at
  `ReduceOps.cpp:832`). Implemented at `ops/cumulative.rs:251-282`
  using `>=` after the NaN-poison short-circuit; verified by
  `test_cummax_backward_tie` (input `[1, 2, 2, 3]` → indices `[0, 1, 2, 3]`).
- [x] AC-10: `cummin_forward` tie-breaking matches upstream — on equal
  values the LATER index wins (upstream `std::less_equal` at
  `ReduceOps.cpp:871`). Implemented at `ops/cumulative.rs:315-345`
  using `<=`; verified by updated `test_cummin_1d` (input
  `[3, 1, 4, 1, 5]` → indices `[0, 1, 1, 3, 3]`) and
  `test_cummin_backward_tie`.
- [x] AC-11: `cummax_forward` / `cummin_forward` propagate NaN through
  the running max/min per upstream `cummax_cummin_helper` at
  `ReduceOps.cpp:819 if(isnan_(curr_elem) || (!isnan_(out) &&
  op(curr_elem, out)))`. Implemented at `ops/cumulative.rs:264-280`
  (cummax) and `:329-339` (cummin) with the matching update predicate;
  verified by `test_cummax_forward_nan_propagates` (input
  `[1.0, NaN, 3.0, 4.0]` → values `[1, NaN, NaN, NaN]`, indices
  `[0, 1, 1, 1]`).

## Architecture

### Layer split (kernel vs autograd)

This file is the kernel layer; the autograd wrapper at
`ferrotorch-core/src/grad_fns/cumulative.rs` (914 LOC) is the only
production consumer today. The split mirrors PyTorch's:
- `cumsum_stub` / `cumprod_stub` / `logcumsumexp_stub` (dispatch declarations
  at `ReduceOps.cpp:460-462`) registered by `cumsum_cpu_kernel` /
  `cumprod_cpu_kernel` / `logcumsumexp_cpu_kernel` at
  `cpu/ReduceOpsKernel.cpp:79, 98, 117`.
- `cummax_helper_cpu` / `cummin_helper_cpu` at `ReduceOps.cpp:828, 867`
  dispatching to the templated `cummax_cummin_helper<scalar_t, int64_t,
  Op>` at `:811-826`.

The user-facing autograd-aware `cumsum`/`cumprod`/`cummax`/`cummin`/
`logcumsumexp` namespace functions in PyTorch attach `cummaxmin_backward` /
`cumsum_backward` / etc. per `tools/autograd/derivatives.yaml:521-539`;
ferrotorch's `grad_fns::cumulative` is the byte-equivalent of that layer.

### Stride factorization (`dim_strides`, `:41-46`)

`dim_strides(shape, dim) -> (outer, dim_size, inner)` factorises the flat
storage layout for a scan along `dim`:
- `outer` = product of dims before `dim`
- `dim_size` = `shape[dim]`
- `inner` = product of dims after `dim`

Element `(o, i, k)` has flat index `o * dim_size * inner + i * inner + k`.
This is the ferrotorch analog of PyTorch's `TensorIterator` reorder
pattern at `cpu_cum_base_kernel` (`cpu/ReduceOpsKernel.cpp:35-77`), which
folds non-scan dims into a single batch axis before dispatching the
per-axis scan lambda.

### REQ-1 / REQ-2 / REQ-5: scan kernels (`cumsum_forward`, `cumprod_forward`, `logcumsumexp_forward`)

All three follow the same scaffold (`ops/cumulative.rs:66-104, 135-173,
352-414`):
1. `validate_dim(input.ndim(), dim, op_name)` — REQ-7.
2. `dim_strides(shape, norm_dim)` — stride factorisation.
3. GPU fast path: if `input.is_cuda() && (is_f32::<T>() || is_f64::<T>())`,
   call `gpu_backend().<op>_{f32,f64}(handle, outer, dim_size, inner)` and
   return a GPU-resident tensor.
4. CUDA + non-{f32,f64} → `NotImplementedOnCuda { op }` early-out.
5. CPU triple-loop over `(outer, inner, dim_size)` accumulating into the
   output. For `cumsum`/`cumprod` the accumulator is a `T` scalar; for
   `logcumsumexp` it's a `-inf`-seeded scalar folded through
   `fn log_add_exp in ops/cumulative.rs` (the `_log_add_exp_helper` port).
6. `Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)`
   returns a fresh tensor with `requires_grad=false` (the autograd layer
   attaches the grad-fn after this returns).

### REQ-3 / REQ-4: cummax/cummin kernels (`cummax_forward`, `cummin_forward`)

Same scaffold but returns `CumExtremeResult { values, indices_tensor,
indices }`. The CPU loop carries the same NaN-poisoning and later-tie
predicate as upstream:

```
let mut cur = in_data[base];
let mut cur_idx = 0usize;
for i in 0..dim_size {
    let idx = base + i * inner;
    let curr = in_data[idx];
    if curr.is_nan() || (!cur.is_nan() && curr >= cur) {
        cur = curr;
        cur_idx = i;
    }
    out_vals[idx] = cur;
    out_idxs[idx] = cur_idx;
}
```

Cummin uses the symmetric `curr <= cur` predicate. The GPU fast path returns
two GPU handles (values + i64 indices). The i64 indices handle is wrapped as
`IntTensor<i64>` and kept resident; `CumExtremeResult.indices` is not
populated on non-scalar CUDA outputs. Callers that need a host copy must use
the explicit `indices_host()` readback method or `indices_tensor.to(Cpu)`.

### REQ-6: `reverse_cumsum` helper (`:109-125`)

```rust
pub fn reverse_cumsum<T: Float>(data: &[T], shape: &[usize], dim: usize) -> Vec<T>
```

Operates on raw `&[T]` + `&[usize]` + a non-negative `dim` (caller is
expected to have already normalised dim via `normalize_axis`). Mirrors
the upstream three-step `w.flip(dim).cumsum(dim).flip(dim)` at
`ReduceOps.cpp:527-529` but unrolled into a single reverse triple-loop
that accumulates from `i = dim_size-1` down to `0`. Used by
`CumsumBackward::backward` and `LogcumsumexpBackward::backward` in
`grad_fns/cumulative.rs`.

### REQ-7: `validate_dim` (`:49-56`)

```rust
fn validate_dim(ndim: usize, dim: i64, op_name: &str) -> FerrotorchResult<usize> {
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("{op_name}: cannot operate on a scalar (0-D) tensor"),
        });
    }
    normalize_axis(dim as isize, ndim)
}
```

Private (`fn`, not `pub fn`) — called only from this file's five
`*_forward` entry points. Defense-in-depth: the autograd layer
short-circuits 0-D inputs *before* reaching the kernel (post-#1233), so
the `ndim == 0` rejection is unreachable through the user-facing
`grad_fns::cumulative::cumsum` etc. surface. A direct external caller
(none exist today) would still see the rejection.

### REQ-8: `pub struct CumExtremeResult in ops/cumulative.rs`

```rust
#[derive(Debug)]
pub struct CumExtremeResult<T: Float> {
    pub values: Tensor<T>,
    pub indices_tensor: IntTensor<i64>,
    pub indices: Vec<usize>,
}
```

Public struct re-exported as `pub use ops::cumulative::CumExtremeResult
in lib.rs`. `indices_tensor` is the PyTorch-equivalent `LongTensor`
result. `indices: Vec<usize>` is retained as a CPU/scalar host cache for
existing Rust callers, but CUDA forwards intentionally leave it empty so
the kernel layer never performs an implicit GPU→CPU transfer. Autograd saves
both fields: CPU backward uses the host cache, while CUDA backward scatters
through the resident `indices_tensor`.

### `is_f32` / `is_f64` type-id helpers (`:17-25`)

Inline `TypeId::of::<T>() == TypeId::of::<f32>()` checks that gate the
GPU fast paths. The `Float` trait bound permits both f32 and f64
monomorphisations; the GPU backend only carries f32/f64 entry points, so
the type-id branch is mandatory.

## Parity contract

The route's `parity_ops` list is **`[]`** — this kernel-layer file owns no
direct parity-sweep entries. The five op names (`cumsum`, `cumprod`,
`cummax`, `cummin`, `logcumsumexp`) belong to the sibling autograd-layer
route at `ferrotorch-core/src/grad_fns/cumulative.rs` (per
`tooling/translate-routes.toml:431-435`), which is the natural production
consumer of every public function in this file. The kernel's correctness
is therefore validated through the autograd layer's parity-sweep runs.

| Upstream op | Upstream entry | Kernel-layer function | Edge cases |
|---|---|---|---|
| `cumsum` | `cumsum_cpu_kernel` at `cpu/ReduceOpsKernel.cpp:79-96` (registered as `cumsum_stub`, dispatched by `TORCH_IMPL_FUNC(cumsum_out)` at `ReduceOps.cpp:511-517`) | `cumsum_forward` at `ops/cumulative.rs:66-104` | NaN propagates (float arithmetic); ±Inf preserved; non-contiguous: CPU iterates by computed flat indices so stride doesn't matter; GPU path requires contiguous storage. |
| `cumprod` | `cumprod_cpu_kernel` at `cpu/ReduceOpsKernel.cpp:98-115` | `cumprod_forward` at `ops/cumulative.rs:135-173` | Zeros propagate (`0 * x = 0` once seen); NaN propagates; non-contiguous: same as cumsum. |
| `cummax` | `cummax_helper_cpu` at `aten/src/ATen/native/ReduceOps.cpp:828-834` dispatching `cummax_cummin_helper<..., std::greater_equal>` at `aten/src/ATen/native/ReduceOps.cpp:811-826` | `pub fn cummax_forward in ops/cumulative.rs` | Tie-break and NaN behavior mirror upstream's `isnan_(curr_elem)` branch at `aten/src/ATen/native/ReduceOps.cpp:819` (NaN propagates forever; later index wins on ties via `std::greater_equal`). |
| `cummin` | `cummin_helper_cpu` at `aten/src/ATen/native/ReduceOps.cpp:867-873` dispatching `cummax_cummin_helper<..., std::less_equal>` | `pub fn cummin_forward in ops/cumulative.rs` | Symmetric to cummax (`std::less_equal` later-index ties; NaN propagation). |
| `logcumsumexp` | `logcumsumexp_cpu_kernel` at `cpu/ReduceOpsKernel.cpp:117-136` (uses `_log_add_exp_helper` at `cpu/LogAddExp.h:22-33`) | `logcumsumexp_forward in ops/cumulative.rs` | Numerical stability via the sequential `fn log_add_exp in ops/cumulative.rs` fold (`log1p(exp(min - max)) + max`, NaN propagation, equal infinities pass through — CORE-133 / #1827); `logcumsumexp([-inf, x]) == [-inf, x]` and `logcumsumexp([x, inf]) == [x, inf]` match upstream. |

End-to-end verification flows through the autograd layer's parity-sweep
runs (see `.design/ferrotorch-core/grad_fns/cumulative.md` Verification
section). The kernel layer carries no per-file `parity_ops` because every
parity op routes through the autograd-attaching wrapper as the runner's
entry point (`tools/parity-sweep/runner/src/main.rs:471-541` dispatches
`grad_fns::cumulative::cumsum / cumprod / cummax / cummin / logcumsumexp`,
which in turn calls the `*_forward` kernels here).

Parity-sweep audit status (read via `tools/parity-sweep/parity_audit.json`):
the five entries (`cumsum`, `cumprod`, `cummax`, `cummin`, `logcumsumexp`)
are currently marked `diverges` in the audit JSON — the
sibling grad_fns design doc cites the post-#1233 fresh
`32/32 / 80/80 / 24/24 / 24/24 / 48/48 passed` results that should
upgrade them to `verified`. Updating the audit JSON is part of the
sibling autograd-layer iter, not this kernel-layer doc.

## Verification

### Indirect coverage via autograd-layer tests

The kernel layer has **no in-file `#[cfg(test)] mod tests`** of its own;
correctness is exercised entirely through `grad_fns/cumulative.rs:355-913`
(20+ tests) and the parity-sweep:

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

The integer smoke grep count (`grep -c "passed (0 skipped, 0 failed)"`)
is `1` per op. The cummax/cummin runs are the runner's Option A
(the `"cummax" =>` arm in `tools/parity-sweep/runner/src/main.rs` returns
`result.values` only); indices-parity remains tracked under blocker #1231.

### Per-crate test command

```bash
cargo test -p ferrotorch-core --lib grad_fns::cumulative
```

Note: a kernel-layer-direct test module would tighten the gauntlet (e.g.
exercising `cummax_forward` on a tie input to pin the EARLIER-index
divergence as a failing test). That is the natural ride-along when #1231
is worked — both the tie-break fix at `ops/cumulative.rs:247, :326` AND
a kernel-direct characterization test would land in the same fixer iter.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (cumsum_forward) | SHIPPED | impl: `cumsum_forward in ferrotorch-core/src/ops/cumulative.rs` mirrors `cumsum_cpu_kernel` at `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:79-96` (registered as `cumsum_stub` per `:564 REGISTER_DISPATCH(cumsum_stub, &cumsum_cpu_kernel)`, dispatched by `TORCH_IMPL_FUNC(cumsum_out)` at `aten/src/ATen/native/ReduceOps.cpp:511-517`). Non-test production consumer: `cumsum_forward in ferrotorch-core/src/grad_fns/cumulative.rs let result = cumsum_forward(input, dim)?;` inside `pub fn cumsum` (which itself is reached by `Tensor::cumsum_t` at `cumsum in methods.rs` per the sibling doc REQ-1). Indirect parity: `[cumsum] 32/32 passed (0 skipped, 0 failed)`. |
| REQ-2 (cumprod_forward) | SHIPPED | impl: `cumprod_forward` at `ferrotorch-core/src/ops/cumulative.rs:135-173` mirrors `cumprod_cpu_kernel` at `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:98-115` (registered as `cumprod_stub` per `:563`). Non-test production consumer: `ferrotorch-core/src/grad_fns/cumulative.rs:343 let result = cumprod_forward(input, dim)?;` inside `pub fn cumprod`. Indirect parity: `[cumprod] 80/80 passed (0 skipped, 0 failed)`. |
| REQ-3 (cummax_forward) | SHIPPED | impl: `pub fn cummax_forward in ops/cumulative.rs` mirrors `cummax_helper_cpu` at `aten/src/ATen/native/ReduceOps.cpp:828-834` dispatching `cummax_cummin_helper<scalar_t, int64_t, std::greater_equal<scalar_t>>` at `aten/src/ATen/native/ReduceOps.cpp:811-826`. CPU kernel uses `>=` tie-break (mirrors `std::greater_equal<scalar_t>` at `aten/src/ATen/native/ReduceOps.cpp:832` — later index wins on ties) with NaN-poison predicate `isnan(curr) || (!isnan(cur) && curr >= cur)` mirroring `aten/src/ATen/native/ReduceOps.cpp:819`. Seed `cur = in_data[base]` mirrors upstream's `T1 out = c10::load(self_data)` at `aten/src/ATen/native/ReduceOps.cpp:815`. **Non-test production consumer**: the `let result = cummax_forward(input, dim)?;` call inside `pub fn cummax in grad_fns/cumulative.rs`, ultimately reached by the `EinopsReduction::Max` arm in `einops.rs`. Indirect parity: `[cummax] 24/24 passed (0 skipped, 0 failed)` (Option A — values-only return per runner). Tie-break + NaN verified by `fn test_cummax_backward_tie in grad_fns/cumulative.rs` and `fn test_cummax_forward_nan_propagates in grad_fns/cumulative.rs` (live-traced torch 2.11.0 indices). Closes #1231. |
| REQ-4 (cummin_forward) | SHIPPED | impl: `pub fn cummin_forward in ops/cumulative.rs` mirrors `cummin_helper_cpu` at `aten/src/ATen/native/ReduceOps.cpp:867-873` dispatching `cummax_cummin_helper<..., std::less_equal>`. CPU kernel uses `<=` tie-break (mirrors `std::less_equal<scalar_t>` at `aten/src/ATen/native/ReduceOps.cpp:871`) with the same NaN-poison predicate. **Non-test production consumer**: the `let result = cummin_forward(input, dim)?;` call inside `pub fn cummin in grad_fns/cumulative.rs`, reached by the `EinopsReduction::Min` arm in `einops.rs`. Indirect parity: `[cummin] 24/24 passed (0 skipped, 0 failed)`. Tie-break verified by updated `fn test_cummin_1d in grad_fns/cumulative.rs` (input `[3, 1, 4, 1, 5]` → indices `[0, 1, 1, 3, 3]`) and `fn test_cummin_backward_tie in grad_fns/cumulative.rs`. Closes #1231. |
| REQ-5 (logcumsumexp_forward) | SHIPPED | impl: `pub fn logcumsumexp_forward in ops/cumulative.rs` mirrors `logcumsumexp_cpu_kernel` at `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:117-136` (registered as `logcumsumexp_stub` per `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:565`, dispatched via `_logcumsumexp_out_cpu` at `aten/src/ATen/native/ReduceOps.cpp:470-473`). Two-pass running-max numerical-stability trick matches the `_log_add_exp_helper` contract. Non-test production consumer: the `let result = logcumsumexp_forward(input, dim)?;` call inside `pub fn logcumsumexp in grad_fns/cumulative.rs`. Indirect parity: `[logcumsumexp] 48/48 passed (0 skipped, 0 failed)`. Numerical stability verified by `fn test_logcumsumexp_numerical_stability in grad_fns/cumulative.rs`. |
| REQ-6 (reverse_cumsum helper) | SHIPPED | impl: `pub fn reverse_cumsum in ops/cumulative.rs` mirrors `static Tensor reversed_cumsum(const Tensor& w, int64_t dim) { return w.flip(dim).cumsum(dim).flip(dim); }` at `aten/src/ATen/native/ReduceOps.cpp:527-529`. Two non-test production consumers: the `let grad_data = reverse_cumsum(go_data, shape, self.dim);` call inside `impl GradFn for CumsumBackward in grad_fns/cumulative.rs`, and the `let rev = reverse_cumsum(&product, shape, self.dim);` call inside `impl GradFn for LogcumsumexpBackward in grad_fns/cumulative.rs`. End-to-end exercised by `fn test_cumsum_backward_numerical in grad_fns/cumulative.rs` and `fn test_logcumsumexp_backward_1d in grad_fns/cumulative.rs`. |
| REQ-7 (validate_dim) | SHIPPED | impl: `fn validate_dim in ops/cumulative.rs` wraps `crate::shape::normalize_axis` mirroring `maybe_wrap_dim` at `c10/core/WrapDimMinimal.h:34-39` (used by upstream's `impl_func_cum_ops` at `aten/src/ATen/native/ReduceOps.cpp:506`, `cummax_out` at `aten/src/ATen/native/ReduceOps.cpp:851`, `cummin_out` at `aten/src/ATen/native/ReduceOps.cpp:890`, and `_logcumsumexp_out_cpu` implicitly via the kernel). Five non-test production consumers (one per `*_forward`): `pub fn cumsum_forward`, `pub fn cumprod_forward`, `pub fn cummax_forward`, `pub fn cummin_forward`, and `pub fn logcumsumexp_forward` (all in `ops/cumulative.rs`). Defense-in-depth 0-D rejection is the intentional deviation from upstream's `wrap_scalar=true` — the autograd layer's 0-D fast paths in `pub fn cumsum`, `pub fn cumprod`, `pub fn cummax`, `pub fn cummin`, and `pub fn logcumsumexp` (all in `grad_fns/cumulative.rs`) short-circuit before reaching the kernel. Exercised through `fn test_cumsum_dim_out_of_bounds in grad_fns/cumulative.rs`. |
| REQ-8 (CumExtremeResult struct) | SHIPPED | impl: `pub struct CumExtremeResult in ops/cumulative.rs` (with fields `values: Tensor<T>`, `indices_tensor: IntTensor<i64>`, and legacy `indices: Vec<usize>`) mirrors the `std::tuple<Tensor, Tensor>` return of `Tensor cummax(...)` at `aten/src/ATen/native/ReduceOps.cpp:860-865`; PyTorch allocates the indices tensor as `self.options().dtype(at::kLong)`, so CUDA indices stay CUDA-resident. Ferrotorch now keeps non-scalar CUDA indices resident in `indices_tensor` and leaves the host cache empty unless the caller explicitly requests `indices_host()` / `indices_tensor.to(Cpu)`. Re-exported as `pub use ops::cumulative::CumExtremeResult in lib.rs`. Non-test production consumers: `pub fn cummax in grad_fns/cumulative.rs`, `pub fn cummin in grad_fns/cumulative.rs`, and `fn cumextreme_scalar_identity in grad_fns/cumulative.rs` all construct or return this type. |
