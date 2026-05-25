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
  NaN with `cur_idx` pinned to the first-NaN position. Implemented at
  `ops/cumulative.rs:251-282` with `>=` tie-break and the matching NaN
  predicate; seed is `cur = in_data[base]` (the first element) mirroring
  upstream's `T1 out = c10::load(self_data)` at `:815`.

- REQ-4: `cummin_forward(input, dim)` — returns `CumExtremeResult { values,
  indices }` symmetric to cummax. Mirrors `cummin_helper_cpu` at
  `aten/src/ATen/native/ReduceOps.cpp:867-873` dispatching
  `cummax_cummin_helper<..., std::less_equal<scalar_t>>`. Same NaN-poison
  predicate as REQ-3. Implemented at `ops/cumulative.rs:315-345` with
  `<=` tie-break and the matching NaN predicate.

- REQ-5: `logcumsumexp_forward(input, dim)` — numerically stable
  `out[..., i, ...] = log(sum(exp(input[..., 0..=i, ...])))` along `dim`.
  Mirrors `logcumsumexp_cpu_kernel` at
  `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:117-136` (uses
  `_log_add_exp_helper` for stability). ferrotorch uses a two-pass
  running-max trick at `ops/cumulative.rs:382-410`: first pass computes
  `maxes[i] = max(input[..0..=i])`, second pass accumulates
  `exp(in_data[idx] - m)` and writes `m + ln(acc)`. When the running max
  changes between iterations, the accumulator is rescaled by
  `(prev_max - m).exp()` to preserve numerical correctness. GPU dispatch
  via `logcumsumexp_{f32,f64}`.

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
  layer at `grad_fns/cumulative.rs:89, :339, :376, :393, :534`
  short-circuits 0-D inputs into `cumulative_scalar_identity` /
  `cumextreme_scalar_identity` *before* reaching the kernel, mirroring
  the upstream `if (self.dim() == 0) { result.fill_(self); }` branch at
  `ReduceOps.cpp:501-504, 847-849, 886-888`. Direct callers of the kernel
  layer (none today outside `grad_fns/cumulative.rs`) would see the
  rejection.

- REQ-8: `CumExtremeResult<T: Float> { values: Tensor<T>, indices:
  Vec<usize> }` public struct — Rust analog of upstream's
  `std::tuple<Tensor, Tensor>` return type from `cummax`/`cummin` (see
  `Tensor cummax(const Tensor& self, int64_t dim)` at `ReduceOps.cpp:860-865
  return std::make_tuple(std::move(values), std::move(indices))`). The
  `indices` field uses `Vec<usize>` rather than a second `Tensor<T>`
  because ferrotorch does not yet have an `i64`-dtype tensor class; the
  upstream indices tensor is `at::kLong` per `:838 self.options().dtype(at::kLong)`.
  Re-exported from the crate root as `ferrotorch_core::CumExtremeResult`
  at `ferrotorch-core/src/lib.rs:173`.

## Acceptance Criteria

- [x] AC-1: `cumsum_forward` produces the correct prefix-sum for a 1D
  tensor — exercised indirectly through `grad_fns::cumulative::cumsum`'s
  forward tests at `ferrotorch-core/src/grad_fns/cumulative.rs:376-443`
  (`test_cumsum_1d`, `test_cumsum_2d_dim0`, `test_cumsum_2d_dim1`,
  `test_cumsum_negative_dim`, `test_cumsum_3d`) and the
  `[cumsum] 32/32 passed (0 skipped, 0 failed)` parity-sweep at
  `--seeds 8`.
- [x] AC-2: `cumprod_forward` produces the correct prefix-product for a
  2D tensor — exercised through `grad_fns::cumulative::cumprod`'s forward
  tests at `:512-551` and `[cumprod] 80/80 passed (0 skipped, 0 failed)`.
- [x] AC-3: `cummax_forward` returns the correct running-max values for a
  monotonic sequence — exercised through `grad_fns::cumulative::cummax`'s
  forward tests at `:618-646` (`test_cummax_1d`, `test_cummax_2d_dim1`)
  and the `[cummax] 24/24 passed (0 skipped, 0 failed)` parity-sweep
  (Option A — values-only return per `runner/src/main.rs:500 "cummax" =>`,
  per blocker #1230).
- [x] AC-4: `cummin_forward` returns the correct running-min values —
  exercised through `:652-678` (`test_cummin_1d`, `test_cummin_2d_dim0`)
  and `[cummin] 24/24 passed (0 skipped, 0 failed)`.
- [x] AC-5: `logcumsumexp_forward` is numerically stable for large
  inputs — exercised by `test_logcumsumexp_numerical_stability` at
  `grad_fns/cumulative.rs:719-736` (inputs at scale ~1000 stay finite)
  and `[logcumsumexp] 48/48 passed (0 skipped, 0 failed)`.
- [x] AC-6: `reverse_cumsum` produces the upper-triangular reverse-cumsum
  expected by the cumsum backward — exercised by
  `test_cumsum_backward_1d`, `test_cumsum_backward_2d_dim0`,
  `test_cumsum_backward_numerical` at
  `grad_fns/cumulative.rs:449-484, 880-913`.
- [x] AC-7: `validate_dim` rejects out-of-range dim with
  `InvalidArgument` — exercised by `test_cumsum_dim_out_of_bounds` at
  `grad_fns/cumulative.rs:830-835` (reaches the kernel because the
  autograd 0-D fast path only short-circuits on `ndim == 0`, not on
  out-of-range `dim`).
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
   `logcumsumexp` it's the two-pass running-max algorithm.
6. `Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)`
   returns a fresh tensor with `requires_grad=false` (the autograd layer
   attaches the grad-fn after this returns).

### REQ-3 / REQ-4: cummax/cummin kernels (`cummax_forward`, `cummin_forward`)

Same scaffold but returns `CumExtremeResult { values, indices }`. The CPU
loop at `:240-262` (cummax) and `:319-341` (cummin) carries:

```
let mut cur_max = T::neg_infinity();
let mut cur_idx = 0usize;
for i in 0..dim_size {
    let idx = base + i * inner;
    if in_data[idx] > cur_max {   // strict — diverges from upstream
        cur_max = in_data[idx];
        cur_idx = i;
    }
    out_vals[idx] = cur_max;
    out_idxs[idx] = cur_idx;
}
```

The strict `>` (resp. `<` for cummin) is the divergence from upstream's
`std::greater_equal` / `std::less_equal` tracked under blocker #1231.

The GPU fast path (`:200-228`, `:281-308`) is more involved: the kernel
returns two GPU handles (values + indices). For f32 the indices are
stored as f32 bits; for f64 the PTX converter rewrites
`st.global.f32` → `st.global.f64` (per inline reference to #787 in the
source) so the indices buffer is f64-width. Both are read back to host
and cast to `Vec<usize>` for the `CumExtremeResult.indices` field.

### REQ-6: `reverse_cumsum` helper (`:109-125`)

```rust
pub fn reverse_cumsum<T: Float>(data: &[T], shape: &[usize], dim: usize) -> Vec<T>
```

Operates on raw `&[T]` + `&[usize]` + a non-negative `dim` (caller is
expected to have already normalised dim via `normalize_axis`). Mirrors
the upstream three-step `w.flip(dim).cumsum(dim).flip(dim)` at
`ReduceOps.cpp:527-529` but unrolled into a single reverse triple-loop
that accumulates from `i = dim_size-1` down to `0`. Used by
`grad_fns/cumulative.rs:60` (`CumsumBackward::backward`) and `:495`
(`LogcumsumexpBackward::backward`).

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

### REQ-8: `CumExtremeResult<T>` (`:181-184`)

```rust
#[derive(Debug)]
pub struct CumExtremeResult<T: Float> {
    pub values: Tensor<T>,
    pub indices: Vec<usize>,
}
```

Public struct re-exported at `lib.rs:173`. The `indices: Vec<usize>`
rather than a second `Tensor<T>` is the Rust-ecosystem-better-fit
deviation (R-DEV-7) — ferrotorch does not yet expose an integer-dtype
tensor and the consumer (`grad_fns::cumulative::cummax` / `einops.rs`)
only needs a flat usize vector along the scan axis. When CummaxBackward
ships (blocker #1231) the indices will be saved into the grad-fn struct
as `Vec<usize>` to drive a scatter_add VJP.

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
| `cummax` | `cummax_helper_cpu` at `ReduceOps.cpp:828-834` dispatching `cummax_cummin_helper<..., std::greater_equal>` at `:811-826` | `cummax_forward` at `ops/cumulative.rs:191-262` | **Tie-break DIVERGES** — upstream `std::greater_equal` picks LATER index; ferrotorch strict `>` picks EARLIER. **NaN DIVERGES** — upstream's `isnan_(curr_elem)` branch at `:819` propagates NaN forever; ferrotorch's strict `>` returns false on NaN so prior non-NaN max is retained. Both folded into #1231. |
| `cummin` | `cummin_helper_cpu` at `ReduceOps.cpp:867-873` dispatching `cummax_cummin_helper<..., std::less_equal>` | `cummin_forward` at `ops/cumulative.rs:272-341` | Symmetric divergences to cummax (strict `<` vs `<=`, NaN). #1231. |
| `logcumsumexp` | `logcumsumexp_cpu_kernel` at `cpu/ReduceOpsKernel.cpp:117-136` (uses `_log_add_exp_helper`) | `logcumsumexp_forward` at `ops/cumulative.rs:352-414` | Numerical stability via two-pass running-max trick (`maxes[i] = max(input[0..=i])` then `exp(in - m)` accumulator); `(-inf).exp() == 0`, `0.ln() == -inf` so `logcumsumexp([-inf, x]) == [-inf, x]` matches upstream. |

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
(`runner/src/main.rs:500 "cummax" =>` returns `result.values` only);
indices-parity remains tracked under blocker #1231.

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
| REQ-1 (cumsum_forward) | SHIPPED | impl: `cumsum_forward` at `ferrotorch-core/src/ops/cumulative.rs:66-104` mirrors `cumsum_cpu_kernel` at `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:79-96` (registered as `cumsum_stub` per `:564 REGISTER_DISPATCH(cumsum_stub, &cumsum_cpu_kernel)`, dispatched by `TORCH_IMPL_FUNC(cumsum_out)` at `aten/src/ATen/native/ReduceOps.cpp:511-517`). Non-test production consumer: `ferrotorch-core/src/grad_fns/cumulative.rs:93 let result = cumsum_forward(input, dim)?;` inside `pub fn cumsum` (which itself is reached by `Tensor::cumsum_t` at `methods.rs:282` per the sibling doc REQ-1). Indirect parity: `[cumsum] 32/32 passed (0 skipped, 0 failed)`. |
| REQ-2 (cumprod_forward) | SHIPPED | impl: `cumprod_forward` at `ferrotorch-core/src/ops/cumulative.rs:135-173` mirrors `cumprod_cpu_kernel` at `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:98-115` (registered as `cumprod_stub` per `:563`). Non-test production consumer: `ferrotorch-core/src/grad_fns/cumulative.rs:343 let result = cumprod_forward(input, dim)?;` inside `pub fn cumprod`. Indirect parity: `[cumprod] 80/80 passed (0 skipped, 0 failed)`. |
| REQ-3 (cummax_forward) | SHIPPED | impl: `cummax_forward` at `ferrotorch-core/src/ops/cumulative.rs:201-291` mirrors `cummax_helper_cpu` at `aten/src/ATen/native/ReduceOps.cpp:828-834` dispatching `cummax_cummin_helper<scalar_t, int64_t, std::greater_equal<scalar_t>>` at `:811-826`. CPU kernel at `ops/cumulative.rs:251-282` uses `>=` tie-break (mirrors `:832 std::greater_equal<scalar_t>` — later index wins on ties) with NaN-poison predicate `isnan(curr) || (!isnan(cur) && curr >= cur)` mirroring `:819`. Seed `cur = in_data[base]` mirrors upstream's `T1 out = c10::load(self_data)` at `:815`. **Non-test production consumer**: `ferrotorch-core/src/grad_fns/cumulative.rs:476 let result = cummax_forward(input, dim)?;` inside `pub fn cummax`, ultimately reached by `ferrotorch-core/src/einops.rs:796` (`EinopsReduction::Max` arm). Indirect parity: `[cummax] 24/24 passed (0 skipped, 0 failed)` (Option A — values-only return per runner). Tie-break + NaN verified by `test_cummax_backward_tie` and `test_cummax_forward_nan_propagates` at `grad_fns/cumulative.rs` (live-traced torch 2.11.0 indices). Closes #1231. |
| REQ-4 (cummin_forward) | SHIPPED | impl: `cummin_forward` at `ferrotorch-core/src/ops/cumulative.rs:301-371` mirrors `cummin_helper_cpu` at `aten/src/ATen/native/ReduceOps.cpp:867-873` dispatching `cummax_cummin_helper<..., std::less_equal>`. CPU kernel at `ops/cumulative.rs:315-345` uses `<=` tie-break (mirrors `:871 std::less_equal<scalar_t>`) with the same NaN-poison predicate. **Non-test production consumer**: `ferrotorch-core/src/grad_fns/cumulative.rs:508 let result = cummin_forward(input, dim)?;` inside `pub fn cummin`, reached by `ferrotorch-core/src/einops.rs:802` (`EinopsReduction::Min` arm). Indirect parity: `[cummin] 24/24 passed (0 skipped, 0 failed)`. Tie-break verified by updated `test_cummin_1d` (input `[3, 1, 4, 1, 5]` → indices `[0, 1, 1, 3, 3]`) and `test_cummin_backward_tie`. Closes #1231. |
| REQ-5 (logcumsumexp_forward) | SHIPPED | impl: `logcumsumexp_forward` at `ferrotorch-core/src/ops/cumulative.rs:352-414` mirrors `logcumsumexp_cpu_kernel` at `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:117-136` (registered as `logcumsumexp_stub` per `:565`, dispatched via `_logcumsumexp_out_cpu` at `aten/src/ATen/native/ReduceOps.cpp:470-473`). Two-pass running-max numerical-stability trick matches the `_log_add_exp_helper` contract. Non-test production consumer: `ferrotorch-core/src/grad_fns/cumulative.rs:541 let result = logcumsumexp_forward(input, dim)?;` inside `pub fn logcumsumexp`. Indirect parity: `[logcumsumexp] 48/48 passed (0 skipped, 0 failed)`. Numerical stability verified by `test_logcumsumexp_numerical_stability` at `grad_fns/cumulative.rs:719-736`. |
| REQ-6 (reverse_cumsum helper) | SHIPPED | impl: `reverse_cumsum` at `ferrotorch-core/src/ops/cumulative.rs:109-125` mirrors `static Tensor reversed_cumsum(const Tensor& w, int64_t dim) { return w.flip(dim).cumsum(dim).flip(dim); }` at `aten/src/ATen/native/ReduceOps.cpp:527-529`. Two non-test production consumers: `ferrotorch-core/src/grad_fns/cumulative.rs:60 let grad_data = reverse_cumsum(go_data, shape, self.dim);` (CumsumBackward::backward) and `ferrotorch-core/src/grad_fns/cumulative.rs:495 let rev = reverse_cumsum(&product, shape, self.dim);` (LogcumsumexpBackward::backward). End-to-end exercised by `test_cumsum_backward_numerical` at `grad_fns/cumulative.rs:880-913` and `test_logcumsumexp_backward_1d` at `:742-779`. |
| REQ-7 (validate_dim) | SHIPPED | impl: `validate_dim` at `ferrotorch-core/src/ops/cumulative.rs:49-56` wraps `crate::shape::normalize_axis` mirroring `maybe_wrap_dim` at `c10/core/WrapDimMinimal.h:34-39` (used by upstream's `impl_func_cum_ops` at `aten/src/ATen/native/ReduceOps.cpp:506`, `cummax_out` at `:851`, `cummin_out` at `:890`, and `_logcumsumexp_out_cpu` implicitly via the kernel). Five non-test production consumers (one per `*_forward`): `ops/cumulative.rs:67, :136, :195, :276, :353`. Defense-in-depth 0-D rejection is the intentional deviation from upstream's `wrap_scalar=true` — the autograd layer's 0-D fast paths at `grad_fns/cumulative.rs:89, 339, 376, 393, 534` short-circuit before reaching the kernel. Exercised through `test_cumsum_dim_out_of_bounds` at `grad_fns/cumulative.rs:830-835`. |
| REQ-8 (CumExtremeResult struct) | SHIPPED | impl: `pub struct CumExtremeResult<T: Float> { pub values: Tensor<T>, pub indices: Vec<usize> }` at `ferrotorch-core/src/ops/cumulative.rs:181-184` mirrors the `std::tuple<Tensor, Tensor>` return of `Tensor cummax(...)` at `aten/src/ATen/native/ReduceOps.cpp:860-865` (per R-DEV-7, using `Vec<usize>` rather than an integer tensor since ferrotorch lacks i64-dtype tensors). Re-exported at `ferrotorch-core/src/lib.rs:173 pub use ops::cumulative::CumExtremeResult;`. Non-test production consumer: `ferrotorch-core/src/grad_fns/cumulative.rs:375, 392, 417, 434` all construct or return this type inside `pub fn cummax` / `pub fn cummin` / `cumextreme_scalar_identity`. |
