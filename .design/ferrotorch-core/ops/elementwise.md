# Elementwise Op Primitives

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/
  - c10/
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/ops/elementwise.rs` is the CPU primitive layer
underneath the autograd-tracking `grad_fns::arithmetic` and
`grad_fns::transcendental` modules. It ships the `simd_*` SIMD-
specialised f32/f64 paths, the `fast_*` broadcast-aware variants,
the `unary_map` / `binary_map` / `scalar_map` generic walkers, and
the `sum` / `sum_axis` / `mean` / `nansum` / `nanmean` / `logsumexp`
reductions. The functions here are NOT differentiable on their own —
they are the building-blocks `grad_fns::*` modules use under
`#[no_grad]` after capturing operands for backward.

## Requirements

- REQ-1: SIMD-specialised f32/f64 elementwise — `simd_add_f32`,
  `simd_mul_f32`, `simd_exp_f32`, `simd_log_f32`, `simd_sqrt_f32`,
  and their f64 siblings. Use the `wide` crate / portable SIMD when
  available. Mirror the per-element ops in
  `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp` /
  `UnaryOpsKernel.cpp` for the CPU backend.
- REQ-2: Generic broadcast-aware `fast_*` ops — `fast_add`,
  `fast_mul`, `fast_sub`, `fast_div`, `fast_exp`, `fast_log`,
  `fast_sigmoid`, `fast_tanh`, `fast_sin`, `fast_cos`. Each handles
  the broadcast walk (with stride iteration) and dispatches to the
  SIMD path for f32/f64 same-shape inputs.
- REQ-3: Generic walkers — `unary_map`, `binary_map`, `scalar_map`
  apply a `Fn(T) -> T` / `Fn(T, T) -> T` to each element with
  broadcast support.
- REQ-4: Reductions — `sum`, `sum_axis`, `mean`, `nansum`, `nanmean`,
  `logsumexp`, `logsumexp_dim`. Each returns a CPU tensor (no GPU
  path at this layer; the autograd-tracking `grad_fns::reduction`
  has the GPU paths via `gpu_dispatch`).
- REQ-5: `logsumexp` numerical stability — subtracts the max before
  exponentiating; an infinite max is masked to zero for the subtraction
  and restored after the log, mirroring `logsumexp_out_impl` at
  `aten/src/ATen/native/ReduceOps.cpp:1512-1521`
  (`maxes_squeezed.masked_fill_(maxes_squeezed.abs() == INFINITY, 0)`)
  so a `+inf` element yields `+inf` and an all-`-inf` input/slice yields
  `-inf` instead of NaN (CORE-134 / #1828; same mechanism in
  `logsumexp_dim`). Empty input returns `-inf`.
- REQ-6: NaN-aware sum/mean — `nansum` treats NaN as 0 and skips it
  from the count; `nanmean` returns NaN when every element is NaN.
  Mirrors `torch.nansum` / `torch.nanmean`.

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib ops::elementwise`
  passes.
- [x] AC-2: `fast_add` with same-shape f32 inputs takes the SIMD path
  (verified via `simd_add_f32` direct call sites in
  `grad_fns::arithmetic`).
- [x] AC-3: `nansum(&[1.0, NaN, 3.0])` returns `4.0`.
- [x] AC-4: `nanmean(&[NaN, NaN, NaN])` returns `NaN`.
- [x] AC-5: `logsumexp` of `[-inf, -inf, -inf]` returns `-inf`
  (not NaN — matches torch's documented behaviour).
- [x] AC-6: `mean(empty)` errors with `NotImplementedOnCuda` is
  rejected only for CUDA inputs; CPU empty inputs return the
  natural `NaN` / `0/0` shape.

## Architecture

`simd_*` ops at `ops/elementwise.rs:67-144` are the type-specific
f32/f64 wrappers around the SIMD-accelerated arithmetic. They take
`Tensor<f32>` / `Tensor<f64>` directly (not generic) because the SIMD
lane width depends on the type.

`fast_add` / `fast_mul` / `fast_sub` / `fast_div` at `:139-540`
broadcast over the input shapes — they compute the broadcast shape
via `crate::shape::broadcast_shapes`, build per-axis stride pairs,
and walk the output index, mapping each output coord into per-operand
flat indices with size-1 broadcast collapsing. For same-shape f32/f64
inputs they take a fast path delegating to `simd_*`.

`fast_exp` / `fast_log` / `fast_sigmoid` / `fast_tanh` / `fast_sin` /
`fast_cos` at `:610-928` are unary versions: same-shape f32/f64 take
the SIMD path; other dtypes route to `unary_map(input, |v| v.exp())`
etc.

`unary_map` / `binary_map` / `scalar_map` at `:930-1108` are the
generic broadcast walkers. They accept arbitrary `Fn(T) -> T` /
`Fn(T, T) -> T` callbacks and run a host-side index loop.

Reductions at `:1113-1342`:

- `sum` walks `data` once accumulating into a `T::zero()` accumulator.
- `sum_axis` decomposes flat indices into per-axis coords, skipping
  the reduced axis when computing the output flat index.
- `mean` is `sum / numel` — CUDA inputs error with
  `NotImplementedOnCuda` (the autograd-tracking `mean_dim` in
  `grad_fns::reduction` has GPU support).
- `nansum` filters NaN out; `nanmean` returns NaN when count == 0.
- `logsumexp` is the numerically stable `log(sum(exp(input - max))) +
  max` pattern; `logsumexp_dim` is the per-axis variant.

**Non-test consumers**: `crate::grad_fns::arithmetic::add` / `mul` /
`sub` / `div` at `grad_fns/arithmetic.rs:950`, `1726`, etc., call
`fast_add` / `fast_mul` / `fast_sub` / `crate::ops::elementwise::fast_div`
directly. `crate::grad_fns::transcendental::exp` / `log` /
`sin` / `cos` at `grad_fns/transcendental.rs:281,384,491,568` call
`fast_exp`, `fast_log`, `fast_sin`, `fast_cos`. `grad_fns::activation::sigmoid`
/ `tanh` at `grad_fns/activation.rs:928,988` call `fast_sigmoid` /
`fast_tanh`. The `unary_map` / `scalar_map` / `binary_map` are used
by `grad_fns::arithmetic::scale_tensor` (`grad_fns/arithmetic.rs:38`
import) and by every `crate::special::*` op (the polynomial families,
gamma family, etc.). These are the production consumers — they live
in the autograd path, not in `#[cfg(test)]` blocks.

## Parity contract

`parity_ops = []`. The parity-sweep coverage runs through the
autograd-tracking `grad_fns::arithmetic` (add/sub/mul/div/pow/etc.)
which calls into this module. By transitivity, every passing
`add`/`sub`/`mul`/`div`/`exp`/`log`/`sin`/`cos`/`tanh`/`sigmoid`
parity-sweep at `--seeds 8` is also evidence of the underlying
`fast_*` correctness here.

## Verification

`cargo test -p ferrotorch-core --lib ops::elementwise` covers the
reductions + a sampling of the broadcast paths. Transitive parity-
sweep coverage via `add` (88/88 passed), `mul` (72/72 passed),
`sub` (88/88 passed), `div` (72/72 passed), `exp` / `log` /
`sin` / `cos` parity ops.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `simd_add_f32` at `ops/elementwise.rs:67` etc.; non-test consumer: `grad_fns::arithmetic::add_inner` calls into `fast_add` at `grad_fns/arithmetic.rs:950`, which routes to `simd_add_f32` for same-shape f32 inputs |
| REQ-2 | SHIPPED | impl: `fast_add`/`fast_mul`/`fast_sub`/`fast_div` at `ops/elementwise.rs:185,266,351,437`; `fast_exp`/`fast_log`/`fast_sigmoid`/`fast_tanh`/`fast_sin`/`fast_cos` at `:610,649,742,782,830,878`; non-test consumer: `grad_fns::arithmetic::add_inner` at `grad_fns/arithmetic.rs:950` (`fast_add`), `grad_fns::arithmetic::mul_inner` at `:1726` (`fast_mul`), `grad_fns::transcendental::exp` at `grad_fns/transcendental.rs:281` (`fast_exp`), etc. |
| REQ-3 | SHIPPED | impl: `unary_map`/`binary_map`/`scalar_map` at `ops/elementwise.rs:924,944,1027`; non-test consumer: `grad_fns::arithmetic::scale_tensor` at `grad_fns/arithmetic.rs:1123` calls `scalar_map`; every `crate::special::*` op uses `unary_map` (`special.rs:676` etc.) |
| REQ-4 | SHIPPED | impl: `sum`/`sum_axis`/`mean`/`nansum`/`nanmean`/`logsumexp`/`logsumexp_dim` at `ops/elementwise.rs:1091,1101,1150,1167,1185,1211,1255`; non-test consumer: `grad_fns::reduction::sum` chains into `ops::elementwise::sum` for the CPU fallback path. The reduction surface is re-exported transitively via `grad_fns::reduction::*` |
| REQ-5 | SHIPPED | impl: `logsumexp` numerical-stability flow at `ops/elementwise.rs:1233-1262`; non-test consumer: `grad_fns::reduction::logsumexp` at `grad_fns/reduction.rs` invokes this for the CPU path |
| REQ-6 | SHIPPED | impl: `nansum`/`nanmean` at `ops/elementwise.rs:1189,1207`; non-test consumer: re-exported via `ferrotorch_core::ops::elementwise::{nansum, nanmean}` public path |
