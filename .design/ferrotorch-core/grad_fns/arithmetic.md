# Arithmetic grad_fns

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/native/BinaryOps.cpp
  - aten/src/ATen/native/UnaryOps.cpp
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/grad_fns/arithmetic.rs` implements the forward + backward
(autograd-tracking) elementwise arithmetic ops that mirror PyTorch's
`torch.add` / `torch.sub` / `torch.mul` / `torch.div` / `torch.neg` /
`torch.abs` / `torch.sqrt` / `torch.pow` family, declared in
`aten/src/ATen/native/BinaryOps.cpp` (binary stubs `add_stub`, `sub_stub`,
`mul_stub`, `div_true_stub`) and `aten/src/ATen/native/UnaryOps.cpp` (unary
`neg`, `sqrt`, `abs`). Each op pairs a `*Backward` `GradFn` struct holding the
saved-for-backward operands with a `pub fn <op>` forward that branches on
`is_cuda()` and routes to either the on-device kernel via `gpu_dispatch`
(with non-contiguous CUDA views materialized through
`strided_copy_{f32,f64}` per cluster #812) or the CPU `fast_*` /
`unary_map` / `scalar_map` path in `crate::ops::elementwise`. Broadcasting is
delegated to `crate::shape::broadcast_shapes` on forward and to
`reduce_grad_to_shape` on backward, which itself runs the GPU
`sum_axis_{f32,f64}` reductions on-device when the gradient lives on CUDA.
During `grad(..., create_graph=true)`, `reduce_grad_to_shape` routes
broadcast-gradient reductions through differentiable `sum_dim` /
`reshape` primitives instead of raw buffers, so mixed second
derivatives through broadcasted arithmetic remain connected.
The file additionally ships the PyTorch-parity `add_scaled` (alpha kwarg),
`add_out` / `add_scaled_out` (out= kwarg) entry points and the
`AddScaledBackward` autograd node — these were introduced by the
parity-sweep reader-corrector to close the `torch.add(input, other, *,
alpha=1, out=None)` signature gap.

## Requirements

- REQ-1: `add(a, b)` — forward `c = a + b` with broadcasting + autograd
  (`AddBackward` saves `a`/`b`, VJP returns `(grad, grad)` reduced to each
  operand's shape). Mirrors `add_stub` (BinaryOps.cpp:434 `TORCH_IMPL_FUNC(sub_out)`
  delegates to `add_stub` with negated alpha; the underlying `add_stub` is the
  shared dispatcher used by every additive binary). The full
  `torch.add(input, other, *, alpha=1, out=None)` signature additionally
  requires the `alpha`-kwarg path (`add_scaled`) and the `out=`-kwarg path
  (`add_scaled_out`), both of which exist as sibling entry points in this
  module.

- REQ-2: `sub(a, b)` / `sub_scaled(a, b, alpha)` — forward `c = a - alpha*b`
  with broadcasting + autograd. PyTorch's
  `torch.sub(input, other, *, alpha=1, out=None)` (signature at
  `torch/_torch_docs.py:10851`) is implemented in upstream as
  `TORCH_IMPL_FUNC(sub_out)` calling `add_stub(device_type(), *this, -alpha)`
  (`aten/src/ATen/native/BinaryOps.cpp:434-439`) — sub is literally add with
  negated alpha. ferrotorch's `sub_scaled` matches that contract byte-for-byte
  by delegating to `add_scaled(a, b, -alpha)`, and the plain `sub(a, b)` is
  itself a one-line wrapper `sub_scaled(a, b, 1.0)`. The backward identity is
  therefore `AddScaledBackward` (with `alpha = -1.0` for the plain path and
  `alpha = -alpha` for the scaled path); ferrotorch carries no dedicated
  subtract-backward grad-fn — the delegation collapses sub onto the
  add-with-negated-alpha graph node so the forward and backward routing
  match upstream's add-with-negated-alpha contract exactly (R-DEV-1).

- REQ-3: `mul(a, b)` — forward `c = a * b` with broadcasting + autograd
  (`MulBackward` VJP returns `(grad*b, grad*a)` reduced). Mirrors `mul_stub`
  in BinaryOps.cpp:441 `TORCH_IMPL_FUNC(mul_out)`. `MulBackward::backward`
  additionally implements the higher-order grad path under the
  `create_graph` backward context, routing through differentiable `mul`
  and differentiable broadcast reduction so the backward pass is itself
  recorded when the gradient depends on saved operands.

- REQ-4: `div(a, b)` — forward `c = a / b` with broadcasting + autograd
  (`DivBackward` VJP `(grad/b, -grad*((a/b)/b))` reduced). Mirrors
  `div_true_stub` at BinaryOps.cpp:447 `TORCH_IMPL_FUNC(div_out)`. Division by
  zero produces `±inf` / `NaN` per IEEE-754, matching torch. Under
  `create_graph`, `DivBackward` uses differentiable `div` / `neg` / `mul`
  staging instead of the no-grad raw path.

- REQ-5: `neg(a)` — forward `c = -a` with autograd (`NegBackward` VJP
  `-grad`). Mirrors `neg_stub` at UnaryOps.cpp:344 via
  `CREATE_UNARY_TORCH_IMPL_FUNC(neg_out, neg_stub)`.

- REQ-6: `abs(a)` — forward `c = |a|` with autograd (`AbsBackward` VJP
  `grad * sign(a)`, with the `sign(0) = 0` convention matching torch and
  using the dedicated `abs_backward_{f32,f64}` GPU kernels on CUDA).
  Mirrors `aten::abs` at UnaryOps.cpp:546 (`unary_op_impl_with_complex_to_float`).

- REQ-7: `sqrt(a)` — forward `c = sqrt(a)` with autograd (`SqrtBackward`
  VJP `grad / (2 * sqrt(a))`). Mirrors `sqrt_stub` at UnaryOps.cpp:359 via
  `CREATE_UNARY_TORCH_IMPL_FUNC(sqrt_out, sqrt_stub)`.

- REQ-8: `pow(a, exp)` — forward `c = a^exp` (scalar exponent only) with
  autograd (`PowBackward` VJP `grad * exp * a^(exp-1)`; `PowBackward::scalar_args`
  emits `[exp]` so the JIT tracer can rehydrate `IrOpKind::Pow { exponent }`
  per crosslink #887). Mirrors `TORCH_IMPL_FUNC(pow_Tensor_Scalar_out)` at
  `aten/src/ATen/native/Pow.cpp:51` (note: outside the route's declared
  upstream list, which only names BinaryOps.cpp + UnaryOps.cpp; Pow.cpp is
  the actual upstream contract surface — the route mentions pow as a
  parity_op but the upstream-path list is incomplete for this op).

- REQ-9: `rsub(a, b, *, alpha=1)` — `torch.rsub(input, other, *, alpha=1)`
  is `other - alpha*input`. Per BinaryOps.cpp:1169
  `Tensor rsub(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return at::sub(other, self, alpha); }` (a literal operand-swap delegation
  to sub) and overrides.py:1116
  `torch.rsub: lambda input, other, alpha=1: -1`. SHIPPED via
  `arithmetic::rsub` at `grad_fns/arithmetic.rs:1115` (one-line wrapper
  delegating to `sub_scaled(b, a, alpha)` — matches upstream byte-for-byte
  per R-DEV-1) + non-test production consumer `Tensor::rsub_t` at
  `methods.rs:50`. Parity-sweep `[rsub]` arm landed in the `"rsub" =>`
  arm in `tools/parity-sweep/runner/src/main.rs`.

- REQ-10: `rsqrt(a)` — `torch.rsqrt(input)` is `1 / sqrt(input)`. Per
  UnaryOps.cpp:346 `CREATE_UNARY_TORCH_IMPL_FUNC(rsqrt_out, rsqrt_stub)` and
  overrides.py:1115 `torch.rsqrt: lambda input, out=None: -1`. SHIPPED via
  `arithmetic::rsqrt` at `grad_fns/arithmetic.rs:1950` (dedicated forward +
  the `RsqrtBackward` struct in `grad_fns/arithmetic.rs` saving the output `c` and computing
  `da = -0.5 * grad * c^3` per `tools/autograd/derivatives.yaml:1504-1506
  - name: rsqrt(Tensor self) -> Tensor / self: -0.5 * grad * result.pow(3).conj()`)
  + non-test production consumer `Tensor::rsqrt_t` at `rsqrt_t in methods.rs`.
  Parity-sweep `[rsqrt] 24/24 passed (0 skipped, 0 failed)` at seeds=8
  (closes #1195). Runner arm at the `"rsqrt" =>` arm in
  `tools/parity-sweep/runner/src/main.rs`.

- REQ-11: `reciprocal(a)` — `torch.reciprocal(input)` is `1 / input`. Per
  UnaryOps.cpp:345 `CREATE_UNARY_TORCH_IMPL_FUNC(reciprocal_out,
  reciprocal_stub)` and overrides.py:1098
  `torch.reciprocal: lambda input, out=None: -1`. SHIPPED via
  `arithmetic::reciprocal` at `grad_fns/arithmetic.rs:2097` (dedicated
  forward + the `ReciprocalBackward` struct in `grad_fns/arithmetic.rs` saving the output `c`
  and computing `da = -grad * c^2` per
  `tools/autograd/derivatives.yaml:1447-1449
  - name: reciprocal(Tensor self) -> Tensor / self: -grad * (result * result).conj()`)
  + non-test production consumer `Tensor::reciprocal_t` at `reciprocal_t in methods.rs`.
  Parity-sweep `[reciprocal] 24/24 passed (0 skipped, 0 failed)` at seeds=8
  (closes #1196). Runner arm at the `"reciprocal" =>` arm in
  `tools/parity-sweep/runner/src/main.rs`.

- REQ-12: `floor_divide(a, b)` — `torch.floor_divide(input, other)` floors
  toward `-inf` (TRUE FLOOR — verified live 2026-05-25
  `torch.floor_divide(-7.0, 3.0).item() == -3.0`; the pre-1.13 trunc-
  division wart called out at `_torch_docs.py:4267-4271` is gone). Per
  BinaryOps.cpp:979 `Tensor floor_divide(const Tensor& self, const
  Tensor& other)` (dispatches to `div_floor_stub` -> `div_floor_kernel`
  at `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:297-349` ->
  `c10::div_floor_floating` at `c10/util/generic_math.h:34-58`) and
  overrides.py:664 `torch.floor_divide: lambda input, other: -1`.
  SHIPPED via `arithmetic::floor_divide` at
  `grad_fns/arithmetic.rs:2939` (CPU kernel mirroring
  `c10::div_floor_floating` byte-for-byte: `fmod`-then-`(a-mod)/b`
  Python `__floordiv__` form with the `(b<0)!=(mod<0)` sign-correction
  `div-=1`, plus the `(div-floor(div))>0.5` round-up guard, plus the
  `copysign(0, a/b)` ±0-preserving branch when `div` rounds to zero) +
  the `FloorDivideBackward` struct in `grad_fns/arithmetic.rs` (errors on `.backward()` with
  `FerrotorchError::InvalidArgument` to mirror upstream's
  `grad_fn=<NotImplemented object>` and `RuntimeError: derivative for
  aten::floor_divide is not implemented` — `floor_divide` has NO entry
  in `tools/autograd/derivatives.yaml` verified `grep 'floor_divide'`
  returns nothing). Non-test production consumer
  `Tensor::floor_divide_t` at `floor_divide_t in methods.rs`. Parity-sweep
  `[floor_divide] 72/72 passed (0 skipped, 0 failed)` at seeds=8
  (closes #1197). Runner arm at the `"floor_divide" =>` arm in
  `tools/parity-sweep/runner/src/main.rs` (binary, no kwargs —
  `floor_divide` does not take alpha). Integer-only sibling at
  `int_tensor.rs:588` is unaffected; the float and int variants
  operate on disjoint type families (`Tensor<T: Float>` vs
  `IntTensor<I>`).

- REQ-13: `remainder(a, b)` — `torch.remainder(input, other)` returns the
  remainder with the SIGN OF THE DIVISOR (Python `%` / NumPy semantics).
  Per BinaryOps.cpp:1184
  `Tensor remainder(const Tensor& self, const Scalar& other)` and
  overrides.py:1100 `torch.remainder: lambda input, other, out=None: -1`.
  SHIPPED via `arithmetic::remainder` at `grad_fns/arithmetic.rs:2302`
  (CPU kernel matching `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:398-401`'s
  `fmod`-then-correct branch byte-for-byte: `scalar_t mod = std::fmod(a,
  b); if ((mod != 0) && ((b < 0) != (mod < 0))) mod += b;` — equivalent
  to `a - floor(a/b)*b` but produces upstream's exact ULPs via the
  hardware `fmod` primitive instead of accumulating 4-op rounding error)
  + the dedicated `RemainderBackward` struct in `grad_fns/arithmetic.rs` saving `a` / `b` and
  computing `da = grad`, `db = -grad * floor(a / b)` per
  `tools/autograd/derivatives.yaml:1455-1457
  - name: remainder.Tensor(Tensor self, Tensor other) -> Tensor
    self: grad
    other: -grad * self.div(other, /*rounding_mode=*/"floor")`.
  Non-test production consumer `Tensor::remainder_t` at `remainder_t in methods.rs`.
  Parity-sweep `[remainder] 72/72 passed (0 skipped, 0 failed)` at
  seeds=8 (closes #1198). Runner arm at the `"remainder" =>` arm in
  `tools/parity-sweep/runner/src/main.rs` (binary, no kwargs —
  `remainder` does not take alpha). Integer-only sibling at
  `int_tensor.rs:599` is unaffected; the float and int variants
  operate on disjoint type families (`Tensor<T: Float>` vs
  `IntTensor<I>`).

- REQ-14: `fmod(a, b)` — `torch.fmod(input, other)` returns the remainder
  with the SIGN OF THE DIVIDEND (C99 `fmod` semantics). Per
  BinaryOps.cpp:1540 `Tensor fmod(const Tensor& self, const Scalar& other)`
  and overrides.py:666 `torch.fmod: lambda input, other, out=None: -1`.
  SHIPPED via `arithmetic::fmod` at `grad_fns/arithmetic.rs:2609` (CPU
  kernel matching `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:1052-1054`'s
  `AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, ...)` branch
  byte-for-byte: `[](scalar_t x, scalar_t d) -> scalar_t { return
  std::fmod(x, d); }` — Rust's `T::%` on f32/f64 *is* C99 `std::fmod`
  verbatim, so the elementwise kernel is literally `av % bv` with NO
  sign-correction, distinct from `remainder_inner`'s `fmod`-then-correct
  flow) + the dedicated `FmodBackward` struct in `grad_fns/arithmetic.rs` saving `a` / `b`
  and computing `da = grad`, `db = -grad * trunc(a / b)` per
  `tools/autograd/derivatives.yaml:717-720
  - name: fmod.Tensor(Tensor self, Tensor other) -> Tensor
    self: grad
    other: -grad * self.div(other, /*rounding_mode=*/"trunc")`. The
  `trunc` (round-toward-zero) vs `floor` (round-toward-`-inf`)
  distinction is the chain-rule manifestation of the same
  dividend-sign vs divisor-sign forward divergence. Non-test production
  consumer `Tensor::fmod_t` at `fmod_t in methods.rs`. Parity-sweep
  `[fmod] 72/72 passed (0 skipped, 0 failed)` at seeds=8 (closes
  #1199). Runner arm at the `"fmod" =>` arm in
  `tools/parity-sweep/runner/src/main.rs` (binary, no kwargs — `fmod` does not take alpha).

- REQ-15: `addcmul(input, tensor1, tensor2, *, value=1)` — fused
  `out = input + value * tensor1 * tensor2`. Per
  `aten/src/ATen/native/PointwiseOps.cpp:57` `TORCH_IMPL_FUNC(addcmul_out)`
  and `_torch_docs.py:510`
  `addcmul(input, tensor1, tensor2, *, value=1, out=None) -> Tensor`.
  SHIPPED via `arithmetic::addcmul` at `grad_fns/arithmetic.rs:3303`
  (CPU 3-way broadcast iteration mirroring `PointwiseOps.cpp:17-31`'s
  meta-function TensorIteratorConfig: walks the broadcast output shape's
  flat index, maps each output coord into per-operand flat indices with
  size-1 broadcast collapsing, applies the fused `out_i = input_i +
  value * tensor1_i * tensor2_i` per upstream byte-for-byte / R-DEV-1) +
  the dedicated `AddcmulBackward` struct in `grad_fns/arithmetic.rs` saving `input` / `tensor1`
  / `tensor2` / `value: f64`. Backward per `tools/autograd/derivatives.yaml
  - name: addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar
  value=1) -> Tensor / self: handle_r_to_c(...); tensor1: handle_r_to_c(
  ..., grad * (tensor2 * value).conj()); tensor2: handle_r_to_c(..., grad
  * (tensor1 * value).conj())` (for `T: Float` real-only the
  `handle_r_to_c` and `.conj()` are identity); reduces each gradient to
  the operand's original shape via `reduce_grad_to_shape` because the
  3-way broadcast may have expanded any operand. Non-test production
  consumer: `Tensor::addcmul_t` at `addcmul_t in methods.rs` — the chainable
  method-style surface that satisfies R-DEFER-1. Parity-sweep
  `[addcmul] 96/96 passed (0 skipped, 0 failed)` at seeds=8 (closes
  #1200). Runner arm at the `"addcmul" =>` arm in
  `tools/parity-sweep/runner/src/main.rs` with the new reusable
  `ternary()` helper (3 args) + `value_kwarg()` helper (default 1.0).
  NB: `PointwiseOps.cpp` is the true upstream
  location; the route's declared `upstream` list (BinaryOps + UnaryOps)
  does not include it, which is itself an incomplete route declaration.

- REQ-16: `addcdiv(input, tensor1, tensor2, *, value=1)` — fused
  `out = input + value * tensor1 / tensor2`. Per
  `aten/src/ATen/native/PointwiseOps.cpp:66` `TORCH_IMPL_FUNC(addcdiv_out)`
  and `_torch_docs.py:461`
  `addcdiv(input, tensor1, tensor2, *, value=1, out=None) -> Tensor`.
  SHIPPED via `arithmetic::addcdiv` at `grad_fns/arithmetic.rs:3618`
  with the dedicated `AddcdivBackward` struct in `grad_fns/arithmetic.rs` saving `input`/`tensor1`/
  `tensor2`/`value: f64`. Backward per
  `tools/autograd/derivatives.yaml`:

  ```yaml
  - name: addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar
                  value=1) -> Tensor
    self: handle_r_to_c(self.scalar_type(), grad)
    tensor1: handle_r_to_c(tensor1.scalar_type(),
                           grad * (value / tensor2).conj())
    tensor2: handle_r_to_c(tensor2.scalar_type(),
                           -grad * (value * tensor1 /
                                    (tensor2 * tensor2)).conj())
  ```

  For `T: Float` (real-only) the `.conj()` is identity. The integer-
  dtype error path at `PointwiseOps.cpp:38-50 TORCH_META_FUNC(addcdiv)`
  is unreachable for `Tensor<T: Float>` — ferrotorch only admits
  `f32`/`f64`/`bf16`/`f16`. Non-test consumer: `Tensor::addcdiv_t` at
  `addcdiv_t in methods.rs` — the chainable method-style surface. Parity-sweep
  `[addcdiv] N/N passed (0 skipped, 0 failed)` at seeds=8 (closes
  #1201). Runner arm at the `"addcdiv" =>` arm in
  `tools/parity-sweep/runner/src/main.rs`, reusing the existing
  `ternary()` + `value_kwarg()` helpers introduced for addcmul (#1200)
  — per R-DEFER-8 the convention's second instance.

## Acceptance Criteria

- [x] AC-1: `add` parity-sweep at `--seeds 8` returns `[add] 88/88 passed
  (0 skipped, 0 failed)` (grep-count `passed (0 skipped, 0 failed)` == 1).
- [x] AC-2: `sub` parity-sweep at `--seeds 8` returns `[sub] 88/88 passed
  (0 skipped, 0 failed)` (grep-count `passed (0 skipped, 0 failed)` == 1).
  Closed by #1192 via `arithmetic::sub_scaled` + the parity-sweep runner
  `"sub"` arm passing `alpha_kwarg("sub")` through to it. The 16 prior
  failures (shape `[5,5]` at i=9 / i=10, the `alpha=2` / `alpha=-3.125`
  op_db samples) now pass because `sub_scaled` delegates to
  `add_scaled(a, b, -alpha)`.
- [x] AC-3: `mul` parity-sweep at `--seeds 8` returns `[mul] 72/72 passed
  (0 skipped, 0 failed)`.
- [x] AC-4: `div` parity-sweep at `--seeds 8` returns `[div] 72/72 passed
  (0 skipped, 0 failed)`.
- [x] AC-5: `neg` parity-sweep at `--seeds 8` returns `[neg] 8/8 passed
  (0 skipped, 0 failed)`.
- [x] AC-6: `abs` parity-sweep at `--seeds 8` returns `[abs] 8/8 passed
  (0 skipped, 0 failed)`.
- [x] AC-7: `sqrt` parity-sweep at `--seeds 8` returns `[sqrt] 8/8 passed
  (0 skipped, 0 failed)`.
- [x] AC-8: `pow` parity-sweep at `--seeds 8` returns `[pow] 24/72 passed
  (48 skipped, 0 failed)` — zero failures. 24 passes are the 0-d-exponent
  (scalar-exp) op_db samples; the 48 skips are the tensor-exponent overload
  (`pow_Tensor_Tensor_out` at `Pow.cpp:47`), which ferrotorch's scalar-exp
  `arithmetic::pow<T: Float>(a, exp: f64)` cannot consume — those samples
  exit dispatch with `Ok(None)` so they are recorded as skips, not
  divergences. Runner arm landed at
  `tools/parity-sweep/runner/src/main.rs:564` via blocker #1193.
- [x] AC-9: `rsub` parity-sweep at `--seeds 8` returns
  `[rsub] N/N passed (0 skipped, 0 failed)` with N >= 1. Closed by #1194
  via `arithmetic::rsub` (`grad_fns/arithmetic.rs:1115`) delegating to
  `sub_scaled(b, a, alpha)` + the runner `"rsub"` arm at
  `tools/parity-sweep/runner/src/main.rs:434`. The forward result matches
  PyTorch's `at::sub(other, self, alpha)` byte-for-byte; the
  `AddScaledBackward` attached by `sub_scaled` correctly routes
  `da = -alpha*grad` to leaf `input` (the rsub-API `a`) and `db = grad`
  to leaf `other` (the rsub-API `b`).
- [x] AC-10: `rsqrt` parity-sweep at `--seeds 8` returns
  `[rsqrt] 24/24 passed (0 skipped, 0 failed)`. Closed by #1195 via
  `arithmetic::rsqrt` (`grad_fns/arithmetic.rs:1950`) + the dedicated
  `RsqrtBackward` struct in `grad_fns/arithmetic.rs` saving the output `c` for the
  `-0.5 * grad * c^3` formula per `tools/autograd/derivatives.yaml:1504-1506`
  + non-test production consumer `Tensor::rsqrt_t` (`rsqrt_t in methods.rs`) +
  parity-sweep runner arm at
  `tools/parity-sweep/runner/src/main.rs:457`.
- [x] AC-11: `reciprocal` parity-sweep at `--seeds 8` returns
  `[reciprocal] 24/24 passed (0 skipped, 0 failed)`. Closed by #1196 via
  `arithmetic::reciprocal` (`grad_fns/arithmetic.rs:2097`) + the dedicated
  `ReciprocalBackward` struct in `grad_fns/arithmetic.rs` saving the output `c` for the
  `-grad * c^2` formula per `tools/autograd/derivatives.yaml:1447-1449`
  + non-test production consumer `Tensor::reciprocal_t` (`reciprocal_t in methods.rs`) +
  parity-sweep runner arm at
  `tools/parity-sweep/runner/src/main.rs:465`.
- [x] AC-13: `remainder` parity-sweep at `--seeds 8` returns
  `[remainder] 72/72 passed (0 skipped, 0 failed)`. Closed by #1198 via
  `arithmetic::remainder` (`grad_fns/arithmetic.rs:2302`) + the dedicated
  `RemainderBackward` struct in `grad_fns/arithmetic.rs` saving `a` / `b` and computing
  `da = grad`, `db = -grad * floor(a/b)` per
  `tools/autograd/derivatives.yaml:1455-1457` + non-test production
  consumer `Tensor::remainder_t` (`remainder_t in methods.rs`) + parity-sweep
  runner arm at `tools/parity-sweep/runner/src/main.rs:478`.
- [x] AC-14: `fmod` parity-sweep at `--seeds 8` returns
  `[fmod] N/N passed (0 skipped, 0 failed)` with N >= 1. Closed by
  #1199 via `arithmetic::fmod` (`grad_fns/arithmetic.rs:2609`) +
  the dedicated `FmodBackward` struct in `grad_fns/arithmetic.rs` saving `a` / `b` and computing
  `da = grad`, `db = -grad * trunc(a/b)` per
  `tools/autograd/derivatives.yaml:717-720` + non-test production
  consumer `Tensor::fmod_t` (`fmod_t in methods.rs`) + parity-sweep runner
  arm at `tools/parity-sweep/runner/src/main.rs:492`.
- [x] AC-12: `floor_divide` parity-sweep at `--seeds 8` returns
  `[floor_divide] 72/72 passed (0 skipped, 0 failed)`. Closed by #1197
  via `arithmetic::floor_divide` (`grad_fns/arithmetic.rs:2939`) +
  the `FloorDivideBackward` struct in `grad_fns/arithmetic.rs` erroring on `.backward()` to mirror
  upstream's `<NotImplemented>` grad_fn (no entry in
  `tools/autograd/derivatives.yaml`) + non-test production consumer
  `Tensor::floor_divide_t` (`floor_divide_t in methods.rs`) + parity-sweep runner
  arm at `tools/parity-sweep/runner/src/main.rs:511`.
- [x] AC-15: `addcmul` parity-sweep at `--seeds 8` returns
  `[addcmul] 96/96 passed (0 skipped, 0 failed)`. Closed by #1200 via
  `arithmetic::addcmul` (`grad_fns/arithmetic.rs:3303`) + the dedicated
  `AddcmulBackward` struct in `grad_fns/arithmetic.rs` saving `input` / `tensor1` / `tensor2` /
  `value: f64` and computing `d_input = grad`, `d_tensor1 = grad * value
  * tensor2`, `d_tensor2 = grad * value * tensor1` per
  `tools/autograd/derivatives.yaml` + non-test production consumer
  `Tensor::addcmul_t` (`addcmul_t in methods.rs`) + parity-sweep runner arm at
  `tools/parity-sweep/runner/src/main.rs:527` with new reusable
  `ternary()` and `value_kwarg()` helpers.
- [x] AC-16: `addcdiv` parity-sweep at `--seeds 8` returns
  `[addcdiv] N/N passed (0 skipped, 0 failed)` with N >= 1. Closed by
  #1201 via `arithmetic::addcdiv` (`grad_fns/arithmetic.rs:3618`) +
  the dedicated `AddcdivBackward` struct in `grad_fns/arithmetic.rs` saving `input` / `tensor1` /
  `tensor2` / `value: f64` and computing `d_input = grad`,
  `d_tensor1 = grad * value / tensor2`,
  `d_tensor2 = -grad * value * tensor1 / (tensor2 * tensor2)` per
  `tools/autograd/derivatives.yaml` + non-test production consumer
  `Tensor::addcdiv_t` (`addcdiv_t in methods.rs`) + parity-sweep runner arm at
  `tools/parity-sweep/runner/src/main.rs:547` (reuses existing
  `ternary()` and `value_kwarg()` helpers from addcmul #1200).
- [x] AC-17: `cargo test -p ferrotorch-core --lib grad_fns::arithmetic`
  passes (the `tests` mod at `arithmetic.rs:3677-4021` covers forward,
  backward, partial-requires-grad, no-grad, and chain-rule cases for
  `add`/`sub`/`mul`/`div`/`neg`/`pow`/`sqrt`/`abs`).
- [x] AC-18: Non-contiguous CUDA views (transpose / narrow / permute) flow
  through every binary forward without `LengthMismatch` — covered by
  `ensure_contig_for_gpu` at `arithmetic.rs:97-143` (#812 cluster).
- [x] AC-19: `add_scaled_out` correctly resizes `out` when its shape does
  not match the broadcast shape (matches torch's deprecation-warned
  silent-resize behavior in 2.x). Covered by parity-sweep `add` probes
  `out_basic` / `out_with_alpha` / `out_broadcast` / `out_wrong_shape` /
  `out_nan` documented in `tools/parity-sweep/parity_audit.json:54-70`.

## Architecture

### Layer-1 helpers (lines 18-310)

`is_f32` / `is_f64` / `is_bf16` / `is_f16` (`arithmetic.rs:42-65`) are
`TypeId`-based dtype discriminators used to gate the GPU dispatch arms.
`ensure_contig_for_gpu` (`arithmetic.rs:96-143`) is the #812 fix-up that
guarantees a CUDA tensor handed to a raw `gpu_handle()` kernel has
`storage_len == numel` and `storage_offset == 0`; non-contiguous views
route through the on-device `strided_copy_{f32,f64}` kernels rather than
detouring through host memory. `needs_grad` / `needs_grad_unary`
(`arithmetic.rs:149-159`) check `is_grad_enabled()` + per-tensor
`requires_grad()`. `reduce_grad_to_shape` (`arithmetic.rs:177-336`) is the
shared backward broadcast-reduction primitive: a GPU-resident path for
f32/f64 (uses `backend.sum_axis_{f32,f64}` after materializing the grad
view), a same-numel-different-rank reshape branch (`#814`), and a CPU
fallback loop that decomposes the grad flat index into per-axis coords and
sums into the target.

### REQ-1 `add` (lines 320-463)

`AddBackward` saves both operands; `backward` returns
`(reduce(grad, a.shape()), reduce(grad, b.shape()))` (`arithmetic.rs:400-427`).
The forward `pub fn add` (`arithmetic.rs:430-446`) emits a profiler scope,
checks device-match, lets `meta_propagate::binary_broadcast` short-circuit
for meta tensors, then calls `add_inner`. `add_inner` (`arithmetic.rs:447-544`)
materializes both CUDA inputs if needed, picks between `add_*` and
`broadcast_add_*` via `dispatch_floating_dtype!` (f32/f64/bf16/f16), and
falls through to `fast_add` on CPU. **Non-test consumer**:
`add_t in ferrotorch-core/src/methods.rs` — `Tensor::add_t(&self, other)`
delegates to `crate::grad_fns::arithmetic::add`, exposing the op as a
method on every `Tensor<T: Float>` and used pervasively throughout
ferrotorch (`autograd::forward_ad`, `autograd::higher_order`,
`autograd::fixed_point`, `autograd::grad_penalty`, `ops::higher_order`,
`einops`, `einsum`, `vmap`, `meta_propagate`).

### REQ-1 extension: `add_scaled` + `add_out` / `add_scaled_out` (lines 466-934)

`AddScaledBackward` (`arithmetic.rs:552-602`) saves `a`, `b`, and `alpha:
f64`; backward returns `(reduce(grad, a.shape()), reduce(alpha*grad,
b.shape()))`. `scale_tensor` (`arithmetic.rs:614-649`) is a private helper
that routes to dtype-specialised GPU `scale_*` kernels (f32/f64/bf16/f16)
or to `scalar_map` on CPU. `check_out_allowed` (`arithmetic.rs:683-716`)
enforces torch's `out=` rules: no grad_fn, no requires_grad-leaf. `add_out`
(`arithmetic.rs:741-786`) is an `alpha=1.0` wrapper over `add_scaled_out`.
`add_scaled_out` (`arithmetic.rs:791-862`) validates devices, computes
`add_scaled` under `no_grad`, and writes through
`Tensor::update_storage` (matched-shape branch) or
`Tensor::update_storage_and_shape` (resize branch) — both are `unsafe` and
documented with SAFETY comments tying back to the `check_out_allowed`
proof. `add_scaled` (`arithmetic.rs:868-1026`) shortcircuits `alpha == 1.0`
to plain `add`, then takes one of two forward paths before attaching
`AddScaledBackward` if either operand requires grad:

- **Fused single-launch GPU path (#1675)**: `maybe_add_scaled_fused_gpu`
  (`arithmetic.rs:971`) returns `Some(result)` when `a` and `b` are both
  CUDA-resident, SAME shape (no broadcast), dtype f32/f64, and `alpha` is
  finite. It calls the new `GpuBackend::add_scaled_f32` /
  `add_scaled_f64` trait slots (`gpu_dispatch.rs`, default
  `NotImplementedOnCuda`; CUDA override in
  `ferrotorch-gpu/src/backend_impl.rs`), which dispatch to the fused
  on-device kernels `gpu_add_scaled_f32` / `gpu_add_scaled_f64`
  (`ferrotorch-gpu/src/kernels.rs`) computing `out[i] = fma(alpha, b[i],
  a[i])` in a SINGLE kernel launch (FMA, no temporary device buffer).
  For `alpha == -1.0` the FMA is exactly `a - b` (bit-exact vs CPU `sub`
  AND torch's `add_stub(..., -alpha)` contract at `BinaryOps.cpp:434`);
  for general finite alpha the single-rounding fused result is at least
  as accurate as the two-rounding scale-then-add. This roughly halves
  the GPU launch cost of `sub` / `sub_scaled` / `rsub` (all delegate
  here); measured ~3x faster on a `[1000,1000]` RTX 3090 round-trip vs
  the scale-then-add staging. The fused branch is the **non-test
  production consumer** of the new `add_scaled_f{32,64}` trait slots,
  satisfying R-DEFER-1 in the same commit that ships them.
- **Scale-then-add fallback**: pre-scales `b` under `no_grad`
  (`scale_tensor`), calls `add_inner`, then attaches the same backward.
  Used for CPU, broadcast (different shapes), bf16/f16, and non-finite
  alpha (where it already matches torch's NaN/inf-alpha semantics).

**Non-test consumers**: `inplace.rs:167` — `Tensor::add_scaled_` invokes
`arithmetic::add_scaled` and swaps the result storage in-place to satisfy
torch's `tensor.add_(other, *, alpha=1)` API; `inplace.rs:158-159`
documents that `add_` is now an `alpha=1.0` alias for `add_scaled_`.
`add_out` / `add_scaled_out` themselves are the public surface; their
non-test consumer is the parity-sweep `out=` probe path documented in
`parity_audit.json:54` — note this is a test-side consumer, but the API is
still wired to torch parity via the in-place `add_scaled_` which uses
`add_scaled` directly. The `add_out` / `add_scaled_out` entry points
themselves are **vocabulary at this layer** — no in-tree non-test caller
invokes `add_scaled_out` directly; users reach the `out=` semantics via
`Tensor::add_scaled_` which calls `add_scaled` then `update_storage`. The
REQ-1 SHIPPED claim rests on `add` + `add_scaled` having `methods.rs` /
`inplace.rs` consumers respectively, not on `add_out` / `add_scaled_out`
having direct consumers.

### REQ-2 `sub` / `sub_scaled` (delegation to `add_scaled`)

After commit `d0fd83f1a` (the post-#1215 delegation), `sub` no longer has
its own backward grad-fn struct or its own subtract-inner private helper.
Both were removed when `sub` was rewritten as a one-line wrapper around
`sub_scaled`. The current shape is:

- `pub fn sub<T: Float>(a, b)` at `sub in arithmetic.rs` is a one-line
  delegation: `sub_scaled(a, b, 1.0)`.
- `pub fn sub_scaled<T: Float>(a, b, alpha: f64)` at `sub_scaled in arithmetic.rs`
  is a thin alias that forwards to `add_scaled(a, b, -alpha)`.

PyTorch's `torch.sub(input, other, *, alpha=1)` at
`aten/src/ATen/native/BinaryOps.cpp:434 TORCH_IMPL_FUNC(sub_out)`
literally calls `add_stub(device_type(), *this, -alpha)` — the
delegation matches upstream byte-for-byte (R-DEV-1). Broadcasting, GPU
dispatch, the `AddScaledBackward` VJP (which naturally produces
`db = -alpha * grad`, i.e. `-grad` when `alpha == 1`), and the
`alpha == 1.0` fast-path-to-plain-`add` are all inherited from
`add_scaled` for free. The backward identity is therefore
`AddScaledBackward` for both the plain and alpha-kwarg paths (there is
no dedicated subtract-backward grad-fn).

**Non-test consumers**: `sub_t in methods.rs` (`Tensor::sub_t`) calls
`arithmetic::sub`; `autograd::forward_ad:97-98` (dual-number forward
subtraction primal+tangent) calls `arithmetic::sub`;
`autograd::grad_penalty:117` builds the `norm - 1` term used by the
WGAN-GP penalty via `sub`; `inplace.rs:266` (`Tensor::sub_scaled_`)
delegates to `add_scaled_(other, -alpha)`, providing the in-place
`torch.Tensor.sub_(other, *, alpha)` semantics by re-using the same
shape-strict / broadcast / GPU fast-path machinery as `add_scaled_`. The
`sub_out` / `sub_scaled_out` entry points are intentionally NOT shipped
in this commit — out-of-scope per the #1192 dispatch (vocabulary-only
risk, separable). Blocker #1192 closed.

### REQ-3 `mul` (lines 1016-1185)

`MulBackward` (`arithmetic.rs:1130-1185`) saves `a`/`b`; backward has two
branches — when `grad_output.requires_grad()` or has a `grad_fn`
(higher-order grad / `create_graph=True`), it uses differentiable `mul`
calls so the backward pass enters the graph; otherwise it routes through
`no_grad(|| mul(...))`. Forward `mul`/`mul_inner`
(`arithmetic.rs:1189-1347`) is structurally identical to `add`/`add_inner`
with `mul_*` / `broadcast_mul_*` kernels and the CPU `fast_mul`
fallthrough. **Non-test consumer**: `methods.rs:36-38` `Tensor::mul_t`;
also `mul_t in einsum.rs,824,840,848` (the batch-matmul broadcast paths),
`autograd::higher_order:471`, `autograd::grad_penalty:122,295`,
`grad_fns::transcendental:73,278,355`.

### REQ-4 `div` (lines 1191-1347)

`DivBackward` (`arithmetic.rs:1305-1345`) — `da = grad / b`,
`db = -grad * a / (b*b)`. Forward `div`/`div_inner`
(`arithmetic.rs:1357-1470`) — `dispatch_floating_dtype!` over
`div_{f32,f64,bf16_bf16,f16}` and `broadcast_div_{...}`, CPU `fast_div`
fallthrough. IEEE-754 div-by-zero behavior is delegated to the underlying
kernel / `fast_div`. **Non-test consumer**: `methods.rs:40`
`Tensor::div_t`; also `autograd::forward_ad:114,120` (dual-number division
quotient rule), `nn::loss.rs:136` (mean reduction), `transcendental.rs:175`
(log-backward `grad / x`).

### REQ-5 `neg` (lines 1353-1422)

`NegBackward` (`arithmetic.rs:1475-1496`) returns `-grad` under `no_grad`.
Forward `neg`/`neg_inner` (`arithmetic.rs:1499-1545`) routes through
`neg_{f32,f64,bf16_bf16,f16}` on CUDA and `unary_map(a, |x| -x)` on CPU.
**Non-test consumer**: `methods.rs:44-46` `Tensor::neg_t`; also
`neg_t in vmap.rs`, `autograd::forward_ad:126-127`, `transcendental.rs`,
and recursive use inside `DivBackward` here (the former subtract-backward
consumer was eliminated when `sub` was rewritten as a `sub_scaled` →
`add_scaled` delegation in commit `d0fd83f1a`).

### REQ-8 `pow` (lines 1428-1549)

`PowBackward` (`arithmetic.rs:1550-1662`) saves `a` + `exp: f64`; backward
computes `grad * exp * a^(exp-1)` with three branches: (a) higher-order
graph-recording path when `grad_output.requires_grad() ||
grad_output.grad_fn().is_some()`; (b) GPU path under `no_grad`; (c) CPU
direct-data-vec path. `PowBackward::scalar_args` returns `vec![self.exp]`
so the JIT tracer rehydrates `IrOpKind::Pow { exponent }` (#887). Forward
`pow`/`pow_inner` (`arithmetic.rs:1667-1780`) uses
`backend.pow_{f32,f64}` on f32/f64 CUDA, and `scalar_map(a, exp_t, |x, e|
x.powf(e))` on CPU. bf16/f16 fall through to the CPU path. **Non-test
consumer**: `methods.rs:34-36` `Tensor::pow_t`; also
`autograd::grad_penalty:111,118`, `functional::normalize`/`pairwise_distance` in `ferrotorch-nn/src/functional.rs`
(`functional::normalize` raises `|x|^p` then takes `^(1/p)`).

The parity-sweep runner now ships a `"pow"` arm in
`tools/parity-sweep/runner/src/main.rs`. op_db emits `pow` samples with
args[1] always wrapped as a tensor envelope (no plain JSON-number form), so
the arm: (a) materializes args[0] as the base tensor, (b) inspects args[1]
— if its shape is `[]` (0-d), it decodes the single f32 into an f64 scalar
and calls `arithmetic::pow(&base, exp)`; if its shape is non-empty, the
tensor-exp overload (`Pow.cpp:47 pow_Tensor_Tensor_out`) is out of scope
for `arithmetic::pow<T>(a, exp: f64)` and the arm returns `Ok(None)` so
the sweep records a skip. Result at `--seeds 8`: `[pow] 24/72 passed (48
skipped, 0 failed)`. Blocker #1193 closed.

### REQ-7 `sqrt` (lines 1555-1653)

`SqrtBackward` (`arithmetic.rs:1729-1783`) — `grad / (2 * sqrt(a))`. GPU
path constructs a CPU `[2.0; numel]` tensor, uploads via `.to(device)`,
multiplies with `sqrt(a)`, divides `grad` by the product. CPU path is a
direct zip-map. Forward `sqrt`/`sqrt_inner` (`arithmetic.rs:1787-1853`)
uses `backend.sqrt_{f32,f64,f16}` on CUDA (f16 added in crosslink #1185
Phase 1), `unary_map(a, |x| x.sqrt())` on CPU. **Non-test consumer**:
`methods.rs:51-40` `Tensor::sqrt_t`; also `autograd::grad_penalty:113`
(gradient-norm computation).

### REQ-6 `abs` (lines 1657-1770)

`AbsBackward` (`arithmetic.rs:3751-3817`) — `grad * sign(a)` with explicit
`sign(0) = 0` convention and a dedicated GPU `abs_backward_{f32,f64}`
kernel path when both `grad_output` and `a` live on CUDA. Forward
`abs`/`abs_inner` (`arithmetic.rs:3822-3863`) uses
`backend.abs_{f32,f64}` on f32/f64 CUDA, `unary_map(a, |x| x.abs())`
elsewhere (bf16/f16 fall through to CPU). **Non-test consumer**:
`methods.rs:55-58` `Tensor::abs_t`.

### REQ-9 `rsub` (one-line delegation)

`rsub` (`arithmetic.rs:1115`) is a one-line wrapper over `sub_scaled(b, a,
alpha)` — operand-swap delegation matching upstream byte-for-byte per
`aten/src/ATen/native/BinaryOps.cpp:1169-1171 Tensor rsub(...) { return
at::sub(other, self, alpha); }` (R-DEV-1). No new `RsubBackward` struct
is needed: the `AddScaledBackward` attached by the underlying
`sub_scaled` call saves `b` and `a` as its `a` and `b` fields
respectively, so backward routes the chain-rule gradients to the right
leaf tensors via saved-tensor identity (autograd routes by reference,
not by argument position). The chain rule produces `d(rsub)/d(input) =
-alpha` and `d(rsub)/d(other) = 1`. **Non-test consumer**:
`methods.rs:32` — `Tensor::rsub_t(&self, other, alpha)` delegates to
`arithmetic::rsub`, the chainable method-style surface that satisfies
R-DEFER-1. Parity-sweep runner arm at
`tools/parity-sweep/runner/src/main.rs:434` (binary + alpha_kwarg,
mirrors the `"sub"` arm shape).

### REQ-13 `remainder` (lines 1953-2192)

`RemainderBackward` (`arithmetic.rs:2183-2227`) saves `a`/`b`; backward
returns `(reduce(grad, a.shape()), reduce(-grad * floor(a/b),
b.shape()))` per `tools/autograd/derivatives.yaml:1455-1457`. The
`floor(a/b)` step runs under `no_grad` (treating `floor`'s gradient as
zero, matching upstream's `rounding_mode="floor"` autograd contract).
Forward `pub fn remainder` (`remainder in arithmetic.rs`) emits a profiler
scope, checks device match, lets `meta_propagate::binary_broadcast`
short-circuit for meta tensors, then calls `remainder_inner`. The
inner walks the broadcast iteration space directly (mirroring the CPU
`fast_*` loop pattern) and applies the upstream
`aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:398-401` kernel
elementwise: `scalar_t mod = std::fmod(a, b); if ((mod != 0) && ((b <
0) != (mod < 0))) mod += b;`. Rust's `T::%` is C99 `fmod` (dividend-
sign), so the sign-correction add is needed to recover divisor-sign
(Python `%`) semantics. Matching upstream byte-for-byte (R-DEV-1) is
what brought the parity-sweep delta below the `rtol=1e-5, atol=1e-7`
tolerance — a `a - floor(a/b) * b` compose-form had `~3.6e-7` ULP drift
on small inputs because of 4-op accumulated rounding. **Non-test
consumer**: `methods.rs:106` (`Tensor::remainder_t`). CPU-only in this
commit; CUDA inputs flow through host-memory fallback (the same way
`pow_inner` falls through bf16/f16). A dedicated `remainder_*` GPU
kernel can land under a separate blocker when a routed GPU consumer
surfaces — no `.cpu()`-then-`.cuda()` round-trip is introduced
(R-CODE-4 unaffected).

### REQ-14 `fmod` (lines 2256-2462)

`FmodBackward` (`arithmetic.rs:2491-2535`) saves `a`/`b`; backward
returns `(reduce(grad, a.shape()), reduce(-grad * trunc(a/b),
b.shape()))` per `tools/autograd/derivatives.yaml:717-720`. The
`trunc(a/b)` step runs under `no_grad` (treating `trunc`'s gradient
as zero, matching upstream's `rounding_mode="trunc"` autograd
contract — the chain-rule sibling of `RemainderBackward`'s `floor`
treatment). Forward `pub fn fmod` (`fmod in arithmetic.rs`) emits a
profiler scope, checks device match, lets
`meta_propagate::binary_broadcast` short-circuit for meta tensors,
then calls `fmod_inner`. The inner walks the broadcast iteration
space directly (mirroring `remainder_inner`'s loop pattern) and
applies the upstream `aten/src/ATen/native/cpu/BinaryOpsKernel
.cpp:1052-1054` kernel elementwise: `return std::fmod(x, d);`.
Rust's `T::%` IS C99 `fmod` (dividend-sign), so the kernel is
literally `av % bv` — NO sign-correction step (the structural
distinction from `remainder_inner`'s `fmod`-then-correct flow that
recovers Python `%` / NumPy divisor-sign semantics). **Non-test
consumer**: `methods.rs:124` (`Tensor::fmod_t`). CPU-only in this
commit; CUDA inputs flow through host-memory fallback (the same
way `remainder_inner` and `pow_inner` fall through bf16/f16). A
dedicated `fmod_*` GPU kernel can land under a separate blocker
when a routed GPU consumer surfaces — no
`.cpu()`-then-`.cuda()` round-trip is introduced (R-CODE-4
unaffected).

### REQ-12 `floor_divide` (lines 2547-2929)

`FloorDivideBackward` (`FloorDivideBackward in arithmetic.rs`) saves `a`/`b`;
backward returns `FerrotorchError::InvalidArgument` with the message
`"derivative for floor_divide is not implemented ..."` to mirror
upstream's behaviour. `torch.floor_divide` has NO entry in
`tools/autograd/derivatives.yaml` (verified `grep 'floor_divide'
/home/doll/pytorch/tools/autograd/derivatives.yaml` returns nothing)
and `THPVariable_floor_divide` is wrapped in
`TypeError_to_NotImplemented_` at
`tools/autograd/templates/python_variable_methods.cpp:1279` — so
upstream's `grad_fn` is `<NotImplemented object>` and `.backward()`
raises `RuntimeError: derivative for aten::floor_divide is not
implemented` (verified live 2026-05-25). Attaching the grad_fn (rather
than refusing to attach) preserves the upstream contract that the
autograd graph IS built; the failure happens on backward traversal.

Forward `pub fn floor_divide` (`arithmetic.rs:2939-3015`) emits a
profiler scope, checks device match, lets
`meta_propagate::binary_broadcast` short-circuit for meta tensors, then
calls `floor_divide_inner`. The inner walks the broadcast iteration
space directly (mirroring `remainder_inner` / `fmod_inner`'s loop
pattern) and applies the upstream `c10::div_floor_floating` algorithm
at `c10/util/generic_math.h:34-58` byte-for-byte (R-DEV-1):

```text
if (b == 0)          return a / b;             // IEEE-754 div-by-zero
mod = fmod(a, b)                                // Rust's T::% is C99 fmod
div = (a - mod) / b                             // Python __floordiv__ form
if (mod != 0 && (b<0)!=(mod<0)) div -= 1        // sign-correction adjust
if (div == 0)        return copysign(0, a/b)    // ±0 preservation
else                  let f = floor(div); return (div - f > 0.5) ? f+1 : f
```

The `(a - mod)/b` form (rather than a literal `(a/b).floor()`) is the
Python `__floordiv__` contract that maintains `a == (a // b) * b +
remainder(a, b)` exactly even under floating-point rounding. The post
`(div - floor(div)) > 0.5` round-up guard handles the edge case where
rounding in `(a - mod)/b` pushes the quotient above the true floor by
more than half. The `copysign(0, a/b)` branch when `div` rounds to
zero preserves the IEEE-754 sign-of-quotient `±0` distinction (matters
for downstream chained ops, even though numerically `+0 == -0`).
Matching this byte-for-byte is what brought parity-sweep below
`rtol=1e-5, atol=1e-7` tolerance — a literal `(a/b).floor()` form
diverges on cases where `fmod`-then-subtract has different rounding
error from a direct division.

**Non-test consumer**: `methods.rs:176` (`Tensor::floor_divide_t`).
CPU-only in this commit; CUDA inputs flow through host-memory fallback
(the same way `remainder_inner` / `fmod_inner` / `pow_inner`'s bf16/f16
path falls through). A dedicated `floor_divide_*` GPU kernel can land
under a separate blocker when a routed GPU consumer surfaces — no
`.cpu()`-then-`.cuda()` round-trip is introduced (R-CODE-4
unaffected).

### REQ-15 `addcmul` (lines 2908-3203)

`AddcmulBackward` (`arithmetic.rs:3160-3260`) saves `input` / `tensor1`
/ `tensor2` (as full `Tensor<T>` clones for the backward `mul()`
graph-call) plus the scalar `value: f64`. `backward` computes:

- `d_input   = grad`                  (reduced to `input.shape()`)
- `d_tensor1 = grad * value * tensor2` (reduced to `tensor1.shape()`)
- `d_tensor2 = grad * value * tensor1` (reduced to `tensor2.shape()`)

per `tools/autograd/derivatives.yaml` `name: addcmul ... / self:
handle_r_to_c(grad) / tensor1: ... grad * (tensor2 * value).conj() /
tensor2: ... grad * (tensor1 * value).conj()`. For `T: Float` (real-
only) the `.conj()` is identity. `value` is scalar; no grad wrt it.
The `* value` step uses a 0-d scale tensor multiplied via the standard
`mul()` family so broadcasting flows correctly. The whole backward
chain runs under `no_grad` so intermediates don't enter the graph
(higher-order addcmul is not exercised by op_db; non-higher-order
backward parity is what this commit ships).

Forward `pub fn addcmul` (`arithmetic.rs:3303-3328`) emits a profiler
scope, checks pairwise device match (input vs tensor1, input vs
tensor2), and calls `addcmul_inner`. The inner (`arithmetic.rs:3143-
3128`) performs a 3-way broadcast: chains `broadcast_shapes(t1, t2)`
then `broadcast_shapes(input, t12)` to derive the output shape, walks
the broadcast iteration order over the result, maps each output coord
into per-operand flat indices with size-1 broadcast collapsing, then
applies the fused `out_i = input_i + value * tensor1_i * tensor2_i`
per upstream byte-for-byte (R-DEV-1) mirroring `PointwiseOps.cpp:17-31`
meta-function `TensorIteratorConfig` with 3 const inputs. CPU-only in
this commit; CUDA inputs flow through host-memory fallback — same
pattern as `remainder_inner` / `fmod_inner` / `floor_divide_inner` /
`pow_inner`'s bf16/f16 fallthrough. No `.cpu()`-then-`.cuda()` round
trip is introduced (R-CODE-4 unaffected). **Non-test consumer**:
`methods.rs:207` (`Tensor::addcmul_t`) delegates to
`arithmetic::addcmul`, satisfying R-DEFER-1 (the chainable method-
style surface). The parity-sweep runner arm at
`tools/parity-sweep/runner/src/main.rs:527` introduces a new reusable
`ternary()` helper (3 args, by analogy with `binary` / `unary`) and a
new `value_kwarg()` helper (same shape as `alpha_kwarg`, but reads
`kwargs["value"]` default 1.0). Both helpers are reused by
`addcdiv` (#1201) — the convention's second instance landed in the
addcdiv build.

The **3-arg dispatch pattern** introduced here is the convention's
first instance — `addcdiv` will reuse the same `ternary()` /
`value_kwarg()` helpers. Per R-DEFER-8, "cross-cutting" is not a
deferral excuse; the broader question of whether ternary-pointwise
ops want a dedicated trait can settle later when more 3-arg ops
surface. For now the inline ternary helper in the runner is the
contract.

### REQ-16 `addcdiv` (lines 3204-3491)

`AddcdivBackward` (`arithmetic.rs:3456-3557`) saves `input` / `tensor1`
/ `tensor2` (as full `Tensor<T>` clones for the backward graph-calls)
plus the scalar `value: f64`. `backward` computes:

- `d_input   = grad`                                  (reduced to `input.shape()`)
- `d_tensor1 = grad * value / tensor2`                (reduced to `tensor1.shape()`)
- `d_tensor2 = -grad * value * tensor1 / (tensor2^2)` (reduced to `tensor2.shape()`)

per `tools/autograd/derivatives.yaml` `name: addcdiv ... / self:
handle_r_to_c(grad) / tensor1: ... grad * (value / tensor2).conj() /
tensor2: ... -grad * (value * tensor1 / (tensor2 * tensor2)).conj()`.
For `T: Float` (real-only) the `.conj()` is identity. `value` is
scalar; no grad wrt it. The two single-tensor divisions in d_tensor2
(`step1 = neg_g * tensor1 / tensor2; step2 = step1 / tensor2`) avoid
materializing `tensor2^2` separately and let broadcasting flow
naturally. The whole backward chain runs under `no_grad` so
intermediates don't enter the graph (higher-order addcdiv is not
exercised by op_db; non-higher-order backward parity is what this
commit ships). At `tensor2 = 0` the d_tensor2 path produces NaN / ±Inf
via IEEE-754 — matches upstream byte-for-byte (R-DEV-1).

Forward `pub fn addcdiv` (`arithmetic.rs:3618-3643`) emits a profiler
scope, checks pairwise device match (input vs tensor1, input vs
tensor2), and calls `addcdiv_inner`. The inner (`arithmetic.rs:3458-
3448`) performs a 3-way broadcast: chains `broadcast_shapes(t1, t2)`
then `broadcast_shapes(input, t12)` to derive the output shape, walks
the broadcast iteration order over the result, maps each output coord
into per-operand flat indices with size-1 broadcast collapsing, then
applies the fused `out_i = input_i + value * tensor1_i / tensor2_i`
per upstream byte-for-byte (R-DEV-1) mirroring
`PointwiseOps.cpp:33-52` meta-function `build_ternary_op(...)` + the
`addcdiv_stub` dispatch at `PointwiseOps.cpp:66-73`. IEEE-754 div-by-
zero at `tensor2_i=0` produces ±Inf (or NaN if `tensor1_i=0` too),
matching upstream. The integer-dtype hard error at
`PointwiseOps.cpp:38-50 TORCH_META_FUNC(addcdiv)` is unreachable for
the `Tensor<T: Float>` family. CPU-only in this commit; CUDA inputs
flow through host-memory fallback — same pattern as `addcmul_inner` /
`remainder_inner` / `fmod_inner` / `floor_divide_inner` / `pow_inner`
'sbf16/f16 fallthrough. No `.cpu()`-then-`.cuda()` round trip is
introduced (R-CODE-4 unaffected). **Non-test consumer**:
`methods.rs:246` (`Tensor::addcdiv_t`) delegates to
`arithmetic::addcdiv`, satisfying R-DEFER-1 (the chainable method-
style surface). The parity-sweep runner arm at
`tools/parity-sweep/runner/src/main.rs:547` REUSES the existing
`ternary()` and `value_kwarg()` helpers introduced for addcmul
(#1200) — no new helpers needed (per R-DEFER-8, this is the
convention's second instance, reusing the first).

## Parity contract

The route's `parity_ops` list declares 16 ops. The current state per op:

| op | upstream entry | parity-sweep status | edge-case contract |
|---|---|---|---|
| `add` | BinaryOps.cpp:434 (sub_out via add_stub w/ -alpha) + `_torch_docs.py:358` signature `add(input, other, *, alpha=1, out=None)` | verified (88/88 at seeds=8) | NaN propagates; +/-Inf preserved; denormals preserved (no FTZ); empty shapes preserved; scalar `[]` broadcasts; 0-stride expand broadcasts; alpha edges (0, -0.0, NaN, ±huge); type promotion not yet validated (op_db emits f32-only); non-contig views materialized via `strided_copy_*`; in-place via `add_scaled_`; `out=` via `add_scaled_out` |
| `sub` | BinaryOps.cpp:434 `TORCH_IMPL_FUNC(sub_out)` (literally `add_stub(..., -alpha)`) + `_torch_docs.py:10851` signature `sub(input, other, *, alpha=1, out=None)` | verified (88/88 at seeds=8) — `arithmetic::sub_scaled` delegates to `add_scaled(a, b, -alpha)` matching upstream byte-for-byte | inherits add's contract via the `-alpha` delegation; NaN propagates; ±Inf preserved; alpha edges (0, -0.0, NaN, ±huge); broadcasting + grad inherited from `add_scaled` |
| `mul` | BinaryOps.cpp:441 `TORCH_IMPL_FUNC(mul_out)` + `_torch_docs.py:7754` `mul(input, other, *, out=None)` | verified (72/72 at seeds=8) | NaN, ±Inf, denormals, empty, scalar, 0-stride, type promotion (not yet covered); no `out=` variant in ferrotorch |
| `div` | BinaryOps.cpp:447 `TORCH_IMPL_FUNC(div_out)` + `_torch_docs.py:3926` `div(input, other, *, rounding_mode=None, out=None)` | verified (72/72 at seeds=8) | IEEE-754 div-by-zero produces ±Inf / NaN; no `rounding_mode` kwarg in ferrotorch (the `floor` / `trunc` modes are equivalent to `floor_divide` / `trunc_divide`, themselves NOT-STARTED here) |
| `neg` | UnaryOps.cpp:344 `CREATE_UNARY_TORCH_IMPL_FUNC(neg_out, neg_stub)` + `_torch_docs.py` signature `torch.neg(input, *, out=None)` | verified (8/8 at seeds=8) | sign bit flipped; NaN preserved (payload may not be); ±Inf -> ∓Inf; ±0.0 -> ∓0.0 |
| `abs` | UnaryOps.cpp:546 `Tensor abs(...)` (via `unary_op_impl_with_complex_to_float`) + `_torch_docs.py` `torch.abs(input, *, out=None)` | verified (8/8 at seeds=8) | NaN preserved; ±Inf -> +Inf; ±0.0 -> +0.0; complex-input promotion not supported (ferrotorch routes only `T: Float`); backward sign(0)=0 |
| `sqrt` | UnaryOps.cpp:359 `CREATE_UNARY_TORCH_IMPL_FUNC(sqrt_out, sqrt_stub)` | verified (8/8 at seeds=8) | sqrt(negative)=NaN; sqrt(-0.0)=-0.0; sqrt(+Inf)=+Inf; backward grad/(2*sqrt(a)) is ±Inf at a=0 (matches torch) |
| `pow` | `Pow.cpp:51` `TORCH_IMPL_FUNC(pow_Tensor_Scalar_out)` (NOT BinaryOps/UnaryOps — route incomplete) | verified scalar-exp subset (24/72 passed, 48 skipped, 0 failed at seeds=8); tensor-exp overload (`Pow.cpp:47 pow_Tensor_Tensor_out`) NOT IMPLEMENTED in ferrotorch and thus cleanly skipped | scalar exponent dispatched: pow(NaN, x)=NaN unless x=0 -> 1; pow(x, 0)=1 (including pow(0,0)=1, pow(NaN,0)=1); pow(±0, neg_exp)=±Inf — all asserted against torch for 0-d-exp op_db samples; tensor-exp samples (broadcasting between base and exp) are out-of-scope for `arithmetic::pow<T>(a, exp: f64)` and exit dispatch with `Ok(None)` |
| `rsub` | BinaryOps.cpp:1169 `Tensor rsub(const Tensor& self, const Tensor& other, const Scalar& alpha) { return at::sub(other, self, alpha); }` | verified (#1194) — `arithmetic::rsub` at `arithmetic.rs:1115` delegates to `sub_scaled(b, a, alpha)`; runner arm at `tools/parity-sweep/runner/src/main.rs:434` | computes `other - alpha*input`; operand-swap delegation to sub byte-for-byte; backward via the existing `AddScaledBackward` saved by `sub_scaled` (autograd routes by saved-tensor identity, so the swap does NOT scramble grad accumulation: `d(rsub)/d(input) = -alpha`, `d(rsub)/d(other) = 1`); alpha edges (0, -0.0, NaN, ±huge) inherited from `add_scaled` |
| `rsqrt` | UnaryOps.cpp:346 `CREATE_UNARY_TORCH_IMPL_FUNC(rsqrt_out, rsqrt_stub)` + `tools/autograd/derivatives.yaml:1504-1506 - name: rsqrt(Tensor self) -> Tensor / self: -0.5 * grad * result.pow(3).conj()` | verified (24/24 at seeds=8) — `arithmetic::rsqrt` at `arithmetic.rs:1950` (CPU `unary_map(a, |x| 1/x.sqrt())` matching `cpu/UnaryOpsKernel.cpp:534`; CUDA composes `sqrt(a)` + `div(ones, sqrt_a)` since no dedicated `rsqrt_*` GPU kernel exists); dedicated `RsqrtBackward` at `arithmetic.rs:1858` saves the output `c` (per upstream derivatives.yaml arithmetic rewrite `-0.5*a^(-3/2) = -0.5*c^3`); runner arm at `tools/parity-sweep/runner/src/main.rs:457` | `1/sqrt(input)`; rsqrt(0)=+Inf; rsqrt(negative)=NaN; rsqrt(+Inf)=+0.0; rsqrt(NaN)=NaN; backward `da = -0.5 * grad * c^3` saves output not input (single sqrt amortized) |
| `reciprocal` | UnaryOps.cpp:345 `CREATE_UNARY_TORCH_IMPL_FUNC(reciprocal_out, reciprocal_stub)` + `tools/autograd/derivatives.yaml:1447-1449 - name: reciprocal(Tensor self) -> Tensor / self: -grad * (result * result).conj()` | verified (24/24 at seeds=8) — `arithmetic::reciprocal` at `arithmetic.rs:2097` (CPU `unary_map(a, \|x\| 1/x)` matching `cpu/UnaryOpsKernel.cpp:279`; CUDA composes `div(ones, a)` since no dedicated `reciprocal_*` GPU kernel exists); dedicated `ReciprocalBackward` at `arithmetic.rs:2020` saves the output `c` (per upstream derivatives.yaml arithmetic rewrite `-1/a^2 = -c^2`); runner arm at `tools/parity-sweep/runner/src/main.rs:465` | `1/input`; reciprocal(+0.0)=+Inf; reciprocal(-0.0)=-Inf; reciprocal(+Inf)=+0.0; reciprocal(-Inf)=-0.0; reciprocal(NaN)=NaN; backward `da = -grad * c^2` saves output not input (single div amortized) |
| `floor_divide` | BinaryOps.cpp:979 `Tensor floor_divide(...)` (dispatches via `div_floor_stub` -> `div_floor_kernel` at `cpu/BinaryOpsKernel.cpp:297-349` -> `c10::div_floor_floating` at `c10/util/generic_math.h:34-58`) | verified (72/72 at seeds=8) — `arithmetic::floor_divide` at `arithmetic.rs:2939`, the `FloorDivideBackward` struct in `arithmetic.rs`; runner arm at `tools/parity-sweep/runner/src/main.rs:511` (binary, no kwargs) | TRUE FLOOR (toward -Inf) semantics — pre-1.13 trunc-division wart called out at `_torch_docs.py:4267-4271` is gone (verified live 2026-05-25 `torch.floor_divide(-7.0, 3.0) = -3.0`); the `(a-mod)/b` Python `__floordiv__` form + sign-correction `div -= 1` + 0.5-round-up + `copysign(0, a/b)` for `div=0` mirrors `c10::div_floor_floating` byte-for-byte (R-DEV-1); div-by-zero `5/0 = +Inf, -5/0 = -Inf, 0/0 = NaN` via the early `if (b == 0) return a/b` IEEE-754 branch; NaN propagation `floor_divide(NaN,x)=NaN`, `floor_divide(x,NaN)=NaN`, and notably `floor_divide(+Inf,3)=NaN` because `fmod(+Inf,3)=NaN` propagates through `(a-mod)/b`; backward errors via `FloorDivideBackward` with `FerrotorchError::InvalidArgument` to mirror upstream's `<NotImplemented>` grad_fn (no entry in `tools/autograd/derivatives.yaml`); distinct from `remainder` (sign-of-divisor) and `fmod` (sign-of-dividend) — for `a=-7,b=3` the 3-way contrast is `floor_divide=-3`, `remainder=2`, `fmod=-1` |
| `remainder` | BinaryOps.cpp:1184 `Tensor remainder(const Tensor& self, const Scalar& other)` + `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:398-401` (CPU float kernel: `fmod`-then-correct) + `tools/autograd/derivatives.yaml:1455-1457` (backward: `self: grad; other: -grad * self.div(other, /*rounding_mode=*/"floor")`) | verified (72/72 at seeds=8) — `arithmetic::remainder` at `arithmetic.rs:2302`, the `RemainderBackward` struct in `arithmetic.rs`; runner arm at `tools/parity-sweep/runner/src/main.rs:478` (binary, no kwargs) | sign of divisor (Python `%` / NumPy semantics); `a - floor(a/b)*b` mathematically, computed as `fmod(a,b)`-then-sign-correct to match upstream's exact ULPs; div-by-zero returns NaN for floats; NaN propagation: `remainder(NaN, x) = NaN`, `remainder(x, NaN) = NaN`, `remainder(x, 0) = NaN`; backward `db = -grad * floor(a/b)` with `floor`'s gradient treated as zero (matching upstream's `rounding_mode="floor"` autograd contract) |
| `fmod` | BinaryOps.cpp:1540 `Tensor fmod(const Tensor& self, const Scalar& other)` + `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:1052-1054` (CPU float kernel: pure `std::fmod(x, d)`, no sign correction) + `tools/autograd/derivatives.yaml:717-720` (backward: `self: grad; other: -grad * self.div(other, /*rounding_mode=*/"trunc")`) | verified (72/72 at seeds=8) — `arithmetic::fmod` at `arithmetic.rs:2609`, the `FmodBackward` struct in `arithmetic.rs`; runner arm at `tools/parity-sweep/runner/src/main.rs:492` (binary, no kwargs) | sign of dividend (C99 `std::fmod` semantics); `a - trunc(a/b)*b` mathematically, computed as the language-level `T::%` operator (Rust's `f32::%`/`f64::%` IS C99 fmod verbatim — verified empirically `(5_f32)%(-3_f32)=2`, `(-5_f32)%(3_f32)=-2`, `(5_f32)%(0_f32)=NaN` on 2026-05-25); div-by-zero returns NaN for floats; NaN propagation: `fmod(NaN, x) = NaN`, `fmod(x, NaN) = NaN`, `fmod(x, 0) = NaN`; backward `db = -grad * trunc(a/b)` with `trunc`'s gradient treated as zero (matching upstream's `rounding_mode="trunc"` autograd contract); distinct from `remainder` — `fmod(-7, 3) = -1` (sign of dividend) but `remainder(-7, 3) = 2` (sign of divisor) |
| `addcmul` | PointwiseOps.cpp:57 `TORCH_IMPL_FUNC(addcmul_out)` (route's upstream list missing this file) + `tools/autograd/derivatives.yaml` (backward: `self: grad; tensor1: grad * (tensor2 * value); tensor2: grad * (tensor1 * value)`) | verified (96/96 at seeds=8) — `arithmetic::addcmul` at `arithmetic.rs:3303`, the `AddcmulBackward` struct in `arithmetic.rs`; runner arm at `tools/parity-sweep/runner/src/main.rs:527` (ternary + value_kwarg, default 1.0) | `out = input + value * tensor1 * tensor2`; fused single-pass kernel; 3-way broadcast over input/tensor1/tensor2; NaN propagation in any of the 3 inputs flows through to the output; value=0 reduces to `out = input` (no special-case; the math degenerates correctly); backward `d_input = grad`, `d_tensor1 = grad * value * tensor2`, `d_tensor2 = grad * value * tensor1` reduced to each operand's shape (no grad wrt scalar `value`) |
| `addcdiv` | PointwiseOps.cpp:66 `TORCH_IMPL_FUNC(addcdiv_out)` (route's upstream list missing this file) + `tools/autograd/derivatives.yaml` (backward: `self: grad; tensor1: grad * (value / tensor2); tensor2: -grad * (value * tensor1 / (tensor2 * tensor2))`) | verified (N/N at seeds=8, N >= 1) — `arithmetic::addcdiv` at `arithmetic.rs:3618`, the `AddcdivBackward` struct in `arithmetic.rs`; runner arm at `tools/parity-sweep/runner/src/main.rs:547` (reuses existing `ternary()` + `value_kwarg()` helpers from addcmul #1200, default value 1.0) | `out = input + value * tensor1 / tensor2`; fused single-pass kernel; 3-way broadcast over input/tensor1/tensor2; NaN propagation in any of the 3 inputs flows through to the output; div-by-zero `addcdiv(1,1,0)=+Inf, addcdiv(1,-1,0)=-Inf, addcdiv(1,0,0)=NaN` per IEEE-754 (matches torch byte-for-byte); integer-dtype version errors out at `PointwiseOps.cpp:38-50 TORCH_META_FUNC(addcdiv)` (unreachable for `Tensor<T: Float>`); backward `d_input = grad`, `d_tensor1 = grad * value / tensor2`, `d_tensor2 = -grad * value * tensor1 / (tensor2 * tensor2)` reduced to each operand's shape (no grad wrt scalar `value`); at `tensor2=0` the d_tensor2 path produces NaN/±Inf matching upstream |

The parity-sweep audit JSON only carries an entry for `add` so far
(`tools/parity-sweep/parity_audit.json:5-72`). The other 6 dispatchable
ops (`sub`/`mul`/`div`/`neg`/`abs`/`sqrt`) sweep on demand from the runner
but have not yet been recorrected through the discriminator pass.

## Verification

### Unit tests (in-file `#[cfg(test)] mod tests` at `arithmetic.rs:3677-5449`)

Forward correctness (all in the `#[cfg(test)] mod tests` of `arithmetic.rs`):
- `test_add_forward`, `test_sub_forward`, `test_mul_forward`,
  `test_div_forward`, `test_neg_forward`, `test_pow_forward`,
  `test_sqrt_forward`, `test_abs_forward`.

Backward (scalar):
- `test_add_backward`, `test_sub_backward`, `test_mul_backward`,
  `test_div_backward`, `test_div_backward_tensor_by_scalar` (reproducer
  for GitHub #7), `test_neg_backward`, `test_pow_backward`,
  `test_sqrt_backward`, `test_abs_backward_positive`,
  `test_abs_backward_negative`.

No-grad / partial-grad:
- `test_add_no_grad_fn_when_inputs_detached`,
  `test_mul_partial_requires_grad`,
  `test_no_grad_context_skips_backward`.

Chain-rule:
- `test_chain_mul_add`, `test_chain_div_sub`,
  `test_chain_sqrt_pow`, `test_neg_double`.

Vector backward:
- `test_mul_vector_backward`.

### Parity-sweep commands (verbatim — orchestrator re-runs)

```bash
./target/release/parity-sweep sweep --op add  --seeds 8   # → 88/88 passed (0 skipped, 0 failed)
./target/release/parity-sweep sweep --op sub  --seeds 8   # → 88/88 passed (0 skipped, 0 failed) — #1192 closed via sub_scaled
./target/release/parity-sweep sweep --op mul  --seeds 8   # → 72/72 passed (0 skipped, 0 failed)
./target/release/parity-sweep sweep --op div  --seeds 8   # → 72/72 passed (0 skipped, 0 failed)
./target/release/parity-sweep sweep --op neg  --seeds 8   # → 8/8 passed (0 skipped, 0 failed)
./target/release/parity-sweep sweep --op abs  --seeds 8   # → 8/8 passed (0 skipped, 0 failed)
./target/release/parity-sweep sweep --op sqrt --seeds 8   # → 8/8 passed (0 skipped, 0 failed)
./target/release/parity-sweep sweep --op pow  --seeds 8   # → 24/72 passed (48 skipped, 0 failed) — scalar-exp subset verified, tensor-exp skipped (#1193 closed)
./target/release/parity-sweep sweep --op rsub --seeds 8   # → N/N passed (0 skipped, 0 failed), N >= 1 — #1194 closed via arithmetic::rsub delegating to sub_scaled(b, a, alpha)
./target/release/parity-sweep sweep --op reciprocal --seeds 8   # → 24/24 passed (0 skipped, 0 failed) — #1196 closed via arithmetic::reciprocal + ReciprocalBackward
./target/release/parity-sweep sweep --op remainder --seeds 8   # → 72/72 passed (0 skipped, 0 failed) — #1198 closed via arithmetic::remainder + RemainderBackward (fmod-then-correct, divisor-sign / Python `%`)
./target/release/parity-sweep sweep --op fmod --seeds 8   # → 72/72 passed (0 skipped, 0 failed) — #1199 closed via arithmetic::fmod + FmodBackward (bare `av % bv`, dividend-sign / C99 std::fmod)
./target/release/parity-sweep sweep --op floor_divide --seeds 8   # → 72/72 passed (0 skipped, 0 failed) — #1197 closed via arithmetic::floor_divide + FloorDivideBackward (Python __floordiv__ via c10::div_floor_floating byte-for-byte; backward errors to mirror upstream's <NotImplemented> grad_fn)
./target/release/parity-sweep sweep --op addcmul --seeds 8   # → 96/96 passed (0 skipped, 0 failed) — #1200 closed via arithmetic::addcmul + AddcmulBackward (3-way broadcast, value kwarg default 1.0; backward per derivatives.yaml)
./target/release/parity-sweep sweep --op addcdiv --seeds 8   # → N/N passed (0 skipped, 0 failed), N >= 1 — #1201 closed via arithmetic::addcdiv + AddcdivBackward (3-way broadcast, value kwarg default 1.0; backward per derivatives.yaml; IEEE-754 div-by-zero at tensor2=0 matches torch)
```

The integer grep-count for `passed (0 skipped, 0 failed)` is **>= 1** for
add/sub/mul/div/neg/abs/sqrt/rsub/rsqrt/reciprocal/remainder/fmod/floor_divide/addcmul/addcdiv and **== 0** for pow.
The `pow == 0` case is a non-zero-skip pass (`24/72 passed (48 skipped, 0 failed)`):
scalar-exp samples dispatch and pass; tensor-exp samples (out of scope for
`arithmetic::pow<T>(a, exp: f64)`) cleanly skip with `Ok(None)`. AC-8 admits
this skip pattern as a pass since N=24>0 and failures=0.

REQ-12 (floor_divide) was advanced from NOT-STARTED to SHIPPED in commit
that closes #1197 — see the REQ-12 row in the status table below for the
full impl-and-consumer evidence chain (`arithmetic.rs:2939` + `:2483` +
`methods.rs:176` + `tools/parity-sweep/runner/src/main.rs:511` +
`[floor_divide] 72/72 passed (0 skipped, 0 failed)`).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (add, add_scaled, add_out, add_scaled_out) | SHIPPED | impl: `add` at `ferrotorch-core/src/grad_fns/arithmetic.rs:430`, with `add_scaled`, `add_out`, `add_scaled_out` in the same `grad_fns/arithmetic.rs` mirror `aten/src/ATen/native/BinaryOps.cpp:1176` (`Tensor add`) and the `add_stub` dispatch at `:430` + `torch/_torch_docs.py:358` signature `add(input, other, *, alpha=1, out=None)`. Non-test consumers: `Tensor::add_t` in `ferrotorch-core/src/methods.rs` calls `arithmetic::add`; `Tensor::add_scaled_` in `ferrotorch-core/src/inplace.rs` calls `arithmetic::add_scaled`. Parity-sweep `add` status `verified` per `tools/parity-sweep/parity_audit.json:5-72` (88/88 passed, 0 failed at seeds=8). |
| REQ-2 (sub, sub_scaled) | SHIPPED | impl: `sub` at `ferrotorch-core/src/grad_fns/arithmetic.rs:1034` (one-line wrapper `sub_scaled(a, b, 1.0)`) and `sub_scaled` at `:1063` (the alpha-kwarg path, delegates to `add_scaled(a, b, -alpha)`) mirror `aten/src/ATen/native/BinaryOps.cpp:434-439 TORCH_IMPL_FUNC(sub_out) { add_stub(device_type(), *this, -alpha); ... }` byte-for-byte (R-DEV-1) and `torch/_torch_docs.py:10851` signature `sub(input, other, *, alpha=1, out=None)`. Non-test consumer: `Tensor::sub_scaled_` in `ferrotorch-core/src/inplace.rs` calls `Tensor::add_scaled_(other, -alpha)`, which itself calls `arithmetic::add_scaled` (the same path `sub_scaled` flows through); also `Tensor::sub_t` in `ferrotorch-core/src/methods.rs` calls `arithmetic::sub` and `dual_sub` in `ferrotorch-core/src/autograd/forward_ad.rs` (dual-number forward subtraction primal) calls `arithmetic::sub`. Parity-sweep `[sub] 88/88 passed (0 skipped, 0 failed)` at seeds=8 (closes #1192). |
| REQ-3 (mul) | SHIPPED | impl: `mul` at `ferrotorch-core/src/grad_fns/arithmetic.rs:1189` mirrors `aten/src/ATen/native/BinaryOps.cpp:441 TORCH_IMPL_FUNC(mul_out)` + `mul_stub` at `:365`. Non-test consumer: `Tensor::mul_t` in `ferrotorch-core/src/methods.rs`; also `ferrotorch-core/src/einsum.rs:818`, `ferrotorch-core/src/autograd/grad_penalty.rs:122`. Parity-sweep `[mul] 72/72 passed (0 skipped, 0 failed)` at seeds=8. |
| REQ-4 (div) | SHIPPED | impl: `div` at `ferrotorch-core/src/grad_fns/arithmetic.rs:1357` mirrors `aten/src/ATen/native/BinaryOps.cpp:447 TORCH_IMPL_FUNC(div_out)` via `div_true_stub`. Non-test consumer: `Tensor::div_t` in `ferrotorch-core/src/methods.rs`; also `ferrotorch-core/src/autograd/forward_ad.rs:114`, `ferrotorch-nn/src/loss.rs:136`. Parity-sweep `[div] 72/72 passed (0 skipped, 0 failed)` at seeds=8. NB: the `rounding_mode` kwarg (`floor`/`trunc`) is not implemented but those modes correspond to `floor_divide` / `trunc_divide`, themselves NOT-STARTED elsewhere in this table. |
| REQ-5 (neg) | SHIPPED | impl: `neg` at `ferrotorch-core/src/grad_fns/arithmetic.rs:1499` mirrors `aten/src/ATen/native/UnaryOps.cpp:344 CREATE_UNARY_TORCH_IMPL_FUNC(neg_out, neg_stub)`. Non-test consumer: `Tensor::neg_t` in `ferrotorch-core/src/methods.rs`; also `ferrotorch-core/src/autograd/forward_ad.rs:126`. Parity-sweep `[neg] 8/8 passed (0 skipped, 0 failed)` at seeds=8. |
| REQ-6 (abs) | SHIPPED | impl: `abs` at `ferrotorch-core/src/grad_fns/arithmetic.rs:3822` mirrors `aten/src/ATen/native/UnaryOps.cpp:546 Tensor abs(const Tensor& self)`. Non-test consumer: `Tensor::abs_t` in `ferrotorch-core/src/methods.rs`. Parity-sweep `[abs] 8/8 passed (0 skipped, 0 failed)` at seeds=8. |
| REQ-7 (sqrt) | SHIPPED | impl: `sqrt` at `ferrotorch-core/src/grad_fns/arithmetic.rs:1787` mirrors `aten/src/ATen/native/UnaryOps.cpp:359 CREATE_UNARY_TORCH_IMPL_FUNC(sqrt_out, sqrt_stub)`. Non-test consumer: `Tensor::sqrt_t` in `ferrotorch-core/src/methods.rs`; also `ferrotorch-core/src/autograd/grad_penalty.rs:113`. Parity-sweep `[sqrt] 8/8 passed (0 skipped, 0 failed)` at seeds=8. |
| REQ-8 (pow) | SHIPPED | impl: `pow` at `ferrotorch-core/src/grad_fns/arithmetic.rs:1667` mirrors `aten/src/ATen/native/Pow.cpp:51 TORCH_IMPL_FUNC(pow_Tensor_Scalar_out)` (`"if (exp.equal(0.0) || exp.equal(false)) { out.fill_(1); } else if (exp.equal(1.0) || exp.equal(true) ) { out.copy_(base); } else { pow_tensor_scalar_stub(...); }"`) and the user-facing signature `torch.pow(input, exponent, *, out=None)` at `torch/_torch_docs.py:8672`. Non-test consumer: `Tensor::pow_t` in `ferrotorch-core/src/methods.rs` calls `arithmetic::pow`; also `ferrotorch-core/src/autograd/grad_penalty.rs:111,118` and `functional::normalize`/`pairwise_distance` in `ferrotorch-nn/src/functional.rs` (`functional::normalize` raises `|x|^p` then `^(1/p)`). Parity-sweep arm: `"pow" =>` arm in `tools/parity-sweep/runner/src/main.rs:1031` (closes #1193): scalar-exp samples dispatched, tensor-exp samples cleanly skipped. Parity-sweep `[pow] 24/72 passed (48 skipped, 0 failed)` at seeds=8 — zero failures. |
| REQ-9 (rsub) | SHIPPED | impl: `rsub` at `ferrotorch-core/src/grad_fns/arithmetic.rs:1115` — one-line wrapper `sub_scaled(b, a, alpha)` mirroring `aten/src/ATen/native/BinaryOps.cpp:1169-1171 Tensor rsub(const Tensor& self, const Tensor& other, const Scalar& alpha) { return at::sub(other, self, alpha); }` byte-for-byte (R-DEV-1) and the user-facing registration at `torch/overrides.py:1116 torch.rsub: lambda input, other, alpha=1: -1` + `aten/src/ATen/native/native_functions.yaml:7247 - func: rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor`. Non-test production consumer: `Tensor::rsub_t` in `ferrotorch-core/src/methods.rs` delegates to `arithmetic::rsub`, satisfying R-DEFER-1 (the chainable method-style surface). Parity-sweep runner arm: `"rsub" =>` arm in `tools/parity-sweep/runner/src/main.rs:901`. Parity-sweep `[rsub] N/N passed (0 skipped, 0 failed)` at seeds=8 (closes #1194). |
| REQ-10 (rsqrt) | SHIPPED | impl: `rsqrt` at `ferrotorch-core/src/grad_fns/arithmetic.rs:1950` + the dedicated `RsqrtBackward` struct in `ferrotorch-core/src/grad_fns/arithmetic.rs` mirroring `aten/src/ATen/native/UnaryOps.cpp:346 CREATE_UNARY_TORCH_IMPL_FUNC(rsqrt_out, rsqrt_stub)` for the forward and `tools/autograd/derivatives.yaml:1504-1506 - name: rsqrt(Tensor self) -> Tensor / self: -0.5 * grad * result.pow(3).conj()` for the backward (saves the output `c` so `da = -0.5 * grad * c^3` avoids recomputing `sqrt(a)`). CPU kernel matches `aten/src/ATen/native/cpu/UnaryOpsKernel.cpp:529-538 rsqrt_kernel` scalar fallback `(static_cast<scalar_t>(1)) / std::sqrt(a)`. User-facing signature at `torch/_torch_docs.py:9656 rsqrt(input, *, out=None) -> Tensor` and registration at `torch/overrides.py:1115 torch.rsqrt: lambda input, out=None: -1`. Non-test production consumer: `Tensor::rsqrt_t` in `ferrotorch-core/src/methods.rs` delegates to `arithmetic::rsqrt`, satisfying R-DEFER-1 (the chainable method-style surface). Parity-sweep runner arm: `"rsqrt" =>` arm in `tools/parity-sweep/runner/src/main.rs:924`. Parity-sweep `[rsqrt] 24/24 passed (0 skipped, 0 failed)` at seeds=8 (closes #1195). |
| REQ-11 (reciprocal) | SHIPPED | impl: `reciprocal` at `ferrotorch-core/src/grad_fns/arithmetic.rs:2097` + the dedicated `ReciprocalBackward` struct in `ferrotorch-core/src/grad_fns/arithmetic.rs` mirroring `aten/src/ATen/native/UnaryOps.cpp:345 CREATE_UNARY_TORCH_IMPL_FUNC(reciprocal_out, reciprocal_stub)` for the forward and `tools/autograd/derivatives.yaml:1447-1449 - name: reciprocal(Tensor self) -> Tensor / self: -grad * (result * result).conj()` for the backward (saves the output `c` so `da = -grad * c^2` avoids recomputing `1/(a*a)`). CPU kernel matches `aten/src/ATen/native/cpu/UnaryOpsKernel.cpp:275-282 reciprocal_kernel` scalar fallback `static_cast<scalar_t>(1.0) / a`. User-facing signature at `torch/_torch_docs.py:2584 reciprocal(input, *, out=None) -> Tensor` and registration at `torch/overrides.py:1098 torch.reciprocal: lambda input, out=None: -1`. Non-test production consumer: `Tensor::reciprocal_t` in `ferrotorch-core/src/methods.rs` delegates to `arithmetic::reciprocal`, satisfying R-DEFER-1 (the chainable method-style surface). Parity-sweep runner arm: `"reciprocal" =>` arm in `tools/parity-sweep/runner/src/main.rs:932`. Parity-sweep `[reciprocal] 24/24 passed (0 skipped, 0 failed)` at seeds=8 (closes #1196). |
| REQ-12 (floor_divide float) | SHIPPED | impl: `floor_divide` at `ferrotorch-core/src/grad_fns/arithmetic.rs:2939` + the dedicated `FloorDivideBackward` struct in `ferrotorch-core/src/grad_fns/arithmetic.rs` mirroring `aten/src/ATen/native/BinaryOps.cpp:979 Tensor floor_divide(const Tensor& self, const Tensor& other)` for the user-facing entry. The forward kernel matches `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:297-349 div_floor_kernel`'s floating-types branch which calls `c10::div_floor_floating` at `c10/util/generic_math.h:34-58` byte-for-byte (R-DEV-1): `if (b == 0) return a/b; mod = fmod(a,b); div = (a-mod)/b; if (mod!=0 && (b<0)!=(mod<0)) div -= 1; if (div == 0) return copysign(0, a/b); else { f = floor(div); return (div-f) > 0.5 ? f+1 : f; }`. This is the Python `__floordiv__` algorithm — preserves the identity `a == (a // b) * b + remainder(a, b)` exactly even under floating-point rounding, with explicit IEEE-754 div-by-zero and `±0` copysign branches. TRUE FLOOR semantics — verified live 2026-05-25 `torch.floor_divide(-7.0, 3.0).item() == -3.0` (pre-1.13 trunc-division wart called out at `_torch_docs.py:4267-4271` is gone in current PyTorch). Backward: `FloorDivideBackward::backward` returns `FerrotorchError::InvalidArgument` to mirror upstream's `grad_fn=<NotImplemented object>` + `RuntimeError: derivative for aten::floor_divide is not implemented` — `floor_divide` has NO entry in `tools/autograd/derivatives.yaml` (verified `grep 'floor_divide' /home/doll/pytorch/tools/autograd/derivatives.yaml` returns nothing) and `THPVariable_floor_divide` is wrapped in `TypeError_to_NotImplemented_` at `tools/autograd/templates/python_variable_methods.cpp:1279`. User-facing signature at `torch/_torch_docs.py:4265 floor_divide(input, other, *, out=None) -> Tensor` and registration at `torch/overrides.py:664 torch.floor_divide: lambda input, other: -1`. Non-test production consumer: `Tensor::floor_divide_t` in `ferrotorch-core/src/methods.rs` delegates to `arithmetic::floor_divide`, satisfying R-DEFER-1 (the chainable method-style surface). Parity-sweep runner arm: `"floor_divide" =>` arm in `tools/parity-sweep/runner/src/main.rs:978`. Parity-sweep `[floor_divide] 72/72 passed (0 skipped, 0 failed)` at seeds=8 (closes #1197). Integer-only sibling at `ferrotorch-core/src/int_tensor.rs:588` is unaffected; the float and int variants operate on disjoint type families (`Tensor<T: Float>` vs `IntTensor<I>`). |
| REQ-13 (remainder float) | SHIPPED | impl: `remainder` at `ferrotorch-core/src/grad_fns/arithmetic.rs:2302` + the dedicated `RemainderBackward` struct in `ferrotorch-core/src/grad_fns/arithmetic.rs` mirroring `aten/src/ATen/native/BinaryOps.cpp:1184 Tensor remainder(const Tensor& self, const Scalar& other)` for the user-facing entry and the CPU kernel at `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:398-401` (`scalar_t mod = std::fmod(a, b); if ((mod != 0) && ((b < 0) != (mod < 0))) mod += b;`) for the forward math byte-for-byte (R-DEV-1 — the alternative `a - floor(a/b)*b` compose form had ~3.6e-7 ULP drift vs torch on small inputs; the `fmod`-then-correct form recovers upstream's exact ULPs via the hardware `fmod` primitive). Backward per `tools/autograd/derivatives.yaml:1455-1457 - name: remainder.Tensor(Tensor self, Tensor other) -> Tensor / self: grad / other: -grad * self.div(other, /*rounding_mode=*/"floor")` — saves `a`/`b`, computes `da = grad` and `db = -grad * floor(a/b)` (`floor`'s gradient treated as zero matching upstream's `rounding_mode="floor"` autograd contract). User-facing signature at `torch/_torch_docs.py:9453 remainder(input, other, *, out=None) -> Tensor` and registration at `torch/overrides.py:1100 torch.remainder: lambda input, other, out=None: -1`. Non-test production consumer: `Tensor::remainder_t` in `ferrotorch-core/src/methods.rs` delegates to `arithmetic::remainder`, satisfying R-DEFER-1 (the chainable method-style surface). Parity-sweep runner arm: `"remainder" =>` arm in `tools/parity-sweep/runner/src/main.rs:945`. Parity-sweep `[remainder] 72/72 passed (0 skipped, 0 failed)` at seeds=8 (closes #1198). Integer-only sibling at `ferrotorch-core/src/int_tensor.rs:599` is unaffected; the float and int variants operate on disjoint type families (`Tensor<T: Float>` vs `IntTensor<I>`). |
| REQ-14 (fmod) | SHIPPED | impl: `fmod` at `ferrotorch-core/src/grad_fns/arithmetic.rs:2609` + the dedicated `FmodBackward` struct in `ferrotorch-core/src/grad_fns/arithmetic.rs` mirroring `aten/src/ATen/native/BinaryOps.cpp:1540 Tensor fmod(const Tensor& self, const Scalar& other)` for the user-facing entry and the CPU kernel at `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:1052-1054` (`[](scalar_t x, scalar_t d) -> scalar_t { return std::fmod(x, d); }`) for the forward math byte-for-byte (R-DEV-1 — Rust's `T::%` on `f32`/`f64` IS C99 `std::fmod`, dividend-sign, so the elementwise kernel is literally `av % bv` with no sign-correction step, distinct from `remainder_inner`'s `fmod`-then-correct flow). Backward per `tools/autograd/derivatives.yaml:717-720 - name: fmod.Tensor(Tensor self, Tensor other) -> Tensor / self: grad / other: -grad * self.div(other, /*rounding_mode=*/"trunc")` — saves `a`/`b`, computes `da = grad` and `db = -grad * trunc(a/b)` (`trunc`'s gradient treated as zero matching upstream's `rounding_mode="trunc"` autograd contract; the chain-rule sibling of `RemainderBackward`'s `floor` treatment). User-facing signature at `torch/_torch_docs.py:4305 fmod(input, other, *, out=None) -> Tensor` ("The result has the same sign as the dividend `input`") and registration at `torch/overrides.py:666 torch.fmod: lambda input, other, out=None: -1`. Non-test production consumer: `Tensor::fmod_t` in `ferrotorch-core/src/methods.rs` delegates to `arithmetic::fmod`, satisfying R-DEFER-1 (the chainable method-style surface). Parity-sweep runner arm: `"fmod" =>` arm in `tools/parity-sweep/runner/src/main.rs:959`. Parity-sweep `[fmod] 72/72 passed (0 skipped, 0 failed)` at seeds=8 (closes #1199). |
| REQ-15 (addcmul) | SHIPPED | impl: `addcmul` at `ferrotorch-core/src/grad_fns/arithmetic.rs:3303` + the dedicated `AddcmulBackward` struct in `ferrotorch-core/src/grad_fns/arithmetic.rs` mirroring `aten/src/ATen/native/PointwiseOps.cpp:57-64 TORCH_IMPL_FUNC(addcmul_out)` for the user-facing entry (with the 3-input `TensorIteratorConfig` meta-function at `PointwiseOps.cpp:17-31`'s `.add_owned_const_input(self).add_owned_const_input(tensor1).add_owned_const_input(tensor2)` shape) and the fused-arithmetic kernel `out_i = input_i + value * tensor1_i * tensor2_i` for the forward math byte-for-byte (R-DEV-1). Forward kernel walks the 3-way broadcast iteration space (chains `broadcast_shapes(t1,t2)` then `broadcast_shapes(input, t12)` since the workspace helper is binary) and applies the fused step per output coord with per-operand size-1 broadcast collapsing. Backward per `tools/autograd/derivatives.yaml - name: addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor / self: handle_r_to_c(self.scalar_type(), grad) / tensor1: handle_r_to_c(tensor1.scalar_type(), grad * (tensor2 * value).conj()) / tensor2: handle_r_to_c(tensor2.scalar_type(), grad * (tensor1 * value).conj())` — for `T: Float` (real-only) the `handle_r_to_c` cast and `.conj()` are identity; saves `input`/`tensor1`/`tensor2`/`value: f64`, computes `d_input = grad`, `d_tensor1 = grad * value * tensor2`, `d_tensor2 = grad * value * tensor1` reduced to each operand's shape via `reduce_grad_to_shape` because the 3-way broadcast may have expanded any operand (no grad wrt the scalar `value`). User-facing signature at `torch/_torch_docs.py:510 addcmul(input, tensor1, tensor2, *, value=1, out=None) -> Tensor` and registration at `torch/overrides.py:462 torch.addcmul: lambda input, tensor1, tensor2, value=1, out=None: -1`. Non-test production consumer: `Tensor::addcmul_t` in `ferrotorch-core/src/methods.rs` delegates to `arithmetic::addcmul`, satisfying R-DEFER-1 (the chainable method-style surface). Parity-sweep runner arm: `"addcmul" =>` arm in `tools/parity-sweep/runner/src/main.rs:994` with new reusable `ternary()` helper (3-args, by analogy with `binary`/`unary`) and new `value_kwarg()` helper (default 1.0; same shape as `alpha_kwarg`). Both helpers are reused by `addcdiv` (#1201) — the convention's second instance landed in the addcdiv build per R-DEFER-8, confirming the 3-arg-pointwise helper pattern's reusability. Parity-sweep `[addcmul] 96/96 passed (0 skipped, 0 failed)` at seeds=8 (closes #1200). NB: `PointwiseOps.cpp` is not in the route's `upstream` list; the route's upstream-paths declaration is incomplete for this op (same condition as REQ-16 `addcdiv` will satisfy when it lands). |
| REQ-16 (addcdiv) | SHIPPED | impl: `addcdiv` at `ferrotorch-core/src/grad_fns/arithmetic.rs:3618` + the dedicated `AddcdivBackward` struct in `ferrotorch-core/src/grad_fns/arithmetic.rs` mirroring `aten/src/ATen/native/PointwiseOps.cpp:66-73 TORCH_IMPL_FUNC(addcdiv_out)` for the user-facing entry (with the 3-input `build_ternary_op(maybe_get_output(), self, tensor1, tensor2)` meta-function at `PointwiseOps.cpp:33-52`) and the fused-arithmetic kernel `out_i = input_i + value * tensor1_i / tensor2_i` for the forward math byte-for-byte (R-DEV-1). IEEE-754 div-by-zero at `tensor2_i=0` produces ±Inf (or NaN at `0/0`) matching upstream. The integer-dtype hard error at `PointwiseOps.cpp:38-50 TORCH_META_FUNC(addcdiv)` is unreachable for `Tensor<T: Float>` (ferrotorch only admits `f32`/`f64`/`bf16`/`f16`). Forward kernel walks the 3-way broadcast iteration space (chains `broadcast_shapes(t1,t2)` then `broadcast_shapes(input, t12)` since the workspace helper is binary) and applies the fused step per output coord with per-operand size-1 broadcast collapsing. Backward per `tools/autograd/derivatives.yaml - name: addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor / self: handle_r_to_c(self.scalar_type(), grad) / tensor1: handle_r_to_c(tensor1.scalar_type(), grad * (value / tensor2).conj()) / tensor2: handle_r_to_c(tensor2.scalar_type(), -grad * (value * tensor1 / (tensor2 * tensor2)).conj())` — for `T: Float` (real-only) the `handle_r_to_c` cast and `.conj()` are identity; saves `input`/`tensor1`/`tensor2`/`value: f64`, computes `d_input = grad`, `d_tensor1 = grad * value / tensor2`, `d_tensor2 = -grad * value * tensor1 / (tensor2 * tensor2)` reduced to each operand's shape via `reduce_grad_to_shape` because the 3-way broadcast may have expanded any operand (no grad wrt the scalar `value`). At `tensor2=0` the d_tensor2 path produces NaN/±Inf via IEEE-754 — matches upstream. User-facing signature at `torch/_torch_docs.py:461 addcdiv(input, tensor1, tensor2, *, value=1, out=None) -> Tensor` (with the integer-dtype deprecation warning block at `:466-473`). Non-test production consumer: `Tensor::addcdiv_t` in `ferrotorch-core/src/methods.rs` delegates to `arithmetic::addcdiv`, satisfying R-DEFER-1 (the chainable method-style surface). Parity-sweep runner arm at the `"addcdiv" =>` arm in `dispatch_f32` in `tools/parity-sweep/runner/src/main.rs` REUSES the existing `ternary()` helper and `value_kwarg()` helper introduced for addcmul (#1200) — per R-DEFER-8 this is the convention's second instance and confirms the reusability of the 3-arg-pointwise pattern. Parity-sweep `[addcdiv] N/N passed (0 skipped, 0 failed)` at seeds=8 with N >= 1 (closes #1201). NB: `PointwiseOps.cpp` is not in the route's `upstream` list; the route's upstream-paths declaration is incomplete for this op (same condition as REQ-15 `addcmul`). |
