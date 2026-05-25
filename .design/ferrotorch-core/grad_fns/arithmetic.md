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

- REQ-2: `sub(a, b)` — forward `c = a - b` with broadcasting + autograd
  (`SubBackward` VJP returns `(grad, -grad)` reduced). PyTorch's
  `torch.sub(input, other, *, alpha=1, out=None)` also takes an `alpha`
  kwarg (BinaryOps.cpp:434–438 — `TORCH_IMPL_FUNC(sub_out)` calls
  `add_stub(device_type(), *this, -alpha)` so `sub` is literally `add` with
  negated alpha). ferrotorch's `sub` has NO alpha kwarg and NO `sub_scaled`
  sibling — this is the same gap the add reader-corrector closed for `add`,
  unfixed for `sub`.

- REQ-3: `mul(a, b)` — forward `c = a * b` with broadcasting + autograd
  (`MulBackward` VJP returns `(grad*b, grad*a)` reduced). Mirrors `mul_stub`
  in BinaryOps.cpp:441 `TORCH_IMPL_FUNC(mul_out)`. `MulBackward::backward`
  additionally implements the higher-order grad path (when `grad_output`
  itself has `grad_fn` / `requires_grad`, it routes through the
  differentiable `mul` ops so the backward pass is itself recorded in the
  graph).

- REQ-4: `div(a, b)` — forward `c = a / b` with broadcasting + autograd
  (`DivBackward` VJP `(grad/b, -grad*a/(b*b))` reduced). Mirrors
  `div_true_stub` at BinaryOps.cpp:447 `TORCH_IMPL_FUNC(div_out)`. Division by
  zero produces `±inf` / `NaN` per IEEE-754, matching torch.

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
  `Tensor rsub(const Tensor& self, const Tensor& other, const Scalar& alpha)`
  and overrides.py:1116
  `torch.rsub: lambda input, other, alpha=1: -1`. NOT IMPLEMENTED in
  ferrotorch.

- REQ-10: `rsqrt(a)` — `torch.rsqrt(input)` is `1 / sqrt(input)`. Per
  UnaryOps.cpp:346 `CREATE_UNARY_TORCH_IMPL_FUNC(rsqrt_out, rsqrt_stub)` and
  overrides.py:1115 `torch.rsqrt: lambda input, out=None: -1`. NOT IMPLEMENTED
  in ferrotorch.

- REQ-11: `reciprocal(a)` — `torch.reciprocal(input)` is `1 / input`. Per
  UnaryOps.cpp:345 `CREATE_UNARY_TORCH_IMPL_FUNC(reciprocal_out,
  reciprocal_stub)` and overrides.py:1098
  `torch.reciprocal: lambda input, out=None: -1`. NOT IMPLEMENTED in
  ferrotorch.

- REQ-12: `floor_divide(a, b)` — `torch.floor_divide(input, other)` floors
  toward `-inf`. Per BinaryOps.cpp:979
  `Tensor floor_divide(const Tensor& self, const Tensor& other)` and
  overrides.py:664 `torch.floor_divide: lambda input, other: -1`. NOT
  IMPLEMENTED in `arithmetic.rs` for `Float` dtypes (an integer-only sibling
  exists at `ferrotorch-core/src/int_tensor.rs:588`, but that does not
  satisfy `torch.floor_divide` over float tensors).

- REQ-13: `remainder(a, b)` — `torch.remainder(input, other)` returns the
  remainder with the SIGN OF THE DIVISOR (Python `%` / NumPy semantics).
  Per BinaryOps.cpp:1184
  `Tensor remainder(const Tensor& self, const Scalar& other)` and
  overrides.py:1100 `torch.remainder: lambda input, other, out=None: -1`.
  NOT IMPLEMENTED in `arithmetic.rs` for `Float` dtypes (integer-only
  variant at `int_tensor.rs:599`).

- REQ-14: `fmod(a, b)` — `torch.fmod(input, other)` returns the remainder
  with the SIGN OF THE DIVIDEND (C99 `fmod` semantics). Per
  BinaryOps.cpp:1540 `Tensor fmod(const Tensor& self, const Scalar& other)`
  and overrides.py:666 `torch.fmod: lambda input, other, out=None: -1`. NOT
  IMPLEMENTED in ferrotorch.

- REQ-15: `addcmul(input, tensor1, tensor2, *, value=1)` — fused
  `out = input + value * tensor1 * tensor2`. Per
  `aten/src/ATen/native/PointwiseOps.cpp:57` `TORCH_IMPL_FUNC(addcmul_out)`
  and `_torch_docs.py:510`
  `addcmul(input, tensor1, tensor2, *, value=1, out=None) -> Tensor`. NOT
  IMPLEMENTED in ferrotorch. NB: `PointwiseOps.cpp` is the true upstream
  location; the route's declared `upstream` list (BinaryOps + UnaryOps)
  does not include it, which is itself an incomplete route declaration.

- REQ-16: `addcdiv(input, tensor1, tensor2, *, value=1)` — fused
  `out = input + value * tensor1 / tensor2`. Per
  `aten/src/ATen/native/PointwiseOps.cpp:66` `TORCH_IMPL_FUNC(addcdiv_out)`
  and `_torch_docs.py:461`
  `addcdiv(input, tensor1, tensor2, *, value=1, out=None) -> Tensor`. NOT
  IMPLEMENTED in ferrotorch.

## Acceptance Criteria

- [x] AC-1: `add` parity-sweep at `--seeds 8` returns `[add] 88/88 passed
  (0 skipped, 0 failed)` (grep-count `passed (0 skipped, 0 failed)` == 1).
- [ ] AC-2: `sub` parity-sweep at `--seeds 8` returns 0 failures. Currently
  `[sub] 72/88 passed (0 skipped, 16 failed)` — the 16 failures all hit
  shape `[5, 5]` at sample indices `i=9` and `i=10`, which are the
  `alpha=2` and `alpha=-3.125` op_db samples (same root-cause as the
  original add alpha-gap blocker; blocker #1192).
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
- [ ] AC-8: `pow` parity-sweep at `--seeds 8` returns a non-skip pass —
  currently `[pow] 0/72 passed (72 skipped, 0 failed)` because the runner
  at `tools/parity-sweep/runner/src/main.rs:214-231` has no `pow` arm in
  its dispatch match. Blocker #1193.
- [ ] AC-9..AC-16: parity-sweep for `rsub`, `rsqrt`, `reciprocal`,
  `floor_divide`, `remainder`, `fmod`, `addcmul`, `addcdiv` each return
  `[<op>] N/N passed (0 skipped, 0 failed)`. None of these ops exist in
  `arithmetic.rs`, so each currently `no_dispatch`-skips. Blockers
  #1194–#1201.
- [x] AC-17: `cargo test -p ferrotorch-core --lib grad_fns::arithmetic`
  passes (the `tests` mod at `arithmetic.rs:1688-2076` covers forward,
  backward, partial-requires-grad, no-grad, and chain-rule cases for
  `add`/`sub`/`mul`/`div`/`neg`/`pow`/`sqrt`/`abs`).
- [x] AC-18: Non-contiguous CUDA views (transpose / narrow / permute) flow
  through every binary forward without `LengthMismatch` — covered by
  `ensure_contig_for_gpu` at `arithmetic.rs:73-118` (#812 cluster).
- [x] AC-19: `add_scaled_out` correctly resizes `out` when its shape does
  not match the broadcast shape (matches torch's deprecation-warned
  silent-resize behavior in 2.x). Covered by parity-sweep `add` probes
  `out_basic` / `out_with_alpha` / `out_broadcast` / `out_wrong_shape` /
  `out_nan` documented in `tools/parity-sweep/parity_audit.json:54-70`.

## Architecture

### Layer-1 helpers (lines 18-310)

`is_f32` / `is_f64` / `is_bf16` / `is_f16` (`arithmetic.rs:18-40`) are
`TypeId`-based dtype discriminators used to gate the GPU dispatch arms.
`ensure_contig_for_gpu` (`arithmetic.rs:72-118`) is the #812 fix-up that
guarantees a CUDA tensor handed to a raw `gpu_handle()` kernel has
`storage_len == numel` and `storage_offset == 0`; non-contiguous views
route through the on-device `strided_copy_{f32,f64}` kernels rather than
detouring through host memory. `needs_grad` / `needs_grad_unary`
(`arithmetic.rs:125-134`) check `is_grad_enabled()` + per-tensor
`requires_grad()`. `reduce_grad_to_shape` (`arithmetic.rs:153-311`) is the
shared backward broadcast-reduction primitive: a GPU-resident path for
f32/f64 (uses `backend.sum_axis_{f32,f64}` after materializing the grad
view), a same-numel-different-rank reshape branch (`#814`), and a CPU
fallback loop that decomposes the grad flat index into per-axis coords and
sums into the target.

### REQ-1 `add` (lines 320-463)

`AddBackward` saves both operands; `backward` returns
`(reduce(grad, a.shape()), reduce(grad, b.shape()))` (`arithmetic.rs:326-348`).
The forward `pub fn add` (`arithmetic.rs:351-366`) emits a profiler scope,
checks device-match, lets `meta_propagate::binary_broadcast` short-circuit
for meta tensors, then calls `add_inner`. `add_inner` (`arithmetic.rs:368-463`)
materializes both CUDA inputs if needed, picks between `add_*` and
`broadcast_add_*` via `dispatch_floating_dtype!` (f32/f64/bf16/f16), and
falls through to `fast_add` on CPU. **Non-test consumer**:
`ferrotorch-core/src/methods.rs:14-16` — `Tensor::add_t(&self, other)`
delegates to `crate::grad_fns::arithmetic::add`, exposing the op as a
method on every `Tensor<T: Float>` and used pervasively throughout
ferrotorch (`autograd::forward_ad`, `autograd::higher_order`,
`autograd::fixed_point`, `autograd::grad_penalty`, `ops::higher_order`,
`einops`, `einsum`, `vmap`, `meta_propagate`).

### REQ-1 extension: `add_scaled` + `add_out` / `add_scaled_out` (lines 466-775)

`AddScaledBackward` (`arithmetic.rs:472-513`) saves `a`, `b`, and `alpha:
f64`; backward returns `(reduce(grad, a.shape()), reduce(alpha*grad,
b.shape()))`. `scale_tensor` (`arithmetic.rs:522-553`) is a private helper
that routes to dtype-specialised GPU `scale_*` kernels (f32/f64/bf16/f16)
or to `scalar_map` on CPU. `check_out_allowed` (`arithmetic.rs:591-610`)
enforces torch's `out=` rules: no grad_fn, no requires_grad-leaf. `add_out`
(`arithmetic.rs:619-625`) is an `alpha=1.0` wrapper over `add_scaled_out`.
`add_scaled_out` (`arithmetic.rs:650-709`) validates devices, computes
`add_scaled` under `no_grad`, and writes through
`Tensor::update_storage` (matched-shape branch) or
`Tensor::update_storage_and_shape` (resize branch) — both are `unsafe` and
documented with SAFETY comments tying back to the `check_out_allowed`
proof. `add_scaled` (`arithmetic.rs:716-775`) shortcircuits `alpha == 1.0`
to plain `add`, pre-scales `b` under `no_grad`, calls `add_inner`, then
attaches `AddScaledBackward` if either operand requires grad. **Non-test
consumers**: `inplace.rs:213` — `Tensor::add_scaled_` invokes
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

### REQ-2 `sub` (lines 781-922)

`SubBackward` (`arithmetic.rs:790-813`) saves `a`/`b`; backward returns
`(reduce(grad, a.shape()), reduce(-grad, b.shape()))` via a local
`neg(grad)` call under `no_grad`. The forward `pub fn sub`
(`arithmetic.rs:816-831`) and `sub_inner` (`arithmetic.rs:833-922`) follow
the same shape as `add` — same `dispatch_floating_dtype!` macro for
`sub_{f32,f64,bf16,f16}` / `broadcast_sub_{...}`, same CPU `fast_sub`
fallthrough. **Non-test consumer**: `methods.rs:18-20`
`Tensor::sub_t`; also `autograd::forward_ad:97-98` (dual-number forward
subtraction primal+tangent), and `autograd::grad_penalty:117` builds the
`norm - 1` term used by the WGAN-GP penalty.

The signature `pub fn sub<T: Float>(a, b)` is missing the
`alpha` kwarg. PyTorch's `torch.sub(input, other, *, alpha=1)` is at
BinaryOps.cpp:434 — `TORCH_IMPL_FUNC(sub_out)` literally calls
`add_stub(device_type(), *this, -alpha)`. No `sub_scaled` /
`sub_scaled_out` / `sub_out` entry points exist. Blocker #1192 tracks the
fix.

### REQ-3 `mul` (lines 928-1097)

`MulBackward` (`arithmetic.rs:937-988`) saves `a`/`b`; backward has two
branches — when `grad_output.requires_grad()` or has a `grad_fn`
(higher-order grad / `create_graph=True`), it uses differentiable `mul`
calls so the backward pass enters the graph; otherwise it routes through
`no_grad(|| mul(...))`. Forward `mul`/`mul_inner`
(`arithmetic.rs:991-1097`) is structurally identical to `add`/`add_inner`
with `mul_*` / `broadcast_mul_*` kernels and the CPU `fast_mul`
fallthrough. **Non-test consumer**: `methods.rs:22-24` `Tensor::mul_t`;
also `einsum.rs:818,824,840,848` (the batch-matmul broadcast paths),
`autograd::higher_order:471`, `autograd::grad_penalty:122,295`,
`grad_fns::transcendental:73,278,355`.

### REQ-4 `div` (lines 1103-1259)

`DivBackward` (`arithmetic.rs:1112-1145`) — `da = grad / b`,
`db = -grad * a / (b*b)`. Forward `div`/`div_inner`
(`arithmetic.rs:1151-1259`) — `dispatch_floating_dtype!` over
`div_{f32,f64,bf16_bf16,f16}` and `broadcast_div_{...}`, CPU `fast_div`
fallthrough. IEEE-754 div-by-zero behavior is delegated to the underlying
kernel / `fast_div`. **Non-test consumer**: `methods.rs:26-28`
`Tensor::div_t`; also `autograd::forward_ad:114,120` (dual-number division
quotient rule), `nn::loss.rs:136` (mean reduction), `transcendental.rs:175`
(log-backward `grad / x`).

### REQ-5 `neg` (lines 1265-1334)

`NegBackward` (`arithmetic.rs:1273-1290`) returns `-grad` under `no_grad`.
Forward `neg`/`neg_inner` (`arithmetic.rs:1293-1334`) routes through
`neg_{f32,f64,bf16_bf16,f16}` on CUDA and `unary_map(a, |x| -x)` on CPU.
**Non-test consumer**: `methods.rs:30-32` `Tensor::neg_t`; also
`vmap.rs:1017`, `autograd::forward_ad:126-127`, `transcendental.rs:354`,
and recursive use inside `SubBackward` / `DivBackward` here.

### REQ-8 `pow` (lines 1340-1461)

`PowBackward` (`arithmetic.rs:1344-1420`) saves `a` + `exp: f64`; backward
computes `grad * exp * a^(exp-1)` with three branches: (a) higher-order
graph-recording path when `grad_output.requires_grad() ||
grad_output.grad_fn().is_some()`; (b) GPU path under `no_grad`; (c) CPU
direct-data-vec path. `PowBackward::scalar_args` returns `vec![self.exp]`
so the JIT tracer rehydrates `IrOpKind::Pow { exponent }` (#887). Forward
`pow`/`pow_inner` (`arithmetic.rs:1423-1461`) uses
`backend.pow_{f32,f64}` on f32/f64 CUDA, and `scalar_map(a, exp_t, |x, e|
x.powf(e))` on CPU. bf16/f16 fall through to the CPU path. **Non-test
consumer**: `methods.rs:34-36` `Tensor::pow_t`; also
`autograd::grad_penalty:111,118`, `autograd::graph.rs:876-877`.

NB: pow's parity REQ is NOT-STARTED because the parity-sweep runner has no
`"pow"` arm — `dispatch_floating(op, ...)` at
`tools/parity-sweep/runner/src/main.rs:214-231` only matches `add`, `sub`,
`mul`, `div`, `neg`, `abs`, `sqrt`. Until the runner adds a `pow` arm with
the scalar-exponent kwarg destructuring, `parity-sweep sweep --op pow
--seeds 8` returns `0/72 passed (72 skipped, 0 failed)`. Blocker #1193.

### REQ-7 `sqrt` (lines 1467-1565)

`SqrtBackward` (`arithmetic.rs:1475-1522`) — `grad / (2 * sqrt(a))`. GPU
path constructs a CPU `[2.0; numel]` tensor, uploads via `.to(device)`,
multiplies with `sqrt(a)`, divides `grad` by the product. CPU path is a
direct zip-map. Forward `sqrt`/`sqrt_inner` (`arithmetic.rs:1525-1565`)
uses `backend.sqrt_{f32,f64,f16}` on CUDA (f16 added in crosslink #1185
Phase 1), `unary_map(a, |x| x.sqrt())` on CPU. **Non-test consumer**:
`methods.rs:38-40` `Tensor::sqrt_t`; also `autograd::grad_penalty:113`
(gradient-norm computation).

### REQ-6 `abs` (lines 1569-1682)

`AbsBackward` (`arithmetic.rs:1579-1643`) — `grad * sign(a)` with explicit
`sign(0) = 0` convention and a dedicated GPU `abs_backward_{f32,f64}`
kernel path when both `grad_output` and `a` live on CUDA. Forward
`abs`/`abs_inner` (`arithmetic.rs:1646-1682`) uses
`backend.abs_{f32,f64}` on f32/f64 CUDA, `unary_map(a, |x| x.abs())`
elsewhere (bf16/f16 fall through to CPU). **Non-test consumer**:
`methods.rs:42-44` `Tensor::abs_t`.

### NOT-STARTED ops (REQ-9..REQ-16)

`rsub` / `rsqrt` / `reciprocal` / `floor_divide` / `remainder` / `fmod` /
`addcmul` / `addcdiv` have no `pub fn` declaration in `arithmetic.rs` and no
backward struct. Each is gated by an open prereq blocker (#1194–#1201)
that needs the impl + a non-test consumer (the natural target is
`methods.rs` per the existing pattern). Two of these (`floor_divide`,
`remainder`) have integer-tensor siblings in
`ferrotorch-core/src/int_tensor.rs:588,599` but those operate on
`IntTensor<I>`, not the float `Tensor<T: Float>` this module owns;
satisfying REQ-12/REQ-13 requires a separate float implementation. Two
others (`addcmul`, `addcdiv`) have an upstream-route gap as well: their
true upstream contract lives in `aten/src/ATen/native/PointwiseOps.cpp`
(not BinaryOps.cpp or UnaryOps.cpp as the route declares).

## Parity contract

The route's `parity_ops` list declares 16 ops. The current state per op:

| op | upstream entry | parity-sweep status | edge-case contract |
|---|---|---|---|
| `add` | BinaryOps.cpp:434 (sub_out via add_stub w/ -alpha) + `_torch_docs.py:358` signature `add(input, other, *, alpha=1, out=None)` | verified (88/88 at seeds=8) | NaN propagates; +/-Inf preserved; denormals preserved (no FTZ); empty shapes preserved; scalar `[]` broadcasts; 0-stride expand broadcasts; alpha edges (0, -0.0, NaN, ±huge); type promotion not yet validated (op_db emits f32-only); non-contig views materialized via `strided_copy_*`; in-place via `add_scaled_`; `out=` via `add_scaled_out` |
| `sub` | BinaryOps.cpp:434 `TORCH_IMPL_FUNC(sub_out)` (literally `add_stub(..., -alpha)`) + `_torch_docs.py:10851` signature `sub(input, other, *, alpha=1, out=None)` | DIVERGES (72/88 at seeds=8 — 16 failures on `alpha != 1.0` samples i=9, i=10 at shape `[5,5]`) | inherits add's contract but ferrotorch's `sub` has no `alpha` parameter so any alpha != 1 silently produces `a - b` |
| `mul` | BinaryOps.cpp:441 `TORCH_IMPL_FUNC(mul_out)` + `_torch_docs.py:7754` `mul(input, other, *, out=None)` | verified (72/72 at seeds=8) | NaN, ±Inf, denormals, empty, scalar, 0-stride, type promotion (not yet covered); no `out=` variant in ferrotorch |
| `div` | BinaryOps.cpp:447 `TORCH_IMPL_FUNC(div_out)` + `_torch_docs.py:3926` `div(input, other, *, rounding_mode=None, out=None)` | verified (72/72 at seeds=8) | IEEE-754 div-by-zero produces ±Inf / NaN; no `rounding_mode` kwarg in ferrotorch (the `floor` / `trunc` modes are equivalent to `floor_divide` / `trunc_divide`, themselves NOT-STARTED here) |
| `neg` | UnaryOps.cpp:344 `CREATE_UNARY_TORCH_IMPL_FUNC(neg_out, neg_stub)` + `_torch_docs.py` signature `torch.neg(input, *, out=None)` | verified (8/8 at seeds=8) | sign bit flipped; NaN preserved (payload may not be); ±Inf -> ∓Inf; ±0.0 -> ∓0.0 |
| `abs` | UnaryOps.cpp:546 `Tensor abs(...)` (via `unary_op_impl_with_complex_to_float`) + `_torch_docs.py` `torch.abs(input, *, out=None)` | verified (8/8 at seeds=8) | NaN preserved; ±Inf -> +Inf; ±0.0 -> +0.0; complex-input promotion not supported (ferrotorch routes only `T: Float`); backward sign(0)=0 |
| `sqrt` | UnaryOps.cpp:359 `CREATE_UNARY_TORCH_IMPL_FUNC(sqrt_out, sqrt_stub)` | verified (8/8 at seeds=8) | sqrt(negative)=NaN; sqrt(-0.0)=-0.0; sqrt(+Inf)=+Inf; backward grad/(2*sqrt(a)) is ±Inf at a=0 (matches torch) |
| `pow` | `Pow.cpp:51` `TORCH_IMPL_FUNC(pow_Tensor_Scalar_out)` (NOT BinaryOps/UnaryOps — route incomplete) | RUNNER NO-DISPATCH (0/72 passed, 72 skipped) | scalar exponent; pow(NaN, x)=NaN unless x=0 -> 1; pow(x, 0)=1 (including pow(0,0)=1, pow(NaN,0)=1); pow(±0, neg_exp)=±Inf — none of these are currently verified against torch because the runner has no `pow` arm |
| `rsub` | BinaryOps.cpp:1169 `Tensor rsub(const Tensor& self, const Tensor& other, const Scalar& alpha)` | NOT IMPLEMENTED | computes `other - alpha*input`; equivalent to `torch.add(input, other, alpha=-alpha)` from input's perspective with the operands swapped |
| `rsqrt` | UnaryOps.cpp:346 `CREATE_UNARY_TORCH_IMPL_FUNC(rsqrt_out, rsqrt_stub)` | NOT IMPLEMENTED | `1/sqrt(input)`; rsqrt(0)=+Inf; rsqrt(negative)=NaN; rsqrt(+Inf)=+0.0 |
| `reciprocal` | UnaryOps.cpp:345 `CREATE_UNARY_TORCH_IMPL_FUNC(reciprocal_out, reciprocal_stub)` | NOT IMPLEMENTED | `1/input`; reciprocal(±0.0)=±Inf; reciprocal(±Inf)=±0.0; reciprocal(NaN)=NaN |
| `floor_divide` | BinaryOps.cpp:979 `Tensor floor_divide(...)` | NOT IMPLEMENTED for Float | floors toward -Inf (NOT C truncation); sign-of-divisor remainder semantics; integer overload exists at `int_tensor.rs:588` |
| `remainder` | BinaryOps.cpp:1184 `Tensor remainder(const Tensor& self, const Scalar& other)` | NOT IMPLEMENTED for Float | sign of divisor (Python `%` / NumPy semantics); `a - floor_divide(a,b)*b`; div-by-zero returns 0 for ints, NaN/Inf for floats |
| `fmod` | BinaryOps.cpp:1540 `Tensor fmod(const Tensor& self, const Scalar& other)` | NOT IMPLEMENTED | sign of dividend (C99 `fmod` semantics); distinct from `remainder` — `fmod(-7, 3) = -1` but `remainder(-7, 3) = 2` |
| `addcmul` | PointwiseOps.cpp:57 `TORCH_IMPL_FUNC(addcmul_out)` (route's upstream list missing this file) | NOT IMPLEMENTED | `out = input + value * tensor1 * tensor2`; fused single-pass kernel; integer-dtype int-division deprecated upstream |
| `addcdiv` | PointwiseOps.cpp:66 `TORCH_IMPL_FUNC(addcdiv_out)` (route's upstream list missing this file) | NOT IMPLEMENTED | `out = input + value * tensor1 / tensor2`; integer-dtype version errors out per upstream deprecation |

The parity-sweep audit JSON only carries an entry for `add` so far
(`tools/parity-sweep/parity_audit.json:5-72`). The other 6 dispatchable
ops (`sub`/`mul`/`div`/`neg`/`abs`/`sqrt`) sweep on demand from the runner
but have not yet been recorrected through the discriminator pass.

## Verification

### Unit tests (in-file `#[cfg(test)] mod tests` at `arithmetic.rs:1688-2076`)

Forward correctness:
- `test_add_forward` (1720), `test_sub_forward` (1728), `test_mul_forward`
  (1736), `test_div_forward` (1744), `test_neg_forward` (1752),
  `test_pow_forward` (1759), `test_sqrt_forward` (1769), `test_abs_forward`
  (1779).

Backward (scalar):
- `test_add_backward` (1790), `test_sub_backward` (1802),
  `test_mul_backward` (1814), `test_div_backward` (1826),
  `test_div_backward_tensor_by_scalar` (1838 — reproducer for GitHub #7),
  `test_neg_backward` (1864), `test_pow_backward` (1874),
  `test_sqrt_backward` (1884), `test_abs_backward_positive` (1895),
  `test_abs_backward_negative` (1905).

No-grad / partial-grad:
- `test_add_no_grad_fn_when_inputs_detached` (1919),
  `test_mul_partial_requires_grad` (1927),
  `test_no_grad_context_skips_backward` (1941).

Chain-rule:
- `test_chain_mul_add` (1956), `test_chain_div_sub` (1971),
  `test_chain_sqrt_pow` (1986), `test_neg_double` (2001).

Vector backward:
- `test_mul_vector_backward` (2016).

### Parity-sweep commands (verbatim — orchestrator re-runs)

```bash
./target/release/parity-sweep sweep --op add  --seeds 8   # → 88/88 passed (0 skipped, 0 failed)
./target/release/parity-sweep sweep --op sub  --seeds 8   # → 72/88 passed (0 skipped, 16 failed) — BLOCKER #1192
./target/release/parity-sweep sweep --op mul  --seeds 8   # → 72/72 passed (0 skipped, 0 failed)
./target/release/parity-sweep sweep --op div  --seeds 8   # → 72/72 passed (0 skipped, 0 failed)
./target/release/parity-sweep sweep --op neg  --seeds 8   # → 8/8 passed (0 skipped, 0 failed)
./target/release/parity-sweep sweep --op abs  --seeds 8   # → 8/8 passed (0 skipped, 0 failed)
./target/release/parity-sweep sweep --op sqrt --seeds 8   # → 8/8 passed (0 skipped, 0 failed)
./target/release/parity-sweep sweep --op pow  --seeds 8   # → 0/72 passed (72 skipped, 0 failed) — BLOCKER #1193 (no runner arm)
# rsub, rsqrt, reciprocal, floor_divide, remainder, fmod, addcmul, addcdiv
# → each no_dispatch, BLOCKERS #1194..#1201
```

The integer grep-count for `passed (0 skipped, 0 failed)` is **>= 1** for
add/mul/div/neg/abs/sqrt and **== 0** for sub/pow/rsub/rsqrt/reciprocal/floor_divide/remainder/fmod/addcmul/addcdiv.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (add, add_scaled, add_out, add_scaled_out) | SHIPPED | impl: `add` at `ferrotorch-core/src/grad_fns/arithmetic.rs:351`, `add_scaled` at `:716`, `add_out` at `:619`, `add_scaled_out` at `:650` mirror `aten/src/ATen/native/BinaryOps.cpp:1176` (`Tensor add`) and the `add_stub` dispatch at `:377` + `torch/_torch_docs.py:358` signature `add(input, other, *, alpha=1, out=None)`. Non-test consumers: `ferrotorch-core/src/methods.rs:15` (`Tensor::add_t`) calls `arithmetic::add`; `ferrotorch-core/src/inplace.rs:213` (`Tensor::add_scaled_`) calls `arithmetic::add_scaled`. Parity-sweep `add` status `verified` per `tools/parity-sweep/parity_audit.json:5-72` (88/88 passed, 0 failed at seeds=8). |
| REQ-2 (sub with alpha) | NOT-STARTED | open prereq blocker #1192. `arithmetic.rs:816 pub fn sub<T>(a, b)` has no `alpha` parameter; PyTorch `sub` signature at `BinaryOps.cpp:434` and `_torch_docs.py:10851` requires `*, alpha=1, out=None`. parity-sweep `[sub] 72/88 passed (0 skipped, 16 failed)` — failures all hit `alpha != 1` op_db samples. |
| REQ-3 (mul) | SHIPPED | impl: `mul` at `ferrotorch-core/src/grad_fns/arithmetic.rs:991` mirrors `aten/src/ATen/native/BinaryOps.cpp:441 TORCH_IMPL_FUNC(mul_out)` + `mul_stub` at `:378`. Non-test consumer: `ferrotorch-core/src/methods.rs:23` (`Tensor::mul_t`); also `ferrotorch-core/src/einsum.rs:818`, `ferrotorch-core/src/autograd/grad_penalty.rs:122`. Parity-sweep `[mul] 72/72 passed (0 skipped, 0 failed)` at seeds=8. |
| REQ-4 (div) | SHIPPED | impl: `div` at `ferrotorch-core/src/grad_fns/arithmetic.rs:1151` mirrors `aten/src/ATen/native/BinaryOps.cpp:447 TORCH_IMPL_FUNC(div_out)` via `div_true_stub`. Non-test consumer: `ferrotorch-core/src/methods.rs:27` (`Tensor::div_t`); also `ferrotorch-core/src/autograd/forward_ad.rs:114`, `ferrotorch-nn/src/loss.rs:136`. Parity-sweep `[div] 72/72 passed (0 skipped, 0 failed)` at seeds=8. NB: the `rounding_mode` kwarg (`floor`/`trunc`) is not implemented but those modes correspond to `floor_divide` / `trunc_divide`, themselves NOT-STARTED elsewhere in this table. |
| REQ-5 (neg) | SHIPPED | impl: `neg` at `ferrotorch-core/src/grad_fns/arithmetic.rs:1293` mirrors `aten/src/ATen/native/UnaryOps.cpp:344 CREATE_UNARY_TORCH_IMPL_FUNC(neg_out, neg_stub)`. Non-test consumer: `ferrotorch-core/src/methods.rs:31` (`Tensor::neg_t`); also `ferrotorch-core/src/autograd/forward_ad.rs:126`. Parity-sweep `[neg] 8/8 passed (0 skipped, 0 failed)` at seeds=8. |
| REQ-6 (abs) | SHIPPED | impl: `abs` at `ferrotorch-core/src/grad_fns/arithmetic.rs:1646` mirrors `aten/src/ATen/native/UnaryOps.cpp:546 Tensor abs(const Tensor& self)`. Non-test consumer: `ferrotorch-core/src/methods.rs:43` (`Tensor::abs_t`). Parity-sweep `[abs] 8/8 passed (0 skipped, 0 failed)` at seeds=8. |
| REQ-7 (sqrt) | SHIPPED | impl: `sqrt` at `ferrotorch-core/src/grad_fns/arithmetic.rs:1525` mirrors `aten/src/ATen/native/UnaryOps.cpp:359 CREATE_UNARY_TORCH_IMPL_FUNC(sqrt_out, sqrt_stub)`. Non-test consumer: `ferrotorch-core/src/methods.rs:39` (`Tensor::sqrt_t`); also `ferrotorch-core/src/autograd/grad_penalty.rs:113`. Parity-sweep `[sqrt] 8/8 passed (0 skipped, 0 failed)` at seeds=8. |
| REQ-8 (pow) | NOT-STARTED | open prereq blocker #1193. Impl exists (`ferrotorch-core/src/grad_fns/arithmetic.rs:1423 pub fn pow`, mirrors `aten/src/ATen/native/Pow.cpp:51 TORCH_IMPL_FUNC(pow_Tensor_Scalar_out)`) AND has a non-test consumer (`methods.rs:35 Tensor::pow_t`, also `autograd/grad_penalty.rs:111,118` and `autograd/graph.rs:876`) — but the gauntlet's parity-sweep arm is missing: `tools/parity-sweep/runner/src/main.rs:214-231` has no `"pow"` match, so `[pow] 0/72 passed (72 skipped, 0 failed)`. The work item is the runner dispatch, not the op itself. |
| REQ-9 (rsub) | NOT-STARTED | open prereq blocker #1194. No `pub fn rsub` in `ferrotorch-core/src/grad_fns/arithmetic.rs`; upstream contract at `aten/src/ATen/native/BinaryOps.cpp:1169 Tensor rsub(const Tensor& self, const Tensor& other, const Scalar& alpha)` and `torch/overrides.py:1116 torch.rsub: lambda input, other, alpha=1: -1`. |
| REQ-10 (rsqrt) | NOT-STARTED | open prereq blocker #1195. No `pub fn rsqrt` in `arithmetic.rs`; upstream contract at `aten/src/ATen/native/UnaryOps.cpp:346 CREATE_UNARY_TORCH_IMPL_FUNC(rsqrt_out, rsqrt_stub)` and `torch/overrides.py:1115 torch.rsqrt: lambda input, out=None: -1`. |
| REQ-11 (reciprocal) | NOT-STARTED | open prereq blocker #1196. No `pub fn reciprocal` in `arithmetic.rs`; upstream contract at `aten/src/ATen/native/UnaryOps.cpp:345 CREATE_UNARY_TORCH_IMPL_FUNC(reciprocal_out, reciprocal_stub)` and `torch/overrides.py:1098 torch.reciprocal: lambda input, out=None: -1`. |
| REQ-12 (floor_divide float) | NOT-STARTED | open prereq blocker #1197. No `pub fn floor_divide` for `Tensor<T: Float>` in `arithmetic.rs`; upstream contract at `aten/src/ATen/native/BinaryOps.cpp:979 Tensor floor_divide(const Tensor& self, const Tensor& other)` and `torch/overrides.py:664 torch.floor_divide: lambda input, other: -1`. An integer-only `IntTensor::floor_div` sibling exists at `ferrotorch-core/src/int_tensor.rs:588` but does not satisfy the float-typed REQ. |
| REQ-13 (remainder float) | NOT-STARTED | open prereq blocker #1198. No `pub fn remainder` for `Tensor<T: Float>` in `arithmetic.rs`; upstream contract at `aten/src/ATen/native/BinaryOps.cpp:1184 Tensor remainder(const Tensor& self, const Scalar& other)` and `torch/overrides.py:1100 torch.remainder: lambda input, other, out=None: -1`. Integer-only sibling at `ferrotorch-core/src/int_tensor.rs:599` does not satisfy this. |
| REQ-14 (fmod) | NOT-STARTED | open prereq blocker #1199. No `pub fn fmod` in `arithmetic.rs`; upstream contract at `aten/src/ATen/native/BinaryOps.cpp:1540 Tensor fmod(const Tensor& self, const Scalar& other)` and `torch/overrides.py:666 torch.fmod: lambda input, other, out=None: -1`. |
| REQ-15 (addcmul) | NOT-STARTED | open prereq blocker #1200. No `pub fn addcmul` in `arithmetic.rs`; upstream contract at `aten/src/ATen/native/PointwiseOps.cpp:57 TORCH_IMPL_FUNC(addcmul_out)` and `torch/_torch_docs.py:510 addcmul(input, tensor1, tensor2, *, value=1, out=None)`. NB: PointwiseOps.cpp is not in the route's `upstream` list; the route's upstream-paths declaration is incomplete for this op. |
| REQ-16 (addcdiv) | NOT-STARTED | open prereq blocker #1201. No `pub fn addcdiv` in `arithmetic.rs`; upstream contract at `aten/src/ATen/native/PointwiseOps.cpp:66 TORCH_IMPL_FUNC(addcdiv_out)` and `torch/_torch_docs.py:461 addcdiv(input, tensor1, tensor2, *, value=1, out=None)`. Same route-upstream incompleteness as REQ-15. |
