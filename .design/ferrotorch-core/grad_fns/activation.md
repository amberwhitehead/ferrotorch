# Activation grad_fns

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/native/Activation.cpp
  - torch/nn/functional.py
  - torch/_torch_docs.py
-->

## Summary

`ferrotorch-core/src/grad_fns/activation.rs` is the autograd-tracking wrapper
layer for the activation-function family declared in
`aten/src/ATen/native/Activation.cpp` and exposed at the Python user surface
via `torch/nn/functional.py`. Eighteen of the twenty-two ops in the route's
`parity_ops` list are implemented here as `pub fn` + `*Backward`
`GradFn` struct pairs with f32/f64 GPU fast paths (where the cuDNN/cuBLAS-side
PTX kernels exist) and CPU fallbacks. The remaining four (`threshold`,
`rrelu`, `celu`, `softmin`) are NOT-STARTED in this file — `softmin` is
implemented as `softmax(-x)` in `ferrotorch-nn/src/functional.rs` (not a
fused VJP), and `threshold` / `rrelu` / `celu` live as Module wrappers
in `ferrotorch-nn/src/activation.rs` without any dedicated `*Backward`
`GradFn` here. The file is 3539 LOC, of which ~2378 LOC are production
code and ~1161 LOC are `#[cfg(test)]`.

## Requirements

- REQ-1: `relu(x) = max(0, x)` — forward `clamp_min(x, 0)` + `ReluBackward`
  VJP `grad * (x > 0)`. Mirrors `Tensor relu(const Tensor& self)` at
  `aten/src/ATen/native/Activation.cpp:514-517` (`return at::clamp_min(self, 0)`)
  and `torch.nn.functional.relu` at `torch/nn/functional.py:1718`. GPU path
  for f32/f64 via `backend.relu_f32` / `backend.relu_f64` PTX kernels.

- REQ-2: `sigmoid(x) = 1 / (1 + exp(-x))` — forward + `SigmoidBackward`
  VJP `grad * s * (1 - s)` where `s = sigmoid(x)` (the output). Mirrors
  `torch._C._nn.sigmoid` exposed via `torch/nn/functional.py:2302`. GPU
  path for f32/f64/bf16/f16 via `dispatch_floating_dtype!` macro.

- REQ-3: `tanh(x)` — forward + `TanhBackward` VJP `grad * (1 - t^2)`
  where `t = tanh(x)` (the output). Mirrors `torch/nn/functional.py:2291`.
  GPU path for f32/f64/bf16/f16 via `dispatch_floating_dtype!`.

- REQ-4: `gelu(x, approximate)` — forward + `GeluBackward` with three
  approximation modes (`None` / `Tanh` / `Sigmoid`). Mirrors PyTorch's
  `nn.GELU(approximate)` per `gelu = _add_docstr(torch._C._nn.gelu, ...)`
  at `torch/nn/functional.py:2012-2015`. Exact mode `x * 0.5 * (1 + erf(x / sqrt(2)))`
  matches `TORCH_IMPL_FUNC(gelu_out_cpu)` at
  `aten/src/ATen/native/Activation.cpp:392-415`. ferrotorch adds a
  `Sigmoid` mode (`x * sigmoid(1.702 * x)`) as a fast approximation not
  present upstream — explicit ferrotorch extension.

- REQ-5: `silu(x) = x * sigmoid(x)` (Swish) — forward + `SiluBackward`
  VJP `grad * (s + x * s * (1 - s))`. Mirrors
  `TORCH_IMPL_FUNC(silu_out)` at `aten/src/ATen/native/Activation.cpp:290`
  and `torch.nn.functional.silu` at `torch/nn/functional.py:2381`. GPU
  path for f32/f64.

- REQ-6: `softmax(x)` along the last axis — forward + `SoftmaxBackward`
  VJP `softmax * (grad - sum(grad * softmax, axis=-1, keepdim))`. Mirrors
  `torch.nn.functional.softmax` at `torch/nn/functional.py:2128` (the
  `dim=-1` default last-axis case is what's wired). Stores the softmax
  output (not input) for backward efficiency. GPU path for f32/f64/bf16/f16
  with bf16 promoting the row-max + sum_exp accumulator to f32 on CPU
  (numerical stability for narrow dynamic range).

- REQ-7: `log_softmax(x)` along the last axis — forward + `LogSoftmaxBackward`
  VJP `grad - softmax * sum(grad, axis=-1, keepdim)`. Mirrors
  `torch.nn.functional.log_softmax` at `torch/nn/functional.py:2245`. Stores
  `exp(log_softmax)` (= softmax) for backward. GPU path for f32/f64.

- REQ-8: `softplus(x; beta, threshold) = log(1 + exp(beta * x)) / beta`
  with stability branch `softplus(x) = x` when `beta * x > threshold`.
  Forward + `SoftplusBackward` VJP `grad * sigmoid(beta * x)` (gradient
  through the threshold branch is 1 — `grad` passes through). Mirrors
  `TORCH_IMPL_FUNC(softplus_out)` at
  `aten/src/ATen/native/Activation.cpp:308` and `torch.nn.functional.softplus`
  declared at `torch/nn/functional.py:2067-2070`. GPU backward path
  builds `sigmoid(beta * x)` from primitives (`scale_f64` + `sigmoid_f64`
  + `mul_f64`).

- REQ-9: `elu(x; alpha) = x` if `x > 0` else `alpha * (exp(x) - 1)` —
  forward + `EluBackward` VJP `grad * 1` (x > 0) or `grad * alpha * exp(x)`
  (x <= 0). Mirrors `TORCH_IMPL_FUNC(elu_out)` at
  `aten/src/ATen/native/Activation.cpp:272-277` and
  `torch.nn.functional.elu` at `torch/nn/functional.py:1821`. GPU path
  for f32/f64.

- REQ-10: `mish(x) = x * tanh(softplus(x))` — forward + `MishBackward`
  VJP `tanh(sp) + x * sigmoid(x) * (1 - tanh(sp)^2)`. Mirrors
  `TORCH_IMPL_FUNC(mish_out)` at
  `aten/src/ATen/native/Activation.cpp:302` and
  `torch.nn.functional.mish` at `torch/nn/functional.py:2406`. GPU path
  for f32/f64.

- REQ-11: `leaky_relu(x; negative_slope)` — forward + `LeakyReluBackward`
  VJP `grad * 1` (x > 0) or `grad * negative_slope` (x <= 0). Mirrors
  `TORCH_IMPL_FUNC(leaky_relu_out)` at
  `aten/src/ATen/native/Activation.cpp:324` and
  `torch.nn.functional.leaky_relu` at `torch/nn/functional.py:1907`.
  CPU-only at present; CUDA backward builds the per-element mask via
  `unary_map` + `crate::grad_fns::arithmetic::mul`.

- REQ-12: `hardtanh(x; min, max) = clamp(x, min, max)` and `relu6(x) = hardtanh(x, 0, 6)`
  — forward + `HardtanhBackward` VJP `grad` if `min < x < max` else 0.
  Mirrors `Tensor hardtanh(...)` at
  `aten/src/ATen/native/Activation.cpp:436-468`,
  `Tensor relu6(...)` at `aten/src/ATen/native/Activation.cpp:528-530` (which calls `hardtanh(self, 0, 6)`),
  and `torch.nn.functional.hardtanh` / `torch.nn.functional.relu6` at
  `torch/nn/functional.py:1770` / `torch/nn/functional.py:1805`. CPU-only.

- REQ-13: `hardsigmoid(x) = clamp((x + 3) / 6, 0, 1)` — forward +
  `HardsigmoidBackward` VJP `grad * (1/6)` when `-3 < x < 3` else 0.
  Mirrors `TORCH_IMPL_FUNC(hardsigmoid_out)` at
  `aten/src/ATen/native/Activation.cpp:340` and
  `torch.nn.functional.hardsigmoid` at `torch/nn/functional.py:2312`.
  CPU-only.

- REQ-14: `hardswish(x) = x * hardsigmoid(x)` — forward +
  `HardswishBackward` VJP `grad * (1 if x >= 3, 0 if x <= -3, else (2x + 3)/6)`.
  Mirrors `Tensor hardswish(const Tensor& self)` at
  `aten/src/ATen/native/Activation.cpp:477-505` (delegating to
  `hardswish_stub`), and `torch.nn.functional.hardswish` at
  `torch/nn/functional.py:2426`. CPU-only.

- REQ-15: `selu(x) = scale * elu(x, alpha)` with canonical Klambauer et al. 2017
  constants `alpha ≈ 1.6732632`, `scale ≈ 1.0507009873554805`. Forward
  + `SeluBackward` VJP `grad * scale * (1 if x > 0 else alpha * exp(x))`.
  Mirrors `Tensor selu(const Tensor& self)` at
  `aten/src/ATen/native/Activation.cpp:524-526` (returns
  `at::elu(self, SELU_ALPHA, SELU_SCALE)`) and
  `torch.nn.functional.selu` at `torch/nn/functional.py:1845`. CPU-only.

- REQ-16: `softsign(x) = x / (1 + |x|)` — forward + `SoftsignBackward`
  VJP `grad / (1 + |x|)^2`. Mirrors `torch.nn.functional.softsign` at
  `torch/nn/functional.py:2055`. PyTorch C++ implements this as a
  composite (no dedicated dispatch stub in Activation.cpp); ferrotorch
  fuses both forward and backward. CPU-only.

- REQ-17: `prelu(x, alpha)` — forward + `PReluBackward` fused VJP
  (single-pass over `x`, single backward node). Returns two gradients:
  `dL/dx[i] = grad[i] * (x[i] >= 0 ? 1 : alpha_v)` and
  `dL/dalpha = sum_i grad[i] * (x[i] < 0 ? x[i] : 0)`. Mirrors
  `Tensor prelu(const Tensor& self, const Tensor& weight_)` at
  `aten/src/ATen/native/Activation.cpp:696-726` + `_prelu_kernel` at
  `aten/src/ATen/native/Activation.cpp:729-749`, and `torch.prelu` exposed via
  `torch/nn/functional.py:1941-1943`. ferrotorch restricts `alpha`
  to a scalar (numel == 1) tensor; full per-channel `alpha` matching
  upstream's `weight.reshape_symint(dim_w)` branch is NOT yet supported.

- REQ-18: `glu(x; dim) = a * sigmoid(b)` where `(a, b) = split(x, dim/2)`.
  Forward + `GluBackward` fused VJP. Mirrors `torch.nn.functional.glu`
  at `torch/nn/functional.py:1743`. Caches `(a, sigmoid_b)` in the
  backward struct to avoid re-computation. CPU-only.

- REQ-19: `threshold(x; threshold, value) = x` if `x > threshold` else
  `value` — forward + backward per `TORCH_IMPL_FUNC(threshold_out)` at
  `aten/src/ATen/native/Activation.cpp:688-690` and
  `TORCH_IMPL_FUNC(threshold_backward_out)` at
  `aten/src/ATen/native/Activation.cpp:692-694` (VJP =
  `grad if x > threshold else 0`). The `Threshold` Module wrapper lives
  in `ferrotorch-nn/src/activation.rs` but has NO dedicated `GradFn`
  here — the function is reachable only as the threshold-clipped output
  of a forward-only nn::Module. Open prereq blocker #1341.

- REQ-20: `rrelu(x; lower, upper, training)` — randomized leaky ReLU.
  Forward + backward per `Tensor rrelu_with_noise_cpu(...)` at
  `aten/src/ATen/native/Activation.cpp:633-654` and
  `torch.nn.functional.rrelu` at `torch/nn/functional.py:1962`.
  `RReLU` Module wrapper exists in `ferrotorch-nn/src/activation.rs`
  but no `GradFn`-bearing function exists in
  `ferrotorch-core/src/grad_fns/activation.rs`. Open prereq blocker #1341.

- REQ-21: `celu(x; alpha) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))`.
  Mirrors `Tensor celu(const Tensor& self, const Scalar& alpha)` at
  `aten/src/ATen/native/Activation.cpp:540-545` (which delegates to
  `at::elu(self, alpha, Scalar(1.0), Scalar(inv_alpha))`) and
  `torch.nn.functional.celu` at `torch/nn/functional.py:1874`. The
  `CELU` Module wrapper exists in `ferrotorch-nn/src/activation.rs` but
  no `GradFn`-bearing function exists in
  `ferrotorch-core/src/grad_fns/activation.rs`. Open prereq blocker #1341.

- REQ-22: `softmin(x) = softmax(-x)` — mirrors
  `torch.nn.functional.softmin` at `torch/nn/functional.py:2095`.
  ferrotorch-nn implements it as a composition (`neg` then `softmax`)
  in `ferrotorch-nn/src/functional.rs`, which routes through two
  separate `GradFn` nodes. A fused `softmin` + `SoftminBackward` in
  `ferrotorch-core/src/grad_fns/activation.rs` would match the file's
  one-node-per-op convention. Open prereq blocker #1341.

- REQ-23: `is_grad_enabled()` + `requires_grad()` gating — every public
  forward function in the module wraps the forward in
  `if is_grad_enabled() && input.requires_grad() { Tensor::from_operation }`
  / else returns the no-grad output. This mirrors the GradMode
  dispatcher gating PyTorch's autograd attachment at
  `aten/src/ATen/native/Activation.cpp:417` `TORCH_IMPL_FUNC(gelu_backward_out_cpu)`
  (which itself is only registered when `GradMode::is_enabled()`).

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core grad_fns::activation` passes all
  forward and backward unit tests in the `#[cfg(test)] mod tests` block
  inside `ferrotorch-core/src/grad_fns/activation.rs`.
- [x] AC-2: `relu` / `sigmoid` / `tanh` / `gelu` (all three modes) / `silu`
  / `softmax` / `log_softmax` / `softplus` / `elu` / `mish` backward
  correctness verified against `numerical_grad_scalar` at residual
  `< 1e-4` for representative non-trivial inputs.
- [x] AC-3: `leaky_relu` / `hardtanh` / `relu6` / `hardsigmoid` /
  `hardswish` / `selu` / `softsign` backward correctness verified by
  closed-form expected values at residual `< 1e-9`.
- [x] AC-4: `prelu` backward routes gradient to BOTH `input` and `alpha`
  in a single VJP (verified by `fn prelu_backward_routes_to_input_and_alpha`
  in `activation.rs`).
- [x] AC-5: `glu` forward matches `a * sigmoid(b)` and backward matches
  the decomposition (verified by `fn glu_backward_matches_decomposition`
  in `activation.rs`).
- [x] AC-6: `no_grad` context disables grad-fn attachment for every
  activation in the file (verified by the `test_*_no_grad` family and
  `fn test_activation_tail_no_grad_does_not_attach_grad_fn` in
  `activation.rs`).
- [x] AC-7: `fn prelu_rejects_nonscalar_alpha` and `fn glu_rejects_odd_dim`
  enforce argument-shape contracts via `FerrotorchError::ShapeMismatch`
  and `FerrotorchError::InvalidArgument`.
- [x] AC-8: GELU exact-mode (`approximate=GeluApproximate::None`) uses
  `crate::special::erf_scalar` (not a private A&S polynomial),
  inheriting f64 ~1 ulp precision via the SunPro fdlibm path (per the
  doc-comment in `fn erf_approx` in `activation.rs`).
- [x] AC-9: Softmax bf16 forward path promotes the row-max + sum_exp
  accumulator to f32 to preserve numerical resolution (the `is_bf16`
  branch in `fn softmax_inner` in `activation.rs`).
- [x] AC-10: GPU fast paths for f32/f64 exist for relu/sigmoid/tanh/gelu
  (all 3 modes)/silu/softmax/log_softmax/softplus/elu/mish — kernels
  delegate to `crate::gpu_dispatch::gpu_backend()` PTX shims.
- [ ] AC-11: All 22 `parity_ops` from the route table return
  `[<op>] N/N passed (0 skipped, 0 failed)` under
  `./target/release/parity-sweep sweep --op <op> --seeds 8`. **Currently
  FAILING**: 22/22 ops report `oracle: unknown op` — the runner has no
  dispatch arm for any activation. Tracked by umbrella runner-arm
  blocker #1338. Per goal.md S5 this is a TEST-INFRASTRUCTURE gap, not
  a REQ blocker.
- [ ] AC-12: `threshold` / `rrelu` / `celu` / `softmin` `GradFn`-bearing
  fused implementations land in `ferrotorch-core/src/grad_fns/activation.rs`.
  Tracked by blocker #1341.

## Architecture

### Module-level public surface

The file exposes 18 forward-entry-point public functions, 14 `*Backward`
`GradFn` struct implementations, and one `pub enum GeluApproximate {
None, Tanh, Sigmoid }` configuration enum plus a `pub fn gelu_with`
variant accepting that enum. Every forward function follows the same
scaffold:

1. **Meta-propagate fast path**: call
   `crate::meta_propagate::unary_same_shape(input)?` — if `input` is a
   meta tensor, short-circuit with the propagated output (avoids the
   full forward when only the shape is meaningful).
2. **Profiler hook**: wrap the call in
   `crate::profiler_hook::profile_op_scope` so the autograd profiler
   can attribute the op.
3. **GPU fast path**: if `input.is_cuda() && (is_f32::<T>() || is_f64::<T>())`
   AND a `gpu_backend()` is available, delegate to the appropriate PTX
   kernel via the backend's per-dtype methods. For sigmoid, tanh, and
   softmax this is broadened to bf16/f16 via
   `crate::dispatch_floating_dtype!`.
4. **CPU path**: build the output via `crate::ops::elementwise::unary_map`
   (or `fast_sigmoid` / `fast_tanh` for the SIMD-accelerated transcendentals).
5. **Grad-fn attach**: if `is_grad_enabled() && input.requires_grad()`,
   wrap the storage in `Tensor::from_operation(storage, shape, grad_fn)`
   with the appropriate `*Backward` node.

### REQ-1 relu

The `pub struct ReluBackward` in `activation.rs` saves `input:
Tensor<T>` and emits `grad_input[i] = grad[i] * (input[i] > 0 ? 1 : 0)`
— the step-function mask. GPU path dispatches `relu_backward_f32` /
`relu_backward_f64`. Non-test production consumer: `pub fn relu` (the
`Tensor::relu` method) in `ferrotorch-core/src/methods.rs` is the
canonical chainable method-style PyTorch-API surface. Additional
non-test production consumer: the forward-AD primal `relu` invocation
in `ferrotorch-core/src/autograd/forward_ad.rs`.

### REQ-2 sigmoid

The `pub struct SigmoidBackward` in `activation.rs` saves both `input`
and `output` (the sigmoid result); the VJP only needs `output`
mathematically, but `input` is saved so `GradFn::inputs(&self)`
returns the right autograd-graph reference. CPU path uses
`fast_sigmoid` (SIMD + rayon). Non-test production consumers: `pub fn
sigmoid` (the `Tensor::sigmoid` method) in
`ferrotorch-core/src/methods.rs`; the `use ferrotorch_core::grad_fns::activation::sigmoid`
in `ferrotorch-nn/src/rnn.rs` (RNN gate computation); the
`use ferrotorch_core::grad_fns::activation::sigmoid` in
`ferrotorch-nn/src/loss.rs` (BCE-with-logits); the forward-AD primal
in `ferrotorch-core/src/autograd/forward_ad.rs`.

### REQ-3 tanh

`pub struct TanhBackward` in `activation.rs` saves both `input` and
`output`. VJP uses `output` (`1 - tanh(x)^2`). CPU path uses
`fast_tanh`. Non-test production consumer: `pub fn tanh_t` (the
`Tensor::tanh_t` method) in `ferrotorch-core/src/methods.rs`;
`use ferrotorch_core::grad_fns::activation::{relu, sigmoid, tanh}`
in `ferrotorch-nn/src/rnn.rs`; the forward-AD primal in
`ferrotorch-core/src/autograd/forward_ad.rs`.

### REQ-4 gelu

`pub struct GeluBackward` in `activation.rs` saves `input` and
`approximate: GeluApproximate`. The backward dispatches on `approximate`:
- `None`: `g * (cdf + x * pdf)` with `cdf = 0.5 * (1 + erf(x / sqrt(2)))`
  and `pdf = (1 / sqrt(2π)) * exp(-x²/2)`. Uses
  `crate::special::erf_scalar` for the `erf` (f64 fdlibm rational
  approximation; f32 A&S 7.1.26).
- `Tanh`: full derivative of `0.5 * x * (1 + tanh(sqrt(2/π) * (x + c*x³)))`
  where `c = 0.044715`.
- `Sigmoid`: derivative of `x * sigmoid(1.702 * x)`.

GPU path has all three modes per `gelu_backward_f32` /
`gelu_backward_tanh_f32` / `gelu_backward_erf_f32` (and f64 variants).
Non-test production consumer: `pub fn gelu` (`Tensor::gelu`) and
`pub fn gelu_with` (`Tensor::gelu_with`) methods in
`ferrotorch-core/src/methods.rs`. Public re-export in
`ferrotorch-core/src/lib.rs`:
`pub use grad_fns::activation::{GeluApproximate, gelu, gelu_with, sigmoid, tanh}`.

### REQ-5 silu

`pub struct SiluBackward` in `activation.rs` saves `input`. VJP:
`grad * (s + x * s * (1 - s))` where `s = sigmoid(x)`. Non-test
production consumer: `pub fn silu` (`Tensor::silu` method) in
`ferrotorch-core/src/methods.rs`;
`use ferrotorch_core::grad_fns::activation::silu` in
`ferrotorch-nn/src/transformer.rs` (SwiGLU / transformer feed-forward).

### REQ-6 softmax

`pub struct SoftmaxBackward` in `activation.rs` saves `input` and
`output` (the softmax). Backward operates per-row along the last axis:
`grad_input[i, j] = s[i, j] * (grad[i, j] - sum_k(grad[i, k] * s[i, k]))`.
The CPU backward does a per-row dot product `dot = sum(g * s)`
followed by a per-element `s * (g - dot)`. GPU path dispatches
`softmax_backward_f32` / `softmax_backward_f64` passing the last-dim
size as `cols`. Non-test production consumer: `pub fn softmax`
(`Tensor::softmax` method) in `ferrotorch-core/src/methods.rs`;
`use ferrotorch_core::grad_fns::activation::softmax` in
`ferrotorch-nn/src/attention.rs`; `crate::grad_fns::activation::softmax`
invocation in `ferrotorch-core/src/flex_attention.rs` (attention-mask
post-modification softmax).

### REQ-7 log_softmax

`pub struct LogSoftmaxBackward` in `activation.rs` saves `input` and
`softmax_output` (= `exp(log_softmax)`). Backward:
`grad_input[i, j] = grad[i, j] - softmax[i, j] * sum_k(grad[i, k])`.
GPU path: when `requires_grad`, computes `softmax = exp(log_softmax)`
on GPU via `backend.exp_f32` / `exp_f64` so backward can reference it
without a CPU round-trip. Non-test production consumer: `pub fn
log_softmax` (`Tensor::log_softmax` method) in
`ferrotorch-core/src/methods.rs`.

### REQ-8 softplus

`pub struct SoftplusBackward` in `activation.rs` saves `input` and
`(beta, threshold)` as non-template `f64` scalar fields. Backward
branches on `bx > threshold` (passes `grad` through unchanged — the
threshold branch IS the identity function) vs the smooth
`grad * sigmoid(beta * x)` elsewhere. GPU backward builds the sigmoid
result from `scale_f64` → `sigmoid_f64` → `mul_f64` rather than
relying on a fused softplus_backward kernel (the fix landed in #796 —
the prior f64 lane raised `NotImplementedOnCuda`). Non-test
production consumer: `pub fn softplus` and `pub fn softplus_with` in
`ferrotorch-nn/src/functional.rs` (delegate with `beta=1,
threshold=20` default). The `Softplus` nn::Module wrapper in
`ferrotorch-nn/src/activation.rs` calls this through `act::softplus`.

### REQ-9 elu

`pub struct EluBackward` in `activation.rs` saves `input` and `alpha:
f64`. VJP: `g * 1` (x > 0) or `g * alpha * exp(x)` (x <= 0). GPU
kernels `elu_backward_f32`/`_f64`. Non-test production consumer:
`pub fn elu` and `pub fn elu_with` in `ferrotorch-nn/src/functional.rs`
(`elu` defaults `alpha=1.0`). The `ELU` nn::Module wrapper composes
through these.

### REQ-10 mish

`pub struct MishBackward` in `activation.rs` saves `input`. VJP
closed form: `dmish = tanh(sp) + x * sigmoid(x) * (1 - tanh(sp)^2)`
where `sp = softplus(x) = ln(1 + exp(x))`. GPU kernels
`mish_backward_f32`/`_f64`. Non-test production consumer: `pub fn
mish` in `ferrotorch-nn/src/functional.rs`.

### REQ-11 leaky_relu

`pub struct LeakyReluBackward` in `activation.rs` saves `input` and
`negative_slope: f64`. The CUDA backward is the file's notable
cross-device path: it computes the mask via `unary_map` (which keeps
the result on the input's device) then multiplies by `grad_output`
using `crate::grad_fns::arithmetic::mul`. The fix (#796) replaced an
earlier unconditional `self.input.data()?` / `grad_output.data()?`
that failed with `GpuTensorNotAccessible` for CUDA-resident saved
state. Non-test production consumer: `pub fn leaky_relu` in
`ferrotorch-nn/src/functional.rs` (which short-circuits to `act::relu`
for `negative_slope == 0.0` and to identity for `negative_slope == 1.0`).

### REQ-12 hardtanh / relu6

`pub struct HardtanhBackward` in `activation.rs` saves `input` +
`min_val` + `max_val: f64`. VJP: `g if min < x < max else 0`. `relu6`
is a hardtanh with `(0, 6)` bounds. CPU-only. Non-test production
consumer: `pub fn hardtanh`, `pub fn hardtanh_with`, and `pub fn
relu6` in `ferrotorch-nn/src/functional.rs`.

### REQ-13 hardsigmoid

`pub struct HardsigmoidBackward` in `activation.rs` saves `input`.
VJP: `g * (1/6)` inside `(-3, 3)` else 0. CPU-only. Non-test
production consumer: `pub fn hardsigmoid` in
`ferrotorch-nn/src/functional.rs`.

### REQ-14 hardswish

`pub struct HardswishBackward` in `activation.rs` saves `input`. VJP:
`0 if x <= -3, g if x >= 3, else g * (2x + 3)/6`. The middle region's
slope `(2x + 3)/6` is the derivative of `x * (x + 3)/6`. CPU-only.
Non-test production consumer: `pub fn hardswish` in
`ferrotorch-nn/src/functional.rs`.

### REQ-15 selu

`pub struct SeluBackward` in `activation.rs` saves `input`. VJP:
`g * scale * (1 if x > 0 else alpha * exp(x))` where constants are
the same `SELU_ALPHA` / `SELU_SCALE` used in the forward. CPU-only.
Non-test production consumer: `pub fn selu` in
`ferrotorch-nn/src/functional.rs` (which ALSO does the equivalent
composition through `act::elu(input, ALPHA)` then multiply by SCALE
— both routes exist; the route through `grad_fns::activation::selu`
is the fused single-GradFn variant).

### REQ-16 softsign

`pub struct SoftsignBackward` in `activation.rs` saves `input`. VJP:
`g / (1 + |x|)^2`. CPU-only. Non-test production consumer: `pub fn
softsign` in `ferrotorch-nn/src/functional.rs` (which also exposes a
`neg → abs → add → div` decomposition; both routes ship).

### REQ-17 prelu

`pub struct PReluBackward` in `activation.rs` saves both `input` and
`alpha` (the latter as a full `Tensor<T>`, not unwrapped to scalar —
so `GradFn::inputs(&self)` returns `vec![&self.input, &self.alpha]`
and gradient flows back to the learnable `alpha` parameter). The
backward emits two gradients: `grad_input` (per-element) and
`grad_alpha` (a length-1 tensor accumulating `sum_i grad[i] * x[i]`
over negatives). This fused single-pass replaces the prior decomposed
`(1 - alpha) * relu(x) + alpha * x` path that walked three separate
GradFn nodes. CPU-only. Non-test production consumer: `pub fn prelu`
in `ferrotorch-nn/src/functional.rs` and the `nn::PReLU` Module
wrapper.

### REQ-18 glu

`pub struct GluBackward` in `activation.rs` saves `input`, `a:
Vec<T>`, `sigmoid_b: Vec<T>`, and `dim: usize`. Caching the split
halves + sigmoid avoids re-running the forward in the backward.
Backward concatenates `[grad_a, grad_b]` back into input shape via a
manual outer/inner/k three-level loop. Forward validates `dim`
(negative → resolved, out-of-range → error, odd length → error).
CPU-only. Non-test production consumer: `pub fn glu` in
`ferrotorch-nn/src/functional.rs`.

### REQ-19 / 20 / 21 / 22 NOT-STARTED

`threshold`, `rrelu`, `celu`, `softmin` are listed in the route's
`parity_ops` but have no impl in `ferrotorch-core/src/grad_fns/activation.rs`.
They exist in alternate forms:
- `threshold`: `pub struct Threshold` in `ferrotorch-nn/src/activation.rs`
  is a forward-only Module without dedicated `GradFn`.
- `rrelu`: `pub struct RReLU` in `ferrotorch-nn/src/activation.rs`
  has default constants but no fused backward in core.
- `celu`: `pub struct CELU` in `ferrotorch-nn/src/activation.rs` is a
  Module wrapper; the function form is missing.
- `softmin`: `pub fn softmin` in `ferrotorch-nn/src/functional.rs` is
  a composition `neg → softmax` (two `GradFn` nodes), not a fused
  `SoftminBackward`. Tracked by blocker #1341.

### REQ-23 autograd gating

Each forward function checks `is_grad_enabled() && input.requires_grad()`
before attaching a `*Backward` node. Inside `no_grad` blocks
(`crate::autograd::no_grad::no_grad(...)`) the attachment is skipped
and the output carries `grad_fn().is_none()`. Verified by the
`fn test_relu_no_grad`, `fn test_sigmoid_no_grad`,
`fn test_softplus_no_grad`, `fn test_elu_no_grad`, `fn test_mish_no_grad`,
and `fn test_activation_tail_no_grad_does_not_attach_grad_fn` in
`activation.rs`.

## Parity contract

| Op | Upstream entry | Backward formula source | Edge cases mirrored |
|---|---|---|---|
| `relu` | `aten/src/ATen/native/Activation.cpp:514 Tensor relu(...)` | `clamp_min` derivative is step | Boolean inputs upstream `TORCH_CHECK` reject; ferrotorch's `Float` trait excludes bool by type-system. NaN: ferrotorch `x > 0` is `false` on NaN, so grad becomes 0 — mirrors upstream's `at::clamp_min(self, 0)` NaN propagation (NaN > 0 = false). |
| `sigmoid` | `torch/nn/functional.py:2302 def sigmoid(input)` | `s*(1-s)` | NaN propagates through `1/(1+exp(-NaN)) = NaN`. f32/f64/bf16/f16 supported on GPU. |
| `tanh` | `torch/nn/functional.py:2291 def tanh(input)` | `1 - tanh²` | Saturates to ±1 at ±20; subnormals flush. |
| `gelu` | `aten/src/ATen/native/Activation.cpp:392 TORCH_IMPL_FUNC(gelu_out_cpu)` | erf-based + tanh approx + ferrotorch sigmoid extension | None/Tanh match upstream byte-for-byte (modulo `erf_scalar` precision: ~1 ulp f64, ~1.5e-7 f32); Sigmoid is a ferrotorch-only fast mode. |
| `silu` | `aten/src/ATen/native/Activation.cpp:290 TORCH_IMPL_FUNC(silu_out)` | `s + x*s*(1-s)` | NaN propagates; `0 * sigmoid(0) = 0`. |
| `softmax` | `torch/nn/functional.py:2128 def softmax(input, dim=...)` | `s*(g - sum(g*s))` per row | bf16 forward promotes accumulator to f32 (numerical-stability fix). Last-axis only (`dim=-1`); explicit-`dim` softmax not yet routed through this file. |
| `log_softmax` | `torch/nn/functional.py:2245 def log_softmax(input, dim=...)` | `g - softmax*sum(g)` per row | Two-pass: max-subtract then log-sum-exp. Last-axis only. |
| `softplus` | `aten/src/ATen/native/Activation.cpp:308 TORCH_IMPL_FUNC(softplus_out)` | `sigmoid(βx)` (passes through above threshold) | Default `threshold=20` matches torch default. |
| `elu` | `aten/src/ATen/native/Activation.cpp:272 TORCH_IMPL_FUNC(elu_out)` | `1 (x>0) or alpha*exp(x)` | Default `alpha=1.0` matches. ferrotorch does not yet wire the `(input_scale, output_scale)` extra args that elu accepts upstream for celu. |
| `mish` | `aten/src/ATen/native/Activation.cpp:302 TORCH_IMPL_FUNC(mish_out)` | `tanh(sp) + x*sig(x)*(1-tanh(sp)²)` | Saturates to `x` for large `x`; verified by `fn test_mish_forward_positive` at `x=20`. |
| `leaky_relu` | `aten/src/ATen/native/Activation.cpp:324 TORCH_IMPL_FUNC(leaky_relu_out)` | `1 if x>0 else slope` | Default `negative_slope=0.01` (matches torch); short-circuits to relu/identity at boundary slopes. |
| `hardtanh` | `aten/src/ATen/native/Activation.cpp:436 Tensor hardtanh(...)` | `1 inside (min,max) else 0` | Default `(min=-1, max=1)`. `relu6 = hardtanh(0, 6)`. |
| `hardsigmoid` | `aten/src/ATen/native/Activation.cpp:340 TORCH_IMPL_FUNC(hardsigmoid_out)` | `1/6 inside (-3,3) else 0` | Endpoints map to 0/1 exactly. |
| `hardswish` | `aten/src/ATen/native/Activation.cpp:477 Tensor hardswish(...)` | piecewise `(2x+3)/6 inside` | Zero below -3; identity above +3. |
| `selu` | `aten/src/ATen/native/Activation.cpp:524 Tensor selu(...)` | `scale * elu_grad` | Canonical Klambauer constants; ferrotorch uses the same constants verbatim. |
| `softsign` | `torch/nn/functional.py:2055 def softsign(input)` | `1 / (1+|x|)²` | Bounded asymptotically by ±1; numerically stable everywhere. |
| `prelu` | `aten/src/ATen/native/Activation.cpp:696 Tensor prelu(...)` | dual VJP `(input, alpha)` | ferrotorch restricts `alpha.numel() == 1`; full per-channel alpha is NOT yet supported. |
| `glu` | `torch/nn/functional.py:1743 def glu(input, dim=-1)` | `(s, a*s*(1-s))` concat | Rejects odd split dim; rejects 0-D input. |
| `threshold` | `aten/src/ATen/native/Activation.cpp:688 TORCH_IMPL_FUNC(threshold_out)` | `1 if x > threshold else 0` | NOT-STARTED in this file. |
| `rrelu` | `aten/src/ATen/native/Activation.cpp:633 Tensor rrelu_with_noise_cpu(...)` | noise-mask-based VJP | NOT-STARTED in this file; requires RNG-state-aware backward. |
| `celu` | `aten/src/ATen/native/Activation.cpp:540 Tensor celu(...)` | `1 (x>0) or exp(x/alpha)` | NOT-STARTED in this file; upstream delegates to `at::elu(self, alpha, 1.0, inv_alpha)`. |
| `softmin` | `torch/nn/functional.py:2095 def softmin(input, dim=...)` | softmax(-x) VJP via chain rule | NOT-STARTED in this file; ferrotorch-nn composes `neg → softmax` (two GradFn nodes). |

Parity-sweep audit reference: all 22 ops are **MISSING** from
`tools/parity-sweep/parity_audit.json`. The runner's match arms in
`tools/parity-sweep/runner/src/main.rs` have no dispatch for any of
them — running `parity-sweep sweep --op relu` returns `oracle: unknown op: relu`.
Per goal.md S5 this is a test-infrastructure gap; tracked under
umbrella blocker #1338.

## Verification

### Existing unit tests (all passing)

Located in the `#[cfg(test)] mod tests` block at the bottom of
`ferrotorch-core/src/grad_fns/activation.rs` (1158 LOC of tests).
Key test functions in `activation.rs`:

- **ReLU**: `fn test_relu_forward_positive`, `fn test_relu_forward_negative`,
  `fn test_relu_backward_positive`, `fn test_relu_backward_negative`,
  `fn test_relu_forward_vector`.
- **Sigmoid**: `fn test_sigmoid_forward`, `fn test_sigmoid_backward`,
  `fn test_sigmoid_backward_nonzero` (vs numerical gradient).
- **Tanh**: `fn test_tanh_forward`, `fn test_tanh_backward_at_zero`,
  `fn test_tanh_backward_nonzero`.
- **GELU**: `fn test_gelu_forward_zero` (all 3 modes),
  `fn test_gelu_exact_forward_values`, `fn test_gelu_tanh_forward_values`,
  `fn test_gelu_sigmoid_forward_values`, `fn test_gelu_backward_exact`,
  `fn test_gelu_backward_tanh`, `fn test_gelu_backward_sigmoid`,
  `fn test_gelu_default_is_exact`.
- **SiLU**: `fn test_silu_forward_zero`, `fn test_silu_backward`.
- **Softmax**: `fn test_softmax_forward_1d`, `fn test_softmax_backward_1d`.
- **LogSoftmax**: `fn test_log_softmax_forward_1d`,
  `fn test_log_softmax_backward_1d`.
- **no_grad gating**: `fn test_relu_no_grad`, `fn test_sigmoid_no_grad`.
- **Softplus**: `fn test_softplus_forward_zero`,
  `fn test_softplus_forward_large` (threshold branch),
  `fn test_softplus_backward_at_zero`, `fn test_softplus_backward_positive`,
  `fn test_softplus_backward_negative`,
  `fn test_softplus_backward_custom_beta`,
  `fn test_softplus_backward_vector`, `fn test_softplus_no_grad`.
- **ELU**: `fn test_elu_forward_positive`, `fn test_elu_forward_negative`,
  `fn test_elu_backward_positive`, `fn test_elu_backward_negative`,
  `fn test_elu_backward_custom_alpha`, `fn test_elu_no_grad`.
- **Mish**: `fn test_mish_forward_zero`, `fn test_mish_forward_positive`
  (saturation at x=20), `fn test_mish_backward_at_zero`,
  `fn test_mish_backward_positive`, `fn test_mish_backward_negative`,
  `fn test_mish_no_grad`.
- **Activation tail**: `fn test_leaky_relu_forward_positive_unchanged`,
  `fn test_leaky_relu_forward_negative_scaled`,
  `fn test_leaky_relu_backward`, `fn test_hardtanh_clamps_and_grad`,
  `fn test_relu6_clamps_top_at_6`,
  `fn test_hardsigmoid_endpoints_and_slope`,
  `fn test_hardswish_zero_below_minus_three`,
  `fn test_hardswish_backward_matches_numerical`,
  `fn test_selu_zero_at_origin`, `fn test_selu_backward_at_one_is_scale`,
  `fn test_softsign_bounded_and_zero_origin`,
  `fn test_softsign_backward_closed_form`,
  `fn test_softsign_backward_at_zero_is_one`,
  `fn test_softsign_backward_matches_numerical`,
  `fn test_activation_tail_no_grad_does_not_attach_grad_fn`.
- **PReLU + GLU**: `fn prelu_forward_matches_definition`,
  `fn prelu_backward_routes_to_input_and_alpha`,
  `fn prelu_rejects_nonscalar_alpha`,
  `fn glu_forward_matches_split_sigmoid_mul`,
  `fn glu_backward_matches_decomposition`,
  `fn glu_rejects_odd_dim`, `fn glu_2d_dim1`.

### Parity-sweep status

All 22 parity_ops report `oracle: unknown op` because the runner
(`tools/parity-sweep/runner/src/main.rs`) has no dispatch arm for any
of them. Reproducer:

```
./target/release/parity-sweep sweep --op relu --seeds 8
  => FAIL: seed=0..7  oracle: unknown op: relu
```

Per goal.md S5: **missing runner arms are a TEST-INFRASTRUCTURE gap,
not a REQ blocker.** The 18 SHIPPED REQs satisfy the criteria
(impl + non-test consumer + lib tests pass + cargo clippy clean) and
the runner-arm gap is captured as ONE file-family umbrella blocker
#1338 covering all 22 ops.

### Cargo test command

```
cargo test -p ferrotorch-core grad_fns::activation
```

All forward and backward tests pass at residual `< 1e-9` (closed-form
expectations) or `< 1e-4` (numerical-gradient comparisons).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (relu) | SHIPPED | impl: `pub fn relu` + `pub struct ReluBackward` in `ferrotorch-core/src/grad_fns/activation.rs` mirroring `Tensor relu(const Tensor& self)` at `aten/src/ATen/native/Activation.cpp:514-517` (`return at::clamp_min(self, 0)`); non-test production consumer: `pub fn relu` (the `Tensor::relu` method) in `ferrotorch-core/src/methods.rs` and `pub fn relu` in `ferrotorch-nn/src/functional.rs`; forward-AD primal consumer in `ferrotorch-core/src/autograd/forward_ad.rs`. Runner arm pending per #1338. |
| REQ-2 (sigmoid) | SHIPPED | impl: `pub fn sigmoid` + `pub struct SigmoidBackward` in `ferrotorch-core/src/grad_fns/activation.rs` mirroring `torch._C._nn.sigmoid` per `torch/nn/functional.py:2302`; non-test consumer: `pub fn sigmoid` (the `Tensor::sigmoid` method) in `ferrotorch-core/src/methods.rs`, `use ferrotorch_core::grad_fns::activation::{relu, sigmoid, tanh}` in `ferrotorch-nn/src/rnn.rs` (RNN gates), `use ferrotorch_core::grad_fns::activation::sigmoid` in `ferrotorch-nn/src/loss.rs` (BCE-with-logits). Runner arm pending per #1338. |
| REQ-3 (tanh) | SHIPPED | impl: `pub fn tanh` + `pub struct TanhBackward` in `ferrotorch-core/src/grad_fns/activation.rs` mirroring `torch/nn/functional.py:2291`; non-test consumer: `pub fn tanh_t` (the `Tensor::tanh_t` method) in `ferrotorch-core/src/methods.rs`, RNN gate consumer in `ferrotorch-nn/src/rnn.rs`, forward-AD primal in `ferrotorch-core/src/autograd/forward_ad.rs`. Runner arm pending per #1338. |
| REQ-4 (gelu) | SHIPPED | impl: `pub fn gelu` + `pub fn gelu_with` + `pub struct GeluBackward` + `pub enum GeluApproximate { None, Tanh, Sigmoid }` in `ferrotorch-core/src/grad_fns/activation.rs` mirroring `TORCH_IMPL_FUNC(gelu_out_cpu)` at `aten/src/ATen/native/Activation.cpp:392-415` and `gelu = _add_docstr(torch._C._nn.gelu, ...)` at `torch/nn/functional.py:2012-2015`; non-test consumer: `pub fn gelu` + `pub fn gelu_with` (the `Tensor::gelu` and `Tensor::gelu_with` methods) in `ferrotorch-core/src/methods.rs`; public re-export `pub use grad_fns::activation::{GeluApproximate, gelu, gelu_with, sigmoid, tanh}` in `ferrotorch-core/src/lib.rs`. Runner arm pending per #1338. |
| REQ-5 (silu) | SHIPPED | impl: `pub fn silu` + `pub struct SiluBackward` in `ferrotorch-core/src/grad_fns/activation.rs` mirroring `TORCH_IMPL_FUNC(silu_out)` at `aten/src/ATen/native/Activation.cpp:290`; non-test consumer: `pub fn silu` (the `Tensor::silu` method) in `ferrotorch-core/src/methods.rs`, `use ferrotorch_core::grad_fns::activation::silu` in `ferrotorch-nn/src/transformer.rs` (SwiGLU). Runner arm pending per #1338. |
| REQ-6 (softmax) | SHIPPED | impl: `pub fn softmax` + `pub struct SoftmaxBackward` in `ferrotorch-core/src/grad_fns/activation.rs` mirroring `torch/nn/functional.py:2128`; non-test consumer: `pub fn softmax` (the `Tensor::softmax` method) in `ferrotorch-core/src/methods.rs`, `use ferrotorch_core::grad_fns::activation::softmax` in `ferrotorch-nn/src/attention.rs`, `crate::grad_fns::activation::softmax` invocation in `ferrotorch-core/src/flex_attention.rs`. Runner arm pending per #1338. |
| REQ-7 (log_softmax) | SHIPPED | impl: `pub fn log_softmax` + `pub struct LogSoftmaxBackward` in `ferrotorch-core/src/grad_fns/activation.rs` mirroring `torch/nn/functional.py:2245`; non-test consumer: `pub fn log_softmax` (the `Tensor::log_softmax` method) in `ferrotorch-core/src/methods.rs`. Runner arm pending per #1338. |
| REQ-8 (softplus) | SHIPPED | impl: `pub fn softplus` + `pub struct SoftplusBackward` in `ferrotorch-core/src/grad_fns/activation.rs` mirroring `TORCH_IMPL_FUNC(softplus_out)` at `aten/src/ATen/native/Activation.cpp:308` and `softplus = _add_docstr(torch._C._nn.softplus, ...)` at `torch/nn/functional.py:2067-2070`; non-test consumer: `pub fn softplus` + `pub fn softplus_with` in `ferrotorch-nn/src/functional.rs`. Runner arm pending per #1338. |
| REQ-9 (elu) | SHIPPED | impl: `pub fn elu` + `pub struct EluBackward` in `ferrotorch-core/src/grad_fns/activation.rs` mirroring `TORCH_IMPL_FUNC(elu_out)` at `aten/src/ATen/native/Activation.cpp:272-277` and `torch/nn/functional.py:1821`; non-test consumer: `pub fn elu` + `pub fn elu_with` in `ferrotorch-nn/src/functional.rs`. Runner arm pending per #1338. |
| REQ-10 (mish) | SHIPPED | impl: `pub fn mish` + `pub struct MishBackward` in `ferrotorch-core/src/grad_fns/activation.rs` mirroring `TORCH_IMPL_FUNC(mish_out)` at `aten/src/ATen/native/Activation.cpp:302` and `torch/nn/functional.py:2406`; non-test consumer: `pub fn mish` in `ferrotorch-nn/src/functional.rs`. Runner arm pending per #1338. |
| REQ-11 (leaky_relu) | SHIPPED | impl: `pub fn leaky_relu` + `pub struct LeakyReluBackward` in `ferrotorch-core/src/grad_fns/activation.rs` mirroring `TORCH_IMPL_FUNC(leaky_relu_out)` at `aten/src/ATen/native/Activation.cpp:324` and `torch/nn/functional.py:1907`; non-test consumer: `pub fn leaky_relu` in `ferrotorch-nn/src/functional.rs`. Runner arm pending per #1338. |
| REQ-12 (hardtanh + relu6) | SHIPPED | impl: `pub fn hardtanh` + `pub fn hardtanh_with` + `pub fn relu6` + `pub struct HardtanhBackward` in `ferrotorch-core/src/grad_fns/activation.rs` mirroring `Tensor hardtanh(...)` at `aten/src/ATen/native/Activation.cpp:436-468` + `Tensor relu6(...)` at `aten/src/ATen/native/Activation.cpp:528-530`; non-test consumer: `pub fn hardtanh` + `pub fn hardtanh_with` + `pub fn relu6` in `ferrotorch-nn/src/functional.rs`. Runner arm pending per #1338. |
| REQ-13 (hardsigmoid) | SHIPPED | impl: `pub fn hardsigmoid` + `pub struct HardsigmoidBackward` in `ferrotorch-core/src/grad_fns/activation.rs` mirroring `TORCH_IMPL_FUNC(hardsigmoid_out)` at `aten/src/ATen/native/Activation.cpp:340` and `torch/nn/functional.py:2312`; non-test consumer: `pub fn hardsigmoid` in `ferrotorch-nn/src/functional.rs`. Runner arm pending per #1338. |
| REQ-14 (hardswish) | SHIPPED | impl: `pub fn hardswish` + `pub struct HardswishBackward` in `ferrotorch-core/src/grad_fns/activation.rs` mirroring `Tensor hardswish(...)` at `aten/src/ATen/native/Activation.cpp:477-505` and `torch/nn/functional.py:2426`; non-test consumer: `pub fn hardswish` in `ferrotorch-nn/src/functional.rs`. Runner arm pending per #1338. |
| REQ-15 (selu) | SHIPPED | impl: `pub fn selu` + `pub struct SeluBackward` in `ferrotorch-core/src/grad_fns/activation.rs` mirroring `Tensor selu(const Tensor& self)` at `aten/src/ATen/native/Activation.cpp:524-526` (which delegates to `at::elu(self, SELU_ALPHA, SELU_SCALE)`) and `torch/nn/functional.py:1845`; non-test consumer: `pub fn selu` in `ferrotorch-nn/src/functional.rs`. Runner arm pending per #1338. |
| REQ-16 (softsign) | SHIPPED | impl: `pub fn softsign` + `pub struct SoftsignBackward` in `ferrotorch-core/src/grad_fns/activation.rs` mirroring `torch/nn/functional.py:2055`; non-test consumer: `pub fn softsign` in `ferrotorch-nn/src/functional.rs`. Runner arm pending per #1338. |
| REQ-17 (prelu) | SHIPPED | impl: `pub fn prelu` + `pub struct PReluBackward` (fused dual-VJP) in `ferrotorch-core/src/grad_fns/activation.rs` mirroring `Tensor prelu(const Tensor& self, const Tensor& weight_)` at `aten/src/ATen/native/Activation.cpp:696-726` and `prelu = _add_docstr(torch.prelu, ...)` at `torch/nn/functional.py:1941-1943`; non-test consumer: `pub fn prelu` in `ferrotorch-nn/src/functional.rs`. ferrotorch restricts `alpha.numel() == 1`; per-channel `alpha` (upstream `weight.reshape_symint(dim_w)` branch) is not yet supported — a known divergence, but the scalar-alpha contract IS shipped. Runner arm pending per #1338. |
| REQ-18 (glu) | SHIPPED | impl: `pub fn glu` + `pub struct GluBackward` (fused VJP caching `(a, sigmoid_b)`) in `ferrotorch-core/src/grad_fns/activation.rs` mirroring `torch/nn/functional.py:1743`; non-test consumer: `pub fn glu` in `ferrotorch-nn/src/functional.rs`. Runner arm pending per #1338. |
| REQ-19 (threshold) | NOT-STARTED | open prereq blocker #1341. No `pub fn threshold` / `ThresholdBackward` in `ferrotorch-core/src/grad_fns/activation.rs`. The `Threshold` nn::Module wrapper in `ferrotorch-nn/src/activation.rs` is forward-only and does not carry a fused `GradFn`. Upstream: `TORCH_IMPL_FUNC(threshold_out)` at `aten/src/ATen/native/Activation.cpp:688-690` and `TORCH_IMPL_FUNC(threshold_backward_out)` at `aten/src/ATen/native/Activation.cpp:692-694`. |
| REQ-20 (rrelu) | NOT-STARTED | open prereq blocker #1341. No `pub fn rrelu` / `RReluBackward` in `ferrotorch-core/src/grad_fns/activation.rs`. The `RReLU` nn::Module wrapper in `ferrotorch-nn/src/activation.rs` defines default constants but no fused autograd path; would also require an RNG-state-aware backward to mirror `Tensor rrelu_with_noise_cpu(...)` at `aten/src/ATen/native/Activation.cpp:633-654`. |
| REQ-21 (celu) | NOT-STARTED | open prereq blocker #1341. No `pub fn celu` / `CeluBackward` in `ferrotorch-core/src/grad_fns/activation.rs`. The `CELU` nn::Module wrapper in `ferrotorch-nn/src/activation.rs` is a Module; the function form is missing. Upstream: `Tensor celu(const Tensor& self, const Scalar& alpha)` at `aten/src/ATen/native/Activation.cpp:540-545` (delegates to `at::elu(self, alpha, Scalar(1.0), Scalar(inv_alpha))`) and `torch/nn/functional.py:1874`. |
| REQ-22 (softmin) | NOT-STARTED | open prereq blocker #1341. No fused `pub fn softmin` / `SoftminBackward` in `ferrotorch-core/src/grad_fns/activation.rs`. `pub fn softmin` in `ferrotorch-nn/src/functional.rs` is a composition `neg → softmax` that walks TWO `GradFn` nodes per call. A fused VJP matching this file's one-node-per-op convention has not been written. Upstream: `torch/nn/functional.py:2095 def softmin`. |
| REQ-23 (autograd gating) | SHIPPED | impl: every forward function in `ferrotorch-core/src/grad_fns/activation.rs` wraps in `if is_grad_enabled() && input.requires_grad() { Tensor::from_operation } else { Tensor::from_storage }`. Non-test production consumer: every `Tensor::<op>` method in `ferrotorch-core/src/methods.rs` reaches this gating through the public forward functions. Verified by `fn test_relu_no_grad` / `fn test_sigmoid_no_grad` / `fn test_softplus_no_grad` / `fn test_elu_no_grad` / `fn test_mish_no_grad` / `fn test_activation_tail_no_grad_does_not_attach_grad_fn` in `activation.rs` (six tests across the file). |
