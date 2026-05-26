# Transcendental grad_fns

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/native/UnaryOps.cpp
  - aten/src/ATen/native/TensorCompare.cpp
  - aten/src/ATen/native/BinaryOps.cpp
  - tools/autograd/derivatives.yaml
  - torch/_torch_docs.py
-->

## Summary

`ferrotorch-core/src/grad_fns/transcendental.rs` is the autograd-tracking
wrapper layer for the transcendental elementwise ops PyTorch declares
through the `CREATE_UNARY_TORCH_IMPL_FUNC` family in
`aten/src/ATen/native/UnaryOps.cpp` (lines 316-363) — `exp`, `log`, `sin`,
`cos`, `tan`, the inverse and hyperbolic trig family, the rounding family
(`ceil`/`floor`/`round`/`trunc`/`frac`), `sign`/`signbit`/`sinc`, the
expanded log family (`log2`/`log10`/`log1p`), the expanded exp family
(`exp2`/`expm1`) — plus the clamp family declared in
`aten/src/ATen/native/TensorCompare.cpp:831 TORCH_IMPL_FUNC(clamp_out)` and
its `clip` alias at `TensorCompare.cpp:918-930 Tensor clip(...)`, and the
binary scalar-field ops declared in `aten/src/ATen/native/BinaryOps.cpp`
(`atan2_out` at `:795`, `copysign_out` at `:865`, `hypot_out` and
`nextafter_out` macro-generated at `:548-551`). Each shipped op pairs a
`*Backward` `GradFn` struct holding the saved-for-backward operand(s)
with a `pub fn <op>` forward that branches on `is_cuda()` and routes to
either the on-device kernel via `gpu_dispatch` or the CPU
`fast_<op>` kernels in `crate::ops::elementwise` (`fast_exp`, `fast_log`,
`fast_sin`, `fast_cos`, plus `unary_map` for clamp). The file is 882 LOC,
of which ~565 LOC are production code and ~317 LOC are `#[cfg(test)]`
covering forward, backward, no-grad, chain-rule, and numerical-gradient
checks for the five shipped ops. Twenty-eight further ops named in the
route's `parity_ops` list are NOT-STARTED — they have no kernel, no
backward struct, no public function, and no method-surface entry.

## Requirements

The route's `parity_ops` list declares 33 ops. Five have shipped
forward+backward implementations with non-test production consumers
(REQ-1 through REQ-5). The remaining 28 (REQ-6 through REQ-33) are
NOT-STARTED. Each NOT-STARTED REQ has its own concrete open prereq
blocker referenced by `#` number.

- REQ-1: `exp(x)` — forward `c = exp(x)` with autograd. Mirrors
  `aten/src/ATen/native/UnaryOps.cpp:334 CREATE_UNARY_TORCH_IMPL_FUNC(exp_out, exp_stub)`.
  Backward per `tools/autograd/derivatives.yaml
  - name: exp(Tensor self) -> Tensor / self: grad * result.conj()` — the
  `ExpBackward` struct saves the output (not the input) so the VJP
  `dx = grad * exp(x) = grad * output` avoids re-running the kernel. GPU
  path dispatches through `dispatch_floating_dtype!` over `f32`/`f64`/
  `bf16`/`f16` (the `bf16` arm uses `exp_bf16_bf16` which routes through
  PTX `ex2.approx.f32` with an f32 internal accumulator and bf16 RNE
  store-back per crosslink #23). CPU path uses `crate::ops::elementwise::fast_exp`.

- REQ-2: `log(x)` — forward `c = ln(x)` (natural log) with autograd.
  Mirrors `UnaryOps.cpp:340 CREATE_UNARY_TORCH_IMPL_FUNC(log_out, log_stub)`.
  Backward per `tools/autograd/derivatives.yaml
  - name: log(Tensor self) -> Tensor / self: grad.div(self.conj())` —
  `LogBackward` saves the input and computes `dx = grad / x`. GPU path
  dispatches the same four dtypes (`log_bf16_bf16` uses PTX
  `lg2.approx.f32 * ln(2)`). CPU path uses
  `crate::ops::elementwise::fast_log`.

- REQ-3: `sin(x)` — forward `c = sin(x)` with autograd. Mirrors
  `UnaryOps.cpp:349 CREATE_UNARY_TORCH_IMPL_FUNC(sin_out, sin_stub)`.
  Backward per `tools/autograd/derivatives.yaml
  - name: sin(Tensor self) -> Tensor / self: grad * self.cos().conj()` —
  `SinBackward` saves the input and re-computes `cos(x)` on backward.
  CPU/GPU dispatch is delegated to `crate::ops::elementwise::fast_sin`
  (no transcendental-layer GPU dispatch wrapper).

- REQ-4: `cos(x)` — forward `c = cos(x)` with autograd. Mirrors
  `UnaryOps.cpp:328 CREATE_UNARY_TORCH_IMPL_FUNC(cos_out, cos_stub)`.
  Backward per `tools/autograd/derivatives.yaml
  - name: cos(Tensor self) -> Tensor / self: grad * -self.sin().conj()` —
  `CosBackward` saves the input and computes `dx = grad * (-sin(x))`.
  CPU/GPU dispatch is delegated to `crate::ops::elementwise::fast_cos`.

- REQ-5: `clamp(x, min, max)` — forward `c = x.clamp(min, max)` (Scalar
  bounds only) with autograd. Mirrors
  `aten/src/ATen/native/TensorCompare.cpp:831 TORCH_IMPL_FUNC(clamp_out)`
  which dispatches via `clamp_scalar_stub` / `clamp_max_scalar_stub` /
  `clamp_min_scalar_stub` based on which bounds are present. Backward
  per `tools/autograd/derivatives.yaml
  - name: clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
    self: clamp_backward(grad, self, min, max)` — `ClampBackward` saves
  the input and both bounds and computes
  `dx[i] = grad[i] if (min <= x[i] <= max) else 0`. GPU fast path for
  `f32`/`f64` via `backend.clamp_f32` / `backend.clamp_f64` (forward) and
  `backend.clamp_backward_f32` / `backend.clamp_backward_f64` (backward,
  closes #524). CPU path uses `unary_map` with the
  `if x < min { min } else if x > max { max } else { x }` lambda.
  **Diverges from upstream**: ferrotorch's `clamp` accepts BOTH bounds
  as required arguments (`T, T`), while upstream supports
  `Scalar? min=None` and `Scalar? max=None` Optional bounds; the
  one-sided forms `clamp_min` and `clamp_max` (`derivatives.yaml`
  `clamp_min`/`clamp_max` entries) are NOT-STARTED. **Diverges from
  upstream**: ferrotorch's `clamp` does NOT support tensor-valued
  bounds (`TORCH_IMPL_FUNC(clamp_Tensor_out)` at
  `TensorCompare.cpp:856`); the `clamp.Tensor` derivative-yaml entry
  is unreachable. **Diverges from upstream**: ferrotorch's clamp does
  NOT special-case `min == NaN` / `max == NaN` to fill the result with
  NaN (upstream `clamp_out` at `TensorCompare.cpp:839-846` does this);
  ferrotorch will compare with NaN bounds and produce a behavior the
  IEEE-754 comparison predicates dictate (likely all-NaN-or-untouched
  output depending on dtype) — see blocker #1298 for the parity-sweep
  arm that would surface this divergence.

- REQ-6: `exp2(x)` — `c = 2^x`, mirror of
  `UnaryOps.cpp:335 CREATE_UNARY_TORCH_IMPL_FUNC(exp2_out, exp2_stub)`.
  Backward per `derivatives.yaml
  - name: exp2(Tensor self) -> Tensor / self: grad * result.conj() * M_LN2`.
  NOT-STARTED — no `Exp2Backward`, no `pub fn exp2`, no `fast_exp2` kernel.
  Open prereq blocker #1303.

- REQ-7: `expm1(x)` — `c = exp(x) - 1` numerically-stable for small `x`,
  mirror of `UnaryOps.cpp:336 CREATE_UNARY_TORCH_IMPL_FUNC(expm1_out, expm1_stub)`.
  Backward per `derivatives.yaml
  - name: expm1(Tensor self) -> Tensor / self: grad * (result.conj() + 1)`.
  NOT-STARTED — no `Expm1Backward`, no `pub fn expm1`, no `fast_expm1`
  kernel. Open prereq blocker #1305.

- REQ-8: `log2(x)` — `c = log_2(x)`, mirror of
  `UnaryOps.cpp:343 CREATE_UNARY_TORCH_IMPL_FUNC(log2_out, log2_stub)`.
  Backward per `derivatives.yaml
  - name: log2(Tensor self) -> Tensor / self: grad / (self.conj() * 0.6931471805599453)`.
  NOT-STARTED — no `Log2Backward`, no `pub fn log2`, no `fast_log2` kernel.
  Open prereq blocker #1307.

- REQ-9: `log10(x)` — `c = log_10(x)`, mirror of
  `UnaryOps.cpp:341 CREATE_UNARY_TORCH_IMPL_FUNC(log10_out, log10_stub)`.
  Backward per `derivatives.yaml
  - name: log10(Tensor self) -> Tensor / self: grad / (self.conj() * 2.3025850929940456)`.
  NOT-STARTED — no `Log10Backward`, no `pub fn log10`, no `fast_log10` kernel.
  Open prereq blocker #1309.

- REQ-10: `log1p(x)` — `c = ln(1 + x)` numerically stable for small `x`,
  mirror of `UnaryOps.cpp:342 CREATE_UNARY_TORCH_IMPL_FUNC(log1p_out, log1p_stub)`.
  Backward per `derivatives.yaml
  - name: log1p(Tensor self) -> Tensor / self: log1p_backward(grad, self)`
  (which expands to `grad / (1 + self)` in the real-only path).
  NOT-STARTED — no `Log1pBackward`, no `pub fn log1p`, no `fast_log1p`
  kernel. Open prereq blocker #1311.

- REQ-11: `tan(x)` — `c = tan(x)`, mirror of
  `UnaryOps.cpp:360 CREATE_UNARY_TORCH_IMPL_FUNC(tan_out, tan_stub)`.
  Backward per `derivatives.yaml
  - name: tan(Tensor self) -> Tensor / self: grad * (1 + result.pow(2)).conj()`
  — saves the output `result` and uses `1 + result^2 = 1 + tan^2 = sec^2`.
  NOT-STARTED. Open prereq blocker #1313.

- REQ-12: `asin(x)` — `c = arcsin(x)`, mirror of
  `UnaryOps.cpp:323 CREATE_UNARY_TORCH_IMPL_FUNC(asin_out, asin_stub)`.
  Backward per `derivatives.yaml
  - name: asin(Tensor self) -> Tensor / self: grad * (-self * self + 1).rsqrt().conj()`.
  NOT-STARTED. Open prereq blocker #1315.

- REQ-13: `acos(x)` — `c = arccos(x)`, mirror of
  `UnaryOps.cpp:321 CREATE_UNARY_TORCH_IMPL_FUNC(acos_out, acos_stub)`.
  Backward per `derivatives.yaml
  - name: acos(Tensor self) -> Tensor / self: grad * -((-self * self + 1).rsqrt()).conj()`.
  NOT-STARTED. Open prereq blocker #1316.

- REQ-14: `atan(x)` — `c = arctan(x)`, mirror of
  `UnaryOps.cpp:325 CREATE_UNARY_TORCH_IMPL_FUNC(atan_out, atan_stub)`.
  Backward per `derivatives.yaml
  - name: atan(Tensor self) -> Tensor / self: grad / (self * self + 1).conj()`.
  NOT-STARTED. Open prereq blocker #1317.

- REQ-15: `atan2(y, x)` — `c = arctan(y / x)` two-argument with quadrant
  selection, mirror of
  `aten/src/ATen/native/BinaryOps.cpp:795 TORCH_IMPL_FUNC(atan2_out)`.
  Backward per `derivatives.yaml
  - name: atan2(Tensor self, Tensor other) -> Tensor / self, other:
  atan2_backward(grad, self, other, grad_input_mask)` (the
  `atan2_backward` helper computes
  `(self_grad, other_grad) = (grad * x / (x^2 + y^2), -grad * y / (x^2 + y^2))`).
  NOT-STARTED — binary form, requires broadcasting support layered onto a
  new backward. Open prereq blocker #1318.

- REQ-16: `sinh(x)` — `c = sinh(x)`, mirror of
  `UnaryOps.cpp:351 CREATE_UNARY_TORCH_IMPL_FUNC(sinh_out, sinh_stub)`.
  Backward per `derivatives.yaml
  - name: sinh(Tensor self) -> Tensor / self: grad * self.cosh().conj()`.
  NOT-STARTED. Open prereq blocker #1319.

- REQ-17: `cosh(x)` — `c = cosh(x)`, mirror of
  `UnaryOps.cpp:329 CREATE_UNARY_TORCH_IMPL_FUNC(cosh_out, cosh_stub)`.
  Backward per `derivatives.yaml
  - name: cosh(Tensor self) -> Tensor / self: grad * self.sinh().conj()`.
  NOT-STARTED. Open prereq blocker #1320.

- REQ-18: `tanh(x)` — `c = tanh(x)`, mirror of
  `UnaryOps.cpp:361 CREATE_UNARY_TORCH_IMPL_FUNC(tanh_out, tanh_stub)`.
  Backward per `derivatives.yaml
  - name: tanh(Tensor self) -> Tensor / self: tanh_backward(grad, result)`
  where `tanh_backward(grad, result) = grad * (1 - result^2)`. **Note**:
  the comment block in `transcendental.rs` (`// tanh (delegated)`)
  documents that `tanh` lives in `grad_fns::activation` because it is
  also an activation function. The implementation EXISTS in
  `grad_fns/activation.rs` (verified by the non-test consumer
  `pub fn tanh_t` in `methods.rs`), but it is NOT in the file
  under design. The route's `parity_ops` list includes `tanh` because
  the parity-sweep audit table treats `tanh` as one of the
  transcendental ops the route covers — but the IMPLEMENTATION DOES
  NOT LIVE IN `transcendental.rs`. NOT-STARTED **at this file's
  level** — the route attribution is ambiguous (the impl is real but
  belongs to a sibling file's design contract). Open prereq blocker
  #1321 tracks the route re-attribution (move `tanh` to the activation
  route, or split it). For the purposes of THIS doc the REQ is
  NOT-STARTED because no code in `transcendental.rs` satisfies it.

- REQ-19: `asinh(x)` — mirror of
  `UnaryOps.cpp:324 CREATE_UNARY_TORCH_IMPL_FUNC(asinh_out, asinh_stub)`.
  Backward per `derivatives.yaml
  - name: asinh(Tensor self) -> Tensor / self: grad * (self.pow(2) + 1).rsqrt().conj()`.
  NOT-STARTED. Open prereq blocker #1322.

- REQ-20: `acosh(x)` — mirror of
  `UnaryOps.cpp:322 CREATE_UNARY_TORCH_IMPL_FUNC(acosh_out, acosh_stub)`.
  Backward per `derivatives.yaml`: uses `sqrt(self - 1) * sqrt(self + 1)`
  decomposition. NOT-STARTED. Open prereq blocker #1323.

- REQ-21: `atanh(x)` — mirror of
  `UnaryOps.cpp:326 CREATE_UNARY_TORCH_IMPL_FUNC(atanh_out, atanh_stub)`.
  Backward per `derivatives.yaml
  - name: atanh(Tensor self) -> Tensor / self: grad * 1 / (1 - self.pow(2)).conj()`.
  NOT-STARTED. Open prereq blocker #1324.

- REQ-22: `sinc(x)` — `c = sin(pi*x) / (pi*x)` (the unnormalized
  `sinc(0) = 1` convention), mirror of
  `UnaryOps.cpp:350 CREATE_UNARY_TORCH_IMPL_FUNC(sinc_out, sinc_stub)`.
  Backward per `derivatives.yaml
  - name: sinc(Tensor self) -> Tensor / self: sinc_backward(grad, self)`.
  NOT-STARTED. Open prereq blocker #1325.

- REQ-23: `ceil(x)` — `c = ceil(x)`, mirror of
  `UnaryOps.cpp:316 CREATE_UNARY_TORCH_IMPL_INTEGER_NO_OP_FUNC(ceil_out, ceil_stub)`
  (the macro variant that no-ops on integer inputs). Backward per
  `derivatives.yaml
  - name: ceil(Tensor self) -> Tensor / self: zeros_like(grad)`. NOT-STARTED.
  Open prereq blocker #1326.

- REQ-24: `floor(x)` — mirror of
  `UnaryOps.cpp:317 CREATE_UNARY_TORCH_IMPL_INTEGER_NO_OP_FUNC(floor_out, floor_stub)`.
  Backward per `derivatives.yaml
  - name: floor(Tensor self) -> Tensor / self: zeros_like(grad)`.
  NOT-STARTED. Open prereq blocker #1327.

- REQ-25: `round(x)` — mirror of
  `UnaryOps.cpp:318 CREATE_UNARY_TORCH_IMPL_INTEGER_NO_OP_FUNC(round_out, round_stub)`,
  plus the `round.decimals` variant per
  `derivatives.yaml
  - name: round.decimals(Tensor self, *, int decimals) -> Tensor / self: zeros_like(grad)`.
  Round-half-to-even (banker's rounding) per upstream's `nearbyint`-based
  kernel. NOT-STARTED. Open prereq blocker #1328.

- REQ-26: `trunc(x)` — mirror of
  `UnaryOps.cpp:319 CREATE_UNARY_TORCH_IMPL_INTEGER_NO_OP_FUNC(trunc_out, trunc_stub)`.
  Backward per `derivatives.yaml
  - name: trunc(Tensor self) -> Tensor / self: zeros_like(grad)`.
  NOT-STARTED. Open prereq blocker #1329.

- REQ-27: `frac(x)` — `c = x - trunc(x)`, mirror of
  `UnaryOps.cpp:337 CREATE_UNARY_TORCH_IMPL_FUNC(frac_out, frac_stub)`.
  Backward per `derivatives.yaml
  - name: frac(Tensor self) -> Tensor / self: grad` (the gradient passes
  through directly because `frac` is piecewise linear with slope 1).
  NOT-STARTED. Open prereq blocker #1330.

- REQ-28: `sign(x)` — `c = sign(x)` returning `-1` / `0` / `+1`, mirror of
  `UnaryOps.cpp:348 CREATE_UNARY_TORCH_IMPL_FUNC(sign_out, sign_stub)`.
  Backward per `derivatives.yaml
  - name: sign(Tensor self) -> Tensor / self: zeros_like(grad)` (sign is
  a piecewise constant function, so the gradient is zero almost everywhere
  upstream defines it as exactly zero). NOT-STARTED. Open prereq blocker
  #1331.

- REQ-29: `signbit(x)` — `c[i] = (signbit(x[i]) ? true : false)` returning
  a Bool tensor, mirror of
  `UnaryOps.cpp:279 TORCH_META_FUNC(signbit)` (which uses
  `build_borrowing_unary_force_boolean_op`). Backward: signbit is
  bool-output so has no `derivatives.yaml` entry — it is non-differentiable.
  NOT-STARTED — additionally requires a Bool-output Tensor variant
  ferrotorch's `Tensor<T: Float>` cannot produce; would need an
  `IntTensor<bool>` or a new `BoolTensor` type. Open prereq blocker #1332.

- REQ-30: `clip(x, min, max)` — Python-level alias of `clamp` per
  `TensorCompare.cpp:902-930 Tensor clip(...)` (literal pass-through to
  `at::clamp(self, min, max)`). NOT-STARTED — ferrotorch has no
  `clip` symbol; a user who writes `tensor.clip_t(...)` gets a method
  not-found error. The fix is a one-line `pub fn clip = clamp` alias
  plus a `Tensor::clip_t` method. Open prereq blocker #1333.

- REQ-31: `copysign(magnitude, sign)` — `c = magnitude * sign(sign)`,
  mirror of `BinaryOps.cpp:865 TORCH_IMPL_FUNC(copysign_out)`. Backward
  per `derivatives.yaml
  - name: copysign.Tensor(Tensor self, Tensor other) -> Tensor / self:
  copysign_tensor_self_backward(grad, self, result)` (which masks the
  gradient by the sign-agreement predicate). NOT-STARTED — binary,
  requires broadcasting + backward. Open prereq blocker #1334.

- REQ-32: `nextafter(self, other)` — `c[i]` is the next representable
  floating-point value from `self[i]` toward `other[i]`, mirror of
  `BinaryOps.cpp:551 CREATE_BINARY_TORCH_IMPL_FUNC(nextafter_out, nextafter_stub)`.
  Backward per `derivatives.yaml
  - name: nextafter(Tensor self, Tensor other) -> Tensor / self:
  at::where(self != other, grad, 0)`. NOT-STARTED. Open prereq blocker
  #1335.

- REQ-33: `hypot(self, other)` — `c = sqrt(self^2 + other^2)`, mirror of
  `BinaryOps.cpp:548 CREATE_BINARY_TORCH_IMPL_FUNC(hypot_out, hypot_stub)`.
  Backward per `derivatives.yaml
  - name: hypot(Tensor self, Tensor other) -> Tensor / self: grad * self / result`
  (saves the output to avoid recomputing the sqrt). NOT-STARTED. Open
  prereq blocker #1336.

## Acceptance Criteria

- [ ] AC-1: `exp` parity-sweep at `--seeds 8` returns
  `[exp] N/N passed (0 skipped, 0 failed)` with `N >= 1`. NOT MET —
  currently `[exp] 0/12 passed (12 skipped, 0 failed)` because the
  parity-sweep runner has no `"exp" =>` arm in
  `tools/parity-sweep/runner/src/main.rs`. The forward+backward
  implementation exists but is unreached by the sweep. Blocker #1298.

- [ ] AC-2: `log` parity-sweep at `--seeds 8` returns
  `[log] N/N passed (0 skipped, 0 failed)` with `N >= 1`. NOT MET —
  currently `[log] 0/12 passed (12 skipped, 0 failed)`. Blocker #1298.

- [ ] AC-3: `sin` parity-sweep at `--seeds 8` returns
  `[sin] N/N passed (0 skipped, 0 failed)` with `N >= 1`. NOT MET —
  currently `[sin] 0/4 passed (4 skipped, 0 failed)`. Blocker #1298.

- [ ] AC-4: `cos` parity-sweep at `--seeds 8` returns
  `[cos] N/N passed (0 skipped, 0 failed)` with `N >= 1`. NOT MET —
  currently `[cos] 0/12 passed (12 skipped, 0 failed)`. Blocker #1298.

- [ ] AC-5: `clamp` parity-sweep at `--seeds 8` returns
  `[clamp] N/N passed (0 skipped, 0 failed)` with `N >= 1`. NOT MET —
  currently `[clamp] 0/28 passed (28 skipped, 0 failed)`. Blocker #1298.

- [x] AC-6: `cargo test -p ferrotorch-core grad_fns::transcendental`
  passes every forward, backward, no-grad, chain-rule, and
  numerical-gradient test in
  `ferrotorch-core/src/grad_fns/transcendental.rs` `mod tests` covering
  `exp` / `log` / `sin` / `cos` / `clamp`.

- [x] AC-7: `exp` forward dispatches through `dispatch_floating_dtype!`
  to the on-device kernel for CUDA inputs of dtype `f32`/`f64`/`bf16`/`f16`,
  picking up the #23 bf16 routing (`exp_bf16_bf16`).

- [x] AC-8: `log` forward dispatches through `dispatch_floating_dtype!`
  to the on-device kernel for CUDA inputs of dtype `f32`/`f64`/`bf16`/`f16`,
  picking up the #23 bf16 routing (`log_bf16_bf16`).

- [x] AC-9: `clamp` GPU fast path for `f32` and `f64` invokes
  `backend.clamp_f32` / `backend.clamp_f64` (forward) and
  `backend.clamp_backward_f32` / `backend.clamp_backward_f64` (backward),
  closing #524. CUDA inputs of dtype `bf16` or `f16` return
  `FerrotorchError::NotImplementedOnCuda { op: "ClampBackward" }` on
  backward (the `ClampBackward::backward` else-branch in
  `transcendental.rs` covers the unsupported-dtype path).

- [x] AC-10: `requires_grad=false` inputs return tensors with
  `grad_fn().is_none()` for all five shipped ops — verified by
  `test_exp_no_grad_fn_when_not_tracking`,
  `test_log_no_grad_fn_when_not_tracking`,
  `test_clamp_no_grad_fn_when_not_tracking` in `mod tests`. Sin/cos do
  not have a no-grad-fn test, but the structural property
  (`needs_grad_unary` gate before `from_operation`) is the same.

- [x] AC-11: Chain-rule tests pass —
  `test_chain_exp_log` (`c = log(exp(a))`, gradient `da = 1`) and
  `test_chain_sin_cos` (`c = cos(sin(a))`) live in `mod tests` and
  verify gradient flow across composed transcendental ops.

- [x] AC-12: Numerical-gradient (central finite difference) checks
  pass for every shipped op — `test_exp_numerical_grad`,
  `test_log_numerical_grad`, `test_sin_numerical_grad`,
  `test_cos_numerical_grad`, `test_clamp_numerical_grad_interior` in
  `mod tests`.

## Architecture

### Layer-0 dtype discriminators

Four `TypeId`-based helpers gate the GPU dispatch arms:
`fn is_f32`, `fn is_f64`, `fn is_bf16`, `fn is_f16` in `transcendental.rs`.
The `is_bf16` discriminator was added under crosslink #23 to route
bf16 buffers through the per-dtype f32-accumulator GPU kernels; the
`is_f16` discriminator landed under crosslink #1185 Phase 1.

### Layer-0 grad gate

`fn needs_grad_unary` in `transcendental.rs` returns
`is_grad_enabled() && a.requires_grad()`. Every `pub fn` in the file
checks this before attaching a grad-fn — when the gate is false the
function returns the raw forward output, skipping the
`Tensor::from_operation` wrap and avoiding the `Arc` allocation for
the backward struct.

### REQ-1 `exp` — `pub fn exp` + `struct ExpBackward`

`struct ExpBackward<T>` in `transcendental.rs` saves `input: Tensor<T>`
and `output: Tensor<T>`. The output is saved because the VJP is
`dx = grad * exp(x) = grad * output` — re-using the output avoids
re-running the kernel. `fn inputs` returns `vec![&self.input]` (not
`&self.output`) so the autograd-graph topological walk correctly
identifies the saved-for-backward operand. `fn backward` splits on
`grad_output.is_cuda()`: GPU goes through `arithmetic::mul` (under
`no_grad`); CPU walks the underlying `Vec<T>` data with a fused
`g * o` map. The forward `pub fn exp` wraps `fn exp_inner` in
`profile_op_scope`; `fn exp_inner` picks between the GPU dispatch arm
(via `dispatch_floating_dtype!` over the four floating dtypes) and the
CPU fallback via `crate::ops::elementwise::fast_exp`. **Non-test
production consumers** (>= 6, all live):

- `pub fn exp_t` in `ferrotorch-core/src/methods.rs` — the chainable
  `Tensor::exp_t()` method delegating to
  `crate::grad_fns::transcendental::exp`. This is the canonical
  R-DEFER-1 pub-API boundary (per S5 grandfathered as production
  consumer).
- `pub fn dual_exp` in `ferrotorch-core/src/autograd/forward_ad.rs` —
  uses `transcendental::exp` as the primal in a dual-number forward-AD
  step.
- `pub fn interpret` and `fn apply_elementwise_op` in
  `ferrotorch-jit/src/interpreter.rs` — the JIT interpreter's
  `IrOpKind::Exp` arm dispatches to `transcendental::exp` to evaluate
  traced graphs (two callsites).
- `impl Transform<T> for ExpTransform` in
  `ferrotorch-distributions/src/transforms.rs` —
  `transcendental::exp as exp_op` is the forward of `ExpTransform`
  used by `LogNormal` and `Gumbel` distributions.
- `pub fn sample_with_seed` (and surrounding
  `impl DiagonalGaussianDistribution<T>`) in
  `ferrotorch-diffusion/src/vae_encoder.rs` —
  `transcendental::exp(&scaled_logvar)` materializes the standard
  deviation in the VAE reparameterization trick.
- `impl<T> GradFn<T> for PoissonNLLBackward<T>` in
  `ferrotorch-nn/src/loss.rs` — `use ... transcendental::exp` inside
  the `PoissonNLLLoss` `log_input=False` branch.

The breadth of consumers (autograd, JIT, distributions, diffusion,
nn losses) is the strongest single signal that this is shipped
production surface, not vocabulary-only.

### REQ-2 `log` — `pub fn log` + `struct LogBackward`

`struct LogBackward<T>` in `transcendental.rs` saves `input: Tensor<T>`.
The VJP is `dx = grad / x`. The CPU backward walks the underlying
`Vec<T>` data with a fused `g / x` map; the GPU backward delegates to
`arithmetic::div` under `no_grad`. The forward `pub fn log` follows the
same structure as `exp` — GPU `dispatch_floating_dtype!` over four
floating dtypes or CPU fallback through
`crate::ops::elementwise::fast_log`. **Non-test production consumers**:

- `pub fn log_t` in `ferrotorch-core/src/methods.rs` — the
  `Tensor::log_t()` method (S5).
- `pub fn dual_log` in `ferrotorch-core/src/autograd/forward_ad.rs` —
  forward-AD primal.
- `pub fn interpret` and `fn apply_elementwise_op` in
  `ferrotorch-jit/src/interpreter.rs` — JIT `IrOpKind::Log` arm (two
  callsites).
- `impl<T> MultivariateNormal<T>` (file-level
  `use ferrotorch_core::grad_fns::transcendental::log as log_op`) in
  `ferrotorch-distributions/src/multivariate_normal.rs` — log-prob and
  half-log-det computations.
- `impl Transform<T> for ExpTransform` in
  `ferrotorch-distributions/src/transforms.rs` —
  `transcendental::log as log_op` is the inverse of `ExpTransform`.
- `ferrotorch-distributions/src/dirichlet.rs` (file-level
  `use ... transcendental::log as log_op`) — log-prob computation.

### REQ-3 `sin` — `pub fn sin` + `struct SinBackward`

`struct SinBackward<T>` in `transcendental.rs` saves `input: Tensor<T>`.
The VJP is `dx = grad * cos(x)`. Backward: the GPU path calls
`crate::grad_fns::transcendental::cos(&self.input)` (recursive
transcendental call) followed by `arithmetic::mul`, all under
`no_grad`. The CPU path computes `x.cos()` directly per element with
the `Float::cos` trait method. The forward delegates to
`crate::ops::elementwise::fast_sin` without a transcendental-layer
GPU-dispatch wrapper (unlike `exp`/`log` which have explicit
`dispatch_floating_dtype!` blocks for backend selection — `sin`/`cos`
rely on `fast_sin`/`fast_cos` to handle device routing internally).
**Non-test production consumers**:

- `pub fn sin_t` in `ferrotorch-core/src/methods.rs` — the
  `Tensor::sin_t()` method (S5).
- `pub fn dual_sin` in `ferrotorch-core/src/autograd/forward_ad.rs` —
  forward-AD primal.
- Recursive consumer through `CosBackward::backward` GPU path, which
  calls `transcendental::sin(&self.input)` inside the cosine VJP.

### REQ-4 `cos` — `pub fn cos` + `struct CosBackward`

`struct CosBackward<T>` in `transcendental.rs` saves `input: Tensor<T>`.
The VJP is `dx = grad * (-sin(x))`. Backward: GPU path calls
`crate::grad_fns::transcendental::sin(&self.input)`, then
`arithmetic::neg`, then `arithmetic::mul`, under `no_grad`. CPU path
computes `-x.sin()` directly per element. **Non-test production
consumers**:

- `pub fn cos_t` in `ferrotorch-core/src/methods.rs` — the
  `Tensor::cos_t()` method (S5).
- `pub fn dual_cos` in `ferrotorch-core/src/autograd/forward_ad.rs` —
  forward-AD primal.
- Recursive consumer through `SinBackward::backward` GPU path
  (symmetric to REQ-3).

### REQ-5 `clamp` — `pub fn clamp` + `struct ClampBackward`

`struct ClampBackward<T>` in `transcendental.rs` saves `input`,
`min: T`, `max: T`. The forward `pub fn clamp` checks
`input.is_cuda() && (is_f32::<T>() || is_f64::<T>())` and dispatches
to `backend.clamp_f32` / `backend.clamp_f64` with scalar bounds
converted via `min.to_f32()` / `max.to_f32()` (saturating to
`f32::MIN`/`f32::MAX` if the source `T` doesn't fit). Otherwise the
CPU path uses `unary_map` with the
`if x < min { min } else if x > max { max } else { x }` lambda.
Backward: the `fn backward` impl on `ClampBackward` enters the GPU
fast path when
`grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>())` and
invokes `backend.clamp_backward_f32` / `backend.clamp_backward_f64`,
which apply the mask `(min <= x <= max) ? grad : 0` on-device per
crosslink #524. The CUDA-non-f32/f64 path (bf16, f16) returns
`FerrotorchError::NotImplementedOnCuda { op: "ClampBackward" }`. CPU
path walks the data with the
`if x >= min && x <= max { g } else { zero }` mask. **Non-test
production consumers**:

- `pub fn clamp_t` in `ferrotorch-core/src/methods.rs` —
  `Tensor::clamp_t(min, max)` method (S5).
- `impl Transform<T> for SigmoidTransform` in
  `ferrotorch-distributions/src/transforms.rs` —
  `transcendental::clamp(y, eps, one - eps)` keeps the
  inverse-sigmoid argument in `(eps, 1-eps)`.
- `impl<T> HuberBackward<T>` block in `ferrotorch-nn/src/loss.rs`
  (surrounding `BCEWithLogitsLoss` use-site at the imports line
  `use ferrotorch_core::grad_fns::transcendental::clamp;`) — the
  numerically-stable `clamp(prob, eps, 1-eps)`-then-`log` formulation.
- `impl_activation_module!(ReLU6)` body via
  `impl ReLU6 { pub fn forward<T>(...) }` in
  `ferrotorch-nn/src/activation.rs` —
  `transcendental::clamp(input, zero, six)` is the body of the
  differentiable ReLU6 activation.
- `impl_activation_module!(Hardtanh)` body via
  `impl Hardtanh { pub fn forward<T>(...) }` in
  `ferrotorch-nn/src/activation.rs` —
  `transcendental::clamp(input, min, max)` is the body of the
  Hardtanh activation.

### REQ-18 `tanh` cross-reference

The comment block in `transcendental.rs` titled "tanh (delegated)"
documents that `tanh` and `sigmoid` live in `grad_fns::activation`
because they are activation functions. The non-test consumer
`pub fn tanh_t` in `methods.rs` calls
`crate::grad_fns::activation::tanh`, not
`crate::grad_fns::transcendental::tanh`. The route's inclusion of
`tanh` in `parity_ops` for this file is a route-attribution drift:
the implementation does not live here. Re-attribution is tracked by
blocker #1321.

### REQ-6 through REQ-17 + REQ-19 through REQ-33 (NOT-STARTED ops)

None of the 27 remaining ops have an implementation in this file or in
its dependency layer. There is no `fast_<op>` kernel in
`ferrotorch-core/src/ops/elementwise.rs` for any of them (verified
2026-05-25 via `grep -nE "pub fn fast_(exp2|expm1|log2|log10|log1p|tan|asin|acos|atan|sinh|cosh|asinh|acosh|atanh|sinc|ceil|floor|round|trunc|frac|sign|signbit|clip|copysign|nextafter|hypot)"
ferrotorch-core/src/ops/elementwise.rs` returns no matches). There is
no `<Op>Backward` struct, no `pub fn <op>`, no method-surface entry in
`methods.rs`. The parity-sweep runner has no `"<op>" =>` arm in
`tools/parity-sweep/runner/src/main.rs`. Each NOT-STARTED REQ tracks
its own open prereq blocker (#1303-#1336 with gaps; see REQ-status
table).

## Parity contract

| Op | Status | Upstream entry | Backward formula source | Edge cases mirrored |
|---|---|---|---|---|
| `exp` | impl SHIPPED, parity arm BLOCKED #1298 | `UnaryOps.cpp:334 CREATE_UNARY_TORCH_IMPL_FUNC(exp_out, exp_stub)` | `derivatives.yaml` `exp: grad * result.conj()` | NaN propagates (Rust `f32::exp(NaN) == NaN`). `+inf -> +inf`, `-inf -> 0`. Denormals: rely on Rust hardware `exp`; bf16 path uses PTX `ex2.approx.f32` with f32 accumulator and bf16 RNE store-back per #23. Non-contiguous CUDA: routed through the same `gpu_handle()` materialize the kernel expects (see `transcendental.rs` GPU dispatch block). |
| `log` | impl SHIPPED, parity arm BLOCKED #1298 | `UnaryOps.cpp:340 CREATE_UNARY_TORCH_IMPL_FUNC(log_out, log_stub)` | `derivatives.yaml` `log: grad.div(self.conj())` | `log(0) = -inf`, `log(negative) = NaN`, `log(NaN) = NaN`. Backward at `x=0` produces `grad / 0 = +/-inf` per IEEE-754 — matches upstream. bf16 path uses PTX `lg2.approx.f32 * ln(2)` per #23. |
| `sin` | impl SHIPPED, parity arm BLOCKED #1298 | `UnaryOps.cpp:349 CREATE_UNARY_TORCH_IMPL_FUNC(sin_out, sin_stub)` | `derivatives.yaml` `sin: grad * self.cos().conj()` | NaN/Inf propagates. Large-magnitude inputs lose precision from argument reduction; `fast_sin` documents its reduction algorithm. Backward re-computes `cos` rather than saving it. |
| `cos` | impl SHIPPED, parity arm BLOCKED #1298 | `UnaryOps.cpp:328 CREATE_UNARY_TORCH_IMPL_FUNC(cos_out, cos_stub)` | `derivatives.yaml` `cos: grad * -self.sin().conj()` | Symmetric to sin. |
| `clamp` | impl SHIPPED, parity arm BLOCKED #1298 | `TensorCompare.cpp:831 TORCH_IMPL_FUNC(clamp_out)` | `derivatives.yaml` `clamp: clamp_backward(grad, self, min, max)` | NaN inputs: `x.is_nan() && (NaN < min) == false && (NaN > max) == false` so the CPU `unary_map` passes NaN through unchanged. Upstream `clamp_out` fills with NaN when EITHER BOUND is NaN — ferrotorch does NOT replicate this special case. Boundary tie: at `x == min` and `x == max` ferrotorch's CPU backward returns `g` (interior gradient) because the test is `>=` and `<=`; upstream's `clamp_backward` may differ in edge handling. CUDA bf16/f16 backward: `NotImplementedOnCuda` (#524 deferred bf16 path). |
| All 28 NOT-STARTED ops | — | (see upstream cites in REQ list) | (see derivatives.yaml cites in REQ list) | No implementation — edge-case parity is by definition not yet defined for the ferrotorch side. Each op has a dedicated open prereq blocker referenced in the REQ status table. |

Parity-sweep audit reference: all 33 op entries are **MISSING** from
`tools/parity-sweep/parity_audit.json` (verified 2026-05-25 via
audit-key check script). Adding the audit entries lands as part of
blocker #1298 (the parity-runner-arm blocker) for the 5 shipped ops;
the 28 NOT-STARTED ops will pick up audit entries when their
impl-and-arm blocker closes.

## Verification

### Existing unit tests (all passing, AC-6)

Located at `ferrotorch-core/src/grad_fns/transcendental.rs` inside the
`#[cfg(test)] mod tests` block at the bottom of the file. Key tests:

- Forward: `test_exp_forward`, `test_log_forward`, `test_sin_forward`,
  `test_cos_forward`, `test_clamp_forward`.
- Backward scalar: `test_exp_backward`, `test_log_backward`,
  `test_sin_backward`, `test_sin_backward_pi_over_3`,
  `test_cos_backward`, `test_cos_backward_pi_over_2`,
  `test_clamp_backward_interior`, `test_clamp_backward_clamped_low`,
  `test_clamp_backward_clamped_high`.
- Chain-rule: `test_chain_exp_log` (verifies `log(exp(x))` gradient
  is exactly 1), `test_chain_sin_cos` (verifies
  `cos(sin(x))` gradient against the analytic expression
  `-sin(sin(x)) * cos(x)`).
- No-grad: `test_exp_no_grad_fn_when_not_tracking`,
  `test_log_no_grad_fn_when_not_tracking`,
  `test_clamp_no_grad_fn_when_not_tracking`.
- Numerical-gradient (central finite difference, AC-12):
  `test_exp_numerical_grad`, `test_log_numerical_grad`,
  `test_sin_numerical_grad`, `test_cos_numerical_grad`,
  `test_clamp_numerical_grad_interior`.

Run:

```bash
cargo test -p ferrotorch-core grad_fns::transcendental
```

### Parity-sweep status

All 33 ops named in the route's `parity_ops` list currently report
`[<op>] 0/N passed (N skipped, 0 failed)` (verified 2026-05-25 via
`./target/release/parity-sweep sweep --op <op> --seeds 4`). The
"skipped" rows are NOT failures — they indicate the runner returned
`Ok(None)` because there is no `"<op>" =>` arm in
`tools/parity-sweep/runner/src/main.rs`. The five shipped ops
(`exp`/`log`/`sin`/`cos`/`clamp`) have working implementations and
tests pass; they are unreached by parity-sweep solely because the
dispatch arm hasn't been wired. Blocker #1298 tracks adding the arms.

```bash
# Once #1298 closes, all five should report 'passed' (the integer
# count depends on op_db sample density per op):
./target/release/parity-sweep sweep --op exp   --seeds 8
./target/release/parity-sweep sweep --op log   --seeds 8
./target/release/parity-sweep sweep --op sin   --seeds 8
./target/release/parity-sweep sweep --op cos   --seeds 8
./target/release/parity-sweep sweep --op clamp --seeds 8
```

The expected `grep -c "passed (0 skipped, 0 failed)"` count is `1`
per op once the runner arms land. The other 28 ops require both an
implementation (per the per-op blockers) and a runner arm.

## REQ status table

Five SHIPPED (impl + non-test production consumer present + tests
pass + parity-arm BLOCKED awaiting #1298). Twenty-eight NOT-STARTED
(no impl + concrete prereq blocker filed).

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (exp) | SHIPPED | impl: `pub fn exp` in `transcendental.rs` + `struct ExpBackward<T>` in `transcendental.rs` mirroring `aten/src/ATen/native/UnaryOps.cpp:334 CREATE_UNARY_TORCH_IMPL_FUNC(exp_out, exp_stub)` with backward formula `grad * result.conj()` per `tools/autograd/derivatives.yaml` `exp`. Non-test production consumers: `pub fn exp_t` in `ferrotorch-core/src/methods.rs` (S5-grandfathered boundary method), `pub fn dual_exp` in `ferrotorch-core/src/autograd/forward_ad.rs` (forward-AD primal), `pub fn interpret` and `fn apply_elementwise_op` in `ferrotorch-jit/src/interpreter.rs` (JIT `IrOpKind::Exp` two sites), `impl Transform<T> for ExpTransform` in `ferrotorch-distributions/src/transforms.rs`, `impl<T> DiagonalGaussianDistribution<T>` in `ferrotorch-diffusion/src/vae_encoder.rs` (VAE reparameterization), `impl<T> GradFn<T> for PoissonNLLBackward<T>` in `ferrotorch-nn/src/loss.rs`. Parity-sweep arm BLOCKED #1298 (currently `[exp] 0/12 passed (12 skipped)`). |
| REQ-2 (log) | SHIPPED | impl: `pub fn log` in `transcendental.rs` + `struct LogBackward<T>` mirroring `UnaryOps.cpp:340 CREATE_UNARY_TORCH_IMPL_FUNC(log_out, log_stub)` with backward `grad / x` per `derivatives.yaml` `log`. Non-test production consumers: `pub fn log_t` in `methods.rs` (S5), `pub fn dual_log` in `autograd/forward_ad.rs`, `pub fn interpret` and `fn apply_elementwise_op` in `ferrotorch-jit/src/interpreter.rs` (JIT `IrOpKind::Log` two sites), `impl<T> MultivariateNormal<T>` in `ferrotorch-distributions/src/multivariate_normal.rs`, `impl Transform<T> for ExpTransform` in `ferrotorch-distributions/src/transforms.rs`, `ferrotorch-distributions/src/dirichlet.rs` (file-level `use` consumer). Parity-sweep arm BLOCKED #1298 (currently `[log] 0/12 passed`). |
| REQ-3 (sin) | SHIPPED | impl: `pub fn sin` in `transcendental.rs` + `struct SinBackward<T>` mirroring `UnaryOps.cpp:349 CREATE_UNARY_TORCH_IMPL_FUNC(sin_out, sin_stub)` with backward `grad * cos(x)` per `derivatives.yaml` `sin`. Non-test production consumers: `pub fn sin_t` in `methods.rs` (S5), `pub fn dual_sin` in `autograd/forward_ad.rs`, plus recursive consumer through `CosBackward::backward` GPU path which invokes `transcendental::sin`. Parity-sweep arm BLOCKED #1298 (currently `[sin] 0/4 passed`). |
| REQ-4 (cos) | SHIPPED | impl: `pub fn cos` in `transcendental.rs` + `struct CosBackward<T>` mirroring `UnaryOps.cpp:328 CREATE_UNARY_TORCH_IMPL_FUNC(cos_out, cos_stub)` with backward `grad * (-sin(x))` per `derivatives.yaml` `cos`. Non-test production consumers: `pub fn cos_t` in `methods.rs` (S5), `pub fn dual_cos` in `autograd/forward_ad.rs`, plus recursive consumer through `SinBackward::backward` GPU path which invokes `transcendental::cos`. Parity-sweep arm BLOCKED #1298 (currently `[cos] 0/12 passed`). |
| REQ-5 (clamp) | SHIPPED | impl: `pub fn clamp` in `transcendental.rs` + `struct ClampBackward<T>` mirroring `aten/src/ATen/native/TensorCompare.cpp:831 TORCH_IMPL_FUNC(clamp_out)` with backward `clamp_backward(grad, self, min, max)` per `derivatives.yaml` `clamp`. GPU fast path via `backend.clamp_f32`/`backend.clamp_f64` (forward) and `backend.clamp_backward_f32`/`backend.clamp_backward_f64` (backward) per #524. Non-test production consumers: `pub fn clamp_t` in `methods.rs` (S5), `impl Transform<T> for SigmoidTransform` in `ferrotorch-distributions/src/transforms.rs`, `impl BCEWithLogitsLoss` family in `ferrotorch-nn/src/loss.rs`, `impl ReLU6` in `ferrotorch-nn/src/activation.rs` (the ReLU6 activation forward), `impl Hardtanh` in `ferrotorch-nn/src/activation.rs` (the Hardtanh activation forward). Diverges from upstream on three points documented in REQ-5: scalar-only bounds (no Tensor bounds), bilateral (no Optional one-sided), no NaN-bound special case. Parity-sweep arm BLOCKED #1298 (currently `[clamp] 0/28 passed`). |
| REQ-6 (exp2) | NOT-STARTED | open prereq blocker #1303 — no `Exp2Backward`, no `pub fn exp2`, no `fast_exp2` kernel. Upstream `UnaryOps.cpp:335 CREATE_UNARY_TORCH_IMPL_FUNC(exp2_out, exp2_stub)`. |
| REQ-7 (expm1) | NOT-STARTED | open prereq blocker #1305 — no `Expm1Backward`, no `pub fn expm1`, no `fast_expm1` kernel. Upstream `UnaryOps.cpp:336`. |
| REQ-8 (log2) | NOT-STARTED | open prereq blocker #1307 — no `Log2Backward`, no `pub fn log2`, no `fast_log2` kernel. Upstream `UnaryOps.cpp:343`. |
| REQ-9 (log10) | NOT-STARTED | open prereq blocker #1309 — no `Log10Backward`, no `pub fn log10`, no `fast_log10` kernel. Upstream `UnaryOps.cpp:341`. |
| REQ-10 (log1p) | NOT-STARTED | open prereq blocker #1311 — no `Log1pBackward`, no `pub fn log1p`, no `fast_log1p` kernel. Upstream `UnaryOps.cpp:342`. |
| REQ-11 (tan) | NOT-STARTED | open prereq blocker #1313 — no `TanBackward`, no `pub fn tan`, no `fast_tan` kernel. Upstream `UnaryOps.cpp:360`. |
| REQ-12 (asin) | NOT-STARTED | open prereq blocker #1315 — no `AsinBackward`, no `pub fn asin`, no `fast_asin` kernel. Upstream `UnaryOps.cpp:323`. |
| REQ-13 (acos) | NOT-STARTED | open prereq blocker #1316 — no `AcosBackward`, no `pub fn acos`, no `fast_acos` kernel. Upstream `UnaryOps.cpp:321`. |
| REQ-14 (atan) | NOT-STARTED | open prereq blocker #1317 — no `AtanBackward`, no `pub fn atan`, no `fast_atan` kernel. Upstream `UnaryOps.cpp:325`. |
| REQ-15 (atan2) | NOT-STARTED | open prereq blocker #1318 — binary op, no `Atan2Backward`, no `pub fn atan2`. Upstream `BinaryOps.cpp:795 TORCH_IMPL_FUNC(atan2_out)`. |
| REQ-16 (sinh) | NOT-STARTED | open prereq blocker #1319 — no `SinhBackward`, no `pub fn sinh`, no `fast_sinh` kernel. Upstream `UnaryOps.cpp:351`. |
| REQ-17 (cosh) | NOT-STARTED | open prereq blocker #1320 — no `CoshBackward`, no `pub fn cosh`, no `fast_cosh` kernel. Upstream `UnaryOps.cpp:329`. |
| REQ-18 (tanh) | NOT-STARTED at this file's level | open prereq blocker #1321 — `tanh` impl exists in `grad_fns::activation` (with non-test consumer `pub fn tanh_t` in `methods.rs`), but it does NOT live in `transcendental.rs`. Route attribution drift: the route includes `tanh` in this file's `parity_ops` but the implementation belongs to the activation file's design contract. Blocker #1321 tracks re-attribution. |
| REQ-19 (asinh) | NOT-STARTED | open prereq blocker #1322 — no `AsinhBackward`, no `pub fn asinh`, no `fast_asinh` kernel. Upstream `UnaryOps.cpp:324`. |
| REQ-20 (acosh) | NOT-STARTED | open prereq blocker #1323 — no `AcoshBackward`, no `pub fn acosh`, no `fast_acosh` kernel. Upstream `UnaryOps.cpp:322`. |
| REQ-21 (atanh) | NOT-STARTED | open prereq blocker #1324 — no `AtanhBackward`, no `pub fn atanh`, no `fast_atanh` kernel. Upstream `UnaryOps.cpp:326`. |
| REQ-22 (sinc) | NOT-STARTED | open prereq blocker #1325 — no `SincBackward`, no `pub fn sinc`, no `fast_sinc` kernel. Upstream `UnaryOps.cpp:350`. |
| REQ-23 (ceil) | NOT-STARTED | open prereq blocker #1326 — no `CeilBackward`, no `pub fn ceil`, no `fast_ceil` kernel. Upstream `UnaryOps.cpp:316 CREATE_UNARY_TORCH_IMPL_INTEGER_NO_OP_FUNC(ceil_out, ceil_stub)` — backward is `zeros_like(grad)`. |
| REQ-24 (floor) | NOT-STARTED | open prereq blocker #1327 — no `FloorBackward`, no `pub fn floor`, no `fast_floor` kernel. Upstream `UnaryOps.cpp:317`. |
| REQ-25 (round) | NOT-STARTED | open prereq blocker #1328 — no `RoundBackward`, no `pub fn round`, no `fast_round` kernel. Upstream `UnaryOps.cpp:318`. |
| REQ-26 (trunc) | NOT-STARTED | open prereq blocker #1329 — no `TruncBackward`, no `pub fn trunc`, no `fast_trunc` kernel. Upstream `UnaryOps.cpp:319`. |
| REQ-27 (frac) | NOT-STARTED | open prereq blocker #1330 — no `FracBackward`, no `pub fn frac`, no `fast_frac` kernel. Upstream `UnaryOps.cpp:337`. |
| REQ-28 (sign) | NOT-STARTED | open prereq blocker #1331 — no `SignBackward`, no `pub fn sign`, no `fast_sign` kernel. Upstream `UnaryOps.cpp:348` — backward is `zeros_like(grad)`. |
| REQ-29 (signbit) | NOT-STARTED | open prereq blocker #1332 — requires Bool-output tensor variant (ferrotorch's `Tensor<T: Float>` cannot produce). Upstream `UnaryOps.cpp:279 TORCH_META_FUNC(signbit)` uses `build_borrowing_unary_force_boolean_op`. |
| REQ-30 (clip) | NOT-STARTED | open prereq blocker #1333 — one-line alias of `clamp` not yet exposed. Upstream `TensorCompare.cpp:918-930 Tensor clip(...)` is a literal pass-through to `at::clamp`. |
| REQ-31 (copysign) | NOT-STARTED | open prereq blocker #1334 — binary op, no `CopysignBackward`, no `pub fn copysign`. Upstream `BinaryOps.cpp:865 TORCH_IMPL_FUNC(copysign_out)`. |
| REQ-32 (nextafter) | NOT-STARTED | open prereq blocker #1335 — binary op, no `NextafterBackward`, no `pub fn nextafter`. Upstream `BinaryOps.cpp:551 CREATE_BINARY_TORCH_IMPL_FUNC(nextafter_out, nextafter_stub)`. |
| REQ-33 (hypot) | NOT-STARTED | open prereq blocker #1336 — binary op, no `HypotBackward`, no `pub fn hypot`. Upstream `BinaryOps.cpp:548 CREATE_BINARY_TORCH_IMPL_FUNC(hypot_out, hypot_stub)`. |
