# ferrotorch-nn — `activation` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/activation.py
  - aten/src/ATen/native/Activation.cpp
-->

## Summary

`ferrotorch-nn/src/activation.rs` ships the stateful `Module<T>` wrappers
around the differentiable activation primitives in
`ferrotorch_core::grad_fns::activation`. Each struct is a zero-parameter
module (except `PReLU<T>`, which owns a single `alpha: Parameter<T>`)
that delegates `forward` to the core differentiable primitive while
exposing a `training` flag for `Module<T>` consistency. Mirrors the class
surface in `torch/nn/modules/activation.py` and the math primitives in
`aten/src/ATen/native/Activation.cpp`.

## Requirements

- REQ-1: `pub struct ReLU` zero-param module with `forward(x) = max(0, x)`.
  Mirrors `torch.nn.ReLU` at `torch/nn/modules/activation.py:104-152`.

- REQ-2: `pub struct Sigmoid`, `pub struct Tanh` — element-wise logistic
  and hyperbolic tangent. Mirror `torch.nn.{Sigmoid,Tanh}` at
  `torch/nn/modules/activation.py:337-434`.

- REQ-3: `pub struct GELU` with `pub enum GeluApproximate` re-exported from
  `ferrotorch_core::grad_fns::activation::GeluApproximate`. Modes:
  `None` (exact erf), `Tanh` (the `approximate="tanh"` mode used by GPT-2 /
  Llama), `Sigmoid` (the `x * sigmoid(1.702 * x)` fast approximation).
  Mirrors `torch.nn.GELU` at `torch/nn/modules/activation.py:777-824`.

- REQ-4: `pub struct SiLU` — `x * sigmoid(x)`. Mirrors `torch.nn.SiLU` at
  `torch/nn/modules/activation.py:435-484`.

- REQ-5: `pub struct Softmax`, `pub struct LogSoftmax`, `pub struct Softmin`,
  `pub struct Softmax2d` — softmax variants. The 1-D/N-D variants accept a
  `dim` argument; `Softmax2d` operates over channel dim of a 4-D input.
  Mirror `torch.nn.{Softmax,LogSoftmax,Softmin,Softmax2d}` at
  `torch/nn/modules/activation.py:1709-1929`.

- REQ-6: `pub struct LeakyReLU`, `pub struct PReLU<T: Float>`,
  `pub struct ELU`, `pub struct CELU`, `pub struct SELU`,
  `pub struct RReLU`. Each carries the parameter (negative_slope / alpha /
  lower-upper bound for RReLU). Mirror
  `torch.nn.{LeakyReLU,PReLU,ELU,CELU,SELU,RReLU}` at
  `torch/nn/modules/activation.py:153-218, 575-735, 874-931, 1575-1656`.

- REQ-7: `pub struct Hardtanh`, `pub struct ReLU6`, `pub struct HardSigmoid`,
  `pub struct HardSwish`, `pub struct Hardshrink`, `pub struct Softshrink`,
  `pub struct Tanhshrink`, `pub struct Softsign`, `pub struct LogSigmoid`,
  `pub struct Threshold`, `pub struct Softplus`, `pub struct Mish`,
  `pub struct GLU` — the remaining piecewise / shrink / threshold modules.
  Mirror their PyTorch counterparts in
  `torch/nn/modules/activation.py:219-336, 364-406, 485-574, 530-574,
  680-735, 736-776, 825-873, 958-1056`.

- REQ-8: PReLU owns a learnable `alpha: Parameter<T>` and reports it via
  `parameters` / `named_parameters("alpha")`. Mirrors
  `torch.nn.PReLU.weight` at `torch/nn/modules/activation.py:1575-1656`.

- REQ-9: Every module implements `Module<T>::{forward, parameters,
  parameters_mut, named_parameters, train, eval, is_training}`. Mirrors
  the upstream `Module` ABI from `torch/nn/modules/module.py`.

- REQ-10: Every numeric activation has a backward node attached when
  `requires_grad` is set on the input (and grad is enabled), via the
  underlying `act::*` primitive in
  `ferrotorch_core::grad_fns::activation`. Mirrors the autograd nodes
  registered at `aten/src/ATen/native/Activation.cpp`.

- REQ-11: `Softmax2d` accepts only 4-D input `[N, C, H, W]`. On CUDA input
  with a registered GPU backend it dispatches to
  `GpuBackend::softmax2d_f32` (channel-axis softmax PTX kernel); CPU input
  uses the per-position loop. CUDA input with no backend registered is
  rejected with `NotImplementedOnCuda`. Forward-only (no backward node),
  matching the CPU path. Closed by #1451.

## Acceptance Criteria

- [x] AC-1: `pub struct ReLU::new()` and `pub struct ReLU::default()`.
- [x] AC-2: `forward` of every activation matches the underlying
  `act::*` primitive byte-for-byte.
- [x] AC-3: `Module<T>` implementations report `parameters` (0 for stateless,
  1 for `PReLU`).
- [x] AC-4: `GeluApproximate::{None,Tanh,Sigmoid}` are reachable through
  `pub use act::GeluApproximate` at the top of `activation.rs`.
- [x] AC-5: `PReLU::new(init)?` constructs a 1-element `alpha` parameter
  and `named_parameters` returns `("alpha", ..)`.
- [x] AC-6: `Threshold::new(threshold, value)` clamps `x <= threshold` to
  `value` and leaves `x > threshold` unchanged.
- [x] AC-7: `RReLU::new(lower, upper)` samples a per-element negative slope
  from `U(lower, upper)` in training mode and uses the midpoint in eval.
- [x] AC-8: `train()` / `eval()` flips the internal `training` flag.
- [x] AC-9: `Softmax2d` GPU forward — `Softmax2d::forward` dispatches to
  `backend.softmax2d_f32(input, n, c, h*w)` for CUDA input; the
  `softmax2d_kernel` PTX lives in `ferrotorch-gpu/src/group_norm.rs`,
  wired through `CudaBackendImpl::softmax2d_f32`. Closed by #1451.
  Runtime parity pinned by `#[ignore]`'d `softmax2d_forward_gpu_matches_cpu`.

## Architecture

### `impl_activation_module!` macro

A declarative macro defined at the top of `activation.rs` (lines 21-53)
implements `Module<T>::{forward, parameters, parameters_mut,
named_parameters, train, eval, is_training}` for every zero-parameter
activation. The macro is the single source of `Module` trait
implementations across REQ-1, REQ-2, REQ-3, REQ-4, REQ-5 (excluding
`PReLU` which has its own hand-written impl for the `alpha` parameter).

### Standard zero-param activations (REQ-1, REQ-2, REQ-3, REQ-4)

`pub struct ReLU`, `pub struct Sigmoid`, `pub struct Tanh`,
`pub struct GELU`, `pub struct SiLU` all carry a single `training: bool`
field and a `forward<T: Float>` method that delegates to the
corresponding `act::*` primitive in `ferrotorch_core::grad_fns::activation`.
Each has a `Default` impl pointing at `Self::new()`.

`GELU` carries an additional `approximate: GeluApproximate` field and
exposes `GELU::with_approximate(mode)`. The re-export
`pub use act::GeluApproximate` at the top of the module surfaces the enum
to users who write `nn::GELU` directly.

### Softmax family (REQ-5)

`pub struct Softmax`, `pub struct LogSoftmax`, `pub struct Softmin` accept
a `dim: i64` field at construction; `forward` delegates to
`act::{softmax,log_softmax,softmin}` with the resolved dim.

`pub struct Softmax2d` requires 4-D `[N, C, H, W]` input and runs a CPU
loop that softmaxes over the channel dim (`dim=1`) at each `(n, h, w)`
spatial position. CUDA input is rejected with `NotImplementedOnCuda`;
blocker #1451 tracks the GPU forward.

### Parameterised activations (REQ-6, REQ-8)

`pub struct LeakyReLU` holds `negative_slope: f64`. `pub struct ELU` and
`pub struct CELU` hold `alpha: f64`. `pub struct SELU` is a zero-param
wrapper around `ELU(alpha=1.6732632...)` then scaled by
`1.0507009...`. `pub struct RReLU` carries `lower` and `upper` (CPU only,
samples a fresh per-element slope each forward in training mode).

`pub struct PReLU<T: Float>` owns a `pub alpha: Parameter<T>` with the
`alpha` slot exposed via `named_parameters()` as the string
`"alpha"` — see `pub fn forward` / `pub fn parameters` / `pub fn
named_parameters` in the PReLU impl in `activation.rs`.

### Saturating / piecewise activations (REQ-7)

`pub struct Hardtanh` holds `min_val: f64, max_val: f64`. `pub struct ReLU6`
is a fixed `Hardtanh(0, 6)`. `pub struct HardSigmoid` and
`pub struct HardSwish` implement the MobileNetV3 piecewise approximations.
`pub struct Hardshrink`, `pub struct Softshrink`, `pub struct Tanhshrink`,
`pub struct Softsign`, `pub struct LogSigmoid` delegate to the
corresponding `act::*` primitives.

`pub struct Threshold` holds `threshold: f64, value: f64` and clamps
`x <= threshold` to `value`. `pub struct Softplus` holds `beta: f64,
threshold: f64` (matching PyTorch defaults `beta=1, threshold=20`).
`pub struct Mish` is zero-param. `pub struct GLU` holds `dim: i64`
and applies gated linear units (split input along `dim` into halves
`(a, b)`, return `a * sigmoid(b)`).

### Module trait surface (REQ-9)

Every type implements `Module<T>` via either the `impl_activation_module!`
macro (zero-param) or a hand-written impl (`PReLU`). All types are
generic-free at the struct level (carry only `training: bool` plus their
parameter scalars) — generic `T: Float` is bound at the `forward<T>` and
`Module<T>::forward` call sites. `PReLU<T>` is the lone exception with
the parameter carrying the generic.

### Backward (REQ-10)

`forward` calls the underlying `act::*` primitive in
`ferrotorch_core::grad_fns::activation`. Each primitive attaches the
appropriate backward node (`ReluBackward`, `SigmoidBackward`,
`TanhBackward`, `GeluBackward`, `SiluBackward`, `SoftmaxBackward`,
`LogSoftmaxBackward`, `LeakyReluBackward`, `EluBackward`, `MishBackward`,
`SoftplusBackward`, `GLUBackward`, `PReluBackward`) when grad is
enabled. The activation modules do NOT re-attach backward — they're a
thin pass-through.

### Non-test production consumers

- `ferrotorch-optim/src/sgd.rs:818` — `let relu = ferrotorch_nn::ReLU::new();`
  inside the SGD module-flow conformance harness (production-side build
  of an MLP for verifying optimiser convergence).
- `ferrotorch-rl/src/mlp_policy.rs` —
  `use ferrotorch_nn::activation::Tanh;` for the actor-critic head.
- `ferrotorch-vision/src/models/vgg.rs,25` —
  `use ferrotorch_nn::activation::ReLU; use ferrotorch_nn::{Conv2d,
  Dropout, Linear};` — VGG-16/19 stem.
- `ferrotorch-vision/src/models/mobilenet.rs` —
  `use ferrotorch_nn::activation::{HardSigmoid, HardSwish, ReLU, ReLU6};`.
- `ferrotorch-diffusion/src/vae.rs` —
  `use ferrotorch_nn::{Conv2d, GroupNorm, SiLU};` — Stable Diffusion VAE
  ResnetBlock uses `SiLU` between every conv.
- `ferrotorch-bert/src/layer.rs`, `ferrotorch-whisper/src/encoder.rs`,
  `ferrotorch-whisper/src/layer.rs` —
  `use ferrotorch_nn::{GELU, LayerNorm, Linear};` — BERT / Whisper FFN.
- `ferrotorch-nn/src/lib.rs:189-193` — re-exports every public type, making
  them addressable as `ferrotorch_nn::{ReLU, GELU, SiLU, ...}` for
  downstream crates.

## Parity contract

`parity_ops = []`. The activation modules themselves do not own dedicated
parity-sweep ops; instead they delegate to the differentiable primitives in
`ferrotorch_core::grad_fns::activation`, which are covered under
`ferrotorch-core`'s own parity ops (`relu`, `sigmoid`, `tanh`, `gelu`,
`silu`, `softmax`, `log_softmax`, `leaky_relu`, `elu`, `softplus`,
`mish`, `glu`, `prelu`). Refer to the `.design/ferrotorch-core/grad_fns_activation.md`
audit for those.

Edge-case behaviour preserved:

- **NaN propagation**: every elementwise activation passes NaN through
  unchanged, matching upstream `aten/src/ATen/native/Activation.cpp`.
- **Infinity**: `sigmoid(inf) = 1.0`, `sigmoid(-inf) = 0.0`,
  `tanh(inf) = 1.0`, `gelu(inf) = inf`, `relu(inf) = inf`. Matches
  upstream.
- **Softmax numerical stability**: max-subtraction before `exp`, matching
  `aten/src/ATen/native/SoftMax.cpp`.
- **Softmax2d on CUDA**: returns `NotImplementedOnCuda` rather than
  silently round-tripping through CPU — see blocker #1451.

## Verification

In-file `#[test]` block: 87 tests (count via
`grep -c "^    #\[test\]" activation.rs`). Coverage spans:

- Functional correctness: `test_relu_forward`, `test_sigmoid_forward`,
  `test_tanh_forward`, `test_gelu_forward_default`,
  `test_gelu_forward_tanh_approximate`, `test_silu_forward`,
  `test_softmax_forward`, `test_log_softmax_forward`,
  `test_leaky_relu_forward`, `test_elu_forward`, `test_celu_forward`,
  `test_selu_forward`, `test_hard_sigmoid_forward`,
  `test_hardswish_forward`, `test_softplus_forward`,
  `test_glu_forward`, `test_relu6_forward`, `test_hardtanh_forward`,
  `test_log_sigmoid_forward`, `test_softmin_forward`,
  `test_threshold_forward`, `test_softshrink_forward`,
  `test_hardshrink_forward`, `test_tanhshrink_forward`,
  `test_softsign_forward`, `test_rrelu_forward`, `test_prelu_forward`,
  `test_mish_forward`, `test_softmax2d_forward`.
- Module trait: `test_relu_module_trait`, `test_gelu_module_trait`,
  `test_prelu_module_trait`, `test_prelu_has_parameter`, plus a
  `assert_zero_param_module<T>` helper exercised against every zero-param
  module.
- Train/eval: `test_*_module_trait` covers `train()`/`eval()` toggling.

```bash
cargo test -p ferrotorch-nn --lib activation:: 2>&1 | tail -3
```

Expected: `87 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct ReLU` and `pub fn forward` at the top of `activation.rs`, mirroring `torch/nn/modules/activation.py:104-152`; non-test consumer: `new in ferrotorch-vision/src/models/vgg.rs` (`use ferrotorch_nn::activation::ReLU;`) and `new in ferrotorch-optim/src/sgd.rs` (`let relu = ferrotorch_nn::ReLU::new();`). Test `test_relu_forward` pins. |
| REQ-2 | SHIPPED | impl: `pub struct Sigmoid`, `pub struct Tanh` in `activation.rs`, mirroring `torch/nn/modules/activation.py:337-434`; non-test consumer: `Tanh in ferrotorch-rl/src/mlp_policy.rs` (`use ferrotorch_nn::activation::Tanh;`). Tests `test_sigmoid_forward`, `test_tanh_forward` pin. |
| REQ-3 | SHIPPED | impl: `pub struct GELU` + re-exported `pub use act::GeluApproximate` in `activation.rs`, mirroring `torch/nn/modules/activation.py:777-824`; non-test consumer: `GELU in ferrotorch-bert/src/layer.rs` and `ferrotorch-whisper/src/encoder.rs` use `GELU`. Tests `test_gelu_forward_default`, `test_gelu_forward_tanh_approximate` pin both modes. |
| REQ-4 | SHIPPED | impl: `pub struct SiLU` in `activation.rs`, mirroring `torch/nn/modules/activation.py:435-484`; non-test consumer: `ferrotorch-diffusion/src/vae.rs` (`use ferrotorch_nn::{Conv2d, GroupNorm, SiLU};`). Test `test_silu_forward` pins. |
| REQ-5 | SHIPPED | impl: `pub struct Softmax`, `pub struct LogSoftmax`, `pub struct Softmin` plus `pub struct Softmax2d` in `activation.rs`, mirroring `torch/nn/modules/activation.py:1709-1929`; non-test consumer: `ferrotorch-nn/src/lib.rs:189-193` re-exports each, and `ferrotorch-nn-derive`-generated forward chains in downstream model crates call them. `Softmax2d` is CPU-only (blocker #1451 for GPU). |
| REQ-6 | SHIPPED | impl: `pub struct LeakyReLU`, `pub struct PReLU<T>`, `pub struct ELU`, `pub struct CELU`, `pub struct SELU`, `pub struct RReLU` in `activation.rs`, mirroring `torch/nn/modules/activation.py:153-218, 575-735, 874-931, 1575-1656`; non-test consumer: `ferrotorch-nn/src/lib.rs:189-193` re-exports each. Tests `test_leaky_relu_forward`, `test_prelu_forward`, `test_elu_forward`, `test_celu_forward`, `test_selu_forward`, `test_rrelu_forward` pin. |
| REQ-7 | SHIPPED | impl: `pub struct Hardtanh`, `pub struct ReLU6`, `pub struct HardSigmoid`, `pub struct HardSwish`, `pub struct Hardshrink`, `pub struct Softshrink`, `pub struct Tanhshrink`, `pub struct Softsign`, `pub struct LogSigmoid`, `pub struct Threshold`, `pub struct Softplus`, `pub struct Mish`, `pub struct GLU` in `activation.rs`, mirroring their PyTorch counterparts in `torch/nn/modules/activation.py`; non-test consumer: `ferrotorch-vision/src/models/mobilenet.rs` consumes `HardSigmoid, HardSwish, ReLU, ReLU6`. |
| REQ-8 | SHIPPED | impl: `pub struct PReLU<T: Float>` with `pub alpha: Parameter<T>` and a `named_parameters` impl returning `("alpha", ..)` in `activation.rs`, mirroring `torch/nn/modules/activation.py:1575-1656`; non-test consumer: re-exported via `activation in ferrotorch-nn/src/lib.rs`; test `test_prelu_has_parameter` pins. |
| REQ-9 | SHIPPED | impl: `macro_rules! impl_activation_module` at the top of `activation.rs` synthesises `Module<T>::{forward, parameters, parameters_mut, named_parameters, train, eval, is_training}` for every zero-param activation; PReLU has a hand-written `Module<T>` impl. Non-test consumer: `ferrotorch-vision/src/models/vgg.rs` builds `Module<f32>` chains using these. |
| REQ-10 | SHIPPED | impl: every `pub fn forward` delegates to `act::*` in `ferrotorch_core::grad_fns::activation`, which attaches the appropriate backward node when grad is enabled; non-test consumer: `ferrotorch-optim/src/sgd.rs:818` uses `ReLU` in an end-to-end SGD training loop that exercises backward. |
| REQ-11 | SHIPPED | impl: `Softmax2d::forward` in `activation.rs` dispatches to `GpuBackend::softmax2d_f32` (declared in `ferrotorch-core/src/gpu_dispatch.rs`, overridden by `CudaBackendImpl::softmax2d_f32` in `ferrotorch-gpu/src/backend_impl.rs` calling `crate::group_norm::gpu_softmax2d_f32`) when `input.is_cuda()` and a backend is registered; CUDA-with-no-backend still returns `NotImplementedOnCuda`. Channel-axis softmax over `[N, C, H*W]`, forward-only. Non-test consumer: `ferrotorch-nn::Softmax2d::forward` is the dispatcher itself, reachable from downstream segmentation heads via the `Softmax2d` re-export at `ferrotorch-nn/src/lib.rs`. Runtime parity pinned by `#[ignore]`'d `softmax2d_forward_gpu_matches_cpu`. Closed #1451. |
