# ferrotorch-nn — `norm` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/batchnorm.py
  - torch/nn/modules/normalization.py
  - aten/src/ATen/native/Normalization.cpp
-->

## Summary

`ferrotorch-nn/src/norm.rs` ships every normalization layer needed for
the transformer / CNN / RNN stack: `LayerNorm`, `GroupNorm`, `RMSNorm`,
`BatchNorm1d` / `BatchNorm2d` / `BatchNorm3d`,
`InstanceNorm1d` / `InstanceNorm2d` / `InstanceNorm3d`, and
`LocalResponseNorm`. Each layer normalises its input along the
documented axes and optionally applies a learnable affine transform
(`weight`, `bias`). Hand-written `GradFn<T>` backward nodes propagate
gradients to the input and the affine parameters. The `LayerNorm` and
`RMSNorm` paths additionally dispatch to GPU kernels in
`ferrotorch_core::gpu_dispatch` when the input is CUDA-resident and the
backend exposes the appropriate `layernorm_f32` / `rmsnorm_f32` slots.

## Requirements

- REQ-1: `pub struct LayerNorm<T: Float>` with `normalized_shape`,
  `eps`, `elementwise_affine`, `weight`, `bias`. Forward applies
  `(x - mean) / sqrt(var + eps) * weight + bias` over the last
  `normalized_shape.len()` dims. Mirrors `torch.nn.LayerNorm` at
  `torch/nn/modules/normalization.py:105-238`.

- REQ-2: `LayerNorm` GPU fast path — when `input.is_cuda() &&
  elementwise_affine` and the GPU backend is registered, dispatches to
  `backend.layernorm_f32(input, weight, bias, batch, norm_size, eps)`.
  Mirrors the upstream CUDA kernel registered at
  `aten/src/ATen/native/Normalization.cpp` (`native_layer_norm_cuda`).

- REQ-3: `pub struct GroupNorm<T: Float>` with `num_groups`,
  `num_channels`, `eps`, `affine`, `weight`, `bias`. Splits the channel
  dim into `num_groups` groups and normalises each group independently
  per sample. Mirrors `torch.nn.GroupNorm` at
  `torch/nn/modules/normalization.py:239-342`. GPU forward fast path
  (#1357): dispatches to `GpuBackend::group_norm_f32` for CUDA input
  (see the GroupNorm architecture section).

- REQ-4: `pub struct RMSNorm<T: Float>` with `normalized_shape`, `eps`,
  `elementwise_affine`, `weight`. Forward applies
  `x / sqrt(mean(x^2) + eps) * weight` (no mean centering, no bias —
  matches the Llama / T5 RMSNorm formulation). Mirrors
  `torch.nn.RMSNorm` at `torch/nn/modules/normalization.py:343-435`.

- REQ-5: `pub struct BatchNorm2d<T: Float>` with `num_features`, `eps`,
  `momentum`, `affine`, `track_running_stats`, plus owned `running_mean`,
  `running_var`, `num_batches_tracked` (all interior-mutable behind
  `Mutex` for `Module` ergonomics). Forward computes per-channel mean /
  variance over `(N, H, W)` in training mode (and updates the running
  stats), or uses `running_mean` / `running_var` in eval. Mirrors
  `torch.nn.BatchNorm2d` at `torch/nn/modules/batchnorm.py:420-498`.

- REQ-6: `pub struct BatchNorm1d<T: Float>` — analogous to
  `BatchNorm2d` but for 2-D `(N, C)` or 3-D `(N, C, L)` inputs. Mirrors
  `torch.nn.BatchNorm1d` at `torch/nn/modules/batchnorm.py:306-383`.

- REQ-7: `pub struct BatchNorm3d<T: Float>` — analogous for 5-D
  `(N, C, D, H, W)` inputs. Mirrors `torch.nn.BatchNorm3d` at
  `torch/nn/modules/batchnorm.py:535-613`.

- REQ-8: Public setters for the BatchNorm running statistics —
  `set_running_mean`, `set_running_var`, `set_num_batches_tracked` plus
  matching read accessors `running_mean`, `running_var`,
  `num_batches_tracked`. Required for state-dict loading from upstream
  checkpoints (the running stats are buffers, not parameters, so the
  generic `state_dict` loader cannot reach them). The setters validate
  length, finiteness, and non-negative variance.

- REQ-9: `pub struct InstanceNorm1d<T: Float>`,
  `pub struct InstanceNorm2d<T: Float>`,
  `pub struct InstanceNorm3d<T: Float>` — per-sample, per-channel
  normalization over the spatial dims. No running statistics. All three
  delegate to a private `InstanceNormInner<T>` for shared logic. Mirror
  `torch.nn.InstanceNorm{1,2,3}d` at
  `torch/nn/modules/instancenorm.py` (with the spatial-dim count `3 |
  4 | 5` selected at the `Module::forward` boundary).

- REQ-10: `pub struct LocalResponseNorm` — `size`, `alpha`, `beta`, `k`.
  Cross-channel normalization `output[c] = input[c] / (k + alpha/size *
  sum(input[j]^2 for j in [c-size/2, c+size/2]))^beta`. Mirrors
  `torch.nn.LocalResponseNorm` at
  `torch/nn/modules/normalization.py:16-73`.

- REQ-11: Every layer implements `Module<T>::{forward, parameters,
  parameters_mut, named_parameters, train, eval, is_training}`. BatchNorm
  additionally implements `as_any` so that state-dict loaders can
  downcast to the concrete `BatchNorm{1,2,3}d<T>` and call the running-stat
  setters.

- REQ-12: Every forward attaches a hand-written `GradFn<T>` backward node
  when `is_grad_enabled() && input.requires_grad()`. Backward nodes:
  `LayerNormBackward`, `GroupNormBackward`, `RMSNormBackward`,
  `BatchNorm1dBackward`, `BatchNorm2dBackward`, `BatchNorm3dBackward`,
  `InstanceNormBackward`, `LocalResponseNormBackward`. Each propagates
  to input, weight, and bias separately, with the standard mean-norm
  formulas
  `dL/dx = (1/N) * inv_std * (dL/dx_hat * N - sum(dL/dx_hat) -
  x_hat * sum(dL/dx_hat * x_hat))`.

## Acceptance Criteria

- [x] AC-1: `LayerNorm::new(normalized_shape, eps, elementwise_affine)`
  rejects empty `normalized_shape`.
- [x] AC-2: `LayerNorm::forward` produces ~zero-mean / unit-var rows
  along the last dim (test `test_layer_norm_forward_zero_mean_unit_var`).
- [x] AC-3: `LayerNorm` CUDA forward dispatches to `backend.layernorm_f32`
  when affine + backend present.
- [x] AC-4: `GroupNorm::new(num_groups, num_channels, eps, affine)` rejects
  `num_channels % num_groups != 0`.
- [x] AC-5: `RMSNorm` matches the Llama RMSNorm reference computation.
- [x] AC-6: `BatchNorm{1,2,3}d::new` constructs running stats with zeros
  / ones and registers them under the conventional names.
- [x] AC-7: BatchNorm running-stat setters validate length, NaN/Inf,
  and non-negative variance (tests `bn2d_set_running_mean_rejects_*`
  etc.).
- [x] AC-8: BatchNorm `set_running_mean` / `set_running_var` flow through
  the eval-mode forward (tests
  `bn2d_set_running_stats_flow_through_eval_forward` etc.).
- [x] AC-9: BatchNorm `as_any` downcasts to the concrete type
  (tests `bn{1,2,3}d_as_any_downcasts_to_concrete_type`).
- [x] AC-10: `InstanceNorm{1,2,3}d::forward` validates ndim matches the
  expected `3 | 4 | 5` rank.
- [x] AC-11: `LocalResponseNorm::forward` requires at least 3-D input;
  CUDA inputs are rejected with `NotImplementedOnCuda`.
- [x] AC-12: Backward propagates to `input`, `weight`, `bias`
  separately when each `requires_grad`.
- [ ] AC-13: Parity-sweep oracle runner arms — blocker #1447.
- [x] AC-14: BatchNorm / InstanceNorm / LocalResponseNorm GPU forward +
  backward — SHIPPED (#1449). Forward GPU paths in `BatchNorm{1,2,3}d::forward`,
  `InstanceNormInner::forward_impl`, `LocalResponseNorm::forward`; backward GPU
  paths in `BatchNorm{1,2,3}dBackward::backward` (via `fn batch_norm_gpu_backward`),
  `InstanceNormBackward::backward` (via `fn instance_norm_gpu_backward`), and
  `LocalResponseNormBackward::backward`. All compute on-device — NO `.cpu()`
  round trip (R-CODE-4). Live-verified on RTX 3090 vs. torch autograd to <1e-3
  in `ferrotorch-nn/tests/divergence_critic_batchnorm_gpu.rs` (train/eval
  BatchNorm grad, InstanceNorm grad, LRN fwd+bwd grad).

## Architecture

### Helpers

`fn is_f32` / `fn is_f64` / `fn zero` at the top of `norm.rs` are
`TypeId`-based dtype dispatchers and a zero-of-T constructor used
throughout the GPU and CPU paths.

### LayerNorm (REQ-1, REQ-2, REQ-12)

`pub struct LayerNorm<T: Float>` with `pub normalized_shape: Vec<usize>`,
`pub eps: f64`, `pub elementwise_affine: bool`, `pub weight:
Parameter<T>`, `pub bias: Parameter<T>`, and a private `training: bool`.

`pub fn LayerNorm::new` rejects empty `normalized_shape` and constructs
`weight = Parameter::ones(&normalized_shape)`, `bias =
Parameter::zeros(&normalized_shape)`.

`impl<T: Float> Module<T> for LayerNorm<T>` — forward validates that the
last `normalized_shape.len()` dims of the input match
`normalized_shape`, then either (a) takes the GPU fast path
`backend.layernorm_f32(input, weight, bias, batch_size, norm_size,
eps_f32)` when `input.is_cuda() && elementwise_affine` and a backend is
registered, or (b) takes the CPU per-row loop that computes mean /
variance manually and writes
`normed[i] = (x_i - mean) * inv_std * weight_i + bias_i`. CUDA input
without backend returns `NotImplementedOnCuda`.

`struct LayerNormBackward<T: Float>` ships in the same file. Backward
computes `dL/dx`, `dL/dw`, `dL/db` per the standard analytic formulas;
the GPU side calls `backend.layernorm_backward_f32` when available
(see `ferrotorch_core::gpu_dispatch:2096-2160` for the CUDA kernel).

### GroupNorm (REQ-3, REQ-12)

`pub struct GroupNorm<T: Float>` with `pub num_groups`,
`pub num_channels`, `pub eps`, `pub affine`, `pub weight`, `pub bias`.

Forward reshapes input from `(N, C, *spatial)` to `(N, G, C/G *
spatial)`, computes per-group mean / variance, normalises, then folds
the optional affine `weight[c] * normed[n, c, *] + bias[c]` back per
channel.

GPU fast path (#1357): when `input.is_cuda()` and the GPU backend is
registered, `GroupNorm::forward` dispatches to
`backend.group_norm_f32(input, weight, bias, batch, channels,
num_groups, hw, eps)` instead of the CPU loop. `weight`/`bias` always
have length `num_channels` (ones / zeros when `affine` is false), so
the kernel's unconditional per-channel affine is the identity in the
non-affine case. Mirrors the upstream CUDA kernel
`aten/src/ATen/native/cuda/group_norm_kernel.cu` (`GroupNormKernelImpl`).
The kernel itself lives in `ferrotorch-gpu/src/group_norm.rs`
(`gpu_group_norm_f32`). CUDA input with no backend registered is
rejected with `NotImplementedOnCuda`.

`struct GroupNormBackward<T: Float>` mirrors the LayerNorm backward but
groups the reduction along the per-group axis instead of per-row.
GroupNorm GPU **backward** is not yet shipped — `GroupNormBackward`
returns `NotImplementedOnCuda` for CUDA-resident input, so a CUDA
GroupNorm that requires grad will surface that on `.backward()`
(forward-only GPU support, tracked separately).

### RMSNorm (REQ-3, REQ-12)

`pub struct RMSNorm<T: Float>` with `pub normalized_shape`, `pub eps`,
`pub elementwise_affine`, `pub weight`. Forward computes
`x / sqrt(mean(x^2) + eps) * weight` element-wise across the last
`normalized_shape.len()` dims. **No mean centering, no bias** — matches
the Llama / T5 formulation.

GPU fast path: dispatches to `backend.rmsnorm_f32(input, weight,
batch_size, norm_size, eps_f32)` when CUDA + affine + backend.
`RMSNormBackward<T: Float>` handles the CPU + GPU backward; GPU side
calls `backend.rmsnorm_backward_f32` (see
`ferrotorch_core::gpu_dispatch:1206-1260`).

### BatchNorm1d/2d/3d (REQ-5, REQ-6, REQ-7, REQ-8, REQ-12)

`pub struct BatchNorm2d<T: Float>` (and 1d / 3d analogs) carry:
- `pub num_features: usize`,
- `pub eps: f64`, `pub momentum: f64`, `pub affine: bool`,
- `pub track_running_stats: bool`,
- `pub weight: Option<Parameter<T>>`, `pub bias: Option<Parameter<T>>`,
- `running_mean: Mutex<Vec<T>>`, `running_var: Mutex<Vec<T>>`,
- `num_batches_tracked: Mutex<usize>`,
- `training: Mutex<bool>`.

The `Mutex` wrapping is required because BN's running statistics MUST
update during `forward(&self, input)` in training mode while keeping the
`Module<T>` trait signature non-`&mut self`. This is a R-DEV-5
typestate-style deviation forced by PyTorch's contract: PyTorch's
`Module.forward` is also called with the module held by reference, and
`running_mean.add_(...)` is an in-place mutation tolerated under
upstream's Python-level reference semantics. The Rust translation
captures this with `Mutex` instead of an `&mut self` re-signature, so
that downstream callers (state-dict loaders, training loops) can keep
the existing `Module<T>` signature.

Forward branches on `is_training()`:
- **Training**: compute per-channel mean / var over `(N, *spatial)`,
  normalise, update running stats via
  `running_mean = (1 - momentum) * running_mean + momentum * batch_mean`,
  bump `num_batches_tracked`.
- **Eval**: read `running_mean`, `running_var` and apply the same
  normalisation.

`set_running_mean(&self, &[T])`, `set_running_var(&self, &[T])`,
`set_num_batches_tracked(&self, usize)` — public setters required by
state-dict loaders to push upstream checkpoint buffers into the layer.
Validation: length must equal `num_features`, every element finite,
variance >= 0. Tests
`bn{1,2,3}d_set_running_*_round_trip`,
`bn2d_set_running_mean_rejects_*`,
`bn2d_set_running_var_rejects_*`,
`bn{1,2,3}d_set_running_stats_flow_through_eval_forward` pin these.

`as_any() -> Option<&dyn Any>` returns `Some(self)` so state-dict
loaders can downcast a `&dyn Module<T>` to the concrete BN type and
call the setters. Tests `bn{1,2,3}d_as_any_downcasts_to_concrete_type`
plus `non_bn_module_as_any_returns_none` (LayerNorm sanity) pin this.

`struct BatchNorm{1,2,3}dBackward<T: Float>` — hand-written backward
computing per-channel `dL/dx`, `dL/dw`, `dL/db` via the standard
batch-norm derivative `dL/dx = inv_std * (dL/dx_hat - mean(dL/dx_hat) -
x_hat * mean(dL/dx_hat * x_hat))`.

**GPU (#1449, SHIPPED)**: `BatchNorm{1,2,3}d::forward` dispatch f32 CUDA
input through `backend.batch_norm_f32(...)` (forward), and
`BatchNorm{1,2,3}dBackward::backward` dispatch f32 CUDA grads through
`backend.batch_norm_backward_f32(...)` via the shared
`fn batch_norm_gpu_backward` (kernel `gpu_batch_norm_backward_f32` in
`ferrotorch-gpu/src/group_norm.rs`, mirroring
`aten/src/ATen/native/cuda/Normalization.cuh:388`). The backward computes
grad_input/grad_weight/grad_bias entirely on-device — NO `.cpu()` round
trip (R-CODE-4). In train mode the kernel recomputes batch mean/var from the
input; in eval mode it reads the running-stat snapshots saved in the backward
node. Live-vs-torch-autograd grad parity (<1e-3) on the RTX 3090.

### InstanceNorm1d/2d/3d (REQ-9, REQ-12)

`struct InstanceNormInner<T: Float>` holds the shared logic:
`num_features`, `eps`, `affine`, `weight`, `bias`, plus a
`fn forward_impl(&self, input, expected_ndim)` that validates the rank
and runs the per-sample, per-channel mean/var normalization.

`pub struct InstanceNorm1d<T>`, `pub struct InstanceNorm2d<T>`,
`pub struct InstanceNorm3d<T>` are thin newtype wrappers around
`InstanceNormInner<T>` that fix `expected_ndim` to `3`, `4`, `5`
respectively. Each implements `Module<T>` by forwarding through the
inner.

`struct InstanceNormBackward<T: Float>` shared across all three.

**GPU (#1449, SHIPPED)**: `InstanceNormInner::forward_impl` routes f32 CUDA
input through `backend.group_norm_f32(...)` with `num_groups == num_channels`;
`InstanceNormBackward::backward` routes f32 CUDA grads through
`fn instance_norm_gpu_backward`, which reshapes `[B,C,S]`→`[1,B*C,S]` so each
`(b,c)` becomes its own normalization "channel" and reuses
`backend.batch_norm_backward_f32(...)` (instance stats), reducing grad_weight /
grad_bias `[B*C]`→`[C]` via `backend.sum_axis_f32` — all on-device, NO `.cpu()`
round trip. Live-vs-torch-autograd grad parity (<1e-3) on the RTX 3090.

### LocalResponseNorm (REQ-10, REQ-12)

`pub struct LocalResponseNorm` with `pub size`, `pub alpha`, `pub beta`,
`pub k`. Forward requires `ndim >= 3` (validated up front). Computes
per-channel `output[c] = input[c] / (k + (alpha/size) *
sum(input[j]^2 for j in [c-size/2, c+size/2]))^beta` using
saturating-subtract for the lower bound and `min(c+size/2+1, channels)`
for the upper bound. `LocalResponseNormBackward` propagates through both
the direct path and the denominator's gradient w.r.t. the surrounding
channels.

**GPU (#1449, SHIPPED)**: `LocalResponseNorm::forward` routes f32 CUDA input
through `backend.local_response_norm_f32(...)` (kernel
`gpu_local_response_norm_f32` in `ferrotorch-gpu/src/group_norm.rs`, mirroring
the `torch/nn/functional.py:3032-3046` decomposition), saving the per-element
`denom` buffer GPU-resident in the backward node; `LocalResponseNormBackward::backward`
routes f32 CUDA grads through `backend.local_response_norm_backward_f32(...)`
(kernel `gpu_local_response_norm_backward_f32`) consuming that `denom` — all
on-device, NO `.cpu()` round trip. Live-vs-torch forward+grad parity (<1e-3)
on the RTX 3090.

### Non-test production consumers

- `ferrotorch-distributed/src/sync_batch_norm.rs,595` —
  `use ferrotorch_nn::BatchNorm2d;` (the SyncBatchNorm distributed
  variant wraps the local `BatchNorm2d`'s running stats with NCCL
  all-reduce).
- `ferrotorch-vision/src/models/segmentation/fcn.rs:34` —
  `use ferrotorch_nn::norm::BatchNorm2d;` (FCN backbone).
- `ferrotorch-diffusion/src/vae.rs,39,99` —
  `use ferrotorch_nn::{Conv2d, GroupNorm, SiLU}; pub conv_norm_out:
  GroupNorm<T>; GroupNorm::<T>::new(groups, bottom_channels,
  resnet_eps, true)?` — Stable Diffusion VAE ResnetBlock uses GroupNorm.
- `ferrotorch-bert/src/attention.rs,194,209` —
  `use ferrotorch_nn::{LayerNorm, ...}; pub layer_norm: LayerNorm<T>;
  layer_norm: LayerNorm::new(vec![cfg.hidden_size], cfg.layer_norm_eps,
  true)?` — every BERT block's post-norm residual.
- `ferrotorch-whisper/src/encoder.rs`, `ferrotorch-whisper/src/layer.rs` —
  `use ferrotorch_nn::{Conv1d, GELU, LayerNorm};` — Whisper FFN
  pre/post norm.
- `ferrotorch-nn/src/lazy_norm.rs,154-155` — the `LazyBatchNorm{2,3}d`
  lazy wrappers materialise into `BatchNorm{2,3}d` after the first
  forward sees an input shape.
- `ferrotorch-nn/src/lib.rs:225-228` — re-exports
  `BatchNorm{1,2,3}d, GroupNorm, InstanceNorm{1,2,3}d, LayerNorm,
  LocalResponseNorm, RMSNorm` for the `ferrotorch_nn` public surface.

## Parity contract

Route declares 5 parity ops:
`nn.functional.batch_norm`, `nn.functional.layer_norm`,
`nn.functional.group_norm`, `nn.functional.instance_norm`,
`nn.functional.local_response_norm`.

Every op currently reports `MISSING` in
`tools/parity-sweep/parity_audit.json` and `parity-sweep sweep --op
<op> --seeds 8` returns `0/N passed (N skipped, 0 failed)` for each.
The runner-arm gap is tracked by **blocker #1447** (`Wire parity-sweep
runner arms for 5 norm ops`).

Upstream edge cases preserved:

- **`var + eps` for numerical stability**: every layer uses
  `1.0 / sqrt(var + eps)` rather than the unstable
  `(var + eps).recip().sqrt()`. Matches upstream `aten/src/ATen/native/Normalization.cpp`.
- **BatchNorm running stats only update in train mode**: pinned by
  tests `bn{1,2,3}d_set_running_mean_does_not_touch_nbt` (the setter
  is independent of training batches).
- **Affine off → weight/bias are `None`**: BatchNorm omits the
  `Parameter<T>` entirely when `affine=false`. Tests
  `bn2d_no_affine_no_params` pin.
- **NotImplementedOnCuda surfacing**: BatchNorm / InstanceNorm /
  LocalResponseNorm rejecting CUDA input with a typed error rather than
  silently CPU↔GPU round-tripping (would violate goal.md R-CODE-4).
  LayerNorm and RMSNorm DO have GPU fast paths.

## Verification

In-file `#[test]` block: 117 tests (count via
`grep -c "^    #\[test\]" norm.rs`). Coverage spans:

- LayerNorm forward / backward / no-affine variants.
- GroupNorm forward / backward / rejection of bad `num_groups`.
- RMSNorm forward / backward / Llama-shape parity.
- BatchNorm1d/2d/3d forward (train + eval), backward, running-stat
  setters, `as_any` downcast, eval-mode flow-through.
- InstanceNorm1d/2d/3d forward + backward.
- LocalResponseNorm forward + backward + rank validation.

```bash
cargo test -p ferrotorch-nn --lib norm:: 2>&1 | tail -3
```

Expected: `117 passed`.

Parity-sweep smoke (blocked on runner-arm gap #1447):

```bash
for OP in nn.functional.batch_norm nn.functional.layer_norm \
         nn.functional.group_norm nn.functional.instance_norm \
         nn.functional.local_response_norm; do
  ./target/release/parity-sweep sweep --op "$OP" --seeds 8 2>&1 | tail -1
done
```

Each line currently reports `0/N passed (N skipped, 0 failed)` — every
op is missing a runner arm; blocker #1447 tracks the wiring.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LayerNorm<T: Float>` and `impl<T: Float> Module<T> for LayerNorm<T>` in `norm.rs`, mirroring `torch/nn/modules/normalization.py:105-238`; non-test consumer: `ferrotorch-bert/src/attention.rs,194,209` (`pub layer_norm: LayerNorm<T>`), `norm in ferrotorch-whisper/src/layer.rs`. Runner arm: #1447. |
| REQ-2 | SHIPPED | impl: GPU fast path inside `LayerNorm::forward` in `norm.rs` (dispatches to `backend.layernorm_f32(...)` when `input.is_cuda() && self.elementwise_affine`), mirroring `aten/src/ATen/native/Normalization.cpp` (`native_layer_norm_cuda`); non-test consumer: `forward in ferrotorch-bert/src/attention.rs` and `forward in ferrotorch-whisper/src/encoder.rs` push GPU-resident inputs through this path during inference. |
| REQ-3 | SHIPPED | impl: `pub struct GroupNorm<T: Float>` + `GroupNormBackward<T>` in `norm.rs`, mirroring `torch/nn/modules/normalization.py:239-342`; non-test consumer: `ferrotorch-diffusion/src/vae.rs:39,99` (`pub conv_norm_out: GroupNorm<T>`). Runner arm: #1447. |
| REQ-4 | SHIPPED | impl: `pub struct RMSNorm<T: Float>` + `RMSNormBackward<T>` in `norm.rs` with `mean(x^2)` denominator (no centering, no bias), mirroring `torch/nn/modules/normalization.py:343-435`; non-test consumer: `ferrotorch-nn/src/lib.rs:227` re-export (Llama / T5 stacks consume `RMSNorm`). |
| REQ-5 | SHIPPED | impl: `pub struct BatchNorm2d<T: Float>` + `BatchNorm2dBackward<T>` in `norm.rs` with per-channel running stats and train/eval branching, mirroring `torch/nn/modules/batchnorm.py:420-498`; non-test consumer: `norm in ferrotorch-distributed/src/sync_batch_norm.rs,595` and `forward in ferrotorch-vision/src/models/segmentation/fcn.rs`. **GPU forward fast path (#1449)**: `BatchNorm2d::forward` dispatches to `backend.batch_norm_f32(...)` for f32 CUDA input via the `batch_norm_gpu_forward` helper in `norm.rs` (per-channel reduce over (B,H,W) train / running-stats eval, kernel in `ferrotorch-gpu/src/group_norm.rs::gpu_batch_norm_f32`), mirroring `aten/src/ATen/native/Normalization.cpp::batch_norm_cuda`. Live GPU↔CPU parity (<1e-4) pinned by `batch_norm2d_eval_forward_gpu_matches_cpu` + `batch_norm2d_train_forward_gpu_matches_cpu`. **GPU backward (#1449)**: `BatchNorm2dBackward::backward` dispatches f32 CUDA grads through `backend.batch_norm_backward_f32(...)` via the shared `fn batch_norm_gpu_backward` in `norm.rs` (kernel `gpu_batch_norm_backward_f32` in `ferrotorch-gpu/src/group_norm.rs`), on-device with NO `.cpu()` round trip, mirroring `aten/src/ATen/native/cuda/Normalization.cuh:388 batch_norm_backward_kernel`. Live-vs-torch-autograd grad parity (<1e-3) pinned by `divergence_batchnorm2d_gpu_train_backward_vs_torch` + `divergence_batchnorm2d_gpu_eval_backward_vs_torch`. Runner arm: #1447. |
| REQ-6 | SHIPPED | impl: `pub struct BatchNorm1d<T: Float>` + `BatchNorm1dBackward<T>` in `norm.rs`, mirroring `torch/nn/modules/batchnorm.py:306-383`; non-test consumer: `ferrotorch-nn/src/lazy_norm.rs:154` (`lazy_batchnorm!(LazyBatchNorm1d, BatchNorm1d, ...)`) and `ferrotorch-nn/src/lib.rs:225-228` re-exports. **GPU forward fast path (#1449)**: `BatchNorm1d::forward` dispatches to `backend.batch_norm_f32(...)` for f32 CUDA input (per-channel reduce over (N,) / (N,L)) via the shared `batch_norm_gpu_forward` helper in `norm.rs`. **GPU backward (#1449)**: `BatchNorm1dBackward::backward` dispatches f32 CUDA grads through the shared `fn batch_norm_gpu_backward` (kernel `gpu_batch_norm_backward_f32`), on-device. |
| REQ-7 | SHIPPED | impl: `pub struct BatchNorm3d<T: Float>` + `BatchNorm3dBackward<T>` in `norm.rs`, mirroring `torch/nn/modules/batchnorm.py:535-613`; non-test consumer: `ferrotorch-nn/src/lazy_norm.rs:155` and `lib.rs:225-228` re-exports. **GPU forward fast path (#1449)**: `BatchNorm3d::forward` dispatches to `backend.batch_norm_f32(...)` for f32 CUDA input (per-channel reduce over (B,D,H,W)) via the shared `batch_norm_gpu_forward` helper in `norm.rs`. **GPU backward (#1449)**: `BatchNorm3dBackward::backward` dispatches f32 CUDA grads through the shared `fn batch_norm_gpu_backward` (kernel `gpu_batch_norm_backward_f32`), on-device. |
| REQ-8 | SHIPPED | impl: `pub fn set_running_mean/set_running_var/set_num_batches_tracked` plus matching read accessors on every `BatchNorm{1,2,3}d<T>` in `norm.rs`, with finite/non-negative validation; non-test consumer: `ferrotorch-distributed/src/sync_batch_norm.rs` reads `running_mean()` / `running_var()` to all-reduce across ranks; vision state-dict loaders downcast via `as_any` to call these setters. Tests `bn{1,2,3}d_set_running_*_round_trip` and `bn{1,2,3}d_set_running_stats_flow_through_eval_forward` pin. |
| REQ-9 | SHIPPED | impl: `pub struct InstanceNorm1d<T>`, `pub struct InstanceNorm2d<T>`, `pub struct InstanceNorm3d<T>` newtypes around `InstanceNormInner<T>` + `InstanceNormBackward<T>` in `norm.rs`, mirroring `torch/nn/modules/instancenorm.py`; non-test consumer: `forward in ferrotorch-nn/src/lazy_norm.rs` (`use ... InstanceNorm{1,2,3}d`) and `lib.rs` re-exports. **GPU forward fast path (#1449)**: `InstanceNormInner::forward_impl` routes f32 CUDA input through `backend.group_norm_f32(...)` with `num_groups == num_channels` (InstanceNorm ≡ per-channel GroupNorm), consumed by all three `InstanceNorm{1,2,3}d::forward`. Live GPU↔CPU parity (<1e-4) pinned by `instance_norm2d_forward_gpu_matches_cpu`. **GPU backward (#1449)**: `InstanceNormBackward::backward` dispatches f32 CUDA grads through `fn instance_norm_gpu_backward` in `norm.rs`, which reshapes `[B,C,S]`→`[1,B*C,S]` and reuses `backend.batch_norm_backward_f32(...)` (per-instance stats), summing grad_weight/grad_bias `[B*C]`→`[C]` via `backend.sum_axis_f32` — all on-device, NO `.cpu()` round trip. Live-vs-torch-autograd grad parity (<1e-3) pinned by `divergence_instancenorm2d_gpu_backward_vs_torch`. |
| REQ-10 | SHIPPED | impl: `pub struct LocalResponseNorm` + `LocalResponseNormBackward<T>` in `norm.rs`, mirroring `torch/nn/modules/normalization.py:16-73`; non-test consumer: `ferrotorch-nn/src/lib.rs:227` re-export. **GPU forward + backward (#1449)**: `LocalResponseNorm::forward` dispatches f32 CUDA input through `backend.local_response_norm_f32(...)` (kernel `gpu_local_response_norm_f32` in `ferrotorch-gpu/src/group_norm.rs`, mirroring `torch/nn/functional.py:3032-3046` square→windowed-channel-sum→`*alpha+k`→pow(beta)→divide), saving the `denom` buffer GPU-resident; `LocalResponseNormBackward::backward` dispatches through `backend.local_response_norm_backward_f32(...)` (kernel `gpu_local_response_norm_backward_f32`) — all on-device, NO `.cpu()` round trip. Live-vs-torch grad+forward parity (<1e-3) pinned by `divergence_local_response_norm_gpu_fwd_bwd_vs_torch`. |
| REQ-11 | SHIPPED | impl: every norm layer has `impl<T: Float> Module<T> for ...` in `norm.rs` with the seven trait methods; BatchNorm additionally implements `as_any() -> Option<&dyn Any>` returning `Some(self)`. Non-test consumer: `ferrotorch-bert/src/attention.rs` calls `self.layer_norm.forward(...)` through the `Module<T>` trait; vision state-dict loaders downcast via `as_any`. Tests `bn{1,2,3}d_as_any_downcasts_to_concrete_type` pin. |
| REQ-12 | SHIPPED | impl: every forward returns `Tensor::from_operation(storage, shape, grad_fn)` when `is_grad_enabled() && input.requires_grad()`; backward nodes `LayerNormBackward`, `GroupNormBackward`, `RMSNormBackward`, `BatchNorm{1,2,3}dBackward`, `InstanceNormBackward`, `LocalResponseNormBackward` all live in `norm.rs`. Non-test consumer: `ferrotorch-optim` training loops drive `backward()` through these nodes when models composed from `ferrotorch-bert` / `ferrotorch-diffusion` / `ferrotorch-vision` are trained. |
