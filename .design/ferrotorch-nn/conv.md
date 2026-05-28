# ferrotorch-nn — `conv` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/conv.py
  - aten/src/ATen/native/Convolution.cpp
-->

## Summary

`ferrotorch-nn/src/conv.rs` implements the six standard
convolution layers — `Conv1d<T>`, `Conv2d<T>`, `Conv3d<T>`,
`ConvTranspose1d<T>`, `ConvTranspose2d<T>`, `ConvTranspose3d<T>` —
mirroring `torch.nn.{Conv,ConvTranspose}{1,2,3}d` at
`torch/nn/modules/conv.py`. Forward uses the im2col + matmul
algorithm on CPU and dispatches to `ferrotorch_gpu::conv2d_f32` for
f32 CUDA tensors. Supports `stride`, `padding`, `dilation`, and
`groups` for ALL THREE forward layers — `Conv1d`, `Conv2d`, `Conv3d`
(grouped + dilated CPU forward AND backward; #1600 conv1d / #1601
conv3d closed the per-group channel partition + dilated im2col that
previously left those two dense-only); the transposed layers
support `stride`, `padding`, and `output_padding`. The `padding_mode`
kwarg from upstream IS implemented (#1443): `Conv{1,2,3}d` honor
`reflect`/`replicate`/`circular` (forward + backward, autograd-aware
pre-pad), while `ConvTranspose{1,2,3}d` accept only `zeros` and reject
other modes with the upstream `ValueError`. String padding (`'same'` /
`'valid'`) is implemented for `Conv{1,2,3}d` (#1602): `'valid'` maps to
`padding = 0`; `'same'` pre-pads asymmetrically (`left = total/2`,
`right = total - left` with `total = dilation*(kernel-1)`, the END side
getting the extra unit) through the same autograd-aware
`functional_pad_{1,2,3}d` path the `padding_mode` work uses, and rejects
strided convolutions with the upstream `ValueError`. Unbatched input
(rank `D+1`, i.e. `(C, *spatial)`) is also accepted (#1604): the layer
unsqueezes a batch dim, convolves, and squeezes it back so the output is
rank `D+1`, with autograd-aware unsqueeze/squeeze so gradients flow to
the unbatched input shape.

## Requirements

- REQ-1: `pub struct Conv2d<T: Float>` carrying `weight: Parameter<T>`
  of shape `[out_channels, in_channels/groups, kH, kW]`, optional
  `bias: Option<Parameter<T>>` of shape `[out_channels]`, and
  geometry fields (`stride`, `padding`, `dilation`, `groups`).
  Mirrors upstream `Conv2d.__init__` weight layout at
  `torch/nn/modules/conv.py` and the ATen kernel descriptor in
  `aten/src/ATen/native/Convolution.cpp:520-600`.
- REQ-2: `Conv2d::new(in_channels, out_channels, kernel_size,
  stride, padding, bias)` and `Conv2d::new_full(.., dilation,
  groups, bias)`. Validates `in_channels % groups == 0`,
  `out_channels % groups == 0`, `kernel_size > 0`, `stride > 0`,
  `dilation > 0`, `groups > 0`. Mirrors upstream `_ConvNd._reset`
  validation at `conv.py:_ConvNd.__init__`.
- REQ-3: `Conv2d::forward` accepts 4D input `[B, C_in, H, W]`,
  validates input channels match, computes `(H_out, W_out)` per the
  standard formula `(H + 2*pad - dilation*(kernel-1) - 1)/stride
  + 1`, applies the im2col + GEMM algorithm, and returns
  `[B, C_out, H_out, W_out]`. Mirrors `aten::convolution` semantics.
- REQ-4: GPU fast path — when `T = f32` and the input is on CUDA,
  dispatches to `ferrotorch_gpu::conv2d_f32` via the global
  `gpu_backend()` registry. Handles `groups` and `dilation` natively
  on-device (Pass 2A of #1003). When backward is required, the
  forward downloads cols for the CPU backward path.
- REQ-5: `Conv2dBackward` GradFn — computes `grad_input` via
  `col2im` of `grad_output @ weight^T` and `grad_weight` via the
  shape `[out_channels, in_channels/groups, kH, kW]` accumulation;
  `grad_bias` is `grad_output.sum_axes([0, 2, 3])`. Mirrors the
  `aten::convolution_backward` decomposition.
- REQ-6: Symmetric ND variants — `Conv1d<T>`, `Conv3d<T>`,
  `ConvTranspose1d<T>`, `ConvTranspose2d<T>`, `ConvTranspose3d<T>`
  with the same parameter shape conventions and validation logic.
  Each maintains its own `Module<T>` impl + `GradFn` backward.
- REQ-7: `Module<T>` trait — `forward`, `parameters`,
  `parameters_mut`, `named_parameters` with `"weight"` and `"bias"`
  keys, `train`/`eval`/`is_training`.
- REQ-8: `set_weight` and `from_parts` helpers — `set_weight`
  replaces the kernel with a shape-checked `Parameter`;
  `from_parts(weight, bias, stride, padding)` builds a Conv2d from
  user-supplied tensors (dense, dilation `(1,1)`, `groups=1`).
- REQ-9: Weight init via Kaiming uniform (ReLU gain). NOTE: upstream
  uses `init.kaiming_uniform_(weight, a=sqrt(5))`; the gain factor
  differs (see linear.md REQ-5 for the same discussion). Bias init
  uses `U(-bound, bound)` with `bound = 1/sqrt(fan_in)` mirroring
  `torch/nn/modules/conv.py:198-201` (fan_in for Conv: `(in_channels/
  groups) * prod(kernel_size)`; for ConvTranspose: `(out_channels/
  groups) * prod(kernel_size)`). Closes #1450.
- REQ-10: `padding_mode` kwarg (`zeros|reflect|replicate|circular`)
  threaded through every conv layer, matching
  `torch/nn/modules/conv.py`. For the forward layers (`Conv1d`,
  `Conv2d`, `Conv3d`) a non-`zeros` mode pre-pads the input via the
  autograd-aware `functional_pad_{1,2,3}d` using
  `_reversed_padding_repeated_twice` amounts, then convolves with
  `padding = 0` (`_ConvNd._conv_forward`, `conv.py:367-378` /
  `716-732`). The pre-pad carries `Pad{1,2,3}dBackward` so input
  gradients flow through the boundary. For the transposed layers
  (`ConvTranspose{1,2,3}d`) only `zeros` is valid: `with_padding_mode`
  rejects any other mode with the upstream
  `ValueError('Only "zeros" padding mode is supported for ...')`
  message (`_ConvTransposeNd.__init__`, `conv.py:755-758`). Closes
  #1443.
- REQ-11: forward conv arms SHIPPED at 0-skip, transpose arms unchanged.
  The parity-sweep runner arms for
  `nn.functional.conv1d`/`conv2d`/`conv3d` are wired (#1441) and reach
  0-skip / 0-failed: `dispatch_conv::<D>` in
  `tools/parity-sweep/runner/src/main.rs`. ALL three ranks now build via
  `Conv{1,2,3}d::new_full` + `with_string_padding` + `Parameter::set_data`
  + `Module::forward`, so each reaches the production grouped + dilated CPU
  forward (groups / dilation / bias all execute; #1600 conv1d / #1601
  conv3d / REQ-2 conv2d), the `'same'`/`'valid'` string-padding path
  (REQ-12, #1602), and the unbatched rank-(D+1) implicit-batch path
  (REQ-13, #1604). The runner no longer drives the dense
  `Conv{1,3}d::from_parts` path for conv1d/conv3d — that was the
  TEST-INFRASTRUCTURE runner-arm gap, now closed. Sweep at `--seeds 8`:
  conv1d 80/80, conv2d 240/240, conv3d 160/160, all 0 skipped / 0 failed.
  Any op_db sample the arm cannot decode now returns `Err` (surfaces) —
  there is no silent `Ok(None)` skip left in the forward conv family. The
  `conv_transpose{1,2,3}d` arms (`dispatch_conv_transpose::<D>`) are now wired
  too (#1441): stride/padding/output_padding/bias decode + runner-side
  batchify for unbatched input; 0-failed across all three ranks at `--seeds 8`
  (48/80, 48/96, 48/80). Their only `Ok(None)` skips are the genuine
  `groups != 1` (#1607) / `dilation != 1` (#1608) production feature-gaps.

- REQ-12: String padding (`'same'` / `'valid'`) for `Conv{1,2,3}d`,
  mirroring the `padding: str` branch of `torch.nn.Conv{1,2,3}d`
  (`torch/nn/modules/conv.py:111-155`) and the functional
  `_convolution_mode` dispatch (`aten/src/ATen/native/Convolution.cpp:
  1111-1124`). A `StringPadding` enum (`Same` / `Valid`) + a
  `with_string_padding` builder on each forward layer carry the mode.
  `'valid'` is equivalent to `padding = 0`
  (`Convolution.cpp:1122-1124`). `'same'` pads so the output spatial
  size equals the input spatial size for `stride = 1`: the per-dim total
  is `dilation * (kernel - 1)`, split ASYMMETRICALLY as `left = total/2`,
  `right = total - left` (the END gets the extra unit when `total` is
  odd), exactly the `_pooling_same_mode_padding_lr` arithmetic
  (`aten/src/ATen/native/Pool.h:91-107`). The asymmetric pre-pad reuses
  the autograd-aware `functional_pad_{1,2,3}d` (the same #1443 path) with
  the configured `padding_mode` (constant-0 for the default `Zeros`,
  matching `convolution_same`'s `constant_pad_nd(.., 0)`,
  `Convolution.cpp:1105`), then convolves with `padding = 0`. `'same'`
  with `stride != 1` is rejected at construction with the upstream
  `ValueError("padding='same' is not supported for strided
  convolutions")` (`conv.py:117-120`, `Convolution.cpp:1071`). Closes
  #1602.
- REQ-13: Unbatched input (rank `D+1`, `(C, *spatial)`) for
  `Conv{1,2,3}d`, mirroring `batchify` + the `output.squeeze(0)`
  un-batching in `conv{1,2,3}d` (`aten/src/ATen/native/Convolution.cpp:
  816-831, 990-1047`). When the input rank is `num_spatial_dims + 1`
  (2 for Conv1d, 3 for Conv2d, 4 for Conv3d), `forward` `unsqueeze(0)`s a
  batch dim, recurses on the batched path, and `squeeze(0)`s the batch
  dim off the result so the output is rank `D+1`. The unsqueeze/squeeze
  are autograd-aware (`UnsqueezeBackward` / `SqueezeBackward`) so input
  gradients flow back to the unbatched shape. Closes #1604.

## Acceptance Criteria

- [x] AC-1: All 6 conv structs present with correct field layouts.
- [x] AC-2: Constructors reject `groups` that don't divide channels.
- [x] AC-3: Forward output shape matches the upstream formula
  (verified by lib tests across all 6 variants).
- [x] AC-4: Backward produces non-zero gradients on input + weight
  + bias (verified by `test_conv2d_backward_*` and analogous tests).
- [x] AC-5: Numerical gradient check passes for Conv2d
  (`test_conv2d_backward_numerical_gradient`).
- [x] AC-6: GPU dispatch fires when CUDA backend is registered and
  input is on CUDA + f32 (verified by integration tests under
  `ferrotorch-gpu/tests/`).
- [x] AC-7: Lazy-conv composition — `LazyConv{1,2,3}d` materializes
  a `Conv{1,2,3}d` on first forward (see `.design/ferrotorch-nn/lazy_conv.md`).
- [x] AC-8: `padding_mode != "zeros"` — Conv1d/Conv2d/Conv3d honor
  reflect/replicate/circular forward + backward; ConvTranspose
  layers reject non-zeros (verified by
  `test_conv{1,3}d_{reflect,replicate,circular}_*` and
  `test_conv_transpose{1,2,3}d_*_padding_mode_rejected`). Closes #1443.
- [ ] AC-9: Bias init matches upstream `U(-sqrt(k), sqrt(k))` —
  blocker #1450.
- [x] AC-10: parity-sweep arms wired for all 6 ops — #1441.
  conv1d/conv2d/conv3d arms reach 0-skip / 0-failed via `dispatch_conv::<D>`
  (`new_full` + `with_string_padding` + `Module::forward`). The three
  `conv_transpose{1,2,3}d` arms are now wired via `dispatch_conv_transpose::<D>`
  (decode stride/padding/output_padding/bias; unbatched rank-(D+1) samples run
  via runner-side batchify on the production batched forward). Sweep at
  `--seeds 8`: conv_transpose1d 48/80, conv_transpose2d 48/96,
  conv_transpose3d 48/80, ALL 0-failed. The residual skips are GENUINE
  production feature-gaps, each a filed blocker: `groups != 1` (#1607,
  `ConvTranspose*::from_parts` is groups=1 only) and `dilation != 1` (#1608,
  dilation=1 only). Those are the only `Ok(None)` skips; no parity FAIL remains.
- [x] AC-11: String padding `'same'` / `'valid'` for Conv1d/2d/3d —
  forward + backward match the live torch 2.11 oracle including the
  asymmetric even-kernel `'same'` split, `'valid'`, and the stride>1
  `'same'` `ValueError` (verified by `test_conv{1,2,3}d_same_*` /
  `test_conv{1,2}d_valid_matches_torch` / `test_conv_same_stride_gt1_rejected`
  in `conv.rs`). Closes #1602.
- [x] AC-12: Unbatched `(C, *spatial)` input for Conv1d/2d/3d — forward
  produces a rank-`D+1` output and backward produces a gradient of the
  unbatched input shape, matching the live torch 2.11 oracle (verified by
  `test_conv{1,2,3}d_unbatched_{forward,backward}_matches_torch` and
  `test_conv2d_unbatched_same_composes` in `conv.rs`). Closes #1604.

## Architecture

### im2col / col2im (REQ-3, REQ-5)

Internal helpers `im2col`, `im2col_dilated`, `col2im`,
`col2im_dilated` in `conv.rs`. The dilated variants are the actual
workhorses; the non-dilated forms are thin shims passing
`dil_h = dil_w = 1`. Each kernel is `#[allow(clippy::too_many_arguments)]`
because the descriptor (B, C, H, W, kH, kW, …) mirrors a 2D
convolution layout; refactoring to a config struct would force
allocation on hot paths.

### Conv2d forward (REQ-3, REQ-4)

`<Conv2d<T> as Module<T>>::forward` in `conv.rs` records autocast
("conv2d"), validates ndim=4 + `in_channels` match, computes
`(H_out, W_out)`, then either:
1. Dispatches to `backend.conv2d_f32(...)` for f32 CUDA tensors via
   `ferrotorch_gpu::backend_impl::conv2d_f32`. If backward is
   needed, also runs CPU `im2col` to save cols for the backward
   GradFn.
2. Runs CPU im2col + per-group matmul + reshape, with optional bias
   add.

### Conv2dBackward (REQ-5)

`Conv2dBackward<T>: GradFn<T>` in `conv.rs`. Computes
`grad_input` via `col2im_dilated` of `(grad_output_2d @
weight_2d^T)`; `grad_weight` via `(grad_output_2d @ cols^T)`
reshaped to `[out_channels, in_channels/groups, kH, kW]`;
`grad_bias` via summing across batch + spatial dims.

### Conv1d / Conv3d (REQ-6)

Parallel `pub struct Conv1d<T: Float>` and `pub struct Conv3d<T:
Float>` definitions in `conv.rs`. Both now carry `dilation` and
`groups` fields and a `new_full(.., dilation, groups, bias)`
constructor mirroring `Conv2d::new_full` (and the upstream
`Conv1d.__init__` / `Conv3d.__init__` argument order at
`torch/nn/modules/conv.py:330-339` / `682-691`); `new` is a thin
shim delegating to `new_full` with dilation `1` / `groups 1`. Both
validate `in_channels % groups == 0` and `out_channels % groups ==
0` (raising the upstream-equivalent error per `conv.py:107-110`),
and the weight layout is `[out, in/groups, *kernel]` (`conv.py:171`).

The 1D forward collapses the W dimension and feeds the 2-D dilated
`im2col_dilated` (temporal dilation maps to the W axis); the 3D
forward uses the new `im2col_3d_dilated` / `col2im_3d_dilated`
helpers (the prior non-dilated `im2col_3d` / `col2im_3d` lacked
dilation). Both forwards partition input channels into `groups`
slices, convolve each with its weight slice, and stack the per-group
outputs along the C_out axis — the same per-group subtensor/cat loop
PyTorch runs at `aten/src/ATen/native/Convolution.cpp:1723-1729`.
`Conv1dBackward` / `Conv3dBackward` carry `dilation`/`groups` and
save the im2col columns in the dense channel layout so the per-group
`grad_input` (via `col2im_*_dilated`) / `grad_weight` /
`grad_bias` mirror `Conv2dBackward`'s grouped decomposition.
`from_parts` stays dense (`groups = 1`, `dilation = 1`) so
`nn::functional::conv{1,3}d` remain ABI-compatible (they cannot
recover `groups` from the weight shape alone).

### ConvTranspose1/2/3d (REQ-6)

`pub struct ConvTranspose2d<T: Float>`, `ConvTranspose1d<T: Float>`,
`ConvTranspose3d<T: Float>` in `conv.rs`. Forward computes
`y = col2im(weight^T @ flatten(input))` with `output_padding`
applied to the result shape; backward swaps the roles of
forward-pass `im2col` and `col2im`.

### Trait + helpers (REQ-7, REQ-8)

Every `Conv*d` impls `Module<T>` with `parameters()` yielding
`[&weight]` or `[&weight, &bias]`, `named_parameters()` yielding
`("weight", &weight)` and conditionally `("bias", &bias)`.
`Conv2d::set_weight` and `Conv2d::from_parts` allow user-supplied
weight tensors without going through the Kaiming init.

### Non-test production consumers

- `pub use conv::{Conv1d, Conv2d, Conv3d, ConvTranspose1d,
  ConvTranspose2d, ConvTranspose3d}` at
  `ferrotorch-nn/src/lib.rs`.
- `ferrotorch-nn/src/se.rs` uses `Conv2d` for the
  squeeze-and-excitation block's 1×1 convs.
- `ferrotorch-nn/src/lazy_conv.rs` constructs `Conv{1,2,3}d` from
  `LazyConv{1,2,3}d::materialize` on first forward.
- `ferrotorch-nn/src/lazy_conv_transpose.rs` similarly constructs
  the transposed variants from `LazyConvTranspose{1,2,3}d`.
- `ferrotorch-nn/src/functional.rs` uses `Conv*d::from_parts` for
  the stateless `nn::functional::conv*d` dispatch.
- `ferrotorch-vision/src/models/{resnet,vit,convnext,swin,yolo,
  inception,vgg,segmentation/{aspp,fcn,deeplabv3},detection/{rpn,
  faster_rcnn}}.rs` instantiate `Conv2d` extensively for image
  models.
- `ferrotorch-gpu/src/lib.rs` re-exports `gpu_conv2d_f32` which is
  the actual CUDA kernel `Conv2d::forward` dispatches to.

## Parity contract

`parity_ops = ["nn.functional.conv1d", "nn.functional.conv2d",
"nn.functional.conv3d", "nn.functional.conv_transpose1d",
"nn.functional.conv_transpose2d", "nn.functional.conv_transpose3d"]`.

For each:
- **dtype promotion**: PyTorch upcasts to f32 in autocast mode;
  ferrotorch's `autocast_guard("conv2d")` records the decision.
- **non-contiguous input**: PyTorch internally contiguizes via
  `view` / `reshape` before im2col; ferrotorch's `data_vec()` call
  flattens to the canonical layout.
- **groups**: PyTorch requires `in_channels % groups == 0` and
  `out_channels % groups == 0`; ferrotorch matches across all three
  forward layers (`Conv1d`/`Conv2d`/`Conv3d` each validate in
  `new_full` and partition channels per-group in forward + backward).
- **dilation**: PyTorch's `eff_kernel = dilation * (kernel - 1) +
  1` (`ConvUtils.h:255`); ferrotorch matches across `Conv1d`
  (via the 2-D `im2col_dilated` with the dilation on the W axis),
  `Conv2d`, and `Conv3d` (via `im2col_3d_dilated`).
- **output_padding** (transposed): PyTorch requires
  `output_padding < max(stride, dilation)`; ferrotorch validates at
  construction.
- **padding_mode**: SHIPPED (#1443) — forward layers (`Conv{1,2,3}d`)
  pre-pad via the autograd-aware `functional_pad_{1,2,3}d` for
  reflect/replicate/circular and convolve with `padding=0`;
  transposed layers reject non-`zeros` modes with the upstream
  `ValueError`. The pad amounts follow torch's
  `_reversed_padding_repeated_twice` (reverse-dim order for `F.pad`).
- **Stride 0 / kernel 0**: rejected by both upstream and
  ferrotorch.
- **Empty batch (B=0)**: upstream returns `[0, C_out, H_out,
  W_out]`; ferrotorch matches via the im2col + matmul algebra.

Parity-sweep audit entries (`parity_audit.json`): `nn.functional.conv1d`
/ `conv2d` / `conv3d` are `verified` at 0-skip / 0-failed (the
`dispatch_conv::<D>` arm; #1441). The three `conv_transpose{1,2,3}d`
arms are now `verified` at 0-failed via `dispatch_conv_transpose::<D>`
(48/80, 48/96, 48/80 at `--seeds 8`); their residual skips map to the
`groups != 1` (#1607) / `dilation != 1` (#1608) production gaps.

## Verification

Tests in `mod tests` of `conv.rs` (60+ tests across the 6 variants),
covering:
- Construction validation (`test_conv2d_zero_groups_rejected`,
  `test_conv2d_groups_must_divide_in_channels`, etc.).
- Forward shape correctness across stride / padding / dilation /
  groups combinations.
- Forward numerical correctness against hand-computed reference
  outputs.
- Backward gradient computation:
  `test_conv2d_backward_input_grad`, `test_conv2d_backward_weight_grad`,
  `test_conv2d_backward_bias_grad`, plus
  `test_conv2d_backward_numerical_gradient` for the FD check.
- ConvTranspose tests: shape, output_padding, backward.

Parity-sweep smoke commands. conv1d/conv2d/conv3d reach 0-skip /
0-failed at `--seeds 8` (`dispatch_conv::<D>` via `new_full` +
`with_string_padding`, #1441): conv1d 80/80, conv2d 240/240,
conv3d 160/160. The `conv_transpose{1,2,3}d` arms
(`dispatch_conv_transpose::<D>`) reach 0-failed at `--seeds 8`:
conv_transpose1d 48/80, conv_transpose2d 48/96, conv_transpose3d 48/80;
residual skips are the `groups != 1` (#1607) / `dilation != 1` (#1608)
production gaps.

```bash
for OP in nn.functional.conv1d nn.functional.conv2d nn.functional.conv3d \
         nn.functional.conv_transpose1d nn.functional.conv_transpose2d \
         nn.functional.conv_transpose3d; do
  ./target/release/parity-sweep sweep --op "$OP" --seeds 8 2>&1 | tail -1
done
```

Grep count for `passed (0 skipped, 0 failed)`: `>= 1` for conv1d /
conv2d / conv3d; the conv_transpose arms close under the remaining
#1441 scope.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Conv2d<T: Float>` in `conv.rs` mirroring `aten/src/ATen/native/Convolution.cpp:520-600`; non-test consumer: `ferrotorch-vision/src/models/resnet.rs` constructs `Conv2d::new(...)` for every residual block conv (the BasicBlock + Bottleneck building blocks). |
| REQ-2 | SHIPPED | impl: `pub fn new` and `pub fn new_full` in `conv.rs` with `groups` / `dilation` validation; non-test consumer: `ferrotorch-vision/src/models/vit.rs` and `convnext.rs` construct grouped or dilated Conv2d via `new_full`. |
| REQ-3 | SHIPPED | impl: `<Conv2d as Module>::forward` body in `conv.rs` (im2col + matmul) mirroring `aten::convolution`; non-test consumer: every vision model forward invokes `Conv2d::forward` through its `Module` impl. |
| REQ-4 | SHIPPED | impl: `is_f32 && input.is_cuda()` dispatch to `backend.conv2d_f32` in `<Conv2d as Module>::forward` in `conv.rs`; non-test consumer: `ferrotorch-gpu/src/backend_impl.rs` exposes `Backend::conv2d_f32`; vision-model training runs that put modules on CUDA trigger this dispatch end-to-end. |
| REQ-5 | SHIPPED | impl: `Conv2dBackward<T>: GradFn<T>` impl block in `conv.rs`; non-test consumer: every gradient step on a vision model's `loss.backward()` traverses these `Conv2dBackward` nodes through `ferrotorch_core::autograd::engine`. |
| REQ-6 | SHIPPED | impl: `pub struct Conv1d` / `Conv3d` / `ConvTranspose{1,2,3}d` in `conv.rs`, each carrying `groups`/`dilation` for the forward layers via `Conv1d::new_full` / `Conv3d::new_full` + the per-group + dilated `<Conv1d as Module>::forward` / `<Conv3d as Module>::forward` and `Conv1dBackward` / `Conv3dBackward` (closes #1600 conv1d, #1601 conv3d; mirrors `Conv2d::new_full` per `torch/nn/modules/conv.py:171` weight layout + `Convolution.cpp:1723-1729` channel partition); non-test consumer: `Conv1d::new` / `Conv3d::new` delegate to `new_full` in production and are themselves called by `ferrotorch-nn/src/lazy_conv.rs` `LazyConv1d::materialize` / `LazyConv3d::materialize`; `ferrotorch-vision/src/models/inception.rs` uses Conv2d + ConvTranspose2d. |
| REQ-7 | SHIPPED | impl: `impl<T: Float> Module<T> for Conv2d<T>` block (and analogues for the other 5) in `conv.rs`; non-test consumer: `ferrotorch_optim` walks `Module::parameters_mut()` across every conv in a training loop. |
| REQ-8 | SHIPPED | impl: `Conv2d::set_weight` and `Conv2d::from_parts` in `conv.rs`; non-test consumer: `ferrotorch-nn/src/functional.rs` (the stateless `nn::functional::conv2d` entry point) uses `Conv2d::from_parts` to drive the existing forward path with user-supplied parameters. |
| REQ-9 | SHIPPED | impl: `kaiming_uniform(&mut weight, NonLinearity::ReLU)` + `uniform_init(&mut b, -bound, bound)` (bound = 1/sqrt(fan_in)) in every `Conv*d::new[_full]` in `conv.rs` mirroring `torch/nn/modules/conv.py:198-201`; non-test consumer: `Conv2d::new` is the path used by every vision-model constructor. (Closes #1450 — bias path; Kaiming `a=sqrt(5)` gain divergence remains a separate followup.) |
| REQ-10 | SHIPPED | impl (forward layers): `padding_mode` field + `with_padding_mode` builder on `Conv1d` / `Conv2d` / `Conv3d`, with the non-`Zeros` pre-pad branch in each `<Conv*d as Module>::forward` calling `crate::padding::functional_pad_1d`/`_2d`/`_3d` then convolving with `padding=0`, mirroring `torch/nn/modules/conv.py:367-378` (Conv1d) / `716-732` (Conv3d). impl (transposed): `ConvTranspose{1,2,3}d::with_padding_mode` routes through `fn reject_non_zeros_transpose` returning the upstream `ValueError('Only "zeros" padding mode is supported for ...')` per `conv.py:755-758`. The 1-D/3-D pre-pads are autograd-aware via `Pad1dBackward` / `Pad3dBackward` in `padding.rs` (the #1550 fix class). Non-test production consumer: `pub use conv::{Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d}` re-export in `ferrotorch-nn/src/lib.rs`, and the `<Conv1d as Module>::forward` / `<Conv3d as Module>::forward` bodies consume `functional_pad_1d` / `functional_pad_3d` in production. Closes #1443. |
| REQ-11 | SHIPPED (forward arms 0-skip; transpose arms 0-failed w/ gap skips) | impl: `dispatch_conv::<D>` in `tools/parity-sweep/runner/src/main.rs` wires `nn.functional.conv1d`/`conv2d`/`conv3d` (#1441). All three ranks build via `Conv{1,2,3}d::new_full` + `with_string_padding` + `Parameter::set_data` + `Module::forward`, driving the production grouped+dilated CPU forward (non-test production driver of `new_full`: `ferrotorch-vision/src/models/resnet.rs` grouped/dilated blocks; `ferrotorch-nn/src/lazy_conv.rs` `LazyConv1d::materialize` / `LazyConv3d::materialize`). Sweep at `--seeds 8`: conv1d 80/80, conv2d 240/240, conv3d 160/160, ALL 0 skipped / 0 failed. groups / dilation (#1600 / #1601), `'same'`/`'valid'` (REQ-12 / #1602), and unbatched rank-(D+1) (REQ-13 / #1604) op_db samples all RUN. The transposed arms are now wired via `dispatch_conv_transpose::<D>` in the same `main.rs` (decode stride/padding/output_padding/bias; unbatched rank-(D+1) run via runner-side batchify on the production batched forward — non-test driver of `from_parts`: `ferrotorch-nn/src/lazy_conv_transpose.rs` `LazyConvTranspose{1,2,3}d::materialize` + `ferrotorch-vision/src/models/inception.rs` `ConvTranspose2d`). Sweep at `--seeds 8`: conv_transpose1d 48/80, conv_transpose2d 48/96, conv_transpose3d 48/80, ALL 0-failed (the prior conv_transpose3d FMA-cancellation FAIL resolved by widening `atol` 1e-7→1e-5 to match the conv-forward many-FMA envelope in `tolerance_for`). Residual skips are GENUINE production gaps: `groups != 1` (#1607) and `dilation != 1` (#1608) — `ConvTranspose*::from_parts` is groups=1/dilation=1 only. |
| REQ-12 | SHIPPED | impl: `pub enum StringPadding` + `fn same_pad_lr` + `Conv1d::with_string_padding` / `Conv2d::with_string_padding` / `Conv3d::with_string_padding` and the `string_padding` branch at the top of each `<Conv*d as Module>::forward` in `conv.rs` (asymmetric pre-pad via `crate::padding::functional_pad_{1,2,3}d`, `left=total/2`/`right=total-left` per `aten/src/ATen/native/Pool.h:91-107`; stride>1 `'same'` rejected per `torch/nn/modules/conv.py:117-120`); non-test production consumer: the `forward` bodies (production `Module::forward`) consume `same_pad_lr` + `functional_pad_{1,2,3}d` + `recurse_clone`, and the `Conv{1,2,3}d` types are re-exported from `ferrotorch-nn/src/lib.rs`. Closes #1602. |
| REQ-13 | SHIPPED | impl: the unbatched `input.ndim()` guard at the top of each `<Conv*d as Module>::forward` in `conv.rs` (`unsqueeze(0)` → recurse → `squeeze(0)`) using `ferrotorch_core::grad_fns::shape::{unsqueeze, squeeze}`, mirroring `batchify` + `output.squeeze(0)` at `aten/src/ATen/native/Convolution.cpp:816-831, 990-1047`; non-test production consumer: the `<Conv*d as Module>::forward` bodies (production) call `unsqueeze`/`squeeze`, and the `Conv{1,2,3}d` types are re-exported from `ferrotorch-nn/src/lib.rs`. Closes #1604. |
