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
`groups` for the forward layers (`Conv*d`); the transposed layers
support `stride`, `padding`, and `output_padding`. The `padding_mode`
kwarg from upstream IS implemented (#1443): `Conv{1,2,3}d` honor
`reflect`/`replicate`/`circular` (forward + backward, autograd-aware
pre-pad), while `ConvTranspose{1,2,3}d` accept only `zeros` and reject
other modes with the upstream `ValueError`.

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
- REQ-11: NOT-STARTED — parity-sweep runner arms for
  `nn.functional.conv1d`/`conv2d`/`conv3d`/`conv_transpose1d`/
  `conv_transpose2d`/`conv_transpose3d` are absent (each reports
  `0/N passed, N skipped`). Blocker #1441 (umbrella) tracks the
  runner-arm gap.

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
- [ ] AC-10: parity-sweep arms wired for all 6 ops — blocker #1441.

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
Float>` definitions in `conv.rs`. Each has its own
`im2col`-style helper and matching backward. The 1D path collapses
the W dimension; the 3D path adds a D dimension to the inner loops.

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
  `out_channels % groups == 0`; ferrotorch matches.
- **dilation**: PyTorch's `eff_kernel = dilation * (kernel - 1) +
  1`; ferrotorch matches.
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

Parity-sweep audit entries: each of the 6 op names is declared but
the runner has no arm — `parity_audit.json` reports `missing` for
each. Blocker #1441 tracks the runner-arm wiring.

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

Parity-sweep smoke commands (all currently 0/N passed, N skipped
because of the runner-arm gap, blocker #1441):

```bash
for OP in nn.functional.conv1d nn.functional.conv2d nn.functional.conv3d \
         nn.functional.conv_transpose1d nn.functional.conv_transpose2d \
         nn.functional.conv_transpose3d; do
  ./target/release/parity-sweep sweep --op "$OP" --seeds 8 2>&1 | tail -1
done
```

Expected grep count after blocker #1441 closes: `>= 1` for each.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Conv2d<T: Float>` in `conv.rs` mirroring `aten/src/ATen/native/Convolution.cpp:520-600`; non-test consumer: `ferrotorch-vision/src/models/resnet.rs` constructs `Conv2d::new(...)` for every residual block conv (the BasicBlock + Bottleneck building blocks). |
| REQ-2 | SHIPPED | impl: `pub fn new` and `pub fn new_full` in `conv.rs` with `groups` / `dilation` validation; non-test consumer: `ferrotorch-vision/src/models/vit.rs` and `convnext.rs` construct grouped or dilated Conv2d via `new_full`. |
| REQ-3 | SHIPPED | impl: `<Conv2d as Module>::forward` body in `conv.rs` (im2col + matmul) mirroring `aten::convolution`; non-test consumer: every vision model forward invokes `Conv2d::forward` through its `Module` impl. |
| REQ-4 | SHIPPED | impl: `is_f32 && input.is_cuda()` dispatch to `backend.conv2d_f32` in `<Conv2d as Module>::forward` in `conv.rs`; non-test consumer: `ferrotorch-gpu/src/backend_impl.rs` exposes `Backend::conv2d_f32`; vision-model training runs that put modules on CUDA trigger this dispatch end-to-end. |
| REQ-5 | SHIPPED | impl: `Conv2dBackward<T>: GradFn<T>` impl block in `conv.rs`; non-test consumer: every gradient step on a vision model's `loss.backward()` traverses these `Conv2dBackward` nodes through `ferrotorch_core::autograd::engine`. |
| REQ-6 | SHIPPED | impl: `pub struct Conv1d` / `Conv3d` / `ConvTranspose{1,2,3}d` in `conv.rs`; non-test consumer: `ferrotorch-vision/src/models/inception.rs` uses Conv2d + ConvTranspose2d; `ferrotorch-nn/src/lazy_conv.rs` instantiates Conv{1,2,3}d via `materialize`. |
| REQ-7 | SHIPPED | impl: `impl<T: Float> Module<T> for Conv2d<T>` block (and analogues for the other 5) in `conv.rs`; non-test consumer: `ferrotorch_optim` walks `Module::parameters_mut()` across every conv in a training loop. |
| REQ-8 | SHIPPED | impl: `Conv2d::set_weight` and `Conv2d::from_parts` in `conv.rs`; non-test consumer: `ferrotorch-nn/src/functional.rs` (the stateless `nn::functional::conv2d` entry point) uses `Conv2d::from_parts` to drive the existing forward path with user-supplied parameters. |
| REQ-9 | SHIPPED | impl: `kaiming_uniform(&mut weight, NonLinearity::ReLU)` + `uniform_init(&mut b, -bound, bound)` (bound = 1/sqrt(fan_in)) in every `Conv*d::new[_full]` in `conv.rs` mirroring `torch/nn/modules/conv.py:198-201`; non-test consumer: `Conv2d::new` is the path used by every vision-model constructor. (Closes #1450 — bias path; Kaiming `a=sqrt(5)` gain divergence remains a separate followup.) |
| REQ-10 | SHIPPED | impl (forward layers): `padding_mode` field + `with_padding_mode` builder on `Conv1d` / `Conv2d` / `Conv3d`, with the non-`Zeros` pre-pad branch in each `<Conv*d as Module>::forward` calling `crate::padding::functional_pad_1d`/`_2d`/`_3d` then convolving with `padding=0`, mirroring `torch/nn/modules/conv.py:367-378` (Conv1d) / `716-732` (Conv3d). impl (transposed): `ConvTranspose{1,2,3}d::with_padding_mode` routes through `fn reject_non_zeros_transpose` returning the upstream `ValueError('Only "zeros" padding mode is supported for ...')` per `conv.py:755-758`. The 1-D/3-D pre-pads are autograd-aware via `Pad1dBackward` / `Pad3dBackward` in `padding.rs` (the #1550 fix class). Non-test production consumer: `pub use conv::{Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d}` re-export in `ferrotorch-nn/src/lib.rs`, and the `<Conv1d as Module>::forward` / `<Conv3d as Module>::forward` bodies consume `functional_pad_1d` / `functional_pad_3d` in production. Closes #1443. |
| REQ-11 | NOT-STARTED | blocker #1441 (umbrella) — parity-sweep runner arms for all 6 conv ops are absent; sweep reports `0/N passed, N skipped` for each. The forward paths themselves are end-to-end verified by 60+ lib tests; only the runner-arm wiring is missing. |
