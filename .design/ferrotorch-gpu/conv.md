# cuDNN/cuBLAS GPU 2-D convolution (im2col + GEMM)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cudnn/ConvShared.cpp
  - aten/src/ATen/native/cudnn/Conv_v7.cpp
  - aten/src/ATen/native/cudnn/Conv_v8.cpp
  - aten/src/ATen/native/Convolution.cpp
-->

## Summary

`ferrotorch-gpu/src/conv.rs` is the GPU-resident 2-D convolution path
used by `ferrotorch-nn::Conv2d`. The implementation uses **im2col + cuBLAS
SGEMM** rather than calling cuDNN's `cudnnConvolutionForward` directly,
because (a) the cuBLAS path covers the dtypes the rest of ferrotorch-gpu
already handles, (b) im2col on the GPU is a single custom PTX kernel
(`crate::kernels::im2col_2d_f32`), and (c) cuDNN's algorithm-selection
heuristics are not required for the shape regimes the test fleet exercises.
The output of `gpu_conv2d_f32` is `(buffer, shape)` — both stay on-device.

This module mirrors PyTorch's `cudnn_convolution_forward_out` semantics
(NCHW row-major, stride / padding / dilation / groups support) but the
implementation strategy is the im2col-GEMM fallback path that
`aten/src/ATen/native/Convolution.cpp` falls back to when cuDNN is
disabled or unsuitable. See R-DEV-7 — the upstream contract (NCHW conv
with stride/padding/dilation/groups) is preserved; the implementation
deviates from upstream's cuDNN-default path because cuBLAS+PTX is the
simpler, dtype-uniform Rust-side answer.

## Requirements

- REQ-1: `pub fn gpu_conv2d_f32` — computes 2-D convolution `output =
  weight @ im2col(input) + bias` for NCHW row-major tensors. Inputs:
  `input: &CudaBuffer<f32>` (`[B, C_in, H, W]`), `weight:
  &CudaBuffer<f32>` (`[C_out, C_in/groups, kH, kW]`), optional `bias:
  Option<&CudaBuffer<f32>>` (`[C_out]`), plus stride / padding /
  dilation / groups scalars. Returns
  `(CudaBuffer<f32>, [B, C_out, H_out, W_out])`. Mirrors PyTorch
  `cudnn_convolution(input, weight, bias, ...)` user-API contract.
- REQ-2: Output spatial dims follow the standard formula
  `H_out = (H + 2*pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1`
  (same as `aten/src/ATen/native/Convolution.cpp::conv_output_size`).
- REQ-3: Groups support — when `groups > 1` the input channels and
  weight channels are partitioned into `groups` independent
  convolution sub-problems. Each group's GEMM runs separately; the
  outputs are concatenated along the channel dimension.
- REQ-4: Bias broadcasting — when `bias` is `Some(...)`, the `C_out`
  per-channel bias is added to each spatial output position via the
  on-device broadcast-bias kernel (`crate::kernels` family).
- REQ-5: GPU-resident pipeline — im2col, GEMM, and bias add all run on
  the GPU. Zero CPU round-trips per `rust-gpu-discipline §3`.
- REQ-6: No-CUDA stub — `cfg(not(feature = "cuda"))` returns
  `GpuError::NoCudaFeature`.

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-gpu --features cuda conv::` passes
  the 13 in-module tests (stride / padding / dilation / groups / bias
  combinations, plus shape correctness against hand-computed expected
  outputs).
- [x] AC-2: `output_shape` matches PyTorch's `conv_output_size` for
  representative `[stride, pad, dilation, kernel]` tuples — pinned by
  the `output_shape_matches_pytorch_*` tests.
- [x] AC-3: Bias is added per-channel; identity when `bias = None`.
- [x] AC-4: Groups partitioning is correct for `groups > 1` — pinned
  by the `groups_partition` test.
- [x] AC-5: No-CUDA stub returns `GpuError::NoCudaFeature`.

## Architecture

### Forward pipeline (REQ-1, REQ-5)

`pub fn gpu_conv2d_f32 in conv.rs` runs:

1. Compute `H_out`, `W_out` from `H`, `W`, kernel dims, stride, padding,
   dilation.
2. Allocate `col_buf = CudaBuffer<f32>` of size
   `[C_in * kH * kW, B * H_out * W_out]` on-device.
3. Launch `crate::kernels::im2col_2d_f32` (custom PTX, one thread per
   output element of the col matrix) to materialise the patched view
   of `input` into `col_buf`.
4. Reshape `weight` to `[C_out, C_in/groups * kH * kW]` (this is a
   shape reinterpret — no data movement; the row-major layout of the
   weight tensor already aligns).
5. Call `crate::blas::gpu_matmul_f32_into(weight_buf, col_buf,
   out_buf)` (or its grouped variant) to compute the result.
6. If `bias.is_some()`, launch the broadcast-add bias kernel.
7. Return `(out_buf, [B, C_out, H_out, W_out])`.

The non-test production consumer is `ferrotorch-gpu/src/backend_impl.rs`
at line 2521 (the cuda backend's `conv2d` dispatch arm). This arm is
called from `ferrotorch-core/src/gpu_dispatch.rs` when a Tensor's
`conv2d` op routes to GPU. The consumer in turn is reached from
`ferrotorch-nn::Conv2d::forward` (the public API users call).

### Output shape (REQ-2)

Computed in-line in `gpu_conv2d_f32`:

```text
H_out = (H + 2*pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1
W_out = (W + 2*pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1
```

This formula mirrors `aten/src/ATen/native/Convolution.cpp` (see the
`conv_output_size` helper around line 144). The
`output_shape_matches_pytorch_*` tests pin it against tabulated
expected dimensions for the common `[3x3,2x2,5x5]` x `[stride 1, 2]`
x `[pad 0, 1]` x `[dilation 1, 2]` matrix.

### Groups (REQ-3)

When `groups > 1`, the loop slices the channel dim into `groups`
sub-problems. Each sub-problem has weight `[C_out/groups, C_in/groups
* kH * kW]` and im2col output `[C_in/groups * kH * kW, B * H_out *
W_out]`; the per-group GEMM produces `[C_out/groups, B * H_out *
W_out]`. The outputs are concatenated along the channel dim in the
same `out_buf` via offset writes.

Mirrors PyTorch's groups semantics in
`aten/src/ATen/native/cudnn/ConvShared.cpp` and
`aten/src/ATen/native/Convolution.cpp::groups`.

### Bias (REQ-4)

Bias broadcast is one custom PTX kernel launch that adds
`bias[c]` to each `[b, c, h_out, w_out]` output element. The kernel
is generic over the spatial size; one thread per output element.

### No-CUDA stub (REQ-6)

`#[cfg(not(feature = "cuda"))] pub fn gpu_conv2d_f32 in conv.rs` (line
816 region) returns `Err(GpuError::NoCudaFeature)` — preserves the
signature so the crate compiles without the cuda feature.

## Parity contract

`parity_ops = []` for this module. Reason: conv2d is an op-level entry
in `ferrotorch-core`'s parity surface; the cuDNN/im2col GEMM path is
reached transitively when `Tensor::conv2d` routes to the GPU backend.
The conv2d op-level parity sweep (driven from `ferrotorch-core`)
exercises this dispatcher indirectly.

Edge cases mirrored from upstream:

- **`pad >= H + dilation * (kH - 1) + 1`**: produces an empty output
  (`H_out = 0`). The im2col kernel handles this cleanly; the GEMM
  short-circuits.
- **Asymmetric stride / padding / dilation**: each is a per-dim 2-tuple;
  the formula handles each spatial dim independently.
- **Bias of wrong shape**: caller is responsible — the consumer in
  `backend_impl.rs:2521` validates `bias.shape == [C_out]` before
  dispatch, matching PyTorch's `addmm_check_bias_shape` semantics.

## Verification

Tests in `#[cfg(all(test, feature = "cuda"))] mod tests in conv.rs`
(13 functions):

- `output_shape_matches_pytorch_simple_3x3`
- `output_shape_matches_pytorch_stride_2`
- `output_shape_matches_pytorch_padding`
- `output_shape_matches_pytorch_dilation_2`
- `conv2d_identity_kernel`
- `conv2d_bias_per_channel`
- `conv2d_bias_none_equals_no_bias`
- `conv2d_stride_2`
- `conv2d_padding`
- `conv2d_dilation`
- `conv2d_groups_partition`
- `conv2d_large_batch`
- `conv2d_non_square_kernel`

Smoke command:

```bash
cargo test -p ferrotorch-gpu --features cuda conv:: 2>&1 | tail -3
```

Expected: all 13 tests pass. `parity_ops = []` — no per-op parity-sweep
smoke applies; the op-level `conv2d` smoke in `ferrotorch-core` covers
this dispatcher.

## REQ status table

Per S5 (existing pub-API grandfather): `gpu_conv2d_f32` is the public
boundary API consumed by `backend_impl.rs:2521` (the cuda backend's
conv2d arm). That arm is reached from `ferrotorch-core::gpu_dispatch`
when `Tensor::conv2d` routes to GPU.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn gpu_conv2d_f32 in conv.rs` mirrors `cudnn_convolution` user-API per upstream `aten/src/ATen/native/cudnn/ConvShared.cpp:243`. Non-test consumer: `ferrotorch-gpu/src/backend_impl.rs:2521` (cuda backend's conv2d dispatch arm). |
| REQ-2 | SHIPPED | impl: in-line output-shape computation in `gpu_conv2d_f32 in conv.rs` mirroring `aten/src/ATen/native/Convolution.cpp::conv_output_size`. Non-test consumer: same call site at `backend_impl.rs:2521`. |
| REQ-3 | SHIPPED | impl: groups partitioning loop in `gpu_conv2d_f32 in conv.rs` matching upstream groups semantics. Non-test consumer: `backend_impl.rs:2521` passes the user-supplied groups arg through. |
| REQ-4 | SHIPPED | impl: bias broadcast kernel launch in `gpu_conv2d_f32 in conv.rs` (Option<&CudaBuffer<f32>> branch). Non-test consumer: `backend_impl.rs:2521` passes `bias` through unwrapped. |
| REQ-5 | SHIPPED | impl: all three phases (im2col, GEMM, bias) launch on-device; the result `CudaBuffer<f32>` never touches host. Non-test consumer: `backend_impl.rs:2521` keeps the resulting buffer on-device for the downstream cuda backend ops. |
| REQ-6 | SHIPPED | impl: `#[cfg(not(feature = "cuda"))] pub fn gpu_conv2d_f32 in conv.rs` returns `Err(GpuError::NoCudaFeature)`. Non-test consumer: the same `backend_impl.rs` arm under the no-cuda compile path. |
