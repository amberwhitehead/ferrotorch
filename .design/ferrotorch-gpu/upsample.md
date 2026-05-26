# GPU nearest-neighbour 2x upsample (f32)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/UpSampleNearest2d.cu
  - aten/src/ATen/native/UpSample.cpp
-->

## Summary

`ferrotorch-gpu/src/upsample.rs` implements GPU nearest-neighbour
upsampling by exactly 2x on `[B, C, H, W]` f32 tensors. A single
hand-written PTX kernel maps each output element `(oh, ow)` to its
source `(oh / 2, ow / 2)` and reads / writes one element per
thread. Mirrors `torch.nn.functional.interpolate(..., mode="nearest",
scale_factor=2.0)` for the canonical SD VAE / UNet up-block use
case, where the upstream `aten/src/ATen/native/cuda/UpSampleNearest2d.cu`
kernel does the same nearest-2x mapping (with a slightly more
general per-axis scale parameter).

## Requirements

- REQ-1: Public f32 forward entry point
  `pub fn gpu_nearest_upsample2x_f32(input, batch, channels, h_in,
  w_in, device) -> CudaBuffer<f32>` returning the upsampled
  `[B, C, 2*h_in, 2*w_in]` buffer.
- REQ-2: Hand-written PTX kernel `nearest_upsample2x_kernel`
  (`NEAREST_UPSAMPLE2X_PTX`) with the documented ABI
  `(in_ptr, out_ptr, batch, channels, h_in, w_in, total)`. Each
  thread decomposes its output index into `(b, c, oh, ow)`,
  computes `(hi, wi) = (oh / 2, ow / 2)`, reads the source, and
  writes the output.
- REQ-3: Length / device validation: `input.len() == batch *
  channels * h_in * w_in`, all on the same device.
- REQ-4: Empty / degenerate handling (`batch == 0` or `channels ==
  0` or `h_in == 0` or `w_in == 0`): return a zero-allocation
  buffer of the correct (zero) output length.
- REQ-5: Non-test production consumer wiring — ferrotorch-diffusion's
  SD VAE up-block and UNet up-block use this kernel directly.

## Acceptance Criteria

- [x] AC-1: `pub fn gpu_nearest_upsample2x_f32` exists at line 152.
- [x] AC-2: `pub(crate) const NEAREST_UPSAMPLE2X_PTX` exists at
  line 43 with the documented ABI.
- [x] AC-3: Length + device validation in the function body
  (the standard ShapeMismatch / DeviceMismatch pattern shared with
  other GPU kernels).
- [x] AC-4: Empty-input short-circuit in the function body before
  kernel launch.
- [x] AC-5: Non-test consumer at
  `ferrotorch-diffusion/src/gpu/unet.rs:1700` and
  `ferrotorch-diffusion/src/gpu/vae.rs:880`.

## Architecture

`pub fn gpu_nearest_upsample2x_f32 in upsample.rs` does:

1. Validates `input.len() == batch * channels * h_in * w_in`,
   all on the same device.
2. Short-circuits empty inputs by returning a zero-length buffer
   without launching.
3. Resolves `nearest_upsample2x_kernel` via
   `crate::module_cache::get_or_compile`.
4. Allocates the f32 output via
   `alloc_zeros_f32(batch * channels * 4 * h_in * w_in, device)`.
5. Launches with `block_dim = 256`, `grid_dim = ceil(total / 256)`,
   shared_mem = 0. One thread per output element.
6. Returns the `CudaBuffer<f32>` output.

The PTX kernel (`pub(crate) const NEAREST_UPSAMPLE2X_PTX in upsample.rs`)
per thread:

1. Computes its global output index `out_idx`.
2. Decomposes into `(b, c, oh, ow)` via the
   `chw_out = channels * (2*h_in) * (2*w_in)` /
   `hw_out = (2*h_in) * (2*w_in)` / `w_out = 2*w_in` strides.
3. Computes `(hi, wi) = (oh / 2, ow / 2)` — the upstream-PyTorch
   nearest-floor mapping.
4. Computes `in_idx = ((b * channels + c) * h_in + hi) * w_in + wi`.
5. Issues one `ld.global.f32` from the source and one
   `st.global.f32` to the output.

### Non-test production consumers (REQ-5)

`ferrotorch-diffusion/src/gpu/unet.rs:1700`:

```rust
let upsampled = gpu_nearest_upsample2x_f32(x, b, c, h, w, device)
    .map_err(gpu_err)?;
```

This is the SD UNet up-block path — the input passes through a
nearest-2x upsample between resnet stacks before the per-block
convolutions.

`ferrotorch-diffusion/src/gpu/vae.rs:880` — the matching call in
the SD VAE decoder's up-block. Both are end-to-end GPU paths:
the input never leaves the device.

The line-46 / line-30 `use ferrotorch_gpu::{... gpu_nearest_upsample2x_f32 ...}`
imports in both files demonstrate the production-API surface.

## Parity contract

`parity_ops = []` for this route. Nearest-upsample parity is
enforced at the ferrotorch-diffusion / ferrotorch-nn layer where
the higher-level `interpolate(mode="nearest")` op lives; this
file is the 2x-specialised primitive.

Edge cases preserved:

- **Empty input** (`numel == 0`): returns a zero-length buffer
  without launching.
- **Floor-division mapping**: `(oh / 2, ow / 2)` is integer
  division, matching PyTorch's nearest-floor contract. For an
  odd output index (e.g. `oh = 5`), the source is `hi = 2` —
  the lower neighbour.
- **2x-only specialisation**: the kernel hard-codes `scale = 2`
  via `shl.b32 %hi2, %h_r, 1` and the floor-divide-by-2 mapping.
  Other scale factors are NOT supported by this kernel; callers
  must route to a different (CPU or generic-GPU) path.
- **Contiguity**: input and output are both contiguous
  `[B, C, H, W]` row-major. Non-contiguous inputs require
  `.contiguous()` upstream.

## Verification

Unit tests in `ferrotorch-gpu/src/upsample.rs` (gated
`#[cfg(test)] #[cfg(feature = "cuda")]`) cover: 2x upsample
correctness on a small `[1, 1, 2, 2]` fixture against a hand-
computed expected output, the empty-input short-circuit, and
the validation error paths.

Cross-cutting integration is exercised by the SD VAE / UNet
forward tests in `ferrotorch-diffusion/tests/`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda upsample:: 2>&1 | tail -3
```

Expected: ≥ 1 `test result: ok` line.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn gpu_nearest_upsample2x_f32 in ferrotorch-gpu/src/upsample.rs` (line 152); non-test consumer: `ferrotorch-diffusion/src/gpu/unet.rs:1700` and `ferrotorch-diffusion/src/gpu/vae.rs:880` both call it. |
| REQ-2 | SHIPPED | impl: `pub(crate) const NEAREST_UPSAMPLE2X_PTX in upsample.rs` (line 43) with the documented 7-arg ABI; loaded via `module_cache::get_or_compile` in the function body. |
| REQ-3 | SHIPPED | impl: shape + device validation in `upsample.rs` after the launch boundary (the standard `ShapeMismatch` / `DeviceMismatch` pattern). |
| REQ-4 | SHIPPED | impl: empty / degenerate short-circuit in `upsample.rs` returns `alloc_zeros_f32(0, device)` before launching when `total == 0`. |
| REQ-5 | SHIPPED | impl: `pub use upsample::gpu_nearest_upsample2x_f32` at `ferrotorch-gpu/src/lib.rs:250`; non-test consumer: `ferrotorch-diffusion/src/gpu/unet.rs:1700` (SD UNet up-block) and `ferrotorch-diffusion/src/gpu/vae.rs:880` (SD VAE decoder up-block). |
