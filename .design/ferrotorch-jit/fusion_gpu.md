# ferrotorch-jit — `fusion_gpu` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/_inductor/codegen/cuda/cuda_kernel.py
  - aten/src/ATen/native/cuda/Activation.cu
  - aten/src/ATen/native/cuda/jit_utils.cu
-->

## Summary

`ferrotorch-jit/src/fusion_gpu.rs` is the GPU runtime executor for
`crate::fusion::FusedChain`. Gated behind the `cuda` feature; the
module is omitted from the default workspace build. Mirrors
PyTorch Inductor's CUDA kernel-launch-via-NVRTC path
(`torch._inductor.codegen.cuda.cuda_kernel`) + ATen's
jit_utils.cu launch helpers. f32 chains lower to PTX directly;
f64 chains route through CUDA C source → NVRTC + libdevice → PTX.

## Requirements

- REQ-1: `pub fn apply_fused_gpu<T: Float>(input, chain) ->
  FerrotorchResult<Tensor<T>>` — the GPU dispatch entry point.
  Dispatches on `TypeId::of::<T>()`: f32 → `apply_fused_gpu_f32_internal`;
  f64 → `apply_fused_gpu_f64_internal`; otherwise
  `NotImplementedOnCuda`. Returns a device-resident `Tensor<T>` on
  the same CUDA device as the input.

- REQ-2: f32 dispatch — `apply_fused_gpu_f32_internal` generates
  the chain's PTX directly via
  `FusedChain::generate_ptx_named(FUSED_F32_KERNEL_NAME)`,
  compiles+caches via
  `ferrotorch-gpu::module_cache::get_or_compile_owned`, allocates a
  fresh output `CudaBuffer<f32>` via
  `ferrotorch-gpu::transfer::alloc_zeros_f32`, and launches with
  the standard 256-thread/block config.

- REQ-3: f64 dispatch — `apply_fused_gpu_f64_internal` generates
  CUDA C source via
  `FusedChain::generate_cuda_source_f64_named`, compiles via
  `crate::nvrtc::compile_cuda_source_to_ptx`, then follows the
  same `module_cache` + alloc + launch flow at scalar width f64.
  Bounds-checks `n <= i32::MAX` to avoid the i32 parameter
  wrapping (the f64 kernel ABI uses `int n`).

- REQ-4: Module cache hit — repeated `apply_fused_gpu` calls with
  the same chain reuse the compiled CUDA function from
  `module_cache` (the cache key is PTX hash × device ordinal, so
  the first call pays compilation cost and subsequent calls
  launch directly).

- REQ-5: Kernel-launch SAFETY documentation — every `unsafe { ...
  }` block carries a SAFETY comment naming the ABI contract,
  buffer length, alias-freedom, and stream-lifetime invariants.
  `#![allow(unsafe_code, reason = "cudarc kernel launches are
  fundamentally unsafe")]` is at the module top with documented
  precedent (matches `codegen_jit.rs`).

- REQ-6: Device preservation — the returned `Tensor<T>` lives on
  the input tensor's exact CUDA device ordinal. `wrap_output_f32`
  / `wrap_output_f64` build a `TensorStorage::gpu` from the
  `CudaBuffer<T>` and the input's `device_ordinal()`.

- REQ-7: Binary op rejection — chains containing Add/Sub/Mul/Div
  must surface `Err(InvalidArgument)` (since the f32 PTX /
  f64 CUDA-C single-input kernel ABI cannot supply a second
  operand). Both internal dispatchers propagate the
  `generate_ptx_named` / `generate_cuda_source_f64_named` error
  unchanged.

- REQ-8: Launch config — 1-D launch with 256-threads / block, grid
  ceil-div the element count by 256 (with `n_u32.saturating_add`
  to avoid overflow). `n > u32::MAX` returns `Err(InvalidArgument)`.
  Matches the project-wide convention from
  `ferrotorch-gpu/src/conv.rs::launch_cfg`.

## Acceptance Criteria

- [x] AC-1: `apply_fused_gpu(&cuda_f32_tensor, &chain)` on a 7-elt
  `Relu → Neg` chain returns a device-resident `Tensor<f32>` whose
  CPU readback matches the CPU reference within 1e-5.
- [x] AC-2: Same chain on `Tensor<f64>` returns the f64 result
  matching CPU within 1e-12.
- [x] AC-3: Transcendental chains on f32 match CPU within 1e-3
  (loose tolerance for the `*.approx.f32` instructions).
- [x] AC-4: Transcendental chains on f64 match CPU within 1e-9
  (libdevice-tight tolerance).
- [x] AC-5: A chain with `FusedOp::Mul` returns
  `Err(InvalidArgument { message: contains "binary op" })`.
- [x] AC-6: The returned tensor's `device()` equals the input's
  `device()` for any CUDA input.
- [x] AC-7: Two back-to-back calls with the same chain produce
  identical results (cache hit on the second).
- [x] AC-8: Non-f32, non-f64 dtype `T` returns
  `NotImplementedOnCuda` with the documented op message.

## Architecture

### `apply_fused_gpu` dispatch (REQ-1, REQ-7)

`pub fn apply_fused_gpu` at `pub fn apply_fused_gpu in fusion_gpu.rs`
checks `TypeId::of::<T>()` against `TypeId::of::<f32>()` and
`TypeId::of::<f64>()`, calls the matching internal dispatcher, and
returns `Err(NotImplementedOnCuda { op: "..." })` for any other
dtype. Binary-op rejection happens inside the internal
dispatchers' PTX / CUDA-C generators
(`generate_ptx_named` / `generate_cuda_source_f64_named`).

### f32 + f64 internal dispatchers (REQ-2, REQ-3)

`fn apply_fused_gpu_f32_internal` at
`fn apply_fused_gpu_f32_internal in fusion_gpu.rs`:

1. `input.gpu_handle()?.downcast_ref::<CudaBuffer<f32>>()`
   extracts the typed buffer; mismatch returns
   `Err(InvalidArgument)`.
2. `GpuDevice::new(handle.device_ordinal())` re-acquires the
   device (cudarc context lookup).
3. `chain.generate_ptx_named(FUSED_F32_KERNEL_NAME)?` generates
   PTX with the stable kernel-entry name `"fused_chain_f32"`.
4. `module_cache::get_or_compile_owned(device.context(), ptx,
   FUSED_F32_KERNEL_NAME.to_string(), device.ordinal() as u32)`
   compiles or fetches the cached `CudaFunction`.
5. `alloc_zeros_f32(n, &device)` allocates the output buffer.
6. `launch_cfg(n)?` builds the launch config.
7. `stream.launch_builder(&func).arg(buffer.inner())
   .arg(out_buf.inner_mut()).arg(&n_u32).launch(cfg)?` inside an
   `unsafe { ... }` block.
8. `wrap_output_f32(out_buf, shape, device_ordinal)` returns the
   device-resident `Tensor<T>` (where `T == f32`).

`fn apply_fused_gpu_f64_internal` follows the same pattern but
generates CUDA C source via `generate_cuda_source_f64_named` and
NVRTC-compiles via `crate::nvrtc::compile_cuda_source_to_ptx`
before the module-cache lookup. The `n <= i32::MAX` bound check
exists because the f64 CUDA-C kernel ABI uses `int n` rather than
`uint32_t n`.

### Launch config + tensor wrapping (REQ-6, REQ-8)

`fn launch_cfg` at `fn launch_cfg in fusion_gpu.rs` returns the
standard 256-threads/block 1-D launch. `n > u32::MAX` returns
`Err(InvalidArgument)`.

`fn wrap_output_f32` / `fn wrap_output_f64` at the same file
build a `GpuBufferHandle::new(Box::new(buf), device_ordinal, len,
DType::F32 | DType::F64)` and wrap it via
`TensorStorage::gpu(handle)` then `Tensor::from_storage(storage,
shape, false)`. The `debug_assert_eq!(TypeId::of::<T>(),
TypeId::of::<f32 | f64>())` guards the calling-convention promise.

### SAFETY documentation (REQ-5)

`#![allow(unsafe_code, reason = "cudarc kernel launches are
fundamentally unsafe — caller is responsible for ABI matching
the bound argument list")]` at the module top mirrors the
precedent in `codegen_jit.rs`. Every `unsafe { ... }` block has a
SAFETY comment naming:

1. The PTX / CUDA-C signature the function pointer was compiled
   from (and which `FusedChain` emitter produced it).
2. The buffer non-null-and-length-n invariant.
3. The output-buffer freshness (no aliasing with the input).
4. The stream lifetime — `cudarc` queues the kernel and
   sync-on-readback is the caller's responsibility.

### Non-test production consumers

- The module is `#[cfg(feature = "cuda")] pub mod fusion_gpu;` at
  `ferrotorch-jit/src/lib.rs:85`.
- `ferrotorch-jit/src/fusion.rs:1253` (inside `pub fn apply_fused`)
  `crate::fusion_gpu::apply_fused_gpu(input, chain)` — the
  canonical tensor-level GPU dispatch.

## Parity contract

`parity_ops = []`. The module is the runtime side of the GPU
fused-chain path; parity tests are inline in this module. Tests
exercise every documented numerical contract end-to-end:

- **f32 transcendentals** — `Exp → Log → Sigmoid` chain tested
  with 1e-3 tolerance (the loose bound reflects the ~1 ULP error
  of `*.approx.f32`).
- **f64 transcendentals** — same chain at 1e-9 tolerance
  (libdevice-tight; libdevice double-precision transcendentals
  are IEEE-accurate to within a few ULPs).
- **Multi-op f32** — `Abs → ScalarAdd → Sqrt → ScalarMul → Neg`
  chain matches CPU within 1e-4.
- **Device preservation** — `gpu_out.device() ==
  cuda_tensor.device()` invariant tested directly.
- **Module cache** — back-to-back calls with the same chain
  succeed and produce identical results; the cache hit is
  confirmed by direct timing observation in
  `apply_fused_gpu_cache_hit_second_call`.
- **Binary-op rejection** — explicitly tested on both f32 and
  f64.

## Verification

Tests in `mod tests in fusion_gpu.rs`: every test calls
`cuda_or_skip` to short-circuit on systems without CUDA. Test
coverage:
`apply_fused_gpu_f32_scalar_add_relu_neg_roundtrip`,
`apply_fused_gpu_f64_scalar_add_relu_neg_roundtrip`,
`apply_fused_gpu_f32_with_transcendentals`,
`apply_fused_gpu_f64_with_transcendentals`,
`apply_fused_gpu_preserves_device_for_cuda_input`,
`apply_fused_gpu_errs_on_binary_op_chain_f32`,
`apply_fused_gpu_errs_on_binary_op_chain_f64`,
`apply_fused_gpu_multi_op_chain_f32_matches_cpu`,
`apply_fused_gpu_cache_hit_second_call`.

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-jit --lib fusion_gpu:: --features cuda 2>&1 | tail -3
```

Expected: all tests pass (or skip if no CUDA device is available).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn apply_fused_gpu` in `fusion_gpu.rs`; non-test consumer: `apply_fused_gpu in ferrotorch-jit/src/fusion.rs` `return crate::fusion_gpu::apply_fused_gpu(input, chain);` inside `pub fn apply_fused` (under `#[cfg(feature = "cuda")]`). |
| REQ-2 | SHIPPED | impl: `fn apply_fused_gpu_f32_internal` + `const FUSED_F32_KERNEL_NAME: &str = "fused_chain_f32"` in `fusion_gpu.rs`; non-test consumer: invoked by `pub fn apply_fused_gpu` when `TypeId::of::<T>() == TypeId::of::<f32>()`; that public fn is called from `fusion in fusion.rs`. |
| REQ-3 | SHIPPED | impl: `fn apply_fused_gpu_f64_internal` + `const FUSED_F64_KERNEL_NAME: &str = "fused_chain_f64"` + the `n <= i32::MAX` check in `fusion_gpu.rs`; non-test consumer: invoked by `pub fn apply_fused_gpu` when `TypeId::of::<T>() == TypeId::of::<f64>()`; that public fn is called from `fusion in fusion.rs`. |
| REQ-4 | SHIPPED | impl: `module_cache::get_or_compile_owned(device.context(), ptx, FUSED_F32_KERNEL_NAME.to_string(), device.ordinal() as u32)` (f32 path) + same call with `FUSED_F64_KERNEL_NAME` (f64 path) in `fusion_gpu.rs`; non-test consumer: every call to `apply_fused_gpu` (via `fusion in fusion.rs`) services its compiled-function lookup through this cache, which is keyed on PTX hash × device ordinal. |
| REQ-5 | SHIPPED | impl: `#![allow(unsafe_code, reason = "...")]` at the module top + 2 per-block `// SAFETY:` comments in `fusion_gpu.rs`; non-test consumer: every `apply_fused_gpu` dispatch flows through one of the two SAFETY-documented `unsafe { ... }` launch blocks. |
| REQ-6 | SHIPPED | impl: `fn wrap_output_f32` + `fn wrap_output_f64` building a `GpuBufferHandle::new(..., device_ordinal, ..., DType::F32 \| DType::F64)` in `fusion_gpu.rs`; non-test consumer: invoked at the end of each internal dispatcher; the returned `Tensor<T>` is what `fusion in fusion.rs` ultimately yields. |
| REQ-7 | SHIPPED | impl: binary-op rejection inside `FusedChain::generate_ptx_named` (f32 path) / `FusedChain::generate_cuda_source_f64_named` (f64 path), propagated unchanged through `fn apply_fused_gpu_f32_internal` / `fn apply_fused_gpu_f64_internal` in `fusion_gpu.rs`; non-test consumer: callers receive the error via `fusion in fusion.rs` → `apply_fused_gpu` → the internal dispatcher. |
| REQ-8 | SHIPPED | impl: `fn launch_cfg` with `BLOCK = 256` and the `n > u32::MAX` overflow check in `fusion_gpu.rs`; non-test consumer: invoked by both internal dispatchers (call sites at fusion_gpu.rs `launch_cfg(n)?` inside f32 and f64 paths). |
