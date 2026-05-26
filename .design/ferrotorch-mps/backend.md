# MtlBackend — Apple Metal implementation of GpuBackend

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - aten/src/ATen/mps/MPSDevice.h
  - aten/src/ATen/mps/MPSDevice.mm
  - aten/src/ATen/mps/MPSAllocator.h
  - aten/src/ATen/mps/MPSAllocator.mm
  - aten/src/ATen/mps/MPSStream.h
  - aten/src/ATen/native/mps/OperationUtils.h
  - aten/src/ATen/native/mps/OperationUtils.mm
-->

## Summary

`ferrotorch-mps/src/backend.rs` is the macOS-only Apple Metal
implementation of `ferrotorch-core`'s `GpuBackend` dispatch trait. It
wraps a `Retained<dyn MTLDevice>` + command queue + 10 pre-compiled
`MTLComputePipelineState` handles, exposes the singleton lifecycle
`init_mps_backend_metal` that registers a global `MtlBackend` via
`ferrotorch_core::gpu_dispatch::register_gpu_backend`, and bridges the
type-erased `GpuBufferHandle` surface to `Arc<MtlBuffer>` storage on
shared-mode Metal buffers. Mirrors upstream's `at::mps::MPSDevice` +
`MPSAllocator` + `at::native::mps::runMPSGraph` role from
`aten/src/ATen/mps/MPSDevice.{h,mm}` + `aten/src/ATen/mps/MPSAllocator.{h,mm}` +
`aten/src/ATen/native/mps/OperationUtils.{h,mm}` collapsed into a
single Rust file because Sprint C.7 covers only 10 kernels.

## Requirements

- REQ-1: `pub struct MtlBackend` carrying a retained `MTLDevice`, a
  retained `MTLCommandQueue`, and a `Pipelines` struct of 10 compiled
  `MTLComputePipelineState` handles (matmul_f32, bmm_f32, add_f32,
  sub_f32, mul_f32, div_f32, relu_f32, sigmoid_f32, softmax_f32,
  sum_axis_f32). Implements
  `ferrotorch_core::gpu_dispatch::GpuBackend`. The entire module is
  `#![cfg(target_os = "macos")]`.

- REQ-2: `pub fn MtlBackend::new` — fail-fast construction that
  resolves `MTLCreateSystemDefaultDevice`, creates a command queue,
  and eagerly compiles all 10 MSL sources via the
  `compile_pipeline` helper. Any failure returns
  `FerrotorchError::DeviceUnavailable` (no device) or
  `FerrotorchError::InvalidArgument` (MSL compilation failed,
  indicating a ferrotorch bug). No lazy degrade-to-CPU path.

- REQ-3: `Arc<MtlBuffer>` buffer representation. `pub struct MtlBuffer`
  wraps `Retained<ProtocolObject<dyn MTLBuffer>>` plus an
  element-count field. Stored type-erased in
  `GpuBufferHandle::inner` via
  `Box::new(Arc::new(MtlBuffer { ... }))`; downcastable via
  `handle.downcast_ref::<Arc<MtlBuffer>>()`. The unsafe
  `Send + Sync` impls are documented with `// SAFETY:` comments noting
  that Metal serialises GPU buffer access through the command queue.

- REQ-4: Memory management trait surface. `cpu_to_gpu`, `gpu_to_cpu`,
  `clone_buffer`, `alloc_zeros`, `buffer_elem_size` — all using
  shared-mode buffers (CPU-accessible after `commit_and_wait`).
  Allocation goes through `MTLDevice::newBufferWithLength_options`
  with `MTLResourceOptions::StorageModeShared`. Buffers are zeroed
  on `alloc_zeros` for defence-in-depth even though Metal's runtime
  guarantees zero-init. Mirrors the
  `MPSAllocator::allocate_with_buffer` shared-mode path from
  `aten/src/ATen/mps/MPSAllocator.h:36-40` (`SHARED` usage flag).

- REQ-5: Elementwise / activation kernel surface. `add_f32`, `sub_f32`,
  `mul_f32`, `div_f32`, `relu_f32`, `sigmoid_f32` — all dispatch
  through `launch_binary_f32` / `launch_unary_f32` helpers with
  1-D launch geometry (one thread per element, `dispatchThreads`,
  threadgroup width capped at 256). Mirrors PyTorch's elementwise MPS
  kernel dispatch in `aten/src/ATen/native/mps/OperationUtils.mm`.

- REQ-6: Matmul kernel surface. `matmul_f32(a, b, m, k, n)` and
  `bmm_f32(a, b, batch, m, k, n)` dispatch the MSL GEMM / batched
  GEMM kernels with 2-D / 3-D launch geometry (`MTLSize { width: n,
  height: m[, depth: batch] }`, 16×16 threadgroups). Mirrors
  `torch.mm` / `torch.bmm` on MPS (`aten/src/ATen/native/mps/operations/LinearAlgebra.mm`
  upstream — we use a hand-written MSL GEMM rather than calling
  MPSGraph for Sprint C.7 simplicity).

- REQ-7: Reduction kernel surface. `softmax_f32(a, rows, cols)` and
  `sum_axis_f32(a, shape, axis)` (plus the convenience `sum_f32(a,
  len) = sum_axis_f32(a, &[len], 0)`) dispatch the MSL softmax /
  sum-axis kernels with one threadgroup per output row / element.
  The threadgroup width is computed by `pow2_tg_width` to satisfy the
  in-kernel `stride = tcount / 2; stride >>= 1` tree-reduction
  contract (issue #1101). Mirrors PyTorch's MPS softmax in
  `aten/src/ATen/native/mps/operations/SoftMax.mm` and sum in
  `aten/src/ATen/native/mps/operations/ReduceOps.mm`.

- REQ-8: `pow2_tg_width` threadgroup-width helper. Rounds the input
  up to the next power of two, capped at the Metal threadgroup limit
  of 1024 (`pow2_tg_width(0)=1`, `pow2_tg_width(13)=16`,
  `pow2_tg_width(1023)=1024`, `pow2_tg_width(2000)=1024`). The cap
  matches the documented Metal hardware limit; the round-up is
  required because the in-kernel tree reduction otherwise silently
  drops upper-half elements when the launch width is not pow-2. See
  #1101 for the bug this fixes.

- REQ-9: Unimplemented-kernel error contract. Every `GpuBackend`
  trait method without a Sprint C.7 MSL kernel returns
  `Err(FerrotorchError::InvalidArgument { message: "MSL kernel
  needed: <name> — follow-up #626" })` — matching PyTorch's
  `NotImplementedError` shape for unregistered backends. No silent
  CPU fallback (§3 of `rust-gpu-discipline`); no
  `todo!()`/`unimplemented!()` (R-APG-1). Follow-up issue #626 is the
  umbrella tracker.

- REQ-10: `pub fn init_mps_backend_metal` — idempotent global
  registration. Constructs `MtlBackend::new()` and registers it via
  `ferrotorch_core::gpu_dispatch::register_gpu_backend`. The only
  failure mode (besides `MtlBackend::new` errors) is that a backend
  is already registered, surfaced as `FerrotorchError::InvalidArgument`.

## Acceptance Criteria

- [x] AC-1: `pub struct MtlBackend in backend.rs` carrying
  `device`, `queue`, `pipelines: Pipelines` fields; `impl GpuBackend
  for MtlBackend in backend.rs` body present.
- [x] AC-2: `pub fn MtlBackend::new in backend.rs` compiles 10
  pipelines fail-fast; returns `Ok(Self)` only when all pipelines
  resolve.
- [x] AC-3: `pub struct MtlBuffer in backend.rs` with
  `inner: Retained<dyn MTLBuffer>` + `elem_count: usize`;
  `wrap_buffer` / `downcast_buf` helpers on `impl MtlBackend`.
- [x] AC-4: `cpu_to_gpu`, `gpu_to_cpu`, `clone_buffer`, `alloc_zeros`,
  `buffer_elem_size` trait method bodies in `impl GpuBackend for
  MtlBackend in backend.rs`.
- [x] AC-5: `add_f32`, `sub_f32`, `mul_f32`, `div_f32`, `relu_f32`,
  `sigmoid_f32` delegate to `launch_binary_f32` / `launch_unary_f32`.
- [x] AC-6: `matmul_f32`, `bmm_f32` build a command buffer +
  encoder, set 3 / 4 buffers + 3 / 4 setBytes scalars, dispatch with
  16×16 threadgroups.
- [x] AC-7: `softmax_f32`, `sum_axis_f32`, `sum_f32` use
  `pow2_tg_width(...)` for threadgroup-width and
  `dispatchThreadgroups` (not `dispatchThreads`) so each threadgroup
  handles one output row / element.
- [x] AC-8: `fn pow2_tg_width in backend.rs` is `n.min(1024).next_power_of_two()`;
  tests `pow2_tg_width_rounds_up_for_non_powers_of_two` and
  `pow2_tg_width_passes_through_powers_of_two` pass.
- [x] AC-9: All trait methods without Sprint C.7 MSL kernels return
  `Err(InvalidArgument { message: "MSL kernel needed: <op> — follow-up #626" })`.
- [x] AC-10: `pub fn init_mps_backend_metal in backend.rs` constructs
  `MtlBackend::new()` and registers via `register_gpu_backend`.

## Architecture

### Module gating

`#![cfg(target_os = "macos")]` at the top of `backend.rs` ensures the
entire module compiles only on macOS. Linux/WSL builds compile only
through `kernels` (raw MSL strings via `include_str!`) and the
platform-agnostic facade in `lib.rs`; the `objc2-metal` dependency is
never linked on non-Apple platforms.

### Type-erasure bridge (REQ-3)

`pub struct MtlBuffer in backend.rs` is the typed storage; it's
wrapped in `Arc<MtlBuffer>` so that `Send + Sync` is satisfied trivially
(Metal's ObjC ARC is thread-safe; access is serialised through the
command queue). The `wrap_buffer` helper on `impl MtlBackend` boxes the
`Arc<MtlBuffer>` into a `GpuBufferHandle` carrying
`(device_ordinal, elem_count, DType::F32)`. `downcast_buf` validates
the boxed type via `Any::downcast_ref::<Arc<MtlBuffer>>` and returns
`Err(InvalidArgument { message: "GpuBufferHandle does not contain an
Arc<MtlBuffer> (wrong backend?)" })` on a type mismatch — the
fast gate that catches an `Arc<CudaBuffer>` being read as
`Arc<MtlBuffer>`.

### Lifecycle (REQ-2, REQ-10)

`pub fn MtlBackend::new in backend.rs`:
1. `MTLCreateSystemDefaultDevice` → `Err(DeviceUnavailable)` on None.
2. `device.newCommandQueue()` → `Err(InvalidArgument)` on None.
3. Compile all 10 MSL pipelines via `fn compile_pipeline in backend.rs`
   (`newLibraryWithSource_options_error` → `newFunctionWithName` →
   `newComputePipelineStateWithFunction_error`).

`pub fn init_mps_backend_metal in backend.rs`:
1. `MtlBackend::new()?` — propagate `DeviceUnavailable` / compile
   errors.
2. `register_gpu_backend(Box::new(backend))` — the `OnceLock::set`
   inside `gpu_dispatch` fails only if a backend is already registered;
   we surface that as `FerrotorchError::InvalidArgument` with a
   diagnostic message (the rejected `Box<dyn GpuBackend>` doesn't
   implement Display so we can't include it in the message verbatim).

### Pipeline compilation (REQ-2)

`fn compile_pipeline in backend.rs` is the single MSL→`MTLComputePipelineState`
funnel. Failures at `newLibraryWithSource_options_error`,
`newFunctionWithName`, or `newComputePipelineStateWithFunction_error`
each map to `FerrotorchError::InvalidArgument` with a `format!`
diagnostic naming the failing kernel — these are ferrotorch bugs
(the MSL didn't compile), not user errors, so the error variant is
documented in the doc-comment as such.

### Kernel-launch helpers (REQ-5, REQ-6, REQ-7)

The launcher pattern is identical across all 10 kernels:
1. Downcast the input `GpuBufferHandle`s via `Self::downcast_buf`.
2. Allocate an output `Arc<MtlBuffer>` via `Self::alloc_buffer(byte_len, elem_count)`.
3. Create a `MTLCommandBuffer` via `self.queue.commandBuffer()`.
4. Create a `MTLComputeCommandEncoder` via `cmd_buf.computeCommandEncoder()`.
5. `enc.setComputePipelineState(&pipeline.state)`.
6. Set N `MTLBuffer` arguments via `enc.setBuffer_offset_atIndex(...)`.
7. Set M scalar `u32` arguments via `enc.setBytes_length_atIndex(...)`
   with the canonical `&n_u32 as *const u32 as *mut _` cast pattern
   (this is what justifies the crate-root `ref_as_ptr` /
   `borrow_as_ptr` allowances).
8. Compute grid + threadgroup size.
9. `dispatchThreads_threadsPerThreadgroup` (1-D / 2-D / 3-D ops) or
   `dispatchThreadgroups_threadsPerThreadgroup` (softmax /
   sum_axis where each tg handles one output row).
10. `enc.endEncoding()`.
11. `Self::commit_and_wait(&cmd_buf)` — sync dispatch so the output
    is CPU-readable immediately on shared-mode buffers.
12. `Self::wrap_buffer(out_buf, a.device_ordinal())`.

The `fn launch_binary_f32 in backend.rs` and `fn launch_unary_f32 in backend.rs`
factor out steps 1-11 for the 4 binary + 2 unary ops; the larger
matmul / bmm / softmax / sum_axis methods inline the same pattern
because their shape parameters and threadgroup geometry differ.

### Pow-2 threadgroup-width helper (REQ-8)

`fn pow2_tg_width in backend.rs` rounds up to the next power of two
with a 1024 cap. The contract is documented in the in-source comment
block above the function: the in-kernel `stride = tcount / 2; stride
>>= 1` tree reduction (in `softmax_f32.metal` and `sum_axis_f32.metal`)
requires `tcount` to be a power of two; on non-pow-2 inputs the
upper-half elements would be silently dropped (#1101). The
companion kernel-side fix is the strided-init pattern in the MSL
source that leaves inactive threads (`tid >= cols` /
`tid >= axis_len`) at identity sentinels (`-INFINITY` for max,
`0.0` for sum) so the reduction over the rounded-up width reads
identity elements where there is no real data.

### Unimplemented-kernel surface (REQ-9)

The Sprint C.7 surface implements 10 of the ~80 `GpuBackend` trait
methods. The remaining methods (broadcast_add_f32, transpose_2d_f32,
gelu_f32, dropout_f32, layernorm_f32, slice_write_f32, slice_read_f32,
embed_lookup_f32, scatter_add_*, scale_f32, relu_backward_f32,
gelu_backward_*, index_select_*, masked_fill_f32, masked_zero_f32,
has_inf_nan_f32, etc.) return
`Err(FerrotorchError::InvalidArgument { message: "MSL kernel needed:
<op> — follow-up #626" })`. This is the load-bearing contract: the
trait impl compiles cleanly (so `register_gpu_backend(Box::new(MtlBackend))`
succeeds), but invoking an unimplemented op surfaces a structured
error instead of a panic or silent fallback. PyTorch's MPS backend
follows the same shape — ops without an MPS dispatch raise
`NotImplementedError("the operator 'aten::xxx' is not currently
implemented for the MPS device")`.

### Non-test production consumers

- `pub fn init_mps_backend in lib.rs` calls
  `backend::init_mps_backend_metal()` on macOS.
- `pub use backend::MtlBackend in lib.rs` re-exports
  the type so application code on macOS can construct it directly
  (e.g. for tests that need a specific instance rather than the
  global).
- `ferrotorch/src/lib.rs:137` — `pub use ferrotorch_mps::*;` propagates
  `MtlBackend` + `init_mps_backend` to the meta-crate's public API.
- The trait impl itself (`impl GpuBackend for MtlBackend in
  backend.rs`) makes `MtlBackend` consumable wherever
  `&dyn ferrotorch_core::gpu_dispatch::GpuBackend` is wanted — after
  `init_mps_backend()` registers the global, every `Tensor::add`,
  `Tensor::matmul`, etc. in `ferrotorch-core` dispatches through it
  on macOS hosts via the type-erased trait.

## Parity contract

`parity_ops = []` for this route. Per-op numeric parity is the
responsibility of the kernel ops in `ferrotorch-core` (the parity-sweep
runner exercises them); this file is the dispatch layer that forwards
the type-erased `GpuBufferHandle` arguments to MSL pipelines.

Edge cases preserved across the dispatch layer:

- **Synchronous dispatch**: every kernel commits the command buffer
  and `waitUntilCompleted`s before returning. Callers can read the
  output via `gpu_to_cpu` immediately. A future async path is
  documented as `addScheduledHandler` + `addCompletedHandler` but
  Sprint C.7 keeps the synchronous contract for simplicity.
- **Shared-mode buffer semantics**: `cpu_to_gpu` copies via the
  CPU-accessible `MTLBuffer::contents()` pointer. `gpu_to_cpu` reads
  from the same pointer after `commit_and_wait`. No explicit
  `didModifyRange:` is needed for shared-mode buffers on Apple
  Silicon.
- **DType tag is fixed at F32**: `Self::wrap_buffer` always tags
  `DType::F32` and `buffer_elem_size` returns 4. Adding bf16 / f16
  support requires a second pipeline family + dtype-aware wrap
  helpers (out of scope for Sprint C.7).
- **Single-device contract**: `device_ordinal()` is preserved across
  ops but no `MpsDevice` validation is done at the kernel boundary
  (Apple Silicon is single-device and `MpsDevice::new` already
  rejected non-zero ordinals upstream).
- **Pow-2 threadgroup**: `softmax_f32` and `sum_axis_f32` round the
  threadgroup width up to next-pow-2 via `pow2_tg_width`. The
  kernel side handles inactive threads with sentinel-identity values.
- **MSL compile fail-fast**: any of the 10 MSL kernels failing to
  compile in `MtlBackend::new()` returns `Err(InvalidArgument)`
  immediately — no partial-pipeline backend can be returned.
- **Idempotent registration**: `init_mps_backend_metal` calling
  `register_gpu_backend` twice produces
  `Err(InvalidArgument { message: "MPS backend registration failed:
  a GPU backend is already registered" })` rather than overwriting
  the registered backend.
- **Unimplemented op signal**: every non-Sprint-C.7 trait method
  returns the structured "MSL kernel needed" error so callers get a
  textual diagnostic naming the missing op + the follow-up issue #626.

## Verification

Unit tests in `mod tests in backend.rs` (4 tests):

- `pow2_tg_width_rounds_up_for_non_powers_of_two` — exercises
  `pow2_tg_width(0)=1`, `(13)=16`, `(257)=512`, `(1023)=1024`,
  `(2000)=1024`.
- `pow2_tg_width_passes_through_powers_of_two` — pow-2 inputs
  round-trip unchanged.
- `mtl_backend_new_succeeds_or_unavailable` — `MtlBackend::new()`
  either returns `Ok` or `Err(DeviceUnavailable)`; never panics.
- `mtl_buffer_round_trip` — `cpu_to_gpu` → `gpu_to_cpu` round-trips
  4 f32 values exactly; `cascade_skip` on hosts without a Metal
  device.

The integration suite in `ferrotorch-mps/tests/conformance_mps.rs`
exercises:

- All 6 kernel-source presence tests (compile-platform-agnostic).
- 11 live-MPS cascade_skip tests (Apple-only).
- 3 #1101 regression tests for `softmax_f32` / `sum_axis_f32` at
  non-pow-2 widths.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-mps --no-default-features 2>&1 | tail -5
```

Expected: ≥ 5 `test result: ok` lines across the lib unit tests +
conformance tests + doctests.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct MtlBackend in backend.rs` with `Pipelines` field of 10 `MTLComputePipelineState` slots; `impl GpuBackend for MtlBackend in backend.rs`; module-level `#![cfg(target_os = "macos")]`; non-test consumer: `pub use backend::MtlBackend in lib.rs` re-exports it, and `ferrotorch/src/lib.rs:137` `pub use ferrotorch_mps::*;` propagates it to applications on macOS. |
| REQ-2 | SHIPPED | impl: `pub fn MtlBackend::new in backend.rs` runs `MTLCreateSystemDefaultDevice` then compiles all 10 pipelines via `fn compile_pipeline in backend.rs` (matmul_f32, bmm_f32, add/sub/mul/div_f32, relu/sigmoid_f32, softmax_f32, sum_axis_f32); non-test consumer: `pub fn init_mps_backend_metal in backend.rs` invokes `MtlBackend::new()?` and propagates the error. |
| REQ-3 | SHIPPED | impl: `pub struct MtlBuffer in backend.rs` + `unsafe impl Send/Sync for MtlBuffer` with SAFETY comments; `fn wrap_buffer in backend.rs` and `fn downcast_buf in backend.rs` on `impl MtlBackend`; non-test consumer: every `impl GpuBackend` trait method body that reads or writes a buffer (e.g. `add_f32` → `launch_binary_f32` → `Self::downcast_buf(a)?`). |
| REQ-4 | SHIPPED | impl: `fn cpu_to_gpu`, `fn gpu_to_cpu`, `fn clone_buffer`, `fn alloc_zeros`, `fn buffer_elem_size` in `impl GpuBackend for MtlBackend in backend.rs`, all using `MTLResourceOptions::StorageModeShared` per `aten/src/ATen/mps/MPSAllocator.h:36`; non-test consumer: every kernel launcher calls `Self::alloc_buffer` for the output (e.g. `fn launch_binary_f32 in backend.rs` allocates the output buffer before encoding). |
| REQ-5 | SHIPPED | impl: `fn add_f32`, `fn sub_f32`, `fn mul_f32`, `fn div_f32`, `fn relu_f32`, `fn sigmoid_f32` in `impl GpuBackend for MtlBackend in backend.rs`, all delegating to `fn launch_binary_f32 in backend.rs` / `fn launch_unary_f32 in backend.rs`; non-test consumer: `ferrotorch_core::gpu_dispatch::gpu_backend()` returns this `&dyn GpuBackend` on macOS post-`init_mps_backend()` and ferrotorch-core's `Tensor::add` / etc. dispatch through it. |
| REQ-6 | SHIPPED | impl: `fn matmul_f32 in backend.rs` and `fn bmm_f32 in backend.rs` inline the command-buffer + encoder + setBuffer + setBytes pattern, dispatching with 16×16 threadgroups over a `(n, m[, batch])` grid; non-test consumer: same trait dispatch path through `gpu_backend()` for `Tensor::matmul` / `Tensor::bmm` on macOS. |
| REQ-7 | SHIPPED | impl: `fn softmax_f32 in backend.rs`, `fn sum_f32 in backend.rs`, `fn sum_axis_f32 in backend.rs` with `dispatchThreadgroups_threadsPerThreadgroup` (one threadgroup per output row / element) and `pow2_tg_width(cols)` / `pow2_tg_width(axis_len)` for the threadgroup width; non-test consumer: trait dispatch path through `gpu_backend()` for `Tensor::softmax` / `Tensor::sum`. |
| REQ-8 | SHIPPED | impl: `fn pow2_tg_width in backend.rs` = `n.min(1024).next_power_of_two()` with the documented #1101 contract above the function; non-test consumer: `fn softmax_f32 in backend.rs` and `fn sum_axis_f32 in backend.rs` invoke it for their threadgroup-width parameter, ensuring the in-kernel tree reduction sees a pow-2 width even at non-pow-2 `cols`/`axis_len`. |
| REQ-9 | SHIPPED | impl: every non-Sprint-C.7 trait method in `impl GpuBackend for MtlBackend in backend.rs` returns `Err(FerrotorchError::InvalidArgument { message: "MSL kernel needed: <op> — follow-up #626" })` (e.g. `neg_f32`, `gelu_f32`, `dropout_f32`, `broadcast_add_f32`, `transpose_2d_f32`, `permute_0213_f32`, `layernorm_f32`, `slice_write_f32`, `slice_read_f32`, `embed_lookup_f32`, `scale_f32`, `relu_backward_f32`, `gelu_backward_f32`, `index_select_1d_f32`, `masked_fill_f32`, `has_inf_nan_f32`); non-test consumer: trait impl compiles so `register_gpu_backend(Box::new(MtlBackend))` succeeds; invoking an unimplemented op surfaces the structured error to ferrotorch-core callers. |
| REQ-10 | SHIPPED | impl: `pub fn init_mps_backend_metal in backend.rs` constructs `MtlBackend::new()` and calls `ferrotorch_core::gpu_dispatch::register_gpu_backend(Box::new(backend))`; non-test consumer: `pub fn init_mps_backend in lib.rs` delegates here on macOS, and `pub use ferrotorch_mps::* in ferrotorch/src/lib.rs` glob re-exports it as `ferrotorch::init_mps_backend()`. |
