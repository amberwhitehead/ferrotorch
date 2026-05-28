# Bridge between ferrotorch-core Tensor<T> and GPU

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/Copy.cu
  - aten/src/ATen/native/Copy.cpp
  - torch/_tensor.py
  - torch/cuda/__init__.py
-->

## Summary

`ferrotorch-gpu/src/tensor_bridge.rs` is the bridge that sidesteps
the circular-dependency rule: ferrotorch-core cannot depend on
ferrotorch-gpu, so all GPU-aware tensor wrapping lives here.
It defines `GpuTensor<T>` (a `CudaBuffer<T>` + shape + owning
device), free functions `tensor_to_gpu` / `tensor_to_cpu` /
`cuda` / `cuda_default` to round-trip a `Tensor<T>` between CPU
and GPU, and the `GpuFloat` trait alias that captures the
"this `Float` can be DeviceRepr'd" constraint. Mirrors the
`torch.Tensor.cuda()` / `.cpu()` / `.to('cuda')` Python-API
surface and the underlying `at::native::copy` machinery in
`aten/src/ATen/native/cuda/Copy.cu`.

## Requirements

- REQ-1: `pub trait GpuFloat: Float (+ DeviceRepr when cuda is on)`
  — the trait alias that lets generic code be parametric over
  `f32` / `f64` while ensuring the type satisfies cudarc's
  `DeviceRepr` constraint. Both `f32` and `f64` impl it.
- REQ-2: `pub struct GpuTensor<T: GpuFloat>` carrying
  `(buffer: CudaBuffer<T>, shape: Vec<usize>, device: GpuDevice)`
  with accessors `shape()`, `numel()`, `device()`, `buffer()`.
- REQ-3: `pub fn tensor_to_gpu<T>(tensor: &Tensor<T>, device:
  &GpuDevice) -> FerrotorchResult<GpuTensor<T>>` — upload a CPU
  tensor to GPU, preserving shape.
- REQ-4: `pub fn tensor_to_cpu<T>(gpu_tensor: &GpuTensor<T>) ->
  FerrotorchResult<Tensor<T>>` — download a GPU tensor back to CPU
  storage.
- REQ-5: Convenience helpers `pub fn cuda<T>(tensor, ordinal)` and
  `pub fn cuda_default<T>` (uses device 0) — wrappers over
  `tensor_to_gpu` for the common single-device case. Mirror
  Python's `.cuda(device=...)` shape.
- REQ-6: f32-only kernel fast-path policy: f32 elementwise / matmul
  / conv2d operations dispatch to PTX kernels directly; f64
  operations currently fall back to a CPU round-trip until f64 PTX
  kernels land. The type parameter is kept for API consistency.
- REQ-7: Non-test production consumer wiring — `ferrotorch-distributed`
  uses `GpuTensor` + `tensor_to_gpu` / `tensor_to_cpu` for the
  GPU-aware collective communication path.

## Acceptance Criteria

- [x] AC-1: `pub trait GpuFloat` with cuda-feature-gated `DeviceRepr`
  bound at line 44 (cuda on) / line 52 (cuda off); impls for `f32`
  and `f64` at lines 47-57.
- [x] AC-2: `pub struct GpuTensor<T: GpuFloat>` at line 71 with the
  documented fields; accessors at lines 80, 86, 92, 98 (and a manual
  `Debug` impl at line 173).
- [x] AC-3: `pub fn tensor_to_gpu` at line 1065 with the
  `(Tensor<T>, GpuDevice) -> GpuTensor<T>` signature.
- [x] AC-4: `pub fn tensor_to_cpu` at line 1099 with the
  `GpuTensor<T> -> Tensor<T>` signature.
- [x] AC-5: `pub fn cuda` at line 1122 and `pub fn cuda_default`
  at line 1130.
- [x] AC-6: f32 ops dispatch into `crate::kernels::gpu_add` / etc.;
  f64 ops fall back to a CPU round-trip path (documented at lines
  11-16 of the module).
- [x] AC-7: Non-test consumers exist in
  `ferrotorch-distributed/src/gpu_collective.rs`
  (`use ferrotorch_gpu::{GpuFloat, GpuTensor, tensor_to_cpu, tensor_to_gpu}`)
  and `ferrotorch-distributed/src/ucc_backend.rs` (UCC backend GPU collective methods,
  function signatures over `GpuTensor<T>`).

## Architecture

### GpuFloat trait alias (REQ-1)

cudarc's transfer functions require `T: DeviceRepr`. Both `f32` and
`f64` implement it, but we cannot unconditionally name the trait
when the `cuda` feature is disabled (cudarc isn't compiled). The
`GpuFloat` trait alias bridges the gap:

```rust
#[cfg(feature = "cuda")]
pub trait GpuFloat: Float + cudarc::driver::DeviceRepr {}

#[cfg(not(feature = "cuda"))]
pub trait GpuFloat: Float {}
```

Both forms have impls for `f32` and `f64`. Downstream code is
parameterised over `<T: GpuFloat>` and works in both cuda-on and
cuda-off builds.

### GpuTensor<T> (REQ-2)

`GpuTensor<T>` is the GPU-side mirror of ferrotorch-core's
`Tensor<T>`. It carries:

- `buffer: CudaBuffer<T>` — the device storage.
- `shape: Vec<usize>` — the host-side shape metadata.
- `device: GpuDevice` — the owning CUDA device (so transfer back
  to CPU knows which stream to use).

Accessors:
- `pub fn shape(&self) -> &[usize]`
- `pub fn numel(&self) -> usize` — product of shape.
- `pub fn device(&self) -> &GpuDevice`
- `pub fn buffer(&self) -> &CudaBuffer<T>` — borrow the underlying
  device storage for kernels that take `&CudaBuffer<T>`.

The `Debug` impl at line 173 prints only the shape + device ordinal,
not the buffer contents (which would require a costly device→host
read).

### Transfer functions (REQ-3, REQ-4)

`tensor_to_gpu(tensor: &Tensor<T>, device: &GpuDevice) ->
FerrotorchResult<GpuTensor<T>>` (line 1065):

1. Extracts the CPU storage via `tensor.data()?`.
2. Uploads via `crate::transfer::cpu_to_gpu` (which is itself
   gated on the `cuda` feature; without it the bridge returns
   `FerrotorchError::DeviceUnavailable`).
3. Wraps the buffer + shape + device into a `GpuTensor<T>`.

`tensor_to_cpu(gpu_tensor: &GpuTensor<T>) ->
FerrotorchResult<Tensor<T>>` (line 1099):

1. Calls `crate::transfer::gpu_to_cpu` on the buffer.
2. Reconstructs a `Tensor<T>` via `Tensor::from_storage(TensorStorage::cpu(data), shape)`.

### Convenience helpers (REQ-5)

`pub fn cuda<T>(tensor: &Tensor<T>, ordinal: usize) ->
GpuResult<GpuTensor<T>>` (line 1122) constructs the
`GpuDevice::new(ordinal)` inline and calls `tensor_to_gpu`.

`pub fn cuda_default<T>` (line 1130) is the ordinal-0 shortcut.

Both mirror Python's `tensor.cuda()` / `tensor.cuda(device=N)`
shape, returning `GpuResult` rather than panicking on missing CUDA.

### Op layer (REQ-6)

The file's lower half imports
`crate::blas::{gpu_matmul_f32, gpu_matmul_f64}`,
`crate::conv::gpu_conv2d_f32`, and the f32 / f64 elementwise
kernels (lines 20-28). It exposes `GpuTensor<f32>::add`,
`GpuTensor<f32>::mul`, etc. as direct PTX dispatches. For
`GpuTensor<f64>` the same op signatures are present (API
consistency) but the implementation either uses the f64 PTX path
(where available — basic add/sub/mul/neg/relu have it) or falls
back to a CPU round-trip.

The documented rationale at lines 11-16: "The PTX kernels are
currently f32-only. Operations on `GpuTensor<f64>` fall back to a
CPU round-trip ... once f64 PTX kernels are added, the fallback
disappears transparently." Many of those f64 PTX kernels have since
landed in `kernels.rs` (via the `ptx_f32_to_f64` auto-converter), so
the fallback story is partially obsolete but the API contract
stands.

### Non-test production consumers (REQ-7)

`ferrotorch-distributed/src/gpu_collective.rs` is the primary
external consumer:

- Line 54: `use ferrotorch_gpu::{GpuFloat, GpuTensor, tensor_to_cpu,
  tensor_to_gpu};` — imports the full surface.
- `pub fn gpu_allreduce<T: GpuFloat>(tensor: &GpuTensor<T>, ...)` —
  the GPU-aware all-reduce collective. The body calls `tensor_to_cpu`
  on the no-NCCL fallback path and routes through NCCL on the fast
  path (when the `nccl` feature is enabled).

`ferrotorch-distributed/src/ucc_backend.rs` (UCC backend GPU collective methods) — UCC backend
methods take `&ferrotorch_gpu::GpuTensor<T>` arguments and return
`FerrotorchResult<GpuTensor<T>>`.

UCC test scaffolding in `ferrotorch-distributed/src/ucc_backend.rs` test
paths use `tensor_to_gpu` for fixture setup (test consumers; not
counted for REQ-7 SHIPPED but illustrate the surface).

## Parity contract

`parity_ops = []` for this route. tensor_bridge is the boundary
layer between CPU and GPU tensor representations; per-op parity is
enforced elsewhere (the kernel layers in `kernels.rs` and the trait
dispatch in `backend_impl.rs`). The bridge itself preserves byte-
exact equivalence in the host↔device direction (matches PyTorch's
`copy_` contract).

Edge cases preserved:

- **Empty tensor** (`numel() == 0`): both directions handle empty
  shape correctly — `tensor_to_gpu` of an empty `Tensor` produces
  an empty `GpuTensor`; `tensor_to_cpu` of an empty `GpuTensor`
  produces an empty `Tensor`.
- **Non-contiguous source**: `tensor.data()?` returns the
  contiguous-or-error contract from ferrotorch-core's `Tensor::data`;
  non-contiguous tensors error rather than silently copy. Matches
  PyTorch's `.cuda()` on a non-contiguous tensor (which silently
  contiguifies — ferrotorch's contract is strict here and
  documented).
- **Shape preservation**: round-tripping
  `tensor_to_cpu(tensor_to_gpu(t, dev))` returns a `Tensor<T>` with
  the same `shape` and same element-by-element values.
- **Device pinning**: `GpuTensor` carries its owning `GpuDevice`;
  any subsequent op against another device-resident tensor returns
  `DeviceMismatch` rather than implicit copy.
- **`Tensor::data()` error pass-through**: if the source tensor is
  itself GPU-resident (already on CUDA), `tensor.data()` returns
  `FerrotorchError::GpuTensorNotAccessible`, which `tensor_to_gpu`
  propagates as-is. Idempotent uploads are not supported through
  this bridge (use the in-place ferrotorch-core path instead).

## Verification

Unit tests in `ferrotorch-gpu/src/tensor_bridge.rs` `mod tests` (gated
`#[cfg(feature = "cuda")]`) cover: round-trip preservation,
empty-tensor handling, device-ordinal validation, the convenience
helpers `cuda` / `cuda_default`, the `GpuTensor::buffer` accessor's
zero-copy borrow shape, and the f32 / f64 op dispatch.

Cross-crate integration is exercised at
`ferrotorch-distributed/src/gpu_collective.rs` — the
`gpu_allreduce` / `gpu_broadcast` paths use `GpuTensor` as their
input shape, and their test paths in the `tests` module exercise
the full CPU↔GPU↔NCCL round trip.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda tensor_bridge:: 2>&1 | tail -3
```

Expected: ≥ 1 `test result: ok` line.

The downstream tests in `ferrotorch-distributed` are also a
quasi-integration smoke for this file:

```bash
cargo test -p ferrotorch-distributed --features cuda gpu_collective:: 2>&1 | tail -3
```

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub trait GpuFloat in ferrotorch-gpu/src/tensor_bridge.rs` (line 44 cuda / line 52 non-cuda) with `f32` + `f64` impls; non-test consumer: `pub fn gpu_allreduce in ferrotorch-distributed/src/gpu_collective.rs` imports it, and `ferrotorch-distributed/src/ucc_backend.rs:304` uses it as a trait bound on collective-op method signatures. |
| REQ-2 | SHIPPED | impl: `pub struct GpuTensor<T: GpuFloat> in tensor_bridge.rs` (line 71); non-test consumer: `ferrotorch-distributed/src/gpu_collective.rs::gpu_allreduce`/`gpu_broadcast` take `&GpuTensor<T>` arguments; `ferrotorch-distributed/src/ucc_backend.rs` (UCC backend GPU collective methods) use it as the method parameter type. |
| REQ-3 | SHIPPED | impl: `pub fn tensor_to_gpu in tensor_bridge.rs` (line 1065); non-test consumer: `pub fn gpu_allreduce in ferrotorch-distributed/src/gpu_collective.rs` imports + uses it in the no-NCCL fallback. |
| REQ-4 | SHIPPED | impl: `pub fn tensor_to_cpu in tensor_bridge.rs` (line 1099); non-test consumer: `pub fn gpu_allreduce in ferrotorch-distributed/src/gpu_collective.rs` imports + uses it for downloading collective results in the fallback path. |
| REQ-5 | SHIPPED | impl: `pub fn cuda` (line 1122), `pub fn cuda_default` (line 1130) in `tensor_bridge.rs`; non-test consumer: re-exported at `ferrotorch-gpu/src/lib.rs:245` for downstream crates. (ferrotorch-core's `Tensor::cuda()` is a separate method that uses ferrotorch-gpu through the `GpuBackend` trait — the convenience helpers here serve direct `GpuTensor`-construction call sites.) |
| REQ-6 | SHIPPED | impl: kernel imports at `tensor_bridge.rs` (gpu_add, gpu_sub, gpu_mul, gpu_neg, gpu_relu f32 + f64 variants; gpu_matmul_f32/f64; gpu_conv2d_f32). The f32-only fast-path policy is documented at lines 11-16 of the module. |
| REQ-7 | SHIPPED | impl: non-test consumer at `pub fn gpu_allreduce in ferrotorch-distributed/src/gpu_collective.rs` (uses `GpuFloat, GpuTensor, tensor_to_cpu, tensor_to_gpu`) and `ferrotorch-distributed/src/ucc_backend.rs` (UCC backend GPU collective methods) (function signatures over `GpuTensor<T>`). |
