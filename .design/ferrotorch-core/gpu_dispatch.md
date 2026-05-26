# GPU backend dispatch

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/Dispatch.h
  - aten/src/ATen/core/dispatch/Dispatcher.h
  - aten/src/ATen/cuda/CUDAContext.h
  - c10/cuda/CUDAFunctions.h
  - c10/core/ScalarType.h
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/gpu_dispatch.rs` defines the `GpuBackend` trait —
the contract that any GPU crate (`ferrotorch-gpu`, `ferrotorch-cubecl`,
…) implements and registers globally via `register_gpu_backend`. The
trait is the seam that lets `ferrotorch-core` dispatch onto CUDA
without depending on `ferrotorch-gpu`. Mirrors the `at::DispatchStub`
+ `at::native::DispatchStub::register_backend` pattern in
`aten/src/ATen/Dispatch.h` and `aten/src/ATen/native/DispatchStub.h`.

The file is ~4.3k LOC and exposes:

- `enum CompareOp` (eq/ne/lt/le/gt/ge) — comparison-op selector.
- `struct GpuRngState` — serializable RNG snapshot for checkpoints.
- `struct GpuBufferHandle` — type-erased device buffer
  (`Box<dyn Any + Send + Sync>`) tagged with `(device_ordinal, len,
  dtype)`. The `dtype` field is the authoritative `ScalarType` analog
  (PyTorch parity — bf16 vs f16 differ only by tag, not by byte width).
- `trait GpuBackend` — ~350 method slots covering elementwise math
  (f16/bf16/f32/f64), reductions, linalg (matmul / GEMM / TRSM /
  cusolver), conv2d/conv3d, RNN/LSTM/GRU, FFT, dropout, indexing,
  cuSPARSE, RNG, profiling primitives, and integer / boolean ops.
- `register_gpu_backend(backend)` / `gpu_backend()` /
  `has_gpu_backend()` — global registration plus `OnceLock`-backed
  accessor.

## Requirements

- REQ-1: `enum CompareOp` — six elementwise comparison operators with
  a stable `suffix()` for kernel-name dispatch. Mirrors the
  comparison kernels behind `torch.eq` / `ne` / `lt` / `le` / `gt`
  / `ge`.
- REQ-2: `struct GpuRngState` — `(counter, seed, offset, device)`
  RNG snapshot with public accessors and crate-private fields. Used
  by checkpoint save/restore. Mirrors
  `c10::cuda::CUDAGeneratorImpl::Philox4_32_10`'s state.
- REQ-3: `struct GpuBufferHandle` — `(inner: Box<dyn Any + Send + Sync>,
  device_ordinal, len, dtype)`. Accessors: `device_ordinal`, `dtype`,
  `len`, `is_empty`, `downcast_ref`, `downcast_mut`, `into_inner`.
  The `dtype` field is the authoritative tag (per the documented
  PyTorch-parity rationale at `gpu_dispatch.rs:128-141`).
- REQ-4: `trait GpuBackend: Send + Sync` — the canonical dispatch
  surface. `as_any()` for backend-specific downcast,
  `cpu_to_gpu(data, dtype, device)` for H2D,
  `gpu_to_cpu(handle)` for D2H, `alloc_zeros(len, dtype, device)`
  for initialised device allocation. Mirrors
  `at::cuda::CUDABlas`'s upload/download surface plus the
  per-backend kernel registry.
- REQ-5: Elementwise math kernels — `add_*`, `sub_*`, `mul_*`,
  `div_*`, `pow_*`, `neg_*`, `abs_*`, `sqrt_*`, `relu_*`, `tanh_*`,
  `sigmoid_*`, `gelu_*`, `silu_*` for f32 and f64 (with selected
  bf16/f16 paths). Each dispatches to a PTX kernel; the trait method
  signature `(a, b) -> handle` returns a fresh GPU buffer.
- REQ-6: Broadcasting variants — `broadcast_add_*`,
  `broadcast_mul_*`, etc. Take per-input strides + the output shape;
  no host-side materialisation.
- REQ-7: `scale_*` family — multiply a tensor by an f64 scalar in
  place (no Python "alpha * tensor" temporary).
- REQ-8: Strided copy / scatter — `strided_copy_{f32,f64}`,
  `strided_scatter_{f32,f64}`. Take `(handle, shape, stride,
  offset)` and produce a contiguous device buffer (copy) or
  overwrite specified positions (scatter). The strided_copy kernels
  are the substrate for `as_strided_copy` on CUDA
  (`stride_tricks.rs:406-414`), for non-contiguous CUDA→CPU
  materialise (`tensor.rs:872-880`), and for memory-format permute
  (`tensor.rs:1585-1602`).
- REQ-9: Reductions — `sum_axis_*`, `mean_axis_*`, `max_axis_*`,
  `min_axis_*`, `prod_axis_*`, plus full-tensor variants. Carry the
  reduction axis (or `None` for all).
- REQ-10: Linalg — `matmul_*`, `bmm_*`, `gemm_*`, `addmm_*`,
  `trsm_*`, `syevd_*` (cuSOLVER symmetric eigendecomposition),
  `getrf_*` (LU), `geqrf_*` (QR), `potrf_*` (Cholesky), `gesdd_*`
  (SVD), `inverse_*`. Most CUDA decompositions route through
  cuSOLVER per the include comment.
- REQ-11: Convolution + pooling — `conv2d_*`, `conv3d_*`, plus
  `max_pool2d`, `avg_pool2d`, etc. cuDNN-backed when available.
- REQ-12: Recurrent layers — `lstm_*`, `gru_*`, `rnn_tanh_*`,
  `rnn_relu_*`. cuDNN-backed.
- REQ-13: FFT — `fft_*`, `ifft_*`, `rfft_*`, `irfft_*`. cuFFT-backed.
- REQ-14: Dropout / RNG — `dropout_*` with a per-call mask
  (Philox-based), `normal_*`, `uniform_*`, `randint_*`,
  `bernoulli_*`. `save_rng_state` / `restore_rng_state` for
  checkpoint integration.
- REQ-15: Indexing — `index_select_intidx`, `gather_intidx`,
  `masked_fill_*`, `masked_select`, `masked_scatter`, `where_cond`.
  GPU-resident; `masked_select` returns the compacted output and the
  output-length integer (the only host crossing).
- REQ-16: Sparse — `dense_to_sparse_csr_{f32,f64}`,
  `sparse_csr_to_dense_*`, `csr_spmm_*`. Wrappers around cuSPARSE
  `cusparseDenseToSparse` / `cusparseSpMM`.
- REQ-17: Integer ops — `int_add`, `int_sub`, `int_mul`, `int_neg`,
  `int_floor_div`, `int_remainder`, bitwise ops, shifts, integer
  reductions, plus integer↔float / integer↔integer casts. Mostly
  default-`Err(NotImplementedOnCuda)` slots that concrete backends
  can override.
- REQ-18: Boolean ops — `compare`, `bool_and`, `bool_or`, `bool_xor`,
  `bool_not`, `bool_any`, `bool_all`, `cast_bool_to_f`,
  `cast_f_to_i`, `cast_i_to_f`, `cast_i_to_i`.
- REQ-19: Stream + sync — `synchronize`, `stream_count`,
  `strided_cat`. Default-impl methods that backends override when
  they expose multi-stream / fused-cat capability.
- REQ-20: Global registration — `register_gpu_backend(backend)`
  returns `Result<(), Box<dyn GpuBackend>>` (errors if already
  registered, returning the un-installed backend back to the caller).
  `gpu_backend() -> Option<&'static dyn GpuBackend>`. `has_gpu_backend()
  -> bool`. The registry is a single `OnceLock<Box<dyn GpuBackend>>`.

## Acceptance Criteria

- [x] AC-1: `GpuBufferHandle::new(_, _, _, dtype)` carries the dtype
  tag through `dtype()` accessor (`gpu_dispatch.rs:168-171`).
- [x] AC-2: `CompareOp::Lt.suffix() == "lt"`
  (`gpu_dispatch.rs:41-50`).
- [x] AC-3: `GpuRngState::new(counter, seed, offset, device)`
  preserves all four fields through getters
  (`gpu_dispatch.rs:84-119`).
- [x] AC-4: `register_gpu_backend(b1)` succeeds; a second
  `register_gpu_backend(b2)` returns `Err(b2)`
  (`gpu_dispatch.rs:4268-4271`).
- [x] AC-5: `has_gpu_backend()` toggles with registration
  (`gpu_dispatch.rs:4278-4280`).
- [x] AC-6: A concrete backend implementing the trait (i.e.
  `ferrotorch-gpu::CudaBackendImpl`) registers successfully and
  serves dispatch calls.
- [x] AC-7: `cargo test -p ferrotorch-core --lib gpu_dispatch`
  passes.

## Architecture

The file is organised as:

- **Lines 7-51**: `CompareOp` + `suffix()`.
- **Lines 53-120**: `GpuRngState` + accessors.
- **Lines 122-206**: `GpuBufferHandle` + downcast methods + Debug.
- **Lines 208-4254**: `trait GpuBackend` — ~350 method slots. Most
  carry default impls that return `Err(NotImplementedOnCuda)` so
  concrete backends can override only what they support; the core
  methods (`cpu_to_gpu`, `gpu_to_cpu`, `clone_buffer`, `alloc_zeros`,
  `add_f32`, `sub_f32`, `mul_f32`, `neg_f32`, `relu_f32`,
  `matmul_f32`) are unimplemented-by-default and MUST be provided.
- **Lines 4266-4282**: `register_gpu_backend`, `gpu_backend`,
  `has_gpu_backend` — the registration plumbing built on a single
  `OnceLock<Box<dyn GpuBackend>>`.
- **Lines 4286-4300**: in-file test mod (handle construction +
  debug formatting).

Non-test production consumers:

- `ferrotorch-gpu/src/backend_impl.rs:7037` (`impl GpuBackend for
  CudaBackendImpl` + the `register_gpu_backend(...)` call at
  `:7011`) is the canonical concrete implementation.
- Inside `ferrotorch-core`, every CUDA-dispatched op routes through
  `gpu_backend()` — see e.g. `tensor.rs:803`, `tensor.rs:834`,
  `tensor.rs:1213`, `tensor.rs:1576`, `stride_tricks.rs:397`,
  `stride_tricks.rs:440`, `storage.rs:153`, `storage.rs:404`,
  `storage.rs:451`, `grad_fns/arithmetic.rs` CUDA branches.

## Parity contract

`parity_ops = []`. The `GpuBackend` trait is the dispatch substrate;
parity is enforced at the op level (the parity sweeps for `add`,
`sub`, `mul`, etc. each exercise the registered backend on CUDA).
The trait contract itself — that the methods accept tagged
`GpuBufferHandle`s and produce tagged outputs of the right
device/dtype/shape — is asserted by the concrete `CudaBackendImpl`
tests in `ferrotorch-gpu`.

The PyTorch-parity invariants pinned by this module:

- The `dtype` tag is authoritative (R-CITE comment at
  `gpu_dispatch.rs:128-141`).
- Byte width never determines element type (bf16 vs f16
  disambiguated by tag).
- Comparison ops return `DType::Bool`-tagged (u8 0/1) regardless of
  input dtype.

## Verification

```bash
cargo test -p ferrotorch-core --lib gpu_dispatch
```

Expected: in-file tests pass (handle, debug formatting). End-to-end
parity is the responsibility of the parity-sweep runner, which
indirectly exercises every backend method through the per-op sweeps.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `enum CompareOp` at `ferrotorch-core/src/gpu_dispatch.rs:22-35` with `suffix()` at `:41-50`; non-test consumer: `GpuBackend::compare(_, _, CompareOp)` method slot dispatches to the suffix-named PTX kernel; the concrete impl in `ferrotorch-gpu/src/backend_impl.rs` reads `op.suffix()` to pick the kernel. |
| REQ-2 | SHIPPED | impl: `GpuRngState` at `ferrotorch-core/src/gpu_dispatch.rs:69` with accessors at `:84-119`; non-test consumer: `GpuBackend::save_rng_state` / `restore_rng_state` at `:2268, :2275` produce / consume this struct; `ferrotorch-core::checkpoint` (downstream) serialises it. |
| REQ-3 | SHIPPED | impl: `GpuBufferHandle` at `ferrotorch-core/src/gpu_dispatch.rs:137` with `new` at `:145`, accessors at `:160-194`; non-test consumer: `TensorStorage::Gpu(GpuBufferHandle)` variant at `storage.rs:67` plus every CUDA op that reads / writes the handle. |
| REQ-4 | SHIPPED | impl: `trait GpuBackend` at `ferrotorch-core/src/gpu_dispatch.rs:211`; non-test consumer: `ferrotorch-gpu/src/backend_impl.rs:7037`'s `impl GpuBackend for CudaBackendImpl` registers via `register_gpu_backend(...)`. |
| REQ-5 | SHIPPED | impl: elementwise method slots `add_f32` at `gpu_dispatch.rs:274`, `sub_f32` at `:279`, `mul_f32` at `:284`, `neg_f32` at `:289`, `relu_f32` at `:290` (plus the rest of the elementwise zoo throughout the file); non-test consumer: `Tensor::accumulate_grad` GPU path at `tensor.rs:588-592` calls `backend.add_f32` / `add_f64`; `grad_fns/arithmetic.rs::add_inner` dispatches the CUDA branches. |
| REQ-6 | SHIPPED | impl: broadcast variants are sibling slots on the trait alongside the non-broadcast methods (e.g. `broadcast_add_f32`); non-test consumer: `grad_fns/arithmetic.rs::add_inner` picks the broadcast variant via the dispatch macro when input shapes differ. |
| REQ-7 | SHIPPED | impl: `scale_*` trait slots in the same elementwise section; non-test consumer: `grad_fns/arithmetic.rs::scale_tensor` (`:547-578`) dispatches `backend.scale_f32` / `scale_f64` for the `alpha`-kwarg path in `add_scaled` / `sub_scaled`. |
| REQ-8 | SHIPPED | impl: `strided_copy_f32` / `strided_copy_f64` slots, `strided_scatter_f32` / `strided_scatter_f64` slots on the trait; non-test consumer: `stride_tricks.rs:407-409, 470-472`, `tensor.rs:874-876, 1586-1597` route through these for materialise + CUDA→CPU readback + memory-format permute. |
| REQ-9 | SHIPPED | impl: `sum_axis_f32` / `sum_axis_f64` plus full-reduction variants on the trait; non-test consumer: `grad_fns/arithmetic.rs::reduce_grad_to_shape` (around `:178-336`) calls `backend.sum_axis_f32` / `sum_axis_f64` for the GPU-resident gradient reduction path. |
| REQ-10 | SHIPPED | impl: `matmul_f32` at `gpu_dispatch.rs:293`, `bmm_*`, `gemm_*`, `syevd_*` (cuSOLVER), `getrf_*`, `geqrf_*`, `potrf_*`, `gesdd_*`, `inverse_*` slots; non-test consumer: `ops/linalg.rs::matmul` dispatches `backend.matmul_f32` on CUDA; `linalg.rs::eigh` (`:569`) routes through `backend.syevd_*` per the CUDA fast path. |
| REQ-11 | SHIPPED | impl: convolution + pooling trait slots; non-test consumer: `ferrotorch-nn::Conv2d::forward` (downstream) dispatches via these slots. |
| REQ-12 | SHIPPED | impl: recurrent-layer trait slots; non-test consumer: `ferrotorch-nn::LSTM` / `GRU` / `RNN` forward dispatches. |
| REQ-13 | SHIPPED | impl: FFT trait slots; non-test consumer: `ferrotorch-core::fft` dispatches `backend.fft_*` on CUDA. |
| REQ-14 | SHIPPED | impl: `dropout_*`, `normal_*`, `uniform_*`, etc. trait slots; `save_rng_state` at `gpu_dispatch.rs:2268`, `restore_rng_state` at `:2275`; non-test consumer: `nn::Dropout::forward` for dropout; `creation::randn` / `randn_like` for normal/uniform sampling. |
| REQ-15 | SHIPPED | impl: `masked_fill_dt` at `gpu_dispatch.rs:1867`, `where_cond` at `:1883`, `masked_select` at `:1900`, `masked_scatter` at `:1917`, `argmax` at `:4088`, `argmin` at `:4099`, `index_select_intidx` at `:4116`, `gather_intidx` at `:4137`; non-test consumer: `Tensor::masked_fill` / `masked_select` at `tensor.rs:1126, 1142` dispatch via these slots; `grad_fns/indexing.rs` consumes them in production. |
| REQ-16 | SHIPPED | impl: cuSPARSE dispatch slots in the `sparse.rs:2960-3334` documented region of the trait; non-test consumer: `SparseTensor::from_dense` at `sparse.rs:178-195` dispatches `backend.dense_to_sparse_csr_*` for the CUDA path. |
| REQ-17 | SHIPPED | impl: `int_add` at `gpu_dispatch.rs:3947`, `int_sub` at `:3956`, `int_mul` at `:3965`, `int_neg` at `:3974`, `int_floor_div` at `:3980`, `int_remainder` at `:3992`, `int_bitand` at `:4003`, `int_bitor` at `:4012`, `int_bitxor` at `:4021`, `int_bitnot` at `:4030`, `int_shl` at `:4035`, `int_shr` at `:4045`, `int_sum` at `:4055`, `int_prod` at `:4060`, `int_min` at `:4065`, `int_max` at `:4070`, `cast_f_to_i` at `:4154`, `cast_i_to_f` at `:4164`, `cast_i_to_i` at `:4175`; non-test consumer: `int_tensor.rs` int-tensor op forwarders. |
| REQ-18 | SHIPPED | impl: `compare` at `gpu_dispatch.rs:4198`, `bool_and` at `:4208`, `bool_or` at `:4217`, `bool_xor` at `:4226`, `bool_not` at `:4235`, `bool_any` at `:4241`, `bool_all` at `:4247`, `cast_bool_to_f` at `:4254`; non-test consumer: `bool_tensor.rs` bool-tensor op forwarders. |
| REQ-19 | SHIPPED | impl: `synchronize` at `gpu_dispatch.rs:3269`, `stream_count` at `:3274`, `strided_cat` at `:2237`; non-test consumer: `ferrotorch-gpu::CudaBackendImpl` overrides `synchronize` to call `cudaDeviceSynchronize`. |
| REQ-20 | SHIPPED | impl: `register_gpu_backend` at `gpu_dispatch.rs:4268`, `gpu_backend` at `:4273`, `has_gpu_backend` at `:4278`; non-test consumer: `ferrotorch-gpu/src/backend_impl.rs:7037` (`if has_gpu_backend()`) checks before registering, and every CUDA op in core calls `gpu_backend().ok_or(DeviceUnavailable)?` to obtain `&dyn GpuBackend`. |
