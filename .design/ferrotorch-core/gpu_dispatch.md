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
  materialise (`tensor.rs`), and for memory-format permute
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
  `bernoulli_*`, plus factory RNG slots
  `rand_uniform_{f32,f64,f16,bf16}` and
  `randn_normal_{f32,f64,f16,bf16}` for CUDA-resident
  `torch.rand` / `torch.randn` parity. `save_rng_state` /
  `restore_rng_state` for checkpoint integration.
- REQ-15: Indexing — `check_int_indices_in_bounds`,
  `expand_index_select_indices_i64`, `index_select_intidx`,
  `gather_intidx`, `gather_intidx_nd`, `masked_fill_*`, `masked_select`,
  `masked_scatter`, `where_cond`.
  GPU-resident; `masked_select` returns the compacted output and the
  output-length integer (the only host crossing). Also the
  predicate-mask slots `isfinite_mask` / `ne_scalar_mask` (#1545):
  compute a `DType::Bool` mask on-device from a float value buffer so
  the masked-tensor constructors `masked_invalid` / `masked_equal`
  need not download the value data to host (only the boolean result is
  read back).
- REQ-16: Sparse — `dense_to_sparse_csr_{f32,f64}`,
  `sparse_csr_to_dense_*`, `csr_spmm_*`. Wrappers around cuSPARSE
  `cusparseDenseToSparse` / `cusparseSpMM`.
- REQ-17: Integer ops — `int_add`, `int_sub`, `int_mul`, `int_neg`,
  `int_floor_div`, `int_remainder`, bitwise ops, shifts, integer
  reductions, plus integer↔float / integer↔integer casts. Mostly
  default-`Err(NotImplementedOnCuda)` slots that concrete backends
  can override.
- REQ-18: Boolean ops — `compare`, `compare_broadcast`, `bool_and`,
  `bool_or`, `bool_xor`, `bool_not`, `bool_any`, `bool_all`, `cast_bool_to_f`,
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
  (`gpu_dispatch.rs`).
- [x] AC-3: `GpuRngState::new(counter, seed, offset, device)`
  preserves all four fields through getters
  (`gpu_dispatch.rs:84-119`).
- [x] AC-4: `register_gpu_backend(b1)` succeeds; a second
  `register_gpu_backend(b2)` returns `Err(b2)`
  (`gpu_dispatch.rs` (GpuBackend trait methods)).
- [x] AC-5: `has_gpu_backend()` toggles with registration
  (`gpu_dispatch.rs:8521`).
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
- **Lines 310-8507**: `trait GpuBackend` — ~350 method slots. Most
  carry default impls that return `Err(NotImplementedOnCuda)` so
  concrete backends can override only what they support; the core
  methods (`cpu_to_gpu`, `gpu_to_cpu`, `clone_buffer`, `alloc_zeros`,
  `add_f32`, `sub_f32`, `mul_f32`, `neg_f32`, `relu_f32`,
  `matmul_f32`) are unimplemented-by-default and MUST be provided.
- **Lines 8511-8521**: `register_gpu_backend`, `gpu_backend`,
  `has_gpu_backend` — the registration plumbing built on a single
  `OnceLock<Box<dyn GpuBackend>>`.
- **Lines 4286-4300**: in-file test mod (handle construction +
  debug formatting).

Non-test production consumers:

- `ferrotorch-gpu/src/backend_impl.rs:970` (`impl GpuBackend for
  CudaBackendImpl` + the `register_gpu_backend(...)` call at
  `:14001`) is the canonical concrete implementation.
- Inside `ferrotorch-core`, every CUDA-dispatched op routes through
  `gpu_backend()` — see e.g. `gpu_backend in tensor.rs`, `gpu_backend in tensor.rs`,
  `tensor.rs`, `tensor.rs`, `tensor in stride_tricks.rs`,
  `stride_tricks.rs:440`, `storage.rs:353`, `storage.rs:414`,
  `storage.rs:556`, `grad_fns/arithmetic.rs` CUDA branches.

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
| REQ-2 | SHIPPED | impl: `GpuRngState` at `ferrotorch-core/src/gpu_dispatch.rs:94-145` with accessors at `:105-145`; non-test consumer: `GpuBackend::save_rng_state` / `restore_rng_state` produce / consume this struct; `ferrotorch-core::checkpoint` (downstream) serialises it. |
| REQ-3 | SHIPPED | impl: `GpuBufferHandle` at `ferrotorch-core/src/gpu_dispatch.rs:162-235` with `new` at `:185-197`, accessors at `:199-233`; non-test consumer: `TensorStorage::Gpu(GpuBufferHandle)` variant plus every CUDA op that reads / writes the handle. |
| REQ-4 | SHIPPED | impl: `trait GpuBackend` at `ferrotorch-core/src/gpu_dispatch.rs:310`; non-test consumer: `ferrotorch-gpu/src/backend_impl.rs:970`'s `impl GpuBackend for CudaBackendImpl` registers via `register_gpu_backend(...)`. |
| REQ-5 | SHIPPED | impl: elementwise method slots `add_f32` at `gpu_dispatch.rs:401`, `sub_f32` at `:406`, `mul_f32` at `:411`, `neg_f32` at `:444`, `relu_f32` at `:445` (plus the rest of the elementwise zoo throughout the file); non-test consumer: `Tensor::accumulate_grad` GPU path at `tensor.rs:1049-1077` calls `backend.add_f32` / `add_f64`; `grad_fns/arithmetic.rs::add_inner` dispatches the CUDA branches. |
| REQ-6 | SHIPPED | impl: broadcast variants are sibling slots on the trait alongside the non-broadcast methods (e.g. `broadcast_add_f32`); non-test consumer: `grad_fns/arithmetic.rs::add_inner` picks the broadcast variant via the dispatch macro when input shapes differ. |
| REQ-7 | SHIPPED | impl: `scale_*` trait slots in the same elementwise section; non-test consumer: `grad_fns/arithmetic.rs::scale_tensor` (`:547-578`) dispatches `backend.scale_f32` / `scale_f64` for the `alpha`-kwarg path in `add_scaled` / `sub_scaled`. |
| REQ-8 | SHIPPED | impl: `strided_copy_f32` / `strided_copy_f64` slots, `strided_scatter_f32` / `strided_scatter_f64` slots on the trait; non-test consumer: `strided_scatter_f64 in stride_tricks.rs, 470-472`, `strided_copy_f64 in tensor.rs, 1586-1597` route through these for materialise + CUDA→CPU readback + memory-format permute. |
| REQ-9 | SHIPPED | impl: `sum_axis_f32` / `sum_axis_f64` plus full-reduction variants on the trait; non-test consumer: `grad_fns/arithmetic.rs::reduce_grad_to_shape` (around `:178-336`) calls `backend.sum_axis_f32` / `sum_axis_f64` for the GPU-resident gradient reduction path. |
| REQ-10 | SHIPPED | impl: `matmul_f32 in gpu_dispatch.rs`, `bmm_*`, `gemm_*`, `syevd_*` (cuSOLVER), `getrf_*`, `geqrf_*`, `potrf_*`, `gesdd_*`, `inverse_*` slots; non-test consumer: `ops/linalg.rs::matmul` dispatches `backend.matmul_f32` on CUDA; `linalg.rs::eigh` (`eigh in gpu_dispatch.rs`) routes through `backend.syevd_*` per the CUDA fast path. |
| REQ-11 | SHIPPED | impl: convolution + pooling trait slots; non-test consumer: `ferrotorch-nn::Conv2d::forward` (downstream) dispatches via these slots. |
| REQ-12 | SHIPPED | impl: recurrent-layer trait slots; non-test consumer: `ferrotorch-nn::LSTM` / `GRU` / `RNN` forward dispatches. |
| REQ-13 | SHIPPED | impl: FFT trait slots; non-test consumer: `ferrotorch-core::fft` dispatches `backend.fft_*` on CUDA. |
| REQ-14 | SHIPPED | impl: `dropout_f32` / `dropout_philox_f32`, `save_rng_state` / `restore_rng_state`, `manual_seed_gpu`, and the on-device factory generation slots `rand_uniform_{f32,f64,f16,bf16}` / `randn_normal_{f32,f64,f16,bf16}` in `gpu_dispatch.rs` (#1682, default `NotImplementedOnCuda` / no-op so non-CUDA backends compile unchanged); non-test consumer: `nn::Dropout::forward` for dropout; `creation::rand_on_device` / `randn_on_device` dispatch CUDA floating dtypes through these dtype-specific slots; `crate::rng::manual_seed` forwards to `manual_seed_gpu` (`torch.manual_seed` -> `torch.cuda.manual_seed_all`, `torch/random.py:67`). CUDA impls in `CudaBackendImpl in ferrotorch-gpu/src/backend_impl.rs`. |
| REQ-15 | SHIPPED | impl: `masked_fill_dt` / `where_cond` / `masked_select` / `masked_scatter` / `argmax` / `argmin` / `check_int_indices_in_bounds` / `expand_index_select_indices_i64` / `index_select_intidx` / `gather_intidx` / `gather_intidx_nd in gpu_dispatch.rs`; non-test consumer: `Tensor::masked_fill` / `masked_select in tensor.rs` dispatch via these slots; `Tensor::index_select`, `Tensor::gather`, `IntTensor::index_select`, and `IntTensor::gather in ops/phase2c.rs` call `check_int_indices_in_bounds` before unchecked integer-index copy kernels; `IndexSelectDimBackward in grad_fns/indexing.rs` calls `expand_index_select_indices_i64` to keep tracked CUDA backward resident; `Tensor::gather` and `IntTensor::gather in ops/phase2c.rs` dispatch through `gather_intidx_nd` for PyTorch-legal smaller non-axis index shapes; `grad_fns/indexing.rs` consumes indexing paths in production. Predicate-mask slots `isfinite_mask` / `ne_scalar_mask` (`isfinite_mask in gpu_dispatch.rs`); non-test consumer: `ferrotorch_core::masked_invalid` / `masked_equal` (`masked.rs`) CUDA branches; backend impl `CudaBackendImpl::isfinite_mask` / `ne_scalar_mask in ferrotorch-gpu/src/backend_impl.rs`. |
| REQ-16 | SHIPPED | impl: cuSPARSE dispatch slots in the `sparse in sparse.rs` documented region of the trait; non-test consumer: `SparseTensor::from_dense` at `sparse in sparse.rs` dispatches `backend.dense_to_sparse_csr_*` for the CUDA path. |
| REQ-17 | SHIPPED | impl: `int_add in gpu_dispatch.rs`, `int_sub in gpu_dispatch.rs`, `int_mul in gpu_dispatch.rs`, `int_neg in gpu_dispatch.rs`, `int_floor_div in gpu_dispatch.rs`, `int_remainder in gpu_dispatch.rs`, `int_bitand in gpu_dispatch.rs`, `int_bitor in gpu_dispatch.rs`, `int_bitxor in gpu_dispatch.rs`, `int_bitnot in gpu_dispatch.rs`, `int_shl in gpu_dispatch.rs`, `int_shr in gpu_dispatch.rs`, `int_sum in gpu_dispatch.rs`, `int_prod in gpu_dispatch.rs`, `int_min in gpu_dispatch.rs`, `int_max in gpu_dispatch.rs`, `cast_f_to_i in gpu_dispatch.rs`, `cast_i_to_f in gpu_dispatch.rs`, `cast_i_to_i in gpu_dispatch.rs`; non-test consumer: `int_tensor.rs` int-tensor op forwarders. |
| REQ-18 | SHIPPED | impl: `compare in gpu_dispatch.rs`, `compare_broadcast in gpu_dispatch.rs`, `bool_and in gpu_dispatch.rs`, `bool_or in gpu_dispatch.rs`, `bool_xor in gpu_dispatch.rs`, `bool_not in gpu_dispatch.rs`, `bool_any in gpu_dispatch.rs`, `bool_all in gpu_dispatch.rs`, `cast_bool_to_f in gpu_dispatch.rs`; non-test consumer: `bool_tensor.rs` bool-tensor op forwarders. `BoolTensor::compare_int` uses `compare_broadcast` for CUDA i32/i64 operands with broadcast-compatible differing shapes so the backend can keep operand values and bool output device-resident. |
| REQ-19 | SHIPPED | impl: `synchronize` / `stream_count` / `strided_cat in gpu_dispatch.rs`; non-test consumer: `ferrotorch-gpu::CudaBackendImpl` overrides `synchronize` to call `cudaDeviceSynchronize`. |
| REQ-20 | SHIPPED | impl: `register_gpu_backend in gpu_dispatch.rs`, `gpu_backend in gpu_dispatch.rs`, `has_gpu_backend in gpu_dispatch.rs`; non-test consumer: `has_gpu_backend in ferrotorch-gpu/src/backend_impl.rs` (`if has_gpu_backend()`) checks before registering, and every CUDA op in core calls `gpu_backend().ok_or(DeviceUnavailable)?` to obtain `&dyn GpuBackend`. |
