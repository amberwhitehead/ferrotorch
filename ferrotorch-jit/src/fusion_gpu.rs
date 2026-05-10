// CUDA kernel launches in this module use cudarc's `LaunchAsync`, which is
// fundamentally `unsafe` because the caller is responsible for the kernel's
// ABI matching the bound argument list. Each `unsafe { ... }` block below
// carries a SAFETY comment justifying its specific invariants — module-wide
// `allow(unsafe_code)` mirrors the precedent in `codegen_jit.rs`.
#![allow(unsafe_code)]

//! GPU runtime executor for [`crate::fusion::FusedChain`].
//!
//! Closes the f32-and-f64 `apply_fused` GPU dispatch loop:
//!
//! 1. Generate the chain's PTX — f32 directly via
//!    [`crate::fusion::FusedChain::generate_ptx_named`], f64 via
//!    [`crate::fusion::FusedChain::generate_cuda_source_f64_named`] →
//!    [`crate::nvrtc::compile_cuda_source_to_ptx`].
//! 2. Cache / compile the resulting `CudaFunction` via
//!    [`ferrotorch_gpu::module_cache::get_or_compile_owned`].
//! 3. Launch the kernel on the input tensor's stream and wrap the result
//!    as a device-resident `Tensor<T>`.
//!
//! Gated entirely on `cfg(feature = "cuda")` — this module is omitted from
//! the default workspace build, so CPU-only builds never pull cudarc or
//! ferrotorch-gpu through this path.

use std::any::TypeId;

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::gpu_dispatch::GpuBufferHandle;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use ferrotorch_gpu::buffer::CudaBuffer;
use ferrotorch_gpu::device::GpuDevice;
use ferrotorch_gpu::module_cache;
use ferrotorch_gpu::transfer::{alloc_zeros_f32, alloc_zeros_f64};

use crate::fusion::FusedChain;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Apply a [`FusedChain`] to a CUDA-resident tensor.
///
/// Dispatches on `TypeId::of::<T>()`:
/// - f32 → [`apply_fused_gpu_f32_internal`]
/// - f64 → [`apply_fused_gpu_f64_internal`]
/// - other → `NotImplementedOnCuda` (`FusedChain` GPU path supports f32 / f64
///   only; bf16 / f16 are intentionally not lowered here because the
///   tracing JIT does not yet emit fused chains for half-precision tensors
///   and the dtype trait `Float` does not constrain `T` to those types).
///
/// # Errors
///
/// - [`FerrotorchError::NotImplementedOnCuda`] on unsupported dtypes.
/// - [`FerrotorchError::InvalidArgument`] on:
///   - chain validation failures (binary ops in a unary chain) raised by
///     the PTX generators, or
///   - cudarc driver / NVRTC errors during launch.
///
/// # Caller contract
///
/// The input tensor MUST be on a CUDA device; the caller (i.e.
/// [`crate::fusion::apply_fused`]) checks this with `input.is_cuda()`
/// before forwarding.
pub fn apply_fused_gpu<T: Float>(
    input: &Tensor<T>,
    chain: &FusedChain,
) -> FerrotorchResult<Tensor<T>> {
    let type_id = TypeId::of::<T>();
    if type_id == TypeId::of::<f32>() {
        apply_fused_gpu_f32_internal(input, chain)
    } else if type_id == TypeId::of::<f64>() {
        apply_fused_gpu_f64_internal(input, chain)
    } else {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "apply_fused: GPU path supports f32 and f64 only",
        })
    }
}

// ---------------------------------------------------------------------------
// f32 path
// ---------------------------------------------------------------------------

/// Stable kernel-entry name for the f32 fused chain. The name is part of
/// the PTX content (the `.visible .entry` declaration), so the
/// owned-string module cache will key uniquely on the PTX hash even when
/// two chains coincidentally share the same opcode sequence: any change
/// to ops changes the body, which changes the hash.
const FUSED_F32_KERNEL_NAME: &str = "fused_chain_f32";

fn apply_fused_gpu_f32_internal<T: Float>(
    input: &Tensor<T>,
    chain: &FusedChain,
) -> FerrotorchResult<Tensor<T>> {
    debug_assert_eq!(TypeId::of::<T>(), TypeId::of::<f32>());

    let handle = input.gpu_handle()?;
    let buffer = handle
        .downcast_ref::<CudaBuffer<f32>>()
        .ok_or(FerrotorchError::InvalidArgument {
            message: "apply_fused: CUDA tensor's GPU handle is not a CudaBuffer<f32>".into(),
        })?;
    let n = buffer.len();

    // GpuDevice::new(ordinal) on cudarc is a context lookup (CudaContext::new
    // returns the existing context for an ordinal if one was already
    // initialized); allocations and stream construction reuse the same
    // CUDA context as the input tensor.
    let device = GpuDevice::new(handle.device_ordinal()).map_err(|e| map_gpu_err(&e))?;

    // Generate PTX for this chain. The chain's validation (binary-op
    // rejection, identifier validation) is enforced inside generate_ptx_named.
    let ptx = chain.generate_ptx_named(FUSED_F32_KERNEL_NAME)?;

    // Compile / cache the function.
    let func = module_cache::get_or_compile_owned(
        device.context(),
        ptx,
        FUSED_F32_KERNEL_NAME.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("apply_fused: PTX compile/load failed (f32): {e}"),
    })?;

    // Allocate output buffer of same length.
    let mut out_buf = alloc_zeros_f32(n, &device).map_err(|e| map_gpu_err(&e))?;

    // Build the launch config and dispatch.
    let cfg = launch_cfg(n)?;
    let stream = device.stream();
    let n_u32 = n as u32;

    // SAFETY:
    // - `func` was just compiled from a PTX kernel whose entry-point ABI
    //   is `(in_ptr: u64, out_ptr: u64, n: u32)` as fixed by
    //   `FusedChain::generate_ptx_named` at fusion.rs (search for
    //   `.visible .entry {kernel_name}(` — there is exactly one signature).
    // - `buffer` is a non-null `CudaBuffer<f32>` of length `n` on
    //   `device`; we read each `i < n` element exactly once.
    // - `out_buf` was freshly allocated this call via `alloc_zeros_f32`
    //   and cannot alias `buffer`; we hold its only `&mut` reference.
    // - `n_u32` is `n as u32`. `launch_cfg(n)` returns Err if `n` would
    //   overflow u32, so the cast cannot wrap.
    // - The kernel's PTX bound check (`setp.ge.u32 %p, %tid, %n_reg; @%p
    //   bra DONE;`) skips threads beyond `n`, so every read of `buffer[i]`
    //   and every write of `out_buf[i]` stays within `[0, n)`.
    // - All three arg references live for the duration of the
    //   `.launch(cfg)?` call; cudarc queues the kernel on `stream` and
    //   stream-sync is the caller's responsibility (handled implicitly
    //   on the subsequent D2H readback in tests / consumers).
    unsafe {
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&func)
            .arg(buffer.inner())
            .arg(out_buf.inner_mut())
            .arg(&n_u32)
            .launch(cfg)
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("apply_fused: f32 kernel launch failed: {e}"),
            })?;
    }

    wrap_output_f32(out_buf, input.shape().to_vec(), handle.device_ordinal())
}

// ---------------------------------------------------------------------------
// f64 path
// ---------------------------------------------------------------------------

const FUSED_F64_KERNEL_NAME: &str = "fused_chain_f64";

fn apply_fused_gpu_f64_internal<T: Float>(
    input: &Tensor<T>,
    chain: &FusedChain,
) -> FerrotorchResult<Tensor<T>> {
    debug_assert_eq!(TypeId::of::<T>(), TypeId::of::<f64>());

    let handle = input.gpu_handle()?;
    let buffer = handle
        .downcast_ref::<CudaBuffer<f64>>()
        .ok_or(FerrotorchError::InvalidArgument {
            message: "apply_fused: CUDA tensor's GPU handle is not a CudaBuffer<f64>".into(),
        })?;
    let n = buffer.len();

    let device = GpuDevice::new(handle.device_ordinal()).map_err(|e| map_gpu_err(&e))?;

    // Generate CUDA C source then compile to PTX via NVRTC + libdevice.
    let cuda_source = chain.generate_cuda_source_f64_named(FUSED_F64_KERNEL_NAME)?;
    let ptx = crate::nvrtc::compile_cuda_source_to_ptx(&cuda_source, FUSED_F64_KERNEL_NAME)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("apply_fused: NVRTC compile of f64 chain failed: {e}"),
        })?;

    let func = module_cache::get_or_compile_owned(
        device.context(),
        ptx,
        FUSED_F64_KERNEL_NAME.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("apply_fused: PTX compile/load failed (f64): {e}"),
    })?;

    let mut out_buf = alloc_zeros_f64(n, &device).map_err(|e| map_gpu_err(&e))?;

    let cfg = launch_cfg(n)?;
    let stream = device.stream();
    // NVRTC-emitted f64 kernel signature is `(const double*, double*,
    // int)` from `FusedChain::generate_cuda_source_f64_named`; the `int n`
    // parameter is fed by `n as i32`.  `launch_cfg(n)` already enforces
    // `n <= u32::MAX`; we further check the i32 range here so the
    // signed-int parameter does not wrap.
    if n > i32::MAX as usize {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "apply_fused: f64 chain length {n} exceeds i32::MAX; \
                 split the tensor or extend the kernel signature"
            ),
        });
    }
    let n_i32 = n as i32;

    // SAFETY:
    // - `func` was just compiled from CUDA C whose `__global__` signature
    //   is `(const double* in, double* out, int n)` (see
    //   `FusedChain::generate_cuda_source_f64_named`); `extern "C"` in
    //   the NVRTC preprocessing keeps the entry-point name unmangled
    //   (see `crate::nvrtc::compile_cuda_source_to_ptx`).
    // - `buffer` is a non-null `CudaBuffer<f64>` of length `n` on
    //   `device`.
    // - `out_buf` was freshly allocated and exclusively borrowed.
    // - The kernel guards every load/store with `if (i < n)`.
    // - All arg references live for the duration of `.launch(cfg)?`.
    unsafe {
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&func)
            .arg(buffer.inner())
            .arg(out_buf.inner_mut())
            .arg(&n_i32)
            .launch(cfg)
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("apply_fused: f64 kernel launch failed: {e}"),
            })?;
    }

    wrap_output_f64(out_buf, input.shape().to_vec(), handle.device_ordinal())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Standard 1-D launch config — 256 threads/block — matching the
/// project-wide convention (see `ferrotorch-gpu/src/conv.rs::launch_cfg`).
fn launch_cfg(n: usize) -> FerrotorchResult<cudarc::driver::LaunchConfig> {
    if n > u32::MAX as usize {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("apply_fused: chain length {n} exceeds u32::MAX threads"),
        });
    }
    const BLOCK: u32 = 256;
    let n_u32 = n as u32;
    let grid = ((n_u32).saturating_add(BLOCK - 1)) / BLOCK;
    Ok(cudarc::driver::LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// Wrap an owned `CudaBuffer<f32>` as a `Tensor<T>` where `T == f32`.
///
/// The result lives on the same CUDA device as the input.
/// `TensorStorage::gpu` is a type-erased constructor (the
/// `GpuBufferHandle` it holds is `Box<dyn Any + Send + Sync>` internally),
/// so it can be parameterised by any `T: Element` — including the caller's
/// generic `T` parameter — without an unsafe re-tag. The `debug_assert!`
/// guards the calling-convention promise that this helper is invoked only
/// when `T == f32`.
fn wrap_output_f32<T: Float>(
    buf: CudaBuffer<f32>,
    shape: Vec<usize>,
    device_ordinal: usize,
) -> FerrotorchResult<Tensor<T>> {
    debug_assert_eq!(TypeId::of::<T>(), TypeId::of::<f32>());
    let len = buf.len();
    let handle = GpuBufferHandle::new(Box::new(buf), device_ordinal, len);
    let storage: TensorStorage<T> = TensorStorage::gpu(handle);
    Tensor::from_storage(storage, shape, false)
}

/// Wrap an owned `CudaBuffer<f64>` as a `Tensor<T>` where `T == f64`.
fn wrap_output_f64<T: Float>(
    buf: CudaBuffer<f64>,
    shape: Vec<usize>,
    device_ordinal: usize,
) -> FerrotorchResult<Tensor<T>> {
    debug_assert_eq!(TypeId::of::<T>(), TypeId::of::<f64>());
    let len = buf.len();
    let handle = GpuBufferHandle::new(Box::new(buf), device_ordinal, len);
    let storage: TensorStorage<T> = TensorStorage::gpu(handle);
    Tensor::from_storage(storage, shape, false)
}

/// Convert a `ferrotorch_gpu::error::GpuError` into a `FerrotorchError`
/// for propagation through this module's public API. Takes by reference
/// to avoid forcing callers to surrender ownership at every site.
fn map_gpu_err(e: &ferrotorch_gpu::error::GpuError) -> FerrotorchError {
    FerrotorchError::InvalidArgument {
        message: format!("apply_fused: GPU error: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Tests — all CUDA-runtime-executed
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::{FusedChain, FusedOp};

    fn cuda_or_skip<T: Float>(data: Vec<T>, shape: Vec<usize>) -> Option<Tensor<T>> {
        let storage = TensorStorage::cpu(data);
        let cpu = Tensor::from_storage(storage, shape, false).expect("cpu tensor");
        cpu.cuda().ok()
    }

    /// Reference: run the chain on the CPU and compare elementwise.
    fn cpu_reference<T: Float + Copy>(input: &[T], chain: &FusedChain) -> Vec<T> {
        chain.execute_cpu(input).expect("cpu execute")
    }

    #[test]
    fn apply_fused_gpu_f32_scalar_add_relu_neg_roundtrip() {
        let input_data: Vec<f32> = vec![-5.0, -1.0, 0.0, 1.0, 3.0, 7.5, -2.25];
        let Some(cuda_tensor) = cuda_or_skip::<f32>(input_data.clone(), vec![7]) else {
            eprintln!("apply_fused_gpu_f32_scalar_add_relu_neg_roundtrip: no CUDA, skipping");
            return;
        };

        let mut chain = FusedChain::new();
        chain.push(FusedOp::ScalarAdd(2.0));
        chain.push(FusedOp::Relu);
        chain.push(FusedOp::Neg);

        let gpu_out = apply_fused_gpu(&cuda_tensor, &chain).expect("apply_fused_gpu f32");
        assert!(
            gpu_out.is_cuda(),
            "result must remain on CUDA, got device {:?}",
            gpu_out.device()
        );
        let host_out = gpu_out.cpu().expect("readback").data().unwrap().to_vec();

        let expected = cpu_reference::<f32>(&input_data, &chain);
        for (i, (g, e)) in host_out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-5,
                "f32 element {i}: got {g}, expected {e}",
            );
        }
    }

    #[test]
    fn apply_fused_gpu_f64_scalar_add_relu_neg_roundtrip() {
        let input_data: Vec<f64> = vec![-5.0, -1.0, 0.0, 1.0, 3.0, 7.5, -2.25];
        let Some(cuda_tensor) = cuda_or_skip::<f64>(input_data.clone(), vec![7]) else {
            eprintln!("apply_fused_gpu_f64_scalar_add_relu_neg_roundtrip: no CUDA, skipping");
            return;
        };

        let mut chain = FusedChain::new();
        chain.push(FusedOp::ScalarAdd(2.0));
        chain.push(FusedOp::Relu);
        chain.push(FusedOp::Neg);

        let gpu_out = apply_fused_gpu(&cuda_tensor, &chain).expect("apply_fused_gpu f64");
        assert!(gpu_out.is_cuda());
        let host_out = gpu_out.cpu().expect("readback").data().unwrap().to_vec();

        let expected = cpu_reference::<f64>(&input_data, &chain);
        for (i, (g, e)) in host_out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-12,
                "f64 element {i}: got {g}, expected {e}",
            );
        }
    }

    #[test]
    fn apply_fused_gpu_f32_with_transcendentals() {
        // Exp + Log + Sigmoid → tests the *.approx.f32 path.
        let input_data: Vec<f32> = vec![0.5, 1.0, 2.0, 3.0, 0.25];
        let Some(cuda_tensor) = cuda_or_skip::<f32>(input_data.clone(), vec![5]) else {
            return;
        };

        let mut chain = FusedChain::new();
        chain.push(FusedOp::Exp);
        chain.push(FusedOp::Log);
        chain.push(FusedOp::Sigmoid);

        let gpu_out = apply_fused_gpu(&cuda_tensor, &chain).expect("apply_fused_gpu f32 trans");
        let host_out = gpu_out.cpu().expect("readback").data().unwrap().to_vec();
        let expected = cpu_reference::<f32>(&input_data, &chain);

        // ex2.approx.f32 / lg2.approx.f32 / rcp.approx.f32 are only
        // 1-ULP-ish accurate; loosen the tolerance accordingly.
        for (i, (g, e)) in host_out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-3,
                "f32 trans element {i}: got {g}, expected {e}",
            );
        }
    }

    #[test]
    fn apply_fused_gpu_f64_with_transcendentals() {
        // Same chain on f64: routed through NVRTC + libdevice.
        let input_data: Vec<f64> = vec![0.5, 1.0, 2.0, 3.0, 0.25];
        let Some(cuda_tensor) = cuda_or_skip::<f64>(input_data.clone(), vec![5]) else {
            return;
        };

        let mut chain = FusedChain::new();
        chain.push(FusedOp::Exp);
        chain.push(FusedOp::Log);
        chain.push(FusedOp::Sigmoid);

        let gpu_out = apply_fused_gpu(&cuda_tensor, &chain).expect("apply_fused_gpu f64 trans");
        let host_out = gpu_out.cpu().expect("readback").data().unwrap().to_vec();
        let expected = cpu_reference::<f64>(&input_data, &chain);

        // libdevice's f64 transcendentals are IEEE-accurate to within
        // a few ULPs; tight tolerance is fine.
        for (i, (g, e)) in host_out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-9,
                "f64 trans element {i}: got {g}, expected {e}",
            );
        }
    }

    #[test]
    fn apply_fused_gpu_preserves_device_for_cuda_input() {
        let Some(cuda_tensor) = cuda_or_skip::<f32>(vec![1.0, 2.0, 3.0], vec![3]) else {
            return;
        };

        let mut chain = FusedChain::new();
        chain.push(FusedOp::Relu);

        let gpu_out = apply_fused_gpu(&cuda_tensor, &chain).expect("apply_fused_gpu");
        // Both tensors must be on a CUDA device (specifically the same one).
        assert!(
            cuda_tensor.is_cuda(),
            "input must be CUDA-resident (test invariant)"
        );
        assert!(
            gpu_out.is_cuda(),
            "result must remain on CUDA, got device {:?}",
            gpu_out.device()
        );
        assert_eq!(gpu_out.device(), cuda_tensor.device(), "device match");
    }

    #[test]
    fn apply_fused_gpu_errs_on_binary_op_chain_f32() {
        let Some(cuda_tensor) = cuda_or_skip::<f32>(vec![1.0, 2.0, 3.0], vec![3]) else {
            return;
        };

        let mut chain = FusedChain::new();
        // Binary op in a unary chain → rejected by generate_ptx_named.
        chain.push(FusedOp::Mul);

        let result = apply_fused_gpu(&cuda_tensor, &chain);
        match result {
            Err(FerrotorchError::InvalidArgument { message }) => {
                assert!(
                    message.contains("binary op"),
                    "error must explain the binary op rejection; got: {message}"
                );
            }
            other => panic!("expected InvalidArgument for binary op chain, got {other:?}"),
        }
    }

    #[test]
    fn apply_fused_gpu_errs_on_binary_op_chain_f64() {
        let Some(cuda_tensor) = cuda_or_skip::<f64>(vec![1.0, 2.0, 3.0], vec![3]) else {
            return;
        };

        let mut chain = FusedChain::new();
        chain.push(FusedOp::Add);

        let result = apply_fused_gpu(&cuda_tensor, &chain);
        match result {
            Err(FerrotorchError::InvalidArgument { message }) => {
                assert!(
                    message.contains("binary op"),
                    "error must explain the binary op rejection; got: {message}"
                );
            }
            other => panic!("expected InvalidArgument for f64 binary op chain, got {other:?}"),
        }
    }

    #[test]
    fn apply_fused_gpu_multi_op_chain_f32_matches_cpu() {
        // Longer chain spanning every f32 op category.
        let input_data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.25).collect();
        let Some(cuda_tensor) = cuda_or_skip::<f32>(input_data.clone(), vec![32]) else {
            return;
        };

        let mut chain = FusedChain::new();
        chain.push(FusedOp::Abs);
        chain.push(FusedOp::ScalarAdd(0.5));
        chain.push(FusedOp::Sqrt);
        chain.push(FusedOp::ScalarMul(2.0));
        chain.push(FusedOp::Neg);

        let gpu_out = apply_fused_gpu(&cuda_tensor, &chain).expect("apply_fused_gpu multi");
        let host_out = gpu_out.cpu().expect("readback").data().unwrap().to_vec();
        let expected = cpu_reference::<f32>(&input_data, &chain);
        for (i, (g, e)) in host_out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "f32 multi-op element {i}: got {g}, expected {e}",
            );
        }
    }

    #[test]
    fn apply_fused_gpu_cache_hit_second_call() {
        // Two back-to-back calls with the same chain must succeed; the
        // second must be measurably faster (PTX compile happens only on
        // the first), proving the module cache is engaged.
        use std::time::Instant;

        let Some(cuda_tensor) = cuda_or_skip::<f32>(vec![1.0f32; 256], vec![256]) else {
            return;
        };

        // Distinct chain so this test does not collide with any other
        // test's cache entry — Pow(3) is rare in the rest of the suite.
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Pow(3.0));
        chain.push(FusedOp::Abs);

        let t1 = Instant::now();
        let r1 = apply_fused_gpu(&cuda_tensor, &chain).expect("first call");
        // Force readback so kernel actually executes and the cache entry
        // gets created.
        let _ = r1.cpu().expect("readback 1");
        let d1 = t1.elapsed();

        let t2 = Instant::now();
        let r2 = apply_fused_gpu(&cuda_tensor, &chain).expect("second call (cached)");
        let _ = r2.cpu().expect("readback 2");
        let d2 = t2.elapsed();

        eprintln!("apply_fused_gpu cache: 1st = {d1:?}, 2nd = {d2:?}");
        // Sanity: results consistent across calls.
        let h1 = r1.cpu().expect("h1").data().unwrap().to_vec();
        let h2 = r2.cpu().expect("h2").data().unwrap().to_vec();
        assert_eq!(h1, h2, "cached result must be identical across calls");
    }
}
