//! Apple Metal backend for ferrotorch-mps (#626).
//!
//! [`MtlBackend`] implements [`GpuBackend`] from `ferrotorch-core` using MSL
//! kernels compiled at runtime via the `objc2-metal` crate.
//!
//! # Platform gating
//!
//! This entire module is `#[cfg(target_os = "macos")]`. On Linux/WSL the
//! module is absent from the compilation unit, so none of the `objc2-metal`
//! bindings are referenced and the workspace build stays clean.
//!
//! # PyTorch parity (§3)
//!
//! ferrotorch is a PyTorch reimplementation. Every method that cannot execute
//! a real Metal kernel on the current platform returns
//! `Err(FerrotorchError::DeviceUnavailable)` — never a silent CPU detour.
//! On macOS, methods for the 10 implemented kernels compile and launch the
//! MSL source; the remaining ~70 GpuBackend methods return
//! `Err(FerrotorchError::InvalidArgument { message: "MSL kernel needed: ..." })`,
//! matching PyTorch's `NotImplementedError` shape for unimplemented ops.
//!
//! # Runtime kernel compilation
//!
//! `MtlBackend::new()` eagerly compiles all 10 MSL shader libraries. Compilation
//! failures return `Err` immediately — there is no lazy degrade-to-CPU path.
//! The compiled `MTLComputePipelineState` handles are cached in `MtlBackend`
//! for the lifetime of the backend.
//!
//! # Buffer representation
//!
//! GPU buffers are `Arc<MtlBuffer>` (a newtype around `Retained<ProtocolObject<dyn MTLBuffer>>`)
//! stored in `GpuBufferHandle::inner` via type-erasure. Downcast via
//! `handle.downcast_ref::<Arc<MtlBuffer>>()`.
//!
//! ## REQ status (per `.design/ferrotorch-mps/backend.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream
//! cites) live in the design doc; this synopsis is a one-line summary per
//! REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`MtlBackend` struct + `impl GpuBackend`) | SHIPPED | `pub struct MtlBackend { device, queue, pipelines }` + `impl GpuBackend for MtlBackend` in `backend.rs`; consumer `pub use backend::MtlBackend` in `lib.rs` re-exports it, and `ferrotorch/src/lib.rs:137` `pub use ferrotorch_mps::*;` propagates on macOS |
//! | REQ-2 (`MtlBackend::new` fail-fast pipeline compile) | SHIPPED | `pub fn MtlBackend::new` resolves `MTLCreateSystemDefaultDevice` then eagerly compiles all 10 MSL pipelines via `fn compile_pipeline` in `backend.rs`; consumer `pub fn init_mps_backend_metal` invokes `MtlBackend::new()?` |
//! | REQ-3 (`Arc<MtlBuffer>` type-erasure bridge) | SHIPPED | `pub struct MtlBuffer` + `unsafe impl Send/Sync` (SAFETY: Metal ObjC ARC + command-queue serialisation) + `fn wrap_buffer`/`fn downcast_buf` in `backend.rs`; consumer every `impl GpuBackend` method body calls `Self::downcast_buf(handle)?` |
//! | REQ-4 (memory mgmt via shared-mode buffers) | SHIPPED | `fn cpu_to_gpu / gpu_to_cpu / clone_buffer / alloc_zeros / buffer_elem_size` in `impl GpuBackend for MtlBackend` using `MTLResourceOptions::StorageModeShared` per `aten/src/ATen/mps/MPSAllocator.h:36`; consumer every kernel launcher calls `Self::alloc_buffer` for the output |
//! | REQ-5 (elementwise + activation kernels) | SHIPPED | `fn add_f32 / sub_f32 / mul_f32 / div_f32 / relu_f32 / sigmoid_f32` delegating to `fn launch_binary_f32` / `fn launch_unary_f32` in `backend.rs`; consumer `ferrotorch_core::gpu_dispatch::gpu_backend()` returns this `&dyn GpuBackend` on macOS post-`init_mps_backend()` |
//! | REQ-6 (matmul / bmm kernels) | SHIPPED | `fn matmul_f32` + `fn bmm_f32` in `backend.rs` build command-buffer + encoder + setBuffer + setBytes + 16x16 threadgroup dispatch; consumer trait dispatch via `gpu_backend()` for `Tensor::matmul` / `Tensor::bmm` on macOS |
//! | REQ-7 (softmax + sum_axis reductions) | SHIPPED | `fn softmax_f32 / sum_f32 / sum_axis_f32` using `dispatchThreadgroups_threadsPerThreadgroup` (one tg per output row/element) and `pow2_tg_width(cols)` / `pow2_tg_width(axis_len)`; consumer trait dispatch via `gpu_backend()` for `Tensor::softmax` / `Tensor::sum` |
//! | REQ-8 (`pow2_tg_width` helper) | SHIPPED | `fn pow2_tg_width(n) = n.min(1024).next_power_of_two()` in `backend.rs` with documented #1101 contract; consumer `fn softmax_f32` and `fn sum_axis_f32` invoke it before threadgroup dispatch |
//! | REQ-9 (unimplemented-op error contract) | SHIPPED | every non-Sprint-C.7 trait method (neg/gelu/dropout/broadcast_*/transpose/permute/layernorm/slice_*/embed_*/scale/relu_backward/gelu_backward/index_select/masked_*/has_inf_nan) returns `Err(InvalidArgument { message: "MSL kernel needed: <op> — follow-up #626" })`; consumer trait impl compiles so `register_gpu_backend(Box::new(MtlBackend))` succeeds, invoking an unimplemented op surfaces the structured error |
//! | REQ-10 (`init_mps_backend_metal` global registration) | SHIPPED | `pub fn init_mps_backend_metal` constructs `MtlBackend::new()` and calls `ferrotorch_core::gpu_dispatch::register_gpu_backend(Box::new(backend))`; consumer `pub fn init_mps_backend` in `lib.rs` delegates here on macOS, and `ferrotorch/src/lib.rs:137` re-exports it |

#![cfg(target_os = "macos")]

use std::sync::Arc;

use ferrotorch_core::dtype::DType;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::gpu_dispatch::{GpuBackend, GpuBufferHandle};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLFunction, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use crate::kernels;

// ---------------------------------------------------------------------------
// MtlBuffer — owned Metal buffer handle
// ---------------------------------------------------------------------------

/// Newtype around a retained `MTLBuffer` so it can live in a `GpuBufferHandle`
/// via `Box<dyn Any + Send + Sync>`.
///
/// `MTLBuffer` is reference-counted by `objc2::rc::Retained`; wrapping in
/// `Arc` makes the `Send + Sync` bound trivially satisfied because we only
/// access buffer contents through the Metal command queue (which serialises
/// access internally).
///
/// # Safety
///
/// `objc2-metal` marks `MTLBuffer` as `!Send + !Sync` on non-macOS (the type
/// doesn't exist there). On macOS the Metal runtime guarantees thread-safe
/// reference-counting for retained objects; the `Arc` wrapper here expresses
/// that invariant at the Rust type level.
pub struct MtlBuffer {
    pub(crate) inner: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Number of elements (not bytes) in the buffer.
    pub(crate) elem_count: usize,
}

impl std::fmt::Debug for MtlBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtlBuffer")
            .field("elem_count", &self.elem_count)
            .finish_non_exhaustive()
    }
}

// SAFETY: Metal buffers use ObjC ARC for memory management, which is
// thread-safe. Access to buffer contents is serialised through the command
// queue; no two Rust threads write to the same buffer concurrently.
unsafe impl Send for MtlBuffer {}
unsafe impl Sync for MtlBuffer {}

// ---------------------------------------------------------------------------
// Compiled pipeline cache
// ---------------------------------------------------------------------------

/// Lazily-compiled `MTLComputePipelineState` for a single MSL kernel function.
struct Pipeline {
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

// SAFETY: Same rationale as MtlBuffer — ObjC ARC, Metal serialises access.
unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

/// All 10 compiled pipelines, cached after `MtlBackend::new()`.
struct Pipelines {
    matmul_f32: Pipeline,
    bmm_f32: Pipeline,
    add_f32: Pipeline,
    sub_f32: Pipeline,
    mul_f32: Pipeline,
    div_f32: Pipeline,
    relu_f32: Pipeline,
    sigmoid_f32: Pipeline,
    softmax_f32: Pipeline,
    sum_axis_f32: Pipeline,
}

// ---------------------------------------------------------------------------
// Helper: compile one MTLLibrary + extract one function + build pipeline
// ---------------------------------------------------------------------------

fn compile_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
    fn_name: &str,
) -> FerrotorchResult<Pipeline> {
    // All objc2-metal calls go through the safe Rust bindings from
    // objc2-metal 0.3.2 — the methods used here are not marked `unsafe`.
    let src = NSString::from_str(source);
    let options = None; // use default MTLCompileOptions
    let lib: Retained<ProtocolObject<dyn MTLLibrary>> = device
        .newLibraryWithSource_options_error(&src, options)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("MSL compile failed for `{fn_name}`: {e:?}"),
        })?;

    let name = NSString::from_str(fn_name);
    let func: Retained<ProtocolObject<dyn MTLFunction>> = lib
        .newFunctionWithName(&name)
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("MSL function `{fn_name}` not found in library"),
        })?;

    let pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>> = device
        .newComputePipelineStateWithFunction_error(&func)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("MTLComputePipelineState creation failed for `{fn_name}`: {e:?}"),
        })?;

    Ok(Pipeline { state: pipeline })
}

// ---------------------------------------------------------------------------
// Threadgroup-width helper (#1101)
// ---------------------------------------------------------------------------

// Power-of-two threadgroup width is required by the in-kernel
// `stride = tcount / 2; stride >>= 1` reduction (softmax_f32, sum_axis_f32).
//
// The reduction loop in those MSL kernels assumes `tcount` is a power of two.
// When `tcount` is not pow-2, the first stride `tcount / 2` rounds *down*, so
// elements in the upper half (indices `2 * stride .. tcount`) are silently
// dropped — producing wrong-but-not-NaN row maxes and partial sums on Apple
// Silicon. PyTorch parity (§3) forbids that silent corruption, so the
// dispatcher rounds the threadgroup width *up* to the next power of two and
// caps it at the Metal threadgroup limit of 1024. The kernel side then
// handles inactive threads (`tid >= cols` / `tid >= axis_len`) by leaving
// the per-thread sentinels untouched (`-INFINITY` for max, `0.0` for sum)
// — the strided init loop short-circuits for those threads and the reduction
// reads the sentinels but they are identity elements for the operation.
//
// Behavioural contract:
//   pow2_tg_width(0)    = 1     // sentinel: zero-width dispatch is a bug
//                                // upstream; we still return a valid Metal
//                                // threadgroup width.
//   pow2_tg_width(1)    = 1
//   pow2_tg_width(13)   = 16
//   pow2_tg_width(257)  = 512
//   pow2_tg_width(1023) = 1024
//   pow2_tg_width(1024) = 1024
//   pow2_tg_width(2000) = 1024  // capped
fn pow2_tg_width(n: usize) -> usize {
    n.min(1024).next_power_of_two()
}

// ---------------------------------------------------------------------------
// MtlBackend
// ---------------------------------------------------------------------------

/// Apple Metal backend implementing [`GpuBackend`] for ferrotorch-mps.
///
/// Holds a reference to the system default Metal device, a command queue,
/// and the compiled pipeline states for all 10 Sprint C.7 kernels.
pub struct MtlBackend {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: Pipelines,
}

// SAFETY: ObjC ARC + command queue serialises all Metal API access.
unsafe impl Send for MtlBackend {}
unsafe impl Sync for MtlBackend {}

impl std::fmt::Debug for MtlBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtlBackend")
            .field("device", &"<MTLDevice>")
            .finish()
    }
}

impl MtlBackend {
    /// Create a new `MtlBackend`, compiling all 10 MSL kernels eagerly.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::DeviceUnavailable`] if no Metal device is
    /// found (VM, CI without GPU passthrough, or non-Apple hardware).
    /// Returns [`FerrotorchError::InvalidArgument`] if any MSL kernel fails
    /// to compile — which indicates a ferrotorch bug, not a user error.
    pub fn new() -> FerrotorchResult<Self> {
        // MTLCreateSystemDefaultDevice returns None when no Metal device is
        // present; the `?` propagates that as DeviceUnavailable.
        let device: Retained<ProtocolObject<dyn MTLDevice>> =
            MTLCreateSystemDefaultDevice().ok_or(FerrotorchError::DeviceUnavailable)?;

        let queue: Retained<ProtocolObject<dyn MTLCommandQueue>> = device
            .newCommandQueue()
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLDevice::newCommandQueue returned nil".into(),
            })?;

        // Compile all MSL sources — fail fast on any compilation error.
        let mat = compile_pipeline(&device, kernels::MATMUL_F32, "matmul_f32")?;
        let bmm = compile_pipeline(&device, kernels::BMM_F32, "bmm_f32")?;
        let add = compile_pipeline(&device, kernels::ELEMENTWISE_F32, "add_f32")?;
        let sub = compile_pipeline(&device, kernels::ELEMENTWISE_F32, "sub_f32")?;
        let mul = compile_pipeline(&device, kernels::ELEMENTWISE_F32, "mul_f32")?;
        let div_p = compile_pipeline(&device, kernels::ELEMENTWISE_F32, "div_f32")?;
        let relu = compile_pipeline(&device, kernels::ACTIVATIONS_F32, "relu_f32")?;
        let sigmoid = compile_pipeline(&device, kernels::ACTIVATIONS_F32, "sigmoid_f32")?;
        let softmax = compile_pipeline(&device, kernels::SOFTMAX_F32, "softmax_f32")?;
        let sum_ax = compile_pipeline(&device, kernels::SUM_AXIS_F32, "sum_axis_f32")?;

        Ok(Self {
            device,
            queue,
            pipelines: Pipelines {
                matmul_f32: mat,
                bmm_f32: bmm,
                add_f32: add,
                sub_f32: sub,
                mul_f32: mul,
                div_f32: div_p,
                relu_f32: relu,
                sigmoid_f32: sigmoid,
                softmax_f32: softmax,
                sum_axis_f32: sum_ax,
            },
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Allocate a new `MTLBuffer` of `byte_len` bytes (shared storage mode).
    fn alloc_buffer(&self, byte_len: usize, elem_count: usize) -> FerrotorchResult<Arc<MtlBuffer>> {
        // Metal manages the buffer memory; Rust holds a retained ref.
        let buf: Retained<ProtocolObject<dyn MTLBuffer>> = self
            .device
            .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!("MTLDevice::newBufferWithLength({byte_len}) returned nil"),
            })?;
        Ok(Arc::new(MtlBuffer {
            inner: buf,
            elem_count,
        }))
    }

    /// Wrap an `Arc<MtlBuffer>` in a `GpuBufferHandle`.
    fn wrap_buffer(buf: Arc<MtlBuffer>, device_ordinal: usize) -> GpuBufferHandle {
        let len = buf.elem_count;
        // This backend's buffers are always f32 (see `buffer_elem_size`); tag
        // the handle accordingly so the authoritative dtype matches the bytes.
        GpuBufferHandle::new(Box::new(buf), device_ordinal, len, DType::F32)
    }

    /// Downcast a `GpuBufferHandle` to `&Arc<MtlBuffer>`.
    fn downcast_buf(handle: &GpuBufferHandle) -> FerrotorchResult<&Arc<MtlBuffer>> {
        handle
            .downcast_ref::<Arc<MtlBuffer>>()
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "GpuBufferHandle does not contain an Arc<MtlBuffer> (wrong backend?)"
                    .into(),
            })
    }

    /// Commit a command buffer and wait for completion (synchronous).
    ///
    /// All current kernels use synchronous dispatch so callers can read
    /// results immediately. A future async path can replace this with
    /// addScheduledHandler + addCompletedHandler without changing the API.
    fn commit_and_wait(cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>) {
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
    }

    /// Launch a 1-D elementwise binary kernel (add/sub/mul/div).
    fn launch_binary_f32(
        &self,
        pipeline: &Pipeline,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::downcast_buf(a)?;
        let b_buf = Self::downcast_buf(b)?;
        let n = a.len();

        let out_buf = self.alloc_buffer(n * 4, n)?;

        let cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>> = self
            .queue
            .commandBuffer()
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandQueue::commandBuffer returned nil".into(),
            })?;

        let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandBuffer::computeCommandEncoder returned nil".into(),
            })?;

        let n_u32 = n as u32;
        unsafe {
            enc.setComputePipelineState(&pipeline.state);
            enc.setBuffer_offset_atIndex(Some(&a_buf.inner), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b_buf.inner), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&out_buf.inner), 0, 2);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&n_u32 as *const u32 as *mut _),
                4,
                3,
            );

            let tg_size = pipeline.state.maxTotalThreadsPerThreadgroup().min(256);
            let grid = MTLSize {
                width: n,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: tg_size,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
        }

        Self::commit_and_wait(&cmd_buf);
        Ok(Self::wrap_buffer(out_buf, a.device_ordinal()))
    }

    /// Launch a 1-D elementwise unary kernel (relu/sigmoid).
    fn launch_unary_f32(
        &self,
        pipeline: &Pipeline,
        a: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::downcast_buf(a)?;
        let n = a.len();
        let out_buf = self.alloc_buffer(n * 4, n)?;

        let cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>> = self
            .queue
            .commandBuffer()
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandQueue::commandBuffer returned nil".into(),
            })?;

        let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandBuffer::computeCommandEncoder returned nil".into(),
            })?;

        let n_u32 = n as u32;
        unsafe {
            enc.setComputePipelineState(&pipeline.state);
            enc.setBuffer_offset_atIndex(Some(&a_buf.inner), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&out_buf.inner), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&n_u32 as *const u32 as *mut _),
                4,
                2,
            );

            let tg_size = pipeline.state.maxTotalThreadsPerThreadgroup().min(256);
            let grid = MTLSize {
                width: n,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: tg_size,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
        }

        Self::commit_and_wait(&cmd_buf);
        Ok(Self::wrap_buffer(out_buf, a.device_ordinal()))
    }
}

// ---------------------------------------------------------------------------
// GpuBackend implementation
// ---------------------------------------------------------------------------

impl GpuBackend for MtlBackend {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    // -- Memory management ---------------------------------------------------

    fn cpu_to_gpu(
        &self,
        data: &[u8],
        dtype: DType,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let elem_size = dtype.size_of();
        let elem_count = data.len().checked_div(elem_size).unwrap_or(0);
        let buf = self.alloc_buffer(data.len(), elem_count)?;

        // Shared-mode buffers expose a CPU-accessible pointer directly.
        // SAFETY: buffer is exclusively owned here; no GPU command is in
        // flight at this point. `contents()` returns `NonNull<c_void>` —
        // non-null by construction for shared-mode buffers.
        unsafe {
            let ptr = buf.inner.contents();
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr().cast::<u8>(), data.len());
        }

        Ok(Self::wrap_buffer(buf, device))
    }

    fn gpu_to_cpu(&self, handle: &GpuBufferHandle) -> FerrotorchResult<Vec<u8>> {
        let buf = Self::downcast_buf(handle)?;
        let byte_len = buf.inner.length();

        // SAFETY: Shared-mode buffer contents are CPU-accessible after the
        // most recent command buffer completes (guaranteed by commit_and_wait
        // in every kernel dispatch path). `contents()` returns
        // `NonNull<c_void>` — non-null by construction.
        let slice = unsafe {
            let ptr = buf.inner.contents();
            std::slice::from_raw_parts(ptr.as_ptr().cast::<u8>(), byte_len)
        };
        Ok(slice.to_vec())
    }

    fn clone_buffer(&self, handle: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let src_bytes = self.gpu_to_cpu(handle)?;
        self.cpu_to_gpu(&src_bytes, handle.dtype(), handle.device_ordinal())
    }

    fn alloc_zeros(
        &self,
        len: usize,
        dtype: DType,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let byte_len = len * dtype.size_of();
        let buf = self.alloc_buffer(byte_len, len)?;

        // Shared-mode buffers are zero-initialised by the Metal runtime.
        // Explicitly zero for clarity / defence-in-depth. `contents()` returns
        // `NonNull<c_void>` — non-null by construction.
        unsafe {
            let ptr = buf.inner.contents();
            std::ptr::write_bytes(ptr.as_ptr().cast::<u8>(), 0u8, byte_len);
        }

        Ok(Self::wrap_buffer(buf, device))
    }

    fn buffer_elem_size(&self, _handle: &GpuBufferHandle) -> usize {
        // MPS buffers in this backend are always f32 (4 bytes).
        4
    }

    // -- Elementwise f32 binary ops ------------------------------------------

    fn add_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        self.launch_binary_f32(&self.pipelines.add_f32, a, b)
    }

    fn sub_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        self.launch_binary_f32(&self.pipelines.sub_f32, a, b)
    }

    fn mul_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        self.launch_binary_f32(&self.pipelines.mul_f32, a, b)
    }

    fn div_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        self.launch_binary_f32(&self.pipelines.div_f32, a, b)
    }

    fn neg_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: neg_f32 — follow-up #626".into(),
        })
    }

    // -- Unary activations f32 -----------------------------------------------

    fn relu_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        self.launch_unary_f32(&self.pipelines.relu_f32, a)
    }

    fn sigmoid_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        self.launch_unary_f32(&self.pipelines.sigmoid_f32, a)
    }

    // -- Linalg f32 ----------------------------------------------------------

    fn matmul_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::downcast_buf(a)?;
        let b_buf = Self::downcast_buf(b)?;
        let out_len = m * n;
        let out_buf = self.alloc_buffer(out_len * 4, out_len)?;

        let cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>> = self
            .queue
            .commandBuffer()
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandQueue::commandBuffer returned nil".into(),
            })?;

        let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandBuffer::computeCommandEncoder returned nil".into(),
            })?;

        let m_u32 = m as u32;
        let k_u32 = k as u32;
        let n_u32 = n as u32;

        unsafe {
            enc.setComputePipelineState(&self.pipelines.matmul_f32.state);
            enc.setBuffer_offset_atIndex(Some(&a_buf.inner), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b_buf.inner), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&out_buf.inner), 0, 2);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&m_u32 as *const u32 as *mut _),
                4,
                3,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&k_u32 as *const u32 as *mut _),
                4,
                4,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&n_u32 as *const u32 as *mut _),
                4,
                5,
            );

            let tg = MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            };
            let grid = MTLSize {
                width: n,
                height: m,
                depth: 1,
            };
            enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
        }

        Self::commit_and_wait(&cmd_buf);
        Ok(Self::wrap_buffer(out_buf, a.device_ordinal()))
    }

    fn bmm_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::downcast_buf(a)?;
        let b_buf = Self::downcast_buf(b)?;
        let out_len = batch * m * n;
        let out_buf = self.alloc_buffer(out_len * 4, out_len)?;

        let cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>> = self
            .queue
            .commandBuffer()
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandQueue::commandBuffer returned nil".into(),
            })?;

        let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandBuffer::computeCommandEncoder returned nil".into(),
            })?;

        let batch_u32 = batch as u32;
        let m_u32 = m as u32;
        let k_u32 = k as u32;
        let n_u32 = n as u32;

        unsafe {
            enc.setComputePipelineState(&self.pipelines.bmm_f32.state);
            enc.setBuffer_offset_atIndex(Some(&a_buf.inner), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b_buf.inner), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&out_buf.inner), 0, 2);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&batch_u32 as *const u32 as *mut _),
                4,
                3,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&m_u32 as *const u32 as *mut _),
                4,
                4,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&k_u32 as *const u32 as *mut _),
                4,
                5,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&n_u32 as *const u32 as *mut _),
                4,
                6,
            );

            let tg = MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            };
            let grid = MTLSize {
                width: n,
                height: m,
                depth: batch,
            };
            enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
        }

        Self::commit_and_wait(&cmd_buf);
        Ok(Self::wrap_buffer(out_buf, a.device_ordinal()))
    }

    // -- Softmax f32 ---------------------------------------------------------

    fn softmax_f32(
        &self,
        a: &GpuBufferHandle,
        rows: usize,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::downcast_buf(a)?;
        let out_len = rows * cols;
        let out_buf = self.alloc_buffer(out_len * 4, out_len)?;

        let cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>> = self
            .queue
            .commandBuffer()
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandQueue::commandBuffer returned nil".into(),
            })?;

        let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandBuffer::computeCommandEncoder returned nil".into(),
            })?;

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        // Each threadgroup handles one row. The kernel's tree reduction
        // (`stride = tcount / 2; stride >>= 1`) requires a pow-2 threadgroup
        // width; pow2_tg_width rounds up and caps at the Metal limit. See
        // pow2_tg_width docs and #1101 for the bug this fixes.
        let tg_w = pow2_tg_width(cols);

        unsafe {
            enc.setComputePipelineState(&self.pipelines.softmax_f32.state);
            enc.setBuffer_offset_atIndex(Some(&a_buf.inner), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&out_buf.inner), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&rows_u32 as *const u32 as *mut _),
                4,
                2,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&cols_u32 as *const u32 as *mut _),
                4,
                3,
            );

            // One threadgroup per row.
            let grid = MTLSize {
                width: rows,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
        }

        Self::commit_and_wait(&cmd_buf);
        Ok(Self::wrap_buffer(out_buf, a.device_ordinal()))
    }

    // -- Reductions f32 ------------------------------------------------------

    fn sum_f32(&self, a: &GpuBufferHandle, len: usize) -> FerrotorchResult<GpuBufferHandle> {
        // Reduce full tensor to scalar: treat as (1, len, 1) sum_axis.
        self.sum_axis_f32(a, &[len], 0)
    }

    fn sum_axis_f32(
        &self,
        a: &GpuBufferHandle,
        shape: &[usize],
        axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::downcast_buf(a)?;

        let outer: usize = shape[..axis].iter().product::<usize>().max(1);
        let axis_len: usize = shape.get(axis).copied().unwrap_or(1);
        let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
        let out_len = outer * inner;
        let out_buf = self.alloc_buffer(out_len * 4, out_len)?;

        let cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>> = self
            .queue
            .commandBuffer()
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandQueue::commandBuffer returned nil".into(),
            })?;

        let enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandBuffer::computeCommandEncoder returned nil".into(),
            })?;

        let outer_u32 = outer as u32;
        let axis_u32 = axis_len as u32;
        let inner_u32 = inner as u32;
        // The kernel's tree reduction (`stride = tcount / 2; stride >>= 1`)
        // requires a pow-2 threadgroup width; pow2_tg_width rounds up and
        // caps at the Metal limit. See pow2_tg_width docs and #1101.
        let tg_w = pow2_tg_width(axis_len);

        unsafe {
            enc.setComputePipelineState(&self.pipelines.sum_axis_f32.state);
            enc.setBuffer_offset_atIndex(Some(&a_buf.inner), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&out_buf.inner), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&outer_u32 as *const u32 as *mut _),
                4,
                2,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&axis_u32 as *const u32 as *mut _),
                4,
                3,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&inner_u32 as *const u32 as *mut _),
                4,
                4,
            );

            // One threadgroup per output element.
            let grid = MTLSize {
                width: out_len,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
        }

        Self::commit_and_wait(&cmd_buf);
        Ok(Self::wrap_buffer(out_buf, a.device_ordinal()))
    }

    // -- Required abstract methods without Sprint C.7 implementations --------
    // These return structured errors matching PyTorch's NotImplementedError
    // for unregistered backends. Follow-up issues are tracked per method.

    fn gelu_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: gelu_f32 — follow-up #626".into(),
        })
    }

    fn dropout_f32(
        &self,
        _a: &GpuBufferHandle,
        _threshold: u32,
        _scale: f32,
        _seed: u32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: dropout_f32 — follow-up #626".into(),
        })
    }

    // -- GpuBackend trait surface that has accumulated upstream without MSL
    //    implementations. Each method returns Err so MPSBackend is a valid
    //    trait impl on macOS-CI and the build stays green. Real Metal
    //    implementations are tracked in follow-up #626 (the same issue every
    //    other "MSL kernel needed" stub above references).

    fn broadcast_add_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: broadcast_add_f32 — follow-up #626".into(),
        })
    }

    fn broadcast_sub_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: broadcast_sub_f32 — follow-up #626".into(),
        })
    }

    fn broadcast_mul_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: broadcast_mul_f32 — follow-up #626".into(),
        })
    }

    fn broadcast_div_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: broadcast_div_f32 — follow-up #626".into(),
        })
    }

    fn transpose_2d_f32(
        &self,
        _a: &GpuBufferHandle,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: transpose_2d_f32 — follow-up #626".into(),
        })
    }

    fn permute_0213_f32(
        &self,
        _a: &GpuBufferHandle,
        _d0: usize,
        _d1: usize,
        _d2: usize,
        _d3: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: permute_0213_f32 — follow-up #626".into(),
        })
    }

    fn layernorm_f32(
        &self,
        _input: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _bias: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _eps: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: layernorm_f32 — follow-up #626".into(),
        })
    }

    fn slice_write_f32(
        &self,
        _src: &GpuBufferHandle,
        _dst: &mut GpuBufferHandle,
        _n_batch: usize,
        _d: usize,
        _max_len: usize,
        _pos: usize,
    ) -> FerrotorchResult<()> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: slice_write_f32 — follow-up #626".into(),
        })
    }

    fn slice_read_f32(
        &self,
        _src: &GpuBufferHandle,
        _n_batch: usize,
        _d: usize,
        _len: usize,
        _max_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: slice_read_f32 — follow-up #626".into(),
        })
    }

    fn embed_lookup_f32(
        &self,
        _idx: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: embed_lookup_f32 — follow-up #626".into(),
        })
    }

    fn embed_lookup_batch_f32(
        &self,
        _indices: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _n: usize,
        _d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: embed_lookup_batch_f32 — follow-up #626".into(),
        })
    }

    fn scatter_add_rows_f32(
        &self,
        _grad_output: &GpuBufferHandle,
        _indices: &GpuBufferHandle,
        _num_embeddings: usize,
        _d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: scatter_add_rows_f32 — follow-up #626".into(),
        })
    }

    fn scale_f32(&self, _a: &GpuBufferHandle, _scalar: f32) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: scale_f32 — follow-up #626".into(),
        })
    }

    fn relu_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: relu_backward_f32 — follow-up #626".into(),
        })
    }

    fn gelu_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: gelu_backward_f32 — follow-up #626".into(),
        })
    }

    fn gelu_backward_erf_f32(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: gelu_backward_erf_f32 — follow-up #626".into(),
        })
    }

    fn index_select_1d_f32(
        &self,
        _input: &GpuBufferHandle,
        _indices: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: index_select_1d_f32 — follow-up #626".into(),
        })
    }

    fn scatter_add_1d_f32(
        &self,
        _grad_output: &GpuBufferHandle,
        _indices: &GpuBufferHandle,
        _input_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: scatter_add_1d_f32 — follow-up #626".into(),
        })
    }

    fn index_select_dim_f32(
        &self,
        _input: &GpuBufferHandle,
        _indices: &GpuBufferHandle,
        _outer: usize,
        _in_dim_size: usize,
        _out_dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: index_select_dim_f32 — follow-up #626".into(),
        })
    }

    fn masked_fill_f32(
        &self,
        _input: &GpuBufferHandle,
        _mask: &GpuBufferHandle,
        _value: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: masked_fill_f32 — follow-up #626".into(),
        })
    }

    fn masked_zero_f32(
        &self,
        _grad: &GpuBufferHandle,
        _mask: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: masked_zero_f32 — follow-up #626".into(),
        })
    }

    fn has_inf_nan_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<bool> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: has_inf_nan_f32 — follow-up #626".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// init_mps_backend entry point
// ---------------------------------------------------------------------------

/// Initialize the MPS Metal backend and register it with `ferrotorch-core`.
///
/// Call once at startup. Returns [`FerrotorchError::DeviceUnavailable`] if no
/// Metal device is present (non-macOS platform or VM without GPU passthrough).
///
/// # Errors
///
/// - [`FerrotorchError::DeviceUnavailable`]: no Metal device found.
/// - [`FerrotorchError::InvalidArgument`]: MSL compilation failed (ferrotorch bug).
pub fn init_mps_backend_metal() -> FerrotorchResult<()> {
    let backend = MtlBackend::new()?;
    ferrotorch_core::gpu_dispatch::register_gpu_backend(Box::new(backend)).map_err(|_| {
        // `register_gpu_backend` uses `OnceLock::set` — the only failure mode
        // is that a backend has already been registered. The Err payload is
        // the rejected `Box<dyn GpuBackend>`, which doesn't implement Display.
        FerrotorchError::InvalidArgument {
            message: "MPS backend registration failed: a GPU backend is already registered".into(),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests are macOS-only (entire module is cfg(target_os = "macos")).
    // On CI without Apple hardware they are excluded by the cfg gate.
    //
    // Note: the `pow2_tg_width_*` tests below are pure-Rust and have no
    // Metal dependency, but they live here because the helper is module-
    // private. They compile and run on macOS only by virtue of the parent
    // `cfg(target_os = "macos")` gate on `pub mod backend` in `lib.rs`.

    /// Pow-2 round-up contract for the threadgroup-width helper (#1101):
    /// non-pow-2 inputs must round up so the in-kernel `stride = tcount/2`
    /// reduction does not silently drop upper-half elements. Cap at 1024
    /// (Metal threadgroup limit).
    #[test]
    fn pow2_tg_width_rounds_up_for_non_powers_of_two() {
        assert_eq!(pow2_tg_width(0), 1);
        assert_eq!(pow2_tg_width(1), 1);
        assert_eq!(pow2_tg_width(2), 2);
        assert_eq!(pow2_tg_width(13), 16);
        assert_eq!(pow2_tg_width(257), 512);
        assert_eq!(pow2_tg_width(1023), 1024);
        assert_eq!(pow2_tg_width(1024), 1024);
        assert_eq!(pow2_tg_width(2000), 1024);
    }

    /// Pow-2 inputs must round-trip unchanged — the helper is idempotent
    /// for already-pow-2 widths within the [1, 1024] Metal cap.
    #[test]
    fn pow2_tg_width_passes_through_powers_of_two() {
        for &n in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            assert_eq!(
                pow2_tg_width(n),
                n,
                "pow-2 input {n} must round-trip unchanged"
            );
        }
    }

    /// Verify [`MtlBackend::new`] either succeeds or returns
    /// [`FerrotorchError::DeviceUnavailable`]. Never panics, never returns an
    /// unexpected error variant.
    #[test]
    fn mtl_backend_new_succeeds_or_unavailable() {
        match MtlBackend::new() {
            Ok(b) => {
                // If a Metal device exists, the debug repr should be non-empty.
                let dbg = format!("{b:?}");
                assert!(dbg.contains("MtlBackend"));
            }
            Err(FerrotorchError::DeviceUnavailable) => {
                // Acceptable: CI macOS runner without GPU passthrough.
            }
            Err(e) => {
                panic!("unexpected error from MtlBackend::new(): {e:?}");
            }
        }
    }

    /// Round-trip f32 data through `cpu_to_gpu` → `gpu_to_cpu` on a real
    /// Metal device. `cascade_skip` if no Metal device is present (CI
    /// without GPU).
    #[test]
    fn mtl_buffer_round_trip() {
        let backend = match MtlBackend::new() {
            Ok(b) => b,
            Err(FerrotorchError::DeviceUnavailable) => {
                eprintln!(
                    "  [cascade_skip] mtl_buffer_round_trip — no Metal device, \
                     tracking issue #626"
                );
                return;
            }
            Err(e) => panic!("MtlBackend::new() error: {e:?}"),
        };

        let src: Vec<f32> = vec![1.0_f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = src.iter().flat_map(|f| f.to_le_bytes()).collect();
        let handle = backend
            .cpu_to_gpu(&bytes, DType::F32, 0)
            .expect("cpu_to_gpu");
        assert_eq!(handle.len(), 4);

        let back = backend.gpu_to_cpu(&handle).expect("gpu_to_cpu");
        assert_eq!(back.len(), bytes.len());
        let floats: Vec<f32> = back
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(floats, src);
    }
}
