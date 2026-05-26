//! CUDA device management.
//!
//! [`GpuDevice`] wraps a `cudarc::driver::CudaContext` and its default stream,
//! providing a safe, ergonomic entry point for all GPU operations.
//!
//! ## REQ status (per `.design/ferrotorch-gpu/device.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream
//! cites) live in the design doc; this synopsis is a one-line summary per
//! REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (struct shape) | SHIPPED | `pub struct GpuDevice in device.rs` holds `(Arc<CudaContext>, Arc<CudaStream>, CudaBlas, usize)`; consumer `CudaBackendImpl::new in backend_impl.rs` constructs the struct |
//! | REQ-2 (`GpuDevice::new`) | SHIPPED | `pub fn GpuDevice::new in device.rs` calls `CudaContext::new(ordinal)? + default_stream() + CudaBlas::new(stream)?`; consumer `ferrotorch-jit/src/fusion_gpu.rs` invokes `GpuDevice::new(handle.device_ordinal())` |
//! | REQ-3 (`fork_for_capture`) | SHIPPED | `pub fn fork_for_capture in device.rs` forks the parent stream and rebinds cuBLAS; consumer `crate::graph` capture suite consumes the forked stream |
//! | REQ-4 (accessors) | SHIPPED | `pub fn context / default_stream / stream / blas / ordinal in device.rs`; consumer `crate::conv::gpu_conv2d_f32 in conv.rs` calls `dev.context()` for `module_cache::get_or_compile` and every kernel call site uses `dev.stream()` |
//! | REQ-5 (`impl Clone`) | SHIPPED | `impl Clone for GpuDevice in device.rs` constructs a fresh `CudaBlas` bound to the same shared stream; consumer meta-crate `ferrotorch/src/lib.rs` re-exports `GpuDevice` and tensor-bridge users clone per-thread |
//! | REQ-6 (host-only stub) | SHIPPED | `#[cfg(not(feature = "cuda"))] pub struct GpuDevice in device.rs` with stub `::new` returning `GpuError::NoCudaFeature`; consumer `ferrotorch-data/src/transforms.rs` threads the stub error through cleanly under `--no-default-features` |

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::cublas::CudaBlas;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaStream};

#[cfg(not(feature = "cuda"))]
use crate::error::GpuError;
use crate::error::GpuResult;

/// Handle to a single CUDA GPU device.
///
/// Holds a CUDA context, default stream, and a **cached cuBLAS handle**.
/// The cuBLAS handle is created once and reused for all matmul/bmm ops,
/// eliminating the ~1.7ms `cuModuleLoadData` overhead that occurs when
/// creating a new `CudaBlas` per operation.
#[cfg(feature = "cuda")]
pub struct GpuDevice {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    ordinal: usize,
}

#[cfg(feature = "cuda")]
impl GpuDevice {
    /// Initialize the CUDA device at the given ordinal.
    ///
    /// Creates a fresh `CudaContext`, takes its default stream, and
    /// constructs a cached `CudaBlas` handle bound to that stream so
    /// subsequent matmul/bmm ops reuse it instead of paying the
    /// `cuModuleLoadData` cost per call.
    pub fn new(ordinal: usize) -> GpuResult<Self> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())?;
        Ok(Self {
            ctx,
            stream,
            blas,
            ordinal,
        })
    }

    /// Create a `GpuDevice` with a non-blocking stream forked from the
    /// given device's default stream. The forked stream supports CUDA graph
    /// capture (which the legacy default stream does not).
    pub fn fork_for_capture(parent: &GpuDevice) -> GpuResult<Self> {
        let stream = parent.stream.fork()?;
        let blas = CudaBlas::new(stream.clone())?;
        Ok(Self {
            ctx: Arc::clone(&parent.ctx),
            stream,
            blas,
            ordinal: parent.ordinal,
        })
    }

    /// The shared `CudaContext` underlying this device.
    ///
    /// Required by `cudarc::driver::CudaModule` loaders and other low-level
    /// APIs that need a context handle separate from the stream.
    #[inline]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// The device's default (legacy) stream.
    ///
    /// Prefer [`stream`](Self::stream) which respects the
    /// thread-local stream override set by [`crate::stream::StreamGuard`].
    #[inline]
    pub fn default_stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The active stream for this device on the current thread.
    ///
    /// Returns the thread-local stream set by [`crate::stream::StreamGuard`]
    /// if one is active, otherwise falls back to the device's default stream.
    /// All kernel launches and memory operations should use this.
    #[inline]
    pub fn stream(&self) -> Arc<CudaStream> {
        crate::stream::current_stream_or_default(self)
    }

    /// The cached cuBLAS handle — reused for all matmul/bmm operations.
    #[inline]
    pub fn blas(&self) -> &CudaBlas {
        &self.blas
    }

    /// The 0-based ordinal of this CUDA device, as reported by the driver.
    #[inline]
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
}

#[cfg(feature = "cuda")]
impl Clone for GpuDevice {
    fn clone(&self) -> Self {
        let blas =
            CudaBlas::new(self.stream.clone()).expect("CudaBlas::new failed in GpuDevice::clone");
        Self {
            ctx: Arc::clone(&self.ctx),
            stream: Arc::clone(&self.stream),
            blas,
            ordinal: self.ordinal,
        }
    }
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevice")
            .field("ordinal", &self.ordinal)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Stub when `cuda` feature is disabled
// ---------------------------------------------------------------------------

/// Stub `GpuDevice` when the `cuda` feature is not enabled.
///
/// Every method returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
#[derive(Clone, Debug)]
pub struct GpuDevice {
    ordinal: usize,
}

#[cfg(not(feature = "cuda"))]
impl GpuDevice {
    /// Always returns an error — compile with `features = ["cuda"]`.
    pub fn new(ordinal: usize) -> GpuResult<Self> {
        let _ = ordinal;
        Err(GpuError::NoCudaFeature)
    }

    /// The device ordinal.
    #[inline]
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
}
