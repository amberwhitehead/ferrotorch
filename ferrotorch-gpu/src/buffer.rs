//! GPU memory buffer with pool-aware Drop.
//!
//! [`CudaBuffer`] owns a region of device memory via `cudarc::driver::CudaSlice`
//! and tracks its length and originating device ordinal. When dropped, pooled
//! buffers are returned to the global GPU memory pool for reuse instead of
//! being freed back to the CUDA driver.
//!
//! ## REQ status (per `.design/ferrotorch-gpu/buffer.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream
//! cites) live in the design doc; this synopsis is a one-line summary per
//! REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`CudaBuffer<T>` shape) | SHIPPED | `pub struct CudaBuffer<T> in buffer.rs` with `Option<CudaSlice<T>> + len + alloc_len + device_ordinal + pool_fn`; consumer `ferrotorch-jit/src/fusion_gpu.rs` downcasts `GpuBufferHandle` to `&CudaBuffer<f32>` |
//! | REQ-2 (pool-return fns) | SHIPPED | `type PoolReturnFn<T>` + `fn return_f32 / return_f64 in buffer.rs`; consumer `crate::pool::pool_return::<CudaSlice<f32>>` wiring fires from buffer Drop |
//! | REQ-3 (`new_pooled`) | SHIPPED | `impl CudaBuffer<f32>::new_pooled / impl CudaBuffer<f64>::new_pooled in buffer.rs`; consumer `crate::transfer::alloc_zeros_f32` builds pooled buffers via this path |
//! | REQ-4 (`Drop`) | SHIPPED | `impl<T> Drop for CudaBuffer<T> in buffer.rs`; consumer every dropped buffer in `ferrotorch-llama/src/gpu.rs` flows through this Drop (KV cache, attention outputs) |
//! | REQ-5 (accessors) | SHIPPED | `pub fn len / alloc_len / is_empty / device_ordinal / inner / inner_mut in buffer.rs`; consumer `ferrotorch-diffusion/src/gpu/vae_encoder.rs` consumes `&CudaBuffer<f32>` via `x.inner()` for kernel launches |
//! | REQ-6 (`Debug` impl) | SHIPPED | `impl<T> std::fmt::Debug for CudaBuffer<T> in buffer.rs`; consumer structured logging via `format!("{buf:?}")` across the workspace |
//! | REQ-7 (host-only stub) | SHIPPED | `#[cfg(not(feature = "cuda"))] pub struct CudaBuffer<T> in buffer.rs`; consumer host-only build path keeps `use ferrotorch_gpu::CudaBuffer` valid at `ferrotorch-jit/src/fusion_gpu.rs` |

#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

/// Type-erased function pointer that returns a `CudaSlice<T>` to the pool.
/// Stored as `Option` — `None` means "don't pool, just drop normally."
#[cfg(feature = "cuda")]
type PoolReturnFn<T> = Option<fn(usize, usize, CudaSlice<T>)>;

/// Return a `CudaSlice<f32>` to the global pool.
#[cfg(feature = "cuda")]
fn return_f32(device: usize, len: usize, slice: CudaSlice<f32>) {
    crate::pool::pool_return::<CudaSlice<f32>>(device, len, 4, slice);
}

/// Return a `CudaSlice<f64>` to the global pool.
#[cfg(feature = "cuda")]
fn return_f64(device: usize, len: usize, slice: CudaSlice<f64>) {
    crate::pool::pool_return::<CudaSlice<f64>>(device, len, 8, slice);
}

/// Owned GPU memory buffer holding `len` elements of type `T`.
///
/// When `pool_fn` is `Some`, dropping returns the inner `CudaSlice` to the
/// global pool ([`crate::pool`]) instead of freeing GPU memory.
///
/// `alloc_len` is the rounded allocation size used as the pool key.
/// `len` is the logical element count visible to callers.
#[cfg(feature = "cuda")]
pub struct CudaBuffer<T> {
    /// The underlying CUDA device memory. Wrapped in `Option` so
    /// `Drop` can `take()` it without double-free.
    pub(crate) data: Option<CudaSlice<T>>,
    pub(crate) len: usize,
    /// Rounded allocation length — used as the pool key so that
    /// buffers are always findable on pool lookup.
    pub(crate) alloc_len: usize,
    pub(crate) device_ordinal: usize,
    /// If `Some`, this function is called in Drop to return the slice
    /// to the pool. If `None`, CudaSlice::Drop frees normally.
    pub(crate) pool_fn: PoolReturnFn<T>,
}

/// Helper to create a pooled f32 buffer.
#[cfg(feature = "cuda")]
impl CudaBuffer<f32> {
    /// Create a pooled f32 buffer that returns to the global pool on drop.
    ///
    /// `alloc_len` is the rounded allocation size used as the pool key.
    /// `len` is the logical element count visible to callers.
    pub(crate) fn new_pooled(
        slice: CudaSlice<f32>,
        len: usize,
        alloc_len: usize,
        device: usize,
    ) -> Self {
        Self {
            data: Some(slice),
            len,
            alloc_len,
            device_ordinal: device,
            pool_fn: Some(return_f32),
        }
    }
}

#[cfg(feature = "cuda")]
impl CudaBuffer<f64> {
    /// Create a pooled f64 buffer that returns to the global pool on drop.
    ///
    /// `alloc_len` is the rounded allocation size used as the pool key.
    /// `len` is the logical element count visible to callers.
    pub(crate) fn new_pooled(
        slice: CudaSlice<f64>,
        len: usize,
        alloc_len: usize,
        device: usize,
    ) -> Self {
        Self {
            data: Some(slice),
            len,
            alloc_len,
            device_ordinal: device,
            pool_fn: Some(return_f64),
        }
    }
}

#[cfg(feature = "cuda")]
impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        if let Some(slice) = self.data.take()
            && let Some(return_fn) = self.pool_fn
        {
            // Use alloc_len (rounded) as the pool key so the buffer
            // is findable on the next pool_take with the same rounded len.
            return_fn(self.device_ordinal, self.alloc_len, slice);
        }
        // else: CudaSlice::Drop fires naturally (cuMemFreeAsync)
    }
}

#[cfg(feature = "cuda")]
impl<T> CudaBuffer<T> {
    /// Number of logical elements in this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Rounded allocation length used as the pool key.
    ///
    /// For pooled buffers, this is `round_len(len)`. For non-pooled
    /// buffers, this equals `len`. Stats (hits, misses, returns) use
    /// `len` consistently within the allocator for user-facing reporting;
    /// `alloc_len` is an internal detail for pool key stability.
    #[inline]
    pub fn alloc_len(&self) -> usize {
        self.alloc_len
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The ordinal of the device that owns this memory.
    #[inline]
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    /// Borrow the underlying `CudaSlice` for use with cudarc APIs.
    #[inline]
    pub fn inner(&self) -> &CudaSlice<T> {
        self.data
            .as_ref()
            .expect("CudaBuffer: inner slice already taken")
    }

    /// Mutably borrow the underlying `CudaSlice`.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut CudaSlice<T> {
        self.data
            .as_mut()
            .expect("CudaBuffer: inner slice already taken")
    }
}

#[cfg(feature = "cuda")]
impl<T> std::fmt::Debug for CudaBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBuffer")
            .field("len", &self.len)
            .field("device_ordinal", &self.device_ordinal)
            .field("pooled", &self.pool_fn.is_some())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Stub when `cuda` feature is disabled
// ---------------------------------------------------------------------------

/// Stub `CudaBuffer` when the `cuda` feature is not enabled.
#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct CudaBuffer<T> {
    pub(crate) _phantom: std::marker::PhantomData<T>,
    pub(crate) len: usize,
    pub(crate) device_ordinal: usize,
}

#[cfg(not(feature = "cuda"))]
impl<T> CudaBuffer<T> {
    /// Number of elements in this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The ordinal of the device that owns this memory.
    #[inline]
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }
}
