//! GPU backend dispatch layer.
//!
//! ferrotorch-core defines the [`GpuBackend`] trait and [`GpuBufferHandle`].
//! ferrotorch-gpu (or any other GPU crate) implements and registers a backend.
//! This avoids circular dependencies: core doesn't depend on gpu.
//!
//! ## REQ status (per `.design/ferrotorch-core/gpu_dispatch.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl `enum CompareOp` + `suffix`; non-test consumer `GpuBackend::compare` dispatch. |
//! | REQ-2 | SHIPPED | impl `GpuRngState`; non-test consumer `GpuBackend::save_rng_state`/`restore_rng_state`. |
//! | REQ-3 | SHIPPED | impl `GpuBufferHandle`; non-test consumer `StorageBuffer::Gpu` variant + every CUDA op. |
//! | REQ-4 | SHIPPED | impl `trait GpuBackend`; non-test consumer `ferrotorch-gpu::CudaBackendImpl`. |
//! | REQ-5 | SHIPPED | impl elementwise method slots `add_f32`, `sub_f32`, `mul_f32`, `neg_f32`, `relu_f32`; non-test consumer `Tensor::accumulate_grad` GPU path + `grad_fns::arithmetic` CUDA branches. |
//! | REQ-6 | SHIPPED | impl broadcast-* trait slots; non-test consumer `grad_fns::arithmetic::add_inner` broadcast branch. |
//! | REQ-7 | SHIPPED | impl `scale_*` trait slots; non-test consumer `grad_fns::arithmetic::scale_tensor`. |
//! | REQ-8 | SHIPPED | impl `strided_copy_*`, `strided_scatter_*` trait slots; non-test consumer `stride_tricks` materialise, `Tensor::to(Cpu)` non-contiguous, `Tensor::materialize_format` GPU fast path. |
//! | REQ-9 | SHIPPED | impl reduction trait slots; non-test consumer `grad_fns::arithmetic::reduce_grad_to_shape`. |
//! | REQ-10 | SHIPPED | impl linalg trait slots (matmul, gemm, syevd, getrf, geqrf, potrf, gesdd, inverse); non-test consumer `ops::linalg::matmul`, `linalg::eigh`. |
//! | REQ-11 | SHIPPED | impl conv2d/conv3d/pooling trait slots; non-test consumer `ferrotorch-nn::Conv2d`. |
//! | REQ-12 | SHIPPED | impl recurrent trait slots; non-test consumer `ferrotorch-nn::LSTM`/`GRU`/`RNN`. |
//! | REQ-13 | SHIPPED | impl FFT trait slots; non-test consumer `ferrotorch-core::fft`. |
//! | REQ-14 | SHIPPED | impl dropout/RNG/`save_rng_state`/`restore_rng_state` trait slots; non-test consumer `nn::Dropout`, `creation::randn`. |
//! | REQ-15 | SHIPPED | impl `masked_fill_dt`, `where_cond`, `masked_select`, `masked_scatter`, `argmax`, `argmin`, `index_select_intidx`, `gather_intidx`; non-test consumer `Tensor::masked_fill`/`masked_select`, `grad_fns::indexing`. |
//! | REQ-16 | SHIPPED | impl cuSPARSE dispatch slots; non-test consumer `SparseTensor::from_dense` CUDA path. |
//! | REQ-17 | SHIPPED | impl int_* trait slots; non-test consumer `int_tensor.rs` op forwarders. |
//! | REQ-18 | SHIPPED | impl `compare`, `bool_*`, cast slots; non-test consumer `bool_tensor.rs` op forwarders. |
//! | REQ-19 | SHIPPED | impl `synchronize`, `stream_count`, `strided_cat`; non-test consumer `CudaBackendImpl::synchronize` override. |
//! | REQ-20 | SHIPPED | impl `register_gpu_backend`, `gpu_backend`, `has_gpu_backend`; non-test consumer `ferrotorch-gpu::backend_impl::register` + every CUDA op in core. |

use std::any::Any;
use std::sync::OnceLock;

use crate::dtype::DType;
use crate::error::{FerrotorchError, FerrotorchResult};

// ---------------------------------------------------------------------------
// CompareOp — the comparison operator for `GpuBackend::compare` (Phase 3b)
// ---------------------------------------------------------------------------

/// The six elementwise comparison operators (`torch.{eq,ne,lt,le,gt,ge}`),
/// passed to [`GpuBackend::compare`] which produces a `DType::Bool`-tagged
/// (u8 0/1) output. PyTorch parity: the comparison's result dtype is `bool`
/// regardless of the input value dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    /// `a == b`.
    Eq,
    /// `a != b`.
    Ne,
    /// `a < b`.
    Lt,
    /// `a <= b`.
    Le,
    /// `a > b`.
    Gt,
    /// `a >= b`.
    Ge,
}

impl CompareOp {
    /// Stable kernel-name suffix (`"eq"`, `"ne"`, …) used to select the PTX
    /// entry point in the CUDA backend.
    #[must_use]
    pub fn suffix(self) -> &'static str {
        match self {
            CompareOp::Eq => "eq",
            CompareOp::Ne => "ne",
            CompareOp::Lt => "lt",
            CompareOp::Le => "le",
            CompareOp::Gt => "gt",
            CompareOp::Ge => "ge",
        }
    }
}

// ---------------------------------------------------------------------------
// GpuRngState — serializable GPU RNG state for checkpoint save/restore
// ---------------------------------------------------------------------------

/// Serializable snapshot of a GPU device's RNG state.
///
/// This is defined in `ferrotorch-core` (not `ferrotorch-gpu`) so that the
/// checkpoint module can save/restore GPU RNG state without depending on the
/// GPU crate directly. The GPU backend implementation is responsible for
/// converting this to/from its internal representation (e.g., `PhiloxState`).
///
/// Fields are crate-private; construct via [`GpuRngState::new`] and read via
/// the [`Self::counter`], [`Self::seed`], [`Self::offset`], [`Self::device`]
/// accessors. Encapsulation lets the layout evolve (e.g. add a Philox-key
/// pair) without a workspace-wide breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuRngState {
    /// RNG counter value.
    pub(crate) counter: u64,
    /// RNG seed.
    pub(crate) seed: u64,
    /// Offset within the current random number group.
    pub(crate) offset: u64,
    /// Device ordinal this state belongs to.
    pub(crate) device: usize,
}

impl GpuRngState {
    /// Construct a new RNG state snapshot.
    #[inline]
    #[must_use]
    pub fn new(counter: u64, seed: u64, offset: u64, device: usize) -> Self {
        Self {
            counter,
            seed,
            offset,
            device,
        }
    }

    /// RNG counter value at the time of the snapshot.
    #[inline]
    #[must_use]
    pub fn counter(&self) -> u64 {
        self.counter
    }

    /// RNG seed used by this generator.
    #[inline]
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Offset within the current random number group.
    #[inline]
    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Device ordinal this RNG state belongs to.
    #[inline]
    #[must_use]
    pub fn device(&self) -> usize {
        self.device
    }
}

/// Opaque handle to GPU memory.
///
/// ferrotorch-core doesn't know what's inside -- the GPU backend provides
/// the concrete type (e.g., `CudaBuffer<f32>`). We store it as
/// `Box<dyn Any + Send + Sync>` for type erasure.
///
/// # The `dtype` tag (PyTorch parity)
///
/// PyTorch's `StorageImpl` holds raw bytes with no dtype; the `ScalarType`
/// tag lives above storage on `TensorImpl::data_type_` as runtime metadata.
/// `Half` and `BFloat16` are both 2 bytes and are told apart *only* by that
/// tag, never by byte width. This handle mirrors that: [`Self::dtype`] is the
/// authoritative element-type tag. Backends dispatch on the tag, not on the
/// erased concrete type or the byte width — so f16 vs. bf16 (both 2 bytes)
/// and (in later phases) i32 vs. f32 (both 4 bytes) never collide.
pub struct GpuBufferHandle {
    pub(crate) inner: Box<dyn Any + Send + Sync>,
    pub(crate) device_ordinal: usize,
    pub(crate) len: usize,
    pub(crate) dtype: DType,
}

impl GpuBufferHandle {
    pub fn new(
        inner: Box<dyn Any + Send + Sync>,
        device_ordinal: usize,
        len: usize,
        dtype: DType,
    ) -> Self {
        Self {
            inner,
            device_ordinal,
            len,
            dtype,
        }
    }

    #[inline]
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    /// The authoritative element-type tag for the bytes this handle owns.
    ///
    /// This is the PyTorch `ScalarType` analog: it, not the byte width or the
    /// erased concrete buffer type, decides how the data is interpreted.
    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.inner.downcast_ref()
    }

    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.inner.downcast_mut()
    }

    /// Consume the handle and extract the inner value as a concrete type.
    pub fn into_inner<T: 'static>(self) -> Result<T, Box<dyn Any + Send + Sync>> {
        self.inner.downcast::<T>().map(|b| *b)
    }
}

impl std::fmt::Debug for GpuBufferHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBufferHandle")
            .field("inner", &"<dyn Any + Send + Sync>")
            .field("device", &self.device_ordinal)
            .field("len", &self.len)
            .field("dtype", &self.dtype)
            .finish()
    }
}

/// True when the f32 GPU `cdist` kernel covers the norm exponent `p`.
///
/// The on-device f32 kernel ([`GpuBackend::cdist_f32`]) handles `p == 1`,
/// `p == 2`, `p == inf`, and general finite `p > 0` (via an on-device `pow`).
/// The only excluded case is the `p == 0` count-of-nonzeros norm, which is
/// delegated to the CPU path. Mirrors the upstream dispatch in
/// `aten/src/ATen/native/cuda/DistanceKernel.cu:230-240` (which special-cases
/// `0`, `1`, `2`, `inf` and falls through to the general kernel otherwise).
#[must_use]
// reason: `p` is a discrete norm selector, not a measured value; PyTorch
// itself special-cases the norm with exact `p == 0.0` / `1.0` / `2.0`
// comparisons at `aten/src/ATen/native/cuda/DistanceKernel.cu:232-238`, so
// the exact compare is the correct upstream-mirroring behaviour here.
#[allow(
    clippy::float_cmp,
    reason = "discrete norm selector, mirrors upstream exact p compares"
)]
pub(crate) fn cdist_supported_f32(p: f64) -> bool {
    p != 0.0
}

/// True when the f64 GPU `cdist` kernel covers the norm exponent `p`.
///
/// The on-device f64 kernel ([`GpuBackend::cdist_f64`]) covers `p == 1`,
/// `p == 2`, and `p == inf`; the `p == 0` count-norm and general finite `p`
/// (which would need an accurate f64 `pow` the base PTX ISA does not provide)
/// fall back to the CPU path.
#[must_use]
// reason: see `cdist_supported_f32` — `p` is a discrete norm selector and the
// exact compare mirrors the upstream norm dispatch.
#[allow(
    clippy::float_cmp,
    reason = "discrete norm selector, mirrors upstream exact p compares"
)]
pub(crate) fn cdist_supported_f64(p: f64) -> bool {
    p == 1.0 || p == 2.0 || p.is_infinite()
}

/// Trait that GPU backends implement to handle tensor operations.
///
/// ferrotorch-core calls these methods; ferrotorch-gpu provides the implementation.
pub trait GpuBackend: Send + Sync {
    /// Downcast to `&dyn Any` for backend-specific access (e.g., getting the
    /// underlying `GpuDevice` for CUDA graph capture).
    fn as_any(&self) -> &dyn std::any::Any;
    /// Copy CPU bytes to GPU, tagging the resulting handle with `dtype`.
    ///
    /// `dtype` is the PyTorch `ScalarType` analog: it is the authoritative
    /// element-type tag for `data` and decides the concrete on-device buffer
    /// type. The element size is derived internally via `dtype.size_of()`.
    fn cpu_to_gpu(
        &self,
        data: &[u8],
        dtype: DType,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn gpu_to_cpu(&self, handle: &GpuBufferHandle) -> FerrotorchResult<Vec<u8>>;

    /// Get the raw CUDA device pointer from a buffer handle.
    ///
    /// Returns null if the handle type is not recognized or the backend
    /// doesn't support raw pointer access.
    fn raw_device_ptr(&self, _handle: &GpuBufferHandle) -> *const std::ffi::c_void {
        std::ptr::null()
    }

    /// Get a mutable raw CUDA device pointer from a buffer handle.
    fn raw_device_ptr_mut(&self, _handle: &mut GpuBufferHandle) -> *mut std::ffi::c_void {
        std::ptr::null_mut()
    }

    /// Get the element size (in bytes) of the data stored in a buffer handle.
    ///
    /// Derived from the handle's authoritative [`GpuBufferHandle::dtype`] tag
    /// (PyTorch parity: byte width is a function of the `ScalarType`, never the
    /// other way around). Returns 0 if unknown.
    fn buffer_elem_size(&self, handle: &GpuBufferHandle) -> usize {
        handle.dtype().size_of()
    }

    /// Copy CPU data to GPU via pinned (page-locked) host memory.
    ///
    /// ~2x faster than [`cpu_to_gpu`] for large buffers due to DMA transfers.
    /// Falls back to regular `cpu_to_gpu` by default. `dtype` is the
    /// authoritative element-type tag for `data` (see [`Self::cpu_to_gpu`]).
    fn cpu_to_gpu_pinned(
        &self,
        data: &[u8],
        dtype: DType,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        self.cpu_to_gpu(data, dtype, device)
    }
    fn clone_buffer(&self, handle: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
    /// Allocate a zero-initialised device buffer of `len` elements, tagged
    /// with `dtype`. Element size is derived via `dtype.size_of()`.
    fn alloc_zeros(
        &self,
        len: usize,
        dtype: DType,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;

    // Elementwise f32
    fn add_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn sub_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn mul_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn neg_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
    fn relu_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;

    // Linalg f32
    fn matmul_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;

    /// Mixed-precision matmul: cast f32 inputs to f16, multiply, accumulate
    /// back to f32. Used by autocast when the category is `ReducedPrecision`.
    ///
    /// Default implementation falls back to `matmul_f32` (no precision
    /// reduction) until a real f16 GEMM kernel is available.
    ///
    /// # NaN / Inf propagation
    ///
    /// f16 has a much smaller dynamic range than f32 (max ~65504). Values
    /// outside that range will overflow to inf or underflow to zero when cast.
    /// Callers relying on autocast should ensure their model weights stay
    /// within f16-representable bounds (which is normal for trained networks).
    fn matmul_f16_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        // Fallback: no f16 kernel available, use full-precision f32.
        self.matmul_f32(a, b, m, k, n)
    }

    // Reduction f32
    fn sum_f32(&self, a: &GpuBufferHandle, len: usize) -> FerrotorchResult<GpuBufferHandle>;

    /// f32 product reduction. Returns a 1-element buffer holding the
    /// product of all elements. (#524)
    fn prod_f32(&self, _a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "GPU reduce_prod not implemented for this backend".into(),
        })
    }

    /// f32 parallel min reduction. Returns a 1-element buffer holding the
    /// minimum element of `a`. Default impl returns the
    /// "not yet implemented" error so existing backends compile unchanged
    /// — concrete backends override. (#627)
    fn min_f32(&self, _a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "GPU reduce_min not implemented for this backend".into(),
        })
    }

    /// f32 parallel max reduction. Counterpart of [`Self::min_f32`]. (#627)
    fn max_f32(&self, _a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "GPU reduce_max not implemented for this backend".into(),
        })
    }

    /// f32 fused masked-min reduction (#627). Single-pass kernel that
    /// folds `(data, mask_f) -> min` directly, where `mask_f[i]` is 1.0
    /// for valid entries and 0.0 for masked. Avoids the
    /// `mul + add + reduce` chain that the unfused path requires.
    fn masked_min_f32(
        &self,
        _data: &GpuBufferHandle,
        _mask_f: &GpuBufferHandle,
        _len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "GPU masked_reduce_min not implemented for this backend".into(),
        })
    }

    /// f32 fused masked-max counterpart of [`Self::masked_min_f32`]. (#627)
    fn masked_max_f32(
        &self,
        _data: &GpuBufferHandle,
        _mask_f: &GpuBufferHandle,
        _len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "GPU masked_reduce_max not implemented for this backend".into(),
        })
    }

    // Elementwise f64 (default impls return "not yet implemented" errors)
    fn add_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU ops not yet implemented".into(),
        })
    }
    fn sub_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU ops not yet implemented".into(),
        })
    }
    fn mul_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU ops not yet implemented".into(),
        })
    }
    fn neg_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU ops not yet implemented".into(),
        })
    }
    fn relu_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU ops not yet implemented".into(),
        })
    }

    // Linalg f64
    fn matmul_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU ops not yet implemented".into(),
        })
    }

    // -- Vector matmul kernels (#816 / #817 / #818) ---------------------------
    //
    // These cover the rank combinations that PyTorch's `torch.matmul` (and
    // `torch.dot` / `torch.mv`) accepts on CUDA but ferrotorch previously
    // routed through CPU-only specialised paths, surfacing as
    // `GpuTensorNotAccessible` for CUDA inputs.
    //
    // - `dot_*` : 1D x 1D inner product (cuBLAS `{S,D}dot`)
    // - `mv_*`  : 2D x 1D matrix-vector product (cuBLAS `{S,D}gemv`, OP_T)
    // - `vm_*`  : 1D x 2D vector-matrix product (cuBLAS `{S,D}gemv`, OP_N)
    //
    // CUDA is the primary GPU backend; backends that don't implement these
    // (or aren't CUDA at all) inherit the default `Err` impl. CUDA itself
    // overrides them with real cuBLAS kernels.

    /// 1D x 1D dot product on GPU. Returns a 1-element buffer.
    /// `n` is the shared length of both inputs (each buffer holds `n` elements).
    fn dot_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "dot_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// 1D x 1D dot product on GPU (f64 dtype).
    fn dot_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "dot_f64 GPU op not implemented for this backend".into(),
        })
    }

    /// 2D x 1D matrix-vector product `y[m] = A[m,k] @ x[k]`. Returns a
    /// buffer of length `m`.
    fn mv_f32(
        &self,
        _a: &GpuBufferHandle,
        _x: &GpuBufferHandle,
        _m: usize,
        _k: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "mv_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// 2D x 1D matrix-vector product on GPU (f64 dtype).
    fn mv_f64(
        &self,
        _a: &GpuBufferHandle,
        _x: &GpuBufferHandle,
        _m: usize,
        _k: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "mv_f64 GPU op not implemented for this backend".into(),
        })
    }

    /// 1D x 2D vector-matrix product `y[n] = x[k] @ B[k,n]`. Returns a
    /// buffer of length `n`. Implemented via `gemv` with the transpose flag
    /// — does NOT materialise a transposed copy of `B`.
    fn vm_f32(
        &self,
        _x: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "vm_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// 1D x 2D vector-matrix product on GPU (f64 dtype).
    fn vm_f64(
        &self,
        _x: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "vm_f64 GPU op not implemented for this backend".into(),
        })
    }

    // -- Broadcast / 4D matmul kernel (#819) ----------------------------------
    //
    // PyTorch's `torch.matmul` supports arbitrary leading-dim broadcast on
    // CUDA — `(B, M, K) @ (K, N)`, `(M, K) @ (B, K, N)`, full 4D bmm,
    // `(2, 1, M, K) @ (2, 4, K, N)`, etc. Pre-fix these shapes fell through
    // to `linalg::matmul` (CPU path) and surfaced as `GpuTensorNotAccessible`
    // for CUDA inputs. Post-fix, `matmul_differentiable` routes them to
    // `broadcast_bmm_f{32,64}` which lower to `cublas{S,D}gemmStridedBatched`
    // — stride=0 on broadcasted axes, no `expand` materialisation.
    //
    // Inputs:
    // - `a`/`b`: GPU buffers, contiguous, in row-major batch layout. The
    //   caller has already ensured non-broadcasted dims match and that
    //   `a.len() == a_batch_count * m * k`, `b.len() == b_batch_count * k * n`.
    // - `out_lead`: the broadcasted leading-dim shape (excluding `m, n`).
    //   `batch = product(out_lead)`. Output shape is `out_lead + [m, n]`.
    // - `a_lead`/`b_lead`: per-leading-axis sizes for A and B. Where the
    //   axis size is 1 vs `out_lead[i]`, that axis is treated as broadcast.
    //   Lengths may be shorter than `out_lead` (implicit batch=1 prefix).
    // - `m`, `k`, `n`: per-batch matmul dims.

    /// Broadcast / batched matmul on GPU (f32 dtype).
    fn broadcast_bmm_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_lead: &[usize],
        _b_lead: &[usize],
        _out_lead: &[usize],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "broadcast_bmm_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// Broadcast / batched matmul on GPU (f64 dtype).
    fn broadcast_bmm_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_lead: &[usize],
        _b_lead: &[usize],
        _out_lead: &[usize],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "broadcast_bmm_f64 GPU op not implemented for this backend".into(),
        })
    }

    /// Broadcast / batched matmul on GPU (bf16 dtype, bf16 in/out, f32 accum).
    ///
    /// Same calling convention as [`Self::broadcast_bmm_f32`]: handles the 4D
    /// bmm, 3D × 2D, 2D × 3D, and arbitrary leading-dim broadcasts that
    /// `matmul_differentiable` routes here for `Tensor<bf16>` on CUDA.
    /// Pre-fix (GH forecast-bio/ferrotorch#25 / local #1543) bf16 fell through
    /// to the CPU `broadcast_matmul` round-trip; downstream of that path the
    /// ViT 3D × 2D `(1, 200, 4096) @ (4096, 768)` matmul reported a 50× worse
    /// `max|Δ|` than CPU bf16, because the GPU→CPU→GPU code-path is what the
    /// reporter actually measured (the CPU bf16 path uses an f64 accumulator;
    /// the device round-trip silently changes which kernel runs). Routing
    /// bf16 directly through `gpu_matmul_bf16_bf16_strided_batched`
    /// (`CUDA_R_16BF` in/out, `CUBLAS_COMPUTE_32F` accumulator) restores the
    /// standard ~1.5e-3 cuBLAS bf16+f32-accum floor that the upstream issue
    /// expects.
    fn broadcast_bmm_bf16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_lead: &[usize],
        _b_lead: &[usize],
        _out_lead: &[usize],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "broadcast_bmm_bf16 GPU op not implemented for this backend".into(),
        })
    }

    /// Broadcast / batched matmul on GPU (IEEE f16 dtype, f16 in/out, f32 accum).
    ///
    /// Symmetric trait surface to [`Self::broadcast_bmm_bf16`] for `Tensor<f16>`
    /// on CUDA. The default impl returns `InvalidArgument`; the CUDA backend
    /// has no `gpu_matmul_f16_f16_strided_batched` kernel today, so f16 GPU
    /// 3D × 2D matmul continues to fall back to the CPU `broadcast_matmul`
    /// round-trip until that kernel lands (see `.design/ferrotorch-gpu/blas.md`
    /// REQ-11 status row). This trait surface is preserved so the
    /// `matmul_differentiable` dispatcher in
    /// `ferrotorch-core/src/grad_fns/linalg.rs` can branch uniformly on
    /// `is_bf16` / `is_f16` and an opt-in backend can override later without
    /// further dispatcher churn.
    fn broadcast_bmm_f16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_lead: &[usize],
        _b_lead: &[usize],
        _out_lead: &[usize],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "broadcast_bmm_f16 GPU op not implemented for this backend".into(),
        })
    }

    // Reduction f64
    fn sum_f64(&self, _a: &GpuBufferHandle, _numel: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU ops not yet implemented".into(),
        })
    }

    /// f64 product reduction. (#524)
    fn prod_f64(&self, _a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU reduce_prod not implemented for this backend".into(),
        })
    }

    /// f32 backward of the global `prod` reduction (#785).
    ///
    /// Returns `grad_input[i] = grad_output * (prod_{j != i} input[j])`,
    /// which matches PyTorch's exact zero-handling semantics:
    /// no zeros → `grad_input = grad_output * total / input`; one zero
    /// at index z → only `grad_input[z]` is nonzero (the product of the
    /// remaining elements); two or more zeros → all zero.
    ///
    /// `grad_output` is a scalar (`numel == 1`).
    fn prod_backward_f32(
        &self,
        _input: &GpuBufferHandle,
        _grad_output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "prod_backward_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 backward of the global `prod` reduction (#785). Companion of
    /// [`Self::prod_backward_f32`].
    fn prod_backward_f64(
        &self,
        _input: &GpuBufferHandle,
        _grad_output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "prod_backward_f64 GPU op not yet implemented".into(),
        })
    }

    /// f64 parallel min reduction. (#627)
    fn min_f64(&self, _a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU reduce_min not implemented for this backend".into(),
        })
    }

    /// f64 parallel max reduction. (#627)
    fn max_f64(&self, _a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU reduce_max not implemented for this backend".into(),
        })
    }

    /// f64 fused masked-min reduction (#627).
    fn masked_min_f64(
        &self,
        _data: &GpuBufferHandle,
        _mask_f: &GpuBufferHandle,
        _len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU masked_reduce_min not implemented for this backend".into(),
        })
    }

    /// f64 fused masked-max reduction (#627).
    fn masked_max_f64(
        &self,
        _data: &GpuBufferHandle,
        _mask_f: &GpuBufferHandle,
        _len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU masked_reduce_max not implemented for this backend".into(),
        })
    }

    // Broadcast binary f32
    fn broadcast_add_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn broadcast_add_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "broadcast_add_f64 GPU op not yet implemented".into(),
        })
    }
    fn broadcast_sub_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn broadcast_sub_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "broadcast_sub_f64 GPU op not yet implemented".into(),
        })
    }
    fn broadcast_mul_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn broadcast_mul_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "broadcast_mul_f64 GPU op not yet implemented".into(),
        })
    }
    fn broadcast_div_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn broadcast_div_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "broadcast_div_f64 GPU op not yet implemented".into(),
        })
    }

    // Softmax f32 (row-wise over last dim)
    fn softmax_f32(
        &self,
        a: &GpuBufferHandle,
        rows: usize,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn softmax_f64(
        &self,
        _a: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "softmax_f64 GPU op not yet implemented".into(),
        })
    }

    // Dropout f32 (inverted dropout)
    fn dropout_f32(
        &self,
        a: &GpuBufferHandle,
        threshold: u32,
        scale: f32,
        seed: u32,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn dropout_f64(
        &self,
        _a: &GpuBufferHandle,
        _threshold: u32,
        _scale: f64,
        _seed: u32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "dropout_f64 GPU op not yet implemented".into(),
        })
    }

    /// Dropout using the Philox CBRNG for deterministic, reproducible mask generation.
    ///
    /// Instead of a simple u32 seed, this takes a `GpuRngState` that specifies the
    /// exact Philox counter and key to use. This enables gradient checkpointing to
    /// reproduce identical dropout masks by restoring the RNG state.
    ///
    /// The method also advances the global GPU RNG state by `ceil(n/4)` counters.
    ///
    /// Returns the dropped-out buffer and the Philox state that was used (for
    /// backward mask regeneration).
    fn dropout_philox_f32(
        &self,
        a: &GpuBufferHandle,
        threshold: u32,
        scale: f32,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuRngState)> {
        // Default: fall back to the non-Philox version with a dummy seed.
        // The returned state has device=0 as a placeholder.
        let result = self.dropout_f32(a, threshold, scale, 0)?;
        Ok((
            result,
            GpuRngState {
                counter: 0,
                seed: 0,
                offset: 0,
                device: 0,
            },
        ))
    }
    fn dropout_philox_f64(
        &self,
        _a: &GpuBufferHandle,
        _threshold: u32,
        _scale: f64,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuRngState)> {
        Err(FerrotorchError::InvalidArgument {
            message: "dropout_philox_f64 GPU op not yet implemented".into(),
        })
    }

    // 2D transpose f32
    fn transpose_2d_f32(
        &self,
        a: &GpuBufferHandle,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn transpose_2d_f64(
        &self,
        _a: &GpuBufferHandle,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "transpose_2d_f64 GPU op not yet implemented".into(),
        })
    }

    // 4D permute (0,2,1,3) f32 — swap dims 1 and 2
    fn permute_0213_f32(
        &self,
        a: &GpuBufferHandle,
        d0: usize,
        d1: usize,
        d2: usize,
        d3: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn permute_0213_f64(
        &self,
        _a: &GpuBufferHandle,
        _d0: usize,
        _d1: usize,
        _d2: usize,
        _d3: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "permute_0213_f64 GPU op not yet implemented".into(),
        })
    }

    // Batched matmul f32: C[i] = A[i] @ B[i] for i in 0..batch
    fn bmm_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn bmm_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _batch: usize,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "bmm_f64 GPU op not yet implemented".into(),
        })
    }

    /// Batched matmul with f16 Tensor Core acceleration.
    /// Takes f32 handles, converts to f16 internally, accumulates in f32.
    fn bmm_f16_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _batch: usize,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "bmm_f16_f32 GPU op not yet implemented".into(),
        })
    }

    // -- bf16 × f32-accumulator mixed-precision kernels (#518) ---------------
    //
    // PyTorch parity (rust-gpu-discipline §3): under `torch.autocast(device_type
    // ="cuda", dtype=torch.bfloat16)`, `torch.matmul`, `torch.bmm`, and
    // `torch.softmax` on CUDA tensors use bf16 inputs with f32 accumulation via
    // cuBLAS GemmEx (CUDA_R_16BF / CUBLAS_COMPUTE_32F).
    //
    // Default impls return `Err` so existing backends compile unchanged. The
    // CUDA backend overrides these three methods with real cuBLAS / PTX kernels.
    // There is NO silent CPU fallback (§3 hard requirement).

    /// Matrix multiply: bf16 inputs (f32-buffers converted to bf16 on-device)
    /// → f32 output via cuBLAS GemmEx (CUDA_R_16BF / CUBLAS_COMPUTE_32F).
    ///
    /// Signature mirrors [`Self::matmul_f16_f32`]; dtype is bf16 instead of f16.
    /// bf16 has a wider exponent range than f16 (same 8-bit exponent as f32),
    /// making it more robust to large weight values typical in transformers.
    fn matmul_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "matmul_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// Batched matrix multiply: bf16 inputs → f32 output.
    ///
    /// `a` is `[batch, m, k]`, `b` is `[batch, k, n]`, result is `[batch, m, n]`.
    /// All tensors are passed as f32 handles; inputs are converted to bf16
    /// on-device before the cuBLAS GemmStridedBatchedEx call.
    fn bmm_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _batch: usize,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "bmm_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// Matrix multiply: bf16 inputs → bf16 output via cuBLAS GemmEx
    /// (`CUDA_R_16BF` operands, `CUBLAS_COMPUTE_32F` accumulator).
    ///
    /// Both input handles must carry a `CudaSlice<u16>` (each u16 is a bf16
    /// bit pattern — top 16 bits of an f32). The result is also a
    /// `CudaSlice<u16>` of shape `[m, n]`. This is the foundational op for
    /// bf16-resident inference (weights + activations stay bf16 in VRAM,
    /// halving VRAM vs. f32) and is what unblocks `Tensor<bf16> @
    /// Tensor<bf16>` on CUDA without an upstream-side cast to f32.
    fn matmul_bf16_bf16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "matmul_bf16_bf16 GPU op not implemented for this backend".into(),
        })
    }

    /// Batched matrix multiply: bf16 inputs → bf16 output via cuBLAS
    /// GemmStridedBatchedEx.
    ///
    /// Each batch element is a row-major matmul of shapes `A:[M,K]`,
    /// `B:[K,N]`, producing `C:[M,N]`. Per-batch strides default to
    /// `m*k`, `k*n`, `m*n` (contiguous batches). Both input handles
    /// must carry `CudaSlice<u16>` (bf16 bit patterns); the output is
    /// a `CudaSlice<u16>` of total size `batch * m * n`.
    fn bmm_bf16_bf16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _batch: usize,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "bmm_bf16_bf16 GPU op not implemented for this backend".into(),
        })
    }

    /// Row-wise softmax: bf16 input (stored as `u16` bit-pattern buffer) →
    /// f32 output via PTX kernel with f32 accumulator.
    ///
    /// `rows` = product of all dims except the last; `cols` = last dim size.
    /// The input handle must contain a `CudaSlice<u16>` (bf16 bit patterns);
    /// the output is a `CudaBuffer<f32>` of the same shape.
    ///
    /// All phases (max-find, exp-sum, normalize) accumulate in f32 for
    /// numerical stability — matches PyTorch's bf16 softmax contract.
    fn softmax_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "softmax_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    // -- bf16 elementwise (#963) ---------------------------------------------

    /// Elementwise add: bf16 inputs (u16 bit-pattern buffers) -> f32 output.
    ///
    /// PyTorch parity: `torch.add(a.bfloat16(), b.bfloat16())` under autocast
    /// uses f32 accumulators on CUDA. Both handles must contain
    /// `CudaSlice<u16>` (bf16 bit patterns) of the same length `n`.
    fn add_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "add_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// Elementwise subtract: bf16 inputs -> f32 output.
    fn sub_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sub_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// Elementwise multiply: bf16 inputs -> f32 output.
    fn mul_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "mul_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// Elementwise divide: bf16 inputs -> f32 output.
    fn div_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "div_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    // -- bf16 reductions (#963) ----------------------------------------------

    /// Sum along an axis: bf16 input [outer, axis_size, inner] (u16) -> f32
    /// [outer, inner]. Accumulates in f32.
    fn sum_axis_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _axis_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sum_axis_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// Mean along an axis: bf16 input [outer, axis_size, inner] (u16) -> f32
    /// [outer, inner]. Accumulates in f32, divides by axis_size in f32.
    fn mean_axis_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _axis_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "mean_axis_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    // -- bf16 activations (#963) ---------------------------------------------

    /// ReLU activation: bf16 input (u16) -> f32 output.
    /// `out[i] = max(0.0, bf16_to_f32(a[i]))`.
    fn relu_bf16_f32(&self, _a: &GpuBufferHandle, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "relu_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// Sigmoid activation: bf16 input (u16) -> f32 output.
    /// `out[i] = 1 / (1 + exp(-bf16_to_f32(a[i])))`.
    fn sigmoid_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sigmoid_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    // GELU activation f32 (sigmoid approximation)
    fn gelu_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
    fn gelu_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "gelu_f64 GPU op not yet implemented".into(),
        })
    }
    // GELU activation f32 (tanh approximation: PyTorch approximate="tanh")
    fn gelu_tanh_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "gelu_tanh_f32 GPU op not yet implemented".into(),
        })
    }
    fn gelu_tanh_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "gelu_tanh_f64 GPU op not yet implemented".into(),
        })
    }
    // GELU activation f32 (exact erf: PyTorch approximate="none")
    fn gelu_erf_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "gelu_erf_f32 GPU op not yet implemented".into(),
        })
    }
    fn gelu_erf_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "gelu_erf_f64 GPU op not yet implemented".into(),
        })
    }

    // LayerNorm f32 (row-wise, with affine)
    fn layernorm_f32(
        &self,
        input: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        bias: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn layernorm_f64(
        &self,
        _input: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _bias: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _eps: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "layernorm_f64 GPU op not yet implemented".into(),
        })
    }

    // GroupNorm f32 (#1356 / #1357).
    //
    // Channel-group normalization over a `[batch, channels, hw]`-laid-out f32
    // buffer (PyTorch `[N, C, H, W]` flattened to `hw = H*W`). `channels` is
    // split into `groups` groups; per-`(batch, group)` mean/var are taken over
    // `(channels/groups) * hw` elements, then the per-channel affine
    // `weight[c] * normed + bias[c]` is applied. `weight`/`bias` have length
    // `channels`. Mirrors `aten/src/ATen/native/cuda/group_norm_kernel.cu`
    // (`GroupNormKernelImpl`). The default impl returns `InvalidArgument` so
    // existing backends compile unchanged; the CUDA backend overrides it with
    // the `gpu_group_norm_f32` PTX kernel. Non-test production consumer:
    // `ferrotorch-nn::GroupNorm::forward` GPU fast path.
    fn group_norm_f32(
        &self,
        _input: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _bias: &GpuBufferHandle,
        _batch: usize,
        _channels: usize,
        _groups: usize,
        _hw: usize,
        _eps: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "group_norm_f32 GPU op not implemented for this backend".into(),
        })
    }

    // BatchNorm f32 (#1449).
    //
    // Per-channel normalization over a `[batch, channels, hw]`-laid-out f32
    // buffer (PyTorch `[N, C, *spatial]` flattened to `hw = ∏ spatial`). In
    // **training** mode (`training == true`) the per-channel mean / variance
    // are computed over `(batch, hw)` (biased, like PyTorch) and the computed
    // batch stats are returned as the second / third tuple element so the
    // caller can update its running statistics; in **eval** mode the
    // caller-supplied `mean` / `var` (the running stats) are used and returned
    // unchanged. `weight`/`bias` have length `channels` (ones / zeros when the
    // layer is non-affine). Mirrors `aten/src/ATen/native/Normalization.cpp`
    // `batch_norm_cpu_transform_input_template`. The default impl returns
    // `InvalidArgument` so existing backends compile unchanged; the CUDA
    // backend overrides it with the `gpu_batch_norm_f32` PTX kernel. Non-test
    // production consumer: `ferrotorch-nn::BatchNorm{1,2,3}d::forward` GPU fast
    // path.
    //
    // Returns `(output, mean_out, var_out)` where `output` has the input shape
    // and `mean_out` / `var_out` have length `channels`.
    #[allow(clippy::too_many_arguments)]
    fn batch_norm_f32(
        &self,
        _input: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _bias: &GpuBufferHandle,
        _mean: &GpuBufferHandle,
        _var: &GpuBufferHandle,
        _batch: usize,
        _channels: usize,
        _hw: usize,
        _eps: f32,
        _training: bool,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "batch_norm_f32 GPU op not implemented for this backend".into(),
        })
    }

    // BatchNorm backward f32 (#1449).
    //
    // On-device gradient for the BatchNorm family over a
    // `[batch, channels, hw]`-laid-out f32 buffer. Mirrors
    // `aten/src/ATen/native/cuda/Normalization.cuh:388 batch_norm_backward_kernel`:
    // one reduction per channel computes `grad_output_sum = Σ go` and
    // `dot_p = Σ (x - mean) * go`, then
    //   grad_input  = train ? (go - (x-mean)*proj_scale - grad_mean) * grad_scale
    //                       : go * grad_scale
    //   grad_weight = dot_p * invstd ; grad_bias = grad_output_sum
    // where `proj_scale = dot_p/N * invstd²`, `grad_mean = grad_output_sum/N`,
    // `grad_scale = invstd * weight[c]`. In **training** mode `mean`/`invstd` are
    // recomputed from `input` (biased var, +eps); in **eval** mode they come from
    // `running_mean`/`running_var`. `weight` has length `channels` (all-ones for
    // the non-affine case so `grad_scale = invstd`). The default impl returns
    // `InvalidArgument`; the CUDA backend overrides it with
    // `gpu_batch_norm_backward_f32`. Non-test production consumer:
    // `ferrotorch-nn::BatchNorm{1,2,3}dBackward::backward` / `InstanceNormBackward`
    // GPU fast path.
    //
    // Returns `(grad_input, grad_weight, grad_bias)`: `grad_input` has the input
    // shape, `grad_weight` / `grad_bias` have length `channels`.
    #[allow(clippy::too_many_arguments)]
    fn batch_norm_backward_f32(
        &self,
        _input: &GpuBufferHandle,
        _grad_output: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _running_mean: &GpuBufferHandle,
        _running_var: &GpuBufferHandle,
        _batch: usize,
        _channels: usize,
        _hw: usize,
        _eps: f32,
        _training: bool,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "batch_norm_backward_f32 GPU op not implemented for this backend".into(),
        })
    }

    // LocalResponseNorm forward f32 (#1449).
    //
    // Per-element cross-channel normalization over a
    // `[batch, channels, spatial]`-laid-out f32 buffer. Mirrors
    // `torch/nn/functional.py:3032-3046 local_response_norm` (square → windowed
    // channel sum → `* alpha + k` → `pow(beta)` → divide):
    //   denom[i] = (Σ_window x² / size) * alpha + k
    //   out[i]   = x[i] / denom[i]^beta
    // Returns `(output, denom)`; `denom` (input shape) is the saved buffer the
    // backward consumes. The default impl returns `InvalidArgument`; the CUDA
    // backend overrides it with `gpu_local_response_norm_f32`. Non-test
    // production consumer: `ferrotorch-nn::LocalResponseNorm::forward` GPU path.
    #[allow(clippy::too_many_arguments)]
    fn local_response_norm_f32(
        &self,
        _input: &GpuBufferHandle,
        _batch: usize,
        _channels: usize,
        _spatial: usize,
        _size: usize,
        _alpha: f32,
        _beta: f32,
        _k: f32,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "local_response_norm_f32 GPU op not implemented for this backend".into(),
        })
    }

    // LocalResponseNorm backward f32 (#1449).
    //
    // On-device VJP for `local_response_norm`, consuming the `denom` buffer
    // saved by `local_response_norm_f32`. One thread per element:
    //   term1 = denom[i]^(-beta) * go[i]
    //   cross = Σ_{c in window(i)} go[c] * x[c] * denom[c]^(-beta-1)
    //   grad_input[i] = term1 - 2*beta*alpha/size * x[i] * cross
    // The default impl returns `InvalidArgument`; the CUDA backend overrides it
    // with `gpu_local_response_norm_backward_f32`. Non-test production consumer:
    // `ferrotorch-nn::LocalResponseNormBackward::backward` GPU path.
    #[allow(clippy::too_many_arguments)]
    fn local_response_norm_backward_f32(
        &self,
        _input: &GpuBufferHandle,
        _grad_output: &GpuBufferHandle,
        _denom: &GpuBufferHandle,
        _batch: usize,
        _channels: usize,
        _spatial: usize,
        _size: usize,
        _alpha: f32,
        _beta: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "local_response_norm_backward_f32 GPU op not implemented for this backend"
                .into(),
        })
    }

    // Softmax2d f32 (#1451).
    //
    // Channel-axis softmax over a `[n, c, hw]`-laid-out f32 buffer
    // (PyTorch `[N, C, H, W]` flattened to `hw = H*W`). For each `(n, p)`
    // spatial position, softmax is taken over the `c` channel values that are
    // strided `hw` apart in the flat buffer:
    // `out[n,c,p] = exp(x[n,c,p] - max_c') / Σ_c' exp(x[n,c',p] - max_c')`.
    // Mirrors `torch.nn.Softmax2d` (`torch/nn/modules/activation.py`). The
    // default impl returns `InvalidArgument` so existing backends compile
    // unchanged; the CUDA backend overrides it with the `gpu_softmax2d_f32`
    // PTX kernel. Non-test production consumer:
    // `ferrotorch-nn::Softmax2d::forward` GPU fast path.
    fn softmax2d_f32(
        &self,
        _input: &GpuBufferHandle,
        _n: usize,
        _c: usize,
        _hw: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "softmax2d_f32 GPU op not implemented for this backend".into(),
        })
    }

    // RMSNorm f32 (row-wise, weight only — no bias, no mean centering)
    fn rmsnorm_f32(
        &self,
        _input: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _eps: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "rmsnorm_f32 GPU op not yet implemented".into(),
        })
    }
    fn rmsnorm_f64(
        &self,
        _input: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _eps: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "rmsnorm_f64 GPU op not yet implemented".into(),
        })
    }

    // RMSNorm backward f32: returns (grad_input, grad_weight)
    fn rmsnorm_backward_f32(
        &self,
        _input: &GpuBufferHandle,
        _grad_output: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _eps: f32,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "rmsnorm_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn rmsnorm_backward_f64(
        &self,
        _input: &GpuBufferHandle,
        _grad_output: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _eps: f64,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "rmsnorm_backward_f64 GPU op not yet implemented".into(),
        })
    }

    // Slice write: write [N, D] into row `pos` of [N, max_len, D] (in-place)
    fn slice_write_f32(
        &self,
        src: &GpuBufferHandle,
        dst: &mut GpuBufferHandle,
        n_batch: usize,
        d: usize,
        max_len: usize,
        pos: usize,
    ) -> FerrotorchResult<()>;
    fn slice_write_f64(
        &self,
        _src: &GpuBufferHandle,
        _dst: &mut GpuBufferHandle,
        _n_batch: usize,
        _d: usize,
        _max_len: usize,
        _pos: usize,
    ) -> FerrotorchResult<()> {
        Err(FerrotorchError::InvalidArgument {
            message: "slice_write_f64 GPU op not yet implemented".into(),
        })
    }

    // Slice read: read first `len` rows from [N, max_len, D] → [N, len, D]
    fn slice_read_f32(
        &self,
        src: &GpuBufferHandle,
        n_batch: usize,
        d: usize,
        len: usize,
        max_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn slice_read_f64(
        &self,
        _src: &GpuBufferHandle,
        _n_batch: usize,
        _d: usize,
        _len: usize,
        _max_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "slice_read_f64 GPU op not yet implemented".into(),
        })
    }

    // Embedding lookup: gather row `idx` from weight [V, D] → [D]
    fn embed_lookup_f32(
        &self,
        idx: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        d: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn embed_lookup_f64(
        &self,
        _idx: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "embed_lookup_f64 GPU op not yet implemented".into(),
        })
    }

    // Batch embedding lookup: gather N rows from weight [V, D] → [N, D]
    // `indices` contains N f32 values encoding integer row indices.
    fn embed_lookup_batch_f32(
        &self,
        indices: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        n: usize,
        d: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn embed_lookup_batch_f64(
        &self,
        _indices: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _n: usize,
        _d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "embed_lookup_batch_f64 GPU op not yet implemented".into(),
        })
    }

    // Scatter-add rows: grad_weight[indices[i], :] += grad_output[i, :] for embedding backward
    // `indices` contains N f32 values, grad_output is [N, D], output is [num_embeddings, D]
    fn scatter_add_rows_f32(
        &self,
        grad_output: &GpuBufferHandle,
        indices: &GpuBufferHandle,
        num_embeddings: usize,
        d: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn scatter_add_rows_f64(
        &self,
        _grad_output: &GpuBufferHandle,
        _indices: &GpuBufferHandle,
        _num_embeddings: usize,
        _d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "scatter_add_rows_f64 GPU op not yet implemented".into(),
        })
    }

    // Scalar multiply: out[i] = a[i] * scalar
    fn scale_f32(&self, a: &GpuBufferHandle, scalar: f32) -> FerrotorchResult<GpuBufferHandle>;
    fn scale_f64(&self, _a: &GpuBufferHandle, _scalar: f64) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "scale_f64 GPU op not yet implemented".into(),
        })
    }

    // Backward activation kernels
    // relu_backward: out[i] = (input[i] > 0) ? grad[i] : 0
    fn relu_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn relu_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "relu_backward_f64 GPU op not yet implemented".into(),
        })
    }
    // abs_backward: out[i] = grad[i] * sign(input[i])  (sign(0) = 0)
    fn abs_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "abs_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn abs_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "abs_backward_f64 GPU op not yet implemented".into(),
        })
    }
    // fill: allocate an n-element device buffer filled with `scalar`.
    // Used by sum/mean backward so the grad is built entirely on-device.
    fn fill_f32(
        &self,
        _n: usize,
        _scalar: f32,
        _ordinal: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fill_f32 GPU op not yet implemented".into(),
        })
    }
    fn fill_f64(
        &self,
        _n: usize,
        _scalar: f64,
        _ordinal: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fill_f64 GPU op not yet implemented".into(),
        })
    }
    // gelu_backward (sigmoid approx): out[i] = grad[i] * (sig + 1.702*x*sig*(1-sig))
    fn gelu_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn gelu_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "gelu_backward_f64 GPU op not yet implemented".into(),
        })
    }
    // gelu_backward (tanh approx)
    fn gelu_backward_tanh_f32(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "gelu_backward_tanh_f32 GPU op not yet implemented".into(),
        })
    }
    fn gelu_backward_tanh_f64(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "gelu_backward_tanh_f64 GPU op not yet implemented".into(),
        })
    }
    // gelu_backward (exact erf): out[i] = grad[i] * (Φ(x) + x·φ(x))
    // where Φ = normal CDF, φ = normal PDF
    fn gelu_backward_erf_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn gelu_backward_erf_f64(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "gelu_backward_erf_f64 GPU op not yet implemented".into(),
        })
    }

    // Cumulative scan operations along a dimension.
    // Parameters: (input, outer, dim_size, inner) factorize the tensor shape.
    fn cumsum_f32(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "cumsum_f32 GPU op not yet implemented".into(),
        })
    }
    fn cumsum_f64(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "cumsum_f64 GPU op not yet implemented".into(),
        })
    }
    fn cumprod_f32(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "cumprod_f32 GPU op not yet implemented".into(),
        })
    }
    fn cumprod_f64(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "cumprod_f64 GPU op not yet implemented".into(),
        })
    }
    // Returns (values, indices_as_f32)
    fn cummax_f32(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "cummax_f32 GPU op not yet implemented".into(),
        })
    }
    fn cummax_f64(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "cummax_f64 GPU op not yet implemented".into(),
        })
    }
    // Returns (values, indices_as_f32)
    fn cummin_f32(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "cummin_f32 GPU op not yet implemented".into(),
        })
    }
    fn cummin_f64(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "cummin_f64 GPU op not yet implemented".into(),
        })
    }
    fn logcumsumexp_f32(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "logcumsumexp_f32 GPU op not yet implemented".into(),
        })
    }
    fn logcumsumexp_f64(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "logcumsumexp_f64 GPU op not yet implemented".into(),
        })
    }

    // Roll (cyclic shift) along a single axis. The `(outer, dim_size, inner)`
    // factorization mirrors the cumulative ops above; `shift_norm` is the
    // already-normalized non-negative shift with `0 <= shift_norm < dim_size`.
    // Forward and backward both call this method (the backward simply
    // negates the original shift before normalizing).
    fn roll_f32(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
        _shift_norm: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "roll_f32 GPU op not yet implemented".into(),
        })
    }

    // -- Triangular masks: triu / tril (#1545 / sub #1535) -------------------
    //
    // `[batch.., rows, cols]` C-contiguous masks (`batch` = product of the
    // leading dims, `1` for plain 2-D). The mask is applied to EVERY trailing
    // `[rows, cols]` matrix, batching over the leading dims. Element
    // `(row, col)` is preserved when the predicate holds and zeroed otherwise:
    //   - triu keeps `col - row >= k`
    //   - tril keeps `col - row <= k`
    // matching `aten/src/ATen/native/cuda/TriangularOps.cu:100` (predicate) and
    // `:120` (`N_padded = multiply_integers(sizes[..last]) * last_dim_padded`,
    // i.e. batches over leading dims) and the ferrotorch CPU `triu`/`tril`.
    // `k` is the signed diagonal offset. The result stays GPU-resident (no host
    // round-trip). Default bodies return `NotImplementedOnCuda` so non-CUDA
    // backends compile unchanged; the CUDA backend overrides all four. Non-test
    // consumer: the `input.is_cuda()` branch of `triu`/`tril` in
    // `ferrotorch-core/src/ops/tensor_ops.rs`.

    /// Upper-triangular mask over an f32 `[batch.., rows, cols]` buffer.
    fn triu_f32(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _rows: usize,
        _cols: usize,
        _k: i64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "triu_f32" })
    }

    /// Lower-triangular mask over an f32 `[batch.., rows, cols]` buffer.
    fn tril_f32(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _rows: usize,
        _cols: usize,
        _k: i64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "tril_f32" })
    }

    /// Upper-triangular mask over an f64 `[batch.., rows, cols]` buffer.
    fn triu_f64(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _rows: usize,
        _cols: usize,
        _k: i64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "triu_f64" })
    }

    /// Lower-triangular mask over an f64 `[batch.., rows, cols]` buffer.
    fn tril_f64(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _rows: usize,
        _cols: usize,
        _k: i64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "tril_f64" })
    }

    // -- Diagonal: diag_embed / diag_extract (#1545 / sub #1535) -------------
    //
    // `torch.diag` is `diag_embed` (1-D -> 2-D scatter onto the k-th diagonal)
    // for a 1-D input and `diagonal_copy` (2-D -> 1-D gather of the k-th
    // diagonal) for a 2-D input, mirroring
    // `aten/src/ATen/native/TensorShape.cpp:4610`. Both are pure gather/scatter
    // (no arithmetic), so the GPU result is bit-for-bit identical to the
    // ferrotorch CPU `diag`. `k` is the signed diagonal offset. The result
    // stays GPU-resident (no host round-trip). Default bodies return
    // `NotImplementedOnCuda` so non-CUDA backends compile unchanged; the CUDA
    // backend overrides all four. Non-test consumer: the `input.is_cuda()`
    // branch of `diag`/`diagflat` in `ferrotorch-core/src/ops/tensor_ops.rs`.

    /// `diag` of a 1-D f32 buffer: scatter `n` elements onto the `k`-th
    /// diagonal of a `[size, size]` matrix (`size = n + |k|`). Returns the
    /// resident `size*size`-element output.
    fn diag_embed_f32(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
        _k: i64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "diag_embed_f32",
        })
    }

    /// `diag` of a 1-D f64 buffer. See [`Self::diag_embed_f32`].
    fn diag_embed_f64(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
        _k: i64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "diag_embed_f64",
        })
    }

    /// `diag` of a 2-D f32 `[rows, cols]` buffer: gather the `k`-th diagonal
    /// into a 1-D vector of `min(rows-start_r, cols-start_c)` elements.
    fn diag_extract_f32(
        &self,
        _a: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _k: i64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "diag_extract_f32",
        })
    }

    /// `diag` of a 2-D f64 `[rows, cols]` buffer. See [`Self::diag_extract_f32`].
    fn diag_extract_f64(
        &self,
        _a: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _k: i64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "diag_extract_f64",
        })
    }

    // -- Pairwise distance: cdist (#1545 / sub #1535) ------------------------
    //
    // `torch.cdist(x1, x2, p)` is the batched Lp pairwise distance matrix:
    // `x1` is `[b, p_dim, m]`, `x2` is `[b, r_dim, m]`, the result is
    // `[b, p_dim, r_dim]`, `out[b,i,j] = (sum_k |x1[b,i,k]-x2[b,j,k]|^p)^(1/p)`.
    // Mirrors `aten/src/ATen/native/cuda/DistanceKernel.cu:195`
    // (`cdist_kernel_cuda_impl`) and the per-norm `dists<scalar_t>::{p,one,
    // two,inf}` accumulate/finish at `:50-86`. The result stays GPU-resident.
    // The CUDA backend covers `p in {1, 2, inf}` and general `p` for f32; the
    // `p == 0` count-norm (and general-p f64) fall back to the CPU path. Non-
    // test consumer: the `is_cuda()` branch of `cdist` in
    // `ferrotorch-core/src/ops/tensor_ops.rs`.

    /// Batched f32 `cdist`. `x1`/`x2` are `[b, p_dim, m]` / `[b, r_dim, m]`
    /// flattened; result is `[b, p_dim, r_dim]` flattened. Returns the
    /// resident `b * p_dim * r_dim`-element output.
    #[allow(clippy::too_many_arguments)]
    fn cdist_f32(
        &self,
        _x1: &GpuBufferHandle,
        _x2: &GpuBufferHandle,
        _b: usize,
        _p_dim: usize,
        _r_dim: usize,
        _m: usize,
        _p: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "cdist_f32" })
    }

    /// Batched f64 `cdist`. The f64 GPU kernel covers `p in {1, 2, inf}`;
    /// see [`Self::cdist_f32`] and [`cdist_supported_f64`].
    #[allow(clippy::too_many_arguments)]
    fn cdist_f64(
        &self,
        _x1: &GpuBufferHandle,
        _x2: &GpuBufferHandle,
        _b: usize,
        _p_dim: usize,
        _r_dim: usize,
        _m: usize,
        _p: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "cdist_f64" })
    }

    // -- Orthogonal-polynomial special functions (#1545 / #1533) -------------
    //
    // Each evaluates the n-th degree basis polynomial pointwise on a CUDA
    // buffer via an on-device three-term recurrence (one thread per element,
    // no host round-trip). The math mirrors the ferrotorch CPU recurrences in
    // `ferrotorch_core::special` so the GPU result equals the CPU result
    // bit-for-relevant-tolerance. The upstream recurrence reference is
    // `aten/src/ATen/native/Math.h` `chebyshev_polynomial_t_forward` et al.
    //
    // The chebyshev method folds T/U/V/W and their shifted variants into one
    // entry via `(seed_a, seed_b, shift)`: `q1 = seed_a*xx + seed_b` with
    // `xx = shift ? 2x-1 : x` (T: 1,0; U: 2,0; V: 2,-1; W: 2,1). Defaults
    // return `InvalidArgument` so non-CUDA backends compile unchanged; the
    // CUDA backend overrides all ten.

    /// Chebyshev polynomial (T/U/V/W + shifted) forward, f32. See the module
    /// comment for the `(seed_a, seed_b, shift)` kind selector.
    fn chebyshev_poly_f32(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
        _seed_a: f32,
        _seed_b: f32,
        _shift: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "chebyshev_poly_f32 GPU op not implemented for this backend".into(),
        })
    }
    /// Chebyshev polynomial (T/U/V/W + shifted) forward, f64.
    fn chebyshev_poly_f64(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
        _seed_a: f64,
        _seed_b: f64,
        _shift: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "chebyshev_poly_f64 GPU op not implemented for this backend".into(),
        })
    }
    /// Hermite (physicist's) `H_n` forward, f32.
    fn hermite_h_poly_f32(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "hermite_h_poly_f32 GPU op not implemented for this backend".into(),
        })
    }
    /// Hermite (physicist's) `H_n` forward, f64.
    fn hermite_h_poly_f64(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "hermite_h_poly_f64 GPU op not implemented for this backend".into(),
        })
    }
    /// Hermite (probabilist's) `He_n` forward, f32.
    fn hermite_he_poly_f32(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "hermite_he_poly_f32 GPU op not implemented for this backend".into(),
        })
    }
    /// Hermite (probabilist's) `He_n` forward, f64.
    fn hermite_he_poly_f64(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "hermite_he_poly_f64 GPU op not implemented for this backend".into(),
        })
    }
    /// Laguerre `L_n` forward, f32.
    fn laguerre_poly_f32(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "laguerre_poly_f32 GPU op not implemented for this backend".into(),
        })
    }
    /// Laguerre `L_n` forward, f64.
    fn laguerre_poly_f64(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "laguerre_poly_f64 GPU op not implemented for this backend".into(),
        })
    }
    /// Legendre `P_n` forward, f32.
    fn legendre_poly_f32(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "legendre_poly_f32 GPU op not implemented for this backend".into(),
        })
    }
    /// Legendre `P_n` forward, f64.
    fn legendre_poly_f64(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "legendre_poly_f64 GPU op not implemented for this backend".into(),
        })
    }

    // -- Normal-distribution trio: entr / ndtr / ndtri (#1651, batch 1) ------
    //
    // Each method launches an on-device elementwise PTX kernel (one thread per
    // element, no host round-trip). The math mirrors the ferrotorch CPU scalar
    // evaluators (`entr_scalar`, `ndtr_scalar`, `ndtri_f64`) so the GPU result
    // equals the CPU result bit-for-relevant-tolerance. Upstream kernels:
    // `entr_string` / `ndtri_string` (`aten/src/ATen/native/cuda/Math.cuh:463-480,
    // 48-173`) and `calc_ndtr` (`aten/src/ATen/native/UnaryOps.cpp:715-718`).
    // Defaults return `InvalidArgument` so non-CUDA backends compile unchanged;
    // the CUDA backend overrides all six.

    /// Entropy `entr(x)` forward, f32.
    fn entr_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "entr_f32 GPU op not implemented for this backend".into(),
        })
    }
    /// Entropy `entr(x)` forward, f64.
    fn entr_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "entr_f64 GPU op not implemented for this backend".into(),
        })
    }
    /// Standard-normal CDF `ndtr(x)` forward, f32.
    fn ndtr_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "ndtr_f32 GPU op not implemented for this backend".into(),
        })
    }
    /// Standard-normal CDF `ndtr(x)` forward, f64.
    fn ndtr_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "ndtr_f64 GPU op not implemented for this backend".into(),
        })
    }
    /// Inverse standard-normal CDF `ndtri(p)` forward, f32.
    fn ndtri_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "ndtri_f32 GPU op not implemented for this backend".into(),
        })
    }
    /// Inverse standard-normal CDF `ndtri(p)` forward, f64.
    fn ndtri_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "ndtri_f64 GPU op not implemented for this backend".into(),
        })
    }

    // Clamp: out[i] = max(min_val, min(max_val, x[i]))
    fn clamp_f32(
        &self,
        _a: &GpuBufferHandle,
        _min_val: f32,
        _max_val: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "clamp_f32 GPU op not yet implemented".into(),
        })
    }
    fn clamp_f64(
        &self,
        _a: &GpuBufferHandle,
        _min_val: f64,
        _max_val: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "clamp_f64 GPU op not yet implemented".into(),
        })
    }

    /// VJP for `clamp(x, min, max)`: `out[i] = grad[i]` when `x[i]` is in
    /// `[min, max]`, else `0`. (#524)
    fn clamp_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
        _min_val: f32,
        _max_val: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "clamp_backward_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 counterpart. (#524)
    fn clamp_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
        _min_val: f64,
        _max_val: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "clamp_backward_f64 GPU op not yet implemented".into(),
        })
    }

    // SiLU activation: out[i] = x * sigmoid(x)
    fn silu_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "silu_f32 GPU op not yet implemented".into(),
        })
    }
    fn silu_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "silu_f64 GPU op not yet implemented".into(),
        })
    }
    fn silu_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "silu_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn silu_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "silu_backward_f64 GPU op not yet implemented".into(),
        })
    }

    // ELU activation: out[i] = x > 0 ? x : alpha*(exp(x)-1)
    fn elu_f32(&self, _a: &GpuBufferHandle, _alpha: f32) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "elu_f32 GPU op not yet implemented".into(),
        })
    }
    fn elu_f64(&self, _a: &GpuBufferHandle, _alpha: f64) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "elu_f64 GPU op not yet implemented".into(),
        })
    }
    fn elu_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
        _alpha: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "elu_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn elu_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
        _alpha: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "elu_backward_f64 GPU op not yet implemented".into(),
        })
    }

    // Mish activation: out[i] = x * tanh(softplus(x))
    fn mish_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "mish_f32 GPU op not yet implemented".into(),
        })
    }
    fn mish_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "mish_f64 GPU op not yet implemented".into(),
        })
    }
    fn mish_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "mish_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn mish_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "mish_backward_f64 GPU op not yet implemented".into(),
        })
    }

    // LogSoftmax: out[i] = x[i] - log(sum(exp(x))) (row-wise)
    fn log_softmax_f32(
        &self,
        _a: &GpuBufferHandle,
        _cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "log_softmax_f32 GPU op not yet implemented".into(),
        })
    }
    fn log_softmax_f64(
        &self,
        _a: &GpuBufferHandle,
        _cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "log_softmax_f64 GPU op not yet implemented".into(),
        })
    }
    // LogSoftmax backward: out[i] = grad[i] - softmax[i] * sum(grad) (row-wise)
    fn log_softmax_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _output: &GpuBufferHandle,
        _cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "log_softmax_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn log_softmax_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _output: &GpuBufferHandle,
        _cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "log_softmax_backward_f64 GPU op not yet implemented".into(),
        })
    }

    // Indexing operations
    // index_select_1d: out[i] = input[indices[i]]  (indices stored as f32)
    fn index_select_1d_f32(
        &self,
        input: &GpuBufferHandle,
        indices: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn index_select_1d_f64(
        &self,
        _input: &GpuBufferHandle,
        _indices: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "index_select_1d_f64 GPU op not yet implemented".into(),
        })
    }
    // scatter_add_1d: out = zeros(input_len); for i: out[indices[i]] += grad_output[i]  (atomic)
    fn scatter_add_1d_f32(
        &self,
        grad_output: &GpuBufferHandle,
        indices: &GpuBufferHandle,
        input_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn scatter_add_1d_f64(
        &self,
        _grad_output: &GpuBufferHandle,
        _indices: &GpuBufferHandle,
        _input_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "scatter_add_1d_f64 GPU op not yet implemented".into(),
        })
    }
    // index_select_dim: gather slices along an arbitrary axis (N-D).
    //
    // Forward layout (contract):
    //   `input`  has logical shape `[outer, in_dim_size, inner]` after
    //   collapsing the axes before/after `dim`. `indices` is an
    //   `out_dim_size`-long f32 buffer encoding integer offsets into
    //   the `in_dim_size` axis (caller-validated, non-negative,
    //   in-range). The kernel writes
    //     `output[o, i, k] = input[o, indices[i], k]`
    //   for `o in [0, outer), i in [0, out_dim_size), k in [0, inner)`.
    // The output buffer has length `outer * out_dim_size * inner`.
    //
    // This subsumes the 1-D `index_select_1d_*` ops for ndim>=2 with
    // arbitrary `dim`. The 1-D ops are kept for the
    // `IndexSelectBackward` (single-axis 1-D index) call site, which
    // predates this op.
    fn index_select_dim_f32(
        &self,
        input: &GpuBufferHandle,
        indices: &GpuBufferHandle,
        outer: usize,
        in_dim_size: usize,
        out_dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn index_select_dim_f64(
        &self,
        _input: &GpuBufferHandle,
        _indices: &GpuBufferHandle,
        _outer: usize,
        _in_dim_size: usize,
        _out_dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "index_select_dim_f64 GPU op not yet implemented".into(),
        })
    }
    // masked_fill: out[i] = mask[i] ? value : input[i]  (mask stored as f32, 1.0/0.0)
    fn masked_fill_f32(
        &self,
        input: &GpuBufferHandle,
        mask: &GpuBufferHandle,
        value: f32,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn masked_fill_f64(
        &self,
        _input: &GpuBufferHandle,
        _mask: &GpuBufferHandle,
        _value: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "masked_fill_f64 GPU op not yet implemented".into(),
        })
    }
    // masked_fill with a GPU-resident Bool (u8) mask, dispatched on input.dtype()
    // (crosslink #1185 Phase 3c). `out[i] = mask[i]!=0 ? value : input[i]`.
    // Covers f32/f64/bf16/f16 (+ i32/i64). The scalar `value` is passed as f64
    // and converted to the input dtype in the backend (for bf16/f16 it is
    // narrowed in-kernel). `mask` MUST be tagged `DType::Bool` and have the same
    // numel as `input`; the result keeps `input`'s dtype and stays GPU-resident.
    // Unlike `masked_fill_f32`, the mask is the resident bool buffer — no
    // float-mask upload, no host crossing.
    fn masked_fill_dt(
        &self,
        _input: &GpuBufferHandle,
        _mask: &GpuBufferHandle,
        _value: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "masked_fill_dt",
        })
    }

    // where_cond / torch.where with a GPU-resident Bool (u8) condition
    // (crosslink #1185 Phase 3c). `out[i] = cond[i]!=0 ? x[i] : y[i]`, dispatched
    // on x.dtype(). `cond` MUST be tagged `DType::Bool`; `x.dtype() == y.dtype()`
    // and all three buffers MUST have equal numel. The result keeps x's dtype and
    // stays GPU-resident. Covers f32/f64/bf16/f16 (+ i32/i64).
    fn where_cond(
        &self,
        _cond: &GpuBufferHandle,
        _x: &GpuBufferHandle,
        _y: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "where_cond" })
    }

    // masked_select stream compaction (crosslink #1185 Phase 3c). Returns
    // `(out, len)` where `out` is a 1-D GPU-resident buffer of x's dtype holding
    // the `len` elements of `input` where `mask` is true. `mask` MUST be tagged
    // `DType::Bool` with the same numel as `input`. `len` is the on-device true
    // count read once to the host to size the data-dependent output — that single
    // integer is the result SHAPE, not a data round-trip (PyTorch parity: a CUDA
    // sync sizes `torch.masked_select`'s output). Covers f32/f64/bf16/f16
    // (+ i32/i64).
    fn masked_select(
        &self,
        _input: &GpuBufferHandle,
        _mask: &GpuBufferHandle,
    ) -> FerrotorchResult<(GpuBufferHandle, usize)> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "masked_select",
        })
    }

    // masked_scatter — the resident VJP of `masked_select` (crosslink #1187
    // Phase 3d). Scatter the compacted `grad_compact` (length = #true) back into
    // a zeros buffer of `out_numel` elements at the flat C-order positions where
    // `mask` is true: `out[i] = mask[i]!=0 ? grad_compact[j++] : 0`. `mask` MUST
    // be tagged `DType::Bool` with `mask.len() == out_numel`; the result keeps
    // `grad_compact`'s dtype and stays GPU-resident. Covers f32/f64/bf16/f16
    // (+ i32/i64). This is the inverse of the Phase-3c compaction kernel.
    fn masked_scatter(
        &self,
        _grad_compact: &GpuBufferHandle,
        _mask: &GpuBufferHandle,
        _out_numel: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "masked_scatter",
        })
    }

    // masked_zero: out[i] = mask[i] ? 0.0 : grad[i]  (backward of masked_fill)
    fn masked_zero_f32(
        &self,
        grad: &GpuBufferHandle,
        mask: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn masked_zero_f64(
        &self,
        _grad: &GpuBufferHandle,
        _mask: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "masked_zero_f64 GPU op not yet implemented".into(),
        })
    }

    // Elementwise unary/binary f32 (default impls for forward ops)
    fn div_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "div_f32 GPU op not yet implemented".into(),
        })
    }
    fn div_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "div_f64 GPU op not yet implemented".into(),
        })
    }
    fn exp_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "exp_f32 GPU op not yet implemented".into(),
        })
    }
    fn exp_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "exp_f64 GPU op not yet implemented".into(),
        })
    }
    fn log_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "log_f32 GPU op not yet implemented".into(),
        })
    }
    fn log_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "log_f64 GPU op not yet implemented".into(),
        })
    }
    fn sqrt_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sqrt_f32 GPU op not yet implemented".into(),
        })
    }
    fn sqrt_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sqrt_f64 GPU op not yet implemented".into(),
        })
    }
    fn pow_f32(&self, _a: &GpuBufferHandle, _exponent: f32) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "pow_f32 GPU op not yet implemented".into(),
        })
    }
    fn pow_f64(&self, _a: &GpuBufferHandle, _exponent: f64) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "pow_f64 GPU op not yet implemented".into(),
        })
    }
    fn abs_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "abs_f32 GPU op not yet implemented".into(),
        })
    }
    fn abs_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "abs_f64 GPU op not yet implemented".into(),
        })
    }
    fn sigmoid_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sigmoid_f32 GPU op not yet implemented".into(),
        })
    }
    fn sigmoid_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sigmoid_f64 GPU op not yet implemented".into(),
        })
    }
    fn tanh_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "tanh_f32 GPU op not yet implemented".into(),
        })
    }
    fn tanh_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "tanh_f64 GPU op not yet implemented".into(),
        })
    }

    // Sigmoid backward: out[i] = grad[i] * output[i] * (1 - output[i])
    fn sigmoid_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sigmoid_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn sigmoid_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sigmoid_backward_f64 GPU op not yet implemented".into(),
        })
    }

    // Tanh backward: out[i] = grad[i] * (1 - output[i]^2)
    fn tanh_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "tanh_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn tanh_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "tanh_backward_f64 GPU op not yet implemented".into(),
        })
    }

    // Softmax backward: out[i] = output[i] * (grad[i] - dot(grad_row, output_row))
    fn softmax_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _output: &GpuBufferHandle,
        _cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "softmax_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn softmax_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _output: &GpuBufferHandle,
        _cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "softmax_backward_f64 GPU op not yet implemented".into(),
        })
    }

    // LayerNorm backward: computes grad_input, grad_weight, grad_bias on GPU
    fn layernorm_backward_f32(
        &self,
        _input: &GpuBufferHandle,
        _grad_output: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _eps: f32,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "layernorm_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn layernorm_backward_f64(
        &self,
        _input: &GpuBufferHandle,
        _grad_output: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _eps: f64,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "layernorm_backward_f64 GPU op not yet implemented".into(),
        })
    }

    // Sum along one axis of a tensor
    fn sum_axis_f32(
        &self,
        _a: &GpuBufferHandle,
        _shape: &[usize],
        _axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sum_axis_f32 GPU op not yet implemented".into(),
        })
    }
    fn sum_axis_f64(
        &self,
        _a: &GpuBufferHandle,
        _shape: &[usize],
        _axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sum_axis_f64 GPU op not yet implemented".into(),
        })
    }

    // Strided split: extract a sub-tensor along one axis entirely on GPU.
    fn strided_split_f32(
        &self,
        _input: &GpuBufferHandle,
        _total_along_axis: usize,
        _split_offset: usize,
        _split_size: usize,
        _inner_size: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "strided_split_f32 GPU op not yet implemented".into(),
        })
    }
    fn strided_split_f64(
        &self,
        _input: &GpuBufferHandle,
        _total_along_axis: usize,
        _split_offset: usize,
        _split_size: usize,
        _inner_size: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "strided_split_f64 GPU op not yet implemented".into(),
        })
    }

    // Strided copy: gather an N-d strided view into a contiguous
    // output buffer entirely on GPU. CL-496.
    fn strided_copy_f32(
        &self,
        _input: &GpuBufferHandle,
        _out_shape: &[usize],
        _src_strides: &[isize],
        _src_offset: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "strided_copy_f32 GPU op not yet implemented".into(),
        })
    }
    fn strided_copy_f64(
        &self,
        _input: &GpuBufferHandle,
        _out_shape: &[usize],
        _src_strides: &[isize],
        _src_offset: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "strided_copy_f64 GPU op not yet implemented".into(),
        })
    }

    // Strided scatter: write a contiguous src into strided positions of
    // dst (in-place). Inverse of strided_copy. Used by
    // `Tensor::as_strided_scatter` for CUDA tensors. (#574)
    fn strided_scatter_f32(
        &self,
        _src: &GpuBufferHandle,
        _dst: &mut GpuBufferHandle,
        _view_shape: &[usize],
        _dst_strides: &[isize],
        _dst_offset: usize,
    ) -> FerrotorchResult<()> {
        Err(FerrotorchError::InvalidArgument {
            message: "strided_scatter_f32 GPU op not yet implemented".into(),
        })
    }
    fn strided_scatter_f64(
        &self,
        _src: &GpuBufferHandle,
        _dst: &mut GpuBufferHandle,
        _view_shape: &[usize],
        _dst_strides: &[isize],
        _dst_offset: usize,
    ) -> FerrotorchResult<()> {
        Err(FerrotorchError::InvalidArgument {
            message: "strided_scatter_f64 GPU op not yet implemented".into(),
        })
    }

    // Strided cat: write a sub-tensor into a larger buffer at an offset along
    // one axis on GPU. Dtype-generic via `elem_size`: PyTorch's
    // `aten::cat_out_cuda` (`aten/src/ATen/native/cuda/Shape.cu`) does the same
    // — the host computes the scalar size once, then dispatches into a
    // strided-memcpy kernel whose body only depends on element width (no
    // arithmetic). Concrete backends are expected to support at least
    // `elem_size in {2, 4, 8}` (covers `bf16`/`f16`, `f32`, `f64`); other
    // widths must return an error so the caller can fall back rather than
    // silently produce wrong data.
    #[allow(clippy::too_many_arguments)]
    fn strided_cat(
        &self,
        _src: &GpuBufferHandle,
        _dst: &mut GpuBufferHandle,
        _total_along_axis: usize,
        _offset: usize,
        _t_axis_size: usize,
        _inner: usize,
        _t_numel: usize,
        elem_size: usize,
    ) -> FerrotorchResult<()> {
        Err(FerrotorchError::InvalidArgument {
            message: format!(
                "strided_cat (elem_size={elem_size}) GPU op not implemented for this backend"
            ),
        })
    }

    /// Check if a GPU buffer contains any inf or NaN values.
    ///
    /// Required method (no default impl): backends must provide an
    /// implementation rather than silently fall back to a host-readback
    /// scan. The previous default impl made the host detour invisible at
    /// the call site, hiding a synchronous device-to-host round-trip behind
    /// a trait-method default. Removing it forces the badness to be visible
    /// at each backend's impl site.
    fn has_inf_nan_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<bool>;

    // GPU RNG state management (for gradient checkpointing)
    /// Save the current GPU RNG state for a device. Used by checkpoint to
    /// ensure dropout masks are identical on recomputation.
    fn save_rng_state(&self, device: usize) -> FerrotorchResult<GpuRngState> {
        Err(FerrotorchError::InvalidArgument {
            message: format!("save_rng_state not implemented for device {device}"),
        })
    }

    /// Restore a previously saved GPU RNG state for a device.
    fn restore_rng_state(&self, state: GpuRngState) -> FerrotorchResult<()> {
        let _ = state;
        Err(FerrotorchError::InvalidArgument {
            message: "restore_rng_state not implemented".into(),
        })
    }

    // GPU linear algebra via cuSOLVER
    fn svd_f32(
        &self,
        _a: &GpuBufferHandle,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "svd_f32 GPU op not yet implemented".into(),
        })
    }
    fn svd_f64(
        &self,
        _a: &GpuBufferHandle,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "svd_f64 GPU op not yet implemented".into(),
        })
    }
    fn cholesky_f32(&self, _a: &GpuBufferHandle, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "cholesky_f32 GPU op not yet implemented".into(),
        })
    }
    fn cholesky_f64(&self, _a: &GpuBufferHandle, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "cholesky_f64 GPU op not yet implemented".into(),
        })
    }
    fn solve_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _n: usize,
        _nrhs: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "solve_f32 GPU op not yet implemented".into(),
        })
    }
    fn solve_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _n: usize,
        _nrhs: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "solve_f64 GPU op not yet implemented".into(),
        })
    }
    fn qr_f32(
        &self,
        _a: &GpuBufferHandle,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "qr_f32 GPU op not yet implemented".into(),
        })
    }
    fn qr_f64(
        &self,
        _a: &GpuBufferHandle,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "qr_f64 GPU op not yet implemented".into(),
        })
    }

    /// LU factorization in cuSOLVER's packed form: returns
    /// `(LU_packed, pivots)` where `LU_packed` is an `n×n` row-major GPU
    /// tensor handle (strict lower = `L`, upper = `U`), and `pivots` is a
    /// host `Vec<i32>` of length `n` (1-based row-permutation indices,
    /// LAPACK convention). The pivot vector is small (O(n)) and inherently
    /// host-readable, so we return it materialized on host rather than
    /// inventing a typed-int GPU handle. Mirrors `torch.linalg.lu_factor`.
    /// (#604)
    fn lu_factor_f32(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, Vec<i32>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "lu_factor_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 counterpart of [`Self::lu_factor_f32`]. (#604)
    fn lu_factor_f64(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, Vec<i32>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "lu_factor_f64 GPU op not yet implemented".into(),
        })
    }

    /// GPU-resident least-squares solver via cuSOLVER `cusolverDnSSgels`
    /// (iterative refinement). Solves `min ||A X - B||_F` for `A: m×n`,
    /// `B: m×nrhs`. Returns `X: n×nrhs`. Mirrors `torch.linalg.lstsq`'s
    /// solution output. (#630)
    fn lstsq_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _m: usize,
        _n: usize,
        _nrhs: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "lstsq_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 counterpart. (#630)
    fn lstsq_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _m: usize,
        _n: usize,
        _nrhs: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "lstsq_f64 GPU op not yet implemented".into(),
        })
    }

    /// Non-symmetric eigendecomposition via cuSOLVER `cusolverDnXgeev`.
    /// Returns `(eigenvalues, eigenvectors)` as **complex** GPU tensors:
    ///   - eigenvalues: length `2n` interleaved re/im (logical `[n, 2]`)
    ///   - eigenvectors: length `2 * n * n` row-major interleaved
    ///     (logical `[n, n, 2]`)
    ///
    /// Mirrors `torch.linalg.eig`. (#631)
    fn eig_f32(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "eig_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 counterpart. (#631)
    fn eig_f64(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "eig_f64 GPU op not yet implemented".into(),
        })
    }
    /// Symmetric eigendecomposition (eigenvalues + eigenvectors) of an
    /// `n × n` real symmetric matrix. Returns `(eigenvalues, eigenvectors)`
    /// where eigenvectors is row-major with column `j` the `j`-th eigenvector.
    fn eigh_f32(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "eigh_f32 GPU op not yet implemented".into(),
        })
    }
    fn eigh_f64(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "eigh_f64 GPU op not yet implemented".into(),
        })
    }
    /// Eigenvalues only of an `n × n` real symmetric matrix.
    fn eigvalsh_f32(&self, _a: &GpuBufferHandle, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "eigvalsh_f32 GPU op not yet implemented".into(),
        })
    }
    fn eigvalsh_f64(&self, _a: &GpuBufferHandle, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "eigvalsh_f64 GPU op not yet implemented".into(),
        })
    }

    // GPU 1-D FFT primitives via cuFFT. (#579)
    //
    // - C2C: input/output layout `[batch * n * 2]` interleaved (re, im).
    // - R2C: input `[batch * n]` real → output `[batch * (n/2+1) * 2]` complex.
    // - C2R: input `[batch * (n_out/2+1) * 2]` complex → output `[batch * n_out]` real.
    // - Inverse transforms include 1/n normalization to match torch / numpy.
    fn fft_c2c_f32(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _n: usize,
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fft_c2c_f32 GPU op not yet implemented".into(),
        })
    }
    fn fft_c2c_f64(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _n: usize,
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fft_c2c_f64 GPU op not yet implemented".into(),
        })
    }

    /// GPU pad/truncate for complex tensors stored as `[batch, n, 2]`
    /// (#605). Used by the FFT path when the user passes `n != input_n` —
    /// allocates a `[batch, dst_n, 2]` output, copies the visible portion
    /// from `src`, and zero-fills the tail. Single PTX kernel, no host
    /// bounce.
    fn pad_truncate_complex_f32(
        &self,
        _src: &GpuBufferHandle,
        _batch: usize,
        _src_n: usize,
        _dst_n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "pad_truncate_complex_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 counterpart of [`Self::pad_truncate_complex_f32`]. (#605)
    fn pad_truncate_complex_f64(
        &self,
        _src: &GpuBufferHandle,
        _batch: usize,
        _src_n: usize,
        _dst_n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "pad_truncate_complex_f64 GPU op not yet implemented".into(),
        })
    }

    /// 2-D complex-to-complex FFT via cufftPlan2d. Input/output layout
    /// `[h, w, 2]` interleaved complex. `inverse=true` divides by `h*w`.
    /// (#634)
    fn fft2_c2c_f32(
        &self,
        _a: &GpuBufferHandle,
        _h: usize,
        _w: usize,
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fft2_c2c_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 2-D FFT counterpart. (#634)
    fn fft2_c2c_f64(
        &self,
        _a: &GpuBufferHandle,
        _h: usize,
        _w: usize,
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fft2_c2c_f64 GPU op not yet implemented".into(),
        })
    }

    /// Broadcast a `[outer, inner]` tensor into `[outer, repeat_count, inner]`
    /// by replicating along the inserted middle dim. Used for sum_dim /
    /// mean_dim backward where the gradient must be expanded along the
    /// previously-reduced dim. (#524)
    fn repeat_along_dim_f32(
        &self,
        _input: &GpuBufferHandle,
        _outer: usize,
        _repeat_count: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "repeat_along_dim_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 counterpart. (#524)
    fn repeat_along_dim_f64(
        &self,
        _input: &GpuBufferHandle,
        _outer: usize,
        _repeat_count: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "repeat_along_dim_f64 GPU op not yet implemented".into(),
        })
    }
    fn rfft_r2c_f32(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "rfft_r2c_f32 GPU op not yet implemented".into(),
        })
    }
    fn rfft_r2c_f64(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "rfft_r2c_f64 GPU op not yet implemented".into(),
        })
    }
    fn irfft_c2r_f32(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _n_out: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "irfft_c2r_f32 GPU op not yet implemented".into(),
        })
    }
    fn irfft_c2r_f64(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _n_out: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "irfft_c2r_f64 GPU op not yet implemented".into(),
        })
    }

    /// Hermitian FFT: `hfft(x, n) = irfft(conj(x), n)` on GPU via cuFFT. (#636)
    ///
    /// Input `[batch, half_in, 2]` complex; output `[batch * n_out]` real.
    /// `half_in` must equal `n_out / 2 + 1`.
    fn hfft_f32(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _half_in: usize,
        _n_out: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "hfft_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 Hermitian FFT counterpart. (#636)
    fn hfft_f64(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _half_in: usize,
        _n_out: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "hfft_f64 GPU op not yet implemented".into(),
        })
    }

    /// Inverse Hermitian FFT: `ihfft(x) = conj(rfft(x)) / n` on GPU. (#636)
    ///
    /// Input `[batch * n]` real; output `[batch, n/2+1, 2]` complex.
    fn ihfft_f32(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "ihfft_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 inverse Hermitian FFT counterpart. (#636)
    fn ihfft_f64(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "ihfft_f64 GPU op not yet implemented".into(),
        })
    }

    /// 3-D complex-to-complex FFT via `cufftPlan3d`. (#636)
    ///
    /// Input/output layout `[d, h, w, 2]` interleaved complex.
    /// `inverse=true` divides by `d*h*w`.
    fn fftn3d_c2c_f32(
        &self,
        _a: &GpuBufferHandle,
        _d: usize,
        _h: usize,
        _w: usize,
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fftn3d_c2c_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 3-D FFT counterpart. (#636)
    fn fftn3d_c2c_f64(
        &self,
        _a: &GpuBufferHandle,
        _d: usize,
        _h: usize,
        _w: usize,
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fftn3d_c2c_f64 GPU op not yet implemented".into(),
        })
    }

    /// 2-D complex-to-complex FFT via `cufftPlanMany` for f32. (#636)
    ///
    /// Input/output layout `[h, w, 2]` interleaved complex.
    /// `inverse=true` divides by `h*w`.
    fn fftn2d_c2c_f32(
        &self,
        _a: &GpuBufferHandle,
        _h: usize,
        _w: usize,
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fftn2d_c2c_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 2-D FFT counterpart. (#636)
    fn fftn2d_c2c_f64(
        &self,
        _a: &GpuBufferHandle,
        _h: usize,
        _w: usize,
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fftn2d_c2c_f64 GPU op not yet implemented".into(),
        })
    }

    // -- axes-aware N-D FFT via cufftPlanMany (#966) -------------------------

    /// Axes-aware N-D complex-to-complex FFT for f32 via `cufftPlanMany`. (#966)
    ///
    /// Transforms over the specified `axes` (normalized to non-negative in
    /// `[0, ndim)`) of the complex input tensor. Input layout: interleaved
    /// `[..., 2]` (re/im pairs); the trailing complex dim is always the last
    /// and is NOT included in `axes`. `shape` is the tensor's spatial dims
    /// (excluding the trailing 2).
    ///
    /// `inverse=true` applies `1 / product(shape[ax] for ax in axes)`
    /// normalization to match `torch.fft.ifftn`.
    ///
    /// Default impl returns `Err` so existing backends compile unchanged.
    fn fftn_axes_c2c_f32(
        &self,
        _a: &GpuBufferHandle,
        _shape: &[usize],
        _axes: &[usize],
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fftn_axes_c2c_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// f64 variant of [`Self::fftn_axes_c2c_f32`]. (#966)
    fn fftn_axes_c2c_f64(
        &self,
        _a: &GpuBufferHandle,
        _shape: &[usize],
        _axes: &[usize],
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fftn_axes_c2c_f64 GPU op not implemented for this backend".into(),
        })
    }

    /// Fused Adam optimizer step: updates param, exp_avg, and exp_avg_sq
    /// in a single kernel launch.
    ///
    /// All four buffers (`param`, `grad`, `exp_avg`, `exp_avg_sq`) must have
    /// the same length. `param`, `exp_avg`, and `exp_avg_sq` are modified
    /// in-place.
    #[allow(clippy::too_many_arguments)]
    fn fused_adam_f32(
        &self,
        _param: &mut GpuBufferHandle,
        _grad: &GpuBufferHandle,
        _exp_avg: &mut GpuBufferHandle,
        _exp_avg_sq: &mut GpuBufferHandle,
        _beta1: f32,
        _beta2: f32,
        _lr: f32,
        _eps: f32,
        _bc1: f32,
        _bc2: f32,
        _weight_decay: f32,
    ) -> FerrotorchResult<()> {
        Err(FerrotorchError::InvalidArgument {
            message: "fused_adam_f32 GPU op not yet implemented".into(),
        })
    }

    /// Fused GRU cell forward: pointwise gate computation on pre-computed
    /// gate matrices. Returns `(hy_handle, workspace_handle)`.
    ///
    /// `input_gates` and `hidden_gates` are `[batch, 3*hsz]` from cuBLAS GEMMs.
    /// `bias_ih` and `bias_hh` are `[3*hsz]`. `hx` is `[batch, hsz]`.
    /// `workspace` is `[batch, 5*hsz]` saved for backward.
    fn fused_gru_cell_f32(
        &self,
        _input_gates: &GpuBufferHandle,
        _hidden_gates: &GpuBufferHandle,
        _bias_ih: &GpuBufferHandle,
        _bias_hh: &GpuBufferHandle,
        _hx: &GpuBufferHandle,
        _hidden_size: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "fused_gru_cell_f32 GPU op not yet implemented".into(),
        })
    }

    /// GPU MaxPool2d forward.
    #[allow(clippy::too_many_arguments)]
    fn maxpool2d_f32(
        &self,
        _input: &GpuBufferHandle,
        _batch: usize,
        _channels: usize,
        _h_in: usize,
        _w_in: usize,
        _kh: usize,
        _kw: usize,
        _sh: usize,
        _sw: usize,
        _ph: usize,
        _pw: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        Err(FerrotorchError::InvalidArgument {
            message: "maxpool2d_f32 GPU op not yet implemented".into(),
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn maxpool2d_f64(
        &self,
        _input: &GpuBufferHandle,
        _batch: usize,
        _channels: usize,
        _h_in: usize,
        _w_in: usize,
        _kh: usize,
        _kw: usize,
        _sh: usize,
        _sw: usize,
        _ph: usize,
        _pw: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        Err(FerrotorchError::InvalidArgument {
            message: "maxpool2d_f64 GPU op not yet implemented".into(),
        })
    }

    /// GPU AvgPool2d forward.
    #[allow(clippy::too_many_arguments)]
    fn avgpool2d_f32(
        &self,
        _input: &GpuBufferHandle,
        _batch: usize,
        _channels: usize,
        _h_in: usize,
        _w_in: usize,
        _kh: usize,
        _kw: usize,
        _sh: usize,
        _sw: usize,
        _ph: usize,
        _pw: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        Err(FerrotorchError::InvalidArgument {
            message: "avgpool2d_f32 GPU op not yet implemented".into(),
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn avgpool2d_f64(
        &self,
        _input: &GpuBufferHandle,
        _batch: usize,
        _channels: usize,
        _h_in: usize,
        _w_in: usize,
        _kh: usize,
        _kw: usize,
        _sh: usize,
        _sw: usize,
        _ph: usize,
        _pw: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        Err(FerrotorchError::InvalidArgument {
            message: "avgpool2d_f64 GPU op not yet implemented".into(),
        })
    }

    /// GPU Conv2d forward: im2col + GEMM + bias add, entirely on-device.
    ///
    /// Supports the full `Conv2d::new_full` parameter surface: `groups`
    /// partitions input/output channels and `dilation` spaces the kernel
    /// taps. The dispatch happens on the GPU for every value of these
    /// parameters; there is no CPU detour. Pass `groups = 1` and
    /// `dilation = (1, 1)` for the dense convolution case.
    ///
    /// Returns `(output_handle, output_shape)` where output_shape is `[B, C_out, H_out, W_out]`.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_f32(
        &self,
        _input: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _bias: Option<&GpuBufferHandle>,
        _input_shape: [usize; 4],
        _weight_shape: [usize; 4],
        _stride: (usize, usize),
        _padding: (usize, usize),
        _dilation: (usize, usize),
        _groups: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        Err(FerrotorchError::InvalidArgument {
            message: "conv2d_f32 GPU op not yet implemented".into(),
        })
    }
    /// GPU Conv2d forward (f64). See [`Self::conv2d_f32`] for parameter
    /// semantics — this is the f64 companion.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_f64(
        &self,
        _input: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _bias: Option<&GpuBufferHandle>,
        _input_shape: [usize; 4],
        _weight_shape: [usize; 4],
        _stride: (usize, usize),
        _padding: (usize, usize),
        _dilation: (usize, usize),
        _groups: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        Err(FerrotorchError::InvalidArgument {
            message: "conv2d_f64 GPU op not yet implemented".into(),
        })
    }

    // -- Sparse SpMM (cuSPARSE, CSR format) -----------------------------------
    //
    // These cover `SparseTensor::spmm` when the dense operand is a CUDA
    // tensor — PyTorch's `torch.sparse.mm` runs on cuSPARSE in that case.
    // Just-in-time CSR upload from the caller's host-side `(crow_indices,
    // col_indices, values)`; the dense operand is already device-resident.
    // Output `[m, n]` row-major lives on the same device as the dense input.
    //
    // See ferrotorch-core/src/sparse.rs `SparseTensor::spmm` for the call
    // site and ferrotorch-gpu/src/sparse.rs for the cuSPARSE implementation.

    /// CSR sparse-dense matmul on GPU (f32 dtype).
    ///
    /// `crow_indices`: `m + 1` host `u32` row pointers in CSR order.
    /// `col_indices`: `nnz` host `u32` column indices.
    /// `values`: `nnz` host f32 non-zero values.
    /// `dense`: device buffer holding a `[k, n]` row-major dense matrix.
    /// Returns a device buffer holding the `[m, n]` row-major dense result.
    fn spmm_csr_f32(
        &self,
        _crow_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f32],
        _dense: &GpuBufferHandle,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "spmm_csr_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// CSR sparse-dense matmul on GPU (f64 dtype). Companion of
    /// [`Self::spmm_csr_f32`].
    fn spmm_csr_f64(
        &self,
        _crow_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f64],
        _dense: &GpuBufferHandle,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "spmm_csr_f64 GPU op not implemented for this backend".into(),
        })
    }

    // -- Sparse <-> Dense conversion (cuSPARSE) -------------------------------
    //
    // P3 covers `SparseTensor::to_dense_on(Device::Cuda)` and the GPU branch
    // of `SparseTensor::from_dense` when the dense tensor lives on CUDA.
    // PyTorch parity (rust-gpu-discipline §3): `torch.Tensor.to_dense()` and
    // `torch.Tensor.to_sparse()` keep the result on the input device and
    // dispatch to cuSPARSE on CUDA. We mirror that.
    //
    // These are CSR-shaped: ferrotorch's internal SpMM path is CSR, and
    // cuSPARSE's `*_to_dense`/`*_to_sparse` accept CSR descriptors. The COO
    // → CSR build (host-side row-pointer prefix sum) reuses the same code
    // path as `SparseTensor::spmm`.
    //
    // See ferrotorch-gpu/src/sparse.rs for the implementation.

    /// CSR-form sparse → dense materialization on GPU (f32 dtype).
    ///
    /// Inputs:
    /// - `crow_indices`: `m + 1` host `u32` row pointers.
    /// - `col_indices`: `nnz` host `u32` column indices.
    /// - `values`: `nnz` host f32 non-zero values.
    /// - `device_ordinal`: target CUDA ordinal; output buffer lives there.
    /// - `m`, `n`: output dense shape `[m, n]`, row-major.
    ///
    /// Returns a device buffer holding the `[m, n]` row-major dense result.
    fn sparse_to_dense_csr_f32(
        &self,
        _crow_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f32],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sparse_to_dense_csr_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// CSR-form sparse → dense materialization on GPU (f64 dtype).
    /// Companion of [`Self::sparse_to_dense_csr_f32`].
    fn sparse_to_dense_csr_f64(
        &self,
        _crow_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f64],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sparse_to_dense_csr_f64 GPU op not implemented for this backend".into(),
        })
    }

    /// Dense → CSR-form sparse extraction on GPU (f32 dtype).
    ///
    /// Reads a row-major `[m, n]` device dense matrix and returns the CSR
    /// triplet `(crow_indices, col_indices, values)` with **only exact-zero**
    /// entries dropped (PyTorch's `torch.Tensor.to_sparse()` semantics — non-
    /// zero thresholds must be applied by the caller).
    ///
    /// Returns host-side `Vec`s — the caller decides whether to coalesce or
    /// store on device. ferrotorch's `SparseTensor` is CPU-resident so the
    /// host-side return matches that storage model.
    fn dense_to_sparse_csr_f32(
        &self,
        _dense: &GpuBufferHandle,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f32>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "dense_to_sparse_csr_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// Dense → CSR-form sparse extraction on GPU (f64 dtype).
    /// Companion of [`Self::dense_to_sparse_csr_f32`].
    fn dense_to_sparse_csr_f64(
        &self,
        _dense: &GpuBufferHandle,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f64>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "dense_to_sparse_csr_f64 GPU op not implemented for this backend".into(),
        })
    }

    // -- CSR/CSC/COO format-conversion + to_dense (cuSPARSE) -- P7 ------------
    //
    // PyTorch parity (rust-gpu-discipline §3): `torch.sparse_csr_tensor` /
    // `torch.sparse_csc_tensor` / `torch.sparse_coo_tensor` keep the result on
    // the input device, and the format-conversion helpers (`.to_sparse_csr()`,
    // `.to_sparse_csc()`, `.to_dense()` on a CSR/CSC/COO tensor) run on
    // cuSPARSE when the data lives on CUDA. ferrotorch routes CSR↔CSC via
    // `cusparseCsr2cscEx2` and COO↔CSR via `cusparseXcoo2csr` /
    // `cusparseXcsr2coo` (host inputs uploaded JIT, dense output stays on
    // device).
    //
    // The CSR-shaped sparse-to-dense path already exists as
    // `sparse_to_dense_csr_f{32,64}` (P3); the CSC variants below are dual
    // paths that take a CSC triplet and materialise the dense matrix on
    // device. Per-component dispatch from `CscTensor::to_dense_on` and
    // `CooTensor::to_dense_on`.
    //
    // See ferrotorch-gpu/src/sparse.rs for the cuSPARSE implementations.

    /// CSC-form sparse → dense materialization on GPU (f32 dtype).
    ///
    /// Inputs:
    /// - `col_ptrs`: `n + 1` host `u32` column pointers in CSC order.
    /// - `row_indices`: `nnz` host `u32` row indices.
    /// - `values`: `nnz` host f32 non-zero values.
    /// - `device_ordinal`: target CUDA ordinal; output buffer lives there.
    /// - `m`, `n`: output dense shape `[m, n]`, row-major.
    ///
    /// Returns a device buffer holding the `[m, n]` row-major dense result.
    fn csc_to_dense_f32(
        &self,
        _col_ptrs: &[u32],
        _row_indices: &[u32],
        _values: &[f32],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "csc_to_dense_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// CSC-form sparse → dense materialization on GPU (f64 dtype).
    /// Companion of [`Self::csc_to_dense_f32`].
    fn csc_to_dense_f64(
        &self,
        _col_ptrs: &[u32],
        _row_indices: &[u32],
        _values: &[f64],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "csc_to_dense_f64 GPU op not implemented for this backend".into(),
        })
    }

    /// CSR → CSC format conversion on GPU (f32 dtype).
    ///
    /// Uses `cusparseCsr2cscEx2`. Returns the CSC triplet
    /// `(col_ptrs, row_indices, values)` in host buffers — ferrotorch's
    /// `CscTensor` is CPU-resident so the host return matches that storage
    /// model. `m`, `n` are the source CSR shape.
    fn csr_to_csc_f32(
        &self,
        _crow_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f32],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f32>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "csr_to_csc_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// CSR → CSC format conversion on GPU (f64 dtype).
    /// Companion of [`Self::csr_to_csc_f32`].
    fn csr_to_csc_f64(
        &self,
        _crow_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f64],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f64>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "csr_to_csc_f64 GPU op not implemented for this backend".into(),
        })
    }

    /// COO → CSR format conversion on GPU (f32 dtype).
    ///
    /// Wraps `cusparseXcoo2csr`. Caller supplies row-sorted COO (cuSPARSE
    /// requires it); ferrotorch's `CooTensor` is host-resident so the
    /// caller pre-sorts on the host before invoking. Values are passed
    /// through unchanged (only the row indices are compacted into a
    /// `crow_indices` row-pointer array).
    ///
    /// Returns the host CSR triplet `(crow_indices, col_indices, values)`.
    fn coo_to_csr_f32(
        &self,
        _row_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f32],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f32>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "coo_to_csr_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// COO → CSR format conversion on GPU (f64 dtype).
    /// Companion of [`Self::coo_to_csr_f32`].
    fn coo_to_csr_f64(
        &self,
        _row_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f64],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f64>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "coo_to_csr_f64 GPU op not implemented for this backend".into(),
        })
    }

    /// CSR → COO row-index expansion on GPU (f32 dtype).
    ///
    /// Wraps `cusparseXcsr2coo`. Returns the host COO triplet
    /// `(row_indices, col_indices, values)`. `col_indices` and `values`
    /// pass through unchanged from the source CSR; only the `crow_indices`
    /// row-pointer array is expanded into per-entry `row_indices`.
    fn csr_to_coo_f32(
        &self,
        _crow_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f32],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f32>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "csr_to_coo_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// CSR → COO row-index expansion on GPU (f64 dtype).
    /// Companion of [`Self::csr_to_coo_f32`].
    fn csr_to_coo_f64(
        &self,
        _crow_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f64],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f64>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "csr_to_coo_f64 GPU op not implemented for this backend".into(),
        })
    }

    /// Synchronize the current stream on the given device, blocking until
    /// all enqueued operations have completed.
    fn synchronize(&self, _device: usize) -> FerrotorchResult<()> {
        Err(FerrotorchError::DeviceUnavailable)
    }

    /// Return the number of streams in the pool for the given device.
    fn stream_count(&self, _device: usize) -> usize {
        1
    }

    // ---------------------------------------------------------------------
    // FlashAttention forward (P5).
    //
    // Per-component dispatch from `nested_scaled_dot_product_attention`.
    // Computes `softmax(Q @ K^T * scale) @ V` with on-device tiled
    // online-softmax (FlashAttention-2 forward) without materialising the
    // full [seq_q, seq_k] scores matrix.
    //
    // Default impls return `InvalidArgument` so backends without CUDA
    // declare unsupported, and the caller falls through to the GPU
    // composite path (`bmm + softmax_rows + bmm`).
    // ---------------------------------------------------------------------

    /// FlashAttention forward (f32). `query`/`key` are `[seq_q, d]` /
    /// `[seq_k, d]`; `value` is `[seq_k, d_v]`. Returns `[seq_q, d_v]`.
    /// `scale` is typically `1 / sqrt(d)`. Single-head (no batch dim) —
    /// the caller folds batch into per-component dispatch.
    fn flash_attention_forward_f32(
        &self,
        _query: &GpuBufferHandle,
        _key: &GpuBufferHandle,
        _value: &GpuBufferHandle,
        _seq_q: usize,
        _seq_k: usize,
        _d: usize,
        _d_v: usize,
        _scale: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "flash_attention_forward_f32 GPU op not yet implemented".into(),
        })
    }

    /// FlashAttention forward (f64). See [`flash_attention_forward_f32`].
    fn flash_attention_forward_f64(
        &self,
        _query: &GpuBufferHandle,
        _key: &GpuBufferHandle,
        _value: &GpuBufferHandle,
        _seq_q: usize,
        _seq_k: usize,
        _d: usize,
        _d_v: usize,
        _scale: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "flash_attention_forward_f64 GPU op not yet implemented".into(),
        })
    }

    // ---------------------------------------------------------------------
    // P6: 2:4 structured sparse matmul (cuSPARSELt).
    //
    // PyTorch parity (rust-gpu-discipline §3): `torch._C._sparse_semi_
    // structured_apply` (and the `SparseSemiStructuredTensor` user API)
    // dispatches to NVIDIA cuSPARSELt on Ampere+ Tensor Cores when the
    // 2:4 sparse weight is on CUDA. ferrotorch mirrors that by routing
    // `SemiStructuredSparseTensor::sparse_matmul_24` through these
    // hooks.
    //
    // The dense `b_dense_decompressed` operand is the dense
    // representation of the structured 2:4 matrix (mask applied → zeros
    // in non-retained positions). cuSPARSELt repacks it into the
    // Tensor-Core-friendly layout via `cusparseLtSpMMACompress`
    // internally.
    //
    // Default impls return `InvalidArgument` so backends without the
    // `cusparselt` cargo feature declare unsupported and the caller
    // falls through to the existing decompress + dense matmul reference
    // path. This is the §3-correct opt-in mechanism — no silent CPU
    // fallback. See `ferrotorch-gpu/src/cusparselt.rs` for the live
    // implementation under the feature.
    // ---------------------------------------------------------------------

    /// 2:4 structured sparse matmul, FP32 (TF32-Tensor-Core compute).
    /// Computes `[m, n] = a @ b_dense_decompressed` where `b` carries
    /// 2:4 structured zeros along its inner dimension.
    fn sparse_matmul_24_f32(
        &self,
        _a: &GpuBufferHandle,
        _b_dense_decompressed: &GpuBufferHandle,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sparse_matmul_24_f32 GPU op not implemented for this backend (build with --features cusparselt)".into(),
        })
    }

    /// 2:4 structured sparse matmul, FP16 (FP32 accumulator).
    /// f16 inputs/outputs are passed as raw `u16` buffers (bf16/f16 in
    /// ferrotorch are `u16`-bit-pattern carriers).
    fn sparse_matmul_24_f16(
        &self,
        _a: &GpuBufferHandle,
        _b_dense_decompressed: &GpuBufferHandle,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sparse_matmul_24_f16 GPU op not implemented for this backend (build with --features cusparselt)".into(),
        })
    }

    /// 2:4 structured sparse matmul, BF16 (FP32 accumulator).
    /// See [`Self::sparse_matmul_24_f16`].
    fn sparse_matmul_24_bf16(
        &self,
        _a: &GpuBufferHandle,
        _b_dense_decompressed: &GpuBufferHandle,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sparse_matmul_24_bf16 GPU op not implemented for this backend (build with --features cusparselt)".into(),
        })
    }

    // -- bf16 → bf16 native dispatch (#17) -----------------------------------
    //
    // These trait methods stay in bf16 end-to-end (inputs *and* outputs are
    // `CudaSlice<u16>` bit-pattern handles, matching the storage convention
    // used by every `*_bf16` PTX kernel in `ferrotorch-gpu::bf16` and the
    // `gpu_matmul_bf16_bf16` family in `ferrotorch-gpu::blas`). The earlier
    // `*_bf16_f32` family widens to f32 on the output (PyTorch-autocast
    // parity); these `*_bf16_bf16` variants preserve bf16 storage end-to-end
    // for the ViT / CLIP-style inference pipeline.
    //
    // Default impls return `Unsupported` so non-CUDA backends compile
    // unchanged — there is NO silent CPU fallback (rust-gpu-discipline §3).
    // The CUDA implementation is in `ferrotorch-gpu::backend_impl`.

    /// bf16 fused-transpose matmul `C = A @ B^T`.
    ///
    /// `A: [m, k]`, `B: [n, k]` (row-major; the transpose is folded into
    /// the cuBLAS `transb` flag with no extra memory traffic). Returns a
    /// `[m, n]` bf16 buffer.
    fn matmul_bf16_bf16_nt(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "matmul_bf16_bf16_nt GPU op not implemented for this backend".into(),
        })
    }

    /// Row-wise softmax: bf16 input → bf16 output via PTX kernel with
    /// f32 accumulator (max-find, exp-sum, normalize all in f32; only the
    /// final store rounds back to bf16). The bf16 round-trip is the
    /// HuggingFace bf16 attention contract.
    fn softmax_bf16_bf16(
        &self,
        _a: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "softmax_bf16_bf16 GPU op not implemented for this backend".into(),
        })
    }

    /// bf16 LayerNorm with per-channel γ/β (also bf16).
    /// `input: [rows, cols]`, `gamma: [cols]`, `beta: [cols]`. The per-row
    /// mean and variance reduce in f32; the final scale-shift result rounds
    /// back to bf16 with round-to-nearest-even.
    fn layernorm_bf16_bf16(
        &self,
        _input: &GpuBufferHandle,
        _gamma: &GpuBufferHandle,
        _beta: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _eps: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "layernorm_bf16_bf16 GPU op not implemented for this backend".into(),
        })
    }

    /// bf16 GELU activation `out = 0.5 * x * (1 + erf(x / sqrt(2)))`,
    /// computed in f32 (Hastings degree-5 erf polynomial; ≤1.5e-7 max abs
    /// error, well below bf16 ULP) and rounded back to bf16. ViT and CLIP
    /// MLP blocks use this exact formulation.
    fn gelu_bf16_bf16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "gelu_bf16_bf16 GPU op not implemented for this backend".into(),
        })
    }

    /// bf16 SiLU activation `out = x * sigmoid(x)`, f32 internal, bf16 RNE
    /// store back.
    fn silu_bf16_bf16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "silu_bf16_bf16 GPU op not implemented for this backend".into(),
        })
    }

    /// bf16 ReLU activation `out = max(0, x)` (clamp on the bf16 sign bit).
    fn relu_bf16_bf16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "relu_bf16_bf16 GPU op not implemented for this backend".into(),
        })
    }

    /// bf16 elementwise add `out = a + b`. f32 internal, bf16 RNE store back.
    fn add_bf16_bf16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "add_bf16_bf16 GPU op not implemented for this backend".into(),
        })
    }

    /// bf16 elementwise multiply `out = a * b`. f32 internal, bf16 RNE
    /// store back.
    fn mul_bf16_bf16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "mul_bf16_bf16 GPU op not implemented for this backend".into(),
        })
    }

    /// bf16 scalar multiply `out = a * scalar`. Used to fold
    /// `1 / sqrt(head_dim)` into attention scores.
    fn scale_bf16_bf16(
        &self,
        _a: &GpuBufferHandle,
        _scalar: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "scale_bf16_bf16 GPU op not implemented for this backend".into(),
        })
    }

    // ─── Issue #23: bf16 dispatch-gap closure ──────────────────────────────
    //
    // These trait methods close the dispatcher gap surfaced by
    // forecast-bio/ferrotorch#23. They cover sub / div / neg, broadcast
    // {add, sub, mul, div}, sum / mean (both scalar + axis), and the
    // transcendentals exp / log / tanh / sigmoid. Each has a default
    // `Err(NotImplementedOnCuda)` body so non-CUDA backends (cubecl,
    // mps, xpu) compile untouched; the CUDA backend overrides them in
    // `ferrotorch-gpu::backend_impl`.

    /// bf16 elementwise subtract `out = a - b`. f32 internal, bf16 RNE store back.
    fn sub_bf16_bf16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "sub_bf16_bf16",
        })
    }

    /// bf16 elementwise divide `out = a / b`. f32 internal, bf16 RNE store back.
    fn div_bf16_bf16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "div_bf16_bf16",
        })
    }

    /// bf16 elementwise negate `out = -a`. Implemented as a sign-bit XOR
    /// on the u16 bit pattern — no f32 round-trip.
    fn neg_bf16_bf16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "neg_bf16_bf16",
        })
    }

    /// bf16 broadcast add. `a_shape`, `b_shape` are the original shapes;
    /// `out_shape` is the numpy-style broadcasted output shape.
    fn broadcast_add_bf16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "broadcast_add_bf16",
        })
    }

    /// bf16 broadcast sub.
    fn broadcast_sub_bf16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "broadcast_sub_bf16",
        })
    }

    /// bf16 broadcast mul.
    fn broadcast_mul_bf16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "broadcast_mul_bf16",
        })
    }

    /// bf16 broadcast div.
    fn broadcast_div_bf16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "broadcast_div_bf16",
        })
    }

    /// bf16 sum-reduce to scalar. PyTorch parity: accumulator is f32, final
    /// store rounds back to bf16 with round-to-nearest-even.
    fn sum_bf16_bf16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "sum_bf16_bf16",
        })
    }

    /// bf16 mean-reduce to scalar. Computed via sum_bf16_bf16 / n on-device.
    fn mean_bf16_bf16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mean_bf16_bf16",
        })
    }

    /// bf16 axis-reduce sum. `shape` is the full input shape; `axis` is
    /// the index of the dimension being reduced. Output has the same shape
    /// minus the reduced dim (caller may keepdim if desired). f32
    /// accumulator, bf16 round-back.
    fn sum_axis_bf16_bf16(
        &self,
        _a: &GpuBufferHandle,
        _shape: &[usize],
        _axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "sum_axis_bf16_bf16",
        })
    }

    /// bf16 axis-reduce mean. Same shape/axis contract as
    /// [`sum_axis_bf16_bf16`]. f32 accumulator, divides by `shape[axis]`
    /// before bf16 round-back.
    fn mean_axis_bf16_bf16(
        &self,
        _a: &GpuBufferHandle,
        _shape: &[usize],
        _axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mean_axis_bf16_bf16",
        })
    }

    /// bf16 elementwise exp. f32 internal via `ex2.approx.f32(x * log2(e))`,
    /// bf16 RNE store back.
    fn exp_bf16_bf16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "exp_bf16_bf16",
        })
    }

    /// bf16 elementwise natural log. f32 internal via
    /// `lg2.approx.f32(x) * ln(2)`, bf16 RNE store back.
    fn log_bf16_bf16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "log_bf16_bf16",
        })
    }

    /// bf16 elementwise tanh. f32 internal via `(e^(2x) - 1)/(e^(2x) + 1)`,
    /// bf16 RNE store back.
    fn tanh_bf16_bf16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "tanh_bf16_bf16",
        })
    }

    /// bf16 elementwise sigmoid `1 / (1 + exp(-x))`. f32 internal, bf16 RNE
    /// store back.
    fn sigmoid_bf16_bf16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "sigmoid_bf16_bf16",
        })
    }

    // ── IEEE float16 (f16) ops — crosslink #1185 Phase 1 ─────────────────────
    //
    // f16 storage is `CudaSlice<u16>` (same width as bf16) but the
    // `GpuBufferHandle` carries `DType::F16`, so `unwrap_buffer_f16` asserts
    // the F16 tag and rejects a BF16-tagged handle (and vice-versa). All math
    // happens in f32 registers per thread (native `cvt.f32.f16` /
    // `cvt.rn.f16.f32`); reductions accumulate in f32 (PyTorch parity). These
    // default bodies return a structured error so non-CUDA backends compile
    // unchanged; `CudaBackendImpl` overrides each one.

    /// f16 elementwise `out = a + b`, f32 compute, f16 RNE store.
    fn add_f16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "add_f16" })
    }

    /// f16 elementwise `out = a - b`, f32 compute, f16 RNE store.
    fn sub_f16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "sub_f16" })
    }

    /// f16 elementwise `out = a * b`, f32 compute, f16 RNE store.
    fn mul_f16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "mul_f16" })
    }

    /// f16 elementwise `out = a / b`, f32 compute, f16 RNE store.
    fn div_f16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "div_f16" })
    }

    /// f16 elementwise `out = -a`.
    fn neg_f16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "neg_f16" })
    }

    /// f16 multiply every element by an f32 scalar (`out = a * scale`).
    fn scale_f16(&self, _a: &GpuBufferHandle, _scale: f32) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "scale_f16" })
    }

    /// f16 broadcast add over N-D broadcast shapes.
    fn broadcast_add_f16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "broadcast_add_f16",
        })
    }

    /// f16 broadcast sub over N-D broadcast shapes.
    fn broadcast_sub_f16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "broadcast_sub_f16",
        })
    }

    /// f16 broadcast mul over N-D broadcast shapes.
    fn broadcast_mul_f16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "broadcast_mul_f16",
        })
    }

    /// f16 broadcast div over N-D broadcast shapes.
    fn broadcast_div_f16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "broadcast_div_f16",
        })
    }

    /// f16 sum-reduce to scalar. f32 accumulator (PyTorch parity).
    fn sum_f16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "sum_f16" })
    }

    /// f16 mean-reduce to scalar. Computed via `sum_f16 / n` on-device.
    fn mean_f16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "mean_f16" })
    }

    /// f16 axis sum-reduce. f32 accumulator; collapses `shape[axis]`.
    fn sum_axis_f16(
        &self,
        _a: &GpuBufferHandle,
        _shape: &[usize],
        _axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "sum_axis_f16" })
    }

    /// f16 axis mean-reduce. f32 accumulator; divides by `shape[axis]`.
    fn mean_axis_f16(
        &self,
        _a: &GpuBufferHandle,
        _shape: &[usize],
        _axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "mean_axis_f16",
        })
    }

    /// f16 elementwise `out = exp(a)`. f32 internal, f16 RNE store.
    fn exp_f16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "exp_f16" })
    }

    /// f16 elementwise `out = ln(a)`. f32 internal, f16 RNE store.
    fn log_f16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "log_f16" })
    }

    /// f16 elementwise tanh. f32 internal, f16 RNE store.
    fn tanh_f16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "tanh_f16" })
    }

    /// f16 elementwise sigmoid `1 / (1 + exp(-x))`. f32 internal, f16 RNE store.
    fn sigmoid_f16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "sigmoid_f16" })
    }

    /// f16 elementwise `out = sqrt(a)`. f32 internal, f16 RNE store.
    fn sqrt_f16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "sqrt_f16" })
    }

    /// f16 elementwise ReLU `max(0, a)`.
    fn relu_f16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "relu_f16" })
    }

    /// f16 elementwise SiLU `a * sigmoid(a)`. f32 internal, f16 RNE store.
    fn silu_f16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "silu_f16" })
    }

    /// f16 elementwise GELU `0.5 * x * (1 + erf(x / sqrt(2)))`. f32 internal.
    fn gelu_f16(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "gelu_f16" })
    }

    /// f16 row-wise softmax over `[rows, cols]`. f32 accumulator, f16 store.
    fn softmax_f16(
        &self,
        _a: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "softmax_f16" })
    }

    /// f16 LayerNorm over `[rows, cols]` with f16 gamma/beta. f32 reductions.
    fn layernorm_f16(
        &self,
        _input: &GpuBufferHandle,
        _gamma: &GpuBufferHandle,
        _beta: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _eps: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "layernorm_f16",
        })
    }

    /// f16 RMSNorm over `[rows, cols]` with f16 weight. f32 reductions.
    fn rmsnorm_f16(
        &self,
        _input: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _eps: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "rmsnorm_f16" })
    }

    /// f16-resident matmul `C = A @ B` (cuBLAS GemmEx, `CUDA_R_16F` operands,
    /// f32 compute). `A: [m, k]`, `B: [k, n]`, `C: [m, n]`.
    fn matmul_f16_f16(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "matmul_f16_f16",
        })
    }

    // ── Integer (i32 / i64) ops — crosslink #1185 Phase 2b ───────────────────
    //
    // Runtime dispatch on the ScalarType tag (PyTorch style): ONE trait method
    // per op, which switches on `a.dtype()` internally in the CUDA backend
    // (`DType::I32` → i32 kernel, `DType::I64` → i64 kernel, else structured
    // error). This mirrors PyTorch's dispatcher routing on `ScalarType` rather
    // than minting a separate symbol per (op, width). Native `CudaBuffer<i32>`
    // / `CudaBuffer<i64>` storage — no `CudaSlice<u16>` bit-pattern trick, no
    // f32/f64 detour, no host round-trip. Each default body returns a
    // structured error so non-CUDA backends compile unchanged; `CudaBackendImpl`
    // overrides them in `ferrotorch-gpu::backend_impl`.

    /// Integer elementwise `out = a + b` (i32 / i64, wrapping on overflow).
    /// Dispatches on `a.dtype()`. Inputs must be same width and length.
    fn int_add(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "int_add" })
    }

    /// Integer elementwise `out = a - b` (i32 / i64, wrapping).
    fn int_sub(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "int_sub" })
    }

    /// Integer elementwise `out = a * b` (i32 / i64, wrapping).
    fn int_mul(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "int_mul" })
    }

    /// Integer elementwise negate `out = -a`.
    fn int_neg(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "int_neg" })
    }

    /// Integer floor division `out = floor_divide(a, b)` (floors toward −∞,
    /// `torch.floor_divide` semantics — NOT C truncation).
    fn int_floor_div(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "int_floor_div",
        })
    }

    /// Integer remainder `out = remainder(a, b)` (sign of the DIVISOR,
    /// `torch.remainder` / Python semantics — NOT C `%`).
    fn int_remainder(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "int_remainder",
        })
    }

    /// Integer elementwise bitwise AND.
    fn int_bitand(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "int_bitand" })
    }

    /// Integer elementwise bitwise OR.
    fn int_bitor(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "int_bitor" })
    }

    /// Integer elementwise bitwise XOR.
    fn int_bitxor(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "int_bitxor" })
    }

    /// Integer elementwise bitwise NOT `out = !a`.
    fn int_bitnot(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "int_bitnot" })
    }

    /// Integer elementwise left shift `out = a << b`.
    fn int_shl(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "int_shl" })
    }

    /// Integer elementwise arithmetic right shift `out = a >> b`
    /// (sign-extending, matching PyTorch `__rshift__` on signed dtypes).
    fn int_shr(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "int_shr" })
    }

    /// Integer sum-reduce to a 1-element buffer (same width accumulator,
    /// wrapping — PyTorch does NOT upcast integer `sum`).
    fn int_sum(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "int_sum" })
    }

    /// Integer product-reduce to a 1-element buffer (same width, wrapping).
    fn int_prod(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "int_prod" })
    }

    /// Integer min-reduce to a 1-element buffer.
    fn int_min(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "int_min" })
    }

    /// Integer max-reduce to a 1-element buffer.
    fn int_max(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "int_max" })
    }

    // ── argmax / argmin / gather / cast — crosslink #1185 Phase 2c ───────────
    //
    // Cross-world integer ops that unblock the GPU-resident token/sampling path
    // (Llama generation loop). All dispatch on the relevant `DType` tag(s)
    // internally; the result stays GPU-resident. Default bodies return a
    // structured error so non-CUDA backends compile (PyTorch parity §3).

    /// Argmax over a value buffer (any float/int `DType`), returning an
    /// **I64-tagged** index handle (PyTorch returns int64 indices).
    ///
    /// Logical layout `[outer, dim_size, inner]` (contiguous, C-order). Global
    /// reduction = `outer=1, inner=1, dim_size=numel`; along-dim = the obvious
    /// factorisation. Tie-break is the FIRST occurrence. The output handle has
    /// `outer * inner` elements. Dispatches on `src.dtype()` for the value type.
    fn argmax(
        &self,
        _src: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "argmax" })
    }

    /// Argmin over a value buffer; see [`Self::argmax`]. Returns an I64 handle.
    fn argmin(
        &self,
        _src: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "argmin" })
    }

    /// On-device `searchsorted` / `bucketize` over a sorted 1-D `boundaries`
    /// buffer (#1545). For each element of `values`, returns the insertion
    /// index into `boundaries`:
    ///
    /// - `right == false` (PyTorch `side="left"`): first `i` with
    ///   `boundaries[i] >= v` (lower_bound).
    /// - `right == true` (PyTorch `side="right"`): first `i` with
    ///   `boundaries[i] > v` (upper_bound).
    ///
    /// Both `values` and `boundaries` are GPU-resident value buffers of the
    /// same `DType` (∈ {F32, F64}); the result is an `I64`-tagged handle of
    /// `values.len()` indices (PyTorch returns `ScalarType::Long`). Mirrors
    /// `searchsorted_cuda_kernel` (`is_1d_boundaries == true`) in
    /// `aten/src/ATen/native/cuda/Bucketization.cu`. The default impl errors;
    /// the CUDA backend overrides it with the `gpu_searchsorted_*` PTX kernel.
    fn searchsorted_1d(
        &self,
        _values: &GpuBufferHandle,
        _boundaries: &GpuBufferHandle,
        _right: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "searchsorted_1d",
        })
    }

    /// On-device `topk` over a GPU-resident `[outer, last_dim]` value buffer
    /// (#1545). Selects the `k` extrema along the last dim for every one of the
    /// `outer` slices, returning `(values, indices)`:
    ///
    /// - `values` — a `GpuBufferHandle` of `outer * k` elements with the SAME
    ///   `DType` as `values_in` (∈ {F32, F64}), in sorted order.
    /// - `indices` — an `I64`-tagged `GpuBufferHandle` of `outer * k` original
    ///   indices into `[0, last_dim)` (PyTorch returns `ScalarType::Long`).
    ///
    /// `largest == true` → descending value order; else ascending. Ties are
    /// broken by ascending original index, which is a valid `torch.topk`
    /// result (upstream `topk_out_cuda` gathers then sorts the top-k with
    /// `stable=false`, leaving the per-tie index order unspecified) and matches
    /// the CPU `ops::search::topk` path bit-for-bit. Mirrors
    /// `topk_out_cuda` in `aten/src/ATen/native/cuda/TensorTopK.cpp` for the
    /// last-dim, sorted case. The default impl errors; the CUDA backend
    /// overrides it with the `gpu_topk_*` PTX kernel.
    fn topk_1d(
        &self,
        _values_in: &GpuBufferHandle,
        _outer: usize,
        _last_dim: usize,
        _k: usize,
        _largest: bool,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "topk_1d" })
    }

    /// On-device `histc` over a GPU-resident value buffer (#1545). Counts the
    /// `input` elements falling in each of `bins` equal-width bins spanning the
    /// inclusive range `[min_val, max_val]`, returning a value handle of `bins`
    /// counts with the SAME `DType` as `input` (∈ {F32, F64}) — PyTorch's
    /// `_histc_cuda` allocates the output with `self.scalar_type()`.
    ///
    /// Bin semantics mirror `getBin` + `kernelHistogram1D` in
    /// `aten/src/ATen/native/cuda/SummaryOps.cu`: `bin = (int)((v - min) *
    /// bins / (max - min))`, the last bin is closed at both ends (a value `==
    /// max` lands in `bins-1`), and values outside `[min, max]` (and NaN) are
    /// not counted. The caller guarantees `bins > 0` and `min_val < max_val`.
    /// The default impl errors; the CUDA backend overrides it with the
    /// `gpu_histc_*` PTX kernel.
    fn histc_1d(
        &self,
        _input: &GpuBufferHandle,
        _bins: usize,
        _min_val: f64,
        _max_val: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "histc_1d" })
    }

    /// On-device `meshgrid` grid for ONE axis over a GPU-resident 1-D
    /// coordinate buffer (#1545, `indexing='ij'`). `input` is the axis's
    /// coordinate vector (length `axis_len`); the result is a value handle of
    /// `total` elements with the same `DType` as `input` (∈ {F32, F64}) where
    /// `out[flat] = input[(flat / inner) % axis_len]`, `inner =
    /// product(shapes[axis+1..])`, `total = product(shapes)`.
    ///
    /// This is the `view(view_shape).expand(shape)` decomposition that upstream
    /// `meshgrid` uses (`aten/src/ATen/native/TensorShape.cpp:4462-4467`) lowered
    /// to a single gather — no intermediate strided `expand` is materialised.
    /// The default impl errors; the CUDA backend overrides it with the
    /// `gpu_meshgrid_*` PTX kernel.
    fn meshgrid_grid(
        &self,
        _input: &GpuBufferHandle,
        _total: usize,
        _inner: usize,
        _axis_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "meshgrid_grid",
        })
    }

    /// `index_select(dim)` driven by a GPU-resident integer index handle.
    ///
    /// `src` is the value buffer (layout `[outer, in_dim, inner]`); `index` is
    /// an I32/I64-tagged handle of `out_dim` entries. Output is a value handle
    /// of `outer * out_dim * inner` elements with the same `DType` as `src`.
    /// Dispatches on `(src.dtype(), index.dtype())`.
    #[allow(clippy::too_many_arguments)]
    fn index_select_intidx(
        &self,
        _src: &GpuBufferHandle,
        _index: &GpuBufferHandle,
        _outer: usize,
        _in_dim: usize,
        _out_dim: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "index_select_intidx",
        })
    }

    /// `gather(dim)` driven by a GPU-resident integer index handle.
    ///
    /// `src` layout `[outer, in_dim, inner]`; `index` (I32/I64) AND output both
    /// have layout `[outer, out_dim, inner]` (the index is parallel to the
    /// output). Output `DType` matches `src`. Dispatches on
    /// `(src.dtype(), index.dtype())`.
    #[allow(clippy::too_many_arguments)]
    fn gather_intidx(
        &self,
        _src: &GpuBufferHandle,
        _index: &GpuBufferHandle,
        _outer: usize,
        _in_dim: usize,
        _out_dim: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "gather_intidx",
        })
    }

    /// Cast a float buffer (`src.dtype()` ∈ {F32,F64,BF16,F16}) to an integer
    /// buffer tagged `dst` (∈ {I32,I64}), truncating toward zero (PyTorch
    /// `.to(int)`). Result stays GPU-resident.
    fn cast_f_to_i(
        &self,
        _src: &GpuBufferHandle,
        _dst: DType,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "cast_f_to_i" })
    }

    /// Cast an integer buffer (`src.dtype()` ∈ {I32,I64}) to a float buffer
    /// tagged `dst` (∈ {F32,F64,BF16,F16}), round-to-nearest-even.
    fn cast_i_to_f(
        &self,
        _src: &GpuBufferHandle,
        _dst: DType,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "cast_i_to_f" })
    }

    /// Cast an integer buffer between i32 and i64 (`src.dtype()` and `dst`
    /// each ∈ {I32,I64}). Widen sign-extends; narrow wraps (PyTorch CUDA
    /// `.to(int)` semantics).
    fn cast_i_to_i(
        &self,
        _src: &GpuBufferHandle,
        _dst: DType,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "cast_i_to_i" })
    }

    // ── Boolean / comparison ops — crosslink #1185 Phase 3b ──────────────────
    //
    // Comparisons read a VALUE buffer (`a.dtype()` ∈ {F32,F64,BF16,F16,I32,I64})
    // and produce a `DType::Bool`-tagged output (u8, 0/1) — PyTorch parity: the
    // comparison result dtype is always `bool`. Logical ops read and write Bool
    // (u8) buffers. Reductions any/all fold a Bool buffer to a 1-element Bool
    // buffer. All dispatch on the relevant `DType` tag internally in the CUDA
    // backend; results stay GPU-resident (no host round-trip). Default bodies
    // return a structured error so non-CUDA backends compile (PyTorch parity §3).

    /// Elementwise comparison `out[i] = (a[i] OP b[i]) ? 1u8 : 0u8`.
    ///
    /// Reads the value dtype from `a.dtype()` to pick the kernel; `a` and `b`
    /// must carry the same dtype and length. The returned handle is tagged
    /// `DType::Bool` (u8 storage). `op` selects the operator.
    fn compare(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _op: CompareOp,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "compare" })
    }

    /// Elementwise logical AND of two Bool (u8) buffers → Bool (u8).
    fn bool_and(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "bool_and" })
    }

    /// Elementwise logical OR of two Bool (u8) buffers → Bool (u8).
    fn bool_or(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "bool_or" })
    }

    /// Elementwise logical XOR of two Bool (u8) buffers → Bool (u8).
    fn bool_xor(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "bool_xor" })
    }

    /// Elementwise logical NOT of a Bool (u8) buffer → Bool (u8).
    fn bool_not(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "bool_not" })
    }

    /// Global OR-reduction (`torch.any`) of a Bool (u8) buffer → 1-element
    /// Bool (u8) buffer holding 1 if any element is nonzero, else 0.
    fn bool_any(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "bool_any" })
    }

    /// Global AND-reduction (`torch.all`) of a Bool (u8) buffer → 1-element
    /// Bool (u8) buffer holding 1 if all elements are nonzero, else 0.
    fn bool_all(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "bool_all" })
    }

    /// Cast a Bool (u8) buffer to a float buffer tagged `dst`
    /// (∈ {F32,F64,BF16,F16}): `true → 1.0`, `false → 0.0`. Result stays
    /// GPU-resident.
    fn cast_bool_to_f(
        &self,
        _src: &GpuBufferHandle,
        _dst: DType,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "cast_bool_to_f",
        })
    }

    // ── #1545 / #1534: predicate masks for masked-tensor constructors ────────
    //
    // `MaskedTensor`'s mask is a host `Vec<bool>` by design. These methods
    // compute the boolean predicate ON-DEVICE from the (CUDA-resident) data
    // buffer, returning a `DType::Bool` (u8 0/1) handle. The core
    // `masked_invalid` / `masked_equal` constructors then read that mask back
    // ONCE to populate the host `Vec<bool>` — a one-way readback of the
    // freshly-computed predicate, not a CPU↔GPU round trip of the value data
    // (which never leaves and returns to the device). Dispatched on
    // `input.dtype()`; covers F32/F64 (the dtypes `MaskedTensor<T: Float>`
    // currently lowers to GPU). Default bodies return a structured error so
    // non-CUDA backends compile.

    /// `isfinite` mask: `out[i] = (v==v) && (|v| != +inf)` as a `DType::Bool`
    /// (u8 0/1) buffer. PyTorch parity with `at::isfinite`
    /// (`aten/src/ATen/native/TensorCompare.cpp:484` —
    /// `(self == self) * (self.abs() != inf)`). Consumer:
    /// `ferrotorch_core::masked_invalid` GPU branch.
    fn isfinite_mask(&self, _input: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "isfinite_mask",
        })
    }

    /// `ne_scalar` mask: `out[i] = (v != value)` as a `DType::Bool` (u8 0/1)
    /// buffer (`value` passed as f64, narrowed to the input dtype). This is the
    /// VALID mask for `numpy.ma.masked_equal` under the torch convention.
    /// Consumer: `ferrotorch_core::masked_equal` GPU branch.
    fn ne_scalar_mask(
        &self,
        _input: &GpuBufferHandle,
        _value: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "ne_scalar_mask",
        })
    }
}

static GPU_BACKEND: OnceLock<Box<dyn GpuBackend>> = OnceLock::new();

/// Register a GPU backend. Called once by the GPU crate on init.
pub fn register_gpu_backend(backend: Box<dyn GpuBackend>) -> Result<(), Box<dyn GpuBackend>> {
    GPU_BACKEND.set(backend)
}

/// Get the registered GPU backend, if any.
pub fn gpu_backend() -> Option<&'static dyn GpuBackend> {
    GPU_BACKEND.get().map(|b| b.as_ref())
}

/// Returns `true` if a GPU backend has been registered.
pub fn has_gpu_backend() -> bool {
    GPU_BACKEND.get().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_buffer_handle() {
        // Inner type is arbitrary here (this exercises the type-erasure
        // mechanics, not dtype dispatch); tag with F32 as a placeholder.
        let handle = GpuBufferHandle::new(Box::new(42u64), 0, 100, DType::F32);
        assert_eq!(handle.device_ordinal(), 0);
        assert_eq!(handle.len(), 100);
        assert!(!handle.is_empty());
        assert_eq!(handle.downcast_ref::<u64>(), Some(&42));
        assert_eq!(handle.dtype(), DType::F32);
    }

    #[test]
    fn test_gpu_buffer_handle_debug() {
        let handle = GpuBufferHandle::new(Box::new(()), 1, 50, DType::F64);
        let s = format!("{handle:?}");
        assert!(s.contains("device: 1"));
        // The Debug impl now surfaces the dtype tag.
        assert!(s.contains("dtype"));
    }
}
