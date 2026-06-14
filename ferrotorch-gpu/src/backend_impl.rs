//! CUDA implementation of the [`GpuBackend`] trait from ferrotorch-core.
//!
//! This module bridges the existing GPU operations (`gpu_add`, `gpu_matmul_f32`,
//! etc.) to the type-erased [`GpuBackend`] dispatch interface, enabling
//! ferrotorch-core to call GPU operations without depending on this crate
//! directly.
//!
//! # Initialization
//!
//! Call [`init_cuda_backend`] once at startup (typically via `ferrotorch::init()`).
//! This creates a [`CudaBackendImpl`], initializes CUDA device 0, and registers
//! it with [`ferrotorch_core::gpu_dispatch::register_gpu_backend`].
//!
//! ## REQ status (per `.design/ferrotorch-gpu/backend_impl.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream
//! cites) live in the design doc; this synopsis is a one-line summary per
//! REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`CudaBackendImpl` + `new`) | SHIPPED | `pub struct CudaBackendImpl in backend_impl.rs` + `pub fn new`; consumer `pub fn init_cuda_backend in backend_impl.rs` constructs it; `ferrotorch/examples/ferrotorch_bench.rs` calls `ferrotorch_gpu::init_cuda_backend()` |
//! | REQ-2 (wrap/unwrap helpers) | SHIPPED | 12 `wrap_* / unwrap_*` helpers on `impl CudaBackendImpl in backend_impl.rs`; consumer 348+ call sites across the `impl GpuBackend` block (every trait method body) |
//! | REQ-3 (cached cuSPARSE/cuSPARSELt handles) | SHIPPED | `cusparse_handle: OnceLock<CusparseHandle>` + `cusparselt_handle: OnceLock<CusparseLtHandle>` fields in `backend_impl.rs` with lazy accessor methods; consumer SpMM / 2:4-sparse matmul trait method bodies in `impl GpuBackend` block |
//! | REQ-4 (`init_cuda_backend`) | SHIPPED | `pub fn init_cuda_backend in backend_impl.rs`; consumer `ferrotorch/examples/ferrotorch_bench.rs`; re-exported at `lib.rs` |
//! | REQ-5 (`get_cuda_device`) | SHIPPED | `pub fn get_cuda_device in backend_impl.rs`; consumer re-exported at `lib.rs`; the downcast-via-`as_any` pattern is the canonical accessor for shared `GpuDevice` from any registered-backend caller |
//! | REQ-6 (`impl GpuBackend`) | SHIPPED | `impl GpuBackend for CudaBackendImpl in backend_impl.rs` with 348+ method bodies forwarding to `crate::kernels::*` / siblings; consumer ferrotorch-core's `gpu_dispatch::gpu_backend()` returns the registered global `&dyn GpuBackend`, every CUDA-aware tensor op dispatches through it |
//! | REQ-7 (`map_gpu_err`) | SHIPPED | `fn map_gpu_err in backend_impl.rs`; consumer every `.map_err(Self::map_gpu_err)?` site in trait-method bodies (hundreds) |
//! | REQ-8 (`gather_or_select`) | SHIPPED | `fn gather_or_select in backend_impl.rs` IS the `GpuBackend::gather_or_select` trait method body; consumer ferrotorch-core's `Tensor::index_select / Tensor::gather` dispatch through it via the trait when source is CUDA-resident |

use std::sync::{Arc, OnceLock};

use ferrotorch_core::dtype::DType;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::gpu_dispatch::{GpuBackend, GpuBufferHandle, GpuRngState};

use crate::buffer::CudaBuffer;
#[cfg(all(feature = "cuda", feature = "cusparselt"))]
use crate::cusparselt::CusparseLtHandle;
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use crate::sparse::CusparseHandle;

// ---------------------------------------------------------------------------
// CudaBackendImpl
// ---------------------------------------------------------------------------

/// CUDA implementation of the [`GpuBackend`] trait.
///
/// Holds one or more [`GpuDevice`] handles (currently device 0 only) and
/// delegates every trait method to the corresponding function in
/// [`crate::kernels`], [`crate::blas`], or [`crate::transfer`].
pub struct CudaBackendImpl {
    devices: Vec<Arc<GpuDevice>>,
    /// Lazily-initialised cuSPARSE handle, cached for `SparseTensor::spmm`.
    /// One handle is sufficient because all current devices share the
    /// primary CUDA context; the handle's stream is rebound per call via
    /// `cusparseSetStream`. Wrapped in `OnceLock` so the first SpMM pays
    /// the `cusparseCreate` cost and subsequent calls reuse the handle.
    #[cfg(feature = "cuda")]
    cusparse_handle: OnceLock<CusparseHandle>,
    /// Lazily-initialised cuSPARSELt handle, cached for the 2:4
    /// structured sparse matmul path. Pays the `cusparseLtInit` cost
    /// only on first use; subsequent matmuls reuse the handle. Only
    /// present when the `cusparselt` cargo feature is enabled.
    #[cfg(all(feature = "cuda", feature = "cusparselt"))]
    cusparselt_handle: OnceLock<CusparseLtHandle>,
}

impl CudaBackendImpl {
    /// Create a new CUDA backend, initializing device 0.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if CUDA initialization fails
    /// (e.g. no GPU available, driver not loaded).
    pub fn new() -> FerrotorchResult<Self> {
        let device = Arc::new(
            GpuDevice::new(0).map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("CUDA init failed: {e}"),
            })?,
        );
        Ok(Self {
            devices: vec![device],
            #[cfg(feature = "cuda")]
            cusparse_handle: OnceLock::new(),
            #[cfg(all(feature = "cuda", feature = "cusparselt"))]
            cusparselt_handle: OnceLock::new(),
        })
    }

    /// Get or lazily create the cached cuSPARSELt handle. The first
    /// call pays `cusparseLtInit`; subsequent calls reuse the handle.
    #[cfg(all(feature = "cuda", feature = "cusparselt"))]
    fn cusparselt(&self) -> FerrotorchResult<&CusparseLtHandle> {
        if let Some(h) = self.cusparselt_handle.get() {
            return Ok(h);
        }
        let new_handle = CusparseLtHandle::new().map_err(Self::map_gpu_err)?;
        let _ = self.cusparselt_handle.set(new_handle);
        self.cusparselt_handle
            .get()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "cuSPARSELt handle slot empty after init".into(),
            })
    }

    /// Get or lazily create the cached cuSPARSE handle. The first call
    /// pays the `cusparseCreate` cost; subsequent calls reuse the handle.
    /// The stream is rebound per SpMM via `cusparseSetStream`.
    #[cfg(feature = "cuda")]
    fn cusparse(&self) -> FerrotorchResult<&CusparseHandle> {
        if let Some(h) = self.cusparse_handle.get() {
            return Ok(h);
        }
        let new_handle = CusparseHandle::new().map_err(Self::map_gpu_err)?;
        // OnceLock::get_or_init can't return Err, so we set+ignore the
        // race where a competing thread set first; either resulting handle
        // is valid for the same context.
        let _ = self.cusparse_handle.set(new_handle);
        self.cusparse_handle
            .get()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "cuSPARSE handle slot empty after init".into(),
            })
    }

    /// Get the device for ordinal 0 (the default device).
    pub fn default_device(&self) -> FerrotorchResult<&Arc<GpuDevice>> {
        self.device(0)
    }

    /// Look up a device by ordinal.
    fn device(&self, ordinal: usize) -> FerrotorchResult<&Arc<GpuDevice>> {
        self.devices
            .get(ordinal)
            .ok_or(FerrotorchError::InvalidArgument {
                message: format!("CUDA device {ordinal} not available"),
            })
    }

    /// Wrap a `CudaBuffer<f32>` into a type-erased [`GpuBufferHandle`],
    /// tagging it `DType::F32` (the authoritative element-type tag).
    fn wrap_buffer(buf: CudaBuffer<f32>, ordinal: usize) -> GpuBufferHandle {
        let len = buf.len();
        GpuBufferHandle::new(Box::new(buf), ordinal, len, DType::F32)
    }

    /// Wrap a `CudaBuffer<f64>` into a type-erased [`GpuBufferHandle`],
    /// tagging it `DType::F64`.
    fn wrap_buffer_f64(buf: CudaBuffer<f64>, ordinal: usize) -> GpuBufferHandle {
        let len = buf.len();
        GpuBufferHandle::new(Box::new(buf), ordinal, len, DType::F64)
    }

    /// Extract a `&CudaBuffer<f32>` from a [`GpuBufferHandle`].
    ///
    /// The `dtype` tag is the fast, authoritative check (PyTorch parity); the
    /// `downcast_ref` is the safety net that catches a tag/storage mismatch.
    /// In later phases this tag check is what stops an i32 handle (also 4
    /// bytes) from being silently read as f32.
    fn unwrap_buffer(handle: &GpuBufferHandle) -> FerrotorchResult<&CudaBuffer<f32>> {
        if handle.dtype() != DType::F32 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("expected F32 buffer, handle is tagged {}", handle.dtype()),
            });
        }
        handle
            .downcast_ref::<CudaBuffer<f32>>()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "GPU handle does not contain a CudaBuffer<f32>".into(),
            })
    }

    /// Extract a `&mut CudaBuffer<f32>` from a [`GpuBufferHandle`].
    fn unwrap_buffer_mut(handle: &mut GpuBufferHandle) -> FerrotorchResult<&mut CudaBuffer<f32>> {
        if handle.dtype() != DType::F32 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("expected F32 buffer, handle is tagged {}", handle.dtype()),
            });
        }
        handle
            .downcast_mut::<CudaBuffer<f32>>()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "GPU handle does not contain a CudaBuffer<f32>".into(),
            })
    }

    /// Extract a `&mut CudaBuffer<f64>` from a [`GpuBufferHandle`].
    fn unwrap_buffer_f64_mut(
        handle: &mut GpuBufferHandle,
    ) -> FerrotorchResult<&mut CudaBuffer<f64>> {
        if handle.dtype() != DType::F64 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("expected F64 buffer, handle is tagged {}", handle.dtype()),
            });
        }
        handle
            .downcast_mut::<CudaBuffer<f64>>()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "GPU handle does not contain a CudaBuffer<f64>".into(),
            })
    }

    /// Extract a `&CudaBuffer<f64>` from a [`GpuBufferHandle`].
    fn unwrap_buffer_f64(handle: &GpuBufferHandle) -> FerrotorchResult<&CudaBuffer<f64>> {
        if handle.dtype() != DType::F64 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("expected F64 buffer, handle is tagged {}", handle.dtype()),
            });
        }
        handle
            .downcast_ref::<CudaBuffer<f64>>()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "GPU handle does not contain a CudaBuffer<f64>".into(),
            })
    }

    /// Wrap a `CudaSlice<u16>` (bf16 bit-pattern storage) into a type-erased
    /// [`GpuBufferHandle`]. bf16 buffers are stored as `CudaSlice<u16>` rather
    /// than `CudaBuffer<T>` so they match the input type of every `*_bf16`
    /// PTX kernel in [`crate::bf16`] and `gpu_matmul_bf16_bf16` in
    /// [`crate::blas`] without an extra unwrap step.
    #[cfg(feature = "cuda")]
    fn wrap_buffer_bf16(buf: cudarc::driver::CudaSlice<u16>, ordinal: usize) -> GpuBufferHandle {
        let len = buf.len();
        // bf16 storage is a `CudaSlice<u16>` bit pattern; the `DType::BF16` tag
        // is what tells it apart from a (future) f16 `CudaSlice<u16>` — same
        // byte width, distinguished only by the tag (PyTorch parity).
        GpuBufferHandle::new(Box::new(buf), ordinal, len, DType::BF16)
    }

    /// Extract a `&CudaSlice<u16>` (bf16 bit-pattern storage) from a
    /// [`GpuBufferHandle`].
    #[cfg(feature = "cuda")]
    fn unwrap_buffer_bf16(
        handle: &GpuBufferHandle,
    ) -> FerrotorchResult<&cudarc::driver::CudaSlice<u16>> {
        // Tag check first: bf16 and (future) f16 both store as `CudaSlice<u16>`,
        // so the `downcast_ref` alone cannot tell them apart. The `DType::BF16`
        // tag is the authoritative discriminator (PyTorch parity).
        if handle.dtype() != DType::BF16 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("expected BF16 buffer, handle is tagged {}", handle.dtype()),
            });
        }
        handle
            .downcast_ref::<cudarc::driver::CudaSlice<u16>>()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "GPU handle does not contain a CudaSlice<u16> (bf16)".into(),
            })
    }

    /// Wrap a `CudaSlice<u16>` (IEEE f16 bit-pattern storage) into a
    /// type-erased [`GpuBufferHandle`] tagged `DType::F16`.
    ///
    /// f16 and bf16 share `CudaSlice<u16>` storage; the `DType::F16` tag is
    /// the *only* thing that disambiguates them (crosslink #1185 Phase 1).
    #[cfg(feature = "cuda")]
    fn wrap_buffer_f16(buf: cudarc::driver::CudaSlice<u16>, ordinal: usize) -> GpuBufferHandle {
        let len = buf.len();
        GpuBufferHandle::new(Box::new(buf), ordinal, len, DType::F16)
    }

    /// Extract a `&CudaSlice<u16>` (IEEE f16 bit-pattern storage) from a
    /// [`GpuBufferHandle`], asserting the `DType::F16` tag first.
    ///
    /// MANDATORY disambiguation: f16 and bf16 both downcast to
    /// `CudaSlice<u16>`, so the `downcast_ref` alone cannot tell them apart.
    /// The tag check rejects a BF16-tagged handle here (and
    /// [`Self::unwrap_buffer_bf16`] rejects an F16-tagged handle), preventing
    /// an f16 buffer from being silently fed to a bf16 kernel (PyTorch
    /// parity — `ScalarType` is authoritative, not storage width).
    #[cfg(feature = "cuda")]
    fn unwrap_buffer_f16(
        handle: &GpuBufferHandle,
    ) -> FerrotorchResult<&cudarc::driver::CudaSlice<u16>> {
        if handle.dtype() != DType::F16 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("expected F16 buffer, handle is tagged {}", handle.dtype()),
            });
        }
        handle
            .downcast_ref::<cudarc::driver::CudaSlice<u16>>()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "GPU handle does not contain a CudaSlice<u16> (f16)".into(),
            })
    }

    /// Wrap a `CudaBuffer<i32>` into a type-erased [`GpuBufferHandle`] tagged
    /// `DType::I32`.
    ///
    /// Integer device storage (GPU dtype-parity epic, crosslink #1185 Phase
    /// 2a). i32 has cudarc `DeviceRepr` natively, so — unlike bf16/f16 which
    /// stash a bit pattern in `CudaSlice<u16>` — the buffer holds a real
    /// `CudaBuffer<i32>`. The `DType::I32` tag is what stops it being read as
    /// f32 (same 4-byte width). No integer compute kernels exist yet (Phase
    /// 2b); this is storage/transport only.
    fn wrap_buffer_i32(buf: CudaBuffer<i32>, ordinal: usize) -> GpuBufferHandle {
        let len = buf.len();
        GpuBufferHandle::new(Box::new(buf), ordinal, len, DType::I32)
    }

    /// Extract a `&CudaBuffer<i32>` from a [`GpuBufferHandle`], asserting the
    /// `DType::I32` tag first (PyTorch parity — the ScalarType tag is
    /// authoritative; the `downcast_ref` is the safety net).
    fn unwrap_buffer_i32(handle: &GpuBufferHandle) -> FerrotorchResult<&CudaBuffer<i32>> {
        if handle.dtype() != DType::I32 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("expected I32 buffer, handle is tagged {}", handle.dtype()),
            });
        }
        handle
            .downcast_ref::<CudaBuffer<i32>>()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "GPU handle does not contain a CudaBuffer<i32>".into(),
            })
    }

    /// Wrap a `CudaBuffer<i64>` into a type-erased [`GpuBufferHandle`] tagged
    /// `DType::I64` (crosslink #1185 Phase 2a). i64 has cudarc `DeviceRepr`
    /// natively. The `DType::I64` tag is what stops it being read as f64 (same
    /// 8-byte width).
    fn wrap_buffer_i64(buf: CudaBuffer<i64>, ordinal: usize) -> GpuBufferHandle {
        let len = buf.len();
        GpuBufferHandle::new(Box::new(buf), ordinal, len, DType::I64)
    }

    /// Extract a `&CudaBuffer<i64>` from a [`GpuBufferHandle`], asserting the
    /// `DType::I64` tag first.
    fn unwrap_buffer_i64(handle: &GpuBufferHandle) -> FerrotorchResult<&CudaBuffer<i64>> {
        if handle.dtype() != DType::I64 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("expected I64 buffer, handle is tagged {}", handle.dtype()),
            });
        }
        handle
            .downcast_ref::<CudaBuffer<i64>>()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "GPU handle does not contain a CudaBuffer<i64>".into(),
            })
    }

    /// Wrap a `CudaBuffer<u8>` (boolean storage) into a type-erased
    /// [`GpuBufferHandle`] tagged `DType::Bool` (crosslink #1185 Phase 3a).
    ///
    /// A `bool` is one byte holding 0 or 1; on device it is stored as a native
    /// `CudaBuffer<u8>` (u8 has cudarc `DeviceRepr`). The `DType::Bool` tag is
    /// what stops a bool buffer being read as a (future) i8/u8 integer — the
    /// same role the F16/BF16 tags play for the two 2-byte float types.
    fn wrap_buffer_bool(buf: CudaBuffer<u8>, ordinal: usize) -> GpuBufferHandle {
        let len = buf.len();
        GpuBufferHandle::new(Box::new(buf), ordinal, len, DType::Bool)
    }

    /// Extract a `&CudaBuffer<u8>` (boolean storage) from a [`GpuBufferHandle`],
    /// asserting the `DType::Bool` tag first (PyTorch parity — the ScalarType
    /// tag is authoritative; the `downcast_ref` is the safety net).
    fn unwrap_buffer_bool(handle: &GpuBufferHandle) -> FerrotorchResult<&CudaBuffer<u8>> {
        if handle.dtype() != DType::Bool {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("expected Bool buffer, handle is tagged {}", handle.dtype()),
            });
        }
        handle
            .downcast_ref::<CudaBuffer<u8>>()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "GPU handle does not contain a CudaBuffer<u8> (bool)".into(),
            })
    }

    /// Wrap a raw `CudaSlice<u8>` (the native bool kernel result type) into a
    /// non-pooled [`CudaBuffer<u8>`] and then a `DType::Bool`-tagged
    /// [`GpuBufferHandle`]. Mirror of `wrap_slice_i32` for the
    /// [`crate::bool_kernels`] (#1185 Phase 3b) output, which produces a bare
    /// `CudaSlice<u8>` rather than a `CudaBuffer`.
    #[cfg(feature = "cuda")]
    fn wrap_slice_bool(slice: cudarc::driver::CudaSlice<u8>, ordinal: usize) -> GpuBufferHandle {
        let len = slice.len();
        let buf = CudaBuffer {
            data: Some(slice),
            len,
            alloc_len: len,
            device_ordinal: ordinal,
            pool_fn: None,
        };
        Self::wrap_buffer_bool(buf, ordinal)
    }

    /// Wrap a raw `CudaSlice<i32>` (the native integer kernel result type)
    /// into a non-pooled [`CudaBuffer<i32>`] and then a `DType::I32`-tagged
    /// [`GpuBufferHandle`]. Mirror of `wrap_buffer_i32` for the
    /// `crate::int_kernels` (#1185 Phase 2b) output, which produces a bare
    /// `CudaSlice` rather than a `CudaBuffer`.
    #[cfg(feature = "cuda")]
    fn wrap_slice_i32(slice: cudarc::driver::CudaSlice<i32>, ordinal: usize) -> GpuBufferHandle {
        let len = slice.len();
        let buf = CudaBuffer {
            data: Some(slice),
            len,
            alloc_len: len,
            device_ordinal: ordinal,
            pool_fn: None,
        };
        Self::wrap_buffer_i32(buf, ordinal)
    }

    /// Wrap a raw `CudaSlice<i64>` into a `DType::I64`-tagged
    /// [`GpuBufferHandle`] (#1185 Phase 2b). Counterpart of
    /// [`Self::wrap_slice_i32`].
    #[cfg(feature = "cuda")]
    fn wrap_slice_i64(slice: cudarc::driver::CudaSlice<i64>, ordinal: usize) -> GpuBufferHandle {
        let len = slice.len();
        let buf = CudaBuffer {
            data: Some(slice),
            len,
            alloc_len: len,
            device_ordinal: ordinal,
            pool_fn: None,
        };
        Self::wrap_buffer_i64(buf, ordinal)
    }

    /// Wrap a raw `CudaSlice<f32>` (a Phase-2c cast/gather kernel result) into
    /// a non-pooled `CudaBuffer<f32>` then a `DType::F32`-tagged handle.
    #[cfg(feature = "cuda")]
    fn wrap_slice_f32(slice: cudarc::driver::CudaSlice<f32>, ordinal: usize) -> GpuBufferHandle {
        let len = slice.len();
        let buf = CudaBuffer {
            data: Some(slice),
            len,
            alloc_len: len,
            device_ordinal: ordinal,
            pool_fn: None,
        };
        Self::wrap_buffer(buf, ordinal)
    }

    /// Wrap a raw `CudaSlice<f64>` into a `DType::F64`-tagged handle.
    #[cfg(feature = "cuda")]
    fn wrap_slice_f64(slice: cudarc::driver::CudaSlice<f64>, ordinal: usize) -> GpuBufferHandle {
        let len = slice.len();
        let buf = CudaBuffer {
            data: Some(slice),
            len,
            alloc_len: len,
            device_ordinal: ordinal,
            pool_fn: None,
        };
        Self::wrap_buffer_f64(buf, ordinal)
    }

    /// Shared dispatch for `index_select` / `gather` with a GPU-resident
    /// integer index (crosslink #1185 Phase 2c). `is_gather=false` selects
    /// rows by `index[i]`; `is_gather=true` reads `index[t]` per output element.
    /// Dispatches on `(src.dtype(), index.dtype())`; result keeps `src`'s dtype.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn gather_or_select(
        &self,
        src: &GpuBufferHandle,
        index: &GpuBufferHandle,
        outer: usize,
        in_dim: usize,
        out_dim: usize,
        inner: usize,
        is_gather: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        use crate::gather_int as gi;
        let dev = self.device(src.device_ordinal())?;
        let ord = src.device_ordinal();
        let op = if is_gather { "gather" } else { "index_select" };
        match index.dtype() {
            DType::I32 | DType::I64 => {}
            other => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("{op}: index dtype must be I32/I64, got {other}"),
                });
            }
        }
        let i32idx = index.dtype() == DType::I32;
        // Macro: pick the right (gather|isel)_<vty>_<ity> entry, run it, wrap.
        macro_rules! run {
            ($val:expr, $g32:path, $g64:path, $s32:path, $s64:path, $wrap:expr) => {{
                let r = if is_gather {
                    if i32idx {
                        $g32(
                            $val,
                            Self::unwrap_buffer_i32(index)?.inner(),
                            outer,
                            in_dim,
                            out_dim,
                            inner,
                            dev,
                        )
                    } else {
                        $g64(
                            $val,
                            Self::unwrap_buffer_i64(index)?.inner(),
                            outer,
                            in_dim,
                            out_dim,
                            inner,
                            dev,
                        )
                    }
                } else if i32idx {
                    $s32(
                        $val,
                        Self::unwrap_buffer_i32(index)?.inner(),
                        outer,
                        in_dim,
                        out_dim,
                        inner,
                        dev,
                    )
                } else {
                    $s64(
                        $val,
                        Self::unwrap_buffer_i64(index)?.inner(),
                        outer,
                        in_dim,
                        out_dim,
                        inner,
                        dev,
                    )
                }
                .map_err(Self::map_gpu_err)?;
                Ok($wrap(r, ord))
            }};
        }
        match src.dtype() {
            DType::F32 => run!(
                Self::unwrap_buffer(src)?.inner(),
                gi::gather_f32_i32,
                gi::gather_f32_i64,
                gi::isel_f32_i32,
                gi::isel_f32_i64,
                Self::wrap_slice_f32
            ),
            DType::F64 => run!(
                Self::unwrap_buffer_f64(src)?.inner(),
                gi::gather_f64_i32,
                gi::gather_f64_i64,
                gi::isel_f64_i32,
                gi::isel_f64_i64,
                Self::wrap_slice_f64
            ),
            DType::I32 => run!(
                Self::unwrap_buffer_i32(src)?.inner(),
                gi::gather_i32_i32,
                gi::gather_i32_i64,
                gi::isel_i32_i32,
                gi::isel_i32_i64,
                Self::wrap_slice_i32
            ),
            DType::I64 => run!(
                Self::unwrap_buffer_i64(src)?.inner(),
                gi::gather_i64_i32,
                gi::gather_i64_i64,
                gi::isel_i64_i32,
                gi::isel_i64_i64,
                Self::wrap_slice_i64
            ),
            DType::F16 => run!(
                Self::unwrap_buffer_f16(src)?,
                gi::gather_u16_i32,
                gi::gather_u16_i64,
                gi::isel_u16_i32,
                gi::isel_u16_i64,
                Self::wrap_buffer_f16
            ),
            DType::BF16 => run!(
                Self::unwrap_buffer_bf16(src)?,
                gi::gather_u16_i32,
                gi::gather_u16_i64,
                gi::isel_u16_i32,
                gi::isel_u16_i64,
                Self::wrap_buffer_bf16
            ),
            other => Err(FerrotorchError::InvalidArgument {
                message: format!("{op}: unsupported value dtype {other}"),
            }),
        }
    }

    fn gather_nd_metadata(
        src: &GpuBufferHandle,
        index: &GpuBufferHandle,
        input_shape: &[usize],
        index_shape: &[usize],
        dim: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>)> {
        if input_shape.len() != index_shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "gather_intidx_nd: rank mismatch input rank {} index rank {}",
                    input_shape.len(),
                    index_shape.len()
                ),
            });
        }
        if dim >= input_shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "gather_intidx_nd: dim {dim} out of range for rank {}",
                    input_shape.len()
                ),
            });
        }
        let input_numel = input_shape.iter().try_fold(1usize, |acc, &d| {
            acc.checked_mul(d).ok_or(FerrotorchError::InvalidArgument {
                message: "gather_intidx_nd: input shape product overflow".to_string(),
            })
        })?;
        let index_numel = index_shape.iter().try_fold(1usize, |acc, &d| {
            acc.checked_mul(d).ok_or(FerrotorchError::InvalidArgument {
                message: "gather_intidx_nd: index shape product overflow".to_string(),
            })
        })?;
        if input_numel != src.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "gather_intidx_nd: input shape product {input_numel} != buffer len {}",
                    src.len()
                ),
            });
        }
        if index_numel != index.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "gather_intidx_nd: index shape product {index_numel} != buffer len {}",
                    index.len()
                ),
            });
        }

        let mut input_strides = vec![0u32; input_shape.len()];
        let mut stride = 1usize;
        for axis in (0..input_shape.len()).rev() {
            input_strides[axis] =
                u32::try_from(stride).map_err(|_| FerrotorchError::InvalidArgument {
                    message: format!(
                        "gather_intidx_nd: input stride {stride} exceeds u32 kernel limit"
                    ),
                })?;
            stride =
                stride
                    .checked_mul(input_shape[axis])
                    .ok_or(FerrotorchError::InvalidArgument {
                        message: "gather_intidx_nd: input stride overflow".to_string(),
                    })?;
        }
        let index_dims: Vec<u32> = index_shape
            .iter()
            .map(|&d| {
                u32::try_from(d).map_err(|_| FerrotorchError::InvalidArgument {
                    message: format!(
                        "gather_intidx_nd: index dimension {d} exceeds u32 kernel limit"
                    ),
                })
            })
            .collect::<Result<_, _>>()?;
        Ok((input_strides, index_dims))
    }

    /// Convert a [`crate::error::GpuError`] into a [`FerrotorchError`].
    fn map_gpu_err(e: crate::error::GpuError) -> FerrotorchError {
        FerrotorchError::InvalidArgument {
            message: format!("{e}"),
        }
    }
}

// ---------------------------------------------------------------------------
// GpuBackend implementation
// ---------------------------------------------------------------------------

impl GpuBackend for CudaBackendImpl {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn raw_device_ptr(&self, handle: &GpuBufferHandle) -> *const std::ffi::c_void {
        use cudarc::driver::DevicePtr;
        let dev = match self.device(handle.device_ordinal()) {
            Ok(d) => d,
            Err(_) => return std::ptr::null(),
        };
        let stream = dev.stream();
        if let Ok(buf) = Self::unwrap_buffer(handle) {
            let (ptr, _sync) = buf.inner().device_ptr(&stream);
            ptr as *const std::ffi::c_void
        } else if let Ok(buf) = Self::unwrap_buffer_f64(handle) {
            let (ptr, _sync) = buf.inner().device_ptr(&stream);
            ptr as *const std::ffi::c_void
        } else if let Ok(slice) = Self::unwrap_buffer_bf16(handle) {
            // bf16 storage: `CudaSlice<u16>` (u16 holds bf16 bit pattern).
            let (ptr, _sync) = slice.device_ptr(&stream);
            ptr as *const std::ffi::c_void
        } else if let Ok(slice) = Self::unwrap_buffer_f16(handle) {
            // f16 storage: `CudaSlice<u16>` (u16 holds IEEE f16 bit pattern),
            // distinguished from bf16 only by the `DType::F16` tag.
            let (ptr, _sync) = slice.device_ptr(&stream);
            ptr as *const std::ffi::c_void
        } else if let Ok(buf) = Self::unwrap_buffer_i32(handle) {
            // i32 storage: real `CudaBuffer<i32>` (crosslink #1185 Phase 2a).
            let (ptr, _sync) = buf.inner().device_ptr(&stream);
            ptr as *const std::ffi::c_void
        } else if let Ok(buf) = Self::unwrap_buffer_i64(handle) {
            // i64 storage: real `CudaBuffer<i64>` (crosslink #1185 Phase 2a).
            let (ptr, _sync) = buf.inner().device_ptr(&stream);
            ptr as *const std::ffi::c_void
        } else if let Ok(buf) = Self::unwrap_buffer_bool(handle) {
            // bool storage: real `CudaBuffer<u8>` (crosslink #1185 Phase 3a).
            let (ptr, _sync) = buf.inner().device_ptr(&stream);
            ptr as *const std::ffi::c_void
        } else {
            std::ptr::null()
        }
    }

    fn raw_device_ptr_mut(&self, handle: &mut GpuBufferHandle) -> *mut std::ffi::c_void {
        use cudarc::driver::DevicePtrMut;
        let ordinal = handle.device_ordinal();
        let dev = match self.device(ordinal) {
            Ok(d) => d,
            Err(_) => return std::ptr::null_mut(),
        };
        let stream = dev.stream();
        if let Some(buf) = handle.downcast_mut::<CudaBuffer<f32>>() {
            let (ptr, _sync) = buf.inner_mut().device_ptr_mut(&stream);
            ptr as *mut std::ffi::c_void
        } else if let Some(buf) = handle.downcast_mut::<CudaBuffer<f64>>() {
            let (ptr, _sync) = buf.inner_mut().device_ptr_mut(&stream);
            ptr as *mut std::ffi::c_void
        } else if let Some(slice) = handle.downcast_mut::<cudarc::driver::CudaSlice<u16>>() {
            let (ptr, _sync) = slice.device_ptr_mut(&stream);
            ptr as *mut std::ffi::c_void
        } else if let Some(buf) = handle.downcast_mut::<CudaBuffer<i32>>() {
            // i32 storage (crosslink #1185 Phase 2a).
            let (ptr, _sync) = buf.inner_mut().device_ptr_mut(&stream);
            ptr as *mut std::ffi::c_void
        } else if let Some(buf) = handle.downcast_mut::<CudaBuffer<i64>>() {
            // i64 storage (crosslink #1185 Phase 2a).
            let (ptr, _sync) = buf.inner_mut().device_ptr_mut(&stream);
            ptr as *mut std::ffi::c_void
        } else if let Some(buf) = handle.downcast_mut::<CudaBuffer<u8>>() {
            // bool storage: `CudaBuffer<u8>` (crosslink #1185 Phase 3a).
            let (ptr, _sync) = buf.inner_mut().device_ptr_mut(&stream);
            ptr as *mut std::ffi::c_void
        } else {
            std::ptr::null_mut()
        }
    }

    fn buffer_elem_size(&self, handle: &GpuBufferHandle) -> usize {
        // PyTorch parity: byte width is derived from the authoritative dtype
        // tag, not probed from the erased concrete buffer type.
        handle.dtype().size_of()
    }

    fn cpu_to_gpu(
        &self,
        data: &[u8],
        dtype: DType,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(device)?;
        match dtype {
            DType::F32 => {
                let count = data.len() / 4;
                // SAFETY:
                // - The caller (ferrotorch-core) guarantees that `data` is the
                //   byte serialisation of a contiguous `&[f32]`; this is the
                //   contract of the `cpu_to_gpu` trait method whose signature
                //   accepts `&[u8]` + `elem_size` as a type-erased façade
                //   across f32/f64 (see trait definition in
                //   ferrotorch-core/src/gpu_dispatch.rs:146-151).
                // - The `elem_size == 4` arm is only entered when the upstream
                //   caller asserted f32 layout. The canonical caller pattern
                //   (ferrotorch-core/src/storage.rs:117-133, `on_device`)
                //   originates a `Vec<T>` and forwards `size_of::<T>()` as
                //   `elem_size`; reaching this arm means `T == f32` upstream.
                // - Alignment: f32 has size 4 and align 4 (Rust reference,
                //   primitive data layout). The source `Vec<f32>` allocation
                //   is 4-byte-aligned; that alignment propagates through
                //   `data.as_ptr()` because the `&[u8]` view of an f32 slice
                //   is always at least 4-byte-aligned at its start.
                // - Length: `count = data.len() / 4` is exact when the caller
                //   honours the contract (`data.len() == count * size_of::<f32>()`);
                //   any remainder is a caller-contract violation, not a bug
                //   we must defend against here.
                // - Provenance: `slice::from_raw_parts` reads `count * 4` bytes,
                //   which equals `data.len()` exactly, so the new slice spans
                //   no memory beyond the input.
                // - Lifetime: the reinterpreted `&[f32]` is bound by `data`
                //   (a `&[u8]` parameter borrowed for the duration of this
                //   call). The slice is consumed by `cpu_to_gpu` on the next
                //   line before this stack frame returns; no dangling
                //   reference can escape.
                // - No `&mut` aliases: `data: &[u8]` is a shared borrow, so
                //   no concurrent `&mut [f32]` to the same allocation can
                //   exist while this call is active.
                let f32_data: &[f32] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, count) };
                let buf = crate::transfer::cpu_to_gpu(f32_data, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer(buf, device))
            }
            DType::F64 => {
                let count = data.len() / 8;
                // SAFETY:
                // - The caller (ferrotorch-core) guarantees that `data` is the
                //   byte serialisation of a contiguous `&[f64]`; this is the
                //   `elem_size == 8` arm's precondition documented at the
                //   trait method (ferrotorch-core/src/gpu_dispatch.rs:146-151).
                //   Canonical caller path: ferrotorch-core/src/storage.rs:117-133
                //   forwards `size_of::<T>()`; reaching this arm means
                //   `T == f64` upstream.
                // - Alignment: f64 has size 8 and align 8 (Rust reference,
                //   primitive data layout). The source `Vec<f64>` allocation
                //   guarantees 8-byte alignment which propagates to
                //   `data.as_ptr()` for the byte view of an f64 slice.
                // - Length: `count = data.len() / 8` is exact when the caller
                //   honours the contract; remainder bytes would be a
                //   caller-contract violation upstream of this site.
                // - Provenance: `slice::from_raw_parts` reads `count * 8`
                //   bytes which equals `data.len()` exactly, so the new
                //   slice covers exactly the byte range of `data` with no
                //   overrun.
                // - Lifetime: the reinterpreted `&[f64]` is bounded by `data`
                //   and consumed by `cpu_to_gpu` on the next line, never
                //   escaping this stack frame.
                // - No `&mut` aliases: shared `&[u8]` input rules out any
                //   concurrent `&mut [f64]` aliasing of the same allocation.
                let f64_data: &[f64] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, count) };
                let buf = crate::transfer::cpu_to_gpu(f64_data, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer_f64(buf, device))
            }
            DType::BF16 => {
                // bf16 bit patterns stored as u16. The handle
                // carries a raw `CudaSlice<u16>` (not a `CudaBuffer<T>` wrapper)
                // so every consumer in this crate — `softmax_bf16_f32`,
                // `add_bf16_f32`, `gpu_matmul_bf16_bf16`, all `*_bf16` PTX
                // kernels — downcasts to the same type without an extra
                // `unwrap_buffer_*` indirection.
                //
                // `clone_htod` requires `&Vec<T: DeviceRepr>` (cudarc 0.19 API);
                // `u16` satisfies DeviceRepr. We reinterpret the `&[u8]` bytes
                // as a `Vec<u16>` by reading pairs of bytes. `bytemuck::cast_slice`
                // is not available here, so we reinterpret via from_raw_parts
                // then collect into a Vec so the &Vec reference is valid for
                // clone_htod's lifetime requirement.
                let count = data.len() / 2;
                // SAFETY:
                // - Caller guarantees `data` is the byte view of a `Vec<u16>`
                //   with `data.len() == count * 2`.
                // - u16 has size 2 and align 2. The source `Vec<u16>` allocation
                //   is 2-byte-aligned; `data.as_ptr()` inherits that alignment.
                // - `slice::from_raw_parts` reads exactly `count * 2 == data.len()`
                //   bytes — no overrun.
                // - The `&[u16]` is immediately `.to_vec()`'d into an owned
                //   `Vec<u16>` before `data` is touched again; no aliasing.
                // - No `&mut` aliases: `data: &[u8]` is a shared borrow throughout.
                let u16_vec: Vec<u16> = unsafe {
                    let slice = std::slice::from_raw_parts(data.as_ptr() as *const u16, count);
                    slice.to_vec()
                };
                let slice = dev
                    .stream()
                    .clone_htod(&u16_vec)
                    .map_err(|e| Self::map_gpu_err(crate::error::GpuError::Driver(e)))?;
                Ok(Self::wrap_buffer_bf16(slice, device))
            }
            DType::F16 => {
                // IEEE float16 bit patterns stored as u16. Byte-identical
                // transport to the BF16 arm above — the ONLY difference is
                // the handle is tagged `DType::F16` (crosslink #1185 Phase 1),
                // which `unwrap_buffer_f16` asserts so an f16 buffer is never
                // fed to a bf16 kernel.
                let count = data.len() / 2;
                // SAFETY:
                // - Caller guarantees `data` is the byte view of a `Vec<u16>`
                //   (f16 `repr(transparent)` over u16) with
                //   `data.len() == count * 2`.
                // - u16 has size 2 and align 2; the source allocation is
                //   2-byte-aligned and `data.as_ptr()` inherits that.
                // - `from_raw_parts` reads exactly `count * 2 == data.len()`
                //   bytes — no overrun.
                // - The `&[u16]` is immediately `.to_vec()`'d into an owned
                //   `Vec<u16>` before `data` is touched again; no aliasing.
                // - No `&mut` aliases: `data: &[u8]` is a shared borrow.
                let u16_vec: Vec<u16> = unsafe {
                    let slice = std::slice::from_raw_parts(data.as_ptr() as *const u16, count);
                    slice.to_vec()
                };
                let slice = dev
                    .stream()
                    .clone_htod(&u16_vec)
                    .map_err(|e| Self::map_gpu_err(crate::error::GpuError::Driver(e)))?;
                Ok(Self::wrap_buffer_f16(slice, device))
            }
            DType::I32 => {
                // Integer device storage (crosslink #1185 Phase 2a). i32 has
                // cudarc `DeviceRepr` natively — no `CudaSlice<u16>` bit-pattern
                // trick (bf16/f16) — so we upload a real `&[i32]` and tag the
                // handle `DType::I32`. No integer compute kernels exist yet
                // (Phase 2b); this is storage/transport only.
                let count = data.len() / 4;
                // SAFETY:
                // - Caller (ferrotorch-core `IntTensor::to` / `TensorStorage::
                //   on_device`) guarantees `data` is the byte serialisation of a
                //   contiguous `&[i32]` with `data.len() == count * 4`.
                // - i32 has size 4 and align 4 (Rust reference, primitive data
                //   layout); the source `Vec<i32>` allocation is 4-byte-aligned
                //   and `data.as_ptr()` inherits that alignment.
                // - `from_raw_parts` reads exactly `count * 4 == data.len()`
                //   bytes — no overrun.
                // - Lifetime: the reinterpreted `&[i32]` is bounded by `data`
                //   and consumed by `transfer::cpu_to_gpu` before this frame
                //   returns; no dangling reference escapes.
                // - No `&mut` aliases: `data: &[u8]` is a shared borrow.
                let i32_data: &[i32] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i32, count) };
                let buf = crate::transfer::cpu_to_gpu(i32_data, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer_i32(buf, device))
            }
            DType::I64 => {
                // Integer device storage (crosslink #1185 Phase 2a). i64 has
                // cudarc `DeviceRepr` natively; upload a real `&[i64]` and tag
                // the handle `DType::I64`.
                let count = data.len() / 8;
                // SAFETY: same invariants as the I32 arm, with element width 8
                // (i64 size 8, align 8). `count = data.len() / 8` is exact when
                // the caller honours the `&[i64]` byte-serialisation contract;
                // `from_raw_parts` reads exactly `data.len()` bytes; the slice
                // is consumed by `transfer::cpu_to_gpu` before this frame
                // returns; `data` is a shared borrow so no `&mut [i64]` alias.
                let i64_data: &[i64] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i64, count) };
                let buf = crate::transfer::cpu_to_gpu(i64_data, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer_i64(buf, device))
            }
            DType::Bool => {
                // Boolean device storage (crosslink #1185 Phase 3a). A `bool` is
                // one byte holding 0 or 1; cudarc has no `DeviceRepr` for `bool`,
                // so we store it as a native `CudaBuffer<u8>` (u8 IS `DeviceRepr`)
                // and tag the handle `DType::Bool`. The `&[u8]` view of the
                // serialised `&[bool]` is byte-identical — no value translation.
                let count = data.len(); // 1 byte per bool
                // `data` is already `&[u8]`; the bytes are exactly the `bool`
                // values (each 0 or 1) reinterpreted, which `u8` reads back
                // identically. No `from_raw_parts` reinterpret is needed (unlike
                // the multi-byte arms): a bool slice IS a byte slice here.
                debug_assert_eq!(count, data.len());
                let buf = crate::transfer::cpu_to_gpu(data, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer_bool(buf, device))
            }
            // PyTorch parity (rust-gpu-discipline §3): unsupported (dtype,
            // CUDA) combinations return a structured error rather than a
            // silent CPU detour. Remaining integer dtypes land here
            // until their respective dtype-parity phases add real CUDA buffer
            // support.
            other => Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "cpu_to_gpu: dtype {other} not supported on CUDA \
                     (supported: F32, F64, BF16, F16, I32, I64, Bool)"
                ),
            }),
        }
    }

    fn cpu_to_gpu_pinned(
        &self,
        data: &[u8],
        dtype: DType,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(device)?;
        match dtype {
            DType::F32 => {
                let count = data.len() / 4;
                // SAFETY:
                // - The caller (ferrotorch-core) guarantees that `data` is the
                //   byte serialisation of a contiguous `&[f32]`; this is the
                //   contract of the `cpu_to_gpu_pinned` trait method whose
                //   signature accepts `&[u8]` + `elem_size` as a type-erased
                //   façade across f32/f64 (see trait definition in
                //   ferrotorch-core gpu_dispatch.rs).
                // - The `elem_size == 4` arm is only entered when the upstream
                //   caller asserted f32 layout. f32 has size 4 and align 4
                //   (Rust reference: <https://doc.rust-lang.org/reference/type-layout.html#primitive-data-layout>),
                //   so `count = data.len() / 4` is exact and `data.as_ptr()`
                //   inherits 4-byte alignment from any f32 source Vec.
                // - Lifetime: the reinterpreted `&[f32]` is bounded by `data`
                //   (a `&[u8]` parameter borrowed for the duration of this
                //   call). The slice is consumed by `cpu_to_gpu_pinned` on
                //   line 226 before this stack frame returns; no dangling
                //   reference can escape.
                // - Provenance: `slice::from_raw_parts` requires the entire
                //   `count * 4` byte range to be readable; that is exactly
                //   `data.len()` bytes (since `count = data.len() / 4`), so
                //   the new slice spans no memory beyond the input.
                // - No `&mut` aliases: `data` is a shared `&[u8]`, so no
                //   concurrent `&mut [f32]` to the same allocation can exist.
                let f32_data: &[f32] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, count) };
                let buf =
                    crate::transfer::cpu_to_gpu_pinned(f32_data, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer(buf, device))
            }
            DType::F64 => {
                let count = data.len() / 8;
                // SAFETY:
                // - The caller (ferrotorch-core) guarantees that `data` is the
                //   byte serialisation of a contiguous `&[f64]`; this is the
                //   `DType::F64` arm's precondition documented at the
                //   trait method (see gpu_dispatch.rs).
                // - f64 has size 8 and align 8 (Rust reference: primitive data
                //   layout). `count = data.len() / 8` is exact; the source
                //   `Vec<f64>` allocation guarantees 8-byte alignment which
                //   propagates to `data.as_ptr()`.
                // - Lifetime: the reinterpreted `&[f64]` is bounded by `data`
                //   and consumed by `cpu_to_gpu_pinned` on line 234, never
                //   escaping this stack frame.
                // - Provenance: `count * 8 == data.len()` so the new slice
                //   covers exactly the byte range of `data` with no overrun.
                // - No `&mut` aliases: shared `&[u8]` input rules out any
                //   concurrent `&mut [f64]` aliasing of the same allocation.
                let f64_data: &[f64] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, count) };
                let buf =
                    crate::transfer::cpu_to_gpu_pinned(f64_data, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer_f64(buf, device))
            }
            DType::I32 => {
                // Integer pinned transport (crosslink #1185 Phase 2a). i32 has
                // cudarc `DeviceRepr + ValidAsZeroBits + Copy`, so the generic
                // pinned path applies. Tag the handle `DType::I32`.
                let count = data.len() / 4;
                // SAFETY: identical invariants to the `cpu_to_gpu` I32 arm —
                // `data` is the byte serialisation of a contiguous `&[i32]`
                // (size 4, align 4), `count = data.len() / 4` is exact, the
                // reinterpreted `&[i32]` is bounded by `data` and consumed by
                // `cpu_to_gpu_pinned` before this frame returns, and `data` is
                // a shared borrow so no `&mut [i32]` alias exists.
                let i32_data: &[i32] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i32, count) };
                let buf =
                    crate::transfer::cpu_to_gpu_pinned(i32_data, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer_i32(buf, device))
            }
            DType::I64 => {
                // Integer pinned transport (crosslink #1185 Phase 2a). i64 has
                // cudarc `DeviceRepr + ValidAsZeroBits + Copy`. Tag `DType::I64`.
                let count = data.len() / 8;
                // SAFETY: same as the I32 pinned arm with element width 8.
                let i64_data: &[i64] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i64, count) };
                let buf =
                    crate::transfer::cpu_to_gpu_pinned(i64_data, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer_i64(buf, device))
            }
            DType::Bool => {
                // Boolean pinned transport (crosslink #1185 Phase 3a). `bool` is
                // one byte; stored as `CudaBuffer<u8>` (u8 is DeviceRepr). The
                // `&[u8]` is already the byte view of the `&[bool]`.
                let buf =
                    crate::transfer::cpu_to_gpu_pinned(data, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer_bool(buf, device))
            }
            // PyTorch parity: any other dtype is a structured error, not a
            // silent fallback.
            other => Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "cpu_to_gpu_pinned: dtype {other} not supported on CUDA \
                     (supported: F32, F64, I32, I64, Bool)"
                ),
            }),
        }
    }

    fn gpu_to_cpu(&self, handle: &GpuBufferHandle) -> FerrotorchResult<Vec<u8>> {
        let dev = self.device(handle.device_ordinal())?;

        // Try f32 first, then f64, then bf16 (CudaSlice<u16> bit pattern).
        if let Ok(buf) = Self::unwrap_buffer(handle) {
            let f32_data = crate::transfer::gpu_to_cpu(buf, dev).map_err(Self::map_gpu_err)?;

            // Reinterpret Vec<f32> as Vec<u8> without copying.
            // SAFETY: f32 has alignment 4 and size 4. We adjust len and capacity
            // accordingly. The original Vec is consumed via ManuallyDrop so its
            // destructor won't free the allocation.
            let bytes = unsafe {
                let mut v = std::mem::ManuallyDrop::new(f32_data);
                let ptr = v.as_mut_ptr() as *mut u8;
                let len = v.len() * 4;
                let cap = v.capacity() * 4;
                Vec::from_raw_parts(ptr, len, cap)
            };
            Ok(bytes)
        } else if let Ok(buf) = Self::unwrap_buffer_f64(handle) {
            let f64_data = crate::transfer::gpu_to_cpu(buf, dev).map_err(Self::map_gpu_err)?;

            // Reinterpret Vec<f64> as Vec<u8> without copying.
            // SAFETY: f64 has alignment 8 and size 8. We adjust len and capacity
            // accordingly. The original Vec is consumed via ManuallyDrop so its
            // destructor won't free the allocation.
            let bytes = unsafe {
                let mut v = std::mem::ManuallyDrop::new(f64_data);
                let ptr = v.as_mut_ptr() as *mut u8;
                let len = v.len() * 8;
                let cap = v.capacity() * 8;
                Vec::from_raw_parts(ptr, len, cap)
            };
            Ok(bytes)
        } else if let Ok(slice) = Self::unwrap_buffer_bf16(handle) {
            // bf16 storage: raw `CudaSlice<u16>` (no `CudaBuffer<T>` wrapper).
            // `clone_dtoh` copies the whole slice including any rounded-up
            // tail; bf16 buffers do not currently use the pool, so the slice
            // length equals the logical length and no truncation is needed.
            let u16_data = dev
                .stream()
                .clone_dtoh(slice)
                .map_err(|e| Self::map_gpu_err(crate::error::GpuError::Driver(e)))?;

            // Reinterpret Vec<u16> as Vec<u8> without copying.
            // SAFETY: u16 has alignment 2 and size 2. We adjust len and capacity
            // accordingly; the original Vec is consumed via ManuallyDrop so its
            // destructor never runs and the allocation stays live under its
            // new u8-typed handle (Layout::array::<u16>(cap) and
            // Layout::array::<u8>(cap*2) describe the same byte range).
            let bytes = unsafe {
                let mut v = std::mem::ManuallyDrop::new(u16_data);
                let ptr = v.as_mut_ptr() as *mut u8;
                let len = v.len() * 2;
                let cap = v.capacity() * 2;
                Vec::from_raw_parts(ptr, len, cap)
            };
            Ok(bytes)
        } else if let Ok(slice) = Self::unwrap_buffer_f16(handle) {
            // f16 storage: raw `CudaSlice<u16>` tagged `DType::F16`. Same
            // dtoh path as bf16 (byte-identical width); the tag check in
            // `unwrap_buffer_f16` is what routes us here vs. the bf16 arm.
            let u16_data = dev
                .stream()
                .clone_dtoh(slice)
                .map_err(|e| Self::map_gpu_err(crate::error::GpuError::Driver(e)))?;

            // Reinterpret Vec<u16> as Vec<u8> without copying.
            // SAFETY: u16 has alignment 2 and size 2; len/capacity adjusted
            // accordingly. The original Vec is consumed via ManuallyDrop so
            // its destructor never runs and the allocation stays live under
            // its new u8-typed handle.
            let bytes = unsafe {
                let mut v = std::mem::ManuallyDrop::new(u16_data);
                let ptr = v.as_mut_ptr() as *mut u8;
                let len = v.len() * 2;
                let cap = v.capacity() * 2;
                Vec::from_raw_parts(ptr, len, cap)
            };
            Ok(bytes)
        } else if let Ok(buf) = Self::unwrap_buffer_i32(handle) {
            // i32 device storage (crosslink #1185 Phase 2a). Real
            // `CudaBuffer<i32>` (not a `CudaSlice<u16>` bit pattern), so the
            // generic `transfer::gpu_to_cpu` D2H path applies directly.
            let i32_data = crate::transfer::gpu_to_cpu(buf, dev).map_err(Self::map_gpu_err)?;

            // Reinterpret Vec<i32> as Vec<u8> without copying.
            // SAFETY: i32 has alignment 4 and size 4; len/capacity adjusted
            // accordingly. The original Vec is consumed via ManuallyDrop so its
            // destructor never runs and the allocation stays live under its new
            // u8-typed handle (the byte ranges Layout::array::<i32>(cap) and
            // Layout::array::<u8>(cap*4) describe are identical).
            let bytes = unsafe {
                let mut v = std::mem::ManuallyDrop::new(i32_data);
                let ptr = v.as_mut_ptr() as *mut u8;
                let len = v.len() * 4;
                let cap = v.capacity() * 4;
                Vec::from_raw_parts(ptr, len, cap)
            };
            Ok(bytes)
        } else if let Ok(buf) = Self::unwrap_buffer_i64(handle) {
            // i64 device storage (crosslink #1185 Phase 2a). Real
            // `CudaBuffer<i64>`; generic D2H path applies.
            let i64_data = crate::transfer::gpu_to_cpu(buf, dev).map_err(Self::map_gpu_err)?;

            // Reinterpret Vec<i64> as Vec<u8> without copying.
            // SAFETY: i64 has alignment 8 and size 8; len/capacity adjusted
            // accordingly. ManuallyDrop keeps the allocation live under the new
            // u8-typed handle (identical byte range to the i64 Vec).
            let bytes = unsafe {
                let mut v = std::mem::ManuallyDrop::new(i64_data);
                let ptr = v.as_mut_ptr() as *mut u8;
                let len = v.len() * 8;
                let cap = v.capacity() * 8;
                Vec::from_raw_parts(ptr, len, cap)
            };
            Ok(bytes)
        } else if let Ok(buf) = Self::unwrap_buffer_bool(handle) {
            // bool device storage (crosslink #1185 Phase 3a). Real
            // `CudaBuffer<u8>`; the D2H readback is already a `Vec<u8>`, and the
            // raw-byte transport's `Vec<u8>` IS the byte serialisation — no
            // reinterpret needed (each byte is 0 or 1, a valid `bool`). The
            // caller (`BoolTensor::to(Cpu)`) reconstructs `Vec<bool>` from these
            // bytes.
            let u8_data = crate::transfer::gpu_to_cpu(buf, dev).map_err(Self::map_gpu_err)?;
            Ok(u8_data)
        } else {
            Err(FerrotorchError::InvalidArgument {
                message: "gpu_to_cpu: handle is not a recognised dtype \
                          (expected CudaBuffer<f32>, CudaBuffer<f64>, \
                          CudaSlice<u16> for bf16/f16, CudaBuffer<i32>/<i64>, \
                          or CudaBuffer<u8> for bool)"
                    .into(),
            })
        }
    }

    fn clone_buffer(&self, handle: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        // Device-to-device copy — NEVER through host RAM (crosslink #1185
        // hardening, task #36). Every `.clone()` of a GPU tensor of any dtype
        // routes here, so the old GPU->CPU->GPU round trip was a silent host
        // crossing in a universal hot op. Each arm unwraps the concrete typed
        // buffer and calls `CudaSlice::try_clone`, which allocates a fresh
        // device slice and schedules a `memcpy_dtod_async` (cudarc 0.19.4
        // `CudaSlice::try_clone` -> `CudaStream::clone_dtod` ->
        // `CudaStream::memcpy_dtod`); when source and destination share the
        // same CUDA context — always true for a same-handle clone — that is a
        // pure `result::memcpy_dtod_async`, with no host staging.
        //
        // The dtype tag is authoritative and is preserved exactly: each arm is
        // selected on `handle.dtype()` and rewraps with the matching dtype tag
        // (PyTorch parity — a clone preserves the ScalarType, never re-infers
        // it from bytes).
        //
        // LENGTH PRESERVATION: for the `CudaBuffer<T>`-backed dtypes the
        // underlying `CudaSlice` may be longer than the logical element count
        // — pooled buffers round their allocation up (see `pool::round_len`,
        // 256-element granularity), so a 4-element f32 tensor sits in a
        // 256-element slice. `try_clone` copies the whole slice, but the new
        // handle MUST carry the SAME logical `len`/`alloc_len` as the source
        // (otherwise a clone of a 4-element grad buffer would report length
        // 256 and break downstream length checks). We therefore rebuild a
        // non-pooled `CudaBuffer` with the source's logical `len`/`alloc_len`
        // (the clone is non-pooled: it never returns to the pool on drop,
        // matching how `wrap_slice_*` non-pool their inputs). bf16/f16 store a
        // bare `CudaSlice<u16>` (never pooled — slice length == logical
        // length), so they wrap the cloned slice directly.
        let ordinal = handle.device_ordinal();
        let map_drv = |e| Self::map_gpu_err(crate::error::GpuError::Driver(e));
        match handle.dtype() {
            DType::F32 => {
                let buf = Self::unwrap_buffer(handle)?;
                let slice = buf.inner().try_clone().map_err(map_drv)?;
                let cloned = CudaBuffer {
                    data: Some(slice),
                    len: buf.len(),
                    alloc_len: buf.alloc_len(),
                    device_ordinal: ordinal,
                    pool_fn: None,
                };
                Ok(Self::wrap_buffer(cloned, ordinal))
            }
            DType::F64 => {
                let buf = Self::unwrap_buffer_f64(handle)?;
                let slice = buf.inner().try_clone().map_err(map_drv)?;
                let cloned = CudaBuffer {
                    data: Some(slice),
                    len: buf.len(),
                    alloc_len: buf.alloc_len(),
                    device_ordinal: ordinal,
                    pool_fn: None,
                };
                Ok(Self::wrap_buffer_f64(cloned, ordinal))
            }
            DType::BF16 => {
                // bf16 storage is a bare `CudaSlice<u16>` bit pattern (never
                // pooled — slice length equals the logical length).
                let slice = Self::unwrap_buffer_bf16(handle)?;
                let cloned = slice.try_clone().map_err(map_drv)?;
                Ok(Self::wrap_buffer_bf16(cloned, ordinal))
            }
            DType::F16 => {
                // f16 shares `CudaSlice<u16>` storage with bf16; the F16 tag
                // is the only discriminator and is preserved by rewrapping
                // via `wrap_buffer_f16`.
                let slice = Self::unwrap_buffer_f16(handle)?;
                let cloned = slice.try_clone().map_err(map_drv)?;
                Ok(Self::wrap_buffer_f16(cloned, ordinal))
            }
            DType::I32 => {
                let buf = Self::unwrap_buffer_i32(handle)?;
                let slice = buf.inner().try_clone().map_err(map_drv)?;
                let cloned = CudaBuffer {
                    data: Some(slice),
                    len: buf.len(),
                    alloc_len: buf.alloc_len(),
                    device_ordinal: ordinal,
                    pool_fn: None,
                };
                Ok(Self::wrap_buffer_i32(cloned, ordinal))
            }
            DType::I64 => {
                let buf = Self::unwrap_buffer_i64(handle)?;
                let slice = buf.inner().try_clone().map_err(map_drv)?;
                let cloned = CudaBuffer {
                    data: Some(slice),
                    len: buf.len(),
                    alloc_len: buf.alloc_len(),
                    device_ordinal: ordinal,
                    pool_fn: None,
                };
                Ok(Self::wrap_buffer_i64(cloned, ordinal))
            }
            DType::Bool => {
                // Bool storage is a native `CudaBuffer<u8>`.
                let buf = Self::unwrap_buffer_bool(handle)?;
                let slice = buf.inner().try_clone().map_err(map_drv)?;
                let cloned = CudaBuffer {
                    data: Some(slice),
                    len: buf.len(),
                    alloc_len: buf.alloc_len(),
                    device_ordinal: ordinal,
                    pool_fn: None,
                };
                Ok(Self::wrap_buffer_bool(cloned, ordinal))
            }
            // PyTorch parity: any other dtype has no on-device storage type in
            // this backend, so cloning it is a structured error — never a
            // silent host round trip.
            other => Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "clone_buffer: dtype {other} has no device-to-device copy \
                     path on CUDA (supported: F32, F64, BF16, F16, I32, I64, Bool)"
                ),
            }),
        }
    }

    fn has_inf_nan_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<bool> {
        // #687: dispatch to the real GPU reduction kernel. The kernel writes
        // a single 4-byte flag on device; only that flag is read back to host,
        // not the whole buffer.
        let buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        crate::kernels::gpu_has_inf_nan(buf, dev).map_err(Self::map_gpu_err)
    }

    fn alloc_zeros(
        &self,
        len: usize,
        dtype: DType,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(device)?;
        match dtype {
            DType::BF16 => {
                // bf16 (u16 bit pattern).
                let slice =
                    crate::transfer::alloc_zeros_bf16(len, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer_bf16(slice, device))
            }
            DType::F16 => {
                // f16 (u16 bit pattern). Zero is 0x0000 in both f16 and bf16,
                // so the bf16 zero-alloc helper produces a byte-correct f16
                // buffer; only the handle tag (F16) differs.
                let slice =
                    crate::transfer::alloc_zeros_bf16(len, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer_f16(slice, device))
            }
            DType::F32 => {
                let buf = crate::transfer::alloc_zeros_f32(len, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer(buf, device))
            }
            DType::F64 => {
                let buf = crate::transfer::alloc_zeros_f64(len, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer_f64(buf, device))
            }
            DType::I32 => {
                // Integer zero-alloc (crosslink #1185 Phase 2a). i32 has cudarc
                // `DeviceRepr + ValidAsZeroBits`; the generic (non-pooled)
                // `alloc_zeros::<i32>` applies. The integer pool path is a
                // follow-up (the pool is keyed by elem_size and currently only
                // stocks f32/f64 slices); zero-init IntTensors are rare so the
                // pool-miss cost is acceptable for Phase 2a.
                let buf: CudaBuffer<i32> =
                    crate::transfer::alloc_zeros(len, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer_i32(buf, device))
            }
            DType::I64 => {
                // Integer zero-alloc (crosslink #1185 Phase 2a). i64 has cudarc
                // `DeviceRepr + ValidAsZeroBits`; generic `alloc_zeros::<i64>`.
                let buf: CudaBuffer<i64> =
                    crate::transfer::alloc_zeros(len, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer_i64(buf, device))
            }
            DType::Bool => {
                // Boolean zero-alloc (crosslink #1185 Phase 3a). bool stored as
                // `CudaBuffer<u8>` (u8 has `DeviceRepr + ValidAsZeroBits`); a
                // zero byte is `false`, so generic `alloc_zeros::<u8>` is correct.
                let buf: CudaBuffer<u8> =
                    crate::transfer::alloc_zeros(len, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer_bool(buf, device))
            }
            // PyTorch parity: unsupported (dtype, CUDA) → structured error.
            other => Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "alloc_zeros: dtype {other} not supported on CUDA \
                     (supported: F32, F64, BF16, F16, I32, I64, Bool)"
                ),
            }),
        }
    }

    // -- Elementwise f32 ------------------------------------------------------

    fn add_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_add(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn sub_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_sub(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn add_scaled_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        alpha: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        // `alpha` arrives as f64 (the dtype-agnostic trait scalar); narrow to
        // f32 for the f32 fused kernel. `alpha as f32` is the same rounding
        // PyTorch applies when an f64 Python scalar is bound to an f32 tensor
        // op, and it preserves NaN / +-inf bit patterns.
        let result = crate::kernels::gpu_add_scaled_f32(a_buf, b_buf, alpha as f32, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn mul_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_mul(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn neg_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_neg(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn relu_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_relu(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn div_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_div(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn exp_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_exp(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn log_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_log(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn sqrt_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_sqrt(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn pow_f32(&self, a: &GpuBufferHandle, exponent: f32) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_pow(a_buf, exponent, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn abs_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_abs(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn sigmoid_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_sigmoid(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn tanh_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_tanh(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    // -----------------------------------------------------------------------
    // f64 elementwise ops
    // -----------------------------------------------------------------------

    fn add_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_add_f64(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn sub_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_sub_f64(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn add_scaled_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        alpha: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        // f64 alpha is used at full precision by the f64 fused kernel.
        let result = crate::kernels::gpu_add_scaled_f64(a_buf, b_buf, alpha, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn mul_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_mul_f64(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn div_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_div_f64(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn neg_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_neg_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn relu_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_relu_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn scale_f64(&self, a: &GpuBufferHandle, scalar: f64) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_scale_f64(a_buf, scalar, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn exp_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_exp_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn log_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_log_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn sqrt_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_sqrt_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn pow_f64(&self, a: &GpuBufferHandle, exponent: f64) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_pow_f64(a_buf, exponent, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn abs_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_abs_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn sigmoid_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_sigmoid_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn tanh_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_tanh_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    // f64 backward ops
    fn relu_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let i_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result =
            crate::kernels::gpu_relu_backward_f64(g_buf, i_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    fn abs_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let i_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result =
            crate::kernels::gpu_abs_backward_f64(g_buf, i_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    fn sigmoid_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let o_buf = Self::unwrap_buffer_f64(output)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_sigmoid_backward_f64(g_buf, o_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    fn tanh_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let o_buf = Self::unwrap_buffer_f64(output)?;
        let dev = self.device(grad.device_ordinal())?;
        let result =
            crate::kernels::gpu_tanh_backward_f64(g_buf, o_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    // f64 activation forward ops

    fn gelu_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn gelu_tanh_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_tanh_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn gelu_erf_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_erf_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn silu_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_silu_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn elu_f64(&self, a: &GpuBufferHandle, alpha: f64) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_elu_f64(a_buf, alpha, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn mish_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_mish_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn clamp_f64(
        &self,
        a: &GpuBufferHandle,
        min_val: f64,
        max_val: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_clamp_f64(a_buf, min_val, max_val, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn clamp_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
        min_val: f64,
        max_val: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let i_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_clamp_backward_f64(g_buf, i_buf, min_val, max_val, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    // f64 activation backward ops

    fn gelu_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let i_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result =
            crate::kernels::gpu_gelu_backward_f64(g_buf, i_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    fn gelu_backward_tanh_f64(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let i_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_backward_tanh_f64(g_buf, i_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    fn gelu_backward_erf_f64(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let i_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_backward_erf_f64(g_buf, i_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    fn silu_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let i_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result =
            crate::kernels::gpu_silu_backward_f64(g_buf, i_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    fn elu_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
        alpha: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let i_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_elu_backward_f64(g_buf, i_buf, alpha, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    fn mish_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let i_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result =
            crate::kernels::gpu_mish_backward_f64(g_buf, i_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    // f64 cumulative ops
    fn cumsum_f64(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_cumsum_f64(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn cumprod_f64(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_cumprod_f64(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn cummax_f64(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (vals, idxs) = crate::kernels::gpu_cummax_f64(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((
            Self::wrap_buffer_f64(vals, ord),
            Self::wrap_buffer_i64(idxs, ord),
        ))
    }

    fn cummin_f64(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (vals, idxs) = crate::kernels::gpu_cummin_f64(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((
            Self::wrap_buffer_f64(vals, ord),
            Self::wrap_buffer_i64(idxs, ord),
        ))
    }

    fn max_with_dim_f64(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (vals, idxs) = crate::kernels::gpu_max_with_dim_f64(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((
            Self::wrap_buffer_f64(vals, ord),
            Self::wrap_buffer_i64(idxs, ord),
        ))
    }

    fn min_with_dim_f64(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (vals, idxs) = crate::kernels::gpu_min_with_dim_f64(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((
            Self::wrap_buffer_f64(vals, ord),
            Self::wrap_buffer_i64(idxs, ord),
        ))
    }

    fn logcumsumexp_f64(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_logcumsumexp_f64(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    // f64 shape ops
    fn transpose_2d_f64(
        &self,
        a: &GpuBufferHandle,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_transpose_2d_f64(a_buf, m, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn permute_0213_f64(
        &self,
        a: &GpuBufferHandle,
        d0: usize,
        d1: usize,
        d2: usize,
        d3: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_permute_0213_f64(a_buf, d0, d1, d2, d3, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    // f64 broadcast ops
    fn broadcast_add_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_broadcast_add_f64(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn broadcast_sub_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_broadcast_sub_f64(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn broadcast_mul_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_broadcast_mul_f64(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn broadcast_div_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_broadcast_div_f64(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    // f64 reduction ops
    fn sum_f64(&self, a: &GpuBufferHandle, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_reduce_sum_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn prod_f64(&self, a: &GpuBufferHandle, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_reduce_prod_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn prod_backward_f64(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_f64(input)?;
        let grad_buf = Self::unwrap_buffer_f64(grad_output)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_prod_backward_f64(input_buf, grad_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    fn prod_axis_f64(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_prod_axis_f64(a_buf, outer, axis_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn prod_axis_backward_f64(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_f64(input)?;
        let grad_buf = Self::unwrap_buffer_f64(grad_output)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_prod_axis_backward_f64(
            input_buf, grad_buf, outer, axis_size, inner, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    fn std_var_axis_f64(
        &self,
        input: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
        correction: f64,
        take_sqrt: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_std_var_axis_f64(
            input_buf, outer, axis_size, inner, correction, take_sqrt, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    fn std_var_axis_backward_f64(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        result: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
        correction: f64,
        take_sqrt: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_f64(input)?;
        let grad_buf = Self::unwrap_buffer_f64(grad_output)?;
        let result_buf = Self::unwrap_buffer_f64(result)?;
        let dev = self.device(input.device_ordinal())?;
        let grad = crate::kernels::gpu_std_var_axis_backward_f64(
            input_buf, grad_buf, result_buf, outer, axis_size, inner, correction, take_sqrt, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(grad, input.device_ordinal()))
    }

    fn logsumexp_axis_f64(
        &self,
        input: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(input.device_ordinal())?;
        let (sum, shift) =
            crate::kernels::gpu_logsumexp_sum_shift_f64(input_buf, outer, axis_size, inner, dev)
                .map_err(Self::map_gpu_err)?;
        let result = crate::kernels::gpu_logsumexp_finalize_f64(&sum, &shift, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    fn nan_reduce_axis_f64(
        &self,
        input: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
        take_mean: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::nan_reductions::nan_reduce_axis_f64(
            input_buf, outer, axis_size, inner, take_mean, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    fn nan_reduce_axis_backward_f64(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
        take_mean: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_f64(input)?;
        let grad_buf = Self::unwrap_buffer_f64(grad_output)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::nan_reductions::nan_reduce_axis_backward_f64(
            input_buf, grad_buf, outer, axis_size, inner, take_mean, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    fn min_f64(&self, a: &GpuBufferHandle, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_reduce_min_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn max_f64(&self, a: &GpuBufferHandle, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_reduce_max_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn extreme_backward_f64(
        &self,
        input: &GpuBufferHandle,
        extreme: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_f64(input)?;
        let extreme_buf = Self::unwrap_buffer_f64(extreme)?;
        let grad_buf = Self::unwrap_buffer_f64(grad_output)?;
        let dev = self.device(input.device_ordinal())?;
        let result =
            crate::kernels::gpu_extreme_backward_f64(input_buf, extreme_buf, grad_buf, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    fn masked_min_f64(
        &self,
        data: &GpuBufferHandle,
        mask_f: &GpuBufferHandle,
        _len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let d_buf = Self::unwrap_buffer_f64(data)?;
        let m_buf = Self::unwrap_buffer_f64(mask_f)?;
        let dev = self.device(data.device_ordinal())?;
        let result = crate::kernels::gpu_masked_reduce_min_f64(d_buf, m_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, data.device_ordinal()))
    }

    fn masked_max_f64(
        &self,
        data: &GpuBufferHandle,
        mask_f: &GpuBufferHandle,
        _len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let d_buf = Self::unwrap_buffer_f64(data)?;
        let m_buf = Self::unwrap_buffer_f64(mask_f)?;
        let dev = self.device(data.device_ordinal())?;
        let result = crate::kernels::gpu_masked_reduce_max_f64(d_buf, m_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, data.device_ordinal()))
    }

    fn sum_axis_f64(
        &self,
        a: &GpuBufferHandle,
        shape: &[usize],
        axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let outer: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let result = crate::kernels::gpu_sum_axis_f64(a_buf, outer, axis_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn min_axis_f64(
        &self,
        a: &GpuBufferHandle,
        shape: &[usize],
        axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let outer: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let result =
            crate::kernels::gpu_extreme_axis_f64(a_buf, outer, axis_size, inner, false, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn max_axis_f64(
        &self,
        a: &GpuBufferHandle,
        shape: &[usize],
        axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let outer: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let result =
            crate::kernels::gpu_extreme_axis_f64(a_buf, outer, axis_size, inner, true, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn extreme_axis_backward_f64(
        &self,
        input: &GpuBufferHandle,
        result: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        shape: &[usize],
        axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_f64(input)?;
        let result_buf = Self::unwrap_buffer_f64(result)?;
        let grad_buf = Self::unwrap_buffer_f64(grad_output)?;
        let dev = self.device(input.device_ordinal())?;
        let outer: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let out = crate::kernels::gpu_extreme_axis_backward_f64(
            input_buf, result_buf, grad_buf, outer, axis_size, inner, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, input.device_ordinal()))
    }

    // f64 softmax / log-softmax / layernorm / rmsnorm

    fn softmax_f64(
        &self,
        a: &GpuBufferHandle,
        rows: usize,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_softmax_f64(a_buf, rows, cols, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn softmax_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        output: &GpuBufferHandle,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer_f64(grad)?;
        let output_buf = Self::unwrap_buffer_f64(output)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_softmax_backward_f64(grad_buf, output_buf, cols, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    fn log_softmax_f64(
        &self,
        a: &GpuBufferHandle,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_log_softmax_f64(a_buf, cols, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn log_softmax_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        output: &GpuBufferHandle,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer_f64(grad)?;
        let output_buf = Self::unwrap_buffer_f64(output)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_log_softmax_backward_f64(grad_buf, output_buf, cols, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    fn layernorm_f64(
        &self,
        input: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        bias: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer_f64(input)?;
        let w_buf = Self::unwrap_buffer_f64(weight)?;
        let b_buf = Self::unwrap_buffer_f64(bias)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_layernorm_f64(in_buf, w_buf, b_buf, rows, cols, eps, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    fn layernorm_backward_f64(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f64,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        let in_buf = Self::unwrap_buffer_f64(input)?;
        let go_buf = Self::unwrap_buffer_f64(grad_output)?;
        let w_buf = Self::unwrap_buffer_f64(weight)?;
        let dev = self.device(input.device_ordinal())?;
        let (gi, gw, gb) =
            crate::kernels::gpu_layernorm_backward_f64(in_buf, go_buf, w_buf, rows, cols, eps, dev)
                .map_err(Self::map_gpu_err)?;
        let ordinal = input.device_ordinal();
        Ok((
            Self::wrap_buffer_f64(gi, ordinal),
            Self::wrap_buffer_f64(gw, ordinal),
            Self::wrap_buffer_f64(gb, ordinal),
        ))
    }

    fn rmsnorm_f64(
        &self,
        input: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer_f64(input)?;
        let w_buf = Self::unwrap_buffer_f64(weight)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_rmsnorm_f64(in_buf, w_buf, rows, cols, eps, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    fn rmsnorm_backward_f64(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f64,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let in_buf = Self::unwrap_buffer_f64(input)?;
        let go_buf = Self::unwrap_buffer_f64(grad_output)?;
        let w_buf = Self::unwrap_buffer_f64(weight)?;
        let dev = self.device(input.device_ordinal())?;
        let (gi, gw) =
            crate::kernels::gpu_rmsnorm_backward_f64(in_buf, go_buf, w_buf, rows, cols, eps, dev)
                .map_err(Self::map_gpu_err)?;
        let ordinal = input.device_ordinal();
        Ok((
            Self::wrap_buffer_f64(gi, ordinal),
            Self::wrap_buffer_f64(gw, ordinal),
        ))
    }

    // f64 embedding / scatter / indexing

    fn embed_lookup_f64(
        &self,
        idx: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        // indices are always f32-encoded
        let idx_buf = Self::unwrap_buffer(idx)?;
        let w_buf = Self::unwrap_buffer_f64(weight)?;
        let dev = self.device(idx.device_ordinal())?;
        let result = crate::kernels::gpu_embed_lookup_f64(idx_buf, w_buf, d, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, idx.device_ordinal()))
    }

    fn embed_lookup_batch_f64(
        &self,
        indices: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        n: usize,
        d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        // indices are always f32-encoded
        let idx_buf = Self::unwrap_buffer(indices)?;
        let w_buf = Self::unwrap_buffer_f64(weight)?;
        let dev = self.device(indices.device_ordinal())?;
        let result = crate::kernels::gpu_embed_lookup_batch_f64(idx_buf, w_buf, n, d, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, indices.device_ordinal()))
    }

    fn scatter_add_rows_f64(
        &self,
        grad_output: &GpuBufferHandle,
        indices: &GpuBufferHandle,
        num_embeddings: usize,
        d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let go_buf = Self::unwrap_buffer_f64(grad_output)?;
        // indices are always f32-encoded
        let idx_buf = Self::unwrap_buffer(indices)?;
        let dev = self.device(grad_output.device_ordinal())?;
        let result =
            crate::kernels::gpu_scatter_add_rows_f64(go_buf, idx_buf, num_embeddings, d, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad_output.device_ordinal()))
    }

    // f64 masked fill / masked zero
    //
    // The f64 kernels expect CudaBuffer<u8> for the mask, but the trait
    // provides a GpuBufferHandle containing CudaBuffer<f32> (1.0/0.0 encoding).
    // We convert f32 mask -> u8 mask via a CPU roundtrip.

    fn masked_fill_f64(
        &self,
        input: &GpuBufferHandle,
        mask: &GpuBufferHandle,
        value: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_f64(input)?;
        let mask_f32 = Self::unwrap_buffer(mask)?;
        let dev = self.device(input.device_ordinal())?;
        // Convert f32 mask to u8 mask on GPU via CPU roundtrip
        let mask_host = crate::transfer::gpu_to_cpu(mask_f32, dev).map_err(Self::map_gpu_err)?;
        let mask_u8: Vec<u8> = mask_host
            .iter()
            .map(|&v| if v != 0.0 { 1u8 } else { 0u8 })
            .collect();
        let mask_gpu = crate::transfer::cpu_to_gpu(&mask_u8, dev).map_err(Self::map_gpu_err)?;
        let result = crate::kernels::gpu_masked_fill_f64(input_buf, &mask_gpu, value, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    fn masked_zero_f64(
        &self,
        grad: &GpuBufferHandle,
        mask: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer_f64(grad)?;
        let mask_f32 = Self::unwrap_buffer(mask)?;
        let dev = self.device(grad.device_ordinal())?;
        // Convert f32 mask to u8 mask on GPU via CPU roundtrip
        let mask_host = crate::transfer::gpu_to_cpu(mask_f32, dev).map_err(Self::map_gpu_err)?;
        let mask_u8: Vec<u8> = mask_host
            .iter()
            .map(|&v| if v != 0.0 { 1u8 } else { 0u8 })
            .collect();
        let mask_gpu = crate::transfer::cpu_to_gpu(&mask_u8, dev).map_err(Self::map_gpu_err)?;
        let result = crate::kernels::gpu_masked_zero_f64(grad_buf, &mask_gpu, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    // f64 slice ops

    fn slice_write_f64(
        &self,
        src: &GpuBufferHandle,
        dst: &mut GpuBufferHandle,
        n_batch: usize,
        d: usize,
        max_len: usize,
        pos: usize,
    ) -> FerrotorchResult<()> {
        let src_buf = Self::unwrap_buffer_f64(src)?;
        let dst_buf = Self::unwrap_buffer_f64_mut(dst)?;
        let dev = self.device(src.device_ordinal())?;
        crate::kernels::gpu_slice_write_f64(src_buf, dst_buf, n_batch, d, max_len, pos, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(())
    }

    fn slice_read_f64(
        &self,
        src: &GpuBufferHandle,
        n_batch: usize,
        d: usize,
        len: usize,
        max_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let src_buf = Self::unwrap_buffer_f64(src)?;
        let dev = self.device(src.device_ordinal())?;
        let result = crate::kernels::gpu_slice_read_f64(src_buf, n_batch, d, len, max_len, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, src.device_ordinal()))
    }

    // f64 strided split / cat

    fn strided_split_f64(
        &self,
        input: &GpuBufferHandle,
        total_along_axis: usize,
        split_offset: usize,
        split_size: usize,
        inner_size: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_strided_split_f64(
            in_buf,
            total_along_axis,
            split_offset,
            split_size,
            inner_size,
            n,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    // f64 indexing ops

    fn index_select_1d_f64(
        &self,
        input: &GpuBufferHandle,
        indices: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_f64(input)?;
        // indices are always f32-encoded
        let idx_buf = Self::unwrap_buffer(indices)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_index_select_1d_f64(input_buf, idx_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    fn scatter_add_1d_f64(
        &self,
        grad_output: &GpuBufferHandle,
        indices: &GpuBufferHandle,
        input_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let go_buf = Self::unwrap_buffer_f64(grad_output)?;
        // indices are always f32-encoded
        let idx_buf = Self::unwrap_buffer(indices)?;
        let dev = self.device(grad_output.device_ordinal())?;
        let result = crate::kernels::gpu_scatter_add_1d_f64(go_buf, idx_buf, input_len, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad_output.device_ordinal()))
    }

    fn index_select_dim_f64(
        &self,
        input: &GpuBufferHandle,
        indices: &GpuBufferHandle,
        outer: usize,
        in_dim_size: usize,
        out_dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_f64(input)?;
        // indices are always f32-encoded
        let idx_buf = Self::unwrap_buffer(indices)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_index_select_dim_f64(
            input_buf,
            idx_buf,
            outer,
            in_dim_size,
            out_dim_size,
            inner,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    fn bmm_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::blas::gpu_bmm_f64(a_buf, b_buf, batch, m, k, n, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn broadcast_bmm_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_lead: &[usize],
        b_lead: &[usize],
        out_lead: &[usize],
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::blas::gpu_broadcast_bmm_f64(
            a_buf, b_buf, a_lead, b_lead, out_lead, m, k, n, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    #[allow(clippy::too_many_arguments)]
    fn fused_adam_f32(
        &self,
        param: &mut GpuBufferHandle,
        grad: &GpuBufferHandle,
        exp_avg: &mut GpuBufferHandle,
        exp_avg_sq: &mut GpuBufferHandle,
        beta1: f32,
        beta2: f32,
        lr: f32,
        eps: f32,
        bc1: f32,
        bc2: f32,
        weight_decay: f32,
    ) -> FerrotorchResult<()> {
        let ordinal = param.device_ordinal();
        let dev = self.device(ordinal)?;
        let p_buf = Self::unwrap_buffer_mut(param)?;
        let g_buf = Self::unwrap_buffer(grad)?;
        let m_buf = Self::unwrap_buffer_mut(exp_avg)?;
        let v_buf = Self::unwrap_buffer_mut(exp_avg_sq)?;
        crate::kernels::gpu_fused_adam(
            p_buf,
            g_buf,
            m_buf,
            v_buf,
            beta1,
            beta2,
            lr,
            eps,
            bc1,
            bc2,
            weight_decay,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn maxpool2d_f32(
        &self,
        input: &GpuBufferHandle,
        batch: usize,
        channels: usize,
        h_in: usize,
        w_in: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        let buf = Self::unwrap_buffer(input)?;
        let dev = self.device(input.device_ordinal())?;
        let (out, shape) = crate::kernels::gpu_maxpool2d(
            buf, batch, channels, h_in, w_in, kh, kw, sh, sw, ph, pw, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok((Self::wrap_buffer(out, input.device_ordinal()), shape))
    }

    #[allow(clippy::too_many_arguments)]
    fn avgpool2d_f32(
        &self,
        input: &GpuBufferHandle,
        batch: usize,
        channels: usize,
        h_in: usize,
        w_in: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        let buf = Self::unwrap_buffer(input)?;
        let dev = self.device(input.device_ordinal())?;
        let (out, shape) = crate::kernels::gpu_avgpool2d(
            buf, batch, channels, h_in, w_in, kh, kw, sh, sw, ph, pw, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok((Self::wrap_buffer(out, input.device_ordinal()), shape))
    }

    #[allow(clippy::too_many_arguments)]
    fn conv2d_f32(
        &self,
        input: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        bias: Option<&GpuBufferHandle>,
        input_shape: [usize; 4],
        weight_shape: [usize; 4],
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        let input_buf = Self::unwrap_buffer(input)?;
        let weight_buf = Self::unwrap_buffer(weight)?;
        let bias_buf = match bias {
            Some(b) => Some(Self::unwrap_buffer(b)?),
            None => None,
        };
        let dev = self.device(input.device_ordinal())?;
        let (out_buf, out_shape) = crate::conv::gpu_conv2d_f32(
            input_buf,
            weight_buf,
            bias_buf,
            input_shape,
            weight_shape,
            stride,
            padding,
            dilation,
            groups,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok((
            Self::wrap_buffer(out_buf, input.device_ordinal()),
            out_shape,
        ))
    }

    fn fused_gru_cell_f32(
        &self,
        input_gates: &GpuBufferHandle,
        hidden_gates: &GpuBufferHandle,
        bias_ih: &GpuBufferHandle,
        bias_hh: &GpuBufferHandle,
        hx: &GpuBufferHandle,
        hidden_size: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let ig = Self::unwrap_buffer(input_gates)?;
        let hg = Self::unwrap_buffer(hidden_gates)?;
        let bih = Self::unwrap_buffer(bias_ih)?;
        let bhh = Self::unwrap_buffer(bias_hh)?;
        let hx_buf = Self::unwrap_buffer(hx)?;
        let dev = self.device(input_gates.device_ordinal())?;
        let (hy, ws) =
            crate::kernels::gpu_fused_gru_forward(ig, hg, bih, bhh, hx_buf, hidden_size, dev)
                .map_err(Self::map_gpu_err)?;
        let ord = input_gates.device_ordinal();
        Ok((Self::wrap_buffer(hy, ord), Self::wrap_buffer(ws, ord)))
    }

    fn synchronize(&self, device: usize) -> FerrotorchResult<()> {
        let dev = self.device(device)?;
        dev.stream()
            .synchronize()
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("CUDA synchronize failed: {e}"),
            })?;
        Ok(())
    }

    fn stream_count(&self, device: usize) -> usize {
        crate::stream::StreamPool::pool_size(device)
    }

    // -- Linalg f32 -----------------------------------------------------------

    fn matmul_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::blas::gpu_matmul_f32(a_buf, b_buf, m, k, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn matmul_f32_nt(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::blas::gpu_matmul_f32_nt(a_buf, b_buf, m, k, n, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    // -- Reduction f32 --------------------------------------------------------

    fn sum_f32(&self, a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_reduce_sum(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn prod_f32(&self, a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_reduce_prod(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn prod_backward_f32(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer(input)?;
        let grad_buf = Self::unwrap_buffer(grad_output)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_prod_backward_f32(input_buf, grad_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn prod_axis_f32(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_prod_axis(a_buf, outer, axis_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn prod_axis_backward_f32(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer(input)?;
        let grad_buf = Self::unwrap_buffer(grad_output)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_prod_axis_backward(
            input_buf, grad_buf, outer, axis_size, inner, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn std_var_axis_f32(
        &self,
        input: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
        correction: f64,
        take_sqrt: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_std_var_axis(
            input_buf, outer, axis_size, inner, correction, take_sqrt, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn std_var_axis_backward_f32(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        result: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
        correction: f64,
        take_sqrt: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer(input)?;
        let grad_buf = Self::unwrap_buffer(grad_output)?;
        let result_buf = Self::unwrap_buffer(result)?;
        let dev = self.device(input.device_ordinal())?;
        let grad = crate::kernels::gpu_std_var_axis_backward(
            input_buf, grad_buf, result_buf, outer, axis_size, inner, correction, take_sqrt, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(grad, input.device_ordinal()))
    }

    fn std_var_axis_bf16(
        &self,
        input: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
        correction: f64,
        take_sqrt: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_bf16(input)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::bf16::gpu_std_var_axis_bf16(
            input_buf, outer, axis_size, inner, correction, take_sqrt, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, input.device_ordinal()))
    }

    fn std_var_axis_backward_bf16(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        result: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
        correction: f64,
        take_sqrt: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_bf16(input)?;
        let grad_buf = Self::unwrap_buffer_bf16(grad_output)?;
        let result_buf = Self::unwrap_buffer_bf16(result)?;
        let dev = self.device(input.device_ordinal())?;
        let grad = crate::bf16::gpu_std_var_axis_backward_bf16(
            input_buf, grad_buf, result_buf, outer, axis_size, inner, correction, take_sqrt, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(grad, input.device_ordinal()))
    }

    fn logsumexp_axis_f32(
        &self,
        input: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(input.device_ordinal())?;
        let (sum, shift) =
            crate::kernels::gpu_logsumexp_sum_shift(input_buf, outer, axis_size, inner, dev)
                .map_err(Self::map_gpu_err)?;
        let log_sum = crate::kernels::gpu_log(&sum, dev).map_err(Self::map_gpu_err)?;
        let result = crate::kernels::gpu_add(&log_sum, &shift, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn nan_reduce_axis_f32(
        &self,
        input: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
        take_mean: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::nan_reductions::nan_reduce_axis_f32(
            input_buf, outer, axis_size, inner, take_mean, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn nan_reduce_axis_backward_f32(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
        take_mean: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer(input)?;
        let grad_buf = Self::unwrap_buffer(grad_output)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::nan_reductions::nan_reduce_axis_backward_f32(
            input_buf, grad_buf, outer, axis_size, inner, take_mean, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn min_f32(&self, a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_reduce_min(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn max_f32(&self, a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_reduce_max(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn extreme_backward_f32(
        &self,
        input: &GpuBufferHandle,
        extreme: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer(input)?;
        let extreme_buf = Self::unwrap_buffer(extreme)?;
        let grad_buf = Self::unwrap_buffer(grad_output)?;
        let dev = self.device(input.device_ordinal())?;
        let result =
            crate::kernels::gpu_extreme_backward_f32(input_buf, extreme_buf, grad_buf, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn masked_min_f32(
        &self,
        data: &GpuBufferHandle,
        mask_f: &GpuBufferHandle,
        _len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let d_buf = Self::unwrap_buffer(data)?;
        let m_buf = Self::unwrap_buffer(mask_f)?;
        let dev = self.device(data.device_ordinal())?;
        let result =
            crate::kernels::gpu_masked_reduce_min(d_buf, m_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, data.device_ordinal()))
    }

    fn masked_max_f32(
        &self,
        data: &GpuBufferHandle,
        mask_f: &GpuBufferHandle,
        _len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let d_buf = Self::unwrap_buffer(data)?;
        let m_buf = Self::unwrap_buffer(mask_f)?;
        let dev = self.device(data.device_ordinal())?;
        let result =
            crate::kernels::gpu_masked_reduce_max(d_buf, m_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, data.device_ordinal()))
    }

    // -- Linalg f64 (cuBLAS DGEMM) --------------------------------------------

    fn matmul_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::blas::gpu_matmul_f64(a_buf, b_buf, m, k, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn matmul_f64_nt(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::blas::gpu_matmul_f64_nt(a_buf, b_buf, m, k, n, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    // -- Vector matmul kernels (#816 / #817 / #818) --------------------------

    fn dot_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::blas::gpu_dot_f32(a_buf, b_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn dot_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::blas::gpu_dot_f64(a_buf, b_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn mv_f32(
        &self,
        a: &GpuBufferHandle,
        x: &GpuBufferHandle,
        m: usize,
        k: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let x_buf = Self::unwrap_buffer(x)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::blas::gpu_mv_f32(a_buf, x_buf, m, k, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn mv_f64(
        &self,
        a: &GpuBufferHandle,
        x: &GpuBufferHandle,
        m: usize,
        k: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let x_buf = Self::unwrap_buffer_f64(x)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::blas::gpu_mv_f64(a_buf, x_buf, m, k, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn vm_f32(
        &self,
        x: &GpuBufferHandle,
        b: &GpuBufferHandle,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let x_buf = Self::unwrap_buffer(x)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(x.device_ordinal())?;
        let result = crate::blas::gpu_vm_f32(x_buf, b_buf, k, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, x.device_ordinal()))
    }

    fn vm_f64(
        &self,
        x: &GpuBufferHandle,
        b: &GpuBufferHandle,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let x_buf = Self::unwrap_buffer_f64(x)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(x.device_ordinal())?;
        let result = crate::blas::gpu_vm_f64(x_buf, b_buf, k, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, x.device_ordinal()))
    }

    // -- Broadcast binary f32 -------------------------------------------------

    fn broadcast_add_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_broadcast_add(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn broadcast_sub_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_broadcast_sub(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn broadcast_mul_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_broadcast_mul(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn broadcast_div_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_broadcast_div(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn softmax_f32(
        &self,
        a: &GpuBufferHandle,
        rows: usize,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_softmax(a_buf, rows, cols, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn dropout_f32(
        &self,
        a: &GpuBufferHandle,
        threshold: u32,
        scale: f32,
        seed: u32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_dropout(a_buf, threshold, scale, seed, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn dropout_philox_f32(
        &self,
        a: &GpuBufferHandle,
        threshold: u32,
        scale: f32,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuRngState)> {
        let device_ordinal = a.device_ordinal();
        let n = a.len();

        // Snapshot the current RNG state and advance it.
        let rng_state = {
            let mut mgr = crate::rng::cuda_rng_manager().lock().map_err(|_| {
                FerrotorchError::InvalidArgument {
                    message: "failed to lock CUDA RNG manager".into(),
                }
            })?;
            let philox_gen = mgr.generator(device_ordinal);
            let state = philox_gen.get_state();
            // Advance by ceil(n/4) counters (each counter produces 4 u32 values)
            let counters_needed = n.div_ceil(4);
            philox_gen.advance(counters_needed as u64);
            state
        };

        // Use the Philox state as the seed for the dropout kernel.
        // We encode the Philox counter+seed into a u32 seed that the existing
        // dropout kernel can use. For full correctness on GPU, we should use
        // the Philox uniform kernel to generate the mask, then apply it.
        // However, for consistency between GPU forward and CPU backward mask
        // regeneration, we use the Philox state to deterministically derive a
        // seed for the existing kernel.
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(device_ordinal)?;

        // Use the Philox counter XOR seed as the dropout kernel's seed.
        // This gives us deterministic behavior tied to the Philox state.
        let derived_seed = (rng_state.counter ^ rng_state.seed) as u32;
        let result = crate::kernels::gpu_dropout(a_buf, threshold, scale, derived_seed, dev)
            .map_err(Self::map_gpu_err)?;

        let gpu_rng_state = GpuRngState::new(
            rng_state.counter,
            rng_state.seed,
            rng_state.offset,
            device_ordinal,
        );

        Ok((Self::wrap_buffer(result, device_ordinal), gpu_rng_state))
    }

    fn dropout_f64(
        &self,
        a: &GpuBufferHandle,
        threshold: u32,
        scale: f64,
        seed: u32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_dropout_f64(a_buf, threshold, scale, seed, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn dropout_philox_f64(
        &self,
        a: &GpuBufferHandle,
        threshold: u32,
        scale: f64,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuRngState)> {
        let device_ordinal = a.device_ordinal();
        let n = a.len();

        let rng_state = {
            let mut mgr = crate::rng::cuda_rng_manager().lock().map_err(|_| {
                FerrotorchError::InvalidArgument {
                    message: "failed to lock CUDA RNG manager".into(),
                }
            })?;
            let philox_gen = mgr.generator(device_ordinal);
            let state = philox_gen.get_state();
            let counters_needed = n.div_ceil(4);
            philox_gen.advance(counters_needed as u64);
            state
        };

        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(device_ordinal)?;
        let derived_seed = (rng_state.counter ^ rng_state.seed) as u32;
        let result = crate::kernels::gpu_dropout_f64(a_buf, threshold, scale, derived_seed, dev)
            .map_err(Self::map_gpu_err)?;

        let gpu_rng_state = GpuRngState::new(
            rng_state.counter,
            rng_state.seed,
            rng_state.offset,
            device_ordinal,
        );

        Ok((Self::wrap_buffer_f64(result, device_ordinal), gpu_rng_state))
    }

    fn transpose_2d_f32(
        &self,
        a: &GpuBufferHandle,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_transpose_2d(a_buf, m, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn permute_0213_f32(
        &self,
        a: &GpuBufferHandle,
        d0: usize,
        d1: usize,
        d2: usize,
        d3: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_permute_0213(a_buf, d0, d1, d2, d3, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
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
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::blas::gpu_bmm_f32(a_buf, b_buf, batch, m, k, n, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn broadcast_bmm_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_lead: &[usize],
        b_lead: &[usize],
        out_lead: &[usize],
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::blas::gpu_broadcast_bmm_f32(
            a_buf, b_buf, a_lead, b_lead, out_lead, m, k, n, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    /// Broadcast-bmm for `Tensor<bf16>` on CUDA — closes the GH#25 / local
    /// #1543 regression where 3D × 2D bf16 matmul fell through to the CPU
    /// `broadcast_matmul` round-trip and reported a 50× worse error than CPU
    /// bf16 against the f32 oracle.
    ///
    /// The strided-batched bf16 kernel
    /// (`gpu_matmul_bf16_bf16_strided_batched`) takes one
    /// `(stride_a, stride_b, batch_count)` triple per call. The trait method
    /// here supports the "single-run" broadcast patterns — those where one
    /// operand is either fully broadcast (its leading dims are empty) or
    /// exactly aligned to `out_lead`. That set covers all the cases the
    /// dispatcher in `ferrotorch-core/src/grad_fns/linalg.rs` routes through
    /// today (3D × 2D, 2D × 3D, ND × ND with matching leads) which were
    /// pre-fix going through the CPU bf16 round-trip. Less-uniform broadcast
    /// patterns (broadcast mid-axes, ragged leads) return `InvalidArgument`;
    /// the dispatcher detects this and skips the GPU path for that call,
    /// leaving the existing CPU fallback in place — no regression.
    #[cfg(feature = "cuda")]
    fn broadcast_bmm_bf16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_lead: &[usize],
        b_lead: &[usize],
        out_lead: &[usize],
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_bf16(a)?;
        let b_buf = Self::unwrap_buffer_bf16(b)?;
        let dev = self.device(a.device_ordinal())?;

        // Per-batch matrix element counts.
        let a_mat = m * k;
        let b_mat = k * n;
        let batch: usize = out_lead.iter().product();

        // Shape contracts: `a_buf.len() == product(a_lead) * a_mat` (1 when
        // `a_lead` is empty), similarly for `b_buf`. These mirror the f32
        // path's `validate_broadcast_shapes` invariants enforced by the
        // upstream `gpu_broadcast_bmm_f32` in `blas.rs`.
        let a_batch_count: usize = if a_lead.is_empty() {
            1
        } else {
            a_lead.iter().product()
        };
        let b_batch_count: usize = if b_lead.is_empty() {
            1
        } else {
            b_lead.iter().product()
        };
        let expected_a = a_batch_count * a_mat;
        let expected_b = b_batch_count * b_mat;
        if a_buf.len() != expected_a {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "broadcast_bmm_bf16: a buffer len {} != expected {} (a_lead={:?}, m={}, k={})",
                    a_buf.len(),
                    expected_a,
                    a_lead,
                    m,
                    k,
                ),
            });
        }
        if b_buf.len() != expected_b {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "broadcast_bmm_bf16: b buffer len {} != expected {} (b_lead={:?}, k={}, n={})",
                    b_buf.len(),
                    expected_b,
                    b_lead,
                    k,
                    n,
                ),
            });
        }

        // Zero-numel fast path.
        if batch == 0 || m == 0 || k == 0 || n == 0 {
            let zeros = dev
                .stream()
                .alloc_zeros::<u16>(batch * m * n)
                .map_err(|e| FerrotorchError::InvalidArgument {
                    message: format!("broadcast_bmm_bf16: alloc_zeros failed: {e}"),
                })?;
            return Ok(Self::wrap_buffer_bf16(zeros, a.device_ordinal()));
        }

        // Decide single-run stride encoding. `a` is single-run iff its lead is
        // empty (fully broadcast across all out_lead — stride 0) OR exactly
        // equals out_lead (per-batch contiguous — stride a_mat). Same for `b`.
        // These two cases cover every shape the dispatcher routes here today.
        let stride_a: usize = if a_lead.is_empty() {
            0
        } else if a_lead == out_lead {
            a_mat
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "broadcast_bmm_bf16: non-uniform broadcast pattern for a_lead={:?} out_lead={:?} not single-run-encodable",
                    a_lead, out_lead,
                ),
            });
        };
        let stride_b: usize = if b_lead.is_empty() {
            0
        } else if b_lead == out_lead {
            b_mat
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "broadcast_bmm_bf16: non-uniform broadcast pattern for b_lead={:?} out_lead={:?} not single-run-encodable",
                    b_lead, out_lead,
                ),
            });
        };

        let result = crate::blas::gpu_matmul_bf16_bf16_strided_batched(
            a_buf, b_buf, m, k, n, batch, stride_a, stride_b, 1.0, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    fn bmm_f16_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::blas::gpu_bmm_f16(a_buf, b_buf, batch, m, k, n, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    // -- bf16 mixed-precision (#518) -----------------------------------------

    fn matmul_bf16_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::blas::gpu_matmul_bf16(a_buf, b_buf, m, k, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn bmm_bf16_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::blas::gpu_bmm_bf16(a_buf, b_buf, batch, m, k, n, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn matmul_bf16_bf16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_bf16(a)?;
        let b_buf = Self::unwrap_buffer_bf16(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::blas::gpu_matmul_bf16_bf16(a_buf, b_buf, m, k, n, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    fn bmm_bf16_bf16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_bf16(a)?;
        let b_buf = Self::unwrap_buffer_bf16(b)?;
        let dev = self.device(a.device_ordinal())?;
        // Contiguous-batch strides: per-batch shapes A:[M,K], B:[K,N].
        let stride_a = m * k;
        let stride_b = k * n;
        let result = crate::blas::gpu_matmul_bf16_bf16_strided_batched(
            a_buf, b_buf, m, k, n, batch, stride_a, stride_b, 1.0, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    fn softmax_bf16_f32(
        &self,
        a: &GpuBufferHandle,
        rows: usize,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        // The bf16 softmax handle carries a CudaSlice<u16> (bf16 bit patterns).
        let buf = a.downcast_ref::<cudarc::driver::CudaSlice<u16>>().ok_or(
            FerrotorchError::InvalidArgument {
                message: "softmax_bf16_f32: GPU handle does not contain a CudaSlice<u16> (bf16)"
                    .into(),
            },
        )?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_softmax_bf16_f32(buf, rows, cols, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    // -- bf16 elementwise (#963) ---------------------------------------------

    fn add_bf16_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = a.downcast_ref::<cudarc::driver::CudaSlice<u16>>().ok_or(
            FerrotorchError::InvalidArgument {
                message: "add_bf16_f32: handle `a` does not contain CudaSlice<u16> (bf16)".into(),
            },
        )?;
        let b_buf = b.downcast_ref::<cudarc::driver::CudaSlice<u16>>().ok_or(
            FerrotorchError::InvalidArgument {
                message: "add_bf16_f32: handle `b` does not contain CudaSlice<u16> (bf16)".into(),
            },
        )?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_add_bf16_f32(a_buf, b_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn sub_bf16_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = a.downcast_ref::<cudarc::driver::CudaSlice<u16>>().ok_or(
            FerrotorchError::InvalidArgument {
                message: "sub_bf16_f32: handle `a` does not contain CudaSlice<u16> (bf16)".into(),
            },
        )?;
        let b_buf = b.downcast_ref::<cudarc::driver::CudaSlice<u16>>().ok_or(
            FerrotorchError::InvalidArgument {
                message: "sub_bf16_f32: handle `b` does not contain CudaSlice<u16> (bf16)".into(),
            },
        )?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_sub_bf16_f32(a_buf, b_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn mul_bf16_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = a.downcast_ref::<cudarc::driver::CudaSlice<u16>>().ok_or(
            FerrotorchError::InvalidArgument {
                message: "mul_bf16_f32: handle `a` does not contain CudaSlice<u16> (bf16)".into(),
            },
        )?;
        let b_buf = b.downcast_ref::<cudarc::driver::CudaSlice<u16>>().ok_or(
            FerrotorchError::InvalidArgument {
                message: "mul_bf16_f32: handle `b` does not contain CudaSlice<u16> (bf16)".into(),
            },
        )?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_mul_bf16_f32(a_buf, b_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn div_bf16_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = a.downcast_ref::<cudarc::driver::CudaSlice<u16>>().ok_or(
            FerrotorchError::InvalidArgument {
                message: "div_bf16_f32: handle `a` does not contain CudaSlice<u16> (bf16)".into(),
            },
        )?;
        let b_buf = b.downcast_ref::<cudarc::driver::CudaSlice<u16>>().ok_or(
            FerrotorchError::InvalidArgument {
                message: "div_bf16_f32: handle `b` does not contain CudaSlice<u16> (bf16)".into(),
            },
        )?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_div_bf16_f32(a_buf, b_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    // -- bf16 reductions (#963) ----------------------------------------------

    fn sum_axis_bf16_f32(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = a.downcast_ref::<cudarc::driver::CudaSlice<u16>>().ok_or(
            FerrotorchError::InvalidArgument {
                message: "sum_axis_bf16_f32: handle does not contain CudaSlice<u16> (bf16)".into(),
            },
        )?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_sum_axis_bf16_f32(a_buf, outer, axis_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn mean_axis_bf16_f32(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = a.downcast_ref::<cudarc::driver::CudaSlice<u16>>().ok_or(
            FerrotorchError::InvalidArgument {
                message: "mean_axis_bf16_f32: handle does not contain CudaSlice<u16> (bf16)".into(),
            },
        )?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_mean_axis_bf16_f32(a_buf, outer, axis_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    // -- bf16 activations (#963) ---------------------------------------------

    fn relu_bf16_f32(&self, a: &GpuBufferHandle, n: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = a.downcast_ref::<cudarc::driver::CudaSlice<u16>>().ok_or(
            FerrotorchError::InvalidArgument {
                message: "relu_bf16_f32: handle does not contain CudaSlice<u16> (bf16)".into(),
            },
        )?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_relu_bf16_f32(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn sigmoid_bf16_f32(&self, a: &GpuBufferHandle, n: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = a.downcast_ref::<cudarc::driver::CudaSlice<u16>>().ok_or(
            FerrotorchError::InvalidArgument {
                message: "sigmoid_bf16_f32: handle does not contain CudaSlice<u16> (bf16)".into(),
            },
        )?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_sigmoid_bf16_f32(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn gelu_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_gelu(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn gelu_tanh_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_tanh(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn gelu_erf_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_erf(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn layernorm_f32(
        &self,
        input: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        bias: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer(input)?;
        let w_buf = Self::unwrap_buffer(weight)?;
        let b_buf = Self::unwrap_buffer(bias)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_layernorm(in_buf, w_buf, b_buf, rows, cols, eps, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn group_norm_f32(
        &self,
        input: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        bias: &GpuBufferHandle,
        batch: usize,
        channels: usize,
        groups: usize,
        hw: usize,
        eps: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer(input)?;
        let w_buf = Self::unwrap_buffer(weight)?;
        let b_buf = Self::unwrap_buffer(bias)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::group_norm::gpu_group_norm_f32(
            in_buf, w_buf, b_buf, batch, channels, groups, hw, eps, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn batch_norm_f32(
        &self,
        input: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        bias: &GpuBufferHandle,
        mean: &GpuBufferHandle,
        var: &GpuBufferHandle,
        batch: usize,
        channels: usize,
        hw: usize,
        eps: f32,
        training: bool,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        let in_buf = Self::unwrap_buffer(input)?;
        let w_buf = Self::unwrap_buffer(weight)?;
        let b_buf = Self::unwrap_buffer(bias)?;
        let dev = self.device(input.device_ordinal())?;
        // The kernel reads (eval) or writes (training) per-channel mean/var, so
        // it needs owned mutable buffers. Stage the (small `[channels]`) input
        // stats through host memory into fresh device buffers so the caller's
        // handles are never mutated in place; the (possibly updated) stats are
        // returned to the caller as new handles.
        let mean_host = crate::transfer::gpu_to_cpu(Self::unwrap_buffer(mean)?, dev)
            .map_err(Self::map_gpu_err)?;
        let var_host = crate::transfer::gpu_to_cpu(Self::unwrap_buffer(var)?, dev)
            .map_err(Self::map_gpu_err)?;
        let mut mean_buf =
            crate::transfer::cpu_to_gpu(&mean_host, dev).map_err(Self::map_gpu_err)?;
        let mut var_buf = crate::transfer::cpu_to_gpu(&var_host, dev).map_err(Self::map_gpu_err)?;
        let result = crate::group_norm::gpu_batch_norm_f32(
            in_buf,
            w_buf,
            b_buf,
            &mut mean_buf,
            &mut var_buf,
            batch,
            channels,
            hw,
            eps,
            training,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        let ord = input.device_ordinal();
        Ok((
            Self::wrap_buffer(result, ord),
            Self::wrap_buffer(mean_buf, ord),
            Self::wrap_buffer(var_buf, ord),
        ))
    }

    fn batch_norm_backward_f32(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        running_mean: &GpuBufferHandle,
        running_var: &GpuBufferHandle,
        batch: usize,
        channels: usize,
        hw: usize,
        eps: f32,
        training: bool,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        let in_buf = Self::unwrap_buffer(input)?;
        let go_buf = Self::unwrap_buffer(grad_output)?;
        let w_buf = Self::unwrap_buffer(weight)?;
        let rm_buf = Self::unwrap_buffer(running_mean)?;
        let rv_buf = Self::unwrap_buffer(running_var)?;
        let dev = self.device(input.device_ordinal())?;
        let (gi, gw, gb) = crate::group_norm::gpu_batch_norm_backward_f32(
            in_buf, go_buf, w_buf, rm_buf, rv_buf, batch, channels, hw, eps, training, dev,
        )
        .map_err(Self::map_gpu_err)?;
        let ord = input.device_ordinal();
        Ok((
            Self::wrap_buffer(gi, ord),
            Self::wrap_buffer(gw, ord),
            Self::wrap_buffer(gb, ord),
        ))
    }

    fn local_response_norm_f32(
        &self,
        input: &GpuBufferHandle,
        batch: usize,
        channels: usize,
        spatial: usize,
        size: usize,
        alpha: f32,
        beta: f32,
        k: f32,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let in_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(input.device_ordinal())?;
        let (out, denom) = crate::group_norm::gpu_local_response_norm_f32(
            in_buf, batch, channels, spatial, size, alpha, beta, k, dev,
        )
        .map_err(Self::map_gpu_err)?;
        let ord = input.device_ordinal();
        Ok((Self::wrap_buffer(out, ord), Self::wrap_buffer(denom, ord)))
    }

    fn local_response_norm_backward_f32(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        denom: &GpuBufferHandle,
        batch: usize,
        channels: usize,
        spatial: usize,
        size: usize,
        alpha: f32,
        beta: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer(input)?;
        let go_buf = Self::unwrap_buffer(grad_output)?;
        let dn_buf = Self::unwrap_buffer(denom)?;
        let dev = self.device(input.device_ordinal())?;
        let gi = crate::group_norm::gpu_local_response_norm_backward_f32(
            in_buf, go_buf, dn_buf, batch, channels, spatial, size, alpha, beta, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(gi, input.device_ordinal()))
    }

    fn softmax2d_f32(
        &self,
        input: &GpuBufferHandle,
        n: usize,
        c: usize,
        hw: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::group_norm::gpu_softmax2d_f32(in_buf, n, c, hw, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn rmsnorm_f32(
        &self,
        input: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer(input)?;
        let w_buf = Self::unwrap_buffer(weight)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_rmsnorm(in_buf, w_buf, rows, cols, eps, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn rmsnorm_backward_f32(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let in_buf = Self::unwrap_buffer(input)?;
        let go_buf = Self::unwrap_buffer(grad_output)?;
        let w_buf = Self::unwrap_buffer(weight)?;
        let dev = self.device(input.device_ordinal())?;
        let (gi, gw) =
            crate::kernels::gpu_rmsnorm_backward(in_buf, go_buf, w_buf, rows, cols, eps, dev)
                .map_err(Self::map_gpu_err)?;
        let ordinal = input.device_ordinal();
        Ok((
            Self::wrap_buffer(gi, ordinal),
            Self::wrap_buffer(gw, ordinal),
        ))
    }

    fn slice_write_f32(
        &self,
        src: &GpuBufferHandle,
        dst: &mut GpuBufferHandle,
        n_batch: usize,
        d: usize,
        max_len: usize,
        pos: usize,
    ) -> FerrotorchResult<()> {
        let src_buf = Self::unwrap_buffer(src)?;
        let dst_buf =
            dst.downcast_mut::<CudaBuffer<f32>>()
                .ok_or(FerrotorchError::InvalidArgument {
                    message: "slice_write_f32: dst is not CudaBuffer<f32>".into(),
                })?;
        let dev = self.device(src.device_ordinal())?;
        crate::kernels::gpu_slice_write(src_buf, dst_buf, n_batch, d, max_len, pos, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(())
    }

    fn slice_read_f32(
        &self,
        src: &GpuBufferHandle,
        n_batch: usize,
        d: usize,
        len: usize,
        max_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let src_buf = Self::unwrap_buffer(src)?;
        let dev = self.device(src.device_ordinal())?;
        let result = crate::kernels::gpu_slice_read(src_buf, n_batch, d, len, max_len, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, src.device_ordinal()))
    }

    fn embed_lookup_f32(
        &self,
        idx: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let idx_buf = Self::unwrap_buffer(idx)?;
        let w_buf = Self::unwrap_buffer(weight)?;
        let dev = self.device(idx.device_ordinal())?;
        let result =
            crate::kernels::gpu_embed_lookup(idx_buf, w_buf, d, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, idx.device_ordinal()))
    }

    fn embed_lookup_batch_f32(
        &self,
        indices: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        n: usize,
        d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let idx_buf = Self::unwrap_buffer(indices)?;
        let w_buf = Self::unwrap_buffer(weight)?;
        let dev = self.device(indices.device_ordinal())?;
        let result = crate::kernels::gpu_embed_lookup_batch(idx_buf, w_buf, n, d, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, indices.device_ordinal()))
    }

    fn scatter_add_rows_f32(
        &self,
        grad_output: &GpuBufferHandle,
        indices: &GpuBufferHandle,
        num_embeddings: usize,
        d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let go_buf = Self::unwrap_buffer(grad_output)?;
        let idx_buf = Self::unwrap_buffer(indices)?;
        let dev = self.device(grad_output.device_ordinal())?;
        let result = crate::kernels::gpu_scatter_add_rows(go_buf, idx_buf, num_embeddings, d, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad_output.device_ordinal()))
    }

    fn scale_f32(&self, a: &GpuBufferHandle, scalar: f32) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_scale(a_buf, scalar, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn relu_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let input_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_relu_backward(grad_buf, input_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn abs_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let input_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_abs_backward(grad_buf, input_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn fill_f32(&self, n: usize, scalar: f32, ordinal: usize) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(ordinal)?;
        let result = crate::kernels::gpu_fill_f32(n, scalar, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, ordinal))
    }

    fn fill_f64(&self, n: usize, scalar: f64, ordinal: usize) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(ordinal)?;
        let result = crate::kernels::gpu_fill_f64(n, scalar, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, ordinal))
    }

    fn fill_bf16_bf16(
        &self,
        n: usize,
        scalar: f32,
        ordinal: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(ordinal)?;
        let result = crate::bf16::gpu_fill_bf16(n, scalar, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, ordinal))
    }

    fn fill_f16(&self, n: usize, scalar: f32, ordinal: usize) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(ordinal)?;
        let result = crate::f16::gpu_fill_f16(n, scalar, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, ordinal))
    }

    fn gelu_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let input_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_backward(grad_buf, input_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn gelu_backward_tanh_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let input_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_backward_tanh(grad_buf, input_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn gelu_backward_erf_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let input_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_backward_erf(grad_buf, input_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn cumsum_f32(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_cumsum(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn cumprod_f32(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_cumprod(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn cummax_f32(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (vals, idxs) = crate::kernels::gpu_cummax(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((
            Self::wrap_buffer(vals, ord),
            Self::wrap_buffer_i64(idxs, ord),
        ))
    }

    fn cummin_f32(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (vals, idxs) = crate::kernels::gpu_cummin(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((
            Self::wrap_buffer(vals, ord),
            Self::wrap_buffer_i64(idxs, ord),
        ))
    }

    fn max_with_dim_f32(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (vals, idxs) = crate::kernels::gpu_max_with_dim(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((
            Self::wrap_buffer(vals, ord),
            Self::wrap_buffer_i64(idxs, ord),
        ))
    }

    fn min_with_dim_f32(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (vals, idxs) = crate::kernels::gpu_min_with_dim(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((
            Self::wrap_buffer(vals, ord),
            Self::wrap_buffer_i64(idxs, ord),
        ))
    }

    fn logcumsumexp_f32(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_logcumsumexp(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn roll_f32(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
        shift_norm: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::roll::gpu_roll_f32(a_buf, outer, dim_size, inner, shift_norm, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn roll_f64(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
        shift_norm: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::roll::gpu_roll_f64(a_buf, outer, dim_size, inner, shift_norm, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    // -- Triangular masks: triu / tril (#1545 / sub #1535) -------------------
    //
    // Gated `#[cfg(feature = "cuda")]` (the `triangular` module is cuda-only,
    // mirroring `reduce_arg`); when the feature is off, the `GpuBackend`
    // trait default `NotImplementedOnCuda` is used instead.

    #[cfg(feature = "cuda")]
    fn triu_f32(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        rows: usize,
        cols: usize,
        k: i64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::triangular::gpu_triu_f32(a_buf, batch, rows, cols, k, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn tril_f32(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        rows: usize,
        cols: usize,
        k: i64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::triangular::gpu_tril_f32(a_buf, batch, rows, cols, k, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn triu_f64(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        rows: usize,
        cols: usize,
        k: i64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::triangular::gpu_triu_f64(a_buf, batch, rows, cols, k, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn tril_f64(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        rows: usize,
        cols: usize,
        k: i64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::triangular::gpu_tril_f64(a_buf, batch, rows, cols, k, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    // -- Diagonal: diag_embed / diag_extract (#1545 / sub #1535) -------------
    //
    // Gated `#[cfg(feature = "cuda")]` (the `diag` module is cuda-only,
    // mirroring `triangular`); when the feature is off, the trait default
    // `NotImplementedOnCuda` is used instead. Non-test consumer: the
    // `is_cuda()` branch of `diag`/`diagflat` in `ops::tensor_ops`.

    #[cfg(feature = "cuda")]
    fn diag_embed_f32(
        &self,
        a: &GpuBufferHandle,
        n: usize,
        k: i64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::diag::gpu_diag_embed_f32(a_buf, n, k, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn diag_embed_f64(
        &self,
        a: &GpuBufferHandle,
        n: usize,
        k: i64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::diag::gpu_diag_embed_f64(a_buf, n, k, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn diag_extract_f32(
        &self,
        a: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        k: i64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::diag::gpu_diag_extract_f32(a_buf, rows, cols, k, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn diag_extract_f64(
        &self,
        a: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        k: i64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::diag::gpu_diag_extract_f64(a_buf, rows, cols, k, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    // -- Pairwise distance: cdist (#1545 / sub #1535) ------------------------

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn cdist_f32(
        &self,
        x1: &GpuBufferHandle,
        x2: &GpuBufferHandle,
        b: usize,
        p_dim: usize,
        r_dim: usize,
        m: usize,
        p: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let x1_buf = Self::unwrap_buffer(x1)?;
        let x2_buf = Self::unwrap_buffer(x2)?;
        let dev = self.device(x1.device_ordinal())?;
        let result = crate::distance::gpu_cdist_f32(x1_buf, x2_buf, b, p_dim, r_dim, m, p, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, x1.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn cdist_f64(
        &self,
        x1: &GpuBufferHandle,
        x2: &GpuBufferHandle,
        b: usize,
        p_dim: usize,
        r_dim: usize,
        m: usize,
        p: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let x1_buf = Self::unwrap_buffer_f64(x1)?;
        let x2_buf = Self::unwrap_buffer_f64(x2)?;
        let dev = self.device(x1.device_ordinal())?;
        let result = crate::distance::gpu_cdist_f64(x1_buf, x2_buf, b, p_dim, r_dim, m, p, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, x1.device_ordinal()))
    }

    // -- Orthogonal-polynomial special functions (#1545 / #1533) -------------

    fn chebyshev_poly_f32(
        &self,
        a: &GpuBufferHandle,
        n: usize,
        seed_a: f32,
        seed_b: f32,
        shift: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::special::gpu_chebyshev_poly_f32(a_buf, n, seed_a, seed_b, shift, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn chebyshev_poly_f64(
        &self,
        a: &GpuBufferHandle,
        n: usize,
        seed_a: f64,
        seed_b: f64,
        shift: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::special::gpu_chebyshev_poly_f64(a_buf, n, seed_a, seed_b, shift, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn hermite_h_poly_f32(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::special::gpu_hermite_h_poly_f32(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn hermite_h_poly_f64(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::special::gpu_hermite_h_poly_f64(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn hermite_he_poly_f32(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::special::gpu_hermite_he_poly_f32(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn hermite_he_poly_f64(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::special::gpu_hermite_he_poly_f64(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn laguerre_poly_f32(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::special::gpu_laguerre_poly_f32(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn laguerre_poly_f64(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::special::gpu_laguerre_poly_f64(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn legendre_poly_f32(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::special::gpu_legendre_poly_f32(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn legendre_poly_f64(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::special::gpu_legendre_poly_f64(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    // -- Normal-distribution trio: entr / ndtr / ndtri (#1651, batch 1) ------
    //
    // f32 runs on-device (PTX in `crate::special`). f64 returns
    // `NotImplementedOnCuda`: base PTX has no `lg2.approx.f64` / `ex2.approx.f64`,
    // so the f64 log/exp these transcendentals need cannot be evaluated at f64
    // precision on-device — and silently bouncing f64 through the host would
    // violate R-CODE-4. The f64 CUDA path is honestly unimplemented (the CPU
    // `pub fn entr/ndtr/ndtri` covers f64 for host tensors).

    fn entr_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::special::gpu_entr_f32(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn entr_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "entr_f64" })
    }

    fn ndtr_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::special::gpu_ndtr_f32(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn ndtr_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "ndtr_f64" })
    }

    fn ndtri_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::special::gpu_ndtri_f32(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn ndtri_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "ndtri_f64" })
    }

    // Modified-Bessel-I family (#1651 batch 2). f32 runs on-device; f64 ->
    // NotImplementedOnCuda (base PTX has no lg2/ex2.approx.f64). bf16/f16 are
    // rejected in `special_gpu_simple` before reaching here.

    fn i0_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::special::gpu_i0_f32(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn i0_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "i0_f64" })
    }

    fn i0e_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::special::gpu_i0e_f32(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn i0e_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "i0e_f64" })
    }

    fn i1_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::special::gpu_i1_f32(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn i1_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "i1_f64" })
    }

    fn i1e_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::special::gpu_i1e_f32(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn i1e_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "i1e_f64" })
    }

    fn spherical_bessel_j0_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::special::gpu_spherical_bessel_j0_f32(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn spherical_bessel_j0_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "spherical_bessel_j0_f64",
        })
    }

    // Modified-Bessel-K family (#1651 batch 3b). f32 runs on-device; f64 ->
    // NotImplementedOnCuda (base PTX has no lg2/ex2.approx.f64). bf16/f16 are
    // rejected in `special_gpu_simple` before reaching here.

    fn modified_bessel_k0_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::special::gpu_modified_bessel_k0_f32(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn modified_bessel_k0_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "modified_bessel_k0_f64",
        })
    }

    fn scaled_modified_bessel_k0_f32(
        &self,
        a: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::special::gpu_scaled_modified_bessel_k0_f32(a_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn scaled_modified_bessel_k0_f64(
        &self,
        _a: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "scaled_modified_bessel_k0_f64",
        })
    }

    fn modified_bessel_k1_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::special::gpu_modified_bessel_k1_f32(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn modified_bessel_k1_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "modified_bessel_k1_f64",
        })
    }

    fn scaled_modified_bessel_k1_f32(
        &self,
        a: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::special::gpu_scaled_modified_bessel_k1_f32(a_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn scaled_modified_bessel_k1_f64(
        &self,
        _a: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda {
            op: "scaled_modified_bessel_k1_f64",
        })
    }

    // Airy Ai + Hurwitz zeta (#1651 GPU tail). f32 runs on-device; f64 ->
    // NotImplementedOnCuda (base PTX has no lg2/ex2.approx.f64). bf16/f16 are
    // rejected in the core dispatch before reaching here.

    fn airy_ai_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::special::gpu_airy_ai_f32(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn airy_ai_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "airy_ai_f64" })
    }

    fn zeta_f32(
        &self,
        x: &GpuBufferHandle,
        q: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let x_buf = Self::unwrap_buffer(x)?;
        let q_buf = Self::unwrap_buffer(q)?;
        let dev = self.device(x.device_ordinal())?;
        let result = crate::special::gpu_zeta_f32(x_buf, q_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, x.device_ordinal()))
    }

    fn zeta_f64(
        &self,
        _x: &GpuBufferHandle,
        _q: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::NotImplementedOnCuda { op: "zeta_f64" })
    }

    fn clamp_f32(
        &self,
        a: &GpuBufferHandle,
        min_val: f32,
        max_val: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_clamp(a_buf, min_val, max_val, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn clamp_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
        min_val: f32,
        max_val: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer(grad)?;
        let i_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_clamp_backward(g_buf, i_buf, min_val, max_val, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn silu_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_silu(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn silu_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let input_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_silu_backward(grad_buf, input_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn elu_f32(&self, a: &GpuBufferHandle, alpha: f32) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_elu(a_buf, alpha, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn elu_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
        alpha: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let input_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_elu_backward(grad_buf, input_buf, alpha, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn mish_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_mish(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn mish_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let input_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_mish_backward(grad_buf, input_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn log_softmax_f32(
        &self,
        a: &GpuBufferHandle,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_log_softmax(a_buf, cols, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn log_softmax_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        output: &GpuBufferHandle,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let output_buf = Self::unwrap_buffer(output)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_log_softmax_backward(grad_buf, output_buf, cols, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn index_select_1d_f32(
        &self,
        input: &GpuBufferHandle,
        indices: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer(input)?;
        let idx_buf = Self::unwrap_buffer(indices)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_index_select_1d(input_buf, idx_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn scatter_add_1d_f32(
        &self,
        grad_output: &GpuBufferHandle,
        indices: &GpuBufferHandle,
        input_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let go_buf = Self::unwrap_buffer(grad_output)?;
        let idx_buf = Self::unwrap_buffer(indices)?;
        let dev = self.device(grad_output.device_ordinal())?;
        let result = crate::kernels::gpu_scatter_add_1d(go_buf, idx_buf, input_len, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad_output.device_ordinal()))
    }

    fn index_select_dim_f32(
        &self,
        input: &GpuBufferHandle,
        indices: &GpuBufferHandle,
        outer: usize,
        in_dim_size: usize,
        out_dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer(input)?;
        let idx_buf = Self::unwrap_buffer(indices)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_index_select_dim(
            input_buf,
            idx_buf,
            outer,
            in_dim_size,
            out_dim_size,
            inner,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn masked_fill_f32(
        &self,
        input: &GpuBufferHandle,
        mask: &GpuBufferHandle,
        value: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer(input)?;
        let mask_buf = Self::unwrap_buffer(mask)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_masked_fill(input_buf, mask_buf, value, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn masked_zero_f32(
        &self,
        grad: &GpuBufferHandle,
        mask: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let mask_buf = Self::unwrap_buffer(mask)?;
        let dev = self.device(grad.device_ordinal())?;
        let result =
            crate::kernels::gpu_masked_zero(grad_buf, mask_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn sigmoid_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let output_buf = Self::unwrap_buffer(output)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_sigmoid_backward(grad_buf, output_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn tanh_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let output_buf = Self::unwrap_buffer(output)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_tanh_backward(grad_buf, output_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn softmax_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        output: &GpuBufferHandle,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let output_buf = Self::unwrap_buffer(output)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_softmax_backward(grad_buf, output_buf, cols, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn layernorm_backward_f32(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        let in_buf = Self::unwrap_buffer(input)?;
        let go_buf = Self::unwrap_buffer(grad_output)?;
        let w_buf = Self::unwrap_buffer(weight)?;
        let dev = self.device(input.device_ordinal())?;
        let (gi, gw, gb) =
            crate::kernels::gpu_layernorm_backward(in_buf, go_buf, w_buf, rows, cols, eps, dev)
                .map_err(Self::map_gpu_err)?;
        let ordinal = input.device_ordinal();
        Ok((
            Self::wrap_buffer(gi, ordinal),
            Self::wrap_buffer(gw, ordinal),
            Self::wrap_buffer(gb, ordinal),
        ))
    }

    fn sum_axis_f32(
        &self,
        a: &GpuBufferHandle,
        shape: &[usize],
        axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let outer: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
        let result = crate::kernels::gpu_sum_axis(a_buf, outer, axis_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn min_axis_f32(
        &self,
        a: &GpuBufferHandle,
        shape: &[usize],
        axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let outer: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
        let result =
            crate::kernels::gpu_extreme_axis_f32(a_buf, outer, axis_size, inner, false, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn max_axis_f32(
        &self,
        a: &GpuBufferHandle,
        shape: &[usize],
        axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let outer: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
        let result =
            crate::kernels::gpu_extreme_axis_f32(a_buf, outer, axis_size, inner, true, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn extreme_axis_backward_f32(
        &self,
        input: &GpuBufferHandle,
        result: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        shape: &[usize],
        axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer(input)?;
        let result_buf = Self::unwrap_buffer(result)?;
        let grad_buf = Self::unwrap_buffer(grad_output)?;
        let dev = self.device(input.device_ordinal())?;
        let outer: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
        let out = crate::kernels::gpu_extreme_axis_backward_f32(
            input_buf, result_buf, grad_buf, outer, axis_size, inner, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, input.device_ordinal()))
    }

    fn matmul_f16_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::blas::gpu_matmul_f16(a_buf, b_buf, m, k, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn rand_uniform_f32(&self, numel: usize) -> FerrotorchResult<GpuBufferHandle> {
        // On-device generation: `gpu_philox_uniform` snapshots and advances the
        // per-device Philox counter inside the CudaRngManager (same pattern as
        // `dropout_philox_f32`), launches the PHILOX_UNIFORM_PTX kernel, and
        // returns a `CudaBuffer<f32>` filled on the GPU — no CPU round trip.
        // Mirrors `torch.rand(size, device='cuda')` =
        // `at::empty(...).uniform_(0,1)` at TensorFactories.cpp:1075-1076.
        let dev = self.default_device()?;
        let buf = crate::rng::gpu_philox_uniform(numel, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(buf, dev.ordinal()))
    }

    fn rand_uniform_f64(&self, numel: usize) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.default_device()?;
        let buf = crate::rng::gpu_philox_uniform_f64(numel, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(buf, dev.ordinal()))
    }

    fn rand_uniform_f16(&self, numel: usize) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.default_device()?;
        let buf = crate::rng::gpu_philox_uniform_f16(numel, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(buf, dev.ordinal()))
    }

    fn rand_uniform_bf16(&self, numel: usize) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.default_device()?;
        let buf = crate::rng::gpu_philox_uniform_bf16(numel, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(buf, dev.ordinal()))
    }

    fn randn_normal_f32(&self, numel: usize) -> FerrotorchResult<GpuBufferHandle> {
        // f32 standard-normal counterpart of `rand_uniform_f32`, via the
        // Box-Muller PHILOX_NORMAL_PTX kernel. Mirrors
        // `torch.randn(size, device='cuda')` = `at::empty(...).normal_(0,1)`.
        let dev = self.default_device()?;
        let buf = crate::rng::gpu_philox_normal(numel, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(buf, dev.ordinal()))
    }

    fn randn_normal_f64(&self, numel: usize) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.default_device()?;
        let buf = crate::rng::gpu_philox_normal_f64(numel, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(buf, dev.ordinal()))
    }

    fn randn_normal_f16(&self, numel: usize) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.default_device()?;
        let buf = crate::rng::gpu_philox_normal_f16(numel, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(buf, dev.ordinal()))
    }

    fn randn_normal_bf16(&self, numel: usize) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.default_device()?;
        let buf = crate::rng::gpu_philox_normal_bf16(numel, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(buf, dev.ordinal()))
    }

    fn manual_seed_gpu(&self, seed: u64) -> FerrotorchResult<()> {
        // `torch.cuda.manual_seed_all` analog (torch/cuda/random.py:112).
        // Called by `ferrotorch_core::manual_seed` after seeding the CPU
        // generator, mirroring `torch.manual_seed` -> `cuda.manual_seed_all`
        // (torch/random.py:67).
        let mut mgr = crate::rng::cuda_rng_manager().lock().map_err(|_| {
            FerrotorchError::InvalidArgument {
                message: "failed to lock CUDA RNG manager".into(),
            }
        })?;
        mgr.manual_seed_all(seed);
        Ok(())
    }

    fn save_rng_state(&self, device: usize) -> FerrotorchResult<GpuRngState> {
        let mut mgr = crate::rng::cuda_rng_manager().lock().map_err(|_| {
            FerrotorchError::InvalidArgument {
                message: "failed to lock CUDA RNG manager".into(),
            }
        })?;
        let state = mgr.get_rng_state(device);
        Ok(GpuRngState::new(
            state.counter,
            state.seed,
            state.offset,
            device,
        ))
    }

    fn restore_rng_state(&self, state: GpuRngState) -> FerrotorchResult<()> {
        let mut mgr = crate::rng::cuda_rng_manager().lock().map_err(|_| {
            FerrotorchError::InvalidArgument {
                message: "failed to lock CUDA RNG manager".into(),
            }
        })?;
        let philox =
            crate::rng::PhiloxState::from_parts(state.counter(), state.seed(), state.offset())
                .map_err(Self::map_gpu_err)?;
        mgr.set_rng_state(state.device(), philox);
        Ok(())
    }

    fn strided_split_f32(
        &self,
        input: &GpuBufferHandle,
        total_along_axis: usize,
        split_offset: usize,
        split_size: usize,
        inner_size: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_strided_split(
            in_buf,
            total_along_axis,
            split_offset,
            split_size,
            inner_size,
            n,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn strided_copy_f32(
        &self,
        input: &GpuBufferHandle,
        out_shape: &[usize],
        src_strides: &[isize],
        src_offset: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(input.device_ordinal())?;
        let result =
            crate::kernels::gpu_strided_copy(in_buf, out_shape, src_strides, src_offset, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn strided_copy_f64(
        &self,
        input: &GpuBufferHandle,
        out_shape: &[usize],
        src_strides: &[isize],
        src_offset: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(input.device_ordinal())?;
        let result =
            crate::kernels::gpu_strided_copy_f64(in_buf, out_shape, src_strides, src_offset, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    fn strided_copy_u16(
        &self,
        input: &GpuBufferHandle,
        out_shape: &[usize],
        src_strides: &[isize],
        src_offset: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = match input.dtype() {
            DType::F16 => Self::unwrap_buffer_f16(input)?,
            DType::BF16 => Self::unwrap_buffer_bf16(input)?,
            other => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("strided_copy_u16 expected F16/BF16, got {other}"),
                });
            }
        };
        let dev = self.device(input.device_ordinal())?;
        let result =
            crate::kernels::gpu_strided_copy_u16(in_buf, out_shape, src_strides, src_offset, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(match input.dtype() {
            DType::F16 => Self::wrap_buffer_f16(result, input.device_ordinal()),
            DType::BF16 => Self::wrap_buffer_bf16(result, input.device_ordinal()),
            _ => unreachable!("dtype checked above"),
        })
    }

    fn strided_scatter_f32(
        &self,
        src: &GpuBufferHandle,
        dst: &mut GpuBufferHandle,
        view_shape: &[usize],
        dst_strides: &[isize],
        dst_offset: usize,
    ) -> FerrotorchResult<()> {
        let ord = src.device_ordinal();
        if dst.device_ordinal() != ord {
            return Err(FerrotorchError::DeviceMismatch {
                expected: ferrotorch_core::Device::Cuda(ord),
                got: ferrotorch_core::Device::Cuda(dst.device_ordinal()),
            });
        }
        let src_buf_ptr = Self::unwrap_buffer(src)? as *const CudaBuffer<f32>;
        let dst_buf = Self::unwrap_buffer_mut(dst)?;
        let dev = self.device(ord)?;
        // SAFETY: `src` and `dst` are distinct GpuBufferHandles supplied
        // by the caller; the borrow checker forbids overlapping &/&mut
        // through the same handle, and CudaBuffer<f32> doesn't share
        // mutable state with anything reachable from the &CudaBuffer
        // pointer. The `*const CudaBuffer<f32>` is reborrowed as `&` for
        // the kernel call only.
        let src_ref = unsafe { &*src_buf_ptr };
        crate::kernels::gpu_strided_scatter(
            src_ref,
            dst_buf,
            view_shape,
            dst_strides,
            dst_offset,
            dev,
        )
        .map_err(Self::map_gpu_err)
    }

    fn strided_scatter_f64(
        &self,
        src: &GpuBufferHandle,
        dst: &mut GpuBufferHandle,
        view_shape: &[usize],
        dst_strides: &[isize],
        dst_offset: usize,
    ) -> FerrotorchResult<()> {
        let ord = src.device_ordinal();
        if dst.device_ordinal() != ord {
            return Err(FerrotorchError::DeviceMismatch {
                expected: ferrotorch_core::Device::Cuda(ord),
                got: ferrotorch_core::Device::Cuda(dst.device_ordinal()),
            });
        }
        let src_buf_ptr = Self::unwrap_buffer_f64(src)? as *const CudaBuffer<f64>;
        let dst_buf = Self::unwrap_buffer_f64_mut(dst)?;
        let dev = self.device(ord)?;
        // SAFETY:
        // - `src` and `dst` are distinct `GpuBufferHandle` parameters
        //   supplied by the caller. The borrow checker forbids the same
        //   handle being passed as both `&` and `&mut` simultaneously, so
        //   `src` and `dst` necessarily refer to non-aliasing handles for
        //   the duration of this call.
        // - `unwrap_buffer_f64` (line 109) produces a `&CudaBuffer<f64>`
        //   borrowed from `src`; we cast to `*const CudaBuffer<f64>` only
        //   to release the borrow on `src` so we can subsequently call
        //   `unwrap_buffer_f64_mut(dst)` without the borrow checker
        //   flagging `dst`'s mutable borrow as conflicting with `src`'s
        //   shared borrow on `self`. The two handles' inner storage is
        //   disjoint by the previous bullet.
        // - `CudaBuffer<f64>` has no interior mutability that would let
        //   `dst_buf`'s mutation affect what `src_ref` reads: it owns a
        //   `cudarc::driver::CudaSlice<f64>` (a device pointer + length)
        //   plus pool-ticket metadata. Mutating one buffer's contents on
        //   the device cannot alias another buffer's allocation because
        //   each `CudaBuffer` owns its own `CudaSlice` allocation.
        // - The `*const CudaBuffer<f64>` is reborrowed as `&CudaBuffer<f64>`
        //   for the duration of the kernel call only; the resulting
        //   reference does not outlive this stack frame.
        // - Pointer is non-null and aligned: it originates from the
        //   `&CudaBuffer<f64>` returned by `unwrap_buffer_f64` (a Rust
        //   reference, by definition non-null and aligned).
        // - Reads through `src_ref` see a fully-initialised
        //   `CudaBuffer<f64>` because the source reference was obtained
        //   from a live `Box<CudaBuffer<f64>>` inside the handle.
        // (Same shape as the f32 sibling at `strided_scatter_f32`; the
        // only differences are the buffer dtype and the kernel name.)
        let src_ref = unsafe { &*src_buf_ptr };
        crate::kernels::gpu_strided_scatter_f64(
            src_ref,
            dst_buf,
            view_shape,
            dst_strides,
            dst_offset,
            dev,
        )
        .map_err(Self::map_gpu_err)
    }

    fn strided_scatter_u16(
        &self,
        src: &GpuBufferHandle,
        dst: &mut GpuBufferHandle,
        view_shape: &[usize],
        dst_strides: &[isize],
        dst_offset: usize,
    ) -> FerrotorchResult<()> {
        let ord = src.device_ordinal();
        if dst.device_ordinal() != ord {
            return Err(FerrotorchError::DeviceMismatch {
                expected: ferrotorch_core::Device::Cuda(ord),
                got: ferrotorch_core::Device::Cuda(dst.device_ordinal()),
            });
        }
        if dst.dtype() != src.dtype() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "strided_scatter_u16: source dtype {} does not match destination dtype {}",
                    src.dtype(),
                    dst.dtype()
                ),
            });
        }
        let src_slice = match src.dtype() {
            DType::F16 => Self::unwrap_buffer_f16(src)?,
            DType::BF16 => Self::unwrap_buffer_bf16(src)?,
            other => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("strided_scatter_u16 expected F16/BF16, got {other}"),
                });
            }
        };
        let dst_slice = dst.downcast_mut::<cudarc::driver::CudaSlice<u16>>().ok_or(
            FerrotorchError::InvalidArgument {
                message: "strided_scatter_u16: destination is not a u16 CUDA slice".into(),
            },
        )?;
        let dev = self.device(ord)?;
        crate::kernels::gpu_strided_scatter_u16(
            src_slice,
            dst_slice,
            view_shape,
            dst_strides,
            dst_offset,
            dev,
        )
        .map_err(Self::map_gpu_err)
    }

    fn strided_cat(
        &self,
        src: &GpuBufferHandle,
        dst: &mut GpuBufferHandle,
        total_along_axis: usize,
        offset: usize,
        t_axis_size: usize,
        inner: usize,
        t_numel: usize,
        elem_size: usize,
    ) -> FerrotorchResult<()> {
        // Mirrors PyTorch's `aten::cat_out_cuda`
        // (`aten/src/ATen/native/cuda/Shape.cu`): host-level dispatch on the
        // scalar size, then a pure-memcpy kernel whose body only differs in
        // element width. For each supported `elem_size` we route to the
        // matching specialized kernel (no arithmetic — the data type is only
        // a copy width).
        let dev = self.device(src.device_ordinal())?;
        match elem_size {
            2 => {
                // bf16 / f16 — both stored as `CudaSlice<u16>`.
                if dst.dtype() != src.dtype() {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "strided_cat: source dtype {} does not match destination dtype {}",
                            src.dtype(),
                            dst.dtype()
                        ),
                    });
                }
                let in_slice = match src.dtype() {
                    DType::BF16 => Self::unwrap_buffer_bf16(src)?,
                    DType::F16 => Self::unwrap_buffer_f16(src)?,
                    other => {
                        return Err(FerrotorchError::InvalidArgument {
                            message: format!("strided_cat: expected F16/BF16, got {other}"),
                        });
                    }
                };
                let out_slice = dst.downcast_mut::<cudarc::driver::CudaSlice<u16>>().ok_or(
                    FerrotorchError::InvalidArgument {
                        message: "strided_cat: output is not a 2-byte (u16) buffer".into(),
                    },
                )?;
                crate::kernels::gpu_strided_cat_u16(
                    in_slice,
                    out_slice,
                    total_along_axis,
                    offset,
                    t_axis_size,
                    inner,
                    t_numel,
                    dev,
                )
                .map_err(Self::map_gpu_err)?;
                Ok(())
            }
            4 => {
                // f32.
                let in_buf = Self::unwrap_buffer(src)?;
                let out_buf = dst.downcast_mut::<CudaBuffer<f32>>().ok_or(
                    FerrotorchError::InvalidArgument {
                        message: "strided_cat: output is not CudaBuffer<f32>".into(),
                    },
                )?;
                crate::kernels::gpu_strided_cat(
                    in_buf,
                    out_buf,
                    total_along_axis,
                    offset,
                    t_axis_size,
                    inner,
                    t_numel,
                    dev,
                )
                .map_err(Self::map_gpu_err)?;
                Ok(())
            }
            8 => {
                // f64.
                let in_buf = Self::unwrap_buffer_f64(src)?;
                let out_buf = Self::unwrap_buffer_f64_mut(dst)?;
                crate::kernels::gpu_strided_cat_f64(
                    in_buf,
                    out_buf,
                    total_along_axis,
                    offset,
                    t_axis_size,
                    inner,
                    t_numel,
                    dev,
                )
                .map_err(Self::map_gpu_err)?;
                Ok(())
            }
            other => Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "strided_cat: unsupported elem_size={other} on CUDA (supported: 2, 4, 8)"
                ),
            }),
        }
    }

    // -- cuSOLVER linear algebra -------------------------------------------------

    fn svd_f32(
        &self,
        a: &GpuBufferHandle,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        // (#896 / #635) Device-resident SVD: no host bounce of matrix data.
        // gpu_svd_f32_dev accepts &CudaBuffer<f32> and returns (U, S, Vh) as
        // CudaBuffer<f32> — all on device. Matches torch.linalg.svd behaviour
        // on CUDA (outputs stay on the input device).
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (u_buf, s_buf, vt_buf) =
            crate::cusolver::gpu_svd_f32_dev(a_buf, m, n, dev).map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((
            Self::wrap_buffer(u_buf, ord),
            Self::wrap_buffer(s_buf, ord),
            Self::wrap_buffer(vt_buf, ord),
        ))
    }

    fn svd_f64(
        &self,
        a: &GpuBufferHandle,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        // (#896 / #635) Device-resident SVD — f64. See svd_f32 for rationale.
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (u_buf, s_buf, vt_buf) =
            crate::cusolver::gpu_svd_f64_dev(a_buf, m, n, dev).map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((
            Self::wrap_buffer_f64(u_buf, ord),
            Self::wrap_buffer_f64(s_buf, ord),
            Self::wrap_buffer_f64(vt_buf, ord),
        ))
    }

    fn cholesky_f32(&self, a: &GpuBufferHandle, n: usize) -> FerrotorchResult<GpuBufferHandle> {
        // (#632) Device-resident Cholesky: cuSOLVER potrf operates on a
        // memcpy_dtod clone of A, then a small host-side mask of the
        // upper triangle.
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let l_buf =
            crate::cusolver::gpu_cholesky_f32_dev(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(l_buf, a.device_ordinal()))
    }

    fn cholesky_f64(&self, a: &GpuBufferHandle, n: usize) -> FerrotorchResult<GpuBufferHandle> {
        // (#632) Device-resident Cholesky.
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let l_buf =
            crate::cusolver::gpu_cholesky_f64_dev(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(l_buf, a.device_ordinal()))
    }

    fn solve_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        n: usize,
        nrhs: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        // (#632) Device-resident path: no host bounce; on-device transposes
        // + cuSOLVER getrf/getrs working on column-major copies.
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let x = crate::cusolver::gpu_solve_f32_dev(a_buf, b_buf, n, nrhs, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(x, a.device_ordinal()))
    }

    fn solve_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        n: usize,
        nrhs: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        // (#632) Device-resident path: no host bounce.
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let x = crate::cusolver::gpu_solve_f64_dev(a_buf, b_buf, n, nrhs, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(x, a.device_ordinal()))
    }

    fn qr_f32(
        &self,
        a: &GpuBufferHandle,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        // (#896 / #635) Device-resident QR: no host bounce of matrix data.
        // gpu_qr_f32_dev uses on-device transposes (gpu_transpose_2d) and
        // on-device R/Q extraction kernels. Matches torch.linalg.qr on CUDA.
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (q_buf, r_buf) =
            crate::cusolver::gpu_qr_f32_dev(a_buf, m, n, dev).map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((Self::wrap_buffer(q_buf, ord), Self::wrap_buffer(r_buf, ord)))
    }

    fn qr_f64(
        &self,
        a: &GpuBufferHandle,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        // (#896 / #635) Device-resident QR — f64. See qr_f32 for rationale.
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (q_buf, r_buf) =
            crate::cusolver::gpu_qr_f64_dev(a_buf, m, n, dev).map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((
            Self::wrap_buffer_f64(q_buf, ord),
            Self::wrap_buffer_f64(r_buf, ord),
        ))
    }

    // GPU-resident LU factorization (no host bounces). Returns (LU_packed, pivots).
    fn lu_factor_f32(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, Vec<i32>)> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (lu, ipiv) =
            crate::cusolver::gpu_lu_factor_f32(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        // Pivots are O(n) ints — download to host. The LU matrix (O(n²))
        // stays on device.
        let ipiv_host = crate::transfer::gpu_to_cpu(&ipiv, dev).map_err(Self::map_gpu_err)?;
        Ok((Self::wrap_buffer(lu, a.device_ordinal()), ipiv_host))
    }

    fn lu_factor_f64(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, Vec<i32>)> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (lu, ipiv) =
            crate::cusolver::gpu_lu_factor_f64(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        let ipiv_host = crate::transfer::gpu_to_cpu(&ipiv, dev).map_err(Self::map_gpu_err)?;
        Ok((Self::wrap_buffer_f64(lu, a.device_ordinal()), ipiv_host))
    }

    fn lstsq_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        n: usize,
        nrhs: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let x = crate::cusolver::gpu_lstsq_f32(a_buf, b_buf, m, n, nrhs, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(x, a.device_ordinal()))
    }

    fn lstsq_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        n: usize,
        nrhs: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let x = crate::cusolver::gpu_lstsq_f64(a_buf, b_buf, m, n, nrhs, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(x, a.device_ordinal()))
    }

    fn eig_f32(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (w, v) = crate::cusolver::gpu_eig_f32(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((Self::wrap_buffer(w, ord), Self::wrap_buffer(v, ord)))
    }

    fn eig_f64(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (w, v) = crate::cusolver::gpu_eig_f64(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((Self::wrap_buffer_f64(w, ord), Self::wrap_buffer_f64(v, ord)))
    }

    // GPU-resident eigh / eigvalsh (no host bounces — see cusolver::gpu_eigh_*).
    fn eigh_f32(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (w, v) = crate::cusolver::gpu_eigh_f32(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((Self::wrap_buffer(w, ord), Self::wrap_buffer(v, ord)))
    }

    fn eigh_f64(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (w, v) = crate::cusolver::gpu_eigh_f64(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((Self::wrap_buffer_f64(w, ord), Self::wrap_buffer_f64(v, ord)))
    }

    fn eigvalsh_f32(&self, a: &GpuBufferHandle, n: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let w = crate::cusolver::gpu_eigvalsh_f32(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(w, a.device_ordinal()))
    }

    fn eigvalsh_f64(&self, a: &GpuBufferHandle, n: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let w = crate::cusolver::gpu_eigvalsh_f64(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(w, a.device_ordinal()))
    }

    // GPU 1-D FFT via cuFFT (#579). All paths are GPU-resident — see
    // `crate::cufft` for layout / normalization details.
    fn fft_c2c_f32(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        n: usize,
        inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out = crate::cufft::gpu_fft_c2c_f32(a_buf, batch, n, inverse, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, a.device_ordinal()))
    }

    fn fft_c2c_f64(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        n: usize,
        inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out = crate::cufft::gpu_fft_c2c_f64(a_buf, batch, n, inverse, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, a.device_ordinal()))
    }

    fn pad_truncate_complex_f32(
        &self,
        src: &GpuBufferHandle,
        batch: usize,
        src_n: usize,
        dst_n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let src_buf = Self::unwrap_buffer(src)?;
        let dev = self.device(src.device_ordinal())?;
        let out = crate::kernels::gpu_pad_truncate_complex_f32(src_buf, batch, src_n, dst_n, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, src.device_ordinal()))
    }

    fn pad_truncate_complex_f64(
        &self,
        src: &GpuBufferHandle,
        batch: usize,
        src_n: usize,
        dst_n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let src_buf = Self::unwrap_buffer_f64(src)?;
        let dev = self.device(src.device_ordinal())?;
        let out = crate::kernels::gpu_pad_truncate_complex_f64(src_buf, batch, src_n, dst_n, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, src.device_ordinal()))
    }

    fn fft2_c2c_f32(
        &self,
        a: &GpuBufferHandle,
        h: usize,
        w: usize,
        inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out =
            crate::cufft::gpu_fft2_c2c_f32(a_buf, h, w, inverse, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, a.device_ordinal()))
    }

    fn fft2_c2c_f64(
        &self,
        a: &GpuBufferHandle,
        h: usize,
        w: usize,
        inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out =
            crate::cufft::gpu_fft2_c2c_f64(a_buf, h, w, inverse, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, a.device_ordinal()))
    }

    fn repeat_along_dim_f32(
        &self,
        input: &GpuBufferHandle,
        outer: usize,
        repeat_count: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(input.device_ordinal())?;
        let out = crate::kernels::gpu_repeat_along_dim(in_buf, outer, repeat_count, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, input.device_ordinal()))
    }

    fn repeat_along_dim_f64(
        &self,
        input: &GpuBufferHandle,
        outer: usize,
        repeat_count: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(input.device_ordinal())?;
        let out = crate::kernels::gpu_repeat_along_dim_f64(in_buf, outer, repeat_count, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, input.device_ordinal()))
    }

    fn rfft_r2c_f32(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out =
            crate::cufft::gpu_rfft_r2c_f32(a_buf, batch, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, a.device_ordinal()))
    }

    fn rfft_r2c_f64(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out =
            crate::cufft::gpu_rfft_r2c_f64(a_buf, batch, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, a.device_ordinal()))
    }

    fn irfft_c2r_f32(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        n_out: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out =
            crate::cufft::gpu_irfft_c2r_f32(a_buf, batch, n_out, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, a.device_ordinal()))
    }

    fn irfft_c2r_f64(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        n_out: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out =
            crate::cufft::gpu_irfft_c2r_f64(a_buf, batch, n_out, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, a.device_ordinal()))
    }

    // -- Hermitian FFT (hfft / ihfft) (#636) ---------------------------------

    fn hfft_f32(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        half_in: usize,
        n_out: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out = crate::cufft::gpu_hfft_f32(a_buf, batch, half_in, n_out, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, a.device_ordinal()))
    }

    fn hfft_f64(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        half_in: usize,
        n_out: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out = crate::cufft::gpu_hfft_f64(a_buf, batch, half_in, n_out, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, a.device_ordinal()))
    }

    fn ihfft_f32(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out = crate::cufft::gpu_ihfft_f32(a_buf, batch, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, a.device_ordinal()))
    }

    fn ihfft_f64(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out = crate::cufft::gpu_ihfft_f64(a_buf, batch, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, a.device_ordinal()))
    }

    // -- N-D FFT 3-D (fftn / ifftn via cufftPlan3d) (#636) ------------------

    fn fftn3d_c2c_f32(
        &self,
        a: &GpuBufferHandle,
        d: usize,
        h: usize,
        w: usize,
        inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out = crate::cufft::gpu_fftn3d_c2c_f32(a_buf, d, h, w, inverse, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, a.device_ordinal()))
    }

    fn fftn3d_c2c_f64(
        &self,
        a: &GpuBufferHandle,
        d: usize,
        h: usize,
        w: usize,
        inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out = crate::cufft::gpu_fftn3d_c2c_f64(a_buf, d, h, w, inverse, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, a.device_ordinal()))
    }

    // -- N-D FFT 2-D (fftn / ifftn via cufftPlanMany) (#636) -----------------

    fn fftn2d_c2c_f32(
        &self,
        a: &GpuBufferHandle,
        h: usize,
        w: usize,
        inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out = crate::cufft::gpu_fftn2d_c2c_f32(a_buf, h, w, inverse, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, a.device_ordinal()))
    }

    fn fftn2d_c2c_f64(
        &self,
        a: &GpuBufferHandle,
        h: usize,
        w: usize,
        inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out = crate::cufft::gpu_fftn2d_c2c_f64(a_buf, h, w, inverse, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, a.device_ordinal()))
    }

    // -- axes-aware N-D FFT via cufftPlanMany (#966) -------------------------

    fn fftn_axes_c2c_f32(
        &self,
        a: &GpuBufferHandle,
        shape: &[usize],
        axes: &[usize],
        inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out = crate::cufft::gpu_fftn_axes_c2c_f32(a_buf, shape, axes, inverse, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, a.device_ordinal()))
    }

    fn fftn_axes_c2c_f64(
        &self,
        a: &GpuBufferHandle,
        shape: &[usize],
        axes: &[usize],
        inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out = crate::cufft::gpu_fftn_axes_c2c_f64(a_buf, shape, axes, inverse, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, a.device_ordinal()))
    }

    // -- Sparse SpMM (cuSPARSE) ----------------------------------------------
    //
    // PyTorch parity (rust-gpu-discipline §3): `torch.sparse.mm` runs on
    // cuSPARSE when the dense operand is CUDA. The just-in-time CSR upload
    // and the actual `cusparseSpMM` call live in `crate::sparse` so that
    // module-level SAFETY substantiation stays adjacent to the FFI.

    fn spmm_csr_f32(
        &self,
        crow_indices: &[u32],
        col_indices: &[u32],
        values: &[f32],
        dense: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dense_buf = Self::unwrap_buffer(dense)?;
        let dev = self.device(dense.device_ordinal())?;
        let handle = self.cusparse()?;
        let out = crate::sparse::gpu_spmm_csr_f32(
            handle,
            crow_indices,
            col_indices,
            values,
            dense_buf,
            m,
            k,
            n,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, dense.device_ordinal()))
    }

    fn spmm_csr_f64(
        &self,
        crow_indices: &[u32],
        col_indices: &[u32],
        values: &[f64],
        dense: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dense_buf = Self::unwrap_buffer_f64(dense)?;
        let dev = self.device(dense.device_ordinal())?;
        let handle = self.cusparse()?;
        let out = crate::sparse::gpu_spmm_csr_f64(
            handle,
            crow_indices,
            col_indices,
            values,
            dense_buf,
            m,
            k,
            n,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, dense.device_ordinal()))
    }

    // -- Sparse <-> Dense conversion (cuSPARSE) -- P3 -------------------------
    //
    // PyTorch parity (rust-gpu-discipline §3): `.to_dense()` and `.to_sparse()`
    // dispatch to `cusparseSparseToDense` / `cusparseDenseToSparse` when the
    // input lives on CUDA. Implementations live in `crate::sparse`.

    fn sparse_to_dense_csr_f32(
        &self,
        crow_indices: &[u32],
        col_indices: &[u32],
        values: &[f32],
        device_ordinal: usize,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(device_ordinal)?;
        let handle = self.cusparse()?;
        let out = crate::sparse::gpu_sparse_to_dense_csr_f32(
            handle,
            crow_indices,
            col_indices,
            values,
            m,
            n,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, device_ordinal))
    }

    fn sparse_to_dense_csr_f64(
        &self,
        crow_indices: &[u32],
        col_indices: &[u32],
        values: &[f64],
        device_ordinal: usize,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(device_ordinal)?;
        let handle = self.cusparse()?;
        let out = crate::sparse::gpu_sparse_to_dense_csr_f64(
            handle,
            crow_indices,
            col_indices,
            values,
            m,
            n,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, device_ordinal))
    }

    fn dense_to_sparse_csr_f32(
        &self,
        dense: &GpuBufferHandle,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f32>)> {
        let dense_buf = Self::unwrap_buffer(dense)?;
        let dev = self.device(dense.device_ordinal())?;
        let handle = self.cusparse()?;
        crate::sparse::gpu_dense_to_sparse_csr_f32(handle, dense_buf, m, n, dev)
            .map_err(Self::map_gpu_err)
    }

    fn dense_to_sparse_csr_f64(
        &self,
        dense: &GpuBufferHandle,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f64>)> {
        let dense_buf = Self::unwrap_buffer_f64(dense)?;
        let dev = self.device(dense.device_ordinal())?;
        let handle = self.cusparse()?;
        crate::sparse::gpu_dense_to_sparse_csr_f64(handle, dense_buf, m, n, dev)
            .map_err(Self::map_gpu_err)
    }

    // -- CSR/CSC/COO format-conversion + CSC → dense (cuSPARSE) — P7 ---------
    //
    // PyTorch parity (rust-gpu-discipline §3): `torch.sparse_csr_tensor` /
    // `torch.sparse_csc_tensor` / `torch.sparse_coo_tensor` route format
    // conversions through cuSPARSE on CUDA. Implementations live in
    // `crate::sparse`; this section is the type-erased trait wiring.

    fn csc_to_dense_f32(
        &self,
        col_ptrs: &[u32],
        row_indices: &[u32],
        values: &[f32],
        device_ordinal: usize,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(device_ordinal)?;
        let handle = self.cusparse()?;
        let out =
            crate::sparse::gpu_csc_to_dense_f32(handle, col_ptrs, row_indices, values, m, n, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, device_ordinal))
    }

    fn csc_to_dense_f64(
        &self,
        col_ptrs: &[u32],
        row_indices: &[u32],
        values: &[f64],
        device_ordinal: usize,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(device_ordinal)?;
        let handle = self.cusparse()?;
        let out =
            crate::sparse::gpu_csc_to_dense_f64(handle, col_ptrs, row_indices, values, m, n, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, device_ordinal))
    }

    fn csr_to_csc_f32(
        &self,
        crow_indices: &[u32],
        col_indices: &[u32],
        values: &[f32],
        device_ordinal: usize,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f32>)> {
        let dev = self.device(device_ordinal)?;
        let handle = self.cusparse()?;
        crate::sparse::gpu_csr_to_csc_f32(handle, crow_indices, col_indices, values, m, n, dev)
            .map_err(Self::map_gpu_err)
    }

    fn csr_to_csc_f64(
        &self,
        crow_indices: &[u32],
        col_indices: &[u32],
        values: &[f64],
        device_ordinal: usize,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f64>)> {
        let dev = self.device(device_ordinal)?;
        let handle = self.cusparse()?;
        crate::sparse::gpu_csr_to_csc_f64(handle, crow_indices, col_indices, values, m, n, dev)
            .map_err(Self::map_gpu_err)
    }

    fn coo_to_csr_f32(
        &self,
        row_indices: &[u32],
        col_indices: &[u32],
        values: &[f32],
        device_ordinal: usize,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f32>)> {
        let dev = self.device(device_ordinal)?;
        let handle = self.cusparse()?;
        crate::sparse::gpu_coo_to_csr_f32(handle, row_indices, col_indices, values, m, n, dev)
            .map_err(Self::map_gpu_err)
    }

    fn coo_to_csr_f64(
        &self,
        row_indices: &[u32],
        col_indices: &[u32],
        values: &[f64],
        device_ordinal: usize,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f64>)> {
        let dev = self.device(device_ordinal)?;
        let handle = self.cusparse()?;
        crate::sparse::gpu_coo_to_csr_f64(handle, row_indices, col_indices, values, m, n, dev)
            .map_err(Self::map_gpu_err)
    }

    fn csr_to_coo_f32(
        &self,
        crow_indices: &[u32],
        col_indices: &[u32],
        values: &[f32],
        device_ordinal: usize,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f32>)> {
        let dev = self.device(device_ordinal)?;
        let handle = self.cusparse()?;
        crate::sparse::gpu_csr_to_coo_f32(handle, crow_indices, col_indices, values, m, n, dev)
            .map_err(Self::map_gpu_err)
    }

    fn csr_to_coo_f64(
        &self,
        crow_indices: &[u32],
        col_indices: &[u32],
        values: &[f64],
        device_ordinal: usize,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f64>)> {
        let dev = self.device(device_ordinal)?;
        let handle = self.cusparse()?;
        crate::sparse::gpu_csr_to_coo_f64(handle, crow_indices, col_indices, values, m, n, dev)
            .map_err(Self::map_gpu_err)
    }

    // -- FlashAttention forward (P5) ---------------------------------------

    fn flash_attention_forward_f32(
        &self,
        query: &GpuBufferHandle,
        key: &GpuBufferHandle,
        value: &GpuBufferHandle,
        seq_q: usize,
        seq_k: usize,
        d: usize,
        d_v: usize,
        scale: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let q_buf = Self::unwrap_buffer(query)?;
        let k_buf = Self::unwrap_buffer(key)?;
        let v_buf = Self::unwrap_buffer(value)?;
        let dev = self.device(query.device_ordinal())?;
        let result = crate::flash_attention::gpu_flash_attention_f32(
            q_buf, k_buf, v_buf, seq_q, seq_k, d, d_v, /* batch_heads */ 1, scale,
            /* causal */ false, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, query.device_ordinal()))
    }

    fn flash_attention_forward_f64(
        &self,
        query: &GpuBufferHandle,
        key: &GpuBufferHandle,
        value: &GpuBufferHandle,
        seq_q: usize,
        seq_k: usize,
        d: usize,
        d_v: usize,
        scale: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let q_buf = Self::unwrap_buffer_f64(query)?;
        let k_buf = Self::unwrap_buffer_f64(key)?;
        let v_buf = Self::unwrap_buffer_f64(value)?;
        let dev = self.device(query.device_ordinal())?;
        let result = crate::flash_attention::gpu_flash_attention_f64(
            q_buf, k_buf, v_buf, seq_q, seq_k, d, d_v, /* batch_heads */ 1, scale,
            /* causal */ false, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, query.device_ordinal()))
    }

    // -- 2:4 Structured sparse matmul (cuSPARSELt) — P6 ----------------------
    //
    // Live implementation gated on the `cusparselt` cargo feature; without
    // the feature these methods inherit the trait's default `Err(...)`
    // shape so the dispatch site falls through to the dense reference
    // path (decompress + dense matmul). The active path is in
    // `crate::cusparselt::gpu_sparse_matmul_24`.

    #[cfg(feature = "cusparselt")]
    fn sparse_matmul_24_f32(
        &self,
        a: &GpuBufferHandle,
        b_dense_decompressed: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b_dense_decompressed)?;
        let dev = self.device(a.device_ordinal())?;
        let handle = self.cusparselt()?;
        let out = crate::cusparselt::gpu_sparse_matmul_24::<f32>(
            handle,
            a_buf,
            b_buf,
            m,
            k,
            n,
            crate::cusparselt::CuSpLtDType::F32,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, a.device_ordinal()))
    }

    // f16 / bf16 trait methods inherit the default `Err(InvalidArgument)`
    // because ferrotorch's `GpuBufferHandle` does not yet expose a
    // `CudaBuffer<u16>` downcast convention for storing f16/bf16 bit
    // patterns. Wiring those is a follow-up — the f32 path (TF32 mode)
    // is the only one currently exercised through this trait surface,
    // matching the SemiStructuredSparseTensor<T: Float> = f32/f64
    // generic constraint.

    // -- bf16 → bf16 native dispatch (#17) -----------------------------------
    //
    // Trait surface for bf16-resident inference (ViT, CLIP, Llama-style
    // transformer blocks). Each method extracts the underlying
    // `CudaSlice<u16>` from the handle (Maxine's storage convention from
    // #19), launches the matching PTX/cuBLAS kernel, and re-wraps the
    // resulting `CudaSlice<u16>` into a `GpuBufferHandle`. No `.cpu()` /
    // host readback / silent fallback — bf16 ops without a backing kernel
    // would `Err(InvalidArgument)` via the trait default (rust-gpu-discipline
    // §3 PyTorch parity).

    #[cfg(feature = "cuda")]
    fn matmul_bf16_bf16_nt(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_bf16(a)?;
        let b_buf = Self::unwrap_buffer_bf16(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::blas::gpu_matmul_bf16_bf16_nt(a_buf, b_buf, m, k, n, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn softmax_bf16_bf16(
        &self,
        a: &GpuBufferHandle,
        rows: usize,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_bf16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::bf16::gpu_softmax_bf16(buf, rows, cols, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn layernorm_bf16_bf16(
        &self,
        input: &GpuBufferHandle,
        gamma: &GpuBufferHandle,
        beta: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer_bf16(input)?;
        let g_buf = Self::unwrap_buffer_bf16(gamma)?;
        let b_buf = Self::unwrap_buffer_bf16(beta)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::bf16::gpu_layernorm_bf16(in_buf, g_buf, b_buf, rows, cols, eps, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, input.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn gelu_bf16_bf16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_bf16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::bf16::gpu_gelu_bf16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn silu_bf16_bf16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_bf16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::bf16::gpu_silu_bf16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn relu_bf16_bf16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_bf16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::bf16::gpu_relu_bf16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn add_bf16_bf16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_bf16(a)?;
        let b_buf = Self::unwrap_buffer_bf16(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::bf16::gpu_add_bf16(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn mul_bf16_bf16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_bf16(a)?;
        let b_buf = Self::unwrap_buffer_bf16(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::bf16::gpu_mul_bf16(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn scale_bf16_bf16(
        &self,
        a: &GpuBufferHandle,
        scalar: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_bf16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::bf16::gpu_scale_bf16(buf, scalar, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    // ─── Issue #23: bf16 dispatch-gap closure ──────────────────────────────
    //
    // Trait method impls covering the gaps surfaced by
    // `_probe_23_bf16_op_sweep.rs`. Each delegates to the PTX kernel in
    // `crate::bf16`. No silent CPU fallback (rust-gpu-discipline §3):
    // failures propagate as `Err(GpuError::*)` via `map_gpu_err`.

    #[cfg(feature = "cuda")]
    fn sub_bf16_bf16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_bf16(a)?;
        let b_buf = Self::unwrap_buffer_bf16(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::bf16::gpu_sub_bf16(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn div_bf16_bf16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_bf16(a)?;
        let b_buf = Self::unwrap_buffer_bf16(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::bf16::gpu_div_bf16(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn neg_bf16_bf16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_bf16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::bf16::gpu_neg_bf16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn broadcast_add_bf16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_bf16(a)?;
        let b_buf = Self::unwrap_buffer_bf16(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::bf16::gpu_broadcast_add_bf16(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn broadcast_sub_bf16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_bf16(a)?;
        let b_buf = Self::unwrap_buffer_bf16(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::bf16::gpu_broadcast_sub_bf16(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn broadcast_mul_bf16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_bf16(a)?;
        let b_buf = Self::unwrap_buffer_bf16(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::bf16::gpu_broadcast_mul_bf16(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn broadcast_div_bf16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_bf16(a)?;
        let b_buf = Self::unwrap_buffer_bf16(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::bf16::gpu_broadcast_div_bf16(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn sum_bf16_bf16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_bf16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::bf16::gpu_sum_bf16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn mean_bf16_bf16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_bf16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::bf16::gpu_mean_bf16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn prod_bf16_bf16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_bf16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::bf16::gpu_prod_bf16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn prod_backward_bf16_bf16(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_bf16(input)?;
        let grad_buf = Self::unwrap_buffer_bf16(grad_output)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::bf16::gpu_prod_backward_bf16(input_buf, grad_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, input.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn sum_axis_bf16_bf16(
        &self,
        a: &GpuBufferHandle,
        shape: &[usize],
        axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        if axis >= shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "sum_axis_bf16_bf16: axis {axis} out of bounds for shape {shape:?}"
                ),
            });
        }
        let outer: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let buf = Self::unwrap_buffer_bf16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::bf16::gpu_sum_axis_bf16_bf16(buf, outer, axis_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn mean_axis_bf16_bf16(
        &self,
        a: &GpuBufferHandle,
        shape: &[usize],
        axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        if axis >= shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "mean_axis_bf16_bf16: axis {axis} out of bounds for shape {shape:?}"
                ),
            });
        }
        let outer: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let buf = Self::unwrap_buffer_bf16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::bf16::gpu_mean_axis_bf16_bf16(buf, outer, axis_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn prod_axis_bf16_bf16(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_bf16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::bf16::gpu_prod_axis_bf16(buf, outer, axis_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn prod_axis_backward_bf16_bf16(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_bf16(input)?;
        let grad_buf = Self::unwrap_buffer_bf16(grad_output)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::bf16::gpu_prod_axis_backward_bf16(
            input_buf, grad_buf, outer, axis_size, inner, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, input.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn exp_bf16_bf16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_bf16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::bf16::gpu_exp_bf16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn log_bf16_bf16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_bf16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::bf16::gpu_log_bf16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn tanh_bf16_bf16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_bf16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::bf16::gpu_tanh_bf16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn sigmoid_bf16_bf16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_bf16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::bf16::gpu_sigmoid_bf16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_bf16(result, a.device_ordinal()))
    }

    // ── IEEE float16 (f16) ops — crosslink #1185 Phase 1 ─────────────────────
    //
    // Each unwraps with `unwrap_buffer_f16` (asserts the F16 tag, rejecting a
    // BF16-tagged handle), dispatches to the matching `crate::f16::gpu_*_f16`
    // PTX/cuBLAS kernel, and re-wraps with `wrap_buffer_f16` (F16 tag).

    #[cfg(feature = "cuda")]
    fn add_f16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f16(a)?;
        let b_buf = Self::unwrap_buffer_f16(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_add_f16(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn sub_f16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f16(a)?;
        let b_buf = Self::unwrap_buffer_f16(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_sub_f16(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn mul_f16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f16(a)?;
        let b_buf = Self::unwrap_buffer_f16(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_mul_f16(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn div_f16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f16(a)?;
        let b_buf = Self::unwrap_buffer_f16(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_div_f16(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn neg_f16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_f16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_neg_f16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn scale_f16(&self, a: &GpuBufferHandle, scale: f32) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_f16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_scale_f16(buf, scale, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn broadcast_add_f16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f16(a)?;
        let b_buf = Self::unwrap_buffer_f16(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::f16::gpu_broadcast_add_f16(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn broadcast_sub_f16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f16(a)?;
        let b_buf = Self::unwrap_buffer_f16(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::f16::gpu_broadcast_sub_f16(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn broadcast_mul_f16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f16(a)?;
        let b_buf = Self::unwrap_buffer_f16(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::f16::gpu_broadcast_mul_f16(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn broadcast_div_f16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f16(a)?;
        let b_buf = Self::unwrap_buffer_f16(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::f16::gpu_broadcast_div_f16(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn sum_f16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_f16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_sum_f16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn mean_f16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_f16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_mean_f16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn prod_f16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_f16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_prod_f16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn prod_backward_f16(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_f16(input)?;
        let grad_buf = Self::unwrap_buffer_f16(grad_output)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::f16::gpu_prod_backward_f16(input_buf, grad_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, input.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn sum_axis_f16(
        &self,
        a: &GpuBufferHandle,
        shape: &[usize],
        axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        if axis >= shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("sum_axis_f16: axis {axis} out of bounds for shape {shape:?}"),
            });
        }
        let outer: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let buf = Self::unwrap_buffer_f16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_sum_axis_f16(buf, outer, axis_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn mean_axis_f16(
        &self,
        a: &GpuBufferHandle,
        shape: &[usize],
        axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        if axis >= shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("mean_axis_f16: axis {axis} out of bounds for shape {shape:?}"),
            });
        }
        let outer: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let buf = Self::unwrap_buffer_f16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_mean_axis_f16(buf, outer, axis_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn prod_axis_f16(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_f16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_prod_axis_f16(buf, outer, axis_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn prod_axis_backward_f16(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_f16(input)?;
        let grad_buf = Self::unwrap_buffer_f16(grad_output)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::f16::gpu_prod_axis_backward_f16(
            input_buf, grad_buf, outer, axis_size, inner, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, input.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn std_var_axis_f16(
        &self,
        input: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
        correction: f64,
        take_sqrt: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_f16(input)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::f16::gpu_std_var_axis_f16(
            input_buf, outer, axis_size, inner, correction, take_sqrt, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, input.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn std_var_axis_backward_f16(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        result: &GpuBufferHandle,
        outer: usize,
        axis_size: usize,
        inner: usize,
        correction: f64,
        take_sqrt: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_f16(input)?;
        let grad_buf = Self::unwrap_buffer_f16(grad_output)?;
        let result_buf = Self::unwrap_buffer_f16(result)?;
        let dev = self.device(input.device_ordinal())?;
        let grad = crate::f16::gpu_std_var_axis_backward_f16(
            input_buf, grad_buf, result_buf, outer, axis_size, inner, correction, take_sqrt, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(grad, input.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn exp_f16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_f16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_exp_f16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn log_f16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_f16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_log_f16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn tanh_f16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_f16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_tanh_f16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn sigmoid_f16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_f16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_sigmoid_f16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn sqrt_f16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_f16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_sqrt_f16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn relu_f16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_f16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_relu_f16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn silu_f16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_f16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_silu_f16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn gelu_f16(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_f16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::f16::gpu_gelu_f16(buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn softmax_f16(
        &self,
        a: &GpuBufferHandle,
        rows: usize,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let buf = Self::unwrap_buffer_f16(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::f16::gpu_softmax_f16(buf, rows, cols, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn layernorm_f16(
        &self,
        input: &GpuBufferHandle,
        gamma: &GpuBufferHandle,
        beta: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer_f16(input)?;
        let g_buf = Self::unwrap_buffer_f16(gamma)?;
        let b_buf = Self::unwrap_buffer_f16(beta)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::f16::gpu_layernorm_f16(in_buf, g_buf, b_buf, rows, cols, eps, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, input.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn rmsnorm_f16(
        &self,
        input: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer_f16(input)?;
        let w_buf = Self::unwrap_buffer_f16(weight)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::f16::gpu_rmsnorm_f16(in_buf, w_buf, rows, cols, eps, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, input.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn matmul_f16_f16(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f16(a)?;
        let b_buf = Self::unwrap_buffer_f16(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::blas::gpu_matmul_f16_f16(a_buf, b_buf, m, k, n, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f16(result, a.device_ordinal()))
    }

    // ── Integer (i32 / i64) ops — crosslink #1185 Phase 2b ───────────────────
    //
    // Runtime dispatch on the ScalarType tag (PyTorch style): each method
    // switches on `a.dtype()`, unwraps with the tag-asserting Phase-2a
    // `unwrap_buffer_i32`/`i64` (rejecting a mismatched tag), launches the
    // matching `crate::int_kernels::gpu_*_i{32,64}` PTX kernel on the native
    // integer buffer, and re-wraps the resident `CudaSlice` result with
    // `wrap_slice_i{32,64}` (correct DType tag). No f32/f64 detour, no host
    // round-trip. An unsupported tag returns `NotImplementedOnCuda` (PyTorch
    // parity — rust-gpu-discipline §3).

    #[cfg(feature = "cuda")]
    fn int_add(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        match a.dtype() {
            DType::I32 => {
                let av = Self::unwrap_buffer_i32(a)?;
                let bv = Self::unwrap_buffer_i32(b)?;
                let r = crate::int_kernels::gpu_add_i32(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i32(r, a.device_ordinal()))
            }
            DType::I64 => {
                let av = Self::unwrap_buffer_i64(a)?;
                let bv = Self::unwrap_buffer_i64(b)?;
                let r = crate::int_kernels::gpu_add_i64(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i64(r, a.device_ordinal()))
            }
            _ => Err(FerrotorchError::NotImplementedOnCuda { op: "int_add" }),
        }
    }

    #[cfg(feature = "cuda")]
    fn int_sub(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        match a.dtype() {
            DType::I32 => {
                let av = Self::unwrap_buffer_i32(a)?;
                let bv = Self::unwrap_buffer_i32(b)?;
                let r = crate::int_kernels::gpu_sub_i32(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i32(r, a.device_ordinal()))
            }
            DType::I64 => {
                let av = Self::unwrap_buffer_i64(a)?;
                let bv = Self::unwrap_buffer_i64(b)?;
                let r = crate::int_kernels::gpu_sub_i64(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i64(r, a.device_ordinal()))
            }
            _ => Err(FerrotorchError::NotImplementedOnCuda { op: "int_sub" }),
        }
    }

    #[cfg(feature = "cuda")]
    fn int_mul(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        match a.dtype() {
            DType::I32 => {
                let av = Self::unwrap_buffer_i32(a)?;
                let bv = Self::unwrap_buffer_i32(b)?;
                let r = crate::int_kernels::gpu_mul_i32(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i32(r, a.device_ordinal()))
            }
            DType::I64 => {
                let av = Self::unwrap_buffer_i64(a)?;
                let bv = Self::unwrap_buffer_i64(b)?;
                let r = crate::int_kernels::gpu_mul_i64(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i64(r, a.device_ordinal()))
            }
            _ => Err(FerrotorchError::NotImplementedOnCuda { op: "int_mul" }),
        }
    }

    #[cfg(feature = "cuda")]
    fn int_neg(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        match a.dtype() {
            DType::I32 => {
                let av = Self::unwrap_buffer_i32(a)?;
                let r =
                    crate::int_kernels::gpu_neg_i32(av.inner(), dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i32(r, a.device_ordinal()))
            }
            DType::I64 => {
                let av = Self::unwrap_buffer_i64(a)?;
                let r =
                    crate::int_kernels::gpu_neg_i64(av.inner(), dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i64(r, a.device_ordinal()))
            }
            _ => Err(FerrotorchError::NotImplementedOnCuda { op: "int_neg" }),
        }
    }

    #[cfg(feature = "cuda")]
    fn int_floor_div(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        match a.dtype() {
            DType::I32 => {
                let av = Self::unwrap_buffer_i32(a)?;
                let bv = Self::unwrap_buffer_i32(b)?;
                let r = crate::int_kernels::gpu_floor_div_i32(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i32(r, a.device_ordinal()))
            }
            DType::I64 => {
                let av = Self::unwrap_buffer_i64(a)?;
                let bv = Self::unwrap_buffer_i64(b)?;
                let r = crate::int_kernels::gpu_floor_div_i64(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i64(r, a.device_ordinal()))
            }
            _ => Err(FerrotorchError::NotImplementedOnCuda {
                op: "int_floor_div",
            }),
        }
    }

    #[cfg(feature = "cuda")]
    fn int_remainder(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        match a.dtype() {
            DType::I32 => {
                let av = Self::unwrap_buffer_i32(a)?;
                let bv = Self::unwrap_buffer_i32(b)?;
                let r = crate::int_kernels::gpu_remainder_i32(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i32(r, a.device_ordinal()))
            }
            DType::I64 => {
                let av = Self::unwrap_buffer_i64(a)?;
                let bv = Self::unwrap_buffer_i64(b)?;
                let r = crate::int_kernels::gpu_remainder_i64(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i64(r, a.device_ordinal()))
            }
            _ => Err(FerrotorchError::NotImplementedOnCuda {
                op: "int_remainder",
            }),
        }
    }

    #[cfg(feature = "cuda")]
    fn int_bitand(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        match a.dtype() {
            DType::I32 => {
                let av = Self::unwrap_buffer_i32(a)?;
                let bv = Self::unwrap_buffer_i32(b)?;
                let r = crate::int_kernels::gpu_bitand_i32(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i32(r, a.device_ordinal()))
            }
            DType::I64 => {
                let av = Self::unwrap_buffer_i64(a)?;
                let bv = Self::unwrap_buffer_i64(b)?;
                let r = crate::int_kernels::gpu_bitand_i64(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i64(r, a.device_ordinal()))
            }
            _ => Err(FerrotorchError::NotImplementedOnCuda { op: "int_bitand" }),
        }
    }

    #[cfg(feature = "cuda")]
    fn int_bitor(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        match a.dtype() {
            DType::I32 => {
                let av = Self::unwrap_buffer_i32(a)?;
                let bv = Self::unwrap_buffer_i32(b)?;
                let r = crate::int_kernels::gpu_bitor_i32(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i32(r, a.device_ordinal()))
            }
            DType::I64 => {
                let av = Self::unwrap_buffer_i64(a)?;
                let bv = Self::unwrap_buffer_i64(b)?;
                let r = crate::int_kernels::gpu_bitor_i64(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i64(r, a.device_ordinal()))
            }
            _ => Err(FerrotorchError::NotImplementedOnCuda { op: "int_bitor" }),
        }
    }

    #[cfg(feature = "cuda")]
    fn int_bitxor(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        match a.dtype() {
            DType::I32 => {
                let av = Self::unwrap_buffer_i32(a)?;
                let bv = Self::unwrap_buffer_i32(b)?;
                let r = crate::int_kernels::gpu_bitxor_i32(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i32(r, a.device_ordinal()))
            }
            DType::I64 => {
                let av = Self::unwrap_buffer_i64(a)?;
                let bv = Self::unwrap_buffer_i64(b)?;
                let r = crate::int_kernels::gpu_bitxor_i64(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i64(r, a.device_ordinal()))
            }
            _ => Err(FerrotorchError::NotImplementedOnCuda { op: "int_bitxor" }),
        }
    }

    #[cfg(feature = "cuda")]
    fn int_bitnot(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        match a.dtype() {
            DType::I32 => {
                let av = Self::unwrap_buffer_i32(a)?;
                let r = crate::int_kernels::gpu_bitnot_i32(av.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i32(r, a.device_ordinal()))
            }
            DType::I64 => {
                let av = Self::unwrap_buffer_i64(a)?;
                let r = crate::int_kernels::gpu_bitnot_i64(av.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i64(r, a.device_ordinal()))
            }
            _ => Err(FerrotorchError::NotImplementedOnCuda { op: "int_bitnot" }),
        }
    }

    #[cfg(feature = "cuda")]
    fn int_shl(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        match a.dtype() {
            DType::I32 => {
                let av = Self::unwrap_buffer_i32(a)?;
                let bv = Self::unwrap_buffer_i32(b)?;
                let r = crate::int_kernels::gpu_shl_i32(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i32(r, a.device_ordinal()))
            }
            DType::I64 => {
                let av = Self::unwrap_buffer_i64(a)?;
                let bv = Self::unwrap_buffer_i64(b)?;
                let r = crate::int_kernels::gpu_shl_i64(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i64(r, a.device_ordinal()))
            }
            _ => Err(FerrotorchError::NotImplementedOnCuda { op: "int_shl" }),
        }
    }

    #[cfg(feature = "cuda")]
    fn int_shr(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        match a.dtype() {
            DType::I32 => {
                let av = Self::unwrap_buffer_i32(a)?;
                let bv = Self::unwrap_buffer_i32(b)?;
                let r = crate::int_kernels::gpu_shr_i32(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i32(r, a.device_ordinal()))
            }
            DType::I64 => {
                let av = Self::unwrap_buffer_i64(a)?;
                let bv = Self::unwrap_buffer_i64(b)?;
                let r = crate::int_kernels::gpu_shr_i64(av.inner(), bv.inner(), dev)
                    .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i64(r, a.device_ordinal()))
            }
            _ => Err(FerrotorchError::NotImplementedOnCuda { op: "int_shr" }),
        }
    }

    #[cfg(feature = "cuda")]
    fn int_sum(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        match a.dtype() {
            DType::I32 => {
                let av = Self::unwrap_buffer_i32(a)?;
                let r =
                    crate::int_kernels::gpu_sum_i32(av.inner(), dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i32(r, a.device_ordinal()))
            }
            DType::I64 => {
                let av = Self::unwrap_buffer_i64(a)?;
                let r =
                    crate::int_kernels::gpu_sum_i64(av.inner(), dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i64(r, a.device_ordinal()))
            }
            _ => Err(FerrotorchError::NotImplementedOnCuda { op: "int_sum" }),
        }
    }

    #[cfg(feature = "cuda")]
    fn int_prod(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        match a.dtype() {
            DType::I32 => {
                let av = Self::unwrap_buffer_i32(a)?;
                let r =
                    crate::int_kernels::gpu_prod_i32(av.inner(), dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i32(r, a.device_ordinal()))
            }
            DType::I64 => {
                let av = Self::unwrap_buffer_i64(a)?;
                let r =
                    crate::int_kernels::gpu_prod_i64(av.inner(), dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i64(r, a.device_ordinal()))
            }
            _ => Err(FerrotorchError::NotImplementedOnCuda { op: "int_prod" }),
        }
    }

    #[cfg(feature = "cuda")]
    fn int_min(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        match a.dtype() {
            DType::I32 => {
                let av = Self::unwrap_buffer_i32(a)?;
                let r =
                    crate::int_kernels::gpu_min_i32(av.inner(), dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i32(r, a.device_ordinal()))
            }
            DType::I64 => {
                let av = Self::unwrap_buffer_i64(a)?;
                let r =
                    crate::int_kernels::gpu_min_i64(av.inner(), dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i64(r, a.device_ordinal()))
            }
            _ => Err(FerrotorchError::NotImplementedOnCuda { op: "int_min" }),
        }
    }

    #[cfg(feature = "cuda")]
    fn int_max(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        match a.dtype() {
            DType::I32 => {
                let av = Self::unwrap_buffer_i32(a)?;
                let r =
                    crate::int_kernels::gpu_max_i32(av.inner(), dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i32(r, a.device_ordinal()))
            }
            DType::I64 => {
                let av = Self::unwrap_buffer_i64(a)?;
                let r =
                    crate::int_kernels::gpu_max_i64(av.inner(), dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_i64(r, a.device_ordinal()))
            }
            _ => Err(FerrotorchError::NotImplementedOnCuda { op: "int_max" }),
        }
    }

    // ── argmax / argmin / gather / cast — crosslink #1185 Phase 2c ───────────

    #[cfg(feature = "cuda")]
    fn argmax(
        &self,
        src: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(src.device_ordinal())?;
        let ord = src.device_ordinal();
        let r = match src.dtype() {
            DType::F32 => crate::reduce_arg::gpu_argmax_f32(
                Self::unwrap_buffer(src)?.inner(),
                outer,
                dim_size,
                inner,
                dev,
            ),
            DType::F64 => crate::reduce_arg::gpu_argmax_f64(
                Self::unwrap_buffer_f64(src)?.inner(),
                outer,
                dim_size,
                inner,
                dev,
            ),
            DType::F16 => crate::reduce_arg::gpu_argmax_f16(
                Self::unwrap_buffer_f16(src)?,
                outer,
                dim_size,
                inner,
                dev,
            ),
            DType::BF16 => crate::reduce_arg::gpu_argmax_bf16(
                Self::unwrap_buffer_bf16(src)?,
                outer,
                dim_size,
                inner,
                dev,
            ),
            DType::I32 => crate::reduce_arg::gpu_argmax_i32(
                Self::unwrap_buffer_i32(src)?.inner(),
                outer,
                dim_size,
                inner,
                dev,
            ),
            DType::I64 => crate::reduce_arg::gpu_argmax_i64(
                Self::unwrap_buffer_i64(src)?.inner(),
                outer,
                dim_size,
                inner,
                dev,
            ),
            other => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("argmax: unsupported value dtype {other}"),
                });
            }
        }
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_slice_i64(r, ord))
    }

    #[cfg(feature = "cuda")]
    fn argmin(
        &self,
        src: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(src.device_ordinal())?;
        let ord = src.device_ordinal();
        let r = match src.dtype() {
            DType::F32 => crate::reduce_arg::gpu_argmin_f32(
                Self::unwrap_buffer(src)?.inner(),
                outer,
                dim_size,
                inner,
                dev,
            ),
            DType::F64 => crate::reduce_arg::gpu_argmin_f64(
                Self::unwrap_buffer_f64(src)?.inner(),
                outer,
                dim_size,
                inner,
                dev,
            ),
            DType::F16 => crate::reduce_arg::gpu_argmin_f16(
                Self::unwrap_buffer_f16(src)?,
                outer,
                dim_size,
                inner,
                dev,
            ),
            DType::BF16 => crate::reduce_arg::gpu_argmin_bf16(
                Self::unwrap_buffer_bf16(src)?,
                outer,
                dim_size,
                inner,
                dev,
            ),
            DType::I32 => crate::reduce_arg::gpu_argmin_i32(
                Self::unwrap_buffer_i32(src)?.inner(),
                outer,
                dim_size,
                inner,
                dev,
            ),
            DType::I64 => crate::reduce_arg::gpu_argmin_i64(
                Self::unwrap_buffer_i64(src)?.inner(),
                outer,
                dim_size,
                inner,
                dev,
            ),
            other => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("argmin: unsupported value dtype {other}"),
                });
            }
        }
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_slice_i64(r, ord))
    }

    #[cfg(feature = "cuda")]
    fn searchsorted_1d(
        &self,
        values: &GpuBufferHandle,
        boundaries: &GpuBufferHandle,
        right: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        if values.dtype() != boundaries.dtype() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "searchsorted_1d: values ({}) and boundaries ({}) must share a dtype",
                    values.dtype(),
                    boundaries.dtype()
                ),
            });
        }
        let dev = self.device(values.device_ordinal())?;
        let ord = values.device_ordinal();
        let n_vals = values.len();
        let n_bounds = boundaries.len();
        let r = match values.dtype() {
            DType::F32 => crate::search::gpu_searchsorted_f32(
                Self::unwrap_buffer(values)?.inner(),
                Self::unwrap_buffer(boundaries)?.inner(),
                n_vals,
                n_bounds,
                right,
                dev,
            ),
            DType::F64 => crate::search::gpu_searchsorted_f64(
                Self::unwrap_buffer_f64(values)?.inner(),
                Self::unwrap_buffer_f64(boundaries)?.inner(),
                n_vals,
                n_bounds,
                right,
                dev,
            ),
            other => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("searchsorted_1d: unsupported value dtype {other}"),
                });
            }
        }
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_slice_i64(r, ord))
    }

    #[cfg(feature = "cuda")]
    fn topk_1d(
        &self,
        values_in: &GpuBufferHandle,
        outer: usize,
        last_dim: usize,
        k: usize,
        largest: bool,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let dev = self.device(values_in.device_ordinal())?;
        let ord = values_in.device_ordinal();
        match values_in.dtype() {
            DType::F32 => {
                let (vals, idx) = crate::search::gpu_topk_f32(
                    Self::unwrap_buffer(values_in)?.inner(),
                    outer,
                    last_dim,
                    k,
                    largest,
                    dev,
                )
                .map_err(Self::map_gpu_err)?;
                Ok((
                    Self::wrap_slice_f32(vals, ord),
                    Self::wrap_slice_i64(idx, ord),
                ))
            }
            DType::F64 => {
                let (vals, idx) = crate::search::gpu_topk_f64(
                    Self::unwrap_buffer_f64(values_in)?.inner(),
                    outer,
                    last_dim,
                    k,
                    largest,
                    dev,
                )
                .map_err(Self::map_gpu_err)?;
                Ok((
                    Self::wrap_slice_f64(vals, ord),
                    Self::wrap_slice_i64(idx, ord),
                ))
            }
            other => Err(FerrotorchError::InvalidArgument {
                message: format!("topk_1d: unsupported value dtype {other}"),
            }),
        }
    }

    #[cfg(feature = "cuda")]
    fn histc_1d(
        &self,
        input: &GpuBufferHandle,
        bins: usize,
        min_val: f64,
        max_val: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let n = input.len();
        match input.dtype() {
            DType::F32 => {
                let out = crate::search::gpu_histc_f32(
                    Self::unwrap_buffer(input)?.inner(),
                    n,
                    bins,
                    min_val as f32,
                    max_val as f32,
                    dev,
                )
                .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_f32(out, ord))
            }
            DType::F64 => {
                let out = crate::search::gpu_histc_f64(
                    Self::unwrap_buffer_f64(input)?.inner(),
                    n,
                    bins,
                    min_val,
                    max_val,
                    dev,
                )
                .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_f64(out, ord))
            }
            other => Err(FerrotorchError::InvalidArgument {
                message: format!("histc_1d: unsupported value dtype {other}"),
            }),
        }
    }

    #[cfg(feature = "cuda")]
    fn meshgrid_grid(
        &self,
        input: &GpuBufferHandle,
        total: usize,
        inner: usize,
        axis_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        match input.dtype() {
            DType::F32 => {
                let out = crate::search::gpu_meshgrid_f32(
                    Self::unwrap_buffer(input)?.inner(),
                    total,
                    inner,
                    axis_len,
                    dev,
                )
                .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_f32(out, ord))
            }
            DType::F64 => {
                let out = crate::search::gpu_meshgrid_f64(
                    Self::unwrap_buffer_f64(input)?.inner(),
                    total,
                    inner,
                    axis_len,
                    dev,
                )
                .map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_slice_f64(out, ord))
            }
            other => Err(FerrotorchError::InvalidArgument {
                message: format!("meshgrid_grid: unsupported value dtype {other}"),
            }),
        }
    }

    #[cfg(feature = "cuda")]
    fn unique_consecutive_1d(
        &self,
        input: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, Vec<usize>, Vec<usize>)> {
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        match input.dtype() {
            DType::F32 => {
                let (values, inverse, counts) =
                    crate::search::gpu_unique_consecutive_f32(Self::unwrap_buffer(input)?, n, dev)
                        .map_err(Self::map_gpu_err)?;
                Ok((Self::wrap_buffer(values, ord), inverse, counts))
            }
            DType::F64 => {
                let (values, inverse, counts) = crate::search::gpu_unique_consecutive_f64(
                    Self::unwrap_buffer_f64(input)?,
                    n,
                    dev,
                )
                .map_err(Self::map_gpu_err)?;
                Ok((Self::wrap_buffer_f64(values, ord), inverse, counts))
            }
            other => Err(FerrotorchError::InvalidArgument {
                message: format!("unique_consecutive_1d: unsupported value dtype {other}"),
            }),
        }
    }

    #[cfg(feature = "cuda")]
    fn unique_1d(
        &self,
        input: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, Vec<usize>, Vec<usize>)> {
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        match input.dtype() {
            DType::F32 => {
                let (values, inverse, counts) =
                    crate::search::gpu_unique_f32(Self::unwrap_buffer(input)?, n, dev)
                        .map_err(Self::map_gpu_err)?;
                Ok((Self::wrap_buffer(values, ord), inverse, counts))
            }
            DType::F64 => {
                let (values, inverse, counts) =
                    crate::search::gpu_unique_f64(Self::unwrap_buffer_f64(input)?, n, dev)
                        .map_err(Self::map_gpu_err)?;
                Ok((Self::wrap_buffer_f64(values, ord), inverse, counts))
            }
            other => Err(FerrotorchError::InvalidArgument {
                message: format!("unique_1d: unsupported value dtype {other}"),
            }),
        }
    }

    #[cfg(feature = "cuda")]
    fn index_select_intidx(
        &self,
        src: &GpuBufferHandle,
        index: &GpuBufferHandle,
        outer: usize,
        in_dim: usize,
        out_dim: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        self.gather_or_select(src, index, outer, in_dim, out_dim, inner, false)
    }

    #[cfg(feature = "cuda")]
    fn gather_intidx(
        &self,
        src: &GpuBufferHandle,
        index: &GpuBufferHandle,
        outer: usize,
        in_dim: usize,
        out_dim: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        self.gather_or_select(src, index, outer, in_dim, out_dim, inner, true)
    }

    #[cfg(feature = "cuda")]
    fn gather_intidx_nd(
        &self,
        src: &GpuBufferHandle,
        index: &GpuBufferHandle,
        input_shape: &[usize],
        index_shape: &[usize],
        dim: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        use crate::gather_int as gi;
        let dev = self.device(src.device_ordinal())?;
        let ord = src.device_ordinal();
        match index.dtype() {
            DType::I32 | DType::I64 => {}
            other => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("gather_intidx_nd: index dtype must be I32/I64, got {other}"),
                });
            }
        }
        let (input_strides, index_dims) =
            Self::gather_nd_metadata(src, index, input_shape, index_shape, dim)?;
        let i32idx = index.dtype() == DType::I32;
        macro_rules! run {
            ($val:expr, $g32:path, $g64:path, $wrap:expr) => {{
                let r = if i32idx {
                    $g32(
                        $val,
                        Self::unwrap_buffer_i32(index)?.inner(),
                        &input_strides,
                        &index_dims,
                        dim,
                        dev,
                    )
                } else {
                    $g64(
                        $val,
                        Self::unwrap_buffer_i64(index)?.inner(),
                        &input_strides,
                        &index_dims,
                        dim,
                        dev,
                    )
                }
                .map_err(Self::map_gpu_err)?;
                Ok($wrap(r, ord))
            }};
        }
        match src.dtype() {
            DType::F32 => run!(
                Self::unwrap_buffer(src)?.inner(),
                gi::gather_nd_f32_i32,
                gi::gather_nd_f32_i64,
                Self::wrap_slice_f32
            ),
            DType::F64 => run!(
                Self::unwrap_buffer_f64(src)?.inner(),
                gi::gather_nd_f64_i32,
                gi::gather_nd_f64_i64,
                Self::wrap_slice_f64
            ),
            DType::I32 => run!(
                Self::unwrap_buffer_i32(src)?.inner(),
                gi::gather_nd_i32_i32,
                gi::gather_nd_i32_i64,
                Self::wrap_slice_i32
            ),
            DType::I64 => run!(
                Self::unwrap_buffer_i64(src)?.inner(),
                gi::gather_nd_i64_i32,
                gi::gather_nd_i64_i64,
                Self::wrap_slice_i64
            ),
            DType::F16 => run!(
                Self::unwrap_buffer_f16(src)?,
                gi::gather_nd_u16_i32,
                gi::gather_nd_u16_i64,
                Self::wrap_buffer_f16
            ),
            DType::BF16 => run!(
                Self::unwrap_buffer_bf16(src)?,
                gi::gather_nd_u16_i32,
                gi::gather_nd_u16_i64,
                Self::wrap_buffer_bf16
            ),
            other => Err(FerrotorchError::InvalidArgument {
                message: format!("gather_intidx_nd: unsupported value dtype {other}"),
            }),
        }
    }

    // -- dim-aware gather / scatter family (#1545 / sub #1535) ----------------

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn gather_dim_f32(
        &self,
        input: &GpuBufferHandle,
        index: &GpuBufferHandle,
        outer: usize,
        in_dim: usize,
        out_dim: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let out = crate::scatter_gather_kernels::gpu_gather_dim_f32(
            Self::unwrap_buffer(input)?,
            Self::unwrap_buffer_i64(index)?.inner(),
            outer,
            in_dim,
            out_dim,
            inner,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, ord))
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn gather_dim_f64(
        &self,
        input: &GpuBufferHandle,
        index: &GpuBufferHandle,
        outer: usize,
        in_dim: usize,
        out_dim: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let out = crate::scatter_gather_kernels::gpu_gather_dim_f64(
            Self::unwrap_buffer_f64(input)?,
            Self::unwrap_buffer_i64(index)?.inner(),
            outer,
            in_dim,
            out_dim,
            inner,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, ord))
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn scatter_dim_f32(
        &self,
        input: &GpuBufferHandle,
        index: &GpuBufferHandle,
        src: &GpuBufferHandle,
        outer: usize,
        out_dim: usize,
        idx_dim: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let out = crate::scatter_gather_kernels::gpu_scatter_dim_f32(
            Self::unwrap_buffer(input)?,
            Self::unwrap_buffer_i64(index)?.inner(),
            Self::unwrap_buffer(src)?,
            outer,
            out_dim,
            idx_dim,
            inner,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, ord))
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn scatter_dim_f64(
        &self,
        input: &GpuBufferHandle,
        index: &GpuBufferHandle,
        src: &GpuBufferHandle,
        outer: usize,
        out_dim: usize,
        idx_dim: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let out = crate::scatter_gather_kernels::gpu_scatter_dim_f64(
            Self::unwrap_buffer_f64(input)?,
            Self::unwrap_buffer_i64(index)?.inner(),
            Self::unwrap_buffer_f64(src)?,
            outer,
            out_dim,
            idx_dim,
            inner,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, ord))
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn scatter_value_dim_f32(
        &self,
        input: &GpuBufferHandle,
        index: &GpuBufferHandle,
        value: f32,
        outer: usize,
        out_dim: usize,
        idx_dim: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let out = crate::scatter_gather_kernels::gpu_scatter_value_dim_f32(
            Self::unwrap_buffer(input)?,
            Self::unwrap_buffer_i64(index)?.inner(),
            value,
            outer,
            out_dim,
            idx_dim,
            inner,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, ord))
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn scatter_value_dim_f64(
        &self,
        input: &GpuBufferHandle,
        index: &GpuBufferHandle,
        value: f64,
        outer: usize,
        out_dim: usize,
        idx_dim: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let out = crate::scatter_gather_kernels::gpu_scatter_value_dim_f64(
            Self::unwrap_buffer_f64(input)?,
            Self::unwrap_buffer_i64(index)?.inner(),
            value,
            outer,
            out_dim,
            idx_dim,
            inner,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, ord))
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn scatter_add_dim_f32(
        &self,
        input: &GpuBufferHandle,
        index: &GpuBufferHandle,
        src: &GpuBufferHandle,
        outer: usize,
        out_dim: usize,
        idx_dim: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let out = crate::scatter_gather_kernels::gpu_scatter_add_dim_f32(
            Self::unwrap_buffer(input)?,
            Self::unwrap_buffer_i64(index)?.inner(),
            Self::unwrap_buffer(src)?,
            outer,
            out_dim,
            idx_dim,
            inner,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, ord))
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn scatter_add_dim_f64(
        &self,
        input: &GpuBufferHandle,
        index: &GpuBufferHandle,
        src: &GpuBufferHandle,
        outer: usize,
        out_dim: usize,
        idx_dim: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let out = crate::scatter_gather_kernels::gpu_scatter_add_dim_f64(
            Self::unwrap_buffer_f64(input)?,
            Self::unwrap_buffer_i64(index)?.inner(),
            Self::unwrap_buffer_f64(src)?,
            outer,
            out_dim,
            idx_dim,
            inner,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, ord))
    }

    #[cfg(feature = "cuda")]
    fn scatter_nd_f32(
        &self,
        input: &GpuBufferHandle,
        index: &GpuBufferHandle,
        src: &GpuBufferHandle,
        input_shape: &[usize],
        index_shape: &[usize],
        dim: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let out = crate::scatter_gather_kernels::gpu_scatter_nd_f32(
            Self::unwrap_buffer(input)?,
            Self::unwrap_buffer_i64(index)?.inner(),
            Self::unwrap_buffer(src)?,
            input_shape,
            index_shape,
            dim,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, ord))
    }

    #[cfg(feature = "cuda")]
    fn scatter_nd_f64(
        &self,
        input: &GpuBufferHandle,
        index: &GpuBufferHandle,
        src: &GpuBufferHandle,
        input_shape: &[usize],
        index_shape: &[usize],
        dim: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let out = crate::scatter_gather_kernels::gpu_scatter_nd_f64(
            Self::unwrap_buffer_f64(input)?,
            Self::unwrap_buffer_i64(index)?.inner(),
            Self::unwrap_buffer_f64(src)?,
            input_shape,
            index_shape,
            dim,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, ord))
    }

    #[cfg(feature = "cuda")]
    fn scatter_value_nd_f32(
        &self,
        input: &GpuBufferHandle,
        index: &GpuBufferHandle,
        value: f32,
        input_shape: &[usize],
        index_shape: &[usize],
        dim: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let out = crate::scatter_gather_kernels::gpu_scatter_value_nd_f32(
            Self::unwrap_buffer(input)?,
            Self::unwrap_buffer_i64(index)?.inner(),
            value,
            input_shape,
            index_shape,
            dim,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, ord))
    }

    #[cfg(feature = "cuda")]
    fn scatter_value_nd_f64(
        &self,
        input: &GpuBufferHandle,
        index: &GpuBufferHandle,
        value: f64,
        input_shape: &[usize],
        index_shape: &[usize],
        dim: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let out = crate::scatter_gather_kernels::gpu_scatter_value_nd_f64(
            Self::unwrap_buffer_f64(input)?,
            Self::unwrap_buffer_i64(index)?.inner(),
            value,
            input_shape,
            index_shape,
            dim,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, ord))
    }

    #[cfg(feature = "cuda")]
    fn scatter_add_nd_f32(
        &self,
        input: &GpuBufferHandle,
        index: &GpuBufferHandle,
        src: &GpuBufferHandle,
        input_shape: &[usize],
        index_shape: &[usize],
        dim: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let out = crate::scatter_gather_kernels::gpu_scatter_add_nd_f32(
            Self::unwrap_buffer(input)?,
            Self::unwrap_buffer_i64(index)?.inner(),
            Self::unwrap_buffer(src)?,
            input_shape,
            index_shape,
            dim,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, ord))
    }

    #[cfg(feature = "cuda")]
    fn scatter_add_nd_f64(
        &self,
        input: &GpuBufferHandle,
        index: &GpuBufferHandle,
        src: &GpuBufferHandle,
        input_shape: &[usize],
        index_shape: &[usize],
        dim: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let out = crate::scatter_gather_kernels::gpu_scatter_add_nd_f64(
            Self::unwrap_buffer_f64(input)?,
            Self::unwrap_buffer_i64(index)?.inner(),
            Self::unwrap_buffer_f64(src)?,
            input_shape,
            index_shape,
            dim,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, ord))
    }

    #[cfg(feature = "cuda")]
    fn scatter_add_segments_f32(
        &self,
        src: &GpuBufferHandle,
        index: &GpuBufferHandle,
        e: usize,
        d: usize,
        dim_size: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(src.device_ordinal())?;
        let ord = src.device_ordinal();
        let out = crate::scatter_gather_kernels::gpu_scatter_add_segments_f32(
            Self::unwrap_buffer(src)?,
            Self::unwrap_buffer_i64(index)?.inner(),
            e,
            d,
            dim_size,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, ord))
    }

    #[cfg(feature = "cuda")]
    fn scatter_add_segments_f64(
        &self,
        src: &GpuBufferHandle,
        index: &GpuBufferHandle,
        e: usize,
        d: usize,
        dim_size: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(src.device_ordinal())?;
        let ord = src.device_ordinal();
        let out = crate::scatter_gather_kernels::gpu_scatter_add_segments_f64(
            Self::unwrap_buffer_f64(src)?,
            Self::unwrap_buffer_i64(index)?.inner(),
            e,
            d,
            dim_size,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, ord))
    }

    #[cfg(feature = "cuda")]
    fn cast_f_to_i(&self, src: &GpuBufferHandle, dst: DType) -> FerrotorchResult<GpuBufferHandle> {
        use crate::cast_kernels as ck;
        let dev = self.device(src.device_ordinal())?;
        let ord = src.device_ordinal();
        match (src.dtype(), dst) {
            (DType::F32, DType::I32) => Ok(Self::wrap_slice_i32(
                ck::cast_f32_to_i32(Self::unwrap_buffer(src)?.inner(), src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (DType::F32, DType::I64) => Ok(Self::wrap_slice_i64(
                ck::cast_f32_to_i64(Self::unwrap_buffer(src)?.inner(), src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (DType::F64, DType::I32) => Ok(Self::wrap_slice_i32(
                ck::cast_f64_to_i32(Self::unwrap_buffer_f64(src)?.inner(), src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (DType::F64, DType::I64) => Ok(Self::wrap_slice_i64(
                ck::cast_f64_to_i64(Self::unwrap_buffer_f64(src)?.inner(), src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (DType::F16, DType::I32) => Ok(Self::wrap_slice_i32(
                ck::cast_f16_to_i32(Self::unwrap_buffer_f16(src)?, src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (DType::F16, DType::I64) => Ok(Self::wrap_slice_i64(
                ck::cast_f16_to_i64(Self::unwrap_buffer_f16(src)?, src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (DType::BF16, DType::I32) => Ok(Self::wrap_slice_i32(
                ck::cast_bf16_to_i32(Self::unwrap_buffer_bf16(src)?, src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (DType::BF16, DType::I64) => Ok(Self::wrap_slice_i64(
                ck::cast_bf16_to_i64(Self::unwrap_buffer_bf16(src)?, src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (s, d) => Err(FerrotorchError::InvalidArgument {
                message: format!("cast_f_to_i: unsupported {s} -> {d}"),
            }),
        }
    }

    #[cfg(feature = "cuda")]
    fn cast_i_to_f(&self, src: &GpuBufferHandle, dst: DType) -> FerrotorchResult<GpuBufferHandle> {
        use crate::cast_kernels as ck;
        let dev = self.device(src.device_ordinal())?;
        let ord = src.device_ordinal();
        match (src.dtype(), dst) {
            (DType::I32, DType::F32) => Ok(Self::wrap_slice_f32(
                ck::cast_i32_to_f32(Self::unwrap_buffer_i32(src)?.inner(), src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (DType::I32, DType::F64) => Ok(Self::wrap_slice_f64(
                ck::cast_i32_to_f64(Self::unwrap_buffer_i32(src)?.inner(), src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (DType::I32, DType::F16) => Ok(Self::wrap_buffer_f16(
                ck::cast_i32_to_f16(Self::unwrap_buffer_i32(src)?.inner(), src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (DType::I32, DType::BF16) => Ok(Self::wrap_buffer_bf16(
                ck::cast_i32_to_bf16(Self::unwrap_buffer_i32(src)?.inner(), src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (DType::I64, DType::F32) => Ok(Self::wrap_slice_f32(
                ck::cast_i64_to_f32(Self::unwrap_buffer_i64(src)?.inner(), src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (DType::I64, DType::F64) => Ok(Self::wrap_slice_f64(
                ck::cast_i64_to_f64(Self::unwrap_buffer_i64(src)?.inner(), src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (DType::I64, DType::F16) => Ok(Self::wrap_buffer_f16(
                ck::cast_i64_to_f16(Self::unwrap_buffer_i64(src)?.inner(), src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (DType::I64, DType::BF16) => Ok(Self::wrap_buffer_bf16(
                ck::cast_i64_to_bf16(Self::unwrap_buffer_i64(src)?.inner(), src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (s, d) => Err(FerrotorchError::InvalidArgument {
                message: format!("cast_i_to_f: unsupported {s} -> {d}"),
            }),
        }
    }

    #[cfg(feature = "cuda")]
    fn cast_i_to_i(&self, src: &GpuBufferHandle, dst: DType) -> FerrotorchResult<GpuBufferHandle> {
        use crate::cast_kernels as ck;
        let dev = self.device(src.device_ordinal())?;
        let ord = src.device_ordinal();
        match (src.dtype(), dst) {
            (DType::I32, DType::I64) => Ok(Self::wrap_slice_i64(
                ck::cast_i32_to_i64(Self::unwrap_buffer_i32(src)?.inner(), src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (DType::I64, DType::I32) => Ok(Self::wrap_slice_i32(
                ck::cast_i64_to_i32(Self::unwrap_buffer_i64(src)?.inner(), src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            // Same-dtype "cast" is a full-value-preserving on-device copy
            // (NO host round trip — `clone_buffer` round-trips via CPU, which
            // §3 forbids here; and a narrow-then-widen would corrupt i64 values
            // outside the i32 range).
            (DType::I32, DType::I32) => Ok(Self::wrap_slice_i32(
                ck::cast_i32_copy(Self::unwrap_buffer_i32(src)?.inner(), src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (DType::I64, DType::I64) => Ok(Self::wrap_slice_i64(
                ck::cast_i64_copy(Self::unwrap_buffer_i64(src)?.inner(), src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (s, d) => Err(FerrotorchError::InvalidArgument {
                message: format!("cast_i_to_i: unsupported {s} -> {d}"),
            }),
        }
    }

    // REQ-8 / issue #29: cross-float cast. This covers bf16/f16 ↔ f32;
    // f32↔f64, bf16↔f16 etc. are tracked in the #29 follow-up issue.
    // Same-dtype is not exposed here — `Tensor::to_dtype<U>()` short-circuits
    // T == U at the public API boundary before this dispatch is reached.
    #[cfg(feature = "cuda")]
    fn cast_f_to_f(&self, src: &GpuBufferHandle, dst: DType) -> FerrotorchResult<GpuBufferHandle> {
        use crate::cast_kernels as ck;
        let dev = self.device(src.device_ordinal())?;
        let ord = src.device_ordinal();
        match (src.dtype(), dst) {
            (DType::BF16, DType::F32) => Ok(Self::wrap_slice_f32(
                ck::cast_bf16_to_f32(Self::unwrap_buffer_bf16(src)?, src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (DType::F32, DType::BF16) => Ok(Self::wrap_buffer_bf16(
                ck::cast_f32_to_bf16(Self::unwrap_buffer(src)?.inner(), src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (DType::F16, DType::F32) => Ok(Self::wrap_slice_f32(
                ck::cast_f16_to_f32(Self::unwrap_buffer_f16(src)?, src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (DType::F32, DType::F16) => Ok(Self::wrap_buffer_f16(
                ck::cast_f32_to_f16(Self::unwrap_buffer(src)?.inner(), src.len(), dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            (s, d) => Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "cast_f_to_f: unsupported {s} -> {d} (tracked in ferrotorch#29 follow-up)"
                ),
            }),
        }
    }

    // ── Boolean / comparison ops — crosslink #1185 Phase 3b ──────────────────
    //
    // `compare` reads the value dtype from `a.dtype()` to pick the kernel
    // (PyTorch-style dispatch on the ScalarType tag) and produces a
    // `DType::Bool`-tagged (u8 0/1) output. Logical ops read/write Bool (u8)
    // buffers. any/all fold a Bool buffer to a 1-element Bool buffer. All keep
    // the result GPU-resident — no host round-trip — and unsupported tags
    // return a structured error (PyTorch parity, rust-gpu-discipline §3).

    #[cfg(feature = "cuda")]
    fn compare(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        op: ferrotorch_core::gpu_dispatch::CompareOp,
    ) -> FerrotorchResult<GpuBufferHandle> {
        use crate::bool_kernels as bk;
        if a.dtype() != b.dtype() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "compare: operand dtypes differ ({} vs {})",
                    a.dtype(),
                    b.dtype()
                ),
            });
        }
        let dev = self.device(a.device_ordinal())?;
        let ord = a.device_ordinal();
        let suffix = op.suffix();
        // #1660: the comparison kernels launch on the LOGICAL element count, not
        // the raw `CudaSlice::len()`. A `.contiguous()`-materialised operand is
        // backed by a POOLED buffer whose raw slice is rounded up to a multiple
        // of `ROUND_ELEMENTS`, while a `clone_htod` operand is exact-length; the
        // two raw lens (e.g. 256 vs 6) then differ even though the logical numels
        // match. We validate the LOGICAL lens here (the dispatch contract) and
        // pass that `n` down so the kernel treats each raw slice as a backing
        // store that need only be `>= n`.
        let n = a.len();
        let n_b = b.len();
        if n != n_b {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("compare: operand numel differs ({n} vs {n_b})"),
            });
        }
        let r = match a.dtype() {
            DType::F32 => bk::gpu_cmp_f32(
                Self::unwrap_buffer(a)?.inner(),
                Self::unwrap_buffer(b)?.inner(),
                n,
                suffix,
                dev,
            ),
            DType::F64 => bk::gpu_cmp_f64(
                Self::unwrap_buffer_f64(a)?.inner(),
                Self::unwrap_buffer_f64(b)?.inner(),
                n,
                suffix,
                dev,
            ),
            DType::I32 => bk::gpu_cmp_i32(
                Self::unwrap_buffer_i32(a)?.inner(),
                Self::unwrap_buffer_i32(b)?.inner(),
                n,
                suffix,
                dev,
            ),
            DType::I64 => bk::gpu_cmp_i64(
                Self::unwrap_buffer_i64(a)?.inner(),
                Self::unwrap_buffer_i64(b)?.inner(),
                n,
                suffix,
                dev,
            ),
            DType::BF16 => bk::gpu_cmp_bf16(
                Self::unwrap_buffer_bf16(a)?,
                Self::unwrap_buffer_bf16(b)?,
                n,
                suffix,
                dev,
            ),
            DType::F16 => bk::gpu_cmp_f16(
                Self::unwrap_buffer_f16(a)?,
                Self::unwrap_buffer_f16(b)?,
                n,
                suffix,
                dev,
            ),
            _ => return Err(FerrotorchError::NotImplementedOnCuda { op: "compare" }),
        };
        Ok(Self::wrap_slice_bool(r.map_err(Self::map_gpu_err)?, ord))
    }

    #[cfg(feature = "cuda")]
    fn bool_and(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        let r = crate::bool_kernels::gpu_and_bool(
            Self::unwrap_buffer_bool(a)?.inner(),
            Self::unwrap_buffer_bool(b)?.inner(),
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_slice_bool(r, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn bool_or(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        let r = crate::bool_kernels::gpu_or_bool(
            Self::unwrap_buffer_bool(a)?.inner(),
            Self::unwrap_buffer_bool(b)?.inner(),
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_slice_bool(r, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn bool_xor(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        let r = crate::bool_kernels::gpu_xor_bool(
            Self::unwrap_buffer_bool(a)?.inner(),
            Self::unwrap_buffer_bool(b)?.inner(),
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_slice_bool(r, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn bool_not(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        let r = crate::bool_kernels::gpu_not_bool(Self::unwrap_buffer_bool(a)?.inner(), dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_slice_bool(r, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn bool_any(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        let r = crate::bool_kernels::gpu_any_bool(Self::unwrap_buffer_bool(a)?.inner(), dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_slice_bool(r, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn bool_all(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(a.device_ordinal())?;
        let r = crate::bool_kernels::gpu_all_bool(Self::unwrap_buffer_bool(a)?.inner(), dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_slice_bool(r, a.device_ordinal()))
    }

    #[cfg(feature = "cuda")]
    fn float_any(
        &self,
        src: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(src.device_ordinal())?;
        let ord = src.device_ordinal();
        let r = match src.dtype() {
            DType::F32 => crate::bool_kernels::gpu_any_f32(
                Self::unwrap_buffer(src)?.inner(),
                outer,
                dim_size,
                inner,
                dev,
            ),
            DType::F64 => crate::bool_kernels::gpu_any_f64(
                Self::unwrap_buffer_f64(src)?.inner(),
                outer,
                dim_size,
                inner,
                dev,
            ),
            DType::F16 => crate::bool_kernels::gpu_any_f16(
                Self::unwrap_buffer_f16(src)?,
                outer,
                dim_size,
                inner,
                dev,
            ),
            DType::BF16 => crate::bool_kernels::gpu_any_bf16(
                Self::unwrap_buffer_bf16(src)?,
                outer,
                dim_size,
                inner,
                dev,
            ),
            other => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("float_any: unsupported dtype {other}"),
                });
            }
        }
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_slice_bool(r, ord))
    }

    #[cfg(feature = "cuda")]
    fn float_all(
        &self,
        src: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(src.device_ordinal())?;
        let ord = src.device_ordinal();
        let r = match src.dtype() {
            DType::F32 => crate::bool_kernels::gpu_all_f32(
                Self::unwrap_buffer(src)?.inner(),
                outer,
                dim_size,
                inner,
                dev,
            ),
            DType::F64 => crate::bool_kernels::gpu_all_f64(
                Self::unwrap_buffer_f64(src)?.inner(),
                outer,
                dim_size,
                inner,
                dev,
            ),
            DType::F16 => crate::bool_kernels::gpu_all_f16(
                Self::unwrap_buffer_f16(src)?,
                outer,
                dim_size,
                inner,
                dev,
            ),
            DType::BF16 => crate::bool_kernels::gpu_all_bf16(
                Self::unwrap_buffer_bf16(src)?,
                outer,
                dim_size,
                inner,
                dev,
            ),
            other => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("float_all: unsupported dtype {other}"),
                });
            }
        }
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_slice_bool(r, ord))
    }

    #[cfg(feature = "cuda")]
    fn float_count_nonzero(
        &self,
        src: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(src.device_ordinal())?;
        let ord = src.device_ordinal();
        let r = match src.dtype() {
            DType::F32 => crate::bool_kernels::gpu_count_nonzero_f32(
                Self::unwrap_buffer(src)?.inner(),
                outer,
                dim_size,
                inner,
                dev,
            ),
            DType::F64 => crate::bool_kernels::gpu_count_nonzero_f64(
                Self::unwrap_buffer_f64(src)?.inner(),
                outer,
                dim_size,
                inner,
                dev,
            ),
            DType::F16 => crate::bool_kernels::gpu_count_nonzero_f16(
                Self::unwrap_buffer_f16(src)?,
                outer,
                dim_size,
                inner,
                dev,
            ),
            DType::BF16 => crate::bool_kernels::gpu_count_nonzero_bf16(
                Self::unwrap_buffer_bf16(src)?,
                outer,
                dim_size,
                inner,
                dev,
            ),
            other => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("float_count_nonzero: unsupported dtype {other}"),
                });
            }
        }
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_slice_i64(r, ord))
    }

    #[cfg(feature = "cuda")]
    fn cast_bool_to_f(
        &self,
        src: &GpuBufferHandle,
        dst: DType,
    ) -> FerrotorchResult<GpuBufferHandle> {
        use crate::cast_kernels as ck;
        if src.dtype() != DType::Bool {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "cast_bool_to_f: src is tagged {}, expected Bool",
                    src.dtype()
                ),
            });
        }
        let dev = self.device(src.device_ordinal())?;
        let ord = src.device_ordinal();
        let inb = Self::unwrap_buffer_bool(src)?.inner();
        match dst {
            DType::F32 => Ok(Self::wrap_slice_f32(
                ck::cast_bool_to_f32(inb, src.len(), dev).map_err(Self::map_gpu_err)?,
                ord,
            )),
            DType::F64 => Ok(Self::wrap_slice_f64(
                ck::cast_bool_to_f64(inb, src.len(), dev).map_err(Self::map_gpu_err)?,
                ord,
            )),
            DType::F16 => Ok(Self::wrap_buffer_f16(
                ck::cast_bool_to_f16(inb, src.len(), dev).map_err(Self::map_gpu_err)?,
                ord,
            )),
            DType::BF16 => Ok(Self::wrap_buffer_bf16(
                ck::cast_bool_to_bf16(inb, src.len(), dev).map_err(Self::map_gpu_err)?,
                ord,
            )),
            d => Err(FerrotorchError::InvalidArgument {
                message: format!("cast_bool_to_f: unsupported Bool -> {d}"),
            }),
        }
    }

    // ── #1545 / #1534: predicate masks for masked-tensor constructors ────────

    #[cfg(feature = "cuda")]
    fn isfinite_mask(&self, input: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        use crate::masked_kernels as mk;
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let r = match input.dtype() {
            DType::F32 => mk::isfinite_mask_f32(Self::unwrap_buffer(input)?.inner(), dev),
            DType::F64 => mk::isfinite_mask_f64(Self::unwrap_buffer_f64(input)?.inner(), dev),
            _ => {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "isfinite_mask",
                });
            }
        };
        Ok(Self::wrap_slice_bool(r.map_err(Self::map_gpu_err)?, ord))
    }

    #[cfg(feature = "cuda")]
    fn ne_scalar_mask(
        &self,
        input: &GpuBufferHandle,
        value: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        use crate::masked_kernels as mk;
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let r = match input.dtype() {
            DType::F32 => {
                mk::ne_scalar_mask_f32(Self::unwrap_buffer(input)?.inner(), value as f32, dev)
            }
            DType::F64 => {
                mk::ne_scalar_mask_f64(Self::unwrap_buffer_f64(input)?.inner(), value, dev)
            }
            _ => {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "ne_scalar_mask",
                });
            }
        };
        Ok(Self::wrap_slice_bool(r.map_err(Self::map_gpu_err)?, ord))
    }

    // ── Phase 3c: mask-driven ops with a GPU-resident Bool mask ──────────────

    #[cfg(feature = "cuda")]
    fn masked_fill_dt(
        &self,
        input: &GpuBufferHandle,
        mask: &GpuBufferHandle,
        value: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        use crate::masked_kernels as mk;
        if mask.dtype() != DType::Bool {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "masked_fill: mask is tagged {}, expected Bool",
                    mask.dtype()
                ),
            });
        }
        // #1661: validate the LOGICAL element counts here (the dispatch
        // contract). A `.contiguous()`-materialised input (a row-narrowed view
        // packed on-device, ferrotorch-core indexing.rs) is backed by a POOLED
        // `CudaSlice` rounded up to a multiple of `ROUND_ELEMENTS`, so its raw
        // slice len (e.g. 256) exceeds its logical numel (6) while the mask is
        // exact-length 6. We pass this logical `n` down so `launch_masked_fill`
        // treats the input slice as a backing store that need only be `>= n` and
        // launches exactly `n` threads, EXACTLY like launch_where/launch_cmp
        // (#1660). Comparing the raw input slice len to the mask len would
        // spuriously reject `256 vs 6`.
        let n = input.len();
        let mask_n = mask.len();
        if n != mask_n {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("masked_fill: input numel {n} != mask numel {mask_n}"),
            });
        }
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let mb = Self::unwrap_buffer_bool(mask)?.inner();
        match input.dtype() {
            DType::F32 => Ok(Self::wrap_slice_f32(
                mk::masked_fill_f32(
                    Self::unwrap_buffer(input)?.inner(),
                    mb,
                    value as f32,
                    n,
                    dev,
                )
                .map_err(Self::map_gpu_err)?,
                ord,
            )),
            DType::F64 => Ok(Self::wrap_slice_f64(
                mk::masked_fill_f64(Self::unwrap_buffer_f64(input)?.inner(), mb, value, n, dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            DType::F16 => Ok(Self::wrap_buffer_f16(
                mk::masked_fill_f16(Self::unwrap_buffer_f16(input)?, mb, value as f32, n, dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            DType::BF16 => Ok(Self::wrap_buffer_bf16(
                mk::masked_fill_bf16(Self::unwrap_buffer_bf16(input)?, mb, value as f32, n, dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            DType::I32 => Ok(Self::wrap_slice_i32(
                mk::masked_fill_i32(
                    Self::unwrap_buffer_i32(input)?.inner(),
                    mb,
                    value as i32,
                    n,
                    dev,
                )
                .map_err(Self::map_gpu_err)?,
                ord,
            )),
            DType::I64 => Ok(Self::wrap_slice_i64(
                mk::masked_fill_i64(
                    Self::unwrap_buffer_i64(input)?.inner(),
                    mb,
                    value as i64,
                    n,
                    dev,
                )
                .map_err(Self::map_gpu_err)?,
                ord,
            )),
            _ => Err(FerrotorchError::NotImplementedOnCuda {
                op: "masked_fill_dt",
            }),
        }
    }

    #[cfg(feature = "cuda")]
    fn where_cond(
        &self,
        cond: &GpuBufferHandle,
        x: &GpuBufferHandle,
        y: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        use crate::masked_kernels as mk;
        if cond.dtype() != DType::Bool {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("where_cond: cond is tagged {}, expected Bool", cond.dtype()),
            });
        }
        if x.dtype() != y.dtype() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "where_cond: x/y dtypes differ ({} vs {})",
                    x.dtype(),
                    y.dtype()
                ),
            });
        }
        if x.len() != y.len() || x.len() != cond.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "where_cond: numel mismatch (cond {}, x {}, y {})",
                    cond.len(),
                    x.len(),
                    y.len()
                ),
            });
        }
        let dev = self.device(x.device_ordinal())?;
        let ord = x.device_ordinal();
        let cb = Self::unwrap_buffer_bool(cond)?.inner();
        // #1660: launch on the LOGICAL element count (already validated equal to
        // cond/y above), not the raw `CudaSlice::len()`. A `.contiguous()`
        // operand is backed by a pooled, over-allocated slice (rounded to
        // `ROUND_ELEMENTS`) while a `clone_htod` operand is exact-length; the
        // kernel treats each raw slice as a backing store of `>= n` elements.
        let n = x.len();
        match x.dtype() {
            DType::F32 => Ok(Self::wrap_slice_f32(
                mk::where_32::<f32>(
                    cb,
                    Self::unwrap_buffer(x)?.inner(),
                    Self::unwrap_buffer(y)?.inner(),
                    n,
                    dev,
                )
                .map_err(Self::map_gpu_err)?,
                ord,
            )),
            DType::F64 => Ok(Self::wrap_slice_f64(
                mk::where_64::<f64>(
                    cb,
                    Self::unwrap_buffer_f64(x)?.inner(),
                    Self::unwrap_buffer_f64(y)?.inner(),
                    n,
                    dev,
                )
                .map_err(Self::map_gpu_err)?,
                ord,
            )),
            DType::F16 => Ok(Self::wrap_buffer_f16(
                mk::where_16(
                    cb,
                    Self::unwrap_buffer_f16(x)?,
                    Self::unwrap_buffer_f16(y)?,
                    n,
                    dev,
                )
                .map_err(Self::map_gpu_err)?,
                ord,
            )),
            DType::BF16 => Ok(Self::wrap_buffer_bf16(
                mk::where_16(
                    cb,
                    Self::unwrap_buffer_bf16(x)?,
                    Self::unwrap_buffer_bf16(y)?,
                    n,
                    dev,
                )
                .map_err(Self::map_gpu_err)?,
                ord,
            )),
            DType::I32 => Ok(Self::wrap_slice_i32(
                mk::where_32::<i32>(
                    cb,
                    Self::unwrap_buffer_i32(x)?.inner(),
                    Self::unwrap_buffer_i32(y)?.inner(),
                    n,
                    dev,
                )
                .map_err(Self::map_gpu_err)?,
                ord,
            )),
            DType::I64 => Ok(Self::wrap_slice_i64(
                mk::where_64::<i64>(
                    cb,
                    Self::unwrap_buffer_i64(x)?.inner(),
                    Self::unwrap_buffer_i64(y)?.inner(),
                    n,
                    dev,
                )
                .map_err(Self::map_gpu_err)?,
                ord,
            )),
            _ => Err(FerrotorchError::NotImplementedOnCuda { op: "where_cond" }),
        }
    }

    #[cfg(feature = "cuda")]
    fn masked_select(
        &self,
        input: &GpuBufferHandle,
        mask: &GpuBufferHandle,
    ) -> FerrotorchResult<(GpuBufferHandle, usize)> {
        use crate::masked_kernels as mk;
        if mask.dtype() != DType::Bool {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "masked_select: mask is tagged {}, expected Bool",
                    mask.dtype()
                ),
            });
        }
        if input.len() != mask.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "masked_select: input numel {} != mask numel {}",
                    input.len(),
                    mask.len()
                ),
            });
        }
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let mb = Self::unwrap_buffer_bool(mask)?.inner();
        match input.dtype() {
            DType::F32 => {
                let (out, len) =
                    mk::masked_select_32::<f32>(Self::unwrap_buffer(input)?.inner(), mb, dev)
                        .map_err(Self::map_gpu_err)?;
                Ok((Self::wrap_slice_f32(out, ord), len))
            }
            DType::F64 => {
                let (out, len) =
                    mk::masked_select_64::<f64>(Self::unwrap_buffer_f64(input)?.inner(), mb, dev)
                        .map_err(Self::map_gpu_err)?;
                Ok((Self::wrap_slice_f64(out, ord), len))
            }
            DType::F16 => {
                let (out, len) = mk::masked_select_16(Self::unwrap_buffer_f16(input)?, mb, dev)
                    .map_err(Self::map_gpu_err)?;
                Ok((Self::wrap_buffer_f16(out, ord), len))
            }
            DType::BF16 => {
                let (out, len) = mk::masked_select_16(Self::unwrap_buffer_bf16(input)?, mb, dev)
                    .map_err(Self::map_gpu_err)?;
                Ok((Self::wrap_buffer_bf16(out, ord), len))
            }
            DType::I32 => {
                let (out, len) =
                    mk::masked_select_32::<i32>(Self::unwrap_buffer_i32(input)?.inner(), mb, dev)
                        .map_err(Self::map_gpu_err)?;
                Ok((Self::wrap_slice_i32(out, ord), len))
            }
            DType::I64 => {
                let (out, len) =
                    mk::masked_select_64::<i64>(Self::unwrap_buffer_i64(input)?.inner(), mb, dev)
                        .map_err(Self::map_gpu_err)?;
                Ok((Self::wrap_slice_i64(out, ord), len))
            }
            _ => Err(FerrotorchError::NotImplementedOnCuda {
                op: "masked_select",
            }),
        }
    }

    #[cfg(feature = "cuda")]
    fn masked_scatter(
        &self,
        grad_compact: &GpuBufferHandle,
        mask: &GpuBufferHandle,
        out_numel: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        use crate::masked_kernels as mk;
        if mask.dtype() != DType::Bool {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "masked_scatter: mask is tagged {}, expected Bool",
                    mask.dtype()
                ),
            });
        }
        if mask.len() != out_numel {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "masked_scatter: mask numel {} != out_numel {}",
                    mask.len(),
                    out_numel
                ),
            });
        }
        let dev = self.device(grad_compact.device_ordinal())?;
        let ord = grad_compact.device_ordinal();
        let mb = Self::unwrap_buffer_bool(mask)?.inner();
        match grad_compact.dtype() {
            DType::F32 => Ok(Self::wrap_slice_f32(
                mk::masked_scatter_32::<f32>(
                    Self::unwrap_buffer(grad_compact)?.inner(),
                    mb,
                    out_numel,
                    dev,
                )
                .map_err(Self::map_gpu_err)?,
                ord,
            )),
            DType::F64 => Ok(Self::wrap_slice_f64(
                mk::masked_scatter_64::<f64>(
                    Self::unwrap_buffer_f64(grad_compact)?.inner(),
                    mb,
                    out_numel,
                    dev,
                )
                .map_err(Self::map_gpu_err)?,
                ord,
            )),
            DType::F16 => Ok(Self::wrap_buffer_f16(
                mk::masked_scatter_16(Self::unwrap_buffer_f16(grad_compact)?, mb, out_numel, dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            DType::BF16 => Ok(Self::wrap_buffer_bf16(
                mk::masked_scatter_16(Self::unwrap_buffer_bf16(grad_compact)?, mb, out_numel, dev)
                    .map_err(Self::map_gpu_err)?,
                ord,
            )),
            DType::I32 => Ok(Self::wrap_slice_i32(
                mk::masked_scatter_32::<i32>(
                    Self::unwrap_buffer_i32(grad_compact)?.inner(),
                    mb,
                    out_numel,
                    dev,
                )
                .map_err(Self::map_gpu_err)?,
                ord,
            )),
            DType::I64 => Ok(Self::wrap_slice_i64(
                mk::masked_scatter_64::<i64>(
                    Self::unwrap_buffer_i64(grad_compact)?.inner(),
                    mb,
                    out_numel,
                    dev,
                )
                .map_err(Self::map_gpu_err)?,
                ord,
            )),
            _ => Err(FerrotorchError::NotImplementedOnCuda {
                op: "masked_scatter",
            }),
        }
    }

    #[cfg(feature = "cuda")]
    fn masked_scatter_forward(
        &self,
        input: &GpuBufferHandle,
        source: &GpuBufferHandle,
        mask: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        use crate::masked_kernels as mk;
        if mask.dtype() != DType::Bool {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "masked_scatter_forward: mask is tagged {}, expected Bool",
                    mask.dtype()
                ),
            });
        }
        if input.dtype() != source.dtype() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "masked_scatter_forward: input dtype {} != source dtype {}",
                    input.dtype(),
                    source.dtype()
                ),
            });
        }
        if input.len() != n || mask.len() != n {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "masked_scatter_forward: input numel {} / mask numel {} != n {}",
                    input.len(),
                    mask.len(),
                    n
                ),
            });
        }
        let dev = self.device(input.device_ordinal())?;
        let ord = input.device_ordinal();
        let mb = Self::unwrap_buffer_bool(mask)?.inner();
        // Single-integer shape sync (the on-device true count), mirroring
        // upstream `masked_scatter_size_check` (`IndexKernel.cu:394-401`): the
        // serial source cursor would over-read `source` if more positions are
        // true than `source` has elements. NOT a data round trip — only the count
        // crosses to host. `source.len()` is the resident source buffer length.
        let true_count = mk::count_true(mb, dev).map_err(Self::map_gpu_err)?;
        if source.len() < true_count {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "masked_scatter: source has {} elements, but mask has {} true positions",
                    source.len(),
                    true_count
                ),
            });
        }
        match input.dtype() {
            DType::F32 => Ok(Self::wrap_slice_f32(
                mk::masked_scatter_forward_32::<f32>(
                    Self::unwrap_buffer(input)?.inner(),
                    Self::unwrap_buffer(source)?.inner(),
                    mb,
                    n,
                    dev,
                )
                .map_err(Self::map_gpu_err)?,
                ord,
            )),
            DType::F64 => Ok(Self::wrap_slice_f64(
                mk::masked_scatter_forward_64::<f64>(
                    Self::unwrap_buffer_f64(input)?.inner(),
                    Self::unwrap_buffer_f64(source)?.inner(),
                    mb,
                    n,
                    dev,
                )
                .map_err(Self::map_gpu_err)?,
                ord,
            )),
            _ => Err(FerrotorchError::NotImplementedOnCuda {
                op: "masked_scatter_forward",
            }),
        }
    }

    fn broadcast_bool(
        &self,
        mask: &GpuBufferHandle,
        in_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        if mask.dtype() != DType::Bool {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "broadcast_bool: mask is tagged {}, expected Bool",
                    mask.dtype()
                ),
            });
        }
        let in_numel: usize = if in_shape.is_empty() {
            1
        } else {
            in_shape.iter().product()
        };
        if mask.len() != in_numel {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "broadcast_bool: mask numel {} != product(in_shape {:?}) = {in_numel}",
                    mask.len(),
                    in_shape,
                ),
            });
        }
        let out_ndim = out_shape.len();
        let in_ndim = in_shape.len();
        if in_ndim > out_ndim {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "broadcast_bool: input ndim {in_ndim} > target ndim {out_ndim} \
                     (shapes {in_shape:?} -> {out_shape:?})"
                ),
            });
        }
        // Per-output-dim broadcast input element strides. Walk output dims;
        // right-align the input. The input element stride for a dim is the
        // contiguous input stride where `in_dim == out_dim`, or 0 where the
        // input dim is size-1 or absent (the NumPy / torch broadcast rule that
        // the CPU `broadcast_in_flat` index map encodes).
        let mut in_contig = vec![0usize; in_ndim];
        if in_ndim > 0 {
            in_contig[in_ndim - 1] = 1;
            for d in (0..in_ndim - 1).rev() {
                in_contig[d] = in_contig[d + 1] * in_shape[d + 1];
            }
        }
        let mut src_strides = vec![0usize; out_ndim];
        for (d_off, src_stride_slot) in src_strides.iter_mut().rev().enumerate() {
            let out_dim = out_shape[out_ndim - 1 - d_off];
            if d_off < in_ndim {
                let d_in = in_ndim - 1 - d_off;
                let in_dim = in_shape[d_in];
                if in_dim == 1 {
                    // broadcast — stride 0
                } else if in_dim == out_dim {
                    *src_stride_slot = in_contig[d_in];
                } else {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "broadcast_bool: cannot broadcast {in_shape:?} -> {out_shape:?} \
                             (axis {} mismatch: {in_dim} vs {out_dim})",
                            out_ndim - 1 - d_off
                        ),
                    });
                }
            }
            // d_off >= in_ndim: leading output axis with no input counterpart —
            // stride stays 0 (replicate).
        }
        let ord = mask.device_ordinal();
        let dev = self.device(ord)?;
        let mb = Self::unwrap_buffer_bool(mask)?.inner();
        let out = crate::bool_kernels::gpu_broadcast_bool(mb, out_shape, &src_strides, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_slice_bool(out, ord))
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Get the `GpuDevice` from the registered CUDA backend.
///
/// This retrieves the device that was created during [`init_cuda_backend`],
/// ensuring all kernel modules and cuBLAS handles are shared. Creating a
/// second `GpuDevice` via `GpuDevice::new(0)` would create a separate
/// CUDA context with its own module cache, which is not interoperable.
pub fn get_cuda_device() -> FerrotorchResult<Arc<GpuDevice>> {
    let backend =
        ferrotorch_core::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    // The global backend is a &dyn GpuBackend. We know it's CudaBackendImpl
    // because init_cuda_backend registered it. Downcast via Any.
    let cuda_backend = backend.as_any().downcast_ref::<CudaBackendImpl>().ok_or(
        FerrotorchError::InvalidArgument {
            message: "registered GPU backend is not CudaBackendImpl".into(),
        },
    )?;
    Ok(Arc::clone(cuda_backend.default_device()?))
}

/// Initialize the CUDA backend and register it with ferrotorch-core.
///
/// This must be called before any GPU tensor operations. It creates a
/// [`CudaBackendImpl`] (initializing CUDA device 0) and registers it via
/// [`ferrotorch_core::gpu_dispatch::register_gpu_backend`].
///
/// Calling this a second time returns an error (the backend is already
/// registered).
///
/// # Errors
///
/// - [`FerrotorchError::InvalidArgument`] if CUDA initialization fails.
/// - [`FerrotorchError::InvalidArgument`] if a GPU backend is already registered.
pub fn init_cuda_backend() -> FerrotorchResult<()> {
    // Idempotent: if already registered, return Ok silently.
    if ferrotorch_core::gpu_dispatch::has_gpu_backend() {
        return Ok(());
    }
    let backend = CudaBackendImpl::new()?;
    // OnceLock::set can still race if two threads call init concurrently —
    // if that happens, the second set() fails but the backend is registered
    // by the first. We treat that as success.
    let _ = ferrotorch_core::gpu_dispatch::register_gpu_backend(Box::new(backend));
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use ferrotorch_core::gpu_dispatch;

    // Note: Because `register_gpu_backend` uses a `OnceLock`, only the first
    // test to call `init_cuda_backend()` will succeed at registration. The
    // others will see the backend as already registered. We handle this by
    // checking `has_gpu_backend()` before calling init.

    /// Ensure the backend can be initialized (or was already initialized).
    fn ensure_init() {
        if !gpu_dispatch::has_gpu_backend() {
            init_cuda_backend().expect("init_cuda_backend");
        }
    }

    #[test]
    fn test_init_cuda_backend() {
        // First call succeeds (or backend was already registered by another test).
        ensure_init();
        assert!(gpu_dispatch::has_gpu_backend());
    }

    #[test]
    fn test_gpu_backend_returns_some() {
        ensure_init();
        assert!(gpu_dispatch::gpu_backend().is_some());
    }

    #[test]
    fn test_roundtrip_cpu_gpu_cpu() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend registered");

        let host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // SAFETY:
        // - `host` is a `Vec<f32>` whose backing allocation is at least
        //   4-byte-aligned (Vec follows the alignment of T = f32, per
        //   `Vec::as_ptr` documentation). Reinterpreting the pointer as
        //   `*const u8` is always sound: u8 has alignment 1, so any pointer
        //   alignment is sufficient for u8 access.
        // - The byte length `host.len() * size_of::<f32>()` exactly matches
        //   the byte extent of the f32 allocation (5 elements × 4 bytes =
        //   20 bytes); the resulting slice never overruns the allocation.
        // - Every f32 bit pattern is a valid u8 sequence (u8 has no invalid
        //   bit patterns), so the byte read is well-defined for all elements
        //   including IEEE 754 NaNs.
        // - Lifetime: the resulting `&[u8]` is bound by `host` (used on the
        //   very next line as `cpu_to_gpu(bytes, ...)` and not escaping the
        //   test scope); `host` is not dropped before line 3176.
        // - No `&mut` aliases: `host` is read via `host.as_ptr()` (shared);
        //   no `&mut [f32]` to the same allocation exists.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                host.as_ptr() as *const u8,
                host.len() * std::mem::size_of::<f32>(),
            )
        };

        let handle = backend
            .cpu_to_gpu(bytes, DType::F32, 0)
            .expect("cpu_to_gpu");
        assert_eq!(handle.len(), 5);
        assert_eq!(handle.device_ordinal(), 0);

        let back_bytes = backend.gpu_to_cpu(&handle).expect("gpu_to_cpu");
        // SAFETY:
        // - `back_bytes` is a `Vec<u8>` returned by `gpu_to_cpu` (line 245)
        //   which constructs it via `Vec::from_raw_parts` from a
        //   `Vec<f32>::ManuallyDrop` (see line 256-262 in this file). The
        //   resulting `Vec<u8>` therefore inherits f32's 4-byte alignment
        //   on the original allocation pointer.
        // - `back_bytes.len() / 4` is the original f32 element count (the
        //   byte length is `host.len() * 4 == 20`, so divided by 4 yields
        //   `5`, the original `host.len()`).
        // - Every u32 bit pattern is a valid f32 (including all NaN
        //   payloads) — the round-trip CUDA memcpy preserves bytes exactly,
        //   so reinterpreting the bytes as f32 yields the original values.
        // - Lifetime: `&[f32]` is bound to `&back_bytes`; both are dropped
        //   at end of scope after the `assert_eq!` on line 3184.
        // - No `&mut` aliases: `back_bytes` is read via `as_ptr()` only.
        let back: &[f32] = unsafe {
            std::slice::from_raw_parts(back_bytes.as_ptr() as *const f32, back_bytes.len() / 4)
        };
        assert_eq!(back, &host[..]);
    }

    #[test]
    fn test_add_f32() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend registered");

        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let expected: Vec<f32> = vec![11.0, 22.0, 33.0, 44.0];

        // SAFETY:
        // - `a_data` is a `Vec<f32>` (4 elements, 4-byte-aligned allocation).
        //   Reinterpret as `*const u8` always sound (u8 alignment is 1; f32
        //   alignment 4 ≥ 1).
        // - Length `a_data.len() * 4 == 16` matches the f32 byte extent
        //   exactly; no overrun.
        // - Every f32 bit pattern is a valid u8 byte sequence; bytes are
        //   well-defined for all stored values (1.0..4.0 here).
        // - Lifetime: `&[u8]` borrows `a_data`'s allocation; `a_data` lives
        //   to the end of this test function (line 3216), and the slice is
        //   consumed by `cpu_to_gpu` on line 3201 before reuse.
        // - No `&mut` aliases: `a_data` is accessed via shared `as_ptr()`.
        let a_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(a_data.as_ptr() as *const u8, a_data.len() * 4) };
        // SAFETY:
        // - `b_data` is a `Vec<f32>` (4 elements, 4-byte-aligned). All
        //   alignment, length, validity, lifetime, and aliasing properties
        //   are identical to `a_bytes` above. The byte length
        //   `b_data.len() * 4 == 16` matches the f32 byte extent exactly.
        // - The `&[u8]` is consumed by `cpu_to_gpu` on line 3202 before
        //   `b_data` is read again.
        let b_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(b_data.as_ptr() as *const u8, b_data.len() * 4) };

        let a_handle = backend
            .cpu_to_gpu(a_bytes, DType::F32, 0)
            .expect("cpu_to_gpu a");
        let b_handle = backend
            .cpu_to_gpu(b_bytes, DType::F32, 0)
            .expect("cpu_to_gpu b");

        let result = backend.add_f32(&a_handle, &b_handle).expect("add_f32");
        assert_eq!(result.len(), 4);

        let result_bytes = backend.gpu_to_cpu(&result).expect("gpu_to_cpu");
        // SAFETY:
        // - `result_bytes` is a `Vec<u8>` constructed by `gpu_to_cpu` (line
        //   245) via `Vec::from_raw_parts` from a `Vec<f32>::ManuallyDrop`
        //   so its allocation pointer inherits 4-byte alignment from the
        //   original f32 allocation.
        // - `result_bytes.len() / 4` is the original f32 element count
        //   (`add_f32` produces 4 f32s → 16 bytes / 4 = 4 elements), so the
        //   reinterpreted slice covers exactly the f32 region.
        // - Every u32 bit pattern is a valid f32 (the bytes that
        //   `gpu_to_cpu` returned are exactly those produced by the GPU
        //   `gpu_add` kernel, so each 4-byte word is a valid IEEE 754 f32).
        // - Lifetime: `&[f32]` is bound by `&result_bytes`; the iterator on
        //   line 3212 consumes the slice within the same scope.
        // - No `&mut` aliases: `result_bytes` is accessed only via shared
        //   `as_ptr()`.
        let result_f32: &[f32] = unsafe {
            std::slice::from_raw_parts(result_bytes.as_ptr() as *const f32, result_bytes.len() / 4)
        };

        for (i, (&got, &exp)) in result_f32.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "element {i}: got {got}, expected {exp}",
            );
        }
    }

    #[test]
    fn test_matmul_f32() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend registered");

        // A = [[1, 2, 3],
        //      [4, 5, 6]]  (2x3)
        // B = [[7, 8],
        //      [9, 10],
        //      [11, 12]]   (3x2)
        // C = [[58, 64],
        //      [139, 154]] (2x2)
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let expected: Vec<f32> = vec![58.0, 64.0, 139.0, 154.0];

        // SAFETY:
        // - `a_data` is a `Vec<f32>` of 6 elements (24 bytes total). f32
        //   alignment 4 ≥ u8 alignment 1, so the pointer cast is sound.
        // - `a_data.len() * 4 == 24` exactly matches the f32 byte extent;
        //   no overrun, no misalignment.
        // - All values (1.0..=6.0) are finite; every f32 byte sequence is a
        //   valid u8 sequence regardless.
        // - Lifetime: `&[u8]` is bound by `a_data` through `cpu_to_gpu` on
        //   line 3242 (within the same test scope). `a_data` is not dropped
        //   before line 3252.
        // - No `&mut` aliases: shared `as_ptr()` access only.
        let a_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(a_data.as_ptr() as *const u8, a_data.len() * 4) };
        // SAFETY:
        // - `b_data` is a `Vec<f32>` of 6 elements; same alignment,
        //   length-exactness, validity, lifetime, and aliasing properties
        //   as `a_bytes` above. Byte length `b_data.len() * 4 == 24`
        //   matches the f32 byte extent exactly.
        // - Slice is consumed by `cpu_to_gpu` on line 3243.
        let b_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(b_data.as_ptr() as *const u8, b_data.len() * 4) };

        let a_handle = backend
            .cpu_to_gpu(a_bytes, DType::F32, 0)
            .expect("cpu_to_gpu a");
        let b_handle = backend
            .cpu_to_gpu(b_bytes, DType::F32, 0)
            .expect("cpu_to_gpu b");

        let result = backend
            .matmul_f32(&a_handle, &b_handle, 2, 3, 2)
            .expect("matmul_f32");
        assert_eq!(result.len(), 4);

        let result_bytes = backend.gpu_to_cpu(&result).expect("gpu_to_cpu");
        // SAFETY:
        // - `result_bytes` is a `Vec<u8>` returned by `gpu_to_cpu` (line
        //   245), constructed via `Vec::from_raw_parts` from a
        //   `ManuallyDrop<Vec<f32>>` (lines 256-262), so its underlying
        //   allocation has f32's 4-byte alignment.
        // - `result_bytes.len() / 4` is the original f32 element count
        //   (matmul produces a 2×2 matrix → 4 f32 → 16 bytes; 16/4 = 4).
        // - Every u32 bit pattern is a valid f32; the bytes were produced
        //   by the cuBLAS GEMM kernel (so each 4-byte word is a finite
        //   IEEE 754 f32 modulo NaN, all of which are valid f32 values).
        // - Lifetime: `&[f32]` is bound by `&result_bytes`; the iterator on
        //   line 3284 consumes the slice in the same scope and `result_bytes`
        //   outlives it.
        // - No `&mut` aliases: shared `as_ptr()` only.
        let result_f32: &[f32] = unsafe {
            std::slice::from_raw_parts(result_bytes.as_ptr() as *const f32, result_bytes.len() / 4)
        };

        for (i, (&got, &exp)) in result_f32.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "element {i}: got {got}, expected {exp}",
            );
        }
    }

    /// Round-trip `Vec<u16>` (bf16 bit patterns) → CudaSlice<u16> → `Vec<u16>`
    /// via the type-erased dispatcher. Reproduces the upstream-decode
    /// failure mode (#15 / #18: "gpu_to_cpu: handle is neither
    /// CudaBuffer<f32> nor CudaBuffer<f64>") and confirms the bf16 branch
    /// now matches.
    #[test]
    fn test_roundtrip_bf16() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend registered");

        // bf16 bit patterns for {0.0, 1.0, -1.0, 2.5, -3.5, 100.0}. We
        // pre-quantize via half::bf16 so the round-trip is lossless (no
        // f32→bf16 conversion kernel involved).
        let host: Vec<u16> = [0.0_f32, 1.0, -1.0, 2.5, -3.5, 100.0]
            .iter()
            .map(|&x| half::bf16::from_f32(x).to_bits())
            .collect();

        // SAFETY:
        // - `host` is a `Vec<u16>` (2-byte-aligned allocation). Cast to
        //   `*const u8` is sound (u8 align 1 ≤ u16 align 2).
        // - `host.len() * 2 == 12` matches the u16 byte extent exactly.
        // - The slice is consumed by `cpu_to_gpu` before `host` is reused.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                host.as_ptr() as *const u8,
                host.len() * std::mem::size_of::<u16>(),
            )
        };

        let handle = backend
            .cpu_to_gpu(bytes, DType::BF16, 0)
            .expect("cpu_to_gpu bf16");
        assert_eq!(handle.len(), host.len());
        assert_eq!(handle.device_ordinal(), 0);
        assert_eq!(backend.buffer_elem_size(&handle), 2);

        let back_bytes = backend.gpu_to_cpu(&handle).expect("gpu_to_cpu bf16");
        assert_eq!(back_bytes.len(), host.len() * 2);

        // SAFETY:
        // - `back_bytes` was constructed by `gpu_to_cpu` (bf16 branch) via
        //   `Vec::from_raw_parts` from a `ManuallyDrop<Vec<u16>>`; the
        //   allocation pointer inherits u16's 2-byte alignment.
        // - `back_bytes.len() / 2 == host.len()` covers exactly the u16
        //   byte extent.
        // - Every u16 bit pattern is a valid u16 — the bytes are unchanged
        //   by the CUDA memcpy.
        let back: &[u16] = unsafe {
            std::slice::from_raw_parts(back_bytes.as_ptr() as *const u16, back_bytes.len() / 2)
        };
        assert_eq!(back, &host[..]);

        // clone_buffer must also work on bf16 handles (#15 / #18 ask for
        // a full round-trip including device-side clones).
        let cloned = backend.clone_buffer(&handle).expect("clone_buffer bf16");
        assert_eq!(cloned.len(), host.len());
        let cloned_bytes = backend.gpu_to_cpu(&cloned).expect("gpu_to_cpu cloned");
        let cloned_back: &[u16] = unsafe {
            std::slice::from_raw_parts(cloned_bytes.as_ptr() as *const u16, cloned_bytes.len() / 2)
        };
        assert_eq!(cloned_back, &host[..]);
    }

    /// Round-trip f16 (IEEE float16) through the type-erased dispatcher and
    /// prove the f16/bf16 DType-tag disambiguation (crosslink #1185 Phase 1).
    ///
    /// Both dtypes store as `CudaSlice<u16>`, so the ONLY thing keeping them
    /// apart is the `GpuBufferHandle` tag. This test asserts:
    ///   1. an f16 upload is tagged `DType::F16` and round-trips bit-exact;
    ///   2. an F16-tagged handle fed to a bf16 backend op returns `Err`
    ///      (NOT garbage) — i.e. `unwrap_buffer_bf16` rejects the F16 tag;
    ///   3. a BF16-tagged handle fed to an f16 backend op returns `Err` too.
    #[test]
    fn test_f16_bf16_tag_disambiguation() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend registered");

        // f16 bit patterns for {0.0, 1.0, -1.0, 2.5, -3.5, 100.0}.
        let host: Vec<u16> = [0.0_f32, 1.0, -1.0, 2.5, -3.5, 100.0]
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        // SAFETY: `host` is a 2-byte-aligned `Vec<u16>`; cast to `*const u8`
        // is sound and `host.len() * 2` matches the byte extent exactly. The
        // slice is consumed by `cpu_to_gpu` before `host` is reused.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                host.as_ptr() as *const u8,
                host.len() * std::mem::size_of::<u16>(),
            )
        };

        let f16_handle = backend
            .cpu_to_gpu(bytes, DType::F16, 0)
            .expect("cpu_to_gpu f16");
        // (1) tag is F16, NOT BF16; elem_size derives from the tag.
        assert_eq!(f16_handle.dtype(), DType::F16);
        assert_ne!(f16_handle.dtype(), DType::BF16);
        assert_eq!(backend.buffer_elem_size(&f16_handle), 2);

        // f16 round-trips bit-exact through the F16 gpu_to_cpu branch.
        let back_bytes = backend.gpu_to_cpu(&f16_handle).expect("gpu_to_cpu f16");
        // SAFETY: `back_bytes` came from `gpu_to_cpu` via from_raw_parts on a
        // `ManuallyDrop<Vec<u16>>` (2-byte aligned); `len / 2 == host.len()`.
        let back: &[u16] = unsafe {
            std::slice::from_raw_parts(back_bytes.as_ptr() as *const u16, back_bytes.len() / 2)
        };
        assert_eq!(back, &host[..], "f16 round-trip must be bit-exact");

        // (2) Feed the F16-tagged handle to a *bf16* backend op. The bf16
        // path unwraps via `unwrap_buffer_bf16`, whose tag check must reject
        // the F16 tag → structured Err, never silent garbage.
        let mismatch = backend.add_bf16_bf16(&f16_handle, &f16_handle);
        assert!(
            mismatch.is_err(),
            "F16-tagged handle fed to add_bf16_bf16 must Err, got Ok"
        );
        if let Err(e) = mismatch {
            let msg = format!("{e}");
            assert!(
                msg.contains("BF16") || msg.contains("expected BF16"),
                "error must name the BF16 tag mismatch, got: {msg}"
            );
        }

        // (3) Symmetric: a BF16-tagged handle fed to an f16 backend op Errs.
        let bf16_host: Vec<u16> = [0.0_f32, 1.0]
            .iter()
            .map(|&x| half::bf16::from_f32(x).to_bits())
            .collect();
        // SAFETY: same as `bytes` above with a 2-element u16 source.
        let bf16_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                bf16_host.as_ptr() as *const u8,
                bf16_host.len() * std::mem::size_of::<u16>(),
            )
        };
        let bf16_handle = backend
            .cpu_to_gpu(bf16_bytes, DType::BF16, 0)
            .expect("cpu_to_gpu bf16");
        assert_eq!(bf16_handle.dtype(), DType::BF16);
        let rev_mismatch = backend.add_f16(&bf16_handle, &bf16_handle);
        assert!(
            rev_mismatch.is_err(),
            "BF16-tagged handle fed to add_f16 must Err, got Ok"
        );
        if let Err(e) = rev_mismatch {
            let msg = format!("{e}");
            assert!(
                msg.contains("F16") || msg.contains("expected F16"),
                "error must name the F16 tag mismatch, got: {msg}"
            );
        }
    }

    /// Allocate a zeroed bf16 buffer via the dispatcher and confirm the
    /// elem_size=2 branch wires through to `alloc_zeros_bf16`.
    #[test]
    fn test_alloc_zeros_bf16() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend registered");
        let handle = backend
            .alloc_zeros(8, DType::BF16, 0)
            .expect("alloc_zeros bf16");
        assert_eq!(handle.len(), 8);
        assert_eq!(backend.buffer_elem_size(&handle), 2);

        let bytes = backend.gpu_to_cpu(&handle).expect("gpu_to_cpu zeros");
        assert_eq!(bytes.len(), 16);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    /// 4×4 bf16 matmul via the type-erased dispatcher (matmul_bf16_bf16).
    /// Inputs and outputs are bf16-stored; the cuBLAS path accumulates in
    /// f32 and rounds back to bf16 on store. Tolerance accounts for the
    /// ~3e-3 bf16 mantissa quantum on each accumulator output.
    #[test]
    fn test_matmul_bf16_bf16_dispatcher() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend registered");

        // A: 2x3, B: 3x2, expected C: 2x2 = [[58, 64], [139, 154]] (matches
        // test_matmul_f32 layout so the contract maps 1:1).
        let a_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_f32: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let expected_f32: Vec<f32> = vec![58.0, 64.0, 139.0, 154.0];

        let a_bf16: Vec<u16> = a_f32
            .iter()
            .map(|&x| half::bf16::from_f32(x).to_bits())
            .collect();
        let b_bf16: Vec<u16> = b_f32
            .iter()
            .map(|&x| half::bf16::from_f32(x).to_bits())
            .collect();

        // SAFETY: u16 → u8 byte view (u8 align 1 ≤ u16 align 2; byte length
        // exactly matches u16 byte extent; slice consumed before reuse).
        let a_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(a_bf16.as_ptr() as *const u8, a_bf16.len() * 2) };
        let b_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(b_bf16.as_ptr() as *const u8, b_bf16.len() * 2) };

        let a_handle = backend
            .cpu_to_gpu(a_bytes, DType::BF16, 0)
            .expect("cpu_to_gpu a");
        let b_handle = backend
            .cpu_to_gpu(b_bytes, DType::BF16, 0)
            .expect("cpu_to_gpu b");

        let result = backend
            .matmul_bf16_bf16(&a_handle, &b_handle, 2, 3, 2)
            .expect("matmul_bf16_bf16");
        assert_eq!(result.len(), 4);
        assert_eq!(backend.buffer_elem_size(&result), 2);

        let result_bytes = backend.gpu_to_cpu(&result).expect("gpu_to_cpu result");
        // SAFETY: result was wrapped as CudaSlice<u16>; gpu_to_cpu's bf16
        // branch returns a u16-aligned byte buffer of length 8 (4 elements
        // × 2 bytes), and every u16 bit pattern is a valid bf16.
        let result_bf16: &[u16] = unsafe {
            std::slice::from_raw_parts(result_bytes.as_ptr() as *const u16, result_bytes.len() / 2)
        };

        // Convert bf16 → f32 for comparison; bf16's 8-bit mantissa puts the
        // ULP at this scale (~58..154) around 0.5, so tolerance 1.0 is
        // generous-but-tight (matches the existing f32→bf16→cuBLAS path's
        // tolerance bands in blas.rs).
        for (i, (&got_bits, &exp)) in result_bf16.iter().zip(expected_f32.iter()).enumerate() {
            let got = half::bf16::from_bits(got_bits).to_f32();
            assert!(
                (got - exp).abs() < 1.0,
                "element {i}: got {got}, expected {exp}"
            );
        }
    }
}
