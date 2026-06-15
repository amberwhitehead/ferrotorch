//! ## REQ status (per `.design/ferrotorch-core/storage.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl `enum StorageBuffer` 4 variants; non-test consumer `Tensor::storage` plus every variant-dispatched op. |
//! | REQ-2 | SHIPPED | impl `trait CubeStorageHandle`; non-test consumer ferrotorch-cubecl concrete impl + `Tensor::cubecl_handle`. |
//! | REQ-3 | SHIPPED | constructors `cpu`, `gpu`, `xpu_from_handle`, `meta`, `meta_filled`, `on_device`, `on_device_pinned`; non-test consumers across `creation::*` and `Tensor::to`. |
//! | REQ-4 | SHIPPED | impl `try_as_slice`, `try_as_mut_slice` (+ crate-internal `try_as_mut_slice_aliased` post-CORE-001); non-test consumer `Tensor::data` (`try_as_slice`), `Tensor::data_mut` (`try_as_mut_slice_aliased`). |
//! | REQ-5 | SHIPPED | impl variant predicates + handle accessors; non-test consumer every CUDA-dispatched op via `is_gpu` / `gpu_handle`. |
//! | REQ-6 | SHIPPED | impl `try_clone` + `Clone`; non-test consumer `Tensor::accumulate_grad`. |
//! | REQ-7 | SHIPPED | impl `try_clone_subregion`; non-test consumer `Tensor::into_storage_and_shape`. |
//! | REQ-8 | SHIPPED | impl `Drop`; non-test consumer every CPU temp tensor returning Vec to `cpu_pool::pool_return_cpu`. |
//! | REQ-9 | SHIPPED | impl `meta_fill_value`; non-test consumer `Tensor::meta_fill_value` for `creation::full_meta` round-trip. |
//!
//! ## Aliasing & synchronization contract (CORE-001 / #1695)
//!
//! `Tensor` clones and views share their `TensorStorage` through `Arc`s
//! (PyTorch parity: `c10/core/StorageImpl.h:38` — aliasing storages are the
//! design), and in-place ops mutate through those aliased handles. The crate
//! deliberately chose **interior mutability + a documented residual
//! contract** over rejecting aliased mutation outright (which would break
//! the PyTorch semantic model where `t.clone(); t.fill_(0.)` is the normal
//! case):
//!
//! - The buffer sits behind `UnsafeCell` ([`TensorStorage`]`.data`) and CPU
//!   elements additionally sit in element-level cells ([`CpuBuffer`]). Every
//!   aliased mutation goes through raw pointers derived from those cells —
//!   **never** through a `&mut` manufactured behind an aliased `Arc` — so
//!   all sequenced aliased read/write patterns are UB-free. The Miri gate
//!   `ferrotorch-core/tests/audit_core001_aliasing_miri.rs` pins them.
//! - **Residual contract** (caller obligations, identical to PyTorch's
//!   informal storage contract; violations are caller bugs, not library UB
//!   sanctioned by this module):
//!   1. A `&[T]` obtained from `try_as_slice` / `Tensor::data()` must not
//!      be *used* across a mutation of the same buffer performed through
//!      another alias — re-borrow after the write instead.
//!   2. Cross-thread access requires external synchronization when at least
//!      one side writes (single-thread mutation XOR synchronization).
//!
//! The one operation that cannot be expressed under this model — resizing
//! (shape-metadata rewrite + different-length buffer swap) while aliased —
//! returns a structured error instead (`Tensor::update_storage_and_shape`).

use std::cell::UnsafeCell;

use crate::device::Device;
use crate::dtype::Element;
use crate::gpu_dispatch::GpuBufferHandle;
use crate::shape::checked_byte_count;

// ---------------------------------------------------------------------------
// CpuBuffer — interior-mutable CPU element buffer
// ---------------------------------------------------------------------------

/// CPU element buffer with *interior mutability* at the element level.
///
/// `Tensor` shares its `TensorStorage` through `Arc`s (clones and views
/// alias the same buffer — PyTorch parity per `c10/core/StorageImpl.h:38`,
/// "storage is supposed to uniquely own a data pointer; two non-null data
/// pointers alias if and only if they are from the same storage"). In-place
/// ops therefore must write through *aliased* handles. Manufacturing a
/// `&mut [T]` behind an aliased `Arc` is undefined behavior (CORE-001 /
/// #1695); instead the elements live in [`UnsafeCell`]s and every aliased
/// write goes through raw pointers derived via [`UnsafeCell::raw_get`],
/// which the aliasing model sanctions for shared-readwrite access.
///
/// # Synchronization contract (documented, PyTorch-equivalent)
///
/// The cells make aliased mutation *expressible without instant UB*; they do
/// not make it race-free. Callers must uphold, exactly as in PyTorch:
///
/// 1. **Single-thread mutation XOR external synchronization** — concurrent
///    unsynchronized access from multiple threads where at least one writes
///    is a data race.
/// 2. **Sequenced borrows** — a `&[T]` obtained from [`Self::as_slice`] (or
///    `Tensor::data()`) must not be *used* across a write performed through
///    another alias; re-borrow after the write instead. (Miri flags
///    violations; see `ferrotorch-core/tests/audit_core001_aliasing_miri.rs`.)
pub struct CpuBuffer<T: Element> {
    cells: Vec<UnsafeCell<T>>,
}

impl<T: Element> CpuBuffer<T> {
    /// Wrap an owned `Vec<T>` without copying.
    pub(crate) fn from_vec(v: Vec<T>) -> Self {
        let mut v = std::mem::ManuallyDrop::new(v);
        let (ptr, len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
        // SAFETY: `UnsafeCell<T>` is guaranteed to have the same in-memory
        // representation (size, alignment, ABI) as `T` (std documents this
        // on `UnsafeCell`; it is `#[repr(transparent)]`). Therefore
        // `Layout::array::<UnsafeCell<T>>(cap) == Layout::array::<T>(cap)`
        // and rebuilding the Vec over the same allocation with the same
        // (len, cap) is sound. Ownership of the allocation moved into the
        // new Vec; `ManuallyDrop` prevents a double-free.
        let cells = unsafe { Vec::from_raw_parts(ptr.cast::<UnsafeCell<T>>(), len, cap) };
        Self { cells }
    }

    /// Unwrap back into an owned `Vec<T>` without copying (used by the
    /// `Drop` impl of [`TensorStorage`] to return buffers to the CPU pool).
    pub(crate) fn into_vec(self) -> Vec<T> {
        let mut cells = std::mem::ManuallyDrop::new(self.cells);
        let (ptr, len, cap) = (cells.as_mut_ptr(), cells.len(), cells.capacity());
        // SAFETY: exact inverse of `from_vec` — same layout-equality
        // guarantee, same ownership transfer.
        unsafe { Vec::from_raw_parts(ptr.cast::<T>(), len, cap) }
    }

    /// Number of elements.
    pub(crate) fn len(&self) -> usize {
        self.cells.len()
    }

    /// Whether the buffer holds zero elements.
    pub(crate) fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Raw base pointer to the first element.
    ///
    /// Deriving the pointer is safe; reads/writes through it must uphold
    /// the synchronization contract in the type-level docs. The pointer is
    /// obtained via [`UnsafeCell::raw_get`], so it carries write provenance
    /// even when derived from a shared `&CpuBuffer` — this is what lets
    /// in-place ops write through aliased `Arc<TensorStorage>` handles
    /// without ever creating a `&mut` to shared memory.
    pub(crate) fn base_ptr(&self) -> *mut T {
        UnsafeCell::raw_get(self.cells.as_ptr())
    }

    /// Borrow the elements as a plain shared slice.
    ///
    /// Sound while no write overlaps the *use* of the returned borrow (the
    /// type-level synchronization contract, rule 2). All sequenced
    /// read-after-write access patterns are fine; holding the slice across
    /// an aliased write and then reading it is not.
    pub(crate) fn as_slice(&self) -> &[T] {
        // SAFETY: the cells are initialized `T`s; the pointer and length
        // come from the live Vec. Creating `&[T]` over `UnsafeCell`
        // contents is sound provided no mutation overlaps the borrow's
        // use, which is the documented contract above.
        unsafe { std::slice::from_raw_parts(self.base_ptr().cast_const(), self.cells.len()) }
    }

    /// Borrow the elements as a mutable slice.
    ///
    /// `&mut self` statically proves exclusive access through this
    /// reference, so no aliasing contract is needed beyond the usual
    /// borrow rules.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        let len = self.cells.len();
        // SAFETY: `&mut self` guarantees no other reference derived from
        // this `CpuBuffer` borrow is live; pointer/length come from the
        // live Vec and all elements are initialized.
        unsafe { std::slice::from_raw_parts_mut(self.base_ptr(), len) }
    }

    /// Copy `src` into elements `[offset, offset + src.len())`.
    ///
    /// # Safety
    ///
    /// Caller must uphold the type-level synchronization contract: no
    /// concurrent access from other threads, and no outstanding `&[T]`
    /// borrow overlapping the written range may be used after this call.
    /// Caller must also ensure `offset + src.len() <= self.len()` (checked
    /// by the `TensorStorage::cpu_write_at` wrapper).
    pub(crate) unsafe fn copy_from_slice_at(&self, offset: usize, src: &[T])
    where
        T: Copy,
    {
        // SAFETY: per the function contract the range is in bounds and no
        // conflicting access overlaps the write. `src` is a live shared
        // slice and cannot overlap the destination cells: the destination
        // is reached through `UnsafeCell`-typed memory owned by this
        // buffer, while `src` is a plain `&[T]` — if it pointed into this
        // buffer the caller would already be violating the borrow contract
        // (callers pass freshly computed host vectors).
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.base_ptr().add(offset), src.len());
        }
    }
}

impl<T: Element> Default for CpuBuffer<T> {
    fn default() -> Self {
        Self { cells: Vec::new() }
    }
}

impl<T: Element> std::fmt::Debug for CpuBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CpuBuffer({} elements)", self.cells.len())
    }
}

// SAFETY: `CpuBuffer` is a plain heap buffer of `T: Element` values
// (`Element: Send + Sync`). `UnsafeCell` removes auto-`Sync` because it
// permits shared mutation; we restore it under the documented
// synchronization contract above — the same informal contract PyTorch's
// storages have. Unsynchronized cross-thread mutation is a documented
// caller violation, not a library guarantee.
unsafe impl<T: Element> Sync for CpuBuffer<T> {}

// ---------------------------------------------------------------------------
// CubeStorageHandle — trait-erased CubeCL device handle
// ---------------------------------------------------------------------------

/// Trait-erased handle to a CubeCL device-resident buffer.
///
/// `ferrotorch-cubecl` provides the concrete implementation; `ferrotorch-core`
/// defines only this interface so there is no circular dependency. The concrete
/// type wraps a `cubecl::server::Handle` plus an `Arc<CubeRuntime>` so the
/// runtime remains alive as long as any handle exists.
///
/// This mirrors the `GpuBufferHandle` / `GpuBackend` pattern used for CUDA.
/// Issue #673.
pub trait CubeStorageHandle: std::fmt::Debug + Send + Sync {
    /// Upcast to `&dyn Any` for concrete-type downcasting.
    ///
    /// Implementors must return `self` via `self as &dyn std::any::Any`.
    /// This mirrors the `GpuBackend::as_any` pattern.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Number of `f32` elements in the buffer.
    fn len(&self) -> usize;

    /// Whether the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Device ordinal this buffer resides on.
    fn ordinal(&self) -> usize;

    /// Read the buffer contents back to the host as `Vec<f32>`.
    ///
    /// Performs a device-to-host transfer (D2H). Call sparingly — this is the
    /// explicit readback that the PyTorch-parity design requires users to opt
    /// into via `.cpu()`.
    fn read_to_host(&self) -> crate::error::FerrotorchResult<Vec<f32>>;

    /// Clone the handle (allocate a new device buffer with the same contents).
    fn clone_handle(&self) -> Box<dyn CubeStorageHandle>;
}

// ---------------------------------------------------------------------------
// TensorStorage / StorageBuffer
// ---------------------------------------------------------------------------

/// The underlying data buffer for a tensor, tagged with its device.
///
/// Owns the data directly ([`CpuBuffer`] for CPU, `GpuBufferHandle` for
/// CUDA, `Box<dyn CubeStorageHandle>` for CubeCL/XPU). GPU handles are
/// type-erased; the backend crates provide concrete implementations.
///
/// # Interior mutability + synchronization contract (CORE-001 / #1695)
///
/// `TensorStorage` is shared between `Tensor` clones and views through
/// `Arc` (PyTorch parity: `c10/core/StorageImpl.h:38` — aliasing storages
/// are the design). In-place ops mutate through aliased handles, so the
/// buffer sits behind an [`UnsafeCell`] and CPU elements additionally sit
/// in element-level cells ([`CpuBuffer`]). All mutation goes through raw
/// pointers derived from those cells — never through a `&mut` manufactured
/// behind an aliased `Arc`.
///
/// Callers must uphold the PyTorch-equivalent contract documented on
/// [`CpuBuffer`]: single-thread mutation XOR external synchronization, and
/// no shared borrow (`try_as_slice` / `gpu_handle` / `Tensor::data()`
/// results) may be *used* across a mutation performed through another
/// alias. The Miri gate at
/// `ferrotorch-core/tests/audit_core001_aliasing_miri.rs` pins the
/// supported patterns.
#[derive(Debug)]
pub struct TensorStorage<T: Element> {
    pub(crate) data: UnsafeCell<StorageBuffer<T>>,
    pub(crate) device: Device,
}

// SAFETY: `StorageBuffer<T: Element>` is `Send + Sync` by composition
// (`Element: Send + Sync`, GPU/CubeCL handles are `Send + Sync` by trait
// bound). The `UnsafeCell` removes auto-`Sync`; we restore it under the
// documented synchronization contract above (single-thread mutation XOR
// external synchronization — the informal contract PyTorch's storages
// carry). Unsynchronized cross-thread mutation is a documented caller
// violation.
unsafe impl<T: Element> Sync for TensorStorage<T> {}

/// Device-specific data buffer.
pub enum StorageBuffer<T: Element> {
    /// CPU heap-allocated data with element-level interior mutability
    /// (see [`CpuBuffer`]).
    Cpu(CpuBuffer<T>),
    /// CUDA device memory, accessed via the registered `GpuBackend`.
    Gpu(GpuBufferHandle),
    /// CubeCL device-resident buffer (XPU / portable GPU via wgpu/CUDA/ROCm).
    ///
    /// The concrete handle type is provided by `ferrotorch-cubecl`; core sees
    /// only the `CubeStorageHandle` trait object. Issue #673.
    Cubecl(Box<dyn CubeStorageHandle>),
    /// Meta storage — no backing memory, only the element count is
    /// recorded. Tensors built on this variant carry shape and dtype
    /// info but cannot be read or written. Used for shape inference
    /// and dry-run model construction. CL-395.
    ///
    /// An optional `fill_value` records the (single, scalar) value a
    /// `full_meta`-style constructor would have materialised if the
    /// tensor were on a real device. The data is still not stored —
    /// callers cannot read individual elements — but the metadata lets
    /// `full_meta(shape, value)` round-trip the fill through e.g.
    /// `meta_fill_value()` and lets shape-inference code that cares
    /// distinguish "uninitialised meta" from "would-be filled meta".
    Meta { numel: usize, fill_value: Option<T> },
}

impl<T: Element> TensorStorage<T> {
    /// Shared view of the buffer behind the cell.
    ///
    /// The returned reference is valid for reads under the synchronization
    /// contract documented on the type: it must not be *used* across a
    /// buffer replacement performed through another alias (element-level
    /// CPU writes do not invalidate it — they go through the element
    /// cells, which this borrow does not freeze).
    fn buffer(&self) -> &StorageBuffer<T> {
        // SAFETY: the cell always holds an initialized `StorageBuffer`.
        // Creating a shared reference is sound while no conflicting write
        // overlaps its use — the documented contract above. No `&mut` to
        // the buffer is ever created while aliased (all aliased mutation
        // is raw-pointer based), so this shared borrow cannot conflict
        // with a `&mut` within this crate's own access discipline.
        unsafe { &*self.data.get() }
    }

    /// Exclusive view of the buffer; `&mut self` proves uniqueness
    /// statically, so no contract is needed.
    fn buffer_mut(&mut self) -> &mut StorageBuffer<T> {
        self.data.get_mut()
    }

    /// Create a new CPU storage from a `Vec<T>`.
    pub fn cpu(data: Vec<T>) -> Self {
        Self {
            data: UnsafeCell::new(StorageBuffer::Cpu(CpuBuffer::from_vec(data))),
            device: Device::Cpu,
        }
    }

    /// Create a meta storage with the given element count. No memory is
    /// allocated for the elements; only the size is recorded. Reading
    /// the data of a meta tensor returns an error.
    pub fn meta(numel: usize) -> Self {
        Self {
            data: UnsafeCell::new(StorageBuffer::Meta {
                numel,
                fill_value: None,
            }),
            device: Device::Meta,
        }
    }

    /// Create a meta storage with the given element count and a recorded
    /// fill value. The fill is metadata only — no backing memory is
    /// allocated and individual elements cannot be read — but
    /// [`Self::meta_fill_value`] will return `Some(value)` so callers
    /// (e.g. `full_meta(shape, value)`) can round-trip the requested
    /// fill.
    pub fn meta_filled(numel: usize, value: T) -> Self {
        Self {
            data: UnsafeCell::new(StorageBuffer::Meta {
                numel,
                fill_value: Some(value),
            }),
            device: Device::Meta,
        }
    }

    /// Recorded fill value for a meta tensor, if one was supplied at
    /// construction time. Returns `None` for non-meta storage and for
    /// meta storage created without a fill (i.e. via [`Self::meta`]).
    pub fn meta_fill_value(&self) -> Option<&T> {
        match self.buffer() {
            StorageBuffer::Meta { fill_value, .. } => fill_value.as_ref(),
            _ => None,
        }
    }

    /// Create storage on `target_device` from CPU data.
    ///
    /// If `target_device` is CPU, wraps the `Vec` directly (zero-copy).
    /// If `target_device` is CUDA, uploads the data and returns GPU storage.
    ///
    /// Use this instead of `TensorStorage::cpu(data).to(device)` to avoid
    /// injecting a `ToDeviceBackward` node into the autograd graph.
    ///
    /// Note: `Device::Xpu` is not supported here because an H2D upload for XPU
    /// requires a `CubeRuntime`, which core does not own. Use
    /// `Tensor::to(Device::Xpu(n))` instead, which routes through
    /// `ferrotorch-xpu`'s `XpuDevice`.
    pub fn on_device(data: Vec<T>, target_device: Device) -> crate::error::FerrotorchResult<Self> {
        match target_device {
            Device::Cpu => Ok(Self::cpu(data)),
            Device::Cuda(ordinal) => {
                let backend = crate::gpu_dispatch::gpu_backend()
                    .ok_or(crate::error::FerrotorchError::DeviceUnavailable)?;
                let byte_len = checked_byte_count(
                    data.len(),
                    std::mem::size_of::<T>(),
                    "TensorStorage::on_device",
                )?;
                let bytes: &[u8] = unsafe {
                    // SAFETY: `data` is a valid, aligned `Vec<T>` on the heap.
                    // Reinterpreting as `&[u8]` is safe because we only use
                    // the bytes to copy to the GPU; the vec is not dropped
                    // until after `cpu_to_gpu` returns. `byte_len` is checked
                    // before constructing this raw byte view.
                    std::slice::from_raw_parts(
                        data.as_ptr().cast::<u8>(),
                        byte_len,
                    )
                };
                let handle = backend.cpu_to_gpu(bytes, T::dtype(), ordinal)?;
                Ok(Self::gpu(handle))
            }
            Device::Xpu(_) => Err(crate::error::FerrotorchError::InvalidArgument {
                message: "XPU storage requires a CubeRuntime; use Tensor::to(Device::Xpu(n)) \
                          via ferrotorch-xpu instead of TensorStorage::on_device. Issue #673."
                    .into(),
            }),
            Device::Mps(_) => Err(crate::error::FerrotorchError::InvalidArgument {
                message: "MPS storage requires the ferrotorch-mps backend; not yet wired into TensorStorage".into(),
            }),
            Device::Meta => {
                // Discard the data; only the element count matters.
                Ok(Self::meta(data.len()))
            }
        }
    }

    /// Create storage on `target_device` from CPU data, using pinned host
    /// memory for the CPU→CUDA transfer (~2x faster for large tensors).
    ///
    /// Falls back to regular transfer if no GPU backend or if target is CPU.
    pub fn on_device_pinned(
        data: Vec<T>,
        target_device: Device,
    ) -> crate::error::FerrotorchResult<Self> {
        match target_device {
            Device::Cpu => Ok(Self::cpu(data)),
            Device::Cuda(ordinal) => {
                let backend = crate::gpu_dispatch::gpu_backend()
                    .ok_or(crate::error::FerrotorchError::DeviceUnavailable)?;
                let byte_len = checked_byte_count(
                    data.len(),
                    std::mem::size_of::<T>(),
                    "TensorStorage::on_device_pinned",
                )?;
                let bytes: &[u8] = unsafe {
                    // SAFETY: same invariant as in `on_device`; `byte_len`
                    // is checked before constructing this raw byte view.
                    std::slice::from_raw_parts(
                        data.as_ptr().cast::<u8>(),
                        byte_len,
                    )
                };
                let handle = backend.cpu_to_gpu_pinned(bytes, T::dtype(), ordinal)?;
                Ok(Self::gpu(handle))
            }
            Device::Xpu(_) => Err(crate::error::FerrotorchError::InvalidArgument {
                message: "XPU storage requires a CubeRuntime; use Tensor::to(Device::Xpu(n)) \
                          via ferrotorch-xpu instead of TensorStorage::on_device_pinned. Issue #673."
                    .into(),
            }),
            Device::Mps(_) => Err(crate::error::FerrotorchError::InvalidArgument {
                message: "MPS storage requires the ferrotorch-mps backend; not yet wired into TensorStorage".into(),
            }),
            Device::Meta => Ok(Self::meta(data.len())),
        }
    }

    /// Create XPU (CubeCL device-resident) storage from a trait-erased handle.
    ///
    /// The handle wraps a `cubecl::server::Handle` and holds an `Arc<CubeRuntime>`
    /// so the device stays alive. This is the correct post-#673 constructor:
    /// XPU storage is truly device-resident, not a CPU `Vec<T>`.
    ///
    /// Called by `ferrotorch-xpu` (and `ferrotorch-cubecl`) after uploading data
    /// to the device.
    pub fn xpu_from_handle(handle: Box<dyn CubeStorageHandle>, ordinal: usize) -> Self {
        Self {
            data: UnsafeCell::new(StorageBuffer::Cubecl(handle)),
            device: Device::Xpu(ordinal),
        }
    }

    /// Create a new CUDA storage from a handle.
    pub fn gpu(handle: GpuBufferHandle) -> Self {
        let device = Device::Cuda(handle.device_ordinal());
        Self {
            data: UnsafeCell::new(StorageBuffer::Gpu(handle)),
            device,
        }
    }

    /// The device this storage resides on.
    #[inline]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Total number of elements in the buffer.
    pub fn len(&self) -> usize {
        match self.buffer() {
            StorageBuffer::Cpu(v) => v.len(),
            StorageBuffer::Gpu(h) => h.len(),
            StorageBuffer::Cubecl(h) => h.len(),
            StorageBuffer::Meta { numel, .. } => *numel,
        }
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the data as a slice. Only available for CPU storage.
    ///
    /// # Panics
    /// Panics if the tensor is on a GPU or XPU device. Call `.cpu()` first.
    /// Panics if the tensor is a meta tensor.
    #[deprecated(
        since = "0.4.5",
        note = "use try_as_slice() instead; this version panics on non-CPU storage"
    )]
    pub fn as_slice(&self) -> &[T] {
        match self.buffer() {
            StorageBuffer::Cpu(v) => v.as_slice(),
            StorageBuffer::Gpu(_) => {
                panic!("cannot access GPU tensor as CPU slice -- call .cpu() first")
            }
            StorageBuffer::Cubecl(_) => {
                panic!("cannot access XPU tensor as CPU slice -- call .cpu() first")
            }
            StorageBuffer::Meta { .. } => {
                panic!("cannot access meta tensor as a slice -- meta tensors carry no data")
            }
        }
    }

    /// Borrow the data as a mutable slice. Only available for CPU storage.
    ///
    /// # Panics
    /// Panics if the tensor is on a GPU or XPU device. Call `.cpu()` first.
    /// Panics if the tensor is a meta tensor.
    #[deprecated(
        since = "0.4.5",
        note = "use try_as_mut_slice() instead; this version panics on non-CPU storage"
    )]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self.buffer_mut() {
            StorageBuffer::Cpu(v) => v.as_mut_slice(),
            StorageBuffer::Gpu(_) => {
                panic!("cannot mutate GPU tensor as CPU slice -- call .cpu() first")
            }
            StorageBuffer::Cubecl(_) => {
                panic!("cannot mutate XPU tensor as CPU slice -- call .cpu() first")
            }
            StorageBuffer::Meta { .. } => {
                panic!("cannot mutate meta tensor as a slice -- meta tensors carry no data")
            }
        }
    }

    /// Fallible borrow of the data as a slice — same as `as_slice` but returns
    /// `Err(GpuTensorNotAccessible)` instead of panicking when the tensor is
    /// not CPU-resident (GPU, XPU, or meta).
    ///
    /// # Errors
    /// Returns [`FerrotorchError::GpuTensorNotAccessible`] when the storage is
    /// on a GPU or XPU device, or is a meta tensor.
    pub fn try_as_slice(&self) -> crate::error::FerrotorchResult<&[T]> {
        match self.buffer() {
            StorageBuffer::Cpu(v) => Ok(v.as_slice()),
            StorageBuffer::Gpu(_) | StorageBuffer::Cubecl(_) | StorageBuffer::Meta { .. } => {
                Err(crate::error::FerrotorchError::GpuTensorNotAccessible)
            }
        }
    }

    /// Fallible mutable borrow of the data as a slice — same as `as_mut_slice`
    /// but returns `Err(GpuTensorNotAccessible)` instead of panicking when the
    /// tensor is not CPU-resident (GPU, XPU, or meta).
    ///
    /// # Errors
    /// Returns [`FerrotorchError::GpuTensorNotAccessible`] when the storage is
    /// on a GPU or XPU device, or is a meta tensor.
    pub fn try_as_mut_slice(&mut self) -> crate::error::FerrotorchResult<&mut [T]> {
        match self.buffer_mut() {
            StorageBuffer::Cpu(v) => Ok(v.as_mut_slice()),
            StorageBuffer::Gpu(_) | StorageBuffer::Cubecl(_) | StorageBuffer::Meta { .. } => {
                Err(crate::error::FerrotorchError::GpuTensorNotAccessible)
            }
        }
    }

    /// Returns `true` if this storage is on CPU.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        matches!(self.buffer(), StorageBuffer::Cpu(_))
    }

    /// Returns `true` if this storage is a CUDA device buffer.
    #[inline]
    pub fn is_gpu(&self) -> bool {
        matches!(self.buffer(), StorageBuffer::Gpu(_))
    }

    /// Returns `true` if this storage is a CubeCL device-resident buffer (XPU).
    #[inline]
    pub fn is_cubecl(&self) -> bool {
        matches!(self.buffer(), StorageBuffer::Cubecl(_))
    }

    /// Returns `true` if this storage is a meta (no-data) tensor.
    #[inline]
    pub fn is_meta(&self) -> bool {
        matches!(self.buffer(), StorageBuffer::Meta { .. })
    }

    /// Get the CUDA buffer handle. Returns `None` for CPU, XPU, and Meta storage.
    pub fn gpu_handle(&self) -> Option<&GpuBufferHandle> {
        match self.buffer() {
            StorageBuffer::Gpu(h) => Some(h),
            StorageBuffer::Cpu(_) | StorageBuffer::Cubecl(_) | StorageBuffer::Meta { .. } => None,
        }
    }

    /// Get a mutable CUDA buffer handle. Returns `None` for CPU, XPU, and Meta storage.
    ///
    /// `&mut self` statically proves exclusive access, so this accessor is
    /// safe with no extra contract.
    pub fn gpu_handle_mut(&mut self) -> Option<&mut GpuBufferHandle> {
        match self.buffer_mut() {
            StorageBuffer::Gpu(h) => Some(h),
            StorageBuffer::Cpu(_) | StorageBuffer::Cubecl(_) | StorageBuffer::Meta { .. } => None,
        }
    }

    /// Get a mutable CUDA buffer handle through a *shared* (possibly
    /// `Arc`-aliased) storage reference. Returns `None` for CPU, XPU, and
    /// Meta storage.
    ///
    /// This is the aliased counterpart of [`Self::gpu_handle_mut`] for the
    /// GPU-native optimizer fast path (`Tensor::with_gpu_handle_mut`), where
    /// the storage is reached through an `Arc` that other `Tensor` handles
    /// also hold. The `&mut` is derived through the [`UnsafeCell`], never
    /// from the `Arc` itself, so creating it does not invalidate outstanding
    /// `&TensorStorage` metadata borrows held by other handles.
    ///
    /// # Safety
    ///
    /// Caller must uphold the type-level synchronization contract: for the
    /// lifetime of the returned `&mut GpuBufferHandle` no other access to
    /// this storage's *buffer* may occur (no concurrent thread, no other
    /// outstanding buffer/handle borrow used during that lifetime).
    /// Metadata access (`device()`) through other aliases remains fine.
    #[allow(
        clippy::mut_from_ref,
        reason = "aliased-storage primitive: the &mut is derived through the \
                  UnsafeCell (interior mutability), not from &self; exclusivity \
                  is the unsafe fn's documented caller contract"
    )]
    pub(crate) unsafe fn gpu_handle_mut_aliased(&self) -> Option<&mut GpuBufferHandle> {
        // SAFETY: `self.data.get()` grants read-write provenance via the
        // `UnsafeCell`; the caller contract above guarantees the produced
        // `&mut StorageBuffer` (and the `&mut GpuBufferHandle` reborrowed
        // from it) is exclusive for its lifetime.
        match unsafe { &mut *self.data.get() } {
            StorageBuffer::Gpu(h) => Some(h),
            StorageBuffer::Cpu(_) | StorageBuffer::Cubecl(_) | StorageBuffer::Meta { .. } => None,
        }
    }

    /// Get the CubeCL storage handle. Returns `None` for non-Cubecl storage.
    pub fn cubecl_handle(&self) -> Option<&dyn CubeStorageHandle> {
        match self.buffer() {
            StorageBuffer::Cubecl(h) => Some(h.as_ref()),
            _ => None,
        }
    }

    /// Write `src` into the CPU buffer at element offset `offset`, through a
    /// *shared* (possibly `Arc`-aliased) storage reference.
    ///
    /// This is the aliased-write primitive behind `Tensor::update_data` and
    /// the trailing-underscore in-place ops: the write goes through the
    /// element-level [`UnsafeCell`]s ([`CpuBuffer::copy_from_slice_at`]), so
    /// it never manufactures a `&mut` behind the aliased `Arc` and never
    /// invalidates outstanding `&TensorStorage` metadata borrows.
    ///
    /// # Errors
    ///
    /// - [`crate::error::FerrotorchError::GpuTensorNotAccessible`] when the
    ///   storage is not CPU-resident (GPU, XPU, or meta).
    /// - [`crate::error::FerrotorchError::InvalidArgument`] when
    ///   `offset + src.len()` exceeds the buffer length.
    ///
    /// # Safety
    ///
    /// Caller must uphold the type-level synchronization contract:
    /// 1. no other thread accesses this storage concurrently with the write
    ///    (single-thread mutation XOR external synchronization), and
    /// 2. any outstanding `&[T]` borrow overlapping the written range
    ///    (`try_as_slice` / `Tensor::data()` results) is not *used* after
    ///    this call — re-borrow instead.
    pub(crate) unsafe fn cpu_write_at(
        &self,
        offset: usize,
        src: &[T],
    ) -> crate::error::FerrotorchResult<()>
    where
        T: Copy,
    {
        match self.buffer() {
            StorageBuffer::Cpu(v) => {
                let end = offset
                    .checked_add(src.len())
                    .filter(|&end| end <= v.len())
                    .ok_or_else(|| crate::error::FerrotorchError::InvalidArgument {
                        message: format!(
                            "cpu_write_at: write of {} elements at offset {} exceeds \
                             buffer length {}",
                            src.len(),
                            offset,
                            v.len(),
                        ),
                    })?;
                let _ = end;
                // SAFETY: bounds were checked above; the synchronization
                // contract is forwarded verbatim from this function's own
                // `# Safety` section to the caller.
                unsafe { v.copy_from_slice_at(offset, src) };
                Ok(())
            }
            StorageBuffer::Gpu(_) | StorageBuffer::Cubecl(_) | StorageBuffer::Meta { .. } => {
                Err(crate::error::FerrotorchError::GpuTensorNotAccessible)
            }
        }
    }

    /// Mutable borrow of the CPU elements through a *shared* (possibly
    /// `Arc`-aliased) storage reference.
    ///
    /// The `&mut [T]` is derived from the element-level [`UnsafeCell`]s, so
    /// creating it is sanctioned by the aliasing model even though `self` is
    /// shared. This is the primitive behind `Tensor::data_mut` (the
    /// optimizer-step pattern).
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::FerrotorchError::GpuTensorNotAccessible`] when
    /// the storage is not CPU-resident (GPU, XPU, or meta).
    ///
    /// # Safety
    ///
    /// Caller must guarantee genuinely exclusive access for the lifetime of
    /// the returned slice: no other thread touches this storage, and no
    /// other borrow of the buffer's elements (from this or any aliasing
    /// handle) is *used* while the `&mut [T]` is live.
    #[allow(
        clippy::mut_from_ref,
        reason = "aliased-storage primitive: the &mut [T] is derived through the \
                  element-level UnsafeCells, not from &self; exclusivity is the \
                  unsafe fn's documented caller contract"
    )]
    pub(crate) unsafe fn try_as_mut_slice_aliased(
        &self,
    ) -> crate::error::FerrotorchResult<&mut [T]> {
        match self.buffer() {
            StorageBuffer::Cpu(v) => {
                let len = v.len();
                // SAFETY: pointer and length come from the live element Vec;
                // all elements are initialized; the caller contract above
                // guarantees exclusivity for the returned slice's lifetime.
                Ok(unsafe { std::slice::from_raw_parts_mut(v.base_ptr(), len) })
            }
            StorageBuffer::Gpu(_) | StorageBuffer::Cubecl(_) | StorageBuffer::Meta { .. } => {
                Err(crate::error::FerrotorchError::GpuTensorNotAccessible)
            }
        }
    }

    /// Replace this storage's buffer with `new_storage`'s buffer, through a
    /// *shared* (possibly `Arc`-aliased) storage reference. The old buffer
    /// is dropped (CPU buffers return to the pool).
    ///
    /// This is the aliased-swap primitive behind `Tensor::update_storage`
    /// (and, transitively, the GPU branch of `update_data`): the
    /// replacement goes through the buffer-level [`UnsafeCell`], so
    /// outstanding `&TensorStorage` metadata borrows held by aliasing
    /// handles survive the swap. Element borrows (`try_as_slice` results)
    /// into the *old* buffer dangle after the swap — using them is the
    /// documented residual-contract violation.
    ///
    /// **View rule (#1938):** swapping the buffer is only observationally
    /// correct when every tensor viewing this storage maps the WHOLE
    /// buffer (`storage_offset == 0`, `numel == len`, C-contiguous). A
    /// caller holding a sub-view must NOT swap — it would shrink/reorder
    /// the shared buffer under every other alias — and must write the
    /// view's region in place instead ([`Self::cpu_write_at`] /
    /// [`Self::try_as_mut_slice_aliased`] on CPU, `strided_scatter_*` on
    /// CUDA). `Tensor::update_storage` enforces this dispatch; new callers
    /// of this primitive must uphold it themselves.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::FerrotorchError::DeviceMismatch`] when
    /// `new_storage` resides on a different device — the `device` field is
    /// plain (not behind the cell) and immutable after construction, so a
    /// cross-device swap cannot be represented.
    ///
    /// # Safety
    ///
    /// Caller must uphold the type-level synchronization contract: no other
    /// thread accesses this storage concurrently, and no outstanding borrow
    /// of the buffer or its elements (from this or any aliasing handle) is
    /// *used* after this call.
    pub(crate) unsafe fn replace_buffer_aliased(
        &self,
        new_storage: TensorStorage<T>,
    ) -> crate::error::FerrotorchResult<()> {
        if new_storage.device != self.device {
            return Err(crate::error::FerrotorchError::DeviceMismatch {
                expected: self.device,
                got: new_storage.device,
            });
        }
        let new_buffer = new_storage.into_buffer();
        // SAFETY: `self.data.get()` grants read-write provenance via the
        // `UnsafeCell`; the caller contract guarantees no conflicting access
        // overlaps the replacement. `ptr::replace` returns the OLD buffer so
        // its destructor runs (no leak of GPU handles).
        let old_buffer = unsafe { std::ptr::replace(self.data.get(), new_buffer) };
        // Re-wrap the old buffer in a `TensorStorage` so its `Drop` impl
        // runs (returning CPU buffers to the pool) — dropping a bare
        // `StorageBuffer` would bypass the pool.
        drop(Self {
            data: UnsafeCell::new(old_buffer),
            device: self.device,
        });
        Ok(())
    }

    /// Consume `self` and return the buffer, bypassing the pool-returning
    /// `Drop` impl (ownership of the buffer transfers to the caller).
    fn into_buffer(self) -> StorageBuffer<T> {
        let this = std::mem::ManuallyDrop::new(self);
        // SAFETY: `this` is `ManuallyDrop`, so `TensorStorage::drop` never
        // runs and the buffer is not read again through `this`; reading it
        // out by value is a plain ownership transfer of an initialized
        // `StorageBuffer`.
        unsafe { std::ptr::read(this.data.get()) }
    }

    /// Fallible clone — same as `Clone::clone` but returns `Result` instead
    /// of panicking when a backend call fails.
    ///
    /// Reads the source buffer, so it falls under the type-level
    /// synchronization contract: cloning concurrently with an unsynchronized
    /// write from another thread is a caller violation.
    pub fn try_clone(&self) -> crate::error::FerrotorchResult<Self> {
        match self.buffer() {
            StorageBuffer::Cpu(v) => Ok(Self {
                data: UnsafeCell::new(StorageBuffer::Cpu(CpuBuffer::from_vec(
                    v.as_slice().to_vec(),
                ))),
                device: self.device,
            }),
            StorageBuffer::Gpu(h) => {
                let backend = crate::gpu_dispatch::gpu_backend()
                    .ok_or(crate::error::FerrotorchError::DeviceUnavailable)?;
                let cloned = backend.clone_buffer(h)?;
                Ok(Self {
                    data: UnsafeCell::new(StorageBuffer::Gpu(cloned)),
                    device: self.device,
                })
            }
            StorageBuffer::Cubecl(h) => {
                let cloned = h.clone_handle();
                Ok(Self {
                    data: UnsafeCell::new(StorageBuffer::Cubecl(cloned)),
                    device: self.device,
                })
            }
            StorageBuffer::Meta { numel, fill_value } => Ok(Self {
                data: UnsafeCell::new(StorageBuffer::Meta {
                    numel: *numel,
                    fill_value: fill_value.clone(),
                }),
                device: self.device,
            }),
        }
    }

    /// Clone a contiguous sub-region `[offset..offset+numel]` of this storage.
    ///
    /// For CPU, slices the `Vec` directly. For CUDA/XPU, round-trips through the
    /// host to extract the sub-region. Returns an error instead of panicking
    /// on backend failures.
    pub fn try_clone_subregion(
        &self,
        offset: usize,
        numel: usize,
    ) -> crate::error::FerrotorchResult<Self> {
        if offset == 0 && numel == self.len() {
            return self.try_clone();
        }
        match self.buffer() {
            StorageBuffer::Cpu(v) => {
                let end = offset
                    .checked_add(numel)
                    .filter(|&end| end <= v.len())
                    .ok_or_else(|| crate::error::FerrotorchError::InvalidArgument {
                        message: format!(
                            "try_clone_subregion: range {offset}..{} exceeds buffer \
                             length {}",
                            offset.saturating_add(numel),
                            v.len(),
                        ),
                    })?;
                let slice = &v.as_slice()[offset..end];
                Ok(Self::cpu(slice.to_vec()))
            }
            StorageBuffer::Gpu(h) => {
                let backend = crate::gpu_dispatch::gpu_backend()
                    .ok_or(crate::error::FerrotorchError::DeviceUnavailable)?;
                let bytes = backend.gpu_to_cpu(h)?;
                let elem_size = std::mem::size_of::<T>();
                let elem_end = offset
                    .checked_add(numel)
                    .filter(|&end| end <= h.len())
                    .ok_or_else(|| crate::error::FerrotorchError::InvalidArgument {
                        message: format!(
                            "try_clone_subregion: range {offset}..{} exceeds buffer \
                             length {}",
                            offset.saturating_add(numel),
                            h.len(),
                        ),
                    })?;
                let start = offset.checked_mul(elem_size).ok_or_else(|| {
                    crate::error::FerrotorchError::InvalidArgument {
                        message: format!(
                            "try_clone_subregion: byte offset {offset} * {elem_size} overflows"
                        ),
                    }
                })?;
                let end = elem_end.checked_mul(elem_size).ok_or_else(|| {
                    crate::error::FerrotorchError::InvalidArgument {
                        message: format!(
                            "try_clone_subregion: byte end {elem_end} * {elem_size} overflows"
                        ),
                    }
                })?;
                // Re-upload the sliced bytes under the *source handle's* dtype
                // tag so the subregion preserves the original ScalarType.
                let handle =
                    backend.cpu_to_gpu(&bytes[start..end], h.dtype(), h.device_ordinal())?;
                Ok(Self {
                    data: UnsafeCell::new(StorageBuffer::Gpu(handle)),
                    device: self.device,
                })
            }
            StorageBuffer::Cubecl(h) => {
                // D2H readback, slice, then re-upload via a new handle.
                // The new handle reuses the same runtime (held by the original
                // handle's Arc<CubeRuntime>).
                let all = h.read_to_host()?;
                let end = offset
                    .checked_add(numel)
                    .filter(|&end| end <= all.len())
                    .ok_or_else(|| crate::error::FerrotorchError::InvalidArgument {
                        message: format!(
                            "try_clone_subregion: range {offset}..{} exceeds CubeCL buffer \
                             length {}",
                            offset.saturating_add(numel),
                            all.len(),
                        ),
                    })?;
                let slice = all[offset..end].to_vec();
                // Re-upload: the concrete impl's `clone_handle` clones the full
                // buffer; for sub-regions we go through host for now (correct,
                // can be optimised later with a device-side copy).
                // We need a new handle wrapping just `slice` — but
                // `CubeStorageHandle` doesn't expose an upload method (that
                // lives in ferrotorch-cubecl). Return an error directing the
                // caller to use `.cpu()` for sub-region reads instead.
                //
                // This path is only hit for non-contiguous XPU tensors, which
                // are rare in practice. If this becomes a bottleneck, add an
                // `upload_slice` method to `CubeStorageHandle`. Issue #673.
                let _ = slice;
                Err(crate::error::FerrotorchError::InvalidArgument {
                    message: format!(
                        "try_clone_subregion on XPU storage is not yet supported \
                         (offset={offset}, numel={numel}); call .cpu() first. Issue #673."
                    ),
                })
            }
            StorageBuffer::Meta { .. } => Ok(Self::meta(numel)),
        }
    }
}

impl<T: Element> Clone for TensorStorage<T> {
    /// Clone the storage. Delegates to [`Self::try_clone`] so the GPU/CubeCL
    /// branches share one fallible-clone implementation.
    ///
    /// # Panics
    /// Panics with a structured message naming the underlying [`crate::error::FerrotorchError`]
    /// (most commonly [`crate::error::FerrotorchError::DeviceUnavailable`] when no GPU backend
    /// is registered, or a backend `clone_buffer` failure). Use
    /// [`Self::try_clone`] when you need to handle the failure explicitly
    /// instead of panicking.
    fn clone(&self) -> Self {
        match self.try_clone() {
            Ok(cloned) => cloned,
            Err(e) => panic!(
                "TensorStorage::clone failed: {e}. \
                 Use TensorStorage::try_clone() to handle this case explicitly."
            ),
        }
    }
}

impl<T: Element> Drop for TensorStorage<T> {
    fn drop(&mut self) {
        // Return CPU buffers to the pool for reuse. `get_mut` is the safe
        // exclusive accessor — `&mut self` in `drop` proves uniqueness.
        if let StorageBuffer::Cpu(v) = self.data.get_mut()
            && !v.is_empty()
        {
            // Take the buffer out, replacing with an empty one (no alloc),
            // and unwrap the `UnsafeCell` element wrappers (zero-copy).
            let buf = std::mem::take(v);
            crate::cpu_pool::pool_return_cpu(buf.into_vec());
        }
        // GPU/CubeCL buffers are dropped normally (runtime handles cleanup).
    }
}

impl<T: Element> std::fmt::Debug for StorageBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageBuffer::Cpu(v) => write!(f, "Cpu({} elements)", v.len()),
            StorageBuffer::Gpu(h) => write!(f, "Gpu({h:?})"),
            StorageBuffer::Cubecl(h) => {
                write!(f, "Cubecl(ordinal={}, len={})", h.ordinal(), h.len())
            }
            StorageBuffer::Meta { numel, .. } => write!(f, "Meta({numel} elements)"),
        }
    }
}
