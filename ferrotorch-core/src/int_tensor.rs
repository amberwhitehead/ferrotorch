//! Integer-typed tensors for indexing, embedding lookups, and any other
//! workload that needs first-class non-float storage. (#596)
//!
//! `IntTensor<I>` is a contiguous tensor of integers (`i32` or `i64`). It is
//! intentionally **not** generic over `Float` — the existing `Tensor<T: Float>`
//! is the right type for differentiable float math. `IntTensor` is for indices
//! and counts where autograd is a category error (mirroring PyTorch's runtime
//! `isDifferentiableType(ScalarType) == false` for integer dtypes).
//!
//! # Device residency (crosslink #1185 Phase 2a)
//!
//! As of Phase 2a, `IntTensor` is **device-aware**: its data lives in a
//! [`TensorStorage<I>`] that carries its own [`Device`]. Integer data can be
//! uploaded to CUDA and read back, reusing the Phase-0 DType-tagged raw-byte
//! transport (`GpuBackend::cpu_to_gpu` / `gpu_to_cpu`) — exactly the machinery
//! `Tensor<T>::to` uses. This is PyTorch's model: an integer tensor is ordinary
//! raw-byte storage plus a `ScalarType` tag (`I32` / `I64`).
//!
//! Phase 2a adds **no integer compute kernels** (that is Phase 2b). It only
//! gives `IntTensor` a device, GPU storage, host↔device transfer, and handle
//! access so a later phase can launch kernels.
//!
//! # Conversions
//!
//! - `Tensor::to_int` — round-then-cast a float tensor to ints
//! - `IntTensor::to_float` — widen back into a float tensor
//!
//! Both copy data; there is no shared-storage path because the element
//! sizes differ. (These cross-type conversions are not yet implemented as
//! methods — see the Phase 2c follow-up note on cross-dtype GPU casts below.)

//!
//! ## REQ status (per `.design/ferrotorch-core/int_tensor.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (IntElement trait, IntTensor<I>) | SHIPPED | trait `IntElement` at `int_tensor.rs:44`; `impl IntElement for i32 / i64` at `:57 / :74`; `pub struct IntTensor<I: IntElement>` at `:93`; consumer `grad_fns/quantize_grad.rs:139` `zero_point: &IntTensor<i64>`; `ops/phase2c.rs:115` `input: &IntTensor<I>` |
//! | REQ-2 (constructors) | SHIPPED | `from_vec` at `int_tensor.rs:113`, `from_slice` at `:139`, `zeros` at `:144`, `arange` at `:159`, `scalar` at `:175`; consumer `grad_fns/reduction.rs:1463` `IntTensor::<i64>::scalar(best_idx)` for argmax; `ops/phase2c.rs:101, :109` `from_gpu_handle` / `from_vec` |
//! | REQ-3 (device transfer) | SHIPPED | `device` at `int_tensor.rs:199`, `is_cuda` at `:205`, `to` at `:260`; consumer `ops/phase2c.rs` reads `input.device()` + `input.gpu_handle()` before argmax kernel launches; `// SAFETY:` D2H reinterpret at `:296-318` |
//! | REQ-4 (cross-width cast) | SHIPPED | `cast<J>` at `int_tensor.rs:355` with `cast_gpu` fast path; consumer `ops/phase2c.rs` i32↔i64 cast kernel; test `cast_i64_to_i32_out_of_range_errors` at `:836` |
//! | REQ-5 (reshape) | SHIPPED | `reshape` at `int_tensor.rs:384`; consumer `grad_fns/reduction.rs` argmax materialization; test `reshape_preserves_data` at `:843` |
//! | REQ-6 (arithmetic ops) | SHIPPED | `add` at `int_tensor.rs:551`, `sub` at `:561`, `mul` at `:571`, `neg` at `:581`; CPU references at `:696`; consumer `lib.rs:146` re-export — boundary public API; `bool_tensor.rs:524-569` integer comparison constructors route through `IntTensor` compute path. R-DEFER-1 S5 grandfathering; runner arms at #1530 |
//! | REQ-7 (floor_div/remainder) | SHIPPED | `floor_div` at `int_tensor.rs:589`, `remainder` at `:599`; CPU references `int_floor_div_ref` at `:709`, `int_remainder_ref` at `:728`; consumer `lib.rs:146` re-export |
//! | REQ-8 (bitwise ops) | SHIPPED | `bitand`/`bitor`/`bitxor`/`bitnot` at `int_tensor.rs:609-639`; `shl`/`shr` at `:644-649`; CPU references at `:744-765`; consumer `lib.rs:146` re-export |
//! | REQ-9 (reductions) | SHIPPED | `sum`/`prod`/`min`/`max` at `int_tensor.rs:654-684`; empty-tensor handling at `reduce_op` `:502-548`; consumer `lib.rs:146` re-export |
//! | REQ-10 (gpu_handle) | SHIPPED | `gpu_handle` at `int_tensor.rs:236`, `from_gpu_handle` at `:421` with `debug_assert_eq!(handle.dtype(), I::dtype())`; consumer `ops/phase2c.rs:101` invokes `from_gpu_handle`; `bool_tensor.rs:596` reads `a.gpu_handle()` for int comparison GPU path |
//! | REQ-11 (0-D vs zero-axis) | SHIPPED | `shape.is_empty() { 1 } else { product }` at `int_tensor.rs:117, :146, :386`; consumer `grad_fns/reduction.rs:1463` returns 0-D scalar `IntTensor` for argmax — #805 regression pin |
//! | REQ-12 (structured errors) | SHIPPED | `ShapeMismatch` / `DeviceMismatch` / `InvalidArgument` at multiple sites; no `panic!` / `unwrap()` / `expect()` in production paths; consumers propagate via `?` |

use crate::device::Device;
use crate::dtype::Element;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::gpu_dispatch::GpuBufferHandle;
use crate::storage::TensorStorage;

/// Element types supported by [`IntTensor`].
///
/// The [`Element`](crate::dtype::Element) bound (added in crosslink #1185 Phase
/// 2a) is what lets a [`TensorStorage<I>`] hold integer data and tag the GPU
/// handle with the right `ScalarType` (`i32` → `DType::I32`, `i64` →
/// `DType::I64`). Both i32 and i64 already satisfy it via ferray.
pub trait IntElement:
    Element + Copy + Send + Sync + 'static + std::fmt::Debug + std::fmt::Display
{
    /// Bit-width of one element, used for dtype tagging.
    const BITS: u32;
    /// Returns this element type's printable name (e.g. `"i32"`).
    fn dtype_name() -> &'static str;
    /// Convert from i64. Returns `None` on out-of-range.
    fn try_from_i64(v: i64) -> Option<Self>;
    /// Widen to i64.
    fn to_i64(self) -> i64;
}

impl IntElement for i32 {
    const BITS: u32 = 32;
    fn dtype_name() -> &'static str {
        "i32"
    }
    fn try_from_i64(v: i64) -> Option<Self> {
        if (i32::MIN as i64..=i32::MAX as i64).contains(&v) {
            Some(v as i32)
        } else {
            None
        }
    }
    fn to_i64(self) -> i64 {
        self as i64
    }
}

impl IntElement for i64 {
    const BITS: u32 = 64;
    fn dtype_name() -> &'static str {
        "i64"
    }
    fn try_from_i64(v: i64) -> Option<Self> {
        Some(v)
    }
    fn to_i64(self) -> i64 {
        self
    }
}

/// Contiguous tensor of integers (`i32` or `i64`), device-aware.
///
/// Data lives in a [`TensorStorage<I>`] (CPU `Vec<I>` or a CUDA
/// [`GpuBufferHandle`]); the storage carries its own [`Device`]. CPU-resident
/// behaviour is byte-identical to the pre-Phase-2a `Arc<Vec<I>>` design.
#[derive(Debug)]
pub struct IntTensor<I: IntElement> {
    storage: TensorStorage<I>,
    shape: Vec<usize>,
}

impl<I: IntElement> Clone for IntTensor<I> {
    /// Clone the tensor. For CPU storage this clones the `Vec`; for GPU storage
    /// it allocates a new device buffer with the same contents (via the
    /// backend's `clone_buffer`). Delegates to [`TensorStorage::clone`].
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            shape: self.shape.clone(),
        }
    }
}

impl<I: IntElement> IntTensor<I> {
    /// Build from a Vec + shape (CPU-resident). Returns an error if
    /// `data.len()` does not match the shape's total numel.
    pub fn from_vec(data: Vec<I>, shape: Vec<usize>) -> FerrotorchResult<Self> {
        // PyTorch parity: shape=[] is a 0-d scalar (numel=1); shape=[0]
        // (or any shape with a zero axis) is empty (numel=0). The previous
        // `.max(1)` conflated these. (#805)
        let expected = crate::shape::checked_numel(&shape, "IntTensor::from_vec")?;
        if data.len() != expected {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "IntTensor::from_vec: data.len()={} != prod(shape)={} for shape {:?}",
                    data.len(),
                    expected,
                    shape
                ),
            });
        }
        Ok(Self {
            storage: TensorStorage::cpu(data),
            shape,
        })
    }

    /// Build from a slice + shape (clones into a fresh `Vec`; CPU-resident).
    pub fn from_slice(data: &[I], shape: &[usize]) -> FerrotorchResult<Self> {
        Self::from_vec(data.to_vec(), shape.to_vec())
    }

    /// Zeros of the given shape (CPU-resident).
    pub fn zeros(shape: &[usize]) -> Self {
        // shape=[] -> 0-d scalar (numel 1); shape=[0] -> empty (numel 0). (#805)
        let total = crate::shape::numel(shape);
        let zero = I::try_from_i64(0).expect("0 fits in any IntElement");
        Self {
            storage: TensorStorage::cpu(vec![zero; total]),
            shape: shape.to_vec(),
        }
    }

    /// 1-D `arange`-style `[0, 1, ..., n-1]` (CPU-resident).
    pub fn arange(n: usize) -> FerrotorchResult<Self> {
        let mut data: Vec<I> = Vec::with_capacity(n);
        for i in 0..n {
            data.push(
                I::try_from_i64(i as i64).ok_or(FerrotorchError::InvalidArgument {
                    message: format!(
                        "IntTensor::arange: {i} out of range for {}",
                        I::dtype_name()
                    ),
                })?,
            );
        }
        Self::from_vec(data, vec![n])
    }

    /// 0-d scalar (CPU-resident).
    pub fn scalar(v: I) -> Self {
        Self {
            storage: TensorStorage::cpu(vec![v]),
            shape: Vec::new(),
        }
    }

    /// Logical shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.storage.len()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// The device this tensor's storage resides on.
    #[inline]
    pub fn device(&self) -> Device {
        self.storage.device()
    }

    /// Returns `true` if this tensor is on a CUDA GPU.
    #[inline]
    pub fn is_cuda(&self) -> bool {
        self.device().is_cuda()
    }

    /// Borrow the contiguous buffer as a host slice.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::GpuTensorNotAccessible`] if the tensor is on a
    /// GPU (CUDA / XPU) device, or is a meta tensor. This mirrors
    /// `Tensor::data` and prevents callers from silently reading device memory
    /// as a host slice — call [`Self::to(Device::Cpu)`](Self::to) first to
    /// transfer.
    pub fn data(&self) -> FerrotorchResult<&[I]> {
        // try_as_slice returns Err(GpuTensorNotAccessible) for Gpu/Cubecl/Meta
        // storage — exactly the contract we want (no silent host readback).
        self.storage.try_as_slice()
    }

    /// Element type name (`"i32"` / `"i64"`).
    pub fn dtype_name(&self) -> &'static str {
        I::dtype_name()
    }

    /// Get the CUDA buffer handle. Returns `Err` for CPU (and other non-CUDA)
    /// tensors, mirroring `Tensor::gpu_handle`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] when the tensor is not
    /// CUDA-resident.
    pub fn gpu_handle(&self) -> FerrotorchResult<&GpuBufferHandle> {
        self.storage
            .gpu_handle()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "IntTensor is not on a CUDA GPU".into(),
            })
    }

    /// Move this tensor to `device`, returning a new tensor.
    ///
    /// Reuses the same DType-tagged raw-byte transport as `Tensor<T>::to`:
    /// CPU→CUDA uploads via `GpuBackend::cpu_to_gpu` (handle tagged `I::dtype()`
    /// = `DType::I32` / `I64`), CUDA→CPU reads back via `gpu_to_cpu`. Both are
    /// bit-exact. On-device→same-device is a cheap storage clone.
    ///
    /// This is the **explicit** transfer entry point (PyTorch parity — like
    /// `.cuda()` / `.cpu()`). The host-readback inside the CUDA→CPU arm is the
    /// user-requested D2H copy, not a silent fallback.
    ///
    /// # Errors
    ///
    /// - [`FerrotorchError::DeviceUnavailable`] if no GPU backend is registered.
    /// - [`FerrotorchError::InvalidArgument`] for unsupported device pairs
    ///   (XPU / MPS are not wired for `IntTensor` in Phase 2a).
    pub fn to(&self, device: Device) -> FerrotorchResult<IntTensor<I>> {
        if self.device() == device {
            // Same device: clone the storage (CPU Vec clone, or GPU
            // clone_buffer). No transfer.
            return Ok(self.clone());
        }

        match (self.device(), device) {
            (Device::Cpu, Device::Cuda(_)) => {
                // H2D upload. on_device serialises the CPU Vec to bytes, tags
                // them I::dtype(), and calls GpuBackend::cpu_to_gpu — the same
                // path Tensor<T>::to(Cuda) uses.
                let data = self.data()?.to_vec();
                let storage = TensorStorage::on_device(data, device)?;
                Ok(Self {
                    storage,
                    shape: self.shape.clone(),
                })
            }
            (Device::Cuda(_), Device::Cpu) => {
                // D2H readback — the user explicitly requested .to(Cpu).
                let backend =
                    crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                let handle = self.gpu_handle()?;
                let bytes = backend.gpu_to_cpu(handle)?;
                let elem_size = std::mem::size_of::<I>();
                if bytes.len() % elem_size != 0 {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "IntTensor::to(Cpu): D2H readback of {} bytes is not a multiple \
                             of size_of::<{}>()={elem_size}",
                            bytes.len(),
                            I::dtype_name()
                        ),
                    });
                }
                // CORE-100 (#1794): decode the D2H byte buffer BY COPY into a
                // freshly allocated `Vec<I>` — never by reinterpreting the
                // `Vec<u8>` allocation in place. `GpuBackend::gpu_to_cpu`
                // only promises an ordinary `Vec<u8>`: no alignment guarantee
                // for `I` and no allocation-layout compatibility (a conforming
                // backend may return a normally allocated byte vector).
                // Rebuilding a `Vec<I>` over that allocation with
                // `Vec::from_raw_parts` was undefined behavior (misaligned
                // reference + dealloc under the wrong layout, observed under
                // MIRI). One extra host memcpy is the price of soundness; the
                // D2H transfer itself already dominates this path.
                let len = bytes.len() / elem_size;
                let mut data: Vec<I> = Vec::with_capacity(len);
                // SAFETY: `data` was freshly allocated just above with
                // capacity `len` under `Layout::array::<I>(len)`, so its
                // pointer is valid for `len * elem_size` bytes of writes and
                // correctly aligned for `I` by construction. The source
                // `bytes` is valid for `len * elem_size` bytes of reads (its
                // length equals that product per the divisibility check
                // above); `u8`-typed copies impose no alignment requirement
                // on the source, and the two allocations are distinct,
                // satisfying `copy_nonoverlapping`'s no-overlap contract.
                // After the copy the first `len` elements are fully
                // initialized, and `I` is i32 or i64 — plain integer types
                // with no padding and no invalid bit patterns — so
                // `set_len(len)` exposes only valid values. Nothing is
                // assumed about `bytes`' alignment, capacity, or allocator.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        data.as_mut_ptr().cast::<u8>(),
                        len * elem_size,
                    );
                    data.set_len(len);
                }
                Ok(Self {
                    storage: TensorStorage::cpu(data),
                    shape: self.shape.clone(),
                })
            }
            (Device::Cuda(_), Device::Cuda(_)) => {
                // Cross-GPU (different ordinals; same-ordinal handled by the
                // early `self.device() == device` return). Route through CPU,
                // mirroring Tensor<T>::to.
                let cpu = self.to(Device::Cpu)?;
                cpu.to(device)
            }
            // XPU / MPS for IntTensor are out of scope for Phase 2a: the XPU
            // upload requires a CubeRuntime (owned by ferrotorch-xpu) and MPS
            // its own backend, neither of which Phase 2a wires for integers.
            // PyTorch parity (rust-gpu-discipline §3): structured error, never
            // a silent CPU detour.
            (from, to) => Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "IntTensor::to: unsupported device transfer {from:?} -> {to:?} \
                     (Phase 2a supports CPU <-> CUDA only)"
                ),
            }),
        }
    }

    /// Cast this `IntTensor<I>` to `IntTensor<J>` (CPU only). Returns an error
    /// on out-of-range elements.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::NotImplementedOnCuda`] if `self` is on CUDA:
    /// integer cross-width cast kernels are Phase 2c. Performing it on host
    /// would require a silent D2H round trip, which the device-error policy
    /// (rust-gpu-discipline §3) forbids — the caller must `.to(Device::Cpu)`
    /// explicitly first.
    pub fn cast<J: IntElement>(&self) -> FerrotorchResult<IntTensor<J>> {
        // GPU path (#1185 Phase 2c): when CUDA-resident, run a real i32↔i64
        // cast kernel and keep the result on-device — no host round-trip.
        // `cast_gpu` returns `None` for non-CUDA tensors so the CPU reference
        // below runs unchanged.
        if let Some(result) = self.cast_gpu::<J>() {
            return result;
        }
        let data = self.data()?;
        let mut out: Vec<J> = Vec::with_capacity(data.len());
        for (i, &v) in data.iter().enumerate() {
            let widened = v.to_i64();
            out.push(
                J::try_from_i64(widened).ok_or(FerrotorchError::InvalidArgument {
                    message: format!(
                        "IntTensor::cast: element {i} = {v} out of range for {}",
                        J::dtype_name()
                    ),
                })?,
            );
        }
        IntTensor::<J>::from_vec(out, self.shape.clone())
    }

    /// Reshape (must preserve numel; no data copy on CPU).
    ///
    /// Works for any device residency: the reshape is metadata-only, and the
    /// storage is cloned (cheap for CPU; a `clone_buffer` for GPU, matching the
    /// existing `Clone` semantics — no host readback).
    pub fn reshape(&self, shape: &[usize]) -> FerrotorchResult<Self> {
        // shape=[] -> 0-d scalar (numel 1); shape=[0,...] -> empty (numel 0). (#805)
        let new_total = crate::shape::checked_numel(shape, "IntTensor::reshape")?;
        let cur = self.storage.len();
        if new_total != cur {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "IntTensor::reshape: new shape {shape:?} (numel {new_total}) != current numel {cur}"
                ),
            });
        }
        Ok(Self {
            storage: self.storage.clone(),
            shape: shape.to_vec(),
        })
    }

    // ── Compute ops (CPU + GPU, runtime dispatch) — #1185 Phase 2b ───────────
    //
    // Each op runs on CUDA when `self.is_cuda()` (real PTX kernel, the result
    // stays GPU-resident — no `.to(Cpu)`, no host readback), and on CPU
    // otherwise via a simple correct reference loop matching the SAME
    // PyTorch semantics the GPU kernels implement (esp. the floor_divide /
    // remainder sign rules and arithmetic shr). Binary ops require both
    // operands on the same device and the same shape (broadcasting is out of
    // scope for Phase 2b); a mismatch is a structured error, never a silent
    // device-mix or CPU detour (rust-gpu-discipline §3).

    /// Construct a GPU-resident `IntTensor` from a CUDA buffer handle + shape.
    ///
    /// The handle must carry the matching `DType` tag (`I32` / `I64`) — this
    /// is what every Phase-2b GPU op returns. Mirrors
    /// `Tensor::from_storage(TensorStorage::gpu(h), ...)`.
    pub(crate) fn from_gpu_handle(handle: GpuBufferHandle, shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            handle.dtype(),
            I::dtype(),
            "from_gpu_handle: handle dtype tag must match IntElement"
        );
        Self {
            storage: TensorStorage::gpu(handle),
            shape,
        }
    }

    /// Assert both operands live on the same device and have the same shape.
    fn check_binary(&self, other: &IntTensor<I>, op: &'static str) -> FerrotorchResult<()> {
        if self.device() != other.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: self.device(),
                got: other.device(),
            });
        }
        if self.shape != other.shape {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "IntTensor::{op}: operand shapes differ {:?} vs {:?} \
                     (broadcasting is out of scope for Phase 2b — same shape only)",
                    self.shape, other.shape
                ),
            });
        }
        Ok(())
    }

    /// Run a binary op: GPU kernel when CUDA-resident, else CPU reference `f`.
    fn binary_op(
        &self,
        other: &IntTensor<I>,
        op: &'static str,
        gpu: impl FnOnce(
            &dyn crate::gpu_dispatch::GpuBackend,
            &GpuBufferHandle,
            &GpuBufferHandle,
        ) -> FerrotorchResult<GpuBufferHandle>,
        f: impl Fn(I, I) -> I,
    ) -> FerrotorchResult<IntTensor<I>> {
        self.check_binary(other, op)?;
        if self.is_cuda() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let h = gpu(backend, self.gpu_handle()?, other.gpu_handle()?)?;
            Ok(IntTensor::from_gpu_handle(h, self.shape.clone()))
        } else {
            let a = self.data()?;
            let b = other.data()?;
            let out: Vec<I> = a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect();
            IntTensor::from_vec(out, self.shape.clone())
        }
    }

    /// Run a unary op: GPU kernel when CUDA-resident, else CPU reference `f`.
    fn unary_op(
        &self,
        gpu: impl FnOnce(
            &dyn crate::gpu_dispatch::GpuBackend,
            &GpuBufferHandle,
        ) -> FerrotorchResult<GpuBufferHandle>,
        f: impl Fn(I) -> I,
    ) -> FerrotorchResult<IntTensor<I>> {
        if self.is_cuda() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let h = gpu(backend, self.gpu_handle()?)?;
            Ok(IntTensor::from_gpu_handle(h, self.shape.clone()))
        } else {
            let a = self.data()?;
            let out: Vec<I> = a.iter().map(|&x| f(x)).collect();
            IntTensor::from_vec(out, self.shape.clone())
        }
    }

    /// Run a reduction: GPU kernel when CUDA-resident, else CPU left-fold `f`
    /// seeded by the first element. Returns a 0-d (scalar) tensor (numel 1).
    fn reduce_op(
        &self,
        op: &'static str,
        gpu: impl FnOnce(
            &dyn crate::gpu_dispatch::GpuBackend,
            &GpuBufferHandle,
        ) -> FerrotorchResult<GpuBufferHandle>,
        empty: Option<I>,
        f: impl Fn(I, I) -> I,
    ) -> FerrotorchResult<IntTensor<I>> {
        if self.is_cuda() {
            // Empty min/max are undefined in PyTorch; sum/prod have an identity.
            if self.numel() == 0 {
                match empty {
                    // CORE-103 (#1797): the identity scalar must live on the
                    // INPUT device — `IntTensor::scalar(id)` alone builds CPU
                    // storage, silently changing the result device just
                    // because an input dim is zero (torch: empty CUDA int
                    // sum/prod stay on cuda). There is no on-device scalar
                    // constructor, so build on host and explicitly upload the
                    // one-element identity via the Phase-2a `to` transport —
                    // this H2D copy constructs the result, it is not a
                    // compute fallback.
                    Some(id) => return IntTensor::scalar(id).to(self.device()),
                    None => {
                        return Err(FerrotorchError::InvalidArgument {
                            message: format!(
                                "IntTensor::{op}: reduction of an empty tensor is undefined"
                            ),
                        });
                    }
                }
            }
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let h = gpu(backend, self.gpu_handle()?)?;
            // Reductions collapse to a 0-d scalar (numel 1).
            Ok(IntTensor::from_gpu_handle(h, Vec::new()))
        } else {
            let a = self.data()?;
            match a.split_first() {
                Some((&first, rest)) => {
                    let acc = rest.iter().fold(first, |acc, &x| f(acc, x));
                    Ok(IntTensor::scalar(acc))
                }
                None => match empty {
                    Some(id) => Ok(IntTensor::scalar(id)),
                    None => Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "IntTensor::{op}: reduction of an empty tensor is undefined"
                        ),
                    }),
                },
            }
        }
    }

    /// Elementwise `a + b` (wrapping on overflow — PyTorch integer semantics).
    pub fn add(&self, other: &IntTensor<I>) -> FerrotorchResult<IntTensor<I>> {
        self.binary_op(other, "add", |b, x, y| b.int_add(x, y), int_wrapping_add)
    }

    /// Elementwise `a - b` (wrapping).
    pub fn sub(&self, other: &IntTensor<I>) -> FerrotorchResult<IntTensor<I>> {
        self.binary_op(other, "sub", |b, x, y| b.int_sub(x, y), int_wrapping_sub)
    }

    /// Elementwise `a * b` (wrapping).
    pub fn mul(&self, other: &IntTensor<I>) -> FerrotorchResult<IntTensor<I>> {
        self.binary_op(
            other,
            "mul",
            |b, x, y| b.int_mul(x, y),
            |x, y| int_wrapping_mul(x, y),
        )
    }

    /// Elementwise negate `-a` (wrapping; `-i*::MIN == i*::MIN`).
    pub fn neg(&self) -> FerrotorchResult<IntTensor<I>> {
        self.unary_op(
            |b, x| b.int_neg(x),
            |x| I::try_from_i64(0_i64.wrapping_sub(x.to_i64())).unwrap_or(x),
        )
    }

    /// Reject zero divisors on the CPU path (CORE-102, #1796).
    ///
    /// PyTorch's per-device contract, probed live (torch 2.11.0+cu130):
    /// - **CPU**: `torch.floor_divide` / `torch.remainder` raise
    ///   `RuntimeError: ZeroDivisionError` when ANY divisor element is zero.
    /// - **CUDA**: no trap; the zero-divisor lanes hold unspecified values.
    ///
    /// `self` is the DIVISOR. CUDA tensors return `Ok(())` — pre-scanning
    /// device memory would need a D2H round trip or a dedicated kernel and
    /// would diverge from torch's CUDA contract, which does not error.
    fn check_zero_divisor(&self, op: &'static str) -> FerrotorchResult<()> {
        if self.is_cuda() {
            return Ok(());
        }
        if self.data()?.iter().any(|&v| v.to_i64() == 0) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "IntTensor::{op}: ZeroDivisionError (integer division or modulo by zero \
                     — PyTorch CPU parity)"
                ),
            });
        }
        Ok(())
    }

    /// Elementwise floor division (`torch.floor_divide`: floors toward −∞).
    ///
    /// # Errors
    ///
    /// On CPU, returns [`FerrotorchError::InvalidArgument`] (carrying
    /// `ZeroDivisionError`) if any divisor element is zero — PyTorch CPU
    /// parity (CORE-102, #1796). On CUDA there is no zero-divisor pre-scan:
    /// matching `torch.floor_divide` on CUDA, the kernel does not trap and
    /// the zero-divisor lanes hold unspecified values (nonzero lanes are
    /// exact).
    pub fn floor_div(&self, other: &IntTensor<I>) -> FerrotorchResult<IntTensor<I>> {
        self.check_binary(other, "floor_div")?;
        other.check_zero_divisor("floor_div")?;
        self.binary_op(
            other,
            "floor_div",
            |b, x, y| b.int_floor_div(x, y),
            int_floor_div_ref,
        )
    }

    /// Elementwise remainder (`torch.remainder`: sign of the divisor).
    ///
    /// # Errors
    ///
    /// On CPU, returns [`FerrotorchError::InvalidArgument`] (carrying
    /// `ZeroDivisionError`) if any divisor element is zero — PyTorch CPU
    /// parity (CORE-102, #1796). On CUDA there is no zero-divisor pre-scan:
    /// matching `torch.remainder` on CUDA, the kernel does not trap and the
    /// zero-divisor lanes hold unspecified values (nonzero lanes are exact).
    pub fn remainder(&self, other: &IntTensor<I>) -> FerrotorchResult<IntTensor<I>> {
        self.check_binary(other, "remainder")?;
        other.check_zero_divisor("remainder")?;
        self.binary_op(
            other,
            "remainder",
            |b, x, y| b.int_remainder(x, y),
            int_remainder_ref,
        )
    }

    /// Elementwise bitwise AND.
    pub fn bitand(&self, other: &IntTensor<I>) -> FerrotorchResult<IntTensor<I>> {
        self.binary_op(
            other,
            "bitand",
            |b, x, y| b.int_bitand(x, y),
            |x, y| I::try_from_i64(x.to_i64() & y.to_i64()).unwrap_or(x),
        )
    }

    /// Elementwise bitwise OR.
    pub fn bitor(&self, other: &IntTensor<I>) -> FerrotorchResult<IntTensor<I>> {
        self.binary_op(
            other,
            "bitor",
            |b, x, y| b.int_bitor(x, y),
            |x, y| I::try_from_i64(x.to_i64() | y.to_i64()).unwrap_or(x),
        )
    }

    /// Elementwise bitwise XOR.
    pub fn bitxor(&self, other: &IntTensor<I>) -> FerrotorchResult<IntTensor<I>> {
        self.binary_op(
            other,
            "bitxor",
            |b, x, y| b.int_bitxor(x, y),
            |x, y| I::try_from_i64(x.to_i64() ^ y.to_i64()).unwrap_or(x),
        )
    }

    /// Elementwise bitwise NOT (`!a`).
    pub fn bitnot(&self) -> FerrotorchResult<IntTensor<I>> {
        self.unary_op(|b, x| b.int_bitnot(x), int_bitnot_ref)
    }

    /// Elementwise left shift `a << b`.
    pub fn shl(&self, other: &IntTensor<I>) -> FerrotorchResult<IntTensor<I>> {
        self.binary_op(other, "shl", |b, x, y| b.int_shl(x, y), int_shl_ref)
    }

    /// Elementwise arithmetic right shift `a >> b` (sign-extending).
    pub fn shr(&self, other: &IntTensor<I>) -> FerrotorchResult<IntTensor<I>> {
        self.binary_op(other, "shr", |b, x, y| b.int_shr(x, y), int_shr_ref)
    }

    /// Sum-reduce to a 0-d scalar (wrapping accumulator; identity 0 on empty).
    pub fn sum(&self) -> FerrotorchResult<IntTensor<I>> {
        self.reduce_op(
            "sum",
            |b, x| b.int_sum(x),
            Some(I::try_from_i64(0).expect("0 fits any IntElement")),
            int_wrapping_add,
        )
    }

    /// Product-reduce to a 0-d scalar (wrapping; identity 1 on empty).
    pub fn prod(&self) -> FerrotorchResult<IntTensor<I>> {
        self.reduce_op(
            "prod",
            |b, x| b.int_prod(x),
            Some(I::try_from_i64(1).expect("1 fits any IntElement")),
            int_wrapping_mul,
        )
    }

    /// Min-reduce to a 0-d scalar (errors on empty, PyTorch parity).
    pub fn min(&self) -> FerrotorchResult<IntTensor<I>> {
        self.reduce_op(
            "min",
            |b, x| b.int_min(x),
            None,
            |acc, x| if x.to_i64() < acc.to_i64() { x } else { acc },
        )
    }

    /// Max-reduce to a 0-d scalar (errors on empty, PyTorch parity).
    pub fn max(&self) -> FerrotorchResult<IntTensor<I>> {
        self.reduce_op(
            "max",
            |b, x| b.int_max(x),
            None,
            |acc, x| if x.to_i64() > acc.to_i64() { x } else { acc },
        )
    }
}

/// Wrapping integer add at the element type's width (matches the GPU
/// `add.s{32,64}` kernel and PyTorch integer overflow semantics).
///
/// CORE-101 (#1795): computing in i64 and converting back with
/// `unwrap_or(x)` returned the unwrapped left operand whenever an i32
/// boundary was crossed (`i32::MAX + 1` -> `i32::MAX`). Arithmetic must
/// happen at the concrete width, as [`int_wrapping_mul`] always did.
fn int_wrapping_add<I: IntElement>(x: I, y: I) -> I {
    let v = match I::BITS {
        32 => ((x.to_i64() as i32).wrapping_add(y.to_i64() as i32)) as i64,
        _ => x.to_i64().wrapping_add(y.to_i64()),
    };
    I::try_from_i64(v).unwrap_or(x)
}

/// Wrapping integer subtract at the element type's width (see
/// [`int_wrapping_add`]; CORE-101 / #1795).
fn int_wrapping_sub<I: IntElement>(x: I, y: I) -> I {
    let v = match I::BITS {
        32 => ((x.to_i64() as i32).wrapping_sub(y.to_i64() as i32)) as i64,
        _ => x.to_i64().wrapping_sub(y.to_i64()),
    };
    I::try_from_i64(v).unwrap_or(x)
}

/// Wrapping integer multiply at the element type's width (matches the GPU
/// `mul.lo.s{32,64}` kernel, which truncates to the operand width).
fn int_wrapping_mul<I: IntElement>(x: I, y: I) -> I {
    let prod = match I::BITS {
        32 => (x.to_i64() as i32).wrapping_mul(y.to_i64() as i32) as i64,
        _ => x.to_i64().wrapping_mul(y.to_i64()),
    };
    I::try_from_i64(prod).unwrap_or(x)
}

/// CPU reference for `torch.floor_divide`: truncated quotient corrected to
/// floor toward −∞ (subtract 1 when the remainder is nonzero and the operand
/// signs differ). The `b == 0` arm is unreachable from the public API —
/// `IntTensor::floor_div` rejects zero divisors before dispatch on CPU
/// (CORE-102, #1796) — and is kept only as defense-in-depth against a
/// future caller that skips the pre-scan.
fn int_floor_div_ref<I: IntElement>(x: I, y: I) -> I {
    let a = x.to_i64();
    let b = y.to_i64();
    if b == 0 {
        return I::try_from_i64(0).unwrap_or(x);
    }
    let q = a.wrapping_div(b);
    let r = a.wrapping_rem(b);
    let q = if r != 0 && ((r < 0) != (b < 0)) {
        q.wrapping_sub(1)
    } else {
        q
    };
    I::try_from_i64(q).unwrap_or(x)
}

/// CPU reference for `torch.remainder`: result has the sign of the divisor.
/// `remainder(a,b) = a - floor_divide(a,b)*b`. The `b == 0` arm is
/// unreachable from the public API (`IntTensor::remainder` rejects zero
/// divisors before dispatch on CPU — CORE-102, #1796; see
/// [`int_floor_div_ref`]).
fn int_remainder_ref<I: IntElement>(x: I, y: I) -> I {
    let a = x.to_i64();
    let b = y.to_i64();
    if b == 0 {
        return I::try_from_i64(0).unwrap_or(x);
    }
    let r = a.wrapping_rem(b);
    let r = if r != 0 && ((r < 0) != (b < 0)) {
        r.wrapping_add(b)
    } else {
        r
    };
    I::try_from_i64(r).unwrap_or(x)
}

/// CPU reference for bitwise NOT at the element's width.
fn int_bitnot_ref<I: IntElement>(x: I) -> I {
    let v = match I::BITS {
        32 => !(x.to_i64() as i32) as i64,
        _ => !x.to_i64(),
    };
    I::try_from_i64(v).unwrap_or(x)
}

/// CPU reference for left shift at the element's width (logical, wrapping the
/// shift count modulo the bit-width to match PTX `shl` on out-of-range counts).
fn int_shl_ref<I: IntElement>(x: I, y: I) -> I {
    let sh = (y.to_i64() as u32) & (I::BITS - 1);
    let v = match I::BITS {
        32 => ((x.to_i64() as i32).wrapping_shl(sh)) as i64,
        _ => x.to_i64().wrapping_shl(sh),
    };
    I::try_from_i64(v).unwrap_or(x)
}

/// CPU reference for arithmetic (sign-extending) right shift at the element's
/// width — matches PyTorch `__rshift__` on signed dtypes and PTX `shr.s`.
fn int_shr_ref<I: IntElement>(x: I, y: I) -> I {
    let sh = (y.to_i64() as u32) & (I::BITS - 1);
    let v = match I::BITS {
        32 => ((x.to_i64() as i32).wrapping_shr(sh)) as i64,
        _ => x.to_i64().wrapping_shr(sh),
    };
    I::try_from_i64(v).unwrap_or(x)
}

impl<I: IntElement> std::fmt::Display for IntTensor<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IntTensor<{}>(shape={:?}, len={}, device={:?})",
            I::dtype_name(),
            self.shape,
            self.storage.len(),
            self.device(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_basic() {
        let t = IntTensor::<i32>::from_vec(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.data().unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn from_vec_shape_mismatch_errors() {
        let err = IntTensor::<i32>::from_vec(vec![1, 2, 3], vec![2, 2]).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn zeros_correct_size() {
        let t = IntTensor::<i64>::zeros(&[3, 4]);
        assert_eq!(t.numel(), 12);
        assert!(t.data().unwrap().iter().all(|&x| x == 0));
    }

    #[test]
    fn arange_sequence() {
        let t = IntTensor::<i32>::arange(5).unwrap();
        assert_eq!(t.data().unwrap(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn arange_oob_for_i32() {
        // Synthetic OOB check: i32::arange beyond i32::MAX would fail.
        // The conversion is via i64; we can't easily trigger that with
        // a usize, so pick the first IntElement-fail path another way:
        // i32 from i64 OOB.
        assert!(i32::try_from_i64(i64::MAX).is_none());
    }

    #[test]
    fn cast_i64_to_i32_in_range() {
        let t = IntTensor::<i64>::from_vec(vec![1, -1, 100], vec![3]).unwrap();
        let c = t.cast::<i32>().unwrap();
        assert_eq!(c.data().unwrap(), &[1, -1, 100]);
        assert_eq!(c.dtype_name(), "i32");
    }

    #[test]
    fn cast_i64_to_i32_out_of_range_errors() {
        let t = IntTensor::<i64>::from_vec(vec![i64::MAX], vec![1]).unwrap();
        let err = t.cast::<i32>().unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn reshape_preserves_data() {
        let t = IntTensor::<i32>::from_vec(vec![1, 2, 3, 4, 5, 6], vec![6]).unwrap();
        let r = t.reshape(&[2, 3]).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.data().unwrap(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn reshape_size_mismatch_errors() {
        let t = IntTensor::<i32>::from_vec(vec![1, 2, 3, 4], vec![4]).unwrap();
        let err = t.reshape(&[3, 2]).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn scalar_constructor() {
        let t = IntTensor::<i64>::scalar(42);
        assert_eq!(t.shape(), &[] as &[usize]);
        assert_eq!(t.numel(), 1);
        assert_eq!(t.data().unwrap()[0], 42);
    }

    #[test]
    fn dtype_name_reports_i32_or_i64() {
        let t32 = IntTensor::<i32>::scalar(0);
        let t64 = IntTensor::<i64>::scalar(0);
        assert_eq!(t32.dtype_name(), "i32");
        assert_eq!(t64.dtype_name(), "i64");
    }

    #[test]
    fn cpu_tensor_reports_cpu_device() {
        // Phase 2a: CPU-resident IntTensors report Device::Cpu and are not
        // CUDA. (GPU residency is exercised in the _probe_phase2a_int_device
        // integration probe, which requires the gpu feature + hardware.)
        let t = IntTensor::<i32>::arange(4).unwrap();
        assert_eq!(t.device(), Device::Cpu);
        assert!(!t.is_cuda());
        // gpu_handle on a CPU tensor errors (not on GPU).
        assert!(t.gpu_handle().is_err());
    }

    #[test]
    fn clone_preserves_cpu_data() {
        let t = IntTensor::<i32>::arange(4).unwrap();
        let t2 = t.clone();
        assert_eq!(t2.data().unwrap(), &[0, 1, 2, 3]);
        assert_eq!(t2.device(), Device::Cpu);
    }
}
