//! Boolean tensors for masks and logical operations. (#596)
//!
//! `BoolTensor` is a contiguous tensor of `bool`s, used for `masked_fill`,
//! `where`, and any predicate-driven indexing where a float-valued mask would
//! lose semantic clarity.
//!
//! Construction:
//! - [`BoolTensor::zeros`] / [`ones`] — uniform fill
//! - [`BoolTensor::from_predicate`] — build from a `Tensor<T>` + closure
//! - [`BoolTensor::from_vec`] / [`from_slice`] — explicit data + shape
//!
//! Logical ops: `not`, `and`, `or`, `xor` are pointwise. `count_true`
//! and `any` / `all` are reductions.
//!
//! # Device residency (crosslink #1185 Phase 3a/3b)
//!
//! As of Phase 3a, `BoolTensor` is **device-aware**: its data lives in a
//! [`TensorStorage<bool>`] that carries its own [`Device`]. `bool` is a ferray
//! [`Element`](crate::dtype::Element) (`<bool as Element>::dtype() ==
//! DType::Bool`, size 1), so `TensorStorage<bool>` reuses the same DType-tagged
//! raw-byte transport (`GpuBackend::cpu_to_gpu` / `gpu_to_cpu`) as `Tensor<T>`
//! and `IntTensor<I>`. On device, a bool is stored as a `CudaSlice<u8>`
//! (cudarc has no `DeviceRepr` for `bool`; each byte is 0 or 1, byte-identical
//! to the host `&[bool]`), disambiguated by the `DType::Bool` handle tag.
//!
//! Phase 3b adds GPU compute: comparisons (`gt`/`lt`/… over float and integer
//! tensors → resident `BoolTensor`), logical ops (`and`/`or`/`xor`/`not`), and
//! the global reductions `any`/`all`. Each runs a real PTX kernel on CUDA when
//! the operands are resident; the result stays GPU-resident (no host round-trip)
//! — except `any`/`all`, whose single reduced byte is the scalar result the
//! caller asked for (PyTorch parity: `Tensor.any() -> bool`), the same as
//! `has_inf_nan`'s one-flag readback.

use crate::device::Device;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::gpu_dispatch::{CompareOp, GpuBufferHandle};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// Contiguous tensor of booleans, device-aware.
///
/// Data lives in a [`TensorStorage<bool>`] (CPU `Vec<bool>` or a CUDA
/// [`GpuBufferHandle`]); the storage carries its own [`Device`]. CPU-resident
/// behaviour is byte-identical to the pre-Phase-3a `Arc<Vec<bool>>` design.
#[derive(Debug)]
pub struct BoolTensor {
    storage: TensorStorage<bool>,
    shape: Vec<usize>,
}

impl Clone for BoolTensor {
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

impl BoolTensor {
    /// Build from a Vec + shape. Errors on numel mismatch.
    pub fn from_vec(data: Vec<bool>, shape: Vec<usize>) -> FerrotorchResult<Self> {
        // PyTorch parity: shape=[] is a 0-d scalar (numel=1); shape=[0]
        // (or any shape with a zero axis) is empty (numel=0). The previous
        // `.max(1)` conflated these. (#805)
        let expected: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        if data.len() != expected {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BoolTensor::from_vec: data.len()={} != prod(shape)={} for shape {:?}",
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

    /// Build from a slice + shape (clones into a fresh `Vec`).
    pub fn from_slice(data: &[bool], shape: &[usize]) -> FerrotorchResult<Self> {
        Self::from_vec(data.to_vec(), shape.to_vec())
    }

    /// All-false tensor of the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        // shape=[] -> 0-d scalar (numel 1); shape=[0] -> empty (numel 0). (#805)
        let total: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        Self {
            storage: TensorStorage::cpu(vec![false; total]),
            shape: shape.to_vec(),
        }
    }

    /// All-true tensor of the given shape.
    pub fn ones(shape: &[usize]) -> Self {
        // shape=[] -> 0-d scalar (numel 1); shape=[0] -> empty (numel 0). (#805)
        let total: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        Self {
            storage: TensorStorage::cpu(vec![true; total]),
            shape: shape.to_vec(),
        }
    }

    /// Build a mask by applying `pred` to every element of `t`.
    /// Useful for `Tensor < 0`, `Tensor.is_finite()`, etc.
    pub fn from_predicate<T: Float>(
        t: &Tensor<T>,
        pred: impl Fn(T) -> bool,
    ) -> FerrotorchResult<Self> {
        let data = t.data_vec()?;
        let mask: Vec<bool> = data.iter().map(|&v| pred(v)).collect();
        Self::from_vec(mask, t.shape().to_vec())
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
    /// [`IntTensor::data`](crate::int_tensor::IntTensor::data) and prevents
    /// callers from silently reading device memory as a host slice — call
    /// [`Self::to(Device::Cpu)`](Self::to) first to transfer.
    pub fn data(&self) -> FerrotorchResult<&[bool]> {
        self.storage.try_as_slice()
    }

    /// Get the CUDA buffer handle. Returns `Err` for CPU (and other non-CUDA)
    /// tensors, mirroring [`Tensor::gpu_handle`](crate::tensor::Tensor::gpu_handle).
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] when the tensor is not
    /// CUDA-resident.
    pub fn gpu_handle(&self) -> FerrotorchResult<&GpuBufferHandle> {
        self.storage
            .gpu_handle()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "BoolTensor is not on a CUDA GPU".into(),
            })
    }

    /// Construct a GPU-resident `BoolTensor` from a CUDA buffer handle + shape.
    ///
    /// The handle must carry the `DType::Bool` tag — this is what every
    /// Phase-3b GPU op returns. Mirrors
    /// [`IntTensor::from_gpu_handle`](crate::int_tensor::IntTensor).
    pub fn from_gpu_handle(handle: GpuBufferHandle, shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            handle.dtype(),
            <bool as crate::dtype::Element>::dtype(),
            "from_gpu_handle: handle dtype tag must be Bool"
        );
        Self {
            storage: TensorStorage::gpu(handle),
            shape,
        }
    }

    /// Move this tensor to `device`, returning a new tensor.
    ///
    /// Reuses the same DType-tagged raw-byte transport as
    /// [`Tensor::to`](crate::tensor::Tensor) / `IntTensor::to`: CPU→CUDA uploads
    /// via `GpuBackend::cpu_to_gpu` (handle tagged `DType::Bool`), CUDA→CPU reads
    /// back via `gpu_to_cpu`. Both are bit-exact. On-device→same-device is a
    /// cheap storage clone.
    ///
    /// This is the **explicit** transfer entry point (PyTorch parity — like
    /// `.cuda()` / `.cpu()`). The host-readback inside the CUDA→CPU arm is the
    /// user-requested D2H copy, not a silent fallback.
    ///
    /// # Errors
    ///
    /// - [`FerrotorchError::DeviceUnavailable`] if no GPU backend is registered.
    /// - [`FerrotorchError::InvalidArgument`] for unsupported device pairs
    ///   (XPU / MPS are not wired for `BoolTensor`).
    pub fn to(&self, device: Device) -> FerrotorchResult<BoolTensor> {
        if self.device() == device {
            return Ok(self.clone());
        }
        match (self.device(), device) {
            (Device::Cpu, Device::Cuda(_)) => {
                // H2D upload via the DType::Bool-tagged transport.
                let data = self.data()?.to_vec();
                let storage = TensorStorage::on_device(data, device)?;
                Ok(Self {
                    storage,
                    shape: self.shape.clone(),
                })
            }
            (Device::Cuda(_), Device::Cpu) => {
                // D2H readback — the user explicitly requested .to(Cpu). The
                // backend returns the raw bytes (one byte per bool, each 0/1);
                // reconstruct Vec<bool> from them.
                let backend = crate::gpu_dispatch::gpu_backend()
                    .ok_or(FerrotorchError::DeviceUnavailable)?;
                let handle = self.gpu_handle()?;
                let bytes = backend.gpu_to_cpu(handle)?;
                // Each byte is 0 or 1 (kernels emit canonical 0/1; uploads were
                // of `bool` values). Map to bool defensively (nonzero -> true)
                // so a stray nonzero never produces an invalid `bool` bit
                // pattern.
                let data: Vec<bool> = bytes.iter().map(|&b| b != 0).collect();
                Ok(Self {
                    storage: TensorStorage::cpu(data),
                    shape: self.shape.clone(),
                })
            }
            (Device::Cuda(_), Device::Cuda(_)) => {
                // Cross-GPU: route through CPU (same as Tensor/IntTensor).
                let cpu = self.to(Device::Cpu)?;
                cpu.to(device)
            }
            (from, to) => Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "BoolTensor::to: unsupported device transfer {from:?} -> {to:?} \
                     (CPU <-> CUDA only)"
                ),
            }),
        }
    }

    /// Pointwise NOT. GPU path (resident) when CUDA-resident, else CPU.
    pub fn not(&self) -> Self {
        if self.is_cuda() {
            // GPU path: a CUDA NOT failure is a programmer/driver error here
            // (the kernel is unconditional), so unwrap into a panic rather than
            // silently degrading — `not` has the infallible `-> Self` signature.
            return self
                .unary_gpu(|b, h| b.bool_not(h))
                .expect("BoolTensor::not GPU kernel");
        }
        let out: Vec<bool> = self
            .data()
            .expect("CPU BoolTensor data")
            .iter()
            .map(|&b| !b)
            .collect();
        Self {
            storage: TensorStorage::cpu(out),
            shape: self.shape.clone(),
        }
    }

    /// Pointwise AND. Errors on shape/device mismatch.
    pub fn and(&self, other: &Self) -> FerrotorchResult<Self> {
        self.binary_op(other, |b, a, c| b.bool_and(a, c), |a, b| a && b, "and")
    }

    /// Pointwise OR.
    pub fn or(&self, other: &Self) -> FerrotorchResult<Self> {
        self.binary_op(other, |b, a, c| b.bool_or(a, c), |a, b| a || b, "or")
    }

    /// Pointwise XOR.
    pub fn xor(&self, other: &Self) -> FerrotorchResult<Self> {
        self.binary_op(other, |b, a, c| b.bool_xor(a, c), |a, b| a ^ b, "xor")
    }

    /// Run a logical unary op on GPU (CUDA-resident input → resident output).
    fn unary_gpu(
        &self,
        gpu: impl FnOnce(
            &dyn crate::gpu_dispatch::GpuBackend,
            &GpuBufferHandle,
        ) -> FerrotorchResult<GpuBufferHandle>,
    ) -> FerrotorchResult<Self> {
        let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let h = gpu(backend, self.gpu_handle()?)?;
        Ok(Self::from_gpu_handle(h, self.shape.clone()))
    }

    /// Run a logical binary op: GPU kernel when CUDA-resident, else CPU `f`.
    fn binary_op(
        &self,
        other: &Self,
        gpu: impl FnOnce(
            &dyn crate::gpu_dispatch::GpuBackend,
            &GpuBufferHandle,
            &GpuBufferHandle,
        ) -> FerrotorchResult<GpuBufferHandle>,
        f: impl Fn(bool, bool) -> bool,
        op_name: &str,
    ) -> FerrotorchResult<Self> {
        if self.device() != other.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: self.device(),
                got: other.device(),
            });
        }
        if self.shape != other.shape {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BoolTensor::{op_name}: shapes {:?} vs {:?}",
                    self.shape, other.shape
                ),
            });
        }
        if self.is_cuda() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let h = gpu(backend, self.gpu_handle()?, other.gpu_handle()?)?;
            return Ok(Self::from_gpu_handle(h, self.shape.clone()));
        }
        let out: Vec<bool> = self
            .data()?
            .iter()
            .zip(other.data()?.iter())
            .map(|(&a, &b)| f(a, b))
            .collect();
        Ok(Self {
            storage: TensorStorage::cpu(out),
            shape: self.shape.clone(),
        })
    }

    /// Reshape (must preserve numel). Metadata-only; the storage is cloned
    /// (cheap for CPU; a `clone_buffer` for GPU — no host readback).
    pub fn reshape(&self, shape: &[usize]) -> FerrotorchResult<Self> {
        // shape=[] -> 0-d scalar (numel 1); shape=[0,...] -> empty (numel 0). (#805)
        let new_total: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        let cur = self.storage.len();
        if new_total != cur {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BoolTensor::reshape: new shape {shape:?} (numel {new_total}) != current numel {cur}"
                ),
            });
        }
        Ok(Self {
            storage: self.storage.clone(),
            shape: shape.to_vec(),
        })
    }

    /// Number of true elements (CPU only).
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::GpuTensorNotAccessible`] for a CUDA-resident
    /// tensor: a host count would require a full-buffer D2H copy, which the
    /// device-error policy forbids. Use [`Self::any`] / [`Self::all`] (which run
    /// the reduction on-device) or `.to(Device::Cpu)` explicitly first.
    pub fn count_true(&self) -> FerrotorchResult<usize> {
        Ok(self.data()?.iter().filter(|&&b| b).count())
    }

    /// True if any element is true.
    ///
    /// On CUDA the OR-reduction runs on the GPU (real PTX kernel); only the
    /// single reduced byte crosses to the host — that byte IS the scalar result
    /// (PyTorch parity: `Tensor.any() -> bool`), NOT a full-buffer round trip.
    pub fn any(&self) -> FerrotorchResult<bool> {
        if self.is_cuda() {
            return self.reduce_gpu(|b, h| b.bool_any(h));
        }
        Ok(self.data()?.iter().any(|&b| b))
    }

    /// True if all elements are true.
    ///
    /// On CUDA the AND-reduction runs on the GPU; only the single reduced byte
    /// crosses to the host (the scalar result, not a buffer round trip).
    pub fn all(&self) -> FerrotorchResult<bool> {
        if self.is_cuda() {
            return self.reduce_gpu(|b, h| b.bool_all(h));
        }
        Ok(self.data()?.iter().all(|&b| b))
    }

    /// Run a global bool reduction on GPU and read back ONLY the single reduced
    /// byte (the scalar result). The reduction itself ran on-device.
    fn reduce_gpu(
        &self,
        gpu: impl FnOnce(
            &dyn crate::gpu_dispatch::GpuBackend,
            &GpuBufferHandle,
        ) -> FerrotorchResult<GpuBufferHandle>,
    ) -> FerrotorchResult<bool> {
        let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let reduced = gpu(backend, self.gpu_handle()?)?;
        // The reduced handle is a 1-element Bool buffer. Read back the single
        // scalar byte (allowed: it is the result, like has_inf_nan's flag).
        let bytes = backend.gpu_to_cpu(&reduced)?;
        Ok(bytes.first().is_some_and(|&b| b != 0))
    }

    // ── Float comparison constructors (#615; GPU path #1185 Phase 3b) ────────
    //
    // Each compares two float `Tensor<T>` of the SAME shape and device. On CUDA
    // it launches the value-typed comparison PTX kernel and the resulting
    // `BoolTensor` stays GPU-resident; on CPU it runs the reference closure.

    /// Pointwise `>` comparing two float tensors of the same shape;
    /// produces a `BoolTensor` of matching shape. Mirrors
    /// `torch.gt(a, b)` returning a bool tensor. (#615)
    pub fn gt<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Self> {
        Self::compare_float(a, b, CompareOp::Gt, |x, y| x > y)
    }

    /// Pointwise `<`. (#615)
    pub fn lt<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Self> {
        Self::compare_float(a, b, CompareOp::Lt, |x, y| x < y)
    }

    /// Pointwise `>=`. (#615)
    pub fn ge<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Self> {
        Self::compare_float(a, b, CompareOp::Ge, |x, y| x >= y)
    }

    /// Pointwise `<=`. (#615)
    pub fn le<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Self> {
        Self::compare_float(a, b, CompareOp::Le, |x, y| x <= y)
    }

    /// Pointwise `==`. (#615)
    pub fn eq_t<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Self> {
        Self::compare_float(a, b, CompareOp::Eq, |x, y| x == y)
    }

    /// Pointwise `!=`. (#615)
    pub fn ne<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Self> {
        Self::compare_float(a, b, CompareOp::Ne, |x, y| x != y)
    }

    fn compare_float<T: Float>(
        a: &Tensor<T>,
        b: &Tensor<T>,
        op: CompareOp,
        f: impl Fn(T, T) -> bool,
    ) -> FerrotorchResult<Self> {
        if a.shape() != b.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BoolTensor::{}: shapes {:?} vs {:?}",
                    op.suffix(),
                    a.shape(),
                    b.shape()
                ),
            });
        }
        if a.device() != b.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: a.device(),
                got: b.device(),
            });
        }
        if a.is_cuda() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let h = backend.compare(a.gpu_handle()?, b.gpu_handle()?, op)?;
            return Ok(Self::from_gpu_handle(h, a.shape().to_vec()));
        }
        let a_data = a.data_vec()?;
        let b_data = b.data_vec()?;
        let result: Vec<bool> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| f(x, y))
            .collect();
        Self::from_vec(result, a.shape().to_vec())
    }

    // ── Integer comparison constructors (#1185 Phase 3b) ─────────────────────
    //
    // Parallel to the float constructors, taking `&IntTensor<I>`. On CUDA they
    // launch the i32/i64 comparison kernel (dispatched on the handle's tag);
    // on CPU they compare the host slices.

    /// Pointwise `a > b` over two integer tensors → `BoolTensor`. (#1185)
    pub fn gt_int<I: crate::int_tensor::IntElement>(
        a: &crate::int_tensor::IntTensor<I>,
        b: &crate::int_tensor::IntTensor<I>,
    ) -> FerrotorchResult<Self> {
        Self::compare_int(a, b, CompareOp::Gt, |x, y| x > y)
    }

    /// Pointwise `a < b` over two integer tensors. (#1185)
    pub fn lt_int<I: crate::int_tensor::IntElement>(
        a: &crate::int_tensor::IntTensor<I>,
        b: &crate::int_tensor::IntTensor<I>,
    ) -> FerrotorchResult<Self> {
        Self::compare_int(a, b, CompareOp::Lt, |x, y| x < y)
    }

    /// Pointwise `a >= b` over two integer tensors. (#1185)
    pub fn ge_int<I: crate::int_tensor::IntElement>(
        a: &crate::int_tensor::IntTensor<I>,
        b: &crate::int_tensor::IntTensor<I>,
    ) -> FerrotorchResult<Self> {
        Self::compare_int(a, b, CompareOp::Ge, |x, y| x >= y)
    }

    /// Pointwise `a <= b` over two integer tensors. (#1185)
    pub fn le_int<I: crate::int_tensor::IntElement>(
        a: &crate::int_tensor::IntTensor<I>,
        b: &crate::int_tensor::IntTensor<I>,
    ) -> FerrotorchResult<Self> {
        Self::compare_int(a, b, CompareOp::Le, |x, y| x <= y)
    }

    /// Pointwise `a == b` over two integer tensors. (#1185)
    pub fn eq_int<I: crate::int_tensor::IntElement>(
        a: &crate::int_tensor::IntTensor<I>,
        b: &crate::int_tensor::IntTensor<I>,
    ) -> FerrotorchResult<Self> {
        Self::compare_int(a, b, CompareOp::Eq, |x, y| x == y)
    }

    /// Pointwise `a != b` over two integer tensors. (#1185)
    pub fn ne_int<I: crate::int_tensor::IntElement>(
        a: &crate::int_tensor::IntTensor<I>,
        b: &crate::int_tensor::IntTensor<I>,
    ) -> FerrotorchResult<Self> {
        Self::compare_int(a, b, CompareOp::Ne, |x, y| x != y)
    }

    fn compare_int<I: crate::int_tensor::IntElement>(
        a: &crate::int_tensor::IntTensor<I>,
        b: &crate::int_tensor::IntTensor<I>,
        op: CompareOp,
        f: impl Fn(i64, i64) -> bool,
    ) -> FerrotorchResult<Self> {
        if a.shape() != b.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BoolTensor::{}_int: shapes {:?} vs {:?}",
                    op.suffix(),
                    a.shape(),
                    b.shape()
                ),
            });
        }
        if a.device() != b.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: a.device(),
                got: b.device(),
            });
        }
        if a.is_cuda() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let h = backend.compare(a.gpu_handle()?, b.gpu_handle()?, op)?;
            return Ok(Self::from_gpu_handle(h, a.shape().to_vec()));
        }
        let result: Vec<bool> = a
            .data()?
            .iter()
            .zip(b.data()?.iter())
            .map(|(&x, &y)| f(x.to_i64(), y.to_i64()))
            .collect();
        Self::from_vec(result, a.shape().to_vec())
    }

    /// Convert to a float tensor: true → 1.0, false → 0.0.
    ///
    /// On CUDA the cast runs on the GPU and the result stays resident (no host
    /// round-trip); on CPU it maps the host slice.
    pub fn to_float<T: Float>(&self) -> FerrotorchResult<Tensor<T>> {
        if self.is_cuda() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let h = backend.cast_bool_to_f(self.gpu_handle()?, <T as crate::dtype::Element>::dtype())?;
            return Tensor::from_storage(TensorStorage::gpu(h), self.shape.clone(), false);
        }
        let one = T::from(1.0).unwrap();
        let zero = T::from(0.0).unwrap();
        let data: Vec<T> = self
            .data()?
            .iter()
            .map(|&b| if b { one } else { zero })
            .collect();
        Tensor::from_storage(TensorStorage::cpu(data), self.shape.clone(), false)
    }
}

impl std::fmt::Display for BoolTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BoolTensor(shape={:?}, len={}, device={:?})",
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
    fn zeros_and_ones() {
        let z = BoolTensor::zeros(&[2, 3]);
        let o = BoolTensor::ones(&[2, 3]);
        assert_eq!(z.numel(), 6);
        assert_eq!(o.numel(), 6);
        assert!(z.data().unwrap().iter().all(|&b| !b));
        assert!(o.data().unwrap().iter().all(|&b| b));
    }

    #[test]
    fn from_vec_shape_mismatch_errors() {
        let err = BoolTensor::from_vec(vec![true, false], vec![3]).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn from_predicate_builds_mask() {
        let t = crate::creation::from_slice::<f32>(&[-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();
        let mask = BoolTensor::from_predicate(&t, |x| x > 0.0).unwrap();
        assert_eq!(mask.data().unwrap(), &[false, false, true, true]);
    }

    #[test]
    fn pointwise_not() {
        let m = BoolTensor::from_vec(vec![true, false, true], vec![3]).unwrap();
        let n = m.not();
        assert_eq!(n.data().unwrap(), &[false, true, false]);
    }

    #[test]
    fn pointwise_and_or_xor() {
        let a = BoolTensor::from_vec(vec![true, false, true, false], vec![4]).unwrap();
        let b = BoolTensor::from_vec(vec![true, true, false, false], vec![4]).unwrap();
        assert_eq!(a.and(&b).unwrap().data().unwrap(), &[true, false, false, false]);
        assert_eq!(a.or(&b).unwrap().data().unwrap(), &[true, true, true, false]);
        assert_eq!(a.xor(&b).unwrap().data().unwrap(), &[false, true, true, false]);
    }

    #[test]
    fn binary_op_shape_mismatch() {
        let a = BoolTensor::ones(&[3]);
        let b = BoolTensor::ones(&[2]);
        assert!(matches!(
            a.and(&b).unwrap_err(),
            FerrotorchError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn count_true_any_all() {
        let m = BoolTensor::from_vec(vec![true, false, true], vec![3]).unwrap();
        assert_eq!(m.count_true().unwrap(), 2);
        assert!(m.any().unwrap());
        assert!(!m.all().unwrap());

        let z = BoolTensor::zeros(&[3]);
        assert!(!z.any().unwrap());
        assert_eq!(z.count_true().unwrap(), 0);

        let o = BoolTensor::ones(&[3]);
        assert!(o.all().unwrap());
        assert_eq!(o.count_true().unwrap(), 3);
    }

    #[test]
    fn reshape_preserves_data() {
        let m = BoolTensor::from_vec(vec![true, false, true, false, true, false], vec![6]).unwrap();
        let r = m.reshape(&[2, 3]).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.data().unwrap(), m.data().unwrap());
    }

    #[test]
    fn to_float_emits_zeros_and_ones() {
        let m = BoolTensor::from_vec(vec![true, false, true], vec![3]).unwrap();
        let f = m.to_float::<f32>().unwrap();
        assert_eq!(f.data().unwrap(), &[1.0_f32, 0.0, 1.0]);
    }

    #[test]
    fn cpu_tensor_reports_cpu_device() {
        // Phase 3a: CPU-resident BoolTensors report Device::Cpu and are not
        // CUDA. (GPU residency is exercised in the _probe_phase3_bool_ops
        // integration probe, which requires the gpu feature + hardware.)
        let m = BoolTensor::ones(&[5]);
        assert_eq!(m.device(), Device::Cpu);
        assert!(!m.is_cuda());
        // gpu_handle on a CPU tensor errors (not on GPU).
        assert!(m.gpu_handle().is_err());
    }

    #[test]
    fn clone_preserves_cpu_data() {
        let m = BoolTensor::from_vec(vec![true, false, true, true, false], vec![5]).unwrap();
        let m2 = m.clone();
        assert_eq!(m2.data().unwrap(), &[true, false, true, true, false]);
        assert_eq!(m2.device(), Device::Cpu);
    }

    // -----------------------------------------------------------------------
    // Comparison ops returning BoolTensor (#615)
    // -----------------------------------------------------------------------

    #[test]
    fn compare_gt_basic() {
        let a = crate::creation::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = crate::creation::from_slice::<f32>(&[0.0, 3.0, 3.0, 5.0], &[4]).unwrap();
        let m = BoolTensor::gt(&a, &b).unwrap();
        assert_eq!(m.data().unwrap(), &[true, false, false, false]);
    }

    #[test]
    fn compare_lt_basic() {
        let a = crate::creation::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = crate::creation::from_slice::<f32>(&[2.0, 2.0, 4.0], &[3]).unwrap();
        let m = BoolTensor::lt(&a, &b).unwrap();
        assert_eq!(m.data().unwrap(), &[true, false, true]);
    }

    #[test]
    fn compare_ge_le() {
        let a = crate::creation::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = crate::creation::from_slice::<f32>(&[1.0, 3.0, 2.0], &[3]).unwrap();
        assert_eq!(BoolTensor::ge(&a, &b).unwrap().data().unwrap(), &[true, false, true]);
        assert_eq!(BoolTensor::le(&a, &b).unwrap().data().unwrap(), &[true, true, false]);
    }

    #[test]
    fn compare_eq_ne() {
        let a = crate::creation::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = crate::creation::from_slice::<f32>(&[1.0, 5.0, 3.0], &[3]).unwrap();
        assert_eq!(
            BoolTensor::eq_t(&a, &b).unwrap().data().unwrap(),
            &[true, false, true]
        );
        assert_eq!(
            BoolTensor::ne(&a, &b).unwrap().data().unwrap(),
            &[false, true, false]
        );
    }

    #[test]
    fn compare_int_basic() {
        use crate::int_tensor::IntTensor;
        let a = IntTensor::<i32>::from_vec(vec![1, 5, 3, 8], vec![4]).unwrap();
        let b = IntTensor::<i32>::from_vec(vec![2, 5, 1, 8], vec![4]).unwrap();
        assert_eq!(
            BoolTensor::gt_int(&a, &b).unwrap().data().unwrap(),
            &[false, false, true, false]
        );
        assert_eq!(
            BoolTensor::eq_int(&a, &b).unwrap().data().unwrap(),
            &[false, true, false, true]
        );
        assert_eq!(
            BoolTensor::le_int(&a, &b).unwrap().data().unwrap(),
            &[true, true, false, true]
        );
    }

    #[test]
    fn compare_rejects_shape_mismatch() {
        let a = crate::creation::from_slice::<f32>(&[1.0, 2.0], &[2]).unwrap();
        let b = crate::creation::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let err = BoolTensor::gt(&a, &b).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }
}
