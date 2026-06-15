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

//!
//! ## REQ status (per `.design/ferrotorch-core/bool_tensor.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (constructors) | SHIPPED | `BoolTensor` at `bool_tensor.rs:64`; `from_vec` at `:83`, `from_slice` at `:109`, `zeros` at `:114`, `ones` at `:128`, `from_predicate` at `:161`; consumer `grad_fns/comparison.rs:234` `BoolTensor::from_vec(...)` for `where_cond` mask; `grad_fns/indexing.rs:407` `BoolTensor::from_slice(...)` for masked-fill mask |
//! | REQ-2 (device methods) | SHIPPED | `device` at `bool_tensor.rs:195`, `is_cuda` at `:201`, `to` at `:306`; consumer `ops/indexing.rs:398` `where_cond` reads `cond.device()` to dispatch GPU vs CPU |
//! | REQ-3 (logical ops) | SHIPPED | `not` at `bool_tensor.rs:353`, `and` / `or` / `xor` at `:377-389`; consumer `grad_fns/indexing.rs` consumes mask buffers — `binary_op` helper at `:416` dispatches GPU PTX kernels (`bool_and` / `bool_or` / `bool_xor` / `bool_not`) |
//! | REQ-4 (reductions) | SHIPPED | `count_true` at `bool_tensor.rs:516`, `any` at `:525`, `all` at `:536`; consumer `grad_fns/indexing.rs` uses `BoolTensor::any` to detect empty-mask before dependent kernel launches |
//! | REQ-5 (float comparisons) | SHIPPED | `gt` / `lt` / `ge` / `le` / `eq_t` / `ne` at `bool_tensor.rs:573-598` + `compare_float` at `:636`; consumer `grad_fns/comparison.rs` invokes `BoolTensor::eq_t` etc. mirroring `torch.gt(a, b)` (`aten/src/ATen/native/Compare.cpp`) |
//! | REQ-6 (integer comparisons) | SHIPPED | `gt_int` / `lt_int` / `ge_int` / `le_int` / `eq_int` / `ne_int` at `bool_tensor.rs:721-761` + `compare_int` at `:782`; CUDA same-shape and broadcasted i32/i64 operands stay resident through `GpuBackend::compare` / `compare_broadcast`; consumer `lib.rs:135` re-export; downstream integer-tensor predicate code |
//! | REQ-7 (to_float) | SHIPPED | `to_float<T: Float>` at `bool_tensor.rs:843`; consumer `grad_fns/indexing.rs` `masked_select` materializes float tensors from `BoolTensor` masks; test `to_float_emits_zeros_and_ones` at `:989` |
//! | REQ-8 (reshape) | SHIPPED | `reshape` at `bool_tensor.rs:487`; consumer `grad_fns/indexing.rs` reshapes mask buffers to match broadcast shape; test `reshape_preserves_data` at `:981` |
//! | REQ-9 (gpu_handle) | SHIPPED | `from_gpu_handle` at `bool_tensor.rs:250` (fallible — CORE-104/#1798), `gpu_handle` at `:225`; consumer every GPU comparison-op return path (`compare_float` at `:682`, `binary_op` at `:453`, `unary_gpu` at `:404`) |
//! | REQ-10 (0-D vs zero-axis) | SHIPPED | `shape.is_empty() { 1 } else { product }` at `bool_tensor.rs:87, :116, :130, :489`; consumer `grad_fns/indexing.rs` 0-D mask handling — #805 regression pin |
//! | REQ-11 (structured errors) | SHIPPED | `ShapeMismatch` / `DeviceMismatch` / `InvalidArgument` at multiple sites; no `panic!` in production paths; consumer `grad_fns/comparison.rs` and `grad_fns/indexing.rs` propagate via `?` |

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
        let expected = crate::shape::checked_numel(&shape, "BoolTensor::from_vec")?;
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
        let total = crate::shape::numel(shape);
        Self {
            storage: TensorStorage::cpu(vec![false; total]),
            shape: shape.to_vec(),
        }
    }

    /// All-true tensor of the given shape.
    pub fn ones(shape: &[usize]) -> Self {
        // shape=[] -> 0-d scalar (numel 1); shape=[0] -> empty (numel 0). (#805)
        let total = crate::shape::numel(shape);
        Self {
            storage: TensorStorage::cpu(vec![true; total]),
            shape: shape.to_vec(),
        }
    }

    /// Build a mask by applying `pred` to every element of `t`.
    /// Useful for `Tensor < 0`, `Tensor.is_finite()`, etc.
    ///
    /// # Device behavior (CORE-105 / #1799, R-LOUD-2)
    ///
    /// The returned mask lives on **`t`'s device** — torch parity:
    /// predicate-style mask ops keep the input device
    /// (`torch.gt(t_cuda, 0).device == cuda:0`, live torch 2.11.0+cu130).
    ///
    /// `pred` is an arbitrary host closure, so for a CUDA input this is an
    /// explicit, documented host round trip: the values are read back via
    /// `data_vec` (full-device D2H copy), the predicate runs on the host,
    /// and the mask is uploaded back to `t`'s device. Resident alternatives
    /// for the common predicates are the GPU comparison constructors
    /// ([`Self::gt`], [`Self::lt`], …), which never leave the device.
    ///
    /// # Errors
    ///
    /// Propagates `data_vec` errors (e.g. meta tensors) and `.to(device)`
    /// errors for device pairs without a registered transfer path.
    pub fn from_predicate<T: Float>(
        t: &Tensor<T>,
        pred: impl Fn(T) -> bool,
    ) -> FerrotorchResult<Self> {
        // Documented D2H readback for non-CPU inputs (see doc-comment).
        let data = t.data_vec()?;
        let mask: Vec<bool> = data.iter().map(|&v| pred(v)).collect();
        let cpu_mask = Self::from_vec(mask, t.shape().to_vec())?;
        if t.device() == Device::Cpu {
            Ok(cpu_mask)
        } else {
            // Re-upload: the mask's device is the input's device (torch
            // parity), never a silent CPU demotion.
            cpu_mask.to(t.device())
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
    /// Phase-3b GPU op returns.
    ///
    /// # Errors (CORE-104 / #1798)
    ///
    /// Construction is fallible — every invariant later kernels and readback
    /// trust is validated here, in release builds too (formerly a
    /// `debug_assert` for the dtype and no length check at all):
    ///
    /// - [`FerrotorchError::DtypeMismatch`] when `handle.dtype()` is not
    ///   `DType::Bool`.
    /// - [`FerrotorchError::ShapeMismatch`] when the shape's element count
    ///   overflows `usize`, or when `handle.len()` differs from it
    ///   (`shape == []` is the 0-d scalar, numel 1 — the same #805
    ///   convention as [`Self::from_vec`]).
    pub fn from_gpu_handle(handle: GpuBufferHandle, shape: Vec<usize>) -> FerrotorchResult<Self> {
        let expected_dtype = <bool as crate::dtype::Element>::dtype();
        if handle.dtype() != expected_dtype {
            return Err(FerrotorchError::DtypeMismatch {
                expected: format!("{expected_dtype:?}"),
                got: format!("{:?}", handle.dtype()),
            });
        }
        // shape=[] -> 0-d scalar (numel 1); zero axes -> empty (numel 0). (#805)
        let expected_numel = if shape.is_empty() {
            Some(1usize)
        } else {
            shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
        };
        let Some(expected_numel) = expected_numel else {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BoolTensor::from_gpu_handle: element count of shape {shape:?} \
                     overflows usize"
                ),
            });
        };
        if handle.len() != expected_numel {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BoolTensor::from_gpu_handle: handle.len()={} != prod(shape)={} \
                     for shape {:?}",
                    handle.len(),
                    expected_numel,
                    shape
                ),
            });
        }
        Ok(Self {
            storage: TensorStorage::gpu(handle),
            shape,
        })
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
                let backend =
                    crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
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

    /// Pointwise AND, broadcasting compatible shapes (mirrors
    /// `torch.logical_and` — #1800). Errors on device mismatch or
    /// non-broadcastable shapes.
    pub fn and(&self, other: &Self) -> FerrotorchResult<Self> {
        self.binary_op(other, |b, a, c| b.bool_and(a, c), |a, b| a && b, "and")
    }

    /// Pointwise OR, broadcasting compatible shapes (mirrors
    /// `torch.logical_or` — #1800).
    pub fn or(&self, other: &Self) -> FerrotorchResult<Self> {
        self.binary_op(other, |b, a, c| b.bool_or(a, c), |a, b| a || b, "or")
    }

    /// Pointwise XOR, broadcasting compatible shapes (mirrors
    /// `torch.logical_xor` — #1800).
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
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let h = gpu(backend, self.gpu_handle()?)?;
        Self::from_gpu_handle(h, self.shape.clone())
    }

    /// Run a logical binary op: GPU kernel when CUDA-resident, else CPU `f`.
    ///
    /// Operands broadcast to their common shape per torch's
    /// `logical_and`/`or`/`xor` semantics (CORE-106 / #1800). On CUDA each
    /// non-common-shape operand is expanded ENTIRELY on device through the
    /// `broadcast_bool` kernel (#1663) before the logical kernel runs — no
    /// host round trip. Incompatible shapes are a structured
    /// [`FerrotorchError::ShapeMismatch`] from
    /// [`crate::shape::broadcast_shapes`].
    fn binary_op(
        &self,
        other: &Self,
        gpu: impl FnOnce(
            &dyn crate::gpu_dispatch::GpuBackend,
            &GpuBufferHandle,
            &GpuBufferHandle,
        ) -> FerrotorchResult<GpuBufferHandle>,
        f: impl Fn(bool, bool) -> bool,
        _op_name: &str,
    ) -> FerrotorchResult<Self> {
        if self.device() != other.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: self.device(),
                got: other.device(),
            });
        }
        let common = crate::shape::broadcast_shapes(&self.shape, &other.shape)?;
        if self.is_cuda() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            // Expand mismatched operands on-device (broadcast_bool, #1663).
            let a_expanded;
            let a_handle = if self.shape == common {
                self.gpu_handle()?
            } else {
                a_expanded = backend.broadcast_bool(self.gpu_handle()?, &self.shape, &common)?;
                &a_expanded
            };
            let b_expanded;
            let b_handle = if other.shape == common {
                other.gpu_handle()?
            } else {
                b_expanded = backend.broadcast_bool(other.gpu_handle()?, &other.shape, &common)?;
                &b_expanded
            };
            let h = gpu(backend, a_handle, b_handle)?;
            return Self::from_gpu_handle(h, common);
        }
        let a_data = self.data()?;
        let b_data = other.data()?;
        if self.shape == other.shape {
            // Fast path: element counts agree, plain zip.
            let out: Vec<bool> = a_data
                .iter()
                .zip(b_data.iter())
                .map(|(&a, &b)| f(a, b))
                .collect();
            return Ok(Self {
                storage: TensorStorage::cpu(out),
                shape: common,
            });
        }
        let numel: usize = if common.is_empty() {
            1
        } else {
            crate::shape::numel(&common)
        };
        let out: Vec<bool> = (0..numel)
            .map(|i| {
                f(
                    a_data[broadcast_src_flat(i, &common, &self.shape)],
                    b_data[broadcast_src_flat(i, &common, &other.shape)],
                )
            })
            .collect();
        Self::from_vec(out, common)
    }

    /// Reshape (must preserve numel). Metadata-only; the storage is cloned
    /// (cheap for CPU; a `clone_buffer` for GPU — no host readback).
    pub fn reshape(&self, shape: &[usize]) -> FerrotorchResult<Self> {
        // shape=[] -> 0-d scalar (numel 1); shape=[0,...] -> empty (numel 0). (#805)
        let new_total = crate::shape::checked_numel(shape, "BoolTensor::reshape")?;
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
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let reduced = gpu(backend, self.gpu_handle()?)?;
        // The reduced handle is a 1-element Bool buffer. Read back the single
        // scalar byte (allowed: it is the result, like has_inf_nan's flag).
        let bytes = backend.gpu_to_cpu(&reduced)?;
        Ok(bytes.first().is_some_and(|&b| b != 0))
    }

    // ── Float comparison constructors (#615; GPU path #1185 Phase 3b) ────────
    //
    // Each compares two float `Tensor<T>` on the same device, broadcasting
    // compatible shapes to their common shape per torch's comparison
    // semantics (CORE-106 / #1800). On CUDA it launches the value-typed
    // comparison PTX kernel (mismatched operands are first expanded ENTIRELY
    // on device — see `expand_float_handle_gpu`) and the resulting
    // `BoolTensor` stays GPU-resident; on CPU it runs the reference closure.

    /// Pointwise `>` comparing two float tensors (broadcasting compatible
    /// shapes); produces a `BoolTensor` of the common shape. Mirrors
    /// `torch.gt(a, b)` returning a bool tensor. (#615, broadcast #1800)
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

    /// Expand a CUDA value buffer from `in_shape` to `out_shape` ENTIRELY on
    /// device, by broadcast-adding a 1-element zeros buffer of the same
    /// dtype — the identical trick `grad_fns::shape::expand` uses for its
    /// f32/f64 GPU fast path, extended over every comparison value dtype
    /// with an implemented `broadcast_add_*` kernel (f32/f64/bf16/f16).
    /// `x + 0` is exact for comparison purposes: every finite value is
    /// unchanged, NaN stays NaN (compares false either way), and `-0.0`
    /// normalising to `+0.0` is invisible to every comparison operator
    /// (`-0.0 == 0.0` in IEEE 754). No host round trip (R-CODE-4).
    /// (CORE-106 / #1800)
    fn expand_float_handle_gpu(
        backend: &dyn crate::gpu_dispatch::GpuBackend,
        h: &GpuBufferHandle,
        in_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        use crate::dtype::DType;
        let zeros = backend.alloc_zeros(1, h.dtype(), h.device_ordinal())?;
        // 0-d inputs (shape []) broadcast like [1] under right alignment.
        let in_shape: &[usize] = if in_shape.is_empty() { &[1] } else { in_shape };
        match h.dtype() {
            DType::F32 => backend.broadcast_add_f32(h, &zeros, in_shape, &[1], out_shape),
            DType::F64 => backend.broadcast_add_f64(h, &zeros, in_shape, &[1], out_shape),
            DType::BF16 => backend.broadcast_add_bf16(h, &zeros, in_shape, &[1], out_shape),
            DType::F16 => backend.broadcast_add_f16(h, &zeros, in_shape, &[1], out_shape),
            other => Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "BoolTensor comparison: no GPU broadcast-expand path for \
                     value dtype {other:?}"
                ),
            }),
        }
    }

    fn compare_float<T: Float>(
        a: &Tensor<T>,
        b: &Tensor<T>,
        op: CompareOp,
        f: impl Fn(T, T) -> bool,
    ) -> FerrotorchResult<Self> {
        if a.device() != b.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: a.device(),
                got: b.device(),
            });
        }
        // Broadcast to the common shape (CORE-106 / #1800); incompatible
        // shapes get broadcast_shapes' structured ShapeMismatch.
        let common = crate::shape::broadcast_shapes(a.shape(), b.shape())?;
        if a.is_cuda() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            // #1660: normalise BOTH narrowed-offset CUDA operands to packed
            // offset-0 buffers before the value-typed comparison kernel reads
            // element 0 (#1658 class). A row-narrowed view's BASE buffer is
            // longer than `numel`, which the kernel rejects ("buffer length
            // mismatch: 8 vs 6"); `.contiguous()` materialises the logical view
            // on-device (strided_copy; cheap clone when already offset-0). The
            // BoolTensor result stays GPU-resident.
            let a = a.contiguous()?;
            let b = b.contiguous()?;
            // Expand mismatched operands on-device, then run the comparison
            // kernel on the (now equal-length) buffers. No host crossing.
            let a_expanded;
            let a_handle = if a.shape() == common.as_slice() {
                a.gpu_handle()?
            } else {
                a_expanded =
                    Self::expand_float_handle_gpu(backend, a.gpu_handle()?, a.shape(), &common)?;
                &a_expanded
            };
            let b_expanded;
            let b_handle = if b.shape() == common.as_slice() {
                b.gpu_handle()?
            } else {
                b_expanded =
                    Self::expand_float_handle_gpu(backend, b.gpu_handle()?, b.shape(), &common)?;
                &b_expanded
            };
            let h = backend.compare(a_handle, b_handle, op)?;
            return Self::from_gpu_handle(h, common);
        }
        let a_data = a.data_vec()?;
        let b_data = b.data_vec()?;
        if a.shape() == b.shape() {
            // Fast path: equal shapes, plain zip.
            let result: Vec<bool> = a_data
                .iter()
                .zip(b_data.iter())
                .map(|(&x, &y)| f(x, y))
                .collect();
            return Self::from_vec(result, common);
        }
        let numel: usize = if common.is_empty() {
            1
        } else {
            crate::shape::numel(&common)
        };
        let result: Vec<bool> = (0..numel)
            .map(|i| {
                f(
                    a_data[broadcast_src_flat(i, &common, a.shape())],
                    b_data[broadcast_src_flat(i, &common, b.shape())],
                )
            })
            .collect();
        Self::from_vec(result, common)
    }

    // ── Integer comparison constructors (#1185 Phase 3b) ─────────────────────
    //
    // Parallel to the float constructors, taking `&IntTensor<I>`, with the
    // same broadcasting semantics (CORE-106 / #1800). On CUDA, same-shape
    // operands launch the i32/i64 comparison kernel and broadcasted operands
    // launch the rank-general comparison-broadcast kernel; the bool mask stays
    // GPU-resident in both cases.

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

    /// Shared integer comparison dispatch. Broadcasts compatible shapes to
    /// their common shape (CORE-106 / #1800).
    ///
    /// # Device behavior
    ///
    /// CUDA operands run entirely on-device. Equal-shape operands use the
    /// ordinary comparison kernel; broadcast-compatible differing shapes use
    /// the rank-general integer broadcast-comparison kernel. The result is a
    /// CUDA-resident `BoolTensor`, matching PyTorch's comparison TensorIterator
    /// behavior without CPU fallback or value round trips.
    fn compare_int<I: crate::int_tensor::IntElement>(
        a: &crate::int_tensor::IntTensor<I>,
        b: &crate::int_tensor::IntTensor<I>,
        op: CompareOp,
        f: impl Fn(i64, i64) -> bool,
    ) -> FerrotorchResult<Self> {
        if a.device() != b.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: a.device(),
                got: b.device(),
            });
        }
        // Broadcast to the common shape (CORE-106 / #1800); incompatible
        // shapes get broadcast_shapes' structured ShapeMismatch.
        let common = crate::shape::broadcast_shapes(a.shape(), b.shape())?;
        if a.is_cuda() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let h = if a.shape() == b.shape() {
                backend.compare(a.gpu_handle()?, b.gpu_handle()?, op)?
            } else {
                backend.compare_broadcast(
                    a.gpu_handle()?,
                    b.gpu_handle()?,
                    a.shape(),
                    b.shape(),
                    &common,
                    op,
                )?
            };
            return Self::from_gpu_handle(h, common);
        }
        let a_data = a.data()?;
        let b_data = b.data()?;
        if a.shape() == b.shape() {
            // Fast path: equal shapes, plain zip.
            let result: Vec<bool> = a_data
                .iter()
                .zip(b_data.iter())
                .map(|(&x, &y)| f(x.to_i64(), y.to_i64()))
                .collect();
            return Self::from_vec(result, common);
        }
        let numel: usize = if common.is_empty() {
            1
        } else {
            crate::shape::numel(&common)
        };
        let result: Vec<bool> = (0..numel)
            .map(|i| {
                f(
                    a_data[broadcast_src_flat(i, &common, a.shape())].to_i64(),
                    b_data[broadcast_src_flat(i, &common, b.shape())].to_i64(),
                )
            })
            .collect();
        Self::from_vec(result, common)
    }

    /// Convert to a float tensor: true → 1.0, false → 0.0.
    ///
    /// On CUDA the cast runs on the GPU and the result stays resident (no host
    /// round-trip); on CPU it maps the host slice.
    pub fn to_float<T: Float>(&self) -> FerrotorchResult<Tensor<T>> {
        if self.is_cuda() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let h = backend
                .cast_bool_to_f(self.gpu_handle()?, <T as crate::dtype::Element>::dtype())?;
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

/// Map an output flat index to the source flat index of an operand being
/// broadcast from `in_shape` to `out_shape` (right-aligned NumPy/torch
/// rules: a size-1 or absent input axis replicates). `in_shape == []` (0-d)
/// always maps to source index 0. (CORE-106 / #1800)
fn broadcast_src_flat(mut out_flat: usize, out_shape: &[usize], in_shape: &[usize]) -> usize {
    let out_ndim = out_shape.len();
    let in_ndim = in_shape.len();
    let mut src = 0usize;
    let mut stride = 1usize;
    for i in (0..out_ndim).rev() {
        let dim = out_shape[i];
        // dim == 0 only for empty tensors (numel 0 — the loop body never
        // runs for any real element); checked_* keeps clippy's
        // manual-checked-division lint and the panic-freedom guarantee.
        let coord = out_flat.checked_rem(dim).unwrap_or(0);
        out_flat = out_flat.checked_div(dim).unwrap_or(0);
        // Right-aligned: out axis i pairs with in axis i - (out_ndim - in_ndim).
        if i + in_ndim >= out_ndim {
            let in_dim = in_shape[i + in_ndim - out_ndim];
            if in_dim != 1 {
                src += coord * stride;
            }
            stride *= in_dim;
        }
    }
    src
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
        assert_eq!(
            a.and(&b).unwrap().data().unwrap(),
            &[true, false, false, false]
        );
        assert_eq!(
            a.or(&b).unwrap().data().unwrap(),
            &[true, true, true, false]
        );
        assert_eq!(
            a.xor(&b).unwrap().data().unwrap(),
            &[false, true, true, false]
        );
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
        assert_eq!(
            BoolTensor::ge(&a, &b).unwrap().data().unwrap(),
            &[true, false, true]
        );
        assert_eq!(
            BoolTensor::le(&a, &b).unwrap().data().unwrap(),
            &[true, true, false]
        );
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
