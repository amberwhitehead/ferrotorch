//! ## REQ status (per `.design/ferrotorch-core/device.md`)
//!
//! Tensor location enum mirroring `c10::Device` (`c10/core/Device.h:31`).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (Cpu variant) | SHIPPED | variant `Device::Cpu` at `device.rs:15` with `#[default]`; consumer `storage.rs` `TensorStorage::cpu(...).device() == Device::Cpu`; also `bool_tensor.rs:152` returns `Cpu` for any `TensorStorage::cpu`-backed `BoolTensor` |
//! | REQ-2 (Cuda variant) | SHIPPED | variant `Device::Cuda(usize)` at `device.rs:18`; consumer `int_tensor.rs:268-323` `IntTensor::to` matches `(Cpu, Cuda(_))` / `(Cuda(_), Cpu)` arms for H2D / D2H transfer |
//! | REQ-3 (Xpu variant) | SHIPPED | variant `Device::Xpu(usize)` at `device.rs:22`; consumer `error.rs:259` `FerrotorchError::DeviceMismatch { expected, got }` carries Xpu values; `int_tensor.rs:336` rejects `Xpu` destination via structured error |
//! | REQ-4 (Mps variant) | SHIPPED | variant `Device::Mps(usize)` at `device.rs:26`; consumer `bool_tensor.rs:261-266` `(from, to) => Err(InvalidArgument)` arm pattern-matches on `Mps(_)` |
//! | REQ-5 (Meta variant) | SHIPPED | variant `Device::Meta` at `device.rs:31`; consumer `storage.rs` `TensorStorage::Meta` arm — `try_as_slice` returns `GpuTensorNotAccessible` for Meta variant |
//! | REQ-6 (predicates) | SHIPPED | `is_cpu` / `is_cuda` / `is_xpu` / `is_mps` / `is_meta` at `device.rs:36-64`; consumer `bool_tensor.rs:158`, `int_tensor.rs:205`, every `if a.device().is_cuda()` branch across `grad_fns/*.rs` |
//! | REQ-7 (Display) | SHIPPED | `Display` impl at `device.rs:66-76` matching `c10::Device::str()` (`c10/core/Device.h:167`); consumer `error.rs:11` `#[error("device mismatch: expected {expected}, got {got}")]` |
//! | REQ-8 (Copy/Hash derives) | SHIPPED | `#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]` at `device.rs:12`; consumer `gpu_dispatch.rs` registry + `Tensor<T>::device() == other.device()` PartialEq compares in `bool_tensor.rs:333`, `int_tensor.rs:436` |

/// Device on which a tensor's data resides.
///
/// `Meta` is a special device that does not allocate any backing memory:
/// meta tensors carry shape, dtype, and device information but no data.
/// They are useful for shape inference, dry-run model construction, and
/// inspecting parameter counts of huge models without actually allocating
/// the weights. Mirrors `torch.device("meta")`.
///
/// `Xpu` mirrors PyTorch's `torch.device("xpu")` and addresses Intel
/// GPUs (Arc series, Data Center GPU Max) via the portable CubeCL
/// wgpu runtime that the `ferrotorch-xpu` crate wraps. CL-452.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU main memory.
    #[default]
    Cpu,
    /// CUDA GPU with the given device index.
    Cuda(usize),
    /// Intel XPU (Arc / Data Center GPU Max) with the given device index.
    /// Accessed via `ferrotorch-xpu` which wraps a CubeCL wgpu runtime.
    /// CL-452.
    Xpu(usize),
    /// Apple Silicon Metal Performance Shaders. The `usize` is the Metal
    /// device index (`0` is the system default GPU). Mirrors
    /// `torch.device("mps")`. Implemented via `ferrotorch-mps`. (#451)
    Mps(usize),
    /// Meta device — shape-only, no backing storage. Operations that need
    /// data return an error; operations that only manipulate metadata
    /// (reshape, view, permute, narrow, transpose, …) work normally and
    /// produce meta tensors as output. CL-395.
    Meta,
}

impl Device {
    /// Returns `true` if this is a CPU device.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Returns `true` if this is a CUDA device.
    #[inline]
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }

    /// Returns `true` if this is an Intel XPU device. CL-452.
    #[inline]
    pub fn is_xpu(&self) -> bool {
        matches!(self, Device::Xpu(_))
    }

    /// Returns `true` if this is an Apple MPS device. (#451)
    #[inline]
    pub fn is_mps(&self) -> bool {
        matches!(self, Device::Mps(_))
    }

    /// Returns `true` if this is the meta device (shape-only, no data).
    #[inline]
    pub fn is_meta(&self) -> bool {
        matches!(self, Device::Meta)
    }
}

impl core::fmt::Display for Device {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(id) => write!(f, "cuda:{id}"),
            Device::Xpu(id) => write!(f, "xpu:{id}"),
            Device::Mps(id) => write!(f, "mps:{id}"),
            Device::Meta => write!(f, "meta"),
        }
    }
}
