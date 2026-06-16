//! Module doc — REQ status table follows.
//!
//! ## REQ status (per `.design/ferrotorch-core/creation.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `zeros`/`ones`/`full` at `creation.rs:7,14,21`; consumer: `stride_tricks::AsStridedBackward::backward` at `stride_tricks.rs:366` |
//! | REQ-2 | SHIPPED | `from_slice`/`from_vec`/`tensor`/`scalar` at `creation.rs:28-46`; consumer: `flex_attention` at `flex_attention.rs:183`, `einops::reduce` at `einops.rs:790` |
//! | REQ-3 | SHIPPED | `eye` at `creation.rs:49`; consumer: re-export at `lib.rs:138` |
//! | REQ-4 | SHIPPED | `arange` at `creation.rs:58`; consumer: re-export at `lib.rs:138` |
//! | REQ-5 | SHIPPED | `linspace` at `creation.rs:81`; consumer: re-export at `lib.rs:138` |
//! | REQ-6 | SHIPPED | `rand`/`randn` at `creation.rs:112,145`; consumer: `autograd::grad_penalty::grad_penalty` at `grad_penalty.rs:81`. Prereq blocker #1537 tracks `torch.manual_seed` |
//! | REQ-7 | SHIPPED | `*_like` family at `creation.rs:288-314`; consumer: `grad_fns::cumulative` at `cumulative.rs:501` |
//! | REQ-8 | SHIPPED | `zeros_meta`/`ones_meta`/`full_meta`/`meta_like` at `creation.rs:253-289`; consumer: `tensor::Tensor::meta_fill_value` at `tensor.rs:1078` (CL-395) |

use crate::device::Device;
use crate::dtype::{DType, Element, Float};
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::rng::with_thread_rng;
use crate::shape::{checked_byte_count, checked_numel};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// Create a tensor filled with zeros.
pub fn zeros<T: Float>(shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    let numel = checked_numel(shape, "zeros")?;
    let data = vec![<T as num_traits::Zero>::zero(); numel];
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
}

/// Create a tensor filled with ones.
pub fn ones<T: Float>(shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    let numel = checked_numel(shape, "ones")?;
    let data = vec![<T as num_traits::One>::one(); numel];
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
}

/// Create a tensor filled with a given value.
pub fn full<T: Float>(shape: &[usize], value: T) -> FerrotorchResult<Tensor<T>> {
    let numel = checked_numel(shape, "full")?;
    let data = vec![value; numel];
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
}

pub(crate) fn full_on_device<T: Float>(
    shape: &[usize],
    value: T,
    device: Device,
    op: &'static str,
) -> FerrotorchResult<Tensor<T>> {
    let numel = checked_numel(shape, op)?;
    let storage = match device {
        Device::Cpu => TensorStorage::cpu(vec![value; numel]),
        Device::Meta => TensorStorage::meta_filled(numel, value),
        Device::Cuda(ordinal) => {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let handle = match T::dtype() {
                DType::F32 => backend.fill_f32(
                    numel,
                    num_traits::ToPrimitive::to_f32(&value).ok_or_else(|| {
                        FerrotorchError::InvalidArgument {
                            message: format!("{op}: fill value is not representable as f32"),
                        }
                    })?,
                    ordinal,
                )?,
                DType::F64 => backend.fill_f64(
                    numel,
                    num_traits::ToPrimitive::to_f64(&value).ok_or_else(|| {
                        FerrotorchError::InvalidArgument {
                            message: format!("{op}: fill value is not representable as f64"),
                        }
                    })?,
                    ordinal,
                )?,
                DType::BF16 => backend.fill_bf16_bf16(
                    numel,
                    num_traits::ToPrimitive::to_f32(&value).ok_or_else(|| {
                        FerrotorchError::InvalidArgument {
                            message: format!("{op}: fill value is not representable as f32"),
                        }
                    })?,
                    ordinal,
                )?,
                DType::F16 => backend.fill_f16(
                    numel,
                    num_traits::ToPrimitive::to_f32(&value).ok_or_else(|| {
                        FerrotorchError::InvalidArgument {
                            message: format!("{op}: fill value is not representable as f32"),
                        }
                    })?,
                    ordinal,
                )?,
                dtype => {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("{op}: unsupported floating dtype {dtype}"),
                    });
                }
            };
            TensorStorage::gpu(handle)
        }
        Device::Xpu(_) | Device::Mps(_) => {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "{op}: device-native tensor creation is not wired for {device}; \
                     refusing to return a CPU fallback"
                ),
            });
        }
    };
    Tensor::from_storage(storage, shape.to_vec(), false)
}

/// Create a tensor from a slice, copying the data.
pub fn from_slice<T: Float>(data: &[T], shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
}

/// Create a tensor from a `Vec<T>`, taking ownership.
pub fn from_vec<T: Float>(data: Vec<T>, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
}

/// Create a 1-D tensor from a slice (shape inferred).
pub fn tensor<T: Float>(data: &[T]) -> FerrotorchResult<Tensor<T>> {
    let shape = vec![data.len()];
    from_slice(data, &shape)
}

/// Create a scalar (0-D) tensor.
pub fn scalar<T: Float>(value: T) -> FerrotorchResult<Tensor<T>> {
    Tensor::from_storage(TensorStorage::cpu(vec![value]), vec![], false)
}

/// Create an identity matrix of size `n x n`.
pub fn eye<T: Float>(n: usize) -> FerrotorchResult<Tensor<T>> {
    let numel = n
        .checked_mul(n)
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("eye: shape [{n}, {n}] element count overflows usize"),
        })?;
    let mut data = vec![<T as num_traits::Zero>::zero(); numel];
    for i in 0..n {
        data[i * n + i] = <T as num_traits::One>::one();
    }
    Tensor::from_storage(TensorStorage::cpu(data), vec![n, n], false)
}

fn arange_arg<T: Float>(value: T) -> String {
    value.to_f64().map_or_else(
        || "<not representable as f64>".to_string(),
        |v| v.to_string(),
    )
}

fn arange_len<T: Float>(start: T, end: T, step: T) -> FerrotorchResult<usize> {
    let zero = <T as num_traits::Zero>::zero();

    if !start.is_finite() || !end.is_finite() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "arange: unsupported range: {} -> {}",
                arange_arg(start),
                arange_arg(end)
            ),
        });
    }
    if step == zero || step.is_nan() {
        return Err(FerrotorchError::InvalidArgument {
            message: "arange: step must be nonzero".into(),
        });
    }
    if start == end {
        return Ok(0);
    }

    let ascending = start < end;
    if (ascending && step < zero) || (!ascending && step > zero) {
        return Err(FerrotorchError::InvalidArgument {
            message: "arange: upper bound and lower bound inconsistent with step sign".into(),
        });
    }
    if step.is_infinite() {
        return Ok(0);
    }

    let start_f = start
        .to_f64()
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "arange: start is not representable as f64".into(),
        })?;
    let end_f = end
        .to_f64()
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "arange: end is not representable as f64".into(),
        })?;
    let step_f = step
        .to_f64()
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "arange: step is not representable as f64".into(),
        })?;

    let len_f = ((end_f - start_f) / step_f).ceil();
    if !len_f.is_finite() || len_f < 0.0 || len_f >= usize::MAX as f64 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "arange: length is not representable for range {} -> {} with step {}",
                arange_arg(start),
                arange_arg(end),
                arange_arg(step)
            ),
        });
    }

    let len = len_f as usize;
    let bytes = checked_byte_count(len, std::mem::size_of::<T>(), "arange")?;
    if isize::try_from(bytes).is_err() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "arange: storage size calculation overflowed for {len} elements of {} bytes",
                std::mem::size_of::<T>()
            ),
        });
    }
    Ok(len)
}

/// Create a 1-D tensor with values from `start` to `end` (exclusive) with step `step`.
pub fn arange<T: Float>(start: T, end: T, step: T) -> FerrotorchResult<Tensor<T>> {
    let len = arange_len(start, end, step)?;
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|err| FerrotorchError::InvalidArgument {
            message: format!("arange: could not allocate {len} elements: {err}"),
        })?;
    for i in 0..len {
        let idx = T::from(i).ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("arange: index {i} is not representable in output dtype"),
        })?;
        data.push(start + step * idx);
    }
    Tensor::from_storage(TensorStorage::cpu(data), vec![len], false)
}

/// Create a 1-D tensor of `num` evenly spaced values from `start` to `end` (inclusive).
pub fn linspace<T: Float>(start: T, end: T, num: usize) -> FerrotorchResult<Tensor<T>> {
    if num == 0 {
        return Tensor::from_storage(TensorStorage::cpu(vec![]), vec![0], false);
    }
    if num == 1 {
        return Tensor::from_storage(TensorStorage::cpu(vec![start]), vec![1], false);
    }
    let n = T::from(num - 1).unwrap();
    let step = (end - start) / n;
    let data: Vec<T> = (0..num)
        .map(|i| start + step * T::from(i).unwrap())
        .collect();
    Tensor::from_storage(TensorStorage::cpu(data), vec![num], false)
}

/// Create a tensor with random values uniformly distributed in [0, 1).
///
/// Uses the process-global default [`crate::rng::Generator`] (MT19937),
/// mirroring PyTorch CPU's `default_generator` / `at::CPUGeneratorImpl` in
/// `aten/src/ATen/CPUGeneratorImpl.cpp:52-57,226-228`. Call
/// [`crate::manual_seed`] for reproducible output.
///
/// # Byte-exact parity vs `torch.rand`
///
/// After `ferrotorch_core::manual_seed(s)`, this function consumes the same
/// MT19937 bit stream as `torch.manual_seed(s); torch.rand(...)`. For f32
/// element type the values are byte-identical
/// (`aten/src/ATen/core/DistributionsHelper.h:106-113` `uniform_real<float>`
/// transform: `(random() & ((1<<24)-1)) * (1.0 / (1<<24))`).
///
/// # Threading
///
/// CPU random creation serializes access to one process-global default
/// generator, like PyTorch's `GeneratorImpl::mutex_` convention. A
/// `manual_seed` call in any thread resets the stream subsequently consumed by
/// all threads.
pub fn rand<T: Float>(shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    let numel = checked_numel(shape, "rand")?;
    let mut data: Vec<T> = Vec::with_capacity(numel);

    let is_f32 = std::mem::size_of::<T>() == 4;
    with_thread_rng(|g| {
        if is_f32 {
            for _ in 0..numel {
                data.push(T::from(g.next_uniform_f32()).unwrap());
            }
        } else {
            for _ in 0..numel {
                data.push(T::from(g.next_uniform_f64()).unwrap());
            }
        }
    });

    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
}

/// Create a tensor with random values from a standard normal distribution.
///
/// Uses the process-global default [`crate::rng::Generator`] (MT19937 +
/// Box-Muller), mirroring `at::normal_distribution<T>(0, 1)` at
/// `aten/src/ATen/core/DistributionsHelper.h:172-201`. Call
/// [`crate::manual_seed`] for reproducible output.
///
/// The Box-Muller pair (`r * cos(theta)`, `r * sin(theta)`) order matches
/// torch CPU: `cos` is returned, `sin` is cached for the next call.
pub fn randn<T: Float>(shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    let numel = checked_numel(shape, "randn")?;
    let mut data: Vec<T> = Vec::with_capacity(numel);

    let is_f32 = std::mem::size_of::<T>() == 4;
    with_thread_rng(|g| {
        if is_f32 {
            for _ in 0..numel {
                data.push(T::from(g.next_normal_f32()).unwrap());
            }
        } else {
            for _ in 0..numel {
                data.push(T::from(g.next_normal_f64()).unwrap());
            }
        }
    });

    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
}

/// Device-aware uniform-`[0, 1)` random tensor creation.
///
/// PyTorch parity: `torch.rand(size, device=...)` lowers to
/// `at::empty(size, options).uniform_(0, 1)`
/// (`aten/src/ATen/native/TensorFactories.cpp:1075-1076`). The tensor is
/// created ON the requested device and filled in place — for a CUDA device
/// the fill runs as an on-device curand/Philox kernel, with NO CPU
/// generate-then-upload.
///
/// # Behaviour by device / dtype
///
/// - `Device::Cuda(_)` + floating dtypes: generated through the dtype-specific
///   [`crate::gpu_dispatch::GpuBackend`] RNG slot and returned as an
///   `is_cuda()` tensor with no host round trip (R-CODE-4). Reproducible after
///   [`crate::manual_seed`] (which seeds the GPU generator too).
/// - `Device::Cpu`: identical to [`rand`] (byte-exact with `torch.rand` for
///   f32 via the process-global MT19937 default generator).
/// - `Device::Meta`: falls back to the CPU path then `.to(Meta)`.
pub fn rand_on_device<T: Float>(
    shape: &[usize],
    device: crate::device::Device,
) -> FerrotorchResult<Tensor<T>> {
    use crate::device::Device;

    match device {
        Device::Cuda(_) => {
            let numel = checked_numel(shape, "rand_on_device")?;
            let dtype = <T as Element>::dtype();
            checked_byte_count(numel, dtype.size_of(), "rand_on_device")?;
            let backend = crate::gpu_dispatch::gpu_backend()
                .ok_or(crate::error::FerrotorchError::DeviceUnavailable)?;
            let handle = match dtype {
                DType::F32 => backend.rand_uniform_f32(numel)?,
                DType::F64 => backend.rand_uniform_f64(numel)?,
                DType::F16 => backend.rand_uniform_f16(numel)?,
                DType::BF16 => backend.rand_uniform_bf16(numel)?,
                dtype => {
                    return Err(crate::error::FerrotorchError::InvalidArgument {
                        message: format!("rand_on_device: unsupported floating dtype {dtype}"),
                    });
                }
            };
            let storage = TensorStorage::gpu(handle);
            Tensor::from_storage(storage, shape.to_vec(), false)
        }
        Device::Cpu => rand::<T>(shape),
        other => rand::<T>(shape)?.to(other),
    }
}

/// Device-aware standard-normal random tensor creation.
///
/// Standard-normal counterpart of [`rand_on_device`]. PyTorch parity:
/// `torch.randn(size, device=...)` = `at::empty(...).normal_(0, 1)`
/// (`aten/src/ATen/native/TensorFactories.cpp:1379`). For `Device::Cuda(_)` +
/// CUDA values are generated through dtype-specific backend RNG slots; CPU and
/// meta keep the existing CPU factory behaviour.
pub fn randn_on_device<T: Float>(
    shape: &[usize],
    device: crate::device::Device,
) -> FerrotorchResult<Tensor<T>> {
    use crate::device::Device;

    match device {
        Device::Cuda(_) => {
            let numel = checked_numel(shape, "randn_on_device")?;
            let dtype = <T as Element>::dtype();
            checked_byte_count(numel, dtype.size_of(), "randn_on_device")?;
            let backend = crate::gpu_dispatch::gpu_backend()
                .ok_or(crate::error::FerrotorchError::DeviceUnavailable)?;
            let handle = match dtype {
                DType::F32 => backend.randn_normal_f32(numel)?,
                DType::F64 => backend.randn_normal_f64(numel)?,
                DType::F16 => backend.randn_normal_f16(numel)?,
                DType::BF16 => backend.randn_normal_bf16(numel)?,
                dtype => {
                    return Err(crate::error::FerrotorchError::InvalidArgument {
                        message: format!("randn_on_device: unsupported floating dtype {dtype}"),
                    });
                }
            };
            let storage = TensorStorage::gpu(handle);
            Tensor::from_storage(storage, shape.to_vec(), false)
        }
        Device::Cpu => randn::<T>(shape),
        other => randn::<T>(shape)?.to(other),
    }
}

/// Create a meta (no-data) tensor with the given shape. Carries shape and
/// dtype information but allocates no backing memory. Useful for shape
/// inference, dry-run model construction, and inspecting parameter counts
/// of huge models without committing to allocation. CL-395.
pub fn zeros_meta<T: Float>(shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    let numel = checked_numel(shape, "zeros_meta")?;
    Tensor::from_storage(TensorStorage::meta(numel), shape.to_vec(), false)
}

/// Create a meta tensor with the given shape. Identical in behavior to
/// [`zeros_meta`] — meta tensors carry no data, so the value parameter
/// has no effect, but the function exists for API symmetry with the
/// regular [`ones`] / [`full`] constructors. CL-395.
pub fn ones_meta<T: Float>(shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    zeros_meta(shape)
}

/// Create a meta tensor with the given shape, recording `value` as the
/// fill that would have been materialised on a real device.
///
/// Meta tensors carry no element-wise data, so individual elements still
/// cannot be read. The recorded fill is available via
/// [`crate::storage::TensorStorage::meta_fill_value`] (and the
/// [`Tensor::meta_fill_value`] convenience wrapper) so that shape-inference
/// code can distinguish "uninitialised meta" (`zeros_meta`, `ones_meta`)
/// from "would-be filled meta" (`full_meta(shape, value)`) and so that the
/// constructor's `value` parameter is observable rather than silently
/// discarded. CL-395.
pub fn full_meta<T: Float>(shape: &[usize], value: T) -> FerrotorchResult<Tensor<T>> {
    let numel = checked_numel(shape, "full_meta")?;
    Tensor::from_storage(
        TensorStorage::meta_filled(numel, value),
        shape.to_vec(),
        false,
    )
}

/// Create a meta tensor matching the shape of `other`. Always allocates
/// on the meta device regardless of `other`'s device.
pub fn meta_like<T: Float>(other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    zeros_meta(other.shape())
}

/// Create a tensor of zeros with the same shape as `other`.
pub fn zeros_like<T: Float>(other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    full_on_device(
        other.shape(),
        <T as num_traits::Zero>::zero(),
        other.device(),
        "zeros_like",
    )
}

/// Create a tensor of ones with the same shape as `other`.
pub fn ones_like<T: Float>(other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    full_on_device(
        other.shape(),
        <T as num_traits::One>::one(),
        other.device(),
        "ones_like",
    )
}

/// Create a tensor filled with `value` with the same shape as `other`.
pub fn full_like<T: Float>(other: &Tensor<T>, value: T) -> FerrotorchResult<Tensor<T>> {
    full_on_device(other.shape(), value, other.device(), "full_like")
}

/// Create a random tensor [0,1) with the same shape as `other`.
pub fn rand_like<T: Float>(other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    rand_on_device(other.shape(), other.device())
}

/// Create a random normal tensor with the same shape as `other`.
pub fn randn_like<T: Float>(other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    randn_on_device(other.shape(), other.device())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t: Tensor<f32> = zeros(&[2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.data().unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    // reason: ones() writes the exact bit pattern of 1.0; sentinel-value
    // identity check (no arithmetic), so equality is the right predicate.
    #[allow(clippy::float_cmp)]
    fn test_ones() {
        let t: Tensor<f64> = ones(&[4]).unwrap();
        assert_eq!(t.shape(), &[4]);
        assert!(t.data().unwrap().iter().all(|&x| x == 1.0));
    }

    #[test]
    #[allow(clippy::approx_constant)] // 3.14 is an arbitrary test fill value, not π.
    fn test_full() {
        let t: Tensor<f32> = full(&[2, 2], 3.14).unwrap();
        assert!(t.data().unwrap().iter().all(|&x| (x - 3.14).abs() < 1e-6));
    }

    #[test]
    fn test_from_slice() {
        let t: Tensor<f32> = from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_1d() {
        let t = tensor(&[1.0f32, 2.0, 3.0]).unwrap();
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    // reason: scalar(42.0) stores exactly 42.0; round-trip read returns
    // the same bit pattern (no arithmetic), so equality is the right check.
    #[allow(clippy::float_cmp)]
    fn test_scalar() {
        let t = scalar(42.0f64).unwrap();
        assert!(t.is_scalar());
        assert_eq!(t.item().unwrap(), 42.0);
    }

    #[test]
    // reason: identity-matrix sentinel — eye() fills exact 1.0 on the
    // diagonal and exact 0.0 elsewhere (no arithmetic), so bit-equality
    // is the right check.
    #[allow(clippy::float_cmp)]
    fn test_eye() {
        let t: Tensor<f32> = eye(3).unwrap();
        assert_eq!(t.shape(), &[3, 3]);
        let d = t.data().unwrap();
        assert_eq!(d[0], 1.0); // [0,0]
        assert_eq!(d[1], 0.0); // [0,1]
        assert_eq!(d[4], 1.0); // [1,1]
        assert_eq!(d[8], 1.0); // [2,2]
    }

    #[test]
    fn test_arange() {
        let t: Tensor<f32> = arange(0.0, 5.0, 1.0).unwrap();
        assert_eq!(t.shape(), &[5]);
        let d = t.data().unwrap();
        assert_eq!(d, &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_arange_step() {
        let t: Tensor<f64> = arange(0.0, 1.0, 0.25).unwrap();
        assert_eq!(t.shape(), &[4]);
    }

    #[test]
    fn test_arange_negative_step() {
        let t: Tensor<f32> = arange(5.0, 1.0, -1.5).unwrap();
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.data().unwrap(), &[5.0, 3.5, 2.0]);
    }

    #[test]
    fn test_arange_wrong_step_sign_errors() {
        let forward: Result<Tensor<f32>, _> = arange(5.0, 1.0, 1.0);
        let backward: Result<Tensor<f32>, _> = arange(1.0, 5.0, -1.0);
        for result in [forward, backward] {
            let err = result.expect_err("wrong-sign arange must error");
            assert!(
                err.to_string()
                    .contains("upper bound and lower bound inconsistent with step sign"),
                "{err}"
            );
        }
    }

    #[test]
    fn test_arange_zero_step() {
        let result: Result<Tensor<f32>, _> = arange(0.0, 5.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_arange_nonfinite_bounds_error() {
        let result: Result<Tensor<f32>, _> = arange(0.0, f32::INFINITY, 1.0);
        let err = result.expect_err("infinite end must error");
        assert!(err.to_string().contains("unsupported range"), "{err}");

        let result: Result<Tensor<f32>, _> = arange(f32::NAN, 5.0, 1.0);
        let err = result.expect_err("NaN start must error");
        assert!(err.to_string().contains("unsupported range"), "{err}");
    }

    #[test]
    fn test_arange_nan_step_errors() {
        let result: Result<Tensor<f32>, _> = arange(0.0, 5.0, f32::NAN);
        let err = result.expect_err("NaN step must error");
        assert!(err.to_string().contains("step must be nonzero"), "{err}");
    }

    #[test]
    fn test_arange_infinite_step_matching_sign_is_empty() {
        let forward: Tensor<f32> = arange(0.0, 5.0, f32::INFINITY).unwrap();
        let backward: Tensor<f32> = arange(5.0, 0.0, f32::NEG_INFINITY).unwrap();
        assert_eq!(forward.shape(), &[0]);
        assert_eq!(backward.shape(), &[0]);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_arange_f32_sub_ulp_step_uses_index_generation() {
        let t: Tensor<f32> = arange(16_777_216.0, 16_777_220.0, 1.0).unwrap();
        assert_eq!(t.shape(), &[4]);
        assert_eq!(
            t.data().unwrap(),
            &[16_777_216.0, 16_777_216.0, 16_777_218.0, 16_777_220.0]
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_arange_f16_sub_ulp_step_uses_index_generation() {
        let t: Tensor<half::f16> = arange(
            half::f16::from_f32(2048.0),
            half::f16::from_f32(2052.0),
            half::f16::from_f32(0.5),
        )
        .unwrap();
        let data: Vec<f32> = t.data().unwrap().iter().map(|v| v.to_f32()).collect();
        assert_eq!(
            data,
            vec![
                2048.0, 2048.0, 2048.0, 2050.0, 2050.0, 2050.0, 2052.0, 2052.0
            ]
        );
    }

    #[test]
    fn test_arange_rejects_unrepresentable_length_without_allocation() {
        let err = arange_len(0.0f64, f64::MAX, f64::MIN_POSITIVE)
            .expect_err("unrepresentable length must error before allocation");
        assert!(
            err.to_string().contains("length is not representable"),
            "{err}"
        );
    }

    #[test]
    fn test_linspace() {
        let t: Tensor<f32> = linspace(0.0, 1.0, 5).unwrap();
        assert_eq!(t.shape(), &[5]);
        let d = t.data().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-6);
        assert!((d[2] - 0.5).abs() < 1e-6);
        assert!((d[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    // reason: linspace(start, start, 1) is a single-point degenerate case
    // that returns exactly `start` — no arithmetic happens, so equality
    // is the right check.
    #[allow(clippy::float_cmp)]
    fn test_linspace_single() {
        let t: Tensor<f32> = linspace(3.0, 3.0, 1).unwrap();
        assert_eq!(t.shape(), &[1]);
        assert_eq!(t.item().unwrap(), 3.0);
    }

    #[test]
    fn test_linspace_empty() {
        let t: Tensor<f32> = linspace(0.0, 1.0, 0).unwrap();
        assert_eq!(t.shape(), &[0]);
    }

    #[test]
    fn test_rand_shape() {
        let _guard = crate::rng::default_rng_test_lock();
        let t: Tensor<f32> = rand(&[10, 20]).unwrap();
        assert_eq!(t.shape(), &[10, 20]);
        // Values should be in [0, 1).
        assert!(t.data().unwrap().iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_randn_shape() {
        let _guard = crate::rng::default_rng_test_lock();
        let t: Tensor<f32> = randn(&[100]).unwrap();
        assert_eq!(t.shape(), &[100]);
        // Mean should be roughly 0 for 100 samples.
        let mean: f32 = t.data().unwrap().iter().sum::<f32>() / 100.0;
        assert!(mean.abs() < 1.0); // Very loose check.
    }

    #[test]
    fn test_zeros_empty() {
        let t: Tensor<f32> = zeros(&[0, 3]).unwrap();
        assert_eq!(t.shape(), &[0, 3]);
        assert_eq!(t.numel(), 0);
    }

    #[test]
    fn test_zeros_like() {
        let _guard = crate::rng::default_rng_test_lock();
        let t: Tensor<f32> = rand(&[3, 4]).unwrap();
        let z = zeros_like(&t).unwrap();
        assert_eq!(z.shape(), &[3, 4]);
        assert!(z.data().unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    // reason: ones_like writes the exact bit pattern of 1.0; sentinel-value
    // identity check (no arithmetic), so equality is the right predicate.
    #[allow(clippy::float_cmp)]
    fn test_ones_like() {
        let t: Tensor<f64> = zeros(&[2, 5]).unwrap();
        let o = ones_like(&t).unwrap();
        assert_eq!(o.shape(), &[2, 5]);
        assert!(o.data().unwrap().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_full_like() {
        let t: Tensor<f32> = zeros(&[4, 3]).unwrap();
        let f = full_like(&t, 7.0).unwrap();
        assert_eq!(f.shape(), &[4, 3]);
        assert!(f.data().unwrap().iter().all(|&x| (x - 7.0).abs() < 1e-6));
    }

    #[test]
    fn test_rand_like() {
        let _guard = crate::rng::default_rng_test_lock();
        let t: Tensor<f32> = zeros(&[5, 6]).unwrap();
        let r = rand_like(&t).unwrap();
        assert_eq!(r.shape(), &[5, 6]);
        assert!(r.data().unwrap().iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_randn_like() {
        let _guard = crate::rng::default_rng_test_lock();
        let t: Tensor<f32> = zeros(&[50]).unwrap();
        let r = randn_like(&t).unwrap();
        assert_eq!(r.shape(), &[50]);
    }

    // -----------------------------------------------------------------------
    // Meta device tests (CL-395)
    //
    // Meta tensors carry shape and dtype info but no backing memory.
    // The tests below verify that:
    //   1. Construction works for arbitrarily large shapes (no allocation)
    //   2. Metadata accessors return the right values
    //   3. Data access errors with a clear message
    //   4. Moving TO meta drops data; moving FROM meta errors
    // -----------------------------------------------------------------------

    #[test]
    fn test_zeros_meta_basic_shape() {
        let t: Tensor<f32> = zeros_meta(&[2, 3, 4]).unwrap();
        assert_eq!(t.shape(), &[2, 3, 4]);
        assert_eq!(t.numel(), 24);
        assert!(t.is_meta());
        assert_eq!(t.device(), crate::device::Device::Meta);
    }

    #[test]
    fn test_zeros_meta_huge_shape_no_allocation() {
        // 100M elements would be 400MB if allocated -- meta tensor must
        // not actually allocate, otherwise this test would either OOM
        // or take a long time on a memory-constrained machine.
        let t: Tensor<f32> = zeros_meta(&[10_000, 10_000]).unwrap();
        assert_eq!(t.shape(), &[10_000, 10_000]);
        assert_eq!(t.numel(), 100_000_000);
        assert!(t.is_meta());
    }

    #[test]
    fn test_meta_data_access_returns_clear_error() {
        let t: Tensor<f32> = zeros_meta(&[3]).unwrap();
        let err = t.data().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("meta tensor"),
            "expected meta-tensor error message, got: {msg}"
        );
    }

    #[test]
    fn test_meta_data_vec_returns_clear_error() {
        let t: Tensor<f32> = zeros_meta(&[3]).unwrap();
        let err = t.data_vec().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("meta tensor"));
    }

    #[test]
    fn test_meta_like_matches_shape() {
        let t: Tensor<f32> = zeros(&[7, 11]).unwrap();
        let m = meta_like(&t).unwrap();
        assert_eq!(m.shape(), t.shape());
        assert!(m.is_meta());
        assert!(!t.is_meta());
    }

    #[test]
    fn test_to_meta_from_cpu_drops_data() {
        let t: Tensor<f32> = zeros(&[5]).unwrap();
        let m = t.to(crate::device::Device::Meta).unwrap();
        assert!(m.is_meta());
        assert_eq!(m.shape(), &[5]);
        // Original is unchanged.
        assert!(!t.is_meta());
    }

    #[test]
    fn test_to_from_meta_errors() {
        let m: Tensor<f32> = zeros_meta(&[3]).unwrap();
        let result = m.to(crate::device::Device::Cpu);
        let err_msg = match result {
            Ok(_) => panic!("expected error moving from meta to CPU"),
            Err(e) => format!("{e}"),
        };
        assert!(err_msg.contains("meta tensor"));
    }

    #[test]
    fn test_meta_device_display() {
        let d = crate::device::Device::Meta;
        assert_eq!(format!("{d}"), "meta");
    }

    #[test]
    fn test_meta_device_clone_is_cheap() {
        let m: Tensor<f32> = zeros_meta(&[1024, 1024, 1024]).unwrap();
        // Cloning should not allocate (Arc share + new TensorInner).
        let c = m.clone();
        assert_eq!(c.shape(), m.shape());
        assert!(c.is_meta());
    }

    #[test]
    #[allow(clippy::approx_constant)] // 3.14 is an arbitrary test fill value, not π.
    fn test_meta_constructors_share_shape_and_meta_flag() {
        let z: Tensor<f64> = zeros_meta(&[2, 2]).unwrap();
        let o: Tensor<f64> = ones_meta(&[2, 2]).unwrap();
        let f: Tensor<f64> = full_meta(&[2, 2], 3.14).unwrap();
        // All three are meta tensors of the same shape.
        assert_eq!(z.shape(), o.shape());
        assert_eq!(z.shape(), f.shape());
        assert!(z.is_meta() && o.is_meta() && f.is_meta());
    }

    #[test]
    // reason: 2.5 and 0.0 are sentinel fill values; the test asserts
    // the exact recorded scalar round-trips, so equality is correct.
    #[allow(clippy::float_cmp)]
    fn test_full_meta_records_value_and_discriminates_by_fill() {
        // Discriminating fixture for the "_value silently ignored" audit:
        // two `full_meta` tensors of identical shape but different fills
        // MUST be distinguishable through the meta_fill_value() metadata.
        let a: Tensor<f64> = full_meta(&[2, 3], 2.5).unwrap();
        let b: Tensor<f64> = full_meta(&[2, 3], 0.0).unwrap();
        let z: Tensor<f64> = zeros_meta(&[2, 3]).unwrap();

        // All three are meta tensors of the same shape.
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(z.shape(), &[2, 3]);
        assert!(a.is_meta() && b.is_meta() && z.is_meta());

        // The fill parameter is observable, not silently discarded.
        assert_eq!(a.meta_fill_value(), Some(&2.5));
        assert_eq!(b.meta_fill_value(), Some(&0.0));

        // The two `full_meta` results discriminate on the recorded fill —
        // 2.5 vs 0.0 must produce different metadata.
        assert_ne!(a.meta_fill_value(), b.meta_fill_value());

        // Plain `zeros_meta` records no fill (distinguishable from
        // `full_meta(_, 0.0)` at the metadata layer).
        assert_eq!(z.meta_fill_value(), None);
        assert_ne!(z.meta_fill_value(), b.meta_fill_value());
    }
}
