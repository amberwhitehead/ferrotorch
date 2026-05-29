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

use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::rng::with_thread_rng;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// Create a tensor filled with zeros.
pub fn zeros<T: Float>(shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    let numel: usize = shape.iter().product();
    let data = vec![<T as num_traits::Zero>::zero(); numel];
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
}

/// Create a tensor filled with ones.
pub fn ones<T: Float>(shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    let numel: usize = shape.iter().product();
    let data = vec![<T as num_traits::One>::one(); numel];
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
}

/// Create a tensor filled with a given value.
pub fn full<T: Float>(shape: &[usize], value: T) -> FerrotorchResult<Tensor<T>> {
    let numel: usize = shape.iter().product();
    let data = vec![value; numel];
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
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
    let mut data = vec![<T as num_traits::Zero>::zero(); n * n];
    for i in 0..n {
        data[i * n + i] = <T as num_traits::One>::one();
    }
    Tensor::from_storage(TensorStorage::cpu(data), vec![n, n], false)
}

/// Create a 1-D tensor with values from `start` to `end` (exclusive) with step `step`.
pub fn arange<T: Float>(start: T, end: T, step: T) -> FerrotorchResult<Tensor<T>> {
    let mut data = Vec::new();
    let mut val = start;
    if step > <T as num_traits::Zero>::zero() {
        while val < end {
            data.push(val);
            val += step;
        }
    } else if step < <T as num_traits::Zero>::zero() {
        while val > end {
            data.push(val);
            val += step;
        }
    } else {
        return Err(crate::error::FerrotorchError::InvalidArgument {
            message: "arange: step cannot be zero".into(),
        });
    }
    let len = data.len();
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
/// Uses the thread-local [`crate::rng::Generator`] (MT19937), mirroring
/// PyTorch CPU's `at::CPUGeneratorImpl` in
/// `aten/src/ATen/CPUGeneratorImpl.cpp:226-228`. Call
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
/// # Thread-local RNG and rayon
///
/// The RNG state is a per-thread `RefCell<Generator>`. Each rayon worker
/// gets its own thread-local generator seeded from `SystemTime` + thread id
/// unless [`crate::manual_seed`] is called on that worker. The randn f32
/// parallel fast path (`numel >= 32_768`) derives per-chunk seeds from the
/// main thread's generator so the result is still deterministic given a
/// `manual_seed`.
pub fn rand<T: Float>(shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    let numel: usize = shape.iter().product();
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
/// Uses the thread-local [`crate::rng::Generator`] (MT19937 + Box-Muller),
/// mirroring `at::normal_distribution<T>(0, 1)` at
/// `aten/src/ATen/core/DistributionsHelper.h:172-201`. Call
/// [`crate::manual_seed`] for reproducible output.
///
/// The Box-Muller pair (`r * cos(theta)`, `r * sin(theta)`) order matches
/// torch CPU: `cos` is returned, `sin` is cached for the next call.
pub fn randn<T: Float>(shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    let numel: usize = shape.iter().product();
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
/// - `Device::Cuda(_)` + f32: generated entirely on-device via
///   [`crate::gpu_dispatch::GpuBackend::rand_uniform_f32`], wrapped as an
///   `is_cuda()` tensor with no host round trip (R-CODE-4). Reproducible after
///   [`crate::manual_seed`] (which seeds the GPU generator too).
/// - `Device::Cpu`: identical to [`rand`] (byte-exact with `torch.rand` for
///   f32 via the thread-local MT19937).
/// - `Device::Cuda(_)` + non-f32 (e.g. f64): the on-device Philox kernel is
///   f32-only, so this falls back to the CPU `rand` path then transfers to the
///   device via [`Tensor::to`]. The values are still correct; only the
///   generation site (CPU vs GPU) differs. f64 on-device RNG is a documented
///   follow-up.
/// - `Device::Meta`: falls back to the CPU path then `.to(Meta)`.
pub fn rand_on_device<T: Float>(
    shape: &[usize],
    device: crate::device::Device,
) -> FerrotorchResult<Tensor<T>> {
    use crate::device::Device;

    let is_f32 = std::mem::size_of::<T>() == 4;
    match device {
        Device::Cuda(_) if is_f32 => {
            let numel: usize = shape.iter().product();
            let backend = crate::gpu_dispatch::gpu_backend()
                .ok_or(crate::error::FerrotorchError::DeviceUnavailable)?;
            let handle = backend.rand_uniform_f32(numel)?;
            let storage = TensorStorage::gpu(handle);
            Tensor::from_storage(storage, shape.to_vec(), false)
        }
        Device::Cpu => rand::<T>(shape),
        // f64-on-CUDA and Meta: generate on CPU (byte-exact / correct
        // distribution) then move to the target device.
        other => rand::<T>(shape)?.to(other),
    }
}

/// Device-aware standard-normal random tensor creation.
///
/// Standard-normal counterpart of [`rand_on_device`]. PyTorch parity:
/// `torch.randn(size, device=...)` = `at::empty(...).normal_(0, 1)`
/// (`aten/src/ATen/native/TensorFactories.cpp:1379`). For `Device::Cuda(_)` +
/// f32 the values are generated on-device via the Box-Muller Philox normal
/// kernel ([`crate::gpu_dispatch::GpuBackend::randn_normal_f32`]); other
/// device/dtype combinations follow the same CPU-then-transfer fall-back as
/// [`rand_on_device`].
pub fn randn_on_device<T: Float>(
    shape: &[usize],
    device: crate::device::Device,
) -> FerrotorchResult<Tensor<T>> {
    use crate::device::Device;

    let is_f32 = std::mem::size_of::<T>() == 4;
    match device {
        Device::Cuda(_) if is_f32 => {
            let numel: usize = shape.iter().product();
            let backend = crate::gpu_dispatch::gpu_backend()
                .ok_or(crate::error::FerrotorchError::DeviceUnavailable)?;
            let handle = backend.randn_normal_f32(numel)?;
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
    let numel: usize = shape.iter().product();
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
    let numel: usize = shape.iter().product();
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
    zeros(other.shape())
}

/// Create a tensor of ones with the same shape as `other`.
pub fn ones_like<T: Float>(other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    ones(other.shape())
}

/// Create a tensor filled with `value` with the same shape as `other`.
pub fn full_like<T: Float>(other: &Tensor<T>, value: T) -> FerrotorchResult<Tensor<T>> {
    full(other.shape(), value)
}

/// Create a random tensor [0,1) with the same shape as `other`.
pub fn rand_like<T: Float>(other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    rand(other.shape())
}

/// Create a random normal tensor with the same shape as `other`.
pub fn randn_like<T: Float>(other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    randn(other.shape())
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
    fn test_arange_zero_step() {
        let result: Result<Tensor<f32>, _> = arange(0.0, 5.0, 0.0);
        assert!(result.is_err());
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
        let t: Tensor<f32> = rand(&[10, 20]).unwrap();
        assert_eq!(t.shape(), &[10, 20]);
        // Values should be in [0, 1).
        assert!(t.data().unwrap().iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_randn_shape() {
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
        let t: Tensor<f32> = zeros(&[5, 6]).unwrap();
        let r = rand_like(&t).unwrap();
        assert_eq!(r.shape(), &[5, 6]);
        assert!(r.data().unwrap().iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_randn_like() {
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
