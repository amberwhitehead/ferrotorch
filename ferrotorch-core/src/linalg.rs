//! Advanced linear algebra operations bridging to ferray-linalg.
//!
//! Complements `ops::linalg` (matmul, mm, mv, dot, bmm, transpose) with
//! decompositions, solvers, norms, and related functions. Each delegates to
//! the corresponding ferray-linalg routine via the same Array bridge pattern.
//!
//! **Backward support**: `solve` is autograd-aware on CPU and CUDA. `det`,
//! `inv`, `trace`, `outer`, `qr`, `cholesky`, `slogdet`, `svd`, `eig`,
//! `eigvals`, `eigh`, `eigvalsh`, `pinv`, `lstsq`, `lu`, `lu_factor`,
//! `vector_norm`, `matrix_norm`, and `householder_product` are autograd-aware
//! on their implemented forward devices. `svdvals`, `matrix_power`,
//! `tensorsolve`, and `tensorinv` compose existing differentiable primitives
//! where those primitives exist; `matrix_power`/`tensorsolve`/`tensorinv` also
//! run on CUDA for f32/f64 via resident `solve`/matmul. Remaining tracked
//! forward-only paths return structured errors instead of detached tensors.
//!
//! ## REQ status (per `.design/ferrotorch-core/linalg.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl `svd` via `ferray_linalg::svd`; non-test consumer `pinv`, `svdvals`, `matrix_rank`, `cond`. |
//! | REQ-2 | SHIPPED | impl `solve` (grad-aware: delegates to `grad_fns::linalg::solve_differentiable`); non-test consumer `ferrotorch-distributions::multivariate_normal`. |
//! | REQ-3 | SHIPPED | impl `det` (grad-aware: delegates to `grad_fns::linalg::det_differentiable`); non-test consumer `slogdet`. |
//! | REQ-4 | SHIPPED | impl `inv` (grad-aware: delegates to `grad_fns::linalg::inv_differentiable`); non-test consumer `inv_ex`. |
//! | REQ-5 | SHIPPED | impl `qr`; non-test consumer pub API + linear-regression downstream. |
//! | REQ-6 | SHIPPED | impl `cholesky`; non-test consumer `ferrotorch-distributions::multivariate_normal`. |
//! | REQ-7 | SHIPPED | impl `matrix_norm`; non-test consumer pub API. |
//! | REQ-8 | SHIPPED | impl `pinv`; non-test consumer composes with `svd`. |
//! | REQ-9 | SHIPPED | impl `eigh`, `eigvalsh` (CUDA via cuSOLVER `syevd`); non-test consumer `matrix_norm`, `cond`. |
//! | REQ-10 | SHIPPED | impl `eig`, `eigvals`; non-test consumer pub API. |
//! | REQ-11 | SHIPPED | impl `lu`; non-test consumer pub API. |
//! | REQ-12 | SHIPPED | impl `lu_factor`; non-test consumer `solve` CUDA path + `tensorsolve`. |
//! | REQ-13 | SHIPPED | impl `svdvals`; non-test consumer `matrix_rank`, `cond`. |
//! | REQ-14 | SHIPPED | impl `lstsq_solve`, `lstsq`; non-test consumer pub API. |
//! | REQ-15 | SHIPPED | impl `matrix_power`, `matrix_exp`; non-test consumer `ferrotorch-distributions` continuous-time models. |
//! | REQ-16 | SHIPPED | impl `tensorsolve`, `tensorinv`; non-test consumer pub API. |
//! | REQ-17 | SHIPPED | impl `vector_norm`; non-test consumer pub API. |
//! | REQ-18 | SHIPPED | impl `slogdet`; non-test consumer log-likelihood computations in `ferrotorch-distributions`. |
//! | REQ-19 | SHIPPED | impl `matrix_rank`, `cond`; non-test consumer pub API. |
//! | REQ-20 | SHIPPED | impl `cross`; non-test consumer pub API. |
//! | REQ-21 | SHIPPED | impl `multi_dot`; non-test consumer pub API. |
//! | REQ-22 | SHIPPED | impl `diagonal`; non-test consumer pub API. |
//! | REQ-23 | SHIPPED | impl `solve_triangular`; non-test consumer `cholesky_solve` paths. |
//! | REQ-24 | SHIPPED | impl `ldl_factor`, `ldl_solve`; non-test consumer pub API. |
//! | REQ-25 | SHIPPED | impl `householder_product`; non-test consumer `qr` reconstruction. |
//! | REQ-26 | SHIPPED | impl `cholesky_ex`, `inv_ex`, `solve_ex`; non-test consumer pub API. |
//! | REQ-27 | SHIPPED | impl `trace` (sum of main diagonal; grad-aware: delegates to `grad_fns::linalg::trace_differentiable` when grad enabled). |
//! | REQ-28 | SHIPPED | impl `outer` (1-D × 1-D outer product; grad-aware: delegates to `grad_fns::linalg::outer_differentiable` when grad enabled). |

use crate::device::Device;
use crate::dtype::{DType, Element, Float};
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::int_tensor::IntTensor;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;
use libloading::Library;
use std::any::TypeId;
use std::ffi::OsString;
use std::os::raw::{c_char, c_int};
use std::sync::OnceLock;

/// Return type for `torch.linalg.lstsq`-style calls.
pub type LstsqResult<T> = (Tensor<T>, Tensor<T>, IntTensor<i64>, Tensor<T>);

/// LAPACK/cuSOLVER driver selection for [`lstsq_with_driver`].
///
/// Mirrors `torch.linalg.lstsq(driver=...)`: CPU accepts all four LAPACK
/// drivers, CUDA accepts only [`LstsqDriver::Gels`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LstsqDriver {
    /// QR/LQ solve; assumes full-rank input. Does not compute rank or singular values.
    Gels,
    /// Complete orthogonal factorization with pivoting. PyTorch's CPU default.
    Gelsy,
    /// Divide-and-conquer SVD solve. Computes rank and singular values.
    Gelsd,
    /// Classic SVD solve. Computes rank and singular values.
    Gelss,
}

impl LstsqDriver {
    fn default_for_device(device: Device) -> Self {
        if device.is_cuda() {
            Self::Gels
        } else {
            Self::Gelsy
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Gels => "gels",
            Self::Gelsy => "gelsy",
            Self::Gelsd => "gelsd",
            Self::Gelss => "gelss",
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: convert Tensor<T> data to ferray Array<T, Ix2> or IxDyn
// ---------------------------------------------------------------------------

/// Build a ferray `Array<f64, Ix2>` from a 2-D tensor's data (f64 path).
fn tensor_to_array2_f64<T: Float>(
    t: &Tensor<T>,
) -> FerrotorchResult<ferray_core::Array<f64, ferray_core::Ix2>> {
    let shape = t.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("expected 2-D tensor, got {shape:?}"),
        });
    }
    let data: Vec<f64> = t.data()?.iter().map(|&v| v.to_f64().unwrap()).collect();
    ferray_core::Array::from_vec(ferray_core::Ix2::new([shape[0], shape[1]]), data)
        .map_err(FerrotorchError::Ferray)
}

/// Build a ferray `Array<f32, Ix2>` from a 2-D tensor's data (f32 path).
fn tensor_to_array2_f32<T: Float>(
    t: &Tensor<T>,
) -> FerrotorchResult<ferray_core::Array<f32, ferray_core::Ix2>> {
    let shape = t.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("expected 2-D tensor, got {shape:?}"),
        });
    }
    let data: Vec<f32> = t
        .data()?
        .iter()
        .map(|&v| v.to_f64().unwrap() as f32)
        .collect();
    ferray_core::Array::from_vec(ferray_core::Ix2::new([shape[0], shape[1]]), data)
        .map_err(FerrotorchError::Ferray)
}

/// Build a ferray `Array<f64, IxDyn>` from a tensor's data (any dimensionality).
fn tensor_to_arraydyn_f64<T: Float>(
    t: &Tensor<T>,
) -> FerrotorchResult<ferray_core::Array<f64, ferray_core::IxDyn>> {
    let data: Vec<f64> = t.data()?.iter().map(|&v| v.to_f64().unwrap()).collect();
    ferray_core::Array::from_vec(ferray_core::IxDyn::new(t.shape()), data)
        .map_err(FerrotorchError::Ferray)
}

/// Build a ferray `Array<f32, IxDyn>` from a tensor's data (any dimensionality).
fn tensor_to_arraydyn_f32<T: Float>(
    t: &Tensor<T>,
) -> FerrotorchResult<ferray_core::Array<f32, ferray_core::IxDyn>> {
    let data: Vec<f32> = t
        .data()?
        .iter()
        .map(|&v| v.to_f64().unwrap() as f32)
        .collect();
    ferray_core::Array::from_vec(ferray_core::IxDyn::new(t.shape()), data)
        .map_err(FerrotorchError::Ferray)
}

/// Convert a slice of f64 back to `Vec<T>`.
fn slice_to_vec<T: Float>(s: &[f64]) -> Vec<T> {
    s.iter().map(|&v| T::from(v).unwrap()).collect()
}

/// Convert a slice of f32 back to `Vec<T>`.
fn slice_f32_to_vec<T: Float>(s: &[f32]) -> Vec<T> {
    s.iter().map(|&v| T::from(v).unwrap()).collect()
}

/// True when `T` is f32 (4-byte float), used to pick the f32 vs f64 path.
fn is_f32<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

/// True when `T` is f64 (8-byte float).
fn is_f64<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

fn is_f16<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<half::f16>()
}

fn is_bf16<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<half::bf16>()
}

const LAPACK_ROW_MAJOR: c_int = 101;

type LapackeSgels = unsafe extern "C" fn(
    c_int,
    c_char,
    c_int,
    c_int,
    c_int,
    *mut f32,
    c_int,
    *mut f32,
    c_int,
) -> c_int;
type LapackeDgels = unsafe extern "C" fn(
    c_int,
    c_char,
    c_int,
    c_int,
    c_int,
    *mut f64,
    c_int,
    *mut f64,
    c_int,
) -> c_int;
type LapackeSgelsy = unsafe extern "C" fn(
    c_int,
    c_int,
    c_int,
    c_int,
    *mut f32,
    c_int,
    *mut f32,
    c_int,
    *mut c_int,
    f32,
    *mut c_int,
) -> c_int;
type LapackeDgelsy = unsafe extern "C" fn(
    c_int,
    c_int,
    c_int,
    c_int,
    *mut f64,
    c_int,
    *mut f64,
    c_int,
    *mut c_int,
    f64,
    *mut c_int,
) -> c_int;
type LapackeSgelsd = unsafe extern "C" fn(
    c_int,
    c_int,
    c_int,
    c_int,
    *mut f32,
    c_int,
    *mut f32,
    c_int,
    *mut f32,
    f32,
    *mut c_int,
) -> c_int;
type LapackeDgelsd = unsafe extern "C" fn(
    c_int,
    c_int,
    c_int,
    c_int,
    *mut f64,
    c_int,
    *mut f64,
    c_int,
    *mut f64,
    f64,
    *mut c_int,
) -> c_int;
type LapackeSgelss = LapackeSgelsd;
type LapackeDgelss = LapackeDgelsd;

struct LapackeBackend {
    _lib: Library,
    sgels: LapackeSgels,
    dgels: LapackeDgels,
    sgelsy: LapackeSgelsy,
    dgelsy: LapackeDgelsy,
    sgelsd: LapackeSgelsd,
    dgelsd: LapackeDgelsd,
    sgelss: LapackeSgelss,
    dgelss: LapackeDgelss,
}

impl LapackeBackend {
    fn load() -> Result<Self, String> {
        let mut candidates: Vec<OsString> = Vec::new();
        if let Some(path) = std::env::var_os("FERROTORCH_LAPACKE_LIB") {
            candidates.push(path);
        }
        candidates.extend(
            [
                "libopenblas.so.0",
                "libopenblas.so",
                "libflexiblas.so.3",
                "libflexiblas.so",
                "liblapacke.so.3",
                "liblapacke.so",
                "libopenblas.dylib",
                "/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
                "/usr/local/opt/openblas/lib/libopenblas.dylib",
            ]
            .into_iter()
            .map(OsString::from),
        );

        let mut errors = Vec::new();
        for candidate in candidates {
            let lib = match unsafe { Library::new(&candidate) } {
                Ok(lib) => lib,
                Err(err) => {
                    errors.push(format!("{}: {err}", candidate.to_string_lossy()));
                    continue;
                }
            };
            match unsafe { Self::from_library(lib) } {
                Ok(backend) => return Ok(backend),
                Err(err) => errors.push(format!(
                    "{}: missing required LAPACKE lstsq symbol: {err}",
                    candidate.to_string_lossy()
                )),
            }
        }

        Err(format!(
            "could not load LAPACKE lstsq symbols from OpenBLAS/FlexiBLAS/LAPACKE \
             candidates. Set FERROTORCH_LAPACKE_LIB to a library exporting \
             LAPACKE_sgels/LAPACKE_dgels/LAPACKE_sgelsy/LAPACKE_dgelsy/\
             LAPACKE_sgelsd/LAPACKE_dgelsd/LAPACKE_sgelss/LAPACKE_dgelss. \
             Attempts: {}",
            errors.join("; ")
        ))
    }

    unsafe fn from_library(lib: Library) -> Result<Self, String> {
        let sgels = unsafe { Self::symbol::<LapackeSgels>(&lib, b"LAPACKE_sgels\0")? };
        let dgels = unsafe { Self::symbol::<LapackeDgels>(&lib, b"LAPACKE_dgels\0")? };
        let sgelsy = unsafe { Self::symbol::<LapackeSgelsy>(&lib, b"LAPACKE_sgelsy\0")? };
        let dgelsy = unsafe { Self::symbol::<LapackeDgelsy>(&lib, b"LAPACKE_dgelsy\0")? };
        let sgelsd = unsafe { Self::symbol::<LapackeSgelsd>(&lib, b"LAPACKE_sgelsd\0")? };
        let dgelsd = unsafe { Self::symbol::<LapackeDgelsd>(&lib, b"LAPACKE_dgelsd\0")? };
        let sgelss = unsafe { Self::symbol::<LapackeSgelss>(&lib, b"LAPACKE_sgelss\0")? };
        let dgelss = unsafe { Self::symbol::<LapackeDgelss>(&lib, b"LAPACKE_dgelss\0")? };
        Ok(Self {
            _lib: lib,
            sgels,
            dgels,
            sgelsy,
            dgelsy,
            sgelsd,
            dgelsd,
            sgelss,
            dgelss,
        })
    }

    unsafe fn symbol<T: Copy>(lib: &Library, name: &'static [u8]) -> Result<T, String> {
        unsafe { lib.get::<T>(name) }
            .map(|sym| *sym)
            .map_err(|err| format!("{} ({err})", String::from_utf8_lossy(name)))
    }
}

static LAPACKE_BACKEND: OnceLock<Result<LapackeBackend, String>> = OnceLock::new();

fn lapacke_backend() -> FerrotorchResult<&'static LapackeBackend> {
    match LAPACKE_BACKEND.get_or_init(LapackeBackend::load) {
        Ok(backend) => Ok(backend),
        Err(message) => Err(FerrotorchError::InvalidArgument {
            message: format!("lstsq: LAPACKE backend unavailable: {message}"),
        }),
    }
}

fn checked_lapack_i32(value: usize, name: &str) -> FerrotorchResult<c_int> {
    c_int::try_from(value).map_err(|_| FerrotorchError::InvalidArgument {
        message: format!("lstsq: {name}={value} exceeds LP64 LAPACK i32 limit"),
    })
}

/// Guard: linalg decompositions are CPU-only. Return an explicit error for
/// GPU tensors instead of silently downloading data to host.
fn require_cpu<T: Float>(t: &Tensor<T>, op: &str) -> FerrotorchResult<()> {
    if t.is_cuda() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "{op}: GPU tensors are not supported for linalg decompositions. \
                 Call `.cpu()` explicitly before calling `{op}`."
            ),
        });
    }
    Ok(())
}

fn tracking_enabled_for<T: Float>(tensors: &[&Tensor<T>]) -> bool {
    crate::autograd::no_grad::is_grad_enabled() && tensors.iter().any(|t| t.requires_grad())
}

fn unsupported_linalg_autograd(op: &str) -> FerrotorchError {
    FerrotorchError::InvalidArgument {
        message: format!(
            "{op}: autograd is not implemented for this path; refusing to return a detached tensor"
        ),
    }
}

fn reject_forward_only_autograd<T: Float>(
    op: &str,
    tensors: &[&Tensor<T>],
) -> FerrotorchResult<()> {
    if tracking_enabled_for(tensors) {
        return Err(unsupported_linalg_autograd(op));
    }
    Ok(())
}

fn checked_product(dims: &[usize], op: &str) -> FerrotorchResult<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!("{op}: shape product overflows usize for dims {dims:?}"),
            })
    })
}

fn eye_on_device<T: Float>(n: usize, device: Device) -> FerrotorchResult<Tensor<T>> {
    let eye = crate::creation::eye::<T>(n)?;
    if device == Device::Cpu {
        Ok(eye)
    } else {
        eye.to(device)
    }
}

fn full_like_on_device<T: Float>(
    shape: &[usize],
    value: T,
    device: Device,
    op: &str,
) -> FerrotorchResult<Tensor<T>> {
    let numel = checked_product(shape, op)?;
    let t = Tensor::from_storage(
        TensorStorage::cpu(vec![value; numel]),
        shape.to_vec(),
        false,
    )?;
    if device == Device::Cpu {
        Ok(t)
    } else {
        t.to(device)
    }
}

fn effective_triangular_for_solve<T: Float>(
    a: &Tensor<T>,
    upper: bool,
    transpose: bool,
    unit_diagonal: bool,
) -> FerrotorchResult<Tensor<T>> {
    let n = a.shape()[0];
    let effective = if unit_diagonal {
        let strict = if upper {
            crate::ops::tensor_ops::triu(a, 1)?
        } else {
            crate::ops::tensor_ops::tril(a, -1)?
        };
        let eye = eye_on_device(n, a.device())?;
        strict.add_t(&eye)?
    } else if upper {
        crate::ops::tensor_ops::triu(a, 0)?
    } else {
        crate::ops::tensor_ops::tril(a, 0)?
    };

    if transpose {
        effective.transpose(0, 1)?.contiguous()
    } else {
        effective.contiguous()
    }
}

#[allow(
    clippy::float_cmp,
    reason = "p is a user-provided discrete norm selector; accepting near-2 values would diverge from torch"
)]
fn is_cond_svd_selector(p: f64) -> Option<bool> {
    if p == 2.0 {
        Some(false)
    } else if p == -2.0 {
        Some(true)
    } else {
        None
    }
}

#[allow(
    clippy::float_cmp,
    reason = "torch.linalg.cond accepts an exact discrete set of numeric norm selectors"
)]
fn validate_cond_selector(p: f64) -> FerrotorchResult<()> {
    if p == 1.0
        || p == -1.0
        || p == 2.0
        || p == -2.0
        || p == f64::INFINITY
        || p == f64::NEG_INFINITY
    {
        Ok(())
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: format!("linalg.cond got an invalid norm type: {p}"),
        })
    }
}

// ---------------------------------------------------------------------------
// Singular Value Decomposition
// ---------------------------------------------------------------------------

/// Singular Value Decomposition: `A = U @ diag(S) @ Vh`.
///
/// Returns `(U, S, Vh)` where `U` and `Vh` are unitary and `S` contains
/// singular values in descending order. Uses reduced (thin) SVD.
///
/// # Backward
/// Autograd-aware (CPU): when grad tracking is active for `input`, this routes
/// through `crate::grad_fns::linalg::svd_differentiable` (the real reduced-SVD
/// VJP mirroring `svd_backward` at `FunctionsManual.cpp:3605`, split across the
/// three `U`/`S`/`Vh` outputs and accumulated into `A.grad`, including the
/// rectangular `m != n` projector terms). The CUDA forward stays forward-only.
pub fn svd<T: Float>(input: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>, Tensor<T>)> {
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("svd requires a 2-D tensor, got {shape:?}"),
        });
    }

    // Autograd path: delegate to the differentiable wrapper. CUDA backward
    // composes resident matmul, diag, and broadcast arithmetic.
    if crate::autograd::no_grad::is_grad_enabled() && input.requires_grad() {
        return crate::grad_fns::linalg::svd_differentiable(input);
    }

    if input.is_cuda() {
        // GPU dispatch via cuSOLVER. Reduced SVD shapes:
        //   U: [m, k], S: [k], Vh: [k, n], k = min(m, n)
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let m = shape[0];
        let n = shape[1];
        let k = m.min(n);
        let buf = input.gpu_handle()?;
        let (u_h, s_h, vh_h) = if is_f32::<T>() {
            backend.svd_f32(buf, m, n)?
        } else if is_f64::<T>() {
            backend.svd_f64(buf, m, n)?
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: "svd requires f32 or f64".into(),
            });
        };
        return Ok((
            Tensor::from_storage(TensorStorage::gpu(u_h), vec![m, k], false)?,
            Tensor::from_storage(TensorStorage::gpu(s_h), vec![k], false)?,
            Tensor::from_storage(TensorStorage::gpu(vh_h), vec![k, n], false)?,
        ));
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(input)?;
        let (u, s, vh) = ferray_linalg::svd(&arr, false).map_err(FerrotorchError::Ferray)?;
        let u_data = slice_f32_to_vec::<T>(u.as_slice().unwrap());
        let s_data = slice_f32_to_vec::<T>(s.as_slice().unwrap());
        let vh_data = slice_f32_to_vec::<T>(vh.as_slice().unwrap());
        let u_shape = u.shape().to_vec();
        let s_shape = s.shape().to_vec();
        let vh_shape = vh.shape().to_vec();
        Ok((
            Tensor::from_storage(TensorStorage::cpu(u_data), u_shape, false)?,
            Tensor::from_storage(TensorStorage::cpu(s_data), s_shape, false)?,
            Tensor::from_storage(TensorStorage::cpu(vh_data), vh_shape, false)?,
        ))
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(input)?;
        let (u, s, vh) = ferray_linalg::svd(&arr, false).map_err(FerrotorchError::Ferray)?;
        let u_data = slice_to_vec::<T>(u.as_slice().unwrap());
        let s_data = slice_to_vec::<T>(s.as_slice().unwrap());
        let vh_data = slice_to_vec::<T>(vh.as_slice().unwrap());
        let u_shape = u.shape().to_vec();
        let s_shape = s.shape().to_vec();
        let vh_shape = vh.shape().to_vec();
        Ok((
            Tensor::from_storage(TensorStorage::cpu(u_data), u_shape, false)?,
            Tensor::from_storage(TensorStorage::cpu(s_data), s_shape, false)?,
            Tensor::from_storage(TensorStorage::cpu(vh_data), vh_shape, false)?,
        ))
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Solve linear system
// ---------------------------------------------------------------------------

/// Solve the linear system `A @ x = b`.
///
/// `a` must be a square 2-D tensor. `b` can be 1-D (single RHS) or 2-D
/// (multiple RHS columns).
///
/// # Backward
/// Autograd-aware (CPU): when grad tracking is active for `a` or `b`, this
/// routes through `crate::grad_fns::linalg::solve_differentiable` (the real
/// `linalg_solve_backward` VJP: `gB = A^{-T} @ gX`, `gA = -gB @ X^T`). The
/// CUDA forward stays forward-only.
pub fn solve<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.ndim() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("solve: `a` must be 2-D, got {:?}", a.shape()),
        });
    }
    if a.shape()[0] != a.shape()[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "solve: `a` must be square, got {}x{}",
                a.shape()[0],
                a.shape()[1]
            ),
        });
    }
    if a.is_cuda() != b.is_cuda() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }

    // Autograd path: CPU and CUDA both route through the same wrapper. The
    // wrapper's CUDA backward stays resident via cuSOLVER + CUDA mm_bt.
    if crate::autograd::no_grad::is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        return crate::grad_fns::linalg::solve_differentiable(a, b);
    }

    if a.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let n = a.shape()[0];
        // b can be [n] (single RHS) or [n, nrhs].
        let nrhs = if b.ndim() == 1 { 1 } else { b.shape()[1] };
        let x_h = if is_f32::<T>() {
            backend.solve_f32(a.gpu_handle()?, b.gpu_handle()?, n, nrhs)?
        } else if is_f64::<T>() {
            backend.solve_f64(a.gpu_handle()?, b.gpu_handle()?, n, nrhs)?
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: "solve requires f32 or f64".into(),
            });
        };
        let out_shape: Vec<usize> = if b.ndim() == 1 {
            vec![n]
        } else {
            vec![n, nrhs]
        };
        return Tensor::from_storage(TensorStorage::gpu(x_h), out_shape, false);
    }

    if is_f32::<T>() {
        let a_arr = tensor_to_array2_f32(a)?;
        let b_arr = tensor_to_arraydyn_f32(b)?;
        let x = ferray_linalg::solve(&a_arr, &b_arr).map_err(FerrotorchError::Ferray)?;
        let x_data = slice_f32_to_vec::<T>(x.as_slice().unwrap());
        let x_shape = x.shape().to_vec();
        Tensor::from_storage(TensorStorage::cpu(x_data), x_shape, false)
    } else if is_f64::<T>() {
        let a_arr = tensor_to_array2_f64(a)?;
        let b_arr = tensor_to_arraydyn_f64(b)?;
        let x = ferray_linalg::solve(&a_arr, &b_arr).map_err(FerrotorchError::Ferray)?;
        let x_data = slice_to_vec::<T>(x.as_slice().unwrap());
        let x_shape = x.shape().to_vec();
        Tensor::from_storage(TensorStorage::cpu(x_data), x_shape, false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Determinant
// ---------------------------------------------------------------------------

/// Matrix determinant of a square 2-D tensor.
///
/// Returns a scalar tensor.
///
/// # Backward
/// Autograd-aware (CPU): when grad tracking is active for `input`, this routes
/// through `crate::grad_fns::linalg::det_differentiable` (the invertible-branch
/// VJP `dA = grad * det(A) * inv(A)^T`).
pub fn det<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    require_cpu(input, "det")?;
    let shape = input.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("det requires a square 2-D tensor, got {shape:?}"),
        });
    }

    // Autograd path: delegate to the differentiable wrapper, which computes
    // the forward (and the VJP's internal `inv`) inside `no_grad` (preventing
    // re-entry here) and attaches `LinalgDetBackward`.
    if crate::autograd::no_grad::is_grad_enabled() && input.requires_grad() {
        return crate::grad_fns::linalg::det_differentiable(input);
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(input)?;
        let d: f32 = ferray_linalg::det(&arr).map_err(FerrotorchError::Ferray)?;
        let val = T::from(d).unwrap();
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(input)?;
        let d: f64 = ferray_linalg::det(&arr).map_err(FerrotorchError::Ferray)?;
        let val = T::from(d).unwrap();
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Matrix inverse
// ---------------------------------------------------------------------------

/// Matrix inverse of a square 2-D tensor.
///
/// # Backward
/// Autograd-aware (CPU): when grad tracking is active for `input`, this routes
/// through `crate::grad_fns::linalg::inv_differentiable` (the VJP
/// `dA = -Y^T @ grad @ Y^T`, `Y = A^{-1}`).
pub fn inv<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    require_cpu(input, "inv")?;
    let shape = input.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("inv requires a square 2-D tensor, got {shape:?}"),
        });
    }

    // Autograd path: delegate to the differentiable wrapper, which computes
    // the forward inside `no_grad` (preventing re-entry here) and attaches
    // `LinalgInvBackward`.
    if crate::autograd::no_grad::is_grad_enabled() && input.requires_grad() {
        return crate::grad_fns::linalg::inv_differentiable(input);
    }

    let n = shape[0];

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(input)?;
        let r = ferray_linalg::inv(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(r.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), vec![n, n], false)
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(input)?;
        let r = ferray_linalg::inv(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(r.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), vec![n, n], false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// QR decomposition
// ---------------------------------------------------------------------------

/// QR decomposition: `A = Q @ R`.
///
/// Returns `(Q, R)` in reduced form.
///
/// # Backward
/// Autograd-aware (CPU, reduced mode, `m >= n`): when grad tracking is active
/// for `input`, this routes through `crate::grad_fns::linalg::qr_differentiable`
/// (the real `linalg_qr_backward` VJP). CUDA forward stays forward-only.
pub fn qr<T: Float>(input: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("qr requires a 2-D tensor, got {shape:?}"),
        });
    }

    // Autograd path: delegate to the differentiable wrapper. CUDA backward
    // composes resident matmul, triangular masks, transpose, and solve.
    if crate::autograd::no_grad::is_grad_enabled() && input.requires_grad() {
        return crate::grad_fns::linalg::qr_differentiable(input);
    }

    if input.is_cuda() {
        // Reduced QR shapes: Q [m, k], R [k, n], k = min(m, n)
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let m = shape[0];
        let n = shape[1];
        let k = m.min(n);
        let (q_h, r_h) = if is_f32::<T>() {
            backend.qr_f32(input.gpu_handle()?, m, n)?
        } else if is_f64::<T>() {
            backend.qr_f64(input.gpu_handle()?, m, n)?
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: "qr requires f32 or f64".into(),
            });
        };
        return Ok((
            Tensor::from_storage(TensorStorage::gpu(q_h), vec![m, k], false)?,
            Tensor::from_storage(TensorStorage::gpu(r_h), vec![k, n], false)?,
        ));
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(input)?;
        let (q, r) = ferray_linalg::qr(&arr, ferray_linalg::QrMode::Reduced)
            .map_err(FerrotorchError::Ferray)?;
        let q_data = slice_f32_to_vec::<T>(q.as_slice().unwrap());
        let r_data = slice_f32_to_vec::<T>(r.as_slice().unwrap());
        let q_shape = q.shape().to_vec();
        let r_shape = r.shape().to_vec();
        Ok((
            Tensor::from_storage(TensorStorage::cpu(q_data), q_shape, false)?,
            Tensor::from_storage(TensorStorage::cpu(r_data), r_shape, false)?,
        ))
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(input)?;
        let (q, r) = ferray_linalg::qr(&arr, ferray_linalg::QrMode::Reduced)
            .map_err(FerrotorchError::Ferray)?;
        let q_data = slice_to_vec::<T>(q.as_slice().unwrap());
        let r_data = slice_to_vec::<T>(r.as_slice().unwrap());
        let q_shape = q.shape().to_vec();
        let r_shape = r.shape().to_vec();
        Ok((
            Tensor::from_storage(TensorStorage::cpu(q_data), q_shape, false)?,
            Tensor::from_storage(TensorStorage::cpu(r_data), r_shape, false)?,
        ))
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Cholesky decomposition
// ---------------------------------------------------------------------------

/// Cholesky decomposition of a symmetric positive-definite matrix.
///
/// Returns the lower-triangular factor `L` such that `A = L @ L^T`.
///
/// # Backward
/// Autograd-aware: when grad tracking is active for `input`, this routes
/// through `crate::grad_fns::linalg::cholesky_differentiable` (the
/// Phi-symmetrisation VJP). CUDA forward stays forward-only.
pub fn cholesky<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("cholesky requires a square 2-D tensor, got {shape:?}"),
        });
    }

    let n = shape[0];

    // Autograd path: delegate to the differentiable wrapper. CUDA backward
    // stays resident by composing CUDA triangular masks, matmul, and solve.
    if crate::autograd::no_grad::is_grad_enabled() && input.requires_grad() {
        return crate::grad_fns::linalg::cholesky_differentiable(input);
    }

    if input.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let l_h = if is_f32::<T>() {
            backend.cholesky_f32(input.gpu_handle()?, n)?
        } else if is_f64::<T>() {
            backend.cholesky_f64(input.gpu_handle()?, n)?
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: "cholesky requires f32 or f64".into(),
            });
        };
        return Tensor::from_storage(TensorStorage::gpu(l_h), vec![n, n], false);
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(input)?;
        let l = ferray_linalg::cholesky(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(l.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), vec![n, n], false)
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(input)?;
        let l = ferray_linalg::cholesky(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(l.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), vec![n, n], false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Matrix norm (Frobenius)
// ---------------------------------------------------------------------------

/// Matrix norm (Frobenius by default).
///
/// Returns a scalar tensor containing the Frobenius norm.
///
/// # Backward
/// Not yet implemented. Returns non-grad tensors.
pub fn matrix_norm<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("matrix_norm requires a 2-D tensor, got {shape:?}"),
        });
    }

    // Autograd path: CPU keeps the existing closed-form wrapper. CUDA composes
    // resident primitives (`sqrt(sum(x*x))`) so backward flows through the
    // existing Mul/Sum/Sqrt CUDA nodes instead of saving a host scalar.
    if crate::autograd::no_grad::is_grad_enabled() && input.requires_grad() {
        if input.is_cuda() {
            let squared = input.mul_t(input)?;
            return squared.sum_all()?.sqrt_t();
        }
        return crate::grad_fns::linalg::matrix_norm_differentiable(input);
    }

    if input.is_cuda() {
        // Frobenius norm: sqrt(sum_ij A_ij^2). Composes existing GPU
        // primitives (mul → reduce_sum → sqrt) — three kernel launches but
        // fully GPU-resident; result lands as a 0-d tensor on device.
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let buf = input.gpu_handle()?;
        let numel = crate::shape::numel(shape);
        let h = if is_f32::<T>() {
            let sq = backend.mul_f32(buf, buf)?;
            let s = backend.sum_f32(&sq, numel)?;
            backend.sqrt_f32(&s)?
        } else if is_f64::<T>() {
            let sq = backend.mul_f64(buf, buf)?;
            let s = backend.sum_f64(&sq, numel)?;
            backend.sqrt_f64(&s)?
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: "matrix_norm requires f32 or f64".into(),
            });
        };
        return Tensor::from_storage(TensorStorage::gpu(h), vec![], false);
    }

    if is_f32::<T>() {
        let arr = tensor_to_arraydyn_f32(input)?;
        let n: f32 = ferray_linalg::norm(&arr, ferray_linalg::NormOrder::Fro)
            .map_err(FerrotorchError::Ferray)?;
        let val = T::from(n).unwrap();
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)
    } else if is_f64::<T>() {
        let arr = tensor_to_arraydyn_f64(input)?;
        let n: f64 = ferray_linalg::norm(&arr, ferray_linalg::NormOrder::Fro)
            .map_err(FerrotorchError::Ferray)?;
        let val = T::from(n).unwrap();
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Pseudo-inverse
// ---------------------------------------------------------------------------

/// Moore-Penrose pseudo-inverse of a 2-D tensor.
///
/// # Backward
/// Autograd-aware on CPU and CUDA f32/f64. CUDA forward composes the resident
/// SVD path with tensor reductions/comparisons/where, so no host value round
/// trip is needed for the singular-value cutoff or reconstruction.
pub fn pinv<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("pinv requires a 2-D tensor, got {shape:?}"),
        });
    }

    // Autograd path: delegate to the differentiable wrapper, which computes
    // the forward under `no_grad` (preventing re-entry here) and attaches
    // `PinvBackward` (the algebraic full-rank Moore-Penrose VJP).
    if crate::autograd::no_grad::is_grad_enabled() && input.requires_grad() {
        return crate::grad_fns::linalg::pinv_differentiable(input);
    }

    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        return pinv_svd_cuda(input);
    }
    require_cpu(input, "pinv")?;

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(input)?;
        let r = ferray_linalg::pinv(&arr, None).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(r.as_slice().unwrap());
        let r_shape = r.shape().to_vec();
        Tensor::from_storage(TensorStorage::cpu(data), r_shape, false)
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(input)?;
        let r = ferray_linalg::pinv(&arr, None).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(r.as_slice().unwrap());
        let r_shape = r.shape().to_vec();
        Tensor::from_storage(TensorStorage::cpu(data), r_shape, false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

fn pinv_svd_cuda<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let m = input.shape()[0];
    let n = input.shape()[1];
    if m == 0 || n == 0 {
        return Tensor::from_storage(
            TensorStorage::on_device(Vec::new(), input.device())?,
            vec![n, m],
            false,
        );
    }

    let (u, s, vh) = svd(input)?;
    let smax = crate::grad_fns::reduction::amax(&s)?;
    let eps = T::epsilon();
    let rtol = T::from((m.max(n)) as f64).ok_or_else(|| FerrotorchError::InvalidArgument {
        message: "pinv: shape is not representable in dtype".into(),
    })? * eps;
    let rtol_t = crate::creation::full_like(&smax, rtol)?;
    let threshold = smax.mul_t(&rtol_t)?;
    let keep = crate::bool_tensor::BoolTensor::gt(&s, &threshold)?;
    let ones = crate::creation::ones_like(&s)?;
    let inv_s = ones.div_t(&s)?;
    let zeros = crate::creation::zeros_like(&s)?;
    let inv_s = crate::grad_fns::comparison::where_bt(&keep, &inv_s, &zeros)?;
    let sigma_pinv = crate::ops::tensor_ops::diag(&inv_s, 0)?;
    let v = vh.transpose(0, 1)?.contiguous()?;
    let ut = u.transpose(0, 1)?.contiguous()?;
    v.mm(&sigma_pinv)?.mm(&ut)
}

// ===========================================================================
// Eigendecomposition (Hermitian / general)
// ===========================================================================

/// Symmetric / Hermitian eigendecomposition: `A = Q diag(w) Q^T`.
///
/// `a` must be square and (numerically) symmetric. Returns `(w, Q)` where
/// `w` are real eigenvalues in ascending order and `Q` is the orthogonal
/// matrix of eigenvectors (column `i` of `Q` is the eigenvector for `w[i]`).
///
/// Mirrors `torch.linalg.eigh`. GPU-resident on CUDA via cuSOLVER `syevd`.
pub fn eigh<T: Float>(a: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("eigh requires a square 2-D tensor, got {shape:?}"),
        });
    }
    let n = shape[0];

    // Autograd path: delegate to the differentiable wrapper. CUDA backward
    // composes resident matmul, diag, broadcast arithmetic, and masks.
    if crate::autograd::no_grad::is_grad_enabled() && a.requires_grad() {
        return crate::grad_fns::linalg::eigh_differentiable(a);
    }

    if a.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let buf = a.gpu_handle()?;
        let (w_h, v_h) = if is_f32::<T>() {
            backend.eigh_f32(buf, n)?
        } else if is_f64::<T>() {
            backend.eigh_f64(buf, n)?
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: "eigh requires f32 or f64".into(),
            });
        };
        return Ok((
            Tensor::from_storage(TensorStorage::gpu(w_h), vec![n], false)?,
            Tensor::from_storage(TensorStorage::gpu(v_h), vec![n, n], false)?,
        ));
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let (w, q) = ferray_linalg::eigh(&arr).map_err(FerrotorchError::Ferray)?;
        let w_data = slice_f32_to_vec::<T>(w.as_slice().unwrap());
        let mut q_data = slice_f32_to_vec::<T>(q.as_slice().unwrap());
        canonicalize_eigenvector_signs(&mut q_data, n);
        Ok((
            Tensor::from_storage(TensorStorage::cpu(w_data), vec![n], false)?,
            Tensor::from_storage(TensorStorage::cpu(q_data), vec![n, n], false)?,
        ))
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let (w, q) = ferray_linalg::eigh(&arr).map_err(FerrotorchError::Ferray)?;
        let w_data = slice_to_vec::<T>(w.as_slice().unwrap());
        let mut q_data = slice_to_vec::<T>(q.as_slice().unwrap());
        canonicalize_eigenvector_signs(&mut q_data, n);
        Ok((
            Tensor::from_storage(TensorStorage::cpu(w_data), vec![n], false)?,
            Tensor::from_storage(TensorStorage::cpu(q_data), vec![n, n], false)?,
        ))
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// Canonicalize the per-column SIGN of a row-major `n×n` eigenvector matrix `q`
/// (column `j` is the eigenvector for `w[j]`) to a deterministic convention:
/// the entry of largest absolute value in each column is made non-negative.
///
/// # Gauge freedom (R-DEV-1 caveat, cite `FunctionsManual.cpp:3877-3880`)
///
/// Eigenvectors of a symmetric matrix are defined only up to a sign (for real
/// matrices, up to multiplication by `e^{i phi}` in the complex case — upstream
/// documents this at `torch/csrc/autograd/FunctionsManual.cpp:3877-3880`:
/// "The eigenvectors ... are specified up to multiplication by e^{i phi}. The
/// specified loss function depends on this quantity, so it is ill-defined.").
/// LAPACK `syevd` (what `torch.linalg.eigh` returns) emits an implementation-
/// defined sign per column; ferray (faer-backed) emits its own. Neither is
/// "more correct". This routine gives ferrotorch a STABLE, REPRODUCIBLE sign
/// contract so two `eigh` calls on the same input return identical eigenvectors
/// — it does NOT (and cannot, without replicating LAPACK) match torch's signs.
///
/// The downstream `EighBackwardV` VJP is sign-consistent: flipping a column of
/// `U` flips the same column of the cotangent `gU`, and the skew-symmetric
/// projection + `U @ ret @ U^T` conjugation is invariant under that joint flip.
/// So for SIGN-INVARIANT losses (`L(U) = L(U·diag(±1))` — PCA, whitening,
/// `U @ diag(f(w)) @ U^T` reconstructions, every well-posed objective on
/// eigenvectors) the gradient is convention-independent and matches torch
/// byte-for-byte. Only gauge-DEPENDENT losses (a raw `<W, U>` linear functional)
/// observe the convention, and for those no implementation can match torch's
/// arbitrary LAPACK signs.
fn canonicalize_eigenvector_signs<T: Float>(q: &mut [T], n: usize) {
    let zero = <T as num_traits::Zero>::zero();
    for col in 0..n {
        // Find the row of the largest-magnitude entry in this column.
        let mut best_row = 0usize;
        let mut best_abs = zero;
        for row in 0..n {
            let v = q[row * n + col].abs();
            if v > best_abs {
                best_abs = v;
                best_row = row;
            }
        }
        // If that pivot entry is negative, flip the whole column.
        if q[best_row * n + col] < zero {
            for row in 0..n {
                q[row * n + col] = -q[row * n + col];
            }
        }
    }
}

/// Eigenvalues of a symmetric / Hermitian matrix (real, ascending).
///
/// Mirrors `torch.linalg.eigvalsh`. GPU-resident on CUDA via cuSOLVER `syevd`
/// with `jobz=NOVECTOR`.
pub fn eigvalsh<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("eigvalsh requires a square 2-D tensor, got {shape:?}"),
        });
    }
    let n = shape[0];

    // Autograd path: delegate to the differentiable wrapper. CUDA backward
    // composes resident matmul/diag kernels.
    if crate::autograd::no_grad::is_grad_enabled() && a.requires_grad() {
        return crate::grad_fns::linalg::eigvalsh_differentiable(a);
    }

    if a.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let buf = a.gpu_handle()?;
        let w_h = if is_f32::<T>() {
            backend.eigvalsh_f32(buf, n)?
        } else if is_f64::<T>() {
            backend.eigvalsh_f64(buf, n)?
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: "eigvalsh requires f32 or f64".into(),
            });
        };
        return Tensor::from_storage(TensorStorage::gpu(w_h), vec![n], false);
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let w = ferray_linalg::eigvalsh(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(w.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), vec![n], false)
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let w = ferray_linalg::eigvalsh(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(w.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), vec![n], false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// General (non-symmetric) eigendecomposition.
///
/// Returns `(w, V)` where eigenvalues `w` and eigenvectors `V` are
/// **complex-valued**, encoded as tensors with a trailing dimension of
/// size 2 representing `[real, imag]` (matching ferrotorch's complex
/// convention used by [`fft`](crate::fft)). `w` has shape `[n, 2]` and
/// `V` has shape `[n, n, 2]`.
///
/// Mirrors `torch.linalg.eig`. CPU-only today.
pub fn eig<T: Float>(a: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    require_cpu(a, "eig")?;
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("eig requires a square 2-D tensor, got {shape:?}"),
        });
    }
    let n = shape[0];

    // Autograd path (CPU): delegate to the differentiable wrapper, which
    // computes the forward under `no_grad` (preventing re-entry here) and
    // attaches the split `EigBackwardW` / `EigBackwardV` nodes (complex VJP,
    // real `A.grad` via `at::real`). #1345.
    if crate::autograd::no_grad::is_grad_enabled() && a.requires_grad() {
        return crate::grad_fns::linalg::eig_differentiable(a);
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let (w, v) = ferray_linalg::eig(&arr).map_err(FerrotorchError::Ferray)?;
        let w_data: Vec<T> = w
            .as_slice()
            .unwrap()
            .iter()
            .flat_map(|c| [T::from(c.re).unwrap(), T::from(c.im).unwrap()])
            .collect();
        let mut v_data: Vec<T> = v
            .as_slice()
            .unwrap()
            .iter()
            .flat_map(|c| [T::from(c.re).unwrap(), T::from(c.im).unwrap()])
            .collect();
        normalize_complex_eigenvector_columns(&mut v_data, n);
        canonicalize_complex_eigenvector_phase(&mut v_data, n);
        Ok((
            Tensor::from_storage(TensorStorage::cpu(w_data), vec![n, 2], false)?,
            Tensor::from_storage(TensorStorage::cpu(v_data), vec![n, n, 2], false)?,
        ))
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let (w, v) = ferray_linalg::eig(&arr).map_err(FerrotorchError::Ferray)?;
        let w_data: Vec<T> = w
            .as_slice()
            .unwrap()
            .iter()
            .flat_map(|c| [T::from(c.re).unwrap(), T::from(c.im).unwrap()])
            .collect();
        let mut v_data: Vec<T> = v
            .as_slice()
            .unwrap()
            .iter()
            .flat_map(|c| [T::from(c.re).unwrap(), T::from(c.im).unwrap()])
            .collect();
        normalize_complex_eigenvector_columns(&mut v_data, n);
        canonicalize_complex_eigenvector_phase(&mut v_data, n);
        Ok((
            Tensor::from_storage(TensorStorage::cpu(w_data), vec![n, 2], false)?,
            Tensor::from_storage(TensorStorage::cpu(v_data), vec![n, n, 2], false)?,
        ))
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// Normalize each COLUMN of a complex `n×n` eigenvector matrix `v` (encoded
/// row-major `[n,n,2]` as interleaved `[re, im]`) to UNIT 2-NORM, matching
/// `torch.linalg.eig`'s documented contract that eigenvectors have norm one
/// (`torch/csrc/autograd/FunctionsManual.cpp:3837-3839` — "the eigenvalue
/// decomposition is returned with eigenvectors normalized to have norm one").
///
/// ferray's faer-backed `eig` forward returns eigenvectors with an arbitrary
/// per-column scale (e.g. a pivot entry forced to 1), NOT unit norm. The
/// `linalg_eig_backward` VJP's unit-norm-tangent projection term
/// `-V^H V real(diag(V^H gV))` (`FunctionsManual.cpp:3887-3889`) is only correct
/// when columns are unit-norm, so normalizing here (R-DEV-1, match torch's
/// numerical contract) makes BOTH the forward V and the backward consistent with
/// torch. After normalization ferrotorch's V matches torch's V up to a per-column
/// phase `e^{i phi}` (a genuine gauge freedom torch documents at
/// `FunctionsManual.cpp:3877-3880`); a phase-invariant loss (`sum(|V_ij|^2 M)`)
/// is therefore comparable byte-for-byte.
fn normalize_complex_eigenvector_columns<T: Float>(v: &mut [T], n: usize) {
    let zero = <T as num_traits::Zero>::zero();
    for col in 0..n {
        // Column 2-norm: sqrt(sum_i (re^2 + im^2)).
        let mut sumsq = zero;
        for row in 0..n {
            let base = 2 * (row * n + col);
            let re = v[base];
            let im = v[base + 1];
            sumsq = sumsq + re * re + im * im;
        }
        let norm = sumsq.sqrt();
        if norm > zero {
            for row in 0..n {
                let base = 2 * (row * n + col);
                v[base] = v[base] / norm;
                v[base + 1] = v[base + 1] / norm;
            }
        }
    }
}

/// Canonicalize the PHASE of each complex eigenvector COLUMN deterministically
/// (encoded row-major `[n,n,2]` as interleaved `[re, im]`).
///
/// Complex eigenvectors are defined only up to multiplication by a per-column
/// phase `e^{i phi}` — a genuine gauge freedom that `torch.linalg.eig`
/// documents at `torch/csrc/autograd/FunctionsManual.cpp:3867-3879` ("The
/// eigenvectors in the complex case are specified up to multiplication by
/// e^{i phi}. The specified loss function depends on this quantity, so it is
/// ill-defined."). ferray's faer-backed `eig` emits per-column phases that
/// differ matrix-by-matrix from torch's LAPACK `geev` gauge; matching torch's
/// arbitrary LAPACK phase would require replicating `geev` (impractical and
/// not the contract). Instead we pick a DETERMINISTIC, reproducible gauge:
/// multiply each column by `e^{-i phi}` so that its LARGEST-MAGNITUDE component
/// becomes real-POSITIVE (its imaginary part driven to 0 and real part > 0).
///
/// This mirrors `canonicalize_eigenvector_signs` for the real `eigh` case
/// (which forces the largest-magnitude component non-negative — the real-axis
/// specialization of the same idea). It does NOT match torch's gauge (that is
/// impossible without LAPACK), but it makes ferrotorch's eig output
/// REPRODUCIBLE: calling `eig` twice on the same input yields identical `V`.
///
/// For PHASE-INVARIANT losses (`sum(|V_ij|^2 M)`, reconstructions, any
/// well-posed objective) the gradient is gauge-free, so this rotation does NOT
/// change `A.grad` — ferrotorch still matches torch. For PHASE-DEPENDENT losses
/// the value is ill-defined regardless of gauge; the `EigBackwardV` guard
/// rejects grossly-phase-dependent losses, but its exact threshold is
/// gauge-dependent and may differ from torch's LAPACK-gauge boundary (the
/// losses in any divergent window are mathematically meaningless anyway).
fn canonicalize_complex_eigenvector_phase<T: Float>(v: &mut [T], n: usize) {
    let zero = <T as num_traits::Zero>::zero();
    for col in 0..n {
        // Find the row of the largest-magnitude entry (|re|^2 + |im|^2) in this
        // column — the canonical pivot whose phase we rotate to the real axis.
        let mut best_row = 0usize;
        let mut best_mag = zero;
        for row in 0..n {
            let base = 2 * (row * n + col);
            let mag = v[base] * v[base] + v[base + 1] * v[base + 1];
            if mag > best_mag {
                best_mag = mag;
                best_row = row;
            }
        }
        if best_mag <= zero {
            continue;
        }
        // Pivot p = a + bi; |p| = sqrt(best_mag). Rotating by e^{-i phi} where
        // phi = arg(p) makes the pivot real-positive: every component is
        // multiplied by conj(p)/|p| = (a - bi)/|p|.
        let base_p = 2 * (best_row * n + col);
        let a = v[base_p];
        let b = v[base_p + 1];
        let mag = best_mag.sqrt();
        let cr = a / mag; // real part of unit phase rotation conj(p)/|p|
        let ci = -b / mag; // imag part
        for row in 0..n {
            let base = 2 * (row * n + col);
            let re = v[base];
            let im = v[base + 1];
            // (re + im i) * (cr + ci i) = (re*cr - im*ci) + (re*ci + im*cr) i
            v[base] = re * cr - im * ci;
            v[base + 1] = re * ci + im * cr;
        }
    }
}

/// General eigenvalues only (complex, encoded `[n, 2]`).
///
/// Mirrors `torch.linalg.eigvals`. CPU-only today.
pub fn eigvals<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    require_cpu(a, "eigvals")?;
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("eigvals requires a square 2-D tensor, got {shape:?}"),
        });
    }
    let n = shape[0];

    // Autograd path (CPU): delegate to the differentiable wrapper, which
    // computes the forward (and the eigenvectors the VJP needs) under `no_grad`
    // (preventing re-entry here) and attaches `EigvalsBackward` (complex VJP,
    // real `A.grad` via `at::real`). #1345.
    if crate::autograd::no_grad::is_grad_enabled() && a.requires_grad() {
        return crate::grad_fns::linalg::eigvals_differentiable(a);
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let w = ferray_linalg::eigvals(&arr).map_err(FerrotorchError::Ferray)?;
        let data: Vec<T> = w
            .as_slice()
            .unwrap()
            .iter()
            .flat_map(|c| [T::from(c.re).unwrap(), T::from(c.im).unwrap()])
            .collect();
        Tensor::from_storage(TensorStorage::cpu(data), vec![n, 2], false)
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let w = ferray_linalg::eigvals(&arr).map_err(FerrotorchError::Ferray)?;
        let data: Vec<T> = w
            .as_slice()
            .unwrap()
            .iter()
            .flat_map(|c| [T::from(c.re).unwrap(), T::from(c.im).unwrap()])
            .collect();
        Tensor::from_storage(TensorStorage::cpu(data), vec![n, 2], false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

// ===========================================================================
// LU decomposition
// ===========================================================================

/// LU decomposition with partial pivoting: `A = P L U`.
///
/// Returns `(P, L, U)` where `P` is the permutation matrix (m × m), `L`
/// is unit-lower-triangular (m × k), and `U` is upper-triangular (k × n)
/// with `k = min(m, n)`.
///
/// `P` follows torch's convention (`A = P L U`). ferray's `lu` returns the
/// permutation satisfying `P_f A = L U` (the inverse/transpose of torch's
/// `P`), so the matrix is transposed before returning (CORE-144 / #1838) —
/// for non-involutory pivot sequences (any 3-cycle) the two conventions
/// genuinely differ.
///
/// Mirrors `torch.linalg.lu`. CPU uses ferray; CUDA f32/f64 uses cuSOLVER
/// `getrf` through [`lu_factor`] and unpacks on device.
pub fn lu<T: Float>(a: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>, Tensor<T>)> {
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("lu requires a 2-D tensor, got {shape:?}"),
        });
    }
    let (m, n) = (shape[0], shape[1]);

    // Autograd path: delegate to the differentiable wrapper, which computes the
    // forward under `no_grad` (preventing re-entry here) and attaches the split
    // `LuBackwardL` / `LuBackwardU` nodes.
    if crate::autograd::no_grad::is_grad_enabled() && a.requires_grad() {
        return crate::grad_fns::linalg::lu_differentiable(a);
    }

    if a.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let (lu_packed, pivots) = lu_factor(a)?;
        return lu_unpack_from_factor(&lu_packed, &pivots);
    }
    if a.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "lu" });
    }

    if m == 0 || n == 0 {
        let k = m.min(n);
        return Ok((
            rectangular_eye_on_device(m, m, Device::Cpu)?,
            rectangular_eye_on_device(m, k, Device::Cpu)?,
            full_like_on_device(&[k, n], <T as num_traits::Zero>::zero(), Device::Cpu, "lu")?,
        ));
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let (p, l, u) = ferray_linalg::lu(&arr).map_err(FerrotorchError::Ferray)?;
        // ferray convention: `P_f A = L U`. torch convention (this fn's
        // contract): `A = P L U` with `P = P_f^T` (CORE-144 / #1838).
        let p_data = transpose_square_to_vec_f32::<T>(p.as_slice().unwrap(), p.shape()[0]);
        let l_data = slice_f32_to_vec::<T>(l.as_slice().unwrap());
        let u_data = slice_f32_to_vec::<T>(u.as_slice().unwrap());
        Ok((
            Tensor::from_storage(TensorStorage::cpu(p_data), p.shape().to_vec(), false)?,
            Tensor::from_storage(TensorStorage::cpu(l_data), l.shape().to_vec(), false)?,
            Tensor::from_storage(TensorStorage::cpu(u_data), u.shape().to_vec(), false)?,
        ))
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let (p, l, u) = ferray_linalg::lu(&arr).map_err(FerrotorchError::Ferray)?;
        // ferray convention: `P_f A = L U`. torch convention (this fn's
        // contract): `A = P L U` with `P = P_f^T` (CORE-144 / #1838).
        let p_data = transpose_square_to_vec_f64::<T>(p.as_slice().unwrap(), p.shape()[0]);
        let l_data = slice_to_vec::<T>(l.as_slice().unwrap());
        let u_data = slice_to_vec::<T>(u.as_slice().unwrap());
        Ok((
            Tensor::from_storage(TensorStorage::cpu(p_data), p.shape().to_vec(), false)?,
            Tensor::from_storage(TensorStorage::cpu(l_data), l.shape().to_vec(), false)?,
            Tensor::from_storage(TensorStorage::cpu(u_data), u.shape().to_vec(), false)?,
        ))
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

fn rectangular_eye_on_device<T: Float>(
    rows: usize,
    cols: usize,
    device: Device,
) -> FerrotorchResult<Tensor<T>> {
    let numel = checked_product(&[rows, cols], "lu_unpack")?;
    let mut data = vec![<T as num_traits::Zero>::zero(); numel];
    let one = <T as num_traits::One>::one();
    for i in 0..rows.min(cols) {
        data[i * cols + i] = one;
    }
    Tensor::from_storage(
        TensorStorage::on_device(data, device)?,
        vec![rows, cols],
        false,
    )
}

fn permutation_from_lapack_pivots<T: Float>(
    pivots: &[i32],
    rows: usize,
    device: Device,
) -> FerrotorchResult<Tensor<T>> {
    let numel = checked_product(&[rows, rows], "lu_unpack")?;
    let mut p_f = vec![<T as num_traits::Zero>::zero(); numel];
    let one = <T as num_traits::One>::one();
    for i in 0..rows {
        p_f[i * rows + i] = one;
    }
    for (i, &pivot) in pivots.iter().enumerate() {
        if pivot <= 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("lu_unpack: pivot at index {i} is not 1-based positive: {pivot}"),
            });
        }
        let j = (pivot - 1) as usize;
        if i >= rows || j >= rows {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "lu_unpack: pivot at index {i} points to row {j}, outside {rows} rows"
                ),
            });
        }
        for col in 0..rows {
            p_f.swap(i * rows + col, j * rows + col);
        }
    }

    // LAPACK/cuSOLVER pivots encode P_f such that P_f @ A = L @ U. PyTorch's
    // `torch.linalg.lu` returns P in the documented A = P @ L @ U convention,
    // so expose P_f^T.
    let mut p_torch = vec![<T as num_traits::Zero>::zero(); numel];
    for r in 0..rows {
        for c in 0..rows {
            p_torch[r * rows + c] = p_f[c * rows + r];
        }
    }
    Tensor::from_storage(
        TensorStorage::on_device(p_torch, device)?,
        vec![rows, rows],
        false,
    )
}

pub(crate) fn lu_unpack_from_factor<T: Float>(
    lu_packed: &Tensor<T>,
    pivots: &[i32],
) -> FerrotorchResult<(Tensor<T>, Tensor<T>, Tensor<T>)> {
    let shape = lu_packed.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("lu_unpack: LU must be 2-D, got {shape:?}"),
        });
    }
    let (m, n) = (shape[0], shape[1]);
    let k = m.min(n);
    if pivots.len() != k {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "lu_unpack: expected {} pivots for LU shape {:?}, got {}",
                k,
                shape,
                pivots.len()
            ),
        });
    }
    let device = lu_packed.device();
    let p = permutation_from_lapack_pivots::<T>(pivots, m, device)?;

    if k == 0 {
        let l = rectangular_eye_on_device(m, k, device)?;
        let u = full_like_on_device(
            &[k, n],
            <T as num_traits::Zero>::zero(),
            device,
            "lu_unpack",
        )?;
        return Ok((p, l, u));
    }

    let l_cols = lu_packed.narrow(1, 0, k)?;
    let strict_l = crate::ops::tensor_ops::tril(&l_cols, -1)?;
    let l = strict_l.add_t(&rectangular_eye_on_device(m, k, device)?)?;

    let u_rows = lu_packed.narrow(0, 0, k)?;
    let u = crate::ops::tensor_ops::triu(&u_rows, 0)?;
    Ok((p, l, u))
}

/// Transpose an `n × n` row-major f32 slice into a `Vec<T>` (used to convert
/// ferray's `P_f A = L U` permutation into torch's `A = P L U` one; a
/// permutation matrix's inverse is its transpose). CORE-144 / #1838.
fn transpose_square_to_vec_f32<T: Float>(src: &[f32], n: usize) -> Vec<T> {
    let mut out = vec![<T as num_traits::Zero>::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = T::from(src[j * n + i]).unwrap();
        }
    }
    out
}

/// See [`transpose_square_to_vec_f32`] — f64 source variant.
fn transpose_square_to_vec_f64<T: Float>(src: &[f64], n: usize) -> Vec<T> {
    let mut out = vec![<T as num_traits::Zero>::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = T::from(src[j * n + i]).unwrap();
        }
    }
    out
}

/// LU factorization in cuSOLVER's packed form: returns `(LU_packed, pivots)`
/// where `LU_packed` has the same `m×n` shape as `a`, the strict lower
/// triangle stores `L` (unit diagonal implicit), the upper triangle stores
/// `U`, and `pivots` is a length-`min(m, n)` host `Vec<i32>` of 1-based
/// row-permutation indices (cuSOLVER /
/// LAPACK convention). Mirrors `torch.linalg.lu_factor`. (#604)
///
/// On CUDA f32/f64, dispatches to the native `gpu_lu_factor` kernel
/// (cuSOLVER `getrf` with on-device row→col→row transpose). The LU matrix
/// stays on device (O(mn) values); only the pivot vector (O(min(m,n)) ints) is
/// downloaded to host as a `Vec<i32>` since `Tensor<T>` requires
/// `T: Float`. Other dtypes and CPU inputs fall back to `ferray-linalg::lu`
/// and pack the result locally.
pub fn lu_factor<T: Float>(a: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Vec<i32>)> {
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("lu_factor requires a 2-D tensor, got {shape:?}"),
        });
    }
    let (m, n) = (shape[0], shape[1]);
    let k = m.min(n);

    // Autograd path: delegate to the differentiable wrapper. The forward value
    // still follows the resident CUDA or CPU path under `no_grad`.
    if crate::autograd::no_grad::is_grad_enabled() && a.requires_grad() {
        return crate::grad_fns::linalg::lu_factor_differentiable(a);
    }

    if k == 0 {
        let lu = full_like_on_device(
            shape,
            <T as num_traits::Zero>::zero(),
            a.device(),
            "lu_factor",
        )?;
        return Ok((lu, Vec::new()));
    }

    if a.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let (lu_h, ipiv) = if is_f32::<T>() {
            backend.lu_factor_f32(a.gpu_handle()?, m, n)?
        } else {
            backend.lu_factor_f64(a.gpu_handle()?, m, n)?
        };
        // The LU matrix stays on device; pivots are returned as a host
        // Vec<i32> directly from the trait (O(min(m,n)) ints, not worth a
        // typed GPU int handle).
        let lu = Tensor::from_storage(TensorStorage::gpu(lu_h), vec![m, n], false)?;
        return Ok((lu, ipiv));
    }
    if a.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "lu_factor" });
    }

    // CPU fallback: get full PLU from ferray-linalg, then collapse to packed
    // LU + pivots (the cuSOLVER convention). The packed form is L's strict
    // lower triangle plus U's upper triangle (incl. diagonal); ipiv comes
    // from the row permutation P encoded as 1-based indices.
    let (p, l, u) = if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let (p, l, u) = ferray_linalg::lu(&arr).map_err(FerrotorchError::Ferray)?;
        (
            slice_f32_to_vec::<T>(p.as_slice().unwrap()),
            slice_f32_to_vec::<T>(l.as_slice().unwrap()),
            slice_f32_to_vec::<T>(u.as_slice().unwrap()),
        )
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let (p, l, u) = ferray_linalg::lu(&arr).map_err(FerrotorchError::Ferray)?;
        (
            slice_to_vec::<T>(p.as_slice().unwrap()),
            slice_to_vec::<T>(l.as_slice().unwrap()),
            slice_to_vec::<T>(u.as_slice().unwrap()),
        )
    } else {
        return Err(FerrotorchError::InvalidArgument {
            message: "lu_factor requires f32 or f64".into(),
        });
    };

    // Build packed LU buffer: lower triangle = strict-L, upper = U (incl. diag).
    let mut packed = vec![<T as num_traits::Zero>::zero(); m * n];
    for i in 0..m {
        for j in 0..n {
            packed[i * n + j] = if j < k && j < i {
                l[i * k + j] // strict lower of L
            } else if i < k {
                u[i * n + j] // U upper triangle
            } else {
                <T as num_traits::Zero>::zero()
            };
        }
    }
    // Convert P (an n×n permutation matrix) to ipiv in cuSOLVER /
    // LAPACK swap-sequence form so the CPU and GPU paths produce
    // interchangeable output. cuSOLVER's `ipiv[i]` (1-based) is the
    // index of the row swapped INTO position `i` at step `i` of the
    // factorization.
    //
    // Two-step conversion:
    //   1. Read P as a permutation vector `perm` where `perm[i]` is the
    //      column with a 1 in row `i` of P (i.e. row `i` of `P @ A`
    //      equals row `perm[i]` of `A`).
    //   2. Convert `perm` → swap-sequence by replaying the swaps. At
    //      step `i`, the algorithm wants `perm[i]` at position `i`.
    //      Find where `perm[i]` lives in the running array `work`
    //      (originally identity), record the swap, and apply it.
    let mut perm = vec![0_usize; m];
    let one = T::from(1.0).unwrap();
    for i in 0..m {
        for j in 0..m {
            if p[i * m + j] == one {
                perm[i] = j;
                break;
            }
        }
    }
    let mut work: Vec<usize> = (0..m).collect();
    let mut ipiv = vec![0_i32; k];
    for i in 0..k {
        let target = perm[i];
        let j = (i..m).find(|&row| work[row] == target).unwrap_or(i);
        ipiv[i] = (j + 1) as i32;
        work.swap(i, j);
    }
    let lu = Tensor::from_storage(TensorStorage::cpu(packed), vec![m, n], false)?;
    Ok((lu, ipiv))
}

// ===========================================================================
// Singular values only / least squares
// ===========================================================================

/// Singular values (descending) of a 2-D tensor.
///
/// Mirrors `torch.linalg.svdvals`.
pub fn svdvals<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("svdvals requires a 2-D tensor, got {shape:?}"),
        });
    }

    if tracking_enabled_for(&[a]) {
        let (_u, s, _vh) = svd(a)?;
        return Ok(s);
    }

    if a.is_cuda() {
        let (_u, s, _vh) = svd(a)?;
        return Ok(s);
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let s = ferray_linalg::svdvals(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(s.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), s.shape().to_vec(), false)
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let s = ferray_linalg::svdvals(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(s.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), s.shape().to_vec(), false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct LstsqDims {
    m: usize,
    n: usize,
    b_is_1d: bool,
    nrhs: usize,
}

fn validate_lstsq_inputs<T: Float>(
    op: &str,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> FerrotorchResult<LstsqDims> {
    if a.ndim() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("{op}: `a` must be 2-D, got {:?}", a.shape()),
        });
    }
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }
    let m = a.shape()[0];
    let n = a.shape()[1];
    let (b_is_1d, nrhs) = match b.ndim() {
        1 if b.shape()[0] == m => (true, 1),
        2 if b.shape()[0] == m => (false, b.shape()[1]),
        _ => {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "{op}: `b` must be 1-D [{m}] or 2-D [{m}, K], got {:?}",
                    b.shape()
                ),
            });
        }
    };
    Ok(LstsqDims {
        m,
        n,
        b_is_1d,
        nrhs,
    })
}

fn empty_rank(device: Device) -> FerrotorchResult<IntTensor<i64>> {
    let rank = IntTensor::<i64>::from_vec(Vec::new(), vec![0])?;
    if device == Device::Cpu {
        Ok(rank)
    } else {
        rank.to(device)
    }
}

fn scalar_rank(rank: usize, device: Device) -> FerrotorchResult<IntTensor<i64>> {
    let rank_i64 = i64::try_from(rank).map_err(|_| FerrotorchError::InvalidArgument {
        message: format!("lstsq: rank {rank} does not fit in i64"),
    })?;
    let rank = IntTensor::<i64>::from_vec(vec![rank_i64], vec![])?;
    if device == Device::Cpu {
        Ok(rank)
    } else {
        rank.to(device)
    }
}

fn empty_float_tensor<T: Float>(device: Device) -> FerrotorchResult<Tensor<T>> {
    Tensor::from_storage(
        TensorStorage::on_device(Vec::new(), device)?,
        vec![0],
        false,
    )
}

fn default_lstsq_rcond<T: Float>(dims: LstsqDims) -> FerrotorchResult<f64> {
    let eps = if is_f32::<T>() {
        f32::EPSILON as f64
    } else if is_f64::<T>() {
        f64::EPSILON
    } else {
        return Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        });
    };
    Ok(eps * dims.m.max(dims.n) as f64)
}

fn check_lapack_lstsq_info(info: c_int, driver: LstsqDriver) -> FerrotorchResult<()> {
    if info == 0 {
        return Ok(());
    }
    if info < 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "lstsq({}): LAPACKE reported illegal argument {}",
                driver.name(),
                -info
            ),
        });
    }
    let message = match driver {
        LstsqDriver::Gels => format!(
            "lstsq(gels): the least-squares solution could not be computed \
             because the input matrix does not have full rank (error code: {info})"
        ),
        LstsqDriver::Gelsd | LstsqDriver::Gelss => format!(
            "lstsq({}): SVD-based LAPACK driver failed to converge (error code: {info})",
            driver.name()
        ),
        LstsqDriver::Gelsy => format!(
            "lstsq(gelsy): complete orthogonal-factorization driver failed \
             (error code: {info})"
        ),
    };
    Err(FerrotorchError::InvalidArgument { message })
}

fn pack_b_f64<T: Float>(b: &Tensor<T>, dims: LstsqDims) -> FerrotorchResult<Vec<f64>> {
    let b_data = b.data_vec()?;
    let rows = dims.m.max(dims.n);
    let mut out = vec![0.0; rows.saturating_mul(dims.nrhs)];
    if dims.b_is_1d {
        for i in 0..dims.m {
            out[i * dims.nrhs] = b_data[i].to_f64().unwrap();
        }
    } else {
        for i in 0..dims.m {
            for j in 0..dims.nrhs {
                out[i * dims.nrhs + j] = b_data[i * dims.nrhs + j].to_f64().unwrap();
            }
        }
    }
    Ok(out)
}

fn pack_b_f32<T: Float>(b: &Tensor<T>, dims: LstsqDims) -> FerrotorchResult<Vec<f32>> {
    let b_data = b.data_vec()?;
    let rows = dims.m.max(dims.n);
    let mut out = vec![0.0; rows.saturating_mul(dims.nrhs)];
    if dims.b_is_1d {
        for i in 0..dims.m {
            out[i * dims.nrhs] = b_data[i].to_f64().unwrap() as f32;
        }
    } else {
        for i in 0..dims.m {
            for j in 0..dims.nrhs {
                out[i * dims.nrhs + j] = b_data[i * dims.nrhs + j].to_f64().unwrap() as f32;
            }
        }
    }
    Ok(out)
}

fn solution_shape(dims: LstsqDims) -> Vec<usize> {
    if dims.b_is_1d {
        vec![dims.n]
    } else {
        vec![dims.n, dims.nrhs]
    }
}

fn residuals_f64<T: Float>(
    b_work: &[f64],
    dims: LstsqDims,
    driver: LstsqDriver,
    rank: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    let compute = dims.m > dims.n
        && driver != LstsqDriver::Gelsy
        && (driver == LstsqDriver::Gels || rank == Some(dims.n));
    let residuals = if compute {
        let mut out = vec![0.0; dims.nrhs];
        for col in 0..dims.nrhs {
            for row in dims.n..dims.m {
                let r = b_work[row * dims.nrhs + col];
                out[col] += r * r;
            }
        }
        out
    } else {
        Vec::new()
    };
    Tensor::from_storage(
        TensorStorage::cpu(slice_to_vec::<T>(&residuals)),
        vec![residuals.len()],
        false,
    )
}

fn residuals_f32<T: Float>(
    b_work: &[f32],
    dims: LstsqDims,
    driver: LstsqDriver,
    rank: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    let compute = dims.m > dims.n
        && driver != LstsqDriver::Gelsy
        && (driver == LstsqDriver::Gels || rank == Some(dims.n));
    let residuals = if compute {
        let mut out = vec![0.0; dims.nrhs];
        for col in 0..dims.nrhs {
            for row in dims.n..dims.m {
                let r = b_work[row * dims.nrhs + col];
                out[col] += r * r;
            }
        }
        out
    } else {
        Vec::new()
    };
    Tensor::from_storage(
        TensorStorage::cpu(slice_f32_to_vec::<T>(&residuals)),
        vec![residuals.len()],
        false,
    )
}

fn solution_from_b_f64<T: Float>(b_work: &[f64], dims: LstsqDims) -> FerrotorchResult<Tensor<T>> {
    let mut out = vec![0.0; dims.n.saturating_mul(dims.nrhs)];
    for row in 0..dims.n {
        for col in 0..dims.nrhs {
            out[row * dims.nrhs + col] = b_work[row * dims.nrhs + col];
        }
    }
    Tensor::from_storage(
        TensorStorage::cpu(slice_to_vec::<T>(&out)),
        solution_shape(dims),
        false,
    )
}

fn solution_from_b_f32<T: Float>(b_work: &[f32], dims: LstsqDims) -> FerrotorchResult<Tensor<T>> {
    let mut out = vec![0.0; dims.n.saturating_mul(dims.nrhs)];
    for row in 0..dims.n {
        for col in 0..dims.nrhs {
            out[row * dims.nrhs + col] = b_work[row * dims.nrhs + col];
        }
    }
    Tensor::from_storage(
        TensorStorage::cpu(slice_f32_to_vec::<T>(&out)),
        solution_shape(dims),
        false,
    )
}

fn lstsq_degenerate_f64<T: Float>(
    b: &Tensor<T>,
    dims: LstsqDims,
    driver: LstsqDriver,
) -> FerrotorchResult<LstsqResult<T>> {
    let sol = Tensor::from_storage(
        TensorStorage::cpu(vec![
            <T as ferray_core::Element>::zero();
            dims.n.saturating_mul(dims.nrhs)
        ]),
        solution_shape(dims),
        false,
    )?;
    let b_work = pack_b_f64(b, dims)?;
    let residuals = residuals_f64(&b_work, dims, driver, Some(0))?;
    let rank = if driver == LstsqDriver::Gels {
        empty_rank(Device::Cpu)?
    } else {
        scalar_rank(0, Device::Cpu)?
    };
    let sv_len = if matches!(driver, LstsqDriver::Gelsd | LstsqDriver::Gelss) {
        dims.m.min(dims.n)
    } else {
        0
    };
    let sv = Tensor::from_storage(
        TensorStorage::cpu(vec![<T as ferray_core::Element>::zero(); sv_len]),
        vec![sv_len],
        false,
    )?;
    Ok((sol, residuals, rank, sv))
}

fn lstsq_degenerate_f32<T: Float>(
    b: &Tensor<T>,
    dims: LstsqDims,
    driver: LstsqDriver,
) -> FerrotorchResult<LstsqResult<T>> {
    let sol = Tensor::from_storage(
        TensorStorage::cpu(vec![
            <T as ferray_core::Element>::zero();
            dims.n.saturating_mul(dims.nrhs)
        ]),
        solution_shape(dims),
        false,
    )?;
    let b_work = pack_b_f32(b, dims)?;
    let residuals = residuals_f32(&b_work, dims, driver, Some(0))?;
    let rank = if driver == LstsqDriver::Gels {
        empty_rank(Device::Cpu)?
    } else {
        scalar_rank(0, Device::Cpu)?
    };
    let sv_len = if matches!(driver, LstsqDriver::Gelsd | LstsqDriver::Gelss) {
        dims.m.min(dims.n)
    } else {
        0
    };
    let sv = Tensor::from_storage(
        TensorStorage::cpu(vec![<T as ferray_core::Element>::zero(); sv_len]),
        vec![sv_len],
        false,
    )?;
    Ok((sol, residuals, rank, sv))
}

fn cpu_lapack_lstsq_f64<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    dims: LstsqDims,
    rcond: Option<f64>,
    driver: LstsqDriver,
) -> FerrotorchResult<LstsqResult<T>> {
    if dims.m == 0 || dims.n == 0 {
        return lstsq_degenerate_f64(b, dims, driver);
    }
    let backend = lapacke_backend()?;
    let mut a_work: Vec<f64> = a.data_vec()?.iter().map(|v| v.to_f64().unwrap()).collect();
    let mut b_work = pack_b_f64(b, dims)?;
    let m = checked_lapack_i32(dims.m, "m")?;
    let n = checked_lapack_i32(dims.n, "n")?;
    let nrhs = checked_lapack_i32(dims.nrhs, "nrhs")?;
    let lda = checked_lapack_i32(dims.n.max(1), "lda")?;
    let ldb = checked_lapack_i32(dims.nrhs.max(1), "ldb")?;
    let rcond = rcond.unwrap_or(default_lstsq_rcond::<T>(dims)?);
    let trans = b'N' as c_char;

    let mut rank_i32 = 0;
    let mut singular_values = Vec::<f64>::new();
    let info = unsafe {
        match driver {
            LstsqDriver::Gels => (backend.dgels)(
                LAPACK_ROW_MAJOR,
                trans,
                m,
                n,
                nrhs,
                a_work.as_mut_ptr(),
                lda,
                b_work.as_mut_ptr(),
                ldb,
            ),
            LstsqDriver::Gelsy => {
                let mut jpvt = vec![0 as c_int; dims.n];
                (backend.dgelsy)(
                    LAPACK_ROW_MAJOR,
                    m,
                    n,
                    nrhs,
                    a_work.as_mut_ptr(),
                    lda,
                    b_work.as_mut_ptr(),
                    ldb,
                    jpvt.as_mut_ptr(),
                    rcond,
                    std::ptr::addr_of_mut!(rank_i32),
                )
            }
            LstsqDriver::Gelsd => {
                singular_values.resize(dims.m.min(dims.n), 0.0);
                (backend.dgelsd)(
                    LAPACK_ROW_MAJOR,
                    m,
                    n,
                    nrhs,
                    a_work.as_mut_ptr(),
                    lda,
                    b_work.as_mut_ptr(),
                    ldb,
                    singular_values.as_mut_ptr(),
                    rcond,
                    std::ptr::addr_of_mut!(rank_i32),
                )
            }
            LstsqDriver::Gelss => {
                singular_values.resize(dims.m.min(dims.n), 0.0);
                (backend.dgelss)(
                    LAPACK_ROW_MAJOR,
                    m,
                    n,
                    nrhs,
                    a_work.as_mut_ptr(),
                    lda,
                    b_work.as_mut_ptr(),
                    ldb,
                    singular_values.as_mut_ptr(),
                    rcond,
                    std::ptr::addr_of_mut!(rank_i32),
                )
            }
        }
    };
    check_lapack_lstsq_info(info, driver)?;

    let rank = if driver == LstsqDriver::Gels {
        None
    } else {
        Some(
            usize::try_from(rank_i32).map_err(|_| FerrotorchError::InvalidArgument {
                message: format!(
                    "lstsq({}): LAPACK returned negative rank {rank_i32}",
                    driver.name()
                ),
            })?,
        )
    };
    let sol = solution_from_b_f64(&b_work, dims)?;
    let residuals = residuals_f64(&b_work, dims, driver, rank)?;
    let rank_tensor = match rank {
        Some(rank) => scalar_rank(rank, Device::Cpu)?,
        None => empty_rank(Device::Cpu)?,
    };
    let sv = Tensor::from_storage(
        TensorStorage::cpu(slice_to_vec::<T>(&singular_values)),
        vec![singular_values.len()],
        false,
    )?;
    Ok((sol, residuals, rank_tensor, sv))
}

fn cpu_lapack_lstsq_f32<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    dims: LstsqDims,
    rcond: Option<f64>,
    driver: LstsqDriver,
) -> FerrotorchResult<LstsqResult<T>> {
    if dims.m == 0 || dims.n == 0 {
        return lstsq_degenerate_f32(b, dims, driver);
    }
    let backend = lapacke_backend()?;
    let mut a_work: Vec<f32> = a
        .data_vec()?
        .iter()
        .map(|v| v.to_f64().unwrap() as f32)
        .collect();
    let mut b_work = pack_b_f32(b, dims)?;
    let m = checked_lapack_i32(dims.m, "m")?;
    let n = checked_lapack_i32(dims.n, "n")?;
    let nrhs = checked_lapack_i32(dims.nrhs, "nrhs")?;
    let lda = checked_lapack_i32(dims.n.max(1), "lda")?;
    let ldb = checked_lapack_i32(dims.nrhs.max(1), "ldb")?;
    let rcond = rcond.unwrap_or(default_lstsq_rcond::<T>(dims)?) as f32;
    let trans = b'N' as c_char;

    let mut rank_i32 = 0;
    let mut singular_values = Vec::<f32>::new();
    let info = unsafe {
        match driver {
            LstsqDriver::Gels => (backend.sgels)(
                LAPACK_ROW_MAJOR,
                trans,
                m,
                n,
                nrhs,
                a_work.as_mut_ptr(),
                lda,
                b_work.as_mut_ptr(),
                ldb,
            ),
            LstsqDriver::Gelsy => {
                let mut jpvt = vec![0 as c_int; dims.n];
                (backend.sgelsy)(
                    LAPACK_ROW_MAJOR,
                    m,
                    n,
                    nrhs,
                    a_work.as_mut_ptr(),
                    lda,
                    b_work.as_mut_ptr(),
                    ldb,
                    jpvt.as_mut_ptr(),
                    rcond,
                    std::ptr::addr_of_mut!(rank_i32),
                )
            }
            LstsqDriver::Gelsd => {
                singular_values.resize(dims.m.min(dims.n), 0.0);
                (backend.sgelsd)(
                    LAPACK_ROW_MAJOR,
                    m,
                    n,
                    nrhs,
                    a_work.as_mut_ptr(),
                    lda,
                    b_work.as_mut_ptr(),
                    ldb,
                    singular_values.as_mut_ptr(),
                    rcond,
                    std::ptr::addr_of_mut!(rank_i32),
                )
            }
            LstsqDriver::Gelss => {
                singular_values.resize(dims.m.min(dims.n), 0.0);
                (backend.sgelss)(
                    LAPACK_ROW_MAJOR,
                    m,
                    n,
                    nrhs,
                    a_work.as_mut_ptr(),
                    lda,
                    b_work.as_mut_ptr(),
                    ldb,
                    singular_values.as_mut_ptr(),
                    rcond,
                    std::ptr::addr_of_mut!(rank_i32),
                )
            }
        }
    };
    check_lapack_lstsq_info(info, driver)?;

    let rank = if driver == LstsqDriver::Gels {
        None
    } else {
        Some(
            usize::try_from(rank_i32).map_err(|_| FerrotorchError::InvalidArgument {
                message: format!(
                    "lstsq({}): LAPACK returned negative rank {rank_i32}",
                    driver.name()
                ),
            })?,
        )
    };
    let sol = solution_from_b_f32(&b_work, dims)?;
    let residuals = residuals_f32(&b_work, dims, driver, rank)?;
    let rank_tensor = match rank {
        Some(rank) => scalar_rank(rank, Device::Cpu)?,
        None => empty_rank(Device::Cpu)?,
    };
    let sv = Tensor::from_storage(
        TensorStorage::cpu(slice_f32_to_vec::<T>(&singular_values)),
        vec![singular_values.len()],
        false,
    )?;
    Ok((sol, residuals, rank_tensor, sv))
}

/// Least-squares solution `X` minimizing `||A X - B||_F`. Just the
/// solution — no residuals / rank / singular values. (#630)
///
/// On CUDA f32/f64, dispatches to cuSOLVER `cusolverDnSSgels` /
/// `cusolverDnDDgels` (iterative refinement, no host bounce). CPU and
/// other dtypes route through `ferray-linalg::lstsq` and discard the
/// extra outputs. `A` is `[M, N]`; `B` is `[M, K]` (or `[M]` treated as
/// `[M, 1]`); output is `[N, K]` (or `[N]` for the 1-D case).
pub fn lstsq_solve<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let dims = validate_lstsq_inputs("lstsq_solve", a, b)?;

    if crate::autograd::no_grad::is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        return crate::grad_fns::linalg::lstsq_solve_differentiable(a, b);
    }

    if a.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let x_h = if is_f32::<T>() {
            backend.lstsq_f32(a.gpu_handle()?, b.gpu_handle()?, dims.m, dims.n, dims.nrhs)?
        } else {
            backend.lstsq_f64(a.gpu_handle()?, b.gpu_handle()?, dims.m, dims.n, dims.nrhs)?
        };
        let out_shape = solution_shape(dims);
        return Tensor::from_storage(TensorStorage::gpu(x_h), out_shape, false);
    }
    if a.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "lstsq_solve" });
    }

    // CPU: route through full ferray lstsq and take the solution slice.
    let (sol, _r, _rank, _sv) = lstsq(a, b, None)?;
    Ok(sol)
}

/// Least-squares solution to `A x ≈ b` using PyTorch's default driver for the device.
///
/// Returns `(solution, residuals, rank, singular_values)`. `rcond`
/// controls the singular-value cutoff for rank determination; if `None`,
/// uses a sensible default (`max(m, n) * eps`).
///
/// Mirrors `torch.linalg.lstsq` for the crate's 2-D, real-valued surface:
/// CPU defaults to `gelsy`, CUDA defaults to `gels`, and `rank` is an
/// `IntTensor<i64>`.
pub fn lstsq<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    rcond: Option<f64>,
) -> FerrotorchResult<LstsqResult<T>> {
    lstsq_with_driver(a, b, rcond, None)
}

/// Least-squares solution to `A x ≈ b` with explicit PyTorch-compatible driver
/// selection.
pub fn lstsq_with_driver<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    rcond: Option<f64>,
    driver: Option<LstsqDriver>,
) -> FerrotorchResult<LstsqResult<T>> {
    let dims = validate_lstsq_inputs("lstsq", a, b)?;
    let driver = driver.unwrap_or_else(|| LstsqDriver::default_for_device(a.device()));

    // Autograd path: delegate to the differentiable wrapper, which computes the
    // forward under `no_grad` (preventing re-entry here) and attaches
    // `LstsqBackward` to the differentiable `solution` and `residuals` outputs
    // (full-rank, via pinv_backward plus PyTorch's residual branch).
    if crate::autograd::no_grad::is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        return crate::grad_fns::linalg::lstsq_differentiable(a, b, rcond, Some(driver));
    }

    if a.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        if driver != LstsqDriver::Gels {
            return Err(FerrotorchError::InvalidArgument {
                message: "torch.linalg.lstsq: `driver` other than `gels` is not supported on CUDA"
                    .into(),
            });
        }
        let sol = lstsq_solve(a, b)?;
        let residuals = if dims.m > dims.n {
            let sol_m = if dims.b_is_1d {
                sol.view_reshape(vec![dims.n, 1])?
            } else {
                sol.clone()
            };
            let b_m = if dims.b_is_1d {
                b.view_reshape(vec![dims.m, 1])?
            } else {
                b.clone()
            };
            let r = a.mm(&sol_m)?.sub_t(&b_m)?;
            let r2 = r.mul_t(&r)?;
            let residuals = crate::grad_fns::reduction::sum_dim(&r2, 0, false)?;
            debug_assert_eq!(residuals.shape(), &[dims.nrhs]);
            residuals
        } else {
            empty_float_tensor(a.device())?
        };
        let rank = empty_rank(a.device())?;
        let sv = empty_float_tensor(a.device())?;
        return Ok((sol, residuals, rank, sv));
    }
    if a.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "lstsq" });
    }
    require_cpu(a, "lstsq")?;
    require_cpu(b, "lstsq")?;

    if is_f32::<T>() {
        cpu_lapack_lstsq_f32(a, b, dims, rcond, driver)
    } else if is_f64::<T>() {
        cpu_lapack_lstsq_f64(a, b, dims, rcond, driver)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

// ===========================================================================
// Higher-order solvers (matrix_power, tensorsolve, tensorinv)
// ===========================================================================

/// Compute `A^n` for integer `n`. For `n >= 0`, uses repeated squaring;
/// for `n < 0`, computes the inverse first.
///
/// Mirrors `torch.linalg.matrix_power`.
pub fn matrix_power<T: Float>(a: &Tensor<T>, n: i64) -> FerrotorchResult<Tensor<T>> {
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("matrix_power requires a square 2-D tensor, got {shape:?}"),
        });
    }

    if tracking_enabled_for(&[a]) || a.is_cuda() {
        return matrix_power_composite(a, n);
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let r = ferray_linalg::matrix_power(&arr, n).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(r.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), r.shape().to_vec(), false)
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let r = ferray_linalg::matrix_power(&arr, n).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(r.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), r.shape().to_vec(), false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

fn matrix_power_composite<T: Float>(a: &Tensor<T>, n: i64) -> FerrotorchResult<Tensor<T>> {
    let shape = a.shape();
    let dim = shape[0];
    let device = a.device();
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();

    if n == 0 {
        let zeros = full_like_on_device(shape, zero, device, "matrix_power")?;
        let eye = eye_on_device(dim, device)?;
        return a.mul_t(&zeros)?.add_t(&eye);
    }
    if n == 1 {
        let ones = full_like_on_device(shape, one, device, "matrix_power")?;
        return a.mul_t(&ones);
    }

    let mut exp = n
        .checked_abs()
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "matrix_power: exponent i64::MIN is not representable as a positive power"
                .into(),
        })? as u64;

    let base = if n < 0 {
        let eye = eye_on_device(dim, device)?;
        solve(a, &eye)?
    } else {
        a.clone()
    };

    if exp == 1 {
        return Ok(base);
    }
    if exp == 2 {
        return base.mm(&base);
    }
    if exp == 3 {
        return base.mm(&base)?.mm(&base);
    }

    let mut z: Option<Tensor<T>> = None;
    let mut result: Option<Tensor<T>> = None;
    while exp > 0 {
        let bit = exp & 1;
        exp >>= 1;
        z = Some(match z {
            Some(ref current) => current.mm(current)?,
            None => base.clone(),
        });
        if bit == 1 {
            let z_ref = z.as_ref().expect("z set before use");
            result = Some(match result {
                Some(ref current) => current.mm(z_ref)?,
                None => z_ref.clone(),
            });
        }
    }

    result.ok_or_else(|| FerrotorchError::Internal {
        message: "matrix_power: exponentiation produced no result".into(),
    })
}

/// Solve `tensordot(a, x, axes) = b` for a tensor `x`.
///
/// Mirrors `torch.linalg.tensorsolve` for the default `dims=None` case.
pub fn tensorsolve<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }
    if b.ndim() > a.ndim() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "tensorsolve: b rank {} cannot exceed a rank {}",
                b.ndim(),
                a.ndim()
            ),
        });
    }
    let result_shape = a.shape()[b.ndim()..].to_vec();
    let result_product = checked_product(&result_shape, "tensorsolve")?;
    let b_product = checked_product(b.shape(), "tensorsolve")?;
    if result_product != b_product {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "tensorsolve: expected prod(a.shape[b.ndim():]) == prod(b.shape), got {result_product} != {b_product}"
            ),
        });
    }

    if tracking_enabled_for(&[a, b]) || a.is_cuda() {
        let a_2d = crate::grad_fns::shape::reshape(
            a,
            &[result_product as isize, result_product as isize],
        )?;
        let b_flat = crate::grad_fns::shape::reshape(b, &[b_product as isize])?;
        let x = solve(&a_2d, &b_flat)?;
        let out_shape: Vec<isize> = result_shape.iter().map(|&d| d as isize).collect();
        return crate::grad_fns::shape::reshape(&x, &out_shape);
    }

    if is_f32::<T>() {
        let a_arr = tensor_to_arraydyn_f32(a)?;
        let b_arr = tensor_to_arraydyn_f32(b)?;
        let x =
            ferray_linalg::tensorsolve(&a_arr, &b_arr, None).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(x.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), x.shape().to_vec(), false)
    } else if is_f64::<T>() {
        let a_arr = tensor_to_arraydyn_f64(a)?;
        let b_arr = tensor_to_arraydyn_f64(b)?;
        let x =
            ferray_linalg::tensorsolve(&a_arr, &b_arr, None).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(x.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), x.shape().to_vec(), false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// Tensor inverse with respect to the partition at `ind`.
///
/// Mirrors `torch.linalg.tensorinv`.
pub fn tensorinv<T: Float>(a: &Tensor<T>, ind: usize) -> FerrotorchResult<Tensor<T>> {
    let shape = a.shape();
    if ind == 0 || ind > shape.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "tensorinv: expected 0 < ind <= ndim ({}), got {ind}",
                shape.len()
            ),
        });
    }
    let left_shape = &shape[..ind];
    let right_shape = &shape[ind..];
    let left_product = checked_product(left_shape, "tensorinv")?;
    let right_product = checked_product(right_shape, "tensorinv")?;
    if left_product != right_product {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "tensorinv: expected prod(shape[..ind]) == prod(shape[ind..]), got {left_product} != {right_product}"
            ),
        });
    }

    if tracking_enabled_for(&[a]) || a.is_cuda() {
        let a_2d =
            crate::grad_fns::shape::reshape(a, &[right_product as isize, right_product as isize])?;
        let eye = eye_on_device(right_product, a.device())?;
        let inv_2d = solve(&a_2d, &eye)?;
        let mut out_shape = right_shape.to_vec();
        out_shape.extend_from_slice(left_shape);
        let out_shape: Vec<isize> = out_shape.iter().map(|&d| d as isize).collect();
        return crate::grad_fns::shape::reshape(&inv_2d, &out_shape);
    }

    if is_f32::<T>() {
        let arr = tensor_to_arraydyn_f32(a)?;
        let inv = ferray_linalg::tensorinv(&arr, ind).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(inv.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), inv.shape().to_vec(), false)
    } else if is_f64::<T>() {
        let arr = tensor_to_arraydyn_f64(a)?;
        let inv = ferray_linalg::tensorinv(&arr, ind).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(inv.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), inv.shape().to_vec(), false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

// ===========================================================================
// Norms (vector / slogdet / matrix_rank / cond)
// ===========================================================================

/// p-norm of a tensor.
///
/// Returns a scalar tensor. `ord` may be `2.0` (L2/Frobenius), `1.0` (L1),
/// `f64::INFINITY`, or any positive real. Matches `torch.linalg.vector_norm`'s
/// scalar reduction (full-tensor) form.
///
/// CPU-only today.
pub fn vector_norm<T: Float>(input: &Tensor<T>, ord: f64) -> FerrotorchResult<Tensor<T>> {
    require_cpu(input, "vector_norm")?;
    let order = float_to_norm_order(ord);

    // Autograd path: delegate to the differentiable wrapper, which attaches
    // `NormBackward` for EVERY accepted `ord` (per-ord `norm_backward`
    // branches; CORE-047 / #1741).
    if crate::autograd::no_grad::is_grad_enabled() && input.requires_grad() {
        return crate::grad_fns::linalg::vector_norm_differentiable(input, ord);
    }

    if is_f32::<T>() {
        let arr = tensor_to_arraydyn_f32(input)?;
        let r = ferray_linalg::vector_norm(&arr, order, None, false)
            .map_err(FerrotorchError::Ferray)?;
        // Result is a 0-d (or 1-d singleton) array.
        let val = T::from(r.as_slice().unwrap()[0]).unwrap();
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)
    } else if is_f64::<T>() {
        let arr = tensor_to_arraydyn_f64(input)?;
        let r = ferray_linalg::vector_norm(&arr, order, None, false)
            .map_err(FerrotorchError::Ferray)?;
        let val = T::from(r.as_slice().unwrap()[0]).unwrap();
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// Sign and natural log of `|det(A)|`.
///
/// Returns `(sign, logabsdet)` as scalar tensors. For singular matrices,
/// `sign` is `0` and `logabsdet` is `-inf`. Mirrors `torch.linalg.slogdet`.
/// CPU-only today.
pub fn slogdet<T: Float>(a: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    require_cpu(a, "slogdet")?;
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("slogdet requires a square 2-D tensor, got {shape:?}"),
        });
    }

    // Autograd path: delegate to the differentiable wrapper, which computes the
    // forward inside `no_grad` (preventing re-entry) and attaches
    // `SlogdetBackward` to the `logabsdet` output.
    if crate::autograd::no_grad::is_grad_enabled() && a.requires_grad() {
        return crate::grad_fns::linalg::slogdet_differentiable(a);
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let (sign, logabs) = ferray_linalg::slogdet(&arr).map_err(FerrotorchError::Ferray)?;
        Ok((
            Tensor::from_storage(
                TensorStorage::cpu(vec![T::from(sign).unwrap()]),
                vec![],
                false,
            )?,
            Tensor::from_storage(
                TensorStorage::cpu(vec![T::from(logabs).unwrap()]),
                vec![],
                false,
            )?,
        ))
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let (sign, logabs) = ferray_linalg::slogdet(&arr).map_err(FerrotorchError::Ferray)?;
        Ok((
            Tensor::from_storage(
                TensorStorage::cpu(vec![T::from(sign).unwrap()]),
                vec![],
                false,
            )?,
            Tensor::from_storage(
                TensorStorage::cpu(vec![T::from(logabs).unwrap()]),
                vec![],
                false,
            )?,
        ))
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// Numerical rank of `a`.
///
/// Returns a scalar (0-d) `i64`-valued tensor encoded as `T`. `tol`, when
/// `Some(t)`, is the absolute tolerance below which singular values are
/// treated as zero; default is `max(m, n) * eps * sigma_max`.
///
/// Mirrors `torch.linalg.matrix_rank`. CPU-only today.
pub fn matrix_rank<T: Float>(a: &Tensor<T>, tol: Option<f64>) -> FerrotorchResult<Tensor<T>> {
    require_cpu(a, "matrix_rank")?;
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("matrix_rank requires a 2-D tensor, got {shape:?}"),
        });
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let r = ferray_linalg::matrix_rank(&arr, tol.map(|t| t as f32))
            .map_err(FerrotorchError::Ferray)?;
        Tensor::from_storage(
            TensorStorage::cpu(vec![T::from(r as f32).unwrap()]),
            vec![],
            false,
        )
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let r = ferray_linalg::matrix_rank(&arr, tol).map_err(FerrotorchError::Ferray)?;
        Tensor::from_storage(
            TensorStorage::cpu(vec![T::from(r as f64).unwrap()]),
            vec![],
            false,
        )
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// Condition number of `a` under the given norm order (`p = 2.0` for the
/// 2-norm, `1.0`, `f64::INFINITY`, etc.).
///
/// Mirrors `torch.linalg.cond`. CPU-only today.
pub fn cond<T: Float>(a: &Tensor<T>, p: f64) -> FerrotorchResult<Tensor<T>> {
    require_cpu(a, "cond")?;
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("cond requires a 2-D tensor, got {shape:?}"),
        });
    }
    validate_cond_selector(p)?;
    let order = float_to_norm_order(p);

    if tracking_enabled_for(&[a]) {
        if let Some(invert_ratio) = is_cond_svd_selector(p) {
            let s = svdvals(a)?;
            let max = s.amax()?;
            let min = s.amin()?;
            return if invert_ratio {
                min.div_t(&max)
            } else {
                max.div_t(&min)
            };
        }
        reject_forward_only_autograd("cond", &[a])?;
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let val: f32 = ferray_linalg::cond(&arr, order).map_err(FerrotorchError::Ferray)?;
        Tensor::from_storage(
            TensorStorage::cpu(vec![T::from(val).unwrap()]),
            vec![],
            false,
        )
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let val: f64 = ferray_linalg::cond(&arr, order).map_err(FerrotorchError::Ferray)?;
        Tensor::from_storage(
            TensorStorage::cpu(vec![T::from(val).unwrap()]),
            vec![],
            false,
        )
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// Map a torch-style `ord` float to ferray's `NormOrder`.
///
/// `2.0` -> Fro/L2, `1.0` -> L1, `f64::INFINITY` -> Inf, `f64::NEG_INFINITY`
/// -> NegInf, anything else -> P(p as the underlying float type).
// reason: PyTorch dispatches `ord` by exact magic values (1.0, 2.0, ±inf).
// 1.0 and 2.0 are exactly representable in f64 and callers pass these
// literals directly, so equality (not epsilon) is the correct dispatch
// predicate; an epsilon check would route nearby user values like 1.0001
// to L1 silently and break parity with torch.linalg.norm.
#[allow(clippy::float_cmp)]
fn float_to_norm_order<T: Into<f64>>(ord: T) -> ferray_linalg::NormOrder {
    let v: f64 = ord.into();
    if v == f64::INFINITY {
        ferray_linalg::NormOrder::Inf
    } else if v == f64::NEG_INFINITY {
        ferray_linalg::NormOrder::NegInf
    } else if v == 1.0 {
        ferray_linalg::NormOrder::L1
    } else if v == 2.0 {
        ferray_linalg::NormOrder::Fro
    } else {
        ferray_linalg::NormOrder::P(v)
    }
}

// ===========================================================================
// Vector products (cross / multi_dot)
// ===========================================================================

/// Vector cross product `a × b` along the given axis.
///
/// The axis selected by `dim` must have size 3 for both inputs, and the two
/// tensors must share the same shape. `dim` follows torch conventions:
/// non-negative indices count from the front, negative indices count from
/// the back (e.g. `-1` selects the last axis).
///
/// For a 1-D length-3 input pair this matches `numpy.cross` and
/// `torch.linalg.cross(..., dim=-1)`; for N-D tensors with one length-3
/// axis it produces the same shape with the cross product computed across
/// the 3 elements along `dim`, mirroring `torch.linalg.cross`.
///
pub fn cross<T: Float>(a: &Tensor<T>, b: &Tensor<T>, dim: i64) -> FerrotorchResult<Tensor<T>> {
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }
    let a_shape = a.shape();
    let b_shape = b.shape();
    if !is_f32::<T>() && !is_f64::<T>() && !is_f16::<T>() && !is_bf16::<T>() {
        return Err(FerrotorchError::InvalidArgument {
            message: "cross requires f32, f64, f16, or bf16".into(),
        });
    }
    if a_shape.len() != b_shape.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "linalg.cross: inputs must have the same number of dimensions, got {} and {}",
                a_shape.len(),
                b_shape.len()
            ),
        });
    }
    if a_shape.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "cross: inputs must have at least one dimension".into(),
        });
    }

    let rank = a_shape.len() as i64;
    let axis = if dim < 0 { dim + rank } else { dim };
    if axis < 0 || axis >= rank {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("cross: dim {dim} is out of range for tensor of rank {rank}"),
        });
    }
    let axis = axis as usize;
    if a_shape[axis] != 3 || b_shape[axis] != 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "linalg.cross: inputs dimension {axis} must have length 3. Got {} and {}",
                a_shape[axis], b_shape[axis]
            ),
        });
    }

    let out_shape = crate::shape::broadcast_shapes(a_shape, b_shape)?;

    // Autograd path: delegate to the differentiable wrapper, which computes
    // the forward under `no_grad` (preventing re-entry here) and attaches
    // `CrossBackward` (`da = cross(b, grad)`, `db = cross(grad, a)`).
    if crate::autograd::no_grad::is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        return crate::grad_fns::linalg::cross_differentiable(a, b, dim);
    }

    if a.is_cuda() {
        return cross_cuda(a, b, axis, &out_shape);
    }

    let stride_axis = cross_stride_axis(&out_shape, axis);
    let numel: usize = crate::shape::numel(&out_shape);
    let a_data = a.data_vec()?;
    let b_data = b.data_vec()?;
    let mut out: Vec<T> = vec![<T as num_traits::Zero>::zero(); numel];

    for flat in 0..numel {
        let coord = (flat / stride_axis) % 3;
        let base = flat - coord * stride_axis;
        let p0 = base;
        let p1 = base + stride_axis;
        let p2 = base + 2 * stride_axis;
        let a0 = a_data[cross_broadcast_src_flat(p0, &out_shape, a_shape)];
        let a1 = a_data[cross_broadcast_src_flat(p1, &out_shape, a_shape)];
        let a2 = a_data[cross_broadcast_src_flat(p2, &out_shape, a_shape)];
        let b0 = b_data[cross_broadcast_src_flat(p0, &out_shape, b_shape)];
        let b1 = b_data[cross_broadcast_src_flat(p1, &out_shape, b_shape)];
        let b2 = b_data[cross_broadcast_src_flat(p2, &out_shape, b_shape)];
        out[flat] = match coord {
            0 => a1 * b2 - a2 * b1,
            1 => a2 * b0 - a0 * b2,
            _ => a0 * b1 - a1 * b0,
        };
    }

    Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)
}

fn cross_stride_axis(shape: &[usize], axis: usize) -> usize {
    shape[axis + 1..].iter().product::<usize>().max(1)
}

fn cross_broadcast_src_flat(mut out_flat: usize, out_shape: &[usize], in_shape: &[usize]) -> usize {
    let out_ndim = out_shape.len();
    let mut src = 0usize;
    let mut stride = 1usize;
    for i in (0..out_ndim).rev() {
        let dim = out_shape[i];
        let coord = out_flat.checked_rem(dim).unwrap_or(0);
        out_flat = out_flat.checked_div(dim).unwrap_or(0);
        let in_dim = in_shape[i];
        if in_dim != 1 {
            src += coord * stride;
        }
        stride *= in_dim;
    }
    src
}

fn cross_cuda<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    axis: usize,
    out_shape: &[usize],
) -> FerrotorchResult<Tensor<T>> {
    let numel = crate::shape::numel(out_shape);
    if numel == 0 {
        return Tensor::from_storage(
            TensorStorage::on_device(Vec::<T>::new(), a.device())?,
            out_shape.to_vec(),
            false,
        );
    }

    let a_expanded = if a.shape() == out_shape {
        a.contiguous()?
    } else {
        crate::grad_fns::shape::expand(a, out_shape)?.contiguous()?
    };
    let b_expanded = if b.shape() == out_shape {
        b.contiguous()?
    } else {
        crate::grad_fns::shape::expand(b, out_shape)?.contiguous()?
    };
    let stride_axis = cross_stride_axis(out_shape, axis);
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let handle = match <T as Element>::dtype() {
        DType::F32 => backend.cross_f32(
            a_expanded.gpu_handle()?,
            b_expanded.gpu_handle()?,
            stride_axis,
        )?,
        DType::F64 => backend.cross_f64(
            a_expanded.gpu_handle()?,
            b_expanded.gpu_handle()?,
            stride_axis,
        )?,
        DType::F16 => backend.cross_f16(
            a_expanded.gpu_handle()?,
            b_expanded.gpu_handle()?,
            stride_axis,
        )?,
        DType::BF16 => backend.cross_bf16(
            a_expanded.gpu_handle()?,
            b_expanded.gpu_handle()?,
            stride_axis,
        )?,
        _ => {
            return Err(FerrotorchError::InvalidArgument {
                message: "cross requires f32, f64, f16, or bf16".into(),
            });
        }
    };
    Tensor::from_storage(TensorStorage::gpu(handle), out_shape.to_vec(), false)
}

/// Chained matmul `A1 @ A2 @ ... @ Ak`, ordered to minimise intermediate
/// flop count.
///
/// Mirrors `torch.linalg.multi_dot`. CPU-only today.
pub fn multi_dot<T: Float>(matrices: &[&Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
    if matrices.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "multi_dot requires at least one matrix".into(),
        });
    }
    for m in matrices {
        require_cpu(m, "multi_dot")?;
    }

    if is_f32::<T>() {
        let arrs: Vec<_> = matrices
            .iter()
            .map(|m| tensor_to_arraydyn_f32(m))
            .collect::<Result<_, _>>()?;
        let refs: Vec<_> = arrs.iter().collect();
        let r = ferray_linalg::multi_dot(&refs).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(r.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), r.shape().to_vec(), false)
    } else if is_f64::<T>() {
        let arrs: Vec<_> = matrices
            .iter()
            .map(|m| tensor_to_arraydyn_f64(m))
            .collect::<Result<_, _>>()?;
        let refs: Vec<_> = arrs.iter().collect();
        let r = ferray_linalg::multi_dot(&refs).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(r.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), r.shape().to_vec(), false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// Diagonal of a 2-D tensor, optionally offset.
///
/// Returns a 1-D tensor of length `min(m, n) - |offset|` containing
/// `a[i, i + offset]`. Implemented in-house (no ferray dep) since it's a
/// pure-shape operation.
///
/// Mirrors `torch.linalg.diagonal` (and `torch.diagonal` with `dim1=0,
/// dim2=1`).
///
/// # Backward
/// Autograd-aware (CPU): when grad tracking is active for `a`, this routes
/// through `crate::grad_fns::linalg::diagonal_differentiable` (the VJP
/// scatters `grad` back onto the `offset`-th diagonal of a zero matrix, per
/// `diagonal_backward_symint`, upstream `tools/autograd/derivatives.yaml:573`).
pub fn diagonal<T: Float>(a: &Tensor<T>, offset: i64) -> FerrotorchResult<Tensor<T>> {
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("diagonal requires a 2-D tensor, got {shape:?}"),
        });
    }

    // Autograd path: delegate to the differentiable wrapper, which computes
    // the forward inside `no_grad` (preventing re-entry here) and attaches
    // `DiagonalBackward`.
    if crate::autograd::no_grad::is_grad_enabled() && a.requires_grad() {
        return crate::grad_fns::linalg::diagonal_differentiable(a, offset);
    }

    if a.is_cuda() {
        return crate::ops::tensor_ops::diag(a, offset);
    }

    let (row_start, col_start) = if offset >= 0 {
        (0usize, offset as usize)
    } else {
        let row_start = usize::try_from(offset.unsigned_abs()).map_err(|_| {
            FerrotorchError::InvalidArgument {
                message: format!("diagonal: offset {offset} overflows usize"),
            }
        })?;
        (row_start, 0usize)
    };
    if col_start >= shape[1] || row_start >= shape[0] {
        return Tensor::from_storage(TensorStorage::cpu(Vec::<T>::new()), vec![0], false);
    }
    let len = (shape[0] - row_start).min(shape[1] - col_start);
    let data = a.data_vec()?;
    let mut out: Vec<T> = Vec::with_capacity(len);
    for i in 0..len {
        let r = row_start + i;
        let c = col_start + i;
        out.push(data[r * shape[1] + c]);
    }
    Tensor::from_storage(TensorStorage::cpu(out), vec![len], false)
}

// ---------------------------------------------------------------------------
// Trace (sum of main diagonal) and outer product
// ---------------------------------------------------------------------------

/// Sum of the main-diagonal elements of a 2-D tensor: `sum_i A[i, i]`.
///
/// Returns a scalar tensor. Mirrors `torch.trace` (`aten/src/ATen/native/
/// ReduceOps.cpp` `Tensor trace_cpu` and `cuda/TriangularOps.cu`
/// `trace_cuda`); `torch.trace` requires a 2-D input, so a non-2-D tensor is an
/// error here too. CUDA follows upstream and computes `self.diagonal().sum()`
/// on device.
///
/// # Backward
/// Autograd-aware: when grad tracking is active for `a`, this routes
/// through `crate::grad_fns::linalg::trace_differentiable` (the VJP
/// `dA = grad * I`, `trace_backward_symint`).
pub fn trace<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("trace requires a 2-D tensor, got {shape:?}"),
        });
    }

    // Autograd path: delegate to the differentiable wrapper, which computes
    // the forward inside `no_grad` (preventing re-entry here) and attaches
    // `TraceBackward`.
    if crate::autograd::no_grad::is_grad_enabled() && a.requires_grad() {
        return crate::grad_fns::linalg::trace_differentiable(a);
    }

    if a.is_cuda() {
        if shape[0].min(shape[1]) == 0 {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let handle = backend.alloc_zeros(1, T::dtype(), a.gpu_handle()?.device_ordinal())?;
            return Tensor::from_storage(TensorStorage::gpu(handle), vec![], false);
        }
        let diagonal = crate::ops::tensor_ops::diag(a, 0)?;
        return crate::grad_fns::reduction::sum(&diagonal);
    }

    let (m, n) = (shape[0], shape[1]);
    let k = m.min(n);
    let data = a.data_vec()?;
    let mut acc = <T as num_traits::Zero>::zero();
    for i in 0..k {
        acc += data[i * n + i];
    }
    Tensor::from_storage(TensorStorage::cpu(vec![acc]), vec![], false)
}

/// Outer product of two 1-D tensors: `out[i, j] = a[i] * b[j]`.
///
/// `a` is length `m`, `b` is length `n`; the result is `[m, n]`. Mirrors
/// `torch.outer` (`aten/src/ATen/native/LinearAlgebra.cpp:1337-1342`):
/// check both operands are 1-D, then compute `a.reshape({m, 1}) * b`.
///
/// # Backward
/// Autograd-aware on CPU and CUDA: the differentiable wrapper builds the same
/// composite reshape/broadcast-mul graph PyTorch uses, so gradients stay on
/// the original device and reduce over broadcast axes through `MulBackward`
/// and `ReshapeBackward`.
pub fn outer<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if crate::autograd::no_grad::is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        return crate::grad_fns::linalg::outer_differentiable(a, b);
    }

    outer_composite(a, b)
}

pub(crate) fn outer_composite<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "outer requires 1-D tensors, got {:?} and {:?}",
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

    let m = a.shape()[0];
    let m_dim = isize::try_from(m).map_err(|_| FerrotorchError::InvalidArgument {
        message: format!("outer: dimension {m} exceeds supported reshape bound"),
    })?;
    let lhs = crate::grad_fns::shape::reshape(a, &[m_dim, 1])?;
    crate::grad_fns::arithmetic::mul(&lhs, b)
}

// ---------------------------------------------------------------------------
// Fused-affine family + Kronecker (grad-aware public forwards)
// ---------------------------------------------------------------------------
//
// These are the `torch.addmm` / `torch.addbmm` / `torch.baddbmm` /
// `torch.addmv` / `torch.addr` / `torch.kron` public surface. Each delegates
// to the matching `crate::grad_fns::linalg::*_differentiable` wrapper, which
// computes the fused-affine forward inline and attaches the `*Backward`
// `GradFn` when `is_grad_enabled() && any-operand.requires_grad()` (else
// returns a plain tensor). The wrapper does NOT call back into these forwards,
// so no `no_grad` re-entry guard is needed.

/// `addmm(self, mat1, mat2, beta, alpha) = beta*self + alpha*(mat1 @ mat2)`.
///
/// Mirrors `torch.addmm`; upstream `TORCH_META_FUNC(addmm)` /
/// `TORCH_IMPL_FUNC(addmm_out_cpu)` in
/// `aten/src/ATen/native/LinearAlgebra.cpp:194,1620` (`self` is broadcast to
/// the `mat1 @ mat2` shape). VJP per `addmm` at
/// `tools/autograd/derivatives.yaml:256`.
///
/// # Backward
/// Autograd-aware (CPU): delegates to
/// `crate::grad_fns::linalg::addmm_differentiable`.
pub fn addmm<T: Float>(
    self_: &Tensor<T>,
    mat1: &Tensor<T>,
    mat2: &Tensor<T>,
    beta: T,
    alpha: T,
) -> FerrotorchResult<Tensor<T>> {
    crate::grad_fns::linalg::addmm_differentiable(self_, mat1, mat2, beta, alpha)
}

/// `addmv(self, mat, vec, beta, alpha) = beta*self + alpha*(mat @ vec)`.
///
/// Mirrors `torch.addmv`; upstream `TORCH_META_FUNC(addmv)` /
/// `TORCH_IMPL_FUNC(addmv_out_cpu)` in `aten/src/ATen/native/Blas.cpp:40,72`.
/// VJP per `addmv` at `tools/autograd/derivatives.yaml:267`.
///
/// # Backward
/// Autograd-aware (CPU): delegates to
/// `crate::grad_fns::linalg::addmv_differentiable`.
pub fn addmv<T: Float>(
    self_: &Tensor<T>,
    mat: &Tensor<T>,
    vec: &Tensor<T>,
    beta: T,
    alpha: T,
) -> FerrotorchResult<Tensor<T>> {
    crate::grad_fns::linalg::addmv_differentiable(self_, mat, vec, beta, alpha)
}

/// `addr(self, vec1, vec2, beta, alpha) = beta*self + alpha*outer(vec1, vec2)`.
///
/// Mirrors `torch.addr`; upstream `Tensor addr(...)` in
/// `aten/src/ATen/native/LinearAlgebra.cpp:1200`. VJP per `addr` at
/// `tools/autograd/derivatives.yaml:273`.
///
/// # Backward
/// Autograd-aware (CPU): delegates to
/// `crate::grad_fns::linalg::addr_differentiable`.
pub fn addr<T: Float>(
    self_: &Tensor<T>,
    vec1: &Tensor<T>,
    vec2: &Tensor<T>,
    beta: T,
    alpha: T,
) -> FerrotorchResult<Tensor<T>> {
    crate::grad_fns::linalg::addr_differentiable(self_, vec1, vec2, beta, alpha)
}

/// `addbmm(self, batch1, batch2, beta, alpha) = beta*self + alpha*sum_b(batch1[b] @ batch2[b])`.
///
/// Mirrors `torch.addbmm`; upstream `Tensor addbmm(...)` in
/// `aten/src/ATen/native/LinearAlgebra.cpp:1615`. VJP per `addbmm` at
/// `tools/autograd/derivatives.yaml:238`.
///
/// # Backward
/// Autograd-aware (CPU): delegates to
/// `crate::grad_fns::linalg::addbmm_differentiable`.
pub fn addbmm<T: Float>(
    self_: &Tensor<T>,
    batch1: &Tensor<T>,
    batch2: &Tensor<T>,
    beta: T,
    alpha: T,
) -> FerrotorchResult<Tensor<T>> {
    crate::grad_fns::linalg::addbmm_differentiable(self_, batch1, batch2, beta, alpha)
}

/// `baddbmm(self, batch1, batch2, beta, alpha) = beta*self + alpha*bmm(batch1, batch2)`.
///
/// Mirrors `torch.baddbmm`; upstream `TORCH_META_FUNC(baddbmm)` /
/// `TORCH_IMPL_FUNC(baddbmm_out_cpu)` in
/// `aten/src/ATen/native/LinearAlgebra.cpp:340,1886`. VJP per `baddbmm` at
/// `tools/autograd/derivatives.yaml:359`.
///
/// # Backward
/// Autograd-aware (CPU): delegates to
/// `crate::grad_fns::linalg::baddbmm_differentiable`.
pub fn baddbmm<T: Float>(
    self_: &Tensor<T>,
    batch1: &Tensor<T>,
    batch2: &Tensor<T>,
    beta: T,
    alpha: T,
) -> FerrotorchResult<Tensor<T>> {
    crate::grad_fns::linalg::baddbmm_differentiable(self_, batch1, batch2, beta, alpha)
}

/// `kron(self, other)` — Kronecker product (2-D × 2-D here).
///
/// Mirrors `torch.kron`; upstream `Tensor kron(const Tensor& self, const
/// Tensor& other)` in `aten/src/ATen/native/LinearAlgebra.cpp:3530`.
///
/// # Backward
/// Autograd-aware (CPU): delegates to
/// `crate::grad_fns::linalg::kron_differentiable` (per-Kron-block VJP
/// `dA = sum grad·B^T`, `dB = sum A^T·grad`).
pub fn kron<T: Float>(self_: &Tensor<T>, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    crate::grad_fns::linalg::kron_differentiable(self_, other)
}

// ===========================================================================
// Linalg tail: solve_triangular / matrix_exp / ldl / householder / *_ex (#581)
// ===========================================================================

/// Solve `A x = b` (or `x A = b`) where `A` is triangular.
///
/// `upper`: if `true`, treat `A` as upper-triangular; else lower-triangular.
/// `transpose`: if `true`, solve `A^T x = b` (or `x A^T = b`).
/// `unit_diagonal`: if `true`, ignore the diagonal entries of `A` and treat
/// them as 1 (the matrix's strict-triangular part still defines the system).
///
/// `b` may be 1-D (`[n]`) for a single right-hand side or 2-D (`[n, k]`) for
/// `k` simultaneous RHS columns. Output has the same shape as `b`.
///
/// Mirrors `torch.linalg.solve_triangular`. CPU uses forward/back substitution
/// in pure Rust at f64 internally; CUDA materializes the effective triangular
/// system on device and solves it through the resident cuSOLVER path.
pub fn solve_triangular<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    upper: bool,
    transpose: bool,
    unit_diagonal: bool,
) -> FerrotorchResult<Tensor<T>> {
    if a.ndim() != 2 || a.shape()[0] != a.shape()[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "solve_triangular: a must be square 2-D, got {:?}",
                a.shape()
            ),
        });
    }
    let n = a.shape()[0];
    let (b_shape, k) = match b.ndim() {
        1 => (vec![n], 1usize),
        2 => {
            if b.shape()[0] != n {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("solve_triangular: b leading dim {} ≠ n={n}", b.shape()[0]),
                });
            }
            (vec![n, b.shape()[1]], b.shape()[1])
        }
        _ => {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "solve_triangular: b must be 1-D or 2-D, got {:?}",
                    b.shape()
                ),
            });
        }
    };
    if a.is_cuda() != b.is_cuda() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }
    if tracking_enabled_for(&[a, b]) {
        return crate::grad_fns::linalg::solve_triangular_differentiable(
            a,
            b,
            upper,
            transpose,
            unit_diagonal,
        );
    }

    if a.is_cuda() {
        if !(is_f32::<T>() || is_f64::<T>()) {
            return Err(FerrotorchError::InvalidArgument {
                message: "solve_triangular requires f32 or f64".into(),
            });
        }
        let effective = effective_triangular_for_solve(a, upper, transpose, unit_diagonal)?;
        let rhs = b.contiguous()?;
        return solve(&effective, &rhs);
    }

    // Materialize to f64 internally; the existing helpers use the same
    // strategy. Final cast back to T.
    let a_f64: Vec<f64> = a.data()?.iter().map(|&v| v.to_f64().unwrap()).collect();
    let mut x: Vec<f64> = b.data()?.iter().map(|&v| v.to_f64().unwrap()).collect();

    // Effective `upper`: when transposed, an upper-triangular A becomes
    // lower-triangular and vice versa. Fold it here so the loop only handles
    // two cases.
    let effective_upper = upper ^ transpose;

    let a_at = |row: usize, col: usize| -> f64 {
        if transpose {
            a_f64[col * n + row]
        } else {
            a_f64[row * n + col]
        }
    };

    for col in 0..k {
        let stride = if b.ndim() == 1 { 0 } else { k };
        let xj = |i: usize, j: usize, x: &[f64]| -> f64 {
            if b.ndim() == 1 {
                x[i]
            } else {
                x[i * stride + j]
            }
        };
        let xj_set = |i: usize, j: usize, val: f64, x: &mut [f64]| {
            if b.ndim() == 1 {
                x[i] = val;
            } else {
                x[i * stride + j] = val;
            }
        };

        if effective_upper {
            // Back-substitute from row n-1 → 0.
            for i in (0..n).rev() {
                let mut sum = xj(i, col, &x);
                for j in (i + 1)..n {
                    sum -= a_at(i, j) * xj(j, col, &x);
                }
                let diag = if unit_diagonal { 1.0 } else { a_at(i, i) };
                if diag == 0.0 {
                    return Err(FerrotorchError::InvalidArgument {
                        message: "solve_triangular: zero on diagonal".into(),
                    });
                }
                xj_set(i, col, sum / diag, &mut x);
            }
        } else {
            // Forward-substitute from row 0 → n-1.
            for i in 0..n {
                let mut sum = xj(i, col, &x);
                for j in 0..i {
                    sum -= a_at(i, j) * xj(j, col, &x);
                }
                let diag = if unit_diagonal { 1.0 } else { a_at(i, i) };
                if diag == 0.0 {
                    return Err(FerrotorchError::InvalidArgument {
                        message: "solve_triangular: zero on diagonal".into(),
                    });
                }
                xj_set(i, col, sum / diag, &mut x);
            }
        }
    }

    let out: Vec<T> = x.into_iter().map(|v| T::from(v).unwrap()).collect();
    Tensor::from_storage(TensorStorage::cpu(out), b_shape, false)
}

/// LDL^T factorization of a real symmetric matrix (no pivoting).
///
/// Returns `(L, D)` where `L` is unit lower-triangular and `D` is diagonal
/// (returned as a length-`n` vector), with `A = L D L^T`.
///
/// Mirrors `torch.linalg.ldl_factor` for the no-pivot case. Numerically
/// reliable on positive-definite or strongly diagonally-dominant inputs;
/// for indefinite or rank-deficient matrices use `eigh` or a pivoted
/// factorization (Bunch-Kaufman) — see follow-up.
pub fn ldl_factor<T: Float>(a: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    require_cpu(a, "ldl_factor")?;
    if a.ndim() != 2 || a.shape()[0] != a.shape()[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("ldl_factor: a must be square 2-D, got {:?}", a.shape()),
        });
    }
    reject_forward_only_autograd("ldl_factor", &[a])?;
    let n = a.shape()[0];
    let a_f64: Vec<f64> = a.data()?.iter().map(|&v| v.to_f64().unwrap()).collect();

    let mut l = vec![0.0f64; n * n];
    let mut d = vec![0.0f64; n];

    // No-pivot LDL^T: for j in 0..n,
    //   D_j = A_jj - sum_{k<j} L_jk^2 * D_k
    //   L_ij = (A_ij - sum_{k<j} L_ik * L_jk * D_k) / D_j  for i > j
    //   L_jj = 1
    for j in 0..n {
        let mut diag = a_f64[j * n + j];
        for k in 0..j {
            diag -= l[j * n + k] * l[j * n + k] * d[k];
        }
        d[j] = diag;
        if diag == 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("ldl_factor: zero pivot at column {j} (no-pivot path)"),
            });
        }
        l[j * n + j] = 1.0;
        for i in (j + 1)..n {
            let mut sum = a_f64[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k] * d[k];
            }
            l[i * n + j] = sum / diag;
        }
    }

    let l_out: Vec<T> = l.into_iter().map(|v| T::from(v).unwrap()).collect();
    let d_out: Vec<T> = d.into_iter().map(|v| T::from(v).unwrap()).collect();
    Ok((
        Tensor::from_storage(TensorStorage::cpu(l_out), vec![n, n], false)?,
        Tensor::from_storage(TensorStorage::cpu(d_out), vec![n], false)?,
    ))
}

/// Solve `A x = b` using a precomputed LDL^T factorization.
///
/// Given `(L, D)` from [`ldl_factor`] and a right-hand side `b` (1-D or 2-D),
/// returns `x` such that `(L D L^T) x = b`. Same `b` shape conventions as
/// [`solve`] / [`solve_triangular`].
pub fn ldl_solve<T: Float>(
    l: &Tensor<T>,
    d: &Tensor<T>,
    b: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    require_cpu(l, "ldl_solve")?;
    require_cpu(d, "ldl_solve")?;
    require_cpu(b, "ldl_solve")?;
    reject_forward_only_autograd("ldl_solve", &[l, d, b])?;
    if l.ndim() != 2 || l.shape()[0] != l.shape()[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("ldl_solve: L must be square 2-D, got {:?}", l.shape()),
        });
    }
    if d.ndim() != 1 || d.shape()[0] != l.shape()[0] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "ldl_solve: D must be 1-D of length {}, got {:?}",
                l.shape()[0],
                d.shape()
            ),
        });
    }

    // Step 1: solve L y = b (forward substitution, unit diagonal).
    let y = solve_triangular(
        l, b, /* upper */ false, /* transpose */ false, /* unit_diag */ true,
    )?;
    // Step 2: scale by D^{-1}: z_i = y_i / d_i (broadcast across columns of y).
    let n = d.shape()[0];
    let d_data = d.data()?.to_vec();
    let y_data = y.data()?.to_vec();
    let z_shape = y.shape().to_vec();
    let k = if y.ndim() == 1 { 1 } else { y.shape()[1] };
    let mut z = vec![T::from(0.0).unwrap(); y_data.len()];
    for i in 0..n {
        let di = d_data[i].to_f64().unwrap();
        if di == 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("ldl_solve: zero diagonal at index {i}"),
            });
        }
        for j in 0..k {
            let val = if y.ndim() == 1 {
                y_data[i].to_f64().unwrap()
            } else {
                y_data[i * k + j].to_f64().unwrap()
            };
            let scaled = T::from(val / di).unwrap();
            if y.ndim() == 1 {
                z[i] = scaled;
            } else {
                z[i * k + j] = scaled;
            }
        }
    }
    let z_t = Tensor::from_storage(TensorStorage::cpu(z), z_shape, false)?;
    // Step 3: solve L^T x = z (back substitution via transpose).
    solve_triangular(
        l, &z_t, /* upper */ false, /* transpose */ true, /* unit_diag */ true,
    )
}

/// Apply the implicit Householder representation `(V, tau)` from a QR
/// factorization to recover the orthogonal matrix `Q`.
///
/// `v` is `[m, k]` whose `j`-th column is the Householder vector for the
/// `j`-th reflection (with implicit unit at row `j`). `tau` is `[k]` of
/// scalar coefficients. Returns the first `k` columns of `Q`, shape `[m, k]`,
/// where `Q = (I - tau_0 v_0 v_0^T)(I - tau_1 v_1 v_1^T) ...`.
///
/// Mirrors `torch.linalg.householder_product` (which returns the leading `k`
/// columns of the reconstructed orthogonal factor — shape `[m, k]`, NOT the
/// full `[m, m]` matrix). When grad is enabled and either input requires grad,
/// delegates to `crate::grad_fns::linalg::householder_product_differentiable`
/// to attach the reflector-recursion VJP.
pub fn householder_product<T: Float>(
    v: &Tensor<T>,
    tau: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    if crate::autograd::no_grad::is_grad_enabled() && (v.requires_grad() || tau.requires_grad()) {
        return crate::grad_fns::linalg::householder_product_differentiable(v, tau);
    }
    let q = householder_product_full(v, tau)?;
    let m = v.shape()[0];
    let k = v.shape()[1];
    if k == m {
        return Ok(q);
    }
    // Slice the leading k columns of the row-major [m, m] product.
    let q_data = q.data()?;
    let mut out = Vec::with_capacity(m * k);
    for i in 0..m {
        for j in 0..k {
            out.push(q_data[i * m + j]);
        }
    }
    Tensor::from_storage(TensorStorage::cpu(out), vec![m, k], false)
}

/// Reconstructs the FULL `[m, m]` orthogonal product
/// `Q = (I - tau_0 v_0 v_0^T)(I - tau_1 v_1 v_1^T) ...` from the compact
/// `(v, tau)` representation. The public `householder_product` returns only the
/// leading `k` columns of this (torch contract); the backward VJP
/// (`HouseholderProductBackward`) needs the full square `Q` for its
/// `K = Q_full @ grad^T` step, so this is exposed crate-wide.
pub fn householder_product_full<T: Float>(
    v: &Tensor<T>,
    tau: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    require_cpu(v, "householder_product")?;
    require_cpu(tau, "householder_product")?;
    if v.ndim() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("householder_product: v must be 2-D, got {:?}", v.shape()),
        });
    }
    if tau.ndim() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "householder_product: tau must be 1-D, got {:?}",
                tau.shape()
            ),
        });
    }
    let m = v.shape()[0];
    let k = v.shape()[1];
    if tau.shape()[0] != k {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "householder_product: tau length {} ≠ v cols {k}",
                tau.shape()[0]
            ),
        });
    }
    if k > m {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("householder_product: k={k} must be ≤ m={m}"),
        });
    }
    reject_forward_only_autograd("householder_product_full", &[v, tau])?;

    let v_f64: Vec<f64> = v.data()?.iter().map(|&x| x.to_f64().unwrap()).collect();
    let tau_f64: Vec<f64> = tau.data()?.iter().map(|&x| x.to_f64().unwrap()).collect();

    // Initialize Q = I_m (row-major).
    let mut q = vec![0.0f64; m * m];
    for i in 0..m {
        q[i * m + i] = 1.0;
    }

    // Apply reflections in reverse order so the cumulative product
    // (I - τ_0 v_0 v_0^T) (I - τ_1 v_1 v_1^T) ... lands in Q. We update
    // Q ← H_j Q where H_j is the j-th reflector. Stepping right-to-left
    // makes that Q = H_0 H_1 ... H_{k-1}.
    for j in (0..k).rev() {
        let tau_j = tau_f64[j];
        if tau_j == 0.0 {
            continue;
        }
        // Extract v_j: column j of V with implicit unit at row j and zeros
        // above row j.
        let mut vj = vec![0.0f64; m];
        vj[j] = 1.0;
        for i in (j + 1)..m {
            vj[i] = v_f64[i * k + j];
        }

        // For each column c of Q, compute Q[:, c] -= τ * v_j * (v_j^T Q[:, c]).
        for c in 0..m {
            let mut dot = 0.0f64;
            for i in 0..m {
                dot += vj[i] * q[i * m + c];
            }
            let scale = tau_j * dot;
            for i in 0..m {
                q[i * m + c] -= scale * vj[i];
            }
        }
    }

    let out: Vec<T> = q.into_iter().map(|x| T::from(x).unwrap()).collect();
    Tensor::from_storage(TensorStorage::cpu(out), vec![m, m], false)
}

/// Matrix exponential `expm(A)` via Padé(13) with scaling and squaring.
///
/// Uses the Higham 2005 algorithm: choose `s` so `||A/2^s||_∞ ≤ θ_13`,
/// compute the Padé(13) approximation of `exp(A/2^s)`, then square `s`
/// times to recover `exp(A)`. Mirrors `torch.linalg.matrix_exp`.
///
/// CPU works in f64 internally. CUDA composes resident matmul/add/solve kernels
/// and uses the same Padé(13) scaling-and-squaring algorithm.
pub fn matrix_exp<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.ndim() != 2 || a.shape()[0] != a.shape()[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("matrix_exp: a must be square 2-D, got {:?}", a.shape()),
        });
    }
    if tracking_enabled_for(&[a]) {
        return crate::grad_fns::linalg::matrix_exp_differentiable(a);
    }
    let n = a.shape()[0];
    if a.is_cuda() {
        if !(is_f32::<T>() || is_f64::<T>()) {
            return Err(FerrotorchError::InvalidArgument {
                message: "matrix_exp requires f32 or f64".into(),
            });
        }
        return matrix_exp_pade13_tensor(a);
    }
    if n == 0 {
        return Tensor::from_storage(TensorStorage::cpu(Vec::<T>::new()), vec![0, 0], false);
    }
    let a_data: Vec<f64> = a.data()?.iter().map(|&v| v.to_f64().unwrap()).collect();
    // Trivial 1x1 case mirrors upstream `linalg_matrix_exp` (pytorch
    // `aten/src/ATen/native/LinearAlgebra.cpp:2795`: `n == 1` returns
    // `a.exp()`). Exact for extreme magnitudes — `exp(1e20) = inf`,
    // `exp(-1e20) = 0`, `exp(inf) = inf` — where scaling-and-squaring
    // either drifts or (for `inf`) would poison the Padé solve into NaN
    // (CORE-148 / #1842).
    if n == 1 {
        let out: Vec<T> = vec![T::from(a_data[0].exp()).unwrap()];
        return Tensor::from_storage(TensorStorage::cpu(out), vec![1, 1], false);
    }
    let result = matrix_exp_pade13(&a_data, n)?;
    let out: Vec<T> = result.into_iter().map(|v| T::from(v).unwrap()).collect();
    Tensor::from_storage(TensorStorage::cpu(out), vec![n, n], false)
}

// --- helpers for matrix_exp ------------------------------------------------

fn mat_eye(n: usize) -> Vec<f64> {
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

fn mat_inf_norm(a: &[f64], n: usize) -> f64 {
    (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j].abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            // NO zero-skip shortcut (CORE-148 / #1842): skipping `aik == 0`
            // suppresses IEEE `0 × inf = NaN`, which torch's dense matmul in
            // the squaring phase DOES produce — e.g.
            // `matrix_exp([[1e308, 0], [0, -1e308]])` is all-NaN in torch
            // (live 2.11.0+cu130) precisely because the overflowing diagonal
            // poisons the off-diagonal zeros. For finite inputs the skip was
            // value-neutral; for non-finite intermediates it silently
            // diverged.
            for j in 0..n {
                out[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    out
}

fn mat_axpby(a: &[f64], alpha: f64, b: &[f64], beta: f64) -> Vec<f64> {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| alpha * x + beta * y)
        .collect()
}

fn tensor_scale<T: Float>(a: &Tensor<T>, alpha: f64) -> FerrotorchResult<Tensor<T>> {
    let zeros = crate::creation::zeros_like(a)?;
    crate::grad_fns::arithmetic::add_scaled(&zeros, a, alpha)
}

fn tensor_axpby<T: Float>(
    a: &Tensor<T>,
    alpha: f64,
    b: &Tensor<T>,
    beta: f64,
) -> FerrotorchResult<Tensor<T>> {
    let scaled_a = tensor_scale(a, alpha)?;
    crate::grad_fns::arithmetic::add_scaled(&scaled_a, b, beta)
}

fn matrix_exp_pade13_tensor<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    const THETA13: f64 = 5.371920351148152;
    let b: [f64; 14] = [
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    ];

    let n = a.shape()[0];
    if n == 0 {
        return a.contiguous();
    }
    if n == 1 {
        return a.exp_t();
    }

    let norm_t = a.abs_t()?.sum_dim(1, false)?.amax()?;
    let norm_data = norm_t.data_vec()?;
    let norm = norm_data.first().and_then(|v| v.to_f64()).ok_or_else(|| {
        FerrotorchError::InvalidArgument {
            message: "matrix_exp: norm is not representable as f64".into(),
        }
    })?;
    if norm.is_nan() || norm.is_infinite() {
        let nan = T::from(f64::NAN).ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "matrix_exp: NaN is not representable in dtype".into(),
        })?;
        return full_like_on_device(a.shape(), nan, a.device(), "matrix_exp");
    }

    let s = if norm <= THETA13 {
        0
    } else {
        ((norm / THETA13).log2().ceil() as i32).clamp(0, 1023)
    };
    let scale = 2f64.powi(s);
    let a_scaled = tensor_scale(a, 1.0 / scale)?;
    let id = eye_on_device(n, a.device())?;
    let a2 = a_scaled.mm(&a_scaled)?;
    let a4 = a2.mm(&a2)?;
    let a6 = a4.mm(&a2)?;

    let inner_u = {
        let t1 = tensor_axpby(&a6, b[13], &a4, b[11])?;
        let t2 = crate::grad_fns::arithmetic::add_scaled(&t1, &a2, b[9])?;
        a6.mm(&t2)?
    };
    let mid_u = crate::grad_fns::arithmetic::add_scaled(&inner_u, &a6, b[7])?;
    let mid_u = crate::grad_fns::arithmetic::add_scaled(&mid_u, &a4, b[5])?;
    let mid_u = crate::grad_fns::arithmetic::add_scaled(&mid_u, &a2, b[3])?;
    let mid_u = crate::grad_fns::arithmetic::add_scaled(&mid_u, &id, b[1])?;
    let u = a_scaled.mm(&mid_u)?;

    let inner_v = {
        let t1 = tensor_axpby(&a6, b[12], &a4, b[10])?;
        let t2 = crate::grad_fns::arithmetic::add_scaled(&t1, &a2, b[8])?;
        a6.mm(&t2)?
    };
    let v = crate::grad_fns::arithmetic::add_scaled(&inner_v, &a6, b[6])?;
    let v = crate::grad_fns::arithmetic::add_scaled(&v, &a4, b[4])?;
    let v = crate::grad_fns::arithmetic::add_scaled(&v, &a2, b[2])?;
    let v = crate::grad_fns::arithmetic::add_scaled(&v, &id, b[0])?;

    let p = crate::grad_fns::arithmetic::add_scaled(&v, &u, -1.0)?;
    let q = crate::grad_fns::arithmetic::add_scaled(&v, &u, 1.0)?;
    let mut r = solve(&p, &q)?;
    for _ in 0..s.max(0) {
        r = r.mm(&r)?;
    }
    Ok(r)
}

/// Solve `(I - U)^{-1} (I + U)`-style linear system used in Padé approximant.
/// Solves `(P) X = Q` for `X` via LU with partial pivoting in pure Rust.
fn solve_dense_pivoted(p: &[f64], q: &[f64], n: usize) -> FerrotorchResult<Vec<f64>> {
    // Augmented matrix [P | Q] in row-major; size n × 2n.
    let mut aug = vec![0.0f64; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * (2 * n) + j] = p[i * n + j];
            aug[i * (2 * n) + n + j] = q[i * n + j];
        }
    }
    // Gaussian elimination with partial pivoting.
    for col in 0..n {
        // Find pivot row.
        let mut pivot_row = col;
        let mut pivot_val = aug[col * (2 * n) + col].abs();
        for r in (col + 1)..n {
            let v = aug[r * (2 * n) + col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = r;
            }
        }
        if pivot_val == 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "matrix_exp: singular Padé denominator (numerical)".into(),
            });
        }
        if pivot_row != col {
            for j in 0..(2 * n) {
                aug.swap(col * (2 * n) + j, pivot_row * (2 * n) + j);
            }
        }
        // Eliminate below.
        let pivot = aug[col * (2 * n) + col];
        for r in (col + 1)..n {
            let factor = aug[r * (2 * n) + col] / pivot;
            if factor == 0.0 {
                continue;
            }
            for j in col..(2 * n) {
                aug[r * (2 * n) + j] -= factor * aug[col * (2 * n) + j];
            }
        }
    }
    // Back-substitute.
    let mut x = vec![0.0f64; n * n];
    for c in 0..n {
        for i in (0..n).rev() {
            let mut sum = aug[i * (2 * n) + n + c];
            for j in (i + 1)..n {
                sum -= aug[i * (2 * n) + j] * x[j * n + c];
            }
            let diag = aug[i * (2 * n) + i];
            x[i * n + c] = sum / diag;
        }
    }
    Ok(x)
}

fn matrix_exp_pade13(a: &[f64], n: usize) -> FerrotorchResult<Vec<f64>> {
    // Higham 2005 thresholds and Padé(13) coefficients.
    const THETA13: f64 = 5.371920351148152;
    let b: [f64; 14] = [
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    ];

    let norm = mat_inf_norm(a, n);
    // CORE-148 / #1842: an INFINITE norm (an `inf` entry, or finite entries
    // whose absolute row sum overflows) cannot be scaled into the Padé
    // convergence region at all. torch returns an all-NaN matrix for every
    // such case (live 2.11.0+cu130 probes: `[[inf,1],[0,1]]`,
    // `[[1e308,1e308],[0,0]]`, `[[1e308,0],[0,-1e308]]` → all `nan`; the
    // `n == 1` trivial case is handled by `matrix_exp` before this fn).
    // NaN entries don't reach this branch: `mat_inf_norm`'s
    // `fold(0.0, f64::max)` ignores NaN row sums, and the Padé arithmetic
    // below propagates the NaNs itself.
    if norm.is_infinite() {
        return Ok(vec![f64::NAN; n * n]);
    }
    // Scaling exponent. The `as i32` cast saturates, and `s` is clamped to
    // 1023: a FINITE norm is < f64::MAX ≈ 1.8e308, so
    // `ceil(log2(norm / θ13)) ≤ ceil(log2(1.8e308 / 5.37)) = 1021` — the
    // clamp is unreachable for finite norms and exists only to keep
    // `2f64.powi(s)` finite (2^1023 < f64::MAX) against float pathologies.
    // Pre-#1842 this computed `1u64 << s`, which PANICS (debug) or WRAPS
    // mod 64 (release) for `s ≥ 64`, i.e. `norm > θ13·2^63 ≈ 4.95e19`.
    let s = if norm <= THETA13 {
        0
    } else {
        ((norm / THETA13).log2().ceil() as i32).clamp(0, 1023)
    };
    let scale = 2f64.powi(s);
    let a_scaled: Vec<f64> = a.iter().map(|&v| v / scale).collect();

    let id = mat_eye(n);
    let a2 = mat_mul(&a_scaled, &a_scaled, n);
    let a4 = mat_mul(&a2, &a2, n);
    let a6 = mat_mul(&a4, &a2, n);

    // U = A * (A6 * (b13 A6 + b11 A4 + b9 A2) + b7 A6 + b5 A4 + b3 A2 + b1 I)
    // V = A6 * (b12 A6 + b10 A4 + b8 A2) + b6 A6 + b4 A4 + b2 A2 + b0 I
    let inner_u = {
        let t1 = mat_axpby(&a6, b[13], &a4, b[11]);
        let t2 = mat_axpby(&t1, 1.0, &a2, b[9]);
        mat_mul(&a6, &t2, n)
    };
    let mid_u = mat_axpby(&inner_u, 1.0, &a6, b[7]);
    let mid_u = mat_axpby(&mid_u, 1.0, &a4, b[5]);
    let mid_u = mat_axpby(&mid_u, 1.0, &a2, b[3]);
    let mid_u = mat_axpby(&mid_u, 1.0, &id, b[1]);
    let u = mat_mul(&a_scaled, &mid_u, n);

    let inner_v = {
        let t1 = mat_axpby(&a6, b[12], &a4, b[10]);
        let t2 = mat_axpby(&t1, 1.0, &a2, b[8]);
        mat_mul(&a6, &t2, n)
    };
    let v = mat_axpby(&inner_v, 1.0, &a6, b[6]);
    let v = mat_axpby(&v, 1.0, &a4, b[4]);
    let v = mat_axpby(&v, 1.0, &a2, b[2]);
    let v = mat_axpby(&v, 1.0, &id, b[0]);

    let p = mat_axpby(&v, 1.0, &u, -1.0); // V - U
    let q = mat_axpby(&v, 1.0, &u, 1.0); // V + U
    let mut r = solve_dense_pivoted(&p, &q, n)?;

    // Squaring phase.
    for _ in 0..s.max(0) {
        r = mat_mul(&r, &r, n);
    }
    Ok(r)
}

// ---------------------------------------------------------------------------
// `_ex` variants — return `(value, info)` with non-throwing semantics for
// NUMERICAL failures only (CORE-145 / #1839)
// ---------------------------------------------------------------------------

/// Classify an error from an `_ex`-family forward: `Some(info)` for a
/// NUMERICAL failure (the LAPACK/cuSOLVER `info > 0` class that
/// `torch.linalg.*_ex` suppresses), `None` for a structural error
/// (shape/dim/dtype/device/backend) that the `_ex` wrapper must PROPAGATE
/// — torch's `_ex` variants still raise on those (CORE-145 / #1839).
///
/// `info` provenance:
/// - **CPU**: ferray-linalg reports `FerrayError::SingularMatrix` for both
///   not-positive-definite (`cholesky`) and singular (`inv`/`solve`)
///   inputs. Probed at ferray-linalg 0.4.9: the error carries NO
///   minor/pivot index ("matrix is not positive definite"), so CPU `info`
///   is the documented constant `1` (#1944 tracks surfacing the true
///   LAPACK index; torch reports e.g. `2` for a minor-2 failure).
/// - **CUDA**: cuSOLVER `devInfo` failures surface as
///   `InvalidArgument` whose message ferrotorch-gpu's `map_gpu_err` builds
///   from `GpuError::ShapeMismatch { op: "gpu_…", got: vec![devInfo] }`
///   (e.g. `"gpu_cholesky_f64_dev: potrf failed (matrix not positive
///   definite): shape mismatch, expected [0], got [2]"`). The true
///   `devInfo` index — identical to torch's `info` — is recovered from the
///   trailing `got [k]`. String-matching is the only classification
///   channel available here: `ferrotorch-core` cannot depend on
///   `ferrotorch-gpu`'s `GpuError` type (workspace dep cycle), and
///   `map_gpu_err` erases the variant into `InvalidArgument`. The CUDA
///   conformance tests (`gpu_*_1839`) pin this contract on real hardware.
fn ex_numerical_info(err: &FerrotorchError) -> Option<i32> {
    match err {
        FerrotorchError::Ferray(ferray_core::FerrayError::SingularMatrix { .. }) => Some(1),
        FerrotorchError::InvalidArgument { message }
            if (message.starts_with("gpu_cholesky")
                && (message.contains("not positive definite")
                    || message.contains("not positive-definite")))
                || (message.starts_with("gpu_solve")
                    && message.contains("LU factorization failed")) =>
        {
            Some(parse_trailing_dev_info(message).unwrap_or(1))
        }
        _ => None,
    }
}

/// Recover the cuSOLVER `devInfo` value from a `map_gpu_err`-formatted
/// message ending in `… got [k]` (see [`ex_numerical_info`]).
fn parse_trailing_dev_info(message: &str) -> Option<i32> {
    let rest = &message[message.rfind("got [")? + 5..];
    rest[..rest.find(']')?].parse().ok()
}

/// Build the 0-d `info` scalar on `like`'s device (torch returns `info` on
/// the input device; the pre-#1839 code always allocated it on CPU).
///
/// `info` is a `T`-typed 0-d tensor — a documented deviation from torch's
/// `int32` (`Tensor<T>` is the only tensor type this signature can carry).
fn ex_info_scalar<T: Float>(value: i32, like: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let t = Tensor::from_storage(
        TensorStorage::cpu(vec![T::from(value).unwrap()]),
        vec![],
        false,
    )?;
    if like.is_cuda() {
        t.to(like.device())
    } else {
        Ok(t)
    }
}

/// Build an all-zeros fallback value tensor of `shape` on `like`'s device.
/// torch documents the value output as UNDEFINED when `info != 0` (it
/// returns the partial factor); deterministic zeros are a legal choice.
fn ex_zeros_like<T: Float>(shape: Vec<usize>, like: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let total: usize = crate::shape::numel(&shape);
    let t = Tensor::from_storage(
        TensorStorage::cpu(vec![T::from(0.0).unwrap(); total]),
        shape,
        false,
    )?;
    if like.is_cuda() {
        t.to(like.device())
    } else {
        Ok(t)
    }
}

/// `cholesky` that doesn't error on a NUMERICAL failure: returns
/// `(L, info)` where `info` is `0` on success and non-zero when `A` is not
/// positive-definite (`L` is then same-shape zeros — torch documents the
/// value as undefined). Structural errors (non-square, non-2-D, dtype,
/// device, missing backend) PROPAGATE as `Err`, exactly like
/// `torch.linalg.cholesky_ex` raises (CORE-145 / #1839).
///
/// `info` index: the true cuSOLVER failing-minor index on CUDA; the
/// documented constant `1` on CPU (ferray carries no index — #1944).
/// Returned as a `T`-typed 0-d scalar on the input device (torch: `int32`).
///
/// Mirrors `torch.linalg.cholesky_ex`.
pub fn cholesky_ex<T: Float>(input: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    match cholesky(input) {
        Ok(l) => Ok((l, ex_info_scalar(0, input)?)),
        Err(e) => match ex_numerical_info(&e) {
            Some(info) => {
                // `cholesky` validated square 2-D before dispatch, so a
                // numerical failure implies shape [n, n].
                let n = input.shape()[0];
                Ok((
                    ex_zeros_like(vec![n, n], input)?,
                    ex_info_scalar(info, input)?,
                ))
            }
            None => Err(e),
        },
    }
}

/// `inv` that doesn't error on singular input: returns `(A^{-1}, info)`;
/// on a NUMERICAL (singular) failure the value is same-shape zeros and
/// `info != 0`. Structural errors PROPAGATE (torch raises) — CORE-145 /
/// #1839. CPU `info` is the documented constant `1` (#1944); torch
/// reports the first zero pivot. Mirrors `torch.linalg.inv_ex`.
pub fn inv_ex<T: Float>(input: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    match inv(input) {
        Ok(out) => Ok((out, ex_info_scalar(0, input)?)),
        Err(e) => match ex_numerical_info(&e) {
            Some(info) => {
                let n = input.shape()[0];
                Ok((
                    ex_zeros_like(vec![n, n], input)?,
                    ex_info_scalar(info, input)?,
                ))
            }
            None => Err(e),
        },
    }
}

/// `solve` that doesn't error on singular `A`: returns `(x, info)`; on a
/// NUMERICAL (singular) failure `x` is zeros shaped like `b` and
/// `info != 0`. Structural errors — including `DeviceMismatch` —
/// PROPAGATE (torch raises) — CORE-145 / #1839. `info` is the true
/// cuSOLVER getrf pivot index on CUDA, the documented constant `1` on CPU
/// (#1944). Mirrors `torch.linalg.solve_ex`.
pub fn solve_ex<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    match solve(a, b) {
        Ok(x) => Ok((x, ex_info_scalar(0, b)?)),
        Err(e) => match ex_numerical_info(&e) {
            Some(info) => Ok((
                ex_zeros_like(b.shape().to_vec(), b)?,
                ex_info_scalar(info, b)?,
            )),
            None => Err(e),
        },
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    // Helper: build a symmetric positive-definite matrix.
    // A = M^T M + I where M = [[2,1,0],[1,3,1],[0,1,2]].
    fn spd_3x3() -> Tensor<f64> {
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            6.0, 5.0, 1.0,
            5.0, 12.0, 5.0,
            1.0, 5.0, 6.0,
        ];
        t(&a, &[3, 3])
    }

    #[test]
    fn test_svd_reconstructs() {
        // A = [[1, 2], [3, 4], [5, 6]]  (3x2)
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let (u, s, vh) = svd(&a).unwrap();

        // Reconstruct: U @ diag(S) @ Vh
        let u_data = u.data().unwrap();
        let s_data = s.data().unwrap();
        let vh_data = vh.data().unwrap();
        let u_shape = u.shape();
        let vh_shape = vh.shape();

        let m = u_shape[0]; // 3
        let k = u_shape[1]; // 2 (reduced)
        let n = vh_shape[1]; // 2

        // U @ diag(S): scale columns of U by S
        let mut us = vec![0.0f64; m * k];
        for i in 0..m {
            for j in 0..k {
                us[i * k + j] = u_data[i * k + j] * s_data[j];
            }
        }

        // (US) @ Vh
        let mut recon = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += us[i * k + p] * vh_data[p * n + j];
                }
                recon[i * n + j] = acc;
            }
        }

        let a_data = a.data().unwrap();
        for i in 0..m * n {
            assert!(
                (recon[i] - a_data[i]).abs() < 1e-10,
                "SVD reconstruction failed at index {}: {} vs {}",
                i,
                recon[i],
                a_data[i]
            );
        }
    }

    #[test]
    fn test_solve_ax_eq_b() {
        // A = [[2, 1], [1, 3]], b = [5, 10]
        // Solution: x = [1, 3]  (2*1+1*3=5, 1*1+3*3=10)
        let a = t(&[2.0, 1.0, 1.0, 3.0], &[2, 2]);
        let b = t(&[5.0, 10.0], &[2]);
        let x = solve(&a, &b).unwrap();
        let x_data = x.data().unwrap();
        assert!((x_data[0] - 1.0).abs() < 1e-10);
        assert!((x_data[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_det_identity() {
        let eye = t(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]);
        let d = det(&eye).unwrap();
        assert!(d.is_scalar());
        assert!((d.item().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inv_identity() {
        // inv(A) @ A ~ I
        let a = t(&[2.0, 1.0, 1.0, 3.0], &[2, 2]);
        let a_inv = inv(&a).unwrap();
        let a_inv_data = a_inv.data().unwrap();
        let a_data = a.data().unwrap();
        let n = 2;

        // Compute a_inv @ a
        let mut product = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += a_inv_data[i * n + k] * a_data[k * n + j];
                }
                product[i * n + j] = acc;
            }
        }

        // Should be approximately identity
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[i * n + j] - expected).abs() < 1e-10,
                    "inv(A) @ A [{},{}] = {} (expected {})",
                    i,
                    j,
                    product[i * n + j],
                    expected,
                );
            }
        }
    }

    #[test]
    fn test_qr_reconstructs() {
        // A = [[1, 2], [3, 4], [5, 6]]
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let (q, r) = qr(&a).unwrap();
        let q_data = q.data().unwrap();
        let r_data = r.data().unwrap();
        let q_shape = q.shape();
        let r_shape = r.shape();

        let m = q_shape[0]; // 3
        let k = q_shape[1]; // 2 (reduced)
        let n = r_shape[1]; // 2

        // Reconstruct Q @ R
        let mut recon = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += q_data[i * k + p] * r_data[p * n + j];
                }
                recon[i * n + j] = acc;
            }
        }

        let a_data = a.data().unwrap();
        for i in 0..m * n {
            assert!(
                (recon[i] - a_data[i]).abs() < 1e-10,
                "QR reconstruction failed at index {}: {} vs {}",
                i,
                recon[i],
                a_data[i]
            );
        }

        // Q should be orthogonal: Q^T @ Q ~ I_k
        let mut qtq = vec![0.0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut acc = 0.0;
                for p in 0..m {
                    acc += q_data[p * k + i] * q_data[p * k + j];
                }
                qtq[i * k + j] = acc;
            }
        }
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[i * k + j] - expected).abs() < 1e-10,
                    "Q^T Q [{},{}] = {} (expected {})",
                    i,
                    j,
                    qtq[i * k + j],
                    expected,
                );
            }
        }
    }

    #[test]
    fn test_cholesky_spd() {
        let a = spd_3x3();
        let l = cholesky(&a).unwrap();
        let l_data = l.data().unwrap();
        let n = 3;

        // Verify lower-triangular: upper entries should be zero
        for i in 0..n {
            for j in (i + 1)..n {
                assert!(
                    l_data[i * n + j].abs() < 1e-10,
                    "L[{},{}] = {} should be 0",
                    i,
                    j,
                    l_data[i * n + j]
                );
            }
        }

        // Reconstruct: L @ L^T should equal A
        let a_data = a.data().unwrap();
        let mut llt = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..n {
                    acc += l_data[i * n + p] * l_data[j * n + p]; // L @ L^T
                }
                llt[i * n + j] = acc;
            }
        }

        for i in 0..n * n {
            assert!(
                (llt[i] - a_data[i]).abs() < 1e-10,
                "L @ L^T failed at index {}: {} vs {}",
                i,
                llt[i],
                a_data[i]
            );
        }
    }

    #[test]
    fn test_matrix_norm_identity() {
        // Frobenius norm of n x n identity = sqrt(n)
        let eye = t(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]);
        let n = matrix_norm(&eye).unwrap();
        assert!(n.is_scalar());
        let expected = (3.0f64).sqrt();
        assert!(
            (n.item().unwrap() - expected).abs() < 1e-10,
            "Frobenius norm of 3x3 identity = {} (expected {})",
            n.item().unwrap(),
            expected,
        );
    }

    #[test]
    fn test_pinv_full_rank_square() {
        // For a full-rank square matrix, pinv(A) == inv(A)
        let a = t(&[2.0, 1.0, 1.0, 3.0], &[2, 2]);
        let a_pinv = pinv(&a).unwrap();
        let a_inv = inv(&a).unwrap();
        let pinv_data = a_pinv.data().unwrap();
        let inv_data = a_inv.data().unwrap();
        for i in 0..4 {
            assert!(
                (pinv_data[i] - inv_data[i]).abs() < 1e-10,
                "pinv vs inv at index {}: {} vs {}",
                i,
                pinv_data[i],
                inv_data[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // eigh / eigvalsh (symmetric)
    // -----------------------------------------------------------------------

    #[test]
    fn test_eigh_diagonal_matrix() {
        // Diagonal matrix: eigenvalues are the diagonal entries (sorted),
        // eigenvectors are standard basis vectors.
        let a = t(&[3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0], &[3, 3]);
        let (w, _q) = eigh(&a).unwrap();
        let w_data = w.data().unwrap();
        // Ascending order: 1, 2, 3
        assert!((w_data[0] - 1.0).abs() < 1e-10);
        assert!((w_data[1] - 2.0).abs() < 1e-10);
        assert!((w_data[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_eigvalsh_matches_eigh() {
        // eigvalsh should agree with the eigenvalue half of eigh.
        let a = t(&[2.0, 1.0, 1.0, 2.0], &[2, 2]);
        let w_only = eigvalsh(&a).unwrap();
        let (w_full, _q) = eigh(&a).unwrap();
        let a_data = w_only.data().unwrap();
        let b_data = w_full.data().unwrap();
        for i in 0..2 {
            assert!(
                (a_data[i] - b_data[i]).abs() < 1e-10,
                "eigvalsh[{i}]={} vs eigh.0[{i}]={}",
                a_data[i],
                b_data[i]
            );
        }
    }

    #[test]
    fn test_eigh_reconstructs() {
        // A symmetric -> A = Q diag(w) Q^T.
        let a = t(&[4.0, 1.0, 1.0, 3.0], &[2, 2]);
        let (w, q) = eigh(&a).unwrap();
        let w_data = w.data().unwrap();
        let q_data = q.data().unwrap();
        // Reconstruct: result[i,j] = sum_k q[i,k] * w[k] * q[j,k]
        let n = 2;
        let mut recon = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += q_data[i * n + k] * w_data[k] * q_data[j * n + k];
                }
                recon[i * n + j] = acc;
            }
        }
        let a_data = a.data().unwrap();
        for i in 0..n * n {
            assert!(
                (recon[i] - a_data[i]).abs() < 1e-9,
                "eigh reconstruction at {i}: {} vs {}",
                recon[i],
                a_data[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // eig / eigvals (general; complex output)
    // -----------------------------------------------------------------------

    #[test]
    fn test_eigvals_diagonal_real() {
        // Diagonal: eigenvalues are diagonal entries (real).
        let a = t(&[2.0, 0.0, 0.0, 5.0], &[2, 2]);
        let w = eigvals(&a).unwrap();
        // Shape is [n, 2] with last dim = (re, im).
        assert_eq!(w.shape(), &[2, 2]);
        let d = w.data().unwrap();
        // The two eigenvalues should be {2, 5} in some order; collect real parts.
        let reals: Vec<f64> = (0..2).map(|i| d[i * 2]).collect();
        let imags: Vec<f64> = (0..2).map(|i| d[i * 2 + 1]).collect();
        for im in imags {
            assert!(im.abs() < 1e-10, "imag part should be 0, got {im}");
        }
        let mut sorted = reals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 2.0).abs() < 1e-10);
        assert!((sorted[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_eig_returns_complex_eigenvectors_shape() {
        let a = t(&[0.0, -1.0, 1.0, 0.0], &[2, 2]); // rotation 90°: complex eigenvalues ±i
        let (w, v) = eig(&a).unwrap();
        assert_eq!(w.shape(), &[2, 2]);
        assert_eq!(v.shape(), &[2, 2, 2]);
    }

    // -----------------------------------------------------------------------
    // lu
    // -----------------------------------------------------------------------

    #[test]
    fn test_lu_reconstructs() {
        let a = t(&[2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0, 8.0, 9.0], &[3, 3]);
        let (p, l, u) = lu(&a).unwrap();
        let p_data = p.data().unwrap();
        let l_data = l.data().unwrap();
        let u_data = u.data().unwrap();
        // Reconstruct PLU
        let n = 3;
        // L is [3,3] (k=min(3,3)=3); U is [3,3]
        let mut lu_prod = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += l_data[i * n + k] * u_data[k * n + j];
                }
                lu_prod[i * n + j] = acc;
            }
        }
        let mut plu = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += p_data[i * n + k] * lu_prod[k * n + j];
                }
                plu[i * n + j] = acc;
            }
        }
        let a_data = a.data().unwrap();
        for i in 0..n * n {
            assert!(
                (plu[i] - a_data[i]).abs() < 1e-9,
                "lu reconstruction at {i}: {} vs {}",
                plu[i],
                a_data[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // svdvals / lstsq
    // -----------------------------------------------------------------------

    #[test]
    fn test_svdvals_descending() {
        let a = t(&[3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0], &[3, 3]);
        let s = svdvals(&a).unwrap();
        let d = s.data().unwrap();
        // Descending: 3, 2, 1
        assert!((d[0] - 3.0).abs() < 1e-9);
        assert!((d[1] - 2.0).abs() < 1e-9);
        assert!((d[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_lstsq_overdetermined() {
        // y = 2x + 1 fit through (0,1), (1,3), (2,5), (3,7).
        let a = t(&[0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0], &[4, 2]);
        let b = t(&[1.0, 3.0, 5.0, 7.0], &[4]);
        let (sol, _resid, rank, sv) = lstsq(&a, &b, None).unwrap();
        let s = sol.data_vec().unwrap();
        // Coefficients should be (2, 1).
        assert!((s[0] - 2.0).abs() < 1e-9, "slope = {}", s[0]);
        assert!((s[1] - 1.0).abs() < 1e-9, "intercept = {}", s[1]);
        // Rank 2 (full column rank).
        assert_eq!(rank.data().unwrap(), &[2]);
        assert_eq!(sv.shape(), &[0]);
    }

    // -----------------------------------------------------------------------
    // matrix_power, matrix_rank, slogdet, cond
    // -----------------------------------------------------------------------

    #[test]
    fn test_matrix_power_zero_is_identity() {
        let a = t(&[2.0, 1.0, 0.0, 3.0], &[2, 2]);
        let r = matrix_power(&a, 0).unwrap();
        let d = r.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!(d[1].abs() < 1e-10);
        assert!(d[2].abs() < 1e-10);
        assert!((d[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_power_two_equals_self_squared() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let a2 = matrix_power(&a, 2).unwrap();
        // Hand-compute A @ A: [[7, 10], [15, 22]]
        let d = a2.data().unwrap();
        assert!((d[0] - 7.0).abs() < 1e-10);
        assert!((d[1] - 10.0).abs() < 1e-10);
        assert!((d[2] - 15.0).abs() < 1e-10);
        assert!((d[3] - 22.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_rank_full_rank_2x2() {
        let a = t(&[1.0, 2.0, 3.0, 5.0], &[2, 2]);
        let r = matrix_rank(&a, None).unwrap();
        assert!((r.data().unwrap()[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_rank_singular_2x2() {
        // Rows are scalar multiples — rank 1.
        let a = t(&[1.0, 2.0, 2.0, 4.0], &[2, 2]);
        let r = matrix_rank(&a, None).unwrap();
        assert!((r.data().unwrap()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_slogdet_identity() {
        let a = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let (sign, logabs) = slogdet(&a).unwrap();
        assert!((sign.data().unwrap()[0] - 1.0).abs() < 1e-10);
        assert!(logabs.data().unwrap()[0].abs() < 1e-10);
    }

    #[test]
    fn test_slogdet_negative_det() {
        // det = 1*4 - 2*3 = -2, sign = -1, log|det| = log(2)
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let (sign, logabs) = slogdet(&a).unwrap();
        assert!((sign.data().unwrap()[0] - (-1.0)).abs() < 1e-10);
        assert!((logabs.data().unwrap()[0] - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_cond_identity_is_one() {
        let a = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let c = cond(&a, 2.0).unwrap();
        assert!((c.data().unwrap()[0] - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // vector_norm
    // -----------------------------------------------------------------------

    #[test]
    fn test_vector_norm_l2() {
        let v = t(&[3.0, 4.0], &[2]);
        let n = vector_norm(&v, 2.0).unwrap();
        assert!((n.data().unwrap()[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_norm_l1() {
        let v = t(&[1.0, -2.0, 3.0, -4.0], &[4]);
        let n = vector_norm(&v, 1.0).unwrap();
        assert!((n.data().unwrap()[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_norm_inf() {
        let v = t(&[1.0, -7.0, 3.0, -4.0], &[4]);
        let n = vector_norm(&v, f64::INFINITY).unwrap();
        assert!((n.data().unwrap()[0] - 7.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // multi_dot, cross, diagonal
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_dot_chains_three() {
        // (A @ B) @ C
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let c = t(&[2.0, 0.0, 0.0, 2.0], &[2, 2]);
        let r = multi_dot(&[&a, &b, &c]).unwrap();
        // A @ I @ 2I = 2A
        let d = r.data().unwrap();
        assert!((d[0] - 2.0).abs() < 1e-10);
        assert!((d[1] - 4.0).abs() < 1e-10);
        assert!((d[2] - 6.0).abs() < 1e-10);
        assert!((d[3] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_basis_vectors() {
        // e1 × e2 = e3
        let e1 = t(&[1.0, 0.0, 0.0], &[3]);
        let e2 = t(&[0.0, 1.0, 0.0], &[3]);
        let r = cross(&e1, &e2, -1).unwrap();
        let d = r.data().unwrap();
        assert!(d[0].abs() < 1e-10);
        assert!(d[1].abs() < 1e-10);
        assert!((d[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_dim_zero_vs_dim_last_differ() {
        // Discriminating fixture for the "dim silently ignored" audit:
        // a [3, 3] tensor pair where dim=0 (column-wise 3-vectors) and
        // dim=-1 (row-wise 3-vectors) MUST produce different outputs.
        //
        // Row-major layout for the same nine elements:
        //   a = [[1, 2, 3],
        //        [4, 5, 6],
        //        [7, 8, 9]]
        //   b = [[9, 8, 7],
        //        [6, 5, 4],
        //        [3, 2, 1]]
        #[rustfmt::skip]
        let a = t(
            &[1.0, 2.0, 3.0,
              4.0, 5.0, 6.0,
              7.0, 8.0, 9.0],
            &[3, 3],
        );
        #[rustfmt::skip]
        let b = t(
            &[9.0, 8.0, 7.0,
              6.0, 5.0, 4.0,
              3.0, 2.0, 1.0],
            &[3, 3],
        );

        // dim=-1: cross product along each row.
        //   row 0: (1,2,3) × (9,8,7)
        //     = (2*7 - 3*8, 3*9 - 1*7, 1*8 - 2*9) = (-10, 20, -10)
        //   row 1: (4,5,6) × (6,5,4) = (5*4 - 6*5, 6*6 - 4*4, 4*5 - 5*6)
        //     = (-10, 20, -10)
        //   row 2: (7,8,9) × (3,2,1) = (8*1 - 9*2, 9*3 - 7*1, 7*2 - 8*3)
        //     = (-10, 20, -10)
        let r_last = cross(&a, &b, -1).unwrap();
        assert_eq!(r_last.shape(), &[3, 3]);
        let d_last = r_last.data().unwrap();
        let expect_last = [-10.0, 20.0, -10.0, -10.0, 20.0, -10.0, -10.0, 20.0, -10.0];
        for (got, exp) in d_last.iter().zip(expect_last.iter()) {
            assert!((got - exp).abs() < 1e-10, "dim=-1 got {got}, exp {exp}");
        }

        // dim=0: each column is a 3-vector. Column j of a × column j of b.
        //   col 0: (1,4,7) × (9,6,3)
        //     = (4*3 - 7*6, 7*9 - 1*3, 1*6 - 4*9) = (-30, 60, -30)
        //   col 1: (2,5,8) × (8,5,2)
        //     = (5*2 - 8*5, 8*8 - 2*2, 2*5 - 5*8) = (-30, 60, -30)
        //   col 2: (3,6,9) × (7,4,1)
        //     = (6*1 - 9*4, 9*7 - 3*1, 3*4 - 6*7) = (-30, 60, -30)
        // Laid out row-major in the [3, 3] output:
        //   [[-30, -30, -30],
        //    [ 60,  60,  60],
        //    [-30, -30, -30]]
        let r_first = cross(&a, &b, 0).unwrap();
        assert_eq!(r_first.shape(), &[3, 3]);
        let d_first = r_first.data().unwrap();
        let expect_first = [-30.0, -30.0, -30.0, 60.0, 60.0, 60.0, -30.0, -30.0, -30.0];
        for (got, exp) in d_first.iter().zip(expect_first.iter()) {
            assert!((got - exp).abs() < 1e-10, "dim=0 got {got}, exp {exp}");
        }

        // The two outputs MUST differ on this non-trivial fixture —
        // the audit failure was that they were identical because the
        // `dim` parameter was being silently ignored.
        assert_ne!(d_last, d_first);
    }

    #[test]
    fn test_cross_dim_out_of_range_errors() {
        let a = t(&[1.0, 0.0, 0.0], &[3]);
        let b = t(&[0.0, 1.0, 0.0], &[3]);
        assert!(cross(&a, &b, 5).is_err());
        assert!(cross(&a, &b, -2).is_err());
    }

    #[test]
    fn test_cross_dim_size_must_be_three() {
        // shape [2, 4] — neither axis is length 3.
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
        let b = t(&[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], &[2, 4]);
        assert!(cross(&a, &b, 0).is_err());
        assert!(cross(&a, &b, -1).is_err());
    }

    #[test]
    fn test_diagonal_main() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
        let d = diagonal(&a, 0).unwrap();
        assert_eq!(d.shape(), &[3]);
        assert_eq!(d.data().unwrap(), &[1.0, 5.0, 9.0]);
    }

    #[test]
    fn test_diagonal_offset_positive() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
        let d = diagonal(&a, 1).unwrap();
        // a[0,1], a[1,2] = 2, 6
        assert_eq!(d.data().unwrap(), &[2.0, 6.0]);
    }

    #[test]
    fn test_diagonal_offset_negative() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
        let d = diagonal(&a, -1).unwrap();
        // a[1,0], a[2,1] = 4, 8
        assert_eq!(d.data().unwrap(), &[4.0, 8.0]);
    }

    // -----------------------------------------------------------------------
    // tensorinv / tensorsolve smoke
    // -----------------------------------------------------------------------

    #[test]
    fn test_tensorinv_2x2_matrix_form() {
        // For a [2,2]-shaped matrix viewed as a (2,2) tensor at ind=1, the
        // tensor inverse is the matrix inverse.
        let a = t(&[4.0, 7.0, 2.0, 6.0], &[2, 2]);
        let inv_a = tensorinv(&a, 1).unwrap();
        // Hand-compute A^-1 = (1/10) * [[6, -7], [-2, 4]]
        let d = inv_a.data().unwrap();
        assert!((d[0] - 0.6).abs() < 1e-10);
        assert!((d[1] - (-0.7)).abs() < 1e-10);
        assert!((d[2] - (-0.2)).abs() < 1e-10);
        assert!((d[3] - 0.4).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // GPU discipline: every new fn returns InvalidArgument on CUDA tensors,
    // never silently downloads. We can't construct CUDA tensors in this
    // CPU-only test, but the require_cpu gate is exercised by the GPU
    // tests in ferrotorch-gpu.
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Linalg tail: solve_triangular / matrix_exp / ldl / householder / *_ex (#581)
    // -----------------------------------------------------------------------

    #[test]
    fn solve_triangular_lower_1d_b() {
        // L = [[1, 0], [2, 3]], b = [1, 8] → x: 1·x0 = 1 → x0=1; 2·1 + 3·x1 = 8 → x1 = 2.
        let a = t(&[1.0, 0.0, 2.0, 3.0], &[2, 2]);
        let b = t(&[1.0, 8.0], &[2]);
        let x = solve_triangular(&a, &b, false, false, false).unwrap();
        assert_eq!(x.shape(), &[2]);
        let d = x.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn solve_triangular_upper_1d_b() {
        // U = [[2, 1], [0, 4]], b = [4, 8] → x1=2, 2·x0 + 1·2 = 4 → x0=1.
        let a = t(&[2.0, 1.0, 0.0, 4.0], &[2, 2]);
        let b = t(&[4.0, 8.0], &[2]);
        let x = solve_triangular(&a, &b, true, false, false).unwrap();
        let d = x.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn solve_triangular_2d_b_multi_rhs() {
        // L = [[1, 0], [2, 3]], B = [[1, 2], [8, 13]] → X = [[1, 2], [2, 3]].
        let a = t(&[1.0, 0.0, 2.0, 3.0], &[2, 2]);
        let b = t(&[1.0, 2.0, 8.0, 13.0], &[2, 2]);
        let x = solve_triangular(&a, &b, false, false, false).unwrap();
        assert_eq!(x.shape(), &[2, 2]);
        let d = x.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[1] - 2.0).abs() < 1e-10);
        assert!((d[2] - 2.0).abs() < 1e-10);
        assert!((d[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn solve_triangular_unit_diag() {
        // L_unit = [[*, 0], [2, *]] (diag treated as 1), b = [1, 5]
        // → x0 = 1, x1 = 5 - 2·1 = 3.
        let a = t(&[99.0, 0.0, 2.0, 99.0], &[2, 2]);
        let b = t(&[1.0, 5.0], &[2]);
        let x = solve_triangular(&a, &b, false, false, true).unwrap();
        let d = x.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn solve_triangular_transpose_lower() {
        // A = lower [[2, 0], [3, 4]]. solve A^T x = b with A^T = [[2, 3], [0, 4]].
        // For b=[5, 8]: x1 = 2, then 2·x0 + 3·2 = 5 → x0 = -0.5.
        let a = t(&[2.0, 0.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 8.0], &[2]);
        let x = solve_triangular(&a, &b, false, true, false).unwrap();
        let d = x.data().unwrap();
        assert!((d[0] - (-0.5)).abs() < 1e-10);
        assert!((d[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn ldl_factor_pd_matrix() {
        // A = [[4, 2], [2, 3]] is PD. Expected L=[[1,0],[0.5,1]], D=[4, 2].
        // Verify A = L diag(D) L^T.
        let a = t(&[4.0, 2.0, 2.0, 3.0], &[2, 2]);
        let (l, d) = ldl_factor(&a).unwrap();
        let l_d = l.data().unwrap();
        let d_d = d.data().unwrap();
        // Reconstruct A_recon[i,j] = sum_k L[i,k] * D[k] * L[j,k].
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += l_d[i * n + k] * d_d[k] * l_d[j * n + k];
                }
                let expected = a.data().unwrap()[i * n + j];
                assert!(
                    (acc - expected).abs() < 1e-10,
                    "LDL reconstruction A[{i},{j}]: {acc} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn ldl_solve_pd_matches_solve() {
        // A = [[4, 2], [2, 3]], b = [6, 5]. Solve via ldl and via direct solve.
        let a = t(&[4.0, 2.0, 2.0, 3.0], &[2, 2]);
        let b = t(&[6.0, 5.0], &[2]);
        let (l, d) = ldl_factor(&a).unwrap();
        let x_ldl = ldl_solve(&l, &d, &b).unwrap();
        let x_ref = solve(&a, &b).unwrap();
        let xd = x_ldl.data().unwrap();
        let rd = x_ref.data().unwrap();
        for i in 0..2 {
            assert!(
                (xd[i] - rd[i]).abs() < 1e-9,
                "ldl_solve[{i}]={} vs {}",
                xd[i],
                rd[i]
            );
        }
    }

    #[test]
    fn matrix_exp_zero_is_identity() {
        let a = t(&[0.0, 0.0, 0.0, 0.0], &[2, 2]);
        let e = matrix_exp(&a).unwrap();
        let d = e.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1]).abs() < 1e-12);
        assert!((d[2]).abs() < 1e-12);
        assert!((d[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn matrix_exp_diagonal() {
        // expm(diag(a, b)) = diag(e^a, e^b).
        let a = t(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);
        let e = matrix_exp(&a).unwrap();
        let d = e.data().unwrap();
        assert!((d[0] - 1.0_f64.exp()).abs() < 1e-10);
        assert!(d[1].abs() < 1e-10);
        assert!(d[2].abs() < 1e-10);
        assert!((d[3] - 2.0_f64.exp()).abs() < 1e-10);
    }

    #[test]
    fn matrix_exp_skew_symmetric_2x2_is_rotation() {
        // expm([[0, -t], [t, 0]]) = [[cos t, -sin t], [sin t, cos t]].
        let theta = 0.5_f64;
        let a = t(&[0.0, -theta, theta, 0.0], &[2, 2]);
        let e = matrix_exp(&a).unwrap();
        let d = e.data().unwrap();
        assert!((d[0] - theta.cos()).abs() < 1e-10);
        assert!((d[1] + theta.sin()).abs() < 1e-10);
        assert!((d[2] - theta.sin()).abs() < 1e-10);
        assert!((d[3] - theta.cos()).abs() < 1e-10);
    }

    #[test]
    fn cholesky_ex_succeeds_for_pd() {
        let a = t(&[4.0, 2.0, 2.0, 3.0], &[2, 2]);
        let (_l, info) = cholesky_ex(&a).unwrap();
        assert_eq!(info.shape(), &[] as &[usize]);
        assert!(info.data().unwrap()[0].abs() < 1e-12);
    }

    #[test]
    fn cholesky_ex_returns_nonzero_info_for_indefinite() {
        // Negative-definite-ish: ferray cholesky should fail.
        let a = t(&[-1.0, 0.0, 0.0, -1.0], &[2, 2]);
        let (_l, info) = cholesky_ex(&a).unwrap();
        assert!(info.data().unwrap()[0] != 0.0);
    }

    #[test]
    fn inv_ex_succeeds_for_invertible() {
        let a = t(&[2.0, 0.0, 0.0, 4.0], &[2, 2]);
        let (inv_a, info) = inv_ex(&a).unwrap();
        assert!(info.data().unwrap()[0].abs() < 1e-12);
        let d = inv_a.data().unwrap();
        assert!((d[0] - 0.5).abs() < 1e-10);
        assert!((d[3] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn inv_ex_singular_returns_nonzero_info() {
        let a = t(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
        let (_inv_a, info) = inv_ex(&a).unwrap();
        assert!(info.data().unwrap()[0] != 0.0);
    }

    #[test]
    fn solve_ex_succeeds() {
        let a = t(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);
        let b = t(&[3.0, 4.0], &[2]);
        let (x, info) = solve_ex(&a, &b).unwrap();
        assert!(info.data().unwrap()[0].abs() < 1e-12);
        let d = x.data().unwrap();
        assert!((d[0] - 3.0).abs() < 1e-10);
        assert!((d[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn solve_ex_singular_returns_nonzero_info() {
        let a = t(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
        let b = t(&[1.0, 2.0], &[2]);
        let (_x, info) = solve_ex(&a, &b).unwrap();
        assert!(info.data().unwrap()[0] != 0.0);
    }

    #[test]
    fn householder_product_identity_when_no_reflectors() {
        // k=0 → tau is empty → torch returns the leading 0 columns, shape [3,0].
        // (torch.linalg.householder_product(zeros(3,0), zeros(0)) → shape [3,0].)
        let v =
            Tensor::from_storage(TensorStorage::cpu(Vec::<f64>::new()), vec![3, 0], false).unwrap();
        let tau =
            Tensor::from_storage(TensorStorage::cpu(Vec::<f64>::new()), vec![0], false).unwrap();
        let q = householder_product(&v, &tau).unwrap();
        assert_eq!(q.shape(), &[3, 0]);
        assert!(q.data().unwrap().is_empty());
        // The FULL reconstruction (used by the backward) is still I_3.
        let q_full = householder_product_full(&v, &tau).unwrap();
        assert_eq!(q_full.shape(), &[3, 3]);
        let d = q_full.data().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((d[i * 3 + j] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn householder_product_single_reflection_is_orthogonal() {
        // Single Householder vector v0 = [1, 0]^T (unit at row 0; below is 0)
        // with tau = 2 → I - 2·v0·v0^T = I - 2·e_0·e_0^T = diag(-1, 1).
        // V is [m=2, k=1]: v[0,0] is the implicit unit (we store anything;
        // householder_product overrides with 1), v[1,0] = 0 (below row 0).
        // torch.linalg.householder_product returns the leading k=1 column → [2,1].
        let v = Tensor::from_storage(TensorStorage::cpu(vec![0.0_f64, 0.0]), vec![2, 1], false)
            .unwrap();
        let tau = Tensor::from_storage(TensorStorage::cpu(vec![2.0_f64]), vec![1], false).unwrap();
        let q = householder_product(&v, &tau).unwrap();
        assert_eq!(q.shape(), &[2, 1]);
        let d = q.data().unwrap();
        // First column of Q = diag(-1, 1) is [-1, 0]^T.
        assert!((d[0] + 1.0).abs() < 1e-12);
        assert!(d[1].abs() < 1e-12);
    }
}
