//! Cross-world integer ops — crosslink #1185 Phase 2c.
//!
//! `argmax` / `argmin` (→ `IntTensor<i64>`), `index_select` / `gather` driven
//! by a GPU-resident `IntTensor` index, and dtype casts (`Tensor::to_int`,
//! `IntTensor::to_float`, `IntTensor::cast` GPU path). Each op runs on CUDA when
//! the input is `is_cuda()` (real PTX kernel; the result stays GPU-resident —
//! no `.to(Cpu)`, no host readback) and on CPU otherwise via a reference loop
//! matching the same PyTorch semantics the GPU kernels implement.
//!
//! These unblock the Llama generation loop, which today round-trips to CPU for
//! argmax sampling and uses raw cudarc slices for token-id embedding gather.

use crate::device::Device;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::int_tensor::{IntElement, IntTensor};
use crate::shape::normalize_axis;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// Factorise `shape` around `dim` into `(outer, dim_size, inner)` for the
/// `[outer, dim_size, inner]` kernel layout. `outer = prod(shape[..dim])`,
/// `inner = prod(shape[dim+1..])`.
fn factor(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let outer: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    (outer, dim_size, inner)
}

/// Shape after removing axis `dim` (the argmax/argmin along-dim output shape).
fn shape_without(shape: &[usize], dim: usize) -> Vec<usize> {
    let mut s = shape.to_vec();
    s.remove(dim);
    s
}

// ── argmax / argmin reference (CPU), generic over a comparator ──────────────

/// First-occurrence argmax/argmin over `data` laid out `[outer, dim_size,
/// inner]`. `better(candidate, current)` returns true iff `candidate` strictly
/// beats `current` (so ties keep the earliest index — PyTorch parity).
fn arg_reduce_ref<V: Copy>(
    data: &[V],
    outer: usize,
    dim_size: usize,
    inner: usize,
    better: impl Fn(V, V) -> bool,
) -> Vec<i64> {
    let mut out = vec![0i64; outer * inner];
    for o in 0..outer {
        for k in 0..inner {
            let base = o * dim_size * inner + k;
            let mut best_j = 0usize;
            let mut best = data[base];
            for j in 1..dim_size {
                let v = data[base + j * inner];
                if better(v, best) {
                    best = v;
                    best_j = j;
                }
            }
            out[o * inner + k] = best_j as i64;
        }
    }
    out
}

/// Run argmax/argmin on a float `Tensor<T>`, returning `IntTensor<i64>`.
/// `dim = None` reduces the flattened tensor to a 0-d scalar index.
fn tensor_arg<T: Float>(
    input: &Tensor<T>,
    dim: Option<isize>,
    is_max: bool,
) -> FerrotorchResult<IntTensor<i64>> {
    let op = if is_max { "argmax" } else { "argmin" };
    if input.numel() == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("{op}: cannot reduce an empty tensor"),
        });
    }
    let input = input.contiguous()?;
    let (outer, dim_size, inner, out_shape) = match dim {
        None => (1usize, input.numel(), 1usize, Vec::new()),
        Some(d) => {
            let d = normalize_axis(d, input.ndim())?;
            let (o, ds, inn) = factor(input.shape(), d);
            (o, ds, inn, shape_without(input.shape(), d))
        }
    };

    if input.is_cuda() {
        let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let h = input.gpu_handle()?;
        let out_h = if is_max {
            backend.argmax(h, outer, dim_size, inner)?
        } else {
            backend.argmin(h, outer, dim_size, inner)?
        };
        Ok(IntTensor::from_gpu_handle(out_h, out_shape))
    } else {
        let data = input.data_vec()?;
        let out = if is_max {
            arg_reduce_ref(&data, outer, dim_size, inner, |c, b| c > b)
        } else {
            arg_reduce_ref(&data, outer, dim_size, inner, |c, b| c < b)
        };
        IntTensor::<i64>::from_vec(out, out_shape)
    }
}

/// Run argmax/argmin on an `IntTensor<I>`, returning `IntTensor<i64>`.
fn inttensor_arg<I: IntElement>(
    input: &IntTensor<I>,
    dim: Option<isize>,
    is_max: bool,
) -> FerrotorchResult<IntTensor<i64>> {
    let op = if is_max { "argmax" } else { "argmin" };
    if input.numel() == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("{op}: cannot reduce an empty tensor"),
        });
    }
    let (outer, dim_size, inner, out_shape) = match dim {
        None => (1usize, input.numel(), 1usize, Vec::new()),
        Some(d) => {
            let d = normalize_axis(d, input.ndim())?;
            let (o, ds, inn) = factor(input.shape(), d);
            (o, ds, inn, shape_without(input.shape(), d))
        }
    };

    if input.is_cuda() {
        let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let h = input.gpu_handle()?;
        let out_h = if is_max {
            backend.argmax(h, outer, dim_size, inner)?
        } else {
            backend.argmin(h, outer, dim_size, inner)?
        };
        Ok(IntTensor::from_gpu_handle(out_h, out_shape))
    } else {
        let data: Vec<i64> = input.data()?.iter().map(|v| v.to_i64()).collect();
        let out = if is_max {
            arg_reduce_ref(&data, outer, dim_size, inner, |c, b| c > b)
        } else {
            arg_reduce_ref(&data, outer, dim_size, inner, |c, b| c < b)
        };
        IntTensor::<i64>::from_vec(out, out_shape)
    }
}

// ── index_select / gather host references (raw element copy) ────────────────

fn index_select_ref<V: Copy>(
    data: &[V],
    indices: &[i64],
    outer: usize,
    in_dim: usize,
    inner: usize,
    zero: V,
) -> Vec<V> {
    let out_dim = indices.len();
    let mut out = vec![zero; outer * out_dim * inner];
    for o in 0..outer {
        for (i, &sel) in indices.iter().enumerate() {
            let sel = sel as usize;
            for k in 0..inner {
                let src = o * in_dim * inner + sel * inner + k;
                out[(o * out_dim + i) * inner + k] = data[src];
            }
        }
    }
    out
}

fn gather_ref<V: Copy>(
    data: &[V],
    indices: &[i64],
    outer: usize,
    in_dim: usize,
    out_dim: usize,
    inner: usize,
    zero: V,
) -> Vec<V> {
    let mut out = vec![zero; outer * out_dim * inner];
    for o in 0..outer {
        for i in 0..out_dim {
            for k in 0..inner {
                let t = (o * out_dim + i) * inner + k;
                let sel = indices[t] as usize;
                let src = o * in_dim * inner + sel * inner + k;
                out[t] = data[src];
            }
        }
    }
    out
}

/// Read an `IntTensor<I>` index as host `Vec<i64>` (CPU references only — the
/// GPU path never calls this, so no host round-trip happens for GPU inputs).
fn index_as_i64<I: IntElement>(index: &IntTensor<I>) -> FerrotorchResult<Vec<i64>> {
    Ok(index.data()?.iter().map(|v| v.to_i64()).collect())
}

// ===========================================================================
// Tensor<T> high-level API
// ===========================================================================

impl<T: Float> Tensor<T> {
    /// Index of the maximum value (PyTorch `torch.argmax`), as `IntTensor<i64>`.
    ///
    /// `dim = None` flattens and returns a 0-d index. `dim = Some(d)` reduces
    /// along `d` (negative indices allowed). Ties resolve to the FIRST (lowest)
    /// index. GPU-resident result when `self` is on CUDA.
    pub fn argmax(&self, dim: Option<isize>) -> FerrotorchResult<IntTensor<i64>> {
        tensor_arg(self, dim, true)
    }

    /// Index of the minimum value (PyTorch `torch.argmin`). See [`Self::argmax`].
    pub fn argmin(&self, dim: Option<isize>) -> FerrotorchResult<IntTensor<i64>> {
        tensor_arg(self, dim, false)
    }

    /// `index_select(dim, indices)` (PyTorch `torch.index_select`) using a
    /// GPU-resident-or-CPU `IntTensor` index. The `indices` tensor must be 1-D.
    /// Output keeps `self`'s dtype; shape is `self.shape` with `shape[dim]`
    /// replaced by `indices.numel()`. On CUDA, `self` and `indices` must be on
    /// the same device; the result stays GPU-resident.
    pub fn index_select<I: IntElement>(
        &self,
        dim: isize,
        indices: &IntTensor<I>,
    ) -> FerrotorchResult<Tensor<T>> {
        if indices.ndim() > 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "index_select: indices must be 1-D, got shape {:?}",
                    indices.shape()
                ),
            });
        }
        let input = self.contiguous()?;
        let d = normalize_axis(dim, input.ndim())?;
        let (outer, in_dim, inner) = factor(input.shape(), d);
        let out_dim = indices.numel();
        let mut out_shape = input.shape().to_vec();
        out_shape[d] = out_dim;

        if input.is_cuda() {
            check_same_device(input.device(), indices.device(), "index_select")?;
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let h = backend.index_select_intidx(
                input.gpu_handle()?,
                indices.gpu_handle()?,
                outer,
                in_dim,
                out_dim,
                inner,
            )?;
            Tensor::from_storage(TensorStorage::gpu(h), out_shape, false)
        } else {
            let data = input.data_vec()?;
            let idx = index_as_i64(&indices.to(Device::Cpu)?)?;
            let out =
                index_select_ref(&data, &idx, outer, in_dim, inner, <T as num_traits::Zero>::zero());
            Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)
        }
    }

    /// `gather(dim, index)` (PyTorch `torch.gather`) using a GPU-resident-or-CPU
    /// `IntTensor` index. `index` must have the same ndim as `self`; output has
    /// `index`'s shape and `self`'s dtype. On CUDA the result stays resident.
    pub fn gather<I: IntElement>(
        &self,
        dim: isize,
        index: &IntTensor<I>,
    ) -> FerrotorchResult<Tensor<T>> {
        let input = self.contiguous()?;
        let d = normalize_axis(dim, input.ndim())?;
        gather_check_shapes(input.shape(), index.shape(), d, "gather")?;
        let (outer, in_dim, inner) = factor(input.shape(), d);
        let out_dim = index.shape()[d];
        let out_shape = index.shape().to_vec();

        if input.is_cuda() {
            check_same_device(input.device(), index.device(), "gather")?;
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let h = backend.gather_intidx(
                input.gpu_handle()?,
                index.gpu_handle()?,
                outer,
                in_dim,
                out_dim,
                inner,
            )?;
            Tensor::from_storage(TensorStorage::gpu(h), out_shape, false)
        } else {
            let data = input.data_vec()?;
            let idx = index_as_i64(&index.to(Device::Cpu)?)?;
            let out = gather_ref(
                &data, &idx, outer, in_dim, out_dim, inner, <T as num_traits::Zero>::zero(),
            );
            Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)
        }
    }

    /// Cast this float tensor to `IntTensor<I>` (PyTorch `.to(int)`):
    /// **truncate toward zero**. GPU-resident result when `self` is on CUDA.
    pub fn to_int<I: IntElement>(&self) -> FerrotorchResult<IntTensor<I>> {
        let input = self.contiguous()?;
        let shape = input.shape().to_vec();
        if input.is_cuda() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let h = backend.cast_f_to_i(input.gpu_handle()?, I::dtype())?;
            Ok(IntTensor::from_gpu_handle(h, shape))
        } else {
            let data = input.data_vec()?;
            let mut out: Vec<I> = Vec::with_capacity(data.len());
            for &v in &data {
                // Truncate toward zero (PyTorch `.to(int)`): drop the fraction.
                let truncated = num_traits::Float::trunc(v);
                let as_i64 = float_to_i64_trunc(truncated);
                out.push(I::try_from_i64(as_i64).ok_or(FerrotorchError::InvalidArgument {
                    message: format!("to_int: value out of range for {}", I::dtype_name()),
                })?);
            }
            IntTensor::<I>::from_vec(out, shape)
        }
    }
}

/// Convert an already-truncated float to i64 (saturating at the i64 range,
/// matching the CPU `as` cast which saturates rather than wraps).
fn float_to_i64_trunc<T: Float>(v: T) -> i64 {
    // `T: Float` -> f64 is lossless for f32/bf16/f16 and exact enough for the
    // integer range here; `as i64` on f64 saturates (Rust 1.45+ semantics),
    // matching PyTorch's clamp-on-overflow for `.to(int64)`.
    let f: f64 = num_traits::ToPrimitive::to_f64(&v).unwrap_or(0.0);
    f as i64
}

// ===========================================================================
// IntTensor<I> high-level API
// ===========================================================================

impl<I: IntElement> IntTensor<I> {
    /// Index of the maximum value, as `IntTensor<i64>`. See
    /// [`Tensor::argmax`](crate::tensor::Tensor::argmax).
    pub fn argmax(&self, dim: Option<isize>) -> FerrotorchResult<IntTensor<i64>> {
        inttensor_arg(self, dim, true)
    }

    /// Index of the minimum value, as `IntTensor<i64>`.
    pub fn argmin(&self, dim: Option<isize>) -> FerrotorchResult<IntTensor<i64>> {
        inttensor_arg(self, dim, false)
    }

    /// `index_select(dim, indices)` on integer data (1-D `indices`). Output
    /// keeps this tensor's int dtype; GPU-resident when on CUDA.
    pub fn index_select<J: IntElement>(
        &self,
        dim: isize,
        indices: &IntTensor<J>,
    ) -> FerrotorchResult<IntTensor<I>> {
        if indices.ndim() > 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "index_select: indices must be 1-D, got shape {:?}",
                    indices.shape()
                ),
            });
        }
        let d = normalize_axis(dim, self.ndim())?;
        let (outer, in_dim, inner) = factor(self.shape(), d);
        let out_dim = indices.numel();
        let mut out_shape = self.shape().to_vec();
        out_shape[d] = out_dim;

        if self.is_cuda() {
            check_same_device(self.device(), indices.device(), "index_select")?;
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let h = backend.index_select_intidx(
                self.gpu_handle()?,
                indices.gpu_handle()?,
                outer,
                in_dim,
                out_dim,
                inner,
            )?;
            Ok(IntTensor::from_gpu_handle(h, out_shape))
        } else {
            let data = self.data()?;
            let idx = index_as_i64(&indices.to(Device::Cpu)?)?;
            let zero = I::try_from_i64(0).expect("0 is in range for i32/i64");
            let out = index_select_ref(data, &idx, outer, in_dim, inner, zero);
            IntTensor::<I>::from_vec(out, out_shape)
        }
    }

    /// `gather(dim, index)` on integer data. `index` must match this tensor's
    /// ndim; output has `index`'s shape and this tensor's int dtype.
    pub fn gather<J: IntElement>(
        &self,
        dim: isize,
        index: &IntTensor<J>,
    ) -> FerrotorchResult<IntTensor<I>> {
        let d = normalize_axis(dim, self.ndim())?;
        gather_check_shapes(self.shape(), index.shape(), d, "gather")?;
        let (outer, in_dim, inner) = factor(self.shape(), d);
        let out_dim = index.shape()[d];
        let out_shape = index.shape().to_vec();

        if self.is_cuda() {
            check_same_device(self.device(), index.device(), "gather")?;
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let h = backend.gather_intidx(
                self.gpu_handle()?,
                index.gpu_handle()?,
                outer,
                in_dim,
                out_dim,
                inner,
            )?;
            Ok(IntTensor::from_gpu_handle(h, out_shape))
        } else {
            let data = self.data()?;
            let idx = index_as_i64(&index.to(Device::Cpu)?)?;
            let zero = I::try_from_i64(0).expect("0 is in range for i32/i64");
            let out = gather_ref(data, &idx, outer, in_dim, out_dim, inner, zero);
            IntTensor::<I>::from_vec(out, out_shape)
        }
    }

    /// Cast this integer tensor to a float `Tensor<T>` (PyTorch `.to(float)`),
    /// round-to-nearest-even. GPU-resident result when `self` is on CUDA.
    pub fn to_float<T: Float>(&self) -> FerrotorchResult<Tensor<T>> {
        let shape = self.shape().to_vec();
        if self.is_cuda() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let h = backend.cast_i_to_f(self.gpu_handle()?, T::dtype())?;
            Tensor::from_storage(TensorStorage::gpu(h), shape, false)
        } else {
            let data = self.data()?;
            let mut out: Vec<T> = Vec::with_capacity(data.len());
            for &v in data {
                out.push(
                    num_traits::NumCast::from(v.to_i64()).ok_or(FerrotorchError::InvalidArgument {
                        message: "to_float: integer not representable in target float".into(),
                    })?,
                );
            }
            Tensor::from_storage(TensorStorage::cpu(out), shape, false)
        }
    }

    /// GPU path for [`IntTensor::cast`] (i32 ↔ i64). Returns `None` so the
    /// caller's CPU path handles non-CUDA tensors; `Some(Ok/Err)` on CUDA.
    pub(crate) fn cast_gpu<J: IntElement>(&self) -> Option<FerrotorchResult<IntTensor<J>>> {
        if !self.is_cuda() {
            return None;
        }
        let shape = self.shape().to_vec();
        let backend = match crate::gpu_dispatch::gpu_backend() {
            Some(b) => b,
            None => return Some(Err(FerrotorchError::DeviceUnavailable)),
        };
        let h = match self.gpu_handle() {
            Ok(h) => h,
            Err(e) => return Some(Err(e)),
        };
        Some(
            backend
                .cast_i_to_i(h, J::dtype())
                .map(|out_h| IntTensor::from_gpu_handle(out_h, shape)),
        )
    }
}

// ── shared validation ───────────────────────────────────────────────────────

fn check_same_device(a: Device, b: Device, op: &str) -> FerrotorchResult<()> {
    if a != b {
        return Err(FerrotorchError::DeviceMismatch { expected: a, got: b });
    }
    let _ = op;
    Ok(())
}

/// `gather` shape rule: `index.ndim() == input.ndim()`, and for every axis
/// `ax != dim`, `index.shape[ax] <= input.shape[ax]` (PyTorch allows the index
/// to be smaller off the gather axis). The gather axis itself is unconstrained
/// in size (each element selects independently).
fn gather_check_shapes(
    input_shape: &[usize],
    index_shape: &[usize],
    dim: usize,
    op: &str,
) -> FerrotorchResult<()> {
    if index_shape.len() != input_shape.len() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "{op}: index ndim {} != input ndim {}",
                index_shape.len(),
                input_shape.len()
            ),
        });
    }
    for (ax, (&isz, &xsz)) in index_shape.iter().zip(input_shape.iter()).enumerate() {
        if ax != dim && isz > xsz {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "{op}: index dim {ax} size {isz} exceeds input size {xsz}"
                ),
            });
        }
    }
    Ok(())
}

