//! ## REQ status (per `.design/ferrotorch-core/tensor.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl `Tensor<T>` + `TensorInner`; non-test consumer every workspace op. |
//! | REQ-2 | SHIPPED | impl `from_storage`; non-test consumer `creation::zeros`/`ones`/`tensor`. |
//! | REQ-3 | SHIPPED | impl `from_operation`; non-test consumer every grad-attaching forward op (e.g. `grad_fns::arithmetic::add_inner`). |
//! | REQ-4 | SHIPPED | impl `view_reshape` / `view_operation` / `stride_view` / `stride_view_operation`; non-test consumer shape grad_fns + `methods::view_t`. |
//! | REQ-5 | SHIPPED | impl `to(device)` incl. non-contig CUDA->CPU materialise (#802); non-test consumer `Tensor::cuda`, `Tensor::cpu`, state-dict transfer. |
//! | REQ-6 | SHIPPED | impl `to_pinned`; non-test consumer `ferrotorch-data::DataLoader` `pin_memory(true)`. |
//! | REQ-7 | SHIPPED | impl `data`, `data_ref`, `data_vec`; non-test consumer every CPU tensor reader (`pruning`, `signal::windows`). |
//! | REQ-8 | SHIPPED | impl `grad`, `set_grad`, `zero_grad`, `accumulate_grad` GPU fast path; non-test consumer `autograd::backward`. |
//! | REQ-9 | SHIPPED | impl `detach`, `requires_grad_`; non-test consumer `autograd::no_grad` + model init. |
//! | REQ-10 | SHIPPED | impl `is_contiguous`, `is_contiguous_for`, `to_memory_format`, `materialize_format`; non-test consumer `ferrotorch-nn::Conv2d` channels-last. |
//! | REQ-11 | SHIPPED | impl `as_strided` family in `stride_tricks.rs`; non-test consumer `crate::einsum`. |
//! | REQ-12 | SHIPPED | impl `gpu_handle`; non-test consumer every CUDA kernel dispatch. |
//! | REQ-13 | SHIPPED | impl `update_data` / `update_storage` / `update_storage_and_shape` / `with_gpu_handle_mut`; non-test consumer optimizer `step()` plus `add_scaled_out`. |
//! | REQ-14 | SHIPPED | impl `register_hook`, `register_post_accumulate_grad_hook`, `remove_hook`; non-test consumer `autograd::hooks` integration. |
//! | REQ-15 | SHIPPED | impl `masked_fill`, `masked_select`; non-test consumer `grad_fns::indexing` + `ops::indexing`. |
//! | REQ-16 | SHIPPED | impl `item`; non-test consumer scalar-loss readout in training loops. |
//! | REQ-17 | SHIPPED | impl `into_storage_and_shape`; non-test consumer `accumulate_grad`. |
//! | REQ-18 | SHIPPED | impl identity / refcount inspection; non-test consumer `autograd::graph` in-place gating. |
//! | REQ-19 | SHIPPED | impl `trait GradFn<T>`; non-test consumer every grad_fn struct in `grad_fns/*`. |
//! | REQ-20 | SHIPPED | impl `enum MemoryFormat`; non-test consumer `Tensor::to_memory_format`, `ferrotorch-nn::Conv2d`. |

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::device::Device;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::shape::{
    checked_byte_count, checked_c_contiguous_strides, checked_channels_last_3d_strides,
    checked_channels_last_strides, checked_numel,
};
use crate::storage::TensorStorage;

/// Describes the physical memory layout of a tensor.
///
/// The *shape* (logical dimension order) never changes — only the strides are
/// rearranged so that the underlying data is stored in a different order.
///
/// [CL-309] WU-05: channels-last memory format support
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryFormat {
    /// Standard C-contiguous / row-major layout (NCHW for 4D tensors).
    Contiguous,
    /// Channels-last layout for 4D tensors: physical order is NHWC.
    /// The shape remains `[N, C, H, W]`, but strides are `[H*W*C, 1, W*C, C]`.
    ChannelsLast,
    /// Channels-last layout for 5D tensors: physical order is NDHWC.
    /// The shape remains `[N, C, D, H, W]`, but strides are `[D*H*W*C, 1, H*W*C, W*C, C]`.
    ChannelsLast3d,
}

/// Unique identifier for a tensor, used for gradient accumulation.
static NEXT_TENSOR_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A unique, monotonically increasing tensor identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(u64);

impl TensorId {
    fn next() -> Self {
        Self(NEXT_TENSOR_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

fn decode_device_bytes<T: Float>(op: &str, bytes: &[u8]) -> FerrotorchResult<Vec<T>> {
    let elem_size = std::mem::size_of::<T>();
    if !bytes.len().is_multiple_of(elem_size) {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "{op}: D2H readback of {} bytes is not a multiple of \
                 size_of::<{}>()={elem_size}",
                bytes.len(),
                std::any::type_name::<T>()
            ),
        });
    }

    let len = bytes.len() / elem_size;
    let mut data: Vec<T> = Vec::with_capacity(len);
    // SAFETY: `data` was freshly allocated just above with capacity `len`
    // under `Layout::array::<T>(len)`, so its pointer is valid for
    // `len * elem_size` bytes of writes and correctly aligned for `T` by
    // construction. The source `bytes` is valid for the same number of bytes
    // of reads (checked above); `u8` copies impose no source alignment
    // requirement, and the allocations are distinct. `T` is a plain numeric
    // Float element type with no padding or invalid bit patterns.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            data.as_mut_ptr().cast::<u8>(),
            len * elem_size,
        );
        data.set_len(len);
    }
    Ok(data)
}

fn gather_logical_from_slice<T: Float>(
    op: &'static str,
    storage: &[T],
    shape: &[usize],
    strides: &[isize],
    offset: usize,
) -> FerrotorchResult<Vec<T>> {
    crate::stride_tricks::validate_bounds(op, shape, strides, offset, storage.len())?;
    let numel = checked_numel(shape, op)?;
    let ndim = shape.len();
    let mut result = Vec::with_capacity(numel);
    let mut indices = vec![0usize; ndim];

    for _ in 0..numel {
        let mut flat = offset as isize;
        for d in 0..ndim {
            flat += indices[d] as isize * strides[d];
        }
        result.push(storage[flat as usize]);

        for d in (0..ndim).rev() {
            indices[d] += 1;
            if indices[d] < shape[d] {
                break;
            }
            indices[d] = 0;
        }
    }

    Ok(result)
}

/// The backward function trait for reverse-mode automatic differentiation.
///
/// Every differentiable operation implements this trait. The autograd engine
/// calls `backward()` during the reverse pass, passing the upstream gradient
/// and receiving gradients for each input.
pub trait GradFn<T: Float>: Send + Sync + fmt::Debug {
    /// Compute gradients of inputs given gradient of output.
    ///
    /// Returns one `Option<Tensor<T>>` per input: `None` for inputs that
    /// don't require gradients.
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>>;

    /// References to input tensors for graph traversal.
    fn inputs(&self) -> Vec<&Tensor<T>>;

    /// Name of this operation (e.g., "AddBackward", "MatmulBackward").
    fn name(&self) -> &'static str;

    /// Scalar parameters saved by this backward node (e.g. the exponent for
    /// `PowBackward`).  The JIT tracer uses these values to faithfully
    /// reconstruct parameterised IR ops.
    ///
    /// The default implementation returns an empty slice; only backward nodes
    /// that carry scalar hyperparameters need to override this method.
    fn scalar_args(&self) -> Vec<f64> {
        vec![]
    }
}

/// Inner storage for a tensor, shared via `Arc`.
///
/// `Tensor<T>` is a thin `Arc` wrapper around this struct. Cloning a tensor
/// clones the `Arc`, so all copies share the same identity, data, and grad
/// storage. This is essential for autograd: the backward engine writes
/// gradients to the same `TensorInner` that the user holds.
struct TensorInner<T: Float> {
    id: TensorId,
    storage: Arc<TensorStorage<T>>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
    grad: Mutex<Option<Box<Tensor<T>>>>,
    grad_fn: Mutex<Option<Arc<dyn GradFn<T>>>>,
    requires_grad: AtomicBool,
    is_leaf: AtomicBool,
    /// Hook storage for gradient hooks and post-accumulate-grad hooks.
    hooks: Mutex<crate::autograd::hooks::HookStorage<T>>,
}

/// The central type. A dynamically-shaped tensor with gradient tracking
/// and device placement.
///
/// Internally an `Arc<TensorInner>` — cloning a tensor is cheap and
/// preserves identity. Two clones of the same tensor share the same
/// data, grad, and TensorId.
///
/// # Type parameter
///
/// `T` must implement [`Float`] — currently `f32` or `f64`. This bound
/// ensures the tensor can participate in gradient computation.
pub struct Tensor<T: Float = f32> {
    inner: Arc<TensorInner<T>>,
    saved_version: Option<u64>,
}

// --- Construction ---

impl<T: Float> Tensor<T> {
    /// Create a new leaf tensor from raw components.
    pub fn from_storage(
        storage: TensorStorage<T>,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> FerrotorchResult<Self> {
        let numel = checked_numel(&shape, "Tensor::from_storage")?;

        if numel > storage.len() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "shape {:?} requires {} elements but storage has {}",
                    shape,
                    numel,
                    storage.len()
                ),
            });
        }

        let strides = checked_c_contiguous_strides(&shape, "Tensor::from_storage")?;

        Ok(Self {
            inner: Arc::new(TensorInner {
                id: TensorId::next(),
                storage: Arc::new(storage),
                shape,
                strides,
                offset: 0,
                grad: Mutex::new(None),
                grad_fn: Mutex::new(None),
                requires_grad: AtomicBool::new(requires_grad),
                is_leaf: AtomicBool::new(true),
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
            }),
            saved_version: None,
        })
    }

    /// Create a view of this tensor with a different shape, sharing the
    /// same underlying storage. Zero-copy — no data movement.
    ///
    /// The new shape must have the same total number of elements.
    /// Non-contiguous tensors are materialized first (requires a copy).
    pub fn view_reshape(&self, new_shape: Vec<usize>) -> FerrotorchResult<Self> {
        // Non-contiguous tensors must be materialized first — a view over
        // non-contiguous storage with new strides would read wrong elements.
        // Use the device-aware `contiguous()` path so a non-contiguous CUDA
        // tensor is gathered on-device via `strided_copy_*` rather than
        // demoted to CPU storage. The previous `data_vec() + TensorStorage::cpu`
        // path silently moved GPU tensors to host on every reshape after
        // a stride-view op (narrow / select / permute), which broke the
        // §3 PyTorch-parity contract for downstream LSTM/GRU/RNN forward
        // paths that compose narrow + squeeze on GPU inputs (#750).
        if !self.is_contiguous() {
            return self.contiguous()?.view_reshape(new_shape);
        }

        let new_numel = checked_numel(&new_shape, "view_reshape")?;
        if new_numel != self.numel() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "view_reshape: new shape {:?} ({} elements) vs old {:?} ({} elements)",
                    new_shape,
                    new_numel,
                    self.shape(),
                    self.numel()
                ),
            });
        }
        let strides = checked_c_contiguous_strides(&new_shape, "view_reshape")?;
        Ok(Self {
            inner: Arc::new(TensorInner {
                id: TensorId::next(),
                storage: Arc::clone(&self.inner.storage),
                shape: new_shape,
                strides,
                offset: self.inner.offset,
                grad: Mutex::new(None),
                grad_fn: Mutex::new(None),
                requires_grad: AtomicBool::new(false),
                is_leaf: AtomicBool::new(true),
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
            }),
            saved_version: None,
        })
    }

    /// Create a zero-copy view with a grad_fn attached. Used for shape ops
    /// (squeeze, unsqueeze, reshape, etc.) that don't change data layout.
    /// Shares the underlying storage with the source tensor.
    ///
    /// Non-contiguous tensors are materialized first (requires a copy).
    pub fn view_operation(
        &self,
        new_shape: Vec<usize>,
        grad_fn: Arc<dyn GradFn<T>>,
    ) -> FerrotorchResult<Self> {
        // Non-contiguous tensors must be materialized first — a view over
        // non-contiguous storage with new strides would read wrong elements.
        // Use the device-aware `contiguous()` path so a non-contiguous CUDA
        // tensor is gathered on-device via `strided_copy_*` rather than
        // demoted to CPU storage — the same #750 fix `view_reshape` carries.
        // The previous `data_vec() + TensorStorage::cpu` path silently moved
        // GPU tensors to host on every grad-tracking reshape / flatten /
        // squeeze / unsqueeze after a stride-view op (CORE-011, #1705).
        // The caller-supplied `grad_fn` references the ORIGINAL input, so
        // discarding the materialized intermediate from the graph is sound:
        // the copy is the identity on values.
        if !self.is_contiguous() {
            return self.contiguous()?.view_operation(new_shape, grad_fn);
        }

        let new_numel = checked_numel(&new_shape, "view_operation")?;
        if new_numel != self.numel() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "view_operation: new shape {:?} ({} elements) vs {:?} ({} elements)",
                    new_shape,
                    new_numel,
                    self.shape(),
                    self.numel()
                ),
            });
        }
        let strides = checked_c_contiguous_strides(&new_shape, "view_operation")?;
        Ok(Self {
            inner: Arc::new(TensorInner {
                id: TensorId::next(),
                storage: Arc::clone(&self.inner.storage),
                shape: new_shape,
                strides,
                offset: self.inner.offset,
                grad: Mutex::new(None),
                grad_fn: Mutex::new(Some(grad_fn)),
                requires_grad: AtomicBool::new(true),
                is_leaf: AtomicBool::new(false),
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
            }),
            saved_version: None,
        })
    }

    /// Create a zero-copy view with explicit shape, strides, and offset.
    ///
    /// This is the lowest-level view constructor — used by permute, transpose,
    /// narrow, and other operations that change the logical layout without
    /// copying data.
    ///
    /// Panics if the metadata is invalid for this tensor's storage. Prefer
    /// [`Self::try_stride_view`] or [`Self::as_strided`] when forwarding
    /// user-controlled shape/stride arguments so errors can be propagated.
    #[track_caller]
    pub fn stride_view(
        &self,
        new_shape: Vec<usize>,
        new_strides: Vec<isize>,
        new_offset: usize,
    ) -> Self {
        self.try_stride_view(new_shape, new_strides, new_offset)
            .expect("Tensor::stride_view invalid view metadata")
    }

    /// Fallible form of [`Self::stride_view`].
    pub fn try_stride_view(
        &self,
        new_shape: Vec<usize>,
        new_strides: Vec<isize>,
        new_offset: usize,
    ) -> FerrotorchResult<Self> {
        crate::stride_tricks::validate_bounds(
            "stride_view",
            &new_shape,
            &new_strides,
            new_offset,
            self.storage_len(),
        )?;
        Ok(self.stride_view_unchecked(new_shape, new_strides, new_offset, None))
    }

    fn stride_view_unchecked(
        &self,
        new_shape: Vec<usize>,
        new_strides: Vec<isize>,
        new_offset: usize,
        grad_fn: Option<Arc<dyn GradFn<T>>>,
    ) -> Self {
        let requires_grad = grad_fn.is_some();
        let is_leaf = grad_fn.is_none();
        Self {
            inner: Arc::new(TensorInner {
                id: TensorId::next(),
                storage: Arc::clone(&self.inner.storage),
                shape: new_shape,
                strides: new_strides,
                offset: new_offset,
                grad: Mutex::new(None),
                grad_fn: Mutex::new(grad_fn),
                requires_grad: AtomicBool::new(requires_grad),
                is_leaf: AtomicBool::new(is_leaf),
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
            }),
            saved_version: None,
        }
    }

    /// Create a zero-copy view with explicit shape, strides, and offset,
    /// with an attached gradient function for autograd.
    ///
    /// Panics if the metadata is invalid for this tensor's storage. Prefer
    /// [`Self::try_stride_view_operation`] in fallible operation code.
    #[track_caller]
    pub fn stride_view_operation(
        &self,
        new_shape: Vec<usize>,
        new_strides: Vec<isize>,
        new_offset: usize,
        grad_fn: Arc<dyn GradFn<T>>,
    ) -> Self {
        self.try_stride_view_operation(new_shape, new_strides, new_offset, grad_fn)
            .expect("Tensor::stride_view_operation invalid view metadata")
    }

    /// Fallible form of [`Self::stride_view_operation`].
    pub fn try_stride_view_operation(
        &self,
        new_shape: Vec<usize>,
        new_strides: Vec<isize>,
        new_offset: usize,
        grad_fn: Arc<dyn GradFn<T>>,
    ) -> FerrotorchResult<Self> {
        crate::stride_tricks::validate_bounds(
            "stride_view_operation",
            &new_shape,
            &new_strides,
            new_offset,
            self.storage_len(),
        )?;
        Ok(self.stride_view_unchecked(new_shape, new_strides, new_offset, Some(grad_fn)))
    }

    /// Create a tensor that is the result of an operation (non-leaf).
    ///
    /// The resulting tensor has `requires_grad = true`, `is_leaf = false`,
    /// and the given `grad_fn` attached for reverse-mode autodiff.
    pub fn from_operation(
        storage: TensorStorage<T>,
        shape: Vec<usize>,
        grad_fn: Arc<dyn GradFn<T>>,
    ) -> FerrotorchResult<Self> {
        // In inference mode, skip all autograd bookkeeping — create a plain
        // tensor without grad_fn. This avoids allocating autograd metadata
        // and makes operations faster for pure inference.
        if crate::autograd::no_grad::is_inference_mode() {
            return Self::from_storage(storage, shape, false);
        }

        let numel = checked_numel(&shape, "Tensor::from_operation")?;

        if numel > storage.len() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "shape {:?} requires {} elements but storage has {}",
                    shape,
                    numel,
                    storage.len()
                ),
            });
        }

        let strides = checked_c_contiguous_strides(&shape, "Tensor::from_operation")?;

        Ok(Self {
            inner: Arc::new(TensorInner {
                id: TensorId::next(),
                storage: Arc::new(storage),
                shape,
                strides,
                offset: 0,
                grad: Mutex::new(None),
                grad_fn: Mutex::new(Some(grad_fn)),
                requires_grad: AtomicBool::new(true),
                is_leaf: AtomicBool::new(false),
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
            }),
            saved_version: None,
        })
    }

    /// Create a non-leaf operation result while also handing the backward node
    /// a saved output tensor that shares the result's storage and version
    /// counter. This mirrors PyTorch formulas such as `sqrt_backward(grad,
    /// result)`: mutating `result.detach()` after forward must invalidate the
    /// saved result observed during backward, without making the grad_fn hold a
    /// strong reference cycle to the output tensor itself.
    pub(crate) fn from_operation_saving_output<F>(
        storage: TensorStorage<T>,
        shape: Vec<usize>,
        make_grad_fn: F,
    ) -> FerrotorchResult<Self>
    where
        F: FnOnce(Tensor<T>) -> Arc<dyn GradFn<T>>,
    {
        if crate::autograd::no_grad::is_inference_mode() {
            return Self::from_storage(storage, shape, false);
        }

        let numel = checked_numel(&shape, "Tensor::from_operation_saving_output")?;
        if numel > storage.len() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "shape {:?} requires {} elements but storage has {}",
                    shape,
                    numel,
                    storage.len()
                ),
            });
        }

        let strides = checked_c_contiguous_strides(&shape, "Tensor::from_operation_saving_output")?;
        let storage = Arc::new(storage);
        let saved_output = Tensor {
            inner: Arc::new(TensorInner {
                id: TensorId::next(),
                storage: Arc::clone(&storage),
                shape: shape.clone(),
                strides: strides.clone(),
                offset: 0,
                grad: Mutex::new(None),
                grad_fn: Mutex::new(None),
                requires_grad: AtomicBool::new(false),
                is_leaf: AtomicBool::new(true),
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
            }),
            saved_version: Some(storage.version()),
        };
        let grad_fn = make_grad_fn(saved_output);

        Ok(Self {
            inner: Arc::new(TensorInner {
                id: TensorId::next(),
                storage,
                shape,
                strides,
                offset: 0,
                grad: Mutex::new(None),
                grad_fn: Mutex::new(Some(grad_fn)),
                requires_grad: AtomicBool::new(true),
                is_leaf: AtomicBool::new(false),
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
            }),
            saved_version: None,
        })
    }
}

// --- ToDeviceBackward ---

/// Backward for `Tensor::to(device)`.
///
/// Copies the gradient back to the source tensor's device so that
/// gradients flow through device transfers.
#[derive(Debug)]
struct ToDeviceBackward<T: Float> {
    source: Tensor<T>,
}

impl<T: Float> GradFn<T> for ToDeviceBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let target_device = self.source.device();
        if grad_output.device() == target_device {
            Ok(vec![Some(grad_output.clone())])
        } else {
            Ok(vec![Some(grad_output.to(target_device)?)])
        }
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.source]
    }

    fn name(&self) -> &'static str {
        "ToDeviceBackward"
    }
}

// --- CastBackward ---

/// Backward for `Tensor::to_dtype`.
///
/// The core autograd engine is monomorphized by tensor dtype, so a normal
/// `GradFn<U>` cannot list a `Tensor<T>` source in `inputs()` when `T != U`.
/// This bridge keeps the public PyTorch contract anyway: it casts the upstream
/// gradient back to the source dtype, runs the source's hooks, then either
/// accumulates on the source leaf or recursively drives the source graph.
#[derive(Debug)]
struct CastBackward<Src: Float, Dst: Float> {
    source: Tensor<Src>,
    _dst: std::marker::PhantomData<Dst>,
}

impl<Src: Float, Dst: Float> CastBackward<Src, Dst> {
    fn propagate_source_grad(&self, grad: Tensor<Src>) -> FerrotorchResult<()> {
        let hooks = self.source.hooks();
        let has_hooks = {
            let guard = hooks.lock().map_err(|e| FerrotorchError::LockPoisoned {
                message: format!("hook storage mutex: {e}"),
            })?;
            (guard.has_grad_hooks(), guard.has_post_accumulate_hooks())
        };
        let grad = if has_hooks.0 {
            crate::autograd::hooks::run_grad_hooks(hooks, grad)?
        } else {
            grad
        };

        if self.source.is_leaf() {
            self.source.accumulate_grad(&grad)?;
            if has_hooks.1 {
                crate::autograd::hooks::run_post_accumulate_hooks(hooks, &self.source)?;
            }
        } else {
            crate::autograd::graph::backward_with_grad(&self.source, Some(&grad))?;
        }
        Ok(())
    }
}

impl<Src: Float, Dst: Float> GradFn<Dst> for CastBackward<Src, Dst> {
    fn backward(&self, grad_output: &Tensor<Dst>) -> FerrotorchResult<Vec<Option<Tensor<Dst>>>> {
        let source_grad = grad_output.to_dtype_backward::<Src>()?;
        self.propagate_source_grad(source_grad)?;
        Ok(Vec::new())
    }

    fn inputs(&self) -> Vec<&Tensor<Dst>> {
        Vec::new()
    }

    fn name(&self) -> &'static str {
        "CastBackward"
    }
}

// --- MemoryFormatBackward ---

/// Backward for the physical materialization in
/// [`Tensor::to_memory_format`] / [`Tensor::contiguous_in`].
///
/// A memory-format change permutes the PHYSICAL order only; logical values
/// are untouched, so its gradient is the identity (torch records
/// `ToCopyBackward0` / `CloneBackward0` on these paths and the source's
/// grad equals the output's grad — probed live, CORE-013 / #1707).
#[derive(Debug)]
struct MemoryFormatBackward<T: Float> {
    source: Tensor<T>,
}

impl<T: Float> GradFn<T> for MemoryFormatBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if self.source.requires_grad() {
            Ok(vec![Some(grad_output.clone())])
        } else {
            Ok(vec![None])
        }
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.source]
    }

    fn name(&self) -> &'static str {
        "MemoryFormatBackward"
    }
}

// --- Accessors ---

impl<T: Float> Tensor<T> {
    #[inline]
    pub fn id(&self) -> TensorId {
        self.inner.id
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.inner.shape
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.inner.shape.len()
    }

    #[inline]
    pub fn numel(&self) -> usize {
        checked_numel(&self.inner.shape, "Tensor::numel")
            .expect("Tensor::numel: stored shape element count overflows usize")
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.inner.strides
    }

    /// Offset (in number of elements) into the underlying storage.
    ///
    /// Non-zero for views created by narrow, select, or other subregion ops.
    #[inline]
    pub fn storage_offset(&self) -> usize {
        self.inner.offset
    }

    /// Number of elements in the underlying storage buffer.
    ///
    /// May be larger than [`numel()`](Self::numel) for views (transpose,
    /// narrow, as_strided, etc.) that address only a subset of the
    /// storage. Used by stride-manipulation ops (`as_strided`,
    /// `as_strided_copy`) for bounds validation.
    #[inline]
    pub fn storage_len(&self) -> usize {
        self.inner.storage.len()
    }

    /// Borrow the underlying [`TensorStorage`]. Used by ops that need
    /// access to the GPU buffer handle or to share storage Arc-wise.
    #[inline]
    pub fn storage(&self) -> &TensorStorage<T> {
        &self.inner.storage
    }

    /// Current shared in-place version for this tensor's storage.
    #[inline]
    pub(crate) fn storage_version(&self) -> u64 {
        self.inner.storage.version()
    }

    /// Clone this tensor as a value saved for backward, recording the shared
    /// storage version PyTorch's SavedVariable would expect during backward.
    #[inline]
    pub(crate) fn saved_for_backward(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            saved_version: Some(self.storage_version()),
        }
    }

    #[inline]
    fn bump_storage_version(&self) {
        self.inner.storage.bump_version();
    }

    #[inline]
    fn check_saved_version(&self, access: &'static str) -> FerrotorchResult<()> {
        if let Some(expected) = self.saved_version {
            let current = self.storage_version();
            if current != expected {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "one of the variables needed for gradient computation has been modified \
                         by an inplace operation: tensor is at version {current}; expected \
                         version {expected} instead ({access})"
                    ),
                });
            }
        }
        Ok(())
    }

    #[inline]
    pub fn device(&self) -> Device {
        self.inner.storage.device()
    }

    #[inline]
    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad.load(Ordering::Acquire)
    }

    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.inner.is_leaf.load(Ordering::Acquire)
    }

    #[inline]
    pub fn grad_fn(&self) -> Option<Arc<dyn GradFn<T>>> {
        let guard = self
            .inner
            .grad_fn
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.as_ref().cloned()
    }

    pub(crate) fn autograd_snapshot(&self) -> FerrotorchResult<Tensor<T>> {
        let (storage, shape) = self.clone().into_storage_and_shape()?;
        let strides = checked_c_contiguous_strides(&shape, "autograd_snapshot")?;
        Ok(Tensor {
            inner: Arc::new(TensorInner {
                id: TensorId::next(),
                storage: Arc::new(storage),
                shape,
                strides,
                offset: 0,
                grad: Mutex::new(None),
                grad_fn: Mutex::new(self.grad_fn()),
                requires_grad: AtomicBool::new(self.requires_grad()),
                is_leaf: AtomicBool::new(self.is_leaf()),
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
            }),
            saved_version: None,
        })
    }

    pub(crate) fn replace_autograd_metadata(
        &self,
        requires_grad: bool,
        is_leaf: bool,
        grad_fn: Option<Arc<dyn GradFn<T>>>,
    ) -> FerrotorchResult<()> {
        {
            let mut guard =
                self.inner
                    .grad_fn
                    .lock()
                    .map_err(|e| FerrotorchError::LockPoisoned {
                        message: format!("grad_fn mutex: {e}"),
                    })?;
            *guard = grad_fn;
        }
        self.inner
            .requires_grad
            .store(requires_grad, Ordering::Release);
        self.inner.is_leaf.store(is_leaf, Ordering::Release);
        if !is_leaf {
            self.set_grad(None)?;
        }
        Ok(())
    }

    /// Access the hook storage for this tensor.
    pub(crate) fn hooks(&self) -> &Mutex<crate::autograd::hooks::HookStorage<T>> {
        &self.inner.hooks
    }

    /// Register a gradient hook on this tensor.
    ///
    /// The hook is called during backward whenever a gradient is computed for
    /// this tensor. It receives the gradient and may return `Some(new_grad)` to
    /// replace it, or `None` to keep the original.
    ///
    /// Returns a [`HookHandle`](crate::autograd::hooks::HookHandle) that can
    /// be used to remove the hook later via [`remove_hook`](Self::remove_hook).
    pub fn register_hook<F>(&self, func: F) -> FerrotorchResult<crate::autograd::hooks::HookHandle>
    where
        F: Fn(&Tensor<T>) -> Option<Tensor<T>> + Send + Sync + 'static,
    {
        let mut guard = self
            .inner
            .hooks
            .lock()
            .map_err(|e| FerrotorchError::LockPoisoned {
                message: format!("hook storage mutex: {e}"),
            })?;
        Ok(guard.add_grad_hook(func))
    }

    /// Register a post-accumulate-grad hook on this tensor.
    ///
    /// The hook is called after gradient accumulation completes on a leaf
    /// tensor. It receives a reference to the tensor itself (so the hook can
    /// read `.grad()`). Cannot modify the gradient — use
    /// [`register_hook`](Self::register_hook) for that.
    pub fn register_post_accumulate_grad_hook<F>(
        &self,
        func: F,
    ) -> FerrotorchResult<crate::autograd::hooks::HookHandle>
    where
        F: Fn(&Tensor<T>) + Send + Sync + 'static,
    {
        if !self.is_leaf() {
            return Err(FerrotorchError::InvalidArgument {
                message: "post accumulate grad hooks cannot be registered on non-leaf tensors"
                    .into(),
            });
        }
        let mut guard = self
            .inner
            .hooks
            .lock()
            .map_err(|e| FerrotorchError::LockPoisoned {
                message: format!("hook storage mutex: {e}"),
            })?;
        Ok(guard.add_post_accumulate_hook(func))
    }

    /// Remove a previously registered hook by its handle.
    ///
    /// Returns `true` if the hook was found and removed.
    pub fn remove_hook(
        &self,
        handle: crate::autograd::hooks::HookHandle,
    ) -> FerrotorchResult<bool> {
        let mut guard = self
            .inner
            .hooks
            .lock()
            .map_err(|e| FerrotorchError::LockPoisoned {
                message: format!("hook storage mutex: {e}"),
            })?;
        Ok(guard.remove(handle))
    }

    /// Read the accumulated gradient. Returns `None` if no gradient has
    /// been computed yet.
    pub fn grad(&self) -> FerrotorchResult<Option<Tensor<T>>> {
        let guard = self
            .inner
            .grad
            .lock()
            .map_err(|e| FerrotorchError::LockPoisoned {
                message: format!("grad mutex: {e}"),
            })?;
        Ok(guard.as_ref().map(|b| (**b).clone()))
    }

    /// Set or replace the accumulated gradient.
    pub fn set_grad(&self, grad: Option<Tensor<T>>) -> FerrotorchResult<()> {
        let mut guard = self
            .inner
            .grad
            .lock()
            .map_err(|e| FerrotorchError::LockPoisoned {
                message: format!("grad mutex: {e}"),
            })?;
        *guard = grad.map(Box::new);
        Ok(())
    }

    /// Zero out the gradient of this tensor.
    ///
    /// Equivalent to `self.set_grad(None)`. Typically called before each
    /// training iteration to prevent gradient accumulation across steps.
    pub fn zero_grad(&self) -> FerrotorchResult<()> {
        self.set_grad(None)
    }

    /// Accumulate a gradient additively (used by the backward engine).
    ///
    /// Keeps gradients on their original device to avoid GPU↔CPU round-trips.
    /// When both the existing gradient and the incoming gradient are on the
    /// same CUDA device, accumulation uses the dtype-specific GPU add kernel
    /// entirely on-device (`f32`, `f64`, `f16`, or `bf16`).
    pub(crate) fn accumulate_grad(&self, incoming: &Tensor<T>) -> FerrotorchResult<()> {
        let mut guard = self
            .inner
            .grad
            .lock()
            .map_err(|e| FerrotorchError::LockPoisoned {
                message: format!("grad mutex: {e}"),
            })?;
        match guard.as_mut() {
            None => {
                // First gradient: store a detached copy on the same device.
                let (storage, shape) = incoming.clone().into_storage_and_shape()?;
                let tensor = Tensor::from_storage(storage, shape, false)?;
                *guard = Some(Box::new(tensor));
            }
            Some(existing) => {
                // Accumulate: existing_grad += incoming_grad.
                // GPU-native path: both grads on the same CUDA device. Dispatch
                // by dtype tag, not Rust element size: f16 and bf16 are both
                // two-byte floats with distinct CUDA handle tags.
                if existing.is_cuda()
                    && incoming.is_cuda()
                    && existing.device() == incoming.device()
                {
                    let backend = crate::gpu_dispatch::gpu_backend()
                        .ok_or(FerrotorchError::DeviceUnavailable)?;
                    if existing.numel() != incoming.numel() {
                        return Err(FerrotorchError::ShapeMismatch {
                            message: format!(
                                "gradient accumulation shape mismatch: {:?} vs {:?}",
                                existing.shape(),
                                incoming.shape()
                            ),
                        });
                    }
                    let a_handle = existing.gpu_handle()?;
                    let b_handle = incoming.gpu_handle()?;
                    let sum_handle: crate::gpu_dispatch::GpuBufferHandle = crate::dispatch_floating_dtype!(
                        T,
                        "gradient accumulation add",
                        f32 => backend.add_f32(a_handle, b_handle),
                        f64 => backend.add_f64(a_handle, b_handle),
                        bf16 => backend.add_bf16_bf16(a_handle, b_handle),
                        f16 => backend.add_f16(a_handle, b_handle),
                    )?;
                    let storage = TensorStorage::gpu(sum_handle);
                    let combined = Tensor::from_storage(storage, existing.shape().to_vec(), false)?;
                    *guard = Some(Box::new(combined));
                } else {
                    // CPU path (or mixed-device): download if needed and
                    // accumulate on the host.
                    let incoming_data = incoming.data_vec()?;
                    let mut buf = existing.data_vec()?;
                    if buf.len() != incoming_data.len() {
                        return Err(FerrotorchError::ShapeMismatch {
                            message: format!(
                                "gradient accumulation shape mismatch: {:?} vs {:?}",
                                existing.shape(),
                                incoming.shape()
                            ),
                        });
                    }
                    for (e, &n) in buf.iter_mut().zip(incoming_data.iter()) {
                        *e += n;
                    }
                    // Store on the parameter's device.
                    let device = existing.device();
                    let combined = Tensor::from_storage(
                        TensorStorage::on_device(buf, device)?,
                        existing.shape().to_vec(),
                        false,
                    )?;
                    *guard = Some(Box::new(combined));
                }
            }
        }
        Ok(())
    }

    /// Borrow the underlying data as a flat slice.
    ///
    /// Returns `Err(GpuTensorNotAccessible)` if the tensor is on a GPU.
    /// Call `.cpu()` first to transfer it.
    ///
    /// Returns `Err` if the tensor is not contiguous — the raw storage
    /// slice would not correspond to the logical element order. Use
    /// [`data_vec()`](Self::data_vec) or call `.contiguous()` first.
    ///
    /// # Borrow contract (CORE-001 / #1695)
    ///
    /// Clones and views share storage (PyTorch parity), so in-place ops on
    /// *any* aliasing handle write the same buffer this slice points into.
    /// The returned `&[T]` must not be **used** after an in-place mutation
    /// (`fill_`, `add_`, `update_data`, ..., or a storage-swapping op such
    /// as the broadcast `add_scaled_` path) performed through this tensor
    /// or any clone/view — re-call `data()` after the mutation instead.
    /// Sequenced read-after-write access is always fine. Cross-thread use
    /// additionally requires external synchronization. See the
    /// synchronization contract on [`crate::storage::TensorStorage`].
    pub fn data(&self) -> FerrotorchResult<&[T]> {
        self.check_saved_version("Tensor::data")?;
        if self.inner.storage.is_gpu() {
            return Err(FerrotorchError::GpuTensorNotAccessible);
        }
        if self.inner.storage.is_cubecl() {
            return Err(FerrotorchError::GpuTensorNotAccessible);
        }
        if self.inner.storage.is_meta() {
            return Err(FerrotorchError::InvalidArgument {
                message: "cannot read data from a meta tensor; meta tensors carry shape only. \
                     Call .to(Device::Cpu) to materialize, or use .shape() / .numel() / .device() \
                     for metadata access."
                    .into(),
            });
        }
        if !self.is_contiguous() {
            return Err(FerrotorchError::InvalidArgument {
                message: "tensor is not contiguous; call .contiguous() or use .data_vec()".into(),
            });
        }
        let slice = self.inner.storage.try_as_slice()?;
        if self.numel() == 0 {
            return Ok(&slice[0..0]);
        }
        let end = self.inner.offset.checked_add(self.numel()).ok_or_else(|| {
            FerrotorchError::InvalidArgument {
                message: "tensor view offset + numel overflows usize".into(),
            }
        })?;
        if end > slice.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: "tensor view extends beyond storage".into(),
            });
        }
        Ok(&slice[self.inner.offset..end])
    }

    /// Borrow the underlying data as a flat slice (CPU-only alias for `data()`).
    ///
    /// Identical to [`data()`](Self::data) — returns a zero-copy `&[T]` reference
    /// to the tensor's storage. Returns `Err(GpuTensorNotAccessible)` if the
    /// tensor lives on a GPU; call `.cpu()` first to transfer.
    ///
    /// This alias exists for call-site clarity: use `data_ref()` when you want
    /// to emphasise that no copy is made, vs `data_vec()` which always copies.
    #[inline]
    pub fn data_ref(&self) -> FerrotorchResult<&[T]> {
        self.data()
    }

    /// Get tensor data as an owned `Vec<T>`, transparently transferring from
    /// GPU if needed and correctly handling non-contiguous tensors.
    ///
    /// For contiguous CPU tensors this copies the slice. For non-contiguous
    /// CPU tensors it gathers elements in logical (C-order) sequence. For
    /// GPU tensors it performs a device-to-host transfer.
    pub fn data_vec(&self) -> FerrotorchResult<Vec<T>> {
        self.check_saved_version("Tensor::data_vec")?;
        if self.inner.storage.is_meta() {
            return Err(FerrotorchError::InvalidArgument {
                message: "cannot read data from a meta tensor; meta tensors carry shape only. \
                     Call .to(Device::Cpu) to materialize, or use .shape() / .numel() / .device() \
                     for metadata access."
                    .into(),
            });
        }
        if self.is_cuda() || self.inner.storage.is_cubecl() {
            let cpu_tensor = self.cpu()?;
            Ok(cpu_tensor.data()?.to_vec())
        } else if self.is_contiguous() {
            Ok(self.data()?.to_vec())
        } else {
            // Non-contiguous: gather elements by walking strides.
            // is_cuda/is_cubecl branches above already routed GPU storage,
            // so try_as_slice on this CPU/Meta arm only Errs on Meta — which
            // the data_vec entry guard at the top of the function rejects.
            let slice = self.inner.storage.try_as_slice()?;
            let shape = &self.inner.shape;
            let strides = &self.inner.strides;
            let offset = self.inner.offset;
            let numel = self.numel();
            let ndim = shape.len();
            crate::stride_tricks::validate_bounds("data_vec", shape, strides, offset, slice.len())?;

            let mut result = Vec::with_capacity(numel);
            let mut indices = vec![0usize; ndim];
            for _ in 0..numel {
                let mut flat = offset as isize;
                for d in 0..ndim {
                    flat += indices[d] as isize * strides[d];
                }
                result.push(slice[flat as usize]);
                // Increment multi-index (rightmost first).
                for d in (0..ndim).rev() {
                    indices[d] += 1;
                    if indices[d] < shape[d] {
                        break;
                    }
                    indices[d] = 0;
                }
            }
            Ok(result)
        }
    }

    /// Consume this tensor and return its storage and shape.
    ///
    /// If this is the only reference to the underlying data, the storage Vec
    /// is extracted without copying. Otherwise falls back to cloning.
    /// Used internally to avoid double-copies when rewrapping op results.
    pub fn into_storage_and_shape(self) -> FerrotorchResult<(TensorStorage<T>, Vec<usize>)> {
        // Non-contiguous tensors must be materialized — the raw storage
        // does not match the logical element order.
        if !self.is_contiguous() {
            let data = self.data_vec()?;
            let shape = self.shape().to_vec();
            let device = self.device();
            return Ok((TensorStorage::on_device(data, device)?, shape));
        }

        let shape = self.inner.shape.clone();
        let offset = self.inner.offset;
        let numel = checked_numel(&shape, "into_storage_and_shape")?;

        // Try to unwrap the inner Arc to get ownership of TensorInner.
        match Arc::try_unwrap(self.inner) {
            Ok(inner) => {
                // We own the inner. Try to unwrap the storage Arc.
                match Arc::try_unwrap(inner.storage) {
                    Ok(storage) if offset == 0 && storage.len() == numel => {
                        // Fast path: sole owner, no offset — zero-copy return.
                        Ok((storage, shape))
                    }
                    Ok(storage) => {
                        // Sole owner but offset or extra elements — extract
                        // the subregion. For CPU we can slice the owned Vec
                        // directly; for GPU we round-trip through the host.
                        let sub = storage.try_clone_subregion(offset, numel)?;
                        Ok((sub, shape))
                    }
                    Err(arc_storage) => {
                        // Storage is shared — clone the relevant subregion.
                        let sub = arc_storage.try_clone_subregion(offset, numel)?;
                        Ok((sub, shape))
                    }
                }
            }
            Err(arc_inner) => {
                // Inner is shared — clone the relevant subregion.
                let sub = arc_inner
                    .storage
                    .try_clone_subregion(arc_inner.offset, numel)?;
                Ok((sub, shape))
            }
        }
    }

    /// Move this tensor to a device, returning a new tensor.
    ///
    /// If the tensor is already on the target device, returns a cheap clone
    /// (shared Arc storage).
    pub fn to(&self, device: Device) -> FerrotorchResult<Tensor<T>> {
        if self.device() == device {
            return Ok(self.clone());
        }

        // CORE-012 (#1706): torch treats a differentiable `.to(device)` as a
        // copy WITH a backward edge even when the source is a LEAF — the
        // transferred tensor is a non-leaf (`is_leaf=False`, grad_fn
        // `ToCopyBackward0`) and backward reaches the ORIGINAL source leaf.
        // The previous `&& !self.is_leaf()` silently severed the graph for
        // leaf transfers. When tracking is off (no_grad, or the source does
        // not require grad), the output is an untracked fresh leaf with
        // `requires_grad = false` (torch: `.to()` under `no_grad` yields
        // requires_grad=False) — never a bare copied flag, per R-LOUD-3.
        let needs_grad_fn = self.requires_grad() && crate::autograd::no_grad::is_grad_enabled();

        match (self.device(), device) {
            (Device::Cpu, Device::Cuda(ordinal)) => {
                // Non-contiguous tensors must be materialized before GPU upload.
                let contiguous_self = if self.is_contiguous() {
                    self.clone()
                } else {
                    crate::methods::contiguous_t(self)?
                };
                let backend =
                    crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                let cpu_data = contiguous_self.data()?;
                // SAFETY: `cpu_data` is a borrow into the contiguous CPU
                // storage of contiguous_self, guaranteed to live for the
                // duration of this scope. Reinterpreting as `&[u8]` is sound
                // because every value of T is fully initialized (it's a
                // numeric Float type with no padding bytes); the byte length
                // is checked before building the raw slice.
                let byte_len =
                    checked_byte_count(cpu_data.len(), std::mem::size_of::<T>(), "Tensor::to")?;
                let bytes =
                    unsafe { std::slice::from_raw_parts(cpu_data.as_ptr().cast::<u8>(), byte_len) };
                let handle = backend.cpu_to_gpu(bytes, T::dtype(), ordinal)?;
                let storage = TensorStorage::gpu(handle);
                if needs_grad_fn {
                    let grad_fn = Arc::new(ToDeviceBackward {
                        source: self.clone(),
                    });
                    Tensor::from_operation(storage, self.shape().to_vec(), grad_fn)
                } else {
                    Tensor::from_storage(storage, self.shape().to_vec(), false)
                }
            }
            (Device::Cuda(_), Device::Cpu) => {
                use std::any::TypeId;

                let backend =
                    crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                let handle = self.gpu_handle()?;
                let numel = self.numel();

                // Issue #802 — When the source GPU tensor is a stride-view
                // (narrow / select / transpose / permute / chained views),
                // its storage Arc is shared with the parent and so the
                // raw `gpu_to_cpu(handle)` D2H readback would copy the
                // *entire* underlying buffer. The receiving CPU tensor is
                // then constructed via `from_storage`, which resets
                // `storage_offset` to 0 and recomputes C-contiguous strides
                // for the view's shape — silently dropping the original
                // `storage_offset()` and `strides()`. The result: `.data()`
                // returns the first `numel` elements of the full buffer
                // rather than the actual view's elements.
                //
                // PyTorch parity (rust-gpu-discipline §3): `tensor.cpu()`
                // must materialize the view. We do this by gathering the
                // view on-device via the existing `strided_copy_{f32,f64,u16}`
                // kernel (rank ≤ 8, positive strides, float dtypes) into a
                // fresh contiguous device buffer, *then* D2H. This mirrors
                // the GPU fast path already used by `methods::contiguous_t`
                // for CUDA tensors — there is no host-side gather.
                //
                // The fast path is only taken when the view is *already*
                // a full contiguous representation of the underlying
                // buffer: contiguous, zero offset, and the buffer length
                // matches numel. In every other case we materialize.
                let needs_materialize =
                    !self.is_contiguous() || self.storage_offset() != 0 || handle.len() != numel;

                let data = if needs_materialize {
                    if self.shape().len() > 8 {
                        let bytes = backend.gpu_to_cpu(handle)?;
                        let storage_data = decode_device_bytes::<T>(
                            "Tensor::to(Cpu): high-rank CUDA storage readback",
                            &bytes,
                        )?;
                        gather_logical_from_slice(
                            "Tensor::to(Cpu): high-rank CUDA view materialization",
                            &storage_data,
                            self.shape(),
                            self.strides(),
                            self.storage_offset(),
                        )?
                    } else {
                        let view_shape = self.shape().to_vec();
                        let src_strides = self.strides().to_vec();
                        let src_offset = self.storage_offset();
                        let materialized = if TypeId::of::<T>() == TypeId::of::<f32>() {
                            backend.strided_copy_f32(
                                handle,
                                &view_shape,
                                &src_strides,
                                src_offset,
                            )?
                        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                            backend.strided_copy_f64(
                                handle,
                                &view_shape,
                                &src_strides,
                                src_offset,
                            )?
                        } else if TypeId::of::<T>() == TypeId::of::<half::f16>()
                            || TypeId::of::<T>() == TypeId::of::<half::bf16>()
                        {
                            backend.strided_copy_u16(
                                handle,
                                &view_shape,
                                &src_strides,
                                src_offset,
                            )?
                        } else {
                            return Err(FerrotorchError::NotImplementedOnCuda {
                                op: "Tensor::to(Cpu): materialize CUDA view for dtype",
                            });
                        };
                        let bytes = backend.gpu_to_cpu(&materialized)?;
                        decode_device_bytes("Tensor::to(Cpu): CUDA view materialization", &bytes)?
                    }
                } else {
                    let bytes = backend.gpu_to_cpu(handle)?;
                    decode_device_bytes("Tensor::to(Cpu): CUDA readback", &bytes)?
                };
                let storage = TensorStorage::cpu(data);
                if needs_grad_fn {
                    let grad_fn = Arc::new(ToDeviceBackward {
                        source: self.clone(),
                    });
                    Tensor::from_operation(storage, self.shape().to_vec(), grad_fn)
                } else {
                    Tensor::from_storage(storage, self.shape().to_vec(), false)
                }
            }
            (Device::Cuda(a), Device::Cuda(b)) if a != b => {
                // Cross-GPU: go through CPU for now
                let cpu = self.to(Device::Cpu)?;
                cpu.to(Device::Cuda(b))
            }
            // CPU → XPU: requires a CubeRuntime owned by ferrotorch-xpu.
            // Core cannot perform the H2D upload here. Use
            // `ferrotorch_xpu::XpuDevice::upload(tensor)` or
            // `tensor.to(Device::Xpu(n))` via the xpu crate's integration.
            // Issue #673.
            (Device::Cpu, Device::Xpu(_)) => Err(FerrotorchError::InvalidArgument {
                message: "CPU→XPU transfer requires a CubeRuntime. \
                              Use ferrotorch_xpu::make_xpu_tensor or \
                              ferrotorch_xpu::XpuDevice::upload instead. Issue #673."
                    .into(),
            }),
            // XPU → CPU: real D2H readback via the CubeStorageHandle.
            // This is the explicit transfer path — mirrors PyTorch's
            // `.cpu()` which requires a synchronous D2H copy. Issue #673.
            (Device::Xpu(_), Device::Cpu) => {
                let handle = self.inner.storage.cubecl_handle().ok_or_else(|| {
                    FerrotorchError::InvalidArgument {
                        message: "XPU→CPU transfer: storage does not contain a CubeCL handle. \
                                  This tensor may have been created before issue #673 was applied."
                            .into(),
                    }
                })?;
                // f32 readback only for now — T=f32 is the only supported XPU dtype.
                let host_f32 = handle.read_to_host()?;
                // SAFETY: `T` on XPU is always `f32` (the only dtype the cubecl
                // kernels support). `host_f32` contains exactly `handle.len()`
                // f32 values. We transmute via bytecast rather than assuming
                // T=f32 at the type level, so this is gated at runtime.
                let data: Vec<T> = {
                    if std::mem::size_of::<T>() != std::mem::size_of::<f32>() {
                        return Err(FerrotorchError::InvalidArgument {
                            message: format!(
                                "XPU→CPU: expected f32 storage (size 4), got size {}; \
                                 only f32 XPU tensors are supported. Issue #673.",
                                std::mem::size_of::<T>()
                            ),
                        });
                    }
                    // SAFETY: T and f32 have the same size (checked above).
                    // Both are plain-data numeric types with no padding.
                    // The bytes came from a cubecl kernel that wrote f32 values.
                    unsafe {
                        let mut md = std::mem::ManuallyDrop::new(host_f32);
                        Vec::from_raw_parts(md.as_mut_ptr().cast::<T>(), md.len(), md.capacity())
                    }
                };
                let storage = TensorStorage::cpu(data);
                if needs_grad_fn {
                    let grad_fn = Arc::new(ToDeviceBackward {
                        source: self.clone(),
                    });
                    Tensor::from_operation(storage, self.shape().to_vec(), grad_fn)
                } else {
                    Tensor::from_storage(storage, self.shape().to_vec(), false)
                }
            }
            // XPU → XPU on a different ordinal: route through CPU for
            // now. CL-452.
            (Device::Xpu(a), Device::Xpu(b)) if a != b => {
                let cpu = self.to(Device::Cpu)?;
                cpu.to(Device::Xpu(b))
            }
            // CUDA ↔ XPU: round-trip via CPU. CL-452.
            (Device::Cuda(_), Device::Xpu(_)) | (Device::Xpu(_), Device::Cuda(_)) => {
                let cpu = self.to(Device::Cpu)?;
                cpu.to(device)
            }
            // Move TO the meta device: drop the data, keep shape only.
            // Works from any source device.
            //
            // Autograd contract (CORE-012, #1706): torch attaches
            // `ToCopyBackward0` to `.to('meta')` like any other transfer;
            // driving backward through it then raises `NotImplementedError:
            // Cannot copy out of meta tensor; no data!` (probed live on
            // torch 2.11.0+cu130). ferrotorch mirrors this: the edge is
            // attached here, and backward fails with the structured
            // `InvalidArgument("cannot move a meta tensor to …")` when
            // `ToDeviceBackward` tries to copy the meta gradient out to the
            // source device — a loud error per R-LOUD-1, never a silently
            // severed graph.
            (_, Device::Meta) => {
                let storage = TensorStorage::meta(self.numel());
                if needs_grad_fn {
                    let grad_fn = Arc::new(ToDeviceBackward {
                        source: self.clone(),
                    });
                    Tensor::from_operation(storage, self.shape().to_vec(), grad_fn)
                } else {
                    Tensor::from_storage(storage, self.shape().to_vec(), false)
                }
            }
            // Move FROM the meta device: cannot materialize random data,
            // so this errors. Users should construct fresh tensors with
            // creation::zeros / randn / etc. on the target device instead.
            (Device::Meta, _) => Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "cannot move a meta tensor to {device} -- meta tensors carry no data. \
                     Construct a real tensor on {device} via creation::zeros/randn/etc."
                ),
            }),
            _ => Ok(self.clone()),
        }
    }

    /// Like [`to`](Self::to), but uses pinned (page-locked) host memory for
    /// the CPU→CUDA transfer when applicable.
    ///
    /// On CPU→CUDA, allocates a temporary pinned host buffer, copies the
    /// tensor data into it, and uses DMA to transfer to the device. This is
    /// roughly 2x faster than the regular `to()` path for large buffers
    /// because it avoids one extra page-locked staging copy inside the CUDA
    /// driver. For small buffers (< ~64KB) the pinning overhead may
    /// outweigh the gain — measure before defaulting to this path.
    ///
    /// Behaves identically to [`to`](Self::to) for CPU→CPU, CUDA→CPU, and
    /// cross-GPU paths (which all bypass pinned memory).
    ///
    /// Used by `ferrotorch_data::DataLoader` when `pin_memory(true)` is set
    /// alongside a target device.
    pub fn to_pinned(&self, device: Device) -> FerrotorchResult<Tensor<T>> {
        if self.device() == device {
            return Ok(self.clone());
        }

        // Only the CPU→CUDA case benefits from pinned memory; for everything
        // else fall through to the regular `to()` path.
        match (self.device(), device) {
            (Device::Cpu, Device::Cuda(_)) => {
                // Same leaf-inclusive tracking rule as `to` (CORE-012,
                // #1706); torch's pinned chain is differentiable too
                // (`PinMemoryBackward0` / `ToCopyBackward0`).
                let needs_grad_fn =
                    self.requires_grad() && crate::autograd::no_grad::is_grad_enabled();

                // Materialize non-contiguous tensors before upload.
                let contiguous_self = if self.is_contiguous() {
                    self.clone()
                } else {
                    crate::methods::contiguous_t(self)?
                };
                let cpu_data = contiguous_self.data()?;
                // on_device_pinned takes ownership of the Vec, so we copy.
                let owned: Vec<T> = cpu_data.to_vec();
                let storage = TensorStorage::on_device_pinned(owned, device)?;
                if needs_grad_fn {
                    let grad_fn = Arc::new(ToDeviceBackward {
                        source: self.clone(),
                    });
                    Tensor::from_operation(storage, self.shape().to_vec(), grad_fn)
                } else {
                    Tensor::from_storage(storage, self.shape().to_vec(), false)
                }
            }
            _ => self.to(device),
        }
    }

    /// Move to CUDA device 0.
    pub fn cuda(&self) -> FerrotorchResult<Tensor<T>> {
        self.to(Device::Cuda(0))
    }

    /// Move to CPU.
    pub fn cpu(&self) -> FerrotorchResult<Tensor<T>> {
        self.to(Device::Cpu)
    }

    fn to_dtype_backward<U: Float>(&self) -> FerrotorchResult<Tensor<U>> {
        use std::any::TypeId;

        if TypeId::of::<T>() == TypeId::of::<U>() {
            let cloned = self.clone();
            // SAFETY: `T == U` at this monomorphisation, so the tensor layout is
            // identical. This path is used only for detached gradient values.
            return Ok(unsafe {
                let md = std::mem::ManuallyDrop::new(cloned);
                std::mem::transmute_copy::<Tensor<T>, Tensor<U>>(&md)
            });
        }

        match self.device() {
            Device::Cpu => {
                let materialised = if self.is_contiguous() {
                    self.clone()
                } else {
                    crate::methods::contiguous_t(self)?
                };
                let src = materialised.data()?;
                let mut out: Vec<U> = Vec::with_capacity(src.len());
                for (i, &v) in src.iter().enumerate() {
                    out.push(<U as num_traits::NumCast>::from(v).ok_or_else(|| {
                        FerrotorchError::InvalidArgument {
                            message: format!(
                                "Tensor::to_dtype backward: gradient element {i} = {v:?} not castable to {}",
                                U::dtype()
                            ),
                        }
                    })?);
                }
                Tensor::<U>::from_storage(TensorStorage::cpu(out), self.shape().to_vec(), false)
            }
            Device::Cuda(_) => {
                let materialised = if self.is_contiguous() {
                    self.clone()
                } else {
                    crate::methods::contiguous_t(self)?
                };
                let backend =
                    crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                let new_handle = backend.cast_f_to_f(materialised.gpu_handle()?, U::dtype())?;
                Tensor::<U>::from_storage(
                    TensorStorage::gpu(new_handle),
                    self.shape().to_vec(),
                    false,
                )
            }
            _ => Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Tensor::to_dtype backward: unsupported source device {:?}",
                    self.device()
                ),
            }),
        }
    }

    /// Cast this tensor to a different float dtype, preserving device + shape.
    ///
    /// `U: Float` — any of `f32` / `f64` / `bf16` / `f16`. PyTorch parity:
    /// `tensor.to(dtype)` / `tensor.to(torch.float32)`.
    ///
    /// - **Same dtype (`T == U`)**: zero-copy `Arc`-shared clone.
    /// - **CPU**: per-element cast with Rust numeric conversion semantics for
    ///   the target float. Narrowing overflow becomes `±∞`, matching PyTorch.
    /// - **GPU**: dispatched through [`crate::gpu_dispatch::GpuBackend::cast_f_to_f`];
    ///   stays GPU-resident and covers every float pair in
    ///   `{f32, f64, f16, bf16}`.
    ///
    /// # Autograd
    ///
    /// When gradient tracking is enabled and the source requires gradients,
    /// the returned tensor is a non-leaf carrying `CastBackward`. Backward
    /// casts the upstream gradient back to the source dtype, matching PyTorch's
    /// `ToCopyBackward0` behavior for floating dtype conversions.
    pub fn to_dtype<U: Float>(&self) -> FerrotorchResult<Tensor<U>> {
        use std::any::TypeId;

        // Same-dtype: zero-copy clone (PyTorch parity for `tensor.to(same_dtype)`).
        if TypeId::of::<T>() == TypeId::of::<U>() {
            let cloned = self.clone();
            // SAFETY: `TypeId::of::<T>() == TypeId::of::<U>()` means T and U are
            // the same concrete type at this monomorphisation. `Tensor<X>` is
            // `Arc<TensorInner<X>>`; with X identical the byte layout matches
            // exactly. `ManuallyDrop` blocks the source's destructor since we
            // moved ownership into the transmuted value.
            return Ok(unsafe {
                let md = std::mem::ManuallyDrop::new(cloned);
                std::mem::transmute_copy::<Tensor<T>, Tensor<U>>(&md)
            });
        }

        let needs_grad_fn = self.requires_grad() && crate::autograd::no_grad::is_grad_enabled();
        let shape = self.shape().to_vec();
        let storage = match self.device() {
            Device::Cpu => {
                // Non-contiguous source: materialise so `.data()` is logical order.
                let materialised = if self.is_contiguous() {
                    self.clone()
                } else {
                    crate::methods::contiguous_t(self)?
                };
                let src = materialised.data()?;
                let mut out: Vec<U> = Vec::with_capacity(src.len());
                for (i, &v) in src.iter().enumerate() {
                    out.push(<U as num_traits::NumCast>::from(v).ok_or_else(|| {
                        FerrotorchError::InvalidArgument {
                            message: format!(
                                "Tensor::to_dtype: element {i} = {v:?} not castable to {}",
                                U::dtype()
                            ),
                        }
                    })?);
                }
                TensorStorage::cpu(out)
            }
            Device::Cuda(_) => {
                // Non-contiguous GPU view: materialise via the existing
                // CPU/GPU `contiguous_t` fast path so the cast sees a fresh
                // contiguous buffer matching the logical numel.
                let materialised = if self.is_contiguous() {
                    self.clone()
                } else {
                    crate::methods::contiguous_t(self)?
                };
                let backend =
                    crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                let src_handle = materialised.gpu_handle()?;
                let new_handle = backend.cast_f_to_f(src_handle, U::dtype())?;
                TensorStorage::gpu(new_handle)
            }
            _ => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "Tensor::to_dtype: unsupported source device {:?}",
                        self.device()
                    ),
                });
            }
        };

        if needs_grad_fn {
            let grad_fn = Arc::new(CastBackward::<T, U> {
                source: self.clone(),
                _dst: std::marker::PhantomData,
            });
            Tensor::<U>::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::<U>::from_storage(storage, shape, false)
        }
    }

    /// Returns `true` if this tensor is on CPU.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        self.device().is_cpu()
    }

    /// Returns `true` if this tensor is on the meta device (no backing data).
    #[inline]
    pub fn is_meta(&self) -> bool {
        self.device().is_meta()
    }

    /// Recorded fill value for a meta tensor, if it was constructed with one
    /// (e.g. via [`crate::creation::full_meta`]). Returns `None` for any
    /// non-meta tensor and for meta tensors created without a fill (e.g.
    /// via [`crate::creation::zeros_meta`] / [`crate::creation::ones_meta`]
    /// / [`crate::creation::meta_like`]).
    ///
    /// Meta tensors carry no element-wise data, so the per-element fill
    /// cannot be read back — this is metadata only — but it lets callers
    /// distinguish a `full_meta(shape, 2.5)` tensor from a `full_meta(shape,
    /// 0.0)` tensor (or from a plain `zeros_meta(shape)`), which closes the
    /// "`_value` is silently ignored" gap.
    #[inline]
    pub fn meta_fill_value(&self) -> Option<&T> {
        self.inner.storage.meta_fill_value()
    }

    /// Returns `true` if this tensor is on a CUDA GPU.
    #[inline]
    pub fn is_cuda(&self) -> bool {
        self.device().is_cuda()
    }

    /// Returns `true` if this tensor is on an XPU (CubeCL / Intel GPU) device.
    #[inline]
    pub fn is_xpu(&self) -> bool {
        matches!(self.device(), crate::device::Device::Xpu(_))
    }

    /// Get the GPU buffer handle. Returns `Err` for CPU tensors.
    pub fn gpu_handle(&self) -> FerrotorchResult<&crate::gpu_dispatch::GpuBufferHandle> {
        self.check_saved_version("Tensor::gpu_handle")?;
        self.inner
            .storage
            .gpu_handle()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "tensor is on CPU, not GPU".into(),
            })
    }

    /// `masked_fill(mask, value)` — `out[i] = mask[i] ? value : self[i]`,
    /// returning a new tensor of the same shape (mask convention "true → fill",
    /// matching `torch.Tensor.masked_fill`). `mask` must have the same numel as
    /// `self` and live on the same device.
    ///
    /// When both `self` and `mask` are CUDA-resident, the fill runs on the GPU
    /// (real PTX kernel dispatched on `self`'s dtype) and the result stays
    /// GPU-resident — NO host crossing (crosslink #1185 Phase 3c). Otherwise it
    /// takes the CPU path. Carries a `MaskedFillBackward` grad_fn when grad is
    /// required.
    #[inline]
    pub fn masked_fill(
        &self,
        mask: &crate::bool_tensor::BoolTensor,
        value: T,
    ) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::indexing::masked_fill_bt(self, mask, value)
    }

    /// `masked_select(mask)` — return a 1-D tensor of the elements of `self`
    /// where `mask` is true, in flat C-order (`torch.Tensor.masked_select`).
    ///
    /// On CUDA (self + mask resident, same device) this runs a GPU stream
    /// compaction; the result stays GPU-resident. The single output-length
    /// integer crosses to the host to size the data-dependent output (the result
    /// shape, not a data round-trip — PyTorch parity).
    #[inline]
    pub fn masked_select(
        &self,
        mask: &crate::bool_tensor::BoolTensor,
    ) -> FerrotorchResult<Tensor<T>> {
        crate::ops::indexing::masked_select(self, mask)
    }

    /// Borrow the underlying data as a mutable flat slice.
    ///
    /// The `&mut [T]` is derived through the storage's element-level
    /// interior mutability ([`TensorStorage::try_as_mut_slice_aliased`]),
    /// never from the shared `Arc` itself, so calling this does not
    /// invalidate metadata borrows (`shape()`, `storage()`) held through
    /// aliasing `Tensor` handles. See the synchronization contract on
    /// [`crate::storage::TensorStorage`] (CORE-001 / #1695).
    ///
    /// # Safety
    ///
    /// The caller must ensure genuinely exclusive access to this tensor's
    /// *buffer* for the lifetime of the returned slice:
    ///
    /// 1. no other thread reads or writes this tensor's storage while the
    ///    slice is live, and
    /// 2. no other borrow of the same buffer's elements — a `&[T]` from
    ///    [`Self::data`] on this tensor or any clone/view sharing the
    ///    storage — is *used* while the slice is live.
    ///
    /// Optimizer `step()` methods satisfy this requirement: they run inside
    /// `no_grad()` (no graph is being built) and hold `&mut self` (exclusive
    /// access to the optimizer's parameter copies).
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn data_mut(&self) -> FerrotorchResult<&mut [T]> {
        if !self.is_contiguous() {
            return Err(FerrotorchError::InvalidArgument {
                message: "data_mut requires a contiguous tensor".into(),
            });
        }
        // Returns Err(GpuTensorNotAccessible) for GPU/Cubecl/Meta storage —
        // the deprecated `as_mut_slice` panicked here. Optimizer step
        // implementations now get a clean error path for misuse against
        // GPU-resident parameters.
        //
        // SAFETY: forwarded verbatim — this function's own `# Safety`
        // section restates `try_as_mut_slice_aliased`'s exclusivity
        // contract for the caller.
        let slice = unsafe { self.inner.storage.try_as_mut_slice_aliased()? };
        if self.numel() == 0 {
            return Ok(&mut slice[0..0]);
        }
        let end = self.inner.offset.checked_add(self.numel()).ok_or_else(|| {
            FerrotorchError::InvalidArgument {
                message: "tensor view offset + numel overflows usize".into(),
            }
        })?;
        if end > slice.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: "tensor view extends beyond storage".into(),
            });
        }
        self.bump_storage_version();
        Ok(&mut slice[self.inner.offset..end])
    }

    /// Write `new_data` into this tensor's storage, preserving tensor identity.
    ///
    /// `new_data` holds the tensor's elements in C-contiguous order of
    /// `self.shape()`.
    ///
    /// - **CPU**: copies data into the existing storage Vec — a single run
    ///   at the storage offset for contiguous tensors, a strided scatter
    ///   for non-contiguous views (#1938).
    /// - **GPU**: uploads data to the device, then applies the
    ///   [`update_storage`] view rule (#1938): whole-storage tensors swap
    ///   the buffer; sub-views scatter into the existing device buffer.
    ///
    /// This is the device-transparent alternative to `data_mut()` for
    /// optimizer step implementations.
    ///
    /// All branches mutate through the storage's interior mutability
    /// ([`TensorStorage::cpu_write_at`] / the element-level scatter for CPU
    /// writes, [`TensorStorage::replace_buffer_aliased`] for the GPU
    /// whole-buffer swap), never through a `&mut` manufactured behind the
    /// shared `Arc` — so metadata borrows held by aliasing handles survive
    /// the call. See the synchronization contract on
    /// [`crate::storage::TensorStorage`] (CORE-001 / #1695).
    ///
    /// # Safety
    ///
    /// Same requirements as `data_mut()` — caller must ensure exclusive
    /// access: no other thread touches this tensor's storage during the
    /// call, and no outstanding `&[T]` borrow of this storage's data (from
    /// this tensor or any clone/view) is *used* after the call. Optimizer
    /// `step()` methods satisfy this by running inside `no_grad()` with
    /// `&mut self`.
    pub unsafe fn update_data(&self, new_data: &[T]) -> FerrotorchResult<()> {
        let numel = self.numel();
        if new_data.len() != numel {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "update_data: new data has {} elements but tensor has {}",
                    new_data.len(),
                    numel,
                ),
            });
        }

        let storage = &*self.inner.storage;

        if storage.is_gpu() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let ordinal = match storage.device() {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };
            // SAFETY: `new_data: &[T]` is borrowed for the duration of this
            // function and is fully initialized (it's a slice of T values
            // with no padding bytes — T is a numeric Float type). Reading
            // its bytes via reinterpretation is sound; the requested length
            // `size_of_val(new_data) = new_data.len() * size_of::<T>()`
            // matches the actual byte size of the underlying allocation.
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    new_data.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(new_data),
                )
            };
            let new_handle = backend.cpu_to_gpu(bytes, T::dtype(), ordinal)?;
            // Route through `update_storage`, which enforces the #1938 view
            // rule: a whole-storage tensor swaps the buffer (the previous
            // behavior); a sub-view scatters into the EXISTING device
            // buffer via `strided_scatter_*` instead of replacing the
            // shared storage behind aliasing views' backs.
            //
            // SAFETY: caller guarantees exclusive access per this function's
            // `# Safety` contract (forwarded verbatim); the fresh handle is
            // on `ordinal`, i.e. the same device the storage already
            // resides on, so the device check inside `update_storage`
            // passes.
            unsafe {
                self.update_storage(TensorStorage::gpu(new_handle))?;
            }
            return Ok(());
        } else if self.is_contiguous() {
            // is_gpu() branch above already routed Gpu storage; cpu_write_at
            // here Errs only on Cubecl/Meta — neither of which an update_data
            // caller in the optimizer hot path is expected to hit, so the
            // explicit Err is a strict improvement over the deprecated panic.
            //
            // SAFETY: caller guarantees exclusive access per this function's
            // `# Safety` contract; bounds are checked inside `cpu_write_at`.
            unsafe {
                storage.cpu_write_at(self.inner.offset, new_data)?;
            }
        } else {
            // Non-contiguous CPU view: scatter `new_data` (C-order of
            // `self.shape()`) through the strides. Pre-#1938 this wrote a
            // contiguous run at the storage offset, landing elements in
            // the wrong slots and clobbering interleaved base elements.
            self.check_view_write("update_data")?;
            // SAFETY: caller guarantees exclusive access per this
            // function's `# Safety` contract; `check_view_write` above
            // validated bounds and rejected overlapping views.
            unsafe {
                self.cpu_scatter_through_view(new_data)?;
            }
        }

        self.bump_storage_version();
        Ok(())
    }

    /// Replace this tensor's storage AND shape/strides in-place, matching
    /// PyTorch's `Tensor.resize_(new_shape)` + storage swap.
    ///
    /// This is the rare case where both the underlying buffer and the
    /// shape metadata in `TensorInner` need to change in lockstep — used
    /// by the `out=` write path of `torch.add(a, b, *, out=out)` when
    /// `out.shape() != broadcast_shape` (PyTorch silently resizes `out`,
    /// with a deprecation warning, in current versions). The new strides
    /// are computed as C-contiguous for `new_shape`.
    ///
    /// # Errors
    ///
    /// In addition to the shape/device errors below, returns
    /// [`FerrotorchError::InvalidArgument`] when this tensor is **aliased**
    /// (another `Tensor` clone shares its `TensorInner`, or another
    /// tensor/view shares its storage `Arc`). A resize rewrites the
    /// shape/strides metadata inside the shared `TensorInner` and replaces
    /// the buffer with one of a *different length* — neither is soundly
    /// expressible while another handle may be observing them, so the
    /// aliased case is rejected instead of mutating behind the alias
    /// (CORE-001 / #1695).
    ///
    /// # Safety
    ///
    /// Same as [`update_storage`]: caller must ensure exclusive access —
    /// no other thread touches this tensor during the call, and no
    /// outstanding borrow of its data (`data()`), shape (`shape()`,
    /// `strides()`), or storage (`storage()`) is *used* after the call.
    /// The `out=`-style call sites this method exists for own a unique
    /// `&Tensor` for the duration of the write.
    pub unsafe fn update_storage_and_shape(
        &self,
        new_storage: TensorStorage<T>,
        new_shape: Vec<usize>,
    ) -> FerrotorchResult<()> {
        let new_numel = checked_numel(&new_shape, "update_storage_and_shape")?;
        if new_storage.len() != new_numel {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "update_storage_and_shape: new storage has {} elements but \
                     new shape {:?} requires {}",
                    new_storage.len(),
                    new_shape,
                    new_numel,
                ),
            });
        }
        if new_storage.device() != self.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: self.device(),
                got: new_storage.device(),
            });
        }

        // Alias gate: rewriting shape/strides inside the Arc-shared
        // `TensorInner` — and swapping in a buffer of a different length —
        // is only sound when this handle is the unique owner of both Arcs.
        // A `&TensorStorage` or shape borrow held through another clone or
        // view would otherwise be invalidated behind that handle's back.
        if Arc::strong_count(&self.inner) > 1
            || Arc::weak_count(&self.inner) > 0
            || Arc::strong_count(&self.inner.storage) > 1
        {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "update_storage_and_shape: cannot resize tensor {:?} -> {:?}: \
                     the tensor is aliased (other clones or views share its \
                     metadata or storage); resize requires a uniquely-owned \
                     tensor (CORE-001 / #1695)",
                    self.shape(),
                    new_shape,
                ),
            });
        }

        let new_strides = checked_c_contiguous_strides(&new_shape, "update_storage_and_shape")?;

        // Swap the buffer through the storage's interior mutability. The
        // numel differs from self.numel() by design (this is the resize
        // path), so this bypasses update_storage's length check; the
        // metadata is rewritten in lockstep below.
        //
        // SAFETY: caller guarantees exclusive access per this function's
        // `# Safety` contract; the alias gate above proved both Arcs are
        // uniquely owned by this handle. Device equality was checked above,
        // satisfying `replace_buffer_aliased`'s device invariant.
        unsafe {
            self.inner.storage.replace_buffer_aliased(new_storage)?;
        }

        // Now mutate the shape/strides/offset fields through the
        // Arc<TensorInner>. The alias gate above proved this Arc is
        // uniquely owned (strong == 1, weak == 0), so no other handle can
        // reach these fields.
        let inner_ptr = Arc::as_ptr(&self.inner).cast_mut();
        // SAFETY: the Arc is uniquely owned (checked above) and the caller
        // guarantees no outstanding borrows of shape/strides are used after
        // this call (this function's `# Safety` contract). Field
        // assignments below are individual `Vec<_>` writes that drop the
        // old vectors in place — no leak.
        unsafe {
            (*inner_ptr).shape = new_shape;
            (*inner_ptr).strides = new_strides;
            (*inner_ptr).offset = 0;
        }

        self.bump_storage_version();
        Ok(())
    }

    /// Write `new_storage`'s elements into this tensor, in-place.
    ///
    /// `new_storage` must hold exactly `self.numel()` elements, laid out in
    /// C-contiguous order of `self.shape()`, on `self`'s device. Used by
    /// GPU-native optimizer steps (whole-parameter update without a CPU
    /// round-trip) and by the `out=` / trailing-underscore in-place op
    /// family to land a freshly-computed result in the target tensor.
    ///
    /// **The view rule (#1938 / CORE-001 residual).** Which write strategy
    /// runs depends on whether this tensor covers its entire storage:
    ///
    /// - **Whole-storage swap** — only when `storage_offset == 0`,
    ///   `numel == storage.len()` AND the tensor is C-contiguous. The
    ///   buffer is swapped through
    ///   [`TensorStorage::replace_buffer_aliased`] (the optimizer `step()`
    ///   fast path: zero copies). The swap goes through the storage's
    ///   buffer-level interior mutability, never through a `&mut`
    ///   manufactured behind the shared `Arc` — so `&TensorStorage`
    ///   metadata borrows held by aliasing handles survive the swap, and
    ///   aliasing clones/views observe the new buffer on their next
    ///   (sequenced) read. `&[T]` borrows into the *old* buffer dangle
    ///   after the swap; using them is the documented residual-contract
    ///   violation (see [`crate::storage::TensorStorage`], CORE-001 /
    ///   #1695).
    /// - **Region write** — when this tensor is a sub-view (offset, fewer
    ///   elements than the storage) or a non-contiguous view, the elements
    ///   are written INTO the existing buffer at this tensor's
    ///   `storage_offset`, honoring its strides, so every other view of
    ///   the same storage keeps its elements. This matches PyTorch's
    ///   matched-shape `out=` semantics
    ///   (`aten/src/ATen/native/Resize.cpp:27`: equal sizes ⇒ no
    ///   resize/swap; the TensorIterator writes elementwise into `out`'s
    ///   storage). Pre-#1938 this case incorrectly swapped the whole
    ///   shared buffer, shrinking it to the view's numel and destroying
    ///   the base tensor's other elements. On CPU the write goes through
    ///   the CORE-001 aliased-write primitives
    ///   ([`TensorStorage::cpu_write_at`] for contiguous views, a strided
    ///   scatter over [`TensorStorage::try_as_mut_slice_aliased`]
    ///   otherwise); on CUDA through the `strided_scatter_{f32,f64,u16}`
    ///   kernels into the existing device buffer (no handle swap — aliased
    ///   views keep pointing at live memory). The u16 path preserves f16/bf16
    ///   bit patterns and dtype tags.
    ///
    /// # Errors
    ///
    /// - [`FerrotorchError::ShapeMismatch`] when the element counts differ.
    /// - [`FerrotorchError::DeviceMismatch`] when `new_storage` resides on
    ///   a different device.
    /// - [`FerrotorchError::InvalidArgument`] when the region write targets
    ///   an internally-overlapping view (a dim of size > 1 with stride 0 —
    ///   torch: "more than one element of the written-to tensor refers to
    ///   a single memory location") or a view whose extent escapes the
    ///   storage bounds.
    /// - [`FerrotorchError::NotImplementedOnCuda`] for CUDA region writes
    ///   with a dtype outside the backend's floating strided-scatter set.
    ///   Returned BEFORE any mutation.
    /// - [`FerrotorchError::GpuTensorNotAccessible`] for region writes on
    ///   CubeCL/meta storage (no in-place region primitive).
    ///
    /// # Safety
    ///
    /// Same as [`update_data`]: caller must ensure exclusive access — no
    /// other thread touches this tensor's storage during the call, and no
    /// outstanding borrow of the storage's data (from this tensor or any
    /// clone/view) is *used* after the call.
    pub unsafe fn update_storage(&self, new_storage: TensorStorage<T>) -> FerrotorchResult<()> {
        let numel = self.numel();
        if new_storage.len() != numel {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "update_storage: new storage has {} elements but tensor has {}",
                    new_storage.len(),
                    numel,
                ),
            });
        }
        if new_storage.device() != self.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: self.device(),
                got: new_storage.device(),
            });
        }

        let storage = &*self.inner.storage;

        // Whole-storage fast path (the only case where a buffer swap is
        // observationally equivalent to an elementwise write): this tensor
        // covers the entire buffer as one C-contiguous run.
        if self.inner.offset == 0 && storage.len() == numel && self.is_contiguous() {
            // SAFETY: caller guarantees exclusive access (optimizer step
            // inside no_grad) per this function's `# Safety` contract.
            // `replace_buffer_aliased` drops the OLD buffer (returning CPU
            // buffers to the pool) so nothing leaks — in the GPU case a
            // leaked storage would leak its `GpuBufferHandle` ->
            // `CudaBuffer` -> pooled `CudaSlice`, climbing toward VRAM
            // exhaustion across optimizer steps (e.g. AdamW foreach calls
            // this once per parameter per step).
            unsafe {
                storage.replace_buffer_aliased(new_storage)?;
            }
            self.bump_storage_version();
            return Ok(());
        }

        // Region write: this tensor is a sub-view / non-contiguous view of
        // a (possibly shared) storage — write its region in place (#1938).
        self.check_view_write("update_storage")?;

        if storage.is_gpu() {
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let src = new_storage
                .gpu_handle()
                .ok_or(FerrotorchError::DeviceUnavailable)?;
            // SAFETY: the `&mut GpuBufferHandle` is derived through the
            // storage's buffer-level `UnsafeCell` (never from the `Arc`);
            // the caller's exclusive-access contract guarantees no other
            // reference to this storage is used while it is live, and it
            // does not outlive this call.
            let dst = unsafe { storage.gpu_handle_mut_aliased() }
                .ok_or(FerrotorchError::DeviceUnavailable)?;
            match <T as crate::dtype::Element>::dtype() {
                crate::dtype::DType::F32 => backend.strided_scatter_f32(
                    src,
                    dst,
                    &self.inner.shape,
                    &self.inner.strides,
                    self.inner.offset,
                )?,
                crate::dtype::DType::F64 => backend.strided_scatter_f64(
                    src,
                    dst,
                    &self.inner.shape,
                    &self.inner.strides,
                    self.inner.offset,
                )?,
                crate::dtype::DType::F16 | crate::dtype::DType::BF16 => backend
                    .strided_scatter_u16(
                        src,
                        dst,
                        &self.inner.shape,
                        &self.inner.strides,
                        self.inner.offset,
                    )?,
                other => {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "update_storage: CUDA sub-view/strided write unsupported dtype {other}"
                        ),
                    });
                }
            }
            self.bump_storage_version();
            return Ok(());
        }

        // CPU storage (CubeCL/meta fall out of the primitives below with a
        // structured GpuTensorNotAccessible — never a silent wrong write).
        let src = new_storage.try_as_slice()?;
        if self.is_contiguous() {
            // SAFETY: caller's exclusive-access contract is forwarded
            // verbatim; bounds are checked inside `cpu_write_at`.
            unsafe {
                storage.cpu_write_at(self.inner.offset, src)?;
            }
        } else {
            // SAFETY: caller's exclusive-access contract is forwarded
            // verbatim; `check_view_write` above validated bounds and
            // rejected overlapping views.
            unsafe {
                self.cpu_scatter_through_view(src)?;
            }
        }
        self.bump_storage_version();
        Ok(())
    }

    /// Validate that this tensor's view geometry can receive an in-place
    /// region write (#1938): rejects internally-overlapping views (a dim
    /// of size > 1 with stride 0 — torch raises "unsupported operation:
    /// more than one element of the written-to tensor refers to a single
    /// memory location") and views whose reachable extent escapes the
    /// storage bounds.
    fn check_view_write(&self, op: &'static str) -> FerrotorchResult<()> {
        if self
            .inner
            .shape
            .iter()
            .zip(self.inner.strides.iter())
            .any(|(&d, &s)| d > 1 && s == 0)
        {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "{op}: cannot write through an internally-overlapping view \
                     (shape {:?}, strides {:?}): more than one element of the \
                     written-to tensor refers to a single memory location",
                    self.inner.shape, self.inner.strides,
                ),
            });
        }
        crate::stride_tricks::validate_bounds(
            op,
            &self.inner.shape,
            &self.inner.strides,
            self.inner.offset,
            self.inner.storage.len(),
        )
    }

    /// Scatter `src` (exactly `self.numel()` elements, C-order of
    /// `self.shape()`) into this tensor's CPU storage region, honoring
    /// `storage_offset` and strides (#1938). Odometer walk: the rightmost
    /// index advances fastest; the running offset is updated incrementally
    /// so each element write is O(1) amortized.
    ///
    /// # Safety
    ///
    /// Same exclusive-access contract as
    /// [`TensorStorage::try_as_mut_slice_aliased`]: no other thread touches
    /// this storage during the call, and no outstanding borrow of its
    /// elements (from this or any aliasing handle) is *used* afterwards.
    /// The caller must have validated the view geometry via
    /// [`Self::check_view_write`] first — it proves every offset the walk
    /// reaches is in `[0, storage.len())` and that no two logical elements
    /// alias one slot.
    unsafe fn cpu_scatter_through_view(&self, src: &[T]) -> FerrotorchResult<()> {
        debug_assert_eq!(src.len(), self.numel());
        // SAFETY: exclusivity is forwarded verbatim from this function's
        // own `# Safety` contract.
        let dst = unsafe { self.inner.storage.try_as_mut_slice_aliased()? };
        let shape = &self.inner.shape;
        let strides = &self.inner.strides;
        let ndim = shape.len();
        let mut idx = vec![0usize; ndim];
        let mut off = self.inner.offset as isize;
        for &v in src {
            // In-bounds per the caller's `check_view_write` obligation
            // (validate_bounds covered the full reachable extent).
            dst[off as usize] = v;
            for d in (0..ndim).rev() {
                idx[d] += 1;
                off += strides[d];
                if idx[d] < shape[d] {
                    break;
                }
                off -= shape[d] as isize * strides[d];
                idx[d] = 0;
            }
        }
        Ok(())
    }

    /// Run `f` with mutable access to this tensor's underlying
    /// [`GpuBufferHandle`], in-place.
    ///
    /// This is a safe wrapper for the optimizer fast-path that fuses the
    /// parameter update directly into a GPU kernel: the kernel needs an
    /// `&mut GpuBufferHandle` aliased into the param tensor's storage, but
    /// `Tensor` is `Arc`-shared so a naïve `&self -> &mut Storage` route
    /// requires `unsafe` at every call site. By centralizing the
    /// `Arc::as_ptr -> *mut TensorStorage<T>` cast inside this single
    /// method and returning `Err(FerrotorchError::DeviceUnavailable)` for
    /// non-GPU storage, callers do not need to write any `unsafe` of their
    /// own.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::DeviceUnavailable`] when this tensor's
    /// storage is not GPU-resident.
    ///
    /// # Safety contract (encapsulated)
    ///
    /// This method is **safe** because the caller cannot violate any
    /// invariant exposed through it: the closure receives a fresh
    /// `&mut GpuBufferHandle` whose lifetime is bounded by the body of
    /// this method, and no other reference to the storage can be created
    /// concurrently from within the closure (the closure is `FnOnce`).
    /// The only remaining hazard is concurrent access to the same `Arc`
    /// from another thread — `Tensor` is not `Sync` for storage mutation
    /// purposes, and the optimizer step that drives this method holds
    /// `&mut self` on the outer `Optimizer` for the whole step, so no
    /// other handle can be observing or mutating this storage during the
    /// call. This is the same exclusive-access guarantee that
    /// [`update_data`] and [`update_storage`] depend on; this method
    /// simply lets the optimizer mutate the GPU handle in place rather
    /// than swap the entire `TensorStorage`.
    pub fn with_gpu_handle_mut<R>(
        &self,
        f: impl FnOnce(&mut crate::gpu_dispatch::GpuBufferHandle) -> FerrotorchResult<R>,
    ) -> FerrotorchResult<R> {
        // SAFETY: the `&mut GpuBufferHandle` is derived through the
        // storage's buffer-level `UnsafeCell` (never from the `Arc`
        // itself), and its exclusivity contract holds here: the
        // optimizer-step caller holds `&mut self` on the outer optimizer
        // for the whole step, so no other handle into this storage can be
        // observed or mutated concurrently — neither this thread (the
        // closure is `FnOnce` and consumes the borrow chain) nor any other
        // thread (the storage is not exposed to a sharing primitive that
        // would let another thread reach it during the optimizer step).
        // The `&mut GpuBufferHandle` is bounded by the body of this method;
        // no reference produced here outlives the call.
        let handle = unsafe { self.inner.storage.gpu_handle_mut_aliased() }
            .ok_or(FerrotorchError::DeviceUnavailable)?;
        self.bump_storage_version();
        f(handle)
    }

    /// Detach this tensor from the computation graph, returning a new
    /// tensor that shares storage but has no grad_fn.
    pub fn detach(&self) -> Self {
        Self {
            inner: Arc::new(TensorInner {
                id: TensorId::next(),
                storage: Arc::clone(&self.inner.storage),
                shape: self.inner.shape.clone(),
                strides: self.inner.strides.clone(),
                offset: self.inner.offset,
                grad: Mutex::new(None),
                grad_fn: Mutex::new(None),
                requires_grad: AtomicBool::new(false),
                is_leaf: AtomicBool::new(true),
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
            }),
            saved_version: None,
        }
    }

    /// Return a new tensor with `requires_grad` set.
    pub fn requires_grad_(self, requires_grad: bool) -> Self {
        // Must create a new inner since Arc<TensorInner> is immutable.
        Self {
            inner: Arc::new(TensorInner {
                id: self.inner.id,
                storage: Arc::clone(&self.inner.storage),
                shape: self.inner.shape.clone(),
                strides: self.inner.strides.clone(),
                offset: self.inner.offset,
                grad: Mutex::new(None),
                grad_fn: Mutex::new(self.grad_fn()),
                requires_grad: AtomicBool::new(requires_grad),
                is_leaf: AtomicBool::new(self.is_leaf()),
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
            }),
            saved_version: None,
        }
    }

    /// Whether this tensor is contiguous in memory (C-order).
    ///
    /// Dimensions with size 1 can have any stride without affecting
    /// contiguity, since they contribute no index offset.
    pub fn is_contiguous(&self) -> bool {
        if self.inner.shape.is_empty() {
            return true;
        }
        let mut expected_stride: isize = 1;
        for d in (0..self.ndim()).rev() {
            if self.inner.shape[d] == 0 {
                return true;
            }
            if self.inner.shape[d] != 1 && self.inner.strides[d] != expected_stride {
                return false;
            }
            if self.inner.shape[d] != 1 {
                expected_stride *= self.inner.shape[d] as isize;
            }
        }
        true
    }

    /// Check whether this tensor is contiguous in a specific memory format.
    ///
    /// - `MemoryFormat::Contiguous` — standard C-order (NCHW for 4D).
    /// - `MemoryFormat::ChannelsLast` — NHWC stride pattern for 4D tensors.
    /// - `MemoryFormat::ChannelsLast3d` — NDHWC stride pattern for 5D tensors.
    ///
    /// Dimensions of size 1 are treated as matching any stride, consistent
    /// with PyTorch behaviour.
    ///
    /// [CL-309] WU-05: channels-last memory format support
    pub fn is_contiguous_for(&self, format: MemoryFormat) -> bool {
        match format {
            MemoryFormat::Contiguous => self.is_contiguous(),
            MemoryFormat::ChannelsLast => {
                if self.ndim() != 4 {
                    return false;
                }
                let Ok(expected) =
                    checked_channels_last_strides(&self.inner.shape, "Tensor::is_contiguous_for")
                else {
                    return false;
                };
                strides_match_with_size1(&self.inner.shape, &self.inner.strides, &expected)
            }
            MemoryFormat::ChannelsLast3d => {
                if self.ndim() != 5 {
                    return false;
                }
                let Ok(expected) = checked_channels_last_3d_strides(
                    &self.inner.shape,
                    "Tensor::is_contiguous_for",
                ) else {
                    return false;
                };
                strides_match_with_size1(&self.inner.shape, &self.inner.strides, &expected)
            }
        }
    }

    /// Rearrange this tensor to the target memory format.
    ///
    /// If the tensor is already contiguous in the target format, returns a
    /// cheap clone (shared storage). Otherwise, physically rearranges the
    /// data and returns a new tensor with the correct strides.
    ///
    /// The *shape* is never changed — only the strides (and possibly the
    /// underlying data order) are altered.
    ///
    /// [CL-309] WU-05: channels-last memory format support
    pub fn to_memory_format(&self, format: MemoryFormat) -> FerrotorchResult<Self> {
        if self.is_contiguous_for(format) {
            return Ok(self.clone());
        }
        self.materialize_format(format)
    }

    /// Return a tensor that is contiguous in the given memory format,
    /// materializing (copying) the data if necessary.
    ///
    /// Equivalent to `.to_memory_format(format)` — both names are provided
    /// for API familiarity: `contiguous()` is the PyTorch-style entry point
    /// while `to_memory_format()` is the explicit variant.
    ///
    /// [CL-309] WU-05: channels-last memory format support
    pub fn contiguous_in(&self, format: MemoryFormat) -> FerrotorchResult<Self> {
        self.to_memory_format(format)
    }

    /// Assemble the result tensor of a physical memory-format
    /// materialization, wiring the autograd edge (CORE-013, #1707).
    ///
    /// When gradient tracking applies (grad mode on, not inference mode,
    /// source requires grad), the output is a non-leaf carrying a
    /// [`MemoryFormatBackward`] identity edge back to `self` — torch parity
    /// for `.to(memory_format=…)` / `.contiguous(memory_format=…)`
    /// (`ToCopyBackward0` / `CloneBackward0`). Otherwise the output is an
    /// honestly untracked fresh leaf (`requires_grad = false`); the flag is
    /// never copied bare onto a disconnected tensor (R-LOUD-3).
    ///
    /// `Tensor::from_operation` cannot be used here: it derives C-contiguous
    /// strides, while format materialization needs the target format's
    /// stride pattern over an offset-0 buffer.
    fn format_materialized(&self, storage: TensorStorage<T>, target_strides: Vec<isize>) -> Self {
        let track = !crate::autograd::no_grad::is_inference_mode()
            && crate::autograd::no_grad::is_grad_enabled()
            && self.requires_grad();
        let grad_fn: Option<Arc<dyn GradFn<T>>> = if track {
            Some(Arc::new(MemoryFormatBackward {
                source: self.clone(),
            }))
        } else {
            None
        };
        Self {
            inner: Arc::new(TensorInner {
                id: TensorId::next(),
                storage: Arc::new(storage),
                shape: self.inner.shape.clone(),
                strides: target_strides,
                offset: 0,
                grad: Mutex::new(None),
                grad_fn: Mutex::new(grad_fn),
                requires_grad: AtomicBool::new(track),
                is_leaf: AtomicBool::new(!track),
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
            }),
            saved_version: None,
        }
    }

    /// Physically rearrange data into the target memory format.
    ///
    /// Called when the tensor is NOT already contiguous in `format`.
    /// Gathers elements in the physical order dictated by the target strides
    /// and writes them into a fresh, contiguous-in-format buffer.
    ///
    /// [CL-309] WU-05: channels-last memory format support
    fn materialize_format(&self, format: MemoryFormat) -> FerrotorchResult<Self> {
        let shape = &self.inner.shape;
        let ndim = shape.len();

        match format {
            MemoryFormat::ChannelsLast if ndim != 4 => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("ChannelsLast requires a 4D tensor, got {ndim}D"),
                });
            }
            MemoryFormat::ChannelsLast3d if ndim != 5 => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("ChannelsLast3d requires a 5D tensor, got {ndim}D"),
                });
            }
            _ => {}
        }

        let target_strides = match format {
            MemoryFormat::Contiguous => {
                checked_c_contiguous_strides(shape, "Tensor::materialize_format")?
            }
            MemoryFormat::ChannelsLast => {
                checked_channels_last_strides(shape, "Tensor::materialize_format")?
            }
            MemoryFormat::ChannelsLast3d => {
                checked_channels_last_3d_strides(shape, "Tensor::materialize_format")?
            }
        };

        // GPU fast path (CL-455): for non-meta CUDA tensors of rank
        // <= 8, gather the source view directly into a fresh contiguous
        // device buffer using gpu_strided_copy. The trick is that
        // gpu_strided_copy iterates its output linearly using
        // c-contiguous output strides — so we pass a *permuted* shape
        // and src_stride pair such that the linear iteration order
        // matches the target memory layout (e.g., NHWC for
        // ChannelsLast). The gathered buffer is then naturally
        // contiguous in the target format.
        if self.is_cuda()
            && ndim <= 8
            && let Some(backend) = crate::gpu_dispatch::gpu_backend()
        {
            use std::any::TypeId;
            let perm = format_permutation(format, ndim);
            let permuted_shape: Vec<usize> = perm.iter().map(|&d| shape[d]).collect();
            let permuted_src_strides: Vec<isize> =
                perm.iter().map(|&d| self.inner.strides[d]).collect();
            let in_handle = self.gpu_handle()?;
            let src_offset = self.inner.offset;

            let out_handle = if TypeId::of::<T>() == TypeId::of::<f32>() {
                backend.strided_copy_f32(
                    in_handle,
                    &permuted_shape,
                    &permuted_src_strides,
                    src_offset,
                )
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                backend.strided_copy_f64(
                    in_handle,
                    &permuted_shape,
                    &permuted_src_strides,
                    src_offset,
                )
            } else if TypeId::of::<T>() == TypeId::of::<half::f16>()
                || TypeId::of::<T>() == TypeId::of::<half::bf16>()
            {
                backend.strided_copy_u16(
                    in_handle,
                    &permuted_shape,
                    &permuted_src_strides,
                    src_offset,
                )
            } else {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "materialize_format: CUDA dtype",
                });
            };

            let handle = out_handle?;
            let storage = TensorStorage::gpu(handle);
            return Ok(self.format_materialized(storage, target_strides));
        }

        if self.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "materialize_format: CUDA rank > 8 or missing backend",
            });
        }

        self.materialize_format_cpu(format, target_strides)
    }

    /// CPU path for [`materialize_format`]. Always valid for CPU tensors.
    fn materialize_format_cpu(
        &self,
        _format: MemoryFormat,
        target_strides: Vec<isize>,
    ) -> FerrotorchResult<Self> {
        let shape = &self.inner.shape;
        let ndim = shape.len();
        let numel = self.numel();
        let src_strides = &self.inner.strides;
        let offset = self.inner.offset;

        let device = self.device();
        let src_owned: Vec<T>;
        let src_ref: &[T] = if self.is_cuda() {
            src_owned = self.data_vec()?;
            &src_owned
        } else {
            // is_cuda branch above routes Gpu storage; try_as_slice
            // here Errs only on Cubecl/Meta — both signal a misuse of
            // materialize_format_cpu and propagate cleanly.
            self.inner.storage.try_as_slice()?
        };

        let mut dst = vec![<T as num_traits::Zero>::zero(); numel];

        let mut indices = vec![0usize; ndim];
        for _ in 0..numel {
            let mut src_flat = offset as isize;
            let mut dst_flat: isize = 0;
            for d in 0..ndim {
                src_flat += indices[d] as isize * src_strides[d];
                dst_flat += indices[d] as isize * target_strides[d];
            }
            dst[dst_flat as usize] = src_ref[src_flat as usize];

            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < shape[d] {
                    break;
                }
                indices[d] = 0;
            }
        }

        let storage = TensorStorage::on_device(dst, device)?;
        Ok(self.format_materialized(storage, target_strides))
    }

    /// Returns `true` if this is a scalar (0-dimensional) tensor.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.inner.shape.is_empty()
    }

    /// For a scalar tensor, extract the single value.
    pub fn item(&self) -> FerrotorchResult<T> {
        if !self.is_scalar() && self.numel() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "item() requires a scalar or single-element tensor, got shape {:?}",
                    self.shape()
                ),
            });
        }
        let data = self.data()?;
        Ok(data[0])
    }

    /// Returns true if two tensors are the same object (same Arc).
    pub fn is_same(&self, other: &Self) -> bool {
        self.inner.id == other.inner.id
    }

    /// Number of strong references to the outer `Arc<TensorInner>`.
    ///
    /// Used by the backward engine to decide whether in-place gradient
    /// accumulation is safe (refcount == 1 means exclusive ownership).
    #[inline]
    pub(crate) fn inner_refcount(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Number of strong references to the inner `Arc<TensorStorage>`.
    ///
    /// Even when `inner_refcount() == 1`, the storage may be shared
    /// (e.g. via `view_reshape` or `detach`). Both must be 1 for
    /// in-place mutation to be safe.
    #[inline]
    pub(crate) fn storage_refcount(&self) -> usize {
        Arc::strong_count(&self.inner.storage)
    }

    /// Get a reference to the inner storage `Arc`.
    ///
    /// Exposed for optimizer kernels that need to modify the param's GPU
    /// buffer in-place via `unsafe` pointer cast (same pattern as
    /// `update_data`).
    #[inline]
    pub fn inner_storage_arc(&self) -> &Arc<TensorStorage<T>> {
        &self.inner.storage
    }

    /// Returns true if two tensors share the same underlying storage allocation.
    ///
    /// Used by tests to verify that view operations (squeeze, unsqueeze, flatten)
    /// are zero-copy.
    #[cfg(test)]
    pub(crate) fn shares_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner.storage, &other.inner.storage)
    }
}

/// Compare actual strides against expected strides, treating dimensions
/// of size 1 as wildcards (any stride is acceptable for size-1 dims).
///
/// This matches PyTorch's contiguity semantics where a size-1 dimension
/// does not constrain the stride because it only ever indexes at 0.
///
/// Permutation that maps *output position d* (in linear iteration
/// order over a buffer that's contiguous in `format`) to the
/// corresponding *input dim*. Used by [`materialize_format`] to
/// reuse [`gpu_strided_copy`](crate::gpu_dispatch::GpuBackend::strided_copy_f32)
/// for non-c-contiguous targets like channels-last.
///
/// For an `ndim`-dimensional input shape `[d0, d1, ..., d_{n-1}]`:
///
/// - `Contiguous` → identity `[0, 1, 2, ..., n-1]`. Linear iteration
///   over the c-contiguous output is the same as iterating dim 0
///   slowest.
/// - `ChannelsLast` (4D, NCHW input → NHWC output): `[0, 2, 3, 1]`.
///   Iterating output[n*HWC + h*WC + w*C + c] linearly produces the
///   coordinate sequence (n, h, w, c), so we want input dim mapping
///   {output 0 → input N=0, output 1 → input H=2, output 2 → input
///   W=3, output 3 → input C=1}.
/// - `ChannelsLast3d` (5D, NCDHW → NDHWC): `[0, 2, 3, 4, 1]`. Same
///   reasoning extended to volumetric tensors.
///
/// `ndim` is taken as a parameter so callers can pass it without
/// re-computing from the input. The function only returns
/// non-identity permutations for the formats whose rank constraints
/// are satisfied; everything else returns the identity.
///
/// CL-455.
fn format_permutation(format: MemoryFormat, ndim: usize) -> Vec<usize> {
    match format {
        MemoryFormat::ChannelsLast if ndim == 4 => vec![0, 2, 3, 1],
        MemoryFormat::ChannelsLast3d if ndim == 5 => vec![0, 2, 3, 4, 1],
        // Contiguous (or any rank-mismatched format that the caller
        // already validated): identity permutation.
        _ => (0..ndim).collect(),
    }
}

/// [CL-309] WU-05: channels-last memory format support
fn strides_match_with_size1(shape: &[usize], actual: &[isize], expected: &[isize]) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    for i in 0..shape.len() {
        if shape[i] != 1 && actual[i] != expected[i] {
            return false;
        }
    }
    true
}

// --- Trait impls ---

impl<T: Float> Clone for Tensor<T> {
    /// Clone is cheap — it increments the Arc refcount. Both copies
    /// share the same data, grad, and identity.
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            saved_version: self.saved_version,
        }
    }
}

impl<T: Float> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("id", &self.inner.id)
            .field("shape", &self.inner.shape)
            .field("device", &self.device())
            .field("requires_grad", &self.requires_grad())
            .field("is_leaf", &self.is_leaf())
            .field("grad_fn", &self.grad_fn().as_ref().map(|gf| gf.name()))
            .field("saved_version", &self.saved_version)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    #[test]
    fn test_tensor_from_storage() {
        let storage = TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = Tensor::from_storage(storage, vec![2, 3], false).unwrap();

        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.strides(), &[3, 1]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.numel(), 6);
        assert!(t.is_contiguous());
        assert!(t.is_leaf());
        assert!(!t.requires_grad());
        assert_eq!(t.device(), Device::Cpu);
    }

    #[test]
    fn test_tensor_shape_mismatch() {
        let storage = TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]);
        let result = Tensor::from_storage(storage, vec![2, 3], false);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_data_access() {
        let storage = TensorStorage::cpu(vec![1.0f64, 2.0, 3.0]);
        let t = Tensor::from_storage(storage, vec![3], false).unwrap();
        assert_eq!(t.data().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    // reason: round-trip bit-equality — the storage contains the exact bit
    // pattern of 42.0 and item() reads it without arithmetic, so equality is
    // the correct check.
    #[allow(clippy::float_cmp)]
    fn test_tensor_scalar() {
        let storage = TensorStorage::cpu(vec![42.0f32]);
        let t = Tensor::from_storage(storage, vec![], false).unwrap();
        assert!(t.is_scalar());
        assert_eq!(t.item().unwrap(), 42.0);
    }

    #[test]
    fn test_tensor_detach() {
        let storage = TensorStorage::cpu(vec![1.0f32, 2.0]);
        let t = Tensor::from_storage(storage, vec![2], true).unwrap();
        assert!(t.requires_grad());

        let d = t.detach();
        assert!(!d.requires_grad());
        assert!(d.is_leaf());
        assert!(d.grad_fn().is_none());
    }

    #[test]
    fn test_tensor_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Tensor<f32>>();
        assert_send_sync::<Tensor<f64>>();
    }

    #[test]
    fn test_clone_shares_identity() {
        let storage = TensorStorage::cpu(vec![1.0f32, 2.0]);
        let t = Tensor::from_storage(storage, vec![2], true).unwrap();
        let t2 = t.clone();

        assert!(t.is_same(&t2));
        assert_eq!(t.id(), t2.id());
    }

    #[test]
    fn test_view_operation_shares_storage() {
        use crate::grad_fns::shape::FlattenBackward;
        let storage = TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = Tensor::from_storage(storage, vec![2, 3], true).unwrap();
        let grad_fn = Arc::new(FlattenBackward::new(t.clone(), t.shape().to_vec()));
        let view = t.view_operation(vec![6], grad_fn).unwrap();
        assert!(t.shares_storage(&view), "view_operation must share storage");
        assert!(
            !t.is_same(&view),
            "view_operation creates new tensor identity"
        );
    }

    #[test]
    fn test_clone_shares_grad() {
        let storage = TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]);
        let t = Tensor::from_storage(storage, vec![3], true).unwrap();
        let t2 = t.clone();

        // Accumulate grad via one clone.
        let g =
            Tensor::from_storage(TensorStorage::cpu(vec![0.1, 0.2, 0.3]), vec![3], false).unwrap();
        t.accumulate_grad(&g).unwrap();

        // Visible from the other clone.
        let grad = t2.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!((data[0] - 0.1).abs() < 1e-7);
    }

    #[test]
    fn test_tensor_grad_accumulation() {
        let storage = TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]);
        let t = Tensor::from_storage(storage, vec![3], true).unwrap();

        assert!(t.grad().unwrap().is_none());

        let g1 =
            Tensor::from_storage(TensorStorage::cpu(vec![0.1, 0.2, 0.3]), vec![3], false).unwrap();
        t.accumulate_grad(&g1).unwrap();

        let grad = t.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!((data[0] - 0.1).abs() < 1e-7);

        let g2 =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0, 1.0, 1.0]), vec![3], false).unwrap();
        t.accumulate_grad(&g2).unwrap();

        let grad = t.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!((data[0] - 1.1).abs() < 1e-6);
        assert!((data[1] - 1.2).abs() < 1e-6);
        assert!((data[2] - 1.3).abs() < 1e-6);
    }

    // ── to_dtype (REQ-8 / issue #29) ─────────────────────────────────────────

    #[test]
    fn to_dtype_same_dtype_is_zero_copy_clone() {
        let storage = TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]);
        let t = Tensor::from_storage(storage, vec![3], false).unwrap();
        let same: Tensor<f32> = t.to_dtype::<f32>().unwrap();
        assert_eq!(same.shape(), &[3usize]);
        assert_eq!(same.data().unwrap(), &[1.0_f32, 2.0, 3.0]);
        // Same underlying Arc storage — id() compares by Arc address via TensorId.
        assert_eq!(same.id(), t.id());
    }

    #[test]
    fn to_dtype_cpu_f32_to_bf16_round_trips_bf16_representable_values() {
        let storage = TensorStorage::cpu(vec![1.0f32, -2.0, 0.5, 100.0]);
        let t = Tensor::from_storage(storage, vec![4], false).unwrap();
        let bf16 = t.to_dtype::<half::bf16>().unwrap();
        assert_eq!(bf16.shape(), &[4usize]);
        let bits: Vec<u16> = bf16.data().unwrap().iter().map(|b| b.to_bits()).collect();
        let expect: Vec<u16> = [1.0f32, -2.0, 0.5, 100.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_bits())
            .collect();
        assert_eq!(bits, expect);
    }

    #[test]
    fn to_dtype_cpu_bf16_to_f32_widens_exactly() {
        let bf16_data: Vec<half::bf16> = [1.0f32, 1.5, -2.25, 100.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let storage = TensorStorage::cpu(bf16_data.clone());
        let t = Tensor::from_storage(storage, vec![4], false).unwrap();
        let f32_t = t.to_dtype::<f32>().unwrap();
        let got = f32_t.data().unwrap();
        let want: Vec<f32> = bf16_data.iter().map(|b| b.to_f32()).collect();
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.to_bits(), w.to_bits());
        }
    }

    #[test]
    fn to_dtype_cpu_narrow_overflow_matches_torch_inf() {
        let big = TensorStorage::cpu(vec![1e300_f64, -1e300_f64]);
        let t = Tensor::from_storage(big, vec![2], false).unwrap();
        let result = t.to_dtype::<f32>().unwrap();
        assert_eq!(result.data().unwrap(), &[f32::INFINITY, f32::NEG_INFINITY]);
    }

    #[test]
    fn to_dtype_cpu_preserves_shape() {
        let storage = TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = Tensor::from_storage(storage, vec![2, 3], false).unwrap();
        let bf16 = t.to_dtype::<half::bf16>().unwrap();
        assert_eq!(bf16.shape(), &[2usize, 3]);
        assert_eq!(bf16.numel(), 6);
    }

    #[test]
    fn to_dtype_cpu_leaf_backward_reaches_source() {
        let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]), vec![3], true)
            .unwrap();
        let y = x.to_dtype::<f64>().unwrap();
        assert!(y.requires_grad(), "torch: cast output tracks gradients");
        assert!(
            !y.is_leaf(),
            "torch: cast output is a non-leaf ToCopyBackward0"
        );
        let loss = crate::grad_fns::reduction::sum(
            &crate::grad_fns::arithmetic::mul(&y, &y).expect("mul cast output"),
        )
        .expect("sum cast output");
        loss.backward().expect("backward through dtype cast");
        let grad = x
            .grad()
            .expect("grad lookup")
            .expect("cast backward must reach the original f32 leaf");
        assert_eq!(grad.data().expect("grad data"), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn to_dtype_cpu_nonleaf_backward_reaches_source_leaf() {
        let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]), vec![3], true)
            .unwrap();
        let z = crate::grad_fns::arithmetic::mul(&x, &x).expect("x*x");
        let y = z.to_dtype::<f64>().unwrap();
        assert!(y.requires_grad());
        assert!(!y.is_leaf());
        crate::grad_fns::reduction::sum(&y)
            .expect("sum cast nonleaf")
            .backward()
            .expect("backward through nonleaf dtype cast");
        let grad = x
            .grad()
            .expect("grad lookup")
            .expect("cast backward must traverse the source f32 graph");
        assert_eq!(grad.data().expect("grad data"), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn to_dtype_cpu_under_no_grad_does_not_track() {
        let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32]), vec![1], true).unwrap();
        let y = crate::autograd::no_grad::no_grad(|| x.to_dtype::<f64>()).unwrap();
        assert!(!y.requires_grad(), "torch no_grad cast does not track");
        assert!(y.is_leaf());
        assert!(y.grad_fn().is_none());
    }
}
