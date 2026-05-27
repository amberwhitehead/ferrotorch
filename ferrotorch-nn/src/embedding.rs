//! Embedding layer: a lookup table of fixed-size vectors.
//!
//! Maps integer indices (stored as `T` values and cast to `usize`) to
//! dense vectors. This is the standard way to represent discrete tokens
//! (words, subwords, categorical features) as continuous vectors for
//! gradient-based learning.
//!
//! The backward pass implements a sparse scatter-add: only the rows that
//! were accessed receive gradient, and duplicate indices accumulate.
//!
//! ## REQ status (per `.design/ferrotorch-nn/embedding.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl: `pub struct Embedding<T: Float>` here with `weight` / `num_embeddings` / `embedding_dim` / `padding_idx` / `sparse` fields, mirroring `torch/nn/modules/sparse.py:37-50`; non-test consumer: `ferrotorch-llama/src/model.rs` declares `pub embed_tokens: Embedding<T>` as a model field. |
//! | REQ-2 | SHIPPED | impl: the `Embedding::new` constructor here (with `padding_idx` validation + N(0,1) init + padding-row zero); non-test consumer: `Embedding::new(cfg.vocab_size, cfg.hidden_size, None)?` in `ferrotorch-llama/src/model.rs` is the Llama model's token-embedding constructor. |
//! | REQ-3 | SHIPPED | impl: `<Embedding as Module>::forward` body here (gather + grad-attach); non-test consumer: `ferrotorch-llama` model's forward path calls `self.embed_tokens.forward(input_ids)` on every training step and inference token. |
//! | REQ-4 | SHIPPED | impl: `pub struct EmbeddingBackward<T>` and its `GradFn::backward` body here; non-test consumer: every `loss.backward()` call in the Llama training scaffolding traverses `EmbeddingBackward` nodes via `ferrotorch_core::autograd::engine`. |
//! | REQ-5 | SHIPPED | impl: `grad_output.is_cuda()` + `scatter_add_rows_f32/f64` dispatch inside `EmbeddingBackward::backward` here; non-test consumer: `ferrotorch-gpu/src/backend_impl.rs` exposes `Backend::scatter_add_rows_f32`; GPU training-loop runs on the Llama model trigger this on every embedding backward. |
//! | REQ-6 | SHIPPED | impl: the `Embedding::sparse_grad` accessor here returning a `SparseGrad<T>`; non-test consumer: `ferrotorch_optim::SparseAdam` consumes this view in its update path (the `sparse=True` codepath in optim). |
//! | REQ-7 | SHIPPED | impl: `pub struct EmbeddingBag<T: Float>` + `pub enum EmbeddingBagMode` + `Module` impl here; non-test consumer: `pub use embedding::{EmbeddingBag, EmbeddingBagMode}` in `lib.rs` exposes the type for downstream models. |
//! | REQ-8 | SHIPPED | impl: both `Module<T> for Embedding<T>` and `Module<T> for EmbeddingBag<T>` impl blocks here; non-test consumer: `ferrotorch_optim::Optimizer` iterates `model.parameters_mut()` which surfaces the embedding's weight parameter for every step. |
//! | REQ-9 | SHIPPED | impl: free fn `renorm_weight_rows_in_place` here (faithful translation of `embedding_renorm_cpu_` at `aten/src/ATen/native/Embedding.cpp:181-212` — sort+dedup touched rows, row norm via `at::norm` special-cased per `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:191-203` for `norm_type` 0/+inf/-inf, scale rows with norm > max_norm by `max_norm/(norm+1e-7)`, persist via `Tensor::update_data`), called by `Embedding::renorm_weight_in_place` and `EmbeddingBag::forward_bag`; `with_max_norm`/`with_norm_type`/`with_scale_grad_by_freq` builders on `Embedding<T>`, plus `EmbeddingBag::new_with` + `with_max_norm`/`with_norm_type`/`with_scale_grad_by_freq`/`with_sparse`/`with_include_last_offset` and `padding_idx` exclusion in `forward_bag`. `EmbeddingBackward::scale_grad_by_freq` divides each touched row's grad by its forward count (`torch/nn/functional.py:2499-2500`). Renorm runs BEFORE the gather, matching `F.embedding`/`F.embedding_bag` (`functional.py:2561-2573`, `2766-2771`). Consumer surface: per goal.md S5, `Embedding`/`EmbeddingBag` ARE boundary public API (the module mirrors `torch.nn.Embedding`/`torch.nn.EmbeddingBag` field-for-field — the user-facing kwargs ARE the deliverable), grandfathered SHIPPED with no further downstream caller required. The renorm is on the live forward path: `<Embedding as Module>::forward` here calls `self.renorm_weight_in_place(&indices)?` on every forward (no-op when `max_norm` unset), and `EmbeddingBag::forward_bag` / `<EmbeddingBag as Module>::forward` consume the bag kwargs; both types are re-exported via `pub use embedding::{Embedding, EmbeddingBag, EmbeddingBagMode}` in `lib.rs` as the public consumer surface. (NB #1566: the prior cite to `ferrotorch-llama/src/model.rs embed_tokens` as the renorm consumer was FALSE — `model.rs` constructs `Embedding::new(.., None)` with no `max_norm`/`EmbeddingBag`; corrected to the S5 boundary-API rationale.) |
//! | REQ-10 | NOT-STARTED | blocker #1441 (umbrella) — parity-sweep runner arms absent for both `nn.functional.embedding` and `nn.functional.embedding_bag`. Lib tests verify the impl end-to-end. |

use std::any::TypeId;
use std::sync::Arc;

use ferrotorch_core::autograd::no_grad::is_grad_enabled;
use ferrotorch_core::device::Device;
use ferrotorch_core::dtype::DType;
use ferrotorch_core::gpu_dispatch::{GpuBufferHandle, gpu_backend};
use ferrotorch_core::tensor::GradFn;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

use crate::init;
use crate::module::Module;
use crate::parameter::Parameter;

/// Returns `true` if `T` is `f32`.
#[inline]
fn is_f32<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

/// Returns `true` if `T` is `f64`.
#[inline]
fn is_f64<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

/// Upload a CPU `&[f32]` slice to a GPU buffer on the given device ordinal.
fn upload_f32_to_gpu(data: &[f32], ordinal: usize) -> FerrotorchResult<GpuBufferHandle> {
    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    // SAFETY: `data` is a live `&[f32]` borrow; its memory is valid for reads of
    // `data.len() * 4` bytes (every `f32` is exactly 4 bytes — `size_of::<f32>() == 4`,
    // guaranteed by the language and verified by `mem::size_of`). The cast from
    // `*const f32` to `*const u8` does not violate alignment (alignment of `u8` is 1,
    // strictly weaker than `f32`'s alignment of 4). The resulting `&[u8]` is borrowed
    // for the duration of this expression and consumed by `backend.cpu_to_gpu` before
    // `data` goes out of scope, so the lifetime never outlives the source borrow.
    // No interior mutability — `data` is a shared reference and `f32` has no padding.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    backend.cpu_to_gpu(bytes, DType::F32, ordinal)
}

/// Renormalise the rows of `weight` touched by `indices`, IN PLACE, so each
/// touched row's `norm_type`-norm is at most `max_norm`.
///
/// Faithful translation of `embedding_renorm_cpu_`
/// (`aten/src/ATen/native/Embedding.cpp:181-212`): indices are sorted and
/// de-duplicated; each unique row whose norm exceeds `max_norm` is scaled by
/// `max_norm / (norm + 1e-7)`. Rows within `max_norm`, and rows never indexed
/// this forward, are left untouched. PyTorch runs this BEFORE the gather under
/// `torch.no_grad()` (`torch/nn/functional.py:2561-2573`), mutating the
/// persisted `weight`, so the change survives across forward calls — this
/// function matches that by writing the renormed rows back via
/// [`Tensor::update_data`].
///
/// Shared by `Embedding` and `EmbeddingBag` so both layers' `max_norm`
/// semantics stay byte-identical. CUDA weights have no on-device renorm kernel
/// yet, so this returns `NotImplementedOnCuda` rather than silently skipping.
fn renorm_weight_rows_in_place<T: Float>(
    weight: &Tensor<T>,
    indices: &[usize],
    dim: usize,
    max_norm: f64,
    norm_type: f64,
    op: &'static str,
) -> FerrotorchResult<()> {
    if weight.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op });
    }

    // Sort + dedup, mirroring `std::sort` + the `sorted[i]==sorted[i-1]` skip
    // at Embedding.cpp:193-201. Visiting each unique row once is required:
    // re-scaling an already-clipped row would shrink it below max_norm.
    let mut sorted: Vec<usize> = indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    let weight_data = weight.data()?;
    let mut new_data: Option<Vec<T>> = None;
    for &idx in &sorted {
        let row_start = idx * dim;
        let row = &weight_data[row_start..row_start + dim];
        // `row.norm(norm_type)` = `at::norm`, which special-cases the
        // non-finite / degenerate orders rather than evaluating the generic
        // `(Σ|x|^p)^(1/p)` formula — that formula gives `inf^0 = 1` for
        // `p = +inf` and `x^0 = 1` for `p = 0`, both wrong. Mirror the kernel
        // dispatch at `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:191-203`:
        //   p == 0     -> NormZeroOps : count of nonzero elements (L0)
        //   p == +inf  -> AbsMaxOps   : max_i |x_i|  (infinity norm)
        //   p == -inf  -> AbsMinOps   : min_i |x_i|  (acc seeded +inf)
        //   else       -> NormOps     : (Σ|x|^p)^(1/p)
        // (p == 1 / p == 2 are exact under the generic formula, so they need
        // no separate arm here.)
        let norm = if norm_type == 0.0 {
            // NormZeroOps (`SharedReduceOps.h:285`): count of nonzeros.
            row.iter()
                .filter(|&&v| num_traits::ToPrimitive::to_f64(&v).unwrap_or(0.0) != 0.0)
                .count() as f64
        } else if norm_type == f64::INFINITY {
            // AbsMaxOps (`SharedReduceOps.h:216`): max_i |x_i|.
            row.iter().fold(0.0f64, |acc, &v| {
                acc.max(num_traits::ToPrimitive::to_f64(&v).unwrap_or(0.0).abs())
            })
        } else if norm_type == f64::NEG_INFINITY {
            // AbsMinOps (`SharedReduceOps.h:186`): min_i |x_i|, acc seeded +inf.
            row.iter().fold(f64::INFINITY, |acc, &v| {
                acc.min(num_traits::ToPrimitive::to_f64(&v).unwrap_or(0.0).abs())
            })
        } else {
            // NormOps: generic finite p-norm `(Σ|x|^p)^(1/p)`.
            let mut acc = 0.0f64;
            for &v in row {
                let vf = num_traits::ToPrimitive::to_f64(&v).unwrap_or(0.0);
                acc += vf.abs().powf(norm_type);
            }
            acc.powf(1.0 / norm_type)
        };
        if norm > max_norm {
            // Lazily materialise the mutable copy only when a row needs
            // clipping, so the no-clip case never touches the buffer.
            let buf = new_data.get_or_insert_with(|| weight_data.to_vec());
            let scale = max_norm / (norm + 1e-7);
            let scale_t = T::from(scale).unwrap();
            for v in &mut buf[row_start..row_start + dim] {
                *v = *v * scale_t;
            }
        }
    }

    if let Some(buf) = new_data {
        // SAFETY: `update_data` requires exclusive access to the weight's
        // storage for the duration of the write. The renorm runs inside the
        // forward, which holds the only live borrow of `weight_data` (a
        // `&[T]` over the same Arc); that borrow ends before this call (the
        // slice is fully consumed into `buf` above). No backward node captures
        // a mutable view, and the autograd engine is not concurrently reading
        // the weight: PyTorch performs this exact mutation under
        // `torch.no_grad()` (`functional.py:2567-2572`), a grad-disabled,
        // single-threaded in-place edit of the persisted weight. `buf` has
        // exactly `num_embeddings * dim` elements, matching the tensor's numel.
        #[allow(
            clippy::undocumented_unsafe_blocks,
            reason = "SAFETY comment above documents the exclusive-access invariant; torch embedding_renorm_ mutates weight in place under no_grad (functional.py:2567-2572), matching the optimizer step()'s update_data contract"
        )]
        unsafe {
            weight.update_data(&buf)?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// EmbeddingBackward
// ---------------------------------------------------------------------------

/// Backward function for the embedding lookup.
///
/// Forward: `output[i, :] = weight[indices[i], :]`
///
/// VJP: `grad_weight = zeros(num_embeddings, embedding_dim);`
///       `for i, idx in indices: grad_weight[idx, :] += grad_output[i, :]`
///
/// This is a sparse gradient — only accessed rows are non-zero.
/// Duplicate indices accumulate their corresponding `grad_output` rows.
#[derive(Debug)]
pub struct EmbeddingBackward<T: Float> {
    /// The weight tensor (needed for graph traversal and shape).
    weight: Tensor<T>,
    /// Indices used in the forward pass.
    indices: Vec<usize>,
    /// Total number of embedding rows.
    num_embeddings: usize,
    /// Width of each embedding vector.
    embedding_dim: usize,
    /// If set, this row's gradient is always zero.
    padding_idx: Option<usize>,
    /// If `true`, divide each row's accumulated gradient by the number of
    /// times the index appeared in the forward pass — mirrors
    /// `torch/nn/functional.py:2374-2388`'s `scale_grad_by_freq=True`
    /// branch. (Closes #1445.)
    scale_grad_by_freq: bool,
}

impl<T: Float> GradFn<T> for EmbeddingBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None]);
        }

        let dim = self.embedding_dim;

        // GPU fast path: scatter-add rows entirely on GPU for f32/f64 tensors.
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let ordinal = match self.weight.device() {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };

            let indices_f32: Vec<f32> = self.indices.iter().map(|&i| i as f32).collect();
            let idx_handle = upload_f32_to_gpu(&indices_f32, ordinal)?;
            let go_handle = grad_output.gpu_handle()?;
            let f64_path = is_f64::<T>();
            let elem_size: usize = if f64_path { 8 } else { 4 };

            let mut gw_handle = if f64_path {
                backend.scatter_add_rows_f64(go_handle, &idx_handle, self.num_embeddings, dim)?
            } else {
                backend.scatter_add_rows_f32(go_handle, &idx_handle, self.num_embeddings, dim)?
            };

            if let Some(pad_idx) = self.padding_idx {
                let mut gw_bytes = backend.gpu_to_cpu(&gw_handle)?;
                let start_byte = pad_idx * dim * elem_size;
                let end_byte = start_byte + dim * elem_size;
                for b in &mut gw_bytes[start_byte..end_byte] {
                    *b = 0;
                }
                let gw_dtype = if f64_path { DType::F64 } else { DType::F32 };
                gw_handle = backend.cpu_to_gpu(&gw_bytes, gw_dtype, ordinal)?;
            }

            let grad_tensor = Tensor::from_storage(
                TensorStorage::gpu(gw_handle),
                vec![self.num_embeddings, dim],
                false,
            )?;
            return Ok(vec![Some(grad_tensor)]);
        }

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "EmbeddingBackward",
            });
        }

        let go_data = grad_output.data()?;

        // Allocate a full-size gradient for the weight matrix, initialized to zero.
        let mut grad_weight = vec![<T as num_traits::Zero>::zero(); self.num_embeddings * dim];

        // Scatter-add: for each index position, accumulate the corresponding
        // grad_output row into the weight gradient at the accessed index.
        for (i, &idx) in self.indices.iter().enumerate() {
            let go_row = &go_data[i * dim..(i + 1) * dim];
            let gw_row = &mut grad_weight[idx * dim..(idx + 1) * dim];
            for (gw, &go) in gw_row.iter_mut().zip(go_row.iter()) {
                *gw += go;
            }
        }

        // scale_grad_by_freq: divide each touched row by its appearance
        // count in the forward pass (mirrors
        // `torch/nn/functional.py:2374-2388`). Untouched rows have grad
        // identically zero, so the divide is a no-op there.
        if self.scale_grad_by_freq {
            let mut counts: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            for &idx in &self.indices {
                *counts.entry(idx).or_insert(0) += 1;
            }
            for (&idx, &cnt) in &counts {
                if cnt <= 1 {
                    continue;
                }
                let scale = T::from(1.0 / cnt as f64).unwrap();
                let row_start = idx * dim;
                for v in &mut grad_weight[row_start..row_start + dim] {
                    *v = *v * scale;
                }
            }
        }

        // If padding_idx is set, zero that row's gradient unconditionally.
        if let Some(pad_idx) = self.padding_idx {
            let start = pad_idx * dim;
            for v in &mut grad_weight[start..start + dim] {
                *v = <T as num_traits::Zero>::zero();
            }
        }

        Ok(vec![Some(Tensor::from_storage(
            TensorStorage::cpu(grad_weight),
            vec![self.num_embeddings, dim],
            false,
        )?)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.weight]
    }

    fn name(&self) -> &'static str {
        "EmbeddingBackward"
    }
}

// ---------------------------------------------------------------------------
// Embedding layer
// ---------------------------------------------------------------------------

/// A simple lookup table that stores embeddings of a fixed dictionary.
///
/// Given a 1-D tensor of integer indices (stored as float values, cast to
/// `usize`), returns a 2-D tensor `[len, embedding_dim]` by gathering the
/// corresponding rows from the weight matrix.
///
/// # Padding index
///
/// If `padding_idx` is set, the embedding vector at that index is always
/// zero and receives no gradient updates. This is commonly used to
/// represent a padding token.
///
/// # Example
///
/// ```ignore
/// let emb = Embedding::<f32>::new(1000, 64, None)?;
/// let indices = ferrotorch_core::tensor(&[1.0, 5.0, 3.0])?;
/// let output = emb.forward(&indices)?;
/// assert_eq!(output.shape(), &[3, 64]);
/// ```
#[derive(Debug)]
pub struct Embedding<T: Float> {
    /// The learnable weight matrix, shape `[num_embeddings, embedding_dim]`.
    pub weight: Parameter<T>,
    /// Number of entries in the lookup table.
    pub num_embeddings: usize,
    /// Dimensionality of each embedding vector.
    pub embedding_dim: usize,
    /// If set, this row is kept at zero and receives no gradient.
    pub padding_idx: Option<usize>,
    /// If set, every row touched by a forward call is renormalised in-place
    /// so its `norm_type`-norm is at most `max_norm`, mirroring
    /// `torch/nn/functional.py:2306-2370` (`_no_grad_embedding_renorm_`).
    /// Carried as `f64` for the upstream scalar type (kwarg is `float`).
    /// (Closes #1445.)
    pub max_norm: Option<f64>,
    /// Order of the row-norm used when `max_norm` is active. Defaults to
    /// `2.0` (Euclidean) per `torch/nn/functional.py:2316`. (Closes #1445.)
    pub norm_type: f64,
    /// If `true`, `EmbeddingBackward` divides each accumulated row gradient
    /// by the number of times that index appeared in the forward pass,
    /// matching `torch/nn/functional.py:2374-2388`. (Closes #1445.)
    pub scale_grad_by_freq: bool,
    /// Whether the module is in training mode.
    training: bool,
    /// If true, advertise a sparse gradient pattern (the only rows touched
    /// are the ones actually indexed in the most recent forward call).
    /// This is purely a flag — autograd still populates a dense grad on
    /// the weight; callers can extract a `SparseGrad` view via
    /// [`Self::sparse_grad`] to feed `optim::SparseAdam` or
    /// `SparseGrad::apply_sgd` without scanning the full dense matrix.
    /// Mirrors `torch.nn.Embedding(sparse=True)`. (#623)
    pub sparse: bool,
    /// Cached unique indices touched by the most recent forward pass. None
    /// if `sparse == false` or no forward has run yet. We dedupe here so
    /// callers don't have to coalesce the SparseGrad themselves.
    last_indices: std::sync::Mutex<Option<Vec<usize>>>,
}

impl<T: Float> Embedding<T> {
    /// Create a new embedding layer.
    ///
    /// Weight is initialized from N(0, 1). If `padding_idx` is set, that
    /// row is zeroed after initialization.
    ///
    /// # Errors
    ///
    /// Returns an error if `padding_idx >= num_embeddings`.
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
    ) -> FerrotorchResult<Self> {
        // Validate padding_idx.
        if let Some(idx) = padding_idx {
            if idx >= num_embeddings {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "padding_idx {idx} is out of range for num_embeddings {num_embeddings}"
                    ),
                });
            }
        }

        // Initialize weight from N(0, 1).
        let mut weight = Parameter::zeros(&[num_embeddings, embedding_dim])?;
        init::normal(&mut weight, 0.0, 1.0)?;

        // Zero the padding row if requested.
        if let Some(idx) = padding_idx {
            let data = weight.data()?.to_vec();
            let mut new_data = data;
            let start = idx * embedding_dim;
            for v in &mut new_data[start..start + embedding_dim] {
                *v = <T as num_traits::Zero>::zero();
            }
            weight = Parameter::new(Tensor::from_storage(
                TensorStorage::cpu(new_data),
                vec![num_embeddings, embedding_dim],
                true,
            )?);
        }

        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
            training: true,
            sparse: false,
            last_indices: std::sync::Mutex::new(None),
        })
    }

    /// Create an embedding layer from an existing weight tensor.
    ///
    /// The tensor must have shape `[num_embeddings, embedding_dim]`.
    pub fn from_pretrained(
        weight: Tensor<T>,
        padding_idx: Option<usize>,
    ) -> FerrotorchResult<Self> {
        if weight.ndim() != 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Embedding weight must be 2-D, got shape {:?}",
                    weight.shape()
                ),
            });
        }
        let num_embeddings = weight.shape()[0];
        let embedding_dim = weight.shape()[1];

        if let Some(idx) = padding_idx {
            if idx >= num_embeddings {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "padding_idx {idx} is out of range for num_embeddings {num_embeddings}"
                    ),
                });
            }
        }

        Ok(Self {
            weight: Parameter::new(weight),
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
            training: true,
            sparse: false,
            last_indices: std::sync::Mutex::new(None),
        })
    }

    /// Builder: set the maximum row norm. After every forward pass, rows
    /// of `weight` touched by the input have their `norm_type`-norm clipped
    /// to `max_norm` via in-place renormalisation, matching
    /// `torch.nn.Embedding(max_norm=...)`. Closes #1445.
    pub fn with_max_norm(mut self, max_norm: f64) -> Self {
        self.max_norm = Some(max_norm);
        self
    }

    /// Builder: set the order of the row-norm used by `max_norm` (default
    /// `2.0`). Closes #1445.
    pub fn with_norm_type(mut self, norm_type: f64) -> Self {
        self.norm_type = norm_type;
        self
    }

    /// Builder: if `true`, `EmbeddingBackward` divides each touched row's
    /// gradient by the number of times the index appeared in the forward
    /// (`torch.nn.Embedding(scale_grad_by_freq=True)`). Closes #1445.
    pub fn with_scale_grad_by_freq(mut self, scale: bool) -> Self {
        self.scale_grad_by_freq = scale;
        self
    }

    /// Renormalise the rows of `self.weight` that `indices` touched, IN
    /// PLACE, so each touched row's `norm_type`-norm is at most `max_norm`.
    ///
    /// This is a faithful translation of the aten kernel
    /// `embedding_renorm_cpu_` (`aten/src/ATen/native/Embedding.cpp:181-212`):
    /// the touched indices are sorted and de-duplicated, and for each unique
    /// row whose current norm exceeds `max_norm` the row is scaled by
    /// `max_norm / (norm + 1e-7)`. Rows already within `max_norm` are left
    /// untouched, and rows never indexed in this forward are not visited.
    ///
    /// PyTorch's `F.embedding` (`torch/nn/functional.py:2561-2573`) runs this
    /// renorm BEFORE the gather, under `torch.no_grad()`, mutating the
    /// persisted `weight` tensor — so the change survives across forward
    /// calls. We match that by writing the renormed rows back into
    /// `self.weight` via [`Tensor::update_data`], the same in-place storage
    /// mutation the optimizer `step()` uses. The write is performed only when
    /// at least one row actually exceeded `max_norm`, keeping the common
    /// "nothing to clip" path allocation-free on the weight buffer.
    ///
    /// Returns `Ok(())` when `max_norm` is unset (no-op) or after the
    /// in-place mutation completes.
    fn renorm_weight_in_place(&self, indices: &[usize]) -> FerrotorchResult<()> {
        let Some(max_norm) = self.max_norm else {
            return Ok(());
        };
        renorm_weight_rows_in_place(
            self.weight.tensor(),
            indices,
            self.embedding_dim,
            max_norm,
            self.norm_type,
            "Embedding(max_norm) weight renorm",
        )
    }

    /// Toggle the sparse-grad mode. When enabled, [`Self::sparse_grad`]
    /// returns a `SparseGrad<T>` populated only with the rows actually
    /// touched by the most recent forward pass. Off by default. Returns
    /// `&mut self` for chaining.
    pub fn with_sparse(mut self, sparse: bool) -> Self {
        self.sparse = sparse;
        self
    }

    /// Record the unique row indices touched by the most recent forward pass.
    /// No-op when sparse mode is off — keeps the hot path zero-overhead for
    /// the common dense-grad case.
    fn cache_touched_rows(&self, indices: &[usize]) {
        if !self.sparse {
            return;
        }
        // Dedupe (sorted) so callers don't have to coalesce later.
        let mut uniq: Vec<usize> = indices.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        if let Ok(mut g) = self.last_indices.lock() {
            *g = Some(uniq);
        }
    }

    /// Materialize a [`SparseGrad`] from the current dense weight gradient,
    /// keyed on the indices touched by the most recent forward pass.
    ///
    /// Returns `None` when sparse mode is off, no forward has been run yet,
    /// or the parameter has no gradient (e.g. before the first backward
    /// call). The returned grad is already coalesced (each touched row
    /// appears once with its full gradient slab) — feed it directly into
    /// [`SparseGrad::apply_sgd`] or `optim::SparseAdam`.
    ///
    /// Mirrors PyTorch's `embedding_bag(..., sparse=True)` → `SparseAdam`
    /// flow. The dense grad is unchanged; `sparse_grad` just provides a
    /// compact view for optimizers that benefit from skipping zero rows.
    pub fn sparse_grad(&self) -> FerrotorchResult<Option<ferrotorch_core::SparseGrad<T>>> {
        if !self.sparse {
            return Ok(None);
        }
        let last = match self.last_indices.lock() {
            Ok(g) => g,
            Err(_) => return Ok(None),
        };
        let indices = match last.as_ref() {
            Some(v) => v.clone(),
            None => return Ok(None),
        };
        let grad = match self.weight.tensor().grad()? {
            Some(g) => g,
            None => return Ok(None),
        };
        let grad_data = grad.data_vec()?;
        let dim = self.embedding_dim;
        let mut values = Vec::with_capacity(indices.len() * dim);
        for &idx in &indices {
            let row_start = idx * dim;
            let row_end = row_start + dim;
            values.extend_from_slice(&grad_data[row_start..row_end]);
        }
        let sg = ferrotorch_core::SparseGrad::new(indices, values, vec![dim])?;
        Ok(Some(sg))
    }
}

impl<T: Float> Module<T> for Embedding<T> {
    /// Forward pass: look up embedding vectors for the given indices.
    ///
    /// `input` is an index tensor of ANY shape whose values are non-negative
    /// integers stored as floats. Each value is cast to `usize` and used to
    /// index into the weight matrix. The lookup operates on the flattened
    /// indices (row-major), exactly mirroring upstream `embedding_symint`
    /// (`aten/src/ATen/native/Embedding.cpp:43-53`):
    /// `weight.index_select(0, indices.reshape(-1)).view_symint(size)` where
    /// `size = (*indices.sizes(), weight.size(1))`.
    ///
    /// Returns a tensor of shape `(*input.shape(), embedding_dim)`. A 1-D
    /// index of length `n` therefore yields `[n, embedding_dim]`, and a 2-D
    /// index `[a, b]` yields `[a, b, embedding_dim]`, matching `F.embedding`.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let dim = self.embedding_dim;

        // Output shape is the index shape with `embedding_dim` appended, per
        // upstream `embedding_symint` (`Embedding.cpp:48-53`): the gather runs
        // over the flattened indices and the result is viewed back to
        // `(*indices.sizes(), weight.size(1))`. A 1-D input keeps the existing
        // `[n, dim]` behavior (the empty-prefix special-case is implicit).
        let mut output_shape: Vec<usize> = input.shape().to_vec();
        output_shape.push(dim);

        // GPU fast path for f32/f64 embeddings: gather rows entirely on GPU.
        if self.weight.tensor().is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let device = self.weight.tensor().device();
            let ordinal = match device {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };

            let input_data = input.data_vec()?;
            let n = input_data.len();

            let mut indices = Vec::with_capacity(n);
            let mut indices_f32 = Vec::with_capacity(n);
            for (i, &val) in input_data.iter().enumerate() {
                let idx = num_traits::ToPrimitive::to_usize(&val).ok_or_else(|| {
                    FerrotorchError::InvalidArgument {
                        message: format!(
                            "Embedding index at position {i} cannot be converted to usize: {val:?}"
                        ),
                    }
                })?;
                if idx >= self.num_embeddings {
                    return Err(FerrotorchError::IndexOutOfBounds {
                        index: idx,
                        axis: 0,
                        size: self.num_embeddings,
                    });
                }
                indices.push(idx);
                indices_f32.push(idx as f32);
            }

            self.cache_touched_rows(&indices);

            // max_norm with a CUDA weight has no on-device renorm kernel yet;
            // surface that explicitly rather than silently returning
            // un-renormed rows (which would diverge from torch's in-place
            // mutation at functional.py:2561-2573). No-op when max_norm unset.
            self.renorm_weight_in_place(&indices)?;

            let idx_handle = upload_f32_to_gpu(&indices_f32, ordinal)?;
            let weight_handle = self.weight.tensor().gpu_handle()?;

            let output_handle = if is_f64::<T>() {
                backend.embed_lookup_batch_f64(&idx_handle, weight_handle, n, dim)?
            } else {
                backend.embed_lookup_batch_f32(&idx_handle, weight_handle, n, dim)?
            };

            // Padding index: if set, zero the corresponding output rows on GPU.
            // For padding_idx, the weight row should already be zero, so output
            // rows at padding positions should already be zero. Be defensive
            // only if padding_idx is actually referenced.
            // (The weight is zeroed at init, so we skip extra GPU work here.)

            let storage = TensorStorage::gpu(output_handle);

            if self.weight.requires_grad() && is_grad_enabled() {
                let grad_fn = Arc::new(EmbeddingBackward {
                    weight: self.weight.tensor().clone(),
                    indices,
                    num_embeddings: self.num_embeddings,
                    embedding_dim: dim,
                    padding_idx: self.padding_idx,
                    scale_grad_by_freq: self.scale_grad_by_freq,
                });
                return Tensor::from_operation(storage, output_shape, grad_fn);
            } else {
                return Tensor::from_storage(storage, output_shape, false);
            }
        }

        // CPU path — non-f32 GPU tensors have no GPU kernel, error out.
        if self.weight.tensor().is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "Embedding" });
        }
        let input_data = input.data_vec()?;
        let n = input_data.len();

        // Convert float indices to usize and validate bounds.
        let mut indices = Vec::with_capacity(n);
        for (i, &val) in input_data.iter().enumerate() {
            let idx = num_traits::ToPrimitive::to_usize(&val).ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: format!(
                        "Embedding index at position {i} cannot be converted to usize: {val:?}"
                    ),
                }
            })?;
            if idx >= self.num_embeddings {
                return Err(FerrotorchError::IndexOutOfBounds {
                    index: idx,
                    axis: 0,
                    size: self.num_embeddings,
                });
            }
            indices.push(idx);
        }

        self.cache_touched_rows(&indices);

        // max_norm: renormalise the touched rows of the PERSISTED weight
        // IN PLACE, BEFORE the gather. This mirrors
        // `torch/nn/functional.py:2561-2573`, where `F.embedding` calls
        // `_no_grad_embedding_renorm_(weight, ...)` (which mutates `weight`
        // via `torch.embedding_renorm_`) and only THEN does the lookup.
        // The mutation persists across forward calls — a second forward with
        // the same indices is a no-op because the rows now satisfy max_norm.
        // Closes #1445 (CPU path).
        self.renorm_weight_in_place(&indices)?;

        // Re-read the (possibly mutated) weight buffer for the gather.
        let cpu_weight = self.weight.tensor().clone();
        let weight_data = cpu_weight.data()?;

        // Gather rows from weight.
        let mut output_data = Vec::with_capacity(n * dim);
        for &idx in &indices {
            let row_start = idx * dim;
            output_data.extend_from_slice(&weight_data[row_start..row_start + dim]);
        }

        // If padding_idx is set, ensure those rows are zeros in the output
        // (they should already be zero in the weight, but be defensive).
        if let Some(pad_idx) = self.padding_idx {
            for (i, &idx) in indices.iter().enumerate() {
                if idx == pad_idx {
                    let start = i * dim;
                    for v in &mut output_data[start..start + dim] {
                        *v = <T as num_traits::Zero>::zero();
                    }
                }
            }
        }

        // Output device matches the weight's device (GPU if model is on GPU).
        let device = self.weight.tensor().device();

        // Build storage on the target device first, then attach grad_fn.
        // This avoids to() stripping the grad_fn by creating a leaf tensor.
        let storage = if device.is_cuda() {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let ordinal = match device {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };
            // SAFETY: `output_data` is a live owned `Vec<T>` whose contents we borrow
            // shared for the duration of this expression. Its underlying buffer is valid
            // for reads of `output_data.len() * size_of::<T>()` bytes — `T: Float`
            // is one of f32/f64/bf16/f16, none of which have padding bytes (no struct
            // wrappers, no niches), so the byte-length calculation is exact. The cast
            // `*const T` -> `*const u8` does not violate alignment because `u8`'s
            // alignment (1) is at most `T`'s alignment. The resulting `&[u8]` is
            // consumed by `backend.cpu_to_gpu` before `output_data` is moved into
            // `TensorStorage::cpu` on the else branch (mutually exclusive paths) or
            // dropped here, so the borrow never outlives the source.
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    output_data.as_ptr() as *const u8,
                    output_data.len() * std::mem::size_of::<T>(),
                )
            };
            let handle = backend.cpu_to_gpu(bytes, T::dtype(), ordinal)?;
            TensorStorage::gpu(handle)
        } else {
            TensorStorage::cpu(output_data)
        };

        if self.weight.requires_grad() && is_grad_enabled() {
            let grad_fn = Arc::new(EmbeddingBackward {
                weight: self.weight.tensor().clone(),
                indices,
                num_embeddings: self.num_embeddings,
                embedding_dim: dim,
                padding_idx: self.padding_idx,
                scale_grad_by_freq: self.scale_grad_by_freq,
            });
            Tensor::from_operation(storage, output_shape, grad_fn)
        } else {
            Tensor::from_storage(storage, output_shape, false)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![&mut self.weight]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![("weight".to_string(), &self.weight)]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// EmbeddingBag — fused lookup + reduce
// ---------------------------------------------------------------------------

/// Reduction mode for [`EmbeddingBag`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingBagMode {
    /// Sum all embeddings in each bag.
    Sum,
    /// Mean of all embeddings in each bag.
    Mean,
    /// Element-wise max across embeddings in each bag.
    Max,
}

/// Computes sums or means of bags of embeddings without instantiating the
/// full intermediate embeddings. This is more efficient than `Embedding`
/// followed by a reduction for variable-length sequences.
///
/// # Input format
///
/// - `input`: 1-D tensor of indices [total_indices]
/// - `offsets`: 1-D tensor [num_bags] giving the start index of each bag
///   in `input`. Must be sorted and non-negative. Example: if `input` has
///   indices for 3 bags with lengths [2, 3, 1], then `offsets = [0, 2, 5]`.
///
/// # Modes
///
/// - `Sum`: output[b] = sum of weight[input[offsets[b]:offsets[b+1]]]
/// - `Mean`: output[b] = mean of weight[input[offsets[b]:offsets[b+1]]]
/// - `Max`: output[b] = element-wise max of weight[input[offsets[b]:offsets[b+1]]]
#[derive(Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "scale_grad_by_freq/sparse/include_last_offset/training each mirror a distinct torch.nn.EmbeddingBag kwarg (sparse.py:376-380) — R-DEV-2 requires matching the upstream Python API surface field-for-field, so collapsing them into a flags enum would diverge from the user-facing kwarg contract"
)]
pub struct EmbeddingBag<T: Float> {
    weight: Parameter<T>,
    num_embeddings: usize,
    embedding_dim: usize,
    mode: EmbeddingBagMode,
    training: bool,
    /// If set, each touched weight row is renormalised in place to at most
    /// `max_norm` under the `norm_type`-norm before the bag reduction,
    /// mirroring `torch.nn.EmbeddingBag(max_norm=...)`
    /// (`torch/nn/modules/sparse.py:374`, `functional.py:2766-2771`).
    pub max_norm: Option<f64>,
    /// Order of the row-norm used when `max_norm` is active. Defaults to
    /// `2.0` per `torch/nn/modules/sparse.py:375`.
    pub norm_type: f64,
    /// If `true`, future gradient accumulation scales each row by the inverse
    /// frequency of its index in the mini-batch. Carried to mirror the
    /// upstream kwarg (`sparse.py:376`); `max` mode forbids it
    /// (`functional.py:2755-2758`).
    pub scale_grad_by_freq: bool,
    /// Advertises a sparse-gradient pattern, mirroring
    /// `torch.nn.EmbeddingBag(sparse=True)` (`sparse.py:378`). `max` mode
    /// forbids it (`functional.py:2760-2761`).
    pub sparse: bool,
    /// When `true`, `offsets` has `num_bags + 1` entries and its last entry
    /// is the total index count (CSR-style), mirroring
    /// `torch.nn.EmbeddingBag(include_last_offset=True)` (`sparse.py:380`,
    /// `functional.py:2621-2624`).
    pub include_last_offset: bool,
    /// If set, indices equal to `padding_idx` are excluded from each bag's
    /// reduction (and the mean divisor), and the corresponding weight row is
    /// zeroed at construction — matching `torch.nn.EmbeddingBag(padding_idx)`
    /// (`sparse.py:381`, `aten/src/ATen/native/EmbeddingBag.cpp:140-156`).
    pub padding_idx: Option<usize>,
}

impl<T: Float> EmbeddingBag<T> {
    /// Create a new EmbeddingBag with default kwargs (no `max_norm`,
    /// `norm_type = 2.0`, `scale_grad_by_freq = false`, `sparse = false`,
    /// `include_last_offset = false`, no `padding_idx`), matching the
    /// `torch.nn.EmbeddingBag(num_embeddings, embedding_dim, mode=...)`
    /// defaults at `torch/nn/modules/sparse.py:370-381`.
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        mode: EmbeddingBagMode,
    ) -> FerrotorchResult<Self> {
        Self::new_with(num_embeddings, embedding_dim, mode, None)
    }

    /// Create a new EmbeddingBag, optionally with a `padding_idx`.
    ///
    /// Mirrors the `padding_idx` validation + zero-fill in
    /// `torch.nn.EmbeddingBag.__init__` / `_fill_padding_idx_with_zero`
    /// (`torch/nn/modules/sparse.py:392-423`): `padding_idx` must be within
    /// `num_embeddings`, and that weight row is zeroed after init.
    ///
    /// # Errors
    ///
    /// Returns an error if `padding_idx >= num_embeddings`.
    pub fn new_with(
        num_embeddings: usize,
        embedding_dim: usize,
        mode: EmbeddingBagMode,
        padding_idx: Option<usize>,
    ) -> FerrotorchResult<Self> {
        if let Some(idx) = padding_idx {
            if idx >= num_embeddings {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "padding_idx {idx} must be within num_embeddings {num_embeddings}"
                    ),
                });
            }
        }

        let mut weight = Parameter::zeros(&[num_embeddings, embedding_dim])?;
        init::normal(&mut weight, 0.0, 1.0)?;

        // Zero the padding row if requested (mirrors
        // `_fill_padding_idx_with_zero`, sparse.py:420-423).
        if let Some(idx) = padding_idx {
            let data = weight.data()?.to_vec();
            let mut new_data = data;
            let start = idx * embedding_dim;
            for v in &mut new_data[start..start + embedding_dim] {
                *v = <T as num_traits::Zero>::zero();
            }
            weight = Parameter::new(Tensor::from_storage(
                TensorStorage::cpu(new_data),
                vec![num_embeddings, embedding_dim],
                true,
            )?);
        }

        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
            mode,
            training: true,
            max_norm: None,
            norm_type: 2.0,
            scale_grad_by_freq: false,
            sparse: false,
            include_last_offset: false,
            padding_idx,
        })
    }

    /// Builder: set the maximum row norm. Touched rows of `weight` have their
    /// `norm_type`-norm clipped to `max_norm` in place before each bag
    /// reduction, mirroring `torch.nn.EmbeddingBag(max_norm=...)`. Closes #1445.
    pub fn with_max_norm(mut self, max_norm: f64) -> Self {
        self.max_norm = Some(max_norm);
        self
    }

    /// Builder: set the order of the row-norm used by `max_norm` (default
    /// `2.0`, `sparse.py:375`). Closes #1445.
    pub fn with_norm_type(mut self, norm_type: f64) -> Self {
        self.norm_type = norm_type;
        self
    }

    /// Builder: set `scale_grad_by_freq` (`sparse.py:376`). Rejected for
    /// `max` mode by [`Self::forward_bag`], matching `functional.py:2755-2758`.
    /// Closes #1445.
    pub fn with_scale_grad_by_freq(mut self, scale: bool) -> Self {
        self.scale_grad_by_freq = scale;
        self
    }

    /// Builder: set `sparse` (`sparse.py:378`). Rejected for `max` mode by
    /// [`Self::forward_bag`], matching `functional.py:2760-2761`. Closes #1445.
    pub fn with_sparse(mut self, sparse: bool) -> Self {
        self.sparse = sparse;
        self
    }

    /// Builder: set `include_last_offset` (`sparse.py:380`). When `true`,
    /// `offsets` carries `num_bags + 1` entries with the last being the total
    /// index count, matching the CSR convention in `functional.py:2621-2624`.
    /// Closes #1445.
    pub fn with_include_last_offset(mut self, include_last_offset: bool) -> Self {
        self.include_last_offset = include_last_offset;
        self
    }

    /// Forward pass: compute bag-reduced embeddings.
    ///
    /// `input`: 1-D tensor of indices `[total_indices]`.
    /// `offsets`: bag start offsets. When `include_last_offset == false`,
    /// this has `num_bags` entries (bag `b` spans `offsets[b]..offsets[b+1]`,
    /// the last bag running to the end of `input`). When
    /// `include_last_offset == true`, it has `num_bags + 1` entries with the
    /// final entry being the total index count (CSR style), matching
    /// `torch/nn/functional.py:2621-2624`.
    ///
    /// Honors `max_norm` (in-place weight renorm before the reduction,
    /// mirroring `functional.py:2766-2771`) and `padding_idx` (indices equal
    /// to it are excluded from both the reduction and the mean divisor,
    /// mirroring `aten/src/ATen/native/EmbeddingBag.cpp:140-156`). `max` mode
    /// rejects `scale_grad_by_freq` / `sparse`
    /// (`functional.py:2755-2761`).
    pub fn forward_bag(&self, input: &Tensor<T>, offsets: &[usize]) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("EmbeddingBag input must be 1-D, got {:?}", input.shape()),
            });
        }

        // mode='max' forbids scale_grad_by_freq and sparse, matching
        // functional.py:2755-2761.
        if self.mode == EmbeddingBagMode::Max {
            if self.scale_grad_by_freq {
                return Err(FerrotorchError::InvalidArgument {
                    message: "max mode does not support scaling the gradient by the frequency"
                        .into(),
                });
            }
            if self.sparse {
                return Err(FerrotorchError::InvalidArgument {
                    message: "max mode does not support sparse weights".into(),
                });
            }
        }

        let input_data = input.data_vec()?;
        let dim = self.embedding_dim;
        let total = input_data.len();

        // Materialise the bag boundaries from `offsets`, honoring
        // include_last_offset (CSR layout: trailing entry == total count).
        let num_bags = if self.include_last_offset {
            offsets.len().saturating_sub(1)
        } else {
            offsets.len()
        };

        // Validate + collect indices.
        let mut indices = Vec::with_capacity(total);
        for (i, &val) in input_data.iter().enumerate() {
            let idx = num_traits::ToPrimitive::to_usize(&val).ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: format!("EmbeddingBag index {i} invalid: {val:?}"),
                }
            })?;
            if idx >= self.num_embeddings {
                return Err(FerrotorchError::IndexOutOfBounds {
                    index: idx,
                    axis: 0,
                    size: self.num_embeddings,
                });
            }
            indices.push(idx);
        }

        // max_norm: renormalise the touched rows of the persisted weight IN
        // PLACE before the reduction (functional.py:2766-2771 runs the renorm
        // before torch.embedding_bag). No-op when max_norm unset.
        if let Some(max_norm) = self.max_norm {
            renorm_weight_rows_in_place(
                self.weight.tensor(),
                &indices,
                dim,
                max_norm,
                self.norm_type,
                "EmbeddingBag(max_norm) weight renorm",
            )?;
        }

        // Re-read the (possibly renormed) weight for the reduction.
        let weight_data = self.weight.tensor().data()?;

        let mut output = vec![<T as num_traits::Zero>::zero(); num_bags * dim];

        for b in 0..num_bags {
            let start = offsets[b];
            // With include_last_offset, every bag (including the last) reads
            // its end from offsets[b+1]; otherwise the final bag runs to total.
            let end = if self.include_last_offset || b + 1 < num_bags {
                offsets[b + 1]
            } else {
                total
            };

            match self.mode {
                EmbeddingBagMode::Sum | EmbeddingBagMode::Mean => {
                    // Count of non-padding entries; the mean divides by this,
                    // mirroring the bag_size decrement at EmbeddingBag.cpp:151-156.
                    let mut count: usize = 0;
                    let out_start = b * dim;
                    for &idx in &indices[start..end] {
                        // padding_idx entries are excluded from the reduction
                        // (EmbeddingBag.cpp:147 `if (idx != padding_idx)`).
                        if self.padding_idx == Some(idx) {
                            continue;
                        }
                        let row_start = idx * dim;
                        for d in 0..dim {
                            output[out_start + d] += weight_data[row_start + d];
                        }
                        count += 1;
                    }
                    if self.mode == EmbeddingBagMode::Mean && count > 0 {
                        let scale = T::from(count).unwrap();
                        for d in 0..dim {
                            output[out_start + d] = output[out_start + d] / scale;
                        }
                    }
                }
                EmbeddingBagMode::Max => {
                    let out_start = b * dim;
                    // Initialize with -inf; an all-padding (or empty) bag stays
                    // at zero (torch leaves max-mode empty bags at zero too).
                    let mut any = false;
                    for d in 0..dim {
                        output[out_start + d] = T::neg_infinity();
                    }
                    for &idx in &indices[start..end] {
                        if self.padding_idx == Some(idx) {
                            continue;
                        }
                        any = true;
                        let row_start = idx * dim;
                        for d in 0..dim {
                            let val = weight_data[row_start + d];
                            if val > output[out_start + d] {
                                output[out_start + d] = val;
                            }
                        }
                    }
                    if !any {
                        for d in 0..dim {
                            output[out_start + d] = <T as num_traits::Zero>::zero();
                        }
                    }
                }
            }
        }

        Tensor::from_storage(TensorStorage::cpu(output), vec![num_bags, dim], false)
    }

    /// Number of embeddings in the table.
    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    /// Dimension of each embedding vector.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// The reduction mode.
    pub fn mode(&self) -> EmbeddingBagMode {
        self.mode
    }

    /// The `padding_idx`, if set. Indices equal to it are excluded from each
    /// bag's reduction. Mirrors `torch.nn.EmbeddingBag.padding_idx`.
    pub fn padding_idx(&self) -> Option<usize> {
        self.padding_idx
    }
}

impl<T: Float> Module<T> for EmbeddingBag<T> {
    /// Forward pass using the input as both indices and offsets.
    ///
    /// If input is 2-D [num_bags, bag_size], each row is a bag.
    /// If input is 1-D, treats the entire input as a single bag.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() == 2 {
            // 2D input: [num_bags, bag_size] — each row is a fixed-length bag.
            // torch forces include_last_offset=False for 2D input
            // (functional.py:2735); we build offsets in whichever convention
            // `forward_bag` will read, so a configured `include_last_offset`
            // flag stays consistent here too.
            let shape = input.shape();
            let num_bags = shape[0];
            let bag_size = shape[1];
            let mut offsets: Vec<usize> = (0..num_bags).map(|b| b * bag_size).collect();
            if self.include_last_offset {
                // CSR layout: trailing entry is the total index count.
                offsets.push(num_bags * bag_size);
            }
            let flat = input.view_reshape(vec![num_bags * bag_size])?;
            self.forward_bag(&flat, &offsets)
        } else if input.ndim() == 1 {
            // 1D input: single bag. With include_last_offset the CSR boundary
            // is [0, total]; otherwise a single [0] start offset.
            if self.include_last_offset {
                let total = input.shape()[0];
                self.forward_bag(input, &[0, total])
            } else {
                self.forward_bag(input, &[0])
            }
        } else {
            Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "EmbeddingBag input must be 1-D or 2-D, got {:?}",
                    input.shape()
                ),
            })
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![&mut self.weight]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![("weight".to_string(), &self.weight)]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::autograd::graph::backward;
    use ferrotorch_core::storage::TensorStorage;

    /// Helper: create a 1-D tensor of float indices.
    fn index_tensor(indices: &[f32]) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(indices.to_vec()),
            vec![indices.len()],
            false,
        )
        .unwrap()
    }

    // --- Forward tests ---

    #[test]
    fn test_forward_shape() {
        let emb = Embedding::<f32>::new(10, 4, None).unwrap();
        let indices = index_tensor(&[0.0, 3.0, 7.0]);
        let output = emb.forward(&indices).unwrap();
        assert_eq!(output.shape(), &[3, 4]);
    }

    #[test]
    fn test_forward_correct_values() {
        // Build an embedding with known weights.
        let weight_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let weight =
            Tensor::from_storage(TensorStorage::cpu(weight_data), vec![4, 3], true).unwrap();
        let emb = Embedding::from_pretrained(weight, None).unwrap();

        // Look up rows 2 and 0.
        let indices = index_tensor(&[2.0, 0.0]);
        let output = emb.forward(&indices).unwrap();
        let data = output.data().unwrap();

        // Row 2 = [6, 7, 8], Row 0 = [0, 1, 2]
        assert_eq!(data.len(), 6);
        assert!((data[0] - 6.0).abs() < 1e-6);
        assert!((data[1] - 7.0).abs() < 1e-6);
        assert!((data[2] - 8.0).abs() < 1e-6);
        assert!((data[3] - 0.0).abs() < 1e-6);
        assert!((data[4] - 1.0).abs() < 1e-6);
        assert!((data[5] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_forward_single_index() {
        let emb = Embedding::<f32>::new(5, 8, None).unwrap();
        let indices = index_tensor(&[3.0]);
        let output = emb.forward(&indices).unwrap();
        assert_eq!(output.shape(), &[1, 8]);
    }

    // --- Padding index tests ---

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_padding_idx_zeros() {
        let emb = Embedding::<f32>::new(5, 3, Some(2)).unwrap();

        // The padding row in the weight should be zero.
        let w_data = emb.weight.data().unwrap();
        let pad_start = 2 * 3;
        for i in 0..3 {
            assert!(
                (w_data[pad_start + i] - 0.0).abs() < 1e-6,
                "padding row weight[2][{i}] should be 0, got {}",
                w_data[pad_start + i]
            );
        }

        // Forward with the padding index should return zeros.
        let indices = index_tensor(&[2.0]);
        let output = emb.forward(&indices).unwrap();
        let data = output.data().unwrap();
        for i in 0..3 {
            assert!(
                (data[i] - 0.0).abs() < 1e-6,
                "padding output[0][{i}] should be 0, got {}",
                data[i]
            );
        }
    }

    #[test]
    fn test_padding_idx_mixed() {
        // Build known weights, set padding_idx=1.
        let weight_data: Vec<f32> = vec![
            1.0, 2.0, // row 0
            0.0, 0.0, // row 1 (padding — will be zeroed)
            5.0, 6.0, // row 2
        ];
        let weight =
            Tensor::from_storage(TensorStorage::cpu(weight_data), vec![3, 2], true).unwrap();
        let emb = Embedding::from_pretrained(weight, Some(1)).unwrap();

        let indices = index_tensor(&[0.0, 1.0, 2.0]);
        let output = emb.forward(&indices).unwrap();
        let data = output.data().unwrap();

        // Row 0: [1, 2]
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        // Row 1 (padding): [0, 0]
        assert!((data[2] - 0.0).abs() < 1e-6);
        assert!((data[3] - 0.0).abs() < 1e-6);
        // Row 2: [5, 6]
        assert!((data[4] - 5.0).abs() < 1e-6);
        assert!((data[5] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_padding_idx_out_of_range() {
        let result = Embedding::<f32>::new(5, 3, Some(10));
        assert!(result.is_err());
    }

    // --- Out-of-bounds error ---

    #[test]
    fn test_out_of_bounds_error() {
        let emb = Embedding::<f32>::new(5, 3, None).unwrap();
        let indices = index_tensor(&[0.0, 5.0]); // 5 is out of bounds for num_embeddings=5
        let result = emb.forward(&indices);
        assert!(result.is_err());
    }

    #[test]
    fn test_negative_index_error() {
        let emb = Embedding::<f32>::new(5, 3, None).unwrap();
        let indices = index_tensor(&[-1.0]); // Negative cannot convert to usize
        let result = emb.forward(&indices);
        assert!(result.is_err());
    }

    // --- N-D index input (matches upstream F.embedding) ---

    #[test]
    fn test_2d_index_input_shape() {
        // Upstream `embedding_symint` (aten/src/ATen/native/Embedding.cpp:48-53)
        // accepts ANY index shape and returns `(*indices.sizes(), embedding_dim)`.
        // A [2,2] index against a [5,3] weight => output shape [2,2,3].
        let emb = Embedding::<f32>::new(5, 3, None).unwrap();
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![0.0f32, 1.0, 2.0, 3.0]),
            vec![2, 2],
            false,
        )
        .unwrap();
        let output = emb.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 2, 3]);
    }

    // --- Backward tests ---

    #[test]
    fn test_backward_simple() {
        // weight shape [3, 2], look up indices [0, 2]
        // output shape [2, 2]
        // grad_output = [[1, 1], [1, 1]]
        // grad_weight = [[1, 1], [0, 0], [1, 1]]
        let weight_data: Vec<f32> = vec![
            10.0, 20.0, // row 0
            30.0, 40.0, // row 1
            50.0, 60.0, // row 2
        ];
        let weight =
            Tensor::from_storage(TensorStorage::cpu(weight_data), vec![3, 2], true).unwrap();
        let emb = Embedding::from_pretrained(weight, None).unwrap();

        let indices = index_tensor(&[0.0, 2.0]);
        let output = emb.forward(&indices).unwrap();

        assert!(output.requires_grad());
        assert_eq!(output.grad_fn().unwrap().name(), "EmbeddingBackward");

        // Manually call backward on the grad_fn.
        let grad_output =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0f32; 4]), vec![2, 2], false).unwrap();

        let grad_fn = output.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        let grad_weight = grads[0].as_ref().unwrap();
        assert_eq!(grad_weight.shape(), &[3, 2]);
        let gd = grad_weight.data().unwrap();

        // Row 0: accessed once -> [1, 1]
        assert!((gd[0] - 1.0).abs() < 1e-6);
        assert!((gd[1] - 1.0).abs() < 1e-6);
        // Row 1: not accessed -> [0, 0]
        assert!((gd[2] - 0.0).abs() < 1e-6);
        assert!((gd[3] - 0.0).abs() < 1e-6);
        // Row 2: accessed once -> [1, 1]
        assert!((gd[4] - 1.0).abs() < 1e-6);
        assert!((gd[5] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_backward_duplicate_indices() {
        // weight shape [3, 2], look up indices [1, 1, 0, 1]
        // output shape [4, 2]
        //
        // grad_output = [[1, 2], [3, 4], [5, 6], [7, 8]]
        //
        // grad_weight[0] = grad_output[2] = [5, 6]       (index 0 appears once, at position 2)
        // grad_weight[1] = grad_output[0] + grad_output[1] + grad_output[3]
        //                = [1, 2] + [3, 4] + [7, 8] = [11, 14]
        // grad_weight[2] = [0, 0]                          (index 2 never accessed)
        let weight_data: Vec<f32> = vec![
            10.0, 20.0, // row 0
            30.0, 40.0, // row 1
            50.0, 60.0, // row 2
        ];
        let weight =
            Tensor::from_storage(TensorStorage::cpu(weight_data), vec![3, 2], true).unwrap();
        let emb = Embedding::from_pretrained(weight, None).unwrap();

        let indices = index_tensor(&[1.0, 1.0, 0.0, 1.0]);
        let output = emb.forward(&indices).unwrap();

        let grad_output = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            vec![4, 2],
            false,
        )
        .unwrap();

        let grad_fn = output.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        let grad_weight = grads[0].as_ref().unwrap();
        let gd = grad_weight.data().unwrap();

        // Row 0: [5, 6]
        assert!((gd[0] - 5.0).abs() < 1e-6, "gd[0] = {}, expected 5", gd[0]);
        assert!((gd[1] - 6.0).abs() < 1e-6, "gd[1] = {}, expected 6", gd[1]);
        // Row 1: [1+3+7, 2+4+8] = [11, 14]
        assert!(
            (gd[2] - 11.0).abs() < 1e-6,
            "gd[2] = {}, expected 11",
            gd[2]
        );
        assert!(
            (gd[3] - 14.0).abs() < 1e-6,
            "gd[3] = {}, expected 14",
            gd[3]
        );
        // Row 2: [0, 0]
        assert!((gd[4] - 0.0).abs() < 1e-6, "gd[4] = {}, expected 0", gd[4]);
        assert!((gd[5] - 0.0).abs() < 1e-6, "gd[5] = {}, expected 0", gd[5]);
    }

    #[test]
    fn test_backward_padding_idx_zeroed() {
        // Even if padding_idx is accessed, its gradient should be zero.
        let weight_data: Vec<f32> = vec![
            1.0, 2.0, // row 0
            0.0, 0.0, // row 1 (padding)
            5.0, 6.0, // row 2
        ];
        let weight =
            Tensor::from_storage(TensorStorage::cpu(weight_data), vec![3, 2], true).unwrap();
        let emb = Embedding::from_pretrained(weight, Some(1)).unwrap();

        let indices = index_tensor(&[0.0, 1.0, 2.0]);
        let output = emb.forward(&indices).unwrap();

        let grad_output =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0f32; 6]), vec![3, 2], false).unwrap();

        let grad_fn = output.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        let grad_weight = grads[0].as_ref().unwrap();
        let gd = grad_weight.data().unwrap();

        // Row 0: [1, 1]
        assert!((gd[0] - 1.0).abs() < 1e-6);
        assert!((gd[1] - 1.0).abs() < 1e-6);
        // Row 1 (padding): must be [0, 0] even though it was accessed
        assert!((gd[2] - 0.0).abs() < 1e-6, "padding grad[1][0] should be 0");
        assert!((gd[3] - 0.0).abs() < 1e-6, "padding grad[1][1] should be 0");
        // Row 2: [1, 1]
        assert!((gd[4] - 1.0).abs() < 1e-6);
        assert!((gd[5] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_backward_end_to_end() {
        // End-to-end test: use the autograd engine to verify gradients
        // flow all the way to the weight parameter.
        let weight_data: Vec<f32> = vec![
            1.0, 2.0, // row 0
            3.0, 4.0, // row 1
            5.0, 6.0, // row 2
        ];
        let weight =
            Tensor::from_storage(TensorStorage::cpu(weight_data), vec![3, 2], true).unwrap();
        let emb = Embedding::from_pretrained(weight, None).unwrap();

        let indices = index_tensor(&[1.0, 0.0]);
        let output = emb.forward(&indices).unwrap();
        // output = [[3, 4], [1, 2]], shape [2, 2]

        // Sum all elements to get a scalar for backward.
        let out_data = output.data().unwrap();
        let total: f32 = out_data.iter().sum();

        // Build a SumBackward that broadcasts scalar grad to output shape.
        #[derive(Debug)]
        struct SumBackward<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackward<T> {
            fn backward(
                &self,
                grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let go_val = grad_output.data()?[0];
                let grad = vec![go_val; self.input.numel()];
                let t = Tensor::from_storage(
                    TensorStorage::cpu(grad),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(t)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(SumBackward {
                input: output.clone(),
            }),
        )
        .unwrap();

        backward(&loss).unwrap();

        // The weight should now have a gradient.
        let grad = emb.weight.tensor().grad().unwrap().unwrap();
        let gd = grad.data().unwrap();
        assert_eq!(gd.len(), 6);

        // Row 0 accessed once (position 1): grad = [1, 1]
        assert!((gd[0] - 1.0).abs() < 1e-6, "grad[0][0] = {}", gd[0]);
        assert!((gd[1] - 1.0).abs() < 1e-6, "grad[0][1] = {}", gd[1]);
        // Row 1 accessed once (position 0): grad = [1, 1]
        assert!((gd[2] - 1.0).abs() < 1e-6, "grad[1][0] = {}", gd[2]);
        assert!((gd[3] - 1.0).abs() < 1e-6, "grad[1][1] = {}", gd[3]);
        // Row 2 not accessed: grad = [0, 0]
        assert!((gd[4] - 0.0).abs() < 1e-6, "grad[2][0] = {}", gd[4]);
        assert!((gd[5] - 0.0).abs() < 1e-6, "grad[2][1] = {}", gd[5]);
    }

    // --- Module trait tests ---

    #[test]
    fn test_module_parameters() {
        let emb = Embedding::<f32>::new(10, 4, None).unwrap();
        assert_eq!(emb.parameters().len(), 1);
        assert_eq!(emb.parameters()[0].shape(), &[10, 4]);
    }

    #[test]
    fn test_module_named_parameters() {
        let emb = Embedding::<f32>::new(5, 3, None).unwrap();
        let named = emb.named_parameters();
        assert_eq!(named.len(), 1);
        assert_eq!(named[0].0, "weight");
    }

    #[test]
    fn test_module_train_eval() {
        let mut emb = Embedding::<f32>::new(5, 3, None).unwrap();
        assert!(emb.is_training());
        emb.eval();
        assert!(!emb.is_training());
        emb.train();
        assert!(emb.is_training());
    }

    #[test]
    fn test_embedding_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Embedding<f32>>();
        assert_send_sync::<Embedding<f64>>();
    }

    #[test]
    fn test_f64_embedding() {
        let emb = Embedding::<f64>::new(5, 3, None).unwrap();
        let indices =
            Tensor::from_storage(TensorStorage::cpu(vec![0.0f64, 2.0, 4.0]), vec![3], false)
                .unwrap();
        let output = emb.forward(&indices).unwrap();
        assert_eq!(output.shape(), &[3, 3]);
    }

    // -------------------------------------------------------------------
    // SparseGrad integration (#623)
    // -------------------------------------------------------------------

    #[test]
    fn sparse_grad_returns_none_when_sparse_off() {
        let emb = Embedding::<f32>::new(8, 4, None).unwrap();
        // Default ctor leaves sparse off.
        assert!(!emb.sparse);
        let idx =
            Tensor::from_storage(TensorStorage::cpu(vec![0.0f32, 1.0]), vec![2], false).unwrap();
        let _ = emb.forward(&idx).unwrap();
        assert!(emb.sparse_grad().unwrap().is_none());
    }

    #[test]
    fn sparse_grad_returns_none_before_first_forward() {
        let emb = Embedding::<f32>::new(8, 4, None).unwrap().with_sparse(true);
        // No forward run yet -> no last_indices recorded.
        assert!(emb.sparse_grad().unwrap().is_none());
    }

    #[test]
    fn sparse_grad_emits_only_touched_rows() {
        // Vocabulary 8, dim 4. Touch only rows 1, 3, 5.
        let emb = Embedding::<f32>::new(8, 4, None).unwrap().with_sparse(true);
        let idx = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 3.0, 5.0, 1.0]),
            vec![4],
            false,
        )
        .unwrap();
        let _out = emb.forward(&idx).unwrap();

        // Manually attach a synthetic dense gradient to weight, simulating
        // post-backward state. The gradient has known per-row values so we
        // can verify slab extraction.
        let grad_data: Vec<f32> = (0..8 * 4).map(|i| i as f32).collect();
        let grad_tensor =
            Tensor::from_storage(TensorStorage::cpu(grad_data), vec![8, 4], false).unwrap();
        emb.weight.tensor().set_grad(Some(grad_tensor)).unwrap();

        let sg = emb.sparse_grad().unwrap().expect("sparse grad");
        // Touched rows are deduped + sorted: {1, 3, 5}.
        assert_eq!(sg.indices(), &[1, 3, 5]);
        assert_eq!(sg.slab_shape(), &[4]);
        // Row 1 of grad: indices 4..8 -> values [4,5,6,7]
        // Row 3 -> [12,13,14,15]
        // Row 5 -> [20,21,22,23]
        assert_eq!(
            sg.values(),
            &[
                4.0, 5.0, 6.0, 7.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0, 22.0, 23.0
            ]
        );
    }

    #[test]
    fn sparse_grad_apply_sgd_updates_only_touched_rows() {
        // End-to-end: forward → set synthetic grad → sparse_grad → apply_sgd.
        // Verifies that untouched rows stay at their original values.
        let mut emb = Embedding::<f32>::new(4, 2, None).unwrap().with_sparse(true);
        // Pin weight to known values for a tractable assertion.
        let init: Vec<f32> = (0..4 * 2).map(|i| i as f32 * 10.0).collect();
        emb.weight = Parameter::new(
            Tensor::from_storage(TensorStorage::cpu(init.clone()), vec![4, 2], true).unwrap(),
        );

        let idx =
            Tensor::from_storage(TensorStorage::cpu(vec![0.0f32, 2.0]), vec![2], false).unwrap();
        let _ = emb.forward(&idx).unwrap();

        // Synthetic gradient: each row is its index repeated.
        let grad_vec: Vec<f32> = (0..4_usize)
            .flat_map(|r| vec![r as f32, r as f32])
            .collect();
        let grad_tensor =
            Tensor::from_storage(TensorStorage::cpu(grad_vec), vec![4, 2], false).unwrap();
        emb.weight.tensor().set_grad(Some(grad_tensor)).unwrap();

        let sg = emb.sparse_grad().unwrap().unwrap();
        let mut weight = emb.weight.tensor().clone();
        sg.apply_sgd(&mut weight, 0.5_f32).unwrap();

        // init pattern is `i*10` row-major over [4, 2] → rows
        //   r0=[0, 10], r1=[20, 30], r2=[40, 50], r3=[60, 70].
        // Touched rows: {0, 2} (deduped). Synthetic per-row grad slabs:
        //   r0=[0,0], r1=[1,1], r2=[2,2], r3=[3,3].
        // SparseGrad pulls only touched rows -> {0: [0,0], 2: [2,2]}.
        // apply_sgd(lr=0.5):
        //   r0 -= 0.5 * [0, 0]  → [0, 10]      (no change, grad zero)
        //   r1                  → [20, 30]     (untouched, no update)
        //   r2 -= 0.5 * [2, 2]  → [40-1, 50-1] = [39, 49]
        //   r3                  → [60, 70]     (untouched)
        let updated = weight.data().unwrap().to_vec();
        assert_eq!(updated, vec![0.0, 10.0, 20.0, 30.0, 39.0, 49.0, 60.0, 70.0]);
    }

    // -------------------------------------------------------------------
    // #1445 — max_norm persisted in-place weight renorm (Embedding)
    // -------------------------------------------------------------------
    //
    // Oracle (live torch 2.11.0):
    //   W = [[3,4],[0,0.5],[6,8],[1,1]]
    //   F.embedding(torch.tensor([0,2]), w, max_norm=5.0, norm_type=2.0)
    //   -> w mutated to [[3,4],[0,0.5],[3,4],[1,1]]
    //   row0 norm == 5.0 == max_norm (NOT > so untouched)
    //   row2 norm == 10 > 5 -> scale 5/(10+1e-7) ≈ 0.5 -> [3,4]
    //   2nd forward leaves w unchanged (rows now satisfy max_norm).
    //   See torch/nn/functional.py:2561-2573 (renorm before gather),
    //   aten/src/ATen/native/Embedding.cpp:181-212 (embedding_renorm_cpu_).

    fn pretrained_embedding(rows: &[[f32; 2]]) -> Embedding<f32> {
        let mut data = Vec::with_capacity(rows.len() * 2);
        for r in rows {
            data.extend_from_slice(r);
        }
        let w = Tensor::from_storage(TensorStorage::cpu(data), vec![rows.len(), 2], true).unwrap();
        Embedding::from_pretrained(w, None).unwrap()
    }

    #[test]
    fn test_max_norm_mutates_persisted_weight() {
        let emb = pretrained_embedding(&[[3.0, 4.0], [0.0, 0.5], [6.0, 8.0], [1.0, 1.0]])
            .with_max_norm(5.0)
            .with_norm_type(2.0);

        // Look up rows 0 and 2.
        let idx = index_tensor(&[0.0, 2.0]);
        let out = emb.forward(&idx).unwrap();
        let od = out.data().unwrap();
        // Row 0 untouched ([3,4], norm exactly 5); row 2 clipped to [3,4].
        assert!((od[0] - 3.0).abs() < 1e-4, "out r0[0]={}", od[0]);
        assert!((od[1] - 4.0).abs() < 1e-4, "out r0[1]={}", od[1]);
        assert!((od[2] - 3.0).abs() < 1e-4, "out r2[0]={}", od[2]);
        assert!((od[3] - 4.0).abs() < 1e-4, "out r2[1]={}", od[3]);

        // The PERSISTED weight must be mutated in place, not just the output.
        let w_after = emb.weight.data().unwrap().to_vec();
        // [[3,4],[0,0.5],[3,4],[1,1]] — only the touched, over-norm row 2 moved.
        assert!((w_after[0] - 3.0).abs() < 1e-4); // row0 untouched
        assert!((w_after[1] - 4.0).abs() < 1e-4);
        assert!((w_after[2] - 0.0).abs() < 1e-4); // row1 not indexed, untouched
        assert!((w_after[3] - 0.5).abs() < 1e-4);
        assert!(
            (w_after[4] - 3.0).abs() < 1e-4,
            "row2[0] persisted={}",
            w_after[4]
        );
        assert!(
            (w_after[5] - 4.0).abs() < 1e-4,
            "row2[1] persisted={}",
            w_after[5]
        );
        assert!((w_after[6] - 1.0).abs() < 1e-4); // row3 not indexed
        assert!((w_after[7] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_max_norm_second_forward_is_stable() {
        // Forward twice: the first call clips the over-norm row in place; the
        // second call sees the already-clipped weight and is a no-op on it.
        let emb = pretrained_embedding(&[[3.0, 4.0], [0.0, 0.5], [6.0, 8.0], [1.0, 1.0]])
            .with_max_norm(5.0)
            .with_norm_type(2.0);
        let idx = index_tensor(&[0.0, 2.0]);

        let _ = emb.forward(&idx).unwrap();
        let w_after_first = emb.weight.data().unwrap().to_vec();

        let _ = emb.forward(&idx).unwrap();
        let w_after_second = emb.weight.data().unwrap().to_vec();

        // Stable: a second renorm of already-clipped rows changes nothing.
        for (a, b) in w_after_first.iter().zip(w_after_second.iter()) {
            assert!((a - b).abs() < 1e-7, "weight drifted: {a} vs {b}");
        }
        // And the clipped row really did change relative to the original [6,8].
        assert!((w_after_first[4] - 3.0).abs() < 1e-4);
        assert!((w_after_first[5] - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_max_norm_untouched_rows_not_renormed() {
        // Row 2 ([6,8], norm 10) exceeds max_norm but is NOT indexed this
        // forward; only row 0 is looked up. The persisted weight's row 2 must
        // stay at its original over-norm value (renorm visits only touched
        // rows — Embedding.cpp:198-202).
        let emb = pretrained_embedding(&[[3.0, 4.0], [0.0, 0.5], [6.0, 8.0], [1.0, 1.0]])
            .with_max_norm(5.0);
        let idx = index_tensor(&[0.0]);
        let _ = emb.forward(&idx).unwrap();
        let w = emb.weight.data().unwrap().to_vec();
        assert!(
            (w[4] - 6.0).abs() < 1e-6,
            "row2 should be untouched: {}",
            w[4]
        );
        assert!(
            (w[5] - 8.0).abs() < 1e-6,
            "row2 should be untouched: {}",
            w[5]
        );
    }

    // -------------------------------------------------------------------
    // #1445 — scale_grad_by_freq divides duplicate-index grad rows
    // -------------------------------------------------------------------
    //
    // Oracle (live torch 2.11.0): indices [1,1,0], grad_output ones[3,2].
    //   scale_grad_by_freq=True  -> grad rows: r0=[1,1], r1=[1,1], r2=[0,0]
    //   scale_grad_by_freq=False -> grad rows: r0=[1,1], r1=[2,2], r2=[0,0]
    //   torch/nn/functional.py:2499-2500 + aten embedding_dense_backward.

    #[test]
    fn test_scale_grad_by_freq_divides_duplicates() {
        let weight =
            Tensor::from_storage(TensorStorage::cpu(vec![0.0f32; 6]), vec![3, 2], true).unwrap();
        let emb = Embedding::from_pretrained(weight, None)
            .unwrap()
            .with_scale_grad_by_freq(true);

        // Index 1 appears twice, index 0 once.
        let idx = index_tensor(&[1.0, 1.0, 0.0]);
        let out = emb.forward(&idx).unwrap();

        let grad_output =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0f32; 6]), vec![3, 2], false).unwrap();
        let grads = out.grad_fn().unwrap().backward(&grad_output).unwrap();
        let gd = grads[0].as_ref().unwrap().data().unwrap();

        // Row 0 (1 occurrence): [1,1]; row 1 (2 occurrences /2): [1,1]; row 2: [0,0].
        assert!((gd[0] - 1.0).abs() < 1e-6, "r0[0]={}", gd[0]);
        assert!((gd[1] - 1.0).abs() < 1e-6);
        assert!(
            (gd[2] - 1.0).abs() < 1e-6,
            "r1[0]={} (should be 1, scaled)",
            gd[2]
        );
        assert!((gd[3] - 1.0).abs() < 1e-6);
        assert!((gd[4] - 0.0).abs() < 1e-6);
        assert!((gd[5] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_grad_by_freq_off_accumulates() {
        // Same indices, flag OFF: row 1's grad accumulates to [2,2].
        let weight =
            Tensor::from_storage(TensorStorage::cpu(vec![0.0f32; 6]), vec![3, 2], true).unwrap();
        let emb = Embedding::from_pretrained(weight, None).unwrap();
        let idx = index_tensor(&[1.0, 1.0, 0.0]);
        let out = emb.forward(&idx).unwrap();
        let grad_output =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0f32; 6]), vec![3, 2], false).unwrap();
        let grads = out.grad_fn().unwrap().backward(&grad_output).unwrap();
        let gd = grads[0].as_ref().unwrap().data().unwrap();
        assert!(
            (gd[2] - 2.0).abs() < 1e-6,
            "r1[0]={} (should be 2, unscaled)",
            gd[2]
        );
        assert!((gd[3] - 2.0).abs() < 1e-6);
    }

    // -------------------------------------------------------------------
    // #1445 — EmbeddingBag kwargs (max_norm / padding_idx / include_last_offset)
    // -------------------------------------------------------------------

    fn pretrained_bag(rows: &[Vec<f32>], mode: EmbeddingBagMode) -> EmbeddingBag<f32> {
        let dim = rows[0].len();
        let mut data = Vec::new();
        for r in rows {
            data.extend_from_slice(r);
        }
        let mut bag = EmbeddingBag::<f32>::new(rows.len(), dim, mode).unwrap();
        bag.weight = Parameter::new(
            Tensor::from_storage(TensorStorage::cpu(data), vec![rows.len(), dim], true).unwrap(),
        );
        bag
    }

    #[test]
    fn test_bag_modes_match_torch() {
        // Oracle (torch 2.11.0): W=[[1,2,3],[4,5,6],[7,8,9],[10,11,12]],
        // input [0,1,2,3], offsets [0,2] -> bag0=rows{0,1}, bag1=rows{2,3}.
        //   sum:  [[5,7,9],[17,19,21]]
        //   mean: [[2.5,3.5,4.5],[8.5,9.5,10.5]]
        //   max:  [[4,5,6],[10,11,12]]
        let rows = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![10.0, 11.0, 12.0],
        ];
        let inp = index_tensor(&[0.0, 1.0, 2.0, 3.0]);
        let offs = [0usize, 2];

        let sum = pretrained_bag(&rows, EmbeddingBagMode::Sum)
            .forward_bag(&inp, &offs)
            .unwrap();
        assert_eq!(sum.data().unwrap(), &[5.0, 7.0, 9.0, 17.0, 19.0, 21.0]);

        let mean = pretrained_bag(&rows, EmbeddingBagMode::Mean)
            .forward_bag(&inp, &offs)
            .unwrap();
        assert_eq!(mean.data().unwrap(), &[2.5, 3.5, 4.5, 8.5, 9.5, 10.5]);

        let max = pretrained_bag(&rows, EmbeddingBagMode::Max)
            .forward_bag(&inp, &offs)
            .unwrap();
        assert_eq!(max.data().unwrap(), &[4.0, 5.0, 6.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_bag_max_norm_mutates_weight() {
        // Oracle (torch 2.11.0): W=[[1,2,3],[4,5,6],[7,8,9],[10,11,12]],
        // input [0,1,2,3], offsets [0,2], mode=sum, max_norm=5.0.
        // row0 norm sqrt(14)≈3.74 < 5 untouched; rows 1,2,3 over-norm scaled.
        // Persisted weight row1 -> ~[2.279212, 2.849014, 3.418817].
        let rows = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![10.0, 11.0, 12.0],
        ];
        let bag = pretrained_bag(&rows, EmbeddingBagMode::Sum)
            .with_max_norm(5.0)
            .with_norm_type(2.0);
        let inp = index_tensor(&[0.0, 1.0, 2.0, 3.0]);
        let offs = [0usize, 2];
        let out = bag.forward_bag(&inp, &offs).unwrap();
        // bag0 = renormed row0 + renormed row1 = [1,2,3] + [2.279212,2.849014,3.418817]
        let od = out.data().unwrap();
        assert!((od[0] - 3.279212).abs() < 1e-4, "bag0[0]={}", od[0]);
        assert!((od[1] - 4.849014).abs() < 1e-4, "bag0[1]={}", od[1]);
        assert!((od[2] - 6.418818).abs() < 1e-4, "bag0[2]={}", od[2]);

        // Persisted weight row0 untouched (under norm), row1 renormed.
        let w = bag.weight.data().unwrap().to_vec();
        assert!((w[0] - 1.0).abs() < 1e-6, "row0 untouched");
        assert!(
            (w[3] - 2.279212).abs() < 1e-4,
            "row1 persisted renorm: {}",
            w[3]
        );
        assert!((w[4] - 2.849014).abs() < 1e-4);
        assert!((w[5] - 3.418817).abs() < 1e-4);
    }

    #[test]
    fn test_bag_padding_idx_excluded_from_reduction() {
        // Oracle (torch 2.11.0): W=[[1,1],[2,2],[4,4],[8,8]], padding_idx=1,
        // single bag input [0,1,2]. idx 1 is padding -> excluded.
        //   mean: ([1,1]+[4,4])/2 = [2.5,2.5]   (divides by non-pad count 2)
        //   sum:  [1,1]+[4,4]      = [5,5]
        let rows = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![4.0, 4.0],
            vec![8.0, 8.0],
        ];
        let inp = index_tensor(&[0.0, 1.0, 2.0]);
        let offs = [0usize];

        let mut mean = pretrained_bag(&rows, EmbeddingBagMode::Mean);
        mean.padding_idx = Some(1);
        let mo = mean.forward_bag(&inp, &offs).unwrap();
        assert_eq!(mo.data().unwrap(), &[2.5, 2.5]);

        let mut sum = pretrained_bag(&rows, EmbeddingBagMode::Sum);
        sum.padding_idx = Some(1);
        let so = sum.forward_bag(&inp, &offs).unwrap();
        assert_eq!(so.data().unwrap(), &[5.0, 5.0]);
    }

    #[test]
    fn test_bag_include_last_offset() {
        // Oracle (torch 2.11.0): W=[[1,2],[3,4],[5,6],[7,8]], input [0,1,2,3],
        // offsets [0,2,4], include_last_offset=True, mode=sum.
        //   bag0 = row0+row1 = [4,6]; bag1 = row2+row3 = [12,14].
        let rows = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let bag = pretrained_bag(&rows, EmbeddingBagMode::Sum).with_include_last_offset(true);
        let inp = index_tensor(&[0.0, 1.0, 2.0, 3.0]);
        let offs = [0usize, 2, 4];
        let out = bag.forward_bag(&inp, &offs).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.data().unwrap(), &[4.0, 6.0, 12.0, 14.0]);
    }

    #[test]
    fn test_bag_max_mode_rejects_sparse_and_scale_grad() {
        let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let inp = index_tensor(&[0.0, 1.0]);
        let offs = [0usize];

        let scaled = pretrained_bag(&rows, EmbeddingBagMode::Max).with_scale_grad_by_freq(true);
        assert!(scaled.forward_bag(&inp, &offs).is_err());

        let sparse = pretrained_bag(&rows, EmbeddingBagMode::Max).with_sparse(true);
        assert!(sparse.forward_bag(&inp, &offs).is_err());
    }

    #[test]
    fn test_bag_padding_idx_validated_and_zeroed() {
        // padding_idx out of range rejected; in range -> that row zeroed.
        assert!(EmbeddingBag::<f32>::new_with(3, 2, EmbeddingBagMode::Sum, Some(5)).is_err());

        let bag = EmbeddingBag::<f32>::new_with(4, 3, EmbeddingBagMode::Sum, Some(2)).unwrap();
        let w = bag.weight.data().unwrap();
        let pad_start = 2 * 3;
        for i in 0..3 {
            assert!(
                w[pad_start + i].abs() < 1e-6,
                "padding row not zeroed at {i}: {}",
                w[pad_start + i]
            );
        }
        assert_eq!(bag.padding_idx(), Some(2));
    }
}
