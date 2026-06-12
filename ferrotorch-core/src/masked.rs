//! Masked tensors — `torch.masked.MaskedTensor` analog.
//!
//! A [`MaskedTensor`] pairs a data tensor with a boolean mask, where mask
//! entries indicate which positions are "valid". Reductions, arithmetic,
//! and `to_tensor` / `filled` all honour the mask.
//!
//! # Mask convention
//!
//! Matches `torch.masked.MaskedTensor`: `mask[i] == true` means the value
//! is valid (use it); `mask[i] == false` means the value is masked out
//! (ignored by reductions, replaced by `fill_value` when materialised).
//! This is the **opposite** of NumPy's `numpy.ma`, which uses
//! `mask=True` to mean "invalid". Helpers below translate at the
//! boundary when delegating to [`ferray_ma`].
//!
//! # Device contract (CORE-065 → #1759)
//!
//! Every value-producing op (`filled` / `to_tensor`, `masked_sum` /
//! `masked_mean` / `masked_min` / `masked_max` / `masked_count`) returns a
//! tensor ON THE DATA TENSOR'S DEVICE — including the all-masked edge-case
//! scalars, which are uploaded rather than silently demoted to CPU
//! (torch parity, live 2.11.0+cu130 probe: `torch.masked.amax(tc,
//! mask=all_false)` → `tensor(-inf, device='cuda:0')`).
//!
//! # Autograd contract (CORE-064 → #1758)
//!
//! Every value-producing op attaches a real backward edge when the data
//! tensor tracks gradients (R-LOUD-3) — `filled` / `to_tensor` via
//! `MaskedFillBackward`, the reductions via the `Masked*Backward` nodes in
//! this module. `masked_count` is non-differentiable (constant in the data
//! values) and stays honestly `requires_grad = false`. Per-op gradient
//! contracts (including the even tie split for extrema and the zero-grad
//! all-masked edge) are quoted from live torch on each node.
//!
//! # GPU discipline
//!
//! No silent CPU↔GPU round trips. Reductions (`masked_sum` / `masked_mean` /
//! `masked_min` / `masked_max`) lower to on-device kernels for f32/f64
//! (#597 / #627); `masked_mean`'s division runs on-device (`div_f32` /
//! `div_f64` against the uploaded count scalar — the GPU sum never crosses
//! back to the host). The constructors `masked_invalid` / `masked_equal`
//! compute their boolean predicate ON-DEVICE for f32/f64 CUDA inputs via
//! `GpuBackend::isfinite_mask` / `ne_scalar_mask` (#1545); only the resulting
//! boolean mask is read back to populate the host-resident `Vec<bool>` (the
//! mask is host-side BY DESIGN — this is a one-way readback of the freshly
//! computed predicate, not a round trip of the value data, which never leaves
//! the device). The same predicate path drives `MaskedExtremumBackward`'s
//! on-device tie detection. `masked_where` takes a host `&[bool]` condition
//! and is device-agnostic. bf16/f16 lowering: `filled` / `to_tensor` run the
//! dtype-generic resident `masked_fill_dt` kernel on CUDA; the bf16/f16
//! extremum forward (and its backward tie walk) takes the documented host
//! readback (#616/#627); bf16/f16 `masked_sum` / `masked_mean` and the
//! constructors still error `NotImplementedOnCuda` / take the host walk.
//!
//! ## REQ status (per `.design/ferrotorch-core/masked.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `MaskedTensor::new` at `masked.rs:60`; consumer: re-export `ferrotorch_core::MaskedTensor` at `lib.rs:167` |
//! | REQ-2 | SHIPPED | `MaskedTensor::from_data` at `masked.rs:78`; consumer: re-export at `lib.rs:167` |
//! | REQ-3 | SHIPPED | `with_fill_value` at `masked.rs:84`; consumer: re-export at `lib.rs:167` |
//! | REQ-4 | SHIPPED | `fn filled` / `fn to_tensor` in `masked.rs` (autograd via `MaskedFillBackward`, result on the data device — #1758/#1759); consumer: re-export at `lib.rs:167` |
//! | REQ-5 | SHIPPED | `masked_sum`/`masked_mean`/`masked_min`/`masked_max`/`masked_count` in `masked.rs` (autograd via `MaskedSumBackward`/`MaskedMeanBackward`/`MaskedExtremumBackward`, results on the data device — #1758/#1759); consumer: re-export at `lib.rs:167-170` |
//! | REQ-6 | SHIPPED | `masked_where`/`masked_invalid`/`masked_equal` (`masked_invalid`/`masked_equal` in `masked.rs`); consumer: re-export at `lib.rs`. GPU predicate masks for `masked_invalid`/`masked_equal` (f32/f64) via `GpuBackend::isfinite_mask`/`ne_scalar_mask` (#1545); consumer: those constructors' CUDA branches in `masked.rs` |
//! | REQ-7 | SHIPPED | `to_ferray` at `masked.rs:165`; consumer: `to_ferray_round_trip_mean_matches_inhouse` pins the bridge |

use std::sync::Arc;

use ferray_core::{Array as FerrayArray, IxDyn as FerrayIxDyn};
use ferray_ma::masked_array::MaskedArray;

use crate::autograd::no_grad::is_grad_enabled;
use crate::bool_tensor::BoolTensor;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

// ---------------------------------------------------------------------------
// MaskedTensor
// ---------------------------------------------------------------------------

/// A tensor paired with a boolean mask.
///
/// `mask[i] == true` means the entry is **valid**; `false` means it is
/// **masked out**. This matches `torch.masked.MaskedTensor`.
///
/// `fill_value` is substituted for masked entries when [`to_tensor`] /
/// [`filled`](Self::filled) is called. Defaults to zero.
#[derive(Clone, Debug)]
pub struct MaskedTensor<T: Float> {
    data: Tensor<T>,
    /// Length equals `data.numel()`. Stored flat in C-order to match the
    /// underlying tensor layout.
    mask: Vec<bool>,
    fill_value: T,
}

impl<T: Float> MaskedTensor<T> {
    /// Build a masked tensor from a data tensor + boolean mask.
    ///
    /// `mask` must have exactly `data.numel()` elements. Accepts both CPU
    /// and CUDA tensors — GPU paths in [`masked_sum`] / [`masked_mean`]
    /// lower to `mul + reduce_sum`. (#597)
    ///
    /// Mask convention: `mask[i] == true` means VALID (torch convention).
    pub fn new(data: Tensor<T>, mask: Vec<bool>) -> FerrotorchResult<Self> {
        if mask.len() != data.numel() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "MaskedTensor::new: mask length {} != data numel {}",
                    mask.len(),
                    data.numel()
                ),
            });
        }
        Ok(Self {
            data,
            mask,
            fill_value: <T as num_traits::Zero>::zero(),
        })
    }

    /// Build a masked tensor from data only, with all entries marked valid.
    pub fn from_data(data: Tensor<T>) -> FerrotorchResult<Self> {
        let n = data.numel();
        Self::new(data, vec![true; n])
    }

    /// Override the fill value used by [`Self::filled`] / [`Self::to_tensor`].
    pub fn with_fill_value(mut self, fill_value: T) -> Self {
        self.fill_value = fill_value;
        self
    }

    /// The underlying data tensor (regardless of mask).
    #[inline]
    pub fn data(&self) -> &Tensor<T> {
        &self.data
    }

    /// Borrow the boolean mask. `true` = valid, `false` = masked out.
    #[inline]
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }

    /// The fill value used when materialising masked entries.
    #[inline]
    pub fn fill_value(&self) -> T {
        self.fill_value
    }

    /// Logical shape (same as `data().shape()`).
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Total number of entries, masked or not.
    #[inline]
    pub fn numel(&self) -> usize {
        self.data.numel()
    }

    /// Number of entries currently marked valid.
    pub fn count_valid(&self) -> usize {
        self.mask.iter().filter(|&&v| v).count()
    }

    /// Number of entries currently masked out.
    pub fn count_masked(&self) -> usize {
        self.mask.iter().filter(|&&v| !v).count()
    }

    /// Materialise into a plain `Tensor<T>` by substituting `fill_value`
    /// at every masked-out position.
    ///
    /// **Autograd (#1758 / CORE-064):** when the data tensor tracks
    /// gradients the result carries a [`MaskedFillBackward`] edge, matching
    /// torch's `MaskedTensor.to_tensor` (live 2.11.0 probe:
    /// `grad_fn=<MaskedFillBackward0>`): the upstream gradient passes
    /// through at valid positions and is zero where the constant fill
    /// replaced the value. `fill_value` is a constant — it never receives
    /// gradient.
    ///
    /// **Device (#1759 / CORE-065):** the result lives on the data tensor's
    /// device. On CUDA the fill runs on-device via the dtype-generic
    /// resident `masked_fill_dt` kernel
    /// ([`crate::grad_fns::indexing::masked_fill_bt`]); only the
    /// host-resident boolean mask is uploaded — the value data never leaves
    /// the device. Torch parity: `masked_tensor(tc, mc).to_tensor(0.0)` on a
    /// CUDA input returns `device='cuda:0'`.
    ///
    /// [`MaskedFillBackward`]: crate::grad_fns::indexing::MaskedFillBackward
    pub fn filled(&self) -> FerrotorchResult<Tensor<T>> {
        // `masked_fill` convention: fill where its mask is TRUE → invert the
        // torch-convention valid mask.
        let fill_positions: Vec<bool> = self.mask.iter().map(|&v| !v).collect();
        if self.data.is_cuda() {
            // Resident-mask path → dtype-generic `masked_fill_dt` kernel
            // (f32/f64/bf16/f16). Result stays on the data device; the
            // autograd edge is attached inside `masked_fill_bt`.
            let mask_bt = BoolTensor::from_slice(&fill_positions, self.data.shape())?
                .to(self.data.device())?;
            return crate::grad_fns::indexing::masked_fill_bt(
                &self.data,
                &mask_bt,
                self.fill_value,
            );
        }
        // CPU: logical-order walk (handles non-contiguous views, matching the
        // pre-existing forward contract) with the same `MaskedFillBackward`
        // edge the delegated path attaches.
        let data_vec = self.data.data_vec()?;
        let out: Vec<T> = data_vec
            .iter()
            .zip(self.mask.iter())
            .map(|(&v, &valid)| if valid { v } else { self.fill_value })
            .collect();
        let shape = self.data.shape().to_vec();
        let storage = TensorStorage::cpu(out);
        if is_grad_enabled() && self.data.requires_grad() {
            let grad_fn = Arc::new(crate::grad_fns::indexing::MaskedFillBackward {
                input: self.data.clone(),
                mask: BoolTensor::from_slice(&fill_positions, &shape)?,
            });
            return Tensor::from_operation(storage, shape, grad_fn);
        }
        Tensor::from_storage(storage, shape, false)
    }

    /// Alias of [`Self::filled`] mirroring `torch.Tensor`'s naming.
    #[inline]
    pub fn to_tensor(&self) -> FerrotorchResult<Tensor<T>> {
        self.filled()
    }
}

// ---------------------------------------------------------------------------
// ferray-ma bridge
//
// ferray-ma's MaskedArray uses NumPy semantics (mask=true means INVALID).
// We invert at the boundary so internal callers see the torch convention
// (mask=true means VALID).
// ---------------------------------------------------------------------------

impl<T: Float> MaskedTensor<T> {
    /// Convert to a `ferray_ma::MaskedArray<U, IxDyn>` for delegating to
    /// ferray-ma's wider op surface (var/std, masked sort, ufunc support,
    /// etc.). Element type is generic over `U: Float + Element` because
    /// ferray-ma's bound is more restrictive than ferrotorch's `Float`
    /// trait — typical choices are `f32` or `f64`.
    ///
    /// Inverts the mask to match NumPy semantics (`true` = invalid)
    /// since ferrotorch uses the torch convention (`true` = valid).
    pub fn to_ferray<U>(&self, op: &'static str) -> FerrotorchResult<MaskedArray<U, FerrayIxDyn>>
    where
        U: ferray_core::Element + Copy + num_traits::Float + 'static,
    {
        let data_vec = self.data.data_vec()?;
        let data_u: Vec<U> = data_vec
            .into_iter()
            .map(|v| U::from(v.to_f64().unwrap()).unwrap())
            .collect();
        let arr =
            FerrayArray::<U, FerrayIxDyn>::from_vec(FerrayIxDyn::new(self.data.shape()), data_u)
                .map_err(FerrotorchError::Ferray)?;
        // Invert mask: ferrotorch true=valid → numpy true=invalid.
        let inv: Vec<bool> = self.mask.iter().map(|&v| !v).collect();
        let mask_arr =
            FerrayArray::<bool, FerrayIxDyn>::from_vec(FerrayIxDyn::new(self.data.shape()), inv)
                .map_err(FerrotorchError::Ferray)?;
        MaskedArray::new(arr, mask_arr).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("{op}: {e}"),
        })
    }
}

// ---------------------------------------------------------------------------
// Reductions (sum / mean / count)
// ---------------------------------------------------------------------------

/// Sum of valid entries; returns a 0-d tensor on the data tensor's device.
///
/// Mirrors `torch.masked.MaskedTensor.sum()` (torch.masked uses the same
/// "ignore masked, sum the rest" semantics as numpy.ma).
///
/// On GPU, lowers to `data * mask_as_float → reduce_sum` (#597). The mask
/// is uploaded once and reused for `masked_mean`'s denominator if both
/// are computed.
///
/// **Autograd (#1758 / CORE-064):** when the data tensor tracks gradients
/// the result carries a [`MaskedSumBackward`] edge. Live torch
/// 2.11.0+cu130 oracle: `torch.masked.sum(t, mask=m).backward()` →
/// `t.grad == tensor([1., 0., 1., 1.])` for `m=[T,F,T,T]` — the upstream
/// gradient is routed to valid positions, zero to masked ones.
pub fn masked_sum<T: Float>(mt: &MaskedTensor<T>) -> FerrotorchResult<Tensor<T>> {
    let result = masked_sum_forward(mt)?;
    if is_grad_enabled() && mt.data.requires_grad() {
        let grad_fn = Arc::new(MaskedSumBackward {
            input: mt.data.clone(),
            mask: mt.mask.clone(),
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        return Tensor::from_operation(storage, shape, grad_fn);
    }
    Ok(result)
}

/// Forward-only sum lowering (no autograd bookkeeping).
fn masked_sum_forward<T: Float>(mt: &MaskedTensor<T>) -> FerrotorchResult<Tensor<T>> {
    if mt.data.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        return masked_sum_gpu(mt);
    }
    if mt.data.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "masked_sum" });
    }
    // Walk the data + mask in one pass.
    let data = mt.data.data_vec()?;
    let mut acc = <T as num_traits::Zero>::zero();
    for (&v, &valid) in data.iter().zip(mt.mask.iter()) {
        if valid {
            acc += v;
        }
    }
    Tensor::from_storage(TensorStorage::cpu(vec![acc]), vec![], false)
}

/// GPU lowering: build a float-valued mask tensor, multiply, reduce-sum.
fn masked_sum_gpu<T: Float>(mt: &MaskedTensor<T>) -> FerrotorchResult<Tensor<T>> {
    let device = mt.data.device();
    let mask_t: Tensor<T> = mask_as_float_tensor(&mt.mask, mt.data.shape(), device)?;
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let numel = mt.data.numel();
    // #1658: normalise a narrowed-offset CUDA data tensor to a packed offset-0
    // buffer before the elementwise mul reads element 0. The mask tensor is
    // freshly built offset-0, so only the data side needs it.
    let data = mt.data.contiguous()?;
    let prod_h = if is_f32::<T>() {
        backend.mul_f32(data.gpu_handle()?, mask_t.gpu_handle()?)?
    } else {
        backend.mul_f64(data.gpu_handle()?, mask_t.gpu_handle()?)?
    };
    let sum_h = if is_f32::<T>() {
        backend.sum_f32(&prod_h, numel)?
    } else {
        backend.sum_f64(&prod_h, numel)?
    };
    Tensor::from_storage(TensorStorage::gpu(sum_h), vec![], false)
}

/// Build a float Tensor<T> on `device` from a bool mask, with shape
/// matching the masked-tensor data. true → 1, false → 0.
fn mask_as_float_tensor<T: Float>(
    mask: &[bool],
    shape: &[usize],
    device: crate::device::Device,
) -> FerrotorchResult<Tensor<T>> {
    let one = T::from(1.0).unwrap();
    let zero = <T as num_traits::Zero>::zero();
    let data: Vec<T> = mask.iter().map(|&b| if b { one } else { zero }).collect();
    let cpu = Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)?;
    if device.is_cuda() {
        cpu.to(device)
    } else {
        Ok(cpu)
    }
}

/// Helper: are we operating on `f32`?
#[inline]
fn is_f32<T: Float>() -> bool {
    std::mem::size_of::<T>() == 4
}

/// Helper: are we operating on `f64`?
#[inline]
fn is_f64<T: Float>() -> bool {
    std::mem::size_of::<T>() == 8
}

/// Fallible `T::from(f64)` with a structured error (R-CODE-2: no
/// production `unwrap`).
fn t_from_f64<T: Float>(v: f64, what: &str) -> FerrotorchResult<T> {
    T::from(v).ok_or_else(|| FerrotorchError::InvalidArgument {
        message: format!("{what}: value {v} not representable in the target dtype"),
    })
}

// ---------------------------------------------------------------------------
// Autograd backward nodes (#1758 / CORE-064)
//
// Every value-producing masked op attaches a real backward edge when the
// data tensor tracks gradients (R-LOUD-3: never a silently detached
// result). Gradient contracts probed from live torch 2.11.0+cu130
// (2026-06-11; quoted per-op below and in
// tests/audit_core064_masked_autograd.rs). Gradients are constructed from
// the host-resident boolean mask and uploaded once to the data device —
// a one-way upload (the mask never lives on the device), not a round trip.
// ---------------------------------------------------------------------------

/// Read the single element of a 0-d tensor (one-element D2H for CUDA —
/// the same pattern `SumBackward` / `MeanBackward` use for the upstream
/// gradient scalar).
fn scalar_of<T: Float>(t: &Tensor<T>) -> FerrotorchResult<T> {
    let v = t.data_vec()?;
    v.first()
        .copied()
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "masked backward: expected a 0-d (1-element) tensor".into(),
        })
}

/// Build the gradient tensor `out[i] = scale if select[i] else 0` shaped
/// like `input`, on `input`'s device.
fn mask_scaled_grad<T: Float>(
    input: &Tensor<T>,
    select: &[bool],
    scale: T,
) -> FerrotorchResult<Tensor<T>> {
    let zero = <T as num_traits::Zero>::zero();
    let data: Vec<T> = select
        .iter()
        .map(|&s| if s { scale } else { zero })
        .collect();
    let cpu = Tensor::from_storage(TensorStorage::cpu(data), input.shape().to_vec(), false)?;
    if input.is_cuda() {
        return cpu.to(input.device());
    }
    Ok(cpu)
}

/// Backward node for [`masked_sum`].
///
/// VJP (torch oracle: `torch.masked.sum(t, mask=[T,F,T,T]).backward()` →
/// `t.grad == tensor([1., 0., 1., 1.])`): `grad_input[i] = grad_output`
/// at valid positions, `0` at masked ones. All-masked degenerates to all
/// zeros (torch probed: `tensor([0., 0.])`).
#[derive(Debug)]
pub struct MaskedSumBackward<T: Float> {
    input: Tensor<T>,
    mask: Vec<bool>,
}

impl<T: Float> GradFn<T> for MaskedSumBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let go = scalar_of(grad_output)?;
        Ok(vec![Some(mask_scaled_grad(&self.input, &self.mask, go)?)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "MaskedSumBackward"
    }
}

/// Backward node for [`masked_mean`].
///
/// VJP (torch oracle: `(torch.masked.mean(t2, mask=m2) * 6.0).backward()`
/// with 3 valid → `t2.grad == tensor([2., 0., 2., 0., 2.])`):
/// `grad_input[i] = grad_output / count_valid` at valid positions, `0` at
/// masked ones. All-masked (`count == 0`): the forward is NaN and torch
/// routes ZERO grads to the leaf (probed: `tensor([0., 0.])`) — matched
/// here without dividing by zero.
#[derive(Debug)]
pub struct MaskedMeanBackward<T: Float> {
    input: Tensor<T>,
    mask: Vec<bool>,
    count: usize,
}

impl<T: Float> GradFn<T> for MaskedMeanBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let zero = <T as num_traits::Zero>::zero();
        if self.count == 0 {
            // All-masked: zero grads (mask is all false, so the scale is
            // never applied anyway — keep it finite for clarity).
            return Ok(vec![Some(mask_scaled_grad(&self.input, &self.mask, zero)?)]);
        }
        let go = scalar_of(grad_output)?;
        let scale = go / t_from_f64::<T>(self.count as f64, "MaskedMeanBackward: count")?;
        Ok(vec![Some(mask_scaled_grad(
            &self.input,
            &self.mask,
            scale,
        )?)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "MaskedMeanBackward"
    }
}

/// Backward node shared by [`masked_min`] / [`masked_max`].
///
/// VJP: the upstream gradient splits EVENLY among the VALID positions
/// whose value equals the saved forward result — torch's
/// `(self == result) / count` extremum rule restricted to the mask. Live
/// torch 2.11.0+cu130 tie oracle:
/// `(torch.masked.amax([5,5,1,5], mask=[T,T,T,F]) * 4).backward()` →
/// `grad == tensor([2., 2., 0., 0.])` (two valid maxima share; the
/// masked-out third 5.0 gets 0).
///
/// Edge contracts:
/// - All-masked: the forward is the #1924-pinned NaN sentinel; torch
///   routes ZERO grads to the leaf (probed: `tensor([0., 0.])`) — matched.
/// - NaN extremum payload (valid NaN data): `NaN == NaN` is false, so no
///   position matches; zero grads are routed rather than dividing 0/0
///   (torch's `scale_grad_by_count` would emit NaN here — see #1932's
///   reduction-NaN family for the forward-side divergences).
///
/// On CUDA f32/f64 the tie predicate runs on-device via
/// `GpuBackend::ne_scalar_mask` and only the boolean mask is read back
/// (the #1545 `predicate_mask_gpu` pattern — the value data never leaves
/// the device). Other CUDA dtypes take the same documented host readback
/// as their forward path ([`masked_extremum_cpu`]).
#[derive(Debug)]
pub struct MaskedExtremumBackward<T: Float> {
    input: Tensor<T>,
    mask: Vec<bool>,
    /// Saved 0-d forward result (same device as the forward output).
    result: Tensor<T>,
    pick_min: bool,
}

impl<T: Float> GradFn<T> for MaskedExtremumBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let zero = <T as num_traits::Zero>::zero();
        // All-masked: forward is the #1924-pinned NaN sentinel; torch
        // routes zero gradient to the data leaf — match it.
        if !self.mask.iter().any(|&m| m) {
            return Ok(vec![Some(mask_scaled_grad(&self.input, &self.mask, zero)?)]);
        }
        let r = scalar_of(&self.result)?;
        let ties: Vec<bool> = if self.input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            // On-device `v != r` predicate; one-way boolean readback only.
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let r_f64 = r.to_f64().ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MaskedExtremumBackward: result not representable as f64".into(),
            })?;
            // #1658: normalise a narrowed-offset CUDA view before the
            // predicate kernel reads element 0 (same as the forward paths).
            let data_c = self.input.contiguous()?;
            let ne_h = backend.ne_scalar_mask(data_c.gpu_handle()?, r_f64)?;
            let ne = predicate_mask_gpu(backend, &ne_h, self.input.numel())?;
            self.mask
                .iter()
                .zip(ne)
                .map(|(&valid, ne_i)| valid && !ne_i)
                .collect()
        } else {
            // Host walk (CPU data, or the documented CUDA bf16/f16 host
            // readback matching the forward lowering).
            let data = self.input.data_vec()?;
            data.iter()
                .zip(self.mask.iter())
                .map(|(&v, &valid)| valid && v == r)
                .collect()
        };
        let n_ties = ties.iter().filter(|&&t| t).count();
        if n_ties == 0 {
            // NaN payload (see doc-comment): nothing matched — zero grads.
            return Ok(vec![Some(mask_scaled_grad(&self.input, &self.mask, zero)?)]);
        }
        let go = scalar_of(grad_output)?;
        let scale = go / t_from_f64::<T>(n_ties as f64, "MaskedExtremumBackward: tie count")?;
        Ok(vec![Some(mask_scaled_grad(&self.input, &ties, scale)?)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        if self.pick_min {
            "MaskedMinBackward"
        } else {
            "MaskedMaxBackward"
        }
    }
}

/// Mean of valid entries; returns a 0-d tensor on the data tensor's device.
///
/// If every entry is masked, returns `NaN` (matches torch.masked; live
/// CUDA probe: `torch.masked.mean(tc, mask=all_false)` →
/// `tensor(nan, device='cuda:0')` — the NaN stays on the data device).
///
/// GPU path computes `sum(data * mask_f) / count_valid` using the same
/// `mul + reduce_sum` lowering as [`masked_sum`] (#597), then divides
/// ON-DEVICE by the uploaded count scalar (#1759 / CORE-065 — the GPU sum
/// is never downloaded; pre-fix this op silently returned a CPU scalar).
///
/// **Autograd (#1758 / CORE-064):** when the data tensor tracks gradients
/// the result carries a [`MaskedMeanBackward`] edge. Live torch oracle:
/// `(torch.masked.mean(t2, mask=m2) * 6.0).backward()` with 3 valid
/// entries → `t2.grad == tensor([2., 0., 2., 0., 2.])` (go / count at
/// valid positions). All-masked: torch routes zero grads (probed) — so
/// does [`MaskedMeanBackward`].
pub fn masked_mean<T: Float>(mt: &MaskedTensor<T>) -> FerrotorchResult<Tensor<T>> {
    let result = masked_mean_forward(mt)?;
    if is_grad_enabled() && mt.data.requires_grad() {
        let grad_fn = Arc::new(MaskedMeanBackward {
            input: mt.data.clone(),
            mask: mt.mask.clone(),
            count: mt.count_valid(),
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        return Tensor::from_operation(storage, shape, grad_fn);
    }
    Ok(result)
}

/// Forward-only mean lowering (no autograd bookkeeping).
fn masked_mean_forward<T: Float>(mt: &MaskedTensor<T>) -> FerrotorchResult<Tensor<T>> {
    if mt.data.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        return masked_mean_gpu(mt);
    }
    if mt.data.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "masked_mean" });
    }
    let data = mt.data.data_vec()?;
    let mut acc = <T as num_traits::Zero>::zero();
    let mut count: usize = 0;
    for (&v, &valid) in data.iter().zip(mt.mask.iter()) {
        if valid {
            acc += v;
            count += 1;
        }
    }
    let val = if count == 0 {
        T::nan()
    } else {
        acc / t_from_f64::<T>(count as f64, "masked_mean: valid count")?
    };
    Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)
}

fn masked_mean_gpu<T: Float>(mt: &MaskedTensor<T>) -> FerrotorchResult<Tensor<T>> {
    let device = mt.data.device();
    let count = mt.count_valid();
    if count == 0 {
        // All-masked → NaN ON THE DATA DEVICE (#1759; torch keeps the NaN
        // on cuda:0). Skip GPU reduction work entirely; the scalar is a
        // fresh host value uploaded once — not a round trip.
        let cpu = Tensor::from_storage(TensorStorage::cpu(vec![T::nan()]), vec![], false)?;
        return cpu.to(device);
    }
    let sum = masked_sum_gpu(mt)?;
    // sum is a 0-d tensor on GPU. Divide ON-DEVICE by the count scalar
    // (uploaded once — the count is host-derived from the host-resident
    // mask, so this is a one-way upload, not a round trip). True division
    // matches the CPU walk and torch's `sum / count` bit-for-bit; the GPU
    // sum never crosses back to the host (#1759).
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let count_t: Tensor<T> = Tensor::from_storage(
        TensorStorage::cpu(vec![t_from_f64::<T>(
            count as f64,
            "masked_mean: valid count",
        )?]),
        vec![],
        false,
    )?
    .to(device)?;
    let mean_h = if is_f32::<T>() {
        backend.div_f32(sum.gpu_handle()?, count_t.gpu_handle()?)?
    } else {
        backend.div_f64(sum.gpu_handle()?, count_t.gpu_handle()?)?
    };
    Tensor::from_storage(TensorStorage::gpu(mean_h), vec![], false)
}

/// Min of valid entries; returns a 0-d tensor on the data tensor's device
/// (NaN if all masked — the #1924-pinned sentinel).
///
/// GPU path: uses the fused `masked_reduce_min` PTX kernel (#627). Single
/// launch reads `(data, mask_f)` directly and combines `mask_f != 0 ?
/// data : +inf` into the running min accumulator — no intermediate
/// buffers, no CPU-side sentinel construction. Same f32/f64-only gate as
/// `masked_sum` / `masked_mean`; other dtypes (bf16/f16) take the CPU
/// walk, matching the existing masked surface.
///
/// **Autograd (#1758 / CORE-064):** when the data tensor tracks gradients
/// the result carries a [`MaskedExtremumBackward`] edge. Live torch
/// 2.11.0+cu130 tie contract: the gradient splits EVENLY among the VALID
/// positions equal to the extremum (`torch.masked.amin` of
/// `[-2, 7, -2, -2]` with mask `[T,T,T,F]` → grad
/// `tensor([0.5, 0., 0.5, 0.])` — the masked-out tied value gets 0).
pub fn masked_min<T: Float>(mt: &MaskedTensor<T>) -> FerrotorchResult<Tensor<T>> {
    masked_extremum(mt, true)
}

/// Max of valid entries; returns a 0-d tensor on the data tensor's device
/// (NaN if all masked — the #1924-pinned sentinel). Autograd + tie
/// contract as for [`masked_min`].
pub fn masked_max<T: Float>(mt: &MaskedTensor<T>) -> FerrotorchResult<Tensor<T>> {
    masked_extremum(mt, false)
}

/// Shared min/max entry: forward lowering + autograd edge.
fn masked_extremum<T: Float>(mt: &MaskedTensor<T>, pick_min: bool) -> FerrotorchResult<Tensor<T>> {
    let result = if mt.data.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        masked_extremum_gpu(mt, pick_min)?
    } else {
        masked_extremum_cpu(mt, pick_min)?
    };
    if is_grad_enabled() && mt.data.requires_grad() {
        let grad_fn = Arc::new(MaskedExtremumBackward {
            input: mt.data.clone(),
            mask: mt.mask.clone(),
            result: result.clone(),
            pick_min,
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        return Tensor::from_operation(storage, shape, grad_fn);
    }
    Ok(result)
}

/// CPU implementation: walk data + mask in one pass.
fn masked_extremum_cpu<T: Float>(
    mt: &MaskedTensor<T>,
    pick_min: bool,
) -> FerrotorchResult<Tensor<T>> {
    let device = mt.data.device();
    let data = mt.data.data_vec()?;
    let mut best: Option<T> = None;
    for (&v, &valid) in data.iter().zip(mt.mask.iter()) {
        if !valid {
            continue;
        }
        best = Some(match best {
            None => v,
            Some(b) if pick_min => {
                if v < b {
                    v
                } else {
                    b
                }
            }
            Some(b) => {
                if v > b {
                    v
                } else {
                    b
                }
            }
        });
    }
    let val = best.unwrap_or_else(T::nan);
    let cpu = Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)?;
    if device.is_cuda() {
        cpu.to(device)
    } else {
        Ok(cpu)
    }
}

/// GPU lowering via the **fused** masked-reduce kernel (#627).
///
/// Single PTX launch that combines `mask_f[i] != 0 ? data[i] : ±inf`
/// directly into the running min/max accumulator. No intermediate
/// `prod` / `filled` buffers, no CPU-side sentinel construction — the
/// only data uploaded is the float mask itself, which we already need
/// for the indicator role.
fn masked_extremum_gpu<T: Float>(
    mt: &MaskedTensor<T>,
    pick_min: bool,
) -> FerrotorchResult<Tensor<T>> {
    // All-masked → the #1924-pinned NaN sentinel ON THE DATA DEVICE
    // (#1759 / CORE-065: pre-fix this edge silently demoted to a CPU
    // scalar; torch keeps its all-masked payload on cuda:0). Short-circuit
    // before allocating GPU reduction buffers — the scalar is a fresh host
    // value uploaded once, not a round trip.
    if mt.count_valid() == 0 {
        let cpu = Tensor::from_storage(TensorStorage::cpu(vec![T::nan()]), vec![], false)?;
        return cpu.to(mt.data.device());
    }

    let device = mt.data.device();
    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let numel = mt.data.numel();

    // Build the [0/1] float mask on device. This is the only host upload —
    // the mask is fundamentally a boolean Vec on the host side, so it has
    // to land on the device once per call regardless. The fused kernel
    // reads it directly and folds the sentinel-fill into the reduce.
    let mask_t: Tensor<T> = mask_as_float_tensor(&mt.mask, mt.data.shape(), device)?;

    // #1658: normalise a narrowed-offset CUDA data tensor to a packed offset-0
    // buffer before the fused masked reduce reads element 0. The mask tensor is
    // freshly built offset-0, so only the data side needs it.
    let data = mt.data.contiguous()?;
    let result_h = if pick_min {
        if is_f32::<T>() {
            backend.masked_min_f32(data.gpu_handle()?, mask_t.gpu_handle()?, numel)?
        } else {
            backend.masked_min_f64(data.gpu_handle()?, mask_t.gpu_handle()?, numel)?
        }
    } else if is_f32::<T>() {
        backend.masked_max_f32(data.gpu_handle()?, mask_t.gpu_handle()?, numel)?
    } else {
        backend.masked_max_f64(data.gpu_handle()?, mask_t.gpu_handle()?, numel)?
    };

    Tensor::from_storage(TensorStorage::gpu(result_h), vec![], false)
}

/// Number of valid (unmasked) entries; returns a 0-d tensor in `T` on the
/// data tensor's device.
///
/// The count itself is computed from the host-resident boolean mask (the
/// mask is host-side BY DESIGN), then the scalar result is uploaded once
/// when the data tensor lives on CUDA (#1759 / CORE-065 device contract:
/// value-producing ops return on the data device). Non-differentiable —
/// the count is constant in the data values, so the result is honestly
/// `requires_grad = false` (torch.masked has no count counterpart).
pub fn masked_count<T: Float>(mt: &MaskedTensor<T>) -> FerrotorchResult<Tensor<T>> {
    let n = mt.count_valid();
    let cpu = Tensor::from_storage(
        TensorStorage::cpu(vec![t_from_f64::<T>(
            n as f64,
            "masked_count: valid count",
        )?]),
        vec![],
        false,
    )?;
    if mt.data.is_cuda() {
        return cpu.to(mt.data.device());
    }
    Ok(cpu)
}

// ---------------------------------------------------------------------------
// Constructors mirroring numpy.ma / torch.masked
// ---------------------------------------------------------------------------

/// Wrap `data` with `condition` interpreted as "where condition is true,
/// mask the value out". Matches `numpy.ma.masked_where`. The resulting
/// [`MaskedTensor`] has `mask = !condition` under the torch convention.
pub fn masked_where<T: Float>(
    data: Tensor<T>,
    condition: &[bool],
) -> FerrotorchResult<MaskedTensor<T>> {
    if condition.len() != data.numel() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "masked_where: condition length {} != data numel {}",
                condition.len(),
                data.numel()
            ),
        });
    }
    let mask: Vec<bool> = condition.iter().map(|&c| !c).collect();
    MaskedTensor::new(data, mask)
}

/// Mask out non-finite entries (NaN, ±∞). Matches `numpy.ma.masked_invalid`.
///
/// On CUDA the `isfinite` predicate runs on-device via the
/// `GpuBackend::isfinite_mask` PTX kernel (#1545); only the resulting boolean
/// mask is read back to populate the host-resident `Vec<bool>` (see
/// [`predicate_mask_gpu`] — this is NOT a CPU↔GPU round trip of the value
/// data, which never leaves the device). f32/f64 only; other dtypes take the
/// host walk.
pub fn masked_invalid<T: Float>(data: Tensor<T>) -> FerrotorchResult<MaskedTensor<T>> {
    if data.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the isfinite predicate reads element 0. The mask is in
        // logical (offset-honouring) order, matching `data`'s own `data_vec`
        // order, so the original `data` is stored unchanged in the result.
        let data_c = data.contiguous()?;
        let mask_h = backend.isfinite_mask(data_c.gpu_handle()?)?;
        let mask = predicate_mask_gpu(backend, &mask_h, data.numel())?;
        return MaskedTensor::new(data, mask);
    }
    if data.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "masked_invalid",
        });
    }
    let data_vec = data.data_vec()?;
    // mask=true means VALID, so finite -> true.
    let mask: Vec<bool> = data_vec
        .iter()
        .map(|v| {
            let f = v.to_f64().unwrap();
            f.is_finite()
        })
        .collect();
    MaskedTensor::new(data, mask)
}

/// Mask out entries equal to `value`. Matches `numpy.ma.masked_equal`.
///
/// On CUDA the `v != value` predicate (the VALID mask under the torch
/// convention) runs on-device via `GpuBackend::ne_scalar_mask` (#1545); only
/// the boolean mask is read back ([`predicate_mask_gpu`]). f32/f64 only.
pub fn masked_equal<T: Float + PartialEq>(
    data: Tensor<T>,
    value: T,
) -> FerrotorchResult<MaskedTensor<T>> {
    if data.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let value_f = value
            .to_f64()
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "masked_equal: value not representable as f64".into(),
            })?;
        // #1658: normalise a narrowed-offset CUDA view to a packed offset-0
        // buffer before the `!= value` predicate reads element 0. The mask is in
        // logical order, matching `data`'s `data_vec` order, so the original
        // `data` is stored unchanged.
        let data_c = data.contiguous()?;
        let mask_h = backend.ne_scalar_mask(data_c.gpu_handle()?, value_f)?;
        let mask = predicate_mask_gpu(backend, &mask_h, data.numel())?;
        return MaskedTensor::new(data, mask);
    }
    if data.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "masked_equal" });
    }
    let data_vec = data.data_vec()?;
    let mask: Vec<bool> = data_vec.iter().map(|&v| v != value).collect();
    MaskedTensor::new(data, mask)
}

/// Read a device-resident `DType::Bool` (u8 0/1) predicate buffer back into the
/// host `Vec<bool>` that backs a [`MaskedTensor`]. The mask is host-resident by
/// design, so this one-way readback of the freshly-computed predicate is the
/// intended data path — the value tensor stays on the device (no R-CODE-4
/// round trip). Each byte is normalised `b != 0` so a stray nonzero never
/// produces an invalid `bool` bit pattern (mirrors `BoolTensor::to(Cpu)`).
///
/// The predicate kernel is launched over the data buffer's RAW cudarc slice
/// length, which the pool over-allocates to a multiple of `ROUND_ELEMENTS`
/// (#1659): a 6-element tensor lands in a 256-element slice, so the readback
/// `bytes` may be longer than the tensor's logical `numel`. The leading `numel`
/// bytes are the valid predicate results; the pooled tail is zeroed garbage to
/// be discarded. Truncate to `numel` so the returned mask matches
/// `data.numel()` (a no-op for an already-packed offset-0 buffer where the raw
/// slice length already equals `numel`).
fn predicate_mask_gpu(
    backend: &dyn crate::gpu_dispatch::GpuBackend,
    mask_h: &crate::gpu_dispatch::GpuBufferHandle,
    numel: usize,
) -> FerrotorchResult<Vec<bool>> {
    let bytes = backend.gpu_to_cpu(mask_h)?;
    Ok(bytes.iter().take(numel).map(|&b| b != 0).collect())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::tensor;

    fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ----- Construction --------------------------------------------------

    #[test]
    fn new_with_matching_mask() {
        let d = t(&[1.0, 2.0, 3.0], &[3]);
        let m = MaskedTensor::new(d, vec![true, false, true]).unwrap();
        assert_eq!(m.shape(), &[3]);
        assert_eq!(m.numel(), 3);
        assert_eq!(m.count_valid(), 2);
        assert_eq!(m.count_masked(), 1);
    }

    #[test]
    fn new_rejects_mask_length_mismatch() {
        let d = t(&[1.0, 2.0, 3.0], &[3]);
        let err = MaskedTensor::new(d, vec![true, false]).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn from_data_marks_all_valid() {
        let d = t(&[1.0, 2.0, 3.0], &[3]);
        let m = MaskedTensor::from_data(d).unwrap();
        assert_eq!(m.count_valid(), 3);
        assert_eq!(m.count_masked(), 0);
    }

    // ----- masked_where (numpy-style) ------------------------------------

    #[test]
    fn masked_where_inverts_condition() {
        // condition=[F, T, F, T] → mask=[T, F, T, F] (i.e. positions 1 and 3
        // are masked OUT in torch convention).
        let d = t(&[10.0, 20.0, 30.0, 40.0], &[4]);
        let mt = masked_where(d, &[false, true, false, true]).unwrap();
        assert_eq!(mt.mask(), &[true, false, true, false]);
        assert_eq!(mt.count_valid(), 2);
    }

    // ----- masked_invalid ------------------------------------------------

    #[test]
    fn masked_invalid_masks_nan() {
        let d = t(&[1.0, f64::NAN, 3.0, f64::INFINITY], &[4]);
        let mt = masked_invalid(d).unwrap();
        // 1.0 finite → valid; NaN → invalid; 3.0 finite → valid; inf → invalid
        assert_eq!(mt.mask(), &[true, false, true, false]);
    }

    // ----- masked_equal --------------------------------------------------

    #[test]
    fn masked_equal_masks_matching() {
        let d = t(&[1.0, 5.0, 5.0, 2.0], &[4]);
        let mt = masked_equal(d, 5.0).unwrap();
        // 5.0 → masked OUT; others → valid.
        assert_eq!(mt.mask(), &[true, false, false, true]);
    }

    // ----- Reductions ----------------------------------------------------

    #[test]
    fn masked_sum_skips_masked_entries() {
        let d = t(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
        // Mask out 2 and 4: valid = 1, 3, 5 → sum 9.
        let mt = MaskedTensor::new(d, vec![true, false, true, false, true]).unwrap();
        let s = masked_sum(&mt).unwrap();
        assert!(close(s.data().unwrap()[0], 9.0, 1e-12));
    }

    #[test]
    fn masked_mean_divides_by_valid_count() {
        let d = t(&[10.0, 0.0, 30.0, 0.0, 50.0], &[5]);
        // valid: 10, 30, 50 → mean 30
        let mt = MaskedTensor::new(d, vec![true, false, true, false, true]).unwrap();
        let r = masked_mean(&mt).unwrap();
        assert!(close(r.data().unwrap()[0], 30.0, 1e-12));
    }

    #[test]
    fn masked_mean_all_masked_returns_nan() {
        let d = t(&[1.0, 2.0, 3.0], &[3]);
        let mt = MaskedTensor::new(d, vec![false, false, false]).unwrap();
        let r = masked_mean(&mt).unwrap();
        assert!(r.data().unwrap()[0].is_nan());
    }

    #[test]
    fn masked_min_max_skip_masked() {
        let d = t(&[5.0, 1.0, 9.0, 2.0], &[4]);
        // Mask out the 9.0 (max) and 1.0 (min) → among valids: 5.0, 2.0
        // min=2.0, max=5.0
        let mt = MaskedTensor::new(d, vec![true, false, false, true]).unwrap();
        assert!(close(
            masked_min(&mt).unwrap().data().unwrap()[0],
            2.0,
            1e-12
        ));
        assert!(close(
            masked_max(&mt).unwrap().data().unwrap()[0],
            5.0,
            1e-12
        ));
    }

    #[test]
    // reason: masked_count returns an integer count cast to float; 3 is
    // exactly representable, so equality (not epsilon) is the right check.
    #[allow(clippy::float_cmp)]
    fn masked_count_returns_valid_count() {
        let d = t(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let mt = MaskedTensor::new(d, vec![true, false, true, true]).unwrap();
        let c = masked_count(&mt).unwrap();
        assert_eq!(c.data().unwrap()[0], 3.0);
    }

    // ----- filled / to_tensor --------------------------------------------

    #[test]
    fn filled_substitutes_default_zero() {
        let d = t(&[1.0, 2.0, 3.0], &[3]);
        let mt = MaskedTensor::new(d, vec![true, false, true]).unwrap();
        let f = mt.filled().unwrap();
        assert_eq!(f.data().unwrap(), &[1.0, 0.0, 3.0]);
    }

    #[test]
    fn filled_uses_fill_value() {
        let d = t(&[1.0, 2.0, 3.0], &[3]);
        let mt = MaskedTensor::new(d, vec![true, false, true])
            .unwrap()
            .with_fill_value(-99.0);
        let f = mt.filled().unwrap();
        assert_eq!(f.data().unwrap(), &[1.0, -99.0, 3.0]);
    }

    #[test]
    fn to_tensor_is_alias_for_filled() {
        let d = t(&[1.0, 2.0, 3.0], &[3]);
        let mt = MaskedTensor::new(d, vec![true, false, true]).unwrap();
        let a = mt.filled().unwrap();
        let b = mt.to_tensor().unwrap();
        assert_eq!(a.data().unwrap(), b.data().unwrap());
    }

    // ----- ferray-ma bridge ----------------------------------------------

    #[test]
    fn to_ferray_round_trip_mean_matches_inhouse() {
        // Cross-check our in-house masked_mean against ferray-ma's
        // MaskedArray::mean() to confirm the mask-inversion bridge is
        // semantically correct.
        let d = t(&[2.0, 4.0, 6.0, 8.0], &[4]);
        let mt = MaskedTensor::new(d, vec![true, false, true, false]).unwrap();
        let inhouse = masked_mean(&mt).unwrap().data().unwrap()[0];
        // Build ferray-ma view via our internal bridge.
        let ferray_ma_view: MaskedArray<f64, FerrayIxDyn> = mt.to_ferray("test").unwrap();
        let ferray_mean = ferray_ma_view.mean().unwrap();
        assert!(close(inhouse, ferray_mean, 1e-12));
        // Sanity: in-house value matches the closed form (2 + 6) / 2 = 4.
        assert!(close(inhouse, 4.0, 1e-12));
    }

    // ----- GPU discipline -------------------------------------------------

    #[test]
    fn constructors_accept_cpu_tensors() {
        // Sanity: every constructor path is reachable for a CPU input.
        let d = tensor(&[1.0_f64, 2.0, 3.0]).unwrap();
        assert!(MaskedTensor::from_data(d.clone()).is_ok());
        assert!(masked_where(d.clone(), &[false, true, false]).is_ok());
        assert!(masked_invalid(d.clone()).is_ok());
        assert!(masked_equal(d, 2.0).is_ok());
    }

    // -------------------------------------------------------------------
    // #616: masked_min/max no longer error on GPU — they fall back to a
    // host-bounce reduce. CPU branch is exercised here; the GPU branch
    // shares the same data_vec() entry point so the same code drives both.
    // -------------------------------------------------------------------

    #[test]
    fn masked_min_max_match_cpu_definition() {
        let d = tensor(&[1.0_f64, -3.0, 5.0, 7.0]).unwrap();
        // mask: [valid, masked, valid, masked] -> visible = {1.0, 5.0}
        let mt = MaskedTensor::new(d, vec![true, false, true, false]).unwrap();
        assert_eq!(masked_min(&mt).unwrap().data().unwrap(), &[1.0]);
        assert_eq!(masked_max(&mt).unwrap().data().unwrap(), &[5.0]);
    }

    #[test]
    fn masked_min_max_all_masked_returns_nan() {
        let d = tensor(&[1.0_f64, 2.0]).unwrap();
        let mt = MaskedTensor::new(d, vec![false, false]).unwrap();
        assert!(masked_min(&mt).unwrap().data().unwrap()[0].is_nan());
        assert!(masked_max(&mt).unwrap().data().unwrap()[0].is_nan());
    }
}
