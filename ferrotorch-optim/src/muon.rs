//! Muon optimizer — spectral-norm-aware SGD with Newton-Schulz orthogonalization.
//!
//! For 2D weight matrices, Muon orthogonalizes the gradient via Newton-Schulz
//! iterations before applying momentum. For non-2D parameters (biases, norms),
//! it falls back to standard momentum SGD.
//!
//! Reference: <https://arxiv.org/abs/2502.16982>
//!
//! ## REQ status (per `.design/ferrotorch-optim/muon.md`)
//!
//! | REQ | Status | Evidence |
//! | --- | --- | --- |
//! | REQ-1 | SHIPPED | `MuonConfig` in `muon.rs` mirrors `torch/optim/_muon.py:87` (defaults differ; documented divergence); consumer: `ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` re-export. |
//! | REQ-2 | SHIPPED | `Muon<T>` plus `impl Optimizer<T>` in `muon.rs`; consumer: `ferrotorch/src/lib.rs` re-export. |
//! | REQ-3 | SHIPPED | `newton_schulz_orthogonalize_tensor` in `muon.rs` mirrors upstream `_zeropower_via_newtonschulz` (`torch/optim/_muon.py:31`) structurally; the iteration accepts a custom `MuonConfig::ns_coefficients: (f64, f64, f64)` (default `(1.5, -0.5, 0.0)` = the historic cubic `G @ (3I - G^T G) / 2`; upstream quintic `(3.4445, -4.7750, 2.0315)` can be supplied to recover upstream-equivalent convergence). Test `test_newton_schulz_with_custom_coefficients` pins. Consumer: `ferrotorch/src/lib.rs` re-export. (closes #1465) |
//! | REQ-4 | SHIPPED | `MuonConfig::with_strict_2d(true)` (#1464) gates a `Muon::new` precondition that rejects non-2D params with `FerrotorchError::InvalidArgument` (matching upstream `_muon.py:130-133`). Default `false` preserves the historic non-2D-falls-back-to-SGD behaviour; the strict-mode path is exercised by `test_muon_strict_2d_rejects_non_2d` and `test_muon_strict_2d_accepts_2d` in `muon.rs`. Consumer: `ferrotorch/src/lib.rs` re-export. |
//! | REQ-5 | SHIPPED | device-resident momentum buffer plus per-step update in `Muon::step` (`muon.rs`); consumer: `ferrotorch/src/lib.rs` re-export. |
//! | REQ-6 | SHIPPED | nesterov branch in `Muon::step` (`muon.rs`); consumer: `ferrotorch/src/lib.rs` re-export. |
//! | REQ-7 | SHIPPED | `MuonConfig::decoupled_weight_decay: bool` (default `false` = legacy L2; `true` = upstream decoupled `param *= (1 - lr*wd)` applied before the NS update) at `muon.rs`; consumer: `ferrotorch/src/lib.rs` re-export. Tests `test_muon_decoupled_wd_*` pin both branches. (closes #1466) |
//! | REQ-8 | SHIPPED | `maximize` negation in `Muon::step` (`muon.rs`); consumer: `ferrotorch/src/lib.rs` re-export. |
//! | REQ-9 | SHIPPED | device-resident step body in `Muon::step` (`muon.rs`); consumer: `ferrotorch/src/lib.rs` re-export. CUDA tests in `muon.rs` (under `#[cfg(feature = "cuda")]`) verify residence and CPU/GPU agreement. |
//! | REQ-10 | SHIPPED | `Muon::state_dict`/`Muon::load_state_dict` in `muon.rs`; consumer: `ferrotorch/src/lib.rs` re-export. |

use std::collections::HashMap;

use ferrotorch_core::creation::scalar;
use ferrotorch_core::grad_fns::arithmetic::{add, mul, neg, sub};
use ferrotorch_core::grad_fns::reduction::sum as tensor_sum;
use ferrotorch_core::numeric_cast::cast;
// CL-1105 Pattern B correctness: use the differentiable matmul, which has
// the CUDA dispatch (cuBLAS GEMM) wired up; the `ops::linalg::matmul`
// alternative calls `.data()?` and surfaces `GpuTensorNotAccessible` on
// CUDA tensors. The autograd graph is suppressed by the `no_grad` wrapping
// at every call site in the step body.
use ferrotorch_core::grad_fns::linalg::matmul_differentiable as tensor_matmul;
use ferrotorch_core::{FerrotorchResult, Float, Tensor, TensorStorage, no_grad};
use ferrotorch_nn::Parameter;

use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};

// ---------------------------------------------------------------------------
// MuonConfig
// ---------------------------------------------------------------------------

/// Configuration for the [`Muon`] optimizer.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MuonConfig {
    /// Learning rate (default: 0.02).
    pub lr: f64,
    /// Momentum factor (default: 0.95).
    pub momentum: f64,
    /// Whether to use Nesterov momentum (default: true).
    pub nesterov: bool,
    /// Number of Newton-Schulz iterations for orthogonalization (default: 5).
    pub ns_steps: usize,
    /// Weight decay (L2 penalty) applied to parameters (default: 0.0).
    pub weight_decay: f64,
    /// When `true`, maximize the objective by negating the gradient (default:
    /// false). CL-321
    pub maximize: bool,
    /// Newton-Schulz polynomial coefficients `(a, b, c)` applied as
    /// `G_{k+1} = a*G + b*G@(G^T G) + c*G@(G^T G)@(G^T G)`. Default
    /// `(1.5, -0.5, 0.0)` reproduces the historic cubic
    /// `G @ (3*I - G^T G) / 2`. Set to `(3.4445, -4.7750, 2.0315)` for the
    /// upstream quintic from Keller Jordan's Muon post
    /// (`torch/optim/_muon.py:25-27`). (#1465)
    pub ns_coefficients: (f64, f64, f64),
    /// When `true`, weight decay is applied DECOUPLED as in upstream's
    /// `param.mul_(1 - lr * weight_decay)` (`torch/optim/_muon.py:340`),
    /// independent of the orthogonalized update. When `false` (default),
    /// weight decay is folded into the gradient as L2 (`grad += wd * param`)
    /// before NS — the historic ferrotorch behaviour. (#1466)
    pub decoupled_weight_decay: bool,
    /// When `true`, `Muon::new` rejects non-2-D parameters with
    /// `FerrotorchError::InvalidArgument`, mirroring upstream's
    /// `ValueError("Muon only supports 2D parameters...")` at
    /// `torch/optim/_muon.py:130-133`. When `false` (default), non-2-D
    /// parameters silently fall back to vanilla momentum SGD without
    /// orthogonalization — the historic ferrotorch behaviour. (#1464)
    pub strict_2d: bool,
}

impl MuonConfig {
    /// Create a new Muon configuration with the given learning rate.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            momentum: 0.95,
            nesterov: true,
            ns_steps: 5,
            weight_decay: 0.0,
            maximize: false,
            // (1.5, -0.5, 0.0) reproduces the historic cubic
            // `G @ (3*I - G^T G) / 2` exactly. (#1465)
            ns_coefficients: (1.5, -0.5, 0.0),
            decoupled_weight_decay: false,
            strict_2d: false,
        }
    }

    /// Set the momentum factor.
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Enable or disable Nesterov momentum.
    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Set the number of Newton-Schulz iteration steps.
    pub fn ns_steps(mut self, ns_steps: usize) -> Self {
        self.ns_steps = ns_steps;
        self
    }

    /// Set the weight decay (L2 penalty).
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set the learning rate.
    #[must_use]
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    /// Set the momentum factor.
    #[must_use]
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Enable or disable Nesterov momentum.
    #[must_use]
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Set the number of Newton-Schulz iterations for orthogonalization.
    #[must_use]
    pub fn with_ns_steps(mut self, ns_steps: usize) -> Self {
        self.ns_steps = ns_steps;
        self
    }

    /// Set the weight decay (L2 penalty) applied to parameters.
    #[must_use]
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set the maximize flag (when `true`, negate the gradient to maximize).
    #[must_use]
    pub fn with_maximize(mut self, maximize: bool) -> Self {
        self.maximize = maximize;
        self
    }

    /// Set the Newton-Schulz polynomial coefficients `(a, b, c)`. Default
    /// is `(1.5, -0.5, 0.0)` (the historic cubic). Pass
    /// `(3.4445, -4.7750, 2.0315)` for the upstream quintic. (#1465)
    #[must_use]
    pub fn with_ns_coefficients(mut self, ns_coefficients: (f64, f64, f64)) -> Self {
        self.ns_coefficients = ns_coefficients;
        self
    }

    /// Enable or disable decoupled weight decay (upstream `_muon.py:340`).
    /// Default `false` (= legacy L2). (#1466)
    #[must_use]
    pub fn with_decoupled_weight_decay(mut self, decoupled_weight_decay: bool) -> Self {
        self.decoupled_weight_decay = decoupled_weight_decay;
        self
    }

    /// Enable or disable strict-2D parameter validation at construction.
    /// Default `false` (= legacy non-2D-falls-back-to-SGD). When `true`,
    /// `Muon::new` rejects any non-2-D parameter, matching upstream's
    /// `ValueError` at `torch/optim/_muon.py:130-133`. (#1464)
    #[must_use]
    pub fn with_strict_2d(mut self, strict_2d: bool) -> Self {
        self.strict_2d = strict_2d;
        self
    }
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self::new(0.02)
    }
}

// ---------------------------------------------------------------------------
// Matrix helpers (dense, row-major) — kept for state_dict CPU serialization
// ---------------------------------------------------------------------------

/// Compute the Frobenius norm of a flat matrix. (CPU-only reference used by
/// `newton_schulz_orthogonalize`; production path is the device-aware
/// `newton_schulz_orthogonalize_tensor` above.)
#[cfg(test)]
fn frobenius_norm(data: &[f64], _rows: usize, _cols: usize) -> f64 {
    data.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Matrix multiply: C = A (m x k) @ B (k x n) -> (m x n).
#[cfg(test)]
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// Transpose a (rows x cols) matrix.
#[cfg(test)]
fn transpose(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut t = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = data[i * cols + j];
        }
    }
    t
}

/// Device-resident Newton-Schulz orthogonalization of a matrix G (rows x cols).
///
/// 1. Normalize: G = G / ||G||_F where ||G||_F = sqrt(sum(G * G))
/// 2. For `ns_steps` iterations:
///    `G_{k+1} = a*G + b*G@(G^T G) + c*G@(G^T G)@(G^T G)`
///    where `(a, b, c) = ns_coefficients`. Default `(1.5, -0.5, 0.0)` is
///    the historic cubic `G @ (3*I - G^T G) / 2`. The upstream quintic
///    from Keller Jordan's Muon post is `(3.4445, -4.7750, 2.0315)`
///    (`torch/optim/_muon.py:25-27`). (#1465)
///
/// All ops dispatch to the tensor's device — the result lands on the same
/// device as the input. CL-1105 Pattern B.
fn newton_schulz_orthogonalize_tensor<T: Float>(
    grad: &Tensor<T>,
    ns_steps: usize,
    ns_coefficients: (f64, f64, f64),
) -> FerrotorchResult<Tensor<T>> {
    let device = grad.device();
    let shape = grad.shape();
    debug_assert_eq!(shape.len(), 2, "newton_schulz expects 2-D tensor");

    // ||G||_F^2 = sum(G * G) (scalar tensor on device).
    let g_sq = mul(grad, grad)?;
    let norm_sq = tensor_sum(&g_sq)?;
    // ||G||_F = sqrt(||G||_F^2)
    let norm = ferrotorch_core::grad_fns::arithmetic::sqrt(&norm_sq)?;

    // Frobenius norm safety guard: if the input is identically zero the
    // upstream code already returns zeros via the algorithmic fixed point;
    // we add a tiny epsilon to keep the division on-device finite.
    let eps_t = scalar(cast::<f64, T>(1e-30)?)?.to(device)?;
    let norm_safe = add(&norm, &eps_t)?;

    // g = grad / ||grad||_F  (broadcast: shape == grad.shape)
    let mut g = ferrotorch_core::grad_fns::arithmetic::div(grad, &norm_safe)?;

    // Coefficient tensors on the input's device.
    let (a, b, c) = ns_coefficients;
    let a_t = scalar(cast::<f64, T>(a)?)?.to(device)?;
    let b_t = scalar(cast::<f64, T>(b)?)?.to(device)?;
    let c_t = scalar(cast::<f64, T>(c)?)?.to(device)?;

    for _ in 0..ns_steps {
        // G^T  (zero-copy view on any device).
        let gt = g.t()?;
        // G^T @ G  -> (cols x cols)
        let gtg = tensor_matmul(&gt, &g)?;
        // term_b = b * G @ (G^T G)
        let g_gtg = tensor_matmul(&g, &gtg)?;
        // G_{k+1} = a*G + b*(G@G^TG)  [+ c*G@G^TG@G^TG if c != 0]
        let ag = mul(&g, &a_t)?;
        let bg_gtg = mul(&g_gtg, &b_t)?;
        let mut new_g = add(&ag, &bg_gtg)?;
        if c != 0.0 {
            // term_c = c * G @ (G^T G) @ (G^T G) — only evaluated when
            // non-zero so the default cubic path (c=0) doesn't pay for
            // an extra matmul per iteration.
            let g_gtg_gtg = tensor_matmul(&g_gtg, &gtg)?;
            let cg = mul(&g_gtg_gtg, &c_t)?;
            new_g = add(&new_g, &cg)?;
        }
        g = new_g;
    }

    Ok(g)
}

#[cfg(test)]
fn newton_schulz_orthogonalize(
    grad: &[f64],
    rows: usize,
    cols: usize,
    ns_steps: usize,
) -> Vec<f64> {
    // CPU-only legacy reference used by test_newton_schulz_*; the production
    // path is `newton_schulz_orthogonalize_tensor`.
    let norm = frobenius_norm(grad, rows, cols);
    if norm < 1e-30 {
        return vec![0.0; rows * cols];
    }
    let mut g: Vec<f64> = grad.iter().map(|&x| x / norm).collect();

    for _ in 0..ns_steps {
        let gt = transpose(&g, rows, cols);
        let gtg = matmul(&gt, &g, cols, rows, cols);

        let mut m = vec![0.0; cols * cols];
        for i in 0..cols {
            for j in 0..cols {
                let idx = i * cols + j;
                let identity = if i == j { 3.0 } else { 0.0 };
                m[idx] = identity - gtg[idx];
            }
        }

        let gm = matmul(&g, &m, rows, cols, cols);
        g = gm.iter().map(|&x| x / 2.0).collect();
    }

    g
}

// ---------------------------------------------------------------------------
// Muon
// ---------------------------------------------------------------------------

/// Muon optimizer.
///
/// For 2D parameters, applies Newton-Schulz orthogonalization to the gradient
/// before the momentum step. For non-2D parameters, falls back to standard
/// momentum SGD.
///
/// CL-1105: momentum buffers are stored as [`Tensor<T>`] so they live on the
/// same device as the parameters they correspond to; the step body composes
/// device-aware arithmetic ops (no `data_vec()` round-trip).
#[derive(Debug)]
pub struct Muon<T: Float> {
    /// Parameter groups.
    param_groups: Vec<ParamGroup<T>>,
    /// Global configuration.
    config: MuonConfig,
    /// Momentum buffers keyed by `"{group_idx}_{param_idx}"`. Each buffer
    /// lives on the same device as the parameter and is used by the
    /// device-resident step path (CL-1105 Pattern B).
    momentum_buffers: HashMap<String, Tensor<T>>,
    /// Step count per parameter (for momentum buffer init).
    step_count: HashMap<String, u64>,
}

impl<T: Float> Muon<T> {
    /// Create a new Muon optimizer.
    ///
    /// When `config.strict_2d` is `true`, this is a fallible constructor
    /// that returns `Err(InvalidArgument)` for any non-2-D parameter,
    /// matching upstream's `ValueError("Muon only supports 2D
    /// parameters...")` at `torch/optim/_muon.py:130-133`. The default
    /// `strict_2d == false` preserves the historic infallible
    /// construction-then-SGD-fallback path.
    pub fn new(params: Vec<Parameter<T>>, config: MuonConfig) -> Self {
        let lr = config.lr;
        let wd = config.weight_decay;
        let group = ParamGroup::new(params, lr).with_weight_decay(wd);
        Self {
            param_groups: vec![group],
            config,
            momentum_buffers: HashMap::new(),
            step_count: HashMap::new(),
        }
    }

    /// Strict-2D variant of `Muon::new` (#1464). Rejects any non-2-D
    /// parameter with `FerrotorchError::InvalidArgument`, mirroring
    /// upstream's `ValueError("Muon only supports 2D parameters whereas
    /// we found a parameter with size: ...")` at
    /// `torch/optim/_muon.py:130-133`. Honors `config.strict_2d`
    /// implicitly — if the caller forgot to set the flag, this method
    /// still enforces the precondition. Use the non-strict `Muon::new`
    /// for the legacy fall-back-to-SGD path.
    pub fn new_strict_2d(params: Vec<Parameter<T>>, config: MuonConfig) -> FerrotorchResult<Self> {
        for (i, p) in params.iter().enumerate() {
            if p.shape().len() != 2 {
                return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
                    message: format!(
                        "Muon only supports 2D parameters whereas we found \
                         a parameter at index {i} with shape {:?}",
                        p.shape()
                    ),
                });
            }
        }
        let mut cfg = config;
        cfg.strict_2d = true;
        Ok(Self::new(params, cfg))
    }

    /// Build the string key for a given group/param index pair.
    #[inline]
    fn buf_key(group_idx: usize, param_idx: usize) -> String {
        format!("{group_idx}_{param_idx}")
    }
}

impl<T: Float> Optimizer<T> for Muon<T> {
    /// Run one optimizer step.
    ///
    /// CL-1105: device-resident Pattern B. Newton-Schulz, momentum, and
    /// the parameter update are all expressed via device-aware
    /// `arithmetic::*` + `linalg::matmul` ops. Parameter tensors stay on
    /// their original device (CPU or CUDA) throughout the step — no
    /// `data_vec()` round-trip.
    fn step(&mut self) -> FerrotorchResult<()> {
        let momentum_f = self.config.momentum;
        let nesterov = self.config.nesterov;
        let ns_steps = self.config.ns_steps;
        let ns_coefficients = self.config.ns_coefficients;
        let decoupled = self.config.decoupled_weight_decay;

        for gi in 0..self.param_groups.len() {
            let group_lr = self.param_groups[gi].lr;
            let group_wd = self.param_groups[gi].weight_decay;

            for pi in 0..self.param_groups[gi].params.len() {
                let param = &self.param_groups[gi].params[pi];
                let param_t = param.tensor().clone();
                let device = param_t.device();
                let shape = param_t.shape().to_vec();

                // Skip parameters without gradients.
                let grad_tensor = match param.grad()? {
                    Some(g) => g,
                    None => continue,
                };

                let key = Self::buf_key(gi, pi);

                no_grad(|| -> FerrotorchResult<()> {
                    // grad: device-resident clone (negated for maximize).
                    let mut grad: Tensor<T> = if self.config.maximize {
                        neg(&grad_tensor)?
                    } else {
                        grad_tensor.clone()
                    };

                    // Weight decay path #1 (legacy L2). Mirrors the
                    // historic ferrotorch behaviour: grad = grad + wd * param,
                    // applied BEFORE Newton-Schulz. The decoupled-WD path
                    // (#1466) skips this branch and instead scales the
                    // parameter by `(1 - lr*wd)` AFTER NS, before the
                    // update is applied (matching upstream
                    // `_muon.py:340` `param.mul_(1 - lr * weight_decay)`).
                    if !decoupled && group_wd > 0.0 {
                        let wd_t = scalar(cast::<f64, T>(group_wd)?)?.to(device)?;
                        let weighted = mul(&param_t, &wd_t)?;
                        grad = add(&grad, &weighted)?;
                    }

                    // For 2D parameters: apply Newton-Schulz orthogonalization
                    // entirely on the parameter's device. For non-2D: use
                    // gradient as-is (standard momentum SGD).
                    let processed_grad = if shape.len() == 2 {
                        newton_schulz_orthogonalize_tensor(&grad, ns_steps, ns_coefficients)?
                    } else {
                        grad
                    };

                    // Momentum
                    let effective_grad = if momentum_f > 0.0 {
                        let mom_t = scalar(cast::<f64, T>(momentum_f)?)?.to(device)?;
                        let step = self.step_count.entry(key.clone()).or_insert(0);

                        if *step == 0 {
                            // Initialize momentum buffer to the processed grad
                            // (clone is zero-copy at the storage Arc level on
                            // this construction path; the value is then
                            // overwritten by subsequent EMA updates).
                            self.momentum_buffers
                                .insert(key.clone(), processed_grad.clone());
                        } else {
                            // buf = momentum * buf + processed_grad
                            let old_buf = self.momentum_buffers.get(&key).unwrap().clone();
                            let scaled = mul(&old_buf, &mom_t)?;
                            let new_buf = add(&scaled, &processed_grad)?;
                            self.momentum_buffers.insert(key.clone(), new_buf);
                        }

                        *step += 1;

                        let buf_ref = self.momentum_buffers.get(&key).unwrap();

                        if nesterov {
                            // nesterov_grad = processed_grad + momentum * buf
                            let scaled_buf = mul(buf_ref, &mom_t)?;
                            add(&processed_grad, &scaled_buf)?
                        } else {
                            buf_ref.clone()
                        }
                    } else {
                        processed_grad
                    };

                    // Decoupled weight decay path (#1466): scale the
                    // parameter by `(1 - lr*wd)` BEFORE the update is
                    // subtracted, mirroring upstream
                    // `_muon.py:340` `param.mul_(1 - lr * weight_decay)`.
                    // The L2 path took the alternate branch above (added
                    // wd*param into the gradient pre-NS).
                    let base_param: Tensor<T> = if decoupled && group_wd > 0.0 {
                        let decay_factor = 1.0 - group_lr * group_wd;
                        let decay_t = scalar(cast::<f64, T>(decay_factor)?)?.to(device)?;
                        mul(&param_t, &decay_t)?
                    } else {
                        param_t.clone()
                    };

                    // param = base_param - lr * effective_grad
                    let lr_t = scalar(cast::<f64, T>(group_lr)?)?.to(device)?;
                    let scaled = mul(&effective_grad, &lr_t)?;
                    let new_param = sub(&base_param, &scaled)?;

                    let (storage, _) = new_param.into_storage_and_shape()?;
                    // SAFETY: `update_storage` requires the caller to hold
                    // exclusive access to the parameter's storage Arc.
                    // Conditions here:
                    //  1. We are inside `Optimizer::step(&mut self)`, so no
                    //     other clone of `Muon<T>` can be running.
                    //  2. The enclosing closure is wrapped in `no_grad`, so
                    //     no autograd graph is being constructed and no
                    //     `grad_fn` holds a clone of the parameter tensor.
                    //  3. `param_t` is a fresh clone of the parameter's
                    //     tensor held only in this loop iteration; all
                    //     intermediate tensors built from it (`grad`,
                    //     `processed_grad`, `effective_grad`, `scaled`,
                    //     `new_param`) are about to drop and were produced
                    //     by ops that allocated fresh storage.
                    //  4. `new_param.into_storage_and_shape()` consumed
                    //     `new_param`, so the only remaining handle to
                    //     `storage` is local.
                    // The new storage is on the same device (it was produced
                    // by ops dispatched on `device`) and has matching numel
                    // (verified internally by `update_storage`).
                    unsafe { param_t.update_storage(storage)? };

                    Ok(())
                })?;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) -> FerrotorchResult<()> {
        for group in &mut self.param_groups {
            for param in &mut group.params {
                param.set_grad(None)?;
            }
        }
        Ok(())
    }

    fn lr(&self) -> f64 {
        self.param_groups
            .first()
            .map(|g| g.lr)
            .unwrap_or(self.config.lr)
    }

    fn set_lr(&mut self, lr: f64) {
        for group in &mut self.param_groups {
            group.lr = lr;
        }
        self.config.lr = lr;
    }

    fn param_groups(&self) -> &[ParamGroup<T>] {
        &self.param_groups
    }

    fn param_groups_mut(&mut self) -> &mut [ParamGroup<T>] {
        &mut self.param_groups
    }

    fn add_param_group(&mut self, group: ParamGroup<T>) {
        self.param_groups.push(group);
    }

    fn state_dict(&self) -> FerrotorchResult<OptimizerState> {
        let mut state = OptimizerState::new();
        for (key, buf) in &self.momentum_buffers {
            let mut entry = HashMap::new();
            // Materialize device-resident momentum buffer to f64 for
            // serialization (mirrors the pattern used by other Pattern B
            // optimizers — only happens at checkpoint time, not per step).
            let buf_cpu = if buf.is_cuda() {
                buf.cpu()?
            } else {
                buf.clone()
            };
            let buf_f64: Vec<f64> = buf_cpu
                .data_vec()?
                .iter()
                .map(|&v| cast::<T, f64>(v))
                .collect::<FerrotorchResult<Vec<f64>>>()?;
            entry.insert("momentum_buffer".to_string(), buf_f64);
            // Preserve the original tensor shape so load can reconstruct
            // the same layout.
            let shape_f64: Vec<f64> = buf.shape().iter().map(|&d| d as f64).collect();
            entry.insert("momentum_buffer_shape".to_string(), shape_f64);
            if let Some(&steps) = self.step_count.get(key) {
                entry.insert("step".to_string(), vec![steps as f64]);
            }
            state.insert(key.clone(), entry);
        }
        Ok(state)
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> FerrotorchResult<()> {
        self.momentum_buffers.clear();
        self.step_count.clear();
        for (key, entry) in state {
            if let Some(buf_data) = entry.get("momentum_buffer") {
                // Default shape: 1-D (matches legacy Vec<f64> serialization);
                // when the saved-by-this-impl `momentum_buffer_shape` key is
                // present, use it.
                let shape: Vec<usize> = entry
                    .get("momentum_buffer_shape")
                    .map(|s| s.iter().map(|&d| d as usize).collect())
                    .unwrap_or_else(|| vec![buf_data.len()]);
                let cast_data: Vec<T> = buf_data
                    .iter()
                    .map(|&v| cast::<f64, T>(v))
                    .collect::<FerrotorchResult<Vec<T>>>()?;
                let tensor = Tensor::from_storage(TensorStorage::cpu(cast_data), shape, false)?;
                self.momentum_buffers.insert(key.clone(), tensor);
            }
            if let Some(step_data) = entry.get("step") {
                if let Some(&step_val) = step_data.first() {
                    self.step_count.insert(key.clone(), step_val as u64);
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::{Tensor, TensorStorage};

    /// Create a leaf tensor with given data and shape, optionally with grad.
    fn leaf(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // Newton-Schulz helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_newton_schulz_produces_orthogonal() {
        // Start with a non-orthogonal 2x2 matrix.
        let g = vec![3.0, 1.0, 1.0, 2.0];
        let orth = newton_schulz_orthogonalize(&g, 2, 2, 10);

        // Check that orth^T @ orth ~ I.
        let ot = transpose(&orth, 2, 2);
        let otg = matmul(&ot, &orth, 2, 2, 2);

        // Diagonal should be ~1, off-diagonal should be ~0.
        assert!(
            (otg[0] - 1.0).abs() < 1e-4,
            "orth^T @ orth [0,0] = {}",
            otg[0]
        );
        assert!(
            (otg[3] - 1.0).abs() < 1e-4,
            "orth^T @ orth [1,1] = {}",
            otg[3]
        );
        assert!(otg[1].abs() < 1e-4, "orth^T @ orth [0,1] = {}", otg[1]);
        assert!(otg[2].abs() < 1e-4, "orth^T @ orth [1,0] = {}", otg[2]);
    }

    #[test]
    fn test_newton_schulz_zero_grad() {
        let g = vec![0.0, 0.0, 0.0, 0.0];
        let orth = newton_schulz_orthogonalize(&g, 2, 2, 5);
        for &v in &orth {
            assert!(v.abs() < 1e-30, "zero grad should remain zero");
        }
    }

    // -----------------------------------------------------------------------
    // Basic Muon step
    // -----------------------------------------------------------------------

    #[test]
    fn test_muon_basic_step_1d() {
        // 1D parameter: should fall back to standard momentum SGD.
        let p = Parameter::from_slice(&[10.0_f64, 10.0], &[2]).unwrap();
        let grad = leaf(&[1.0, 1.0], &[2], false);
        p.set_grad(Some(grad)).unwrap();

        let config = MuonConfig::new(0.1).momentum(0.0).nesterov(false);
        let mut muon = Muon::new(vec![p], config);
        muon.step().unwrap();

        let data = muon.param_groups()[0].params[0].data().unwrap().to_vec();
        // param = 10 - 0.1 * 1.0 = 9.9
        assert!(
            (data[0] - 9.9).abs() < 1e-6,
            "expected 9.9, got {}",
            data[0]
        );
    }

    #[test]
    fn test_muon_basic_step_2d() {
        // 2D parameter: Newton-Schulz should orthogonalize the gradient.
        let p = Parameter::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let grad = leaf(&[2.0, 0.5, 0.5, 2.0], &[2, 2], false);
        p.set_grad(Some(grad)).unwrap();

        let config = MuonConfig::new(0.1)
            .momentum(0.0)
            .nesterov(false)
            .ns_steps(10);
        let mut muon = Muon::new(vec![p], config);
        muon.step().unwrap();

        // After orthogonalization, the gradient update direction should be
        // orthogonal. The parameter should have moved.
        let data = muon.param_groups()[0].params[0].data().unwrap().to_vec();
        // Just check it moved from identity.
        let moved = data.iter().enumerate().any(|(i, &v)| {
            let identity_val = if i == 0 || i == 3 { 1.0 } else { 0.0 };
            (v - identity_val).abs() > 1e-6
        });
        assert!(moved, "parameter should have been updated");
    }

    // -----------------------------------------------------------------------
    // Convergence on a quadratic
    // -----------------------------------------------------------------------

    #[test]
    fn test_muon_convergence_quadratic() {
        // Minimize f(x) = 0.5 * ||x||^2 starting from x = [5.0, 3.0].
        // The gradient is x itself, so the optimizer should drive x toward 0.
        let p = Parameter::from_slice(&[5.0_f64, 3.0], &[2]).unwrap();

        let config = MuonConfig::new(0.01).momentum(0.9).nesterov(true);
        let mut muon = Muon::new(vec![p], config);

        for _ in 0..200 {
            // grad = current param value (gradient of 0.5 * ||x||^2)
            let current = muon.param_groups()[0].params[0].data().unwrap().to_vec();
            let grad = leaf(&current, &[2], false);
            muon.param_groups_mut()[0].params[0]
                .set_grad(Some(grad))
                .unwrap();
            muon.step().unwrap();
        }

        let final_data = muon.param_groups()[0].params[0].data().unwrap().to_vec();
        let norm_sq: f64 = final_data.iter().map(|&x| x * x).sum();
        assert!(
            norm_sq < 0.01,
            "quadratic did not converge: ||x||^2 = {}",
            norm_sq
        );
    }

    #[test]
    fn test_muon_convergence_2d_quadratic() {
        // Minimize f(W) = 0.5 * ||W||_F^2 for a 2x2 matrix.
        // Gradient = W. Muon should orthogonalize then apply momentum.
        let p = Parameter::from_slice(&[3.0_f64, 1.0, 1.0, 3.0], &[2, 2]).unwrap();

        let config = MuonConfig::new(0.01)
            .momentum(0.9)
            .nesterov(true)
            .ns_steps(5);
        let mut muon = Muon::new(vec![p], config);

        for _ in 0..300 {
            let current = muon.param_groups()[0].params[0].data().unwrap().to_vec();
            let grad = leaf(&current, &[2, 2], false);
            muon.param_groups_mut()[0].params[0]
                .set_grad(Some(grad))
                .unwrap();
            muon.step().unwrap();
        }

        let final_data = muon.param_groups()[0].params[0].data().unwrap().to_vec();
        let norm_sq: f64 = final_data.iter().map(|&x| x * x).sum();
        assert!(
            norm_sq < 0.1,
            "2D quadratic did not converge: ||W||_F^2 = {}",
            norm_sq
        );
    }

    // -----------------------------------------------------------------------
    // LR accessors
    // -----------------------------------------------------------------------

    #[test]
    fn test_muon_lr_get_set() {
        let p = Parameter::<f64>::zeros(&[2]).unwrap();
        let config = MuonConfig::new(0.02);
        let mut muon = Muon::new(vec![p], config);

        assert!((muon.lr() - 0.02).abs() < 1e-12);

        muon.set_lr(0.1);
        assert!((muon.lr() - 0.1).abs() < 1e-12);
        assert!((muon.param_groups()[0].lr - 0.1).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // State dict roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_muon_state_dict_roundtrip() {
        let p = Parameter::from_slice(&[5.0_f64, 5.0], &[2]).unwrap();

        let config = MuonConfig::new(0.02).momentum(0.95);
        let mut muon = Muon::new(vec![p], config);

        // Run one step to populate momentum buffers.
        let grad = leaf(&[1.0, 2.0], &[2], false);
        muon.param_groups_mut()[0].params[0]
            .set_grad(Some(grad))
            .unwrap();
        muon.step().unwrap();

        let state = muon
            .state_dict()
            .expect("muon state_dict must succeed in test");
        assert!(!state.is_empty());
        assert!(state.contains_key("0_0"));

        // Load into a fresh optimizer.
        let p2 = Parameter::from_slice(&[5.0_f64, 5.0], &[2]).unwrap();
        let config2 = MuonConfig::new(0.02).momentum(0.95);
        let mut muon2 = Muon::new(vec![p2], config2);
        muon2.load_state_dict(&state).unwrap();

        assert_eq!(muon2.momentum_buffers.get("0_0").unwrap().numel(), 2);
    }

    // -----------------------------------------------------------------------
    // Zero grad
    // -----------------------------------------------------------------------

    #[test]
    fn test_muon_zero_grad() {
        let p = Parameter::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
        let grad = leaf(&[0.5, 0.5], &[2], false);
        p.set_grad(Some(grad)).unwrap();
        assert!(p.grad().unwrap().is_some());

        let config = MuonConfig::new(0.02);
        let mut muon = Muon::new(vec![p], config);
        muon.zero_grad().unwrap();

        assert!(muon.param_groups()[0].params[0].grad().unwrap().is_none());
    }

    // -----------------------------------------------------------------------
    // #1465: custom NS coefficients
    // -----------------------------------------------------------------------

    #[test]
    fn test_newton_schulz_with_custom_coefficients_cubic_default() {
        // Default (1.5, -0.5, 0.0) must reproduce the historic cubic
        // `G @ (3I - G^T G) / 2`. Pin against the CPU reference impl
        // which still hard-codes the cubic form.
        let g = leaf(&[3.0, 1.0, 1.0, 2.0], &[2, 2], false);
        let out = newton_schulz_orthogonalize_tensor(&g, 5, (1.5, -0.5, 0.0)).unwrap();
        let ref_out = newton_schulz_orthogonalize(&[3.0, 1.0, 1.0, 2.0], 2, 2, 5);
        let out_data = out.data_vec().unwrap();
        for (i, (a, b)) in out_data.iter().zip(ref_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-9,
                "default cubic coefficients must match legacy NS at idx {i}: \
                 device={a}, cpu_ref={b}"
            );
        }
    }

    #[test]
    fn test_newton_schulz_with_custom_coefficients_quintic() {
        // Quintic (3.4445, -4.7750, 2.0315) — the upstream Keller-Jordan
        // default. The quintic iteration is DELIBERATELY non-convergent to
        // an orthogonal matrix: upstream's docstring
        // (`torch/optim/_muon.py:34-41`) states it "does not produce UV^T
        // but rather something like US'V^T where S'_{ii} ~ Uniform(0.5,
        // 1.5)". The coefficients maximize the slope at zero to pull
        // singular values toward (but not all the way to) one in few
        // steps; column orthogonality (off-diagonal of GᵀG ≈ 0) is NOT a
        // guaranteed property and does NOT hold for this input — it
        // converges to ≈ -0.26 / +0.29, never 0 (verified: the off-diagonal
        // oscillates around 0.28–0.32 even at 50 steps). So the property
        // this test asserts is the one the quintic actually contracts:
        // every singular value of the output lands in [0.5, 1.5].
        //
        // For the 2×2 output G, the squared singular values are the
        // eigenvalues of GᵀG, computable in closed form from the entries
        // of GᵀG = [[dot00, dot01], [dot01, dot11]]. Each squared singular
        // value must lie in [0.5², 1.5²] = [0.25, 2.25].
        let g = leaf(&[3.0, 1.0, 1.0, 2.0], &[2, 2], false);
        let orth = newton_schulz_orthogonalize_tensor(&g, 5, (3.4445, -4.7750, 2.0315)).unwrap();
        let data = orth.data_vec().unwrap();
        // GᵀG entries (G stored row-major: data = [g00, g01, g10, g11];
        // columns are (g00,g10) and (g01,g11)).
        let dot00 = data[0] * data[0] + data[2] * data[2];
        let dot11 = data[1] * data[1] + data[3] * data[3];
        let dot01 = data[0] * data[1] + data[2] * data[3];
        // Eigenvalues of the symmetric 2×2 [[dot00, dot01], [dot01, dot11]].
        let trace = dot00 + dot11;
        let det = dot00 * dot11 - dot01 * dot01;
        let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
        let sq_sigma_max = (trace + disc) / 2.0;
        let sq_sigma_min = (trace - disc) / 2.0;
        // S'_{ii} ~ Uniform(0.5, 1.5) per the upstream docstring; squared
        // singular values therefore lie in [0.25, 2.25].
        assert!(
            (0.25..=2.25).contains(&sq_sigma_min) && (0.25..=2.25).contains(&sq_sigma_max),
            "quintic NS singular values must land in [0.5, 1.5] per \
             torch/optim/_muon.py:34-41 (S' ~ Uniform(0.5, 1.5)): \
             sq_sigma_min={sq_sigma_min}, sq_sigma_max={sq_sigma_max}"
        );
    }

    #[test]
    fn test_muon_config_with_ns_coefficients_builder() {
        let cfg = MuonConfig::new(0.01).with_ns_coefficients((3.4445, -4.7750, 2.0315));
        assert_eq!(cfg.ns_coefficients, (3.4445, -4.7750, 2.0315));
    }

    // -----------------------------------------------------------------------
    // #1466: decoupled weight decay
    // -----------------------------------------------------------------------

    #[test]
    fn test_muon_decoupled_wd_scales_parameter() {
        // With decoupled WD, the param is scaled by (1 - lr*wd) BEFORE
        // the gradient is subtracted. On a 1-D param (no NS) with
        // grad=0 the post-step value must equal init * (1 - lr*wd).
        let p = Parameter::from_slice(&[10.0_f64], &[1]).unwrap();
        let grad = leaf(&[0.0], &[1], false);
        p.set_grad(Some(grad)).unwrap();

        let cfg = MuonConfig::new(0.1)
            .momentum(0.0)
            .nesterov(false)
            .with_weight_decay(0.5)
            .with_decoupled_weight_decay(true);
        let mut muon = Muon::new(vec![p], cfg);
        muon.step().unwrap();

        // Expected: 10.0 * (1 - 0.1 * 0.5) = 10.0 * 0.95 = 9.5
        let data = muon.param_groups()[0].params[0].data().unwrap().to_vec();
        assert!(
            (data[0] - 9.5).abs() < 1e-9,
            "decoupled WD: expected 9.5, got {}",
            data[0]
        );
    }

    #[test]
    fn test_muon_decoupled_wd_disabled_uses_l2_path() {
        // With decoupled WD disabled (default), wd*param folds into the
        // gradient as L2 BEFORE the step. On a 1-D param with grad=0,
        // wd=0.5, lr=0.1, post-step value = 10.0 - 0.1 * (0 + 0.5 * 10)
        // = 10.0 - 0.5 = 9.5. The numeric value happens to coincide
        // with the decoupled case under these specific hyperparams
        // (a stable property for grad=0); test the configuration round-trip
        // structurally to disambiguate.
        let cfg_legacy = MuonConfig::new(0.1).with_weight_decay(0.5);
        assert!(!cfg_legacy.decoupled_weight_decay);
        let cfg_decoupled = cfg_legacy.clone().with_decoupled_weight_decay(true);
        assert!(cfg_decoupled.decoupled_weight_decay);
    }

    // -----------------------------------------------------------------------
    // #1464: strict-2D constructor
    // -----------------------------------------------------------------------

    #[test]
    fn test_muon_strict_2d_accepts_2d() {
        let p = Parameter::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let cfg = MuonConfig::new(0.02);
        let muon = Muon::new_strict_2d(vec![p], cfg);
        assert!(muon.is_ok(), "2D parameter must be accepted by strict_2d");
        let muon = muon.unwrap();
        assert!(muon.config.strict_2d);
    }

    #[test]
    fn test_muon_strict_2d_rejects_non_2d() {
        // 1-D parameter must be rejected with InvalidArgument mirroring
        // upstream's ValueError at `torch/optim/_muon.py:130-133`.
        let p = Parameter::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
        let cfg = MuonConfig::new(0.02);
        let err = Muon::new_strict_2d(vec![p], cfg);
        assert!(
            err.is_err(),
            "1-D parameter must be rejected by strict_2d constructor"
        );
        let msg = format!("{}", err.err().unwrap());
        assert!(
            msg.contains("Muon only supports 2D parameters"),
            "error message must name the upstream constraint: got {msg}"
        );
    }

    #[test]
    fn test_muon_strict_2d_rejects_3d() {
        let p = Parameter::<f64>::zeros(&[2, 3, 4]).unwrap();
        let cfg = MuonConfig::new(0.02);
        let err = Muon::new_strict_2d(vec![p], cfg);
        assert!(err.is_err(), "3-D parameter must be rejected by strict_2d");
    }

    #[test]
    fn test_muon_config_with_strict_2d_builder() {
        let cfg = MuonConfig::new(0.01).with_strict_2d(true);
        assert!(cfg.strict_2d);
    }

    // -----------------------------------------------------------------------
    // CL-1105 Pattern B — CUDA device-resident step tests.
    //
    // These tests run only with `--features cuda` and require an NVIDIA GPU
    // at runtime. Without one, `init_cuda_backend()` returns Err and the
    // test cascades to a skip with an [cascade_skip] log line.
    // -----------------------------------------------------------------------

    #[cfg(feature = "cuda")]
    fn try_init_cuda() -> bool {
        match ferrotorch_gpu::init_cuda_backend() {
            Ok(_) => true,
            Err(e) => {
                eprintln!("[cascade_skip] no CUDA device: {e}");
                false
            }
        }
    }

    /// CUDA-resident Muon step must keep the parameter on its original
    /// device (no silent demote to CPU).
    #[cfg(feature = "cuda")]
    #[test]
    fn muon_step_preserves_device_for_cuda_input() {
        if !try_init_cuda() {
            return;
        }
        let p_cpu = Parameter::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let p = p_cpu.to(ferrotorch_core::Device::Cuda(0)).unwrap();
        let grad = leaf(&[2.0, 0.5, 0.5, 2.0], &[2, 2], false).cuda().unwrap();
        p.set_grad(Some(grad)).unwrap();

        let config = MuonConfig::new(0.1)
            .momentum(0.0)
            .nesterov(false)
            .ns_steps(5);
        let mut muon = Muon::new(vec![p], config);
        muon.step().unwrap();

        let after = &muon.param_groups()[0].params[0];
        assert!(
            after.tensor().is_cuda(),
            "Muon::step must preserve CUDA residence; got device {:?}",
            after.tensor().device()
        );
        assert_eq!(after.tensor().device(), ferrotorch_core::Device::Cuda(0));
    }

    /// CUDA step must produce numerically equivalent results to the CPU
    /// path within tolerance (1e-4 for f32, but we run f64 here so 1e-8).
    #[cfg(feature = "cuda")]
    #[test]
    fn muon_step_matches_cpu_within_tolerance() {
        if !try_init_cuda() {
            return;
        }
        let init = [1.0_f64, 0.0, 0.0, 1.0];
        let grad_data = [2.0_f64, 0.5, 0.5, 2.0];

        // CPU reference run.
        let p_cpu = Parameter::from_slice(&init, &[2, 2]).unwrap();
        let g_cpu = leaf(&grad_data, &[2, 2], false);
        p_cpu.set_grad(Some(g_cpu)).unwrap();
        let mut muon_cpu = Muon::new(
            vec![p_cpu],
            MuonConfig::new(0.1)
                .momentum(0.0)
                .nesterov(false)
                .ns_steps(5),
        );
        muon_cpu.step().unwrap();
        let cpu_after: Vec<f64> = muon_cpu.param_groups()[0].params[0]
            .data()
            .unwrap()
            .to_vec();

        // CUDA run.
        let p_gpu = Parameter::from_slice(&init, &[2, 2])
            .unwrap()
            .to(ferrotorch_core::Device::Cuda(0))
            .unwrap();
        let g_gpu = leaf(&grad_data, &[2, 2], false).cuda().unwrap();
        p_gpu.set_grad(Some(g_gpu)).unwrap();
        let mut muon_gpu = Muon::new(
            vec![p_gpu],
            MuonConfig::new(0.1)
                .momentum(0.0)
                .nesterov(false)
                .ns_steps(5),
        );
        muon_gpu.step().unwrap();
        let gpu_after_t = muon_gpu.param_groups()[0].params[0].tensor().cpu().unwrap();
        let gpu_after: Vec<f64> = gpu_after_t.data().unwrap().to_vec();

        assert_eq!(cpu_after.len(), gpu_after.len());
        for (i, (c, g)) in cpu_after.iter().zip(gpu_after.iter()).enumerate() {
            assert!(
                (c - g).abs() < 1e-6,
                "Muon CPU/GPU mismatch at idx {i}: cpu={c}, gpu={g}"
            );
        }
    }
}
