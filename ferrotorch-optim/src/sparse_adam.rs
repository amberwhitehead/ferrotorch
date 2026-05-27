//! SparseAdam optimizer — Adam variant for sparse gradients.
//!
//! `torch.optim.SparseAdam` is intended for parameters of a DENSE layout
//! whose gradient arrives in a SPARSE layout — the case
//! `nn.Embedding(sparse=True)` produces, where only the looked-up rows have
//! gradient (`torch/optim/sparse_adam.py:132-161`). It REQUIRES a sparse
//! gradient: a dense `p.grad` raises
//! `RuntimeError("SparseAdam does not support dense gradients, please
//! consider Adam instead")` (`torch/optim/sparse_adam.py:88-92`).
//!
//! ferrotorch mirrors this with [`ferrotorch_core::SparseGrad`]: the caller
//! registers the parameter's sparse gradient via
//! [`SparseAdam::set_sparse_grad`] (or [`SparseAdam::collect_sparse_grad_from_embedding`]
//! for the `nn::Embedding(sparse=True)` producer), and [`SparseAdam::step`]
//! runs the masked update. A parameter that has a DENSE `.grad` set but no
//! registered sparse grad is rejected with the same message as upstream.
//!
//! The masked update faithfully mirrors `torch.optim._functional.sparse_adam`
//! (`torch/optim/_functional.py:24-84`): the grad is coalesced (duplicate
//! indices summed, "the update is non-linear so indices must be unique",
//! `_functional.py:44`); the moment buffers are EMA'd ONLY at the coalesced
//! indices (`exp_avg.add_(make_sparse(exp_avg_update_values))`,
//! `_functional.py:65-72`); and the parameter rows are updated with the
//! sparse-Adam step size `step_size = lr * sqrt(bc2) / bc1` and
//! `param -= step_size * numer / (sqrt(v) + eps)` (`_functional.py:80-84`).
//! Note: this eps placement differs from dense Adam's `m_hat/(sqrt(v_hat)+eps)`.
//!
//! ## REQ status (per `.design/ferrotorch-optim/sparse_adam.md`)
//!
//! | REQ | Status | Evidence |
//! | --- | --- | --- |
//! | REQ-1 | SHIPPED | `SparseAdamConfig` at `sparse_adam.rs` mirrors `torch/optim/sparse_adam.py:14`; consumer: `ferrotorch/src/lib.rs:61` re-export. |
//! | REQ-2 | SHIPPED | `SparseAdam<T>` plus `impl Optimizer<T>` at `sparse_adam.rs` mirrors `torch.optim.SparseAdam` (`torch/optim/sparse_adam.py:13`); consumer: `ferrotorch/src/lib.rs:61` re-export plus `ferrotorch-nn/src/embedding.rs:20` documented chain. |
//! | REQ-3 | SHIPPED | sparse-COO gradient contract: `SparseAdam::step` requires a registered `ferrotorch_core::SparseGrad` and rejects a dense `.grad` with torch's exact message (`fn step` here, mirroring `torch/optim/sparse_adam.py:88-92`); the masked step mirrors `torch/optim/_functional.py:24-84`. Producer→consumer chain: `Embedding::sparse_grad` (`ferrotorch-nn/src/embedding.rs`) feeds `SparseAdam::collect_sparse_grad_from_embedding` (here) which calls `set_sparse_grad`; non-test production consumer: `SparseAdam::step` reads the registry. Closes #1463. |
//! | REQ-4 | SHIPPED | per-`ParamKey` state map at `sparse_adam.rs` mirrors `torch/optim/sparse_adam.py:98`; consumer: `ferrotorch/src/lib.rs:61` re-export. |
//! | REQ-5 | SHIPPED | bias-corrected sparse-Adam step (`step_size = lr*sqrt(bc2)/bc1`, `param -= step_size*numer/(sqrt(v)+eps)`) in `fn sparse_step` here, mirroring `torch/optim/_functional.py:80-84`; consumer: `ferrotorch/src/lib.rs:61` re-export. |
//! | REQ-6 | NOT-STARTED | `state_dict`/`load_state_dict` are no-op stubs here; blocked on follow-up (state-dict serialisation for the masked moment buffers is unfiled). |
//! | REQ-7 | SHIPPED | `FerrotorchError::NotImplementedOnCuda` early-return in `fn step` here (intentional divergence tracked by #1468); consumer: `ferrotorch/src/lib.rs:61` re-export. |

use std::collections::HashMap;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, SparseGrad};
use ferrotorch_nn::{Embedding, Parameter};

use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};
use crate::param_key::ParamKey;

/// Hyperparameters for the SparseAdam optimizer.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct SparseAdamConfig {
    /// Learning rate (default: 0.001).
    pub lr: f64,
    /// Exponential decay rates for first/second moment (default: (0.9, 0.999)).
    pub betas: (f64, f64),
    /// Numerical stability term (default: 1e-8).
    pub eps: f64,
}

impl Default for SparseAdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
        }
    }
}

impl SparseAdamConfig {
    /// Set the learning rate.
    #[must_use]
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    /// Set the exponential decay rates for the first and second moment estimates.
    #[must_use]
    pub fn with_betas(mut self, betas: (f64, f64)) -> Self {
        self.betas = betas;
        self
    }

    /// Set the numerical stability term added to the denominator.
    #[must_use]
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
}

#[derive(Debug)]
struct SparseAdamState {
    step_count: u64,
    /// Dense moment buffer; only the indices touched by a coalesced sparse
    /// grad are EMA'd each step, mirroring `state["exp_avg"]` updated via
    /// `exp_avg.add_(make_sparse(...))` (`torch/optim/_functional.py:67`).
    exp_avg: Vec<f64>,
    /// Dense second-moment buffer, masked-updated like `exp_avg`
    /// (`torch/optim/_functional.py:72`).
    exp_avg_sq: Vec<f64>,
}

/// SparseAdam optimizer.
///
/// Like Adam, but only updates moment estimates for elements with non-zero
/// gradients. This avoids unnecessary computation for large sparse parameter
/// matrices (e.g., embedding tables).
///
/// # Differences from Adam
///
/// - No weight decay support (sparse updates and weight decay don't mix well).
/// - No AMSGrad variant.
/// - Moment estimates are only updated at indices where the gradient is non-zero.
#[derive(Debug)]
pub struct SparseAdam<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: SparseAdamConfig,
    /// CL-1122: typed key replaces per-step `format!("g{}_p{}")` heap
    /// allocation (wire format preserved via Display, though SparseAdam
    /// state is not currently serialised).
    state: HashMap<ParamKey, SparseAdamState>,
    /// Per-`ParamKey` registered sparse gradients. This is the ferrotorch
    /// analog of `p.grad` being a sparse-COO tensor in
    /// `torch.optim.SparseAdam` (`torch/optim/sparse_adam.py:88-92`): a
    /// parameter whose sparse gradient is registered here gets the masked
    /// update; a parameter with a dense `.grad` and no entry here is
    /// rejected. Cleared per consumed key after each `step` so a stale
    /// gradient is not re-applied (mirrors autograd re-populating `p.grad`
    /// every backward pass).
    sparse_grads: HashMap<ParamKey, SparseGrad<T>>,
}

impl<T: Float> SparseAdam<T> {
    /// Create a new SparseAdam optimizer.
    pub fn new(params: Vec<Parameter<T>>, config: SparseAdamConfig) -> Self {
        let group = ParamGroup::new(params, config.lr);
        Self {
            param_groups: vec![group],
            config,
            state: HashMap::new(),
            sparse_grads: HashMap::new(),
        }
    }

    /// CL-1122: typed `ParamKey` replaces the legacy `String` key built
    /// by `format!("g{}_p{}")` on every step.
    fn param_key(gi: usize, pi: usize) -> ParamKey {
        ParamKey::new(gi, pi)
    }

    /// Register the sparse gradient for the parameter at
    /// `(group_idx, param_idx)`, the ferrotorch analog of assigning
    /// `p.grad = <sparse-COO tensor>` before calling
    /// `torch.optim.SparseAdam.step()`. The registered grad is consumed
    /// (and cleared) by the next [`SparseAdam::step`].
    ///
    /// This is the boundary by which a sparse-grad producer
    /// (`nn::Embedding(sparse=True)`) hands its gradient to the optimizer;
    /// see [`SparseAdam::collect_sparse_grad_from_embedding`] for the wired
    /// producer path.
    pub fn set_sparse_grad(&mut self, group_idx: usize, param_idx: usize, grad: SparseGrad<T>) {
        self.sparse_grads
            .insert(Self::param_key(group_idx, param_idx), grad);
    }

    /// Pull the sparse gradient from an `nn::Embedding(sparse=True)` and
    /// register it for the parameter at `(group_idx, param_idx)`.
    ///
    /// This is the production wiring of [`SparseAdam::set_sparse_grad`]:
    /// it consumes [`Embedding::sparse_grad`], the sparse-grad producer for
    /// the looked-up rows of the most recent forward/backward pass, and
    /// feeds it to this optimizer — mirroring the upstream
    /// `nn.Embedding(sparse=True)` → `torch.optim.SparseAdam` flow
    /// (`torch/optim/sparse_adam.py:132-161`).
    ///
    /// Returns `true` if a sparse gradient was available and registered,
    /// `false` if the embedding had no sparse grad yet (sparse mode off, no
    /// forward run, or no backward populated the weight grad). A `false`
    /// return registers nothing, so a subsequent dense `.grad` on the same
    /// parameter would be rejected by [`SparseAdam::step`] exactly as torch
    /// rejects dense gradients.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`Embedding::sparse_grad`] (e.g. a failed
    /// gradient read).
    pub fn collect_sparse_grad_from_embedding(
        &mut self,
        embedding: &Embedding<T>,
        group_idx: usize,
        param_idx: usize,
    ) -> FerrotorchResult<bool> {
        match embedding.sparse_grad()? {
            Some(grad) => {
                self.set_sparse_grad(group_idx, param_idx, grad);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Apply one masked sparse-Adam update for a single parameter, mirroring
    /// `torch.optim._functional.sparse_adam` (`torch/optim/_functional.py:24-84`).
    ///
    /// `grad` is coalesced first (duplicate indices summed —
    /// `_functional.py:44`). For each coalesced index `r` and each element
    /// `j` of its slab:
    ///
    /// - `exp_avg[r,j]   <- beta1*exp_avg[r,j]   + (1-beta1)*g`
    /// - `exp_avg_sq[r,j]<- beta2*exp_avg_sq[r,j]+ (1-beta2)*g^2`
    /// - `numer = exp_avg[r,j]`, `denom = sqrt(exp_avg_sq[r,j]) + eps`
    /// - `step_size = lr * sqrt(bc2) / bc1`  (`_functional.py:80-82`)
    /// - `param[r,j] -= step_size * numer / denom`  (`_functional.py:84`)
    ///
    /// Moment buffers and parameter elements OUTSIDE the coalesced indices
    /// are left untouched this step (the "sparse mask" — `_functional.py:65-72`).
    fn sparse_step(
        param: &Parameter<T>,
        grad: &SparseGrad<T>,
        state: &mut SparseAdamState,
        beta1: f64,
        beta2: f64,
        eps: f64,
        lr: f64,
    ) -> FerrotorchResult<()> {
        // "the update is non-linear so indices must be unique"
        // (_functional.py:44).
        let coalesced = grad.coalesce();
        let slab_size = coalesced.slab_size();

        // Empty grad: torch skips the update entirely (_functional.py:47-49).
        if coalesced.nnz() == 0 {
            return Ok(());
        }

        let tensor = param.tensor();
        if tensor.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "SparseAdam" });
        }

        let shape = tensor.shape().to_vec();
        let leading = *shape.first().unwrap_or(&0);
        // Validate the param layout matches the slab layout (leading dim is
        // the indexed dim; the rest is the slab).
        if shape.len() != 1 + coalesced.slab_shape().len()
            || shape[1..] != coalesced.slab_shape()[..]
        {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "SparseAdam: param shape {:?} incompatible with sparse-grad slab shape {:?}",
                    shape,
                    coalesced.slab_shape()
                ),
            });
        }
        for (k, &idx) in coalesced.indices().iter().enumerate() {
            if idx >= leading {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("SparseAdam: sparse-grad index {idx} >= {leading} (slot {k})"),
                });
            }
        }

        // step is recorded AFTER the increment, like torch
        // (`sparse_adam.py:112-114`).
        state.step_count += 1;
        let t = state.step_count as f64;
        let bias_correction1 = 1.0 - beta1.powf(t);
        let bias_correction2 = 1.0 - beta2.powf(t);
        // step_size = lr * sqrt(bc2) / bc1  (_functional.py:82).
        let step_size = lr * bias_correction2.sqrt() / bias_correction1;

        let values = coalesced.values();
        let mut data = tensor.data_vec()?;

        for (k, &idx) in coalesced.indices().iter().enumerate() {
            let row = idx * slab_size;
            let vslab = k * slab_size;
            for j in 0..slab_size {
                let g = num_traits::ToPrimitive::to_f64(&values[vslab + j]).ok_or_else(|| {
                    FerrotorchError::InvalidArgument {
                        message: "SparseAdam: sparse-grad value not representable as f64".into(),
                    }
                })?;
                let pos = row + j;
                // Masked moment EMA at this coordinate only.
                state.exp_avg[pos] = beta1 * state.exp_avg[pos] + (1.0 - beta1) * g;
                state.exp_avg_sq[pos] = beta2 * state.exp_avg_sq[pos] + (1.0 - beta2) * g * g;
                let numer = state.exp_avg[pos];
                let denom = state.exp_avg_sq[pos].sqrt() + eps;
                let update = step_size * numer / denom;
                let p = num_traits::ToPrimitive::to_f64(&data[pos]).ok_or_else(|| {
                    FerrotorchError::InvalidArgument {
                        message: "SparseAdam: param value not representable as f64".into(),
                    }
                })?;
                data[pos] =
                    T::from(p - update).ok_or_else(|| FerrotorchError::InvalidArgument {
                        message: "SparseAdam: updated param value not representable as T".into(),
                    })?;
            }
        }

        // SAFETY: `update_data` mutates the parameter's CPU storage through
        // the storage Arc. Invariants:
        //  1. `step(&mut self)` holds the only mutable handle to this
        //     optimiser and iterates params sequentially, so no other thread
        //     mutates this parameter concurrently.
        //  2. SparseAdam refuses CUDA tensors (early return above), so the
        //     storage is CPU-only and no live GPU view exists.
        //  3. `data` is an owned `Vec<T>` produced by `tensor.data_vec()`
        //     (a copy), so its allocation is independent of the parameter's
        //     storage — the write does not alias a borrow held here.
        //  4. The optimizer update is grad-free by contract (the
        //     `Optimizer::step` doc requires it run detached); torch wraps
        //     SparseAdam.step in `@torch.no_grad()` (`sparse_adam.py:63`),
        //     so no autograd node captures this storage.
        #[allow(
            clippy::undocumented_unsafe_blocks,
            reason = "SAFETY comment above documents the sole-writer CPU-only invariant; mirrors torch SparseAdam.step running under @torch.no_grad() (sparse_adam.py:63)"
        )]
        unsafe {
            tensor.update_data(&data)?;
        }
        Ok(())
    }
}

impl<T: Float> Optimizer<T> for SparseAdam<T> {
    fn step(&mut self) -> FerrotorchResult<()> {
        let (beta1, beta2) = self.config.betas;
        let eps = self.config.eps;

        // PASS 1 — validate-all-before-apply (atomicity, #1593).
        // torch.optim.SparseAdam collects every param's grad in one loop and
        // RAISES the dense-grad `RuntimeError` the moment it sees a non-sparse
        // `p.grad` — BEFORE `F.sparse_adam` writes ANY param
        // (`sparse_adam.py:85-92,116`). So on a dense grad anywhere, NO param
        // is mutated, even earlier params with a valid sparse grad. We mirror
        // that by checking every group/param FIRST (without consuming the
        // registered sparse grads and without mutating any param), returning
        // the rejection before pass 2 applies a single update.
        for gi in 0..self.param_groups.len() {
            for pi in 0..self.param_groups[gi].params.len() {
                let key = Self::param_key(gi, pi);
                let has_sparse_grad = self.sparse_grads.contains_key(&key);

                let param = &self.param_groups[gi].params[pi];
                let tensor = param.tensor();

                if has_sparse_grad {
                    // A param consumed by pass 2 must not be on CUDA; torch's
                    // device dispatch would likewise fail before any write.
                    if tensor.is_cuda() {
                        return Err(FerrotorchError::NotImplementedOnCuda { op: "SparseAdam" });
                    }
                    continue;
                }

                // No sparse grad registered. A DENSE `p.grad` is rejected
                // exactly as torch does (sparse_adam.py:88-92); a param with no
                // gradient at all is simply skipped in pass 2.
                if tensor.grad()?.is_some() {
                    return Err(FerrotorchError::InvalidArgument {
                        message:
                            "SparseAdam does not support dense gradients, please consider Adam instead"
                                .to_string(),
                    });
                }
            }
        }

        // PASS 2 — apply the masked update now that every grad is validated
        // sparse. No `Err` after this point depends on a not-yet-mutated param,
        // so the step is atomic-on-error (the only post-validation failures are
        // internal numeric/shape faults that torch would also surface).
        for gi in 0..self.param_groups.len() {
            let lr = self.param_groups[gi].lr;

            for pi in 0..self.param_groups[gi].params.len() {
                let key = Self::param_key(gi, pi);

                // Consume the registered sparse gradient (the ferrotorch analog
                // of `p.grad.is_sparse == True`), validated in pass 1.
                let Some(sparse_grad) = self.sparse_grads.remove(&key) else {
                    // Validated in pass 1 as having no grad — skip.
                    continue;
                };

                let param = &self.param_groups[gi].params[pi];
                let tensor = param.tensor();
                let numel = tensor.numel();
                // Clone the parameter handle (Arc clone) so the per-key
                // `sparse_step` can borrow `self.state` mutably without
                // colliding with the `self.param_groups` borrow.
                let param_handle = param.clone();

                let state = self.state.entry(key).or_insert_with(|| SparseAdamState {
                    step_count: 0,
                    exp_avg: vec![0.0; numel],
                    exp_avg_sq: vec![0.0; numel],
                });

                Self::sparse_step(&param_handle, &sparse_grad, state, beta1, beta2, eps, lr)?;
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self) -> FerrotorchResult<()> {
        for group in &mut self.param_groups {
            for param in &mut group.params {
                param.zero_grad()?;
            }
        }
        Ok(())
    }

    fn lr(&self) -> f64 {
        self.param_groups.first().map(|g| g.lr).unwrap_or(0.0)
    }

    fn set_lr(&mut self, lr: f64) {
        for group in &mut self.param_groups {
            group.lr = lr;
        }
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
        Ok(OptimizerState::default())
    }

    fn load_state_dict(&mut self, _state: &OptimizerState) -> FerrotorchResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::Tensor;
    use ferrotorch_core::storage::TensorStorage;

    /// `[leading, slab]`-shaped f64 parameter from a flat row-major buffer.
    fn make_param_2d(data: &[f64], leading: usize, slab: usize) -> Parameter<f64> {
        assert_eq!(data.len(), leading * slab);
        let t = Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![leading, slab], true)
            .unwrap();
        Parameter::new(t)
    }

    fn read(param: &Parameter<f64>) -> Vec<f64> {
        param.tensor().data_vec().unwrap()
    }

    /// Oracle: `torch.optim.SparseAdam` 2.11.0, single step, lr=0.1,
    /// betas=(0.9,0.999), eps=1e-8. Param `[3,3]` =
    /// `[[1,2,3],[4,5,6],[7,8,9]]`; sparse grad indices `[0,2,0]` (duplicate
    /// row 0 to exercise coalesce) with slab values rows
    /// `[0.1,0.2,0.3]`, `[0.4,0.5,0.6]`, `[0.05,0.05,0.05]`.
    /// Reproduce with:
    /// ```text
    /// p = nn.Parameter(torch.tensor([[1.,2,3],[4,5,6],[7,8,9]], dtype=torch.float64))
    /// idx = torch.tensor([[0,2,0]]);
    /// vals = torch.tensor([[.1,.2,.3],[.4,.5,.6],[.05,.05,.05]], dtype=torch.float64)
    /// p.grad = torch.sparse_coo_tensor(idx, vals, (3,3))
    /// torch.optim.SparseAdam([p], lr=.1, betas=(.9,.999), eps=1e-8).step()
    /// ```
    /// (`torch/optim/_functional.py:24-84`).
    #[test]
    fn sparse_adam_matches_torch_oracle_one_step_with_coalesce() {
        let param = make_param_2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        // indices [0,2,0] with the three slabs above (NOT coalesced — the
        // optimizer coalesces internally, matching _functional.py:44).
        let grad = SparseGrad::<f64>::new(
            vec![0, 2, 0],
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.05, 0.05, 0.05],
            vec![3],
        )
        .unwrap();

        let mut opt = SparseAdam::new(
            vec![param.clone()],
            SparseAdamConfig::default()
                .with_lr(0.1)
                .with_betas((0.9, 0.999))
                .with_eps(1e-8),
        );
        opt.set_sparse_grad(0, 0, grad);
        opt.step().unwrap();

        // Live torch.optim.SparseAdam 2.11.0 output (flattened row-major).
        let expected = [
            0.9000002108180662,
            1.9000001264909465,
            2.9000000903507086,
            4.0,
            5.0,
            6.0,
            6.900000079056879,
            7.900000063245513,
            8.9000000527046,
        ];
        let got = read(&opt.param_groups[0].params[0]);
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() <= 1e-12,
                "row-major[{i}]: expected {e:.17}, got {g:.17} (diff {:.2e})",
                (g - e).abs()
            );
        }
    }

    /// Oracle: same config, 3 sequential steps, param `[[5.0]]`, constant
    /// sparse grad row 0 = `[1.0]`. torch 2.11.0 trajectory below.
    #[test]
    fn sparse_adam_matches_torch_oracle_multi_step() {
        let param = make_param_2d(&[5.0], 1, 1);
        let mut opt = SparseAdam::new(
            vec![param.clone()],
            SparseAdamConfig::default()
                .with_lr(0.1)
                .with_betas((0.9, 0.999))
                .with_eps(1e-8),
        );
        let expected = [4.900000031622767, 4.800000053989034, 4.700000072255582];
        for &exp in &expected {
            let grad = SparseGrad::<f64>::new(vec![0], vec![1.0], vec![1]).unwrap();
            opt.set_sparse_grad(0, 0, grad);
            opt.step().unwrap();
            let got = read(&opt.param_groups[0].params[0])[0];
            assert!(
                (got - exp).abs() <= 1e-12,
                "expected {exp:.17}, got {got:.17}"
            );
        }
    }

    /// Untouched rows (index 1 above) keep their moments at zero and their
    /// param byte-stable: the masked update touches ONLY coalesced indices
    /// (`torch/optim/_functional.py:65-72`).
    #[test]
    fn sparse_adam_leaves_untouched_rows_byte_stable() {
        let param = make_param_2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let grad = SparseGrad::<f64>::new(vec![0], vec![0.5, 0.5], vec![2]).unwrap();
        let mut opt = SparseAdam::new(vec![param.clone()], SparseAdamConfig::default());
        opt.set_sparse_grad(0, 0, grad);
        opt.step().unwrap();
        let got = read(&opt.param_groups[0].params[0]);
        // Row 1 (indices 2,3) never indexed -> unchanged byte-for-byte.
        assert_eq!(got[2], 3.0, "row 1 elem 0 must be byte-stable");
        assert_eq!(got[3], 4.0, "row 1 elem 1 must be byte-stable");
        // Row 0 moved.
        assert_ne!(got[0], 1.0, "row 0 elem 0 should move");
        assert_ne!(got[1], 2.0, "row 0 elem 1 should move");
    }

    /// torch rejects a DENSE gradient with this exact message
    /// (`torch/optim/sparse_adam.py:88-92`).
    #[test]
    fn sparse_adam_rejects_dense_grad_like_torch() {
        let param = make_param_2d(&[1.0, 2.0, 3.0], 3, 1);
        // Dense grad set, NO sparse grad registered.
        let dense = Tensor::from_storage(
            TensorStorage::cpu(vec![0.1f64, 0.0, 0.2]),
            vec![3, 1],
            false,
        )
        .unwrap();
        param.tensor().set_grad(Some(dense)).unwrap();
        let mut opt = SparseAdam::new(vec![param], SparseAdamConfig::default());
        let err = opt.step().unwrap_err();
        match err {
            FerrotorchError::InvalidArgument { message } => {
                assert_eq!(
                    message,
                    "SparseAdam does not support dense gradients, please consider Adam instead",
                    "must mirror torch RuntimeError verbatim"
                );
            }
            other => panic!("expected dense-grad rejection, got {other:?}"),
        }
    }

    /// No grad at all (neither sparse nor dense) is a no-op for that param.
    #[test]
    fn sparse_adam_no_grad_is_noop() {
        let param = make_param_2d(&[1.0, 2.0], 2, 1);
        let mut opt = SparseAdam::new(vec![param.clone()], SparseAdamConfig::default());
        opt.step().unwrap();
        assert_eq!(read(&opt.param_groups[0].params[0]), vec![1.0, 2.0]);
    }

    #[test]
    fn sparse_adam_zero_grad_clears() {
        let param = make_param_2d(&[1.0, 2.0], 2, 1);
        let dense =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0f64, 1.0]), vec![2, 1], false).unwrap();
        param.tensor().set_grad(Some(dense)).unwrap();
        let mut opt = SparseAdam::new(vec![param], SparseAdamConfig::default());
        opt.zero_grad().unwrap();
        let g = opt.param_groups[0].params[0].tensor().grad().unwrap();
        assert!(g.is_none(), "grad should be None after zero_grad");
    }
}
