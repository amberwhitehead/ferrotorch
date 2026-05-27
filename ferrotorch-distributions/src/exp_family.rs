//! Generic exponential-family KL divergence (Bregman-divergence fallback).
//!
//! Mirrors PyTorch's `_kl_expfamily_expfamily` registration
//! (`torch/distributions/kl.py:282-300`), the *generic* `@register_kl`
//! fallback for `(ExponentialFamily, ExponentialFamily)` pairs. It fires for
//! two distributions of the **same** [`ExponentialFamily`](crate::ExponentialFamily)
//! subclass when no more-specific `(P, Q)` arm matched
//! (`kl.py:284`: `if type(p) is not type(q): raise NotImplementedError`).
//!
//! ## REQ status (per `.design/ferrotorch-distributions/exp_family.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`ExponentialFamily` trait `natural_params`/`log_normalizer`/`mean_params`) | SHIPPED | `pub trait ExponentialFamily<T>` (with the analytic `mean_params`) in `lib.rs` mirrors `torch/distributions/exp_family.py:11-66`; consumers: `impl ExponentialFamily for {Normal,Poisson,Gamma,Exponential,Beta,Bernoulli}` + `fn kl_expfamily_expfamily` below reads all three methods |
//! | REQ-2 (Bregman KL `kl_expfamily_expfamily`) | SHIPPED | `pub fn kl_expfamily_expfamily` below computes `A(η_q) − A(η_p) − Σ_i⟨η_q,i − η_p,i, ∇A(η_p)_i⟩` mirroring `torch/distributions/kl.py:282-297`; consumer: `fn try_kl_expfamily` (the dispatch hook) calls it, and `kl_divergence_dyn` in `kl.rs` calls `try_kl_expfamily` |
//! | REQ-3 (same-family dispatch hook) | SHIPPED | `pub fn try_kl_expfamily` below downcasts both operands to each registered exp-family type and fires only when they are the same type (`kl.py:284`); consumer: `kl_divergence_dyn` fall-through in `kl.rs` |
//!
//! ## Why analytic gradients instead of autograd (R-DEV-7)
//!
//! Upstream computes `∇A(η_p)` by reverse-mode autograd through
//! `_log_normalizer` (`kl.py:292`:
//! `torch.autograd.grad(lg_normal.sum(), p_nparams, create_graph=True)`).
//! ferrotorch's `_log_normalizer` impls evaluate on host-resident
//! `data_vec()` and build no autograd graph, so differentiating through them
//! is impossible. Rather than retrofit a differentiable host path, each
//! [`ExponentialFamily`](crate::ExponentialFamily) impl supplies
//! [`mean_params`](crate::ExponentialFamily::mean_params) — the same gradient
//! `∇A(η) = E[t(X)]` in **closed form**. The shipped tests verify the
//! Bregman KL built from these analytic gradients equals both the
//! specific-pair closed-form KL and live PyTorch to ~1e-10.

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::shape::broadcast_shapes;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::{
    Bernoulli, Beta, Distribution, Exponential, ExponentialFamily, Gamma, Normal, Poisson,
};

/// Compute `KL(p ‖ q)` for two distributions of the same exponential family
/// via the Bregman divergence of the log-normalizer.
///
/// Mirrors `_kl_expfamily_expfamily` (`torch/distributions/kl.py:282-297`):
/// ```text
/// p_nparams = p._natural_params           # η_p
/// q_nparams = q._natural_params           # η_q
/// lg_normal = p._log_normalizer(*p_nparams)        # A(η_p)
/// gradients = ∇A(η_p)                      # = p.mean_params() here (analytic)
/// result = q._log_normalizer(*q_nparams) - lg_normal     # A(η_q) − A(η_p)
/// for pnp, qnp, g in zip(p_nparams, q_nparams, gradients):
///     result -= _sum_rightmost((qnp - pnp) * g, len(q.event_shape))
/// ```
///
/// The `(η_q − η_p) · ∇A(η_p)` inner product runs over each natural-param
/// component `i`; for univariate families (`event_shape == []`) the
/// `_sum_rightmost(..., 0)` is the identity, so the term is added
/// element-wise. `p`'s tensors (`A(η_p)`, `η_p,i`, `∇A(η_p)_i`) share `p`'s
/// batch shape and `q`'s tensors (`A(η_q)`, `η_q,i`) share `q`'s batch shape;
/// the result is broadcast across `broadcast(p_batch, q_batch)` (mirroring
/// upstream tensor broadcasting in the formula).
///
/// # Errors
///
/// - `InvalidArgument` if `p` and `q` expose different numbers of natural
///   parameters (they cannot be the same family) or if a parameter tensor's
///   element count is inconsistent with its sibling components.
/// - propagates any error from the underlying `natural_params` /
///   `log_normalizer` / `mean_params` evaluations.
pub fn kl_expfamily_expfamily<T: Float>(
    p: &dyn ExponentialFamily<T>,
    q: &dyn ExponentialFamily<T>,
) -> FerrotorchResult<Tensor<T>> {
    let p_nparams = p.natural_params()?;
    let q_nparams = q.natural_params()?;
    if p_nparams.len() != q_nparams.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "kl_divergence(ExponentialFamily, ExponentialFamily): operands expose \
                 {} vs {} natural parameters; the Bregman fallback requires the same \
                 family (mirrors NotImplementedError at torch/distributions/kl.py:284)",
                p_nparams.len(),
                q_nparams.len()
            ),
        });
    }
    // A(η_p) and A(η_q): the log-normalizer evaluated at each operand's own η.
    let a_p = p.log_normalizer(&p_nparams)?;
    let a_q = q.log_normalizer(&q_nparams)?;
    // ∇A(η_p): the analytic mean parameters (expected sufficient statistics).
    let grad_p = p.mean_params()?;
    if grad_p.len() != p_nparams.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "kl_divergence(ExponentialFamily, ExponentialFamily): mean_params \
                 returned {} components but natural_params returned {}",
                grad_p.len(),
                p_nparams.len()
            ),
        });
    }

    // Host-resident views of every operand. Each operand's parameter tensors
    // share one batch shape (e.g. Normal::new requires loc.shape ==
    // scale.shape), so `a_p`/`grad_p[i]`/`p_nparams[i]` are all `p`-shaped and
    // `a_q`/`q_nparams[i]` are all `q`-shaped.
    let a_p_v = a_p.data_vec()?;
    let a_q_v = a_q.data_vec()?;
    let p_np_v: Vec<Vec<T>> = p_nparams
        .iter()
        .map(|t| t.data_vec())
        .collect::<FerrotorchResult<_>>()?;
    let q_np_v: Vec<Vec<T>> = q_nparams
        .iter()
        .map(|t| t.data_vec())
        .collect::<FerrotorchResult<_>>()?;
    let grad_p_v: Vec<Vec<T>> = grad_p
        .iter()
        .map(|t| t.data_vec())
        .collect::<FerrotorchResult<_>>()?;

    // Broadcast p's batch shape against q's batch shape.
    let plan = expfamily_broadcast_index_pairs(a_p.shape(), a_q.shape())?;

    let out: Vec<T> = plan
        .p_idx
        .iter()
        .zip(plan.q_idx.iter())
        .map(|(&pi, &qi)| {
            // result = A(η_q) − A(η_p)
            let mut result = a_q_v[qi] - a_p_v[pi];
            // − Σ_i (η_q,i − η_p,i) · ∇A(η_p)_i
            for comp in 0..p_np_v.len() {
                let eta_q = q_np_v[comp][qi];
                let eta_p = p_np_v[comp][pi];
                let g = grad_p_v[comp][pi];
                result = result - (eta_q - eta_p) * g;
            }
            result
        })
        .collect();

    Tensor::from_storage(TensorStorage::cpu(out), plan.out_shape, false)
}

/// Try the generic exponential-family KL fallback for an already-`dyn`-erased
/// `(p, q)` pair.
///
/// This is the dispatch hook `kl_divergence_dyn` (in `kl.rs`) falls through to
/// after the concrete `kl_dispatch` chain reports no registered formula. It
/// downcasts **both** operands to each registered exponential-family concrete
/// type and invokes [`kl_expfamily_expfamily`] only when both succeed for the
/// *same* type — the Rust analog of `_kl_expfamily_expfamily`'s
/// `if type(p) is not type(q): raise NotImplementedError`
/// (`torch/distributions/kl.py:284`). Because `Any::downcast_ref` cannot
/// express a `match any T: ExponentialFamily` arm, the registered families are
/// enumerated explicitly here (the same closed-crate pattern `kl_dispatch`
/// uses for concrete pairs, per `kl.md` REQ-8).
///
/// Returns `Ok(None)` when neither operand pair matches a registered
/// exponential family (the caller then raises the no-formula error), or
/// `Ok(Some(result))` when a same-family pair fired.
pub fn try_kl_expfamily<T: Float>(
    p: &dyn Distribution<T>,
    q: &dyn Distribution<T>,
) -> FerrotorchResult<Option<Tensor<T>>> {
    let pa = p.as_dist_any();
    let qa = q.as_dist_any();
    // Each arm: if BOTH operands are this concrete exponential family, run the
    // Bregman fallback. The `type(p) is type(q)` guard is implicit — both
    // downcasts must succeed for the *same* `T_i`.
    macro_rules! arm {
        ($ty:ty) => {
            if let (Some(pp), Some(qq)) = (pa.downcast_ref::<$ty>(), qa.downcast_ref::<$ty>()) {
                return Ok(Some(kl_expfamily_expfamily::<T>(pp, qq)?));
            }
        };
    }
    arm!(Normal<T>);
    arm!(Poisson<T>);
    arm!(Gamma<T>);
    arm!(Exponential<T>);
    arm!(Beta<T>);
    arm!(Bernoulli<T>);
    Ok(None)
}

/// Per-output-element broadcast index plan: maps each element of the broadcast
/// output to the flat row-major index into `p`'s and `q`'s parameter vectors.
///
/// A standalone copy of the broadcasting logic `kl.rs` uses for its concrete
/// arms (which keeps that helper private to `kl.rs`). Each exp-family operand's
/// parameter tensors share one batch shape, so broadcasting `p` against `q`
/// reduces to broadcasting one representative `p` shape against one `q` shape.
struct ExpFamilyBroadcastPlan {
    out_shape: Vec<usize>,
    p_idx: Vec<usize>,
    q_idx: Vec<usize>,
}

fn expfamily_broadcast_index_pairs(
    p_shape: &[usize],
    q_shape: &[usize],
) -> FerrotorchResult<ExpFamilyBroadcastPlan> {
    let out_shape = broadcast_shapes(p_shape, q_shape)?;
    let out_strides = row_major_strides(&out_shape);
    let p_strides = row_major_strides(p_shape);
    let q_strides = row_major_strides(q_shape);
    let numel: usize = out_shape.iter().product();
    let out_ndim = out_shape.len();

    let mut p_idx = Vec::with_capacity(numel);
    let mut q_idx = Vec::with_capacity(numel);
    for out_flat in 0..numel {
        p_idx.push(broadcast_flat_index(
            out_flat,
            &out_strides,
            out_ndim,
            p_shape,
            &p_strides,
        ));
        q_idx.push(broadcast_flat_index(
            out_flat,
            &out_strides,
            out_ndim,
            q_shape,
            &q_strides,
        ));
    }
    Ok(ExpFamilyBroadcastPlan {
        out_shape,
        p_idx,
        q_idx,
    })
}

/// Row-major (C-contiguous) strides for `shape`.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Map a flat output index to the flat source index, honoring NumPy-style
/// right-aligned broadcasting (size-1 source dims map to source index 0).
fn broadcast_flat_index(
    out_flat: usize,
    out_strides: &[usize],
    out_ndim: usize,
    src_shape: &[usize],
    src_strides: &[usize],
) -> usize {
    let src_ndim = src_shape.len();
    let mut src_flat = 0usize;
    let mut rem = out_flat;
    for (axis, &os) in out_strides.iter().enumerate() {
        let coord = rem / os;
        rem %= os;
        // Right-align: this out axis corresponds to src axis
        // `axis - (out_ndim - src_ndim)` if that is non-negative.
        if axis + src_ndim >= out_ndim {
            let src_axis = axis + src_ndim - out_ndim;
            if src_shape[src_axis] != 1 {
                src_flat += coord * src_strides[src_axis];
            }
        }
    }
    src_flat
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{scalar, tensor};

    // Live-torch oracle (PyTorch 2.11, float64), reproducible:
    //   import torch; torch.set_default_dtype(torch.float64)
    //   from torch.distributions import Normal, Gamma, Exponential, Beta, \
    //       Poisson, Bernoulli, kl_divergence
    //   from torch.distributions.kl import _kl_expfamily_expfamily
    //   # both kl_divergence(p,q) and _kl_expfamily_expfamily(p,q) agree to 1e-10
    // Each constant below is the torch value (NOT copied from the ferrotorch
    // side), satisfying R-CHAR-3.

    fn sc(x: f64) -> Tensor<f64> {
        scalar(x).unwrap()
    }

    /// Bregman KL == the value torch's `_kl_expfamily_expfamily` produces.
    macro_rules! assert_bregman {
        ($p:expr, $q:expr, $torch:expr, $name:literal) => {{
            let kl = kl_expfamily_expfamily(&$p, &$q).unwrap().item().unwrap();
            let t: f64 = $torch;
            assert!(
                (kl - t).abs() < 1e-9,
                "{}: ferrotorch Bregman KL = {kl}, torch = {t} (delta {})",
                $name,
                (kl - t).abs()
            );
        }};
    }

    #[test]
    fn bregman_normal() {
        // _kl_expfamily_expfamily(Normal(0,1), Normal(1,2)) = 0.4431471805599454
        assert_bregman!(
            Normal::new(sc(0.0), sc(1.0)).unwrap(),
            Normal::new(sc(1.0), sc(2.0)).unwrap(),
            0.4431471805599454,
            "Normal(0,1)->Normal(1,2)"
        );
        // Normal(-0.5,2) -> Normal(0.3,0.7) = 3.184871753052344
        assert_bregman!(
            Normal::new(sc(-0.5), sc(2.0)).unwrap(),
            Normal::new(sc(0.3), sc(0.7)).unwrap(),
            3.184871753052344,
            "Normal(-0.5,2)->Normal(0.3,0.7)"
        );
    }

    #[test]
    fn bregman_gamma() {
        // _kl_expfamily(Gamma(2,1.5), Gamma(3,0.5)) = 2.2328663781324742
        assert_bregman!(
            Gamma::new(sc(2.0), sc(1.5)).unwrap(),
            Gamma::new(sc(3.0), sc(0.5)).unwrap(),
            2.2328663781324742,
            "Gamma(2,1.5)->Gamma(3,0.5)"
        );
    }

    #[test]
    fn bregman_exponential() {
        // _kl_expfamily(Exponential(1.5), Exponential(0.5)) = 0.43194562200144293
        assert_bregman!(
            Exponential::new(sc(1.5)).unwrap(),
            Exponential::new(sc(0.5)).unwrap(),
            0.43194562200144293,
            "Exponential(1.5)->Exponential(0.5)"
        );
    }

    #[test]
    fn bregman_beta() {
        // _kl_expfamily(Beta(2,3), Beta(1,1)) = 0.23490664978800058
        assert_bregman!(
            Beta::new(sc(2.0), sc(3.0)).unwrap(),
            Beta::new(sc(1.0), sc(1.0)).unwrap(),
            0.23490664978800058,
            "Beta(2,3)->Beta(1,1)"
        );
        // Beta(0.5,0.5) -> Beta(2,5) = 3.7718388992077854
        assert_bregman!(
            Beta::new(sc(0.5), sc(0.5)).unwrap(),
            Beta::new(sc(2.0), sc(5.0)).unwrap(),
            3.7718388992077854,
            "Beta(0.5,0.5)->Beta(2,5)"
        );
    }

    #[test]
    fn bregman_poisson() {
        // _kl_expfamily(Poisson(2), Poisson(3.5)) = 0.38076842412915446
        assert_bregman!(
            Poisson::new(sc(2.0)).unwrap(),
            Poisson::new(sc(3.5)).unwrap(),
            0.38076842412915446,
            "Poisson(2)->Poisson(3.5)"
        );
    }

    #[test]
    fn bregman_bernoulli() {
        // _kl_expfamily(Bernoulli(0.3), Bernoulli(0.6)) = 0.1837868973868123
        assert_bregman!(
            Bernoulli::new(sc(0.3)).unwrap(),
            Bernoulli::new(sc(0.6)).unwrap(),
            0.1837868973868123,
            "Bernoulli(0.3)->Bernoulli(0.6)"
        );
    }

    #[test]
    fn bregman_normal_broadcast_p_vs_q() {
        // torch: Normal([0,1],[1,2]) -> Normal([0.5],[1.5])
        //   = [0.18324288588594217, 0.15676237199266352]
        let p = Normal::new(
            tensor(&[0.0f64, 1.0]).unwrap(),
            tensor(&[1.0f64, 2.0]).unwrap(),
        )
        .unwrap();
        let q = Normal::new(tensor(&[0.5f64]).unwrap(), tensor(&[1.5f64]).unwrap()).unwrap();
        let kl = kl_expfamily_expfamily(&p, &q).unwrap();
        assert_eq!(kl.shape(), &[2]);
        let v = kl.data_vec().unwrap();
        assert!((v[0] - 0.18324288588594217).abs() < 1e-9, "got {}", v[0]);
        assert!((v[1] - 0.15676237199266352).abs() < 1e-9, "got {}", v[1]);
    }

    #[test]
    fn try_hook_same_family_fires_and_mismatch_skips() {
        let n0: Normal<f64> = Normal::new(sc(0.0), sc(1.0)).unwrap();
        let n1: Normal<f64> = Normal::new(sc(1.0), sc(2.0)).unwrap();
        // Same family -> Some(..)
        let r = try_kl_expfamily::<f64>(&n0, &n1).unwrap();
        assert!(r.is_some(), "Normal-Normal must hit the exp-family hook");
        // Different families (Normal vs Gamma) -> None (no Bregman cross-family).
        let g: Gamma<f64> = Gamma::new(sc(2.0), sc(1.0)).unwrap();
        let r2 = try_kl_expfamily::<f64>(&n0, &g).unwrap();
        assert!(
            r2.is_none(),
            "cross-family exp pair must NOT fire (mirrors kl.py:284 NotImplementedError)"
        );
    }
}
