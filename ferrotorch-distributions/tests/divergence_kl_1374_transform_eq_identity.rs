//! Divergence audit of commit `0d61d0bdd` (#1374, recursion-based KL pairs:
//! Independent-Independent + TransformedDistribution-TransformedDistribution).
//!
//! FOCUS: the `Transform::transform_eq_key` fidelity flagged by the builder.
//!
//! `_kl_transformed_transformed` (`torch/distributions/kl.py:496-502`) guards:
//!     if p.transforms != q.transforms: raise NotImplementedError
//! where `!=` invokes each `Transform.__eq__`. For the UNPARAMETERISED
//! transforms whose `__eq__` is `isinstance(other, X)` (ExpTransform
//! `transforms.py:586`, Sigmoid :657, Softplus :683, Tanh :724, Abs :747, …),
//! two DISTINCT same-type instances ARE equal, so ferrotorch's `name()`-based
//! `transform_eq_key` default is faithful.
//!
//! BUT several transforms ferrotorch ships do NOT define their own `__eq__` and
//! therefore inherit the BASE `Transform.__eq__`, which is IDENTITY:
//!     `torch/distributions/transforms.py:148-149  def __eq__: return self is other`
//! These are: ReshapeTransform (class at transforms.py:500, no __eq__),
//! IndependentTransform (:422), CatTransform (:1081), StackTransform (:1223),
//! CorrCholeskyTransform (:864), CumulativeDistributionTransform (:1324).
//! For ALL of these, two DISTINCT instances — even with identical parameters —
//! compare UNEQUAL in torch, so `p.transforms != q.transforms` is TRUE and
//! `_kl_transformed_transformed` RAISES NotImplementedError.
//!
//! ferrotorch's `transform_eq_key` default (`transforms.rs:166-168`) returns
//! `self.name().to_string()`, and these transforms do NOT override it
//! (only AffineTransform `transforms.rs:302` and PowerTransform :733 do).
//! So two distinct `ReshapeTransform`/`IndependentTransform`/… with the same
//! name produce IDENTICAL fingerprints → the `pf != qf` guard in
//! `kl_recurse_pair` (`kl.rs:210`) is FALSE → ferrotorch recurses into the base
//! and RETURNS a finite KL where torch RAISES.
//!
//! Reference behavior from live torch 2.11.0 (this machine, 2026-05-27):
//!   ReshapeTransform((6,),(2,3)) distinct instances: tp.transforms==tq.transforms
//!     -> False; kl_divergence(tp,tq) -> NotImplementedError.
//!   The base Normal-Normal KL ferrotorch WRONGLY returns is
//!     [0.11634933457173241, 1.3068528194400546, 0.8181471805599453,
//!      0.1939320724517809, 0.25341894868579024, 0.32342903865933026].
//! Non-tautological per R-CHAR-3: the expected behavior (Err) traces to torch's
//! identity __eq__, NOT copied from the ferrotorch side.
//!
//! Tracking: #1576.

use ferrotorch_core::creation::from_slice;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_distributions::kl::kl_divergence;
use ferrotorch_distributions::{
    Distribution, ExpTransform, IndependentTransform, Normal, ReshapeTransform,
    TransformedDistribution,
};

fn vec2(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    from_slice(data, shape).unwrap()
}

fn boxed_normal(loc: &[f64], scale: &[f64]) -> Box<dyn Distribution<f64>> {
    Box::new(Normal::new(vec2(loc, &[loc.len()]), vec2(scale, &[scale.len()])).unwrap())
}

/// Divergence: ferrotorch's `TransformedDistribution::kl_recurse` uses
/// `transform_eq_key() == name()` for `ReshapeTransform`, so two DISTINCT
/// `ReshapeTransform` instances compare EQUAL — but
/// `torch/distributions/transforms.py:148-149` `Transform.__eq__` is
/// `self is other` (ReshapeTransform at :500 defines no `__eq__`), so torch
/// treats them as UNEQUAL and `_kl_transformed_transformed`
/// (`torch/distributions/kl.py:498`) RAISES NotImplementedError.
///
/// Input: two TransformedDistributions over Normal bases, each with its OWN
/// `ReshapeTransform((6,),(2,3))` instance.
/// Upstream: `kl_divergence` raises (transforms unequal under identity __eq__).
/// ferrotorch: returns the base Normal-Normal KL (Ok), a value torch never
/// produces for this pair.
/// Tracking: #1576.
#[ignore = "divergence: TD-TD KL equates distinct identity-__eq__ transforms (Reshape); torch raises; tracking #1576"]
#[test]
fn divergence_kl_td_distinct_reshape_must_raise() {
    let bp = vec![0.0, 1.0, 2.0, -1.0, 0.5, 3.0];
    let sp = vec![1.0, 2.0, 0.5, 1.5, 1.0, 2.0];
    let bq = vec![0.5, 0.0, 1.0, 0.0, 1.0, 2.0];
    let sq = vec![1.2, 1.0, 1.0, 2.0, 0.8, 1.5];

    let tp = TransformedDistribution::new(
        boxed_normal(&bp, &sp),
        vec![Box::new(ReshapeTransform::new(vec![6], vec![2, 3]).unwrap())],
    );
    let tq = TransformedDistribution::new(
        boxed_normal(&bq, &sq),
        vec![Box::new(ReshapeTransform::new(vec![6], vec![2, 3]).unwrap())],
    );

    let result = kl_divergence(&tp, &tq);
    assert!(
        result.is_err(),
        "torch raises NotImplementedError for two DISTINCT ReshapeTransform \
         instances (identity __eq__ at transforms.py:148-149); ferrotorch's \
         name()-based transform_eq_key wrongly equates them and returned {:?}",
        result.map(|t| t.data().unwrap().to_vec())
    );
}

/// Divergence: same identity-`__eq__` bug for `IndependentTransform`
/// (class at `torch/distributions/transforms.py:422`, no `__eq__`, inherits
/// `self is other`). Two distinct `IndependentTransform(ExpTransform, 1)`
/// instances are UNEQUAL in torch -> `_kl_transformed_transformed` raises.
/// ferrotorch's `transform_eq_key` default returns "IndependentTransform" for
/// both -> equal fingerprints -> wrongly returns the base Normal-Normal KL.
///
/// Verified live torch 2.11: tp.transforms==tq.transforms -> False;
/// kl_divergence(tp,tq) -> NotImplementedError.
/// Tracking: #1576.
#[ignore = "divergence: TD-TD KL equates distinct identity-__eq__ transforms (Independent); torch raises; tracking #1576"]
#[test]
fn divergence_kl_td_distinct_independent_transform_must_raise() {
    let tp = TransformedDistribution::new(
        boxed_normal(&[0.0, 1.0], &[1.0, 2.0]),
        vec![Box::new(IndependentTransform::<f64>::new(
            Box::new(ExpTransform),
            1,
        ))],
    );
    let tq = TransformedDistribution::new(
        boxed_normal(&[0.5, 0.0], &[1.5, 1.0]),
        vec![Box::new(IndependentTransform::<f64>::new(
            Box::new(ExpTransform),
            1,
        ))],
    );

    let result = kl_divergence(&tp, &tq);
    assert!(
        result.is_err(),
        "torch raises NotImplementedError for two DISTINCT IndependentTransform \
         instances (identity __eq__); ferrotorch returned {:?}",
        result.map(|t| t.data().unwrap().to_vec())
    );
}
