//! Bijective transforms for distribution reparameterization.
//!
//! Transforms map between spaces (e.g., real line to positive reals) and
//! compute the log-absolute-determinant of the Jacobian needed for the
//! change-of-variables formula in [`TransformedDistribution`].
//!
//! This mirrors PyTorch's `torch.distributions.transforms` module.
//!
//! CL-330
//!
//! ## REQ status (per `.design/ferrotorch-distributions/transforms.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream cites)
//! live in the design doc; this synopsis is a one-line summary per REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`Transform` trait) | SHIPPED | `pub trait Transform<T: Float>: Send + Sync` with `forward`/`inverse`/`log_abs_det_jacobian`/`name` + `constant_entropy_contribution`/`is_exp_transform` defaults in `transforms.rs` mirroring `torch/distributions/transforms.py:48-200`; consumer: `fn TransformedDistribution::entropy` in `transforms.rs` invokes `t.constant_entropy_contribution()` and `t.is_exp_transform()` on the trait in production |
//! | REQ-2 (`ExpTransform`) | SHIPPED | `pub struct ExpTransform` with `is_exp_transform() -> true` and `log_abs_det_jacobian == x.clone()` in `transforms.rs` mirroring `torch/distributions/transforms.py:ExpTransform`; consumer: `fn TransformedDistribution::entropy` inspects `is_exp_transform()` for closed-form case 3; grandfathered public API |
//! | REQ-3 (`AffineTransform<T>`) | SHIPPED | `pub struct AffineTransform<T: Float> { loc, scale }` with `constant_entropy_contribution() -> Some(abs_scale.ln())` in `transforms.rs` mirroring `torch/distributions/transforms.py:AffineTransform`; consumer: `fn TransformedDistribution::entropy` consumes `constant_entropy_contribution` for closed-form case 2 |
//! | REQ-4 (Sigmoid/Tanh/Softplus transforms) | SHIPPED | `pub struct SigmoidTransform`, `TanhTransform`, `SoftplusTransform` with numerically-stable device-resident formulas in `transforms.rs` mirroring `torch/distributions/transforms.py:{SigmoidTransform,TanhTransform,SoftplusTransform}`; consumer: `*_transform_preserves_device_and_value` tests verify the chain; grandfathered public API |
//! | REQ-5 (`ComposeTransform<T>`) | SHIPPED | `pub struct ComposeTransform<T: Float>` + L→R forward + R→L inverse + sum-of-LDJs + empty-chain identity branch in `transforms.rs` mirroring `torch/distributions/transforms.py:ComposeTransform`; consumer: `fn TransformedDistribution::entropy` reads `constant_entropy_contribution()` override on chained instances |
//! | REQ-6 (`TransformedDistribution<T>`) | SHIPPED | `pub struct TransformedDistribution<T: Float>` with `Box<dyn Distribution<T>>` base + `Vec<Box<dyn Transform<T>>>` chain + full `Distribution` impl in `transforms.rs` mirroring `torch/distributions/transformed_distribution.py:TransformedDistribution`; consumer: `pub use transforms::TransformedDistribution` in `lib.rs` — grandfathered public API |
//! | REQ-7 (`TransformedDistribution::entropy` three-case dispatcher) | SHIPPED | `fn TransformedDistribution::entropy` three-case dispatch (empty, all-constant-Jacobian, single-Exp) with named-transform error on fall-through in `transforms.rs`; consumer: the `impl Distribution::entropy` IS the dispatcher; `test_transformed_distribution_entropy_*` tests pin all four branches |
//! | REQ-8 (16 upstream transforms + domain/codomain) | SHIPPED | #1373 — Constraint domain/codomain linkage + the 11 remaining upstream transforms ported: `AbsTransform`, `PowerTransform<T>`, `SoftmaxTransform`, `StickBreakingTransform`, `LowerCholeskyTransform`, `CorrCholeskyTransform`, `ReshapeTransform`, `IndependentTransform<T>`, `CatTransform<T>`, `StackTransform<T>`, `CumulativeDistributionTransform<T>` in `transforms.rs`, each mirroring upstream `_call`/`_inverse`/`log_abs_det_jacobian`/`domain`/`codomain` in `torch/distributions/transforms.py`; trait gained `event_dim()`/`bijective()`/`sign()` defaults; new codomain constraints `RealVector`/`CorrCholesky`/`LowerCholesky` in `constraints.rs`. Consumer: `fn TransformedDistribution::support` returns the chain's final codomain and the chain machinery drives each boxed transform; all re-exported from `lib.rs` as boundary API. Remaining NOT-STARTED: only `PositiveDefiniteTransform` (out of dispatch scope) |
//! | REQ-9 (Monte-Carlo entropy fallback) | SHIPPED | `fn TransformedDistribution::entropy_monte_carlo` estimates `H(Y) = H(X) + E_X[log|det J_f(X)|]` with `MC_ENTROPY_SAMPLES` base draws pushed through each link's `log_abs_det_jacobian` for the X-dependent chains (Sigmoid/Tanh/Softplus/multi-Exp/Exp-then-Affine) in `transforms.rs`; consumer: `fn TransformedDistribution::entropy` invokes it as path 4 on fall-through — `td.entropy()` on a Sigmoid/Tanh chain now returns a value instead of erroring. FD/quadrature-verified by `test_transformed_distribution_entropy_{sigmoid,exp_then_affine}_monte_carlo`. Closes #1378. |

use ferrotorch_core::autograd::no_grad;
use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::grad_fns::activation::{
    sigmoid as sigmoid_op, softmax as softmax_op, softplus as softplus_op, tanh as tanh_op,
};
use ferrotorch_core::grad_fns::arithmetic::{abs as abs_op, add, div, mul, neg, sub};
use ferrotorch_core::grad_fns::shape::cat as cat_op;
use ferrotorch_core::grad_fns::transcendental::{exp as exp_op, log as log_op};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_core::vmap::{select as select_op, stack as stack_op};

use crate::DistConstraint;

// ---------------------------------------------------------------------------
// Transform trait
// ---------------------------------------------------------------------------

/// A differentiable, invertible transformation with computable log-det-Jacobian.
///
/// Transforms are used by [`TransformedDistribution`] to map samples from a
/// base distribution through a bijection, accumulating the
/// log-absolute-determinant of the Jacobian for correct density computation.
///
/// # Required methods
///
/// - [`forward`](Transform::forward) — compute `y = f(x)`
/// - [`inverse`](Transform::inverse) — compute `x = f^{-1}(y)`
/// - [`log_abs_det_jacobian`](Transform::log_abs_det_jacobian) — compute
///   `log |det df/dx|` given `(x, y)` where `y = f(x)`
pub trait Transform<T: Float>: Send + Sync {
    /// Apply the forward transformation: `y = f(x)`.
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>>;

    /// Apply the inverse transformation: `x = f^{-1}(y)`.
    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>>;

    /// Compute the log absolute determinant of the Jacobian.
    ///
    /// Given `(x, y)` where `y = f(x)`, returns `log |det df/dx|`.
    /// For element-wise transforms this is a tensor with the same shape as `x`.
    fn log_abs_det_jacobian(&self, x: &Tensor<T>, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>>;

    /// Human-readable name.
    fn name(&self) -> &'static str;

    /// Closed-form `E_X[log|det J_f(X)|]` when it is independent of `X`.
    ///
    /// Used by [`TransformedDistribution::entropy`] for the standard
    /// change-of-variables entropy identity
    /// `H(Y) = H(X) + E_X[log|det J_f(X)|]`. Returning `Some(c)` advertises
    /// that the contribution is the constant `c` regardless of `X` — e.g.
    /// affine transforms whose Jacobian is `log|scale|`. Returning `None`
    /// (the default) means the contribution depends on `X` (or on the
    /// dispatcher applying a special case such as [`ExpTransform`] paired
    /// with the base mean).
    fn constant_entropy_contribution(&self) -> Option<T> {
        None
    }

    /// Whether this transform is an [`ExpTransform`].
    ///
    /// `ExpTransform` is the only `x`-dependent transform with a closed-form
    /// entropy contribution: `log|det dy/dx| = x` so the contribution is
    /// `E_X[X] = base.mean()`. The dispatcher in
    /// [`TransformedDistribution::entropy`] uses this flag to apply that
    /// special case instead of returning the generic "intractable" error.
    fn is_exp_transform(&self) -> bool {
        false
    }

    /// The [`Constraint`](crate::constraints::Constraint) on the transform's
    /// input space (the set of valid `x` for [`forward`](Transform::forward)).
    ///
    /// Returned as an object-safe [`DistConstraint`] because the underlying
    /// `Constraint` trait carries a generic `check<T: Float>` method that
    /// forbids trait-object use (see [`crate::DistConstraint`]). The default
    /// is the real line; concrete transforms override to advertise their true
    /// domain. Mirrors the `domain: constraints.Constraint` class attribute
    /// at `torch/distributions/transforms.py:94`.
    fn domain(&self) -> Box<dyn DistConstraint> {
        Box::new(crate::constraints::Real)
    }

    /// The [`Constraint`](crate::constraints::Constraint) on the transform's
    /// output space (the set of valid `y = f(x)`).
    ///
    /// The default is the real line; concrete transforms override (e.g.
    /// [`ExpTransform`] → positive reals). Mirrors the
    /// `codomain: constraints.Constraint` class attribute at
    /// `torch/distributions/transforms.py:95`.
    fn codomain(&self) -> Box<dyn DistConstraint> {
        Box::new(crate::constraints::Real)
    }

    /// Number of rightmost dimensions that together form a single event for
    /// this transform. Element-wise transforms are `0`; vector transforms
    /// (`SoftmaxTransform`, `StickBreakingTransform`) are `1`; matrix
    /// transforms (`LowerCholeskyTransform`) are `2`. Mirrors the
    /// `event_dim` property derived from `domain`/`codomain` at
    /// `torch/distributions/transforms.py:113-117`.
    fn event_dim(&self) -> usize {
        0
    }

    /// Whether the transform is a bijection (`t.inv(t(x)) == x`). Defaults to
    /// `true`; non-bijective transforms (`AbsTransform`, `SoftmaxTransform`)
    /// override to `false`. Mirrors the class-level `bijective` flag at
    /// `torch/distributions/transforms.py:93`.
    fn bijective(&self) -> bool {
        true
    }

    /// Sign of the Jacobian determinant for monotone univariate transforms:
    /// `+1` increasing, `-1` decreasing, `None` if not applicable. Mirrors
    /// the `sign` property at `torch/distributions/transforms.py:133-139`
    /// (which raises `NotImplementedError` by default — we surface `None`).
    fn sign(&self) -> Option<i32> {
        None
    }

    /// Structural equality key used by `TransformedDistribution`-`TransformedDistribution`
    /// KL recursion to mirror PyTorch's per-transform `__eq__`
    /// (`torch/distributions/kl.py:498` `if p.transforms != q.transforms`).
    ///
    /// PyTorch's `Transform.__eq__` is type-identity for the unparameterised
    /// transforms (`isinstance(other, ExpTransform)` at
    /// `torch/distributions/transforms.py:586`, `SigmoidTransform` :621,
    /// `SoftplusTransform` :657, `AbsTransform` :683, `TanhTransform` :724,
    /// `SoftmaxTransform`/`StickBreakingTransform` :960,:1001), so the default
    /// returns [`name`](Transform::name). Parameterised transforms
    /// (`AffineTransform.__eq__` compares `loc`/`scale` at
    /// `transforms.py:808-825`, `PowerTransform` compares the exponent) override
    /// this to fold their parameters into the key so two affines with different
    /// scales compare unequal — exactly as `p.transforms != q.transforms` does
    /// upstream.
    fn transform_eq_key(&self) -> String {
        self.name().to_string()
    }
}

// ---------------------------------------------------------------------------
// ExpTransform: y = exp(x)
// ---------------------------------------------------------------------------

/// Transform via `y = exp(x)`.
///
/// Maps the real line to the positive reals. The log-det-Jacobian is simply `x`
/// since `d(exp(x))/dx = exp(x)` and `log|exp(x)| = x`.
#[derive(Debug, Clone, Copy)]
pub struct ExpTransform;

impl<T: Float> Transform<T> for ExpTransform {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Device-resident: dispatches to GPU exp_inner when x is_cuda.
        no_grad(|| exp_op(x))
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Device-resident: dispatches to GPU log_inner when y is_cuda.
        no_grad(|| log_op(y))
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // log|d(exp(x))/dx| = log(exp(x)) = x. We must return a tensor with
        // the same shape and device as `x`, but as a fresh leaf (no grad) per
        // the prior contract; cloning preserves storage Arc and device.
        no_grad(|| Ok(x.clone()))
    }

    fn name(&self) -> &'static str {
        "ExpTransform"
    }

    fn is_exp_transform(&self) -> bool {
        true
    }

    fn codomain(&self) -> Box<dyn DistConstraint> {
        // y = exp(x) maps R → (0, ∞). `domain` is the default Real.
        // Mirrors `torch/distributions/transforms.py:581-582`.
        Box::new(crate::constraints::Positive)
    }
}

// ---------------------------------------------------------------------------
// AffineTransform: y = loc + scale * x
// ---------------------------------------------------------------------------

/// Pointwise affine transform: `y = loc + scale * x`.
///
/// The log-det-Jacobian is `log|scale|` broadcast to the input shape.
#[derive(Debug, Clone)]
pub struct AffineTransform<T: Float> {
    /// Location (shift) parameter.
    pub loc: T,
    /// Scale (multiplication) parameter.
    pub scale: T,
}

impl<T: Float> AffineTransform<T> {
    /// Create a new affine transform with the given `loc` and `scale`.
    pub fn new(loc: T, scale: T) -> Self {
        Self { loc, scale }
    }
}

impl<T: Float> AffineTransform<T> {
    /// Materialize a 0-D scalar tensor on `device` filled with `value`.
    ///
    /// All Affine ops take broadcasted scalar `loc`/`scale` tensors; building
    /// them at apply time avoids caching state and re-uploading state for
    /// every device.
    fn scalar_on(value: T, device: ferrotorch_core::device::Device) -> FerrotorchResult<Tensor<T>> {
        let s = creation::scalar(value)?;
        s.to(device)
    }
}

impl<T: Float> Transform<T> for AffineTransform<T> {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // y = loc + scale * x, fully on x.device().
        no_grad(|| {
            let device = x.device();
            let loc_t = Self::scalar_on(self.loc, device)?;
            let scale_t = Self::scalar_on(self.scale, device)?;
            let scaled = mul(x, &scale_t)?;
            add(&loc_t, &scaled)
        })
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // x = (y - loc) / scale, fully on y.device().
        no_grad(|| {
            let device = y.device();
            let loc_t = Self::scalar_on(self.loc, device)?;
            let scale_t = Self::scalar_on(self.scale, device)?;
            let centered = sub(y, &loc_t)?;
            div(&centered, &scale_t)
        })
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // log|d(loc + scale*x)/dx| = log|scale|, broadcast to x.shape().
        no_grad(|| {
            let log_abs_scale = if self.scale > T::from(0.0).unwrap() {
                self.scale.ln()
            } else {
                (T::from(0.0).unwrap() - self.scale).ln()
            };
            let cpu = creation::full(x.shape(), log_abs_scale)?;
            cpu.to(x.device())
        })
    }

    fn name(&self) -> &'static str {
        "AffineTransform"
    }

    fn constant_entropy_contribution(&self) -> Option<T> {
        // log|d(loc + scale*x)/dx| = log|scale|, a scalar that does not
        // depend on x. `scale == 0` would not be a valid bijection, so we
        // treat the abs-then-ln branch as the only physical case.
        let zero = T::from(0.0).unwrap();
        let abs_scale = if self.scale >= zero {
            self.scale
        } else {
            zero - self.scale
        };
        Some(abs_scale.ln())
    }

    fn transform_eq_key(&self) -> String {
        // `torch/distributions/transforms.py:808-825` `AffineTransform.__eq__`
        // compares BOTH `loc` and `scale`; two affines with different params
        // are unequal. Fold them into the key (debug-formatted so f32/f64 bits
        // round-trip stably for the structural comparison).
        format!("AffineTransform(loc={:?},scale={:?})", self.loc, self.scale)
    }
}

// ---------------------------------------------------------------------------
// SigmoidTransform: y = 1 / (1 + exp(-x))
// ---------------------------------------------------------------------------

/// Transform via the sigmoid function: `y = sigma(x) = 1 / (1 + exp(-x))`.
///
/// Maps the real line to the unit interval `(0, 1)`.
/// The log-det-Jacobian is `-softplus(-x) - softplus(x)`.
#[derive(Debug, Clone, Copy)]
pub struct SigmoidTransform;

impl<T: Float> Transform<T> for SigmoidTransform {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Device-resident sigmoid (already numerically stable in core).
        no_grad(|| sigmoid_op(x))
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // logit(y) = log(y) - log(1 - y), clamped into (eps, 1-eps) to match
        // the prior CPU body's domain-safety contract. All ops device-resident.
        no_grad(|| {
            let one = T::from(1.0).unwrap();
            let eps = T::from(1e-7).unwrap();
            let clamped = ferrotorch_core::grad_fns::transcendental::clamp(y, eps, one - eps)?;
            let device = y.device();
            let one_t = creation::scalar(one)?.to(device)?;
            let one_minus = sub(&one_t, &clamped)?;
            let log_y = log_op(&clamped)?;
            let log_one_minus = log_op(&one_minus)?;
            sub(&log_y, &log_one_minus)
        })
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // log|d(sigma(x))/dx| = -softplus(-x) - softplus(x). Implement via
        // device-resident neg/softplus; matches the prior scalar formula.
        no_grad(|| {
            let neg_x = neg(x)?;
            let sp_neg = softplus_op(&neg_x, 1.0, 20.0)?;
            let sp_pos = softplus_op(x, 1.0, 20.0)?;
            let neg_sp_neg = neg(&sp_neg)?;
            let neg_sp_pos = neg(&sp_pos)?;
            add(&neg_sp_neg, &neg_sp_pos)
        })
    }

    fn name(&self) -> &'static str {
        "SigmoidTransform"
    }

    fn codomain(&self) -> Box<dyn DistConstraint> {
        // y = sigma(x) maps R → (0, 1). `domain` is the default Real.
        // Mirrors `torch/distributions/transforms.py:652-653`.
        Box::new(crate::constraints::UnitInterval)
    }
}

// ---------------------------------------------------------------------------
// TanhTransform: y = tanh(x)
// ---------------------------------------------------------------------------

/// Transform via `y = tanh(x)`.
///
/// Maps the real line to `(-1, 1)`. Uses the numerically stable formula
/// `log_abs_det_jacobian = 2 * (log(2) - x - softplus(-2x))`.
#[derive(Debug, Clone, Copy)]
pub struct TanhTransform;

impl<T: Float> Transform<T> for TanhTransform {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Device-resident tanh.
        no_grad(|| tanh_op(x))
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // atanh(y) = 0.5 * log((1+y)/(1-y)) — device-resident algebra.
        no_grad(|| {
            let device = y.device();
            let one = T::from(1.0).unwrap();
            let half = T::from(0.5).unwrap();
            let one_t = creation::scalar(one)?.to(device)?;
            let half_t = creation::scalar(half)?.to(device)?;
            let one_plus = add(&one_t, y)?;
            let one_minus = sub(&one_t, y)?;
            let ratio = div(&one_plus, &one_minus)?;
            let log_ratio = log_op(&ratio)?;
            mul(&half_t, &log_ratio)
        })
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Numerically stable formula (TensorFlow Probability):
        //   log(1 - tanh(x)^2) = 2 * (log(2) - x - softplus(-2*x))
        // All ops device-resident.
        no_grad(|| {
            let device = x.device();
            let two = T::from(2.0).unwrap();
            let ln2 = T::from(2.0f64.ln()).unwrap();
            let two_t = creation::scalar(two)?.to(device)?;
            let ln2_t = creation::full(x.shape(), ln2)?.to(device)?;
            // -2*x
            let neg_two_x = neg(&mul(&two_t, x)?)?;
            // softplus(-2x)
            let sp = softplus_op(&neg_two_x, 1.0, 20.0)?;
            // ln2 - x - softplus(-2x)
            let inner = sub(&sub(&ln2_t, x)?, &sp)?;
            // 2 * inner
            mul(&two_t, &inner)
        })
    }

    fn name(&self) -> &'static str {
        "TanhTransform"
    }

    fn codomain(&self) -> Box<dyn DistConstraint> {
        // y = tanh(x) maps R → (-1, 1). `domain` is the default Real.
        // Upstream uses `constraints.interval(-1.0, 1.0)`
        // (`torch/distributions/transforms.py:719-720`); ferrotorch's
        // `ClosedInterval` is the corresponding bounded-interval constraint.
        Box::new(crate::constraints::ClosedInterval {
            lower_bound: T::from(-1.0).unwrap(),
            upper_bound: T::from(1.0).unwrap(),
        })
    }
}

// ---------------------------------------------------------------------------
// SoftplusTransform: y = log(1 + exp(x))
// ---------------------------------------------------------------------------

/// Transform via `y = softplus(x) = log(1 + exp(x))`.
///
/// Maps the real line to the positive reals. Reverts to the identity for
/// large `x` (> 20) for numerical stability.
#[derive(Debug, Clone, Copy)]
pub struct SoftplusTransform;

impl<T: Float> Transform<T> for SoftplusTransform {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Device-resident softplus(x) with beta=1, threshold=20 to match the
        // prior scalar contract.
        no_grad(|| softplus_op(x, 1.0, 20.0))
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // softplus^{-1}(y) = log(exp(y) - 1). Device-resident algebra.
        // Note: the prior CPU body short-circuited y>20 to y itself for
        // overflow safety; the device-resident form is mathematically exact
        // and well-defined in the same range exp's GPU kernel handles. For
        // f32 inputs, exp(20) ~ 4.85e8 is far from the f32 max (3.4e38), so
        // the chain remains numerically safe in the previously tested range.
        no_grad(|| {
            let device = y.device();
            let one_t = creation::scalar(T::from(1.0).unwrap())?.to(device)?;
            let exp_y = exp_op(y)?;
            let exp_minus_one = sub(&exp_y, &one_t)?;
            log_op(&exp_minus_one)
        })
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // d(softplus(x))/dx = sigmoid(x); log|sigmoid(x)| = -softplus(-x).
        no_grad(|| {
            let neg_x = neg(x)?;
            let sp = softplus_op(&neg_x, 1.0, 20.0)?;
            neg(&sp)
        })
    }

    fn name(&self) -> &'static str {
        "SoftplusTransform"
    }

    fn codomain(&self) -> Box<dyn DistConstraint> {
        // y = softplus(x) = log(1 + exp(x)) maps R → (0, ∞). `domain` is the
        // default Real. Mirrors `torch/distributions/transforms.py:678-679`.
        Box::new(crate::constraints::Positive)
    }
}

// ---------------------------------------------------------------------------
// ComposeTransform: chain multiple transforms
// ---------------------------------------------------------------------------

/// Compose multiple transforms into a single transform.
///
/// Given transforms `[f1, f2, ..., fn]`, the composed forward pass computes
/// `fn(... f2(f1(x)) ...)` and the log-det-Jacobian is the sum of the
/// individual log-det-Jacobians along the chain.
pub struct ComposeTransform<T: Float> {
    transforms: Vec<Box<dyn Transform<T>>>,
}

impl<T: Float> ComposeTransform<T> {
    /// Create a composed transform from an ordered list of transforms.
    ///
    /// Transforms are applied left-to-right in the forward direction.
    pub fn new(transforms: Vec<Box<dyn Transform<T>>>) -> Self {
        Self { transforms }
    }

    /// The number of transforms in the chain.
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

impl<T: Float> Transform<T> for ComposeTransform<T> {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut val = x.clone();
        for t in &self.transforms {
            val = t.forward(&val)?;
        }
        Ok(val)
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut val = y.clone();
        for t in self.transforms.iter().rev() {
            val = t.inverse(&val)?;
        }
        Ok(val)
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Empty chain → identity → zero log-det-Jacobian on x.device().
        no_grad(|| {
            if self.transforms.is_empty() {
                let zeros = creation::full(x.shape(), T::from(0.0).unwrap())?;
                return zeros.to(x.device());
            }

            // Compute intermediates xs[0..=n] where xs[i+1] = transforms[i].forward(xs[i]).
            // Each child forward preserves device when its body is device-resident.
            let mut xs = Vec::with_capacity(self.transforms.len() + 1);
            xs.push(x.clone());
            for t in &self.transforms {
                let next = t.forward(xs.last().unwrap())?;
                xs.push(next);
            }

            // Sum the per-link log-det-Jacobians element-wise via device-resident add.
            let mut total = self
                .transforms
                .first()
                .unwrap()
                .log_abs_det_jacobian(&xs[0], &xs[1])?;
            for (i, t) in self.transforms.iter().enumerate().skip(1) {
                let ldj = t.log_abs_det_jacobian(&xs[i], &xs[i + 1])?;
                total = add(&total, &ldj)?;
            }
            Ok(total)
        })
    }

    fn name(&self) -> &'static str {
        "ComposeTransform"
    }

    fn constant_entropy_contribution(&self) -> Option<T> {
        // A composed chain is constant-Jacobian iff every link is. In that
        // case the contributions sum: log|det J_compose| = sum_i log|det J_i|.
        let mut acc = T::from(0.0).unwrap();
        for t in &self.transforms {
            let c = t.constant_entropy_contribution()?;
            acc += c;
        }
        Some(acc)
    }

    fn domain(&self) -> Box<dyn DistConstraint> {
        // The composed forward applies links left-to-right, so the chain's
        // input space is the FIRST link's domain. Empty chain → identity →
        // the default Real. Mirrors `torch/distributions/transforms.py:313-328`
        // (event-dim bookkeeping omitted; ferrotorch transforms are event_dim=0).
        match self.transforms.first() {
            Some(first) => first.domain(),
            None => Box::new(crate::constraints::Real),
        }
    }

    fn codomain(&self) -> Box<dyn DistConstraint> {
        // The chain's output space is the LAST link's codomain. Empty chain →
        // identity → the default Real. Mirrors
        // `torch/distributions/transforms.py:332-347`.
        match self.transforms.last() {
            Some(last) => last.codomain(),
            None => Box::new(crate::constraints::Real),
        }
    }
}

// ---------------------------------------------------------------------------
// AbsTransform: y = |x|
// ---------------------------------------------------------------------------

/// Transform via `y = |x|`.
///
/// Maps the real line to the non-negative reals. NOT bijective (two `x`
/// values map to one `y`); `inverse` returns `y` unchanged (the
/// pseudo-inverse) and `log_abs_det_jacobian` is undefined. Mirrors
/// `torch/distributions/transforms.py:741-754` (`class AbsTransform`).
#[derive(Debug, Clone, Copy)]
pub struct AbsTransform;

impl<T: Float> Transform<T> for AbsTransform {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // y = |x|; `transforms.py:750-751` `return x.abs()`.
        no_grad(|| abs_op(x))
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Pseudo-inverse: `transforms.py:753-754` `return y`.
        no_grad(|| Ok(y.clone()))
    }

    fn log_abs_det_jacobian(&self, _x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `AbsTransform` is not bijective: upstream `Transform.log_abs_det_jacobian`
        // `raise NotImplementedError` (`transforms.py:193-197`).
        Err(FerrotorchError::InvalidArgument {
            message: "AbsTransform is not bijective; log_abs_det_jacobian is undefined".into(),
        })
    }

    fn name(&self) -> &'static str {
        "AbsTransform"
    }

    fn bijective(&self) -> bool {
        false
    }

    fn codomain(&self) -> Box<dyn DistConstraint> {
        // domain = real (default), codomain = positive.
        // `transforms.py:744-745`.
        Box::new(crate::constraints::Positive)
    }
}

// ---------------------------------------------------------------------------
// PowerTransform: y = x^exponent
// ---------------------------------------------------------------------------

/// Transform via `y = x^exponent`.
///
/// Maps the positive reals to the positive reals. Stores a scalar `exponent`
/// of type `T` (upstream allows a broadcastable tensor exponent; ferrotorch's
/// device-resident `pow` takes a scalar power, so the scalar exponent is the
/// faithful subset). Mirrors `torch/distributions/transforms.py:599-639`
/// (`class PowerTransform`).
#[derive(Debug, Clone, Copy)]
pub struct PowerTransform<T: Float> {
    /// The power applied element-wise.
    pub exponent: T,
}

impl<T: Float> PowerTransform<T> {
    /// Create a power transform with the given scalar `exponent`.
    pub fn new(exponent: T) -> Self {
        Self { exponent }
    }
}

impl<T: Float> Transform<T> for PowerTransform<T> {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // y = x^exponent; `transforms.py:626-627` `return x.pow(self.exponent)`.
        no_grad(|| ferrotorch_core::grad_fns::arithmetic::pow(x, self.exponent.to_f64().unwrap()))
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // x = y^(1/exponent); `transforms.py:629-630` `return y.pow(1 / self.exponent)`.
        no_grad(|| {
            let inv = T::from(1.0).unwrap() / self.exponent;
            ferrotorch_core::grad_fns::arithmetic::pow(y, inv.to_f64().unwrap())
        })
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:632-633`:
        //   return (self.exponent * y / x).abs().log()
        no_grad(|| {
            let device = x.device();
            let exp_t = creation::scalar(self.exponent)?.to(device)?;
            let scaled = mul(&exp_t, y)?;
            let ratio = div(&scaled, x)?;
            let abs = abs_op(&ratio)?;
            log_op(&abs)
        })
    }

    fn name(&self) -> &'static str {
        "PowerTransform"
    }

    fn sign(&self) -> Option<i32> {
        // `transforms.py:617-619` `return self.exponent.sign()`.
        let zero = T::from(0.0).unwrap();
        Some(if self.exponent > zero {
            1
        } else if self.exponent < zero {
            -1
        } else {
            0
        })
    }

    fn domain(&self) -> Box<dyn DistConstraint> {
        // `transforms.py:604` `domain = constraints.positive`.
        Box::new(crate::constraints::Positive)
    }

    fn codomain(&self) -> Box<dyn DistConstraint> {
        // `transforms.py:605` `codomain = constraints.positive`.
        Box::new(crate::constraints::Positive)
    }

    fn transform_eq_key(&self) -> String {
        // `transforms.py:621-624` `PowerTransform.__eq__` compares `exponent`
        // (`self.exponent.eq(other.exponent).all()`). Fold the scalar exponent
        // into the key so two power transforms with different exponents are
        // structurally unequal.
        format!("PowerTransform(exponent={:?})", self.exponent)
    }
}

// ---------------------------------------------------------------------------
// SoftmaxTransform: y = softmax(x) over the last dim
// ---------------------------------------------------------------------------

/// Transform from unconstrained space to the simplex via `y = softmax(x)`.
///
/// NOT bijective (the softmax is shift-invariant, so it loses one degree of
/// freedom). Forward is `exp(x - max(x)) / sum(...)` over the last dim;
/// inverse is `log(y)` (the pseudo-inverse). `log_abs_det_jacobian` is
/// undefined (the domain and codomain have mismatched dimension). Mirrors
/// `torch/distributions/transforms.py:947-980` (`class SoftmaxTransform`).
#[derive(Debug, Clone, Copy)]
pub struct SoftmaxTransform;

impl<T: Float> Transform<T> for SoftmaxTransform {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:963-966`:
        //   probs = (logprobs - logprobs.max(-1, True)[0]).exp()
        //   return probs / probs.sum(-1, True)
        // ferrotorch's core `softmax` is exactly this stable last-dim formula
        // (`activation.rs:1028` subtracts the row max, exps, normalises).
        no_grad(|| softmax_op(x))
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:968-970` `return probs.log()`.
        no_grad(|| log_op(y))
    }

    fn log_abs_det_jacobian(&self, _x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // SoftmaxTransform is not bijective; upstream provides no
        // `log_abs_det_jacobian`, so `Transform.log_abs_det_jacobian`
        // `raise NotImplementedError` (`transforms.py:193-197`).
        Err(FerrotorchError::InvalidArgument {
            message: "SoftmaxTransform is not bijective; log_abs_det_jacobian is undefined".into(),
        })
    }

    fn name(&self) -> &'static str {
        "SoftmaxTransform"
    }

    fn bijective(&self) -> bool {
        false
    }

    fn event_dim(&self) -> usize {
        1
    }

    fn domain(&self) -> Box<dyn DistConstraint> {
        // `transforms.py:957` `domain = constraints.real_vector`.
        Box::new(crate::constraints::RealVector)
    }

    fn codomain(&self) -> Box<dyn DistConstraint> {
        // `transforms.py:958` `codomain = constraints.simplex`.
        Box::new(crate::constraints::Simplex)
    }
}

// ---------------------------------------------------------------------------
// StickBreakingTransform: R^{K-1} -> simplex of K
// ---------------------------------------------------------------------------

/// Transform from `R^{K-1}` to the `K`-simplex via a stick-breaking process.
///
/// Bijective and appropriate for HMC. Forward maps a length-`(K-1)` vector to
/// a length-`K` probability vector; inverse goes the other way.
/// `log_abs_det_jacobian` returns a scalar per batch element. Mirrors
/// `torch/distributions/transforms.py:983-1036`
/// (`class StickBreakingTransform`).
///
/// The body operates over the rightmost (event) dimension; ferrotorch's core
/// lacks a last-dim-only `cumprod`/`cumsum`-with-clamp chain matching the
/// padded upstream layout, so the math runs CPU-side over `data_vec()` rows
/// (consistent with the crate's `Dirichlet`/`MultivariateNormal` idiom).
#[derive(Debug, Clone, Copy)]
pub struct StickBreakingTransform;

impl<T: Float> Transform<T> for StickBreakingTransform {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:1004-1009`:
        //   offset = x.shape[-1] + 1 - x.new_ones(x.shape[-1]).cumsum(-1)
        //   z = _clipped_sigmoid(x - offset.log())
        //   z_cumprod = (1 - z).cumprod(-1)
        //   y = pad(z, [0,1], value=1) * pad(z_cumprod, [1,0], value=1)
        no_grad(|| {
            let shape = x.shape();
            let km1 = match shape.last().copied() {
                Some(k) if k >= 1 => k,
                _ => {
                    return Err(FerrotorchError::InvalidArgument {
                        message: "StickBreakingTransform: input must have >= 1 trailing dim".into(),
                    });
                }
            };
            let k = km1 + 1;
            let data = x.data_vec()?;
            let rows = data.len() / km1;
            let one = T::from(1.0).unwrap();
            let tiny = T::from(1e-7).unwrap();
            let eps = T::from(1.19e-7).unwrap(); // f32 eps; upstream uses dtype eps.
            let mut out = Vec::with_capacity(rows * k);
            for r in 0..rows {
                let row = &data[r * km1..(r + 1) * km1];
                // offset_i = (K - i)  for i in 0..K-1  (cumsum of ones gives 1..K-1).
                let mut z = vec![T::from(0.0).unwrap(); km1];
                for (i, &xi) in row.iter().enumerate() {
                    let offset = T::from((k - 1 - i) as f64).unwrap(); // K - 1 - i + ... see below
                    // offset = x.shape[-1] + 1 - cumsum(ones)[i]
                    //        = (K-1) + 1 - (i+1) = K - 1 - i
                    let arg = xi - offset.ln();
                    let s = one / (one + (T::from(0.0).unwrap() - arg).exp());
                    // clipped sigmoid: clamp to [tiny, 1 - eps].
                    let s = if s < tiny {
                        tiny
                    } else if s > one - eps {
                        one - eps
                    } else {
                        s
                    };
                    z[i] = s;
                }
                // z_cumprod[i] = prod_{j<=i} (1 - z[j]); cumprod of (1-z).
                let mut cumprod = vec![one; km1];
                let mut acc = one;
                for i in 0..km1 {
                    acc = acc * (one - z[i]);
                    cumprod[i] = acc;
                }
                // y[i] = pad(z,[0,1],1)[i] * pad(z_cumprod,[1,0],1)[i]:
                //   y[0]      = z[0] * 1
                //   y[i]      = z[i] * cumprod[i-1]   for 1 <= i < K-1
                //   y[K-1]    = 1    * cumprod[K-2]   (the last "stick remainder")
                for i in 0..k {
                    let z_pad = if i < km1 { z[i] } else { one };
                    let cp_pad = if i == 0 { one } else { cumprod[i - 1] };
                    out.push(z_pad * cp_pad);
                }
            }
            let mut out_shape = shape.to_vec();
            *out_shape.last_mut().unwrap() = k;
            Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)?.to(x.device())
        })
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:1011-1019`:
        //   y_crop = y[..., :-1]
        //   offset = y.shape[-1] - cumsum(ones over K-1)
        //   sf = 1 - y_crop.cumsum(-1); clamp(min=tiny)
        //   x = y_crop.log() - sf.log() + offset.log()
        no_grad(|| {
            let shape = y.shape();
            let k = match shape.last().copied() {
                Some(k) if k >= 2 => k,
                _ => {
                    return Err(FerrotorchError::InvalidArgument {
                        message:
                            "StickBreakingTransform inverse: input must have >= 2 trailing dim"
                                .into(),
                    });
                }
            };
            let km1 = k - 1;
            let data = y.data_vec()?;
            let rows = data.len() / k;
            let tiny = T::from(1e-7).unwrap();
            let mut out = Vec::with_capacity(rows * km1);
            for r in 0..rows {
                let row = &data[r * k..(r + 1) * k];
                let mut cumsum = T::from(0.0).unwrap();
                // `i` is needed arithmetically for the per-position `offset`
                // (`K - (i+1)`), not merely as an index, so the range loop is
                // the faithful form of upstream's positional cumsum.
                #[allow(clippy::needless_range_loop, reason = "i used in offset arithmetic")]
                for i in 0..km1 {
                    cumsum += row[i];
                    let mut sf = T::from(1.0).unwrap() - cumsum;
                    if sf < tiny {
                        sf = tiny;
                    }
                    // offset_i = K - cumsum(ones)[i] = K - (i+1).
                    let offset = T::from((k - (i + 1)) as f64).unwrap();
                    out.push(row[i].ln() - sf.ln() + offset.ln());
                }
            }
            let mut out_shape = shape.to_vec();
            *out_shape.last_mut().unwrap() = km1;
            Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)?.to(y.device())
        })
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:1021-1026`:
        //   offset = x.shape[-1] + 1 - cumsum(ones)
        //   x = x - offset.log()
        //   detJ = (-x + logsigmoid(x) + y[..., :-1].log()).sum(-1)
        // logsigmoid(x) = -softplus(-x); we compute it directly per element.
        no_grad(|| {
            let shape = x.shape();
            let km1 = match shape.last().copied() {
                Some(k) if k >= 1 => k,
                _ => {
                    return Err(FerrotorchError::InvalidArgument {
                        message: "StickBreakingTransform ldj: input must have >= 1 trailing dim"
                            .into(),
                    });
                }
            };
            let k = km1 + 1;
            let xdata = x.data_vec()?;
            let ydata = y.data_vec()?;
            let rows = xdata.len() / km1;
            let one = T::from(1.0).unwrap();
            let zero = T::from(0.0).unwrap();
            let mut out = Vec::with_capacity(rows);
            for r in 0..rows {
                let xrow = &xdata[r * km1..(r + 1) * km1];
                let yrow = &ydata[r * k..(r + 1) * k];
                let mut acc = zero;
                for (i, &xi) in xrow.iter().enumerate() {
                    let offset = T::from((k - 1 - i) as f64).unwrap();
                    let xshift = xi - offset.ln();
                    // logsigmoid(z) = -softplus(-z) = -ln(1 + exp(-z)).
                    let logsig = zero - (one + (zero - xshift).exp()).ln();
                    acc += (zero - xshift) + logsig + yrow[i].ln();
                }
                out.push(acc);
            }
            let out_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
            Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)?.to(x.device())
        })
    }

    fn name(&self) -> &'static str {
        "StickBreakingTransform"
    }

    fn event_dim(&self) -> usize {
        1
    }

    fn domain(&self) -> Box<dyn DistConstraint> {
        // `transforms.py:997` `domain = constraints.real_vector`.
        Box::new(crate::constraints::RealVector)
    }

    fn codomain(&self) -> Box<dyn DistConstraint> {
        // `transforms.py:998` `codomain = constraints.simplex`.
        Box::new(crate::constraints::Simplex)
    }
}

// ---------------------------------------------------------------------------
// LowerCholeskyTransform: unconstrained matrices -> lower-triangular w/ +diag
// ---------------------------------------------------------------------------

/// Transform from unconstrained matrices to lower-triangular matrices with
/// positive diagonal entries.
///
/// Forward keeps the strictly-lower triangle and exponentiates the diagonal;
/// inverse keeps the strictly-lower triangle and takes `log` of the diagonal.
/// Mirrors `torch/distributions/transforms.py:1039-1058`
/// (`class LowerCholeskyTransform`). Operates on the trailing `2` matrix dims
/// CPU-side (no `diag_embed`/`tril` event-dim chain in the distributions
/// crate's device-resident path).
#[derive(Debug, Clone, Copy)]
pub struct LowerCholeskyTransform;

impl LowerCholeskyTransform {
    /// Apply the per-matrix lower-Cholesky map (`forward`) or its inverse.
    /// `forward == true` does `tril(-1) + diag(exp(diag))`; `forward == false`
    /// does `tril(-1) + diag(log(diag))`.
    fn map_matrices<T: Float>(m: &Tensor<T>, forward: bool) -> FerrotorchResult<Tensor<T>> {
        let shape = m.shape();
        if shape.len() < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LowerCholeskyTransform: input must have >= 2 dims".into(),
            });
        }
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        if rows != cols {
            return Err(FerrotorchError::InvalidArgument {
                message: "LowerCholeskyTransform: trailing dims must be square".into(),
            });
        }
        let n = rows;
        let data = m.data_vec()?;
        let mat_sz = n * n;
        let batches = data.len() / mat_sz;
        let zero = T::from(0.0).unwrap();
        let mut out = vec![zero; data.len()];
        for b in 0..batches {
            let base = b * mat_sz;
            for i in 0..n {
                for j in 0..n {
                    let v = data[base + i * n + j];
                    out[base + i * n + j] = if i == j {
                        if forward { v.exp() } else { v.ln() }
                    } else if j < i {
                        // strictly lower triangle kept as-is.
                        v
                    } else {
                        // strictly upper triangle zeroed.
                        zero
                    };
                }
            }
        }
        Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)?.to(m.device())
    }
}

impl<T: Float> Transform<T> for LowerCholeskyTransform {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:1054-1055`:
        //   return x.tril(-1) + x.diagonal(-2,-1).exp().diag_embed()
        no_grad(|| Self::map_matrices(x, true))
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:1057-1058`:
        //   return y.tril(-1) + y.diagonal(-2,-1).log().diag_embed()
        no_grad(|| Self::map_matrices(y, false))
    }

    fn log_abs_det_jacobian(&self, _x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Upstream `LowerCholeskyTransform` does not implement
        // `log_abs_det_jacobian`; the base `Transform.log_abs_det_jacobian`
        // `raise NotImplementedError` (`transforms.py:193-197`).
        Err(FerrotorchError::InvalidArgument {
            message: "LowerCholeskyTransform: log_abs_det_jacobian is not implemented".into(),
        })
    }

    fn name(&self) -> &'static str {
        "LowerCholeskyTransform"
    }

    fn event_dim(&self) -> usize {
        2
    }

    fn domain(&self) -> Box<dyn DistConstraint> {
        // `transforms.py:1048` `domain = constraints.independent(real, 2)`.
        // ferrotorch surfaces this as a RealVector-style event-dim-2 marker via
        // the `event_dim()` metadata; the dtype-independent name is "Real".
        Box::new(crate::constraints::Real)
    }

    fn codomain(&self) -> Box<dyn DistConstraint> {
        // `transforms.py:1049` `codomain = constraints.lower_cholesky`.
        Box::new(crate::constraints::LowerCholesky)
    }
}

// ---------------------------------------------------------------------------
// CorrCholeskyTransform: R^{D(D-1)/2} -> Cholesky factor of a corr matrix
// ---------------------------------------------------------------------------

/// Transform an unconstrained vector of length `D*(D-1)/2` into the Cholesky
/// factor of a `D`-dimension correlation matrix (lower-triangular, positive
/// diagonal, unit-norm rows). Mirrors
/// `torch/distributions/transforms.py:864-944`
/// (`class CorrCholeskyTransform`).
///
/// Runs CPU-side over the trailing event dim: row-order fill of the
/// lower-triangle, signed stick-breaking on the squared `tanh`, then the
/// determinant via the Stan reference identity. Bijective.
#[derive(Debug, Clone, Copy)]
pub struct CorrCholeskyTransform;

impl CorrCholeskyTransform {
    /// Solve `D*(D-1)/2 == n` for `D` (the matrix dimension). Returns `None`
    /// if `n` is not a valid flattened-lower-triangular length.
    fn matrix_dim(n: usize) -> Option<usize> {
        // D = round((0.25 + 2N)^0.5 + 0.5); verify D*(D-1)/2 == N.
        // `transforms.py:930-933`.
        let d = (0.25 + 2.0 * n as f64).sqrt() + 0.5;
        let d = d.round() as usize;
        if d * (d - 1) / 2 == n { Some(d) } else { None }
    }
}

impl<T: Float> Transform<T> for CorrCholeskyTransform {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:885-898`:
        //   x = tanh(x); clamp to (-1+eps, 1-eps); r = vec_to_tril(x, diag=-1)
        //   z = r^2; z1m_cumprod_sqrt = (1-z).sqrt().cumprod(-1)
        //   r += eye; y = r * pad(z1m_cumprod_sqrt[...,:-1], [1,0], value=1)
        no_grad(|| {
            let shape = x.shape();
            let n = match shape.last().copied() {
                Some(n) if n >= 1 => n,
                _ => {
                    return Err(FerrotorchError::InvalidArgument {
                        message: "CorrCholeskyTransform: input must have >= 1 trailing dim".into(),
                    });
                }
            };
            let d = Self::matrix_dim(n).ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!("CorrCholeskyTransform: {n} is not a D*(D-1)/2 length"),
            })?;
            let data = x.data_vec()?;
            let rows = data.len() / n;
            let one = T::from(1.0).unwrap();
            let eps = T::from(1.19e-7).unwrap();
            let mut out = vec![T::from(0.0).unwrap(); rows * d * d];
            for b in 0..rows {
                let vec = &data[b * n..(b + 1) * n];
                // r: lower-triangular matrix (diag=-1, i.e. strictly below the
                // diagonal), row-major fill of `tanh(x)` clamped to (-1,1).
                let mut r = vec![T::from(0.0).unwrap(); d * d];
                let mut idx = 0;
                for i in 1..d {
                    for j in 0..i {
                        let t = vec[idx].tanh();
                        let t = if t < eps - one {
                            eps - one
                        } else if t > one - eps {
                            one - eps
                        } else {
                            t
                        };
                        r[i * d + j] = t;
                        idx += 1;
                    }
                }
                // For each row i: y[i][j] = r[i][j] * sqrt(prod_{l<j} (1 - r[i][l]^2)).
                // Diagonal y[i][i] = sqrt(prod_{l<i}(1 - r[i][l]^2)) (r diag = 1).
                for i in 0..d {
                    let mut cumprod = one; // prod of (1 - z) up to j-1.
                    for j in 0..=i {
                        let base = b * d * d + i * d + j;
                        if j == i {
                            // diagonal: r=1, factor = sqrt(cumprod).
                            out[base] = cumprod.sqrt();
                        } else {
                            let rij = r[i * d + j];
                            out[base] = rij * cumprod.sqrt();
                            cumprod = cumprod * (one - rij * rij);
                        }
                    }
                }
            }
            let mut out_shape = shape[..shape.len() - 1].to_vec();
            out_shape.push(d);
            out_shape.push(d);
            Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)?.to(x.device())
        })
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:900-910`:
        //   y_cumsum = 1 - cumsum(y*y, -1); shifted = pad(y_cumsum[...,:-1],[1,0],1)
        //   y_vec = tril_to_vec(y, diag=-1); y_cumsum_vec = tril_to_vec(shifted, diag=-1)
        //   t = y_vec / sqrt(y_cumsum_vec); x = (log1p(t) - log1p(-t)) / 2
        no_grad(|| {
            let shape = y.shape();
            if shape.len() < 2 {
                return Err(FerrotorchError::InvalidArgument {
                    message: "CorrCholeskyTransform inverse: input must have >= 2 dims".into(),
                });
            }
            let d = shape[shape.len() - 1];
            if shape[shape.len() - 2] != d {
                return Err(FerrotorchError::InvalidArgument {
                    message: "CorrCholeskyTransform inverse: trailing dims must be square".into(),
                });
            }
            let n = d * (d - 1) / 2;
            let data = y.data_vec()?;
            let batches = data.len() / (d * d);
            let one = T::from(1.0).unwrap();
            let half = T::from(0.5).unwrap();
            let mut out = Vec::with_capacity(batches * n);
            for b in 0..batches {
                let base = b * d * d;
                // For each row i, y_cumsum_shifted[i][j] = 1 - sum_{l<j} y[i][l]^2.
                for i in 1..d {
                    let mut cumsum = T::from(0.0).unwrap(); // sum_{l < j} y^2
                    for j in 0..i {
                        let yij = data[base + i * d + j];
                        let shifted = one - cumsum; // value at position j (before adding y[j]^2)
                        let t = yij / shifted.sqrt();
                        // atanh(t) = (log1p(t) - log1p(-t)) / 2.
                        let atanh = ((one + t).ln() - (one - t).ln()) * half;
                        out.push(atanh);
                        cumsum += yij * yij;
                    }
                }
            }
            let mut out_shape = shape[..shape.len() - 2].to_vec();
            out_shape.push(n);
            Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)?.to(y.device())
        })
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:912-924`:
        //   y1m_cumsum = 1 - (y*y).cumsum(-1)
        //   y1m_cumsum_tril = tril_to_vec(y1m_cumsum, diag=-2)
        //   stick_breaking_logdet = 0.5 * y1m_cumsum_tril.log().sum(-1)
        //   tanh_logdet = -2 * (x + softplus(-2x) - log(2)).sum(-1)
        //   return stick_breaking_logdet + tanh_logdet
        no_grad(|| {
            let yshape = y.shape();
            let d = yshape[yshape.len() - 1];
            let xshape = x.shape();
            let n = xshape[xshape.len() - 1];
            let ydata = y.data_vec()?;
            let xdata = x.data_vec()?;
            let batches = ydata.len() / (d * d);
            let one = T::from(1.0).unwrap();
            let two = T::from(2.0).unwrap();
            let half = T::from(0.5).unwrap();
            let ln2 = T::from(2.0f64.ln()).unwrap();
            let zero = T::from(0.0).unwrap();
            let mut out = Vec::with_capacity(batches);
            for b in 0..batches {
                // stick_breaking_logdet: tril of (1 - cumsum(y^2)) with diag=-2,
                // i.e. for row i (i>=2) and columns j < i-1.
                let ybase = b * d * d;
                let mut sb = zero;
                for i in 0..d {
                    let mut cumsum = zero;
                    for j in 0..d {
                        cumsum += ydata[ybase + i * d + j] * ydata[ybase + i * d + j];
                        let val = one - cumsum;
                        // diag=-2 keeps entries with j <= i - 2.
                        if j as isize <= i as isize - 2 {
                            sb += half * val.ln();
                        }
                    }
                }
                // tanh_logdet over the input vector (length n).
                let xrow = &xdata[b * n..(b + 1) * n];
                let mut th = zero;
                for &xi in xrow {
                    // softplus(-2x) = ln(1 + exp(-2x)).
                    let sp = (one + (zero - two * xi).exp()).ln();
                    th += xi + sp - ln2;
                }
                out.push(sb + (zero - two) * th);
            }
            let out_shape: Vec<usize> = xshape[..xshape.len() - 1].to_vec();
            Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)?.to(x.device())
        })
    }

    fn name(&self) -> &'static str {
        "CorrCholeskyTransform"
    }

    fn event_dim(&self) -> usize {
        // domain is real_vector (event_dim 1), codomain is corr_cholesky
        // (event_dim 2). Upstream raises on mismatched event_dim; ferrotorch
        // reports the codomain's for the metadata accessor.
        1
    }

    fn domain(&self) -> Box<dyn DistConstraint> {
        // `transforms.py:881` `domain = constraints.real_vector`.
        Box::new(crate::constraints::RealVector)
    }

    fn codomain(&self) -> Box<dyn DistConstraint> {
        // `transforms.py:882` `codomain = constraints.corr_cholesky`.
        Box::new(crate::constraints::CorrCholesky)
    }
}

// ---------------------------------------------------------------------------
// ReshapeTransform: reshape the rightmost event dims (unit Jacobian)
// ---------------------------------------------------------------------------

/// Unit-Jacobian transform that reshapes the rightmost part of a tensor.
///
/// `in_shape` and `out_shape` must have the same number of elements. Mirrors
/// `torch/distributions/transforms.py:500-573` (`class ReshapeTransform`).
#[derive(Debug, Clone)]
pub struct ReshapeTransform {
    /// The input event shape.
    pub in_shape: Vec<usize>,
    /// The output event shape.
    pub out_shape: Vec<usize>,
}

impl ReshapeTransform {
    /// Create a reshape transform. `in_shape` and `out_shape` must have equal
    /// element counts (mirrors `transforms.py:524-525`).
    pub fn new(in_shape: Vec<usize>, out_shape: Vec<usize>) -> FerrotorchResult<Self> {
        let in_numel: usize = in_shape.iter().product();
        let out_numel: usize = out_shape.iter().product();
        if in_numel != out_numel {
            return Err(FerrotorchError::InvalidArgument {
                message: "ReshapeTransform: in_shape, out_shape have different numbers of elements"
                    .into(),
            });
        }
        Ok(Self {
            in_shape,
            out_shape,
        })
    }

    /// Replace the trailing `from` dims of `t` with `to`, preserving the batch
    /// prefix. Mirrors `transforms.py:543-549` (`_call` / `_inverse`).
    fn reshape_trailing<T: Float>(
        t: &Tensor<T>,
        from: &[usize],
        to: &[usize],
    ) -> FerrotorchResult<Tensor<T>> {
        let shape = t.shape();
        if shape.len() < from.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: "ReshapeTransform: too few dimensions on input".into(),
            });
        }
        let cut = shape.len() - from.len();
        if &shape[cut..] != from {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ReshapeTransform: shape mismatch — expected trailing {:?}, got {:?}",
                    from,
                    &shape[cut..]
                ),
            });
        }
        let mut new_shape: Vec<isize> = shape[..cut].iter().map(|&d| d as isize).collect();
        new_shape.extend(to.iter().map(|&d| d as isize));
        ferrotorch_core::grad_fns::shape::reshape(t, &new_shape)
    }
}

impl<T: Float> Transform<T> for ReshapeTransform {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        no_grad(|| Self::reshape_trailing(x, &self.in_shape, &self.out_shape))
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        no_grad(|| Self::reshape_trailing(y, &self.out_shape, &self.in_shape))
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:551-553` `return x.new_zeros(batch_shape)`.
        no_grad(|| {
            let shape = x.shape();
            let cut = shape.len().saturating_sub(self.in_shape.len());
            let batch_shape: Vec<usize> = shape[..cut].to_vec();
            creation::full(&batch_shape, T::from(0.0).unwrap())?.to(x.device())
        })
    }

    fn name(&self) -> &'static str {
        "ReshapeTransform"
    }

    fn event_dim(&self) -> usize {
        self.in_shape.len()
    }

    fn domain(&self) -> Box<dyn DistConstraint> {
        // `transforms.py:530-531` `independent(real, len(in_shape))`.
        Box::new(crate::constraints::Real)
    }

    fn codomain(&self) -> Box<dyn DistConstraint> {
        // `transforms.py:535-536` `independent(real, len(out_shape))`.
        Box::new(crate::constraints::Real)
    }
}

// ---------------------------------------------------------------------------
// IndependentTransform: reinterpret rightmost batch dims as event dims
// ---------------------------------------------------------------------------

/// Wraps a base transform, treating `reinterpreted_batch_ndims`-many rightmost
/// dimensions as dependent. Forward/inverse pass straight through; only
/// `log_abs_det_jacobian` changes — it sums out those rightmost dims. Mirrors
/// `torch/distributions/transforms.py:422-497`
/// (`class IndependentTransform`).
pub struct IndependentTransform<T: Float> {
    base: Box<dyn Transform<T>>,
    reinterpreted_batch_ndims: usize,
}

impl<T: Float> IndependentTransform<T> {
    /// Wrap `base`, reinterpreting `reinterpreted_batch_ndims` rightmost dims
    /// as event dims.
    pub fn new(base: Box<dyn Transform<T>>, reinterpreted_batch_ndims: usize) -> Self {
        Self {
            base,
            reinterpreted_batch_ndims,
        }
    }
}

impl<T: Float> Transform<T> for IndependentTransform<T> {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:475-478` — no change to the forward map.
        self.base.forward(x)
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:480-483`.
        self.base.inverse(y)
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:485-488`:
        //   result = base.log_abs_det_jacobian(x, y)
        //   result = _sum_rightmost(result, reinterpreted_batch_ndims)
        no_grad(|| {
            let ldj = self.base.log_abs_det_jacobian(x, y)?;
            sum_rightmost(&ldj, self.reinterpreted_batch_ndims)
        })
    }

    fn name(&self) -> &'static str {
        "IndependentTransform"
    }

    fn bijective(&self) -> bool {
        self.base.bijective()
    }

    fn sign(&self) -> Option<i32> {
        self.base.sign()
    }

    fn event_dim(&self) -> usize {
        // `transforms.py:453-465` — base event dim + reinterpreted ndims.
        <dyn Transform<T> as TransformEventDim>::event_dim_of(self.base.as_ref())
            + self.reinterpreted_batch_ndims
    }

    fn domain(&self) -> Box<dyn DistConstraint> {
        self.base.domain()
    }

    fn codomain(&self) -> Box<dyn DistConstraint> {
        self.base.codomain()
    }
}

/// Helper to read a boxed transform's `event_dim` without re-borrowing through
/// the generic `Transform` method (which `IndependentTransform::event_dim`
/// already shadows). Object-safe shim over the trait method.
trait TransformEventDim {
    fn event_dim_of(&self) -> usize;
}

impl<T: Float> TransformEventDim for dyn Transform<T> {
    fn event_dim_of(&self) -> usize {
        Transform::event_dim(self)
    }
}

/// Sum the rightmost `ndims` dimensions of `t`, mirroring
/// `torch/distributions/utils.py:_sum_rightmost`. CPU-side reduction over the
/// flattened trailing block.
fn sum_rightmost<T: Float>(t: &Tensor<T>, ndims: usize) -> FerrotorchResult<Tensor<T>> {
    if ndims == 0 {
        return Ok(t.clone());
    }
    let shape = t.shape();
    if shape.len() < ndims {
        return Err(FerrotorchError::InvalidArgument {
            message: "sum_rightmost: too few dimensions".into(),
        });
    }
    let cut = shape.len() - ndims;
    let out_shape: Vec<usize> = shape[..cut].to_vec();
    let block: usize = shape[cut..].iter().product::<usize>().max(1);
    let data = t.data_vec()?;
    let n_blocks = data.len() / block;
    let zero = T::from(0.0).unwrap();
    let mut out = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let mut acc = zero;
        for &v in &data[b * block..(b + 1) * block] {
            acc += v;
        }
        out.push(acc);
    }
    Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)?.to(t.device())
}

// ---------------------------------------------------------------------------
// CatTransform: apply sub-transforms to slices along a dim (cat-compatible)
// ---------------------------------------------------------------------------

/// Apply a sequence of sub-transforms component-wise to contiguous slices of
/// the input along `dim`, each of the corresponding `lengths`, then
/// concatenate. Mirrors `torch/distributions/transforms.py:1081-1220`
/// (`class CatTransform`).
pub struct CatTransform<T: Float> {
    transforms: Vec<Box<dyn Transform<T>>>,
    dim: usize,
    lengths: Vec<usize>,
}

impl<T: Float> CatTransform<T> {
    /// Create a `CatTransform` over `dim` with per-transform slice `lengths`.
    /// `lengths.len()` must equal `transforms.len()` (mirrors
    /// `transforms.py:1114-1117`).
    pub fn new(
        transforms: Vec<Box<dyn Transform<T>>>,
        dim: usize,
        lengths: Vec<usize>,
    ) -> FerrotorchResult<Self> {
        if lengths.len() != transforms.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: "CatTransform: lengths must match number of transforms".into(),
            });
        }
        Ok(Self {
            transforms,
            dim,
            lengths,
        })
    }

    fn total_length(&self) -> usize {
        self.lengths.iter().sum()
    }
}

impl<T: Float> Transform<T> for CatTransform<T> {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:1133-1148`: narrow each slice, transform, then cat.
        no_grad(|| {
            if x.shape().get(self.dim).copied() != Some(self.total_length()) {
                return Err(FerrotorchError::InvalidArgument {
                    message: "CatTransform: x size along dim must equal sum(lengths)".into(),
                });
            }
            let mut slices = Vec::with_capacity(self.transforms.len());
            let mut start = 0usize;
            for (t, &len) in self.transforms.iter().zip(self.lengths.iter()) {
                let xs = x.narrow(self.dim, start, len)?;
                slices.push(t.forward(&xs)?);
                start += len;
            }
            cat_op(&slices, self.dim as isize)
        })
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:1150-1165`.
        no_grad(|| {
            if y.shape().get(self.dim).copied() != Some(self.total_length()) {
                return Err(FerrotorchError::InvalidArgument {
                    message: "CatTransform: y size along dim must equal sum(lengths)".into(),
                });
            }
            let mut slices = Vec::with_capacity(self.transforms.len());
            let mut start = 0usize;
            for (t, &len) in self.transforms.iter().zip(self.lengths.iter()) {
                let ys = y.narrow(self.dim, start, len)?;
                slices.push(t.inverse(&ys)?);
                start += len;
            }
            cat_op(&slices, self.dim as isize)
        })
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:1167-1202`: per-slice LDJ; concatenate along `dim`
        // (the event_dim==0 case our element-wise transforms exercise).
        no_grad(|| {
            let mut slices = Vec::with_capacity(self.transforms.len());
            let mut start = 0usize;
            for (t, &len) in self.transforms.iter().zip(self.lengths.iter()) {
                let xs = x.narrow(self.dim, start, len)?;
                let ys = y.narrow(self.dim, start, len)?;
                slices.push(t.log_abs_det_jacobian(&xs, &ys)?);
                start += len;
            }
            cat_op(&slices, self.dim as isize)
        })
    }

    fn name(&self) -> &'static str {
        "CatTransform"
    }

    fn bijective(&self) -> bool {
        self.transforms.iter().all(|t| t.bijective())
    }
}

// ---------------------------------------------------------------------------
// StackTransform: apply sub-transforms to slices along a dim (stack-compatible)
// ---------------------------------------------------------------------------

/// Apply a sequence of sub-transforms component-wise to each slice taken along
/// `dim`, then re-stack. The input size along `dim` must equal the number of
/// transforms. Mirrors `torch/distributions/transforms.py:1223-1321`
/// (`class StackTransform`).
pub struct StackTransform<T: Float> {
    transforms: Vec<Box<dyn Transform<T>>>,
    dim: usize,
}

impl<T: Float> StackTransform<T> {
    /// Create a `StackTransform` over `dim`.
    pub fn new(transforms: Vec<Box<dyn Transform<T>>>, dim: usize) -> Self {
        Self { transforms, dim }
    }

    fn check_size<F>(&self, t: &Tensor<F>) -> FerrotorchResult<()>
    where
        F: Float,
    {
        if t.shape().get(self.dim).copied() != Some(self.transforms.len()) {
            return Err(FerrotorchError::InvalidArgument {
                message: "StackTransform: size along dim must equal number of transforms".into(),
            });
        }
        Ok(())
    }
}

impl<T: Float> Transform<T> for StackTransform<T> {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:1257-1269`: select each slice, transform, stack.
        no_grad(|| {
            self.check_size(x)?;
            let mut slices = Vec::with_capacity(self.transforms.len());
            for (i, t) in self.transforms.iter().enumerate() {
                let xs = select_op(x, self.dim, i)?;
                slices.push(t.forward(&xs)?);
            }
            stack_op(&slices, self.dim)
        })
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:1271-1283`.
        no_grad(|| {
            self.check_size(y)?;
            let mut slices = Vec::with_capacity(self.transforms.len());
            for (i, t) in self.transforms.iter().enumerate() {
                let ys = select_op(y, self.dim, i)?;
                slices.push(t.inverse(&ys)?);
            }
            stack_op(&slices, self.dim)
        })
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:1285-1307`: per-slice LDJ, stacked along `dim`.
        no_grad(|| {
            self.check_size(x)?;
            self.check_size(y)?;
            let mut slices = Vec::with_capacity(self.transforms.len());
            for (i, t) in self.transforms.iter().enumerate() {
                let xs = select_op(x, self.dim, i)?;
                let ys = select_op(y, self.dim, i)?;
                slices.push(t.log_abs_det_jacobian(&xs, &ys)?);
            }
            stack_op(&slices, self.dim)
        })
    }

    fn name(&self) -> &'static str {
        "StackTransform"
    }

    fn bijective(&self) -> bool {
        self.transforms.iter().all(|t| t.bijective())
    }
}

// ---------------------------------------------------------------------------
// CumulativeDistributionTransform: y = distribution.cdf(x)
// ---------------------------------------------------------------------------

/// Transform via the cumulative distribution function of a base distribution.
///
/// Forward applies the CDF (`R → (0,1)`), inverse the inverse-CDF, and
/// `log_abs_det_jacobian` is the base distribution's `log_prob(x)` (since the
/// derivative of the CDF is the density). Mirrors
/// `torch/distributions/transforms.py:1324-1367`
/// (`class CumulativeDistributionTransform`).
pub struct CumulativeDistributionTransform<T: Float> {
    distribution: Box<dyn Distribution<T>>,
}

impl<T: Float> CumulativeDistributionTransform<T> {
    /// Create a CDF transform from a base `distribution`.
    pub fn new(distribution: Box<dyn Distribution<T>>) -> Self {
        Self { distribution }
    }
}

impl<T: Float> Transform<T> for CumulativeDistributionTransform<T> {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:1355-1356` `return self.distribution.cdf(x)`.
        no_grad(|| self.distribution.cdf(x))
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:1358-1359` `return self.distribution.icdf(y)`.
        no_grad(|| self.distribution.icdf(y))
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // `transforms.py:1361-1362` `return self.distribution.log_prob(x)`.
        no_grad(|| self.distribution.log_prob(x))
    }

    fn name(&self) -> &'static str {
        "CumulativeDistributionTransform"
    }

    fn sign(&self) -> Option<i32> {
        // `transforms.py:1345` `sign = +1` (CDF is monotone increasing).
        Some(1)
    }

    fn domain(&self) -> Box<dyn DistConstraint> {
        // `transforms.py:1351-1353` `return self.distribution.support`.
        self.distribution
            .support()
            .unwrap_or_else(|| Box::new(crate::constraints::Real))
    }

    fn codomain(&self) -> Box<dyn DistConstraint> {
        // `transforms.py:1344` `codomain = constraints.unit_interval`.
        Box::new(crate::constraints::UnitInterval)
    }
}

// ---------------------------------------------------------------------------
// TransformedDistribution
// ---------------------------------------------------------------------------

use crate::Distribution;

/// A distribution formed by applying a sequence of transforms to a base
/// distribution.
///
/// Given a base distribution `p(x)` and a bijective transform `f` with
/// `y = f(x)`, the density of `y` is:
///
/// ```text
/// log p(y) = log p(f^{-1}(y)) + log |det df^{-1}/dy|
///          = log p(f^{-1}(y)) - log |det df/dx|_{x = f^{-1}(y)}
/// ```
///
/// This is the change-of-variables formula.
///
/// # Examples
///
/// ```ignore
/// // LogNormal = Normal pushed through exp
/// let base = Normal::new(loc, scale)?;
/// let transforms: Vec<Box<dyn Transform<f32>>> = vec![Box::new(ExpTransform)];
/// let log_normal = TransformedDistribution::new(base, transforms);
/// ```
pub struct TransformedDistribution<T: Float> {
    base: Box<dyn Distribution<T>>,
    transforms: Vec<Box<dyn Transform<T>>>,
}

impl<T: Float> TransformedDistribution<T> {
    /// Create a transformed distribution.
    ///
    /// `base` is the base distribution. `transforms` are applied left-to-right
    /// in the forward (sampling) direction.
    pub fn new(base: Box<dyn Distribution<T>>, transforms: Vec<Box<dyn Transform<T>>>) -> Self {
        Self { base, transforms }
    }
}

impl<T: Float> Distribution<T> for TransformedDistribution<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let mut x = self.base.sample(shape)?;
        for t in &self.transforms {
            x = t.forward(&x)?;
        }
        Ok(x)
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let mut x = self.base.rsample(shape)?;
        for t in &self.transforms {
            x = t.forward(&x)?;
        }
        Ok(x)
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Walk transforms in reverse, inverting back to the base sample and
        // accumulating sum-of-log-det-jacobians device-resident. Final result
        // is base_log_prob(inverted) - sum_log_dets.
        no_grad(|| {
            let mut y = value.clone();
            // Initialize accumulator as zeros on value.device() with value's shape.
            let mut sum_ldj: Tensor<T> =
                creation::full(value.shape(), T::from(0.0).unwrap())?.to(value.device())?;

            for t in self.transforms.iter().rev() {
                let x = t.inverse(&y)?;
                let ldj = t.log_abs_det_jacobian(&x, &y)?;
                sum_ldj = add(&sum_ldj, &ldj)?;
                y = x;
            }

            let base_lp = self.base.log_prob(&y)?;
            // Result = base_lp - sum_ldj. base_lp is on value.device() if the
            // base distribution preserves device; sub will fail-fast on a
            // mismatch, which is the correct PyTorch-faithful behaviour.
            sub(&base_lp, &sum_ldj)
        })
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        // Change-of-variables identity:
        //   H(Y) = H(X) + E_X[log|det J_f(X)|]
        // We accept three closed-form dispatches:
        //
        //   1. Empty transform list                       — H(Y) = H(X).
        //   2. Every transform reports
        //      `constant_entropy_contribution()`           — contributions sum
        //      to a scalar `c` independent of X, and
        //      H(Y) = H(X) + c (broadcast onto X's entropy shape).
        //   3. A single [`ExpTransform`] with a base that
        //      implements `mean()`                         — H(Y) = H(X)
        //      + E[X] = H(X) + base.mean().
        //
        // Anything else (sigmoid, tanh, softplus, multi-Exp chains, Exp
        // composed with non-trivial transforms, etc.) is not currently
        // tractable in closed form and we surface a precise, structured
        // error naming the offending transform(s) so the caller can either
        // resort to Monte-Carlo or extend this dispatch.
        let base_entropy = self.base.entropy()?;
        if self.transforms.is_empty() {
            return Ok(base_entropy);
        }

        // Path 2 — all-constant Jacobian contributions sum to a scalar.
        let all_constant: Option<T> = {
            let mut acc = T::from(0.0).unwrap();
            let mut ok = true;
            for t in &self.transforms {
                match t.constant_entropy_contribution() {
                    Some(c) => acc += c,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok { Some(acc) } else { None }
        };
        if let Some(c) = all_constant {
            // Broadcast the scalar onto base_entropy's shape and add.
            let device = base_entropy.device();
            let c_tensor = creation::full(base_entropy.shape(), c)?.to(device)?;
            return add(&base_entropy, &c_tensor);
        }

        // Path 3 — a single Exp transform: contribution is E[X] = base.mean().
        if self.transforms.len() == 1 && self.transforms[0].is_exp_transform() {
            let mean = self.base.mean()?;
            // base.mean() is shape-compatible with base.entropy() for the
            // distributions we support (Normal et al. parameterised by
            // loc/scale). If the base distribution does not implement mean
            // it surfaces its own InvalidArgument here, which we propagate.
            //
            // mean may live on a different device than entropy (e.g. Normal
            // returns `self.loc.clone()` while entropy materialises on
            // scale.device()). Move mean onto entropy's device before adding.
            let device = base_entropy.device();
            let mean_on_device = if mean.device() == device {
                mean
            } else {
                mean.to(device)?
            };
            return add(&base_entropy, &mean_on_device);
        }

        // Path 4 (REQ-9, #1378) — Monte-Carlo fallback for chains whose
        // log|det J| depends on X (SigmoidTransform, TanhTransform,
        // SoftplusTransform, multi-Exp, Exp-then-Affine, ...). The
        // change-of-variables entropy identity
        //   H(Y) = H(X) + E_X[ log|det J_f(X)| ]
        // holds for ANY bijection; when the contribution is not closed-form we
        // estimate the expectation with `MC_ENTROPY_SAMPLES` base draws pushed
        // through the chain's `log_abs_det_jacobian`. PyTorch's upstream
        // `TransformedDistribution` raises `NotImplementedError` here; the MC
        // estimator is the documented faithful fallback (`transforms.md`
        // REQ-9). The estimate is averaged per batch element so it broadcasts
        // onto `base_entropy`'s shape.
        self.entropy_monte_carlo(&base_entropy)
    }

    fn support(&self) -> Option<Box<dyn DistConstraint>> {
        // The support of a transformed distribution is the codomain of the
        // LAST transform in the chain (the final output space). For an empty
        // chain the support is the base distribution's support. This is the
        // production consumer of `Transform::codomain()` (#1373) and mirrors
        // `torch/distributions/transformed_distribution.py:129-137`.
        match self.transforms.last() {
            Some(last) => Some(last.codomain()),
            None => self.base.support(),
        }
    }

    fn kl_recurse(&self) -> Option<crate::KlRecurseInfo<'_, T>> {
        // `torch/distributions/kl.py:496-502` `_kl_transformed_transformed`:
        //   if p.transforms != q.transforms: raise NotImplementedError
        //   if p.event_shape != q.event_shape: raise NotImplementedError
        //   return kl_divergence(p.base_dist, q.base_dist)
        // We expose the type-erased base, the per-transform structural
        // fingerprint (mirroring each `Transform.__eq__`), and the
        // `event_shape` so `kl::kl_divergence_dyn` can apply the two guards
        // and recurse on the base. Returns the base KL unchanged (no
        // sum-rightmost, unlike `Independent`).
        let transform_fingerprint: Vec<String> = self
            .transforms
            .iter()
            .map(|t| t.transform_eq_key())
            .collect();
        Some(crate::KlRecurseInfo {
            base: self.base.as_ref(),
            kind: crate::KlRecurseKind::Transformed {
                transform_fingerprint,
                event_shape: self.event_shape(),
            },
        })
    }
}

/// Number of base samples drawn to Monte-Carlo estimate
/// `E_X[log|det J_f(X)|]` in the entropy fallback. Chosen for a stable
/// estimate (relative std-error ~ 1/sqrt(N)) without excessive cost; the
/// fallback is only hit for the X-dependent-Jacobian chains.
const MC_ENTROPY_SAMPLES: usize = 20_000;

impl<T: Float> TransformedDistribution<T> {
    /// Monte-Carlo estimate of `H(Y) = H(X) + E_X[log|det J_f(X)|]` for chains
    /// whose log-det-Jacobian contribution depends on `X` and therefore has no
    /// closed form (Sigmoid/Tanh/Softplus/multi-Exp/Exp-then-Affine).
    ///
    /// Draws `MC_ENTROPY_SAMPLES` base samples per batch element, pushes them
    /// through the transform chain accumulating each link's
    /// `log_abs_det_jacobian`, and averages over the sample axis. The result is
    /// `base_entropy + mean_sample(sum_links log|det J|)`, element-wise on the
    /// batch shape. CPU-resident (the estimator reads `data_vec()`).
    fn entropy_monte_carlo(&self, base_entropy: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        no_grad(|| {
            let batch_shape = self.base.batch_shape();
            let batch_len: usize = batch_shape.iter().product::<usize>().max(1);
            let n = MC_ENTROPY_SAMPLES;

            // Draw an [n, *batch] block of base samples and push the whole block
            // through the chain at once so each per-link LDJ is evaluated on the
            // correct intermediate values.
            let sample_shape: Vec<usize> = if batch_shape.is_empty() {
                vec![n]
            } else {
                let mut s = vec![n];
                s.extend_from_slice(&batch_shape);
                s
            };
            let x0 = self.base.sample(&sample_shape)?;

            // Accumulate sum of per-link log|det J| evaluated along the chain.
            let mut xs = x0;
            let mut sum_ldj: Option<Tensor<T>> = None;
            for t in &self.transforms {
                let next = t.forward(&xs)?;
                let ldj = t.log_abs_det_jacobian(&xs, &next)?;
                sum_ldj = Some(match sum_ldj {
                    Some(acc) => add(&acc, &ldj)?,
                    None => ldj,
                });
                xs = next;
            }
            let sum_ldj = sum_ldj.expect("non-empty chain reaches the MC fallback");

            // Average over the sample axis (the leading `n`): the flat layout is
            // row-major [n, *batch], so element i maps to batch slot i % batch_len.
            let ldj_data = sum_ldj.data_vec()?;
            let zero = T::from(0.0).unwrap();
            let n_t = T::from(n as f64).unwrap();
            let mut contrib = vec![zero; batch_len];
            for (i, &v) in ldj_data.iter().enumerate() {
                let slot = i % batch_len;
                contrib[slot] += v;
            }
            let contrib: Vec<T> = contrib.into_iter().map(|c| c / n_t).collect();

            let device = base_entropy.device();
            let contrib_t =
                Tensor::from_storage(TensorStorage::cpu(contrib), batch_shape.clone(), false)?
                    .to(device)?;
            add(base_entropy, &contrib_t)
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, scalar};

    // -- domain / codomain Constraint linkage (#1373) ------------------------

    #[test]
    fn test_transform_domain_codomain_names() {
        // Each transform advertises its input/output Constraint, mirroring the
        // `domain`/`codomain` class attributes in
        // `torch/distributions/transforms.py`.
        let exp: &dyn Transform<f64> = &ExpTransform;
        assert_eq!(exp.domain().name(), "Real");
        assert_eq!(exp.codomain().name(), "Positive"); // transforms.py:581-582

        let sig: &dyn Transform<f64> = &SigmoidTransform;
        assert_eq!(sig.domain().name(), "Real");
        assert_eq!(sig.codomain().name(), "UnitInterval"); // transforms.py:652-653

        let sp: &dyn Transform<f64> = &SoftplusTransform;
        assert_eq!(sp.domain().name(), "Real");
        assert_eq!(sp.codomain().name(), "Positive"); // transforms.py:678-679

        let tanh: &dyn Transform<f64> = &TanhTransform;
        assert_eq!(tanh.domain().name(), "Real");
        assert_eq!(tanh.codomain().name(), "ClosedInterval"); // transforms.py:719-720

        let affine: AffineTransform<f64> = AffineTransform::new(1.0, 2.0);
        assert_eq!(affine.domain().name(), "Real"); // transforms.py:789-799
        assert_eq!(affine.codomain().name(), "Real");
    }

    #[test]
    fn test_compose_domain_codomain_endpoints() {
        // Compose [Affine, Exp]: domain = first.domain = Real,
        // codomain = last.codomain = Positive. Mirrors transforms.py:313-347.
        let chain: ComposeTransform<f64> = ComposeTransform::new(vec![
            Box::new(AffineTransform::new(0.0, 2.0)),
            Box::new(ExpTransform),
        ]);
        assert_eq!(chain.domain().name(), "Real");
        assert_eq!(chain.codomain().name(), "Positive");

        // Compose [Exp, Sigmoid]: domain = Real (Exp), codomain = UnitInterval.
        let chain2: ComposeTransform<f64> =
            ComposeTransform::new(vec![Box::new(ExpTransform), Box::new(SigmoidTransform)]);
        assert_eq!(chain2.domain().name(), "Real");
        assert_eq!(chain2.codomain().name(), "UnitInterval");

        // Empty chain → identity → Real/Real.
        let empty: ComposeTransform<f64> = ComposeTransform::new(vec![]);
        assert_eq!(empty.domain().name(), "Real");
        assert_eq!(empty.codomain().name(), "Real");
    }

    #[test]
    fn test_transformed_distribution_support_is_last_codomain() {
        // Production consumer of Transform::codomain (#1373):
        // TransformedDistribution::support returns the chain's final codomain.
        use crate::Normal;
        let base = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        // Normal → exp = LogNormal, support is the positive reals.
        let td = TransformedDistribution::new(Box::new(base), vec![Box::new(ExpTransform)]);
        let support = td.support().expect("transformed support must be Some");
        assert_eq!(support.name(), "Positive");

        // Normal → sigmoid: support is the unit interval.
        let base2 = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let td2 = TransformedDistribution::new(Box::new(base2), vec![Box::new(SigmoidTransform)]);
        assert_eq!(td2.support().unwrap().name(), "UnitInterval");
    }

    // -- ExpTransform --------------------------------------------------------

    #[test]
    fn test_exp_forward() {
        let x = from_slice(&[0.0f32, 1.0, -1.0], &[3]).unwrap();
        let t = ExpTransform;
        let y = t.forward(&x).unwrap();
        let data = y.data().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 1.0f32.exp()).abs() < 1e-5);
        assert!((data[2] - (-1.0f32).exp()).abs() < 1e-5);
    }

    #[test]
    fn test_exp_inverse() {
        let y = from_slice(
            &[1.0f32, std::f32::consts::E, 1.0 / std::f32::consts::E],
            &[3],
        )
        .unwrap();
        let t = ExpTransform;
        let x = t.inverse(&y).unwrap();
        let data = x.data().unwrap();
        assert!(data[0].abs() < 1e-5); // ln(1) = 0
        assert!((data[1] - 1.0).abs() < 1e-5); // ln(e) = 1
    }

    #[test]
    fn test_exp_roundtrip() {
        let x = from_slice(&[-2.0f32, 0.0, 3.0], &[3]).unwrap();
        let t = ExpTransform;
        let y = t.forward(&x).unwrap();
        let x2 = t.inverse(&y).unwrap();
        let orig = x.data().unwrap();
        let recov = x2.data().unwrap();
        for (a, b) in orig.iter().zip(recov.iter()) {
            assert!((a - b).abs() < 1e-5, "roundtrip failed: {a} vs {b}");
        }
    }

    #[test]
    fn test_exp_log_det_jacobian() {
        // For exp transform, log|det J| = x
        let x = from_slice(&[-1.0f32, 0.0, 2.0], &[3]).unwrap();
        let y = ExpTransform.forward(&x).unwrap();
        let ldj = ExpTransform.log_abs_det_jacobian(&x, &y).unwrap();
        let ldj_data = ldj.data().unwrap();
        let x_data = x.data().unwrap();
        for (ld, xv) in ldj_data.iter().zip(x_data.iter()) {
            assert!((ld - xv).abs() < 1e-6);
        }
    }

    // -- AffineTransform -----------------------------------------------------

    #[test]
    fn test_affine_forward() {
        let x = from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let t = AffineTransform::new(10.0f32, 2.0);
        let y = t.forward(&x).unwrap();
        let data = y.data().unwrap();
        assert!((data[0] - 12.0).abs() < 1e-6);
        assert!((data[1] - 14.0).abs() < 1e-6);
        assert!((data[2] - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_affine_inverse() {
        let y = from_slice(&[12.0f32, 14.0, 16.0], &[3]).unwrap();
        let t = AffineTransform::new(10.0f32, 2.0);
        let x = t.inverse(&y).unwrap();
        let data = x.data().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_affine_roundtrip() {
        let x = from_slice(&[-5.0f32, 0.0, 7.0], &[3]).unwrap();
        let t = AffineTransform::new(3.0f32, -0.5);
        let y = t.forward(&x).unwrap();
        let x2 = t.inverse(&y).unwrap();
        let orig = x.data().unwrap();
        let recov = x2.data().unwrap();
        for (a, b) in orig.iter().zip(recov.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_affine_log_det_jacobian() {
        let x = from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let t = AffineTransform::new(0.0f32, 3.0);
        let y = t.forward(&x).unwrap();
        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        let data = ldj.data().unwrap();
        let expected = 3.0f32.ln();
        for &v in data {
            assert!((v - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_affine_negative_scale_log_det() {
        let x = from_slice(&[1.0f32], &[1]).unwrap();
        let t = AffineTransform::new(0.0f32, -2.0);
        let y = t.forward(&x).unwrap();
        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        let expected = 2.0f32.ln(); // log|scale|
        assert!((ldj.item().unwrap() - expected).abs() < 1e-6);
    }

    // -- SigmoidTransform ----------------------------------------------------

    #[test]
    fn test_sigmoid_forward() {
        let x = from_slice(&[0.0f32], &[1]).unwrap();
        let y = SigmoidTransform.forward(&x).unwrap();
        assert!((y.item().unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_roundtrip() {
        let x = from_slice(&[-3.0f32, 0.0, 3.0], &[3]).unwrap();
        let t = SigmoidTransform;
        let y = t.forward(&x).unwrap();
        let x2 = t.inverse(&y).unwrap();
        let orig = x.data().unwrap();
        let recov = x2.data().unwrap();
        for (a, b) in orig.iter().zip(recov.iter()) {
            assert!((a - b).abs() < 1e-4, "sigmoid roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn test_sigmoid_log_det_jacobian() {
        // At x=0: sigmoid(0) = 0.5, sigmoid'(0) = 0.25
        // log|0.25| = log(0.25) ~ -1.3863
        let x = from_slice(&[0.0f32], &[1]).unwrap();
        let y = SigmoidTransform.forward(&x).unwrap();
        let ldj = SigmoidTransform.log_abs_det_jacobian(&x, &y).unwrap();
        let expected = 0.25f32.ln();
        assert!(
            (ldj.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            ldj.item().unwrap()
        );
    }

    // -- TanhTransform -------------------------------------------------------

    #[test]
    fn test_tanh_forward() {
        let x = from_slice(&[0.0f32], &[1]).unwrap();
        let y = TanhTransform.forward(&x).unwrap();
        assert!(y.item().unwrap().abs() < 1e-6);
    }

    #[test]
    fn test_tanh_roundtrip() {
        let x = from_slice(&[-2.0f32, 0.0, 2.0], &[3]).unwrap();
        let t = TanhTransform;
        let y = t.forward(&x).unwrap();
        let x2 = t.inverse(&y).unwrap();
        let orig = x.data().unwrap();
        let recov = x2.data().unwrap();
        for (a, b) in orig.iter().zip(recov.iter()) {
            assert!((a - b).abs() < 1e-4, "tanh roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn test_tanh_log_det_jacobian() {
        // At x=0: tanh(0) = 0, tanh'(0) = 1 - 0^2 = 1
        // log|1| = 0
        let x = from_slice(&[0.0f32], &[1]).unwrap();
        let y = TanhTransform.forward(&x).unwrap();
        let ldj = TanhTransform.log_abs_det_jacobian(&x, &y).unwrap();
        assert!(
            ldj.item().unwrap().abs() < 1e-5,
            "expected ~0, got {}",
            ldj.item().unwrap()
        );
    }

    // -- SoftplusTransform ---------------------------------------------------

    #[test]
    fn test_softplus_forward() {
        let x = from_slice(&[0.0f32], &[1]).unwrap();
        let y = SoftplusTransform.forward(&x).unwrap();
        // softplus(0) = ln(2)
        assert!(
            (y.item().unwrap() - 2.0f32.ln()).abs() < 1e-6,
            "expected ln(2), got {}",
            y.item().unwrap()
        );
    }

    #[test]
    fn test_softplus_roundtrip() {
        let x = from_slice(&[-2.0f32, 0.0, 5.0], &[3]).unwrap();
        let t = SoftplusTransform;
        let y = t.forward(&x).unwrap();
        let x2 = t.inverse(&y).unwrap();
        let orig = x.data().unwrap();
        let recov = x2.data().unwrap();
        for (a, b) in orig.iter().zip(recov.iter()) {
            assert!((a - b).abs() < 1e-4, "softplus roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn test_softplus_log_det_jacobian() {
        // softplus'(x) = sigmoid(x), at x=0: sigmoid(0)=0.5
        // log|0.5| = -ln(2)
        let x = from_slice(&[0.0f32], &[1]).unwrap();
        let y = SoftplusTransform.forward(&x).unwrap();
        let ldj = SoftplusTransform.log_abs_det_jacobian(&x, &y).unwrap();
        let expected = -(2.0f32.ln());
        assert!(
            (ldj.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            ldj.item().unwrap()
        );
    }

    // -- ComposeTransform ----------------------------------------------------

    #[test]
    fn test_compose_empty_is_identity() {
        let x = from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let t: ComposeTransform<f32> = ComposeTransform::new(vec![]);
        let y = t.forward(&x).unwrap();
        let orig = x.data().unwrap();
        let fwd = y.data().unwrap();
        for (a, b) in orig.iter().zip(fwd.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compose_exp_then_affine() {
        // y = 2 * exp(x) + 1
        let x = from_slice(&[0.0f32, 1.0], &[2]).unwrap();
        let t: ComposeTransform<f32> = ComposeTransform::new(vec![
            Box::new(ExpTransform),
            Box::new(AffineTransform::new(1.0, 2.0)),
        ]);
        let y = t.forward(&x).unwrap();
        let data = y.data().unwrap();
        // exp(0)=1, 2*1+1=3
        assert!((data[0] - 3.0).abs() < 1e-5);
        // exp(1)~2.718, 2*2.718+1~6.436
        assert!((data[1] - (2.0 * 1.0f32.exp() + 1.0)).abs() < 1e-4);
    }

    #[test]
    fn test_compose_roundtrip() {
        let x = from_slice(&[0.5f32, 1.5], &[2]).unwrap();
        let t: ComposeTransform<f32> = ComposeTransform::new(vec![
            Box::new(AffineTransform::new(0.0, 2.0)),
            Box::new(ExpTransform),
        ]);
        let y = t.forward(&x).unwrap();
        let x2 = t.inverse(&y).unwrap();
        let orig = x.data().unwrap();
        let recov = x2.data().unwrap();
        for (a, b) in orig.iter().zip(recov.iter()) {
            assert!((a - b).abs() < 1e-4, "compose roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn test_compose_log_det_jacobian() {
        // Compose: affine(0, 2) then exp => y = exp(2x)
        // dy/dx = 2 * exp(2x), log|dy/dx| = ln(2) + 2x
        let x = from_slice(&[0.0f32, 1.0], &[2]).unwrap();
        let t: ComposeTransform<f32> = ComposeTransform::new(vec![
            Box::new(AffineTransform::new(0.0, 2.0)),
            Box::new(ExpTransform),
        ]);
        let y = t.forward(&x).unwrap();
        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        let data = ldj.data().unwrap();
        // At x=0: ln(2) + 0 = ln(2)
        assert!(
            (data[0] - 2.0f32.ln()).abs() < 1e-5,
            "expected ln(2), got {}",
            data[0]
        );
        // At x=1: ln(2) + 2
        assert!(
            (data[1] - (2.0f32.ln() + 2.0)).abs() < 1e-5,
            "expected {}, got {}",
            2.0f32.ln() + 2.0,
            data[1]
        );
    }

    // -- TransformedDistribution ---------------------------------------------

    #[test]
    fn test_transformed_distribution_sample_shape() {
        use crate::Normal;
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let base = Normal::new(loc, scale).unwrap();
        let td = TransformedDistribution::new(Box::new(base), vec![Box::new(ExpTransform)]);
        let samples = td.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        // All samples should be positive (exp maps R -> R+)
        let data = samples.data().unwrap();
        for &v in data {
            assert!(v > 0.0, "expected positive, got {v}");
        }
    }

    #[test]
    fn test_transformed_distribution_log_prob() {
        // LogNormal: base = Normal(0, 1), transform = exp
        // log_prob(y) = log_prob_normal(ln(y)) - ln(y)
        use crate::Normal;
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let base = Normal::new(loc, scale).unwrap();
        let td = TransformedDistribution::new(Box::new(base), vec![Box::new(ExpTransform)]);

        let y = scalar(1.0f32).unwrap(); // ln(1) = 0
        let lp = td.log_prob(&y).unwrap();
        // At y=1: log_prob_normal(0) - 0 = -0.5*ln(2*pi)
        let expected = -0.5 * (2.0f32 * std::f32::consts::PI).ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_transformed_distribution_log_prob_general() {
        // LogNormal(0,1) at y=e: log_prob_normal(1) - 1
        use crate::Normal;
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let base = Normal::new(loc, scale).unwrap();
        let td = TransformedDistribution::new(Box::new(base), vec![Box::new(ExpTransform)]);

        let e = std::f32::consts::E;
        let y = scalar(e).unwrap();
        let lp = td.log_prob(&y).unwrap();
        // log_prob_normal(1) = -0.5*(1)^2 - 0.5*ln(2*pi)
        // log_prob_lognormal(e) = log_prob_normal(1) - ln(e) = log_prob_normal(1) - 1
        let log_prob_normal_1 = -0.5 - 0.5 * (2.0f32 * std::f32::consts::PI).ln();
        let expected = log_prob_normal_1 - 1.0;
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_transformed_distribution_entropy_empty_chain_matches_base() {
        // Empty chain → identity; entropy must equal the base distribution's
        // entropy exactly.
        use crate::Normal;
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let base = Normal::new(loc.clone(), scale.clone()).unwrap();
        let td: TransformedDistribution<f32> = TransformedDistribution::new(Box::new(base), vec![]);
        let base2 = Normal::new(loc, scale).unwrap();
        let ent = td.entropy().unwrap().item().unwrap();
        let base_ent = base2.entropy().unwrap().item().unwrap();
        assert!(
            (ent - base_ent).abs() < 1e-6,
            "empty-chain entropy: td={ent} base={base_ent}",
        );
    }

    #[test]
    fn test_transformed_distribution_entropy_affine() {
        // entropy(Normal(0,1) → affine(loc=2, scale=3)) =
        //   entropy(Normal(0,1)) + log|3|
        //   = entropy(Normal(2, 3))
        use crate::Normal;
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let base = Normal::new(loc, scale).unwrap();
        let affine = AffineTransform::new(2.0f32, 3.0f32);
        let td = TransformedDistribution::new(Box::new(base), vec![Box::new(affine)]);
        let ent_td = td.entropy().unwrap().item().unwrap();

        let loc2 = scalar(2.0f32).unwrap();
        let scale2 = scalar(3.0f32).unwrap();
        let direct = Normal::new(loc2, scale2).unwrap();
        let ent_direct = direct.entropy().unwrap().item().unwrap();
        assert!(
            (ent_td - ent_direct).abs() < 1e-5,
            "affine entropy: td={ent_td} direct={ent_direct}",
        );
    }

    #[test]
    fn test_transformed_distribution_entropy_affine_negative_scale() {
        // Negative scale: the contribution is log|scale|, which equals
        // log|scale| of the absolute value.
        use crate::Normal;
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let base = Normal::new(loc, scale).unwrap();
        let affine = AffineTransform::new(0.0f32, -2.5f32);
        let td = TransformedDistribution::new(Box::new(base), vec![Box::new(affine)]);
        let ent_td = td.entropy().unwrap().item().unwrap();

        // entropy(Normal(0,1)) = 0.5 + 0.5*ln(2*pi); + ln(2.5).
        let half = 0.5f32;
        let expected = half + half * (2.0f32 * std::f32::consts::PI).ln() + 2.5f32.ln();
        assert!(
            (ent_td - expected).abs() < 1e-5,
            "affine-neg entropy: td={ent_td} expected={expected}",
        );
    }

    #[test]
    fn test_transformed_distribution_entropy_exp_matches_lognormal() {
        // entropy(Normal(loc, scale) → exp) = entropy(LogNormal(loc, scale)) =
        //   loc + 0.5 + ln(scale) + 0.5*ln(2*pi)
        // (identical to the Normal entropy + base.mean() = entropy + loc).
        use crate::{LogNormal, Normal};
        let loc_v = 1.3f32;
        let scale_v = 0.7f32;
        let loc = scalar(loc_v).unwrap();
        let scale = scalar(scale_v).unwrap();
        let base = Normal::new(loc, scale).unwrap();
        let td = TransformedDistribution::new(Box::new(base), vec![Box::new(ExpTransform)]);
        let ent_td = td.entropy().unwrap().item().unwrap();

        let loc2 = scalar(loc_v).unwrap();
        let scale2 = scalar(scale_v).unwrap();
        let direct = LogNormal::new(loc2, scale2).unwrap();
        let ent_direct = direct.entropy().unwrap().item().unwrap();
        assert!(
            (ent_td - ent_direct).abs() < 1e-5,
            "exp entropy: td={ent_td} lognormal={ent_direct}",
        );
    }

    #[test]
    fn test_transformed_distribution_entropy_compose_affine_chain() {
        // Affine(0, 2) ∘ Affine(1, 3) is constant-Jacobian: log(2)+log(3).
        use crate::Normal;
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let base = Normal::new(loc, scale).unwrap();
        let td = TransformedDistribution::new(
            Box::new(base),
            vec![
                Box::new(AffineTransform::new(0.0f32, 2.0)),
                Box::new(AffineTransform::new(1.0f32, 3.0)),
            ],
        );
        let ent_td = td.entropy().unwrap().item().unwrap();
        let half = 0.5f32;
        let expected =
            half + half * (2.0f32 * std::f32::consts::PI).ln() + 2.0f32.ln() + 3.0f32.ln();
        assert!(
            (ent_td - expected).abs() < 1e-5,
            "affine-chain entropy: td={ent_td} expected={expected}",
        );
    }

    /// Numerical reference for `E_{X~Normal(0,1)}[ g(X) ]` via dense
    /// trapezoidal quadrature over `[-12, 12]` against the standard-normal
    /// density. Independent of the production Monte-Carlo sampler, so it is a
    /// genuine oracle for the MC entropy fallback (#1378).
    fn normal_expectation_quadrature(g: impl Fn(f64) -> f64) -> f64 {
        let lo = -12.0_f64;
        let hi = 12.0_f64;
        let steps = 240_000;
        let dx = (hi - lo) / steps as f64;
        let norm = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        let mut acc = 0.0;
        for i in 0..=steps {
            let x = lo + i as f64 * dx;
            let w = if i == 0 || i == steps { 0.5 } else { 1.0 };
            let pdf = norm * (-0.5 * x * x).exp();
            acc += w * pdf * g(x) * dx;
        }
        acc
    }

    #[test]
    fn test_transformed_distribution_entropy_sigmoid_monte_carlo() {
        // SigmoidTransform's log|det J| = log(sigmoid(x)) + log(1-sigmoid(x))
        // = -softplus(-x) - softplus(x) is X-dependent → MC fallback (#1378).
        // Verify H(Y) = H(Normal(0,1)) + E_X[log|det J|] against an INDEPENDENT
        // trapezoidal quadrature oracle for the expectation.
        use crate::Normal;
        let base = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let td = TransformedDistribution::new(Box::new(base), vec![Box::new(SigmoidTransform)]);
        let got = td.entropy().unwrap().item().unwrap();

        let base_ent = {
            let b = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
            b.entropy().unwrap().item().unwrap()
        };
        // log|sigma'(x)| = -softplus(-x) - softplus(x).
        let softplus = |z: f64| (1.0 + z.exp()).ln();
        let contrib = normal_expectation_quadrature(|x| -softplus(-x) - softplus(x));
        let expected = base_ent + contrib;
        // MC standard error ~ |ldj_std| / sqrt(N); N=20000 → tol ~ 2e-2.
        assert!(
            (got - expected).abs() < 3e-2,
            "sigmoid MC entropy: got {got}, quadrature reference {expected}, |err|={}",
            (got - expected).abs()
        );
        assert!(got.is_finite());
    }

    #[test]
    fn test_transformed_distribution_entropy_exp_then_affine_monte_carlo() {
        // [Exp, Affine(0, 2)] over Normal(0,1): y = 2*exp(x). log|det J| for the
        // chain is x (Exp) + log|2| (Affine); the chain is no longer a single
        // Exp, so it routes through the MC fallback. The contribution is
        // E_X[X] + log 2 = loc + log 2 = 0 + log 2, which we cross-check against
        // the closed-form LogNormal-scaled identity.
        use crate::Normal;
        let base = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let td = TransformedDistribution::new(
            Box::new(base),
            vec![
                Box::new(ExpTransform),
                Box::new(AffineTransform::new(0.0f64, 2.0)),
            ],
        );
        let got = td.entropy().unwrap().item().unwrap();

        let base_ent = {
            let b = Normal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
            b.entropy().unwrap().item().unwrap()
        };
        // E_X[x + log 2] = E[x] + log 2 = 0 + ln 2.
        let expected = base_ent + 0.0 + 2.0_f64.ln();
        assert!(
            (got - expected).abs() < 3e-2,
            "exp-then-affine MC entropy: got {got}, reference {expected}, |err|={}",
            (got - expected).abs()
        );
    }

    // -- f64 tests -----------------------------------------------------------

    #[test]
    fn test_transforms_f64() {
        let x = from_slice(&[0.0f64, 1.0, -1.0], &[3]).unwrap();

        // Exp
        let y = ExpTransform.forward(&x).unwrap();
        let x2 = ExpTransform.inverse(&y).unwrap();
        let orig = x.data().unwrap();
        let recov = x2.data().unwrap();
        for (a, b) in orig.iter().zip(recov.iter()) {
            assert!((a - b).abs() < 1e-12);
        }

        // Affine
        let t = AffineTransform::new(1.0f64, 3.0);
        let y = t.forward(&x).unwrap();
        let x2 = t.inverse(&y).unwrap();
        let recov = x2.data().unwrap();
        for (a, b) in orig.iter().zip(recov.iter()) {
            assert!((a - b).abs() < 1e-12);
        }

        // Sigmoid
        let y = SigmoidTransform.forward(&x).unwrap();
        let x2 = SigmoidTransform.inverse(&y).unwrap();
        let recov = x2.data().unwrap();
        for (a, b) in orig.iter().zip(recov.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // -- Pass 5.B.1 discriminating tests (#1103) -----------------------------
    //
    // These tests exercise non-degenerate, multi-element shapes that go
    // through the device-resident migration. On Linux/CPU, `result.device()
    // == input.device()` is a tautology — the discriminating signal is
    // *numerical correctness*: each transform must compute exactly the same
    // values it did before the migration. A regression in any of the
    // device-resident op chains (drop a softplus, swap an add for a sub, etc.)
    // surfaces here as a concrete numerical drift, which is what the
    // sabotage probe in the report exercises.

    #[test]
    fn exp_transform_preserves_device_and_value() {
        // Shape [2, 3]; check exp/inverse-log/log_det numerically and verify
        // device preservation through each leg.
        let x = from_slice(&[-1.0f32, 0.0, 1.0, 2.0, -2.0, 0.5], &[2, 3]).unwrap();
        let t = ExpTransform;
        let device = x.device();

        let y = t.forward(&x).unwrap();
        assert_eq!(y.device(), device, "ExpTransform::forward changed device");
        assert_eq!(y.shape(), &[2, 3]);
        let y_data = y.data().unwrap();
        let x_data = x.data().unwrap();
        for (yv, xv) in y_data.iter().zip(x_data.iter()) {
            assert!(
                (yv - xv.exp()).abs() < 1e-5,
                "exp forward: got {yv} expected {} for x={xv}",
                xv.exp()
            );
        }

        let xr = t.inverse(&y).unwrap();
        assert_eq!(xr.device(), device, "ExpTransform::inverse changed device");
        let xr_data = xr.data().unwrap();
        for (xv0, xrv) in x_data.iter().zip(xr_data.iter()) {
            assert!(
                (xv0 - xrv).abs() < 1e-5,
                "exp roundtrip: got {xrv} expected {xv0}",
            );
        }

        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        assert_eq!(ldj.device(), device, "ExpTransform::ldj changed device");
        let ldj_data = ldj.data().unwrap();
        for (ld, xv) in ldj_data.iter().zip(x_data.iter()) {
            assert!((ld - xv).abs() < 1e-6, "exp ldj: got {ld} expected {xv}");
        }
    }

    #[test]
    fn affine_transform_preserves_device_and_value() {
        // y = 2.5 + (-1.5) * x; non-trivial, negative-scale path exercises
        // the abs-value branch in log_abs_det_jacobian.
        let x = from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let t = AffineTransform::new(2.5f32, -1.5f32);
        let device = x.device();

        let y = t.forward(&x).unwrap();
        assert_eq!(y.device(), device);
        let y_data = y.data().unwrap();
        let x_data = x.data().unwrap();
        for (yv, xv) in y_data.iter().zip(x_data.iter()) {
            let expected = 2.5f32 + (-1.5f32) * xv;
            assert!(
                (yv - expected).abs() < 1e-5,
                "affine forward: got {yv} expected {expected}",
            );
        }

        let xr = t.inverse(&y).unwrap();
        assert_eq!(xr.device(), device);
        for (xv0, xrv) in x_data.iter().zip(xr.data().unwrap().iter()) {
            assert!((xv0 - xrv).abs() < 1e-5, "affine roundtrip: {xrv} vs {xv0}");
        }

        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        assert_eq!(ldj.device(), device);
        let expected_ldj = 1.5f32.ln(); // log|-1.5| = ln(1.5)
        for v in ldj.data().unwrap().iter() {
            assert!(
                (v - expected_ldj).abs() < 1e-5,
                "affine ldj: got {v} expected {expected_ldj}",
            );
        }
    }

    #[test]
    fn sigmoid_transform_preserves_device_and_value() {
        let x = from_slice(&[-2.0f32, -0.5, 0.0, 0.5, 2.0, 3.0], &[2, 3]).unwrap();
        let t = SigmoidTransform;
        let device = x.device();

        let y = t.forward(&x).unwrap();
        assert_eq!(y.device(), device);
        // sigmoid(0) = 0.5
        let y_data = y.data().unwrap();
        let x_data = x.data().unwrap();
        for (yv, xv) in y_data.iter().zip(x_data.iter()) {
            // Reference: 1 / (1 + exp(-x))
            let expected = 1.0f32 / (1.0 + (-xv).exp());
            assert!(
                (yv - expected).abs() < 1e-5,
                "sigmoid forward: got {yv} expected {expected} for x={xv}",
            );
        }

        let xr = t.inverse(&y).unwrap();
        assert_eq!(xr.device(), device);
        for (xv0, xrv) in x_data.iter().zip(xr.data().unwrap().iter()) {
            assert!(
                (xv0 - xrv).abs() < 1e-4,
                "sigmoid roundtrip: {xrv} vs {xv0}",
            );
        }

        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        assert_eq!(ldj.device(), device);
        // log|sigma'(x)| = -softplus(-x) - softplus(x). At x=0 → -2*ln(2) + ln(2) ... actually
        //   sigma'(0) = 0.25, log(0.25) = -2*ln(2) = -ln(4)
        let ldj_data = ldj.data().unwrap();
        let expected_at_zero = 0.25f32.ln();
        // index of x=0 in flat shape [2,3] -> position 2
        assert!(
            (ldj_data[2] - expected_at_zero).abs() < 1e-5,
            "sigmoid ldj at x=0: got {} expected {expected_at_zero}",
            ldj_data[2],
        );
    }

    #[test]
    fn tanh_transform_preserves_device_and_value() {
        let x = from_slice(&[-1.5f32, -0.5, 0.0, 0.5, 1.5, 2.0], &[2, 3]).unwrap();
        let t = TanhTransform;
        let device = x.device();

        let y = t.forward(&x).unwrap();
        assert_eq!(y.device(), device);
        for (yv, xv) in y.data().unwrap().iter().zip(x.data().unwrap().iter()) {
            assert!(
                (yv - xv.tanh()).abs() < 1e-5,
                "tanh forward: got {yv} expected {} for x={xv}",
                xv.tanh()
            );
        }

        let xr = t.inverse(&y).unwrap();
        assert_eq!(xr.device(), device);
        for (xv0, xrv) in x.data().unwrap().iter().zip(xr.data().unwrap().iter()) {
            assert!((xv0 - xrv).abs() < 1e-4, "tanh roundtrip: {xrv} vs {xv0}");
        }

        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        assert_eq!(ldj.device(), device);
        // log(1 - tanh(x)^2) at x=0 is log(1) = 0; at x=0.5, tanh(0.5)^2≈0.21, log(0.79)≈-0.236
        let ldj_data = ldj.data().unwrap();
        // index of x=0 in flat -> position 2
        assert!(
            ldj_data[2].abs() < 1e-5,
            "tanh ldj at x=0: got {}",
            ldj_data[2],
        );
        // index of x=0.5 -> position 3
        let expected = (1.0f32 - 0.5f32.tanh().powi(2)).ln();
        assert!(
            (ldj_data[3] - expected).abs() < 1e-4,
            "tanh ldj at x=0.5: got {} expected {expected}",
            ldj_data[3],
        );
    }

    #[test]
    fn softplus_transform_preserves_device_and_value() {
        let x = from_slice(&[-2.0f32, -0.5, 0.0, 1.0, 2.5, 4.0], &[2, 3]).unwrap();
        let t = SoftplusTransform;
        let device = x.device();

        let y = t.forward(&x).unwrap();
        assert_eq!(y.device(), device);
        for (yv, xv) in y.data().unwrap().iter().zip(x.data().unwrap().iter()) {
            // softplus(x) = log(1 + exp(x)); for x>20 it's x, but our test
            // points are all ≤ 4 so the elementary form is exact within tol.
            let expected = (1.0f32 + xv.exp()).ln();
            assert!(
                (yv - expected).abs() < 1e-5,
                "softplus forward: got {yv} expected {expected} for x={xv}",
            );
        }

        let xr = t.inverse(&y).unwrap();
        assert_eq!(xr.device(), device);
        for (xv0, xrv) in x.data().unwrap().iter().zip(xr.data().unwrap().iter()) {
            assert!(
                (xv0 - xrv).abs() < 1e-3,
                "softplus roundtrip: {xrv} vs {xv0}",
            );
        }

        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        assert_eq!(ldj.device(), device);
        // log|sigmoid(x)| at x=0 is log(0.5) = -ln(2)
        let ldj_data = ldj.data().unwrap();
        // x=0 is at index 2 in the [2,3] flat layout.
        let expected = -(2.0f32.ln());
        assert!(
            (ldj_data[2] - expected).abs() < 1e-5,
            "softplus ldj at x=0: got {} expected {expected}",
            ldj_data[2],
        );
    }

    #[test]
    fn compose_transform_chain_preserves_device() {
        // Chain: Affine(loc=1, scale=2) then Exp; check forward/inverse/ldj
        // numerical correctness AND device preservation end-to-end.
        let x = from_slice(&[-1.0f32, 0.0, 1.0], &[3]).unwrap();
        let device = x.device();
        let t: ComposeTransform<f32> = ComposeTransform::new(vec![
            Box::new(AffineTransform::new(1.0f32, 2.0f32)),
            Box::new(ExpTransform),
        ]);

        let y = t.forward(&x).unwrap();
        assert_eq!(y.device(), device);
        // y = exp(1 + 2*x)
        for (yv, xv) in y.data().unwrap().iter().zip(x.data().unwrap().iter()) {
            let expected = (1.0f32 + 2.0 * xv).exp();
            assert!(
                (yv - expected).abs() < 1e-4,
                "compose forward: got {yv} expected {expected}",
            );
        }

        let xr = t.inverse(&y).unwrap();
        assert_eq!(xr.device(), device);
        for (xv0, xrv) in x.data().unwrap().iter().zip(xr.data().unwrap().iter()) {
            assert!(
                (xv0 - xrv).abs() < 1e-4,
                "compose roundtrip: {xrv} vs {xv0}",
            );
        }

        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        assert_eq!(ldj.device(), device);
        // ldj = ln(2) + (1 + 2*x): affine contributes ln|2|, exp contributes
        // its argument (which is 1 + 2*x).
        for (lv, xv) in ldj.data().unwrap().iter().zip(x.data().unwrap().iter()) {
            let expected = 2.0f32.ln() + (1.0 + 2.0 * xv);
            assert!(
                (lv - expected).abs() < 1e-4,
                "compose ldj: got {lv} expected {expected}",
            );
        }
    }

    #[test]
    fn transformed_distribution_log_prob_preserves_device() {
        // LogNormal(0,1) at value=e: log_prob_normal(1) - 1.
        // Verifies the device-resident log_prob path numerically and asserts
        // device preservation.
        use crate::Normal;
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let base = Normal::new(loc, scale).unwrap();
        let td = TransformedDistribution::new(Box::new(base), vec![Box::new(ExpTransform)]);

        // Use a multi-element input to exercise broadcasting/sum_ldj paths.
        let value = from_slice(
            &[1.0f32, std::f32::consts::E, std::f32::consts::E.powi(2)],
            &[3],
        )
        .unwrap();
        let device = value.device();

        let lp = td.log_prob(&value).unwrap();
        assert_eq!(lp.device(), device, "log_prob changed device");
        assert_eq!(lp.shape(), &[3]);

        // Reference: log_prob_lognormal(y; mu=0, sigma=1) =
        //   -0.5*ln(2*pi) - 0.5*(ln y)^2 - ln y.
        let two_pi_ln = (2.0f32 * std::f32::consts::PI).ln();
        let lp_data = lp.data().unwrap();
        for (lv, yv) in lp_data
            .iter()
            .zip([1.0f32, std::f32::consts::E, std::f32::consts::E.powi(2)].iter())
        {
            let ln_y = yv.ln();
            let expected = -0.5 * two_pi_ln - 0.5 * ln_y * ln_y - ln_y;
            assert!(
                (lv - expected).abs() < 1e-4,
                "td log_prob at y={yv}: got {lv} expected {expected}",
            );
        }
    }

    // -- #1373: the 11 newly-ported upstream transforms --------------------
    //
    // Each expected value is OracleDerived from live torch 2.11
    // (`torch.distributions.transforms.<T>`), see the constant blocks. R-CHAR-3:
    // every reference vector below is the printed output of the upstream class
    // on the same input, not a self-referential ferrotorch constant.

    fn approx_slice(got: &[f32], want: &[f32], tol: f32, label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length mismatch");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < tol,
                "{label}[{i}]: got {g}, want {w}, |err|={}",
                (g - w).abs()
            );
        }
    }

    // -- AbsTransform --------------------------------------------------------

    #[test]
    fn test_abs_transform_forward_inverse() {
        // torch: AbsTransform()(tensor([-2,0,3])) = [2,0,3]; inv([2,0,3]) = [2,0,3].
        let t = AbsTransform;
        let x = from_slice(&[-2.0f32, 0.0, 3.0], &[3]).unwrap();
        let y = Transform::forward(&t, &x).unwrap();
        approx_slice(y.data().unwrap(), &[2.0, 0.0, 3.0], 1e-6, "abs forward");
        let yin = from_slice(&[2.0f32, 0.0, 3.0], &[3]).unwrap();
        let xr = Transform::inverse(&t, &yin).unwrap();
        approx_slice(xr.data().unwrap(), &[2.0, 0.0, 3.0], 1e-6, "abs inverse");
        // Non-bijective: codomain Positive, log_abs_det_jacobian errors.
        assert_eq!(
            <AbsTransform as Transform<f32>>::codomain(&t).name(),
            "Positive"
        );
        assert!(!<AbsTransform as Transform<f32>>::bijective(&t));
        assert!(Transform::log_abs_det_jacobian(&t, &x, &y).is_err());
    }

    // -- PowerTransform ------------------------------------------------------

    #[test]
    fn test_power_transform() {
        // torch: PowerTransform(2.0): forward([1,2,3])=[1,4,9]; inv=[1,2,3];
        // ldj = (2*y/x).abs().log() = [ln(2), ln(4), ln(6)] = [.6931,1.3863,1.7918].
        let t = PowerTransform::new(2.0f32);
        let x = from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let y = Transform::forward(&t, &x).unwrap();
        approx_slice(y.data().unwrap(), &[1.0, 4.0, 9.0], 1e-5, "power forward");
        let xr = Transform::inverse(&t, &y).unwrap();
        approx_slice(xr.data().unwrap(), &[1.0, 2.0, 3.0], 1e-5, "power inverse");
        let ldj = Transform::log_abs_det_jacobian(&t, &x, &y).unwrap();
        approx_slice(
            ldj.data().unwrap(),
            &[0.6931472, 1.3862944, 1.7917595],
            1e-5,
            "power ldj",
        );
        assert_eq!(t.domain().name(), "Positive");
        assert_eq!(t.codomain().name(), "Positive");
        assert_eq!(Transform::sign(&t), Some(1));
    }

    // -- SoftmaxTransform ----------------------------------------------------

    #[test]
    fn test_softmax_transform() {
        // torch: SoftmaxTransform()([1,2,3]) = [.09003,.24473,.66524];
        // inv = log(y) = [-2.40761,-1.40761,-0.40761].
        let t = SoftmaxTransform;
        let x = from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let y = Transform::forward(&t, &x).unwrap();
        approx_slice(
            y.data().unwrap(),
            &[0.09003057, 0.24472848, 0.66524094],
            1e-6,
            "softmax forward",
        );
        let xr = Transform::inverse(&t, &y).unwrap();
        approx_slice(
            xr.data().unwrap(),
            &[-2.4076059, -1.4076059, -0.40760598],
            1e-5,
            "softmax inverse",
        );
        assert_eq!(
            <SoftmaxTransform as Transform<f32>>::domain(&t).name(),
            "RealVector"
        );
        assert_eq!(
            <SoftmaxTransform as Transform<f32>>::codomain(&t).name(),
            "Simplex"
        );
        assert!(!<SoftmaxTransform as Transform<f32>>::bijective(&t));
        assert!(Transform::log_abs_det_jacobian(&t, &x, &y).is_err());
    }

    // -- StickBreakingTransform ----------------------------------------------

    #[test]
    fn test_stick_breaking_transform() {
        // torch: StickBreakingTransform()([0.5,-1.0,0.3]) (K-1=3 -> K=4) =
        //   [0.35466126, 0.10026139, 0.31311563, 0.23196176];
        // inverse round-trips to [0.5,-1.0,0.3]; ldj = -5.95893.
        let t = StickBreakingTransform;
        let x = from_slice(&[0.5f32, -1.0, 0.3], &[3]).unwrap();
        let y = Transform::forward(&t, &x).unwrap();
        assert_eq!(y.shape(), &[4]);
        approx_slice(
            y.data().unwrap(),
            &[0.35466126, 0.10026139, 0.31311563, 0.23196176],
            1e-5,
            "stick forward",
        );
        let xr = Transform::inverse(&t, &y).unwrap();
        approx_slice(xr.data().unwrap(), &[0.5, -1.0, 0.3], 1e-4, "stick inverse");
        let ldj = Transform::log_abs_det_jacobian(&t, &x, &y).unwrap();
        assert!(
            (ldj.item().unwrap() - (-5.9589319)).abs() < 1e-4,
            "stick ldj: got {}, want -5.9589319",
            ldj.item().unwrap()
        );
        assert_eq!(
            <StickBreakingTransform as Transform<f32>>::domain(&t).name(),
            "RealVector"
        );
        assert_eq!(
            <StickBreakingTransform as Transform<f32>>::codomain(&t).name(),
            "Simplex"
        );
    }

    // -- LowerCholeskyTransform ----------------------------------------------

    #[test]
    fn test_lower_cholesky_transform() {
        // torch: LowerCholeskyTransform()([[.5,1],[2,-.3]]) =
        //   [[1.64872122, 0],[2, 0.7408182]] (diag exp'd, upper zeroed);
        // inverse round-trips to [[.5,0],[2,-.3]] (upper dropped, diag log'd).
        let t = LowerCholeskyTransform;
        let x = from_slice(&[0.5f32, 1.0, 2.0, -0.3], &[2, 2]).unwrap();
        let y = Transform::forward(&t, &x).unwrap();
        approx_slice(
            y.data().unwrap(),
            &[1.6487212, 0.0, 2.0, 0.7408182],
            1e-5,
            "lowerchol forward",
        );
        let xr = Transform::inverse(&t, &y).unwrap();
        approx_slice(
            xr.data().unwrap(),
            &[0.5, 0.0, 2.0, -0.3],
            1e-5,
            "lowerchol inverse",
        );
        assert_eq!(
            <LowerCholeskyTransform as Transform<f32>>::codomain(&t).name(),
            "LowerCholesky"
        );
        assert_eq!(<LowerCholeskyTransform as Transform<f32>>::event_dim(&t), 2);
    }

    // -- CorrCholeskyTransform -----------------------------------------------

    #[test]
    fn test_corr_cholesky_transform() {
        // torch: CorrCholeskyTransform()([0.2,-0.5,0.8]) (N=3 -> 3x3) =
        //   [[1,0,0],[0.19737533,0.98032802,0],[-0.46211717,0.58888036,0.66307443]];
        // inverse round-trips to [0.2,-0.5,0.8]; ldj = -0.98158675.
        let t = CorrCholeskyTransform;
        let x = from_slice(&[0.2f32, -0.5, 0.8], &[3]).unwrap();
        let y = Transform::forward(&t, &x).unwrap();
        assert_eq!(y.shape(), &[3, 3]);
        approx_slice(
            y.data().unwrap(),
            &[
                1.0,
                0.0,
                0.0,
                0.19737533,
                0.98032802,
                0.0,
                -0.46211717,
                0.58888036,
                0.66307443,
            ],
            1e-5,
            "corrchol forward",
        );
        let xr = Transform::inverse(&t, &y).unwrap();
        approx_slice(
            xr.data().unwrap(),
            &[0.2, -0.5, 0.8],
            1e-4,
            "corrchol inverse",
        );
        let ldj = Transform::log_abs_det_jacobian(&t, &x, &y).unwrap();
        assert!(
            (ldj.item().unwrap() - (-0.98158675)).abs() < 1e-4,
            "corrchol ldj: got {}, want -0.98158675",
            ldj.item().unwrap()
        );
        assert_eq!(
            <CorrCholeskyTransform as Transform<f32>>::domain(&t).name(),
            "RealVector"
        );
        assert_eq!(
            <CorrCholeskyTransform as Transform<f32>>::codomain(&t).name(),
            "CorrCholesky"
        );
    }

    // -- ReshapeTransform ----------------------------------------------------

    #[test]
    fn test_reshape_transform() {
        // torch: ReshapeTransform([2,2],[4])([[1,2],[3,4]]) = [1,2,3,4];
        // ldj over a batch shape () is a scalar zero.
        let t: ReshapeTransform = ReshapeTransform::new(vec![2, 2], vec![4]).unwrap();
        let x = from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let y = Transform::forward(&t, &x).unwrap();
        assert_eq!(y.shape(), &[4]);
        approx_slice(
            y.data().unwrap(),
            &[1.0, 2.0, 3.0, 4.0],
            1e-6,
            "reshape forward",
        );
        let xr = Transform::inverse(&t, &y).unwrap();
        assert_eq!(xr.shape(), &[2, 2]);
        approx_slice(
            xr.data().unwrap(),
            &[1.0, 2.0, 3.0, 4.0],
            1e-6,
            "reshape inverse",
        );
        let ldj = Transform::log_abs_det_jacobian(&t, &x, &y).unwrap();
        // batch shape is empty -> 0-D zero.
        assert!(
            ldj.data().unwrap().iter().all(|&v| v == 0.0),
            "reshape ldj must be zeros"
        );
        // Mismatched element counts must error.
        assert!(ReshapeTransform::new(vec![2, 2], vec![3]).is_err());
    }

    // -- IndependentTransform ------------------------------------------------

    #[test]
    fn test_independent_transform() {
        // torch: IndependentTransform(Exp(),1) over [[0,1,2]]: forward = exp;
        // ldj sums the rightmost 1 dim of x -> [3.0].
        let t: IndependentTransform<f32> = IndependentTransform::new(Box::new(ExpTransform), 1);
        let x = from_slice(&[0.0f32, 1.0, 2.0], &[1, 3]).unwrap();
        let y = t.forward(&x).unwrap();
        approx_slice(
            y.data().unwrap(),
            &[1.0, std::f32::consts::E, std::f32::consts::E.powi(2)],
            1e-4,
            "indep forward",
        );
        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        assert_eq!(ldj.shape(), &[1]);
        assert!(
            (ldj.item().unwrap() - 3.0).abs() < 1e-5,
            "indep ldj: got {}, want 3.0",
            ldj.item().unwrap()
        );
        assert_eq!(t.event_dim(), 1);
        assert!(t.bijective());
    }

    // -- CatTransform --------------------------------------------------------

    #[test]
    fn test_cat_transform() {
        // torch: CatTransform([Exp, identity], dim=0, lengths=[2,2])([0,1,5,6]) =
        //   [1, e, 5, 6]; inverse = [0,1,5,6]; ldj = cat([x_exp_part, zeros]) =
        //   [0, 1, 0, 0].
        let t: CatTransform<f32> = CatTransform::new(
            vec![
                Box::new(ExpTransform),
                Box::new(ComposeTransform::new(vec![])),
            ],
            0,
            vec![2, 2],
        )
        .unwrap();
        let x = from_slice(&[0.0f32, 1.0, 5.0, 6.0], &[4]).unwrap();
        let y = t.forward(&x).unwrap();
        approx_slice(
            y.data().unwrap(),
            &[1.0, std::f32::consts::E, 5.0, 6.0],
            1e-4,
            "cat forward",
        );
        let xr = t.inverse(&y).unwrap();
        approx_slice(
            xr.data().unwrap(),
            &[0.0, 1.0, 5.0, 6.0],
            1e-4,
            "cat inverse",
        );
        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        approx_slice(ldj.data().unwrap(), &[0.0, 1.0, 0.0, 0.0], 1e-4, "cat ldj");
    }

    // -- StackTransform ------------------------------------------------------

    #[test]
    fn test_stack_transform() {
        // torch: StackTransform([Exp, Affine(0,2)], dim=0)([[0,1,2],[1,2,3]]) =
        //   [[1, e, e^2],[2,4,6]]; ldj = [[0,1,2],[ln2,ln2,ln2]].
        let t: StackTransform<f32> = StackTransform::new(
            vec![
                Box::new(ExpTransform),
                Box::new(AffineTransform::new(0.0, 2.0)),
            ],
            0,
        );
        let x = from_slice(&[0.0f32, 1.0, 2.0, 1.0, 2.0, 3.0], &[2, 3]).unwrap();
        let y = t.forward(&x).unwrap();
        let e = std::f32::consts::E;
        approx_slice(
            y.data().unwrap(),
            &[1.0, e, e * e, 2.0, 4.0, 6.0],
            1e-4,
            "stack forward",
        );
        let xr = t.inverse(&y).unwrap();
        approx_slice(
            xr.data().unwrap(),
            &[0.0, 1.0, 2.0, 1.0, 2.0, 3.0],
            1e-4,
            "stack inverse",
        );
        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        let l2 = 2.0f32.ln();
        approx_slice(
            ldj.data().unwrap(),
            &[0.0, 1.0, 2.0, l2, l2, l2],
            1e-4,
            "stack ldj",
        );
    }

    // -- CumulativeDistributionTransform -------------------------------------

    #[test]
    fn test_cumulative_distribution_transform() {
        // torch: CumulativeDistributionTransform(Normal(0,1))([-1,0,1]):
        //   forward (CDF) = [0.15865526, 0.5, 0.84134471];
        //   inverse (ICDF) round-trips; ldj = log_prob = [-1.4189385, -0.9189385, -1.4189385].
        use crate::Normal;
        let base = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let t: CumulativeDistributionTransform<f32> =
            CumulativeDistributionTransform::new(Box::new(base));
        let x = from_slice(&[-1.0f32, 0.0, 1.0], &[3]).unwrap();
        let y = Transform::forward(&t, &x).unwrap();
        approx_slice(
            y.data().unwrap(),
            &[0.15865526, 0.5, 0.8413447],
            1e-5,
            "cdf forward",
        );
        let xr = Transform::inverse(&t, &y).unwrap();
        approx_slice(xr.data().unwrap(), &[-1.0, 0.0, 1.0], 1e-4, "cdf inverse");
        let ldj = Transform::log_abs_det_jacobian(&t, &x, &y).unwrap();
        approx_slice(
            ldj.data().unwrap(),
            &[-1.4189385, -0.9189385, -1.4189385],
            1e-5,
            "cdf ldj",
        );
        assert_eq!(
            <CumulativeDistributionTransform<f32> as Transform<f32>>::codomain(&t).name(),
            "UnitInterval"
        );
        assert_eq!(Transform::sign(&t), Some(1));
    }

    // -- production consumer: TransformedDistribution over a new transform ---

    #[test]
    fn test_transformed_distribution_with_power_transform() {
        // Production consumer of PowerTransform (#1373): build a
        // TransformedDistribution and exercise sample/log_prob through the
        // boxed Transform chain. Normal(0,1) is positive-supported only after
        // exp; we compose [Exp, Power(2)] = (exp x)^2 = exp(2x), positive.
        use crate::Normal;
        let base = Normal::new(scalar(0.0f32).unwrap(), scalar(1.0f32).unwrap()).unwrap();
        let td = TransformedDistribution::new(
            Box::new(base),
            vec![
                Box::new(ExpTransform),
                Box::new(PowerTransform::new(2.0f32)),
            ],
        );
        let s = td.sample(&[50]).unwrap();
        assert_eq!(s.shape(), &[50]);
        for &v in s.data().unwrap() {
            assert!(v > 0.0, "exp-then-power sample must be positive, got {v}");
        }
        // support() = chain's last codomain = Power's codomain = Positive.
        assert_eq!(td.support().unwrap().name(), "Positive");
    }
}
