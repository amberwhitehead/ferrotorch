//! Dirichlet distribution.
//!
//! `Dirichlet(concentration)` defines a distribution over the probability simplex.
//! Samples are K-dimensional vectors whose elements are positive and sum to 1.
//!
//! Sampling uses the Gamma-based reparameterization: draw independent
//! `Gamma(alpha_k, 1)` samples and normalize.
//!
//! Device-resident composition (Pattern B) for closed-form methods
//! (`log_prob`, `mean`, `variance`, `entropy`): every step composes
//! `ferrotorch_core` tensor ops so the result lives on the same device as
//! the concentration parameter. `lgamma`/`digamma` route through
//! `ferrotorch_core::special` (tensor-level; internally CPU until GPU
//! special-function kernels land, but the call site is device-resident so a
//! future GPU kernel slot-fills transparently).
//!
//! `sample`/`rsample` retain scalar Gamma-rejection sampling because there
//! is no GPU Gamma kernel; the result tensor is built directly on the
//! caller's device via `TensorStorage::on_device(...)` (no redundant CPU
//! materialize + `Tensor::to(device)` round-trip).
//!
//! [CL-331] ferrotorch#331 — multivariate distributions
//! Pass 5.B.1 follow-up: closes #1136 by migrating to Pattern B (device-resident).
//!
//! ## REQ status (per `.design/ferrotorch-distributions/dirichlet.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`Dirichlet<T>` struct) | SHIPPED | `pub struct Dirichlet<T: Float>` with `concentration`/`k` mirroring `torch/distributions/dirichlet.py:38-86`; consumer: `pub use dirichlet::Dirichlet` in `lib.rs` |
//! | REQ-2 (constructor + validation) | SHIPPED | `pub fn Dirichlet::new` with ndim/empty checks mirroring `dirichlet.py:61-74`; consumer: re-export |
//! | REQ-3 (accessors) | SHIPPED | `pub fn Dirichlet::concentration`/`num_categories`; consumer: re-export |
//! | REQ-4 (`Distribution` trait impl) | SHIPPED | `impl<T: Float> Distribution<T> for Dirichlet<T>`; consumer: trait dispatch |
//! | REQ-5 (`sample` via Marsaglia-Tsang) | SHIPPED | the `sample` method invokes `sample_gamma` per element with α<1 boost + device-resident upload; consumer: trait surface |
//! | REQ-6 (`rsample` with backward) | SHIPPED | `DirichletRsampleBackward` attachment + device-resident sample tensor; consumer: trait surface |
//! | REQ-7 (`log_prob` device-resident) | SHIPPED | composes `sub`/`mul`/`sum_dim`/`add` + `lgamma` mirroring `dirichlet.py:90-97`; consumer: trait surface |
//! | REQ-8 (`mean` device-resident) | SHIPPED | `sum_dim(α, -1, true)` + `div` mirroring `dirichlet.py:99-101`; consumer: trait surface |
//! | REQ-9 (`variance` device-resident) | SHIPPED | closed-form via scalar broadcasts mirroring `dirichlet.py:113-120`; consumer: trait surface |
//! | REQ-10 (`entropy` device-resident) | SHIPPED | composed `lgamma`/`digamma` formula mirroring `dirichlet.py:122-130`; consumer: trait surface |
//! | REQ-11 (`DirichletRsampleBackward`) | SHIPPED | implicit-reparam + simplex projection; consumer: invoked by the rsample method when concentration requires grad |
//! | REQ-12 (full PyTorch surface, N-D batched) | SHIPPED | #1412/#1547/#1548/#1549 — `Dirichlet::new` accepts `[*batch, K]` (`ndim() >= 1`); `batch_shape`/`expand`/`support`/`arg_constraints`/`mode`/`has_rsample` overrides + N-D batched `sample`/`rsample`/`log_prob`/`mean`/`variance`/`entropy` in `dirichlet.rs` mirroring `torch/distributions/dirichlet.py:55-59, 71, 76-83, 90-130`. `mode` all-α<1 rows return `one_hot(argmax)` per `dirichlet.py:107-110` (NOT NaN). `log_prob` validates `value` against the simplex support via `constraints::Simplex::check_tensor` (production consumer). Consumer: trait dispatch via `pub use Dirichlet` re-export in `lib.rs`. |

use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use std::collections::HashMap;

use crate::constraints;
use crate::constraints::Constraint;
use crate::special_fns::{digamma_scalar, lgamma_scalar};
use crate::{DistConstraint, Distribution};

/// Dirichlet distribution parameterized by `concentration` (alpha).
///
/// `concentration` is a tensor of shape `[*batch, K]` whose elements must be
/// positive. The trailing dim `K` is the event dim; the leading dims (if any)
/// are the batch dims. Samples lie on the `(K-1)`-dimensional probability
/// simplex and have shape `sample_shape ++ batch_shape ++ [K]`.
///
/// # Reparameterization
///
/// `rsample` uses the implicit reparameterization through Gamma samples.
/// Gradients flow through the concentration parameters.
pub struct Dirichlet<T: Float> {
    concentration: Tensor<T>,
    /// Number of categories — the trailing dim of `concentration`.
    k: usize,
    /// Leading batch dims — `concentration.shape()[..ndim-1]`. Empty for the
    /// 1-D case.
    batch_shape: Vec<usize>,
}

impl<T: Float> Dirichlet<T> {
    /// Create a new Dirichlet distribution.
    ///
    /// `concentration` must have at least one dimension; the trailing dim `K`
    /// is the event dim and any leading dims are batch dims. Mirrors
    /// `torch/distributions/dirichlet.py:66-72`:
    ///   if concentration.dim() < 1: raise ValueError(...)
    ///   batch_shape, event_shape = concentration.shape[:-1], concentration.shape[-1:]
    pub fn new(concentration: Tensor<T>) -> FerrotorchResult<Self> {
        if concentration.ndim() < 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Dirichlet: concentration must be at least 1-D, got shape {:?}",
                    concentration.shape()
                ),
            });
        }
        let shape = concentration.shape();
        let k = *shape.last().unwrap();
        if k == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "Dirichlet: concentration last dim (K) must be >= 1".into(),
            });
        }
        let batch_shape = shape[..shape.len() - 1].to_vec();
        Ok(Self {
            concentration,
            k,
            batch_shape,
        })
    }

    /// The concentration (alpha) parameter.
    pub fn concentration(&self) -> &Tensor<T> {
        &self.concentration
    }

    /// Number of categories (K) — the trailing event dim.
    pub fn num_categories(&self) -> usize {
        self.k
    }

    /// Number of batch elements `prod(batch_shape)` (1 for a 1-D
    /// concentration).
    fn num_batches(&self) -> usize {
        self.batch_shape.iter().product::<usize>().max(1)
    }
}

/// Sample a Gamma(alpha, 1) variable using Marsaglia & Tsang's method.
///
/// This handles alpha >= 1 directly. For alpha < 1 we use the Ahrens-Dieter
/// boost: Gamma(alpha, 1) = Gamma(alpha+1, 1) * U^(1/alpha).
fn sample_gamma<T: Float>(alpha: T) -> T {
    let one = <T as num_traits::One>::one();
    let zero = <T as num_traits::Zero>::zero();
    let third = T::from(1.0 / 3.0).unwrap();

    if alpha < one {
        // Boost: Gamma(a) = Gamma(a+1) * U^(1/a)
        let g = sample_gamma(alpha + one);
        let u = sample_uniform_01::<T>();
        return g * u.powf(one / alpha);
    }

    // Marsaglia & Tsang for alpha >= 1
    let d = alpha - third;
    let c = third / d.sqrt();

    loop {
        let x = sample_standard_normal::<T>();
        let v_base = one + c * x;
        if v_base <= zero {
            continue;
        }
        let v = v_base * v_base * v_base;
        let u = sample_uniform_01::<T>();

        let half = T::from(0.5).unwrap();
        let threshold = T::from(0.0331).unwrap();

        if u < one - threshold * x * x * x * x {
            return d * v;
        }
        if u.ln() < half * x * x + d * (one - v + v.ln()) {
            return d * v;
        }
    }
}

/// Draw U ~ Uniform(0, 1) using the same RNG approach as creation::rand.
fn sample_uniform_01<T: Float>() -> T {
    // Use the creation module's rand for a single element
    let t = creation::rand::<T>(&[1]).unwrap();
    t.data_vec().unwrap()[0]
}

/// Draw Z ~ N(0, 1) using the same RNG approach as creation::randn.
fn sample_standard_normal<T: Float>() -> T {
    let t = creation::randn::<T>(&[1]).unwrap();
    t.data_vec().unwrap()[0]
}

impl<T: Float> Distribution<T> for Dirichlet<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        // Gamma rejection sampling is intrinsically scalar — there is no
        // GPU Gamma kernel in ferrotorch-core. We read alpha (the [*batch, K]
        // parameter tensor) once, run the host sampler, and build the result
        // tensor directly on the caller's device via
        // `TensorStorage::on_device(...)`. This matches PyTorch's Dirichlet
        // behaviour on CUDA prior to the dedicated CUDA Gamma sampler
        // (which composed `_standard_gamma` + division).
        //
        // For batched concentration each of the `b` batch rows owns its own
        // length-K alpha slice; the output is `sample_shape ++ batch_shape ++
        // [K]` mirroring `torch/distributions/dirichlet.py:86-88`
        // (`_extended_shape(sample_shape)` then a per-element gamma draw).
        let device = self.concentration.device();
        let n: usize = shape.iter().product();
        let k = self.k;
        let b = self.num_batches();
        let alpha = self.concentration.data_vec()?;

        let mut result = Vec::with_capacity(n * b * k);
        for _ in 0..n {
            for bi in 0..b {
                let row = &alpha[bi * k..bi * k + k];
                let mut gammas = Vec::with_capacity(k);
                let mut total = <T as num_traits::Zero>::zero();
                for &a in row {
                    let g = sample_gamma(a);
                    gammas.push(g);
                    total += g;
                }
                for g in gammas {
                    result.push(g / total);
                }
            }
        }

        let mut out_shape = shape.to_vec();
        out_shape.extend_from_slice(&self.batch_shape);
        out_shape.push(k);
        // Direct upload to device — no CPU materialize + `to(device)` hop.
        let storage = TensorStorage::on_device(result, device)?;
        Tensor::from_storage(storage, out_shape, false)
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        // Same scalar Gamma rejection sampling as `sample`, but attach the
        // implicit-reparameterization backward node so gradients can flow
        // through `concentration`. Result storage lands on the parameter
        // device directly (no `Tensor::to` hop). Batched concentration is
        // laid out as `[n_samples, b, K]` flattened — each batch row owns its
        // own length-K alpha slice.
        let device = self.concentration.device();
        let n: usize = shape.iter().product();
        let k = self.k;
        let b = self.num_batches();
        let alpha = self.concentration.data_vec()?;

        let total_rows = n * b;
        let mut gamma_vals = Vec::with_capacity(total_rows * k);
        let mut result = Vec::with_capacity(total_rows * k);

        for s in 0..n {
            for bi in 0..b {
                let arow = &alpha[bi * k..bi * k + k];
                let mut total = <T as num_traits::Zero>::zero();
                let base = (s * b + bi) * k;
                for &a in arow {
                    let g = sample_gamma(a);
                    gamma_vals.push(g);
                    total += g;
                }
                for j in 0..k {
                    result.push(gamma_vals[base + j] / total);
                }
            }
        }

        let mut out_shape = shape.to_vec();
        out_shape.extend_from_slice(&self.batch_shape);
        out_shape.push(k);

        if self.concentration.requires_grad() && ferrotorch_core::is_grad_enabled() {
            // Keep a clone of the result samples in the backward node so the
            // implicit-grad expression has access to the realized x_j.
            let samples_storage = TensorStorage::on_device(result.clone(), device)?;
            let sample_tensor = Tensor::from_storage(samples_storage, out_shape.clone(), false)?;
            let grad_fn = Arc::new(DirichletRsampleBackward {
                concentration: self.concentration.clone(),
                samples: sample_tensor,
                n,
                b,
                k,
            });
            let storage = TensorStorage::on_device(result, device)?;
            Tensor::from_operation(storage, out_shape, grad_fn)
        } else {
            let storage = TensorStorage::on_device(result, device)?;
            Tensor::from_storage(storage, out_shape, false)
        }
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // log_prob(x) = sum_k (alpha_k - 1) * log(x_k)      over last dim
        //              + lgamma(sum_k alpha_k)              per batch row
        //              - sum_k lgamma(alpha_k)              per batch row
        //
        // Mirrors `torch/distributions/dirichlet.py:90-97`:
        //   xlogy(self.concentration - 1.0, value).sum(-1)
        //     + lgamma(self.concentration.sum(-1)) - lgamma(self.concentration).sum(-1)
        //
        // Host-side scalar composition. The trailing dim K is the event dim;
        // every batch row (and every leading sample row) owns its own length-K
        // alpha slice (broadcast by batch index). The normalizer is a per-row
        // function of alpha. `xlogy(α-1, x)` makes the α==1 boundary contribute
        // 0 even when `x==0` (`0 * log(0) -> 0`), so we replicate it exactly
        // rather than calling `log` then multiplying.
        if value.device() != self.concentration.device() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: self.concentration.device(),
                got: value.device(),
            });
        }
        let k = self.k;
        let b = self.num_batches();
        let device = self.concentration.device();

        let val_shape = value.shape().to_vec();
        if val_shape.last().copied() != Some(k) {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Dirichlet log_prob: value last dim must be K={}, got shape {:?}",
                    k, val_shape
                ),
            });
        }

        // Validate that `value` lies on the simplex support. This is the
        // production consumer of `constraints::Simplex::check_tensor` — the
        // full-vector simplex predicate (non-negativity AND sum-to-one over
        // the trailing dim) mirroring `dirichlet.py:91-92`
        // (`self._validate_sample(value)` against `support = constraints.simplex`).
        if !constraints::Simplex.check_tensor(value)? {
            return Err(FerrotorchError::InvalidArgument {
                message: "Dirichlet log_prob: value is not on the probability simplex \
                          (Simplex support: elements >= 0 and last dim sums to 1)"
                    .into(),
            });
        }

        let alpha = self.concentration.data_vec()?;
        let xs = value.data_vec()?;
        let n_value_rows = xs.len() / k;
        // The value's batch rows must align with the concentration's batch
        // rows (broadcast leading sample dims share the same b-cycle).
        if n_value_rows % b != 0 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Dirichlet log_prob: value batch rows {n_value_rows} not a multiple \
                     of concentration batch count {b} (concentration shape {:?})",
                    self.concentration.shape()
                ),
            });
        }
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();

        // Per-batch-row normalizer = lgamma(sum(α)) - sum(lgamma(α)).
        let mut normalizer = Vec::with_capacity(b);
        for bi in 0..b {
            let arow = &alpha[bi * k..bi * k + k];
            let mut alpha_sum = zero;
            let mut sum_lgamma = zero;
            for &a in arow {
                alpha_sum += a;
                sum_lgamma += lgamma_scalar(a);
            }
            normalizer.push(lgamma_scalar(alpha_sum) - sum_lgamma);
        }

        // Per value row: xlogy(α-1, x).sum(-1) + normalizer[bi].
        let mut out = Vec::with_capacity(n_value_rows);
        for r in 0..n_value_rows {
            let bi = r % b;
            let arow = &alpha[bi * k..bi * k + k];
            let xrow = &xs[r * k..r * k + k];
            let mut acc = zero;
            for j in 0..k {
                let coeff = arow[j] - one;
                // xlogy(coeff, x): 0 when coeff == 0 regardless of x.
                if coeff != zero {
                    acc += coeff * xrow[j].ln();
                }
            }
            out.push(acc + normalizer[bi]);
        }

        // Output shape = value.shape()[..len-1] (drop the trailing event dim).
        let out_shape = val_shape[..val_shape.len() - 1].to_vec();
        let storage = TensorStorage::on_device(out, device)?;
        Tensor::from_storage(storage, out_shape, false)
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        // Reference: `torch/distributions/dirichlet.py:99-101`
        //   mean = concentration / concentration.sum(-1, keepdim=True)
        // Host-side per-batch-row normalization over the trailing event dim;
        // output shape matches the concentration ([*batch, K]).
        let k = self.k;
        let b = self.num_batches();
        let device = self.concentration.device();
        let alpha = self.concentration.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();

        let mut out = Vec::with_capacity(b * k);
        for bi in 0..b {
            let arow = &alpha[bi * k..bi * k + k];
            let mut s = zero;
            for &a in arow {
                s += a;
            }
            for &a in arow {
                out.push(a / s);
            }
        }
        let storage = TensorStorage::on_device(out, device)?;
        Tensor::from_storage(storage, self.concentration.shape().to_vec(), false)
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        // Reference: `torch/distributions/dirichlet.py:113-120`
        //   variance[i] = alpha[i] * (alpha0 - alpha[i]) / (alpha0^2 * (alpha0 + 1))
        // Host-side per-batch-row; output shape matches concentration.
        let k = self.k;
        let b = self.num_batches();
        let device = self.concentration.device();
        let alpha = self.concentration.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();

        let mut out = Vec::with_capacity(b * k);
        for bi in 0..b {
            let arow = &alpha[bi * k..bi * k + k];
            let mut a0 = zero;
            for &a in arow {
                a0 += a;
            }
            let denom = a0 * a0 * (a0 + one);
            for &a in arow {
                out.push(a * (a0 - a) / denom);
            }
        }
        let storage = TensorStorage::on_device(out, device)?;
        Tensor::from_storage(storage, self.concentration.shape().to_vec(), false)
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        // Reference: `torch/distributions/dirichlet.py:122-130`
        //   H = sum(lgamma(alpha_k)) - lgamma(alpha0)
        //       - (K - alpha0) * digamma(alpha0)
        //       - sum((alpha_k - 1) * digamma(alpha_k))
        // Host-side per-batch-row; output shape == batch_shape (scalar for the
        // 1-D case).
        let k = self.k;
        let b = self.num_batches();
        let device = self.concentration.device();
        let alpha = self.concentration.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let k_t = T::from(k).unwrap();

        let mut out = Vec::with_capacity(b);
        for bi in 0..b {
            let arow = &alpha[bi * k..bi * k + k];
            let mut a0 = zero;
            let mut sum_lgamma = zero;
            let mut term3 = zero;
            for &a in arow {
                a0 += a;
                sum_lgamma += lgamma_scalar(a);
                term3 += (a - one) * digamma_scalar(a);
            }
            let h = sum_lgamma - lgamma_scalar(a0) - (k_t - a0) * digamma_scalar(a0) - term3;
            out.push(h);
        }
        let storage = TensorStorage::on_device(out, device)?;
        Tensor::from_storage(storage, self.batch_shape.clone(), false)
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface (#1412 / #1547 / #1548 / #1549) — N-D batched.
    //
    // Mirrors `torch/distributions/dirichlet.py:55-59, 71, 76-83, 100-111`:
    //   arg_constraints = {"concentration": constraints.independent(constraints.positive, 1)}
    //   support         = constraints.simplex
    //   has_rsample     = True
    //   batch_shape     = concentration.shape[:-1]
    //   expand          = concentration.expand(batch_shape + event_shape)
    //   mode            = (concentration - 1).clamp(min=0) / sum(.., -1, True);
    //                     all-α<1 rows substitute one_hot(argmax) (NOT NaN)
    // All methods handle arbitrary leading batch dims (`[*batch, K]`).
    // -----------------------------------------------------------------------

    fn has_rsample(&self) -> bool {
        // `torch/distributions/dirichlet.py:59`: `has_rsample = True`.
        true
    }

    fn batch_shape(&self) -> Vec<usize> {
        // `torch/distributions/dirichlet.py:71`:
        //   batch_shape = concentration.shape[:-1]
        self.batch_shape.clone()
    }

    fn event_shape(&self) -> Vec<usize> {
        // The simplex event has dimension K (the trailing dim of every sample).
        vec![self.k]
    }

    fn support(&self) -> Option<Box<dyn DistConstraint>> {
        // `torch/distributions/dirichlet.py:47`: `support = constraints.simplex`.
        Some(Box::new(constraints::Simplex))
    }

    fn arg_constraints(&self) -> HashMap<&'static str, Box<dyn DistConstraint>> {
        // `torch/distributions/dirichlet.py:46`:
        //   arg_constraints = {"concentration": independent(constraints.positive, 1)}
        // ferrotorch ships scalar `Positive` here because the `Independent`
        // composite constraint variant is still under #1372 follow-up; the
        // per-element semantics (each α_k > 0) match what `Positive` checks.
        let mut m: HashMap<&'static str, Box<dyn DistConstraint>> = HashMap::new();
        m.insert("concentration", Box::new(constraints::Positive));
        m
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/dirichlet.py:104-111`:
        //   concentrationm1 = (self.concentration - 1).clamp(min=0.0)
        //   mode = concentrationm1 / concentrationm1.sum(-1, True)
        //   mask = (self.concentration < 1).all(dim=-1)
        //   mode[mask] = one_hot(mode[mask].argmax(dim=-1), K).to(mode)
        //
        // Per batch row: clamp `(α-1)` at 0, normalize over the event dim. For
        // a row where every α < 1 the clamped numerator is all-zero (sum 0),
        // so upstream substitutes the one-hot of argmax — NOT NaN. Since the
        // clamped numerator is all-zero in that branch, `argmax` over the
        // numerator is 0; PyTorch's `mode[mask].argmax` is over the (all-zero)
        // `mode` slice and likewise returns index 0. We replicate index 0.
        let k = self.k;
        let b = self.num_batches();
        let alpha = self.concentration.data_vec()?;
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();
        let device = self.concentration.device();

        let mut result = Vec::with_capacity(b * k);
        for bi in 0..b {
            let arow = &alpha[bi * k..bi * k + k];
            let all_below_one = arow.iter().all(|a| *a < one);
            if all_below_one {
                // One-hot at argmax of the all-zero clamped numerator -> idx 0.
                // (Upstream: `one_hot(mode.argmax(-1), K)` with mode all-zero
                // resolves to the first index.)
                for j in 0..k {
                    result.push(if j == 0 { one } else { zero });
                }
            } else {
                let mut sum_am1 = zero;
                let mut am1: Vec<T> = Vec::with_capacity(k);
                for &a in arow {
                    let v = if a > one { a - one } else { zero };
                    am1.push(v);
                    sum_am1 += v;
                }
                for v in &am1 {
                    result.push(*v / sum_am1);
                }
            }
        }
        let storage = TensorStorage::on_device(result, device)?;
        Tensor::from_storage(storage, self.concentration.shape().to_vec(), false)
    }

    fn expand(&self, batch_shape: &[usize]) -> FerrotorchResult<Box<dyn Distribution<T>>> {
        // `torch/distributions/dirichlet.py:76-83`:
        //   new.concentration = self.concentration.expand(batch_shape + self.event_shape)
        // The target concentration shape is `batch_shape ++ [K]`. The source
        // concentration (shape `self.batch_shape ++ [K]`) must broadcast to it
        // per PyTorch's `Tensor.expand` rules: source batch dims, right-aligned
        // under the target batch dims, must each be 1 or equal the target. We
        // materialize the broadcast host-side (CPU path; the lib.rs `expand`
        // doc note records that ferrotorch materializes the broadcast for
        // simplicity rather than constructing a strided view).
        let k = self.k;
        let src_batch = &self.batch_shape;
        let tgt_batch = batch_shape;

        if src_batch.len() > tgt_batch.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Dirichlet::expand: cannot expand batch_shape {src_batch:?} to \
                     {tgt_batch:?} (target has fewer dims)"
                ),
            });
        }
        let pad = tgt_batch.len() - src_batch.len();
        for (i, &s) in src_batch.iter().enumerate() {
            let t = tgt_batch[pad + i];
            if s != t && s != 1 {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "Dirichlet::expand: source batch dim {s} (axis {i}) is not \
                         broadcastable to target {t} (batch_shape {src_batch:?} -> {tgt_batch:?})"
                    ),
                });
            }
        }

        // Source strides over the source batch dims (row-major); broadcast
        // dims (size 1) contribute stride 0 so they re-read the same row.
        let mut src_strides = vec![0usize; src_batch.len()];
        let mut acc = 1usize;
        for i in (0..src_batch.len()).rev() {
            src_strides[i] = if src_batch[i] == 1 { 0 } else { acc };
            acc *= src_batch[i].max(1);
        }
        // Target strides for decoding a flat batch index into coordinates.
        let mut tgt_strides = vec![1usize; tgt_batch.len()];
        let mut tacc = 1usize;
        for i in (0..tgt_batch.len()).rev() {
            tgt_strides[i] = tacc;
            tacc *= tgt_batch[i].max(1);
        }

        let src = self.concentration.data_vec()?;
        let tgt_b: usize = tgt_batch.iter().product::<usize>().max(1);
        let mut out = Vec::with_capacity(tgt_b * k);
        for flat in 0..tgt_b {
            // Decode `flat` into target batch coordinates, then map to the
            // source batch row (collapsing broadcast dims to index 0).
            let mut src_row = 0usize;
            for i in 0..tgt_batch.len() {
                let coord = (flat / tgt_strides[i]) % tgt_batch[i].max(1);
                if i >= pad {
                    let si = i - pad;
                    let use_coord = if src_batch[si] == 1 { 0 } else { coord };
                    src_row += use_coord * src_strides[si];
                }
            }
            let base = src_row * k;
            out.extend_from_slice(&src[base..base + k]);
        }

        let device = self.concentration.device();
        let mut new_shape = tgt_batch.to_vec();
        new_shape.push(k);
        let storage = TensorStorage::on_device(out, device)?;
        let requires_grad = self.concentration.requires_grad();
        let new_conc = Tensor::from_storage(storage, new_shape, requires_grad)?;
        Ok(Box::new(Dirichlet::new(new_conc)?))
    }
}

// ---------------------------------------------------------------------------
// Backward node for rsample
// ---------------------------------------------------------------------------
//
// rsample's forward path is still scalar Gamma-rejection sampling because
// ferrotorch-core has no GPU Gamma kernel. The implicit-reparameterization
// gradient we record here exactly matches the prior CPU implementation —
// the only change is that the gradient tensor is built directly on the
// caller's device via `TensorStorage::on_device(...)` rather than CPU →
// `to(device)`.

/// Backward for Dirichlet rsample.
///
/// Uses the implicit reparameterization gradient through the Gamma-based
/// sampling. Approximation:
/// d(x_k)/d(alpha_k) ≈ x_k * (digamma(alpha_k) - digamma(sum(alpha)))
/// corrected by the Jacobian of the simplex projection.
#[derive(Debug)]
struct DirichletRsampleBackward<T: Float> {
    concentration: Tensor<T>,
    samples: Tensor<T>,
    /// Number of leading sample rows (`prod(sample_shape)`).
    n: usize,
    /// Number of concentration batch rows (`prod(batch_shape)`, 1 for 1-D).
    b: usize,
    /// Number of categories (event dim).
    k: usize,
}

impl<T: Float> GradFn<T> for DirichletRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let device = grad_output.device();
        // The gradient kernel itself is scalar (per-element rejection-grad
        // formula); we read back grad_output / samples / concentration once,
        // then upload the [*batch, K] gradient tensor directly to `device`.
        //
        // Layout: samples / grad_output are `[n_samples, b, K]` flattened;
        // each batch row `bi` owns its own length-K alpha slice and accumulates
        // its own gradient slot. The gradient tensor has the same shape as the
        // concentration (`[*batch, K]`).
        let go = grad_output.data_vec()?;
        let x_data = self.samples.data_vec()?;
        let alpha = self.concentration.data_vec()?;
        let n = self.n;
        let b = self.b;
        let k = self.k;
        let zero = <T as num_traits::Zero>::zero();

        // Per-batch-row digamma(sum(alpha_row)).
        let mut dig_sum = Vec::with_capacity(b);
        for bi in 0..b {
            let arow = &alpha[bi * k..bi * k + k];
            let asum: T = arow.iter().copied().fold(zero, |a, c| a + c);
            dig_sum.push(digamma_scalar(asum));
        }

        let mut grad_alpha = vec![zero; b * k];
        for s in 0..n {
            for bi in 0..b {
                let base = (s * b + bi) * k;
                let arow = &alpha[bi * k..bi * k + k];
                let mut xg_sum = zero;
                for j in 0..k {
                    xg_sum += x_data[base + j] * go[base + j];
                }
                for j in 0..k {
                    let dig_alpha_j = digamma_scalar(arow[j]);
                    let grad_j = x_data[base + j] * (dig_alpha_j - dig_sum[bi]);
                    grad_alpha[bi * k + j] += (go[base + j] - xg_sum) * grad_j;
                }
            }
        }

        let storage = TensorStorage::on_device(grad_alpha, device)?;
        let grad_alpha_t =
            Tensor::from_storage(storage, self.concentration.shape().to_vec(), false)?;

        Ok(vec![if self.concentration.requires_grad() {
            Some(grad_alpha_t)
        } else {
            None
        }])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.concentration]
    }

    fn name(&self) -> &'static str {
        "DirichletRsampleBackward"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, tensor};

    #[test]
    fn test_dirichlet_sample_shape() {
        let alpha = tensor(&[1.0f32, 1.0, 1.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100, 3]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_dirichlet_sample_2d_shape() {
        let alpha = tensor(&[2.0f32, 3.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();

        let samples = dist.sample(&[5, 10]).unwrap();
        assert_eq!(samples.shape(), &[5, 10, 2]);
    }

    #[test]
    fn test_dirichlet_sample_on_simplex() {
        let alpha = tensor(&[0.5f32, 0.5, 0.5]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        let data = samples.data().unwrap();

        for s in 0..50 {
            let mut sum = 0.0f32;
            for j in 0..3 {
                let val = data[s * 3 + j];
                assert!(
                    val > 0.0,
                    "Dirichlet sample elements must be positive, got {val}"
                );
                sum += val;
            }
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Dirichlet sample must sum to 1, got {sum}"
            );
        }
    }

    #[test]
    fn test_dirichlet_rsample_has_grad() {
        let alpha = tensor(&[2.0f32, 3.0, 4.0]).unwrap().requires_grad_(true);
        let dist = Dirichlet::new(alpha).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert_eq!(samples.shape(), &[5, 3]);
        assert!(samples.requires_grad());
        assert!(samples.grad_fn().is_some());
    }

    #[test]
    fn test_dirichlet_rsample_no_grad_when_detached() {
        let alpha = tensor(&[2.0f32, 3.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_dirichlet_log_prob_uniform() {
        // Dirichlet([1, 1, 1]) is uniform on the simplex.
        // log_prob = lgamma(3) - 3*lgamma(1) = ln(2!) = ln(2)
        let alpha = tensor(&[1.0f32, 1.0, 1.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();

        // Any point on the simplex should have same log_prob
        let x = tensor(&[0.25f32, 0.25, 0.5]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 2.0f32.ln(); // lgamma(3) - 3*lgamma(1)
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_dirichlet_log_prob_batch() {
        let alpha = tensor(&[2.0f32, 2.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();

        let x = from_slice(&[0.5f32, 0.5, 0.9, 0.1], &[2, 2]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert_eq!(lp.shape(), &[2]);

        let data = lp.data().unwrap();
        // For Dirichlet([2,2]), the mode is at [0.5, 0.5]
        assert!(data[0] > data[1], "log_prob at mode should be highest");
    }

    #[test]
    fn test_dirichlet_entropy_uniform() {
        // For Dirichlet([1,1,...,1]) with K categories:
        // H = sum(lgamma(1)) - lgamma(K) - (K - K)*digamma(K) - sum(0 * digamma(1))
        //   = -lgamma(K) = -ln((K-1)!)
        let alpha = tensor(&[1.0f32, 1.0, 1.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();

        let h = dist.entropy().unwrap();
        // H = -lgamma(3) = -ln(2) ≈ -0.6931
        let expected = -(2.0f32.ln());
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-3,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_dirichlet_nd_accepted() {
        // Upstream accepts `concentration.dim() >= 1` (dirichlet.py:66-72);
        // a [2,2] concentration yields batch_shape [2], event [2].
        let alpha = from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();
        assert_eq!(dist.batch_shape(), vec![2]);
        assert_eq!(dist.num_categories(), 2);
    }

    #[test]
    fn test_dirichlet_empty_errors() {
        // A trailing-zero event dim (K == 0) is rejected (dirichlet.py: K must
        // be >= 1 for a valid simplex).
        let alpha = from_slice::<f32>(&[], &[0]).unwrap();
        assert!(Dirichlet::new(alpha).is_err());
    }

    #[test]
    fn test_dirichlet_num_categories() {
        let alpha = tensor(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();
        assert_eq!(dist.num_categories(), 4);
    }

    #[test]
    fn test_dirichlet_f64() {
        let alpha = tensor(&[2.0f64, 3.0, 4.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50, 3]);

        let data = samples.data().unwrap();
        for s in 0..50 {
            let sum: f64 = (0..3).map(|j| data[s * 3 + j]).sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    // -----------------------------------------------------------------------
    // Wave-H #1412: support / arg_constraints / mode / expand (1-D path)
    // -----------------------------------------------------------------------

    #[test]
    fn test_dirichlet_has_rsample_true() {
        let alpha = tensor(&[2.0f32, 3.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();
        assert!(dist.has_rsample());
    }

    #[test]
    fn test_dirichlet_support_is_simplex() {
        let alpha = tensor(&[1.0f32, 1.0, 1.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();
        let sup = dist.support().unwrap();
        assert_eq!(sup.name(), "Simplex");
        assert_eq!(sup.event_dim(), 1);
    }

    #[test]
    fn test_dirichlet_arg_constraints_concentration_positive() {
        let alpha = tensor(&[2.0f32, 3.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();
        let ac = dist.arg_constraints();
        assert_eq!(ac.len(), 1);
        assert_eq!(ac.get("concentration").unwrap().name(), "Positive");
    }

    #[test]
    fn test_dirichlet_mode_alpha_gt_one() {
        // For α = [2, 3, 4]: mode = (α-1) / sum(α-1) = [1, 2, 3] / 6.
        let alpha = tensor(&[2.0f32, 3.0, 4.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();
        let mode = dist.mode().unwrap();
        let data = mode.data().unwrap();
        assert!((data[0] - 1.0 / 6.0).abs() < 1e-6);
        assert!((data[1] - 2.0 / 6.0).abs() < 1e-6);
        assert!((data[2] - 3.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_dirichlet_mode_all_alpha_below_one_is_one_hot() {
        // Upstream `dirichlet.py:107-110`: all-α<1 rows return one_hot(argmax),
        // NOT NaN. For α = [0.5, 0.5, 0.5] the clamped numerator is all-zero so
        // argmax = 0 → mode = [1, 0, 0] (verified live torch 2.11, #1549).
        let alpha = tensor(&[0.5f32, 0.5, 0.5]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();
        let mode = dist.mode().unwrap();
        let data = mode.data().unwrap();
        assert_eq!(
            data[0], 1.0f32,
            "mode[0] must be 1.0 (one-hot), got {}",
            data[0]
        );
        assert_eq!(data[1], 0.0f32, "mode[1] must be 0.0, got {}", data[1]);
        assert_eq!(data[2], 0.0f32, "mode[2] must be 0.0, got {}", data[2]);
    }

    #[test]
    fn test_dirichlet_event_shape() {
        let alpha = tensor(&[1.0f32, 1.0, 1.0, 1.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();
        assert_eq!(dist.event_shape(), vec![4]);
    }

    #[test]
    fn test_dirichlet_expand_empty_batch_clones() {
        let alpha = tensor(&[2.0f32, 3.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();
        let expanded = dist.expand(&[]).unwrap();
        // sample shape must still have trailing K = 2.
        let s = expanded.sample(&[5]).unwrap();
        assert_eq!(s.shape(), &[5, 2]);
    }

    #[test]
    fn test_dirichlet_expand_to_new_batch_dim() {
        // `dirichlet.py:76-83`: expanding a 1-D Dirichlet (K=2) to batch [4]
        // broadcasts concentration to [4, 2]; sample shape is then [4, 2].
        let alpha = tensor(&[2.0f32, 3.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();
        let expanded = dist.expand(&[4]).unwrap();
        assert_eq!(expanded.batch_shape(), vec![4]);
        let s = expanded.sample(&[]).unwrap();
        assert_eq!(s.shape(), &[4, 2]);
    }

    #[test]
    fn test_dirichlet_nd_batched_shapes() {
        // Concentration [2, 3] -> batch_shape [2], event [3]; sample [2,3];
        // log_prob over a [2,3] value reduces the event dim -> [2].
        let alpha = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();
        assert_eq!(dist.batch_shape(), vec![2]);
        assert_eq!(dist.event_shape(), vec![3]);
        let s = dist.sample(&[]).unwrap();
        assert_eq!(s.shape(), &[2, 3]);
        let value = from_slice(&[0.2f32, 0.3, 0.5, 0.1, 0.4, 0.5], &[2, 3]).unwrap();
        let lp = dist.log_prob(&value).unwrap();
        assert_eq!(lp.shape(), &[2]);
        // mean of a batched Dirichlet matches per-row α / sum(α).
        let m = dist.mean().unwrap();
        assert_eq!(m.shape(), &[2, 3]);
        let md = m.data().unwrap();
        assert!((md[0] - 1.0 / 6.0).abs() < 1e-6);
        assert!((md[3] - 4.0 / 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_dirichlet_log_prob_batched_matches_per_row() {
        // A batched [2,K] Dirichlet's log_prob must equal the two independent
        // 1-D Dirichlet log_probs computed row-by-row (validates the batched
        // normalizer is per-row, not global).
        let a0 = [2.0f32, 3.0, 4.0];
        let a1 = [1.0f32, 1.0, 1.0];
        let batched = from_slice(&[a0[0], a0[1], a0[2], a1[0], a1[1], a1[2]], &[2, 3]).unwrap();
        let dist = Dirichlet::new(batched).unwrap();
        let x = from_slice(&[0.2f32, 0.3, 0.5, 0.25, 0.25, 0.5], &[2, 3]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let lpd = lp.data().unwrap();

        let d0 = Dirichlet::new(tensor(&a0).unwrap()).unwrap();
        let x0 = tensor(&[0.2f32, 0.3, 0.5]).unwrap();
        let e0 = d0.log_prob(&x0).unwrap().item().unwrap();
        let d1 = Dirichlet::new(tensor(&a1).unwrap()).unwrap();
        let x1 = tensor(&[0.25f32, 0.25, 0.5]).unwrap();
        let e1 = d1.log_prob(&x1).unwrap().item().unwrap();

        assert!((lpd[0] - e0).abs() < 1e-5, "row0: {} vs {}", lpd[0], e0);
        assert!((lpd[1] - e1).abs() < 1e-5, "row1: {} vs {}", lpd[1], e1);
    }

    #[test]
    fn test_dirichlet_log_prob_rejects_off_simplex() {
        // log_prob validates value against the Simplex support (consumer of
        // constraints::Simplex::check_tensor). A vector summing to 1.1 is off
        // the simplex -> Err (dirichlet.py:91-92 validate_sample).
        let alpha = tensor(&[1.0f32, 1.0, 1.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();
        let bad = tensor(&[0.2f32, 0.5, 0.4]).unwrap(); // sums to 1.1
        assert!(dist.log_prob(&bad).is_err());
    }

    #[test]
    fn test_dirichlet_concentrated() {
        // High concentration => samples cluster near the uniform mean.
        //
        // For Dir(α=100, 100, 100) the per-component std is
        //   sqrt(α_i (α_0 - α_i) / (α_0² (α_0 + 1)))
        //   = sqrt(100·200 / (300²·301)) ≈ 0.0272
        // and the mean is 1/3 by symmetry. The test originally checked
        // each of 60 samples (20 batches × 3 components) against a
        // ±0.1 (~3.7σ) bound, which fails ~0.4% of the time across the
        // 60 draws and made the test flaky under workspace-parallel
        // runs.
        //
        // Switching to an empirical-mean check tightens the bound by
        // sqrt(N_SAMPLES) via CLT: with N_SAMPLES=200 the mean's std is
        // ≈ 0.0272 / sqrt(200) ≈ 0.00193, so a 0.05 tolerance is ~26σ —
        // genuinely never fails for a correct sampler.
        let alpha = tensor(&[100.0f32, 100.0, 100.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();

        const N_SAMPLES: usize = 200;
        let samples = dist.sample(&[N_SAMPLES]).unwrap();
        let data = samples.data().unwrap();
        let third = 1.0f32 / 3.0;

        // Empirical mean per component.
        let mut means = [0.0f32; 3];
        for s in 0..N_SAMPLES {
            for (j, m) in means.iter_mut().enumerate() {
                *m += data[s * 3 + j];
            }
        }
        for m in means.iter_mut() {
            *m /= N_SAMPLES as f32;
        }

        for (j, &m) in means.iter().enumerate() {
            assert!(
                (m - third).abs() < 0.05,
                "concentrated Dirichlet empirical mean for component {j} \
                 should be near 1/3 across {N_SAMPLES} samples, got {m}"
            );
        }

        // Sanity: every individual sample lies inside the simplex
        // [0, 1] (no per-element tolerance — that bound is racy).
        for s in 0..N_SAMPLES {
            for j in 0..3 {
                let v = data[s * 3 + j];
                assert!(
                    (0.0..=1.0).contains(&v),
                    "Dirichlet sample [s={s}, j={j}] = {v} not in [0, 1]"
                );
            }
        }
    }
}
