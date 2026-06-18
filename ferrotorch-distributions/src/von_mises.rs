//! Von Mises distribution (circular normal).
//!
//! `VonMises(loc, concentration)` — distribution on the circle [-pi, pi].
//!
//! ## REQ status (per `.design/ferrotorch-distributions/von_mises.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`VonMises` struct) | SHIPPED | `pub struct VonMises` in `von_mises.rs`; re-exported as `pub use von_mises::VonMises` in `lib.rs:123`; mirrors `torch/distributions/von_mises.py:133-142`. |
//! | REQ-2 (`new` constructor, shape match) | SHIPPED | `VonMises::new` rejecting shape mismatch; registered in `tests/conformance/_surface_inventory.toml:497`. |
//! | REQ-3 (`loc` + `concentration` accessors) | SHIPPED | accessors in `von_mises.rs`. |
//! | REQ-4 (`log_bessel_i0` private helper) | SHIPPED | private `log_bessel_i0` two-branch Abramowitz-Stegun approximation in `von_mises.rs`; coefficients match upstream `_I0_COEF_SMALL`/`_I0_COEF_LARGE` at `von_mises.py:23-42`; 2 production call sites in this module. |
//! | REQ-5 (`Distribution::sample` via Best's rejection) | SHIPPED | `impl Distribution::sample` in `von_mises.rs` via Best & Fisher 1979 rejection algorithm; mirrors `_rejection_sample` at `von_mises.py:92-107`. Known divergences in REQ-11/REQ-12. |
//! | REQ-6 (`Distribution::log_prob`) | SHIPPED | `impl Distribution::log_prob` returns `kappa*cos(value-loc) - log(2*pi) - log_bessel_i0(kappa)`; mirrors `von_mises.py:144-153` exactly. |
//! | REQ-7 (`Distribution::entropy` approximation) | SHIPPED | `impl Distribution::entropy` uses I_1/I_0 ratio approximation; R-DEV-7 enhancement (upstream lacks closed-form entropy). |
//! | REQ-8 (`Distribution::mean`) | SHIPPED | `impl Distribution::mean` returns `loc.clone()`; mirrors `von_mises.py:199-204` (circular mean). |
//! | REQ-9 (`Distribution::rsample` errors) | SHIPPED | `impl Distribution::rsample` returns `InvalidArgument`; mirrors upstream's `has_rsample = False` at `von_mises.py:131`. |
//! | REQ-10 (`mode`/`variance`/`expand`/`support`/`_log_modified_bessel_fn(order=1)`) | SHIPPED | `fn mode` returns `loc.clone()` mirroring `torch/distributions/von_mises.py:206-208`; `fn variance` invokes `log_bessel_i1` - `log_bessel_i0` ratio mirroring `von_mises.py:210-221`; `fn log_bessel_i1` mirrors `_log_modified_bessel_fn(order=1)` at `von_mises.py:43-89` using `_I1_COEF_SMALL`/`_I1_COEF_LARGE`; `fn support` returns `Real`; `fn expand` broadcasts both parameter tensors mirroring `von_mises.py:190-197`. Consumer: trait dispatch via `pub use VonMises` re-export. Closes #1431. |
//! | REQ-11 (RNG: `creation::rand` instead of xorshift) | SHIPPED | the `sample` body in `von_mises.rs` now draws all uniforms via a bulk `ferrotorch_core::creation::rand` allocation up-front, indexing into the resulting vec for each rejection step. The xorshift hash on `SystemTime + ThreadId` is gone; non-test consumer: `pub use von_mises::VonMises` re-export — any caller using `dist.sample(...)` now routes through the workspace-RNG which honours `ferrotorch_core::manual_seed`. Closes #1432. |
//! | REQ-12 (small-kappa Taylor fallback) | SHIPPED | the `sample` body branches to `_proposal_r_taylor = 1.0 / kappa + kappa` for `kappa < 1e-5` per `torch/distributions/von_mises.py:170-171` (`_rejection_sample_with_taylor_fallback`); avoids the Best's rejection loop hanging when the proposal radius collapses. Non-test consumer: same as REQ-11. Closes #1433. |
//! | REQ-13 (entropy override uses exact log_bessel I_1/I_0 ratio + Stirling tails) | SHIPPED | `fn entropy` in `von_mises.rs` evaluates `log(2π) + log_bessel_i0(kappa) - kappa * exp(log_bessel_i1(kappa) - log_bessel_i0(kappa))` using the upstream-Bessel polynomial coefficient table from `log_bessel_i1`/`log_bessel_i0` (already SHIPPED for variance under REQ-10). Replaces the prior 1-term `1 - 1/(2*kappa)` asymptote with the exact ratio. Closes #1434. |

use std::collections::HashMap;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::constraints;
use crate::{DistConstraint, Distribution};

/// Von Mises distribution parameterized by `loc` (mean direction) and
/// `concentration` (kappa, analogous to inverse variance).
///
/// PDF: `f(x) = exp(kappa * cos(x - loc)) / (2 * pi * I_0(kappa))`
/// where `I_0` is the modified Bessel function of the first kind, order 0.
///
/// Values are on [-pi, pi].
pub struct VonMises<T: Float> {
    loc: Tensor<T>,
    concentration: Tensor<T>,
}

impl<T: Float> VonMises<T> {
    pub fn new(loc: Tensor<T>, concentration: Tensor<T>) -> FerrotorchResult<Self> {
        if loc.shape() != concentration.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "VonMises: loc shape {:?} != concentration shape {:?}",
                    loc.shape(),
                    concentration.shape()
                ),
            });
        }
        Ok(Self { loc, concentration })
    }

    pub fn loc(&self) -> &Tensor<T> {
        &self.loc
    }
    pub fn concentration(&self) -> &Tensor<T> {
        &self.concentration
    }
}

/// Approximate log of modified Bessel function I_0(x).
/// Mirrors `_log_modified_bessel_fn(x, order=0)` at
/// `torch/distributions/von_mises.py:68-89` using the
/// Abramowitz-Stegun polynomial coefficients (`_I0_COEF_SMALL` at
/// `von_mises.py:23-31` + `_I0_COEF_LARGE` at `von_mises.py:32-42`).
fn log_bessel_i0<T: Float>(x: T) -> T {
    let xf = num_traits::ToPrimitive::to_f64(&x).unwrap();
    let result = if xf < 3.75 {
        // Small argument: I_0(x) ≈ polynomial
        let t = (xf / 3.75).powi(2);
        let i0 = 1.0
            + t * (3.5156229
                + t * (3.0899424
                    + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
        i0.ln()
    } else {
        // Large argument: asymptotic expansion
        let t = 3.75 / xf;
        let factor = 0.39894228
            + t * (0.01328592
                + t * (0.00225319
                    + t * (-0.00157565
                        + t * (0.00916281
                            + t * (-0.02057706
                                + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))));
        xf - 0.5 * xf.ln() + factor.ln()
    };
    T::from(result).unwrap()
}

/// Approximate log of modified Bessel function I_1(x), for x > 0.
/// Mirrors `_log_modified_bessel_fn(x, order=1)` at
/// `torch/distributions/von_mises.py:68-89` using the Abramowitz-Stegun
/// coefficients (`_I1_COEF_SMALL` at `von_mises.py:43-51` and
/// `_I1_COEF_LARGE` at `von_mises.py:52-62`).
fn log_bessel_i1<T: Float>(x: T) -> T {
    let xf = num_traits::ToPrimitive::to_f64(&x).unwrap();
    let result = if xf < 3.75 {
        // Small argument: small = |x| * poly(t); log(small) = log|x| + log(poly)
        let t = (xf / 3.75).powi(2);
        let poly = 0.5
            + t * (0.87890594
                + t * (0.51498869
                    + t * (0.15084934 + t * (0.02658733 + t * (0.00301532 + t * 0.00032411)))));
        xf.abs().ln() + poly.ln()
    } else {
        // Large argument: x - 0.5 * log(x) + log(poly(3.75/x))
        let t = 3.75 / xf;
        let factor = 0.39894228
            + t * (-0.03988024
                + t * (-0.00362018
                    + t * (0.00163801
                        + t * (-0.01031555
                            + t * (0.02282967
                                + t * (-0.02895312 + t * (0.01787654 + t * (-0.00420059))))))));
        xf - 0.5 * xf.ln() + factor.ln()
    };
    T::from(result).unwrap()
}

impl<T: Float> Distribution<T> for VonMises<T> {
    #[allow(clippy::needless_range_loop)]
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.loc, &self.concentration],
            "VonMises::sample",
        )?;
        // Best's algorithm for Von Mises sampling — closes #1432 (uses
        // workspace `creation::rand` instead of a per-call xorshift seeded
        // from `SystemTime + ThreadId`) and #1433 (small-kappa Taylor
        // fallback so the rejection loop never collapses).
        let l_data = self.loc.data()?;
        let k_data = self.concentration.data()?;
        let numel: usize = shape.iter().product();

        let pi = T::from(std::f64::consts::PI).unwrap();
        let two = T::from(2.0).unwrap();
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();
        let half = T::from(0.5).unwrap();
        // Per upstream `torch/distributions/von_mises.py:170-171`, when
        // `concentration < 1e-5` the Bessel ratio `(rho - sqrt(2*rho))/...`
        // becomes numerically unstable; switch to the Taylor expansion
        // `_proposal_r_taylor(kappa) = 1/kappa + kappa` which keeps `r`
        // bounded away from `1`. The threshold matches PyTorch.
        let small_kappa_threshold = T::from(1e-5).unwrap();

        // Bulk-draw uniforms via the workspace RNG (`creation::rand` honours
        // `ferrotorch_core::manual_seed`). The rejection loop is bounded in
        // expectation but unbounded in the worst case, so we pre-allocate a
        // generous buffer and top up if we exhaust it. Each accepted draw
        // consumes 3 uniforms on average (u1 for z, u2 for the accept test,
        // u3 for the sign), and Best's algorithm accepts with probability
        // > 0.5 for all kappa > 0.
        let mut uniforms = creation::rand::<T>(&[numel * 8])?.data_vec()?;
        let mut u_idx = 0usize;
        // Closure-free helper: refresh the buffer when we run out.
        let refresh_if_needed = |idx: &mut usize, buf: &mut Vec<T>| -> FerrotorchResult<()> {
            if *idx >= buf.len() {
                let more = creation::rand::<T>(&[buf.len()])?.data_vec()?;
                buf.extend(more);
            }
            Ok(())
        };

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let li = if l_data.len() == 1 {
                0
            } else {
                i % l_data.len()
            };
            let ki = if k_data.len() == 1 {
                0
            } else {
                i % k_data.len()
            };
            let kappa = k_data[ki];

            // _proposal_r mirroring `von_mises.py:163-171`: stable Bessel
            // ratio for kappa >= 1e-5, Taylor expansion otherwise.
            let r = if kappa < small_kappa_threshold {
                // Taylor: r ≈ 1/kappa + kappa  (PyTorch's _proposal_r_taylor)
                one / kappa + kappa
            } else {
                let tau = one + (one + T::from(4.0).unwrap() * kappa * kappa).sqrt();
                let rho = (tau - (two * tau).sqrt()) / (two * kappa);
                (one + rho * rho) / (two * rho)
            };

            let sample = loop {
                refresh_if_needed(&mut u_idx, &mut uniforms)?;
                let u1 = uniforms[u_idx];
                u_idx += 1;
                let z = (pi * u1).cos();
                let w = (one + r * z) / (r + z);

                refresh_if_needed(&mut u_idx, &mut uniforms)?;
                let u2 = uniforms[u_idx];
                u_idx += 1;
                let c = kappa * (r - w);

                if c * (two - c) > u2 || c.ln() >= u2.ln() + one - c {
                    refresh_if_needed(&mut u_idx, &mut uniforms)?;
                    let u3 = uniforms[u_idx];
                    u_idx += 1;
                    let sign = if u3 > half { one } else { zero - one };
                    break sign * w.acos() + l_data[li];
                }
            };

            // Wrap to [-pi, pi]
            let wrapped = ((sample + pi) % (two * pi) + two * pi) % (two * pi) - pi;
            out.push(wrapped);
        }

        Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        // Reference: torch.distributions.VonMises.mean — returns self.loc (the mean direction).
        // The mean of a VonMises distribution is loc (modulo 2π, but torch returns loc directly).
        Ok(self.loc.clone())
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/von_mises.py:206-208`: `mode = self.loc`.
        // The density is peaked at the mean direction loc.
        Ok(self.loc.clone())
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        // `torch/distributions/von_mises.py:210-221`:
        //   variance = 1 - exp(log I_1(kappa) - log I_0(kappa))
        // which is the circular variance.
        crate::fallback::check_gpu_fallback_opt_in(&[&self.concentration], "VonMises::variance")?;
        let k = self.concentration.data()?;
        let one = <T as num_traits::One>::one();
        let mut out = Vec::with_capacity(k.len());
        for &ki in k.iter() {
            let log_ratio = log_bessel_i1(ki) - log_bessel_i0(ki);
            out.push(one - log_ratio.exp());
        }
        Tensor::from_storage(
            TensorStorage::cpu(out),
            self.concentration.shape().to_vec(),
            false,
        )
    }

    // -----------------------------------------------------------------------
    // Full PyTorch surface — `von_mises.py:128-131` declares
    //   arg_constraints = {"loc": real, "concentration": positive}
    //   support = constraints.real
    //   has_rsample = False
    // -----------------------------------------------------------------------

    fn has_rsample(&self) -> bool {
        // `torch/distributions/von_mises.py:131`: `has_rsample = False`.
        false
    }

    fn batch_shape(&self) -> Vec<usize> {
        self.loc.shape().to_vec()
    }

    fn support(&self) -> Option<Box<dyn DistConstraint>> {
        // `torch/distributions/von_mises.py:130`: `support = constraints.real`.
        Some(Box::new(constraints::Real))
    }

    fn arg_constraints(&self) -> HashMap<&'static str, Box<dyn DistConstraint>> {
        // `torch/distributions/von_mises.py:129`:
        //   arg_constraints = {"loc": real, "concentration": positive}
        let mut m: HashMap<&'static str, Box<dyn DistConstraint>> = HashMap::new();
        m.insert("loc", Box::new(constraints::Real));
        m.insert("concentration", Box::new(constraints::Positive));
        m
    }

    fn expand(&self, batch_shape: &[usize]) -> FerrotorchResult<Box<dyn Distribution<T>>> {
        // Mirrors `von_mises.py:190-197`.
        let l_data = self.loc.data_vec()?;
        let k_data = self.concentration.data_vec()?;
        let n: usize = batch_shape.iter().product::<usize>().max(1);
        let l_out: Vec<T> = (0..n).map(|i| l_data[i % l_data.len()]).collect();
        let k_out: Vec<T> = (0..n).map(|i| k_data[i % k_data.len()]).collect();
        let new_loc = Tensor::from_storage(TensorStorage::cpu(l_out), batch_shape.to_vec(), false)?;
        let new_conc =
            Tensor::from_storage(TensorStorage::cpu(k_out), batch_shape.to_vec(), false)?;
        Ok(Box::new(VonMises::new(new_loc, new_conc)?))
    }

    fn rsample(&self, _shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "VonMises: rsample not supported (discrete rejection sampling)".into(),
        })
    }

    #[allow(clippy::needless_range_loop)]
    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.loc, &self.concentration, value],
            "VonMises::log_prob",
        )?;
        let v = value.data()?;
        let l = self.loc.data()?;
        let k = self.concentration.data()?;
        let numel = v.len();
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let li = if l.len() == 1 { 0 } else { i % l.len() };
            let ki = if k.len() == 1 { 0 } else { i % k.len() };
            // log_prob = kappa * cos(x - loc) - log(2*pi*I_0(kappa))
            let lp = k[ki] * (v[i] - l[li]).cos() - two_pi.ln() - log_bessel_i0(k[ki]);
            out.push(lp);
        }

        Tensor::from_storage(TensorStorage::cpu(out), value.shape().to_vec(), false)
    }

    #[allow(clippy::needless_range_loop)]
    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.concentration], "VonMises::entropy")?;
        // H = log(2*pi*I_0(kappa)) - kappa * I_1(kappa)/I_0(kappa)
        // Closes #1434: compute the I_1/I_0 ratio via the upstream Bessel
        // coefficient tables (`log_bessel_i1` − `log_bessel_i0` shipped under
        // REQ-10) instead of the prior 1-term asymptote. The Bessel
        // approximations are the Abramowitz-Stegun polynomial coefficients
        // (`von_mises.py:23-62`), which are accurate to ~1e-7 across the
        // small/large argument split at 3.75.
        let k = self.concentration.data()?;
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let zero = <T as num_traits::Zero>::zero();

        let mut out = Vec::with_capacity(k.len());
        for i in 0..k.len() {
            let ratio = if k[i] > zero {
                // Exact I_1(k)/I_0(k) via log-bessel difference.
                (log_bessel_i1(k[i]) - log_bessel_i0(k[i])).exp()
            } else {
                // I_1(0)/I_0(0) = 0/1 = 0.
                zero
            };
            let h = two_pi.ln() + log_bessel_i0(k[i]) - k[i] * ratio;
            out.push(h);
        }

        Tensor::from_storage(
            TensorStorage::cpu(out),
            self.concentration.shape().to_vec(),
            false,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(v: f64) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vec![v]), vec![1], false).unwrap()
    }

    #[test]
    fn test_von_mises_sample_range() {
        let d = VonMises::new(scalar(0.0), scalar(2.0)).unwrap();
        let s = d.sample(&[500]).unwrap();
        let pi = std::f64::consts::PI;
        for &v in s.data().unwrap() {
            assert!(
                v >= -pi && v <= pi,
                "VonMises sample should be in [-pi,pi], got {v}"
            );
        }
    }

    #[test]
    fn test_von_mises_log_prob_at_mode() {
        let d = VonMises::new(scalar(0.0), scalar(5.0)).unwrap();
        let at_mode = Tensor::from_storage(TensorStorage::cpu(vec![0.0]), vec![1], false).unwrap();
        let away = Tensor::from_storage(
            TensorStorage::cpu(vec![std::f64::consts::PI]),
            vec![1],
            false,
        )
        .unwrap();
        let lp_mode = d.log_prob(&at_mode).unwrap().data().unwrap()[0];
        let lp_away = d.log_prob(&away).unwrap().data().unwrap()[0];
        assert!(lp_mode > lp_away, "log_prob should be highest at mode");
    }

    #[test]
    fn test_von_mises_small_kappa_terminates() {
        // Pre-fix: kappa < 1e-5 made Best's loop hang because rho → 0.
        // Post-fix (REQ-12): Taylor fallback gives finite r, so sample
        // terminates promptly. The actual value just needs to be in [-pi, pi].
        let d = VonMises::new(scalar(0.0), scalar(1e-8)).unwrap();
        let s = d.sample(&[20]).unwrap();
        let pi = std::f64::consts::PI;
        for &v in s.data().unwrap() {
            assert!(
                v >= -pi && v <= pi,
                "small-kappa VonMises sample out of [-pi, pi]: {v}"
            );
        }
    }

    #[test]
    fn test_von_mises_seed_reproducibility() {
        // Closes #1432: with a fixed manual_seed, two independent samples
        // should match byte-for-byte because we now route through
        // creation::rand (which honours the workspace RNG).
        ferrotorch_core::manual_seed(0xC0FFEE).unwrap();
        let d1 = VonMises::new(scalar(0.0), scalar(2.0)).unwrap();
        let s1 = d1.sample(&[50]).unwrap();
        let v1 = s1.data().unwrap().to_vec();

        ferrotorch_core::manual_seed(0xC0FFEE).unwrap();
        let d2 = VonMises::new(scalar(0.0), scalar(2.0)).unwrap();
        let s2 = d2.sample(&[50]).unwrap();
        let v2 = s2.data().unwrap().to_vec();

        assert_eq!(
            v1, v2,
            "seed reproducibility broken — VonMises::sample should be deterministic under manual_seed"
        );
    }

    #[test]
    fn test_von_mises_entropy_positive() {
        let d = VonMises::new(scalar(0.0), scalar(1.0)).unwrap();
        let h = d.entropy().unwrap();
        assert!(h.data().unwrap()[0] > 0.0, "entropy should be positive");
    }
}
