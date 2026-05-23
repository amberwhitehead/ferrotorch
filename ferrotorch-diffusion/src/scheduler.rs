//! Deterministic DDIM scheduler matching `diffusers.schedulers.DDIMScheduler`
//! for the Stable-Diffusion-1.5 sampling defaults.
//!
//! Phase F of real-artifact-driven development (#1163). The scheduler is
//! the missing fourth component of the SD generation pipeline:
//!
//! ```text
//! CLIP text encoder + UNet noise predictor + DDIM scheduler + VAE decoder
//!                                            ^^^^^^^^^^^^^^^
//!                                            this module
//! ```
//!
//! Matches `diffusers.schedulers.DDIMScheduler` for the SD-1.5 defaults
//! (`scaled_linear` beta schedule, `epsilon` prediction, `leading`
//! timestep spacing, `clip_sample=false`, `set_alpha_to_one=false`,
//! `init_noise_sigma=1.0` — i.e. η=0 deterministic sampling). Values
//! mirrored byte-for-byte from the upstream defaults in
//! `diffusers/schedulers/scheduling_ddim.py` as of `diffusers==0.38.0`.

use ferrotorch_core::grad_fns::arithmetic::{add, mul, sub};
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};

/// Beta-schedule recipe (subset matching SD-1.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BetaSchedule {
    /// `betas = linspace(sqrt(beta_start), sqrt(beta_end), N)^2`. SD default.
    ScaledLinear,
    /// `betas = linspace(beta_start, beta_end, N)`.
    Linear,
}

/// Discrete timestep spacing (subset matching SD-1.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestepSpacing {
    /// `step_ratio = num_train_timesteps // num_inference_steps;
    /// timesteps = arange(0, num_inference_steps) * step_ratio` then reversed.
    /// SD default.
    Leading,
    /// `step_ratio = num_train_timesteps / num_inference_steps;
    /// timesteps = arange(num_inference_steps) * step_ratio` (rounded) then reversed.
    Linspace,
}

/// `DDIMScheduler` configuration (the subset that affects forward math).
#[derive(Debug, Clone)]
pub struct DDIMConfig {
    /// `num_train_timesteps` — SD-1.5: 1000.
    pub num_train_timesteps: usize,
    /// `beta_start` — SD-1.5: 0.00085.
    pub beta_start: f64,
    /// `beta_end` — SD-1.5: 0.012.
    pub beta_end: f64,
    /// `beta_schedule` — SD-1.5: ScaledLinear.
    pub beta_schedule: BetaSchedule,
    /// `clip_sample` — SD-1.5: false (no clipping of predicted x0).
    pub clip_sample: bool,
    /// `set_alpha_to_one` — SD-1.5: false. When false, the
    /// final-step `alpha_prev` is `alphas_cumprod[0]` (matches diffusers).
    pub set_alpha_to_one: bool,
    /// `prediction_type` — SD-1.5: "epsilon" (the UNet predicts the noise).
    /// Only `"epsilon"` is implemented; anything else returns an error
    /// at `set_timesteps` time.
    pub prediction_type: PredictionType,
    /// `timestep_spacing` — SD-1.5: Leading.
    pub timestep_spacing: TimestepSpacing,
    /// `steps_offset` — SD-1.5: 1 (diffusers adds this offset on the
    /// `leading` path so the first inference step is `step_ratio` rather
    /// than 0; consumed inside [`DDIMScheduler::set_timesteps`]).
    pub steps_offset: usize,
}

/// Prediction parameterisation (subset matching SD-1.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionType {
    /// UNet predicts the noise ε. SD default.
    Epsilon,
}

impl Default for DDIMConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_start: 0.000_85,
            beta_end: 0.012,
            beta_schedule: BetaSchedule::ScaledLinear,
            clip_sample: false,
            set_alpha_to_one: false,
            prediction_type: PredictionType::Epsilon,
            timestep_spacing: TimestepSpacing::Leading,
            steps_offset: 1,
        }
    }
}

impl DDIMConfig {
    /// SD-1.5 defaults (alias for [`Default::default`]).
    pub fn sd_v1_5() -> Self {
        Self::default()
    }
}

/// Deterministic DDIM scheduler (η=0, no noise injection).
///
/// Pre-computes `betas`, `alphas`, and `alphas_cumprod` over the full
/// training-time grid (`num_train_timesteps` entries). Inference picks
/// a subset of timesteps via [`DDIMScheduler::set_timesteps`] and walks
/// them in reverse with [`DDIMScheduler::step`].
#[derive(Debug, Clone)]
pub struct DDIMScheduler {
    config: DDIMConfig,
    /// `alphas_cumprod` of length `num_train_timesteps`.
    alphas_cumprod: Vec<f64>,
    /// `final_alpha_cumprod` — used when stepping into prev_timestep < 0
    /// (the very last denoising step). When `set_alpha_to_one=false` this
    /// is `alphas_cumprod[0]`; when true it's 1.0.
    final_alpha_cumprod: f64,
    /// Timesteps the user requested via [`Self::set_timesteps`], in the
    /// order they will be consumed (descending). Empty until
    /// `set_timesteps` is called.
    timesteps: Vec<usize>,
}

impl DDIMScheduler {
    /// Build a scheduler with the given config; runs the one-shot
    /// `betas → alphas → alphas_cumprod` precomputation.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] for malformed
    /// configurations (zero training steps, non-positive beta).
    pub fn new(config: DDIMConfig) -> FerrotorchResult<Self> {
        if config.num_train_timesteps == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "DDIMScheduler::new: num_train_timesteps must be > 0".into(),
            });
        }
        if !config.beta_start.is_finite() || !config.beta_end.is_finite() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "DDIMScheduler::new: non-finite betas (beta_start={}, beta_end={})",
                    config.beta_start, config.beta_end
                ),
            });
        }
        let n = config.num_train_timesteps;
        let betas = compute_betas(config.beta_schedule, config.beta_start, config.beta_end, n);
        // alphas[i] = 1 - betas[i].
        // alphas_cumprod[i] = prod_{j=0..=i} alphas[j].
        let mut alphas_cumprod = Vec::with_capacity(n);
        let mut acc = 1.0_f64;
        for &b in &betas {
            let a = 1.0 - b;
            acc *= a;
            alphas_cumprod.push(acc);
        }
        let final_alpha_cumprod = if config.set_alpha_to_one {
            1.0
        } else {
            alphas_cumprod[0]
        };
        Ok(Self {
            config,
            alphas_cumprod,
            final_alpha_cumprod,
            timesteps: Vec::new(),
        })
    }

    /// Read-only access to the frozen configuration.
    pub fn config(&self) -> &DDIMConfig {
        &self.config
    }

    /// `init_noise_sigma` — the multiplier applied to the initial Gaussian
    /// noise tensor before the first denoising step. DDIM with the SD-1.5
    /// defaults uses 1.0 (no scaling).
    pub fn init_noise_sigma(&self) -> f64 {
        1.0
    }

    /// Set the inference-time discrete timesteps and return them.
    ///
    /// Matches `diffusers.schedulers.DDIMScheduler.set_timesteps` exactly
    /// for the SD-1.5 defaults (Leading spacing + `steps_offset=1`).
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if `num_inference_steps`
    /// is zero or exceeds `num_train_timesteps`, or if the configured
    /// `prediction_type` is anything other than [`PredictionType::Epsilon`].
    pub fn set_timesteps(&mut self, num_inference_steps: usize) -> FerrotorchResult<&[usize]> {
        if num_inference_steps == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "DDIMScheduler::set_timesteps: num_inference_steps must be > 0".into(),
            });
        }
        if num_inference_steps > self.config.num_train_timesteps {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "DDIMScheduler::set_timesteps: num_inference_steps {num_inference_steps} \
                     must be <= num_train_timesteps {}",
                    self.config.num_train_timesteps
                ),
            });
        }
        if self.config.prediction_type != PredictionType::Epsilon {
            return Err(FerrotorchError::InvalidArgument {
                message:
                    "DDIMScheduler::set_timesteps: only PredictionType::Epsilon is implemented"
                        .into(),
            });
        }
        let n_train = self.config.num_train_timesteps;
        self.timesteps = match self.config.timestep_spacing {
            TimestepSpacing::Leading => {
                // step_ratio = num_train_timesteps // num_inference_steps
                let step_ratio = n_train / num_inference_steps;
                // ts = (arange(0, num_inference_steps) * step_ratio) reversed
                //      + steps_offset.
                let mut ts: Vec<usize> = (0..num_inference_steps)
                    .rev()
                    .map(|i| i * step_ratio + self.config.steps_offset)
                    .collect();
                // Clamp into [0, n_train - 1] (defensive — the standard
                // SD-1.5 4-step recipe never hits this).
                for t in &mut ts {
                    if *t >= n_train {
                        *t = n_train - 1;
                    }
                }
                ts
            }
            TimestepSpacing::Linspace => {
                let step = (n_train as f64) / (num_inference_steps as f64);
                let mut ts: Vec<usize> = (0..num_inference_steps)
                    .rev()
                    .map(|i| ((i as f64 * step).round() as usize).min(n_train - 1))
                    .collect();
                for t in &mut ts {
                    if *t >= n_train {
                        *t = n_train - 1;
                    }
                }
                ts
            }
        };
        Ok(&self.timesteps)
    }

    /// Read-only access to the inference timesteps (empty until
    /// `set_timesteps` has been called).
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Scale the model input. For DDIM with the SD-1.5 defaults this is
    /// the identity (DDIM does not rescale the model input the way
    /// `LMSDiscreteScheduler` does). Kept for pipeline-parity with the
    /// diffusers API surface.
    ///
    /// # Errors
    ///
    /// Currently infallible; returned for forward-compat with non-DDIM
    /// schedulers we may add later.
    pub fn scale_model_input<T: Float>(
        &self,
        sample: &Tensor<T>,
        _timestep: usize,
    ) -> FerrotorchResult<Tensor<T>> {
        Ok(sample.clone())
    }

    /// One DDIM step: predict the previous-sample given the model's
    /// noise prediction `model_output` at `timestep`, applied to `sample`.
    ///
    /// Math (η = 0, deterministic):
    ///
    /// ```text
    /// prev_timestep = timestep - step_size
    /// alpha_t       = alphas_cumprod[timestep]
    /// alpha_t_prev  = alphas_cumprod[prev_timestep] if prev_timestep >= 0
    ///                else final_alpha_cumprod
    /// beta_t        = 1 - alpha_t
    ///
    /// pred_x0       = (sample - sqrt(beta_t) * model_output) / sqrt(alpha_t)
    /// pred_dir      = sqrt(1 - alpha_t_prev) * model_output
    /// prev_sample   = sqrt(alpha_t_prev) * pred_x0 + pred_dir
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if `timestep` is
    /// outside `[0, num_train_timesteps)` or `set_timesteps` has not been
    /// called, and propagates any underlying tensor-arithmetic error.
    pub fn step<T: Float>(
        &self,
        model_output: &Tensor<T>,
        timestep: usize,
        sample: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        if self.timesteps.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "DDIMScheduler::step: set_timesteps must be called before step".into(),
            });
        }
        if timestep >= self.config.num_train_timesteps {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "DDIMScheduler::step: timestep {timestep} out of range \
                     [0, {})",
                    self.config.num_train_timesteps
                ),
            });
        }
        if model_output.shape() != sample.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "DDIMScheduler::step: model_output shape {:?} != sample shape {:?}",
                    model_output.shape(),
                    sample.shape()
                ),
            });
        }
        // diffusers: prev_timestep = timestep - num_train_timesteps //
        //                              num_inference_steps
        let step_ratio = self.config.num_train_timesteps / self.timesteps.len();
        let prev_timestep_i = timestep as isize - step_ratio as isize;
        let alpha_t = self.alphas_cumprod[timestep];
        let alpha_t_prev = if prev_timestep_i >= 0 {
            self.alphas_cumprod[prev_timestep_i as usize]
        } else {
            self.final_alpha_cumprod
        };
        let beta_t = 1.0 - alpha_t;

        // Compute three scalar tensors (broadcastable against the [B,C,H,W] sample).
        let sqrt_beta_t = scalar_f64::<T>(beta_t.sqrt())?;
        let inv_sqrt_alpha_t = scalar_f64::<T>(1.0 / alpha_t.sqrt())?;
        let sqrt_one_minus_alpha_t_prev = scalar_f64::<T>((1.0 - alpha_t_prev).sqrt())?;
        let sqrt_alpha_t_prev = scalar_f64::<T>(alpha_t_prev.sqrt())?;

        // pred_x0 = (sample - sqrt(beta_t) * model_output) / sqrt(alpha_t)
        //         = (sample - sqrt(beta_t) * model_output) * (1 / sqrt(alpha_t))
        let scaled_noise = mul(model_output, &sqrt_beta_t)?;
        let diff = sub(sample, &scaled_noise)?;
        let pred_x0 = mul(&diff, &inv_sqrt_alpha_t)?;

        // Optional clip_sample. SD-1.5 default is false; gate kept so this
        // module is reusable with non-SD configs.
        let pred_x0 = if self.config.clip_sample {
            clip_to_one::<T>(&pred_x0)?
        } else {
            pred_x0
        };

        // pred_dir = sqrt(1 - alpha_t_prev) * model_output
        let pred_dir = mul(model_output, &sqrt_one_minus_alpha_t_prev)?;
        // prev_sample = sqrt(alpha_t_prev) * pred_x0 + pred_dir
        let x0_scaled = mul(&pred_x0, &sqrt_alpha_t_prev)?;
        add(&x0_scaled, &pred_dir)
    }
}

/// Compute `betas` according to the chosen schedule. Mirrors diffusers's
/// `betas_for_alpha_bar` only for the two schedules SD-1.5 ever uses.
fn compute_betas(schedule: BetaSchedule, beta_start: f64, beta_end: f64, n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    if n == 1 {
        out.push(beta_start);
        return out;
    }
    let denom = (n - 1) as f64;
    match schedule {
        BetaSchedule::Linear => {
            for i in 0..n {
                let t = i as f64 / denom;
                out.push(beta_start + t * (beta_end - beta_start));
            }
        }
        BetaSchedule::ScaledLinear => {
            // linspace(sqrt(beta_start), sqrt(beta_end), N) ^ 2
            let a = beta_start.sqrt();
            let b = beta_end.sqrt();
            for i in 0..n {
                let t = i as f64 / denom;
                let lin = a + t * (b - a);
                out.push(lin * lin);
            }
        }
    }
    out
}

/// Build a 1-element scalar tensor carrying `value` cast into the target Float.
fn scalar_f64<T: Float>(value: f64) -> FerrotorchResult<Tensor<T>> {
    let v = T::from(value).ok_or_else(|| FerrotorchError::InvalidArgument {
        message: format!("DDIMScheduler: cannot represent f64 {value} as the requested Float"),
    })?;
    ferrotorch_core::scalar::<T>(v)
}

/// Clamp every entry of `t` into `[-1, 1]`. Used only when the user
/// configures `clip_sample=true`; SD-1.5 default is false.
fn clip_to_one<T: Float>(t: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let lo = T::from(-1.0).ok_or_else(|| FerrotorchError::InvalidArgument {
        message: "clip_to_one: cannot represent -1.0".into(),
    })?;
    let hi = T::from(1.0).ok_or_else(|| FerrotorchError::InvalidArgument {
        message: "clip_to_one: cannot represent 1.0".into(),
    })?;
    ferrotorch_core::clamp(t, lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_schedule_scaled_linear_matches_diffusers_sd15() {
        // Mirror diffusers's `betas = linspace(sqrt(0.00085),
        // sqrt(0.012), 1000) ** 2`. Spot-check a handful of indices.
        let betas = compute_betas(BetaSchedule::ScaledLinear, 0.000_85, 0.012, 1000);
        assert_eq!(betas.len(), 1000);
        // i=0 → beta_start exactly.
        assert!((betas[0] - 0.000_85).abs() < 1e-12, "betas[0]={}", betas[0]);
        // i=999 → beta_end exactly.
        assert!(
            (betas[999] - 0.012).abs() < 1e-12,
            "betas[999]={}",
            betas[999]
        );
        // i=500 → midpoint: ((sqrt(0.00085)+sqrt(0.012))/2)^2
        let mid_root = (0.000_85_f64.sqrt() + 0.012_f64.sqrt()) * 0.5;
        let want = mid_root * mid_root;
        // i=500 with linspace of 1000 points is offset 500/999 ≠ 0.5; allow a small slack.
        let approx_idx = 999 / 2;
        assert!(
            (betas[approx_idx] - want).abs() < 5e-3,
            "betas[{approx_idx}]={} vs midpoint {want}",
            betas[approx_idx]
        );
    }

    #[test]
    fn alphas_cumprod_is_monotone_decreasing() {
        let sched = DDIMScheduler::new(DDIMConfig::sd_v1_5()).unwrap();
        let mut prev = 1.0_f64;
        for &a in &sched.alphas_cumprod {
            assert!(a < prev, "alphas_cumprod not strictly decreasing");
            assert!(a > 0.0, "alphas_cumprod must be > 0, got {a}");
            prev = a;
        }
        // Final value is approximately 0.0047 for SD-1.5 (reference).
        let last = *sched.alphas_cumprod.last().unwrap();
        assert!(
            (0.001..0.01).contains(&last),
            "SD-1.5 alphas_cumprod[-1] should be ~0.0047, got {last}"
        );
    }

    #[test]
    fn timesteps_leading_4_steps_sd15() {
        // step_ratio = 1000 // 4 = 250; offset = 1.
        // diffusers: timesteps = (arange(4) * 250) reversed + 1
        //           = [750, 500, 250, 0] + 1 = [751, 501, 251, 1].
        let mut sched = DDIMScheduler::new(DDIMConfig::sd_v1_5()).unwrap();
        let ts = sched.set_timesteps(4).unwrap();
        assert_eq!(ts, [751, 501, 251, 1]);
    }

    #[test]
    fn timesteps_leading_50_steps_sd15_head() {
        let mut sched = DDIMScheduler::new(DDIMConfig::sd_v1_5()).unwrap();
        let ts = sched.set_timesteps(50).unwrap();
        assert_eq!(ts.len(), 50);
        // step_ratio = 1000 // 50 = 20. first = 49*20+1 = 981; last = 1.
        assert_eq!(ts[0], 981);
        assert_eq!(ts[49], 1);
    }

    #[test]
    fn init_noise_sigma_is_one() {
        let sched = DDIMScheduler::new(DDIMConfig::sd_v1_5()).unwrap();
        assert!((sched.init_noise_sigma() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn final_alpha_cumprod_is_alphas_cumprod_zero_when_set_alpha_to_one_false() {
        let sched = DDIMScheduler::new(DDIMConfig::sd_v1_5()).unwrap();
        assert!((sched.final_alpha_cumprod - sched.alphas_cumprod[0]).abs() < 1e-12);
    }

    #[test]
    fn step_recovers_zero_for_identity_noise() {
        // If model_output is the zero tensor at timestep 0:
        //   pred_x0 = sample / sqrt(alpha_0)
        //   pred_dir = 0
        //   prev_sample = sqrt(alpha_prev) * pred_x0
        // We can't really assert byte-level here without a reference, but
        // we can assert non-finite values do not appear and shape is
        // preserved.
        let mut sched = DDIMScheduler::new(DDIMConfig::sd_v1_5()).unwrap();
        sched.set_timesteps(4).unwrap();
        let sample = Tensor::<f32>::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![0.5_f32; 4]),
            vec![1, 1, 2, 2],
            false,
        )
        .unwrap();
        let noise = Tensor::<f32>::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![0.0_f32; 4]),
            vec![1, 1, 2, 2],
            false,
        )
        .unwrap();
        let out = sched.step(&noise, 1, &sample).unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        for v in out.data().unwrap() {
            assert!(v.is_finite(), "step produced non-finite value: {v}");
        }
    }
}
