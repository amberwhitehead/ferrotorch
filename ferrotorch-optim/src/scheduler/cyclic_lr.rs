//! Cyclic learning rate scheduler.
//!
//! Cycles the learning rate between `base_lr` and `max_lr` with a triangular
//! wave. Three built-in policies are provided: `triangular`, `triangular2`,
//! and `exp_range`.
//!
//! Reference: "Cyclical Learning Rates for Training Neural Networks"
//! (Smith, 2017).
//!
//! [CL-320]
//!
//! ## REQ status (per `.design/ferrotorch-optim/scheduler/cyclic_lr.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub enum CyclicMode { Triangular, Triangular2, ExpRange }` in `scheduler/cyclic_lr.rs` mirrors `torch/optim/lr_scheduler.py:1893`; consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`. |
//! | REQ-2 | SHIPPED | `pub struct CyclicLR` in `scheduler/cyclic_lr.rs` mirrors `torch/optim/lr_scheduler.py:1886-1965`; consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`; user code boxes for `Learner::with_scheduler` at `ferrotorch-train/src/learner.rs:105`. |
//! | REQ-3 | SHIPPED | `pub fn CyclicLR::new(... step_size_down: Option<usize>, mode, gamma)` in `scheduler/cyclic_lr.rs` mirrors `torch/optim/lr_scheduler.py:1886-1965`; consumer: re-exported via `lib.rs:47-52`. |
//! | REQ-4 | SHIPPED | `impl<T: Float> LrScheduler<T> for CyclicLR` triangular-wave + amplitude scaling in `scheduler/cyclic_lr.rs` mirrors `torch/optim/lr_scheduler.py:1999-2098`; consumer: `Learner` per-epoch `sched.step` at `ferrotorch-train/src/learner.rs:306-308`. |
//! | REQ-5 | SHIPPED | `cycle_momentum` + `base_momentum` + `max_momentum` fields on `CyclicLR` + `with_cycle_momentum` builder + `compute_momentum` per-step writeback via `optimizer.set_momentum` mirror `torch/optim/lr_scheduler.py:1840-1862, 1935-1963`; consumer: `LrScheduler::step` invokes both `set_lr` and `set_momentum` so SGD-family optimizers see the cycled momentum coefficient. |
//! | REQ-6 | SHIPPED | `with_scale_fn(Box<dyn Fn(f64) -> f64 + Send + Sync>)` builder on `CyclicLR` mirrors `torch/optim/lr_scheduler.py:1830-1834` `_scale_fn_custom`; consumer: `compute_lr` dispatches to the user closure when present, replacing the built-in Triangular/Triangular2/ExpRange amplitude formula. |

use std::sync::Arc;

use ferrotorch_core::Float;

use super::LrScheduler;
use crate::optimizer::Optimizer;

/// Policy for the cyclic learning rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CyclicMode {
    /// Basic triangular cycle without amplitude scaling.
    Triangular,
    /// Triangular cycle that halves the amplitude each cycle.
    Triangular2,
    /// Cycle that scales amplitude by `gamma^(iteration)` each iteration.
    ExpRange,
}

/// User-provided amplitude scaling closure for [`CyclicLR`].
///
/// Mirrors PyTorch's `scale_fn` parameter at
/// `torch/optim/lr_scheduler.py:1830-1834`. The closure receives a single
/// `f64` (the cycle position when `scale_mode == "cycle"`, or the absolute
/// iteration count when `scale_mode == "iterations"`; ferrotorch follows
/// upstream's `Triangular*` default and passes the cycle number) and
/// returns an amplitude factor in `[0, 1]`.
pub type ScaleFn = Arc<dyn Fn(f64) -> f64 + Send + Sync>;

/// Cyclic learning rate scheduler.
///
/// Cycles the learning rate between `base_lr` and `max_lr` using a
/// triangular wave pattern. The amplitude can be scaled per-cycle or
/// per-iteration depending on the mode.
///
/// # Policies
///
/// - **Triangular**: constant amplitude `(max_lr - base_lr)`.
/// - **Triangular2**: amplitude is halved each complete cycle.
/// - **ExpRange**: amplitude scaled by `gamma^iteration`.
///
/// # Momentum cycling
///
/// When [`Self::with_cycle_momentum`] is configured, the scheduler also
/// drives the optimizer's momentum coefficient inversely with the learning
/// rate — i.e. momentum is at its maximum (`max_momentum`) when LR is at
/// `base_lr`, and at its minimum (`base_momentum`) when LR is at `max_lr`.
/// Requires the underlying optimizer to override
/// [`Optimizer::set_momentum`]; today only SGD-family optimizers do.
///
/// # Example
///
/// ```ignore
/// let scheduler = CyclicLR::new(0.001, 0.01, 2000, None, CyclicMode::Triangular, 1.0);
/// ```
#[derive(Clone)]
pub struct CyclicLR {
    /// Lower boundary learning rate.
    base_lr: f64,
    /// Upper boundary learning rate.
    max_lr: f64,
    /// Total cycle size (step_size_up + step_size_down).
    total_size: f64,
    /// Ratio of the up-phase to the full cycle.
    step_ratio: f64,
    /// Scaling mode.
    mode: CyclicMode,
    /// Gamma for exp_range mode.
    gamma: f64,
    /// Current step count.
    current_step: usize,
    /// Current computed learning rate.
    current_lr: f64,
    /// Optional user-provided amplitude scaling closure. When `Some`,
    /// overrides the built-in `Triangular`/`Triangular2`/`ExpRange`
    /// formula. Mirrors `torch/optim/lr_scheduler.py:1830-1834`.
    scale_fn: Option<ScaleFn>,
    /// When `true`, also cycle momentum inversely with the learning rate.
    /// Mirrors `cycle_momentum=True` at `torch/optim/lr_scheduler.py:1840`.
    cycle_momentum: bool,
    /// Lower momentum boundary (used when `cycle_momentum == true`).
    /// Mirrors `base_momentum` at `torch/optim/lr_scheduler.py:1843`.
    base_momentum: f64,
    /// Upper momentum boundary (used when `cycle_momentum == true`).
    /// Mirrors `max_momentum` at `torch/optim/lr_scheduler.py:1848`.
    max_momentum: f64,
    /// Current computed momentum (cached for introspection).
    current_momentum: f64,
}

impl std::fmt::Debug for CyclicLR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CyclicLR")
            .field("base_lr", &self.base_lr)
            .field("max_lr", &self.max_lr)
            .field("total_size", &self.total_size)
            .field("step_ratio", &self.step_ratio)
            .field("mode", &self.mode)
            .field("gamma", &self.gamma)
            .field("current_step", &self.current_step)
            .field("current_lr", &self.current_lr)
            .field("scale_fn", &self.scale_fn.as_ref().map(|_| "<closure>"))
            .field("cycle_momentum", &self.cycle_momentum)
            .field("base_momentum", &self.base_momentum)
            .field("max_momentum", &self.max_momentum)
            .field("current_momentum", &self.current_momentum)
            .finish()
    }
}

impl CyclicLR {
    /// Create a new `CyclicLR` scheduler.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Lower learning rate boundary.
    /// * `max_lr` - Upper learning rate boundary.
    /// * `step_size_up` - Number of iterations in the increasing half of a cycle.
    /// * `step_size_down` - Number of iterations in the decreasing half.
    ///   If `None`, defaults to `step_size_up`.
    /// * `mode` - One of `Triangular`, `Triangular2`, or `ExpRange`.
    /// * `gamma` - Constant for `ExpRange` mode: `gamma^(iteration)`.
    pub fn new(
        base_lr: f64,
        max_lr: f64,
        step_size_up: usize,
        step_size_down: Option<usize>,
        mode: CyclicMode,
        gamma: f64,
    ) -> Self {
        let step_size_down = step_size_down.unwrap_or(step_size_up);
        let total_size = (step_size_up + step_size_down) as f64;
        let step_ratio = step_size_up as f64 / total_size;

        Self {
            base_lr,
            max_lr,
            total_size,
            step_ratio,
            mode,
            gamma,
            current_step: 0,
            current_lr: base_lr,
            scale_fn: None,
            // Match upstream PyTorch defaults at
            // `torch/optim/lr_scheduler.py:1897-1899`: `cycle_momentum=True`
            // with `base_momentum=0.8, max_momentum=0.9`. Even though we
            // default `cycle_momentum=false` here (Rust constructor takes
            // a smaller arg surface than the upstream `__init__`), we keep
            // the same default boundary values so a `with_cycle_momentum`
            // call with no arguments matches upstream's behaviour.
            cycle_momentum: false,
            base_momentum: 0.8,
            max_momentum: 0.9,
            current_momentum: 0.9,
        }
    }

    /// Replace the built-in amplitude scaling with a user-provided closure.
    ///
    /// When set, `compute_lr` evaluates `scale_fn(cycle)` instead of the
    /// built-in `Triangular`/`Triangular2`/`ExpRange` amplitude. The closure
    /// receives the (1-indexed) cycle number and must return an amplitude
    /// factor (typically in `[0, 1]`, though no bound is enforced — upstream
    /// is equally permissive).
    ///
    /// Mirrors PyTorch's `scale_fn` parameter at
    /// `torch/optim/lr_scheduler.py:1830-1834`.
    #[must_use]
    pub fn with_scale_fn(mut self, scale_fn: ScaleFn) -> Self {
        self.scale_fn = Some(scale_fn);
        self
    }

    /// Enable momentum cycling.
    ///
    /// When enabled, the scheduler drives the optimizer's momentum
    /// coefficient inversely with the learning rate: momentum is at
    /// `max_momentum` when LR is at `base_lr`, and at `base_momentum` when
    /// LR is at `max_lr`. The underlying optimizer must override
    /// [`Optimizer::set_momentum`] (only SGD-family today).
    ///
    /// Mirrors `cycle_momentum=True` at
    /// `torch/optim/lr_scheduler.py:1840-1862, 1935-1963`.
    #[must_use]
    pub fn with_cycle_momentum(mut self, base_momentum: f64, max_momentum: f64) -> Self {
        self.cycle_momentum = true;
        self.base_momentum = base_momentum;
        self.max_momentum = max_momentum;
        self.current_momentum = max_momentum;
        self
    }

    /// Return the current learning rate.
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Return the current momentum coefficient (only meaningful when
    /// `cycle_momentum` is enabled).
    pub fn get_momentum(&self) -> f64 {
        self.current_momentum
    }

    /// Compute the amplitude factor (in `[0, 1]`) at the given step.
    ///
    /// When a user `scale_fn` is configured, dispatches to it (passing the
    /// 1-indexed cycle number). Otherwise applies the built-in
    /// `Triangular`/`Triangular2`/`ExpRange` formula.
    fn compute_amplitude(&self, step: usize, cycle: f64, x: f64) -> f64 {
        // Triangular wave: ramp up then ramp down.
        let triangular_scale = if x <= self.step_ratio {
            x / self.step_ratio
        } else {
            (x - 1.0) / (self.step_ratio - 1.0)
        };

        if let Some(ref scale_fn) = self.scale_fn {
            // Upstream passes the cycle number (or step count, depending on
            // `scale_mode`) to the user closure. Mirror upstream's default
            // `scale_mode="cycle"` and pass the cycle number; the closure
            // multiplies the triangular wave amplitude.
            return triangular_scale * scale_fn(cycle);
        }

        match self.mode {
            CyclicMode::Triangular => triangular_scale,
            CyclicMode::Triangular2 => triangular_scale / 2.0_f64.powf(cycle - 1.0),
            CyclicMode::ExpRange => triangular_scale * self.gamma.powi(step as i32),
        }
    }

    /// Compute the learning rate at the given step.
    fn compute_lr(&self, step: usize) -> f64 {
        // Cycle number (1-indexed).
        let cycle = (1.0 + step as f64 / self.total_size).floor();
        // Position within the cycle [0, 1).
        let x = 1.0 + step as f64 / self.total_size - cycle;

        let amplitude = self.compute_amplitude(step, cycle, x);
        let base_height = (self.max_lr - self.base_lr) * amplitude;
        self.base_lr + base_height
    }

    /// Compute the momentum at the given step.
    ///
    /// Momentum cycles inversely with LR: at the LR peak (`x ==
    /// step_ratio`), momentum is at `base_momentum`; at the LR valley,
    /// momentum is at `max_momentum`. Mirrors upstream's formula at
    /// `torch/optim/lr_scheduler.py:1935-1963`.
    fn compute_momentum(&self, step: usize) -> f64 {
        let cycle = (1.0 + step as f64 / self.total_size).floor();
        let x = 1.0 + step as f64 / self.total_size - cycle;
        let amplitude = self.compute_amplitude(step, cycle, x);
        // Inverse: at amplitude=1.0 (LR peak), momentum is base_momentum;
        // at amplitude=0.0 (LR valley), momentum is max_momentum.
        let momentum_range = self.max_momentum - self.base_momentum;
        self.max_momentum - momentum_range * amplitude
    }
}

impl<T: Float> LrScheduler<T> for CyclicLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer<T>) {
        self.current_step += 1;
        self.current_lr = self.compute_lr(self.current_step);
        optimizer.set_lr(self.current_lr);

        if self.cycle_momentum {
            let new_momentum = self.compute_momentum(self.current_step);
            self.current_momentum = new_momentum;
            // Best-effort writeback for every group. Optimizers without
            // a settable momentum (Adam family) return Err from the
            // default trait impl; we silently ignore those — upstream
            // PyTorch also raises only at construction time, not on each
            // step, so the schedule continues for the LR side and the
            // momentum write is treated as a no-op when unsupported.
            for gi in 0..optimizer.param_groups().len() {
                let _ = optimizer.set_momentum(gi, new_momentum);
            }
        }
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockOptimizer {
        lr: f64,
    }

    impl MockOptimizer {
        fn new(lr: f64) -> Self {
            Self { lr }
        }
    }

    impl Optimizer<f32> for MockOptimizer {
        fn step(&mut self) -> ferrotorch_core::FerrotorchResult<()> {
            Ok(())
        }
        fn zero_grad(&mut self) -> ferrotorch_core::FerrotorchResult<()> {
            Ok(())
        }
        fn lr(&self) -> f64 {
            self.lr
        }
        fn set_lr(&mut self, lr: f64) {
            self.lr = lr;
        }
        fn param_groups(&self) -> &[crate::optimizer::ParamGroup<f32>] {
            &[]
        }
        fn param_groups_mut(&mut self) -> &mut [crate::optimizer::ParamGroup<f32>] {
            &mut []
        }
        fn add_param_group(&mut self, _group: crate::optimizer::ParamGroup<f32>) {}
        fn state_dict(
            &self,
        ) -> ferrotorch_core::FerrotorchResult<crate::optimizer::OptimizerState> {
            Ok(Default::default())
        }
        fn load_state_dict(
            &mut self,
            _state: &crate::optimizer::OptimizerState,
        ) -> ferrotorch_core::FerrotorchResult<()> {
            Ok(())
        }
    }

    #[test]
    fn test_cyclic_triangular_peak() {
        // step_size_up=10, step_size_down=10 => total=20, step_ratio=0.5
        // At step 10, x = 1 + 10/20 - 1 = 0.5, scale_factor = 0.5/0.5 = 1.0
        // => lr = base + (max - base) * 1.0 = max_lr
        let mut sched = CyclicLR::new(0.001, 0.01, 10, None, CyclicMode::Triangular, 1.0);
        let mut opt = MockOptimizer::new(0.001);

        for _ in 0..10 {
            sched.step(&mut opt);
        }
        assert!(
            (opt.lr - 0.01).abs() < 1e-10,
            "expected peak 0.01, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_cyclic_triangular_valley() {
        // After a full cycle (20 steps), should be back at base_lr.
        let mut sched = CyclicLR::new(0.001, 0.01, 10, None, CyclicMode::Triangular, 1.0);
        let mut opt = MockOptimizer::new(0.001);

        for _ in 0..20 {
            sched.step(&mut opt);
        }
        // At step 20: cycle=2, x = 1 + 20/20 - 2 = 0.0
        // scale_factor = 0.0/0.5 = 0.0 => lr = base_lr
        assert!(
            (opt.lr - 0.001).abs() < 1e-10,
            "expected valley 0.001, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_cyclic_triangular2_halves_amplitude() {
        let base = 0.0;
        let max = 1.0;
        let mut sched = CyclicLR::new(base, max, 10, None, CyclicMode::Triangular2, 1.0);
        let mut opt = MockOptimizer::new(base);

        // Peak of cycle 1 (step 10): full amplitude.
        for _ in 0..10 {
            sched.step(&mut opt);
        }
        let peak1 = opt.lr;

        // Complete cycle 1, then peak of cycle 2 (step 30).
        for _ in 0..20 {
            sched.step(&mut opt);
        }
        let peak2 = opt.lr;

        // Peak 2 should be half of peak 1.
        assert!(
            (peak2 - peak1 / 2.0).abs() < 1e-10,
            "expected peak2={}, got {}",
            peak1 / 2.0,
            peak2
        );
    }

    #[test]
    fn test_cyclic_exp_range_decays() {
        let base = 0.0;
        let max = 1.0;
        let gamma = 0.99;
        let mut sched = CyclicLR::new(base, max, 10, None, CyclicMode::ExpRange, gamma);
        let mut opt = MockOptimizer::new(base);

        // Peak of cycle 1.
        for _ in 0..10 {
            sched.step(&mut opt);
        }
        let peak1 = opt.lr;

        // Peak of cycle 2.
        for _ in 0..20 {
            sched.step(&mut opt);
        }
        let peak2 = opt.lr;

        // exp_range should cause peak2 < peak1.
        assert!(
            peak2 < peak1,
            "exp_range should decay: peak1={peak1}, peak2={peak2}"
        );
    }

    #[test]
    fn test_cyclic_asymmetric_cycle() {
        // step_size_up=5, step_size_down=15 => faster ascent, slower descent.
        let mut sched = CyclicLR::new(0.0, 1.0, 5, Some(15), CyclicMode::Triangular, 1.0);
        let mut opt = MockOptimizer::new(0.0);

        // At step 5: should be at peak.
        for _ in 0..5 {
            sched.step(&mut opt);
        }
        assert!(
            (opt.lr - 1.0).abs() < 1e-10,
            "expected peak 1.0, got {}",
            opt.lr
        );

        // At step 20: should be back at base.
        for _ in 0..15 {
            sched.step(&mut opt);
        }
        assert!(opt.lr.abs() < 1e-10, "expected valley 0.0, got {}", opt.lr);
    }

    #[test]
    fn test_cyclic_midpoint_ramp_up() {
        // step_size_up=10, symmetric.
        // At step 5: halfway up.
        let base = 0.0;
        let max = 1.0;
        let mut sched = CyclicLR::new(base, max, 10, None, CyclicMode::Triangular, 1.0);
        let mut opt = MockOptimizer::new(base);

        for _ in 0..5 {
            sched.step(&mut opt);
        }
        // x = 1 + 5/20 - 1 = 0.25, scale_factor = 0.25/0.5 = 0.5
        assert!((opt.lr - 0.5).abs() < 1e-10, "expected 0.5, got {}", opt.lr);
    }
}
