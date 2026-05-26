//! One-cycle learning rate scheduler.
//!
//! Implements the 1cycle policy from "Super-Convergence: Very Fast Training
//! of Neural Networks Using Large Learning Rates" (Smith & Topin, 2018).
//!
//! The learning rate anneals from `initial_lr = max_lr / div_factor` up to
//! `max_lr`, then back down to `min_lr = initial_lr / final_div_factor`.
//! Supports both cosine and linear annealing strategies, and an optional
//! three-phase variant.
//!
//! [CL-320]
//!
//! ## REQ status (per `.design/ferrotorch-optim/scheduler/one_cycle_lr.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub enum AnnealStrategy { Cos, Linear }` in `scheduler/one_cycle_lr.rs` mirrors `torch/optim/lr_scheduler.py:2329`; consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`. |
//! | REQ-2 | SHIPPED | `pub struct OneCycleLR` + private `SchedulePhase` in `scheduler/one_cycle_lr.rs` mirror `torch/optim/lr_scheduler.py:2454-2520`; consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`; user code boxes for `Learner::with_scheduler` at `ferrotorch-train/src/learner.rs:105`. |
//! | REQ-3 | SHIPPED | `pub fn OneCycleLR::new(max_lr, total_steps, pct_start, ..., three_phase)` with `assert!`s in `scheduler/one_cycle_lr.rs` mirrors `torch/optim/lr_scheduler.py:2358-2520`; consumer: re-exported via `lib.rs:47-52`. |
//! | REQ-4 | SHIPPED | `impl<T: Float> LrScheduler<T> for OneCycleLR` with phase-aware compute in `scheduler/one_cycle_lr.rs` mirrors `torch/optim/lr_scheduler.py:2538-2602`; consumer: `Learner` per-epoch `sched.step` at `ferrotorch-train/src/learner.rs:306-308`. |
//! | REQ-5 | SHIPPED | Two-phase vs three-phase phase-table branching in the constructor (`scheduler/one_cycle_lr.rs`) mirrors `torch/optim/lr_scheduler.py:2454-2520`; consumer: `Learner` per-epoch `sched.step` at `ferrotorch-train/src/learner.rs:306-308` dispatches into the chosen phase table. |
//! | REQ-6 | SHIPPED | `cycle_momentum`/`base_momentum`/`max_momentum` fields + `with_cycle_momentum` builder + `compute_momentum` per-step writeback via `optimizer.set_momentum` mirror `torch/optim/lr_scheduler.py:2342-2350, 2391-2453`; consumer: `<OneCycleLR as LrScheduler<T>>::step` invokes both `set_lr` and `set_momentum`, and `Learner::with_scheduler` at `ferrotorch-train/src/learner.rs:142` boxes any `OneCycleLR` (with or without momentum cycling) and drives `sched.step` per epoch (closes #1474). |

use ferrotorch_core::Float;

use super::LrScheduler;
use crate::optimizer::Optimizer;

/// Annealing strategy for the one-cycle policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnealStrategy {
    /// Cosine annealing between phase endpoints.
    Cos,
    /// Linear annealing between phase endpoints.
    Linear,
}

/// A single phase of the one-cycle schedule.
///
/// Each phase carries both LR endpoints and momentum endpoints; when
/// `cycle_momentum` is disabled the momentum endpoints are unused but
/// still stored (avoids a parallel `MomentumPhase` table and the risk of
/// the two tables drifting out of sync). Mirrors upstream's
/// `_schedule_phases` list-of-dicts at
/// `torch/optim/lr_scheduler.py:2454-2520` where each entry holds both
/// `(end_step, start_lr, end_lr, start_momentum, end_momentum)`.
#[derive(Debug, Clone)]
struct SchedulePhase {
    /// End step of this phase (inclusive).
    end_step: f64,
    /// Starting LR for this phase.
    start_lr: f64,
    /// Ending LR for this phase.
    end_lr: f64,
    /// Starting momentum for this phase (only consulted when
    /// `cycle_momentum == true`).
    start_momentum: f64,
    /// Ending momentum for this phase (only consulted when
    /// `cycle_momentum == true`).
    end_momentum: f64,
}

/// One-cycle learning rate scheduler.
///
/// # Two-phase mode (default)
///
/// 1. Ramp from `initial_lr` to `max_lr` over `pct_start * total_steps`.
/// 2. Anneal from `max_lr` to `min_lr` over the remaining steps.
///
/// # Three-phase mode
///
/// 1. Ramp from `initial_lr` to `max_lr` over `pct_start * total_steps`.
/// 2. Anneal from `max_lr` back to `initial_lr` over `pct_start * total_steps`.
/// 3. Anneal from `initial_lr` to `min_lr` over the remaining steps.
///
/// # Momentum cycling
///
/// When [`Self::with_cycle_momentum`] is configured, the scheduler also
/// drives the optimizer's momentum coefficient inversely with the
/// learning rate — momentum is at `max_momentum` when LR is at the
/// initial / minimum value and at `base_momentum` when LR is at
/// `max_lr`. Requires the underlying optimizer to override
/// [`Optimizer::set_momentum`] (only SGD-family today). Mirrors
/// `torch/optim/lr_scheduler.py:2342-2350, 2391-2453`.
///
/// # Example
///
/// ```ignore
/// let scheduler = OneCycleLR::new(
///     0.01,   // max_lr
///     1000,   // total_steps
///     0.3,    // pct_start
///     AnnealStrategy::Cos,
///     25.0,   // div_factor
///     1e4,    // final_div_factor
///     false,  // three_phase
/// )
/// .with_cycle_momentum(0.85, 0.95);
/// ```
#[derive(Debug, Clone)]
pub struct OneCycleLR {
    /// Schedule phases.
    phases: Vec<SchedulePhase>,
    /// Total number of steps (retained for state introspection).
    #[allow(dead_code)]
    total_steps: usize,
    /// Annealing strategy.
    anneal_strategy: AnnealStrategy,
    /// Current step count.
    current_step: usize,
    /// Current computed learning rate.
    current_lr: f64,
    /// When `true`, also cycle momentum inversely with the learning
    /// rate. Mirrors `cycle_momentum=True` at
    /// `torch/optim/lr_scheduler.py:2342`.
    cycle_momentum: bool,
    /// Lower momentum boundary (used when `cycle_momentum == true`).
    /// Mirrors `base_momentum` at `torch/optim/lr_scheduler.py:2345`.
    base_momentum: f64,
    /// Upper momentum boundary (used when `cycle_momentum == true`).
    /// Mirrors `max_momentum` at `torch/optim/lr_scheduler.py:2348`.
    max_momentum: f64,
    /// Current computed momentum (cached for introspection).
    current_momentum: f64,
}

impl OneCycleLR {
    /// Create a new `OneCycleLR` scheduler.
    ///
    /// # Arguments
    ///
    /// * `max_lr` - Upper learning rate boundary.
    /// * `total_steps` - Total number of training steps.
    /// * `pct_start` - Fraction of steps spent in the increasing phase (0..1).
    /// * `anneal_strategy` - `Cos` or `Linear` annealing.
    /// * `div_factor` - Determines initial LR: `initial_lr = max_lr / div_factor`.
    /// * `final_div_factor` - Determines final LR: `min_lr = initial_lr / final_div_factor`.
    /// * `three_phase` - If `true`, use the three-phase variant.
    ///
    /// # Panics
    ///
    /// Panics if `total_steps == 0` or `pct_start` is not in `[0, 1]`.
    pub fn new(
        max_lr: f64,
        total_steps: usize,
        pct_start: f64,
        anneal_strategy: AnnealStrategy,
        div_factor: f64,
        final_div_factor: f64,
        three_phase: bool,
    ) -> Self {
        assert!(total_steps > 0, "total_steps must be > 0");
        assert!(
            (0.0..=1.0).contains(&pct_start),
            "pct_start must be in [0, 1], got {pct_start}"
        );

        let initial_lr = max_lr / div_factor;
        let min_lr = initial_lr / final_div_factor;

        // Default momentum boundaries match upstream
        // `torch/optim/lr_scheduler.py:2344-2348`: `base_momentum=0.85`,
        // `max_momentum=0.95`. Phase tables encode the inverse coupling
        // (momentum is at `max_momentum` at the LR valley/start and at
        // `base_momentum` at the LR peak). When `cycle_momentum` is
        // disabled the momentum endpoints in the phase table are
        // simply not consulted; we still populate them so the table is
        // structurally complete and `with_cycle_momentum` toggling
        // post-construction is symmetric.
        let default_base_momentum = 0.85_f64;
        let default_max_momentum = 0.95_f64;

        let phases = if three_phase {
            vec![
                SchedulePhase {
                    end_step: pct_start * total_steps as f64 - 1.0,
                    start_lr: initial_lr,
                    end_lr: max_lr,
                    start_momentum: default_max_momentum,
                    end_momentum: default_base_momentum,
                },
                SchedulePhase {
                    end_step: 2.0 * pct_start * total_steps as f64 - 2.0,
                    start_lr: max_lr,
                    end_lr: initial_lr,
                    start_momentum: default_base_momentum,
                    end_momentum: default_max_momentum,
                },
                SchedulePhase {
                    end_step: total_steps as f64 - 1.0,
                    start_lr: initial_lr,
                    end_lr: min_lr,
                    start_momentum: default_max_momentum,
                    end_momentum: default_max_momentum,
                },
            ]
        } else {
            vec![
                SchedulePhase {
                    end_step: pct_start * total_steps as f64 - 1.0,
                    start_lr: initial_lr,
                    end_lr: max_lr,
                    start_momentum: default_max_momentum,
                    end_momentum: default_base_momentum,
                },
                SchedulePhase {
                    end_step: total_steps as f64 - 1.0,
                    start_lr: max_lr,
                    end_lr: min_lr,
                    start_momentum: default_base_momentum,
                    end_momentum: default_max_momentum,
                },
            ]
        };

        Self {
            phases,
            total_steps,
            anneal_strategy,
            current_step: 0,
            current_lr: initial_lr,
            cycle_momentum: false,
            base_momentum: default_base_momentum,
            max_momentum: default_max_momentum,
            current_momentum: default_max_momentum,
        }
    }

    /// Enable momentum cycling on this `OneCycleLR`.
    ///
    /// Mirrors `cycle_momentum=True` at
    /// `torch/optim/lr_scheduler.py:2342-2350`. The scheduler will
    /// drive the optimizer's momentum coefficient inversely with the
    /// learning rate: momentum is at `max_momentum` when LR is at the
    /// initial / minimum value, and at `base_momentum` when LR is at
    /// `max_lr`. The underlying optimizer must override
    /// [`Optimizer::set_momentum`] (only SGD-family today); for
    /// optimizers without a momentum coefficient the per-step
    /// `set_momentum` call is silently treated as a no-op (mirroring
    /// upstream's permissive behaviour on each step).
    ///
    /// # Arguments
    ///
    /// * `base_momentum` — lower momentum boundary (reached at the LR
    ///   peak).
    /// * `max_momentum` — upper momentum boundary (reached at the LR
    ///   valley / initial step).
    #[must_use]
    pub fn with_cycle_momentum(mut self, base_momentum: f64, max_momentum: f64) -> Self {
        self.cycle_momentum = true;
        self.base_momentum = base_momentum;
        self.max_momentum = max_momentum;
        self.current_momentum = max_momentum;

        // Rebuild the phase table's momentum endpoints from the new
        // boundary values. The LR phase structure is unchanged; only
        // the momentum endpoints per phase are rewritten.
        let three_phase = self.phases.len() == 3;
        if three_phase {
            // Phase 1: ramp-up — momentum max → base.
            self.phases[0].start_momentum = max_momentum;
            self.phases[0].end_momentum = base_momentum;
            // Phase 2: anneal back — momentum base → max.
            self.phases[1].start_momentum = base_momentum;
            self.phases[1].end_momentum = max_momentum;
            // Phase 3: further-anneal — momentum stays at max.
            self.phases[2].start_momentum = max_momentum;
            self.phases[2].end_momentum = max_momentum;
        } else {
            // Two-phase: ramp-up max → base; anneal-down base → max.
            self.phases[0].start_momentum = max_momentum;
            self.phases[0].end_momentum = base_momentum;
            self.phases[1].start_momentum = base_momentum;
            self.phases[1].end_momentum = max_momentum;
        }
        self
    }

    /// Return the current momentum coefficient (only meaningful when
    /// `cycle_momentum` is enabled).
    pub fn get_momentum(&self) -> f64 {
        self.current_momentum
    }

    /// Return the current learning rate.
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Cosine anneal from `start` to `end` as `pct` goes from 0 to 1.
    fn anneal_cos(start: f64, end: f64, pct: f64) -> f64 {
        let cos_out = (std::f64::consts::PI * pct).cos() + 1.0;
        end + (start - end) / 2.0 * cos_out
    }

    /// Linear anneal from `start` to `end` as `pct` goes from 0 to 1.
    fn anneal_linear(start: f64, end: f64, pct: f64) -> f64 {
        (end - start) * pct + start
    }

    /// Compute the learning rate at the given step.
    fn compute_lr(&self, step: usize) -> f64 {
        let step_num = step as f64;
        let mut start_step = 0.0_f64;

        for (i, phase) in self.phases.iter().enumerate() {
            if step_num <= phase.end_step || i == self.phases.len() - 1 {
                let pct = if (phase.end_step - start_step).abs() < 1e-12 {
                    1.0
                } else {
                    (step_num - start_step) / (phase.end_step - start_step)
                };
                return match self.anneal_strategy {
                    AnnealStrategy::Cos => Self::anneal_cos(phase.start_lr, phase.end_lr, pct),
                    AnnealStrategy::Linear => {
                        Self::anneal_linear(phase.start_lr, phase.end_lr, pct)
                    }
                };
            }
            start_step = phase.end_step;
        }

        // Shouldn't reach here, but return the last phase's end_lr.
        self.phases.last().map(|p| p.end_lr).unwrap_or(0.0)
    }

    /// Compute the momentum coefficient at the given step.
    ///
    /// Walks the same phase table as [`Self::compute_lr`] but
    /// interpolates between the per-phase `(start_momentum,
    /// end_momentum)` endpoints. Mirrors upstream's twin-loop
    /// momentum lookup at `torch/optim/lr_scheduler.py:2391-2453`,
    /// which uses the SAME per-phase `pct` for the momentum
    /// interpolation as it does for the LR interpolation, so the two
    /// schedules stay perfectly phase-aligned.
    fn compute_momentum(&self, step: usize) -> f64 {
        let step_num = step as f64;
        let mut start_step = 0.0_f64;

        for (i, phase) in self.phases.iter().enumerate() {
            if step_num <= phase.end_step || i == self.phases.len() - 1 {
                let pct = if (phase.end_step - start_step).abs() < 1e-12 {
                    1.0
                } else {
                    (step_num - start_step) / (phase.end_step - start_step)
                };
                return match self.anneal_strategy {
                    AnnealStrategy::Cos => {
                        Self::anneal_cos(phase.start_momentum, phase.end_momentum, pct)
                    }
                    AnnealStrategy::Linear => {
                        Self::anneal_linear(phase.start_momentum, phase.end_momentum, pct)
                    }
                };
            }
            start_step = phase.end_step;
        }

        self.phases.last().map(|p| p.end_momentum).unwrap_or(0.0)
    }
}

impl<T: Float> LrScheduler<T> for OneCycleLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer<T>) {
        self.current_lr = self.compute_lr(self.current_step);
        optimizer.set_lr(self.current_lr);

        if self.cycle_momentum {
            // Best-effort writeback to every group, matching CyclicLR's
            // wiring at `cyclic_lr.rs`. Optimizers without a settable
            // momentum coefficient (Adam family) return Err from the
            // default trait impl; we silently ignore those — upstream
            // PyTorch raises only at construction time, not on each
            // step, so the LR schedule keeps progressing while the
            // momentum write is treated as a no-op when unsupported.
            // Mirrors `torch/optim/lr_scheduler.py:2391-2453`.
            let new_momentum = self.compute_momentum(self.current_step);
            self.current_momentum = new_momentum;
            for gi in 0..optimizer.param_groups().len() {
                let _ = optimizer.set_momentum(gi, new_momentum);
            }
        }

        self.current_step += 1;
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
    fn test_one_cycle_initial_lr() {
        let max_lr = 0.01;
        let div_factor = 25.0;
        let sched = OneCycleLR::new(
            max_lr,
            100,
            0.3,
            AnnealStrategy::Cos,
            div_factor,
            1e4,
            false,
        );
        let expected_initial = max_lr / div_factor;
        assert!(
            (sched.get_lr() - expected_initial).abs() < 1e-12,
            "expected {expected_initial}, got {}",
            sched.get_lr()
        );
    }

    #[test]
    fn test_one_cycle_reaches_max_lr_cos() {
        let max_lr = 0.1;
        let total_steps = 100;
        let pct_start = 0.3;
        let mut sched = OneCycleLR::new(
            max_lr,
            total_steps,
            pct_start,
            AnnealStrategy::Cos,
            25.0,
            1e4,
            false,
        );
        let mut opt = MockOptimizer::new(0.004);

        // Step to the end of phase 1.
        let phase1_end = (pct_start * total_steps as f64) as usize;
        for _ in 0..phase1_end {
            sched.step(&mut opt);
        }

        // At the boundary, LR should be close to max_lr.
        // It won't be exact due to phase boundary math, but should be within tolerance.
        assert!(
            (opt.lr - max_lr).abs() < 0.01,
            "at phase boundary: expected ~{max_lr}, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_one_cycle_end_lr() {
        let max_lr = 0.1;
        let total_steps = 100;
        let div_factor = 25.0;
        let final_div_factor = 1e4;
        let mut sched = OneCycleLR::new(
            max_lr,
            total_steps,
            0.3,
            AnnealStrategy::Cos,
            div_factor,
            final_div_factor,
            false,
        );
        let mut opt = MockOptimizer::new(0.004);

        // Run all steps.
        for _ in 0..total_steps {
            sched.step(&mut opt);
        }

        let initial_lr = max_lr / div_factor;
        let min_lr = initial_lr / final_div_factor;
        assert!(
            (opt.lr - min_lr).abs() < 1e-10,
            "expected min_lr={min_lr}, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_one_cycle_linear_monotonic_ramp() {
        let max_lr = 1.0;
        let total_steps = 100;
        let pct_start = 0.3;
        let mut sched = OneCycleLR::new(
            max_lr,
            total_steps,
            pct_start,
            AnnealStrategy::Linear,
            25.0,
            1e4,
            false,
        );
        let mut opt = MockOptimizer::new(0.04);

        let phase1_steps = (pct_start * total_steps as f64) as usize;
        let mut prev_lr = 0.0;
        for i in 0..phase1_steps {
            sched.step(&mut opt);
            if i > 0 {
                assert!(
                    opt.lr >= prev_lr - 1e-12,
                    "step {i}: LR should be monotonically increasing in ramp phase"
                );
            }
            prev_lr = opt.lr;
        }
    }

    #[test]
    fn test_one_cycle_three_phase() {
        let max_lr = 0.1;
        let total_steps = 100;
        let pct_start = 0.3;
        let div_factor = 25.0;
        let final_div_factor = 1e4;
        let mut sched = OneCycleLR::new(
            max_lr,
            total_steps,
            pct_start,
            AnnealStrategy::Cos,
            div_factor,
            final_div_factor,
            true,
        );
        let mut opt = MockOptimizer::new(0.004);

        let initial_lr = max_lr / div_factor;
        let min_lr = initial_lr / final_div_factor;

        // Run all steps.
        for _ in 0..total_steps {
            sched.step(&mut opt);
        }

        // At the end of three-phase, LR should approach min_lr.
        assert!(
            (opt.lr - min_lr).abs() < 1e-10,
            "three_phase end: expected min_lr={min_lr}, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_one_cycle_lr_never_negative() {
        let mut sched = OneCycleLR::new(0.01, 200, 0.3, AnnealStrategy::Cos, 25.0, 1e4, false);
        let mut opt = MockOptimizer::new(0.0004);

        for step in 0..200 {
            sched.step(&mut opt);
            assert!(
                opt.lr >= 0.0,
                "step {step}: LR should never be negative, got {}",
                opt.lr
            );
        }
    }

    #[test]
    #[should_panic(expected = "total_steps must be > 0")]
    fn test_one_cycle_zero_steps_panics() {
        OneCycleLR::new(0.01, 0, 0.3, AnnealStrategy::Cos, 25.0, 1e4, false);
    }

    // -----------------------------------------------------------------------
    // cycle_momentum tests (#1474)
    // -----------------------------------------------------------------------

    /// `with_cycle_momentum` toggles the flag and updates the cached
    /// boundary values. Without calling `step()`, `current_momentum`
    /// equals `max_momentum` (the boundary the schedule starts at).
    #[test]
    fn test_one_cycle_with_cycle_momentum_initial_value() {
        let sched = OneCycleLR::new(0.01, 100, 0.3, AnnealStrategy::Cos, 25.0, 1e4, false)
            .with_cycle_momentum(0.85, 0.95);
        assert!((sched.get_momentum() - 0.95).abs() < 1e-12);
    }

    /// At the LR peak (step == pct_start * total_steps), momentum is
    /// at `base_momentum`. Verified against a real SGD optimizer (the
    /// only optimizer family whose `set_momentum` override actually
    /// writes through).
    #[test]
    fn test_one_cycle_momentum_at_peak_is_base_momentum() {
        use crate::{Sgd, SgdConfig};
        use ferrotorch_nn::Parameter;

        let p = Parameter::<f32>::from_slice(&[1.0_f32], &[1]).unwrap();
        let mut sgd = Sgd::new(vec![p], SgdConfig::new(0.001).with_momentum(0.95));

        let total_steps = 100;
        let pct_start = 0.3;
        let mut sched = OneCycleLR::new(
            0.01,
            total_steps,
            pct_start,
            AnnealStrategy::Cos,
            25.0,
            1e4,
            false,
        )
        .with_cycle_momentum(0.85, 0.95);

        // Step through the entire ramp-up phase.
        let phase1_steps = (pct_start * total_steps as f64) as usize;
        for _ in 0..phase1_steps {
            sched.step(&mut sgd);
        }

        // At the peak boundary, momentum must be at base_momentum=0.85.
        let m = sgd.momentum(0).expect("read momentum");
        assert!(
            (m - 0.85).abs() < 1e-3,
            "at LR peak momentum must be base_momentum=0.85; got {m}"
        );
    }

    /// At the LR valley (step == total_steps), momentum returns to
    /// `max_momentum`. Inverse coupling guarantee.
    #[test]
    fn test_one_cycle_momentum_returns_to_max_at_valley() {
        use crate::{Sgd, SgdConfig};
        use ferrotorch_nn::Parameter;

        let p = Parameter::<f32>::from_slice(&[1.0_f32], &[1]).unwrap();
        let mut sgd = Sgd::new(vec![p], SgdConfig::new(0.001).with_momentum(0.95));

        let total_steps = 100;
        let mut sched = OneCycleLR::new(
            0.01,
            total_steps,
            0.3,
            AnnealStrategy::Cos,
            25.0,
            1e4,
            false,
        )
        .with_cycle_momentum(0.85, 0.95);

        for _ in 0..total_steps {
            sched.step(&mut sgd);
        }

        let m = sgd.momentum(0).expect("read momentum");
        // At total_steps the LR is at min_lr (near zero); momentum
        // should be back at max_momentum=0.95.
        assert!(
            (m - 0.95).abs() < 1e-3,
            "at LR valley momentum must return to max_momentum=0.95; got {m}"
        );
    }

    /// Without `with_cycle_momentum`, the optimizer's momentum is
    /// never touched — a sabotage probe against a "wired regardless"
    /// regression.
    #[test]
    fn test_one_cycle_without_cycle_momentum_does_not_write() {
        use crate::{Sgd, SgdConfig};
        use ferrotorch_nn::Parameter;

        let p = Parameter::<f32>::from_slice(&[1.0_f32], &[1]).unwrap();
        let mut sgd = Sgd::new(vec![p], SgdConfig::new(0.001).with_momentum(0.7));

        let mut sched = OneCycleLR::new(0.01, 100, 0.3, AnnealStrategy::Cos, 25.0, 1e4, false);

        for _ in 0..50 {
            sched.step(&mut sgd);
        }

        let m = sgd.momentum(0).expect("read momentum");
        assert!(
            (m - 0.7).abs() < 1e-12,
            "without cycle_momentum the optimizer momentum must stay at its initial value 0.7; got {m}"
        );
    }

    /// `get_momentum` accessor tracks the per-step computed value.
    #[test]
    fn test_one_cycle_get_momentum_accessor_tracks_compute() {
        use crate::{Sgd, SgdConfig};
        use ferrotorch_nn::Parameter;

        let p = Parameter::<f32>::from_slice(&[1.0_f32], &[1]).unwrap();
        let mut sgd = Sgd::new(vec![p], SgdConfig::new(0.001).with_momentum(0.95));

        let mut sched = OneCycleLR::new(0.01, 100, 0.3, AnnealStrategy::Cos, 25.0, 1e4, false)
            .with_cycle_momentum(0.85, 0.95);

        // Take a few steps and verify get_momentum() reflects the
        // computed value (not the constructor default).
        for _ in 0..10 {
            sched.step(&mut sgd);
        }
        let mom = sched.get_momentum();
        // The exact value depends on the cosine interpolation; just
        // assert it moved off the initial boundary toward base_momentum.
        assert!(
            (0.85..=0.95).contains(&mom),
            "get_momentum() should be in [base, max]; got {mom}"
        );
        assert!(
            (mom - 0.95).abs() > 1e-6,
            "get_momentum() should have moved off the initial max_momentum; got {mom}"
        );
    }
}
