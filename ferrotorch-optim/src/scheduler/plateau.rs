//! Reduce learning rate on plateau scheduler.
//!
//! Monitors a metric and reduces the learning rate when the metric has
//! stopped improving for `patience` steps.
//!
//! ## REQ status (per `.design/ferrotorch-optim/scheduler/plateau.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub enum PlateauMode { Min, Max }` in `scheduler/plateau.rs` mirrors `torch/optim/lr_scheduler.py:1650`; consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`. |
//! | REQ-2 | SHIPPED | `pub struct ReduceLROnPlateau` schedule state in `scheduler/plateau.rs` mirrors `torch/optim/lr_scheduler.py:1647-1687`; consumer: re-exported via `lib.rs:47-52`; non-test consumer: `Learner::with_metric_scheduler` at `ferrotorch-train/src/learner.rs` boxes a `ReduceLROnPlateau` and drives its `step(metric)` per epoch (closes #1475). |
//! | REQ-3 | SHIPPED | `pub fn ReduceLROnPlateau::new(mode)` + builder methods (`factor`/`patience`/`min_lr`/`threshold`/`with_cooldown`/`with_eps`/`with_threshold_mode`/`with_per_group_min_lr`) in `scheduler/plateau.rs` mirror `torch/optim/lr_scheduler.py:1583-1786` (R-DEV-7 builder); consumer: re-exported via `lib.rs:47-52`. |
//! | REQ-4 | SHIPPED | `pub trait MetricScheduler<T: Float>` in `scheduler/plateau.rs` mirrors the `step(metrics)` signature at `torch/optim/lr_scheduler.py:1695`; consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`; non-test consumer: `Learner` boxes `dyn MetricScheduler<T>` via `with_metric_scheduler` and dispatches `step(opt, metric)` per epoch (closes #1475). |
//! | REQ-5 | SHIPPED | `impl<T: Float> MetricScheduler<T> for ReduceLROnPlateau` first-call snapshot + best-tracking + cooldown + reduction in `scheduler/plateau.rs` mirrors `torch/optim/lr_scheduler.py:1695-1742`; non-test consumer: `Learner::fit` invokes `metric_sched.step(self.optimizer.as_mut(), val_loss)` after each epoch (closes #1475). |
//! | REQ-6 | SHIPPED | `Learner::with_metric_scheduler` at `ferrotorch-train/src/learner.rs` accepts `Box<dyn MetricScheduler<T>>`, and `Learner::fit` invokes `metric_sched.step(opt, metric)` per epoch with the validation loss (falling back to training loss when validation is absent); closes #1475. |
//! | REQ-7 | SHIPPED | `with_cooldown(usize)` + `with_eps(f64)` + `with_threshold_mode(ThresholdMode)` + `with_per_group_min_lr(Vec<f64>)` builders on `ReduceLROnPlateau` mirror `torch/optim/lr_scheduler.py:1625-1632, 1684`; consumer: re-exported at `lib.rs:47-52`; the `MetricScheduler` step impl uses these fields per-group; closes #1476. |

use ferrotorch_core::Float;

use crate::optimizer::Optimizer;

/// Mode for plateau detection: minimize or maximize the metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlateauMode {
    /// Reduce LR when the metric stops decreasing.
    Min,
    /// Reduce LR when the metric stops increasing.
    Max,
}

/// Threshold mode for plateau improvement detection.
///
/// Mirrors `threshold_mode: Literal["rel", "abs"]` at
/// `torch/optim/lr_scheduler.py:1626-1632`.
///
/// * `Rel` — relative threshold: improvement requires
///   `metric < best * (1 - threshold)` in Min mode or
///   `metric > best * (1 + threshold)` in Max mode.
/// * `Abs` — absolute threshold: improvement requires
///   `metric < best - threshold` in Min mode or
///   `metric > best + threshold` in Max mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdMode {
    /// Relative threshold (default; matches `"rel"` upstream).
    Rel,
    /// Absolute threshold (matches `"abs"` upstream).
    Abs,
}

/// A metric-aware scheduler that reduces the learning rate when a metric
/// plateaus.
///
/// Unlike [`LrScheduler`](super::LrScheduler), this scheduler requires
/// a metric value at each step, so it implements its own
/// [`MetricScheduler`] trait instead.
///
/// # Algorithm
///
/// 1. Track the best metric value seen so far.
/// 2. If the metric has not improved for `patience` consecutive calls,
///    multiply the current learning rate by `factor`.
/// 3. Optionally enforce a minimum learning rate (`min_lr`).
///
/// # Example
///
/// ```ignore
/// let mut scheduler = ReduceLROnPlateau::new(PlateauMode::Min)
///     .factor(0.1)
///     .patience(10)
///     .min_lr(1e-6);
///
/// for epoch in 0..100 {
///     let val_loss = train_one_epoch();
///     scheduler.step(&mut optimizer, val_loss);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ReduceLROnPlateau {
    /// Whether we are minimizing or maximizing the metric.
    mode: PlateauMode,
    /// Factor by which the learning rate is reduced. `new_lr = lr * factor`.
    factor: f64,
    /// Number of steps with no improvement before reducing LR.
    patience: usize,
    /// Minimum learning rate when no per-group floor is set. LR will not
    /// be reduced below this value.
    min_lr: f64,
    /// Optional per-parameter-group minimum learning rate (#1476).
    /// When `Some`, must have `len() == optimizer.param_groups().len()`
    /// at step time; each group's floor is `per_group_min_lr[i]`. When
    /// `None`, the scalar `min_lr` is the floor for every group.
    /// Mirrors `min_lr: Union[List[float], float]` at
    /// `torch/optim/lr_scheduler.py:1684`.
    per_group_min_lr: Option<Vec<f64>>,
    /// Threshold for measuring improvement.
    threshold: f64,
    /// Threshold mode (`Rel` or `Abs`). Mirrors upstream
    /// `threshold_mode` at `torch/optim/lr_scheduler.py:1626-1632`.
    threshold_mode: ThresholdMode,
    /// Number of epochs to wait after a reduction before resuming
    /// plateau detection. Mirrors `cooldown` at
    /// `torch/optim/lr_scheduler.py:1625`.
    cooldown: usize,
    /// Minimum decay applied to LR — if the would-be `new_lr` is
    /// within `eps` of the current LR (i.e. the relative change is
    /// less than `eps`), the reduction is skipped. Mirrors `eps` at
    /// `torch/optim/lr_scheduler.py:1632`.
    eps: f64,
    /// Best metric value seen so far.
    best: f64,
    /// Number of steps since the last improvement.
    num_bad_steps: usize,
    /// Remaining cooldown steps (decremented each `step` until 0).
    cooldown_counter: usize,
    /// Current learning rate (tracked for `get_lr()`). When
    /// `per_group_min_lr` is in use this still tracks group 0's LR for
    /// `get_lr` reporting; the per-group writebacks happen via
    /// `param_groups_mut()` directly.
    current_lr: f64,
    /// Whether we have received at least one metric value.
    initialized: bool,
}

impl ReduceLROnPlateau {
    /// Create a new plateau scheduler with the given mode.
    ///
    /// Defaults:
    /// - `factor = 0.1`
    /// - `patience = 10`
    /// - `min_lr = 0.0`
    /// - `threshold = 1e-4`
    pub fn new(mode: PlateauMode) -> Self {
        let best = match mode {
            PlateauMode::Min => f64::INFINITY,
            PlateauMode::Max => f64::NEG_INFINITY,
        };
        Self {
            mode,
            factor: 0.1,
            patience: 10,
            min_lr: 0.0,
            per_group_min_lr: None,
            threshold: 1e-4,
            threshold_mode: ThresholdMode::Rel,
            cooldown: 0,
            eps: 1e-8,
            best,
            num_bad_steps: 0,
            cooldown_counter: 0,
            current_lr: 0.0,
            initialized: false,
        }
    }

    /// Set the multiplicative factor for LR reduction.
    pub fn factor(mut self, factor: f64) -> Self {
        self.factor = factor;
        self
    }

    /// Set the number of patience steps.
    pub fn patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Set the minimum learning rate.
    pub fn min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Set the threshold for measuring improvement.
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the cooldown period (number of post-reduction steps to wait
    /// before resuming plateau detection).
    ///
    /// Mirrors `cooldown` at `torch/optim/lr_scheduler.py:1625`.
    #[must_use]
    pub fn with_cooldown(mut self, cooldown: usize) -> Self {
        self.cooldown = cooldown;
        self
    }

    /// Set the minimum-decay threshold `eps`. A reduction is skipped
    /// when the relative change `(current_lr - new_lr) / current_lr`
    /// is less than `eps`.
    ///
    /// Mirrors `eps` at `torch/optim/lr_scheduler.py:1632`.
    #[must_use]
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Switch between relative (`ThresholdMode::Rel`, default) and
    /// absolute (`ThresholdMode::Abs`) improvement detection.
    ///
    /// Mirrors `threshold_mode` at
    /// `torch/optim/lr_scheduler.py:1626-1632`.
    #[must_use]
    pub fn with_threshold_mode(mut self, mode: ThresholdMode) -> Self {
        self.threshold_mode = mode;
        self
    }

    /// Configure per-parameter-group LR floors.
    ///
    /// When set, the `min_lr` floor used during reduction is taken
    /// from `per_group_min_lr[group_idx]` for each optimizer parameter
    /// group. The number of entries must match the optimizer's group
    /// count at step time; mismatch falls back to the scalar
    /// `min_lr` for that group.
    ///
    /// Mirrors `min_lr: List[float]` at
    /// `torch/optim/lr_scheduler.py:1684`.
    #[must_use]
    pub fn with_per_group_min_lr(mut self, per_group_min_lr: Vec<f64>) -> Self {
        self.per_group_min_lr = Some(per_group_min_lr);
        self
    }

    /// Return the current learning rate.
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Check whether the metric has improved relative to the best.
    fn is_better(&self, metric: f64) -> bool {
        match (self.mode, self.threshold_mode) {
            (PlateauMode::Min, ThresholdMode::Rel) => metric < self.best * (1.0 - self.threshold),
            (PlateauMode::Min, ThresholdMode::Abs) => metric < self.best - self.threshold,
            (PlateauMode::Max, ThresholdMode::Rel) => metric > self.best * (1.0 + self.threshold),
            (PlateauMode::Max, ThresholdMode::Abs) => metric > self.best + self.threshold,
        }
    }

    /// Resolve the LR floor for the given parameter group index.
    fn floor_for_group(&self, group_idx: usize) -> f64 {
        self.per_group_min_lr
            .as_ref()
            .and_then(|v| v.get(group_idx).copied())
            .unwrap_or(self.min_lr)
    }
}

/// Trait for schedulers that require a metric value each step.
///
/// This is separate from [`LrScheduler`](super::LrScheduler) because the
/// signature differs -- plateau schedulers need a metric to decide whether
/// to reduce the learning rate.
pub trait MetricScheduler<T: Float> {
    /// Perform one scheduler step with the given metric value.
    fn step(&mut self, optimizer: &mut dyn Optimizer<T>, metric: f64);

    /// Return the current learning rate.
    fn get_lr(&self) -> f64;
}

impl<T: Float> MetricScheduler<T> for ReduceLROnPlateau {
    fn step(&mut self, optimizer: &mut dyn Optimizer<T>, metric: f64) {
        // On first call, snapshot the optimizer's current LR.
        if !self.initialized {
            self.current_lr = optimizer.lr();
            self.initialized = true;
        }

        if self.is_better(metric) {
            self.best = metric;
            self.num_bad_steps = 0;
        } else if self.cooldown_counter > 0 {
            // In cooldown: the metric counts as "improved" for the
            // purposes of patience even when it would otherwise be a
            // bad step. Mirrors upstream's `in_cooldown` short-circuit
            // at `torch/optim/lr_scheduler.py:1718-1735` where the
            // bad-step counter is held at zero during cooldown.
            self.num_bad_steps = 0;
            self.cooldown_counter -= 1;
        } else {
            self.num_bad_steps += 1;
        }

        if self.num_bad_steps > self.patience {
            // Per-group reduction (#1476): walk each parameter group and
            // apply the per-group floor when set. Falls back to the
            // scalar `min_lr` floor for any group whose index is out of
            // range in `per_group_min_lr`.
            let num_groups = optimizer.param_groups().len();
            if num_groups > 1 || self.per_group_min_lr.is_some() {
                let mut any_lowered = false;
                let mut new_lrs: Vec<f64> = Vec::with_capacity(num_groups);
                for gi in 0..num_groups {
                    let cur = optimizer.param_groups()[gi].lr;
                    let floor = self.floor_for_group(gi);
                    let candidate = (cur * self.factor).max(floor);
                    // `eps` gate (#1476): skip the reduction when the
                    // relative change is smaller than `eps`, mirroring
                    // upstream's `if old_lr - new_lr > self.eps`
                    // check at `torch/optim/lr_scheduler.py:1748`.
                    if cur - candidate > self.eps {
                        new_lrs.push(candidate);
                        any_lowered = true;
                    } else {
                        new_lrs.push(cur);
                    }
                }
                if any_lowered {
                    // Apply per-group writeback.
                    let groups = optimizer.param_groups_mut();
                    for (gi, lr) in new_lrs.iter().enumerate() {
                        if let Some(group) = groups.get_mut(gi) {
                            group.lr = *lr;
                        }
                    }
                    // Track group 0's LR for `get_lr` reporting; the
                    // other groups are observable via
                    // `optimizer.param_groups()[gi].lr`.
                    self.current_lr = new_lrs[0];
                    self.cooldown_counter = self.cooldown;
                }
            } else {
                // Single-group fast path (preserves legacy behaviour).
                let floor = self.floor_for_group(0);
                let new_lr = (self.current_lr * self.factor).max(floor);
                if self.current_lr - new_lr > self.eps {
                    self.current_lr = new_lr;
                    optimizer.set_lr(new_lr);
                    self.cooldown_counter = self.cooldown;
                }
            }
            self.num_bad_steps = 0;
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
    fn test_plateau_no_reduction_when_improving() {
        let mut sched = ReduceLROnPlateau::new(PlateauMode::Min)
            .patience(3)
            .factor(0.5);
        let mut opt = MockOptimizer::new(0.1);

        // Steadily improving metric (decreasing for Min mode).
        for i in 0..10 {
            let metric = 1.0 - 0.1 * i as f64;
            <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, metric);
        }
        assert!(
            (opt.lr - 0.1).abs() < 1e-12,
            "LR should not change when improving; got {}",
            opt.lr
        );
    }

    #[test]
    fn test_plateau_reduces_after_patience() {
        let patience = 3;
        let mut sched = ReduceLROnPlateau::new(PlateauMode::Min)
            .patience(patience)
            .factor(0.5)
            .threshold(0.0);
        let mut opt = MockOptimizer::new(0.1);

        // Give it one good value, then plateau.
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
        assert!((opt.lr - 0.1).abs() < 1e-12);

        // patience + 1 steps of no improvement triggers reduction.
        for _ in 0..=patience {
            <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
        }
        assert!(
            (opt.lr - 0.05).abs() < 1e-12,
            "expected 0.05, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_plateau_respects_min_lr() {
        let mut sched = ReduceLROnPlateau::new(PlateauMode::Min)
            .patience(0)
            .factor(0.1)
            .min_lr(0.01)
            .threshold(0.0);
        let mut opt = MockOptimizer::new(0.1);

        // Each step with a non-improving metric should reduce LR, but not below min.
        for _ in 0..20 {
            <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 999.0);
        }
        assert!(
            opt.lr >= 0.01 - 1e-12,
            "LR should not go below min_lr; got {}",
            opt.lr
        );
    }

    #[test]
    fn test_plateau_max_mode() {
        let mut sched = ReduceLROnPlateau::new(PlateauMode::Max)
            .patience(2)
            .factor(0.5)
            .threshold(0.0);
        let mut opt = MockOptimizer::new(0.1);

        // Improving metric in max mode (increasing).
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 2.0);
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 3.0);
        assert!(
            (opt.lr - 0.1).abs() < 1e-12,
            "should not reduce when improving in max mode"
        );

        // Stagnant metric.
        for _ in 0..=2 {
            <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 3.0);
        }
        assert!(
            (opt.lr - 0.05).abs() < 1e-12,
            "expected 0.05 after plateau in max mode, got {}",
            opt.lr
        );
    }

    // -----------------------------------------------------------------------
    // #1476 — cooldown / eps / threshold_mode / per_group_min_lr
    // -----------------------------------------------------------------------

    /// `with_cooldown(N)` suppresses further reductions for N steps
    /// after each reduction.
    #[test]
    fn test_plateau_cooldown_suppresses_reduction() {
        let mut sched = ReduceLROnPlateau::new(PlateauMode::Min)
            .patience(0)
            .factor(0.5)
            .threshold(0.0)
            .with_cooldown(3);
        let mut opt = MockOptimizer::new(1.0);

        // First bad step triggers a reduction: 1.0 → 0.5.
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
        assert!(
            (opt.lr - 0.5).abs() < 1e-12,
            "first reduction; got {}",
            opt.lr
        );

        // Three cooldown steps: even with "bad" metrics, no reduction.
        for _ in 0..3 {
            <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
        }
        assert!(
            (opt.lr - 0.5).abs() < 1e-12,
            "during cooldown LR must stay at 0.5; got {}",
            opt.lr
        );

        // After cooldown, the next patience+1 bad steps trigger the
        // next reduction.
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
        assert!(
            (opt.lr - 0.25).abs() < 1e-12,
            "post-cooldown reduction; got {}",
            opt.lr
        );
    }

    /// `with_eps(eps)` skips the reduction when the relative change is
    /// below `eps`.
    #[test]
    fn test_plateau_eps_skips_tiny_reductions() {
        // current=1.0, factor=0.999 → candidate=0.999. With eps=0.01
        // the change 1e-3 is below eps, so the reduction is skipped.
        let mut sched = ReduceLROnPlateau::new(PlateauMode::Min)
            .patience(0)
            .factor(0.999)
            .threshold(0.0)
            .with_eps(0.01);
        let mut opt = MockOptimizer::new(1.0);

        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
        assert!(
            (opt.lr - 1.0).abs() < 1e-12,
            "eps must suppress reduction below threshold; got {}",
            opt.lr
        );
    }

    /// `with_threshold_mode(Abs)` switches to absolute improvement
    /// detection.
    #[test]
    fn test_plateau_threshold_mode_abs_min() {
        // Min mode + Abs threshold=0.1: improvement requires
        // metric < best - 0.1.
        let mut sched = ReduceLROnPlateau::new(PlateauMode::Min)
            .patience(2)
            .factor(0.5)
            .threshold(0.1)
            .with_threshold_mode(ThresholdMode::Abs);
        let mut opt = MockOptimizer::new(0.1);

        // Step 1: snapshot.
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
        // 0.95 < 1.0 - 0.1 = 0.9? No (0.95 > 0.9): bad.
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 0.95);
        // Another small "improvement" that doesn't pass abs(0.1).
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 0.93);
        // Third bad step crosses patience+1 → reduction.
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 0.92);
        assert!(
            (opt.lr - 0.05).abs() < 1e-12,
            "Abs threshold must NOT treat 0.05-step deltas as improvement; got {}",
            opt.lr
        );

        // A real Abs-passing improvement (drop of 0.5) resets the
        // counter without reducing further.
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 0.4);
        assert!(
            (opt.lr - 0.05).abs() < 1e-12,
            "Abs improvement must NOT reduce; got {}",
            opt.lr
        );
    }

    /// `with_per_group_min_lr` honors per-group floors.
    #[test]
    fn test_plateau_per_group_min_lr_uses_per_group_floor() {
        // Build a fake two-group optimizer via SGD with add_param_group.
        use crate::{Sgd, SgdConfig};
        use ferrotorch_nn::Parameter;

        let p1 = Parameter::<f32>::from_slice(&[1.0_f32], &[1]).unwrap();
        let p2 = Parameter::<f32>::from_slice(&[1.0_f32], &[1]).unwrap();
        let mut sgd = Sgd::new(vec![p1], SgdConfig::new(0.1));
        let g2 = crate::optimizer::ParamGroup::new(vec![p2], 0.1);
        sgd.add_param_group(g2);

        // Per-group floors: group 0 floored at 0.05; group 1 at 0.01.
        let mut sched = ReduceLROnPlateau::new(PlateauMode::Min)
            .patience(0)
            .factor(0.1)
            .threshold(0.0)
            .with_per_group_min_lr(vec![0.05, 0.01]);

        // First step seeds best with 1.0.
        sched.step(&mut sgd, 1.0);
        // patience+1 bad steps trigger reduction.
        sched.step(&mut sgd, 1.0);

        // candidate = max(0.1 * 0.1, floor) = max(0.01, floor).
        // Group 0 floor=0.05 ⇒ result 0.05.
        // Group 1 floor=0.01 ⇒ result 0.01.
        let g0_lr = sgd.param_groups()[0].lr;
        let g1_lr = sgd.param_groups()[1].lr;
        assert!(
            (g0_lr - 0.05).abs() < 1e-12,
            "group 0 LR should hit floor 0.05; got {g0_lr}"
        );
        assert!(
            (g1_lr - 0.01).abs() < 1e-12,
            "group 1 LR should hit floor 0.01; got {g1_lr}"
        );
    }

    #[test]
    fn test_plateau_resets_bad_count_on_improvement() {
        let mut sched = ReduceLROnPlateau::new(PlateauMode::Min)
            .patience(3)
            .factor(0.5)
            .threshold(0.0);
        let mut opt = MockOptimizer::new(0.1);

        // Start with a good value.
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);

        // 2 bad steps (below patience).
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);

        // Improvement resets the counter.
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 0.5);

        // 3 more bad steps -- should NOT trigger because counter was reset.
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 0.5);
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 0.5);
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 0.5);

        assert!(
            (opt.lr - 0.1).abs() < 1e-12,
            "LR should not have been reduced; got {}",
            opt.lr
        );
    }
}
