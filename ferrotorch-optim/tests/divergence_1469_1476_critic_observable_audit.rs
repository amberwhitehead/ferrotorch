//! Discriminator probes for #1469, #1470, #1474, #1476 — observable
//! behavior, not vocab. (The #1475 Learner-side probe lives in
//! `ferrotorch-train/tests/divergence_1475_critic_observable_audit.rs`
//! because it requires the `Learner` type which is in the
//! `ferrotorch-train` crate.)
//!
//! These probes are the critic's red-team against "vocab-only" landings:
//! each probe constructs a state where a no-op or stub implementation
//! would silently pass the in-source unit tests (e.g. a builder that
//! mutates a field but `step()` ignores it), then asserts the
//! end-to-end observable that only a genuinely-wired implementation can
//! satisfy.
//!
//! References:
//! * #1469 — LBFGS per-group LR. Upstream:
//!   `pytorch/torch/optim/lbfgs.py:288-310` iterates `self.param_groups`
//!   and reads `group["lr"]` per group.
//! * #1470 — SWA `AveragedModel` from `&dyn Module<T>`. Upstream:
//!   `pytorch/torch/optim/swa_utils.py:165-240`.
//! * #1474 — `OneCycleLR::with_cycle_momentum` drives `set_momentum`.
//!   Upstream: `pytorch/torch/optim/lr_scheduler.py:2342-2350,
//!   2391-2453`.
//! * #1476 — `ReduceLROnPlateau` extra builders consumed in `step`.
//!   Upstream: `pytorch/torch/optim/lr_scheduler.py:1625-1684, 1748`.

use ferrotorch_core::{FerrotorchResult, Tensor};
use ferrotorch_nn::Parameter;
use ferrotorch_optim::scheduler::{
    AnnealStrategy, LrScheduler, MetricScheduler, OneCycleLR, PlateauMode, ReduceLROnPlateau,
    ThresholdMode,
};
use ferrotorch_optim::{Lbfgs, LbfgsConfig, Optimizer, ParamGroup, Sgd, SgdConfig};

// ---------------------------------------------------------------------------
// Mock optimizer used by the probes that need a settable LR but no real
// parameter machinery.
//
// `add_param_group` panics — this mock holds no real groups, so any test
// that tries to add one has been miswritten. Mirrors the precedent in
// `ferrotorch-optim/src/swa.rs::tests::MockOptimizer` which panics for
// the same reason. (Empty `{}` bodies are not OK here: they would
// silently swallow a real-group attach and let downstream assertions
// drift onto the wrong path.)
// ---------------------------------------------------------------------------

struct MockOpt {
    lr: f64,
}

impl Optimizer<f32> for MockOpt {
    fn step(&mut self) -> FerrotorchResult<()> {
        Ok(())
    }
    fn zero_grad(&mut self) -> FerrotorchResult<()> {
        Ok(())
    }
    fn lr(&self) -> f64 {
        self.lr
    }
    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
    fn param_groups(&self) -> &[ParamGroup<f32>] {
        &[]
    }
    fn param_groups_mut(&mut self) -> &mut [ParamGroup<f32>] {
        &mut []
    }
    fn add_param_group(&mut self, _g: ParamGroup<f32>) {
        panic!(
            "MockOpt::add_param_group called — this mock holds no real param \
             groups; the calling test is miswritten"
        );
    }
    fn state_dict(&self) -> FerrotorchResult<ferrotorch_optim::OptimizerState> {
        Ok(Default::default())
    }
    fn load_state_dict(
        &mut self,
        _s: &ferrotorch_optim::OptimizerState,
    ) -> FerrotorchResult<()> {
        Ok(())
    }
}

fn scalar_param(v: f32) -> Parameter<f32> {
    Parameter::from_slice(&[v], &[1]).unwrap()
}

// ===========================================================================
// #1469 — LBFGS per-group LR ratio assertion
// ===========================================================================

/// Two L-BFGS param groups at lr=0.1 and lr=0.01. On the very first step
/// (no curvature history) the search direction equals `-grad`, so each
/// group's delta is `lr * (-grad)`. With identical starting points the
/// ratio of |delta_0| / |delta_1| must equal 10. A regression to
/// "global lr from first group" would yield ratio 1; a regression to
/// "global lr from second group" would yield 1 or 0.1.
#[test]
fn critic_1469_lbfgs_per_group_lr_ratio_is_ten() {
    let p1 = Parameter::<f64>::from_slice(&[5.0_f64], &[1]).unwrap();
    let p2 = Parameter::<f64>::from_slice(&[5.0_f64], &[1]).unwrap();

    // LbfgsConfig is #[non_exhaustive]; use the builder chain.
    let cfg = LbfgsConfig::default().with_lr(0.1);
    let mut opt = Lbfgs::new(vec![p1], cfg);
    let g2 = ParamGroup::new(vec![p2], 0.01);
    opt.add_param_group(g2);

    // f = p1^2 + p2^2 ⇒ grad(pi) = 2*pi = 10 at the starting point.
    opt.zero_grad().unwrap();
    let x1 = opt.param_groups()[0].params()[0].tensor().clone();
    let x2 = opt.param_groups()[1].params()[0].tensor().clone();
    let s1 = ferrotorch_core::grad_fns::arithmetic::pow(&x1, 2.0).unwrap();
    let s2 = ferrotorch_core::grad_fns::arithmetic::pow(&x2, 2.0).unwrap();
    let loss = ferrotorch_core::grad_fns::arithmetic::add(&s1, &s2).unwrap();
    loss.backward().unwrap();
    opt.step().unwrap();

    let v1 = opt.param_groups()[0].params()[0].tensor().data().unwrap()[0];
    let v2 = opt.param_groups()[1].params()[0].tensor().data().unwrap()[0];

    // grad=10, direction=-10. delta = lr * (-10). So:
    //   v1 = 5 + 0.1 * (-10) = 4.0
    //   v2 = 5 + 0.01 * (-10) = 4.9
    let d1 = (5.0_f64 - v1).abs();
    let d2 = (5.0_f64 - v2).abs();
    assert!(
        (d1 - 1.0).abs() < 1e-9,
        "expected group-0 delta=1.0 (lr=0.1 * |grad|=10); got d1={d1} (v1={v1})"
    );
    assert!(
        (d2 - 0.1).abs() < 1e-9,
        "expected group-1 delta=0.1 (lr=0.01 * |grad|=10); got d2={d2} (v2={v2})"
    );
    // Ratio must be 10×.
    assert!(
        (d1 / d2 - 10.0).abs() < 1e-9,
        "expected |delta_0|/|delta_1| == 10; got {}",
        d1 / d2
    );
}

// ===========================================================================
// #1470 — AveragedModel from &dyn Module<T> drives correct running mean
// ===========================================================================

/// `AveragedModel::from_module(&dyn Module<T>, ...)` plus
/// `update_parameters_from_module` after 3 distinct weight updates must
/// produce an averaged value equal to the SWA running mean of all 3
/// snapshots — the canonical SWA correctness check.
#[test]
fn critic_1470_swa_from_dyn_module_running_mean_matches_arithmetic_mean() {
    use ferrotorch_nn::Module;
    use ferrotorch_optim::{AveragedModel, AveragingStrategy};

    struct Tiny {
        w: Parameter<f32>,
        training: bool,
    }
    impl Module<f32> for Tiny {
        fn forward(&self, x: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
            Ok(x.clone())
        }
        fn parameters(&self) -> Vec<&Parameter<f32>> {
            vec![&self.w]
        }
        fn parameters_mut(&mut self) -> Vec<&mut Parameter<f32>> {
            vec![&mut self.w]
        }
        fn named_parameters(&self) -> Vec<(String, &Parameter<f32>)> {
            vec![("w".to_string(), &self.w)]
        }
        fn train(&mut self) {
            self.training = true;
        }
        fn eval(&mut self) {
            self.training = false;
        }
        fn is_training(&self) -> bool {
            self.training
        }
    }

    let module = Tiny {
        w: Parameter::from_slice(&[1.0_f32], &[1]).unwrap(),
        training: true,
    };

    // Pass the &dyn Module<T> at construction time (#1470).
    let dyn_module: &dyn Module<f32> = &module;
    let mut avg = AveragedModel::from_module(dyn_module, AveragingStrategy::Swa);

    // Three snapshots: 2.0, 4.0, 6.0. SWA running mean ends at 4.0.
    for v in [2.0_f32, 4.0, 6.0] {
        ferrotorch_core::no_grad(|| unsafe { module.w.tensor().update_data(&[v]) }).unwrap();
        avg.update_parameters_from_module(&module).unwrap();
    }
    let m = avg.averaged_values(0).unwrap()[0];
    // SWA running mean: 2.0 (first copy), then 2+(4-2)/2=3, then 3+(6-3)/3=4.
    assert!((m - 4.0).abs() < 1e-5, "expected SWA mean=4.0, got {m}");

    // apply_to_module writes the averaged value into the module weight.
    avg.apply_to_module(&module).unwrap();
    let w_now = module.w.data().unwrap()[0];
    assert!((w_now - 4.0).abs() < 1e-5, "expected w=4.0, got {w_now}");
}

// ===========================================================================
// #1474 — OneCycleLR with_cycle_momentum drives SGD's momentum coefficient
// ===========================================================================

/// Drive 100 OneCycleLR steps with `with_cycle_momentum(0.85, 0.95)` and
/// log the sequence of momentum values pulled back via
/// `sgd.momentum(0)`. The min and max observed values must straddle
/// the configured (base, max) boundaries — a no-op `set_momentum`
/// would leave SGD's momentum at its constructor value 0.5.
#[test]
fn critic_1474_one_cycle_momentum_observed_range_straddles_boundaries() {
    let p = scalar_param(1.0);
    let mut sgd = Sgd::new(vec![p], SgdConfig::new(0.001).with_momentum(0.5));

    let mut sched = OneCycleLR::new(0.01, 100, 0.3, AnnealStrategy::Cos, 25.0, 1e4, false)
        .with_cycle_momentum(0.85, 0.95);

    let mut observed: Vec<f64> = Vec::with_capacity(100);
    for _ in 0..100 {
        sched.step(&mut sgd);
        observed.push(sgd.momentum(0).expect("read SGD momentum"));
    }
    let min_obs = observed.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_obs = observed.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // The sweep must hit base_momentum=0.85 at the LR peak and
    // max_momentum=0.95 at the valley (within cosine interpolation
    // tolerance).
    assert!(
        (min_obs - 0.85).abs() < 1e-3,
        "min observed momentum should be ~0.85; got {min_obs}"
    );
    assert!(
        (max_obs - 0.95).abs() < 1e-3,
        "max observed momentum should be ~0.95; got {max_obs}"
    );
    // Sabotage probe: a `let _ = optimizer.set_momentum(..)` that
    // silently swallowed the call but never wrote through would leave
    // every observed value at the SGD constructor's 0.5.
    assert!(
        observed.iter().all(|&m| m != 0.5),
        "no observed momentum should equal the SGD constructor default 0.5; \
         got at least one 0.5 entry in {observed:?}"
    );
}

// ===========================================================================
// #1476 — Plateau extra builders are CONSUMED in step (not just stored)
// ===========================================================================

/// Sabotage probe for `with_cooldown`: if `cooldown` were stored but
/// never decremented in `step`, the second reduction would fire
/// immediately on the next bad epoch. We verify that across `cooldown`
/// post-reduction bad steps the LR is held constant.
#[test]
fn critic_1476_cooldown_holds_lr_during_cooldown_window() {
    let mut sched = ReduceLROnPlateau::new(PlateauMode::Min)
        .patience(0)
        .factor(0.5)
        .threshold(0.0)
        .with_cooldown(4);
    let mut opt = MockOpt { lr: 1.0 };

    // snapshot epoch.
    <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
    // Bad step 1 → reduction 1.0 → 0.5.
    <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
    assert!(
        (opt.lr - 0.5).abs() < 1e-12,
        "first reduction must drop lr to 0.5; got {}",
        opt.lr
    );

    // 4 cooldown steps: lr held at 0.5.
    for i in 0..4 {
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
        assert!(
            (opt.lr - 0.5).abs() < 1e-12,
            "during cooldown (step {i}) lr must stay at 0.5; got {}",
            opt.lr
        );
    }

    // Post-cooldown: next bad step reduces again.
    <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
    assert!(
        (opt.lr - 0.25).abs() < 1e-12,
        "post-cooldown reduction must drop lr to 0.25; got {}",
        opt.lr
    );
}

/// Sabotage probe for `with_threshold_mode(Abs)`: a builder that
/// flipped the field but `is_better` still used Rel would treat tiny
/// `0.99 * best`-style improvements as real improvements and never
/// reduce. With Abs and threshold=0.5 a drop from 1.0 to 0.7 is NOT
/// an improvement (delta < threshold), so reduction must fire.
#[test]
fn critic_1476_threshold_mode_abs_rejects_sub_threshold_improvement() {
    // Min mode + Abs threshold=0.5: an improvement requires
    // metric < best - 0.5.
    let mut sched = ReduceLROnPlateau::new(PlateauMode::Min)
        .patience(0)
        .factor(0.5)
        .threshold(0.5)
        .with_threshold_mode(ThresholdMode::Abs);
    let mut opt = MockOpt { lr: 1.0 };

    // snapshot epoch with metric=1.0, best=1.0.
    <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
    // metric=0.7: is 0.7 < 1.0 - 0.5 = 0.5? No (0.7 > 0.5): NOT
    // an improvement under Abs. With patience=0 this fires reduction.
    <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 0.7);
    assert!(
        (opt.lr - 0.5).abs() < 1e-12,
        "Abs threshold=0.5 must reject 0.3-step improvement and reduce; got {}",
        opt.lr
    );

    // A real Abs-passing improvement (drop of 0.6: 0.7 → 0.1) IS an
    // improvement: 0.1 < 0.7 - 0.5 = 0.2 yes.
    <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 0.1);
    assert!(
        (opt.lr - 0.5).abs() < 1e-12,
        "real Abs improvement must NOT trigger further reduction; got {}",
        opt.lr
    );
}

/// Sabotage probe for `with_eps`: with eps=0.5 and factor=0.6, the
/// candidate new_lr = 1.0 * 0.6 = 0.6, change = 0.4 which is <
/// eps=0.5, so reduction must be SKIPPED. A builder that ignored eps
/// would still write through to 0.6.
#[test]
fn critic_1476_eps_suppresses_below_threshold_reductions() {
    let mut sched = ReduceLROnPlateau::new(PlateauMode::Min)
        .patience(0)
        .factor(0.6)
        .threshold(0.0)
        .with_eps(0.5);
    let mut opt = MockOpt { lr: 1.0 };

    <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
    <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
    // candidate=0.6, cur-candidate=0.4 < eps=0.5 → skip.
    assert!(
        (opt.lr - 1.0).abs() < 1e-12,
        "eps=0.5 must suppress 0.4 reduction; got {}",
        opt.lr
    );
}

/// Sabotage probe for `with_per_group_min_lr`: an unfloored reduction
/// would let group 0's LR drop below the per-group floor.
#[test]
fn critic_1476_per_group_min_lr_floors_each_group_distinctly() {
    let p1 = scalar_param(1.0);
    let p2 = scalar_param(1.0);
    let mut sgd = Sgd::new(vec![p1], SgdConfig::new(1.0));
    let g2 = ParamGroup::new(vec![p2], 1.0);
    sgd.add_param_group(g2);

    // Floors: group 0 hard-floored at 0.5; group 1 at 0.01.
    let mut sched = ReduceLROnPlateau::new(PlateauMode::Min)
        .patience(0)
        .factor(0.1)
        .threshold(0.0)
        .with_per_group_min_lr(vec![0.5, 0.01]);

    <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut sgd, 1.0);
    <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut sgd, 1.0);

    let g0_lr = sgd.param_groups()[0].lr;
    let g1_lr = sgd.param_groups()[1].lr;
    // candidate = max(1.0 * 0.1, floor) = max(0.1, floor).
    // Group 0 floor=0.5 → 0.5; group 1 floor=0.01 → 0.1.
    assert!(
        (g0_lr - 0.5).abs() < 1e-12,
        "group 0 LR must be floored at 0.5; got {g0_lr}"
    );
    assert!(
        (g1_lr - 0.1).abs() < 1e-12,
        "group 1 should NOT have hit its floor 0.01 yet (candidate=0.1 > floor); got {g1_lr}"
    );
}
