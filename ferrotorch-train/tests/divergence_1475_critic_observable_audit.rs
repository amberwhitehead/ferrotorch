//! Discriminator probe for #1475 — `Learner::with_metric_scheduler`
//! drives the metric scheduler per epoch with the up-to-date loss.
//!
//! This probe is the critic's red-team against a "vocab-only" landing
//! where the `MetricScheduler` trait + `Learner::with_metric_scheduler`
//! builder exist but `Learner::fit` never calls `metric_sched.step(...)`.
//!
//! Reference: PyTorch's
//! `ReduceLROnPlateau.step(metric)` recipe at
//! `pytorch/torch/optim/lr_scheduler.py:1695-1742`; the Learner's
//! upstream-equivalent driver site is `Learner::fit` in
//! `ferrotorch-train/src/learner.rs` (the per-epoch
//! `metric_sched.step(optimizer, val_loss.unwrap_or(train_loss))`).

use ferrotorch_core::{FerrotorchResult, Tensor};
use ferrotorch_nn::{Module, Parameter};
use ferrotorch_optim::scheduler::{PlateauMode, ReduceLROnPlateau};
use ferrotorch_optim::{Optimizer, OptimizerState, ParamGroup};
use ferrotorch_train::{Learner, LossFn};

// ---------------------------------------------------------------------------
// MockOpt + identity Module fixture
//
// Both fixtures explicitly panic in `add_param_group` to surface miswritten
// tests (mirrors `ferrotorch-optim/src/swa.rs::tests::MockOptimizer`).
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
    fn state_dict(&self) -> FerrotorchResult<OptimizerState> {
        Ok(Default::default())
    }
    fn load_state_dict(&mut self, _s: &OptimizerState) -> FerrotorchResult<()> {
        Ok(())
    }
}

struct M {
    w: Parameter<f32>,
    training: bool,
}

impl Module<f32> for M {
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

/// Reproduces the audit prompt's protocol for #1475: descending loss
/// across an epoch must NOT trigger reduction; stable loss across an
/// epoch must. The loss closure returns `pred`, so the per-batch loss
/// equals the input scalar — we drive the trajectory by varying the
/// input across epochs.
///
/// A "vocab-only" landing (Learner stores the metric scheduler but
/// `fit` never calls its `step`) would leave `EpochResult::lr` at the
/// constructor's 1.0 throughout, failing this probe.
#[test]
fn critic_1475_learner_fit_drives_metric_scheduler_per_epoch_observable() {
    // --- Phase 1: 3 epochs of descending loss (1.0, 0.5, 0.25) ---
    let model = M {
        w: Parameter::from_slice(&[0.0_f32], &[1]).unwrap(),
        training: true,
    };
    let optimizer: Box<dyn Optimizer<f32>> = Box::new(MockOpt { lr: 1.0 });
    let loss_fn: LossFn<f32> = Box::new(|pred, _target| Ok(pred.clone()));

    let plateau = ReduceLROnPlateau::new(PlateauMode::Min)
        .patience(0)
        .factor(0.5)
        .threshold(0.0);

    let mut learner =
        Learner::new(model, optimizer, loss_fn).with_metric_scheduler(Box::new(plateau));

    // Per-epoch data: one batch whose input scalar dictates the loss.
    // We use a shared cell to advance through the schedule across
    // successive `fit(.., n_epochs=1)` calls.
    let trajectory = std::cell::RefCell::new(vec![1.0_f32, 0.5, 0.25]);
    let next_batch = || {
        let v = trajectory.borrow_mut().remove(0);
        std::iter::once(Ok((
            ferrotorch_core::scalar(v).unwrap(),
            ferrotorch_core::scalar(0.0_f32).unwrap(),
        )))
        .collect::<Vec<_>>()
        .into_iter()
    };

    // Drive 3 descending epochs.
    let hist1 = learner
        .fit(
            &next_batch,
            None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>,
            3,
        )
        .expect("descending fit");

    // The LR captured at each EpochResult is the post-step LR.
    //
    // Epoch 0: best=1.0 (snapshot), num_bad=0. No reduction. LR=1.0.
    // Epoch 1: metric=0.5 < 1.0*(1-0)=1.0 (Rel threshold), is_better=
    //          true ⇒ num_bad=0, no reduction. LR=1.0.
    // Epoch 2: metric=0.25 < best=0.5, is_better=true ⇒ no reduction.
    //          LR=1.0.
    for (i, ep) in hist1.epochs.iter().enumerate() {
        assert!(
            (ep.lr - 1.0).abs() < 1e-12,
            "descending epoch {i}: LR must remain 1.0; got {}",
            ep.lr
        );
    }

    // --- Phase 2: 3 epochs of stable loss (0.25, 0.25, 0.25) ---
    let stable = std::cell::RefCell::new(vec![0.25_f32, 0.25, 0.25]);
    let next_stable = || {
        let v = stable.borrow_mut().remove(0);
        std::iter::once(Ok((
            ferrotorch_core::scalar(v).unwrap(),
            ferrotorch_core::scalar(0.0_f32).unwrap(),
        )))
        .collect::<Vec<_>>()
        .into_iter()
    };
    let hist2 = learner
        .fit(
            &next_stable,
            None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>,
            3,
        )
        .expect("stable fit");

    // Epoch 0 of stable: metric=0.25, best=0.25. is_better requires
    // metric < best*(1-0)=best, so 0.25 < 0.25 is false ⇒ num_bad=1,
    // crosses patience=0, reduction fires 1.0 → 0.5.
    // Epoch 1 stable: 0.25 < 0.25 false ⇒ reduction 0.5 → 0.25.
    // Epoch 2 stable: reduction 0.25 → 0.125.
    let final_lr = hist2.epochs.last().unwrap().lr;
    assert!(
        final_lr < 1.0,
        "stable loss must trigger plateau reduction; got final LR={final_lr}"
    );
    assert!(
        final_lr <= 0.125 + 1e-12,
        "3 stable epochs must trigger at least 3 halvings: 1.0→0.5→0.25→0.125; got {final_lr}"
    );

    // Sabotage probe: a Learner that stored the metric scheduler but
    // never called step would leave every LR at 1.0.
    assert!(
        hist2.epochs.iter().any(|ep| ep.lr != 1.0),
        "metric_scheduler must have fired at least once in the stable phase; \
         all LRs still 1.0 in {:?}",
        hist2.epochs.iter().map(|e| e.lr).collect::<Vec<_>>()
    );
}
