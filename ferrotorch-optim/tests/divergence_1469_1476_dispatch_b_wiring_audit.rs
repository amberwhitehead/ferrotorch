//! Audit probes for Dispatch B (#1469-#1476) — REQ cluster on
//! `ferrotorch-optim` schedulers + optimizer trait extension.
//!
//! Each test asserts the observable behavior claimed in the dispatch
//! goal — set_momentum mutation, CyclicLR cycle_momentum drive, custom
//! scale_fn override, etc. A passing test confirms genuine wiring;
//! a failing test points at vocab-only / no-op stubs.
//!
//! Untouched blockers (#1469 lbfgs, #1470 swa &dyn Module, #1471 nadam/radam
//! with_foreach builder, #1474 one_cycle_lr cycle_momentum, #1475 plateau
//! with_metric_scheduler, #1476 plateau cooldown/eps/threshold_mode) get
//! documentation probes that explicitly call out the missing symbols.
//!
//! Active tracking: #1542 (audit dispatch).

use std::sync::Arc;

use ferrotorch_nn::Parameter;
use ferrotorch_optim::scheduler::cyclic_lr::ScaleFn;
use ferrotorch_optim::scheduler::{CyclicLR, CyclicMode, LrScheduler};
use ferrotorch_optim::{Optimizer, Sgd, SgdConfig};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

fn sgd_fixture(lr: f64, momentum: f64) -> Sgd<f32> {
    let p = Parameter::<f32>::from_slice(&[1.0_f32, 2.0_f32], &[2]).unwrap();
    let cfg = SgdConfig::new(lr).with_momentum(momentum);
    Sgd::new(vec![p], cfg)
}

// ---------------------------------------------------------------------------
// #1472 — CyclicLR cycle_momentum drives SGD momentum field
// ---------------------------------------------------------------------------

/// Probe: `with_cycle_momentum(base=0.8, max=0.95)` on a Triangular
/// cyclic schedule must mutate SGD's `momentum` field via
/// `set_momentum`. At the LR peak (step = step_size_up), momentum must
/// be at `base_momentum` (LR-momentum inverse coupling).
#[test]
fn audit_1472_cyclic_lr_cycle_momentum_writes_through_to_sgd() {
    let mut sgd = sgd_fixture(0.001, 0.95);
    assert!(
        (sgd.momentum(0).unwrap() - 0.95).abs() < 1e-12,
        "initial momentum must be 0.95"
    );

    let mut sched = CyclicLR::new(0.001, 0.01, 10, None, CyclicMode::Triangular, 1.0)
        .with_cycle_momentum(0.85, 0.95);

    // Step 10 times → LR at peak; momentum at base_momentum (0.85).
    for _ in 0..10 {
        sched.step(&mut sgd);
    }

    let m = sgd.momentum(0).expect("read momentum");
    assert!(
        (m - 0.85).abs() < 1e-6,
        "at LR peak, momentum must be base_momentum=0.85; got {m}. \
         If still 0.95, CyclicLR::step is not invoking optimizer.set_momentum \
         (or SGD's set_momentum override is not writing through)."
    );
}

/// Probe: at the LR valley (after a full cycle), momentum must return
/// to `max_momentum`. The LR-momentum inverse coupling is a 1-1 map.
#[test]
fn audit_1472_cyclic_lr_cycle_momentum_returns_to_max_at_valley() {
    let mut sgd = sgd_fixture(0.001, 0.95);
    let mut sched = CyclicLR::new(0.001, 0.01, 10, None, CyclicMode::Triangular, 1.0)
        .with_cycle_momentum(0.85, 0.95);

    for _ in 0..20 {
        sched.step(&mut sgd);
    }

    let m = sgd.momentum(0).expect("read");
    assert!(
        (m - 0.95).abs() < 1e-6,
        "at LR valley, momentum must be max_momentum=0.95; got {m}"
    );
}

/// Probe: `get_momentum` accessor on `CyclicLR` must track the
/// computed value, not stay at the constructor default.
#[test]
fn audit_1472_cyclic_lr_get_momentum_accessor_reflects_compute() {
    let mut sgd = sgd_fixture(0.001, 0.95);
    let mut sched = CyclicLR::new(0.001, 0.01, 10, None, CyclicMode::Triangular, 1.0)
        .with_cycle_momentum(0.80, 0.99);

    // At step 5 (halfway up): amplitude = 0.5, momentum = 0.99 - 0.19*0.5 = 0.895.
    for _ in 0..5 {
        sched.step(&mut sgd);
    }

    let expected = 0.99 - (0.99 - 0.80) * 0.5;
    let got = sched.get_momentum();
    assert!(
        (got - expected).abs() < 1e-9,
        "get_momentum() should reflect the computed value; expected {expected}, got {got}"
    );
}

// ---------------------------------------------------------------------------
// #1473 — CyclicLR with_scale_fn user closure overrides amplitude formula
// ---------------------------------------------------------------------------

/// Probe: a user `scale_fn` returning a constant 0.5 must give an
/// amplitude that's half of the built-in Triangular shape — and crucially
/// must be SEEN by `compute_lr`. If `with_scale_fn` is vocab-only and
/// the closure is never invoked, the result equals the default
/// Triangular formula.
#[test]
fn audit_1473_cyclic_lr_with_scale_fn_overrides_triangular_amplitude() {
    let mut sgd = sgd_fixture(0.001, 0.0);

    // Builtin Triangular at step 10 (peak): LR = max_lr = 0.01.
    let mut sched_default =
        CyclicLR::new(0.001, 0.01, 10, None, CyclicMode::Triangular, 1.0);
    for _ in 0..10 {
        sched_default.step(&mut sgd);
    }
    let lr_default = sched_default.get_lr();

    // With scale_fn(cycle) = 0.5: amplitude = triangular_scale * 0.5 = 0.5
    // → LR = base + 0.5 * (max - base) = 0.001 + 0.0045 = 0.0055.
    let mut sgd2 = sgd_fixture(0.001, 0.0);
    let custom: ScaleFn = Arc::new(|_cycle: f64| 0.5_f64);
    let mut sched_custom =
        CyclicLR::new(0.001, 0.01, 10, None, CyclicMode::Triangular, 1.0)
            .with_scale_fn(custom);
    for _ in 0..10 {
        sched_custom.step(&mut sgd2);
    }
    let lr_custom = sched_custom.get_lr();

    assert!(
        (lr_custom - 0.0055).abs() < 1e-9,
        "with scale_fn returning 0.5, LR at peak must be 0.0055; got {lr_custom}"
    );
    assert!(
        (lr_default - 0.01).abs() < 1e-9,
        "default Triangular LR at peak must be 0.01; got {lr_default}"
    );
    assert!(
        (lr_default - lr_custom).abs() > 1e-3,
        "scale_fn must produce a materially different LR than default ({lr_default} vs {lr_custom}); \
         if equal, the closure was never invoked — vocab-only wiring."
    );
}

/// Probe: the user `scale_fn` is invoked at least once per step.
/// We pin this by setting the closure to mutate an atomic counter.
#[test]
fn audit_1473_cyclic_lr_scale_fn_closure_actually_invoked() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let count: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
    let c = count.clone();
    let probe: ScaleFn = Arc::new(move |_cycle: f64| {
        c.fetch_add(1, Ordering::SeqCst);
        1.0_f64
    });

    let mut sched =
        CyclicLR::new(0.0, 1.0, 5, None, CyclicMode::Triangular, 1.0).with_scale_fn(probe);
    let mut sgd = sgd_fixture(0.0, 0.0);
    for _ in 0..7 {
        sched.step(&mut sgd);
    }

    assert!(
        count.load(Ordering::SeqCst) >= 7,
        "scale_fn must be invoked once per step; saw {} calls in 7 steps",
        count.load(Ordering::SeqCst)
    );
}

// ---------------------------------------------------------------------------
// #1472/#1473 — Optimizer::set_momentum trait extension on SGD
// ---------------------------------------------------------------------------

/// Probe: SGD's `set_momentum` override must write `config.momentum`
/// (read back via `momentum()`) — the smoke for the trait extension.
#[test]
fn audit_set_momentum_writes_through_on_sgd() {
    let mut sgd = sgd_fixture(0.001, 0.5);
    assert!((sgd.momentum(0).unwrap() - 0.5).abs() < 1e-12);
    sgd.set_momentum(0, 0.9).expect("set_momentum");
    assert!(
        (sgd.momentum(0).unwrap() - 0.9).abs() < 1e-12,
        "set_momentum(0.9) must update SGD config; if get returns 0.5, the override is vocab-only"
    );
}

/// Probe: SGD's `set_momentum` rejects out-of-range group_idx.
#[test]
fn audit_set_momentum_rejects_out_of_range_group_idx() {
    let mut sgd = sgd_fixture(0.001, 0.5);
    let r = sgd.set_momentum(99, 0.9);
    assert!(r.is_err(), "out-of-range group_idx must Err; got {r:?}");
}

// ---------------------------------------------------------------------------
// #1469 — lbfgs per-group lr + uniform step API (UNTOUCHED in dispatch)
// ---------------------------------------------------------------------------

/// Documentation probe: the dispatch goal for #1469 was a uniform
/// `step()` API on Lbfgs that accepts no closure (matching every other
/// Optimizer impl). The dispatch did NOT modify `lbfgs.rs`; the
/// existing `Optimizer<T>::step` impl on Lbfgs returns `InvalidArgument`
/// when `line_search_fn.is_some()` (per `lbfgs.rs` REQ-8). No new
/// builder/API surface was added.
///
/// Verifies that the divergence is observable: the no-closure `step`
/// on Lbfgs cannot be used with `line_search_fn=Some(StrongWolfe)`.
#[test]
#[ignore = "vocab-only on dispatch B; tracking #1469 — uniform Optimizer::step on Lbfgs requires upstream-style step_with_closure or a closure-less re-entry path"]
fn audit_1469_lbfgs_uniform_step_api_not_implemented() {
    // Intentionally ignored to document the unmet dispatch goal.
    unreachable!("dispatch goal #1469 was not addressed in dispatch B; lbfgs.rs unchanged")
}

// ---------------------------------------------------------------------------
// #1470 — swa AveragedModel &dyn Module<T> (UNTOUCHED in dispatch)
// ---------------------------------------------------------------------------

/// Documentation probe: #1470's goal was to accept `&dyn Module<T>` in
/// `AveragedModel::new` rather than requiring `&[Parameter<T>]`. The
/// dispatch did NOT modify `swa.rs`.
#[test]
#[ignore = "vocab-only on dispatch B; tracking #1470 — AveragedModel::new still takes &[Parameter<T>], not &dyn Module<T>"]
fn audit_1470_swa_dyn_module_constructor_not_implemented() {
    unreachable!("dispatch goal #1470 was not addressed in dispatch B; swa.rs unchanged")
}

// ---------------------------------------------------------------------------
// #1471 — nadam/radam with_foreach builder (UNTOUCHED in dispatch)
// ---------------------------------------------------------------------------

/// Documentation probe: `with_foreach` already exists on both
/// `NAdamConfig` and `RAdamConfig` (per pre-dispatch state). The
/// dispatch did NOT touch these files. The REQ tables in
/// `nadam.rs:23` and `radam.rs:24` already say "SHIPPED" with
/// "partial-parity divergence tracked by #1471" — meaning #1471 is
/// the partial-parity bug for `any_cuda` auto-routing, not a
/// missing-builder bug.
///
/// Confirms the builder is callable today.
#[test]
fn audit_1471_nadam_with_foreach_builder_exists() {
    use ferrotorch_optim::{NAdam, NAdamConfig};

    let p = Parameter::<f32>::from_slice(&[1.0_f32], &[1]).unwrap();
    let cfg = NAdamConfig::default().with_foreach(true);
    let _opt = NAdam::new(vec![p], cfg);
    // No assertion — the dispatch goal here is the builder existing,
    // which it does. The OPEN work tracked by #1471 is partial-parity
    // on auto-route; orthogonal to a vocab-only check.
}

// ---------------------------------------------------------------------------
// #1474 — one_cycle_lr cycle_momentum (UNTOUCHED in dispatch)
// ---------------------------------------------------------------------------

/// Probe: `OneCycleLR` does NOT have a `with_cycle_momentum` builder.
/// The dispatch did NOT modify `one_cycle_lr.rs`. The cyclic_lr.rs
/// path got the builder, but OneCycleLR is its own scheduler in
/// `scheduler/one_cycle_lr.rs` and the REQ table at REQ-6 still says
/// "NOT-STARTED, blocker #1474".
#[test]
#[ignore = "vocab-only on dispatch B; tracking #1474 — OneCycleLR has no with_cycle_momentum builder"]
fn audit_1474_one_cycle_lr_cycle_momentum_not_implemented() {
    unreachable!("dispatch goal #1474 was not addressed in dispatch B; one_cycle_lr.rs unchanged")
}

// ---------------------------------------------------------------------------
// #1475 — plateau with_metric_scheduler in Learner (UNTOUCHED in dispatch)
// ---------------------------------------------------------------------------

/// Documentation probe: `Learner::with_metric_scheduler` does NOT
/// exist. The dispatch did NOT modify `plateau.rs` or `learner.rs` to
/// add a metric-aware scheduler attachment point.
#[test]
#[ignore = "vocab-only on dispatch B; tracking #1475 — Learner has no with_metric_scheduler builder for ReduceLROnPlateau"]
fn audit_1475_plateau_metric_scheduler_not_implemented() {
    unreachable!("dispatch goal #1475 was not addressed in dispatch B; plateau.rs + learner.rs unchanged")
}

// ---------------------------------------------------------------------------
// #1476 — plateau cooldown/eps/threshold_mode/per_group_min_lr (UNTOUCHED)
// ---------------------------------------------------------------------------

/// Documentation probe: ReduceLROnPlateau builder is missing
/// `cooldown`, `eps`, `threshold_mode='abs'`, per-group `min_lr`.
/// Dispatch did NOT modify `plateau.rs`. REQ-7 in `plateau.rs:16`
/// still says "NOT-STARTED, blocker #1476".
#[test]
#[ignore = "vocab-only on dispatch B; tracking #1476 — plateau builder missing cooldown/eps/threshold_mode/per_group_min_lr"]
fn audit_1476_plateau_extra_builder_args_not_implemented() {
    unreachable!("dispatch goal #1476 was not addressed in dispatch B; plateau.rs unchanged")
}

// ---------------------------------------------------------------------------
// Adam-family probe: trait default returns Err for set_momentum
// ---------------------------------------------------------------------------

/// Probe: optimizers that do NOT own a momentum coefficient (Adam,
/// AdamW, etc.) must Err from the trait default. This pins R-CHAR-3:
/// the divergence is the Adam optimizer accepting `set_momentum` as a
/// no-op vs. legitimately rejecting it.
#[test]
fn audit_adam_set_momentum_rejects_with_invalid_argument() {
    use ferrotorch_optim::{Adam, AdamConfig};
    let p = Parameter::<f32>::from_slice(&[1.0_f32], &[1]).unwrap();
    let mut adam = Adam::new(vec![p], AdamConfig::default());

    let r = adam.set_momentum(0, 0.9);
    assert!(
        r.is_err(),
        "Adam must reject set_momentum (no momentum field); got Ok ({r:?})"
    );
}
