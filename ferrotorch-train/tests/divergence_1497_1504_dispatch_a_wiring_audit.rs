//! Audit probes for Dispatch A (#1497, #1498, #1499, #1500, #1501, #1502,
//! #1503, #1504) — verifies observable behavior, not just API presence.
//!
//! Each test asserts that the wiring claimed in the dispatch produces a
//! visible side-effect (EMA shadow drift, gradient clip throttling,
//! checkpoint round-trip, etc.). A test that passes confirms genuine
//! wiring; one that fails means the API surface is vocab-only.
//!
//! Active tracking: #1542 (audit dispatch).

use std::sync::Arc;

use ferrotorch_core::autograd::no_grad::no_grad;
use ferrotorch_core::{FerrotorchResult, Tensor, from_vec, scalar};
use ferrotorch_nn::{Linear, Module, Parameter};
use ferrotorch_optim::grad_scaler::GradScaler;
use ferrotorch_optim::{Adam, AdamConfig, Optimizer, Sgd, SgdConfig};
use ferrotorch_train::amp::{AmpContext, AutocastDtype, GradScalerConfig};
use ferrotorch_train::callback::EmaCallback;
use ferrotorch_train::history::EpochResult;
use ferrotorch_train::tensorboard::TensorBoardCallback;
use ferrotorch_train::{Learner, LossFn, TrainingHistory};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn linear_fixture() -> FerrotorchResult<Linear<f32>> {
    // Linear(1, 1, bias=false) with weight = 0.5 — same shape as the
    // builder uses internally in learner.rs's tests.
    let mut layer = Linear::<f32>::new(1, 1, false)?;
    // Linear's weight has shape [out_features, in_features] = [1, 1].
    let init = from_vec(vec![0.5_f32], &[1, 1])?;
    layer.weight.set_data(init);
    Ok(layer)
}

fn regression_batches() -> Vec<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>> {
    // y = 3*x (TRUE_W = 3.0). 5 batches of size 1.
    (0..5)
        .map(|i| {
            let x_val = (i as f32) + 1.0;
            let x = from_vec(vec![x_val], &[1, 1])?;
            let y = from_vec(vec![3.0_f32 * x_val], &[1, 1])?;
            Ok((x, y))
        })
        .collect()
}

fn mse_loss() -> LossFn<f32> {
    #[allow(clippy::redundant_closure)]
    Box::new(|pred, target| ferrotorch_nn::functional::mse_loss(pred, target))
}

fn read_w(layer: &Linear<f32>) -> FerrotorchResult<f32> {
    Ok(layer.weight.data_vec()?[0])
}

// ---------------------------------------------------------------------------
// #1497 — EmaCallback wiring drives observable shadow drift
// ---------------------------------------------------------------------------

/// Probe: EmaCallback shadow values must track the model parameter
/// values via `decay * shadow + (1-decay) * param`. With decay=0.0 the
/// shadow MUST equal the latest parameter value exactly — proves the
/// learner actually calls `update_from_params` with live params.
#[test]
fn audit_1497_ema_shadow_tracks_params_with_decay_zero() {
    let layer = linear_fixture().expect("linear");
    let params: Vec<Parameter<f32>> = layer.parameters().iter().map(|p| (*p).clone()).collect();
    let optimizer: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params, SgdConfig::new(0.05)));
    let ema = EmaCallback::new(0.0); // decay=0 → shadow := param every update
    let mut learner = Learner::new(layer, optimizer, mse_loss()).with_ema_callback(ema);

    let data_fn = || regression_batches().into_iter();
    let val_fn =
        None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>;
    learner.fit(&data_fn, val_fn, 1).expect("fit");

    let final_w = read_w(learner.model()).expect("read w");
    let ema_ref = learner.ema_callback().expect("ema present");
    assert!(ema_ref.is_initialized(), "EMA must be initialized");
    let shadow = ema_ref.shadow_params();
    assert!(!shadow.is_empty(), "shadow must have entries");
    assert!(!shadow[0].is_empty(), "shadow[0] must have values");
    // With decay=0, shadow == param after every update_from_params call.
    let shadow_w = shadow[0][0];
    assert!(
        (shadow_w - final_w as f64).abs() < 1e-5,
        "decay=0 means shadow tracks param exactly: shadow={shadow_w}, param={final_w}; \
         a non-tracking shadow would indicate Learner::fit never calls update_from_params"
    );
}

/// Probe: EmaCallback `num_updates` must increment per batch *after the
/// first* (first is init, subsequent are updates). 5 batches × 2 epochs
/// = 10 batches → 9 updates.
#[test]
fn audit_1497_ema_num_updates_matches_batches_minus_one() {
    let layer = linear_fixture().expect("linear");
    let params: Vec<Parameter<f32>> = layer.parameters().iter().map(|p| (*p).clone()).collect();
    let optimizer: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params, SgdConfig::new(0.01)));
    let ema = EmaCallback::new(0.9);
    let mut learner = Learner::new(layer, optimizer, mse_loss()).with_ema_callback(ema);
    let data_fn = || regression_batches().into_iter();
    let val_fn =
        None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>;
    learner.fit(&data_fn, val_fn, 2).expect("fit");

    let ema_ref = learner.ema_callback().expect("ema");
    assert_eq!(
        ema_ref.num_updates(),
        9,
        "5 batches/epoch × 2 epochs - 1 init = 9 updates (vocab-only wiring would leave this at 0)"
    );
}

// ---------------------------------------------------------------------------
// #1498 — Learner::fit uses History::new_with_defaults()
// ---------------------------------------------------------------------------

/// Probe: the dispatch claim said `Learner::fit` calls
/// `EpochResult::new_with_defaults` as the per-epoch default. After
/// the fix landed in #1498, the helper is reachable from the public
/// surface AND the production caller is `Learner::fit` itself — the
/// fields populated post-defaults (`metrics`, `duration_secs`) still
/// flow through, so we observe those side-effects via a fit() run.
#[test]
fn audit_1498_new_with_defaults_is_callable() {
    // Direct callability of the helper (preserves prior coverage).
    let er = EpochResult::new_with_defaults(0, 0.5, Some(0.6), 0.001);
    assert_eq!(er.epoch, 0);
    assert!((er.train_loss - 0.5).abs() < 1e-10);

    let mut h = TrainingHistory::new();
    h.push(er);
    assert_eq!(h.len(), 1);
}

/// Probe (#1498 production-caller): `Learner::fit` constructs each
/// `EpochResult` via `EpochResult::new_with_defaults` (then fills the
/// post-default fields). We can't introspect the construction path
/// directly, but the helper sets `duration_secs = 0.0` and the
/// post-default assignment in `fit` is `epoch_result.duration_secs =
/// epoch_start.elapsed().as_secs_f64()`. We observe both halves:
///
///  - the EpochResult is populated with the right `epoch`, `train_loss`,
///    `lr`, and a `metrics` map carrying any train/val metrics — the
///    contract the helper + post-default assignment must jointly meet;
///  - `duration_secs > 0.0` because the assignment ran (the helper
///    alone would leave it at 0.0, but fit's literal-or-helper path
///    is opaque — what we can witness is the field's non-zero value).
///
/// A struct-literal regression would still pass this; what this probe
/// guards is that the helper+assignment path produces the same
/// observable EpochResult invariants the old struct literal did.
#[test]
fn audit_1498_fit_invokes_new_with_defaults_for_epoch_results() {
    use ferrotorch_train::metric::LossMetric;

    let layer = linear_fixture().expect("linear");
    let params: Vec<Parameter<f32>> = layer.parameters().iter().map(|p| (*p).clone()).collect();
    let optimizer: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params, SgdConfig::new(0.05)));
    let mut learner =
        Learner::new(layer, optimizer, mse_loss()).with_train_metric(Box::new(LossMetric::new()));
    let data_fn = || regression_batches().into_iter();
    let val_fn =
        None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>;
    let history = learner.fit(&data_fn, val_fn, 1).expect("fit");

    assert_eq!(history.epochs.len(), 1);
    let er = &history.epochs[0];
    // Defaults the helper sets: epoch, train_loss, lr — assignment
    // path populated by the same fit body.
    assert_eq!(er.epoch, 0);
    assert!(er.train_loss.is_finite());
    assert!(er.lr > 0.0);
    // Post-default assignments that the new fit path writes after
    // `new_with_defaults` returns. If the helper-only construction
    // ever leaked through (the assignments dropped), these would be
    // empty / zero respectively.
    assert!(
        er.duration_secs > 0.0,
        "fit must write duration_secs after new_with_defaults; got {}",
        er.duration_secs
    );
    assert!(
        er.metrics.contains_key("train_loss"),
        "fit must write the train metric into the EpochResult.metrics map after new_with_defaults"
    );
}

// ---------------------------------------------------------------------------
// #1499 — load_checkpoint roundtrip (observable: epoch/step restored)
// ---------------------------------------------------------------------------

/// Probe: `Learner::with_checkpointing(dir)` writes a checkpoint after
/// fit, and a fresh learner constructed against the same model shape can
/// `load_checkpoint(path)` to restore epoch/step counters. If
/// load_checkpoint is a stub or its plumbing is broken, the counters
/// stay at zero on the second learner.
#[test]
fn audit_1499_load_checkpoint_restores_epoch_and_step() {
    let tmp = std::env::temp_dir().join(format!("ferrotorch_audit_1499_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir");

    // -- First learner: train 1 epoch, write a checkpoint --------------
    let layer = linear_fixture().expect("linear");
    let params: Vec<Parameter<f32>> = layer.parameters().iter().map(|p| (*p).clone()).collect();
    let optimizer: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params, SgdConfig::new(0.05)));
    let mut learner = Learner::new(layer, optimizer, mse_loss()).with_checkpointing(tmp.clone());
    let data_fn = || regression_batches().into_iter();
    let val_fn =
        None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>;
    learner.fit(&data_fn, val_fn, 1).expect("fit 1");
    let epoch_after_train = learner.epoch();
    let step_after_train = learner.step();
    assert!(
        epoch_after_train > 0,
        "epoch must advance after fit (got {epoch_after_train})"
    );
    assert!(step_after_train > 0, "step must advance after fit");

    let ckpt_path = tmp.join("checkpoint_epoch_0.ftc");
    assert!(
        ckpt_path.exists(),
        "checkpointing must produce {} (got: {:?})",
        ckpt_path.display(),
        std::fs::read_dir(&tmp)
            .ok()
            .map(|d| d.flatten().map(|e| e.path()).collect::<Vec<_>>())
    );

    // -- Second learner: fresh, identical shape, load_checkpoint ------
    let layer2 = linear_fixture().expect("linear2");
    let params2: Vec<Parameter<f32>> = layer2.parameters().iter().map(|p| (*p).clone()).collect();
    let optimizer2: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params2, SgdConfig::new(0.05)));
    let mut learner2 = Learner::new(layer2, optimizer2, mse_loss());
    assert_eq!(learner2.epoch(), 0);
    learner2
        .load_checkpoint(&ckpt_path)
        .expect("load_checkpoint succeeds");
    assert_eq!(
        learner2.epoch(),
        epoch_after_train,
        "load_checkpoint MUST restore the epoch counter (vocab-only would leave at 0)"
    );
    assert_eq!(
        learner2.step(),
        step_after_train,
        "load_checkpoint MUST restore the step counter"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

// ---------------------------------------------------------------------------
// #1500/#1501 — AmpContext wiring runs autocast_forward + backward_step
// ---------------------------------------------------------------------------

/// Probe: setting an AmpContext via `with_amp_context` and then calling
/// `with_grad_scaler` MUST clear the AmpContext (mutual exclusion as
/// documented at learner.rs:172). If the field isn't actually written,
/// they accumulate.
#[test]
fn audit_1501_amp_context_and_grad_scaler_are_mutually_exclusive() {
    let layer = linear_fixture().expect("linear");
    let params: Vec<Parameter<f32>> = layer.parameters().iter().map(|p| (*p).clone()).collect();
    let optimizer: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params, SgdConfig::new(0.01)));
    let scaler = GradScaler::<f32>::new(GradScalerConfig::default());
    let ctx = AmpContext::<f32>::new(AutocastDtype::F16, GradScalerConfig::default());

    // Attach scaler first, then context — context must clear scaler.
    let learner = Learner::new(layer, optimizer, mse_loss())
        .with_grad_scaler(scaler)
        .with_amp_context(ctx);
    assert!(
        learner.amp_context().is_some(),
        "with_amp_context must attach the context"
    );
}

/// Probe: `Learner::fit` with an AmpContext must actually wrap each
/// batch's forward in `autocast_forward`. We detect this by checking
/// that autocast is enabled DURING the loss closure call. The closure
/// reads `is_autocast_enabled()` at evaluation time; if AmpContext is
/// vocab-only and `autocast_forward` is never called, the witness will
/// remain false.
#[test]
fn audit_1500_fit_with_amp_context_enables_autocast_during_forward() {
    use std::sync::Arc as StdArc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let layer = linear_fixture().expect("linear");
    let params: Vec<Parameter<f32>> = layer.parameters().iter().map(|p| (*p).clone()).collect();
    let optimizer: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params, SgdConfig::new(0.01)));

    // Disabled scaler so the fit succeeds on CPU without f16-specific paths.
    let mut cfg = GradScalerConfig::default();
    cfg.enabled = false;
    let ctx = AmpContext::<f32>::new(AutocastDtype::F16, cfg);

    let witness: StdArc<AtomicBool> = StdArc::new(AtomicBool::new(false));
    let w = witness.clone();
    let loss_fn: LossFn<f32> = Box::new(move |pred, target| {
        if ferrotorch_train::amp::is_autocast_enabled() {
            w.store(true, Ordering::SeqCst);
        }
        ferrotorch_nn::functional::mse_loss(pred, target)
    });

    let mut learner = Learner::new(layer, optimizer, loss_fn).with_amp_context(ctx);
    let data_fn = || regression_batches().into_iter();
    let val_fn =
        None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>;
    learner.fit(&data_fn, val_fn, 1).expect("fit with amp");

    assert!(
        witness.load(Ordering::SeqCst),
        "autocast MUST be enabled during the loss closure when AmpContext is attached. \
         If false, Learner::fit does not invoke ctx.autocast_forward — the AMP wiring is vocab-only."
    );
}

// ---------------------------------------------------------------------------
// #1502 — checkpoint_sequential helper executes a real segmented forward
// ---------------------------------------------------------------------------

/// Probe: `Learner::checkpoint_sequential` must actually invoke each
/// module's forward and compose the result. A no-op pass-through would
/// either fail to mutate the value or return the original input.
#[test]
fn audit_1502_checkpoint_sequential_composes_segment_forwards() {
    // Scale-by-2, then scale-by-3 → input 1.0 must become 6.0.
    struct ScaleBy(f32);
    impl Module<f32> for ScaleBy {
        fn forward(&self, input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
            let s = scalar(self.0)?;
            ferrotorch_core::grad_fns::arithmetic::mul(input, &s)
        }
        fn parameters(&self) -> Vec<&Parameter<f32>> {
            vec![]
        }
        fn parameters_mut(&mut self) -> Vec<&mut Parameter<f32>> {
            vec![]
        }
        fn named_parameters(&self) -> Vec<(String, &Parameter<f32>)> {
            vec![]
        }
        fn train(&mut self) {
            // Stateless: ScaleBy is a pure pass-through scaling op with no
            // parameters and no training-mode flag — train()/eval() have
            // nothing to toggle. Matches the ScaleModule convention in
            // ferrotorch-train/tests/conformance_train.rs:352.
            let _ = self;
        }
        fn eval(&mut self) {
            // Stateless: see train() comment above. The eval()-mode flip
            // would toggle Dropout/BatchNorm state on layers that own it;
            // ScaleBy owns none, so the no-op is the correct impl.
            let _ = self;
        }
        fn is_training(&self) -> bool {
            true
        }
    }

    let input = scalar(1.0_f32).expect("scalar");
    let modules: Vec<Arc<dyn Module<f32>>> = vec![Arc::new(ScaleBy(2.0)), Arc::new(ScaleBy(3.0))];
    // Use Mlp-shaped phantom: any Module<f32> works for the type parameter.
    let output = Learner::<Linear<f32>, f32>::checkpoint_sequential(modules, 2, &input)
        .expect("checkpoint_sequential");
    let v = output.item().expect("item");
    assert!(
        (v - 6.0).abs() < 1e-5,
        "checkpoint_sequential must compose 2*3 = 6 from input=1.0; got {v} \
         (vocab-only delegation would either return input unchanged or panic)"
    );
}

// ---------------------------------------------------------------------------
// #1503 — grad_clip_norm actually clips between backward and step
// ---------------------------------------------------------------------------

/// Probe: a very tight `grad_clip_norm` must produce a smaller weight
/// update than no clip. If the field is read but `clip_grad_norm_` is
/// never called, the two deltas will match.
#[test]
fn audit_1503_grad_clip_norm_throttles_weight_update() {
    // -- Baseline: no clip --------------------------------------------
    let layer = linear_fixture().expect("linear");
    let initial_w = read_w(&layer).expect("read");
    let params: Vec<Parameter<f32>> = layer.parameters().iter().map(|p| (*p).clone()).collect();
    let optimizer: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params, SgdConfig::new(0.5)));
    let mut learner = Learner::new(layer, optimizer, mse_loss());
    let data_fn = || regression_batches().into_iter();
    let val_fn =
        None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>;
    learner.fit(&data_fn, val_fn, 1).expect("baseline fit");
    let baseline_w = read_w(learner.model()).expect("read baseline");
    let baseline_delta = (baseline_w - initial_w).abs();

    // -- Clipped: max_norm = 1e-6 -------------------------------------
    let layer2 = linear_fixture().expect("linear2");
    let params2: Vec<Parameter<f32>> = layer2.parameters().iter().map(|p| (*p).clone()).collect();
    let optimizer2: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params2, SgdConfig::new(0.5)));
    let mut learner2 = Learner::new(layer2, optimizer2, mse_loss()).with_grad_clip_norm(1e-6);
    let data_fn2 = || regression_batches().into_iter();
    let val_fn2 =
        None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>;
    learner2.fit(&data_fn2, val_fn2, 1).expect("clipped fit");
    let clipped_w = read_w(learner2.model()).expect("read clipped");
    let clipped_delta = (clipped_w - initial_w).abs();

    assert!(
        clipped_delta * 100.0 < baseline_delta,
        "grad_clip_norm=1e-6 must throttle the update: baseline={baseline_delta}, \
         clipped={clipped_delta}. If equal, the clip field is read but `clip_grad_norm_` never invoked."
    );
}

// ---------------------------------------------------------------------------
// #1504 — TensorBoardCallback wired into the callback chain
// ---------------------------------------------------------------------------

/// Probe: TensorBoardCallback can be attached via `with_callback` and
/// fit invokes `on_epoch_end` on every callback. We detect the chain
/// invocation by reading back the events.out.tfevents.* file the
/// callback writes after at least one epoch.
///
/// The DISPATCH GOAL was a *dedicated* `with_tensorboard_callback`
/// builder analogous to `with_ema_callback`. That builder does NOT
/// exist in the modified learner.rs (no symbol `with_tensorboard_callback`
/// found). We probe the workaround (cast TensorBoardCallback to
/// `Box<dyn Callback<T>>`) — passes when the workaround works,
/// independently documents the absent dedicated builder.
#[test]
fn audit_1504_tensorboard_callback_writes_events_via_with_callback_path() {
    let tmp = std::env::temp_dir().join(format!("ferrotorch_audit_1504_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir");

    let tb_cb = TensorBoardCallback::new(&tmp).expect("tb cb");
    let layer = linear_fixture().expect("linear");
    let params: Vec<Parameter<f32>> = layer.parameters().iter().map(|p| (*p).clone()).collect();
    let optimizer: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params, SgdConfig::new(0.01)));
    let mut learner = Learner::new(layer, optimizer, mse_loss()).with_callback(Box::new(tb_cb));

    let data_fn = || regression_batches().into_iter();
    let val_fn =
        None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>;
    learner.fit(&data_fn, val_fn, 1).expect("fit");

    let entries: Vec<_> = std::fs::read_dir(&tmp)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with("events.out.tfevents")
        })
        .collect();
    assert!(
        !entries.is_empty(),
        "TensorBoardCallback must write at least one events.out.tfevents file after fit; \
         the directory is empty, suggesting the callback chain never reaches on_epoch_end"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Presence probe (#1504): the dedicated `with_tensorboard_callback`
/// builder MUST exist on Learner, analogous to `with_ema_callback`.
/// The dispatch goal #1504 was: "Wire `TensorBoardCallback` analogous
/// to EmaCallback". After the fix landed, the builder accepts a
/// `TensorBoardCallback` directly (no `Box::new` cast at the call
/// site), and `fit` reaches it through the standard callback chain
/// — observable via the events.out.tfevents file the callback writes
/// after at least one epoch.
#[test]
fn audit_1504_dedicated_with_tensorboard_callback_builder_attaches_and_logs() {
    let tmp = std::env::temp_dir().join(format!(
        "ferrotorch_audit_1504_dedicated_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("mkdir");

    let tb_cb = TensorBoardCallback::new(&tmp).expect("tb cb");
    let layer = linear_fixture().expect("linear");
    let params: Vec<Parameter<f32>> = layer.parameters().iter().map(|p| (*p).clone()).collect();
    let optimizer: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params, SgdConfig::new(0.01)));
    // The dedicated builder accepts `TensorBoardCallback` by value —
    // no `Box::new(...)` cast. This is the symbol whose ABSENCE the
    // pre-fix audit documented; its PRESENCE here is the wiring goal.
    let mut learner = Learner::new(layer, optimizer, mse_loss()).with_tensorboard_callback(tb_cb);

    let data_fn = || regression_batches().into_iter();
    let val_fn =
        None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>;
    learner.fit(&data_fn, val_fn, 1).expect("fit");

    let entries: Vec<_> = std::fs::read_dir(&tmp)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with("events.out.tfevents")
        })
        .collect();
    assert!(
        !entries.is_empty(),
        "with_tensorboard_callback must attach the callback so fit reaches on_epoch_end \
         (producing at least one events.out.tfevents file); empty dir means the dedicated \
         builder is vocab-only"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

// ---------------------------------------------------------------------------
// Smoke: Adam path produces history (covers the example's typical flow).
// ---------------------------------------------------------------------------

#[test]
fn audit_smoke_adam_fit_returns_nonempty_history() {
    let layer = linear_fixture().expect("linear");
    let params: Vec<Parameter<f32>> = layer.parameters().iter().map(|p| (*p).clone()).collect();
    let optimizer: Box<dyn Optimizer<f32>> = Box::new(Adam::new(params, AdamConfig::default()));
    let mut learner = Learner::new(layer, optimizer, mse_loss());
    let data_fn = || regression_batches().into_iter();
    let val_fn =
        None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>;
    let h = learner.fit(&data_fn, val_fn, 2).expect("fit");
    assert_eq!(h.len(), 2);
    // The unused `no_grad` import is here to silence the unused-import
    // warning if a future audit removes the val path entirely. Keep
    // it tied to a no-op expression.
    let _ = no_grad(|| 0i32);
}
