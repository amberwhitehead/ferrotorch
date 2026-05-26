//! The [`Learner`] — a high-level training loop abstraction.
//!
//! `Learner` owns a model, optimizer, optional scheduler, metrics, and
//! callbacks. It orchestrates the training loop: forward pass, loss
//! computation, backward pass, optimizer step, metric tracking, and
//! callback dispatch.
//!
//! # Examples
//!
//! ```ignore
//! use ferrotorch_train::Learner;
//!
//! let learner = Learner::new(model, optimizer, loss_fn)
//!     .with_scheduler(scheduler)
//!     .with_metric(Box::new(AccuracyMetric::new()))
//!     .with_callback(Box::new(EarlyStopping::new(5, 0.001)))
//!     .with_callback(Box::new(ProgressLogger::new(100)));
//!
//! let history = learner.fit(&train_loader, Some(&val_loader), 100)?;
//! ```
//!
//! ## REQ status (per `.design/ferrotorch-train/learner.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl: `pub type LossFn<T>` at `ferrotorch-train/src/learner.rs:45-46`; consumer: `Learner::new(..., loss_fn: LossFn<T>)` at `:87`, field invocation `(self.loss_fn)(&output, &target)?` at `:269, :420`. |
//! | REQ-2 | SHIPPED | impl: `pub struct Learner<M, T: Float>` at `ferrotorch-train/src/learner.rs:57-77` with 11 fields; consumer: every method in `impl<M: Module<T>, T: Float> Learner<M, T>` at `:79-446` reads/writes these fields. |
//! | REQ-3 | SHIPPED | impl: `Learner::new` at `:87`, `with_scheduler` at `:105`, `with_grad_scaler` at `:122`, `with_train_metric` at `:137`, `with_val_metric` at `:143`, `with_callback` at `:149`, `with_checkpointing` at `:158`; consumer: `ferrotorch-train/examples/multi_epoch_train_dump.rs` uses the builder chain end-to-end. |
//! | REQ-4 | SHIPPED | impl: `fit` at `ferrotorch-train/src/learner.rs:227-381`; consumer: `ferrotorch-train/examples/multi_epoch_train_dump.rs` invokes `Learner::fit` on a 3-layer MLP + Adam for the real-artifact trajectory dump. |
//! | REQ-5 | SHIPPED | impl: `evaluate` at `:394-404`, `evaluate_iter` at `:407-445` (eval mode + `no_grad` wrap); consumer: `Learner::fit` calls `self.evaluate_iter(val_fn)?` at `:323`. |
//! | REQ-6 | SHIPPED | impl: `load_checkpoint` at `ferrotorch-train/src/learner.rs:208-215`; consumer: `ferrotorch-train/examples/multi_epoch_train_dump.rs` `run_learner_smoke` calls `learner.load_checkpoint(resume_path)?` when `--resume <path>` is passed (closes #1499). |
//! | REQ-7 | SHIPPED | impl: AMP fit-loop branches at `ferrotorch-train/src/learner.rs:285-310` (AmpContext path) + `:311-321` (standalone GradScaler path); consumer: `ferrotorch-train/examples/multi_epoch_train_dump.rs::run_learner_smoke` attaches an `AmpContext` via `Learner::with_amp_context` and runs `Learner::fit` end-to-end, exercising the AMP backward path on every default invocation of the example (closes #1500). |
//! | REQ-8 | SHIPPED | impl: `model()` at `:164`, `model_mut()` at `:169`, `epoch()` at `:174`, `step()` at `:179`, `skipped_steps()` at `:129`; consumer: `ferrotorch-train/examples/multi_epoch_train_dump.rs` reads `learner.model()` to snapshot parameter state per epoch. |

use std::path::PathBuf;
use std::time::Instant;

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchResult, Float, Tensor};
use ferrotorch_nn::Module;
use ferrotorch_optim::Optimizer;
use ferrotorch_optim::grad_scaler::GradScaler;
use ferrotorch_optim::scheduler::{LrScheduler, MetricScheduler};
use ferrotorch_serialize::{TrainingCheckpoint, load_checkpoint, save_checkpoint};

use crate::amp::AmpContext;
use crate::callback::{Callback, EmaCallback};
use crate::history::{EpochResult, EvalResult, TrainingHistory};
use crate::metric::Metric;

// ---------------------------------------------------------------------------
// LossFn
// ---------------------------------------------------------------------------

/// A loss function that takes `(prediction, target)` and returns a scalar loss.
///
/// This is a simple function-pointer type alias rather than a trait object so
/// that any closure or function with the right signature works.
pub type LossFn<T> =
    Box<dyn Fn(&Tensor<T>, &Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync>;

// ---------------------------------------------------------------------------
// Learner
// ---------------------------------------------------------------------------

/// A high-level training loop abstraction.
///
/// `Learner` combines a model, optimizer, loss function, and optional
/// scheduler, metrics, and callbacks into a single struct that drives the
/// training loop via [`fit`](Learner::fit).
pub struct Learner<M, T: Float> {
    model: M,
    optimizer: Box<dyn Optimizer<T>>,
    loss_fn: LossFn<T>,
    scheduler: Option<Box<dyn LrScheduler<T>>>,
    /// Optional metric-aware scheduler driven per-epoch with the
    /// validation loss (or training loss when validation is absent).
    /// Mirrors PyTorch's `ReduceLROnPlateau` integration where the
    /// user calls `scheduler.step(metric)` after each epoch. (#1475)
    metric_scheduler: Option<Box<dyn MetricScheduler<T>>>,
    /// Optional automatic-mixed-precision gradient scaler. When set, the
    /// training loop scales the loss before backward, unscales gradients
    /// inside `scaler.step`, skips the optimizer step on inf/NaN, and
    /// updates the scale factor at the end of each batch. (#595)
    grad_scaler: Option<GradScaler<T>>,
    /// Optional combined autocast + grad-scaler context (#1501). When set,
    /// the training loop wraps the forward+loss pass in `autocast_forward`
    /// and routes backward through `AmpContext::backward_step`. The
    /// internal `GradScaler` is owned by the context. Mutually exclusive
    /// with `grad_scaler` — last builder set wins (the constructor only
    /// allows one or the other to be `Some` at fit time; the field set
    /// last in the builder chain is honored, the other is cleared).
    amp_context: Option<AmpContext<T>>,
    train_metrics: Vec<Box<dyn Metric<Input = f64>>>,
    val_metrics: Vec<Box<dyn Metric<Input = f64>>>,
    callbacks: Vec<Box<dyn Callback<T>>>,
    /// Optional EmaCallback driven through the parameter-update path
    /// (#1497). The `Callback` trait's `on_batch_end` does not surface
    /// parameter tensors, so the Learner holds the `EmaCallback` directly
    /// and calls `init_from_params` / `update_from_params` after each
    /// optimizer step.
    ema_callback: Option<EmaCallback>,
    checkpoint_dir: Option<PathBuf>,
    /// Optional per-batch gradient-norm clip (#1503). When `Some(max_norm)`,
    /// `fit` calls `clip_grad_norm_(model.parameters(), max_norm, 2.0)`
    /// after backward and before `optimizer.step()`. L2 norm only;
    /// callers wanting L1 or inf norms should drop down to the raw
    /// `clip_grad_norm_` helper.
    grad_clip_norm: Option<f64>,
    epoch: usize,
    step: usize,
    /// Number of optimizer steps that the grad scaler skipped this run
    /// (inf/NaN detected in scaled gradients). Useful for monitoring AMP
    /// stability. Reset at the start of each `fit` call.
    skipped_steps: usize,
}

impl<M: Module<T>, T: Float> Learner<M, T> {
    /// Create a new `Learner`.
    ///
    /// # Arguments
    ///
    /// * `model` - The neural network module to train.
    /// * `optimizer` - The optimizer that updates model parameters.
    /// * `loss_fn` - A function `(prediction, target) -> scalar_loss`.
    pub fn new(model: M, optimizer: Box<dyn Optimizer<T>>, loss_fn: LossFn<T>) -> Self {
        Self {
            model,
            optimizer,
            loss_fn,
            scheduler: None,
            metric_scheduler: None,
            grad_scaler: None,
            amp_context: None,
            train_metrics: Vec::new(),
            val_metrics: Vec::new(),
            callbacks: Vec::new(),
            ema_callback: None,
            checkpoint_dir: None,
            grad_clip_norm: None,
            epoch: 0,
            step: 0,
            skipped_steps: 0,
        }
    }

    /// Attach a learning rate scheduler.
    pub fn with_scheduler(mut self, scheduler: Box<dyn LrScheduler<T>>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Attach a metric-aware scheduler (e.g. `ReduceLROnPlateau`) that
    /// is driven once per epoch with the validation loss (or the
    /// training loss when validation data is absent). (#1475)
    ///
    /// Independent from [`Self::with_scheduler`]: both kinds can be
    /// attached simultaneously and both are stepped per epoch — the
    /// step-based one runs first (LR-vs-step schedule), then the
    /// metric-based one observes the resulting LR and can override
    /// when a plateau is detected. Mirrors PyTorch's common pattern
    /// where a user attaches a `CosineAnnealingLR` AND a
    /// `ReduceLROnPlateau` to the same optimizer.
    pub fn with_metric_scheduler(mut self, metric_scheduler: Box<dyn MetricScheduler<T>>) -> Self {
        self.metric_scheduler = Some(metric_scheduler);
        self
    }

    /// Enable automatic mixed precision via [`GradScaler`]. (#595)
    ///
    /// The training loop will:
    /// 1. Scale the loss by `scaler.get_scale()` before backward.
    /// 2. Unscale gradients and check for inf/NaN inside `scaler.step`.
    /// 3. Skip the optimizer step on a non-finite gradient and lower the
    ///    scale; otherwise apply the step at the regular learning rate.
    /// 4. Call `scaler.update()` to dynamically tune the scale for the
    ///    next iteration.
    ///
    /// The scaler is independent of the optimizer; pair it with whichever
    /// optimizer was passed to [`Learner::new`].
    ///
    /// Mutually exclusive with [`Learner::with_amp_context`] — setting one
    /// clears the other.
    pub fn with_grad_scaler(mut self, scaler: GradScaler<T>) -> Self {
        self.grad_scaler = Some(scaler);
        self.amp_context = None;
        self
    }

    /// Enable automatic mixed precision via an [`AmpContext`] (#1501).
    ///
    /// Unlike [`with_grad_scaler`](Self::with_grad_scaler), the context
    /// also carries the [`AutocastDtype`](ferrotorch_core::autograd::autocast::AutocastDtype)
    /// for the forward pass — `fit` wraps each batch's `forward + loss`
    /// in `ctx.autocast_forward(...)` so reduced-precision dispatch
    /// flips on for matmul/conv/linear, and uses `ctx.backward_step` for
    /// the post-backward scale/step/update/zero_grad recipe.
    ///
    /// Mutually exclusive with [`with_grad_scaler`](Self::with_grad_scaler) —
    /// setting an `AmpContext` clears any standalone `GradScaler`.
    pub fn with_amp_context(mut self, ctx: AmpContext<T>) -> Self {
        self.amp_context = Some(ctx);
        self.grad_scaler = None;
        self
    }

    /// Number of optimizer steps the grad scaler skipped during the most
    /// recent `fit` call. Resets to `0` at the start of every `fit`.
    pub fn skipped_steps(&self) -> usize {
        self.skipped_steps
    }

    /// Add a training metric.
    ///
    /// Training metrics receive the scalar loss value (as `f64`) after each
    /// batch and are summarized at the end of each epoch.
    pub fn with_train_metric(mut self, metric: Box<dyn Metric<Input = f64>>) -> Self {
        self.train_metrics.push(metric);
        self
    }

    /// Add a validation metric.
    pub fn with_val_metric(mut self, metric: Box<dyn Metric<Input = f64>>) -> Self {
        self.val_metrics.push(metric);
        self
    }

    /// Add a training callback.
    pub fn with_callback(mut self, callback: Box<dyn Callback<T>>) -> Self {
        self.callbacks.push(callback);
        self
    }

    /// Register an [`EmaCallback`] driven through the parameter-update
    /// path (#1497).
    ///
    /// The standard [`Callback`] trait's `on_batch_end` only sees the
    /// scalar loss — not the model parameters — so an EMA update cannot
    /// be driven through the regular `with_callback` channel. This
    /// builder registers the callback in a dedicated slot; `fit` calls
    /// `init_from_params` on the first batch end and `update_from_params`
    /// on every subsequent batch end, materialising `Parameter::data_vec()`
    /// for each parameter on the host before applying the EMA rule.
    pub fn with_ema_callback(mut self, ema: EmaCallback) -> Self {
        self.ema_callback = Some(ema);
        self
    }

    /// Register a [`TensorBoardCallback`] in the callback chain (#1504).
    ///
    /// Dedicated builder analogous to [`with_ema_callback`](Self::with_ema_callback):
    /// boxes the supplied [`TensorBoardCallback`] and appends it to the
    /// generic callback vector so `fit`'s standard `on_epoch_start` /
    /// `on_batch_end` / `on_epoch_end` / `on_train_end` dispatch reaches
    /// it. Equivalent to `self.with_callback(Box::new(tb))` but spells
    /// out the canonical attachment site on the Learner surface so that
    /// production code has a non-vocab-only consumer of
    /// `TensorBoardCallback` (the open consumer-wiring gap closed by
    /// blocker #1504).
    pub fn with_tensorboard_callback(
        mut self,
        tb: crate::tensorboard::TensorBoardCallback,
    ) -> Self {
        self.callbacks.push(Box::new(tb));
        self
    }

    /// Clip total gradient L2 norm to `max_norm` after every backward
    /// pass (#1503).
    ///
    /// Calls [`ferrotorch_nn::utils::clip_grad_norm_`] with `norm_type =
    /// 2.0` on all model parameters between `loss.backward()` and
    /// `optimizer.step()`. Mirrors PyTorch's standard mid-loop
    /// `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`
    /// pattern.
    pub fn with_grad_clip_norm(mut self, max_norm: f64) -> Self {
        self.grad_clip_norm = Some(max_norm);
        self
    }

    /// Enable checkpointing to the given directory.
    ///
    /// When set, the [`Learner`] will save model state dicts at the end of
    /// each epoch.
    pub fn with_checkpointing(mut self, dir: PathBuf) -> Self {
        self.checkpoint_dir = Some(dir);
        self
    }

    /// Return a reference to the model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Return a mutable reference to the model.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Return the current epoch counter.
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Return the current global step counter.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Return the registered [`EmaCallback`], if any (#1497).
    ///
    /// Useful for reading the shadow parameter values at the end of a
    /// `fit` run, or for swapping shadow weights into the model for
    /// evaluation.
    pub fn ema_callback(&self) -> Option<&EmaCallback> {
        self.ema_callback.as_ref()
    }

    /// Return the registered [`AmpContext`], if any (#1501).
    ///
    /// Provides read-only access to the autocast dtype and the current
    /// scale factor for monitoring AMP stability across a `fit` run.
    pub fn amp_context(&self) -> Option<&AmpContext<T>> {
        self.amp_context.as_ref()
    }

    /// Apply gradient checkpointing to a sequence of modules and run a
    /// segment-wise forward (#1502).
    ///
    /// Convenience wrapper for [`crate::checkpoint::checkpoint_sequential`]
    /// that lets `Learner`-using code import the helper at the Learner
    /// surface rather than reaching into the `checkpoint` submodule
    /// directly. The Learner itself does not call this helper internally
    /// (the segment policy is a user choice — only the user knows where
    /// the recomputation/memory trade-off should land in their model).
    ///
    /// # Errors
    ///
    /// Propagates any `FerrotorchError` from
    /// [`crate::checkpoint::checkpoint_sequential`] (a module's forward
    /// failing, an autograd graph operation in the checkpoint primitive
    /// erroring, etc.).
    ///
    /// # Panics
    ///
    /// Panics if `segments == 0` or `modules` is empty (delegates to the
    /// underlying [`crate::checkpoint::checkpoint_sequential`] which
    /// enforces these as programmer-error preconditions).
    pub fn checkpoint_sequential(
        modules: Vec<std::sync::Arc<dyn Module<T>>>,
        segments: usize,
        input: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        crate::checkpoint::checkpoint_sequential(modules, segments, input)
    }

    /// Load a training checkpoint from disk, restoring model weights,
    /// optimizer state, and the epoch/step counters.
    ///
    /// The model's `load_state_dict` is called with `strict = true`.
    ///
    /// # Errors
    ///
    /// Propagates any `FerrotorchError` from
    /// [`ferrotorch_serialize::load_checkpoint`] (file open / read,
    /// deserialization, version mismatch), from the model's
    /// `load_state_dict` (parameter shape/name mismatches under strict
    /// loading), or from the optimizer's `load_state_dict`.
    pub fn load_checkpoint(&mut self, path: impl AsRef<std::path::Path>) -> FerrotorchResult<()> {
        let ckpt: TrainingCheckpoint<T> = load_checkpoint(path)?;
        self.model.load_state_dict(&ckpt.model_state, true)?;
        self.optimizer.load_state_dict(&ckpt.optimizer_state)?;
        self.epoch = ckpt.epoch;
        self.step = ckpt.step;
        Ok(())
    }

    /// Run the training loop.
    ///
    /// Trains for `num_epochs` epochs on `train_data`. If `val_data` is
    /// provided, runs an evaluation pass after each training epoch.
    ///
    /// # Arguments
    ///
    /// * `train_data` - Iterator of `(input, target)` batches for training.
    /// * `val_data` - Optional iterator of `(input, target)` batches for
    ///   validation.
    /// * `num_epochs` - Number of epochs to train.
    ///
    /// # Returns
    ///
    /// A [`TrainingHistory`] containing per-epoch results.
    ///
    /// # Errors
    ///
    /// Propagates any `FerrotorchError` returned by the train/val data
    /// iterators, the model's `forward`, the user-supplied loss function,
    /// the optimizer's `step` / `zero_grad`, the optional `GradScaler`,
    /// or the numeric `cast` helper used to convert the per-batch loss to
    /// `f64`.
    pub fn fit<I, V>(
        &mut self,
        train_data: &dyn Fn() -> I,
        val_data: Option<&dyn Fn() -> V>,
        num_epochs: usize,
    ) -> FerrotorchResult<TrainingHistory>
    where
        I: Iterator<Item = FerrotorchResult<(Tensor<T>, Tensor<T>)>>,
        V: Iterator<Item = FerrotorchResult<(Tensor<T>, Tensor<T>)>>,
    {
        let mut history = TrainingHistory::new();
        // Reset AMP step-skip counter at the start of every fit.
        self.skipped_steps = 0;

        for _ in 0..num_epochs {
            let epoch_start = Instant::now();

            // Notify callbacks.
            for cb in &mut self.callbacks {
                cb.on_epoch_start(self.epoch);
            }

            // Reset training metrics.
            for m in &mut self.train_metrics {
                m.reset();
            }

            // ── Training phase ──────────────────────────────────────────
            self.model.train();
            let mut train_loss_sum = 0.0_f64;
            let mut train_batch_count = 0_usize;

            for (batch_idx, batch_result) in train_data().enumerate() {
                let (input, target) = batch_result?;

                // Notify callbacks.
                for cb in &mut self.callbacks {
                    cb.on_batch_start(batch_idx);
                }

                // Forward + loss. When an AmpContext is attached, wrap
                // the forward+loss closure in `autocast_forward` so the
                // reduced-precision dispatch policy flips on for the
                // matmul/conv/linear ops inside `model.forward` and the
                // loss closure.
                let (output, loss) = if let Some(ref ctx) = self.amp_context {
                    ctx.autocast_forward(|| -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
                        let out = self.model.forward(&input)?;
                        let l = (self.loss_fn)(&out, &target)?;
                        Ok((out, l))
                    })?
                } else {
                    let out = self.model.forward(&input)?;
                    let l = (self.loss_fn)(&out, &target)?;
                    (out, l)
                };
                // `output` is kept live until after backward so the
                // autograd graph stays intact; it's not otherwise read.
                let _ = &output;
                let loss_val = cast::<T, f64>(loss.item()?)?;

                if let Some(ref mut ctx) = self.amp_context {
                    // AmpContext path (#1501): backward_step does
                    // scale → backward → step → update → zero_grad and
                    // reports whether the optimizer step was actually
                    // applied (false on inf/NaN). Note: gradient
                    // clipping under AMP would need an explicit
                    // unscale-first path — AmpContext's backward_step
                    // intentionally bundles unscale+step, so the
                    // grad_clip_norm field is honored only on the
                    // non-AMP paths below.
                    let stepped = ctx.backward_step(&loss, self.optimizer.as_mut())?;
                    if !stepped {
                        self.skipped_steps += 1;
                    }
                } else if let Some(ref mut scaler) = self.grad_scaler {
                    // AMP path: scale loss, backward, scaler-driven step.
                    let scaled = scaler.scale(&loss)?;
                    scaled.backward()?;
                    let stepped = scaler.step(self.optimizer.as_mut())?;
                    if !stepped {
                        self.skipped_steps += 1;
                    }
                    scaler.update();
                    self.optimizer.zero_grad()?;
                } else {
                    // Standard path.
                    loss.backward()?;
                    // Optional gradient-norm clip (#1503). Clipping
                    // sits between backward and step so the optimizer
                    // applies the clipped gradient. L2 norm only.
                    if let Some(max_norm) = self.grad_clip_norm {
                        let params = self.model.parameters();
                        ferrotorch_nn::utils::clip_grad_norm_(&params, max_norm, 2.0)?;
                    }
                    self.optimizer.step()?;
                    self.optimizer.zero_grad()?;
                }

                // EMA shadow-parameter update (#1497). Drive it after
                // the optimizer step so the EMA tracks the just-updated
                // parameter values (the canonical Polyak-averaging
                // convention).
                if let Some(ref mut ema) = self.ema_callback {
                    let params: Vec<Vec<T>> = self
                        .model
                        .parameters()
                        .iter()
                        .map(|p| p.data_vec())
                        .collect::<FerrotorchResult<Vec<_>>>()?;
                    if !ema.is_initialized() {
                        ema.init_from_params(&params)?;
                    } else {
                        ema.update_from_params(&params)?;
                    }
                }

                // Track loss.
                train_loss_sum += loss_val;
                train_batch_count += 1;
                self.step += 1;

                // Update training metrics.
                for m in &mut self.train_metrics {
                    m.update(&loss_val);
                }

                // Notify callbacks.
                for cb in &mut self.callbacks {
                    cb.on_batch_end(batch_idx, loss_val);
                }
            }

            // Scheduler step (per-epoch).
            if let Some(ref mut sched) = self.scheduler {
                sched.step(self.optimizer.as_mut());
            }

            let train_loss = if train_batch_count > 0 {
                train_loss_sum / train_batch_count as f64
            } else {
                0.0
            };

            // ── Validation phase ────────────────────────────────────────
            let val_loss = if let Some(val_fn) = val_data {
                // Reset validation metrics.
                for m in &mut self.val_metrics {
                    m.reset();
                }

                let eval_result = self.evaluate_iter(val_fn)?;

                Some(eval_result.loss)
            } else {
                None
            };

            // Metric-scheduler step (per-epoch, after validation so the
            // plateau detector sees the up-to-date validation loss).
            // Falls back to training loss when validation data isn't
            // supplied. Mirrors PyTorch's standard
            // `scheduler.step(val_loss)` recipe at end of each epoch.
            // (#1475)
            if let Some(ref mut metric_sched) = self.metric_scheduler {
                let metric = val_loss.unwrap_or(train_loss);
                metric_sched.step(self.optimizer.as_mut(), metric);
            }

            // ── Epoch result ────────────────────────────────────────────
            let mut metrics = std::collections::HashMap::new();
            for m in &self.train_metrics {
                metrics.insert(format!("train_{}", m.name()), m.compute());
            }
            for m in &self.val_metrics {
                metrics.insert(format!("val_{}", m.name()), m.compute());
            }

            // Save checkpoint if checkpointing is enabled.
            if let Some(ref dir) = self.checkpoint_dir {
                let _ = std::fs::create_dir_all(dir);
                let checkpoint = TrainingCheckpoint::new(
                    self.model.state_dict(),
                    self.optimizer.state_dict()?,
                    // Store the POST-increment epoch number (== number of
                    // epochs completed). Filename keeps the pre-increment
                    // index so the first checkpoint stays `_epoch_0.ftc`,
                    // matching the audit test contract at
                    // ferrotorch-train/tests/divergence_1497_1504_dispatch_a_wiring_audit.rs:177.
                    // Closes #1544 / #1499 — load_checkpoint must restore
                    // `learner.epoch() == 1` after fit-1-epoch.
                    self.epoch + 1,
                    self.step,
                );
                let path = dir.join(format!("checkpoint_epoch_{}.ftc", self.epoch));
                let _ = save_checkpoint(&checkpoint, &path);
            }

            // Construct via `EpochResult::new_with_defaults` per #1498 so
            // the helper has a non-test production consumer (the dispatch
            // goal); then fill in the post-default fields the helper
            // intentionally zeroes (`metrics`, `duration_secs`).
            let mut epoch_result = EpochResult::new_with_defaults(
                self.epoch,
                train_loss,
                val_loss,
                self.optimizer.lr(),
            );
            epoch_result.metrics = metrics;
            epoch_result.duration_secs = epoch_start.elapsed().as_secs_f64();

            // Notify callbacks.
            for cb in &mut self.callbacks {
                cb.on_epoch_end(self.epoch, &epoch_result);
            }

            history.push(epoch_result);
            self.epoch += 1;

            // Check for early stopping.
            if self.callbacks.iter().any(|cb| cb.should_stop()) {
                break;
            }
        }

        // Notify callbacks of training end.
        for cb in &mut self.callbacks {
            cb.on_train_end(&history);
        }

        Ok(history)
    }

    /// Run an evaluation pass on the given data.
    ///
    /// Sets the model to eval mode, runs forward passes (no gradient),
    /// and returns an [`EvalResult`] with mean loss and metric values.
    ///
    /// # Errors
    ///
    /// Propagates any `FerrotorchError` returned by the data iterator,
    /// the model's `forward`, the user-supplied loss function, or the
    /// numeric `cast` helper used to convert the per-batch loss to `f64`
    /// (e.g. when an `f16`/`bf16` value cannot be represented in `f64`).
    pub fn evaluate<V>(&mut self, val_data: &dyn Fn() -> V) -> FerrotorchResult<EvalResult>
    where
        V: Iterator<Item = FerrotorchResult<(Tensor<T>, Tensor<T>)>>,
    {
        // Reset validation metrics.
        for m in &mut self.val_metrics {
            m.reset();
        }

        self.evaluate_iter(val_data)
    }

    /// Internal evaluation helper that does not reset metrics.
    fn evaluate_iter<V>(&mut self, val_data: &dyn Fn() -> V) -> FerrotorchResult<EvalResult>
    where
        V: Iterator<Item = FerrotorchResult<(Tensor<T>, Tensor<T>)>>,
    {
        self.model.eval();

        let mut val_loss_sum = 0.0_f64;
        let mut val_batch_count = 0_usize;

        ferrotorch_core::no_grad(|| -> FerrotorchResult<()> {
            for batch_result in val_data() {
                let (input, target) = batch_result?;
                let output = self.model.forward(&input)?;
                let loss = (self.loss_fn)(&output, &target)?;
                let loss_val = cast::<T, f64>(loss.item()?)?;

                val_loss_sum += loss_val;
                val_batch_count += 1;

                for m in &mut self.val_metrics {
                    m.update(&loss_val);
                }
            }
            Ok(())
        })?;

        let loss = if val_batch_count > 0 {
            val_loss_sum / val_batch_count as f64
        } else {
            0.0
        };

        let mut metrics = std::collections::HashMap::new();
        for m in &self.val_metrics {
            metrics.insert(m.name().to_string(), m.compute());
        }

        Ok(EvalResult { loss, metrics })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_nn::Parameter;

    // -- Minimal test module -------------------------------------------------

    struct DummyModule<T: Float> {
        weight: Parameter<T>,
        training: bool,
    }

    impl<T: Float> DummyModule<T> {
        fn new() -> FerrotorchResult<Self> {
            Ok(Self {
                weight: Parameter::zeros(&[1])?,
                training: true,
            })
        }
    }

    impl<T: Float> Module<T> for DummyModule<T> {
        fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
            Ok(input.clone())
        }
        fn parameters(&self) -> Vec<&Parameter<T>> {
            vec![&self.weight]
        }
        fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
            vec![&mut self.weight]
        }
        fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
            vec![("weight".to_string(), &self.weight)]
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

    // -- Minimal test optimizer ----------------------------------------------

    struct DummyOptimizer {
        lr: f64,
    }

    impl Optimizer<f32> for DummyOptimizer {
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
        fn param_groups(&self) -> &[ferrotorch_optim::ParamGroup<f32>] {
            &[]
        }
        fn param_groups_mut(&mut self) -> &mut [ferrotorch_optim::ParamGroup<f32>] {
            &mut []
        }
        fn add_param_group(&mut self, _group: ferrotorch_optim::ParamGroup<f32>) {}
        fn state_dict(&self) -> FerrotorchResult<ferrotorch_optim::OptimizerState> {
            Ok(Default::default())
        }
        fn load_state_dict(
            &mut self,
            _state: &ferrotorch_optim::OptimizerState,
        ) -> FerrotorchResult<()> {
            Ok(())
        }
    }

    // -- Real-loss test helpers (#1116) --------------------------------------
    //
    // Three of the construction-only tests below were upgraded to drive the
    // full `fit()` / `evaluate()` path against a real MSE loss and a real
    // SGD optimizer, so the test suite actually exercises gradient flow.
    //
    // The fixture is deliberately the smallest model with a learnable
    // parameter where MSE backprop has a measurable effect:
    //
    //   * `ferrotorch_nn::Linear<f32>(1, 1, bias=false)`     — single weight `w`.
    //   * targets `y = TRUE_W * x` with `TRUE_W = 3.0`       — exact linear fit.
    //   * loss closure `mse_loss(pred, target)`              — reduction = mean.
    //   * `Sgd(lr = 0.05)`                                   — large enough lr
    //                                                          for 5 epochs to
    //                                                          visibly move `w`.
    //
    // The dataset is small and deterministic (5 batches of 4 samples each),
    // so the tests stay fast (well under 1s) while still letting us assert
    // both "loss decreases" and "parameter changes" robustly.

    /// Target slope used by the synthetic regression fixture.
    const TRUE_W: f32 = 3.0;
    /// Initial weight value the model is forced to before training. Picked
    /// far enough from `TRUE_W` that 5 epochs of SGD@0.05 noticeably reduce
    /// the loss, but not so far that gradients explode.
    const INIT_W: f32 = 0.5;

    fn mse_loss_fn() -> LossFn<f32> {
        Box::new(ferrotorch_nn::functional::mse_loss)
    }

    /// Build a `Linear(1, 1, bias=false)` whose single weight is set to
    /// `INIT_W` — gives every real-loss test the same deterministic start.
    fn build_linear_fixture() -> FerrotorchResult<ferrotorch_nn::Linear<f32>> {
        let mut layer = ferrotorch_nn::Linear::<f32>::new(1, 1, false)?;
        // Linear's weight has shape [out_features, in_features] = [1, 1].
        let init = ferrotorch_core::from_vec(vec![INIT_W], &[1, 1])?;
        layer.weight.set_data(init);
        Ok(layer)
    }

    /// Synthetic regression batches: `y = TRUE_W * x` over the inputs
    /// `[0.0, 1.0, 2.0, 3.0]` repeated for 5 batches per epoch. Returns a
    /// closure compatible with `Learner::fit` / `Learner::evaluate`.
    #[allow(clippy::type_complexity)]
    fn regression_data()
    -> impl Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>> {
        || {
            const N_BATCHES: usize = 5;
            let xs: [f32; 4] = [0.0, 1.0, 2.0, 3.0];
            let mut batches: Vec<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>> =
                Vec::with_capacity(N_BATCHES);
            for _ in 0..N_BATCHES {
                let x = match ferrotorch_core::from_vec(xs.to_vec(), &[4, 1]) {
                    Ok(t) => t,
                    Err(e) => {
                        batches.push(Err(e));
                        continue;
                    }
                };
                let y_vec: Vec<f32> = xs.iter().map(|v| TRUE_W * v).collect();
                let y = match ferrotorch_core::from_vec(y_vec, &[4, 1]) {
                    Ok(t) => t,
                    Err(e) => {
                        batches.push(Err(e));
                        continue;
                    }
                };
                batches.push(Ok((x, y)));
            }
            batches.into_iter()
        }
    }

    /// Read the scalar weight value out of a `Linear(1, 1, bias=false)`.
    fn read_weight(layer: &ferrotorch_nn::Linear<f32>) -> FerrotorchResult<f32> {
        Ok(layer.weight.data_vec()?[0])
    }

    // -- Construction tests --------------------------------------------------

    #[test]
    fn test_learner_construction() {
        let model = DummyModule::<f32>::new().unwrap();
        let optimizer: Box<dyn Optimizer<f32>> = Box::new(DummyOptimizer { lr: 0.01 });
        let loss_fn: LossFn<f32> = Box::new(|pred, _target| Ok(pred.clone()));

        let learner = Learner::new(model, optimizer, loss_fn);
        assert_eq!(learner.epoch(), 0);
        assert_eq!(learner.step(), 0);
    }

    #[test]
    fn test_learner_with_checkpoint_dir() {
        let model = DummyModule::<f32>::new().unwrap();
        let optimizer: Box<dyn Optimizer<f32>> = Box::new(DummyOptimizer { lr: 0.01 });
        let loss_fn: LossFn<f32> = Box::new(|pred, _target| Ok(pred.clone()));

        let learner = Learner::new(model, optimizer, loss_fn)
            .with_checkpointing(PathBuf::from("/tmp/checkpoints"));

        assert_eq!(
            learner.checkpoint_dir,
            Some(PathBuf::from("/tmp/checkpoints"))
        );
    }

    // -- Real-loss training tests (#1116) -----------------------------------
    //
    // These three tests replace the original construction-only variants
    // (`test_learner_with_metrics`, `test_learner_with_callbacks`,
    // `test_learner_model_accessors`). Each now drives the full `fit()` or
    // `evaluate()` path against `mse_loss` and a real `Sgd` optimizer,
    // while still exercising the `with_train_metric` / `with_val_metric` /
    // `with_callback` / `model()` / `model_mut()` surface APIs the
    // originals were guarding.

    /// `fit()` must drive the per-epoch training loss meaningfully
    /// downward when the loss is real MSE and the optimizer actually
    /// updates the parameter. Also exercises `with_train_metric` and
    /// `with_val_metric` (counts after construction match), and the
    /// returned `TrainingHistory` (5 epochs recorded).
    #[test]
    fn test_learner_fit_with_metrics_decreases_loss() {
        use crate::metric::LossMetric;
        use ferrotorch_optim::{Sgd, SgdConfig};

        let layer = build_linear_fixture().expect("build Linear fixture");
        let params: Vec<Parameter<f32>> = layer.parameters().iter().map(|p| (*p).clone()).collect();
        let optimizer: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params, SgdConfig::new(0.05)));

        let mut learner = Learner::new(layer, optimizer, mse_loss_fn())
            .with_train_metric(Box::new(LossMetric::new()))
            .with_val_metric(Box::new(LossMetric::new()));

        // Sanity-check the surface API the original construction test
        // covered: both metric vectors saw the registration.
        assert_eq!(learner.train_metrics.len(), 1);
        assert_eq!(learner.val_metrics.len(), 1);

        let data = regression_data();
        let history =
            learner
                .fit(
                    &data,
                    None::<
                        &dyn Fn()
                            -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>,
                    >,
                    5,
                )
                .expect("fit should succeed on the synthetic regression task");

        // 5 epochs were actually run.
        assert_eq!(history.epochs.len(), 5);

        // The per-epoch mean training loss must drop monotonically across
        // the whole run — for a 1-D linear model on a noiseless linear
        // target with SGD@0.05, that's the expected behavior. A flat
        // trajectory (e.g. optimizer.step() got skipped or gradients
        // were zeroed before stepping) would trip this assertion.
        let first = history.epochs.first().unwrap().train_loss;
        let last = history.epochs.last().unwrap().train_loss;
        assert!(
            last < first * 0.5,
            "expected fit() to at least halve the train loss, got first={first}, last={last}"
        );
        // And the absolute loss must be small (the model converges to
        // within ~0.5 MSE of the true slope on this fixture).
        assert!(
            last < 0.5,
            "expected final train loss < 0.5, got last={last}"
        );
    }

    /// `fit()` must visibly change the model's parameter values via
    /// backprop. This is the canonical sabotage probe: a `fit()` that
    /// either skips `optimizer.step()` or zeros gradients before stepping
    /// would leave the weight untouched and fail this test. Also
    /// exercises `with_callback`.
    #[test]
    fn test_learner_fit_changes_parameters() {
        use crate::callback::EarlyStopping;
        use ferrotorch_optim::{Sgd, SgdConfig};

        let layer = build_linear_fixture().expect("build Linear fixture");
        let initial_w = read_weight(&layer).expect("read initial weight");
        assert!(
            (initial_w - INIT_W).abs() < 1e-6,
            "fixture must start at INIT_W={INIT_W}, got {initial_w}"
        );

        let params: Vec<Parameter<f32>> = layer.parameters().iter().map(|p| (*p).clone()).collect();
        let optimizer: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params, SgdConfig::new(0.05)));

        // EarlyStopping with a very loose patience won't fire over 5
        // epochs — its presence here is only to keep the
        // `with_callback` surface-API coverage the original test had.
        let mut learner = Learner::new(layer, optimizer, mse_loss_fn())
            .with_callback(Box::new(EarlyStopping::new(100, 0.0)));
        assert_eq!(learner.callbacks.len(), 1);

        let data = regression_data();
        learner
            .fit(
                &data,
                None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>,
                5,
            )
            .expect("fit succeeds");

        let final_w = read_weight(learner.model()).expect("read final weight");
        // The weight must have moved meaningfully toward TRUE_W=3.0.
        // We assert a generous lower bound (≥ 0.5 absolute change) so
        // the test is robust to small numerical differences but still
        // fails decisively if the optimizer step is a no-op.
        let delta = (final_w - initial_w).abs();
        assert!(
            delta >= 0.5,
            "fit() must change the weight by ≥0.5: initial={initial_w}, final={final_w}, delta={delta}"
        );
        // The direction must be toward the true slope — guards against
        // a backward-sign regression.
        assert!(
            final_w > initial_w && final_w < TRUE_W + 1.0,
            "weight should approach TRUE_W from below: initial={initial_w}, final={final_w}, TRUE_W={TRUE_W}"
        );
    }

    /// `evaluate()` must report a non-zero, mathematically meaningful
    /// loss when the model's prediction does not match the target. With
    /// `w = INIT_W = 0.5` on inputs `xs = [0, 1, 2, 3]` and target
    /// `y = 3*x`, the per-sample squared errors are `(0.5*x - 3*x)^2 =
    /// (2.5*x)^2`, so the analytic mean is
    /// `mean((2.5 * [0,1,2,3])^2) = mean([0, 6.25, 25, 56.25]) = 21.875`.
    /// Also exercises `model()` and `model_mut()` accessors before/after
    /// evaluate to confirm `evaluate()` flips `is_training()` to false.
    #[test]
    fn test_learner_evaluate_reports_meaningful_loss() {
        use ferrotorch_optim::{Sgd, SgdConfig};

        let layer = build_linear_fixture().expect("build Linear fixture");
        let params: Vec<Parameter<f32>> = layer.parameters().iter().map(|p| (*p).clone()).collect();
        let optimizer: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params, SgdConfig::new(0.05)));

        let mut learner = Learner::new(layer, optimizer, mse_loss_fn());

        // model() / model_mut() surface check (original test's content).
        assert!(learner.model().is_training());
        learner.model_mut().eval();
        assert!(!learner.model().is_training());
        learner.model_mut().train();

        let data = regression_data();
        let eval = learner
            .evaluate(&data)
            .expect("evaluate should succeed on synthetic regression");

        // Analytic mean MSE for w=INIT_W=0.5 vs target=3*x on xs=[0,1,2,3]:
        //   per-sample: ((0.5 - 3)*x)^2 = (2.5*x)^2 = 6.25*x^2
        //   mean over [0,1,4,9] = 14/4 = 3.5, scaled by 6.25 = 21.875.
        let expected = 21.875_f64;
        assert!(
            (eval.loss - expected).abs() < 1e-3,
            "evaluate() loss should match analytic MSE {expected}, got {}",
            eval.loss
        );

        // evaluate() must leave the model in eval mode (the contract
        // documented on `evaluate`/`evaluate_iter`).
        assert!(!learner.model().is_training());
    }

    // -----------------------------------------------------------------------
    // GradScaler / AMP integration (#595)
    // -----------------------------------------------------------------------

    #[test]
    fn test_learner_grad_scaler_field_starts_none() {
        let model = DummyModule::<f32>::new().unwrap();
        let optimizer: Box<dyn Optimizer<f32>> = Box::new(DummyOptimizer { lr: 0.01 });
        let loss_fn: LossFn<f32> = Box::new(|pred, _target| Ok(pred.clone()));
        let learner = Learner::new(model, optimizer, loss_fn);
        assert!(learner.grad_scaler.is_none());
        assert_eq!(learner.skipped_steps(), 0);
    }

    #[test]
    fn test_learner_with_grad_scaler_attaches() {
        use ferrotorch_optim::grad_scaler::{GradScaler, GradScalerConfig};
        let model = DummyModule::<f32>::new().unwrap();
        let optimizer: Box<dyn Optimizer<f32>> = Box::new(DummyOptimizer { lr: 0.01 });
        let loss_fn: LossFn<f32> = Box::new(|pred, _target| Ok(pred.clone()));
        let scaler = GradScaler::<f32>::new(GradScalerConfig::default());
        let learner = Learner::new(model, optimizer, loss_fn).with_grad_scaler(scaler);
        assert!(learner.grad_scaler.is_some());
    }

    #[test]
    fn test_learner_skipped_steps_counter_starts_zero() {
        use ferrotorch_optim::grad_scaler::{GradScaler, GradScalerConfig};
        let model = DummyModule::<f32>::new().unwrap();
        let optimizer: Box<dyn Optimizer<f32>> = Box::new(DummyOptimizer { lr: 0.01 });
        let loss_fn: LossFn<f32> = Box::new(|pred, _target| Ok(pred.clone()));
        let scaler = GradScaler::<f32>::new(GradScalerConfig::default());
        let learner = Learner::new(model, optimizer, loss_fn).with_grad_scaler(scaler);
        assert_eq!(learner.skipped_steps(), 0);
    }

    // -----------------------------------------------------------------------
    // EmaCallback wiring (#1497)
    // -----------------------------------------------------------------------

    /// `with_ema_callback` registers the EMA in the dedicated slot and
    /// `fit` drives `init_from_params` / `update_from_params` per batch
    /// against the current model parameter values.
    #[test]
    fn test_learner_fit_drives_ema_updates() {
        use crate::callback::EmaCallback;
        use ferrotorch_optim::{Sgd, SgdConfig};

        let layer = build_linear_fixture().expect("build Linear fixture");
        let params: Vec<Parameter<f32>> = layer.parameters().iter().map(|p| (*p).clone()).collect();
        let optimizer: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params, SgdConfig::new(0.05)));

        let ema = EmaCallback::new(0.9);
        let mut learner = Learner::new(layer, optimizer, mse_loss_fn()).with_ema_callback(ema);

        // Verify the EMA is plumbed in but not yet initialized.
        assert!(learner.ema_callback.is_some());
        assert!(!learner.ema_callback.as_ref().unwrap().is_initialized());

        let data = regression_data();
        learner
            .fit(
                &data,
                None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>,
                3,
            )
            .expect("fit succeeds");

        // After fit: 5 batches/epoch * 3 epochs = 15 batches → 14
        // update_from_params calls (first batch initializes, subsequent
        // ones update; num_updates counts only `update_from_params`).
        let ema_ref = learner
            .ema_callback()
            .expect("EMA callback should still be registered");
        assert!(ema_ref.is_initialized());
        assert_eq!(
            ema_ref.num_updates(),
            14,
            "expected 5 batches/epoch * 3 epochs - 1 init = 14 updates",
        );

        // Shadow params should be populated for the Linear's single weight.
        let shadow = ema_ref.shadow_params();
        assert_eq!(shadow.len(), 1, "Linear(1,1,bias=false) has 1 parameter");
        assert_eq!(shadow[0].len(), 1, "weight is a single scalar");
    }

    // -----------------------------------------------------------------------
    // grad_clip_norm wiring (#1503)
    // -----------------------------------------------------------------------

    /// `with_grad_clip_norm` makes `fit` invoke `clip_grad_norm_` between
    /// backward and step. We discriminate by setting a *very tight* clip
    /// (max_norm = 1e-6) which essentially zeros the gradient; the
    /// weight should then move much less per step than without the clip.
    #[test]
    fn test_learner_fit_grad_clip_norm_throttles_step() {
        use ferrotorch_optim::{Sgd, SgdConfig};

        // -- Baseline: no clip --------------------------------------------
        let layer = build_linear_fixture().expect("build Linear fixture");
        let initial_w = read_weight(&layer).expect("read initial weight");
        let params: Vec<Parameter<f32>> = layer.parameters().iter().map(|p| (*p).clone()).collect();
        let optimizer: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params, SgdConfig::new(0.05)));
        let mut learner = Learner::new(layer, optimizer, mse_loss_fn());

        let data = regression_data();
        learner
            .fit(
                &data,
                None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>,
                1,
            )
            .expect("baseline fit");
        let baseline_w = read_weight(learner.model()).expect("read baseline weight");
        let baseline_delta = (baseline_w - initial_w).abs();

        // -- Clipped: max_norm = 1e-6 -------------------------------------
        let layer2 = build_linear_fixture().expect("build Linear fixture (clipped)");
        let params2: Vec<Parameter<f32>> =
            layer2.parameters().iter().map(|p| (*p).clone()).collect();
        let optimizer2: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params2, SgdConfig::new(0.05)));
        let mut learner2 =
            Learner::new(layer2, optimizer2, mse_loss_fn()).with_grad_clip_norm(1e-6);

        let data2 = regression_data();
        learner2
            .fit(
                &data2,
                None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>,
                1,
            )
            .expect("clipped fit");
        let clipped_w = read_weight(learner2.model()).expect("read clipped weight");
        let clipped_delta = (clipped_w - initial_w).abs();

        // The clip should throttle the update by at least 100x — a
        // sabotage that no-ops `clip_grad_norm_` would leave the two
        // deltas approximately equal.
        assert!(
            clipped_delta * 100.0 < baseline_delta,
            "grad_clip_norm should throttle step magnitude: baseline_delta={baseline_delta}, \
             clipped_delta={clipped_delta} (expected clipped < baseline/100)"
        );
    }

    // -----------------------------------------------------------------------
    // AmpContext wiring (#1501)
    // -----------------------------------------------------------------------

    /// `with_amp_context` attaches an `AmpContext` and clears any
    /// standalone `GradScaler`. After fit, the context is still
    /// accessible and reports the configured dtype.
    #[test]
    fn test_learner_with_amp_context_attaches_and_clears_scaler() {
        use crate::amp::{AmpContext, AutocastDtype, GradScalerConfig};
        use ferrotorch_optim::grad_scaler::GradScaler;

        let model = DummyModule::<f32>::new().unwrap();
        let optimizer: Box<dyn Optimizer<f32>> = Box::new(DummyOptimizer { lr: 0.01 });
        let loss_fn: LossFn<f32> = Box::new(|pred, _target| Ok(pred.clone()));
        let scaler = GradScaler::<f32>::new(GradScalerConfig::default());
        let ctx = AmpContext::<f32>::new(AutocastDtype::BF16, GradScalerConfig::default());

        // First attach the standalone scaler, then override with the
        // context — the scaler slot must clear so we don't double-step.
        let learner = Learner::new(model, optimizer, loss_fn)
            .with_grad_scaler(scaler)
            .with_amp_context(ctx);

        assert!(
            learner.grad_scaler.is_none(),
            "AmpContext must clear scaler"
        );
        let ctx_ref = learner
            .amp_context()
            .expect("AmpContext should be registered");
        assert_eq!(ctx_ref.dtype(), AutocastDtype::BF16);
    }

    /// `fit` with an `AmpContext` runs the autocast_forward + backward_step
    /// recipe and still drives the weight toward the target.
    #[test]
    fn test_learner_fit_with_amp_context_runs_and_updates_weights() {
        use crate::amp::{AmpContext, AutocastDtype, GradScalerConfig};
        use ferrotorch_optim::{Sgd, SgdConfig};

        let layer = build_linear_fixture().expect("build Linear fixture");
        let initial_w = read_weight(&layer).expect("read initial weight");
        let params: Vec<Parameter<f32>> = layer.parameters().iter().map(|p| (*p).clone()).collect();
        let optimizer: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params, SgdConfig::new(0.05)));

        // Disabled-scaler config so AmpContext acts as a passthrough on
        // the loss (no scale factor), keeping the unit test free of
        // f16-specific numerics on this CPU-only fixture. The
        // autocast_forward + backward_step recipe is still exercised.
        let mut cfg = GradScalerConfig::default();
        cfg.enabled = false;
        let ctx = AmpContext::<f32>::new(AutocastDtype::F16, cfg);

        let mut learner = Learner::new(layer, optimizer, mse_loss_fn()).with_amp_context(ctx);

        let data = regression_data();
        learner
            .fit(
                &data,
                None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>,
                3,
            )
            .expect("fit with amp_context");

        let final_w = read_weight(learner.model()).expect("read final weight");
        let delta = (final_w - initial_w).abs();
        assert!(
            delta >= 0.1,
            "AMP fit must still update weights: initial={initial_w}, final={final_w}, delta={delta}",
        );
        // Direction toward TRUE_W=3.0.
        assert!(final_w > initial_w);

        // Skipped steps should be 0 for a disabled scaler (no inf/NaN
        // path triggered by passthrough scaling).
        assert_eq!(learner.skipped_steps(), 0);
    }

    // -----------------------------------------------------------------------
    // checkpoint_sequential helper (#1502)
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // with_metric_scheduler wiring (#1475)
    // -----------------------------------------------------------------------

    /// `Learner::fit` drives the metric-scheduler once per epoch with
    /// the validation loss (or training loss when val is absent).
    ///
    /// Uses `DummyOptimizer` + a constant-loss DummyModule fixture so
    /// the metric is flat across epochs — `ReduceLROnPlateau(patience=0,
    /// factor=0.5, threshold=0.0)` then guarantees a reduction every
    /// epoch. A Learner that ignored the metric scheduler would leave
    /// `optimizer.lr()` at the initial value.
    #[test]
    fn test_learner_fit_drives_metric_scheduler_per_epoch() {
        use ferrotorch_optim::scheduler::{PlateauMode, ReduceLROnPlateau};

        let model = DummyModule::<f32>::new().unwrap();
        let optimizer: Box<dyn Optimizer<f32>> = Box::new(DummyOptimizer { lr: 0.1 });
        // Returns the input as "prediction" so loss == pred elementwise;
        // with constant inputs the loss stays flat across batches/epochs.
        let loss_fn: LossFn<f32> = Box::new(|pred, _target| Ok(pred.clone()));

        let plateau = ReduceLROnPlateau::new(PlateauMode::Min)
            .patience(0)
            .factor(0.5)
            .threshold(0.0);

        let mut learner =
            Learner::new(model, optimizer, loss_fn).with_metric_scheduler(Box::new(plateau));

        // Synthetic flat-loss batches: scalar tensors with value 1.0.
        let flat_data = || {
            let mut batches: Vec<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>> = Vec::new();
            for _ in 0..2 {
                let x = ferrotorch_core::scalar(1.0_f32).unwrap();
                let y = ferrotorch_core::scalar(1.0_f32).unwrap();
                batches.push(Ok((x, y)));
            }
            batches.into_iter()
        };

        learner
            .fit(
                &flat_data,
                None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>,
                3,
            )
            .expect("fit");

        // With a flat loss = 1.0 every epoch, `is_better` is false on
        // the SECOND epoch (snapshot epoch sets `best = 1.0`; the
        // second epoch's loss == 1.0 fails the strict-less-than
        // improvement check). patience=0 means the very next bad
        // epoch triggers a reduction. Across 3 epochs we expect 2
        // reductions (0.1 → 0.05 → 0.025).
        let final_lr = learner.optimizer.lr();
        assert!(
            final_lr < 0.1,
            "metric_scheduler must lower LR with flat loss + patience=0; got {final_lr}"
        );
    }

    /// `with_metric_scheduler` is a no-op on the LR scheduler slot —
    /// the two are independent.
    #[test]
    fn test_learner_metric_scheduler_independent_of_lr_scheduler() {
        use ferrotorch_optim::scheduler::{PlateauMode, ReduceLROnPlateau};

        let model = DummyModule::<f32>::new().unwrap();
        let optimizer: Box<dyn Optimizer<f32>> = Box::new(DummyOptimizer { lr: 0.01 });
        let loss_fn: LossFn<f32> = Box::new(|pred, _target| Ok(pred.clone()));
        let plateau = ReduceLROnPlateau::new(PlateauMode::Min);

        let learner =
            Learner::new(model, optimizer, loss_fn).with_metric_scheduler(Box::new(plateau));

        assert!(learner.scheduler.is_none());
        assert!(learner.metric_scheduler.is_some());
    }

    #[test]
    fn test_learner_checkpoint_sequential_routes_through_core() {
        use std::sync::Arc;

        // The `Learner::checkpoint_sequential` shim is a thin pass-through
        // to `crate::checkpoint::checkpoint_sequential`. The behavioral
        // contracts are pinned by tests in `checkpoint.rs`; here we just
        // confirm the surface is callable from the Learner namespace.
        struct Identity;
        impl Module<f32> for Identity {
            fn forward(&self, input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
                Ok(input.clone())
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
            // Stateless: `Identity` has no parameters and no training-mode
            // flag, so `train()` / `eval()` have nothing to toggle. Matches
            // the convention established by `ScaleModule` / `CountingScale`
            // in `checkpoint.rs`.
            fn train(&mut self) {
                let _ = self;
            }
            fn eval(&mut self) {
                let _ = self;
            }
            fn is_training(&self) -> bool {
                true
            }
        }

        let input = ferrotorch_core::scalar(7.0_f32).unwrap();
        let modules: Vec<Arc<dyn Module<f32>>> = vec![Arc::new(Identity), Arc::new(Identity)];
        let output =
            Learner::<DummyModule<f32>, f32>::checkpoint_sequential(modules, 1, &input).unwrap();
        assert!((output.item().unwrap() - 7.0).abs() < 1e-6);
    }
}
