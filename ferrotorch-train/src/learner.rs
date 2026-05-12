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

use std::path::PathBuf;
use std::time::Instant;

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchResult, Float, Tensor};
use ferrotorch_nn::Module;
use ferrotorch_optim::Optimizer;
use ferrotorch_optim::grad_scaler::GradScaler;
use ferrotorch_optim::scheduler::LrScheduler;
use ferrotorch_serialize::{TrainingCheckpoint, load_checkpoint, save_checkpoint};

use crate::callback::Callback;
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
    /// Optional automatic-mixed-precision gradient scaler. When set, the
    /// training loop scales the loss before backward, unscales gradients
    /// inside `scaler.step`, skips the optimizer step on inf/NaN, and
    /// updates the scale factor at the end of each batch. (#595)
    grad_scaler: Option<GradScaler<T>>,
    train_metrics: Vec<Box<dyn Metric<Input = f64>>>,
    val_metrics: Vec<Box<dyn Metric<Input = f64>>>,
    callbacks: Vec<Box<dyn Callback<T>>>,
    checkpoint_dir: Option<PathBuf>,
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
            grad_scaler: None,
            train_metrics: Vec::new(),
            val_metrics: Vec::new(),
            callbacks: Vec::new(),
            checkpoint_dir: None,
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
    pub fn with_grad_scaler(mut self, scaler: GradScaler<T>) -> Self {
        self.grad_scaler = Some(scaler);
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

                // Forward.
                let output = self.model.forward(&input)?;
                let loss = (self.loss_fn)(&output, &target)?;
                let loss_val = cast::<T, f64>(loss.item()?)?;

                if let Some(ref mut scaler) = self.grad_scaler {
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
                    self.optimizer.step()?;
                    self.optimizer.zero_grad()?;
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
                    self.epoch,
                    self.step,
                );
                let path = dir.join(format!("checkpoint_epoch_{}.ftc", self.epoch));
                let _ = save_checkpoint(&checkpoint, &path);
            }

            let epoch_result = EpochResult {
                epoch: self.epoch,
                train_loss,
                val_loss,
                metrics,
                lr: self.optimizer.lr(),
                duration_secs: epoch_start.elapsed().as_secs_f64(),
            };

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
        let history = learner
            .fit(
                &data,
                None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>,
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
}
