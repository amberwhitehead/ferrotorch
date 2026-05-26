//! `ferrotorch-optim` — crate root.
//!
//! Declares every leaf submodule (optimizer family, LR-scheduler tree,
//! AMP / `GradScaler`, `GradientAccumulator`, differentiable-step helpers,
//! `foreach_utils`, `ParamKey`) and re-exports their public surface so
//! downstream crates (`ferrotorch-train`, `ferrotorch`, example /
//! benchmark binaries) consume the optimizer family through
//! `ferrotorch_optim::*` without spelunking into submodule paths.
//! Mirrors `torch/optim/__init__.py`'s flat re-export pattern.
//!
//! ## REQ status (per `.design/ferrotorch-optim/lib.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | the 25 `pub mod` lines below; consumed by `ferrotorch-train/src/learner.rs:28` `use ferrotorch_optim::Optimizer;` |
//! | REQ-2 | SHIPPED | `pub use adam::{Adam, AdamConfig}` etc.; consumed by `ferrotorch-train/examples/multi_epoch_train_dump.rs:63` `use ferrotorch_optim::{Adam, AdamConfig, Optimizer};` |
//! | REQ-3 | SHIPPED | `pub use optimizer::{Optimizer, OptimizerState, ParamGroup}`; consumed by `ferrotorch-train/src/learner.rs:59` `optimizer: Box<dyn Optimizer<T>>` |
//! | REQ-4 | SHIPPED | `pub use scheduler::{...}` re-exports the `LrScheduler` family; consumed by `ferrotorch-train/src/learner.rs:30` `use ferrotorch_optim::scheduler::LrScheduler;` |
//! | REQ-5 | SHIPPED | `pub use grad_scaler::{GradScaler, GradScalerConfig, GradScalerState}`; consumed by `ferrotorch-train/src/amp.rs:55` `pub use ferrotorch_optim::{GradScaler, GradScalerConfig, GradScalerState};` |
//! | REQ-6 | SHIPPED | `pub use grad_accumulator::GradientAccumulator` + `pub use differentiable::{diff_sgd_momentum_step, diff_sgd_step}`; boundary-method public API per goal.md S5 |
//! | REQ-7 | SHIPPED | `pub use param_key::ParamKey`; consumed by `ferrotorch-optim/src/adamw.rs:22` `use crate::param_key::ParamKey;` and 5 other optimizer files |

pub mod adadelta;
pub mod adafactor;
pub mod adagrad;
pub mod adam;
pub mod adamax;
pub mod adamw;
pub mod asgd;
pub mod differentiable;
pub mod ema;
pub mod foreach_utils;
pub mod grad_accumulator;
pub mod grad_scaler;
pub mod lbfgs;
pub mod muon;
pub mod nadam;
pub mod natural_gradient;
pub mod optimizer;
pub mod param_key;
pub mod radam;
pub mod rmsprop;
pub mod rprop;
pub mod scheduler;
pub mod sgd;
pub mod sparse_adam;
pub mod swa;

pub use adadelta::{Adadelta, AdadeltaConfig};
pub use adafactor::{Adafactor, AdafactorConfig};
pub use adagrad::{Adagrad, AdagradConfig};
pub use adam::{Adam, AdamConfig};
pub use adamax::{Adamax, AdamaxConfig};
pub use adamw::{AdamW, AdamWConfig};
pub use asgd::{Asgd, AsgdConfig};
pub use differentiable::{diff_sgd_momentum_step, diff_sgd_step};
pub use ema::ExponentialMovingAverage;
pub use grad_accumulator::GradientAccumulator;
pub use grad_scaler::{GradScaler, GradScalerConfig, GradScalerState};
pub use lbfgs::{Lbfgs, LbfgsConfig, LineSearchFn};
pub use muon::{Muon, MuonConfig};
pub use nadam::{NAdam, NAdamConfig};
pub use natural_gradient::{Kfac, KfacConfig};
pub use optimizer::{Optimizer, OptimizerState, ParamGroup};
pub use param_key::ParamKey;
pub use radam::{RAdam, RAdamConfig};
pub use rmsprop::{Rmsprop, RmspropConfig};
pub use rprop::{Rprop, RpropConfig};
pub use scheduler::{
    AnnealStrategy, ChainedScheduler, ConstantLR, CosineAnnealingLR, CosineAnnealingWarmRestarts,
    CyclicLR, CyclicMode, ExponentialLR, LambdaLR, LinearLR, LinearWarmup, LrScheduler,
    MetricScheduler, MultiStepLR, MultiplicativeLR, OneCycleLR, PlateauMode, PolynomialLR,
    ReduceLROnPlateau, SequentialLr, StepLR, ThresholdMode, cosine_warmup_scheduler,
};
pub use sgd::{Sgd, SgdConfig};
pub use sparse_adam::{SparseAdam, SparseAdamConfig};
pub use swa::{AveragedModel, AveragingStrategy, Swalr};
