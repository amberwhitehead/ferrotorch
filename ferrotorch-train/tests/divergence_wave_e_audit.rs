//! Wave-E audit (#1542): Learner-attached metrics (#1494/#1495/#1496).
//!
//! Builds a tiny Learner around a Linear(C, C) model, attaches
//! `AccuracyMetric` + `TopKAccuracy` + `RunningAverage` via the new builder
//! methods, runs a 1-epoch fit, and asserts the metric values are recorded
//! in `EpochResult::metrics` AND surface via `learner.metric_snapshot()`.

#![allow(clippy::approx_constant)]

use ferrotorch_core::{FerrotorchResult, Tensor, from_vec};
use ferrotorch_nn::{Linear, Module, Parameter};
use ferrotorch_optim::{Optimizer, Sgd, SgdConfig};
use ferrotorch_train::learner::{ClassificationAdapter, Learner, LossFn};
use ferrotorch_train::{AccuracyMetric, Metric, RunningAverage, TopKAccuracy};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const C: usize = 3;

fn linear_cc() -> FerrotorchResult<Linear<f32>> {
    let mut layer = Linear::<f32>::new(C, C, false)?;
    let mut data = vec![0.0_f32; C * C];
    for i in 0..C {
        data[i * C + i] = 1.0;
    }
    layer.weight.set_data(from_vec(data, &[C, C])?);
    Ok(layer)
}

fn one_hot_batches() -> Vec<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>> {
    let b1 = (
        from_vec(vec![0.9, 0.1, 0.0, 0.2, 0.7, 0.1], &[2, C]).unwrap(),
        from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, C]).unwrap(),
    );
    let b2 = (
        from_vec(vec![0.1, 0.8, 0.1, 0.3, 0.3, 0.4], &[2, C]).unwrap(),
        from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[2, C]).unwrap(),
    );
    vec![Ok(b1), Ok(b2)]
}

fn mse_loss() -> LossFn<f32> {
    #[allow(clippy::redundant_closure)]
    Box::new(|pred, target| ferrotorch_nn::functional::mse_loss(pred, target))
}

fn argmax_adapter() -> ClassificationAdapter<f32> {
    Box::new(
        |pred: &Tensor<f32>, target: &Tensor<f32>| -> FerrotorchResult<(usize, usize)> {
            let p = pred.data_vec()?;
            let t = target.data_vec()?;
            let n = pred.shape()[0];
            let c = pred.shape()[1];
            let mut correct = 0;
            for i in 0..n {
                let p_row = &p[i * c..(i + 1) * c];
                let t_row = &t[i * c..(i + 1) * c];
                let pa = p_row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                let ta = t_row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                if pa == ta {
                    correct += 1;
                }
            }
            Ok((correct, n))
        },
    )
}

fn topk_adapter(k: usize) -> ClassificationAdapter<f32> {
    Box::new(
        move |pred: &Tensor<f32>, target: &Tensor<f32>| -> FerrotorchResult<(usize, usize)> {
            let p = pred.data_vec()?;
            let t = target.data_vec()?;
            let n = pred.shape()[0];
            let c = pred.shape()[1];
            let mut correct = 0;
            for i in 0..n {
                let p_row = &p[i * c..(i + 1) * c];
                let t_row = &t[i * c..(i + 1) * c];
                let ta = t_row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                let mut indexed: Vec<(usize, f32)> =
                    p_row.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let top: Vec<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();
                if top.contains(&ta) {
                    correct += 1;
                }
            }
            Ok((correct, n))
        },
    )
}

// ---------------------------------------------------------------------------
// #1494 — AccuracyMetric attached + observed in EpochResult.
// ---------------------------------------------------------------------------
#[test]
fn audit_1494_accuracy_metric_attached_and_updated() {
    let layer = linear_cc().expect("linear");
    let params: Vec<Parameter<f32>> =
        layer.parameters().iter().map(|p| (*p).clone()).collect();
    let optimizer: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params, SgdConfig::new(0.01)));
    let mut learner = Learner::new(layer, optimizer, mse_loss())
        .with_accuracy_metric(AccuracyMetric::new(), argmax_adapter());

    let data_fn = || one_hot_batches().into_iter();
    let val_fn =
        None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>;
    let history = learner.fit(&data_fn, val_fn, 1).expect("fit");

    let epoch = history.epochs.first().expect("at least one epoch");
    assert!(
        epoch.metrics.contains_key("train_accuracy"),
        "epoch metrics must contain `train_accuracy`, got: {:?}",
        epoch.metrics.keys().collect::<Vec<_>>()
    );
    let acc = epoch.metrics["train_accuracy"];
    assert!(
        (acc - 1.0).abs() < 1e-9,
        "expected train_accuracy ≈ 1.0, got {acc}"
    );
}

// ---------------------------------------------------------------------------
// #1495 — TopKAccuracy attached + observed in EpochResult.
// ---------------------------------------------------------------------------
#[test]
fn audit_1495_topk_accuracy_metric_attached() {
    let layer = linear_cc().expect("linear");
    let params: Vec<Parameter<f32>> =
        layer.parameters().iter().map(|p| (*p).clone()).collect();
    let optimizer: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params, SgdConfig::new(0.01)));
    let mut learner = Learner::new(layer, optimizer, mse_loss())
        .with_topk_accuracy_metric(TopKAccuracy::new(3), topk_adapter(3));

    let data_fn = || one_hot_batches().into_iter();
    let val_fn =
        None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>;
    let history = learner.fit(&data_fn, val_fn, 1).expect("fit");

    let epoch = history.epochs.first().expect("epoch present");
    assert!(
        epoch.metrics.contains_key("train_top_k_accuracy"),
        "epoch metrics must contain `train_top_k_accuracy`, got: {:?}",
        epoch.metrics.keys().collect::<Vec<_>>()
    );
    let v = epoch.metrics["train_top_k_accuracy"];
    assert!((v - 1.0).abs() < 1e-9, "expected top-3 accuracy ≈ 1.0, got {v}");
}

// ---------------------------------------------------------------------------
// #1496 — RunningAverage attached + sliding window.
// ---------------------------------------------------------------------------
#[test]
fn audit_1496_running_avg_metric_attached() {
    let layer = linear_cc().expect("linear");
    let params: Vec<Parameter<f32>> =
        layer.parameters().iter().map(|p| (*p).clone()).collect();
    let optimizer: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params, SgdConfig::new(0.01)));
    let mut learner = Learner::new(layer, optimizer, mse_loss())
        .with_running_average_metric(RunningAverage::new(8));

    let data_fn = || one_hot_batches().into_iter();
    let val_fn =
        None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>;
    let history = learner.fit(&data_fn, val_fn, 1).expect("fit");

    let epoch = history.epochs.first().expect("epoch present");
    assert!(
        epoch.metrics.contains_key("running_avg"),
        "epoch metrics must contain `running_avg` (no train_/val_ prefix), got: {:?}",
        epoch.metrics.keys().collect::<Vec<_>>()
    );
    let v = epoch.metrics["running_avg"];
    assert!(
        v.is_finite() && v > 0.0,
        "running_avg must be finite > 0, got {v}"
    );
}

// ---------------------------------------------------------------------------
// metric_snapshot() smoke — all three present after fit.
// ---------------------------------------------------------------------------
#[test]
fn audit_1494_1495_1496_metric_snapshot_contains_all_three() {
    let layer = linear_cc().expect("linear");
    let params: Vec<Parameter<f32>> =
        layer.parameters().iter().map(|p| (*p).clone()).collect();
    let optimizer: Box<dyn Optimizer<f32>> = Box::new(Sgd::new(params, SgdConfig::new(0.01)));
    let mut learner = Learner::new(layer, optimizer, mse_loss())
        .with_accuracy_metric(AccuracyMetric::new(), argmax_adapter())
        .with_topk_accuracy_metric(TopKAccuracy::new(2), topk_adapter(2))
        .with_running_average_metric(RunningAverage::new(8));

    let data_fn = || one_hot_batches().into_iter();
    let val_fn =
        None::<&dyn Fn() -> std::vec::IntoIter<FerrotorchResult<(Tensor<f32>, Tensor<f32>)>>>;
    let _ = learner.fit(&data_fn, val_fn, 1).expect("fit");

    let snap = learner.metric_snapshot();
    let names: Vec<String> = snap.iter().map(|(n, _)| n.clone()).collect();
    assert!(
        names.iter().any(|n| n == "accuracy"),
        "snapshot must contain `accuracy`, got {names:?}"
    );
    assert!(
        names.iter().any(|n| n == "top_k_accuracy"),
        "snapshot must contain `top_k_accuracy`, got {names:?}"
    );
    assert!(
        names.iter().any(|n| n == "running_avg"),
        "snapshot must contain `running_avg`, got {names:?}"
    );
}

// ---------------------------------------------------------------------------
// Trait surface — basic name() smoke.
// ---------------------------------------------------------------------------
#[test]
fn audit_metric_builder_methods_exist() {
    let _ = AccuracyMetric::new();
    let _ = TopKAccuracy::new(3);
    let _ = RunningAverage::new(8);
    let acc = AccuracyMetric::new();
    let n: &str = acc.name();
    assert_eq!(n, "accuracy");
}
