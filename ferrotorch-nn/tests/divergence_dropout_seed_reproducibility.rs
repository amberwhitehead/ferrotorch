//! Regression coverage for `ferrotorch_nn::functional::dropout`'s torch-seed
//! reproducibility (closes #1634; originally a re-audit of commit 8af6a065e /
//! #1441 dropout arm).
//!
//! These tests WERE failing-pinned divergences (#1634): the CPU dropout mask
//! drawn from `ferrotorch_core::rng` (MT19937) via `with_thread_rng` /
//! `next_uniform_f64` used the WRONG Bernoulli comparison (`u < p`, keep on
//! `u >= p`) instead of torch's `noise.bernoulli_(1 - p)` semantics
//! (keep iff `u < (1 - p)`, `aten/src/ATen/native/Dropout.cpp:74`,
//! `aten/src/ATen/native/cpu/DistributionTemplates.h:388-399`,
//! `TransformationHelper.h:171-173`). With the comparison corrected the mask
//! is byte-identical to torch under a shared seed, so the `#[ignore]` is
//! removed and these now serve as permanent regression coverage.
//!
//! Reference values produced by LIVE torch 2.11.0+cu130 (R-CHAR-3 — NOT copied
//! from the ferrotorch side):
//!     torch.manual_seed(0); F.dropout(torch.ones(10), p=0.5, training=True)
//!         -> [0, 0, 2, 0, 0, 0, 2, 2, 0, 2]
//!     torch.manual_seed(1); F.dropout(torch.ones(10), p=0.5, training=True)
//!         -> [2, 2, 2, 2, 0, 2, 2, 0, 0, 2]
//! Upstream: `torch/nn/functional.py:1425 dropout(...)` ->
//! `aten/src/ATen/native/Dropout.cpp` native_dropout (CPU bernoulli mask gen).

use ferrotorch_core::{Tensor, TensorStorage};

fn ones(n: usize) -> Tensor<f32> {
    Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0f32; n]), vec![n], false).unwrap()
}

/// Regression (#1634): ferrotorch's `functional::dropout` mask is byte-identical
/// to `torch.manual_seed(s); F.dropout(...)` under a shared seed — same elements
/// zeroed, same `1/(1-p)` scale on survivors.
#[test]
fn divergence_dropout_seed0_mask_matches_torch() {
    // Live-torch reference for torch.manual_seed(0); F.dropout(ones(10),0.5,True).
    let torch_seed0: [f32; 10] = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0];

    ferrotorch_core::rng::manual_seed(0).unwrap();
    let y = ferrotorch_nn::functional::dropout(&ones(10), 0.5, true).unwrap();
    let got = y.data_vec().unwrap();

    assert_eq!(
        got, torch_seed0,
        "dropout mask under manual_seed(0) must match torch's seeded mask \
         (the functional.rs:351 'matches torch.manual_seed(s); F.dropout' claim)"
    );
}

/// Regression (#1634): same as above for seed=1.
#[test]
fn divergence_dropout_seed1_mask_matches_torch() {
    let torch_seed1: [f32; 10] = [2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0];

    ferrotorch_core::rng::manual_seed(1).unwrap();
    let y = ferrotorch_nn::functional::dropout(&ones(10), 0.5, true).unwrap();
    let got = y.data_vec().unwrap();

    assert_eq!(
        got, torch_seed1,
        "dropout mask under manual_seed(1) must match torch's seeded mask"
    );
}

/// Regression (#1634): the `Dropout` MODULE (struct) CPU forward path — which
/// the fix rewired from xorshift64 to the byte-exact MT19937 generator — also
/// reproduces `torch.manual_seed(0); F.dropout(ones(10), 0.5, True)`
/// byte-for-byte. Guards the `<Dropout as Module>::forward` CPU branch, not just
/// the stateless `functional::dropout`.
#[test]
fn dropout_module_seed0_mask_matches_torch() {
    use ferrotorch_nn::Module;

    let torch_seed0: [f32; 10] = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0];

    ferrotorch_core::rng::manual_seed(0).unwrap();
    let layer = ferrotorch_nn::Dropout::<f32>::new(0.5).unwrap();
    let y = layer.forward(&ones(10)).unwrap();
    let got = y.data_vec().unwrap();

    assert_eq!(
        got, torch_seed0,
        "Dropout module forward under manual_seed(0) must match torch's seeded mask \
         (CPU path rewired to MT19937 generator, #1634)"
    );
}

/// NOT a divergence — pins that the STATISTICAL/structural behavior of
/// ferrotorch dropout IS correct (this guards against the #1634 fix
/// accidentally breaking the property contract): every output element is
/// either 0 or exactly `1/(1-p)` (inverted-dropout scale), per
/// `aten/src/ATen/native/Dropout.cpp` (mask * input * 1/(1-p)).
/// This test PASSES today and must keep passing.
#[test]
fn dropout_structural_inverted_scale_holds() {
    ferrotorch_core::rng::manual_seed(0).unwrap();
    let p = 0.25_f64;
    let y = ferrotorch_nn::functional::dropout(&ones(64), p, true).unwrap();
    let scale = (1.0 / (1.0 - p)) as f32;
    for (i, &e) in y.data_vec().unwrap().iter().enumerate() {
        assert!(
            e == 0.0 || (e - scale).abs() < 1e-5,
            "element {i} = {e} not in {{0, {scale}}} (inverted-dropout structure)"
        );
    }
}
