//! Divergence audit for `ferrotorch_nn::functional::dropout`'s torch-seed
//! reproducibility claim (re-audit of commit 8af6a065e / #1441 dropout arm).
//!
//! `ferrotorch-nn/src/functional.rs:20` (REQ-3 "SHIPPED" row) and
//! `:269-273` / `:349-352` claim functional dropout "mirrors / matches
//! `torch.manual_seed(s); F.dropout(...)`" and "closes #1452". The mask is
//! drawn from `ferrotorch_core::rng` (MT19937) via `with_thread_rng` /
//! `next_uniform_f64`. PyTorch's CPU dropout consumes its RNG stream
//! differently (and uses Philox on CUDA), so the per-element mask is NOT
//! byte-reproducible against torch under a shared seed.
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

/// Divergence: ferrotorch's `functional::dropout` mask diverges from
/// `torch.manual_seed(s); F.dropout(...)` despite the
/// `ferrotorch-nn/src/functional.rs:351` claim that ferrotorch
/// "matches `torch.manual_seed(s); F.dropout(...)`'s contract".
/// Under a shared seed, upstream and ferrotorch zero DIFFERENT elements.
/// Tracking: #1634
#[test]
#[ignore = "divergence: dropout mask not torch-seed-reproducible (MT19937 stream consumed differently than torch CPU dropout / Philox); tracking #1634"]
fn divergence_dropout_seed0_mask_matches_torch() {
    // Live-torch reference for manual_seed(0); F.dropout(ones(10),0.5,True).
    let torch_seed0: [f32; 10] = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0];

    ferrotorch_core::rng::manual_seed(0);
    let y = ferrotorch_nn::functional::dropout(&ones(10), 0.5, true).unwrap();
    let got = y.data_vec().unwrap();

    assert_eq!(
        got, torch_seed0,
        "dropout mask under manual_seed(0) must match torch's seeded mask \
         (the functional.rs:351 'matches torch.manual_seed(s); F.dropout' claim)"
    );
}

/// Divergence: same as above for seed=1. Tracking: #1634
#[test]
#[ignore = "divergence: dropout mask not torch-seed-reproducible; tracking #1634"]
fn divergence_dropout_seed1_mask_matches_torch() {
    let torch_seed1: [f32; 10] = [2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0];

    ferrotorch_core::rng::manual_seed(1);
    let y = ferrotorch_nn::functional::dropout(&ones(10), 0.5, true).unwrap();
    let got = y.data_vec().unwrap();

    assert_eq!(
        got, torch_seed1,
        "dropout mask under manual_seed(1) must match torch's seeded mask"
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
    ferrotorch_core::rng::manual_seed(0);
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
