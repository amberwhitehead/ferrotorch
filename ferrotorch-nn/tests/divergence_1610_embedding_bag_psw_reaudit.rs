//! Re-audit of commit `5e888d42f` — EmbeddingBag per_sample_weights (#1610).
//!
//! The builder shipped `EmbeddingBag::forward_bag_weighted` (sum-mode-only
//! per-sample row scaling) with `EmbeddingBagSumWeightedBackward` flowing grad
//! to BOTH the embedding table and `per_sample_weights`. The +7 lib tests cover
//! single-bag / two-bag / padding / end-to-end / mode-reject / shape-mismatch,
//! but NOT the `scale_grad_by_freq=true` + psw composition.
//!
//! These tests pin the `scale_grad_by_freq` + psw composition against LIVE
//! torch 2.11.0+cu130 (2026-05-28). All expected values were produced by
//! `torch.nn.functional.embedding_bag(input, weight, offsets, mode='sum',
//! per_sample_weights=psw, scale_grad_by_freq=True).backward(grad_output)` —
//! never copied from the ferrotorch side (R-CHAR-3).
//!
//! The dominant finding (`divergence_psw_sgbf_count_uses_sorted_neighbor`,
//! tracking #1618) is that torch's CPU dense-backward divides the per-INDEX
//! weight gradient using `counts[indices_data[i]]` where `i` is the
//! *unique-index iteration counter* over the SORTED indices, NOT `counts[index]`.
//! For an input whose indices are not already sorted this divides one index's
//! gradient by a *neighboring* index's frequency. ferrotorch divides each index
//! by its own frequency, so the two diverge.
//!
//! Upstream sites:
//!   - `aten/src/ATen/native/EmbeddingBag.cpp:1522` — `indices_.sort()` (the
//!     backward operates on SORTED indices + the corresponding `counts`).
//!   - `aten/src/ATen/native/EmbeddingBag.cpp:1569-1571` — `if (scale_grad_by_freq)
//!     { scale /= counts[indices_data[i]]; }` (note: `indices_data[i]`, indexed by
//!     the unique-iteration counter `i`, not by `index`).
//!   - ferrotorch `ferrotorch-nn/src/embedding.rs:138-149` builds a HashMap of
//!     per-index occurrence counts and at `:165-171` divides by
//!     `counts[idx]` (each index's OWN count).

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::{EmbeddingBag, EmbeddingBagMode};

/// Build a sum-mode `EmbeddingBag<f32>` with the given pretrained rows,
/// installing the weight through the public `Module::parameters_mut` surface.
fn bag_from_rows(rows: &[&[f32]], mode: EmbeddingBagMode) -> EmbeddingBag<f32> {
    let dim = rows[0].len();
    let mut data = Vec::new();
    for r in rows {
        data.extend_from_slice(r);
    }
    let mut bag = EmbeddingBag::<f32>::new(rows.len(), dim, mode).unwrap();
    let weight =
        Tensor::from_storage(TensorStorage::cpu(data), vec![rows.len(), dim], true).unwrap();
    bag.parameters_mut()[0].set_data(weight);
    bag
}

fn index_tensor(idx: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(idx.to_vec()), vec![idx.len()], false).unwrap()
}

fn psw_tensor(w: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(w.to_vec()), vec![w.len()], true).unwrap()
}

fn grad_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// Divergence: ferrotorch's `EmbeddingBagSumWeightedBackward` (sum mode,
/// `scale_grad_by_freq=true`, with per_sample_weights) diverges from
/// `pytorch aten/src/ATen/native/EmbeddingBag.cpp:1569-1571` for an input
/// whose indices are NOT pre-sorted.
///
/// Setup (oracle = live torch 2.11.0+cu130, 2026-05-28):
///   W=[[1,2],[3,4],[5,6]], input=[0,1,0], offsets=[0,2] (bag0={0,1}, bag1={0}),
///   psw=[2,3,4], grad_output=[[1,1],[2,2]], mode='sum', scale_grad_by_freq=True.
///
/// Upstream sorts indices to [0,0,1] (`EmbeddingBag.cpp:1522`) and, at the
/// unique-iteration step for index 1, divides by `counts[indices_data[1]]` =
/// counts[sorted[1]] = counts[0] = 2 (the quirk). Upstream therefore returns
///   grad_W = [[5,5],[1.5,1.5],[0,0]].
/// ferrotorch divides index 1 by its OWN count (=1), returning
///   grad_W = [[5,5],[3,3],[0,0]].
/// grad_psw is unaffected by scale_grad_by_freq and matches in both: [3,7,6].
///
/// Tracking: #1618
#[test]
fn divergence_psw_sgbf_count_uses_sorted_neighbor() {
    let bag = bag_from_rows(
        &[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]],
        EmbeddingBagMode::Sum,
    )
    .with_scale_grad_by_freq(true);
    let inp = index_tensor(&[0.0, 1.0, 0.0]);
    let offs = [0usize, 2];
    let psw = psw_tensor(&[2.0, 3.0, 4.0]);

    let out = bag.forward_bag_weighted(&inp, &offs, Some(&psw)).unwrap();
    // forward parity (oracle): bag0 = 2*[1,2] + 3*[3,4] = [11,16]; bag1 = 4*[1,2] = [4,8].
    let od = out.data().unwrap();
    assert!((od[0] - 11.0).abs() < 1e-5, "out[0]={}", od[0]);
    assert!((od[1] - 16.0).abs() < 1e-5, "out[1]={}", od[1]);
    assert!((od[2] - 4.0).abs() < 1e-5, "out[2]={}", od[2]);
    assert!((od[3] - 8.0).abs() < 1e-5, "out[3]={}", od[3]);

    let grad_output = grad_tensor(&[1.0, 1.0, 2.0, 2.0], &[2, 2]);
    let grads = out.grad_fn().unwrap().backward(&grad_output).unwrap();

    // grad to embedding table (input 0). Upstream's sorted-neighbor count quirk.
    let gw = grads[0].as_ref().unwrap().data().unwrap();
    let expect_w = [5.0, 5.0, 1.5, 1.5, 0.0, 0.0]; // live torch 2.11
    for (i, &e) in expect_w.iter().enumerate() {
        assert!(
            (gw[i] - e).abs() < 1e-4,
            "grad_W[{i}]={} exp {e} (torch divides index 1 by counts[sorted[1]]=counts[0]=2)",
            gw[i]
        );
    }

    // grad to per_sample_weights (input 1) — unaffected by scale_grad_by_freq.
    let gp = grads[1].as_ref().unwrap().data().unwrap();
    let expect_psw = [3.0, 7.0, 6.0]; // live torch 2.11
    for (i, &e) in expect_psw.iter().enumerate() {
        assert!((gp[i] - e).abs() < 1e-4, "grad_psw[{i}]={} exp {e}", gp[i]);
    }
}

/// Companion (no scale_grad_by_freq): the SAME setup without the freq scaling
/// must match torch exactly. This isolates the divergence to the
/// scale_grad_by_freq path — if THIS fails too, the divergence is broader.
///
/// Oracle (live torch 2.11): grad_W=[[10,10],[3,3],[0,0]], grad_psw=[3,7,6].
#[test]
fn psw_no_sgbf_crossbag_matches_torch() {
    let bag = bag_from_rows(
        &[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]],
        EmbeddingBagMode::Sum,
    );
    let inp = index_tensor(&[0.0, 1.0, 0.0]);
    let offs = [0usize, 2];
    let psw = psw_tensor(&[2.0, 3.0, 4.0]);

    let out = bag.forward_bag_weighted(&inp, &offs, Some(&psw)).unwrap();
    let grad_output = grad_tensor(&[1.0, 1.0, 2.0, 2.0], &[2, 2]);
    let grads = out.grad_fn().unwrap().backward(&grad_output).unwrap();

    let gw = grads[0].as_ref().unwrap().data().unwrap();
    let expect_w = [10.0, 10.0, 3.0, 3.0, 0.0, 0.0];
    for (i, &e) in expect_w.iter().enumerate() {
        assert!((gw[i] - e).abs() < 1e-4, "grad_W[{i}]={} exp {e}", gw[i]);
    }
    let gp = grads[1].as_ref().unwrap().data().unwrap();
    let expect_psw = [3.0, 7.0, 6.0];
    for (i, &e) in expect_psw.iter().enumerate() {
        assert!((gp[i] - e).abs() < 1e-4, "grad_psw[{i}]={} exp {e}", gp[i]);
    }
}

/// Regression / composition: `include_last_offset=true` + per_sample_weights,
/// two bags, distinct grad_output rows (so a mis-mapped sample->bag is caught).
///
/// Oracle (live torch 2.11): W=[[1,2],[3,4],[5,6]], input=[0,1,2],
/// offsets=[0,2,3] (CSR; last entry == total), psw=[2,3,4], mode='sum',
/// grad_output=[[1,1],[10,10]].
///   out      = [[11,16],[20,24]]
///   grad_W   = [[2,2],[3,3],[40,40]]
///   grad_psw = [3,7,110]
#[test]
fn psw_include_last_offset_matches_torch() {
    let bag = bag_from_rows(
        &[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]],
        EmbeddingBagMode::Sum,
    )
    .with_include_last_offset(true);
    let inp = index_tensor(&[0.0, 1.0, 2.0]);
    let offs = [0usize, 2, 3];
    let psw = psw_tensor(&[2.0, 3.0, 4.0]);

    let out = bag.forward_bag_weighted(&inp, &offs, Some(&psw)).unwrap();
    assert_eq!(out.shape(), &[2, 2], "include_last_offset => 2 bags");
    let od = out.data().unwrap();
    let expect_out = [11.0, 16.0, 20.0, 24.0];
    for (i, &e) in expect_out.iter().enumerate() {
        assert!((od[i] - e).abs() < 1e-4, "out[{i}]={} exp {e}", od[i]);
    }

    let grad_output = grad_tensor(&[1.0, 1.0, 10.0, 10.0], &[2, 2]);
    let grads = out.grad_fn().unwrap().backward(&grad_output).unwrap();
    let gw = grads[0].as_ref().unwrap().data().unwrap();
    let expect_w = [2.0, 2.0, 3.0, 3.0, 40.0, 40.0];
    for (i, &e) in expect_w.iter().enumerate() {
        assert!((gw[i] - e).abs() < 1e-4, "grad_W[{i}]={} exp {e}", gw[i]);
    }
    let gp = grads[1].as_ref().unwrap().data().unwrap();
    let expect_psw = [3.0, 7.0, 110.0];
    for (i, &e) in expect_psw.iter().enumerate() {
        assert!((gp[i] - e).abs() < 1e-4, "grad_psw[{i}]={} exp {e}", gp[i]);
    }
}
