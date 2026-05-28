//! Re-audit of commit `9bdea4fe0` — EmbeddingBag scale_grad_by_freq+psw weight-grad
//! "sorted-neighbour count" fix (#1618).
//!
//! Commit `9bdea4fe0` replaced ferrotorch's per-index-own-count divisor in
//! `EmbeddingBagSumWeightedBackward` with torch's quirky sorted-unique-iteration
//! divisor: for the k-th unique step over the SORTED indices, torch divides that
//! step's index grad by `counts[indices_data[k]]` = `counts[sorted[k]]`, NOT by
//! the index's own occurrence count (`aten/src/ATen/native/EmbeddingBag.cpp:1569-1571`,
//! indexed by the unique-iteration counter `i`; sort at `:1522`; unique-step stride
//! `i += counts[indices_data[i]]` at `:1499`; `compute_counts` at `:1475-1478`).
//!
//! The original #1618 fix-verification test (`divergence_1610_*::
//! divergence_psw_sgbf_count_uses_sorted_neighbor`) pins only ONE pattern,
//! `input=[0,1,0]`, where the quirk happens to coincide with a particular value.
//! This file is the GENERALIZATION audit: it pins quirk-ACTIVE patterns where the
//! sorted-neighbour divisor differs from BOTH a naive per-index count AND from the
//! single pinned case, so a fix that merely memorised `[0,1,0]` would fail here.
//!
//! Every expected value below was produced by LIVE torch 2.11.0+cu130 (2026-05-28)
//! via `torch.nn.functional.embedding_bag(input, weight, offsets, mode='sum',
//! per_sample_weights=psw, scale_grad_by_freq=True).backward(ones_like(out))` —
//! never copied from the ferrotorch side (R-CHAR-3). The python oracle harness is
//! reproduced in the per-test doc comments.
//!
//! These tests are expected to PASS on `9bdea4fe0` (the fix is faithful) and to
//! FAIL on the pre-fix per-index implementation — they are the artifact proving the
//! fix generalizes rather than overfits, and a permanent regression guard against a
//! revert to per-index counting.

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::{EmbeddingBag, EmbeddingBagMode};

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

/// CASE A — the cleanest quirk demonstration. `input=[2,0,2,1,1,1]`, single bag.
///
/// counts = {0:1, 1:3, 2:2}; sorted = [0,1,1,1,2,2].
/// Unique steps: k=0 → index 0, div=counts[sorted[0]]=counts[0]=1;
///               k=1 → index 1, div=counts[sorted[1]]=counts[1]=3;
///               k=2 → index 2, div=counts[sorted[4=2-block start]]... here the
///               quirk: div=counts[sorted[2]]=counts[1]=3, NOT counts[2]=2.
/// So index 2's grad is divided by 3 even though index 2 appears only twice.
///
/// Live torch 2.11 (psw all 1, grad_output ones[1,2]):
///   out      = [[20, 26]]
///   grad_W   = [[1,1],[1,1],[0.6667,0.6667]]   (row 2 = 2/3, the quirk)
///   grad_psw = [11,3,11,7,7,7]
///
/// A naive per-index fix would give grad_W row 2 = 2/2 = [1,1] — this test pins
/// the quirk value, so it FAILS the pre-#1618 implementation and proves the fix
/// is not overfit to the `[0,1,0]` pinned case.
///
/// Tracking: #1618
#[test]
fn sgbf_quirk_index_divided_by_sorted_neighbor_count() {
    let bag = bag_from_rows(
        &[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]],
        EmbeddingBagMode::Sum,
    )
    .with_scale_grad_by_freq(true);
    let inp = index_tensor(&[2.0, 0.0, 2.0, 1.0, 1.0, 1.0]);
    let offs = [0usize];
    let psw = psw_tensor(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

    let out = bag.forward_bag_weighted(&inp, &offs, Some(&psw)).unwrap();
    let od = out.data().unwrap();
    let expect_out = [20.0, 26.0];
    for (i, &e) in expect_out.iter().enumerate() {
        assert!((od[i] - e).abs() < 1e-4, "out[{i}]={} exp {e}", od[i]);
    }

    let grad_output = grad_tensor(&[1.0, 1.0], &[1, 2]);
    let grads = out.grad_fn().unwrap().backward(&grad_output).unwrap();

    let gw = grads[0].as_ref().unwrap().data().unwrap();
    // row 2 = 0.6667 is THE quirk: index 2 (count 2) divided by counts[sorted[2]]=counts[1]=3.
    let expect_w = [1.0, 1.0, 1.0, 1.0, 2.0 / 3.0, 2.0 / 3.0];
    for (i, &e) in expect_w.iter().enumerate() {
        assert!(
            (gw[i] - e).abs() < 1e-4,
            "grad_W[{i}]={} exp {e} (torch sorted-neighbour quirk: index 2 / counts[sorted[2]]=3)",
            gw[i]
        );
    }

    let gp = grads[1].as_ref().unwrap().data().unwrap();
    let expect_psw = [11.0, 3.0, 11.0, 7.0, 7.0, 7.0];
    for (i, &e) in expect_psw.iter().enumerate() {
        assert!((gp[i] - e).abs() < 1e-4, "grad_psw[{i}]={} exp {e}", gp[i]);
    }
}

/// CASE C — quirk-active WITH varied per_sample_weights across two bags, and its
/// `scale_grad_by_freq=False` twin to confirm the freq scaling touches ONLY
/// grad_weight (grad_psw must be identical with the flag on or off).
///
/// W=[[1,2],[3,4],[5,6]], input=[2,2,0,1,1], offsets=[0,3] (bag0={2,2,0}, bag1={1,1}),
/// psw=[2,3,4,5,6], grad_output=ones[2,2].
/// counts = {0:1, 1:2, 2:2}; sorted = [0,1,1,2,2].
///   k=0 → index 0, div=counts[sorted[0]]=counts[0]=1
///   k=1 → index 1, div=counts[sorted[1]]=counts[1]=2
///   k=2 → index 2, div=counts[sorted[2]]=counts[1]=2  (coincidentally == counts[2])
///
/// Live torch 2.11:
///   sgbf=True : grad_W=[[4,4],[5.5,5.5],[2.5,2.5]], grad_psw=[11,11,3,7,7]
///   sgbf=False: grad_W=[[4,4],[11,11],[5,5]],       grad_psw=[11,11,3,7,7]  (psw identical)
///
/// Tracking: #1618
#[test]
fn sgbf_psw_twobag_affects_only_weight_grad_not_psw() {
    let rows: [&[f32]; 3] = [&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]];
    let inp = index_tensor(&[2.0, 2.0, 0.0, 1.0, 1.0]);
    let offs = [0usize, 3];
    let psw_vals = [2.0f32, 3.0, 4.0, 5.0, 6.0];

    // --- scale_grad_by_freq = True ---
    let bag_on = bag_from_rows(&rows, EmbeddingBagMode::Sum).with_scale_grad_by_freq(true);
    let psw_on = psw_tensor(&psw_vals);
    let out_on = bag_on
        .forward_bag_weighted(&inp, &offs, Some(&psw_on))
        .unwrap();
    let go = grad_tensor(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads_on = out_on.grad_fn().unwrap().backward(&go).unwrap();
    let gw_on = grads_on[0].as_ref().unwrap().data().unwrap();
    let expect_w_on = [4.0, 4.0, 5.5, 5.5, 2.5, 2.5];
    for (i, &e) in expect_w_on.iter().enumerate() {
        assert!(
            (gw_on[i] - e).abs() < 1e-4,
            "sgbf=True grad_W[{i}]={} exp {e}",
            gw_on[i]
        );
    }
    let gp_on = grads_on[1].as_ref().unwrap().data().unwrap();
    let expect_psw = [11.0, 11.0, 3.0, 7.0, 7.0];
    for (i, &e) in expect_psw.iter().enumerate() {
        assert!(
            (gp_on[i] - e).abs() < 1e-4,
            "sgbf=True grad_psw[{i}]={} exp {e}",
            gp_on[i]
        );
    }

    // --- scale_grad_by_freq = False (same input/psw) ---
    let bag_off = bag_from_rows(&rows, EmbeddingBagMode::Sum);
    let psw_off = psw_tensor(&psw_vals);
    let out_off = bag_off
        .forward_bag_weighted(&inp, &offs, Some(&psw_off))
        .unwrap();
    let go2 = grad_tensor(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let grads_off = out_off.grad_fn().unwrap().backward(&go2).unwrap();
    let gw_off = grads_off[0].as_ref().unwrap().data().unwrap();
    let expect_w_off = [4.0, 4.0, 11.0, 11.0, 5.0, 5.0];
    for (i, &e) in expect_w_off.iter().enumerate() {
        assert!(
            (gw_off[i] - e).abs() < 1e-4,
            "sgbf=False grad_W[{i}]={} exp {e}",
            gw_off[i]
        );
    }
    let gp_off = grads_off[1].as_ref().unwrap().data().unwrap();
    // grad_psw MUST be identical whether sgbf is on or off (the freq scaling is
    // absent from torch's psw-backward kernel, EmbeddingBag.cpp:1716-1724).
    for (i, &e) in expect_psw.iter().enumerate() {
        assert!(
            (gp_off[i] - e).abs() < 1e-4,
            "sgbf=False grad_psw[{i}]={} exp {e} (psw must match sgbf=True)",
            gp_off[i]
        );
    }
}

/// CASE B — larger vocab, quirk-active, all-distinct-mixed pattern.
/// W is 5x2, input=[3,0,4,0,3,1,2], single bag, psw all 1.
/// counts={0:2,1:1,2:1,3:2,4:1}; sorted=[0,0,1,2,3,3,4].
///   k=0 index 0 div=counts[sorted[0]]=counts[0]=2
///   k=1 index 1 div=counts[sorted[1]]=counts[0]=2  (quirk: index 1 count 1 / 2)
///   k=2 index 2 div=counts[sorted[2]]=counts[1]=1
///   k=3 index 3 div=counts[sorted[3]]=counts[2]=1  (quirk: index 3 count 2 / 1)
///   k=4 index 4 div=counts[sorted[4]]=counts[3]=2  (quirk: index 4 count 1 / 2)
///
/// Live torch 2.11 grad_W col0 = [1.0, 0.5, 1.0, 2.0, 0.5];
/// grad_psw = [15,3,19,3,15,7,11].
///
/// Tracking: #1618
#[test]
fn sgbf_quirk_larger_vocab_mixed_pattern() {
    let bag = bag_from_rows(
        &[
            &[1.0, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
            &[7.0, 8.0],
            &[9.0, 10.0],
        ],
        EmbeddingBagMode::Sum,
    )
    .with_scale_grad_by_freq(true);
    let inp = index_tensor(&[3.0, 0.0, 4.0, 0.0, 3.0, 1.0, 2.0]);
    let offs = [0usize];
    let psw = psw_tensor(&[1.0; 7]);

    let out = bag.forward_bag_weighted(&inp, &offs, Some(&psw)).unwrap();
    let grad_output = grad_tensor(&[1.0, 1.0], &[1, 2]);
    let grads = out.grad_fn().unwrap().backward(&grad_output).unwrap();

    let gw = grads[0].as_ref().unwrap().data().unwrap();
    // col0 (every other element) expected from live torch.
    let expect_col0 = [1.0, 0.5, 1.0, 2.0, 0.5];
    for (row, &e) in expect_col0.iter().enumerate() {
        let v = gw[row * 2];
        assert!(
            (v - e).abs() < 1e-4,
            "grad_W[row {row}][0]={v} exp {e} (sorted-neighbour quirk)",
        );
    }

    let gp = grads[1].as_ref().unwrap().data().unwrap();
    let expect_psw = [15.0, 3.0, 19.0, 3.0, 15.0, 7.0, 11.0];
    for (i, &e) in expect_psw.iter().enumerate() {
        assert!((gp[i] - e).abs() < 1e-4, "grad_psw[{i}]={} exp {e}", gp[i]);
    }
}
