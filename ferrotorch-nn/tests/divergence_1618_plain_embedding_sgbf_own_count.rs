//! Re-audit of commit `9bdea4fe0` (#1618) — SIBLING-PATH verification.
//!
//! The #1618 fix is scoped to `EmbeddingBagSumWeightedBackward` (the bag+psw path,
//! governed by `EmbeddingBag.cpp` which SORTS indices before dividing — the
//! sorted-neighbour quirk). The fixer asserted that the PLAIN `Embedding`
//! `scale_grad_by_freq` path (`EmbeddingBackward`) is correct AS-IS because it is
//! governed by a DIFFERENT upstream kernel that uses each index's OWN count with
//! NO sort:
//!
//!   `aten/src/ATen/native/Embedding.cpp:155-161`:
//!     `index_t k = indices_data[i]; ... if (scale_grad_by_freq) scale /= counts[k];`
//!   where `counts[k]` is keyed by the index's OWN value `k` (built at `:144-149`),
//!   walked in original (UNsorted) order. There is no `indices_.sort()` here, so the
//!   sorted-neighbour quirk that bites EmbeddingBag does NOT apply to plain Embedding.
//!
//! This file VERIFIES that claim directly against LIVE torch 2.11.0+cu130
//! (2026-05-28), independent of per_sample_weights, with multi-duplicate patterns
//! the existing `test_scale_grad_by_freq_divides_duplicates` lib test (only
//! `[1,1,0]`) does not exercise. Expected values came from
//! `torch.nn.functional.embedding(input, weight, padding_idx=..,
//! scale_grad_by_freq=True).backward(ones_like(out))` — never copied from
//! ferrotorch (R-CHAR-3).
//!
//! Crucially this includes an ORDER-INVARIANCE check: `[3,3,3,1,0]` and
//! `[0,3,3,3,1]` (same multiset, different order) must produce IDENTICAL grads,
//! because plain Embedding has no sort step. If the plain path had silently
//! adopted the EmbeddingBag sorted-neighbour quirk, these would diverge — pinning
//! both orders catches that.
//!
//! These tests are expected to PASS on `9bdea4fe0` (the plain path was correctly
//! left untouched); a PASS discharges the fixer's sibling-path claim with a
//! runnable artifact rather than prose.

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_nn::Embedding;
use ferrotorch_nn::module::Module;

fn index_tensor(idx: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(idx.to_vec()), vec![idx.len()], false).unwrap()
}

/// 4-row table, all-ones grad_output. Returns column-0 of grad_weight.
fn plain_grad_col0(idx: &[f32], padding_idx: Option<usize>) -> Vec<f32> {
    let weight =
        Tensor::from_storage(TensorStorage::cpu(vec![0.0f32; 8]), vec![4, 2], true).unwrap();
    let emb = Embedding::from_pretrained(weight, padding_idx)
        .unwrap()
        .with_scale_grad_by_freq(true);
    let inp = index_tensor(idx);
    let out = emb.forward(&inp).unwrap();
    let n = idx.len();
    let grad_output =
        Tensor::from_storage(TensorStorage::cpu(vec![1.0f32; n * 2]), vec![n, 2], false).unwrap();
    let grads = out.grad_fn().unwrap().backward(&grad_output).unwrap();
    let gd = grads[0].as_ref().unwrap().data().unwrap();
    (0..4).map(|r| gd[r * 2]).collect()
}

/// Plain Embedding divides each index by its OWN count (Embedding.cpp:155-161),
/// NOT by a sorted neighbour's. `[3,3,3,1,0]`: index 3 appears 3x → grad 3/3=1.
///
/// Live torch 2.11 grad_W col0 = [1, 1, 0, 1].
#[test]
fn plain_embedding_sgbf_divides_by_own_count() {
    let g = plain_grad_col0(&[3.0, 3.0, 3.0, 1.0, 0.0], None);
    let expect = [1.0, 1.0, 0.0, 1.0]; // live torch 2.11
    for (i, &e) in expect.iter().enumerate() {
        assert!(
            (g[i] - e).abs() < 1e-5,
            "plain grad_W[row {i}][0]={} exp {e} (own-count, no sort)",
            g[i]
        );
    }
}

/// Order invariance: same multiset as above in a different order MUST give the
/// identical gradient, because plain Embedding has no `indices_.sort()`. If the
/// plain path had adopted EmbeddingBag's sorted-neighbour quirk these would differ.
///
/// Live torch 2.11 grad_W col0 for `[0,3,3,3,1]` = [1, 1, 0, 1] (== the other order).
#[test]
fn plain_embedding_sgbf_order_invariant() {
    let g_a = plain_grad_col0(&[3.0, 3.0, 3.0, 1.0, 0.0], None);
    let g_b = plain_grad_col0(&[0.0, 3.0, 3.0, 3.0, 1.0], None);
    let expect = [1.0, 1.0, 0.0, 1.0]; // live torch 2.11, both orders
    for i in 0..4 {
        assert!(
            (g_b[i] - expect[i]).abs() < 1e-5,
            "reordered plain grad_W[row {i}][0]={} exp {}",
            g_b[i],
            expect[i]
        );
        assert!(
            (g_a[i] - g_b[i]).abs() < 1e-6,
            "plain Embedding grad must be order-invariant: row {i} {} vs {}",
            g_a[i],
            g_b[i]
        );
    }
}

/// Plain Embedding with two distinct multi-dup indices.
/// `[2,2,0,0,0]`: index 0 appears 3x → 3/3=1; index 2 appears 2x → 2/2=1.
///
/// Live torch 2.11 grad_W col0 = [1, 0, 1, 0].
#[test]
fn plain_embedding_sgbf_two_dup_groups() {
    let g = plain_grad_col0(&[2.0, 2.0, 0.0, 0.0, 0.0], None);
    let expect = [1.0, 0.0, 1.0, 0.0]; // live torch 2.11
    for (i, &e) in expect.iter().enumerate() {
        assert!(
            (g[i] - e).abs() < 1e-5,
            "plain grad_W[row {i}][0]={} exp {e}",
            g[i]
        );
    }
}
