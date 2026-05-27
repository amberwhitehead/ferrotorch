//! Divergence tests for the in-place `max_norm` renorm added in commit
//! `fbac4e055` (closes #1445).
//!
//! The renorm in `renorm_weight_rows_in_place`
//! (`ferrotorch-nn/src/embedding.rs:84-147`) computes the row norm as a
//! generic finite p-norm:
//!
//! ```text
//! acc += |v|.powf(norm_type);   // per element
//! norm = acc.powf(1.0 / norm_type);
//! ```
//!
//! PyTorch's `embedding_renorm_cpu_`
//! (`aten/src/ATen/native/Embedding.cpp:202-203`) instead calls
//! `row.norm(norm_type)`, i.e. `at::norm`, which special-cases the
//! non-finite `norm_type` values exactly the way the math requires:
//! `norm_type == +inf` is the infinity norm `max_i |x_i|` and
//! `norm_type == 0` is the L0 "count of nonzeros". The generic-powf
//! formula does NOT reproduce those limits, so ferrotorch diverges for
//! the documented `norm_type=inf` constraint.
//!
//! All expected values are taken from live torch 2.11.0
//! (`torch.embedding_renorm_`), R-CHAR-3 — no tautologies.
//!
//! Tracking: #1564

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_nn::Embedding;
use ferrotorch_nn::module::Module;

fn weight_3x2(rows: &[f64; 6]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(rows.to_vec()), vec![3, 2], true).unwrap()
}

fn idx(vals: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(vals.to_vec()), vec![vals.len()], false).unwrap()
}

/// Control test: `norm_type=2` (the default Euclidean norm) MUST match torch.
/// This proves the test harness exercises the renorm path correctly, so the
/// `norm_type=inf` failure below is a real divergence and not a scaffolding bug.
///
/// Live torch 2.11.0:
///   w = [[1,1],[3,4],[9,9]]; torch.embedding_renorm_(w, [1], max_norm=2, norm_type=2)
///   row1 (L2-norm=5 > 2) -> scale=2/(5+1e-7); row1 = [3*s, 4*s].
#[test]
fn renorm_norm_type_2_control_matches_torch() {
    let emb = Embedding::from_pretrained(weight_3x2(&[1.0, 1.0, 3.0, 4.0, 9.0, 9.0]), None)
        .unwrap()
        .with_max_norm(2.0)
        .with_norm_type(2.0);
    let _ = emb.forward(&idx(&[1.0])).unwrap();

    let w = emb.weight.data().unwrap();
    let scale = 2.0 / (5.0 + 1e-7);
    let exp1 = 3.0 * scale; // torch: 1.1999999952...
    let exp2 = 4.0 * scale; // torch: 1.5999999936...
    assert!(
        (w[2] - exp1).abs() < 1e-12,
        "row1[0] got {}, want {}",
        w[2],
        exp1
    );
    assert!(
        (w[3] - exp2).abs() < 1e-12,
        "row1[1] got {}, want {}",
        w[3],
        exp2
    );
}

/// Divergence: ferrotorch's `renorm_weight_rows_in_place` diverges from
/// `pytorch aten/src/ATen/native/Embedding.cpp:202-204` for `norm_type=inf`.
///
/// Input: weight row 1 = `[3, 4]`, `max_norm=2`, `norm_type=+inf`.
/// Upstream (`row.norm(inf)` = max(|3|,|4|) = 4 > 2) scales row 1 by
/// `2/(4+1e-7)`, yielding `[1.4999999625, 1.99999995]`
/// (live torch 2.11.0 `torch.embedding_renorm_`).
/// ferrotorch computes `(3^inf + 4^inf)^(1/inf) = inf^0 = 1.0`, decides
/// `1.0 <= 2.0`, and leaves row 1 = `[3, 4]` UN-renormed.
/// Tracking: #1564
#[test]
fn divergence_renorm_norm_type_inf_max() {
    let emb = Embedding::from_pretrained(weight_3x2(&[1.0, 1.0, 3.0, 4.0, 9.0, 9.0]), None)
        .unwrap()
        .with_max_norm(2.0)
        .with_norm_type(f64::INFINITY);
    let _ = emb.forward(&idx(&[1.0])).unwrap();

    let w = emb.weight.data().unwrap();
    // Live torch 2.11.0: row1 = [1.499999962500001, 1.9999999500000012].
    let scale = 2.0 / (4.0 + 1e-7);
    let exp1 = 3.0 * scale;
    let exp2 = 4.0 * scale;
    assert!(
        (w[2] - exp1).abs() < 1e-9,
        "norm_type=inf row1[0]: ferrotorch {} != torch {}",
        w[2],
        exp1
    );
    assert!(
        (w[3] - exp2).abs() < 1e-9,
        "norm_type=inf row1[1]: ferrotorch {} != torch {}",
        w[3],
        exp2
    );
}
