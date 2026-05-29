//! Discriminator probes for #1651 batch 1 (entr/ndtr/ndtri) coverage gaps not
//! exercised by the in-crate lib tests: ndtr f64 deep tails, entr large/tiny x,
//! ndtri central-region symmetry. Expected values are live
//! `torch.special.*` (torch 2.11.0+cu130, f64) outputs (R-CHAR-3).

use ferrotorch_core::{Tensor, TensorStorage, entr, ndtr, ndtri};

fn t(v: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(v.to_vec()), shape.to_vec(), false).unwrap()
}

/// ndtr f64 deep tails. torch.special.ndtr(f64) at x in {-10,-6,-4,4,6,10}:
/// the lib test only covers [-3,3]; the tail is where the composite
/// `(1+erf(x/sqrt2))*0.5` can diverge from torch's `at::erf`.
#[test]
fn ndtr_f64_deep_tails_vs_torch() {
    let input = t(&[-10.0, -6.0, -4.0, 4.0, 6.0, 10.0], &[6]);
    let r = ndtr(&input).unwrap();
    let d = r.data().unwrap();
    // live torch.special.ndtr f64 oracle:
    let want: [f64; 6] = [
        0.0,
        9.865_876_449_133_282e-10,
        3.167_124_183_311_998e-5,
        0.999_968_328_758_166_9,
        0.999_999_999_013_412_3,
        1.0,
    ];
    for i in 0..6 {
        let tol = 1e-12 * (1.0 + want[i].abs());
        assert!(
            (d[i] - want[i]).abs() <= tol,
            "ndtr tail idx {i}: got {} want {} (diff {})",
            d[i],
            want[i],
            (d[i] - want[i]).abs()
        );
    }
}

/// entr f64 large + tiny x. torch.special.entr(f64).
#[test]
fn entr_f64_large_tiny_vs_torch() {
    let input = t(&[1e-30, 1e-20, 1e5, 1e10, 1e20], &[5]);
    let r = entr(&input).unwrap();
    let d = r.data().unwrap();
    let want: [f64; 5] = [
        6.907_755_278_982_138e-29,
        4.605_170_185_988_091e-19,
        -1_151_292.546_497_023,
        -230_258_509_299.404_57,
        -4.605_170_185_988_091_7e21,
    ];
    for i in 0..5 {
        let tol = 1e-9 * (1.0 + want[i].abs());
        assert!(
            (d[i] - want[i]).abs() <= tol,
            "entr idx {i}: got {} want {}",
            d[i],
            want[i]
        );
    }
}

/// ndtri central-region near 0.5: torch is exactly antisymmetric about 0.5.
#[test]
fn ndtri_near_half_symmetry_vs_torch() {
    let input = t(&[0.5 - 1e-8, 0.5 + 1e-8, 0.4, 0.6], &[4]);
    let r = ndtri(&input).unwrap();
    let d = r.data().unwrap();
    let want: [f64; 4] = [
        -2.506_628_273_311_622_5e-8,
        2.506_628_287_226_204_7e-8,
        -0.253_347_103_135_799_7,
        0.253_347_103_135_799_7,
    ];
    for i in 0..4 {
        let tol = 1e-12 * (1.0 + want[i].abs());
        assert!(
            (d[i] - want[i]).abs() <= tol,
            "ndtri idx {i}: got {} want {}",
            d[i],
            want[i]
        );
    }
}
