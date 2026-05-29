//! Discriminator audit of commit 45375f66b (#1651 batch 3b: torch.special
//! zeta + airy_ai). These probes target the EDGE branches the generator's
//! in-module tests did NOT exercise:
//!
//!  1. `zeta(x, q)` with q <= 0 NON-integer AND x an INTEGER. The generator's
//!     `zeta_edge_ladder_vs_torch` only checks q=-1.5 with x=2.5 (non-integer
//!     x -> NaN). When x is an integer, upstream Cephes
//!     (`aten/src/ATen/native/cuda/Math.cuh:336-343`) SKIPS the `x != floor(x)`
//!     NaN branch and FALLS THROUGH to the `pow(q, -x)` convergence loop with a
//!     NEGATIVE base. This requires C `pow(neg_base, integer_exp)` semantics
//!     (real, finite) and is the most fragile untested path.
//!
//!  2. airy_ai region-boundary continuity: the AFN/AFD<->AGN/AGD oscillatory
//!     region (x < -2.09), the AN/AD decaying region (x >= 2.09 with early
//!     return at x > 8.3203353), the x > 103.892 -> 0 cliff, and a deep
//!     negative argument (x = -100) that stresses the oscillatory series.
//!
//! All oracle values are LIVE torch 2.11.0+cu130 (R-CHAR-3), captured via
//! `torch.special.zeta` / `torch.special.airy_ai` on f64.

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_core::{airy_ai, zeta};

fn cpu(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
}

/// Divergence probe: ferrotorch `zeta` must match
/// `pytorch aten/src/ATen/native/cuda/Math.cuh:336-343` for the case
/// `q <= 0`, `q` non-integer, `x` an INTEGER — where upstream falls through
/// the edge ladder into the convergence loop with a negative `pow` base.
///
/// Live torch 2.11:
///   zeta(2, -1.5) = 9.379246644989124   (even integer x)
///   zeta(3, -1.5) = 0.11810202582086413 (odd  integer x)
///   zeta(2, -0.5) = 8.934802200544679
///   zeta(4, -2.5) = 32.457979369864596
///
/// NOT NaN and NOT +inf. This branch is unexercised by the generator's
/// `zeta_edge_ladder_vs_torch` (which only uses non-integer x=2.5).
#[test]
fn zeta_neg_noninteger_q_integer_x_vs_torch() {
    let xs = vec![2.0, 3.0, 2.0, 4.0];
    let qs = vec![-1.5, -1.5, -0.5, -2.5];
    let want = [
        9.379246644989124,
        0.11810202582086413,
        8.934802200544679,
        32.457979369864596,
    ];
    let r = zeta(&cpu(xs.clone(), vec![4]), &cpu(qs.clone(), vec![4])).unwrap();
    let d = r.data().unwrap();
    for i in 0..4 {
        assert!(
            d[i].is_finite(),
            "zeta(x={}, q={}) must be FINITE (torch={}), got {}",
            xs[i],
            qs[i],
            want[i],
            d[i]
        );
        assert!(
            (d[i] - want[i]).abs() <= 1e-9 * (1.0 + want[i].abs()),
            "zeta idx {i} x={} q={}: got {} want {} (live torch 2.11)",
            xs[i],
            qs[i],
            d[i],
            want[i]
        );
    }
}

/// Divergence probe: airy_ai region boundaries + deep negative argument vs
/// live torch 2.11 (`aten/src/ATen/native/cuda/Math.cuh:1372-1457`).
///
///   airy_ai(-2.09)      = 0.17005055173203007  (oscillatory boundary)
///   airy_ai(-2.0900001) = 0.17005347501314444  (just inside oscillatory)
///   airy_ai(-100.0)     = 0.1767533932395512   (deep oscillatory)
///   airy_ai(2.09)       = 0.03042031836319837  (decaying boundary)
///   airy_ai(8.3203354)  = 1.861309161510959e-08(decaying early-return)
///   airy_ai(103.892)    = 2.240778287387011e-308 (just under cliff)
///   airy_ai(103.8920001)= 0.0                   (x > 103.892 cliff)
#[test]
fn airy_ai_region_boundaries_vs_torch() {
    let xs = vec![
        -2.09,
        -2.0900001,
        -100.0,
        2.09,
        8.3203354,
        103.892,
        103.8920001,
    ];
    let want = [
        0.17005055173203007,
        0.17005347501314444,
        0.1767533932395512,
        0.03042031836319837,
        1.861309161510959e-08,
        2.240778287387011e-308,
        0.0,
    ];
    let r = airy_ai(&cpu(xs.clone(), vec![7])).unwrap();
    let d = r.data().unwrap();
    for i in 0..7 {
        if want[i] == 0.0 {
            assert_eq!(
                d[i], 0.0,
                "airy_ai({}) must be exactly 0 (x>103.892 cliff), got {}",
                xs[i], d[i]
            );
        } else {
            assert!(
                (d[i] - want[i]).abs() <= 1e-9 * (1.0 + want[i].abs()),
                "airy_ai idx {i} x={}: got {} want {} (live torch 2.11)",
                xs[i],
                d[i],
                want[i]
            );
        }
    }
}
