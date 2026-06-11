//! Red-then-green regression tests for audit finding CORE-169 (crosslink
//! #1863): `gammainc`/`gammaincc` cap the Numerical-Recipes power series and
//! Lentz continued fraction at 300 iterations and return the PARTIAL sum as
//! if converged. Near `x ≈ a` the series needs O(√a) terms, so from
//! `a ≈ 1.2e4` upward the cap is exceeded and the ops return silently wrong
//! probabilities (`gammainc(1e5, 1e5)` = 0.329 pre-fix vs torch 0.50042…).
//!
//! Oracle (R-ORACLE-1 path (b)) — live torch 2.11.0+cu130 f64 session,
//! 2026-06-11, this machine:
//!
//! ```python
//! >>> import torch
//! >>> t = lambda v: torch.tensor(v, dtype=torch.float64)
//! >>> [torch.special.gammainc(t(a), t(x)).item() for ...]
//! a=1e4 x=9700.0:             P=0.0012341755842795814  Q=0.9987658244157205
//! a=1e4 x=10000.0:            P=0.5013298083214526     Q=0.4986701916785474
//! a=1e4 x=10300.0:            P=0.9985295051033916     Q=0.0014704948966084723
//! a=1e5 x=99051.31670194949:  P=0.0013127698795359416  Q=0.9986872301204641
//! a=1e5 x=100000.0:           P=0.5004205221045142     Q=0.49957947789548585
//! a=1e5 x=100948.68329805051: P=0.9986124863191003     Q=0.001387513680899791
//! a=1e6 x=997000.0:           P=0.001338104167293426   Q=0.9986618958327066
//! a=1e6 x=1000000.0:          P=0.5001329807590223     Q=0.49986701924097765
//! a=1e6 x=1003000.0:          P=0.9986382593537616     Q=0.001361740646238383
//! # small-a regime (must stay green through the fix):
//! a=0.5 x=0.2:   P=0.47291074313446185
//! a=2.0 x=1.0:   P=0.2642411176571153   Q=0.7357588823428847
//! a=4.0 x=5.0:   P=0.7349740847026385
//! a=1.2e4 x=1.2e4: P=0.5012139432456733 Q=0.4987860567543267
//! ```
//!
//! Upstream algorithm being ported: `pytorch/aten/src/ATen/native/Math.h`
//! `calc_igamma` (:1144) / `calc_igammac` (:1070) regime selection over
//! `_igam_helper_asymptotic_series` (:713, DLMF 8.12.3/8.12.4),
//! `_igam_helper_series` (:655), `_igamc_helper_series` (:687) and
//! `_igamc_helper_continued_fraction` (:1006).
//!
//! Tolerance justification (R-ORACLE-5): assertions use RELATIVE error
//! ≤ 1e-11. The two oracles themselves (torch's double kernel vs
//! scipy.special.gammainc) differ by up to 1.2e-11 relative at the pinned
//! points (worst: a=1e5, x=1e5 — 0.5004205221045142 vs 0.5004205221103651);
//! the ferrotorch port runs the identical f64 algorithm and differs from
//! torch only through ~1-ulp library kernels (erfc, lgamma), far below that
//! inter-oracle drift. Pre-fix values diverge at the FIRST decimal digit, so
//! the gate has ~10 orders of margin against the regression it pins.

use ferrotorch_core::special::{gammainc, gammaincc};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

#[cfg(feature = "gpu")]
use ferrotorch_core::Device;
#[cfg(feature = "gpu")]
use std::sync::Once;

#[cfg(feature = "gpu")]
static GPU_INIT: Once = Once::new();

#[cfg(feature = "gpu")]
fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU lane of this suite");
    });
}

fn t64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// Assert relative error ≤ 1e-11 (justification in module doc).
fn assert_rel(got: f64, expected: f64, what: &str) {
    let rel = ((got - expected) / expected).abs();
    assert!(
        rel <= 1e-11,
        "{what}: got {got:e}, torch oracle {expected:e}, rel err {rel:e} > 1e-11"
    );
}

/// (a, x, torch P(a,x), torch Q(a,x)) — see module doc for the live session.
const LARGE_A_PINS: &[(f64, f64, f64, f64)] = &[
    (1e4, 9700.0, 0.0012341755842795814, 0.9987658244157205),
    (1e4, 10000.0, 0.5013298083214526, 0.4986701916785474),
    (1e4, 10300.0, 0.9985295051033916, 0.0014704948966084723),
    (
        1e5,
        99051.31670194949,
        0.0013127698795359416,
        0.9986872301204641,
    ),
    (1e5, 100000.0, 0.5004205221045142, 0.49957947789548585),
    (
        1e5,
        100948.68329805051,
        0.9986124863191003,
        0.001387513680899791,
    ),
    (1e6, 997000.0, 0.001338104167293426, 0.9986618958327066),
    (1e6, 1000000.0, 0.5001329807590223, 0.49986701924097765),
    (1e6, 1003000.0, 0.9986382593537616, 0.001361740646238383),
];

#[test]
fn core169_gammainc_large_a_matches_torch() {
    for &(a, x, p_expected, _) in LARGE_A_PINS {
        let r = gammainc(&t64(&[a], &[1]), &t64(&[x], &[1])).unwrap();
        assert_rel(
            r.data().unwrap()[0],
            p_expected,
            &format!("gammainc({a:e}, {x:e})"),
        );
    }
}

#[test]
fn core169_gammaincc_large_a_matches_torch() {
    for &(a, x, _, q_expected) in LARGE_A_PINS {
        let r = gammaincc(&t64(&[a], &[1]), &t64(&[x], &[1])).unwrap();
        assert_rel(
            r.data().unwrap()[0],
            q_expected,
            &format!("gammaincc({a:e}, {x:e})"),
        );
    }
}

/// The small-a regime the pre-fix tests covered must not move (same oracle
/// session as the module doc).
#[test]
fn core169_small_a_regime_unchanged() {
    let pins: &[(f64, f64, f64, f64)] = &[
        (0.5, 0.2, 0.47291074313446185, 0.5270892568655381),
        (2.0, 1.0, 0.2642411176571153, 0.7357588823428847),
        (4.0, 5.0, 0.7349740847026385, 0.2650259152973616),
        (1.2e4, 1.2e4, 0.5012139432456733, 0.4987860567543267),
    ];
    for &(a, x, p_expected, q_expected) in pins {
        let p = gammainc(&t64(&[a], &[1]), &t64(&[x], &[1])).unwrap();
        assert_rel(
            p.data().unwrap()[0],
            p_expected,
            &format!("gammainc({a}, {x})"),
        );
        let q = gammaincc(&t64(&[a], &[1]), &t64(&[x], &[1])).unwrap();
        assert_rel(
            q.data().unwrap()[0],
            q_expected,
            &format!("gammaincc({a}, {x})"),
        );
    }
}

/// CUDA lane: unlike the unary special ops (which round-trip through
/// `unary_map`'s documented host fallback), the BINARY special ops dispatch
/// through `binary_map`, which has no CUDA path — on CUDA inputs they return
/// a structured `Err` (same contract the in-module `zeta_cuda_not_implemented`
/// documents; the unary/binary round-trip inconsistency is audit CORE-177,
/// tracked separately). R-LOUD-1: the error is the contract — what this test
/// pins is that no silently-wrong VALUE can come back from the CUDA lane, so
/// the CORE-169 fix cannot be bypassed by a diverging device kernel.
#[cfg(feature = "gpu")]
#[test]
fn core169_gammainc_cuda_is_structured_error_not_silent_value() {
    ensure_cuda_backend();
    let a = t64(&[1e5], &[1]).to(Device::Cuda(0)).unwrap();
    let x = t64(&[1e5], &[1]).to(Device::Cuda(0)).unwrap();
    let p = gammainc(&a, &x);
    assert!(
        p.is_err(),
        "gammainc on CUDA must be a structured error (no device kernel exists), got {:?}",
        p.map(|t| t.data_vec())
    );
    let q = gammaincc(&a, &x);
    assert!(
        q.is_err(),
        "gammaincc on CUDA must be a structured error (no device kernel exists), got {:?}",
        q.map(|t| t.data_vec())
    );
}
