//! Red-then-green regression tests for audit finding CORE-171 (crosslink
//! #1865): `erfinv`'s edge ladder uses `>= 1` / `<= -1`, so out-of-domain
//! inputs (`|x| > 1`) receive a plausible-looking ±inf instead of NaN.
//!
//! Upstream contract: `pytorch/aten/src/ATen/native/Math.h:152-172
//! calc_erfinv` checks `|y| > 1 → NaN` FIRST, then
//! `|y| == 1 → copysign(inf, y)` — ±inf is reserved for exactly ±1.
//!
//! Oracle (R-ORACLE-1 path (b)) — live torch 2.11.0+cu130 f64 session,
//! 2026-06-11, this machine:
//!
//! ```python
//! >>> t = lambda v: torch.tensor(v, dtype=torch.float64)
//! >>> torch.erfinv(t(2.0)).item()         # nan
//! >>> torch.erfinv(t(-2.0)).item()        # nan
//! >>> torch.erfinv(t(1.0000001)).item()   # nan
//! >>> torch.erfinv(t(-1.0000001)).item()  # nan
//! >>> torch.erfinv(t(1.0)).item()         # inf
//! >>> torch.erfinv(t(-1.0)).item()        # -inf
//! >>> torch.erfinv(t(0.5)).item()         # 0.4769362762044699
//! ```
//!
//! Tolerance justification (R-ORACLE-5): the edge ladder is exact-equality
//! (NaN-ness / signed infinity — no tolerance applies). The single interior
//! pin (0.5) uses 1e-12 absolute: the implementation Newton-refines with the
//! ~1-ulp `erf_f64_hi` to residual < 4·f64::EPSILON, so 1e-12 is ~3 orders
//! above its worst case while 12 orders below the pre/post-fix divergence.

use ferrotorch_core::special::erfinv;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn t64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}
fn t32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

#[test]
fn core171_erfinv_out_of_domain_is_nan_f64() {
    let r = erfinv(&t64(
        &[2.0, -2.0, 1.0000001, -1.0000001, 1e300, -1e300],
        &[6],
    ))
    .unwrap();
    let d = r.data().unwrap();
    for (i, v) in d.iter().enumerate() {
        assert!(
            v.is_nan(),
            "erfinv(|x|>1) must be NaN per torch/calc_erfinv, lane {i} got {v}"
        );
    }
}

#[test]
fn core171_erfinv_out_of_domain_is_nan_f32() {
    let r = erfinv(&t32(&[2.0, -2.0, 1.5, -1.5], &[4])).unwrap();
    let d = r.data().unwrap();
    for (i, v) in d.iter().enumerate() {
        assert!(
            v.is_nan(),
            "erfinv f32 (|x|>1) must be NaN per torch/calc_erfinv, lane {i} got {v}"
        );
    }
}

#[test]
fn core171_erfinv_edge_ladder_exact() {
    // ±1 stays ±inf (exact equality only), NaN propagates, interior intact.
    let r = erfinv(&t64(&[1.0, -1.0, f64::NAN, 0.5], &[4])).unwrap();
    let d = r.data().unwrap();
    assert!(
        d[0].is_infinite() && d[0] > 0.0,
        "erfinv(1.0) must stay +inf, got {}",
        d[0]
    );
    assert!(
        d[1].is_infinite() && d[1] < 0.0,
        "erfinv(-1.0) must stay -inf, got {}",
        d[1]
    );
    assert!(d[2].is_nan(), "erfinv(NaN) must be NaN, got {}", d[2]);
    assert!(
        (d[3] - 0.4769362762044699).abs() < 1e-12,
        "erfinv(0.5) interior value moved: got {}",
        d[3]
    );
}
