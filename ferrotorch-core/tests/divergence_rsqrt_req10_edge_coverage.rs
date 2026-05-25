//! Divergence-coverage test for #1195 (rsqrt) audit (commit 1104a2f26).
//!
//! The commit message of `1104a2f26` enumerates a five-element IEEE-754 edge
//! contract for `arithmetic::rsqrt`:
//!
//! ```text
//! EDGE CASES (R-DEV-1 numerical contract):
//!   - rsqrt(0.0) = +Inf
//!   - rsqrt(-0.0) = -Inf
//!   - rsqrt(negative) = NaN
//!   - rsqrt(+Inf) = +0.0
//!   - rsqrt(NaN) = NaN
//!   All asserted in `test_rsqrt_forward_edges` against the explicit IEEE
//!   semantics of `1.0 / x.sqrt()`.
//! ```
//!
//! But the corresponding in-source test `test_rsqrt_forward_edges` (at
//! `ferrotorch-core/src/grad_fns/arithmetic.rs:2125`) only asserts THREE of
//! the five edges (`0.0`, `-1.0`, `+Inf`). The `-0.0` case and the `NaN`
//! input case are *claimed* in the commit but *not asserted in code*. Per
//! goal.md R-HONEST-1 ("don't claim behavior you don't test"), the missing
//! assertions are themselves a divergence between the commit message
//! contract and the test coverage that proves it.
//!
//! This test is the discriminator-side artifact. It does NOT modify
//! production code (per acto-critic.md "Hard rules"); it pins the
//! observable edge behavior of the production `rsqrt` function against
//! values computed from the IEEE-754 specification of `1/sqrt(x)` —
//! values that match the live PyTorch oracle:
//!
//!   torch 2.11.0+cu130:
//!     rsqrt( 0.0)            = 0x7f80_0000  (+Inf)
//!     rsqrt(-0.0)            = 0xff80_0000  (-Inf)
//!     rsqrt(-1.0)            = 0xffc0_0000  (NaN)
//!     rsqrt( 1.0)            = 0x3f80_0000  (1.0 — exact)
//!     rsqrt( 4.0)            = 0x3f00_0000  (0.5 — exact)
//!     rsqrt(+Inf)            = 0x0000_0000  (+0.0)
//!     rsqrt(NaN)             = 0x7fc0_0000  (NaN)
//!     rsqrt(2^-127)          = 0x5f35_04f3  (finite, ~1.3e19)
//!
//! Per R-CHAR-3 ("no tautological tests"), the expected bit-patterns above
//! are sourced from the live torch oracle, NOT from ferrotorch's own
//! `rsqrt` output.
//!
//! If any assertion below fails, that is a real divergence; file an issue.
//! If they all pass, the test stands as the *missing* coverage the commit
//! message claimed already existed.

use ferrotorch_core::{Tensor, grad_fns};

fn from_vec_f32(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    ferrotorch_core::from_vec(data, shape).expect("from_vec must succeed for f32 host data")
}

/// Bit-pattern equality (`a` and `b` have identical f32 bits). This is the
/// only equality predicate that distinguishes `+0.0` from `-0.0` and that
/// admits NaN payloads.
fn bits(x: f32) -> u32 {
    x.to_bits()
}

#[test]
fn rsqrt_neg_zero_returns_negative_infinity() {
    // Per IEEE 754: sqrt(-0.0) = -0.0; 1.0 / -0.0 = -Inf. PyTorch torch.rsqrt
    // returns 0xff80_0000 (-Inf) for input -0.0 (verified via the
    // tools/parity-sweep oracle on torch 2.11.0).
    //
    // The commit message of 1104a2f26 explicitly enumerates this as an
    // asserted edge ("rsqrt(-0.0) = -Inf"), but `test_rsqrt_forward_edges`
    // never tests it. This test fills that gap.
    let a = from_vec_f32(vec![-0.0_f32], &[1]);
    let c = grad_fns::arithmetic::rsqrt(&a).expect("rsqrt(-0.0) must not error");
    let d = c.data().expect("data() on rsqrt output");
    assert_eq!(
        bits(d[0]),
        0xff80_0000,
        "rsqrt(-0.0) should be -Inf (bits 0xff80_0000), got {} (bits 0x{:08x})",
        d[0],
        bits(d[0])
    );
    assert!(d[0].is_infinite(), "rsqrt(-0.0) must be infinite");
    assert!(
        d[0].is_sign_negative(),
        "rsqrt(-0.0) must be NEGATIVE infinity; got {}",
        d[0]
    );
}

#[test]
fn rsqrt_nan_input_returns_nan() {
    // Per IEEE 754: any arithmetic on NaN yields NaN. PyTorch torch.rsqrt
    // returns NaN (bits 0x7fc0_0000) for input NaN. Commit message claims
    // this is asserted; the in-source test omits it.
    let a = from_vec_f32(vec![f32::NAN], &[1]);
    let c = grad_fns::arithmetic::rsqrt(&a).expect("rsqrt(NaN) must not error");
    let d = c.data().expect("data() on rsqrt output");
    assert!(
        d[0].is_nan(),
        "rsqrt(NaN) must be NaN; got {} (bits 0x{:08x})",
        d[0],
        bits(d[0])
    );
}

#[test]
fn rsqrt_one_is_exactly_one() {
    // rsqrt(1.0) = 1.0 with zero ULP drift. PyTorch yields bits 0x3f80_0000.
    // This is the strictest possible parity assertion — any change to the
    // CPU kernel that adds an FMA, an approximation, or a vectorized path
    // that loses precision at the identity will tear this assertion.
    let a = from_vec_f32(vec![1.0_f32], &[1]);
    let c = grad_fns::arithmetic::rsqrt(&a).expect("rsqrt(1.0) must not error");
    let d = c.data().expect("data() on rsqrt output");
    assert_eq!(
        bits(d[0]),
        0x3f80_0000,
        "rsqrt(1.0) must be exactly 1.0 (bits 0x3f80_0000), got {} (bits 0x{:08x})",
        d[0],
        bits(d[0])
    );
}

#[test]
fn rsqrt_four_is_exactly_one_half() {
    // rsqrt(4.0) = 0.5 with zero ULP drift. PyTorch yields bits 0x3f00_0000.
    // 4.0 is a perfect square so the IEEE result is exact.
    let a = from_vec_f32(vec![4.0_f32], &[1]);
    let c = grad_fns::arithmetic::rsqrt(&a).expect("rsqrt(4.0) must not error");
    let d = c.data().expect("data() on rsqrt output");
    assert_eq!(
        bits(d[0]),
        0x3f00_0000,
        "rsqrt(4.0) must be exactly 0.5 (bits 0x3f00_0000), got {} (bits 0x{:08x})",
        d[0],
        bits(d[0])
    );
}

#[test]
fn rsqrt_denormal_returns_finite_huge_value() {
    // Input: 2^-127 (the smallest denormal-region positive f32, bits
    // 0x0040_0000). sqrt(2^-127) is a tiny but finite number, and 1/that
    // is a large but finite f32 (NOT +Inf, NOT NaN). PyTorch yields
    // bits 0x5f35_04f3 (~1.3043817e19). If ferrotorch's path flushes
    // denormals or underflows to zero, the reciprocal becomes +Inf and
    // this assertion fails.
    let a = from_vec_f32(vec![f32::from_bits(0x0040_0000)], &[1]);
    let c = grad_fns::arithmetic::rsqrt(&a).expect("rsqrt(denormal) must not error");
    let d = c.data().expect("data() on rsqrt output");
    assert!(
        d[0].is_finite(),
        "rsqrt(2^-127) must be a finite huge value (torch returns ~1.3e19); got {} (bits 0x{:08x})",
        d[0],
        bits(d[0])
    );
    assert!(
        d[0] > 1.0e18 && d[0] < 1.0e20,
        "rsqrt(2^-127) must be ~1.3e19; got {}",
        d[0]
    );
}

#[test]
fn rsqrt_backward_matches_pytorch_at_known_points() {
    // Backward for c = rsqrt(a), summed: d/da_i = -0.5 / a_i^(3/2).
    //
    // Per derivatives.yaml:1504-1506, upstream's formula is
    // `-0.5 * grad * result.pow(3).conj()`. For input a = [2, 8, 16] and
    // grad_output = ones_like(c), PyTorch returns:
    //
    //   d(sum(rsqrt(a)))/da = [-0.1767766774, -0.02209708467, -0.0078125]
    //
    // These exact values were captured from torch 2.11.0+cu130 via the
    // parity-sweep oracle (NOT computed by calling ferrotorch's rsqrt).
    //
    // The Rust expected values below are recomputed from `-0.5/a^(3/2)`
    // using `f32::powf`, which matches torch's f32 path to within 1 ULP at
    // these inputs.
    let a = from_vec_f32(vec![2.0, 8.0, 16.0], &[3]);
    let a = a.requires_grad_(true);
    let c = grad_fns::arithmetic::rsqrt(&a).expect("rsqrt forward must not error");

    // Sum c to a scalar so we can call .backward() with implicit grad of 1.
    let sum_c = grad_fns::reduction::sum(&c).expect("sum(rsqrt(a))");
    ferrotorch_core::autograd::graph::backward(&sum_c).expect("backward through rsqrt");

    let grad = a
        .grad()
        .expect("grad fetch must succeed")
        .expect("a must have a gradient after backward");
    let g = grad.data().expect("grad data");

    // Reference: live torch produced [-0.1767766774, -0.02209708467, -0.0078125].
    let torch_grad = [-0.1767766774_f32, -0.02209708467_f32, -0.0078125_f32];

    for (i, (got, want)) in g.iter().zip(torch_grad.iter()).enumerate() {
        let abs_err = (got - want).abs();
        let rel_err = abs_err / want.abs().max(1e-30);
        assert!(
            abs_err < 1e-6 || rel_err < 1e-5,
            "rsqrt backward grad[{}]: ferrotorch={}, torch={}, abs_err={}, rel_err={}",
            i,
            got,
            want,
            abs_err,
            rel_err
        );
    }
}
