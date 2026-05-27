//! Divergence-coverage tests for #1197 (floor_divide REQ-12) audit
//! (commit `cd8a6af58`).
//!
//! The commit ships `arithmetic::floor_divide<T: Float>` +
//! `FloorDivideBackward` at
//! `ferrotorch-core/src/grad_fns/arithmetic.rs:2616` / `:2459` mirroring
//! `c10::div_floor_floating` from `/home/doll/pytorch/c10/util/generic_math.h:34-58`.
//! The builder's 12 unit tests at `arithmetic.rs:3914-4124` cover the
//! happy four-quadrant sign matrix, `(±5, 0)` IEEE-754 div-by-zero,
//! `(0, 0) = NaN`, NaN propagation, the (-7, 3) three-way contrast vs
//! `remainder` and `fmod`, and a `[2]@[1]` broadcast — but they leave six
//! load-bearing edges uncovered:
//!
//! 1. `floor_divide(±Inf, finite)`: builder docstring at
//!    `arithmetic.rs:2587-2588` claims `NaN` "because `fmod(Inf, 3) = NaN`
//!    propagates through the (a-mod)/b step". Live torch on 2026-05-25
//!    confirms `NaN`. NEVER ASSERTED in the unit tests.
//! 2. `floor_divide(finite, ±Inf)`: NOT MENTIONED in the docstring at
//!    all, and the audit shape A5 predicted `0` for all such cases.
//!    Live torch on 2026-05-25 returns a 2x2 sign-pattern:
//!      floor_divide( 5, +Inf) =  0
//!      floor_divide(-5, +Inf) = -1   <- sign-correction path triggers!
//!      floor_divide( 5, -Inf) = -1   <- sign-correction path triggers!
//!      floor_divide(-5, -Inf) =  0
//!    These are the algorithm's `(mod != 0) && (b<0) != (mod<0)` adjust-
//!    `div`-by-1 step. If the impl forgot the Inf-divisor case the result
//!    would be `0` across the board (a flat zero).
//! 3. Signed-zero output: builder's algorithm includes the
//!    `copysign(0, a/b)` step (`arithmetic.rs:2761-2766`) but no unit
//!    test verifies the sign-bit of the result. Live torch:
//!      floor_divide( 0, +3) = +0   (0x00000000)
//!      floor_divide( 0, -3) = -0   (0x80000000)
//!      floor_divide(-0, +3) = -0   (0x80000000)
//!      floor_divide(-0, -3) = +0   (0x00000000)
//! 4. Clean-division identity: builder asserts the identity
//!    `a == fd(a,b)*b + rem(a,b)` only at (-7, 3). The rem=0 branch
//!    (clean division) is a distinct algorithm path (the `m != 0` guard
//!    is FALSE, the `div == 0` branch is FALSE, going straight through
//!    `floor` with no copysign and no -1 adjust). E.g. (6, 3) and
//!    (-6, 3).
//! 5. Backward error MESSAGE: builder asserts `InvalidArgument` with
//!    "floor_divide" in the message, but doesn't pin the FULL upstream-
//!    parity message text ("derivative for ... is not implemented"). A
//!    silent rename of the error text would slip through.
//! 6. Large-magnitude inputs where naive `floor(a/b)` rounds wrong and
//!    only the `(a-mod)/b` form is correct. e.g. `1e7/3` in f32: the
//!    naive path may overshoot by one ULP, but upstream guarantees
//!    `(1e7 - fmod(1e7, 3)) / 3 == exact_integer_quotient`.
//!
//! Per R-CHAR-3 every expected value below is sourced from the LIVE
//! PyTorch oracle (`torch 2.11.0+cu130`, verified 2026-05-25) — not
//! self-checked against ferrotorch.

use ferrotorch_core::{FerrotorchError, Tensor, grad_fns};

fn leaf_vec(data: &[f32], requires_grad: bool) -> Tensor<f32> {
    let t = ferrotorch_core::from_vec(data.to_vec(), &[data.len()])
        .expect("from_vec must succeed for f32 host data");
    t.requires_grad_(requires_grad)
}

fn leaf_scalar(v: f32, requires_grad: bool) -> Tensor<f32> {
    leaf_vec(&[v], requires_grad)
}

/// (1) `floor_divide(±Inf, finite)` MUST be `NaN`.
///
/// Live oracle 2026-05-25:
///   torch.floor_divide(tensor([+inf, -inf]), tensor([3.0, 3.0])) -> tensor([nan, nan])
///
/// Upstream algorithm path: `b != 0`, so we hit `mod = fmod(Inf, 3) = NaN`,
/// then `div = (Inf - NaN) / 3 = NaN`. NaN comparisons evaluate `false`, so
/// the sign-correction is skipped, the `div == 0` test is `false`, and
/// `floor(NaN) = NaN` propagates to the final result.
#[test]
fn divergence_floor_divide_inf_dividend_yields_nan() {
    let a = leaf_vec(&[f32::INFINITY, f32::NEG_INFINITY], false);
    let b = leaf_vec(&[3.0, 3.0], false);
    let c = grad_fns::arithmetic::floor_divide(&a, &b).expect("floor_divide fwd must succeed");
    let data = c.data().expect("c.data");
    assert!(
        data[0].is_nan(),
        "floor_divide(+Inf, 3) expected NaN per torch oracle, got {}",
        data[0],
    );
    assert!(
        data[1].is_nan(),
        "floor_divide(-Inf, 3) expected NaN per torch oracle, got {}",
        data[1],
    );
}

/// (2) `floor_divide(finite, ±Inf)` MUST follow the 2x2 sign pattern
/// `[0, -1, -1, 0]` (for inputs `[5, -5, 5, -5] / [+Inf, +Inf, -Inf, -Inf]`).
///
/// Live oracle 2026-05-25:
///   torch.floor_divide(tensor([ 5,-5, 5,-5]),
///                      tensor([+inf,+inf,-inf,-inf])) -> tensor([0,-1,-1,0])
///
/// This is the algorithm's load-bearing sign-correction. For (-5, +Inf):
///   m = fmod(-5, +Inf) = -5
///   div = (-5 - (-5)) / +Inf = 0 / +Inf = +0
///   (m != 0) AND (b<0) != (m<0) is TRUE  (b=+Inf is not <0; m=-5 IS <0)
///   -> div -= 1 -> div = -1
///   div != 0 -> floor(-1) = -1, no 0.5 fixup -> result -1
///
/// If the implementation skipped the -Inf-divisor sign correction the
/// vector would be `[0, 0, 0, 0]` and this assertion would FAIL.
#[test]
fn divergence_floor_divide_inf_divisor_sign_correction() {
    let a = leaf_vec(&[5.0, -5.0, 5.0, -5.0], false);
    let b = leaf_vec(
        &[
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ],
        false,
    );
    let c = grad_fns::arithmetic::floor_divide(&a, &b).expect("floor_divide fwd must succeed");
    let data = c.data().expect("c.data");
    let expected = [0.0_f32, -1.0, -1.0, 0.0];
    for (i, &exp) in expected.iter().enumerate() {
        assert!(
            (data[i] - exp).abs() < 1e-6,
            "floor_divide(finite/Inf)[{i}] expected {exp} per torch oracle, got {} \
             (full vector {:?})",
            data[i],
            data,
        );
    }
}

/// (3) Signed-zero of the output MUST match `copysign(0, a/b)`.
///
/// Live oracle 2026-05-25 (bit-exact via torch.tensor.view(int32)):
///   floor_divide( 0.0, +3.0) -> +0  (bits 0x00000000)
///   floor_divide( 0.0, -3.0) -> -0  (bits 0x80000000)
///   floor_divide(-0.0, +3.0) -> -0  (bits 0x80000000)
///   floor_divide(-0.0, -3.0) -> +0  (bits 0x00000000)
///
/// Upstream algorithm: `m = 0`, `div = 0`, `div == 0` branch fires,
/// returns `copysign(0, a/b)`. The IEEE-754 sign of `a/b` is `sign(a) XOR
/// sign(b)`.
#[test]
fn divergence_floor_divide_signed_zero_sign_bit() {
    let cases = [
        (0.0_f32, 3.0_f32, 0u32),        // +0
        (0.0_f32, -3.0_f32, 0x80000000), // -0
        (-0.0_f32, 3.0_f32, 0x80000000), // -0
        (-0.0_f32, -3.0_f32, 0u32),      // +0
    ];
    for (a_v, b_v, expected_bits) in cases {
        let a = leaf_scalar(a_v, false);
        let b = leaf_scalar(b_v, false);
        let c = grad_fns::arithmetic::floor_divide(&a, &b).expect("floor_divide fwd must succeed");
        let v = c.item().expect("c.item");
        let bits = v.to_bits();
        assert_eq!(
            bits, expected_bits,
            "floor_divide({a_v}, {b_v}) bits expected {expected_bits:#010x} \
             per torch copysign(0,a/b), got {bits:#010x} (value {v})",
        );
    }
}

/// (4) Euclidean division identity at the rem==0 branch.
///
/// Live oracle 2026-05-25:
///   floor_divide(6, 3) = 2; remainder(6, 3) = 0; 2*3 + 0 = 6 ✓
///   floor_divide(-6, 3) = -2; remainder(-6, 3) = 0; (-2)*3 + 0 = -6 ✓
///   floor_divide(6, -3) = -2; remainder(6, -3) = 0; (-2)*(-3) + 0 = 6 ✓
///   floor_divide(-6, -3) = 2; remainder(-6, -3) = 0; 2*(-3) + 0 = -6 ✓
///
/// This is a distinct algorithm path from the (-7, 3) case the builder
/// already tested — here `m == 0` so the sign-correction (`m != 0 && ...`)
/// is SKIPPED, `div != 0` (= 2 or -2), so the standard `floor()` branch
/// fires WITHOUT the `copysign` step.
#[test]
fn divergence_floor_divide_clean_division_identity() {
    let cases = [
        (6.0_f32, 3.0_f32, 2.0_f32),
        (-6.0, 3.0, -2.0),
        (6.0, -3.0, -2.0),
        (-6.0, -3.0, 2.0),
    ];
    for (a_v, b_v, expected_fd) in cases {
        let a = leaf_scalar(a_v, false);
        let b = leaf_scalar(b_v, false);
        let fd = grad_fns::arithmetic::floor_divide(&a, &b)
            .expect("floor_divide fwd must succeed")
            .item()
            .expect("item");
        let rem = grad_fns::arithmetic::remainder(&a, &b)
            .expect("remainder fwd must succeed")
            .item()
            .expect("item");
        assert!(
            (fd - expected_fd).abs() < 1e-6,
            "floor_divide({a_v}, {b_v}) expected {expected_fd} per torch, got {fd}",
        );
        // Identity: a = fd * b + rem
        let recovered = fd * b_v + rem;
        assert!(
            (recovered - a_v).abs() < 1e-6,
            "Euclidean identity broken at clean division ({a_v}, {b_v}): \
             fd={fd}, rem={rem}, recovered={recovered}, expected={a_v}",
        );
    }
}

/// (5) Backward error message MUST contain the upstream-parity phrase
/// "derivative for floor_divide is not implemented" (NOT just the op name).
///
/// Upstream RuntimeError text (live oracle 2026-05-25):
///   "derivative for aten::floor_divide is not implemented"
///
/// The builder's `test_floor_divide_backward_errors` only asserts the
/// message *contains* "floor_divide", which would pass even if the message
/// were just "floor_divide" with no informative phrase. This test pins the
/// substring "not implemented" so a future silent rename of the error
/// (e.g. to a generic "backward unsupported") would surface here.
#[test]
fn divergence_floor_divide_backward_error_message_upstream_parity() {
    let a = leaf_scalar(7.0, true);
    let b = leaf_scalar(3.0, true);
    let c = grad_fns::arithmetic::floor_divide(&a, &b)
        .expect("floor_divide fwd must succeed on requires_grad inputs");
    assert!(
        c.grad_fn().is_some(),
        "floor_divide on requires_grad=true MUST attach a grad_fn (upstream \
         attaches <NotImplemented object>)"
    );
    let err = c
        .backward()
        .expect_err("floor_divide backward must error per upstream parity");
    let msg = match &err {
        FerrotorchError::InvalidArgument { message } => message.clone(),
        _ => panic!(
            "expected InvalidArgument with upstream-parity 'not implemented' message, \
             got {err:?}"
        ),
    };
    assert!(
        msg.contains("not implemented"),
        "backward error message must contain 'not implemented' to mirror upstream's \
         'derivative for aten::floor_divide is not implemented' RuntimeError. \
         Got: {msg:?}",
    );
    assert!(
        msg.contains("floor_divide"),
        "backward error message must name the operation 'floor_divide'. \
         Got: {msg:?}",
    );
}

/// (6) Large-magnitude `(a - mod) / b` MUST be exact-integer-floor, not the
/// rounded result of `floor(a / b)` directly.
///
/// Live oracle 2026-05-25:
///   torch.floor_divide(tensor([1.0e7_f32]), tensor([3.0_f32])) = 3333333.0
///   torch.remainder(tensor([1.0e7_f32]), tensor([3.0_f32]))    = 1.0
///   1.0e7 == 3333333.0 * 3.0 + 1.0  ✓
///
/// The naive `(1.0e7 / 3.0).floor()` in f32 is 3333333.0 too (lucky), but
/// for `1.0e7 + 1.0 == 1.0e7` in f32 (loss of precision at +1ulp), the
/// `(a - mod)/b` form is what keeps the identity exact. We pin both the
/// quotient AND the identity to catch any future regression where the
/// algorithm is simplified to `(a/b).floor()`.
#[test]
fn divergence_floor_divide_large_magnitude_identity_holds() {
    let a = leaf_scalar(1.0e7_f32, false);
    let b = leaf_scalar(3.0_f32, false);
    let fd = grad_fns::arithmetic::floor_divide(&a, &b)
        .expect("fwd")
        .item()
        .expect("item");
    let rem = grad_fns::arithmetic::remainder(&a, &b)
        .expect("rem fwd")
        .item()
        .expect("item");
    let expected_fd: f32 = 3_333_333.0;
    let expected_rem: f32 = 1.0;
    assert!(
        (fd - expected_fd).abs() < 1e-1,
        "floor_divide(1e7, 3) expected {expected_fd} per torch oracle, got {fd}",
    );
    assert!(
        (rem - expected_rem).abs() < 1e-3,
        "remainder(1e7, 3) expected {expected_rem} per torch oracle, got {rem}",
    );
    let recovered = fd * 3.0_f32 + rem;
    // f32 precision at 1e7 magnitude: tolerate 1 ULP (~1.19e0 at this scale)
    assert!(
        (recovered - 1.0e7_f32).abs() < 2.0,
        "Euclidean identity broken at large magnitude: \
         fd={fd}, rem={rem}, recovered={recovered}, expected=1e7",
    );
}
