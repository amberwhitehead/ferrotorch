//! Divergence-coverage test for #1199 (fmod REQ-14) audit (commit `34b8301f2`).
//!
//! The commit ships `arithmetic::fmod<T: Float>` + `FmodBackward` at
//! `ferrotorch-core/src/grad_fns/arithmetic.rs:2286` / `:2168`. The backward
//! per `tools/autograd/derivatives.yaml:717-720`:
//!
//! ```yaml
//! - name: fmod.Tensor(Tensor self, Tensor other) -> Tensor
//!   self : grad
//!   other: -grad * self.div(other, /*rounding_mode=*/"trunc")
//! ```
//!
//! `FmodBackward::backward` routes through `reduce_grad_to_shape` so that
//! when `a` and `b` were broadcast on forward, the gradient is reduced back to
//! each leaf's shape. The builder's own backward tests
//! (`test_fmod_backward_scalar`, `test_fmod_backward_negative_dividend`) both
//! use scalar leaves — neither exercises the `reduce_grad_to_shape` path on
//! a broadcast dimension. This audit test fills that gap with the
//! `trunc`-rounding contract (vs `remainder`'s `floor`-rounding) which is
//! THE numerical hinge that distinguishes the two ops at the backward
//! boundary.
//!
//! Per R-CHAR-3 the expected gradients are sourced from the live PyTorch
//! oracle, NOT from ferrotorch. Verified against `torch 2.11.0+cu130` on
//! 2026-05-25:
//!
//! ```text
//! # CASE A: a.shape=[4], b.shape=[1]
//! a = tensor([5., -5., 5., -5.], requires_grad=True)
//! b = tensor([3.], requires_grad=True)
//! c = torch.fmod(a, b)
//! c.sum().backward()
//! # c      = [2., -2., 2., -2.]    (sign of dividend; cf. remainder = [2., 1., 2., 1.])
//! # a.grad = [1., 1., 1., 1.]      (shape [4])
//! # b.grad = [0.]                  (shape [1], sum of -trunc(a/b) over broadcast axis)
//! ```
//!
//! Derivation: `db[i] = -trunc(a[i]/b[i])` per-element is
//! `[-trunc(5/3), -trunc(-5/3), -trunc(5/3), -trunc(-5/3)] = [-1, 1, -1, 1]`
//! and reduced (summed) to `b.shape() = [1]` is `-1+1-1+1 = 0 -> [0.0]`.
//!
//! CONTRAST WITH REMAINDER backward at the same input: `b.grad = [2.0]`
//! (remainder uses `floor` rounding so `-floor(-5/3) = 2`, while fmod uses
//! `trunc` rounding so `-trunc(-5/3) = 1`). The 1-unit-per-flipped-sign
//! difference exactly mirrors the forward sign-correction divergence.
//!
//! If `FmodBackward` used `floor` instead of `trunc` (a plausible cross-
//! wiring with `RemainderBackward`), `b.grad` would be `[2.0]` and this
//! assertion would FAIL.

use ferrotorch_core::{Tensor, grad_fns};

fn leaf_vec(data: &[f32], requires_grad: bool) -> Tensor<f32> {
    let t = ferrotorch_core::from_vec(data.to_vec(), &[data.len()])
        .expect("from_vec must succeed for f32 host data");
    t.requires_grad_(requires_grad)
}

/// Pin the upstream-derived expected gradients (NOT self-checked vs
/// ferrotorch — per R-CHAR-3).
///
/// `a.shape = [4]`, `b.shape = [1]`: backward must `reduce_grad_to_shape`
/// the per-element `db` to `b.shape`'s [1] by summing across the broadcast
/// axis. The `trunc`-rounding vs `floor`-rounding distinction means b.grad
/// is `0.0` for fmod and `2.0` for remainder at the very same input.
#[test]
fn divergence_fmod_backward_broadcast_b_scalar() {
    let a = leaf_vec(&[5.0, -5.0, 5.0, -5.0], true);
    let b = leaf_vec(&[3.0], true);

    let c = grad_fns::arithmetic::fmod(&a, &b)
        .expect("fmod fwd must succeed on broadcastable shapes");

    // Forward must already be sign-of-dividend (oracle: [2,-2,2,-2]).
    let c_data = c.data().expect("c.data");
    let expected_c = [2.0_f32, -2.0, 2.0, -2.0];
    for (i, exp) in expected_c.iter().enumerate() {
        assert!(
            (c_data[i] - exp).abs() < 1e-6,
            "fmod fwd[{i}] = {} (expected {exp} — sign of dividend)",
            c_data[i],
        );
    }

    let loss = grad_fns::reduction::sum(&c).expect("sum must succeed");
    loss.backward().expect("backward must succeed");

    let ga = a
        .grad()
        .expect("a.grad() must succeed")
        .expect("a.grad must be present");
    let gb = b
        .grad()
        .expect("b.grad() must succeed")
        .expect("b.grad must be present");

    // a.grad shape must match a.shape() = [4]; values from torch oracle.
    assert_eq!(ga.shape(), &[4], "a.grad shape must match a.shape");
    let ga_data = ga.data().expect("a.grad data");
    let expected_ga = [1.0_f32, 1.0, 1.0, 1.0];
    for (i, exp) in expected_ga.iter().enumerate() {
        assert!(
            (ga_data[i] - exp).abs() < 1e-6,
            "a.grad[{i}] = {} (expected {exp})",
            ga_data[i],
        );
    }

    // b.grad shape must match b.shape() = [1]; value reduced from per-element
    // db by summing across the broadcast axis.
    //   db[i] = -trunc(a[i]/b[i])
    //         = [-trunc(5/3), -trunc(-5/3), -trunc(5/3), -trunc(-5/3)]
    //         = [-1, 1, -1, 1]
    //   sum   = 0
    // (Contrast remainder at same input: db reduces to 2.0 via -floor.)
    assert_eq!(
        gb.shape(),
        &[1],
        "b.grad shape must match b.shape (broadcast reduction)"
    );
    let gb_data = gb.data().expect("b.grad data");
    assert!(
        (gb_data[0] - 0.0_f32).abs() < 1e-6,
        "b.grad[0] = {} (expected 0.0 from torch oracle: \
         sum(-trunc([5,-5,5,-5]/3)) = -1+1-1+1 = 0; \
         remainder at same input yields 2.0 — DO NOT cross-wire)",
        gb_data[0]
    );
}

/// Inverse-broadcast: `a.shape = [1]`, `b.shape = [4]`. Reduction is on the
/// `a` operand. From torch oracle 2026-05-25:
///
/// ```text
/// a = tensor([5.], requires_grad=True)
/// b = tensor([3., 4., -2., -7.], requires_grad=True)
/// c = torch.fmod(a, b)                 # = [2., 1., 1., 5.]
/// c.sum().backward()
/// # a.grad = [4.]                      (sum over broadcast axis of [1,1,1,1])
/// # b.grad = [-1., -1., 2., 0.]        (-trunc(5/b[i]) elementwise)
/// ```
///
/// Verifying both:
/// 1) forward of `fmod(5, -7) = 5` is sign-of-dividend (remainder would give -2).
/// 2) backward elementwise `-trunc` matches torch (remainder would give
///    `[-1, -1, -3, 1]` — every entry differs in the sign-flipped quadrants).
#[test]
fn divergence_fmod_backward_broadcast_a_scalar() {
    let a = leaf_vec(&[5.0], true);
    let b = leaf_vec(&[3.0, 4.0, -2.0, -7.0], true);

    let c = grad_fns::arithmetic::fmod(&a, &b)
        .expect("fmod fwd must succeed on broadcastable shapes");

    // Forward sign-of-dividend (all positive since dividend = +5).
    let c_data = c.data().expect("c.data");
    let expected_c = [2.0_f32, 1.0, 1.0, 5.0];
    for (i, exp) in expected_c.iter().enumerate() {
        assert!(
            (c_data[i] - exp).abs() < 1e-6,
            "fmod fwd[{i}] = {} (expected {exp} — all positive since dividend = +5)",
            c_data[i],
        );
    }

    let loss = grad_fns::reduction::sum(&c).expect("sum must succeed");
    loss.backward().expect("backward must succeed");

    let ga = a
        .grad()
        .expect("a.grad() must succeed")
        .expect("a.grad must be present");
    let gb = b
        .grad()
        .expect("b.grad() must succeed")
        .expect("b.grad must be present");

    // a.grad shape = [1]; value = sum of grad_out across broadcast axis = 4.
    assert_eq!(
        ga.shape(),
        &[1],
        "a.grad shape must match a.shape (broadcast reduction)"
    );
    let ga_data = ga.data().expect("a.grad data");
    assert!(
        (ga_data[0] - 4.0_f32).abs() < 1e-6,
        "a.grad[0] = {} (expected 4.0 = sum([1,1,1,1]))",
        ga_data[0]
    );

    // b.grad shape = [4]; values per torch oracle.
    assert_eq!(gb.shape(), &[4], "b.grad shape must match b.shape");
    let gb_data = gb.data().expect("b.grad data");
    // Compute expected from the closed form using TRUNC (NOT from ferrotorch
    // and NOT using `floor` — cross-wire check):
    //   db[i] = -trunc(5 / b[i])
    let b_vals = [3.0_f32, 4.0, -2.0, -7.0];
    let expected_db = [-1.0_f32, -1.0, 2.0, 0.0];
    for (i, exp) in expected_db.iter().enumerate() {
        // Sanity: hand-checked formula against the closed form.
        let derived = -((5.0_f32 / b_vals[i]).trunc());
        assert!(
            (derived - exp).abs() < 1e-7,
            "formula sanity: -trunc(5/{}) = {derived}, declared {exp}",
            b_vals[i],
        );
        assert!(
            (gb_data[i] - exp).abs() < 1e-6,
            "b.grad[{i}] = {} (expected {exp} = -trunc(5/{}); \
             remainder-cross-wired impl would give -floor(5/{}) here)",
            gb_data[i],
            b_vals[i],
            b_vals[i],
        );
    }
}

/// The signed-zero forward edge: `fmod(-0.0, 1.0) = -0.0`. Std::fmod preserves
/// the dividend's signed-zero sign bit. An impl that drops the sign bit (e.g.
/// uses absolute-value semantics, or normalizes -0 to +0) would diverge.
///
/// Verified against torch oracle on 2026-05-25:
///   torch.fmod(tensor([-0.0]), tensor([1.0])) -> tensor([-0.])
///   raw bytes: 0x80000000 (sign bit set, magnitude zero)
///
/// We assert the sign bit is set, not just that the value compares == 0,
/// because `-0.0 == 0.0` is true in IEEE-754 arithmetic.
#[test]
fn divergence_fmod_forward_preserves_signed_zero() {
    let a = leaf_vec(&[-0.0_f32], false);
    let b = leaf_vec(&[1.0_f32], false);

    let c = grad_fns::arithmetic::fmod(&a, &b).expect("fmod fwd");
    let c_data = c.data().expect("c.data");
    let val = c_data[0];

    assert!(
        val.is_sign_negative() && val == 0.0,
        "fmod(-0.0, 1.0) = {val} (bits {:#x}); expected -0.0 (bits 0x80000000) per torch oracle",
        val.to_bits(),
    );
}
