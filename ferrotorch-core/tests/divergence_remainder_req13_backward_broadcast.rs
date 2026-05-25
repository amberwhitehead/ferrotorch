//! Divergence-coverage test for #1198 (remainder REQ-13) audit (commit `cd813e77d`).
//!
//! The commit ships `arithmetic::remainder<T: Float>` + `RemainderBackward` at
//! `ferrotorch-core/src/grad_fns/arithmetic.rs:1979` / `:1865`. The backward
//! per `tools/autograd/derivatives.yaml:1455-1457`:
//!
//! ```yaml
//! - name: remainder.Tensor(Tensor self, Tensor other) -> Tensor
//!   self : grad
//!   other: -grad * self.div(other, /*rounding_mode=*/"floor")
//! ```
//!
//! `RemainderBackward::backward` routes through `reduce_grad_to_shape` so that
//! when `a` and `b` were broadcast on forward, the gradient is reduced back to
//! each leaf's shape (this mirrors `AddBackward` / `MulBackward`). The
//! builder's own backward tests (`test_remainder_backward_scalar`,
//! `test_remainder_backward_negative_dividend`) both use scalar leaves —
//! neither exercises the `reduce_grad_to_shape` path on the broadcast
//! dimension. This audit test fills that gap.
//!
//! Per R-CHAR-3 the expected gradients are sourced from the live PyTorch
//! oracle, NOT from ferrotorch:
//!
//! ```text
//! a = tensor([5., -5., 5., -5.], requires_grad=True)  # shape [4]
//! b = tensor([3.], requires_grad=True)                # shape [1] -> broadcasts to [4]
//! c = torch.remainder(a, b)
//! c.sum().backward()
//! # a.grad = [1., 1., 1., 1.]      (shape [4])
//! # b.grad = [2.]                  (shape [1], reduced from broadcast)
//! ```
//!
//! Verified against `torch 2.11.0+cu130` on 2026-05-25.
//!
//! Derivation: `da[i] = grad_out[i] = 1` summed nowhere yields `[1,1,1,1]`;
//! `db[i] = -grad_out[i] * floor(a[i]/b[i])` per-element is
//! `[-floor(5/3), -floor(-5/3), -floor(5/3), -floor(-5/3)] = [-1, 2, -1, 2]`
//! and reduced (summed) to `b.shape() = [1]` is `-1 + 2 - 1 + 2 = 2 -> [2.0]`.
//!
//! If `RemainderBackward` does not invoke `reduce_grad_to_shape` correctly
//! when only one operand has a broadcastable shape, this assertion will
//! diverge from the torch oracle.

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
/// axis.
#[test]
fn divergence_remainder_backward_broadcast_b_scalar() {
    let a = leaf_vec(&[5.0, -5.0, 5.0, -5.0], true);
    let b = leaf_vec(&[3.0], true);

    let c = grad_fns::arithmetic::remainder(&a, &b)
        .expect("remainder fwd must succeed on broadcastable shapes");

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
    assert_eq!(
        gb.shape(),
        &[1],
        "b.grad shape must match b.shape (broadcast reduction)"
    );
    let gb_data = gb.data().expect("b.grad data");
    assert!(
        (gb_data[0] - 2.0_f32).abs() < 1e-6,
        "b.grad[0] = {} (expected 2.0 from torch oracle: \
         sum(-1*floor([5,-5,5,-5]/3)) = -1+2-1+2 = 2)",
        gb_data[0]
    );
}

/// Inverse-broadcast: `a.shape = [1]`, `b.shape = [4]`. Reduction is on the
/// `a` operand. From torch oracle 2026-05-25:
///
/// ```text
/// a = tensor([5.], requires_grad=True)
/// b = tensor([3., 4., -2., -7.], requires_grad=True)
/// c = torch.remainder(a, b)             # = [2., 1., -1., -2.]
/// c.sum().backward()
/// # a.grad = [4.]                       (sum over broadcast axis of [1,1,1,1])
/// # b.grad = [-1., -1., 3., 1.]         (-floor(5/b[i]) elementwise)
/// ```
#[test]
fn divergence_remainder_backward_broadcast_a_scalar() {
    let a = leaf_vec(&[5.0], true);
    let b = leaf_vec(&[3.0, 4.0, -2.0, -7.0], true);

    let c = grad_fns::arithmetic::remainder(&a, &b)
        .expect("remainder fwd must succeed on broadcastable shapes");

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
    // Compute expected from the closed form (NOT from ferrotorch):
    // db[i] = -1 * floor(5 / b[i])
    let b_vals = [3.0_f32, 4.0, -2.0, -7.0];
    for (i, &bv) in b_vals.iter().enumerate() {
        let expected_db = -((5.0_f32 / bv).floor());
        assert!(
            (gb_data[i] - expected_db).abs() < 1e-6,
            "b.grad[{i}] = {} (expected {expected_db} = -floor(5/{bv}))",
            gb_data[i],
        );
    }
}
