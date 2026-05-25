//! Divergence-coverage test for #1200 (addcmul REQ-15) audit (commit `9036936a0`).
//!
//! The commit ships `arithmetic::addcmul<T: Float>` + `AddcmulBackward` at
//! `ferrotorch-core/src/grad_fns/arithmetic.rs:2963` / `:2820`. The backward
//! per `tools/autograd/derivatives.yaml:250-253`:
//!
//! ```yaml
//! - name: addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
//!   self    : handle_r_to_c(self.scalar_type(), grad)
//!   tensor1 : handle_r_to_c(tensor1.scalar_type(), grad * (tensor2 * value).conj())
//!   tensor2 : handle_r_to_c(tensor2.scalar_type(), grad * (tensor1 * value).conj())
//! ```
//!
//! For `T: Float` (real-only) `handle_r_to_c` / `.conj()` are identities. The
//! VJP simplifies to:
//! - `d_input   = grad`                       (reduced to input.shape())
//! - `d_tensor1 = grad * value * tensor2`     (reduced to tensor1.shape())
//! - `d_tensor2 = grad * value * tensor1`     (reduced to tensor2.shape())
//!
//! `AddcmulBackward::backward` must invoke `reduce_grad_to_shape` on each
//! leaf's incoming gradient so that broadcast reductions land on the original
//! operand shape (mirrors `AddBackward` / `MulBackward`). The builder's own
//! `test_addcmul_backward_value_two` uses three same-shape `[2]` leaves, so
//! NEITHER the 1-leaf-vs-output broadcast reduction NOR the 3-way mixed
//! broadcast backward is exercised by the commit's own unit tests.
//!
//! This audit test fills the gap with the audit's prescribed shapes:
//!
//! ```text
//! input = tensor([1., 2., 3.], requires_grad=True)              # shape [3]
//! t1    = tensor([[4.], [5.]], requires_grad=True)              # shape [2,1]
//! t2    = tensor([[7.,8.,9.],[10.,11.,12.]], requires_grad=True) # shape [2,3]
//! c = torch.addcmul(input, t1, t2, value=1.0)
//! c.sum().backward()
//! # input.grad = [2., 2., 2.]              (shape [3]; reduced from [2,3] by sum over dim 0)
//! # t1.grad    = [[24.], [33.]]            (shape [2,1]; reduced from [2,3] by sum over dim 1)
//! # t2.grad    = [[4.,4.,4.],[5.,5.,5.]]   (shape [2,3]; not reduced)
//! ```
//!
//! Verified against `torch 2.11.0+cu130` on 2026-05-25 via the parity-sweep
//! oracle (R-CHAR-3: expected values come from torch, not from ferrotorch).
//!
//! Derivation:
//! - `grad_output = ones([2,3])` (sum-loss).
//! - `d_input` raw = `grad = ones([2,3])`; reduce-to-shape `[3]` by summing
//!   over the broadcast dim 0 yields `[2,2,2]`.
//! - `d_t1` raw = `grad * value * t2 = t2 = [[7,8,9],[10,11,12]]`; reduce-to
//!   `[2,1]` by summing over dim 1 yields `[[24],[33]]`.
//! - `d_t2` raw = `grad * value * t1 = t1_broadcast = [[4,4,4],[5,5,5]]`;
//!   reduce-to `[2,3]` is the identity.
//!
//! If `AddcmulBackward` skips `reduce_grad_to_shape` on any leaf, the
//! resulting tensor will have the wrong shape (output shape `[2,3]` instead
//! of the leaf's original shape) and either the shape assertion fires or the
//! values disagree.

use ferrotorch_core::{Tensor, from_vec, grad_fns};

fn leaf(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    from_vec(data.to_vec(), shape)
        .expect("from_vec must succeed")
        .requires_grad_(requires_grad)
}

/// Audit-prescribed shapes: `input=[3]`, `t1=[2,1]`, `t2=[2,3]`, `value=1`.
///
/// Expected (torch oracle 2026-05-25):
/// - `c.shape() = [2, 3]`
/// - `input.grad = [2, 2, 2]` (shape `[3]`)
/// - `t1.grad    = [[24], [33]]` (shape `[2,1]`)
/// - `t2.grad    = [[4,4,4],[5,5,5]]` (shape `[2,3]`)
#[test]
fn divergence_addcmul_backward_mixed_broadcast() {
    let input = leaf(&[1.0, 2.0, 3.0], &[3], true);
    let t1 = leaf(&[4.0, 5.0], &[2, 1], true);
    let t2 = leaf(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[2, 3], true);

    let c = grad_fns::arithmetic::addcmul(&input, &t1, &t2, 1.0)
        .expect("addcmul fwd must succeed on broadcastable shapes");

    assert_eq!(
        c.shape(),
        &[2, 3],
        "forward output must broadcast to [2,3] per torch (input bcast over dim 0; t1 bcast over dim 1)",
    );

    // Sum to scalar so backward sees a unit grad.
    let loss = grad_fns::reduction::sum(&c).expect("sum must succeed");
    loss.backward().expect("backward must succeed");

    // input.grad: shape [3], values [2, 2, 2].
    let g_input = input
        .grad()
        .expect("input.grad() must succeed")
        .expect("input.grad must be present");
    assert_eq!(
        g_input.shape(),
        &[3],
        "input.grad shape must match input.shape() = [3] (reduce_grad_to_shape over broadcast dim 0)",
    );
    let g_input_data = g_input.data().expect("input.grad data");
    for (i, &got) in g_input_data.iter().enumerate() {
        assert!(
            (got - 2.0_f32).abs() < 1e-6,
            "input.grad[{i}] = {got} (expected 2.0 from torch oracle: \
             sum(ones[2,3], dim=0) = [2,2,2])",
        );
    }

    // t1.grad: shape [2,1], values [[24], [33]].
    let g_t1 = t1
        .grad()
        .expect("t1.grad() must succeed")
        .expect("t1.grad must be present");
    assert_eq!(
        g_t1.shape(),
        &[2, 1],
        "t1.grad shape must match t1.shape() = [2,1] (reduce_grad_to_shape over broadcast dim 1)",
    );
    let g_t1_data = g_t1.data().expect("t1.grad data");
    assert!(
        (g_t1_data[0] - 24.0_f32).abs() < 1e-6,
        "t1.grad[0,0] = {} (expected 24.0 from torch oracle: 7+8+9)",
        g_t1_data[0],
    );
    assert!(
        (g_t1_data[1] - 33.0_f32).abs() < 1e-6,
        "t1.grad[1,0] = {} (expected 33.0 from torch oracle: 10+11+12)",
        g_t1_data[1],
    );

    // t2.grad: shape [2,3], values [[4,4,4],[5,5,5]] (broadcast of t1).
    let g_t2 = t2
        .grad()
        .expect("t2.grad() must succeed")
        .expect("t2.grad must be present");
    assert_eq!(
        g_t2.shape(),
        &[2, 3],
        "t2.grad shape must match t2.shape() = [2,3] (no reduction; t2 was not broadcast)",
    );
    let g_t2_data = g_t2.data().expect("t2.grad data");
    let expected_t2 = [4.0_f32, 4.0, 4.0, 5.0, 5.0, 5.0];
    for (i, exp) in expected_t2.iter().enumerate() {
        assert!(
            (g_t2_data[i] - exp).abs() < 1e-6,
            "t2.grad[{i}] = {} (expected {exp} from torch oracle: \
             t1_broadcast[2,3] = [[4,4,4],[5,5,5]])",
            g_t2_data[i],
        );
    }
}

/// Sanity check for the audit's "simple" backward case (the spec called this
/// out as the baseline that the commit's own `test_addcmul_backward_value_two`
/// covers). Re-derived from torch oracle:
///
/// ```text
/// input=[1,2], t1=[3,4], t2=[5,6], value=2.0
/// d_input = [1, 1]
/// d_t1    = [10, 12]   (= 2*5, 2*6)
/// d_t2    = [6, 8]     (= 2*3, 2*4)
/// ```
///
/// This duplicates the builder's `test_addcmul_backward_value_two` but using
/// `requires_grad_(true)` plumbing through the public surface — if the
/// duplicate passes here too, the prior test isn't tautological.
#[test]
fn divergence_addcmul_backward_simple_value_two_oracle_pinned() {
    let input = leaf(&[1.0, 2.0], &[2], true);
    let t1 = leaf(&[3.0, 4.0], &[2], true);
    let t2 = leaf(&[5.0, 6.0], &[2], true);

    let c = grad_fns::arithmetic::addcmul(&input, &t1, &t2, 2.0)
        .expect("addcmul forward must succeed");
    let loss = grad_fns::reduction::sum(&c).expect("sum must succeed");
    loss.backward().expect("backward must succeed");

    let gi = input.grad().expect("input.grad() ok").expect("present");
    let g1 = t1.grad().expect("t1.grad() ok").expect("present");
    let g2 = t2.grad().expect("t2.grad() ok").expect("present");

    let gi_data = gi.data().expect("gi data");
    let g1_data = g1.data().expect("g1 data");
    let g2_data = g2.data().expect("g2 data");

    // From torch oracle (2026-05-25): see module-level docstring derivation.
    let expected_gi = [1.0_f32, 1.0];
    let expected_g1 = [10.0_f32, 12.0];
    let expected_g2 = [6.0_f32, 8.0];
    for (i, exp) in expected_gi.iter().enumerate() {
        assert!(
            (gi_data[i] - exp).abs() < 1e-6,
            "input.grad[{i}] = {} expected {exp}",
            gi_data[i],
        );
    }
    for (i, exp) in expected_g1.iter().enumerate() {
        assert!(
            (g1_data[i] - exp).abs() < 1e-6,
            "t1.grad[{i}] = {} expected {exp}",
            g1_data[i],
        );
    }
    for (i, exp) in expected_g2.iter().enumerate() {
        assert!(
            (g2_data[i] - exp).abs() < 1e-6,
            "t2.grad[{i}] = {} expected {exp}",
            g2_data[i],
        );
    }
}
