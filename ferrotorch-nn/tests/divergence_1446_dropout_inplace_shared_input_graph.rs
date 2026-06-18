//! Divergence probe for in-place dropout (#1446) on a shared-input autograd graph.
//!
//! Upstream contract: `torch/nn/functional.py:1448-1450`
//!   `_VF.dropout_(input, p, training) if inplace else _VF.dropout(...)`.
//! When `inplace=True` mutates a tensor whose pre-mutation value is needed by
//! ANOTHER op's backward, PyTorch's autograd version-counter raises a
//! `RuntimeError`:
//!   "one of the variables needed for gradient computation has been modified
//!    by an inplace operation: ... is at version 1; expected version 0".
//! Confirmed live against torch 2.11.0+cu130:
//!   x leaf; t = x*x; y = t*t [MulBackward saves t]; z = Dropout(p, inplace=True)(t)
//!   [mutates t]; (y.sum()).backward()  ->  RuntimeError.
//!
//! ferrotorch has NO version counter (grep `version` in
//! `ferrotorch-core/src/tensor.rs` / `autograd/` returns nothing). `Tensor::clone`
//! shares the `Arc<TensorInner>` storage (`tensor.rs:1840`), and `write_inplace`
//! (`ferrotorch-nn/src/dropout.rs:169`, used by the `if self.inplace` branch at
//! `dropout.rs:419-421`) calls `Tensor::update_data` (`tensor.rs:1221`) which copies
//! straight into that shared storage. `MulBackward`
//! (`ferrotorch-core/src/grad_fns/arithmetic.rs:912`) saves `t` as a storage-sharing
//! clone and reads `self.a`/`self.b` during backward. So the in-place dropout
//! silently corrupts `y`'s gradient instead of raising.
//!
//! The SAFETY comment in `write_inplace` (`dropout.rs:170-179`) asserts "no
//! backward node captures a read view of the forward input's values" — that
//! invariant is FALSE in any graph where the dropout input is also consumed by
//! another differentiable op that saves it. This test pins that false invariant.

use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_nn::{Dropout, Module};

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

/// Divergence: ferrotorch `Dropout::forward(.., inplace=true)` mutates the
/// input storage shared (via Arc) with a tensor that `MulBackward` saved for a
/// DIFFERENT branch's gradient. Upstream PyTorch raises a RuntimeError
/// (version-counter); ferrotorch silently corrupts `y`'s gradient and returns
/// `Ok`, computing the gradient from the post-dropout (dropped→0, kept→2x) `t`.
///
/// With `x = [1..8]`, `t = x*x`, `loss = sum((t*t))`, the gradient
/// `d loss / d x = 4*x^3` is mathematically INDEPENDENT of the unrelated
/// dropout branch. ferrotorch instead returns `4*x^3` scaled per-element by the
/// random dropout mask applied to `t` (0 for dropped, 2x for kept), e.g. an
/// observed `[0, 0, 0, 512, 1000, 0, 2744, 4096]` vs the correct
/// `[4, 32, 108, 256, 500, 864, 1372, 2048]`.
///
/// Upstream: torch/nn/functional.py:1449 + autograd version counter.
/// ferrotorch: ferrotorch-nn/src/dropout.rs:419-421 (write_inplace) with no guard.
/// Tracking: #1580 (blocker).
#[test]
fn divergence_inplace_dropout_corrupts_shared_branch_grad() {
    let xs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = leaf(&xs, &[8]);

    // t = x*x  (non-leaf, requires grad, no zeros).
    let t = mul(&x, &x).unwrap();
    let t_orig = t.data().unwrap().to_vec();

    // y = t*t -> MulBackward saves t. loss = sum(y). dloss/dx = 4*x^3.
    let y = mul(&t, &t).unwrap();
    let loss = sum(&y).unwrap();

    // In-place dropout on t. `t` is a grad-tracked NON-LEAF whose storage is
    // shared (via Arc) with the copy `MulBackward` saved for `y`'s gradient.
    // The FIX (`apply_inplace_dropout` in `ferrotorch-nn/src/dropout.rs`)
    // declines to mutate that shared storage: ferrotorch has no autograd
    // version counter, so rather than risk silently corrupting `y`'s branch it
    // falls back to out-of-place. The post-fix contract is therefore that `t`'s
    // storage is LEFT UNMUTATED (or, alternatively, that backward errors as
    // torch's version counter would).
    let d = Dropout::<f32>::new(0.5).unwrap().with_inplace(true);
    let z = d.forward(&t).unwrap();

    let t_after = t.data().unwrap().to_vec();
    assert_eq!(
        t_orig, t_after,
        "FIX: in-place dropout on a grad-tracked non-leaf must NOT mutate the \
         shared storage (it falls back to out-of-place); torch raises via its \
         version counter, ferrotorch declines to mutate. Either way the saved \
         copy for y's MulBackward stays intact."
    );

    // PyTorch raises RuntimeError here. ferrotorch must EITHER error (matching
    // torch's version-counter contract) OR deliver the CORRECT gradient
    // 4*x^3 computed from the ORIGINAL t — never the mutated t.
    match loss.backward() {
        Err(_) => { /* acceptable: matches torch's RuntimeError contract */ }
        Ok(()) => {
            let g = x
                .grad()
                .unwrap()
                .expect("input gradient should be populated");
            let g = g.data().unwrap().to_vec();
            let expected: Vec<f32> = xs.iter().map(|&xi| 4.0 * xi * xi * xi).collect();
            for (i, (&got, &exp)) in g.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() <= 1e-3 * exp.max(1.0),
                    "grad[{i}] corrupted by inplace dropout: got {got}, expected {exp} \
                     (4*x^3 from ORIGINAL t). Upstream torch raises RuntimeError; \
                     ferrotorch silently used the post-dropout t.",
                );
            }
        }
    }
    let _ = z;
}
