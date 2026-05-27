//! Divergence test: the backward formula for `index_fill` is documented as
//! `grad.index_fill(dim, index, 0)` per
//! `tools/autograd/derivatives.yaml:884-887`. That formula REQUIRES the
//! same `index` tensor accepted by the forward — including negative
//! indices, since upstream's forward accepts them.
//!
//! Because ferrotorch's forward rejects negative indices (see
//! `divergence_index_fill_1249_negative_index_rejected.rs`), the backward
//! cannot be reached for the negative-index case at all. This test pins
//! the END-TO-END parity: a `requires_grad=True` forward through
//! `index_fill` with a negative index, then `.backward()`, must produce a
//! grad whose filled-position column is zero — matching upstream:
//!
//!     >>> x = torch.tensor([[1.,2.,3.],[4.,5.,6.]], requires_grad=True)
//!     >>> idx = torch.tensor([-1])      # wraps to col 2
//!     >>> y = torch.index_fill(x, 1, idx, -1.0)
//!     >>> y.sum().backward()
//!     >>> x.grad
//!     tensor([[1., 1., 0.],
//!             [1., 1., 0.]])
//!
//! ferrotorch currently errors at the forward, so the backward is
//! unreachable — this test FAILS at the forward call (same root cause as
//! the standalone negative-index test) and pins the END-TO-END grad
//! against the live-torch oracle output to guard against a half-fix that
//! only restores forward negative indices without threading them through
//! to `IndexFillBackward.index` for correct grad zeroing.
//!
//! Tracking: blocker (filed by acto-critic).

use ferrotorch_core::grad_fns::indexing::index_fill;
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

#[test]
fn index_fill_backward_negative_index_matches_upstream_grad() {
    // x = [[1,2,3],[4,5,6]], requires_grad=True
    let x: Tensor<f32> = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![2, 3],
        true,
    )
    .unwrap();
    let idx: IntTensor<i64> = IntTensor::from_vec(vec![-1_i64], vec![1]).unwrap();

    let y = index_fill(&x, 1, &idx, -1.0).expect(
        "forward must accept negative index (upstream wraps); see \
         divergence_index_fill_1249_negative_index_rejected for the \
         standalone forward pin",
    );

    // Drive backward via the grad_fn directly with an ones() grad_output
    // (equivalent to `.sum().backward()` since d(sum)/d(y) = ones).
    let gf = y
        .grad_fn()
        .expect("requires_grad=true input must yield grad_fn on y");
    assert_eq!(gf.name(), "IndexFillBackward");
    let grad_output: Tensor<f32> =
        Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32; 6]), vec![2, 3], false).unwrap();
    let grads = gf
        .backward(&grad_output)
        .expect("IndexFillBackward.backward must succeed");
    let g = grads[0]
        .as_ref()
        .expect("expected Some(grad_input) for requires_grad leaf");
    let gd = g.data().expect("g.data");
    // Upstream live-torch oracle, verified directly:
    //   x.grad = [[1, 1, 0], [1, 1, 0]]
    // (col 2 zeroed because idx=-1 wraps to 2; cols 0/1 pass grad through.)
    assert_eq!(
        gd,
        &[1.0_f32, 1.0, 0.0, 1.0, 1.0, 0.0],
        "VJP must zero the wrapped negative-index column (col 2) and pass \
         grad through elsewhere, per derivatives.yaml:884-887 \
         self: grad.index_fill(dim, index, 0)"
    );
}
