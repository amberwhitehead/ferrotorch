//! DIVERGENCE (#1631, surfaced during the #1629 close-audit): the circular
//! over-crop wrap-PROPAGATION FORWARD fix (`circular_slicecopy_block`, commit
//! `2dfd7cd83`) is NOT matched by the BACKWARD. `PadNdSignedModeBackward`
//! (`ferrotorch-nn/src/padding.rs:2191-2237`) still computes the gradient
//! source index with the OLD per-axis `circular_axis_src` map
//! (`padding.rs:1617-1636`), which for an over-cropped axis returns an index
//! `>= size`, so the scatter `grad_in[in_base + src_lin] += ...`
//! (`padding.rs:2230`) PANICS with `index out of bounds` — an R-CODE-2
//! violation (the public API must never panic; it must Err or compute).
//!
//! Upstream contract (R-CHAR-3, values from live torch 2.11.0+cu130):
//!   x = [[[1.0, 2.0]]]            # shape [1,1,2]
//!   y = F.pad(x, [-1,2,0,1], 'circular')      # FORWARD -> [1,2,3], all 2.0
//!   y.sum().backward()
//!   x.grad  ==  [[[0.0, 4.0]]]    # input[0] dropped by the -1 crop; input[1]
//!                                 # gathered into all 6 output cells -> grad 4
//!   (also: x=[[[1,2,3,4]]], pad [-1,4,0,1] -> grad [[[0,4,4,4]]])
//!
//! ferrotorch FORWARD already matches (the #1629 fix); ferrotorch BACKWARD
//! panics. The forward and backward MUST use the same holistic gather.
//!
//! Tracking: #1631 (negative-pad chain: refs #1611/#1620/#1621/#1623/#1624/
//! #1625/#1626/#1627/#1628/#1629).

use std::panic::{AssertUnwindSafe, catch_unwind};

use ferrotorch_core::Tensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_nn::padding::{PaddingMode, functional_pad_2d_signed};

fn tensor(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

/// The minimal #1629 over-crop propagation case. FORWARD is `[2,2,2,2,2,2]`
/// (already correct). BACKWARD: torch grad is `[0.0, 4.0]`. ferrotorch panics
/// `index out of bounds: the len is 2 but the index is 2` at padding.rs:2230.
///
/// Tracking: #1631
#[test]
fn divergence_circular_overcrop_backward_112_w_neg1_2_h_0_1() {
    let x = tensor(&[1.0, 2.0], &[1, 1, 2], true);
    let y = functional_pad_2d_signed(&x, -1, 2, 0, 1, PaddingMode::Circular, 0.0)
        .expect("forward (#1629 fix) returns [1,2,3] all-2.0");
    assert_eq!(y.shape(), &[1, 2, 3]);
    assert_eq!(y.data().unwrap().to_vec(), vec![2.0; 6]);

    let s = ferrotorch_core::grad_fns::reduction::sum(&y).expect("sum");
    // The backward currently PANICS (R-CODE-2 violation). Catch it so the test
    // reports a clean failed assertion rather than aborting the process.
    let res = catch_unwind(AssertUnwindSafe(|| {
        ferrotorch_core::backward(&s).expect("backward");
        x.grad()
            .expect("grad fetch")
            .expect("grad present")
            .data()
            .unwrap()
            .to_vec()
    }));
    let grad = res.expect(
        "ferrotorch circular over-crop BACKWARD panics (index OOB at padding.rs:2230); \
         torch grad is [0.0, 4.0] — forward/backward gather mismatch (#1631)",
    );
    // torch: x.grad == [0.0, 4.0].
    assert_eq!(
        grad,
        vec![0.0, 4.0],
        "torch routes grad of all-input[1] output back to input[1] (used 4x); input[0] cropped"
    );
}

/// Non-uniform propagation case. x=[[[1,2,3,4]]], pad [-1,4,0,1] -> forward
/// [1,2,7] (the over-cropped center [2,3,4] wrap-extended). torch grad is
/// [0.0, 4.0, 4.0, 4.0] (input[0] cropped; inputs 1..3 each gathered 4x).
///
/// Tracking: #1631
#[test]
fn divergence_circular_overcrop_backward_114_w_neg1_4_h_0_1() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4], true);
    let y = functional_pad_2d_signed(&x, -1, 4, 0, 1, PaddingMode::Circular, 0.0)
        .expect("forward (#1629 fix) returns [1,2,7]");
    assert_eq!(y.shape(), &[1, 2, 7]);

    let s = ferrotorch_core::grad_fns::reduction::sum(&y).expect("sum");
    let res = catch_unwind(AssertUnwindSafe(|| {
        ferrotorch_core::backward(&s).expect("backward");
        x.grad()
            .expect("grad fetch")
            .expect("grad present")
            .data()
            .unwrap()
            .to_vec()
    }));
    let grad = res
        .expect("ferrotorch circular over-crop BACKWARD panics; torch grad is [0,4,4,4] (#1631)");
    assert_eq!(grad, vec![0.0, 4.0, 4.0, 4.0]);
}
