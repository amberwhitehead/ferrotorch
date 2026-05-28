//! FINAL audit of the negative-pad chain (#1611/#1620/#1621/#1623/#1624),
//! commit `7d0f05455`. The exhaustive 2-D re-sweep vs LIVE torch 2.11.0+cu130
//! surfaced a reflect residual the chain's 1-D-only reasoning missed: a reflect
//! pad whose NET extent on a spatial axis is EXACTLY 0 behaves DIFFERENTLY in
//! torch's 1-D vs 2-D reflect kernels.
//!
//!   - 1-D reflect (`reflection_pad1d`) demands `output_w >= 1`
//!     -> net-zero ERRORS ("input (W: N) is too small. Calculated output W: 0").
//!   - 2-D reflect (`reflection_pad2d`) only requires `output >= 0`
//!     -> net-zero on the W axis (and the H axis) ACCEPTS, returning an EMPTY
//!     `[..,H,0]` / `[..,0,W]` tensor (verified reproducible 3x against the
//!     live oracle).
//!
//! ferrotorch's `pad_nd_signed_reflect_circular` (`padding.rs:1734-1746`) uses a
//! SINGLE shared `new_size < 1` Err guard for reflect across ALL ndim, so it
//! Errs on the 2-D net-zero case where torch returns the empty tensor. The
//! #1623 fix correctly made reflect's POSITIVE-pad legality SIGNED, and the
//! 1-D net-zero reject is correct — but the 2-D net-zero accept was never
//! audited (the prior round's reflect regression guard was 1-D only).
//!
//! 18 grid points diverge this way in the 2-D sample (all where torch returns
//! a deterministic empty `[..,0,..]` tensor and ferrotorch Errs).
//!
//! METHOD (R-CHAR-3): expected shapes are from the live torch oracle
//! (`F.pad(torch.arange(1,13).reshape(1,3,4).double(), [..], mode="reflect")`),
//! reproduced 3x. ferrotorch Errs on every case here, so NONE are copied from
//! the ferrotorch side.
//!
//! VERDICT: release-blocker — ferrotorch rejects an input torch accepts and
//! returns an empty tensor for. Tracking: #1626.

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::Tensor;
use ferrotorch_nn::padding::{functional_pad_2d_signed, PaddingMode};

fn tensor(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn plane_3x4() -> Tensor<f64> {
    let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
    tensor(&data, &[1, 3, 4])
}

/// Live torch 2.11.0+cu130 (reproducible 3x):
/// ```python
/// x = torch.arange(1,13).reshape(1,3,4).double()
/// F.pad(x, [-4, 0, 0, 0], mode="reflect")   # shape [1, 3, 0], data []
/// ```
/// W net = 4 + (-4) + 0 = 0. 2-D reflect ACCEPTS net-zero (only 1-D reflect
/// requires `>= 1`). ferrotorch Errs via the shared `new_size < 1` guard.
/// Tracking: #1626
#[test]
fn divergence_reflect_2d_w_netzero_left_crop() {
    let x = plane_3x4();
    let y = functional_pad_2d_signed(&x, -4, 0, 0, 0, PaddingMode::Reflect, 0.0).expect(
        "torch 2-D reflect [-4,0,0,0] on 3x4 returns empty [1,3,0]; ferrotorch must not Err",
    );
    assert_eq!(
        y.shape(),
        &[1, 3, 0],
        "torch 2-D reflect W net-zero -> empty [1,3,0]"
    );
    assert_eq!(y.data_vec().unwrap().len(), 0);
}

/// Live torch (reproducible 3x):
/// ```python
/// F.pad(x, [-2, -2, 0, 0], mode="reflect")  # shape [1, 3, 0], data []
/// ```
/// Both-side net-zero crop on W. Tracking: #1626
#[test]
fn divergence_reflect_2d_w_netzero_both_crop() {
    let x = plane_3x4();
    let y = functional_pad_2d_signed(&x, -2, -2, 0, 0, PaddingMode::Reflect, 0.0)
        .expect("torch 2-D reflect [-2,-2,0,0] returns empty [1,3,0]; ferrotorch must not Err");
    assert_eq!(y.shape(), &[1, 3, 0]);
    assert_eq!(y.data_vec().unwrap().len(), 0);
}

/// Live torch (reproducible 3x):
/// ```python
/// F.pad(x, [0, -4, 0, 0], mode="reflect")   # shape [1, 3, 0], data []
/// ```
/// Right-side net-zero crop on W. Tracking: #1626
#[test]
fn divergence_reflect_2d_w_netzero_right_crop() {
    let x = plane_3x4();
    let y = functional_pad_2d_signed(&x, 0, -4, 0, 0, PaddingMode::Reflect, 0.0)
        .expect("torch 2-D reflect [0,-4,0,0] returns empty [1,3,0]; ferrotorch must not Err");
    assert_eq!(y.shape(), &[1, 3, 0]);
    assert_eq!(y.data_vec().unwrap().len(), 0);
}

/// Live torch (reproducible 3x): net-zero on the H axis instead of W.
/// ```python
/// F.pad(x, [0, 0, -3, 0], mode="reflect")   # shape [1, 0, 4], data []
/// ```
/// H net = 3 - 3 = 0. Tracking: #1626
#[test]
fn divergence_reflect_2d_h_netzero_top_crop() {
    let x = plane_3x4();
    let y = functional_pad_2d_signed(&x, 0, 0, -3, 0, PaddingMode::Reflect, 0.0)
        .expect("torch 2-D reflect [0,0,-3,0] returns empty [1,0,4]; ferrotorch must not Err");
    assert_eq!(y.shape(), &[1, 0, 4]);
    assert_eq!(y.data_vec().unwrap().len(), 0);
}

/// Live torch (reproducible 3x): net-zero on the H axis with a surviving W axis.
/// ```python
/// F.pad(x, [-3, 0, 0, -3], mode="reflect")  # shape [1, 0, 1], data []
/// ```
/// W net = 4 - 3 = 1, H net = 3 - 3 = 0 -> empty. Tracking: #1626
#[test]
fn divergence_reflect_2d_both_axes_netzero_h() {
    let x = plane_3x4();
    let y = functional_pad_2d_signed(&x, -3, 0, 0, -3, PaddingMode::Reflect, 0.0)
        .expect("torch 2-D reflect [-3,0,0,-3] returns empty [1,0,1]; ferrotorch must not Err");
    assert_eq!(y.shape(), &[1, 0, 1]);
    assert_eq!(y.data_vec().unwrap().len(), 0);
}
