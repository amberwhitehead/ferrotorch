//! Divergence pin for #1627 — the FINAL circular-legality residual of the
//! negative-pad chain (#1620 #1621 #1623 #1624 #1626).
//!
//! METHOD (R-CHAR-3): brute-forced the full circular accept/reject grid for
//! `torch.nn.functional.pad(x, [lo, hi], mode="circular")` over sizes 1..6 and
//! all `lo,hi` in `-(size+2)..=(size+2)` against LIVE torch 2.11.0+cu130, then
//! compared ferrotorch's `functional_pad_1d_signed`. After the #1624 fix, the
//! ONLY surviving accept-mismatches (ferrotorch ACCEPTS — returns empty `[..,0]`
//! — where torch ERRORS) are exactly 7 grid points, all sharing:
//!   out_w == size + lo + hi == 0,  lo > 0,  hi < 0,  |hi| > size.
//!     size=3 lo=1 hi=-4   size=4 lo=1 hi=-5   size=4 lo=2 hi=-6
//!     size=5 lo=1 hi=-6   size=5 lo=2 hi=-7   size=6 lo=1 hi=-7   size=6 lo=2 hi=-8
//!
//! UPSTREAM (pytorch working tree). `_pad_circular_symint`
//! (`aten/src/ATen/native/PadNd.cpp:110-189`) enforces THREE legality gates:
//!   - `:140-142` `TORCH_CHECK(pad_l <= size && pad_r <= size, "Padding value
//!     causes wrapping around more than once.")` (already mirrored, #1624).
//!   - `:143-145` `TORCH_CHECK(out_shape >= 0, "Negative padding value is
//!     resulting in an empty dimension")` — net-zero is allowed as an empty dim
//!     (already mirrored, #1624).
//!   - `:158-161` the CENTER copy `out_slice.copy_(in_slice)`, where
//!     `out_slice = out.slice_symint(dim, max(lo,0), out_w - max(hi,0))` has size
//!     `outc` and `in_slice = self.slice_symint(dim, max(-lo,0), size - max(-hi,0))`
//!     has size `inc`. `copy_` broadcasts, so it RAISES `RuntimeError` iff
//!     `outc != inc && outc != 1 && inc != 1`. For the 7 cases `outc = 0` and
//!     `inc = |hi| - size >= 2`, so torch raises "The size of tensor a (0) must
//!     match the size of tensor b (N)".
//!
//! Every expected value/acceptance below is from the live torch oracle; none is
//! copied from ferrotorch (ferrotorch ACCEPTS the divergence cases pre-fix).
//!
//! Tracking: #1627.

use ferrotorch_core::Tensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_nn::padding::{PaddingMode, functional_pad_1d_signed};

fn tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

// ===========================================================================
// DIVERGENCE — circular net-zero over-crop where the opposite-side window is
// larger than the cropped center -> torch's center copy_ broadcast-mismatch.
// ferrotorch ACCEPTS (returns empty [..,0]) pre-fix; torch ERRORS.
// ===========================================================================

/// Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[[1., 2., 3.]]])          # size 3
/// F.pad(x, [1, -4], mode="circular")
/// # RuntimeError: The size of tensor a (0) must match the size of tensor b (2)
/// ```
/// `lo=1, hi=-4`: out_w = 3+1-4 = 0; the `:158-161` center copy reads an
/// `inc = |hi|-size = 1 = 4-3 ... = 2`-element source slice into a 0-element
/// destination -> broadcast-mismatch RuntimeError. ferrotorch returns empty
/// `[1,0]` pre-fix (the gather loop runs `0..0`).
///
/// Tracking: #1627
#[test]
fn divergence_circular_overcrop_size3_1_neg4() {
    let x = tensor(&[1.0, 2.0, 3.0], &[1, 3]);
    let r = functional_pad_1d_signed(&x, 1, -4, PaddingMode::Circular, 0.0);
    assert!(
        r.is_err(),
        "torch rejects circular [1,-4] on size 3 (center copy broadcast-mismatch, \
         out=0 vs in=2); ferrotorch must return Err, got {:?}",
        r.ok()
            .map(|t| (t.shape().to_vec(), t.data().unwrap().to_vec()))
    );
}

/// Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[[1., 2., 3., 4.]]])      # size 4
/// F.pad(x, [2, -6], mode="circular")
/// # RuntimeError: The size of tensor a (0) must match the size of tensor b (2)
/// ```
/// `lo=2, hi=-6`: out_w = 4+2-6 = 0; center copy `inc = 6-4 = 2 != outc = 0`.
///
/// Tracking: #1627
#[test]
fn divergence_circular_overcrop_size4_2_neg6() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let r = functional_pad_1d_signed(&x, 2, -6, PaddingMode::Circular, 0.0);
    assert!(
        r.is_err(),
        "torch rejects circular [2,-6] on size 4 (center copy broadcast-mismatch); \
         ferrotorch must return Err, got {:?}",
        r.ok()
            .map(|t| (t.shape().to_vec(), t.data().unwrap().to_vec()))
    );
}

/// Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[[1., 2., 3., 4., 5.]]])  # size 5
/// F.pad(x, [1, -6], mode="circular")
/// # RuntimeError: The size of tensor a (0) must match the size of tensor b (4)
/// ```
/// `lo=1, hi=-6`: out_w = 5+1-6 = 0; center copy `inc = 6-5+... = 4 != outc = 0`.
///
/// Tracking: #1627
#[test]
fn divergence_circular_overcrop_size5_1_neg6() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5]);
    let r = functional_pad_1d_signed(&x, 1, -6, PaddingMode::Circular, 0.0);
    assert!(
        r.is_err(),
        "torch rejects circular [1,-6] on size 5 (center copy broadcast-mismatch); \
         ferrotorch must return Err, got {:?}",
        r.ok()
            .map(|t| (t.shape().to_vec(), t.data().unwrap().to_vec()))
    );
}

/// Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[[1., 2., 3., 4., 5., 6.]]])  # size 6
/// F.pad(x, [2, -8], mode="circular")
/// # RuntimeError: The size of tensor a (0) must match the size of tensor b (4)
/// ```
/// `lo=2, hi=-8`: out_w = 6+2-8 = 0; center copy `inc = 8-6 = ... 4 != outc = 0`.
///
/// Tracking: #1627
#[test]
fn divergence_circular_overcrop_size6_2_neg8() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 6]);
    let r = functional_pad_1d_signed(&x, 2, -8, PaddingMode::Circular, 0.0);
    assert!(
        r.is_err(),
        "torch rejects circular [2,-8] on size 6 (center copy broadcast-mismatch); \
         ferrotorch must return Err, got {:?}",
        r.ok()
            .map(|t| (t.shape().to_vec(), t.data().unwrap().to_vec()))
    );
}

// ===========================================================================
// REGRESSION GUARDS — net-zero over-crop cases torch ACCEPTS (returns empty
// `[..,0]`). The fix must NOT over-reject these. All have outc == inc (== 0)
// or inc == 1 (broadcastable into the empty destination). Expected from the
// same live torch 2.11.0+cu130 oracle.
// ===========================================================================

/// Live torch: `F.pad([[1,2]], [1,-3], mode="circular")` -> empty `[1,0]`
/// (`lo=1, hi=-3, size=2`: out_w=0, center inc = `|hi|-size = 1` broadcasts into
/// the 0-length destination, so torch accepts). ferrotorch must still accept.
#[test]
fn regression_circular_overcrop_size2_1_neg3_accepts() {
    let x = tensor(&[1.0, 2.0], &[1, 2]);
    let y = functional_pad_1d_signed(&x, 1, -3, PaddingMode::Circular, 0.0)
        .expect("torch circular [1,-3] on size 2 returns empty [1,0]; ferrotorch must accept");
    assert_eq!(y.shape(), &[1, 0]);
    assert_eq!(y.data().unwrap().len(), 0);
}

/// Live torch: `F.pad([[1,2]], [2,-4], mode="circular")` -> empty `[1,0]`
/// (`lo=2, hi=-4, size=2`: out_w=0, center inc=0=outc -> accepts).
#[test]
fn regression_circular_overcrop_size2_2_neg4_accepts() {
    let x = tensor(&[1.0, 2.0], &[1, 2]);
    let y = functional_pad_1d_signed(&x, 2, -4, PaddingMode::Circular, 0.0)
        .expect("torch circular [2,-4] on size 2 returns empty [1,0]; ferrotorch must accept");
    assert_eq!(y.shape(), &[1, 0]);
    assert_eq!(y.data().unwrap().len(), 0);
}

/// Live torch: `F.pad([[1,2,3]], [2,-5], mode="circular")` -> empty `[1,0]`
/// (`lo=2, hi=-5, size=3`: out_w=0, center inc=1 broadcasts -> accepts).
#[test]
fn regression_circular_overcrop_size3_2_neg5_accepts() {
    let x = tensor(&[1.0, 2.0, 3.0], &[1, 3]);
    let y = functional_pad_1d_signed(&x, 2, -5, PaddingMode::Circular, 0.0)
        .expect("torch circular [2,-5] on size 3 returns empty [1,0]; ferrotorch must accept");
    assert_eq!(y.shape(), &[1, 0]);
    assert_eq!(y.data().unwrap().len(), 0);
}
