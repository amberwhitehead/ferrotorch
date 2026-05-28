//! Divergence: ferrotorch's 2-D circular signed pad diverges from
//! `torch.nn.functional.pad(..., mode="circular")` for the CROSS-AXIS net-zero
//! case — where ONE padded axis collapses the whole output to empty (`numel 0`)
//! while ANOTHER padded axis's per-axis circular wrap would (in isolation) read
//! out of bounds.
//!
//! This is the surviving residual of the negative-pad chain
//! (#1611/#1620/#1621/#1623/#1624/#1625/#1626/#1627) AFTER the two claimed-final
//! fixes `1a6d16c5e` and `0501b4ec5`. The `0501b4ec5` fix added the center-copy
//! broadcast gate and the per-axis gather pre-validation in
//! `circular_axis_new_size` (`ferrotorch-nn/src/padding.rs:1690-1742`), which
//! correctly handles a SINGLE axis with `out_w == 0`. But that validation runs
//! PER-AXIS, independently, and rejects an axis whose own wrap gather resolves an
//! index outside `0..size` — even when a DIFFERENT padded axis has already
//! collapsed the total output to `numel 0`, so torch never performs that gather
//! at all.
//!
//! ferrotorch cite (`ferrotorch-nn/src/padding.rs:1731-1740`):
//! ```ignore
//! let out_w = out_w as usize;
//! for j in 0..out_w {
//!     let s = circular_axis_src(j, size, lo, hi);
//!     if !(0..size_i).contains(&s) { return Err(InvalidArgument { .. }); }
//! }
//! ```
//! This loop runs per axis inside `circular_axis_new_size`, called once per axis
//! from `pad_nd_signed_reflect_circular` (`padding.rs:1832-1833`). For the H axis
//! it Errs with "...the wrap would read out of bounds..." even though the W axis
//! already set `out_w == 0`, making the whole output empty.
//!
//! UPSTREAM CONTRACT (pytorch `_pad_circular_symint`,
//! `aten/src/ATen/native/PadNd.cpp:140-187`):
//!   - `:138` `out_shape[...] = size + pad_l + pad_r` per padded axis.
//!   - `:140-142` `TORCH_CHECK(pad_l <= size && pad_r <= size, "...wrapping
//!     around more than once.")` — fires per axis on the PAD magnitude only.
//!   - `:143-145` `TORCH_CHECK(out_shape[...] >= 0, "Negative padding ... empty
//!     dimension")` — net extent of EXACTLY 0 is ALLOWED (empty dim).
//!   - `:148` `auto out = self.new_empty_symint(out_shape, ...)`; the center copy
//!     (`:158-161`) and the wrap copies (`:169-187`) all operate on SLICES of
//!     `out`. When ANY `out_shape` axis is 0 the whole `out` has `numel 0`, every
//!     `slice(...).copy_(...)` is a no-op over an empty extent, and torch returns
//!     the well-defined empty tensor — it NEVER materializes any wrap index on
//!     the other axes.
//!
//! So torch's only PER-AXIS legality is `:142` (pad <= size); the gather/center
//! validity is a property of the WHOLE output, not each axis in isolation.
//!
//! Upstream returns a well-defined empty tensor; ferrotorch returns Err.
//!
//! METHOD (R-CHAR-3): every expected shape below is from LIVE torch 2.11.0+cu130
//! (reproducer Python inlined in each doc comment). The exhaustive grid harness
//! `divergence_negpad_chain_close.rs::definitive_negpad_grid_all_modes` flags
//! 3052 such circular reject-mismatches; these are minimal hand-reduced ones,
//! each independently re-checked against live torch.
//!
//! Tracking: #1628

use ferrotorch_core::Tensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_nn::padding::{PaddingMode, functional_pad_2d_signed};

fn tensor(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// Minimal `[1,1,1]` reproducer. Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[[5.]]])                       # shape [1,1,1]
/// y = F.pad(x, [-2, 1, -1, 1], mode="circular")    # flat = [lo_w,hi_w,lo_h,hi_h]
/// y.shape   # torch.Size([1, 1, 0]) — empty, NOT an error
/// ```
/// W axis (size 1) gets `(lo,hi)=(-2,1)` -> `out_w = 1-2+1 = 0` (collapses the
/// whole output). H axis (size 1) gets `(-1,1)` -> `out_h = 1` and its isolated
/// wrap reads source index 1 (OOB for size 1). torch never gathers because the W
/// axis already made the output empty; ferrotorch's per-axis
/// `circular_axis_new_size` rejects the H axis ("wrap would read out of bounds").
///
/// Tracking: #1628
#[test]
fn divergence_circular_2d_crossaxis_netzero_111() {
    let x = tensor(&[5.0], &[1, 1, 1]);
    let y = functional_pad_2d_signed(&x, -2, 1, -1, 1, PaddingMode::Circular, 0.0).expect(
        "torch circular [-2,1,-1,1] on [1,1,1] returns empty [1,1,0]; ferrotorch must not Err",
    );
    assert_eq!(
        y.shape(),
        &[1, 1, 0],
        "torch returns the well-defined empty tensor [1,1,0]"
    );
    assert_eq!(y.data().unwrap().len(), 0);
}

/// Non-trivial `[1,2,3]` reproducer (so the empty output is obviously a real
/// well-defined tensor, not a degenerate 1-element edge). Live torch:
/// ```python
/// x = torch.arange(1., 7.).reshape(1, 2, 3)        # [[[1,2,3],[4,5,6]]]
/// y = F.pad(x, [-3, 0, -2, 2], mode="circular")    # W:(-3,0) net0; H:(-2,2) on size2
/// y.shape   # torch.Size([1, 2, 0]) — empty
/// ```
/// W axis (size 3) gets `(-3,0)` -> `out_w = 0` (empties the output). H axis
/// (size 2) gets `(-2,2)` -> `out_h = 2`; its isolated wrap gather resolves an
/// OOB source index, so ferrotorch's per-axis pre-validation rejects, while
/// torch returns the well-defined empty `[1,2,0]`.
///
/// Tracking: #1628
#[test]
fn divergence_circular_2d_crossaxis_netzero_123_w_empties() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
    let y = functional_pad_2d_signed(&x, -3, 0, -2, 2, PaddingMode::Circular, 0.0).expect(
        "torch circular [-3,0,-2,2] on [1,2,3] returns empty [1,2,0]; ferrotorch must not Err",
    );
    assert_eq!(y.shape(), &[1, 2, 0]);
    assert_eq!(y.data().unwrap().len(), 0);
}

/// Mirror: the W net-zero comes from the RIGHT side and the H pad is a positive
/// wrap that would (in isolation) read past the cropped center. Live torch:
/// ```python
/// x = torch.arange(1., 7.).reshape(1, 2, 3)
/// y = F.pad(x, [0, -3, 2, -2], mode="circular")    # W:(0,-3) net0; H:(2,-2) on size2
/// y.shape   # torch.Size([1, 2, 0])
/// ```
///
/// Tracking: #1628
#[test]
fn divergence_circular_2d_crossaxis_netzero_123_right() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
    let y = functional_pad_2d_signed(&x, 0, -3, 2, -2, PaddingMode::Circular, 0.0).expect(
        "torch circular [0,-3,2,-2] on [1,2,3] returns empty [1,2,0]; ferrotorch must not Err",
    );
    assert_eq!(y.shape(), &[1, 2, 0]);
    assert_eq!(y.data().unwrap().len(), 0);
}

/// REGRESSION GUARD (PASSES today): when the W axis collapses to net-zero but the
/// H axis is UNTOUCHED, ferrotorch already accepts and returns the empty tensor
/// (the single-axis #1627 fix). This pins that the residual is specifically the
/// CROSS-AXIS interaction (one axis empties while the OTHER axis's wrap is OOB),
/// not net-zero in general. Live torch:
/// ```python
/// x = torch.arange(1., 7.).reshape(1, 2, 3)
/// y = F.pad(x, [-3, 0, 0, 0], mode="circular")     # W net0; H untouched
/// y.shape   # torch.Size([1, 2, 0])
/// ```
#[test]
fn regression_circular_2d_single_axis_netzero_accepts() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
    let y = functional_pad_2d_signed(&x, -3, 0, 0, 0, PaddingMode::Circular, 0.0)
        .expect("torch circular [-3,0,0,0] on [1,2,3] returns empty [1,2,0]; ferrotorch accepts");
    assert_eq!(y.shape(), &[1, 2, 0]);
    assert_eq!(y.data().unwrap().len(), 0);
}
