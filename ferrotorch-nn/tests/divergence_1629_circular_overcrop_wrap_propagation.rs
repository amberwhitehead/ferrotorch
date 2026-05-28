//! DIVERGENCE: circular over-crop-then-wrap PROPAGATION — ferrotorch wrongly
//! rejects a DEFINED torch result.
//!
//! Target: commit `fb19ca311` ("holistic circular allocate-then-copy"),
//! `ferrotorch-nn/src/padding.rs` `circular_slicecopy_block`.
//!
//! THE BUG. torch's `_pad_circular_symint` wrap copies
//! (`aten/src/ATen/native/PadNd.cpp:169-187`) read from `out` LIVE — the
//! `out_slice.copy_(in_slice)` at `:179`/`:185` aliases the SAME `out` buffer
//! that the loop is writing, and the comment at `:163-165` is explicit that
//! "Corners will be written more than once" as the wraps run in sequence. When
//! an axis is over-cropped so its center copy writes only a NARROW band (down to
//! a single cell), the subsequent wrap `copy_`s on that axis and the OTHER axes
//! PROPAGATE that band across the whole output deterministically — torch returns
//! a well-defined, reproducible result (verified below: it is a pure gather of a
//! real input element, LINEAR in the input, and bit-identical across 8 fresh
//! processes and under heap pollution).
//!
//! ferrotorch's `circular_slicecopy_block` instead SNAPSHOTS `out`/`init` BEFORE
//! each wrap copy (`padding.rs:1875-1876`, `:1899-1900`), so each wrap reads the
//! state from BEFORE that copy's own writes. For these over-crop propagation
//! cases the snapshot still shows the wrap-source cells as UNINITIALIZED, so the
//! cell never gets its `init` bit set, and the final leftover-uninit R-DEV-6
//! check (`padding.rs:1928-1934`) fires -> ferrotorch returns `Err`. This is a
//! FALSE R-DEV-6 reject: torch's result here is NOT uninitialized garbage, it is
//! a defined propagation that the live-read (no snapshot) produces.
//!
//! EVIDENCE that the torch result is DEFINED (not R-DEV-6 garbage), gathered by
//! the acto-critic with `torch 2.11.0+cu130`:
//!   - in=[1,1,2] data=[a,b], pads(W=-1,2; H=0,1): out is ALWAYS `[1,2,3]` filled
//!     entirely with `b` (= input[..,1]). Verified for [1,2]->all 2, [10,20]->all
//!     20, [100,200]->all 200, [-3,7]->all 7: a pure LINEAR gather (out(k*x) ==
//!     k*out(x)), and bit-identical across 8 fresh processes (no allocator
//!     dependence). Contrast the genuine R-DEV-6 over-crops (e.g.
//!     in=[1,1,2] pads(W=2,-1; H=1,0)) which return NON-linear allocator-varying
//!     denormals — those ferrotorch correctly rejects.
//!
//! The expected values below come from that live-torch oracle (R-CHAR-3); none
//! are copied from ferrotorch (ferrotorch Errs, so it has no value to copy).
//!
//! Tracking: #1629  (negative-pad chain: refs #1611/#1620/#1621/#1623/#1624/
//! #1625/#1626/#1627/#1628 — this reopens the CODE chain).

use ferrotorch_core::Tensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_nn::padding::{PaddingMode, functional_pad_2d_signed};

fn tensor(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// Smallest case. in `[1,1,2]` = `[1,2]`, W pad `(-1,+2)`, H pad `(0,+1)`.
/// torch (`PadNd.cpp:148-187`) returns `[1,2,3]` filled with `input[..,1]` = `2.0`.
/// ferrotorch Errs (false R-DEV-6 leftover-uninit) — DIVERGENCE.
///
/// Tracking: #1629
#[test]
#[ignore = "divergence: circular over-crop wrap-propagation; torch returns defined [1,2,3] all-2.0 (pure linear gather of input[..,1], stable across 8 fresh processes), ferro false-R-DEV-6 Errs due to snapshot-before-wrap (padding.rs:1875); tracking #1629"]
fn divergence_circular_overcrop_propagation_112_w_neg1_2_h_0_1() {
    let x = tensor(&[1.0, 2.0], &[1, 1, 2]);
    // torch: F.pad(x, [-1, 2, 0, 1], mode='circular') -> shape [1,2,3], all 2.0.
    let y = functional_pad_2d_signed(&x, -1, 2, 0, 1, PaddingMode::Circular, 0.0).expect(
        "ferrotorch rejects this DEFINED circular pad (false R-DEV-6 leftover-uninit); \
         torch returns [1,2,3] all-2.0",
    );
    assert_eq!(y.shape(), &[1, 2, 3], "torch out_shape (PadNd.cpp:138)");
    assert_eq!(
        y.data().unwrap().to_vec(),
        vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        "torch fills the whole output with input[..,1]=2.0 via live-read wrap propagation"
    );
}

/// Larger pure-propagation case. in `[1,2,2]` = `[1,2,3,4]`, W `(-1,+2)`,
/// H `(1,+1)`. torch returns `[1,4,3]`; rows alternate `input[..,1,1]=4.0` /
/// `input[..,0,1]=2.0` (a defined H-circular stack of the propagated W column).
/// ferrotorch Errs. (Oracle values from live torch 2.11.)
///
/// Tracking: #1629
#[test]
#[ignore = "divergence: circular over-crop wrap-propagation 2-D; torch returns defined [1,4,3], ferro false-R-DEV-6 Errs; tracking #1629"]
fn divergence_circular_overcrop_propagation_122_w_neg1_2_h_1_1() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    // torch: F.pad(x, [-1, 2, 1, 1], mode='circular') -> [1,4,3]
    //   [4,4,4, 2,2,2, 4,4,4, 2,2,2]
    let y = functional_pad_2d_signed(&x, -1, 2, 1, 1, PaddingMode::Circular, 0.0)
        .expect("ferrotorch rejects this DEFINED circular pad; torch returns [1,4,3]");
    assert_eq!(y.shape(), &[1, 4, 3], "torch out_shape");
    assert_eq!(
        y.data().unwrap().to_vec(),
        vec![4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0],
        "torch propagates the over-cropped W column then H-wraps it"
    );
}

/// Mixed (non-uniform) propagation: in `[1,1,4]` = `[1,2,3,4]`, W `(-1,+4)`,
/// H `(0,+1)`. torch returns `[1,2,7]` =
/// `[2,3,4, 2,3,4, 2, 2,3,4, 2,3,4, 2]`. The over-cropped W center (`[2,3,4]`)
/// is wrap-extended then H-replicated — a non-trivial DEFINED pattern (it is NOT
/// a single constant, ruling out a coincidental-uniform-garbage explanation).
/// ferrotorch Errs.
///
/// Tracking: #1629
#[test]
#[ignore = "divergence: circular over-crop wrap-propagation (non-uniform); torch returns defined [1,2,7], ferro false-R-DEV-6 Errs; tracking #1629"]
fn divergence_circular_overcrop_propagation_114_w_neg1_4_h_0_1() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    // torch: F.pad(x, [-1, 4, 0, 1], mode='circular') -> [1,2,7]
    let y = functional_pad_2d_signed(&x, -1, 4, 0, 1, PaddingMode::Circular, 0.0)
        .expect("ferrotorch rejects this DEFINED circular pad; torch returns [1,2,7]");
    assert_eq!(y.shape(), &[1, 2, 7], "torch out_shape");
    assert_eq!(
        y.data().unwrap().to_vec(),
        vec![
            2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 2.0, // row 0
            2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 2.0, // row 1 (H-wrap of row 0)
        ],
        "torch wrap-extends the over-cropped center [2,3,4] then H-replicates"
    );
}
