//! DIVERGENCE PIN — `cdist` p=0 ("zero-norm") does NOT propagate NaN.
//!
//! Discriminator audit of the CORE-122 (#1816) remediation in
//! `ferrotorch-core/src/ops/tensor_ops.rs`. The fix introduced an explicit
//! `Norm::Zero` branch that COUNTS unequal coordinates:
//!
//! ```text
//! Norm::Zero => { if abs_diff != zero { acc += one; } }   // tensor_ops.rs:779
//! ```
//!
//! But upstream's zero-"norm" is NOT a `!= 0` count — it is
//! `min(ceil(|diff|), 1)` summed (`map`), then identity `finish`:
//!
//!   pytorch aten/src/ATen/native/cpu/DistanceOpsKernel.cpp:94
//!     `static inline data_t map(const data_t& diff, const data_t& p)
//!        { return min(ceil(abs(diff)), 1); }`
//!   pytorch aten/src/ATen/native/cuda/DistanceKernel.cu  dists::zero
//!     (identical `min(ceil(abs(diff)), 1)` map, sum reduce, identity finish)
//!
//! For a NaN coordinate difference `d`, `abs(NaN)=NaN`, `ceil(NaN)=NaN`,
//! `min(NaN, 1)=NaN`, and `sum(..NaN..)=NaN` — so torch returns **NaN**.
//! ferrotorch's `abs_diff != zero` is `NaN != 0 == true`, so it counts the
//! coordinate as `1.0`, swallowing the NaN.
//!
//! For every FINITE difference the two agree (`ceil(|d|)>=1 → min=1` matches
//! `d!=0 → +1`; `ceil(0)=0` matches `d==0 → +0`), so this is purely the
//! NaN-propagation edge the explicit-branch fix overlooked. The existing
//! CORE-122 battery (`audit_core122_cdist_pnorm_branches.rs`) only checks
//! finite inputs, so it does not catch this.
//!
//! LIVE torch 2.11.0+cu130 oracle (quoted verbatim):
//! ```text
//! >>> import torch
//! >>> a = torch.tensor([[float('nan')]]); b = torch.tensor([[0.0]])
//! >>> torch.cdist(a, b, 0.0).item()
//! nan
//! >>> a = torch.tensor([[1.0, float('nan'), 3.0]])
//! >>> b = torch.tensor([[1.0, 0.0,         3.0]])
//! >>> torch.cdist(a, b, 0.0).item()
//! nan
//! ```
//!
//! Upstream returns NaN; ferrotorch returned 1.0 pre-fix. Tracking: #1816
//! (CORE-122) — the divergence was introduced and fixed within that issue's
//! open dispatch (discriminator re-audit pin; the `Norm::Zero` accumulate now
//! mirrors `min(ceil(|d|), 1)` with a NaN-preserving clamp). NOT
//! `#[ignore]`d: NaN-swallowing in a shipped norm branch is observably wrong
//! output for valid input — a release blocker.

use ferrotorch_core::ops::tensor_ops::cdist;
use ferrotorch_core::{Tensor, TensorStorage};

fn t2d(data: &[f32], rows: usize, cols: usize) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![rows, cols], false).unwrap()
}

/// Divergence: `cdist(_, _, 0.0)` must propagate NaN per upstream
/// `min(ceil(abs(diff)), 1)` map (DistanceOpsKernel.cpp:94), not count it as
/// an unequal coordinate. Single 1-coord pair with a NaN difference.
#[test]
fn divergence_cdist_p0_single_nan_coord() {
    // diff = NaN - 0.0 = NaN. torch: min(ceil(|NaN|),1) = NaN -> dist NaN.
    let x1 = t2d(&[f32::NAN], 1, 1);
    let x2 = t2d(&[0.0], 1, 1);
    let result = cdist(&x1, &x2, 0.0).unwrap();
    let d = result.data().unwrap();
    assert_eq!(d.len(), 1);
    assert!(
        d[0].is_nan(),
        "cdist p=0 must propagate NaN (torch returns NaN); ferrotorch returned {}",
        d[0]
    );
}

/// Divergence: the NaN must dominate the SUM even when other coordinates are
/// equal. torch: 0 + NaN + 0 = NaN. ferrotorch counts the NaN coord as 1
/// (equal coords add 0), yielding 1.0.
#[test]
fn divergence_cdist_p0_nan_among_equal_coords() {
    let x1 = t2d(&[1.0, f32::NAN, 3.0], 1, 3);
    let x2 = t2d(&[1.0, 0.0, 3.0], 1, 3);
    let result = cdist(&x1, &x2, 0.0).unwrap();
    let d = result.data().unwrap();
    assert_eq!(d.len(), 1);
    assert!(
        d[0].is_nan(),
        "cdist p=0 sum with a NaN coord must be NaN (torch returns NaN); \
         ferrotorch returned {}",
        d[0]
    );
}
