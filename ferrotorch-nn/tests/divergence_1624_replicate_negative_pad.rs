//! FINAL audit of the negative-pad chain (#1611/#1620/#1621/#1623/#1624),
//! commit `7d0f05455` ("complete circular pad legality"). The exhaustive
//! re-sweep of ALL FOUR modes (reflect/replicate/circular/constant) vs LIVE
//! torch 2.11.0+cu130 — sizes 1..6, all `(lo,hi)` in `-(size+2)..=(size+2)`,
//! 1-D and a 2-D per-axis sample, every call wrapped in `catch_unwind` —
//! surfaced a residual the chain never touched: `mode="replicate"` with a
//! NEGATIVE (crop) pad that crops a side to size 0 then applies a positive pad
//! PANICS with "attempt to subtract with overflow", while torch returns a
//! clean, deterministic, reproducible value.
//!
//! ROOT CAUSE (read, not asserted): `functional_pad_nd_signed`
//! (`ferrotorch-nn/src/padding.rs:1934-1950`) handles Replicate+negative by
//! composing crop-then-pad: it narrows via the constant signed path, then calls
//! `functional_pad_nd_positive` -> `pad_1d_replicate` / `pad_2d_replicate`. When
//! the crop reduces the axis to size 0, the subsequent positive replicate-pad
//! evaluates `inner - 1` (`pad_1d_replicate:391`) / `h - 1`
//! (`pad_2d_replicate:427`) on a zero-size axis and PANICS. Torch's replicate
//! kernel clamps the gather index to the ORIGINAL boundary
//! (`aten/src/ATen/native/cpu/PaddingKernel.cpp`), so an over-crop still reads
//! the preserved edge — it NEVER panics and NEVER errors for these inputs.
//!
//! METHOD (R-CHAR-3): every expected value below is from the live torch oracle
//! (`F.pad(torch.arange(1,size+1).reshape(...), [lo,hi], mode="replicate")`),
//! reproduced 3x and confirmed identical + finite. NONE are copied from
//! ferrotorch (which PANICS on every case here).
//!
//! VERDICT: release-blocker — a PANIC is never an acceptable substitute for
//! torch's clean value (R-CODE-2). 66 such 1-D grid points + 22 2-D points
//! (all where torch ACCEPTS a finite reproducible value). Tracking: #1625.

use std::panic::{AssertUnwindSafe, catch_unwind};

use ferrotorch_core::Tensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_nn::padding::{PaddingMode, functional_pad_1d_signed, functional_pad_2d_signed};

fn tensor(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn approx(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9)
}

/// Live torch 2.11.0+cu130 (reproducible 3x):
/// ```python
/// x = torch.tensor([[1., 2.]])              # size 2
/// F.pad(x, [2, -2], mode="replicate")       # shape [1, 2], data [1., 1.]
/// ```
/// `hi=-2` crops the right to size 0; torch's clamp gives `[1,1]`. ferrotorch
/// PANICS "attempt to subtract with overflow" because the crop leaves a
/// zero-size axis fed to `pad_1d_replicate`, which does `inner - 1` on
/// `inner == 0` (`padding.rs:391`).
/// Tracking: #1625
#[test]
fn divergence_replicate_1d_pad_left_crop_right_panics() {
    let res = catch_unwind(AssertUnwindSafe(|| {
        let x = tensor(&[1.0, 2.0], &[1, 2]);
        functional_pad_1d_signed(&x, 2, -2, PaddingMode::Replicate, 0.0)
    }));
    let t = res
        .expect("ferrotorch must NOT panic on replicate [2,-2] (torch returns [1,1])")
        .expect("ferrotorch must NOT Err on replicate [2,-2]");
    assert_eq!(t.shape(), &[1, 2]);
    assert!(
        approx(&t.data_vec().unwrap(), &[1.0, 1.0]),
        "torch replicate [2,-2] size2 -> [1,1]"
    );
}

/// Live torch (reproducible 3x):
/// ```python
/// x = torch.tensor([[1., 2.]])
/// F.pad(x, [-2, 1], mode="replicate")       # shape [1, 1], data [2.]
/// ```
/// `lo=-2` crops both source elements off the left (axis -> size 0); torch's
/// replicate clamp still resolves the right pad against the ORIGINAL boundary
/// (element 2) -> `[2.]`. ferrotorch over-crops the `narrow` to size 0 then the
/// positive replicate-pad PANICS on `inner - 1`.
/// Tracking: #1625
#[test]
fn divergence_replicate_1d_deep_left_crop_pad_right_panics() {
    let res = catch_unwind(AssertUnwindSafe(|| {
        let x = tensor(&[1.0, 2.0], &[1, 2]);
        functional_pad_1d_signed(&x, -2, 1, PaddingMode::Replicate, 0.0)
    }));
    let t = res
        .expect("ferrotorch must NOT panic on replicate [-2,1] (torch returns [2.])")
        .expect("ferrotorch must NOT Err on replicate [-2,1]");
    assert_eq!(t.shape(), &[1, 1]);
    assert!(
        approx(&t.data_vec().unwrap(), &[2.0]),
        "torch replicate [-2,1] size2 -> [2.]"
    );
}

/// Live torch (reproducible 3x):
/// ```python
/// x = torch.arange(1,13).reshape(1,3,4).double()    # 3x4 plane
/// F.pad(x, [-4, 1, 0, 0], mode="replicate")  # shape [1,3,1], data [4.,8.,12.]
/// ```
/// `lo=-4` crops W to size 0, `hi=+1` replicates the surviving right-edge column.
/// ferrotorch PANICS "attempt to subtract with overflow" at
/// `pad_2d_replicate` (`padding.rs:427`, `h - 1` style boundary clamp on the
/// zero-size axis).
/// Tracking: #1625
#[test]
fn divergence_replicate_2d_w_overcrop_pad_panics() {
    let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
    let res = catch_unwind(AssertUnwindSafe(|| {
        let x = tensor(&data, &[1, 3, 4]);
        functional_pad_2d_signed(&x, -4, 1, 0, 0, PaddingMode::Replicate, 0.0)
    }));
    let t = res
        .expect("ferrotorch must NOT panic on replicate 2D [-4,1,0,0] (torch -> [1,3,1])")
        .expect("ferrotorch must NOT Err on replicate 2D [-4,1,0,0]");
    assert_eq!(
        t.shape(),
        &[1, 3, 1],
        "torch replicate 2D [-4,1,0,0] on 3x4 -> [1,3,1]"
    );
    assert!(
        approx(&t.data_vec().unwrap(), &[4.0, 8.0, 12.0]),
        "torch replicate 2D [-4,1,0,0] on 3x4 -> [4,8,12]"
    );
}
