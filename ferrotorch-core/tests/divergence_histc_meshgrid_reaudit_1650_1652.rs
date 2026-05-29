//! Acto-critic RE-AUDIT of commit `7cbb0a27d` (histc skip-oob #1650 +
//! histc default-range #1652a + meshgrid indexing='xy' #1652b).
//!
//! The commit shipped tests for the 2-tensor `'xy'` case, the 2-element
//! default-range case, and the all-equal widen case. This file pins the
//! corners those tests do NOT cover, each cross-checked against LIVE torch
//! 2.11 (named references per R-CHAR-3, recorded below — NOT copied from the
//! ferrotorch side):
//!
//!   torch.meshgrid([1,2,3],[4,5],[6,7], indexing='xy')  (3-TENSOR: torch
//!     swaps ONLY the first two dims, leaving the 3rd grid in place)
//!       shapes  -> [[2,3,2],[2,3,2],[2,3,2]]
//!       grid0   -> [1,1,2,2,3,3, 1,1,2,2,3,3]
//!       grid1   -> [4,4,4,4,4,4, 5,5,5,5,5,5]
//!       grid2   -> [6,7,6,7,6,7, 6,7,6,7,6,7]
//!   torch.meshgrid([1,2,3], indexing='xy')  (1-TENSOR: NO swap, < 2 inputs)
//!       shapes  -> [[3]]   grid0 -> [1,2,3]
//!   torch.meshgrid([1,2,3],[4,5], indexing='ij')  (default unregressed)
//!       grid0   -> [1,1,2,2,3,3] (shape [3,2])
//!   torch.histc(tensor([1,2,3]), bins=4, min=2, max=2)  (EXPLICIT non-zero
//!     min==max -> torch still triggers aminmax inference -> range [1,3])
//!       -> [1,0,1,1]
//!   torch.histc(tensor([0,2,1]), bins=2, min=0, max=2)  (value==min in bin 0,
//!     value==max in LAST bin)
//!       -> [1,2]
//!   torch.histc(tensor([1,2,3]), bins=4, min=5, max=1)  (min>max strictly)
//!       -> RuntimeError "torch.histc: max must be larger than min"
//!
//! CPU-host tests; the skip/inference/swap logic is device-agnostic (it runs
//! before the device branch). The commit message reports the GPU-resident
//! variants pass on the RTX 3090 via divergence_histc_meshgrid_gpu.

use ferrotorch_core::{MeshIndexing, Tensor, TensorStorage, histc, meshgrid, meshgrid_indexing};

fn cpu_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("cpu f32 tensor")
}

fn data_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.data_vec().expect("data")
}

/// RE-AUDIT meshgrid 'xy' 3-TENSOR: torch swaps ONLY the first two tensors and
/// the first two output grids; the trailing tensors/grids are untouched
/// (`pytorch aten/src/ATen/native/TensorShape.cpp:4433-4438,4470-4472`).
/// `torch.meshgrid([1,2,3],[4,5],[6,7], indexing='xy')` -> all grids shape
/// [2,3,2]; grid0=[1,1,2,2,3,3,1,1,2,2,3,3], grid1=[4,4,4,4,4,4,5,5,5,5,5,5],
/// grid2=[6,7,6,7,6,7,6,7,6,7,6,7]. This is the case the commit did NOT test.
#[test]
fn reaudit_meshgrid_xy_three_tensors() {
    let x = cpu_f32(&[1.0, 2.0, 3.0]);
    let y = cpu_f32(&[4.0, 5.0]);
    let z = cpu_f32(&[6.0, 7.0]);
    let grids = meshgrid_indexing(&[x, y, z], MeshIndexing::Xy).expect("meshgrid xy 3-tensor");
    assert_eq!(grids.len(), 3);
    assert_eq!(grids[0].shape(), &[2, 3, 2]);
    assert_eq!(grids[1].shape(), &[2, 3, 2]);
    assert_eq!(grids[2].shape(), &[2, 3, 2]);
    assert_eq!(
        data_f32(&grids[0]),
        vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    );
    assert_eq!(
        data_f32(&grids[1]),
        vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    );
    assert_eq!(
        data_f32(&grids[2]),
        vec![6.0, 7.0, 6.0, 7.0, 6.0, 7.0, 6.0, 7.0, 6.0, 7.0, 6.0, 7.0]
    );
}

/// RE-AUDIT meshgrid 'xy' 1-TENSOR: with < 2 inputs torch performs NO swap
/// (`TensorShape.cpp:4434` guards `tensors.size() >= 2`).
/// `torch.meshgrid([1,2,3], indexing='xy')` -> single grid shape [3] = [1,2,3].
#[test]
fn reaudit_meshgrid_xy_single_tensor_no_swap() {
    let x = cpu_f32(&[1.0, 2.0, 3.0]);
    let grids = meshgrid_indexing(&[x], MeshIndexing::Xy).expect("meshgrid xy 1-tensor");
    assert_eq!(grids.len(), 1);
    assert_eq!(grids[0].shape(), &[3]);
    assert_eq!(data_f32(&grids[0]), vec![1.0, 2.0, 3.0]);
}

/// RE-AUDIT meshgrid 'ij' DEFAULT UNREGRESSED: `meshgrid(..)` must still equal
/// `meshgrid_indexing(.., Ij)` and produce torch's 'ij' layout (shape [3,2],
/// grid0=[1,1,2,2,3,3]) after the indexing refactor.
/// `torch.meshgrid([1,2,3],[4,5], indexing='ij')[0]` -> [1,1,2,2,3,3].
#[test]
fn reaudit_meshgrid_ij_default_unregressed() {
    let x = cpu_f32(&[1.0, 2.0, 3.0]);
    let y = cpu_f32(&[4.0, 5.0]);
    let g = meshgrid(&[x, y]).expect("meshgrid ij default");
    assert_eq!(g[0].shape(), &[3, 2]);
    assert_eq!(g[1].shape(), &[3, 2]);
    assert_eq!(data_f32(&g[0]), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    assert_eq!(data_f32(&g[1]), vec![4.0, 5.0, 4.0, 5.0, 4.0, 5.0]);
}

/// RE-AUDIT histc EXPLICIT non-zero min==max: torch triggers aminmax inference
/// for ANY `min == max` (not only the 0/0 default) per
/// `if (min == max && self.numel() > 0)` (`SummaryOps.cu:328`).
/// `torch.histc(tensor([1,2,3]), bins=4, min=2, max=2)` -> range inferred [1,3]
/// -> [1,0,1,1]. The commit only tested the min==max==0 form.
#[test]
fn reaudit_histc_explicit_nonzero_min_eq_max_infers_range() {
    let input = cpu_f32(&[1.0, 2.0, 3.0]);
    let out = histc(&input, 4, 2.0, 2.0).expect("histc min==max==2 infers data range");
    assert_eq!(data_f32(&out), vec![1.0, 0.0, 1.0, 1.0]);
}

/// RE-AUDIT histc boundary inclusion: value == min lands in bin 0, value == max
/// lands in the LAST bin (getBin clamp, `SummaryOps.cu:41,47-48`).
/// `torch.histc(tensor([0,2,1]), bins=2, min=0, max=2)` -> [1,2] (0->bin0,
/// 1->bin1, 2->last bin = bin1).
#[test]
fn reaudit_histc_boundary_min_and_max_inclusive() {
    let input = cpu_f32(&[0.0, 2.0, 1.0]);
    let out = histc(&input, 2, 0.0, 2.0).expect("histc boundary inclusive");
    assert_eq!(data_f32(&out), vec![1.0, 2.0]);
}

/// RE-AUDIT histc strict min>max still errors: torch raises
/// "torch.histc: max must be larger than min" (the aminmax inference is gated on
/// `min == max`, so a strict `min > max` is a hard error, NOT inference).
/// `torch.histc(tensor([1,2,3]), bins=4, min=5, max=1)` -> RuntimeError.
#[test]
fn reaudit_histc_strict_min_gt_max_errors() {
    let input = cpu_f32(&[1.0, 2.0, 3.0]);
    let r = histc(&input, 4, 5.0, 1.0);
    assert!(
        r.is_err(),
        "histc with min > max must error (torch: max must be larger than min)"
    );
}
