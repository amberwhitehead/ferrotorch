//! Discriminator audit of commit `f4703e140` (histc + meshgrid GPU kernels,
//! #1545 / #1535). These tests pin TWO divergences from live torch 2.11.0+cu130
//! that are NOT covered by the already-filed #1650 (CPU histc clamp-vs-skip):
//!
//!   A. `torch.histc(input, bins)` defaults to `min=0, max=0`, and upstream
//!      `_histc_cuda_template` (and `_histc_cpu`) treat `min == max` SPECIALLY:
//!      they recompute the range from the data's `aminmax()`
//!      (`aten/src/ATen/native/cuda/SummaryOps.cu:328-336`):
//!          if (min == max && self.numel() > 0) {
//!              auto [min_tensor, max_tensor] = self.aminmax();
//!              minvalue = min_tensor.item<input_t>();
//!              maxvalue = max_tensor.item<input_t>(); }
//!          if (minvalue == maxvalue) { minvalue -= 1; maxvalue += 1; }
//!      ferrotorch's `histc` (`ferrotorch-core/src/ops/search.rs:267-271`)
//!      instead ERRORS unconditionally when `min_val >= max_val`:
//!          if min_val >= max_val { return Err(InvalidArgument ...) }
//!      This guard runs BEFORE both the CPU and the new GPU branches, so the
//!      single most common call form — `torch.histc(x, bins)` with default
//!      bounds — is rejected outright by ferrotorch on every device.
//!
//!   B. `torch.meshgrid(*tensors, indexing=...)` supports BOTH `'ij'` (default)
//!      and `'xy'` (`aten/src/ATen/native/TensorShape.cpp:4433-4447`). For
//!      `'xy'` upstream swaps the first two tensors (and the first two output
//!      grids), so `meshgrid([1,2,3],[4,5], 'xy')` yields grids of shape
//!      `[2,3]` with `grid0 = [1,2,3,1,2,3]`, `grid1 = [4,4,4,5,5,5]`.
//!      ferrotorch's `meshgrid` (`ferrotorch-core/src/ops/search.rs:312`) takes
//!      NO `indexing` argument — its signature is `meshgrid(tensors)` — and
//!      hard-codes `'ij'`. The `'xy'` contract is unreachable through the
//!      ferrotorch API.
//!
//! Live torch oracle (torch 2.11.0+cu130, recorded here as named references per
//! R-CHAR-3; NOT copied from the ferrotorch side):
//!   torch.histc(tensor([1,2,3,4,5]), bins=4)               -> [1,1,1,2]
//!   torch.histc(tensor([1,2,3,4,5]), bins=4, min=0, max=0) -> [1,1,1,2]
//!   torch.histc(tensor([3,3,3]),     bins=4)               -> [0,0,3,0]
//!   torch.meshgrid([1,2,3],[4,5], indexing='xy')[0]        -> [1,2,3,1,2,3] (shape [2,3])
//!   torch.meshgrid([1,2,3],[4,5], indexing='xy')[1]        -> [4,4,4,5,5,5]
//!
//! These are CPU-host tests (the `min>=max` guard and the `meshgrid` signature
//! are device-agnostic — the divergence reproduces without a GPU).

use ferrotorch_core::{Tensor, TensorStorage, histc, meshgrid};

fn cpu_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("cpu f32 tensor")
}

fn data_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.data_vec().expect("data")
}

/// Divergence A: `ferrotorch_core::histc` rejects the default `min==max==0`
/// bounds, diverging from `torch.histc` which recomputes the range from
/// `self.aminmax()` (`pytorch aten/src/ATen/native/cuda/SummaryOps.cu:328-336`).
/// Upstream `torch.histc(tensor([1,2,3,4,5]), bins=4, min=0, max=0)` returns
/// `[1,1,1,2]` (range inferred as `[1,5]`); ferrotorch returns
/// `Err(InvalidArgument { "histc: min (0) must be < max (0)" })`.
/// Tracking: #1652
#[test]
#[ignore = "divergence: histc(min==max==0) must infer range from data (torch SummaryOps.cu:328); ferrotorch errors; tracking #1652"]
fn divergence_histc_default_minmax_infers_data_range() {
    let input = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    // torch.histc(tensor([1,2,3,4,5]), bins=4) == torch.histc(..., min=0, max=0)
    // -> range inferred [1,5], bins -> [1,1,1,2].
    let out = histc(&input, 4, 0.0, 0.0)
        .expect("histc with default min=max=0 must succeed (torch infers data range)");
    assert_eq!(data_f32(&out), vec![1.0, 1.0, 1.0, 2.0]);
}

/// Divergence A (degenerate sub-case): when `min==max==0` AND the data is all
/// equal, upstream falls through to `minvalue -= 1; maxvalue += 1`
/// (`SummaryOps.cu:333-335`), giving range `[v-1, v+1]`.
/// `torch.histc(tensor([3,3,3]), bins=4)` returns `[0,0,3,0]` (range `[2,4]`,
/// the three `3.0`s land in bin 2). ferrotorch errors at the `min>=max` guard.
/// Tracking: #1652
#[test]
#[ignore = "divergence: histc(min==max==0) all-equal data -> range [v-1,v+1] (torch SummaryOps.cu:333); ferrotorch errors; tracking #1652"]
fn divergence_histc_default_minmax_all_equal_widens_range() {
    let input = cpu_f32(&[3.0, 3.0, 3.0]);
    // torch.histc(tensor([3,3,3]), bins=4) -> range [2,4] -> [0,0,3,0].
    let out = histc(&input, 4, 0.0, 0.0)
        .expect("histc all-equal default range must succeed (torch widens to [v-1,v+1])");
    assert_eq!(data_f32(&out), vec![0.0, 0.0, 3.0, 0.0]);
}

/// Divergence B: `ferrotorch_core::meshgrid` exposes no `indexing` argument and
/// hard-codes `'ij'`, diverging from `torch.meshgrid(*t, indexing='xy')`
/// (`pytorch aten/src/ATen/native/TensorShape.cpp:4433-4447`), which swaps the
/// first two tensors/grids. For inputs `[1,2,3]` and `[4,5]`, upstream `'xy'`
/// returns grids of shape `[2,3]` with `grid0=[1,2,3,1,2,3]`,
/// `grid1=[4,4,4,5,5,5]`. ferrotorch can only produce the `'ij'` result
/// (shape `[3,2]`, `grid0=[1,1,2,2,3,3]`), so its output cannot equal torch's
/// `'xy'` output — this test asserts the `'xy'` contract and fails because the
/// only available API path yields `'ij'`.
/// Tracking: #1652
#[test]
#[ignore = "divergence: meshgrid has no indexing='xy' (torch TensorShape.cpp:4433); ferrotorch hard-codes 'ij'; tracking #1652"]
fn divergence_meshgrid_xy_indexing_unsupported() {
    let a = cpu_f32(&[1.0, 2.0, 3.0]);
    let b = cpu_f32(&[4.0, 5.0]);
    // The ferrotorch API offers only the default 'ij' meshgrid — there is no
    // way to request 'xy'. We compute what ferrotorch CAN produce and assert it
    // equals torch's 'xy' result; it does not, pinning the missing-API gap.
    let grids = meshgrid(&[a, b]).expect("meshgrid 'ij' (only available mode)");
    assert_eq!(grids.len(), 2);
    // torch.meshgrid([1,2,3],[4,5], indexing='xy') -> grids of shape [2,3].
    assert_eq!(
        grids[0].shape(),
        &[2, 3],
        "torch 'xy' grids are shape [2,3]; ferrotorch produces 'ij' shape [3,2]"
    );
    // torch 'xy' grid0 = [1,2,3,1,2,3]; grid1 = [4,4,4,5,5,5].
    assert_eq!(data_f32(&grids[0]), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    assert_eq!(data_f32(&grids[1]), vec![4.0, 4.0, 4.0, 5.0, 5.0, 5.0]);
}
