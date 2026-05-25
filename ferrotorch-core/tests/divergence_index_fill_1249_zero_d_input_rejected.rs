//! Divergence test: ferrotorch `grad_fns::indexing::index_fill` rejects 0-d
//! input even though upstream `torch.index_fill` explicitly supports it.
//!
//! Builder claim in commit c3c1fd57c: "16 skips are for: 0-d input (#1256
//! cross-cutting), multi-d index (upstream TORCH_CHECK), negative index
//! values (ferrotorch IntTensor convention)."
//!
//! The claim that 0-d input is "blocked by cross-cutting #1256" is FALSE.
//! Upstream `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1917` does NOT
//! reject 0-d input — it has an explicit handler:
//!
//!     Tensor self_nonzero_dim =
//!         (self.dim() == 0) ? self.unsqueeze(-1) : self;
//!
//! Then performs the index_fill on the unsqueezed view, which (since the
//! result shares storage with `self`) writes back into the original 0-d
//! tensor. Live torch confirms:
//!
//!     >>> torch.index_fill(torch.tensor(5.0), 0, torch.tensor([0]), -1.0)
//!     tensor(-1.)
//!
//! ferrotorch hard-errors with `InvalidArgument { message: "index_fill:
//! input must have at least 1 dimension" }` at
//! `ferrotorch-core/src/grad_fns/indexing.rs:1484-1486`. This is a forward
//! divergence, NOT a #1256-style cross-cutting deferral — the 0-d path is
//! a per-op responsibility because the upstream handles it INLINE before
//! delegating to the broadcast/strided machinery. Citing #1256 as cover is
//! the deferral pattern goal.md R-DEFER-3 forbids (every divergence is real
//! work to do).
//!
//! Tracking: blocker (filed by acto-critic).

use ferrotorch_core::grad_fns::indexing::index_fill;
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

/// Upstream behavior (confirmed by live torch):
///   torch.index_fill(torch.tensor(5.0), 0, torch.tensor([0]), -1.0)
///   == tensor(-1.)
/// ferrotorch returns an error; this test asserts the upstream-matching
/// behavior and FAILS until the impl is fixed to unsqueeze-and-write-back.
#[test]
fn index_fill_zero_d_input_must_succeed_per_upstream_unsqueeze_handler() {
    let scalar: Tensor<f32> =
        Tensor::from_storage(TensorStorage::cpu(vec![5.0_f32]), vec![], false).unwrap();
    let idx: IntTensor<i64> = IntTensor::from_vec(vec![0_i64], vec![1]).unwrap();
    let result = index_fill(&scalar, 0, &idx, -1.0);
    let out = result.expect(
        "torch.index_fill on a 0-d input is well-defined (upstream unsqueezes \
         to 1-d at TensorAdvancedIndexing.cpp:1917). ferrotorch must mirror.",
    );
    assert_eq!(out.shape(), &[] as &[usize], "0-d output must remain 0-d");
    let data = out.data().expect("data");
    assert_eq!(
        data,
        &[-1.0_f32],
        "torch.index_fill(tensor(5.0), 0, [0], -1.0) returns tensor(-1.), \
         ferrotorch must match per TensorAdvancedIndexing.cpp:1917 \
         self_nonzero_dim = (self.dim() == 0) ? self.unsqueeze(-1) : self"
    );
}
