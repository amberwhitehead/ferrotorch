//! Red regression tests for CORE-153 (#1847) — CLASS-V, Medium.
//!
//! `Tensor::t()` delegates to `grad_fns::shape::transpose_2d`, which
//! errors for ANY rank other than 2 — while `torch.t` documents and
//! implements pass-through for 0-D and 1-D tensors
//! (`aten/src/ATen/native/TensorShape.cpp` `t`:
//! `TORCH_CHECK(self.dim() <= 2, "t() expects a tensor with <= 2
//! dimensions, but self is ", self.dim(), "D");
//! return self.transpose(0, self.dim() < 2 ? 0 : 1);`).
//! `transpose(0, 0)` on a rank-0 tensor likewise errors where PyTorch
//! accepts it (`maybe_wrap_dim` allows dim in `[-1, 0]` for scalars).
//! Rank > 2 `t()` must KEEP erroring, with torch's contract.
//!
//! Oracle (R-ORACLE-1(b)): live torch 2.11.0+cu130, 2026-06-11:
//!
//! ```python
//! torch.tensor(5.).t()                 # tensor(5.)
//! torch.tensor([1., 2., 3.]).t()       # tensor([1., 2., 3.])
//! torch.tensor(5.).transpose(0, 0)     # tensor(5.)
//! torch.randn(2, 2, 2).t()
//! # RuntimeError: t() expects a tensor with <= 2 dimensions, but self is 3D
//! torch.tensor(5.).transpose(0, 1)
//! # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
//! ```

use ferrotorch_core::creation::{from_vec, scalar, tensor};

/// `torch.tensor(5.).t()` -> `tensor(5.)` — 0-D pass-through.
#[test]
// reason: t() on rank < 2 is the identity (pure pass-through, no
// arithmetic), so float equality is the right check.
#[allow(clippy::float_cmp)]
fn t_on_rank0_is_identity() {
    let x = scalar(5.0f32).expect("construct scalar");
    let y = x.t().expect("torch.t accepts 0-D tensors (CORE-153)");
    assert_eq!(y.ndim(), 0, "0-D t() preserves rank");
    assert_eq!(y.item().expect("item"), 5.0);
}

/// `torch.tensor([1., 2., 3.]).t()` -> same 1-D tensor.
#[test]
// reason: t() on rank < 2 is the identity (pure pass-through, no
// arithmetic), so float equality is the right check.
#[allow(clippy::float_cmp)]
fn t_on_rank1_is_identity() {
    let x = tensor(&[1.0f32, 2.0, 3.0]).expect("construct 1-D tensor");
    let y = x.t().expect("torch.t accepts 1-D tensors (CORE-153)");
    assert_eq!(y.shape(), &[3], "1-D t() preserves shape");
    assert_eq!(y.data().expect("data"), &[1.0, 2.0, 3.0]);
}

/// torch returns the SAME tensor (alias) for rank < 2; gradient must
/// keep flowing to the original leaf (R-ORACLE-3: assert flow, not
/// flags).
#[test]
// reason: sum-grad of an identity op is exactly ones — no arithmetic
// rounding anywhere, so float equality is the right check.
#[allow(clippy::float_cmp)]
fn t_on_rank1_propagates_grad_to_leaf() {
    let x = tensor(&[1.0f32, 2.0, 3.0])
        .expect("construct 1-D tensor")
        .requires_grad_(true);
    let y = x.t().expect("1-D t()");
    let loss = y.sum_all().expect("sum");
    ferrotorch_core::backward(&loss).expect("backward");
    let g = x
        .grad()
        .expect("grad access")
        .expect("t() must propagate grad to the original leaf");
    assert_eq!(g.data().expect("grad data"), &[1.0, 1.0, 1.0]);
}

/// Rank-2 behavior is unchanged: still a real transpose.
#[test]
// reason: transpose is pure data movement (no arithmetic), so float
// equality is the right check.
#[allow(clippy::float_cmp)]
fn t_on_rank2_still_transposes() {
    let x = from_vec((0..6).map(|v| v as f32).collect(), &[2, 3]).expect("construct 2-D");
    let y = x.t().expect("2-D t()");
    assert_eq!(y.shape(), &[3, 2]);
    let y = y.contiguous().expect("materialize");
    assert_eq!(y.data().expect("data"), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}

/// Rank > 2 must KEEP erroring, with torch's documented contract:
/// "t() expects a tensor with <= 2 dimensions, but self is 3D".
#[test]
fn t_on_rank3_errors_like_torch() {
    let x = from_vec(vec![0.0f32; 8], &[2, 2, 2]).expect("construct 3-D");
    let err = x.t().expect_err("torch.t rejects rank > 2");
    let msg = err.to_string();
    assert!(
        msg.contains("t() expects a tensor with <= 2 dimensions"),
        "error must carry torch's t() contract, got: {msg}"
    );
}

/// `torch.tensor(5.).transpose(0, 0)` -> `tensor(5.)` — scalars accept
/// dim 0 (`maybe_wrap_dim` range `[-1, 0]`).
#[test]
// reason: transpose(0, 0) is the identity (pure pass-through), so
// float equality is the right check.
#[allow(clippy::float_cmp)]
fn transpose_0_0_on_rank0_is_identity() {
    let x = scalar(5.0f32).expect("construct scalar");
    let y = x
        .transpose(0, 0)
        .expect("torch accepts transpose(0, 0) on scalars (CORE-153)");
    assert_eq!(y.ndim(), 0);
    assert_eq!(y.item().expect("item"), 5.0);
}

/// `torch.tensor(5.).transpose(0, 1)` raises IndexError — dim 1 is out
/// of the scalar's `[-1, 0]` range. Pinned so the rank-0 acceptance
/// does not over-accept.
#[test]
fn transpose_0_1_on_rank0_still_errors() {
    let x = scalar(5.0f32).expect("construct scalar");
    assert!(
        x.transpose(0, 1).is_err(),
        "dim 1 is out of range for a 0-D tensor in torch"
    );
}
