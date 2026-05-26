//! Divergence-coverage tests for commit `ced1df941` (transcendental S1 batch).
//!
//! The commit ships 21 new transcendental ops in
//! `ferrotorch-core/src/grad_fns/transcendental.rs`. This file pins a
//! divergence in `pub fn sign<T: Float>(input: &Tensor<T>)` at
//! `ferrotorch-core/src/grad_fns/transcendental.rs:1444-1459`:
//!
//! ```text
//! pub fn sign<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
//!     let zero = <T as num_traits::Zero>::zero();
//!     let output = unary_map(input, |x| {
//!         if x.is_nan() {
//!             x                          // <-- propagates NaN
//!         } else if x == zero {
//!             zero
//!         } else {
//!             x.signum()
//!         }
//!     })?;
//!     ...
//! }
//! ```
//!
//! The accompanying doc-comment at line 1442 explicitly claims:
//!     "Special: `sign(NaN) = NaN`, matching `num_traits::Float::signum`'s
//!      NaN propagation (which matches PyTorch's `c10::signum(NaN) = NaN`)."
//!
//! **This claim is FALSE.** PyTorch's `sign_kernel` at
//! `/home/doll/pytorch/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp:304`
//! reads:
//!
//! ```cpp
//!   [=](scalar_t a) -> scalar_t { return (0 < a) - c10::is_negative(a); },
//! ```
//!
//! For floating-point `a = NaN`:
//!   - `(0 < NaN)` is **false** (all comparisons against NaN return false
//!     in IEEE-754), so the first term is `0`.
//!   - `c10::is_negative(NaN)` examines the sign bit. For a positive-payload
//!     NaN the sign bit is 0; for a negative-payload NaN the sign bit is 1.
//!     But torch.tensor([float('nan')]) and torch.tensor([-float('nan')])
//!     are both quieted-NaN with sign bit 0 once they round-trip through
//!     Python/numpy, and `c10::is_negative(NaN)` returns **false**.
//!   - Result: `0 - 0 = 0`.
//!
//! Live PyTorch oracle (R-CHAR-3 — expected value from live torch, NOT
//! literal-copied from ferrotorch):
//!
//! ```python
//! >>> import torch
//! >>> torch.sign(torch.tensor([float('nan')])).item()
//! 0.0
//! >>> torch.sign(torch.tensor([-float('nan')])).item()
//! 0.0
//! ```
//!
//! Ferrotorch returns NaN for both inputs (test asserts the divergence by
//! requiring an upstream-correct `0.0`, expecting the test to FAIL until
//! the impl is fixed).
//!
//! Note: `torch.sgn` (a separate op for complex extension) DOES propagate
//! NaN. The doc-comment at transcendental.rs:1442 conflated `sign` with
//! `sgn`. They are deliberately distinct in PyTorch.
//!
//! Tracking: divergence-blocker filed concurrently with this test.

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

/// Divergence: `ferrotorch::sign(NaN)` returns NaN; PyTorch returns 0.0.
///
/// Upstream cite:
///   `/home/doll/pytorch/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp:304`
///     `[=](scalar_t a) -> scalar_t { return (0 < a) - c10::is_negative(a); },`
///   evaluates to `0` when `a` is a positive-payload NaN.
///
/// Ferrotorch cite:
///   `ferrotorch-core/src/grad_fns/transcendental.rs:1447-1449`
///     `if x.is_nan() { x } ...` — returns NaN unchanged.
#[test]
fn divergence_sign_nan_returns_zero_not_nan() {
    // Expected: live PyTorch returns 0.0 for sign(NaN). Reproducible via:
    //   python3 -c "import torch; print(torch.sign(torch.tensor([float('nan')])).item())"
    //   => 0.0
    let expected_for_positive_nan: f32 = 0.0;
    let expected_for_negative_nan: f32 = 0.0;

    // Construct inputs
    let pos_nan = Tensor::from_storage(TensorStorage::cpu(vec![f32::NAN]), vec![1], false).unwrap();
    let neg_nan =
        Tensor::from_storage(TensorStorage::cpu(vec![-f32::NAN]), vec![1], false).unwrap();

    let out_pos = pos_nan.sign_t().unwrap();
    let out_neg = neg_nan.sign_t().unwrap();

    let val_pos = out_pos.data().unwrap()[0];
    let val_neg = out_neg.data().unwrap()[0];

    assert_eq!(
        val_pos,
        expected_for_positive_nan,
        "sign(+NaN) divergence: ferrotorch returned {val_pos} (NaN={}), \
         but live torch.sign(+NaN) = 0.0. \
         Upstream kernel `(0 < a) - c10::is_negative(a)` at \
         pytorch/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp:304 yields 0 \
         when a is NaN because both `0 < NaN` and `is_negative(NaN with \
         positive sign bit)` are false.",
        val_pos.is_nan()
    );
    assert_eq!(
        val_neg,
        expected_for_negative_nan,
        "sign(-NaN) divergence: ferrotorch returned {val_neg} (NaN={}), \
         but live torch.sign(-NaN) = 0.0 (Python's -float('nan') \
         round-trips to a positive-sign-bit NaN through the tensor \
         constructor).",
        val_neg.is_nan()
    );
}

/// Cross-check: PyTorch's sign on the regular branch still matches
/// ferrotorch — confirms the divergence is ISOLATED to the NaN branch
/// and isn't a sweeping algorithmic disagreement.
///
/// Expected values from live PyTorch oracle:
///   sign(+0.0)  =  0.0
///   sign(-0.0) =  0.0  (NOT preserving the sign bit on zero)
///   sign(+inf) =  1.0
///   sign(-inf) = -1.0
///   sign( 2.5) =  1.0
///   sign(-3.0) = -1.0
#[test]
fn sign_non_nan_branch_matches_torch() {
    let inputs = vec![
        0.0f32,
        -0.0f32,
        f32::INFINITY,
        f32::NEG_INFINITY,
        2.5f32,
        -3.0f32,
    ];
    let expected = vec![0.0f32, 0.0f32, 1.0f32, -1.0f32, 1.0f32, -1.0f32];

    let t = Tensor::from_storage(
        TensorStorage::cpu(inputs.clone()),
        vec![inputs.len()],
        false,
    )
    .unwrap();

    let out = t.sign_t().unwrap();
    let data = out.data().unwrap();

    for (i, (&got, &want)) in data.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got, want,
            "sign({}) = {} but torch expects {}",
            inputs[i], got, want
        );
    }
}
