//! Red-then-green regression tests for audit finding CORE-072 (crosslink
//! #1766): public CSR and CSC constructors accept structurally invalid
//! compressed layouts (CLASS-U — malformed pointer/index arrays pass
//! construction, then panic via out-of-bounds indexing inside fallible
//! APIs like `to_dense`, or reach the cuSPARSE backend as malformed
//! descriptors). `CscTensor::to_csr` additionally converted a downstream
//! validation failure into a process panic with `expect`.
//!
//! Observed at HEAD (probe, 2026-06-12, rev 74099dd19):
//! - `CsrTensor::new([1,2,3], [0,1], [1.,2.], 2, 2)` (first ptr != 0):
//!   ACCEPTED.
//! - `CsrTensor::new([0,2,1], ...)` (non-monotonic): ACCEPTED.
//! - `CsrTensor::new([0,1,1], ...)` (final != nnz): ACCEPTED.
//! - `CsrTensor::new([0,1,2], [0,5], ...)` (col 5 in 2 cols): ACCEPTED,
//!   then `to_dense()` panicked: "index out of bounds: the len is 4 but
//!   the index is 7" (sparse.rs:1272).
//! - `CscTensor::new([0,2,5], [0,1], [1.,2.], 2, 2)` (final ptr 5 > nnz
//!   2): ACCEPTED, then `to_dense()` panicked: "index out of bounds: the
//!   len is 2 but the index is 2" (sparse.rs:1883).
//!
//! torch oracle (live session, torch 2.11.0+cu130, the invariant
//! contract `torch.sparse.check_sparse_tensor_invariants` enforces —
//! upstream `aten/src/ATen/native/sparse/SparseCsrTensor.cpp`
//! `_validate_sparse_compressed_tensor_args`):
//!
//! ```python
//! >>> K = dict(check_invariants=True)
//! >>> torch.sparse_csr_tensor(torch.tensor([1,2,3]), torch.tensor([0,1]),
//! ...                         torch.tensor([1.,2.]), (2,2), **K)
//! RuntimeError: `crow_indices[..., 0] == 0` is not satisfied.
//! >>> torch.sparse_csr_tensor(torch.tensor([0,2,1]), ...)  # non-monotonic
//! RuntimeError: `crow_indices[..., -1] == nnz` is not satisfied.
//! >>> torch.sparse_csr_tensor(torch.tensor([0,1,1]), ...)  # final != nnz
//! RuntimeError: `crow_indices[..., -1] == nnz` is not satisfied.
//! >>> torch.sparse_csr_tensor(torch.tensor([0,1,2]), torch.tensor([0,5]), ...)
//! RuntimeError: `0 <= col_indices < ncols` is not satisfied.
//! # CSC duals report `ccol_indices[..., 0] == 0` /
//! # `ccol_indices[..., -1] == nnz` / `0 <= row_indices < nrows`.
//! ```
//!
//! Post-fix contract: both constructors validate (a) zero first pointer,
//! (b) non-decreasing pointers, (c) final pointer == nnz, (d) in-range
//! indices, returning `FerrotorchError::InvalidArgument`; backend-return
//! conversions route through the same constructors; `CscTensor::to_csr`
//! returns `FerrotorchResult` instead of `expect`-panicking.

use ferrotorch_core::{CscTensor, CsrTensor, FerrotorchError};

// ── CSR ────────────────────────────────────────────────────────────────

/// torch: `crow_indices[..., 0] == 0` is not satisfied.
#[test]
fn core072_csr_rejects_nonzero_first_pointer() {
    let r = CsrTensor::new(vec![1, 2, 3], vec![0, 1], vec![1.0f32, 2.0], 2, 2);
    assert!(
        matches!(r, Err(FerrotorchError::InvalidArgument { .. })),
        "first row_ptr != 0 must be rejected, got {r:?}"
    );
}

/// torch: non-monotonic compressed pointers are invalid (reported via
/// the `crow_indices[..., -1] == nnz` / monotonicity checks).
#[test]
fn core072_csr_rejects_non_monotonic_pointers() {
    let r = CsrTensor::new(vec![0, 2, 1], vec![0, 1], vec![1.0f32, 2.0], 2, 2);
    assert!(
        matches!(r, Err(FerrotorchError::InvalidArgument { .. })),
        "non-monotonic row_ptrs must be rejected, got {r:?}"
    );
}

/// torch: `crow_indices[..., -1] == nnz` is not satisfied.
#[test]
fn core072_csr_rejects_final_pointer_not_nnz() {
    let r = CsrTensor::new(vec![0, 1, 1], vec![0, 1], vec![1.0f32, 2.0], 2, 2);
    assert!(
        matches!(r, Err(FerrotorchError::InvalidArgument { .. })),
        "final row_ptr ({}) != nnz (2) must be rejected, got {r:?}",
        1
    );
}

/// torch: `0 <= col_indices < ncols` is not satisfied. Pre-fix this
/// passed construction and panicked inside `to_dense()` ("index out of
/// bounds: the len is 4 but the index is 7").
#[test]
fn core072_csr_rejects_out_of_range_col_index() {
    let r = CsrTensor::new(vec![0, 1, 2], vec![0, 5], vec![1.0f32, 2.0], 2, 2);
    assert!(
        matches!(r, Err(FerrotorchError::InvalidArgument { .. })),
        "col index 5 in a 2-col matrix must be rejected, got {r:?}"
    );
}

// ── CSC ────────────────────────────────────────────────────────────────

/// torch: `ccol_indices[..., 0] == 0` is not satisfied.
#[test]
fn core072_csc_rejects_nonzero_first_pointer() {
    let r = CscTensor::new(vec![1, 2, 3], vec![0, 1], vec![1.0f32, 2.0], 2, 2);
    assert!(
        matches!(r, Err(FerrotorchError::InvalidArgument { .. })),
        "first col_ptr != 0 must be rejected, got {r:?}"
    );
}

/// torch: non-monotonic `ccol_indices` are invalid.
#[test]
fn core072_csc_rejects_non_monotonic_pointers() {
    let r = CscTensor::new(vec![0, 2, 1], vec![0, 1], vec![1.0f32, 2.0], 2, 2);
    assert!(
        matches!(r, Err(FerrotorchError::InvalidArgument { .. })),
        "non-monotonic col_ptrs must be rejected, got {r:?}"
    );
}

/// torch: `ccol_indices[..., -1] == nnz` is not satisfied. Pre-fix the
/// final-ptr-beyond-nnz layout passed construction and `to_dense()`
/// panicked ("index out of bounds: the len is 2 but the index is 2").
#[test]
fn core072_csc_rejects_final_pointer_not_nnz() {
    // final ptr 5 > nnz 2 (the probe's panic reproducer).
    let r = CscTensor::new(vec![0, 2, 5], vec![0, 1], vec![1.0f32, 2.0], 2, 2);
    assert!(
        matches!(r, Err(FerrotorchError::InvalidArgument { .. })),
        "final col_ptr 5 != nnz 2 must be rejected, got {r:?}"
    );
    // final ptr 1 < nnz 2.
    let r = CscTensor::new(vec![0, 1, 1], vec![0, 1], vec![1.0f32, 2.0], 2, 2);
    assert!(
        matches!(r, Err(FerrotorchError::InvalidArgument { .. })),
        "final col_ptr 1 != nnz 2 must be rejected, got {r:?}"
    );
}

/// torch: `0 <= row_indices < nrows` is not satisfied (this one was
/// already rejected at HEAD; pinned so the centralized validator keeps
/// covering it).
#[test]
fn core072_csc_rejects_out_of_range_row_index() {
    let r = CscTensor::new(vec![0, 1, 2], vec![0, 5], vec![1.0f32, 2.0], 2, 2);
    assert!(
        matches!(r, Err(FerrotorchError::InvalidArgument { .. })),
        "row index 5 in a 2-row matrix must be rejected, got {r:?}"
    );
}

// ── to_csr is fallible, not panicking ──────────────────────────────────

/// `CscTensor::to_csr` returns `FerrotorchResult` (the `expect` that
/// turned a downstream validation failure into a process panic is gone)
/// and the valid-layout round-trip still works.
#[test]
fn core072_csc_to_csr_fallible_round_trip() {
    // [[1, 0, 2], [0, 0, 3], [4, 5, 0]]
    let csr = CsrTensor::new(
        vec![0, 2, 3, 5],
        vec![0, 2, 2, 0, 1],
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0],
        3,
        3,
    )
    .expect("valid CSR layout");
    let csc = CscTensor::from_csr(&csr);
    let csr2: CsrTensor<f32> = csc.to_csr().expect("valid CSC -> CSR conversion");
    assert_eq!(csr2.row_ptrs(), csr.row_ptrs());
    assert_eq!(csr2.col_indices(), csr.col_indices());
    assert_eq!(csr2.values(), csr.values());
}

// ── valid layouts must keep passing ────────────────────────────────────

/// Valid layouts (empty rows/cols, empty matrix, full final pointer)
/// still construct: validation must not over-reject.
#[test]
fn core072_valid_layouts_still_accepted() {
    // Empty matrix.
    CsrTensor::<f32>::new(vec![0, 0, 0], vec![], vec![], 2, 3).expect("empty CSR");
    CscTensor::<f32>::new(vec![0, 0, 0, 0], vec![], vec![], 2, 3).expect("empty CSC");
    // Middle row empty.
    CsrTensor::new(vec![0, 1, 1, 2], vec![0, 2], vec![1.0f32, 2.0], 3, 3)
        .expect("CSR with empty middle row");
    // Middle col empty.
    CscTensor::new(vec![0, 1, 1, 2], vec![0, 2], vec![1.0f32, 2.0], 3, 3)
        .expect("CSC with empty middle col");
}
