//! Red-then-green regression tests for audit finding CORE-068 (crosslink
//! #1762): `PackedNestedTensor::from_data_tensor` accepts invalid offset
//! layouts (CLASS-S — silently accepted invalid input).
//!
//! The documented layout invariants (`nested.rs` "Layout invariants" doc
//! block, mirroring torch's jagged-layout NJT offsets contract at
//! `torch/nested/_internal/nested_tensor.py` — offsets[0] == 0 and the
//! final offset addressing the full `_values` extent) are:
//!
//! - `offsets[0] == 0`
//! - `offsets.len() == num_components + 1`
//! - `offsets[i+1] - offsets[i] == lengths[i] * tail_numel`
//! - `offsets[num_components] == data.len()`
//!
//! Pre-fix, `from_data_tensor` checked only non-emptiness, monotonicity,
//! and the final entry. Observed at HEAD (probe, 2026-06-11):
//!
//! - offsets `[2, 5]` accepted → the prefix `data[0..2]` is silently
//!   discarded (component 0 silently starts at element 2);
//! - offsets `[0, 3, 6]` with tail `[2]` accepted → `length(0)` truncates
//!   `3 / 2` to `1`, and `to_nested()` pairs a `[1, 2]` shape with a
//!   3-element slice, silently losing a value;
//! - a 2-D `[2, 3]` data tensor accepted where the doc contract says the
//!   input is the flat 1-D tensor produced by `data_to_tensor`.
//!
//! Post-fix contract: every constructor routes through one centralized
//! packed-layout validation; each violation above is a structured
//! `InvalidArgument`/`ShapeMismatch` error.

use ferrotorch_core::nested::PackedNestedTensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn t_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
}

/// offsets[0] != 0 silently discards a data prefix. Must be rejected.
#[test]
fn core068_rejects_nonzero_first_offset() {
    let data = t_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
    let r = PackedNestedTensor::from_data_tensor(&data, vec![2, 5], vec![]);
    let err = r.expect_err("offsets[0] == 2 must be rejected: it silently discards data[0..2]");
    let msg = format!("{err}");
    assert!(
        msg.contains("offsets[0]"),
        "error should name the violated invariant, got: {msg}"
    );
}

/// A component extent not divisible by tail_numel makes `length()` truncate
/// and `to_nested()` silently lose elements. Must be rejected.
#[test]
fn core068_rejects_non_divisible_extent() {
    let data = t_f32((1..=6).map(|x| x as f32).collect(), vec![6]);
    // extent of component 0 is 3, tail_numel is 2 — 3 % 2 != 0.
    let r = PackedNestedTensor::from_data_tensor(&data, vec![0, 3, 6], vec![2]);
    let err = r.expect_err("extent 3 with tail_numel 2 must be rejected (would truncate)");
    let msg = format!("{err}");
    assert!(
        msg.contains("divisible") || msg.contains("tail"),
        "error should name the divisibility violation, got: {msg}"
    );
}

/// The documented input is the flat 1-D tensor from `data_to_tensor`.
/// A 2-D tensor of the right numel must be rejected, not silently
/// reinterpreted.
#[test]
fn core068_rejects_non_1d_data_tensor() {
    let data2d = t_f32(vec![1.0; 6], vec![2, 3]);
    let r = PackedNestedTensor::from_data_tensor(&data2d, vec![0, 6], vec![]);
    let err = r.expect_err("2-D data tensor must be rejected: contract is the flat 1-D buffer");
    let msg = format!("{err}");
    assert!(
        msg.contains("1-D") || msg.contains("ndim"),
        "error should name the rank violation, got: {msg}"
    );
}

/// Decreasing offsets were already rejected pre-fix; the centralized
/// validation must keep rejecting them.
#[test]
fn core068_still_rejects_non_monotonic_offsets() {
    let data = t_f32(vec![1.0; 5], vec![5]);
    let r = PackedNestedTensor::from_data_tensor(&data, vec![0, 4, 3, 5], vec![]);
    assert!(r.is_err(), "non-monotonic offsets must remain rejected");
}

/// Guard: a valid layout still round-trips exactly (no over-rejection).
#[test]
fn core068_valid_layout_round_trips() {
    let orig = PackedNestedTensor::from_sequences(
        vec![vec![1.0f32, 2.0, 3.0, 4.0], vec![5.0, 6.0]],
        &[2, 1],
        &[2],
    )
    .unwrap();
    let flat = orig.data_to_tensor().unwrap();
    let rebuilt =
        PackedNestedTensor::from_data_tensor(&flat, orig.offsets().to_vec(), vec![2]).unwrap();
    assert_eq!(rebuilt.offsets(), orig.offsets());
    assert_eq!(rebuilt.data(), orig.data());
    assert_eq!(rebuilt.tail_shape(), orig.tail_shape());
    assert_eq!(rebuilt.length(0), 2);
    assert_eq!(rebuilt.length(1), 1);
}

/// Guard: empty trailing components (zero-length ragged entries) are a
/// VALID layout (`offsets[i+1] == offsets[i]`) and must stay accepted.
#[test]
fn core068_zero_length_component_layout_still_valid() {
    let data = t_f32(vec![1.0, 2.0], vec![2]);
    let p = PackedNestedTensor::from_data_tensor(&data, vec![0, 2, 2], vec![]).unwrap();
    assert_eq!(p.num_components(), 2);
    assert_eq!(p.length(0), 2);
    assert_eq!(p.length(1), 0);
    assert_eq!(p.component_slice(1), &[] as &[f32]);
}
