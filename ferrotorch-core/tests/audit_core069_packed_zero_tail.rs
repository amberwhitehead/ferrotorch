//! Red-then-green regression tests for audit finding CORE-069 (crosslink
//! #1763): packed nested tensors mis-handle zero-sized tail dimensions
//! (CLASS-S — `product(tail).max(1)` conflates the empty tail [scalar,
//! factor 1] with a zero-containing tail [zero elements per row]).
//!
//! Observed at HEAD (probe, 2026-06-11):
//! - `from_sequences(vec![vec![1.,2.,3.]], &[3], &[0])` ACCEPTED — three
//!   phantom values stored for the logical shape `[3, 0]` (numel 0);
//!   offsets `[0, 3]`, `to_padded` shape `[1, 3, 0]` with numel 0 while
//!   the packed storage holds 3 values.
//! - The CORRECT empty input (`vec![vec![], vec![]]`, lengths `[3, 2]`,
//!   tail `[0]`) was REJECTED (expected `3 * max(0,1) = 3` elements).
//!
//! torch oracle (live session, torch 2.11.0+cu130):
//!
//! ```python
//! >>> a = torch.zeros(3,0); b = torch.zeros(2,0)
//! >>> ntj = torch.nested.nested_tensor([a,b], layout=torch.jagged)
//! >>> ntj.values().shape, ntj.offsets().tolist()
//! (torch.Size([5, 0]), [0, 3, 5])           # lengths 3 and 2 preserved
//! >>> ntj.to_padded_tensor(0.0).shape
//! torch.Size([2, 3, 0])                      # numel 0
//! >>> c = torch.nested.nested_tensor([torch.tensor([[1.,2.],[3.,4.]]),
//! ...                                 torch.zeros(0,2)], layout=torch.jagged)
//! >>> c.offsets().tolist(), c.to_padded_tensor(-1.0).flatten().tolist()
//! ([0, 2, 2], [1.0, 2.0, 3.0, 4.0, -1.0, -1.0, -1.0, -1.0])
//! ```
//!
//! Post-fix contract: zero-containing tails have zero elements per row
//! (actual product, no `.max(1)`); per-component lengths are carried in
//! the struct (they are NOT derivable from element offsets when
//! `tail_numel == 0` — torch's jagged offsets count ragged-dim rows, so
//! upstream never loses them); round-trips for `[L, 0]`, `[L, 2, 0]`,
//! and zero-length ragged components hold.

use ferrotorch_core::nested::PackedNestedTensor;

/// Non-empty data for a logical `[L, 0]` component shape must be rejected:
/// torch's `[3, 0]` component has numel 0.
#[test]
fn core069_rejects_nonempty_data_for_zero_tail() {
    let r = PackedNestedTensor::from_sequences(vec![vec![1.0f32, 2.0, 3.0]], &[3], &[0]);
    assert!(
        r.is_err(),
        "3 data values for logical shape [3, 0] (numel 0) must be rejected, got {r:?}"
    );
}

/// Round-trip for `[L, 0]` components (lengths 3 and 2, tail `[0]`).
/// torch jagged oracle: values shape (5, 0) — i.e. lengths preserved —
/// and to_padded shape (2, 3, 0), numel 0.
#[test]
fn core069_round_trip_l_0() {
    let p = PackedNestedTensor::<f32>::from_sequences(vec![vec![], vec![]], &[3, 2], &[0])
        .expect("empty data is the CORRECT input for [L, 0] components");
    assert_eq!(p.num_components(), 2);
    assert_eq!(p.length(0), 3, "length 3 must survive a zero tail");
    assert_eq!(p.length(1), 2, "length 2 must survive a zero tail");
    assert_eq!(p.total_numel(), 0);

    // to_padded: torch oracle shape (2, 3, 0), numel 0.
    let padded = p.to_padded(0.0).unwrap();
    assert_eq!(padded.shape(), &[2, 3, 0]);
    assert_eq!(padded.numel(), 0);

    // from_padded round-trip.
    let back = PackedNestedTensor::from_padded(&padded, &[3, 2]).unwrap();
    assert_eq!(back.length(0), 3);
    assert_eq!(back.length(1), 2);
    assert_eq!(back.tail_shape(), &[0]);
    assert_eq!(back.total_numel(), 0);

    // to_nested round-trip: component shapes [3, 0] and [2, 0].
    let nested = p.to_nested().unwrap();
    assert_eq!(nested.tensors()[0].shape(), &[3, 0]);
    assert_eq!(nested.tensors()[1].shape(), &[2, 0]);
}

/// Same for a higher-rank zero-containing tail `[2, 0]` → components
/// `[L, 2, 0]` (numel 0 each).
#[test]
fn core069_round_trip_l_2_0() {
    let p = PackedNestedTensor::<f64>::from_sequences(
        vec![vec![], vec![], vec![]],
        &[1, 4, 0],
        &[2, 0],
    )
    .expect("[L, 2, 0] components have numel 0; empty data is correct");
    assert_eq!(p.length(0), 1);
    assert_eq!(p.length(1), 4);
    assert_eq!(p.length(2), 0);

    let padded = p.to_padded(7.0).unwrap();
    assert_eq!(padded.shape(), &[3, 4, 2, 0]);
    assert_eq!(padded.numel(), 0);

    let back = PackedNestedTensor::from_padded(&padded, &[1, 4, 0]).unwrap();
    assert_eq!(back.length(0), 1);
    assert_eq!(back.length(1), 4);
    assert_eq!(back.length(2), 0);

    let nested = p.to_nested().unwrap();
    assert_eq!(nested.tensors()[1].shape(), &[4, 2, 0]);
}

/// Zero-LENGTH ragged components with a non-zero tail keep working.
/// torch jagged oracle: offsets [0, 2, 2]; padded
/// [1, 2, 3, 4, -1, -1, -1, -1] with shape (2, 2, 2).
#[test]
// reason: to_padded copies values verbatim and writes the literal pad value;
// no float arithmetic happens, so bitwise equality is the right assertion.
#[allow(clippy::float_cmp)]
fn core069_zero_length_ragged_component_round_trip() {
    let p = PackedNestedTensor::from_sequences(
        vec![vec![1.0f32, 2.0, 3.0, 4.0], vec![]],
        &[2, 0],
        &[2],
    )
    .unwrap();
    assert_eq!(p.offsets(), &[0, 4, 4]);
    assert_eq!(p.length(0), 2);
    assert_eq!(p.length(1), 0);

    let padded = p.to_padded(-1.0).unwrap();
    assert_eq!(padded.shape(), &[2, 2, 2]);
    assert_eq!(
        padded.data().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, -1.0, -1.0, -1.0, -1.0],
        "torch jagged oracle to_padded values"
    );

    let back = PackedNestedTensor::from_padded(&padded, &[2, 0]).unwrap();
    assert_eq!(back.data(), p.data());
    assert_eq!(back.offsets(), p.offsets());
    assert_eq!(back.length(1), 0);
}

/// `from_data_tensor` cannot derive per-component lengths from element
/// offsets when the tail contains a zero (every component spans zero
/// elements). The honest contract is a structured error pointing at
/// `from_sequences`, not a silent guess.
#[test]
fn core069_from_data_tensor_zero_tail_errors_structurally() {
    let p = PackedNestedTensor::<f32>::from_sequences(vec![vec![], vec![]], &[3, 2], &[0]).unwrap();
    let flat = p.data_to_tensor().unwrap();
    let r = PackedNestedTensor::<f32>::from_data_tensor(&flat, p.offsets().to_vec(), vec![0]);
    let err = r.expect_err("lengths are not derivable from element offsets with a zero tail");
    let msg = format!("{err}");
    assert!(
        msg.contains("zero") || msg.contains("derivable") || msg.contains("from_sequences"),
        "error should explain why and name the alternative, got: {msg}"
    );
}
