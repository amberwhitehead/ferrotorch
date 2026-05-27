//! Utilities for packing and unpacking variable-length sequences.
//!
//! [`pack_padded_sequence`] converts a padded tensor into a [`PackedSequence`],
//! which RNN modules (LSTM, GRU) can process without wasting computation on
//! padding tokens.  [`pad_packed_sequence`] is the inverse operation.
//!
//! These mirror `torch.nn.utils.rnn.pack_padded_sequence` and
//! `torch.nn.utils.rnn.pad_packed_sequence`.
//!
//! ## REQ status (per `.design/ferrotorch-nn/rnn_utils.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | the `PackedSequence<T>` struct with `data` / `batch_sizes` / `sorted_indices` / `unsorted_indices` here; non-test consumer: re-export at `ferrotorch-nn/src/lib.rs:263` |
//! | REQ-2 | SHIPPED | the `pack_padded_sequence<T>` entry here; non-test consumer: re-export at `lib.rs:263` |
//! | REQ-3 | SHIPPED | the `pad_packed_sequence<T>` entry here; non-test consumer: re-export at `lib.rs:263` |
//! | REQ-4 | SHIPPED | validation guards at the head of `pack_padded_sequence` here; non-test consumer: re-export at `lib.rs:263` |
//! | REQ-5 | SHIPPED | `batch_first` axis-swap logic inside `pack_padded_sequence` here; non-test consumer: re-export at `lib.rs:263` |
//! | REQ-6 | SHIPPED | stable sort + `sorted_indices` / `unsorted_indices` capture inside `pack_padded_sequence` here; non-test consumer: re-export at `lib.rs:263` |
//! | REQ-7 | SHIPPED | per-timestep batch-size accumulation inside `pack_padded_sequence` here; non-test consumer: re-export at `lib.rs:263` |
//! | REQ-8 | SHIPPED | parity op `pack_padded_sequence` wired as a pack/pad round-trip; runner arm `dispatch_pack_unpack_roundtrip` + oracle `_pack_unpack_roundtrip_torch_call`; 48/48 pass — closes #1457 |
//! | REQ-9 | SHIPPED | covered by the same round-trip arm (unpad half calls `pad_packed_sequence`); closes #1457 |
//! | REQ-10 | SHIPPED | `pad_sequence` implemented here; parity op wired via `dispatch_pad_sequence` + oracle `_pad_sequence_torch_call`; 48/48 pass — closes #1457 |

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

// ---------------------------------------------------------------------------
// PackedSequence
// ---------------------------------------------------------------------------

/// A packed representation of variable-length sequences, sorted by length.
///
/// Used with LSTM/GRU to avoid computing on padding tokens.
#[derive(Debug, Clone)]
pub struct PackedSequence<T: Float> {
    /// Concatenated data: all timesteps packed together.
    /// Shape: `[total_elements, features]`
    pub data: Tensor<T>,
    /// Number of sequences present at each timestep.
    /// `batch_sizes[0]` = total sequences (all have at least 1 timestep),
    /// `batch_sizes[t]` = number of sequences with length > t.
    pub batch_sizes: Vec<usize>,
    /// Original indices before sorting (for un-sorting later).
    pub sorted_indices: Vec<usize>,
    /// Inverse of `sorted_indices` (for restoring original order).
    pub unsorted_indices: Vec<usize>,
}

// ---------------------------------------------------------------------------
// pack_padded_sequence
// ---------------------------------------------------------------------------

/// Pack a padded batch of variable-length sequences.
///
/// # Arguments
///
/// * `input` — padded tensor:
///   - `[batch, max_seq_len, features]` if `batch_first` is `true`
///   - `[max_seq_len, batch, features]` if `batch_first` is `false`
/// * `lengths` — actual length of each sequence in the batch.
/// * `batch_first` — whether the batch dimension is first.
/// * `enforce_sorted` — if `true`, the function verifies that `lengths` is
///   sorted in descending order and returns an error otherwise.
///
/// # Returns
///
/// A [`PackedSequence`] with sequences sorted by length (descending).
pub fn pack_padded_sequence<T: Float>(
    input: &Tensor<T>,
    lengths: &[usize],
    batch_first: bool,
    enforce_sorted: bool,
) -> FerrotorchResult<PackedSequence<T>> {
    // --- Validate input shape ---
    if input.ndim() != 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "pack_padded_sequence: expected 3-D input, got {}-D",
                input.ndim()
            ),
        });
    }

    let (batch, max_seq_len, features) = if batch_first {
        (input.shape()[0], input.shape()[1], input.shape()[2])
    } else {
        (input.shape()[1], input.shape()[0], input.shape()[2])
    };

    if lengths.len() != batch {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "pack_padded_sequence: lengths.len() ({}) != batch size ({})",
                lengths.len(),
                batch,
            ),
        });
    }

    if batch == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "pack_padded_sequence: batch size must be >= 1".into(),
        });
    }

    // Validate lengths: all must be in [1, max_seq_len].
    for (i, &len) in lengths.iter().enumerate() {
        if len == 0 || len > max_seq_len {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "pack_padded_sequence: lengths[{i}] = {len} is invalid \
                     (must be in [1, {max_seq_len}])",
                ),
            });
        }
    }

    // --- Sort by length (descending) ---
    let mut sorted_indices: Vec<usize> = (0..batch).collect();
    sorted_indices.sort_by(|&a, &b| lengths[b].cmp(&lengths[a]));

    let sorted_lengths: Vec<usize> = sorted_indices.iter().map(|&i| lengths[i]).collect();

    // Check descending order if enforce_sorted.
    if enforce_sorted {
        for w in lengths.windows(2) {
            if w[0] < w[1] {
                return Err(FerrotorchError::InvalidArgument {
                    message: "pack_padded_sequence: lengths must be sorted in \
                              descending order when enforce_sorted=true"
                        .into(),
                });
            }
        }
    }

    // Build unsorted_indices: the inverse permutation.
    let mut unsorted_indices = vec![0usize; batch];
    for (new_pos, &orig_idx) in sorted_indices.iter().enumerate() {
        unsorted_indices[orig_idx] = new_pos;
    }

    // --- Compute batch_sizes ---
    let max_len = sorted_lengths[0]; // longest after sort
    let mut batch_sizes: Vec<usize> = Vec::with_capacity(max_len);
    for t in 0..max_len {
        let count = sorted_lengths.iter().filter(|&&l| l > t).count();
        batch_sizes.push(count);
    }

    // --- Pack the data ---
    let total_elements: usize = batch_sizes.iter().sum();
    let input_data = input.data()?;
    let mut packed_data: Vec<T> = Vec::with_capacity(total_elements * features);

    for (t, &bs) in batch_sizes.iter().enumerate() {
        for &orig_batch_idx in &sorted_indices[..bs] {
            // Compute offset into the flat input data.
            let offset = if batch_first {
                // input[orig_batch_idx, t, :] — layout [batch, max_seq_len, features]
                orig_batch_idx * max_seq_len * features + t * features
            } else {
                // input[t, orig_batch_idx, :] — layout [max_seq_len, batch, features]
                t * batch * features + orig_batch_idx * features
            };

            packed_data.extend_from_slice(&input_data[offset..offset + features]);
        }
    }

    let data = Tensor::from_storage(
        TensorStorage::cpu(packed_data),
        vec![total_elements, features],
        input.requires_grad(),
    )?;

    Ok(PackedSequence {
        data,
        batch_sizes,
        sorted_indices,
        unsorted_indices,
    })
}

// ---------------------------------------------------------------------------
// pad_packed_sequence
// ---------------------------------------------------------------------------

/// Inverse of [`pack_padded_sequence`]. Pads packed sequences back to a tensor.
///
/// # Arguments
///
/// * `packed` — a [`PackedSequence`] produced by [`pack_padded_sequence`].
/// * `batch_first` — if `true`, the output shape is
///   `[batch, max_seq_len, features]`; otherwise `[max_seq_len, batch, features]`.
/// * `padding_value` — the value used to fill padding positions.
///
/// # Returns
///
/// A tuple `(padded_tensor, lengths)` where `lengths` is in the *original*
/// (unsorted) order matching the input batch indices.
pub fn pad_packed_sequence<T: Float>(
    packed: &PackedSequence<T>,
    batch_first: bool,
    padding_value: T,
) -> FerrotorchResult<(Tensor<T>, Vec<usize>)> {
    let batch = packed.batch_sizes.first().copied().unwrap_or(0);
    if batch == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "pad_packed_sequence: empty PackedSequence".into(),
        });
    }

    let max_seq_len = packed.batch_sizes.len();
    let packed_data = packed.data.data()?;
    let total_elements = packed.data.shape()[0];
    let features = if packed.data.ndim() == 2 {
        packed.data.shape()[1]
    } else {
        return Err(FerrotorchError::InvalidArgument {
            message: "pad_packed_sequence: packed data must be 2-D [total, features]".into(),
        });
    };

    // Verify total_elements consistency.
    let expected_total: usize = packed.batch_sizes.iter().sum();
    if total_elements != expected_total {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "pad_packed_sequence: total elements {} != sum of batch_sizes {}",
                total_elements, expected_total,
            ),
        });
    }

    // Reconstruct sorted lengths from batch_sizes.
    // sorted_lengths[s] = number of timesteps t where batch_sizes[t] > s.
    let mut sorted_lengths = vec![0usize; batch];
    for &bs in &packed.batch_sizes {
        for sl in sorted_lengths[..bs].iter_mut() {
            *sl += 1;
        }
    }

    // Initialize padded output with padding_value.
    let numel = batch * max_seq_len * features;
    let mut output_data = vec![padding_value; numel];

    // Unpack: walk through packed data in the same order pack_padded_sequence
    // produced it.
    let mut data_offset = 0;
    for t in 0..max_seq_len {
        let bs = packed.batch_sizes[t];
        for s in 0..bs {
            // s is the index in sorted order.
            // Map back to original batch index.
            let orig_batch_idx = packed.sorted_indices[s];

            let out_offset = if batch_first {
                orig_batch_idx * max_seq_len * features + t * features
            } else {
                t * batch * features + orig_batch_idx * features
            };

            output_data[out_offset..out_offset + features]
                .copy_from_slice(&packed_data[data_offset..data_offset + features]);
            data_offset += features;
        }
    }

    let out_shape = if batch_first {
        vec![batch, max_seq_len, features]
    } else {
        vec![max_seq_len, batch, features]
    };

    let tensor = Tensor::from_storage(
        TensorStorage::cpu(output_data),
        out_shape,
        packed.data.requires_grad(),
    )?;

    // Return lengths in *original* (unsorted) order.
    let mut original_lengths = vec![0usize; batch];
    for (sorted_pos, &orig_idx) in packed.sorted_indices.iter().enumerate() {
        original_lengths[orig_idx] = sorted_lengths[sorted_pos];
    }

    Ok((tensor, original_lengths))
}

// ---------------------------------------------------------------------------
// pad_sequence
// ---------------------------------------------------------------------------

/// Pad a list of variable-length sequences into a single batched tensor.
///
/// Mirrors `torch.nn.utils.rnn.pad_sequence` at
/// `torch/nn/utils/rnn.py:405-470`. Each input sequence has shape
/// `[L_i, *trailing]` where `L_i` is the (variable) sequence length and
/// `*trailing` is a fixed set of feature dimensions shared by every sequence.
/// The sequences are stacked along a new batch dimension and right-padded
/// with `padding_value` so every row has length `T = max_i L_i`.
///
/// # Arguments
///
/// * `sequences` — slice of `[L_i, *trailing]` tensors. Must be non-empty,
///   and every sequence must share the same trailing dimensions.
/// * `batch_first` — if `true`, the output is `[B, T, *trailing]`; otherwise
///   it is `[T, B, *trailing]` (upstream's default).
/// * `padding_value` — value written into the padded tail of each sequence.
///
/// # Returns
///
/// A single padded tensor (`[B, T, *trailing]` or `[T, B, *trailing]`).
pub fn pad_sequence<T: Float>(
    sequences: &[Tensor<T>],
    batch_first: bool,
    padding_value: T,
) -> FerrotorchResult<Tensor<T>> {
    if sequences.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "pad_sequence: sequences must be non-empty".into(),
        });
    }

    // Every sequence must have ndim >= 1 (a length axis plus optional features)
    // and identical trailing dimensions. Upstream assumes "trailing dimensions
    // and type of all the Tensors in sequences are same" (rnn.py:429-432).
    let first = &sequences[0];
    if first.ndim() < 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: "pad_sequence: each sequence must have ndim >= 1".into(),
        });
    }
    let trailing: Vec<usize> = first.shape()[1..].to_vec();
    let feature_numel: usize = trailing.iter().product();

    for (i, seq) in sequences.iter().enumerate() {
        if seq.ndim() != first.ndim() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "pad_sequence: sequence {i} has ndim {} but sequence 0 has ndim {}",
                    seq.ndim(),
                    first.ndim()
                ),
            });
        }
        if seq.shape()[1..] != trailing[..] {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "pad_sequence: sequence {i} trailing dims {:?} != {:?}",
                    &seq.shape()[1..],
                    trailing
                ),
            });
        }
    }

    let batch = sequences.len();
    let max_len = sequences.iter().map(|s| s.shape()[0]).max().unwrap_or(0);

    // Allocate the padded buffer filled with padding_value.
    let numel = batch * max_len * feature_numel;
    let mut out = vec![padding_value; numel];

    // Whether grad must flow: if any input requires grad, the output does too.
    let requires_grad = sequences.iter().any(|s| s.requires_grad());

    // Scatter each sequence's rows into the batched buffer. The flat layout is
    // [B, T, feature_numel] for batch_first, else [T, B, feature_numel].
    for (b, seq) in sequences.iter().enumerate() {
        let len = seq.shape()[0];
        let src = seq.data()?;
        for t in 0..len {
            let src_off = t * feature_numel;
            let dst_off = if batch_first {
                (b * max_len + t) * feature_numel
            } else {
                (t * batch + b) * feature_numel
            };
            out[dst_off..dst_off + feature_numel]
                .copy_from_slice(&src[src_off..src_off + feature_numel]);
        }
    }

    let mut out_shape = if batch_first {
        vec![batch, max_len]
    } else {
        vec![max_len, batch]
    };
    out_shape.extend_from_slice(&trailing);

    Tensor::from_storage(TensorStorage::cpu(out), out_shape, requires_grad)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: create a batch-first tensor with known values.
    //
    //   input[b, t, f] = (b * 100 + t * 10 + f) as f32
    //
    // This makes it easy to verify which elements ended up where.
    // -----------------------------------------------------------------------
    fn make_test_input(batch: usize, max_seq_len: usize, features: usize) -> Tensor<f32> {
        let mut data = Vec::with_capacity(batch * max_seq_len * features);
        for b in 0..batch {
            for t in 0..max_seq_len {
                for f in 0..features {
                    data.push((b * 100 + t * 10 + f) as f32);
                }
            }
        }
        Tensor::from_storage(
            TensorStorage::cpu(data),
            vec![batch, max_seq_len, features],
            false,
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // batch_sizes for [5, 3, 2]
    // -----------------------------------------------------------------------
    #[test]
    fn test_batch_sizes_5_3_2() {
        let input = make_test_input(3, 5, 4);
        let packed = pack_padded_sequence(&input, &[5, 3, 2], true, true).unwrap();

        assert_eq!(packed.batch_sizes, vec![3, 3, 2, 1, 1]);
    }

    // -----------------------------------------------------------------------
    // Round-trip: pack then unpack preserves data
    // -----------------------------------------------------------------------
    #[test]
    fn test_round_trip_batch_first() {
        let batch = 3;
        let max_seq = 5;
        let feat = 4;
        let lengths = [5, 3, 2];

        let input = make_test_input(batch, max_seq, feat);
        let input_data = input.data().unwrap().to_vec();

        let packed = pack_padded_sequence(&input, &lengths, true, true).unwrap();
        let (output, out_lengths) = pad_packed_sequence(&packed, true, 0.0f32).unwrap();

        assert_eq!(out_lengths, &[5, 3, 2]);
        assert_eq!(output.shape(), &[batch, max_seq, feat]);

        let output_data = output.data().unwrap();

        // For each sequence, non-padding positions must match; padding must be 0.
        for b in 0..batch {
            for t in 0..max_seq {
                for f in 0..feat {
                    let idx = b * max_seq * feat + t * feat + f;
                    if t < lengths[b] {
                        assert_eq!(
                            output_data[idx], input_data[idx],
                            "mismatch at b={b} t={t} f={f}"
                        );
                    } else {
                        assert_eq!(
                            output_data[idx], 0.0,
                            "expected padding=0.0 at b={b} t={t} f={f}"
                        );
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // batch_first=false
    // -----------------------------------------------------------------------
    #[test]
    fn test_round_trip_seq_first() {
        let batch = 3;
        let max_seq = 4;
        let feat = 2;
        let lengths = [4, 2, 1];

        // Build a seq-first tensor: [max_seq, batch, features].
        let mut data = Vec::with_capacity(max_seq * batch * feat);
        for t in 0..max_seq {
            for b in 0..batch {
                for f in 0..feat {
                    data.push((t * 100 + b * 10 + f) as f32);
                }
            }
        }
        let input = Tensor::from_storage(
            TensorStorage::cpu(data.clone()),
            vec![max_seq, batch, feat],
            false,
        )
        .unwrap();

        let packed = pack_padded_sequence(&input, &lengths, false, true).unwrap();
        let (output, out_lengths) = pad_packed_sequence(&packed, false, -1.0f32).unwrap();

        assert_eq!(out_lengths, &lengths);
        assert_eq!(output.shape(), &[max_seq, batch, feat]);

        let output_data = output.data().unwrap();

        for t in 0..max_seq {
            for b in 0..batch {
                for f in 0..feat {
                    let idx = t * batch * feat + b * feat + f;
                    if t < lengths[b] {
                        assert_eq!(output_data[idx], data[idx], "mismatch at t={t} b={b} f={f}");
                    } else {
                        assert_eq!(
                            output_data[idx], -1.0,
                            "expected padding=-1.0 at t={t} b={b} f={f}"
                        );
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // enforce_sorted=true rejects unsorted input
    // -----------------------------------------------------------------------
    #[test]
    fn test_enforce_sorted_rejects_unsorted() {
        let input = make_test_input(3, 5, 2);
        let result = pack_padded_sequence(&input, &[2, 5, 3], true, true);
        assert!(result.is_err(), "should reject unsorted lengths");

        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("descending"),
            "error should mention descending: {err_msg}"
        );
    }

    // -----------------------------------------------------------------------
    // enforce_sorted=false accepts unsorted input
    // -----------------------------------------------------------------------
    #[test]
    fn test_enforce_sorted_false_accepts_unsorted() {
        let input = make_test_input(3, 5, 2);
        let packed = pack_padded_sequence(&input, &[2, 5, 3], true, false).unwrap();

        // After sorting: [5, 3, 2] => batch_sizes = [3, 3, 2, 1, 1]
        assert_eq!(packed.batch_sizes, vec![3, 3, 2, 1, 1]);

        // sorted_indices should map sorted position -> original index.
        // Longest (5) was at index 1, next (3) at index 2, shortest (2) at index 0.
        assert_eq!(packed.sorted_indices, vec![1, 2, 0]);
    }

    // -----------------------------------------------------------------------
    // Single sequence
    // -----------------------------------------------------------------------
    #[test]
    fn test_single_sequence() {
        let input = make_test_input(1, 3, 2);
        let packed = pack_padded_sequence(&input, &[3], true, true).unwrap();

        assert_eq!(packed.batch_sizes, vec![1, 1, 1]);
        assert_eq!(packed.sorted_indices, vec![0]);
        assert_eq!(packed.unsorted_indices, vec![0]);

        // Packed data should equal the original (no padding to remove).
        let packed_flat = packed.data.data().unwrap();
        let input_flat = input.data().unwrap();
        assert_eq!(packed_flat, input_flat);

        // Round-trip.
        let (output, lens) = pad_packed_sequence(&packed, true, 0.0f32).unwrap();
        assert_eq!(lens, vec![3]);
        assert_eq!(output.data().unwrap(), input.data().unwrap());
    }

    // -----------------------------------------------------------------------
    // All same length (no actual packing needed)
    // -----------------------------------------------------------------------
    #[test]
    fn test_all_same_length() {
        let batch = 4;
        let seq_len = 3;
        let feat = 2;
        let lengths = [3, 3, 3, 3];

        let input = make_test_input(batch, seq_len, feat);

        let packed = pack_padded_sequence(&input, &lengths, true, true).unwrap();

        // All timesteps have the full batch.
        assert_eq!(packed.batch_sizes, vec![4, 4, 4]);

        // Round-trip should be exact.
        let (output, out_lengths) = pad_packed_sequence(&packed, true, 0.0f32).unwrap();
        assert_eq!(out_lengths, &[3, 3, 3, 3]);
        assert_eq!(output.data().unwrap(), input.data().unwrap());
    }

    // -----------------------------------------------------------------------
    // Packed data order is correct
    // -----------------------------------------------------------------------
    #[test]
    fn test_packed_data_order() {
        // 2 sequences, lengths [3, 2], features=1, batch_first=true
        // input[0] = [10, 20, 30], input[1] = [40, 50, PAD]
        let data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 0.0];
        let input = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 3, 1], false).unwrap();

        let packed = pack_padded_sequence(&input, &[3, 2], true, true).unwrap();

        assert_eq!(packed.batch_sizes, vec![2, 2, 1]);

        // Expected packed order (timestep-major, sorted batch):
        // t=0: seq0=10, seq1=40
        // t=1: seq0=20, seq1=50
        // t=2: seq0=30
        let packed_flat = packed.data.data().unwrap();
        assert_eq!(packed_flat, &[10.0, 40.0, 20.0, 50.0, 30.0]);
    }

    // -----------------------------------------------------------------------
    // Error: lengths mismatch batch size
    // -----------------------------------------------------------------------
    #[test]
    fn test_error_lengths_mismatch_batch() {
        let input = make_test_input(3, 5, 2);
        let result = pack_padded_sequence(&input, &[5, 3], true, false);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Error: length = 0 is invalid
    // -----------------------------------------------------------------------
    #[test]
    fn test_error_zero_length() {
        let input = make_test_input(2, 3, 2);
        let result = pack_padded_sequence(&input, &[3, 0], true, false);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Error: length exceeds max_seq_len
    // -----------------------------------------------------------------------
    #[test]
    fn test_error_length_exceeds_max() {
        let input = make_test_input(2, 3, 2);
        let result = pack_padded_sequence(&input, &[3, 4], true, false);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Error: non-3D input
    // -----------------------------------------------------------------------
    #[test]
    fn test_error_non_3d_input() {
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
            vec![2, 2],
            false,
        )
        .unwrap();
        let result = pack_padded_sequence(&input, &[2, 1], true, false);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Unsorted input: round-trip with enforce_sorted=false
    // -----------------------------------------------------------------------
    #[test]
    fn test_unsorted_round_trip() {
        let batch = 3;
        let max_seq = 5;
        let feat = 2;
        // Deliberately unsorted lengths.
        let lengths = [3, 5, 2];

        let input = make_test_input(batch, max_seq, feat);
        let input_data = input.data().unwrap().to_vec();

        let packed = pack_padded_sequence(&input, &lengths, true, false).unwrap();
        let (output, out_lengths) = pad_packed_sequence(&packed, true, 0.0f32).unwrap();

        // Lengths should come back in original order.
        assert_eq!(out_lengths, &[3, 5, 2]);

        let output_data = output.data().unwrap();
        for b in 0..batch {
            for t in 0..max_seq {
                for f in 0..feat {
                    let idx = b * max_seq * feat + t * feat + f;
                    if t < lengths[b] {
                        assert_eq!(
                            output_data[idx], input_data[idx],
                            "mismatch at b={b} t={t} f={f}"
                        );
                    } else {
                        assert_eq!(
                            output_data[idx], 0.0,
                            "expected padding at b={b} t={t} f={f}"
                        );
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // pad_sequence: batch_first stacks + right-pads variable-length seqs.
    //
    // Expected values traceable to torch.nn.utils.rnn.pad_sequence
    // (torch/nn/utils/rnn.py:405). Live-verified 2026-05-26:
    //   a = arange(3).reshape(3,1); b = arange(2).reshape(2,1);
    //   c = arange(1).reshape(1,1)
    //   pad_sequence([a,b,c], batch_first=True, padding_value=-9.0).squeeze(-1)
    //     == [[0,1,2],[0,1,-9],[0,-9,-9]]
    // -----------------------------------------------------------------------
    #[test]
    fn test_pad_sequence_batch_first() {
        let a = Tensor::from_storage(
            TensorStorage::cpu(vec![0.0f32, 1.0, 2.0]),
            vec![3, 1],
            false,
        )
        .unwrap();
        let b =
            Tensor::from_storage(TensorStorage::cpu(vec![0.0f32, 1.0]), vec![2, 1], false).unwrap();
        let c = Tensor::from_storage(TensorStorage::cpu(vec![0.0f32]), vec![1, 1], false).unwrap();

        let out = pad_sequence(&[a, b, c], true, -9.0f32).unwrap();
        assert_eq!(out.shape(), &[3, 3, 1]);
        // Flat [B=3, T=3, F=1] row-major.
        assert_eq!(
            out.data().unwrap(),
            &[0.0, 1.0, 2.0, 0.0, 1.0, -9.0, 0.0, -9.0, -9.0]
        );
    }

    // -----------------------------------------------------------------------
    // pad_sequence: seq_first ([T, B, *]) is the upstream default layout.
    //
    // Live-verified 2026-05-26: pad_sequence([a,b,c], batch_first=False,
    // padding_value=0.0) has shape [3, 3, 1].
    // -----------------------------------------------------------------------
    #[test]
    fn test_pad_sequence_seq_first() {
        let a = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]),
            vec![3, 1],
            false,
        )
        .unwrap();
        let b =
            Tensor::from_storage(TensorStorage::cpu(vec![4.0f32, 5.0]), vec![2, 1], false).unwrap();

        let out = pad_sequence(&[a, b], false, 0.0f32).unwrap();
        assert_eq!(out.shape(), &[3, 2, 1]);
        // Flat [T=3, B=2, F=1]:
        //   t=0: a0=1, b0=4
        //   t=1: a1=2, b1=5
        //   t=2: a2=3, b_pad=0
        assert_eq!(out.data().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 0.0]);
    }

    // -----------------------------------------------------------------------
    // pad_sequence: multi-feature trailing dims preserved.
    // -----------------------------------------------------------------------
    #[test]
    fn test_pad_sequence_multifeature() {
        // a: [2, 2], b: [1, 2]
        let a = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
            vec![2, 2],
            false,
        )
        .unwrap();
        let b =
            Tensor::from_storage(TensorStorage::cpu(vec![5.0f32, 6.0]), vec![1, 2], false).unwrap();

        let out = pad_sequence(&[a, b], true, 0.0f32).unwrap();
        assert_eq!(out.shape(), &[2, 2, 2]);
        // [B=2, T=2, F=2]:
        //   b0: [[1,2],[3,4]]
        //   b1: [[5,6],[pad,pad]]
        assert_eq!(
            out.data().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0]
        );
    }

    // -----------------------------------------------------------------------
    // pad_sequence: equal-length sequences need no padding.
    // -----------------------------------------------------------------------
    #[test]
    fn test_pad_sequence_equal_lengths() {
        let a =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0]), vec![2, 1], false).unwrap();
        let b =
            Tensor::from_storage(TensorStorage::cpu(vec![3.0f32, 4.0]), vec![2, 1], false).unwrap();

        let out = pad_sequence(&[a, b], true, -1.0f32).unwrap();
        assert_eq!(out.shape(), &[2, 2, 1]);
        assert_eq!(out.data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    // -----------------------------------------------------------------------
    // pad_sequence error: empty list.
    // -----------------------------------------------------------------------
    #[test]
    fn test_pad_sequence_error_empty() {
        let empty: &[Tensor<f32>] = &[];
        assert!(pad_sequence(empty, true, 0.0f32).is_err());
    }

    // -----------------------------------------------------------------------
    // pad_sequence error: mismatched trailing dims.
    // -----------------------------------------------------------------------
    #[test]
    fn test_pad_sequence_error_trailing_mismatch() {
        let a = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
            vec![2, 2],
            false,
        )
        .unwrap();
        let b =
            Tensor::from_storage(TensorStorage::cpu(vec![5.0f32, 6.0]), vec![2, 1], false).unwrap();
        assert!(pad_sequence(&[a, b], true, 0.0f32).is_err());
    }

    // -----------------------------------------------------------------------
    // f64 type works
    // -----------------------------------------------------------------------
    #[test]
    fn test_f64_pack_unpack() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 3, 1], false).unwrap();

        let packed = pack_padded_sequence(&input, &[3, 2], true, true).unwrap();
        let (output, lens) = pad_packed_sequence(&packed, true, 0.0f64).unwrap();

        assert_eq!(lens, &[3, 2]);
        let out = output.data().unwrap();
        // seq 0: [1, 2, 3]
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 3.0);
        // seq 1: [4, 5, PAD=0]
        assert_eq!(out[3], 4.0);
        assert_eq!(out[4], 5.0);
        assert_eq!(out[5], 0.0);
    }
}
