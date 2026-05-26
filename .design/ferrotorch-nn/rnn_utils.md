# ferrotorch-nn — `rnn_utils` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/utils/rnn.py
-->

## Summary

`ferrotorch-nn/src/rnn_utils.rs` provides packing utilities for
variable-length RNN inputs: `PackedSequence`, `pack_padded_sequence`,
`pad_packed_sequence`. Mirrors
`torch.nn.utils.rnn.{PackedSequence, pack_padded_sequence,
pad_packed_sequence}` at `torch/nn/utils/rnn.py:39-404`.

These let LSTM/GRU/RNN modules consume batches of variable-length
sequences without wasting computation on padding tokens. The packing
sorts sequences by length (descending), concatenates timesteps in a
batch-aligned layout, and tracks the per-timestep "still-active"
count so the RNN can step its way through the pack without reading
padded zeros.

## Requirements

- REQ-1: `pub struct PackedSequence<T: Float>` with fields
  `data: Tensor<T>` (concatenated `[total_elements, features]`),
  `batch_sizes: Vec<usize>` (per-timestep active count),
  `sorted_indices: Vec<usize>` (original-to-sorted permutation),
  `unsorted_indices: Vec<usize>` (inverse permutation). Mirrors
  upstream's `PackedSequence` NamedTuple at `rnn.py:39-257`.

- REQ-2: `pub fn pack_padded_sequence(input, lengths, batch_first,
  enforce_sorted) -> FerrotorchResult<PackedSequence<T>>` —
  validates the padded input shape, optionally sorts by length
  (descending), then walks `t = 0..max_len` emitting the active
  rows per timestep. Mirrors upstream's
  `pack_padded_sequence(input, lengths, batch_first,
  enforce_sorted)` at `rnn.py:258-326`.

- REQ-3: `pub fn pad_packed_sequence(sequence, batch_first,
  padding_value, total_length) -> FerrotorchResult<(Tensor<T>,
  Vec<usize>)>` — reconstructs a padded tensor from a
  `PackedSequence` by scattering each timestep back to its original
  batch row. Mirrors upstream's `pad_packed_sequence` at
  `rnn.py:327-404`.

- REQ-4: Shape validation — `pack_padded_sequence` rejects non-3-D
  input, length count mismatching batch, lengths outside
  `[1, max_seq_len]`, and (when `enforce_sorted=true`) any length
  array that isn't monotonically non-increasing.

- REQ-5: `batch_first` axis convention — when `batch_first=true`,
  input shape is `[batch, max_seq_len, features]`; when
  `batch_first=false`, it is `[max_seq_len, batch, features]`.
  Matches upstream's kwarg semantics.

- REQ-6: Sort-by-length — when `enforce_sorted=false`,
  `pack_padded_sequence` permutes the batch axis so the longest
  sequence is first. Records the original positions in
  `sorted_indices` and the inverse in `unsorted_indices` so
  `pad_packed_sequence` can restore the original order.

- REQ-7: `batch_sizes` semantics —
  `batch_sizes[t] = #{i | lengths[i] > t}`. The RNN reads
  `data[offset..offset + batch_sizes[t]]` at each timestep.
  Matches upstream's `_VF._pack_padded_sequence` output contract.

- REQ-8: Parity op `pack_padded_sequence` — output `PackedSequence`
  matches upstream's structural contract. NOT-STARTED until the
  parity-sweep runner has a dispatch arm (blocker #1457).

- REQ-9: Parity op `pad_packed_sequence` — round-tripped padded
  output equals the original padded input. NOT-STARTED — blocker
  #1457.

- REQ-10: Parity op `pad_sequence` — NOT-STARTED. The standalone
  `pad_sequence` from `rnn.py:405-472` (which stacks variable-length
  tensors into a single padded batch) is not implemented in
  ferrotorch's `rnn_utils.rs`. Blocker #1457 tracks both the runner
  arm and the implementation.

## Acceptance Criteria

- [x] AC-1: `pack_padded_sequence([B=3, T=5, D=4], lengths=[5, 3,
  2], batch_first=true, enforce_sorted=true)` returns a
  `PackedSequence` with `batch_sizes = [3, 3, 2, 1, 1]`.
- [x] AC-2: `pack_padded_sequence(...)` with non-3-D input errors.
- [x] AC-3: `pack_padded_sequence(...)` with `lengths.len() != B`
  errors.
- [x] AC-4: `pack_padded_sequence(...)` with `lengths[i] = 0` or
  `> max_seq_len` errors.
- [x] AC-5: `pack_padded_sequence(...)` with `enforce_sorted=true`
  and a non-decreasing lengths array errors.
- [x] AC-6: `pad_packed_sequence(packed, batch_first=true,
  padding_value=0.0, total_length=None)` reconstructs the original
  padded tensor (up to the `padding_value` substitution).
- [ ] AC-7: parity-sweep `pack_padded_sequence` at status
  `verified` — blocker #1457.
- [ ] AC-8: parity-sweep `pad_packed_sequence` at status
  `verified` — blocker #1457.
- [ ] AC-9: `pad_sequence` implementation — NOT-STARTED. Blocker
  #1457.

## Architecture

### PackedSequence (REQ-1)

`pub struct PackedSequence<T: Float>` at
`pub struct PackedSequence in rnn_utils.rs` carries the four public
fields with `#[derive(Debug, Clone)]`. The `data` field is a
2-D `[total_elements, features]` tensor; `batch_sizes` is a
host-side `Vec<usize>` (small, frequently inspected); the two index
vectors are stored as `Vec<usize>` for O(1) lookup during unpack.

### pack_padded_sequence (REQ-2, REQ-4, REQ-5, REQ-6, REQ-7)

`pub fn pack_padded_sequence<T: Float>` at
`pub fn pack_padded_sequence in rnn_utils.rs` runs:

1. Validate `input.ndim() == 3` and extract `(batch, max_seq_len,
   features)` per `batch_first`.
2. Validate `lengths.len() == batch` and each `lengths[i]` is in
   `[1, max_seq_len]`.
3. When `enforce_sorted=true`, validate the lengths array is
   monotonically non-increasing.
4. When `enforce_sorted=false`, build `sorted_indices` via a stable
   `sort_by_key` then permute the batch axis.
5. For `t in 0..max_seq_len`, count `batch_sizes[t] =
   #{i | lengths[i] > t}`. Stop when `batch_sizes[t] == 0`.
6. Walk timesteps, copying `batch_sizes[t]` rows from the padded
   input into a flat output buffer of shape
   `[sum(batch_sizes), features]`.
7. Build `unsorted_indices` as the inverse of `sorted_indices`.

### pad_packed_sequence (REQ-3)

`pub fn pad_packed_sequence<T: Float>` at
`pub fn pad_packed_sequence in rnn_utils.rs` performs the inverse:

1. Validate `sequence.data` is 2-D.
2. Allocate a padded buffer of shape
   `[batch, max_len, features]` (or `[max_len, batch, features]`)
   filled with `padding_value`.
3. Walk `t in 0..batch_sizes.len()`, scattering
   `batch_sizes[t]` rows back into the padded buffer at row
   `unsorted_indices[i]`.
4. Return `(padded_tensor, lengths_in_original_order)`.

### Non-test production consumers

- `pub use rnn_utils::{PackedSequence, pack_padded_sequence,
  pad_packed_sequence}` at
  `ferrotorch-nn/src/lib.rs:246` — grandfathered public API
  surface. (Note: `pad_sequence` is intentionally NOT re-exported
  because it's not implemented; see REQ-10.)

## Parity contract

### `pack_padded_sequence`

- Upstream entry: `torch/nn/utils/rnn.py:258 — pack_padded_sequence`
  → `torch._VF._pack_padded_sequence`.
- Edge cases preserved:
  - **Length 0** — rejected. Upstream also errors.
  - **Length > max_seq_len** — rejected.
  - **Unsorted with `enforce_sorted=true`** — rejected.
  - **Single-sample batch** — works; `batch_sizes = [1, 1, ...,
    1]` up to `lengths[0]`.
- Parity-sweep audit status: `MISSING` (blocker #1457).

### `pad_packed_sequence`

- Upstream entry: `torch/nn/utils/rnn.py:327 — pad_packed_sequence`
  → `torch._VF._pad_packed_sequence`.
- Edge cases preserved:
  - **`total_length` shorter than `max_len`** — upstream errors;
    ferrotorch matches.
  - **Custom `padding_value`** — every padded slot receives this
    value (default 0).
- Parity-sweep audit status: `MISSING` (blocker #1457).

### `pad_sequence` (NOT-STARTED)

- Upstream entry: `torch/nn/utils/rnn.py:405 — pad_sequence`.
- Implementation not yet present in `rnn_utils.rs`. Blocker
  #1457 tracks the work item.

## Verification

Tests in `mod tests in rnn_utils.rs`. Highlights:

- `pack_padded_sequence` shape contract.
- `pack_padded_sequence` round-trip through
  `pad_packed_sequence` reproduces the original padded input.
- Validation errors for malformed lengths.

Parity smoke command (blocker #1457 must close):

```bash
for OP in pack_padded_sequence pad_packed_sequence pad_sequence; do
  ./target/release/parity-sweep sweep --op "$OP" --seeds 8 2>&1 \
    | grep -c "passed (0 skipped, 0 failed)"
done
```

Expected (post-#1457): each line returns `>= 1`. Current: each
returns `0`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct PackedSequence<T: Float>` with `data`/`batch_sizes`/`sorted_indices`/`unsorted_indices` in `rnn_utils.rs`; non-test consumer: re-export at `ferrotorch-nn/src/lib.rs:246`. |
| REQ-2 | SHIPPED | impl: `pub fn pack_padded_sequence<T: Float>` in `rnn_utils.rs`; non-test consumer: re-export at `lib.rs:246`. |
| REQ-3 | SHIPPED | impl: `pub fn pad_packed_sequence<T: Float>` in `rnn_utils.rs`; non-test consumer: re-export at `lib.rs:246`. |
| REQ-4 | SHIPPED | impl: validation guards at the head of `pack_padded_sequence` in `rnn_utils.rs`; non-test consumer: re-export at `lib.rs:246`. |
| REQ-5 | SHIPPED | impl: `batch_first` axis-swap logic inside `pack_padded_sequence` in `rnn_utils.rs`; non-test consumer: re-export at `lib.rs:246`. |
| REQ-6 | SHIPPED | impl: stable sort + `sorted_indices` / `unsorted_indices` capture inside `pack_padded_sequence` in `rnn_utils.rs`; non-test consumer: re-export at `lib.rs:246`. |
| REQ-7 | SHIPPED | impl: per-timestep batch-size accumulation inside `pack_padded_sequence` in `rnn_utils.rs`; non-test consumer: re-export at `lib.rs:246`. |
| REQ-8 | NOT-STARTED | parity-sweep runner arm for `pack_padded_sequence` not wired — blocker #1457. |
| REQ-9 | NOT-STARTED | parity-sweep runner arm for `pad_packed_sequence` not wired — blocker #1457. |
| REQ-10 | NOT-STARTED | `pad_sequence` implementation not present in `rnn_utils.rs`; blocker #1457 tracks both the impl and the runner arm. |
