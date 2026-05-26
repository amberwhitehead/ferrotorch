# Pruning utilities

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - torch/nn/utils/prune.py
  - aten/src/ATen/native/cuda/Sparse24.cu
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/pruning.rs` ships three free functions that
correspond to the most common pruning primitives in `torch.nn.utils.prune`:
unstructured magnitude pruning, 2:4 structured sparsity masking, and
sparsity ratio measurement. Each operates on CPU tensors and returns a
new `Tensor<T>` with `requires_grad` propagated.

## Requirements

- REQ-1: `magnitude_prune(weights, sparsity) -> Tensor<T>` — zero out
  the `round(sparsity * numel)` smallest-magnitude elements. The
  threshold is computed by sorting absolute values. NaN values do not
  panic. Mirrors `torch.nn.utils.prune.l1_unstructured` for the
  `1 - sparsity` amount semantics.
- REQ-2: `apply_2_4_mask(weights) -> Tensor<T>` — for every group of
  4 contiguous elements, keep the 2 with the largest magnitude and
  zero the other 2. Trailing elements when `numel % 4 != 0` are left
  unchanged. Mirrors the 2:4 structured-sparsity contract behind
  `torch.sparse.SparseSemiStructuredTensor` and the cuSPARSELt 2:4
  layout in `aten/src/ATen/native/cuda/Sparse24.cu`.
- REQ-3: `sparsity_ratio(tensor) -> f64` — return the fraction of
  exact-zero elements. Used to measure post-pruning sparsity.
- REQ-4: Sparsity argument validation — `magnitude_prune` rejects
  `sparsity ∉ [0, 1)` with `InvalidArgument`. The exact closed-open
  interval matches `prune.py`'s `amount` validation.
- REQ-5: `requires_grad` propagation — the output preserves the
  input's `requires_grad` flag so downstream backward passes can
  still flow through (used by sparse-finetuning workflows).

## Acceptance Criteria

- [x] AC-1: `magnitude_prune([1,-4,2,-3], 0.5)` zeros the two
  smallest-magnitude entries and keeps `[-4, -3]`
  (`pruning.rs:143-154`).
- [x] AC-2: `magnitude_prune([1,2,3,4], 0.0)` returns the input
  unchanged (`pruning.rs:157-162`).
- [x] AC-3: `magnitude_prune(_, 1.0)` and `_, -0.1` return
  `InvalidArgument` (`pruning.rs:165-169`).
- [x] AC-4: NaN inputs do not panic in `magnitude_prune` or
  `apply_2_4_mask` (`pruning.rs:174-190`).
- [x] AC-5: `apply_2_4_mask` keeps exactly 2 of every 4 contiguous
  elements (`pruning.rs:200-218`).
- [x] AC-6: `apply_2_4_mask` propagates `requires_grad`
  (`pruning.rs:221-229`).
- [x] AC-7: `sparsity_ratio` reports 0.5 for `[0, 1, 0, 2]`
  (`pruning.rs:235-239`).
- [x] AC-8: `cargo test -p ferrotorch-core --lib pruning` passes.

## Architecture

The file is ~120 production LOC. All three functions allocate a new
`Vec<T>` for the output, wrap it in `TensorStorage::cpu`, and call
`Tensor::from_storage`. NaN safety in the sort step uses
`partial_cmp(_).unwrap_or(Ordering::Equal)` — the `unwrap_or` is
intentional and matches the NaN-doesn't-panic contract documented in
the test at `pruning.rs:174-179`.

- `magnitude_prune` (`pruning.rs:21-67`) — sort by absolute value, pick
  the threshold at index `n_prune - 1`, set anything `|v| <= threshold`
  to zero. The fast path for `n_prune == 0` returns a clone.
- `apply_2_4_mask` (`pruning.rs:84-112`) — chunked iteration over the
  data slice; per 4-element group, sort `(idx, |v|)` and zero the two
  smallest-magnitude positions.
- `sparsity_ratio` (`pruning.rs:115-122`) — count exact zeros, divide
  by `numel`. Returns an `f64`.

Non-test production consumers: `pruning::apply_2_4_mask` is referenced
by the sparse-tensor compress / decompress integration tests in
`sparse.rs:3023-3027` which compare 2:4-mask output against a
`SemiStructuredSparseTensor` round-trip. Top-level re-export at
`lib.rs:178` lets downstream `ferrotorch-nn` callers reach these.

## Parity contract

`parity_ops = []`. Pruning is a model-compression utility, not a
parity-tracked numeric op. The byte-for-byte parity with PyTorch is
achieved only for the determinism of the magnitude threshold (sort by
absolute value, drop the smallest `k`); PyTorch's
`l1_unstructured` uses the same algorithm but exposes a different API
(it returns a mask + pruning hook chain). The numerical output for
the same input + sparsity is identical.

## Verification

- Unit tests at `pruning.rs:124-240` cover correctness for both
  pruning functions, the NaN-no-panic property, `requires_grad`
  propagation, and the sparsity-ratio computation.
- The sparse-tensor cross-check at `sparse.rs:3011-3027` verifies that
  `apply_2_4_mask` produces the same output as the
  `SemiStructuredSparseTensor` compress-then-decompress round-trip.

```bash
cargo test -p ferrotorch-core --lib pruning
```

Expected: 8 tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `magnitude_prune` at `ferrotorch-core/src/pruning.rs:21` mirrors `torch.nn.utils.prune.l1_unstructured` semantics (sort + threshold + zero); non-test consumer: re-exported at `ferrotorch-core/src/lib.rs:178` (`pub use pruning::{apply_2_4_mask, magnitude_prune, sparsity_ratio};`), reachable by downstream `ferrotorch-nn` callers. |
| REQ-2 | SHIPPED | impl: `apply_2_4_mask` at `ferrotorch-core/src/pruning.rs:84` mirrors the 2:4 structured-sparsity contract behind `aten/src/ATen/native/cuda/Sparse24.cu`; non-test consumer: cross-checked from `ferrotorch-core/src/sparse.rs:3023` (semi-structured sparse decompress matches this mask) AND re-exported at `lib.rs:178`. |
| REQ-3 | SHIPPED | impl: `sparsity_ratio` at `ferrotorch-core/src/pruning.rs:115`; non-test consumer: re-exported at `ferrotorch-core/src/lib.rs:178`; used by downstream `ferrotorch-nn` model-statistics utilities. Per S5 the pub API surface is grandfathered. |
| REQ-4 | SHIPPED | impl: `magnitude_prune`'s validation guard at `ferrotorch-core/src/pruning.rs:25-29` — `if !(0.0..1.0).contains(&sparsity) { return Err(InvalidArgument) }`; non-test consumer: same as REQ-1 (the validation is part of the function contract). |
| REQ-5 | SHIPPED | impl: `Tensor::from_storage(..., weights.requires_grad())` propagation at `ferrotorch-core/src/pruning.rs:63-66` and `:107-111`; non-test consumer: any sparse-finetune workflow that calls `apply_2_4_mask` on a learnable parameter (the `requires_grad` flag survives the prune, allowing backward to flow through the surviving weights). Test pin at `pruning.rs:221-229`. |
