# Pruning utilities

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - torch/nn/utils/prune.py
  - aten/src/ATen/native/TopKImpl.h
  - aten/src/ATen/native/cuda/Sparse24.cu
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/pruning.rs` ships three free functions that
correspond to the most common pruning primitives in `torch.nn.utils.prune`:
unstructured magnitude pruning, 2:4 structured sparsity masking, and
sparsity ratio measurement. Each operates on CPU tensors. Both pruning
functions are implemented as a differentiable constant-mask
multiplication (`weights * mask` through the real `mul` op), mirroring
PyTorch's pruning parametrization `weight = weight_orig * weight_mask`
(`torch/nn/utils/prune.py`, `BasePruningMethod.apply_mask`): when the
input requires grad, backward delivers `grad * mask` to the ORIGINAL
parameter — exact zeros at pruned slots (CORE-082 -> #1776).

## Requirements

- REQ-1: `magnitude_prune(weights, sparsity) -> Tensor<T>` — zero out
  EXACTLY `round(sparsity * numel)` smallest-magnitude elements, using
  Python's round-half-to-even count rule. Selection (including which
  members of a magnitude tie are pruned)
  reproduces `torch.nn.utils.prune.l1_unstructured`, which scatters
  zeros at `torch.topk(|t|, k, largest=False).indices`
  (`torch/nn/utils/prune.py`, `L1Unstructured.compute_mask`); torch's
  CPU topk selects via `std::partial_sort` when `k * 64 <= numel`, else
  `std::nth_element` at `k - 1`
  (`aten/src/ATen/native/TopKImpl.h:44-88`), so the tie split follows
  libstdc++'s heap-select / introselect order, ported faithfully in
  `torch_cpu_topk_indices` (CORE-083 -> #1777). NaN values do not
  panic (the topk comparator ranks NaN last for `largest=False`).
- REQ-2: `apply_2_4_mask(weights) -> Tensor<T>` — groups of 4 are formed
  along the FINAL dimension of a non-empty 2-D weight matrix only (groups
  never span row boundaries); per group, keep the 2 largest-magnitude
  elements and zero the other 2. Non-2-D inputs, empty dimensions, and
  shapes whose final dimension is not a multiple of 4 return
  `InvalidArgument`, matching the public PyTorch oracle's rejection
  (`torch.ao.pruning.WeightNormSparsifier(sparsity_level=1.0,
  sparse_block_shape=(1,4), zeros_per_block=2)`; live torch
  2.11.0+cu130: `[4]` -> `ValueError: not enough values to unpack`;
  `[2,6]` -> `AssertionError: mask shape (torch.Size([2, 8])) must match
  x shape (torch.Size([2, 6]))`; empty dimensions are rejected by
  `unfold`) — CORE-084 -> #1778. The kept layout and in-block tie
  selection mirror the 2:4 innermost-dim contract behind
  `torch.sparse.SparseSemiStructuredTensor` / cuSPARSELt
  (`aten/src/ATen/native/cuda/Sparse24.cu`).
- REQ-3: `sparsity_ratio(tensor) -> f64` — return the fraction of
  exact-zero elements. Used to measure post-pruning sparsity.
- REQ-4: Sparsity argument validation — `magnitude_prune` rejects
  `sparsity ∉ [0, 1)` with `InvalidArgument`.
- REQ-5: Differentiable masking — both pruning functions return
  `weights * mask` built by `apply_constant_mask` through
  `grad_fns::arithmetic::mul`, attaching a `MulBackward` edge so
  backward reaches the original parameter with `grad * mask` (exact
  zeros at pruned slots), exactly like torch's
  `weight = weight_orig * weight_mask` parametrization (CORE-082 ->
  #1776). The multiplication also reproduces torch's value semantics at
  pruned slots: `(-w) * 0.0 == -0.0` (sign preserved, #1909) and
  `NaN * 0.0 == NaN`.

## Acceptance Criteria

- [x] AC-1: `magnitude_prune([1,-4,2,-3], 0.5)` zeros the two
  smallest-magnitude entries and keeps `[-4, -3]`
  (`test_magnitude_prune_50_percent`, `pruning.rs:370`).
- [x] AC-2: `magnitude_prune([1,2,3,4], 0.0)` returns the input values
  unchanged (`test_magnitude_prune_zero_sparsity`, `pruning.rs:407`).
- [x] AC-3: `magnitude_prune(_, 1.0)` and `_, -0.1` return
  `InvalidArgument` (`test_magnitude_prune_invalid_sparsity`, `pruning.rs:435`).
- [x] AC-4: NaN inputs do not panic in `magnitude_prune` or
  `apply_2_4_mask` (`pruning.rs:444`).
- [x] AC-5: ties at the prune cut are split EXACTLY as live torch
  (`[1,1,1,1] @ 0.25 -> [1,1,0,1]`, `@ 0.5 -> [1,1,0,0]`;
  `test_magnitude_prune_ties_prune_exact_count`, `pruning.rs:386`, plus the conformance tie fixtures).
- [x] AC-6: `apply_2_4_mask` keeps exactly 2 of every 4 elements along
  the final dimension and never groups across rows
  (`pruning.rs:307-417`).
- [x] AC-6b: `apply_2_4_mask` resolves in-block magnitude ties exactly
  as `WeightNormSparsifier` (`[2,2,2,2] -> [2,2,0,0]`,
  `[1,3,3,3] -> [0,3,0,3]`, `[-2,2,-2,2] -> [-2,2,-0,0]`).
- [x] AC-7: `apply_2_4_mask` returns `InvalidArgument` for shapes the
  public torch sparsifier rejects (`[4]`, `[6]`, scalar, rank-3,
  zero-sized `[0,4]` / `[2,0]`, and `[2,6]`;
  `test_apply_2_4_mask_rejects_weight_norm_sparsifier_invalid_shapes`),
  matching the torch sparsifier's rejection.
- [x] AC-8: backward through either pruning function reaches the
  ORIGINAL parameter with `grad * mask` — exact zeros at pruned slots
  (conformance `*_backward_flows_masked_gradient_to_original_leaf`,
  oracle: live torch `weight_orig.grad == [0, 20, 0, 40]`).
- [x] AC-9: `sparsity_ratio` reports 0.5 for `[0, 1, 0, 2]`
  (`test_sparsity_ratio`, `pruning.rs:607`).
- [x] AC-10: `cargo test -p ferrotorch-core --lib pruning` passes.

## Architecture

`magnitude_prune` and `apply_2_4_mask` have both CPU and CUDA-resident
lanes. CPU uses host masks and torch-CPU-compatible top-k selection; CUDA
uses device masks, device top-k/scatter, and returns CUDA tensors without a
host round trip. Quantization remains CPU-domain in
`conformance_quantize_prune.rs` `gpu_tensor_returns_error_*`.

- `torch_cpu_topk_indices` (`pruning.rs:88`) — faithful port of the
  selection order of torch CPU `topk(largest=False)`
  (`aten/src/ATen/native/TopKImpl.h:44-88`): `heap_select`
  (libstdc++ `__heap_select`, the set-defining half of
  `std::partial_sort`) when `k * 64 <= n`, otherwise `nth_element`
  (libstdc++ `__introselect`: median-of-3 quickselect, insertion-sort
  tail, heap-select depth-limit fallback) at `k - 1`. Comparator is
  `TopKImpl.h`'s `(!isnan(x) && isnan(y)) || x < y` — value-only, NaN
  ranks last, never panics. Only SET membership of the first `k` slots
  feeds the mask, so the final orderings (`sort_heap` / the `sorted`
  re-sort) are omitted.
- `magnitude_prune` (`pruning.rs:71`) — validate sparsity, compute
  `n_prune` with `round_ties_even` to mirror Python `round`, build a
  0/1 mask with zeros at the bottom-k indices, then `apply_constant_mask`.
- `apply_constant_mask` / `apply_cuda_mask` (`pruning.rs`) — wrap the
  non-tracking mask on the same device as the input and return
  `grad_fns::arithmetic::mul(weights, mask)`; `MulBackward` provides the
  masked-gradient backward edge.
- `apply_2_4_mask` (`pruning.rs:133`) — validate the public
  WeightNormSparsifier shape contract (`InvalidArgument` unless the input
  is non-empty 2-D and its final dimension is a multiple of 4), build the
  mask row by row in 4-element groups using `torch_cpu_topk_indices` over
  WeightNormSparsifier's default L2 scores (`w * w`) to zero the same two
  in-block entries as `torch.topk(..., k=2, largest=False)`, then
  `apply_constant_mask`.
- `sparsity_ratio` (`pruning.rs:179`) — count exact zeros, divide by
  `numel`. Returns an `f64`. (`-0.0 == 0.0` counts as zero, matching
  torch's `(t == 0)`.)

Non-test production consumers: top-level re-export at `lib.rs:218`
(`pub use pruning::{apply_2_4_mask, magnitude_prune, sparsity_ratio};`)
for downstream `ferrotorch-nn` callers; `apply_2_4_mask` is
cross-checked against the `SemiStructuredSparseTensor`
compress/decompress round-trip in `sparse.rs`.

## Parity contract

`parity_ops = []`. Pruning is a model-compression utility, not a
parity-tracked numeric op. Output parity with PyTorch is nevertheless
bit-exact on the conformance fixtures (torch-oracle-generated,
CORE-194 -> #1888): exact-count selection with torch's CPU topk tie
order, sign-preserving `-0.0` at pruned negative slots, and the
sparsifier's shape rejection and in-block tie selection. No pinned
pruning divergences remain in this segment.

## Verification

- Unit tests at `pruning.rs:207-423` cover exact-count tie behavior
  (live-torch-quoted), final-dim grouping/rejection, NaN-no-panic,
  `requires_grad` propagation, and the sparsity-ratio computation.
- `conformance_quantize_prune.rs` asserts bit-exact parity against the
  torch-oracle fixtures (`magnitude_prune_bit_exact_and_sparsity`,
  `apply_2_4_mask_bit_exact_and_sparsity`), structured rejection of
  torch-rejected shapes, and gradient FLOW to the original leaf
  (`*_backward_flows_masked_gradient_to_original_leaf`, R-ORACLE-3).
- The sparse-tensor cross-check in `sparse.rs`
  (`semi24_compress_then_decompress_matches_apply_2_4_mask`) verifies
  `apply_2_4_mask` against the compress-then-decompress round-trip.

```bash
cargo test -p ferrotorch-core --lib pruning
cargo test -p ferrotorch-core --test conformance_quantize_prune
```

Expected: 13 lib tests + 31 conformance tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `magnitude_prune` at `ferrotorch-core/src/pruning.rs:71` — Python round-half-even count (`round_ties_even`, #1908) plus exact-count bottom-k via `torch_cpu_topk_indices` (`pruning.rs:88`, faithful `TopKImpl.h` selection-order port; CORE-083 -> #1777); non-test consumer: re-exported at `ferrotorch-core/src/lib.rs:218`, reachable by downstream `ferrotorch-nn` callers. |
| REQ-2 | SHIPPED | impl: `apply_2_4_mask` at `ferrotorch-core/src/pruning.rs:133` — public WeightNormSparsifier shape contract (non-empty 2-D, final-dim grouping with structured `InvalidArgument` for final dims not divisible by 4; CORE-084 -> #1778), and in-block zeros selected by torch CPU/CUDA `topk(largest=False)` over WeightNormSparsifier L2 scores (#1910), mirroring the torch sparsifier and the 2:4 innermost-dim layout of `aten/src/ATen/native/cuda/Sparse24.cu`; non-test consumer: cross-checked from `ferrotorch-core/src/sparse.rs` AND re-exported at `lib.rs:218`. |
| REQ-3 | SHIPPED | impl: `sparsity_ratio` at `ferrotorch-core/src/pruning.rs:179`; non-test consumer: re-exported at `ferrotorch-core/src/lib.rs:218`; used by downstream `ferrotorch-nn` model-statistics utilities. Per S5 the pub API surface is grandfathered. |
| REQ-4 | SHIPPED | impl: `magnitude_prune`'s validation guard at `ferrotorch-core/src/pruning.rs:75-79` — `if !(0.0..1.0).contains(&sparsity) { return Err(InvalidArgument) }`; non-test consumer: same as REQ-1 (the validation is part of the function contract). |
| REQ-5 | SHIPPED | impl: `apply_constant_mask` at `ferrotorch-core/src/pruning.rs:102` returns `grad_fns::arithmetic::mul(weights, mask)` — a real `MulBackward` edge, so backward delivers `grad * mask` to the original parameter (CORE-082 -> #1776; replaces the old disconnected-leaf `requires_grad` flag copy); non-test consumer: any sparse-finetune workflow calling either pruning fn on a learnable parameter. Gradient-flow pins: conformance `*_backward_flows_masked_gradient_to_original_leaf`. |
