# Search & Sort Tensor Operations

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/
  - c10/
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/ops/search.rs` ships the searching / sorting /
discretization primitives: `searchsorted`, `bucketize`, `unique`,
`unique_consecutive`, `histc`, `meshgrid`, and `topk`. Each mirrors
the same-named `torch.*` function. All paths are CPU-only — CUDA
inputs error with `NotImplementedOnCuda` (GPU lowerings tracked
under #1545).

## Requirements

- REQ-1: `searchsorted(boundaries, values, right)` — binary search a
  sorted 1-D `boundaries` tensor for each element of `values`.
  Returns `Vec<usize>`. `right=true` uses upper-bound (>), `right=false`
  uses lower-bound (>=). Mirrors `torch.searchsorted`.
- REQ-2: `bucketize(input, boundaries, right)` — alias for
  `searchsorted(boundaries, input, right)`. Mirrors `torch.bucketize`.
- REQ-3: `unique(input)` — 1-D tensor → `(unique_values, inverse_indices,
  counts)` where `unique_values` is sorted, `inverse[i]` is the
  index of `input[i]` in `unique_values`, `counts[k]` is the
  frequency. Mirrors `torch.unique(sorted=True, return_inverse=True,
  return_counts=True)`.
- REQ-4: `unique_consecutive(input)` — same return tuple but
  collapses only adjacent-equal runs. Mirrors
  `torch.unique_consecutive`.
- REQ-5: `histc(input, bins, min, max)` — `bins`-bucket histogram of
  the flattened input. Out-of-range elements clamp to boundary bins.
  Mirrors `torch.histc`.
- REQ-6: `meshgrid(tensors)` — N 1-D inputs → N N-D coordinate
  tensors of shape `[len0, len1, ..., lenN-1]`. `indexing='ij'`
  convention. Mirrors `torch.meshgrid`.
- REQ-7: `topk(input, k, largest)` — return `(values, indices)` for
  the k largest (or smallest) along the last dim. Both shapes have
  last-dim replaced by `k`. Mirrors `torch.topk`. CUDA f32/f64 inputs
  lower on-device via `GpuBackend::topk_1d` (#1545): the k-selection runs
  on the GPU, the VALUES tensor stays GPU-resident, and only the int64
  indices are read back to host. Ties resolve to ascending original index
  (a valid `torch.topk` result; matches the CPU stable-sort path).

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib ops::search`
  passes (12 tests covering each operation).
- [x] AC-2: `searchsorted([1,3,5,7], [0,2,3,6,8], right=true)` →
  `[0,1,2,3,4]`.
- [x] AC-3: `searchsorted` with empty boundaries returns all zeros
  (test_searchsorted_empty_bounds).
- [x] AC-4: `unique([3,1,2,1,3,2])` → `([1,2,3], [2,0,1,0,2,1], [2,2,2])`.
- [x] AC-5: `unique_consecutive([1,1,2,2,2,3,1,1])` → `([1,2,3,1],
  [...], [2,3,1,2])`.
- [x] AC-6: `histc` clamps out-of-range to boundary bins
  (`test_histc_clamps`).
- [x] AC-7: `topk(input, k>last_dim)` errors with `InvalidArgument`.
- [x] AC-8: GPU paths for `searchsorted` / `bucketize` (f32/f64) — SHIPPED
  (#1545). CUDA inputs lower the binary search on-device via
  `GpuBackend::searchsorted_1d` (`ferrotorch-gpu/src/search.rs`); only the
  int64 result indices are read back.
- [x] AC-9: GPU path for `topk` (f32/f64, last-dim, largest/smallest,
  sorted) — SHIPPED (#1545). CUDA inputs lower the k-selection on-device via
  `GpuBackend::topk_1d` (`ferrotorch-gpu/src/search.rs`); the values tensor
  stays GPU-resident and only the int64 indices are read back. `unique`,
  `unique_consecutive`, `histc`, `meshgrid`, and arbitrary non-last-dim
  `topk` remain CPU-only (sort/dedup/index-arithmetic GPU lowerings tracked
  as a follow-up under #1545).

## Architecture

`searchsorted` at `ops/search.rs:20-55`: validates `boundaries` is
1-D, then `vals.iter().map(|v| bounds.partition_point(|b| *b <= *v))`
for `right=true` or `partition_point(|b| *b < *v)` for
`right=false`. The `Vec<T>::partition_point` is Rust's standard
binary search for ascending-sorted slices.

`bucketize` at `:63-69` is a single-line delegation:
`searchsorted(boundaries, input, right)` with swapped argument
order.

`unique` at `:79-130`:
1. Empty input → return `([], [], [])`.
2. Build `indices: Vec<usize> = (0..n).collect()`, then
   `sort_by(|a, b| data[a].partial_cmp(&data[b]))`.
3. Walk sorted indices, emitting a new entry into `unique_vals` /
   `counts` when the value differs from the previous one;
   update `inverse[orig_idx]` to point at the current unique index.

`unique_consecutive` at `:140-177`:
1. Empty → empty triple.
2. Walk input left-to-right; if `data[i] == data[i-1]`, bump
   `counts.last_mut()`; else push a new entry into `output` /
   `counts`.

`histc` at `:186-219`:
1. Validate `bins > 0`, `min < max`.
2. `bin_width = (max - min) / bins`.
3. For each element, clamp to `[min, max - 1e-30]`, compute
   `idx = ((clamped - min) / bin_width) as usize`, clip to `bins-1`,
   increment `counts[idx]`.
4. Return `Tensor<T>` with shape `[bins]`.

`meshgrid` at `:228-279`:
1. Empty input → empty result.
2. Validate every input is 1-D and CPU.
3. For each input axis `dim`, walk every output flat index
   `flat in 0..total`, compute `coord = (flat / inner) % shapes[dim]`,
   emit `data[coord]`.
4. Return `Vec<Tensor<T>>` one per input axis.

`topk` at `:287-344`:
1. Validate `ndim >= 1`, `k <= last_dim`, CPU.
2. For each outer slice along the last dim, sort indices by value
   (descending for `largest=true`).
3. Take the first `k` of the sorted indices; emit corresponding
   values and original indices.
4. Output shape = input shape with last dim → `k`.

**Non-test consumer**: re-exported at `lib.rs:176` as
`ferrotorch_core::{bucketize, histc, meshgrid, searchsorted, topk,
unique, unique_consecutive}`. The boundary IS the public API per
goal.md S5. Downstream consumers in `ferrotorch-llama` use `topk`
for sampling and `unique` for vocab analysis; `meshgrid` is used by
`ferrotorch-nn::positional_embedding`.

## Parity contract

`parity_ops = []` (no specific parity op declared). Numeric contract
is byte-for-byte parity with the same-named torch ops, verified
through unit tests.

## Verification

`cargo test -p ferrotorch-core --lib ops::search` covers all 7
operations across normal-case + edge-case (empty, all-same, oob,
clamping) paths.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `searchsorted in ops/search.rs`; non-test consumer: re-exported as `ferrotorch_core::searchsorted` (boundary public API per goal.md S5). CUDA f32/f64 lower on-device via `GpuBackend::searchsorted_1d` (`gpu_searchsorted_f32`/`_f64 in ferrotorch-gpu/src/search.rs`); #1545. |
| REQ-2 | SHIPPED | impl: `bucketize in ops/search.rs`; non-test consumer: re-exported as `ferrotorch_core::bucketize`. Inherits the CUDA GPU path through its delegation to `searchsorted`. |
| REQ-3 | SHIPPED | impl: `unique` at `ops/search.rs:79`; non-test consumer: re-exported as `ferrotorch_core::unique` at `lib.rs:176` |
| REQ-4 | SHIPPED | impl: `unique_consecutive` at `ops/search.rs:140`; non-test consumer: re-exported as `ferrotorch_core::unique_consecutive` at `lib.rs:176` |
| REQ-5 | SHIPPED | impl: `histc` at `ops/search.rs:186`; non-test consumer: re-exported as `ferrotorch_core::histc` at `lib.rs:176` |
| REQ-6 | SHIPPED | impl: `meshgrid` at `ops/search.rs:228`; non-test consumer: re-exported as `ferrotorch_core::meshgrid` at `lib.rs:176` |
| REQ-7 | SHIPPED | impl: `topk` at `ops/search.rs:287`; non-test consumer: re-exported as `ferrotorch_core::topk` at `lib.rs:176` |
