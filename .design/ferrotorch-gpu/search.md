# GPU `searchsorted` / `bucketize` / `topk` / `histc` / `meshgrid` / `unique` kernels

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/Bucketization.cu
  - aten/src/ATen/native/BucketizationUtils.h
  - aten/src/ATen/native/cuda/TensorTopK.cpp
  - aten/src/ATen/native/cuda/TensorTopK.cu
  - aten/src/ATen/native/TopKImpl.h
  - aten/src/ATen/native/cuda/SummaryOps.cu
  - aten/src/ATen/native/TensorShape.cpp
  - aten/src/ATen/native/cuda/Unique.cu
  - aten/src/ATen/native/cuda/UniqueCub.cu
-->

## Summary

`ferrotorch-gpu/src/search.rs` provides the on-device binary-search kernel
that lowers `torch.searchsorted` / `torch.bucketize` for CUDA-resident
tensors. It mirrors `aten/src/ATen/native/cuda/Bucketization.cu`
(`searchsorted_cuda_kernel`, `lower_bound`, `upper_bound`) for the 1-D
boundaries case (`is_1d_boundaries = true`), which is the surface the
`ferrotorch-core` `searchsorted` / `bucketize` ops expose (1-D `boundaries`,
flat `values`, no `sorter`).

Each output element is an int64 insertion index (PyTorch returns
`ScalarType::Long` when `out_int32 == false`, which is the ferrotorch
default). The kernel runs one thread per value; each thread binary-searches
the entire boundary array.

## Boundary / tie semantics (the bug-prone part)

Matched byte-for-byte to upstream:

- `right == false` (PyTorch `side="left"`): `lower_bound` — first index `i`
  with `boundaries[i] >= val`. Upstream condition is `!(mid_val >= val)`
  advancing `start`. A value equal to a boundary lands ON that boundary's
  index (`seq[i-1] < v <= seq[i]`).
- `right == true` (PyTorch `side="right"`): `upper_bound` — first index `i`
  with `boundaries[i] > val`. Upstream condition is `!(mid_val > val)`
  advancing `start`. A value equal to a boundary lands AFTER it
  (`seq[i-1] <= v < seq[i]`).

This is the exact pair of half-open comparisons the CPU `partition_point`
path in `ferrotorch-core/src/ops/search.rs` already uses
(`partition_point(|b| *b < *v)` for left, `partition_point(|b| *b <= *v)` for
right), so the GPU and CPU paths agree bit-for-bit including the tie case
where a value equals a boundary.

### topk tie-break (the bug-prone part)

`torch.topk(input, k, dim=-1, largest, sorted=True)` on CUDA gathers the
top-k (unordered) then sorts the gathered slice with
`sortKeyValueInplace(.., stable=false)` (`topk_out_cuda`,
`aten/src/ATen/native/cuda/TensorTopK.cpp:101`). Because the post-gather sort
is NOT stable, the per-tie ORDER of original indices is unspecified by torch
— any permutation within a tie group is a correct `topk` result, and the CPU
path (`TopKImpl.h`, `std::sort`) is likewise non-stable. The ferrotorch GPU
kernel pins the deterministic choice of ascending original index on ties,
which is (a) a valid torch result and (b) bit-identical to the production CPU
path `ops::search::topk` (Rust's stable `sort_by`). The contract is therefore:
VALUES match torch exactly; INDICES match torch exactly outside tie groups
and select identical-value entries within tie groups. Live-oracle verified
against torch 2.11 CUDA (values exact in all cases; gathered values exact).

## Requirements

- REQ-1: `gpu_searchsorted_f32` / `gpu_searchsorted_f64` /
  `gpu_searchsorted_f16` / `gpu_searchsorted_bf16` — given a device
  `values` buffer of length `n_vals`, a device 1-D `boundaries` buffer of
  length `n_bounds`, and a `right: bool`, return a fresh device
  `CudaSlice<i64>` of length `n_vals` holding the per-value insertion index.
  One thread per value; serial binary search over `[0, n_bounds)`. Empty
  boundaries → every index is 0 (matches `partition_point` on an empty slice
  and upstream `start == end == 0`). Mirrors `searchsorted_cuda_kernel`.
- REQ-2: PTX template + ABI. A single `SEARCHSORTED_F32_PTX` /
  `SEARCHSORTED_F64_PTX` / `SEARCHSORTED_F16_PTX` /
  `SEARCHSORTED_BF16_PTX` carrying ABI
  `(vals_ptr, bounds_ptr, out_ptr, n_vals, n_bounds, right)` where `right`
  is a `u32` flag (0 = lower_bound, 1 = upper_bound). Loaded via
  `module_cache::get_or_compile`.
- REQ-3: `GpuBackend::searchsorted_1d` trait surface in
  `ferrotorch-core/src/gpu_dispatch.rs` — `(values, boundaries, right)` →
  `DType::I64` `GpuBufferHandle`. Default impl returns `InvalidArgument`;
  the CUDA backend overrides. Output dtype tag is int64 (PyTorch
  `ScalarType::Long`).
- REQ-4: Dispatch wiring in `CudaBackendImpl::searchsorted_1d`
  (`ferrotorch-gpu/src/backend_impl.rs`) — `match dtype { F32, F64, F16, BF16 }`,
  wrapping the resulting `CudaSlice<i64>` as an `I64` handle.
- REQ-5: Re-export + non-test production consumer. `pub use
  search::{gpu_searchsorted_f32, gpu_searchsorted_f64, gpu_searchsorted_f16,
  gpu_searchsorted_bf16}` in
  `ferrotorch-gpu/src/lib.rs`; the production consumer is
  `ferrotorch-core/src/ops/search.rs::searchsorted`, which dispatches CUDA
  f32/f64/f16/bf16 inputs through `GpuBackend::searchsorted_1d` and reads
  back ONLY the i64 result indices (the value/boundary data never leaves the
  device — no R-CODE-4 round trip).
- REQ-6: `gpu_topk_f32` / `gpu_topk_f64` / `gpu_topk_f16` /
  `gpu_topk_bf16` — given a device `[outer, dim]` value buffer and
  `(k, largest)`, return `(values, indices)`: a fresh device value buffer of
  `outer * k` extrema (same dtype) and a `CudaSlice<i64>` of `outer * k`
  original indices into `[0, dim)`, both in sorted order. One thread per
  output slice; serial selection of `k` extrema (`largest` → descending value,
  else ascending; ties broken by ascending original index). Mirrors the gather
  + `sortKeyValueInplace(stable=false)` contract of `topk_out_cuda` for the
  last-dim, sorted case.
- REQ-7: PTX template + ABI for topk. `TOPK_F32_PTX` / `TOPK_F64_PTX` /
  `TOPK_F16_PTX` / `TOPK_BF16_PTX` carry the 7-arg ABI
  `(in_ptr, vals_ptr, idx_ptr, outer, dim, k, largest)` where `largest` is a
  `u32` flag (1 = largest, 0 = smallest). Loaded via
  `module_cache::get_or_compile`.
- REQ-8: `GpuBackend::topk_1d` trait surface in
  `ferrotorch-core/src/gpu_dispatch.rs` — `(values_in, outer, last_dim, k,
  largest)` → `(values handle same dtype, I64 indices handle)`. Default impl
  returns `NotImplementedOnCuda`; the CUDA backend overrides. The indices
  handle is `DType::I64` (PyTorch `ScalarType::Long`).
- REQ-9: Dispatch wiring in `CudaBackendImpl::topk_1d`
  (`ferrotorch-gpu/src/backend_impl.rs`) — `match dtype { F32, F64, F16, BF16 }`,
  wrapping the value buffer with the same dtype tag and the index
  `CudaSlice<i64>` via `wrap_slice_i64`.
- REQ-10: Re-export + non-test production consumer for topk. `pub use
  search::{gpu_topk_f32, gpu_topk_f64, gpu_topk_f16, gpu_topk_bf16}` in
  `ferrotorch-gpu/src/lib.rs`; the production consumer is
  `ferrotorch-core/src/ops/search.rs::topk`, which on CUDA
  f32/f64/f16/bf16 dispatches through `GpuBackend::topk_1d`, keeps the VALUES
  tensor GPU-resident (`TensorStorage::gpu`), and reads back ONLY the i64
  indices (no R-CODE-4 round trip of the value data).
- REQ-11: `gpu_histc_f32` / `gpu_histc_f64` — given a device value buffer of
  `n` elements, `bins`, and inclusive range bounds `(min_val, max_val)`,
  return a fresh pre-zeroed device `CudaSlice<V>` of `bins` counts (same dtype
  as the input — PyTorch's `_histc_cuda` allocates the output with
  `self.scalar_type()`). One thread per input element; each thread computes its
  bin and `atom.global.add`s `1`. Bin semantics mirror `getBin` +
  `kernelHistogram1D` in `aten/src/ATen/native/cuda/SummaryOps.cu`:
  `bin = (int)((v - min) * bins / (max - min))`, the last bin is closed at both
  ends (`bin == bins -> bins-1`), and values outside `[min, max]` (and NaN) are
  skipped.
- REQ-12: histc PTX + ABI. `HISTC_F32_PTX` / `HISTC_F64_PTX` carry the 6-arg
  ABI `(in_ptr, out_ptr, n, nbins, minv, maxv)`. f32 uses `red.global.add.f32`
  (sm_20+); f64 targets `sm_60` for `red.global.add.f64`. Integer counts are
  exact in f32 up to 2^24 / f64 up to 2^53, matching the CPU path which
  accumulates `T::one()` per element.
- REQ-13: `GpuBackend::histc_1d` trait surface in
  `ferrotorch-core/src/gpu_dispatch.rs` — `(input, bins, min_val, max_val)` →
  same-dtype `GpuBufferHandle` of `bins` counts. Default impl returns
  `NotImplementedOnCuda`; the CUDA backend overrides.
- REQ-14: Dispatch wiring + non-test consumer. `CudaBackendImpl::histc_1d`
  (`ferrotorch-gpu/src/backend_impl.rs`) — `match dtype { F32, F64 }`, wrapping
  via `wrap_slice_{f32,f64}`. The production consumer is
  `ferrotorch-core/src/ops/search.rs::histc`, which on CUDA f32/f64 dispatches
  through `GpuBackend::histc_1d` and keeps the counts GPU-resident
  (`TensorStorage::gpu`) — no R-CODE-4 round trip.
- REQ-15: `gpu_meshgrid_f32` / `gpu_meshgrid_f64` / `gpu_meshgrid_f16` /
  `gpu_meshgrid_bf16` — given an axis's 1-D
  coordinate buffer (length `axis_len`), `total = product(shapes)`, and
  `inner = product(shapes[axis+1..])`, return a fresh device `CudaSlice<V>` of
  `total` elements with `out[flat] = input[(flat / inner) % axis_len]`. One
  thread per output element does the index arithmetic and a single gather load —
  no `expand` materialisation. Mirrors the `view(view_shape).expand(shape)`
  decomposition of upstream `meshgrid` (`indexing='ij'`).
- REQ-16: meshgrid PTX + ABI. `MESHGRID_F32_PTX` / `MESHGRID_F64_PTX` /
  `MESHGRID_U16_PTX` carry the 5-arg ABI
  `(in_ptr, out_ptr, total, inner, axis_len)`. The launcher forces `inner >= 1`
  so the `div.u32` divisor is never zero (last axis). `MESHGRID_U16_PTX`
  bit-copies raw f16/bf16 `u16` payloads without widening through f32; the
  surrounding `GpuBufferHandle` dtype tag distinguishes f16 from bf16.
- REQ-17: `GpuBackend::meshgrid_grid` trait surface in
  `ferrotorch-core/src/gpu_dispatch.rs` — `(input, total, inner, axis_len)` →
  same-dtype `GpuBufferHandle` of `total` elements. Default impl returns
  `NotImplementedOnCuda`; the CUDA backend overrides.
- REQ-18: Dispatch wiring + non-test consumer. `CudaBackendImpl::meshgrid_grid`
  (`ferrotorch-gpu/src/backend_impl.rs`) — `match dtype { F32, F64, F16, BF16 }`,
  wrapping via `wrap_slice_{f32,f64}` or the half/bf16 `wrap_buffer_*` helpers.
  The production consumer is
  `ferrotorch-core/src/ops/search.rs::meshgrid`, which when ALL inputs are CUDA
  f32/f64/f16/bf16 produces each axis grid through `GpuBackend::meshgrid_grid`
  and keeps every grid GPU-resident (`TensorStorage::gpu`) — no R-CODE-4 round
  trip.
- REQ-19: `gpu_unique_consecutive_f32` / `gpu_unique_consecutive_f64` /
  `gpu_unique_consecutive_f16` / `gpu_unique_consecutive_bf16` — given a
  device value buffer of `n` elements, collapse each maximal RUN of equal
  ADJACENT elements into a single output element. The output length is
  DATA-DEPENDENT. The on-device pipeline is: (a) a `run_flag_{f32,f64,f16,bf16}_kernel`
  marks each element `flag[i] = (i==0 || in[i] != in[i-1]) ? 1 : 0` into a
  device f32 buffer; (b) the existing `gpu_cumsum` primitive inclusive-prefix-
  sums the flags (flat axis `outer=1, dim_size=n, inner=1`), so `incl[i]` is the
  number of run-starts in `[0,i]` and `out_len = incl[n-1]`; (c) a
  `compact_{f32,f64,f16,bf16}_kernel` scatters each run-start value `in[i]` to
  `out[(u32)incl[i] - 1]`. Returns `(values, Vec<usize> inverse,
  Vec<usize> counts)`: `values` is the GPU-resident deduplicated output
  (`CudaBuffer<f32/f64>` or raw `CudaSlice<u16>` for f16/bf16);
  `inverse[i] = incl[i] - 1`; `counts[j]` is the run length of output run `j`.
  NaN starts its own run (`setp.neu` — the UNORDERED not-equal — is true for
  `NaN != NaN`, where ordered `setp.ne` would be false; #1656), matching the CPU
  PartialEq path and live torch. The 16-bit paths widen adjacent values to f32
  for the comparison so `-0.0` and `+0.0` collapse, then bit-copy the original
  run-start payload into the output.
- REQ-20: PTX + ABI for unique_consecutive. `RUN_FLAG_F32_PTX` /
  `RUN_FLAG_F64_PTX` / `RUN_FLAG_F16_PTX` / `RUN_FLAG_BF16_PTX` carry the 3-arg
  ABI `(in_ptr, flag_ptr, n)`; `COMPACT_F32_PTX` / `COMPACT_F64_PTX` /
  `COMPACT_F16_PTX` / `COMPACT_BF16_PTX` carry the 4-arg ABI
  `(in_ptr, incl_ptr, out_ptr, n)`. Loaded via `module_cache::get_or_compile`.
  The trait surface is
  `GpuBackend::unique_consecutive_1d(input, n) -> (GpuBufferHandle, Vec<usize>,
  Vec<usize>)` in `ferrotorch-core/src/gpu_dispatch.rs`; dispatch wiring in
  `CudaBackendImpl::unique_consecutive_1d`
  (`match dtype { F32, F64, F16, BF16 }`, wrapping values via
  `wrap_buffer{,_f64,_f16,_bf16}`). The non-test production consumer is
  `ferrotorch-core/src/ops/search.rs::unique_consecutive`, which on CUDA f32/f64/f16/bf16
  keeps the VALUES tensor GPU-resident (`TensorStorage::gpu`) and reads back
  ONLY the derived run-position metadata to build the host `inverse` / `counts`
  vectors (host `Vec<usize>` by the CPU signature) — the value data never leaves
  the device and returns, so this is NOT an R-CODE-4 round trip (the same
  contract as `searchsorted_1d` reading back its i64 indices).

- REQ-21: `gpu_unique_f32` / `gpu_unique_f64` / `gpu_unique_f16` /
  `gpu_unique_bf16` — given a device value buffer of
  `n` elements, return `(CudaBuffer<V>`/`CudaSlice<u16> values, Vec<usize> inverse,
  Vec<usize> counts)` for `torch.unique(sorted=True, return_inverse=True,
  return_counts=True)`: the SORTED-ascending DISTINCT elements (NaN entries last,
  each NaN a DISTINCT unique), the per-input index into `values`, and each
  unique's frequency. The output length is DATA-DEPENDENT. The CUDA `unique`
  ALWAYS sorts (no device hashtable in thrust); the on-device pipeline is: (a) an
  `unique_init_{f32,f64,f16,bf16}_kernel` builds a power-of-2 padded `(key, idx)` pair
  array (`key[i] = i<n ? in[i] : +INF`, `idx[i] = i<n ? i : i32::MAX`); (b) a
  `unique_bitonic_{f32,f64,f16,bf16}_kernel` sort-by-key network sorts ascending under a
  TOTAL-order comparator that ranks pads strictly last (via the `idx == i32::MAX`
  payload) and NaN as the maximum among real values (`setp.neu` self-compare —
  `a != a` iff `a` is NaN), breaking NaN ties by ascending original index and
  finite equal ties by descending original index so compaction preserves
  PyTorch's last-original signed-zero representative;
  (c) the existing `run_flags_and_scan` (`setp.neu` run-flag → `gpu_cumsum`
  inclusive scan) over the first `n` SORTED positions yields `incl`, and
  `launch_compact_{f32,f64,u16}` scatters each distinct-value run-start to its unique
  slot. Mirrors `compute_unique` in `aten/src/ATen/native/cuda/Unique.cu:51-85`
  (sort-by-key → `inverse[sorted_indices[i]] =
  inclusive_scan(adjacent_diff(not_equal))[i]` `:63-66` → run-length `counts`
  `:75-81`) and the `radix_sort_pairs` of `UniqueCub.cu:175`. Only the derived
  index/run metadata is read back (the `incl` scan + the i32 sorted-index
  permutation) to build the host `inverse` / `counts`; the VALUE data never
  leaves the device (no R-CODE-4 round trip).
- REQ-22: PTX + ABI for unique. `UNIQUE_INIT_F32_PTX` / `UNIQUE_INIT_F64_PTX`
  / `UNIQUE_INIT_F16_PTX` / `UNIQUE_INIT_BF16_PTX`
  carry the 5-arg ABI `(in_ptr, key_ptr, idx_ptr, n, npad)`;
  `UNIQUE_BITONIC_F32_PTX` / `UNIQUE_BITONIC_F64_PTX` /
  `UNIQUE_BITONIC_F16_PTX` / `UNIQUE_BITONIC_BF16_PTX` (generated by the
  `unique_bitonic_ptx!` / `unique_bitonic_16_ptx!` macros) carry the 5-arg ABI `(key_ptr, idx_ptr, npad, j,
  k)` — one `(k, j)` bitonic step per launch, the host driving the network
  (`k` doubling `2..=npad`, `j` halving `k/2..=1`). The dedup/compaction reuse
  the `unique_consecutive` `RUN_FLAG_*` / `COMPACT_*` PTX + `gpu_cumsum`. The
  comparator uses only valid PTX (no `setp`/`mov` on `.pred` operands): each
  sub-predicate is materialised to a u32 via `selp.u32` and combined with
  `setp.*.u32` branches.
- REQ-23: `GpuBackend::unique_1d` trait surface in
  `ferrotorch-core/src/gpu_dispatch.rs` — `(input, n)` → `(values handle same
  dtype, Vec<usize> inverse, Vec<usize> counts)`. Default impl returns
  `NotImplementedOnCuda`; the CUDA backend overrides. Dispatch wiring in
  `CudaBackendImpl::unique_1d` (`ferrotorch-gpu/src/backend_impl.rs`, `match
  dtype { F32, F64, F16, BF16 }`, wrapping values via `wrap_buffer{,_f64,_f16,_bf16}`). Re-export +
  non-test production consumer: `pub use search::{gpu_unique_f32, gpu_unique_f64, gpu_unique_f16, gpu_unique_bf16}`
  in `ferrotorch-gpu/src/lib.rs`; the consumer is
  `ferrotorch-core/src/ops/search.rs::unique`, which on CUDA f32/f64/f16/bf16 keeps the
  SORTED-unique VALUES GPU-resident (`TensorStorage::gpu`) and reads back ONLY
  the run-position metadata for the host `inverse` / `counts` (host `Vec<usize>`
  by the CPU signature) — no R-CODE-4 value round trip.

## Acceptance Criteria

- [x] AC-1: `gpu_searchsorted_f32([1,3,5,7], [0,2,3,6,8], right=true)` →
  `[0,1,2,3,4]` (matches CPU `test_searchsorted_right`).
- [x] AC-2: `gpu_searchsorted_f32([1,3,5,7], [1,3,5,7], right=false)` →
  `[0,1,2,3]` (value exactly on a boundary, left side → boundary index).
- [x] AC-3: same values, `right=true` → `[1,2,3,4]` (value on a boundary,
  right side → after it). The right-vs-left tie divergence is the
  bug-prone case and is pinned by a dedicated GPU test.
- [x] AC-4: empty boundaries → all-zero output.
- [x] AC-5: the result buffer `is_cuda()` (lives on device); only the
  decoded indices are read to host.
- [x] AC-6: `gpu_topk_f32([3,1,4,1,5,9], k=3, largest=true)` → values
  `[9,5,4]`, indices `[5,4,2]` (matches CPU `test_topk_largest` and torch).
- [x] AC-7: `gpu_topk_f32([3,1,4,1,5], k=2, largest=false)` → values
  `[1,1]`, indices `[1,3]` — the equal-value tie resolves to ascending
  original index.
- [x] AC-8: `gpu_topk_f32([2,2,2,2,1], k=3, largest=true)` → indices
  `[0,1,2]` (all-ties → ascending index; the divergence-prone tie case,
  pinned by a dedicated GPU test).
- [x] AC-9: multi-row `[2,4]` topk runs one thread per row and matches the
  CPU reference per row.
- [x] AC-10: f64 topk (`gpu_topk_f64`) matches the CPU reference for both
  `largest=true` and `largest=false`.
- [x] AC-11: `gpu_histc_f32(arange(0,11.), bins=5, min=0, max=10)` →
  `[2,2,2,2,3]` (the value `10 == max` lands in the closed last bin), matching
  live `torch.histc` (torch 2.11 CUDA) and the upstream-getBin CPU reference.
- [x] AC-12: `gpu_histc_f32([-1,.5,1.5,2.5,4,5,nan], 4, 0, 4)` →
  `[1,1,1,1]` — values below min, above max, and NaN are all skipped (torch).
- [x] AC-13: `gpu_histc_f64([0,.25,.5,.75,1.], 4, 0, 1)` → `[1,1,1,2]` (f64
  atomic path; `1.0 == max` in the closed last bin), matching live torch.
- [x] AC-14: the histc result buffer `is_cuda()` (lives on device); the
  consumer `ferrotorch_core::histc` returns a CUDA tensor (no value round trip).
- [x] AC-15: `gpu_meshgrid_f32` and `gpu_meshgrid_f16`
  `([1,2,3],[4,5], indexing='ij')` → grid0 `[1,1,2,2,3,3]`, grid1
  `[4,5,4,5,4,5]` (shape `[3,2]`), matching live
  `torch.meshgrid(..., indexing='ij')`; the f16 test checks raw u16 bits.
- [x] AC-16: `gpu_meshgrid_f64` and `gpu_meshgrid_bf16` 3-axis grids match the
  `view(view_shape).expand(shape)` CPU reference per axis; the bf16 test checks
  raw u16 bits.
- [x] AC-17: the meshgrid result grids `is_cuda()`; the consumer
  `ferrotorch_core::meshgrid` returns CUDA tensors when all inputs are CUDA.
- [x] AC-18: `unique_consecutive([1,1,2,3,3,3,1])` → values `[1,2,3,1]`,
  inverse `[0,0,1,2,2,2,3]`, counts `[2,1,3,1]`, matching live
  `torch.unique_consecutive(return_inverse=True, return_counts=True)` and the
  CPU path (f32 + f64). Half coverage additionally pins
  `unique_consecutive([1,1,nan,nan,-0,+0,2,2,1])` for f16/bf16: NaNs split,
  `-0/+0` collapse, and raw output bits preserve the run-start value. Pinned by
  `unique_consecutive_f{16,32,64}_*` and `unique_consecutive_bf16_*` tests.
- [x] AC-19: no-duplicates input `[1,2,3,4,5]` → identity values, counts all
  `1`; all-same input `[7,7,7,7]` → values `[7]`, counts `[4]`; 2-D input
  flattens in C-order. Pinned by
  `unique_consecutive_f32_no_duplicates_is_identity`,
  `unique_consecutive_f32_all_same_collapses_to_one`,
  `unique_consecutive_f64_2d_input_flattens_like_torch`.
- [x] AC-20: the deduplicated values tensor `is_cuda()` (lives on device); only
  the run-position metadata is read back to build `inverse` / `counts` (no
  value round trip).
- [x] AC-21: `unique([3,1,2,1,3])` → values `[1,2,3]`, inverse `[2,0,1,0,2]`,
  counts `[2,1,2]`, matching live `torch.unique(sorted=True,
  return_inverse=True, return_counts=True)` (torch 2.11 CUDA) and the CPU path
  (f32 + f64). Pinned by `unique_f32_basic_matches_torch`,
  `unique_f64_basic_matches_torch`.
- [x] AC-22: SORTED-output coverage — already-sorted, reverse-sorted, all-same
  (`[7,7,7,7]→[7]` cnt `[4]`), all-distinct, single element, negative+positive
  mixed; all match torch. Pinned by `unique_f32_already_sorted`,
  `unique_f32_reverse_sorted`, `unique_f32_all_same`, `unique_f32_all_distinct`,
  `unique_f32_single_element`, `unique_f32_negative_and_positive`.
- [x] AC-23: NON-power-of-2 lengths (5, 7, 100, 1000) — the bitonic +INF padding
  is fully excluded (no sentinel leaks into the output), the inverse reconstructs
  the input bit-for-bit, counts sum to `n`, and the output is sorted. Pinned by
  `unique_f32_len7_non_pow2`, `unique_f64_non_pow2_len7`,
  `unique_f32_len100_non_pow2_invariants`, `unique_f32_len1000_non_pow2_invariants`.
- [x] AC-24: NaN — `torch.unique` does NOT collapse NaNs (each NaN a DISTINCT
  unique sorted to the END), verified live: `unique([nan,1,nan,2,nan])` →
  `[1,2,nan,nan,nan]` inverse `[2,0,3,1,4]` counts `[1,1,1,1,1]`; the GPU
  comparator breaks NaN ties by ascending original index, matching torch's
  radix-stable order. `±inf` sort correctly (`[-inf,1,inf,nan]`). Pinned by
  `unique_f32_nan_each_distinct_at_end`, `unique_f32_inf_and_nan`,
  `unique_f64_nan_distinct_tail`.
- [x] AC-25: the SORTED-unique values tensor `is_cuda()`; only the index/run
  metadata is read back (no value round trip). GPU == CPU on identical finite
  data (`unique_f32_gpu_equals_cpu_finite`). f16/bf16 CUDA inputs run the same
  resident pipeline and match live torch probes for NaN distinctness, inverse,
  counts, and signed-zero representative bits (`unique_f16_cuda_matches_torch_probe`,
  `unique_bf16_cuda_matches_torch_probe`, `unique_f16_signed_zero_keeps_last_original_representative`,
  `unique_bf16_signed_zero_keeps_last_original_representative`).

## Architecture

`gpu_searchsorted_f32` (`search.rs`):
1. Validate device ordinals match across `values` and `boundaries`.
2. `n_vals == 0` → empty i64 buffer (no launch).
3. Range-check `n_vals` / `n_bounds` against `u32::MAX`.
4. `module_cache::get_or_compile(ctx, SEARCHSORTED_F32_PTX, ...)`.
5. Alloc `CudaSlice<i64>` of `n_vals`; launch one thread per value.
6. Each thread loops `lo=0, hi=n_bounds`; mid-value compare per `right`
   flag; writes `lo` (the converged insertion point) as `s64`.

`searchsorted_1d` (`backend_impl.rs`): unwrap f32/f64/f16/bf16 value+boundary
buffers, call the matching `gpu_searchsorted_*`, wrap result via
`wrap_slice_i64`.

Non-test consumer (`ops/search.rs::searchsorted`): on CUDA f32/f64/f16/bf16,
`backend.searchsorted_1d(values.gpu_handle()?, boundaries.gpu_handle()?,
right)?`, then `gpu_to_cpu` the i64 handle, decode 8-byte LE chunks to
`Vec<usize>`. `bucketize` inherits the GPU path through its existing
delegation to `searchsorted`.

## Parity contract

`parity_ops = []` — the `searchsorted` / `bucketize` / `topk` / `histc` /
`meshgrid` route family has no parity-sweep runner arm (a TEST-INFRASTRUCTURE
gap tracked separately per goal.md S5, not a REQ blocker). The numeric
contract is byte-for-byte parity with the CPU path: `partition_point` for
searchsorted (right=true/false tie cases), the stable-sort selection for topk
(largest/smallest, ties, multi-row, k==dim, f64), the upstream-getBin
assignment for histc (interior bins + closed last bin + skipped oob/NaN), and
the `view().expand()` index map for meshgrid. The topk, histc, and meshgrid
contracts were additionally validated live against the torch 2.11 CUDA oracle
(`torch.topk` / `torch.histc` / `torch.meshgrid` outputs exact).

NOTE (spillover, NOT fixed here): the CPU `histc` path
(`ferrotorch-core/src/ops/search.rs::histc`) CLAMPS out-of-range values into
the boundary bins (`clamp(v, min, max-1e-30)`), whereas `torch.histc` (and this
GPU path) SKIP out-of-range values. The two agree for in-range data (which all
parity tests use) but diverge when the input contains values outside
`[min, max]`. This is a pre-existing CPU-only divergence from torch, filed as a
separate blocker for a single-file acto-fixer dispatch.

## Verification

`cargo test -p ferrotorch-gpu --features cuda --lib search::` exercises the
kernels live on the device (skips cleanly when no GPU is present), asserting
both the values and that the output buffer is device-resident. The end-to-end
consumer path (result `is_cuda()` + torch value-match) is pinned by
`ferrotorch-core/tests/divergence_histc_meshgrid_gpu.rs` (5 GPU-gated tests).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | `pub fn gpu_searchsorted_f32` / `gpu_searchsorted_f64` / `gpu_searchsorted_f16` / `gpu_searchsorted_bf16 in search.rs` mirror `lower_bound`/`upper_bound` in `aten/src/ATen/native/cuda/Bucketization.cu:26,44`; consumer `CudaBackendImpl::searchsorted_1d in backend_impl.rs` |
| REQ-2 | SHIPPED | `SEARCHSORTED_F32_PTX` / `SEARCHSORTED_F64_PTX` / `SEARCHSORTED_F16_PTX` / `SEARCHSORTED_BF16_PTX in search.rs` carry the 6-arg ABI; launch site binds args in matching order |
| REQ-3 | SHIPPED | `fn searchsorted_1d in gpu_dispatch.rs` (`GpuBackend` trait method, `DType::I64` output); consumer `ops::search::searchsorted` GPU branch |
| REQ-4 | SHIPPED | `CudaBackendImpl::searchsorted_1d in backend_impl.rs` dispatches `match dtype { F32, F64, F16, BF16 }` and wraps via `wrap_slice_i64` |
| REQ-5 | SHIPPED | `pub use search::{gpu_searchsorted_f32, gpu_searchsorted_f64, gpu_searchsorted_f16, gpu_searchsorted_bf16} in lib.rs`; non-test consumer `ferrotorch_core::ops::search::searchsorted` CUDA f32/f64/f16/bf16 branch (`searchsorted in ops/search.rs`) |
| REQ-6 | SHIPPED | `pub fn gpu_topk_f32` / `gpu_topk_f64` / `gpu_topk_f16` / `gpu_topk_bf16 in search.rs` select k extrema mirroring `topk_out_cuda` gather+`sortKeyValueInplace` at `aten/src/ATen/native/cuda/TensorTopK.cpp:97,106`; consumer `CudaBackendImpl::topk_1d in backend_impl.rs` |
| REQ-7 | SHIPPED | `TOPK_F32_PTX` / `TOPK_F64_PTX` / `TOPK_F16_PTX` / `TOPK_BF16_PTX in search.rs` carry the 7-arg ABI `(in_ptr, vals_ptr, idx_ptr, outer, dim, k, largest)`; launch site binds args in matching order |
| REQ-8 | SHIPPED | `fn topk_1d in gpu_dispatch.rs` (`GpuBackend` trait method, `(values, I64 indices)` output); consumer `ops::search::topk` GPU branch |
| REQ-9 | SHIPPED | `CudaBackendImpl::topk_1d in backend_impl.rs` dispatches `match dtype { F32, F64, F16, BF16 }`, wraps values with the original dtype tag and indices via `wrap_slice_i64` |
| REQ-10 | SHIPPED | `pub use search::{gpu_topk_f32, gpu_topk_f64, gpu_topk_f16, gpu_topk_bf16} in lib.rs`; non-test consumer `ferrotorch_core::ops::search::topk` CUDA f32/f64/f16/bf16 branch keeps VALUES GPU-resident (`TensorStorage::gpu`), reads back only i64 indices (`topk in ops/search.rs`) |
| REQ-11 | SHIPPED | `pub fn gpu_histc_f32` / `gpu_histc_f64 in search.rs` mirror getBin + last-bin clamp + range guard at `aten/src/ATen/native/cuda/SummaryOps.cu:41,47,92`; consumer `CudaBackendImpl::histc_1d in backend_impl.rs` |
| REQ-12 | SHIPPED | `HISTC_F32_PTX` / `HISTC_F64_PTX in search.rs` carry the 6-arg ABI `(in,out,n,nbins,minv,maxv)`; f32 `red.global.add.f32` (sm_52), f64 `red.global.add.f64` (sm_60) |
| REQ-13 | SHIPPED | `fn histc_1d in gpu_dispatch.rs` (`GpuBackend` trait method, same-dtype counts output); consumer `ops::search::histc` GPU branch |
| REQ-14 | SHIPPED | `CudaBackendImpl::histc_1d in backend_impl.rs` dispatches `match dtype { F32, F64 }`, wraps via `wrap_slice_{f32,f64}`; non-test consumer `ferrotorch_core::ops::search::histc` CUDA f32/f64 branch keeps counts GPU-resident (`TensorStorage::gpu`, `histc in ops/search.rs`) |
| REQ-15 | SHIPPED | `pub fn gpu_meshgrid_f32` / `gpu_meshgrid_f64` / `gpu_meshgrid_f16` / `gpu_meshgrid_bf16 in search.rs` mirror `view(view_shape).expand(shape)` at `aten/src/ATen/native/TensorShape.cpp:4462`; consumer `CudaBackendImpl::meshgrid_grid in backend_impl.rs` |
| REQ-16 | SHIPPED | `MESHGRID_F32_PTX` / `MESHGRID_F64_PTX` / `MESHGRID_U16_PTX in search.rs` carry the 5-arg ABI `(in,out,total,inner,axis_len)`; launcher forces `inner >= 1` (no zero-divisor); u16 path bit-copies f16/bf16 payloads |
| REQ-17 | SHIPPED | `fn meshgrid_grid in gpu_dispatch.rs` (`GpuBackend` trait method, same-dtype grid output); consumer `ops::search::meshgrid` GPU branch |
| REQ-18 | SHIPPED | `CudaBackendImpl::meshgrid_grid in backend_impl.rs` dispatches `match dtype { F32, F64, F16, BF16 }`, wraps via `wrap_slice_{f32,f64}` / `wrap_buffer_{f16,bf16}`; non-test consumer `ferrotorch_core::ops::search::meshgrid` CUDA f32/f64/f16/bf16 branch keeps each grid GPU-resident (`TensorStorage::gpu`, `meshgrid in ops/search.rs`) |
| REQ-19 | SHIPPED | `pub fn gpu_unique_consecutive_f32` / `gpu_unique_consecutive_f64` / `gpu_unique_consecutive_f16` / `gpu_unique_consecutive_bf16 in ferrotorch-gpu/src/search.rs` run the on-device run-flag → `gpu_cumsum` prefix-sum → compaction pipeline; consumer `CudaBackendImpl::unique_consecutive_1d in ferrotorch-gpu/src/backend_impl.rs` dispatches `match dtype { F32, F64, F16, BF16 }` and wraps values via `wrap_buffer{,_f64,_f16,_bf16}` |
| REQ-20 | SHIPPED | `RUN_FLAG_F32_PTX`/`RUN_FLAG_F64_PTX`/`RUN_FLAG_F16_PTX`/`RUN_FLAG_BF16_PTX` (3-arg `(in,flag,n)`) + `COMPACT_F32_PTX`/`COMPACT_F64_PTX`/`COMPACT_F16_PTX`/`COMPACT_BF16_PTX` (4-arg `(in,incl,out,n)`) `in ferrotorch-gpu/src/search.rs`; trait `fn unique_consecutive_1d in ferrotorch-core/src/gpu_dispatch.rs`; re-export `pub use search::{gpu_unique_consecutive_f32, gpu_unique_consecutive_f64, gpu_unique_consecutive_f16, gpu_unique_consecutive_bf16} in ferrotorch-gpu/src/lib.rs`; non-test consumer `ferrotorch_core::ops::search::unique_consecutive` CUDA f32/f64/f16/bf16 branch keeps the deduplicated VALUES GPU-resident (`TensorStorage::gpu`, `unique_consecutive in ferrotorch-core/src/ops/search.rs`), reads back only run-position metadata for the host `inverse`/`counts` vectors (no R-CODE-4 value round trip) |
| REQ-21 | SHIPPED | `pub fn gpu_unique_f32` / `gpu_unique_f64` / `gpu_unique_f16` / `gpu_unique_bf16 in ferrotorch-gpu/src/search.rs` run the on-device init → bitonic sort-by-key → run-flag → `gpu_cumsum` → compaction pipeline mirroring `compute_unique` at `aten/src/ATen/native/cuda/Unique.cu:51-85` (sort-by-key `radix_sort_pairs` `UniqueCub.cu:175`, inverse scatter `Unique.cu:63-66`, run-length counts `:75-81`); non-test consumer `CudaBackendImpl::unique_1d in ferrotorch-gpu/src/backend_impl.rs` dispatches `match dtype { F32, F64, F16, BF16 }` and wraps values via `wrap_buffer{,_f64,_f16,_bf16}` |
| REQ-22 | SHIPPED | `UNIQUE_INIT_F32_PTX`/`UNIQUE_INIT_F64_PTX`/`UNIQUE_INIT_F16_PTX`/`UNIQUE_INIT_BF16_PTX` (5-arg `(in,key,idx,n,npad)`) + `UNIQUE_BITONIC_F32_PTX`/`UNIQUE_BITONIC_F64_PTX`/`UNIQUE_BITONIC_F16_PTX`/`UNIQUE_BITONIC_BF16_PTX` (5-arg `(key,idx,npad,j,k)`) `in ferrotorch-gpu/src/search.rs`; the comparator ranks pads (`idx == i32::MAX`) last and NaN (`setp.neu` self-compare) as max, breaking NaN ties by ascending original index and finite equal ties by descending original index for signed-zero parity, using only `selp.u32`/`setp.*.u32` (no `.pred` arithmetic); dedup/compaction reuse `RUN_FLAG_*`/`COMPACT_*`/`gpu_cumsum` |
| REQ-23 | SHIPPED | `fn unique_1d in ferrotorch-core/src/gpu_dispatch.rs` (`GpuBackend` trait method, `(values, inverse, counts)` output, default `NotImplementedOnCuda`); re-export `pub use search::{gpu_unique_f32, gpu_unique_f64, gpu_unique_f16, gpu_unique_bf16} in ferrotorch-gpu/src/lib.rs`; non-test consumer `ferrotorch_core::ops::search::unique` CUDA f32/f64/f16/bf16 branch keeps the SORTED-unique VALUES GPU-resident (`TensorStorage::gpu`, `unique in ops/search.rs`), reads back only the index/run metadata for the host `inverse`/`counts` (no R-CODE-4 value round trip) |
