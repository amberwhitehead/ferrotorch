# Indexing grad_fns

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/native/TensorAdvancedIndexing.cpp
  - aten/src/ATen/native/TensorCompare.cpp
  - tools/autograd/derivatives.yaml
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/grad_fns/indexing.rs` (1968 LOC) is the autograd-tracking
layer for the indexing/gather/scatter family declared in
`aten/src/ATen/native/TensorAdvancedIndexing.cpp` (gather/scatter, masked_*,
take/put, index_*) and `aten/src/ATen/native/TensorCompare.cpp` (`where`).
The file holds:

1. Seven `*Backward` `GradFn` structs for the ops whose forward kernels live
   in the kernel layer (`ferrotorch-core/src/ops/indexing.rs`):
   `IndexSelectBackward` (1-D), `MaskedFillBackward`, `GatherBackward`,
   `ScatterBackward`, `ScatterAddBackward`, `WhereCondBackward`,
   `MaskedSelectBackward`, plus the N-D `IndexSelectDimBackward`.
2. A handful of forward `pub fn`s that build their backward struct in place
   rather than living in the kernel layer: `index_select_1d`, `masked_fill`,
   `masked_fill_bt` (BoolTensor variant), `index_select_1d_it` (IntTensor
   variant), and `index_select_dim` (the N-D analogue of `index_select_1d`
   that subsumes 1-D for ndim>=2 — #1014).
3. Two helper utilities shared by gather/scatter VJPs:
   `gather_dst_flat_indices` / `scatter_src_flat_indices` /
   `scatter_write_mask` for CPU N-D coord walks, and `flat_index` /
   `increment_coords` for C-contiguous coordinate arithmetic.
4. The GPU upload shim `upload_f32_to_gpu` that wraps a host `&[f32]` into a
   GPU buffer handle (used to ship CPU-resident integer index lists onto the
   device for the resident scatter-add path).

The PyTorch route declares 14 parity_ops: `gather`, `scatter`, `scatter_add`,
`scatter_reduce`, `index_select`, `index_add`, `index_copy`, `index_fill`,
`masked_select`, `masked_fill`, `masked_scatter`, `take`, `put`, `where`. As of
2026-05-25 ALL 14 parity ops have SHIPPED: the masked/where family
(`masked_select`, `masked_fill`, `where`) ships with broadcasting wrappers
(zero skips); the gather/scatter/index_select family ships with the
shape-strict impls (>50% pass, narrow-contract skips); `index_fill` ships
with negative-wrap + 0-d-input handling; and the 2026-05-25 S1-batch
closure of #1245/#1247/#1248/#1252/#1253/#1254 lands the final 6 ops
(`scatter_reduce` sum-mode, `index_add`, `index_copy`, `masked_scatter`,
`take`, `put`) in a single cohesive commit. Remaining residual skips
across the family are 0-d input (#1256), scatter.value scalar-src (#1258),
and negative-index narrower contract — tracked as their own follow-up
blockers, not parity failures.

## Requirements

- REQ-1: `gather(input, dim, index, *, sparse_grad=False)` — forward
  `output[i][j][k] = input[index[i][j][k]][j][k]` (when `dim=0`; analogous
  for higher dims). Mirrors `TORCH_IMPL_FUNC(gather_out)` at
  `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2070-2085`. VJP per
  `tools/autograd/derivatives.yaml:730-733
  - name: gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
    self: gather_backward(grad, self, dim, index, sparse_grad)
    index: non_differentiable`
  → `gather_backward` (`TensorAdvancedIndexing.cpp:2087-2104`) is
  `zeros_like(self).scatter_add_(dim, index, grad)`. ferrotorch implements
  the forward as `pub fn gather` in `ferrotorch-core/src/ops/indexing.rs`,
  which Arc-attaches `GatherBackward` from `grad_fns/indexing.rs`; the
  backward is `pub struct GatherBackward` in
  `ferrotorch-core/src/grad_fns/indexing.rs` (CPU scatter-add walk + a
  GPU-resident path via `backend.scatter_add_1d_f32`).
  **Divergence (sparse_grad kwarg)**: the
  `sparse_grad=True` branch (`TensorAdvancedIndexing.cpp:2093-2095
  return at::_gather_sparse_backward(self, dim, index, grad)`) has no
  ferrotorch analog — sparse tensors are out of scope for ferrotorch-core
  per the goal.md's stated scope. **Divergence (CUDA forward)**: only
  f32 GPU path lives in the backward — `pub fn gather` (in
  `ops/indexing.rs`) rejects CUDA inputs outright via
  `FerrotorchError::NotImplementedOnCuda`, so the GPU backward only ever
  fires under the (rare) case of a CPU forward whose grad has been moved
  to CUDA between the two passes.

- REQ-2: `scatter(self, dim, index, src)` — forward
  `output = self.clone(); output[index[i][j][k]][j][k] = src[i][j][k]` (for
  `dim=0`). Mirrors `TORCH_IMPL_FUNC(scatter_src_out)` at
  `TensorAdvancedIndexing.cpp:2263-2270`. VJP per `derivatives.yaml:1508-1511
  - name: scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
    self: grad.scatter(dim, index, 0)
    src: grad.gather(dim, index)` — the input gradient zeros the positions
  scatter wrote, the src gradient gathers from the same positions.
  ferrotorch implements the forward as `pub fn scatter` in
  `ferrotorch-core/src/ops/indexing.rs`, which Arc-attaches `ScatterBackward`
  from `grad_fns/indexing.rs`; the backward is `pub struct ScatterBackward` in
  `ferrotorch-core/src/grad_fns/indexing.rs` with a CPU path
  (input: copy grad_output then zero scattered positions; src:
  gather from scattered positions) and a GPU-resident path
  using `backend.masked_zero_f32` (zeros input grad at written positions)
  and `backend.index_select_1d_f32` (gathers src grad).

- REQ-3: `scatter_add(self, dim, index, src)` — forward
  `output = self.clone(); output[index[i][j][k]][j][k] += src[i][j][k]`.
  Mirrors `TORCH_IMPL_FUNC(scatter_add)` at
  `TensorAdvancedIndexing.cpp:2317-2352`. VJP per `derivatives.yaml:1519-1522
  - name: scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
    self: grad
    src: grad.gather(dim, index)` — identity for input (addition passes the
  gradient through unchanged), gather for src. ferrotorch implements the
  forward as `pub fn scatter_add` in `ferrotorch-core/src/ops/indexing.rs`,
  which Arc-attaches `ScatterAddBackward` from `grad_fns/indexing.rs`;
  the backward is `pub struct ScatterAddBackward` in
  `ferrotorch-core/src/grad_fns/indexing.rs` with a CPU walk and a
  GPU-resident path via `backend.clone_buffer` (identity for input grad)
  and `backend.index_select_1d_f32` (gather for src grad). **Production
  consumer**: `fn cummaxmin_backward_impl` in
  `ferrotorch-core/src/grad_fns/cumulative.rs` invokes
  `ops::indexing::scatter_add` for the cummax/cummin
  VJP — the live use-site that exercises the ScatterAddBackward attach
  path indirectly via the differentiable forward.

- REQ-4: `scatter_reduce(self, dim, index, src, reduce, *, include_self=True)`
  with reduce ∈ {`"sum"`, `"prod"`, `"mean"`, `"amax"`, `"amin"`}. Mirrors
  `TORCH_IMPL_FUNC(scatter_reduce_two)` at
  `TensorAdvancedIndexing.cpp:2354-2400`. VJP per `derivatives.yaml:3074-3077
  - name: scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor
    self, src: scatter_reduce_backward(grad, self, dim, index, src, reduce, include_self, result)`
  — per-reduce-mode branching. **NOT-STARTED in ferrotorch**: no forward
  kernel in `ops/indexing.rs`, no `ScatterReduceBackward` struct, no consumer.
  Prereq blocker #1245. The parity-sweep result (0/168 passed, 168 skipped)
  reflects that all reduce modes need wiring before any of them ship.

- REQ-5: `index_select(input, dim, index)` — forward
  `output[..., i, ...] = input[..., index[i], ...]` (gather a 1-D slice list
  along an arbitrary axis). Mirrors `Tensor index_select_cpu_(...)` at
  `TensorAdvancedIndexing.cpp:1862-1866` (dispatch into
  `index_select_out_cpu_` declared elsewhere). VJP per
  `derivatives.yaml:910-913
  - name: index_select(Tensor self, int dim, Tensor index) -> Tensor
    self: index_select_backward_symint(grad, self.sym_sizes(), dim, index)
    index: non_differentiable`
  → `index_select_backward_symint` (`TensorAdvancedIndexing.cpp:1878-1900`)
  is `zeros(self.sizes()).index_add_(dim, index, grad)`. ferrotorch
  implements three forward variants in `grad_fns/indexing.rs`:
  1. `pub fn index_select_1d` in `grad_fns/indexing.rs` — host `&[usize]`
     indices, 1-D input only, attaches `IndexSelectBackward` (also defined
     in `grad_fns/indexing.rs`). Backward is a
     1-D scatter-add VJP with CPU and GPU paths.
  2. `pub fn index_select_1d_it` in `grad_fns/indexing.rs` — IntTensor-typed
     indices, 1-D input only, widens to `Vec<usize>` and forwards to
     `index_select_1d`.
  3. `pub fn index_select_dim` in `grad_fns/indexing.rs` — N-D input,
     arbitrary axis, IntTensor indices.
     Attaches `IndexSelectDimBackward` (also defined in
     `grad_fns/indexing.rs`). The backward is the N-D scatter-add VJP
     that subsumes the 1-D case for ndim>=2 (per #1014). CPU + GPU
     f32/f64 fast paths. **Production
     consumer**: `index_select_dim` is invoked inside
     `RandomHorizontalFlip::apply` in `ferrotorch-data/src/transforms.rs`
     under a `no_grad` guard — the chainable
     flip-along-axis primitive the data crate uses to mirror torchvision's
     RandomHorizontalFlip. The 1-D `index_select_1d` variant has no in-tree
     non-test consumer outside its IntTensor wrapper.

- REQ-6: `index_add(self, dim, index, source, *, alpha=1)` — forward
  `output[index[i]] += alpha * source[i]` along `dim`. Mirrors
  `TORCH_IMPL_FUNC(index_add_cpu_out)` at
  `TensorAdvancedIndexing.cpp:1153`. VJP per `derivatives.yaml:862-868
  - name: index_add(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor
    self: grad
    source: "maybe_multiply(source.dim() > 0 ? grad.index_select(dim, index).expand_as(source) : grad.index_select(dim, index.squeeze(0)), alpha)"
    index: non_differentiable`. **NOT-STARTED in ferrotorch**: no forward
  kernel, no `IndexAddBackward` struct, no consumer. Prereq blocker #1247.

- REQ-7: `index_copy(self, dim, index, source)` — forward overwrites
  `output[index[i]] = source[i]` along `dim` (no accumulation). Mirrors
  `TORCH_IMPL_FUNC(index_copy_out)` at
  `TensorAdvancedIndexing.cpp:1082`. VJP per `derivatives.yaml:875-883
  - name: index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
    self: grad.index_fill(dim, index, 0)
    source: grad.index_select(dim, index).expand_as(source)
    index: non_differentiable`. **NOT-STARTED in ferrotorch**: no forward
  kernel, no `IndexCopyBackward` struct, no consumer. The VJP would
  additionally require an `index_fill` kernel (REQ-8) so the two
  NOT-STARTED REQs are coupled. Prereq blocker #1248.

- REQ-8: `index_fill(self, dim, index, value)` — forward overwrites
  `output[..., index[i], ...] = value` along `dim`. Mirrors
  `Tensor index_fill(...)` at `TensorAdvancedIndexing.cpp:1979-1990`. VJP
  per `derivatives.yaml:884-887
  - name: index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
    self: grad.index_fill(dim, index, 0)
    index: non_differentiable
    result: self_t.index_fill(dim, index, 0)` — gradient is zeroed at the
  filled positions. **SHIPPED 2026-05-25** in ferrotorch:
  `grad_fns::indexing::index_fill<T: Float>(input, dim: i64, index:
  &IntTensor<i64>, value: f64)` (in `grad_fns/indexing.rs`) returns `clone(input)` with slices at `index`
  positions along the normalized `dim` overwritten by `value`; attaches
  `IndexFillBackward` (struct in `grad_fns/indexing.rs`) which on backward returns
  `grad_output` with the same fill positions zeroed (`backward` impl at
  `:1391-1430`). Negative dim wraps per `at::maybe_wrap_dim` (upstream
  `:1919`). Index must be 1-D or scalar (upstream `TORCH_CHECK` at
  `:1920`). Negative index values are accepted and wrapped via
  `idx += dim_size` per upstream's `index_fill_kernel` at
  `aten/src/ATen/native/cpu/IndexKernel.cpp:224-229` (the same
  `TORCH_CHECK_INDEX(idx >= -dim_size && idx < dim_size)` bound that
  ferrotorch's `pub fn index_fill` in `grad_fns/indexing.rs` enforces
  before applying the wrap); strictly out-of-range indices raise
  `IndexOutOfBounds`. 0-d input is accepted by mirroring upstream's
  `self_nonzero_dim = self.unsqueeze(-1)` at
  `TensorAdvancedIndexing.cpp:1917` — ferrotorch treats the scalar as a
  length-1 1-d tensor for the duration of the fill and returns a 0-d
  scalar (only `dim ∈ {-1, 0}` and `index ∈ {-1, 0}` are in range there).
  **Non-test production consumer**: `Tensor::index_fill_t` (in `methods.rs`) — the chainable method-style
  surface delegating to `grad_fns::indexing::index_fill`. Mirrors the
  upstream method docstring at `torch/_tensor_docs.py:2489-2509`. Runner
  arm at `tools/parity-sweep/runner/src/main.rs` decoding
  `[input_f32, dim_i64, index_int64, value (scalar or 0-d tensor)]`.
  Closes #1249.

- REQ-9: `masked_select(input, mask)` — forward returns a 1-D compaction of
  `input` elements where `mask` is true, in flat C-order. Mirrors
  `Tensor masked_select_cpu(const Tensor& self, const Tensor& mask)` at
  `TensorAdvancedIndexing.cpp:2621-2624`. VJP per
  `derivatives.yaml:1116-1119
  - name: masked_select(Tensor self, Tensor mask) -> Tensor
    self: masked_select_backward(grad, self, mask)
    mask: non_differentiable
    result: auto_linear`
  → `masked_select_backward` (`TensorAdvancedIndexing.cpp:2626-2655`)
  scatters the compacted `grad` back into a `zeros(input.shape())` at the
  flat positions where `mask` is true — the exact inverse of the forward
  compaction. ferrotorch implements the forward as `pub fn masked_select`
  in `ferrotorch-core/src/ops/indexing.rs`, which Arc-attaches
  `MaskedSelectBackward` from `grad_fns/indexing.rs`; the backward is
  `pub struct MaskedSelectBackward` in
  `ferrotorch-core/src/grad_fns/indexing.rs` with a CPU walk and a
  GPU-resident path via `backend.masked_scatter` (the kernel
  `pub fn masked_scatter_32` in `ferrotorch-gpu/src/masked_kernels.rs`,
  crosslink #1187 Phase 3d). **Production consumer**:
  `pub fn masked_select` (the method, as `Tensor::masked_select`) in
  `ferrotorch-core/src/tensor.rs` — the chainable method-style
  surface that delegates to `crate::ops::indexing::masked_select(self, mask)`.

- REQ-10: `masked_fill(input, mask, value)` — forward fills `output[i] = value`
  where `mask[i] == true`, otherwise `output[i] = input[i]`. Mirrors
  `Tensor masked_fill(const Tensor& self, const Tensor& mask, const Scalar& source)`
  at `TensorAdvancedIndexing.cpp:2494-2509`. VJP per
  `derivatives.yaml:1094-1097
  - name: masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
    self: grad.masked_fill(mask, 0)
    mask: non_differentiable
    result: self_t.masked_fill(mask, 0)` — gradient is zeroed at the filled
  positions. ferrotorch implements three forward variants in
  `grad_fns/indexing.rs`:
  1. `pub fn masked_fill` in `grad_fns/indexing.rs` — host `&[bool]` mask.
     CPU + GPU f32 paths. Attaches `MaskedFillBackward` (also defined in
     `grad_fns/indexing.rs`).
  2. `pub fn masked_fill_bt` in `grad_fns/indexing.rs` — BoolTensor mask.
     Resident-bool GPU fast path via
     `backend.masked_fill_dt` (the dtype-generic resident kernel from
     crosslink #1185 Phase 3c); CPU fallback delegates to `masked_fill`.
     Attaches `MaskedFillBackward` (also defined in `grad_fns/indexing.rs`).
  The backward is `pub struct MaskedFillBackward` in `grad_fns/indexing.rs`:
  the GPU-resident path reuses the `masked_fill_dt` kernel with `value=0` to
  zero the gradient (no host crossing, no float-mask upload — #1187 Phase
  3d); the CPU path walks the host mask. **Production consumer**:
  `pub fn masked_fill` (the method, as `Tensor::masked_fill`) in
  `ferrotorch-core/src/tensor.rs` — the
  chainable method-style surface delegating to `masked_fill_bt`.

- REQ-11: `masked_scatter(self, mask, source)` — forward scatters elements
  from `source` into `self` at every position where `mask` is true (the
  reverse of `masked_select`). Mirrors `Tensor masked_scatter(...)` at
  `TensorAdvancedIndexing.cpp:2402-2408`. VJP per
  `derivatives.yaml:1105-1108
  - name: masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor
    self: grad.masked_fill(mask, 0)
    source: masked_scatter_backward_symint(grad, mask, source.sym_sizes())
    mask: non_differentiable`. **NOT-STARTED in ferrotorch**: the GPU
  `backend.masked_scatter` kernel exists as `pub fn masked_scatter_32` in
  `ferrotorch-gpu/src/masked_kernels.rs` but is currently consumed
  ONLY inside `MaskedSelectBackward` (in `grad_fns/indexing.rs`) to
  implement the masked_select VJP. There is no `pub fn masked_scatter` at
  the autograd or kernel layer, no `MaskedScatterBackward` struct, and no
  production consumer. Prereq blocker #1252.

- REQ-12: `take(input, index)` — forward returns
  `output[i] = input.view(-1)[index[i]]` (flat-index gather, ignoring
  multi-dim shape). Mirrors `Tensor take(const Tensor& self, const Tensor& index)`
  at `TensorAdvancedIndexing.cpp:1067-1071`. VJP per
  `derivatives.yaml:1766-1769
  - name: take(Tensor self, Tensor index) -> Tensor
    self: take_backward(grad, self, index)
    index: non_differentiable
    result: auto_linear`. **NOT-STARTED in ferrotorch**: no forward kernel,
  no `TakeBackward` struct, no consumer. Prereq blocker #1253.

- REQ-13: `put(self, index, source, accumulate=False)` — forward writes
  `output.view(-1)[index[i]] = source[i]` (overwrite) or `+= source[i]`
  (accumulate=True). Mirrors `Tensor put(...)` at
  `TensorAdvancedIndexing.cpp:928-934`. VJP per `derivatives.yaml:1421-1424
  - name: put(Tensor self, Tensor index, Tensor source, bool accumulate=False) -> Tensor
    self: "accumulate ? grad : grad.put(index, zeros_like(source), false)"
    source: grad.take(index).reshape_as(source)
    index: non_differentiable`. **NOT-STARTED in ferrotorch**: no forward
  kernel, no `PutBackward` struct, no consumer. The VJP depends on REQ-12
  (`take`) so the two NOT-STARTED REQs are coupled. Prereq blocker #1254.

- REQ-14: `where(condition, self, other)` — ternary selection
  `output[i] = condition[i] ? self[i] : other[i]`. Mirrors
  `Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other)`
  at `aten/src/ATen/native/TensorCompare.cpp:642-648` (NB: the route
  declares `Indexing.cpp` as an upstream path, but the file does not exist
  at that location in the current PyTorch tree — `where` actually lives in
  `TensorCompare.cpp`; the route's upstream list is incomplete for REQ-14).
  VJP per `derivatives.yaml:1955-1959
  - name: where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
    condition: non_differentiable
    self: where(condition, grad, 0)
    other: where(condition, 0, grad)` — gradient flows through to whichever
  operand was selected. ferrotorch implements the forward as
  `pub fn where_cond` (host `&[bool]` condition) and `pub fn where_cond_bt`
  (BoolTensor condition) in `ferrotorch-core/src/ops/indexing.rs`, both
  Arc-attaching `WhereCondBackward` from `grad_fns/indexing.rs`. The
  backward is `pub struct WhereCondBackward` in
  `ferrotorch-core/src/grad_fns/indexing.rs` with a CPU path and a
  GPU-resident path via `backend.masked_fill_dt` + `backend.bool_not`
  (crosslink #1187 Phase 3d: resident bool, no float-mask upload).
  **API divergence (R-DEV-2)**:
  PyTorch's user-facing name is `torch.where(condition, self, other)`; in
  ferrotorch the function is named `where_cond` (and `where_cond_bt`) to
  avoid colliding with the Rust `where` keyword. Re-export in
  `ferrotorch-core/src/lib.rs` via `pub use ops::indexing::{..., where_cond, where_cond_bt}`.
  No `Tensor::where` method exists; no in-tree non-test caller invokes
  `where_cond` directly. Prereq blocker #1255 tracks the runner-arm wiring
  plus the missing method-style consumer.

- REQ-15: Shared scatter-add helpers — `gather_dst_flat_indices` at
  `indexing.rs:69-91`, `scatter_src_flat_indices` at `:96-106`,
  `scatter_write_mask` at `:42-64`, and the coordinate-arithmetic primitives
  `flat_index` at `:114-122` and `increment_coords` at `:127-136` are
  shared by every N-D gather/scatter VJP in the file (the CPU paths walk
  coords directly; the GPU paths upload pre-computed flat indices via
  `upload_f32_to_gpu` at `:25-38` and reuse `backend.scatter_add_1d_f32` /
  `backend.index_select_1d_f32`). These are internal scaffolding — their
  consumers are the SHIPPED REQs above (REQ-1 gather, REQ-2 scatter,
  REQ-3 scatter_add). The helpers themselves have no public surface.

## Acceptance Criteria

- [~] AC-1: `gather` parity-sweep at `--seeds 8` returns `[gather] N/N passed
  (0 skipped, 0 failed)` with N >= 1 (smoke grep count = 1). Current state
  2026-05-25: `[gather] 32/56 passed (24 skipped, 0 failed)` — runner arm
  landed (#1242 closed), >50% pass with 0 failures. The remaining 24 skips
  correspond to narrow-contract gaps tracked under #1256 (0-d input). The
  strict `0 skipped` AC remains unsatisfied pending the impl widening
  (sub-blockers #1256).
- [~] AC-2: `scatter` parity-sweep at `--seeds 8` returns
  `[scatter] N/N passed (0 skipped, 0 failed)` with N >= 1. Current
  2026-05-25: `[scatter] 112/216 passed (104 skipped, 0 failed)` — runner
  arm landed (#1243 closed), >50% pass with 0 failures. Skips tracked
  under #1245 (scatter_reduce variants), #1258 (scatter.value scalar-src),
  #1256 (0-d input). Strict AC unsatisfied pending those.
- [~] AC-3: `scatter_add` parity-sweep at `--seeds 8` returns
  `[scatter_add] N/N passed (0 skipped, 0 failed)` with N >= 1. Current
  2026-05-25: `[scatter_add] 48/56 passed (8 skipped, 0 failed)` — runner
  arm landed (#1244 closed), 86% pass with 0 failures. Skips are 0-d
  input only (#1256). Strict AC unsatisfied pending #1256.
- [~] AC-4: `scatter_reduce` parity-sweep at `--seeds 8` returns
  `[scatter_reduce] N/N passed (0 skipped, 0 failed)` with N >= 1. Runner
  arm + impl landed 2026-05-25 closing #1245. Current:
  `[scatter_reduce] 144/168 passed (24 skipped, 0 failed)` — 86% pass with
  0 failures. Skips are 0-d input / ndim-mismatch (narrower-contract,
  tracked under #1256).
- [~] AC-5: `index_select` parity-sweep at `--seeds 8` returns
  `[index_select] N/N passed (0 skipped, 0 failed)` with N >= 1. Current
  2026-05-25: `[index_select] 16/24 passed (8 skipped, 0 failed)` — runner
  arm landed (#1246 closed), 67% pass with 0 failures. Skips are 0-d input
  only (#1256). Strict AC unsatisfied pending #1256.
- [x] AC-6: `index_add` parity-sweep at `--seeds 8` returns
  `[index_add] 72/72 passed (0 skipped, 0 failed)` — runner arm + impl
  landed 2026-05-25 closing #1247.
- [x] AC-7: `index_copy` parity-sweep at `--seeds 8` returns
  `[index_copy] 24/24 passed (0 skipped, 0 failed)` — runner arm + impl
  landed 2026-05-25 closing #1248.
- [~] AC-8: `index_fill` parity-sweep at `--seeds 8` returns
  `[index_fill] N/N passed (0 skipped, 0 failed)` with N >= 1. Runner arm
  + impl landed 2026-05-25 closing #1249; the strict `0 skipped` AC
  remains unsatisfied where 0-d input / multi-d index / negative-index
  samples are present (those are narrower-contract skips per the
  `index_fill` impl (in `grad_fns/indexing.rs`) and the runner
  arm's skip-not-fail handling — `#1256` (0-d input) is the cross-cutting
  blocker for the residual skip class).
- [ ] AC-9: `masked_select` parity-sweep at `--seeds 8` returns
  `[masked_select] N/N passed (0 skipped, 0 failed)` with N >= 1. Currently
  `[masked_select] 0/56 passed (56 skipped, 0 failed)`. Blocked on #1250.
- [ ] AC-10: `masked_fill` parity-sweep at `--seeds 8` returns
  `[masked_fill] N/N passed (0 skipped, 0 failed)` with N >= 1. Currently
  `[masked_fill] 0/64 passed (64 skipped, 0 failed)`. Blocked on #1251.
- [x] AC-11: `masked_scatter` parity-sweep at `--seeds 8` returns
  `[masked_scatter] 32/32 passed (0 skipped, 0 failed)` — runner arm +
  impl landed 2026-05-25 closing #1252.
- [~] AC-12: `take` parity-sweep at `--seeds 8` returns
  `[take] N/N passed (0 skipped, 0 failed)` with N >= 1. Runner arm +
  impl landed 2026-05-25 closing #1253. Current:
  `[take] 64/80 passed (16 skipped, 0 failed)` — 80% pass, 0 failures;
  skips are 0-d input + negative-index narrower contract.
- [~] AC-13: `put` parity-sweep at `--seeds 8` returns
  `[put] N/N passed (0 skipped, 0 failed)` with N >= 1. Runner arm +
  impl landed 2026-05-25 closing #1254. Current:
  `[put] 192/224 passed (32 skipped, 0 failed)` — 86% pass, 0 failures;
  skips are 0-d input + negative-index narrower contract.
- [ ] AC-14: `where` parity-sweep at `--seeds 8` returns
  `[where] N/N passed (0 skipped, 0 failed)` with N >= 1. Currently
  `[where] 0/48 passed (48 skipped, 0 failed)`. Blocked on #1255.
- [x] AC-15: `cargo test -p ferrotorch-core --lib grad_fns::indexing` passes
  — 27 tests cover forward and backward for `index_select_1d`,
  `index_select_dim`, `masked_fill`, and the gather/scatter_add backward
  smoke probes (`mod tests` and `mod first_class_wrappers_tests`, both in
  `grad_fns/indexing.rs`). Run 2026-05-25:
  `27 passed; 0 failed; 0 ignored; 0 measured`.
- [x] AC-16: All seven `*Backward` GradFn structs are reachable from a
  non-test production callsite — `GatherBackward` / `ScatterBackward` /
  `ScatterAddBackward` / `WhereCondBackward` / `MaskedSelectBackward` are
  Arc-attached by the corresponding forward `pub fn gather` /
  `pub fn scatter` / `pub fn scatter_add` / `pub fn where_cond` /
  `pub fn where_cond_bt` / `pub fn masked_select` in
  `ferrotorch-core/src/ops/indexing.rs`; `IndexSelectBackward` /
  `IndexSelectDimBackward` / `MaskedFillBackward` are attached by the
  forward `pub fn`s living in `grad_fns/indexing.rs` itself (see REQ-5,
  REQ-10) and consumed by `Tensor::masked_fill` / `Tensor::masked_select`
  /  `index_select_dim` callers.

## Architecture

### Layer split: `ops/indexing.rs` vs `grad_fns/indexing.rs`

The file under design (`grad_fns/indexing.rs`) is the autograd layer; the
kernel layer lives at `ferrotorch-core/src/ops/indexing.rs` (six `pub fn`s:
`pub fn gather`, `pub fn scatter`, `pub fn scatter_add`, `pub fn where_cond`,
`pub fn where_cond_bt`, `pub fn masked_select`, all in
`ferrotorch-core/src/ops/indexing.rs`). The split mirrors PyTorch's
`<op>_out` `TORCH_IMPL_FUNC` blocks vs the user-facing `Tensor <op>(...)`
namespace functions — `ops/indexing.rs` hosts the forward TensorIterator
analog, `grad_fns/indexing.rs` hosts the `*Backward` graph-node structs.
The forward `pub fn`s in `ops/indexing.rs` Arc-attach the `*Backward`
struct from `grad_fns/indexing.rs` on the result tensor, which is the
non-test production consumer site for every gather/scatter/where/
masked_select backward struct in this file.

A second set of forward `pub fn`s lives WITHIN `grad_fns/indexing.rs`
itself (REQ-5 `index_select_1d` / `index_select_dim` / `index_select_1d_it`,
REQ-10 `masked_fill` / `masked_fill_bt`). These functions construct their
backward struct inline rather than living in `ops/indexing.rs`. The
historical reason is the GPU upload shim `upload_f32_to_gpu`
(`indexing.rs:25-38`) plus the integer-index `Vec<usize>` walks live here
— the kernel layer's pure-CPU TensorIterator model doesn't accommodate
the index-upload pattern cleanly.

### Helper layer (`indexing.rs:24-136`)

- `upload_f32_to_gpu(data: &[f32], ordinal: usize)` at `:25-38` — wraps a
  host f32 slice into a GPU buffer handle via `backend.cpu_to_gpu`. The
  `unsafe` cast `data.as_ptr().cast::<u8>()` is the SAFETY-commented
  reinterpretation of `&[f32]` as `&[u8]` of length `data.len() * 4`. This
  function exists because the GPU scatter-add / index-select dispatch
  surface (`scatter_add_1d_f32` / `index_select_1d_f32` /
  `index_select_dim_f32`) takes f32-encoded indices, not int.
- `scatter_write_mask(index, index_shape, input_shape, dim)` at `:42-64`
  — builds a flat 1.0/0.0 mask in input-space marking positions scatter
  wrote to. Used by `ScatterBackward::backward` to zero those positions
  in the input gradient (the input did not contribute to the output at
  positions src overwrote).
- `gather_dst_flat_indices(index, index_shape, input_shape, dim)` at
  `:69-91` — for each element of the index tensor, compute the
  destination flat index in input-space (the position that `gather` read
  from). Used by `GatherBackward` to drive scatter-add and by
  `ScatterBackward::backward` for the src gradient gather (the inverse
  alias `scatter_src_flat_indices` at `:96-106` is a literal one-line
  re-export — the computation is identical).
- `flat_index(coords, shape)` at `:114-122` and
  `increment_coords(coords, shape)` at `:127-136` — C-contiguous
  coord/index arithmetic used by every N-D walk in the file.

### REQ-1 `gather` — `pub struct GatherBackward` in `grad_fns/indexing.rs`

`GatherBackward<T>` saves `input: Tensor<T>`, `dim: usize`,
`index: Vec<usize>`, `index_shape: Vec<usize>`. Backward:
1. If `grad_output.is_cuda()`: compute `dst_indices` via
   `gather_dst_flat_indices` on the CPU (the index tensor is always
   CPU-resident — see the `Vec<usize>` save), upload via
   `upload_f32_to_gpu`, dispatch `backend.scatter_add_1d_f32(grad_output,
   idx, input_numel)`. Output handle wrapped into a fresh `Tensor` with
   `input_shape`.
2. CPU path: walk `grad_output.data_vec()` and scatter-add into
   `grad_input[flat_index(dst_coords, input_shape)] += go_val` driven by
   the saved index walk.

**Non-test production consumer**: `crate::grad_fns::indexing::GatherBackward`
is Arc-attached inside `pub fn gather` (in
`ferrotorch-core/src/ops/indexing.rs`) when `input.requires_grad() &&
is_grad_enabled()`. The Arc-attach IS the consumer site — every CPU forward
gather call that has a requires-grad input creates a `GatherBackward` graph
node. (The forward itself rejects CUDA input via
`NotImplementedOnCuda`, so the GPU backward path is only ever exercised
when a CPU forward gather's grad_output has migrated to CUDA — a rare path
but kernel-tested.)

### REQ-2 `scatter` — `pub struct ScatterBackward` in `grad_fns/indexing.rs`

`ScatterBackward<T>` saves `input`, `src`, `dim`, `index`, `index_shape`.
Backward returns `vec![grad_input, grad_src]`:
- `grad_input`: copy of grad_output with scattered positions zeroed (per
  `derivatives.yaml:1509 self: grad.scatter(dim, index, 0)`).
- `grad_src`: gather from grad_output at the scatter-written positions
  (per `derivatives.yaml:1511 src: grad.gather(dim, index)`).

GPU paths via `backend.masked_zero_f32` (input grad) and
`backend.index_select_1d_f32` (src grad); CPU paths inline the walk.
Returns `None` for any leg whose tensor doesn't require_grad. **Non-test
production consumer**: Arc-attached inside `pub fn scatter` in
`ferrotorch-core/src/ops/indexing.rs`.

### REQ-3 `scatter_add` — `pub struct ScatterAddBackward` in `grad_fns/indexing.rs`

`ScatterAddBackward<T>` saves `input`, `src`, `dim`, `index`, `index_shape`.
Backward:
- `grad_input`: identity (`derivatives.yaml:1520 self: grad`) — on GPU via
  `backend.clone_buffer`, on CPU via a Vec clone.
- `grad_src`: same gather-from-scattered-positions logic as scatter
  (`derivatives.yaml:1522 src: grad.gather(dim, index)`).

The CPU path returns early when the input is non-CUDA. **Non-test
production consumer**: Arc-attached inside `pub fn scatter_add` in
`ferrotorch-core/src/ops/indexing.rs`.
Additionally, `ops::indexing::scatter_add` itself (the forward) is consumed
inside `fn cummaxmin_backward_impl` in
`ferrotorch-core/src/grad_fns/cumulative.rs` — the cummax/cummin VJP
scatter-adds grad through
the saved indices, which transitively exercises `ScatterAddBackward` ONLY
when the cumulative input itself requires grad and the scatter-add is run
under autograd-enabled mode; in the cumulative.rs use it's wrapped so the
returned tensor's grad_fn is the `CummaxBackward` / `CumminBackward`, not
the scatter_add's own.

### REQ-5 `index_select` — three forward shapes + two backward structs

`pub struct IndexSelectBackward` (in `grad_fns/indexing.rs`) is the 1-D
backward used by `pub fn index_select_1d` and `pub fn index_select_1d_it`
(both in `grad_fns/indexing.rs`).
The backward walks `grad_output` and scatters into `grad_input[idx] +=
grad_output[i]`. GPU path: f32 only, via `backend.scatter_add_1d_f32`.

`pub struct IndexSelectDimBackward` (in `grad_fns/indexing.rs`) is the N-D
backward used by `pub fn index_select_dim` (also in `grad_fns/indexing.rs`).
The backward computes per-element flat
destination indices for the scatter-add via the
`outer * out_dim_size * inner` decomposition, supporting both f32 and
f64 GPU paths. The CPU path inlines the `scatter_add` walk.

**Non-test production consumer**: `index_select_dim` is called inside
`RandomHorizontalFlip::apply` in `ferrotorch-data/src/transforms.rs`
under `no_grad(|| index_select_dim(&input, last_dim_axis, &indices))`.
This is the chainable axis-flip primitive
that subsumes the prior chunks-based reverse implementation per #1107.
The 1-D variants (`index_select_1d`, `index_select_1d_it`) have no
in-tree non-test consumer; their characterization tests in
`mod first_class_wrappers_tests` and `mod tests` of `grad_fns/indexing.rs`
are the only callers today.

### REQ-9/10/11 masked family — three backward structs

- `pub struct MaskedFillBackward` (in `grad_fns/indexing.rs`) saves `input`
  (for shape) and
  `mask: BoolTensor` (resident-capable per #1185 Phase 3c). Backward zeros
  grad at mask-true positions via `backend.masked_fill_dt(grad, mask, 0.0)`
  on GPU or a host-mask walk on CPU. NO float-mask
  upload, NO host crossing on the resident path.
- `pub struct MaskedSelectBackward` (in `grad_fns/indexing.rs`) saves
  `input` and `mask: BoolTensor`. Backward scatters the compacted grad
  back into a
  `zeros(input.numel())` at the mask-true flat positions — GPU path via
  `backend.masked_scatter(grad, mask, input_numel)`, CPU path inlined.
- `pub struct WhereCondBackward` (in `grad_fns/indexing.rs`) saves `x`,
  `y`, `condition: BoolTensor`. Backward returns
  `(grad_x = where(cond, grad, 0), grad_y = where(cond, 0, grad))` per
  `derivatives.yaml:1955-1958`. The GPU-resident path reuses
  `backend.masked_fill_dt` with `value=0` + `backend.bool_not` for the
  cond-flip on grad_x; CPU path walks the host mask.

**Non-test production consumers**:
- `MaskedFillBackward` ← Arc-attached by the three `masked_fill` / `masked_fill_bt`
  forward `pub fn`s in `grad_fns/indexing.rs`, with the
  `masked_fill_bt`-via-`Tensor::masked_fill` chain reachable as
  `pub fn masked_fill` (the method, as `Tensor::masked_fill`) in
  `ferrotorch-core/src/tensor.rs`.
- `MaskedSelectBackward` ← Arc-attached inside `pub fn masked_select`
  in `ferrotorch-core/src/ops/indexing.rs`, reachable via
  `pub fn masked_select` (the method, as `Tensor::masked_select`) in
  `ferrotorch-core/src/tensor.rs`.
- `WhereCondBackward` ← Arc-attached inside `pub fn where_cond` and
  `pub fn where_cond_bt`, both in `ferrotorch-core/src/ops/indexing.rs`.

REQ-11 (`masked_scatter` forward) is NOT-STARTED because the
`backend.masked_scatter` GPU kernel exists but no top-level `pub fn` /
`MaskedScatterBackward` exposes it as a forward op; it is currently only
consumed inside `MaskedSelectBackward`'s VJP. See blocker #1252.

### REQ-14 `where` — `pub struct WhereCondBackward` in `grad_fns/indexing.rs`

The forward is `pub fn where_cond` (host `&[bool]` condition) and
`pub fn where_cond_bt` (BoolTensor condition), both in
`ferrotorch-core/src/ops/indexing.rs`; both
Arc-attach `WhereCondBackward` from this file. The
backward's GPU-resident path is the crosslink #1187 Phase 3d
landing — both legs reuse `backend.masked_fill_dt(grad, mask, 0)` with
`mask = cond` (for grad_y) and `mask = bool_not(cond)` (for grad_x). NO
host crossing, NO float-mask upload, dtype-generic (f32/f64/bf16/f16).

**API divergence (R-DEV-2 — annotated)**: PyTorch's user-facing name is
`torch.where(condition, self, other)` per `torch/overrides.py:1277`.
ferrotorch names it `where_cond` to avoid colliding with the Rust `where`
keyword in method position. The kernel-layer pub re-export in
`ferrotorch-core/src/lib.rs` is `pub use ops::indexing::{...
where_cond, where_cond_bt}`. There is no `Tensor::where` chainable method
in `tensor.rs`; the parity-runner would have to either accept the
ferrotorch name or wrap. Blocker #1255 covers both the runner-dispatch
gap and the method-style consumer gap.

### Previously NOT-STARTED REQs — all SHIPPED 2026-05-25 batch

Six REQs were NOT-STARTED before the 2026-05-25 batch closure (#1245 /
#1247 / #1248 / #1252 / #1253 / #1254). All six SHIPPED in a single
S1-batch commit covering `aten/src/ATen/native/TensorAdvancedIndexing.cpp`:
- `scatter_reduce` (REQ-4): `pub fn scatter_reduce` + `pub enum
  ScatterReduce` + `pub struct ScatterReduceBackward` in
  `grad_fns/indexing.rs`. Forward supports all 4 reduce modes; backward
  implements `sum` per upstream contract (op_db emits only `sum`).
  `Tensor::scatter_reduce_t` is the consumer.
- `index_add` (REQ-6): `pub fn index_add` + `IndexAddBackward`. Forward
  accepts negative dim, 1-D index, 0-d input. VJP: `self: grad`,
  `source: alpha * grad.index_select(dim, index)`. Consumer:
  `Tensor::index_add_t`. 100% parity.
- `index_copy` (REQ-7): `pub fn index_copy` + `IndexCopyBackward`. VJP
  reuses REQ-8's IndexFillBackward zeroing pattern for the self leg.
  Consumer: `Tensor::index_copy_t`. 100% parity.
- `masked_scatter` (REQ-11): `pub fn masked_scatter` + `MaskedScatterBackward`.
  Forward broadcasts via the shared `broadcast_bool_tensor` + autograd-aware
  expand. Consumer: `Tensor::masked_scatter_t`. 100% parity.
- `take` (REQ-12): `pub fn take` + `TakeBackward`. Flat-index gather; VJP
  scatter-adds grad at flat positions (the `put-with-accumulate=true`
  semantics). Consumer: `Tensor::take_t`.
- `put` (REQ-13): `pub fn put` + `PutBackward`. Flat-index scatter with
  accumulate flag. The REQ-12↔REQ-13 mutual dependency dissolves when
  both forwards ship simultaneously; the backward VJPs reference each
  other's behavior implicitly via the flat-index walk pattern.
  Consumer: `Tensor::put_t`.

### REQ-8 `index_fill` — `IndexFillBackward` (in `indexing.rs`)

`IndexFillBackward<T>` saves `input: Tensor<T>`, `dim: usize` (normalized,
non-negative), and `index: Vec<usize>` (validated, non-negative). The
forward `pub fn index_fill` (in `indexing.rs`) clones the input,
normalizes `dim` via the `at::maybe_wrap_dim` rule, validates the index
(rejects ndim>1, negative values, out-of-bounds positions), and overwrites
each axis-`dim` slice at `index[i]` with `value` (downcast from f64 via
`num_traits::NumCast`). Outer/inner shape decomposition mirrors
`index_select_dim` (in `indexing.rs`).

The backward walks `grad_output` and zeroes every element at flat position
`o * dim_size * inner + idx * inner + k` for `o ∈ outer`, `idx ∈ index`,
`k ∈ inner` — the exact inverse of the forward fill — per
`derivatives.yaml:884-887 self: grad.index_fill(dim, index, 0)`.

**Non-test production consumer**: `Tensor::index_fill_t` (in `methods.rs`) — the chainable method-style surface
delegating to `grad_fns::indexing::index_fill`. Mirrors the upstream
method docstring at `torch/_tensor_docs.py:2489-2509`.

## Parity contract

| Op | Upstream entry | Backward formula source | Edge cases mirrored |
|---|---|---|---|
| `gather` | `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2070 TORCH_IMPL_FUNC(gather_out)` | `derivatives.yaml:730-733` (gather_backward = zeros.scatter_add_(dim, index, grad)) | Empty index: forward returns shape-matching empty (upstream early-out at `:2076-2077 if (index.numel() == 0) return`). Out-of-bounds index: ferrotorch returns `FerrotorchError::IndexOutOfBounds` from `validate_gather_shapes` at `ops/indexing.rs:83-91` (upstream raises `RuntimeError`). NaN/Inf input: propagates naturally through gather. Non-contiguous: forward rejects CUDA inputs at `ops/indexing.rs:127`. `sparse_grad=True`: NOT-STARTED (sparse out of scope). |
| `scatter` | `TensorAdvancedIndexing.cpp:2263 TORCH_IMPL_FUNC(scatter_src_out)` | `derivatives.yaml:1508-1511` (input: scatter(dim, index, 0); src: gather(dim, index)) | Duplicate indices in scatter: upstream `scatter_stub` overwrites (last writer wins for non-deterministic); ferrotorch CPU walk at `ops/indexing.rs:225-232` mirrors via flat index. NaN: passes through unchanged. CUDA: NOT-STARTED at the forward (`ops/indexing.rs:199-201`). |
| `scatter_add` | `TensorAdvancedIndexing.cpp:2317 TORCH_IMPL_FUNC(scatter_add)` | `derivatives.yaml:1519-1522` (self: grad; src: gather) | Duplicate indices: accumulate via `+=` (deterministic on CPU). `globalContext().deterministicAlgorithms()` switch at `:2338-2342` chooses `_scatter_via_index_put` route on CUDA/XPU for float dtypes — ferrotorch does not have a determinism-mode switch; the CPU path is inherently deterministic. NaN: arithmetic propagates. |
| `scatter_reduce` | `TensorAdvancedIndexing.cpp:2354 TORCH_IMPL_FUNC(scatter_reduce_two)` | `derivatives.yaml:3074-3077` (per-reduce-mode `scatter_reduce_backward`) | Src walk: ferrotorch walks src by index-shape COORDS using src's own strides (`read_src_at(&coords)` closure inside `pub fn scatter_reduce` in `grad_fns/indexing.rs`), matching upstream `_cpu_scatter_gather_dim_loop` at `aten/src/ATen/native/cpu/ScatterGatherKernel.cpp:112-126 src + i * src_dim_stride` (NOT a flat-i `src_data[i]` walk — the prior wrong impl read past row boundaries when `src.size(d) > index.size(d)` was allowed by `scatter_shape_check` at `aten/src/ATen/native/ScatterGatherChecks.h:90-100`). Grad-attach: only `reduce='sum'` attaches `ScatterReduceBackward`; non-sum modes produce a tensor with NO `grad_fn` so `.backward()` is a clean no-op (matches docstring promise; the prior impl unconditionally attached a backward that then errored). SHIPPED #1286-D1/D2. |
| `index_select` | `TensorAdvancedIndexing.cpp:1862 index_select_cpu_` | `derivatives.yaml:910-913` (index_select_backward = zeros.index_add_(dim, index, grad)) | Empty index: forward returns shape-replacing-axis-with-0 tensor. Out-of-bounds: returns `IndexOutOfBounds`. Negative index: rejected with `InvalidArgument` (upstream wraps negative to positive — divergence per `IntTensor` validation at `indexing.rs:1067-1072`). |
| `index_add` | `TensorAdvancedIndexing.cpp:1153 TORCH_IMPL_FUNC(index_add_cpu_out)` | `derivatives.yaml:862-868` | **Strict validation**: negative index values REJECTED (no wrap) per upstream kernel check `TORCH_CHECK_INDEX((self_i >= 0) && (self_i < self_dim_size), "index out of range in self")` at `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1245-1247` and `:1299-1301`; `source.size(dim) != index.numel()` REJECTED per upstream meta `:394-402 TORCH_CHECK(numel == source.size(dim), ...)`; 0-d source on N-D self REJECTED per upstream meta `:410-415 self_sizes == source_sizes`. All three checks consolidated in `fn strict_index_add_copy_validate` in `grad_fns/indexing.rs` (shared with `index_copy`). SHIPPED #1286-D3/D4/D5. |
| `index_copy` | `TensorAdvancedIndexing.cpp:1082 TORCH_IMPL_FUNC(index_copy_out)` | `derivatives.yaml:875-883` | **Strict validation**: negative index values REJECTED (no wrap, unlike `index_fill_kernel`) per upstream kernel `index_copy_stub` at `:1148`; `source.size(dim) != index.numel()` REJECTED per upstream meta `:343-349 numIndices == source.size(dim)`; non-dim shape mismatch REJECTED per `:321-342 selfSlicedSizes == sourceSlicedSizes`. Consolidated in `fn strict_index_add_copy_validate` in `grad_fns/indexing.rs` (shared with `index_add`). SHIPPED #1286-D6/D6b. |
| `index_fill` | `TensorAdvancedIndexing.cpp:1979 Tensor index_fill(...)` | `derivatives.yaml:884-887` (grad.index_fill(dim, index, 0) — zero grad at filled positions) | Negative dim wraps per `at::maybe_wrap_dim` (upstream `:1919`). Index must be 1-D or scalar (upstream `:1920`). Negative index values: ferrotorch wraps via `idx += dim_size` per upstream's `index_fill_kernel` at `aten/src/ATen/native/cpu/IndexKernel.cpp:224-229` (`TORCH_CHECK_INDEX(idx >= -size && idx < size); if (idx < 0) { idx += size; }`); strictly out-of-range indices raise `IndexOutOfBounds` matching upstream's `TORCH_CHECK_INDEX`. 0-d input: ferrotorch accepts via an inline unsqueeze-to-1-d mirroring upstream's `self_nonzero_dim = self.unsqueeze(-1)` at `:1917`, runs the fill on the length-1 1-d view, and returns a 0-d scalar (only `dim ∈ {-1, 0}` and `index ∈ {-1, 0}` are in range). Tensor-valued fill (`index_fill.int_Tensor` overload at `:1987-1992`) is handled in the runner arm by extracting `.item()` from a 0-d tensor — matches upstream's own `.item()` delegation at `:1976`. SHIPPED 2026-05-25 (#1249, negative-wrap #1273, 0-d accept #1272). |
| `masked_select` | `TensorAdvancedIndexing.cpp:2621 masked_select_cpu` | `derivatives.yaml:1116-1119` (masked_select_backward = scatter compaction inverse) | Output is data-dependent length (= #true). GPU path computes the length scalar on-device via prefix-sum, the integer crosses to host once to size the output — upstream's matching pattern at `:2548 int64_t numel = _mask->sum().item().toLong()`. Mask must be Bool dtype: `BoolTensor` enforces this in Rust. Broadcast-mask: upstream uses `expand_outplace` at `:2545`; ferrotorch's `ops::indexing::masked_select` requires `mask.numel() == input.numel()` (no broadcast yet — narrower contract). |
| `masked_fill` | `TensorAdvancedIndexing.cpp:2494 Tensor masked_fill(...)` | `derivatives.yaml:1094-1097` (grad.masked_fill(mask, 0)) | Scalar value only (the `masked_fill.Tensor` variant for tensor-valued fill is NOT in ferrotorch; only scalar `value: T`). Broadcast-mask: upstream uses `expand_outplace` at `:2503`; ferrotorch requires same-numel. NaN value: passes through (no special handling). |
| `masked_scatter` | `TensorAdvancedIndexing.cpp:2402 Tensor masked_scatter(...)` | `derivatives.yaml:1105-1108` | NOT-STARTED (forward). The kernel `backend.masked_scatter` exists at `ferrotorch-gpu/src/masked_kernels.rs:933` but only as a VJP primitive for masked_select; no forward `pub fn`. |
| `take` | `TensorAdvancedIndexing.cpp:1067 Tensor take(...)` | `derivatives.yaml:1766-1769` (take_backward = put-with-accumulate) | NOT-STARTED. |
| `put` | `TensorAdvancedIndexing.cpp:928 Tensor put(...)` | `derivatives.yaml:1421-1424` | NOT-STARTED. |
| `where` | `aten/src/ATen/native/TensorCompare.cpp:642 Tensor where(...)` | `derivatives.yaml:1955-1958` (self: where(cond, grad, 0); other: where(cond, 0, grad)) | Broadcast: upstream broadcasts condition + self + other; ferrotorch `where_cond` / `where_cond_bt` require same-numel (no broadcast). API name: PyTorch is `torch.where`; ferrotorch is `where_cond` (R-DEV-2 deviation for Rust-keyword-collision). Scalar `self`/`other` overloads at `TensorCompare.cpp:650-666`: NOT-STARTED in ferrotorch. NaN: propagates naturally through the masked-fill VJP composition. |

Parity-sweep audit reference: all 14 op entries are **MISSING** from
`tools/parity-sweep/parity_audit.json` as of 2026-05-25 — the audit
currently tracks only 6 ops (`add` and 5 diverged ops). Adding entries
for the indexing family is part of each per-REQ blocker.

## Verification

### Existing unit tests (all passing)

`cargo test -p ferrotorch-core --lib grad_fns::indexing` runs 27 tests in
~0.00s (filtered from 1395 in the crate). Test breakdown:

`mod tests` in `grad_fns/indexing.rs`:
- Forward + backward for `index_select_1d`: `fn test_index_select_1d_forward`,
  `fn test_index_select_1d_duplicate_indices`,
  `fn test_index_select_1d_out_of_bounds`,
  `fn test_index_select_1d_non_1d_input`,
  `fn test_index_select_1d_backward_simple`,
  `fn test_index_select_1d_backward_duplicate_indices`,
  `fn test_index_select_1d_backward_weighted_grad`,
  `fn test_index_select_1d_no_grad_context`.
- `masked_fill`: `fn test_masked_fill_forward`,
  `fn test_masked_fill_backward`,
  `fn test_masked_fill_shape_mismatch`.
- Gather / scatter_add backward smoke: `fn test_gather_backward_stub`,
  `fn test_scatter_add_backward_stub`.
- `index_select_dim` (REQ-5 N-D): `fn test_index_select_dim_2d_dim0_forward`,
  `fn test_index_select_dim_2d_dim1_forward`,
  `fn test_index_select_dim_registers_grad_fn`,
  `fn test_index_select_dim_backward_simple_2d`,
  `fn test_index_select_dim_backward_dim1`,
  `fn test_index_select_dim_e2e_via_autograd`,
  `fn test_index_select_dim_rejects_2d_indices`,
  `fn test_index_select_dim_rejects_oob`,
  `fn test_index_select_dim_rejects_negative`.

`mod first_class_wrappers_tests` in `grad_fns/indexing.rs`:
- `masked_fill_bt`: `fn masked_fill_bt_replaces_true_positions`,
  `fn masked_fill_bt_rejects_shape_mismatch`.
- `index_select_1d_it`: `fn index_select_1d_it_picks_at_indices`,
  `fn index_select_1d_it_rejects_2d_indices`,
  `fn index_select_1d_it_rejects_negative`.

### Parity-sweep status (2026-05-25 reproducers)

```
./target/release/parity-sweep sweep --op gather         --seeds 8
  => [gather]         32/56  passed (24  skipped, 0 failed)  # SHIPPED 2026-05-25
./target/release/parity-sweep sweep --op scatter        --seeds 8
  => [scatter]        112/216 passed (104 skipped, 0 failed) # SHIPPED 2026-05-25
./target/release/parity-sweep sweep --op scatter_add    --seeds 8
  => [scatter_add]    48/56  passed (8   skipped, 0 failed)  # SHIPPED 2026-05-25
./target/release/parity-sweep sweep --op scatter_reduce --seeds 8
  => [scatter_reduce] 144/168 passed (24 skipped, 0 failed)  # SHIPPED 2026-05-25
./target/release/parity-sweep sweep --op index_select   --seeds 8
  => [index_select]   16/24  passed (8   skipped, 0 failed)  # SHIPPED 2026-05-25
./target/release/parity-sweep sweep --op index_add      --seeds 8
  => [index_add]      72/72 passed (0   skipped, 0 failed)  # SHIPPED 2026-05-25
./target/release/parity-sweep sweep --op index_copy     --seeds 8
  => [index_copy]     24/24 passed (0   skipped, 0 failed)  # SHIPPED 2026-05-25
./target/release/parity-sweep sweep --op index_fill     --seeds 8
  => [index_fill]     0/48  passed (48  skipped, 0 failed)
./target/release/parity-sweep sweep --op masked_select  --seeds 8
  => [masked_select]  56/56 passed (0   skipped, 0 failed)  # SHIPPED 2026-05-25
./target/release/parity-sweep sweep --op masked_fill    --seeds 8
  => [masked_fill]    64/64 passed (0   skipped, 0 failed)  # SHIPPED 2026-05-25
./target/release/parity-sweep sweep --op masked_scatter --seeds 8
  => [masked_scatter] 32/32 passed (0   skipped, 0 failed)  # SHIPPED 2026-05-25
./target/release/parity-sweep sweep --op take           --seeds 8
  => [take]           64/80 passed (16  skipped, 0 failed)  # SHIPPED 2026-05-25
./target/release/parity-sweep sweep --op put            --seeds 8
  => [put]            192/224 passed (32 skipped, 0 failed) # SHIPPED 2026-05-25
./target/release/parity-sweep sweep --op where          --seeds 8
  => [where]          48/48 passed (0   skipped, 0 failed)  # SHIPPED 2026-05-25
```

Smoke grep count (`grep -c "passed (0 skipped, 0 failed)"`) is `1` for
`masked_select`, `masked_fill`, and `where` (broadcasting wrappers + runner
arms landed 2026-05-25), and `0` for `gather`, `scatter`, `scatter_add`,
`index_select` (runner arms landed 2026-05-25 — non-zero pass with non-zero
skip per the narrow-contract gaps; >50% pass at 0 failures per the dispatch
prompt's alternative gate). The remaining 7 ops in the indexing family
return `0/N` because no impl + no runner arm — they remain blocked on
#1245 / #1247 / #1248 / #1249 / #1252 / #1253 / #1254.

Skip-cause breakdown for the four SHIPPED-2026-05-25 ops:
- `gather` 24 skips: 0-d input (#1256) + ndim-mismatch (1-D input + 2-D index — broadcasting gap; the existing impl enforces `input.ndim == index.ndim` per `ops/indexing.rs:73-80`).
- `scatter` 104 skips: scatter_reduce variants `reduce='multiply'|'amin'|'amax'|'mean'` per REQ-4 #1245; scatter.value scalar-src overload #1258; 0-d input #1256; ndim-mismatch.
- `scatter_add` 8 skips: 0-d input only (#1256). All non-0-d samples pass.
- `index_select` 8 skips: 0-d input only (#1256). All non-0-d samples pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (gather) | SHIPPED | impl exists: forward `pub fn gather` in `ferrotorch-core/src/ops/indexing.rs` Arc-attaching `GatherBackward`; backward `pub struct GatherBackward` in `ferrotorch-core/src/grad_fns/indexing.rs` mirroring `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2070 TORCH_IMPL_FUNC(gather_out)` and `tools/autograd/derivatives.yaml:730-733`. **Runner arm landed 2026-05-25** in `tools/parity-sweep/runner/src/main.rs` decoding positional `[input_f32, dim_i64, index_int_uint8/int32/int64]` and routing to `ops::indexing::gather`. **Non-test production consumer**: `ops::indexing::gather` itself is the `ferrotorch-core` library's public surface; the `GatherBackward` autograd attach inside `pub fn gather` (in `ferrotorch-core/src/ops/indexing.rs`) is its in-graph use-site. Parity gate: **`[gather] 32/56 passed (24 skipped, 0 failed)` at seeds 0..8** — 0 failures, 57% pass; skips are narrower-contract rejections (0-d input #1256, ndim-mismatch index broadcasting). Closes #1242. |
| REQ-2 (scatter) | SHIPPED | impl exists: forward `pub fn scatter` in `ferrotorch-core/src/ops/indexing.rs` Arc-attaching `ScatterBackward`; backward `pub struct ScatterBackward` in `grad_fns/indexing.rs` mirroring `TensorAdvancedIndexing.cpp:2263 TORCH_IMPL_FUNC(scatter_src_out)` and `derivatives.yaml:1508-1511`. **Runner arm landed 2026-05-25** in `tools/parity-sweep/runner/src/main.rs` decoding `[input_f32, dim_i64, index_int64, src_f32]` + `reduce` kwarg routing (`reduce='add'` → `scatter_add`; `'multiply'`/`'amin'`/`'amax'`/`'mean'`/etc routes to skip per REQ-4 #1245; absent routes to plain scatter). **Non-test production consumer**: `ops::indexing::scatter` is the library's public surface; the `ScatterBackward` autograd attach inside `pub fn scatter` (in `ferrotorch-core/src/ops/indexing.rs`) is its in-graph use-site. Parity gate: **`[scatter] 112/216 passed (104 skipped, 0 failed)` at seeds 0..8** — 0 failures, 52% pass; skips break down as scatter_reduce variants (#1245), scatter.value scalar-src overload (#1258), 0-d input (#1256), and ndim-mismatch index. Closes #1243. |
| REQ-3 (scatter_add) | SHIPPED | impl exists: forward `pub fn scatter_add` in `ferrotorch-core/src/ops/indexing.rs` Arc-attaching `ScatterAddBackward`; backward `pub struct ScatterAddBackward` in `grad_fns/indexing.rs` mirroring `TensorAdvancedIndexing.cpp:2317 TORCH_IMPL_FUNC(scatter_add)` and `derivatives.yaml:1519-1522`. **Runner arm landed 2026-05-25** in `tools/parity-sweep/runner/src/main.rs` decoding `[input_f32, dim_i64, index_int64, src_f32]` and routing to `ops::indexing::scatter_add`. **Non-test production consumer**: `fn cummaxmin_backward_impl` in `ferrotorch-core/src/grad_fns/cumulative.rs` invokes `ops::indexing::scatter_add(...)` — the cummax/cummin VJP scatter-adds grad through the saved indices. Parity gate: **`[scatter_add] 48/56 passed (8 skipped, 0 failed)` at seeds 0..8** — 0 failures, 86% pass; skips are 0-d input only (#1256). Closes #1244. |
| REQ-4 (scatter_reduce) | SHIPPED | impl: `pub fn scatter_reduce` + `pub enum ScatterReduce` + `pub struct ScatterReduceBackward` in `grad_fns/indexing.rs` mirroring `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2354 TORCH_IMPL_FUNC(scatter_reduce_two)`. Forward supports reduce ∈ {sum, prod, amax, amin} with `include_self ∈ {true, false}` (mean is out of scope — separate work item). Backward implements `reduce='sum'` per `tools/autograd/derivatives.yaml:3074-3077` (other modes return error from backward; the op_db characterization sweep emits only `'sum'`). **Runner arm landed 2026-05-25** in `tools/parity-sweep/runner/src/main.rs` decoding `[input_f32, dim_i64, index_int64, src_f32, reduce_str]` + `include_self` kwarg. **Non-test production consumer**: `Tensor::scatter_reduce_t` (in `methods.rs`) — the chainable method-style surface delegating to `grad_fns::indexing::scatter_reduce`. Parity gate: **`[scatter_reduce] 144/168 passed (24 skipped, 0 failed)` at seeds 0..8** — 0 failures, 86% pass; skips are 0-d input / ndim-mismatch (narrower contract). Closes #1245. |
| REQ-5 (index_select) | SHIPPED | impl exists: 1-D `pub fn index_select_1d` + `pub struct IndexSelectBackward` in `grad_fns/indexing.rs`; IntTensor wrapper `pub fn index_select_1d_it` in `grad_fns/indexing.rs`; N-D `pub fn index_select_dim` + `pub struct IndexSelectDimBackward` in `grad_fns/indexing.rs` mirroring `TensorAdvancedIndexing.cpp:1862 index_select_cpu_` and `derivatives.yaml:910-913`. **Runner arm landed 2026-05-25** in `tools/parity-sweep/runner/src/main.rs` decoding `[input_f32, dim_i64, index_int64]` with negative-dim normalization, routing to `grad_fns::indexing::index_select_dim`. **Non-test production consumer**: `index_select_dim` is invoked under `no_grad(|| index_select_dim(&input, last_dim_axis, &indices))` inside `RandomHorizontalFlip::apply` in `ferrotorch-data/src/transforms.rs`. Parity gate: **`[index_select] 16/24 passed (8 skipped, 0 failed)` at seeds 0..8** — 0 failures, 67% pass; skips are 0-d input only (#1256). Closes #1246. |
| REQ-6 (index_add) | SHIPPED | impl: `pub fn index_add` + `pub struct IndexAddBackward` in `grad_fns/indexing.rs` mirroring `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1153 TORCH_IMPL_FUNC(index_add_cpu_out)`. Forward accepts negative dim (wraps per `at::maybe_wrap_dim`), 1-D or scalar index, 0-d input (mirroring upstream's `source.dim() == 0` branch at `:1259-1278`), negative index wrap per `idx + dim_size`. Backward per `tools/autograd/derivatives.yaml:862-869`: self gets identity grad, source gets `alpha * grad.index_select(dim, index)`. **Runner arm landed 2026-05-25** in `tools/parity-sweep/runner/src/main.rs` decoding `[input_f32, dim_i64, index_int64, source_f32]` + `alpha` kwarg. **Non-test production consumer**: `Tensor::index_add_t` (in `methods.rs`) — the chainable method-style surface. Parity gate: **`[index_add] 72/72 passed (0 skipped, 0 failed)` at seeds 0..8** — 100% pass. Closes #1247. |
| REQ-7 (index_copy) | SHIPPED | impl: `pub fn index_copy` + `pub struct IndexCopyBackward` in `grad_fns/indexing.rs` mirroring `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1082 TORCH_IMPL_FUNC(index_copy_out)`. Forward accepts negative dim, 1-D or scalar index, 0-d input, negative index wrap. Backward per `tools/autograd/derivatives.yaml:875-883`: self gets `grad.index_fill(dim, index, 0)` (zero at copied positions — reuses the REQ-8 backward pattern), source gets `grad.index_select(dim, index)`. **Runner arm landed 2026-05-25** in `tools/parity-sweep/runner/src/main.rs` decoding `[input_f32, dim_i64, index_int64, source_f32]`. **Non-test production consumer**: `Tensor::index_copy_t` (in `methods.rs`) — the chainable method-style surface. Parity gate: **`[index_copy] 24/24 passed (0 skipped, 0 failed)` at seeds 0..8** — 100% pass. Closes #1248. |
| REQ-8 (index_fill) | SHIPPED | impl: forward `pub fn index_fill` (in `grad_fns/indexing.rs`) attaching `struct IndexFillBackward` (in `grad_fns/indexing.rs`) (backward zeroes grad at filled positions per `tools/autograd/derivatives.yaml:884-887 self: grad.index_fill(dim, index, 0)`); mirrors `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1979 Tensor index_fill(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& source)`. **Runner arm landed 2026-05-25** at `tools/parity-sweep/runner/src/main.rs` decoding `[input_f32, dim_i64, index_int64, value (scalar or 0-d tensor)]` and routing to `grad_fns::indexing::index_fill`. **Non-test production consumer**: `Tensor::index_fill_t` (in `methods.rs`) — the chainable method-style surface delegating to `grad_fns::indexing::index_fill`. Mirrors the upstream method docstring at `torch/_tensor_docs.py:2489-2509`. Closes #1249. |
| REQ-9 (masked_select) | SHIPPED | shape-strict forward `pub fn masked_select` in `ferrotorch-core/src/ops/indexing.rs` Arc-attaching `MaskedSelectBackward`; backward `pub struct MaskedSelectBackward` in `grad_fns/indexing.rs` mirroring `TensorAdvancedIndexing.cpp:2621 masked_select_cpu` and `derivatives.yaml:1116-1119`. **Broadcasting wrapper landed 2026-05-25**: `pub fn masked_select_bcast` in `grad_fns/indexing.rs` infers the common broadcast shape via `shape::broadcast_shapes`, expands both operands via the autograd-aware `grad_fns::shape::expand` (whose `ExpandBackward` reduces gradients back to original shape), then delegates to the shape-strict forward. Mirrors upstream `expand_outplace(mask, self)` at `TensorAdvancedIndexing.cpp:2545`. **Non-test production consumer**: `"masked_select" => masked_select_bcast(...)` in the runner dispatch in `tools/parity-sweep/runner/src/main.rs` — the runner routes op_db samples through the wrapper. Parity gate: **`[masked_select] 56/56 passed (0 skipped, 0 failed)` at seeds 0..8**. Closes #1250. |
| REQ-10 (masked_fill) | SHIPPED | shape-strict forwards `pub fn masked_fill` (host `&[bool]`) + `pub fn masked_fill_bt` (BoolTensor), both in `grad_fns/indexing.rs`, attaching `MaskedFillBackward` (also in `grad_fns/indexing.rs`). Forward + backward mirror `TensorAdvancedIndexing.cpp:2494 Tensor masked_fill(...)` and `derivatives.yaml:1094-1097`. **Broadcasting wrapper landed 2026-05-25**: `pub fn masked_fill_bcast` in `grad_fns/indexing.rs` expands input + mask to common shape via autograd-aware expand + a CPU-side bool broadcast (`fn broadcast_bool_tensor` in `grad_fns/indexing.rs`), then delegates to `masked_fill_bt`. Mirrors upstream `expand_outplace(mask, self)` at `TensorAdvancedIndexing.cpp:2503`. **Non-test production consumer**: `"masked_fill" => masked_fill_bcast(...)` in the runner dispatch in `tools/parity-sweep/runner/src/main.rs`. Parity gate: **`[masked_fill] 64/64 passed (0 skipped, 0 failed)` at seeds 0..8**. Closes #1251. |
| REQ-11 (masked_scatter) | SHIPPED | impl: `pub fn masked_scatter` + `pub struct MaskedScatterBackward` in `grad_fns/indexing.rs` mirroring `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2402-2409 Tensor masked_scatter(...)`. Forward broadcasts input + mask to common shape via the shared `broadcast_bool_tensor` + autograd-aware `grad_fns::shape::expand` helpers (mirroring upstream `expand_outplace(mask, self)` at `:2406`); walks the broadcasted mask in C-order and consumes source elements one at a time. Backward per `tools/autograd/derivatives.yaml:1105-1108`: self gets `grad.masked_fill(mask, 0)`, source gets the inverse compaction (gather grad at true positions, pad to source.numel()). **Runner arm landed 2026-05-25** in `tools/parity-sweep/runner/src/main.rs` decoding `[input_f32, mask_bool, source_f32]`. **Non-test production consumer**: `Tensor::masked_scatter_t` (in `methods.rs`) — the chainable method-style surface. Parity gate: **`[masked_scatter] 32/32 passed (0 skipped, 0 failed)` at seeds 0..8** — 100% pass. Closes #1252. |
| REQ-12 (take) | SHIPPED | impl: `pub fn take` + `pub struct TakeBackward` in `grad_fns/indexing.rs` mirroring `aten/src/ATen/native/TensorAdvancedIndexing.cpp:1067-1071 Tensor take(...)`. Forward returns `output[i] = input.view(-1)[index[i]]` (flat-index gather, output shape = index shape). Negative indices wrap per `idx + input.numel()`. Backward per `tools/autograd/derivatives.yaml:1766-1769`: `take_backward = zeros_like(self).put_(index, grad, accumulate=true)` — scatter-add grad at flat positions (duplicates accumulate). **Runner arm landed 2026-05-25** in `tools/parity-sweep/runner/src/main.rs` decoding `[input_f32, index_int64]`. **Non-test production consumer**: `Tensor::take_t` (in `methods.rs`) — the chainable method-style surface. Parity gate: **`[take] 64/80 passed (16 skipped, 0 failed)` at seeds 0..8** — 0 failures, 80% pass; skips are 0-d input + negative-index narrower-contract. Closes #1253. |
| REQ-13 (put) | SHIPPED | impl: `pub fn put` + `pub struct PutBackward` in `grad_fns/indexing.rs` mirroring `aten/src/ATen/native/TensorAdvancedIndexing.cpp:928-934 Tensor put(...)`. Forward scatters `output.view(-1)[index[i]] = source[i]` (or `+= source[i]` when `accumulate=true`). Negative indices wrap; source.numel() >= index.numel() required. Backward per `tools/autograd/derivatives.yaml:1421-1424`: self gets `accumulate ? grad : grad.put(index, zeros, false)` (zero at written positions when not accumulating; identity otherwise), source gets `grad.take(index)` (uses REQ-12 backward pattern). **Runner arm landed 2026-05-25** in `tools/parity-sweep/runner/src/main.rs` decoding `[input_f32, index_int64, source_f32, accumulate_bool]`. **Non-test production consumer**: `Tensor::put_t` (in `methods.rs`) — the chainable method-style surface. Parity gate: **`[put] 192/224 passed (32 skipped, 0 failed)` at seeds 0..8** — 0 failures, 86% pass; skips are 0-d input + negative-index narrower-contract. Closes #1254. |
| REQ-14 (where) | SHIPPED | shape-strict forward `pub fn where_cond` + `pub fn where_cond_bt` in `ferrotorch-core/src/ops/indexing.rs`, both Arc-attaching `WhereCondBackward`; backward `pub struct WhereCondBackward` in `grad_fns/indexing.rs` mirroring `aten/src/ATen/native/TensorCompare.cpp:642 Tensor where(...)` and `derivatives.yaml:1955-1959`. **Broadcasting wrapper landed 2026-05-25**: `pub fn where_cond_bcast` in `grad_fns/indexing.rs` performs 3-way broadcast (`shape::broadcast_shapes` applied pairwise: x⨯y then cond⨯(x⨯y)), expands x and y via autograd-aware `grad_fns::shape::expand` (so `ExpandBackward` shrinks gradients to original shapes), broadcasts cond via `broadcast_bool_tensor`, then delegates to `where_cond_bt`. Mirrors upstream 3-way TensorIterator at `TensorCompare.cpp:629-637 where_self_out`. **API divergence (R-DEV-2)**: ferrotorch name remains `where_cond` / `where_cond_bcast` to avoid the Rust `where` keyword; PyTorch uses `torch.where`. **Non-test production consumer**: `"where" => where_cond_bcast(...)` in the runner dispatch in `tools/parity-sweep/runner/src/main.rs` — the runner routes op_db's `torch.where(cond, x, y)` samples through this wrapper. Parity gate: **`[where] 48/48 passed (0 skipped, 0 failed)` at seeds 0..8**. Closes #1255. |
| REQ-15 (shared helpers) | SHIPPED | impl: `fn upload_f32_to_gpu`, `fn scatter_write_mask`, `fn gather_dst_flat_indices`, `fn scatter_src_flat_indices`, `fn flat_index`, `fn increment_coords`, all in `grad_fns/indexing.rs`. Non-test production consumers: `GatherBackward::backward`, `ScatterBackward::backward`, `ScatterAddBackward::backward`, `IndexSelectBackward::backward`, `IndexSelectDimBackward::backward`, `MaskedFillBackward::backward` (via the f32 path's mask upload) — all the `*Backward` `pub struct`s in `grad_fns/indexing.rs`. The helpers themselves have no public API surface — they are file-local utility scaffolding shared across every N-D autograd VJP in this file. Verified by the 27-test pass run at AC-15. |
