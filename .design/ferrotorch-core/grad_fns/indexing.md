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
2026-05-25 the masked / where family (`masked_select`, `masked_fill`, `where`)
has SHIPPED with broadcasting wrappers + runner dispatch (zero skips), and
the indexing family (`gather`, `scatter`, `scatter_add`, `index_select`) has
SHIPPED with runner dispatch routed through the existing shape-strict impls
— >50% pass with 0 failures, remaining skips tracked under narrow-contract
sub-blockers (#1256 for 0-d input across the indexing family; #1245 for
scatter_reduce variants; #1258 for scatter.value scalar-src). The remaining
7 op_db entries (`scatter_reduce`, `index_add`, `index_copy`, `index_fill`,
`masked_scatter` forward, `take`, `put`) return 0 passes because there is
no ferrotorch impl at all. Each remaining gap is filed as a NOT-STARTED
REQ with a concrete prereq blocker (#1245, #1247–#1254).

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
  the forward at `ferrotorch-core/src/ops/indexing.rs:112 pub fn gather`
  attaching `GatherBackward` at `:155`; the backward struct is defined at
  `ferrotorch-core/src/grad_fns/indexing.rs:444-520` with the CPU
  scatter-add walk at `:456-510` and a GPU-resident path at `:467-486` that
  uses `backend.scatter_add_1d_f32`. **Divergence (sparse_grad kwarg)**: the
  `sparse_grad=True` branch (`TensorAdvancedIndexing.cpp:2093-2095
  return at::_gather_sparse_backward(self, dim, index, grad)`) has no
  ferrotorch analog — sparse tensors are out of scope for ferrotorch-core
  per the goal.md's stated scope. **Divergence (CUDA forward)**: only
  f32 GPU path lives in the backward — the forward at
  `ops/indexing.rs:127` rejects CUDA inputs outright via
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
  ferrotorch implements the forward at
  `ferrotorch-core/src/ops/indexing.rs:183 pub fn scatter` attaching
  `ScatterBackward` at `:235`; the backward struct lives at
  `ferrotorch-core/src/grad_fns/indexing.rs:534-658` with CPU paths at
  `:602-647` (input: copy grad_output then zero scattered positions; src:
  gather from scattered positions) and a GPU-resident path at `:562-601`
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
  forward at `ferrotorch-core/src/ops/indexing.rs:259 pub fn scatter_add`
  attaching `ScatterAddBackward` at `:311`; the backward struct lives at
  `ferrotorch-core/src/grad_fns/indexing.rs:672-788` with CPU paths at
  `:740-777` and a GPU-resident path at `:699-734` via
  `backend.clone_buffer` (identity for input grad) and
  `backend.index_select_1d_f32` (gather for src grad). **Production
  consumer**: `crate::grad_fns::cumulative::cummaxmin_backward_impl`
  invokes `ops::indexing::scatter_add` at
  `ferrotorch-core/src/grad_fns/cumulative.rs:503` for the cummax/cummin
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
  1. `index_select_1d<T>(input: &Tensor<T>, indices: &[usize])` at
     `indexing.rs:212` — host `&[usize]` indices, 1-D input only, attaches
     `IndexSelectBackward` at `:254`/`:268`. Backward at `:157-205` is a
     1-D scatter-add VJP with CPU and GPU paths.
  2. `index_select_1d_it<T, I>(input, indices: &IntTensor<I>)` at
     `indexing.rs:1053` — IntTensor-typed indices, 1-D input only, widens
     to `Vec<usize>` and forwards to `index_select_1d`.
  3. `index_select_dim<T, I>(input, dim, indices: &IntTensor<I>)` at
     `indexing.rs:1229` — N-D input, arbitrary axis, IntTensor indices.
     Attaches `IndexSelectDimBackward` at `:1325`/`:1357`. Backward at
     `:1101-1212` is the N-D scatter-add VJP that subsumes the 1-D case
     for ndim>=2 (per #1014). CPU + GPU f32/f64 fast paths. **Production
     consumer**: `ferrotorch-data/src/transforms.rs:389
     no_grad(|| index_select_dim(&input, last_dim_axis, &indices))` inside
     `HorizontalFlip::apply` (`transforms.rs:380-390`) — the chainable
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
  filled positions. **NOT-STARTED in ferrotorch**: no forward kernel, no
  `IndexFillBackward` struct, no consumer. Prereq blocker #1249.

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
  compaction. ferrotorch implements the forward at
  `ferrotorch-core/src/ops/indexing.rs:478 pub fn masked_select` attaching
  `MaskedSelectBackward` at `:509`; the backward struct lives at
  `ferrotorch-core/src/grad_fns/indexing.rs:923-987` with CPU walk at
  `:963-977` and a GPU-resident path at `:944-960` via
  `backend.masked_scatter` (the kernel from `ferrotorch-gpu/src/masked_kernels.rs:933`,
  crosslink #1187 Phase 3d). **Production consumer**: `Tensor::masked_select`
  at `ferrotorch-core/src/tensor.rs:1142-1147` — the chainable method-style
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
  1. `masked_fill<T>(input, mask: &[bool], value)` at `indexing.rs:367` —
     host `&[bool]` mask. CPU + GPU f32 paths. Attaches
     `MaskedFillBackward` at `:405`/`:423`.
  2. `masked_fill_bt<T>(input, mask: &BoolTensor, value)` at `indexing.rs:997`
     — BoolTensor mask. Resident-bool GPU fast path at `:1017-1042` via
     `backend.masked_fill_dt` (the dtype-generic resident kernel from
     crosslink #1185 Phase 3c); CPU fallback delegates to `masked_fill`.
     Attaches `MaskedFillBackward` at `:1035`.
  The backward struct lives at `indexing.rs:295-358`: GPU-resident path at
  `:314-329` reuses the resident `masked_fill_dt` kernel with `value=0` to
  zero the gradient (no host crossing, no float-mask upload — #1187 Phase
  3d); CPU path at `:336-348` walks the host mask. **Production consumer**:
  `Tensor::masked_fill` at `ferrotorch-core/src/tensor.rs:1126-1132` — the
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
  `backend.masked_scatter` kernel exists at
  `ferrotorch-gpu/src/masked_kernels.rs:933` but is currently consumed
  ONLY inside `MaskedSelectBackward` (`grad_fns/indexing.rs:952`) to
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
  operand was selected. ferrotorch implements the forward at
  `ferrotorch-core/src/ops/indexing.rs:334 pub fn where_cond` (host
  `&[bool]` condition) and `:397 pub fn where_cond_bt` (BoolTensor
  condition), attaching `WhereCondBackward` at `:378` / `:445`. The backward
  struct lives at `ferrotorch-core/src/grad_fns/indexing.rs:800-904` with
  CPU paths at `:868-893` and a GPU-resident path at `:825-862` via
  `backend.masked_fill_dt` + `backend.bool_not` (crosslink #1187 Phase 3d:
  resident bool, no float-mask upload). **API divergence (R-DEV-2)**:
  PyTorch's user-facing name is `torch.where(condition, self, other)`; in
  ferrotorch the function is named `where_cond` (and `where_cond_bt`) to
  avoid colliding with the Rust `where` keyword. Re-export at
  `ferrotorch-core/src/lib.rs:174 pub use ops::indexing::{..., where_cond, where_cond_bt}`.
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
- [ ] AC-4: `scatter_reduce` parity-sweep at `--seeds 8` returns
  `[scatter_reduce] N/N passed (0 skipped, 0 failed)` with N >= 1.
  Currently `[scatter_reduce] 0/168 passed (168 skipped, 0 failed)`.
  Blocked on #1245.
- [~] AC-5: `index_select` parity-sweep at `--seeds 8` returns
  `[index_select] N/N passed (0 skipped, 0 failed)` with N >= 1. Current
  2026-05-25: `[index_select] 16/24 passed (8 skipped, 0 failed)` — runner
  arm landed (#1246 closed), 67% pass with 0 failures. Skips are 0-d input
  only (#1256). Strict AC unsatisfied pending #1256.
- [ ] AC-6: `index_add` parity-sweep at `--seeds 8` returns
  `[index_add] N/N passed (0 skipped, 0 failed)` with N >= 1. Currently
  `[index_add] 0/72 passed (72 skipped, 0 failed)`. Blocked on #1247.
- [ ] AC-7: `index_copy` parity-sweep at `--seeds 8` returns
  `[index_copy] N/N passed (0 skipped, 0 failed)` with N >= 1. Currently
  `[index_copy] 0/24 passed (24 skipped, 0 failed)`. Blocked on #1248.
- [ ] AC-8: `index_fill` parity-sweep at `--seeds 8` returns
  `[index_fill] N/N passed (0 skipped, 0 failed)` with N >= 1. Currently
  `[index_fill] 0/48 passed (48 skipped, 0 failed)`. Blocked on #1249.
- [ ] AC-9: `masked_select` parity-sweep at `--seeds 8` returns
  `[masked_select] N/N passed (0 skipped, 0 failed)` with N >= 1. Currently
  `[masked_select] 0/56 passed (56 skipped, 0 failed)`. Blocked on #1250.
- [ ] AC-10: `masked_fill` parity-sweep at `--seeds 8` returns
  `[masked_fill] N/N passed (0 skipped, 0 failed)` with N >= 1. Currently
  `[masked_fill] 0/64 passed (64 skipped, 0 failed)`. Blocked on #1251.
- [ ] AC-11: `masked_scatter` parity-sweep at `--seeds 8` returns
  `[masked_scatter] N/N passed (0 skipped, 0 failed)` with N >= 1.
  Currently `[masked_scatter] 0/32 passed (32 skipped, 0 failed)`.
  Blocked on #1252.
- [ ] AC-12: `take` parity-sweep at `--seeds 8` returns
  `[take] N/N passed (0 skipped, 0 failed)` with N >= 1. Currently
  `[take] 0/80 passed (80 skipped, 0 failed)`. Blocked on #1253.
- [ ] AC-13: `put` parity-sweep at `--seeds 8` returns
  `[put] N/N passed (0 skipped, 0 failed)` with N >= 1. Currently
  `[put] 0/224 passed (224 skipped, 0 failed)`. Blocked on #1254.
- [ ] AC-14: `where` parity-sweep at `--seeds 8` returns
  `[where] N/N passed (0 skipped, 0 failed)` with N >= 1. Currently
  `[where] 0/48 passed (48 skipped, 0 failed)`. Blocked on #1255.
- [x] AC-15: `cargo test -p ferrotorch-core --lib grad_fns::indexing` passes
  — 27 tests cover forward and backward for `index_select_1d`,
  `index_select_dim`, `masked_fill`, and the gather/scatter_add backward
  smoke probes (`grad_fns::indexing::tests` mod at `indexing.rs:1428-1968`
  and `first_class_wrappers_tests` at `:1368-1422`). Run 2026-05-25:
  `27 passed; 0 failed; 0 ignored; 0 measured`.
- [x] AC-16: All seven `*Backward` GradFn structs are reachable from a
  non-test production callsite — `GatherBackward` / `ScatterBackward` /
  `ScatterAddBackward` / `WhereCondBackward` / `MaskedSelectBackward` are
  attached at `ferrotorch-core/src/ops/indexing.rs:155, :235, :311, :378,
  :509` by the corresponding forward `pub fn`s; `IndexSelectBackward` /
  `IndexSelectDimBackward` / `MaskedFillBackward` are attached by the
  forward `pub fn`s living in `grad_fns/indexing.rs` itself (see REQ-5,
  REQ-10) and consumed by `Tensor::masked_fill` / `Tensor::masked_select`
  /  `index_select_dim` callers.

## Architecture

### Layer split: `ops/indexing.rs` vs `grad_fns/indexing.rs`

The file under design (`grad_fns/indexing.rs`) is the autograd layer; the
kernel layer lives at `ferrotorch-core/src/ops/indexing.rs` (1004 LOC,
six `pub fn`s: `gather` at `:112`, `scatter` at `:183`, `scatter_add` at
`:259`, `where_cond` at `:334`, `where_cond_bt` at `:397`, `masked_select`
at `:478`). The split mirrors PyTorch's
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

### REQ-1 `gather` — `GatherBackward` (`indexing.rs:444-520`)

`GatherBackward<T>` saves `input: Tensor<T>`, `dim: usize`,
`index: Vec<usize>`, `index_shape: Vec<usize>`. Backward at `:456-510`:
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
is Arc-attached at `ferrotorch-core/src/ops/indexing.rs:155-159` inside
`pub fn gather` (the forward kernel) when `input.requires_grad() &&
is_grad_enabled()`. The Arc-attach IS the consumer site — every CPU forward
gather call that has a requires-grad input creates a `GatherBackward` graph
node. (The forward itself rejects CUDA input at `ops/indexing.rs:127-129`
via `NotImplementedOnCuda`, so the GPU backward path is only ever exercised
when a CPU forward gather's grad_output has migrated to CUDA — a rare path
but kernel-tested.)

### REQ-2 `scatter` — `ScatterBackward` (`indexing.rs:534-658`)

`ScatterBackward<T>` saves `input`, `src`, `dim`, `index`, `index_shape`.
Backward at `:548-649` returns `vec![grad_input, grad_src]`:
- `grad_input`: copy of grad_output with scattered positions zeroed (per
  `derivatives.yaml:1509 self: grad.scatter(dim, index, 0)`).
- `grad_src`: gather from grad_output at the scatter-written positions
  (per `derivatives.yaml:1511 src: grad.gather(dim, index)`).

GPU paths via `backend.masked_zero_f32` (input grad) and
`backend.index_select_1d_f32` (src grad); CPU paths inline the walk.
Returns `None` for any leg whose tensor doesn't require_grad
(`:569-582 / :584-598` for GPU; `:607-625 / :628-645` for CPU). **Non-test
production consumer**: Arc-attached at `ops/indexing.rs:235-241` inside
`pub fn scatter`.

### REQ-3 `scatter_add` — `ScatterAddBackward` (`indexing.rs:672-788`)

`ScatterAddBackward<T>` saves `input`, `src`, `dim`, `index`, `index_shape`.
Backward at `:686-779`:
- `grad_input`: identity (`derivatives.yaml:1520 self: grad`) — on GPU via
  `backend.clone_buffer`, on CPU via a Vec clone.
- `grad_src`: same gather-from-scattered-positions logic as scatter
  (`derivatives.yaml:1522 src: grad.gather(dim, index)`).

There is a dead-code branch at `:736-740 if grad_output.is_cuda() { return
Err(NotImplementedOnCuda { op: "scatter_add backward" }); }` that is
unreachable because the GPU path returns at `:733` above it — a residual
from before the GPU resident path landed. **Non-test production consumer**:
Arc-attached at `ops/indexing.rs:311-318` inside `pub fn scatter_add`.
Additionally, `ops::indexing::scatter_add` itself (the forward) is consumed
at `ferrotorch-core/src/grad_fns/cumulative.rs:503` inside
`cummaxmin_backward_impl` — the cummax/cummin VJP scatter-adds grad through
the saved indices, which transitively exercises `ScatterAddBackward` ONLY
when the cumulative input itself requires grad and the scatter-add is run
under autograd-enabled mode; in the cumulative.rs use it's wrapped so the
returned tensor's grad_fn is the `CummaxBackward` / `CumminBackward`, not
the scatter_add's own.

### REQ-5 `index_select` — three forward shapes + two backward structs

`IndexSelectBackward<T>` at `:149-205` is the 1-D backward used by
`index_select_1d` (`:212-277`) and `index_select_1d_it` (`:1053-1076`).
The backward walks `grad_output` and scatters into `grad_input[idx] +=
grad_output[i]`. GPU path: f32 only (`:165-181`) via
`backend.scatter_add_1d_f32`.

`IndexSelectDimBackward<T>` at `:1091-1212` is the N-D backward used by
`index_select_dim` (`:1229-1366`). The backward computes per-element flat
destination indices for the scatter-add via the
`outer * out_dim_size * inner` decomposition, supporting both f32 and
f64 GPU paths (`:1126-1184`). The CPU path (`:1186-1202`) inlines the
`scatter_add` walk.

**Non-test production consumer**: `index_select_dim` is called at
`ferrotorch-data/src/transforms.rs:389` inside
`HorizontalFlip::apply(...)` — `no_grad(|| index_select_dim(&input,
last_dim_axis, &indices))`. This is the chainable axis-flip primitive
that subsumes the prior chunks-based reverse implementation per #1107.
The 1-D variants (`index_select_1d`, `index_select_1d_it`) have no
in-tree non-test consumer; their characterization tests at
`indexing.rs:1395-1421` and `:1448-1645` are the only callers today.

### REQ-9/10/11 masked family — three backward structs

- `MaskedFillBackward<T>` (`:295-358`) saves `input` (for shape) and
  `mask: BoolTensor` (resident-capable per #1185 Phase 3c). Backward zeros
  grad at mask-true positions via `backend.masked_fill_dt(grad, mask, 0.0)`
  on GPU (`:314-329`) or a host-mask walk on CPU (`:336-348`). NO float-mask
  upload, NO host crossing on the resident path.
- `MaskedSelectBackward<T>` (`:923-987`) saves `input` and `mask:
  BoolTensor`. Backward scatters the compacted grad back into a
  `zeros(input.numel())` at the mask-true flat positions — GPU path via
  `backend.masked_scatter(grad, mask, input_numel)` at `:944-960`, CPU
  path at `:963-977`.
- `WhereCondBackward<T>` (`:800-904`) saves `x`, `y`, `condition:
  BoolTensor`. Backward returns
  `(grad_x = where(cond, grad, 0), grad_y = where(cond, 0, grad))` per
  `derivatives.yaml:1955-1958`. GPU-resident path at `:825-862` reuses
  `backend.masked_fill_dt` with `value=0` + `backend.bool_not` for the
  cond-flip on grad_x; CPU path at `:866-893`.

**Non-test production consumers**:
- `MaskedFillBackward` ← Arc-attached at `indexing.rs:405, :423, :1035`
  (the three forward variants in this file), with the
  `masked_fill_bt`-via-`Tensor::masked_fill` chain reachable at
  `tensor.rs:1126-1132`.
- `MaskedSelectBackward` ← Arc-attached at `ops/indexing.rs:509-512`,
  reachable via `Tensor::masked_select` at `tensor.rs:1142-1147`.
- `WhereCondBackward` ← Arc-attached at `ops/indexing.rs:378, :445`.

REQ-11 (`masked_scatter` forward) is NOT-STARTED because the
`backend.masked_scatter` GPU kernel exists but no top-level `pub fn` /
`MaskedScatterBackward` exposes it as a forward op; it is currently only
consumed inside `MaskedSelectBackward`'s VJP. See blocker #1252.

### REQ-14 `where` — `WhereCondBackward` (`indexing.rs:800-904`)

The forward lives at `ops/indexing.rs:334-394` (`where_cond` with host
`&[bool]` condition) and `:397-466` (`where_cond_bt` with BoolTensor
condition); both Arc-attach `WhereCondBackward` from this file. The
backward's GPU-resident path (`:825-862`) is the crosslink #1187 Phase 3d
landing — both legs reuse `backend.masked_fill_dt(grad, mask, 0)` with
`mask = cond` (for grad_y) and `mask = bool_not(cond)` (for grad_x). NO
host crossing, NO float-mask upload, dtype-generic (f32/f64/bf16/f16).

**API divergence (R-DEV-2 — annotated)**: PyTorch's user-facing name is
`torch.where(condition, self, other)` per `torch/overrides.py:1277`.
ferrotorch names it `where_cond` to avoid colliding with the Rust `where`
keyword in method position. The kernel-layer pub re-export at
`ferrotorch-core/src/lib.rs:174` is `pub use ops::indexing::{...
where_cond, where_cond_bt}`. There is no `Tensor::where` chainable method
in `tensor.rs`; the parity-runner would have to either accept the
ferrotorch name or wrap. Blocker #1255 covers both the runner-dispatch
gap and the method-style consumer gap.

### NOT-STARTED REQs (REQ-4 / REQ-6 / REQ-7 / REQ-8 / REQ-11 / REQ-12 / REQ-13)

Seven REQs have NO ferrotorch implementation today:
- `scatter_reduce` (REQ-4): 4 reduce modes × `include_self` boolean — the
  largest single missing piece in the indexing family. #1245.
- `index_add` (REQ-6): #1247. VJP needs `grad.index_select`.
- `index_copy` (REQ-7): #1248. VJP needs `grad.index_fill` (REQ-8).
- `index_fill` (REQ-8): #1249.
- `masked_scatter` forward (REQ-11): #1252. The GPU kernel exists but no
  forward `pub fn` / `MaskedScatterBackward` struct.
- `take` (REQ-12): #1253. VJP needs `put-with-accumulate`.
- `put` (REQ-13): #1254. VJP needs `take` (REQ-12).

The REQ-7↔REQ-8 and REQ-12↔REQ-13 pairs are mutually dependent — implementing
one requires the other. The blockers document the dependency.

## Parity contract

| Op | Upstream entry | Backward formula source | Edge cases mirrored |
|---|---|---|---|
| `gather` | `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2070 TORCH_IMPL_FUNC(gather_out)` | `derivatives.yaml:730-733` (gather_backward = zeros.scatter_add_(dim, index, grad)) | Empty index: forward returns shape-matching empty (upstream early-out at `:2076-2077 if (index.numel() == 0) return`). Out-of-bounds index: ferrotorch returns `FerrotorchError::IndexOutOfBounds` from `validate_gather_shapes` at `ops/indexing.rs:83-91` (upstream raises `RuntimeError`). NaN/Inf input: propagates naturally through gather. Non-contiguous: forward rejects CUDA inputs at `ops/indexing.rs:127`. `sparse_grad=True`: NOT-STARTED (sparse out of scope). |
| `scatter` | `TensorAdvancedIndexing.cpp:2263 TORCH_IMPL_FUNC(scatter_src_out)` | `derivatives.yaml:1508-1511` (input: scatter(dim, index, 0); src: gather(dim, index)) | Duplicate indices in scatter: upstream `scatter_stub` overwrites (last writer wins for non-deterministic); ferrotorch CPU walk at `ops/indexing.rs:225-232` mirrors via flat index. NaN: passes through unchanged. CUDA: NOT-STARTED at the forward (`ops/indexing.rs:199-201`). |
| `scatter_add` | `TensorAdvancedIndexing.cpp:2317 TORCH_IMPL_FUNC(scatter_add)` | `derivatives.yaml:1519-1522` (self: grad; src: gather) | Duplicate indices: accumulate via `+=` (deterministic on CPU). `globalContext().deterministicAlgorithms()` switch at `:2338-2342` chooses `_scatter_via_index_put` route on CUDA/XPU for float dtypes — ferrotorch does not have a determinism-mode switch; the CPU path is inherently deterministic. NaN: arithmetic propagates. |
| `scatter_reduce` | `TensorAdvancedIndexing.cpp:2354 TORCH_IMPL_FUNC(scatter_reduce_two)` | `derivatives.yaml:3074-3077` (per-reduce-mode `scatter_reduce_backward`) | NOT-STARTED. |
| `index_select` | `TensorAdvancedIndexing.cpp:1862 index_select_cpu_` | `derivatives.yaml:910-913` (index_select_backward = zeros.index_add_(dim, index, grad)) | Empty index: forward returns shape-replacing-axis-with-0 tensor. Out-of-bounds: returns `IndexOutOfBounds`. Negative index: rejected with `InvalidArgument` (upstream wraps negative to positive — divergence per `IntTensor` validation at `indexing.rs:1067-1072`). |
| `index_add` | `TensorAdvancedIndexing.cpp:1153 TORCH_IMPL_FUNC(index_add_cpu_out)` | `derivatives.yaml:862-868` | NOT-STARTED. |
| `index_copy` | `TensorAdvancedIndexing.cpp:1082 TORCH_IMPL_FUNC(index_copy_out)` | `derivatives.yaml:875-883` | NOT-STARTED. |
| `index_fill` | `TensorAdvancedIndexing.cpp:1979 Tensor index_fill(...)` | `derivatives.yaml:884-887` | NOT-STARTED. |
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

`grad_fns::indexing::tests` mod at `indexing.rs:1428-1968`:
- Forward + backward for `index_select_1d`: `test_index_select_1d_forward`
  (`:1448`), `test_index_select_1d_duplicate_indices` (`:1457`),
  `test_index_select_1d_out_of_bounds` (`:1466`),
  `test_index_select_1d_non_1d_input` (`:1473`),
  `test_index_select_1d_backward_simple` (`:1487`),
  `test_index_select_1d_backward_duplicate_indices` (`:1554`),
  `test_index_select_1d_backward_weighted_grad` (`:1597`),
  `test_index_select_1d_no_grad_context` (`:1637`).
- `masked_fill`: `test_masked_fill_forward` (`:1650`),
  `test_masked_fill_backward` (`:1661`),
  `test_masked_fill_shape_mismatch` (`:1684`).
- Gather / scatter_add backward smoke: `test_gather_backward_stub`
  (`:1694`), `test_scatter_add_backward_stub` (`:1710`).
- `index_select_dim` (REQ-5 N-D): `test_index_select_dim_2d_dim0_forward`
  (`:1729`), `test_index_select_dim_2d_dim1_forward` (`:1758`),
  `test_index_select_dim_registers_grad_fn` (`:1779`),
  `test_index_select_dim_backward_simple_2d` (`:1794`),
  `test_index_select_dim_backward_dim1` (`:1837`),
  `test_index_select_dim_e2e_via_autograd` (`:1873`),
  `test_index_select_dim_rejects_2d_indices` (`:1943`),
  `test_index_select_dim_rejects_oob` (`:1952`),
  `test_index_select_dim_rejects_negative` (`:1961`).

`grad_fns::indexing::first_class_wrappers_tests` mod at `:1368-1422`:
- `masked_fill_bt`: `masked_fill_bt_replaces_true_positions` (`:1373`),
  `masked_fill_bt_rejects_shape_mismatch` (`:1386`).
- `index_select_1d_it`: `index_select_1d_it_picks_at_indices` (`:1395`),
  `index_select_1d_it_rejects_2d_indices` (`:1408`),
  `index_select_1d_it_rejects_negative` (`:1416`).

### Parity-sweep status (2026-05-25 reproducers)

```
./target/release/parity-sweep sweep --op gather         --seeds 8
  => [gather]         32/56  passed (24  skipped, 0 failed)  # SHIPPED 2026-05-25
./target/release/parity-sweep sweep --op scatter        --seeds 8
  => [scatter]        112/216 passed (104 skipped, 0 failed) # SHIPPED 2026-05-25
./target/release/parity-sweep sweep --op scatter_add    --seeds 8
  => [scatter_add]    48/56  passed (8   skipped, 0 failed)  # SHIPPED 2026-05-25
./target/release/parity-sweep sweep --op scatter_reduce --seeds 8
  => [scatter_reduce] 0/168 passed (168 skipped, 0 failed)
./target/release/parity-sweep sweep --op index_select   --seeds 8
  => [index_select]   16/24  passed (8   skipped, 0 failed)  # SHIPPED 2026-05-25
./target/release/parity-sweep sweep --op index_add      --seeds 8
  => [index_add]      0/72  passed (72  skipped, 0 failed)
./target/release/parity-sweep sweep --op index_copy     --seeds 8
  => [index_copy]     0/24  passed (24  skipped, 0 failed)
./target/release/parity-sweep sweep --op index_fill     --seeds 8
  => [index_fill]     0/48  passed (48  skipped, 0 failed)
./target/release/parity-sweep sweep --op masked_select  --seeds 8
  => [masked_select]  56/56 passed (0   skipped, 0 failed)  # SHIPPED 2026-05-25
./target/release/parity-sweep sweep --op masked_fill    --seeds 8
  => [masked_fill]    64/64 passed (0   skipped, 0 failed)  # SHIPPED 2026-05-25
./target/release/parity-sweep sweep --op masked_scatter --seeds 8
  => [masked_scatter] 0/32  passed (32  skipped, 0 failed)
./target/release/parity-sweep sweep --op take           --seeds 8
  => [take]           0/80  passed (80  skipped, 0 failed)
./target/release/parity-sweep sweep --op put            --seeds 8
  => [put]            0/224 passed (224 skipped, 0 failed)
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
| REQ-1 (gather) | SHIPPED | impl exists: forward at `ferrotorch-core/src/ops/indexing.rs:112 pub fn gather` attaching `GatherBackward` at `:155`; backward struct at `ferrotorch-core/src/grad_fns/indexing.rs:444-520` mirroring `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2070 TORCH_IMPL_FUNC(gather_out)` and `tools/autograd/derivatives.yaml:730-733`. **Runner arm landed 2026-05-25** at `tools/parity-sweep/runner/src/main.rs` decoding positional `[input_f32, dim_i64, index_int_uint8/int32/int64]` and routing to `ops::indexing::gather`. **Non-test production consumer**: `ops::indexing::gather` itself is the `ferrotorch-core` library's public surface; the `GatherBackward` autograd attach at `ops/indexing.rs:155-159` is its in-graph use-site. Parity gate: **`[gather] 32/56 passed (24 skipped, 0 failed)` at seeds 0..8** — 0 failures, 57% pass; skips are narrower-contract rejections (0-d input #1256, ndim-mismatch index broadcasting). Closes #1242. |
| REQ-2 (scatter) | SHIPPED | impl exists: forward at `ops/indexing.rs:183 pub fn scatter` attaching `ScatterBackward` at `:235`; backward at `grad_fns/indexing.rs:534-658` mirroring `TensorAdvancedIndexing.cpp:2263 TORCH_IMPL_FUNC(scatter_src_out)` and `derivatives.yaml:1508-1511`. **Runner arm landed 2026-05-25** at `tools/parity-sweep/runner/src/main.rs` decoding `[input_f32, dim_i64, index_int64, src_f32]` + `reduce` kwarg routing (`reduce='add'` → `scatter_add`; `'multiply'`/`'amin'`/`'amax'`/`'mean'`/etc routes to skip per REQ-4 #1245; absent routes to plain scatter). **Non-test production consumer**: `ops::indexing::scatter` is the library's public surface; the `ScatterBackward` autograd attach at `ops/indexing.rs:235-241` is its in-graph use-site. Parity gate: **`[scatter] 112/216 passed (104 skipped, 0 failed)` at seeds 0..8** — 0 failures, 52% pass; skips break down as scatter_reduce variants (#1245), scatter.value scalar-src overload (#1258), 0-d input (#1256), and ndim-mismatch index. Closes #1243. |
| REQ-3 (scatter_add) | SHIPPED | impl exists: forward at `ops/indexing.rs:259 pub fn scatter_add` attaching `ScatterAddBackward` at `:311`; backward at `grad_fns/indexing.rs:672-788` mirroring `TensorAdvancedIndexing.cpp:2317 TORCH_IMPL_FUNC(scatter_add)` and `derivatives.yaml:1519-1522`. **Runner arm landed 2026-05-25** at `tools/parity-sweep/runner/src/main.rs` decoding `[input_f32, dim_i64, index_int64, src_f32]` and routing to `ops::indexing::scatter_add`. **Non-test production consumer**: `ferrotorch-core/src/grad_fns/cumulative.rs:503 ops::indexing::scatter_add(...)` inside `cummaxmin_backward_impl` — the cummax/cummin VJP scatter-adds grad through the saved indices. Parity gate: **`[scatter_add] 48/56 passed (8 skipped, 0 failed)` at seeds 0..8** — 0 failures, 86% pass; skips are 0-d input only (#1256). Closes #1244. |
| REQ-4 (scatter_reduce) | NOT-STARTED | no impl. No forward kernel in `ops/indexing.rs`, no `ScatterReduceBackward` struct, no consumer. Upstream `TORCH_IMPL_FUNC(scatter_reduce_two)` at `TensorAdvancedIndexing.cpp:2354` and `derivatives.yaml:3074-3077`. `[scatter_reduce] 0/168 passed`. Blocker #1245. |
| REQ-5 (index_select) | SHIPPED | impl exists: 1-D at `grad_fns/indexing.rs:212 pub fn index_select_1d` + `IndexSelectBackward` at `:149`; IntTensor wrapper at `:1053 pub fn index_select_1d_it`; N-D at `:1229 pub fn index_select_dim` + `IndexSelectDimBackward` at `:1091` mirroring `TensorAdvancedIndexing.cpp:1862 index_select_cpu_` and `derivatives.yaml:910-913`. **Runner arm landed 2026-05-25** at `tools/parity-sweep/runner/src/main.rs` decoding `[input_f32, dim_i64, index_int64]` with negative-dim normalization, routing to `grad_fns::indexing::index_select_dim`. **Non-test production consumer**: `ferrotorch-data/src/transforms.rs:389 no_grad(|| index_select_dim(&input, last_dim_axis, &indices))` inside `HorizontalFlip::apply`. Parity gate: **`[index_select] 16/24 passed (8 skipped, 0 failed)` at seeds 0..8** — 0 failures, 67% pass; skips are 0-d input only (#1256). Closes #1246. |
| REQ-6 (index_add) | NOT-STARTED | no impl. `[index_add] 0/72 passed`. Blocker #1247. |
| REQ-7 (index_copy) | NOT-STARTED | no impl; coupled to REQ-8 via VJP. `[index_copy] 0/24 passed`. Blocker #1248. |
| REQ-8 (index_fill) | NOT-STARTED | no impl; VJP target of REQ-7. `[index_fill] 0/48 passed`. Blocker #1249. |
| REQ-9 (masked_select) | SHIPPED | shape-strict forward at `ops/indexing.rs:478 pub fn masked_select` attaching `MaskedSelectBackward` at `:509`; backward at `grad_fns/indexing.rs:923-987` mirroring `TensorAdvancedIndexing.cpp:2621 masked_select_cpu` and `derivatives.yaml:1116-1119`. **Broadcasting wrapper landed 2026-05-25**: `grad_fns/indexing.rs:1526 pub fn masked_select_bcast` infers the common broadcast shape via `shape::broadcast_shapes`, expands both operands via the autograd-aware `grad_fns::shape::expand` (whose `ExpandBackward` reduces gradients back to original shape), then delegates to the shape-strict forward. Mirrors upstream `expand_outplace(mask, self)` at `TensorAdvancedIndexing.cpp:2545`. **Non-test production consumer**: `tools/parity-sweep/runner/src/main.rs:670 "masked_select" => masked_select_bcast(...)` — the runner dispatch routes op_db samples through the wrapper. Parity gate: **`[masked_select] 56/56 passed (0 skipped, 0 failed)` at seeds 0..8**. Closes #1250. |
| REQ-10 (masked_fill) | SHIPPED | shape-strict forwards at `grad_fns/indexing.rs:367 pub fn masked_fill` (host `&[bool]`) + `:997 pub fn masked_fill_bt` (BoolTensor) attaching `MaskedFillBackward` at `:295`. Forward + backward mirror `TensorAdvancedIndexing.cpp:2494 Tensor masked_fill(...)` and `derivatives.yaml:1094-1097`. **Broadcasting wrapper landed 2026-05-25**: `grad_fns/indexing.rs:1503 pub fn masked_fill_bcast` expands input + mask to common shape via autograd-aware expand + a CPU-side bool broadcast (`broadcast_bool_tensor` at `:1463`), then delegates to `masked_fill_bt`. Mirrors upstream `expand_outplace(mask, self)` at `TensorAdvancedIndexing.cpp:2503`. **Non-test production consumer**: `tools/parity-sweep/runner/src/main.rs:691 "masked_fill" => masked_fill_bcast(...)`. Parity gate: **`[masked_fill] 64/64 passed (0 skipped, 0 failed)` at seeds 0..8**. Closes #1251. |
| REQ-11 (masked_scatter) | NOT-STARTED | no forward impl. The `backend.masked_scatter` GPU kernel exists at `ferrotorch-gpu/src/masked_kernels.rs:933` but is consumed only inside `MaskedSelectBackward` (the VJP of masked_select). No `pub fn masked_scatter` and no `MaskedScatterBackward` struct. Upstream forward at `TensorAdvancedIndexing.cpp:2402` and VJP at `derivatives.yaml:1105-1108`. `[masked_scatter] 0/32 passed`. Blocker #1252. |
| REQ-12 (take) | NOT-STARTED | no impl; VJP requires REQ-13 (`put`). `[take] 0/80 passed`. Blocker #1253. |
| REQ-13 (put) | NOT-STARTED | no impl; VJP requires REQ-12 (`take`). `[put] 0/224 passed`. Blocker #1254. |
| REQ-14 (where) | SHIPPED | shape-strict forward at `ops/indexing.rs:334 pub fn where_cond` + `:397 pub fn where_cond_bt` attaching `WhereCondBackward` at `:378` / `:445`; backward at `grad_fns/indexing.rs:800-904` mirroring `aten/src/ATen/native/TensorCompare.cpp:642 Tensor where(...)` and `derivatives.yaml:1955-1959`. **Broadcasting wrapper landed 2026-05-25**: `grad_fns/indexing.rs:1547 pub fn where_cond_bcast` performs 3-way broadcast (`shape::broadcast_shapes` applied pairwise: x⨯y then cond⨯(x⨯y)), expands x and y via autograd-aware `grad_fns::shape::expand` (so `ExpandBackward` shrinks gradients to original shapes), broadcasts cond via `broadcast_bool_tensor`, then delegates to `where_cond_bt`. Mirrors upstream 3-way TensorIterator at `TensorCompare.cpp:629-637 where_self_out`. **API divergence (R-DEV-2)**: ferrotorch name remains `where_cond` / `where_cond_bcast` to avoid the Rust `where` keyword; PyTorch uses `torch.where`. **Non-test production consumer**: `tools/parity-sweep/runner/src/main.rs:741 "where" => where_cond_bcast(...)` — the runner routes op_db's `torch.where(cond, x, y)` samples through this wrapper. Parity gate: **`[where] 48/48 passed (0 skipped, 0 failed)` at seeds 0..8**. Closes #1255. |
| REQ-15 (shared helpers) | SHIPPED | impl: `upload_f32_to_gpu` at `grad_fns/indexing.rs:25-38`, `scatter_write_mask` at `:42-64`, `gather_dst_flat_indices` at `:69-91`, `scatter_src_flat_indices` at `:96-106`, `flat_index` at `:114-122`, `increment_coords` at `:127-136`. Non-test production consumers: `GatherBackward::backward` (`:474`), `ScatterBackward::backward` (`:572, :587`), `ScatterAddBackward::backward` (`:720`), `IndexSelectBackward::backward` (`:173`), `IndexSelectDimBackward::backward` (`:1159`), `MaskedFillBackward::backward` (`:394`, via the f32 path's mask upload). The helpers themselves have no public API surface — they are file-local utility scaffolding shared across every N-D autograd VJP in this file. Verified by the 27-test pass run at AC-15. |
