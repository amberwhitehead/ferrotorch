# vmap — vectorised map

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - torch/_functorch/vmap.py
  - torch/_functorch/apis.py
  - torch/func/__init__.py
  - aten/src/ATen/functorch/BatchRulesBinaryOps.cpp
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/vmap.rs` implements the `torch.vmap` /
`torch.func.vmap` function transform — vectorise a per-element function
over one or more batch dimensions. The current implementation is
loop-based (correct but not fused); a future version may trace the
function to produce a batched kernel. The module also ships the
foundational `select` (slice along a dim) and `stack` (along a new
dim) helpers that the vmap loop uses internally. Both helpers are
differentiable, device-preserving compositions over the crate's view /
concat primitives (CORE-056 / #1750): gradients flow back to the
batched leaves, CUDA and non-contiguous inputs are supported, and
mixed-device `stack` input lists return a structured `DeviceMismatch`.

## Requirements

- REQ-1: `select(input, dim, index)` — extract a single slice along
  `dim` at position `index`, returning a tensor of one rank lower.
  Mirrors `torch.select(input, dim, index)` including autograd
  (gradient scatters back into the source) and device preservation.
  Used as the per-batch-element extractor inside vmap.
- REQ-2: `stack(tensors, dim)` — stack a slice of tensors along a new
  dimension. Validates that every tensor has the same shape and (via
  `cat`) the same device. Mirrors `torch.stack(tensors, dim=0)`
  including autograd (gradient splits back to each input) and device
  preservation.
- REQ-3: `vmap(f, in_dim, out_dim)` — vectorise `f: Tensor -> Tensor`
  over a single batch dimension. Returns a closure that, when called
  with a batched input, runs `f` per slice and stacks the results.
  Mirrors `torch.vmap(f, in_dims=in_dim, out_dims=out_dim)` for the
  single-arg case.
- REQ-4: `vmap2(f, in_dims, out_dim)` — vectorise a two-argument
  function over a batch dim of each input. Mirrors `torch.vmap` with
  a 2-tuple `in_dims`.
- REQ-5: `vmap3(f, in_dims, out_dim)` — three-argument variant.
  Mirrors `torch.vmap` with a 3-tuple.
- REQ-6: `vmap_many(f, inputs, in_dims, out_dim)` — variadic version
  taking an explicit slice of inputs + a slice of per-input batch
  dims. Mirrors `torch.vmap` with an arbitrary tuple of `in_dims`.
- REQ-7: `vmap_multi_output(f, in_dim, out_dim)` — vectorise a
  function returning multiple tensors, stacking each output along
  `out_dim`. Mirrors `torch.vmap(f)` when `f` returns a tuple.
- REQ-8: `per_sample_grad(f, params, inputs)` — compute per-sample
  gradients via vmap-of-grad. Mirrors
  `torch.func.grad_and_value(f)` composed with `vmap`. Used for
  privacy / influence-function workflows.

## Acceptance Criteria

- [x] AC-1: `select(t, 0, i)` on `[B, M, N]` returns the `[M, N]`
  slice at batch index `i`.
- [x] AC-2: `stack` of `N` tensors of shape `[M, K]` along `dim=0`
  produces `[N, M, K]`.
- [x] AC-3: Index OOB in `select` returns `IndexOutOfBounds`.
- [x] AC-4: Shape mismatch in `stack` returns `ShapeMismatch`.
- [x] AC-5: `vmap(f, 0, 0)` applied to a `[B, M, K]` tensor with
  `f = matmul(_, weights)` produces a `[B, M, N]` output.
- [x] AC-6: `vmap2` of `f(a, b) = a + b` with `in_dims=(0, 0)` on
  two `[B, N]` inputs produces a `[B, N]` output.
- [x] AC-7: `cargo test -p ferrotorch-core --lib vmap` passes.

## Architecture

The implementation is loop-based: every `vmapN` walks the batch
dimension, calls `f` on the per-slice extracts, and stacks the
results. The loop transforms are pure compositions of `select` + `f`
+ `stack`, so autograd, CUDA support, and non-contiguous-input
support are all inherited from those two helpers (CORE-056 / #1750).

- `select` (`vmap.rs:51`) — validates `dim` / `index` (preserving the
  `InvalidArgument` / `IndexOutOfBounds` contract), then composes
  `narrow_t(input, dim, index, 1)` (zero-copy stride view carrying
  `NarrowBackward`, `methods.rs`) with `squeeze(dim)`
  (`SqueezeBackward`, `grad_fns/shape.rs`) — the same decomposition
  as `torch.select` (`aten/src/ATen/native/TensorShape.cpp`). CUDA
  and non-contiguous inputs go through the device-aware
  `contiguous()` staging of the view machinery.
- `stack` (`vmap.rs:94`) — validates emptiness / `dim` / shape
  equality (preserving the `InvalidArgument` / `ShapeMismatch`
  contract), then composes per-input `unsqueeze(dim)`
  (`UnsqueezeBackward`) with `cat(dim)` (`CatBackward`,
  device-validating, on-device fast path — #1749), the same
  decomposition as `torch.stack`. Mixed-device input lists are
  rejected with a structured device-mismatch error.
- `vmap` (`vmap.rs:154`) — returns an `impl Fn` closure that:
  1. Validates `in_dim < ndim`.
  2. Loops `0..batch_size`, calling `select(input, in_dim, i)` and
     then `f(&slice)`.
  3. `stack(&results, out_dim)`.
- `vmap2`, `vmap3`, `vmap_many`, `vmap_multi_output` follow the same
  shape with multiple inputs / outputs.
- `per_sample_grad` (`vmap.rs:509`) — uses the `vmap` loop pattern
  combined with `autograd::backward` on a per-sample loss. Each
  sample's parameter leaf is minted on the parameter's ORIGINAL
  device via `param.detach().requires_grad_(true)` (CORE-057 /
  #1751; torch's `p.detach().requires_grad_(True)` idiom — fresh
  autograd identity, shared storage, no host round trip). Input
  slices are detached so the original `inputs` never accumulates
  side-effect gradients (`torch.func.grad` functional semantics),
  and per-sample gradients are stacked without leaving their device.

Non-test production consumers:

- The vmap-of-jvp pattern is used by
  `autograd::forward_ad::jacfwd` (`autograd/forward_ad.rs:379-388`):
  the docstring there explicitly states "This is the vmap(jvp)
  pattern: we loop over basis vectors, computing one JVP per
  iteration." That implementation does not call the `vmap` helper
  directly (it inlines the loop for efficiency), but the pattern is
  the same and the helper module exists for direct user consumption.
- Re-exports at `lib.rs:233` (`pub use vmap::{select, stack, vmap,
  vmap2};`) make these reachable as `ferrotorch_core::vmap(...)` /
  `ferrotorch_core::select(...)` from any downstream crate.

## Parity contract

`parity_ops = []`. vmap is a function transform; its correctness
follows from `select` + `f` + `stack` being correct. The
implementation is explicitly loop-based — there is no fused-kernel
parity claim. When the future tracing path lands, it will need its
own parity sweep against `torch.vmap`'s batched output.

The shape contract for `select` matches `torch.select` exactly
(remove the indexed dim). The shape contract for `stack` matches
`torch.stack` exactly (insert the new dim). The autograd / device
contract is exercised by `tests/conformance_autograd.rs`: per public
variant, backward-to-batched-leaf gradient values against a live
`torch.vmap` oracle, non-contiguous input forward, and a
`--features gpu` lane asserting CUDA result/grad residency
(CORE-056 / #1750, CORE-057 / #1751). Known boundary: closures that
apply a whole-tensor CUDA reduction directly to a `select`ed slice
hit the storage-aliasing reduction divergence tracked as #1961
(pinned, `#[ignore]`d test `gpu_vmap_multi_output_cuda_sum_closure_pinned`).

## Verification

```bash
cargo test -p ferrotorch-core --lib vmap
```

Expected: a handful of tests covering select / stack / vmap1-2 pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `select` at `ferrotorch-core/src/vmap.rs:51` mirrors `torch.select` (narrow + squeeze composition, differentiable and device-preserving); non-test consumer: re-exported at `ferrotorch-core/src/lib.rs:233` and called inside `vmap` at `:180` as the per-batch-element extractor. |
| REQ-2 | SHIPPED | impl: `stack in ferrotorch-core/src/vmap.rs` mirrors `torch.stack`; non-test consumer: re-exported at `vmap in ferrotorch-core/src/lib.rs` and called inside `vmap in ferrotorch-core/src/lib.rs`. |
| REQ-3 | SHIPPED | impl: `vmap` at `ferrotorch-core/src/vmap.rs:154`; non-test consumer: re-exported at `ferrotorch-core/src/lib.rs:233`, reachable by downstream callers via `ferrotorch_core::vmap(_, _, _)`. The vmap(jvp) pattern is documented at `autograd/forward_ad.rs:379` as the production use case. |
| REQ-4 | SHIPPED | impl: `vmap2` at `ferrotorch-core/src/vmap.rs:198`; non-test consumer: re-exported at `ferrotorch-core/src/lib.rs:233` alongside the single-arg variant. |
| REQ-5 | SHIPPED | impl: `vmap3 in ferrotorch-core/src/vmap.rs`; non-test consumer: pub API surface (grandfathered per S5) — three-arg function transform is a documented `torch.vmap` use case. |
| REQ-6 | SHIPPED | impl: `vmap_many in ferrotorch-core/src/vmap.rs`; non-test consumer: pub API surface for arbitrary-arity vmap; grandfathered per S5. |
| REQ-7 | SHIPPED | impl: `vmap_multi_output in ferrotorch-core/src/vmap.rs`; non-test consumer: pub API surface used for `torch.func.vmap(f)` parity when `f` returns a tuple; grandfathered per S5. |
| REQ-8 | SHIPPED | impl: `per_sample_grad` at `ferrotorch-core/src/vmap.rs:509`; non-test consumer: pub API surface for the privacy / influence-function workflow that needs per-sample gradients; grandfathered per S5. |
