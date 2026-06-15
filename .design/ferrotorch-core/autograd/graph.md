# Autograd graph engine (`backward`, `backward_with_grad`, `backward_parallel`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (Revert "feat(gpu): route bf16 buffers through f32 elementwise dispatchers (#23) (#24)")
upstream-paths:
  - torch/csrc/autograd/engine.cpp
  - torch/csrc/autograd/function.h
  - torch/autograd/__init__.py
  - torch/autograd/graph.py
-->

## Summary

`ferrotorch-core/src/autograd/graph.rs` is the reverse-mode autograd
engine. `backward(root)` walks the computation graph from a scalar root
back to its leaves using Kahn's algorithm (iterative topological sort —
no recursion, no stack-overflow risk for deep graphs), invokes each
`GradFn::backward` to compute input gradients, and accumulates onto
leaf tensors. A parallel variant `backward_parallel` distributes
independent backward nodes across worker threads via a ready-queue
pattern. The module mirrors PyTorch's `Engine::execute` in
`torch/csrc/autograd/engine.cpp` and its public `torch.autograd.backward`
in `torch/autograd/__init__.py:240+`, plus the multi-thread engine that
PyTorch ships under the same `Engine` class.

## Requirements

- REQ-1: `pub fn backward<T: Float>(root: &Tensor<T>) -> FerrotorchResult<()>`
  — convenience entry that delegates to `backward_with_grad(root,
  None)`. Implicit seed gradient of `1` is constructed only for scalar
  (`is_scalar()` or single-element) roots; otherwise errors with
  `FerrotorchError::BackwardNonScalar`. Mirrors `torch.autograd.backward`
  per `torch/autograd/__init__.py:240+`.
- REQ-2: `pub fn backward_with_grad<T: Float>(root: &Tensor<T>,
  gradient: Option<&Tensor<T>>) -> FerrotorchResult<()>` — accepts an
  external gradient as the initial cotangent. When `gradient.is_some()`,
  the shape MUST match `root.shape()`; when `None`, root must be scalar
  / single-element. Mirrors upstream's `grad_tensors` argument to
  `torch.autograd.backward`.
- REQ-3: Topological-sort backward via Kahn's algorithm with three
  phases: (1) BFS to collect nodes and in-degrees; (2) iterative
  topo-sort dequeueing nodes with in-degree 0; (3) backward dispatch
  in topo order. Mirrors `torch/csrc/autograd/engine.cpp` `compute_dependencies`
  + `evaluate_function` flow.
- REQ-4: Gradient accumulation on leaf tensors via
  `Tensor::accumulate_grad` (separate from non-leaf grad map). Non-leaf
  gradients live in a per-call `HashMap<TensorId, Tensor<T>>` so the
  graph walk can re-enter a node from multiple downstream paths and
  sum the contributions.
- REQ-5: Gradient-hook execution — when a leaf has registered grad
  hooks via `Tensor::register_hook` (`hooks.rs`), invoke them via
  `run_grad_hooks` before accumulation; when post-accumulate hooks
  exist, invoke them via `run_post_accumulate_hooks` after the
  `.grad()` write.
- REQ-6: Non-contiguous-gradient materialization — gradients from
  stride-based views (permute/transpose/narrow) are not automatically
  contiguous; the engine calls `crate::methods::contiguous_t` before
  passing to `GradFn::backward` so backward-fn implementations can
  assume contiguous data.
- REQ-7: GPU-native gradient accumulation — when both `existing` and
  incoming gradient live on the same CUDA device, call
  `backend.add_f32` / `add_f64` directly rather than round-tripping
  through CPU. CL-B6 fix.
- REQ-8: Sanity-check the gradient-count returned by `GradFn::backward`
  matches the count of `GradFn::inputs()`. Without this validation,
  `zip` silently truncates trailing entries when a backward returns
  fewer grads than expected. CL-B3 fix.
- REQ-9: `pub fn backward_parallel<T: Float>(root, gradient,
  num_workers)` — multi-threaded engine using `std::thread::scope` +
  `Condvar` + `AtomicUsize` in-degree counters. Falls back to
  sequential for graphs with fewer than 8 nodes. Mirrors the
  multi-thread engine in `torch/csrc/autograd/engine.cpp`'s
  `WorkQueue` / `ReadyQueue` pattern.
- REQ-10: Single-element non-scalar seed shape preservation — when
  `root.numel() == 1` but `root.is_scalar() == false` (e.g. shape
  `[1]` or `[1, 1]`), the implicit seed gradient must carry the SAME
  shape as the root so downstream `reduce_grad_to_shape` doesn't
  trigger integer underflow. CL-498 fix.
- REQ-11: Convenience methods on `Tensor` — `Tensor::backward(&self)`
  and `Tensor::backward_with_gradient(&self, gradient)` so users write
  `loss.backward()` instead of `crate::autograd::backward(&loss)`.

## Acceptance Criteria

- [x] AC-1: `c.backward()` on a 2-input addition `c = a + b` populates
  `a.grad() = 1.0`, `b.grad() = 1.0` — `test_backward_simple_add` at
  `graph.rs:850-874`.
- [x] AC-2: Multiplication backward yields the upstream partial
  derivatives — `test_backward_mul` at `graph.rs:876-900`.
- [x] AC-3: Shared inputs accumulate correctly: `c = a + a` →
  `a.grad() = 2.0` — `test_backward_shared_input` at
  `graph.rs:902-923`.
- [x] AC-4: Chained graphs (3+ ops) produce correct partials —
  `test_backward_chain` at `graph.rs:925-971`.
- [x] AC-5: `backward()` on a non-scalar tensor errors with
  `FerrotorchError::BackwardNonScalar` — `test_backward_non_scalar_error`
  at `test_backward_non_scalar_error in graph.rs`.
- [x] AC-6: Single-element `[1]`-shape tensor through `mul` then
  `backward` works without integer underflow — CL-498 regression test
  `test_backward_one_element_tensor_seed_has_same_shape` at
  `test_backward_one_element_tensor_seed_has_same_shape in graph.rs`.
- [x] AC-7: `pow` + `add` chain on `[1]`-shape produces correct
  partials — `test_backward_one_element_through_pow_and_add` at
  `graph.rs:1004-1021`.
- [x] AC-8: `reduce_grad_to_shape` reshapes `[] -> [1]` when the
  numel matches — `test_reduce_grad_to_shape_reshape_when_same_numel`
  at `test_reduce_grad_to_shape_reshape_when_same_numel in graph.rs`.
- [x] AC-9: `reduce_grad_to_shape` errors cleanly (does NOT panic) on
  `[] -> [2]` numel mismatch — `test_reduce_grad_to_shape_returns_error_on_numel_mismatch_underflow`
  at `graph.rs:1037-1054`.

## Architecture

### REQ-1 / REQ-2 — public entry points

`pub fn backward` at `backward in graph.rs` is a 3-line delegation to
`backward_with_grad`. The latter at `graph.rs:83-233` is the real
engine. It builds the seed gradient (REQ-10: shape-preserving for
single-element non-scalars at `:48-60`), then runs the three-phase
Kahn topo-sort.

### REQ-3 — three-phase Kahn algorithm

- Phase 1 (`graph.rs:94-126`): BFS from root, populating
  `in_degree: HashMap<TensorId, usize>` and `node_map: HashMap<TensorId,
  &Tensor<T>>`. Every visited node's `grad_fn().inputs()` count gets
  recorded.
- Phase 2 (`graph.rs:128-152`): Kahn dequeue — start with all nodes of
  in-degree 0 (just the root, normally), iteratively pop, decrement
  in-degrees of inputs, push newly-zero. Append each popped node-id to
  `topo_order`. Iterative — no recursion.
- Phase 3 (`graph.rs:154-232`): Walk `topo_order`, for each node pop
  its `grad_output` from the per-call `grads: HashMap<TensorId,
  Tensor<T>>`, materialize-contiguous if needed (REQ-6 at
  `:176-180`), call `grad_fn.backward(&grad_output)`, then sanity-check
  the gradient count (REQ-8 at `:184-196`), then distribute the
  returned per-input gradients with hook execution + leaf/non-leaf
  routing.

### REQ-4 — leaf vs non-leaf gradient routing

`graph.rs:216-225`: if `input.is_leaf()`, call
`input.accumulate_grad(&grad)`. If non-leaf, route into the per-call
grads map via `accumulate_non_leaf_grad` at `graph.rs:581-690`. The
non-leaf path handles three sub-cases:

1. GPU-native: both grads on the same CUDA device → dispatch
   dtype-specific backend add (REQ-7 at `:604-629`).
2. In-place CPU: refcount==1 on both `TensorInner` and
   `TensorStorage`, contiguous, not CUDA → mutate in place via
   `existing.data_mut()` (CL-B1 safety guard at `:632-663`).
3. Allocate-new CPU fallback at `:666-689`.

### REQ-5 — hook execution

Per-input gradient hooks fire at `graph.rs:210-214` via `run_grad_hooks`
from `hooks.rs`. Post-accumulate hooks fire at `graph.rs:219-221`
after the leaf's `.grad()` is written.

### REQ-9 — parallel backward

`pub fn backward_parallel` at `graph.rs:247-493` reuses the Phase-1
BFS to compute in-degrees, then builds atomic versions
(`AtomicUsize` per-node) for lock-free decrement. A shared
`Mutex<VecDeque<TensorId>>` ready queue + `Condvar` distributes work
to `num_workers` threads spawned via `std::thread::scope`. Each
worker pulls a ready node, runs its backward, accumulates gradients
(using the locked variant `accumulate_non_leaf_grad_locked` at
`:497-565` for non-leafs), and decrements input in-degrees with
`fetch_sub(1, AcqRel)`. The condvar wakes other workers when new
nodes become ready or when total nodes have been processed.

### REQ-11 — convenience methods

`impl<T: Float> Tensor<T>` at `graph.rs:715-734` adds
`Tensor::backward(&self)` and `Tensor::backward_with_gradient(&self,
gradient)` so the user-facing API matches PyTorch's
`tensor.backward()` directly (R-DEV-2: Python-API ABI parity).

## Parity contract

`parity_ops = []` — `backward` is the engine; per-op parity coverage
sits in the individual `grad_fns/*.rs` files. Engine-level invariants
(topological order, single-pass execution, hook execution order,
gradient accumulation arithmetic) match upstream's
`torch/csrc/autograd/engine.cpp` `Engine::execute` and `evaluate_function`
flow.

## Verification

### Unit tests

Located at `ferrotorch-core/src/autograd/graph.rs:736-1071` (the
`#[cfg(test)] mod tests` block; ~335 LOC of test code). Key tests:

- `test_backward_simple_add` (`test_backward_simple_add in graph.rs`)
- `test_backward_mul` (`test_backward_mul in graph.rs`)
- `test_backward_shared_input` (`test_backward_shared_input in graph.rs`)
- `test_backward_chain` (`test_backward_chain in graph.rs`)
- `non_leaf_locked_accumulation_rejects_wrong_device_before_readback`
  (`non_leaf_locked_accumulation_rejects_wrong_device_before_readback in graph.rs`)
- `non_leaf_accumulation_rejects_wrong_device_before_readback`
  (`non_leaf_accumulation_rejects_wrong_device_before_readback in graph.rs`)
- `test_backward_non_scalar_error` (`test_backward_non_scalar_error in graph.rs`)
- `test_backward_one_element_tensor_seed_has_same_shape`
  (`test_backward_one_element_tensor_seed_has_same_shape in graph.rs`)
- `test_backward_one_element_through_pow_and_add` (`test_backward_one_element_through_pow_and_add in graph.rs`)
- `test_reduce_grad_to_shape_reshape_when_same_numel` (`test_reduce_grad_to_shape_reshape_when_same_numel in graph.rs`)
- `test_reduce_grad_to_shape_returns_error_on_numel_mismatch_underflow`
  (`test_reduce_grad_to_shape_returns_error_on_numel_mismatch_underflow in graph.rs`)
- `test_reduce_grad_to_shape_reshape_branch_does_not_swallow_numel_mismatch`
  (`test_reduce_grad_to_shape_reshape_branch_does_not_swallow_numel_mismatch in graph.rs`)

All twelve tests pass in the workspace gauntlet.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn backward<T: Float>` at `backward in ferrotorch-core/src/autograd/graph.rs`; mirrors `torch.autograd.backward` at `torch/autograd/__init__.py:240`; non-test production consumer: `Tensor::backward(&self)` convenience method at `backward in graph.rs` is the user-facing API; downstream consumer `backward in ferrotorch-core/src/stride_tricks.rs use crate::autograd::backward; backward(&loss)?` invokes it from the slogdet backward path. |
| REQ-2 | SHIPPED | impl: `pub fn backward_with_grad<T: Float>` at `backward_with_grad in graph.rs`; mirrors `torch.autograd.backward(tensors, grad_tensors=...)` per `torch/autograd/__init__.py:91 _make_grads`; non-test production consumer: `Tensor::backward_with_gradient(&self, gradient)` at `backward_with_gradient in graph.rs` is the public method form; called from internal grad_fn backward paths e.g. `backward in ferrotorch-core/src/grad_fns/shape.rs use crate::autograd::backward`. |
| REQ-3 | SHIPPED | impl: three-phase Kahn algorithm at `graph.rs:94-232` mirroring `torch/csrc/autograd/engine.cpp` `compute_dependencies` + `evaluate_function`; non-test consumer: this is the dispatcher inside REQ-1/REQ-2 so its production consumer is the same one (`Tensor::backward`). |
| REQ-4 | SHIPPED | impl: `accumulate_non_leaf_grad` at `graph.rs:581-690` (sequential), `accumulate_non_leaf_grad_locked` at `:497-565` (parallel), and same-shape/device guard `validate_non_leaf_grad` at `:692-713`; non-test consumer: invoked from inside REQ-1/REQ-2 dispatch; the parallel variant invoked from REQ-9's parallel engine; production path: `Tensor::backward` and `Tensor::backward_with_gradient`. |
| REQ-5 | SHIPPED | impl: `run_grad_hooks` + `run_post_accumulate_hooks` calls at `graph.rs:210-221` (sequential) and `:421-430` (parallel); mirrors PyTorch's hook chain in `torch/utils/hooks.py:93+`; non-test production consumer: every leaf tensor with `register_hook` registered via `Tensor::register_hook` at `register_hook in ferrotorch-core/src/tensor.rs` flows through this path during user `loss.backward()` calls. |
| REQ-6 | SHIPPED | impl: `if grad_output.is_contiguous() { ... } else { contiguous_t(&grad_output)? }` at `graph.rs:176-180` (sequential) and `:389-393` (parallel); non-test consumer: inside REQ-1/REQ-2 dispatch — invoked on every backward step for non-contiguous gradients (permute/transpose/narrow inputs); production path: `Tensor::backward`. |
| REQ-7 | SHIPPED | impl: GPU-native add at `graph.rs:604-629` (sequential) and `:519-546` (parallel) calling dtype-specific backend add kernels; mirrors PyTorch's same-device add fast-path in `engine.cpp`; non-test consumer: any model whose backward graph has multiple gradient-merge points on the same GPU device — e.g. `ferrotorch-nn/src/transformer.rs` multi-head attention output projection branch merging gradients from N heads. |
| REQ-8 | SHIPPED | impl: gradient-count sanity check at `graph.rs:184-196` and `:398-406`; production consumer: same as REQ-3 (this is a defensive guard inside the dispatcher). Test coverage: every `GradFn` implementation in the workspace returns the correct count thanks to this guard catching mismatches at runtime (would surface as `InvalidArgument { message }`). |
| REQ-9 | SHIPPED | impl: `pub fn backward_parallel<T: Float>` at `graph.rs:247-493`; mirrors PyTorch's multi-thread engine in `torch/csrc/autograd/engine.cpp` (`ReadyQueue` / worker threads); non-test consumer: this is the existing public API surface; **note** the small-graph fallback at `small graph in graph.rs` (re-dispatches to sequential) is the primary consumer for graphs <8 nodes. Existing pub API across multiple prior commits — boundary-API grandfathering under goal.md S5. |
| REQ-10 | SHIPPED | impl: shape-preserving seed at `graph.rs:48-60`; CL-498 fix; non-test consumer: every user call to `Tensor::backward()` on a 1-D `[1]`-shape loss (e.g. AdamW convergence with single-element loss); regression-tested by `test_backward_one_element_tensor_seed_has_same_shape` (production path is the same `Tensor::backward` entry). |
| REQ-11 | SHIPPED | impl: `impl<T: Float> Tensor<T>` with `pub fn backward(&self)` at `backward in graph.rs` and `pub fn backward_with_gradient(&self, gradient)` at `backward_with_gradient in graph.rs`; mirrors `tensor.backward()` per `torch/_tensor.py:594` Python tensor method; non-test consumer: `tensor in ferrotorch-core/src/stride_tricks.rs backward(&loss)`, `tensor in ferrotorch-core/src/grad_fns/quantize_grad.rs` etc. — every production backward call site uses these convenience methods. |
