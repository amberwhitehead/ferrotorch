# Gradient checkpointing (`checkpoint`, `checkpoint_multi`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (Revert "feat(gpu): route bf16 buffers through f32 elementwise dispatchers (#23) (#24)")
upstream-paths:
  - torch/utils/checkpoint.py
-->

## Summary

`ferrotorch-core/src/autograd/checkpoint.rs` is the activation-recompute
memory-vs-compute trade-off layer for deep networks. The forward pass
runs inside `no_grad` (NO activations saved on the graph), and a
`CheckpointBackward` / `CheckpointMultiBackward` node attaches to the
output instead. During backward, the function is re-executed WITH
gradient tracking, with the same autocast snapshot and a forked RNG
state. The checkpoint saves the current thread's CPU generator plus
every distinct CUDA input-device generator state, restores them for
recompute, and then restores the caller's pre-backward RNG state. This
keeps stochastic recomputation numerically identical to the original
forward without perturbing the surrounding random stream.
Mirrors PyTorch's `torch.utils.checkpoint.checkpoint` at
`torch/utils/checkpoint.py:25-540`.

## Requirements

- REQ-1: `pub fn checkpoint<T, F>(f: F, input: &Tensor<T>) ->
  FerrotorchResult<Tensor<T>>` — single-input gradient checkpoint.
  Forward runs in `no_grad`; if `input.requires_grad()`, attach a
  `CheckpointBackward` node that re-runs `f` during backward. Mirrors
  `torch.utils.checkpoint.checkpoint(f, *args)` at
  `torch/utils/checkpoint.py:339-540`.
- REQ-2: `pub fn checkpoint_multi<T, F>(f: F, inputs: &[Tensor<T>])
  -> FerrotorchResult<Tensor<T>>` — multi-input variant. Same
  semantic as REQ-1 but `f` receives a slice of tensors; gradients
  are computed for every input with `requires_grad=true`.
- REQ-3: Save the autocast `(enabled, dtype)` snapshot at forward
  time (via `current_autocast_snapshot()`) and restore it during
  backward recomputation (via `with_autocast_state`). Without this,
  a checkpoint inside `autocast(F16, ...)` would recompute under f32
  and produce numerically inconsistent gradients.
- REQ-4: Save CPU RNG state at forward time and save CUDA RNG state
  for every distinct CUDA input device. Restore those states during
  backward recomputation so stochastic ops (dropout, rand, randn,
  rrelu training) produce identical masks/values across forward and
  recompute. Without this, the recompute can use different random
  values and produce incorrect gradients. Mirrors upstream
  `torch.get_rng_state`, `_get_device_states`, `torch.set_rng_state`,
  and `_set_device_states` in `torch/utils/checkpoint.py`.
- REQ-5: RAII guard on RNG state — the recompute path must NOT
  permanently rewind the global RNG to the saved-forward state; it
  must restore the caller's pre-backward RNG state on completion.
  Implemented via `CheckpointRngGuard`, with explicit error
  propagation on normal restore and best-effort `Drop` only for panic
  cleanup.
- REQ-6: `TensorId` aliasing invariant — `CheckpointBackward.input`
  stores a clone of the user's input. `Tensor::clone()` is an `Arc`
  clone, so the stored `Tensor` shares its `TensorId` with the
  user's. This is required so autograd's gradient-accumulation path
  writes back to the right leaf identity during recompute backward.
- REQ-7: Recompute backward uses the "weighted sum" trick: build
  `weighted = recomputed * detach(grad_output)`, then `scalar =
  sum(weighted)`, then `scalar.backward()`. This delivers the
  upstream gradient through chain rule onto the recomputed graph,
  yielding the input gradients via the cached graph's accumulation.
- REQ-8: Skip-attach fast path — when `!input.requires_grad()` (or
  no input requires grad in the multi-input case), return the
  forward output directly with NO `grad_fn` attached. This is the
  inference-time fast path matching upstream's
  `checkpoint(...).requires_grad` propagation.

## Acceptance Criteria

- [x] AC-1: Single-input checkpoint `f(x) = (x * x) + x` produces
  the correct forward AND correct backward partial `df/dx = 2x + 1`
  — `test_checkpoint_single_input_basic` at `checkpoint.rs:472`.
- [x] AC-2: When input does not require grad, no `grad_fn` is
  attached — `test_checkpoint_no_grad_input_returns_output_only` at
  `checkpoint.rs:497`.
- [x] AC-3: Multi-input checkpoint with both inputs requiring grad
  produces correct partials for `f(a, b) = a*b + a`:
  `df/da = b + 1`, `df/db = a` —
  `test_checkpoint_multi_two_inputs_both_grad` at
  `checkpoint.rs:520`.
- [x] AC-4: Multi-input partial-grad case (only second input
  requires grad) produces correct grad for that input —
  `test_checkpoint_multi_partial_grad` at `checkpoint.rs:552`.
- [x] AC-5: Single-input stochastic uniform checkpoint recomputes the
  exact forward random values during backward —
  `test_checkpoint_preserves_cpu_uniform_rng_for_recompute`.
- [x] AC-6: Single-input stochastic normal checkpoint preserves the
  full Box-Muller cached normal state, not just seed/counter —
  `test_checkpoint_preserves_cpu_normal_rng_cache_for_recompute`.
- [x] AC-7: Backward recomputation does not advance the caller's CPU
  RNG stream —
  `test_checkpoint_backward_does_not_advance_caller_cpu_rng_stream`.
- [x] AC-8: Multi-input stochastic checkpoint saves/restores the CPU
  RNG stream for recomputation —
  `test_checkpoint_multi_preserves_cpu_uniform_rng_for_recompute`.

## Architecture

### REQ-1 single-input `checkpoint`

`pub fn checkpoint<T, F>` at `checkpoint.rs:76-112`:

1. Capture `saved_rng` via `save_checkpoint_rng_state([input])`.
2. Capture `saved_autocast` via `current_autocast_snapshot()` at `:92`.
3. Forward pass in `no_grad`: `let output = no_grad(|| f(input))?;`
   at `:95`.
4. Fast path: if `!input.requires_grad()`, return `output` directly
   at `:97-99`.
5. Build `CheckpointBackward { func, input.clone(), output_shape,
   saved_rng, saved_autocast }`.
6. Wrap output's data in a fresh storage and attach the backward via
   `Tensor::from_operation` at `:110-111`.

### REQ-2 multi-input `checkpoint_multi`

`pub fn checkpoint_multi<T, F>` at `checkpoint.rs:121-148` is the
symmetric multi-input variant. Validates that at least one input is
provided. RNG state is saved using `save_checkpoint_rng_state(inputs)`,
which includes CPU state plus every distinct CUDA input device rather
than only the first input. Skip-attach if no input requires grad.

### REQ-3 autocast state preservation

`saved_autocast: AutocastSnapshot` field on both `CheckpointBackward`
(`CheckpointBackward in checkpoint.rs`) and `CheckpointMultiBackward` (`CheckpointMultiBackward in checkpoint.rs`). The backward
implementations at `:240` and `:312` wrap the recompute closure in
`with_autocast_state(self.saved_autocast, || ...)`. The
`with_autocast_state` RAII drop guard restores the caller's autocast
state on completion (success or panic).

### REQ-4 CPU/CUDA RNG preservation

`save_checkpoint_rng_state<T: Float>(tensors: &[Tensor<T>]) ->
FerrotorchResult<CheckpointRngState>` clones the current thread's CPU
`Generator` state, including cached normal samples, and captures a
`GpuRngState` for each distinct CUDA input device. CUDA save failures
propagate instead of being converted to `None`.

During backward, `CheckpointRngGuard::activate` snapshots the caller's
current CPU/CUDA states, restores the saved-forward states, and then
the backward recomputation runs under those states. After recompute,
`CheckpointRngGuard::restore` restores the caller states and propagates
restore failures. This mirrors PyTorch's fork-style checkpoint RNG
handling.

### REQ-5 RNG guard

`struct CheckpointRngGuard { previous: CheckpointRngState, restored:
bool }` restores the saved-forward RNG state for recomputation and
then restores the caller's previous RNG state. Normal control flow
calls `restore()` explicitly so errors are visible; `Drop` remains a
panic/unwind cleanup fallback.

### REQ-6 `TensorId` aliasing

`CheckpointBackward.input: Tensor<T>` at `CheckpointBackward in checkpoint.rs` is
populated by `input.clone()` at `input in checkpoint.rs`. The doc comment at
`:182-188` documents the invariant: `Tensor::clone()` is an `Arc`
clone, so `TensorId` is preserved. Autograd's gradient accumulation
keys by `TensorId`, so the cloned `input` and the user's original
input share identity — the gradient written to `input_with_grad.grad()`
inside the backward recompute is visible at the user's input tensor
via the same `TensorId`.

### REQ-7 weighted-sum recompute trick

Inside `CheckpointBackward::backward` at `checkpoint.rs:217-260`:

```rust
let input_with_grad = self.input.clone().requires_grad_(true);
let recomputed = (self.func)(&input_with_grad)?;
let weighted = mul(&recomputed, &grad_output.clone().requires_grad_(false).detach())?;
let scalar = sum(&weighted)?;
scalar.backward()?;
let input_grad = input_with_grad.grad()?;
Ok(vec![input_grad])
```

The `detach()` on `grad_output` (at `detach in checkpoint.rs`) is essential — without
it, the autograd engine would walk back through the grad_output's
own grad_fn and double-process the chain rule. The `requires_grad_(true)`
on the cloned input enables the freshly-recomputed graph to record
its own backward edges.

### REQ-8 skip-attach fast path

Implemented at `checkpoint.rs:97-99` (single-input) and `:140-142`
(multi-input). Required because attaching a `CheckpointBackward`
without a downstream consumer would still allocate the
`func: CheckpointFn<T>` Arc (a non-trivial allocation), and the
recompute would never fire anyway.

## Parity contract

`parity_ops = []` — checkpoint is graph-transformation plumbing.
Behavioral parity:

- Forward output is bit-identical to non-checkpointed forward (the
  same `f` runs in both cases).
- Backward gradients are bit-identical (modulo floating-point
  determinism) to non-checkpointed backward, ASSUMING:
  - CPU/CUDA RNG state was successfully saved + restored (REQ-4), and
  - autocast snapshot was successfully captured + restored (REQ-3).
- Caller's autocast and RNG state outside the backward call are
  unchanged (RAII restoration).
- Memory: peak GPU memory drops by the size of the activations
  inside `f`'s body, paid back by re-executing `f` once during
  backward (a 2x compute cost for `f`).

R-DEV-2 API parity: `checkpoint(f, *args)` in PyTorch maps to
`checkpoint(f, &input)` / `checkpoint_multi(f, &inputs)` in
ferrotorch (Rust closure signatures + slice rather than variadic
args).

## Verification

Tests in `checkpoint.rs:362-754` (~390 LOC of test code). Key tests:

- `test_checkpoint_single_input_basic` (`test_checkpoint_single_input_basic in checkpoint.rs`)
- `test_checkpoint_no_grad_input_returns_output_only` (`test_checkpoint_no_grad_input_returns_output_only in checkpoint.rs`)
- `test_checkpoint_multi_two_inputs_both_grad` (`test_checkpoint_multi_two_inputs_both_grad in checkpoint.rs`)
- `test_checkpoint_multi_partial_grad` (`test_checkpoint_multi_partial_grad in checkpoint.rs`)
- `test_checkpoint_preserves_cpu_uniform_rng_for_recompute`
- `test_checkpoint_preserves_cpu_normal_rng_cache_for_recompute`
- `test_checkpoint_backward_does_not_advance_caller_cpu_rng_stream`
- `test_checkpoint_multi_preserves_cpu_uniform_rng_for_recompute`
- Autocast preservation tests verify the snapshot round-trip across
  the forward/backward boundary using `is_autocast_enabled()`
  / `autocast_dtype()` assertions inside the closure (see test mod's
  use of `AutocastDtype, autocast, is_autocast_enabled` at
  `checkpoint.rs`).

All tests pass in the workspace gauntlet.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn checkpoint<T, F>` at `checkpoint in ferrotorch-core/src/autograd/checkpoint.rs` + `struct CheckpointBackward<T: Float>` at `CheckpointBackward in ferrotorch-core/src/autograd/checkpoint.rs` + `impl<T: Float> crate::tensor::GradFn<T> for CheckpointBackward<T>` at `struct in ferrotorch-core/src/autograd/checkpoint.rs`; mirrors `torch.utils.checkpoint.checkpoint` at `torch/utils/checkpoint.py:339-540`; non-test production consumer: `pub mod checkpoint` is the public sub-module of `autograd` (via `checkpoint in ferrotorch-core/src/autograd/mod.rs`); callers reach it as `crate::autograd::checkpoint::checkpoint` from any user code that wants to memory-trade. Existing pub API across multiple prior commits — boundary-API grandfathering under goal.md S5. |
| REQ-2 | SHIPPED | impl: `pub fn checkpoint_multi<T, F>` at `checkpoint_multi in checkpoint.rs` + `struct CheckpointMultiBackward<T: Float>` at `CheckpointMultiBackward in checkpoint.rs` + `impl GradFn` at `CheckpointMultiBackward in checkpoint.rs`; non-test production consumer: same as REQ-1 — exposed via `crate::autograd::checkpoint` sub-module. Existing pub API — boundary-API grandfathering. |
| REQ-3 | SHIPPED | impl: `saved_autocast: AutocastSnapshot` fields at `checkpoint.rs:298, :373`; `current_autocast_snapshot()` calls at `:92, :135`; `with_autocast_state(self.saved_autocast, || ...)` recompute wraps at `:329, :398`; non-test production consumer: every `checkpoint` / `checkpoint_multi` call (the autocast preservation is unconditional, not opt-in). |
| REQ-4 | SHIPPED | impl: `fn save_gpu_rng_state` at `save_gpu_rng_state in checkpoint.rs`; `saved_gpu_rng: Option<GpuRngState>` fields at `, `; RNG-restore calls at `, `; non-test production consumer: every checkpoint of a CUDA-resident tensor when a GPU backend is registered. |
| REQ-5 | SHIPPED | impl: `struct GpuRngGuard { previous: GpuRngState }` at `GpuRngGuard in checkpoint.rs` with `Drop` impl restoring `previous`; non-test production consumer: instantiated inside both backward impls at `, ` (the `let _rng_guard = ...` pattern); production callers are every backward of a checkpointed CUDA op. |
| REQ-6 | SHIPPED | impl: `CheckpointBackward.input: Tensor<T>` field at `CheckpointBackward in checkpoint.rs` populated by `input.clone()` at `input in checkpoint.rs` — Arc-clone preserves `TensorId`; the doc-comment at `input in checkpoint.rs` documents the invariant; non-test production consumer: every `checkpoint(f, &input)` call where the input requires grad — the `input_with_grad.grad()` read at `checkpoint in checkpoint.rs` relies on the `TensorId` aliasing to project the recompute's gradient back to the user's input tensor. |
| REQ-7 | SHIPPED | impl: weighted-sum trick at `checkpoint.rs:339-347` (`mul(&recomputed, &grad_output.detach())` + `sum(&weighted)` + `scalar.backward()` + `input_with_grad.grad()`); non-test production consumer: every backward call on a checkpointed output — invoked from `Tensor::backward` (REQ-1 of graph.md) → `CheckpointBackward::backward`. |
| REQ-8 | SHIPPED | impl: `if !input.requires_grad() { return Ok(output); }` at `checkpoint.rs:97-99` (single-input) and `if !any_requires_grad { return Ok(output); }` at `:140-142` (multi-input); non-test production consumer: any inference-time checkpoint call (a model whose forward path uses `checkpoint` for memory but whose inputs do not require grad). |
