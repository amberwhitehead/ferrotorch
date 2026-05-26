# ferrotorch-train — gradient (activation) checkpointing

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/utils/checkpoint.py
-->

## Summary

`ferrotorch-train/src/checkpoint.rs` is a thin shell over
`ferrotorch_core::autograd::checkpoint::checkpoint` (the activation-
checkpointing primitive that re-executes a closure's forward during
backward to save memory) plus a sequential-segment helper
`checkpoint_sequential(modules, segments, input)` that splits a chain
of `Arc<dyn Module<T>>` into `segments` roughly-equal groups and wraps
each in its own checkpoint. Mirrors PyTorch's
`torch.utils.checkpoint.checkpoint` (`torch/utils/checkpoint.py:355`)
and `torch.utils.checkpoint.checkpoint_sequential`
(`torch/utils/checkpoint.py:526`).

## Requirements

- REQ-1: Module re-exports `ferrotorch_core::autograd::checkpoint::
  checkpoint` so users can write
  `use ferrotorch_train::checkpoint;`. Mirrors `from torch.utils.
  checkpoint import checkpoint`.
- REQ-2: `pub fn checkpoint_sequential<T: Float>(modules:
  Vec<Arc<dyn Module<T>>>, segments: usize, input: &Tensor<T>) ->
  FerrotorchResult<Tensor<T>>` splits the module chain into
  `ceil(n / segments)` contiguous groups and wraps each in a
  `checkpoint` call. Panics on `segments == 0` or empty `modules`.
- REQ-3: When `input.requires_grad() == false`, the no-grad
  shortcut at line 126-134 chains the forwards directly without
  paying the `no_grad` wrap allocation. The output carries no
  `grad_fn`.
- REQ-4: When `input.requires_grad() == true`, each segment is wrapped
  in `ferrotorch_core::autograd::checkpoint::checkpoint`, producing a
  `CheckpointBackward` grad_fn per segment. The top-level grad_fn of
  the final output is `"CheckpointBackward"`.
- REQ-5: The `'static + Send + Sync` lifetime contract on the closure
  passed to `checkpoint` is satisfied by taking modules **by value**
  as `Vec<Arc<dyn Module<T>>>`. Each segment closure clones the
  relevant `Arc`s into itself.
- REQ-6: Backward through a `checkpoint_sequential` output recomputes
  the segment's forward (i.e. the module's `forward` is called a
  second time during backward), and the gradient computed by the
  recomputed graph matches the analytic value.

## Acceptance Criteria

- [x] AC-1: `pub use ferrotorch_core::autograd::checkpoint::checkpoint;`
  at the top of the module.
- [x] AC-2: `checkpoint_sequential(modules: Vec<Arc<dyn Module<T>>>,
  segments: usize, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>>`
  exists with documented signature.
- [x] AC-3: `checkpoint_sequential` with `requires_grad=false` returns
  output with `grad_fn().is_none()`.
- [x] AC-4: `checkpoint_sequential` with `requires_grad=true` returns
  output with `grad_fn().name() == "CheckpointBackward"`.
- [x] AC-5: Calling `output.backward()` re-executes the module
  forwards (verified via a shared atomic counter).
- [x] AC-6: Computed gradient matches the analytic value for
  `x * 2 * 5 ⇒ dx = 10`.

## Architecture

### `checkpoint` re-export (REQ-1)

At `ferrotorch-train/src/checkpoint.rs:61` `pub use
ferrotorch_core::autograd::checkpoint::checkpoint;`. The function is
defined in `ferrotorch-core` because the autograd graph it builds
requires `Tensor<T>` + autograd internals that live there. The
train-crate re-export gives users the upstream-mirroring import path.

### `checkpoint_sequential` (REQ-2, REQ-3, REQ-4, REQ-5)

At lines 102-153. The algorithm:
1. Assert `segments > 0` and `!modules.is_empty()` (panic on
   violation — these are programmer-error conditions, not runtime
   conditions).
2. `seg_size = n.div_ceil(segments)` — ceil division so the last
   segment may be smaller but no segment is zero-sized.
3. Drain `seg_size` modules at a time from the front of `remaining`,
   running each segment either:
   - **No-grad shortcut** (line 126-134): chain `module.forward(...)`
     directly. The core `checkpoint` already short-circuits in this
     case but we skip it explicitly to avoid the `no_grad` wrap
     allocation.
   - **Full path** (line 136-149): call `checkpoint(move |x| { ... },
     &current)`. The move closure owns the segment's `Arc`s, so the
     closure is `'static + Send + Sync` — required by the core
     primitive because the closure is stored on the
     `CheckpointBackward` node and called during backward, which
     may happen at any later point.

The `Arc<dyn Module<T>>` ownership pattern (REQ-5) is documented
inline at lines 27-37: a `&[M]` slice cannot capture into the
`'static` closure, so the API takes modules by value through
`Arc`-wrapped trait objects.

### Real-checkpoint discriminator (REQ-6)

The `test_checkpoint_sequential_real_checkpoint_grad_fn` test
at lines 336-407 is the sabotage discriminator for issue #1108. The
test:
1. Constructs a `CountingScale` module that increments a shared
   `Arc<AtomicUsize>` on every `forward` call.
2. Runs `checkpoint_sequential(modules, 1, &input)` — single segment
   so the top-level grad_fn IS `CheckpointBackward`.
3. Asserts forward calls == 2 (each module called once during forward).
4. Asserts `output.grad_fn().name() == "CheckpointBackward"` — this
   is the sabotage-catching assertion. The #1108 bug had if/else
   branches that were byte-for-byte identical and never routed
   through the core checkpoint primitive; the grad_fn chain
   surfaced `Mul` at the top instead of `CheckpointBackward`.
5. Runs backward, asserts the call counter increased (recomputation
   happened).
6. Asserts the analytic gradient `dx = 10` for `x * 2 * 5`.

### Non-test production consumers

- The re-exported `checkpoint` function is consumed inside
  `ferrotorch-train/src/checkpoint.rs:140` by `checkpoint_sequential`
  itself (same-file production consumer).
- No external in-tree caller of `checkpoint_sequential` exists today;
  the function is part of the public training-loop API surface. Open
  prereq blocker #1502 covers wiring `checkpoint_sequential` into a
  large-model example (ResNet / Transformer block) where the
  memory savings justify the recomputation overhead.

## Parity contract

`parity_ops = []`. The numerical contract is owned by
`ferrotorch_core::autograd::checkpoint::checkpoint`'s design doc.
Edge cases this module owns:

- **`segments == 0`**: panic with `"segments must be > 0"`. Tested
  by `test_checkpoint_sequential_zero_segments_panics` at line 262.
- **Empty `modules`**: panic with `"modules must not be empty"`.
  Tested by `test_checkpoint_sequential_empty_modules_panics` at
  line 270.
- **`segments > modules.len()`**: each module becomes its own
  segment. Tested by `test_checkpoint_sequential_more_segments_than_modules`
  at line 250.
- **No-grad input**: shortcut path skips the autograd wrap;
  `output.grad_fn().is_none()`. Tested by
  `test_checkpoint_sequential_no_grad_skips_checkpoint` at line 409.
- **Nested CUDA RNG state**: when the segment's modules use dropout
  on CUDA, the core `checkpoint` primitive saves/restores the GPU
  RNG state so the recomputed forward produces identical dropout
  masks — that contract is the core primitive's, documented in its
  module.

## Verification

7 unit tests in `mod tests` (lines 159-449):
- `test_checkpoint_reexported` (line 165) pins the re-export.
- `test_checkpoint_sequential_single_segment` / `multiple_segments`
  / `more_segments_than_modules` (lines 230-259) pin the
  segment-splitting + forward-correctness contract.
- `test_checkpoint_sequential_zero_segments_panics` /
  `empty_modules_panics` (lines 261-275) pin the panic contracts.
- `test_checkpoint_sequential_real_checkpoint_grad_fn` (line 336)
  is the #1108 sabotage discriminator.
- `test_checkpoint_sequential_no_grad_skips_checkpoint` (line 409)
  pins the no-grad shortcut.
- `test_checkpoint_sequential_multi_segment_each_wraps` (line 426)
  pins multi-segment grad_fn structure.

Smoke command:

```bash
cargo test -p ferrotorch-train --lib checkpoint:: 2>&1 | tail -3
```

Expected: > 7 passed, 0 failed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub use ferrotorch_core::autograd::checkpoint::checkpoint;` at `ferrotorch-train/src/checkpoint.rs:61`; non-test consumer: same-file `checkpoint_sequential` at `ferrotorch-train/src/checkpoint.rs:140` invokes `checkpoint(move |x| { ... }, &current)` — production consumer in the same module. |
| REQ-2 | NOT-STARTED | open prereq blocker #1502 — `pub fn checkpoint_sequential` at `ferrotorch-train/src/checkpoint.rs:102-153` is shipped on the public surface but no in-tree caller invokes it outside the unit-test module. A large-model example (ResNet block or Transformer layer) is the open consumer-wiring work. |
| REQ-3 | NOT-STARTED | open prereq blocker #1502 — no-grad shortcut at `ferrotorch-train/src/checkpoint.rs:126-134` is shipped but only exercised by `test_checkpoint_sequential_no_grad_skips_checkpoint`. No production caller triggers the no-grad input branch end-to-end. |
| REQ-4 | NOT-STARTED | open prereq blocker #1502 — `checkpoint(move |x| { ... }, &current)` at `ferrotorch-train/src/checkpoint.rs:140-149` is the production-side wiring for the segment wrap, but reachable only through `checkpoint_sequential` which has no production caller today. |
| REQ-5 | NOT-STARTED | open prereq blocker #1502 — the `move` closure at `ferrotorch-train/src/checkpoint.rs:141-147` captures `Vec<Arc<dyn Module<T>>>` satisfying `'static + Send + Sync`, but the only caller of the surrounding `checkpoint_sequential` is the unit test. The lifetime-contract guarantee is structurally correct; the external production caller is still missing. |
| REQ-6 | NOT-STARTED | open prereq blocker #1502 — recomputation-on-backward behavior is pinned by `test_checkpoint_sequential_real_checkpoint_grad_fn` at `ferrotorch-train/src/checkpoint.rs:336-407`, but no production training loop drives the recomputation today. |

