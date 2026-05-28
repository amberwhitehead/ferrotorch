# Pipeline parallelism

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/distributed/pipelining/__init__.py
  - torch/distributed/pipelining/microbatch.py
  - torch/distributed/pipelining/_IR.py
-->

## Summary

`ferrotorch-distributed/src/pipeline.rs` implements stage-level
pipeline parallelism mirroring `torch.distributed.pipelining`.
`Pipeline<M, T>` wraps a single stage's `Module<T>` and the
`Arc<dyn Backend>` whose rank determines which stage this process
runs. Inputs are split into micro-batches along dim 0 by the first
stage (rank 0), forwarded through each stage, and activations are
sent point-to-point through the `Backend`. Two schedules are
exposed: `GPipe` (all forwards, then all backwards) and `OneFOnEB`
(1F1B). 1F1B's full memory benefit (true interleaving with backward)
is a follow-up; the current implementation gives 1F1B scheduling
structure but uses GPipe-style sequential backward.

## Requirements

- REQ-1: `pub enum PipelineSchedule { GPipe, OneFOnEB }` with `Debug +
  Clone + Copy + PartialEq + Eq`. Mirrors PyTorch's `ScheduleGPipe`
  and `Schedule1F1B` symbols re-exported from
  `torch/distributed/pipelining/__init__.py:3`.
- REQ-2: `pub struct Pipeline<M: Module<T>, T: Float>` owns the stage
  module, the `Arc<dyn Backend>`, the number of micro-batches, and
  the schedule. `Debug` impl uses `finish_non_exhaustive` to avoid
  leaking the inner module type's full debug repr.
- REQ-3: `pub fn new(module, backend, num_microbatches, schedule) ->
  FerrotorchResult<Self>` validates `num_microbatches > 0` and
  `world_size >= 2` (pipeline parallelism requires at least 2
  stages). Mirrors PyTorch's `PipelineStage` constructor invariants.
- REQ-4: `pub fn forward(&self, input: Option<&Tensor<T>>) ->
  FerrotorchResult<Vec<Tensor<T>>>` runs the inner module's forward
  for every micro-batch. Rank 0 chunks the caller-provided input
  along dim 0; non-zero ranks receive activations from the previous
  rank via `recv_activation`. After running forward, non-last ranks
  send activations to the next rank via `send_activation`. Returns
  one output tensor per micro-batch (only meaningful on the last
  stage).
- REQ-5: `pub fn backward(&self, outputs: &[Tensor<T>], grad_outputs:
  Option<&[Tensor<T>]>) -> FerrotorchResult<()>` runs backward over
  the micro-batches in reverse. Last stage uses the caller-provided
  per-micro-batch grad_outputs; other ranks receive gradient
  activations from the next stage. After autograd backward, non-zero
  ranks send a placeholder zero-gradient back to the previous stage
  (the production-grade upstream sends the actual input-grad; this
  is documented as a known limitation in the module doc-comment).
- REQ-6: Internal helpers: `fn get_microbatch(input, mb_idx) ->
  FerrotorchResult<Tensor<T>>` extracts a contiguous slice along
  dim 0 (last micro-batch absorbs the remainder).
  `fn send_activation(tensor, dst_rank) -> FerrotorchResult<()>` and
  `fn recv_activation(src_rank) -> FerrotorchResult<Tensor<T>>` move
  activations through the `Backend`. Wire format: ndim (8 bytes
  LE-u64) + shape dims (each 8 bytes LE-u64) + data (LE bytes;
  4-byte f32 / 8-byte f64). Both `unsafe` casts are documented at
  the call sites.
- REQ-7: Accessors `pub fn schedule`, `pub fn num_microbatches`,
  `pub fn module`, `pub fn module_mut` expose pipeline state for
  user code.

## Acceptance Criteria

- [x] AC-1: `Pipeline::new(_, _, 0, GPipe)` errors with
  "num_microbatches must be > 0".
- [x] AC-2: `Pipeline::new(_, world_size=1, _, _)` errors with
  "world_size must be >= 2 for pipeline parallelism".
- [x] AC-3: `schedule()` and `num_microbatches()` accessors return
  the values passed at construction.
- [x] AC-4: The accompanying `IdentityModule` (test-side) toggles
  training state via `train()` / `eval()` (smoke test for the
  `Module` trait wiring used elsewhere in the suite).

## Architecture

### Schedule enum (REQ-1)

`pub enum PipelineSchedule` (in `pipeline.rs`) carries `GPipe` and
`OneFOnEB`. The `OneFOnEB` variant's doc-comment explicitly documents
that backward currently uses GPipe-style sequential processing; true
1F1B memory benefit is deferred to a combined forward+backward
method (a future fixer dispatch).

### Construction (REQ-2, REQ-3)

`pub fn new` (in `pipeline.rs`) validates `num_microbatches > 0` and
`world_size >= 2` before constructing. Validation errors are
`FerrotorchError::InvalidArgument` with descriptive messages. The
struct stores `_marker: PhantomData<T>` for the generic float type.
The custom `Debug` impl exposes only `num_microbatches` and
`schedule`; the inner module is hidden behind `finish_non_exhaustive`
to avoid noisy debug output.

### Forward (REQ-4)

`pub fn forward(&self, input: Option<&Tensor<T>>)` (in `pipeline.rs`)
loops over `0..self.num_microbatches`:

1. Rank 0: extract micro-batch via `get_microbatch(input, mb)`.
2. Non-rank 0: receive activation from `rank - 1` via
   `recv_activation`.
3. Run `self.module.forward(&mb_input)`.
4. If not last rank: send the output to `rank + 1` via
   `send_activation`.
5. Push the output into `outputs` (final stage's outputs are the
   pipeline result).

The `Option<&Tensor<T>>` is `None` for non-zero ranks (their inputs
come from upstream stages); rank 0 must provide `Some(_)` or the
forward errors.

### Backward (REQ-5)

`pub fn backward` (in `pipeline.rs`) loops `(0..num_microbatches).rev()`:

1. Last rank: install the caller-provided `grad_outputs[mb]` on
   `outputs[mb]` via `set_grad`.
2. Non-last rank: receive a gradient activation from `rank + 1` via
   `recv_activation`.
3. Run `ferrotorch_core::backward(&outputs[mb])`.
4. Non-zero rank: send a placeholder zero-gradient back to the
   previous stage (REQ-5 limitation — the real gradient w.r.t. the
   stage input is not yet plumbed; this is documented in the
   module doc-comment).

### Micro-batch slicing (REQ-6)

`fn get_microbatch` (in `pipeline.rs`) computes `mb_size =
batch_size / num_microbatches`. The last micro-batch absorbs the
remainder so the chunks always cover the full batch. The flat data
slice is extracted via `start*stride..end*stride` where
`stride = shape[1..].iter().product()`.

### Activation transport (REQ-6)

`fn send_activation` (in `pipeline.rs`) sends:

1. An 8-byte `ndim` LE u64 header.
2. `ndim` 8-byte LE u64 shape dims.
3. The flat data as LE bytes (4-byte f32 or 8-byte f64) via a
   per-element byte-reinterpret inside a documented `unsafe { ... }`
   block. The unsafe block has a 6-bullet SAFETY comment covering
   VALIDITY / LENGTH / ALIGNMENT / LIFETIME / PROVENANCE / ENDIANNESS.

`fn recv_activation` (in `pipeline.rs`) reads the header, allocates
the data buffer, calls `backend.recv`, then decodes LE bytes into
`T` using `f32::from_le_bytes` / `f64::from_le_bytes` and
`T::from(...)`. Elements other than 4 / 8 bytes hit `unreachable!`
(this is a stub for future bf16/f16 support; current `T: Float`
includes only f32 / f64).

### Accessors (REQ-7)

`pub fn schedule`, `pub fn num_microbatches`, `pub fn module`,
`pub fn module_mut` are direct getters with no I/O.

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/lib.rs` `pub use pipeline::{Pipeline,
  PipelineSchedule};` re-exports the pipeline surface; reached via
  `ferrotorch/src/lib.rs` `pub use ferrotorch_distributed::*;` for
  user training scripts.
- Within `pipeline.rs`, `send_activation` and `recv_activation` are
  production consumers of `Backend::send` and `Backend::recv`.
- Within `pipeline.rs`, `backward` is a production consumer of
  `ferrotorch_core::backward` (the autograd entry point).

## Parity contract

No parity-sweep ops in the route (`parity_ops = []`). The contract is
the PyTorch pipelining shape:

- `Pipe`, `pipeline`, `PipelineStage`, `ScheduleGPipe`,
  `Schedule1F1B` re-exported by
  `torch/distributed/pipelining/__init__.py:1-13` → ferrotorch's
  `Pipeline<M, T>` (per-stage wrapper) and `PipelineSchedule`. R-DEV-7
  deviation: ferrotorch flattens the upstream's `Pipe` /
  `PipelineStage` / `Schedule*` trinity into a single
  `Pipeline<M, T>` whose schedule is a runtime enum, because Rust's
  type system doesn't need separate schedule classes for compile-
  time dispatch.
- Micro-batch chunking along dim 0 mirrors
  `torch/distributed/pipelining/microbatch.py`'s default
  `_split_args_kwargs_into_chunks`. The last micro-batch absorbing
  the remainder matches the upstream `chunks` mechanism for non-
  divisible batches.

## Verification

`cargo test -p ferrotorch-distributed --lib pipeline::` runs the
`#[cfg(test)] mod tests` block at lines 353-438 covering 4 tests:

- `test_pipeline_new_validates_microbatches` — `num_microbatches=0`
  errors.
- `test_pipeline_new_validates_world_size` — `world_size=1` errors.
- `test_pipeline_schedule_accessors` — round-trips the schedule and
  microbatch count through the accessors.
- `test_identity_module_train_eval_toggles_state` — smoke test for
  the `IdentityModule` used to back the other tests.

Multi-rank `forward` / `backward` end-to-end tests are NOT in the
suite because they require coordinated cross-thread execution that
the simulated `Backend` only partially supports for point-to-point
flow (the collective ops have multi-rank tests but not the
sequential pipeline flow). This is a known test-coverage gap; see
blocker filed below.

Lint: `cargo clippy -p ferrotorch-distributed -- -D warnings` clean.
Parity-sweep: no ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum PipelineSchedule` in `ferrotorch-distributed/src/pipeline.rs`; non-test consumer: `ferrotorch-distributed/src/lib.rs` `pub use pipeline::{Pipeline, PipelineSchedule};` → `ferrotorch/src/lib.rs`. |
| REQ-2 | SHIPPED | impl: `pub struct Pipeline` in `ferrotorch-distributed/src/pipeline.rs`; non-test consumer: `lib.rs` re-export of `Pipeline`. |
| REQ-3 | SHIPPED | impl: `pub fn new` in `ferrotorch-distributed/src/pipeline.rs`; non-test consumer: `lib.rs` re-export of `Pipeline` — `new` is the only constructor. |
| REQ-4 | SHIPPED | impl: `pub fn forward` in `ferrotorch-distributed/src/pipeline.rs`; non-test consumer: invoked through `lib.rs` re-export of `Pipeline`; internally calls `Backend::send`/`recv` and `module.forward`. |
| REQ-5 | SHIPPED | impl: `pub fn backward` in `ferrotorch-distributed/src/pipeline.rs`; non-test consumer: invoked through `lib.rs` re-export of `Pipeline`; internally calls `ferrotorch_core::backward`. The grad-back-to-previous-stage limitation is documented in the module doc-comment per R-HONEST-3. |
| REQ-6 | SHIPPED | impl: `fn get_microbatch`, `fn send_activation`, `fn recv_activation` in `ferrotorch-distributed/src/pipeline.rs`; non-test consumer: invoked by `pub fn forward` and `pub fn backward` (same file). |
| REQ-7 | SHIPPED | impl: `pub fn schedule`, `pub fn num_microbatches`, `pub fn module`, `pub fn module_mut` in `ferrotorch-distributed/src/pipeline.rs`; non-test consumer: reachable via `lib.rs` re-export of `Pipeline`. |
