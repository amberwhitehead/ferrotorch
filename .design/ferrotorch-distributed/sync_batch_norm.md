# Synchronized Batch Normalization (SyncBatchNorm)

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/batchnorm.py
  - torch/nn/modules/_functions.py
-->

## Summary

`ferrotorch-distributed/src/sync_batch_norm.rs` implements
`SyncBatchNorm2d<T>`, the distributed analog of
`ferrotorch_nn::BatchNorm2d` that synchronizes per-channel mean and
variance statistics across all ranks of a `Backend` so the
normalization sees the **global** batch statistics rather than the
per-rank local mini-batch. Mirrors `class SyncBatchNorm(_BatchNorm)`
at `torch/nn/modules/batchnorm.py:650` and the autograd-friendly
`class SyncBatchNorm(Function)` at `torch/nn/modules/_functions.py:7`.
Forward packs per-channel `sum` and `sum_sq` into a single
`[2*C]`-shaped tensor and does **one** allreduce to amortize the
communication cost. Backward packs `sum_dl_dx_hat` and
`sum_dl_dx_hat_x_hat` similarly and synchronizes them across ranks
so `grad_input` is consistent with the synchronized forward.
`grad_weight` and `grad_bias` are accumulated locally — the wrapping
DDP's gradient hook reduces them at the parameter-sync step.

## Requirements

- REQ-1: `pub struct SyncBatchNorm2d<T: Float>` with
  `#[non_exhaustive]` so future configuration knobs can be added
  without breaking external struct-literal construction. Fields:
  `num_features`, `eps`, `momentum`, `affine`, optional
  `weight`/`bias` `Parameter<T>`s, mutex-guarded `running_mean` /
  `running_var` / `num_batches_tracked` / `training`, and optional
  `backend: Option<Arc<dyn Backend>>`. The `backend = None` case
  reduces to plain `BatchNorm2d` semantics.
- REQ-2: `pub fn new(num_features, eps, momentum, affine) ->
  FerrotorchResult<Self>` constructs without backend (acts as plain
  BatchNorm2d). Validates `num_features > 0`. Affine weights are
  ones, biases zeros. Mirrors PyTorch's
  `SyncBatchNorm(num_features, eps=1e-5, momentum=0.1, affine=True,
  track_running_stats=True, process_group=None)`. R-DEV-2 deviation:
  `track_running_stats` is implicitly true (always on).
- REQ-3: `pub fn with_backend(self, backend: Arc<dyn Backend>) ->
  Self` attaches a backend for cross-rank synchronization. Builder-
  style chaining method.
- REQ-4: Accessors `pub fn running_mean`, `pub fn running_var`,
  `pub fn num_batches_tracked` return snapshots of the current
  running statistics for inspection / checkpointing.
- REQ-5: `impl Module<T> for SyncBatchNorm2d<T>` implements forward,
  parameters / parameters_mut / named_parameters,
  train / eval / is_training, so SyncBatchNorm2d can drop into any
  Module-consuming code.
- REQ-6: Forward (training): compute per-channel local `sum` and
  `sum_sq` over `B x H x W` elements; if `backend.world_size() > 1`,
  pack into `[2*C]` tensor and call
  `crate::collective::allreduce(&packed_t, backend, ReduceOp::Sum)`.
  Divide by `global_count = local_count * world_size` to get
  channel-wise mean and biased variance (`E[X²] - E[X]²`). Update
  running stats with Bessel correction (`bessel = N/(N-1)`).
- REQ-7: Forward (eval): bypass synchronization; use the stored
  `running_mean` and `running_var` for normalization regardless of
  whether a backend is attached.
- REQ-8: Forward normalizes via `(x - mean) * inv_std` where
  `inv_std = 1.0 / sqrt(var + eps)`. Optional affine: `out = normed
  * weight + bias`. If `is_grad_enabled() && input.requires_grad()`,
  the normalized intermediate `x_hat` is retained for the backward
  pass. Builds a `Tensor::from_operation` with a
  `SyncBatchNorm2dBackward` GradFn when grad-capture is needed,
  otherwise returns a `from_storage` tensor.
- REQ-9: `struct SyncBatchNorm2dBackward<T: Float> impl GradFn<T>`
  computes `grad_input`, `grad_weight`, `grad_bias`. Per-channel
  packs `sum_dl_dx_hat` and `sum_dl_dx_hat_x_hat` into a single
  `[2*C]` tensor and one allreduce. `grad_input` uses the
  synchronized means in the standard BatchNorm VJP formula so it's
  consistent with the synchronized forward. `grad_weight` and
  `grad_bias` accumulate locally per-rank — DDP at the next
  gradient-sync step sums them across ranks (this matches PyTorch
  SyncBatchNorm semantics; see the module doc-comment).
- REQ-10: Shape and device validation: forward errors on non-4D
  input (`[B, C, H, W]`) and on `channels != num_features`. CUDA
  input errors with `NotImplementedOnCuda` for both forward and
  backward (the CPU path is the only supported path; a GPU kernel
  is a separate follow-up).

## Acceptance Criteria

- [x] AC-1: With `world_size = 1` (no backend), `SyncBatchNorm2d`
  output matches plain `BatchNorm2d` element-for-element.
- [x] AC-2: 2-rank `SyncBatchNorm2d` on split halves of a 4-element
  batch produces the same per-rank statistics AND the same
  concatenated output as a 1-rank `BatchNorm2d` on the full batch.
- [x] AC-3: Both ranks see identical `running_mean` and `running_var`
  after the forward, and those match the 1-rank full-batch running
  statistics.
- [x] AC-4: After warming up running stats in train mode, eval mode
  produces deterministic output regardless of the eval input
  distribution.
- [x] AC-5: Constructing with `num_features = 0` errors.
- [x] AC-6: Forward on a non-4D tensor errors.
- [x] AC-7: Forward with wrong channel count errors.

## Architecture

### Struct (REQ-1)

`pub struct SyncBatchNorm2d<T>` (in `sync_batch_norm.rs`) is
`#[non_exhaustive]` so future fields (running-stats sync mode,
momentum schedule, fused fwd/bwd flags) can land without breaking
external code. Stat fields are `Mutex`-guarded so the same instance
can be shared across forward-pass threads in a multi-rank simulated
setup. The optional `backend: Option<Arc<dyn Backend>>` cleanly
expresses the "no synchronization" case (no backend ↔ plain
BatchNorm2d).

A custom `Debug` impl hides the inner mutex contents and reports
`num_features`, `eps`, `momentum`, `affine`, `world_size` (1 if no
backend), `training`.

### Construction (REQ-2, REQ-3)

`pub fn new` (in `sync_batch_norm.rs`) validates `num_features > 0`,
constructs the affine weights (ones / zeros) only when `affine`,
and initializes running stats to `0.0` / `1.0` (matching PyTorch's
init). `pub fn with_backend(self, backend)` is a builder-style
setter that attaches the backend post-construction.

### Accessors (REQ-4)

`pub fn running_mean()` / `running_var()` / `num_batches_tracked()`
return cloned snapshots. The `Mutex::lock().unwrap()` calls assume
no thread panic during stat update; the only writer is `forward`
which is sequential per instance.

### Forward training (REQ-6, REQ-8)

`forward` (in `sync_batch_norm.rs`):

1. Validate shape `[B, C, H, W]` and `C == num_features`.
2. CPU-only: error on CUDA input.
3. Read input data and weight/bias data.
4. If training:
   - Compute per-channel `sum` and `sum_sq` over `B*H*W` elements.
   - If `backend.world_size() > 1`: pack into `[2*C]`, allreduce
     (Sum), unpack back into `sum` / `sum_sq`.
   - Compute `mean = sum / global_count`,
     `var = sum_sq / global_count - mean^2`.
   - Update running stats with Bessel correction.
5. If eval (REQ-7): copy `running_mean` / `running_var` into the
   per-channel buffers.
6. Compute `inv_std = 1 / sqrt(var + eps)` per channel.
7. For each pixel: `normed = (x - mean) * inv_std`; optionally
   apply affine.
8. If `is_grad_enabled() && input.requires_grad()`: retain `x_hat`
   in a `Vec<T>` and build a `SyncBatchNorm2dBackward` GradFn;
   return via `Tensor::from_operation`. Else return via
   `Tensor::from_storage` (no autograd hookup).

The `#[allow(clippy::manual_memcpy)]` attribute on `forward` is
documented inline — the per-channel unpack of `packed[2*c]` ↔ `sum`
+ `sum_sq` is not a slice-memcpy due to the de-interleaving.

### Backward (REQ-9)

`struct SyncBatchNorm2dBackward<T> impl GradFn<T>` (in
`sync_batch_norm.rs`) stores the inputs needed for the VJP:
`input`, `x_hat`, optional `weight` / `bias`, `chan_var` (per-channel
variance, captured as `f64` for numerical stability), `eps`,
`affine`, `global_count`, optional `backend`.

`fn backward(grad_output)`:

1. CPU-only: error on CUDA.
2. First pass: per-channel local sums of `dl/dx_hat` and
   `dl/dx_hat * x_hat`; also accumulate `grad_weight` and
   `grad_bias` if affine.
3. If `backend.world_size() > 1`: pack the two `[C]` sum vectors
   into `[2*C]` and allreduce (Sum).
4. Second pass: compute `grad_input[idx] = inv_std * (dl_dx_hat -
   dl_dx_hat_mean - x_h * dl_dx_hat_x_hat_mean)` using the
   synchronized means. This is the standard BatchNorm VJP formula
   applied with global statistics.
5. Return `vec![Some(grad_input), grad_weight_out, grad_bias_out]`
   (the latter two are `None` when not affine or when the parameter
   doesn't require_grad).

### Validation (REQ-10)

The 4-D check and channel-count check are at the top of `forward`.
CUDA inputs produce `FerrotorchError::NotImplementedOnCuda` (a
distinct error variant so callers can route to a CPU fallback if
desired).

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/lib.rs` `pub use
  sync_batch_norm::SyncBatchNorm2d;` re-exports the surface. Reached
  via `ferrotorch/src/lib.rs` `pub use ferrotorch_distributed::*;`
  for user code.
- Within `sync_batch_norm.rs`, `forward` is a production consumer
  of `crate::collective::allreduce` (REQ-3 of collective.md); the
  `SyncBatchNorm2dBackward::backward` is the same.

## Parity contract

No parity-sweep ops in the route (`parity_ops = []`). The contract is
the PyTorch SyncBatchNorm shape:

- `class SyncBatchNorm(_BatchNorm)` at
  `torch/nn/modules/batchnorm.py:650` — public API: 5-D tensor
  support, 1-D / 2-D / 3-D normalization. Ferrotorch ships **only
  the 4-D / spatial-2D case** (`SyncBatchNorm2d`), matching the
  most common detection/segmentation use case. The upstream's `1d`
  / `3d` variants are not yet shipped (separate follow-up).
- `class SyncBatchNorm(Function)` at
  `torch/nn/modules/_functions.py:7` — the autograd op that
  performs the cross-rank sync. Ferrotorch's
  `SyncBatchNorm2dBackward` (a `GradFn` impl) is the analog.
  R-DEV-1 (numerical-contract match): biased variance estimator
  (`E[X²] - E[X]²`) matches PyTorch's `batch_norm_stats`. Bessel
  correction is applied only to the running variance update,
  matching upstream.
- `convert_sync_batchnorm` helper (a class method on
  `nn.SyncBatchNorm`) is NOT shipped in ferrotorch (R-DEV-5
  candidate: a typestate-driven conversion would be a separate
  utility; deferring until a need arises in downstream model
  crates).
- `process_group` kwarg → `Option<Arc<dyn Backend>>` field.
  R-DEV-4 deviation: ferrotorch uses `Arc<dyn Backend>` rather than
  a global registry of process groups.

## Verification

`cargo test -p ferrotorch-distributed --lib sync_batch_norm::` runs
the `#[cfg(test)] mod tests` block at lines 571-745 covering 5
tests:

- `test_sync_bn_world_size_1_matches_batch_norm` — no-backend
  SyncBatchNorm2d matches plain BatchNorm2d.
- `test_sync_bn_two_ranks_match_full_batch` — 2-rank sync of split
  halves matches 1-rank full batch (both outputs AND running stats).
- `test_sync_bn_eval_mode_uses_running_stats` — after train-mode
  warmup, eval mode produces deterministic output independent of
  eval input distribution.
- `test_sync_bn_constructor_validates_num_features` — zero
  num_features errors.
- `test_sync_bn_rejects_wrong_input_shape` — non-4D input errors.
- `test_sync_bn_rejects_wrong_channel_count` — mismatched channel
  count errors.

Lint: `cargo clippy -p ferrotorch-distributed -- -D warnings` clean.
The 2 `#[allow(clippy::manual_memcpy)]` attributes (forward and
backward) are documented inline — the packed-allreduce unpack is
not expressible as a memcpy.
Parity-sweep: no ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct SyncBatchNorm2d` (with `#[non_exhaustive]`) in `ferrotorch-distributed/src/sync_batch_norm.rs`; non-test consumer: `ferrotorch-distributed/src/lib.rs` `pub use sync_batch_norm::SyncBatchNorm2d;` → `ferrotorch/src/lib.rs`. |
| REQ-2 | SHIPPED | impl: `pub fn new` in `ferrotorch-distributed/src/sync_batch_norm.rs`; non-test consumer: `lib.rs` re-export of `SyncBatchNorm2d` — `new` is the constructor. |
| REQ-3 | SHIPPED | impl: `pub fn with_backend` in `ferrotorch-distributed/src/sync_batch_norm.rs`; non-test consumer: `lib.rs` re-export of `SyncBatchNorm2d`. |
| REQ-4 | SHIPPED | impl: `pub fn running_mean`, `pub fn running_var`, `pub fn num_batches_tracked` in `ferrotorch-distributed/src/sync_batch_norm.rs`; non-test consumer: surfaced via `lib.rs` re-export; the accessors are the user-facing way to read running stats for checkpointing. |
| REQ-5 | SHIPPED | impl: `impl Module<T> for SyncBatchNorm2d<T>` in `ferrotorch-distributed/src/sync_batch_norm.rs`; non-test consumer: the `Module` trait impl makes `SyncBatchNorm2d` drop-in wherever `impl Module<T>` is accepted, reachable via `lib.rs`. |
| REQ-6 | SHIPPED | impl: `forward` (training arm) in `ferrotorch-distributed/src/sync_batch_norm.rs`; non-test consumer: invokes `crate::collective::allreduce`; surfaced via `lib.rs` re-export. |
| REQ-7 | SHIPPED | impl: `forward` (eval arm) in `ferrotorch-distributed/src/sync_batch_norm.rs`; non-test consumer: surfaced via `lib.rs` re-export. |
| REQ-8 | SHIPPED | impl: `forward` (normalize / autograd hookup) in `ferrotorch-distributed/src/sync_batch_norm.rs`; non-test consumer: `Tensor::from_operation` (autograd machinery) registers `SyncBatchNorm2dBackward` for the backward pass; surfaced via `lib.rs` re-export. |
| REQ-9 | SHIPPED | impl: `struct SyncBatchNorm2dBackward<T>` and `impl GradFn<T>` in `ferrotorch-distributed/src/sync_batch_norm.rs`; non-test consumer: instantiated by `forward` (same file) and registered on the autograd graph via `Tensor::from_operation`. The autograd engine in `ferrotorch_core` is the runtime consumer. |
| REQ-10 | SHIPPED | impl: shape / channel / CUDA validation in `forward` and `SyncBatchNorm2dBackward::backward` in `ferrotorch-distributed/src/sync_batch_norm.rs`; non-test consumer: surfaced via `lib.rs` re-export of `SyncBatchNorm2d`. |
