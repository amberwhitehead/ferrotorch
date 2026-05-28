# FullyShardedDataParallel (FSDP)

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/distributed/fsdp/fully_sharded_data_parallel.py
  - torch/distributed/fsdp/api.py
-->

## Summary

`ferrotorch-distributed/src/fsdp.rs` implements a Rust analog of
`torch.distributed.fsdp.FullyShardedDataParallel`. `FSDP<M, T>` wraps a
`Module<T>` and shards its parameters across the ranks of a
`Backend` so that each rank only stores `1 / world_size` of every
parameter tensor. Forward all-gathers the shards to reconstruct full
parameters, runs the inner module, then optionally re-installs the
shard. Backward fills gradients on the full tensors; `sync_gradients`
reduce-scatters the gradient back to per-rank shards for the
optimizer. Four sharding strategies mirror the upstream
`ShardingStrategy` enum: `FullShard` (ZeRO-3), `ShardGradOp` (ZeRO-2),
`NoShard` (ZeRO-0 / DDP-equivalent), and `HybridShard` (intra-node
FullShard + inter-node DDP).

## Requirements

- REQ-1: `pub enum ShardingStrategy { FullShard, ShardGradOp, NoShard,
  HybridShard { intra_node_size: usize } }` with `Debug + Clone + Copy
  + PartialEq + Eq + Default`. Mirrors
  `torch/distributed/fsdp/api.py:32` `class ShardingStrategy(Enum)`
  with members `FULL_SHARD`, `SHARD_GRAD_OP`, `NO_SHARD`,
  `HYBRID_SHARD`. R-DEV-1 deviation: `_HYBRID_SHARD_ZERO2` is not yet
  shipped (the underscore-prefixed name in upstream signals
  experimental). `Default::default()` is `FullShard`, matching
  PyTorch's default constructor behavior.
- REQ-2: `pub struct FSDP<M, T>` stores the inner module, backend,
  active strategy, original parameter shapes, full-param tensors
  retained across forward → backward → sync_gradients, optional
  pending async all-gather prefetch handles, and (for HybridShard)
  the `Arc<SubBackend>` for intra-node and inter-node subgroups.
  Mirrors `class FullyShardedDataParallel(nn.Module, _FSDPState)` at
  `torch/distributed/fsdp/fully_sharded_data_parallel.py:118`.
- REQ-3: `pub fn new(module, backend) -> FerrotorchResult<Self>` is a
  thin forwarder to `new_with_strategy` with `FullShard`. Mirrors the
  upstream constructor default. `pub fn new_with_strategy(module,
  backend, strategy) -> FerrotorchResult<Self>` performs the initial
  parameter-sharding step based on `strategy`.
- REQ-4: `FullShard` strategy: each parameter is asserted divisible
  by `world_size`; each rank keeps `numel/world_size` contiguous
  elements and the original shape is recorded for reconstruction.
  Mirrors `ShardingStrategy.FULL_SHARD` / ZeRO-3.
- REQ-5: `ShardGradOp` strategy: parameters remain replicated on
  every rank; only gradients (and external optimizer state) are
  sharded at `sync_gradients` time. Mirrors
  `ShardingStrategy.SHARD_GRAD_OP` / ZeRO-2. Requires a follow-up
  `broadcast_updated_params()` call after `optimizer.step()` to
  re-synchronize updated shards across ranks.
- REQ-6: `NoShard` strategy: parameters are kept full on every rank;
  gradients are allreduced (not reduce-scattered) at sync time.
  Equivalent to DDP / ZeRO-0. Provided so the `FSDP` wrapper is a
  superset of DDP for debugging.
- REQ-7: `HybridShard { intra_node_size }` strategy: validates
  `world_size % intra_node_size == 0`, builds `intra_node_group` and
  `inter_node_group` `SubBackend`s from the global backend at
  construction. Parameters are sharded **within** the node only
  (`numel / intra_node_size`); during `sync_gradients` the gradient
  is reduce-scattered intra-node, then allreduced inter-node, so
  every replica of a given intra-rank sees the same gradient.
  Mirrors `ShardingStrategy.HYBRID_SHARD`.
- REQ-8: `pub fn forward(&mut self, input) -> FerrotorchResult<Tensor<T>>`
  reconstructs full parameters before running the inner module's
  `forward`. For `FullShard` and `HybridShard` the full parameters
  are all-gathered (via `crate::collective::all_gather`); for
  `ShardGradOp` and `NoShard` parameters are already full. The
  reconstructed tensors are stored in `self.full_params` so that
  backward can accumulate gradients on them. After forward, the
  full parameters are dropped (FullShard / HybridShard).
- REQ-9: `pub fn prefetch_forward_params(&mut self) ->
  FerrotorchResult<()>` kicks off asynchronous all-gathers via
  `crate::async_collective::async_all_gather` so the next
  `forward()` consumes pre-gathered tensors. Valid only for
  `FullShard`; errors otherwise. Double-call without intervening
  `forward()` errors. Mirrors PyTorch's
  `BackwardPrefetch.BACKWARD_PRE` (and the forward prefetch in
  `_runtime_utils.py`). `pub fn has_pending_prefetch(&self) -> bool`
  exposes the in-flight state for diagnostics.
- REQ-10: `pub fn sync_gradients(&mut self) -> FerrotorchResult<()>`
  reads gradients from `self.full_params` (set by backward), runs
  the strategy-appropriate reduction (`reduce_scatter` for
  FullShard, `reduce_scatter` + zero-pad for ShardGradOp,
  `reduce_scatter` then `allreduce` for HybridShard, plain
  `allreduce` for NoShard), and installs the result on each
  parameter's `.grad()`. After completion, `full_params` is cleared
  to free memory.
- REQ-11: `pub fn broadcast_updated_params(&mut self) ->
  FerrotorchResult<()>` (for `ShardGradOp` only) reconstructs the
  full updated parameter on every rank by all-gathering each
  rank's slice. No-op for `FullShard`, `NoShard`, and `HybridShard`
  (which have consistent params after step). Required after
  `optimizer.step()` in the ZeRO-2 loop.
- REQ-12: `pub fn update_shards(&mut self, flat_data: &[T]) ->
  FerrotorchResult<()>` updates shard parameters from a flat
  optimizer-produced buffer. Asserts `flat_data.len()` equals the
  sum of shard numels.
- REQ-13: Accessors `pub fn strategy`, `pub fn module`, `pub fn
  module_mut`, `pub fn into_inner`, `pub fn backend`. These match
  the PyTorch accessor set used by training scripts.

## Acceptance Criteria

- [x] AC-1: 2 ranks, parameter `[10, 20, 30, 40]`: rank 0 holds
  `[10, 20]`, rank 1 holds `[30, 40]` (FullShard sharding test).
- [x] AC-2: Shard tensors have `requires_grad=true` (so autograd can
  accumulate the local-slice contribution).
- [x] AC-3: After `forward()`, FullShard parameters are restored to
  shard size on every rank.
- [x] AC-4: 2 ranks each set the same full-param gradient
  `[1, 2, 3, 4]`; after `sync_gradients`, rank 0 has shard gradient
  `[1, 2]` and rank 1 has `[3, 4]`.
- [x] AC-5: `update_shards` with the right element count succeeds;
  with wrong count panics (deliberate `assert!`).
- [x] AC-6: `ShardGradOp` keeps params replicated on every rank;
  `sync_gradients` writes a full-shape gradient with zero-padded
  non-shard positions; `broadcast_updated_params` reconstructs the
  full updated parameter via all-gather.
- [x] AC-7: `NoShard` is DDP-equivalent: param stays full,
  `sync_gradients` allreduces, mean of identical contributions is
  the identity.
- [x] AC-8: `HybridShard { intra_node_size: 3 }` with `world_size=4`
  errors at construction.
- [x] AC-9: `HybridShard { intra_node_size: 2 }` with `world_size=4`
  shards within each node; ranks with the same `rank % 2` see the
  same intra-shard slice. Inter-node averaging mixes node-local
  gradients across replicas.
- [x] AC-10: `prefetch_forward_params` followed by `forward()`
  produces the same output as the synchronous path. Double-prefetch
  errors. Prefetch on `ShardGradOp` errors.

## Architecture

### Sharding strategy enum (REQ-1)

`pub enum ShardingStrategy` (in `fsdp.rs`) carries the four variants;
`HybridShard` carries `intra_node_size: usize`. The `Default` impl
returns `FullShard` to match the upstream constructor default.

### Construction (REQ-2, REQ-3, REQ-4, REQ-7)

`pub fn new` is a thin forwarder. `pub fn new_with_strategy` (in
`fsdp.rs`) does:

1. For `HybridShard`, build `intra_node_group` and `inter_node_group`
   `SubBackend`s based on the rank-grid layout. `intra_members` is the
   contiguous block `[node_idx*intra_size, node_idx*intra_size+intra_size)`
   and `inter_members` is the stride-selected `[local_idx,
   intra_size+local_idx, 2*intra_size+local_idx, ...]`. Both are
   wrapped in `Arc<SubBackend>` so they can be cheaply re-borrowed by
   `forward` and `sync_gradients`.
2. For each parameter, record `original_shapes` so later forwards can
   reconstruct the right shape.
3. For `FullShard`: assert `numel % world_size == 0`; replace the
   parameter with a 1-D shard tensor of length `numel/world_size`.
4. For `HybridShard`: assert `numel % intra_size == 0`; replace with
   the intra-rank shard.
5. For `ShardGradOp` / `NoShard`: leave the parameter full; the
   strategy only affects gradient sync.

The two `expect()` calls in the HybridShard arms carry detailed
SAFETY-style INVARIANT comments documenting the typestate-like
coupling between `strategy` and `intra_node_group` /
`inter_node_group` (Category C from rust-fix-discipline).

### Forward (REQ-8, REQ-9)

`pub fn forward(&mut self, input)` (in `fsdp.rs`):

1. Take any pending prefetch handles (`self.pending_prefetch.take()`).
2. Match on strategy:
   - `FullShard`: for each parameter, either consume a pending
     prefetch handle (`handle.wait()?`) or call
     `crate::collective::all_gather(&shard, backend)`. Reshape to
     the original parameter shape with `requires_grad=true`. Push
     into `self.full_params` AND replace the module parameter with
     the full tensor so the inner module's forward sees the full
     parameter.
   - `HybridShard`: intra-node `all_gather` only.
   - `ShardGradOp` / `NoShard`: parameters are already full; just
     wrap with `requires_grad=true` and stash in `full_params`.
3. Call `self.module.forward(input)`.
4. Restore shards (`restore_shards` for FullShard,
   `restore_hybrid_shards` for HybridShard); ShardGradOp / NoShard
   keep full params.

`pub fn prefetch_forward_params` is only valid for `FullShard`;
returns an `InvalidArgument` for other strategies. Errors on double-
call. Stores `Vec<PendingCollective<T>>` in `self.pending_prefetch`,
populated via `crate::async_collective::async_all_gather`.

### Gradient sync (REQ-10)

`pub fn sync_gradients(&mut self)` (in `fsdp.rs`) reads
`self.full_params[i].grad()?` for each parameter (zero-filling if
absent so wire-size matches across ranks), flattens it, then:

- `FullShard`: `crate::collective::reduce_scatter(&flat_grad,
  backend, ReduceOp::Mean)` → shard gradient. Set
  `param.tensor().set_grad(Some(shard_grad))`.
- `ShardGradOp`: reduce_scatter same as FullShard, but the per-rank
  slice is zero-padded back into a full-shape buffer at the shard
  positions; the optimizer at non-shard positions becomes a no-op.
- `HybridShard`: intra-node `reduce_scatter` followed by inter-node
  `allreduce`. Both via the `SubBackend`s built at construction.
- `NoShard`: `allreduce` over the full gradient (DDP path).

After the loop, `self.full_params.clear()` frees the retained
full-parameter tensors.

### Param sync for ZeRO-2 (REQ-11)

`pub fn broadcast_updated_params(&mut self)` (in `fsdp.rs`) is a no-op
for all strategies except `ShardGradOp`. For ShardGradOp it extracts
each rank's slice of the updated full parameter, all-gathers across
ranks, and re-installs the resulting full-shape tensor with
`requires_grad=true`.

### Shard updates from optimizer (REQ-12)

`pub fn update_shards` (in `fsdp.rs`) accepts a flat slice
representing all parameters' shards concatenated; asserts the slice
length matches the total shard numel, then slices it back into per-
parameter shapes. Used by optimizers that produce flat output (e.g.
fused-update kernels).

### Accessors (REQ-13)

`pub fn strategy`, `pub fn module`, `pub fn module_mut`, `pub fn
into_inner`, `pub fn backend` are direct getters; they don't
allocate or perform I/O.

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/lib.rs` `pub use fsdp::FSDP;` exposes
  the wrapper to user code via `ferrotorch/src/lib.rs` `pub use
  ferrotorch_distributed::*;`.
- Within `fsdp.rs`, `forward` is the production consumer of
  `crate::collective::all_gather` (FullShard / HybridShard) and
  `crate::async_collective::async_all_gather` (prefetch).
- Within `fsdp.rs`, `sync_gradients` is the production consumer of
  `crate::collective::reduce_scatter`, `crate::collective::allreduce`,
  and (via SubBackend) all four strategy paths.
- Within `fsdp.rs`, `broadcast_updated_params` is the production
  consumer of `crate::collective::all_gather` for ShardGradOp
  param re-sync.
- `ferrotorch-distributed/src/backend.rs` documents `SubBackend` as
  "Used by FSDP's HybridShard" — the `Arc<SubBackend>` round-trip
  is the FSDP-specific use case.

## Parity contract

No parity-sweep ops in the route (`parity_ops = []`). The contract is
the PyTorch FSDP shape:

- `FullyShardedDataParallel(module, process_group=None,
  sharding_strategy=None, ...)` at
  `torch/distributed/fsdp/fully_sharded_data_parallel.py:118` →
  ferrotorch's `FSDP::new` / `FSDP::new_with_strategy`. The
  `process_group` kwarg's role is taken by `Arc<dyn Backend>` in
  ferrotorch.
- `ShardingStrategy` upstream values `FULL_SHARD` / `SHARD_GRAD_OP` /
  `NO_SHARD` / `HYBRID_SHARD` at `torch/distributed/fsdp/api.py:65-68`
  match ferrotorch's `FullShard` / `ShardGradOp` / `NoShard` /
  `HybridShard { intra_node_size }`. R-DEV-2: kebab-cased Python
  members map to UpperCamel Rust variants; the semantic mapping is
  1:1.
- Backward prefetch via `prefetch_forward_params` mirrors PyTorch's
  `BackwardPrefetch.BACKWARD_PRE` (FSDP supports forward and backward
  prefetch; ferrotorch ships forward prefetch first; backward
  prefetch is a follow-up).
- Reduce-scatter as the gradient sync primitive (not allreduce)
  matches the FSDP design described in
  `torch/distributed/fsdp/api.py:37-42`.

## Verification

`cargo test -p ferrotorch-distributed --lib fsdp::` runs the
`#[cfg(test)] mod tests` block at lines 883-1738 covering 15 tests:

- Sharding: `test_fsdp_sharding`, `test_fsdp_shard_requires_grad`.
- Forward: `test_fsdp_forward_restores_shards`,
  `test_fsdp_forward_produces_correct_output`.
- Update: `test_fsdp_update_shards`,
  `test_fsdp_update_shards_size_validation`.
- Sync gradients: `test_fsdp_sync_gradients_single_rank`,
  `test_fsdp_sync_gradients_multi_rank`.
- ShardGradOp / NoShard:
  `test_fsdp_shard_grad_op_keeps_full_params`,
  `test_fsdp_shard_grad_op_sync_gradients_multi_rank`,
  `test_fsdp_shard_grad_op_broadcast_updated_params`,
  `test_fsdp_no_shard_is_ddp_equivalent`,
  `test_fsdp_no_shard_broadcast_is_noop`.
- Prefetch: `test_fsdp_prefetched_forward_matches_sync_forward`,
  `test_fsdp_forward_without_prefetch_still_works`,
  `test_fsdp_prefetch_rejects_double_call`,
  `test_fsdp_prefetch_rejects_non_fullshard`.
- HybridShard: `test_fsdp_hybrid_shard_rejects_uneven_world_size`,
  `test_fsdp_hybrid_shard_intra_node_sharding`,
  `test_fsdp_hybrid_shard_sync_gradients`,
  `test_fsdp_hybrid_shard_inter_node_averaging`.

Lint: `cargo clippy -p ferrotorch-distributed -- -D warnings` clean.
Parity-sweep: no ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum ShardingStrategy` in `ferrotorch-distributed/src/fsdp.rs`; non-test consumer: `ferrotorch-distributed/src/backend.rs` doc-link `crate::fsdp::ShardingStrategy::HybridShard`; reachable via `lib.rs` re-export of `FSDP` (whose `new_with_strategy` takes the enum). |
| REQ-2 | SHIPPED | impl: `pub struct FSDP` in `ferrotorch-distributed/src/fsdp.rs`; non-test consumer: `ferrotorch-distributed/src/lib.rs` `pub use fsdp::FSDP;` → `ferrotorch/src/lib.rs`. |
| REQ-3 | SHIPPED | impl: `pub fn new` and `pub fn new_with_strategy` in `ferrotorch-distributed/src/fsdp.rs`; non-test consumer: `new` invokes `new_with_strategy` internally (same file), and both are reachable through `lib.rs` re-export of `FSDP`. |
| REQ-4 | SHIPPED | impl: `ShardingStrategy::FullShard` arm of `new_with_strategy` in `ferrotorch-distributed/src/fsdp.rs`; non-test consumer: invoked from `pub fn new` (same file, default path). |
| REQ-5 | SHIPPED | impl: `ShardingStrategy::ShardGradOp` arm of `new_with_strategy` + `pub fn broadcast_updated_params` in `ferrotorch-distributed/src/fsdp.rs`; non-test consumer: surfaced through `lib.rs` re-export of `FSDP`; `broadcast_updated_params` invokes `crate::collective::all_gather`. |
| REQ-6 | SHIPPED | impl: `ShardingStrategy::NoShard` arms across `new_with_strategy`, `forward`, `sync_gradients` in `ferrotorch-distributed/src/fsdp.rs`; non-test consumer: `sync_gradients` invokes `crate::collective::allreduce` on the NoShard path; FSDP re-exported via `lib.rs`. |
| REQ-7 | SHIPPED | impl: `ShardingStrategy::HybridShard` arms across `new_with_strategy` (builds `Arc<SubBackend>` pair), `forward` (`restore_hybrid_shards`), `sync_gradients` (intra-node `reduce_scatter` + inter-node `allreduce`) in `ferrotorch-distributed/src/fsdp.rs`; non-test consumer: production consumer of `crate::backend::SubBackend::new` and `crate::collective::{reduce_scatter, allreduce}`. |
| REQ-8 | SHIPPED | impl: `pub fn forward` in `ferrotorch-distributed/src/fsdp.rs`; non-test consumer: `pub fn forward` is itself a production user-facing API surfaced through `lib.rs` re-export of `FSDP`; internally calls `crate::collective::all_gather`. |
| REQ-9 | SHIPPED | impl: `pub fn prefetch_forward_params` and `pub fn has_pending_prefetch` in `ferrotorch-distributed/src/fsdp.rs`; non-test consumer: `prefetch_forward_params` invokes `crate::async_collective::async_all_gather`; `forward` consumes the resulting `Vec<PendingCollective>` via `handle.wait()?` (same file). |
| REQ-10 | SHIPPED | impl: `pub fn sync_gradients` in `ferrotorch-distributed/src/fsdp.rs`; non-test consumer: production consumer of `crate::collective::{reduce_scatter, allreduce}`; surfaced via `lib.rs` re-export. |
| REQ-11 | SHIPPED | impl: `pub fn broadcast_updated_params` in `ferrotorch-distributed/src/fsdp.rs`; non-test consumer: production consumer of `crate::collective::all_gather`; surfaced via `lib.rs` re-export. |
| REQ-12 | SHIPPED | impl: `pub fn update_shards` in `ferrotorch-distributed/src/fsdp.rs`; non-test consumer: surfaced via `lib.rs` re-export of `FSDP` for downstream optimizers that produce flat parameter buffers. |
| REQ-13 | SHIPPED | impl: `pub fn strategy`, `pub fn module`, `pub fn module_mut`, `pub fn into_inner`, `pub fn backend` in `ferrotorch-distributed/src/fsdp.rs`; non-test consumer: surfaced via `lib.rs` re-export; the accessor surface is the user-facing way to retrieve sub-fields of an `FSDP` during training. |
