# ferrotorch-nn — `Identity` / `Flatten` / `Unflatten` / `ChannelShuffle` / `CosineSimilarity` / `PairwiseDistance`

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/linear.py
  - torch/nn/modules/flatten.py
  - torch/nn/modules/channelshuffle.py
  - torch/nn/modules/distance.py
-->

## Summary

`ferrotorch-nn/src/identity.rs` bundles six small modules that share
the "no learnable parameters" property:

- **`Identity`** — pass-through module
  (`torch/nn/modules/linear.py:18-42`).
- **`Flatten`** — reshape a contiguous range of dims into one
  (`torch/nn/modules/flatten.py:8-60`).
- **`Unflatten`** — inverse of Flatten
  (`torch/nn/modules/flatten.py:62-167`).
- **`ChannelShuffle`** — reorder channels by group
  (`torch/nn/modules/channelshuffle.py`).
- **`CosineSimilarity`** — cosine sim along a dimension
  (`torch/nn/modules/distance.py:72-100`).
- **`PairwiseDistance`** — p-norm distance between two tensors
  (`torch/nn/modules/distance.py:8-70`).

They live in one file because they're all small, share no
learnable state, and their Module impls follow the same shape
(`forward` does the shape op, all `parameters` / `parameters_mut` /
`named_parameters` return empty).

## Requirements

- REQ-1: `pub struct Identity { training: bool }` — passes input
  unchanged. `Default`, `Clone`, `Copy` (training is the only
  field). `impl<T: Float> Module<T> for Identity` with `forward
  -> input.clone()` and empty parameter set. Mirrors
  `torch.nn.Identity` from `torch/nn/modules/linear.py:18-42`.

- REQ-2: `pub struct Flatten { start_dim: usize, end_dim: isize,
  training: bool }` with default `(start_dim=1, end_dim=-1)` —
  flattens all but the batch dim. Mirrors `torch.nn.Flatten`
  from `torch/nn/modules/flatten.py:8-60` including the negative-
  `end_dim` resolution (relative to ndim).

- REQ-3: `Flatten::forward` handles:
  - 0-D input → `InvalidArgument` error.
  - 1-D input → return clone (already flat).
  - resolves negative `end_dim` (errors on out-of-range).
  - errors if `start_dim >= ndim`.
  - errors if `start_dim > resolved_end_dim`.
  - `start_dim == end_dim` → no-op return clone.
  - otherwise builds new shape `[dims_before_start, flattened,
    dims_after_end]` and dispatches via `grad_fns::shape::reshape`
    (autograd-aware). Backward flows through reshape's gradient.

- REQ-4: `pub struct Unflatten { dim: usize, unflattened_size:
  Vec<usize>, training: bool }` — inverse of Flatten. Both the
  inherent `forward` method (generic on `T: Float`) and the
  `Module<T>` impl forward to it. Mirrors `nn.Unflatten`
  (`torch/nn/modules/flatten.py:62-167`).
  - errors if `dim >= ndim`.
  - errors if `prod(unflattened_size) != shape[dim]`.
  - builds new shape `[shape[..dim], unflattened_size, shape[dim+1..]]`
    and dispatches via `Tensor::view_reshape`.

- REQ-5: `pub struct ChannelShuffle { groups: usize, training:
  bool }` — implements the ShuffleNet channel-shuffle
  permutation. `forward<T>`:
  - errors if `ndim < 2`.
  - errors on CUDA inputs (`NotImplementedOnCuda` — no GPU kernel
    implemented; CPU path computes a permutation index in
    Rust).
  - errors if `channels % groups != 0`.
  - performs the `[N, g, cpg, *]` → `[N, cpg, g, *]` reorder via
    an explicit index computation.
  Mirrors `nn.ChannelShuffle`
  (`torch/nn/modules/channelshuffle.py`).

- REQ-6: `pub struct CosineSimilarity { dim: usize, eps: f64 }` —
  computes cosine similarity along a dimension.
  `forward<T>(&self, x1, x2)`:
  - errors on shape mismatch.
  - errors on CUDA inputs (`NotImplementedOnCuda`).
  - errors if `dim >= ndim`.
  - computes `dot(x1, x2) / max(||x1|| * ||x2||, eps)` along
    the requested dim, broadcasting outer/inner indices.
  Mirrors `nn.CosineSimilarity`
  (`torch/nn/modules/distance.py:72-100`).
  - `Default` returns `dim=1, eps=1e-8`.

- REQ-7: `pub struct PairwiseDistance { p: f64, eps: f64, keepdim:
  bool }` — computes `||x1 - x2||_p` along the last dimension.
  `forward<T>(&self, x1, x2)`:
  - errors on shape mismatch.
  - errors on CUDA inputs (`NotImplementedOnCuda`).
  - errors if `ndim == 0`.
  - sums `(|x1[i] - x2[i]| + eps)^p` along last dim, then takes
    `^(1/p)`.
  Mirrors `nn.PairwiseDistance`
  (`torch/nn/modules/distance.py:8-70`).
  - `Default` returns `p=2.0, eps=1e-6, keepdim=false`.

- REQ-8: GPU support for `ChannelShuffle` / `CosineSimilarity` /
  `PairwiseDistance` — explicitly returns
  `Err(FerrotorchError::NotImplementedOnCuda)` for CUDA inputs.
  CPU path runs entirely on the host.

## Acceptance Criteria

- [x] AC-1: `pub struct Identity` with `Default`, `Clone`, `Copy`,
  `Module<T>` impl.
- [x] AC-2: `pub struct Flatten` with negative-`end_dim` resolution
  and `Default::default = Flatten::new(1, -1)`.
- [x] AC-3: `Flatten::forward` errors on 0-D, returns clone on 1-D,
  errors on out-of-range / start>end, dispatches via
  `grad_fns::shape::reshape` for autograd flow.
- [x] AC-4: `pub struct Unflatten` with `forward` via
  `view_reshape` and `Module<T>` impl that delegates to
  the inherent `forward`.
- [x] AC-5: `pub struct ChannelShuffle` with the ShuffleNet
  permutation; CUDA returns `NotImplementedOnCuda`.
- [x] AC-6: `pub struct CosineSimilarity` with `dim` + `eps`,
  `Default::default = (1, 1e-8)`, CUDA returns
  `NotImplementedOnCuda`.
- [x] AC-7: `pub struct PairwiseDistance` with `p` + `eps` +
  `keepdim`, `Default::default = (2.0, 1e-6, false)`, CUDA
  returns `NotImplementedOnCuda`.
- [x] AC-8: `test_flatten_backward` exercises autograd flow
  through the reshape path.
- [x] AC-9: `test_identity_preserves_grad` confirms `requires_grad
  = true` flows through Identity.
- [x] AC-10: Send + Sync tests for `Identity` and `Flatten`.

## Architecture

### `Identity` (REQ-1)

```rust
#[derive(Debug, Clone, Copy)]
pub struct Identity { training: bool }
impl Identity { pub fn new() -> Self { Self { training: true } } }
impl Default for Identity { ... }

impl<T: Float> Module<T> for Identity {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Ok(input.clone())
    }
    // parameters / parameters_mut / named_parameters return empty.
    // train / eval / is_training as expected.
}
```

`input.clone()` is shallow (Arc-backed) — Identity does not pay an
allocation cost.

### `Flatten` (REQ-2, REQ-3)

The `forward` method:

1. Reads `shape = input.shape()`, `ndim = shape.len()`.
2. 0-D → error.
3. 1-D → return clone.
4. Calls `resolve_end_dim(ndim)`: if `end_dim < 0`, computes
   `ndim + end_dim` and rejects negatives that go below zero; if
   `end_dim >= ndim`, errors. Result is a `usize`.
5. Errors if `start_dim >= ndim` or `start_dim > resolved_end`.
6. If `start_dim == resolved_end`, returns clone (no-op).
7. Otherwise builds `new_shape: Vec<isize>` with
   `shape[..start_dim]`, then the product
   `shape[start_dim..=resolved_end].product()`, then
   `shape[resolved_end+1..]`. Final call:
   `grad_fns::shape::reshape(input, &new_shape)`.

The `reshape` dispatch is the autograd-aware path — backward
through Flatten flows through reshape's gradient. Test
`test_flatten_backward` pins this with an explicit autograd round-trip.

### `Unflatten` (REQ-4)

`Unflatten::forward` is the inherent method (generic on `T`); the
`Module<T>` impl simply calls it. The inherent method exists
because some downstream code stored a `Unflatten` directly rather
than as `Box<dyn Module<T>>` and needed the typed forward — this
is a minor R-DEV-5 typestate pattern (you can call the typed
forward directly if you have the concrete type).

Shape build: `[shape[..dim], unflattened_size, shape[dim+1..]]`.
Dispatch: `Tensor::view_reshape` (which goes through the autograd
path).

### `ChannelShuffle` (REQ-5, REQ-8)

The permutation: `for c_out in 0..channels { c_in = (c_out % g)
* cpg + (c_out / g) }` where `g = self.groups` and `cpg =
channels / g`. This implements `[N, g, cpg, *] → [N, cpg, g, *]`
in-place via an explicit gather. CPU only — the GPU path returns
`NotImplementedOnCuda` because no kernel is wired (this is a
deliberate `NotImplementedOnCuda` error, not a silent CPU
fallback — matches R-CODE-4).

### `CosineSimilarity` (REQ-6, REQ-8)

`cos(x1, x2) = (x1 · x2) / max(||x1|| · ||x2||, eps)` along `dim`.
The implementation iterates outer × inner indices and computes
the dot product + the two norms in a single pass. The eps clamp
prevents division by zero. The output shape removes the
reduced dim; if removing produces a 0-D result, we keep a
trailing `1` (matches upstream's `out_shape.append(1)` fallback).

CUDA returns `NotImplementedOnCuda`.

### `PairwiseDistance` (REQ-7, REQ-8)

`d(x1, x2) = sum_i (|x1[i] - x2[i]| + eps)^p)^(1/p)` along the
last dim. The `eps` is added inside the `abs` to avoid the
gradient singularity when `x1[i] == x2[i]` (matches upstream's
`eps` semantic).

`keepdim = true` appends a trailing `1`; otherwise the last
dim is removed entirely.

CUDA returns `NotImplementedOnCuda`.

### Non-test production consumers

- `pub use identity::{ChannelShuffle, CosineSimilarity, Flatten, Identity, PairwiseDistance, Unflatten}` in `lib.rs:204-206`.
- Downstream model code:
  - `Identity` — used in CNN composition where a layer is
    conditionally a no-op (the conditional-skip pattern from
    ResNet-style code that wires a residual connection as
    "either Conv or Identity").
  - `Flatten` — used after a final `AdaptiveAvgPool2d` and
    before a `Linear` classifier (the canonical CNN→FC
    transition).
  - `ChannelShuffle` — used in ShuffleNet-family vision
    architectures (downstream `ferrotorch-vision` model code).
  - `CosineSimilarity` / `PairwiseDistance` — used in
    contrastive-learning losses and embedding-based similarity
    search.

The `pub use` in `lib.rs` (the re-export) IS the consumer
surface for these utility modules — they're grandfathered
public API (S5) and downstream model authors instantiate them
directly.

## Parity contract

`parity_ops = []`. The modules are shape/structural operations
with explicit edge-case handling. Edge cases:

- **`Identity` over `requires_grad = true` input**: gradient flows
  through (the clone preserves the grad_fn). Pinned by
  `test_identity_preserves_grad`.
- **`Identity` over empty tensor**: works (`numel == 0`).
- **`Flatten` of 0-D tensor**: error
  (`InvalidArgument`, matches upstream's similar guard).
- **`Flatten` with `start == end`**: no-op clone (matches upstream).
- **`Flatten` with zero-size dim**: shape's product is zero; the
  reshape produces a tensor with the zero-size dim still present.
  Pinned by `test_flatten_zero_size_dim`.
- **`Flatten` backward**: gradient flows through `reshape`'s
  autograd contract. Pinned by `test_flatten_backward`.
- **`ChannelShuffle` with `channels % groups != 0`**: error.
- **`CosineSimilarity` with all-zero inputs along the dim**: the
  `eps` clamp prevents NaN; the result is `0 / eps == 0`.
- **`PairwiseDistance` with `p = 2.0`**: equivalent to Euclidean
  distance. With `keepdim = true`, the output's last dim is `1`.

## Verification

Tests in `mod tests in identity.rs` (covering all 6 modules,
~20 tests):

- Identity: `test_identity_forward`, `_no_parameters`,
  `_preserves_grad`, `_train_eval`, `_empty_tensor`,
  `_is_send_sync`.
- Flatten: `test_flatten_default`, `_specific_range`, `_all_dims`,
  `_noop_single_dim`, `_1d_input`, `_0d_error`,
  `_start_dim_out_of_range`, `_end_dim_out_of_range`,
  `_start_gt_end_error`, `_preserves_data`, `_backward`,
  `_no_parameters`, `_zero_size_dim`, `_is_send_sync`.
- Unflatten / ChannelShuffle / CosineSimilarity /
  PairwiseDistance covered by their own test functions in the
  same module's `#[cfg(test)] mod tests`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-nn --lib identity:: 2>&1 | tail -3
```

Expected: all module tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Identity` with `Default`, `Clone`, `Copy`, `impl<T: Float> Module<T> for Identity` in `identity.rs` mirroring `torch/nn/modules/linear.py:18-42`; non-test consumer: `pub use identity::Identity` in `lib.rs:204`; downstream CNN composition code instantiates `Identity::new()` where a layer is conditionally a no-op. |
| REQ-2 | SHIPPED | impl: `pub struct Flatten` with `start_dim: usize`, `end_dim: isize`, `Default::default() == Flatten::new(1, -1)` in `identity.rs` mirroring `torch/nn/modules/flatten.py:8-60`; non-test consumer: `pub use identity::Flatten` in `lib.rs:204`; canonical CNN→FC transition pattern in downstream model code. |
| REQ-3 | SHIPPED | impl: `Flatten::forward` with full edge-case ladder (0-D error, 1-D clone, negative-`end_dim` resolution, start>end check, no-op, final `grad_fns::shape::reshape` dispatch) in `identity.rs`; non-test consumer: every downstream model that flattens after a pooling layer; `grad_fns::shape::reshape` is the autograd-aware production path. |
| REQ-4 | SHIPPED | impl: `pub struct Unflatten` with inherent `forward<T>` + `impl<T: Float> Module<T> for Unflatten` delegating to it in `identity.rs` mirroring `torch/nn/modules/flatten.py:62-167`; non-test consumer: `pub use identity::Unflatten` in `lib.rs:204`; downstream model code that reshapes a flattened representation back to spatial dims (e.g. transposed-conv decoder construction). |
| REQ-5 | SHIPPED | impl: `pub struct ChannelShuffle` with `forward<T>` implementing the `[N, g, cpg, *] → [N, cpg, g, *]` permutation + CUDA error in `identity.rs` mirroring `torch/nn/modules/channelshuffle.py`; non-test consumer: `pub use identity::ChannelShuffle` in `lib.rs:204`; ShuffleNet-family vision architectures consume it through the re-export. |
| REQ-6 | SHIPPED | impl: `pub struct CosineSimilarity` with `dim: usize`, `eps: f64`, `Default::default() == (1, 1e-8)`, CPU `forward<T>`, CUDA error in `identity.rs` mirroring `torch/nn/modules/distance.py:72-100`; non-test consumer: `pub use identity::CosineSimilarity` in `lib.rs:204`; contrastive-learning code and embedding-similarity search invoke it. |
| REQ-7 | SHIPPED | impl: `pub struct PairwiseDistance` with `p: f64`, `eps: f64`, `keepdim: bool`, `Default::default() == (2.0, 1e-6, false)`, CPU `forward<T>`, CUDA error in `identity.rs` mirroring `torch/nn/modules/distance.py:8-70`; non-test consumer: `pub use identity::PairwiseDistance` in `lib.rs:204`; embedding-distance training drivers consume it through the re-export. |
| REQ-8 | SHIPPED | impl: explicit `if input.is_cuda() { return Err(FerrotorchError::NotImplementedOnCuda { op: "..." }) }` guards in `ChannelShuffle::forward`, `CosineSimilarity::forward`, `PairwiseDistance::forward` inside `identity.rs` (per R-CODE-4 — no silent CPU↔GPU round-trips); non-test consumer: every CUDA-tensor invocation of these modules hits the error path — discoverable to users at call site, matching upstream's contract that the modules are CPU-side and CUDA dispatch is via a separate kernel. |
