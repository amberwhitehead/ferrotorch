# ferrotorch-data — `collate` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/utils/data/_utils/collate.py
  - torch/utils/data/dataloader.py
-->

## Summary

`ferrotorch-data/src/collate.rs` provides the default collation
strategy for batching tensor samples produced by a `Dataset`: stack a
`Vec<Tensor<T>>` along a new leading dimension. Mirrors the tensor
branch of PyTorch's `default_collate` in
`torch/utils/data/_utils/collate.py:118-243` (the `collate_tensor_fn`
case). The helper `default_collate_pair` extends this to the
common `(input, target)` supervised-learning pattern.

## Requirements

- REQ-1: `pub fn default_collate<T: Float>(samples: Vec<Tensor<T>>)
  -> FerrotorchResult<Tensor<T>>` — stack N tensors of shape `[*S]`
  into one tensor of shape `[N, *S]` along dim 0 via
  `ferrotorch_core::stack`. Empty input returns
  `FerrotorchError::InvalidArgument`. Mirrors
  `torch/utils/data/_utils/collate.py:246-276` `collate_tensor_fn`
  which calls `torch.stack(batch, 0, out=out)`.

- REQ-2: `pub fn default_collate_pair<T: Float>(samples: Vec<(Tensor<T>,
  Tensor<T>)>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)>` — split
  `(input, target)` pairs into two separate stacked tensors. Mirrors
  the tuple-branch fallthrough in `collate` upstream (lines 173-186)
  that recursively collates each tuple element.

- REQ-3: Generic over `T: Float`. Both functions work for `f32`,
  `f64`, and any future numeric type implementing the
  `ferrotorch_core::Float` trait (which currently bounds to `f32` /
  `f64`). Empty-input validation runs BEFORE the type-generic stack
  call so the error message is collate-specific, not a confusing
  inner-`stack`-error.

- REQ-4: Error contract on heterogeneous shapes or devices. Both
  functions delegate to `stack`, which itself returns
  `FerrotorchError::ShapeMismatch` on shape divergence and
  `FerrotorchError::DeviceMismatch` on a mix of CPU/CUDA inputs.
  No silent shape-broadcast and no implicit device migration —
  matches PyTorch's `RuntimeError: stack expects each tensor to be
  equal size, but got [...]` from
  `aten/src/ATen/native/TensorShape.cpp` `at::stack`.

## Acceptance Criteria

- [x] AC-1: `fn default_collate` in `collate.rs` calls `stack(&samples,
  0)` after the empty-input check.
- [x] AC-2: `fn default_collate_pair` in `collate.rs` unzips into
  two parallel `Vec<Tensor<T>>` lists and calls `stack` twice.
- [x] AC-3: `test_default_collate_1d` and `test_default_collate_2d`
  in `mod tests in collate.rs` exercise the basic stack path for
  shape `[3]` → `[2, 3]` and `[2, 2]` → `[2, 2, 2]`.
- [x] AC-4: `test_default_collate_empty` and
  `test_default_collate_pair_empty` exercise the
  `InvalidArgument` empty-input branch.
- [x] AC-5: `test_default_collate_f64` exercises the `T: Float`
  generic over `f64` (not just `f32`).

## Architecture

### `default_collate` (REQ-1, REQ-3)

The function body is three lines after the empty-input guard:

```rust
if samples.is_empty() {
    return Err(FerrotorchError::InvalidArgument {
        message: "default_collate: empty sample list".into(),
    });
}
stack(&samples, 0)
```

The work happens in `ferrotorch_core::stack` (re-exported via
`ferrotorch_core::stack` in `creation.rs`). `stack` itself wraps
`torch.stack(tensors, dim=0)` — inserting a new leading dim and
concatenating along it. The empty-input guard is necessary because
`stack` upstream is documented to require at least one tensor
(`aten/src/ATen/native/TensorShape.cpp::stack` checks
`tensors.size() > 0` and TORCH_CHECKs otherwise).

### `default_collate_pair` (REQ-2)

The pair helper handles the supervised-learning idiom: dataset
samples are typically `(input_tensor, target_tensor)` tuples and the
collated batch is `(batched_inputs, batched_targets)`. The
implementation unzips the input list and calls `stack` on each
half:

```rust
let (inputs, targets): (Vec<_>, Vec<_>) = samples.into_iter().unzip();
Ok((stack(&inputs, 0)?, stack(&targets, 0)?))
```

This corresponds to the tuple-recursion branch of upstream's
`default_collate` (lines 173-186): when the batch element is a
tuple, recurse into each component and rebuild as a tuple of
collated tensors.

### Error propagation (REQ-4)

Both functions are pure pass-throughs to `stack`; shape /
device divergence errors propagate verbatim from the underlying
`stack` impl. Test cases for these branches live in
`ferrotorch-core/tests/conformance_creation.rs` (the `stack` op's
own tests); the collate-level tests focus on the empty-input branch
and the dtype-genericity property.

### Non-test production consumers

- `DataLoader::with_collate` in `dataloader.rs` accepts a
  `Fn(Vec<Sample>) -> Result<Sample>` closure; callers writing a
  custom dataset of tensors typically pass `default_collate` here:

  ```rust
  let loader = DataLoader::new(Arc::new(my_tensor_ds), 32)
      .with_collate(default_collate);
  ```

- `ferrotorch/src/lib.rs` glob re-export propagates
  `default_collate` and `default_collate_pair` to the meta-crate
  surface as `ferrotorch::default_collate` /
  `ferrotorch::default_collate_pair`.

- The downstream training-loop crates (`ferrotorch-llama`,
  `ferrotorch-bert`, etc.) use `default_collate_pair` for
  `(token_ids, target_ids)` minibatches in their training driver
  modules.

## Parity contract

`parity_ops = []`. The numerical contract is delegated to
`ferrotorch_core::stack` (which itself has parity-sweep coverage
under the `stack` op name). Edge cases preserved by this module:

- **Empty sample list** → `Err(InvalidArgument)`. Upstream raises
  `RuntimeError("expected a non-empty list of Tensors")` from
  `at::stack`; we surface a more specific message that names
  `default_collate` so the user sees which collate path failed.
- **Shape mismatch** → propagated from `stack`; matches PyTorch's
  `RuntimeError: stack expects each tensor to be equal size`.
- **Device mismatch** → propagated from `stack`; matches PyTorch's
  device-check failure (silently CPU-promoting is a R-CODE-4
  violation we explicitly do NOT do).
- **Dtype propagation** — all samples must be the same `T`; the
  function signature `<T: Float>` enforces this at compile time.
  Heterogeneous-dtype collation (PyTorch's
  `collate_numpy_array_fn` etc.) is intentionally out of scope —
  the user must promote upstream of `default_collate`.
- **`requires_grad` on the output**: inherited from `stack`'s
  contract; since `stack` propagates grad if any input requires it,
  the collated batch's `requires_grad` is the OR of the inputs.

## Verification

Unit tests in `mod tests in collate.rs` (7 tests):

- `test_default_collate_1d` — `[3]` × 2 → `[2, 3]`, exact data
  verification.
- `test_default_collate_2d` — `[2, 2]` × 2 → `[2, 2, 2]`, shape
  verification.
- `test_default_collate_scalars` — `[]` × 3 → `[3]`, scalar
  stacking edge case.
- `test_default_collate_empty` — empty input → `Err`.
- `test_default_collate_pair` — `([2], [1])` × 2 → `([2, 2], [2, 1])`.
- `test_default_collate_pair_empty` — empty pair input → `Err`.
- `test_default_collate_f64` — exercise `T: Float` over `f64`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-data --lib collate:: 2>&1 | tail -3
```

Expected: 7 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn default_collate<T: Float>(samples: Vec<Tensor<T>>) -> FerrotorchResult<Tensor<T>>` in `collate.rs` calling `stack(&samples, 0)` after empty-input check; mirrors `torch/utils/data/_utils/collate.py:246-276` `collate_tensor_fn`; non-test consumer: `pub use collate::default_collate` in `lib.rs` re-exports to the crate surface, and `DataLoader::with_collate` in `dataloader.rs` accepts callable closures including `default_collate` as `Arc<dyn Fn(Vec<Sample>) -> Result<Sample> + Send + Sync>`. |
| REQ-2 | SHIPPED | impl: `pub fn default_collate_pair<T: Float>(samples: Vec<(Tensor<T>, Tensor<T>)>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)>` in `collate.rs`, unzipping then stacking both halves; mirrors the tuple-recursion branch of `torch/utils/data/_utils/collate.py:173-186`; non-test consumer: `pub use collate::default_collate_pair` in `lib.rs` and downstream training-loop code in the model crates uses it for `(input, target)` batches. |
| REQ-3 | SHIPPED | impl: both functions are `<T: Float>`-generic so f32 and f64 are first-class; non-test consumer: the same `pub use` from `lib.rs` makes the generic surface available to any caller; `test_default_collate_f64` in `mod tests` exercises the `f64` path explicitly. |
| REQ-4 | SHIPPED | impl: both functions delegate to `ferrotorch_core::stack` for the shape/device contract, so `FerrotorchError::ShapeMismatch` and `FerrotorchError::DeviceMismatch` propagate verbatim; non-test consumer: `DataLoader::with_collate` callers (e.g. the meta-crate re-export path) observe these errors through the `FerrotorchResult` return; no silent device migration per R-CODE-4. |
