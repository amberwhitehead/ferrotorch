# NestedTensor / PackedNestedTensor — ragged (jagged) tensors

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/native/nested/
  - torch/nested/_internal/
-->

## Summary

`ferrotorch-core/src/nested.rs` defines:
- `NestedTensor<T: Float>` — a collection of `Tensor<T>` components with
  differing sizes along a single ragged dimension (every other dim must
  match). Mirrors PyTorch's `torch.nested.nested_tensor`
  (`aten/src/ATen/native/nested/NestedTensorMath.cpp`).
- `PackedNestedTensor<T: Float>` — a flat-buffer + offsets layout for
  the same logical content. Mirrors PyTorch's jagged-layout NJT
  (`torch/nested/_internal/nested_tensor.py:55 class NestedTensor` —
  the per-component `_values` flat tensor + `_offsets` carrier).
  CL-291.
- `nested_scaled_dot_product_attention<T: Float>` — variable-length
  attention helper that dispatches per-component to flash-attention on
  GPU when available. Mirrors `torch._C._nested_scaled_dot_product`.

## Requirements

- REQ-1: `NestedTensor::new(tensors, ragged_dim)` constructor — validates
  all components have the same ndim and identical sizes on every
  non-ragged dim. Returns structured error on mismatch. Mirrors
  `torch.nested.nested_tensor([t1, t2, ...])` (`aten/src/ATen/native/nested/NestedTensorFactories.cpp`).
- REQ-2: Accessors: `num_components`, `ragged_dim`, `tensors(&self) ->
  &[Tensor<T>]`, `ndim`, `consistent_shape`, `ragged_lengths`.
- REQ-3: `to_padded(pad_value)` — convert to a dense padded tensor of
  shape `[batch, d_0, ..., max_L, ..., d_{n-1}]`. Has a GPU fast path
  (P4 of #806) using `fill_f{32,64}` + `strided_scatter_f{32,64}` —
  components stay on-device throughout. CPU fallback for mixed-device
  or unsupported dtype. Mirrors `torch.nested.to_padded_tensor`.
- REQ-4: `from_padded(tensor, lengths, ragged_dim)` — inverse of
  `to_padded`. Slices out each component via `narrow` + `.contiguous()`
  on the GPU fast path; CPU fallback for else. Mirrors
  `torch.nested.from_padded`.
- REQ-5: `nested_scaled_dot_product_attention(q, k, v)` — per-component
  scaled-dot-product attention. GPU FlashAttention dispatch when the
  component fits the kernel's regime (`d_k <= 128`, `d_v <= 128`); the
  composite CPU path runs `Q @ K^T * scale → softmax → · V` per
  component as a fallback. Mirrors `torch._C._nested_scaled_dot_product`
  and `torch.nn.functional.scaled_dot_product_attention` on nested
  inputs.
- REQ-6: `PackedNestedTensor` — packed flat-storage layout
  (`data: Vec<T>`, `offsets: Vec<usize>`, `tail_shape: Vec<usize>`).
  Invariants: `offsets.len() == num_components + 1`, `offsets[0] == 0`,
  `offsets[i+1] - offsets[i] == lengths[i] * tail_numel`,
  `offsets[num_components] == data.len()`. Constructor
  `from_sequences(seqs, lengths, tail_shape)` validates the invariants.
- REQ-7: Structured errors on shape / device / component-count mismatch.
  No panics in production. R-CODE-2.

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib nested` passes all
  `#[cfg(test)] mod tests` tests.
- [x] AC-2: `NestedTensor::new` validates per-component ndim + non-ragged
  shape parity (mechanical: `ferrotorch-core/src/nested.rs:50-96`).
- [x] AC-3: `to_padded` GPU fast path materializes on-device when every
  component is on the same CUDA device (`ferrotorch-core/src/nested.rs:258-377`).
- [x] AC-4: `from_padded` GPU fast path routes through `narrow` +
  `.contiguous()` (`ferrotorch-core/src/nested.rs:450-454`).
- [x] AC-5: `nested_scaled_dot_product_attention` errors on
  `num_components` mismatch and dim-count mismatch (mechanical at
  `ferrotorch-core/src/nested.rs:663-691`).
- [x] AC-6: `PackedNestedTensor::from_sequences` validates per-component
  length × `tail_numel` (`ferrotorch-core/src/nested.rs:988-1010`).
- [x] AC-7: The GPU FlashAttention kernel-call path is delegated to the
  `GpuBackend` trait — `try_flash_attention_gpu_component` at `:775`
  returns `Ok(false)` if the backend declines, falling through to the
  composite CPU path. No silent CPU detour.

## Architecture

### `NestedTensor<T>` layout (`nested.rs:29-36`)

```rust
pub struct NestedTensor<T: Float> {
    tensors: Vec<Tensor<T>>,
    ragged_dim: usize,
}
```

Each component is a full `Tensor<T>` with its own (possibly distinct)
shape on the ragged dim and identical shape on every other dim. The
`ragged_dim` is the dimension index that varies across components.

This is the **components-list** layout — diverges from upstream's
preferred jagged-layout (`_values`-flat + `_offsets`) by trading
storage compactness for component-level autograd graph independence
(each `tensors[i]` carries its own grad-fn). The packed layout
(`PackedNestedTensor` below) is the storage-compact alternative when
autograd granularity isn't needed.

### Padded / unpadded round-trip (REQ-3 / REQ-4)

`to_padded(pad_value)` builds a dense `[batch, ..., max_L, ...]` tensor
by allocating a pre-padded buffer then scatter-writing each component's
data into its slot. The GPU fast path uses
`fill_f{32,64}` for step 1 and `strided_scatter_f{32,64}` for step 2;
both kernels exist in the `ferrotorch-gpu` backend trait. Each
component's `.gpu_handle()` is read directly; the `strided_copy_f{32,64}`
kernel first materializes a fresh contiguous CUDA buffer (handles the
storage-offset issue where `narrow(0, k, n)` is stride-contiguous but
has non-zero offset). CPU fallback at `nested.rs:206-240`.

`from_padded(tensor, lengths, ragged_dim)` is the inverse. GPU fast
path at `nested.rs:450-454` uses `narrow` + `.contiguous()` for
zero-copy slicing then per-component materialization.

### Per-component flash attention (REQ-5)

`nested_scaled_dot_product_attention` walks the component list:
1. Shape validation: every (q[i], k[i], v[i]) trio must be 2-D with
   compatible `d_k` and `seq_k`.
2. GPU FlashAttention dispatch: `try_flash_attention_gpu_component`
   (`nested.rs:775`) asks the `GpuBackend` for a flash-attention call.
   Returns `Ok(true)` if the kernel fired, `Ok(false)` if the backend
   declined (unsupported dtype, shape outside kernel regime, etc.).
3. Composite CPU fallback (`nested.rs:722-770`): `Q @ K^T`,
   scale by `1/sqrt(d_k)`, row-wise softmax, multiply by `V`.

The GPU fast path keeps the output `Tensor<T>` GPU-resident. The CPU
fallback materializes via `.data()` and runs a triple-loop.

### `PackedNestedTensor` layout (`nested.rs:937-951`)

```rust
pub struct PackedNestedTensor<T: Float> {
    data: Vec<T>,                 // flat concat in component order
    offsets: Vec<usize>,          // len = num_components + 1
    tail_shape: Vec<usize>,       // shared tail after ragged dim
}
```

Storage-compact: a single contiguous `Vec<T>` instead of `num_components`
independent allocations. Mirrors upstream NJT's `_values` + `_offsets`
shape (`torch/nested/_internal/nested_tensor.py:55-80`).

The `from_sequences(seqs, lengths, tail_shape)` constructor (`nested.rs:967`)
validates:
- `sequences.len() == lengths.len()` (component-count parity).
- Per-component `seqs[i].len() == lengths[i] * prod(tail_shape)`.
- Length × `tail_numel` doesn't overflow (`checked_mul`).
- At least one sequence (empty input errors).

### Production consumers

- `ferrotorch-core/src/lib.rs:172` `pub use nested::{NestedTensor,
  PackedNestedTensor, nested_scaled_dot_product_attention}` — the
  crate-root re-export is the boundary. R-DEFER-1 S5 grandfathering
  applies: existing pub API surface; the type IS the boundary.
- `ferrotorch-core/src/gpu_dispatch.rs:3281` — comment block referencing
  "Per-component dispatch from `nested_scaled_dot_product_attention`"
  documenting how the backend's flash-attention kernel is invoked from
  the per-component dispatch path.

There is no in-tree non-test `NestedTensor` consumer in
`ferrotorch-core/src/**/*.rs` outside `nested.rs` itself plus the
`lib.rs` re-export. End-user code in downstream model crates (LLM
batching with variable-length sequences) is the natural consumer of
this surface.

## Parity contract

`parity_ops = []`. Indirect parity:
- `to_padded` / `from_padded` are inverse operations; the round-trip on
  any nested tensor must reproduce the original component-by-component.
  Verified by `nested_round_trip_padded` (test in the `nested.rs`
  test module).
- `nested_scaled_dot_product_attention` on a non-ragged (uniform-length)
  input must produce element-by-element equal output to a regular
  batched `scaled_dot_product_attention` on the same data — verified
  by the composite-path tests and the GPU-fast-path probe (gated on
  `gpu` feature + hardware).

A direct parity sweep against PyTorch's `torch.nested.*` API would
require a Python oracle that builds an equivalent `NestedTensor` and
calls the same ops; that's achievable but currently out of scope
(tracked under the nested-tensor parity follow-up).

## Verification

```
cargo test -p ferrotorch-core --lib nested::tests
```

Expected: all tests pass, 0 failed.

GPU residency + GPU-kernel paths are exercised by integration probes
under `ferrotorch-core/tests/` gated on the `gpu` feature + hardware.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `NestedTensor::new` at `ferrotorch-core/src/nested.rs:50-96` validating ndim + non-ragged shape parity. Non-test production consumer: `ferrotorch-core/src/lib.rs:172` `pub use nested::{NestedTensor, ...}`. R-DEFER-1 S5 grandfathering: existing pub API surface (#806/#291); the type IS the boundary public API for variable-length-sequence workflows. |
| REQ-2 | SHIPPED | impl: `num_components` at `ferrotorch-core/src/nested.rs:100`, `ragged_dim` at `:106`, `tensors` at `:112`, `ndim` at `:118`, `consistent_shape` at `:123`, `ragged_lengths` at `:128`. Non-test production consumer: `lib.rs:172` re-export + the GPU fast paths within `nested.rs` itself (e.g. `nested.rs:198` `self.tensors.iter().map(|t| t.shape()[self.ragged_dim]).max()` uses `tensors()` indirectly). |
| REQ-3 | SHIPPED | impl: `to_padded` at `ferrotorch-core/src/nested.rs:163-240` with GPU fast path `try_to_padded_gpu` at `:258-377`. Non-test production consumer: `lib.rs:172` re-export. R-DEFER-1 S5 grandfathering applies — boundary public API for the padded-vs-nested data interchange. |
| REQ-4 | SHIPPED | impl: `from_padded` at `ferrotorch-core/src/nested.rs:401`-(continuation), with GPU fast path at `:450-454` via `try_from_padded_gpu`. Non-test production consumer: `lib.rs:172` re-export. R-DEFER-1 S5 grandfathering. |
| REQ-5 | SHIPPED | impl: `pub fn nested_scaled_dot_product_attention<T: Float>` at `ferrotorch-core/src/nested.rs:657-770` with GPU dispatch helper `try_flash_attention_gpu_component` at `:775`. Non-test production consumer: `ferrotorch-core/src/lib.rs:172` re-exports `nested_scaled_dot_product_attention`. R-DEFER-1 S5 grandfathering: existing pub fn (#806). |
| REQ-6 | SHIPPED | impl: `pub struct PackedNestedTensor<T: Float>` at `ferrotorch-core/src/nested.rs:938`; constructor `from_sequences` at `:967-1010+`. Non-test production consumer: `ferrotorch-core/src/lib.rs:172` re-exports `PackedNestedTensor`. R-DEFER-1 S5 grandfathering (#291). |
| REQ-7 | SHIPPED | impl: `FerrotorchError::InvalidArgument` at `nested.rs:52, :59, :408, :415, :437, :664, :682, :973, :978, :990`; `ShapeMismatch` at `:78, :281, :701, :706`; `DeviceMismatch` at `:281`. No `panic!` / `unwrap()` / `expect()` in production paths (the one `.unwrap()` at `nested.rs:726` for `T::from(d_k).unwrap()` is on the type-safe int→T conversion path where the source is always within bounds; should be migrated to `numeric_cast::cast` per #815 as a no-blocker cleanup follow-up). Non-test production consumer: every caller propagates the structured error via `?`. |
