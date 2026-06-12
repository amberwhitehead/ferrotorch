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
  all components have the same ndim, identical sizes on every
  non-ragged dim, and a SINGLE shared device (CORE-070 / #1764: mixed
  CPU/CUDA lists fail at construction with `DeviceMismatch` instead of
  later with an opaque data-access error). Returns structured error on
  mismatch. Mirrors `torch.nested.nested_tensor([t1, t2, ...])`
  (`aten/src/ATen/native/nested/NestedTensorFactories.cpp`).
- REQ-2: Accessors: `num_components`, `ragged_dim`, `tensors(&self) ->
  &[Tensor<T>]`, `ndim`, `consistent_shape`, `ragged_lengths`.
- REQ-3: `to_padded(pad_value)` — convert to a dense padded tensor of
  shape `[batch, d_0, ..., max_L, ..., d_{n-1}]`. Three paths:
  (a) graph-preserving differentiable composition (cat with a constant
  pad filler + unsqueeze + cat) when grad is enabled and any component
  tracks gradients — gradients flow back to the component leaves like
  the differentiable `torch.nested.to_padded_tensor` (CORE-066 / #1760);
  (b) GPU fast path (P4 of #806) using `fill_f{32,64}` +
  `strided_scatter_f{32,64}` — components stay on-device throughout;
  (c) CPU path materializing each component's LOGICAL view via
  `data_vec` (non-contiguous views supported, CORE-070 / #1764).
  CUDA components that reach the CPU path error loudly (R-LOUD-1).
- REQ-4: `from_padded(tensor, lengths, ragged_dim)` — inverse of
  `to_padded`. Graph-preserving narrow → contiguous → reshape chain when
  the padded source tracks gradients (backward scatters component
  cotangents into the source, zeros in the pad region — CORE-066 /
  #1760); GPU fast path via `narrow` + `.contiguous()`; CPU fallback
  materializes the logical view (`data_vec`). Mirrors
  `torch.nested.from_padded`.
- REQ-5: `nested_scaled_dot_product_attention(q, k, v)` — per-component
  scaled-dot-product attention. Dispatch order per component:
  (a) differentiable composite (`mm_bt` → broadcast `mul` by
  `1/sqrt(d_k)` → `softmax` → `matmul`) when grad is enabled and any of
  q/k/v tracks gradients — device-aware, real backward edges
  (CORE-066 / #1760); (b) GPU FlashAttention kernel when the component
  is CUDA and fits the kernel's regime (`d_k <= 128`, `d_v <= 128`);
  (c) the same device-aware composite when the flash kernel DECLINES a
  CUDA component (head dim > 128, unsupported dtype, no backend) —
  result stays on CUDA (CORE-067 / #1761); (d) scalar CPU loop for
  non-grad CPU components (logical views via `data_vec`). Mirrors
  `torch._C._nested_scaled_dot_product` and
  `torch.nn.functional.scaled_dot_product_attention` on nested inputs.
- REQ-6: `PackedNestedTensor` — packed flat-storage layout
  (`data: Vec<T>`, `offsets: Vec<usize>`, `lengths: Vec<usize>`,
  `tail_shape: Vec<usize>`). Invariants (enforced by the centralized
  `validate_packed_layout` from EVERY constructor — CORE-068 / #1762):
  `offsets.len() == num_components + 1`, `lengths.len() ==
  num_components`, `offsets[0] == 0`, monotonic offsets,
  `offsets[i+1] - offsets[i] == lengths[i] * tail_numel`,
  `offsets[num_components] == data.len()`, where `tail_numel` is the
  ACTUAL `product(tail_shape)` — `1` for an empty tail, `0` when a tail
  dim is zero (CORE-069 / #1763; lengths are stored because element
  offsets degenerate for zero tails — torch's jagged `_offsets` count
  rows, `torch/nested/_internal/nested_tensor.py`). `from_data_tensor`
  additionally requires the documented flat 1-D input and rejects zero
  tails (lengths not derivable) with a structured error.
  `from_nested` rejects grad-tracking components loudly (R-LOUD-3 —
  the packed layout drops graphs by design; CORE-066 / #1760).
  `mean_per_component` returns NaN for empty components, matching
  torch's empty floating reduction (`torch.tensor([]).mean()` is nan;
  CORE-071 / #1765).
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

### `NestedTensor<T>` layout (`NestedTensor in nested.rs`)

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
(each `tensors[i]` carries its own grad-fn). Since CORE-066 (#1760)
the conversions honor that promise: `to_padded` / `from_padded` /
`nested_scaled_dot_product_attention` attach real backward edges when
their inputs track gradients (regression suite:
`tests/audit_core066_nested_autograd.rs`, CPU + CUDA, live-torch
oracles). The packed layout (`PackedNestedTensor` below) is the
storage-compact alternative when autograd granularity isn't needed —
and refuses grad-tracking components loudly rather than silently
detaching them. All components share one device (CORE-070 / #1764).

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
2. Differentiable composite (`attention_component_composite`) when grad
   is enabled and any of q/k/v tracks gradients: `mm_bt` → broadcast
   `mul` by the constant `1/sqrt(d_k)` → `softmax` → `matmul`. All
   primitives are device-aware and attach backward edges
   (CORE-066 / #1760).
3. GPU FlashAttention dispatch: `try_flash_attention_gpu_component`
   asks the `GpuBackend` for a flash-attention call. Returns `Ok(true)`
   if the kernel fired, `Ok(false)` if the backend declined
   (unsupported dtype, shape outside kernel regime, etc.).
4. CUDA composite fallback: when the flash kernel declines a CUDA
   component, the SAME device-aware composite from step 2 runs —
   the result stays on CUDA (CORE-067 / #1761; regression at head
   sizes 128/129 in `tests/audit_core067_cuda_attention_fallback.rs`).
5. Scalar CPU loop for non-grad CPU components: `Q @ K^T`, scale by
   `1/sqrt(d_k)`, row-wise softmax, multiply by `V`; inputs are
   materialized as LOGICAL views via `data_vec` so non-contiguous
   views run (CORE-070 / #1764).

The GPU paths keep the output `Tensor<T>` GPU-resident.

### `PackedNestedTensor` layout (`nested.rs:1130-1160`)

```rust
pub struct PackedNestedTensor<T: Float> {
    data: Vec<T>,                 // flat concat in component order
    offsets: Vec<usize>,          // len = num_components + 1
    lengths: Vec<usize>,          // ragged lengths (authoritative; CORE-069)
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
- `ferrotorch-core/src/gpu_dispatch.rs` — comment block referencing
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
cargo test -p ferrotorch-core --test conformance_nested_sparse
# CORE-066..071 remediation regressions (#1760-#1765):
cargo test -p ferrotorch-core --features gpu \
  --test audit_core066_nested_autograd \
  --test audit_core067_cuda_attention_fallback \
  --test audit_core068_packed_offsets_validation \
  --test audit_core069_packed_zero_tail \
  --test audit_core070_nested_device_views \
  --test audit_core071_empty_mean_nan
```

Expected: all tests pass, 0 failed.

GPU residency + GPU-kernel paths are exercised by integration probes
under `ferrotorch-core/tests/` gated on the `gpu` feature + hardware,
with R-ORACLE-3 device assertions on results AND gradients.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `NestedTensor::new` at `ferrotorch-core/src/nested.rs:50-96` validating ndim + non-ragged shape parity. Non-test production consumer: `ferrotorch-core/src/lib.rs:172` `pub use nested::{NestedTensor, ...}`. R-DEFER-1 S5 grandfathering: existing pub API surface (#806/#291); the type IS the boundary public API for variable-length-sequence workflows. |
| REQ-2 | SHIPPED | impl: `num_components in ferrotorch-core/src/nested.rs`, `ragged_dim in ferrotorch-core/src/nested.rs`, `tensors in ferrotorch-core/src/nested.rs`, `ndim in ferrotorch-core/src/nested.rs`, `consistent_shape in ferrotorch-core/src/nested.rs`, `ragged_lengths in ferrotorch-core/src/nested.rs`. Non-test production consumer: `nested in lib.rs` re-export + the GPU fast paths within `nested.rs` itself (e.g. `nested in nested.rs` `self.tensors.iter().map(|t| t.shape()[self.ragged_dim]).max()` uses `tensors()` indirectly). |
| REQ-3 | SHIPPED | impl: `to_padded` at `ferrotorch-core/src/nested.rs:163-240` with GPU fast path `try_to_padded_gpu` at `:258-377`. Non-test production consumer: `lib.rs:172` re-export. R-DEFER-1 S5 grandfathering applies — boundary public API for the padded-vs-nested data interchange. |
| REQ-4 | SHIPPED | impl: `from_padded` at `ferrotorch-core/src/nested.rs:401`-(continuation), with GPU fast path at `:450-454` via `try_from_padded_gpu`. Non-test production consumer: `lib.rs:172` re-export. R-DEFER-1 S5 grandfathering. |
| REQ-5 | SHIPPED | impl: `pub fn nested_scaled_dot_product_attention<T: Float>` at `ferrotorch-core/src/nested.rs:780-905` with the differentiable/CUDA-fallback composite `attention_component_composite` (CORE-066/#1760, CORE-067/#1761) and the GPU dispatch helper `try_flash_attention_gpu_component`. Non-test production consumer: `ferrotorch-core/src/lib.rs:172` re-exports `nested_scaled_dot_product_attention`. R-DEFER-1 S5 grandfathering: existing pub fn (#806). |
| REQ-6 | SHIPPED | impl: `pub struct PackedNestedTensor<T: Float>` at `PackedNestedTensor in ferrotorch-core/src/nested.rs`; constructor `from_sequences in ferrotorch-core/src/nested.rs`. Non-test production consumer: `ferrotorch-core/src/lib.rs` re-exports `PackedNestedTensor`. R-DEFER-1 S5 grandfathering (#291). |
| REQ-7 | SHIPPED | impl: `FerrotorchError::InvalidArgument` at `nested in nested.rs, , , , , , , , , `; `ShapeMismatch` at `, , , `; `DeviceMismatch` at `nested in nested.rs`. No `panic!` / `unwrap()` / `expect()` in production paths (the one `.unwrap()` at `nested in nested.rs` for `T::from(d_k).unwrap()` is on the type-safe int→T conversion path where the source is always within bounds; should be migrated to `numeric_cast::cast` per #815 as a no-blocker cleanup follow-up). Non-test production consumer: every caller propagates the structured error via `?`. |
