# Meta-device propagation

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - torch/_subclasses/meta_utils.py
  - torch/_meta_registrations.py
  - aten/src/ATen/native/ExpandUtils.cpp
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/meta_propagate.rs` is the fast-path helper that lets
every tensor op shortcircuit data computation when its inputs live on
`Device::Meta`. Meta tensors carry shape + dtype + device but no
backing memory; they exist so the user can dry-run a model to
determine output shapes without allocating activations. Mirrors the
`@register_meta` pattern in `torch/_meta_registrations.py` and the
`torch._subclasses.meta_utils.MetaConverter` machinery.

## Requirements

- REQ-1: `unary_same_shape(input) -> Option<Tensor<T>>` — when `input`
  is meta, return `Some(zeros_meta(input.shape()))`; otherwise `None`.
  Mirrors `torch._meta_registrations.meta_unary_inplace` shape rule.
- REQ-2: `binary_broadcast(a, b) -> Option<Tensor<T>>` — when BOTH
  inputs are meta, return the broadcast meta tensor; when neither is
  meta, return `None`; when mixed (one meta + one real), error with
  `DeviceMismatch`. Mirrors
  `torch._meta_registrations.meta_binary_op_with_broadcast`.
- REQ-3: `reduce_dim(input, dim, keepdim) -> Option<Tensor<T>>` —
  reduction along a single (possibly negative) axis. When `keepdim`
  is true the axis becomes size 1; otherwise it is removed. Errors
  with `InvalidArgument` for scalar inputs or out-of-bounds dim.
  Mirrors the `meta_sum_dim` / `meta_mean_dim` shape rules.
- REQ-4: `reduce_all(input) -> Option<Tensor<T>>` — full-reduction
  fast path: returns a scalar (0-D) meta tensor when input is meta.
- REQ-5: `matmul(a, b) -> Option<Tensor<T>>` — matmul shape inference
  for 1×1, 2×1, 1×2, 2×2, and batched (>=3D) cases. Batch dimensions
  broadcast per PyTorch's standard `infer_size` rules. Mirrors
  `torch._meta_registrations.meta_mm` / `meta_bmm` /
  `meta_matmul`.

## Acceptance Criteria

- [x] AC-1: `unary_same_shape` returns the input shape for a meta
  tensor and `None` for a CPU tensor (`meta_propagate.rs:245-258`).
- [x] AC-2: `binary_broadcast` of `[3,1]` × `[1,4]` → `[3,4]`
  (`meta_propagate.rs:264-271`).
- [x] AC-3: `binary_broadcast` with one meta + one real errors
  (`meta_propagate.rs:281-287`).
- [x] AC-4: `reduce_dim(_, 1, false)` removes axis 1;
  `reduce_dim(_, 1, true)` keeps it at size 1
  (`meta_propagate.rs:301-315`).
- [x] AC-5: `reduce_dim` accepts negative axis
  (`meta_propagate.rs:317-322`).
- [x] AC-6: `matmul` shape rules for 1D/2D/batched cases
  (`meta_propagate.rs:360-407`).
- [x] AC-7: End-to-end: a dry-run MLP on meta inputs yields the
  correct logit shape without allocating any forward activations
  (`meta_propagate.rs:513-538`).
- [x] AC-8: `cargo test -p ferrotorch-core --lib meta_propagate`
  passes.

## Architecture

Every helper returns `FerrotorchResult<Option<Tensor<T>>>`:

- `Ok(Some(t))` — the inputs were all meta; here is the meta result.
  Caller short-circuits and returns this without running the data
  kernel.
- `Ok(None)` — no inputs were meta; caller runs the normal compute
  path.
- `Err(e)` — mixed meta + real, or invalid args (negative dim out of
  range, scalar inputs to matmul).

The op authors call these at the top of their forward functions, e.g.
`grad_fns/arithmetic.rs:376-391`'s `add` does:

```rust
if let Some(out) = meta_propagate::binary_broadcast(a, b)? {
    return Ok(out);
}
```

This is the canonical pattern; you can see it in
`grad_fns/activation.rs:695, 737, 792, 853, 938, 992` and
`grad_fns/reduction.rs:107, 234, 414, 772`.

Internally each helper:

1. Checks `is_meta()` on its inputs.
2. Computes the expected output shape (delegating to
   `crate::shape::broadcast_shapes` for binary broadcast / matmul
   batched).
3. Calls `crate::creation::zeros_meta(out_shape)` to construct the
   no-data meta tensor.

## Parity contract

`parity_ops = []`. Meta tensors carry no element values so the
parity-sweep oracle cannot compare numerical output. The shape rules
themselves are stress-tested in the e2e tests at the bottom of the
file (chained arithmetic, reductions, matmul, MLP forward).

The shape rules are byte-for-byte identical to PyTorch's meta
registrations (matmul broadcasting follows `infer_size` exactly,
reduction with `keepdim` matches `torch.sum(x, dim=d, keepdim=True)`).

## Verification

- Unit tests at `meta_propagate.rs:228-538` cover every helper plus
  five end-to-end pipeline tests that route real public ops (`add`,
  `mul`, `neg`, `sqrt`, `sum`, `sum_dim`, `mean_dim`, `matmul`,
  activations) on meta inputs through the production codepaths.
- The MLP dry-run at `meta_propagate.rs:513-538` is the canonical
  user-facing scenario: build a model, run forward on meta inputs,
  observe the logit shape, never allocate the hidden activations.

```bash
cargo test -p ferrotorch-core --lib meta_propagate
```

Expected: 22 tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `unary_same_shape in ferrotorch-core/src/meta_propagate.rs` mirrors `torch._meta_registrations.meta_unary_inplace`; non-test consumers: `unary_same_shape in grad_fns/activation.rs` (`relu`), `relu in grad_fns/activation.rs` (`sigmoid`), `sigmoid in grad_fns/activation.rs` (`tanh`), `tanh in grad_fns/activation.rs` (`gelu`), `gelu in grad_fns/activation.rs` (`silu`), `silu in grad_fns/activation.rs` (`softmax`). |
| REQ-2 | SHIPPED | impl: `binary_broadcast` at `ferrotorch-core/src/meta_propagate.rs:50` mirrors `torch._meta_registrations.meta_binary_op_with_broadcast`; non-test consumer: `grad_fns/arithmetic.rs:380` (called from `add`'s forward and via dispatch from `sub`/`mul`/`div`/... — every binary broadcast op routes through this guard). |
| REQ-3 | SHIPPED | impl: `reduce_dim` at `ferrotorch-core/src/meta_propagate.rs:72` mirrors `meta_sum_dim` shape rule; non-test consumer: `grad_fns/reduction.rs:772` (`sum_dim`) and the e2e test at `meta_propagate.rs:466-481` pins it for `sum_dim`/`mean_dim` via production callsites. |
| REQ-4 | SHIPPED | impl: `reduce_all` at `ferrotorch-core/src/meta_propagate.rs:109`; non-test consumer: `grad_fns/reduction.rs:107` (`sum_all`), `:234` (`mean_all`), `:414` (`prod_all`). |
| REQ-5 | SHIPPED | impl: `matmul` at `ferrotorch-core/src/meta_propagate.rs:127` mirrors `torch._meta_registrations.meta_mm` / `meta_matmul` shape rules; non-test consumer: `ferrotorch-core/src/ops/linalg.rs::matmul` invokes this guard before dispatching to the data kernel — see the e2e test at `meta_propagate.rs:484-491` which routes through the production `op_matmul` entry point. |
