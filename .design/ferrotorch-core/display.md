# Tensor Display Formatting

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - torch/_tensor_str.py
-->

## Summary

`ferrotorch-core/src/display.rs` implements `impl<T: Float>
std::fmt::Display for Tensor<T>` so `format!("{t}")` produces a
PyTorch-style string mirroring the output of `torch._tensor_str._str`
(`torch/_tensor_str.py`). The implementation covers scalar, 1-D, 2-D,
and 3-D+ tensors with the same elision-by-`...` pattern, the same
`grad_fn=<...>` / `requires_grad=true` suffix, and the same
4-decimal-place numeric format.

## Requirements

- REQ-1: 0-D (scalar) tensors print as `tensor(value)` with the value
  formatted via `std::fmt::Display` for `T`. Mirrors
  `torch._tensor_str._scalar_str` (`torch/_tensor_str.py`).
- REQ-2: 1-D tensors print as `tensor([v0, v1, v2, ...])` with up to
  6 visible values; longer tensors are elided as `[v0, v1, v2, ...,
  vn-2, vn-1, vn]`. Mirrors `torch._tensor_str._vector_str`.
- REQ-3: 2-D tensors print as `tensor([[row0], [row1], ...])` with
  newline + 8-space indentation between rows. Up to 6 rows / 6 columns
  visible; larger matrices elide both axes with `...`. Mirrors
  `torch._tensor_str._matrix_str`.
- REQ-4: 3-D and higher tensors print a summary
  `tensor(<{numel} elements>, shape={...})` rather than the full nested
  bracket form — a deliberate divergence (R-DEV-7) to keep the Rust
  `Display` impl compact; PyTorch recursively renders all dims.
- REQ-5: Autograd metadata suffix — `, grad_fn=<NAME>` when the tensor
  has a `grad_fn`, else `, requires_grad=true` when `requires_grad()`
  is set. Mirrors `torch._tensor_str._add_suffixes` parameter coverage.
- REQ-6: `data_vec()`-based read so non-contiguous and CUDA tensors
  display correctly via the existing host-bounce path. The CUDA bounce
  is acceptable HERE because formatting is inherently a debug /
  printf-style operation that is never on a hot path.
- REQ-7: Inaccessible (meta-tensor) fallback — `tensor(<inaccessible>,
  shape=...)` when `data_vec()` returns an error.

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib display::tests`
  passes (6 tests at `display.rs:140-198`).
- [x] AC-2: `format!("{}", scalar(3.14f32).unwrap())` contains `"3.14"`
  (`test_display_scalar`).
- [x] AC-3: `format!("{}", t)` for `t` with `requires_grad=true`
  contains `"grad_fn=<AddBackward>"` when produced by `&a + &b`
  (`test_display_with_grad_fn` at `:169`).
- [x] AC-4: 1-D tensors of length > 6 contain `"..."`
  (`test_display_large_1d_truncated`).
- [x] AC-5: 3-D tensors display `"24 elements"` and `"shape=[2, 3, 4]"`
  (`test_display_3d_summary`).

## Architecture

The single `impl Display for Tensor<T>` at `display.rs:8-138` walks
the tensor's shape and dispatches on dimensionality:

- `shape.is_empty()` branch (`:18-30`): formats the single value, with
  optional `grad_fn=<NAME>` or `requires_grad=true` suffix from
  `self.grad_fn().unwrap().name()` / `self.requires_grad()`.
- `shape.len() == 1` branch (`:33-60`): emits up to 6 entries; longer
  vectors use `[v0, v1, v2, ..., vn-3, vn-2, vn-1]`.
- `shape.len() == 2` branch (`:62-116`): nested via `display_row`
  closure that handles the per-row elision separately; outer-row
  elision uses `,\n        ...` separator.
- `shape.len() >= 3` branch (`:118-128`): summary form
  `tensor(<{numel} elements>, shape={shape:?}{suffix})`.

The data fetch at `:12` uses `self.data_vec()` (NOT `self.data()`).
This is intentional: `data_vec` resolves non-contiguous CPU views
(returns a materialised copy) and downloads CUDA tensors through the
host bounce. For meta tensors `data_vec` errors; the `match Err` arm
emits `tensor(<inaccessible>, shape=...)` (`:14`).

The `4`-decimal-place format (`{v:.4}`) at `:42`, `:50`, etc., matches
PyTorch's default `torch.set_printoptions(precision=4)` — the eager
default that downstream PyTorch users see.

**Non-test consumers**: the `Display` impl is invoked by every
`println!("{tensor}")` / `format!("{tensor}")` / `eprintln!`
call site in tests and downstream crates, AND by the
`Debug` / `Display` derives on structs that embed `Tensor`. In
production it is used by `crate::tensor::Tensor::fmt` (the
`Debug` impl on `tensor.rs` is dependent on this `Display` path
for the `grad_fn` annotation). At the workspace level, the
panic messages from `cargo test` that print failed tensor
comparisons go through this impl.

## Parity contract

`parity_ops = []` (utility file). Display formatting is text-output,
not floating-point computation; no parity-sweep sample compares string
output. The matching with `torch._tensor_str` is informational —
the spec is "looks roughly like torch's output" not "byte-for-byte".

## Verification

`cargo test -p ferrotorch-core --lib display::tests` exercises 6
tests across scalar, 1-D, 2-D, 3-D, grad-fn suffix, requires-grad
suffix, and 1-D-truncation paths.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `Display` 0-D branch at `display.rs:18-30` mirrors `torch._tensor_str._scalar_str`; non-test consumer: every `println!("{t}")` call site in `ferrotorch-core`, the Rust standard library's `format!` machinery is the production consumer |
| REQ-2 | SHIPPED | impl: 1-D branch at `display.rs:33-60`; non-test consumer: panic-message tensor formatting in `ferrotorch-nn` / `ferrotorch-vision` debug paths |
| REQ-3 | SHIPPED | impl: 2-D branch at `display.rs:62-116`; non-test consumer: top-level `Display` impl is the production consumer for every `format!("{t}")` |
| REQ-4 | SHIPPED | impl: 3-D+ summary branch at `display.rs:118-128`; non-test consumer: top-level `Display` impl. NB: this is an R-DEV-7 deviation — PyTorch recursively renders all dims; ferrotorch summarises |
| REQ-5 | SHIPPED | impl: suffix logic at `display.rs:24-28` (scalar) and `:131-135` (1D/2D); non-test consumer: every autograd-graph test's printed tensor goes through this; the `test_display_with_grad_fn` test pins the exact `grad_fn=<AddBackward>` format |
| REQ-6 | SHIPPED | impl: `self.data_vec()` call at `display.rs:12`; non-test consumer: implicit — every non-contiguous and CUDA tensor printed in production hits this path |
| REQ-7 | SHIPPED | impl: `Err` arm at `display.rs:14`; non-test consumer: every meta-tensor printed in shape-inference code (e.g. `creation::zeros_meta` outputs in dry-run model construction) |
