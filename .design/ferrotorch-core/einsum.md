# Einstein Summation (einsum)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/Linear.cpp
  - torch/functional.py
-->

## Summary

`ferrotorch-core/src/einsum.rs` implements `einsum` and the
differentiable wrapper `einsum_differentiable` — the Rust mirror of
`torch.einsum` (`torch/functional.py:einsum`). The forward parser
accepts explicit (`"ij,jk->ik"`) and implicit (`"ij,jk"`) notation,
ASCII uppercase/lowercase labels, ellipsis, shared-label size-1/zero
broadcasting, and one or more operands. Single-input equations
(transpose, axis-sum, trace, diagonal extraction), two-input
contractions (matmul, batched matmul, generic permute+reshape+matmul),
and n-ary contractions lower to Rust/GPU-aware primitive composites
under crosslink #803 / #821 / #822 / #1861 — pure CPU detours for CUDA
operands are forbidden.

## Requirements

- REQ-1: `einsum(equation, &[&Tensor<T>])` — forward einsum. Accepts
  one or more inputs and rejects zero inputs with `InvalidArgument`.
  Mirrors `torch.einsum` (`torch/functional.py:einsum`).
- REQ-2: Equation parser — explicit `"lhs->rhs"` and implicit
  `"lhs"` (output = ellipsis axes first, then sorted unique
  single-occurrence labels). Supports `[A-Za-z]`, ellipsis, explicit
  ellipsis output, and ellipsis reduction. Mirrors `torch.einsum`
  parsing in `torch/functional.py:einsum` and the C++
  `at::native::einsum` (`aten/src/ATen/native/Linear.cpp`).
- REQ-3: Single-input ops — pure permutation (e.g. `"ij->ji"`), axis
  sum / projection (`"ij->i"`), full reduction (`"ij->"`), and the
  repeated-index extension (`"ii->"` trace, `"ii->i"` diagonal,
  implicit `"ii"`) decomposed on-device through the strided_copy GPU
  surface (#821). Mirrors `at::native::einsum` single-operand path.
- REQ-4: Two-input contractions — matmul `"ij,jk->ik"` via
  `grad_fns::linalg::matmul_differentiable`, batched matmul
  `"bij,bjk->bik"` via `grad_fns::linalg::bmm`, generic
  permute+reshape+matmul/bmm decomposition for multi-axis
  contractions (#822). Mirrors `at::native::einsum` two-operand
  contraction path.
- REQ-5: `einsum_differentiable(equation, inputs)` — wraps the
  forward result with `EinsumBackwardSingle`, `EinsumBackwardTwo`, or
  `EinsumBackwardN` when grad is enabled and any input requires grad.
  Participates in autocast (classified `ReducedPrecision` via
  `autocast_guard("einsum")`).
- REQ-6: Dimension-map consistency check — `build_dim_map` validates
  every repeated index inside one operand resolves to the same size,
  and shared labels across operands are PyTorch-broadcast-compatible
  (`same`, `1`, or `0`/`1` zero contraction cases); output indices
  must reference known dims.
- REQ-7: GPU dispatch discipline — `Err(NotImplementedOnCuda)` for
  equations whose structure cannot be lowered to existing GPU
  primitives. No silent CPU detour.

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib einsum::tests` passes
  (covers single-input, two-input, trace, diagonal, batched matmul,
  matrix-vector product, hadamard).
- [x] AC-2: `einsum("ij,jk->ik", &[&a, &b])` produces the same output
  as `crate::grad_fns::linalg::matmul_differentiable(a, b)`.
- [x] AC-3: `einsum("ii->", &[&a])` computes the trace via on-device
  strided diagonal view (#821 path).
- [x] AC-4: `einsum_differentiable` attaches an `EinsumBackward*`
  grad_fn when `any_requires_grad && is_grad_enabled()`.
- [x] AC-5: Parity-sweep `einsum` at `--seeds 8` returns ≥1 passed
  sample — SHIPPED 2026-05-26 (closes #1532). Runner arm at
  `tools/parity-sweep/runner/src/main.rs` decodes op_db's
  `[List[Tensor], equation: str]` envelope and dispatches through
  `ferrotorch_core::einsum::einsum_differentiable`.
- [x] AC-6: CORE-167 coverage proves uppercase labels, ellipsis
  permutation/reduction, shared-label size-1 broadcasting, and n-ary
  forward/backward against live PyTorch-derived oracles.

## Architecture

The parser at `einsum.rs` splits on `->`, tokenises labels and
ellipsis, expands ellipsis to private internal labels, and (in
implicit mode) builds the output as ellipsis axes followed by
sorted-unique single-occurrence labels via a `BTreeMap<char, usize>`.
`build_dim_map` walks every (subscripts, tensor) pair, asserting
matching ndim, exact repeated-label sizes within one operand, and
PyTorch-compatible broadcast sizes across operands; output indices
must be present in the dim map.

`einsum` is the eager forward entry. It dispatches to `einsum_single`
(1 input), `einsum_two` (2 inputs), or left-to-right pairwise
contraction for n-ary inputs. For single-input equations the handler
distinguishes pure permutation, axis
reduction, full reduction, and the repeated-index diagonal/trace
extension (#821). For two-input equations it identifies the
contracting indices (present in BOTH inputs but NOT in output),
batch indices (present in both AND in output), and free indices
(present in one input AND output). The general path permutes each
operand to `[batch_dims, free_dims, contract_dims]` for A and
`[batch_dims, contract_dims, free_dims]` for B, reshapes to 3-D
`[batch, M, K]` / `[batch, K, N]`, applies `bmm`, then reshapes +
permutes back (#822).

`einsum_differentiable` wraps the forward result. It runs
`autocast_guard("einsum")` (classified as `ReducedPrecision` in the
autocast policy), runs forward, and — if grad is enabled and any
input requires grad — attaches `EinsumBackwardSingle { equation,
input }`, `EinsumBackwardTwo { equation, a, b }`, or
`EinsumBackwardN { equation, inputs }`. The backward
implementation recursively builds the partner-input einsum needed
for the VJP: for two-input contractions, `dL/dA = einsum("dL/dC, B
indices on conjugate side", grad_output, b)` and symmetric for `B`.

The storage transfer uses `into_storage_and_shape()` at `into_storage_and_shape in einsum.rs` /
`:1575` to keep the forward's GPU-resident output GPU-resident
through the autograd wrap (avoids the host bounce that an earlier
implementation triggered via `data_vec()`).

**Non-test consumer**: `crate::methods::Tensor::einsum` at
`methods.rs:638-642` is the method-style entry point —
`tensor.einsum("ij,jk->ik", &[other])` invokes
`crate::einsum::einsum_differentiable`. Re-exported at `lib.rs:144`
as `pub use einsum::{einsum, einsum_differentiable}`.

## Parity contract

`parity_ops = ["einsum"]`. The parity-sweep oracle ingests
`torch.einsum` op_db samples and compares against
`ferrotorch_core::einsum_differentiable`. As of 2026-05-26 (#1532
closed) the runner has an `einsum` dispatch arm at
`tools/parity-sweep/runner/src/main.rs`'s `dispatch_f32`. CORE-167
removes the former ellipsis / uppercase / n-ary parser-surface reason
for skips; the runner skip table must not treat those equation classes
as legitimate exclusions.

## Verification

`cargo test -p ferrotorch-core --lib einsum::tests` covers parser,
single-input, two-input, repeated-index, and autograd paths.
`cargo test -p ferrotorch-core --test audit_core167_einsum_surface`
covers the CORE-167 PyTorch surface: uppercase labels, ellipsis
permutation/reduction, size-1 shared-label broadcasting, n-ary
forward, and n-ary backward.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `einsum` in `einsum.rs` mirrors `torch.einsum` (`torch/functional.py:einsum`); non-test consumer: `Tensor::einsum` at `methods.rs:638` invokes `einsum_differentiable` which routes to `einsum` |
| REQ-2 | SHIPPED | impl: `parse_equation` at `einsum in einsum.rs`; non-test consumer: every call into `einsum_differentiable` → `einsum` triggers `parse_equation` first |
| REQ-3 | SHIPPED | impl: `einsum_single` (referenced at `einsum_single in einsum.rs`), repeated-index decomposition through `crate::stride_tricks::as_strided_copy` (#821 path) inside `einsum_single`; non-test consumer: `Tensor::einsum` at `einsum in methods.rs` |
| REQ-4 | SHIPPED | impl: `einsum_two` (referenced at `:1532`); non-test consumer: `Tensor::einsum` at `methods.rs:638` |
| REQ-5 | SHIPPED | impl: `einsum_differentiable` in `einsum.rs` with `EinsumBackwardSingle`/`EinsumBackwardTwo`/`EinsumBackwardN` wrap; non-test consumer: `Tensor::einsum` at `methods.rs:641` invokes `einsum_differentiable` (the method-surface boundary IS the public API per goal.md S5) |
| REQ-6 | SHIPPED | impl: `build_dim_map` at `einsum.rs:149`; non-test consumer: every call into `einsum_differentiable` → `einsum` → `build_dim_map` |
| REQ-7 | SHIPPED | impl: documented in the module-level `//!` comment at `einsum in einsum.rs` and the `Err(NotImplementedOnCuda)` returns inside `einsum_two` for non-decomposable equations; non-test consumer: `Tensor::einsum` |
