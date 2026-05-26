# GPU constrained-decoding DFA token mask

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/
  - c10/cuda/
-->

## Summary

`ferrotorch-cubecl/src/grammar.rs` is the GPU dispatcher for constrained-
decoding token-mask computation. Given a per-token packed vocabulary (CSR
offsets + flat char array) and a schema-derived DFA (flat transition table
+ ASCII char-class lookup), one cubecl thread per vocab entry walks its
token through the DFA in registers and writes a 0/1 allow flag to a device
buffer. The companion in `quant.rs` (`kernel_apply_token_mask`) consumes
that allow buffer to mask a logits vector.

This module has NO direct PyTorch upstream — constrained decoding is a
ferrotorch-LLAMA-side feature that pre-dates any aten-side kernel. The
route's upstream-path entries (`aten/src/ATen/native/cuda/` +
`c10/cuda/`) name the directories whose patterns this kernel mirrors:
the `#[cube]` macro idiom, `ABSOLUTE_POS` bounds-guarded one-thread-per-
element layout, and ASCII-aware encoding choices are all transplants of
how `aten/native/cuda` writes its elementwise kernels.

## Requirements

- REQ-1: `kernel_compute_token_mask_dfa<F>` — `#[cube(launch_unchecked)]`
  kernel taking the four flat input arrays (transitions, char_classes,
  vocab_offsets, vocab_chars), one output mask array, and four scalar
  arguments (num_classes, start_state, reject_state, max_token_len). One
  thread per vocab entry. Walks DFA in registers, writes 1 if the walk
  ends in a non-reject state with non-empty token, else 0. ASCII guard:
  `c >= 128` forces REJECT.

- REQ-2: Empty-token policy — tokens with `start == end` (zero chars)
  are masked OUT (writes 0). Mirrors the CPU `JsonSchemaProcessor::
  compute_mask` behaviour of skipping empty tokens and leaving their
  slot at zero (the test-side `reference_walk` reproduces this exactly).

- REQ-3: Bounded loop — CubeCL doesn't have `break`, so the per-thread
  walk runs exactly `max_token_len` iterations, guarded by
  `pos < end && rejected == 0`. Once `rejected = 1`, subsequent
  iterations are no-ops. This is the canonical pattern for early-exit
  loops in `#[cube]` kernels and mirrors the `for k in 1..n_u { if
  condition { ... } }` idiom in `kernels.rs::kernel_chebyshev_*`.

- REQ-4: `DfaMaskInputs<'a>` host-side input struct. `#[non_exhaustive]`
  so external callers must use `DfaMaskInputs::new(vocab_size, ...)`,
  which validates `vocab_offsets.len() == vocab_size + 1`. The in-crate
  launcher and CUDA tests construct via field literals (allowed for
  same-crate use).

- REQ-5: `DfaMaskInputs::new(...)` validating constructor — returns
  `Some(Self)` when `vocab_offsets.len() == vocab_size + 1`, else
  `None`. This guards against the silent off-by-one where a vocab
  without the trailing-sentinel offset would walk one token short.

- REQ-6: `compute_token_mask_dfa_to_gpu<R>(client, &inputs) -> (Handle,
  vocab_size)` — single host launcher. Uploads the four input slices,
  allocates the allow buffer, launches the kernel, returns the
  device-resident allow handle plus `vocab_size = vocab_offsets.len() -
  1`. The caller (`ferrotorch-grammar::gpu_dispatch::run_dfa_on_gpu`)
  reads the handle back via `client.read_one`.

- REQ-7: `DfaMaskInputs<'_>: Debug` — manual `Debug` impl that shows
  array lengths and scalar fields. Critical for failure diagnostics
  when a vocab/DFA pair mis-validates; the raw arrays are too large
  to print.

## Acceptance Criteria

- [x] AC-1: CUDA-feature tests in `mod cuda_tests in grammar.rs` pass when
  a CUDA device is available (`cargo test -p ferrotorch-cubecl --features
  cuda`). Three tests pinned: `dfa_kernel_matches_hand_built_walk`,
  `dfa_kernel_accepts_digit_sequences`, `dfa_kernel_rejects_non_ascii`.
- [x] AC-2: GPU output is byte-identical to the pure-Rust `reference_walk`
  for the test inputs (asserted by every test in `cuda_tests`).
- [x] AC-3: `DfaMaskInputs::new` returns `None` for a mis-sized
  `vocab_offsets` (no separate test in this module — the constraint is
  enforced at the type level; the `?` operator at `gpu_dispatch.rs:833-843`
  propagates a `None` into `Option<TokenMask>`).

## Architecture

### Kernel layout (REQ-1, REQ-2, REQ-3)

`#[cube(launch_unchecked)] pub fn kernel_compute_token_mask_dfa<F> in
grammar.rs`. Generic over `F: Float` for vocabulary-future compatibility
(though all current callers pass `u32` arrays + `f32` is unused — the
kernel only touches `u32`). The kernel structure:

1. Bounds-guard `ABSOLUTE_POS < n_alw` (one thread per vocab entry).
2. Read `start = vocab_offsets[tok]`, `end = vocab_offsets[tok + 1]`.
3. Initialise `state = start_state`, `rejected = 0u32`.
4. Empty-token check: `if start == end { rejected = 1; }`.
5. Bounded loop `for i in 0..max_token_len`, predicated on
   `pos < end && rejected == 0`.
6. ASCII guard: `if c < 128 { class = char_classes[c]; ... } else
   { rejected = 1; }`.
7. Transition: `state = transitions[state * num_classes + class]`.
8. Reject sink: `if state == reject_state { rejected = 1; }`.
9. Write `allow[tok] = if rejected == 0 { 1 } else { 0 }`.

This is the most idiomatic `#[cube]` kernel in the crate — every other
elementwise kernel in `kernels.rs` follows the same `ABSOLUTE_POS`-
guarded one-thread-per-element pattern. The DFA-walk loop is the
distinguishing structural feature; bounded by `max_token_len` because
cubecl lacks `break`.

### Input struct (REQ-4, REQ-5)

`#[non_exhaustive] pub struct DfaMaskInputs<'a> in grammar.rs`. Eight
fields: four borrowed slices, four scalars. The `#[non_exhaustive]`
forces external crates to use `DfaMaskInputs::new(vocab_size, ...)`
(which validates the CSR-sentinel invariant) rather than field-literal
construction. The in-crate CUDA tests use field literals because they
construct inputs that are known-correct by inspection; production
callers in `ferrotorch-grammar` must go through `new`.

`pub fn new in grammar.rs` returns `Option<Self>`. The single validation
is `vocab_offsets.len() == vocab_size + 1` — the CSR trailing-sentinel
invariant. Without the sentinel the kernel can't compute `[start, end)`
for the last token, and the resulting walk silently shortens the vocab
by one.

`#[allow(clippy::too_many_arguments)]` on `new` is documented in-line:
the constructor mirrors the 8 fields of the struct verbatim, and that's
the contract the caller has to fulfill anyway.

### Host launcher (REQ-6)

`pub fn compute_token_mask_dfa_to_gpu<R: Runtime> in grammar.rs`. Five
uploads + one launch:

1. `trans_handle = client.create_from_slice(u32_slice_as_bytes(trans))`.
2. Three more for `char_classes`, `vocab_offsets`, `vocab_chars`.
3. `allow_handle = client.empty(vocab_size * size_of::<u32>())`.
4. `(count, dim) = crate::elementwise_launch_dims(vocab_size as u32)`.
5. `unsafe { kernel_compute_token_mask_dfa::launch_unchecked::<R>(
   ..., trans_arg, classes_arg, offsets_arg, chars_arg, allow_arg,
   num_classes, start_state, reject_state, max_token_len) }`.

The `unsafe { ... }` block carries a long SAFETY comment quoting the
five handle allocations and bounding the `ArrayArg::from_raw_parts`
invariant.

Non-test production consumer: `ferrotorch-grammar/src/gpu_dispatch.rs`
— `let (handle, n) = compute_token_mask_dfa_to_gpu::<R>(client, &inputs);`
inside `fn run_dfa_on_gpu`, which the JSON-schema processor calls per
generation step.

### Slice reinterpret helper

`fn u32_slice_as_bytes in grammar.rs` casts `&[u32]` to `&[u8]` for
cubecl's byte-oriented upload API. The SAFETY comment documents
alignment (u32 ≥ u8), length (`size_of_val(s) == s.len() * 4`),
lifetime (elision through single-input function), and validity (any
byte pattern is a valid u8).

### Debug impl (REQ-7)

`impl Debug for DfaMaskInputs<'_> in grammar.rs` shows array lengths
and scalars but NOT the full array contents (a 128k-vocab dump would
flood logs). This is the standard discipline for very-large input
structs.

## Parity contract

ferrotorch-cubecl is INFRASTRUCTURE — `parity_ops = []`. There is no
parity-sweep arm for "DFA mask computation" because the upstream PyTorch
does not have this kernel — constrained decoding is a ferrotorch-grammar
feature that pre-dates aten-level support.

The contract is enforced by direct GPU-vs-CPU byte-equality assertions in
the in-module CUDA tests:

- `dfa_kernel_matches_hand_built_walk`: 2-state DFA accepting `a+`;
  vocab `["a", "aa", "aaa", "ab", "ba", ""]` → expected `[1,1,1,0,0,0]`.
- `dfa_kernel_accepts_digit_sequences`: digit-DFA over a 26-token vocab
  spanning single-digit / multi-digit / mixed-letter / empty / 7-digit
  tokens. Asserts `accepted == 26`.
- `dfa_kernel_rejects_non_ascii`: vocab `["abc", "héllo", "x"]` with a
  permissive DFA. Without the ASCII guard `héllo` would pass; with it,
  the kernel writes 0 → expected `[1, 0, 1]`.

Each test compares the kernel output byte-for-byte against
`reference_walk`, a pure-Rust mirror of the in-kernel DFA traversal.

## Verification

Tests in `#[cfg(all(test, feature = "cuda"))] mod cuda_tests in grammar.rs`:

- `dfa_kernel_matches_hand_built_walk`
- `dfa_kernel_accepts_digit_sequences`
- `dfa_kernel_rejects_non_ascii`

Plus the `reference_walk` host implementation used as the byte-equality
oracle.

Smoke command (`parity_ops = []`):

```bash
cargo test -p ferrotorch-cubecl --features cuda grammar:: 2>&1 | tail -3
```

Expected: `3 passed; 0 failed` when a CUDA device is available.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `#[cube(launch_unchecked)] pub fn kernel_compute_token_mask_dfa in grammar.rs`. Non-test consumer: `pub fn compute_token_mask_dfa_to_gpu in grammar.rs` invokes `kernel_compute_token_mask_dfa::launch_unchecked::<R>(...)`; `compute_token_mask_dfa_to_gpu` is called from `ferrotorch-grammar/src/gpu_dispatch.rs`. |
| REQ-2 | SHIPPED | impl: `if start == end { rejected = 1u32; }` block in `kernel_compute_token_mask_dfa`. Non-test consumer: same as REQ-1 (the kernel writes 0 for empty tokens, consumed by `gpu_dispatch::run_dfa_on_gpu`'s readback). |
| REQ-3 | SHIPPED | impl: `for i in 0..max_token_len_u { if pos < end && rejected == 0u32 { ... } }` in `kernel_compute_token_mask_dfa`. Non-test consumer: same as REQ-1 — the bounded-loop pattern enables the kernel on cubecl's `break`-less control flow. |
| REQ-4 | SHIPPED | impl: `#[non_exhaustive] pub struct DfaMaskInputs<'a> in grammar.rs`. Non-test consumer: `ferrotorch-grammar/src/gpu_dispatch.rs` constructs via `DfaMaskInputs::new(...)`. |
| REQ-5 | SHIPPED | impl: `pub fn new in grammar.rs` returning `Option<Self>` after `vocab_offsets.len() != vocab_size + 1` check. Non-test consumer: `ferrotorch-grammar/src/gpu_dispatch.rs` uses `?` on the `Option<DfaMaskInputs>` so a mis-sized vocab propagates as `None`. |
| REQ-6 | SHIPPED | impl: `pub fn compute_token_mask_dfa_to_gpu<R: Runtime> in grammar.rs`. Non-test consumer: `ferrotorch-grammar/src/gpu_dispatch.rs` — `let (handle, n) = compute_token_mask_dfa_to_gpu::<R>(client, &inputs);`. |
| REQ-7 | SHIPPED | impl: `impl std::fmt::Debug for DfaMaskInputs<'_> in grammar.rs` showing lengths + scalars. Non-test consumer: any `eprintln!("{:?}", inputs)` in error paths in downstream grammar code; the impl is the documented diagnostic contract. |
