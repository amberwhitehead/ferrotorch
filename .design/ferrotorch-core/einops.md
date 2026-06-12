# Einops-style Tensor Rearrangement

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - torch/_namedtensor_internals.py
-->

## Summary

`ferrotorch-core/src/einops.rs` implements `rearrange`, `repeat`, and
`reduce` — the three primary operations of the
[einops](https://github.com/arogozhnikov/einops) library. These mirror
no exact PyTorch upstream (the einops package is separate from
`torch`), but the semantics are commonly used in PyTorch programs and
the route's upstream path
`torch/_namedtensor_internals.py` is the closest PyTorch-side analog
(named-tensor axis identification). The implementation parses
`"left -> right"` patterns with parenthesized groups for
merge/split dims, resolves named-axis sizes from input shape +
`axes_lengths`, and emits the corresponding reshape / permute /
reduction sequence.

## Requirements

- REQ-1: `rearrange(input, pattern)` — pure shape transformation
  (transpose / merge / split) without value reduction or replication.
  Mirrors `einops.rearrange`. Errors on axis-count mismatch / size
  contradictions.
- REQ-2: `rearrange_with(input, pattern, axes_lengths)` — same as
  REQ-1 but accepts user-supplied sizes for split-dim sub-axes when
  inference is ambiguous.
- REQ-3: `repeat(input, pattern, axes_lengths)` — replicates the
  tensor along NEW axes (axes present on right but not left). Mirrors
  `einops.repeat`.
- REQ-4: `reduce(input, pattern, reduction, axes_lengths)` — reduces
  along axes present on left but not right. `EinopsReduction` enum
  values: `Mean`, `Sum`, `Max`, `Min`. Mirrors `einops.reduce`.
- REQ-5: Pattern parser — handles parenthesized groups
  `"b (c h w) -> b c h w"`, validates duplicate-axis-name detection,
  rejects empty groups / malformed characters. (`einops.rs:79-195`)
- REQ-6: Named-axis size resolution — `resolve_sizes` infers split-
  dim sub-axes when all-but-one is known, validates products match
  parent dim, accepts user overrides via `axes_lengths`.
  (`einops.rs:211-300`)

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib einops::tests` passes
  (covers `rearrange`, `repeat`, `reduce` with merge/split/transpose
  patterns).
- [x] AC-2: `"b c h w -> b (c h w)"` merges trailing dims via reshape.
- [x] AC-3: `"b (c h) w -> b c h w"` with `c=2` splits the second dim;
  resolves `h = dim_size / 2`.
- [x] AC-4: Duplicate axis name on the same side → error.
- [x] AC-5: Empty parenthesized group → error.
- [x] AC-6: `reduce` with `Sum` along absent-on-right axes — verified
  numerically against direct sum.

## Architecture

The pattern parser (`einops.rs:50-195`) builds a `ParsedPattern` with
`Vec<AxisSpec>` for each side. `AxisSpec::Single(name)` is a bare
axis; `AxisSpec::Group(names)` is a parenthesized list. `parse_side`
walks character-by-character, with `read_axis_name` consuming
`[a-zA-Z0-9_]+` runs. Duplicate-detection runs `flatten_axes` per
side and inserts into a `HashMap<&str, _>`, erroring on second
insert.

`resolve_sizes` at `einops.rs:211` performs the named-axis size pass:

1. For each `AxisSpec::Single` on the left, assign `sizes[name] =
   input_shape[dim_idx]`.
2. For each `AxisSpec::Group(names)` on the left, treat the
   corresponding input dim as a split target. If user `axes_lengths`
   supply sizes for all sub-axes, validate their product equals the
   parent dim. If sizes for all but one sub-axis are known, infer the
   remaining size as `parent / known_product`. If two or more are
   unknown, error.

`rearrange` at `einops.rs:393` is the entry point for pure-shape ops:
parses, resolves sizes, then applies the reshape sequence
(`einops.rs:424` enters `rearrange_with` which routes to the
differentiable `reshape → permute → reshape` composition). `repeat`
at `:514` similarly resolves sizes then replicates along new-axis
positions via `reshape → permute → reshape → expand → reshape`.
`reduce` at `:614` identifies axes-to-reduce as those present on left
but not right, then applies the chosen reduction through
`reshape → permute → reshape → sum_dim/cummax/cummin → reshape`.
Per CORE-061 (#1755) every step participates in autograd; per
CORE-062 (#1756) axis reordering happens by name in the permute step,
never by positional coordinate walks.

The reduction path for `Mean` (`einops.rs`) lifts a scalar
`n_recip = 1.0 / count` via `crate::creation::scalar` so the
mean can be expressed as `sum * n_recip` — this is the only path
that constructs a tensor.

**Non-test consumers**: the einops surface is re-exported at
`lib.rs:143` as `pub use einops::{EinopsReduction, rearrange,
rearrange_with, reduce, repeat}`. Downstream consumers in
`ferrotorch-nn` use these for the standard "patch embedding" / "head
split" / "mixer" patterns: e.g. `rearrange(x, "b (h p1) (w p2) c ->
b (h w) (p1 p2 c)")` for ViT patch embedding. The non-test
consumer for REQ-1 ... REQ-4 is the public surface itself
re-exported through `lib.rs`; per goal.md S5 ("boundary methods ARE
the public API"), the `einops::rearrange` symbol exposed as
`ferrotorch_core::rearrange` IS the production consumer.

## Parity contract

`parity_ops = []` (einops is not in `torch.ops`; no torch.einops
op_db entry). Numerical correctness is unit-tested against
hand-computed expected outputs for each pattern shape.

## Verification

`cargo test -p ferrotorch-core --lib einops::tests` covers parser
edge cases, named-axis resolution, all three operations
(`rearrange`/`repeat`/`reduce`), and the four `EinopsReduction`
variants.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `rearrange` at `einops.rs:393` (closest upstream behavioural analog `einops.rearrange`, no PyTorch counterpart); non-test consumer: re-exported as `ferrotorch_core::rearrange` at `lib.rs:143`; used by `ferrotorch-nn` ViT/MLP-Mixer-style patch-shuffling code |
| REQ-2 | SHIPPED | impl: `rearrange_with` at `einops.rs:424`; non-test consumer: re-exported as `ferrotorch_core::rearrange_with` at `lib.rs:143` |
| REQ-3 | SHIPPED | impl: `repeat` at `einops.rs:514`; non-test consumer: re-exported as `ferrotorch_core::repeat` at `lib.rs:143` |
| REQ-4 | SHIPPED | impl: `reduce in einops.rs` with `EinopsReduction` enum at `EinopsReduction in einops.rs`; non-test consumer: re-exported as `ferrotorch_core::reduce` and `EinopsReduction` at `einops in lib.rs` |
| REQ-5 | SHIPPED | impl: `parse_pattern`/`parse_side`/`read_axis_name` at `einops.rs:79-195`; non-test consumer: all four public APIs (REQ-1..REQ-4) invoke `parse_pattern` as their first step |
| REQ-6 | SHIPPED | impl: `resolve_sizes` at `einops.rs:211`; non-test consumer: all four public APIs invoke `resolve_sizes` after `parse_pattern` |
