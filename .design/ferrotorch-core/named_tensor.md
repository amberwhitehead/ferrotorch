# NamedTensor — dim-name annotations on `Tensor<T>`

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/NamedTensorUtils.h
  - aten/src/ATen/NamedTensor.h
-->

## Summary

`ferrotorch-core/src/named_tensor.rs` defines `NamedTensor<T>` — a
`Tensor<T>` paired with one optional name per dimension. Mirrors
PyTorch's `Tensor.refine_names` / `align_to` / `rename` experimental
named-tensor feature (`aten/src/ATen/NamedTensor.h`). The names are
**advisory**: ops like `align_to` permute dims by name to match a
target ordering, but the underlying tensor math is unchanged. This
gives users a way to *annotate* tensors and reorder them by name
before passing into the position-based op surface, covering the most
common "did I get the batch dim right?" bug class for attention /
einsum prep.

Crosslink #621.

## Requirements

- REQ-1: `pub struct NamedTensor<T: Float>` storing the inner
  `Tensor<T>` + a `Vec<Option<String>>` of per-dim names. `names.len()
  == inner.ndim()` always; `None` entries mark anonymous dims.
- REQ-2: Constructors: `new(inner, names)`, `refined(inner, &[&str])`
  (with `""` for anonymous). `new` validates name count vs ndim and
  rejects duplicate non-None names. Mirrors `tensor.refine_names(...)`
  upstream.
- REQ-3: Accessors: `tensor(&self) -> &Tensor<T>` (borrow), `into_tensor`
  (consume), `names`, `shape`, `ndim`, `numel`.
- REQ-4: Lookups: `dim_index(name)`, `size_of(name)`. Mirrors
  `tensor.size("dim_name")` upstream.
- REQ-5: `rename(&[(old, new)])` — replace some names according to a
  mapping. Unmapped names stay; `None` names are unchanged. Mirrors
  `tensor.rename(...)` upstream.
- REQ-6: `align_to(&[target_names])` — permute dims to match a target
  ordering. Errors if a target name is not present. Mirrors
  `tensor.align_to(...)` upstream.
- REQ-7: `detached()` — strip names (all `None`). Useful before passing
  into ops that don't preserve names.
- REQ-8: `Display` impl prints `NamedTensor(shape=..., names=[...])`.
- REQ-9: Structured errors on duplicate names / unknown names / length
  mismatch. No panics in production. R-CODE-2.

## Acceptance Criteria

- [x] AC-1: `named_tensor_basic_construction` at `named_tensor.rs:202`.
- [x] AC-2: `named_tensor_rejects_length_mismatch` at `named_tensor.rs:212`.
- [x] AC-3: `named_tensor_rejects_duplicate_names` at `named_tensor.rs:218`.
- [x] AC-4: `named_tensor_anonymous_dim_via_empty_string` at
  `named_tensor.rs:224`.
- [x] AC-5: `named_tensor_align_permutes_dims` at `named_tensor.rs:231` —
  `[batch=2, seq=3, feat=4]` aligned to `[feat, batch, seq]` yields shape
  `[4, 2, 3]`.
- [x] AC-6: `named_tensor_align_identity_is_clone` at
  `named_tensor.rs:243`.
- [x] AC-7: `named_tensor_align_rejects_unknown_name` at
  `named_tensor.rs:250`.
- [x] AC-8: `named_tensor_align_rejects_length_mismatch` at
  `named_tensor.rs:257`.
- [x] AC-9: `named_tensor_rename_replaces_specified_names` at
  `named_tensor.rs:264`.
- [x] AC-10: `named_tensor_detached_drops_names` at
  `named_tensor.rs:273`.
- [x] AC-11: `named_tensor_into_tensor_recovers_inner` at
  `named_tensor.rs:282`.
- [x] AC-12: `named_tensor_dim_index_lookup` at `named_tensor.rs:289`.

## Architecture

### Layout (`named_tensor in named_tensor.rs`)

```rust
pub struct NamedTensor<T: Float> {
    inner: Tensor<T>,
    names: Vec<Option<String>>,
}
```

The `Vec<Option<String>>` choice rather than `Vec<String>` + sentinel
("`*`" or "_") is intentional: `None` is mechanically `None`, not a
magic string. Upstream uses `at::Dimname::Wildcard()` for the same
purpose (`aten/src/ATen/NamedTensor.h`), which is similarly a discrim-
inated null distinct from any user-provided name.

### Why "advisory" semantics (REQ-6 callout)

PyTorch's named-tensor experiment aimed to intercept every op and
surface name mismatches at op boundaries (e.g.
`a[batch, seq] + b[seq, batch]` would error rather than broadcast).
ferrotorch does NOT intercept ops; we only let users **annotate**
tensors and **reorder** them by name before passing into the
position-based op surface. This is a strictly weaker contract but
covers the most common practical bug class (attention / einsum prep
where the batch dim moved). Future work could lift this to op-level
interception (mirroring upstream's experimental path), but that's
a separate translation effort.

### Constructors (`named_tensor.rs:32-73`)

- `new(inner, names)` — explicit `Vec<Option<String>>` form. Validates
  `names.len() == ndim` and rejects duplicates.
- `refined(inner, &[&str])` — convenience: maps `""` to `None` and
  passes through `Some(s.to_string())` for the rest. Used for the
  common "all dims named" case.

### `align_to` (`named_tensor.rs:137-163`)

```rust
fn align_to(&self, target_names: &[&str]) -> FerrotorchResult<Self> {
    // 1. Validate target_names.len() == self.ndim().
    // 2. Build a permutation: for each target name, find its index in self.names.
    // 3. Apply the permutation via `crate::methods::permute_t`.
    // 4. Construct a new NamedTensor with the permuted shape + permuted names.
}
```

Errors on:
- length mismatch (target has fewer / more names than ndim),
- target name not present in self.names.

Doesn't support anonymous dims in target (every target slot must be a
named dim). This is the same restriction PyTorch's `align_to` imposes
when called positionally — the upstream `align_to(...)` accepts a
mix of names and `...` to fill anonymous positions, which ferrotorch
does not yet support (future enhancement; tracked as a no-blocker
follow-up).

### Production consumers

- `ferrotorch-core/src/lib.rs:147` `pub use named_tensor::NamedTensor`
  — the crate-root re-export is the boundary. R-DEFER-1 S5
  grandfathering applies: existing pub API surface (#621); the type IS
  the public boundary.

Same status as `ComplexTensor` — there is no in-tree non-test consumer
of `NamedTensor` in `ferrotorch-core/src/**/*.rs` outside `named_tensor.rs`
itself plus the `lib.rs` re-export. Downstream user code that wants
named-tensor annotations imports `ferrotorch_core::NamedTensor` and
uses the type at the boundary; the internal op surface remains
position-based. This is the documented contract: `NamedTensor` is an
**annotation overlay**, not a parallel op set.

## Parity contract

`parity_ops = []`. The parity surface is the indirect `align_to` /
`refine_names` semantic correctness vs upstream:
- `refine_names(["batch", "seq", "feat"])` followed by `align_to(["feat",
  "batch", "seq"])` produces a permuted tensor with shape `[4, 2, 3]` if
  the input was `[2, 3, 4]` — same as upstream.
- Duplicate names error.
- Unknown target names in `align_to` error.

A full parity test against `torch.Tensor.refine_names(...)` /
`align_to(...)` would require a Python oracle that calls those methods
and compares post-permute element-by-element. The 12 in-file tests pin
the semantic surface without needing the Python oracle; an explicit
op-db entry could be added but is currently out of scope.

## Verification

```
cargo test -p ferrotorch-core --lib named_tensor::tests
```

Expected: 12 tests pass, 0 failed (one per AC).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct NamedTensor<T: Float>` at `NamedTensor in ferrotorch-core/src/named_tensor.rs` with `Vec<Option<String>>` names. Non-test production consumer: `ferrotorch-core/src/lib.rs` `pub use named_tensor::NamedTensor` — the boundary public API. R-DEFER-1 S5 grandfathering: existing pub API surface (#621), the type IS the boundary. |
| REQ-2 | SHIPPED | impl: `new in ferrotorch-core/src/named_tensor.rs` (validates name count + rejects duplicates at `new in ferrotorch-core/src/named_tensor.rs`), `refined in ferrotorch-core/src/named_tensor.rs`. Non-test production consumer: `lib.rs` re-export. Tests: `named_tensor_basic_construction` at `lib.rs`, `named_tensor_rejects_duplicate_names` at `lib.rs`. |
| REQ-3 | SHIPPED | impl: `tensor in ferrotorch-core/src/named_tensor.rs`, `into_tensor in ferrotorch-core/src/named_tensor.rs`, `names in ferrotorch-core/src/named_tensor.rs`, `shape in ferrotorch-core/src/named_tensor.rs`, `ndim in ferrotorch-core/src/named_tensor.rs`, `numel in ferrotorch-core/src/named_tensor.rs`. Non-test production consumer: `shape in lib.rs` re-export. Test: `named_tensor_into_tensor_recovers_inner` at `shape in lib.rs`. |
| REQ-4 | SHIPPED | impl: `dim_index in ferrotorch-core/src/named_tensor.rs`, `size_of in ferrotorch-core/src/named_tensor.rs`. Non-test production consumer: `lib.rs` re-export. Test: `named_tensor_dim_index_lookup` at `lib.rs`. |
| REQ-5 | SHIPPED | impl: `rename in ferrotorch-core/src/named_tensor.rs`. Non-test production consumer: `lib.rs` re-export. Test: `named_tensor_rename_replaces_specified_names` at `lib.rs`. |
| REQ-6 | SHIPPED | impl: `align_to in ferrotorch-core/src/named_tensor.rs` using `crate::methods::permute_t` for the permutation. Non-test production consumer: `lib.rs` re-export + internal consumer at `lib.rs` `crate::methods::permute_t(&self.inner, &perm)?`. Tests: `named_tensor_align_permutes_dims` at `lib.rs`, `named_tensor_align_identity_is_clone` at `lib.rs`, `named_tensor_align_rejects_unknown_name` at `lib.rs`. |
| REQ-7 | SHIPPED | impl: `detached in ferrotorch-core/src/named_tensor.rs`. Non-test production consumer: `lib.rs` re-export. Test: `named_tensor_detached_drops_names` at `lib.rs`. |
| REQ-8 | SHIPPED | impl: `Display` impl at `NamedTensor in ferrotorch-core/src/named_tensor.rs`. Non-test production consumer: `lib.rs` re-export — every `format!("{}", nt)` callsite that handles a `NamedTensor`. |
| REQ-9 | SHIPPED | impl: `FerrotorchError::ShapeMismatch` at `named_tensor in named_tensor.rs, `; `InvalidArgument` at `, `. No `panic!` / `unwrap` / `expect` in production paths. Non-test production consumer: callers propagate the structured error via `?`. Tests: `named_tensor_rejects_length_mismatch in named_tensor.rs`, `named_tensor_rejects_duplicate_names in named_tensor.rs`, `named_tensor_align_rejects_length_mismatch in named_tensor.rs`. |
