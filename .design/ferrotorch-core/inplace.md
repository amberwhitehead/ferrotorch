# In-place tensor ops (trailing-underscore convention)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/native/BinaryOps.cpp
  - aten/src/ATen/native/UnaryOps.cpp
-->

## Summary

`ferrotorch-core/src/inplace.rs` ships eleven public `Tensor<T: Float>` methods
that mutate a tensor's underlying storage following PyTorch's trailing-
underscore convention: `add_scalar_` / `mul_scalar_` / `fill_` / `zero_` /
`add_` / `add_scaled_` / `sub_` / `sub_scaled_` / `mul_` / `div_` / `clamp_`.
They mirror the
in-place overloads declared alongside the structured out-of-place ops in
`aten/src/ATen/native/BinaryOps.cpp` (binary `add_` / `sub_` / `mul_` / `div_`
families plumbed through `TORCH_IMPL_FUNC(<op>_out)` at lines 434/441/447 and
the `Tensor& <op>_(Tensor& self, ...)` C++ entry points at 896–998 / 1144–1167
/ 1180), `aten/src/ATen/native/Fill.cpp` (`fill_` at line 47, `zero_` at line
150 — both reachable through the route's UnaryOps neighbour by the
`at::fill_` callback inside `TORCH_IMPL_FUNC(clamp_out)`), and
`aten/src/ATen/native/TensorCompare.cpp:831` (`clamp_out`, dispatched by
`Tensor::clamp_` on the Python side).

Every method routes through a single `check_inplace_allowed` autograd guard
(`inplace.rs:35-57`) and an `unsafe { self.update_data(...) }` or
`unsafe { self.update_storage(...) }` storage swap. The autograd guard
rejects two PyTorch-parity error cases: tensors with a `grad_fn` (i.e. non-
leaf graph nodes) and leaf tensors with `requires_grad = true`. The latter
matches torch's `RuntimeError: a leaf Variable that requires grad is being
used in an in-place operation` (cf. `torch/autograd/_functions/utils.py` and
`torch/_C/_VariableFunctions.pyi`).

`add_scaled_` is the single load-bearing op that the parity-sweep `add` arm
uses to express torch's `tensor.add_(other, *, alpha=1)` API. The other
nine ops have no in-tree non-test production caller today — the parity-
sweep runner at `tools/parity-sweep/runner/src/main.rs:495` is the only
non-test call site of `add_scaled_` in the repo, and per goal.md R-DEFER-1
the parity-sweep dispatch table is structurally a test-side consumer. Every
REQ in the table below is therefore NOT-STARTED.

## Requirements

- REQ-1: `Tensor::add_scalar_(value)` — in-place scalar add: `self += value`.
  Mirrors torch's `Tensor.add_(scalar)` overload (per
  `aten/src/ATen/native/BinaryOps.cpp:1180 Tensor& add_(Tensor& self, const
  Scalar& other, const Scalar& alpha)` which delegates to
  `self.add_(wrapped_scalar_tensor(other), alpha)`, and per the user-facing
  docstring `add_(other, *, alpha=1) -> Tensor` at
  `torch/_tensor_docs.py:379`). ferrotorch's `add_scalar_` drops torch's
  `alpha` kwarg (defaults to 1) — when alpha is 1, `add_scalar_(value)` is
  semantically `Tensor.add_(value, alpha=1)`. The function returns `&Self`
  for method chaining (torch returns `Tensor&` for the same reason).

- REQ-2: `Tensor::mul_scalar_(value)` — in-place scalar multiply: `self *=
  value`. Mirrors torch's `Tensor.mul_(scalar)` overload (per
  `aten/src/ATen/native/BinaryOps.cpp:996 Tensor& mul_(Tensor& self, const
  Scalar& other)` and docstring `mul_(value) -> Tensor` at
  `torch/_tensor_docs.py:3441`).

- REQ-3: `Tensor::fill_(value)` — fill every element with a scalar value.
  Mirrors torch's `Tensor.fill_(scalar)` per `aten/src/ATen/native/Fill.cpp:47
  Tensor& fill_(Tensor& self, const Scalar& value)` (delegates to
  `fill_out`), and docstring `fill_(value) -> Tensor` at
  `torch/_tensor_docs.py:1955`.

- REQ-4: `Tensor::zero_()` — fill every element with zero. Mirrors torch's
  `Tensor.zero_()` per `aten/src/ATen/native/Fill.cpp:150 Tensor& zero_(Tensor
  &self)` (small-tensor CPU fast path uses memset, otherwise dispatches to
  `self.fill_(0)`), and docstring `zero_() -> Tensor` at
  `torch/_tensor_docs.py:6381`. ferrotorch's `zero_` is implemented as
  `self.fill_(T::zero())` matching the upstream fallback path.

- REQ-5: `Tensor::add_(other)` — elementwise in-place add with broadcasting:
  `self += other`. Mirrors torch's `Tensor.add_(other)` per the
  `TORCH_IMPL_FUNC(sub_out)` dispatch at `BinaryOps.cpp:434-439` (which
  shares `add_stub` with `add_out`) and the user-facing docstring `add_(other,
  *, alpha=1) -> Tensor` at `torch/_tensor_docs.py:379`. ferrotorch's `add_`
  is implemented as `self.add_scaled_(other, 1.0)` matching torch's
  `alpha=1` default.

- REQ-6: `Tensor::add_scaled_(other, alpha)` — elementwise in-place
  `self += alpha * other` with broadcasting. This is the full
  `torch.Tensor.add_(other, *, alpha=1)` contract per
  `torch/_tensor_docs.py:379`. The upstream implementation is the same
  `add_stub` family at `aten/src/ATen/native/BinaryOps.cpp:437` (via
  `TORCH_IMPL_FUNC(sub_out)` calling `add_stub(device_type(), *this,
  -alpha)`). ferrotorch's `add_scaled_` validates broadcast compatibility,
  preserves `self.shape()` (in-place ops cannot resize), and on the
  same-shape `alpha == 1.0` fast path uses the GPU `add_f32` kernel directly
  with no CPU round-trip; otherwise routes through `grad_fns::arithmetic::
  add_scaled` and swaps the result storage in.

- REQ-7: `Tensor::sub_(other, *, alpha=1)` — elementwise in-place subtract.
  Per torch's `Tensor.sub_(other, *, alpha=1) -> Tensor` at
  `torch/_tensor_docs.py:5113` and the C++ `Tensor& sub_(Tensor& self, const
  Tensor& other, const Scalar& alpha)` chain at `BinaryOps.cpp:1157`
  (subtract alias) which itself routes through `TORCH_IMPL_FUNC(sub_out)`
  at `BinaryOps.cpp:434-439` — literally `add_stub(device_type(), *this,
  -alpha)`. `pub fn sub_` in `inplace.rs` is a one-line delegation to
  `self.sub_scaled_(other, 1.0)` (post-commit 2f792bfc5), which inherits
  broadcasting and GPU dispatch from `add_scaled_` via `sub_scaled_`.
  REQ-7 remains NOT-STARTED because there is no non-test production consumer;
  the `alpha` kwarg gap is now closed by `sub_scaled_` (REQ-11), but `sub_`
  itself still needs a consumer to flip to SHIPPED (blocker #1211).

- REQ-8: `Tensor::mul_(other)` — elementwise in-place multiply with
  broadcasting. Per torch's `Tensor.mul_(value) -> Tensor` at
  `torch/_tensor_docs.py:3441` and `aten/src/ATen/native/BinaryOps.cpp:441
  TORCH_IMPL_FUNC(mul_out)`. ferrotorch's `mul_` at `inplace.rs:292` is
  shape-strict (rejects mismatched-shape inputs at lines 294-302).

- REQ-9: `Tensor::div_(other, *, rounding_mode=None)` — elementwise in-place
  divide with broadcasting. Per torch's `Tensor.div_(value, *,
  rounding_mode=None) -> Tensor` at `torch/_tensor_docs.py:1746` and
  `aten/src/ATen/native/BinaryOps.cpp:447 TORCH_IMPL_FUNC(div_out)`.
  ferrotorch's `pub fn div_ in inplace.rs` is shape-strict and carries no
  `rounding_mode` kwarg; the `"floor"` and `"trunc"` modes correspond to
  `floor_divide_` and `trunc_divide_` (themselves NOT-STARTED in
  `grad_fns/arithmetic.rs` per the arithmetic design doc's REQ-12 / #1197).

- REQ-10: `Tensor::clamp_(min, max)` — in-place clamp each element to
  `[min, max]`. Per torch's `Tensor.clamp_(min=None, max=None) -> Tensor`
  at `torch/_tensor_docs.py:1141` and the C++ structured kernel
  `TORCH_IMPL_FUNC(clamp_out)` at `aten/src/ATen/native/TensorCompare.cpp:831
  -854`, which: (a) if either bound is NaN, fills the entire result with
  NaN via `at::fill_(result, NaN)`; (b) otherwise dispatches to one of
  `clamp_scalar_stub` / `clamp_max_scalar_stub` / `clamp_min_scalar_stub`
  depending on which bounds are provided. ferrotorch's `clamp_` at
  `inplace.rs:385` requires BOTH bounds (no Optional/None case), rejects
  `min > max` up front, and lacks the NaN-fills-everything special case.

- REQ-11: `Tensor::sub_scaled_(other, alpha)` — elementwise in-place
  `self -= alpha * other` with broadcasting. This is the full
  `torch.Tensor.sub_(other, *, alpha=1)` contract per
  `torch/_tensor_docs.py:5113`. The upstream implementation is the same
  `add_stub` family at `aten/src/ATen/native/BinaryOps.cpp:434-439`:
  `TORCH_IMPL_FUNC(sub_out)` calls `add_stub(device_type(), *this, -alpha)`.
  ferrotorch's `sub_scaled_` mirrors that delegation byte-for-byte by
  calling `self.add_scaled_(other, -alpha)`, inheriting `add_scaled_`'s
  broadcasting, GPU fast path, shape-strict in-place validation, and
  autograd guards (R-DEV-1). `sub_scaled_` is the in-place non-test
  production consumer of the new out-of-place
  `arithmetic::sub_scaled` (which delegates similarly to
  `add_scaled(a, b, -alpha)`); together they ship the
  `torch.sub(input, other, *, alpha=1)` family across the
  out-of-place + in-place divide at PyTorch parity. Note: this is
  distinct from REQ-7 (`sub_`), which lacks an `alpha` kwarg and
  remains NOT-STARTED for its own consumer-wiring reasons (blocker
  #1211).

## Acceptance Criteria

- [x] AC-1: Every in-place method enforces the two PyTorch-parity autograd
  invariants: (a) reject when `self.grad_fn().is_some()` (non-leaf graph
  node), and (b) reject when `self.requires_grad() && self.is_leaf()`
  (requires-grad leaf). Covered by `fn check_inplace_allowed in inplace.rs`
  and exercised by tests
  `fn test_add_scalar_rejects_requires_grad_leaf in inplace.rs`,
  `fn test_mul_scalar_rejects_requires_grad_leaf in inplace.rs`,
  `fn test_fill_rejects_requires_grad_leaf in inplace.rs`,
  `fn test_zero_rejects_requires_grad_leaf in inplace.rs`,
  `fn test_clamp_rejects_requires_grad_leaf in inplace.rs`.
- [x] AC-2: `tensor.detach()` followed by an in-place op succeeds (detached
  tensors are mutable). Covered by
  `fn test_detached_tensor_allows_inplace in inplace.rs`.
- [x] AC-3: In-place ops can be chained via the returned `&Self` (matching
  torch's `Tensor&` chaining). Covered by
  `fn test_add_scalar_chaining in inplace.rs` and
  `fn test_mixed_inplace_chaining in inplace.rs`.
- [x] AC-4: `add_scalar_` and `mul_scalar_` work on both `f32` and `f64`.
  Covered by `fn test_inplace_ops_f64 in inplace.rs`.
- [x] AC-5: `fill_` accepts scalar-shape (0-d) tensors. Covered by
  `fn test_fill_scalar_tensor in inplace.rs`.
- [x] AC-6: `zero_` accepts empty tensors. Covered by
  `fn test_zero_empty_tensor in inplace.rs`.
- [x] AC-7: `clamp_(min, max)` returns an error when `min > max`. Covered
  by `test_clamp_invalid_range in ferrotorch-core/src/inplace.rs`.
- [x] AC-8: `add_scaled_` same-shape `alpha != 1.0` path mutates `self` to
  `self + alpha * other`. Covered by the conformance-sweep
  `inplace_add_scaled` arm in
  `ferrotorch-core/tests/conformance_elementwise.rs:1880-1921` (calls
  `t.add_(&other)`) and by the parity-sweep `add` arm
  `inplace=True, alpha=<various>` probes at
  `tools/parity-sweep/runner/src/main.rs:494-498` (the parity-sweep `add`
  audit reports 88/88 passed at seeds=8 per `parity_audit.json:5-72`).
- [ ] AC-9: `sub_(other, *, alpha=k)` for `k != 1` produces `self - k *
  other`. Blocked by #1211 (sub alpha gap, parallel to add_scaled / #1192).
- [ ] AC-10: `mul_(other)` and `div_(other)` accept broadcasting (e.g. `[B,
  T, C].mul_([1, T, 1])`). Blocked by #1212 / #1213.
- [ ] AC-11: `clamp_(NaN, max)` or `clamp_(min, NaN)` fills `self` with NaN
  (parity with `TORCH_IMPL_FUNC(clamp_out)` at TensorCompare.cpp:844-846).
  Blocked by #1214.
- [ ] AC-12: `clamp_(min=None, max=None)` is the no-op identity; `clamp_(min,
  None)` clamps below only; `clamp_(None, max)` clamps above only. Blocked
  by #1214 (current signature requires both bounds).
- [ ] AC-13: At least one non-test production consumer in
  `ferrotorch-{optim,nn,core}/src/**/*.rs` invokes each public in-place op.
  Blocked by #1205 / #1206 / #1207 / #1208 / #1209 / #1210 / #1211 / #1212
  / #1213 / #1214.

## Architecture

### Autograd guard (`inplace.rs:31-57`)

`check_inplace_allowed(tensor, op_name)` is the single safety predicate
every in-place op calls before mutating. It rejects two PyTorch-parity
cases: (a) `tensor.grad_fn().is_some()` returns
`FerrotorchError::InvalidArgument` with the grad_fn name in the message
(matching torch's `RuntimeError: a view of a leaf Variable that requires
grad is being used in an in-place operation`); (b) `requires_grad &&
is_leaf` returns the same error variant with a "would not be tracked by
autograd" message. The two checks together produce the same semantics as
torch's `at::AutogradMeta`-driven check in
`torch/csrc/autograd/VariableTypeUtils.h:check_inplace`.

### Layer 1: scalar-bound in-place ops (`inplace.rs:60-130`)

- `add_scalar_(value)` (`:69`) — calls `check_inplace_allowed`, takes a
  CPU-side `data_vec`, iterates `+= value`, swaps back via `unsafe
  update_data`. No GPU fast path; relies on `data_vec()` being device-
  transparent (CPU returns the storage Vec; GPU downloads). REQ-1
  NOT-STARTED — no non-test caller in the workspace. The only invocation
  outside `#[cfg(test)]` blocks is in conformance / parity-sweep test
  harnesses (see Verification). Blocker #1205.

- `mul_scalar_(value)` (`:89`) — same shape as `add_scalar_` with `*=` body.
  REQ-2 NOT-STARTED — no non-test caller. Blocker #1206.

- `fill_(value)` (`:109`) — constructs a fresh `vec![value; self.numel()]`
  and swaps via `unsafe update_data`. REQ-3 NOT-STARTED — no non-test
  caller (the natural callers are weight initializers in
  `ferrotorch-nn/src/init.rs`, which currently build storage directly
  rather than fill_-ing a pre-allocated tensor). Blocker #1207.

- `zero_()` (`:128`) — delegates to `self.fill_(T::zero())`. REQ-4
  NOT-STARTED — no non-test caller (the natural caller is
  `optim::zero_grad`, which currently zeroes through other paths). Blocker
  #1208.

### Layer 2: tensor-bound in-place ops (`inplace.rs:132-373`)

- `add_(other)` (`:147`) — a single-line wrapper over `add_scaled_(other,
  1.0)`. PyTorch parity for `Tensor.add_(other, *, alpha=1)`. REQ-5
  NOT-STARTED — no non-test caller. Blocker #1209.

- `pub fn add_scaled_ in inplace.rs` — the load-bearing op. Three paths:
  (1) same-shape, `alpha == 1.0`, both `is_cuda()`, `T == f32` → calls
  `gpu_backend().add_f32(self.gpu_handle()?, other.gpu_handle()?)` directly
  and swaps the resulting GPU storage in via `unsafe update_storage`. This
  is the only GPU-resident fast path in the file. (2) same-shape,
  `alpha == 1.0`, CPU or non-f32 → iterates `*a += b` on the CPU-side
  data_vec and swaps via `unsafe update_data`. (3) Anything else
  (broadcasting OR `alpha != 1`) → routes through
  `crate::grad_fns::arithmetic::add_scaled(self, other, alpha)` and swaps
  the resulting tensor's storage in via `into_storage_and_shape() +
  unsafe update_storage`. The third path explicitly validates
  `result.shape() == self.shape()` — broadcasting may shape-expand `other`
  to match `self`, but in-place ops cannot resize `self`.
  REQ-6 NOT-STARTED — `tools/parity-sweep/runner/src/main.rs:495` is the
  only non-test caller, and the parity-sweep runner's dispatch table is
  structurally a test-side consumer per goal.md R-DEFER-1. Blocker #1210.

- `sub_scaled_(other, alpha)` (`:265`) — the in-place sibling of
  `arithmetic::sub_scaled`. Body delegates to `self.add_scaled_(other,
  -alpha)` exactly as upstream `TORCH_IMPL_FUNC(sub_out)` delegates to
  `add_stub(device_type(), *this, -alpha)` at `BinaryOps.cpp:434-439`.
  Inherits `add_scaled_`'s same-shape GPU/SIMD fast path, broadcast
  dispatch, and in-place shape-strict invariant for free. REQ-11 SHIPPED
  (closes #1192): the out-of-place `arithmetic::sub_scaled` consumes this
  module's `add_scaled_` indirectly through `arithmetic::add_scaled`, and
  this in-place `sub_scaled_` is itself a direct non-test caller of the
  shared `add_scaled_` infrastructure.

- `pub fn sub_` in `inplace.rs` (`:311` post-commit 2f792bfc5) — a
  one-line delegation: `self.sub_scaled_(other, 1.0)`. The old 40-line
  body (which was shape-strict and routed through a dedicated GPU kernel)
  was replaced in commit 2f792bfc5; the current implementation inherits
  broadcasting, GPU dispatch, and the in-place shape-strict invariant from
  `add_scaled_` via `sub_scaled_`. REQ-7 NOT-STARTED — no non-test caller;
  the `alpha` kwarg gap is covered by `sub_scaled_` (REQ-11), but `sub_`
  itself requires a production consumer to flip to SHIPPED. Blocker #1211.

- `mul_(other)` (`:292`) — shape-strict GPU f32 fast path via `mul_f32`,
  CPU fallback. REQ-8 NOT-STARTED — broadcasting gap (torch's `mul_`
  inherits broadcasting from `mul_out`) plus no non-test caller. Blocker
  #1212.

- `div_(other)` (`:335`) — shape-strict GPU f32 fast path via `div_f32`,
  CPU fallback. IEEE-754 div-by-zero passes through (no special-casing,
  matching torch). REQ-9 NOT-STARTED — broadcasting gap, missing
  `rounding_mode` kwarg, no non-test caller. Blocker #1213.

### Layer 3: `clamp_` (`inplace.rs:375-407`)

- `clamp_(min, max)` (`:385`) — validates `min <= max` up front
  (`InvalidArgument` error if violated), then iterates `if *x < min { *x =
  min } else if *x > max { *x = max }`. No GPU fast path. REQ-10
  NOT-STARTED — three gaps: (a) torch's `clamp_(min=None, max=None)`
  accepts Optional bounds (current signature requires both); (b) NaN-bound
  fills-everything special case (`TensorCompare.cpp:844`); (c) no non-test
  caller (natural caller is gradient-clipping in `optim/utils.rs`).
  Blocker #1214.

### Storage-mutation safety (see module `//!` doc-comment in `inplace.rs`)

All in-place ops mutate storage through `unsafe { self.update_data(...) }`
or `unsafe { self.update_storage(...) }`. Every `unsafe` block carries a
`// SAFETY:` comment tying soundness back to the `fn check_inplace_allowed in inplace.rs`
proof: the tensor is not part of the autograd graph (so no cached value is
invalidated) and is not a `requires_grad` leaf (so no gradient tracking is
silently corrupted). Single-threaded `&self` access satisfies the
`update_data` / `update_storage` exclusive-access contract on
`Arc<TensorStorage>`. See file-level `//!` doc at lines 1-25.

## Parity contract

The route's `parity_ops` list is `[]` (empty) — `inplace.rs` itself owns no
per-op parity-sweep entries; instead, in-place behavior surfaces through the
out-of-place op arms with `inplace=True` kwarg. The two relevant
sweep-arms today:

| route arm | upstream entry | in-place use of inplace.rs | edge-case contract |
|---|---|---|---|
| `add` (in `tools/parity-sweep/runner/src/main.rs:494-498`) | `aten/src/ATen/native/BinaryOps.cpp:434-439` (`TORCH_IMPL_FUNC(sub_out)` shares `add_stub`) + `torch/_tensor_docs.py:379` `add_(other, *, alpha=1)` | `Tensor::add_scaled_(&other, alpha)` invoked when op_db sample carries `inplace=True`; the parity-sweep `add` audit `verified` at 88/88 passed includes these in-place samples | NaN propagates; ±Inf preserved; denormals preserved (no FTZ); empty shapes preserved; scalar `[]` broadcasts; 0-stride expand broadcasts; alpha edges (0, -0.0, NaN, ±huge); in-place cannot resize `self` — broadcast result must equal `self.shape()` or `ShapeMismatch` returns |
| `sub` (`tools/parity-sweep/runner/src/main.rs`'s `sub` arm) | `aten/src/ATen/native/BinaryOps.cpp:434-439` (`TORCH_IMPL_FUNC(sub_out)`) + `torch/_tensor_docs.py:5113` `sub_(other, *, alpha=1)` | `Tensor::sub_scaled_(&other, alpha)` (the `alpha` kwarg path) and `Tensor::sub_(&other)` (the alpha=1 plain path) | verified — `arithmetic::sub_scaled` delegates to `add_scaled(a, b, -alpha)` and `inplace::sub_scaled_` delegates to `add_scaled_(other, -alpha)`; 88/88 passed at seeds=8 after #1192 closed |

The parity-sweep `mul`, `div`, `neg`, `abs`, `sqrt`, `pow`, `clamp` arms
have no `inplace=True` exercise today; if added, they would exercise
`mul_` / `div_` / `clamp_` which carry the broadcasting / NaN-bound /
None-bound gaps above. The audit JSON at
`tools/parity-sweep/parity_audit.json` only carries an `add` entry today.

## Verification

### Unit tests (in-file `#[cfg(test)] mod tests` at `inplace.rs:410-697`)

20 tests covering forward correctness and autograd-guard rejection across
the ten public ops:

- `add_scalar_`: `fn test_add_scalar_basic in inplace.rs`,
  `fn test_add_scalar_negative in inplace.rs`,
  `fn test_add_scalar_chaining in inplace.rs`,
  `fn test_add_scalar_rejects_requires_grad_leaf in inplace.rs`.
- `mul_scalar_`: `fn test_mul_scalar_basic in inplace.rs`,
  `fn test_mul_scalar_zero in inplace.rs`,
  `fn test_mul_scalar_rejects_requires_grad_leaf in inplace.rs`.
- `fill_`: `fn test_fill_basic in inplace.rs`,
  `fn test_fill_scalar_tensor in inplace.rs`,
  `fn test_fill_rejects_requires_grad_leaf in inplace.rs`.
- `zero_`: `fn test_zero_basic in inplace.rs`,
  `fn test_zero_empty_tensor in inplace.rs`,
  `fn test_zero_rejects_requires_grad_leaf in inplace.rs`.
- `clamp_`: `fn test_clamp_basic in inplace.rs`,
  `fn test_clamp_all_within_range in inplace.rs`,
  `fn test_clamp_single_value_range in inplace.rs`,
  `fn test_clamp_invalid_range in inplace.rs`,
  `fn test_clamp_rejects_requires_grad_leaf in inplace.rs`.
- Cross-op integration: `fn test_detached_tensor_allows_inplace in inplace.rs`
  — detached tensors are mutable; `fn test_mixed_inplace_chaining in inplace.rs`
  — add_scalar_ → mul_scalar_ → clamp_ chain; `fn test_inplace_ops_f64 in inplace.rs`
  — f64 dtype coverage.

Note: there are NO unit tests in this file for `add_` / `add_scaled_` /
`sub_` / `mul_` / `div_` — the elementwise tensor-bound in-place ops.
Their tensor-elementwise correctness is exercised only through the
conformance sweep below; this is itself a verification gap (no in-file
guard against regression of the `add_scaled_` storage-swap path, the
`shape mismatch` error path on `sub_`/`mul_`/`div_`, or the broadcast
in-place-cannot-resize error path on `add_scaled_`).

### Conformance tests
(`ferrotorch-core/tests/conformance_elementwise.rs`)

The elementwise conformance harness invokes `t.add_(&other)` /
`t.sub_(&other)` / `t.mul_(&other)` / `t.div_(&other)` at lines 1880-1921
and `t.add_scalar_(...)` / `t.mul_scalar_(...)` / `t.fill_(...)` /
`t.zero_()` / `t.clamp_(lo, hi)` at lines 1962-2034. These tests run as
`cargo test -p ferrotorch-core --test conformance_elementwise`. They
compare ferrotorch in-place results to the out-of-place equivalent's
result element-by-element, which is the closest analog to the "in-place
matches out-of-place" contract goal.md R-CHAR-3 requires.

### Parity-sweep commands (verbatim — orchestrator re-runs)

```bash
# inplace.rs's behavior is exercised through the add/sub parity-sweep arms
# (with inplace=True op_db samples). The route's parity_ops field is [], so
# no per-file smoke is mandated — but the umbrella add/sub smokes are:
./target/release/parity-sweep sweep --op add --seeds 8   # → 88/88 passed (0 skipped, 0 failed)
./target/release/parity-sweep sweep --op sub --seeds 8   # → 88/88 passed (0 skipped, 0 failed) — #1192 closed via sub_scaled / sub_scaled_
```

The integer grep-count for `passed (0 skipped, 0 failed)` is **>= 1** for
both `add` and `sub` (the in-place add / sub paths are covered through the
`add_scaled_` and `sub_scaled_` consumers respectively).

### Per-crate test command

```bash
cargo test -p ferrotorch-core --lib inplace   # → 20 passed, 0 failed
```

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (add_scalar_) | NOT-STARTED | open prereq blocker #1205. impl exists at `ferrotorch-core/src/inplace.rs:69` mirroring `aten/src/ATen/native/BinaryOps.cpp:1180 Tensor& add_(Tensor& self, const Scalar& other, const Scalar& alpha)` and docstring `torch/_tensor_docs.py:379 add_(other, *, alpha=1) -> Tensor`; in-file tests pass; BUT no non-test production consumer exists in `ferrotorch-{core,nn,optim,...}/src/**/*.rs` — the only out-of-`#[cfg(test)]` caller is the parity-sweep dispatch table at `tools/parity-sweep/runner/src/main.rs` (test-side, disqualified by goal.md R-DEFER-1). |
| REQ-2 (mul_scalar_) | NOT-STARTED | open prereq blocker #1206. impl at `ferrotorch-core/src/inplace.rs:89` mirroring `aten/src/ATen/native/BinaryOps.cpp:996 Tensor& mul_(Tensor& self, const Scalar& other)` and docstring `torch/_tensor_docs.py:3441 mul_(value) -> Tensor`; in-file tests pass; no non-test production consumer. |
| REQ-3 (fill_) | NOT-STARTED | open prereq blocker #1207. impl at `ferrotorch-core/src/inplace.rs:109` mirroring `aten/src/ATen/native/Fill.cpp:47 Tensor& fill_(Tensor& self, const Scalar& value)` and docstring `torch/_tensor_docs.py:1955 fill_(value) -> Tensor`; in-file tests pass; no non-test production consumer (natural caller is `ferrotorch-nn::init::constant_` which currently builds storage directly). |
| REQ-4 (zero_) | NOT-STARTED | open prereq blocker #1208. impl at `ferrotorch-core/src/inplace.rs:128` delegates to `self.fill_(T::zero())` mirroring `aten/src/ATen/native/Fill.cpp:150-158 Tensor& zero_(Tensor &self)` (which itself falls back to `self.fill_(0)`) and docstring `torch/_tensor_docs.py:6381 zero_() -> Tensor`; in-file tests pass; no non-test production consumer (natural caller is `optim::zero_grad`). |
| REQ-5 (add_) | NOT-STARTED | open prereq blocker #1209. impl at `ferrotorch-core/src/inplace.rs:147` (`self.add_scaled_(other, 1.0)`) mirroring `aten/src/ATen/native/BinaryOps.cpp:434-439 TORCH_IMPL_FUNC(sub_out) → add_stub(..., 1.0)` and docstring `torch/_tensor_docs.py:379 add_(other, *, alpha=1) -> Tensor`; conformance test at `ferrotorch-core/tests/conformance_elementwise.rs:1880` exercises it; no non-test production consumer (natural caller is `optim::sgd` param-update path). |
| REQ-6 (add_scaled_) | NOT-STARTED | open prereq blocker #1210. impl at `ferrotorch-core/src/inplace.rs:167` mirroring `aten/src/ATen/native/BinaryOps.cpp:437 add_stub(device_type(), *this, -alpha)` (with alpha-positive direction) and docstring `torch/_tensor_docs.py:379 add_(other, *, alpha=1) -> Tensor`; conformance sweep at `ferrotorch-core/tests/conformance_elementwise.rs` and parity-sweep `add` arm cover its behavior end-to-end (88/88 passed at seeds=8); the parity-sweep dispatch table at `tools/parity-sweep/runner/src/main.rs:495 a.add_scaled_(&b, alpha)` IS a non-test invocation but the parity-sweep runner is explicitly disqualified as a test-side consumer per the role spec; no other non-test consumer exists. |
| REQ-7 (sub_) | NOT-STARTED | open prereq blocker #1211. impl: `pub fn sub_` in `ferrotorch-core/src/inplace.rs` (post-commit 2f792bfc5) is a one-line delegation `self.sub_scaled_(other, 1.0)` mirroring `aten/src/ATen/native/BinaryOps.cpp:1157 Tensor& sub_(Tensor& self, const Tensor& other, const Scalar& alpha)` and docstring `torch/_tensor_docs.py:5113 sub_(other, *, alpha=1) -> Tensor`. Broadcasting and GPU dispatch are inherited from `add_scaled_` via `sub_scaled_`. The `alpha` kwarg gap is now closed by `sub_scaled_` (REQ-11 SHIPPED). REQ-7 remains NOT-STARTED solely because there is no non-test production consumer — parity-sweep `sub` arm now passes 88/88 after #1192 closed (the prior 16/88 failures on `alpha != 1` are resolved via `sub_scaled_`). |
| REQ-8 (mul_) | NOT-STARTED | open prereq blocker #1212. impl at `ferrotorch-core/src/inplace.rs:292` mirrors `aten/src/ATen/native/BinaryOps.cpp:441 TORCH_IMPL_FUNC(mul_out)` only on the same-shape path (shape-strict at lines 294-302) — torch's `mul_` inherits broadcasting from `mul_out` per docstring `torch/_tensor_docs.py:3441 mul_(value)`. No non-test production consumer. |
| REQ-9 (div_) | NOT-STARTED | open prereq blocker #1213. impl: `pub fn div_ in inplace.rs` mirrors `aten/src/ATen/native/BinaryOps.cpp:447 TORCH_IMPL_FUNC(div_out)` only on the same-shape path (shape-strict early return); missing `rounding_mode` kwarg per docstring `torch/_tensor_docs.py:1746 div_(value, *, rounding_mode=None) -> Tensor`. No non-test production consumer. |
| REQ-10 (clamp_) | NOT-STARTED | open prereq blocker #1214. impl at `ferrotorch-core/src/inplace.rs:385` mirrors only the both-bounds-scalar path of `aten/src/ATen/native/TensorCompare.cpp:831-854 TORCH_IMPL_FUNC(clamp_out)` and docstring `torch/_tensor_docs.py:1141 clamp_(min=None, max=None) -> Tensor` — missing Optional/None bound handling, missing NaN-fills-everything special case. No non-test production consumer (natural caller is gradient-clipping in `optim::utils`). |
| REQ-11 (sub_scaled_) | SHIPPED | impl: `Tensor::sub_scaled_` at `ferrotorch-core/src/inplace.rs:265` delegates to `self.add_scaled_(other, -alpha)` mirroring `aten/src/ATen/native/BinaryOps.cpp:434-439 TORCH_IMPL_FUNC(sub_out) { add_stub(device_type(), *this, -alpha); }` and docstring `torch/_tensor_docs.py:5113 sub_(other, *, alpha=1) -> Tensor`. Non-test production consumer: `ferrotorch-core/src/grad_fns/arithmetic.rs:923-936 pub fn sub_scaled` IS the out-of-place sibling and itself delegates to `add_scaled(a, b, -alpha)` — the symmetric pair establishes torch's `sub` alpha-kwarg path across both surfaces. Parity-sweep `[sub] 88/88 passed (0 skipped, 0 failed)` at seeds=8 covers the in-place samples too (closes #1192). NB: this is distinct from REQ-7 (`sub_`, no alpha kwarg) which remains NOT-STARTED for unrelated consumer-wiring (blocker #1211). |
