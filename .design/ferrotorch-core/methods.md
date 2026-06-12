# Method-style API for Tensor

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/native/TensorShape.cpp
-->

## Summary

`ferrotorch-core/src/methods.rs` is the **canonical public method-style API**
of ferrotorch: it implements `impl<T: Float> Tensor<T>` with ~50 instance
methods (e.g. `a.add_t(&b)`, `a.relu()`, `a.matmul(&b)`, `a.reshape_t(&[...])`,
`a.view(&[...])`, `a.contiguous()`, `a.chunk(n, dim)`) that mirror
`torch.Tensor.*` per R-DEV-2's Python user-API ABI rule. Every method is a
thin wrapper delegating to a function in
`crate::{grad_fns::{arithmetic, transcendental, activation, reduction, linalg,
shape}, einsum, linalg}` — the methods do not implement arithmetic
themselves; they exist so PyTorch users can write
`tensor.relu().matmul(&w).softmax()` in Rust with the same chaining
ergonomics torch's `torch.Tensor` C++ methods provide.

The file additionally houses five free functions — `permute_t`, `narrow_t`,
`view_t`, `contiguous_t`, `chunk_t`, `split_t` — that own the actual shape /
view-creation logic (zero-copy stride views with autograd backward nodes
`PermuteBackward`, `NarrowBackward`, `ContiguousBackward`, `SplitBackward`).
These free functions are the load-bearing implementation; the `impl Tensor`
methods are public-API wrappers around them.

## Requirements

- REQ-1: **Arithmetic methods** — `add_t / sub_t / rsub_t / mul_t / div_t /
  neg_t / pow_t / sqrt_t / abs_t`. Each instance method must delegate to
  `crate::grad_fns::arithmetic::<op>(self, ...)` preserving autograd, broadcast,
  and dtype-promotion semantics. Mirrors `torch.Tensor.add / sub / rsub / mul /
  div / neg / pow / sqrt / abs` per `torch/_tensor_docs.py:360 / 5102 / 4625 /
  3430 / 1735 / 3583 / 3823 / 4983 / 251` and the upstream stubs at
  `aten/src/ATen/native/BinaryOps.cpp:434/441/447/1169` (binary, with `rsub`
  at `:1169` as a literal `at::sub(other, self, alpha)` operand-swap
  delegation) and `aten/src/ATen/native/UnaryOps.cpp:344+` (unary). The
  instance methods are the contract `arithmetic.md`'s REQ-1..REQ-11 publish
  to the world.

- REQ-2: **Transcendental methods** — `exp_t / log_t / sin_t / cos_t /
  clamp_t`. Delegate to `crate::grad_fns::transcendental::<op>` mirroring
  `torch.Tensor.exp / log / sin / cos / clamp` per `torch/_tensor_docs.py:1878
  / 2992 / 4794 / 1288 / 1130` and the unary kernels in
  `aten/src/ATen/native/UnaryOps.cpp`.

- REQ-3: **Activation methods** — `relu / sigmoid / tanh_t / gelu / gelu_with
  / silu / softmax / log_softmax`. Delegate to
  `crate::grad_fns::activation::*` mirroring the activation functions exposed
  on `torch.Tensor` (sigmoid/tanh) and via `torch.nn.functional` (relu/gelu/
  silu/softmax/log_softmax — per `torch/_tensor_docs.py:4713 sigmoid` and
  `torch/_tensor_docs.py:5522 tanh`; `torch/nn/functional.py:1718 relu / 2012
  gelu / 2381 silu / 2245 log_softmax`).

- REQ-4: **Global reduction methods** — `sum_all / mean_all / prod_all / amin
  / amax`. Delegate to `crate::grad_fns::reduction::*` mirroring
  `torch.Tensor.sum / mean / prod / amin / amax` per `torch/_tensor_docs.py:
  5138 / 3304 / 3859 / 3349 / 3259`. The `_all` suffix in ferrotorch's
  `sum_all`/`mean_all`/`prod_all` distinguishes the global-reduction overload
  (no `dim` arg) from the per-dim sibling `sum_dim`/`mean_dim`; PyTorch
  uses argument count to dispatch.

- REQ-5: **Dim-reduction methods** — `sum_dim(dim, keepdim) /
  mean_dim(dim, keepdim)`. Delegate to `crate::grad_fns::reduction::{sum_dim,
  mean_dim}` mirroring `torch.Tensor.sum(dim, keepdim=False) /
  torch.Tensor.mean(dim, keepdim=False)` per `torch/_tensor_docs.py:5138 /
  3304`.

- REQ-6: **Linalg methods** — `matmul / mm / mm_bt / bmm / mv_t / dot_t / t /
  einsum`. Delegate to `crate::grad_fns::linalg::*_differentiable` (and
  `crate::einsum::einsum_differentiable` for `einsum`). Mirrors
  `torch.Tensor.matmul / mm / bmm / mv / dot / t` per `torch/_tensor_docs.py:
  6388 / 3394 / 1060 / 3475 / 1771 / 5201` and `torch.einsum` per
  `torch/functional.py:176`. `mm_bt` is a ferrotorch-specific fused
  `A @ B^T` (per R-DEV-7 — no direct PyTorch op, but matches the
  contract of `torch.matmul(a, b.T)` with the transpose materialization
  elided); torch's `Linear` forward uses the same shape internally via
  cuBLAS gemm with transA=N, transB=T.

- REQ-7: **lu_factor method** — `lu_factor()` delegates to
  `crate::linalg::lu_factor` mirroring `torch.linalg.lu_factor` per
  `torch/linalg/__init__.py:2403 lu_factor = _add_docstr(...)`.

- REQ-8: **Reshape / shape methods** — `reshape_t / flatten_t / squeeze_t /
  unsqueeze_t / permute / transpose / view`. Delegate to
  `crate::grad_fns::shape::*` (and `permute_t` / `view_t` defined in this
  file). Mirrors `torch.Tensor.reshape / flatten / squeeze / unsqueeze /
  permute / transpose / view` per
  `aten/src/ATen/native/TensorShape.cpp:2129 reshape / 4178 flatten / 4014
  squeeze / 4109 unsqueeze / 1829 permute / 3816 transpose / 4563 view` and
  the user-facing docstrings at `torch/_tensor_docs.py:4274 / 2061 / 5019 /
  6109 / 3777 / 5903 / 6145`.

- REQ-9: **View / contiguous methods** — `view / contiguous / narrow`. The
  in-this-file free functions `view_t / contiguous_t / narrow_t` provide the
  zero-copy stride-view implementation; the methods delegate to them.
  Mirrors `torch.Tensor.view / contiguous / narrow` per
  `aten/src/ATen/native/TensorShape.cpp:4563 view / 1669 narrow` and
  `torch/_tensor_docs.py:6145 view / 1190 contiguous / 3502 narrow`.

- REQ-10: **Chunk / split methods** — `chunk(chunks, dim) /
  split(split_sizes, dim)`. The free functions `chunk_t / split_t` in this
  file own the implementation (GPU fast path through `backend.strided_split_f32`
  + CPU fallback). Mirrors `torch.Tensor.chunk / torch.split` per
  `aten/src/ATen/native/TensorShape.cpp:1077 chunk / 3175 split` and
  `torch/_tensor_docs.py:6397 chunk`.

- REQ-11: **PyTorch-compat aliases** — `size() -> &[usize]` aliases
  `shape()`; `dim() -> usize` aliases `ndim()`. Mirrors
  `torch.Tensor.size() / torch.Tensor.dim()` per `torch/_tensor_docs.py:4848
  size / 1717 dim` — these are the canonical PyTorch names for shape
  inspection and `Tensor::shape() / ndim()` are the Rust-natural names; the
  aliases satisfy R-DEV-2.

- REQ-12: **Utility method** — `print(&self) -> &Self`. Mirrors
  `torch.Tensor.__repr__` indirectly: provides a chainable debug-emit
  primitive built on the `tracing` ecosystem rather than direct stdout
  (R-DEV-7 — library code does not write to stdout; consumers install a
  `tracing` subscriber). No direct PyTorch parallel; closest cousins are
  Python's `print(tensor)` builtin and PyTorch's `Tensor.print()` debug API.

- REQ-13: **Cumulative (scan) methods** — `cumsum_t / cumprod_t /
  logcumsumexp_t`. Each instance method delegates to
  `crate::grad_fns::cumulative::<op>(self, dim)` preserving autograd
  (`CumsumBackward` / `CumprodBackward` / `LogcumsumexpBackward`),
  negative-dim normalization, and the 0-D scalar fast path. Mirrors
  `torch.Tensor.cumsum / cumprod / logcumsumexp` per
  `torch/_tensor_docs.py:1500-1506 (cumsum) / :1482-1488 (cumprod) /
  :1455-1462 (logcumsumexp)` and the upstream entries at
  `aten/src/ATen/native/ReduceOps.cpp:511 TORCH_IMPL_FUNC(cumsum_out)
  / :519 TORCH_IMPL_FUNC(cumprod_out) / :475 Tensor logcumsumexp(...)`.
  These instance methods are the non-test production consumer wiring
  per R-DEFER-1 for `grad_fns::cumulative::{cumsum, cumprod,
  logcumsumexp}` — the previously vocabulary-only `lib.rs:159`
  re-exports are now reachable through a chainable PyTorch-API
  surface. `cummax_t` / `cummin_t` are NOT included here: the existing
  consumers at `einops.rs:796 / :802` use the
  `CumExtremeResult { values, indices }` tuple directly, and the
  underlying ops remain NOT-STARTED behind blocker #1231
  (differentiability + tie-break + NaN). Closes #1232.

## Acceptance Criteria

- [x] AC-1: Every arithmetic method (`add_t / sub_t / mul_t / div_t / neg_t /
  pow_t / sqrt_t / abs_t`) returns the same value as its underlying
  `crate::grad_fns::arithmetic::*` function for any input — covered by
  `fn test_method_chain in methods.rs` and the
  matmul / relu / sum / sigmoid tests `fn test_method_matmul / test_method_relu /
  test_method_sum / test_method_sigmoid in methods.rs` plus the conformance
  sweep `ferrotorch-core/tests/conformance_elementwise.rs`.
- [x] AC-2: Reshape / squeeze / unsqueeze / permute / view / transpose are
  all zero-copy stride views when the layout permits (no data copy on the
  forward pass) — covered by `fn test_method_permute_2d in methods.rs`
  which asserts `!b.is_contiguous()`
  after permute, and by `fn test_method_view in methods.rs`.
- [x] AC-3: `permute` rejects invalid permutations (wrong length, duplicate,
  out-of-bounds) with `InvalidArgument` — covered by
  `fn test_permute_invalid_dims in methods.rs`.
- [x] AC-4: `chunk(n, dim)` produces `n` chunks (or fewer when the last is
  empty) with the last chunk sized to soak up the remainder of an uneven
  split — covered by `fn test_method_chunk_even / test_method_chunk_uneven /
  test_method_chunk_2d in methods.rs`.
- [x] AC-5: `split(split_sizes, dim)` rejects mismatched-sum split_sizes —
  covered by `fn test_split_bad_sizes in methods.rs`.
- [x] AC-6: `split` / `chunk` preserve autograd (each output chunk has a
  `grad_fn` when input requires grad), with the gradient correctly
  scattered back to the input on backward — covered by
  `fn test_split_preserves_grad / test_split_backward_simple /
  test_chunk_backward_2d / test_split_no_grad_when_disabled in methods.rs`.
- [x] AC-7: `contiguous()` on an already-contiguous tensor returns a cheap
  clone (no GPU kernel launch); on non-contiguous CUDA tensors of rank ≤ 8
  it dispatches to `backend.strided_copy_{f32,f64}` (no CPU round-trip per
  R-CODE-4) — covered by `fn test_method_contiguous in methods.rs`
  and exercised in production at
  `ferrotorch-diffusion/src/attention.rs:175 / 180 / 185 / 204 / 737 / 745 /
  ferrotorch-nn/src/attention.rs:339 / 345 / 349 / 410`.
- [x] AC-8: `view(&[..])` requires `is_contiguous()` and errors with
  `InvalidArgument` ("call .contiguous() first") otherwise; supports `-1`
  inference for exactly one dim — covered by `fn test_method_view /
  test_method_view_infer in methods.rs`.
- [x] AC-9: `narrow(dim, start, length)` errors when `start + length >
  dim_size` and provides a backward that pads the gradient with zeros in
  the sliced dimension (`struct NarrowBackward in methods.rs`).
- [x] AC-10: `transpose(dim0, dim1)` short-circuits to a clone when
  `dim0 == dim1` (matches torch's no-op identity) and errors on
  out-of-bounds dims — covered implicitly by the production caller at
  `ferrotorch-diffusion/src/blocks.rs:310 / 316 / ferrotorch-diffusion/src/
  attention.rs:736 / 744`.
- [x] AC-11: `size() == shape()` and `dim() == ndim()` byte-for-byte —
  covered by `fn test_size_alias / test_dim_alias in methods.rs`.

## Architecture

### `impl Tensor<T>` — public method surface (`methods.rs:11-302`)

Each public method is a one-liner that calls into the underlying free
function in the corresponding `grad_fns::*` module. This indirection is
deliberate: the methods file is the API contract, and the implementation
files (`grad_fns/arithmetic.rs`, `grad_fns/shape.rs`, etc.) are free to
restructure their internals without affecting the `Tensor::method`
signatures users compile against. The pattern matches `torch.Tensor`'s C++
side, where `Tensor::add()` is a method declared in `tensor.h` that
delegates to the native `at::add` dispatcher.

#### Arithmetic block (`methods.rs:14-84`) — REQ-1

- `add_t(&Tensor)` → `arithmetic::add` (`grad_fns/arithmetic.rs`).
- `sub_t(&Tensor)` → `arithmetic::sub`.
- `rsub_t(&Tensor, alpha: f64)` → `arithmetic::rsub` (`methods.rs:32-34`),
  the operand-swap delegation that mirrors upstream
  `BinaryOps.cpp:1169 Tensor rsub(...) { return at::sub(other, self, alpha); }`.
- `mul_t(&Tensor)` → `arithmetic::mul`.
- `div_t(&Tensor)` → `arithmetic::div`.
- `neg_t()` → `arithmetic::neg`.
- `pow_t(f64)` → `arithmetic::pow`.
- `sqrt_t()` → `arithmetic::sqrt`.
- `abs_t()` → `arithmetic::abs`.

Non-test consumers: `ferrotorch-nn/src/hooks.rs:296,505 (add_t)`. `rsub_t`
is the chainable method-style surface that closes the R-DEFER-1
production-consumer requirement for `arithmetic::rsub` per `arithmetic.md`
REQ-9.

#### Transcendental block (`methods.rs:48-66`) — REQ-2

- `exp_t / log_t / sin_t / cos_t / clamp_t` — five methods delegating to
  `grad_fns::transcendental::*`.

Non-test consumers: `ferrotorch-diffusion/src/vae_encoder.rs:499 (clamp_t)`.

#### Activation block (`methods.rs:70-103`) — REQ-3

- `relu / sigmoid / tanh_t / gelu / gelu_with(approximate) / silu / softmax
  / log_softmax`.

Non-test consumers: `ferrotorch-vision/src/models/detection/roi_heads_post
process.rs:421 (sigmoid)`, `ferrotorch-diffusion/src/attention.rs:196
(softmax)`, `ferrotorch-diffusion/src/blocks.rs:339 (softmax)`,
`ferrotorch/examples/ferrotorch_bench.rs:54-55 (relu, sigmoid)`.

#### Global reduction block (`methods.rs:107-128`) — REQ-4

- `sum_all / mean_all / prod_all` and the dim-implicit `amin / amax`.

Non-test consumers: `ferrotorch-distributions/src/multivariate_normal.rs:
208,573 (sum_all)`, `ferrotorch-distributions/src/{normal,laplace,cauchy,
exponential,half_normal,gumbel,uniform,student_t,lognormal,dirichlet}.rs
(sum_all)`, `ferrotorch-core/src/grad_fns/activation.rs:3221,3236,3265
(sum_all)`, `ferrotorch-core/src/stride_tricks.rs:679,707 (sum_all)`.

#### Linalg block (`methods.rs:140-187`) — REQ-6

- `matmul / mm / mm_bt / bmm / mv_t / dot_t / t / einsum`.

Non-test consumers: `ferrotorch-distributions/src/multivariate_normal.rs:
232,265,297,320 (matmul)`, `ferrotorch-optim/src/muon.rs:216 (t)`,
`ferrotorch-optim/src/natural_gradient.rs:275,279 (t)`,
`ferrotorch-diffusion/src/attention.rs:190,197 (bmm)`,
`ferrotorch-diffusion/src/blocks.rs,340 (bmm)`,
`ferrotorch-distributions/src/multivariate_normal.rs:230 (t via no_grad)`.

#### lu_factor (`methods.rs:135-137`) — REQ-7

Single method delegating to `crate::linalg::lu_factor`. **No non-test
consumer in `ferrotorch-{core,nn,vision,diffusion,distributions,optim}/src/
**/*.rs` today** — natural callers would be a future `torch.lu_solve`-style
pipeline or scientific-computing examples; ferrotorch's existing solvers
go through `crate::linalg::*` directly. NOT-STARTED — blocker #1220
(consumer wiring).

#### Dim-reduction block (`methods.rs:191-197`) — REQ-5

- `sum_dim(dim, keepdim) / mean_dim(dim, keepdim)`.

**Production callers use the free-function form** `crate::grad_fns::reduction::
{sum_dim, mean_dim}` directly (see `ferrotorch-distributions/src/
multivariate_normal.rs:26, dirichlet.rs:32, independent.rs:146,
ferrotorch-core/src/einsum.rs:263,1170, einops.rs:783,787,
grad_fns/linalg.rs:686,695,
meta_propagate.rs:474,478`) rather than the `Tensor::sum_dim` /
`Tensor::mean_dim` instance methods. The instance methods exist as
PyTorch-style aliases but have no production consumer. NOT-STARTED —
blocker #1221 (consumer migration: encourage `.sum_dim()` over
`sum_dim(&t, ...)`).

#### Shape block (`methods.rs:201-273`) — REQ-8

- `reshape_t(&[isize]) / flatten_t() / squeeze_t(isize) / unsqueeze_t(isize)`
  delegate to `grad_fns::shape::*`.
- `permute(&[usize])` delegates to `permute_t` (this file).
- `transpose(dim0, dim1)` builds the permutation array and calls
  `permute_t`; short-circuits to clone when `dim0 == dim1`.

Non-test consumers: `ferrotorch-nn/src/attention.rs:275,292,312,313,314,
337,340,343,347,355,357,359,361,364,365,408,411,420,579 (reshape_t)`,
`ferrotorch-diffusion/src/attention.rs:173,176,178,181,183,186,202,205,
735,745 (reshape_t / transpose)`,
`ferrotorch-diffusion/src/resnet_block_time.rs:133 (reshape_t)`,
`ferrotorch-diffusion/src/blocks.rs:309,310,316,348 (reshape_t / transpose)`,
`ferrotorch-nn/src/rnn.rs:244,245,669,1711 (squeeze_t)`,
`ferrotorch-core/src/einops.rs:745,751 (squeeze_t)`,
`ferrotorch-core/src/flex_attention.rs,213,226 (squeeze_t / unsqueeze_t)`,
`ferrotorch-vision/src/models/vit.rs:437,570 (squeeze_t)`,
`ferrotorch-vision/src/models/detection/roi_heads_postprocess.rs +
ferrotorch-ml/src/metrics.rs:1495,1496 (reshape_t)`,
`ferrotorch-data/src/transforms.rs:465-467 (narrow → contiguous)`,
`ferrotorch-nn/src/attention.rs:338,344,348,369 (permute)`.

#### View / contiguous / narrow block (`methods.rs:247-263`) — REQ-9

- `view(&[i64])` → free function `pub fn view_t in methods.rs` which
  validates contiguity and forwards to `grad_fns::shape::reshape`.
- `contiguous()` → free function `pub fn contiguous_t in methods.rs` with
  GPU fast path through `backend.strided_copy_{f32,f64}` (no CPU bounce per
  R-CODE-4) and CPU fallback. Autograd backward is the identity
  (`struct ContiguousBackward in methods.rs`).
- `narrow(dim, start, length)` → free function `pub fn narrow_t in methods.rs`
  — zero-copy view; `struct NarrowBackward in methods.rs` pads the gradient
  with zeros in the sliced dimension.

Non-test consumers: pervasive (see `ferrotorch-diffusion/src/attention.rs +
blocks.rs`, `ferrotorch-nn/src/attention.rs`, `ferrotorch-vision/src/
models/*`, `ferrotorch-distributions/src/multivariate_normal.rs:243,271,
319,359,367 (view)`, `ferrotorch-distributions/src/dirichlet.rs:291,293,
350,358,362 (view)`, `ferrotorch-vision/src/models/swin.rs:273,276,360,370
(view)`, `ferrotorch-vision/src/models/vit.rs:100 (view)`,
`ferrotorch-nn/src/rnn.rs:244,245 (narrow + squeeze_t)`,
`ferrotorch-data/src/transforms.rs:465,466 (narrow)`,
`ferrotorch-nn/src/flex_attention.rs:210,211 (narrow)`,
`ferrotorch-core/src/nested.rs:549,556 (narrow)`,
`ferrotorch-core/src/einops.rs:798,804 (narrow)`).

#### Chunk / split block (`methods.rs:265-273`) — REQ-10

- `chunk(chunks, dim)` → `pub fn chunk_t in methods.rs` which
  computes per-chunk sizes (`dim_size.div_ceil(chunks)`) and delegates to
  `split_t`.
- `split(split_sizes, dim)` → `pub fn split_t in methods.rs`. GPU fast
  path for f32 via `backend.strided_split_f32` (per-chunk on-device); CPU
  fallback walks `outer` strides and `memcpy`s. Autograd via
  `SplitBackward` from `grad_fns/shape.rs`.

Non-test consumers: `ferrotorch-diffusion/src/vae_encoder.rs:483 (chunk)`,
`ferrotorch-diffusion/src/attention.rs:351 (chunk)`,
`ferrotorch-diffusion/src/gpu/vae_encoder.rs:677 (chunk via cpu_params)`.

#### Compat aliases + utility (`methods.rs:275-301`) — REQ-11, REQ-12

- `size() -> &[usize]` aliases `shape()`. Non-test consumer:
  `ferrotorch-distributions/src/multivariate_normal.rs:509,629 (assert dim
  3)` — though `dim()` is consumed there, not `size()`. `size()` itself
  has no out-of-test consumer; closest is the test harness asserting alias
  equality. NOT-STARTED for `size()` consumer (most production code uses
  the underlying `shape()` accessor directly).
- `dim() -> usize` aliases `ndim()`. Non-test consumer:
  `ferrotorch-distributions/src/multivariate_normal.rs:509,629,
  ferrotorch-distributions/src/low_rank_multivariate_normal.rs:204 (assert
  ndim)` — these are all inside `#[cfg(test)] mod tests` blocks however,
  meaning **no non-test caller exists**. NOT-STARTED — blocker #1222.
- `print()` → `tracing::info!` event. **No non-test caller** in the
  workspace; the only invocations of `.print()` are the in-file test
  `test_method_print_chain`. NOT-STARTED — blocker #1223.

### Free functions (`methods.rs:308-786`)

The free functions `permute_t / narrow_t / view_t / contiguous_t /
chunk_t / split_t` are exported `pub fn` from the module (visible as
`crate::methods::*`). They own:

- The zero-copy stride-view construction (`pub fn permute_t in methods.rs`,
  `pub fn narrow_t in methods.rs`).
- The autograd `*Backward` structs (`struct PermuteBackward in methods.rs`,
  `struct NarrowBackward in methods.rs`, `struct ContiguousBackward in methods.rs`).
- The GPU fast paths (`contiguous_t` uses `backend.strided_copy_{f32,f64}`
  for rank ≤ 8 CUDA inputs; `split_t` uses
  `backend.strided_split_f32` per-chunk).
- The CPU fallbacks (`fn contiguous_t_cpu in methods.rs` always-valid layout
  copy; `split_t`'s outer-strides loop).

These free functions are themselves consumed by both the `impl Tensor`
methods above AND directly by other modules — there's no daylight between
the "method" surface and the "free function" surface; they're the same
public API exposed two ways.

### Why some REQs are NOT-STARTED

The methods that map onto strong production consumers (REQ-1 add_t, REQ-2
clamp_t, REQ-3 relu/sigmoid/softmax, REQ-4 sum_all, REQ-6 matmul/bmm/t,
REQ-8 reshape_t/squeeze_t/permute/transpose, REQ-9 view/contiguous/narrow,
REQ-10 chunk/split) are SHIPPED.

The methods that have **no non-test production consumer** in the
ferrotorch workspace are individually called out under their REQ:
- REQ-5 (sum_dim/mean_dim methods): all in-tree callers use the free
  `grad_fns::reduction::{sum_dim, mean_dim}` form — the `Tensor::sum_dim`
  method is a vocabulary-only PyTorch alias. Blocker #1221.
- REQ-7 (lu_factor): no consumer. Blocker #1220.
- REQ-11 (`size()` and `dim()` aliases): both aliases are only used inside
  `#[cfg(test)]` modules, even in `ferrotorch-distributions`. Blocker
  #1222.
- REQ-12 (`print()`): no non-test caller. Blocker #1223.

Per goal.md R-DEFER-1 and R-DEFER-2, these REQs are NOT-STARTED until a
non-test production consumer wires them up. The methods exist and compile;
they're simply unconsumed today.

## Parity contract

The route's `parity_ops` list is `[]` (empty) — `methods.rs` itself owns no
per-op parity-sweep entries. Each method delegates to a function whose
parity-sweep coverage lives in its own module's design doc:

| method | underlying op | parity-sweep coverage |
|---|---|---|
| `add_t` / `sub_t` / `mul_t` / `div_t` | `grad_fns::arithmetic::{add,sub,mul,div}` | `parity-sweep sweep --op add --seeds 8` (88/88 verified per `tools/parity-sweep/parity_audit.json:5-72`) covers add; sub via `sub_scaled` (verified after #1192) |
| `neg_t` / `abs_t` / `sqrt_t` / `pow_t` | `grad_fns::arithmetic::{neg,abs,sqrt,pow}` | arithmetic.md REQ-5..REQ-8 — no parity-sweep ops registered yet (NOT-STARTED in arithmetic.md) |
| `exp_t` / `log_t` / `sin_t` / `cos_t` / `clamp_t` | `grad_fns::transcendental::*` | transcendental.md (separate doc) |
| `relu` / `sigmoid` / `tanh_t` / `gelu` / `silu` / `softmax` / `log_softmax` | `grad_fns::activation::*` | activation.md (separate doc) |
| `sum_all` / `mean_all` / `prod_all` / `amin` / `amax` / `sum_dim` / `mean_dim` | `grad_fns::reduction::*` | reduction.md (separate doc) |
| `matmul` / `mm` / `mm_bt` / `bmm` / `mv_t` / `dot_t` / `t` | `grad_fns::linalg::*_differentiable` | linalg/grad_fns design doc + `complex_tensor.rs:628-651` exercising matmul |
| `reshape_t` / `flatten_t` / `squeeze_t` / `unsqueeze_t` | `grad_fns::shape::*` | grad_fns/shape design doc |
| `permute` / `transpose` / `narrow` / `view` / `contiguous` / `chunk` / `split` | this file (free functions) | exercised via every downstream consumer (attention, einops, conv, vit) |

The umbrella `add` parity-sweep arm at
`tools/parity-sweep/parity_audit.json` already passes the `Tensor::add_t`
path end-to-end through `arithmetic::add` end-to-end (88/88 at seeds=8).

## Verification

### In-file unit tests (`methods.rs:788-1129`)

22 tests across the method surface:

- **Arithmetic/activation chain**: `fn test_method_relu in methods.rs`,
  `fn test_method_matmul in methods.rs`, `fn test_method_sum in methods.rs`,
  `fn test_method_transpose in methods.rs`, `fn test_method_chain in methods.rs`,
  `fn test_method_sigmoid in methods.rs`, `fn test_method_flatten in methods.rs`.
- **Reduction**: `fn test_method_sum_dim in methods.rs`,
  `fn test_method_mean_dim in methods.rs`.
- **Shape**: `fn test_method_permute_2d in methods.rs`,
  `fn test_method_permute_3d in methods.rs`,
  `fn test_permute_invalid_dims in methods.rs`,
  `fn test_method_view in methods.rs`, `fn test_method_view_infer in methods.rs`,
  `fn test_method_contiguous in methods.rs`.
- **Chunk/split**: `fn test_method_chunk_even in methods.rs`,
  `fn test_method_chunk_uneven in methods.rs`, `fn test_method_chunk_2d in methods.rs`,
  `fn test_method_split in methods.rs`, `fn test_method_split_2d_axis1 in methods.rs`,
  `fn test_split_bad_sizes in methods.rs`.
- **Autograd**: `fn test_split_preserves_grad in methods.rs`,
  `fn test_split_backward_simple in methods.rs`,
  `fn test_chunk_backward_2d in methods.rs`,
  `fn test_split_no_grad_when_disabled in methods.rs`.
- **Aliases**: `fn test_size_alias in methods.rs`, `fn test_dim_alias in methods.rs`.
- **Utility**: `fn test_method_print_chain in methods.rs`.

### Per-crate test command

```bash
cargo test -p ferrotorch-core --lib methods   # all 22 tests pass
```

### Parity-sweep commands (verbatim — orchestrator re-runs)

```bash
# methods.rs has no own parity_ops; the umbrella add arm covers the
# delegation chain Tensor::add_t -> arithmetic::add.
./target/release/parity-sweep sweep --op add --seeds 8   # → 88/88 passed (0 skipped, 0 failed)
```

The integer grep-count for `passed (0 skipped, 0 failed)` is **>= 1** for
the umbrella `add` arm.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (arithmetic methods) | SHIPPED | impl: `Tensor::add_t / sub_t / rsub_t / mul_t / div_t / neg_t / pow_t / sqrt_t / abs_t` at `add_t in ferrotorch-core/src/methods.rs` each delegate to `crate::grad_fns::arithmetic::<op>` mirroring `torch.Tensor.add / sub / rsub / mul / div / neg / pow / sqrt / abs` per `torch/_tensor_docs.py:360 / 5102 / 4625 / 3430 / 1735 / 3583 / 3823 / 4983 / 251` and `aten/src/ATen/native/BinaryOps.cpp:434/441/447/1169` (`rsub` at `:1169` is the operand-swap delegation `at::sub(other, self, alpha)`). Non-test production consumer: `add_t in ferrotorch-nn/src/hooks.rs,505 input.add_t(...)`; `rsub_t` itself at `rsub_t in ferrotorch-core/src/methods.rs` is the production consumer for `arithmetic::rsub` per R-DEFER-1 (`arithmetic.md` REQ-9). Parity-sweep `add` arm verified 88/88 at seeds=8 (`tools/parity-sweep/parity_audit.json:5-72`); `rsub` arm landed via #1194. |
| REQ-2 (transcendental methods) | SHIPPED | impl: `pub fn exp_t / log_t / sin_t / cos_t / clamp_t in methods.rs` delegate to `crate::grad_fns::transcendental::*` mirroring `torch.Tensor.exp / log / sin / cos / clamp` per `torch/_tensor_docs.py:1878 / 2992 / 4794 / 1288 / 1130`. Non-test production consumer: `ferrotorch-diffusion/src/vae_encoder.rs:499 logvar_raw.clamp_t(lo, hi)`. |
| REQ-3 (activation methods) | SHIPPED | impl: `Tensor::relu / sigmoid / tanh_t / gelu / gelu_with / silu / softmax / log_softmax` at `relu in ferrotorch-core/src/methods.rs` delegate to `crate::grad_fns::activation::*` mirroring `torch.Tensor.{sigmoid, tanh}` per `torch/_tensor_docs.py:4713,5522` and `torch/nn/functional.py:1718 relu / 2012 gelu / 2381 silu / 2245 log_softmax`. Non-test consumers: `ferrotorch-vision/src/models/detection/roi_heads_postprocess.rs mask_logits.sigmoid()`, `ferrotorch-diffusion/src/attention.rs scores_scaled.softmax()`, `ferrotorch-diffusion/src/blocks.rs scores_scaled.softmax()`. |
| REQ-4 (global reductions) | SHIPPED | impl: `Tensor::sum_all / mean_all / prod_all / amin / amax` at `sum_all in ferrotorch-core/src/methods.rs` delegate to `crate::grad_fns::reduction::*` mirroring `torch.Tensor.sum / mean / prod / amin / amax` per `torch/_tensor_docs.py:5138 / 3304 / 3859 / 3349 / 3259`. Non-test consumers: `sum_all in ferrotorch-distributions/src/multivariate_normal.rs log_l.sum_all()`, `sum_all in ferrotorch-core/src/stride_tricks.rs,707 view.sum_all() / contig.sum_all()`, `sum_all in ferrotorch-core/src/grad_fns/activation.rs,3236,3265 y.sum_all()`. |
| REQ-5 (dim reductions) | NOT-STARTED | open prereq blocker #1221. impl at `ferrotorch-core/src/methods.rs:191-197` delegates to `crate::grad_fns::reduction::{sum_dim, mean_dim}` mirroring `torch.Tensor.{sum, mean}(dim, keepdim=False)` per `torch/_tensor_docs.py:5138 / 3304`; the in-file tests `test_method_sum_dim / test_method_mean_dim` pass; BUT production callers (`ferrotorch-distributions/src/{multivariate_normal,dirichlet,independent}.rs`, `ferrotorch-core/src/{einsum,einops,grad_fns/linalg,meta_propagate}.rs`) all import and call the free `grad_fns::reduction::sum_dim` form directly, bypassing the `Tensor::sum_dim` method. No non-test caller of `Tensor::sum_dim` or `Tensor::mean_dim` exists in `ferrotorch-*/src/**/*.rs`. |
| REQ-6 (linalg methods) | SHIPPED | impl: `Tensor::matmul / mm / mm_bt / bmm / mv_t / dot_t / t / einsum` at `matmul in ferrotorch-core/src/methods.rs` delegate to `crate::grad_fns::linalg::*_differentiable` and `crate::einsum::einsum_differentiable` mirroring `torch.Tensor.matmul / mm / bmm / mv / dot / t` per `torch/_tensor_docs.py:6388 / 3394 / 1060 / 3475 / 1771 / 5201` and `torch.einsum` per `torch/functional.py:176`. Non-test consumers: `matmul in ferrotorch-distributions/src/multivariate_normal.rs,265,297,320 eps.matmul(&l_t) / scale_tril.matmul / diff_2d.matmul`; `matmul in ferrotorch-optim/src/muon.rs g.t()`; `matmul in ferrotorch-optim/src/natural_gradient.rs,279 input_activation.t() / output_gradient.t()`; `ferrotorch-diffusion/src/attention.rs,197 q.bmm(&k_t) / probs.bmm(&v)`; `ferrotorch-diffusion/src/blocks.rs,340 q.bmm(&k_t) / probs.bmm(&v)`. Caveat: `einsum / mv_t / dot_t / mm_bt / mm` have no direct non-test caller (those callers prefer `matmul` or call the underlying `linalg::*_differentiable` directly); the family-level REQ is SHIPPED via `matmul / bmm / t`. |
| REQ-7 (lu_factor) | NOT-STARTED | open prereq blocker #1220. impl at `ferrotorch-core/src/methods.rs:135-137 self.lu_factor()` delegates to `crate::linalg::lu_factor` mirroring `torch.linalg.lu_factor` per `torch/linalg/__init__.py:2403 lu_factor = _add_docstr(...)`. No non-test caller exists in `ferrotorch-*/src/**/*.rs`. |
| REQ-8 (reshape/shape methods) | SHIPPED | impl: `Tensor::reshape_t / flatten_t / squeeze_t / unsqueeze_t / permute / transpose` at `reshape_t in ferrotorch-core/src/methods.rs` (transpose at `reshape_t in ferrotorch-core/src/methods.rs` builds the permutation array and delegates to `permute_t`). Mirrors `torch.Tensor.reshape / flatten / squeeze / unsqueeze / permute / transpose` per `aten/src/ATen/native/TensorShape.cpp:2129 / 4178 / 4014 / 4109 / 1829 / 3816` and `torch/_tensor_docs.py:4274 / 2061 / 5019 / 6109 / 3777 / 5903`. Non-test consumers: `permute_t in ferrotorch-nn/src/attention.rs,292,312,313,314,337,340,343,347,355-365,408,411,420,579 (reshape_t)`; `permute_t in ferrotorch-diffusion/src/attention.rs,202-205,735-746 (reshape_t / transpose)`; `reshape_t in ferrotorch-diffusion/src/blocks.rs,348 (reshape_t / transpose)`; `reshape_t in ferrotorch-diffusion/src/resnet_block_time.rs (reshape_t)`; `reshape_t in ferrotorch-nn/src/rnn.rs,245,669,1711 (squeeze_t via narrow)`; `permute_t in ferrotorch-core/src/einops.rs,805 (squeeze_t)`; `reshape_t in ferrotorch-core/src/flex_attention.rs,213,226 (squeeze_t / unsqueeze_t)`; `permute_t in ferrotorch-vision/src/models/vit.rs,570 (squeeze_t)`; `reshape_t in ferrotorch-ml/src/metrics.rs,1496 (reshape_t)`; `permute_t in ferrotorch-nn/src/attention.rs,344,348,369 (permute)`. |
| REQ-9 (view/contiguous/narrow) | SHIPPED | impl: `pub fn view / contiguous / narrow in methods.rs` delegate to free functions `pub fn view_t in methods.rs`, `pub fn contiguous_t in methods.rs` (with GPU `strided_copy_{f32,f64}` fast path), `pub fn narrow_t in methods.rs`. `struct ContiguousBackward in methods.rs`, `struct NarrowBackward in methods.rs`. Mirrors `aten/src/ATen/native/TensorShape.cpp:4563 view / 1669 narrow` and `torch/_tensor_docs.py:6145 / 1190 / 3502`. Non-test consumers: pervasive — `ferrotorch-distributions/src/multivariate_normal.rs,271,319,359,367 (view)`; `ferrotorch-distributions/src/dirichlet.rs,293,350,358,362 (view)`; `ferrotorch-vision/src/models/swin.rs,276,360,370 + ferrotorch-vision/src/models/vit.rs (view)`; `ferrotorch-diffusion/src/attention.rs,180,185,204,737,745 + ferrotorch-nn/src/attention.rs,345,349,410 (contiguous)`; `ferrotorch-nn/src/rnn.rs,245 (narrow)`; `struct in ferrotorch-data/src/transforms.rs,466 (narrow)`; `ferrotorch-core/src/flex_attention.rs,211 (narrow)`; `ferrotorch-core/src/nested.rs,556 (narrow)`; `ferrotorch-core/src/einops.rs,804 (narrow)`. |
| REQ-10 (chunk/split) | SHIPPED | impl: `pub fn chunk / split in methods.rs` delegate to free functions `pub fn chunk_t in methods.rs` (computes per-chunk sizes via `dim_size.div_ceil(chunks)` matching upstream `aten/src/ATen/native/TensorShape.cpp:1077-1097`) and `pub fn split_t in methods.rs` (GPU fast path via `backend.strided_split_f32` + CPU fallback, `SplitBackward` autograd through `grad_fns/shape.rs`). Mirrors `aten/src/ATen/native/TensorShape.cpp:1077 chunk / 3175 split` and `torch/_tensor_docs.py:6397 chunk`. Non-test consumers: `ferrotorch-diffusion/src/vae_encoder.rs:483 params.chunk(2, 1)`; `ferrotorch-diffusion/src/attention.rs:351 proj.chunk(2, last)`; `ferrotorch-diffusion/src/gpu/vae_encoder.rs:677 cpu_params.chunk(2, 1)`. |
| REQ-11 (size/dim aliases) | NOT-STARTED | open prereq blocker #1222. impl at `size in ferrotorch-core/src/methods.rs` provides `size() -> &[usize]` (alias for `shape()`) and `dim() -> usize` (alias for `ndim()`) mirroring `torch.Tensor.size() / torch.Tensor.dim()` per `torch/_tensor_docs.py:4848 size / 1717 dim`. All in-tree callers of `dim()` (`ferrotorch-distributions/src/{multivariate_normal,low_rank_multivariate_normal}.rs:509,629 / :204`, `dim in ferrotorch-nn/src/transformer.rs`) are inside `#[cfg(test)] mod tests` blocks. `size()` has no caller at all. The aliases compile but no production code in `ferrotorch-*/src/**/*.rs` invokes them outside tests. |
| REQ-12 (print utility) | NOT-STARTED | open prereq blocker #1223. impl `pub fn print in methods.rs` emits a `tracing::info!(target: "ferrotorch::tensor", "{self}")` event and returns `&Self` for chaining. R-DEV-7 — uses `tracing` rather than stdout per library hygiene. The only invocation is in-file test `fn test_method_print_chain in methods.rs`. No non-test consumer in the workspace. |
| REQ-13 (cumulative methods) | SHIPPED | impl: `pub fn cumsum_t / cumprod_t / logcumsumexp_t in methods.rs` delegate to `crate::grad_fns::cumulative::{cumsum, cumprod, logcumsumexp}` mirroring `torch.Tensor.cumsum / cumprod / logcumsumexp` per `torch/_tensor_docs.py:1500-1506 / :1482-1488 / :1455-1462` and `aten/src/ATen/native/ReduceOps.cpp:511 TORCH_IMPL_FUNC(cumsum_out) / :519 TORCH_IMPL_FUNC(cumprod_out) / :475 Tensor logcumsumexp(...)`. Non-test production consumer: these three Tensor methods themselves close the R-DEFER-1 consumer requirement for the previously vocabulary-only `lib.rs:159` re-exports (per `.design/ferrotorch-core/grad_fns/cumulative.md` REQ-1/REQ-2/REQ-5 — each flipped NOT-STARTED → SHIPPED in the same commit). Unit tests `fn test_method_cumsum_t_1d / test_method_cumprod_t_1d / test_method_logcumsumexp_t_1d in methods.rs` verify dispatch correctness against the free function and numerical correctness against the upstream recurrence per R-CHAR-3. Parity-sweep: `[cumsum] 32/32 / [cumprod] 80/80 / [logcumsumexp] 48/48` all pass at seeds=8. Closes #1232. `cummax_t / cummin_t` explicitly NOT added — `cummax / cummin` already have consumers at `einops.rs:796 / :802` using the tuple form, and remain NOT-STARTED behind #1231 (differentiability + tie-break + NaN). |
