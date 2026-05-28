# DType — float element-type taxonomy

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - c10/core/ScalarType.h
  - aten/src/ATen/Dispatch.h
-->

## Summary

`ferrotorch-core/src/dtype.rs` is the workspace's dtype gateway. It
re-exports ferray's sealed `Element` trait and `DType` enum (the typed-bytes
layer ferrotorch sits on top of) and adds a `Float` marker trait that the
autograd-aware `Tensor<T>` operations bound on. Mirrors PyTorch's
`c10::ScalarType` (`c10/core/ScalarType.h`) — the same taxonomy of
floating-point + integer + complex + quantized + bool element types — but
expressed as Rust generic-`T: Float` constraints rather than a
runtime-tagged `ScalarType` enum match per kernel. The dynamic enum form
is still available (`DType` from ferray) and tags GPU buffers; the static
form is what compile-time generics use.

## Requirements

- REQ-1: Re-export `ferray_core::DType` and `ferray_core::Element` so the
  rest of the workspace has a single import path for the dtype enum and the
  sealed element trait. Mirrors `c10::ScalarType` (`ScalarType.h:38-43
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS`).
- REQ-2: A `Float` marker trait whose bound is
  `Element + num_traits::Float + std::ops::AddAssign`. This is the bound
  used by every autograd-tracking op (`Tensor<T: Float>::add_t`, …) and
  expresses "T is a differentiable floating-point dtype that ferrotorch
  supports for forward + backward". Implementors: `f32`, `f64`, `half::bf16`,
  `half::f16`. Mirrors PyTorch's runtime
  `isDifferentiableType(ScalarType) == true` predicate at `Dispatch.h`'s
  `AT_DISPATCH_FLOATING_TYPES_AND_HALF` macro.
- REQ-3: `bf16` (Google brain-float, 8-bit exponent / 7-bit mantissa) is a
  first-class `Float` implementor — Llama 3 / 8B inference and most modern
  transformer weight storage are bf16. Mirrors PyTorch's `kBFloat16`
  scalar type at `c10/util/BFloat16.h`.
- REQ-4: IEEE `f16` (`half::f16`, 5-bit exponent / 10-bit mantissa) is a
  separate `Float` implementor, distinct from `bf16` despite sharing the
  2-byte storage width. Disambiguated by the `DType::F16` tag returned by
  `<half::f16 as Element>::dtype()`. Mirrors PyTorch's `kHalf` scalar type
  at `c10/util/Half.h`. Closes crosslink #1185 Phase 1.
- REQ-5: Integer and boolean element types exist (they satisfy `Element`
  via ferray) but are intentionally NOT `Float` — autograd is a category
  error on them. Mirrors PyTorch's
  `isDifferentiableType(ScalarType) == false` for integer dtypes (used as
  a runtime check at `aten/src/ATen/autograd/VariableTypeUtils.h`).
- REQ-6: Each `Float` implementor's `<T as Element>::dtype()` returns the
  correct `DType` enum variant for GPU buffer tagging — `f32 -> F32`,
  `f64 -> F64`, `bf16 -> BF16`, `f16 -> F16`. The GPU dispatch path
  (`gpu_dispatch.rs::cpu_to_gpu`) reads this tag to pick the right
  on-device representation.

## Acceptance Criteria

- [x] AC-1: `bf16_is_float` at `dtype.rs:42` — compile-time check
  `assert_float::<half::bf16>()` succeeds; `bf16` satisfies `Float`.
- [x] AC-2: `f16_is_float` at `dtype.rs:56` — compile-time check
  `assert_float::<half::f16>()` succeeds; IEEE `f16` satisfies `Float`,
  distinct from `bf16`.
- [x] AC-3: `bf16_element_dtype` at `dtype.rs:50` —
  `<half::bf16 as Element>::dtype() == DType::BF16`.
- [x] AC-4: `f16_element_dtype` at `dtype.rs:63` —
  `<half::f16 as Element>::dtype() == DType::F16` AND `!= DType::BF16`
  (disambiguation check for crosslink #1185 Phase 1).
- [x] AC-5: `bf16_num_traits_float_ops` at `dtype.rs:86` — `Float::sqrt`
  and `+` operators compose via `num_traits::Float`.
- [x] AC-6: `bf16_add_assign` at `dtype.rs:101` — `+=` works on bf16.
- [x] AC-7: `bf16_tensor_construction_and_shape` at `dtype.rs:108` and
  `bf16_tensor_addition` at `dtype.rs:124` — end-to-end tensor build +
  arithmetic on bf16.

## Architecture

### Re-export surface

```rust
pub use ferray_core::{DType, Element};  // dtype.rs:6
```

`DType` is the runtime enum tag (one variant per scalar type, like
`c10::ScalarType`). `Element` is the sealed trait every tensor element
type implements — sealed because only ferray knows the full set, which
keeps the workspace's dtype universe synchronized with the typed-bytes
layer.

### `Float` marker trait (`dtype.rs:26-35`)

```rust
pub trait Float: Element + num_traits::Float + std::ops::AddAssign {}

impl Float for f32 {}
impl Float for f64 {}
impl Float for half::bf16 {}
impl Float for half::f16 {}
```

The bound has three parts:
1. `Element` — element type known to ferray (storage-compatible).
2. `num_traits::Float` — full IEEE-754 surface (`sqrt`, `exp`, `floor`,
   classification predicates `is_nan` / `is_finite`, …).
3. `std::ops::AddAssign` — `+=` is the accumulator primitive every reduction
   relies on (sum, mean, dot product).

The trait is **the** bound used by `Tensor<T: Float>` — passing it instead
of a concrete type into a generic ferrotorch op signature picks up
`num_traits::Float`'s full vocabulary.

### bf16 vs f16 disambiguation (REQ-3 + REQ-4)

Both `half::bf16` and `half::f16` are 2-byte floats but they have
**different bit layouts**:
- `bf16`: sign | 8-bit exponent | 7-bit mantissa (same exponent range as
  `f32` — well-suited for weights that span wide dynamic range; mantissa
  precision is sacrificed).
- `f16`: sign | 5-bit exponent | 10-bit mantissa (better precision in a
  narrow dynamic range — used by NVIDIA fp16 mixed-precision training
  before bf16 hardware support was widespread).

On disk both occupy 2 bytes; the only way to tell them apart is the dtype
tag. ferrotorch carries that tag via `DType::BF16` vs `DType::F16`, and
the `ScalarType` taxonomy is mirrored byte-for-byte from
`c10::ScalarType::BFloat16` / `c10::ScalarType::Half`. Confusing them on
GPU upload (sending bf16 bits to a fp16 kernel) corrupts every output —
the test `f16_element_dtype` at `dtype.rs:63` pins the disambiguation
with `assert_ne!(F16, BF16)`.

### Integer / bool deliberately not Float (REQ-5)

`i32`, `i64`, `bool` are `Element` implementors (so `TensorStorage<i32>`
and friends work) but **not** `Float` — this surfaces as a compile error
when a caller tries to `Tensor::<i32>::add_t` instead of using
`IntTensor::<i32>::add`. Upstream's `isDifferentiableType(kInt) == false`
is a runtime check; ferrotorch lifts it to a trait-bound (R-DEV-5
typestate-when-ordering-matters: refusing to autograd-track ints is
enforced at the type level rather than at runtime).

### Production consumers

- `ferrotorch-core/src/tensor.rs` — `pub struct Tensor<T: Float>` at the
  module entry. Every method on `Tensor<T>` carries the `Float` bound by
  inheritance, so the rest of the public API surface (`add_t`, `mul_t`,
  `cumsum_t`, …) inherits it transitively.
- `ferrotorch-core/src/grad_fns/arithmetic.rs:45-57` — defines the local
  `is_f32::<T>`, `is_f64::<T>`, `is_bf16::<T>` `Float`-bounded helpers used
  by GPU dispatch arms.
- `is_f32 in ferrotorch-core/src/dtype_dispatch.rs / ` — `is_f32`,
  `is_f64`, `is_bf16`, `is_f16` helpers all take `T: 'static` and key on
  `TypeId`; `dispatch_floating_dtype!` uses them to branch.

## Parity contract

`parity_ops = []` — this file ships infrastructure. The parity surface is
the indirect bf16-vs-f16 dispatch correctness, exercised by every op's
parity-sweep run with seeded inputs in those dtypes. crosslink #1185 Phase 1
documented that conflating the two had been silently routing `bf16` GPU
ops to a `f32` arm; ferrotorch now refuses that fallthrough at the trait
level (`dtype.rs:26-35` defines the supported set; anything outside it
fails to compile).

## Verification

```
cargo test -p ferrotorch-core --lib dtype::tests
```

Expected: 7 tests pass, 0 failed.

Tests (named above in Acceptance Criteria) cover compile-time `Float`
bound satisfaction, the `<T as Element>::dtype()` GPU-tag projection, the
bf16/f16 disambiguation, and the end-to-end tensor-construction +
arithmetic happy path on bf16.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub use ferray_core::{DType, Element}` at `dtype in ferrotorch-core/src/dtype.rs` mirroring `c10::ScalarType` (`c10/core/ScalarType.h:38-43`). Non-test production consumer: `dtype in ferrotorch-core/src/lib.rs` re-exports `DType, Element, Float` for downstream crates, and `ferrotorch-core/src/storage.rs` uses `<T as Element>::dtype()` to tag every GPU buffer (the on-device dtype carrier). |
| REQ-2 | SHIPPED | impl: trait `Float` at `f32 in ferrotorch-core/src/dtype.rs` with bound `Element + num_traits::Float + std::ops::AddAssign`, implementors at `f32 in ferrotorch-core/src/dtype.rs` (`f32`, `f64`, `bf16`, `f16`). Non-test production consumer: `ferrotorch-core/src/tensor.rs` `pub struct Tensor<T: Float>` and every `<T: Float>` op in `ferrotorch-core/src/grad_fns/arithmetic.rs` (e.g. `pub fn add<T: Float>` at `add in arithmetic.rs`). |
| REQ-3 | SHIPPED | impl: `impl Float for half::bf16` at `ferrotorch-core/src/dtype.rs:30`. Non-test production consumer: `ferrotorch-core/src/grad_fns/arithmetic.rs:413` `dispatch_floating_dtype!` `bf16 =>` arm — invokes `backend.add_bf16_bf16(...)`, the bf16 GPU kernel path. End-to-end exercised by `bf16_tensor_addition` at `dtype.rs:124`. |
| REQ-4 | SHIPPED | impl: `impl Float for half::f16` at `ferrotorch-core/src/dtype.rs:35`. Non-test production consumer: `ferrotorch-core/src/dtype_dispatch.rs:111` — `dispatch_floating_dtype!` macro has a dedicated `f16 =>` arm. Disambiguation test `f16_element_dtype` at `dtype.rs:63` pins `DType::F16 != DType::BF16`. Closes crosslink #1185 Phase 1. |
| REQ-5 | SHIPPED | impl: no `impl Float for i32 / i64 / bool` exists in `ferrotorch-core/src/dtype.rs`. The absence is the contract. Non-test production consumer: `IntElement in ferrotorch-core/src/int_tensor.rs` `IntElement: Element + Copy + Send + Sync + 'static` is the integer-specific bound (no `num_traits::Float`); `IntTensor<I: IntElement>` cannot be passed to `<T: Float>` op signatures (compile error). |
| REQ-6 | SHIPPED | impl: ferray's `Element::dtype()` returns the right tag — verified by `bf16_element_dtype` at `dtype in dtype.rs` (`bf16 -> BF16`) and `f16_element_dtype in dtype.rs` (`f16 -> F16`). Non-test production consumer: `ferrotorch-core/src/storage.rs` `TensorStorage::on_device` (the upload path tags every GPU buffer with `<T as Element>::dtype()`); also `dtype in ferrotorch-core/src/int_tensor.rs` `from_gpu_handle` debug-asserts `handle.dtype() == I::dtype()` (the same tag-as-source-of-truth pattern). |
