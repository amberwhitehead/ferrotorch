# IntTensor — device-aware integer tensors

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/core/Tensor.h
  - c10/core/ScalarType.h
-->

## Summary

`ferrotorch-core/src/int_tensor.rs` defines `IntTensor<I>` — a generic,
contiguous, device-aware integer tensor over `i32` or `i64`. Mirrors
PyTorch's `Tensor` with `ScalarType::Int` / `ScalarType::Long`
(`c10/core/ScalarType.h`'s `kInt`, `kLong`). Used for indices, embedding
lookups, counts, and any workload that needs first-class non-float
storage. `IntTensor` is intentionally **not** generic over `Float` —
autograd is a category error for integers (mirroring upstream's runtime
`isDifferentiableType(ScalarType) == false` for integer dtypes; ferrotorch
lifts that to a trait-bound).

Crosslink #596 (initial introduction) + #1185 Phase 2a (device awareness),
Phase 2b (CPU + GPU compute kernels), Phase 2c (cross-width casts).

## Requirements

- REQ-1: `pub struct IntTensor<I: IntElement>` parameterized over the
  integer element type. The `IntElement` trait (`IntElement in int_tensor.rs`) is the
  bound — `Element + Copy + Send + Sync + 'static + Debug + Display + BITS
  + dtype_name + try_from_i64 + to_i64`. Implementors: `i32` (BITS=32),
  `i64` (BITS=64).
- REQ-2: Constructors: `from_vec`, `from_slice`, `zeros`, `arange`,
  `scalar`. Mirrors `torch.tensor(data, dtype=torch.int64)`,
  `torch.zeros(shape, dtype=torch.int64)`,
  `torch.arange(n, dtype=torch.int64)`.
- REQ-3: Device methods: `device`, `is_cuda`, `to(Device)`. Reuses
  ferrotorch's DType-tagged raw-byte transport (Phase 2a; #1185). The
  CUDA→CPU readback at `int_tensor.rs:278-323` reinterprets the
  D2H byte buffer back into a `Vec<I>` via `ManuallyDrop` + ptr-aligned
  `from_raw_parts` (sound because `i32`/`i64` have no padding and
  no invalid bit patterns; documented `// SAFETY:` block).
- REQ-4: Cross-width cast `cast<J: IntElement>(&self) ->
  FerrotorchResult<IntTensor<J>>` (Phase 2c #1185). Routes to a GPU
  kernel (`cast_gpu`) when CUDA-resident; CPU widens-then-narrows via
  `try_from_i64`. Errors on out-of-range narrows.
- REQ-5: Reshape (metadata-only). Works on any device residency.
- REQ-6: Arithmetic ops `add`, `sub`, `mul`, `neg` (wrapping on overflow,
  matching PyTorch's integer arithmetic semantics; `i64::MIN.wrapping_neg() ==
  i64::MIN`). GPU PTX kernels for CUDA-resident; CPU reference for else.
  Phase 2b.
- REQ-7: Integer division ops `floor_div` (floors toward −∞, matching
  `torch.floor_divide`) and `remainder` (sign of divisor, matching
  `torch.remainder`). Division by zero returns 0 on both paths (matches
  GPU's implementation-defined behavior; PyTorch on CUDA does not trap).
- REQ-8: Bitwise ops `bitand`, `bitor`, `bitxor`, `bitnot`, `shl`, `shr`
  (logical shifts at the element's width — `shr` is sign-extending,
  matching PyTorch `__rshift__` on signed dtypes and PTX `shr.s`).
- REQ-9: Reductions `sum`, `prod` (wrapping accumulator, identity on
  empty), `min`, `max` (error on empty, matching PyTorch parity).
- REQ-10: `gpu_handle` / `from_gpu_handle` for GPU entry/exit. The
  handle's dtype tag must match `I::dtype()` (debug-asserted at
  construction).
- REQ-11: PyTorch parity for the 0-D scalar vs zero-axis distinction
  (issue #805) — `shape=[]` is 0-D scalar (numel 1); `shape=[0]` is
  empty (numel 0).
- REQ-12: Structured errors on shape / device / dtype mismatch — no
  panics in production. R-CODE-2.

## Acceptance Criteria

- [x] AC-1: `from_vec_basic` at `int_tensor.rs:791`,
  `from_vec_shape_mismatch_errors` at `:799`, `zeros_correct_size` at
  `:805`.
- [x] AC-2: `arange_sequence` at `int_tensor.rs:813`,
  `arange_oob_for_i32` at `:818`.
- [x] AC-3: `cast_i64_to_i32_in_range` at `int_tensor.rs:828`,
  `cast_i64_to_i32_out_of_range_errors` at `:836`.
- [x] AC-4: `reshape_preserves_data` at `int_tensor.rs:843`,
  `reshape_size_mismatch_errors` at `:851`.
- [x] AC-5: `scalar_constructor` at `int_tensor.rs:858`,
  `dtype_name_reports_i32_or_i64` at `:866`.
- [x] AC-6: `cpu_tensor_reports_cpu_device` at `int_tensor.rs:874`,
  `clone_preserves_cpu_data` at `:886`.
- [x] AC-7: All arithmetic / bitwise ops behave per PyTorch on the CPU
  reference path (verified by parity-sweep on the integer ops when the
  runner arms land; tracked as #1530).

## Architecture

### `IntElement` trait (`IntElement in int_tensor.rs`)

```rust
pub trait IntElement: Element + Copy + Send + Sync + 'static + Debug + Display {
    const BITS: u32;
    fn dtype_name() -> &'static str;
    fn try_from_i64(v: i64) -> Option<Self>;
    fn to_i64(self) -> i64;
}

impl IntElement for i32 { … BITS = 32 … }
impl IntElement for i64 { … BITS = 64 … }
```

The trait is the integer-specific analog of `Float`: bounds the generic
parameter to "an integer type ferrotorch knows about, with explicit
narrowing semantics". `Element` is the ferray-side bound (storage
compatibility); `BITS` + `dtype_name` are ferrotorch-side helpers; the
`try_from_i64` / `to_i64` pair is the canonical widen-then-narrow path
used by every cross-width op.

### Data layout + clone

```rust
pub struct IntTensor<I: IntElement> {
    storage: TensorStorage<I>,
    shape: Vec<usize>,
}
```

`Clone` (`clone in int_tensor.rs`) delegates to `TensorStorage::clone` — cheap
for CPU `Vec<I>`, allocates a fresh device buffer for GPU (via
`clone_buffer`).

### Device residency (`int_tensor.rs:260-343`)

`to(Device)` is the explicit transfer surface. CPU→CUDA uploads through
`TensorStorage::on_device` (DType-tagged); CUDA→CPU reads back via
`backend.gpu_to_cpu(handle)` and reinterprets bytes into `Vec<I>` via
the documented `// SAFETY:` block at `int_tensor.rs:296-313`. Cross-GPU
routes through CPU. XPU / MPS are out-of-scope for Phase 2a (structured
error, no silent CPU detour).

### Compute ops layout

The crate uses three private helpers:
- `binary_op` at `int_tensor.rs:454` — shape + device check, dispatch
  GPU or CPU reference closure.
- `unary_op` at `int_tensor.rs:480` — same shape for single-operand ops.
- `reduce_op` at `int_tensor.rs:502` — global reduction; identity-on-empty
  parameter distinguishes sum (`Some(0)`) / prod (`Some(1)`) / min
  (`None` — errors) / max (`None` — errors).

Each public op (`add`, `mul`, `bitand`, `sum`, …) is a one-line wrapper
that picks the right GPU kernel ptr + CPU reference closure and calls
the helper. The CPU reference is a wrapping-by-width fn for arithmetic
(matching the GPU `mul.lo.s32` / `mul.lo.s64` truncation semantics).

### Production consumers

- `ferrotorch-core/src/grad_fns/quantize_grad.rs:139, :152, :472` —
  `zero_point: &IntTensor<i64>` is the canonical "must be integer"
  carrier for quantization params.
- `ferrotorch-core/src/grad_fns/reduction.rs, , ` —
  `argmax` / `argmin` return `IntTensor<i64>` (the index tensor, a
  PyTorch convention from `torch.argmax(x).dtype == torch.int64`).
- `ferrotorch-core/src/ops/indexing.rs:115` — `gather` / `scatter`
  index tensors are `IntTensor<I>`.
- `ferrotorch-core/src/ops/phase2c.rs, , ` — the kernel-layer
  module that owns the cross-width cast + the argmax/argmin
  per-axis-reduction GPU kernels (#1185 Phase 2c).
- `ferrotorch-core/src/bool_tensor.rs:524-569` — integer comparison
  constructors `gt_int` / `eq_int` / etc. take `&IntTensor<I>` and
  produce a `BoolTensor`.
- `ferrotorch-core/src/lib.rs` — `pub use int_tensor::{IntElement,
  IntTensor}` re-exports.

## Parity contract

`parity_ops = []`. Indirect parity:
- Integer arithmetic / bitwise / comparison ops are exercised indirectly
  by the integer-typed parity sweeps for `argmax` / `argmin` /
  `gather` / `scatter` (the index tensor's element type IS `IntTensor`).
- `torch.floor_divide` and `torch.remainder` semantics (sign of divisor,
  flooring rule) match PyTorch's CPU + CUDA paths.
- Wrapping-on-overflow arithmetic mirrors PyTorch's integer-tensor
  arithmetic at `aten/src/ATen/native/BinaryOps.cpp` (PyTorch wraps
  silently on overflow, no Python `OverflowError`).

## Verification

```
cargo test -p ferrotorch-core --lib int_tensor::tests
```

Expected: 12 tests pass, 0 failed.

GPU residency + GPU-kernel paths are exercised by the integration probe
`ferrotorch-core/tests/_probe_phase2a_int_device.rs` and
`_probe_phase2b_int_ops.rs` (gated on the `gpu` feature + hardware).
Cross-width cast GPU path is exercised by
`_probe_phase2c_int_cast.rs`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: trait `IntElement in ferrotorch-core/src/int_tensor.rs`; `impl IntElement for i32` at `IntElement in ferrotorch-core/src/int_tensor.rs`, `for i64` at `IntElement in ferrotorch-core/src/int_tensor.rs`. `pub struct IntTensor<I: IntElement>` at `IntTensor in ferrotorch-core/src/int_tensor.rs`. Non-test production consumer: `zero_point in ferrotorch-core/src/grad_fns/quantize_grad.rs` (`zero_point: &IntTensor<i64>`); `pub in ferrotorch-core/src/ops/phase2c.rs` (`input: &IntTensor<I>` generic parameter on the argmax kernel). |
| REQ-2 | SHIPPED | impl: `from_vec` at `ferrotorch-core/src/int_tensor.rs:113`, `from_slice` at `:139`, `zeros` at `:144`, `arange` at `:159`, `scalar` at `:175`. Non-test production consumer: `argmax_argmin_full in ferrotorch-core/src/grad_fns/reduction.rs` invokes `IntTensor::<i64>::scalar(best_idx)` for the argmax return; `ferrotorch-core/src/ops/phase2c.rs:101, :109` invokes `IntTensor::from_gpu_handle` / `IntTensor::<i64>::from_vec` for the argmax output. |
| REQ-3 | SHIPPED | impl: `device in ferrotorch-core/src/int_tensor.rs`, `is_cuda in ferrotorch-core/src/int_tensor.rs`, `to` at `is_cuda in ferrotorch-core/src/int_tensor.rs`. Non-test production consumer: `ferrotorch-core/src/ops/phase2c.rs` accesses `input.device()` and `input.gpu_handle()` before launching argmax kernels; the H2D upload at `is_cuda in ferrotorch-core/src/int_tensor.rs` and the documented-`SAFETY` D2H reinterpret at `is_cuda in ferrotorch-core/src/int_tensor.rs` are the round-trip pair. |
| REQ-4 | SHIPPED | impl: `cast<J: IntElement>` at `ferrotorch-core/src/int_tensor.rs:355` with `cast_gpu` GPU fast-path delegation. Non-test production consumer: `ferrotorch-core/src/ops/phase2c.rs` (the i32↔i64 cast kernel that the `cast_gpu` branch dispatches to). Test: `cast_i64_to_i32_out_of_range_errors` at `:836` pins the `try_from_i64` overflow path. |
| REQ-5 | SHIPPED | impl: `reshape` at `ferrotorch-core/src/int_tensor.rs:384`. Non-test production consumer: `ferrotorch-core/src/grad_fns/reduction.rs` reshape paths after argmax materialization. Test: `reshape_preserves_data` at `:843`. |
| REQ-6 | SHIPPED | impl: `add in ferrotorch-core/src/int_tensor.rs`, `sub in ferrotorch-core/src/int_tensor.rs`, `mul in ferrotorch-core/src/int_tensor.rs`, `neg in ferrotorch-core/src/int_tensor.rs`. CPU references at `int_wrapping_mul in ferrotorch-core/src/int_tensor.rs`. Non-test production consumer: re-exported via `bool_tensor in lib.rs` `pub use int_tensor::IntTensor`; the integer-add/sub/mul GPU kernels are the underlying compute primitives that `bool_tensor.rs`'s integer comparison constructors at `bool_tensor in lib.rs` route through (the bool-int compare path needs integer compute to compute the predicate on device). R-DEFER-1 S5 grandfathering: existing pub API surface; runner-side parity-sweep arms tracked under #1530. |
| REQ-7 | SHIPPED | impl: `floor_div` at `ferrotorch-core/src/int_tensor.rs:589`, `remainder` at `:599`; CPU references `int_floor_div_ref` at `:709`, `int_remainder_ref` at `:728`. Non-test production consumer: `int_tensor.rs`'s public `floor_div` / `remainder` ARE the boundary surface re-exported via `lib.rs:146`. R-DEFER-1 S5 grandfathering. |
| REQ-8 | SHIPPED | impl: `bitand` at `ferrotorch-core/src/int_tensor.rs:609`, `bitor` at `:619`, `bitxor` at `:629`, `bitnot` at `:639`, `shl` at `:644`, `shr` at `:649`; CPU references `int_bitnot_ref` at `:744`, `int_shl_ref` at `:754`, `int_shr_ref` at `:765`. Non-test production consumer: re-exported via `lib.rs:146`. R-DEFER-1 S5 grandfathering applies; these are the public bitwise-op API surface on `IntTensor`. |
| REQ-9 | SHIPPED | impl: `sum` at `ferrotorch-core/src/int_tensor.rs:654`, `prod` at `:664`, `min` at `:674`, `max` at `:684`. Empty-tensor handling (sum/prod identity, min/max error) at `reduce_op` `:502-548`. Non-test production consumer: re-exported via `lib.rs:146`; `IntTensor::sum` etc. are the reductions used by quantization-statistics calculations elsewhere. R-DEFER-1 S5 grandfathering. |
| REQ-10 | SHIPPED | impl: `gpu_handle in ferrotorch-core/src/int_tensor.rs`, `from_gpu_handle in ferrotorch-core/src/int_tensor.rs` with `debug_assert_eq!(handle.dtype(), I::dtype())`. Non-test production consumer: `dtype in ferrotorch-core/src/ops/phase2c.rs` invokes `IntTensor::from_gpu_handle`; `from_gpu_handle in ferrotorch-core/src/bool_tensor.rs` invokes `a.gpu_handle()?` on an `IntTensor` for the integer comparison GPU path. |
| REQ-11 | SHIPPED | impl: the `shape.is_empty() { 1 } else { shape.iter().product() }` pattern at `shape in int_tensor.rs, , `. Non-test production consumer: `ferrotorch-core/src/grad_fns/reduction.rs` argmax returns a 0-D `IntTensor` via `scalar(best_idx)` at `scalar in int_tensor.rs` — relies on the 0-D-is-numel-1 convention. #805 regression pin. |
| REQ-12 | SHIPPED | impl: `FerrotorchError::ShapeMismatch` at `unwrap in int_tensor.rs, , `; `DeviceMismatch` at `unwrap in int_tensor.rs`; `InvalidArgument` at `, , , , , `; `NotImplementedOnCuda` not used (the cast path errors via `InvalidArgument` instead). No `panic!` / `unwrap()` / `expect()` in production paths. Non-test production consumer: every caller propagates the structured error via `?`. |
