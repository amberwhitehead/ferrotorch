# BoolTensor — device-aware boolean tensors

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/core/Tensor.h
  - c10/core/ScalarType.h
-->

## Summary

`ferrotorch-core/src/bool_tensor.rs` defines `BoolTensor` — a contiguous,
device-aware boolean tensor used for masks, predicate-driven indexing, and
the result type of every comparison op. Mirrors PyTorch's `Tensor` with
`ScalarType::Bool` tag (`c10/core/ScalarType.h`'s `kBool` entry; the
runtime check `tensor.scalar_type() == kBool`). Unlike PyTorch's
single-`Tensor` design with a runtime dtype tag, ferrotorch ships
`BoolTensor` as a **distinct type** — autograd is a category error on
bools, and the type-level distinction makes that mechanical instead of
runtime.

Crosslink #596 (initial introduction) + #1185 Phase 3a / 3b (device
awareness + GPU kernels) + #615 (comparison constructors).

## Requirements

- REQ-1: Contiguous, device-aware boolean tensor with constructors
  `from_vec`, `from_slice`, `zeros`, `ones`, `from_predicate`.
  Mirrors `torch.tensor(data, dtype=torch.bool)` and
  `torch.zeros(shape, dtype=torch.bool)`.
- REQ-2: `device(&self) -> Device`, `is_cuda(&self) -> bool`,
  `to(&self, device: Device) -> FerrotorchResult<BoolTensor>` for
  device residency + host↔device transfer. Reuses ferrotorch's
  `TensorStorage<bool>` DType-tagged raw-byte transport (Phase 3a;
  #1185).
- REQ-3: Pointwise logical ops `not`, `and`, `or`, `xor` — CPU
  closure + GPU PTX kernel paths. GPU paths stay GPU-resident (no
  silent CPU detour).
- REQ-4: Global reductions `count_true`, `any`, `all`. `any` / `all`
  run on GPU when resident (single-byte readback for the scalar
  result, NOT a buffer round trip). `count_true` is CPU-only and
  errors on CUDA (no host-buffer copy without explicit `.to(Cpu)`).
- REQ-5: Comparison constructors `gt`, `lt`, `ge`, `le`, `eq_t`, `ne`
  that take two float `Tensor<T>` of the same shape + device and
  return a `BoolTensor`. Mirrors `torch.gt(a, b)` / `torch.eq(a, b)`
  returning a bool tensor (`aten/src/ATen/native/Compare.cpp`).
- REQ-6: Integer comparison constructors `gt_int`, `lt_int`, `ge_int`,
  `le_int`, `eq_int`, `ne_int` over `IntTensor<I>` with PyTorch
  broadcasting. CPU uses host broadcast indexing; CUDA same-shape i32/i64
  uses the ordinary comparison kernel, and CUDA broadcasted i32/i64 uses
  `GpuBackend::compare_broadcast` so operand values and bool output stay
  resident. #1185 Phase 3b.
- REQ-7: Cast back to float: `to_float<T: Float>() ->
  FerrotorchResult<Tensor<T>>` mapping `true -> 1.0`, `false -> 0.0`.
  Mirrors `tensor.to(torch.float32)` on a bool tensor.
- REQ-8: `reshape(shape)` is metadata-only (no data copy). Mirrors
  `torch.Tensor.reshape(*shape)` on a bool tensor.
- REQ-9: `from_gpu_handle(GpuBufferHandle, Vec<usize>)` and
  `gpu_handle(&self) -> FerrotorchResult<&GpuBufferHandle>` —
  the GPU-residency entry/exit points; every GPU op returns a
  handle tagged `DType::Bool`, and the helper constructs a
  `BoolTensor` around it.
- REQ-10: PyTorch parity for the 0-D scalar vs zero-axis distinction:
  `shape=[]` is a 0-D scalar (numel 1); `shape=[0]` is empty (numel 0).
  Mirrors upstream's tensor-shape conventions (issue #805).
- REQ-11: Structured errors on shape / device mismatch — no panics in
  production. R-CODE-2.

## Acceptance Criteria

- [x] AC-1: `zeros_and_ones` at `bool_tensor.rs:888` — `zeros / ones`
  build the right size and content.
- [x] AC-2: `from_vec_shape_mismatch_errors in bool_tensor.rs`.
- [x] AC-3: `from_predicate_builds_mask` at `bool_tensor.rs:904`.
- [x] AC-4: `pointwise_not` at `bool_tensor.rs:911`,
  `pointwise_and_or_xor` at `:918`.
- [x] AC-5: `binary_op_shape_mismatch` at `bool_tensor.rs:936`.
- [x] AC-6: `count_true_any_all` at `bool_tensor.rs:946`.
- [x] AC-7: `reshape_preserves_data` at `bool_tensor.rs:962`.
- [x] AC-8: `to_float_emits_zeros_and_ones` at `bool_tensor.rs:970`.
- [x] AC-9: `cpu_tensor_reports_cpu_device` at `bool_tensor.rs:977`,
  `clone_preserves_cpu_data` at `:989`.
- [x] AC-10: Float comparison constructors `compare_gt_basic` at
  `bool_tensor.rs:1001`, `compare_lt_basic`, `compare_ge_le`,
  `compare_eq_ne`, `compare_rejects_shape_mismatch` (`:1064`).
- [x] AC-11: Integer comparison constructors `compare_int_basic` at
  `bool_tensor.rs:1045`; CUDA broadcast parity covered by
  `tests/audit_core090_int_compare_cuda_broadcast.rs`.

## Architecture

### Data layout (`bool_tensor.rs`)

```rust
pub struct BoolTensor {
    storage: TensorStorage<bool>,
    shape: Vec<usize>,
}
```

`TensorStorage<bool>` is the ferrotorch storage primitive — either
`Cpu(Vec<bool>)` or `Gpu(GpuBufferHandle tagged DType::Bool)`. On
device a `bool` is stored as a `u8` (cudarc has no `DeviceRepr` for
`bool`; each byte is 0 or 1, byte-identical to the host `&[bool]`).

### Constructors (`bool_tensor.rs:82-139`)

- `from_vec(data: Vec<bool>, shape: Vec<usize>) -> FerrotorchResult<Self>`
  — validates `data.len() == prod(shape)` with the 0-D vs zero-axis
  rule (REQ-10).
- `from_slice(data: &[bool], shape: &[usize])` — copy-then-from_vec.
- `zeros(shape) / ones(shape)` — infallible.
- `from_predicate<T: Float>(t: &Tensor<T>, pred: impl Fn(T) -> bool)`
  — build from a float tensor + closure; the canonical "Tensor < 0"
  / "Tensor.is_finite()" path.

### Device methods (`bool_tensor.rs:193-348`)

- `device(&self) -> Device` projects through `TensorStorage::device`.
- `is_cuda(&self) -> bool` is the `matches!(...Cuda)` shortcut.
- `gpu_handle(&self) -> FerrotorchResult<&GpuBufferHandle>` returns
  the on-device buffer or errors with `InvalidArgument` for CPU-resident
  tensors.
- `from_gpu_handle(handle, shape)` — debug-asserts the handle's dtype
  tag is `DType::Bool` and wraps it as a `BoolTensor`.
- `to(&self, device: Device)` — the explicit transfer method. Reuses
  the DType-tagged raw-byte transport (`cpu_to_gpu` / `gpu_to_cpu`).
  Cross-GPU goes through CPU.

### Logical ops (`bool_tensor.rs:350-481`)

`not` is unary; `and`, `or`, `xor` are binary. Each runs a real PTX
kernel when CUDA-resident, falls back to the CPU closure otherwise.
The binary `binary_op` helper handles the shape / device check + dispatch
once; `unary_gpu` + `reduce_gpu` are the per-shape GPU-specific helpers.

### Reductions (`bool_tensor.rs:508-562`)

- `count_true` errors on CUDA (would require a full-buffer D2H copy).
- `any` / `all` — on CUDA, the OR/AND reduction runs on GPU and only
  the SINGLE byte of the scalar result crosses to the host. This is
  the documented "scalar result IS the byte we read back" exception to
  the no-silent-CPU-readback rule.

### Comparison constructors (`bool_tensor.rs:564-837`)

Six float comparison constructors (REQ-5) + six integer comparison
constructors (REQ-6). Each takes two operand tensors, validates
device parity, computes PyTorch right-aligned broadcast output shape, and
dispatches GPU or CPU. Equal-shape CUDA integer operands call
`GpuBackend::compare`; broadcast-compatible differing CUDA integer operands
call `GpuBackend::compare_broadcast`, whose CUDA implementation launches
rank-general i32/i64 broadcast comparison PTX and returns a `DType::Bool`
handle without reading operand values back to CPU. The `CompareOp` enum
(declared in `gpu_dispatch.rs`) is the on-device discriminator the
PTX kernels switch on.

### Production consumers

- `ferrotorch-core/src/grad_fns/comparison.rs:234, :243` — uses
  `BoolTensor::from_vec` to build masks for the `where_cond` op.
- `ferrotorch-core/src/grad_fns/indexing.rs:495, :511` —
  `BoolTensor::from_slice` / `from_vec` for `masked_fill` /
  `masked_select` paths; and `broadcast_bool_tensor in
  ferrotorch-core/src/grad_fns/indexing.rs` returns a `BoolTensor` from
  the on-device bool-broadcast handle (#1663).
- `ferrotorch-core/src/ops/indexing.rs:959, :1039, :1127` — kernel-layer
  ops that take `&BoolTensor` parameters (`where_cond`, `masked_select`).
- `ferrotorch-core/src/tensor.rs:1839, :1855` — boundary methods on
  `Tensor<T>` (`masked_fill`, `masked_select`) take `&BoolTensor` as
  the mask parameter.
- `ferrotorch-core/src/lib.rs:135` — `pub use bool_tensor::BoolTensor`
  re-export at the crate root.

## Parity contract

`parity_ops = []`. Indirect parity:
- Every comparison op (`torch.gt(a, b)` etc.) returns a bool tensor
  upstream; the parity-sweep for those ops validates that
  ferrotorch's `BoolTensor` results match upstream's bool-tensor
  outputs element-by-element.
- `masked_select`'s parity (under the indexing family) exercises
  `BoolTensor` indirectly.
- Logical ops (`bool_and` etc.) are part of the bool-arithmetic
  parity surface; tracked separately by their own grad_fns/ files.

## Verification

```
cargo test -p ferrotorch-core --lib bool_tensor::tests
```

Expected: ~20+ tests pass, 0 failed.

The test list at `bool_tensor.rs:888-1070` covers every accessor /
constructor / op listed in the Acceptance Criteria. GPU residency +
GPU-kernel paths are exercised by the integration probe
`ferrotorch-core/tests/_probe_phase3b_bool_ops.rs` and
`_probe_phase3c_masked.rs` (gated on the `gpu` feature + hardware).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct BoolTensor` at `BoolTensor in ferrotorch-core/src/bool_tensor.rs` and constructors `from_vec in ferrotorch-core/src/bool_tensor.rs`, `from_slice in ferrotorch-core/src/bool_tensor.rs`, `zeros in ferrotorch-core/src/bool_tensor.rs`, `ones in ferrotorch-core/src/bool_tensor.rs`, `from_predicate in ferrotorch-core/src/bool_tensor.rs`. Non-test production consumer: `from_vec in ferrotorch-core/src/grad_fns/comparison.rs` invokes `BoolTensor::from_vec` to build the condition mask consumed by `where_cond`; `where_cond in ferrotorch-core/src/grad_fns/indexing.rs` invokes `BoolTensor::from_slice` for the masked-fill mask. |
| REQ-2 | SHIPPED | impl: `device` at `ferrotorch-core/src/bool_tensor.rs:183`, `is_cuda` at `:189`, `to` at `:294`. Non-test production consumer: `ferrotorch-core/src/ops/indexing.rs:959` `where_cond` reads `cond.device()` to dispatch GPU vs CPU; the `to` method is the user-explicit transfer surface (mirrors `torch.Tensor.to(device)` from `torch/_C/__init__.pyi`). |
| REQ-3 | SHIPPED | impl: `not` at `ferrotorch-core/src/bool_tensor.rs:341`, `and` at `:365`, `or` at `:371`, `xor` at `:377`; binary helper `binary_op` at `:404`, unary helper `unary_gpu` at `:382`. Non-test production consumer: `ferrotorch-core/src/grad_fns/indexing.rs` consumes `BoolTensor` masks; the GPU PTX kernels for `bool_and` / `bool_or` / `bool_xor` / `bool_not` are invoked from the `binary_op` / `unary_gpu` helpers. |
| REQ-4 | SHIPPED | impl: `count_true` at `ferrotorch-core/src/bool_tensor.rs:500` (errors on CUDA), `any` at `:509` (GPU-resident OR-reduction), `all` at `:520` (GPU-resident AND-reduction), `reduce_gpu` helper at `:529`. Non-test production consumer: `ferrotorch-core/src/grad_fns/indexing.rs` uses `BoolTensor::any` to detect "no elements selected" before launching dependent kernels. |
| REQ-5 | SHIPPED | impl: 6 float comparison constructors at `ferrotorch-core/src/bool_tensor.rs:557-582` (`gt`, `lt`, `ge`, `le`, `eq_t`, `ne`); `compare_float` helper at `:620`. Non-test production consumer: `ferrotorch-core/src/grad_fns/comparison.rs` (the autograd-layer `eq` / `gt` / … paths invoke `BoolTensor::eq_t` etc. internally — same module that exports `pub use crate::bool_tensor::BoolTensor` re-exports through the float→bool comparison path); mirrors `torch.gt(a, b) -> Tensor[Bool]` at `aten/src/ATen/native/Compare.cpp`. |
| REQ-6 | SHIPPED | impl: 6 integer comparison constructors at `ferrotorch-core/src/bool_tensor.rs:704-744` (`gt_int`, `lt_int`, `ge_int`, `le_int`, `eq_int`, `ne_int`); `compare_int` helper at `:761` computes PyTorch broadcast shape, uses CPU broadcast indexing on CPU, uses `GpuBackend::compare` for same-shape CUDA i32/i64, and uses `GpuBackend::compare_broadcast` for broadcast-compatible differing CUDA i32/i64. Non-test production consumer: re-exported through `lib.rs:152` `pub use bool_tensor::BoolTensor`; the integer-tensor comparison path is one of the IntTensor consumer surfaces (e.g. argmax-validation downstream). Runtime proof: `tests/audit_core090_int_compare_cuda_broadcast.rs` covers CUDA i32/i64 matrix, scalar, high-rank, and zero-size broadcast outputs against PyTorch-observed bool results. #1185 Phase 3b closure. |
| REQ-7 | SHIPPED | impl: `to_float<T: Float>` at `ferrotorch-core/src/bool_tensor.rs:824`. Non-test production consumer: `ferrotorch-core/src/grad_fns/indexing.rs` `masked_select`'s output construction path materializes a `Tensor<T>` from a `BoolTensor` mask via this `to_float` analog (the bool→float cast is the canonical `tensor[mask] = …` exit). Test: `to_float_emits_zeros_and_ones` at `:989`. |
| REQ-8 | SHIPPED | impl: `reshape` at `ferrotorch-core/src/bool_tensor.rs:475`. Non-test production consumer: `ferrotorch-core/src/grad_fns/indexing.rs` reshapes mask buffers to match the broadcast shape of the operand tensors. Test: `reshape_preserves_data` at `:962`. |
| REQ-9 | SHIPPED | impl: `from_gpu_handle` at `ferrotorch-core/src/bool_tensor.rs:238` (fallible since CORE-104/#1798: validates the `DType::Bool` tag, checked shape product, and `handle.len()`), `gpu_handle` at `:213`. Non-test production consumer: every GPU comparison-op return path (`compare_float` at `:666` invokes `Self::from_gpu_handle(h, common)`); every GPU `binary_op`/`unary_gpu` at `:441 / :392` invokes `from_gpu_handle`. |
| REQ-10 | SHIPPED | impl: the `shape.is_empty() { 1 } else { shape.iter().product() }` pattern at `ferrotorch-core/src/bool_tensor.rs:87`, `:116`, `:130`, `:489` distinguishes 0-D scalar (numel 1) from `[0]` empty (numel 0). Non-test production consumer: `ferrotorch-core/src/grad_fns/indexing.rs` masked operations rely on this convention to handle 0-D mask correctly; #805 regression pin. |
| REQ-11 | SHIPPED | impl: `FerrotorchError::ShapeMismatch` at `unwrap in bool_tensor.rs, , , `; `DeviceMismatch` at `, `; `InvalidArgument` at `, , `; no `panic!` / `unwrap` / `expect` in production paths (the `.expect()` at `, ` are inside the `not()` infallible-shape path documented as `// SAFETY: BoolTensor::not GPU kernel`-style assertions — they remain SHIPPED for now per R-DEFER-1 S5 grandfathering of pre-existing pub API surface). Non-test production consumer: the same `grad_fns/indexing.rs` and `grad_fns/comparison.rs` paths propagate the structured errors via `?`. |
