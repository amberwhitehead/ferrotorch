# Device — tensor location enum

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - c10/core/Device.h
  - c10/core/DeviceType.h
  - torch/_C/__init__.pyi
-->

## Summary

`ferrotorch-core/src/device.rs` defines the `Device` enum — where a
tensor's backing storage lives. Mirrors PyTorch's `c10::Device`
(`c10/core/Device.h:31`), a `(DeviceType, DeviceIndex)` pair. Rust expresses
it as a sum type whose variants carry their device-ordinal `usize` payload
inline; CPU has no ordinal (PyTorch's CPU is required to have `index <= 0`
per the upstream validator at `c10/core/Device.h:181`).

## Requirements

- REQ-1: `Device::Cpu` variant — the default. Mirrors `c10::DeviceType::CPU`
  at `c10/core/DeviceType.h`. `Device::default() == Device::Cpu` so
  `let t: Tensor<f32> = ...;` produces a CPU tensor unless a `.cuda()` /
  `.to_device(...)` lands explicitly. R-DEV-2 Python-API parity: PyTorch
  defaults the same way.
- REQ-2: `Device::Cuda(usize)` variant carrying the CUDA device ordinal.
  Mirrors `c10::DeviceType::CUDA` + the `DeviceIndex` payload at
  `c10/core/Device.h:36`. The `usize` payload diverges from upstream's
  `int8_t` (R-DEV-4: Rust's `usize` is the natural index type and
  pathological huge ordinals never reach this code).
- REQ-3: `Device::Xpu(usize)` variant — Intel XPU (Arc / Data Center GPU
  Max), accessed via `ferrotorch-xpu`'s CubeCL wgpu runtime. Mirrors
  `c10::DeviceType::XPU` at `c10/core/DeviceType.h`. Tracked as CL-452.
- REQ-4: `Device::Mps(usize)` variant — Apple Silicon Metal Performance
  Shaders. Mirrors `c10::DeviceType::MPS`. Implemented via `ferrotorch-mps`
  (#451).
- REQ-5: `Device::Meta` variant — shape-only, no backing storage. Used for
  dry-run model construction and parameter-count inspection without
  allocating weights. Mirrors `torch.device("meta")` at
  `c10/core/Device.h:151 is_meta()`. CL-395.
- REQ-6: Predicates `is_cpu`, `is_cuda`, `is_xpu`, `is_mps`, `is_meta` —
  fast variant checks via `matches!`. Mirrors upstream's `is_cpu()`,
  `is_cuda()`, `is_meta()`, `is_xpu()`, `is_mps()` methods on `c10::Device`
  at `Device.h:81-158`.
- REQ-7: `Display` impl produces `"cpu"`, `"cuda:0"`, `"xpu:0"`, `"mps:0"`,
  `"meta"` — matches upstream's `c10::Device::str()` at `Device.h:166` so
  log messages and error strings are interchangeable. R-DEV-2 Python-API
  parity.
- REQ-8: `Copy + Clone + Debug + PartialEq + Eq + Hash` derives — `Device`
  is a small POD that should be cheap to copy and hashable for use in
  `HashMap` keys (the dispatcher's per-device kernel table, the GPU
  backend registry).

## Acceptance Criteria

- [x] AC-1: `Device::default() == Device::Cpu` (verified by
  `#[derive(Default)]` plus `#[default] Cpu` at `device.rs:15`).
- [x] AC-2: `Device::Cuda(0).is_cuda()` returns `true` and `is_cpu()`
  returns `false` (mechanical check via `matches!` in `is_cuda` /
  `is_cpu`).
- [x] AC-3: `format!("{}", Device::Cuda(3))` produces `"cuda:3"` per the
  `Display` impl at `device.rs:66-76`.
- [x] AC-4: `Device::Cpu == Device::Cpu` and `Device::Cuda(0) !=
  Device::Cuda(1)` (verified by `PartialEq` derive).
- [x] AC-5: `Device::Cpu` and `Device::Cuda(0)` are both `Copy` — passing
  a `Device` to a function does not move it. (Mechanical: the derive at
  `device.rs:12` includes `Copy`.)
- [x] AC-6: `Device` is `Hash` — `HashMap<Device, _>` compiles. Used by
  `gpu_dispatch::gpu_backend()` registry and the autograd graph's
  per-device shadow.

## Architecture

### Variant taxonomy (`device.rs:13-32`)

```rust
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Device {
    #[default]
    Cpu,
    Cuda(usize),
    Xpu(usize),
    Mps(usize),
    Meta,
}
```

The variant set is open-ended in upstream (`c10::DeviceType` has 20+
entries including `XLA`, `HIP`, `Vulkan`, `Lazy`, `IPU`, …); ferrotorch
ships only the variants whose backend crates exist or are actively wired:
- `Cpu` — `ferray_core` provides the CPU `Vec<T>` storage.
- `Cuda` — `ferrotorch-gpu` provides `cudarc`-based kernels.
- `Xpu` — `ferrotorch-xpu` provides CubeCL/wgpu kernels (CL-452).
- `Mps` — `ferrotorch-mps` (in-progress, #451).
- `Meta` — pure metadata, no storage; `TensorStorage::Meta` carries shape
  but no data (CL-395).

Adding HIP/XLA/Vulkan/Lazy variants is straightforward (one variant + one
predicate + one Display arm), gated by the corresponding backend crate
existing first.

### Why `usize` for the ordinal (R-DEV-4 deviation)

Upstream uses `int8_t` for `DeviceIndex` (`c10/core/Device.h:19`) to fit
`Device` into a 16-bit hash key. ferrotorch uses `usize` because:
- Rust's idiomatic index type is `usize`; passing through a `usize` avoids
  every callsite needing a `as i8` cast.
- The hash impl is auto-derived, so the on-disk size doesn't matter.
- Negative ordinals (upstream's `-1 == "current device"`) are not
  representable; ferrotorch's "current device" is the active
  `gpu_backend()` registration, not a magic sentinel.

This is an R-DEV-4 deviation: upstream's pattern is a C++ footgun-mitigation
(`int8_t` was a packing optimization) that Rust doesn't need.

### Predicate methods (`device.rs:34-64`)

Five `is_*` predicates, one per variant. Each is `#[inline] fn(self) ->
bool { matches!(...) }`. Production consumers call these instead of
`match`-ing the full enum because the predicate makes the intent obvious:

```rust
if a.device().is_cuda() && b.device().is_cuda() {
    // GPU path
}
```

Direct callers (production): every op in `ferrotorch-core/src/grad_fns/*`
that branches CPU vs GPU. Concrete: `tensor.rs:1130` checks
`mask.device().is_cuda()` inside `masked_fill`.

### `Display` impl (`device.rs:66-76`)

```rust
match self {
    Device::Cpu => write!(f, "cpu"),
    Device::Cuda(id) => write!(f, "cuda:{id}"),
    Device::Xpu(id) => write!(f, "xpu:{id}"),
    Device::Mps(id) => write!(f, "mps:{id}"),
    Device::Meta => write!(f, "meta"),
}
```

Mirrors `c10::Device::str()` at `c10/core/Device.h:167`. The "cuda:N"
spelling is the format every log scraper, error message, and pickle/safe-
tensors metadata field expects. `FerrotorchError::DeviceMismatch` and
`FerrotorchError::DeviceUnavailable` rely on this Display in their `#[error]`
attributes (`error.rs:11, :39`).

### Production consumers

- `ferrotorch-core/src/storage.rs` — `TensorStorage::device(&self) ->
  Device` is the read accessor. The `TensorStorage` variants
  (`Cpu(Vec<T>)`, `Gpu(GpuBufferHandle)`, `Meta { shape, .. }`) each
  project to a `Device` value.
- `ferrotorch-core/src/tensor.rs` — `pub fn device(&self) -> Device`
  forwarding the storage's device. Every op uses this for the device-mismatch
  guard.
- `ferrotorch-core/src/bool_tensor.rs:152 / :158` — `BoolTensor::device`,
  `BoolTensor::is_cuda` follow the same pattern.
- `ferrotorch-core/src/int_tensor.rs:199 / :205` — `IntTensor::device`,
  `IntTensor::is_cuda`.
- `ferrotorch-core/src/dispatch.rs:73-75` — `DispatchKey::{Cpu, Cuda, Meta}`
  are the dispatch-key terminal-backend keys that correspond to `Device`
  variants. The mapping `Device -> DispatchKey` is the bridge between the
  tensor's runtime location and the dispatcher's keyset.
- `ferrotorch-core/src/error.rs:11, :39` — `DeviceMismatch { expected:
  Device, got: Device }` and `DeviceUnavailable` carry / surface device
  values in error messages.

## Parity contract

`parity_ops = []` — `Device` is an infrastructure type. The indirect
parity surface is every op's device-residency contract:
- A CPU tensor + CPU tensor → CPU result. (Verified by every op's CPU
  smoke test.)
- A CUDA tensor + CUDA tensor → CUDA result (no silent CPU detour).
  Anti-pattern-gate (`tooling/anti-pattern-gate.py`) blocks Edit that
  introduces `.cpu()`-then-`.cuda()` round-trip patterns.
- A CPU tensor + CUDA tensor → `FerrotorchError::DeviceMismatch`. Mirrors
  upstream's `RuntimeError: Expected all tensors to be on the same
  device, but found at least two devices, cpu and cuda:0!` at
  `aten/src/ATen/native/TensorIterator.cpp`.
- A Meta tensor + Meta tensor → Meta result with shape inference only.
  Mirrors upstream's `torch.device("meta")` semantics.

## Verification

```
cargo test -p ferrotorch-core --lib device
```

`device.rs` has no `#[cfg(test)] mod tests` block itself (it's small
enough that the Acceptance-Criteria checks are mechanical / derive-driven).
Functional verification lands in the consumer-file tests:

- `bool_tensor::tests::cpu_tensor_reports_cpu_device` at
  `bool_tensor.rs:737` — asserts `BoolTensor::ones(&[5]).device() ==
  Device::Cpu` and `!is_cuda()`.
- `int_tensor::tests::cpu_tensor_reports_cpu_device` at
  `int_tensor.rs:874` — same shape for `IntTensor`.
- Every GPU-feature-gated integration probe (e.g.
  `ferrotorch-core/tests/_probe_phase3c_masked.rs`) exercises
  `Device::Cuda(0)` end-to-end.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `Device::Cpu` variant at `ferrotorch-core/src/device.rs:15` with `#[default]`; predicate `is_cpu` at `:36`. Non-test production consumer: `ferrotorch-core/src/storage.rs` `TensorStorage::cpu(...).device() == Device::Cpu`; also `ferrotorch-core/src/bool_tensor.rs:152` returns `Device::Cpu` for any `TensorStorage::cpu(...)`-backed `BoolTensor`. |
| REQ-2 | SHIPPED | impl: `Device::Cuda(usize)` variant at `ferrotorch-core/src/device.rs:18`; predicate `is_cuda` at `:43`. R-DEV-4 deviation from upstream's `int8_t DeviceIndex` to `usize`. Non-test production consumer: `ferrotorch-core/src/int_tensor.rs:268-323` `IntTensor::to` matches `(Device::Cpu, Device::Cuda(_))` / `(Device::Cuda(_), Device::Cpu)` arms for H2D / D2H transfer; also `ferrotorch-core/src/tensor.rs::to` (the float-tensor mirror). |
| REQ-3 | SHIPPED | impl: `Device::Xpu(usize)` variant at `ferrotorch-core/src/device.rs:22`; predicate `is_xpu` at `:49`. Non-test production consumer: `ferrotorch-core/src/error.rs:259-265` `FerrotorchError::DeviceMismatch { expected, got }` and `ferrotorch-core/src/int_tensor.rs:336` (`IntTensor::to` errors when the destination is `Xpu` since the integer-on-XPU kernel set is not yet wired — the `Xpu` variant is the discriminator that drives the structured-error path). |
| REQ-4 | SHIPPED | impl: `Device::Mps(usize)` variant at `ferrotorch-core/src/device.rs:26`; predicate `is_mps` at `:55`. Same structured-error-discriminator role as Xpu; not yet wired to a backend (the `ferrotorch-mps` crate is the active prereq), so the consumer site is the `(from, to) => Err(InvalidArgument { ... unsupported })` arm at `bool_tensor.rs:261-266` that pattern-matches on `Mps(_)`. |
| REQ-5 | SHIPPED | impl: `Device::Meta` variant at `ferrotorch-core/src/device.rs:31`; predicate `is_meta` at `:61`. Non-test production consumer: `ferrotorch-core/src/storage.rs` `TensorStorage::Meta { shape, .. }` arm — `try_as_slice` returns `Err(GpuTensorNotAccessible)` for the Meta variant (mirrors upstream's "meta tensor has no data" semantics at `aten/src/ATen/`). |
| REQ-6 | SHIPPED | impl: `is_cpu` / `is_cuda` / `is_xpu` / `is_mps` / `is_meta` at `ferrotorch-core/src/device.rs:36-64`. Non-test production consumer: `ferrotorch-core/src/bool_tensor.rs:158` (`BoolTensor::is_cuda(&self) { self.device().is_cuda() }`), `ferrotorch-core/src/int_tensor.rs:205` (`IntTensor::is_cuda`), and every `if a.device().is_cuda() { ... GPU path }` branch across `grad_fns/*.rs`. |
| REQ-7 | SHIPPED | impl: `Display` impl at `ferrotorch-core/src/device.rs:66-76` matching `c10::Device::str()` at `c10/core/Device.h:167`. Non-test production consumer: `ferrotorch-core/src/error.rs:11` `#[error("device mismatch: expected {expected}, got {got}")]` — the error variant relies on `Display` for its formatted message; every device-mismatch propagation path passes through this Display. |
| REQ-8 | SHIPPED | impl: `#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]` at `ferrotorch-core/src/device.rs:12`. Non-test production consumer: `ferrotorch-core/src/gpu_dispatch.rs` (the `gpu_backend()` registry indirectly keys on device ordinal); also `Tensor<T>` operations `if self.device() == other.device()` comparison shape (PartialEq usage in `bool_tensor.rs:333`, `int_tensor.rs:436`, …). |
