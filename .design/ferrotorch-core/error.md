# FerrotorchError — workspace-wide error taxonomy

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - c10/util/Exception.h
-->

## Summary

`ferrotorch-core/src/error.rs` defines `FerrotorchError`, the single error
enum that every fallible op in the workspace returns. It mirrors PyTorch's
`c10::Error` / `TORCH_CHECK` taxonomy (`c10/util/Exception.h:31`), but as a
`#[non_exhaustive]` Rust enum so callers can pattern-match on the variant
without losing forward-compatibility when new variants land. The convenience
alias `FerrotorchResult<T> = Result<T, FerrotorchError>` is the workspace's
universal return type.

## Requirements

- REQ-1: A single `FerrotorchError` enum covering shape mismatch, device
  mismatch, dtype mismatch, index OOB, invalid arguments, lock poisoning,
  GPU/CUDA backend errors, and worker-thread panics — collapsing PyTorch's
  string-mode `RuntimeError` / `IndexError` / `TypeError` raises
  (`c10/util/Exception.h`'s `TORCH_CHECK_MSG` macro family) into typed
  variants. R-DEV-4: deviate from upstream's exception-throwing model and
  use Rust's `Result` instead.
- REQ-2: `Display` impls produce stable, machine-parseable error messages
  matching PyTorch's `TORCH_CHECK(cond, "shape mismatch: ...")` shape — the
  message prefix encodes the variant tag so log scrapers see the same
  vocabulary across ferrotorch and PyTorch.
- REQ-3: `FerrotorchError: Send + Sync + 'static` so async / multi-thread
  pipelines (`ferrotorch_data` worker pools, autograd's parallel-backward
  prototype) can propagate errors across thread boundaries.
- REQ-4: A `Gpu { source: Box<dyn Error + Send + Sync + 'static> }` variant
  for type-erased propagation of backend-specific errors
  (`ferrotorch_gpu::GpuError`, `cudarc::driver::DriverError`) — this lets
  `ferrotorch-core` cite GPU failures without taking a workspace dep cycle
  on `ferrotorch-gpu`. Callers downcast via `std::error::Error::source` +
  `Error::downcast_ref::<T>`.
- REQ-5: `FerrotorchResult<T>` alias for the `Result<T, FerrotorchError>`
  shape — every public ferrotorch fn returns this so call sites compose with
  `?` without a verbose error-type spelling.
- REQ-6: `NotImplementedOnCuda { op: &'static str }` variant — used where
  the CUDA path for an op is missing and the caller must `.cpu()` first.
  Anti-pattern-gate (`tooling/anti-pattern-gate.py`) treats a silent CPU↔GPU
  round trip as a bug; this variant is the structured surface that lets
  ops error cleanly rather than degrade silently.
- REQ-7: `Ferray(#[from] ferray_core::FerrayError)` automatic conversion —
  ferray (the typed-bytes layer ferrotorch sits on top of) returns its own
  error enum; the `#[from]` blanket impl lets `?` propagate ferray failures
  without manual wrapping. R-DEV-7: Rust ecosystem analog is materially
  better than upstream's monolithic c10::Error.

## Acceptance Criteria

- [x] AC-1: `FerrotorchError` is `#[non_exhaustive]` (verified at
  `error in error.rs`). New variants can be added without breaking pattern matches
  in downstream crates.
- [x] AC-2: `FerrotorchError: Send + Sync + 'static` — verified by the
  `Box<dyn Error + Send + Sync + 'static>` source bound at `error.rs:82` and
  by the fact that `FerrotorchResult<T>` is used across thread boundaries
  in `ferrotorch-core/src/cpu_pool.rs` and `ferrotorch_data`'s worker
  pools.
- [x] AC-3: `thiserror::Error` derives `std::error::Error` correctly — the
  `#[source]` attribute on the `Gpu` variant exposes the boxed inner error
  via `source()`. Verified by `gpu_variant_preserves_source_chain` at
  `error.rs:104`.
- [x] AC-4: `Display` output matches the documented schema for each variant
  — `gpu_variant_display` at `error.rs:117` asserts
  `outer.to_string() == "gpu error: test error: oom"`.
- [x] AC-5: The `Ferray` variant has automatic `From<FerrayError>` via
  `#[from]` at `error.rs:89`. `?` operator works across crate boundaries.
- [x] AC-6: No `unsafe` blocks. No `unwrap()` / `expect()` in production
  code paths. No `panic!()` (panics live only in test code).

## Architecture

### Variant taxonomy (`error.rs:5-89`)

The enum has 13 variants today, each carrying structured fields rather than
opaque strings. The categorization mirrors `c10/util/Exception.h:31-100`'s
`c10::Error` extras (`error_msg_` + `context_` + `backtrace_`):

- **Shape / structural errors** — `ShapeMismatch` (`error in error.rs`),
  `BackwardNonScalar` (`error in error.rs`), `IndexOutOfBounds` (`error in error.rs`).
  Mirrors `TORCH_CHECK(self.dim() == ...)` shape-validation sites scattered
  across `aten/src/ATen/native/*`.
- **Type / device errors** — `DtypeMismatch` (`error.rs:19`),
  `DeviceMismatch` (`error in error.rs`). Mirrors
  `TORCH_CHECK_TYPE(t.scalar_type() == kFloat, ...)` and the
  `c10::DeviceTypeName` mismatch checks at `c10/core/Device.h`.
- **Autograd-specific** — `NoGradFn` (`error in error.rs`) for leaf-tensor
  backward calls, mirroring upstream's
  `RuntimeError: element 0 of tensors does not require grad` at
  `torch/autograd/__init__.py`.
- **Backend / device-availability** — `DeviceUnavailable` (`error in error.rs`),
  `GpuTensorNotAccessible` (`error in error.rs`), `NotImplementedOnCuda`
  (`error in error.rs`), `Gpu { source }` (`error in error.rs`).
- **Concurrency** — `LockPoisoned` (`error in error.rs`) for `Mutex` poisoning
  (e.g. when an autograd hook panics inside a `lock()`).
- **External-crate passthrough** — `Ferray(#[from] FerrayError)`
  (`Ferray in error.rs`).
- **Catch-all** — `InvalidArgument { message: String }` (`error in error.rs`),
  `Internal { message: String }` (`error in error.rs`), `WorkerPanic { message }`
  (`error in error.rs`).

### `Gpu` variant — type-erased cross-crate error propagation

`ferrotorch-core` cannot depend on `ferrotorch-gpu` (workspace cycle).
`ferrotorch-gpu`'s `GpuError` enum is the natural type for kernel-launch
failures, cuBLAS errors, OOM, and so on. The `Gpu { source: Box<dyn
Error + Send + Sync + 'static> }` variant at `error.rs:75-83` resolves
this by boxing any `Error + Send + Sync + 'static` value. Callers
recover the concrete inner type with `source.downcast_ref::<GpuError>()`,
which is the standard Rust pattern (documented at `error.rs:46-69` with
a runnable code sample). Production consumer: every callsite in
`ferrotorch-core/src/grad_fns/*.rs` that returns
`FerrotorchResult<GpuBufferHandle>` from backend dispatch arms (e.g.
`grad_fns/arithmetic.rs:721, :755, :1474` `dispatch_floating_dtype!`
arms wrapping `GpuBackend::*` results — the underlying mapping happens in
`gpu_dispatch.rs`'s `?` propagation).

### `FerrotorchResult<T>` alias

`pub type FerrotorchResult<T> = Result<T, FerrotorchError>` at
`error.rs:93`. Used by every fallible public method in the workspace —
1336+ direct references across `ferrotorch-core/src/**/*.rs` (excluding
test files) per `grep -rn FerrotorchError:: ferrotorch-core/src/`.

### Production consumers

Every non-test `.rs` file under `ferrotorch-core/src/` that returns
`FerrotorchResult<T>` is a consumer. The most concentrated callsites are:
- `ferrotorch-core/src/tensor.rs:1855` —
  `ops::indexing::masked_select(self, mask)` returns
  `FerrotorchResult<Tensor<T>>` and propagates via `?` through the
  `Tensor::masked_select` boundary method.
- `ferrotorch-core/src/grad_fns/arithmetic.rs:721, :755, :1474` —
  `dispatch_floating_dtype!` arms return
  `FerrotorchResult<GpuBufferHandle>` and rely on `FerrotorchError::Gpu`
  for backend-error wrapping.
- `ferrotorch-core/src/methods.rs` — every `Tensor::*_t` boundary method
  returns `FerrotorchResult<Tensor<T>>` (one per op surface).

## Parity contract

This file declares `parity_ops = []` in `tooling/translate-routes.toml` —
it ships infrastructure, not numerical ops. The parity contract is
indirect: every op that returns `FerrotorchResult<T>` MUST emit an
`Err(FerrotorchError::*)` whose tag-prefixed `Display` output is the same
substring PyTorch would put into its `RuntimeError`'s `args[0]`. Cross-op
tests (e.g. `tests/conformance_elementwise.rs`) compare a sample of error
messages to upstream's; that's the negative-path parity surface.

## Verification

### Unit tests (passing)

- `gpu_variant_preserves_source_chain` at `error.rs:104` — verifies
  `std::error::Error::source().downcast_ref::<TestError>()` recovers the
  concrete inner type. Pins R-DEV-7 Rust-ecosystem-analog behavior.
- `gpu_variant_display` at `error.rs:117` — verifies the wrapped Display
  output is `"gpu error: test error: oom"`. Pins the schema in REQ-2.

### Cross-crate tests

Every `cargo test -p ferrotorch-core` test that constructs a tensor with a
mismatched shape exercises `FerrotorchError::ShapeMismatch`'s discriminator
and message format. Examples:
- `bool_tensor::tests::from_vec_shape_mismatch_errors` at `from_vec_shape_mismatch_errors in bool_tensor.rs`
- `int_tensor::tests::from_vec_shape_mismatch_errors` at `int_tensor.rs:800`
- `named_tensor::tests::named_tensor_rejects_length_mismatch` at
  `named_tensor.rs:213`
- `complex_tensor::tests::complex_reshape_size_mismatch_errors` at
  `complex_tensor.rs:586`

### Smoke gate

```
cargo test -p ferrotorch-core --lib error::tests
```

Expected: 2 tests pass, 0 failed. (The error.rs `#[cfg(test)] mod tests`
block has 2 tests; both are listed above.)

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: enum `FerrotorchError` at `Error in ferrotorch-core/src/error.rs` with 13 variants mirroring `c10/util/Exception.h:31` `c10::Error` taxonomy under R-DEV-4 (Rust `Result` deviation from C++ exceptions). Non-test production consumer: `masked_select in ferrotorch-core/src/tensor.rs` (`masked_select` returns `FerrotorchResult<Tensor<T>>` and propagates `ShapeMismatch` / `DeviceMismatch` via `?`); also `ferrotorch-core/src/grad_fns/arithmetic.rs` (`dispatch_floating_dtype!` arm returns the same `FerrotorchResult`). 1336+ direct uses across `ferrotorch-core/src/**/*.rs`. |
| REQ-2 | SHIPPED | impl: `#[error("...")]` attributes on every variant from `error.rs:7-89` produce stable Display output. Each variant's prefix encodes its tag (`"shape mismatch: "`, `"device mismatch: "`, `"gpu error: "`, …). Test: `gpu_variant_display` at `error.rs:117` pins the schema. Non-test production consumer: `ferrotorch-core/src/tensor.rs` — every Display invocation via `?`-propagated error formatting in the public API. |
| REQ-3 | SHIPPED | impl: the `Box<dyn Error + Send + Sync + 'static>` source bound at `error.rs:82` propagates `Send + Sync` through the `Gpu` variant; every other variant carries only `Send + Sync` fields. Non-test production consumer: `ferrotorch-core/src/cpu_pool.rs` (worker-pool threads return `FerrotorchResult<T>` across thread boundaries; the `Send + Sync` bound is what makes `JoinHandle<FerrotorchResult<T>>` viable). |
| REQ-4 | SHIPPED | impl: variant `Gpu { source: Box<dyn Error + Send + Sync + 'static> }` at `error in error.rs` with `#[source]` attribute exposing the inner error. Documented downcast pattern at `error in error.rs`. Test: `gpu_variant_preserves_source_chain` at `error in error.rs`. Non-test production consumer: `ferrotorch-core/src/gpu_dispatch.rs` (the `?` propagation path for `GpuBackend::*` results — every backend error is wrapped into `FerrotorchError::Gpu` before crossing the public API surface). |
| REQ-5 | SHIPPED | impl: `pub type FerrotorchResult<T> = Result<T, FerrotorchError>` at `error.rs:93`. Re-exported at `ferrotorch-core/src/lib.rs:145`. Non-test production consumer: every fallible public fn in the workspace; concrete cite `ferrotorch-core/src/tensor.rs:1839` (`pub fn masked_fill -> FerrotorchResult<Tensor<T>>`). |
| REQ-6 | SHIPPED | impl: `NotImplementedOnCuda { op: &'static str }` variant at `error in error.rs`. Non-test production consumer: `ferrotorch-core/src/dtype_dispatch.rs` (`dispatch_floating_dtype!` macro emits `FerrotorchError::NotImplementedOnCuda` for unsupported dtypes); also `cast in ferrotorch-core/src/int_tensor.rs` (`IntTensor::cast` errors cross-width casts on CUDA). |
| REQ-7 | SHIPPED | impl: `Ferray(#[from] ferray_core::FerrayError)` at `error in error.rs` — `#[from]` derives `From<FerrayError> for FerrotorchError`. Non-test production consumer: `ferrotorch-core/src/storage.rs` and `ferrotorch-core/src/tensor.rs` — wherever ferray returns its own error and `?` propagates it into a `FerrotorchResult`; concrete grep returns 30+ ferray callsites in the storage layer. |
