# Portable GPU op dispatch (portable_*)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/
  - c10/cuda/
-->

## Summary

`ferrotorch-cubecl/src/ops.rs` is the high-level op layer that takes
`Tensor<f32>` inputs, decides whether each is already device-resident
(via `cubecl_handle_of`) or needs a host-side upload, and dispatches
the corresponding kernel runner from `kernels.rs`. The boundary contract
is `(cubecl::server::Handle, Vec<usize>)` — device-resident result handle
plus shape — with NO host readback per ADR #663 item 4. This is the
ferrotorch analog of upstream's `aten/src/ATen/native/cuda/`
DispatchStub-registered entry points (e.g. `at::cuda::add_out`,
`at::cuda::matmul`); upstream uses DispatchStub + structured-kernels to
route, while ferrotorch's portability requirement (one source, three
backends via cubecl) drives the macro-dispatch design.

## Requirements

- REQ-1: Per-op `pub fn portable_<op>(args, &CubeRuntime) ->
  FerrotorchResult<(cubecl::server::Handle, Vec<usize>)>`. The
  return type ALWAYS includes the shape so the caller can rebuild a
  `Tensor` without re-deriving the shape from inputs. Mirrors how
  PyTorch CUDA ops return a `Tensor` whose shape is implicit; ferrotorch
  surfaces the shape explicitly because the device-resident handle
  doesn't carry one.

- REQ-2: Handle-direct vs slice-upload dispatch — if every input has a
  `CubeclStorageHandle` (already on the GPU), use the
  `kernels::run_*_handle` path (no H2D upload). Otherwise call
  `contiguous_data` to materialise a host `Vec<f32>` from each input
  and use `kernels::run_*` (slice-upload). Implemented uniformly via
  three macros: `dispatch_binary!`, `dispatch_unary!`,
  `dispatch_matmul!`.

- REQ-3: Per-backend client-arm dispatch — every dispatch macro arm
  matches on `CubeClient::{Cuda,Wgpu,Rocm}` (cfg-gated) to invoke the
  kernel runner with the correctly-typed `ComputeClient<R>`. The
  `CubeClient::Stub` arm is `unreachable!()` per #1083 (Stub never
  reaches kernel dispatch).

- REQ-4: Shape validation BEFORE dispatch — elementwise binary ops
  call `check_same_shape(a, b)?` first; matmul calls
  `check_matmul_shapes(a, b)?` returning `(m, k, n)` after validating
  2-D + inner-dim agreement. Mirrors upstream's
  `TORCH_CHECK(a.dim() == 2, ...)` discipline at
  `aten/src/ATen/native/cuda/Matmul.cu` entry.

- REQ-5: No-backend stubs — every `pub fn portable_*` has a paired
  `#[cfg(not(any(feature = "wgpu", feature = "cuda", feature =
  "rocm")))]` arm returning `Err(FerrotorchError::DeviceUnavailable)`
  AFTER running shape validation. Validation runs even without a
  backend so callers still get a useful error for malformed inputs.

- REQ-6: Macro-generated elementwise ops — `define_portable_unary!`
  and `define_portable_binary!` stamp out 10 unary + 1 extra binary
  (`div`) ops. Each is `pub fn portable_<op>(args, &CubeRuntime) ->
  FerrotorchResult<(Handle, Vec<usize>)>`. The 4 binary ops `add`,
  `sub`, `mul`, `matmul` are written out longhand because they have
  per-op specifics (matmul takes `(m, k, n)`).

- REQ-7: Macro-generated polynomial ops — `define_portable_polynomial!`
  stamps out 8 polynomial families (Chebyshev T/U/V/W, Hermite H/He,
  Laguerre L, Legendre P). Each takes an additional `n: usize` degree
  argument, validates `u32::try_from(n)` (returning
  `FerrotorchError::InvalidArgument` on overflow), and dispatches via
  `dispatch_unary_with_n!`.

- REQ-8: GPU readback is the CALLER's responsibility — `portable_*`
  returns the device handle; the caller decides when to read back via
  `CubeRuntime::read_f32s`. The `cfg(test)` `readback` helper exists
  only for the integration tests; production callers in
  `ferrotorch-xpu` thread the handle through `wrap_kernel_output` into
  a `CubeclStorageHandle` and return a device-resident
  `Tensor::from_storage`.

## Acceptance Criteria

- [x] AC-1: All 15 elementwise + 8 polynomial + 1 matmul `portable_*`
  functions exist as documented (`portable_add/sub/mul/div`,
  `portable_neg/abs/exp/ln/sqrt/sin/cos/tanh/sigmoid/relu`,
  `portable_chebyshev_polynomial_t/u/v/w`,
  `portable_hermite_polynomial_h/he`, `portable_laguerre_polynomial_l`,
  `portable_legendre_polynomial_p`, `portable_matmul`).
- [x] AC-2: Each op returns shape `(handle, Vec<usize>)` matching the
  input shape (for elementwise) or `[m, n]` (for matmul).
- [x] AC-3: Shape mismatch returns `FerrotorchError::ShapeMismatch`
  (verified by `portable_add_rejects_shape_mismatch`,
  `portable_matmul_rejects_rank_mismatch`,
  `portable_matmul_rejects_inner_dim_mismatch`).
- [x] AC-4: No-backend build yields `DeviceUnavailable` after
  validation (verified by
  `no_backend_tests::runtime_construction_errors_without_backend`).
- [x] AC-5: Polynomial degree overflow returns
  `FerrotorchError::InvalidArgument` — `n: usize` exceeding `u32` range.

## Architecture

### Per-op entry point shape (REQ-1)

Every `pub fn portable_<op>` returns `FerrotorchResult
<(cubecl::server::Handle, Vec<usize>)>`. The `Vec<usize>` is the output
shape, derived from the input(s): for elementwise it's `a.shape()
.to_vec()`; for matmul it's `vec![m, n]`. The handle is device-resident;
no readback happens inside this file (ADR #663 item 4).

Non-test production consumer: `ferrotorch-xpu/src/lib.rs` —
the `xpu_binary!`, `xpu_unary!`, `xpu_polynomial!` macro expansions
each invoke `$cubecl(args, xpu.runtime())?`, where `$cubecl` is the
matching `ferrotorch_cubecl::ops::portable_<op>` path.

### Dispatch macros (REQ-2, REQ-3)

`macro_rules! dispatch_binary in ops.rs` is the structural heart of the
file. The macro expands to:

```text
match (cubecl_handle_of($a), cubecl_handle_of($b)) {
    (Some(ha), Some(hb)) => {
        // handle-direct path: no H2D upload
        match $rt.client() {
            CubeClient::Wgpu(c) => $launcher_handle(c, ha_handle, hb_handle, n),
            CubeClient::Cuda(c) => $launcher_handle(c, ha_handle, hb_handle, n),
            CubeClient::Rocm(c) => $launcher_handle(c, ha_handle, hb_handle, n),
            CubeClient::Stub    => unreachable!(),
        }
    }
    _ => {
        // slice-upload fallback for CPU inputs
        let a_data = contiguous_data($a)?;
        let b_data = contiguous_data($b)?;
        match $rt.client() { ... $launcher(c, &a_data, &b_data) ... }
    }
}
```

`dispatch_unary!` is the same with one input; `dispatch_matmul!` adds
`m, k, n` scalars; `dispatch_unary_with_n!` is the polynomial variant
adding a degree scalar.

The structural choice — match on input device-residency FIRST, then on
backend variant — keeps the no-H2D-round-trip optimisation (#673) at
the boundary where it belongs. Inputs that are already on the GPU stay
there; inputs that aren't get uploaded once.

### Shape validation (REQ-4)

`fn check_same_shape in ops.rs` is the elementwise shape gate. It
returns `Err(FerrotorchError::ShapeMismatch { message: ... })` with the
two shapes in the error string. Called at the top of every binary op.

`fn check_matmul_shapes in ops.rs` validates 2-D + `a.shape()[1] ==
b.shape()[0]` and returns `(m, k, n)` for the kernel.

The shape checks run BEFORE the cfg-gate, so even in a no-backend build
calls to `portable_add(a_2elem, b_3elem, &rt)` return a shape-mismatch
error rather than a `DeviceUnavailable`.

### Macro-stamped ops (REQ-6, REQ-7)

`macro_rules! define_portable_unary in ops.rs` invocations:

- `portable_neg, portable_abs, portable_exp, portable_ln, portable_sqrt,
  portable_sin, portable_cos, portable_tanh, portable_sigmoid` — 9
  invocations. (`portable_relu` is written longhand for legacy reasons
  but is structurally identical.)

`macro_rules! define_portable_binary in ops.rs` invocations:

- `portable_div` — 1 invocation. (`add/sub/mul/matmul` longhand for
  reasons noted in REQ-6.)

`macro_rules! define_portable_polynomial in ops.rs` invocations:

- 8 — one per polynomial family. Each takes an extra `n: usize` arg,
  validates `u32::try_from(n)`, and dispatches via
  `dispatch_unary_with_n!`.

Each macro stamps BOTH the feature-on `pub fn` (real dispatch) AND the
feature-off `pub fn` (returns `DeviceUnavailable`) for the matching op
under the appropriate `#[cfg(...)]`.

### Backend-feature gate (REQ-5)

Every `pub fn portable_<op>` has two variants:

```rust
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn portable_<op>(...) -> FerrotorchResult<...> { /* real dispatch */ }

#[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "rocm")))]
pub fn portable_<op>(...) -> FerrotorchResult<...> { /* validate + error */ }
```

This keeps the type signature stable across builds — downstream code
compiles either way; only the runtime behaviour changes. The
`no_backend_tests::runtime_construction_errors_without_backend` test
exercises this path.

### Stub-arm safety (REQ-3)

Every dispatch macro arm includes a `CubeClient::Stub => unreachable!()`
branch with a multi-line message: "test stub should not reach kernel
dispatch — shape check or signature pin should fire first (#1083)".
This is documentation-as-runtime-check: Stub runtimes are reserved for
tests that exercise the shape-validation code path; if a test
accidentally lets a Stub-runtime call reach kernel dispatch, the panic
identifies the test-discipline bug.

`R-CODE-1` permits `unreachable!()` outside leaf primitives only when
the impossibility is enforced by an upstream invariant; the doc comment
on `CubeClient::Stub` in `runtime.rs` (`pub enum CubeClient`) establishes the invariant
that `CubeRuntime::new` and `auto` never produce Stub.

## Parity contract

ferrotorch-cubecl is INFRASTRUCTURE — `parity_ops = []`. Per-op
correctness lives in `.design/ferrotorch-core/ops/*.md` for the
matching CPU op. The integration suite that pins GPU-vs-CPU agreement
runs through `ferrotorch-xpu`'s tests on a wgpu adapter; this file
contributes the GPU dispatch side of those tests.

Edge cases handled by this file directly:

- **Shape mismatch**: elementwise → `ShapeMismatch` error. Matmul →
  rank-mismatch ShapeMismatch + inner-dim-mismatch ShapeMismatch.
- **Empty tensor**: empty shapes pass `check_same_shape`. The
  downstream `dispatch_binary!` extracts `n = ha.len() = 0`; the kernel
  runner allocates a zero-byte output handle and dispatches with
  `count = 1, dim = 256`; the kernel's `ABSOLUTE_POS < out.len()` guard
  ensures no out-of-bounds writes.
- **Non-contiguous input**: `contiguous_data` calls `t.data_vec()` to
  materialise a contiguous copy when `t.data()` returns Err. The
  ferrotorch-xpu path goes through handle-direct dispatch instead,
  bypassing the issue.
- **Polynomial overflow**: `u32::try_from(n)` fails for `n > u32::MAX`;
  returns `FerrotorchError::InvalidArgument` with the offending value.

## Verification

Tests in `#[cfg(all(test, feature = "wgpu"))] mod tests in ops.rs`:

- `portable_add_runs_on_gpu` — small-shape sanity.
- `portable_sub/mul/div/relu/neg/abs/exp/ln/sqrt/sin/cos/tanh/sigmoid
  _runs_on_gpu` — 13 small-shape sanity tests for the
  elementwise family.
- `portable_add_large_shape` — 1024 elements to exercise multi-cube
  launch geometry. f32 bit-equality (no epsilon) because both sides are
  a single non-fused IEEE-754 add.
- `portable_matmul_runs_on_gpu` / `portable_matmul_square_8x8` —
  matmul correctness on small + identity-multiply cases.
- `portable_matmul_rejects_rank_mismatch` /
  `portable_matmul_rejects_inner_dim_mismatch` /
  `portable_add_rejects_shape_mismatch` /
  `portable_div_rejects_shape_mismatch` — error-path coverage.
- `portable_exp_then_ln_is_identity` — round-trip sanity.
- `portable_sigmoid_large_shape` — 1024-element sigmoid for cube-count
  coverage + numerical stability check.
- `portable_chebyshev_t/u/v/w_runs_on_gpu`,
  `portable_hermite_h/he_runs_on_gpu`,
  `portable_laguerre_l_runs_on_gpu`,
  `portable_legendre_p_runs_on_gpu` — one per polynomial family
  against a hand-computed expected vector.
- `portable_chebyshev_t_n0_returns_ones`,
  `portable_chebyshev_t_n1_returns_x` — degree-0 / degree-1 base-case
  coverage.
- `portable_polynomial_handles_large_input` — 1024 elems through
  Chebyshev T_5 with `|T_5(x)| <= 1` invariant check.

`#[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "rocm")))]
mod no_backend_tests`:

- `runtime_construction_errors_without_backend` — pins the
  no-backend error path.

Smoke command (`parity_ops = []`):

```bash
cargo test -p ferrotorch-cubecl --no-default-features 2>&1 | tail -3
```

Expected: ≥ 1 `test result: ok`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: every `pub fn portable_<op> in ops.rs` returns `FerrotorchResult<(cubecl::server::Handle, Vec<usize>)>`. Non-test consumer: `ferrotorch-xpu/src/lib.rs` invokes `$cubecl(args, xpu.runtime())?` (where `$cubecl` is a `portable_*` path) and destructures `(handle, shape)`. |
| REQ-2 | SHIPPED | impl: `macro_rules! dispatch_binary/dispatch_unary/dispatch_matmul/dispatch_unary_with_n in ops.rs` route handle-direct vs slice-upload via `match cubecl_handle_of(...)`. Non-test consumer: every `portable_<op>` in this file invokes one of the four macros; downstream `ferrotorch-xpu` calls those ops. |
| REQ-3 | SHIPPED | impl: each dispatch macro's inner match on `CubeClient::{Wgpu,Cuda,Rocm}` with `Stub => unreachable!()` arm. Non-test consumer: same as REQ-2 — every op routes through these macros, which route through the backend variant. |
| REQ-4 | SHIPPED | impl: `fn check_same_shape in ops.rs` and `fn check_matmul_shapes in ops.rs`. Non-test consumer: every `portable_<binary>` and `portable_matmul` invokes one before dispatch (lines 229, 251, 273, 319, 394). |
| REQ-5 | SHIPPED | impl: per-op `#[cfg(not(...))]` arms returning `Err(FerrotorchError::DeviceUnavailable)` after shape validation. Non-test consumer: `ferrotorch-cubecl/src/ops.rs::no_backend_tests::runtime_construction_errors_without_backend` exercises the path under `--no-default-features`. |
| REQ-6 | SHIPPED | impl: `macro_rules! define_portable_unary/define_portable_binary in ops.rs` invoked for div + 9 unary ops. Non-test consumer: each generated `portable_<op>` is called from `ferrotorch-xpu/src/lib.rs::xpu_*` macro expansions. |
| REQ-7 | SHIPPED | impl: `macro_rules! define_portable_polynomial in ops.rs` invoked 8 times. Non-test consumer: `ferrotorch-xpu/src/lib.rs::xpu_polynomial!` macro expansions invoke `ferrotorch_cubecl::ops::portable_*_polynomial_*`. |
| REQ-8 | SHIPPED | impl: `portable_*` returns the raw `Handle` + shape; the file's `tests` module uses a local `fn readback` helper, but production callers do NOT read back here. Non-test consumer: `ferrotorch-xpu/src/lib.rs` calls `wrap_kernel_output(handle, &shape, ..., xpu.ordinal())` — handle stays device-resident; readback only happens later via `Tensor::cpu()`. |
