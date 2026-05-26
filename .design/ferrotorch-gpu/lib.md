# ferrotorch-gpu — crate root (lib.rs)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2fa9c68b1 (working tree at /home/doll/pytorch)
upstream-paths:
  - aten/src/ATen/cuda/
  - aten/src/ATen/native/cuda/
  - c10/cuda/
  - torch/cuda/
-->

## Summary

`ferrotorch-gpu/src/lib.rs` is the crate root: it declares the lint baseline,
publishes module declarations behind cfg-gated `#[cfg(feature = "cuda")]`
arms, and re-exports the user-facing public API surface (`GpuDevice`,
`GpuError`, `CudaBuffer`, `cpu_to_gpu`, `gpu_add`, the BLAS shims, the
graph capture suite, the memory-guard suite, the pool helpers, and the
tensor-bridge). This is the ferrotorch analog of the umbrella header
behaviour PyTorch gets from `c10/cuda/CUDAStream.h` +
`c10/cuda/CUDAFunctions.h` + `c10/cuda/CUDACachingAllocator.h` being
visible as a unit to anyone who includes `<ATen/cuda/CUDAContext.h>`.

## Requirements

- REQ-1: Lint baseline. `#![warn(clippy::all, clippy::pedantic)]` and
  `#![deny(rust_2018_idioms)]` set the crate-wide ceiling; `#![deny(missing_docs)]`
  enforces rustdoc coverage. Targeted `#![allow]` entries each carry an
  in-source justification — none silence the production-code anti-pattern
  set in goal.md (no `Arc<Mutex<T>>` shield, no `panic!` allow, no
  `unsafe_code` allow). `unsafe_code` is NOT denied because leaf primitives
  (PTX launches, raw pointer slices, FFI to cudarc) need per-block
  `SAFETY:` annotation rather than crate-wide denial — this is the
  documented R-CODE-1 leaf-primitive carve-out.

- REQ-2: Feature flag matrix. The `cuda` feature is default-on; the
  crate compiles with `--no-default-features` to support host-only smoke
  tests and CI runs. Every cuda-only module declaration carries
  `#[cfg(feature = "cuda")]`; every cuda-only re-export carries the
  same. The `cusparselt` feature is a sub-feature gating just the
  cuSPARSELt 2:4 sparse path.

- REQ-3: Module taxonomy. Modules are declared in a flat namespace
  matching the file taxonomy: `allocator`, `backend_impl`, `bf16`,
  `blas`, `buffer`, `cast_kernels`, `conv`, `cufft`, `cusolver`,
  `cusparselt`, `device`, `error`, `f16`, `flash_attention`,
  `gather_int`, `graph`, `group_norm`, `int_kernels`, `kernels`,
  `masked_kernels`, `memory_guard`, `module_cache`, `pool`, `reduce_arg`,
  `rng`, `roll`, `sparse`, `stream`, `tensor_bridge`, `transfer`,
  `upsample`. Each is owned by a single source file.

- REQ-4: Ergonomic re-exports. Top-level types every downstream caller
  needs are re-exported from the crate root: `GpuDevice`, `GpuError`,
  `GpuResult`, `CudaBuffer`, `CudaAllocator`, `MemoryGuard` /
  `MemoryGuardBuilder` family, `CapturedGraph` / `CapturePool` /
  `begin_capture` / `end_capture` graph family, `GpuTensor`, `GpuFloat`,
  `cpu_to_gpu` / `gpu_to_cpu`, the kernel families (`gpu_add`,
  `gpu_matmul_f32`, `gpu_softmax`, etc.). This is the boundary API
  surface — downstream crates import from `ferrotorch_gpu::` rather
  than `ferrotorch_gpu::device::GpuDevice`.

- REQ-5: Compile-time symbol-resolution probe for the cuSPARSE feature
  wiring. The `cusparse_smoke` test module compile-checks that
  `cudarc::cusparse::sys::cusparseHandle_t` resolves with the
  workspace feature set, preventing a silent feature-gating
  regression that would otherwise show up at first SpMM call.

## Acceptance Criteria

- [x] AC-1: `cargo build -p ferrotorch-gpu --no-default-features`
  succeeds (host-only stub build).
- [x] AC-2: `cargo build -p ferrotorch-gpu --features cuda` succeeds
  on a system with the CUDA toolchain.
- [x] AC-3: `cargo clippy -p ferrotorch-gpu --all-features` produces
  no new warnings beyond the documented per-item `#[allow]` set.
- [x] AC-4: Every cuda-only module declaration is gated with
  `#[cfg(feature = "cuda")]` (audit by grep on the `pub mod` lines
  in `lib.rs`).
- [x] AC-5: `cusparse_handle_type_resolves` test passes when the
  workspace's `cudarc` feature set includes `cusparse`.

## Architecture

### Lint baseline (REQ-1)

The crate uses `#![warn(clippy::all, clippy::pedantic)]` plus
`#![deny(rust_2018_idioms)]` and `#![deny(missing_docs)]`. Every
crate-root `#![allow]` is paired with an inline comment explaining the
rationale (module-name repetition, missing-errors-doc backlog, math-kernel
single-char names, PTX-template unreadability, etc.). None of the
allows silence the production-code anti-pattern set goal.md R-CODE-*
forbids. The lint header lives at lines 1–113 of `lib.rs`.

`unsafe_code` is NOT denied — `mod bf16 in lib.rs`, `mod blas in lib.rs`,
`mod stream in lib.rs`, `mod graph in lib.rs`, and the various kernel
modules all contain leaf-primitive `unsafe` blocks with per-block
`SAFETY:` comments. The crate-root preamble documents this explicitly.

### Feature flag matrix (REQ-2)

The default-on `cuda` feature wires cudarc into the build. The flat
list of `#[cfg(feature = "cuda")] pub mod ...` lines at `lib.rs`
holds the cuda-specific modules. Stubs in each module's tail provide a
build-only surface when the feature is off so downstream crates that
unconditionally `use ferrotorch_gpu::GpuDevice` still compile.
`cusparselt` is sub-gated for the sparse 2:4 matmul path —
`#[cfg(all(feature = "cuda", feature = "cusparselt"))] pub mod cusparselt`
at `lib.rs`.

### Module taxonomy (REQ-3)

Each `pub mod X` declaration at `lib.rs` corresponds to one
source file under `ferrotorch-gpu/src/`. This is a flat layout mirroring
PyTorch's `c10/cuda/` directory shape. No deeply-nested module
hierarchies; each file is its own translation unit and routes 1:1 in
`tooling/translate-routes.toml`.

### Ergonomic re-exports (REQ-4)

The re-export block at `lib.rs` is the published public API.
Downstream crates (`ferrotorch-llama/src/gpu.rs`,
`ferrotorch-diffusion/src/gpu/unet.rs`, `ferrotorch-jit/src/fusion_gpu.rs`)
import from the crate root by convention — e.g.
`use ferrotorch_gpu::{CudaBuffer, GpuDevice, GpuError, gpu_bmm_f32, gpu_matmul_f32}`
(`ferrotorch-diffusion/src/gpu/clip.rs`). The flat re-export shape
is the stable contract; intra-crate moves of the underlying definitions
must preserve the names the crate root re-publishes.

Non-test production consumers of the re-export:
- `ferrotorch-llama/src/gpu.rs` — `use ferrotorch_gpu::{ ... }`.
- `ferrotorch-diffusion/src/gpu/unet.rs`, `pipeline.rs`,
  `vae.rs`, `vae_encoder.rs`, `clip.rs`.
- `ferrotorch-jit/src/fusion_gpu.rs`.
- `ferrotorch-distributed/src/gpu_collective.rs`.
- `ferrotorch-nn/src/utils.rs` (`init_cuda_backend`).
- The meta-crate `ferrotorch/src/lib.rs` does
  `pub use ferrotorch_gpu::*;` so application code sees the same surface.

### cuSPARSE symbol-resolution probe (REQ-5)

`mod cusparse_smoke in lib.rs` at `lib.rs` runs at
`cargo test`-time only. It binds a null `cudarc::cusparse::sys::cusparseHandle_t`
to confirm the typedef resolves with the workspace's cudarc feature set.
A regression on the `cudarc/cusparse` feature toggle would otherwise
silently break the SpMM path; this compile-only test catches it before
the first real SpMM call.

## Parity contract

`parity_ops = []`. The crate root is INFRASTRUCTURE — it has no parity
op of its own. The downstream consumer's parity-sweep ops verify that the
re-exported kernel functions are wired correctly; that contract is
enforced per-kernel (`add`, `mul`, `matmul`, etc.) by each kernel
module's own design doc.

Edge cases handled at the crate-root level:
- Disabled `cuda` feature: every cuda-only re-export disappears from
  the public API surface; a downstream `use ferrotorch_gpu::gpu_add_bf16`
  with `--no-default-features` produces a clean "no such symbol" compile
  error rather than a runtime CUDA error.
- Missing `cusparselt` feature: the `cusparselt` re-export disappears;
  callers of the structured-sparse matmul path get a clean compile error.

## Verification

Tests:
- `mod cusparse_smoke::cusparse_handle_type_resolves` at `lib.rs` —
  compile-time symbol resolution probe.
- The crate-level gauntlet (`cargo test -p ferrotorch-gpu --features cuda`,
  ~700 tests including each module's own unit tests) exercises every
  re-export by construction; an import that fails to resolve breaks the
  build before any test runs.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda --lib 2>&1 | tail -3
cargo build -p ferrotorch-gpu --no-default-features 2>&1 | tail -3
cargo clippy -p ferrotorch-gpu --all-features --all-targets -- -D warnings 2>&1 | tail -3
```

Expected: each command's tail prints `Finished` / `0 failed` / `Compiling`
with no `error:` / `warning:` lines beyond the documented allows.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: lint baseline at `lib.rs`, every `#![allow]` carries inline justification, `unsafe_code` carve-out documented. Non-test production consumer: every `ferrotorch-gpu/src/*.rs` module compiles under this baseline (e.g. `bf16.rs`, `blas.rs`, `stream.rs` all use per-block `SAFETY:` annotations the carve-out enables). |
| REQ-2 | SHIPPED | impl: cfg-gated `pub mod` declarations at `lib.rs` and matching cfg-gated `pub use` re-exports at `lib.rs`. Non-test production consumer: `ferrotorch-distributions/src/fallback.rs` calls `ferrotorch_gpu::init_cuda_backend()` under the host-only feature configuration; works without panicking. |
| REQ-3 | SHIPPED | impl: 31 `pub mod` lines at `lib.rs`, each one-file-per-module. Non-test production consumer: `tooling/translate-routes.toml` enumerates one route per `ferrotorch-gpu/src/*.rs` file (33 routes), matching the module declarations. |
| REQ-4 | SHIPPED | impl: re-exports at `lib.rs` (`pub use device::GpuDevice`, `pub use error::{GpuError, GpuResult}`, `pub use buffer::CudaBuffer`, `pub use graph::{CapturedGraph, ...}`, `pub use memory_guard::{MemoryGuard, ...}`). Non-test production consumer: `ferrotorch-diffusion/src/gpu/clip.rs` — `use ferrotorch_gpu::{CudaBuffer, GpuDevice, GpuError, gpu_bmm_f32, gpu_layernorm, gpu_matmul_f32, gpu_softmax};` imports directly off the crate root. |
| REQ-5 | SHIPPED | impl: `mod cusparse_smoke in lib.rs` at `lib.rs`, test `cusparse_handle_type_resolves` binds `cudarc::cusparse::sys::cusparseHandle_t = std::ptr::null_mut()`. Non-test production consumer: `ferrotorch-gpu/src/sparse.rs` (the actual cuSPARSE SpMM caller, gated on the same workspace feature) requires the symbol to resolve at compile time; the smoke test is a regression floor for that build. |
