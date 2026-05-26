# ferrotorch-jit — `nvrtc` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - aten/src/ATen/native/cuda/jit_utils.cpp
  - aten/src/ATen/native/cuda/jit_utils.h
  - torch/utils/cpp_extension.py
-->

## Summary

`ferrotorch-jit/src/nvrtc.rs` is the shared NVRTC compile helper.
It compiles CUDA C source to PTX via cudarc's `nvrtc::compile_ptx_with_opts`,
with two preprocessing steps that the rest of the crate's GPU path
needs: (a) strip `#include <math.h>` (NVRTC has no host-header
include path), (b) rewrite `__global__ void <name>` to `extern "C"
__global__ void <name>` so cudarc's `cuModuleGetFunction` can find
the unmangled symbol. Mirrors the role of
`aten/src/ATen/native/cuda/jit_utils.cpp`'s `jit_pwise_function` and
the `torch.utils.cpp_extension.load_inline(..., is_cuda=True)`
machinery, which similarly invoke NVRTC for runtime-generated CUDA
sources.

## Requirements

- REQ-1: `pub fn compile_cuda_source_to_ptx(cuda_source: &str,
  kernel_name: &str) -> Result<String, JitError>` compiles a CUDA C
  source string to a PTX module string. Returns
  `JitError::CodegenError` on NVRTC failure, with both
  `kernel_name` and the NVRTC compile log in the message.
- REQ-2: Pre-processing strips lines that begin with
  `#include <math.h>` — NVRTC rejects these (no host include
  path). Device-math symbols (`__nv_exp`, `__nv_log`, ...) resolve
  via libdevice automatically.
- REQ-3: Pre-processing rewrites `__global__ void <name>(...)` to
  `extern "C" __global__ void <name>(...)` so the symbol stays
  unmangled — cudarc's `cuModuleGetFunction` keys on the unmangled
  name.
- REQ-4: Compile options pin `arch = "compute_75"` (the floor for
  non-deprecated NVRTC targets in CUDA 13.x; supports f64 hardware
  ops on Turing-and-newer GPUs) and explicitly do NOT enable
  `--use_fast_math` (which would sacrifice f64 precision in
  libdevice's polynomial expansions).
- REQ-5: A `#[cfg(not(feature = "cuda"))]` stub keeps the module's
  exported surface uniform across feature combinations. Callers that
  hit the stub receive `JitError::CodegenError` citing the missing
  `cuda` feature.

## Acceptance Criteria

- [x] AC-1: With `--features cuda` enabled, compiling a trivial
  `extern "C" __global__ void k(float* x) { x[0] = 1.0f; }` source
  returns `Ok(ptx)` where `ptx` starts with `// PTX` or `.version`
  line characteristic of PTX.
- [x] AC-2: Source containing `#include <math.h>` compiles
  successfully (the line is stripped before NVRTC sees it).
- [x] AC-3: Source containing `__global__ void <name>` is
  preprocessed so the produced PTX exports `<name>` unmangled.
- [x] AC-4: Without `--features cuda`, the stub returns
  `Err(JitError::CodegenError)` whose message contains "requires
  the `cuda` feature".
- [x] AC-5: Malformed CUDA source returns `Err(JitError::CodegenError)`
  with the NVRTC compile log embedded.

## Architecture

The function takes a CUDA C source string. It pipes the source
through two stripping/rewriting passes:

1. Filter out lines whose trimmed start equals `#include
   <math.h>`. NVRTC has no host headers in its include path and
   the line causes a hard failure. Device-math intrinsics are
   resolved by libdevice, which NVRTC links automatically when the
   source uses f64 math ops.
2. Rewrite each `__global__ void <name>(...)` line to `extern "C"
   __global__ void <name>(...)`. Without this rewrite, the C++
   mangler turns symbol `k_f64_exp(double*, double*, int)` into
   `_Z9k_f64_expPKdPdi`; cudarc's loader keys on `k_f64_exp` and
   fails to find the function.

Then `compile_ptx_with_opts(&nvrtc_source, opts)` is invoked with
the documented `CompileOptions { arch: Some("compute_75"),
..Default::default() }`. The architecture pin is the deliberate
floor for CUDA 13.x — Volta (sm_70) emits a deprecation warning,
and sm_75 (Turing) is the first universally-supported target with
f64 FMA hardware (libdevice's f64 transcendentals depend on f64
FMA).

The `#[cfg(not(feature = "cuda"))]` stub matches the public
signature so downstream callers can `cfg`-gate at the dispatch
site (e.g. `codegen_gpu.rs:1237`) without conditional re-imports.

### Non-test production consumers

- `ferrotorch-jit/src/codegen_gpu.rs:1237` —
  `crate::nvrtc::compile_cuda_source_to_ptx(&cuda_source, fn_name)`
  is the f64-transcendental PTX path (#748 / #749). The codegen
  emits a CUDA C source using libdevice-resolved math intrinsics
  and routes the source through NVRTC.
- `ferrotorch-jit/src/fusion_gpu.rs:197` —
  `crate::nvrtc::compile_cuda_source_to_ptx(&cuda_source,
  FUSED_F64_KERNEL_NAME)` is the f64-fusion path. The PTX is
  loaded by cudarc and dispatched at apply time
  (`fusion.rs:522`).

## Parity contract

`parity_ops = []`. NVRTC is build-machinery; correctness is "the
emitted PTX runs and produces the expected numerical output". The
parity gauntlet for the ops that use this path (the transcendental
f64 ops on PTX) is enforced by the parity-sweep for those ops
(`exp`, `log`, `tanh`, `sigmoid`, `gelu`, `silu`, `pow` —
covered by `codegen_gpu.md` and the upstream ops' parity audits).

Edge cases:

- **`--use_fast_math` disabled** — explicitly NOT enabled because
  the libdevice polynomial expansions for f64 transcendentals are
  the IEEE-correct path; fast-math swaps them for approximate
  intrinsics.
- **`#[cfg(not(feature = "cuda"))]` stub** — surfaces a clear
  error rather than silently no-oping.

## Verification

Tests for this module live inside the CUDA-feature-gated paths in
`codegen_gpu.rs` and `fusion_gpu.rs`, since the function is only
meaningful with the runtime present. The non-CUDA stub is
exercised by every default-feature `cargo test -p ferrotorch-jit`
that goes through `codegen_gpu.rs:1237` and gets the explicit
`JitError::CodegenError` back.

Smoke command:

```bash
cargo test -p ferrotorch-jit --lib nvrtc:: 2>&1 | tail -3
# under CUDA:
cargo test -p ferrotorch-jit --features cuda --lib 2>&1 | tail -3
```

Expected: default feature set passes; under `--features cuda`, the
f64-transcendental and fused-f64 GPU paths exercise NVRTC.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn compile_cuda_source_to_ptx` in `nvrtc.rs` (both `#[cfg(feature = "cuda")]` and the stub); non-test consumer: `ferrotorch-jit/src/codegen_gpu.rs:1237` `crate::nvrtc::compile_cuda_source_to_ptx(&cuda_source, fn_name)`; `ferrotorch-jit/src/fusion_gpu.rs:197` `crate::nvrtc::compile_cuda_source_to_ptx(&cuda_source, FUSED_F64_KERNEL_NAME)`. |
| REQ-2 | SHIPPED | impl: `.lines().filter(\|l\| !l.trim().starts_with("#include <math.h>"))` chain in `nvrtc.rs::compile_cuda_source_to_ptx`; non-test consumer: every NVRTC invocation from `codegen_gpu.rs:1237` and `fusion_gpu.rs:197` benefits from this strip. |
| REQ-3 | SHIPPED | impl: the `if l.starts_with("__global__ void ")` rewrite in `nvrtc.rs::compile_cuda_source_to_ptx`; non-test consumer: same call sites — cudarc's `cuModuleGetFunction` keys on the unmangled name post-rewrite. |
| REQ-4 | SHIPPED | impl: `CompileOptions { arch: Some("compute_75"), ..Default::default() }` in `nvrtc.rs::compile_cuda_source_to_ptx`; non-test consumer: every PTX produced for `codegen_gpu` and `fusion_gpu` targets sm_75+. |
| REQ-5 | SHIPPED | impl: `#[cfg(not(feature = "cuda"))] pub fn compile_cuda_source_to_ptx(...) -> Err(JitError::CodegenError { ... })` in `nvrtc.rs`; non-test consumer: `codegen_gpu.rs:1237` compiles under both feature configurations and receives the structured error in the no-CUDA build. |
