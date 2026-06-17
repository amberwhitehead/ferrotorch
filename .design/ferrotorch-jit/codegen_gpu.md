# ferrotorch-jit — `codegen_gpu` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/_inductor/codegen/cuda/cuda_kernel.py
  - torch/_inductor/codegen/triton.py
  - torch/_inductor/codegen/triton_split_scan.py
  - torch/_inductor/codegen/common.py
-->

## Summary

`ferrotorch-jit/src/codegen_gpu.rs` emits CUDA C and PTX source code
from a `LoopIR` program. Mirrors PyTorch Inductor's
`torch._inductor.codegen.cuda.cuda_kernel.CUDAKernel` plus the PTX
emission stub: each fusion group is lowered to a single
`__global__` kernel that maps the outermost loop to GPU threads,
chains transcendentals through hardware approximations (f32) or
libdevice (f64), and uses shared-memory tree reductions for sums /
products / means.

## Requirements

- REQ-1: `pub struct GpuCodegen` — zero-sized type carrying the two
  emission entry points. Mirrors Inductor's `CUDAKernel` /
  `TritonKernel` as the emitter object.

- REQ-2: `pub fn generate_cuda_source(loops, fn_name, num_inputs,
  dtype) -> Result<String, JitError>` — emits a CUDA C `__global__`
  kernel with `blockIdx.x * blockDim.x + threadIdx.x` indexing,
  bounds checking against `n`, shared memory for reductions, and
  coalesced access patterns.

- REQ-3: `pub fn generate_ptx_source(loops, fn_name, block_size,
  num_inputs, dtype) -> Result<String, JitError>` — emits PTX
  targeting `sm_52` with hand-scheduled register allocation, the
  `.visible .entry` boilerplate, and approximate transcendentals
  (`ex2.approx.f32`, `lg2.approx.f32`, `rcp.approx.f32`,
  `sqrt.approx.f32`) for f32.

- REQ-4: Per-kernel single-dtype dispatch (#729) — both emitters
  accept a `Dtype` parameter that selects CUDA `float` vs `double`,
  PTX `.f32` vs `.f64` suffixes, register declarations, load/store
  widths, and constant literal encoding (`0f...` 8 hex digits for
  f32; `0d...` 16 hex digits for f64).

- REQ-5: f64 transcendentals on PTX — explicitly rejected with
  `JitError::Unsupported { op, dtype }` (no `*.approx.f64`
  instructions exist). The f64 transcendental path must route
  through `generate_cuda_source` → NVRTC + libdevice instead.

- REQ-6: Reduction emission — both CUDA and PTX paths emit
  tree-reduction in shared memory and use `atomicAdd` (sum/mean) or
  `atomicCAS` (prod) to combine per-block partials. Mean divides
  by `n` via a second kernel entry (`<name>_finalize` for PTX) /
  inline `/n` (for CUDA).

- REQ-7: Coalesced memory access — sequential thread IDs map to
  sequential addresses (`output[tid] = ...`, not `output[tid * N]`).
  Matches CUDA's coalescing requirement for full-bandwidth global
  loads/stores.

- REQ-8: Block-size configurability — `generate_ptx_source` takes
  an explicit `block_size: usize` so callers (notably
  `InductorBackend::with_block_size`) can tune the launch shape.

## Acceptance Criteria

- [x] AC-1: `GpuCodegen::generate_cuda_source(&[], "k", 1, F32)`
  returns a well-formed `__global__ void k(...)` declaration with
  `#include <math.h>`.
- [x] AC-2: A single `IrOpKind::Neg` lowered + emitted contains
  `blockIdx.x * blockDim.x + threadIdx.x`, `if (tid >= n) return;`,
  and `in0[tid]`.
- [x] AC-3: A `IrOpKind::Sum` lowered + emitted contains
  `extern __shared__`, `sdata[`, `__syncthreads()`, and
  `atomicAdd(&output[0]`.
- [x] AC-4: `generate_ptx_source` for the same `Neg` graph contains
  `.visible .entry`, `.f32 %val`, `ld.global.f32`,
  `st.global.f32`.
- [x] AC-5: `generate_ptx_source(..., F64)` for an f64
  transcendental returns `Err(JitError::Unsupported { ... })`.
- [x] AC-6: CUDA `Sigmoid` emission contains `1.0 / (1.0 + exp(-x))`;
  CUDA `Tanh` emission contains `tanh(`.
- [x] AC-7: f64 CUDA emission uses `double` (not `float`); PTX f64
  uses `.f64` and `ld.global.f64` (where supported).

## Architecture

### `GpuCodegen` + CUDA emission (REQ-1, REQ-2, REQ-4)

`pub struct GpuCodegen` at `pub struct GpuCodegen in codegen_gpu.rs`
is the unit struct holding both emission paths.
`pub fn generate_cuda_source` at
`impl GpuCodegen in codegen_gpu.rs` builds the function signature
(`const <scalar>* __restrict__ in0, ..., <scalar>* __restrict__
output, int n`), the thread-index computation
(`tid = blockIdx.x * blockDim.x + threadIdx.x;`), and dispatches to
either `emit_cuda_elementwise` (non-reduction path: every thread
handles one outer-loop iteration) or `emit_cuda_reduction`
(stride-loop + tree-reduction).

### PTX emission (REQ-3, REQ-4, REQ-5)

`pub fn generate_ptx_source` at
`impl GpuCodegen in codegen_gpu.rs` emits the PTX 7.0 / sm_52
preamble, the `.visible .entry` declaration, register declarations
(`.reg .f32 %val, ...`), and the body. Transcendentals on f32 use
the hardware-approximation instructions (`ex2.approx.f32`,
`lg2.approx.f32`, `rcp.approx.f32`, `sqrt.approx.f32`); on f64,
they fail with `Err(JitError::Unsupported { op, dtype })` — there
are no `*.approx.f64` instructions and the f64 transcendental path
must route through CUDA C + NVRTC + libdevice instead. The
`#[allow(unsafe_code, reason = "...")]` annotation at the module
top is for the cudarc kernel-launch unsafe blocks, not for codegen
itself.

### Reductions (REQ-6)

`fn generate_ptx_reduction_source` at `fn generate_ptx_reduction_source in codegen_gpu.rs`
emits the tree-reduction pattern:
1. Each thread accumulates strided elements into `thread_acc`.
2. Threads write `thread_acc` into `sdata[local_tid]` and
   `__syncthreads()`.
3. Tree-reduce via `for (s = blockDim.x/2; s > 0; s >>= 1)`.
4. Thread 0 of each block calls
   `atomicAdd(&output[0], sdata[0])` (sum) or
   `atomicAdd(&output[0], sdata[0] / (<scalar>)n)` (mean).

The PTX reduction emission (called transitively via
`generate_ptx_source` when `loops_contain_accumulate(loops)`) uses
the same shared-memory pattern with `.shared .f32 sdata[1024];`
and `bar.sync 0;` for `__syncthreads`.

### Coalesced access + block size (REQ-7, REQ-8)

The outer loop becomes `tid`, so thread `t` accesses `in[t]` and
writes `out[t]`. This is the canonical coalesced pattern (warp `w`
reads addresses `w*32..(w+1)*32 - 1`). `block_size` is plumbed
through `InductorBackend::with_block_size(target, block_size)` and
emitted into the PTX kernel's body indirectly (the kernel itself is
launch-shape-agnostic; `block_size` informs the launch config).

### Non-test production consumers

- `pub use codegen_gpu::GpuCodegen` at
  `ferrotorch-jit/src/lib.rs` — grandfathered public API.
- `ferrotorch-jit/src/codegen.rs` calls
  `crate::codegen_gpu::GpuCodegen::generate_cuda_source(loops,
  &fn_name, num_inputs, dtype)` from
  `InductorBackend::generate` `GpuCuda` arm.
- `ferrotorch-jit/src/codegen.rs` calls
  `crate::codegen_gpu::GpuCodegen::generate_ptx_source(loops,
  &fn_name, self.block_size, num_inputs, dtype)` from the `GpuPtx`
  arm.

## Parity contract

`parity_ops = []`. This is a GPU source emitter; runtime parity
lives in the executors (`fusion_gpu.rs` for fused chains, the
nvrtc + module-cache path for everything else). Numerical edge
cases preserved:

- **f32 transcendental ULPs** — `ex2.approx.f32` is ~1 ULP;
  `rcp.approx.f32` is ~1 ULP; tolerance vs CPU/PyTorch is ~1e-5 for
  composed chains (consistent with PyTorch's CUDA tolerance).
- **f64 PTX rejection** — `JitError::Unsupported { op, dtype }`
  for transcendentals on PTX f64. Closes #748 follow-up to #729.
- **Mean reduction** — divides by `(scalar)n` (not by
  `(float)n` regardless of dtype) so f64 means stay full-precision.
- **NaN / Inf** — hardware-approximation instructions propagate
  NaN consistent with IEEE 754; tested implicitly via the runtime
  parity tests in `fusion_gpu`.
- **GELU bit-stability** — emits the tanh-approximation matching
  CPU + fusion paths verbatim for cross-backend identity.

## Verification

Tests in `mod tests in codegen_gpu.rs`: shape tests
(`test_cuda_neg_kernel_shape`, `test_ptx_neg_kernel_shape`),
reduction tests (`test_cuda_sum_uses_atomic_add`,
`test_ptx_sum_uses_shared_memory`), dtype dispatch tests
(`test_cuda_f64_uses_double`,
`test_ptx_f64_transcendental_returns_unsupported`).

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-jit --lib codegen_gpu:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct GpuCodegen` in `codegen_gpu.rs`; non-test consumer: re-export at `codegen_gpu in ferrotorch-jit/src/lib.rs` + `generate_cuda_source in ferrotorch-jit/src/codegen.rs` `crate::codegen_gpu::GpuCodegen::generate_cuda_source(loops, &fn_name, num_inputs, dtype)`. |
| REQ-2 | SHIPPED | impl: `pub fn generate_cuda_source` in `codegen_gpu.rs`; non-test consumer: `generate_cuda_source in codegen.rs` (`GpuCuda` arm of `InductorBackend::generate`) + `codegen in codegen.rs` (identity-graph `GpuCuda` fallback). |
| REQ-3 | SHIPPED | impl: `pub fn generate_ptx_source` in `codegen_gpu.rs`; non-test consumer: `generate_ptx_source in codegen.rs` (`GpuPtx` arm of `InductorBackend::generate`) + `codegen in codegen.rs` (identity-graph `GpuPtx` fallback). |
| REQ-4 | SHIPPED | impl: `cuda_scalar_name` + `cuda_zero_literal` + `ptx_dtype_suffix` helpers in `codegen_gpu.rs` switched on the `Dtype` parameter; non-test consumer: every emission path through `codegen in codegen.rs` / `codegen in codegen.rs` passes the resolved group `dtype` from `resolve_group_dtype`. |
| REQ-5 | SHIPPED | impl: f64 transcendental check inside `pub fn generate_ptx_source` returning `Err(JitError::Unsupported { ... })` in `codegen_gpu.rs`; non-test consumer: `codegen in codegen.rs` propagates the error via `.map_err(FerrotorchError::from)` so callers see the structured Unsupported diagnosis. |
| REQ-6 | SHIPPED | impl: `fn generate_ptx_reduction_source` (PTX path) + the PTX `.shared` + `bar.sync 0;` block inside `pub fn generate_ptx_source` in `codegen_gpu.rs`; non-test consumer: transitively via `codegen in codegen.rs` / `codegen in codegen.rs` for any fusion group containing Sum/Mean/Prod. |
| REQ-7 | SHIPPED | impl: `tid = blockIdx.x * blockDim.x + threadIdx.x;` + sequential `output[tid]` write pattern in `fn emit_cuda_elementwise` (`codegen_gpu.rs`); non-test consumer: transitively via `codegen in codegen.rs`. |
| REQ-8 | SHIPPED | impl: `pub fn generate_ptx_source(..., block_size, ...)` parameter in `codegen_gpu.rs`; non-test consumer: `codegen in codegen.rs` passes `self.block_size` from `InductorBackend::with_block_size`. |
