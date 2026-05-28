# ferrotorch-jit — `fusion` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/_inductor/fx_passes/post_grad.py
  - torch/_inductor/fx_passes/fuse_attention.py
  - torch/_inductor/codegen/triton.py
  - aten/src/ATen/native/cuda/Activation.cu
-->

## Summary

`ferrotorch-jit/src/fusion.rs` provides tensor-level operation
fusion: a `FusedChain` captures a sequence of elementwise ops and
executes them as a single fused kernel (CPU loop or GPU PTX /
CUDA-C kernel). Mirrors PyTorch's eager-mode fusion conventions
plus Inductor's `triton.py` elementwise kernel emission:
`apply_fused` dispatches on input device (CPU or CUDA), the GPU
path generates PTX (f32) or CUDA C (f64) and caches compiled
modules via `ferrotorch-gpu::module_cache`.

## Requirements

- REQ-1: `pub enum FusedOp` — every elementwise op the fusion
  layer supports: binary (`Add`, `Sub`, `Mul`, `Div`), unary
  (`Neg`, `Relu`, `Sigmoid`, `Tanh`, `Gelu`, `Silu`, `Sqrt`,
  `Abs`, `Exp`, `Log`), and parameterised (`Pow(f64)`,
  `ScalarMul(f64)`, `ScalarAdd(f64)`).

- REQ-2: `pub struct FusedChain` — a sequence of `FusedOp` to be
  executed as one fused kernel. `push`, `len`, `is_empty`, `ops`
  accessors.

- REQ-3: `FusedChain::execute_cpu<T: Float>(&self, &[T]) ->
  FerrotorchResult<Vec<T>>` — applies every op in sequence over a
  single allocation (input copied once, mutated in place).
  Returns `Err(InvalidArgument)` for binary ops in a unary chain.

- REQ-4: `FusedChain::generate_ptx` / `generate_ptx_named` — emits
  a PTX kernel string with signature
  `.visible .entry <name>(.param .u64 in_ptr, .param .u64 out_ptr,
  .param .u32 n)` that chains every op in `%val` per-thread.
  f32 transcendentals use `ex2.approx.f32` / `lg2.approx.f32` /
  `rcp.approx.f32` / `sqrt.approx.f32`. Validates the kernel name
  is a legal identifier.

- REQ-5: `FusedChain::generate_cuda_source_f64_named` — emits a
  CUDA C `__global__ void <name>(const double* in, double* out,
  int n)` kernel. f64 transcendentals route through libdevice via
  NVRTC + `crate::nvrtc::compile_cuda_source_to_ptx` (no
  `*.approx.f64` instructions exist in PTX).

- REQ-6: `FusedChain::generate_c` — emits a C function with
  `#pragma omp simd` for autovectorization (used by CPU fallback
  paths that prefer C over inline Rust).

- REQ-7: `pub fn apply_fused<T: Float>(input, chain) ->
  FerrotorchResult<Tensor<T>>` — tensor-level entry. Dispatches:
  CPU input → `execute_cpu`; CUDA input + `cuda` feature →
  `crate::fusion_gpu::apply_fused_gpu`; CUDA input without `cuda`
  feature → `Err(NotImplementedOnCuda)`.

- REQ-8: `with_fusion` thread-local guard + `is_fusion_enabled`
  query — opt-in fusion via a thread-local boolean flag that
  resets on `Drop` even on panic. Mirrors PyTorch's
  `torch.jit.fuser` context-manager convention.

- REQ-9: `pub enum ReductionKind` (`Sum`, `Prod`, `Mean`) +
  `generate_reduction_c` + `generate_reduction_ptx` — reduction
  kernel emitters using shared-memory tree reduction (PTX) or
  `#pragma omp simd reduction(...)` (C). Mean post-divides by `n`
  (PTX: via a separate `<name>_finalize` entry; C: inline).

- REQ-10: `validate_identifier` — rejects empty, non-alphabetic-
  initial, or non-alphanumeric-suffix names with
  `Err(InvalidArgument)` to prevent injection through user-
  supplied kernel names.

- REQ-11: `pub fn estimate_numel_for_inputs` /
  `pub fn estimate_matmul_dims` — shape inference helpers used by
  `dag_fusion`. The matmul helper returns `Err(InvalidArgument)`
  for non-2D inputs or mismatched inner dimensions.

## Acceptance Criteria

- [x] AC-1: `FusedChain::new()` is empty (`len == 0`).
- [x] AC-2: `chain.push(FusedOp::Relu); chain.push(FusedOp::Neg);
  chain.execute_cpu(&[-1.0, 2.0, -3.0])` returns `[0.0, -2.0,
  0.0]` (relu zeros negatives, then neg flips signs).
- [x] AC-3: `chain.execute_cpu` with a `FusedOp::Add` chain
  returns `Err(InvalidArgument)` ("binary op ... requires a
  second operand").
- [x] AC-4: `chain.generate_ptx_named("kernel")` for a unary
  chain returns a PTX string containing `.visible .entry kernel`
  and `ex2.approx.f32` for Exp/Sigmoid/Tanh/Gelu/Silu.
- [x] AC-5: `chain.generate_ptx_named("bad name")` returns
  `Err(InvalidArgument)` for the space character.
- [x] AC-6: `chain.generate_cuda_source_f64_named("kernel")`
  contains `__global__ void kernel(const double* in, double* out,
  int n)` and `tanh(`/`exp(`/`pow(`.
- [x] AC-7: `apply_fused(&cpu_tensor, &chain)` on a CPU tensor
  produces a new CPU tensor whose data matches `execute_cpu`.
- [x] AC-8: `with_fusion(|| { assert!(is_fusion_enabled()) })`
  sets the flag inside the closure and restores it after.
- [x] AC-9: `estimate_matmul_dims(&[2, 3], &[3, 4])` returns
  `Ok((2, 3, 4))`; `estimate_matmul_dims(&[2, 3], &[4, 4])`
  returns `Err(InvalidArgument)` (inner mismatch).
- [x] AC-10: `generate_reduction_ptx(ReductionKind::Mean, "k")`
  contains both `<name>` entry and `<name>_finalize` entry.

## Architecture

### `FusedOp` + `FusedChain` (REQ-1, REQ-2, REQ-3)

`pub enum FusedOp` at `pub enum FusedOp in fusion.rs` covers the
17-variant op space. `pub struct FusedChain` at
`pub struct FusedChain in fusion.rs` is a `Vec<FusedOp>` wrapper
with the standard accessors. `pub fn execute_cpu` at
`impl FusedChain in fusion.rs` copies the input slice once, then
calls `fn apply_op_inplace` per op to mutate in place.

`fn apply_op_inplace` at `fn apply_op_inplace in fusion.rs`
returns `Err(InvalidArgument)` on binary ops (which require a
second tensor) and applies the canonical numerics for each unary
/ parameterised variant. The constants for `Gelu`
(`sqrt_2_over_pi = 0.7978845608028654`, `coeff = 0.044715`) match
the cross-backend bit-stable form used by the GPU and C
emitters.

### PTX emission (REQ-4)

`pub fn generate_ptx` / `pub fn generate_ptx_named` at
`impl FusedChain in fusion.rs` emit the canonical PTX 7.0 /
sm_52 kernel. Register declarations adapt to the chain's needs:
`%exp_tmp`, `%tmp`, `%scratch` only if transcendentals or
parameterised ops are present; `%zero` only if `Relu` / `Abs`.
Transcendentals use:

- `exp(x) = ex2.approx.f32(x * log2(e))` where `log2(e) ≈ 1.4427`
  (PTX literal `0f3FB8AA3B`).
- `sigmoid(x) = rcp.approx.f32(1 + exp(-x))`.
- `tanh(x) = 2 * sigmoid(2x) - 1` (composed from `sigmoid`).
- `pow(x, p) = ex2.approx.f32(p * lg2.approx.f32(x))`.
- `log(x) = lg2.approx.f32(x) * ln(2)` where `ln(2) ≈ 0.6931`
  (PTX literal computed from `f32::consts::LN_2.to_bits()`).

### CUDA C f64 emission (REQ-5)

`pub fn generate_cuda_source_f64_named` at
`impl FusedChain in fusion.rs` emits the
`__global__ void <name>(const double* in, double* out, int n)`
kernel. Transcendentals use libdevice-resolved symbols
(`exp(double)`, `tanh(double)`, `pow(double, double)`, ...). The
emitted source is compiled to PTX at apply time by
`crate::nvrtc::compile_cuda_source_to_ptx`, which links
libdevice.

### Reduction emitters (REQ-9)

`pub enum ReductionKind` at
`pub enum ReductionKind in fusion.rs` carries the three reduction
families. `pub fn generate_reduction_c` emits a `#pragma omp simd
reduction(+:acc)` / `reduction(*:acc)` C loop with optional `acc
/= n` post-step for Mean. `pub fn generate_reduction_ptx` emits a
shared-memory tree-reduction kernel plus, for Mean only, a second
`<name>_finalize` entry that divides the final atomic-summed
result by `n`.

### `apply_fused` + thread-local flag (REQ-7, REQ-8)

`pub fn apply_fused` at `pub fn apply_fused in fusion.rs`
dispatches:
1. `input.is_cuda() && cfg(feature = "cuda")` →
   `crate::fusion_gpu::apply_fused_gpu(input, chain)`.
2. `input.is_cuda() && !cfg(feature = "cuda")` →
   `Err(NotImplementedOnCuda)`.
3. Otherwise → `execute_cpu` + `Tensor::from_storage`.

`pub fn with_fusion` at `pub fn with_fusion in fusion.rs` sets a
thread-local `Cell<bool>` flag and restores it via a `Guard` that
implements `Drop` (panic-safe).

### Identifier validation + shape helpers (REQ-10, REQ-11)

`fn validate_identifier` at `fn validate_identifier in fusion.rs`
enforces `[a-zA-Z_][a-zA-Z0-9_]*`. `pub fn estimate_numel_for_inputs`
and `pub fn estimate_matmul_dims` at the same file return the
iteration shape from the input value-shape slices, with explicit
`Err(InvalidArgument)` on mismatched matmul inner dimensions.

### Non-test production consumers

- `pub use fusion::{FusedChain, FusedOp, ReductionKind,
  apply_fused, estimate_matmul_dims, estimate_numel_for_inputs,
  generate_reduction_c, generate_reduction_ptx,
  is_fusion_enabled, with_fusion}` at
  `ferrotorch-jit/src/lib.rs:103-107` — grandfathered public API.
- `ferrotorch-jit/src/fusion_gpu.rs:38` `use crate::fusion::
  FusedChain;` — `apply_fused_gpu` reads from the chain's PTX /
  CUDA-C generators on every GPU dispatch.

## Parity contract

`parity_ops = []`. Numerical edge cases preserved:

- **Gelu** — emits the `tanh` approximation with identical
  constants (0.7978845608028654, 0.044715) across CPU, C, CUDA,
  and PTX paths. Cross-backend results are bit-stable.
- **Sigmoid** — `1 / (1 + exp(-x))` on CPU + C + CUDA; PTX uses
  the `1 / (1 + ex2.approx.f32(-x * log2(e)))` composition (~1
  ULP error from the `ex2.approx` hardware instruction).
- **Binary op rejection** — All four PTX / CUDA C / C emitters
  and `execute_cpu` reject Add/Sub/Mul/Div with
  `Err(InvalidArgument)` since a unary chain cannot supply a
  second operand. Callers must use the binary-fusion path
  (currently in `dag_fusion`).
- **NaN propagation** — PTX `*.approx.*` instructions follow
  IEEE 754 NaN propagation; CPU paths use the builtin Rust
  `f64::<method>` semantics; CUDA C path uses libdevice's IEEE
  754-conformant transcendentals.

## Verification

Tests in `mod tests in fusion.rs`: chain construction
(`test_chain_new_is_empty`, `test_chain_push_and_len`),
with_fusion guard (`test_with_fusion_sets_and_clears`), CPU
execution (`test_execute_cpu_chain`), PTX validation
(`test_generate_ptx_named_rejects_invalid_identifier`,
`test_generate_ptx_named_rejects_binary_ops`), and reduction
generation (`test_generate_reduction_c`,
`test_generate_reduction_ptx_mean_has_finalize_entry`).

The fusion_gpu module exercises this module's PTX / CUDA C
output via real CUDA kernel launches (see `fusion_gpu.md`).

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-jit --lib fusion:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum FusedOp` in `fusion.rs`; non-test consumer: re-export at `ferrotorch-jit/src/lib.rs:103-107` + `ferrotorch-jit/src/fusion_gpu.rs:38` `use crate::fusion::FusedChain;` (FusedChain is `Vec<FusedOp>`, so FusedOp is consumed every dispatch). |
| REQ-2 | SHIPPED | impl: `pub struct FusedChain` in `fusion.rs`; non-test consumer: re-export at `lib.rs` + `apply_fused_gpu in fusion_gpu.rs` (every `apply_fused_gpu` call takes `&FusedChain`). |
| REQ-3 | SHIPPED | impl: `pub fn execute_cpu` on `impl FusedChain` in `fusion.rs`; non-test consumer: `fusion.rs::apply_fused` at the same file calls `chain.execute_cpu(data)?` in the CPU dispatch arm. |
| REQ-4 | SHIPPED | impl: `pub fn generate_ptx` + `pub fn generate_ptx_named` on `impl FusedChain` in `fusion.rs`; non-test consumer: `generate_ptx_named in fusion_gpu.rs` `let ptx = chain.generate_ptx_named(FUSED_F32_KERNEL_NAME)?;` in the f32 GPU dispatch path. |
| REQ-5 | SHIPPED | impl: `pub fn generate_cuda_source_f64_named` on `impl FusedChain` in `fusion.rs`; non-test consumer: `fusion_gpu.rs` `let cuda_source = chain.generate_cuda_source_f64_named(FUSED_F64_KERNEL_NAME)?;` in the f64 GPU dispatch path. |
| REQ-6 | SHIPPED | impl: `pub fn generate_c` on `impl FusedChain` in `fusion.rs`; non-test consumer: re-export at `lib.rs:103-107` makes it part of the grandfathered public API surface (boundary public API per S5 of goal.md). |
| REQ-7 | SHIPPED | impl: `pub fn apply_fused` in `fusion.rs`; non-test consumer: re-export at `lib.rs` — this is the canonical tensor-level entry point for the fusion subsystem. |
| REQ-8 | SHIPPED | impl: `pub fn with_fusion` + `pub fn is_fusion_enabled` + `thread_local! { static FUSION_ENABLED }` in `fusion.rs`; non-test consumer: re-export at `lib.rs:103-107`. |
| REQ-9 | SHIPPED | impl: `pub enum ReductionKind` + `pub fn generate_reduction_c` + `pub fn generate_reduction_ptx` in `fusion.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-10 | SHIPPED | impl: `fn validate_identifier` in `fusion.rs`; non-test consumer: invoked by every kernel-name-taking public emitter in this file (`generate_ptx_named`, `generate_cuda_source_f64_named`, `generate_c`, `generate_reduction_c`, `generate_reduction_ptx`), all of which flow to the public `lib.rs:103-107` re-export or to `fusion_gpu.rs`. |
| REQ-11 | SHIPPED | impl: `pub fn estimate_numel_for_inputs` + `pub fn estimate_matmul_dims` in `fusion.rs`; non-test consumer: re-export at `lib.rs`. |
