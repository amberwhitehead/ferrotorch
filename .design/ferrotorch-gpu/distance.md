# GPU Pairwise Distance (cdist)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/DistanceKernel.cu
  - aten/src/ATen/native/Distance.cpp
-->

## Summary

`ferrotorch-gpu/src/distance.rs` ships an on-device PTX kernel for
`torch.cdist` (crosslink #1545 / sub #1535): the batched Lp pairwise distance
matrix. `x1` is `[B, P, M]`, `x2` is `[B, R, M]`, the result is `[B, P, R]`
with `out[b, i, j] = (sum_k |x1[b,i,k] - x2[b,j,k]|^p)^(1/p)`.

This mirrors the upstream CUDA kernel `cdist_kernel_cuda_impl`
(`aten/src/ATen/native/cuda/DistanceKernel.cu:195`) and the per-norm
accumulate/finish in `dists<scalar_t>::{p,one,two,inf}`
(`DistanceKernel.cu:50-86`). Upstream assigns one CUDA *block* per output cell
and parallelises the `M`-reduction across the block's threads; ferrotorch keeps
the identical arithmetic but assigns one *thread* per output cell with a serial
ascending `M`-loop — the reduction order over `k` is the same, so the float
result matches the CPU ferrotorch `cdist` and `torch.cdist` to the usual fp
tolerance.

`diff` fed to the accumulator is `|x1 - x2|` (`std::abs(*a - *b)`,
`DistanceKernel.cu:210`). Per-norm:

- **two** (`p == 2`): `agg += diff*diff`; `finish = sqrt(agg)`.
- **one** (`p == 1`): `agg += diff`; `finish = agg`.
- **inf** (`p == inf`): `agg = max(agg, diff)`; `finish = agg`.
- **general** (other finite `p`): `agg += diff^p`; `finish = agg^(1/p)`,
  computed on-device via `2^(p*log2(diff))` (f32 only).

The norm dispatch mirrors `DistanceKernel.cu:230-240` (which special-cases
`0`, `1`, `2`, `inf`). The `p == 0` count-of-nonzeros norm is delegated to the
CPU path. The f64 kernel covers `p in {1, 2, inf}` only — the base PTX ISA has
no accurate f64 `pow`, so general-p f64 also falls back to CPU.

## Requirements

- REQ-1: `gpu_cdist_f32` — launch the f32 cdist PTX over `B*P*R` output cells
  (one thread each), covering `p in {1, 2, inf}` and general finite `p > 0`.
  Returns a fresh resident `CudaBuffer<f32>` of `B*P*R` elements.
- REQ-2: `gpu_cdist_f64` — f64 counterpart covering `p in {1, 2, inf}`;
  general-p / `p == 0` return `GpuError::Unsupported` so the caller falls back
  to CPU.
- REQ-3: batched layout — the kernel decodes `(b, i, j)` from the flat output
  index and reads `x1[(b*P+i)*M + k]` / `x2[(b*R+j)*M + k]`, so the same kernel
  serves the unbatched (`B == 1`) and batched cases.
- REQ-4: norm modes — `MODE_ONE` / `MODE_TWO` / `MODE_INF` / `MODE_GENERAL`
  selected by `mode_for_p(p)`, branched in the PTX, matching the upstream norm
  dispatch.

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-gpu --features cuda cdist` passes LIVE on
  the RTX 3090.
- [x] AC-2: `gpu_cdist_f32` L2 from `(0,0)`, `(1,0)`, `(0,1)` to `(1,1)` is
  `[sqrt(2), 1, 1]` and equals the CPU reference.
- [x] AC-3: L1 of the same inputs is `[2, 1, 1]`; Linf is verified against the
  CPU `max|diff|` reference.
- [x] AC-4: general `p == 3` matches the CPU `powf` reference within fp
  tolerance (the `2^(p*log2)` device path).
- [x] AC-5: a `[2,P,M]`/`[2,R,M]` batched call matches the CPU reference cell
  for cell.
- [x] AC-6: f64 L2 matches `sqrt` to 1e-12.
- [x] AC-7: a CUDA tensor passed to `ops::tensor_ops::cdist` returns a tensor
  whose storage `is_cuda()` (NO `.cpu()` round trip) and whose `.cpu()` data
  equals the CPU reference.

## Architecture

`distance.rs` is `#![cfg(feature = "cuda")]` (mirrors `triangular.rs`). Two PTX
template constants (`CDIST_F32_PTX`, `CDIST_F64_PTX`) carry one entry each with
the 9-arg ABI `(x1, x2, out, total, p_dim, r_dim, m, p, mode)`. Thread
`t in [0, total)` decodes `b = t / (P*R)`, `i = (t % (P*R)) / R`,
`j = (t % (P*R)) % R`, loops `k in [0, M)` accumulating `|x1-x2|` per `mode`,
and writes one `out[t]`. `mode_for_p` resolves the norm selector and gates the
GPU-vs-CPU decision (`p == 0` -> CPU; general-p f64 -> CPU). `launch_cdist_f32`
/ `launch_cdist_f64` resolve the entry via `module_cache::get_or_compile`,
validate the output length, short-circuit empty launches, and launch one 1-D
grid.

The backend (`backend_impl.rs`) overrides `cdist_f32`/`cdist_f64`, unwrapping
both dtype-tagged input handles and re-wrapping the result.
`ops::tensor_ops::cdist` gains a CUDA branch (after shape validation) that
requires both inputs on-device, dispatches on `is_f32`/`is_f64` +
`gpu_dispatch::cdist_supported_f32`/`_f64`, and returns a GPU-resident result;
unsupported `(dtype, p)` combinations surface as `NotImplementedOnCuda` rather
than a silent host fallback.

**Non-test consumer**: `ferrotorch_core::ops::tensor_ops::cdist` (re-exported as
`ferrotorch_core::cdist`) calls `GpuBackend::cdist_f32`/`cdist_f64` for
CUDA-resident inputs.

## Parity contract

`parity_ops = []`. Numeric contract is fp-tolerance parity with `torch.cdist`
and the ferrotorch CPU `cdist`. Verified by the LIVE GPU-vs-CPU unit tests in
`distance.rs` and the `tensor_ops.rs` CUDA dispatch test.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn gpu_cdist_f32 in distance.rs`; non-test consumer: `CudaBackendImpl::cdist_f32 in backend_impl.rs` dispatched from the `is_cuda()` branch of `cdist in ops/tensor_ops.rs` |
| REQ-2 | SHIPPED | impl: `pub fn gpu_cdist_f64 in distance.rs`; non-test consumer: `CudaBackendImpl::cdist_f64 in backend_impl.rs` |
| REQ-3 | SHIPPED | impl: `(b, i, j)` decode in the cdist PTX in `distance.rs`; verified by `cdist_f32_batched` unit test |
| REQ-4 | SHIPPED | impl: `MODE_ONE`/`MODE_TWO`/`MODE_INF`/`MODE_GENERAL` branch in the PTX selected by `fn mode_for_p in distance.rs`; verified by `cdist_f32_{l2,l1,linf,p3}` unit tests |
