# Custom PTX kernel registry (the giant `kernels.rs`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/BinaryMulKernel.cu
  - aten/src/ATen/native/cuda/AbsKernel.cu
  - aten/src/ATen/native/cuda/UnaryOpsKernel.cu
  - aten/src/ATen/native/cuda/ActivationGeluKernel.cu
  - aten/src/ATen/native/cuda/ActivationSiluKernel.cu
  - aten/src/ATen/native/cuda/Reduce.cu
  - aten/src/ATen/native/cuda/ReduceSumProdKernel.cu
  - aten/src/ATen/native/cuda/ScanKernels.cu
  - aten/src/ATen/native/cuda/SoftMax.cu
  - aten/src/ATen/native/cuda/layer_norm_kernel.cu
  - aten/src/ATen/native/cuda/Indexing.cu
  - aten/src/ATen/native/cuda/ScatterGatherKernel.cu
  - aten/src/ATen/native/cuda/Dropout.cu
  - aten/src/ATen/native/cuda/AdaptiveAveragePooling.cu
-->

## Summary

`ferrotorch-gpu/src/kernels.rs` is the largest single file in the
workspace (~27.7K LOC) — a registry of hand-written PTX kernels and
their Rust-side launcher wrappers covering the bulk of ferrotorch's
CUDA compute surface. The file is organised by op category, not by
upstream PyTorch file: 362 `pub fn gpu_*` entry points are stamped
out across ~50 logical sections, each backed by one or more
`pub(crate) const *_PTX: &str` PTX template strings. Mirrors the
entire `aten/src/ATen/native/cuda/*Kernel.cu` family — instead of
one Rust file per upstream `.cu`, ferrotorch consolidates into a
single registry because the PTX share scaffolding (launch grids,
shared-memory reductions, broadcast strides) that benefits from
co-location. The `ptx_f32_to_f64` mechanical-substitution helper
near the top of the file expands many `_F32_PTX` constants into their
`_F64_PTX` siblings at runtime, halving the maintenance surface for
dtype-pair kernels.

## Requirements

- REQ-1: **f32 → f64 PTX auto-conversion**
  (`pub(crate) fn ptx_f32_to_f64`, `pub(crate) fn get_f64_ptx`):
  mechanical string substitution from an f32 PTX template to its f64
  equivalent, covering register types, load/store widths, byte offsets,
  shift constants, and the float-hex literals for `0.0`/`1.0`/`-1.0`/
  `2.0`/`0.5`/`±inf`/`log2(e)`/`ln(2)`. Avoids hand-maintaining a
  parallel f64 PTX block for every kernel that's pure linear math.
- REQ-2: **Elementwise binary ops** (`pub fn gpu_add` / `gpu_sub` /
  `gpu_mul` / `gpu_div` for f32 + f64): standard 1-D launch
  geometry, optional vec4-packed fast path for `n >= 16 && n % 4 == 0`
  (`ADD_VEC4_PTX` / `MUL_VEC4_PTX`). Mirrors
  `aten/src/ATen/native/cuda/BinaryMulKernel.cu` family.
- REQ-3: **Broadcast binary ops** (`gpu_broadcast_add/sub/mul/div`
  for f32 + f64): N-dimensional broadcast via per-thread index
  decomposition + per-dim stride lookup. Output shape computed via
  numpy-style broadcast rules. Mirrors upstream's generic
  TensorIterator broadcast.
- REQ-4: **Unary ops** (`pub fn gpu_neg` / `gpu_relu` / `gpu_exp` /
  `gpu_log` / `gpu_sqrt` / `gpu_pow` / `gpu_abs` / `gpu_sigmoid` /
  `gpu_tanh` for f32 + f64): one-thread-per-element with cudarc's
  `ex2.approx.f32` / `lg2.approx.f32` / `sqrt.approx.f32` transcendentals
  where applicable. Mirrors `aten/src/ATen/native/cuda/UnaryOpsKernel.cu`
  family.
- REQ-5: **Activations + backward** (gelu (erf + tanh approximation),
  silu, elu, mish, clamp + backward each + the sigmoid/tanh/abs
  backwards used by activation grad_fns). f64 erf series-evaluation
  PTX is hand-written because the converter can't lift the small-x /
  mid-x / tail polynomial templates correctly. Mirrors
  `ActivationGeluKernel.cu`, `ActivationSiluKernel.cu`,
  `ActivationMishKernel.cu`, `ActivationEluKernel.cu` upstream.
- REQ-6: **Reductions** (`gpu_reduce_sum` / `gpu_reduce_prod` /
  `gpu_reduce_min` / `gpu_reduce_max`, masked variants,
  `gpu_sum_axis`, `gpu_prod_backward_f32/f64`): tree-reduction within
  blocks + cross-block accumulation via second-stage kernel or atomic
  add. f64 atomics require `.target sm_60` (lifted by the auto-converter).
  Mirrors `Reduce.cu` + `ReduceSumProdKernel.cu`.
- REQ-7: **Cumulative scans** (`gpu_cumsum` / `gpu_cumprod` /
  `gpu_cummax` / `gpu_cummin` / `gpu_logcumsumexp`): hand-written
  blelloch-style scan PTX with the documented `0xFF800000` /
  `0x7F800000` seed bit-patterns (the converter has special handling
  for `mov.b32 %acc, 0xFF800000` → `mov.b64 %acc, 0xFFF0000000000000`
  to preserve `-inf` as the cummax init). Mirrors `ScanKernels.cu`.
- REQ-8: **Softmax / log_softmax + backward** (`gpu_softmax` /
  `gpu_log_softmax` / `gpu_softmax_backward` / `gpu_log_softmax_backward`,
  plus the `bf16_f32` mixed-precision softmax for inference):
  row-wise numerically-stable softmax with shared-memory max + sum
  reductions. Mirrors `aten/src/ATen/native/cuda/SoftMax.cu`.
- REQ-9: **Normalisations** (`gpu_layernorm` / `gpu_layernorm_backward` /
  `gpu_rmsnorm` / `gpu_rmsnorm_backward` for f32 + f64): row-wise
  shared-memory mean / variance reductions with per-column affine.
  Backward uses atomic f64 accumulation (lifted to sm_60). Mirrors
  `aten/src/ATen/native/cuda/layer_norm_kernel.cu`.
- REQ-10: **Indexing** (`gpu_index_select_1d` / `gpu_index_select_dim` /
  `gpu_index_select_dim_f64` / `gpu_scatter_add_1d` / strided
  variants): legacy f32-encoded index path (newer integer-index path
  is in `gather_int.rs`). Mirrors `Indexing.cu` +
  `ScatterGatherKernel.cu`.
- REQ-11: **Masked compute** (`gpu_masked_fill` / `gpu_masked_zero` —
  the legacy f32-mask path; the modern u8-mask path is in
  `masked_kernels.rs`). Mirrors upstream's `masked_fill_kernel_cuda`.
- REQ-12: **Strided / shape ops** (`gpu_strided_split` /
  `gpu_strided_cat` / `gpu_transpose_2d` / `gpu_permute_0213` /
  `gpu_strided_copy` / `gpu_strided_scatter` / complex-matrix
  transpose for the FFT path). Mirrors the multi-axis layout ops in
  the native CUDA tree.
- REQ-13: **Embedding / KV cache** (`gpu_embed_lookup` /
  `gpu_embed_lookup_batch` / `gpu_scatter_add_rows` /
  `gpu_slice_write` / `gpu_slice_read` plus indirect variants).
  Mirrors `aten/src/ATen/native/cuda/Embedding.cu` and the KV-cache
  slicing semantics PyTorch's Llama inference uses.
- REQ-14: **Convolution / pooling / normalisation extras**
  (`gpu_small_matmul` cuBLAS-bypass, `gpu_small_bmm`, MaxPool2d /
  AvgPool2d kernels, BatchNorm2d, padded-truncate complex helpers
  for FFT). Mirrors `AdaptiveAveragePooling.cu`, `AveragePool2d.cu`,
  `MaxPoolWithIndices.cu`, `Normalization.cu`.
- REQ-15: **Dropout** (`gpu_dropout` with xorshift RNG for the
  in-kernel `[0, 1)` sample). The Philox-driven dropout path uses
  `crate::rng` for the seed. Mirrors `Dropout.cu`.
- REQ-16: **bf16 elementwise + reduction + activations** — a parallel
  family (`gpu_*_bf16` and `gpu_*_bf16_f32` mixed-precision variants)
  added in #963 for inference workloads. Each bf16 op uses a
  dedicated PTX template that converts the 16-bit input to f32 in
  registers for the arithmetic, then truncates back. Mirrors upstream's
  bf16 dispatch in `aten/src/ATen/Dispatch.h`.
- REQ-17: **Fused optimizer steps** (`gpu_fused_adam_step` and the
  fused GRU cell). Mirrors `FusedAdamWKernels.cu` and the optimised
  GRU forward upstream.
- REQ-18: **Public-API _into helpers** (`gpu_add_into`,
  `gpu_mul_into`, `gpu_softmax_into`, etc.) — write to a pre-allocated
  output buffer to avoid the allocator on the hot path. Used by the
  inplace-op family and graph-capture-safe op variants.
- REQ-19: **f32 → f16 / f16 → f32 conversion** (`gpu_f32_to_f16` /
  `gpu_f16_to_f32`) with PTX `cvt.rn.f16.f32` / `cvt.f32.f16`.
  Mirrors upstream's half-precision dtype-cast utilities.
- REQ-20: **Non-CUDA stubs** (#cfg-gated `pub fn` blocks at lines
  ~14301, 14401, 14653, 14751, 14861, 14970, 15066, 23177-23834,
  24007-24312, 26061+, 26148+) — every CUDA-feature `pub fn` has a
  matching stub returning `GpuError::DeviceUnavailable` so
  downstream crates compile cleanly without the `cuda` feature.

## Acceptance Criteria

- [x] AC-1: `ptx_f32_to_f64` exists at line 39, `get_f64_ptx` at
  line 173, both `pub(crate)`. The substitution covers the
  documented register types, byte offsets, transcendental approx
  variants, and the `mov.b32 ±inf` special-case lift.
- [x] AC-2: 4 elementwise binary entry points (`gpu_add` 13114,
  `gpu_sub` 13146, `gpu_mul` 13168, `gpu_div` in the same section);
  vec4 fast paths use `try_launch_binary_vec4` helper.
- [x] AC-3: 4 broadcast-binary entry points
  (`gpu_broadcast_add/sub/mul/div` at 13200-13284).
- [x] AC-4: 10+ unary entry points: `gpu_neg` 13325, `gpu_relu`
  13342, plus `gpu_exp`, `gpu_log`, `gpu_sqrt`, `gpu_pow`,
  `gpu_abs`, `gpu_sigmoid`, `gpu_tanh` (in the "Public API --
  elementwise transcendentals & math ops" section at 17768).
- [x] AC-5: Activation suite + backwards (gelu erf + tanh, silu,
  elu, mish, clamp, sigmoid bw at 13885, tanh bw at 13909, gelu
  backwards at 13382 / 13402). Erf series-evaluation PTX hand-written
  between lines 1868-1960.
- [x] AC-6: Reductions: `gpu_reduce_sum` (14204 f32 / 14301 stub),
  `gpu_reduce_prod` (14309 / 14401), `gpu_reduce_min` (14560 /
  14653), `gpu_reduce_max` (14660 / 14751), `gpu_sum_axis` (15080),
  masked variants (`gpu_masked_reduce_min/max` 14760 / 14871).
- [x] AC-7: Cumulative scans: `gpu_cumsum` (15174), `gpu_cumprod`
  (15262), `gpu_cummax` (15345), `gpu_cummin` (15436),
  `gpu_logcumsumexp` (15523).
- [x] AC-8: Softmax: `gpu_softmax` and `gpu_log_softmax` (14028) +
  backward (`gpu_softmax_backward` 13937, `gpu_log_softmax_backward`
  14112); bf16→f32 softmax PTX at 9249.
- [x] AC-9: Normalisations: LayerNorm + backward, RMSNorm + backward
  (sections at 21424-21744).
- [x] AC-10: Indexing: `gpu_index_select_1d` (13427), `gpu_scatter_add_1d`
  (13506), `gpu_index_select_dim` (13596), `gpu_index_select_dim_f64`
  (13692).
- [x] AC-11: Masked compute: `gpu_masked_fill` (13789), `gpu_masked_zero`
  (13867).
- [x] AC-12: Strided / shape: `gpu_strided_split` (15615),
  `gpu_strided_cat` (15721), `gpu_transpose_2d` (16678 section),
  `gpu_permute_0213` (16754 section), `gpu_strided_copy` (15799
  section), `gpu_strided_scatter` (16124 section).
- [x] AC-13: Embedding / KV cache: `gpu_embed_lookup` (16948 section),
  `gpu_slice_write` (17019 section), `gpu_slice_read` (17096 section).
- [x] AC-14: `gpu_small_matmul`, pool kernels, batch-norm sections
  at lines 16837, 21030, 21286.
- [x] AC-15: `gpu_dropout` (16503 section).
- [x] AC-16: bf16 elementwise / axis-reduction / activation
  PTX + dispatch functions at lines 9448-9866 (PTX) and
  26477-26884 (dispatch).
- [x] AC-17: Fused Adam (20776 section), fused GRU cell (20905 section).
- [x] AC-18: `_into` helpers at 12603 covering add/mul/scale/layernorm/
  softmax/transpose/permute/slice-read/embed-lookup/small-matmul.
- [x] AC-19: f32-to-f16 GPU conversion at 23835 section.
- [x] AC-20: Non-CUDA stubs at the listed line ranges.

## Architecture

### High-level structure

`kernels.rs` is organised by op CATEGORY, not by upstream PyTorch
file. The top-of-file is the f32→f64 converter machinery; the
remainder is a long sequence of category sections, each containing:

1. PTX source string constants (`pub(crate) const *_PTX: &str = "..."`).
2. Hand-written or converter-generated f64 variants (where applicable).
3. The Rust launcher wrappers (`pub fn gpu_*`) — typically one per
   `(op, dtype)` pair.
4. Optional `_into` variants that write into a pre-allocated output.
5. Non-CUDA stubs at the end of the file (one per CUDA-feature
   `pub fn`).

Every PTX kernel is loaded through `crate::module_cache::get_or_compile`
to avoid the ~1700 us per-call compile cost. The kernel-name strings
are `&'static str` (string literals), enabling the cheap `&'static str`
keyed cache path.

### Helper machinery

- `fn validate_binary(a, b, device)` — shared device + length
  validation for two-input ops.
- `fn try_launch_binary(a, b, device, ptx, name)` — standard 1-D
  launcher for binary kernels.
- `fn try_launch_binary_vec4(a, b, device, ptx, name)` — vec4-packed
  launch path used by `gpu_add` / `gpu_mul` when `n >= 16 && n % 4 == 0`.
  On `PtxCompileFailed` it falls back to the scalar path.
- `fn broadcast_strides(input_shape, out_shape)` — compute the
  per-dim broadcast stride array used by the broadcast PTX templates.
- `fn elementwise_launch_dims(n)` — compute `(grid, block)` for a
  standard 1-D launch.
- Section "Launch configuration helper" at 12297 — these helpers.
- Section "Validation helpers" at 12328 — shape/device checks.
- Section "PTX kernel launch helpers" at 12380 — the `try_launch_*`
  family.
- Section "f64 launch helpers" at 12740 — f64-specialised variants.

### f32 → f64 PTX auto-conversion

`pub(crate) fn ptx_f32_to_f64` (line 39) mechanically translates an
f32 PTX kernel string into its f64 equivalent through ~30 chained
`.replace(...)` substitutions:

- **Register types**: `.reg .f32 → .reg .f64`.
- **Memory ops**: `ld.global.f32 → ld.global.f64`, etc.
- **Arithmetic**: `add.f32 → add.f64`, `fma.rn.f32 → fma.rn.f64`,
  `mul.f32 → mul.f64`, etc.
- **Byte offsets**: `shl.b64 %off, %off, 2 → shl.b64 %off, %off, 3`
  (f32→4B, f64→8B). Covers `%off`, `%off_in`, `%off_out`, `%off_src`,
  `%off_dst`, `%off_a`, `%off_b`, `%row_off` (the broadcast / row /
  gather / scatter offset registers).
- **Transcendentals**: `rcp.approx.f32 → rcp.rn.f64`,
  `sqrt.approx.f32 → sqrt.rn.f64`. There is no `*.approx.f64` in PTX.
- **Atomic ops**: `atom.global.add.f32 → atom.global.add.f64`.
- **Target arch**: `.target sm_52 → .target sm_60` because
  `atom.add.f64.global` requires sm_60+.
- **Special-value seeds**: `mov.b32 %acc, 0xFF800000 → mov.b64 %acc,
  0xFFF0000000000000` (the `-inf` seed used by cummax init), and the
  `+inf` 0x7F800000 → 0x7FF0000000000000 sibling. Documented at
  lines 83-97 — pre-this-fix the kernels returned subnormal
  denormals because the seed never made it through the converter
  intact (issue #787).
- **Float literals**: `0f00000000 → 0d0000000000000000` (the f32 hex
  for `0.0` → f64 hex), and the 8 other common constants at
  lines 156-165 (1.0, -1.0, 2.0, 0.5, ±inf, log2(e), ln(2)).

`get_f64_ptx` (line 173) caches the converted string in a
`OnceLock<String>` per kernel so the conversion only happens once
per process.

### Category sections

Each category section has the structure: PTX constants, launcher
wrappers, optional `_into` variants. The 50+ sections are listed in
the AC-2 through AC-20 mapping above with their starting line
numbers.

The "Public API -- *" comments at e.g. line 13099 ("binary ops"),
13188 ("broadcast binary ops"), 13311 ("unary ops"), etc. delimit
the category boundaries.

### Non-test production consumers

`backend_impl.rs` is the single primary consumer: 171
`crate::kernels::*` call sites across its trait method bodies, each
forwarding the type-erased `GpuBufferHandle` arguments to the
corresponding typed launcher in this file. Selected sites:

- `backend_impl.rs:1330` — `CudaBackendImpl::add_f32` → `kernels::gpu_add`.
- `backend_impl.rs:1445` — `add_f64` → `kernels::gpu_add_f64`.
- `backend_impl.rs:2037` — `softmax_f64` → `kernels::gpu_softmax_f64`.
- `backend_impl.rs:2846` — `softmax_f32` → `kernels::gpu_softmax`.
- `backend_impl.rs:2859` — `dropout_f32` → `kernels::gpu_dropout`.
- `backend_impl.rs:3136` — `softmax_bf16_f32` → `kernels::gpu_softmax_bf16_f32`.
- `backend_impl.rs:3852` — `softmax_backward_f32` → `kernels::gpu_softmax_backward`.

Through `CudaBackendImpl`, every GPU-resident tensor op in
ferrotorch-core (`Tensor::add`, `Tensor::matmul`, `Tensor::softmax`,
`Tensor::layer_norm`, etc.) dispatches into this file.

## Parity contract

`parity_ops = []` for this route — per-op parity is enforced at the
ferrotorch-core op layer where the parity-sweep runner lives. Each
of the 362 `pub fn gpu_*` entries in this file pairs with a CPU
implementation in `ferrotorch-core` whose parity is verified by the
sweep. The kernels here are the GPU side of that pair.

Selected edge cases preserved (categories common to many kernels):

- **NaN propagation**: f32/f64 arithmetic preserves IEEE-754 NaN
  through `add`, `mul`, etc. Special-case `min` / `max` use
  `max.f32` which propagates NaN per `setp.gt.f32` semantics.
- **±Inf seed for cummax / cummin**: lifted through the auto-converter
  (issue #787).
- **Empty input**: every launcher's grid is sized with `.max(1)` and
  every kernel has a `setp.ge.u32` out-of-bounds guard.
- **Vec4 fast path fallback**: `gpu_add` / `gpu_mul` try
  `try_launch_binary_vec4` first; on `PtxCompileFailed` they fall
  back to the scalar kernel. This preserves correctness on devices
  that reject the vec4 PTX (rare in practice — sm_52+ supports
  128-bit loads).
- **f64 atomic add target arch**: lifted to sm_60 automatically by
  the converter for kernels using `atom.global.add.f64` (used in
  the layernorm / rmsnorm backward atomics).
- **Broadcast stride math**: each broadcast kernel computes
  `bcast_a(out_idx)` via per-dim stride lookup; the launcher
  pre-computes the stride array on host and passes it as a u32
  array param.
- **Row-wise softmax stability**: uses the standard
  `max - exp(x - max) / Σ exp(x - max)` decomposition for
  numerical stability. The bf16 mixed-precision softmax accumulates
  in f32 and writes bf16, matching PyTorch's bf16 dispatch.
- **Cumulative-scan associativity**: Blelloch up-sweep + down-sweep,
  shared-memory backed. Output element i equals the prefix sum of
  elements `[0, i]` (left-inclusive, matching PyTorch's
  `inclusive=True` default).

## Verification

Unit tests in `ferrotorch-gpu/src/kernels.rs` `mod tests` (39
tests, gated `#[cfg(test)] #[cfg(feature = "cuda")]`) exercise: each
elementwise op against a CPU reference, broadcast ops on
representative shape pairs, cumulative scans against running-sum
references, softmax / log_softmax against the upstream NumPy formula,
layernorm / rmsnorm against a CPU reference, and embedding lookup
against a hand-computed expected output. All run through the
`GpuDevice::new(0)` graceful-skip pattern.

The fuller integration is in
`ferrotorch-gpu/tests/conformance_gpu_backend.rs` (the trait-surface
conformance suite) and `ferrotorch-core/tests/conformance_elementwise.rs`
(which routes through `CudaBackendImpl` when CUDA is available).

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda kernels:: 2>&1 | tail -3
```

Expected: ≥ 1 `test result: ok` line.

The full crate test pass:

```bash
cargo test -p ferrotorch-gpu --features cuda 2>&1 | tail -5
```

Per-op parity is the responsibility of `ferrotorch-core`'s op
modules (which the parity-sweep runner exercises). The GPU paths in
this file PASS by sharing numerical agreement with the CPU paths in
`ferrotorch-core` to within f32 epsilon (verified by the
conformance tests).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub(crate) fn ptx_f32_to_f64 in ferrotorch-gpu/src/kernels.rs` (line 39), `pub(crate) fn get_f64_ptx` (line 173). Non-test consumer: every f64 op in this file that delegates to an f32 template (e.g., the f64 versions of layernorm_backward / rmsnorm_backward / broadcast_add at the line ranges noted in the auto-converter's documenting comments). |
| REQ-2 | SHIPPED | impl: `pub fn gpu_add` (line 13114), `gpu_sub` (13146), `gpu_mul` (13168); non-test consumer: `backend_impl.rs:1330` (`add_f32`), 1346 (`mul_f32`), 1334 (`sub_f32`), 1372 (`div_f32`). |
| REQ-3 | SHIPPED | impl: `gpu_broadcast_add/sub/mul/div` at lines 13200-13284; non-test consumer: trait methods in `backend_impl.rs` lines 1322-1395 forward to these for broadcast inputs. |
| REQ-4 | SHIPPED | impl: 10 unary `pub fn gpu_*` at lines 13325-13412 (neg, relu, abs, exp, log, sqrt, pow, sigmoid, tanh) plus the broader transcendental section at 17768; non-test consumer: trait methods at `backend_impl.rs:1358-1437` (relu, neg, exp, log, sqrt, pow, abs, sigmoid, tanh f32 + f64 variants). |
| REQ-5 | SHIPPED | impl: GELU / SiLU / ELU / Mish + backward sections (1354, 2295-3555, 4763-4870); non-test consumer: `backend_impl.rs:1612-1740` trait methods (gelu, gelu_tanh, gelu_erf, silu, elu, mish, clamp + backwards). |
| REQ-6 | SHIPPED | impl: reductions at lines 14191-14982 + cumulative-scan ops at 15158+; non-test consumer: `backend_impl.rs` reduce / sum / prod / min / max trait methods. |
| REQ-7 | SHIPPED | impl: `gpu_cumsum/cumprod/cummax/cummin/logcumsumexp` at lines 15174-15598; non-test consumer: `backend_impl.rs` `cumsum_f32`/`cumprod_f32`/etc. trait methods routed through `CudaBackendImpl`. |
| REQ-8 | SHIPPED | impl: `gpu_softmax` / `gpu_log_softmax` + backwards at lines 13885-14191; non-test consumer: `backend_impl.rs:2837, 2028, 3852, 3136` (softmax_f32, softmax_f64, softmax_backward_f32, softmax_bf16_f32). |
| REQ-9 | SHIPPED | impl: LayerNorm / RMSNorm + backwards at lines 21424-21744; non-test consumer: `backend_impl.rs` `layernorm_f32` / `layernorm_backward_f32` / `rmsnorm_f32` / `rmsnorm_backward_f32` trait methods + f64 variants. |
| REQ-10 | SHIPPED | impl: `gpu_index_select_1d` (13427), `gpu_scatter_add_1d` (13506), `gpu_index_select_dim` (13596), `gpu_index_select_dim_f64` (13692); non-test consumer: `backend_impl.rs` `index_select_*` / `scatter_add_*` trait methods (the legacy f32-encoded path; the modern integer-index path is in `gather_int.rs`). |
| REQ-11 | SHIPPED | impl: `gpu_masked_fill` (13789), `gpu_masked_zero` (13867); non-test consumer: `backend_impl.rs` `masked_fill_f32` / `masked_zero_f32` trait methods (the legacy f32-mask path; the modern u8-mask path is in `masked_kernels.rs`). |
| REQ-12 | SHIPPED | impl: strided ops at lines 15615-16678; non-test consumer: `backend_impl.rs` `strided_split` / `strided_cat` / `transpose_2d` / `permute_0213` trait methods. |
| REQ-13 | SHIPPED | impl: embedding / KV-cache section at 16948-17171; non-test consumer: `backend_impl.rs` `embed_lookup_*` / `slice_write_*` / `slice_read_*` trait methods (f32 + f64). |
| REQ-14 | SHIPPED | impl: `gpu_small_matmul` (16837 section), MaxPool2d / AvgPool2d (21030 section), BatchNorm2d (21286 section); non-test consumer: `backend_impl.rs` `matmul_f32` (2578), `conv2d_f32` (2502), pooling and batchnorm trait methods. |
| REQ-15 | SHIPPED | impl: `gpu_dropout` (16503 section); non-test consumer: `backend_impl.rs:2859` (`dropout_f32`), `2864` (`dropout_philox_f32`). |
| REQ-16 | SHIPPED | impl: bf16 binary / axis-reduction / activation PTX at lines 9448-9866, dispatch functions at 26477-26884; non-test consumer: `backend_impl.rs:3161` (`add_bf16_f32`) and the broader bf16 trait method block. |
| REQ-17 | SHIPPED | impl: `gpu_fused_adam_step` (20776 section), fused GRU cell (20905 section); non-test consumer: `ferrotorch-optim` Adam optimizer routes through the fused Adam trait method on `GpuBackend` for the CUDA case. |
| REQ-18 | SHIPPED | impl: `_into` helpers section at line 12603 covering 10+ ops; non-test consumer: `ferrotorch-core` `inplace.rs` op dispatch + graph-capture-safe variants in `backend_impl.rs` use the `_into` shape to avoid mid-op allocations. |
| REQ-19 | SHIPPED | impl: f32-to-f16 conversion section at line 23835; non-test consumer: `backend_impl.rs` dtype-cast paths route through this for f32 ↔ f16. |
| REQ-20 | SHIPPED | impl: non-CUDA stubs at the line ranges listed in AC-20; non-test consumer: workspace `--no-default-features` CI lane verifies the crate compiles without cuda. |
