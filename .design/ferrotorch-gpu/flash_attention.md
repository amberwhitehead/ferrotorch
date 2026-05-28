# GPU FlashAttention (online softmax + shared-memory tiles)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/transformers/cuda/attention.cu
  - aten/src/ATen/native/transformers/cuda/attention_backward.cu
  - aten/src/ATen/native/transformers/cuda/flash_attn
  - aten/src/ATen/native/transformers/cuda/sdp_utils.cpp
-->

## Summary

`ferrotorch-gpu/src/flash_attention.rs` is a from-scratch FlashAttention
implementation in PTX (no Tri-Dao / cutlass dependency). It computes
`softmax(Q @ K^T / sqrt(d)) @ V` **without** materialising the full
`N x N` attention matrix — peak memory is `O(N)` instead of `O(N^2)`.

The kernel uses the online-softmax algorithm from
[FlashAttention](https://arxiv.org/abs/2205.14135): each thread keeps
running `max` (`m`) and `sum` (`l`) statistics, processes `TILE_K`-wide
key/value tiles cooperatively through shared memory, and accumulates
the weighted V contribution into a register-resident output vector.

This mirrors upstream PyTorch's `at::_scaled_dot_product_attention`
dispatch on the GPU path. PyTorch's actual flash-attn-2 / cutlass-fmha
backends are extensive third-party libraries; ferrotorch ships a
self-contained PTX kernel that handles the most common head-dim
regimes (`d ≤ 128`) and falls back to a tiled matmul + softmax + matmul
fallback for larger head dims. (`scaled_dot_product_attention` is
exposed at the ferrotorch-nn level; this file is the GPU-resident
implementation.)

## Requirements

- REQ-1: `pub fn gpu_flash_attention_f32` — single entry point.
  Takes `q, k, v: &CudaBuffer<f32>` (`[B, H, N, d]`), output
  `[B, H, N, d]`. Optional causal mask flag and explicit attn-mask
  buffer. Returns `CudaBuffer<f32>`.
- REQ-2: `pub fn gpu_flash_attention_f64` — f64 mirror of REQ-1.
- REQ-3: PTX kernel layout — each CUDA thread handles one query
  position. Q is loaded into registers; K/V tiles are cooperatively
  loaded into shared memory. `TILE_K = 32` rows of `K` + `TILE_K`
  rows of `V`; shared memory budget `2 * 32 * 128 * 4 = 32 KiB`,
  fitting comfortably in the 48 KiB default shared-mem limit on
  all architectures from sm_52 upwards.
- REQ-4: Online softmax — within the tile loop, the kernel updates
  running `m_new = max(m_old, tile_max)` and `l_new = exp(m_old -
  m_new) * l_old + sum(exp(qk_tile - m_new))`. The output
  accumulator is rescaled by `exp(m_old - m_new)` before adding
  the tile's contribution. Mirrors the FlashAttention paper
  algorithm 1, step 11-14.
- REQ-5: Causal masking — when `causal: bool` is true, the kernel
  applies an upper-triangular mask (`qk[i, j] = -inf` for `j > i`)
  before softmax. Implemented as a per-tile predicate.
- REQ-6: Explicit attention mask — optional `attn_mask:
  Option<&CudaBuffer<f32>>` parameter; when present, added to
  `qk` before softmax. Supports torch's additive-mask convention.
- REQ-7: GPU-resident — all three phases (load Q to registers,
  iterate K/V tiles, write output) happen on-device. Zero CPU
  round-trips per `rust-gpu-discipline §3`.
- REQ-8: No-CUDA stubs — `cfg(not(feature = "cuda"))` returns
  `GpuError::NoCudaFeature`.

## Acceptance Criteria

- [x] AC-1: All 8 in-file `#[test]` units pass under
  `cargo test -p ferrotorch-gpu --features cuda flash_attention::`.
- [x] AC-2: Output of `gpu_flash_attention_f32` matches a naive
  `softmax(Q @ K^T / sqrt(d)) @ V` reference to within `1e-4`
  (rtol) for representative `[B, H, N, d]` tuples — pinned by the
  shape-correctness tests.
- [x] AC-3: Causal mask produces a strictly lower-triangular
  attention pattern (verified by hand-computed reference).
- [x] AC-4: Explicit attn-mask is added correctly.
- [x] AC-5: No-CUDA stub returns `NoCudaFeature`.

## Architecture

### Entry points (REQ-1, REQ-2)

`pub fn gpu_flash_attention_f32 in flash_attention.rs` (line 460):

1. Compute kernel launch dims: grid `[B * H, N]`, block
   `[TILE_K]`.
2. Allocate output `CudaBuffer<f32>` of shape `[B, H, N, d]`.
3. Launch the PTX kernel `flash_attention_f32_kernel` (defined in
   the file's PTX string constant).
4. Return the output buffer.

`pub fn gpu_flash_attention_f64 in flash_attention.rs` (line 1023)
is the f64 mirror with the f64 PTX kernel.

Non-test consumer at `backend_impl.rs` (f32) and `backend_impl.rs` (f64)
— the cuda backend's `scaled_dot_product_attention` arm. The
public re-export is at `lib.rs:212` (`pub use
flash_attention::{gpu_flash_attention_f32, gpu_flash_attention_f64}`).

### Kernel layout (REQ-3, REQ-4)

Per the module-level `//!` doc-comment (lines 1-29):

```text
shared memory layout:
  K_tile: [TILE_K x d]  f32  (TILE_K = 32)
  V_tile: [TILE_K x d]  f32
total: 2 * 32 * 128 * 4 = 32 KiB
```

Each thread:
1. Loads its query vector (`d` floats) into registers.
2. Initialises `m = -inf`, `l = 0`, `o[d] = 0`.
3. Iterates over K/V tiles:
   - Cooperatively loads `K_tile` and `V_tile` into shared mem.
   - `__syncthreads()`.
   - Computes `qk[j] = dot(q, K_tile[j]) / sqrt(d)` for j in
     `[0, TILE_K)`.
   - (Optionally applies causal / explicit mask.)
   - Computes `m_new = max(m, max(qk))`.
   - Rescales `l *= exp(m - m_new)` and `o *= exp(m - m_new)`.
   - Updates `l += sum(exp(qk - m_new))` and
     `o += sum(exp(qk - m_new) * V_tile[j])`.
   - Sets `m = m_new`.
   - `__syncthreads()`.
4. Normalises: `o /= l`.
5. Writes `o` to global memory.

### Causal masking (REQ-5)

When `causal == true`, the kernel sets `qk[j] = -inf` for any
`j > query_idx` (the query position) before the max / sum
accumulation. Implemented as a single predicate in the tile loop.

### Explicit attention mask (REQ-6)

When `attn_mask.is_some()`, the kernel loads the `[B, H, N, N]`
mask tile alongside `K_tile` and adds the per-(query, key) value
to `qk` before softmax. This matches torch's additive-mask
convention (where `-inf` means masked-out, `0.0` means visible).

### GPU-resident (REQ-7)

All buffers stay on device. The output is allocated on-device and
returned to the caller as a `CudaBuffer<f32>` — no host copy in
the hot path.

### No-CUDA stubs (REQ-8)

`#[cfg(not(feature = "cuda"))] pub fn gpu_flash_attention_f32 in
flash_attention.rs` (line 653) returns
`Err(GpuError::NoCudaFeature)`. f64 mirror at line 1189.

## Parity contract

`parity_ops = []` for this module. Reason: scaled-dot-product-attention
is an op-level entry in `ferrotorch-nn` and `ferrotorch-core`'s
parity surface (`sdpa`, `flash_attention`); the cuda dispatcher is
reached transitively when the GPU backend arm is selected.

Edge cases mirrored from upstream:

- **`d > 128`** (very large head dim): the kernel's shared-memory
  budget assumes `d ≤ 128`. For larger head dims, the caller is
  expected to fall back to the naive tiled matmul + softmax +
  matmul path. Mirrors upstream's dispatcher in
  `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp` which
  picks between flash / mem-eff / math kernels by head dim.
- **`N == 0`** (empty sequence): kernel grid is empty; output is
  empty.
- **Causal + non-square `Q.shape[2] != K.shape[2]`**: caller is
  responsible (cross-attention causal is ill-defined and the
  upstream raises). Wrapper does not enforce.
- **NaN / Inf in Q, K, or V**: propagated through the kernel
  arithmetic. The online softmax handles `-inf` masked positions
  cleanly because `exp(-inf) = 0`.

## Verification

Tests in `#[cfg(all(test, feature = "cuda"))] mod tests in
flash_attention.rs` (8 functions at lines 1294, 1359, 1407, 1444,
1481, 1535, 1576, 1598):

- Shape correctness for small `[B=1, H=1, N=8, d=8]` against naive
  reference.
- Larger `[B=2, H=4, N=64, d=64]` round-trip.
- Causal-mask hand-computed reference.
- Explicit attn-mask propagation.
- d=128 boundary case.
- f64 mirror tests.

Smoke commands:

```bash
cargo test -p ferrotorch-gpu --features cuda flash_attention:: 2>&1 | tail -3
cargo build -p ferrotorch-gpu --no-default-features 2>&1 | tail -3
```

Expected: 8 tests pass under cuda; no-cuda compile succeeds.
`parity_ops = []` — no per-op parity-sweep applies at this layer.

## REQ status table

Per S5 (existing pub-API grandfather): `gpu_flash_attention_f32` and
`gpu_flash_attention_f64` are exported from `lib.rs:212` and
consumed by `backend_impl.rs` (the cuda backend's
SDPA dispatch arm). That arm is reached from
`ferrotorch-nn::scaled_dot_product_attention` when the dispatch
routes to GPU.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn gpu_flash_attention_f32 in flash_attention.rs` (line 460). Non-test consumer: `backend_impl.rs` (cuda backend's SDPA f32 arm). Also re-exported at `lib.rs`. |
| REQ-2 | SHIPPED | impl: `pub fn gpu_flash_attention_f64 in flash_attention.rs` (line 1023). Non-test consumer: `backend_impl.rs` (cuda backend's SDPA f64 arm). Also re-exported at `lib.rs`. |
| REQ-3 | SHIPPED | impl: PTX kernel string in `flash_attention.rs` documents `TILE_K = 32`, `d_max = 128`, 32 KiB shared-mem budget; the module `//!` doc-comment at lines 19-29 pins the layout. Non-test consumer: both `gpu_flash_attention_f32` (line 460) and `_f64` (line 1023) launch this kernel; downstream consumers at `_f64 in backend_impl.rs,5058`. |
| REQ-4 | SHIPPED | impl: online-softmax `m`/`l` accumulator + rescale logic is in the PTX kernel string; documented in the module `//!` doc-comment at lines 8-17. Non-test consumer: every call through `gpu_flash_attention_f32/f64` exercises the online-softmax path; `gpu_flash_attention_f32 in backend_impl.rs` is the production consumer. |
| REQ-5 | SHIPPED | impl: causal-mask predicate inside the PTX kernel, activated by the `causal: bool` parameter on `gpu_flash_attention_f32/f64`. Non-test consumer: `gpu_flash_attention_f32 in backend_impl.rs` passes through the user's `causal` flag from `ferrotorch-nn::scaled_dot_product_attention`. |
| REQ-6 | SHIPPED | impl: optional `attn_mask: Option<&CudaBuffer<f32/f64>>` parameter on `gpu_flash_attention_f32/f64`; PTX kernel branches on a mask-pointer-not-null check. Non-test consumer: `gpu_flash_attention_f32 in backend_impl.rs` passes the optional mask through from `ferrotorch-nn`. |
| REQ-7 | SHIPPED | impl: both `gpu_flash_attention_f32/f64` allocate output on device via `CudaBuffer::empty` and return without host pull. Non-test consumer: `empty in backend_impl.rs` keeps the buffer on-device for downstream GPU consumers. |
| REQ-8 | SHIPPED | impl: `#[cfg(not(feature = "cuda"))] pub fn gpu_flash_attention_f32 in flash_attention.rs` (line 653) and `_f64` (line 1189) return `Err(GpuError::NoCudaFeature)`. Non-test consumer: the same `_f64 in backend_impl.rs` SDPA arm under the no-cuda compile path. |
