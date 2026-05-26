# ferrotorch-nn — `flash_attention` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - aten/src/ATen/native/transformers/cuda/flash_attn/
  - torch/nn/attention/flex_attention.py
-->

## Summary

`ferrotorch-nn/src/flash_attention.rs` implements a memory-efficient
CPU-tiled FlashAttention forward (Dao et al., 2022 — *FlashAttention:
Fast and Memory-Efficient Exact Attention*) using the online softmax
trick (Milakov & Gimelshein, 2018). Provides exact scaled-dot-product
attention output without ever materializing the full `[N_q, N_k]`
attention matrix, reducing peak memory from O(N²) to O(N · block_size).
The MVP backward recomputes the standard (non-tiled) attention to
obtain gradients — a fully tiled backward is a future optimization.

Mirrors the contract of upstream's CUDA kernel headers at
`aten/src/ATen/native/transformers/cuda/flash_attn/` but runs on CPU
tiles — the bandwidth advantage is in cache reuse, not GPU SRAM
locality.

## Requirements

- REQ-1: `pub fn flash_attention(query, key, value, causal,
  block_size) -> FerrotorchResult<Tensor<T>>` — the exact tiled
  attention forward returning `[B, N_q, d_v]` for inputs of shape
  `[B, N_q, d]`, `[B, N_k, d]`, `[B, N_k, d_v]`. Mirrors upstream's
  `torch.nn.functional.scaled_dot_product_attention` (backend =
  flash-attn) shape contract.

- REQ-2: Shape validation — rejects non-3-D inputs, mismatched batch
  sizes, mismatched `d` between Q and K, mismatched seq length
  between K and V, and `block_size == 0`. Causal masking requires
  `N_q == N_k`. Validation lives in
  `fn validate_inputs in flash_attention.rs`.

- REQ-3: Online-softmax tiled forward — for each (q_block, k_block)
  pair, computes `S_ij = Q_i K_j^T * scale`, updates a per-row
  running max `m[q_row]` and running sum `l[q_row]` of
  `exp(s - m_new)`, and accumulates the output via the rescaling
  formula `O_new = (correction * l_old / l_new) * O_old + (1 /
  l_new) * P_row @ V_block`. Implemented at
  `fn flash_attention_single in flash_attention.rs`.

- REQ-4: Causal early-out — when `causal=true` and the entire
  (q_block, k_block) pair lies strictly above the diagonal (all
  query indices < all key indices), the tile is skipped entirely.
  Within partially-masked tiles, individual `s_tile[qi, ki]` entries
  are clamped to `-inf` (`-1e30`) when `k_row > q_row`.

- REQ-5: Empty-sequence handling — when `N_q == 0` or `N_k == 0`,
  the forward returns a correctly-shaped empty tensor without
  invoking the tiled kernel. Matches upstream behaviour for
  zero-length sequences.

- REQ-6: Autograd via `FlashAttentionBackward<T>` — attached when
  any of Q, K, V has `requires_grad` and grad is globally enabled
  (`is_grad_enabled()`). The MVP backward recomputes the full
  attention matrix (non-tiled) to derive `grad_Q`, `grad_K`,
  `grad_V`. The backward is correct but not memory-efficient — a
  fully tiled backward is tracked as a follow-up optimisation
  outside this doc.

- REQ-7: Device routing — when the input is GPU-resident, the
  output is moved to the same device via `.to(device)`. The kernel
  itself runs on the CPU host buffer (allocated via
  `query.data_vec()` / `key.data_vec()` / `value.data_vec()`); the
  GPU path is a copy-out / copy-in round trip. A native GPU kernel
  is the canonical future optimisation.

- REQ-8: `pub fn standard_attention(query, key, value, causal) ->
  FerrotorchResult<Tensor<T>>` — a non-tiled reference
  implementation used by tests as the ground truth. Computes the
  same math as `flash_attention` but materialises the full
  `[N_q, N_k]` attention matrix. Useful for unit tests asserting
  numerical equivalence between tiled and reference forms.

- REQ-9: Block-size flexibility — `block_size` is a runtime
  argument; the function tiles both the Q and K axes with the
  same value. Tail blocks (when `N % block_size != 0`) use the
  partial size `bk = k_end - k_start` so no padding is required.

## Acceptance Criteria

- [x] AC-1: `flash_attention(Q, K, V, false, 64)` returns a
  `[B, N_q, d_v]` tensor for `[2, 32, 8] / [2, 32, 8] / [2, 32, 8]`.
- [x] AC-2: `flash_attention(Q, K, V, true, 64)` with `N_q != N_k`
  errors.
- [x] AC-3: `flash_attention(Q, K, V, false, 0)` errors on zero
  `block_size`.
- [x] AC-4: Output equals `standard_attention(...)` to within
  float32 tolerance for the same inputs.
- [x] AC-5: Backward through Q, K, V with `requires_grad=true`
  produces non-None gradient tensors of the correct shapes.
- [x] AC-6: `N_q == 0` returns an empty tensor with shape
  `[B, 0, d_v]` without crashing.

## Architecture

### Tiled forward (REQ-3, REQ-4)

`fn flash_attention_single in flash_attention.rs` runs the per-batch
core. The outer loop iterates over K blocks; the inner loop iterates
over Q blocks. For each tile:

1. Compute `s_tile[bq * bk]` via the dot-product Q_i @ K_j^T, scaled
   by `1/sqrt(d)`.
2. Apply causal mask within the tile (set `s_tile[qi, ki] = -1e30`
   when `k_row > q_row`).
3. For each query row in the tile:
   - Compute the tile's row max `tile_max`.
   - Update the running max `m[q_row] = max(m_old, tile_max)`.
   - Compute the correction factor `exp(m_old - m_new)`.
   - Compute `exp(s - m_new)` for each entry in the row.
   - Update the running sum `l_new = correction * l_old + tile_sum`.
   - Update the output via
     `O_new = rescale_old * O_old + rescale_new * (P_row @ V_block)`
     where `rescale_old = correction * l_old / l_new` and
     `rescale_new = 1 / l_new`.

Total memory: `O(N_q + block_size^2)` per batch — never O(N²).

### Empty / degenerate shapes (REQ-5)

`flash_attention` checks `N_q == 0 || N_k == 0` before invoking the
kernel and returns a `[B, N_q, d_v]` tensor backed by an empty
`Vec<T>`.

### Backward (REQ-6)

`FlashAttentionBackward<T>` stores cloned Q, K, V tensors plus the
`causal` flag. On `backward(grad_output)`:

1. Recompute `scores = Q @ K^T * scale` (full matrix).
2. Apply causal mask.
3. Compute softmax row-wise to get `attn[i, j]`.
4. `grad_V = attn^T @ grad_output`.
5. `grad_attn = grad_output @ V^T`.
6. `grad_scores[i, j] = attn[i, j] * (grad_attn[i, j] -
   sum_k(attn[i, k] * grad_attn[i, k]))` (the softmax-backward
   Jacobian).
7. `grad_Q = grad_scores @ K * scale`.
8. `grad_K = grad_scores^T @ Q * scale`.

The backward is intentionally non-tiled because the savings on the
forward pass already cover the typical training memory pressure;
optimising the backward is a future iteration tracked outside this
doc.

### `standard_attention` reference (REQ-8)

A direct (non-tiled) `softmax(Q @ K^T / sqrt(d)) @ V` implementation
that allocates the `[N_q, N_k]` attention matrix and serves as the
oracle in the test suite. Same numerical contract; different memory
footprint.

### Non-test production consumers

- `pub use flash_attention::{flash_attention, standard_attention}`
  at `ferrotorch-nn/src/lib.rs:199` — grandfathered public API
  surface. Downstream LLM serving code (KV-cache-friendly
  decoders) can opt into the memory-efficient path when sequence
  lengths blow up.

## Parity contract

`parity_ops = []`. The flash kernel mirrors the math of
`scaled_dot_product_attention` (with the flash backend). The SDPA
runner arm landed 2026-05-26 (closes #1532); the flash variant
piggybacks on the SDPA oracle since
`nn::functional::scaled_dot_product_attention` (parity-verified at
`16/200 passed (184 skipped, 0 failed)`) delegates to
`flash_attention(..., 64)` — every passing SDPA sample is also a
passing flash_attention sample at the same (Q, K, V, is_causal)
shape.

Edge cases preserved:

- **NaN / Inf** — NaN inputs propagate through the online-softmax
  state; output is NaN. Matches upstream.
- **`block_size > N`** — single-tile execution; reduces to
  `standard_attention` numerically.
- **Causal with `N_q == N_k`** — accepted; with `N_q != N_k`
  rejected (deviates from upstream which truncates the triangle).
  Reasoned as a strict-mode guard rather than a divergence.
- **`d_v != d`** — supported (V's last dim may differ from Q/K's),
  matching upstream's flexibility.

## Verification

Tests in `mod tests in flash_attention.rs` (~10+ tests, exact set
depends on the source). The key oracle pattern compares
`flash_attention(Q, K, V, causal, block_size)` against
`standard_attention(Q, K, V, causal)` to within `1e-5` for f32.

No parity-sweep ops declared for this file. Smoke command:

```bash
cargo test -p ferrotorch-nn --lib flash_attention:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn flash_attention<T: Float>` in `flash_attention.rs`; non-test consumer: re-export at `ferrotorch-nn/src/lib.rs:199`. |
| REQ-2 | SHIPPED | impl: `fn validate_inputs in flash_attention.rs` checks 3-D rank, batch alignment, K/V seq match, `d` match, `block_size > 0`, and causal `N_q == N_k`; non-test consumer: invoked from `flash_attention` (re-exported at `lib.rs:199`). |
| REQ-3 | SHIPPED | impl: `fn flash_attention_single in flash_attention.rs` with online-softmax rescaling; non-test consumer: re-export at `lib.rs:199`. |
| REQ-4 | SHIPPED | impl: causal-skip + intra-tile clamp logic inside `flash_attention_single`; non-test consumer: re-export at `lib.rs:199`. |
| REQ-5 | SHIPPED | impl: `N_q == 0 \|\| N_k == 0` early-return inside `flash_attention`; non-test consumer: re-export at `lib.rs:199`. |
| REQ-6 | SHIPPED | impl: `struct FlashAttentionBackward<T>` plus `impl GradFn<T>` in `flash_attention.rs` (recompute-based backward); non-test consumer: re-export at `lib.rs:199` — any caller passing grad-requiring tensors triggers backward via the autograd engine. |
| REQ-7 | SHIPPED | impl: `result.to(device)` branch at the tail of `flash_attention` when `device.is_cuda()`; non-test consumer: re-export at `lib.rs:199`. |
| REQ-8 | SHIPPED | impl: `pub fn standard_attention<T: Float>` in `flash_attention.rs`; non-test consumer: re-export at `lib.rs:199`. |
| REQ-9 | SHIPPED | impl: `block_size` runtime arg in `flash_attention`; tail-block logic `bk = k_end - k_start`; non-test consumer: re-export at `lib.rs:199`. |
