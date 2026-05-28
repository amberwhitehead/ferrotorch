# ferrotorch-vision — `models::swin` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: /home/doll/.local/lib/python3.13/site-packages/torchvision/models/swin_transformer.py
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/swin_transformer.py
-->

## Summary

`ferrotorch-vision/src/models/swin.rs` ships Swin Transformer Tiny
(Liu et al. 2021, the swin_t variant) with shifted-window self-attention
end-to-end. Phase 11 (#998) closes this with a torchvision-exact
`named_parameters` schema so the strict value-parity loader can ingest
a torchvision swin_t state dict without per-key remap. The eval-mode
forward is autograd-correct everywhere except the cyclic-shift `roll`,
which requires `no_grad` (training-mode `roll` backward tracked as
#1014).

## Requirements

- REQ-1: A private `PatchEmbed<T: Float>` ships
  `Conv2d(3, embed_dim, k=patch_size, s=patch_size, bias=true) → permute
  to [B, H', W', C] → LayerNorm(embed_dim, eps=1e-5)`. Children index
  `0=conv`, `2=norm` (mirrors torchvision's
  `nn.Sequential(Conv2d, Permute, LayerNorm)` where Permute=1 is
  parameter-free).
- REQ-2: A private `ShiftedWindowAttention<T: Float>` ships
  window-based MHSA with cyclic-shift support and learned
  `relative_position_bias_table` (shape `[(2*ws-1)^2, num_heads]`). A
  precomputed `relative_position_index: Vec<i64>` is used to gather the
  bias per query/key pair (NOT a `Parameter` — failure mode #36).
  Forward implements torchvision's `shifted_window_attention(...)`
  step-for-step: pad → cyclic shift → window partition (via
  `view().permute().contiguous().view()` — NEVER `data_vec()`, failure
  mode #15) → fused QKV → bias add → mask add → softmax → matmul →
  reverse window partition → reverse cyclic shift → unpad.
- REQ-3: A private `Mlp<T: Float>` ships the 2-layer
  `Linear(dim, 4*dim) → GELU → Linear(4*dim, dim)`.
- REQ-4: `pub struct SwinBlock<T: Float>` carries `norm1`, `attn:
  ShiftedWindowAttention`, `norm2`, `mlp`. Forward:
  `LN → attn → +residual → LN → MLP → +residual`.
- REQ-5: A private `PatchMerging<T: Float>` ships the inter-stage
  reduction: `LayerNorm(4*dim) → Linear(4*dim, 2*dim, bias=false)`
  preceded by the `_patch_merging_pad` sampling pattern. Halves
  spatial; doubles channels.
- REQ-6: `pub struct SwinTransformer<T: Float>` carries:
  `features.0` = `PatchEmbed`,
  `features.{1,3,5,7}` = stages of `SwinBlock` (stage configurations
  per swin_t: depths `[2, 2, 6, 2]`, heads `[3, 6, 12, 24]`),
  `features.{2,4,6}` = `PatchMerging`,
  `norm: LayerNorm(8 * embed_dim, eps=1e-5)`,
  `head: Linear(8 * embed_dim, num_classes)`.
- REQ-7: `Module::forward` for `SwinTransformer<T>` runs every
  `features.<i>` in order, then `mean_dim` over the H,W axes to get
  `[B, C]`, then `norm`, then `head`.
- REQ-8: Per-block `shift_size` alternates `0` and `floor(window_size /
  2) = 3`. A runtime guard forces `shift_size = 0` when the spatial dim
  is `≤ window_size` (matches torchvision's
  `if window_size[0] >= pad_H: shift_size[0] = 0`).
- REQ-9: `named_parameters` returns torchvision-exact paths:
  `features.0.0.<n>` (PatchEmbed conv), `features.0.2.<n>` (PatchEmbed
  norm),
  `features.<i_stage>.<j>.norm{1,2}.<n>` (SwinBlock norms),
  `features.<i_stage>.<j>.attn.{qkv,proj}.<n>` (SwinBlock attention),
  `features.<i_stage>.<j>.attn.relative_position_bias_table`,
  `features.<i_stage>.<j>.mlp.{0,3}.<n>` (SwinBlock MLP),
  `features.<i_merge>.{norm,reduction}.<n>` (PatchMerging),
  `norm.<n>`, `head.<n>`.
- REQ-10: `pub fn swin_tiny` is the canonical constructor.

## Acceptance Criteria

- [x] AC-1: `swin_tiny::<f32>(1000)` constructs and forwards on
  `[1, 3, 224, 224]` returning `[1, 1000]` (covered by the
  conformance-vision tests).
- [x] AC-2: `named_parameters` includes the torchvision-shaped prefixes
  listed in REQ-9 (covered by the tests in `mod tests`).
- [x] AC-3: `relative_position_bias_table` exists as a `Parameter` of
  shape `[(2*ws-1)^2, num_heads]` per block.
- [x] AC-4: For the small spatial 7×7 stage, the shift-size guard
  forces `effective_shift = 0` so no cyclic shift is applied.
- [x] AC-5: The cyclic-shift `roll` works under `no_grad` (eval-mode
  value parity); training-mode parity is tracked by the open
  prerequisite blocker.

## Architecture

`PatchEmbed<T: Float>` runs Conv (3→96, kernel=4, stride=4) →
permute→contiguous (NCHW→NHWC) → LayerNorm. Child indexing `0=conv,
2=norm` matches torchvision's
`nn.Sequential(Conv2d, Permute, LayerNorm)` where the Permute slot
(index 1) has no parameters.

`ShiftedWindowAttention<T: Float>` is the cornerstone primitive. The
parameter set matches torchvision's keys exactly:
`qkv.{weight,bias}`, `proj.{weight,bias}`,
`relative_position_bias_table`. The `relative_position_index` field is
deliberately NOT a Parameter — it's a precomputed `Vec<i64>` (length
`ws^2 * ws^2`), serialized in torchvision as an int64 buffer that the
safetensors payload deliberately omits (failure mode #36).

`build_relative_position_bias` uses
`ferrotorch_core::grad_fns::indexing::index_select_dim` to gather the
bias (autograd-correct: `IndexSelectDimBackward` scatter-adds into
`relative_position_bias_table.grad()`). The CUDA path uses the #1098
`gpu_index_select_dim` kernel.

The `forward` impl implements torchvision's
`shifted_window_attention(...)` line-by-line, with the window-partition
chain using `view().permute().contiguous().view()` and the head-split
using `narrow(0, idx, 1).contiguous().view(...)`. The
`compute_relative_position_index` helper builds the `Vec<i64>` once at
construction.

`pub struct SwinBlock<T: Float>` runs the standard pre-norm
ViT-shaped block with `ShiftedWindowAttention` substituted for
`MultiheadAttention`. `Mlp<T: Float>` ships the 2-Linear MLP.
`PatchMerging<T: Float>` halves spatial via the
`_patch_merging_pad` sampling pattern (concat of 4 strided slices)
and projects to 2× channels via `Linear(4*dim, 2*dim, bias=false)`.

`pub struct SwinTransformer<T: Float>` stores `features` as a
`Vec<FeatureChild<T>>` where each child is `PatchEmbed`, a
`Sequential<SwinBlock>`, or `PatchMerging`. Module-tree paths use the
flat `features.<i>` layout matching torchvision exactly.

### Non-test production consumers

- `pub use swin::{SwinBlock, SwinTransformer, swin_tiny}` re-export at
  `ferrotorch-vision/src/models/mod.rs`.
- `default_registry()` registers `"swin_tiny"` via
  `maybe_load_pretrained` at `registry.rs`.

## Parity contract

`parity_ops = []`. Swin composes `Conv2d`, `LayerNorm`, `Linear`,
`GELU`, the differentiable `add` / `mul` / `cat` / `mean_dim` /
`softmax` / `index_select_dim` / `matmul_differentiable`, plus the
non-differentiable `roll` (under `no_grad`). No new op surface.

Edge cases preserved versus torchvision:

- **Shift-size guard**: `if window_size >= pad_H: shift_size = 0` —
  applied per-block per dim. For swin_t at 224 input, only the final
  7×7 stage triggers this guard.
- **`_patch_merging_pad` reflection sampling**: pad reflection on
  `(0, 0, 0, W % 2, 0, H % 2)`, then take strided slices `x[..., 0::2,
  0::2, :]` and `x[..., 1::2, 0::2, :]` etc., concatenate along
  `dim=-1` to land at 4*C channels.
- **`relative_position_bias_table.shape == [(2*ws-1)^2, num_heads]`**:
  matches torchvision's `nn.Parameter(torch.zeros(...))` initialization.
- **`relative_position_index` is NOT a Parameter**: it's a precomputed
  Vec<i64>. The fixture descriptor lists it under
  `skipped_int_buffer_keys` for the safetensors loader.
- **`roll` is `no_grad`**: `ferrotorch_core::ops::tensor_ops::roll` is a
  CPU `data_vec()` cycle that returns `requires_grad=false`. Training
  forward through `ShiftedWindowAttention` requires a future
  device-resident `roll` with backward (tracked separately). For the
  eval-mode value-parity contract this is fine.

## Verification

Tests in `mod tests` in `swin.rs` cover the per-component construction,
forward shape, named-parameter prefixes, and the small spatial 7×7
shift-guard. The `conformance_vision_models.rs` test exercises the full
`swin_tiny` value-parity load.

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib swin:: 2>&1 | tail -3
```

Expected: all tests pass; no parity-sweep ops.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `struct PatchEmbed<T: Float>` + `Module<T>` impl in `swin.rs`; non-test consumer: `SwinTransformer::new` constructs it as `features.0` in `swin.rs`. |
| REQ-2 | SHIPPED | impl: `struct ShiftedWindowAttention<T: Float>` + `Module<T>` impl + `build_relative_position_bias` + `build_attn_mask` in `swin.rs`; non-test consumer: every `SwinBlock::new` constructs one in `swin.rs`. |
| REQ-3 | SHIPPED | impl: `struct Mlp<T: Float>` + `Module<T>` impl in `swin.rs`; non-test consumer: every `SwinBlock::new` constructs one in `swin.rs`. |
| REQ-4 | SHIPPED | impl: `pub struct SwinBlock<T: Float>` + `Module<T>` impl in `swin.rs`; non-test consumer: `SwinTransformer::new` builds 4 stages of SwinBlocks in `swin.rs`. |
| REQ-5 | SHIPPED | impl: `struct PatchMerging<T: Float>` + `Module<T>` impl in `swin.rs`; non-test consumer: `SwinTransformer::new` constructs 3 PatchMergings as `features.{2,4,6}` in `swin.rs`. |
| REQ-6 | SHIPPED | impl: `pub struct SwinTransformer<T: Float>` + `SwinTransformer::new` in `swin.rs`; non-test consumer: `default_registry()` constructs it via `maybe_load_pretrained` at `registry.rs`. |
| REQ-7 | SHIPPED | impl: `Module::forward` for `SwinTransformer<T>` in `swin.rs`; non-test consumer: trait method invoked through `Box<dyn Module<T>>` returned from `registry.rs::get_model`. |
| REQ-8 | SHIPPED | impl: the `effective_shift = if ws >= height || ws >= width { 0 } else { self.shift_size }` guard inside `ShiftedWindowAttention::forward` in `swin.rs`; non-test consumer: the final 7×7 stage of swin_tiny (constructed by `default_registry()` at `registry.rs`) runs through this guard at inference time. |
| REQ-9 | SHIPPED | impl: `named_parameters` for `PatchEmbed`, `ShiftedWindowAttention`, `Mlp`, `SwinBlock`, `PatchMerging`, `SwinTransformer` in `swin.rs`; non-test consumer: `load_state_dict(&state_dict, false)` at `named_parameters in registry.rs` walks the result. |
| REQ-10 | SHIPPED | impl: `pub fn swin_tiny` in `swin.rs`; non-test consumer: `default_registry()` invokes it at `registry.rs`. |
