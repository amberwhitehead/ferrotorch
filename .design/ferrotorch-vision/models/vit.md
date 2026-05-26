# ferrotorch-vision — `models::vit` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: /home/doll/.local/lib/python3.13/site-packages/torchvision/models/vision_transformer.py
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/vision_transformer.py
-->

## Summary

`ferrotorch-vision/src/models/vit.rs` ships the Vision Transformer
(Dosovitskiy et al. 2020) end-to-end, with `vit_b_16` as the canonical
constructor. The patch-embedding `conv_proj` carries `bias=true` to
match torchvision's `vit_b_16` (Phase 4 #999/#1001 closed the prior
`bias=false` divergence). The full cls-token prepend → positional add →
N transformer blocks → final LayerNorm → cls slice → Linear head
pipeline is autograd-correct end-to-end thanks to Pass 2B (#1000), which
migrated the per-batch `data_vec()` patterns to device-resident
`view/permute/expand/cat/narrow/squeeze` primitives.

## Requirements

- REQ-1: `pub struct PatchEmbed<T: Float>` ships a `Conv2d(in_channels,
  embed_dim, patch_size, patch_size, bias=true)` projection. Forward:
  `Conv2d` → `view([B, embed_dim, num_patches])` →
  `permute(0, 2, 1)` → `contiguous` to land in `[B, num_patches,
  embed_dim]`. No `data_vec()` detour.
- REQ-2: `pub struct TransformerBlock<T: Float>` carries `norm1:
  LayerNorm`, `attn: MultiheadAttention`, `norm2: LayerNorm`, `mlp_fc1:
  Linear(embed_dim, mlp_dim)`, `mlp_fc2: Linear(mlp_dim, embed_dim)`,
  `gelu: GELU`. Forward: `LN → MHA → +residual → LN → MLP → +residual`.
  The MLP accepts 3-D `[B, S, C]` input directly (no per-batch slicing).
- REQ-3: `pub struct VisionTransformer<T: Float>` carries
  `patch_embed`, learnable `cls_token: Parameter` (`[1, 1, embed_dim]`),
  learnable `pos_embed: Parameter` (`[1, num_patches+1, embed_dim]`),
  `Vec<TransformerBlock<T>>`, final `norm: LayerNorm`,
  `head: Linear(embed_dim, num_classes)`.
- REQ-4: `Module::forward` for `VisionTransformer<T>` runs: patch
  embed → expand cls_token to `[B, 1, embed_dim]` → `cat` along dim=1
  with the patch sequence → `add` pos_embed (broadcasts batch) → run
  every block → final LN → `narrow(1, 0, 1).squeeze_t(1)` to pull the
  CLS-token features → `head`. All ops are autograd-correct so
  `cls_token` and `pos_embed` receive gradients on backward.
- REQ-5: `named_parameters` returns: `patch_embed.proj.weight`,
  `patch_embed.proj.bias`, `cls_token`, `pos_embed`,
  `blocks.<i>.{norm1,attn,norm2,mlp.fc1,mlp.fc2}.<n>`, `norm.<n>`,
  `head.<n>`.
- REQ-6: `named_children` exposes `patch_embed`, `blocks.<i>`, `norm`,
  `head` so `named_descendants_dyn()` can walk every sub-module.
- REQ-7: `VisionTransformer<T>` implements `IntermediateFeatures<T>`
  exposing `patch_embed`, `embedded`, `block<i>`, `norm`, `head`
  activations.
- REQ-8: `pub fn vit_b_16` is the canonical constructor (image_size=224,
  patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4 →
  ~86M parameters).

## Acceptance Criteria

- [x] AC-1: `PatchEmbed::<f32>::new(3, 768, 16)` forward on
  `[1, 3, 224, 224]` returns `[1, 196, 768]`
  (`test_patch_embed_output_shape`).
- [x] AC-2: PatchEmbed parameter count is `768 * 3 * 16 * 16 + 768`
  (weight + bias) (`test_patch_embed_parameter_count`).
- [x] AC-3: `TransformerBlock::<f32>::new(64, 4, 256)` forward on
  `[1, 10, 64]` returns `[1, 10, 64]`
  (`test_transformer_block_output_shape`).
- [x] AC-4: `vit_b_16::<f32>(1000)` forward on `[1, 3, 224, 224]`
  returns `[1, 1000]` (`test_vit_b16_output_shape`).
- [x] AC-5: `vit_b_16::<f32>(1000)` parameter count is in
  (80 000 000, 90 000 000) (`test_vit_b16_param_count`).
- [x] AC-6: `cls_token` and `pos_embed` have shapes `[1, 1, 768]` and
  `[1, 197, 768]` (`test_vit_cls_token_shape`,
  `test_vit_pos_embed_shape`).
- [x] AC-7: `named_parameters` includes the torchvision-shaped paths
  (`patch_embed.`, `cls_token`, `pos_embed`, `blocks.0.`, `blocks.11.`,
  `norm.`, `head.`) (`test_vit_named_parameters_prefixes`).
- [x] AC-8: Backward through `expand` + `cat` preserves the
  `cls_token` gradient (non-zero on at least one element)
  (`test_vit_cls_token_grad_flows_through_cat`).

## Architecture

`pub struct PatchEmbed<T: Float>` wraps the patchify Conv2d. Forward
uses `view(...).permute(...).contiguous()` instead of the prior
`data_vec()`+manual-index pattern (#996, closes #986/#987). The probe
test `tests/probe_permute_migration.rs` proves element-for-element
equivalence with the legacy CPU loop.

`pub struct TransformerBlock<T: Float>` runs the standard pre-norm
encoder block. The MLP forward is rank-polymorphic so the 3-D `[B, S, C]`
input goes through `Linear(C, mlp_dim)` → `GELU` → `Linear(mlp_dim, C)`
in one dispatch (Pass 2B #1000 migrated callers from per-batch slicing
to a single 3-D dispatch).

`pub struct VisionTransformer<T: Float>`:

```text
input  : [B, 3, H, W]
        ↓ patch_embed                                     # [B, num_patches, embed_dim]
        ↓ expand(cls_token, [B, 1, embed_dim])            # broadcast leading B
        ↓ cat(..., 1)                                     # [B, num_patches+1, embed_dim]
        ↓ add(..., pos_embed)                             # broadcast leading B
        ↓ blocks[0..depth]                                # each TransformerBlock
        ↓ norm
        ↓ narrow(1, 0, 1).squeeze_t(1)                    # [B, embed_dim]
output : Linear(embed_dim, num_classes)                   # [B, num_classes]
```

Every step is autograd-correct: `expand` has `ExpandBackward`, `cat`
has `CatBackward` (scatters grads back to each input), `add` has
`BroadcastAddBackward` (reduces along the broadcast dim so
`pos_embed.grad` has the right shape), and `narrow` has
`NarrowBackward` (zeros for the unsliced positions on backward).
`test_vit_cls_token_grad_flows_through_cat` is the binding regression
test for this autograd path.

### Non-test production consumers

- `pub use vit::{PatchEmbed, TransformerBlock, VisionTransformer, vit_b_16}` re-export at
  `ferrotorch-vision/src/models/mod.rs`.
- `default_registry()` registers `"vit_b_16"` via
  `maybe_load_pretrained` at `registry.rs:179`.

## Parity contract

`parity_ops = []`. ViT composes `Conv2d`, `LayerNorm`,
`MultiheadAttention`, `Linear`, `GELU`, and the differentiable `add` /
`cat` / `expand` / `narrow` / `squeeze` / `view` / `permute` /
`contiguous` ops. All covered upstream. No new op surface.

Edge cases preserved versus torchvision:

- **`conv_proj.bias=true`** matches `nn.Conv2d`'s default. Phase 4
  (#999/#1001) closed the prior `bias=false` divergence that was
  rejecting `conv_proj.bias` as unmapped.
- **`cls_token` and `pos_embed` are zero-initialized**: matches
  torchvision's `nn.Parameter(torch.zeros(...))` initialization
  pattern.
- **CLS slice via `narrow(1, 0, 1).squeeze_t(1)`** is the device-resident
  analog of `x[:, 0, :]`. `NarrowBackward` zeros the unsliced positions
  on backward, matching `Tensor.__getitem__`'s slice semantics.
- **`LayerNorm(eps=1e-6)`** matches torchvision's `nn.LayerNorm`
  default for ViT.

## Verification

Tests in `mod tests` in `vit.rs`:

- `test_patch_embed_{output_shape,batch_2,parameter_count}`
- `test_transformer_block_{output_shape,parameter_count}`
- `test_vit_b16_{output_shape,param_count}`
- `test_vit_{cls_token_shape,pos_embed_shape,named_parameters_prefixes,custom_classes,train_eval,is_send_sync}`
- `test_small_vit_forward`
- `test_vit_cls_token_grad_flows_through_cat` (binding regression test
  for Pass 2B #1000 autograd preservation)

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib vit:: 2>&1 | tail -3
```

Expected: all tests pass; no parity-sweep ops.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct PatchEmbed<T: Float>` + `Module<T>` impl in `vit.rs`; non-test consumer: field of `pub struct VisionTransformer` in `vit.rs`; `VisionTransformer::forward` calls `self.patch_embed.forward(...)`. |
| REQ-2 | SHIPPED | impl: `pub struct TransformerBlock<T: Float>` + `Module<T>` impl in `vit.rs`; non-test consumer: `VisionTransformer::new` builds `Vec<TransformerBlock<T>>` in `vit.rs`. |
| REQ-3 | SHIPPED | impl: `pub struct VisionTransformer<T: Float>` + `VisionTransformer::new` in `vit.rs`; non-test consumer: `default_registry()` constructs `vit_b_16` via `maybe_load_pretrained` at `registry.rs:179`. |
| REQ-4 | SHIPPED | impl: `Module::forward` for `VisionTransformer<T>` in `vit.rs`; non-test consumer: trait method invoked through `Box<dyn Module<T>>` returned from `registry.rs::get_model`. |
| REQ-5 | SHIPPED | impl: `Module::named_parameters` for `VisionTransformer<T>` in `vit.rs`; non-test consumer: `load_state_dict(&state_dict, false)` at `registry.rs:53` walks the result. |
| REQ-6 | SHIPPED | impl: `children` / `named_children` overrides on `PatchEmbed`, `TransformerBlock`, `VisionTransformer` in `vit.rs`; non-test consumer: `apply_bn_buffers_from_state_dict` at `registry.rs:62` walks `named_descendants_dyn()` (no BN buffers for ViT, but the consumer site is real). |
| REQ-7 | SHIPPED | impl: `impl IntermediateFeatures<T> for VisionTransformer<T>` in `vit.rs`; non-test consumer: `pub use feature_extractor::IntermediateFeatures` at `mod.rs`. |
| REQ-8 | SHIPPED | impl: `pub fn vit_b_16` in `vit.rs`; non-test consumer: `default_registry()` invokes it at `registry.rs:182`. |
