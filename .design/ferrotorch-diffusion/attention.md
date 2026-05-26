# Multi-head attention + Transformer2DModel

<!--
tier: 3-component
status: draft
baseline-pytorch: /home/doll/pytorch (HEAD)
upstream-paths:
  - diffusers/src/diffusers/models/attention_processor.py
  - diffusers/src/diffusers/models/attention.py
  - diffusers/src/diffusers/models/transformers/transformer_2d.py
-->

## Summary

The cross-attention stack used by the SD-1.5 UNet's CrossAttn blocks.
Mirrors diffusers's `attention_processor.Attention`,
`attention.FeedForward` (GEGLU), `attention.BasicTransformerBlock`,
and `transformers/transformer_2d.Transformer2DModel` 1:1 — including
the `to_out.0`/`net.0.proj`/`net.2` state-dict prefix convention
that exists because diffusers wraps these in `Sequential`s.

## Requirements

- REQ-1: `Attention::new(query_dim, cross_attention_dim, heads,
  dim_head, bias)` builds q/k/v + out projections matching the SD-1.5
  UNet (`bias = false` on q/k/v, `bias = true` on `to_out.0`).
  `cross_attention_dim = None` selects self-attention (`kv_dim =
  query_dim`); `Some(d)` selects cross-attention (`kv_dim = d`).
- REQ-2: `Attention::forward_xattn(hidden, encoder_hidden_states?)`
  computes scaled-dot-product multi-head attention:
  `softmax(q @ k^T / sqrt(dim_head)) @ v`, with the heads collapsed
  into the batch axis (`B*H`) for the `bmm` path. Output shape
  `[B, N, query_dim]`.
- REQ-3: `FeedForward::new(dim, mult)` constructs the GEGLU FFN:
  `net.0.proj = Linear(dim, 2*dim_ff)`, `net.2 = Linear(dim_ff,
  dim)` where `dim_ff = dim * mult`. Forward: split into `(x, gate)`,
  `out = net.2(x * gelu(gate))`.
- REQ-4: `BasicTransformerBlock::new(dim, heads, dim_head,
  cross_attention_dim)` chains
  `LayerNorm → self-attn → +residual → LayerNorm → cross-attn →
  +residual → LayerNorm → GEGLU → +residual`. State-dict layout:
  `norm1.*`, `attn1.*`, `norm2.*`, `attn2.*`, `norm3.*`, `ff.*`.
- REQ-5: `Transformer2DModel::new(in_channels, heads, dim_head,
  num_layers, cross_attention_dim, norm_num_groups)` wraps
  `GroupNorm → proj_in (Conv2d k=1) → reshape to [B, HW, inner] →
  num_layers × BasicTransformerBlock → reshape back →
  proj_out (Conv2d k=1) → + residual` to plug into the UNet's
  spatial CrossAttn paths.
- REQ-6: `BasicTransformerBlock::forward` and
  `Transformer2DModel::forward` (the `Module<T>` trait method) error
  out with `InvalidArgument` because they require
  `encoder_hidden_states`; callers must use the explicit
  `forward_xattn` method.

## Acceptance Criteria

- [x] AC-1: `attention_self_shape` and `attention_cross_shape` pass
  (`attention.rs:849..879`).
- [x] AC-2: `feedforward_shape_and_keys` confirms the state-dict
  layout (`attention.rs:881..901`).
- [x] AC-3: `basic_transformer_block_shape` cycles a tensor through
  the full pre-LN + self-attn + cross-attn + FF stack
  (`attention.rs:903..920`).
- [x] AC-4: `transformer_2d_shape` runs the wrapper end-to-end
  (`attention.rs:922..939`).
- [x] AC-5: `transformer_2d_named_parameters` confirms diffusers
  state-dict prefixes (`attention.rs:941..958`).

## Architecture

- `Attention<T>` (`attention.rs:58..78`): five `Linear<T>` fields
  (q/k/v + out) plus `dim_head`, `heads`, `inner_dim`, `query_dim`,
  `kv_dim`, and a cached `scale = 1/sqrt(dim_head)`.
- `Attention::new` (`attention.rs:94..121`): allocates the four
  projections with the diffusers bias contract (`bias` on q/k/v,
  always-on for `to_out.0`).
- `Attention::forward_xattn` (`attention.rs:133..209`): the core
  multi-head attention recipe. Reshapes `q` to `[B*H, N, D]`,
  `k`/`v` to `[B*H, S, D]`, computes `q @ k^T * scale → softmax →
  @ v`, then merges heads back to `[B, N, inner]` and projects out.
- `FeedForward<T>` (`attention.rs:314..323`): two `Linear<T>` +
  `GELU`. The `GEGLU` decomposition lives inside `forward`
  (`attention.rs:346..365`): `chunk(2, dim=-1)` to split into `(x,
  gate)`, then `net.2(x * gelu(gate))`.
- `BasicTransformerBlock<T>` (`attention.rs:441..457`): three
  `LayerNorm`s, two `Attention`s, one `FeedForward`. Constructor
  (`attention.rs:465..490`) wires the SD-1.5 contract: self-attn has
  `cross_attention_dim = None`, cross-attn has `Some(cad)`, both
  have `bias=false` on q/k/v.
- `BasicTransformerBlock::forward_xattn` (`attention.rs:499..525`):
  pre-LN + sub-block + residual, three times.
- `Transformer2DModel<T>` (`attention.rs:644..657`): one
  `GroupNorm`, one `proj_in` Conv2d (k=1), `Vec<BasicTransformerBlock>`,
  one `proj_out` Conv2d (k=1). SD-1.5 v1 uses
  `use_linear_projection=False` so both projections are Convs.
- `Transformer2DModel::forward_xattn` (`attention.rs:709..751`):
  norm → proj_in → permute/reshape to sequence layout → run
  transformer blocks → reshape back → proj_out → + residual.

State-dict layout (every block's `named_parameters` /
`load_state_dict`) mirrors diffusers exactly: `to_out.0.*` (not
`to_out.*`), `net.0.proj.*` + `net.2.*` (not `net.0.*` + `net.1.*`),
`transformer_blocks.{i}.*`. The non-standard prefixes exist because
upstream diffusers wraps these layers in `Sequential`s; we preserve
the keys to load HF checkpoints directly.

Non-test production consumers:

- `ferrotorch-diffusion/src/unet.rs:47` imports
  `Transformer2DModel`; `unet.rs:116` (CrossAttnDownBlock2D),
  `unet.rs:469` (UNetMidBlock2DCrossAttn), and `unet.rs:664`
  (CrossAttnUpBlock2D) all call `Transformer2DModel::<T>::new` for
  each cross-attn level.
- `Attention`, `BasicTransformerBlock`, and `FeedForward` are
  consumed inside `Transformer2DModel` (their containing module
  inside `attention.rs`), which is a production consumer of each.

## Parity contract

`parity_ops = []`. Numerical contract: byte-equivalence with
`diffusers.models.attention_processor.Attention` + `attention.BasicTransformerBlock`
+ `transformers.transformer_2d.Transformer2DModel` when loaded with
the SD-1.5 UNet HF checkpoint. Edge cases:

- `hidden_states.shape() != [B, N, query_dim]` → `ShapeMismatch`.
- `encoder_hidden_states.shape() != [B, S, kv_dim]` →
  `ShapeMismatch`.
- `forward` (the trait method) without explicit
  `encoder_hidden_states` → `InvalidArgument` on
  `BasicTransformerBlock` and `Transformer2DModel` since
  cross-attention requires it.

## Verification

Six lib tests in `attention.rs:844..959`:

- `attention_self_shape` / `attention_cross_shape` — self/cross
  attention shape preservation.
- `feedforward_shape_and_keys` — GEGLU shape + state-dict prefixes.
- `basic_transformer_block_shape` — full pre-LN + 3-sub-block cycle.
- `transformer_2d_shape` — full spatial wrapper.
- `transformer_2d_named_parameters` — diffusers state-dict layout
  including `transformer_blocks.0.*`.

No parity-sweep ops apply.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `Attention::new` at `ferrotorch-diffusion/src/attention.rs:94..121`; non-test consumer: `BasicTransformerBlock::new` at `ferrotorch-diffusion/src/attention.rs:475..477` calls `Attention::<T>::new` for both self-attn and cross-attn, and is itself consumed by `Transformer2DModel` in production |
| REQ-2 | SHIPPED | impl: `Attention::forward_xattn` at `ferrotorch-diffusion/src/attention.rs:133..209`; non-test consumer: `BasicTransformerBlock::forward_xattn` at `ferrotorch-diffusion/src/attention.rs:515` and `519` calls `self.attn1.forward_xattn` / `self.attn2.forward_xattn` |
| REQ-3 | SHIPPED | impl: `FeedForward::new` at `ferrotorch-diffusion/src/attention.rs:331..342`; non-test consumer: `BasicTransformerBlock::new` at `ferrotorch-diffusion/src/attention.rs:479` calls `FeedForward::<T>::new(dim, 4)?` |
| REQ-4 | SHIPPED | impl: `BasicTransformerBlock::new` at `ferrotorch-diffusion/src/attention.rs:465..490` and `forward_xattn` at `ferrotorch-diffusion/src/attention.rs:499..525`; non-test consumer: `Transformer2DModel::new` at `ferrotorch-diffusion/src/attention.rs:683..688` constructs `BasicTransformerBlock` instances and the forward pass at `attention.rs:740` invokes them |
| REQ-5 | SHIPPED | impl: `Transformer2DModel::new` at `ferrotorch-diffusion/src/attention.rs:669..699` and `forward_xattn` at `ferrotorch-diffusion/src/attention.rs:709..751`; non-test consumer: `ferrotorch-diffusion/src/unet.rs:116`, `unet.rs:469`, and `unet.rs:664` all call `Transformer2DModel::<T>::new` to build cross-attn levels |
| REQ-6 | SHIPPED | impl: error returns at `ferrotorch-diffusion/src/attention.rs:529..535` (`BasicTransformerBlock::forward`) and `attention.rs:755..761` (`Transformer2DModel::forward`); non-test consumer: the strict-typestate-style guard surfaces a clear error to any production caller that forgets to supply `encoder_hidden_states` |
