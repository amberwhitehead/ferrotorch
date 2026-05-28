# ferrotorch-whisper — `layer` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - /home/doll/pytorch/torch/ (Linear / LayerNorm / GELU at
    torch/nn/modules/linear.py, normalization.py, activation.py;
    HF WhisperEncoderLayer at
    huggingface/transformers/src/transformers/models/whisper/modeling_whisper.py)
-->

## Summary

`ferrotorch-whisper/src/layer.rs` ships one full Whisper encoder
block: `self_attn_layer_norm → self_attn` (PRE-norm residual),
`final_layer_norm → fc1 → GELU → fc2` (PRE-norm FFN residual). The
PRE-norm convention is the structural divergence from `ferrotorch-bert`
(post-norm) and matches the modern transformer convention used by
Llama / Mistral / Whisper.

## Requirements

- REQ-1: `pub struct WhisperEncoderLayer<T: Float>` carries:
  - `self_attn_layer_norm: LayerNorm<T>`
  - `self_attn: WhisperEncoderSelfAttention<T>`
  - `final_layer_norm: LayerNorm<T>`
  - `fc1: Linear<T>` (`[d_model -> encoder_ffn_dim]`, bias=true)
  - `fc2: Linear<T>` (`[encoder_ffn_dim -> d_model]`, bias=true)
  - `activation: GELU` (exact-erf)
  plus a `training: bool` flag.
- REQ-2: `Module::forward` computes:
  1. `attn_residual = input + self_attn(self_attn_layer_norm(input))`
     (PRE-norm self-attention residual)
  2. `layer_out = attn_residual + fc2(GELU(fc1(final_layer_norm(attn_residual))))`
     (PRE-norm FFN residual)
- REQ-3: LayerNorm eps is hard-coded to `1e-5` (the PyTorch default).
  HF Whisper's `config.json` does not carry the eps field; the
  module documents this in a code comment.
- REQ-4: HF state-dict key layout for `WhisperEncoderLayer`:
  `self_attn_layer_norm.{weight,bias}`,
  `self_attn.{q_proj.weight, q_proj.bias, k_proj.weight,
  v_proj.weight, v_proj.bias, out_proj.weight, out_proj.bias}`,
  `final_layer_norm.{weight,bias}`, `fc1.{weight,bias}`,
  `fc2.{weight,bias}`. Loadable in strict mode without rewriting
  keys. The absence of `self_attn.k_proj.bias` matches the
  attention sub-block's bias asymmetry.

## Acceptance Criteria

- [x] AC-1: `WhisperEncoderLayer::<f32>::new(&tiny_cfg)` constructs.
- [x] AC-2: `forward([1, 5, 16])` returns `[1, 5, 16]` with finite
  values.
- [x] AC-3: `named_parameters()` exposes the HF-layout keys
  (`self_attn_layer_norm.weight`, `self_attn.q_proj.weight`,
  `self_attn.k_proj.weight`, `final_layer_norm.weight`,
  `fc1.weight`, `fc2.weight`, etc.).
- [x] AC-4: `named_parameters()` does NOT contain
  `self_attn.k_proj.bias` (HF Whisper convention).

## Architecture

`pub struct WhisperEncoderLayer<T: Float>` in `layer.rs` carries the
five sub-modules plus the GELU activation and training flag. Its
`Module::forward` implementation runs the two PRE-norm residual
blocks in the canonical order. The two `add` calls are the
production consumer for `ferrotorch_core::grad_fns::arithmetic::add`
inside this file.

The eps choice is documented inline:

> HF Whisper uses `layer_norm_eps = 1e-5` (the PyTorch default, not
> BERT's 1e-12). Hard-coded here because Whisper's `config.json`
> does NOT carry the field.

The `load_state_dict` strict path rejects any prefix not in
`{self_attn_layer_norm, self_attn, final_layer_norm, fc1, fc2}` so an
unknown HF schema field cannot silently land into a parameter.

### Non-test production consumers

- `pub use layer::WhisperEncoderLayer` at
  `ferrotorch-whisper/src/lib.rs`.
- `pub layers: Vec<WhisperEncoderLayer<T>>` field of
  `pub struct WhisperEncoder` at
  `WhisperEncoder in ferrotorch-whisper/src/encoder.rs`; `WhisperEncoder::new` at
  `ferrotorch-whisper/src/encoder.rs:170` constructs them in a
  loop; `WhisperEncoder::forward_from_mel` at
  `ferrotorch-whisper/src/encoder.rs:232` iterates through them.

## Parity contract

`parity_ops = []`. The layer composes `linear`, `gelu`, and
`layer_norm` (covered by `ferrotorch-nn` parity) plus
`WhisperEncoderSelfAttention` (covered by `attention.md`).

Numerical / structural edge cases preserved:

- **PRE-norm residual.** Matches HF Whisper. Diverges from BERT
  (post-norm); aligns with Llama / Mistral / the modern LLM
  convention.
- **Exact-erf GELU.** Matches HF Whisper default `gelu`
  (`torch.nn.functional.gelu(approximate="none")`). The
  `gelu_new` (tanh approximation) is intentionally not implemented.
- **`layer_norm_eps = 1e-5` hard-coded.** Documented in the code;
  upstream Whisper config does not carry an eps override field.
- **K projection without bias.** Inherited from the attention
  sub-block; the layer's `named_parameters` block preserves the
  asymmetry.

## Verification

Tests in `mod tests in layer.rs`:

- `layer_forward_shape`
- `named_parameters_match_hf_layout` (also asserts the absence of
  `self_attn.k_proj.bias`)

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-whisper --lib layer:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct WhisperEncoderLayer<T: Float>` + `WhisperEncoderLayer::new` in `layer.rs`; non-test consumer: element of `pub layers: Vec<WhisperEncoderLayer<T>>` at `ferrotorch-whisper/src/encoder.rs:150`. |
| REQ-2 | SHIPPED | impl: `Module::forward` for `WhisperEncoderLayer` in `layer.rs`; non-test consumer: `WhisperEncoder::forward_from_mel` at `ferrotorch-whisper/src/encoder.rs:232` calls `l.forward(&x)` for each layer. |
| REQ-3 | SHIPPED | impl: `let eps = 1e-5_f64;` + the documenting comment block above it at the top of `WhisperEncoderLayer::new` in `layer.rs`; non-test consumer: same call path as REQ-1 (the eps is baked into the LayerNorm sub-modules constructed by `new`). |
| REQ-4 | SHIPPED | impl: `named_parameters` / `load_state_dict` for `WhisperEncoderLayer` in `layer.rs`; non-test consumer: `WhisperEncoder::load_state_dict` at `ferrotorch-whisper/src/encoder.rs:421` recurses through `layers.{i}.*`. |
