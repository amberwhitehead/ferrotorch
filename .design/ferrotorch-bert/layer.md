# ferrotorch-bert — `layer` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - /home/doll/pytorch/torch/ (Linear / LayerNorm / GELU at
    torch/nn/modules/linear.py, normalization.py, activation.py;
    HF BertLayer at
    huggingface/transformers/src/transformers/models/bert/modeling_bert.py)
-->

## Summary

`ferrotorch-bert/src/layer.rs` ships the FFN half of a BERT encoder
block (`BertIntermediate` + `BertOutput`) plus the full block
`BertLayer` that composes the attention sub-block with the FFN. The
FFN uses the exact-erf GELU (HF BERT default) and the residual is the
POST-norm pattern `LayerNorm(attn_out + dense(GELU(linear(attn_out))))`.

## Requirements

- REQ-1: `pub struct BertIntermediate<T: Float>` carries a
  `ferrotorch_nn::Linear<T>` (`[hidden -> intermediate]`, bias) plus a
  `ferrotorch_nn::GELU` activation. `Module::forward` returns
  `GELU(Linear(input))`.
- REQ-2: `pub struct BertOutput<T: Float>` carries a
  `Linear<T>` (`[intermediate -> hidden]`, bias) and a `LayerNorm<T>`
  (`[hidden]`, eps from config). `forward_residual(intermediate,
  attn_out)` returns
  `LayerNorm(attn_out + dense(intermediate))` — the POST-norm FFN
  residual.
- REQ-3: `pub struct BertLayer<T: Float>` composes
  `BertAttention<T>` + `BertIntermediate<T>` + `BertOutput<T>` and
  exposes a single `Module::forward(input)` that runs the full
  encoder block.
- REQ-4: `Module::forward` on `BertOutput` performs ONLY the dense
  projection (the residual partner is unavailable). The full
  sub-block uses `forward_residual`; the `Module::forward` fallback
  keeps the trait satisfied.
- REQ-5: HF state-dict layout for `BertLayer`:
  `attention.{...}` (recurses into `BertAttention`),
  `intermediate.dense.{weight,bias}`,
  `output.{dense.{weight,bias}, LayerNorm.{weight,bias}}`. Loadable
  in strict mode without rewriting keys.
- REQ-6: GELU is the exact erf-based variant (matches HF
  `BertIntermediate` default = `gelu`, not `gelu_new`); validated by
  the config layer rejecting any other activation name on load.

## Acceptance Criteria

- [x] AC-1: `BertLayer::<f32>::new(&tiny_cfg)` constructs.
- [x] AC-2: `BertLayer::forward([1, 5, 8])` returns `[1, 5, 8]` with
  finite values.
- [x] AC-3: `BertLayer::named_parameters()` exposes the HF-layout
  keys (`attention.self.query.weight`, `attention.output.dense.weight`,
  `attention.output.LayerNorm.weight`, `intermediate.dense.weight`,
  `output.dense.weight`, `output.LayerNorm.weight`, etc.).

## Architecture

`pub struct BertIntermediate<T: Float>` in `layer.rs` packages the
expansion linear and the GELU. Its `Module::forward` is the
straightforward `GELU(Linear(input))` pipeline; `Module::parameters`
delegates to the inner linear (GELU has no trainable parameters).

`pub struct BertOutput<T: Float>` in `layer.rs` packages the reduction
linear and the LayerNorm. `forward_residual` is the canonical entry
point; `Module::forward` is the dense-only shim for trait satisfaction.

`pub struct BertLayer<T: Float>` in `layer.rs` is the full block.
Its `Module::forward` calls:

1. `self.attention.forward(input)` — produces the post-norm attention
   output `[1, S, hidden]`.
2. `self.intermediate.forward(&attn_out)` — produces the
   GELU-activated expansion `[1, S, intermediate]`.
3. `self.output.forward_residual(&inter, &attn_out)` — produces the
   post-norm FFN output `[1, S, hidden]`.

The recursive `load_state_dict` strict-mode check rejects any key
whose prefix is not in `{attention, intermediate, output}` so an
unknown HF schema field cannot silently land into a parameter.

### Non-test production consumers

- `pub use layer::{BertIntermediate, BertLayer, BertOutput}` at
  `ferrotorch-bert/src/lib.rs:89`.
- `pub layer: Vec<BertLayer<T>>` field of `pub struct BertEncoder`
  at `BertEncoder in ferrotorch-bert/src/model.rs`; `BertEncoder::new` at
  `ferrotorch-bert/src/model.rs:35` constructs them in a loop;
  `BertEncoder`'s `Module::forward` at
  `ferrotorch-bert/src/model.rs:46` iterates through them.

## Parity contract

`parity_ops = []`. The layer composes `linear`, `gelu`, and
`layer_norm` (covered by `ferrotorch-nn` parity) plus
`BertAttention` (covered by `attention.md`).

Numerical / structural edge cases preserved:

- **Exact-erf GELU.** Matches HF `BertIntermediate` default `gelu`
  (`torch.nn.functional.gelu(approximate="none")`). The
  `gelu_new` (tanh approximation) is intentionally not implemented;
  the config layer rejects it.
- **POST-norm FFN residual.** `LayerNorm(attn_out + dense(intermediate))`
  matches HF BERT and is the same convention as the attention
  sub-block — both norms come AFTER the residual add.
- **LayerNorm `eps = cfg.layer_norm_eps`** (default `1e-12` for HF
  BERT).

## Verification

Tests in `mod tests in layer.rs`:

- `layer_forward_shape`
- `layer_named_parameters_match_hf_layout`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-bert --lib layer:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct BertIntermediate<T: Float>` + its `Module<T>` impl in `layer.rs`; non-test consumer: field `intermediate` of `pub struct BertLayer` in `layer.rs`; `BertLayer::Module::forward` in `layer.rs` calls `self.intermediate.forward(&attn_out)`. |
| REQ-2 | SHIPPED | impl: `BertOutput::forward_residual` in `layer.rs`; non-test consumer: `BertLayer::Module::forward` in `layer.rs` calls `self.output.forward_residual(&inter, &attn_out)`. |
| REQ-3 | SHIPPED | impl: `pub struct BertLayer<T: Float>` + its `Module<T>` impl in `layer.rs`; non-test consumer: element of `pub layer: Vec<BertLayer<T>>` at `ferrotorch-bert/src/model.rs:22`; `BertEncoder::new` at `ferrotorch-bert/src/model.rs:35` and `BertEncoder::Module::forward` at `ferrotorch-bert/src/model.rs:46` consume them. |
| REQ-4 | SHIPPED | impl: `BertOutput::Module::forward` (dense-only) in `layer.rs`; non-test consumer: kept reachable through the `pub use` at `ferrotorch-bert/src/lib.rs:89` (the `Module` blanket trait surface). |
| REQ-5 | SHIPPED | impl: `named_parameters` / `load_state_dict` for `BertLayer` in `layer.rs`; non-test consumer: `BertEncoder::load_state_dict` at `ferrotorch-bert/src/model.rs:105` recurses through `layer.{i}.*`. |
| REQ-6 | SHIPPED | impl: `BertIntermediate::new` in `layer.rs` constructs `GELU::new()` (exact-erf); non-test consumer: the `gelu` activation choice is enforced upstream by `HfBertConfig::validate` in `ferrotorch-bert/src/config.rs` which rejects any other activation name on load. |
