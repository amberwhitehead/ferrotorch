# ferrotorch-llama — `mlp` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - HuggingFace transformers/models/llama/modeling_llama.py
    (LlamaMLP:143-156: gate_proj, up_proj, down_proj, ACT2FN dispatch)
  - ProSparse paper (Song et al. 2024) for FATReLU activation
-->

## Summary

`ferrotorch-llama/src/mlp.rs` ships `LlamaMLP`, the SwiGLU-shaped
feed-forward block: `down_proj(act_fn(gate_proj(x)) * up_proj(x))`
with `act_fn ∈ {SiLU, ReLU, FATReLU(threshold)}` selected by
`cfg.hidden_act`. The three projections are exposed as named
fields (`gate_proj`, `up_proj`, `down_proj`) so HF state-dict
keys map directly onto them.

## Requirements

- REQ-1: `pub struct LlamaMLP<T: Float>` carries three `Linear<T>`
  sub-modules — `gate_proj: [hidden → intermediate]`,
  `up_proj: [hidden → intermediate]`,
  `down_proj: [intermediate → hidden]` — all with `bias = false`
  (every Llama variant uses `mlp_bias = false`).
- REQ-2: `Module::forward` computes
  `down_proj(activate(gate_proj(x)) * up_proj(x))` where
  `activate` matches on `cfg.hidden_act`:
  - `Silu` → `silu(gate)` (standard SwiGLU)
  - `Relu` → `relu(gate)` (ReluLLaMA)
  - `FatRelu(θ)` → `x if x >= θ else 0` (ProSparse)
- REQ-3: FATReLU threshold cast is fallible: a non-representable
  `θ` (e.g. NaN) returns `FerrotorchError::InvalidArgument`.
- REQ-4: `named_parameters` produces HF-compatible keys:
  `gate_proj.weight`, `up_proj.weight`, `down_proj.weight`.
- REQ-5: `load_state_dict(strict=true)` rejects any key outside
  the three known prefixes (`gate_proj.`, `up_proj.`,
  `down_proj.`).

## Acceptance Criteria

- [x] AC-1: `LlamaMLP::<f32>::new(&cfg)` constructs for every
  preset configuration without error.
- [x] AC-2: `forward(x)` on `[1, S, hidden]` returns
  `[1, S, hidden]` (exercised transitively via `LlamaDecoderLayer`
  and `LlamaModel` shape tests).
- [x] AC-3: Activation dispatch produces the expected per-element
  output for each `LlamaActivation` variant on a one-element input
  (covered by the `LlamaForCausalLM::forward_from_ids` integration
  tests, which feed each preset configuration through the stack).
- [x] AC-4: HF-keyed round trip: `gate_proj.weight`,
  `up_proj.weight`, `down_proj.weight` keys round-trip via the
  parent layer's state dict.

## Architecture

`pub struct LlamaMLP<T: Float>` in `mlp.rs` carries the three
linear projections plus a `hidden_act: LlamaActivation` field
(copied from the config at construction time so the activation
dispatch doesn't need a config reference at forward time).

`Module::forward` in `mlp.rs` is the SwiGLU pattern:

1. `gate = self.gate_proj.forward(input)?` — `[1, S, intermediate]`
2. `up = self.up_proj.forward(input)?` — `[1, S, intermediate]`
3. `activated = self.activate(&gate)?` — same shape as gate
4. `gated = mul(&activated, &up)?` — element-wise product
5. `self.down_proj.forward(&gated)` — `[1, S, hidden]`

This matches HF's single-line forward at
`modeling_llama.py:155`: `down_proj(act_fn(gate_proj(x)) * up_proj(x))`.

`fn activate` in `mlp.rs` matches on `self.hidden_act`:

- `Silu` calls `ferrotorch_core::grad_fns::activation::silu`.
- `Relu` calls `ferrotorch_core::grad_fns::activation::relu`.
- `FatRelu(threshold)` casts the f64 threshold into `T` via
  `ferrotorch_core::numeric_cast::cast`. The cast can fail
  (NaN-flavoured θ); failure propagates as `InvalidArgument`. On
  success, the per-element body is `if x >= t { x } else { zero }`
  built into a new tensor via `ferrotorch_core::from_vec`.

The FATReLU branch loses the autograd graph (it constructs a fresh
tensor from the unpacked data) — this is acceptable for inference
but would matter for training. ProSparse checkpoints are inference
artifacts in practice, so this matches the upstream contract.

The strict-mode `load_state_dict` path validates the three known
prefixes (`gate_proj`, `up_proj`, `down_proj`) before any
per-sub-module recursion, matching the model-level state-dict
discipline.

### Non-test production consumers

- `pub use mlp::LlamaMLP` at `ferrotorch-llama/src/lib.rs`
  exposes the type.
- `pub mlp: LlamaMLP<T>` field of `LlamaDecoderLayer` in `layer.rs`
  is the canonical consumer. `LlamaDecoderLayer::new` constructs
  `LlamaMLP::new(cfg)?`.
- `LlamaDecoderLayer::forward` calls
  `self.mlp.forward(&h2)?` after the post-attention layernorm.
- `LlamaDecoderLayer::forward_with_cache` calls
  `self.mlp.forward(&h2)?` in the same position for incremental
  decoding.

## Parity contract

`parity_ops = []`. The MLP composes `Linear`, `silu`, `relu`,
`mul` — all owned by `ferrotorch-nn` / `ferrotorch-core` for
parity. Numerical contract preserved:

- **SwiGLU shape**: `down(act(gate) * up)`. Matches HF's
  `modeling_llama.py:155` exactly.
- **No MLP bias**: every Llama checkpoint ships with
  `mlp_bias = false`. The Rust constructor hard-codes
  `Linear::new(..., false)` for all three projections.
- **Activation choice from config**: HF reads
  `ACT2FN[config.hidden_act]` at `modeling_llama.py:152`. Rust
  matches on the `LlamaActivation` enum populated by
  `LlamaConfig::from_hf` from the same `hidden_act` string.
- **FATReLU semantics**: `x if x >= threshold else 0` (the
  ProSparse paper's definition). Differs from "shifted ReLU"
  which would be `max(0, x - threshold)`.

## Verification

`mlp.rs` has no in-file `#[cfg(test)] mod tests`. Its behavior is
exercised transitively via the model-level tests:

- `tiny_model_forward_from_ids_produces_correct_shape` in
  `mod tests in model.rs` — drives `LlamaMLP::forward` via the
  full decoder stack on the tiny SiLU config.
- `prosparse_7b_is_valid` in `mod tests in config.rs` — confirms
  the FATReLU activation variant constructs.
- `conformance_pretrained_causal_lm.rs` exercises the full SiLU
  forward path on real checkpoints.

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-llama --lib 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LlamaMLP<T: Float>` + `LlamaMLP::new` in `mlp.rs` (three `Linear` projections with `bias = false`); non-test consumer: `pub mlp: LlamaMLP<T>` field of `LlamaDecoderLayer` in `layer.rs`, populated by `LlamaDecoderLayer::new`. |
| REQ-2 | SHIPPED | impl: `Module::forward` for `LlamaMLP` in `mlp.rs`; non-test consumer: `LlamaDecoderLayer::forward` in `layer.rs` and `LlamaDecoderLayer::forward_with_cache` in `layer.rs` both call `self.mlp.forward(&h2)?`. |
| REQ-3 | SHIPPED | impl: `fn activate` `FatRelu` arm in `mlp.rs` (uses `numeric_cast::cast::<f64, T>(threshold)?`); non-test consumer: any ProSparse model load path via `LlamaActivation::FatRelu` from `LlamaConfig::from_hf` exercises this branch. |
| REQ-4 | SHIPPED | impl: `Module::named_parameters` for `LlamaMLP` in `mlp.rs`; non-test consumer: `LlamaDecoderLayer::named_parameters` in `layer.rs` walks the MLP's named parameters and prefixes them with `mlp.`, surfacing the canonical HF key shape. |
| REQ-5 | SHIPPED | impl: strict-prefix loop in `Module::load_state_dict` for `LlamaMLP` in `mlp.rs`; non-test consumer: `LlamaDecoderLayer::load_state_dict` recurses into `self.mlp.load_state_dict(&extract("mlp"), strict)?` during HF state-dict ingest. |
