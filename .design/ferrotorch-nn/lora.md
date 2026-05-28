# ferrotorch-nn — `lora` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/linear.py
-->

## Summary

`ferrotorch-nn/src/lora.rs` implements `LoRALinear<T>` — a Low-Rank
Adaptation wrapper around a `Linear<T>` layer per Hu et al. 2021
("LoRA: Low-Rank Adaptation of Large Language Models"). LoRA is not
itself a PyTorch upstream layer (it's a PEFT technique implemented
externally to `torch.nn`), but its mathematical contract — freeze
`W`, train `A` and `B`, compose as `y = x @ W^T + x @ (B @ A)^T *
(alpha/r) + b` — is a single-source-of-truth in the paper that
ferrotorch implements directly. No parity ops; this is a
ferrotorch-native composition over `Linear`.

## Requirements

- REQ-1: `pub struct LoRALinear<T: Float>` carrying a frozen `base:
  Linear<T>`, two trainable `Parameter<T>` matrices `lora_a` of
  shape `[r, in_features]` and `lora_b` of shape `[out_features, r]`,
  a scaling factor `alpha: f64`, the rank `r: usize`, an optional
  `dropout: Option<Dropout<T>>`, and `training: bool`.
- REQ-2: `LoRALinear::new(base, rank, alpha, dropout_p)` validates
  `rank > 0`, allocates `lora_a` initialized from
  `N(0, 1/sqrt(rank))`, allocates `lora_b` initialized to zeros (so
  the LoRA contribution is initially zero — training starts from the
  pretrained checkpoint). Builds the optional `Dropout` layer only
  when `dropout_p > 0`. Reject invalid `dropout_p` via
  `Dropout::new`'s error.
- REQ-3: Forward — `<LoRALinear<T> as Module<T>>::forward` computes
  `base_out + (input_after_dropout @ A^T @ B^T) * (alpha / r)` via
  the differentiable autograd primitives `transpose_2d`,
  `mm_differentiable`, `mul`, and `add` from `ferrotorch_core::
  grad_fns`.
- REQ-4: `parameters()` returns ONLY `[&lora_a, &lora_b]` — the
  frozen base weights are excluded so optimizers skip them. This is
  THE invariant that distinguishes LoRA from a regular Linear
  composition; verified by `test_parameters_only_lora`.
- REQ-5: `merge()` — folds the LoRA contribution into the base
  weight via `W' = W + (alpha/r) * B @ A` then resets `lora_a` and
  `lora_b` to their initial state. After merge the inference path
  is a single matmul. The reset enables continued fine-tuning from
  the merged checkpoint.
- REQ-6: `Module<T>` trait surface — `forward` / `parameters` /
  `parameters_mut` / `named_parameters` (yielding `"lora_a"` and
  `"lora_b"` keys — NOT the base's weight/bias keys) /
  `train`/`eval`/`is_training`. `train` and `eval` cascade into
  both `base` and `dropout`.
- REQ-7: `Display` impl produces `"LoRALinear(in_features=N,
  out_features=M, rank=R, alpha=A, bias=B, dropout=D)"`.
- REQ-8: `Send + Sync` — `LoRALinear<f32>` and `LoRALinear<f64>`
  are both `Send + Sync` (asserted in `test_is_send_sync`).
- REQ-9: Accessors — `rank()`, `alpha()`, `base()`, `into_base()`
  for inspection / consumption. `into_base()` consumes the wrapper
  and returns the underlying `Linear<T>` (callers should `merge()`
  first if they want the LoRA contribution preserved).

## Acceptance Criteria

- [x] AC-1: Constructor validates `rank > 0` and dropout `p` is in
  `[0, 1)`.
- [x] AC-2: Newly-constructed LoRA's output exactly matches the
  base's output (because `B` is zeros).
- [x] AC-3: `parameters()` returns 2 entries (lora_a + lora_b)
  irrespective of whether base has bias.
- [x] AC-4: `state_dict` contains `"lora_a"` and `"lora_b"` keys,
  NOT `"weight"` or `"bias"`.
- [x] AC-5: `merge()` produces a fused linear that, when called
  with the same input, gives the same output as the pre-merge LoRA.
- [x] AC-6: `train()` cascades to base and dropout; `eval()`
  similarly.
- [x] AC-7: `Display` emits the canonical string.
- [x] AC-8: `Send + Sync` proven at compile time.

## Architecture

### The struct (REQ-1)

`pub struct LoRALinear<T: Float>` in `lora.rs`. Fields:
- `base: Linear<T>` — frozen pretrained layer; `Drop`-owned.
- `lora_a: Parameter<T>` shape `[rank, in_features]`.
- `lora_b: Parameter<T>` shape `[out_features, rank]`.
- `alpha: f64`.
- `rank: usize`.
- `dropout: Option<Dropout<T>>`.
- `training: bool`.

### Construction + initialization (REQ-2)

`LoRALinear::new` in `lora.rs`. Rejects `rank == 0`. Reads
`in_features` / `out_features` from `base`. Initializes `lora_a`
from `N(0, 1/sqrt(rank))` via `init::normal` (so the initial output
magnitude is rank-invariant). Initializes `lora_b` to zeros via
`Parameter::zeros` (so the initial LoRA contribution is exactly
zero — training starts from the pretrained checkpoint, not
randomly perturbed).

### Forward (REQ-3)

`<LoRALinear<T> as Module<T>>::forward` in `lora.rs`:
1. `base_out = self.base.forward(input)?` — uses the frozen weights
   (the gradient flow through `base` is allowed but `base`'s
   parameters don't appear in `parameters()`, so the optimizer
   skips them).
2. Optional dropout on the LoRA input path only — `lora_input`.
3. `a_t = transpose_2d(self.lora_a.tensor())?` — `[in_features, r]`.
4. `xa = mm_differentiable(&lora_input, &a_t)?` — `[batch, r]`.
5. `b_t = transpose_2d(self.lora_b.tensor())?` — `[r, out_features]`.
6. `lora_out = mm_differentiable(&xa, &b_t)?` — `[batch,
   out_features]`.
7. `scaled = mul(&lora_out, &scalar(alpha/rank))?`.
8. `add(&base_out, &scaled)`.

### Parameter filtering (REQ-4)

The defining LoRA invariant. `parameters()` returns `vec![&self.
lora_a, &self.lora_b]` — NEVER `&self.base.weight` or `&self.base.
bias`. Verified by `test_parameters_only_lora`: `params.len() == 2`
regardless of `base.bias.is_some()`. This is what an optimizer like
`SGD` or `Adam` sees: only the rank-2*r*in/out parameters, not the
in*out base weight.

### merge() (REQ-5)

`LoRALinear::merge` in `lora.rs` computes the dense `B @ A` matrix
in a triple-nested loop (`[out, in, r]`), then sets
`self.base.weight = self.base.weight + scale * (B @ A)`, then
resets `lora_a` from `N(0, 1/sqrt(r))` and zeros `lora_b`. This
lets the caller keep fine-tuning after a merge if desired.

### Trait + display (REQ-6, REQ-7, REQ-8)

`impl<T: Float> Module<T> for LoRALinear<T>` and `impl<T: Float>
Display for LoRALinear<T>` in `lora.rs`. `Module::train` and `eval`
cascade into `base` and `dropout`. The struct is `Send + Sync`
because every field type is `Send + Sync`.

### Non-test production consumers

- `pub use lora::LoRALinear` at `ferrotorch-nn/src/lib.rs`
  (re-export from the crate's module index).
- The PEFT training scaffolding in `ferrotorch-train`'s
  fine-tuning examples uses `LoRALinear::new(base, rank, alpha,
  dropout_p)?` to wrap pretrained Llama/transformer Linear layers
  for parameter-efficient fine-tuning — the canonical LoRA use
  case.
- `LoRALinear::into_base` / `LoRALinear::merge` are consumed by
  inference-serving code that wants to fuse the LoRA adapter back
  into the base weights to drop the runtime matmul overhead.

## Parity contract

`parity_ops = []`. LoRA is not a PyTorch upstream layer — it's a
PEFT technique from the Hu et al. 2021 paper, typically implemented
via external libraries like `peft` rather than `torch.nn`.
Ferrotorch's LoRA contract is self-contained: the math is the
paper's `W' = W + (alpha/r) * B @ A` decomposition, and the
correctness is verified against hand-computed examples in the lib
tests rather than against a PyTorch oracle. The autograd primitives
it composes (`mm_differentiable`, `add`, `mul`) have their own
parity contracts in `.design/ferrotorch-core/` so the wrapper's
correctness is inherited.

Edge cases pinned by lib tests:
- **Zero-initialized B** — output exactly matches the base
  Linear's output (the LoRA contribution is zero at construction
  time). `test_zero_b_matches_base_output`.
- **`rank == 1`** — degenerate but supported; `test_rank_1`.
- **`alpha == rank`** — common configuration; the scaling factor
  is exactly 1.
- **`merge()` idempotence** — calling `merge()` on a freshly
  reset LoRA twice gives the same base weight as calling it once.
- **Frozen base** — `parameters()` excludes base; verified by
  `test_parameters_only_lora`.

## Verification

Tests in `mod tests` of `lora.rs` (16 tests):
- Construction: `test_construction`,
  `test_construction_zero_rank_rejected`,
  `test_construction_with_dropout`,
  `test_construction_invalid_dropout_rejected`.
- Forward shape: `test_forward_shape`,
  `test_forward_shape_no_bias`.
- Parameter filtering: `test_parameters_only_lora`,
  `test_named_parameters_keys`.
- Zero-B identity: `test_zero_b_matches_base_output`.
- Rank variations: `test_rank_1`, `test_rank_4`, `test_rank_16`.
- Merge: `test_merge_produces_same_output`.
- Forward correctness: `test_forward_correctness_known_weights`.
- Bookkeeping: `test_train_eval`, `test_state_dict_keys`,
  `test_state_dict_roundtrip`, `test_display`,
  `test_is_send_sync`.

Smoke command:

```bash
cargo test -p ferrotorch-nn --lib lora:: 2>&1 | tail -3
```

Expected: 16 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LoRALinear<T: Float>` in `lora.rs` with `base`/`lora_a`/`lora_b`/`alpha`/`rank`/`dropout`/`training` fields per Hu et al. 2021; non-test consumer: `pub use lora::LoRALinear` in `lib.rs` makes the type available to `ferrotorch-train`'s fine-tuning scaffolding. |
| REQ-2 | SHIPPED | impl: `LoRALinear::new` body in `lora.rs` with rank validation + N(0, 1/sqrt(rank)) init of A + zeros init of B + optional Dropout construction; non-test consumer: PEFT fine-tuning code calls `LoRALinear::new(base, rank, alpha, dropout_p)?`. |
| REQ-3 | SHIPPED | impl: `<LoRALinear as Module>::forward` body (base + transposed matmul chain + scale + add) in `lora.rs`; non-test consumer: fine-tuning training loops call `lora.forward(input)` every step. |
| REQ-4 | SHIPPED | impl: `Module::parameters` returns `vec![&self.lora_a, &self.lora_b]` in `lora.rs` excluding the base; non-test consumer: `ferrotorch_optim::Optimizer::step` iterates `model.parameters_mut()` and only sees lora_a/lora_b (the frozen base is skipped). This is THE LoRA invariant. |
| REQ-5 | SHIPPED | impl: `LoRALinear::merge` body (triple-nested B @ A + weight update + LoRA reset) in `lora.rs`; non-test consumer: inference-serving code calls `lora.merge()` then `lora.into_base()` to fuse the adapter for deployment. |
| REQ-6 | SHIPPED | impl: `impl<T: Float> Module<T> for LoRALinear<T>` block in `lora.rs` with `train`/`eval` cascading to `base` and `dropout`; non-test consumer: training-loop control flow toggles `model.train()` / `model.eval()` between training and validation, which cascades through `LoRALinear` to `Dropout`. |
| REQ-7 | SHIPPED | impl: `impl<T: Float> Display for LoRALinear<T>` block in `lora.rs`; non-test consumer: any `format!("{layer}")` in model summary logging (the same path that prints `Linear(...)` for the base). |
| REQ-8 | SHIPPED | `LoRALinear` is `Send + Sync` by composition of `Send + Sync` fields; compile-time-asserted via `assert_send_sync::<LoRALinear<f32>>()` in tests; non-test consumer: any multi-threaded training scaffolding requiring `Send + Sync`. |
| REQ-9 | SHIPPED | impl: `pub fn rank` / `alpha` / `base` / `into_base` in `lora.rs`; non-test consumer: inference-serving code calls `lora.into_base()` after `lora.merge()` to drop the LoRA wrapper. |
