# ferrotorch-llama — `spec_decode` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - Leviathan et al. 2023 "Fast Inference from Transformers via
    Speculative Decoding", arXiv:2211.17192, Algorithm 1 / §3
    (acceptance criterion: min(1, p(x̃)/q(x̃)); residual sampling:
    norm(max(0, p − q)))
  - HuggingFace transformers/generation/utils.py (assisted_decoding
    / speculative_decoding paths)
-->

## Summary

`ferrotorch-llama/src/spec_decode.rs` ships speculative decoding
per Leviathan et al. 2023 Algorithm 1: a small `draft` model
proposes K tokens autoregressively; the larger `target` model
verifies them in a single batched forward; each draft token is
accepted with probability `min(1, p/q)`; on rejection the
corrected token is sampled from `norm(max(0, p − q))`; on full
acceptance a bonus token is sampled from the target's next-position
distribution. The single-forward verify path (per the
`forward_ids_all_positions` API) drops the per-step verify cost
from `O((K+1) · S)` to `O(S)` (#1129).

## Requirements

- REQ-1: `pub trait ModelHandle<T: Float>` exposes a deliberately
  narrow interface: `forward_ids(ids) → Vec<f64>` (last-position
  logits) and `forward_ids_all_positions(ids) → Vec<Vec<f64>>`
  (per-position logits, length = `ids.len()`). The trait carries
  `vocab_size() → usize` for vocab-equality validation between
  draft and target.
- REQ-2: A default `forward_ids_all_positions` is provided that
  falls back to per-position calls — quadratic but correct so
  existing custom impls don't break.
- REQ-3: `pub struct LlamaHandle<'m, T: Float>` adapts
  `&LlamaForCausalLM<T>` to `ModelHandle<T>`. Its
  `forward_ids_all_positions` is the real single-forward
  implementation: one `forward_from_ids` call returns the
  `[1, S, V]` logits tensor, which is split into S per-position
  `Vec<f64>` slices.
- REQ-4: `pub struct SpecDecodeConfig` carries `draft_k`,
  `max_new_tokens`, `seed`, `eos_token_ids`. `SpecDecodeConfig::
  validate` rejects `draft_k == 0` and `max_new_tokens == 0`
  with `InvalidArgument`.
- REQ-5: `pub struct SpecDecodeOutput` carries the generated
  `tokens`, the `accepted_count`, and the `proposed_count`. The
  `acceptance_rate` method returns `accepted / proposed` (or `1.0`
  on the degenerate `proposed_count == 0` case).
- REQ-6: `pub fn speculative_decode<T: Float>(draft, target,
  prompt_ids, config)` runs the full algorithm:
  - Empty `prompt_ids` → `InvalidArgument`.
  - Draft / target vocab-size mismatch → `InvalidArgument`.
  - Per outer iteration: K draft tokens autoregressively, ONE
    batched target verify over the whole `context + draft_tokens`,
    accept/reject loop with `min(1, p/q)`, residual sampling on
    rejection, bonus sample on full acceptance.
- REQ-7: Numerical stability: `softmax_f64` is the
  max-subtraction variant; degenerate all-`-inf` returns zeros.
  `sample_probs` falls back to argmax when the total probability
  is `<= 0` or non-finite. `sample_residual` falls back to
  `sample_probs(p, ...)` when the residual is zero everywhere
  (p == q exactly).

## Acceptance Criteria

- [x] AC-1: `softmax_f64` sums to 1.0 on a finite input vector.
- [x] AC-2: `softmax_f64` on all-`-inf` returns zeros.
- [x] AC-3: `sample_residual` with `p == q` falls back to
  `sample_probs(p, ...)`.
- [x] AC-4: `SpecDecodeConfig::validate` rejects `draft_k == 0`
  and `max_new_tokens == 0`.
- [x] AC-5: `LlamaHandle::forward_ids_all_positions` returns a
  `Vec<Vec<f64>>` of length `ids.len()` with each inner vector of
  length `vocab_size`.
- [x] AC-6: `speculative_decode` on a tiny model with `draft ==
  target` always produces tokens identical to greedy decoding
  (acceptance rate = 1.0 at every step in this degenerate case).
- [x] AC-7: `speculative_decode` rejects vocab-size-mismatched
  draft/target with `InvalidArgument`.

## Architecture

`pub trait ModelHandle<T: Float>` in `spec_decode.rs` is the
narrow forward-only interface speculative decoding needs. It
exposes both the legacy single-position `forward_ids` and the
batched `forward_ids_all_positions` so the verify step can run
once over the whole prefix instead of K+1 times. The default
`forward_ids_all_positions` is the quadratic fallback for custom
impls; the real single-forward implementation lives on
`LlamaHandle`.

`pub struct LlamaHandle<'m, T: Float>` in `spec_decode.rs` carries
a borrowed reference to a `LlamaForCausalLM<T>`. Its
`ModelHandle` impl validates the `[1, S, V]` shape contract on
every forward and casts logits from `T` to `f64` via
`ferrotorch_core::numeric_cast::cast`.

`pub fn speculative_decode<T: Float>` in `spec_decode.rs` is the
Algorithm 1 main loop. Per outer iteration:

1. **Draft**: produce K tokens autoregressively using `draft.
   forward_ids(&draft_ctx)`. Record the softmax distribution
   `q_j` at each draft position.
2. **Verify**: construct `verify_prefix = context ++ draft_tokens`
   and call `target.forward_ids_all_positions(&verify_prefix)`
   once. Validate the returned vector length matches the prefix.
   Compute the K+1 target distributions `p_0..p_k` from positions
   `context.len() - 1 + j` (for `j ∈ 0..=k`) via `softmax_f64`.
3. **Accept/reject**: for each `j` in `0..k`:
   - Look up `q = draft_probs[j][token]`, `p =
     target_probs[j][token]`.
   - `accept_prob = if q <= 0 { 0 } else { (p / q).min(1.0) }`.
   - Draw `u = xorshift_uniform`. If `u < accept_prob`, accept;
     else sample corrected token from
     `norm(max(0, p_j - q_j))` and stop.
4. **Emit**: emit `n_accepted` draft tokens (stopping on EOS or
   `max_new_tokens`). Then emit either the corrected token
   (rejection case) or the bonus token sampled from `p_k`
   (all-accepted case).

`fn softmax_f64` in `spec_decode.rs` is the
max-subtraction-stable softmax. `fn sample_probs` is the
inverse-cumulative-CDF categorical sampler with fallback to
argmax on degenerate inputs. `fn sample_residual` computes the
positive part of `p - q`, normalizes, and samples; falls back to
`sample_probs(p)` on degenerate all-zero residual.

### Non-test production consumers

- `pub use spec_decode::{LlamaHandle, ModelHandle,
  SpecDecodeConfig, SpecDecodeOutput, speculative_decode}` at
  `ferrotorch-llama/src/lib.rs:176-178`.
- The `ferrotorch::llama` umbrella re-export at
  `ferrotorch/src/lib.rs:155` exposes the full spec-decode
  surface (`speculative_decode`, `LlamaHandle`, `ModelHandle`,
  `SpecDecodeConfig`, `SpecDecodeOutput`) for downstream consumers.

The current crate ships an integration test
(`ferrotorch-llama/tests/spec_decode_test.rs`) plus a probe
(`_probe_c8_spec_decode.rs`) exercising the algorithm against
real Llama checkpoints. Per goal.md S5, the existing pub API
surface is the boundary method that IS the public API —
downstream wiring is not a blocker.

## Parity contract

`parity_ops = []`. The correctness criterion is Leviathan et
al. 2023 §3 Algorithm 1 step-by-step:

- **Acceptance probability**: `min(1, p / q)`. The `q <= 0`
  guard yields acceptance probability 0 (we cannot accept a
  token the draft assigned zero mass to).
- **Residual sampling on rejection**: `norm(max(0, p − q))`.
  When the residual is identically zero (p == q exactly), we
  fall back to `sample_probs(p)` — this matches the paper's
  observation that the rejection-side resample is undefined
  when the distributions agree.
- **Bonus token on full acceptance**: sampled from `p_K`, the
  target distribution at position `context.len() - 1 + k`.
  Matches Algorithm 1 Step 5.
- **Single-forward verify**: the `forward_ids_all_positions`
  contract eliminates the K+1-forward quadratic verify loop
  (#1129). Mathematically equivalent to per-position verifies
  (the target model is autoregressive, so position-j logits
  depend only on positions 0..j of the input).
- **Acceptance rate sanity**: when draft == target,
  `p[token] / q[token] == 1` for every draft token, so
  acceptance prob is `min(1, 1) = 1` and the run produces
  identical output to greedy decoding at the target.

## Verification

Tests in `mod tests in spec_decode.rs` (plus integration tests in
`ferrotorch-llama/tests/spec_decode_test.rs` and
`ferrotorch-llama/tests/_probe_c8_spec_decode.rs`):

- `softmax_sums_to_one`
- `softmax_all_neg_inf_returns_zeros`
- `sample_residual_degenerate_falls_back_to_p`
- `SpecDecodeConfig::validate` rejection paths
- End-to-end speculative-decode correctness on a tiny `draft ==
  target` model (acceptance rate = 1.0 ⇒ output matches greedy).
- Vocab-size mismatch rejection.

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-llama --lib spec_decode:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub trait ModelHandle<T: Float>` in `spec_decode.rs`; non-test consumer: re-exported at `ferrotorch-llama/src/lib.rs:177`. The `LlamaHandle` impl in `spec_decode.rs` is the canonical implementor; `speculative_decode` consumes `&dyn ModelHandle<T>` as both draft and target. |
| REQ-2 | SHIPPED | impl: the default `fn forward_ids_all_positions` body in the `ModelHandle` trait in `spec_decode.rs`; non-test consumer: any custom `ModelHandle` impl that doesn't override gets the quadratic-but-correct fallback. |
| REQ-3 | SHIPPED | impl: `pub struct LlamaHandle<'m, T: Float>` + the explicit `fn forward_ids_all_positions` override in `spec_decode.rs`; non-test consumer: re-exported at `ferrotorch-llama/src/lib.rs:177`; constructed via `LlamaHandle::new(&model)` to bridge a `LlamaForCausalLM` into `speculative_decode`. |
| REQ-4 | SHIPPED | impl: `pub struct SpecDecodeConfig` + `pub fn SpecDecodeConfig::validate` in `spec_decode.rs`; non-test consumer: re-exported at `ferrotorch-llama/src/lib.rs:177`. `speculative_decode` calls `config.validate()?` near the top of the function. |
| REQ-5 | SHIPPED | impl: `pub struct SpecDecodeOutput` + `pub fn acceptance_rate` in `spec_decode.rs`; non-test consumer: re-exported at `spec_decode in ferrotorch-llama/src/lib.rs`; produced by `speculative_decode` as its return value. |
| REQ-6 | SHIPPED | impl: `pub fn speculative_decode<T: Float>` in `spec_decode.rs`; non-test consumer: re-exported at `ferrotorch-llama/src/lib.rs:178`. The function chains the four phases (draft, verify, accept/reject, emit) and is the entry point for every speculative-decode caller. |
| REQ-7 | SHIPPED | impl: `fn softmax_f64` (max-subtraction stable), `fn sample_probs` (argmax fallback on degenerate total), `fn sample_residual` (`sample_probs(p)` fallback on zero residual) in `spec_decode.rs`; non-test consumer: every call to `softmax_f64` / `sample_probs` / `sample_residual` from inside `speculative_decode` is a production call site. |
