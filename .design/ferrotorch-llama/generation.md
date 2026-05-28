# ferrotorch-llama — `generation` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - HuggingFace transformers/generation/utils.py
    (GenerationMixin.generate, greedy/sample/beam_search paths)
  - HuggingFace transformers/generation/logits_process.py
    (TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor)
  - Keskar et al. 2019 (Repetition penalty formula)
-->

## Summary

`ferrotorch-llama/src/generation.rs` ships the token-level sampling
loop (`generate`, `generate_with_streamer`) on top of
`LlamaForCausalLM::forward_from_ids`, the public logit primitives
(`apply_temperature`, `top_k_filter`, `top_p_filter`,
`apply_repetition_penalty`, `argmax`, `sample_softmax`), the
`GenerationConfig` knobs, and the beam-search path
(`beam_search`, `BeamSearchConfig`) which exploits the KV-cache
incremental forward (#1129) for `O(seq)` per-step work.

## Requirements

- REQ-1: `pub struct GenerationConfig` carries the knobs:
  `max_new_tokens`, `temperature`, `top_k`, `top_p`,
  `repetition_penalty`, `eos_token_ids`, `seed`. Defaults match a
  sensible "vanilla sampling" configuration.
- REQ-2: Convenience constructors `GenerationConfig::greedy`
  (`temperature = 0.0`), `GenerationConfig::sampling`
  (temperature only), `GenerationConfig::nucleus`
  (temperature + top_p) for common presets.
- REQ-3: `pub fn generate` is a thin wrapper around
  `generate_with_streamer` with a no-op streamer.
- REQ-4: `pub fn generate_with_streamer` validates inputs (empty
  prompt, negative temperature, top_p outside `[0, 1]`,
  non-positive repetition_penalty all return `InvalidArgument`)
  then runs the autoregressive loop: forward → last-position
  logits → repetition_penalty → (greedy / temperature → top_k →
  top_p → sample_softmax) → append → streamer → EOS check.
- REQ-5: `pub fn apply_temperature`, `pub fn top_k_filter`,
  `pub fn top_p_filter`, `pub fn apply_repetition_penalty`,
  `pub fn argmax`, `pub fn sample_softmax` are exposed so callers
  can roll their own generation loops on top of
  `forward_from_ids`.
- REQ-6: Greedy path (`temperature == 0.0`) picks `argmax(logits)`
  directly; sampling path applies temperature, then top-k, then
  top-p, then categorical sampling via softmax.
- REQ-7: Repetition penalty (Keskar et al. 2019): for each token in
  context, divide its logit by the penalty if positive, multiply
  otherwise. `1.0` is the no-op baseline.
- REQ-8: `pub fn beam_search` runs `num_beams` candidates ranked by
  cumulative log-probability with KV-cache-backed per-step
  expansion. Each beam carries a per-beam `LlamaKvCache`, cloned
  at fork points so divergent continuations don't alias.
- REQ-9: `BeamSearchConfig::length_penalty` exponent normalizes the
  per-beam score by `len^length_penalty`. Returned beams are
  sorted best-first.

## Acceptance Criteria

- [x] AC-1: `argmax(&[0.1, 0.5, 0.2, 0.9, -1.0])` returns 3.
- [x] AC-2: `apply_temperature` divides every logit by the temperature.
- [x] AC-3: `top_k_filter` keeps only the top k entries; the rest
  go to `-inf`. `k = 0` is a no-op.
- [x] AC-4: `top_p_filter` keeps the smallest set whose softmaxed
  cumulative probability reaches `top_p`. `top_p = 1.0` is a no-op.
- [x] AC-5: `apply_repetition_penalty` halves a `+1.0` logit and
  doubles a `-1.0` logit at penalty `2.0`.
- [x] AC-6: `sample_softmax` with one finite logit (rest `-inf`)
  always returns that token's index.
- [x] AC-7: `beam_search` returns exactly `num_beams` candidates of
  exactly `max_new_tokens` tokens each (in the absence of EOS).
- [x] AC-8: `beam_search(num_beams = 1)` matches a hand-rolled
  greedy reference using full-prefix `forward_from_ids` at every
  step — pins the KV-cache equivalence (#1129).
- [x] AC-9: Setting every token as EOS forces every beam to be
  exactly 1 token long.

## Architecture

`pub struct GenerationConfig` in `generation.rs` carries all the
knobs the sampling loop consumes. `Default` sets
`max_new_tokens = 64, temperature = 1.0, top_k = 0, top_p = 1.0,
repetition_penalty = 1.0, eos_token_ids = [], seed = None`.

`pub fn generate_with_streamer` in `generation.rs` is the main
loop:

1. Validate inputs (empty prompt / out-of-range knobs).
2. For each new-token step (up to `max_new_tokens`):
   a. `model.forward_from_ids(&ids)?` — full-prefix forward over
      the running context.
   b. Slice the last-position logits.
   c. Apply repetition penalty (if `penalty != 1.0`).
   d. If `temperature == 0.0`, pick `argmax`; otherwise apply
      temperature, then top-k filter (if `k > 0`), then top-p
      filter (if `p < 1.0`), then `sample_softmax`.
   e. Append the new token, invoke the streamer, check for EOS.

`pub fn apply_temperature` in `generation.rs` divides every
logit by the temperature in-place. The `inv = 1.0 / temperature`
factor is computed once.

`pub fn top_k_filter` in `generation.rs` partial-sorts indices
by logit (descending), then sets every logit below the k-th
largest to `-inf`. `k = 0` and `k >= logits.len()` are no-ops.

`pub fn top_p_filter` in `generation.rs` softmaxes the logits,
sorts by probability (descending), accumulates until the running
sum reaches `top_p`, and sets every NOT-kept logit to `-inf`.
`top_p = 1.0` is a no-op.

`pub fn apply_repetition_penalty` in `generation.rs` walks the
context and for each token id `i < vocab`, divides
`logits[i]` by `penalty` if positive, multiplies otherwise.

`pub fn sample_softmax` in `generation.rs` softmaxes the logits
and draws a categorical sample using the xorshift PRNG state.
On all-`-inf` (every token filtered) it falls back to argmax.

`pub fn beam_search` in `generation.rs` is the beam-search path
(#612, KV-cache-enabled per #1129):

1. Validate inputs (empty prompt, `num_beams = 0`,
   `length_penalty = 0` all `InvalidArgument`).
2. Seed: feed the full prompt through the model once via
   `forward_one_with_cache` token by token, accumulating one
   shared KV cache. The next-token logits at this point are the
   first-new-token distribution.
3. Expand: for each live beam, compute the log-prob of every
   continuation (numerically-stable softmax), partial-sort to
   top-`num_beams`, fork the beam per surviving continuation.
4. Advance: for each surviving `(beam, token)` pair, clone the
   parent's KV cache and call
   `model.forward_one_with_cache(token, &parent.cache)?` to
   advance one step. Cache cloning is intentional — divergent
   continuations need independent K/V trails.
5. Finalize: when EOS is produced or `max_new_tokens` is reached,
   sort the beams by `score / len^length_penalty` (best first).

### Non-test production consumers

- `pub use generation::{GenerationConfig, apply_repetition_penalty,
  apply_temperature, argmax, generate, generate_with_streamer,
  sample_softmax, top_k_filter, top_p_filter}` at
  `ferrotorch-llama/src/lib.rs:166-169` exposes the public surface.
- The crate-level doc-comment block in `lib.rs` documents the
  `LlamaForCausalLM::load_hf_state_dict` →
  `ferrotorch_serialize::load_safetensors_sharded` →
  `generate` pipeline as the canonical text-generation path.
- The `ferrotorch` meta-crate re-exports the entire `ferrotorch-llama`
  surface at `ferrotorch/src/lib.rs:155` (`pub use
  ferrotorch_llama::*;`), making every generation primitive
  reachable from downstream consumers as `ferrotorch::llama::generate`,
  `ferrotorch::llama::GenerationConfig`, etc.
- `beam_search` and `BeamSearchConfig` are pub at the module level
  (not yet re-exported through `lib.rs`); their non-test
  consumers are the `ferrotorch-llama` examples and integration
  tests that import via the module path
  `ferrotorch_llama::generation::beam_search`.

## Parity contract

`parity_ops = []`. Sampling and beam-search are HF/Hub-conventional
behaviors rather than per-op parity probes. Behavioral guarantees:

- **Greedy via `argmax`**: matches HF's
  `do_sample = False` path. The `temperature == 0.0` sentinel
  switches the decode mode without needing a separate `do_sample`
  field.
- **Logit-warper order**: temperature → top-k → top-p, matching HF's
  `LogitsProcessorList` default ordering when those processors are
  attached.
- **Repetition penalty formula**: Keskar et al. 2019. HF's
  `RepetitionPenaltyLogitsProcessor` uses the same `v > 0 ?
  v / penalty : v * penalty` form.
- **Nucleus sampling**: keeps the smallest set whose cumulative
  softmaxed probability reaches `top_p`. HF's
  `TopPLogitsWarper` has the same semantic.
- **Beam search length normalization**: `score / len^length_penalty`.
  HF's `LengthPenalty` uses the same exponent form.
- **KV cache during beam search**: the beam-search path uses
  `forward_one_with_cache` for each step (per #1129). This
  matches HF's `use_cache=True` beam-search behavior on `LlamaModel`.

## Verification

Tests in `mod tests in generation.rs`:

- `argmax_picks_highest`
- `temperature_scales_logits`
- `top_k_keeps_only_k`, `top_k_zero_is_noop`
- `top_p_keeps_just_enough_mass`, `top_p_one_is_noop`
- `repetition_penalty_downweights_seen_tokens`
- `repetition_penalty_negative_logits`
- `sample_softmax_with_one_finite_logit_picks_it`
- `sample_softmax_all_neg_inf_falls_back_to_argmax`
- `sample_softmax_distribution_matches_probs_loosely`
- `generation_config_helpers`
- `beam_search_config_defaults_sensible`
- `beam_search_validates_inputs`
- `beam_search_returns_num_beams_results`
- `beam_search_matches_full_prefix_reference_top1` (discriminating
  test pinning the #1129 KV-cache equivalence)
- `beam_search_eos_finalizes_beam`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-llama --lib generation:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct GenerationConfig` + `Default` impl in `generation.rs`; non-test consumer: re-exported at `ferrotorch-llama/src/lib.rs:167`, reachable from the meta-crate as `ferrotorch::llama::GenerationConfig` via the umbrella re-export at `ferrotorch/src/lib.rs:155`. |
| REQ-2 | SHIPPED | impl: `GenerationConfig::greedy` / `sampling` / `nucleus` in `generation.rs`; non-test consumer: same re-export surface as REQ-1 — any external caller building a config uses one of these helpers. |
| REQ-3 | SHIPPED | impl: `pub fn generate` in `generation.rs`; non-test consumer: re-exported at `generation in ferrotorch-llama/src/lib.rs`; the crate-level doc-comment in `lib.rs` documents `generate` as the public text-generation entry point. |
| REQ-4 | SHIPPED | impl: `pub fn generate_with_streamer` in `generation.rs`; non-test consumer: same re-export surface as REQ-3; `generate` itself calls `generate_with_streamer` with a no-op streamer. |
| REQ-5 | SHIPPED | impl: `pub fn apply_temperature` / `top_k_filter` / `top_p_filter` / `apply_repetition_penalty` / `argmax` / `sample_softmax` in `generation.rs`; non-test consumer: re-exported at `ferrotorch-llama/src/lib.rs:167-169`; the same primitives are called from within `generate_with_streamer` itself, a non-test production caller. |
| REQ-6 | SHIPPED | impl: the `if config.temperature == 0.0 { argmax(...) } else { apply_temperature ...; sample_softmax(...) }` block in `generate_with_streamer` in `generation.rs`; non-test consumer: same call path as REQ-4. |
| REQ-7 | SHIPPED | impl: `pub fn apply_repetition_penalty` in `generation.rs`; non-test consumer: invoked inside `generate_with_streamer` in `generation.rs` when `(penalty - 1.0).abs() > f64::EPSILON`. |
| REQ-8 | SHIPPED | impl: `pub fn beam_search` in `generation.rs` (uses `forward_one_with_cache` and cache cloning); non-test consumer: callable via the `ferrotorch_llama::generation::beam_search` module path; the seed and per-step expansion both invoke `LlamaForCausalLM::forward_one_with_cache` in `model.rs`, which is itself a non-test production code path. |
| REQ-9 | SHIPPED | impl: the length-penalty sort block at the bottom of `beam_search` in `generation.rs`; non-test consumer: same path as REQ-8 — the returned `Vec<Vec<u32>>` is sorted best-first by `score / len^length_penalty`. |
