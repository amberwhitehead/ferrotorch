# ferrotorch-llama — `lib` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - HuggingFace transformers/models/llama/modeling_llama.py (LlamaForCausalLM,
    LlamaModel, LlamaDecoderLayer, LlamaMLP, LlamaAttention,
    LlamaRMSNorm, LlamaRotaryEmbedding)
  - HuggingFace transformers/models/llama/configuration_llama.py
    (LlamaConfig)
  - llama.cpp GGUF tensor naming (blk.{i}.attn_q.weight etc.)
-->

## Summary

`ferrotorch-llama/src/lib.rs` is the crate root. It declares the
crate-wide clippy lint baseline, the public module tree
(`attention`, `config`, `generation`, `gguf_remap`, `gpu`,
`kv_cache`, `layer`, `mlp`, `model`, `quant_loaders`,
`spec_decode`), and the flattened re-export surface that downstream
consumers (the `ferrotorch` meta-crate, `ferrotorch-llama/examples/`)
import from. It does not itself implement any layer or op; its role
is composition and re-export.

## Requirements

- REQ-1: A crate-level doc comment describes the Llama decoder stack
  (Embedding → N × `LlamaDecoderLayer` → final RMSNorm → `lm_head`)
  with a fenced text diagram so consumers can orient before opening
  individual modules.
- REQ-2: `pub mod` declarations for every public sub-module the crate
  ships: `attention`, `config`, `generation`, `gguf_remap`,
  `kv_cache`, `layer`, `mlp`, `model`, `quant_loaders`,
  `spec_decode`, plus `#[cfg(feature = "cuda")] pub mod gpu`.
- REQ-3: Flattened `pub use` re-exports of the user-facing types
  (`LlamaConfig`, `LlamaActivation`, `LlamaForCausalLM`,
  `LlamaModel`, `LlamaDecoderLayer`, `LlamaAttention`, `LlamaMLP`,
  `LlamaKvCache`, `LayerKvCache`, `GenerationConfig`, `generate`,
  `generate_with_streamer`, `apply_temperature`, `top_k_filter`,
  `top_p_filter`, `apply_repetition_penalty`, `argmax`,
  `sample_softmax`, `gguf_key_to_hf`, `gguf_to_hf_state_dict`,
  `GptqQ4`, `AwqQ4`, `dequantize_gptq_q4`, `dequantize_awq_q4`,
  `ModelHandle`, `LlamaHandle`, `SpecDecodeConfig`,
  `SpecDecodeOutput`, `speculative_decode`).
- REQ-4: GPU symbols (`LlamaGpuInferencer`, `LlamaGpuLayer`,
  `ProfiledForwardResult`) re-exported behind
  `#[cfg(feature = "cuda")]` so non-CUDA builds compile cleanly
  without the GPU dependency tree.
- REQ-5: Backward-compatible `pub use ferrotorch_grammar as grammar`
  alias so callers from before v0.5.1 (when the grammar processors
  lived in `ferrotorch_llama::grammar`) still compile unchanged.
- REQ-6: Crate-wide lint baseline that denies correctness/idiom/Debug/
  docs problems and warns pedantic style — with named, justified
  per-lint `#![allow]` exceptions for clippy lints that fire as
  noise in ML kernel code (casts, must-use, doc_markdown, etc.).

## Acceptance Criteria

- [x] AC-1: `cargo check -p ferrotorch-llama` compiles without errors.
- [x] AC-2: `cargo clippy -p ferrotorch-llama --lib -- -D warnings`
  passes (no warnings escape the lint baseline).
- [x] AC-3: `cargo doc -p ferrotorch-llama --no-deps` succeeds
  (no `missing_docs` violations).
- [x] AC-4: The re-exports listed in REQ-3 / REQ-4 are reachable as
  `ferrotorch_llama::<Name>` from external callers.
- [x] AC-5: `ferrotorch_llama::grammar` resolves to the
  `ferrotorch_grammar` crate via the BC alias.

## Architecture

`lib.rs` opens with a `#![deny(unsafe_code)]` / `#![deny(missing_docs)]`
/ `#![warn(clippy::pedantic)]` block, then a series of explicitly
named per-lint `#![allow]` exceptions, each documented with a
comment explaining why the lint fires as noise here (casts for tensor
dimensions, `doc_markdown` for `bf16`/`RoPE`/`cuBLAS` identifiers
inside fenced code, `float_cmp` for the `temperature == 0.0` greedy
sentinel match, etc.). This is the crate-wide lint baseline goal.md
R-CODE-3 permits (per-lint allowances are explicit, not module-root
silence).

The module declarations (`pub mod attention`, `pub mod config`, ...)
expose every sub-module in the crate. `gpu` is gated behind
`#[cfg(feature = "cuda")]` so a non-CUDA build doesn't pull in
`ferrotorch-gpu` / `cudarc` transitively.

The `pub use` block flattens the user-facing API: the sub-module
names are public for discoverability, but typical callers (the
`ferrotorch-llama/examples/llama3_8b.rs` driver, the `ferrotorch`
meta-crate's `pub mod llama` re-export at `ferrotorch/src/lib.rs:155`)
use the unprefixed names. The CUDA-only re-exports
(`LlamaGpuInferencer`, etc.) are inside the same `#[cfg(feature =
"cuda")]` gate so they vanish from the API in non-CUDA builds.

The `pub use ferrotorch_grammar as grammar` line preserves the v0.5.0
API surface: code that called `ferrotorch_llama::grammar::<X>` still
resolves after the extraction into the standalone
`ferrotorch_grammar` crate (R-DEV-7 applies — the Rust analog moved
to a dedicated crate, and the alias preserves the API).

### Non-test production consumers

- `pub mod llama { pub use ferrotorch_llama::*; }` at
  `ferrotorch/src/lib.rs:155` (the workspace meta-crate). This makes
  every `pub use` in `lib.rs` reachable as
  `ferrotorch::llama::<Name>` for every downstream consumer that
  depends on the `ferrotorch` umbrella crate.
- `use ferrotorch_llama::{LlamaConfig, LlamaForCausalLM};` at
  `ferrotorch-llama/examples/llama3_8b.rs:40`.
- `use ferrotorch_llama::{LlamaConfig, LlamaGpuInferencer};` at
  `ferrotorch-llama/examples/llama3_8b_gpu.rs:28`.
- `use ferrotorch_llama::LlamaGpuInferencer;` at
  `ferrotorch-llama/examples/llm_inference_dump.rs:240` (inside the
  `--feature cuda` branch).

## Parity contract

`parity_ops = []`. `lib.rs` declares no parity ops directly. The
behavioral parity surface is owned by the sub-modules each `pub mod`
declares (model / layer / attention / mlp / gpu).

The structural parity contract `lib.rs` enforces:

- Module names mirror upstream class boundaries (HF
  `LlamaDecoderLayer` → `pub mod layer`, HF `LlamaMLP` → `pub mod
  mlp`, HF `LlamaForCausalLM` → `pub mod model`).
- The re-export surface mirrors HF's `from transformers import
  LlamaForCausalLM, LlamaConfig` import shape — the user-facing
  Rust names land at the crate root, not buried in sub-modules.

## Verification

`lib.rs` carries no `#[cfg(test)]` tests of its own. The gauntlet
that protects it is the crate-wide:

```bash
cargo check -p ferrotorch-llama 2>&1 | tail -3
cargo clippy -p ferrotorch-llama --lib -- -D warnings 2>&1 | tail -3
cargo test -p ferrotorch-llama --lib 2>&1 | tail -3
cargo fmt -p ferrotorch-llama --check
```

Expected: all four pass clean. Any new pub item added to a
sub-module without a matching re-export here would fail the
discoverability check in downstream consumers, surfacing the gap.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: crate-level `//!` doc-comment with the decoder-stack fenced diagram in `lib.rs`; non-test consumer: same diagram is the canonical orientation for the `ferrotorch::llama` re-export at `ferrotorch/src/lib.rs:155`. |
| REQ-2 | SHIPPED | impl: `pub mod attention; pub mod config; ...` block in `lib.rs`; non-test consumer: `ferrotorch-llama/examples/llama3_8b.rs:40` reaches `LlamaForCausalLM` via the `model` module export. |
| REQ-3 | SHIPPED | impl: the flattened `pub use attention::LlamaAttention; pub use config::{LlamaActivation, LlamaConfig}; ...` block at the bottom of `lib.rs`; non-test consumer: `use ferrotorch_llama::{LlamaConfig, LlamaForCausalLM};` at `ferrotorch-llama/examples/llama3_8b.rs:40`. |
| REQ-4 | SHIPPED | impl: `#[cfg(feature = "cuda")] pub mod gpu;` and `#[cfg(feature = "cuda")] pub use gpu::{LlamaGpuInferencer, LlamaGpuLayer, ProfiledForwardResult};` in `lib.rs`; non-test consumer: `use ferrotorch_llama::LlamaGpuInferencer;` at `ferrotorch-llama/examples/llm_inference_dump.rs:240` inside the cuda gate. |
| REQ-5 | SHIPPED | impl: `pub use ferrotorch_grammar as grammar;` in `lib.rs`; non-test consumer: the umbrella re-export `pub mod llama { pub use ferrotorch_llama::*; }` at `ferrotorch/src/lib.rs:155` exposes the BC alias to every downstream `ferrotorch::llama::grammar` user. |
| REQ-6 | SHIPPED | impl: the lint baseline block at the top of `lib.rs` (`#![deny(unsafe_code)]` through the per-lint `#![allow]` exceptions, each with a comment); non-test consumer: `cargo clippy -p ferrotorch-llama --lib -- -D warnings` is the production gate that runs on every commit. |
