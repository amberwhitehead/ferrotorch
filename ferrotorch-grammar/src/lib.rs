//! Constrained-decoding grammar processors.
//!
//! Submodules:
//!
//! - [`schema`] — internal `Schema` enum and JSON-Schema parser (subset).
//! - [`state`]  — `JsonGrammar` state machine that tracks where we are in
//!   the partially-emitted JSON value.
//! - [`json_schema`] — public `JsonSchemaProcessor` that wraps a tokenizer
//!   vocabulary and produces per-step token-allow masks for use with
//!   `ferrotorch_cubecl::apply_token_mask_to_gpu`.
//!
//! ## REQ status (per `.design/ferrotorch-grammar/lib.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl: `pub mod json_schema; pub mod schema; pub mod state;` + `#[cfg(feature = "cuda")] pub mod gpu_dispatch;` in `lib.rs`; non-test consumer: `pub use ferrotorch_grammar as grammar;` in `ferrotorch-llama/src/lib.rs:156` makes the submodule tree reachable to every downstream model crate (grandfathered public API per goal.md S5). |
//! | REQ-2 | SHIPPED | impl: `pub use json_schema::{GrammarError, JsonSchemaProcessor, TokenMask};` + `pub use schema::Schema;` + `pub use state::{BooleanEmissionStage, JsonGrammar};` + `#[cfg(feature = "cuda")] pub use gpu_dispatch::{PackedVocab, compute_mask_gpu};` in `lib.rs`; non-test consumer: `ferrotorch-llama/src/lib.rs:156` aliases the whole crate (grandfathered S5). |
//! | REQ-3 | SHIPPED | impl: the re-export chain in `lib.rs`; non-test consumer: `pub use ferrotorch_grammar as grammar;` in `ferrotorch-llama/src/lib.rs:156` documented at lines 150-155 is the literal alias that makes the entire public surface a member of `ferrotorch_llama::grammar`. |

pub mod json_schema;
pub mod schema;
pub mod state;

#[cfg(feature = "cuda")]
pub mod gpu_dispatch;

pub use json_schema::{GrammarError, JsonSchemaProcessor, TokenMask, TokenTransitionCache};
pub use schema::Schema;
pub use state::{BooleanEmissionStage, JsonGrammar};

#[cfg(feature = "cuda")]
pub use gpu_dispatch::{PackedVocab, compute_mask_gpu};
