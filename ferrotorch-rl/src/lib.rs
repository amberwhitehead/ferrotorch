// Crate-level lint baseline. Mirrors the ferrotorch-bert / ferrotorch-graph
// posture: deny correctness / idiom / Debug / docs problems; warn pedantic
// stylistic issues. Specific pedantic lints are allowed crate-wide where the
// lint is consistently wrong for ML / numeric kernel code.

#![deny(unsafe_code)]
#![deny(rust_2018_idioms)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
// Casts: dimension math (`as usize`, `as f32`, `as u32`) is intrinsic to
// tensor indexing.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
// Builder-style accessors don't all need `#[must_use]`.
#![allow(clippy::must_use_candidate)]
// `MLP`, `PPO`, `sb3`, `bf16` flagged as missing backticks even inside fences.
#![allow(clippy::doc_markdown)]
// `needless_pass_by_value` would force `&MlpPolicyConfig` everywhere.
#![allow(clippy::needless_pass_by_value)]
// `unnecessary_wraps` flags `Result`-returning helpers that today always
// succeed but are part of an extensible API surface.
#![allow(clippy::unnecessary_wraps)]
// `format!("x={}", x)` vs `format!("x={x}")` churn.
#![allow(clippy::uninlined_format_args)]

//! Reinforcement-learning policy composition for ferrotorch.
//!
//! Phase D.2 of real-artifact-driven development: the
//! `stable-baselines3` `ActorCriticPolicy` (a.k.a. `MlpPolicy`) for
//! discrete-action environments, mirrored byte-for-byte from the
//! pinned `sb3/ppo-CartPole-v1` checkpoint and verified for forward-
//! pass parity against `stable-baselines3` via
//! `scripts/verify_rl_inference.py`.
//!
//! # Architecture (matches sb3 `MlpPolicy` defaults for CartPole-v1)
//!
//! ```text
//! MlpPolicy
//! ├── features_extractor: FlattenExtractor (identity on 1-D obs)
//! ├── mlp_extractor: MlpExtractor
//! │   ├── policy_net: Linear(obs_dim → 64) → Tanh → Linear(64 → 64) → Tanh
//! │   └── value_net:  Linear(obs_dim → 64) → Tanh → Linear(64 → 64) → Tanh
//! ├── action_net: Linear(64 → n_actions)        ← Categorical logits
//! └── value_net:  Linear(64 → 1)                ← scalar state value
//! ```
//!
//! `share_features_extractor=True` in sb3's default, but the policy /
//! value trunks have *separate* `Linear` weights — the "shared"
//! features extractor is the flatten layer, not the MLP. For a 1-D
//! observation (e.g. CartPole's `[cart_pos, cart_vel, pole_angle,
//! pole_angvel]`) the flatten is a no-op, so the loader treats it as
//! identity.
//!
//! # Loading real weights
//!
//! [`load_ppo_policy`] accepts a path to `model.safetensors` (the
//! ferrotorch mirror of an sb3 zip checkpoint) plus dimensions
//! `(obs_dim, hidden, n_actions)` and returns a populated
//! [`MlpPolicy`] plus a [`DropReport`] documenting any upstream key
//! that was intentionally not consumed. For the canonical
//! `sb3/ppo-CartPole-v1` pin the report should be empty; a non-empty
//! report on a clean pin signals a state-dict-drop bug (#1141 class).
//!
//! # On `log_std`
//!
//! sb3's `ActorCriticPolicy` only emits a `log_std` parameter for
//! continuous-action policies (DiagGaussianDistribution). For the
//! discrete `Categorical` distribution used by CartPole-v1 there is no
//! `log_std`, so the state dict has exactly 12 keys (4 for the policy
//! trunk × 2, 2 for action_net, 2 for value_net).
//!
//! ## REQ status (per `.design/ferrotorch-rl/lib.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl: crate-root attribute block (`#![deny(unsafe_code)]` + per-lint `#![allow]` blocks with documented rationale) at top of `ferrotorch-rl/src/lib.rs:1-29` mirroring `ferrotorch-bert`/`ferrotorch-graph` posture; non-test consumer: the workspace clippy gate (`cargo clippy -p ferrotorch-rl -- -D warnings`) consumes this baseline; the per-lint allows are referenced by every production file in the crate. |
//! | REQ-2 | SHIPPED | impl: `pub mod mlp_policy; pub mod safetensors_loader;` declarations in `ferrotorch-rl/src/lib.rs`; non-test consumer: `ferrotorch-rl/src/safetensors_loader.rs:36` `use crate::mlp_policy::{MlpPolicy, MlpPolicyConfig};` consumes the `mlp_policy` module via this declaration; the production binary `ferrotorch-rl/examples/ppo_policy_dump.rs:34` consumes both modules transitively via the prelude re-exports. |
//! | REQ-3 | SHIPPED | impl: `pub use mlp_policy::{ActionNet, MlpExtractor, MlpPolicy, MlpPolicyConfig, ValueHead}` and `pub use safetensors_loader::{DropReport, load_ppo_policy}` below; non-test consumer: `ferrotorch-rl/examples/ppo_policy_dump.rs:34` `use ferrotorch_rl::{MlpPolicyConfig, load_ppo_policy};` invokes the production binary via the canonical user-facing import path. |
//! | REQ-4 | SHIPPED | impl: crate doc-comment paragraphs at `ferrotorch-rl/src/lib.rs:31-75` including the architectural diagram and CartPole-v1 dimension contract; non-test consumer: `ferrotorch-rl/src/mlp_policy.rs:71-78` `MlpPolicyConfig::cartpole_v1()` returns `(obs_dim: 4, hidden: 64, n_actions: 2)` which is the production realisation of the documented contract; `ferrotorch-rl/examples/ppo_policy_dump.rs:196-200` constructs the same `MlpPolicyConfig` via CLI args. |
//! | REQ-5 | SHIPPED | impl: crate doc-comment paragraph "On `log_std`" at `ferrotorch-rl/src/lib.rs:69-75` documenting the 12-key state-dict contract; non-test consumer: `ferrotorch-rl/src/mlp_policy.rs:547-559` `Module::named_parameters` for `MlpPolicy` emits exactly those 12 keys (verified by the production safetensors loader's expected-key set construction at `ferrotorch-rl/src/safetensors_loader.rs:80-84`).

pub mod mlp_policy;
pub mod safetensors_loader;

pub use mlp_policy::{ActionNet, MlpExtractor, MlpPolicy, MlpPolicyConfig, ValueHead};
pub use safetensors_loader::{DropReport, load_ppo_policy};
