# ferrotorch-rl — Reinforcement-learning policy composition crate root

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/__init__.py
  - torch/nn/modules/module.py
-->

## Summary

`ferrotorch-rl/src/lib.rs` is the crate root for Phase D.2 of
real-artifact-driven development: the `stable-baselines3`
`ActorCriticPolicy` (a.k.a. `MlpPolicy`) for discrete-action
environments, mirrored byte-for-byte from the pinned
`sb3/ppo-CartPole-v1` checkpoint. The crate composes
`ferrotorch-nn` `Module` implementations (`Linear`, `Tanh`) into the
two-trunk Tanh-MLP plus separate action / value heads sb3 ships, then
exposes a safetensors loader that round-trips the upstream
`ActorCriticPolicy.state_dict()` byte-for-byte. Upstream PyTorch
itself does not ship sb3 — sb3 is a third-party Python package built
on PyTorch. ferrotorch-rl is the Rust analog (R-DEV-7) bundled as a
first-party crate to give the parity harness a known-pinned target.

## Requirements

- REQ-1: The crate root establishes a workspace-mirroring lint
  baseline: `deny(unsafe_code, rust_2018_idioms,
  missing_debug_implementations, missing_docs)` + `warn(clippy::all,
  clippy::pedantic)` plus per-lint `#![allow]` blocks for cast lints
  and a small set of pedantic lints the bridge code consistently
  trips. Mirrors `ferrotorch-bert` / `ferrotorch-graph` posture.
- REQ-2: Two public sub-modules — `mlp_policy` and
  `safetensors_loader` — are declared and reachable as
  `ferrotorch_rl::{mlp_policy, safetensors_loader}`.
- REQ-3: The crate-level prelude re-exports five public types from
  `mlp_policy` — `ActionNet`, `MlpExtractor`, `MlpPolicy`,
  `MlpPolicyConfig`, `ValueHead` — and two from
  `safetensors_loader` — `DropReport`, `load_ppo_policy` — so
  `use ferrotorch_rl::{MlpPolicy, MlpPolicyConfig, load_ppo_policy}`
  resolves cleanly (the canonical user-facing import path).
- REQ-4: The crate doc-comment documents the sb3 architectural
  contract for CartPole-v1: `obs_dim=4, hidden=64, n_actions=2`,
  `net_arch=[64, 64]`, `activation_fn=Tanh`, `ortho_init=True`. The
  documented diagram exactly matches the safetensors key layout
  (`mlp_extractor.policy_net.{0,2}.{weight,bias}`,
  `mlp_extractor.value_net.{0,2}.{weight,bias}`, `action_net.{weight,
  bias}`, `value_net.{weight,bias}`).
- REQ-5: The crate doc-comment documents the discrete-vs-continuous
  `log_std` divergence — sb3 only emits `log_std` for continuous
  policies; the discrete CartPole-v1 policy has exactly 12 parameter
  keys.

## Acceptance Criteria

- [x] AC-1: Crate-root lint baseline present with `#![deny(unsafe_code)]`
  and the per-lint allows carrying inline justification.
- [x] AC-2: `pub mod mlp_policy; pub mod safetensors_loader;`
  declarations resolve.
- [x] AC-3: `pub use mlp_policy::{ActionNet, MlpExtractor, MlpPolicy,
  MlpPolicyConfig, ValueHead}` and `pub use safetensors_loader::{DropReport,
  load_ppo_policy}` re-exports compile.
- [x] AC-4: Crate doc-comment includes the architectural diagram and
  the CartPole-v1 dimension contract.
- [x] AC-5: `log_std` paragraph in the doc-comment documents the
  discrete-vs-continuous divergence and the exact 12-key count.

## Architecture

### Lint baseline (REQ-1)

The crate-root attribute block in `lib.rs`:

- Denies: `unsafe_code`, `rust_2018_idioms`,
  `missing_debug_implementations`, `missing_docs`.
- Warns: `clippy::all`, `clippy::pedantic`.
- Allows (per-lint, with documented rationale):
  - cast lints (`cast_possible_truncation`, `cast_precision_loss`,
    `cast_sign_loss`, `cast_possible_wrap`, `cast_lossless`) —
    dimension math intrinsic to tensor indexing.
  - `must_use_candidate`, `doc_markdown`, `needless_pass_by_value`,
    `unnecessary_wraps`, `uninlined_format_args` — pedantic lints
    consistently wrong for ML/numeric kernel code.

Mirrors the `ferrotorch-bert` / `ferrotorch-graph` baseline. The
`deny(unsafe_code)` is load-bearing — the policy composition involves
no FFI and stays inside the safe-Rust `Tensor<f32>` surface.

### Public module declarations (REQ-2)

```rust
pub mod mlp_policy;
pub mod safetensors_loader;
```

Two modules cover the full crate scope:

- `mlp_policy` — the `MlpPolicy` composition + its sub-types
  (`MlpExtractor`, `ActionNet`, `ValueHead`) + `MlpPolicyConfig`.
- `safetensors_loader` — `load_ppo_policy` + the `DropReport` audit
  trail.

### Prelude re-exports (REQ-3)

```rust
pub use mlp_policy::{ActionNet, MlpExtractor, MlpPolicy, MlpPolicyConfig, ValueHead};
pub use safetensors_loader::{DropReport, load_ppo_policy};
```

The canonical user import is:

```rust
use ferrotorch_rl::{MlpPolicy, MlpPolicyConfig, load_ppo_policy};
```

This is the import used by `ferrotorch-rl/examples/ppo_policy_dump.rs`
(the production binary driven by `scripts/verify_rl_inference.py`).

### Architectural diagram (REQ-4)

The doc-comment includes:

```text
MlpPolicy
├── features_extractor: FlattenExtractor (identity on 1-D obs)
├── mlp_extractor: MlpExtractor
│   ├── policy_net: Linear(obs_dim → 64) → Tanh → Linear(64 → 64) → Tanh
│   └── value_net:  Linear(obs_dim → 64) → Tanh → Linear(64 → 64) → Tanh
├── action_net: Linear(64 → n_actions)        ← Categorical logits
└── value_net:  Linear(64 → 1)                ← scalar state value
```

Plus the share-features-extractor explanation: sb3 sets
`share_features_extractor=True` by default, but the "shared" extractor
is the FlattenExtractor (identity on 1-D obs), not the MLP. The
policy and value trunks have entirely separate weights — they only
share the (no-op) FlattenExtractor wrapper.

### `log_std` divergence (REQ-5)

sb3's `ActorCriticPolicy` only emits a `log_std` parameter for
continuous-action policies (DiagGaussianDistribution). For the
discrete `Categorical` distribution used by CartPole-v1, the state
dict has exactly 12 parameter keys:

- 4 for `mlp_extractor.policy_net.{0,2}.{weight,bias}`
- 4 for `mlp_extractor.value_net.{0,2}.{weight,bias}`
- 2 for `action_net.{weight,bias}`
- 2 for `value_net.{weight,bias}`

The doc-comment documents this explicitly so the `DropReport`
behavior on a clean pin (empty) is grounded in the parameter-key
contract.

### Non-test production consumers

- `ferrotorch-rl/examples/ppo_policy_dump.rs:34` —
  `use ferrotorch_rl::{MlpPolicyConfig, load_ppo_policy};` is the
  canonical production consumer (the binary driven by
  `scripts/verify_rl_inference.py` for the parity harness).
- `ferrotorch-rl/src/safetensors_loader.rs:36` —
  `use crate::mlp_policy::{MlpPolicy, MlpPolicyConfig};` consumes
  the policy type via the crate-internal path.
- `ferrotorch-rl/README.md:48` quotes the canonical import as the
  user-facing API contract.

### Upstream PyTorch mapping (R-DEV-7 deviation)

Upstream PyTorch ships `torch.nn.Module` and the building-block
layers; sb3's `ActorCriticPolicy` is a third-party Python package
that composes them. `ferrotorch-rl` is the Rust analog of sb3's
policy composition, mirrored byte-for-byte against the pinned
`sb3/ppo-CartPole-v1` safetensors so the parity harness can compare
forward outputs to the upstream Python model.

The contract preserved is the safetensors key layout and the
forward-pass numerical contract; the implementation is Rust-native
composition over `ferrotorch-nn::Module`.

## Parity contract

`parity_ops = []`. The crate root performs no numerical computation
of its own. Parity is owned by the submodules:

- `mlp_policy.rs` parity → `.design/ferrotorch-rl/mlp_policy.md`
  (forward-pass numerical contract vs sb3, parameter-shape
  contract).
- `safetensors_loader.rs` parity → `.design/ferrotorch-rl/safetensors_loader.md`
  (key-mapping byte-for-byte against sb3 `ActorCriticPolicy.state_dict()`).

End-to-end parity is verified by `scripts/verify_rl_inference.py`,
which drives `ferrotorch-rl/examples/ppo_policy_dump.rs` and
cross-checks its `action_logits` + `value` dumps against
stable-baselines3's Python forward output.

## Verification

The crate root has no lib-level tests directly — each submodule
ships its own `#[cfg(test)] mod tests`. Crate-wide gauntlet:

```bash
cargo test -p ferrotorch-rl --lib 2>&1 | tail -3
cargo clippy -p ferrotorch-rl --lib -- -D warnings 2>&1 | tail -3
cargo fmt -p ferrotorch-rl --check
```

The integration test
`ferrotorch-rl/tests/conformance_ppo_cartpole.rs` exercises the
public symbols end-to-end through the canonical user-import path.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-rl --lib 2>&1 | tail -3
```

Expected: `8 passed` across `mlp_policy::tests` (4) and
`safetensors_loader::tests` (3) plus zero crate-root tests.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: crate-root attribute block (`#![deny(unsafe_code)]` + per-lint `#![allow]` blocks with documented rationale) at top of `ferrotorch-rl/src/lib.rs:1-29` mirroring `ferrotorch-bert`/`ferrotorch-graph` posture; non-test consumer: the workspace clippy gate (`cargo clippy -p ferrotorch-rl -- -D warnings`) consumes this baseline; the per-lint allows are referenced by every production file in the crate (`mlp_policy.rs` cast operations and `safetensors_loader.rs` string formatting would otherwise trip pedantic lints). |
| REQ-2 | SHIPPED | impl: `pub mod mlp_policy; pub mod safetensors_loader;` declarations in `ferrotorch-rl/src/lib.rs:77-78`; non-test consumer: `ferrotorch-rl/src/safetensors_loader.rs:36` `use crate::mlp_policy::{MlpPolicy, MlpPolicyConfig};` consumes the `mlp_policy` module via this declaration; the production binary `ferrotorch-rl/examples/ppo_policy_dump.rs:34` consumes both modules transitively via the prelude re-exports. |
| REQ-3 | SHIPPED | impl: `pub use mlp_policy::{ActionNet, MlpExtractor, MlpPolicy, MlpPolicyConfig, ValueHead}` and `pub use safetensors_loader::{DropReport, load_ppo_policy}` in `ferrotorch-rl/src/lib.rs:80-81`; non-test consumer: `ferrotorch-rl/examples/ppo_policy_dump.rs:34` `use ferrotorch_rl::{MlpPolicyConfig, load_ppo_policy};` invokes the production binary via the canonical user-facing import path. |
| REQ-4 | SHIPPED | impl: crate doc-comment paragraphs at `ferrotorch-rl/src/lib.rs:31-75` including the architectural diagram (lines 42-50) and the CartPole-v1 dimension contract; non-test consumer: `ferrotorch-rl/src/mlp_policy.rs:71-78` `MlpPolicyConfig::cartpole_v1()` returns `Self { obs_dim: 4, hidden: 64, n_actions: 2 }` which is the production realisation of the documented contract; `ferrotorch-rl/examples/ppo_policy_dump.rs:196-200` constructs the same `MlpPolicyConfig` via the CLI arguments. |
| REQ-5 | SHIPPED | impl: crate doc-comment paragraph "On `log_std`" at `ferrotorch-rl/src/lib.rs:69-75` documenting the 12-key state-dict contract; non-test consumer: `ferrotorch-rl/src/mlp_policy.rs:547-559` `Module::named_parameters` for `MlpPolicy` emits exactly those 12 keys (verified by the production safetensors loader's expected-key set construction at `ferrotorch-rl/src/safetensors_loader.rs:80-84`). |
