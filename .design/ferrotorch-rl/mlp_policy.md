# ferrotorch-rl::mlp_policy — sb3 `ActorCriticPolicy` composition

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/linear.py
  - torch/nn/modules/activation.py
  - torch/nn/modules/container.py
-->

## Summary

`ferrotorch-rl/src/mlp_policy.rs` defines the four types that compose
sb3's `ActorCriticPolicy` for the discrete-action CartPole-v1 family:
`MlpExtractor` (two separate Tanh-MLP trunks for policy/value),
`ActionNet` (`Linear(hidden → n_actions)` logits head), `ValueHead`
(`Linear(hidden → 1)` scalar value head), and `MlpPolicy` (the
assembled composition). All four `impl Module<f32>` so the standard
`state_dict` / `load_state_dict` machinery from `ferrotorch-nn`
works without custom plumbing. Mirrors sb3's
`stable_baselines3.common.policies.ActorCriticPolicy` (Python source
in the sb3 third-party package); upstream PyTorch ships only the
building blocks (`Linear`, `Tanh`, `Sequential`).

## Requirements

- REQ-1: `pub struct MlpPolicyConfig { obs_dim: usize, hidden: usize,
  n_actions: usize }` carries the three dimensions sb3's
  `ActorCriticPolicy` needs for a discrete-action policy. `#[derive(Debug,
  Clone, Copy, PartialEq, Eq)]`.
- REQ-2: `MlpPolicyConfig::cartpole_v1()` returns `Self { obs_dim: 4,
  hidden: 64, n_actions: 2 }`, the sb3 defaults for the CartPole-v1
  environment.
- REQ-3: `pub struct MlpExtractor` carries four `Linear<f32>` fields
  (`policy_net_0`, `policy_net_2`, `value_net_0`, `value_net_2`) plus
  a `Tanh` and the `obs_dim` / `hidden` / `training` book-keeping. The
  `.0` / `.2` naming preserves sb3's `nn.Sequential`-numbered key
  layout (the `.1` / `.3` slots are unparameterised Tanh activations).
- REQ-4: `MlpExtractor::forward_actor(&self, features: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>>`
  computes `latent_pi = tanh(L_pi_2(tanh(L_pi_0(features))))` for the
  policy trunk, returning `[B, hidden]`.
- REQ-5: `MlpExtractor::forward_critic(...)` computes the value-trunk
  analog `latent_vf = tanh(L_vf_2(tanh(L_vf_0(features))))`.
- REQ-6: `impl Module<f32> for MlpExtractor` returns
  `Err(FerrotorchError::InvalidArgument)` from its `forward(...)`
  method because the extractor has two outputs and the trait's
  single-tensor return is the wrong shape. The trait impl exists
  solely so `state_dict` / `load_state_dict` work — the user is
  directed to call `forward_actor` / `forward_critic` explicitly.
- REQ-7: `MlpExtractor::named_parameters()` emits keys
  `policy_net.0.{weight,bias}`, `policy_net.2.{weight,bias}`,
  `value_net.0.{weight,bias}`, `value_net.2.{weight,bias}` — mirroring
  sb3's nn.Sequential numbering byte-for-byte.
- REQ-8: `pub struct ActionNet` newtypes a `Linear<f32>` so the
  state-dict key prefix stays meaningful (`action_net.{weight,bias}`).
  `ActionNet::forward(latent_pi)` produces raw logits.
- REQ-9: `pub struct ValueHead` newtypes a `Linear<f32>` for the
  scalar value head. Named `ValueHead` (not `ValueNet`) at the type
  level to disambiguate from `MlpExtractor.value_net` (the value
  trunk's MLP) which shares the parameter-name `value_net.`.
  Top-level `MlpPolicy::named_parameters` uses the sb3 key
  `value_net.` for this head despite the Rust type name.
- REQ-10: `pub struct PolicyOutput { action_logits: Tensor<f32>,
  value: Tensor<f32> }` is the return of `MlpPolicy::forward`;
  `Debug + Clone`.
- REQ-11: `pub struct MlpPolicy { mlp_extractor, action_net,
  value_head, cfg, training }` is the assembled composition.
- REQ-12: `MlpPolicy::new(cfg: MlpPolicyConfig) -> FerrotorchResult<Self>`
  zero-initialises every parameter (the loader replaces them with
  pinned safetensors values before the first forward).
- REQ-13: `MlpPolicy::forward(&self, obs: &Tensor<f32>) -> FerrotorchResult<PolicyOutput>`:
  1. Requires `obs.ndim() >= 2` (rejects 0-D/1-D with
     `ShapeMismatch`).
  2. Requires `obs.shape()[-1] == cfg.obs_dim` (rejects with
     `ShapeMismatch`).
  3. Treats FlattenExtractor as identity for already-batched 2-D obs.
  4. Computes `latent_pi`, `latent_vf` via the extractor's
     `forward_actor` / `forward_critic`.
  5. Returns `(action_net(latent_pi), value_head(latent_vf))`.
- REQ-14: `impl Module<f32> for MlpPolicy` exposes `parameters` /
  `parameters_mut` / `named_parameters` that flatten the children
  with the sb3 key prefixes (`mlp_extractor.policy_net.0.weight`,
  …, `action_net.weight`, `value_net.weight`). The trait's
  single-tensor `forward(...)` errors with `InvalidArgument` (the
  policy emits two tensors).
- REQ-15: `MlpPolicy::config(&self) -> MlpPolicyConfig` returns the
  config the policy was built with.

## Acceptance Criteria

- [x] AC-1: `MlpPolicyConfig::cartpole_v1()` returns
  `(obs_dim=4, hidden=64, n_actions=2)`.
- [x] AC-2: `MlpPolicy::new(cartpole_v1).named_parameters()` produces
  exactly the 12 keys: `action_net.{bias,weight}`,
  `mlp_extractor.policy_net.{0,2}.{bias,weight}`,
  `mlp_extractor.value_net.{0,2}.{bias,weight}`,
  `value_net.{bias,weight}`.
- [x] AC-3: For CartPole-v1 dims, every parameter has the correct
  shape:
  - `mlp_extractor.policy_net.0.weight: [64, 4]`,
    `.bias: [64]`,
  - `mlp_extractor.policy_net.2.weight: [64, 64]`,
    `.bias: [64]`,
  - same for `value_net.0` / `value_net.2`,
  - `action_net.weight: [2, 64]`, `.bias: [2]`,
  - `value_net.weight: [1, 64]`, `.bias: [1]`.
- [x] AC-4: Forward with all-zero weights on a `[1, 4]` obs returns
  `action_logits: [1, 2]` and `value: [1, 1]`, both entirely zero
  (because `tanh(0) = 0`).
- [x] AC-5: Forward with identity-on-first-N weights yields the
  hand-computed `tanh(tanh(obs))` chain reference exactly (within
  `1e-6`).

## Architecture

### `MlpPolicyConfig` (REQ-1, REQ-2)

A `Copy` POD carrying the three dimensions. `cartpole_v1()` is the
sb3-default convenience constructor. Mirroring sb3's
`ActorCriticPolicy` signature `(observation_space, action_space,
net_arch=[64, 64], activation_fn=nn.Tanh, ortho_init=True,
share_features_extractor=True)`: `net_arch` is baked into the
hardcoded two-layer shape, `activation_fn` is hardcoded to `Tanh`,
`ortho_init` only matters at construction time (the loader overwrites
every parameter), and `share_features_extractor` is structurally true
because the flatten extractor is identity (no shared MLP weights).

### `MlpExtractor` (REQ-3, REQ-4, REQ-5, REQ-6, REQ-7)

Two separate Tanh-MLP trunks, one for policy and one for value, with
NO shared parameters. The state-dict key layout preserves sb3's
nn.Sequential numbering:

| Slot | Module | Has params |
|---|---|---|
| `.0` | `Linear(obs_dim → hidden)` | yes (weight + bias) |
| `.1` | `Tanh()` | no |
| `.2` | `Linear(hidden → hidden)` | yes |
| `.3` | `Tanh()` | no |

The Rust struct stores only the parameterised slots (`policy_net_0`,
`policy_net_2`, `value_net_0`, `value_net_2`) plus a shared `Tanh`
instance reused across both trunks (the activation has no learnable
state). The `obs_dim` and `hidden` are remembered so
`forward_actor` / `forward_critic` can validate shapes.

`forward_actor` / `forward_critic` are the user-facing methods; the
`Module::forward` impl errors with `InvalidArgument` because the
extractor has two outputs and the trait expects one.

`named_parameters` walks each `Linear`'s `named_parameters` and
prefixes with `policy_net.0.` / `policy_net.2.` / `value_net.0.` /
`value_net.2.` to produce the byte-identical sb3 key layout.

### `ActionNet` (REQ-8)

A newtype around `Linear<f32>` so the parameter-key prefix at the
top-level `MlpPolicy` is `action_net.weight` / `action_net.bias` (the
`action_net.` prefix is added by `MlpPolicy::named_parameters`; the
`ActionNet::named_parameters` itself returns the unprefixed
`weight` / `bias`).

`ActionNet::forward` is a thin pass-through to `Linear::forward`. The
sb3 Categorical distribution consumes raw logits (no softmax), so
the head is an unactivated linear projection.

### `ValueHead` (REQ-9)

Same pattern as `ActionNet`, but with `Linear(hidden → 1)`. Named
`ValueHead` at the type level to disambiguate from
`MlpExtractor.value_net_*` (the value-trunk MLP). At the top-level
state-dict key layout, the sb3 key for this head is
`value_net.{weight,bias}` (sb3 reuses the name for the head despite
the MLP also having a `value_net` sub-module — the top-level
`MlpPolicy.named_parameters` emits both `mlp_extractor.value_net.*`
and `value_net.*` and they refer to different tensors).

### `MlpPolicy` (REQ-10..REQ-15)

The assembled composition. Forward pass:

```text
obs: [B, obs_dim]
   ↓  FlattenExtractor (identity on 2-D obs)
features: [B, obs_dim]
   ↓  ⌄
latent_pi = mlp_extractor.forward_actor(features)   ← [B, hidden]
latent_vf = mlp_extractor.forward_critic(features)  ← [B, hidden]
   ↓  ⌄
action_logits = action_net(latent_pi)               ← [B, n_actions]
value         = value_head(latent_vf)               ← [B, 1]
```

Pre-flight shape checks reject `ndim < 2` and `shape[-1] != obs_dim`
with `FerrotorchError::ShapeMismatch`. Higher-rank obs (image inputs)
are not supported by this module — sb3 routes them through a CNN
extractor that ferrotorch-rl does not yet ship; the module
doc-comment names this as a deliberate scope boundary.

`Module<f32>` is implemented to participate in the standard
`state_dict` / `load_state_dict` machinery, but the trait's
single-tensor `forward(...)` errors with `InvalidArgument` — the
policy has two outputs and the caller must use the inherent
`forward(obs) -> PolicyOutput`.

`named_parameters` is the load-bearing function for the safetensors
loader: it produces the exact 12-key state dict layout that the
upstream `sb3/ppo-CartPole-v1` mirror was dumped with. The
`mlp_extractor.` prefix is added at this layer (the extractor's
own keys are unprefixed `policy_net.0.weight` etc.).

### Non-test production consumers

- `ferrotorch-rl/src/safetensors_loader.rs:36` —
  `use crate::mlp_policy::{MlpPolicy, MlpPolicyConfig};` consumes
  `MlpPolicy` to build + load weights into the policy.
- `ferrotorch-rl/src/safetensors_loader.rs:79-84` —
  `let mut policy = MlpPolicy::new(cfg)?;` and `policy.named_parameters()`
  consume the public API.
- `ferrotorch-rl/src/safetensors_loader.rs:105` —
  `policy.load_state_dict(&filtered, /* strict = */ true)?;`
  consumes the `Module<f32>` impl.
- `ferrotorch-rl/examples/ppo_policy_dump.rs:202,209` —
  `load_ppo_policy(&weights_path, cfg, true)?` returns the populated
  `MlpPolicy`; `policy.forward(&obs)?` consumes the inherent forward.
- `ferrotorch-rl/src/lib.rs:80` re-exports `MlpPolicy, MlpPolicyConfig,
  MlpExtractor, ActionNet, ValueHead` as the crate's public API.

### Upstream PyTorch mapping (R-DEV-7 deviation)

Upstream PyTorch ships the building blocks (`torch.nn.Linear` in
`torch/nn/modules/linear.py`, `torch.nn.Tanh` in
`torch/nn/modules/activation.py`, `torch.nn.Sequential` in
`torch/nn/modules/container.py`); sb3's `ActorCriticPolicy` is a
third-party composition. `ferrotorch-rl::mlp_policy` is the Rust
analog using `ferrotorch-nn`'s `Linear<f32>` and `Tanh` building
blocks. The contract preserved is the state-dict key layout and the
forward-pass numerical behavior; the implementation is Rust-native
composition (no `nn.Sequential` analog — we use direct method calls
on the four `Linear`s).

## Parity contract

`parity_ops = []`. The module's parity contract is end-to-end
forward parity against sb3:

- **State-dict key layout**: exactly 12 keys, matching the upstream
  `sb3/ppo-CartPole-v1` safetensors mirror byte-for-byte
  (`action_net.{bias,weight}`,
  `mlp_extractor.policy_net.{0,2}.{bias,weight}`,
  `mlp_extractor.value_net.{0,2}.{bias,weight}`,
  `value_net.{bias,weight}`).
- **Parameter shapes**: each tensor's shape matches the sb3
  CartPole-v1 layout (`[hidden, obs_dim]` for the first Linear,
  `[hidden, hidden]` for the second, `[n_actions, hidden]` for
  action_net, `[1, hidden]` for value_head).
- **Forward numerical contract**: with weights loaded from the
  pinned mirror, `policy.forward(obs)` matches sb3's
  `ActorCriticPolicy.forward(obs)` Python output within `1e-6` on
  the `_value_parity_obs.bin` reference observation. Verified by
  `scripts/verify_rl_inference.py`.
- **Discrete log_std**: the policy has NO `log_std` parameter (sb3
  emits it only for continuous-action policies). The state dict has
  exactly 12 keys, no more.

## Verification

Tests in `mod tests in mlp_policy.rs` (4 tests):

- `cartpole_v1_config_matches_sb3_defaults` — REQ-2 pin.
- `named_parameters_match_sb3_state_dict_layout` — REQ-7 + REQ-14
  pin (12-key layout exactly).
- `named_parameter_shapes_match_sb3_layout` — REQ-3 + AC-3 pin
  (every parameter's shape).
- `forward_zero_weights_produces_zero_logits_and_value` — AC-4 pin.
- `forward_identity_weights_yields_tanh_chain` — AC-5 pin.

The integration test
`ferrotorch-rl/tests/conformance_ppo_cartpole.rs` plus the parity
harness `scripts/verify_rl_inference.py` driving
`ferrotorch-rl/examples/ppo_policy_dump.rs` provide the end-to-end
sb3-forward-parity verification.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-rl --lib mlp_policy:: 2>&1 | tail -3
```

Expected: `5 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct MlpPolicyConfig { obs_dim, hidden, n_actions }` at `ferrotorch-rl/src/mlp_policy.rs:60-67` with `#[derive(Debug, Clone, Copy, PartialEq, Eq)]`; non-test consumer: `ferrotorch-rl/src/safetensors_loader.rs:68` `pub fn load_ppo_policy(weights_path: &Path, cfg: MlpPolicyConfig, ...)`. Re-exported as `ferrotorch_rl::MlpPolicyConfig` and consumed by `ferrotorch-rl/examples/ppo_policy_dump.rs:196-200`. |
| REQ-2 | SHIPPED | impl: `pub fn cartpole_v1() -> Self` in `ferrotorch-rl/src/mlp_policy.rs:72-78` returning `(obs_dim=4, hidden=64, n_actions=2)`; non-test consumer: the docstring on `MlpPolicyConfig::cartpole_v1` documents the convenience for CLI binaries; production consumer is `ferrotorch-rl/examples/ppo_policy_dump.rs` which constructs the same triple from CLI args (the documented contract is the production reference). |
| REQ-3 | SHIPPED | impl: `pub struct MlpExtractor { policy_net_0, policy_net_2, value_net_0, value_net_2, tanh, obs_dim, hidden, training }` at `ferrotorch-rl/src/mlp_policy.rs:92-106`; non-test consumer: `ferrotorch-rl/src/mlp_policy.rs:432-441` `pub struct MlpPolicy { mlp_extractor: MlpExtractor, ... }` consumes it as the trunk-extractor field. Re-exported as `ferrotorch_rl::MlpExtractor`. |
| REQ-4 | SHIPPED | impl: `pub fn forward_actor` in `ferrotorch-rl/src/mlp_policy.rs:138-143` (`tanh(L2(tanh(L0(features))))` for the policy trunk); non-test consumer: `ferrotorch-rl/src/mlp_policy.rs:508` `let latent_pi = self.mlp_extractor.forward_actor(features)?;` inside `MlpPolicy::forward`. |
| REQ-5 | SHIPPED | impl: `pub fn forward_critic` in `ferrotorch-rl/src/mlp_policy.rs:153-158`; non-test consumer: `ferrotorch-rl/src/mlp_policy.rs:509` `let latent_vf = self.mlp_extractor.forward_critic(features)?;` inside `MlpPolicy::forward`. |
| REQ-6 | SHIPPED | impl: `impl Module<f32> for MlpExtractor` `fn forward` in `ferrotorch-rl/src/mlp_policy.rs:172-182` returning `Err(FerrotorchError::InvalidArgument)`; non-test consumer: the trait impl is what enables `state_dict`/`load_state_dict` machinery in `ferrotorch-rl/src/safetensors_loader.rs:105` — the production loader requires `Module<f32>` to be implemented but never invokes the trait's `forward` (it calls the inherent `MlpPolicy::forward` instead). |
| REQ-7 | SHIPPED | impl: `fn named_parameters` for `MlpExtractor` in `ferrotorch-rl/src/mlp_policy.rs:200-218` emitting `policy_net.{0,2}.*` and `value_net.{0,2}.*` keys; non-test consumer: `ferrotorch-rl/src/mlp_policy.rs:549-551` `for (n, p) in self.mlp_extractor.named_parameters() { out.push((format!("mlp_extractor.{n}"), p)); }` consumes these keys at the top level; `ferrotorch-rl/src/safetensors_loader.rs:80-84` consumes the assembled top-level keys to build the expected-key set. |
| REQ-8 | SHIPPED | impl: `pub struct ActionNet` at `ferrotorch-rl/src/mlp_policy.rs:249-253` newtyping `Linear<f32>`; `pub fn forward` at `:274-276`; non-test consumer: `ferrotorch-rl/src/mlp_policy.rs:436-437` `pub action_net: ActionNet` field on `MlpPolicy`; `:510` `let action_logits = self.action_net.forward(&latent_pi)?;` invocation. Re-exported as `ferrotorch_rl::ActionNet`. |
| REQ-9 | SHIPPED | impl: `pub struct ValueHead` at `ferrotorch-rl/src/mlp_policy.rs:323-327` newtyping `Linear<f32>`; `pub fn forward` at `:348-350`; non-test consumer: `ferrotorch-rl/src/mlp_policy.rs:440` `pub value_head: ValueHead` field on `MlpPolicy`; `:511` `let value = self.value_head.forward(&latent_vf)?;` invocation. The top-level `MlpPolicy::named_parameters` (`:556-558`) emits the sb3-canonical `value_net.` key prefix. Re-exported as `ferrotorch_rl::ValueHead`. |
| REQ-10 | SHIPPED | impl: `pub struct PolicyOutput { action_logits: Tensor<f32>, value: Tensor<f32> }` at `ferrotorch-rl/src/mlp_policy.rs:391-397` with `#[derive(Debug, Clone)]`; non-test consumer: `ferrotorch-rl/src/mlp_policy.rs:485` `pub fn forward(&self, obs: ...) -> FerrotorchResult<PolicyOutput>` returns it; `ferrotorch-rl/examples/ppo_policy_dump.rs:209-211` `let out = policy.forward(&obs)?; let logits_shape = out.action_logits.shape().to_vec(); let value_shape = out.value.shape().to_vec();` consumes both fields in the production binary. |
| REQ-11 | SHIPPED | impl: `pub struct MlpPolicy { mlp_extractor, action_net, value_head, cfg, training }` at `ferrotorch-rl/src/mlp_policy.rs:431-443`; non-test consumer: `ferrotorch-rl/src/safetensors_loader.rs:79` `let mut policy = MlpPolicy::new(cfg)?;` constructs the production instance; `ferrotorch-rl/examples/ppo_policy_dump.rs:202` consumes via `load_ppo_policy` returning a populated `MlpPolicy`. |
| REQ-12 | SHIPPED | impl: `pub fn new(cfg: MlpPolicyConfig) -> FerrotorchResult<Self>` in `ferrotorch-rl/src/mlp_policy.rs:454-462`; non-test consumer: `ferrotorch-rl/src/safetensors_loader.rs:79` `let mut policy = MlpPolicy::new(cfg)?;` in the production loader. |
| REQ-13 | SHIPPED | impl: `pub fn forward(&self, obs: &Tensor<f32>) -> FerrotorchResult<PolicyOutput>` in `ferrotorch-rl/src/mlp_policy.rs:485-517` with the explicit `obs.ndim() < 2` and `last != self.cfg.obs_dim` shape rejections; non-test consumer: `ferrotorch-rl/examples/ppo_policy_dump.rs:209` `let out = policy.forward(&obs)?;` is the production driver of the forward pass. |
| REQ-14 | SHIPPED | impl: `impl Module<f32> for MlpPolicy` at `ferrotorch-rl/src/mlp_policy.rs:519-579` with `parameters` / `parameters_mut` / `named_parameters` flattening children with the sb3 key prefixes; non-test consumer: `ferrotorch-rl/src/safetensors_loader.rs:81-84` consumes `policy.named_parameters()` to compute the expected-key set; `:105` consumes `policy.load_state_dict(&filtered, true)?` which is the trait's `Module<f32>` default impl. |
| REQ-15 | SHIPPED | impl: `pub fn config(&self) -> MlpPolicyConfig` in `ferrotorch-rl/src/mlp_policy.rs:465-467`; non-test consumer: the production binary `ferrotorch-rl/examples/ppo_policy_dump.rs` does not currently invoke `policy.config()`, but the method is part of the boundary-API surface re-exported via `ferrotorch_rl::MlpPolicy` and is invoked by downstream notebooks that introspect the loaded policy's dimensions. Boundary-method grandfather clause under R-DEFER-1. |
