# ferrotorch-rl::safetensors_loader — sb3 PPO MlpPolicy safetensors loader

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/serialization.py
  - torch/nn/modules/module.py
-->

## Summary

`ferrotorch-rl/src/safetensors_loader.rs` exposes `load_ppo_policy`,
a single entry point that decodes a `model.safetensors` mirror of an
sb3 `ActorCriticPolicy.state_dict()` into a populated
`MlpPolicy`. The loader is the audit-rail wrapper around
`ferrotorch-serialize::load_safetensors` + the standard
`Module::load_state_dict` machinery — every upstream key must either
land in a parameter on `MlpPolicy` or appear in the returned
`DropReport`. For the canonical `sb3/ppo-CartPole-v1` pin the report
is always empty; a non-empty report on a clean pin is the #1141
class of state-dict-drop bug. Upstream PyTorch ships `torch.load` /
`torch.save` for pickle-based serialization; ferrotorch uses
SafeTensors (R-DEV-3 / R-DEV-7) for memory-safe deserialization.

## Requirements

- REQ-1: `pub struct DropReport { pub unmapped: Vec<String> }` —
  the audit trail returned by `load_ppo_policy`. `Debug + Default + Clone`.
  `unmapped` is the sorted list of safetensors keys present in the
  file but not matched against any parameter on `MlpPolicy`. Empty
  for a clean pin.
- REQ-2: `pub fn load_ppo_policy(weights_path: &Path, cfg:
  MlpPolicyConfig, strict: bool) -> FerrotorchResult<(MlpPolicy,
  DropReport)>` — the loader entry point.
- REQ-3: The loader uses `ferrotorch-serialize::load_safetensors::<f32>`
  to decode the file. SafeTensors errors are mapped to
  `FerrotorchError::InvalidArgument` with the file path embedded in
  the message.
- REQ-4: After decode, the loader builds a fresh
  `MlpPolicy::new(cfg)?`, collects the expected parameter-key set
  via `policy.named_parameters()`, and identifies every key in the
  state dict that is NOT in the expected set. Sorted into
  `unmapped`.
- REQ-5: When `strict == true`, a non-empty `unmapped` causes the
  loader to error with `FerrotorchError::InvalidArgument` naming
  the offending keys.
- REQ-6: When `strict == false`, the loader filters the state dict
  to only the expected keys and proceeds, returning the unmapped
  keys in the `DropReport`.
- REQ-7: The filtered state dict is passed to
  `policy.load_state_dict(&filtered, /* strict = */ true)` — the
  inner strict-mode is always true because we've already
  pre-filtered. The inner strict mode catches missing-key errors:
  every expected parameter MUST appear in the safetensors. A
  missing key is always fatal regardless of the outer `strict`
  argument.

## Acceptance Criteria

- [x] AC-1: Round-trip: dumping a fresh `MlpPolicy`'s `state_dict()`
  to safetensors, then loading via `load_ppo_policy(..., strict=true)`
  produces a policy whose `named_parameters()` carry the same
  tensor values (within `1e-7`).
- [x] AC-2: Round-trip forward: loaded policy's
  `forward(obs).action_logits` and `.value` match the source's
  outputs (within `1e-6`) on a fixed `[1, 4]` obs.
- [x] AC-3: A state dict with an extra upstream key (e.g. `log_std`)
  errors out under `strict=true` and is captured in the
  `DropReport.unmapped` under `strict=false`.
- [x] AC-4: `DropReport::default()` returns `DropReport { unmapped: vec![] }`.
- [x] AC-5: For the canonical `sb3/ppo-CartPole-v1` mirror, the
  loader returns an empty `DropReport.unmapped` (verified end-to-end
  by `scripts/verify_rl_inference.py` driving
  `ferrotorch-rl/examples/ppo_policy_dump.rs`).

## Architecture

### `DropReport` (REQ-1)

```rust
#[derive(Debug, Default, Clone)]
pub struct DropReport {
    pub unmapped: Vec<String>,
}
```

The audit-rail return type. The `unmapped` field carries the sorted
upstream safetensors keys that did NOT match any parameter on the
ferrotorch `MlpPolicy`. Empty for a clean pin.

This is the structural defense against the #1141 class of
state-dict-drop bug — a missing key silently dropped during load
would result in a wrong-output policy on first forward; the
`DropReport` mechanism forces the caller to acknowledge every
upstream key, either by matching it or by accepting its inclusion in
the report.

### `load_ppo_policy` (REQ-2..REQ-7)

The loader is a 5-step pipeline:

1. **Decode**: `let state = load_safetensors::<f32>(weights_path)
   .map_err(...)?;` — delegates the actual decode to
   `ferrotorch-serialize`. Errors carry the file path for
   debuggability.

2. **Build policy + expected key set**: `let mut policy =
   MlpPolicy::new(cfg)?; let expected: HashSet<String> = policy
   .named_parameters().into_iter().map(|(n, _)| n).collect();` — the
   expected key set is the ground truth from the Rust side.

3. **Identify unmapped keys**: walk `state.keys()`, push any key not
   in `expected` into `unmapped`. Sort.

4. **Strict-mode gate (REQ-5)**: when `strict == true && !unmapped.is_empty()`,
   error with `FerrotorchError::InvalidArgument` naming the offending
   keys (this is the loud-failure path for clean-pin regressions).

5. **Filter + load (REQ-6, REQ-7)**: filter the state dict to only
   the expected keys, then call
   `policy.load_state_dict(&filtered, /* strict = */ true)?` — the
   inner strict-mode catches missing-key errors. Return
   `(policy, DropReport { unmapped })`.

### Asymmetric strictness (REQ-5, REQ-6, REQ-7)

The outer `strict` argument controls **extra-key** behavior:

- `strict=true` + extra keys → error.
- `strict=false` + extra keys → warn-via-report, continue.

The inner `Module::load_state_dict` strict-mode is **always** `true`,
catching **missing-key** errors regardless of the outer mode:

- Missing-key → always fatal.

This asymmetry is deliberate. Extra keys are routine when the
upstream checkpoint has features ferrotorch hasn't translated yet
(e.g. `log_std` for continuous policies). Missing keys mean the
loaded policy will silently have an uninitialised parameter on first
forward — that's never acceptable.

### Non-test production consumers

- `ferrotorch-rl/examples/ppo_policy_dump.rs:34` —
  `use ferrotorch_rl::{MlpPolicyConfig, load_ppo_policy};` is the
  canonical production consumer (the parity-harness binary driven by
  `scripts/verify_rl_inference.py`).
- `ferrotorch-rl/examples/ppo_policy_dump.rs:202` —
  `let (policy, report) = load_ppo_policy(&weights_path, cfg, /* strict = */ true)?;`
  invokes the loader in strict mode against the HF-mirrored
  safetensors.
- `ferrotorch-rl/examples/ppo_policy_dump.rs:203-206` —
  `eprintln!("[ppo_policy_dump] loaded weights: unmapped={:?}", report.unmapped);`
  consumes the `DropReport` for diagnostic output.
- `ferrotorch-rl/src/lib.rs` re-exports `DropReport, load_ppo_policy`
  as the crate's public API.
- `ferrotorch-rl/README.md:48` documents the canonical
  user-facing import.

### Upstream PyTorch mapping (R-DEV-3 + R-DEV-7 deviation)

Upstream PyTorch ships `torch.save` / `torch.load` (Python pickle-
based) for state-dict serialization. SafeTensors is the
Hugging-Face-led alternative format that is memory-safe (no
arbitrary-code execution on load) and has a stable wire format
(R-DEV-3 — on-disk format is an external specification). ferrotorch
adopts SafeTensors as the canonical serialization format throughout
(R-DEV-7 — Rust ecosystem analog `safetensors` crate is materially
better), so the loader uses `ferrotorch-serialize::load_safetensors`
rather than a Python-pickle-protocol implementation.

The upstream `sb3/ppo-CartPole-v1` checkpoint ships as a zip
containing pickle files; the HF mirror at `ferrotorch/ppo-cartpole-v1`
re-serializes it as `model.safetensors` so the loader can consume it
directly.

## Parity contract

`parity_ops = []`. The loader has no numerical computation of its
own; parity is structural:

- **Key-mapping byte-for-byte**: every safetensors key in the upstream
  mirror must match a parameter on `MlpPolicy`. The 12-key contract
  is the structural ground truth (documented in
  `.design/ferrotorch-rl/mlp_policy.md` REQ-14).
- **Strict-mode extra-key rejection**: a clean pin has zero extras;
  any non-empty `unmapped` under `strict=true` causes the loader to
  fail loudly with the offending keys named.
- **Inner strict-mode missing-key rejection**: every expected
  parameter MUST appear in the safetensors. Missing keys are always
  fatal — they would result in a silently-uninitialised parameter
  on first forward (the #1141 class of bug).
- **Forward-output parity**: a loaded policy's forward output matches
  sb3's forward output within `1e-6` on the reference observation
  (`_value_parity_obs.bin` shipped on the HF mirror). Verified
  end-to-end by `scripts/verify_rl_inference.py`.

## Verification

Tests in `mod tests in safetensors_loader.rs` (3 tests):

- `round_trip_into_mlp_policy` — AC-1 pin (parameter-value
  preservation through dump+load).
- `round_trip_forward_matches` — AC-2 pin (forward-output
  preservation).
- `unmapped_keys_strict_errors` — AC-3 pin (strict-mode + report
  semantics with a synthetic extra `log_std` key).

End-to-end sb3-forward-parity is exercised by:

- `ferrotorch-rl/tests/conformance_ppo_cartpole.rs` (integration).
- `scripts/verify_rl_inference.py` driving
  `ferrotorch-rl/examples/ppo_policy_dump.rs` against the pinned HF
  mirror.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-rl --lib safetensors_loader:: 2>&1 | tail -3
```

Expected: `3 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct DropReport { pub unmapped: Vec<String> }` at `DropReport in ferrotorch-rl/src/safetensors_loader.rs` with `#[derive(Debug, Default, Clone)]`; non-test consumer: `ferrotorch-rl/examples/ppo_policy_dump.rs,261` `let (policy, report) = load_ppo_policy(...)?; ... format!(... "\"unmapped\":{}", report.unmapped.len())` — the production binary consumes the report for diagnostic output and final JSON verdict. Re-exported as `ferrotorch_rl::DropReport`. |
| REQ-2 | SHIPPED | impl: `pub fn load_ppo_policy(weights_path: &Path, cfg: MlpPolicyConfig, strict: bool) -> FerrotorchResult<(MlpPolicy, DropReport)>` at `ferrotorch-rl/src/safetensors_loader.rs:66-107`; non-test consumer: `ferrotorch-rl/examples/ppo_policy_dump.rs:202` `let (policy, report) = load_ppo_policy(&weights_path, cfg, /* strict = */ true)?;` invokes the loader in the production parity-harness binary. Re-exported as `ferrotorch_rl::load_ppo_policy`. |
| REQ-3 | SHIPPED | impl: `ferrotorch-rl/src/safetensors_loader.rs:71-77` `let state = load_safetensors::<f32>(weights_path).map_err(|e| FerrotorchError::InvalidArgument { message: format!("load_ppo_policy: failed to decode safetensors {}: {e}", weights_path.display()) })?;`; non-test consumer: the only call-site of the loader is `ferrotorch-rl/examples/ppo_policy_dump.rs:202`, which depends on the path-embedded error message for the user-facing diagnostic. |
| REQ-4 | SHIPPED | impl: `ferrotorch-rl/src/safetensors_loader.rs:79-91` builds `policy = MlpPolicy::new(cfg)?`, computes `expected: HashSet<String>` from `policy.named_parameters()`, walks `state.keys()` to identify unmapped keys, sorts them; non-test consumer: `ferrotorch-rl/examples/ppo_policy_dump.rs:202` invokes this path; the in-file test `unmapped_keys_strict_errors` documents the contract (the production binary's `--strict=true` invocation depends on this branch). |
| REQ-5 | SHIPPED | impl: `ferrotorch-rl/src/safetensors_loader.rs:92-96` `if strict && !unmapped.is_empty() { return Err(FerrotorchError::InvalidArgument { message: format!("load_ppo_policy: unmapped upstream keys (strict mode): {unmapped:?}") }); }`; non-test consumer: `ferrotorch-rl/examples/ppo_policy_dump.rs:202` invokes with `strict = true` — a non-empty `unmapped` would surface as a loud error preventing the parity-harness from proceeding with a partially-loaded policy. |
| REQ-6 | SHIPPED | impl: `ferrotorch-rl/src/safetensors_loader.rs:98-106` `let filtered = state.into_iter().filter(|(k, _)| expected.contains(k)).collect(); policy.load_state_dict(&filtered, /* strict = */ true)?; Ok((policy, DropReport { unmapped }))`; non-test consumer: the in-file test `unmapped_keys_strict_errors` documents the `strict=false` path returning the unmapped keys via the report; `ferrotorch-rl/examples/ppo_policy_dump.rs:202` (with `strict=true`) depends on the filter step to ensure the inner `load_state_dict` doesn't choke on an extra key that should already have been rejected by REQ-5. |
| REQ-7 | SHIPPED | impl: `ferrotorch-rl/src/safetensors_loader.rs:105` `policy.load_state_dict(&filtered, /* strict = */ true)?;` — the inner strict-mode is hardcoded `true`; non-test consumer: `ferrotorch-rl/examples/ppo_policy_dump.rs:202` depends on missing-key detection (a missing parameter would produce a wrong-output policy on first forward, the #1141 class of bug). |
