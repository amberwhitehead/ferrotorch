# ferrotorch-graph/safetensors_loader — pinned `GcnNet` loader + audit `DropReport`

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/
-->

## Summary

`ferrotorch-graph/src/safetensors_loader.rs` is the entry point that
turns a pinned `model.safetensors` mirror of a PyG `GCNConv` pair
into a ready-to-forward `GcnNet`. The upstream PyG `state_dict`
keys (`conv1.bias`, `conv1.lin.weight`, `conv2.bias`,
`conv2.lin.weight`) match what `GcnNet::named_parameters` produces,
so the loader is mostly a pass-through into the standard
`ferrotorch_nn::Module::load_state_dict` machinery. The wrapper adds
two things on top: a `DropReport` documenting any upstream keys the
ferrotorch model did NOT consume (the #1141 audit rail — every key
must either land in a parameter or appear in the report), and a
`strict` flag that promotes a non-empty `DropReport` into a hard
error.

## Requirements

- REQ-1: `pub struct DropReport` with a single `pub unmapped:
  Vec<String>` field deriving `Debug, Default, Clone`. Empty on a
  canonical pin; non-empty entries are state-dict-drop bugs the
  audit rail surfaces.
- REQ-2: `pub fn load_gcn_net(weights_path, in_features, hidden,
  num_classes, strict) -> FerrotorchResult<(GcnNet, DropReport)>`
  decodes the safetensors via `ferrotorch_serialize::load_safetensors`,
  constructs a fresh `GcnNet`, computes the expected-key set from
  `net.named_parameters()`, filters the loaded state to only those
  keys, drives `Module::load_state_dict(filtered, strict=true)`, and
  returns the `(net, report)` pair.
- REQ-3: When `strict=true` and the safetensors holds any key not in
  the expected set, the function returns
  `FerrotorchError::InvalidArgument` with the unmapped keys listed.
  When `strict=false`, the unmapped keys land in the `DropReport`
  and loading continues with the matching subset.
- REQ-4: Safetensors decode failure is propagated as
  `FerrotorchError::InvalidArgument` with the file path in the
  message (rather than passing through the raw `safetensors` error
  type) so the failure mode is uniform with the rest of
  ferrotorch's error surface.
- REQ-5: The inner `Module::load_state_dict` call always uses
  `strict=true` regardless of the outer flag, because the outer
  flag controls extras (upstream keys not in the ferrotorch model);
  the inner strict mode catches the opposite class of bug
  (ferrotorch parameters with no upstream key), which is always
  fatal — a partial parameter load would silently leave a
  zero-initialised conv2.bias in place and skew every Cora
  prediction.

## Acceptance Criteria

- [x] AC-1: `DropReport::default()` produces an empty `unmapped`
  vector.
- [x] AC-2: A round-trip (save `GcnNet::state_dict()` to a temp
  safetensors, reload via `load_gcn_net`) recovers parameters with
  byte-identical values, and the returned `DropReport.unmapped` is
  empty (`round_trip_into_gcn_net`).
- [x] AC-3: `strict=true` with extra upstream keys errors out with
  `InvalidArgument` (architectural — the strict branch
  unconditionally returns `Err` on non-empty `unmapped`).
- [x] AC-4: Decode failure on a malformed safetensors path produces
  `InvalidArgument` carrying the path (architectural — the
  `.map_err(...)` rewriter at the call site preserves the path).

## Architecture

### `DropReport` (REQ-1)

Defined at `ferrotorch-graph/src/safetensors_loader.rs:37-41` as a
minimal `Debug, Default, Clone` struct with a single `unmapped:
Vec<String>` field. The empty default lets callers handle the
"nothing went wrong" path without unwrapping an Option, and the
struct is intentionally `pub` so the example binary can surface the
report fields to its stdout JSON verdict line
(`ferrotorch-graph/examples/gcn_inference_dump.rs:246, 282`).

### `load_gcn_net` orchestration (REQ-2, REQ-3, REQ-4, REQ-5)

Implementation at `ferrotorch-graph/src/safetensors_loader.rs:57-97`:

1. `load_safetensors::<f32>(weights_path)` decodes the file
   (line 64). The `.map_err(|e| FerrotorchError::InvalidArgument
   { message: ...path display + ': {e}' })?` adapter (lines 65-70)
   surfaces the path in the error message, satisfying REQ-4.
2. `GcnNet::new(in_features, hidden, num_classes)?` constructs the
   target model (line 72).
3. `let expected: HashSet<String> = net.named_parameters().into_iter()
   .map(|(n, _)| n).collect();` (lines 73-74) computes the
   ferrotorch-side expected keys via the canonical
   `Module::named_parameters` contract.
4. The loop over `state.keys()` (lines 76-80) collects every
   upstream key not in `expected` into `unmapped` (sorted at line
   81 for deterministic diagnostic output).
5. The `if strict && !unmapped.is_empty()` branch (lines 82-86)
   converts a non-empty report into `InvalidArgument` — REQ-3's
   strict-true behaviour.
6. The filter at lines 91-94 keeps only the keys the ferrotorch
   model knows about, then `net.load_state_dict(&filtered,
   /* strict = */ true)?` (line 95) drives the standard
   `Module::load_state_dict` machinery. The inner `strict=true` is
   intentional — REQ-5 explains the rationale (catching the missing-
   parameter class of bug).
7. `Ok((net, DropReport { unmapped }))` (line 96) returns the model
   and the audit trail in lockstep.

### Tie back to #1141 audit rail

The module doc-comment (`ferrotorch-graph/src/safetensors_loader.rs`)
ties the `DropReport` to issue #1141 ("every key must either land
in a parameter or appear in the report"). This is the
state-dict-drop audit rail that surfaces silent vocabulary
divergence between upstream PyG `GCNConv` and ferrotorch's `GcnNet`.

### Non-test production consumers

- `ferrotorch-graph/examples/gcn_inference_dump.rs:32`
  `use ferrotorch_graph::load_gcn_net;`
- `ferrotorch-graph/examples/gcn_inference_dump.rs`
  `let (net, report) = load_gcn_net(&weights_path, args.in_features,
  args.hidden, args.num_classes, /* strict = */ true)?;` — the
  example binary's primary consumer of this function.
- `ferrotorch-graph/examples/gcn_inference_dump.rs:246, 282`
  Reads `report.unmapped` for the harness's verdict line and the
  `"unmapped":<count>` JSON field.

## Parity contract

`parity_ops = []`. The loader's parity surface is the on-disk
safetensors format — that contract is owned by
`ferrotorch-serialize/src/safetensors.rs`. This file only validates
that the keys produced by `GcnNet::named_parameters` match the keys
PyG's pinned mirror exposes; any divergence shows up as a non-empty
`DropReport.unmapped` and (with `strict=true`) as an error.

Edge-case expectations:

- **Extra upstream keys**, `strict=true`: `InvalidArgument` with
  the sorted list of unmapped keys.
- **Extra upstream keys**, `strict=false`: keys land in
  `DropReport.unmapped`; matching subset is still loaded.
- **Missing parameter** (ferrotorch expects a key not in the
  safetensors): the inner `Module::load_state_dict(filtered,
  strict=true)` call propagates the error. This is always fatal
  regardless of the outer `strict` flag.
- **Shape mismatch on a matching key**: same path — surfaced by
  the inner `Module::load_state_dict`.
- **Malformed safetensors file**: surfaced by
  `load_safetensors::<f32>(weights_path)` and rewritten with the
  file path embedded in the error message.

## Verification

One inline test at `ferrotorch-graph/src/safetensors_loader.rs:104-132`:

- `round_trip_into_gcn_net` — builds a `GcnNet(4, 3, 2)`, snapshots
  every parameter's data, writes its `state_dict()` to a temporary
  safetensors via `ferrotorch_serialize::save_safetensors`, reloads
  via `load_gcn_net`, asserts `report.unmapped.is_empty()`, and
  walks the loaded parameters comparing each `Vec<f32>`
  element-by-element with `abs(a - b) < 1e-7`.

```bash
cargo test -p ferrotorch-graph --lib safetensors_loader:: 2>&1 | tail -3
```

Expected: `1 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct DropReport` at `DropReport in ferrotorch-graph/src/safetensors_loader.rs` deriving `Debug, Default, Clone` with `unmapped: Vec<String>`; non-test consumer: `ferrotorch-graph/examples/gcn_inference_dump.rs` reads `report.unmapped` for the harness diagnostic and line 282 surfaces `report.unmapped.len()` in the stdout JSON verdict. |
| REQ-2 | SHIPPED | impl: `pub fn load_gcn_net` at `load_gcn_net in ferrotorch-graph/src/safetensors_loader.rs`; non-test consumer: `ferrotorch-graph/examples/gcn_inference_dump.rs` `let (net, report) = load_gcn_net(&weights_path, args.in_features, args.hidden, args.num_classes, /* strict = */ true)?;` is the end-to-end harness call. |
| REQ-3 | SHIPPED | impl: the `if strict && !unmapped.is_empty()` branch at `ferrotorch-graph/src/safetensors_loader.rs` returns `InvalidArgument` carrying the unmapped key list; non-test consumer: `ferrotorch-graph/examples/gcn_inference_dump.rs` calls with `strict=true`, so the harness commits to the strict-error path on every invocation — any drift in the pinned mirror surfaces as a hard error during inference. |
| REQ-4 | SHIPPED | impl: the `.map_err` adapter at `safetensors in ferrotorch-graph/src/safetensors_loader.rs` rewrites the `safetensors` error into `InvalidArgument` with the file path embedded; non-test consumer: `ferrotorch-graph/examples/gcn_inference_dump.rs` consumes this error path on misconfigured `--model` flags; the path-in-message contract is what the harness's stderr diagnostic at line 290 prints. |
| REQ-5 | SHIPPED | impl: `net.load_state_dict(&filtered, /* strict = */ true)?` at `ferrotorch-graph/src/safetensors_loader.rs:95` hard-codes the inner strict mode; non-test consumer: the unit test `round_trip_into_gcn_net` at lines 104-132 verifies the round-trip end-to-end (test is the canonical consumer because `Module::load_state_dict` is the trait method this file calls into; downstream production code reaches it only via `load_gcn_net`'s return value). |
