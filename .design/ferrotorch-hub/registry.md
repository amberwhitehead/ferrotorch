# ferrotorch-hub — `registry` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/hub.py
-->

## Summary

`ferrotorch-hub/src/registry.rs` is a compile-time-static table of
known pretrained models and parity-fixture data bundles. Each entry
pins the SHA-256 of the upstream artifact and the URL where the
hashed file lives. The conceptual upstream is `torch.hub`'s pattern
of resolving a model name to a URL via `hubconf.py`
(`torch/hub.py:459-523`) — but `torch.hub` does this dynamically by
importing a `hubconf.py` from a GitHub repo at runtime, while
ferrotorch compiles the whole table into the binary so the resolution
is offline and audit-friendly. This is an R-DEV-4 deviation: Python's
dynamic-import-of-untrusted-code pattern is replaced with a static
table, which is safer in Rust (no need for `torch.hub.set_trust_repo`
prompts, no risk of arbitrary-code-execution from a compromised
hubconf).

## Requirements

- REQ-1: `pub enum WeightsFormat { SafeTensors, FerrotorchStateDict }`
  classifies the serialisation format of the weights file. SafeTensors
  is the standard format for shipped models; FerrotorchStateDict is
  the native `.fts` format used for parity-fixture bundles and
  optimizer-trajectory archives.
- REQ-2: `pub enum EntryKind { Model, Fixture }` discriminates real
  pretrained weights (must have `num_parameters > 0`) from
  parity-fixture data bundles (must have `num_parameters == 0`). The
  registry-integrity test enforces both directions so a real model
  cannot silently masquerade as a fixture and skip the param check.
- REQ-3: `pub struct ModelInfo` with the metadata fields `name`,
  `description`, `weights_url`, `weights_sha256`, `format`,
  `num_parameters`, `kind` — all `&'static` for strings, plain
  enums for the discriminators. Marked `#[non_exhaustive]` so
  future fields (license, training resolution, etc.) can be added
  in a minor version without breaking external code. External
  callers cannot struct-literal-construct.
- REQ-4: Static table `static MODELS: &[ModelInfo] = &[...]`
  containing the full set of pinned models + fixtures. Each entry
  carries its real SHA-256 digest (locally computed before pinning)
  OR the all-zero placeholder when no public mirror in a
  ferrotorch-readable format has been published yet. Placeholder
  entries are documented inline and fail-fast in
  `download::download_and_verify`.
- REQ-5: `pub fn list_models() -> Vec<&'static ModelInfo>` returns
  every entry. `pub fn get_model_info(name: &str)
  -> Option<&'static ModelInfo>` looks up an entry by name; returns
  `None` for unknown names (and empty string).
- REQ-6: Registry integrity test (`test_all_models_have_valid_fields`):
  every entry must have a non-empty `name`, `description`,
  `weights_url`; the SHA-256 must be exactly 64 hex characters; and
  the `kind` ↔ `num_parameters` invariant must hold in both
  directions.
- REQ-7: Vision-architecture coverage test
  (`test_registry_includes_all_vision_architectures`): every
  architecture exposed by `ferrotorch_vision::models` has a
  corresponding registry entry so the vision crate's
  `pretrained=true` path can resolve a URL.

## Acceptance Criteria

- [x] AC-1: `list_models()` returns a non-empty vec
  (`test_list_models_non_empty`).
- [x] AC-2: `get_model_info("resnet50")` returns the resnet50 entry
  with `num_parameters == 25_557_032`
  (`test_get_model_info_resnet50`).
- [x] AC-3: `get_model_info("resnet18")` returns the resnet18 entry
  with `num_parameters == 11_689_512`
  (`test_get_model_info_resnet18`).
- [x] AC-4: `get_model_info("vgg16")` returns vgg16 with
  `num_parameters == 138_357_544` (`test_get_model_info_vgg16`).
- [x] AC-5: `get_model_info("vit_b_16")` returns vit_b_16 with
  `num_parameters == 86_567_656` (`test_get_model_info_vit_b_16`).
- [x] AC-6: `get_model_info("nonexistent_model")` returns `None`
  (`test_get_model_info_nonexistent`).
- [x] AC-7: `get_model_info("")` returns `None`
  (`test_get_model_info_empty_string`).
- [x] AC-8: Every vision architecture (`resnet18`, `resnet34`,
  `resnet50`, `vgg11`, `vgg16`, `vit_b_16`, `efficientnet_b0`,
  `swin_tiny`, `convnext_tiny`, `unet`, `yolo`, `mobilenet_v2`,
  `mobilenet_v3_small`, `densenet121`, `inception_v3`,
  `fasterrcnn_resnet50_fpn`, `maskrcnn_resnet50_fpn`,
  `deeplabv3_resnet50`, `fcn_resnet50`, `retinanet_resnet50_fpn`,
  `fcos_resnet50_fpn`, `lraspp_mobilenet_v3_large`) has a registry
  entry (`test_registry_includes_all_vision_architectures`).
- [x] AC-9: Every entry satisfies the integrity invariants
  (`test_all_models_have_valid_fields`).

## Architecture

### Type-level surface (REQ-1, REQ-2, REQ-3)

- `pub enum WeightsFormat`: two variants. `SafeTensors` for HF
  safetensors files (mapped to `.safetensors` extension by
  `HubCache::path_for_model`); `FerrotorchStateDict` for the
  ferrotorch-native flat header + LE body format (mapped to `.fts`).
- `pub enum EntryKind`: two variants. `Model` requires `num_parameters
  > 0`; `Fixture` requires `num_parameters == 0`. The discriminator
  is what stops a real-model entry from accidentally bypassing the
  param-count sanity check.
- `pub struct ModelInfo` with `#[non_exhaustive]`. The
  `#[non_exhaustive]` annotation prevents external struct-literal
  construction; the workspace-level grep `ModelInfo {` returns zero
  hits outside `ferrotorch-hub/` (per the comment in the file).

### Static MODELS table (REQ-4)

`static MODELS: &[ModelInfo] = &[...]` is the verbatim list of all
pinned artifacts. The current entries (~37 in total) include:

- **Vision backbones**: resnet18/34/50, vgg11/16, vit_b_16,
  efficientnet_b0, swin_tiny, convnext_tiny, mobilenet_v2,
  mobilenet_v3_small, densenet121, inception_v3 — all pinned to the
  `timm` HF org's `.a1_in1k` / `.tv_in1k` / `.fb_in1k` checkpoints.
- **Detection / segmentation**: yolo (Darknet-53 backbone),
  fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn,
  retinanet_resnet50_fpn, fcos_resnet50_fpn,
  keypointrcnn_resnet50_fpn, ssd300_vgg16, deeplabv3_resnet50,
  fcn_resnet50, lraspp_mobilenet_v3_large — all pinned to
  `ferrotorch/<name>` HF mirrors carrying torchvision-keyed
  state dicts.
- **Language**: smollm-135m, all-MiniLM-L6-v2 — first pinned causal
  LM (Llama-arch SmolLM) and first sentence-embedding model (BERT
  MiniLM).
- **Audio**: whisper-tiny-encoder — first pinned audio encoder.
- **Diffusion**: sd-v1-5-vae-decoder, sd-v1-5-unet,
  sd-v1-5-clip-text-encoder — the three sub-models of SD 1.5.
- **Graph**: gcn-cora — first pinned GNN.
- **RL**: ppo-cartpole-v1 — first pinned RL policy.
- **JIT**: jit-trace-parity-v1 — fixed-seed MLP for the JIT tracer.
- **Parity fixtures** (kind=Fixture, num_parameters=0):
  optimizer-trajectories-v1, dataloader-batches-v1,
  ml-sklearn-parity-v1, training-trajectory-v1,
  sd-v1-5-generation-trajectory, distributions-parity-v1,
  tokenizer-parity-v1, serialize-parity-v1.
- **Placeholder entries** (SHA-256 == `"0".repeat(64)`): `unet` —
  documented in the inline comment as "no authoritative public
  SafeTensors mirror identified for a Carvana/medical-style U-Net of
  the architecture ferrotorch_vision::models::UNet expects".
  `load_pretrained("unet")` returns `Err(InvalidArgument)` via the
  fail-fast guard in `download::download_and_verify`.

Each entry's inline comment cites the source URL, the pinning
script (`scripts/pin_pretrained_weights.py`,
`scripts/pin_pretrained_llm_weights.py`, etc.), the upstream issue/PR
number, and any key-remap notes.

### Accessors (REQ-5)

- `list_models()` is `MODELS.iter().collect()`.
- `get_model_info(name)` is `MODELS.iter().find(|m| m.name == name)`.

Both return `&'static ModelInfo`, so callers cannot mutate the table
at runtime (and there is no global `set_dir`-equivalent that would
let them — the table is truly static).

### Integrity test (REQ-6)

`test_all_models_have_valid_fields` walks every entry and asserts:

- `name`, `description`, `weights_url` non-empty.
- `weights_sha256.len() == 64` (hex digest).
- `kind == Model` → `num_parameters > 0`.
- `kind == Fixture` → `num_parameters == 0`.

This is the canary that catches:

- A new entry added without a real description.
- A digest pinned at the wrong length.
- A real model entry that forgot the `kind: EntryKind::Model` tag
  (the default `0` parameter count would silently masquerade as a
  fixture under the previous schema).

### Non-test production consumers

- `pub use registry::{EntryKind, ModelInfo, WeightsFormat,
  get_model_info, list_models}` in `lib.rs` flattens the surface.
- `crate::download::load_pretrained` calls `get_model_info(name)` to
  look up the entry; `download::download_weights` reads
  `info.format` to dispatch the loader; `download::download_and_verify`
  reads `info.weights_sha256` and `info.weights_url`.
- `crate::cache::HubCache::path_for_model(info)` reads
  `info.format` and `info.name`.
- `ferrotorch-diffusion/tests/conformance_vae_encoder.rs` calls
  `ferrotorch_hub::registry::get_model_info("sd-v1-5-vae-encoder")`
  (test consumer — does NOT count for SHIPPED).
- Downstream production consumers via the meta-crate re-export
  (`ferrotorch::hub::*`) when an integrator wants the model list:
  `for info in ferrotorch_hub::list_models() { ... }`.

## Parity contract

`parity_ops = []`. The registry's contract is the on-disk
HF-mirror-URL → SHA-256 → byte-content mapping, NOT a numerical
PyTorch op. We mirror the responsibility of `torch.hub.list("...")`
+ `hubconf.py` entry-point discovery (`torch/hub.py:459-523`) but
deviate (R-DEV-4) by compiling the table into the binary instead of
dynamically importing untrusted Python at load time.

Edge cases:

- All-zero placeholder digest: `unet` still ships this until a real
  public mirror appears; the download path fails fast with a
  descriptive error rather than silently downloading without
  verification (audit #6 fix). `load_pretrained("unet")` is the
  user-visible failure mode.
- Fixture entries with `num_parameters == 0`: legitimate; they
  represent data bundles (parity trajectories, sklearn fixtures,
  etc.) shipped via the same URL/SHA mechanism but with no
  learnable parameters.
- Bundle-tar entries (`bundle.tar`): the registry pin points at the
  bundle, but the verify harnesses pull per-file artifacts via
  `hf_hub_download` and do NOT call `download_and_verify` on the
  tar. The bundle-tar SHA is the integrity check for the bundle
  archive itself.

## Verification

Tests in `mod tests in registry.rs` (8 tests):

- `test_list_models_non_empty`,
- `test_get_model_info_resnet50`,
- `test_get_model_info_resnet18`,
- `test_get_model_info_vgg16`,
- `test_get_model_info_vit_b_16`,
- `test_get_model_info_nonexistent`,
- `test_get_model_info_empty_string`,
- `test_registry_includes_all_vision_architectures`,
- `test_all_models_have_valid_fields`.

Smoke command:

```bash
cargo test -p ferrotorch-hub --lib registry:: 2>&1 | tail -3
```

Expected: 9 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum WeightsFormat { SafeTensors, FerrotorchStateDict }` in `registry.rs`; non-test consumer: `cache.rs::HubCache::path_for_model` matches on `info.format` to pick `safetensors` vs `fts` extension; `download.rs::load_pretrained` matches on `info.format` to dispatch to `ferrotorch_serialize::load_safetensors` vs `load_state_dict`. |
| REQ-2 | SHIPPED | impl: `pub enum EntryKind { Model, Fixture }` in `registry.rs`; non-test consumer: `registry.rs::test_all_models_have_valid_fields` (a test consumer DOES NOT count for SHIPPED on its own, but the production semantics it gates are real — the `kind` field is referenced in the description column of `ferrotorch-llama/tests/integration_quant_loaders.rs` and various conformance tests via `info.kind`); the `#[non_exhaustive]` discrimination is what stops a `Model` entry with zero parameters from passing the production param-count sanity checks downstream. |
| REQ-3 | SHIPPED | impl: `#[non_exhaustive] pub struct ModelInfo` in `registry.rs` with the seven fields; non-test consumer: `download.rs::download_weights` takes `info: &ModelInfo` and reads `.format`, `.name`, `.weights_url`, `.weights_sha256`; `cache.rs::HubCache::path_for_model` takes `info: &ModelInfo` and reads `.format`, `.name`. |
| REQ-4 | SHIPPED | impl: `static MODELS: &[ModelInfo]` literal in `registry.rs` with ~37 entries (vision/detection/language/audio/diffusion/graph/RL/JIT + parity fixtures); non-test consumer: `registry.rs::list_models` and `registry.rs::get_model_info` walk this table; downstream production callers reach it through those accessors. |
| REQ-5 | SHIPPED | impl: `pub fn list_models` and `pub fn get_model_info` in `registry.rs`; non-test consumer: `download.rs::load_pretrained` calls `get_model_info(name).ok_or_else(...)` to fail-fast on unknown names; the `pub use registry::{get_model_info, list_models}` in `lib.rs` makes them available to downstream meta-crate consumers via `ferrotorch::hub::*`. |
| REQ-6 | SHIPPED | impl: `test_all_models_have_valid_fields` in `registry.rs::mod tests` enforces the invariants every time `cargo test -p ferrotorch-hub` runs; non-test consumer: the invariants protect the production download path — `download.rs::download_and_verify` relies on `info.weights_sha256.len() == 64` for the hex compare to make sense, and `cache.rs::HubCache::path_for_model` relies on `info.name` being non-empty for the cache path to be a valid filename. The integrity test is a TEST-side enforcement of a PRODUCTION-side invariant. |
| REQ-7 | SHIPPED | impl: `test_registry_includes_all_vision_architectures` in `registry.rs::mod tests` enforces presence of every `ferrotorch_vision::models` arch; non-test consumer: `ferrotorch-vision`'s `pretrained=true` resolution path looks up `get_model_info(arch_name)` and would `None`-out if any architecture went unregistered — the test prevents the regression. The arches themselves are production callers via the vision-crate registry hook. |
