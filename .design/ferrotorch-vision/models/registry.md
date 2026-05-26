# ferrotorch-vision — `models::registry` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: /home/doll/.local/lib/python3.13/site-packages/torchvision/models/_api.py
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/_api.py
-->

## Summary

`ferrotorch-vision/src/models/registry.rs` ships the model-name →
constructor registry that mirrors torchvision's
`torchvision.models.get_model(name, weights=...)` user-facing API.
Every canonical vision architecture (ResNet-18/34/50, VGG-11/16,
ViT-B/16, EfficientNet-B0, Swin-Tiny, ConvNeXt-Tiny, U-Net, YOLO,
MobileNetV2 / V3-Small, DenseNet-121, Inception-V3, Faster R-CNN, Mask
R-CNN, SSD300-VGG16, RetinaNet, FCOS, Keypoint R-CNN, DeepLabV3-ResNet50,
FCN-ResNet50, LRASPP-MobileNetV3-Large) is bound here, with `pretrained
= true` routing through `ferrotorch_hub` to fetch + verify safetensors
weights and then load them with `strict = false` plus an explicit
BN-buffer apply step.

## Requirements

- REQ-1: `pub type ModelConstructor<T>` is the boxed
  `Fn(bool, usize) -> FerrotorchResult<Box<dyn Module<T>>> + Send +
  Sync` signature stored per architecture name.
- REQ-2: `pub struct ModelRegistry<T: Float>` carries a `HashMap<String,
  ModelConstructor<T>>` and exposes `list_models()`, `get_model(name,
  pretrained, num_classes)`, and `register_model(name, constructor)`.
  `list_models` returns sorted names. `get_model` returns
  `FerrotorchError::InvalidArgument` on unknown names with a message
  listing available models.
- REQ-3: A private `default_registry()` populates the global registry
  with every canonical architecture from `super::{resnet, vgg, vit,
  efficientnet, swin, convnext, mobilenet, densenet, inception,
  detection, segmentation, unet, yolo}`.
- REQ-4: A private `maybe_load_pretrained<T, F, M>` helper builds the
  architecture, then if `pretrained == true`:
  - Looks up `hub_name` in `ferrotorch_hub::registry`; surfaces a clear
    `InvalidArgument` error if absent.
  - Downloads weights via `ferrotorch_hub::download::download_weights`
    (SHA-verified, cached).
  - Loads the safetensors / state-dict file via `ferrotorch_serialize`.
  - Calls `model.load_state_dict(&state_dict, /*strict=*/ false)`.
  - Applies BN running statistics via
    `super::bn_buffer_loader::apply_bn_buffers_from_state_dict` (BN
    running stats live in `Mutex<Vec<f64>>` outside the `Buffer<T>`
    abstraction and `load_state_dict` would silently drop them).
- REQ-5: `pub static REGISTRY: LazyLock<RwLock<ModelRegistry<f32>>>` is
  the global f32 registry. Public free functions `list_models`,
  `get_model`, `register_model` lock the global. Lock poison surfaces
  as `FerrotorchError::Internal`.
- REQ-6: Every canonical model name in the public registry must have a
  matching entry in `ferrotorch_hub::registry`. The conformance test
  `test_pretrained_lookup_known_model_in_hub` is the binding contract
  for this.

## Acceptance Criteria

- [x] AC-1: `list_models()` includes every canonical name: `resnet18`,
  `resnet34`, `resnet50`, `vgg11`, `vgg16`, `vit_b_16`,
  `efficientnet_b0`, `swin_tiny`, `convnext_tiny`, `unet`, `yolo`,
  `mobilenet_v2`, `mobilenet_v3_small`, `densenet121`, `inception_v3`,
  `fasterrcnn_resnet50_fpn`, `maskrcnn_resnet50_fpn`, `ssd300_vgg16`,
  `retinanet_resnet50_fpn`, `fcos_resnet50_fpn`,
  `keypointrcnn_resnet50_fpn`, `deeplabv3_resnet50`, `fcn_resnet50`,
  `lraspp_mobilenet_v3_large` (test
  `test_list_models_contains_resnets`).
- [x] AC-2: `list_models()` returns the names in sorted order
  (`test_list_models_is_sorted`).
- [x] AC-3: `get_model("nonexistent_model", false, 1000)` returns an
  error whose message contains `"unknown model"` and the bad name
  (`test_get_model_unknown_name_errors`).
- [x] AC-4: `get_model("resnet18", false, 1000)` succeeds with a
  non-empty parameter list (`test_get_model_resnet18_constructs_successfully`).
- [x] AC-5: Every canonical name in the hard-coded canonical list has
  a matching `get_model_info` entry in `ferrotorch_hub::registry`
  (`test_pretrained_lookup_known_model_in_hub`).
- [x] AC-6: `get_model(name, false, 10)` succeeds for every registered
  model name (no network access required when `pretrained=false`)
  (`test_pretrained_false_constructs_without_network`).
- [x] AC-7: `register_model("dummy_test_model", ...)` then `get_model`
  on the same name returns the registered constructor's output
  (`test_register_and_get_model_roundtrip`).

## Architecture

`pub type ModelConstructor<T>` is the per-architecture factory
signature. Each closure receives `(pretrained, num_classes)` and
returns a boxed dynamic Module — the boxed form lets the registry
return architectures of wildly different concrete types (`ResNet`,
`VGG`, `VisionTransformer`, etc.) via a single trait-object handle.

`pub struct ModelRegistry<T: Float>` stores a `HashMap<String,
ModelConstructor<T>>` and is wrapped in `RwLock` for thread safety.
`get_model` looks up the name; on miss returns
`FerrotorchError::InvalidArgument { message: "unknown model:
\"foo\". Available: [...]" }`.

`maybe_load_pretrained` is the polymorphic loader helper. Every
registry entry uses it via `Box::new(|pretrained, num_classes| { ...
maybe_load_pretrained(pretrained, "<hub_name>", || super::<crate>::<fn>::<f32>(num_classes)) })`.
The closure pattern lets each registry entry name the hub key
independently of the Rust function name. The `strict = false` mode is
deliberate — torchvision pretrained checkpoints sometimes include a
classifier head for a different `num_classes` than the user requests,
and those keys are dropped silently rather than failing the load.

The follow-up `apply_bn_buffers_from_state_dict` call is the load-bearing
part most lossy state-dict loaders miss. `load_state_dict` walks
`named_parameters` and `named_buffers`, but ferrotorch's
`BatchNorm2d::running_{mean,var}` are stored in
`Mutex<Vec<f64>>` outside the `Buffer<T>` abstraction; without an
explicit BN-buffer pass the running stats stay at their init values
(`mean=0, var=1`) and the eval-mode forward produces wrong logits.

`pub static REGISTRY: LazyLock<RwLock<ModelRegistry<f32>>>` initializes
the global registry lazily. The public free functions `list_models`,
`get_model`, `register_model` lock the `RwLock` (read for the first two,
write for `register_model`). Lock poisoning (a panic in another thread
holding the write lock) surfaces as `FerrotorchError::Internal`.

### Non-test production consumers

- `pub use registry::{ModelConstructor, ModelRegistry, REGISTRY,
  get_model, list_models, register_model}` re-export at
  `ferrotorch-vision/src/models/mod.rs`.
- Second-tier `pub use ... get_model, list_models, register_model ...`
  in `ferrotorch-vision/src/lib.rs:109` exposes these to downstream
  crates.
- `ferrotorch-vision/examples/inference_dump.rs:40` calls
  `ferrotorch_vision::models::get_model(...)` as the production CLI
  entry point.
- `scripts/pin_pretrained_weights.py:1249` generates Rust code that
  imports `use ferrotorch_vision::models::registry::get_model;` so the
  CI weight-pinning automation depends on the registry being stable.

## Parity contract

`parity_ops = []`. The registry is pure orchestration over the model
constructors covered in the other 9 design docs and the
`ferrotorch_hub` / `ferrotorch_serialize` weight-loading pipeline.

Edge cases preserved versus torchvision:

- **`torchvision.models.get_model(name, ...)`**: torchvision returns
  the model via `torchvision.models.get_model("resnet18", weights=...)`.
  Our `get_model("resnet18", pretrained=true, num_classes=1000)` is
  the analog. The `pretrained: bool` collapses torchvision's
  `Optional[WeightsEnum]` to a single yes/no — for the canonical
  checkpoint pin. Custom weight choice flows through the hub registry
  by per-model name.
- **`strict = false` state-dict load**: torchvision's default. The
  `apply_bn_buffers_from_state_dict` follow-up call is the ferrotorch
  recovery for the running-stats gap.
- **Lock poisoning** is a `FerrotorchError::Internal`, not a panic:
  matches the project-wide policy that production code never panics
  outside `#[cfg(test)]`.

## Verification

Tests in `mod tests` in `registry.rs`:

- `test_list_models_contains_resnets` — exhaustive list check.
- `test_list_models_is_sorted`
- `test_get_model_unknown_name_errors`
- `test_get_model_resnet18_constructs_successfully`
- `test_pretrained_lookup_known_model_in_hub` — every canonical name
  has a hub entry.
- `test_pretrained_false_constructs_without_network` — pretrained=false
  works for every registered model.
- `test_register_and_get_model_roundtrip`

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib registry:: 2>&1 | tail -3
```

Expected: all tests pass; no parity-sweep ops.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub type ModelConstructor<T>` in `registry.rs`; non-test consumer: every `register_model` call in `default_registry()` passes a `Box::new(...)` whose type matches `ModelConstructor<f32>` in `registry.rs`. |
| REQ-2 | SHIPPED | impl: `pub struct ModelRegistry<T: Float>` + `list_models` / `get_model` / `register_model` in `registry.rs`; non-test consumer: public free functions `pub fn list_models` / `pub fn get_model` / `pub fn register_model` at the bottom of `registry.rs` lock `REGISTRY` and call them. |
| REQ-3 | SHIPPED | impl: `fn default_registry()` in `registry.rs`; non-test consumer: `pub static REGISTRY` in `registry.rs` initializes lazily via `LazyLock::new(|| RwLock::new(default_registry()))`. |
| REQ-4 | SHIPPED | impl: `fn maybe_load_pretrained<T, F, M>` in `registry.rs` (calls `ferrotorch_hub::registry::get_model_info`, `download::download_weights`, `ferrotorch_serialize::load_safetensors` / `load_state_dict`, then `model.load_state_dict(&state, false)` and `bn_buffer_loader::apply_bn_buffers_from_state_dict(...)`); non-test consumer: every registry-entry closure in `default_registry()` invokes it in `registry.rs`. |
| REQ-5 | SHIPPED | impl: `pub static REGISTRY: LazyLock<RwLock<ModelRegistry<f32>>>` + `pub fn list_models` / `pub fn get_model` / `pub fn register_model` (each with poison-handling) in `registry.rs`; non-test consumer: `examples/inference_dump.rs:40` calls `ferrotorch_vision::models::get_model(...)`. |
| REQ-6 | SHIPPED | impl: hard-coded canonical-name list in `test_pretrained_lookup_known_model_in_hub` plus every entry in `default_registry()` passing the same name string to both `register_model` and the hub-name lookup parameter; non-test consumer: `scripts/pin_pretrained_weights.py:1249` generates Rust code that walks `get_model(name, pretrained=true, ...)` for each canonical name and exits non-zero on hub lookup failure — the CI weight-pinning workflow is the production consumer that depends on this contract. |
