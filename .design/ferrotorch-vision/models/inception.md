# ferrotorch-vision — `models::inception` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: /home/doll/.local/lib/python3.13/site-packages/torchvision/models/inception.py
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/inception.py
-->

## Summary

`ferrotorch-vision/src/models/inception.rs` ships Inception V3 (Szegedy
et al. 2016) with full torchvision parity (Phase 10, #993, #1012). The
layout matches `torchvision.models.inception_v3(aux_logits=False)`
exactly — same field names, same per-branch `BasicConv2d` shapes, same
BN eps (`1e-3`, NOT the 1e-5 default), and same `bias=false` on every
conv (because BN follows). The `aux_logits=False` variant is shipped;
the AuxLogits auxiliary head is intentionally absent.

## Requirements

- REQ-1: `pub struct BasicConv2d<T: Float>` is the per-conv primitive:
  `Conv2d(bias=false) + BatchNorm2d(eps=1e-3, momentum=0.1, affine=true)
  + ReLU`. Children render as `conv` and `bn`. The BN running buffers
  are exposed via the standard `BatchNorm2d` `as_any` downcast.
- REQ-2: `pub struct InceptionA<T: Float>` (`Mixed_5b/5c/5d`) carries
  the 4 branches: `branch1x1`, `branch5x5_{1,2}`, `branch3x3dbl_{1..3}`,
  `branch_pool` plus a 3×3 stride-1 padding-1 `avg_pool`. Forward
  `cat([b1, b5, b3dbl, branch_pool(avg_pool(x))], 1)`.
- REQ-3: `pub struct InceptionB<T: Float>` (`Mixed_6a` reduction):
  carries `branch3x3`, `branch3x3dbl_{1..3}`, and a 3×3 stride-2
  `max_pool` with NO `branch_pool` conv. Forward
  `cat([b3, b3dbl, max_pool(x)], 1)`.
- REQ-4: `pub struct InceptionC<T: Float>` (`Mixed_6b/6c/6d/6e`):
  carries `branch1x1`, `branch7x7_{1..3}`, `branch7x7dbl_{1..5}`,
  `branch_pool` plus a 3×3 stride-1 padding-1 `avg_pool`.
- REQ-5: `pub struct InceptionD<T: Float>` (`Mixed_7a` reduction):
  carries `branch3x3_{1,2}`, `branch7x7x3_{1..4}`, and a 3×3 stride-2
  `max_pool` (no branch_pool conv).
- REQ-6: `pub struct InceptionE<T: Float>` (`Mixed_7b/7c`):
  carries `branch1x1`, `branch3x3_1`, `branch3x3_{2a,2b}` (run on the
  SAME upstream tensor and concat their outputs — parallel-branch
  fan-out, failure mode #34), `branch3x3dbl_{1,2}`,
  `branch3x3dbl_{3a,3b}` (same parallel-branch shape), `branch_pool`
  plus a 3×3 stride-1 padding-1 `avg_pool`.
- REQ-7: `pub struct InceptionV3<T: Float>` carries the full
  inception_v3 stack: 5 stem `BasicConv2d`s + `maxpool1` + 1
  `BasicConv2d` + 1 `BasicConv2d` + `maxpool2` + 11 Mixed modules +
  `avgpool` + `dropout` (eval-identity) + `fc: Linear(2048,
  num_classes)`.
- REQ-8: `Module::forward` for `InceptionV3<T>` runs the full pipeline
  matching torchvision's `Inception3.forward` (without `transform_input`
  and without `aux_logits`, which are out-of-scope here).
- REQ-9: `named_parameters` returns the torchvision-exact paths
  (`Conv2d_1a_3x3.conv.weight`, ..., `Mixed_5b.branch1x1.conv.weight`,
  ..., `fc.weight`).
- REQ-10: `pub fn inception_v3` is the canonical constructor.

## Acceptance Criteria

- [x] AC-1: `BasicConv2d::<f32>::new(...)` constructs and forwards.
- [x] AC-2: `InceptionA::<f32>::new(192, 32)` (`Mixed_5b`) constructs
  and forwards; output channels = `64 + 64 + 96 + pool_features = 256`.
- [x] AC-3: `InceptionB::<f32>::new(288)` (`Mixed_6a`) constructs and
  forwards; output channels = `in + 384 + 96 = 768`.
- [x] AC-4: `InceptionV3::<f32>::new(1000)` constructs and forwards on
  the canonical 299×299 input returning `[1, 1000]`.
- [x] AC-5: `named_parameters` includes torchvision-exact prefixes
  `Conv2d_1a_3x3.conv.`, `Mixed_5b.branch1x1.conv.`, `Mixed_6a.`,
  `Mixed_7c.`, `fc.`.

## Architecture

`pub struct BasicConv2d<T: Float>` carries `conv` and `bn` plus a
`training` flag. The conv carries `bias=false`; BN uses
`eps=INCEPTION_BN_EPS=1e-3` (NOT 1e-5 — failure mode #32) and
`momentum=INCEPTION_BN_MOM=0.1`. Forward: `conv → bn → relu`. Children
render as `conv` / `bn`.

Each Mixed module decomposes into named branches, runs them in parallel
on the SAME input, and concatenates along axis=1. The `extend_named`
helper at the top of the file walks the branch's `BasicConv2d`s and
prefixes their already-`conv.X` / `bn.X` keys with the branch name to
produce torchvision-exact `named_parameters` output.

`pub struct InceptionV3<T: Float>` ties the full stack together. The
field names match `torchvision.models.inception.Inception3` exactly:

```text
Conv2d_1a_3x3, Conv2d_2a_3x3, Conv2d_2b_3x3,
maxpool1,
Conv2d_3b_1x1, Conv2d_4a_3x3,
maxpool2,
Mixed_5b, Mixed_5c, Mixed_5d,
Mixed_6a,
Mixed_6b, Mixed_6c, Mixed_6d, Mixed_6e,
Mixed_7a,
Mixed_7b, Mixed_7c,
avgpool, dropout, fc
```

Dropout in eval is identity (`Dropout::forward(...)` returns input
unchanged in eval mode), matching torchvision's `nn.Dropout(p=0.5)`
behavior on `model.eval()`. `aux_logits=False` is the only variant
shipped; the AuxLogits submodule is intentionally absent (the design
contract is the eval-mode `inception_v3(aux_logits=False)` path).

### Non-test production consumers

- `pub use inception::{BasicConv2d as InceptionBasicConv2d, InceptionA,
  InceptionB, InceptionC, InceptionD, InceptionE, InceptionV3,
  inception_v3}` re-export at `ferrotorch-vision/src/models/mod.rs`.
- `default_registry()` registers `"inception_v3"` via
  `maybe_load_pretrained` at `registry.rs:257`.

## Parity contract

`parity_ops = []`. Inception V3 composes `Conv2d`, `BatchNorm2d`
(`eps=1e-3`), `Linear`, `MaxPool2d`, `AvgPool2d`,
`AdaptiveAvgPool2d`, `Dropout`, the differentiable `relu`, and `cat`.
No new op surface.

Edge cases preserved versus torchvision:

- **BN `eps=1e-3`** (NOT 1e-5): would change BN output by ~1 ulp per
  element on value-parity logits otherwise. Failure mode #32 in the
  Phase 10 audit.
- **Conv `bias=false`** everywhere (BN follows): failure mode #33.
- **`branch_pool` uses `F.avg_pool2d(..., padding=1)`**:
  `count_include_pad=True` is the default and matches our `AvgPool2d`
  semantics (hardcoded). Divisor matches.
- **InceptionE's `branch3x3_2a/2b` and `branch3x3dbl_3a/3b` run on the
  SAME upstream tensor**: the parallel-branch fan-out shape that
  failure mode #34 flags as easy to break. The forward impl runs the
  upstream branch once, then forks two children from it before
  concatenating.
- **`aux_logits=False`**: no AuxLogits submodule. The design contract
  ships only the eval-mode `inception_v3(aux_logits=False)` path
  (matches the torchvision pretrained checkpoint pinned for this
  registry entry).

## Verification

Tests in `mod tests` in `inception.rs` cover per-block construction +
forward shapes, the full `inception_v3` forward shape on the 299×299
canonical input, parameter-count band check matching torchvision,
named-parameter layout, train/eval propagation, and Send+Sync.

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib inception:: 2>&1 | tail -3
```

Expected: all tests pass; no parity-sweep ops.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct BasicConv2d<T: Float>` + `Module<T>` impl in `inception.rs` mirrors torchvision `BasicConv2d` at `inception.py`; non-test consumer: every Mixed module's `new` constructs `BasicConv2d`s in `inception.rs`. |
| REQ-2 | SHIPPED | impl: `pub struct InceptionA<T: Float>` + `Module<T>` impl in `inception.rs`; non-test consumer: `InceptionV3::new` constructs `Mixed_5b/5c/5d` from it in `inception.rs`. |
| REQ-3 | SHIPPED | impl: `pub struct InceptionB<T: Float>` + `Module<T>` impl in `inception.rs`; non-test consumer: `InceptionV3::new` constructs `Mixed_6a` from it in `inception.rs`. |
| REQ-4 | SHIPPED | impl: `pub struct InceptionC<T: Float>` + `Module<T>` impl in `inception.rs`; non-test consumer: `InceptionV3::new` constructs `Mixed_6b/6c/6d/6e` from it in `inception.rs`. |
| REQ-5 | SHIPPED | impl: `pub struct InceptionD<T: Float>` + `Module<T>` impl in `inception.rs`; non-test consumer: `InceptionV3::new` constructs `Mixed_7a` from it in `inception.rs`. |
| REQ-6 | SHIPPED | impl: `pub struct InceptionE<T: Float>` + `Module<T>` impl in `inception.rs`; non-test consumer: `InceptionV3::new` constructs `Mixed_7b/7c` from it in `inception.rs`. |
| REQ-7 | SHIPPED | impl: `pub struct InceptionV3<T: Float>` + `InceptionV3::new` in `inception.rs`; non-test consumer: `default_registry()` constructs it via `maybe_load_pretrained` at `registry.rs:257`. |
| REQ-8 | SHIPPED | impl: `Module::forward` for `InceptionV3<T>` in `inception.rs`; non-test consumer: trait method invoked through `Box<dyn Module<T>>` returned from `registry.rs::get_model`. |
| REQ-9 | SHIPPED | impl: `named_parameters` for every Mixed type + `InceptionV3` in `inception.rs`; non-test consumer: `load_state_dict(&state_dict, false)` at `registry.rs:53` walks the result. |
| REQ-10 | SHIPPED | impl: `pub fn inception_v3` in `inception.rs`; non-test consumer: `default_registry()` invokes it at `registry.rs:260`. |
