# ferrotorch-nn — `pooling` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/pooling.py
  - aten/src/ATen/native/Pool.cpp
-->

## Summary

`ferrotorch-nn/src/pooling.rs` implements the pooling-layer family:
`MaxPool1d/2d/3d`, `AvgPool1d/2d/3d`, `AdaptiveAvgPool1d/2d/3d`,
`AdaptiveMaxPool1d/2d/3d`, `FractionalMaxPool2d`, `LPPool1d/2d`, and
`MaxUnpool2d`. Mirrors `torch.nn.{MaxPool*, AvgPool*, AdaptiveAvg*,
AdaptiveMax*, FractionalMaxPool2d, LPPool*, MaxUnpool*}` at
`torch/nn/modules/pooling.py:79-1860`.

Every layer is a zero-parameter module operating on
`[B, C, *spatial]` tensors. Each forward attaches a `GradFn<T>` for
reverse-mode autodiff when gradient tracking is enabled.

## Requirements

- REQ-1: `pub struct MaxPool2d` + `Module<T>` impl — slides a kernel
  window over each `[H, W]` spatial plane taking the max per window.
  Carries `kernel_size: [usize; 2]`, `stride: [usize; 2]` (defaults
  to `kernel_size` when constructed as `[0, 0]`), `padding: [usize;
  2]`. Mirrors `torch.nn.MaxPool2d` at `pooling.py:155-234`.

- REQ-2: `pub struct AvgPool2d` + `Module<T>` impl — same shape
  contract; averages instead of maxing. Mirrors
  `torch.nn.AvgPool2d` at `pooling.py:680-789`.

- REQ-3: `pub struct AdaptiveAvgPool2d` + `Module<T>` impl — chooses
  kernel/stride per output spatial size to produce a fixed
  `output_size: (usize, usize)` regardless of input dimensions.
  Mirrors `torch.nn.AdaptiveAvgPool2d` at `pooling.py:1473-1512`.

- REQ-4: `pub struct MaxPool1d` / `pub struct MaxPool3d` /
  `pub struct AvgPool1d` / `pub struct AvgPool3d` — 1-D and 3-D
  variants with `[usize; N]`-typed kernel / stride / padding.
  Mirror `torch.nn.{MaxPool1d, MaxPool3d, AvgPool1d, AvgPool3d}` at
  `pooling.py:79-154`, `235-314`, `595-679`, `790-895`.

- REQ-5: `pub struct AdaptiveAvgPool1d` / `AdaptiveAvgPool3d` /
  `AdaptiveMaxPool1d` / `AdaptiveMaxPool2d` / `AdaptiveMaxPool3d` —
  1/2/3-D adaptive variants. Mirror upstream's
  `AdaptiveAvg*` and `AdaptiveMax*` at
  `pooling.py:1313-1568`.

- REQ-6: `pub struct FractionalMaxPool2d` + `Module<T>` impl —
  fractional-stride max pool (Graham 2014). Picks output positions
  via a deterministic random sequence given a target output ratio.
  Mirrors `torch.nn.FractionalMaxPool2d`.

- REQ-7: `pub struct LPPool1d` / `pub struct LPPool2d` + `Module<T>`
  impls — `LP-norm` pooling: `(sum_window(|x|^p))^(1/p)`. Mirrors
  `torch.nn.LPPool{1d,2d}`.

- REQ-8: `pub struct MaxUnpool2d` + `Module<T>` impl plus
  `pub fn max_unpool2d` — scatter the max values back to their
  original positions given the indices captured during pooling.
  Mirrors `torch.nn.MaxUnpool2d` at `pooling.py:316-426`.

- REQ-9: Functional entry points — `pub fn max_pool1d`,
  `pub fn max_pool2d`, `pub fn max_pool3d`, `pub fn avg_pool1d`,
  `pub fn avg_pool2d`, `pub fn avg_pool3d`,
  `pub fn adaptive_avg_pool1d`, `pub fn adaptive_avg_pool2d`,
  `pub fn adaptive_avg_pool3d`, `pub fn adaptive_max_pool1d`,
  `pub fn adaptive_max_pool2d`, `pub fn adaptive_max_pool3d`,
  `pub fn lp_pool1d`, `pub fn lp_pool2d`, `pub fn max_unpool2d`.
  Mirror `torch.nn.functional.{max_pool*, avg_pool*,
  adaptive_avg_pool*, adaptive_max_pool*, lp_pool*}`.

- REQ-10: Shape validation — every forward calls a private
  `validate_*d` helper that returns an error on rank mismatch,
  zero kernel or stride, and `padded_input < kernel_size`. The
  helpers live at `fn validate_4d in pooling.rs`, `fn
  validate_pool_params in pooling.rs`, etc.

- REQ-11: Autograd via `GradFn<T>` — every pool forward attaches a
  backward node when grad-tracking is enabled. Max pools save
  indices for the scatter on backward; avg pools just divide
  `grad_output` by the window size.

- REQ-12: Parity ops for the canonical entries
  (`max_pool1d/2d/3d`, `avg_pool1d/2d/3d`,
  `adaptive_avg_pool1d/2d`, `adaptive_max_pool1d/2d`) — output
  match upstream to within float tolerance. NOT-STARTED — blocker
  #1458.

## Acceptance Criteria

- [x] AC-1: `MaxPool2d::new([2, 2], [2, 2], [0, 0])` constructs.
- [x] AC-2: `MaxPool2d` forward on `[1, 1, 4, 4]` input produces
  `[1, 1, 2, 2]`.
- [x] AC-3: `AvgPool2d` forward produces the arithmetic mean per
  window.
- [x] AC-4: `AdaptiveAvgPool2d::new((1, 1))` returns global
  average per channel.
- [x] AC-5: `MaxUnpool2d` round-trips with `MaxPool2d` (modulo
  positions zeroed by the max selection).
- [x] AC-6: Backward through `MaxPool2d` propagates only to the
  argmax positions.
- [x] AC-7: Backward through `AvgPool2d` distributes
  `grad_output / window_size` to every input position in the
  window.
- [ ] AC-8: parity-sweep `nn.functional.max_pool2d` at status
  `verified` — blocker #1458.
- [ ] AC-9: parity-sweep `nn.functional.avg_pool2d` at status
  `verified` — blocker #1458.
- [ ] AC-10: parity-sweep `nn.functional.adaptive_avg_pool2d` at
  status `verified` — blocker #1458.

## Architecture

### Standard pools (REQ-1, REQ-2, REQ-4)

`pub struct MaxPool2d` / `pub struct AvgPool2d` / `pub struct
MaxPool1d` / `pub struct MaxPool3d` / `pub struct AvgPool1d` /
`pub struct AvgPool3d` all carry kernel/stride/padding arrays.
`MaxPool2d::new` defaults `stride` to `kernel_size` when callers
pass `[0, 0]` (the PyTorch convention).

Each `impl<T: Float> Module<T> for ...Pool<N>d` dispatches `forward`
to a free `fn ..._pool<N>d_forward` that:

1. Validates the input rank and the kernel parameters.
2. Computes the output size via `pool_output_size(input,
   kernel_size, stride, padding)`.
3. For each output position, iterates over the input window and
   selects the max (or sum) value.
4. Attaches the appropriate `GradFn<T>` if grad is enabled.

### Adaptive pools (REQ-3, REQ-5)

`pub struct AdaptiveAvgPool2d` / sibling variants carry only the
output spatial dimensions. The forward computes per-output
`start = (i * H_in) / H_out` and `end = ((i + 1) * H_in) / H_out`
to determine the input window for each output position, then
averages (or takes the max).

### Functional entries (REQ-9)

The free functions at `pub fn max_pool1d`, `pub fn max_pool2d`,
... in `pooling.rs` construct the corresponding module on the fly
and dispatch its forward. Useful for one-shot calls without
allocating the module ahead of time.

### MaxUnpool (REQ-8)

`pub struct MaxUnpool2d` + `pub fn max_unpool2d` accept the saved
indices from a prior `MaxPool2d` invocation and scatter the max
values back into a zero-initialised tensor of the original spatial
shape. Required for VAE / autoencoder symmetric architectures.

### LP and fractional pools (REQ-6, REQ-7)

`pub struct LPPool1d` / `pub struct LPPool2d` apply
`(sum_window(|x|^p))^(1/p)` per window. `pub struct
FractionalMaxPool2d` uses a deterministic random sequence (seeded
by `torch.Generator`-style state in upstream) to pick output
positions.

### Autograd (REQ-11)

Each forward path attaches its `GradFn<T>` via
`Tensor::from_operation`. Max-pool backwards save the argmax
indices; avg-pool backwards just record the window divisor; LP-pool
backwards record the per-window `(sum)^((1-p)/p)` factor for the
chain rule.

### Non-test production consumers

- `pub use pooling::{AdaptiveAvgPool1d, AdaptiveAvgPool2d,
  AdaptiveAvgPool3d, AdaptiveMaxPool1d, AdaptiveMaxPool2d,
  AdaptiveMaxPool3d, AvgPool1d, AvgPool2d, AvgPool3d,
  FractionalMaxPool2d, LPPool1d, LPPool2d, MaxPool1d, MaxPool2d,
  MaxPool3d, MaxUnpool2d, adaptive_avg_pool1d, adaptive_avg_pool2d,
  adaptive_avg_pool3d, adaptive_max_pool1d, adaptive_max_pool2d,
  adaptive_max_pool3d, avg_pool1d, avg_pool2d, avg_pool3d,
  lp_pool1d, lp_pool2d, max_pool1d, max_pool2d, max_pool3d,
  max_unpool2d}` at `ferrotorch-nn/src/lib.rs:237-243`.
- `ferrotorch-vision/src/models/resnet.rs:23`, `densenet.rs:43`,
  `inception.rs:61`, `vgg.rs:24`, `yolo.rs:36`,
  `unet.rs:34`, `convnext.rs:35`, `efficientnet.rs:38`,
  `mobilenet.rs:55`, `detection/fpn.rs:35`,
  `segmentation/aspp.rs:38`, `segmentation/lraspp.rs:49` — every
  major torchvision-mirror model consumes `MaxPool2d` and / or
  `AdaptiveAvgPool2d` and / or `AvgPool2d`.
- `ferrotorch-nn/src/se.rs` (SqueezeExcitation) uses
  `AdaptiveAvgPool2d::new((1, 1))` for the squeeze stage.
- `ferrotorch-nn/src/prelude` re-exports
  `AdaptiveAvgPool2d, MaxPool2d` at
  `ferrotorch-nn/src/lib.rs:286`.

## Parity contract

### `nn.functional.max_pool2d`

- Upstream entry: `torch.nn.functional.max_pool2d` →
  `aten/src/ATen/native/MaxPooling.cpp`.
- Edge cases preserved by `max_pool2d_forward`:
  - **`padding > kernel_size / 2`** — upstream errors; ferrotorch
    matches.
  - **Padded values participate in the max** — upstream pads with
    `-inf` so padded positions never win; ferrotorch pads with the
    minimum representable `T` value.
  - **Non-default stride** — ferrotorch matches upstream's
    `(input + 2 * padding - kernel) / stride + 1` formula.
- Parity-sweep audit status: `MISSING` (blocker #1458).

### `nn.functional.avg_pool2d`

- Upstream entry: `torch.nn.functional.avg_pool2d`.
- Edge cases:
  - **`count_include_pad=False`** — upstream divides by the
    *actual* (unpadded) window size. ferrotorch divides by the
    full kernel area (matching the default `count_include_pad=True`).
    Custom `count_include_pad=False` is NOT-STARTED.
  - **`divisor_override`** — kwarg from upstream that lets callers
    pick a custom divisor — NOT-STARTED.
- Parity-sweep audit status: `MISSING` (blocker #1458).

### `nn.functional.adaptive_avg_pool2d`

- Upstream entry: `torch.nn.functional.adaptive_avg_pool2d`.
- Edge case: when `H_out / H_in != integer`, upstream's stride
  varies per output position. ferrotorch matches via the
  `start = (i * H_in) / H_out`, `end = ((i + 1) * H_in) / H_out`
  formula.
- Parity-sweep audit status: `MISSING` (blocker #1458).

### Other declared parity ops

`nn.functional.{max_pool1d, max_pool3d, avg_pool1d, avg_pool3d,
adaptive_avg_pool1d, adaptive_max_pool1d, adaptive_max_pool2d}`
all `MISSING` (blocker #1458). Same edge-case stories with N-D
generalisation.

## Verification

Tests in `mod tests in pooling.rs`. Highlights:

- Forward shape contracts for every pool.
- Numerical checks against hand-computed reference values.
- Round-trip tests for `MaxPool2d` → `MaxUnpool2d`.
- Backward correctness via finite differences for at least
  `MaxPool2d` and `AvgPool2d`.

Parity smoke command (blocker #1458 must close):

```bash
for OP in nn.functional.max_pool1d nn.functional.max_pool2d \
          nn.functional.max_pool3d nn.functional.avg_pool1d \
          nn.functional.avg_pool2d nn.functional.avg_pool3d \
          nn.functional.adaptive_avg_pool1d \
          nn.functional.adaptive_avg_pool2d \
          nn.functional.adaptive_max_pool1d \
          nn.functional.adaptive_max_pool2d; do
  ./target/release/parity-sweep sweep --op "$OP" --seeds 8 2>&1 \
    | grep -c "passed (0 skipped, 0 failed)"
done
```

Expected (post-#1458): each line returns `>= 1`. Current: each
returns `0`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct MaxPool2d` plus `impl<T: Float> Module<T> for MaxPool2d` in `pooling.rs`; non-test consumer: re-export at `ferrotorch-nn/src/lib.rs:237` + `ferrotorch-vision/src/models/resnet.rs:23` + many other vision models. |
| REQ-2 | SHIPPED | impl: `pub struct AvgPool2d` plus `impl<T: Float> Module<T> for AvgPool2d` in `pooling.rs`; non-test consumer: `ferrotorch-vision/src/models/densenet.rs:43` + `inception.rs:61` + re-export at `lib.rs:237`. |
| REQ-3 | SHIPPED | impl: `pub struct AdaptiveAvgPool2d` plus `impl<T: Float> Module<T> for AdaptiveAvgPool2d` in `pooling.rs`; non-test consumer: `ferrotorch-vision/src/models/resnet.rs:23` + `convnext.rs:35` + `efficientnet.rs:38` + `mobilenet.rs:55` + `segmentation/aspp.rs:38` + re-export at `lib.rs:237` + the prelude re-export at `lib.rs:286` + `ferrotorch-nn/src/se.rs` (SqueezeExcitation squeeze stage). |
| REQ-4 | SHIPPED | impl: `pub struct MaxPool1d` / `MaxPool3d` / `AvgPool1d` / `AvgPool3d` plus `impl<T: Float> Module<T> for *` in `pooling.rs`; non-test consumer: re-export at `lib.rs:237`. |
| REQ-5 | SHIPPED | impl: `pub struct AdaptiveAvgPool1d` / `AdaptiveAvgPool3d` / `AdaptiveMaxPool1d` / `AdaptiveMaxPool2d` / `AdaptiveMaxPool3d` plus their `impl Module<T> for *` blocks in `pooling.rs`; non-test consumer: re-export at `lib.rs:237`. |
| REQ-6 | SHIPPED | impl: `pub struct FractionalMaxPool2d` plus `impl<T: Float> Module<T>` in `pooling.rs`; non-test consumer: re-export at `lib.rs:237`. |
| REQ-7 | SHIPPED | impl: `pub struct LPPool1d` / `pub struct LPPool2d` plus their `impl Module<T>` blocks in `pooling.rs`; non-test consumer: re-export at `lib.rs:237`. |
| REQ-8 | SHIPPED | impl: `pub struct MaxUnpool2d` plus `pub fn max_unpool2d` in `pooling.rs`; non-test consumer: re-export at `lib.rs:237`. |
| REQ-9 | SHIPPED | impl: 14 `pub fn *_pool*<T: Float>` functional entries in `pooling.rs`; non-test consumer: re-export at `lib.rs:237`. |
| REQ-10 | SHIPPED | impl: `fn validate_4d`, `fn validate_pool_params`, etc. in `pooling.rs`; non-test consumer: invoked from every pool forward (re-exported at `lib.rs:237`). |
| REQ-11 | SHIPPED | impl: per-pool `GradFn<T>` types plus `Tensor::from_operation` calls in `pooling.rs`; non-test consumer: re-export at `lib.rs:237` — autograd engine traverses these GradFns on `backward()`. |
| REQ-12 | NOT-STARTED | parity-sweep runner arms for the 10 declared pooling ops (`max_pool1d/2d/3d`, `avg_pool1d/2d/3d`, `adaptive_avg_pool1d/2d`, `adaptive_max_pool1d/2d`) not wired — blocker #1458. |
