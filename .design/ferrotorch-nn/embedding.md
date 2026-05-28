# ferrotorch-nn — `embedding` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/sparse.py
  - aten/src/ATen/native/Embedding.cpp
-->

## Summary

`ferrotorch-nn/src/embedding.rs` implements `Embedding<T>` (the
standard lookup table) and `EmbeddingBag<T>` (a reduction over
variable-length bags of indices) mirroring `torch.nn.Embedding` and
`torch.nn.EmbeddingBag` at `torch/nn/modules/sparse.py`. The
backward is a sparse scatter-add — only accessed rows accumulate
gradient, with duplicate indices summing. Supports `padding_idx`
(zero-gradient row) and the `sparse=True` flag (carried + acted on
via `Embedding::sparse_grad` SparseGrad view). `max_norm` /
`norm_type` / `scale_grad_by_freq` are implemented on BOTH layers:
`max_norm` performs a faithful translation of the aten
`embedding_renorm_cpu_` kernel — touched rows of the PERSISTED
`weight` are renormalised IN PLACE before the gather/reduction (not
merely clipped on the output), so the mutation survives across
forward calls. `EmbeddingBag` additionally supports `sparse`,
`include_last_offset`, and `padding_idx` (excluded from the
reduction with the mean divisor decremented).

## Requirements

- REQ-1: `pub struct Embedding<T: Float>` with `weight: Parameter<T>`
  shape `[num_embeddings, embedding_dim]`, optional `padding_idx`,
  and `sparse` flag. Mirrors upstream's class attributes at
  `torch/nn/modules/sparse.py:37-50` and the `__init__` signature.
- REQ-2: `Embedding::new(num_embeddings, embedding_dim, padding_idx)`
  validates `padding_idx < num_embeddings`. Initializes weight from
  `N(0, 1)` then zeros the padding row if set. Mirrors upstream's
  `reset_parameters` at `sparse.py:188-194`.
- REQ-3: Forward — `<Embedding as Module>::forward` accepts an
  indices tensor (any shape), casts each element to `usize`, gathers
  the corresponding weight rows. Output shape is `(*input_shape,
  embedding_dim)`. Mirrors upstream's `F.embedding(input, weight,
  padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)`
  call at `sparse.py:181-186`.
- REQ-4: `EmbeddingBackward<T>: GradFn<T>` — sparse scatter-add of
  `grad_output` rows into a zero-initialized `[num_embeddings,
  embedding_dim]` grad_weight at the indices accessed in forward.
  Duplicate indices accumulate. If `padding_idx` is set, that row's
  grad is zeroed unconditionally. Mirrors
  `aten::embedding_backward`.
- REQ-5: GPU fast path — when `grad_output.is_cuda()` and `T` is
  `f32` or `f64`, dispatches to `backend.scatter_add_rows_f32/f64`
  on-device. The `padding_idx` zeroing is done via a small CPU
  roundtrip on the affected row.
- REQ-6: `Embedding::sparse_grad` — exposes a `SparseGrad<T>` view
  (unique-indices + corresponding gradient rows) for sparse
  optimizers like SparseAdam. Mirrors upstream's `sparse=True` flag
  contract.
- REQ-7: `pub struct EmbeddingBag<T: Float>` with `EmbeddingBagMode
  { Sum, Mean, Max }`. Forward accepts a flat indices tensor +
  offsets slice. 2D input `[num_bags, bag_size]` is treated as
  fixed-size bags via implicit offsets. Mirrors upstream
  `EmbeddingBag` at `sparse.py:264-454`.
- REQ-8: `Module<T>` impl on both `Embedding` and `EmbeddingBag` —
  `forward` / `parameters` / `parameters_mut` / `named_parameters`
  / `train` / `eval` / `is_training`.
- REQ-9: `max_norm` / `norm_type` / `scale_grad_by_freq` kwargs on
  BOTH `Embedding` and `EmbeddingBag`, plus `sparse` /
  `include_last_offset` / `padding_idx` on `EmbeddingBag`, mirroring
  `torch.nn.Embedding.__init__` (`sparse.py:134-163`) and
  `torch.nn.EmbeddingBag.__init__` (`sparse.py:370-414`). `max_norm`
  renormalises the touched rows of the PERSISTED weight in place
  before the gather/reduction — a faithful translation of
  `embedding_renorm_cpu_` (`aten/src/ATen/native/Embedding.cpp:181-212`),
  invoked exactly where `F.embedding` / `F.embedding_bag` invoke
  `_no_grad_embedding_renorm_` (`functional.py:2561-2573`,
  `2766-2771`). `EmbeddingBag` excludes `padding_idx` entries from the
  reduction and the mean divisor (`EmbeddingBag.cpp:140-156`); `max`
  mode rejects `scale_grad_by_freq`/`sparse` (`functional.py:2755-2761`).
  `freeze` (Parameter `requires_grad` toggle) is a thin wrapper over
  the existing `Parameter::set_requires_grad` and is not a separate
  surface here. `EmbeddingBag`'s `Module::forward` still returns a
  non-grad-tracked tensor (per-bag backward unchanged this iter).
- REQ-11: `per_sample_weights` (#1610) on `EmbeddingBag` via
  `forward_bag_weighted(input, offsets, per_sample_weights:
  Option<&Tensor<T>>)`, mirroring `F.embedding_bag(..., per_sample_weights)`
  (`torch/nn/functional.py:2576-2791`, `torch/nn/modules/sparse.py:425-473`).
  When `psw` is supplied each gathered embedding row is multiplied by its
  sample weight BEFORE the sum reduction (`output[bag] += weight[idx] * psw`,
  `aten/src/ATen/native/EmbeddingBag.cpp:537-543`). It is ONLY valid for
  `mode='sum'` — any other mode returns torch's byte-identical
  `NotImplementedError` text (`functional.py:2773-2778`) — and `psw` must
  share the input's shape (`functional.py:2698-2702`). The weighted forward is
  grad-tracked: `EmbeddingBagSumWeightedBackward` flows gradient to BOTH the
  embedding table (`grad_weight[idx] += grad[bag] * psw`,
  `EmbeddingBag.cpp:1564-1582`) AND `per_sample_weights` (`grad_psw[i] =
  dot(grad[bag], weight[idx])`, `EmbeddingBag.cpp:1716-1724`). `padding_idx`
  samples contribute 0 to the reduction and to BOTH gradients (skipped at
  `EmbeddingBag.cpp:537,1561,1720`). The unweighted 2-arg `forward_bag`
  delegates to `forward_bag_weighted(.., None)` and is unchanged (non-grad
  reduction). The runner arm now FEEDS the 96 `per_sample_weights` op_db
  samples through `forward_bag_weighted` (#1441): for the 2-D fixed-bag layout
  the indices AND `psw` are both flattened to 1-D with implicit per-row offsets
  exactly as torch does (`functional.py:2736-2738`). The production capability
  is verified by the live-torch oracle lib tests AND the parity sweep.
- REQ-10: SHIPPED — parity-sweep runner arms for
  `nn.functional.embedding` and `nn.functional.embedding_bag` are wired
  in `tools/parity-sweep/runner/src/main.rs` (#1441). The `embedding`
  arm builds `Embedding::from_pretrained` + `with_max_norm` /
  `with_norm_type` / `with_scale_grad_by_freq` + `Module::forward`
  (0-D / 1-D / N-D indices, `max_norm` / `norm_type`,
  `scale_grad_by_freq` + `sparse` forward-inert all RUN); reaches
  80/80 at `--seeds 8` (0 skip, 0 failed). The `embedding_bag` arm
  builds `EmbeddingBag::new_with` + `with_*` + `forward_bag_weighted` (1-D +
  offsets, optional `per_sample_weights`) / flatten-to-1D + implicit per-row
  offsets for 2-D fixed bags, covering sum/mean/max, offsets,
  include_last_offset, padding_idx (negatives wrapped), max_norm/norm_type,
  AND `per_sample_weights` (#1610); reaches 392/392 at `--seeds 8`
  (0 skip, 0 failed). The 96 `per_sample_weights` samples that previously
  `Ok(None)`-skipped now RUN through `forward_bag_weighted`.

## Acceptance Criteria

- [x] AC-1: `Embedding::new` rejects `padding_idx >= num_embeddings`.
- [x] AC-2: Forward returns shape `(*input_shape, embedding_dim)`.
- [x] AC-3: Backward accumulates duplicate indices
  (`test_embedding_backward_duplicate_indices`).
- [x] AC-4: `padding_idx` row stays zero after gradient updates
  (`test_embedding_padding_idx_no_grad`).
- [x] AC-5: GPU scatter-add path returns correct gradients
  (`test_embedding_backward_gpu_*` under
  `ferrotorch-gpu/tests/integration/embedding.rs`).
- [x] AC-6: `EmbeddingBag` Sum/Mean/Max modes produce correct
  reductions over fixed-size and variable-length bags.
- [x] AC-7: `max_norm` enforcement — the persisted weight is mutated
  in place (`test_max_norm_mutates_persisted_weight`,
  `test_max_norm_second_forward_is_stable`,
  `test_bag_max_norm_mutates_weight`), values from live torch 2.11.
- [x] AC-8: `scale_grad_by_freq` weighting
  (`test_scale_grad_by_freq_divides_duplicates`,
  `test_scale_grad_by_freq_off_accumulates`).
- [x] AC-9: parity-sweep arms wired — #1441. `nn.functional.embedding`
  reaches 80/80 (0 skip, 0 failed); `nn.functional.embedding_bag`
  reaches 392/392 (0 skip, 0 failed) — the 96 `per_sample_weights`
  samples now RUN through `forward_bag_weighted` (#1610) instead of
  skipping.

## Architecture

### `Embedding` struct (REQ-1, REQ-2)

`pub struct Embedding<T: Float>` in `embedding.rs` with fields
`weight: Parameter<T>`, `num_embeddings`, `embedding_dim`,
`padding_idx`, `training`, `sparse: bool`, and `last_indices:
Mutex<Option<Vec<usize>>>` (for sparse-grad coalescing).
`Embedding::new` rejects out-of-range `padding_idx`, initialises
weight from `N(0,1)` via `init::normal`, then zeros the padding row
if set.

### Forward (REQ-3)

`<Embedding<T> as Module<T>>::forward` in `embedding.rs`:
1. Casts each element of `input` to `usize`, bounds-checks each
   against `num_embeddings`.
2. Gathers the corresponding weight rows into a flat output buffer.
3. Constructs the output tensor with shape
   `(*input.shape(), embedding_dim)`.
4. Attaches `EmbeddingBackward` GradFn when grad is required.

### Backward (REQ-4, REQ-5)

`pub struct EmbeddingBackward<T: Float>` impls `GradFn<T>` in
`embedding.rs`. On CPU, allocates a `[num_embeddings, embedding_dim]`
zero buffer then scatter-adds `grad_output` rows by index. On GPU
(f32/f64 only), uploads the indices vector to GPU and dispatches
`backend.scatter_add_rows_f32/f64`, then if `padding_idx` is set,
zeroes that row via a small CPU roundtrip.

### Sparse-grad view (REQ-6)

`Embedding::sparse_grad` (declared further down in `embedding.rs`)
returns a `SparseGrad<T>` carrying the deduplicated set of indices
touched by the most recent forward + the corresponding rows of the
dense grad. Consumed by `ferrotorch_optim::SparseAdam`.

### EmbeddingBag (REQ-7)

`pub struct EmbeddingBag<T: Float>` in `embedding.rs`. `pub fn
new(num_embeddings, embedding_dim, mode)` initializes weight from
`N(0,1)`. `pub fn forward_bag(input, offsets)` computes the
per-bag reduction (sum, mean, or elementwise max) directly without
materializing the full per-element embedding. `<EmbeddingBag as
Module>::forward` accepts 1D (single bag) or 2D (`[num_bags,
bag_size]` fixed-size bags) inputs.

### Module impl (REQ-8)

`impl<T: Float> Module<T> for Embedding<T>` and `impl<T: Float>
Module<T> for EmbeddingBag<T>` in `embedding.rs`. `parameters()`
returns `[&weight]`, `named_parameters()` yields `("weight",
&weight)`.

### Non-test production consumers

- `pub use embedding::{Embedding, EmbeddingBag, EmbeddingBagMode}`
  at `ferrotorch-nn/src/lib.rs`.
- `ferrotorch-llama/src/model.rs` carries `pub embed_tokens:
  Embedding<T>` and constructs it via `Embedding::new(cfg.vocab_size,
  cfg.hidden_size, None)?` — the standard Llama embedding table. This
  is the consumer for REQ-1/2/3 (the basic lookup table) only. It
  constructs with `padding_idx=None`, `max_norm=None`, default
  `norm_type`, and never touches `EmbeddingBag` — so it is **NOT** a
  consumer of the REQ-9 `max_norm`/`norm_type`/`scale_grad_by_freq`/
  `EmbeddingBag` kwargs (#1566 corrected the earlier overstatement).
- For REQ-9 specifically: per goal.md S5, `Embedding` and
  `EmbeddingBag` are themselves the **boundary public API** mirroring
  `torch.nn.Embedding` / `torch.nn.EmbeddingBag` — the module IS what
  users call. The builder kwargs + `forward`/`forward_bag` are the
  consumer surface; no further downstream caller is required for the
  REQ to be SHIPPED (S5 grandfathers boundary methods).
- `ferrotorch_optim`'s SparseAdam consumes `Embedding::sparse_grad`
  in its update step.

## Parity contract

`parity_ops = ["nn.functional.embedding",
"nn.functional.embedding_bag"]`.

For `embedding`:
- **Out-of-bound index** — upstream raises
  `RuntimeError: index out of range`; ferrotorch returns
  `IndexOutOfBounds`.
- **Negative index** — upstream wraps via `index += num_embeddings`
  for negative values; ferrotorch's `usize` cast rejects negatives
  (cast fails). Tracked as a follow-up; in practice the indices fed
  into Embedding are non-negative tokens.
- **padding_idx** — both return zeros for that token and zero
  gradient on backward.
- **dtype** — upstream supports `int32/int64` index types;
  ferrotorch stores indices as `T` (f32/f64) and casts to `usize`.
  Mathematically equivalent for token indices < `2^24` (f32) /
  `2^53` (f64); diverges for very large vocabularies — tracked.

For `embedding_bag`:
- **mode** — Sum/Mean/Max parity matches upstream.
- **Empty bag** (offsets imply a 0-length bag) — both return zero
  vector for that bag.
- **per_sample_weights** — IMPLEMENTED in production (#1610) via
  `EmbeddingBag::forward_bag_weighted` (sum-mode-only scaling + grad to
  BOTH weight and psw; `mode!='sum'` and shape-mismatch errors match
  torch). The runner arm now FEEDS these 96 samples through
  `forward_bag_weighted` (#1441): the 2-D fixed-bag samples flatten both the
  indices and `psw` to 1-D with implicit per-row offsets, mirroring torch's
  `functional.py:2736-2738` reshape-to-`-1`. Verified by live-torch 2.11 oracle
  lib tests (`test_bag_psw_*`) AND the parity sweep at 0-skip / 0-failed.
  (Previously mis-cited #1445, which was the max_norm/norm_type work —
  corrected to #1610.)

Parity-sweep audit entries: both ops `verified` (#1441) —
`nn.functional.embedding` at 80/80 (0 skip / 0 failed),
`nn.functional.embedding_bag` at 392/392 (0 skip / 0 failed; the 96
`per_sample_weights` samples now RUN, #1610).

## Verification

Tests in `mod tests` of `embedding.rs` (~20 tests):
- `test_embedding_forward_shape`,
  `test_embedding_forward_correctness`,
  `test_embedding_padding_idx_no_grad`,
  `test_embedding_backward_duplicate_indices`,
  `test_embedding_backward_gpu_*` (CUDA-only).
- `test_embedding_bag_sum`,
  `test_embedding_bag_mean`,
  `test_embedding_bag_max`,
  `test_embedding_bag_2d_fixed_bags`.

Parity-sweep smoke commands (both at 0-skip / 0-failed after #1441):

```bash
./target/release/parity-sweep sweep --op nn.functional.embedding --seeds 8 2>&1 | tail -1
./target/release/parity-sweep sweep --op nn.functional.embedding_bag --seeds 8 2>&1 | tail -1
```

Grep count `passed (0 skipped, 0 failed)` is `>= 1` for each (embedding
80/80, embedding_bag 392/392).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Embedding<T: Float>` in `embedding.rs` with `weight`/`num_embeddings`/`embedding_dim`/`padding_idx`/`sparse` fields mirroring `torch/nn/modules/sparse.py:37-50`; non-test consumer: `ferrotorch-llama/src/model.rs` declares `pub embed_tokens: Embedding<T>` as a model field. |
| REQ-2 | SHIPPED | impl: `Embedding::new` (with `padding_idx` validation + N(0,1) init + padding-row zero) in `embedding.rs`; non-test consumer: `Embedding::new(cfg.vocab_size, cfg.hidden_size, None)?` in `ferrotorch-llama/src/model.rs` is the Llama model's token-embedding constructor. |
| REQ-3 | SHIPPED | impl: `<Embedding as Module>::forward` body in `embedding.rs` (gather + grad-attach); non-test consumer: `ferrotorch-llama` model's `forward` calls `self.embed_tokens.forward(input_ids)` on every training step and inference token. |
| REQ-4 | SHIPPED | impl: `pub struct EmbeddingBackward<T>` and its `GradFn::backward` body in `embedding.rs`; non-test consumer: every `loss.backward()` call in the Llama training scaffolding traverses `EmbeddingBackward` nodes via `ferrotorch_core::autograd::engine`. |
| REQ-5 | SHIPPED | impl: `grad_output.is_cuda()` + `scatter_add_rows_f32/f64` dispatch in `EmbeddingBackward::backward` in `embedding.rs`; non-test consumer: `ferrotorch-gpu/src/backend_impl.rs` exposes `Backend::scatter_add_rows_f32`; GPU training-loop runs on the Llama model trigger this on every embedding backward. |
| REQ-6 | SHIPPED | impl: `Embedding::sparse_grad` defined further in `embedding.rs` returning a `SparseGrad<T>`; non-test consumer: `ferrotorch_optim::SparseAdam` consumes this view in its update path (the `sparse=True` codepath in optim). |
| REQ-7 | SHIPPED | impl: `pub struct EmbeddingBag<T: Float>` + `pub enum EmbeddingBagMode` + `impl Module` in `embedding.rs`; non-test consumer: `pub use embedding::{EmbeddingBag, EmbeddingBagMode}` in `lib.rs` exposes the type for downstream models. |
| REQ-8 | SHIPPED | impl: both `Module<T> for Embedding<T>` and `Module<T> for EmbeddingBag<T>` impl blocks in `embedding.rs`; non-test consumer: `ferrotorch_optim::Optimizer` iterates `model.parameters_mut()` which surfaces the embedding's weight parameter for every step. |
| REQ-9 | SHIPPED | impl: free fn `renorm_weight_rows_in_place` (faithful `embedding_renorm_cpu_` translation, persisted in-place weight mutation via `Tensor::update_data`, row norm via `at::norm` special-cased per `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:191-203` for `norm_type` 0/+inf/-inf; the default `norm_type == 2.0` f32 row reduces via `ferrotorch_core::simd_reduce::l2_norm_f32_torch` per the vectorized last-dim L2 kernel `ReduceOpsKernel.cpp:222-255`, f32 accumulator, so the `norm > max_norm` boundary decision matches torch byte-for-byte — closes the powf-vs-`v*v` summation gap #1612 left open, #1614) in `embedding.rs`, called by `Embedding::renorm_weight_in_place` and `EmbeddingBag::forward_bag`; `with_max_norm`/`with_norm_type`/`with_scale_grad_by_freq` on `Embedding`, plus `EmbeddingBag::new_with` + `with_max_norm`/`with_norm_type`/`with_scale_grad_by_freq`/`with_sparse`/`with_include_last_offset` + `padding_idx` exclusion in `forward_bag`. Consumer surface: per goal.md S5, `Embedding`/`EmbeddingBag` ARE boundary public API mirroring `torch.nn.Embedding`/`torch.nn.EmbeddingBag` (the user-facing kwargs ARE the deliverable) — grandfathered SHIPPED with no further downstream caller required. `<Embedding as Module>::forward` calls `self.renorm_weight_in_place(&indices)?` before every gather (renorm on the live forward path, no-op when `max_norm` unset); `EmbeddingBag::forward_bag`/`<EmbeddingBag as Module>::forward` consume the bag kwargs; both re-exported via `pub use embedding::{Embedding, EmbeddingBag, EmbeddingBagMode}` in `lib.rs` as the public consumer surface. `EmbeddingBag` per-bag backward remains unimplemented (forward returns a non-grad tensor) — tracked separately. (NB #1566: prior cite of `ferrotorch-llama/src/model.rs embed_tokens` as the renorm consumer was FALSE — `model.rs` uses `Embedding::new(.., None)` with no `max_norm`/`EmbeddingBag`; corrected to the S5 boundary-API rationale.) |
| REQ-10 | SHIPPED | impl: the `nn.functional.embedding` arm (builds `Embedding::from_pretrained` + `with_max_norm`/`with_norm_type`/`with_scale_grad_by_freq` + `Module::forward`) and the `nn.functional.embedding_bag` arm (builds `EmbeddingBag::new_with` + `with_*` + `Parameter::set_data` via `Module::parameters_mut` + `forward_bag_weighted` for the 1-D-offsets path / flatten-to-1D + implicit per-row offsets for the 2-D fixed-bag path) in `tools/parity-sweep/runner/src/main.rs` (#1441). Non-test production consumer of the wired surface: `<Embedding as Module>::forward` (driven on every Llama token via `ferrotorch-llama/src/model.rs` `embed_tokens.forward`) and `EmbeddingBag` as boundary public API re-exported at `ferrotorch-nn/src/lib.rs` `pub use embedding::{Embedding, EmbeddingBag, EmbeddingBagMode}` (goal.md S5). Sweep `--seeds 8`: embedding 80/80 (0 skip, 0 failed); embedding_bag 392/392 (0 skip, 0 failed). The 96 `per_sample_weights` samples that previously `Ok(None)`-skipped now RUN through `forward_bag_weighted` (REQ-11 / #1610), flattening indices + `psw` to 1-D for 2-D fixed bags exactly as torch (`functional.py:2736-2738`). `padding_idx` is forward-inert for `F.embedding` (gathers the actual weight row; only backward grad zeroed) so the arm passes `None` to the layer to match torch's functional. |
| REQ-11 | SHIPPED | impl: `EmbeddingBag::forward_bag_weighted(input, offsets, per_sample_weights: Option<&Tensor<T>>)` in `embedding.rs` — sum-mode-only per-sample scaling before the reduction (`aten/src/ATen/native/EmbeddingBag.cpp:537-543`), with `EmbeddingBagSumWeightedBackward` (`embedding.rs`) flowing grad to BOTH the embedding table (`EmbeddingBag.cpp:1564-1582`) and `per_sample_weights` (`EmbeddingBag.cpp:1716-1724`); `mode!='sum'` returns torch's exact `NotImplementedError` text (`torch/nn/functional.py:2773-2778`) and shape-mismatch returns the `functional.py:2698-2702` error; `padding_idx` samples contribute 0 to both grads. Non-test production consumer: the existing 2-arg `EmbeddingBag::forward_bag` (called by the parity-sweep runner's `embedding_bag` arm AND boundary public API re-exported at `ferrotorch-nn/src/lib.rs` `pub use embedding::{EmbeddingBag, ..}`, goal.md S5) is rewired to call `forward_bag_weighted(.., None)` in the same commit — `forward_bag_weighted` is the unified reduction body that `forward_bag` now delegates to (so the new pub method has an in-production caller per R-DEFER-1). Verified by live-torch 2.11.0 oracle lib tests `test_bag_psw_sum_forward_single_bag` / `test_bag_psw_sum_grad_to_weight_and_psw` / `test_bag_psw_sum_two_bags_offsets` / `test_bag_psw_with_padding_idx` / `test_bag_psw_end_to_end_autograd` / `test_bag_psw_rejects_mean_and_max_modes` / `test_bag_psw_rejects_shape_mismatch` in `embedding.rs`. |
