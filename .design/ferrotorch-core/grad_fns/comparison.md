# Comparison-area grad_fns (differentiable `where`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/native/BinaryOps.cpp
  - aten/src/ATen/native/TensorCompare.cpp
-->

## Summary

`ferrotorch-core/src/grad_fns/comparison.rs` (287 LOC) is the autograd-tracking
layer for PyTorch's differentiable conditional-selection op `torch.where`
(declared in `aten/src/ATen/native/TensorCompare.cpp:642 Tensor where(const
Tensor& condition, const Tensor& self, const Tensor& other)`). The file pairs
a forward `pub fn where_` with a `WhereBackward<T>` `GradFn` struct that
routes upstream gradients to the input selected at each position per
`tools/autograd/derivatives.yaml:1955-1959 - name: where.self(...) ...
self: where(condition, grad, 0); other: where(condition, 0, grad)`. A
second public entry `pub fn where_bt` wraps the same kernel for the
first-class `BoolTensor` mask path (closed by upstream tracking #615).

Notwithstanding the file's name (`comparison.rs`) and the parity-sweep
route's declared `parity_ops` list (`eq, ne, lt, le, gt, ge, logical_and,
logical_or, logical_xor, logical_not, max, min, maximum, minimum, isnan,
isinf, isfinite`), **the implementations of those 17 comparison ops do
not live in this file**. They live in `ferrotorch-core/src/bool_tensor.rs`
(see `pub fn gt`, `pub fn lt`, `pub fn ge`, `pub fn le`, `pub fn eq_t`,
`pub fn ne` in `BoolTensor` at `ne in bool_tensor.rs`). Discrepancy
tracked by blocker #1293.

## Requirements

- REQ-1: `where_(condition, x, y)` — pointwise differentiable selection
  taking `condition: &[bool]`, `x: &Tensor<T>`, `y: &Tensor<T>` of
  identical numel and returning a `Tensor<T>` whose `i`-th element is
  `x[i]` if `condition[i]`, else `y[i]`. Forward mirrors `Tensor
  where(const Tensor& condition, const Tensor& self, const Tensor& other)`
  at `aten/src/ATen/native/TensorCompare.cpp:642-648` (dispatched via
  `where_self_out` at `:592-640` → `where_kernel` at `:306`). When
  gradient tracking is enabled and either `x` or `y` requires grad, the
  output carries a `WhereBackward` node that on `.backward(grad_output)`
  routes `grad_output` to `grad_x` at true positions and to `grad_y` at
  false positions, per `tools/autograd/derivatives.yaml:1955-1959 - name:
  where.self(Tensor condition, Tensor self, Tensor other) -> Tensor /
  condition: non_differentiable / self: where(condition, grad, 0) /
  other: where(condition, 0, grad)`. **Diverges from upstream** on the
  condition representation: ferrotorch's `condition: &[bool]` is a CPU
  slice, while upstream's `condition: Tensor` permits a device-resident
  boolean tensor with 3-way broadcasting. The shape-broadcasting and
  device-resident-condition path is handled by a separate function
  `ferrotorch-core/src/grad_fns/indexing.rs:1999 pub fn where_cond_bcast`
  (not this file).

- REQ-2: `where_bt(cond, x, y)` — pointwise differentiable selection
  taking `cond: &BoolTensor` (a first-class boolean tensor wrapper) plus
  same-shaped `x, y`. Validates `cond.numel() == x.numel()` and `x.shape()
  == y.shape()`, then delegates to `where_(cond.data()?, ...)` for the
  numerics + autograd. Same upstream entry as REQ-1
  (`TensorCompare.cpp:642`); the BoolTensor wrapper is a ferrotorch
  internal convenience and has no direct upstream counterpart (PyTorch
  uses a regular `Tensor` with `kBool` scalar type as the condition).
  Shape-checking matches the upstream `TORCH_CHECK(condition_.scalar_type()
  == kBool, ...)` at `TensorCompare.cpp:626-629` (we check shape; upstream
  checks dtype because Python passed a `Tensor`).

- REQ-3: Forward selection matches upstream byte-for-byte (NaN
  propagation, Inf propagation, denormal preservation are all the trivial
  consequence of returning `x[i]` or `y[i]` unmodified — no arithmetic
  occurs). The CPU CPU/GPU dispatch path: forward computes the output
  vector on host via the `condition.iter().zip(...)` iterator, then
  uploads via `TensorStorage::on_device(result, device)?` so the
  returned tensor lives on the same device as `x`. Mirrors upstream's
  `out_device(condition, self_, other_)` at `TensorCompare.cpp:609`.
  **Diverges from upstream** in that ferrotorch always materializes
  `x.data_vec()?` and `y.data_vec()?` to CPU before selecting, instead
  of building a `TensorIteratorConfig` and dispatching `where_kernel` on
  the device. Closing R-CODE-4 would require a GPU-resident `where_`
  path — currently the silent CPU round trip is the contract.

- REQ-4: 17 routed parity ops (`eq, ne, lt, le, gt, ge, logical_and,
  logical_or, logical_xor, logical_not, max, min, maximum, minimum,
  isnan, isinf, isfinite`) are NOT implemented in this file. They are
  implemented in `ferrotorch-core/src/bool_tensor.rs` (the comparison
  cluster at `bool_tensor.rs:450-475`) and as `BoolTensor::*` methods —
  outside the routing of this `.rs` file. The route's `parity_ops` list
  is mis-targeted; the actual file ships `where_` / `where_bt` only.
  Blocker #1293 covers retargeting the route or relocating the ops.

## Acceptance Criteria

- [x] AC-1: `where_` on a same-shape `[bool] / Tensor / Tensor` triplet
  returns the expected pointwise-selection vector — verified by
  `where_bt_picks_correctly` in `first_class_tests` mod of
  `grad_fns/comparison.rs` and by `cpu_where` /
  `run_where_for_device("cpu", Device::Cpu)` at
  `ferrotorch-core/tests/conformance_elementwise.rs` (forward
  parity for float32 and float64 against pre-recorded fixtures).
- [x] AC-2: `where_` backward routes `grad_output` to `grad_x` on
  true positions and `grad_y` on false positions, with zeros on the
  unselected side — verified by `test_where_backward` in the `tests`
  mod of `grad_fns/comparison.rs` (the `sum(out).backward()` flow that
  pins `x_grad == [1, 0, 1, 0]` and `y_grad == [0, 1, 0, 1]` for
  `condition == [true, false, true, false]`) and by the autograd lane in
  `run_where_for_device` at
  `ferrotorch-core/tests/conformance_elementwise.rs:957-974` (float32) /
  `:1000-1017` (float64).
- [x] AC-3: When `is_grad_enabled() == false` (inside `no_grad`), the
  returned tensor has `grad_fn().is_none()` even if `x.requires_grad()`
  is true — verified by `test_where_no_grad` in the `tests` mod of
  `grad_fns/comparison.rs`.
- [x] AC-4: `where_bt` rejects shape-mismatched `cond` with a
  `FerrotorchError::ShapeMismatch` — verified by
  `where_bt_rejects_shape_mismatch` in `first_class_tests` mod of
  `grad_fns/comparison.rs`.
- [x] AC-5: GPU-resident inputs work for both `where_` and `where_bt`:
  the backward materializes `grad_x` / `grad_y` on CPU then uploads via
  `grad_x_tensor.to(device)?` / `grad_y_tensor.to(device)?` (see the
  `if device.is_cuda()` branch in `WhereBackward::backward` of
  `grad_fns/comparison.rs`) — exercised by `cuda_where` /
  `run_where_for_device("cuda:0", Device::Cuda(0))` at
  `Cuda in ferrotorch-core/tests/conformance_elementwise.rs`.
- [ ] AC-6: All 17 routed parity-sweep ops (`eq, ne, lt, le, gt, ge,
  logical_and, logical_or, logical_xor, logical_not, max, min, maximum,
  minimum, isnan, isinf, isfinite`) return `passed (0 skipped, 0 failed)`
  with `grep -c >= 1` at `--seeds 8`. Currently every routed op returns
  `0/N passed (N skipped, 0 failed)` because the parity-sweep runner has
  no dispatch arm for these names. Blocker #1293 covers the wiring.

## Architecture

### Layer placement

`grad_fns::comparison` sits in the autograd layer (parallel to
`grad_fns::activation`, `grad_fns::cumulative`, `grad_fns::shape`, etc.).
The file is unusual in the cluster because it contains NO `ops::*`
kernel-layer delegation — the forward is implemented inline in
`pub fn where_` by walking the `condition` slice and indexing into
`x_data` / `y_data` host vectors. No `ferrotorch-core/src/ops/comparison.rs`
exists; the comparison-op kernels that DO exist (eq/ne/lt/le/gt/ge,
logical_*, isnan/isinf/isfinite) live in `bool_tensor.rs` outside the
routing of this design doc.

### REQ-1 forward (`where_`)

`pub fn where_<T: Float>(condition: &[bool], x: &Tensor<T>, y: &Tensor<T>)
-> FerrotorchResult<Tensor<T>>` in `grad_fns/comparison.rs`:

1. Pull `x.device()` (the output lives on the same device as `x`).
2. Pull `x_data = x.data_vec()?` and `y_data = y.data_vec()?` (forces a
   CPU round trip for GPU-resident inputs — see REQ-3).
3. `debug_assert_eq!` the three lengths.
4. Build `result: Vec<T>` via `condition.iter().zip(x_data.iter()
   .zip(y_data.iter())).map(|(&c, (&xv, &yv))| if c { xv } else { yv })
   .collect()`.
5. Compute `needs_grad = is_grad_enabled() && (x.requires_grad() ||
   y.requires_grad())`.
6. Upload `result` via `TensorStorage::on_device(result, device)?`.
7. If `needs_grad`, build `WhereBackward { condition: condition.to_vec(),
   x: x.clone(), y: y.clone() }`, wrap in `Arc`, and call
   `Tensor::from_operation`. Otherwise call `Tensor::from_storage(..,
   false)`.

The output retains the SHAPE of `x` (which equals the shape of `y` —
caller responsibility to ensure same-shape).

### REQ-1 backward (`WhereBackward::backward`)

`impl<T: Float> GradFn<T> for WhereBackward<T>` in
`grad_fns/comparison.rs`:

1. `let go = grad_output.data_vec()?` — CPU round trip for the upstream
   grad.
2. Build `grad_x: Vec<T>` by zipping `go.iter()` with
   `self.condition.iter()` and projecting `if c { g } else { zero }`.
3. Build `grad_y: Vec<T>` symmetrically projecting `if c { zero } else { g }`.
4. Wrap each in a CPU `TensorStorage` then, if `device.is_cuda()`, upload
   each via `.to(device)?`.
5. Return `vec![Some(grad_x_tensor), Some(grad_y_tensor)]`.

The `inputs()` impl returns `vec![&self.x, &self.y]` (two-input op for
the autograd-graph topological walk; the `condition: Vec<bool>` is NOT a
graph input because the derivatives.yaml line `condition:
non_differentiable` marks it as such — ferrotorch encodes the
non-differentiability structurally by storing `condition` as a non-tensor
`Vec<bool>`).

### REQ-2 `where_bt` (BoolTensor variant)

`pub fn where_bt<T: Float>(cond: &BoolTensor, x: &Tensor<T>, y:
&Tensor<T>) -> FerrotorchResult<Tensor<T>>` in `grad_fns/comparison.rs`:

1. Check `cond.numel() == x.numel()`; on mismatch return
   `FerrotorchError::ShapeMismatch { message: format!("where_bt: cond
   numel={} != x numel={}", cond.numel(), x.numel()) }`.
2. Check `x.shape() == y.shape()`; on mismatch return
   `FerrotorchError::ShapeMismatch { message: format!("where_bt: x
   shape {:?} != y shape {:?}", x.shape(), y.shape()) }`.
3. Delegate to `where_(cond.data()?, x, y)`.

The `where_bt`-vs-`where_` split exists to give ferrotorch users a
type-safe boolean-tensor mask analogous to PyTorch's `kBool` scalar type
without forcing them through the `&[bool]` slice. Closed by tracking
issue #615.

### REQ-3 device handling

`x.device()` determines the output device. The host-side `result: Vec<T>`
is built by indexing CPU-resident `data_vec()` (one round trip per
input), then uploaded via `TensorStorage::on_device(result, device)?`.
The CPU round trip is the divergence from upstream's
`TensorIteratorConfig` device-dispatched kernel — see REQ-3 narrative.
Backward materializes both `grad_x` and `grad_y` on CPU then re-uploads
to the device of `grad_output` (see the `if device.is_cuda()` branch in
`WhereBackward::backward`).

### REQ-4 (route mismatch)

The 17 ops named in the route's `parity_ops` field are not in this file.
They are implemented in `ferrotorch-core/src/bool_tensor.rs`:
- `BoolTensor::gt<T>` (`pub fn gt` in `bool_tensor.rs`) mirroring
  `aten/src/ATen/native/BinaryOps.cpp:greater` registration.
- `BoolTensor::lt<T>` (`pub fn lt` in `bool_tensor.rs`) mirroring
  `BinaryOps.cpp:less`.
- `BoolTensor::ge<T>` (`pub fn ge` in `bool_tensor.rs`).
- `BoolTensor::le<T>` (`pub fn le` in `bool_tensor.rs`).
- `BoolTensor::eq_t<T>` (`pub fn eq_t` in `bool_tensor.rs`).
- `BoolTensor::ne<T>` (`pub fn ne` in `bool_tensor.rs`).
- Their `_int` variants at `ne in bool_tensor.rs`.

`logical_and / logical_or / logical_xor / logical_not / max / min /
maximum / minimum / isnan / isinf / isfinite` are not located in this
file either — they are either in `bool_tensor.rs`,
`ferrotorch-core/src/ops/reduction.rs` (for `max` / `min` reductions),
or absent altogether. Auditing the full coverage of these 11 ops is out
of scope for this design doc (the route mis-targets them); blocker
#1293 covers the cleanup.

## Parity contract

| Op | Upstream entry | Backward formula source | Edge cases mirrored / NOT mirrored |
|---|---|---|---|
| `where` (the differentiable selection) | `aten/src/ATen/native/TensorCompare.cpp:642 Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other)` | `tools/autograd/derivatives.yaml:1955-1959 self: where(condition, grad, 0); other: where(condition, 0, grad)` | **Mirrored**: NaN passthrough (no arithmetic on values; `x[i]` or `y[i]` returned unmodified). Inf passthrough. Denormals preserved. Pure-elementwise so non-contiguous doesn't matter on CPU. **NOT mirrored**: 3-way broadcasting (upstream's `TensorIteratorConfig` handles three-shape broadcast; ferrotorch requires same-numel `&[bool]` + same-shape `x` / `y`). Mixed dtype promotion (`at::native::result_type(self, other)` at `TensorCompare.cpp:597` promotes mismatched dtypes; ferrotorch requires same-type `Tensor<T>`). Scalar overloads (upstream provides three `Tensor where(Tensor cond, Scalar / Tensor, Tensor / Scalar)` at `:650-672`; ferrotorch ships only the `Tensor / Tensor` overload). uint8 deprecation warning (`TORCH_WARN_ONCE` at `:622-624`; ferrotorch has no `uint8` condition path). Backward-on-CUDA: ferrotorch backward forces a CPU materialization of `grad_output.data_vec()` — a silent round trip that upstream's GPU `where_kernel` avoids. |
| `where_bt` (BoolTensor variant) | same upstream entry as `where`; no direct upstream counterpart for the first-class BoolTensor wrapper | same `derivatives.yaml:1955-1959` (delegates to `where_`) | Same edge-case coverage as `where_` since the body delegates. Adds shape-check up front (upstream uses TensorIterator broadcasting; ferrotorch uses strict shape equality). |

Parity-sweep audit reference: the `where` op IS wired in the
parity-sweep runner at `tools/parity-sweep/runner/src/main.rs:799-813
"where" =>` (closes #1255) — but the dispatch routes to
`grad_fns::indexing::where_cond_bcast` (the 3-way-broadcast path),
**NOT** to `grad_fns::comparison::where_` of this file. As of this
writeup, `tools/parity-sweep/parity_audit.json` has no entry for
`where` (run returns `[where] 48/48 passed (0 skipped, 0 failed)` at
`--seeds 8` but the JSON status is `None`). The 17 routed ops in the
route table all return `0/N passed (N skipped)` — none are wired.

## Verification

### Existing unit tests (in `grad_fns/comparison.rs`)

`#[cfg(test)] mod first_class_tests`:
- `where_bt_picks_correctly` — verifies REQ-2 forward on shape `[4]` with
  alternating-condition fixture `[true, false, true, false]` over
  `[1.0, 2.0, 3.0, 4.0]` / `[10.0, 20.0, 30.0, 40.0]` → expected
  `[1.0, 20.0, 3.0, 40.0]`.
- `where_bt_rejects_shape_mismatch` — verifies REQ-2 shape-mismatch
  error path (AC-4).

`#[cfg(test)] mod tests`:
- `test_where_forward` — REQ-1 forward smoke (AC-1, smaller-scope
  variant of `where_bt_picks_correctly`).
- `test_where_backward` — REQ-1 backward verification: pins
  `x_grad = [1, 0, 1, 0]` and `y_grad = [0, 1, 0, 1]` for
  `condition = [true, false, true, false]` via a `sum(out).backward()`
  flow with a locally-defined `SumBackward` helper that injects ones as
  grad to its input (AC-2).
- `test_where_no_grad` — verifies that under `no_grad`, the returned
  tensor has `grad_fn().is_none()` and the data is still correct
  (AC-3).

### Conformance tests (in `ferrotorch-core/tests/conformance_elementwise.rs`)

- `cpu_where` (at `cpu_where in conformance_elementwise.rs`) — calls
  `run_where_for_device("cpu", Device::Cpu)`.
- `cuda_where` (at `conformance_elementwise.rs` inside the
  `#[cfg(feature = "cuda-pytorch-parity")]` block) — calls
  `run_where_for_device("cuda:0", Device::Cuda(0))`.
- `run_where_for_device` (at `run_where_for_device in conformance_elementwise.rs`)
  runs forward and backward parity for both `float32` and `float64`
  against pre-recorded fixtures, exercising AC-1, AC-2, and AC-5.

### Parity-sweep status

```
./target/release/parity-sweep sweep --op where --seeds 8
  => [where] 48/48 passed (0 skipped, 0 failed)
```

Smoke grep count for `passed (0 skipped, 0 failed)` is `1`. **However**,
the runner's `where` dispatch arm routes to
`grad_fns::indexing::where_cond_bcast` (not this file's `where_`); so
the smoke does NOT exercise `grad_fns::comparison::where_` end-to-end.
The conformance fixtures in `tests/conformance_elementwise.rs` are the
only live exercise of `grad_fns::comparison::where_`.

The 17 ops the route declares all return `0/N passed (N skipped, 0
failed)` at `--seeds 4`:
- eq: 0/40 passed (40 skipped)
- ne: 0/36 passed (36 skipped)
- lt: 0/36 passed (36 skipped)
- le: 0/36 passed (36 skipped)
- gt: 0/36 passed (36 skipped)
- ge: 0/36 passed (36 skipped)
- logical_and: 0/36 passed (36 skipped)
- logical_or: 0/36 passed (36 skipped)
- logical_xor: 0/36 passed (36 skipped)
- logical_not: 0/12 passed (12 skipped)
- max: 0/16 passed (16 skipped)
- min: 0/16 passed (16 skipped)
- maximum: 0/36 passed (36 skipped)
- minimum: 0/36 passed (36 skipped)
- isnan: 0/4 passed (4 skipped)
- isinf: 0/4 passed (4 skipped)
- isfinite: 0/12 passed (12 skipped)

The runner has no dispatch arms for these ops — confirmed by
`grep -n '"eq"\|"isnan"' tools/parity-sweep/runner/src/main.rs`
returning no matches in the dispatch table.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (where_ forward + backward) | NOT-STARTED | impl exists: `pub fn where_` in `grad_fns/comparison.rs` mirroring `aten/src/ATen/native/TensorCompare.cpp:642 Tensor where(...)` and `WhereBackward<T>` in `grad_fns/comparison.rs` mirroring `tools/autograd/derivatives.yaml:1955-1959`. Tests pass (`test_where_forward`, `test_where_backward`, `test_where_no_grad` in the `tests` mod + `cpu_where` / `cuda_where` in `conformance_elementwise.rs`). **However**, no non-test production consumer of `grad_fns::comparison::where_` exists in the workspace (`grep -rn "grad_fns::comparison\|where_(\|where_bt(" ferrotorch-*/src/ ferrotorch/src/` returns matches only inside `grad_fns/comparison.rs` itself; production-style callers in `ferrotorch-core/src/lib.rs` re-export `where_cond` / `where_cond_bt` from `grad_fns::indexing`, not from `grad_fns::comparison`). Open prereq blocker #1295 (wire `Tensor::where_t` method-style boundary or migrate to `grad_fns::indexing` variant). |
| REQ-2 (where_bt BoolTensor variant) | NOT-STARTED | impl exists: `pub fn where_bt` in `grad_fns/comparison.rs` delegating to `where_` with shape validation, no direct upstream counterpart (PyTorch uses a `Tensor` with `kBool` for the condition; `TensorCompare.cpp:626-629`). Tests pass (`where_bt_picks_correctly`, `where_bt_rejects_shape_mismatch` in `first_class_tests`; `where_bt` lane in `run_where_for_device` of `conformance_elementwise.rs`). No non-test production consumer of `grad_fns::comparison::where_bt` outside `comparison.rs` test mods (the workspace's only production BoolTensor-`where` path is `where in ferrotorch-core/src/ops/indexing.rs pub fn where_cond_bt` exported from `grad_fns in lib.rs pub use ops::indexing::{...where_cond_bt}`, which is a DIFFERENT function in a DIFFERENT module). Open prereq blocker #1297. |
| REQ-3 (device handling + NaN/Inf passthrough) | NOT-STARTED | partial: CPU and GPU forward both work (the `is_cuda()` upload branch in `where_` and the symmetric branch in `WhereBackward::backward` of `grad_fns/comparison.rs`). NaN / Inf trivially pass through because no arithmetic occurs (the impl is `if c { xv } else { yv }`). But the implementation diverges from upstream by materializing `x.data_vec()?` / `y.data_vec()?` / `grad_output.data_vec()?` on CPU before selecting — a silent CPU round trip that upstream's `where_kernel` at `TensorCompare.cpp:306, 638` avoids on the GPU path. Per R-CODE-4 this round trip is a bug that should be eliminated by a GPU-resident `where_` kernel. **No production consumer is using this code path** (REQ-1 blocker #1295 dominates); when a consumer lands, R-CODE-4 must be re-audited. Open prereq blocker #1295 (consumer) gates this REQ in turn. |
| REQ-4 (17 comparison parity ops the route declares) | NOT-STARTED | The 17 ops named in `tooling/translate-routes.toml` for this file (`eq, ne, lt, le, gt, ge, logical_and, logical_or, logical_xor, logical_not, max, min, maximum, minimum, isnan, isinf, isfinite`) are not implemented in `grad_fns/comparison.rs`. They are implemented elsewhere (eq/ne/lt/le/gt/ge in `bool_tensor.rs:450-475`; logical_*/max/min/maximum/minimum/isnan/isinf/isfinite either elsewhere or absent). The parity-sweep runner has no dispatch arms for them (every op returns `0/N passed (N skipped, 0 failed)`). Open prereq blocker #1293 (retarget the route or relocate the ops). |

---

## Honest under-claim

Every REQ above is classified NOT-STARTED because no non-test production
consumer of either `pub fn where_` or `pub fn where_bt` exists. Under a
strict reading of S5, the `pub fn where_` boundary IS the public API
surface and is grandfathered — this would make REQ-1 / REQ-2 SHIPPED.
The doc takes the conservative reading because the file's `pub` items
are not re-exported from `ferrotorch-core/src/lib.rs` (unlike the
sibling `grad_fns::cumulative::{cumsum, cumprod, ...}` cluster, which
IS re-exported at `lib.rs:159`), so they are not the user-visible API
surface — they are an internal-only `pub` that hasn't yet been wired to
a stable boundary. Blocker #1295 (and #1297 for the BoolTensor variant)
make the consumer-wiring concrete. When that wiring lands, REQ-1 and
REQ-2 promote to SHIPPED automatically.
