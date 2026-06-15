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

`ferrotorch-core/src/grad_fns/comparison.rs` is the autograd-tracking
layer for PyTorch's differentiable conditional-selection op `torch.where`
(declared in `aten/src/ATen/native/TensorCompare.cpp:642 Tensor where(const
Tensor& condition, const Tensor& self, const Tensor& other)`). The host-slice
entry `pub fn where_` treats `condition: &[bool]` as the full flat output mask
for `broadcast_shapes(self, other)`. The first-class `pub fn where_bt` entry
delegates to `grad_fns::indexing::where_cond_bcast`, so condition/self/other
broadcast by PyTorch rules and CUDA tensors run through the resident
`ops::indexing::where_cond_bt` kernel. Same-shape CPU `where_` still uses the
legacy `WhereBackward<T>` node; broadcasted and CUDA cases use
`WhereCondBackward` plus `ExpandBackward` reductions, matching
`tools/autograd/derivatives.yaml:1955-1959`.

Notwithstanding the file's name (`comparison.rs`) and the parity-sweep
route's declared `parity_ops` list (`eq, ne, lt, le, gt, ge, logical_and,
logical_or, logical_xor, logical_not, max, min, maximum, minimum, isnan,
isinf, isfinite`), **this file owns differentiable `where` only**. Float
and integer bool-producing comparisons plus bool logical ops live in
`ferrotorch-core/src/bool_tensor.rs`; CUDA broadcasted i32/i64 comparisons
route through `GpuBackend::compare_broadcast` and stay device-resident.
Value-returning extrema and value-predicate APIs are owned by their dedicated
modules. Discrepancy in the route ownership is tracked by blocker #1293.

## Requirements

- REQ-1: `where_(condition, x, y)` — pointwise differentiable selection
  taking `condition: &[bool]`, `x: &Tensor<T>`, `y: &Tensor<T>` and returning
  a `Tensor<T>` whose `i`-th output element is `x` if `condition[i]`, else `y`.
  The raw condition slice has no independent shape, so its length must equal
  the full output numel of `broadcast_shapes(x.shape(), y.shape())`. Forward
  mirrors `Tensor where(const Tensor& condition, const Tensor& self, const
  Tensor& other)` at `aten/src/ATen/native/TensorCompare.cpp:642-648`
  (dispatched via `where_self_out` at `:592-640` → `where_kernel` at `:306`).
  When gradient tracking is enabled and either `x` or `y` requires grad, the
  output carries a backward node that routes `grad_output` to `grad_x` at true
  positions and to `grad_y` at false positions, per
  `tools/autograd/derivatives.yaml:1955-1959`.
  CUDA inputs upload only the host mask to the device, then use the resident
  `where_cond_bcast` path; value tensors do not round-trip through CPU.

- REQ-2: `where_bt(cond, x, y)` — pointwise differentiable selection
  taking `cond: &BoolTensor` (a first-class boolean tensor wrapper) plus
  `x` and `y`. Delegates directly to `grad_fns::indexing::where_cond_bcast`,
  which broadcasts all three operands to a common shape (TensorIterator parity)
  and preserves CUDA residency. Same upstream entry as REQ-1
  (`TensorCompare.cpp:642`); the BoolTensor wrapper corresponds to PyTorch's
  regular `kBool` condition tensor.

- REQ-3: Forward selection matches upstream byte-for-byte (NaN propagation,
  Inf propagation, denormal preservation are all the trivial consequence of
  returning `x` or `y` unmodified — no arithmetic occurs). Device handling now
  mirrors upstream's `out_device(condition, self_, other_)` at
  `TensorCompare.cpp:609`: CUDA `where_bt` stays resident end-to-end, and CUDA
  host-mask `where_` uploads only the host mask before launching the resident
  where kernel.

- REQ-4: Comparison/logical predicate surface is split across owners. This
  file intentionally ships `where_` / `where_bt` only. Float and integer
  `eq/ne/lt/le/gt/ge` plus bool `logical_and` / `logical_or` /
  `logical_xor` / `logical_not` live in `bool_tensor.rs`; CUDA broadcasted
  i32/i64 comparisons route through `GpuBackend::compare_broadcast` rather
  than a CPU value round trip. Other value-returning extrema / predicate APIs
  are owned by their dedicated modules. Blocker #1293 covers retargeting the
  route metadata.

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
- [x] AC-4: `where_bt` rejects non-broadcastable `cond` with a
  `FerrotorchError::ShapeMismatch` — verified by `where_bt_rejects_shape_mismatch`
  in `first_class_tests` mod of `grad_fns/comparison.rs`.
- [x] AC-5: GPU-resident inputs work for both public method paths without value
  round trips: `Tensor::where_t` uploads only the host mask and keeps result +
  gradients CUDA-resident; `Tensor::where_bt_t` broadcasts a CUDA BoolTensor
  condition and keeps the result CUDA-resident. Pinned by
  `tensor_where_t_host_mask_on_cuda_stays_resident_and_backpropagates` and
  `tensor_where_bt_broadcast_cuda_condition_stays_resident_and_reduces_grads` in
  `ferrotorch-gpu/tests/divergence_indexing_masked_fill_parity_cuda.rs`.
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
The file is unusual in the cluster because it only owns the differentiable
`where` public wrappers and the legacy same-shape CPU fast path. CUDA and
broadcasted cases delegate into `grad_fns::indexing::where_cond_bcast`, which
uses the kernel-layer `ops::indexing::where_cond_bt` resident implementation.
No `ferrotorch-core/src/ops/comparison.rs` exists; the comparison-op kernels
that do exist (eq/ne/lt/le/gt/ge, logical_*, isnan/isinf/isfinite) live in
`bool_tensor.rs` outside the routing of this design doc.

### REQ-1 forward (`where_`)

`pub fn where_<T: Float>(condition: &[bool], x: &Tensor<T>, y: &Tensor<T>)
-> FerrotorchResult<Tensor<T>>` in `grad_fns/comparison.rs`:

1. Pull `x.device()` (the output lives on the same device as `x`).
2. Compute `common = broadcast_shapes(x.shape(), y.shape())` and require
   `condition.len() == prod(common)`.
3. If either operand is CUDA, or if `x/y` require broadcasting, build a
   `BoolTensor` shaped as `common`, move it to the operand device when needed,
   and delegate to `grad_fns::indexing::where_cond_bcast`.
4. Otherwise (same-shape CPU fast path), build `result: Vec<T>` via
   `condition.iter().zip(x_data.iter()
   .zip(y_data.iter())).map(|(&c, (&xv, &yv))| if c { xv } else { yv })
   .collect()`.
5. Compute `needs_grad = is_grad_enabled() && (x.requires_grad() ||
   y.requires_grad())`.
6. If `needs_grad`, build `WhereBackward { condition: condition.to_vec(),
   x: x.clone(), y: y.clone() }`, wrap in `Arc`, and call
   `Tensor::from_operation`. Otherwise call `Tensor::from_storage(..,
   false)`.

The host-slice path can only express a full-output condition mask because
`&[bool]` carries no shape. Use `where_bt` for broadcasted condition tensors.

### REQ-1 backward (`WhereBackward::backward`)

`impl<T: Float> GradFn<T> for WhereBackward<T>` in
`grad_fns/comparison.rs` is now only reached by the same-shape CPU fast path:

1. `let go = grad_output.data_vec()?` — host read of the CPU upstream grad.
2. Build `grad_x: Vec<T>` by zipping `go.iter()` with
   `self.condition.iter()` and projecting `if c { g } else { zero }`.
3. Build `grad_y: Vec<T>` symmetrically projecting `if c { zero } else { g }`.
4. Wrap each in a CPU `TensorStorage`.
5. Return `vec![Some(grad_x_tensor), Some(grad_y_tensor)]`.

The `inputs()` impl returns `vec![&self.x, &self.y]` (two-input op for
the autograd-graph topological walk; the `condition: Vec<bool>` is NOT a
graph input because the derivatives.yaml line `condition:
non_differentiable` marks it as such — ferrotorch encodes the
non-differentiability structurally by storing `condition` as a non-tensor
`Vec<bool>`).

CUDA and broadcasted CPU cases do not use `WhereBackward`. They delegate to
`WhereCondBackward` through `where_cond_bcast`; gradients are routed through
the resident `masked_fill_dt` / `bool_not` CUDA path when applicable and then
through `ContiguousBackward` / `ExpandBackward` reductions for broadcasted
operands.

### REQ-2 `where_bt` (BoolTensor variant)

`pub fn where_bt<T: Float>(cond: &BoolTensor, x: &Tensor<T>, y:
&Tensor<T>) -> FerrotorchResult<Tensor<T>>` in `grad_fns/comparison.rs`
delegates to `grad_fns::indexing::where_cond_bcast(cond, x, y)`. That wrapper:

1. Computes the 3-way common broadcast shape.
2. Broadcasts the BoolTensor condition on its current device.
3. Expands `x` and `y` through autograd-aware `ExpandBackward` nodes.
4. Delegates to `ops::indexing::where_cond_bt`, whose CUDA branch launches
   `backend.where_cond` and whose CPU branch performs the same pointwise select.

The `where_bt`-vs-`where_` split exists to give ferrotorch users a
type-safe boolean-tensor mask analogous to PyTorch's `kBool` scalar type
without forcing them through the `&[bool]` slice. Closed by tracking
issue #615.

### REQ-3 device handling

`x.device()` determines the output device and must match `y.device()`.
`where_bt` additionally requires the condition to live on the same device once
broadcasted; CPU masks remain CPU, CUDA masks remain CUDA. Host-mask `where_`
constructs the condition on CPU, then moves only that mask to CUDA when the
value operands are CUDA-resident.

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
`ferrotorch-core/src/grad_fns/reduction.rs` (for `max` / `min` reductions),
or absent altogether. Auditing the full coverage of these 11 ops is out
of scope for this design doc (the route mis-targets them); blocker
#1293 covers the cleanup.

## Parity contract

| Op | Upstream entry | Backward formula source | Edge cases mirrored / NOT mirrored |
|---|---|---|---|
| `where` (host-slice condition) | `aten/src/ATen/native/TensorCompare.cpp:642 Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other)` | `tools/autograd/derivatives.yaml:1955-1959 self: where(condition, grad, 0); other: where(condition, 0, grad)` | **Mirrored**: value operands broadcast by PyTorch rules; CUDA operands stay resident; NaN/Inf/denormal values pass through unmodified. The raw `&[bool]` has no shape, so ferrotorch treats it as the full flat output mask and requires `condition.len() == prod(broadcast_shape(self, other))`. **NOT mirrored**: no shaped host condition broadcasting, mixed dtype promotion, scalar overloads, or uint8 condition deprecation path. |
| `where_bt` (BoolTensor variant) | same upstream entry as `where`; no direct upstream counterpart for the first-class BoolTensor wrapper | same `derivatives.yaml:1955-1959` through `WhereCondBackward` | **Mirrored**: 3-way broadcasting of condition/self/other, same-device validation, CUDA-resident forward and backward for f32/f64/f16/bf16 value tensors through the indexing where kernels and broadcast reductions. **NOT mirrored**: mixed dtype promotion, scalar overloads, or uint8 conditions. |

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
- `where_bt_broadcasts_three_inputs_and_reduces_grads` — verifies 3-way
  broadcast forward plus `ExpandBackward` reduction to `[1, 3]` and scalar
  operands.

`#[cfg(test)] mod tests`:
- `test_where_forward` — REQ-1 forward smoke (AC-1, smaller-scope
  variant of `where_bt_picks_correctly`).
- `test_where_host_mask_broadcasts_operands` — verifies the host-slice entry
  accepts broadcast-compatible value operands when the mask covers the full
  output.
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
the smoke does not exercise the host-slice `grad_fns::comparison::where_`
entry directly. The public method probes in
`ferrotorch-gpu/tests/divergence_indexing_masked_fill_parity_cuda.rs` and
the unit tests in this file pin that boundary.

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
| REQ-1 (where_ forward + backward) | SHIPPED | `Tensor::where_t` in `methods.rs` calls `grad_fns::comparison::where_`. Same-shape CPU uses `WhereBackward`; CUDA or broadcasted value operands build a full-output BoolTensor mask and route through `where_cond_bcast`, preserving CUDA residency and broadcast-gradient reductions. Pinned by `test_where_forward`, `test_where_backward`, `test_where_host_mask_broadcasts_operands`, and GPU public-method probes. |
| REQ-2 (where_bt BoolTensor variant) | SHIPPED | `Tensor::where_bt_t` in `methods.rs` calls `grad_fns::comparison::where_bt`, which delegates to `grad_fns::indexing::where_cond_bcast`. Condition/self/other broadcast by PyTorch rules. Pinned by `where_bt_broadcasts_three_inputs_and_reduces_grads` and `tensor_where_bt_broadcast_cuda_condition_stays_resident_and_reduces_grads`. |
| REQ-3 (device handling + NaN/Inf passthrough) | SHIPPED | CUDA `where_t` uploads only the host mask; CUDA `where_bt_t` keeps condition/value tensors resident; both produce resident gradients. NaN/Inf/denormals pass through because selection returns input elements unmodified. |
| REQ-4 (comparison / logical predicate surface) | SPLIT-OWNERSHIP | This module owns differentiable `where` only. Float and integer `eq/ne/lt/le/gt/ge` plus bool `logical_and` / `logical_or` / `logical_xor` / `logical_not` live in `bool_tensor.rs`; CUDA broadcasted i32/i64 comparisons route through `GpuBackend::compare_broadcast` and stay CUDA-resident. Other value-returning extrema / predicate APIs are owned by their dedicated modules. Open prereq blocker #1293 covers route retargeting. |
