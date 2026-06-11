# Ferrotorch Core Deep Audit

**Repository:** `forecast-bio/ferrotorch`  
**Audited revision:** `24f587d9402d40c1479ef4ad4146f77162b69ab1`  
**Crate:** `ferrotorch-core` (`0.6.2`)  
**Audit started:** 2026-06-10  
**Status:** Module coverage complete (207 findings; remediation not started)

## Scope And Method

This is a module-by-module deep audit of `ferrotorch-core`, separate from the
earlier workspace-wide audit. A module is marked complete only after reviewing
its public contracts, error and panic behavior, shape/index arithmetic, device
semantics, autograd behavior, unsafe code, and relevant tests.

The crate contains 77 production Rust source files and approximately 113,000
production Rust lines. Its test tree contains approximately 60,000 additional
Rust lines across 170 files.

## Coverage Ledger

| Area | Modules | Status |
|---|---|---|
| Tensor and storage invariants | `tensor`, `storage`, `shape`, `creation`, `device`, `display`, `cpu_pool` | Complete |
| Mutation and dispatch | `inplace`, `dispatch`, `gpu_dispatch`, `ops_trait`, `profiler_hook` | Complete |
| Autograd engine | `autograd/*` | Complete |
| Gradient functions | `grad_fns/*` | Complete |
| Core operation families | `ops/*`, `methods`, `linalg`, `fft`, `einsum`, `special` | Complete |
| Dtypes and typed tensors | `dtype`, `dtype_dispatch`, `numeric_cast`, `bool_tensor`, `int_tensor`, `complex_tensor` | Complete |
| Structured and advanced tensors | `sparse`, `nested`, `named_tensor`, `masked`, `stride_tricks`, `einops`, `vmap` | Complete |
| Quantization and pruning | `quantize`, `pruning`, `grad_fns/quantize_grad` | Complete |
| Remaining utilities | `rng`, `signal/*`, `simd_reduce`, `flex_attention`, `meta_propagate` | Complete |
| Tests and public contract coverage | `ferrotorch-core/tests/*`, in-module tests, CI inclusion | Complete |

## Findings

### CORE-001: Safe in-place APIs violate Rust aliasing requirements

- **Severity:** Critical
- **Confidence:** Confirmed
- **Affected code:** `src/tensor.rs:118-120`, `src/tensor.rs:126-128`,
  `src/tensor.rs:1267-1288`, `src/tensor.rs:1305-1352`,
  `src/tensor.rs:1434-1522`, `src/tensor.rs:1922-1929`,
  `src/inplace.rs:75-687`, `src/grad_fns/arithmetic.rs:646-735`

`Tensor` uses shared `Arc<TensorInner>` ownership, but safe public mutation
operations create mutable references behind potentially aliased `Arc`s.
Cloning a tensor and then calling a trailing-underscore operation is sufficient
to violate Rust's reference aliasing requirements. Cross-thread aliases can
also turn this into a data race reachable from safe code.

**Recommendation:** Redesign mutable storage around enforced synchronization or
provable unique ownership. Audit every public mutation and `out=` path under
Miri with cloned and cross-thread aliases.

### CORE-002: CUDA-to-CPU conversion returns incorrect values for unsupported views

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/tensor.rs:855-935`

CUDA view readback materializes only rank-eight-or-lower `f32`/`f64` views. For
other non-contiguous or offset views, the fallback reads the complete backing
buffer and constructs a fresh contiguous CPU tensor with the view shape,
discarding the logical offset and strides.

**Recommendation:** Correctly materialize all supported view types or return an
explicit unsupported-operation error.

### CORE-003: `Tensor::to_dtype` silently severs autograd graphs

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/tensor.rs` (`to_dtype`)

The dtype-conversion path creates fresh storage without attaching a backward
function. Converting a differentiable tensor therefore silently produces a
detached output.

**Recommendation:** Attach a cast backward operation or explicitly reject dtype
conversion while gradient tracking is enabled.

### CORE-004: Safe stride-view constructors permit out-of-bounds layouts

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/tensor.rs:267-318`, `src/tensor.rs` (`data_vec`),
  CUDA strided-copy call path

Safe public stride-view constructors accept arbitrary shapes, strides, and
offsets without proving that every logical index lies inside the backing
storage. CPU access can panic; the corresponding CUDA strided copy has no
source bounds check and can read out of bounds.

**Recommendation:** Validate the complete reachable storage interval at every
safe view-construction boundary. Keep unchecked construction private or unsafe.

### CORE-005: Fallible subregion cloning panics on invalid ranges

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/storage.rs` (`try_clone_subregion`)

The method returns a result but slices with caller-controlled bounds before
returning, converting invalid ranges into panics.

**Recommendation:** Validate range ordering and bounds before slicing.

### CORE-006: `TensorStorage::clone` panics on routine backend failures

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/storage.rs` (`Clone` implementation)

Cloning GPU-backed storage performs a fallible deep clone and panics if the
backend reports an error, making an ordinary trait operation unexpectedly
fallible through panic.

**Recommendation:** Use shared ownership for storage cloning or expose an
explicit fallible deep-clone API.

### CORE-007: Foundational shape arithmetic can overflow silently

- **Severity:** High
- **Confidence:** Strong
- **Affected code:** `src/shape.rs`, `src/tensor.rs`, `src/creation.rs`, and
  operation-specific shape calculations

Shape products and offset calculations frequently use unchecked `usize`
arithmetic. Crafted or extreme shapes can wrap in release builds, causing
incorrect allocation sizes, validation bypasses, or later indexing failures.

**Recommendation:** Centralize checked shape, product, stride, and byte-count
arithmetic and use it at all public construction boundaries.

### CORE-008: `BoolTensor::not` turns normal backend failures into panics

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/bool_tensor.rs:286-300`

Both CPU data access and the GPU kernel result are unwrapped with `expect` in a
public infallible operation, so ordinary backend failures panic.

**Recommendation:** Return `FerrotorchResult<BoolTensor>` consistently with
other fallible tensor operations.

### CORE-009: Fixed-point backward erases tensor errors with `unwrap`

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/autograd/fixed_point.rs`, `src/autograd/graph.rs`

The fixed-point backward implementation unwraps fallible tensor data and
storage construction. The parallel autograd engine also unwraps poisoned
synchronization primitives, converting recoverable execution failures into
process panics.

**Recommendation:** Propagate structured errors through backward execution and
define explicit poison-recovery behavior.

### CORE-010: Quantization observer constructors accept panic-inducing values

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/quantize.rs:638-879`

Zero channel and zero bin counts are accepted by infallible constructors and
later cause division, modulo, subtraction, or indexing panics in `observe`.
Malformed per-channel lengths are silently ignored.

**Recommendation:** Reject invalid observer configurations and make malformed
observations report errors.

### CORE-011: Autograd-aware reshape of a non-contiguous GPU tensor silently moves it to CPU

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/tensor.rs:223-264`, `src/grad_fns/shape.rs:141-156`,
  `src/grad_fns/shape.rs:197-205`, `src/grad_fns/shape.rs:251-274`,
  `src/grad_fns/shape.rs:319-353`

`Tensor::view_reshape` correctly materializes a non-contiguous tensor through
the device-aware `contiguous()` path. Its autograd-aware sibling
`Tensor::view_operation` instead calls `data_vec()`, wraps the result in
`TensorStorage::cpu`, and recurses. Consequently, reshape, flatten, squeeze, or
unsqueeze on a non-contiguous CUDA tensor remains CUDA when gradients are
disabled but silently becomes CPU when gradients are enabled.

The implementation comment on `view_reshape` explicitly identifies this old
CPU-demotion pattern as a previously fixed bug, but the same pattern remains in
`view_operation`. Existing GPU shape conformance tests do not exercise a
non-contiguous, requires-grad input.

**Recommendation:** Materialize through `self.contiguous()?` in
`view_operation` as well. Add device-residency and backward tests for every
shape operation on non-contiguous CUDA views.

### CORE-012: Device transfer of a gradient-tracking leaf severs the source graph

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/tensor.rs:812-1026`, `src/tensor.rs:1044-1077`,
  `tests/conformance_creation.rs:1151-1172`

`Tensor::to` and `to_pinned` attach `ToDeviceBackward` only when
`self.requires_grad() && !self.is_leaf()`. Moving a requires-grad leaf between
CPU and CUDA/XPU therefore constructs a new independent leaf with no grad
function. Backward on the transferred tensor accumulates on that new leaf and
cannot reach the original source leaf.

PyTorch treats a differentiable `.to(other_device)` as a copy operation with a
backward edge, including when the source is a leaf. The GPU conformance test's
comments say a tracking node must exist, but the test asserts only
`requires_grad` and device, so it misses the defect. Transfers to `Meta` also
always rebuild through `from_storage` and sever any existing graph.

**Recommendation:** Attach a transfer backward node whenever gradient tracking
is enabled and the source requires gradients, regardless of leaf status.
Explicitly define or reject autograd semantics for transfers to `Meta`. Test
that gradients reach the original source leaf after every supported transfer.

### CORE-013: Memory-format conversion discards autograd history while retaining `requires_grad`

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/tensor.rs:1625-1799`

When a tensor must be physically rearranged for `ChannelsLast`,
`ChannelsLast3d`, or contiguous format, both GPU and CPU materialization paths
construct a fresh tensor with `grad_fn: None`, `is_leaf: true`, and
`requires_grad` copied from the input. The output therefore appears
differentiable but is disconnected from the input graph.

**Recommendation:** Attach an identity/permutation-aware backward node to
materialized format conversions. Add backward tests for each memory format on
CPU and CUDA, including non-leaf inputs.

### CORE-014: Floating-point `arange` can loop forever

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/creation.rs:74-94`

`arange` repeatedly performs `val += step` until a floating-point comparison
changes. It does not validate finiteness or verify that each addition makes
progress. At sufficiently large magnitudes, a nonzero step smaller than one
ULP leaves `val` unchanged while the loop condition remains true. For example,
an `f32` start whose next representable value is larger than `step`, paired
with a larger or infinite end, never terminates. Non-finite inputs are also not
rejected consistently.

**Recommendation:** Validate finite arguments and direction, compute a bounded
output length with checked arithmetic, allocate once, and generate by index.
Reject any configuration whose length is not representable or whose generated
values cannot progress.

### CORE-015: Safe storage constructors allow contradictory dtype and device metadata

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/gpu_dispatch.rs:147-220`,
  `src/storage.rs:231-253`, `src/tensor.rs:134-168`

`GpuBufferHandle::new` is safe and public and accepts arbitrary erased storage,
length, dtype, and ordinal metadata. `TensorStorage::gpu` is also safe and
generic over `T`, but never verifies that `handle.dtype() == T::dtype()`.
Similarly, `TensorStorage::xpu_from_handle` accepts an ordinal separate from
`handle.ordinal()` and any element type even though the CubeCL handle contract
states that its length and readback are `f32`.

Safe callers can therefore construct tensors whose Rust element type, handle
dtype, erased allocation, storage length, and advertised device disagree.
Many operations select typed backend methods from `T`, while other backend
operations dispatch from the handle tag. Backends often turn the contradiction
into errors, but the invariant is not enforced at the safe construction
boundary and downstream code cannot rely on tensor metadata being coherent.

**Recommendation:** Make raw handle construction unsafe or backend-private.
Provide fallible typed storage constructors that validate dtype, length,
ordinal, and concrete handle compatibility. Derive XPU device metadata from
the handle rather than accepting it separately.

### CORE-016: CPU buffer pool's memory limit ignores allocation capacity

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/cpu_pool.rs:74-247`, `src/storage.rs:531-540`

The pool claims exact-size reuse and a 64 MiB per-thread bound, but keys and
accounts buffers using `Vec::len()` rather than `Vec::capacity()`. The public
`pool_return_cpu` accepts vectors with arbitrarily larger capacity than length,
and ordinary tensor storage can also own such vectors. A short vector backed by
a very large allocation is charged only `len * size_of::<T>()`, so the pool can
retain substantially more than its documented bound. Reuse is likewise
length-exact, not capacity-exact.

**Recommendation:** Account actual allocation capacity and reject or shrink
oversized buffers before caching. Include capacity-heavy and per-thread memory
bound tests.

### CORE-017: Linux CI explicitly excludes all core integration tests

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `.github/workflows/linux-ci.yml:80-102`,
  `ferrotorch-core/tests/*`

The Linux workflow job is named "cargo test (ferrotorch-core lib + tests)", and
the workflow header says core receives library plus integration coverage.
However, the command passes `--lib`; its comment explicitly states that
`--tests` remains excluded because several integration tests already fail on
`main`. Approximately 170 core integration-test files, including conformance,
GPU probes, and targeted divergence regressions, therefore do not gate changes
in normal Linux CI.

**Recommendation:** Separate stable integration suites from environment-gated
GPU/probe suites and make the stable set mandatory. Treat existing failures as
tracked skips with owners and deadlines rather than excluding the entire test
surface.

### CORE-018: Detached aliases can silently mutate tensors saved for backward

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/tensor.rs:1524-1540`, `src/inplace.rs:51-687`,
  `src/inplace.rs:918-929`, gradient functions that save input tensor clones

`Tensor::detach` creates a new leaf with `requires_grad = false` but deliberately
shares the original tensor's storage. Every trailing-underscore operation then
accepts that detached alias because `check_inplace_allowed` inspects only its
local autograd metadata. The mutation changes the values visible through the
original tensor and through clones saved inside backward nodes.

Unlike PyTorch, the tensor/storage model has no shared version counter that can
detect this mutation when backward later reads a saved tensor. A forward such
as `y = x * x`, followed by `x.detach().fill_(new_value)` and `y.backward()`,
therefore computes a gradient from the mutated value instead of rejecting the
invalidated graph. The existing in-place unit test explicitly verifies that a
detached alias is mutable but does not exercise backward through its source.

**Recommendation:** Introduce a version counter shared by all aliases and save
the expected version in backward nodes, or prohibit mutation whenever storage
is aliased by graph-tracking tensors. Add regressions for detached and view
aliases modified between forward and backward.

### CORE-019: Binary in-place operations discard gradients from tracking source operands

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/inplace.rs:163-549`,
  `src/grad_fns/arithmetic.rs:459-483`, `src/grad_fns/arithmetic.rs:1144-1168`,
  `src/grad_fns/arithmetic.rs:1306-1330`

The in-place guard validates only the destination. A non-tracking destination
can therefore be combined in-place with an `other` tensor that requires
gradients. The arithmetic helpers correctly create a result with a backward
node when either operand tracks gradients, but the in-place implementation
extracts only that result's storage and installs it into `self`; `self` remains
a non-tracking leaf. Same-shape fast paths bypass the differentiable helper
entirely and have the same outcome.

Consequently, operations such as `destination.add_(&tracking_source)` produce
correct values but silently lose the edge to `tracking_source`. PyTorch makes
the mutated result require gradients and records the operation.

**Recommendation:** Either implement graph-aware in-place semantics, including
version tracking, or reject binary in-place operations whenever any source
operand requires gradients. Test gradient propagation from each source operand
through every binary trailing-underscore operation.

### CORE-020: `add_out` silently bypasses autograd for tracking inputs

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/arithmetic.rs:620-738`

`check_out_allowed` validates only the output tensor. `add_scaled_out` then
unconditionally computes inside `no_grad` and swaps the resulting storage into
`out`, even when `a` or `b` requires gradients. This returns a numerically
correct but detached result without warning.

PyTorch rejects `out=` arithmetic when any argument requires gradients because
the operation does not support automatic differentiation. The current
implementation's documentation states that `out=` is non-autograd-tracked but
does not turn tracking inputs into an error.

**Recommendation:** Reject `add_out` and `add_scaled_out` when either input
requires gradients or carries a gradient function. Add tests asserting the
error contract for leaf and non-leaf tracking inputs.

### CORE-021: Parallel backward deadlocks when a backward node returns an error

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/autograd/graph.rs:330-463`

When a worker's node processing returns an error, the worker stores the error
and increments `processed`, but it does not decrement the failed node's input
in-degrees or signal a global cancellation condition. Those downstream nodes
can never enter the ready queue. `processed` therefore remains below
`total_nodes`, and every worker eventually waits on the condition variable
forever instead of returning the collected error.

This affects any graph large enough to take the parallel path when a gradient
function, hook, device operation, or shape check fails.

**Recommendation:** Add a shared cancellation/error flag that wakes all
workers and terminates scheduling immediately on the first failure. Add a
timeout-backed regression with an intentionally failing gradient node.

### CORE-022: Parallel and sequential backward use different implicit-gradient shapes

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/autograd/graph.rs:47-78`, `src/autograd/graph.rs:242-269`

Sequential backward creates an implicit gradient with the same shape as the
root, including single-element shapes such as `[1]` and `[1, 1]`. Parallel
backward instead always constructs one CPU element with scalar shape `[]`
before moving it to the root device. A gradient function that requires its
upstream gradient to match a single-element non-scalar root therefore behaves
differently or errors only on the parallel path.

**Recommendation:** Share one seed-construction helper between both engines
and add parallel regressions for all accepted single-element root shapes.

### CORE-023: Saved-tensor hooks are not connected to production autograd nodes

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/autograd/saved_tensors.rs`, all production
  `GradFn` implementations

The crate exposes `saved_tensors_hooks`, `pack_saved_tensor`, and
`unpack_saved_tensor` as a memory-offloading feature, but repository-wide
production search finds no call to either pack or unpack helper outside their
own module and tests. Gradient functions continue storing ordinary cloned
tensors directly. Installing hooks therefore has no effect on tensors saved by
real forward operations and cannot provide the documented GPU-memory savings.

**Recommendation:** Introduce a saved-tensor wrapper used by every backward
node that needs forward values, and route construction/access through pack and
unpack hooks. Add an integration test proving a normal differentiable
operation invokes both hooks.

### CORE-024: Saved-tensor hooks remain installed after panic unwind

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/autograd/saved_tensors.rs:72-119`

`saved_tensors_hooks` restores the previous thread-local hook pair only after
the user closure returns normally. If the closure panics, unwind skips the
restoration assignments and leaves the hooks active on that thread. Later,
unrelated autograd work can unexpectedly invoke closures whose intended scope
has ended.

**Recommendation:** Restore prior hooks from an RAII guard, matching the
crate's `no_grad`, autocast, anomaly, and inference-mode guards. Add panic
safety tests for both dtypes and nested scopes.

### CORE-025: Gradient hooks execute while holding their registration mutex

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/autograd/hooks.rs:141-173`, `src/tensor.rs:485-527`

Both gradient and post-accumulate callbacks run while the tensor's hook-storage
mutex is locked. A callback can capture the tensor, or receives it directly in
the post-accumulate case, and call `register_hook`, `remove_hook`, or another
operation that tries to lock the same storage. Because the mutex is not
reentrant, this deadlocks the backward pass.

**Recommendation:** Snapshot callbacks or otherwise release the storage lock
before invoking user code. Define how registrations/removals made during
callback execution affect the current pass and test reentrant mutation.

### CORE-026: Higher-order `grad` is not device preserving

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/autograd/higher_order.rs:70-248`,
  `src/autograd/higher_order.rs:283-395`

`grad` always creates its seed on CPU, even when `outputs` is on CUDA/XPU/MPS.
When `create_graph=false`, gradient fan-in accumulation also reads both inputs
to host vectors and constructs the combined gradient in CPU storage. The next
device-resident backward node can consequently receive an upstream gradient on
the wrong device. `jacobian`, `hessian`, and element extraction likewise use
CPU-only data access and always build CPU results.

**Recommendation:** Construct seeds on the output device and use the same
device-aware accumulation paths as the main backward engine. Either preserve
device placement throughout Jacobian/Hessian helpers or reject unsupported
devices explicitly.

### CORE-027: `create_graph=true` can produce disconnected fake gradient leaves

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/autograd/higher_order.rs:193-207` and gradient
  functions that compute backward values through raw data operations

When a backward implementation returns a tensor without autograd metadata,
`grad(..., create_graph=true)` calls `requires_grad_(true)` on that numerical
result. This makes the gradient look differentiable, but it is a new leaf with
no edge to the original forward inputs. Differentiating it again cannot recover
the derivative of the backward computation and yields missing or incorrect
higher-order gradients.

**Recommendation:** Require backward implementations used with
`create_graph=true` to construct genuinely differentiable operations. Return a
clear unsupported-higher-order error where that is not implemented instead of
marking detached values as tracking leaves.

### CORE-028: CPU stochastic checkpointing recomputes different forward values

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/autograd/checkpoint.rs:44-72`,
  `src/autograd/checkpoint.rs:150-184`

Checkpointing preserves only CUDA RNG state. The module documentation confirms
that CPU dropout uses a time-seeded generator and is not restored during
recomputation. A checkpointed stochastic CPU function can therefore use a
different mask during backward than it used during forward, producing
incorrect gradients. GPU RNG save and restore errors are also converted to
`None` or ignored, causing the same silent correctness failure on backend
errors.

**Recommendation:** Capture and restore every relevant device RNG state and
propagate save/restore failures. Until CPU state can be preserved, reject
checkpointing CPU stochastic operations or require an explicit opt-in to
nondeterministic recomputation.

### CORE-029: Inference-mode tensors are not actually inference-only

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/autograd/no_grad.rs:110-170`,
  `src/tensor.rs:324-364`, `src/tensor.rs:1543-1559`

The inference-mode contract says tensors created inside the scope cannot later
participate in autograd. In practice, inference mode only makes
`from_operation` fall back to a plain tensor. No inference-only marker is
stored, and `requires_grad_(true)` can be called after leaving the scope.
Subsequent operations then track gradients normally.

**Recommendation:** Store and enforce an inference-tensor flag across aliases,
or narrow the documented/API contract to the behavior actually implemented.
Add tests that attempt to enable gradients on tensors created in inference
mode.

### CORE-030: Anomaly detection is not integrated with the autograd engine

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/autograd/anomaly.rs`, `src/autograd/graph.rs`,
  `src/tensor.rs`, all production gradient nodes

The anomaly module exposes mode control, forward-backtrace capture, and
gradient validation, but production search finds no call to
`ForwardBacktrace::capture_if_enabled` or `check_gradient_anomaly` outside the
anomaly module and its tests. Tensors and gradient nodes do not store a forward
backtrace, and neither backward engine checks gradients for NaN/Inf.

Enabling `detect_anomaly` therefore has no effect on ordinary model execution
despite the documented PyTorch-style contract.

**Recommendation:** Store optional forward provenance on differentiable nodes
and invoke anomaly checks after each backward result in both engines. Add
end-to-end tests where a normal operation produces a non-finite gradient.

### CORE-031: Forward-mode AD demotes many GPU tangents to CPU

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/autograd/forward_ad.rs:62-82`,
  `src/autograd/forward_ad.rs:169-328`, `src/autograd/forward_ad.rs:421-469`

`DualTensor::constant` always creates its zero tangent in CPU storage. The
ReLU, sigmoid, tanh, exp, log, sin, and cos forward rules read values to host
vectors and also construct CPU tangents regardless of the primal device.
`jacfwd` seeds every basis tangent on CPU and always returns a CPU Jacobian.

A CUDA primal can consequently produce a dual value whose primal is CUDA but
whose tangent is CPU; the next dual arithmetic operation then encounters a
device mismatch or silently leaves the intended execution device.

**Recommendation:** Allocate and compute tangents on the primal's device, with
device-native kernels or differentiable tensor operations. Enforce matching
device as well as shape in `DualTensor::new`.

### CORE-032: `jacfwd` panics for an empty one-dimensional input

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/autograd/forward_ad.rs:421-469`

Shape `[0]` passes the one-dimensional input check. The basis-vector loop then
produces no columns, after which `jacfwd` indexes `columns[0]` to determine the
output size and panics.

**Recommendation:** Define the empty-input Jacobian contract and return a
correctly shaped empty tensor or a structured invalid-argument error.

### CORE-033: `gradcheck` can report success for invalid or non-finite comparisons

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/autograd/gradcheck.rs:54-188`

User-supplied `eps`, `atol`, and `rtol` are not validated for positivity or
finiteness. With `eps = 0`, the numerical derivative becomes `0 / 0 = NaN`.
The subsequent `diff > tolerance` comparison is false for NaN, so the check can
return `Ok(true)`. The same comparison logic also lets non-finite analytical
or numerical gradients evade mismatch detection.

**Recommendation:** Reject non-finite or invalid tolerances and explicitly
fail on any non-finite analytical derivative, numerical derivative,
difference, or tolerance.

### CORE-034: `gradcheck` is stateful and uses previously accumulated gradients

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/autograd/gradcheck.rs:86-111`

Analytical gradients are obtained by calling `output.backward()` on the
caller's input tensors. Backward accumulates into an existing leaf gradient,
and `gradcheck` neither clears it nor uses the non-accumulating `grad` API.
Calling `gradcheck` twice on the same inputs, or calling it after any earlier
backward pass, compares accumulated gradients against a single numerical
derivative and can fail spuriously. It also mutates caller-visible gradient
state as a side effect.

**Recommendation:** Compute analytical values with a non-accumulating gradient
API on fresh inputs, or snapshot/restore gradient state explicitly. Add
repeatability tests.

### CORE-035: Checkpoint recomputation does not force gradient tracking on

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/autograd/checkpoint.rs:216-358`,
  `src/autograd/no_grad.rs`

Checkpoint backward claims to rerun the forward function with gradient
tracking, but it restores only autocast/RNG state and never enters
`enable_grad`. If a user invokes backward inside a `no_grad` scope, the
checkpoint recomputation remains untracked. Its weighted scalar has no graph
back to the recomputed input, so checkpoint backward returns no input
gradient. Ordinary non-checkpointed backward does not require forward graph
construction during backward and is not affected in the same way.

**Recommendation:** Wrap checkpoint recomputation and its weighted backward
construction in `enable_grad`, independent of the caller's ambient grad mode.
Test checkpoint backward from inside `no_grad`.

### CORE-036: Fixed-point solving accepts incompatible iterates and silent non-convergence

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/autograd/fixed_point.rs:91-139`

The fixed-point loop never verifies that `f(x, params)` has the same shape or
device as `x`. Its convergence norm zips the two host vectors, silently
ignoring unmatched trailing elements. An incompatible iterate can therefore be
declared converged. If no iterate meets the tolerance, including invalid
negative or NaN tolerances, the function silently returns the last estimate as
though it were a fixed point. When parameters require gradients, the returned
fixed point is additionally rebuilt in CPU storage regardless of its original
device.

**Recommendation:** Validate iteration shape/device, validate solver
configuration, and return a non-convergence error with residual information.
Preserve device placement when attaching the implicit backward node.

### CORE-037: Backward on a gradient-tracking leaf silently does nothing

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/autograd/graph.rs:37-218`,
  `src/autograd/graph.rs:236-469`

The engines seed the root gradient but accumulate onto leaves only while
processing gradients returned for the inputs of a node's `grad_fn`. A leaf
root has no `grad_fn`, so its seed is removed from the temporary gradient map
and discarded. Calling `backward()` or `backward_with_gradient()` directly on
a scalar leaf with `requires_grad=true` returns success but leaves `.grad()`
unset. Calling it on a non-tracking leaf also returns success rather than
reporting that the tensor does not require gradients.

**Recommendation:** Validate that the root participates in autograd and
directly accumulate the seed when the root is a tracking leaf. Add scalar,
single-element shaped, and external-gradient leaf-root tests.

### CORE-038: External backward gradients are not validated for device compatibility

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/autograd/graph.rs:47-61`,
  `src/autograd/graph.rs:242-255`

Both backward engines validate only the external gradient's shape before using
it as the root seed. A CPU gradient can be supplied for a CUDA root, or vice
versa, and the mismatch is discovered later inside an arbitrary gradient
function or can trigger an unintended host/device path.

**Recommendation:** Require external gradients to match the root's device at
the engine boundary and return `DeviceMismatch` immediately.

### CORE-039: Differentiable FFT wrappers panic before fallible input validation

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/fft.rs:487-559`,
  `src/grad_fns/fft.rs:1627-1660`

Several public differentiable FFT wrappers inspect shape entries before calling
the corresponding fallible forward FFT implementation. `rfft_differentiable`
unwraps the last dimension of a scalar input, `irfft_differentiable` and
`hfft_differentiable` index the second-to-last dimension of undersized inputs,
and `ihfft_differentiable` unwraps the last dimension of a scalar input. The
underlying `rfft` and `irfft` implementations explicitly return
`InvalidArgument` for these malformed ranks, but the differentiable wrappers
panic first. The explicit-dimension wrappers contain related unchecked shape
indexing and subtraction.

The conformance fixtures exercise valid numerical cases and therefore report
full FFT parity without detecting these public `Result` APIs aborting on
invalid inputs.

**Recommendation:** Call and validate the forward operation before deriving
backward metadata, or share a checked shape/axis validation helper. Add scalar,
rank-one complex, empty-axis, and out-of-range-axis tests for every
differentiable FFT entry point.

### CORE-040: Scalar `scatter_reduce` bypasses validation and can panic on empty source

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/indexing.rs:2437-2483`

The scalar-input branch returns before the normal `dim`, index-rank, index
length, source-length, and bounds checks. It therefore accepts arbitrary
dimensions and contradictory `index_shape` metadata. With a non-empty index
and an empty source tensor, `src_data.len() - 1` underflows before indexing and
the public fallible API panics. The same branch also silently repeats the final
source value when the index contains more entries than the source, rather than
rejecting the incompatible arguments.

**Recommendation:** Apply the common validation path before the scalar
special-case computation. Require scalar-compatible index/source shapes and
return a structured shape or argument error instead of clamping source
indices.

### CORE-041: Differentiable fake quantization silently moves CUDA tensors and gradients to CPU

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/quantize_grad.rs:217-285`,
  `src/grad_fns/quantize_grad.rs:421-627`,
  `src/grad_fns/quantize_grad.rs:661-779`

Both per-tensor and per-channel fake-quantization forwards read input data
through `data_vec` and unconditionally construct their outputs with
`TensorStorage::cpu`. They neither preserve a CUDA input's device nor return
`NotImplementedOnCuda`. Their backward nodes likewise unconditionally return
CPU gradients even when the saved input and incoming gradient are CUDA
tensors. This breaks device-preservation and can make a QAT graph fail only
after a later operation encounters the unexpected CPU tensor.

The quantization conformance suite exercises CPU values and gradients but has
no CUDA/device-preservation coverage for these APIs.

**Recommendation:** Either implement device-preserving fake-quantization
kernels and gradients or reject CUDA inputs at the public boundary. Add
forward and backward device assertions to the conformance suite.

### CORE-042: Scalar cumulative operations silently demote CUDA tensors to CPU

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/cumulative.rs:104-221`,
  `src/grad_fns/cumulative.rs:354-628`,
  `src/grad_fns/cumulative.rs:712-752`,
  `src/ops/cumulative.rs:79-467`

The differentiable wrappers special-case scalar inputs before calling the
device-aware forward kernels. The shared scalar identity helper and scalar
`cummax`/`cummin` helper materialize `TensorStorage::cpu` directly. As a
result, valid CUDA scalar calls to `cumsum`, `cumprod`, `cummax`, `cummin`,
and `logcumsumexp` return CPU outputs, while non-scalar CUDA f32/f64 inputs use
CUDA kernels and preserve device placement. The identity backward also passes
through whatever root-gradient device it receives, leaving the saved CUDA
input and returned CPU output inconsistent.

**Recommendation:** Preserve the input device in scalar identity paths, or
route scalars through a device-aware copy helper. Add CUDA scalar forward and
backward tests for all five cumulative operations.

### CORE-043: `where_` relies on debug assertions and silently permits cross-device inputs

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/comparison.rs:34-153`,
  `src/methods.rs:1250-1291`

The public host-mask `where_` entry point uses `debug_assert_eq!` as its only
length validation. In release builds, extra condition or `y` elements are
silently ignored by nested `zip` iteration, and equal-numel tensors with
different shapes are accepted under `x`'s shape. It also never requires `x`
and `y` to share a device: `y` is downloaded through `data_vec`, and the
result is uploaded to `x`'s device. `where_bt` adds some shape checks but
still delegates to the same cross-device behavior and validates only the
condition's element count, not its shape.

**Recommendation:** Replace debug assertions with structured validation of
condition shape/length, operand shapes or supported broadcasting, and operand
devices. Reject mismatches before materializing data.

### CORE-044: `rrelu(training=true)` silently executes inference behavior

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/activation.rs:2650-2702`

The public `rrelu` API accepts a training-mode boolean but names it
`_training`, never branches on it, and always applies the deterministic mean
slope `(lower + upper) / 2`. PyTorch's training path draws and saves an
independent random slope for each negative element. A caller requesting
training behavior therefore receives a valid-looking tensor and backward node
with materially different stochastic behavior, rather than an unsupported
operation error.

The source comment acknowledges this divergence and notes that the parity
sweep exercises only the default `training=false` path.

**Recommendation:** Implement the RNG-stateful training path or reject
`training=true`. Add forward, backward, repeatability, and RNG-state tests for
both modes.

### CORE-045: Several activation APIs fail on CUDA only when autograd is enabled

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/activation.rs:1788-2255`

`hardtanh`/`relu6`, `hardsigmoid`, `hardswish`, `selu`, `softsign`, and
`prelu` compute their forward through `unary_map`, which permits CUDA inputs
via a host round trip and restores the output device. When the input requires
gradients, however, each wrapper immediately calls `output.data()` and rebuilds
CPU storage to attach its backward node. `data()` is unavailable for a CUDA
tensor, so the same otherwise-valid forward fails solely because
`requires_grad` is enabled. Their backward implementations also directly call
CPU-only `data()` without a CUDA implementation or explicit
`NotImplementedOnCuda` boundary.

`leaky_relu` directly beside these functions has already been corrected to
consume and preserve `unary_map`'s storage, demonstrating the intended local
pattern.

**Recommendation:** Apply the storage-preserving `leaky_relu` pattern and
provide device-aware backwards, or reject CUDA consistently before doing any
work. Add paired CUDA tests with grad tracking enabled and disabled for every
activation.

### CORE-046: `var_dim` and `std_dim` silently sever autograd graphs

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/reduction.rs:1920-2030`

The public dimension-reduction variance and standard-deviation APIs always
construct outputs with `requires_grad=false` and never attach a backward node,
even when gradient tracking is enabled and the input requires gradients.
Their comments explicitly describe them as “forward-only,” but callers receive
ordinary successful tensors rather than an unsupported-autograd error.
PyTorch supports gradients for these operations, so any training graph using a
dimension-keyed variance or standard deviation silently stops at that point.

Both functions are listed in the public surface inventory but excluded from
surface conformance coverage.

**Recommendation:** Implement the dimension-aware variance/std VJPs, or reject
tracking inputs until those VJPs exist. Add graph-connectivity and numerical
gradient tests across `keepdim`, correction, empty, and singleton slices.

### CORE-047: `vector_norm_differentiable` silently detaches all norms except `ord=2`

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/linalg.rs:4080-4101`

`vector_norm_differentiable` computes every supported norm order under
`no_grad`, but attaches `NormBackward` only when `ord == 2.0`. For any other
order, a gradient-tracking input produces a successful detached output. The
function name and return type do not signal this conditional loss of autograd,
and PyTorch implements backward formulas for the other norm orders.

**Recommendation:** Implement the remaining norm VJPs or return an explicit
unsupported-autograd error whenever a tracking input requests an unsupported
order. Add graph-connectivity tests for every accepted `ord`, including
zero/infinite norms and non-smooth inputs.

### CORE-048: Advanced indexing APIs silently demote CUDA and accept mixed-device operands

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/indexing.rs:2437-2644`,
  `src/grad_fns/indexing.rs:3055-3495`,
  `src/grad_fns/indexing.rs:3618-4038`

`scatter_reduce`, `index_add`, `index_copy`, `take`, and `put` have no CUDA
boundary or device-equality validation. They read CUDA operands through
`data_vec`, perform the operation on the host, and unconditionally construct
CPU outputs and CPU gradients. Binary forms also accept operands from
different devices and silently combine their downloaded values. The
non-all-CUDA fallback of `masked_scatter` has the same behavior whenever its
mask is host-accessible.

This differs from nearby indexing operations that either use an on-device
kernel or explicitly return `NotImplementedOnCuda`. Existing divergence tests
focus on numerical/index semantics on CPU and do not assert output or gradient
device placement.

**Recommendation:** Enforce same-device operands at entry, preserve device
placement with kernels or explicit upload, and reject unsupported CUDA cases
instead of silently downloading. Add paired CPU/CUDA and mixed-device tests
for every advanced-indexing surface.

### CORE-049: New transcendental forwards preserve CUDA but their backwards do not

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/transcendental.rs:610-1595`

The unary transcendental family implemented through `unary_map` preserves a
CUDA input's device in forward, but most corresponding backward nodes directly
call CPU-only `data()` on the saved input/output and incoming gradient.
Affected differentiable operations include `tan`, `asin`, `acos`, `atan`,
`sinh`, `cosh`, `asinh`, `acosh`, `atanh`, `exp2`, `expm1`, `log2`, `log10`,
`log1p`, `frac`, and `sinc`. Their forward succeeds on CUDA, then backward
fails with GPU data-access errors. The rounding/sign family instead uses
`zeros_like_tensor`, which unconditionally creates CPU storage, silently
returning a CPU gradient for a CUDA input.

**Recommendation:** Implement device-aware backward formulas or reject CUDA
before a forward graph is created. Make `zeros_like_tensor` preserve the saved
input's device. Test every CUDA forward together with backward and gradient
device assertions.

### CORE-050: Core linalg wrappers panic or silently compute truncated results for invalid shapes

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/linalg.rs:831-910`,
  `src/grad_fns/linalg.rs:1033-1089`,
  `src/grad_fns/linalg.rs:1268-1379`,
  `src/grad_fns/linalg.rs:1383-1506`,
  `src/grad_fns/linalg.rs:1587-1611`

Several public, fallible linalg entry points use rank-dependent shape indexes
or raw data indexes before validating their documented contracts:

- `mm_differentiable`, `mm_bt_differentiable`, `linear_fused`, and
  `mv_differentiable` index `shape()[0]`/`shape()[1]` before checking input
  rank, so scalar or vector inputs can panic instead of returning `Err`.
- `mv_differentiable` never validates the vector length and indexes
  `x_data[p]`, so a short vector panics while a long vector is silently
  partially ignored.
- `dot_differentiable` neither requires 1-D inputs nor equal lengths. Its CPU
  path uses `zip`, silently truncating to the shorter operand.
- `linear_fused` does not validate weight inner dimensions or bias length; the
  raw multiply or bias loop can panic or ignore surplus data.
- The direct CUDA 2-D matrix-multiply paths dispatch without checking inner
  dimension compatibility, unlike the CPU `mm_differentiable` path.

These functions back public `Tensor` methods and return `FerrotorchResult`, so
ordinary invalid user input should not unwind the process or produce a
plausible but incorrect result.

**Recommendation:** Centralize and call strict rank, shape, and same-device
validators before every CPU/GPU dispatch. Add negative tests for all rank and
dimension mismatches, asserting an error and no panic on both devices.

### CORE-051: Global `amin` and `amax` knowingly return false identities for empty tensors

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/reduction.rs:575-665`,
  `tests/conformance_reduction.rs:772-813`

The CPU global `amin` and `amax` implementations fold from positive and
negative infinity. Empty tensors therefore successfully return `+inf` and
`-inf`, although these reductions have no identity and PyTorch raises an
error. The conformance test explicitly documents the divergence but accepts
either the correct error or the incorrect infinity sentinel, allowing the
known behavior to remain green.

The CUDA path similarly sends `numel() == 0` to the backend without an
entry-point check, leaving empty-input behavior backend-dependent.

**Recommendation:** Reject empty inputs before dispatch on every device, and
change the conformance test to require `Err` rather than accepting the
sentinel divergence.

### CORE-052: Dimension-keyed value reductions panic on zero-length reduced slices

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/reduction.rs:1573-1643`,
  `src/grad_fns/reduction.rs:2393-2460`,
  `src/grad_fns/reduction.rs:2748-2808`,
  `src/grad_fns/reduction.rs:2911-3013`

The shared implementations for `argmax_dim`/`argmin_dim`,
`amin_dim`/`amax_dim`, `max_with_dim`/`min_with_dim`, and
`median_with_dim`/`nanmedian_with_dim` do not reject a zero-length reduced
dimension. For shapes such as `[2, 0, 3]` reduced along dimension 1, their
per-slice loops index the first nonexistent element. The median helper also
underflows `dim_size - 1` before indexing an empty order vector.

All affected public APIs return `FerrotorchResult`, but these normal invalid
reductions panic instead of reporting that a value or index cannot be selected
from an empty slice.

**Recommendation:** Validate `dim_size > 0` whenever the reduction requires a
selected value/index, while preserving valid empty outputs caused only by
non-reduced zero dimensions. Add tests that distinguish those two cases.

### CORE-053: Broadcast arithmetic panics for tensors with more than 16 dimensions

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/arithmetic.rs:178-335`,
  `src/grad_fns/arithmetic.rs:2125-2255`,
  `src/grad_fns/arithmetic.rs:2429-2547`,
  `src/grad_fns/arithmetic.rs:2762-2916`,
  `src/grad_fns/arithmetic.rs:3116-3208`,
  `src/grad_fns/arithmetic.rs:3431-3524`

The broadcast walkers used by `remainder`, `fmod`, `floor_divide`, `addcmul`,
and `addcdiv`, plus the shared CPU broadcast-gradient reducer, store output
coordinates in a fixed `[usize; 16]`. The tensor constructors and broadcasting
APIs impose no 16-dimensional rank limit. Any non-empty result with rank 17 or
higher writes or reads beyond that fixed array and panics.

The limit is neither validated nor documented, and the affected APIs return
`FerrotorchResult`.

**Recommendation:** Use a dynamically sized coordinate vector or a shared
arbitrary-rank iterator. If a rank ceiling is required, validate it explicitly
and return an error before iteration. Add forward and backward tests at ranks
16, 17, and higher.

### CORE-054: `repeat` and `tile` reject valid zero-repeat requests

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/shape.rs:982-1045`

`repeat` documents that a repeat count of zero should collapse the
corresponding axis to size zero, matching PyTorch. Its zero branch instead
calls `reshape` on the existing non-empty tensor with a zero-sized shape.
`reshape` correctly requires the element count to remain unchanged, so
`repeat(non_empty, [..., 0, ...])` returns a shape-mismatch error rather than
an empty repeated tensor. `tile` delegates to the same implementation.

**Recommendation:** Allocate or construct a genuine empty tensor with the
computed output shape and attach a backward node that returns a zero gradient
of the input shape. Add zero-repeat tests for each axis, scalars, empty inputs,
and gradient-tracking inputs.

### CORE-055: `cat` rejects CPU views and reads incorrect storage for CUDA views

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/shape.rs:1413-1472`,
  `src/grad_fns/shape.rs:1656-1798`,
  `src/gpu_dispatch.rs:3215-3241`,
  `ferrotorch-gpu/src/kernels.rs:16036-16110`

The CPU `cat` path calls `data()` on each input, which explicitly rejects
non-contiguous tensors. Valid inputs such as transposes therefore fail instead
of concatenating their logical values.

The CUDA path is more dangerous: it passes each raw underlying GPU handle and
logical `numel` to `strided_cat`, but supplies no source strides or storage
offset. The backend kernel explicitly reads contiguous `input[i]` from the
start of the underlying allocation. A transposed, narrowed, permuted, or
offset CUDA view can therefore concatenate the wrong values while returning a
successful result. `CatBackward` repeats the same raw-handle assumption for a
non-contiguous or offset upstream gradient. `cat` also does not validate that
all inputs share the first tensor's device before dispatch.

**Recommendation:** Materialize every input whose logical view is not an
offset-zero packed buffer, or extend the backend kernel to receive source
shape, strides, and offset. Validate all devices up front. Add CPU/CUDA tests
using transpose, narrow, permute, offset views, and mixed-device inputs.

### CORE-056: The public `vmap` family detaches autograd and rejects non-CPU inputs

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/vmap.rs:38-159`, `src/vmap.rs:182-508`,
  `src/lib.rs:233`, `.design/ferrotorch-core/vmap.md:18-53`,
  `tests/conformance_autograd.rs:1476-1731`

The public `select` helper copies input values through CPU-only `data()` and
constructs its result with `requires_grad = false`. The public `stack` helper
likewise copies every input through `data()` into a new detached CPU tensor.
Consequently:

- `select` rejects CUDA and non-contiguous tensors and always severs an input
  autograd graph.
- `stack` rejects CUDA and non-contiguous tensors, does not validate devices,
  and always returns a detached CPU result.
- Every `vmap`, `vmap2`, `vmap3`, `vmap_many`, and `vmap_multi_output` call is
  built from those helpers, so it cannot propagate gradients to batched
  inputs and cannot operate on CUDA or valid non-contiguous inputs.

The design document states that the APIs mirror `torch.vmap` and calls the
loop implementation "correct but not fused", while the conformance suite only
checks contiguous CPU forward values. It has no `requires_grad`, backward,
CUDA, non-contiguous, or output-device assertions for this API family.

**Recommendation:** Implement `select` and `stack` as differentiable,
device-preserving operations over logical tensor views, then build the loop
transform from those operations. Until then, document the API as a detached
CPU-only mapping helper rather than `torch.vmap` parity. Add forward/backward,
device, view, and captured-parameter tests for every public variant.

### CORE-057: `per_sample_grad` silently reconstructs parameters on CPU

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/vmap.rs:532-585`,
  `tests/conformance_autograd.rs:1674-1701`

For each sample, `per_sample_grad` downloads or copies the parameter with
`data_vec()` and constructs the differentiable leaf with
`TensorStorage::cpu(...)`, regardless of the original parameter's device. A
CUDA parameter is therefore silently replaced with a CPU parameter before
calling the user-supplied loss function. This can produce a device-mismatch
error, cause an unintended CPU computation, or return CPU gradients for a
nominally CUDA workflow. CUDA inputs fail even earlier because the function
uses the CPU-only `select` helper described in CORE-056.

The sole conformance test uses contiguous CPU inputs and parameters, so it
does not assert parameter device preservation or mixed-device rejection.

**Recommendation:** Clone each isolated leaf on the parameter's original
device, preserve input slices on their original device, explicitly validate
the input/parameter device contract, and stack gradients without moving them.
Add CPU/CUDA parity and output-device tests.

### CORE-058: Overlapping `as_strided` views return mathematically wrong gradients

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/stride_tricks.rs:273-334`,
  `src/stride_tricks.rs:367-384`, `src/stride_tricks.rs:700-724`

`AsStridedBackward` implements its VJP with `as_strided_scatter`, whose CPU and
CUDA paths overwrite each destination position. When multiple output
positions alias the same input element, autograd must sum all corresponding
upstream gradients. The current implementation instead keeps only the final
write.

The in-module regression test explicitly asserts this incorrect
"last-write-wins" result and claims it matches PyTorch. For a length-five
input viewed as sliding windows of shape `[3, 3]` with strides `[1, 1]`, the
gradient of `view.sum()` should be `[1, 2, 3, 2, 1]`; the test requires
`[1, 1, 1, 1, 1]`.

**Recommendation:** Implement overlap-aware accumulation in
`AsStridedBackward`, including zero-stride and negative-stride layouts, and
replace the regression expectation with PyTorch-derived multiplicity
gradients.

### CORE-059: `AsStridedBackward` fails on CUDA and nonzero-offset input views

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/stride_tricks.rs:209-239`,
  `src/stride_tricks.rs:367-384`

The backward node creates its gradient base with `creation::zeros`, which is
always CPU-backed, then scatters the incoming gradient into it. A CUDA
`grad_output` therefore fails the scatter's device check instead of producing
a CUDA gradient.

The saved `storage_offset` is also an absolute offset into the original
backing storage, but backward allocates only a fresh contiguous tensor of
`input.shape()`. If `as_strided` is called on an input view with a nonzero
offset, that absolute offset is validated against the much smaller fresh
gradient buffer and can fail even for a valid forward. More generally, the
backward does not account for the saved input's own logical view geometry.

**Recommendation:** Allocate the gradient on the input device and implement
the full view-geometry-aware `as_strided` backward rather than scattering an
absolute backing-storage offset into a fresh logical-shape buffer. Add CUDA,
narrowed-input, transposed-input, and chained-view backward tests.

### CORE-060: `as_strided_copy` and `as_strided_scatter` silently detach outputs

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/stride_tricks.rs:244-334`,
  `src/stride_tricks.rs:401-488`, `tests/conformance_shape.rs:1773-1904`

Both public operations mirror differentiable PyTorch operations, but every
successful CPU and CUDA return path constructs a fresh tensor with
`requires_grad = false` and no backward node. `as_strided_copy` first creates
a differentiable `as_strided` view, then discards that graph while
materializing it. `as_strided_scatter` similarly discards gradients with
respect to both the base and source tensors.

The conformance suite checks only forward values and contiguity for these
operations, never `requires_grad` or backward behavior.

**Recommendation:** Attach correct backward nodes to both operations, with
device-preserving implementations, and add gradients for base/source inputs
matching PyTorch.

### CORE-061: All public einops transforms can silently sever autograd graphs

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/einops.rs:469-898`,
  `src/tensor.rs:175-220`, `tests/conformance_einops.rs:14-21`

`rearrange`, `repeat`, and `reduce` are exposed as production tensor
transformations but do not provide a consistent differentiable contract:

- Identity-order `rearrange` directly calls `view_reshape`, which creates a
  detached leaf.
- The general rearrange path begins with the same detached `view_reshape`;
  its CPU fallback also constructs a fresh detached tensor.
- `repeat` always constructs a fresh detached tensor.
- `reduce` preserves a graph only when its narrow fast-path decomposition
  succeeds; every fallback constructs a fresh detached tensor.

Thus gradient behavior depends on the pattern, layout, and whether an internal
optimization happens to succeed. The conformance test header explicitly
excludes backward coverage because these APIs produce `requires_grad=false`
outputs by construction.

**Recommendation:** Build the APIs entirely from differentiable tensor
operations or attach dedicated backward nodes. Make fallback paths preserve
the same graph/device semantics as fast paths, and add backward parity for
every pattern class.

### CORE-062: Reordered-axis einops `repeat` and `reduce` use coordinates in the wrong order

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/einops.rs:600-659`, `src/einops.rs:836-898`

The `repeat` loop collects source coordinates while walking axes in *right*
pattern order, then interprets that coordinate vector using the *left* shape.
A valid pattern that both reorders existing axes and adds a new axis, such as
`"a b -> b a c"`, therefore reads wrong elements and can index past the input
buffer.

The fallback `reduce` loop has the inverse mismatch: it collects kept
coordinates in left order, then interprets them using the right-order output
shape. This fallback is specifically selected when kept axes are reordered,
so a valid pattern such as `"a b c -> c a"` can write to the wrong output
position or panic.

Existing fixtures cover replication and reduction without these reorder
combinations.

**Recommendation:** Build coordinates by axis name and explicitly reorder
them into the target shape's axis order before flattening. Add exhaustive
small-shape pattern tests combining split, merge, reorder, repeat, and reduce.

### CORE-063: `as_strided` bounds validation can overflow and approve invalid layouts

- **Severity:** High
- **Confidence:** Strong
- **Affected code:** `src/stride_tricks.rs:111-188`

The bounds checker converts caller-controlled `usize` dimensions and offsets
to `i64`, then multiplies and accumulates stride extents with unchecked signed
arithmetic. Large dimensions can wrap during conversion or extent
calculation. In release builds, a wrapped extent can make an invalid layout
appear in-bounds; the resulting safe `as_strided` view can later drive
out-of-bounds CPU indexing or a CUDA strided-copy kernel.

This is a concrete validation-bypass instance of the crate-wide unchecked
shape arithmetic noted in CORE-007, at a boundary whose documentation claims
bounds are always validated.

**Recommendation:** Compute extents with checked wide arithmetic, reject
values that cannot be represented, and prove the final inclusive range before
constructing a view. Add adversarial near-`usize::MAX` tests in release mode.

### CORE-064: Masked tensor materialization and reductions are not differentiable

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/masked.rs:148-160`, `src/masked.rs:217-440`,
  `tests/conformance_masked.rs`

`MaskedTensor` retains an ordinary `Tensor<T>` as its data field, including
that tensor's `requires_grad` state, but every value-producing API constructs
a detached result:

- `filled`/`to_tensor` copy values into a new `requires_grad = false` tensor.
- `masked_sum`, `masked_mean`, `masked_min`, and `masked_max` return fresh
  detached CPU or GPU tensors on every path.

These operations mirror differentiable masked operations and have direct
derivatives with respect to valid data entries. Silently returning a detached
value makes a masked loss stop training the underlying data. The conformance
suite checks only forward values and contains no `requires_grad` or backward
coverage for this module.

**Recommendation:** Implement autograd-aware masked fill and reductions,
including correct tie behavior for extrema, and test CPU/CUDA gradients with
partial, all-valid, and all-masked masks.

### CORE-065: CUDA masked operations inconsistently and silently return CPU tensors

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/masked.rs:148-160`, `src/masked.rs:296-332`,
  `src/masked.rs:343-447`, `tests/conformance_masked.rs:453-526`

Despite the module-level "no silent CPU/GPU round trips" claim, several public
operations change device based on the operation or edge case:

- `filled` and `to_tensor` always return CPU tensors for CUDA-backed data.
- `masked_mean` downloads its GPU sum and always returns a CPU scalar.
- `masked_count` always returns a CPU scalar.
- CUDA `masked_min`/`masked_max` normally return CUDA scalars, but the
  all-masked case returns a CPU NaN scalar.

The GPU conformance lane feeds every result through a device-transparent
readback helper and never asserts the result device, so these inconsistencies
remain invisible.

**Recommendation:** Define and enforce one device contract, preferably
matching the underlying data tensor for value-producing operations. Perform
mean division on-device, upload edge-case scalars where necessary, and assert
devices in every GPU test.

### CORE-066: Nested dense conversion and attention silently detach component graphs

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/nested.rs:179-255`, `src/nested.rs:392`,
  `src/nested.rs:417-602`, `src/nested.rs:673-902`

The design states that the component-list layout preserves per-component
autograd independence, but its principal operations discard those graphs:

- `NestedTensor::to_padded` constructs a fresh detached tensor on CPU and GPU.
- `NestedTensor::from_padded` constructs detached CPU components; its GPU path
  ends in `view_reshape`, which also creates detached leaves.
- `nested_scaled_dot_product_attention` constructs detached output components
  on both the CPU and FlashAttention paths.

A model using nested components for variable-length training therefore loses
gradient flow as soon as it pads, unpads, or applies the provided attention
helper.

**Recommendation:** Implement graph-preserving scatter/gather conversion
nodes and a differentiable nested attention path. Add end-to-end component
gradient tests on CPU and CUDA.

### CORE-067: CUDA nested attention has no working fallback outside the flash-kernel regime

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/nested.rs:673-902`,
  `.design/ferrotorch-core/nested.md:45-50`

The GPU helper returns `Ok(false)` for valid CUDA inputs when the dtype is not
f32/f64, no backend is registered, or either head dimension exceeds 128. The
caller then enters the purported composite fallback, but immediately calls
CPU-only `data()` on the CUDA query, key, and value tensors. Instead of
running a composite attention path, valid CUDA attention outside the narrow
flash regime fails with a GPU data-access error.

This contradicts the design's claim that the CPU/composite path is a fallback
when the backend declines.

**Recommendation:** Compose a device-aware matmul/softmax/matmul fallback on
CUDA or return an explicit unsupported-shape error before claiming fallback
support. Add tests at head sizes 128 and 129 and with the flash backend
unavailable.

### CORE-068: `PackedNestedTensor::from_data_tensor` accepts invalid offset layouts

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/nested.rs:1185-1230`

The public reconstruction API checks only that offsets are non-empty,
monotonic, and end at `tensor.numel()`. It does not enforce the documented
layout invariants:

- `offsets[0] == 0`
- every component extent is divisible by `product(tail_shape)`
- the input is the documented flat 1-D tensor

For example, offsets beginning above zero silently discard a prefix of the
data. An extent not divisible by the tail size makes `length()` truncate; a
later `to_nested()` can then pair a shape with fewer logical elements than
the component slice and silently lose values.

**Recommendation:** Centralize complete packed-layout validation and call it
from every constructor. Reject nonzero first offsets, non-divisible extents,
and non-1-D data tensors.

### CORE-069: Packed nested tensors mis-handle zero-sized tail dimensions

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/nested.rs:983-1045`, `src/nested.rs:1129-1147`,
  `src/nested.rs:1370-1451`

Packed layout calculations repeatedly compute
`product(tail_shape).max(1)`. An empty `tail_shape` correctly represents a
scalar tail and needs a factor of one, but a non-empty tail shape containing a
zero must have zero elements. Treating both cases identically lets
`from_sequences` accept non-empty data for logical component shapes such as
`[L, 0]`, computes incorrect lengths and offsets, and can construct padded
tensors whose storage contains values despite a zero logical element count.

**Recommendation:** Distinguish an empty tail shape from a tail containing a
zero. Use the actual product for non-empty shapes and add round-trip tests for
`[L, 0]`, `[L, 2, 0]`, and zero-length ragged components.

### CORE-070: Nested CPU paths reject valid views and mixed-device construction creates unusable values

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/nested.rs:66-111`, `src/nested.rs:179-255`,
  `src/nested.rs:417-530`, `src/nested.rs:673-775`,
  `src/nested.rs:1053-1077`, `src/nested.rs:1401-1451`

`NestedTensor::new` validates shapes but not that components share a device.
It can therefore successfully construct a mixed CPU/CUDA nested tensor, even
though `to_padded` falls through to a CPU path that calls `data()` on the CUDA
component and fails. The same raw `data()` dependency causes CPU
`to_padded`, `from_padded`, `from_nested`, packed `from_padded`, and the CPU
attention fallback to reject valid non-contiguous component or source views.

**Recommendation:** Enforce a single-device invariant at construction, and
materialize logical views with device-aware operations rather than requiring
packed CPU storage. Add mixed-device rejection and non-contiguous round-trip
tests.

### CORE-071: Empty packed components report a false mean of zero

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/nested.rs:1348-1364`,
  `src/nested.rs:1812-1819`

`PackedNestedTensor::mean_per_component` explicitly returns zero for an empty
component. An arithmetic mean over zero elements is undefined; PyTorch-style
floating reductions return NaN. Returning zero is a plausible finite value
that can silently bias downstream aggregation, and the in-module test locks
in the divergence.

**Recommendation:** Return NaN for floating packed tensors or expose a
structured error/mask for empty components. Replace the zero-expectation test
with the chosen explicit contract.

### CORE-072: Public CSR and CSC constructors accept structurally invalid compressed layouts

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/sparse.rs:1088-1120`, `src/sparse.rs:1203-1285`,
  `src/sparse.rs:1701-1740`, `src/sparse.rs:1745-1813`,
  `src/sparse.rs:1831-1897`

`CsrTensor::new` checks only pointer-array length and equality of the column
and value counts. It does not require a zero first pointer, monotonic pointers,
a final pointer equal to `nnz`, or in-range column indices. `CscTensor::new`
checks row bounds but likewise omits all compressed-pointer invariants.

Subsequent public operations trust these arrays. Malformed CSR/CSC values can
therefore panic through out-of-bounds indexing in dense conversion and format
conversion. GPU paths pass the malformed descriptors into backend sparse
operations. `CscTensor::to_csr` additionally converts a resulting validation
failure into a process panic with `expect`.

**Recommendation:** Centralize complete compressed-layout validation and use
it in both constructors and every backend-return conversion. Replace infallible
conversion signatures and `expect` with structured errors.

### CORE-073: CSC dense materialization overwrites duplicate entries instead of summing them

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/sparse.rs:1701-1740`, `src/sparse.rs:1831-1897`

`CscTensor::new` permits duplicate row indices within a column, but the CPU
`to_dense_on` path assigns each value with `=`. The last duplicate silently
wins. Other sparse representations in this module explicitly sum duplicates,
and dense materialization of a sparse tensor is expected to accumulate them.

**Recommendation:** Accumulate CSC entries with `+=` or require and validate a
canonical duplicate-free representation. Add direct-construction duplicate
tests for CPU and CUDA.

### CORE-074: Sparse operations on dense tensors silently sever autograd

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/sparse.rs:473-621`, `src/sparse.rs:1415-1494`,
  `src/sparse.rs:1569-1678`, `tests/conformance_nested_sparse.rs`

`SparseTensor::spmm` always constructs a detached output even when its dense
operand requires gradients. `SemiStructuredSparseTensor::compress` extracts
the dense input into ordinary vectors, `decompress` returns a detached tensor,
and `sparse_matmul_24` returns a detached result on both CUDA and reference
paths. A loss using these APIs therefore cannot train the dense operand or the
original semi-structured weight.

The conformance suite checks forward values only and contains no backward or
`requires_grad` assertions for these operations.

**Recommendation:** Add autograd nodes for sparse-dense matmul and
semi-structured projection/matmul, with CPU/CUDA backward parity tests.

### CORE-075: 2:4 compression validates total size instead of the innermost dimension

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/sparse.rs:1405-1468`, `src/sparse.rs:1547-1678`

The documented 2:4 contract groups elements along the innermost dimension and
requires a matrix weight width divisible by four. `compress` checks only
`dense.numel() % 4 == 0` and then groups the entire flat buffer. A shape such
as `[2, 2]` passes, but its only group spans two rows. `sparse_matmul_24` also
omits its documented `n % 4 == 0` check and accepts the malformed compressed
weight.

**Recommendation:** Require a non-scalar input whose last dimension is a
multiple of four, and independently enforce `n % 4 == 0` in
`sparse_matmul_24`. Test shapes whose total size is divisible by four but
whose last dimension is not.

### CORE-076: CUDA semi-structured matmul fallback silently returns a CPU tensor

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/sparse.rs:1601-1678`,
  `tests/conformance_nested_sparse.rs:2250-2260`

When a CUDA `sparse_matmul_24` call is not f32 or the cuSPARSELt backend
declines the shape/runtime, the reference fallback downloads `a` through
`data_vec` and constructs its output with CPU storage. A successful operation
therefore silently changes device. The conformance test for this operation's
GPU lane is an unconditional cascade skip.

**Recommendation:** Run the composite fallback on the input device or return a
clear unsupported-kernel error. Assert the output device for fast-path and
declined-backend cases.

### CORE-077: Sparse GPU paths truncate indices and pointers to 32 bits without validation

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/sparse.rs:307-400`, `src/sparse.rs:501-605`,
  `src/sparse.rs:933-982`, `src/sparse.rs:1220-1260`,
  `src/sparse.rs:1300-1374`, `src/sparse.rs:1831-1875`,
  `src/sparse.rs:1910-2016`

Public sparse representations store shapes, indices, and compressed pointers
as `usize`, but GPU dispatch repeatedly converts them with unchecked `as u32`
casts. Valid public inputs above `u32::MAX` wrap to unrelated coordinates or
pointer values before reaching the backend, creating CPU/GPU disagreement and
potentially malformed GPU descriptors.

**Recommendation:** Use checked conversions and reject unsupported large
layouts before backend dispatch, or expose 64-bit sparse-index backend APIs.

### CORE-078: Zero-sized sparse-gradient slabs can pass validation and panic during SGD

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/sparse.rs:2064-2109`, `src/sparse.rs:2190-2365`

`SparseGrad` calculates `product(slab_shape).max(1)`. This is correct for an
empty scalar slab shape, but wrong for a non-empty shape containing zero. For
example, a gradient with `slab_shape = [0]`, one index, and one value passes
construction and matches a parameter shaped `[2, 0]`. The CPU update then
writes one element into the parameter's empty data buffer and panics.

**Recommendation:** Distinguish an empty shape from a shape containing a zero,
use checked products, and test zero-sized trailing dimensions through
construction, coalescing, and application.

### CORE-079: CUDA sparse SGD loses row-index precision above 2^24

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/sparse.rs:2258-2279`, `src/sparse.rs:2317-2335`

Both f32 and f64 CUDA sparse-SGD lanes encode `usize` row indices as f32.
Integers above `2^24` are not exactly representable, so a validated row index
can round to a neighboring row before `scatter_add_rows` consumes it. Large
embedding tables can therefore update the wrong parameter row.

**Recommendation:** Give scatter kernels an integer index ABI and use checked
integer-width conversion. Add boundary tests around `2^24` and the selected
backend index width.

### CORE-080: Unsupported CUDA sparse-SGD dtypes silently move the parameter to CPU

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/sparse.rs:2223-2365`, `src/tensor.rs:703-720`

The implementation comments claim unsupported CUDA dtypes will fail when
`data_vec` is called. In fact, `Tensor::data_vec` explicitly downloads CUDA
tensors. The generic CPU lane then performs the update and replaces `param`
with CPU storage. A successful optimizer step can silently move an unsupported
CUDA parameter to CPU.

**Recommendation:** Return an explicit unsupported-dtype error before the CPU
lane when `param.is_cuda()`, or implement an on-device update for every
supported dtype. Test device preservation beyond f32/f64.

### CORE-081: Sparse SGD replaces tensor identity and leaves aliases stale

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/sparse.rs:2290-2296`, `src/sparse.rs:2341-2346`,
  `src/sparse.rs:2354-2365`, `src/tensor.rs:58-60`,
  `src/tensor.rs:120`, `src/tensor.rs:400-404`

Every non-empty `SparseGrad::apply_sgd` path assigns a newly constructed
tensor into `*param`. That assigns a fresh `TensorId`, discards the original
tensor's grad and hooks, and does not update any existing clones that shared
the old parameter storage. This contradicts optimizer-style in-place update
semantics and can leave model-held aliases observing stale weights.

**Recommendation:** Implement optimizer updates against stable parameter
storage under a well-defined no-grad mutation mechanism. Add tests that retain
aliases, IDs, hooks, and gradient state across a sparse optimizer step.

### CORE-082: Pruning outputs are disconnected leaves, not differentiable masked weights

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/pruning.rs:31-77`, `src/pruning.rs:94-122`,
  `tests/conformance_quantize_prune.rs:920-941`,
  `.design/ferrotorch-core/pruning.md`

Both pruning functions read raw values and construct a new tensor with
`weights.requires_grad()` copied as a boolean. This creates a fresh leaf with
no `grad_fn`; backward accumulates on the returned leaf and never reaches the
original weight. The design explicitly claims that propagating the flag allows
backward to flow through surviving weights, but the conformance test checks
only the flag and never runs backward.

**Recommendation:** Represent pruning as a differentiable mask multiplication
or attach an explicit backward node that masks the incoming gradient. Test the
original parameter's gradient, including zero gradient at pruned positions.

### CORE-083: Magnitude pruning can remove far more elements than requested

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/pruning.rs:41-75`,
  `.design/ferrotorch-core/pruning.md:25-28`

The function selects the `n_prune`th magnitude as a threshold, then zeros
every value whose magnitude is less than or equal to that threshold. Ties at
the threshold therefore all disappear. For `[1, 1, 1, 1]` with sparsity
`0.25`, `n_prune` is one but all four elements are pruned. This violates the
documented exact-count contract and can catastrophically oversparsify layers.

**Recommendation:** Select exactly `n_prune` stable indices, with a documented
tie-break rule, and construct the mask from those indices.

### CORE-084: The 2:4 pruning mask can group values across row boundaries

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/pruning.rs:80-122`,
  `.design/ferrotorch-core/pruning.md:29-33`

`apply_2_4_mask` groups the entire flat storage in chunks of four and leaves a
single flat remainder. For a multidimensional tensor whose last dimension is
not divisible by four, groups span row boundaries. That is not the
innermost-dimension 2:4 layout required by semi-structured sparse kernels, and
it can produce a mask that cannot be consumed as valid 2:4 weights.

**Recommendation:** Apply groups independently along the final dimension and
reject or explicitly handle rows whose width is not divisible by four.

### CORE-085: Quantization and pruning reject valid non-contiguous CPU views

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/quantize.rs:237-339`, `src/pruning.rs:31-126`,
  `src/tensor.rs:673-709`

The public CPU quantization and pruning APIs all call the raw `data()` accessor,
which requires packed storage and does not gather a view's logical elements.
Valid transposed, sliced, or offset CPU tensors therefore fail rather than
being processed in logical tensor order.

**Recommendation:** Materialize logical values through a view-aware path or
compose tensor operations that naturally respect strides. Add transposed and
offset-view parity tests.

### CORE-086: Quantized matmul can overflow its i32 accumulator

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/quantize.rs:379-480`

The implementation states that i32 accumulation avoids overflow, but sums
products directly with `sum += qa * qb`. Ordinary INT8 products can overflow
i32 for sufficiently large inner dimensions, causing a debug panic or wrapped
release result. Zero-point subtraction can further increase each product's
magnitude.

**Recommendation:** Accumulate in i64 with checked or explicitly defined
requantization, then range-check before narrowing. Add long-inner-dimension
tests near and beyond the i32 boundary.

### CORE-087: The advertised packed INT4 format stores one byte per element

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/quantize.rs:43-50`, `src/quantize.rs:80-105`,
  `src/quantize.rs:267-337`, `.design/ferrotorch-core/quantize.md:91-94`

The design says INT4 packs two values per i8 storage byte, but `quantize`
pushes one `i8` for every tensor element and `dequantize` reads one byte per
element. INT4 therefore receives none of the promised storage compression,
and its data representation is incompatible with consumers expecting packed
nibbles.

**Recommendation:** Implement nibble packing/unpacking with an explicit odd
element policy, or rename and document the representation as unpacked INT4.

### CORE-088: A zero-bin histogram observer panics on its first finite sample

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/quantize.rs:759-810`, `src/quantize.rs:813-879`

`HistogramObserver::new(0)` is accepted and creates an empty bin vector.
Observation then computes `n - 1` and indexes the empty vector. Depending on
overflow checking, this panics during subtraction or during indexing.

**Recommendation:** Make construction fallible and require at least one bin.
Use checked indexing and add zero/one-bin boundary tests.

### CORE-089: `HistogramObserver` ignores its histogram when calculating qparams

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/quantize.rs:745-879`,
  `.design/ferrotorch-core/quantize.md:111-113`

Although the observer maintains and redistributes bins and the design claims
KL-divergence threshold selection, `calculate_qparams` simply applies min/max
quantization to the observed extrema. The histogram has no effect on the
result, so this type behaves like a more expensive `MinMaxObserver` and does
not provide the advertised outlier-resistant calibration.

**Recommendation:** Implement the documented histogram-based clipping
algorithm or remove the unsupported claim and type distinction.

### CORE-090: Disabling fake quantization also silently disables observation

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/quantize.rs:918-973`

`FakeQuantize::forward` returns immediately when `fake_quant_enabled` is false,
before checking `observer_enabled`. These controls are public and documented
as independent. A calibration phase that disables fake quantization while
leaving observation enabled silently collects no statistics and retains stale
or absent qparams.

**Recommendation:** Run the observer update first whenever
`observer_enabled`, then independently decide whether to fake-quantize the
output. Test all four flag combinations.

### CORE-091: CUDA RNG fork/join state is not thread-local or atomic

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/quantize.rs:1100-1167`

The CUDA RNG helper uses one process-global state and one process-global saved
state stack protected by separate mutexes. Concurrent threads can interleave
forks and joins, pop each other's saved states, and restore an unrelated seed.
Even a single `fork_rng` is not atomic because it releases the state lock
before pushing and setting the replacement state.

**Recommendation:** Use scoped/thread-local fork state or lock one structure
containing both current state and stack for the complete transition. Include
concurrent and nested-fork tests.

### CORE-092: Generic f64 quantization silently narrows all values and qparams to f32

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/quantize.rs:237-372`

`quantize<T: Float>` advertises a generic floating input but converts every
value to f32 and stores f32 scales. Finite f64 values outside the f32 range
become infinities, and ordinary f64 values lose precision before min/max,
rounding, and code selection. `dequantize::<f64>` cannot recover that loss
because it first computes each value in f32.

**Recommendation:** Either restrict the API to f32 inputs or compute and store
qparams at a precision appropriate to `T`. Add f64 range and precision tests.

### CORE-093: Top-level `manual_seed` does not seed random generation in other threads

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/rng.rs:250-289`, `src/creation.rs:117-174`,
  `.design/ferrotorch-core/rng.md`

The public function is documented as mirroring `torch.manual_seed`, but it
reseeds only a thread-local generator. Worker threads initialized after or
before the call retain independent entropy-derived streams unless callers
manually execute the seed operation in each thread. Parallel initializers and
random operations are therefore not reproducible from a normal single
top-level seed call.

**Recommendation:** Define a process-level seeded default-generator policy
that deterministically initializes worker streams, or rename the function to
make its current-thread-only semantics explicit. Test seeded work across
fresh and persistent worker threads.

### CORE-094: Reentrant use of the public thread-RNG accessor panics

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/rng.rs:250-289`, `src/creation.rs:135-174`

`with_thread_rng` holds a mutable `RefCell` borrow while invoking an arbitrary
public closure. If that closure calls `rand`, `randn`, another initializer, or
`with_thread_rng` again, the nested mutable borrow panics. This is an
unexpected process-level failure reachable through entirely safe public APIs.

**Recommendation:** Avoid exposing a closure while holding a `RefCell` borrow,
or detect reentrancy and return a structured error. Add nested-access tests.

### CORE-095: `manual_seed` silently discards GPU seeding failures

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/rng.rs:269-282`

When a GPU backend is registered, `manual_seed` calls `manual_seed_gpu` and
unconditionally discards its result. A backend or device-manager failure
therefore leaves CUDA random state unseeded while the public call reports
success, undermining reproducibility without any diagnostic.

**Recommendation:** Return a result from a fallible all-device seeding API, or
record and surface backend failures before claiming CPU/CUDA seed parity.

### CORE-096: Flex attention with a score modifier fails valid zero-batch and zero-head inputs

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/flex_attention.rs:213-250`

When `score_mod` is present, the implementation builds per-batch/head vectors
and reassembles them with `cat`. If `batch == 0`, the final batch vector is
empty; if `heads == 0`, each per-batch head vector is empty. Both paths call
`cat` with no tensors and fail, even though the corresponding empty attention
shapes are structurally valid and the no-modifier path does not impose this
extra restriction.

**Recommendation:** Short-circuit empty batch/head dimensions to a correctly
shaped empty result, and add score-modifier parity tests for both cases.

### CORE-097: Meta matmul panics for valid batched vector cases

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/meta_propagate.rs:137-235`

The matmul shape propagator handles only the four exact 1-D/2-D combinations,
then assumes both operands in every remaining case have rank at least two.
Valid PyTorch matmul forms such as rank-1 vector times rank-3 batched matrix,
or rank-3 batched matrix times rank-1 vector, enter that fallback and evaluate
an index based on `ndim - 2` for the rank-one operand. This underflows and
panics rather than returning the correct meta shape.

**Recommendation:** Implement the full vector promotion/squeeze rules used by
matmul before broadcasting batch dimensions. Add every rank-1 versus rank-N
combination for `N >= 3`.

### CORE-098: Meta propagation silently drops autograd state

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/meta_propagate.rs:46-235`, `src/creation.rs:264-300`

Every successful meta helper returns a fresh `zeros_meta` tensor. These
outputs do not preserve `requires_grad` and have no operation-specific
`grad_fn`, even when meta inputs require gradients. An operation that normally
participates in autograd therefore becomes a detached non-grad tensor solely
because its input device is Meta.

**Recommendation:** Construct meta operation results through the same
autograd-aware operation machinery as real-device outputs while skipping only
the data kernel. Add meta `requires_grad` and backward-graph structure tests.

### CORE-099: SIMD reduction behavior is hard-coded to one AVX2 host

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/simd_reduce.rs:39-222`,
  `.design/ferrotorch-core/simd_reduce.md`

The public reduction helpers permanently use eight f32 lanes and four f64
lanes because the development host used AVX2 without AVX-512. PyTorch selects
different vector widths and reduction trees on AVX-512, other CPU targets,
and potentially scalar builds. Ferrotorch can therefore make different
boundary decisions or sums from PyTorch solely because it runs on another
supported machine.

**Recommendation:** Select the modeled reduction tree from detected/compiled
CPU capabilities, and maintain oracle tests for each supported target class.

### CORE-100: CUDA readback relies on an alignment guarantee absent from the backend trait

- **Severity:** Critical
- **Confidence:** Confirmed
- **Affected code:** `src/gpu_dispatch.rs:274-289`,
  `src/tensor.rs:903-926`, `src/int_tensor.rs:297-336`,
  `../ferrotorch-gpu/src/backend_impl.rs:1004-1119`

The public `GpuBackend::gpu_to_cpu` contract returns an ordinary `Vec<u8>`.
Both `Tensor::to(Cpu)` and `IntTensor::to(Cpu)` then cast that byte vector's
pointer to a wider element pointer and reconstruct a typed `Vec` with
`Vec::from_raw_parts`. This requires the byte allocation to have the typed
element's alignment and a compatible allocation layout, neither of which is
expressed or enforced by the trait. The bundled CUDA backend happens to create
its byte vector by reinterpreting a typed vector, but any conforming custom
backend may return a normally allocated `Vec<u8>`, making the core conversion
undefined behavior. The conversion also validates byte length but not
capacity divisibility or pointer alignment.

**Recommendation:** Return an ownership-safe typed/readback abstraction, or
copy bytes into a newly allocated typed vector. If zero-copy ownership transfer
is retained, encode and validate its allocation-layout contract explicitly.

### CORE-101: CPU i32 add, subtract, and sum do not wrap as documented

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/int_tensor.rs:568-585`,
  `src/int_tensor.rs:671-678`

The CPU implementations widen operands to i64, perform an i64 wrapping
operation, and then attempt to convert the result back to i32. When an i32
boundary is crossed, conversion fails and `unwrap_or` returns the original
left operand or accumulator. For example, `i32::MAX + 1` returns
`i32::MAX` instead of `i32::MIN`; a sum that first overflows simply ignores
that element. This contradicts the methods' documented wrapping semantics and
can diverge from the GPU kernels.

**Recommendation:** Perform arithmetic at the concrete integer width, as the
existing multiplication and shift helpers do. Add boundary-crossing tests for
every arithmetic operation and reduction on both i32 and i64.

### CORE-102: CPU integer division and remainder by zero silently fabricate zero

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/int_tensor.rs:606-623`,
  `src/int_tensor.rs:722-758`

The CPU reference paths explicitly return zero for every division or remainder
whose divisor is zero. This converts an invalid arithmetic operation into
plausible data and diverges from PyTorch CPU integer behavior, which reports
an error. The comment justifies the choice using CUDA's unspecified result,
but that does not make zero a valid CPU result or establish consistent
cross-device semantics.

**Recommendation:** Detect zero divisors and return a structured error before
dispatch, consistently on CPU and GPU. Add mixed zero/nonzero divisor tests.

### CORE-103: Empty CUDA integer sum and product return CPU tensors

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/int_tensor.rs:518-548`

The CUDA reduction path special-cases empty sum and product by returning
`IntTensor::scalar(id)`, which always constructs CPU storage. Non-empty CUDA
reductions return CUDA-resident scalar tensors. Consequently, the result
device changes solely because an input dimension becomes zero.

**Recommendation:** Construct the identity scalar on the input device, or run
a device-resident empty-reduction path. Test empty reductions on every device.

### CORE-104: Public GPU BoolTensor construction accepts malformed handles in release builds

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/bool_tensor.rs:207-222`,
  `src/bool_tensor.rs:255-271`, `src/gpu_dispatch.rs:162-200`

`BoolTensor::from_gpu_handle` is a safe public infallible constructor, but it
checks the handle dtype only with `debug_assert` and never checks that
`handle.len()` equals the shape's element count. Release callers can therefore
construct a bool tensor around a wrong-dtype or wrong-length handle. Later
kernels and readback trust those invariants; CPU readback also retains the
declared shape regardless of the returned byte count.

**Recommendation:** Make construction fallible and validate dtype, checked
shape product, and handle length. Keep any unchecked constructor private and
clearly unsafe.

### CORE-105: Predicate mask construction silently downloads CUDA tensors

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/bool_tensor.rs:141-150`

`BoolTensor::from_predicate` calls `Tensor::data_vec`, evaluates the closure on
the host, and constructs the result through CPU-only `BoolTensor::from_vec`.
A CUDA input therefore incurs an implicit full-device readback and produces a
CPU mask without any device transition in the API or documentation.

**Recommendation:** Reject device tensors for arbitrary host closures, or
provide explicit device-aware predicate operations that keep masks resident.
Document and test result-device behavior.

### CORE-106: Bool comparison APIs reject broadcast-compatible inputs

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/bool_tensor.rs:338-380`,
  `src/bool_tensor.rs:458-540`, `src/bool_tensor.rs:597-631`

The public float and integer comparison constructors claim to mirror operations
such as `torch.gt`, but require exactly equal shapes and zip elements
one-for-one. PyTorch comparisons broadcast compatible operands. Logical bool
binary operations impose the same restriction, leaving basic tensor-mask
workflows unsupported despite the parity claim.

**Recommendation:** Infer a broadcast shape, materialize or index operands
accordingly on CPU and GPU, and return shape errors only for incompatible
inputs. Add scalar, singleton-axis, and multi-axis broadcast tests.

### CORE-107: Complex magnitude overflows and underflows avoidably

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/complex_tensor.rs:246-255`

`ComplexTensor::abs` computes `sqrt(re * re + im * im)` directly. Squaring can
overflow even when the true magnitude is finite, and can underflow small
finite components to zero. This is a standard numerical-stability failure in
a foundational complex operation.

**Recommendation:** Use a scaled hypotenuse algorithm or the element type's
stable `hypot` implementation. Add near-maximum, subnormal, infinity, and NaN
tests for f32 and f64.

### CORE-108: `topk` panics on a valid zero-width, zero-selection input

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/ops/search.rs:629-700`

For an input whose last dimension has size zero, `k = 0` passes validation.
Both CPU and GPU paths then compute `input.numel() / last_dim`, producing an
integer divide-by-zero panic. The valid empty result should preserve all outer
dimensions and replace the final dimension with zero.

**Recommendation:** Short-circuit `last_dim == 0 && k == 0` to correctly
shaped empty values and indices. Add CPU and GPU tests with nonzero and zero
outer dimensions.

### CORE-109: `topk` values are always detached from autograd

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/ops/search.rs:629-733`

Every `topk` path constructs the values result with `requires_grad = false`
and attaches no backward function. In PyTorch, selected values are
differentiable and backward scatters their gradients to the selected input
indices. Here a gradient-tracking input silently produces a detached values
tensor, so ordinary top-k losses cannot train.

**Recommendation:** Attach a backward function that scatters value gradients
through the saved indices, preserving input device and dtype. Add tie, empty,
and repeated-index gradient tests.

### CORE-110: `meshgrid` silently detaches every output

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/ops/search.rs:481-599`

CPU and CUDA meshgrid paths build fresh tensors with `requires_grad = false`
regardless of whether coordinate inputs require gradients. PyTorch meshgrid
outputs remain connected to their corresponding coordinate tensors, with
backward reducing the broadcast gradient over all other axes.

**Recommendation:** Implement meshgrid through autograd-aware reshape and
expand operations or attach an equivalent reduction backward. Test gradients
for `ij`, `xy`, singleton, and empty axes.

### CORE-111: Direct integer-index gather APIs do not validate index values

- **Severity:** Critical
- **Confidence:** Confirmed
- **Affected code:** `src/ops/phase2c.rs:171-218`,
  `src/ops/phase2c.rs:247-334`, `src/ops/phase2c.rs:396-466`,
  `../ferrotorch-gpu/src/gather_int.rs:23-52`

The public `Tensor` and `IntTensor` `index_select`/`gather` methods never check
whether integer index values are negative or exceed the selected axis. CPU
helpers cast signed values directly to `usize` and index slices, so ordinary
invalid input panics. CUDA dispatch forwards those values to PTX kernels that
explicitly omit bounds checks and compute unchecked source addresses, enabling
out-of-bounds device reads from a safe public API.

**Recommendation:** Validate every index before CPU or GPU dispatch. For
resident GPU indices, use a device-side validation/error flag or checked
kernels rather than declaring invalid public input to be undefined behavior.

### CORE-112: `gather` accepts smaller non-axis dimensions but indexes as if they were full-size

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/ops/phase2c.rs:193-216`,
  `src/ops/phase2c.rs:298-334`, `src/ops/phase2c.rs:437-466`,
  `src/ops/phase2c.rs:529-556`

Validation correctly allows the gather index to be smaller than the input on
axes other than the gather axis. Execution then factors `outer` and `inner`
from the input shape and assumes the index/output layout is
`[input_outer, index_dim, input_inner]`. For an accepted case such as input
shape `[2, 3]`, gather axis 1, and index shape `[1, 2]`, the CPU loop reads four
indices from a two-element index tensor and panics. The CUDA kernel similarly
launches for the larger assumed layout and reads beyond the index buffer.

**Recommendation:** Iterate over the actual index/output shape and map each
output coordinate into the input shape, preserving smaller non-axis extents.
Add parity cases for every non-gather axis being smaller.

### CORE-113: The direct argmax/argmin path ignores NaNs on CPU and CUDA

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/ops/phase2c.rs:57-165`,
  `../ferrotorch-gpu/src/reduce_arg.rs:12-48`

The `Tensor::argmax` and `Tensor::argmin` methods in `ops/phase2c` use strict
comparisons, so a NaN after the first element never wins. PyTorch selects the
first NaN for both reductions. The bundled GPU kernel explicitly documents
this known divergence, while the separate reduction implementation already
contains the correct NaN-propagating predicate. Thus two public arg-reduction
surfaces disagree with each other as well as with PyTorch.

**Recommendation:** Route all public arg reductions through one
NaN-propagating implementation and apply the same predicate in GPU kernels.
Add NaNs at every position for global and per-axis reductions.

### CORE-114: Direct `Tensor::index_select` and `Tensor::gather` silently detach autograd

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/ops/phase2c.rs:247-334`

Both public methods always construct their result with `requires_grad = false`
and no backward function. The crate separately implements differentiable
indexing helpers, but callers of these PyTorch-named `Tensor` methods receive a
successful detached result when the input tracks gradients.

**Recommendation:** Make the public methods delegate to the validated,
autograd-aware implementations, leaving only private kernel-level helpers
detached.

### CORE-115: CPU float-to-integer casting disagrees with CUDA and PyTorch on exceptional values

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/ops/phase2c.rs:336-373`,
  `../ferrotorch-gpu/src/cast_kernels.rs:72-100`

The CPU conversion first casts every float to i64 using Rust's saturating
`as` semantics, where NaN becomes zero and infinities/out-of-range finite
values saturate. Narrow i32 conversion may then return an error. The CUDA path
uses PTX `cvt.rzi`, whose invalid-conversion result differs, and PyTorch's
native conversion behavior uses the target-width conversion result rather
than this two-stage saturate-then-fail policy. Consequently the same tensor can
yield zero, an error, or an integer-indefinite sentinel depending on target
width and device.

**Recommendation:** Define and implement one target-width conversion policy
matching the selected PyTorch oracle on CPU and CUDA. Add NaN, both infinities,
just-out-of-range, and very-large finite cases for every float/integer pair.

### CORE-116: Gradient-enabled `cond` and `scan` silently move every result to CPU

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/ops/higher_order.rs:123-143`,
  `src/ops/higher_order.rs:274-304`

When no relevant input requires gradients, both control-flow primitives return
the branch or step tensors unchanged. Once gradients are enabled, each output
is read through `data_vec` and wrapped in fresh CPU storage. CUDA branch
outputs, scan carries, and step outputs therefore change device solely because
autograd is active. Their wrapper's saved target remains on the original
device, so backward can then route a CPU upstream gradient into a CUDA graph.

**Recommendation:** Wrap the existing result storage/device without host
materialization, or attach control-flow metadata within the original graph.
Test forward and backward device preservation for CUDA carries and outputs.

### CORE-117: `cond` and `scan` create fake grad-tracking outputs from detached user results

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/ops/higher_order.rs:123-143`,
  `src/ops/higher_order.rs:274-304`,
  `src/ops/higher_order.rs:520-549`, `src/ops/higher_order.rs:797-827`

The decision to wrap outputs is based only on whether any operand, initial
carry, or input sequence element requires gradients. It does not check whether
the user-supplied branch or step result is actually connected to those inputs.
A branch that constructs a detached tensor therefore returns a wrapper that
claims `requires_grad`, but its backward target has no path to the operands.
The included tests explicitly accept this behavior by checking only the flag,
not whether gradients reach the purported inputs.

**Recommendation:** Preserve the actual output graph and its real
`requires_grad` state. Do not manufacture tracking metadata for detached
results; add negative tests proving disconnected branch and step outputs remain
disconnected.

### CORE-118: `cond` advertises branch-shape validation that it never performs

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/ops/higher_order.rs:69-143`,
  `src/ops/higher_order.rs:147-180`

The public `cond` error contract says it rejects branches returning different
numbers or shapes of tensors, but it executes only the selected branch and
never invokes `validate_cond_branches`. Incompatible branches therefore
succeed silently. The separate optional utility does not enforce the contract
at the operation boundary.

**Recommendation:** Require branch metadata or a tracing/eager-validation
phase that enforces compatible output count, shape, dtype, and device, or
remove the unsupported guarantee from the public API.

### CORE-119: `cond` cannot evaluate a CUDA-resident predicate

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/ops/higher_order.rs:102-116`

After validating that the predicate has one element, `cond` reads it through
`pred.data()`, which rejects CUDA and other non-CPU storage. The operation can
run CUDA branches and accepts device tensors elsewhere, but its selector must
silently be moved by the caller or the operation errors.

**Recommendation:** Provide an explicit scalar synchronization/readback path
for supported devices or document and validate a CPU-only predicate before any
branch work.

### CORE-120: CPU `diag` panics for valid diagonals outside a matrix

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/ops/tensor_ops.rs:318-334`

For 2-D input, the CPU extraction path computes `rows - start_r` and
`cols - start_c` with unchecked subtraction. A diagonal offset beyond the
matrix bounds, which should return an empty diagonal, underflows and then
panics or drives invalid indexing. The CUDA path uses `saturating_sub` for the
same calculation and therefore disagrees with CPU behavior.

**Recommendation:** Use saturating bounds logic on every path and add offsets
just inside, exactly at, and far beyond both matrix edges.

### CORE-121: Extreme triangular and diagonal offsets overflow before masking

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/ops/tensor_ops.rs:75-124`,
  `src/ops/tensor_ops.rs:161-210`, `src/ops/tensor_ops.rs:251-334`

`triu` and `tril` add a row index to caller-controlled `i64` diagonal values,
which overflows for extreme offsets. `diag` converts `unsigned_abs()` to
`usize`, adds it to the input length, and elsewhere negates a negative offset;
`i64::MIN` cannot be negated and very large offsets overflow sizes or request
impossible allocations. These valid offset arguments should simply select no
elements or produce a checked size error, not panic or wrap.

**Recommendation:** Compare offsets without signed addition/negation and use
checked size arithmetic before allocation. Add `i64::MIN`, `i64::MAX`, and
large finite offsets on CPU and CUDA.

### CORE-122: CPU `cdist` computes incorrect results for `p = 0` and `p = infinity`

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/ops/tensor_ops.rs:491-626`

The generic CPU formula sums `abs(diff)^p` and raises the sum to `1/p`.
For `p = 0`, this counts every feature, including equal coordinates, instead
of PyTorch's zero-norm count of unequal coordinates. For `p = infinity`, it
does not compute the maximum absolute difference and instead degenerates
through infinite exponents and a zero final exponent. The GPU dispatch
advertises dedicated implementations for these cases, so devices disagree.

**Recommendation:** Implement explicit zero-, one-, two-, infinity-, and
general-p branches with shared CPU/GPU parity tests, including equal
coordinates and differences above and below one.

### CORE-123: `cdist` silently detaches both differentiable inputs

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/ops/tensor_ops.rs:491-626`

Every CPU and CUDA result is created with `requires_grad = false` and no
backward function. Pairwise distance is differentiable for supported `p`
values, and PyTorch propagates gradients to both input point sets. Ferrotorch
silently turns a training loss using `cdist` into a detached tensor.

**Recommendation:** Attach a device-aware backward for both inputs, with
defined behavior at zero distances and nonsmooth norms. Add gradient checks
for all supported `p` branches.

### CORE-124: `cdist` permits cross-GPU operands without validating ordinals

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/ops/tensor_ops.rs:558-594`

The CUDA guard checks only that both operands are CUDA tensors. It does not
require the same CUDA ordinal before passing both handles to one backend
kernel. Inputs on different GPUs can therefore reach a kernel launched for one
device with a pointer owned by another device, producing backend errors or
invalid device-memory access.

**Recommendation:** Enforce exact device equality before dispatch and add
cross-ordinal negative tests for every binary GPU operation.

### CORE-125: Safe indexing APIs trust contradictory index data and shape metadata

- **Severity:** Critical
- **Confidence:** Confirmed
- **Affected code:** `src/ops/indexing.rs:118-145`,
  `src/ops/indexing.rs:175-224`, `src/ops/indexing.rs:260-285`,
  `src/ops/indexing.rs:339-401`, `src/ops/indexing.rs:494-560`,
  `src/ops/indexing.rs:639-702`

`gather`, `scatter`, `scatter_value`, and `scatter_add` accept index values as
one slice and its claimed shape as a separate slice, but never require
`index.len() == product(index_shape)`. CPU loops index the slice according to
the claimed product and panic when it is shorter. More seriously, every CUDA
fast path runs before `validate_gather_shapes`; the kernels launch for the
shape-derived element count and read past a short uploaded index buffer.
Scatter-family CUDA paths can likewise read past `src`, and no CUDA path checks
index values before using them as device-memory offsets. Kernel safety comments
incorrectly state that the core validator already rejected out-of-range
indices.

**Recommendation:** Validate index rank, exact index length, bounds, source
shape, and source length before any device dispatch. Treat the host slice and
shape as one checked logical tensor, and add malformed-metadata tests on CPU
and CUDA.

### CORE-126: Gather/scatter shape validation omits every non-indexed dimension

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/ops/indexing.rs:118-145`,
  `src/ops/indexing.rs:260-285`, `src/ops/indexing.rs:403-430`,
  `src/ops/indexing.rs:562-579`, `src/ops/indexing.rs:704-731`

Despite documenting PyTorch's per-dimension constraints,
`validate_gather_shapes` checks only equal rank and index-value bounds. It
allows an index dimension other than `dim` to exceed the input, then CPU
coordinate flattening indexes outside the input/output buffer. It also allows
smaller non-`dim` dimensions, but CUDA kernels factor `outer` and `inner` from
the full input shape and therefore launch for a larger layout than the index
and output actually have. This is the same underlying dimensional-assumption
failure as CORE-112, but in the separate autograd-aware public indexing API.

**Recommendation:** Enforce the documented per-axis constraints and execute
using index/output coordinates rather than assuming only the selected axis can
differ. Add larger- and smaller-non-axis parity cases for each operation.

### CORE-127: Scatter reads the wrong source elements when `src` is larger than `index`

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/ops/indexing.rs:405-430`,
  `src/ops/indexing.rs:706-731`, `src/grad_fns/indexing.rs:562-680`,
  `src/grad_fns/indexing.rs:700-810`

The scatter APIs permit `src.numel() >= index.numel()` but consume
`src_data[i]` or device `src[t]` as a flat prefix. PyTorch maps each index
coordinate to the same coordinate in `src`. For example, with index shape
`[2, 1]` and source shape `[2, 3]`, the two consumed source values are at
coordinates `[0,0]` and `[1,0]`, not flat offsets zero and one. The backward
path compounds this by returning a gradient shaped like `index`, rather than
the original larger `src`.

**Recommendation:** Validate source rank and per-axis sizes, address source
values by index coordinates, and return a full source-shaped gradient with
zeros outside the consumed region. Test larger source dimensions before,
after, and across the scatter axis.

### CORE-128: CUDA gather/scatter backward is hard-wired to f32 for f64 tensors

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/indexing.rs:472-525`,
  `src/grad_fns/indexing.rs:562-630`, `src/grad_fns/indexing.rs:700-761`

The forward paths explicitly support both f32 and f64 CUDA tensors. Their
backward implementations unconditionally call `scatter_add_1d_f32`,
`masked_zero_f32`, and `index_select_1d_f32` on `grad_output` handles,
regardless of `T`. An f64 graph therefore reaches kernels with the wrong dtype
contract and cannot produce a valid f64 gradient.

**Recommendation:** Dispatch every backward primitive by `T::dtype()` and
provide matching f64 kernels, or reject f64 forward autograd until its VJP is
implemented. Add end-to-end f64 CUDA backward tests for gather, scatter, and
scatter_add.

### CORE-129: CUDA indexing gradients lose destination precision above 2^24

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/indexing.rs:45-121`,
  `src/grad_fns/indexing.rs:472-525`, `src/grad_fns/indexing.rs:562-630`,
  `src/grad_fns/indexing.rs:700-761`

The indexing backward helpers compute integer flat offsets, cast them to
`f32`, upload them as f32, and feed them to indexing kernels. Flat offsets
above `2^24` are not all exactly representable, so valid gradients for large
tensors can be scattered to or gathered from neighboring elements. This is a
broader instance of the precision hazard noted for sparse SGD in CORE-079.

**Recommendation:** Give all indexing kernels an integer index ABI and keep
offsets as checked i64/usize until dispatch. Add boundary tests around `2^24`
and large multidimensional flat offsets.

### CORE-130: `masked_select` implements same-numel reshaping instead of broadcasting

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/ops/indexing.rs:912-988`

`masked_select` validates only that mask and input have equal element counts.
It therefore accepts incompatible shapes such as input `[2, 2]` with mask
`[4]` and pairs them by flat position. Conversely, it rejects broadcastable
masks such as `[1, 2]` for input `[2, 2]`. PyTorch requires the input and mask
to be broadcastable and selects from their broadcasted layout.

**Recommendation:** Infer and validate the broadcast shape, run selection over
that layout on CPU and CUDA, and make backward reduce broadcasted input
dimensions correctly. Add both equal-numel-incompatible and
different-numel-broadcastable tests.

### CORE-131: SIMD elementwise helpers skip shape validation and return partially computed tensors

- **Severity:** High
- **Confidence:** Confirmed (probe-executed)
- **Affected code:** `src/ops/elementwise.rs:43-58`, `src/ops/elementwise.rs:87-102`

The public, fallible `simd_add_f32`, `simd_mul_f32`, `simd_add_f64`, and
`simd_mul_f64` perform no shape or length validation. The output is sized to
`a`, and the underlying ferray dispatch guards lengths only with
`debug_assert_eq!` before zipping, which silently stops at the shortest slice.
In release builds, adding a `[2, 3]` tensor to a `[2]` tensor returns `Ok`
with shape `[2, 3]` whose tail is the zero-initialized output buffer; in debug
builds the dispatch panics inside a `Result`-returning API. PyTorch either
broadcasts or rejects mismatched shapes; it never returns partially computed
data.

**Recommendation:** Validate equal shapes (or at least equal lengths) and
return a shape error before invoking the kernel in all four functions.

### CORE-132: CPU elementwise, reduction, and cumulative kernels reject valid non-contiguous views

- **Severity:** High
- **Confidence:** Confirmed (probe-executed for add, sigmoid, cumsum;
  identical mechanism in the rest)
- **Affected code:** `src/ops/elementwise.rs:145-149`, `:224-228`, `:307-311`,
  `:391-395`, `:688-689`, `:727-729`, `:866-879`, `:970-979`, `:1034-1245`;
  `src/ops/cumulative.rs:104`, `:177`, `:268`, `:380`, `:453`;
  call sites in `src/special.rs:1982-2333` and
  `src/grad_fns/arithmetic.rs:1401`, `:1528`, `:1632`, `:1791`, `:1945`, `:3635`

`Tensor::data()` rejects non-contiguous tensors, and a large family of CPU
kernels call it without a contiguity fallback: the same-shape fast paths of
`fast_add`/`fast_sub`/`fast_mul`/`fast_div`, `fast_sigmoid`/`fast_tanh`,
`scalar_map`, every reduction in `ops/elementwise.rs` (`sum`, `mean`,
`nansum`, `nanmean`, `logsumexp`, `logsumexp_dim`), the CPU branches of all
five cumulative forwards, and the CPU branch of `unary_map` — through which
roughly twenty `special.rs` operations and the CPU forwards of `neg`, `abs`,
`sqrt`, `rsqrt`, `reciprocal`, and `pow` also fail. Probes confirm that adding
two transposed CPU tensors through the public `add` path, `sigmoid` of a
transpose, and `cumsum` of a transpose all return errors. PyTorch accepts
arbitrary strides for every one of these operations. The contrast is internal
as well: `fast_exp`/`fast_log`, `binary_map`, and the cumulative GPU branches
all materialize via `contiguous()` first.

**Recommendation:** Materialize non-contiguous inputs at the top of each CPU
kernel (or make `unary_map`/`scalar_map` fall back to `data_vec()`), mirroring
the pattern already used by `fast_exp`, `binary_map`, and the GPU branches.

### CORE-133: `logcumsumexp` NaN-poisons scan lines containing infinities

- **Severity:** High
- **Confidence:** Confirmed (probe-executed)
- **Affected code:** `src/ops/cumulative.rs:456-491`

The running-max rescaling computes `(x - m).exp()`; when the element and
running max are both infinite, `inf - inf` is NaN and the accumulator is
poisoned for the rest of the line. `logcumsumexp([-inf, 0])` returns
`[NaN, NaN]` where PyTorch returns `[-inf, 0]`, and `[0, inf]` returns
`[0, NaN]` where PyTorch returns `[0, inf]`. `-inf` is the standard masking
value in log-prob workflows, so a masked first position destroys all
downstream cumulative values. The doc comment claims the implementation
mirrors PyTorch's `_log_add_exp_helper`, but that helper's equal-infinity
guards were dropped in translation.

**Recommendation:** Port the `_log_add_exp_helper` special cases so equal
infinities pass through instead of entering the `exp(x - m)` rescaling.

### CORE-134: `logsumexp` and `logsumexp_dim` return NaN for +inf inputs and all-(-inf) slices

- **Severity:** High
- **Confidence:** Confirmed (probe-executed)
- **Affected code:** `src/ops/elementwise.rs:1154-1183`, `:1215-1235`

Both reductions subtract the max before exponentiating without neutralizing
infinite maxes, so `(inf - inf).exp()` yields NaN: `logsumexp([1, inf])`
returns NaN where PyTorch returns `inf` (ATen masks `|max| == inf` to zero
before the exp-sum). The scalar variant guards only the all-`-inf` case;
`logsumexp_dim` guards neither, so a fully masked row returns NaN instead of
`-inf`. Both docstrings claim to match `torch.logsumexp`; the unit tests cover
only finite inputs.

**Recommendation:** Mask infinite per-slice maxes to zero for the subtraction
step (restoring them in the result), matching ATen, in both functions.

### CORE-135: `vexp_f32` clamps the domain: `exp(-inf)` returns 1.18e-38 and near-threshold inputs fabricate +inf

- **Severity:** Medium
- **Confidence:** Confirmed (probe-executed)
- **Affected code:** `src/ops/elementwise.rs:480-502`, used by `fast_exp` at
  `:548-586`

The vectorized exp kernel clamps inputs to `[-87.34, 88.72]` with no special
value guards. Probes: `fast_exp(-inf)` returns `1.1754997e-38` instead of
`0.0`; `fast_exp(-100)` returns the same clamp value instead of a subnormal;
`fast_exp(88.5)` returns `+inf` instead of the finite `2.73e38` because the
exponent-bit trick overflows once `round(x·log2e)` reaches 128. The file
contains a correct scalar kernel (`fast_exp_f32`) with exactly the missing
guards, but it is used only by `fast_sigmoid`/`fast_tanh`; the test asserting
`exp(-inf) == 0` tests that unused kernel, not the one `Tensor::exp` reaches
on CPU.

**Recommendation:** Add the NaN/overflow/underflow guards from `fast_exp_f32`
to `vexp_f32` and clamp the exponent integer so near-threshold inputs do not
fabricate infinity.

### CORE-136: CUDA `cummax`/`cummin` encode result indices as f32, corrupting positions above 2^24

- **Severity:** Medium
- **Confidence:** Strong
- **Affected code:** `src/ops/cumulative.rs:238-247`, `:352-360`

The f32 CUDA kernels return argmax/argmin positions in an f32 buffer that is
read back via `v as usize`. Integers above 2^24 are not exactly representable
in f32, so any scan line whose running extreme lies past 16.7M elements
reports a silently rounded index, and the cumulative backward scatters
gradients to the wrong source position. PyTorch stores these indices as int64.
This is the cumulative-op instance of the f32-encoded-index class
(CORE-077/079/129).

**Recommendation:** Emit indices in an integer (or f64) buffer, or enforce a
checked `dim_size <= 2^24` limit before dispatch.

### CORE-137: `fast_exp`/`fast_log`/`fast_sigmoid`/`fast_tanh` error on CUDA while `fast_sin`/`fast_cos` fall back

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/ops/elementwise.rs:548-554`, `:591-597`, `:688-689`,
  `:727-729`; contrast `:780-782`, `:827-829`

`fast_sin`/`fast_cos` carry an explicit CUDA guard (added for #796) that
routes device tensors through `unary_map`; the four sibling kernels in the
same file have the identical unconditional `data()?` and never received the
guard, so the public functions return `GpuTensorNotAccessible` for CUDA
inputs. The `grad_fns` wrappers dispatch CUDA earlier, so only direct public
calls hit it, but these are exported APIs with no documented CPU-only
contract, and the file is internally inconsistent.

**Recommendation:** Apply the same CUDA fallback guard to all four functions
or document and reject CUDA explicitly.

### CORE-138: Safe public `mm_raw` family performs unchecked indexing — out-of-bounds reads from safe code

- **Severity:** Critical
- **Confidence:** Confirmed (debug UB-precondition abort reproduced)
- **Affected code:** `src/ops/linalg.rs:1088-1247`, `:1253-1412`, `:1417-1570`

`mm_raw`, `mm_raw_bt`, and `mm_raw_at` are safe `pub fn` in a public module,
but their small-matrix paths (max dimension ≤ 128) index the input slices with
`get_unchecked` whose SAFETY comments rely on a "function contract"
(`a_data.len() >= m*k`, `b_data.len() >= k*n`) that is never checked. Calling
`mm_raw(&[1.0; 4], &[1.0; 4], 64, 64, 64)` from safe code performs genuine
out-of-bounds heap reads in release builds (reproduced as a debug
UB-precondition abort). The large-matrix path happens to panic safely because
faer validates lengths, making the unsoundness size-dependent. Under
`--features mkl` the same unchecked dimensions feed Fortran `sgemm_`/`dgemm_`.
These functions back the entire matmul stack, so any internal shape
miscalculation escalates to UB rather than a panic.

**Recommendation:** Validate slice lengths against `m`/`k`/`n` at entry (or
mark the trio `unsafe fn` with documented preconditions) before the unchecked
loops and FFI calls.

### CORE-139: Broadcast matmul panics on zero-sized batch dimensions

- **Severity:** High
- **Confidence:** Confirmed (probe-executed)
- **Affected code:** `src/ops/linalg.rs:519`, `:533-554`, `:608-618`

`broadcast_matmul` computes `batch_size = product(batch_shape).max(1)`. A
zero-sized batch dimension (e.g. `(0,2,3) @ (0,3,2)`, which PyTorch handles by
returning an empty `(0,2,2)` tensor) makes the product 0, but `.max(1)` forces
one loop iteration, and `batch_linear_index` evaluates `remaining % 0`,
panicking with a remainder-by-zero inside a `Result`-returning API. The path
is reachable from `Tensor::matmul` via the grad_fns broadcast fallback as well
as from the public `ops::linalg::matmul`.

**Recommendation:** Drop the `.max(1)` so a zero batch size skips the loop and
returns the correctly shaped empty output.

### CORE-140: f16 and bf16 matmul accumulate in storage precision instead of PyTorch's f32 opmath

- **Severity:** High
- **Confidence:** Confirmed (probe-measured 0.58% relative error at k=128)
- **Affected code:** `src/ops/linalg.rs:1144-1164`, `:1300-1320`, `:1463-1482`
  (f16 in the generic branch), `:641-647`, `:1649-1655`, `:1680-1686`,
  `:1740-1753` (`dot`, `mv`, `vm`, `bmm` for both f16 and bf16)

The small-matrix `mm_raw*` paths special-case bf16 with an f32 accumulator but
let `half::f16` fall into the generic branch, which accumulates directly in
f16 (11-bit mantissa, max 65504): probes show 0.58% relative error at k=128
and intermediate-sum overflow to `inf` for dots whose partial sums exceed
65504. `dot`, `mv`, `vm`, and `bmm` accumulate in `T` for all dtypes with no
reduced-precision special case at all, so even bf16 — deliberately fixed in
`mm_raw` — suffers ~8-bit accumulation through `ops::bmm` and the
vector-matmul arms. PyTorch accumulates `Half`/`BFloat16` CPU matmuls in
`opmath_type = float`.

**Recommendation:** Extend the f32-accumulator route to f16 in the `mm_raw*`
small paths and give `dot`/`mv`/`vm`/`bmm` an opmath accumulator for f16/bf16.

### CORE-141: `broadcast_matmul` silently computes CUDA matmuls on the host and accepts mixed-device operands

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/ops/linalg.rs:472-573`; contrast `:641-642`,
  `:1612-1614`, `:1645-1646`, `:1733-1734`

The ≥3-D arm pulls both operands through `data_vec()` (a silent
device-to-host copy), runs the batched GEMM on CPU, and uploads the result to
`a`'s device. `b`'s device is never validated, so CPU/CUDA and cross-ordinal
mixes succeed silently where PyTorch raises a device error. The dispatcher is
also internally inconsistent: the 1-D/2-D arms use `data()` and fail loudly on
GPU tensors, so the same public API errors or silently demotes compute
depending only on rank.

**Recommendation:** Validate full device equality (including ordinal) at
`matmul` entry and make the ≥3-D arm either reject GPU tensors or dispatch a
real GPU kernel.

### CORE-142: `dot`/`mv`/`vm`/`bmm`/`transpose` reject non-contiguous views while `mm` materializes them

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/ops/linalg.rs:641-642`, `:1645-1646`, `:1676-1677`,
  `:1733-1734`, `:1768`; contrast `:1584-1594`

`mm` explicitly materializes non-contiguous inputs, but its siblings call
`data()?` directly and therefore fail on any strided view — including rank-3
attention patterns such as `bmm(a.transpose(1,2)?, b)` that PyTorch handles
natively. The same logical matmul succeeds or fails depending on which
rank-arm it lands in, contradicting the file's claim of following
`torch.matmul` semantics exactly.

**Recommendation:** Apply `mm`'s materialization pattern (or `data_vec()`) in
`dot`, `mv`, `vm`, `bmm`, and `transpose`.

### CORE-143: MKL dispatch casts matrix dimensions to i32 unchecked

- **Severity:** Medium
- **Confidence:** Strong
- **Affected code:** `src/ops/linalg.rs:778-817`, `:846-885`, `:919-937`,
  `:962-979`, `:1023-1039`, `:1064-1081`

Under `--features mkl`, every GEMM passes `m as i32`, `n as i32`, `k as i32`,
and the leading dimensions to Fortran BLAS with no range check. A dimension of
2^31 wraps negative (BLAS xerbla abort); 2^32 wraps to zero, making MKL
compute nothing while the function returns its zero-initialized result as
success. PyTorch checks that dimensions fit the BLAS integer type before
dispatch. This is a concrete validation-bypass instance of the CORE-007 class
at an FFI boundary where the failure is silent wrong results.

**Recommendation:** Verify all dimensions fit `i32` before MKL dispatch and
fall back to the faer path (or error) when they do not.

### CORE-144: `lu` returns the inverse of its documented permutation

- **Severity:** High
- **Confidence:** Confirmed (ferray convention pinned by running its own
  3-cycle pivot test)
- **Affected code:** `src/linalg.rs:1078-1128`; non-discriminating tests at
  `src/linalg.rs:3167-3206` and `tests/conformance_linalg.rs:1834-1858`

`lu` documents `A = P L U` ("Mirrors `torch.linalg.lu`") and passes ferray's
`(P, L, U)` straight through — but ferray's contract is `P A = L U`, i.e. the
transpose of torch's `P`. For any pivot sequence composing to a
non-involutory permutation (any 3-cycle; extremely common), the returned
triple satisfies `P L U = P² A ≠ A`. Both existing tests use fixtures whose
pivoting is a single swap (an involution, where the conventions coincide), so
the suite locks in only the non-discriminating case. The backward node
multiplies by `P^T`, which is numerically correct under the true relation, but
its justifying comment repeats the wrong claim.

**Recommendation:** Transpose ferray's `P` before returning so `A = P L U`
holds, and add a non-involutory (3-cycle) pivot fixture to both test suites.

### CORE-145: `cholesky_ex`/`inv_ex`/`solve_ex` convert every error into `info=1` with fabricated outputs

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/linalg.rs:2711-2736`, `:2739-2763`, `:2766-2793`

All three `_ex` variants map **every** `Err` — shape, dtype, and device errors
included — to `(zeros, info=1)`. PyTorch's `_ex` family suppresses only
numerical (LAPACK info) failures; structural errors still raise. Here
`cholesky_ex` of a non-square `[2, 3]` matrix fabricates a bogus `[2, 2]` zero
result, a 1-D `[7]` input yields `[7, 7]` zeros, and `solve_ex` swallows
`DeviceMismatch`. Additionally, `info` is always the constant 1 rather than
the failing leading-minor index the documentation promises, and the fallback
zeros are constructed on CPU even for CUDA inputs (the success-path `info`
scalar is also always CPU).

**Recommendation:** Convert only numerical failures into `info != 0`,
propagate structural errors, report the actual minor index, and allocate
fallback values and `info` on the input device.

### CORE-146: Decomposition and solver autograd is silently severed on CUDA and absent for a dozen CPU ops

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/linalg.rs:175-201`, `:278-306`, `:442-465`,
  `:524-540`, `:695-715`, `:809-826`, `:1155-1174`, `:582-607` (CUDA gates);
  `src/linalg.rs:1257`, `:1430`, `:1459`, `:1487`, `:1614`, `:1651`, `:2159`,
  `:2279`, `:2330`, `:2435`, `:2520` (CPU forward-only ops)

Every grad-aware linalg op gates its differentiable wrapper on
`!input.is_cuda()`, so a CUDA tensor with `requires_grad=true` flowing through
`svd`, `solve`, `qr`, `cholesky`, `eigh`, `eigvalsh`, `lu_factor`, or
`matrix_norm` produces outputs constructed with `requires_grad=false`: the
graph edge disappears with no error and `.grad` silently stays empty, where
PyTorch differentiates all of these on CUDA. Separately, `svdvals`,
`matrix_power`, `tensorsolve`, `tensorinv`, `matrix_rank`, `cond`,
`solve_triangular`, `ldl_factor`/`ldl_solve`, `matrix_exp`, and
`householder_product_full` are differentiable in PyTorch but always return
detached tensors here, even on CPU. The module header's non-grad list is stale
in both directions.

**Recommendation:** Return an explicit unsupported-autograd error when a
tracking input reaches a forward-only path (or fall back to CPU compute with
the differentiable wrapper), and refresh the module documentation.

### CORE-147: `cross` panics on tensors with a zero-sized non-cross dimension

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/linalg.rs:1782-1832`

The base-offset enumeration pushes an offset for the current multi-index
before checking for zero-sized dimensions, so an input of shape `[0, 3]`
produces one base offset while `groups = 0`. The `debug_assert_eq!` aborts
debug builds; release builds proceed to index an empty data slice and panic
inside a `Result`-returning API. PyTorch returns an empty tensor of the input
shape. The differentiable wrapper shares the forward, so autograd is equally
affected.

**Recommendation:** Short-circuit `numel == 0` after validation and return an
empty tensor of the input shape.

### CORE-148: `matrix_exp` integer-shift overflow for extreme norms

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/linalg.rs:2653-2660`

The scaling-and-squaring count is computed as `1u64 << s`. For matrices whose
infinity norm exceeds `θ13 · 2^63` (a single `1e20` entry suffices) or
contains `inf`, `s` reaches 64 or more: debug builds panic on the shift;
release builds mask the shift amount mod 64, producing a tiny scale and
evaluating the Padé approximant far outside its convergence region — silently
wrong results. PyTorch degrades gracefully toward `inf`.

**Recommendation:** Clamp `s` and compute the scale as `2f64.powi(s)` instead
of an integer shift.

### CORE-149: CUDA `solve` dispatches without validating the right-hand side's shape

- **Severity:** Medium
- **Confidence:** Strong
- **Affected code:** `src/linalg.rs:285-306`

The CUDA branch computes `nrhs` from `b` and immediately calls the backend
with raw buffers; nothing checks `b.shape()[0] == n` or restricts `b.ndim()`
to 1 or 2. An undersized `b` hands the kernel a buffer it reads `n` elements
from (out-of-bounds device read); a 3-D `b` is silently treated as 2-D with a
mislabeled result shape. The CPU path is safe only because ferray validates
internally. PyTorch validates `A.shape[-1] == B.shape[-2]` on every device.

**Recommendation:** Hoist the dimension and row-count checks above the device
branch so CPU and CUDA validate identically.

### CORE-150: `eigh` eigenvector sign canonicalization is applied only on the CPU path

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/linalg.rs:699-716` (CUDA, no canonicalization) vs
  `:718-737`, `:747-791` (CPU contract and helper)

The CPU path post-processes eigenvectors through
`canonicalize_eigenvector_signs`, documented in-file as the crate's "STABLE,
REPRODUCIBLE sign contract"; the CUDA path returns raw cuSOLVER output with no
canonicalization, so the same matrix yields sign-flipped columns depending on
device. The file establishes a contract stronger than gauge freedom and
honors it on one device only, silently breaking CPU/GPU comparisons that rely
on the documented convention.

**Recommendation:** Canonicalize the CUDA eigenvectors as well, or scope the
documented sign contract explicitly to CPU outputs.

### CORE-151: `split`/`chunk` CUDA fast path ignores strides and storage offset

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/methods.rs:1751-1798`; reached via
  `src/methods.rs:976-983`, `:1675-1710`

The GPU fast path in `split_t` is gated only on device, dtype, and backend
presence — there is no `is_contiguous()` or `storage_offset() == 0` check. It
passes the raw base buffer (`gpu_handle()`) plus logical-shape-derived
extents to `strided_split_f32`, whose trait signature accepts no source
strides or base offset. Calling `.split()` or `.chunk()` on any CUDA f32 view
— a transpose, permute, or `narrow` result — therefore silently returns
chunks gathered from the wrong elements, while the CPU path (via `data_vec`)
is correct, so CPU and CUDA disagree and CUDA is wrong. This is the same
`gpu_handle()`-drops-view-geometry class that #1657 fixed only in
`contiguous_t`. GPU conformance tests upload fresh contiguous tensors only.
Secondary: a kernel error propagates instead of using the documented CPU
fallback.

**Recommendation:** Gate the fast path on contiguous, offset-zero inputs
(materializing otherwise), mirror the #1657 fix, and fall back to CPU on
kernel errors.

### CORE-152: `chunk` over a zero-sized dimension returns zero chunks

- **Severity:** Medium
- **Confidence:** Confirmed (verified against live torch 2.11)
- **Affected code:** `src/methods.rs:1699-1709`

When the chunked dimension has size zero, the split-size loop never executes
and `chunk` returns an empty vector. ATen special-cases this and returns
`chunks` empty tensors — `torch.empty(0, 3).chunk(3, dim=0)` yields three
`(0, 3)` tensors — and PyTorch guarantees `chunk` never returns an empty
tuple. Translated code that destructures (`q, k, v = x.chunk(3, -1)`) breaks
on empty batches.

**Recommendation:** Emit `chunks` zero-sized split entries when the dimension
is empty, matching ATen.

### CORE-153: `t()` and `transpose` reject rank-0/1 tensors that PyTorch accepts

- **Severity:** Medium
- **Confidence:** Confirmed (verified against live torch 2.11)
- **Affected code:** `src/methods.rs:640-642`, `:790-803`;
  `src/grad_fns/shape.rs:397-406`

`Tensor::t()` delegates to `transpose_2d`, which errors for any rank other
than 2, while `torch.t` documents and implements pass-through for 0-D and 1-D
tensors. `transpose(0, 0)` on a scalar likewise errors where PyTorch accepts
it. The `usize` dimension parameters also make PyTorch's negative-dim
convention inexpressible, unlike the `isize`-taking siblings in the same file.

**Recommendation:** Return `self.clone()` from `t()` for ranks below 2 and
accept dimension 0 on rank-0 inputs in `transpose`.

### CORE-154: `narrow` bounds validation can be bypassed by usize overflow

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/methods.rs:1453`, `:1470`

The bounds check `start + length > dim_size` uses raw addition; with
`start = usize::MAX`, the sum wraps in release builds, passes the check, and
the subsequent offset computation wraps again, producing a garbage offset in a
`stride_view` whose constructor performs no validation. Debug builds panic
inside the fallible API. The non-contiguous gather and GPU consumers
downstream have no saturating guard. This is a concrete validation-bypass
instance of CORE-007 in which the overflow defeats the only bounds check the
operation has.

**Recommendation:** Use checked addition/multiplication in the validation and
offset computation, returning `InvalidArgument` on overflow.

### CORE-155: `irfft`/`hfft` underflow-panic on zero-length frequency axes, even with explicit `n`

- **Severity:** High
- **Confidence:** Confirmed (panic reproduced)
- **Affected code:** `src/fft.rs:466-467`, `:1172`

`irfft_norm` computes `n.unwrap_or(2 * (half_n - 1))` after rank validation
but before any emptiness check. A `[0, 2]` input (zero-length frequency axis)
makes `half_n - 1` underflow — and because `unwrap_or` evaluates its argument
eagerly, the panic fires even when the caller passes an explicit `n`, a case
PyTorch handles by zero-padding and returning zeros. With `n=None` PyTorch
raises a structured error, never a crash. The same unchecked expression sits
in `hfft_norm`'s CUDA gate, and ferray-fft duplicates it internally, so the
wrapper must guard before delegating.

**Recommendation:** Reject `half_n == 0 && n.is_none()` with `InvalidArgument`
and compute the default lazily so an explicit `n` never evaluates the
underflowing expression; apply the same guard in `hfft_norm`.

### CORE-156: `fftshift`/`ifftshift` roll the interleaved complex pair axis, silently corrupting complex spectra

- **Severity:** High
- **Confidence:** Confirmed (probe-executed)
- **Affected code:** `src/fft.rs:1432-1445`, `:1451-1464`

The module convention stores complex tensors with a trailing dimension of
size 2, and every transform entry point strips that axis before resolving
`dim` — except `fftshift`/`ifftshift`, which delegate the full shape to
ferray's all-axes roll. With `axes=None` (or `-1`) the re/im pair axis is
rolled by one, swapping real and imaginary components: the canonical
`fftshift(fft(x))` returns numerically wrong data with the correct shape.
Probes confirm divergence from torch on a length-4 signal. The conformance
fixtures exercise real dtypes only, so the complex-encoded case is untested.
The docstring's "Matches `torch.fft.fftshift`" is false for complex inputs.

**Recommendation:** Exclude the trailing pair axis from `axes=None` resolution
and resolve negative axes against the signal layout, or document the functions
as real-only.

### CORE-157: f32 FFTs return errors where PyTorch returns infinities

- **Severity:** Medium
- **Confidence:** Confirmed (probe-executed)
- **Affected code:** `src/fft.rs:709-734`

The CPU path computes in f64 and casts back through `numeric_cast::cast`,
whose saturation guard rejects finite f64 values that overflow f32. Any
spectrum bin exceeding `f32::MAX` — trivially reachable since the DFT sums n
inputs — turns the whole transform into `InvalidArgument`, where PyTorch
computes natively in f32 and returns `inf` bins. The general-purpose fallible
cast is the wrong tool for a value-domain conversion whose torch contract is
saturate-to-inf.

**Recommendation:** Use a direct float-to-float conversion in the array-to-
tensor bridges so finite overflow saturates to ±inf.

### CORE-158: CUDA FFT fast paths promote zero-sized batches to batch 1

- **Severity:** Medium
- **Confidence:** Strong
- **Affected code:** `src/fft.rs:215`, `:322`, `:399`, `:481`, `:1176`,
  `:1240`

Every 1-D CUDA fast path computes `batch_size = product(batch).max(1)`. The
empty-slice product is already 1, so the only effect of `.max(1)` is to turn a
genuine zero-sized batch into 1, asking the backend to transform data from an
empty GPU buffer. The cuFFT wrapper's length validation converts this into a
`ShapeMismatch` error — but the same valid input succeeds (returns empty) on
the CPU path, so CPU and CUDA disagree where PyTorch returns an empty tensor
on both.

**Recommendation:** Return an empty same-shape tensor when the batch size is
zero instead of clamping to 1.

### CORE-159: `fftn`/`ifftn` CUDA axes-set construction underflows for over-long axis lists

- **Severity:** Medium
- **Confidence:** Strong
- **Affected code:** `src/fft.rs:838-840`, `:954-956`

The innermost-axes check constructs `(spatial_ndim - r)..spatial_ndim` where
`r` is the user-supplied axes count, with no validation that
`r <= spatial_ndim`. An axes list longer than the spatial rank (including
duplicate axes, which torch rejects cleanly) underflows: debug/test builds
panic inside a fallible API; release builds wrap to an empty range and fall
through to the CPU path's proper error.

**Recommendation:** Use `checked_sub` (or validate `r` and axis bounds) before
constructing the range.

### CORE-160: `fftfreq`/`rfftfreq` reject `n = 0` and `d = 0` edge cases PyTorch defines

- **Severity:** Medium
- **Confidence:** Strong
- **Affected code:** `src/fft.rs:1403-1422`

Both helpers inherit ferray's stricter domain: `n == 0` and `d == 0.0` return
errors, where `torch.fft.fftfreq(0)` returns an empty tensor and `d = 0`
produces ±inf-valued bins. The docstrings claim parity with torch.

**Recommendation:** Short-circuit `n == 0` to an empty tensor and either pass
`d = 0` through or document the stricter domain.

### CORE-161: Two-input CPU einsum silently drops summation over lone indices

- **Severity:** High
- **Confidence:** Confirmed (probe-executed)
- **Affected code:** `src/einsum.rs:1242-1254`, `:1492-1514`, `:1278`,
  `:1335-1343`

A subscript appearing in operand A only and absent from the output is pushed
into `free_a_chars` with a comment claiming it "will be summed implicitly" —
nothing ever sums it; the output-permute step returns the index-0 slice along
that axis instead of the sum. A lone subscript in operand B is dropped from
every group, so B is always read at index 0 along that axis. Probes:
`einsum("ij,j->j")` returns `[10, 200]` where torch returns `[40, 600]`;
`einsum("i,ij->i")` and `"ab,cd->ad"` are similarly wrong. The GPU path
(`reduce_lone_axes`) handles this correctly, so CPU and CUDA disagree and CPU
is silently wrong through `einsum`, `einsum_differentiable`, and
`Tensor::einsum`. When the lone dimension has size zero, the `.max(1)` at
line 1278 additionally drives a remainder-by-zero panic.

**Recommendation:** Pre-reduce lone axes on the CPU path exactly as the GPU
path's `reduce_lone_axes` does, which also removes the zero-size panic.

### CORE-162: Einsum backward panics when an operand has repeated subscripts

- **Severity:** High
- **Confidence:** Confirmed (probe-executed)
- **Affected code:** `src/einsum.rs:1894-1919`, `:1505-1513`, `:84-154`

`EinsumBackwardTwo` derives gradient equations by textual swap, producing
equations with repeated *output* subscripts (e.g. grad of `"ii,j->ij"` w.r.t.
A becomes `"ij,j->ii"`), which the implementation neither rejects nor
handles: the CPU permute step indexes past the intermediate buffer and panics
during `.backward()`. On CUDA the same backward returns an `Internal` error.
Beyond the panic, the textual swap is mathematically wrong for repeated-index
operands — the true gradient requires a diagonal-embed scatter that no swapped
equation can express, which torch implements.

**Recommendation:** Compute repeated-subscript gradients via deduped
subscripts followed by a diagonal embed, and make `parse_equation` reject
repeated output subscripts as torch does.

### CORE-163: Stored einsum equations are not whitespace-normalized; backward panics or errors on spaced equations

- **Severity:** High
- **Confidence:** Confirmed (probe-executed)
- **Affected code:** `src/einsum.rs:1597-1614`, `:1648-1654`, `:1850-1859`

`parse_equation` strips spaces, so forwards accept torch-legal equations like
`"ii -> i"`, but `einsum_differentiable` stores the raw string and
`EinsumBackwardSingle` re-parses it by hand without stripping: the output
subscripts become `[' ', 'i']`, panicking in `backward_repeated_index`
(`char_val[&oc]`, no entry for `' '`) or failing with a spurious subset error
for non-repeated equations. torch.einsum is whitespace-tolerant and
differentiable in both cases.

**Recommendation:** Normalize the equation once before storing it in the
grad-fns (or store the `ParsedEquation`), which also fixes the implicit-mode
finding below.

### CORE-164: Implicit-mode einsum equations are not differentiable

- **Severity:** High
- **Confidence:** Confirmed (probe-executed)
- **Affected code:** `src/einsum.rs:1648-1651`, `:1894-1899`

Both backward implementations split the stored equation on `"->"` and default
the output subscripts to the empty string when absent. For implicit-mode
equations the true output is the sorted set of once-occurring labels (the
forward computes this correctly), so `einsum_differentiable("ij,jk", ...)
.backward()` fails with a subscript-count error and `"ji"` fails inside
`view_reshape`; only fully contracting implicit equations work by
coincidence. torch's implicit mode is fully differentiable.

**Recommendation:** Re-derive the implicit output subscripts in both backward
paths (or store the parsed equation).

### CORE-165: Repeated output subscripts are accepted and produce garbage

- **Severity:** Medium
- **Confidence:** Confirmed (probe-executed)
- **Affected code:** `src/einsum.rs:84-154`, `:569-635`

Neither `parse_equation` nor `build_dim_map` rejects an output subscript
appearing twice, which torch refuses with a structured error. Here
`einsum("i->ii", [1,2,3])` succeeds and returns the vector tiled across rows
(the CPU loop silently overwrites the first occurrence's coordinate), while
two-input variants reach the out-of-bounds panic of CORE-162 and GPU paths
fail incidentally inside `permute_t` — garbage, panic, or unrelated error
depending on path.

**Recommendation:** Reject duplicate output subscripts in `parse_equation`.

### CORE-166: Repeated-index einsum backward returns CPU gradients for CUDA inputs

- **Severity:** Medium
- **Confidence:** Strong
- **Affected code:** `src/einsum.rs:1804-1862`

`backward_repeated_index` is pure CPU: it downloads `grad_output` via
`data_vec()` and builds the gradient in CPU storage. The comment justifying
this claims the forward rejects CUDA for these cases, but that is stale —
since #821/#824 the forward succeeds on CUDA via
`einsum_single_repeated_gpu`, so a CUDA parameter ends up with a CPU `.grad`.
PyTorch guarantees gradient device equals parameter device.

**Recommendation:** Upload the constructed gradient to the input's device (or
build it on-device with the forward's diagonal machinery).

### CORE-167: Torch-legal einsum surface rejected: ellipsis, uppercase labels, size-1 broadcasting, n-ary contraction

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/einsum.rs:92`, `:107`, `:140`, `:185-192`,
  `:1559-1566`

Four torch-supported equation classes fail with errors: ellipsis (`"...ij"` —
ubiquitous in attention code), uppercase subscripts, size-1 broadcasting on
shared labels, and equations with more than two operands. All are clean
errors rather than wrong values, but any translated model using these forms
fails at runtime, and the messages ("invalid character") do not name the
unsupported feature.

**Recommendation:** Track as feature work (parser support, broadcast
semantics, pairwise n-ary reduction); at minimum, name the unsupported
feature in the error.

### CORE-168: The re-exported free `einsum` silently detaches autograd

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/einsum.rs:1558-1578`, `src/lib.rs:160`

The crate root re-exports both `einsum` and `einsum_differentiable`. The free
`einsum` never attaches a grad-fn, so a user porting `torch.einsum` who
reaches for the identically named function gets silently detached outputs
even when inputs require gradients; only `Tensor::einsum` routes to the
differentiable variant.

**Recommendation:** Make the exported `einsum` delegate to
`einsum_differentiable`, or stop re-exporting the raw forward under the
torch-colliding name.

### CORE-169: `gammainc`/`gammaincc` return silently truncated partial sums for large `a`

- **Severity:** High
- **Confidence:** Confirmed (numerically measured against the SciPy/torch
  oracle)
- **Affected code:** `src/special.rs:1746-1766`, `:1771-1796`

Both the power series and the Lentz continued fraction cap at 300 iterations
and return the partial result as if converged. The series needs O(√a) terms
near `x ≈ a`, so the cap is exceeded from `a ≈ 1.2e4`: `gammainc(1e5, 1e5)`
returns 0.329 and `gammainc(1e6, 1e6)` returns 0.118 where the true value
(and torch's, via its large-`a` asymptotic branch) is 0.500 — silently wrong
probabilities for routine chi-square/Poisson statistics. In-file tests cover
only `a ≤ 4`.

**Recommendation:** Port torch's large-`a` asymptotic branch (or scale the
iteration caps with √a) and propagate non-convergence instead of returning
partial sums.

### CORE-170: Every special function silently detaches autograd

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/special.rs:1982-2913`; construction sites
  `src/ops/elementwise.rs:866-966`, `src/special.rs:2472-2477`, `:2675-2703`

No function in `special.rs` consults `requires_grad()` or attaches a backward
node; every output is an unconditionally detached leaf, while in PyTorch the
erf family, gamma family, `log1p`/`expm1`/`sinc`/`xlogy`/`entr`/`ndtr`, and
the Bessel/polynomial families (w.r.t. `x`) are differentiable with standard
derivative formulas. A loss containing `lgamma` (any Gamma/Beta NLL) silently
stops training at that node. Distinct from CORE-045/049: there is no backward
at all here, on any device.

**Recommendation:** Attach gradient nodes for the differentiable subset, or
reject tracking inputs explicitly until they exist.

### CORE-171: `erfinv` returns ±inf outside [-1, 1], contradicting its own documentation and torch

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/special.rs:505-510`

The edge ladder uses `>= 1` / `<= -1`, so `erfinv(2.0)` returns `+inf` where
torch (and the function's own rustdoc) specify NaN for `|x| > 1`, with ±inf
reserved for exactly ±1. Out-of-range inputs receive a plausible-looking
infinity instead of the NaN sentinel.

**Recommendation:** Use exact equality for the ±1 cases and return NaN for
`|x| > 1`.

### CORE-172: `beta` drops the sign for negative arguments

- **Severity:** Medium
- **Confidence:** Confirmed (verified against SciPy values)
- **Affected code:** `src/special.rs:1910-1916`, `:2373-2375`

`beta` computes `exp(log_beta)` from `ln|Γ|`, so the result is always
positive, while the declared oracle is signed: `scipy.special.beta(-0.5, 1.5)
= -π` but ferrotorch returns `+π`. The file ships `gammaln_sign_scalar`
implementing exactly the needed sign factor, unused by `beta`.

**Recommendation:** Multiply by the gamma sign factors
(`sgn(a)·sgn(b)/sgn(a+b)`), matching SciPy.

### CORE-173: `digamma` returns huge finite garbage at negative-integer poles

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/special.rs:578-583`, `:629-633`

PyTorch's `calc_digamma` returns NaN at negative-integer poles. The reflection
branch here computes `cot(π·x)` directly; because π is rounded, `sin(π·-1.0)`
is ~1.2e-16 rather than zero, so `digamma(-1.0)` returns -2.57e16 — large
finite garbage that silently poisons downstream sums. Both the f64 and f32
paths share the defect.

**Recommendation:** Guard `x <= 0 && x == floor(x)` with NaN before the
reflection in both paths.

### CORE-174: `lgamma(±inf)` returns NaN instead of +inf

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/special.rs:545-569`

For `+inf` the Lanczos tail computes `inf - inf = NaN`; for `-inf` the
reflection computes `sin(π·-inf) = NaN`, which evades the pole test. C99 and
torch return `+inf` for both. The defect propagates into `multigammaln` and
`log_beta`/`beta` with infinite arguments.

**Recommendation:** Return `T::infinity()` for any infinite input at the top
of `lgamma_scalar`.

### CORE-175: `xlogy(0, NaN)` returns 0 — the NaN-first rule is missing

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/special.rs:679-685`

torch's contract checks `isnan(y)` before the `x == 0` shortcut, so
`xlogy(0, NaN)` is NaN; ferrotorch checks `x == 0` first and returns 0. The
in-file test pins `y ∈ {1, 0, inf}` but not NaN.

**Recommendation:** Return `y` when it is NaN, before the zero branch.

### CORE-176: Chebyshev polynomial families omit torch's closed-form trigonometric branch

- **Severity:** Medium
- **Confidence:** Strong
- **Affected code:** `src/special.rs:2987-3075`, public wrappers `:2711-2762`,
  `:2854-2915`

Upstream evaluates `cos(n·acos(x))` (and sin/half-angle analogues for U/V/W)
for `n > 6`/`n > 8` with `|x| < 1`; the port always runs the three-term
recurrence in native precision. Measured f32 divergence exceeds the crate's
own 1e-5 gate (2.3e-5 at `n=100, x=0.9999`). The GPU kernels mirror the
recurrence, so devices agree with each other but both diverge from torch. The
shifted variants inherit the gap.

**Recommendation:** Add the closed-form branches to the four scalar
evaluators and matching GPU kernels.

### CORE-177: `special.rs` CUDA dispatch policy is three-way inconsistent

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/special.rs:1982-2035`, `:2391-2413` (silent host
  round trip via `unary_map`); `:2040-2042`, `:2343-2375` (leaked
  `GpuTensorNotAccessible`); contrast `:2051-2333`, `:2604-2611`

Within one module, CUDA inputs receive three different treatments: eleven
unary ops (`erf`, `erfc`, `erfinv`, `lgamma`, `digamma`, `log1p`, `expm1`,
`sinc`, `multigammaln`, `mvlgamma`, `gammaln_sign`) silently round-trip
through the host — directly contradicting the module's documented
"no host round trip" rule; five binary ops (`xlogy`, `gammainc`, `gammaincc`,
`log_beta`, `beta`) leak the internal `GpuTensorNotAccessible` error that the
module's own comments identify as the wrong public surface; and the remaining
ops reject cleanly with `NotImplementedOnCuda`. Values are correct in the
round-trip cases, but the device contract is undocumented and inconsistent.

**Recommendation:** Pick one policy — on-device kernels or a clean
`NotImplementedOnCuda` rejection — and apply it to all special ops.

### CORE-178: `PowBackward` omits the `exponent == 0` special case, producing NaN gradients

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/arithmetic.rs:1425-1480`

All three PowBackward branches compute `grad · exp · a^(exp-1)`
unconditionally. PyTorch's `pow_backward` returns zeros for a zero exponent
before evaluating `a^(exp-1)`; ferrotorch instead evaluates `0 · a^{-1} · g`,
which is `0 · inf = NaN` wherever `a == 0`. So `x.pow(0.0).sum().backward()`
produces NaN gradients at every zero element where PyTorch produces exact
zeros, silently poisoning optimizer steps. The forward is pinned by
conformance tests; no test covers backward at exponent zero.

**Recommendation:** Return a zeros tensor of `a`'s shape and device when the
exponent is zero, mirroring upstream.

### CORE-179: Most arithmetic backward nodes and the broadcast gradient reducer are non-differentiable, silently breaking `create_graph`

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/arithmetic.rs:178-336`, `:1188-1212`,
  `:1551-1589`, `:1675-1726`, `:1837-1874`, `:1996-2028`, `:2299-2331`,
  `:2953-3010`, `:3249-3314`, `:3541-3596`

Only `MulBackward` and `PowBackward` have differentiable backward paths; the
backward nodes for div, sqrt, rsqrt, reciprocal, abs, remainder, fmod,
addcmul, and addcdiv compute their VJPs under `no_grad` or via raw vectors,
and `reduce_grad_to_shape` returns a detached tensor for every non-identity
case — severing even MulBackward's differentiable branch whenever the forward
broadcast. Combined with CORE-027's leaf fabrication, second-order gradients
through any of these ops (all of which have nonzero second derivatives) are
silently `None` where PyTorch returns true values. No error or warning exists
anywhere on the path.

**Recommendation:** Implement differentiable backward paths gated like
MulBackward's and make `reduce_grad_to_shape` graph-preserving (it is sum +
reshape, both differentiable), or error loudly when `create_graph=true` meets
a detached node-produced gradient.

### CORE-180: `DivBackward` squares the denominator instead of using PyTorch's nested-division staging

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/arithmetic.rs:1198-1209`

The denominator gradient is computed as `-g·a / (b·b)`, while torch's
`div_tensor_other_backward` computes `-g·((a/b)/b)` specifically because
`b·b` overflows/underflows at half the exponent range: for f32 with
`|b| > ~1.8e19` ferrotorch returns `-0` where torch returns a subnormal, and
for `|b| < ~1e-19` it returns `±inf` where torch is finite; the divergent
range is far larger for f16/bf16. `AddcdivBackward` in the same file already
uses the correct nested staging.

**Recommendation:** Restage as `-g · (a/b) / b` (or reuse the saved forward
output), matching upstream.

### CORE-181: `RemainderBackward` uses naive `floor(a/b)` instead of `div_floor_floating`

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/arithmetic.rs:2013-2025`; the correct
  kernel exists at `:2857-2897`

torch defines remainder's other-gradient via `div(..., rounding_mode="floor")`
= `div_floor_floating` — the fmod-based, fixup-corrected algorithm this very
file reproduces in `floor_divide_inner`, with comments noting that the naive
`div`+`floor` chain produced errors beyond the parity tolerance. The backward
nevertheless computes `floor(a/b)` naively, reintroducing the same rounding
divergence plus the signed-zero and 0.5-fixup gaps at quotient boundaries.
`FmodBackward`'s `trunc(a/b)` is correct (torch's trunc path is the naive
form).

**Recommendation:** Extract the `div_floor_floating` elementwise kernel from
`floor_divide_inner` into a shared helper and use it in `RemainderBackward`.

### CORE-182: `PowBackward`'s higher-order branch builds its exponent tensor on CPU, failing CUDA double-backward

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/arithmetic.rs:1431-1442`

The `create_graph` branch (selected before the GPU branch) constructs the
exponent scalar with `TensorStorage::cpu` and multiplies it against the
CUDA-resident `a^(exp-1)`, hitting `mul`'s device guard: any
`grad(..., create_graph=true)` through a CUDA `pow` fails with an
unrelated-looking `DeviceMismatch`. The non-higher-order GPU branch directly
below performs the missing `.to(device)` hop.

**Recommendation:** Build the exponent tensor on `a`'s device in the
higher-order branch.

### CORE-183: bf16/f16 CUDA forwards succeed but broadcast-gradient reduction and `AbsBackward` always error

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/arithmetic.rs:233-238`, `:3565-3567`

The bf16/f16 CUDA forward kernels (including broadcast variants) attach
grad-fns, but `reduce_grad_to_shape`'s GPU branch handles only f32/f64 and
hard-errors `NotImplementedOnCuda` otherwise — so a bf16 CUDA `x + bias` (the
canonical broadcast bias-add) computes forward and then fails at
`.backward()`. `AbsBackward` similarly errors for non-f32/f64 CUDA although
its own forward succeeds via host round trip. The advertised reduced-precision
CUDA support is therefore unusable for training with broadcasting.

**Recommendation:** Add host-roundtrip fallbacks (download, reduce on CPU,
upload) for non-f32/f64 CUDA gradients in both places.

### CORE-184: `pow` on reduced-precision CUDA dtypes hard-errors while three doc comments cite a nonexistent fallthrough

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/arithmetic.rs:1507-1537`; false claims at
  `:2096-2098`, `:2404-2405`, `:2733-2734`

`pow_inner` takes the GPU path only for f32/f64; all other CUDA dtypes fall
into `scalar_map`, which rejects CUDA inputs outright, so bf16/f16 CUDA `pow`
always fails while torch supports it. Three doc comments in the same file
(remainder, fmod, floor_divide) cite "`pow_inner`'s bf16/f16 fallthrough" as
precedent for a working host fallback — that fallthrough does not exist.

**Recommendation:** Route non-f32/f64 CUDA `pow` through a host round trip
(or fix `scalar_map`), and correct the three comments.

### CORE-185: `addcmul`/`addcdiv` lack the meta-propagation path every sibling binary op has

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/arithmetic.rs:3089-3130`, `:3404-3445`

`add`, `sub`, `mul`, `div`, `remainder`, `fmod`, and `floor_divide` all
short-circuit through meta propagation; `addcmul`/`addcdiv` go straight to
`data_vec()`, which rejects meta tensors, so shape-inference passes over
Adam-style updates error instead of producing the three-way broadcast meta
result. PyTorch has meta kernels for both ops.

**Recommendation:** Add a ternary meta-propagation guard mirroring
`binary_broadcast`.

### CORE-186: 1-D × batched matmul backward returns wrong gradients or errors

- **Severity:** High
- **Confidence:** Confirmed (probe-executed)
- **Affected code:** `src/grad_fns/linalg.rs:642-653`, `:671-823`

The forward correctly implements PyTorch's 1-D promotion for `vec @ ND` and
`ND @ vec`, but `MatmulBackward`'s batched arm never re-promotes the saved
1-D operand. For `a:[2] @ b:[2,2,3]`, the vector gradient is computed by
broadcast-matmuling the squeezed cotangent against every batch element and
sum-reducing the cross terms: a ones-cotangent probe returns `[60, 96]` where
PyTorch returns `[30, 48]`, and non-uniform cotangents produce arbitrary
cross-contamination — silently wrong gradients. The gradient for the other
operand fails outright because `swap_last_two` rejects `ndim < 2`. No test
covers a 1-D × batched backward.

**Recommendation:** Unsqueeze 1-D operands (and the squeezed grad dimension)
before transpose/matmul in the backward, then squeeze the result, mirroring
PyTorch's decomposition.

### CORE-187: `det`/`slogdet` forward fails on singular matrices only when gradients are enabled

- **Severity:** High
- **Confidence:** Confirmed (probe-executed)
- **Affected code:** `src/grad_fns/linalg.rs:2238-2256`, `:2428-2445`

Both differentiable wrappers eagerly compute `inv(A)` at forward time to
stash for backward, so a singular input makes the *forward* return
`Err(SingularMatrix)` when `requires_grad=true`, while the identical call
without gradient tracking succeeds (`det → -0.0`). PyTorch's forward never
fails on singular input regardless of grad mode (its backward has a
non-invertible fallback; slogdet returns `(0, -inf)`). Toggling
`requires_grad` flipping forward success is a semantic divergence; the eager
O(n³) inverse is also wasted whenever backward never runs.

**Recommendation:** Defer the inverse to backward, and on singular input use
the adjugate/zero-gradient fallback torch implements instead of erroring.

### CORE-188: bf16/f16 CUDA matmul attaches a backward node that always errors

- **Severity:** Medium
- **Confidence:** Confirmed (code path; not executed on GPU)
- **Affected code:** `src/grad_fns/linalg.rs:167-174`, `:1587-1620`,
  `:1656-1758`, `:1835-1846`

The dispatcher deliberately ships bf16/f16 CUDA matmul forwards (#1543/GH#25,
including the broadcast `gemm_strided_batched_ex` path) and attaches
`MatmulBackward` when tracking is on, but every backward route for these
dtypes dead-ends in `NotImplementedOnCuda` (MmBackward's dtype gate, or
`bmm`'s rejection on the broadcast arm). A reduced-precision CUDA training
graph builds successfully and fails at the first `.backward()`; the advertised
bf16 path is inference-only without saying so.

**Recommendation:** Implement the bf16/f16 GPU backward with the same GemmEx
kernels, or refuse to attach the grad node at forward time.

### CORE-189: `eig`/`eigvals` backward solves via explicit Gauss-Jordan inverse

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `src/grad_fns/linalg.rs:5850-5924`, used at `:6007`,
  `:6106`

`c_solve` forms `M^{-1}` by Gauss-Jordan elimination and multiplies, where
PyTorch's `linalg_eig_backward` uses a direct LU solve. Explicit
inverse-then-multiply roughly doubles rounding stages and loses accuracy
precisely when the eigenvector matrix is ill-conditioned — the regime where
eig gradients are already delicate. Singularity detection is exact-zero only,
and the f32 path runs the whole complex pipeline in f32, unlike
`HouseholderProductBackward`, which deliberately widens to f64.

**Recommendation:** Keep the existing pivoted elimination as an LU
factorization with forward/back substitution instead of forming the inverse,
and consider widening to f64 internally.

### CORE-190: `permute_0213` dispatches the f32 kernel for every dtype

- **Severity:** Medium
- **Confidence:** Strong
- **Affected code:** `src/grad_fns/linalg.rs:1901-1907`

The GPU branch calls `backend.permute_0213_f32` unconditionally for any
`T: Float`, although the dispatch trait defines `permute_0213_f64`. Per the
crate's typed-buffer convention, an f64/bf16/f16 CUDA tensor through this
public op fails the buffer downcast with a confusing type error where
`torch.permute` works on every dtype and device — a dtype-dispatch bug, not a
missing feature, since the f64 trait method exists and is never called.

**Recommendation:** Branch on the element type to call the matching kernel and
route unsupported dtypes through the CPU fallback.

### CORE-191: The `gpu` feature of ferrotorch-core is enabled nowhere — every GPU test lane is dead in every workflow

- **Severity:** Critical
- **Confidence:** Confirmed
- **Affected code:** `ferrotorch-core/Cargo.toml:15-24`,
  `.github/workflows/nightly.yml:79-97`, `.github/workflows/cuda-ci.yml:153-176`,
  `.github/workflows/linux-ci.yml`; 43 of 51 `tests/_probe_*.rs` files and the
  GPU modules of every conformance suite

The crate's conformance discipline deliberately gates GPU tests behind
`#[cfg(feature = "gpu")]` instead of `#[ignore]` so they are "entirely absent
rather than silently skipped" — but no workflow and no workspace manifest ever
enables `ferrotorch-core/gpu`. Nightly's `--features cuda` resolves only to
`ferrotorch-gpu/cuda`; cuda-ci tests only `-p ferrotorch-gpu` despite
path-triggering on `ferrotorch-core/**` changes. Verified locally: a
gpu-gated divergence suite runs zero tests on a default build. Clippy in
linux-ci (`--all-targets`, default features) and cuda-ci (`-p ferrotorch-gpu`)
never even type-checks this corpus, allowing silent rot. The cfg-not-ignore
design is defeated at the CI layer: every GPU conformance lane, all 43 GPU
probe files, and every `mod gpu` branch compile to empty test binaries on
every configured build, including the CUDA runner.

**Recommendation:** Add `cargo test -p ferrotorch-core --features gpu --tests`
to cuda-ci or nightly, and `cargo clippy -p ferrotorch-core --features gpu
--all-targets` to linux-ci.

### CORE-192: The nightly workflow has never executed a test step

- **Severity:** High
- **Confidence:** Confirmed (GitHub run history inspected)
- **Affected code:** `.github/workflows/nightly.yml:61-62`, `:99-109`;
  contrast `.github/workflows/cuda-ci.yml:127-135`

`nightly.yml` is the only workflow whose commands cover ferrotorch-core's
`--tests` surface, but it runs on the same hardened self-hosted container as
cuda-ci — where `sudo apt-get` fails, a fact cuda-ci documents and works
around by omitting the step. Nightly still contains the apt-get step, so the
one nightly run that received environment approval (2026-06-05) failed during
setup before any build or test; subsequent runs sat pending manual approval
for ~24h and were cancelled. The tier-4 sweep that nominally backstops
CORE-017 has produced zero test executions. The `-- --ignored` step also
carries `continue-on-error: true` with no tracked baseline, and runs only
`-p ferrotorch-gpu`, so core's ignored set runs nowhere.

**Recommendation:** Remove the apt-get step (dependencies are baked into the
runner image), reconsider manual approval for a scheduled run, and give the
ignored-tests step a tracked baseline.

### CORE-193: The failures blocking `--tests` in Linux CI are two doc-hygiene meta-tests; the functional suites pass at HEAD

- **Severity:** Medium
- **Confidence:** Confirmed (nine representative suites executed)
- **Affected code:** `.github/workflows/linux-ci.yml:80`, `:91-101`;
  `ferrotorch-core/tests/divergence_cite_drift_generic.rs`;
  `ferrotorch-core/tests/divergence_b247d7dbc_self_inflicted_gap_c_probe.rs:52`

Updating CORE-017 at HEAD: `cargo test -p ferrotorch-core --tests --no-run`
compiles cleanly, and nine representative suites all pass (conformance
elementwise/autograd/creation/fft/reduction, test_integration, and the
fake-quantize divergence blocker). The only failures are
`all_design_docs_cites_resolve_at_head` (stale `.design/` line citations) and
the meta-test that re-runs it. The entire functional conformance corpus is
excluded from CI to avoid two documentation-drift failures, while the job
remains named "cargo test (ferrotorch-core lib + tests)".

**Recommendation:** Fix the stale citations or move the cite-drift checks to a
dedicated job, then flip linux-ci to include `--tests`; rename the job until
then.

### CORE-194: Quantize/prune conformance fixtures are a Python mirror of ferrotorch's own algorithms, not PyTorch

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `scripts/regenerate_quantize_prune_fixtures.py:125-198`,
  `:480-512`; `scripts/regenerate_nested_sparse_fixtures.py:818-838`, `:860-862`;
  consumed by `tests/conformance_quantize_prune.rs` and
  `tests/conformance_nested_sparse.rs`

The fixture metadata claims torch 2.11 provenance, but torch is used only to
materialize inputs: the expected values for `quantize_per_tensor`,
`quantize_per_channel`, `dequantize`, `compute_scale_zp`, `magnitude_prune`,
and `apply_2_4_mask` come from Python functions whose docstrings say "matching
ferrotorch's ... exactly", reproducing Rust's rounding and accumulation. The
suite then asserts these "bit-exact" — against ferrotorch's own algorithm
round-tripped through Python. PyTorch's actual oracles
(`torch.quantize_per_tensor`, `torch.ao` observers, which clamp zero_point;
`torch.nn.utils.prune.l1_unstructured`, which prunes exactly n elements via
topk rather than zeroing all ties) are never consulted, so any divergence is
structurally invisible. The 2:4 fixtures use the same mirror pattern and
deliberately avoid tie-magnitude inputs — the exact regime where mirror and
torch could disagree. Only `fake_quantize_differentiable` has genuine torch
provenance. This is the fixture-level instance of the inadequate-parity-infra
pattern.

**Recommendation:** Regenerate these fixtures from the real torch oracles and
pin genuine contract differences as documented divergences rather than
re-deriving expectations from the implementation.

### CORE-195: 331 coverage-gate exclusions cite tracking issues that are all closed

- **Severity:** High
- **Confidence:** Confirmed (issue states verified in the tracker)
- **Affected code:**
  `tests/conformance/_surface_exclusions.toml` (331 of 573 entries);
  `tests/conformance_surface_coverage.rs:268-283`

The strict surface-coverage gate's design says an exclusion without a live
follow-up is indefinite deferral, but 331 entries reference "Pending Phase
2.x" issues (#763-#776) that are all closed: the phases shipped suites without
retiring their exclusions. Items like `grad_fns::reduction::argmax`/`argmin`
(classic tie-break divergence sources) remain permanently exempt with no
conformance test anywhere. The gate's stale-entry guard checks only that the
excluded item still exists, not that the cited issue is open — exactly the
state the gate claims to forbid.

**Recommendation:** Make the gate reject exclusions whose tracking issue is
closed, then burn down the 331 entries starting with value-semantics ops.

### CORE-196: Five GPU conformance suites never assert result device; the linalg suite knowingly exercises a CPU fallback

- **Severity:** High
- **Confidence:** Confirmed
- **Affected code:** `tests/conformance_linalg.rs:435-451`, `:2975-2984`;
  `tests/conformance_reduction.rs`, `tests/conformance_fft.rs`,
  `tests/conformance_activation.rs`, `tests/conformance_einops.rs` (zero
  `is_cuda` assertions in each)

The shared readback helper is device-transparent, and the GPU modules of the
linalg, reduction, fft, activation, and einops suites compare values through
it without ever asserting the result is CUDA-resident — so the codebase's
documented recurring failure mode (silent CPU fallback) produces a green GPU
test. `conformance_linalg.rs` admits this openly for `gpu_mv`: the comment
states the op fails on CUDA tensors so the test "exercises it with the CPU
fallback" — a GPU test pinned to passing via CPU execution (and stale, since
the post-#816 probe asserts device). Contrast the elementwise, shape, and
creation suites, which do assert devices.

**Recommendation:** Add result- and gradient-device assertions to the five
suites' GPU runners and update the `gpu_mv` test to the post-#816 reality.

### CORE-197: Masked extremum fixtures pin ferrotorch's NaN where PyTorch raises

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `tests/conformance_masked.rs:555-569`;
  `scripts/regenerate_masked_fixtures.py:168-183`

The generator documents that `MaskedTensor.amax/amin` raises on all-masked
input while ferrotorch returns a NaN scalar "by design", and encodes the
ferrotorch-side answer so the suite asserts ferrotorch equals itself. The
masked oracle overall is numpy.ma rather than torch.masked. Same family as
CORE-051 (error vs sentinel divergence locked in green) in a distinct op
family and file.

**Recommendation:** Pin the divergence with a tracked issue and an assertion
of the torch-side contract, not a self-referential fixture.

### CORE-198: Tie-break regimes are systematically excluded from fixtures while tie semantics are known to diverge

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `tests/conformance_reduction.rs:1084-1088`;
  `scripts/regenerate_nested_sparse_fixtures.py:860-862`

The cumulative-op fixtures use strictly distinct scan values, with a comment
acknowledging the first-tie vs last-tie divergence from PyTorch and claiming a
cascade issue was filed — but the file's `cascade_skip` is empty and no tie
fixture or ignored test exists; the divergence lives only in a comment. The
2:4 fixtures likewise pick distinct magnitudes to keep tie-break "unambiguous".
Index-returning ops with ties are precisely where silent divergence occurs,
and the generators are tuned never to sample them.

**Recommendation:** Add tie-regime fixtures that pin the current divergence
explicitly under a tracked issue.

### CORE-199: Conformance fixture sampling is trivial — tiny, contiguous, finite, well-conditioned

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `tests/conformance/fixtures/*.json`

Maximum fixture numel ranges from 8 (elementwise) to 256 (linalg); every
input is a freshly built contiguous CPU tensor (only shape.json has view
coverage); NaN/Inf appear only in a handful of division cases; SIMD ops are
"tested" on vectors of length 4-8, which cannot exercise multi-chunk loops,
tails, or accumulation drift; linalg decompositions sample only
well-conditioned matrices up to 4×4 with no rank deficiency or pivoting
stress. creation.json was also generated with torch 2.5.1 on Windows while
every other file used 2.11 on Linux — mixed oracle versions.

**Recommendation:** Extend the generators with size sweeps, non-contiguous
lanes, and special-value lanes per op family; regenerate creation.json on the
current torch.

### CORE-200: The `mean` gradient conformance test never exercises mean's backward

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `tests/conformance_elementwise.rs:1593-1634`

The test labeled as mean's gradient computes the expected value by running
*sum's* backward and dividing by n inside the test, so whatever differentiable
path `mean` actually has (or lacks) is untested — a wrong or missing mean VJP
passes green.

**Recommendation:** Drive mean's own backward; if flat mean is genuinely
non-differentiable, pin that as a divergence instead of synthesizing the
gradient.

### CORE-201: `quantized_matmul` tolerance floor of 0.5 absolute swallows up to 50% error

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `tests/conformance_quantize_prune.rs:656-662`

The assertion floors its tolerance at `step.max(0.5)`. The identity fixture's
expected values range 1.0-4.0, so a result off by half the smallest element
passes; combined with the float reference being plain `a @ b` rather than
torch's quantized matmul, the assertion provides little discrimination, and
the 0.5 floor has no analytic justification in the comment.

**Recommendation:** Remove or analytically justify the floor and scale
fixtures so the step-derived bound dominates.

### CORE-202: The surface-coverage gate counts comment mentions as conformance coverage

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `tests/conformance_surface_coverage.rs:86-92`, `:123-145`

An item counts as covered if its identifier appears as a substring anywhere in
any conformance source — including doc comments and skip-rationale prose. The
suites are saturated with op names in comments (one activation skip block
names a dozen ops), so an op whose test was deleted remains "covered" as long
as its name survives in a comment.

**Recommendation:** Strip comments and strings before the substring scan, or
match against a token stream.

### CORE-203: The live fft GPU cascade-skip asserts nothing about the skipped contract

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `tests/conformance_fft.rs:36-56`, `:442-452`

The only active cascade-skip in the conformance tree (`fftn`/`ifftn`
non-innermost axes on CUDA, #966) bare-`continue`s without asserting the
documented `NotImplementedOnCuda` rejection — if the op started silently
returning wrong values instead of erroring, nothing would notice.

**Recommendation:** Replace the bare continue with an assertion that the call
returns the expected error class.

### CORE-204: Vacuous GPU autograd lane and tautological creation assertion

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `tests/conformance_autograd.rs:2482-2492`;
  `tests/conformance_creation.rs:643`

`gpu_lane_present_by_design` is an intentionally empty test body asserting
nothing, resting on a device-transparency claim contradicted by the tree's own
GPU-autograd bug history (#796, #798/#820). Separately, the creation suite's
`assert_eq!(t.numel(), f.numel.unwrap_or(t.numel()))` self-compares whenever
the fixture omits the field — a silent no-op assertion.

**Recommendation:** Add at least one real fixture-driven backward-on-CUDA case
per grad-fn family, and make missing fixture fields hard errors.

### CORE-205: MKL-gated suites containing a self-described release-blocking failing test never run anywhere

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `tests/divergence_mkl_byte_exact_audit.rs:59`, `:287`;
  `tests/divergence_mkl_byte_exact_critic.rs:30`

Both files are `#![cfg(feature = "mkl")]` and run zero tests on a default
build; no workflow passes `--features mkl`. One test is described in-file as
"the release-blocking failing test for #1538" — a release blocker no CI tier
can observe.

**Recommendation:** Install MKL on the self-hosted runner with an opt-in
nightly step, or document that #1538's blocker is operator-run only.

### CORE-206: The live-torch parity gate is entirely outside CI, and the parity runner's tests pass vacuously without torch

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `tools/parity-sweep/` (zero workflow/justfile
  references); `tools/parity-sweep/runner/tests/*.rs` oracle-spawn guards

The only genuinely live PyTorch comparison in the repository (the parity-sweep
oracle and op_db runner) is human-driven and referenced by no CI job. The
runner crate is, however, a workspace member, and each of its test files
early-returns green with an eprintln "SKIP" when the torch oracle is
unavailable — so even if nightly ran, a runner without python3+torch would
pass vacuously. Every CI-visible "parity" signal is therefore pre-recorded
fixtures or skippable.

**Recommendation:** Make oracle absence a hard failure under an env flag set
in nightly, and add a torch-equipped parity-smoke step on the self-hosted
runner.

### CORE-207: Untracked ignores and green-when-absent skip patterns

- **Severity:** Medium
- **Confidence:** Confirmed
- **Affected code:** `tests/divergence_indexing_56e81de88_audit.rs:204`;
  `tests/divergence_cite_drift_generic.rs:2273`;
  `tests/_probe_p6_sparse_matmul_24.rs:100-116`

Of the eight `#[ignore]` attributes in core tests, two carry no tracking-issue
reference (one "documentation pin only", one diagnostic). The cuSPARSELt probe
soft-skips with a bare `return` when the library is missing or an error string
matches — a green-when-absent pattern (doubly feature-gated today, so
CI-invisible). Core's ignored set is additionally never exercised because the
nightly `--ignored` step targets only ferrotorch-gpu.

**Recommendation:** Add issue references to the untracked ignores, gate the P6
skip on an explicit opt-out env var, and include core in the ignored-tests
sweep.
