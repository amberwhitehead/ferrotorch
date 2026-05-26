# ferrotorch-optim — `Optimizer` trait + `ParamGroup` + `OptimizerState`

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/optimizer.py
  - torch/optim/_functional.py
-->

## Summary

`ferrotorch-optim/src/optimizer.rs` defines the `Optimizer<T>` trait
that every concrete optimizer (`Sgd`, `Adam`, `AdamW`, `Adamax`,
`Adadelta`, `Adafactor`, `Adagrad`, `Asgd`, `Lbfgs`, `Muon`, `NAdam`,
`Rmsprop`, `Rprop`, `Kfac`, `RAdam`, `SparseAdam`) implements, the
`ParamGroup<T>` struct that holds a set of parameters sharing a
learning rate + weight-decay (mirroring PyTorch's `param_groups`
list-of-dicts), and the `OptimizerState` checkpoint type alias
(`HashMap<String, HashMap<String, Vec<f64>>>` — outer key is the
per-parameter `ParamKey` string, inner map is the named state vectors
flattened to `f64`). It also exports four crate-internal workspace
helpers (`tensor_to_f64_vec`, `fill_f64_workspace`,
`fill_t_workspace`, `resize_typed_workspace`,
`resize_f64_workspace`, `f64_vec_to_tensor`) that the concrete
optimisers use to amortise per-step allocation.

## Requirements

- REQ-1: `pub struct ParamGroup<T: Float>` holds `params:
  Vec<Parameter<T>>` (crate-private to enforce add-via-`add_param`
  invariants), `lr: f64`, `weight_decay: f64`. Mirrors PyTorch's
  `param_groups` dict entries
  (`torch/optim/optimizer.py:399-450` — `_param_groups` setup).
- REQ-2: `ParamGroup::new(params, lr)` constructor +
  `with_weight_decay(wd)` builder-style mutator +
  `params()` read-only accessor + `add_param(p)` fallible append.
  Defaults match PyTorch (`weight_decay = 0.0`).
- REQ-3: `pub trait Optimizer<T: Float>` — the trait every concrete
  optimizer implements. Methods:
  - `fn step(&mut self) -> FerrotorchResult<()>` — one update step
    (no `closure` overload in this trait; LBFGS handles closures
    via a separate concrete API on its struct).
  - `fn zero_grad(&mut self) -> FerrotorchResult<()>` — clear every
    parameter's gradient.
  - `fn lr(&self) -> f64` — read first-group learning rate (the
    canonical "current LR" reading PyTorch's
    `optimizer.param_groups[0]['lr']` users do).
  - `fn set_lr(&mut self, lr: f64)` — broadcast LR to every group
    (LR schedulers consume this).
  - `fn param_groups(&self) -> &[ParamGroup<T>]` /
    `fn param_groups_mut(&mut self) -> &mut [ParamGroup<T>]` —
    iteration access for schedulers (LR-per-group), gradient
    scalers (per-parameter inf/NaN check), and serialization.
  - `fn add_param_group(&mut self, group)` — append a new group at
    runtime (matches `Optimizer.add_param_group`).
  - `fn state_dict(&self) -> FerrotorchResult<OptimizerState>` /
    `fn load_state_dict(&mut self, state) -> FerrotorchResult<()>`
    — checkpoint serialization. The fallible signature on
    `state_dict()` exists so impls that cast generic `T` values to
    `f64` (e.g. SGD momentum buffers in lower-precision tensors)
    can propagate cast failure rather than panicking.
- REQ-4: All updates execute in autograd's `no_grad()` mode by
  contract (no method on the trait enforces it — the concrete impl
  is responsible). Mirrors PyTorch's
  `@_use_grad_for_differentiable` decorator
  (`torch/optim/optimizer.py:59-87`).
- REQ-5: `pub type OptimizerState = HashMap<String, HashMap<String, Vec<f64>>>`
  — the on-disk wire format. Outer key uses the legacy
  `"g{group_idx}_p{param_idx}"` format (matches the pre-`ParamKey`
  string format, kept for checkpoint round-trip). Inner map keys are
  state-vector names (`"exp_avg"`, `"exp_avg_sq"`, `"momentum_buffer"`,
  etc.). Values are dtype-erased `Vec<f64>` flattened in row-major
  layout. Mirrors `torch.optim.Optimizer.state_dict()`'s
  serialization format (`torch/optim/optimizer.py:700-800`).
- REQ-6: `pub(crate)` workspace helpers
  (`tensor_to_f64_vec`, `fill_f64_workspace`, `fill_t_workspace`,
  `resize_typed_workspace`, `resize_f64_workspace`,
  `f64_vec_to_tensor`) amortise per-step heap allocation across
  steady-state training. Justification recorded in the module
  (`CL-1125`): a 7-billion-parameter model with `data_vec` +
  `collect` per step costs ~28 GB of transient allocation per
  optimizer step.

## Acceptance Criteria

- [x] AC-1: `pub struct ParamGroup<T: Float>` with `pub(crate) params`,
  `pub lr: f64`, `pub weight_decay: f64` fields, deriving `Debug`.
- [x] AC-2: `ParamGroup::new` / `with_weight_decay` / `params` /
  `add_param` all present.
- [x] AC-3: `pub trait Optimizer<T: Float>` declares all 9 methods
  listed in REQ-3.
- [x] AC-4: `pub type OptimizerState = HashMap<String, HashMap<String, Vec<f64>>>`.
- [x] AC-5: Six `pub(crate)` workspace helpers exist with `CL-1125`
  rationale documented inline.
- [x] AC-6: `state_dict` signature returns `FerrotorchResult` so
  cast failures propagate (no `unwrap()` in the trait contract).

## Architecture

### `ParamGroup<T>` (REQ-1, REQ-2)

The crate-private `params` field is the central invariant: external
code must go through `add_param` so future invariants (dtype
homogeneity within a group, device homogeneity, etc.) can be
enforced without a breaking change. Today `add_param` always
succeeds; the `FerrotorchResult` return is reserved for those future
checks.

The `lr` and `weight_decay` are `pub` because schedulers and
diagnostics legitimately need to read/write them directly. (The
trait method `set_lr` is the canonical scheduler driver, but the
struct field is the storage; both paths converge on the same
in-memory mutation.)

### `Optimizer<T>` trait (REQ-3, REQ-4)

The trait is generic over `T: Float` matching the tensor element
type. There are NO default implementations — every concrete
optimizer ships its own `step`, `zero_grad`, etc., because the
update rule IS the optimizer. Generic methods (LR broadcast, state
serialization layout) are not abstracted in the trait because the
state shape differs per-optimizer (SGD's `momentum_buffer` vs Adam's
`exp_avg + exp_avg_sq + step`).

`no_grad` enforcement is by convention: every concrete `step` impl
in this crate wraps the parameter update in a `no_grad`-equivalent
guard (calling the in-place tensor mutators that are documented to
not record autograd). The non-differentiable contract matches
PyTorch's `@_use_grad_for_differentiable` decorator
(`torch/optim/optimizer.py:59-87`); the differentiable-step
alternative is in `ferrotorch-optim/src/differentiable.rs`
(separate functions, not a trait method).

### `OptimizerState` (REQ-5)

The double-nested HashMap mirrors PyTorch's `state_dict` format:
- Outer key: `"g{group_idx}_p{param_idx}"` (the `ParamKey` wire
  format — see `param_key.md`).
- Inner key: per-state-vector name (e.g. `"exp_avg"` for Adam).
- Inner value: flattened `Vec<f64>` (dtype-erased; the loader
  reconstructs the typed tensor via `f64_vec_to_tensor`).

All numerics are stored as `f64` to keep the wire format
dtype-independent — a checkpoint written by an `Adam<f32>` model
can be loaded into an `Adam<f64>` (or vice versa) with the cast
happening at load time via `f64_vec_to_tensor`. Mirrors
`torch.optim.Optimizer.state_dict`'s tensor-detached storage
(PyTorch stores tensors directly; ferrotorch stores flat `Vec<f64>`
because the on-disk format is JSON-friendly).

### Workspace helpers (REQ-6)

The `CL-1125` block is the rationale: every optimizer step would
otherwise heap-allocate two `Vec` workspaces per parameter (one
typed, one `f64`). For large models that is gigabytes of transient
allocation per step. The helpers reuse a single owner-held
workspace across steps; capacity grows monotonically to the largest
parameter the optimizer has seen.

- `tensor_to_f64_vec(t)`: one-shot read (no workspace) — used at
  state-dict serialization time where reuse is irrelevant.
- `fill_f64_workspace(workspace, tensor)`: per-step hot path
  reader, CPU-contiguous slice borrowed zero-copy; CUDA / non-
  contiguous goes through `data_vec`.
- `fill_t_workspace(workspace, tensor)`: same for the typed slice.
  CUDA path uses `mem::swap` to keep the workspace's capacity.
- `resize_typed_workspace(workspace, n)` /
  `resize_f64_workspace(workspace, n)`: prepare an empty buffer of
  exactly `n` zero-initialised elements.
- `f64_vec_to_tensor(data, shape)`: state-dict load path — casts
  `f64` → `T` and constructs a tensor.

### Non-test production consumers

- Every in-tree optimizer: `ferrotorch-optim/src/adam.rs:20`,
  `adamw.rs`, `adadelta.rs:23`, `adamax.rs:18`, `adafactor.rs:19`,
  `asgd.rs:20`, `muon.rs:24`, `radam.rs:22`, `rmsprop.rs:13`,
  `rprop.rs:38`, `sgd.rs`, `nadam.rs`, `lbfgs.rs`,
  `natural_gradient.rs`, `sparse_adam.rs` all
  `use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};`.
- `ferrotorch-optim/src/scheduler/mod.rs:71`
  `use crate::optimizer::Optimizer;` — every LR scheduler drives
  the trait.
- `ferrotorch-optim/src/grad_scaler.rs:18`
  `use crate::optimizer::Optimizer;` — `GradScaler::unscale_` /
  `step` consume `&mut dyn Optimizer<T>`.
- `ferrotorch-train/src/learner.rs:28` `use ferrotorch_optim::Optimizer;`,
  line 59 `optimizer: Box<dyn Optimizer<T>>` — the central training
  loop field.
- `ferrotorch-train/src/amp.rs:58` `use ferrotorch_optim::Optimizer;`.

## Parity contract

`parity_ops = []`. The trait + struct are structural — numerical
parity is owned by each concrete optimizer's design doc. Edge cases
the trait itself owns:

- **`step` failing**: returns `FerrotorchResult::Err`; no partial
  parameter updates expected in a single failed step (concrete
  impls are responsible for atomicity).
- **`set_lr` with NaN / Inf**: not rejected by the trait
  (matches PyTorch's permissive setter — users observing diverging
  loss are expected to inspect their LR schedule).
- **`add_param_group` order**: the new group is appended at the
  end; existing group indices stay stable so that checkpointed
  `ParamKey`s remain valid after adding a group post-load.
- **`state_dict` cast failure**: returns
  `FerrotorchError::InvalidArgument` (via `cast::<T, f64>`).
- **`load_state_dict` key mismatch**: concrete impls decide
  (strict vs lenient). Most impls treat unexpected keys as an
  error; missing keys leave the corresponding parameter's state at
  whatever it was before (so a partial-load resumes training with
  the existing state).

## Verification

Two unit tests in `mod tests` (lib.rs line 210-229):

- `test_param_group_construction` — `lr` defaults, `weight_decay
  = 0.0`, single-parameter group length.
- `test_param_group_with_weight_decay` — builder-style setter.

The trait itself is exercised through every concrete optimizer's
tests (327 lib tests across the crate).

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib optimizer:: 2>&1 | tail -3
```

Expected: `2 passed` for `optimizer::tests`, `327 passed; 0 failed`
for the full lib-test sweep.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct ParamGroup<T: Float>` at `ferrotorch-optim/src/optimizer.rs:20` mirroring `torch/optim/optimizer.py:399-450`; non-test consumer: every in-tree optimizer file (`adam.rs:20`, `adamw.rs:22`, `sgd.rs`, ...) holds `Vec<ParamGroup<T>>` as its `param_groups` field; `ferrotorch-train/src/learner.rs:516` `fn param_groups(&self) -> &[ferrotorch_optim::ParamGroup<f32>]`. |
| REQ-2 | SHIPPED | impl: `ParamGroup::new` / `with_weight_decay` / `params` / `add_param` at `ferrotorch-optim/src/optimizer.rs:31-65`; non-test consumer: `ferrotorch-optim/src/grad_scaler.rs:391` `ParamGroup::new(params, 0.01)` (in MockOptimizer); production `ParamGroup::new` calls are inside every concrete optimizer's `new` constructor; `ferrotorch-train/src/learner.rs` reads `.lr` directly via `.param_groups()[0].lr`. |
| REQ-3 | SHIPPED | impl: `pub trait Optimizer<T: Float>` at `ferrotorch-optim/src/optimizer.rs:68` with all 9 methods declared on lines 72-104, mirroring `torch/optim/optimizer.py:339`; non-test consumer: `ferrotorch-train/src/learner.rs:28` `use ferrotorch_optim::Optimizer;` and line 59 `optimizer: Box<dyn Optimizer<T>>` holds the trait. |
| REQ-4 | SHIPPED | impl: contract documented at `ferrotorch-optim/src/optimizer.rs:72` ("All parameter updates execute inside `no_grad()`."); non-test consumer: every concrete `step` impl (`ferrotorch-optim/src/sgd.rs`, `adam.rs`, ...) wraps the parameter update in the in-place mutators that do not record autograd — matches `torch.optim.optimizer._use_grad_for_differentiable` at `torch/optim/optimizer.py:59-87`. |
| REQ-5 | SHIPPED | impl: `pub type OptimizerState = HashMap<String, HashMap<String, Vec<f64>>>` at `ferrotorch-optim/src/optimizer.rs:8` mirroring `torch.optim.Optimizer.state_dict()` shape; non-test consumer: every concrete optimizer's `state_dict` / `load_state_dict` impl returns / consumes `OptimizerState`; `ferrotorch-train/src/learner.rs:523` `fn state_dict(&self) -> FerrotorchResult<ferrotorch_optim::OptimizerState>` is the trait method override on the learner's mock optimizer plumbing. |
| REQ-6 | SHIPPED | impl: `tensor_to_f64_vec` at line 108, `fill_f64_workspace` at line 129, `fill_t_workspace` at line 159, `resize_typed_workspace` at line 185, `resize_f64_workspace` at line 193, `f64_vec_to_tensor` at line 199 of `ferrotorch-optim/src/optimizer.rs`, all annotated with the `CL-1125` rationale; non-test consumer: the workspace helpers are `pub(crate)` and consumed by `ferrotorch-optim/src/adam.rs`, `adamw.rs`, `sgd.rs`'s per-step paths. |
