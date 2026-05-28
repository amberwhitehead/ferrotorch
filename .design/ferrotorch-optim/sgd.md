# ferrotorch-optim — Stochastic Gradient Descent (SGD)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/sgd.py
-->

## Summary

`ferrotorch-optim/src/sgd.rs` implements the SGD optimizer (stochastic
gradient descent with momentum, dampening, weight decay, Nesterov
acceleration, and the maximize flag), mirroring PyTorch's
`torch.optim.SGD` in `torch/optim/sgd.py`. The Rust impl exposes a
config-builder type `SgdConfig` and the `Sgd<T: Float>` struct, which
implements the workspace-local `Optimizer<T>` trait. Two update paths
are provided: a legacy CPU `data_vec()` loop and a `step_foreach`
on-device tensor-op path that avoids per-step CPU↔GPU round-trips on
CUDA parameters (the Rust analog of PyTorch's `foreach=True` /
`_multi_tensor_sgd`).

## Requirements

- REQ-1: `pub struct SgdConfig` carries `lr, momentum, dampening,
  weight_decay, nesterov, maximize, foreach` with the same defaults as
  PyTorch's `SGD.__init__` (`torch/optim/sgd.py:29-104`).
- REQ-2: `pub struct Sgd<T: Float>` implements
  `Optimizer<T>` (`crate::optimizer::Optimizer`) with `step`, `zero_grad`,
  `lr`, `set_lr`, `param_groups`, `param_groups_mut`, `add_param_group`,
  `state_dict`, `load_state_dict`.
- REQ-3: The legacy CPU `step` path mirrors `_single_tensor_sgd`
  (`torch/optim/sgd.py:322-380`): apply maximize first, then weight
  decay (`grad = grad + wd * param`), then momentum buffer update
  `buf = momentum * buf + (1 - dampening) * grad`, then Nesterov
  blend `grad = grad + momentum * buf` if requested, then
  `param = param - lr * grad`.
- REQ-4: First-step momentum-buffer initialisation matches PyTorch:
  on the first observed gradient the buffer is set to `grad.clone()`,
  not to `(1 - dampening) * grad` (`torch/optim/sgd.py:349-359`).
- REQ-5: The foreach path `step_foreach` keeps parameter and momentum
  buffer storage on the parameter's native device throughout the
  update, mirroring `_multi_tensor_sgd` (`torch/optim/sgd.py:382-481`)
  while preserving the same numerical update.
- REQ-6: `state_dict` serialises the momentum buffer (`Vec<f64>`) and
  step count per `(group_idx, param_idx)` key; `load_state_dict`
  round-trips the same keys. Mirrors PyTorch's
  `Optimizer.state_dict()` shape (per-parameter `state`).
- REQ-7: `zero_grad` clears every parameter's `.grad` slot back to
  `None`, mirroring `Optimizer.zero_grad(set_to_none=True)`
  (the modern default in PyTorch ≥ 1.7).
- REQ-8: Parameters whose `.grad()` is `None` are skipped (not
  treated as zero), mirroring PyTorch's `if p.grad is None: continue`
  in `_single_tensor_sgd` (`torch/optim/sgd.py:325-330`).
- REQ-9: `Sgd::set_momentum(group_idx, value)` + `Sgd::momentum(group_idx)`
  override the `Optimizer<T>` trait defaults and write through to /
  read from `config.momentum`, enabling `CyclicLR` / `OneCycleLR` to
  cycle momentum inversely with the learning rate
  (`torch/optim/lr_scheduler.py:1840-1862`, `:2342-2350`).

## Acceptance Criteria

- [x] AC-1: `SgdConfig` exposes builder methods (`with_lr`,
  `with_momentum`, `with_dampening`, `with_weight_decay`,
  `with_nesterov`, `with_maximize`, `with_foreach`) plus the older
  short-form (`momentum`, `dampening`, ...) and `Default` constructs
  `lr = 0.01`.
- [x] AC-2: `impl<T: Float> Optimizer<T> for Sgd<T>` compiles and
  exposes all eight trait methods.
- [x] AC-3: `test_sgd_basic_step` produces
  `param = param - lr * grad` exactly on the first step (no momentum).
- [x] AC-4: `test_sgd_momentum_works` verifies
  `buf_2 = 0.9 * buf_1 + grad_2` over two steps.
- [x] AC-5: `test_sgd_nesterov_momentum` verifies the
  `grad = grad + momentum * buf` blend produces 9.81 from initial 10.0
  with `lr = 0.1, momentum = 0.9`.
- [x] AC-6: `test_sgd_weight_decay` verifies `grad += wd * param` is
  applied before the update.
- [x] AC-7: `test_state_dict_roundtrip` round-trips the
  `"group_idx_param_idx"` keyed serialisation.
- [x] AC-8: Four foreach-parity tests
  (`test_foreach_basic_parity_no_momentum`,
  `_parity_with_momentum`, `_parity_with_weight_decay`,
  `_parity_with_nesterov`) confirm the legacy CPU path and
  `step_foreach` agree to within 1e-5.
- [x] AC-9: `test_sgd_skips_params_without_grad` confirms parameters
  with `.grad() == None` are not touched.

## Architecture

### `SgdConfig` (REQ-1)

The config is a `#[non_exhaustive]` struct with the seven
hyperparameters listed above, plus a `Default` impl. Per PyTorch
defaults (`torch/optim/sgd.py:38-49`): `lr=required`, `momentum=0`,
`dampening=0`, `weight_decay=0`, `nesterov=false`,
`maximize=false`. Ferrotorch picks `lr=0.01` as the `Default` so the
type is `Default`-able for tests; production callers pass an explicit
`lr`. Builder methods come in two namespaces: bare-name (`momentum(0.9)`)
and `with_*` (`with_momentum(0.9)`); both mutate `self` and return
`Self` for chaining.

### `Sgd<T>` struct (REQ-2)

Holds `param_groups: Vec<ParamGroup<T>>` (the parameters partitioned
by hyperparameter group), the global `config`, two momentum-buffer
maps (`momentum_buffers: HashMap<String, Vec<T>>` for the CPU path,
`foreach_buffers: HashMap<String, Tensor<T>>` for the on-device path),
and a `step_count: HashMap<String, u64>` for first-step initialisation
of the momentum buffer. Keys are `"{group_idx}_{param_idx}"` strings;
this is the same wire format PyTorch uses internally except that
PyTorch keys by the parameter's `id()` rather than by group/param
index.

### CPU `step` path (REQ-3, REQ-4, REQ-7, REQ-8)

Order of operations in the implementation mirrors
`_single_tensor_sgd` exactly:

1. `match param.grad()? { Some(g) => g, None => continue }` skips
   parameters without gradients (REQ-8).
2. `if self.config.maximize { *g = zero - *g; }` negates the
   gradient for ascent (PyTorch's `maximize=True`).
3. `if wd > 0.0 { *g += wd_t * p; }` applies L2 weight decay
   (`torch/optim/sgd.py:340-343`).
4. Momentum update:
   - First step (`step == 0`): `buf = grad.clone()` (REQ-4).
   - Subsequent: `buf = momentum * buf + (1 - dampening) * grad`.
5. Effective gradient: Nesterov blends
   `grad + momentum * buf`; else `effective = buf`.
6. `param = param - lr * effective_grad` via the unsafe
   `Tensor::update_data` (sole-writer guard documented in the
   per-call `SAFETY:` block).
7. `zero_grad` writes `None` to every `param.grad` slot (REQ-7).

### Foreach `step_foreach` path (REQ-5)

Activated when `config.foreach == true`. Replaces the
`data_vec()`/`update_data` round-trip with tensor-op kernels
(`mul`/`add`/`sub`) dispatched on the parameter's device. The
momentum buffer becomes a `Tensor<T>` instead of a `Vec<T>`, owned
by the optimizer in `foreach_buffers`. Numerically equivalent to the
CPU path within `T` precision (CPU path computes intermediate
products in `T` directly; `foreach_buffers` stores `T` tensors). The
per-parameter `update_storage` call swaps the storage Arc with a
documented `SAFETY:` block detailing the four sole-writer
invariants (no other clone of `Sgd<T>`, inside `no_grad`, no live
borrow into the existing storage, matching numel + device).

### State-dict (REQ-6)

`state_dict` serialises each momentum buffer to a `Vec<f64>` via
`cast::<T, f64>` (the `Optimizer::state_dict` signature is
`FerrotorchResult<OptimizerState>` so the cast can fail rather
than panic). `load_state_dict` clears the existing buffers and
re-populates from the deserialised entries.

### Non-test production consumers

- `ferrotorch-optim/src/lib.rs:53` — `pub use sgd::{Sgd, SgdConfig};`
  exposes `Sgd` and `SgdConfig` as crate-level public API.
- `ferrotorch/src/lib.rs:51` — `pub use ferrotorch_optim::{Adam, AdamW,
  Optimizer, Sgd};` re-exports `Sgd` from the umbrella crate's
  `prelude` module.
- `ferrotorch/src/lib.rs:61` — `pub use ferrotorch_optim::*;` exposes
  `SgdConfig` and the rest of the optim surface.
- `ferrotorch-train/src/learner.rs:516-528` consumes the `Optimizer`
  trait implemented by `Sgd<T>` and round-trips its `state_dict`.

## Parity contract

`parity_ops = []`. SGD has no per-op parity-sweep entry — the
optimiser is verified end-to-end by training-loop convergence tests
plus the foreach-parity test suite. Edge-cases the impl owns:

- **First-step momentum**: buffer initialised to `grad.clone()`, not
  `(1 - dampening) * grad`. Matches `torch/optim/sgd.py:349-359`.
- **Nesterov gating**: PyTorch errors if `nesterov=True` with
  `momentum<=0` or `dampening!=0`; ferrotorch follows the math
  silently (a Nesterov blend with zero momentum reduces to the
  bare gradient, which is the mathematically correct limit).
- **Sparse gradients**: PyTorch dispatches a sparse code path when
  `grad.is_sparse`; ferrotorch's `data_vec()` densifies the
  gradient before the loop. Sparse-grad support tracks via
  `ferrotorch-optim/src/sparse_adam.rs` (separate file).
- **`requires_grad == false` parameters**: PyTorch silently skips
  them inside the optimizer's iteration. Ferrotorch's `param.grad()`
  returns `None` for such parameters (because `backward()` never
  populates them), and REQ-8 covers the skip.

## Verification

Tests in `mod tests in sgd.rs` (15 tests):

- Basic update: `test_sgd_basic_step`, `test_sgd_skips_params_without_grad`.
- Zero-grad: `test_zero_grad_clears_grads`.
- Momentum: `test_sgd_momentum_works`,
  `test_sgd_nesterov_momentum`.
- Weight decay: `test_sgd_weight_decay`.
- LR accessors: `test_lr_get_set`, `test_param_groups_different_lr`.
- State-dict: `test_state_dict_roundtrip`.
- Convergence (gated `#[ignore]` for flakiness):
  `test_xor_convergence`.
- Foreach parity:
  `test_foreach_basic_parity_no_momentum`,
  `test_foreach_parity_with_momentum`,
  `test_foreach_parity_with_weight_decay`,
  `test_foreach_parity_with_nesterov`,
  `test_foreach_skips_params_without_grad`.

Smoke command (no parity ops to gate on):

```bash
cargo test -p ferrotorch-optim --lib sgd:: 2>&1 | tail -3
```

Expected: `14 passed; 1 ignored` (XOR convergence is `#[ignore]`'d).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct SgdConfig` in `sgd.rs` mirroring `torch/optim/sgd.py:29-104`; non-test consumer: `sgd in ferrotorch-optim/src/lib.rs` re-exports `SgdConfig`; `ferrotorch/src/lib.rs` re-exports the umbrella crate's optim surface. |
| REQ-2 | SHIPPED | impl: `impl<T: Float> Optimizer<T> for Sgd<T>` block in `sgd.rs` with all eight trait methods; non-test consumer: `Sgd in ferrotorch-train/src/learner.rs` `use ferrotorch_optim::Optimizer;` drives `optimizer.step()` in the training loop, satisfied for any `Sgd<f32>` instance. |
| REQ-3 | SHIPPED | impl: legacy CPU `step` method in `Optimizer<T> for Sgd<T>` (`sgd.rs`) mirroring `_single_tensor_sgd` in `torch/optim/sgd.py:322-380`; non-test consumer: `ferrotorch/examples/train_mnist.rs` uses an `Optimizer<T>` (Adam in the example; Sgd is the prelude-exported sibling on the same trait). |
| REQ-4 | SHIPPED | impl: first-step branch `if *step == 0 { self.momentum_buffers.insert(key, grad_data.clone()); }` in `sgd.rs` (`step` method) mirroring `torch/optim/sgd.py:349-359`; non-test consumer: same as REQ-3 (training-loop callers exercise the first step). |
| REQ-5 | SHIPPED | impl: `Sgd::step_foreach` method in `sgd.rs` mirroring `_multi_tensor_sgd` in `torch/optim/sgd.py:382-481`; non-test consumer: `ferrotorch/src/lib.rs:61` `pub use ferrotorch_optim::*;` re-exports `Sgd` and `SgdConfig` so a downstream consumer can opt into `foreach = true`. |
| REQ-6 | SHIPPED | impl: `state_dict` / `load_state_dict` methods on `Sgd<T>` (`sgd.rs`); non-test consumer: `ferrotorch-serialize/src/checkpoint.rs:48` `use ferrotorch_optim::OptimizerState;` is the on-disk checkpoint writer consuming this map. |
| REQ-7 | SHIPPED | impl: `Sgd::zero_grad` method clearing all parameter grads in `sgd.rs`; non-test consumer: `zero_grad in ferrotorch-train/src/learner.rs` invokes `Optimizer::zero_grad` at the start of every step. |
| REQ-8 | SHIPPED | impl: `let grad_tensor = match param.grad()? { Some(g) => g, None => continue };` inside both `step` and `step_foreach` in `sgd.rs` mirroring `torch/optim/sgd.py:325-330`; non-test consumer: `ferrotorch-train/src/learner.rs` training step relies on this skip when frozen layers are present. |
| REQ-9 | SHIPPED | impl: `fn set_momentum(&mut self, group_idx, value)` + `fn momentum(&self, group_idx)` overrides on `impl<T: Float> Optimizer<T> for Sgd<T>` in `sgd.rs` mirror `torch/optim/lr_scheduler.py:1840-1862, 2342-2350`; non-test consumer: `CyclicLR::step` in `scheduler/cyclic_lr.rs` and `OneCycleLR::step` in `scheduler/one_cycle_lr.rs` invoke `optimizer.set_momentum` when `cycle_momentum` is enabled. |
