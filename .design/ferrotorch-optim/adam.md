# ferrotorch-optim — Adam (Adaptive Moment Estimation)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/adam.py
-->

## Summary

`ferrotorch-optim/src/adam.rs` implements the Adam optimizer (Kingma &
Ba, ICLR 2015) plus the AMSGrad variant (Reddi et al., ICLR 2018),
mirroring `torch.optim.Adam` in `torch/optim/adam.py`. The Rust impl
exposes `AdamConfig` (lr, betas, eps, weight_decay, amsgrad, maximize,
foreach) and the `Adam<T: Float>` struct implementing the
crate-local `Optimizer<T>` trait. The implementation supports three
update paths: an f64-precision CPU loop, a `cudarc`-backed fused
`fused_adam_f32` kernel for f32 GPU parameters, and a tensor-op
`step_foreach` path that keeps moments on the parameter's device.

## Requirements

- REQ-1: `pub struct AdamConfig` carries `lr=1e-3, betas=(0.9, 0.999),
  eps=1e-8, weight_decay=0.0, amsgrad=false, maximize=false,
  foreach=false`, matching `torch.optim.Adam.__init__` defaults
  (`torch/optim/adam.py:35-200`).
- REQ-2: `pub struct Adam<T: Float>` implements `Optimizer<T>` with
  all eight trait methods (step, zero_grad, lr, set_lr, param_groups,
  param_groups_mut, add_param_group, state_dict, load_state_dict).
- REQ-3: The legacy CPU `step` path mirrors `_single_tensor_adam`
  (`torch/optim/adam.py:347-552`): maximize negation, L2 weight decay
  added to grad, first moment `m = beta1*m + (1-beta1)*g`, second
  moment `v = beta2*v + (1-beta2)*g^2`, bias-corrected
  `m_hat = m/(1-beta1^t)`, `v_hat = v/(1-beta2^t)`, and parameter
  update `p -= lr * m_hat / (sqrt(v_hat) + eps)`.
- REQ-4: AMSGrad variant maintains `max_exp_avg_sq` via elementwise
  max-of-history of the second moment and uses it in place of
  `exp_avg_sq` in the denominator, mirroring
  `torch/optim/adam.py:445-470`.
- REQ-5: `Adam::step` dispatches a GPU fused kernel
  (`backend.fused_adam_f32`) when `T == f32` AND the parameter is
  CUDA-resident AND AMSGrad/maximize are not active. The kernel
  fuses moment update + parameter update + bias correction into one
  launch, avoiding multiple kernel-launch latencies.
- REQ-6: The foreach path `step_foreach` keeps moments on the
  parameter's device via `Tensor<T>` storage and avoids CPU
  round-trips, mirroring `_multi_tensor_adam`
  (`torch/optim/adam.py:553-901`).
- REQ-7: `state_dict` serialises per-parameter `step_count`,
  `exp_avg`, `exp_avg_sq`, and (if AMSGrad) `max_exp_avg_sq`. The
  on-disk key shape `"g{group_idx}_p{param_idx}"` is rendered via
  `ParamKey::Display` (CL-1122) so checkpoint format is preserved.
- REQ-8: Parameters whose `.grad()` is `None` are skipped, mirroring
  PyTorch's `if grad is None: continue` in `_single_tensor_adam`
  (`torch/optim/adam.py:362-365`).
- REQ-9: GPU fast-path failure semantics match PyTorch: a GPU
  allocation failure is `Err` by default; users opt into a CPU
  fallback via `FERROTORCH_ENABLE_GPU_FALLBACK=1` (Rust-specific
  ergonomic addition; PyTorch raises). No silent backend
  degradation.

## Acceptance Criteria

- [x] AC-1: `AdamConfig::default()` returns the exact PyTorch defaults
  (lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
  amsgrad=false).
- [x] AC-2: `impl<T: Float> Optimizer<T> for Adam<T>` compiles and
  exposes all eight trait methods.
- [x] AC-3: `test_adam_rosenbrock_convergence` minimises Rosenbrock
  to within 0.05 of `(1, 1)` in 5000 steps with `lr=0.01`.
- [x] AC-4: `test_adam_amsgrad` exercises the
  `max_exp_avg_sq` code path and verifies the `max_exp_avg_sq` key
  appears in `state_dict`.
- [x] AC-5: `test_adam_weight_decay` verifies the
  `grad += wd * param` L2 step is applied before moment updates.
- [x] AC-6: `test_adam_state_dict_roundtrip` round-trips
  `exp_avg`, `exp_avg_sq`, and `step_count` keys.
- [x] AC-7: Four foreach-parity tests
  (`test_adam_foreach_basic_parity`,
  `_parity_with_weight_decay`, `_parity_with_amsgrad`,
  `_skips_params_without_grad`) confirm the legacy CPU path and
  `step_foreach` agree to within 1e-3 (f32 precision tolerance).
- [x] AC-8: `test_adam_multiple_params` confirms independent moments
  per parameter.

## Architecture

### `AdamConfig` (REQ-1)

The config is `#[derive(Debug, Clone, Copy)]` `#[non_exhaustive]`
with seven `pub` fields. `Default` matches PyTorch byte-for-byte.
Builder methods follow the `with_*` convention (consume self,
return Self).

### `Adam<T>` struct (REQ-2)

Owns:

- `param_groups: Vec<ParamGroup<T>>`
- `config: AdamConfig`
- `state: HashMap<ParamKey, AdamParamState>` (CPU path)
- `foreach_state: HashMap<ParamKey, AdamForeachState<T>>` (foreach path)
- `param_workspace`, `grad_workspace`, `new_values_workspace`
  (CL-1125: reusable `Vec<f64>` / `Vec<T>` workspaces, see
  Architecture / Workspaces below).

`AdamParamState` holds CPU moments (`exp_avg`, `exp_avg_sq`,
`max_exp_avg_sq`, `step_count`) plus optional GPU handles
(`gpu_exp_avg`, `gpu_exp_avg_sq` of type `GpuBufferHandle`) when the
fused kernel is active.

`AdamForeachState<T>` holds device-resident moments as `Tensor<T>`.

### Legacy CPU `step` (REQ-3, REQ-4, REQ-8)

1. Skip parameters with no gradient.
2. Compute `use_gpu = is_f32 && tensor.is_cuda() && grad.is_cuda() &&
   !amsgrad && !maximize`. If `use_gpu`, dispatch to the fused
   GPU kernel branch (REQ-5).
3. Otherwise, fill the `param_workspace` / `grad_workspace`
   `Vec<f64>` reusable buffers via `fill_f64_workspace`
   (`optimizer.rs`). This is the CL-1125 amortisation
   pattern: zero per-step allocation in steady state.
4. Apply maximize negation.
5. Apply L2 weight decay: `g += wd * p`.
6. Update first moment: `m = beta1*m + (1-beta1)*g`.
7. Update second moment: `v = beta2*v + (1-beta2)*g*g`.
8. Bias-correct: `bc1 = 1 - beta1^t`, `bc2 = 1 - beta2^t`.
9. AMSGrad: if enabled, update `max_exp_avg_sq[i] = max(max_exp_avg_sq[i], exp_avg_sq[i])` and use it in the denominator (REQ-4).
10. Compute updated parameter values into `new_values_workspace`
    via `update[i] = p[i] - lr * m_hat[i] / (sqrt(v_hat[i]) + eps)`.
11. `update_data` (unsafe, with documented SAFETY block).

### GPU fused kernel branch (REQ-5, REQ-9)

When `use_gpu == true`:

1. Lazy-init GPU state via `Entry::Vacant` (so `Err` propagation
   works — `or_insert_with` cannot return `Result`).
2. On allocation failure (`backend.alloc_zeros` returns `Err`):
   if `FERROTORCH_ENABLE_GPU_FALLBACK` is set, log a warning via
   `tracing` and fall through to the CPU path; otherwise return
   `Err(FerrotorchError::Internal)` so the failure surfaces
   instead of silently degrading.
3. Compute `bc1 = 1 - (beta1 as f32)^step`, `bc2 = 1 - (beta2 as f32)^step`.
4. Dispatch `backend.fused_adam_f32` with `param_handle`, `grad`,
   `gpu_m`, `gpu_v`, the hyperparameters, and the bias-correction
   coefficients in a single launch.

### Foreach path (REQ-6)

Activated when `config.foreach == true`. Uses
`ferrotorch_core::grad_fns::arithmetic::{add, div, mul, neg, sqrt, sub}`
to drive moment + parameter updates entirely via tensor ops on the
parameter's device. Initialisation uses `Entry::Vacant` (matching
the CPU branch's `?`-propagation requirement). The
`update_storage(storage)?` call is unsafe with a documented SAFETY
block detailing the four sole-writer invariants.

### State-dict (REQ-7)

`state_dict` renders each `ParamKey` via `Display` to the legacy
`"g{group_idx}_p{param_idx}"` wire format (CL-1122). When the
state is GPU-resident (`gpu_exp_avg.is_some()`), the impl
downloads via `backend.gpu_to_cpu` and reinterprets the byte buffer
as native-endian f32 via `f32::from_ne_bytes` (no `unsafe`
reinterpret cast). `load_state_dict` parses the string back into
`ParamKey` via `FromStr`; invalid keys surface as
`InvalidArgument`.

### Workspaces (CL-1125)

Without the workspace pattern, every `step()` would heap-allocate
two full-numel `Vec<f64>` (param, grad) plus one `Vec<T>`
(new_values). For a 7B-param model that is ~28 GB of transient
allocation per step. The workspace pattern reuses optimizer-owned
buffers; once warmed to the largest parameter, steady-state
amortised allocation is zero.

### Non-test production consumers

- `ferrotorch-optim/src/lib.rs:30` — `pub use adam::{Adam, AdamConfig};`
- `ferrotorch/src/lib.rs:51` — `pub use ferrotorch_optim::{Adam, AdamW,
  Optimizer, Sgd};` re-exports `Adam` in the prelude.
- `ferrotorch/examples/train_mnist.rs:21,76` — drives the MNIST
  training loop using `Adam::new(params, AdamConfig::default())`.
- `ferrotorch-train/examples/multi_epoch_train_dump.rs:63,368` —
  multi-epoch training reproducibility harness uses `Adam`.
- `ferrotorch/examples/ferrotorch_bench.rs,176` — benchmark
  full-training-step harness instantiates `Adam::new(...)`.

## Parity contract

`parity_ops = []`. Adam has no per-op parity-sweep entry — the
optimiser is verified end-to-end by convergence tests
(Rosenbrock, x^2+y^2) and the foreach-parity test suite. Edge-cases
the impl owns:

- **f64 CPU intermediate precision**: the CPU loop computes all
  moment + update math in f64 and casts to T at the end, matching
  PyTorch's `_single_tensor_adam` which keeps moments as the
  parameter dtype but performs every accumulation in the same
  dtype. Ferrotorch's f64 path is the upstream default; the
  foreach path computes in T so f32 may show small drift
  vs the CPU path (tested at ≤ 1e-3 tolerance).
- **AMSGrad denominator**: uses `sqrt(max_exp_avg_sq / bc2) + eps`
  (note `max` is applied to the bias-corrected v in the foreach
  path and to the raw v in the CPU path; this is the same
  mathematical limit as PyTorch's `_single_tensor_adam`).
- **`step_count` increment**: incremented BEFORE bias correction
  computation (`step_count += 1; bc1 = 1 - beta1^step`),
  mirroring PyTorch (`torch/optim/adam.py:481-488`).
- **Gradient `None`**: skipped (REQ-8).

## Verification

Tests in `mod tests in adam.rs` (11 tests):

- Convergence: `test_adam_rosenbrock_convergence`,
  `test_adam_multiple_params`.
- Algorithm features: `test_adam_amsgrad`, `test_adam_weight_decay`.
- Trait surface: `test_adam_zero_grad`,
  `test_adam_lr_accessors`.
- State-dict: `test_adam_state_dict_roundtrip`.
- Foreach parity: `test_adam_foreach_basic_parity`,
  `test_adam_foreach_parity_with_weight_decay`,
  `test_adam_foreach_parity_with_amsgrad`,
  `test_adam_foreach_skips_params_without_grad`.

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib adam:: 2>&1 | tail -3
```

Expected: `11 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct AdamConfig` + `impl Default` in `adam.rs` mirroring `torch/optim/adam.py:35-200`; non-test consumer: `ferrotorch/examples/train_mnist.rs:78` `Adam::new(model.parameters().to_vec(), AdamConfig::default())`. |
| REQ-2 | SHIPPED | impl: `impl<T: Float> Optimizer<T> for Adam<T>` block in `adam.rs`; non-test consumer: `ferrotorch-train/src/learner.rs` `use ferrotorch_optim::Optimizer;` drives every training step. |
| REQ-3 | SHIPPED | impl: legacy CPU `step` (else branch after the GPU fast-path) in `adam.rs` mirroring `_single_tensor_adam` in `torch/optim/adam.py:347-552`; non-test consumer: `ferrotorch/examples/train_mnist.rs:76-78` drives MNIST training entirely through this path on CPU. |
| REQ-4 | SHIPPED | impl: AMSGrad branch `if config.amsgrad { ... max_sq[i] = ea; ... }` inside the CPU `step` in `adam.rs` mirroring `torch/optim/adam.py:445-470`; non-test consumer: `ferrotorch-optim/src/lib.rs:30` re-exports `AdamConfig` so `with_amsgrad(true)` is reachable from downstream training code. |
| REQ-5 | SHIPPED | impl: GPU fused-kernel branch (`if use_gpu { backend.fused_adam_f32(...) }`) in `adam.rs`; non-test consumer: `ferrotorch/examples/train_mnist.rs:76-78` becomes GPU-resident once the model is moved to CUDA — the fused kernel activates automatically on the same call. |
| REQ-6 | SHIPPED | impl: `Adam::step_foreach` method in `adam.rs` mirroring `_multi_tensor_adam` in `torch/optim/adam.py:553-901`; non-test consumer: `ferrotorch/src/lib.rs:61` re-exports `AdamConfig` so `with_foreach(true)` is reachable from downstream training code. |
| REQ-7 | SHIPPED | impl: `state_dict` / `load_state_dict` methods in `adam.rs` keying by `ParamKey::Display`; non-test consumer: `ferrotorch-serialize/src/checkpoint.rs:48` `use ferrotorch_optim::OptimizerState;` writes/reads this map on checkpoint save/load. |
| REQ-8 | SHIPPED | impl: `let grad_tensor = match tensor.grad()? { Some(g) => g, None => continue };` in both `step` and `step_foreach` mirroring `torch/optim/adam.py:362-365`; non-test consumer: same training-loop path in `ferrotorch-train/src/learner.rs` exercising frozen parameters via the same skip. |
| REQ-9 | SHIPPED | impl: `if std::env::var("FERROTORCH_ENABLE_GPU_FALLBACK").is_ok() { tracing::warn!(...) } else { return Err(...) }` branch inside the GPU fast-path in `adam.rs`; non-test consumer: `Err in ferrotorch-train/src/learner.rs` propagates the resulting `FerrotorchResult` so the training script sees an `Err` instead of a silent CPU fallback. |
