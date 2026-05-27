# `Kfac` — K-FAC natural gradient optimizer (Kronecker-Factored Approximate Curvature)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/optimizer.py   # CONTRACT ONLY (Optimizer base class)
academic-source:
  - Martens & Grosse 2015, "Optimizing Neural Networks with
    Kronecker-factored Approximate Curvature", ICML 2015, arXiv:1503.05671
reference-impl:
  - KFAC-PyTorch (github.com/alecwangcq/KFAC-Pytorch, KFACOptimizer)
rdev7-exemption: yes  # K-FAC has no torch.optim op; custom add
-->

## Summary

`ferrotorch-optim/src/natural_gradient.rs` defines `Kfac<T>` and
`KfacConfig`, an MVP K-FAC natural-gradient optimizer for dense Linear
layers. K-FAC (Martens & Grosse, ICML 2015) approximates the Fisher
Information Matrix per-layer as a Kronecker product of the input and
output-gradient covariance matrices:

```text
F_W ≈ E[a a^T] ⊗ E[g g^T]    (A ⊗ G)
F_W^{-1} ≈ A^{-1} ⊗ G^{-1}
natural_grad_W = G^{-1} @ grad_W @ A^{-1}
```

**K-FAC is NOT in upstream `torch.optim`** — there is no
`torch.optim.KFAC` op and no ATen kernel for it. Per **R-DEV-7**
(Rust-ecosystem analog), this module is a **custom add, exempt from the
op-level upstream-cite rule** (resolved as the answer to #1467). The
exemption is recorded in three places: the metadata header above, the
route comment in `tooling/translate-routes.toml`, and this paragraph.

The cite split is:

- **Algorithm** — the Kronecker-factored Fisher approximation, the EMA
  factor accumulation, Tikhonov damping, the `G^{-1} @ grad_W @ A^{-1}`
  preconditioner, and the periodic inverse-recompute interval all come
  from the **academic source**: Martens & Grosse 2015, *Optimizing
  Neural Networks with Kronecker-factored Approximate Curvature*, ICML
  2015 (**arXiv:1503.05671**), with the **KFAC-PyTorch** reference impl
  (`github.com/alecwangcq/KFAC-Pytorch`, `KFACOptimizer`) as the
  companion engineering reference.
- **Contract** — the only thing preserved from upstream PyTorch is the
  `Optimizer` base-class API surface (`step` / `state_dict` /
  `load_state_dict` / `add_param_group` / `zero_grad`), so the route's
  `upstream` points at `torch/optim/optimizer.py:339` (`class
  Optimizer`) — the file ferrotorch's `impl<T> Optimizer<T> for Kfac<T>`
  mirrors. `step` is the abstract `raise NotImplementedError` at
  `torch/optim/optimizer.py:1094`; `state_dict` at `:681`;
  `add_param_group` at `:1104`.

Because no torch op exists, the correctness contract is between this
implementation and the **closed-form K-FAC math** — verified
non-tautologically (R-CHAR-3) by `test_kronecker_identity_matches_dense_fisher`
(builds the dense Fisher `kron(A+λI, G+λI)`, solves it independently, and
asserts the reshaped solution equals the step's `G_d^{-1} @ grad @
A_d^{-1}` preconditioner — i.e. the Kronecker identity
`(A⊗G)^{-1} vec(grad) = vec(G^{-1} grad A^{-1})` holds) and
`test_damping_limit_recovers_scaled_sgd` (λ→∞ collapses the
preconditioned direction onto the scaled raw gradient).

## Requirements

- REQ-1: `pub struct KfacConfig` with `lr` (1e-3), `damping` (1e-3),
  `momentum` (0.9), `update_freq` (10), `weight_decay` (0.0),
  `maximize` (false). Builder-style `with_*` setters.
- REQ-2: `pub struct Kfac<T: Float>` with `new(params, config)` and
  full `Optimizer<T>` impl.
- REQ-3: `update_factors(param_name: &str, input_activation: &Tensor<T>,
  output_gradient: &Tensor<T>)` — exponential moving average update of
  the Kronecker factors:
  ```text
  A = momentum * A + (1 - momentum) * (a^T a) / batch
  G = momentum * G + (1 - momentum) * (g^T g) / batch
  ```
  Rejects 1-D inputs and batch-size mismatch with
  `FerrotorchError::InvalidArgument`.
- REQ-4: Per-`String`-keyed factor state in `factors: HashMap<String,
  KroneckerFactors<T>>`. K-FAC keeps the string-key scheme (NOT the
  CL-1122 `ParamKey` typed-key scheme) because `update_factors`
  accepts user-supplied names that need not match `"g{}_p{}"`.
- REQ-5: `invert_damped_tensor(matrix, damping)` solves
  `(matrix + damping * I) @ X = I` to obtain `X = (matrix + damping * I)^{-1}`,
  dispatching to `ferrotorch_core::linalg::solve` (cuSOLVER on CUDA,
  LAPACK on CPU). Caches the result in `KroneckerFactors::a_inv`/`g_inv`.
- REQ-6: `Kfac::step` recomputes inverses when
  `step_count % update_freq == 1` (or `update_freq <= 1`), then applies
  the preconditioned gradient `G^{-1} @ grad @ A^{-1}` with momentum
  and (L2) weight decay. CL-1105 Pattern B keeps tensors on the
  parameter's device throughout.
- REQ-7: `state_dict`/`load_state_dict` round-trip the Kronecker
  factors (downloaded to CPU and cast to `f64` for the wire format).
- REQ-8: `maximize: true` negates the gradient before preconditioning.
- REQ-9: Device migration — if the existing factors live on a
  different device than the current parameter (e.g., loaded from a
  CPU state_dict then params moved to CUDA), the factors are migrated
  once on first reuse.

## Acceptance Criteria

- [x] AC-1: `KfacConfig::default()` returns the documented defaults
  (`test_kfac_config_defaults`).
- [x] AC-2: Constructor accepts a parameter list and stores them in a
  single param group (`test_kfac_construction`).
- [x] AC-3: `update_factors` stores running averages of the outer
  products with the configured momentum coefficient
  (`test_update_factors_stores_running_averages`).
- [x] AC-4: EMA blending across multiple `update_factors` calls
  produces the expected exponentially-weighted average
  (`test_update_factors_ema_blending`).
- [x] AC-5: With identity factors (zero `update_factors` calls),
  `step` falls back to vanilla SGD on the gradient
  (`test_step_with_identity_factors_matches_sgd`).
- [x] AC-6: Convergence on a simple quadratic
  (`test_convergence_quadratic`).
- [x] AC-7: Convergence with non-trivial K-FAC factors
  (`test_convergence_with_kfac_factors`).
- [x] AC-8: `state_dict`/`load_state_dict` round-trip preserves the
  factor matrices (`test_state_dict_roundtrip`).
- [x] AC-9: `lr`/`set_lr` accessors (`test_kfac_lr_accessors`).
- [x] AC-10: Weight decay shrinks parameters
  (`test_kfac_weight_decay`).
- [x] AC-11: 1-D input to `update_factors` is rejected
  (`test_update_factors_rejects_1d`).
- [x] AC-12: Batch-size mismatch is rejected
  (`test_update_factors_rejects_batch_mismatch`).
- [x] AC-13 (CUDA): step preserves CUDA residence
  (`kfac_step_preserves_device_for_cuda_input`).
- [x] AC-14 (CUDA): factor inversion uses cuSOLVER on CUDA
  (`kfac_factor_inversion_uses_cuSOLVER_on_cuda`).
- [x] AC-15 (CUDA): CUDA step matches CPU within tolerance
  (`kfac_step_matches_cpu_within_tolerance`).
- [ ] AC-16: Convolutional-layer Kronecker factorization. NOT
  implemented — the module is currently dense-only. Recorded in the
  module-level doc-comment; not blocked because no upstream contract
  pins this.

## Architecture

### Config

`#[derive(Debug, Clone, Copy)]`, `#[non_exhaustive]`, six fields plus
six builder `with_*` setters.

### `KroneckerFactors<T>`

Per-parameter struct holding:

- `a_factor: Tensor<T>` — `[in_features, in_features]` 2-D tensor
  (EMA of `a^T a`).
- `a_size: usize` — `in_features`.
- `g_factor: Tensor<T>` — `[out_features, out_features]` 2-D tensor
  (EMA of `g^T g`).
- `g_size: usize` — `out_features`.
- `a_inv`, `g_inv: Option<Tensor<T>>` — cached inverses
  (recomputed every `update_freq` steps).
- `momentum_buf: Option<Tensor<T>>` — momentum buffer for the
  preconditioned gradient.

### `update_factors`

Accepts user-supplied `param_name`, the input activation `a: [batch,
in_features]`, and the output gradient `g: [batch, out_features]`.
Constructs `A_batch = (a^T @ a) / batch` and `G_batch = (g^T @ g) / batch`
on the activation's device via `tensor_matmul` (cuBLAS GEMM on CUDA),
then blends `A = mom * A + (1 - mom) * A_batch` (same for `G`).
Invalidates the cached inverses. Lazy-initializes the factor entry on
first call.

### `invert_damped_tensor`

Builds an identity on the matrix's device, computes `damped = matrix +
damping * I`, then calls `ferrotorch_core::linalg::solve(&damped,
&identity)` to obtain the inverse. On CUDA this dispatches to
`cusolver::gpu_solve_*` (LU + triangular solve via cusolverDn); on CPU
to LAPACK's `getrs`.

### `Kfac::step`

The trait method (`natural_gradient.rs`):

1. Increment `step_count`.
2. If `step_count % update_freq == 1 || update_freq <= 1`, recompute
   `a_inv`/`g_inv` for every factor entry.
3. For each parameter with a gradient:
   - Optionally negate (maximize).
   - L2 weight decay: `grad = grad + wd * param`.
   - If the parameter has a 2-D shape AND there is a matching factor
     entry: precondition via `G^{-1} @ grad @ A^{-1}` (`tensor_matmul`
     dispatches to cuBLAS GEMM on CUDA).
   - Apply momentum and the LR step.
4. Commit via `update_storage` inside `no_grad`.

### Why K-FAC keeps `String` keys (not `ParamKey`)

`update_factors(param_name: &str, ...)` accepts arbitrary user-supplied
names (layer names, parameter names, etc.) — not necessarily of the form
`"g{}_p{}"`. To avoid forcing the caller to construct a synthetic
`ParamKey`, K-FAC stays on `HashMap<String, KroneckerFactors<T>>`. The
`param_key_buf: String` field is a CL-1122 reuse-buffer that avoids the
per-step `format!()` allocation when `step()` constructs the
`"g{group}_p{param}"` lookup name.

### Non-test production consumers

`ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` re-exports
`Kfac` and `KfacConfig` as `ferrotorch::optim::{Kfac, KfacConfig}`.

## Parity contract

`parity_ops = []`. **K-FAC is not in upstream torch.optim**, so there is
no parity-sweep op and no torch oracle. #1467 is **resolved as the
R-DEV-7 custom-add exemption** (not by selecting a third-party Python
oracle): the correctness contract is between this implementation and the
**closed-form Martens & Grosse 2015 math**, verified by:

- `test_kronecker_identity_matches_dense_fisher` — builds the dense
  Fisher `M = kron(A+λI, G+λI)`, solves `M @ y = vec(grad^T)` via the
  independent `ferrotorch_core::linalg::{kron, solve}` path, reshapes,
  and asserts equality with the step's preconditioner `G_d^{-1} @ grad @
  A_d^{-1}` (`G_d = G+λI`, `A_d = A+λI`). This confirms the Kronecker
  identity `(A⊗G)^{-1} vec(X) = vec(G^{-1} X A^{-1})` holds for the
  damped factors — the algebraic heart of K-FAC.
- `test_damping_limit_recovers_scaled_sgd` — λ→∞ drives `(A+λI)^{-1} →
  (1/λ)I` and `(G+λI)^{-1} → (1/λ)I`, so the preconditioned direction
  collapses onto `(1/λ²) · grad` (scaled gradient descent), confirming
  the damping limit.
- `kfac_*_within_tolerance` CUDA tests cross-validate CPU↔CUDA agreement
  within `1e-6` for f64.

Edge cases the code owns:

- **`damping`** is added to BOTH `A` and `G` before inversion, so the
  preconditioned gradient direction is well-defined even for
  rank-deficient factors.
- **`update_freq <= 1`** forces inverse recomputation every step
  (max-frequency mode).
- **Factor migration across devices** — if the factor lives on CPU
  and the parameter migrates to CUDA, the factor is `.clone().to(device)`-d
  on the next `update_factors` call.
- **Parameter without a matching factor entry** — step falls back to
  vanilla SGD-with-momentum on the raw gradient.
- **1-D input** to `update_factors` — rejected with
  `InvalidArgument`.
- **Batch-size mismatch** between `a` and `g` — rejected with
  `InvalidArgument`.

## Verification

Tests in `mod tests` of `natural_gradient.rs` (15+ tests including
CUDA-conditional):

- `test_kfac_config_defaults`
- `test_kfac_construction`
- `test_update_factors_stores_running_averages`
- `test_update_factors_ema_blending`
- `test_step_with_identity_factors_matches_sgd`
- `test_convergence_quadratic`
- `test_convergence_with_kfac_factors`
- `test_kronecker_identity_matches_dense_fisher` (R-CHAR-3 — the
  Kronecker identity verified against the dense Fisher via
  `linalg::kron` + `linalg::solve`)
- `test_damping_limit_recovers_scaled_sgd` (R-CHAR-3 — λ→∞ damping limit)
- `test_state_dict_roundtrip`
- `test_kfac_lr_accessors`
- `test_kfac_weight_decay`
- `test_update_factors_rejects_1d`
- `test_update_factors_rejects_batch_mismatch`
- `kfac_step_preserves_device_for_cuda_input` (CUDA-conditional)
- `kfac_factor_inversion_uses_cuSOLVER_on_cuda` (CUDA-conditional)
- `kfac_step_matches_cpu_within_tolerance` (CUDA-conditional)

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib natural_gradient:: 2>&1 | tail -3
```

Expected: all CPU-path tests pass; CUDA tests run only with
`--features cuda` against a real GPU.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct KfacConfig` at `ferrotorch-optim/src/natural_gradient.rs` + builder setters at `ferrotorch-optim/src/natural_gradient.rs`; non-test consumer: `ferrotorch/src/lib.rs` `pub use ferrotorch_optim::*;` re-export. |
| REQ-2 | SHIPPED | impl: `pub struct Kfac<T>` at `ferrotorch-optim/src/natural_gradient.rs` + `impl<T: Float> Optimizer<T>` at `ferrotorch-optim/src/natural_gradient.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-3 | SHIPPED | impl: `pub fn update_factors` at `ferrotorch-optim/src/natural_gradient.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-4 | SHIPPED | impl: `factors: HashMap<String, KroneckerFactors<T>>` at `ferrotorch-optim/src/natural_gradient.rs` + lazy-init at `ferrotorch-optim/src/natural_gradient.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-5 | SHIPPED | impl: `invert_damped_tensor` at `ferrotorch-optim/src/natural_gradient.rs` dispatching to `ferrotorch_core::linalg::solve` (cuSOLVER on CUDA); non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-6 | SHIPPED | impl: `step` at `ferrotorch-optim/src/natural_gradient.rs` with the inverse-cache gate at `ferrotorch-optim/src/natural_gradient.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-7 | SHIPPED | impl: `state_dict` at `ferrotorch-optim/src/natural_gradient.rs` + `load_state_dict` at `ferrotorch-optim/src/natural_gradient.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-8 | SHIPPED | impl: maximize branch inside `step`'s gradient pre-processing (file body); non-test consumer: `ferrotorch/src/lib.rs` re-export. |
| REQ-9 | SHIPPED | impl: device migration at `ferrotorch-optim/src/natural_gradient.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-export. |
