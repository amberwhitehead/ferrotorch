# ferrotorch-distributions — `fallback` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/distribution.py
-->

## Summary

`ferrotorch-distributions/src/fallback.rs` is the opt-in CPU-fallback
gate every distribution method invokes before doing a host-side
`data_vec()` readback. It mirrors PyTorch's MPS fallback shape:
error by default when a CUDA tensor is passed to a method with no
GPU kernel, allow opt-in with the `FERROTORCH_ENABLE_GPU_FALLBACK`
environment variable, and log `tracing::warn!` on every fallback so
the slow path is loud. The module is `pub(crate)` — only the
distribution implementations inside this crate call into it.

## Requirements

- REQ-1: `pub(crate) const FALLBACK_ENV_VAR: &str =
  "FERROTORCH_ENABLE_GPU_FALLBACK"` declares the single env var
  every guarded method reads. The string literal IS the public
  contract surfaced to end users via the warning's text and the
  error's `op` field. Mirrors PyTorch's
  `PYTORCH_ENABLE_MPS_FALLBACK` (`torch/__init__.py` user-facing
  documentation; ferrotorch's analog points at CUDA because we ship
  CUDA, not MPS).

- REQ-2: `pub(crate) fn check_gpu_fallback_opt_in<T: Float>(inputs:
  &[&Tensor<T>], op: &'static str) -> FerrotorchResult<()>` is the
  single entry point. Behaviour:
    1. All CPU inputs → `Ok(())`.
    2. Any CUDA input + env-var set → `Ok(())` with
       `tracing::warn!(target = "ferrotorch::gpu_fallback", ...)`.
    3. Any CUDA input + env-var unset →
       `Err(FerrotorchError::NotImplementedOnCuda { op })`.

  The `op` argument is the `"<DistributionName>::<method>"` literal
  that names the call site in both the warning and the error. R-DEV-7
  applies: PyTorch's MPS fallback uses `UserWarning`, we use
  `tracing::warn!` because that is the Rust ecosystem analog.

- REQ-3: The guard is `pub(crate)` not `pub`. End users of
  ferrotorch-distributions do NOT call this function directly; every
  distribution method that does CPU compute (the entire crate today)
  invokes it as the first line of its body before reading host
  buffers. This is enforced by the `pub(crate) fn` visibility plus
  the `pub(crate) mod fallback` declaration in `lib.rs`.

## Acceptance Criteria

- [x] AC-1: `pub(crate) const FALLBACK_ENV_VAR: &str` declares the
  env var literal in `fallback.rs`.
- [x] AC-2: `pub(crate) fn check_gpu_fallback_opt_in<T: Float>` has
  the exact three-arm behaviour above.
- [x] AC-3: `pub(crate) mod fallback;` in `lib.rs` keeps the module
  crate-internal.
- [x] AC-4: Tests in `mod tests` exercise the three arms with
  `with_env_set` / `with_env_unset` helpers serialised through a
  module-local `ENV_LOCK: Mutex<()>` so cargo's parallel test
  threading doesn't race on the process env var. The CUDA-input
  arms are gated `#[cfg(feature = "cuda")]` so they compile out on
  machines without GPUs.

## Architecture

### The fallback policy (REQ-1, REQ-2)

The guard's three-arm semantics are exactly what PyTorch ships for
MPS: silent fallback would hide the CPU↔GPU round trip
(R-CODE-4 forbids), so we error by default. The opt-in is the
environment variable, matching upstream's `PYTORCH_ENABLE_MPS_FALLBACK`
contract. Every call site looks like:

```rust
fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    crate::fallback::check_gpu_fallback_opt_in(
        &[&self.loc, &self.scale, value],
        "Normal::log_prob",
    )?;
    // CPU compute body unchanged...
}
```

The `op` literal is what surfaces in the user's terminal when they
hit the slow path. We use `"<Type>::<method>"` consistently so
diagnostics group naturally.

### `tracing::warn!` per call (REQ-2)

The warning fires on every guarded call — not once-per-process —
because the round-trip cost compounds across batches. A user running
1000 evaluation steps with the env var set should see 1000 warnings
in the trace if they're hitting the slow path; that's the signal to
either disable validation or work the prerequisite blocker that
ships per-distribution GPU kernels. We deliberately do NOT
de-duplicate the warnings via `OnceLock` — loudness is the
contract.

### `pub(crate)` visibility (REQ-3)

`pub(crate) fn check_gpu_fallback_opt_in` + `pub(crate) mod fallback`
in `lib.rs` together mean the function is invisible to consumers
outside the crate. This is intentional: the fallback policy is a
crate-internal implementation detail every distribution
participates in, not a public extension point. If a downstream user
needs the policy, they write their own distribution and the trait's
`fn sample(&self, ...) -> FerrotorchResult<Tensor<T>>` signature
doesn't constrain them to use ours.

### Non-test production consumers

Confirmed via `grep -rn "crate::fallback::check_gpu_fallback_opt_in"
ferrotorch-distributions/src/`:

- `bernoulli.rs` (3 sites: `sample`, `rsample`, `log_prob`)
- `beta.rs` (multiple sites: every CPU compute method)
- `cauchy.rs` (8 sites: `sample`, `rsample`, `log_prob`, `entropy`,
  `cdf`, `icdf`, `mean`, `variance`)
- `dirichlet.rs`, `exponential.rs`, `gamma.rs`, `gumbel.rs`,
  `half_normal.rs`, `kl.rs` (12 sites — one per KL formula),
  `kumaraswamy.rs`, `laplace.rs`, `lognormal.rs`,
  `low_rank_multivariate_normal.rs`, `mixture_same_family.rs`,
  `multinomial.rs`, `multivariate_normal.rs`, `normal.rs`,
  `one_hot_categorical.rs`, `pareto.rs`, `poisson.rs`,
  `relaxed_bernoulli.rs`, `relaxed_one_hot_categorical.rs`,
  `student_t.rs`, `uniform.rs`, `von_mises.rs`, `weibull.rs` (8
  sites: every CPU compute method).

Every method in every distribution that does host readback invokes
the guard as its first statement. The crate-wide invariant is
auditable mechanically: `grep -rL "fallback::check_gpu_fallback_opt_in"`
returns any file that has CPU compute but skipped the guard.

## Parity contract

`parity_ops = []`. The fallback gate is a runtime-policy decision,
not a numerical op. Edge cases the gate preserves:

- **Empty inputs slice**: `inputs: &[]` → `any_cuda = false` →
  `Ok(())`. Exercised by `empty_inputs_ok` test.
- **Mixed CPU + CUDA inputs**: treated as CUDA — any single CUDA
  input triggers the policy. Exercised by
  `mixed_cpu_and_cuda_inputs_treated_as_cuda` test.
- **Env var with empty string value**: `std::env::var` returns
  `Ok("")` for an unset-to-empty var; we treat presence (`.is_ok()`)
  as enabling fallback. This matches PyTorch's
  `PYTORCH_ENABLE_MPS_FALLBACK` which is enabled when set to any
  value including `"0"`.
- **Thread safety of the env-var read**: `std::env::var` takes the
  process-global env lock on every call. We do NOT cache the
  result; if a user toggles the env var mid-training, the next
  guarded call sees the change. The mutex in tests is for the
  serialised mutation arm, not the read arm.

## Verification

Tests in `mod tests in fallback.rs` (5 tests):

- `cpu_inputs_ok_regardless_of_env_var` — always-on path.
- `empty_inputs_ok` — empty slice path.
- `cuda_input_errors_when_env_var_unset` (cfg cuda) — error path.
- `cuda_input_ok_when_env_var_set_with_cpu_correct_result`
  (cfg cuda) — opt-in path + sanity-check the readback returns the
  original values.
- `mixed_cpu_and_cuda_inputs_treated_as_cuda` (cfg cuda) — any-CUDA
  policy.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib fallback:: 2>&1 | tail -3
```

Expected: `2 passed` without `--features cuda` (the three
CUDA-gated tests compile out); `5 passed` with `--features cuda` on
a machine with a CUDA device.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub(crate) const FALLBACK_ENV_VAR: &str = "FERROTORCH_ENABLE_GPU_FALLBACK"` in `fallback.rs`, mirroring PyTorch's `PYTORCH_ENABLE_MPS_FALLBACK` env-var contract (R-DEV-2: user-facing API match); non-test consumer: the literal is read by every `check_gpu_fallback_opt_in` call across 20+ distribution methods; the warning's `env_var = FALLBACK_ENV_VAR` field surfaces it to end users in the `tracing` stream. |
| REQ-2 | SHIPPED | impl: `pub(crate) fn check_gpu_fallback_opt_in<T: Float>` in `fallback.rs` with the three-arm Ok / Ok+warn / Err structure, mirroring PyTorch's MPS-fallback shape; non-test consumer: `fn Normal::log_prob in normal.rs` calls it as the first statement of every CPU-compute method (and likewise for every distribution — confirmed via grep, 20+ production call sites including `fn kl_normal_normal in kl.rs`, `fn LogNormal::sample in lognormal.rs`, `fn Cauchy::cdf in cauchy.rs`). |
| REQ-3 | SHIPPED | impl: `pub(crate) mod fallback;` declaration in `lib.rs` and `pub(crate) fn check_gpu_fallback_opt_in` visibility in `fallback.rs` together make the function callable only from within `ferrotorch-distributions`; non-test consumer: the visibility IS enforced — the crate's own distribution modules call the function, and the public API surface (`pub use Normal`, etc.) does not re-export it. `cargo doc -p ferrotorch-distributions` omits it from the public docs by virtue of `pub(crate)`. |
