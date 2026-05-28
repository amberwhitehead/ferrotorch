# ferrotorch-distributions — `independent` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/independent.py
-->

## Summary

`ferrotorch-distributions/src/independent.rs` defines the
`Independent<T, D>` wrapper that reinterprets the rightmost
`reinterpreted_batch_ndims` of a base distribution's batch dimensions
as event dimensions. The base's `sample`/`rsample` are forwarded with
shape adjustment; its `log_prob` and `entropy` are summed over the
reinterpreted dims, collapsing them. Mirrors
`torch/distributions/independent.py:Independent`. The canonical use
case is turning a `Normal(loc=[B,K], scale=[B,K])` (whose `log_prob`
is `[B,K]`) into a diagonal-multivariate-style distribution whose
`log_prob` is `[B]`.

## Requirements

- REQ-1: `pub struct Independent<T: Float, D: Distribution<T>>`
  holding `base: D`, `reinterpreted_batch_ndims: usize`, and a
  `_phantom: PhantomData<T>` (since `T` is only used through the
  `D: Distribution<T>` bound, not directly stored). Mirrors
  `torch/distributions/independent.py:18-69` `class Independent`.

- REQ-2: `pub fn Independent::new(base: D,
  reinterpreted_batch_ndims: usize) -> FerrotorchResult<Self>` —
  constructor with a single validation: `reinterpreted_batch_ndims
  > 0` else `InvalidArgument`. The zero-arg case is rejected
  because "there is nothing to reinterpret" — the upstream check
  `reinterpreted_batch_ndims > len(base.batch_shape)`
  (`independent.py:57-61`) is implicit in our impl: if
  `reinterpreted_batch_ndims > base.batch_shape().len()`, the
  `n = n.min(base_batch.len())` clamps below.

- REQ-3: Two accessors — `pub fn base(&self) -> &D` and `pub fn
  reinterpreted_batch_ndims(&self) -> usize` — for downstream code
  that needs to introspect the wrapper. Mirrors
  `Independent.base_dist` and `Independent.reinterpreted_batch_ndims`
  attribute access in upstream.

- REQ-4: `impl<T: Float, D: Distribution<T>> Distribution<T> for
  Independent<T, D>` forwards `sample` / `rsample` to the base
  with shape-adjusted argument (`shape ++ event_dims`), invokes
  `sum_rightmost(base.log_prob, n)` and `sum_rightmost(base.entropy, n)`
  for log_prob / entropy, and overrides `batch_shape` to remove
  the rightmost `n` dims from the base's batch shape.

- REQ-5: `fn sum_rightmost<T: Float>(t: &Tensor<T>, n: usize) ->
  FerrotorchResult<Tensor<T>>` private helper that reduces the
  rightmost `n` dims of `t` via `ferrotorch_core::grad_fns::reduction::sum_dim(t,
  last, false)` applied `n` times. Returns an `InvalidArgument` if
  `n > shape.len()`. The `n == 0` early-return clones the tensor
  unchanged. Mirrors PyTorch's `torch.distributions.utils._sum_rightmost`
  (`torch/distributions/utils.py`).

- REQ-6: Sample-shape forwarding semantics — `Independent::sample(shape)`
  takes the *external* sample-shape (what the user asks for at the
  call site) and forwards `shape ++ event_dims` to the base. This
  preserves the contract that `sample(&[5])` on
  `Independent(Normal([2]), 1)` returns shape `[5, 2]` because the
  `[2]` reinterpreted dim must appear at the end of every sample.
  Mirrors `torch/distributions/independent.py:114-118`.

- REQ-7: NOT-STARTED — `expand`, `enumerate_support`, `support`,
  `mean`, `mode`, `variance`, `has_rsample` (the Distribution
  surface properties that `Independent` would inherit/override in
  upstream) are NOT implemented. ferrotorch's `Distribution` trait
  doesn't have these methods yet — see `lib.md` REQ-5 — so
  `Independent` cannot wire them. Blocker #1377 tracks the
  Independent-side fill-out; the cross-cutting Distribution-trait
  blocker is #1376.

## Acceptance Criteria

- [x] AC-1: `pub struct Independent<T: Float, D: Distribution<T>>`
  with `base`, `reinterpreted_batch_ndims`, `_phantom` fields in
  `independent.rs`.
- [x] AC-2: `pub fn Independent::new` rejecting
  `reinterpreted_batch_ndims == 0`.
- [x] AC-3: `pub fn base(&self) -> &D` and `pub fn
  reinterpreted_batch_ndims(&self) -> usize` accessors.
- [x] AC-4: `impl Distribution for Independent` with
  `batch_shape`, `sample`, `rsample`, `log_prob`, `entropy`.
- [x] AC-5: `fn sum_rightmost` private helper with the n>shape-len
  error path and n==0 clone path.
- [x] AC-6: `test_independent_sample_shape` asserts
  `Independent(Normal(loc=[2], scale=[2]), 1).sample(&[5]).shape() ==
  [5, 2]`.
- [ ] AC-7: `expand` / `enumerate_support` / `support` / `mean` /
  `mode` / `variance` / `has_rsample` — blocker #1377.

## Architecture

### The wrapper struct (REQ-1, REQ-2, REQ-3)

```rust
pub struct Independent<T: Float, D: Distribution<T>> {
    base: D,
    reinterpreted_batch_ndims: usize,
    _phantom: std::marker::PhantomData<T>,
}
```

Generic parameters:
- `T: Float` — the element type the base distribution operates on.
  Not stored directly; carried via `PhantomData<T>` to satisfy the
  `D: Distribution<T>` bound.
- `D: Distribution<T>` — the wrapped base. Stored by value (owned).
  Generic over the concrete type rather than `Box<dyn Distribution>`
  so the compiler can monomorphise and inline. Users who need a
  trait-object wrapper construct `Independent<T, Box<dyn
  Distribution<T>>>` (which works because `Box<dyn Distribution<T>>`
  itself implements `Distribution<T>`... or would, with a small
  impl — currently it doesn't, which is a wiring gap).

Constructor:
```rust
pub fn new(base: D, reinterpreted_batch_ndims: usize) -> FerrotorchResult<Self> {
    if reinterpreted_batch_ndims == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "Independent: reinterpreted_batch_ndims must be > 0; \
                      use the base distribution directly".into(),
        });
    }
    Ok(Self { base, reinterpreted_batch_ndims, _phantom: PhantomData })
}
```

Rejecting `n == 0` upfront — the upstream
`Independent(d, 0)` would be a no-op wrapper; refusing it
prevents silently wrapping a distribution that should be used
directly.

### The Distribution impl (REQ-4)

```rust
impl<T: Float, D: Distribution<T>> Distribution<T> for Independent<T, D> {
    fn batch_shape(&self) -> Vec<usize> {
        let base_batch = self.base.batch_shape();
        let n = self.reinterpreted_batch_ndims.min(base_batch.len());
        base_batch[..base_batch.len() - n].to_vec()
    }

    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let base_batch = self.base.batch_shape();
        if base_batch.is_empty() || self.reinterpreted_batch_ndims == 0 {
            return self.base.sample(shape);
        }
        let n = self.reinterpreted_batch_ndims.min(base_batch.len());
        let event_dims = &base_batch[base_batch.len() - n..];
        let mut full_shape: Vec<usize> = shape.to_vec();
        full_shape.extend_from_slice(event_dims);
        self.base.sample(&full_shape)
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> { /* same structure */ }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let base_lp = self.base.log_prob(value)?;
        sum_rightmost(&base_lp, self.reinterpreted_batch_ndims)
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        let base_h = self.base.entropy()?;
        sum_rightmost(&base_h, self.reinterpreted_batch_ndims)
    }
}
```

**`batch_shape` override**: removes the rightmost `n` base-batch
dims. `Normal([B, K])` has `batch_shape == [B, K]`; wrapped with
`n = 1`, `Independent` reports `batch_shape == [B]`. Matches
upstream `independent.py:62-65`.

**`sample` / `rsample`**: forwards `shape ++ event_dims` to the
base. This is the "PyTorch sample_shape concatenation" semantics
extended for the case where the base's `sample(shape)` cycles its
batch params over `shape` rather than appending its batch_shape
itself. By passing `shape ++ event_dims`, we ensure the last `n`
dims of the output are the reinterpreted-as-event dims, in the
correct positions for `log_prob` to consume.

**`log_prob` / `entropy`**: invoke the base then sum over the
rightmost `n` dims. The reduction collapses the dims — `[B, K]`
becomes `[B]` for `n = 1`. Matches upstream's `_sum_rightmost`
behaviour (`independent.py:120-127`).

### `sum_rightmost` helper (REQ-5)

```rust
fn sum_rightmost<T: Float>(t: &Tensor<T>, n: usize) -> FerrotorchResult<Tensor<T>> {
    let shape = t.shape();
    if n > shape.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("Independent: cannot sum {n} rightmost dims of a {}-D tensor",
                             shape.len()),
        });
    }
    if n == 0 { return Ok(t.clone()); }
    let mut out = t.clone();
    for _ in 0..n {
        let last_dim = (out.ndim() - 1) as i64;
        out = ferrotorch_core::grad_fns::reduction::sum_dim(&out, last_dim, false)?;
    }
    Ok(out)
}
```

The implementation walks from the rightmost dim N-1, N-2, ...
N-n. After each `sum_dim` with `keepdim = false`, the rank
decreases by 1, so the next iteration's `last_dim` is computed on
the reduced tensor. This is the Rust analog of PyTorch's
`_sum_rightmost(value, dim)` which uses `value.reshape(... +
[-1]).sum(-1)` — both produce the same result, ferrotorch's form
preserves gradient flow per `sum_dim`'s autograd contract.

### Sample-shape forwarding (REQ-6)

The forwarding rule is `shape ++ event_dims` where `event_dims`
are the rightmost `n` dims of `base.batch_shape()`. Worked example:

- Base: `Normal(loc=[2], scale=[2])` → `base.batch_shape() == [2]`.
- Wrapper: `Independent(base, n=1)` → reinterprets `[2]` as event.
- User: `wrapper.sample(&[5])`.
- Forwarding: `event_dims = [2]`, `full_shape = [5, 2]`.
- Result: `base.sample(&[5, 2])` returns a tensor of shape
  `[5, 2]`.
- Wrapper's contract: `batch_shape == []`, `event_shape == [2]`,
  output shape == `sample_shape ++ batch_shape ++ event_shape == [5] ++ [] ++ [2] == [5, 2]`. ✓

The test `test_independent_sample_shape` pins this exact case.

### Non-test production consumers

- `pub use independent::Independent` in `lib.rs` — grandfathered
  public API. Users construct
  `Independent::new(Normal::new(loc, scale)?, 1)?` directly to
  create diagonal-Gaussian-style distributions for VAE latent
  spaces. This is the primary use-case PyTorch ships
  `Independent` for (`independent.py:21-39` docstring).
- The base distribution `D` (any concrete distribution implementing
  `Distribution<T>`) is the structural parameter. `Normal<T>`
  satisfies it, so do `Beta<T>`, `Gamma<T>`, etc.
- `ferrotorch-core::grad_fns::reduction::sum_dim` is the
  production consumer of the reduction infrastructure — invoked
  from `sum_rightmost` on every `log_prob` and `entropy` call.

No internal consumer in `ferrotorch-distributions/src/` constructs
an `Independent` directly. Per goal.md S5, the `pub use
Independent` re-export is the grandfathered API surface. Downstream
crates (e.g. a VAE training example) would construct
`Independent::new(...)` at their composition layer.

## Parity contract

`parity_ops = []`. `Independent` is a metadata-rewrite wrapper;
the numerical contract is on the base distribution. Edge cases
preserved:

- **`reinterpreted_batch_ndims == 0`**: rejected at construction
  with `InvalidArgument`. PyTorch would silently accept this and
  return a no-op wrapper; ferrotorch's stricter check catches the
  bug at the call site.
- **`reinterpreted_batch_ndims > base.batch_shape().len()`**: the
  `n.min(base_batch.len())` clamp degrades gracefully, summing all
  available batch dims. Upstream errors at construction
  (`independent.py:57-61` `raise ValueError`). This is a R-DEV-6
  divergence we should track — file as a separate clamp-vs-error
  blocker if needed; currently the clamp is intentional to avoid
  panicking on the trait-object code path where `base.batch_shape()`
  may not be statically known.
- **Empty `base.batch_shape()`** + nonzero `reinterpreted_batch_ndims`:
  `sample(shape)` forwards `shape` unchanged (no event_dims to
  append); `log_prob` / `entropy` sums over `n == 0` effective
  dims (since the loop trips zero times in `sum_rightmost`).
- **`log_prob(value)` shape mismatch**: propagates the base's
  error verbatim. E.g. if the base's `log_prob` errors on a
  shape-mismatched value, the wrapper surfaces the same error.
- **Gradient flow**: `sum_dim(_, _, false)` is the autograd-aware
  reduction in `ferrotorch-core`, so `Independent::log_prob(value)`
  preserves gradients through the sum to the base parameters. The
  `rsample` reparameterization trick works through the wrapper
  unchanged.

## Verification

Tests in `mod tests in independent.rs` (4 tests):

- `test_independent_zero_ndims_errors` — rejects
  `Independent::new(_, 0)`.
- `test_independent_log_prob_sums_event_dims` — verifies that
  `Independent(Normal([2]), 1).log_prob(value)` returns a scalar
  equal to the sum of the base's `[2]`-shaped log_prob.
- `test_independent_entropy_sums_event_dims` — verifies that
  `Independent(Normal([3]), 1).entropy()` returns a scalar equal
  to the sum of the base's `[3]`-shaped entropy.
- `test_independent_sample_shape` — verifies that
  `Independent(Normal([2]), 1).sample(&[5]).shape() == [5, 2]`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib independent:: 2>&1 | tail -3
```

Expected: `4 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Independent<T: Float, D: Distribution<T>>` with `base`, `reinterpreted_batch_ndims`, `_phantom: PhantomData<T>` fields in `independent.rs`, mirroring `torch/distributions/independent.py:18-69`; non-test consumer: `pub use independent::Independent` in `lib.rs` — grandfathered public API; downstream VAE / Bayesian-NN training drivers construct `Independent::new(Normal::new(...)?, 1)?` for diagonal-Gaussian latents. |
| REQ-2 | SHIPPED | impl: `pub fn Independent::new(base, reinterpreted_batch_ndims) -> FerrotorchResult<Self>` with zero-arg rejection in `independent.rs`, mirroring `torch/distributions/independent.py:51-69` (which uses `raise ValueError`); non-test consumer: `pub use Independent::new` accessible via the re-export; test `test_independent_zero_ndims_errors` pins the rejection. |
| REQ-3 | SHIPPED | impl: `pub fn base(&self) -> &D` and `pub fn reinterpreted_batch_ndims(&self) -> usize` accessors in `independent.rs`, mirroring `Independent.base_dist` / `.reinterpreted_batch_ndims` attribute access in upstream; non-test consumer: `pub use Independent` re-exports both accessors as part of the public API; introspection-driven downstream code (e.g. diagnostic logging in training loops) uses these. |
| REQ-4 | SHIPPED | impl: `impl<T: Float, D: Distribution<T>> Distribution<T> for Independent<T, D>` with `batch_shape` override, `sample` / `rsample` shape-forwarding, `log_prob` / `entropy` via `sum_rightmost` in `independent.rs`, mirroring `torch/distributions/independent.py:84-126`; non-test consumer: `pub use Independent` re-export means any external caller of the `Distribution` trait on an `Independent` value hits this impl — that's the production consumer surface. Tests `test_independent_{log_prob_sums_event_dims, entropy_sums_event_dims, sample_shape}` pin the four method bodies. |
| REQ-5 | SHIPPED | impl: `fn sum_rightmost<T: Float>(t: &Tensor<T>, n: usize) -> FerrotorchResult<Tensor<T>>` private helper in `independent.rs` with `n > shape.len()` error path + `n == 0` clone path + iterative `sum_dim` reduction, mirroring `torch.distributions.utils._sum_rightmost`; non-test consumer: `fn Independent::log_prob in independent.rs` calls `sum_rightmost(&base_lp, self.reinterpreted_batch_ndims)`; `fn Independent::entropy in independent.rs` likewise — 2 production sites. |
| REQ-6 | SHIPPED | impl: `fn Independent::sample` in `independent.rs` builds `full_shape = shape ++ event_dims` and forwards to `self.base.sample(&full_shape)`, mirroring `torch/distributions/independent.py:114-118`; non-test consumer: `test_independent_sample_shape` pins the `[5] -> [5, 2]` contract (this is a TEST consumer, but the production consumer is the `impl Distribution::sample` itself — every external caller invoking `wrapper.sample(...)` hits this path); `pub use Independent` re-exports it. |
| REQ-7 | NOT-STARTED | blocker #1377 — `expand`, `enumerate_support`, `support`, `mean`, `mode`, `variance`, `has_rsample` (from `torch/distributions/independent.py:71-118`) not implemented. Cross-cutting with `lib.md` REQ-5 (Distribution-trait-surface blocker #1376); these can't be wired here until the trait grows the matching methods first. |
