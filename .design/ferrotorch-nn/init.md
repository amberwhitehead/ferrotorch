# ferrotorch-nn — `init` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/init.py
-->

## Summary

`ferrotorch-nn/src/init.rs` implements the in-place weight-initialization
helpers that mirror `torch.nn.init`. Each function operates on a `Parameter<T>`
and replaces its underlying tensor with freshly sampled / structured data while
preserving the `requires_grad` flag. Covers constant / zeros / ones / uniform /
normal fills, Glorot (Xavier) and He (Kaiming) variants, plus the structured
initializers `trunc_normal_`, `orthogonal_`, `sparse_`, `dirac_`, and `eye_`.

## Requirements

- REQ-1: `pub enum NonLinearity` carrying the gain table used by Kaiming
  initializers. Mirrors `torch.nn.init.calculate_gain` at
  `torch/nn/init.py:173-244` (`linear`, `sigmoid`, `tanh`, `relu`,
  `leaky_relu(negative_slope)`).

- REQ-2: `pub fn constant`, `pub fn zeros`, `pub fn ones` — bulk-fill the
  parameter with a scalar. Mirror `torch.nn.init.{constant_,zeros_,ones_}` at
  `torch/nn/init.py:337-378`.

- REQ-3: `pub fn uniform`, `pub fn normal` — sample from
  `U(low, high)` / `N(mean, std)` using an internal xorshift PRNG seeded
  from system time + thread id. Mirror `torch.nn.init.{uniform_,normal_}` at
  `torch/nn/init.py:247-300`.

- REQ-4: `pub fn xavier_uniform`, `pub fn xavier_normal` — Glorot initializers
  using `fan_in + fan_out`. Mirror `torch.nn.init.{xavier_uniform_,xavier_normal_}`
  at `torch/nn/init.py:479-540`.

- REQ-5: `pub fn kaiming_uniform`, `pub fn kaiming_normal` — He initializers
  parameterised by `NonLinearity`. Mirror `torch.nn.init.{kaiming_uniform_,kaiming_normal_}`
  at `torch/nn/init.py:554-672`. Uses `gain / sqrt(fan_in)` for the std (the
  `fan_in` mode in upstream); `fan_out` mode is NOT-STARTED.

- REQ-6: `pub fn trunc_normal_` — rejection-sampled truncated normal on
  `[a, b]`. Mirrors `torch.nn.init.trunc_normal_` at `torch/nn/init.py:301-336`.

- REQ-7: `pub fn orthogonal_` — modified Gram-Schmidt QR with sign correction,
  scaled by `gain`. Handles 2D and higher-rank via the
  `[rows, cols=product(shape[1..])]` reshape. Mirrors
  `torch.nn.init.orthogonal_` at `torch/nn/init.py:672-722`.

- REQ-8: `pub fn sparse_` — 2D-only column-wise sparsification: per column,
  randomly zero `sparsity` fraction of rows using partial Fisher-Yates;
  non-zero values from `N(0, std)`. Mirrors `torch.nn.init.sparse_` at
  `torch/nn/init.py:723-764`.

- REQ-9: `pub fn dirac_` — center-element identity mapping for conv weights,
  with `groups` support. Mirrors `torch.nn.init.dirac_` at
  `torch/nn/init.py:402-455`.

- REQ-10: `pub fn eye_` — 2D identity matrix (top-left for non-square).
  Mirrors `torch.nn.init.eye_` at `torch/nn/init.py:381-401`.

- REQ-11: Every initializer preserves the `Parameter`'s `requires_grad` flag.
  Mirrors upstream's `with torch.no_grad():` discipline at
  `torch/nn/init.py:69-160`.

## Acceptance Criteria

- [x] AC-1: `pub enum NonLinearity::{Linear,Sigmoid,Tanh,ReLU,LeakyReLU(f64)}`
  with the gain table.
- [x] AC-2: `zeros` / `ones` / `constant` populate every element.
- [x] AC-3: `uniform(low, high)` bounds every element into `[low, high]`.
- [x] AC-4: `normal(mean, std)` produces a sample whose empirical mean and
  variance approximate the targets (test `test_normal_init_stats`).
- [x] AC-5: `xavier_uniform` / `xavier_normal` use `fan_in + fan_out`.
- [x] AC-6: `kaiming_uniform` / `kaiming_normal` use `gain / sqrt(fan_in)`.
- [x] AC-7: `trunc_normal_` rejects `a >= b` and `std <= 0`.
- [x] AC-8: `orthogonal_` satisfies `Q^T Q ≈ I` (gain=1) and `Q^T Q ≈ gain^2 I`.
- [x] AC-9: `sparse_` 2D-only, per-column sparsity ratio matches request.
- [x] AC-10: `dirac_` places center-element 1s along the identity diagonal
  per group.
- [x] AC-11: `eye_` square / tall / wide.
- [x] AC-12: Every initializer preserves `requires_grad`.
- [ ] AC-13: Kaiming `fan_out` mode — NOT-STARTED (blocker #1453).

## Architecture

### Gain table (REQ-1)

`pub enum NonLinearity` in `init.rs`, with `fn gain` returning
`1.0` (linear/sigmoid), `5.0/3.0` (tanh), `sqrt(2)` (relu), and
`sqrt(2 / (1 + slope^2))` (leaky_relu). Matches the Python branches of
`calculate_gain` at `torch/nn/init.py:173-244`.

### Fan computation

The private `fn compute_fans` mirrors `_calculate_fan_in_and_fan_out` at
`torch/nn/init.py:458-476`:
- 0-D rejected.
- 1-D → `fan_in = fan_out = shape[0]` (degenerate; PyTorch raises but
  ferrotorch returns the input length to match the upstream sentinel use
  inside RNN layer constructors).
- 2-D → `fan_in = shape[1]`, `fan_out = shape[0]`.
- N-D → `fan_in = shape[1] * product(shape[2..])`,
  `fan_out = shape[0] * product(shape[2..])`.

### PRNG (REQ-3)

All sampling routes through `ferrotorch_core::rng::Generator` (MT19937 +
Box-Muller). The thread-local generator (seeded from `SystemTime` + thread
id on first use) is the default; every initializer also has an explicit
`*_with_generator` variant that takes `&mut Generator`, mirroring the
`generator` kwarg of the corresponding `torch.nn.init.*_` upstream
helper. Covered variants: `uniform_with_generator`,
`normal_with_generator`, `xavier_uniform_with_generator`,
`xavier_normal_with_generator`, `kaiming_uniform_with_generator`,
`kaiming_normal_with_generator`, `trunc_normal_with_generator`,
`orthogonal_with_generator`, `sparse_with_generator`. `dirac_` and
`eye_` are deterministic (no `generator` kwarg upstream either) so they
have no `*_with_generator` variant. The original blocker #1454
(deterministic-init plumbing) is CLOSED by these wired variants.

### Constant / uniform / normal (REQ-2, REQ-3)

`fn constant`, `fn zeros` (delegates to `constant(zero)`), `fn ones`
(delegates to `constant(one)`), `fn uniform(low, high)`, `fn normal(mean, std)`
all rebuild the parameter with a fresh `Tensor::from_storage(...)` while
preserving `requires_grad=true`.

### Glorot / He (REQ-4, REQ-5)

`fn xavier_uniform` / `fn xavier_normal` compute the limit / std from
`fan_in + fan_out` and delegate to the underlying uniform/normal helpers.

`fn kaiming_uniform` / `fn kaiming_normal` take a `NonLinearity` for the
gain. Upstream supports both `fan_in` (default) and `fan_out` modes at
`torch/nn/init.py:543-552`; ferrotorch implements only `fan_in`. The
`fan_out` mode is NOT-STARTED — tracked by blocker #1453.

### Structured initializers (REQ-6..10)

`trunc_normal_` (`init.rs`) does rejection sampling with batch over-sample of
`remaining * 2 + 64` to amortise the rejection-loop overhead. Rejects
`a >= b` and `std <= 0`.

`orthogonal_` reshapes higher-rank weights to 2-D `[rows,
cols=product(shape[1..])]`, transposes when `rows < cols`, runs modified
Gram-Schmidt with `r_diag` sign correction, scales by `gain`, then writes
back into the parameter's original shape. Rejects rank < 2.

`sparse_` requires a 2D weight, samples `N(0, std)` for all elements, then
per column uses partial Fisher-Yates to pick `ceil(rows * sparsity)`
indices to zero. Rejects rank != 2 and sparsity outside `[0, 1)`.

`dirac_` requires rank >= 3 (conv weights). For each group, places `1.0` at
the kernel center along the channel diagonal. Rejects `groups == 0` and
`out_channels % groups != 0`.

`eye_` requires rank == 2; sets `data[i*cols + i] = 1` for `i in 0..min(rows,cols)`.

### `requires_grad` preservation (REQ-11)

Every initializer calls `Parameter::new(Tensor::from_storage(.., true))`,
matching upstream's `with torch.no_grad():` block at
`torch/nn/init.py:69-160`.

### Non-test production consumers

- `ferrotorch-nn/src/rnn.rs:127-128, 575-576, 932-933, 1138-1139, 1360-1361,
  1618-1619` — every `RNN`/`GRU`/`LSTM` constructor calls
  `init::uniform(&mut weight_*, -k, k)` to seed `[-1/sqrt(hidden_size),
  1/sqrt(hidden_size)]` (matches upstream Pytorch RNN init).
- `ferrotorch-nn/src/embedding.rs:249, 640` —
  `Embedding::new` / `EmbeddingBag::new` call `init::normal(&mut weight, 0.0,
  1.0)` to mirror `torch.nn.Embedding.reset_parameters`.
- `ferrotorch-nn/src/lora.rs:109, 172` — `LoRALinear::new` /
  `LoRALinear::reset_parameters` call `init::normal` on the low-rank
  matrices with `std = 1/sqrt(rank)`.

## Parity contract

`parity_ops = []`. There are no parity-sweep ops for `nn.init` — the
initializers are statistical / structural and not byte-exact against
PyTorch's runtime. Coverage relies on the in-file `#[test]` block (35
tests) which pins:

- Bounds: `test_uniform_init_bounds`, `test_kaiming_uniform_relu`,
  `test_trunc_normal_bounds`.
- Statistics (mean / variance / sparsity ratio): `test_normal_init_stats`,
  `test_xavier_normal_stats`, `test_kaiming_normal_relu`,
  `test_trunc_normal_stats`, `test_sparse_sparsity_ratio`.
- Structural identities: `test_orthogonal_columns_orthonormal`,
  `test_orthogonal_gain`, `test_orthogonal_tall_matrix`,
  `test_orthogonal_wide_matrix`, `test_dirac_3d_identity`,
  `test_dirac_4d_identity`, `test_dirac_groups`, `test_eye_square`,
  `test_eye_tall`, `test_eye_wide`.
- Argument validation: `test_trunc_normal_rejects_bad_bounds`,
  `test_trunc_normal_rejects_zero_std`,
  `test_orthogonal_rejects_1d`, `test_sparse_rejects_non_2d`,
  `test_sparse_rejects_bad_sparsity`, `test_dirac_rejects_2d`,
  `test_eye_rejects_non_2d`.
- `requires_grad` preservation: `test_init_preserves_requires_grad`,
  `test_eye_preserves_requires_grad`.

## Verification

```bash
cargo test -p ferrotorch-nn --lib init:: 2>&1 | tail -3
```

Expected: `35 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum NonLinearity` and `fn gain` in `init.rs`, mirroring `torch/nn/init.py:173-244`; non-test consumer: `ferrotorch-nn/src/lib.rs:207` re-exports `init::NonLinearity` as part of the `ferrotorch_nn` public surface. |
| REQ-2 | SHIPPED | impl: `pub fn constant`, `pub fn zeros`, `pub fn ones` in `init.rs`, mirroring `torch/nn/init.py:337-378`; non-test consumer: parameter-fill discipline relied on across the workspace via the `init` module path (re-export at `lib.rs` makes the namespace public). |
| REQ-3 | SHIPPED | impl: `pub fn uniform`, `pub fn normal` in `init.rs` with xorshift PRNG, mirroring `torch/nn/init.py:247-300`; non-test consumer: `ferrotorch-nn/src/rnn.rs:127-128` (`init::uniform(&mut weight_ih, -k, k)?`) and `ferrotorch-nn/src/embedding.rs:249` (`init::normal(&mut weight, 0.0, 1.0)?`). |
| REQ-4 | SHIPPED | impl: `pub fn xavier_uniform`, `pub fn xavier_normal` in `init.rs`, mirroring `torch/nn/init.py:479-540`; non-test consumer: callable through the public `init` namespace; tests pin the stats. |
| REQ-5 | SHIPPED | impl: `pub fn kaiming_uniform`, `pub fn kaiming_normal` in `init.rs` (fan_in mode only), mirroring `torch/nn/init.py:554-672`; non-test consumer: re-exported via `init` module + ferrotorch-nn `lib.rs:207`. Tests `test_kaiming_uniform_relu`, `test_kaiming_normal_relu` validate the std formula. |
| REQ-6 | SHIPPED | impl: `pub fn trunc_normal_` + `pub fn trunc_normal_with_generator` in `init.rs` (rejection sampling), mirroring `torch/nn/init.py:301-336` (incl. `generator` kwarg); non-test consumer: re-exported via the `init` module; downstream ViT/position-embedding code in the model crates uses it. Tests `test_trunc_normal_bounds`, `test_trunc_normal_stats` pin behaviour; `trunc_normal_with_generator_uses_explicit_stream` in `ferrotorch-nn/tests/divergence_manual_seed_init_threading_extended.rs` pins generator threading. |
| REQ-7 | SHIPPED | impl: `pub fn orthogonal_` + `pub fn orthogonal_with_generator` in `init.rs` (modified Gram-Schmidt with sign correction), mirroring `torch/nn/init.py:672-722` (incl. `generator` kwarg); non-test consumer: re-exported via the `init` module path. Tests `test_orthogonal_columns_orthonormal`, `test_orthogonal_gain`, `test_orthogonal_tall_matrix`, `test_orthogonal_wide_matrix` pin `Q^T Q ≈ gain^2 I`; `orthogonal_with_generator_uses_explicit_stream` pins generator threading. |
| REQ-8 | SHIPPED | impl: `pub fn sparse_` + `pub fn sparse_with_generator` in `init.rs` (column-wise partial Fisher-Yates), mirroring `torch/nn/init.py:723-764` (incl. `generator` kwarg); non-test consumer: re-exported via the `init` module path. Tests `test_sparse_sparsity_ratio`, `test_sparse_nonzero_drawn_from_normal` pin behaviour; `sparse_with_generator_uses_explicit_stream` pins generator threading (covers BOTH N(0,std) sampling AND Fisher-Yates index draws). |
| REQ-9 | SHIPPED | impl: `pub fn dirac_` in `init.rs` (channel-diagonal center placement), mirroring `torch/nn/init.py:402-455`; no `generator` kwarg upstream — `dirac_` is deterministic, consumes 0 random bits; non-test consumer: re-exported via the `init` module path. Tests `test_dirac_3d_identity`, `test_dirac_4d_identity`, `test_dirac_groups` pin behaviour. |
| REQ-10 | SHIPPED | impl: `pub fn eye_` in `init.rs`, mirroring `torch/nn/init.py:381-401`; non-test consumer: re-exported via the `init` module path. Tests `test_eye_square`, `test_eye_tall`, `test_eye_wide`, `test_eye_preserves_requires_grad` pin behaviour. |
| REQ-11 | SHIPPED | impl: every initializer rebuilds the parameter via `Parameter::new(Tensor::from_storage(.., true))?` in `init.rs`; non-test consumer: `ferrotorch-nn/src/rnn.rs:127-128` — calls `init::uniform` and continues using the parameter as a leaf with grad, exercising the preservation contract. Test `test_init_preserves_requires_grad` pins. |
