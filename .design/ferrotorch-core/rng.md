# Seeded RNG (`torch.manual_seed` parity)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/core/MT19937RNGEngine.h
  - aten/src/ATen/core/DistributionsHelper.h
  - aten/src/ATen/core/TransformationHelper.h
  - aten/src/ATen/CPUGeneratorImpl.cpp
  - torch/random.py
-->

## Summary

`ferrotorch-core/src/rng.rs` implements a thread-local seeded RNG that
mirrors `torch.manual_seed` / `torch.Generator`. It exposes:

- `pub struct Generator` — owns an MT19937 (Mersenne Twister 32-bit)
  engine + cached Box-Muller normal-distribution samples for f32 / f64.
- `pub fn manual_seed(seed: u64)` — top-level reseed of the current
  thread's default generator, mirroring `torch.manual_seed` at
  `torch/random.py:46`.
- `pub fn with_thread_rng<R>(f)` — closure-accessor over the
  thread-local generator; consumed by `creation::rand` / `creation::randn`
  and by the `ferrotorch-nn::init` helpers.

The internal MT19937 engine is byte-identical to PyTorch CPU's
`at::mt19937_engine` (`aten/src/ATen/core/MT19937RNGEngine.h:110-150`):
state array of 624 uint32, twist using `MATRIX_A = 0x9908b0df`, tempering
shifts identical (11, 7+0x9d2c5680, 15+0xefc60000, 18). The seeding
algorithm at `init_with_uint32` (`:155-164`) is the well-known
`1812433253 * (state[j-1] ^ (state[j-1] >> 30)) + j` recurrence.

## Requirements

- REQ-1 (MT19937 engine): the engine reproduces `at::mt19937_engine`
  byte-for-byte: `Generator::new(seed).random_u32()` agrees with
  `at::mt19937_engine(seed)()` for every state-array position over the
  full 624-element period (verified for the seed-42 prefix).
- REQ-2 (`Generator` newtype): expose `new(seed)`, `seed_from_entropy()`,
  `manual_seed(seed)`, `seed()`, `random_u32`/`random_u64`,
  `next_uniform_f32`/`f64`, `next_normal_f32`/`f64`. Implement
  `Clone + Debug + Default` (Default = `seed_from_entropy`).
- REQ-3 (`manual_seed` top-level): `pub fn manual_seed(seed: u64)` is
  the analogue of `torch.manual_seed` — reseeds the current thread's
  default generator. Re-exported at `ferrotorch_core::manual_seed`.
- REQ-4 (thread-local state): `thread_local! { static THREAD_RNG:
  RefCell<Generator> }` initialised lazily from `SystemTime` + thread
  id on first use; `with_thread_rng` borrows mutably for callers.
- REQ-5 (byte-exact parity for f32 rand): after
  `ferrotorch_core::manual_seed(s)`, `creation::rand::<f32>(&[N])`
  agrees with `torch.manual_seed(s); torch.rand(N)` byte-for-byte
  (mantissa MASK = (1<<24)-1, divisor = 1.0/(1<<24)f32 — see
  `TransformationHelper.h:84-89`).

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib rng::tests` passes
  (6 tests covering MT19937 seed-42 prefix, manual_seed reset,
  distinct-seed stream separation, generator clone, Box-Muller cache,
  random_u64 ordering).
- [x] AC-2: `cargo test -p ferrotorch-core --test
  divergence_manual_seed_parity` passes (5 tests: byte-exact vs torch
  for rand; deterministic randn; explicit-Generator independence).
- [x] AC-3: `creation::rand` / `creation::randn` consume bits from
  `with_thread_rng` instead of per-call xorshift seed. Verified by
  grep: no `xorshift_seed` / `xorshift_step` calls remain in
  `creation.rs` after the build.
- [x] AC-4: `ferrotorch-nn::init` initialisers expose
  `*_with_generator` variants taking `&mut Generator`. Default
  variants forward to `with_thread_rng`.

## Architecture

`Mt19937` is a private struct holding `state: [u32; 624]`, `next`,
`left`, and `seed`. `random_u32` implements the temper + reload pattern
from `MT19937RNGEngine.h:139-150`. `next_state` walks the state array
in two segments mirroring the C++ pointer-arithmetic loop. `random_u64`
is two `random_u32` calls concatenated as `(hi << 32) | lo`, matching
`CPUGeneratorImpl::random64` at `CPUGeneratorImpl.cpp:235-239`.

`Generator` wraps `Mt19937` and adds the Box-Muller cache slots
(`next_float_normal: Option<f32>`, `next_double_normal: Option<f64>`),
mirroring the `next_float_normal_sample_` / `next_double_normal_sample_`
fields on `at::CPUGeneratorImpl` (`CPUGeneratorImpl.cpp:244-271`).
`manual_seed(seed)` rebuilds the engine and drops the cached samples.

`next_uniform_f32` applies the upstream `uniform_real<float>` transform
(`TransformationHelper.h:84-89`): `(random_u32() & ((1<<24)-1)) * (1.0
/ (1<<24))` in f32. `next_uniform_f64` is the f64 analogue with the
53-bit mantissa mask and a single `random_u64` call.

`next_normal_f32` implements Box-Muller in f32 acctype (matching
`dist_acctype<float> = float` at `TransformationHelper.h:27`): drawn as
`u1 = uniform; u2 = uniform; r = sqrt(-2 * log1p(-u2)); theta = 2π *
u1; sample = r * sin(theta) cached; return r * cos(theta)`. The
`log1p(-u2)` form matches `DistributionsHelper.h:190`.

`THREAD_RNG: RefCell<Generator>` is lazy-initialised from
`SystemTime` + thread id on first use. Each rayon worker thread gets
its own state. `manual_seed(s)` is per-thread — callers wanting a
global seed broadcast must call `manual_seed` from each worker thread
explicitly.

**Non-test consumers**:
- `crate::creation::rand` at `creation.rs:127` invokes
  `with_thread_rng(|g| { for _ in 0..numel { data.push(g.next_uniform_f32()) } })`.
- `crate::creation::randn` at `creation.rs:165` invokes
  `with_thread_rng(|g| g.next_normal_f32())` per element.
- `ferrotorch_nn::init::uniform` / `normal` / `xavier_*` /
  `kaiming_*` / `trunc_normal_` / `orthogonal_` / `sparse_` all route
  through `with_thread_rng`; `*_with_generator` variants take an
  explicit `&mut Generator`.

## Parity contract

`parity_ops = []`. The RNG itself is not a `op_db` entry — it is
the prelude to every parity-sweep sample. The byte-exact agreement is
verified directly by
`ferrotorch-core/tests/divergence_manual_seed_parity.rs` against
captured `torch.manual_seed(42); torch.rand(10)` bit patterns.

`randn` byte-exact parity is NOT in scope of #1537. Torch's `randn`
splits the algorithm on `numel`: `< 16` uses
`cpu_serial_kernel(normal_distribution<double>(...)(gen))` then casts
down to scalar_t; `>= 16` uses `normal_fill` which writes uniform
samples in-place into the output buffer and then runs vectorised
Box-Muller over 16-element blocks pairing element `i` with element
`i+8` (`aten/src/ATen/native/cpu/DistributionTemplates.h:91-218`). The
SIMD path additionally vendors `sincos256_ps` from `avx_mathfun.h` so
that even matching torch's algorithm bit-perfectly would still require
linking the same libm. Tracked as a follow-up.

## Verification

```bash
cargo test -p ferrotorch-core --lib rng::tests
cargo test -p ferrotorch-core --test divergence_manual_seed_parity
```

Plus the standard gauntlet (clippy + fmt).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `Mt19937` engine at `rng.rs:29-115` mirrors `aten/src/ATen/core/MT19937RNGEngine.h:110-150` (state/twist/temper bit-identical); non-test consumer: `Generator::new` at `rng.rs:147-154` constructs the engine. Byte-exact prefix verified by `rng::tests::mt19937_seed_42_matches_torch_rand_f32`. |
| REQ-2 | SHIPPED | impl: `pub struct Generator` + methods at `rng.rs:127-235`; non-test consumer: `ferrotorch_nn::init::uniform_with_generator` at `ferrotorch-nn/src/init.rs:101-115` accepts `&mut Generator`. |
| REQ-3 | SHIPPED | impl: `pub fn manual_seed(seed)` at `rng.rs:265-269` mirrors `torch.manual_seed` (`torch/random.py:46`). Non-test consumer: re-exported at `ferrotorch-core/src/lib.rs` as `ferrotorch_core::manual_seed`. |
| REQ-4 | SHIPPED | impl: `thread_local! { static THREAD_RNG: RefCell<Generator> }` at `rng.rs:258-262`, lazily seeded from `Generator::seed_from_entropy`. Non-test consumer: `creation::rand`/`randn` at `ferrotorch-core/src/creation.rs:127,165` invoke `with_thread_rng`. |
| REQ-5 | SHIPPED | impl: `Generator::next_uniform_f32` at `rng.rs:200-205` applies `(random_u32() & ((1<<24)-1)) * (1.0/(1<<24))` mirroring `aten/src/ATen/core/TransformationHelper.h:84-89`. Non-test consumer: `creation::rand` at `creation.rs:127`. Byte-exact agreement with `torch.manual_seed(42); torch.rand(10)` pinned by `ferrotorch-core/tests/divergence_manual_seed_parity.rs:manual_seed_42_rand_byte_exact_vs_torch_f32`. |
