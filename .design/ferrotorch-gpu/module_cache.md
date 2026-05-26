# Global PTX module + CudaFunction cache

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/cuda/CUDAContextLight.h
  - aten/src/ATen/cuda/jiterator.cpp
  - aten/src/ATen/cuda/CUDAGraphsUtils.cuh
-->

## Summary

`ferrotorch-gpu/src/module_cache.rs` is the global cache that
eliminates the ~1700 us per-kernel PTX compilation overhead by
keeping compiled `CudaFunction`s alive across all calls. Two parallel
caches: one keyed by `(&'static str kernel_name, u32 device_ordinal)`
for the per-kernel string constants in `kernels.rs` / `bf16.rs` /
`f16.rs` / etc., and a second hash-keyed cache
`OWNED_MODULE_CACHE` for runtime-generated PTX strings (the
FusedChain executor's runtime PTX synthesis). Mirrors the JIT
module-cache role that PyTorch's `jiterator.cpp` plays for runtime
fused kernels.

## Requirements

- REQ-1: `pub fn get_or_compile(ctx, ptx_src, kernel_name,
  device_ordinal) -> Result<CudaFunction, DriverError>` — the
  `&'static str`-keyed cache. Compiles PTX on first call for a
  `(kernel_name, device_ordinal)` pair, returns the cached
  `CudaFunction` clone thereafter.
- REQ-2: `pub fn get_or_compile_owned(ctx, ptx_src: &str,
  kernel_name, device_ordinal) -> Result<CudaFunction, DriverError>`
  — the hash-keyed cache for runtime-generated PTX. Computes a
  hash of the PTX body, looks up by `(hash, device_ordinal)`, and
  compiles on miss.
- REQ-3: Per-device-ordinal keying — a kernel compiled for device 0
  cannot be used on device 1, so the device ordinal is part of the
  cache key.
- REQ-4: Thread safety — both caches are
  `LazyLock<Mutex<HashMap<...>>>`. The critical section is a
  hash lookup + optional insert; contention is negligible per the
  module doc.
- REQ-5: Non-test production consumer — every PTX-loading callsite
  in ferrotorch-gpu (`kernels.rs`, `bf16.rs`, `f16.rs`,
  `int_kernels.rs`, `bool_kernels.rs`, `cast_kernels.rs`,
  `masked_kernels.rs`, `reduce_arg.rs`, `gather_int.rs`,
  `roll.rs`, `group_norm.rs`, `flash_attention.rs`, `conv.rs`,
  `upsample.rs`, `rng.rs`, etc.) calls `get_or_compile` /
  `get_or_compile_owned`.

## Acceptance Criteria

- [x] AC-1: `pub fn get_or_compile` exists at line 76 with the
  documented `(ctx, ptx_src: &'static str, kernel_name, device_ordinal)`
  signature.
- [x] AC-2: `pub fn get_or_compile_owned` exists at line 134 with
  the `(ctx, ptx_src: &str, kernel_name, device_ordinal)` signature.
- [x] AC-3: Both caches are static globals at lines 42 and 53,
  initialised via `LazyLock<Mutex<HashMap<...>>>`.
- [x] AC-4: 160+ `crate::module_cache::get_or_compile` callsites
  exist in `ferrotorch-gpu/src/` (verified by
  `grep -c "module_cache::get_or_compile" ferrotorch-gpu/src/*.rs`).
- [x] AC-5: Six unit tests in `mod tests` exercise the cache hit /
  miss / cross-device-key paths.

## Architecture

The module is small (~470 lines) and structurally simple:

1. **Two caches as static globals**:
   - `MODULE_CACHE: LazyLock<Mutex<HashMap<(&'static str, u32), CudaFunction>>>`
     at line 42 — the `&'static str` key is zero-cost because all
     kernel names in `kernels.rs` are string literals.
   - `OWNED_MODULE_CACHE: LazyLock<Mutex<HashMap<(u64, u32), CudaFunction>>>`
     at line 53 — the `u64` key is a blake-style hash computed
     over the PTX body on insert + lookup, avoiding the need to
     keep the `String` alive as a `&'static str`.

2. **`get_or_compile` body** (line 76):
   - Acquire mutex, lookup `(kernel_name, device_ordinal)`.
   - On hit: return `clone()` of the cached `CudaFunction`.
   - On miss: compile via `ctx.load_module(Ptx::from_src(ptx_src))`,
     extract the named function, insert into the cache, return the
     clone.

3. **`get_or_compile_owned` body** (line 134):
   - Compute the hash of `ptx_src` via `DefaultHasher`.
   - Acquire mutex, lookup `(hash, device_ordinal)`.
   - On hit: return the clone.
   - On miss: compile, insert, return the clone.

The `CudaFunction::clone()` is a refcount bump on cudarc's
`Arc<...>`-backed handle, not a recompile — cheap and safe to call
from any thread.

### Non-test production consumers (REQ-5)

Every PTX-loading file in ferrotorch-gpu's `src/` calls
`get_or_compile` (or `get_or_compile_owned` for runtime PTX). A
representative sample:

- `kernels.rs` — ~100 kernel-load sites (one per PTX constant).
- `bf16.rs`, `f16.rs`, `int_kernels.rs`, `bool_kernels.rs` — dtype-
  specialised kernel loads.
- `roll.rs` — `module_cache::get_or_compile(ctx, ROLL_F32_PTX, ...)`.
- `group_norm.rs` — `module_cache::get_or_compile(ctx,
  GROUP_NORM_PTX, ...)`.
- `masked_kernels.rs` — every `masked_fill_*` / `where_*` /
  `masked_select_*` / `masked_scatter_*` launcher.
- `reduce_arg.rs::launch_argreduce` — argmax/argmin PTX load.
- `gather_int.rs::launch_select` — gather/index_select PTX load.
- `flash_attention.rs`, `conv.rs`, `upsample.rs`, `rng.rs`,
  `cast_kernels.rs` — the rest of the GPU compute surface.

Total non-test call sites: 160+ across `ferrotorch-gpu/src/` (counted
via `grep -c "module_cache::" ferrotorch-gpu/src/*.rs`).

## Parity contract

`parity_ops = []` for this route. The cache is INFRASTRUCTURE — it
preserves byte-exact equivalence between cached and freshly compiled
`CudaFunction`s because the underlying `CudaFunction::clone()` is a
refcount-only operation. The cache cannot introduce a numerical
divergence; it can only fail (a `DriverError`) or hit.

Edge cases preserved:

- **First-call cost**: ~1700 us PTX compile per `(name, device)`
  pair, amortised across all subsequent calls.
- **Cross-device keying**: the `(kernel_name, device_ordinal)` key
  ensures kernels are compiled separately per device. Critical for
  multi-GPU setups where a device-0 module cannot be launched on
  device 1.
- **Owned-PTX hash collision**: `OWNED_MODULE_CACHE` uses
  `DefaultHasher::finish() -> u64`. Practical collision probability
  is negligible for the runtime PTX strings produced by FusedChain;
  a collision would surface as a kernel-name resolution failure
  inside `load_module`, which is then propagated as `DriverError`.
- **Mutex contention**: the critical section is a HashMap lookup +
  one of `clone` (hit) or compile-then-insert (miss). The miss
  path's compile is the expensive piece, and any contention is on
  the first call; subsequent hits are sub-microsecond.

## Verification

Unit tests in `ferrotorch-gpu/src/module_cache.rs` `mod tests` (6
tests) cover: cache hit returns the same function pointer, cache
miss compiles and inserts, cross-device-ordinal keys are distinct,
owned-PTX hash collision behaviour, and the `Mutex` poisoning path.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda module_cache:: 2>&1 | tail -3
```

Expected: ≥ 1 `test result: ok` line.

The cache is also indirectly exercised by every other
`cargo test -p ferrotorch-gpu --features cuda` invocation, since
every kernel load goes through this cache.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn get_or_compile in ferrotorch-gpu/src/module_cache.rs` (line 76) mirrors the upstream JIT-module-cache role; non-test consumer: 100+ sites in `ferrotorch-gpu/src/kernels.rs` plus every PTX-loading sibling module (10+ files). E.g. `roll.rs`, `group_norm.rs`, `reduce_arg.rs::launch_argreduce`. |
| REQ-2 | SHIPPED | impl: `pub fn get_or_compile_owned in module_cache.rs` (line 134) with the hash-keyed `OWNED_MODULE_CACHE`; non-test consumer: FusedChain runtime PTX executor (consumed by the workspace's fused-chain JIT path; the hash-keyed shape only exists because of that consumer). |
| REQ-3 | SHIPPED | impl: cache key is `(name, ordinal)` tuple at line 42 (`HashMap<(&'static str, u32), CudaFunction>`) and `(hash, ordinal)` at line 53; the unit-test suite verifies cross-device keys are distinct. |
| REQ-4 | SHIPPED | impl: both caches use `LazyLock<Mutex<HashMap<...>>>` at lines 42, 53; the lock-acquire / lookup / optional-compile / insert / drop pattern is the body of both `get_or_compile` and `get_or_compile_owned`. |
| REQ-5 | SHIPPED | impl: every kernel-loading site in `ferrotorch-gpu/src/` calls into this cache. Non-test consumers include `kernels.rs` (~100 sites), `bf16.rs`, `f16.rs`, `int_kernels.rs`, `bool_kernels.rs`, `cast_kernels.rs`, `masked_kernels.rs`, `reduce_arg.rs`, `gather_int.rs`, `roll.rs`, `group_norm.rs`, `flash_attention.rs`, `conv.rs`, `upsample.rs`, `rng.rs`. Counted via `grep -c "module_cache::" ferrotorch-gpu/src/*.rs` = 160+. |
