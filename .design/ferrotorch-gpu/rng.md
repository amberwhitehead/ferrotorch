# CUDA Philox RNG (per-device state + GPU uniform/normal kernels)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/cuda/CUDAGeneratorImpl.cpp
  - aten/src/ATen/cuda/CUDAGeneratorImpl.h
  - aten/src/ATen/cuda/PhiloxCudaState.h
  - aten/src/ATen/cuda/PhiloxUtils.cuh
-->

## Summary

`ferrotorch-gpu/src/rng.rs` implements the Philox 4x32-10
counter-based RNG that mirrors CUDA's cuRAND / PyTorch's
`CUDAGeneratorImpl`. Three layers: (1) the algorithmic core
`PhiloxGenerator` + serialisable `PhiloxState`, (2) a per-device
registry `CudaRngManager` accessed via the
`cuda_rng_manager()` singleton + `fork_rng` / `join_rng` snapshot
helpers, (3) GPU-side PTX kernels `philox_uniform_kernel` and
`philox_normal_kernel` that generate random buffers directly on
device. Mirrors upstream's `at::CUDAGeneratorImpl` state machine
and the `PhiloxCudaState` capture/restore contract used by
captured CUDA graphs.

## Requirements

- REQ-1: `PhiloxState` — serialisable snapshot of `(counter, seed,
  offset)` with `pub fn new(counter, seed) -> Self`,
  `pub fn from_parts(counter, seed, offset) -> GpuResult<Self>`
  (validates `offset < 4`), and `pub fn offset() -> u64` accessor.
  Mirrors upstream's `PhiloxCudaState { seed_, offset_ }` capture.
- REQ-2: `PhiloxGenerator` — stateful Philox 4x32-10 generator
  tracking `(counter, seed, offset, cached: [u32; 4])`. 10 rounds of
  multiply-xor-key-advance mixing per counter step, producing 4
  uniform u32 outputs. Constants
  `PHILOX_M0=0xD2511F53`, `PHILOX_M1=0xCD9E8D57`,
  `PHILOX_W0=0x9E3779B9`, `PHILOX_W1=0xBB67AE85` match the
  Salmon et al. paper.
- REQ-3: `CudaRngManager` — per-device generator registry keyed by
  `usize` device ordinal, accessed via the global
  `pub fn cuda_rng_manager() -> &'static Mutex<CudaRngManager>`
  singleton. Lazy-construct a per-device `PhiloxGenerator` from the
  default seed on first access. Mirrors PyTorch's
  `at::cuda::detail::getDefaultCUDAGenerator(device_idx)`.
- REQ-4: `fork_rng(devices: &[usize]) -> Vec<PhiloxState>` and
  `join_rng(devices, states) -> GpuResult<()>` — snapshot and
  restore RNG states across multiple devices for DDP rank
  independence. Mirrors `torch.cuda.fork_rng` / the
  `torch.cuda.rng_state` save/restore protocol.
- REQ-5: `gpu_philox_uniform(n, device) -> CudaBuffer<f32>` — GPU
  kernel filling a length-`n` f32 buffer with uniform `[0, 1)` via
  the `PHILOX_UNIFORM_PTX` template. No CPU↔GPU round trip.
- REQ-6: `gpu_philox_normal(n, device) -> CudaBuffer<f32>` — GPU
  kernel filling a length-`n` f32 buffer with standard-normal f32
  via Box-Muller, using `PHILOX_NORMAL_PTX`. No CPU↔GPU round trip.
- REQ-7: Non-test production consumer wiring through
  `CudaBackendImpl` — the dropout-philox and other RNG-consuming
  paths in `backend_impl.rs` call into `crate::rng::cuda_rng_manager()`
  to advance state per launch.

## Acceptance Criteria

- [x] AC-1: `pub struct PhiloxState` at line 75 with `Debug`,
  `Clone`, `Copy`, `PartialEq`, `Eq`, `#[non_exhaustive]`.
- [x] AC-2: `pub struct PhiloxGenerator` at line 140 with the
  documented state.
- [x] AC-3: `pub struct CudaRngManager` at line 380 with per-device
  `HashMap<usize, PhiloxGenerator>` storage.
- [x] AC-4: `pub fn cuda_rng_manager`, `pub fn fork_rng`,
  `pub fn join_rng` at lines 473, 499, 520.
- [x] AC-5: `pub(crate) const PHILOX_UNIFORM_PTX` at line 557 and
  `pub(crate) const PHILOX_NORMAL_PTX` at line 872.
- [x] AC-6: `pub fn gpu_philox_uniform` at line 1235 and
  `pub fn gpu_philox_normal` at line 1307.
- [x] AC-7: 27 unit tests in `mod tests` exercise the algorithmic
  core, fork/join, and the GPU kernel paths.

## Architecture

### Algorithmic core (REQ-1, REQ-2)

`pub struct PhiloxState in rng.rs` is the serialisable triple `(counter,
seed, offset)`. The `offset` field is `pub(crate)` to forbid
external code from constructing an invalid `offset >= 4`; the
typed constructor `from_parts` enforces the range.

`pub struct PhiloxGenerator in rng.rs` carries
`(counter, seed, offset, cached: [u32; 4])`. The `cached` field
holds the last 4-tuple of Philox round outputs, with `offset`
indicating how many have been consumed. The CBRNG advance is:

```text
1. Split 64-bit counter into (counter_lo, counter_hi).
2. Split 64-bit seed/key into (key_lo, key_hi).
3. Run 10 rounds of: hi *= M0/M1; xor with the rotated tail;
   advance key_lo += W0; key_hi += W1.
4. Output the 4 mixed u32 values.
```

### Per-device registry (REQ-3, REQ-4)

`CudaRngManager` holds
`generators: HashMap<usize, PhiloxGenerator>` with helpers to:

- `manual_seed(device, seed)` — set the seed for a device's generator.
- `generator(device)` — lazy-construct from the default seed and
  return `&mut PhiloxGenerator`.
- `state(device) -> PhiloxState` — snapshot.
- `restore(device, state) -> ()` — restore from snapshot.

The global singleton at `pub fn cuda_rng_manager`
(`rng in rng.rs`) is a `&'static Mutex<CudaRngManager>` initialised
via `LazyLock`. The mutex critical section is short (a HashMap
lookup + counter advance), so contention is negligible in practice.

`fork_rng(devices)` and `join_rng(devices, states)` are the
DDP-rank-independence helpers: each rank captures its starting
state, runs independent streams, and re-converges before any
collective.

### GPU kernels (REQ-5, REQ-6)

`pub(crate) const PHILOX_UNIFORM_PTX` (line 557) and
`pub(crate) const PHILOX_NORMAL_PTX` (line 872) are hand-written
PTX templates loaded via `crate::module_cache::get_or_compile`.
Each kernel takes a launch-time `(counter, seed, offset)` from the
host (the current `PhiloxState`), runs the Philox advance
in-register per thread, and writes the resulting f32 values
directly to the output buffer. The host bumps the generator's
counter by `ceil(n / 4)` after the launch returns, keeping the
host- and device-side cursors in sync.

`pub fn gpu_philox_uniform(n, device)` (line 1235):
1. Reads the current `PhiloxState` for `device.ordinal()` from the
   global manager.
2. Allocates the f32 output buffer.
3. Launches `PHILOX_UNIFORM_PTX` with `(counter, seed, offset)`.
4. Advances the manager's counter by `ceil(n / 4)`.
5. Returns the buffer.

`pub fn gpu_philox_normal(n, device)` (line 1307) is structurally
identical with the Box-Muller transform applied in the PTX.

### Non-CUDA stubs

When `feature = "cuda"` is disabled, `gpu_philox_uniform` and
`gpu_philox_normal` are stub functions (lines 1380, 1386) that
return `GpuResult` carrying a "no CUDA backend" error. The
algorithmic core (`PhiloxGenerator`, `PhiloxState`,
`CudaRngManager`) is unconditional and remains usable for
serialisation, deterministic seeding, and CPU-side state tracking
in non-CUDA builds.

### Non-test production consumers

`backend_impl.rs, 2938, 3914, 3929` all do
`crate::rng::cuda_rng_manager().lock()` — these are the
dropout-philox and stochastic-rounding entry points in
`CudaBackendImpl` that advance the Philox cursor per launch.
ferrotorch-core's `Tensor::dropout` (when `train=true`) routes
through `GpuBackend::dropout_philox_f32` (`dropout_philox_f32 in backend_impl.rs`)
which uses this registry.

## Parity contract

`parity_ops = []` for this route. RNG parity is enforced indirectly:
the Philox constants and 10-round structure match PyTorch's
`PhiloxCudaState`/`CUDAGeneratorImpl` byte-for-byte, so any consumer
that pins the same `(seed, counter)` will produce identical
random streams to PyTorch (modulo CUDA stream interleaving, which
is also matched by the per-device generator design).

Edge cases preserved:

- **Offset cursor `[0, 4)`**: the typed `PhiloxState::from_parts`
  rejects `offset >= 4`. Matches upstream's invariant.
- **Per-device independence**: the registry's `HashMap<usize, ...>`
  keying gives each CUDA device its own generator, matching
  PyTorch's `getDefaultCUDAGenerator(device_idx)`.
- **Fork / join symmetry**: `join_rng(devices, fork_rng(devices))`
  is a no-op (verified by unit tests).
- **Counter advance after launch**: `gpu_philox_uniform` /
  `gpu_philox_normal` advance the counter by `ceil(n / 4)`, since
  each Philox round produces 4 u32 values. Two back-to-back calls
  on the same generator produce statistically-independent streams.

## Verification

Unit tests in `ferrotorch-gpu/src/rng.rs` `mod tests` (27 tests
covering): algorithmic state (`PhiloxState::new`, `from_parts`
validation), generator advance (`next_u32`, `next_4`),
`CudaRngManager` lazy init + state save/restore, `fork_rng` /
`join_rng` round-trip, the GPU uniform / normal kernels'
distribution properties (mean / variance / range), and the
multi-device split case.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda rng:: 2>&1 | tail -3
```

Expected: ≥ 1 `test result: ok` line. The GPU-kernel tests use the
`GpuDevice::new(0)` graceful-skip pattern.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct PhiloxState in ferrotorch-gpu/src/rng.rs` (line 75) with the documented constructors; non-test consumer: `CudaRngManager::state` and `restore` (lines 425, 431) use it for save/restore. Re-exported at `lib.rs`. |
| REQ-2 | SHIPPED | impl: `pub struct PhiloxGenerator in rng.rs` (line 140) with Philox 4x32-10 constants (lines 47-52); non-test consumer: `CudaRngManager.generators` map stores instances (line 382); production callers via `backend_impl.rs,2938,3914,3929`. |
| REQ-3 | SHIPPED | impl: `pub struct CudaRngManager in rng.rs` (line 380) and `pub fn cuda_rng_manager` singleton (line 473); non-test consumer: four `crate::rng::cuda_rng_manager().lock()` sites in `backend_impl.rs` at lines 2875, 2938, 3914, 3929. |
| REQ-4 | SHIPPED | impl: `pub fn fork_rng` at `rng in rng.rs`, `pub fn join_rng` at `rng in rng.rs`; non-test consumer: re-exported at `rng in lib.rs`. ferrotorch-core's `quantize in quantize.rs,1132` defines a parallel `cuda_rng::fork_rng`/`join_rng` Python-API surface that wraps these. |
| REQ-5 | SHIPPED | impl: `pub(crate) const PHILOX_UNIFORM_PTX` at `rng in rng.rs`, `pub fn gpu_philox_uniform` at `rng in rng.rs`; non-test consumer: the dropout-philox code path at `backend_impl.rs` derives a seed from the manager and launches the dropout kernel (the Philox-uniform value path used internally by dropout). |
| REQ-6 | SHIPPED | impl: `pub(crate) const PHILOX_NORMAL_PTX` at `rng in rng.rs`, `pub fn gpu_philox_normal` at `rng in rng.rs`; non-test consumer: re-exported through `lib.rs` and consumed by the ferrotorch-distributions Normal sampling path on GPU. |
| REQ-7 | SHIPPED | impl: `use crate::rng` import sites in `backend_impl.rs` (lines 2875, 2938, 3914, 3929) inside the dropout-philox / stochastic-rounding CudaBackendImpl methods; ferrotorch-core dispatches `Tensor::dropout` through `GpuBackend::dropout_philox_f32` which consumes the RNG state. |
