# nccl_sys — raw FFI bindings to NCCL via dlopen/dlsym

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/csrc/distributed/c10d/NCCLUtils.hpp
  - torch/csrc/distributed/c10d/NCCLUtils.cpp
  - torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp
-->

## Summary

`ferrotorch-distributed/src/nccl_sys.rs` is the raw FFI layer for
NVIDIA's NCCL (Collective Communications Library). NCCL is loaded
at runtime via `dlopen` so that the crate compiles and works on
systems without `libnccl2` installed — `is_available()` simply
returns `false` and any call that needs NCCL returns
`NcclError::LibraryNotFound`. The module exposes a typed function
table (`NcclFunctions`) loaded once on first use via `OnceLock`,
plus thin safe wrappers (`get_unique_id`, `comm_init_rank`,
`group_start`, `group_end`, `is_available`) and `unsafe`
wrappers for the bulk collective FFI symbols. Mirrors the role of
PyTorch's `NCCLUtils.{hpp,cpp}` which similarly exposes NCCL as
a typed C++ surface over the raw `nccl.h` symbols.

## Requirements

- REQ-1: `pub type NcclComm = *mut c_void` is the opaque
  communicator handle. `#[repr(C)] pub struct NcclUniqueId {
  pub internal: [u8; 128] }` is the 128-byte bootstrap ID
  matching `nccl.h`'s `ncclUniqueId`. `#[non_exhaustive]` is a
  Rust surface-API annotation only (does NOT affect memory
  layout); it prevents external struct-literal construction so
  the field is forward-compatible.
- REQ-2: `#[repr(C)] pub enum NcclDataType` (Int8=0, Uint8=1,
  Int32=2, Uint32=3, Int64=4, Uint64=5, Float16=6, Float32=7,
  Float64=8, Bfloat16=9) and `#[repr(C)] pub enum NcclRedOp`
  (Sum=0, Prod=1, Max=2, Min=3, Avg=4) match `nccl.h`'s
  `ncclDataType_t` and `ncclRedOp_t` discriminants exactly.
- REQ-3: `#[repr(C)] pub enum NcclResult` (Success=0,
  UnhandledCudaError=1, …, NumResults=8) matches
  `ncclResult_t`. `impl NcclResult::ok(self) -> Result<(),
  NcclError>` converts to the workspace error shape.
- REQ-4: `pub enum NcclError` (LibraryNotFound,
  SymbolNotFound(String), NcclStatus(NcclResult)) with
  `#[derive(thiserror::Error)]` for `Display` impls.
- REQ-5: Function-pointer table `NcclFunctions` carrying 11
  typed `unsafe extern "C" fn` fields: `ncclGetUniqueId`,
  `ncclCommInitRank`, `ncclCommDestroy`, `ncclAllReduce`,
  `ncclBroadcast`, `ncclAllGather`, `ncclReduceScatter`,
  `ncclSend`, `ncclRecv`, `ncclGroupStart`, `ncclGroupEnd`.
  Each signature mirrors the corresponding `nccl.h` declaration
  byte-for-byte at the Rust ABI level.
- REQ-6: `static NCCL_LIB: OnceLock<Result<NcclFunctions,
  NcclError>>` lazily loads the library on first call.
  `fn load_nccl` tries `libnccl.so.2`, `libnccl.so`,
  `/usr/lib/x86_64-linux-gnu/libnccl.so.2`,
  `/usr/local/cuda/lib64/libnccl.so.2` in order and returns
  `LibraryNotFound` if every name fails to `dlopen`.
- REQ-7: `pub fn get_unique_id() -> Result<NcclUniqueId,
  NcclError>` invokes `ncclGetUniqueId` via the table.
- REQ-8: `pub fn comm_init_rank(world_size: i32, rank: i32,
  unique_id: NcclUniqueId) -> Result<NcclComm, NcclError>`
  invokes `ncclCommInitRank`. The function is SAFE despite
  invoking unsafe FFI: it constructs the comm via the table
  and returns the handle; callers must `cudaSetDevice` first
  (documented in the rustdoc, NOT enforced).
- REQ-9: `pub unsafe fn comm_destroy(comm: NcclComm) ->
  Result<(), NcclError>` is unsafe because it requires a
  once-only call on a previously-initialised comm.
- REQ-10: Five `pub unsafe fn` collective FFI wrappers:
  `all_reduce`, `broadcast`, `all_gather`, `reduce_scatter`,
  `send`, `recv`. Each takes `*const c_void` / `*mut c_void`
  buffer pointers, counts, dtype/op, comm, stream pointers.
- REQ-11: `pub fn group_start() / group_end() -> Result<(),
  NcclError>` are safe wrappers around the matching FFI
  symbols. Used by callers wanting to batch multiple
  collective launches into a single CUDA stream submission.
- REQ-12: `pub fn is_available() -> bool` returns `nccl().is_ok()`
  — i.e., does the library load succeed?

## Acceptance Criteria

- [x] AC-1: Module compiles only under `#[cfg(feature = "nccl")]`
  (gated in `lib.rs` line 209).
- [x] AC-2: `NcclUniqueId` has `internal: [u8; 128]` and is
  `#[repr(C)]`.
- [x] AC-3: `NcclDataType` and `NcclRedOp` discriminants match
  `nccl.h` (verified by inspection — the file's enum block).
- [x] AC-4: `is_available()` doesn't panic; verified by
  `test_nccl_availability_doesnt_panic`.
- [x] AC-5: `NCCL_LIB` is a `OnceLock` — the load runs exactly
  once per process (`OnceLock::get_or_init` semantics).
- [x] AC-6: `comm_destroy` is `pub unsafe fn`; its
  `# Safety` rustdoc documents the once-only contract.

## Architecture

The file starts with the `nccl.h`-mirroring type declarations:
`NcclComm` (= `*mut c_void`), `NcclUniqueId` (128-byte
`#[repr(C)]` struct), `NcclDataType` / `NcclRedOp` / `NcclResult`
enums (all `#[repr(C)]`), and the `NcclError` taxonomy. The
`#[non_exhaustive]` on `NcclUniqueId` is a Rust-surface-only
annotation — it does NOT affect memory layout — and prevents
external struct-literal construction so the field is
forward-compatible (e.g., adding a version field later).

`NcclFunctions` is the typed function-pointer table. Each of its
11 fields is an `unsafe extern "C" fn(...) -> NcclResult` whose
signature byte-for-byte matches `nccl.h`'s declaration:
`ncclAllReduce` takes 7 args (sendbuf, recvbuf, count, dtype, op,
comm, stream), returns `NcclResult`. The `unsafe impl Send + Sync
for NcclFunctions` is sound because the table is loaded once and
never mutated.

`static NCCL_LIB: OnceLock<Result<NcclFunctions, NcclError>>` is
the global library handle. `fn nccl()` returns `&'static
NcclFunctions` (or an error if loading failed). The `get_or_init`
closure is `load_nccl`.

`fn load_nccl` tries four library names in order: `libnccl.so.2`
(versioned soname, preferred), `libnccl.so` (unversioned
fallback), `/usr/lib/x86_64-linux-gnu/libnccl.so.2` (Debian path),
`/usr/local/cuda/lib64/libnccl.so.2` (CUDA toolkit default). If
every dlopen fails, return `LibraryNotFound`. Otherwise, dlsym
each of the 11 symbols via the `load_sym!` macro and pack into
`NcclFunctions`. The macro uses `#[allow(clippy::missing_transmute_annotations)]`
inside its body because the transmute target type is
parametrically inferred from the corresponding struct field's
declared type — adding explicit annotations would duplicate the
11 signatures already in `NcclFunctions`. The struct definition is
the single source of truth; the transmute is type-checked through
field-init coercion.

The safe wrappers (`get_unique_id`, `comm_init_rank`,
`group_start`, `group_end`, `is_available`) take no `unsafe`
arguments and have no unsafe preconditions on the caller side.
The `unsafe fn` wrappers (`comm_destroy`, `all_reduce`,
`broadcast`, `all_gather`, `reduce_scatter`, `send`, `recv`)
carry `# Safety` rustdoc enumerating the device-pointer /
buffer-size / comm-validity contract. Inside each `unsafe fn`
body, a SAFETY: comment maps the documented obligations to the
corresponding `nccl.h` precondition (e.g., "buffer valid for
`count * size_of(datatype)` bytes" → `ncclAllReduce` buffer
contract).

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/nccl_backend.rs` — every
  `NcclBackend::*_raw` method invokes the corresponding
  `nccl_sys::*` wrapper.
- `ferrotorch-distributed/src/nccl_collective.rs` — imports
  `NcclDataType` from `nccl_sys` and uses it as the explicit-
  dtype parameter type.
- `ferrotorch-distributed/src/lib.rs` — `pub use
  nccl_sys::NcclUniqueId;` at line 247.
- `ferrotorch-distributed/src/hybrid_backend.rs` — imports
  `NcclUniqueId` from `nccl_sys`; `HybridBackend::new` takes
  the unique ID as a parameter.
- `ferrotorch/src/lib.rs` — meta-crate re-export reaches
  `NcclUniqueId`.

## Parity contract

No parity-sweep ops. The contract is the `nccl.h` FFI shape:

- `NcclDataType` discriminants ↔ `ncclDataType_t` enum
  values 0..9.
- `NcclRedOp` discriminants ↔ `ncclRedOp_t` enum values 0..4.
- `NcclResult` discriminants ↔ `ncclResult_t` enum values 0..8.
- `NcclUniqueId.internal: [u8; 128]` layout ↔ `ncclUniqueId`'s
  128-byte buffer.
- Function-pointer signatures (11 fields) ↔ `nccl.h`
  declarations.
- dlopen lookup order matches the libnccl2 package's typical
  install locations on Debian/Ubuntu/CUDA-toolkit systems.

## Verification

`cargo test -p ferrotorch-distributed --features nccl --lib`
runs one in-file test:

- `test_nccl_availability_doesnt_panic` — `is_available()`
  doesn't crash regardless of whether NCCL is actually
  installed.

The FFI signature correctness can't be verified without a real
`libnccl2.so` and a CUDA device; the `gpu_collective::tests`
hardware-gated tests (in the sibling `gpu_collective.rs` design
doc) cover the end-to-end dispatch.

No parity-sweep ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub type NcclComm` and `pub struct NcclUniqueId` in `ferrotorch-distributed/src/nccl_sys.rs`; non-test consumer: `ferrotorch-distributed/src/nccl_backend.rs` `pub struct NcclBackend.comm: Mutex<NcclComm>` and `pub fn NcclBackend::new(...) -> FerrotorchResult<Self>` takes `unique_id: NcclUniqueId`. |
| REQ-2 | SHIPPED | impl: `pub enum NcclDataType` and `pub enum NcclRedOp` in `ferrotorch-distributed/src/nccl_sys.rs`; non-test consumer: `ferrotorch-distributed/src/nccl_backend.rs` `pub unsafe fn allreduce_raw(...)` takes `datatype: NcclDataType` and `op: NcclRedOp`; `ferrotorch-distributed/src/nccl_collective.rs` `fn infer_dtype` returns `NcclDataType`. |
| REQ-3 | SHIPPED | impl: `pub enum NcclResult` and `impl NcclResult::ok` in `ferrotorch-distributed/src/nccl_sys.rs`; non-test consumer: every `pub unsafe fn` in the same file (`all_reduce`, `broadcast`, etc.) returns `(lib.ncclXxx)(...).ok()` which routes through `NcclResult::ok`. |
| REQ-4 | SHIPPED | impl: `pub enum NcclError` in `ferrotorch-distributed/src/nccl_sys.rs`; non-test consumer: `ferrotorch-distributed/src/nccl_backend.rs` `pub fn NcclBackend::new` maps `NcclError` to `DistributedError::Io { message: format!("NCCL comm_init_rank failed: {e}") }`. |
| REQ-5 | SHIPPED | impl: `struct NcclFunctions` (11 fn-pointer fields) in `ferrotorch-distributed/src/nccl_sys.rs`; non-test consumer: `fn nccl()` (same file) returns `&'static NcclFunctions`; every public `unsafe fn` in the file invokes through the table. |
| REQ-6 | SHIPPED | impl: `static NCCL_LIB: OnceLock<...>` and `fn load_nccl` in `ferrotorch-distributed/src/nccl_sys.rs`; non-test consumer: `fn nccl()` (same file) is the only access path; every public wrapper goes through it. |
| REQ-7 | SHIPPED | impl: `pub fn get_unique_id` in `ferrotorch-distributed/src/nccl_sys.rs`; non-test consumer: re-export at `ferrotorch-distributed/src/lib.rs` line 247 (`pub use nccl_sys::NcclUniqueId`) — production callers (the API docs for `NcclBackend::new` and `HybridBackend::new`) instruct rank 0 to call `nccl_sys::get_unique_id()` and distribute the result via TCP. The `ucc_native_gpu_allreduce_via_nccl_single_rank` test (in `ucc_backend.rs`) demonstrates the dispatch shape but is `#[ignore]`'d. |
| REQ-8 | SHIPPED | impl: `pub fn comm_init_rank` in `ferrotorch-distributed/src/nccl_sys.rs`; non-test consumer: `ferrotorch-distributed/src/nccl_backend.rs` `pub fn NcclBackend::new` and `pub fn with_stream` both invoke `nccl_sys::comm_init_rank(world_size as i32, rank as i32, unique_id)?`. |
| REQ-9 | SHIPPED | impl: `pub unsafe fn comm_destroy` in `ferrotorch-distributed/src/nccl_sys.rs`; non-test consumer: `ferrotorch-distributed/src/nccl_backend.rs` `impl Drop for NcclBackend` calls `nccl_sys::comm_destroy(*comm)` under the mutex guard. |
| REQ-10 | SHIPPED | impl: `pub unsafe fn all_reduce / broadcast / all_gather / reduce_scatter / send / recv` in `ferrotorch-distributed/src/nccl_sys.rs`; non-test consumer: every `NcclBackend::*_raw` method in `ferrotorch-distributed/src/nccl_backend.rs` invokes the corresponding `nccl_sys::*` wrapper; `impl Backend for NcclBackend::barrier` invokes `nccl_sys::all_reduce` with count=0. |
| REQ-11 | SHIPPED | impl: `pub fn group_start` and `pub fn group_end` in `ferrotorch-distributed/src/nccl_sys.rs`; non-test consumer: the symbols are loaded into `NcclFunctions` (REQ-5) so the public wrappers can dispatch. Re-exports: `lib.rs` exposes `NcclUniqueId` only; `group_start` / `group_end` are part of `nccl_sys::*` reachable through the public `nccl_sys` module declaration. The grandfathered surface lets future point-to-point batched calls (e.g., paired ncclSend/ncclRecv) wrap these helpers without surface churn. |
| REQ-12 | SHIPPED | impl: `pub fn is_available` in `ferrotorch-distributed/src/nccl_sys.rs`; non-test consumer: `ferrotorch-distributed/src/nccl_backend.rs` `pub fn is_nccl_available` invokes `nccl_sys::is_available()`. |
