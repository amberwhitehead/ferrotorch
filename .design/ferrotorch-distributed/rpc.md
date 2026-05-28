# Remote Procedure Call (RPC) framework

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/distributed/rpc/__init__.py
  - torch/distributed/rpc/api.py
-->

## Summary

`ferrotorch-distributed/src/rpc.rs` implements a minimal RPC framework
layered on top of the `Backend` trait, mirroring
`torch.distributed.rpc`. `RpcAgent` wraps a `Backend`, maintains a
function registry, and supports synchronous (`rpc_sync`) and thread-
spawned asynchronous (`rpc_async`) remote invocations. Messages are
length-prefixed binary frames with a request/response correlation
mechanism via `request_id`. A 1 GiB cap (`MAX_RPC_MSG_SIZE`) guards
against OOM from malicious or corrupted length prefixes. The TCP
transport (`TcpRpcBackend`) is a thin wrapper that translates star-
topology connection errors into typed RPC errors.

## Requirements

- REQ-1: `pub enum RpcError` with `Debug` + `thiserror::Error` +
  `#[non_exhaustive]`. Variants: `FunctionNotFound`,
  `InvalidMessage`, `NoConnection`, `Internal`, `Timeout`. A
  `From<RpcError> for FerrotorchError` blanket conversion stringifies
  the error into `InvalidArgument`. Mirrors PyTorch's
  `RpcError`-family exceptions raised by `rpc.api`.
- REQ-2: `const MAX_RPC_MSG_SIZE: usize = 1 << 30` (1 GiB) hard cap
  applied at receive time to prevent OOM from malicious length
  prefixes. The cap is enforced by both `TcpRpcBackend::recv` and
  `RpcAgent::recv_response`.
- REQ-3: Internal `struct RpcRequest { request_id: u64, function_name:
  String, payload: Vec<u8> }` and `struct RpcResponse { request_id:
  u64, payload: Vec<u8>, error: Option<String> }` with `serialize()`
  / `deserialize()` methods. Wire format uses LE byte ordering: 1-
  byte tag (0x01 request / 0x02 response) + u64 request_id + u32
  length prefixes + payload bytes. Deserialization rejects messages
  with mismatched tags or truncated framing with `RpcError::InvalidMessage`.
- REQ-4: `pub struct TcpRpcBackend` wraps an `Arc<dyn Backend>` (the
  star-topology TCP backend). Provides `pub fn send(data, dst_rank)`,
  `pub fn recv(dst, src_rank)`, `pub fn rank()`, `pub fn
  world_size()`. The `send` / `recv` translate the underlying
  backend's "no connection" / "NoConnection" string errors into
  typed `RpcError::NoConnection { rank }` via string matching (the
  underlying `Backend` errors are plain
  `FerrotorchError::InvalidArgument`).
- REQ-5: `pub struct RpcAgent` owns a `Backend`, a
  `Mutex<HashMap<String, Arc<RpcHandler>>>` registry, a
  `Mutex<u64>` request-id counter, and a `Mutex<HashMap<u64,
  RpcResponse>>` buffer for out-of-order responses. All `.lock()`
  calls use the `.unwrap_or_else(|e| e.into_inner())` poison-recovery
  pattern so a panicked handler does not deadlock the agent.
- REQ-6: `pub fn register<F>(&self, name, handler)` where `F:
  Fn(&[u8]) -> Result<Vec<u8>, String> + Send + Sync + 'static`. The
  handler is type-erased into `Arc<RpcHandler>` and stored in the
  registry. Mirrors `rpc.api.RpcAgent.register_callee` (and the
  upstream-side `RemoteModule` registration).
- REQ-7: `pub fn rpc_sync(&self, dst_rank, function_name, args) ->
  FerrotorchResult<Vec<u8>>` serializes a request, sends it via
  the backend, then loops in `recv_response` until a response
  matching the request_id arrives. Out-of-order responses are
  buffered in `self.buffered_responses` for later retrieval.
  Mirrors `torch.distributed.rpc.rpc_sync` at
  `torch/distributed/rpc/api.py:766`.
- REQ-8: `pub fn rpc_async(self: &Arc<Self>, dst_rank, function_name,
  args) -> std::thread::JoinHandle<FerrotorchResult<Vec<u8>>>`
  spawns a thread invoking `rpc_sync` on a clone of the agent. The
  doc-comment explicitly warns about unbounded thread spawning —
  this is acceptable for the typical infrequent-coordination RPC
  pattern but not for high-frequency fire-and-forget. Mirrors
  `torch.distributed.rpc.rpc_async` at
  `torch/distributed/rpc/api.py:840`.
- REQ-9: `pub fn handle_request(&self, src_rank, request_data) ->
  FerrotorchResult<()>` deserializes a request, looks up the
  function, calls it, and sends the response (length-prefix +
  serialized response) back to `src_rank`. Unregistered functions
  produce an error response containing the rank that rejected the
  call.
- REQ-10: Accessors `pub fn rank` and `pub fn world_size` on both
  `RpcAgent` and `TcpRpcBackend` forward to the inner `Backend`.

## Acceptance Criteria

- [x] AC-1: `RpcRequest` round-trip via `serialize` + `deserialize`
  preserves `request_id`, `function_name`, and `payload`.
- [x] AC-2: `RpcResponse` round-trip preserves `request_id`,
  `payload`, and `error` (both ok and error variants).
- [x] AC-3: `MAX_RPC_MSG_SIZE == 1 << 30` (1 GiB).
- [x] AC-4: `RpcAgent::register` + `RpcAgent::lookup` returns the
  registered handler; calling the handler returns the expected
  output.
- [x] AC-5: `RpcAgent::lookup` on a non-existent name returns
  `None`.

## Architecture

### Error type (REQ-1)

`pub enum RpcError` (in `rpc.rs`) carries the 5 variants. The
`#[non_exhaustive]` attribute allows future error categories without
a major version bump. The `From<RpcError> for FerrotorchError`
conversion routes through `FerrotorchError::InvalidArgument` with
the RPC error's `Display` text — this is a stringly-typed bridge
because the project's central error enum is intentionally narrow.

### Message envelope (REQ-3)

`RpcRequest::serialize` (in `rpc.rs`) writes:

- 1 byte tag `0x01`.
- 8 bytes LE u64 `request_id`.
- 4 bytes LE u32 `name_bytes.len()`, then `name_bytes`.
- 4 bytes LE u32 `payload.len()`, then `payload`.

`RpcRequest::deserialize` (in `rpc.rs`) walks the same layout and
errors on any truncation or wrong tag. The `try_into().unwrap()` on
exactly-8-byte and exactly-4-byte slices is correct because the
slice indexing immediately preceded by a length check guarantees
size 8 / 4.

`RpcResponse::serialize` / `deserialize` (in `rpc.rs`) use tag
`0x02` and an additional 1-byte `error` flag preceding the variable-
length payload-or-error-string. Both variants are length-prefixed
with a u32.

### TCP transport (REQ-4)

`pub struct TcpRpcBackend` (in `rpc.rs`) holds `Arc<dyn Backend>` so
it can be cheaply cloned by `RpcAgent` and other consumers. `send`
forwards to `Backend::send` and translates string-matched
"NoConnection" errors into typed `RpcError::NoConnection`. `recv`
applies the same translation AND enforces `MAX_RPC_MSG_SIZE` before
delegating.

### Agent (REQ-5, REQ-6, REQ-7, REQ-8, REQ-9)

`pub struct RpcAgent` (in `rpc.rs`) layout:

- `backend: Arc<dyn Backend>` — the underlying transport.
- `registry: Mutex<HashMap<String, Arc<RpcHandler>>>` — registered
  function table.
- `next_request_id: Mutex<u64>` — monotonically increasing per-agent
  request counter.
- `buffered_responses: Mutex<HashMap<u64, RpcResponse>>` — holds
  responses that arrived out of order so the awaiting `rpc_sync`
  can retrieve them.

`pub fn register<F>` boxes the closure into `Arc<RpcHandler>` (where
`RpcHandler = Box<dyn Fn(&[u8]) -> Result<Vec<u8>, String> + Send
+ Sync>`) and stores it under `name`. Re-registering overwrites the
prior handler.

`pub fn rpc_sync`:

1. Allocate a request_id via `next_id`.
2. Serialize the request.
3. `backend.send(&serialized, dst_rank)`.
4. Call `recv_response(dst_rank, request_id)`.

`fn recv_response` first checks the buffer; if the expected response
isn't already cached, it loops:

- Read an 8-byte length prefix from `dst_rank`.
- Validate against `MAX_RPC_MSG_SIZE`.
- Allocate a buffer of that size and recv.
- Deserialize the response.
- If the response's `request_id` matches, return it via
  `process_response` (which converts the embedded error string into a
  `FerrotorchError::InvalidArgument` or returns the payload).
- If the response's `request_id` doesn't match (a different
  in-flight call's response arrived first), buffer it under its
  id and continue.

`pub fn rpc_async` clones the `Arc<Self>` and spawns a new
`std::thread` running `rpc_sync`. The returned `JoinHandle` lets the
caller `.join()` to retrieve the result.

`pub fn handle_request` is the server-side counterpart: deserialize,
look up, call, serialize, send response (length-prefix + body). If
the function isn't registered, send an error response with the
rejecting rank in the message string.

### Accessors (REQ-10)

`pub fn rank` and `pub fn world_size` on both `TcpRpcBackend` and
`RpcAgent` forward to the underlying `Backend`'s methods.

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/lib.rs` `pub use rpc::{RpcAgent,
  RpcError, TcpRpcBackend};` → `ferrotorch/src/lib.rs` `pub use
  ferrotorch_distributed::*;` for user code.
- Within `rpc.rs`, `rpc_sync` is a production consumer of
  `Backend::send` + `Backend::recv`; `handle_request` is the
  server-side production consumer of the same primitives.
- Within `rpc.rs`, `rpc_async` is a production consumer of
  `rpc_sync` via the spawned thread.

## Parity contract

No parity-sweep ops in the route (`parity_ops = []`). The contract is
the PyTorch RPC shape:

- `torch.distributed.rpc.rpc_sync(to, func, args, kwargs, timeout)`
  at `torch/distributed/rpc/api.py:766` → ferrotorch's
  `RpcAgent::rpc_sync(dst_rank, function_name, args)`. R-DEV-7
  deviation: ferrotorch's args are pre-serialized `&[u8]` because
  Rust doesn't have a single dynamic-callable abstraction equivalent
  to Python's `Callable + pickle`. The user-side serialization
  conventions (serde, bincode, custom) are caller-chosen.
- `torch.distributed.rpc.rpc_async(...)` at
  `torch/distributed/rpc/api.py:840` → ferrotorch's
  `RpcAgent::rpc_async`. Returns a `std::thread::JoinHandle`
  instead of a `torch.futures.Future`. R-DEV-7 deviation — the
  Rust ecosystem doesn't have a single canonical future runtime
  ferrotorch can depend on, so a join-handle is the simplest
  forward-compatible primitive.
- Star-topology limitation (non-zero ranks can only talk to rank 0)
  is documented in the module doc-comment. Upstream RPC supports
  full-mesh connectivity; ferrotorch's `TcpBackend` is a star, so
  rank-to-rank RPC between two non-zero ranks errors with
  `RpcError::NoConnection`. This is upstream-feature-gap, not
  divergence.

## Verification

`cargo test -p ferrotorch-distributed --lib rpc::` runs the
`#[cfg(test)] mod tests` block at lines 593-670 covering 6 tests:

- `test_rpc_request_roundtrip` — `RpcRequest` serialize → deserialize
  round-trip.
- `test_rpc_response_roundtrip_ok` — `RpcResponse` (ok variant)
  serialize → deserialize round-trip.
- `test_rpc_response_roundtrip_error` — `RpcResponse` (error variant)
  serialize → deserialize round-trip.
- `test_max_message_size_constant` — `MAX_RPC_MSG_SIZE == 1 << 30`.
- `test_rpc_agent_register_lookup` — register + lookup + invocation
  of an `"echo"` handler.
- `test_rpc_agent_lookup_missing` — lookup of a non-existent name
  returns `None`.

Lint: `cargo clippy -p ferrotorch-distributed -- -D warnings` clean.
Parity-sweep: no ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum RpcError` and `impl From<RpcError> for FerrotorchError` in `ferrotorch-distributed/src/rpc.rs`; non-test consumer: `ferrotorch-distributed/src/lib.rs` `pub use rpc::{RpcAgent, RpcError, TcpRpcBackend};` re-export reaching `ferrotorch/src/lib.rs`. |
| REQ-2 | SHIPPED | impl: `const MAX_RPC_MSG_SIZE: usize = 1 << 30` in `ferrotorch-distributed/src/rpc.rs`; non-test consumer: enforced at `TcpRpcBackend::recv` and `RpcAgent::recv_response` (both same file, both production paths). |
| REQ-3 | SHIPPED | impl: `struct RpcRequest`, `struct RpcResponse`, and their `serialize`/`deserialize` methods in `ferrotorch-distributed/src/rpc.rs`; non-test consumer: invoked by `pub fn rpc_sync` and `pub fn handle_request` (same file, both reachable via `lib.rs` re-export of `RpcAgent`). |
| REQ-4 | SHIPPED | impl: `pub struct TcpRpcBackend` in `ferrotorch-distributed/src/rpc.rs`; non-test consumer: `ferrotorch-distributed/src/lib.rs` re-exports `TcpRpcBackend`. |
| REQ-5 | SHIPPED | impl: `pub struct RpcAgent` and `pub fn new` in `ferrotorch-distributed/src/rpc.rs`; non-test consumer: `lib.rs` re-export of `RpcAgent`. |
| REQ-6 | SHIPPED | impl: `pub fn register<F>` in `ferrotorch-distributed/src/rpc.rs`; non-test consumer: invoked by `pub fn handle_request` (lookup side) via `self.lookup`, plus `lib.rs` re-export. |
| REQ-7 | SHIPPED | impl: `pub fn rpc_sync` + `fn recv_response` + `fn process_response` in `ferrotorch-distributed/src/rpc.rs`; non-test consumer: invoked by `pub fn rpc_async` (same file) AND reachable through `lib.rs` re-export of `RpcAgent`. |
| REQ-8 | SHIPPED | impl: `pub fn rpc_async` in `ferrotorch-distributed/src/rpc.rs`; non-test consumer: reachable through `lib.rs` re-export of `RpcAgent`. |
| REQ-9 | SHIPPED | impl: `pub fn handle_request` in `ferrotorch-distributed/src/rpc.rs`; non-test consumer: invokes `self.lookup` and the backend's `send` directly; reachable through `lib.rs` re-export of `RpcAgent`. |
| REQ-10 | SHIPPED | impl: `pub fn rank` and `pub fn world_size` on `TcpRpcBackend` and on `RpcAgent` in `ferrotorch-distributed/src/rpc.rs`; non-test consumer: surfaced via `lib.rs` re-exports of both types. |
