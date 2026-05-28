# gloo_native::transport — TCP framing primitives

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/csrc/distributed/c10d/ProcessGroupGloo.cpp
  - torch/csrc/distributed/c10d/ProcessGroupGloo.hpp
-->

## Summary

`ferrotorch-distributed/src/gloo_native/transport.rs` is the TCP
framing layer for the native-Rust Gloo backend. Every logical
message is `[u64 length, little-endian][payload]`. The module is
deliberately narrow: it knows nothing about ranks, topology, or
collective algorithms — its only job is to push a frame onto one
`TcpStream` and pull it back off another. Maps to the framing
that PyTorch's `ProcessGroupGloo` delegates to `libgloo` for; in
ferrotorch we own the protocol because we own the transport.

## Requirements

- REQ-1: `pub(crate) const LEN_PREFIX_BYTES: usize =
  std::mem::size_of::<u64>()` is the wire-format length-prefix
  width (8 bytes, little-endian).
- REQ-2: `pub(crate) fn send_msg(stream: &mut TcpStream, payload:
  &[u8]) -> GlooResult<()>` writes 8 LE bytes of length plus
  payload bytes plus a flush, mapping every I/O error to
  `DistributedError::Io` with a context-bearing message.
- REQ-3: `pub(crate) fn recv_msg_into(stream: &mut TcpStream,
  dst: &mut [u8]) -> GlooResult<()>` reads 8 LE bytes of length,
  rejects any mismatch with `dst.len()` via
  `DistributedError::SizeMismatch`, then reads exactly `dst.len()`
  bytes. Hot-path entry point for the ring collectives.
- REQ-4: `pub(crate) fn with_read_timeout<F, R>(stream: &mut
  TcpStream, timeout: Duration, f: F) -> GlooResult<R>` sets a
  per-socket read timeout around `f`, runs it, restores blocking
  mode regardless of outcome, and maps `WouldBlock` / `TimedOut`
  / Linux `EAGAIN` to `DistributedError::Timeout`. The timeout
  message-fingerprinting (`is_timeout_message`) is platform-
  portable across Linux (`Resource temporarily unavailable`),
  macOS/Windows (`timed out` / `operation timed out`), and the
  generic `WouldBlock` Display (`would block`).
- REQ-5: A zero-length frame is well-formed (length = 0, no
  payload bytes) and round-trips correctly through `send_msg`
  / `recv_msg` / `recv_msg_into`.
- REQ-6: A `#[cfg(test)]`-only `recv_msg` helper allocates a
  fresh `Vec<u8>` for the payload. Production code uses
  `recv_msg_into` to reuse caller-allocated chunk buffers.

## Acceptance Criteria

- [x] AC-1: `round_trip_small_payload` — 18-byte payload
  round-trips through a localhost pair.
- [x] AC-2: `round_trip_into_dst_buffer` — 1024-byte payload
  into a pre-allocated `dst` buffer.
- [x] AC-3: `size_mismatch_into_dst_buffer` — wrong-sized
  `dst` produces `SizeMismatch { expected: 16, got: 32 }`.
- [x] AC-4: `zero_length_frame_round_trips` — empty payload
  passes through cleanly.
- [x] AC-5: `read_timeout_surfaces_as_timeout_error` — a
  50ms timeout with no writer produces
  `DistributedError::Timeout { seconds: 0 }`.

## Architecture

`send_msg` is three operations: `write_all` the 8-byte LE
length prefix, `write_all` the payload, `flush`. Each maps to
a `DistributedError::Io` with a context string identifying
which operation failed (length / payload / flush). The flush
is load-bearing: without it, a sender that immediately drops
the stream may not deliver the framed bytes.

`recv_msg_into` mirrors the send: `read_exact` 8 LE bytes,
parse as `u64`, length-check against `dst.len()`, `read_exact`
into `dst`. The length-check rejects any mismatch with
`SizeMismatch { expected: dst.len(), got: parsed_len }` rather
than truncating or extending. The `if len > 0` guard around
the payload read avoids calling `read_exact` on a zero-length
slice (which would be a no-op but adds clarity).

`with_read_timeout` is a closure pattern that maintains the
invariant "the stream's read-timeout setting after `f`
returns equals what it was before" — by always calling
`set_read_timeout(None)` after `f`, regardless of outcome.
The `let _ = stream.set_read_timeout(None)` deliberately
discards the restore-result: if the socket is dead, the
next read will surface the real error.

The platform-portability of `is_timeout_message` is the
trickiest bit. On Linux the kernel surfaces a per-socket
read timeout as `EAGAIN`, which `std::io::Error`'s `Display`
formats as `"Resource temporarily unavailable (os error 11)"`.
On macOS / Windows the same condition surfaces as
`ErrorKind::TimedOut` with `Display` strings `"timed out"`
or `"operation timed out"`. We fingerprint the message
string verbatim — fragile across libc updates but the
simplest portable shape; the alternative (matching on
`io::ErrorKind` AND raw `os error 11`) is brittler in
practice.

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/gloo_native/mod.rs` — `send_inner`
  and `recv_inner` invoke `send_msg` and `with_read_timeout` +
  `recv_msg_into`.
- The `LEN_PREFIX_BYTES` const is internal-only documentation.

## Parity contract

No parity-sweep ops. The wire-protocol contract is the
length-prefix framing: 8-byte LE u64 length followed by
payload bytes. PyTorch's `ProcessGroupGloo` delegates framing
to `libgloo`, which uses a similar (but not identical)
binary-length-prefix shape. The exact byte format does NOT
need to match upstream — Gloo's wire format is internal to
the C++ library and not part of any external contract. What
DOES matter is that ferrotorch's wire format is
self-consistent across the gloo_native module (it is — every
sender and receiver agrees on `LEN_PREFIX_BYTES = 8` and
`u64::to_le_bytes` / `u64::from_le_bytes`).

## Verification

`cargo test -p ferrotorch-distributed --features gloo-backend
--lib` runs five tests in `transport::tests`:

- `round_trip_small_payload` — happy path 18-byte round-trip.
- `round_trip_into_dst_buffer` — happy path 1024-byte
  pre-allocated round-trip.
- `size_mismatch_into_dst_buffer` — discriminates the
  `SizeMismatch` shape.
- `zero_length_frame_round_trips` — empty payload happy path.
- `read_timeout_surfaces_as_timeout_error` — pins the
  platform-portable timeout-discriminator.

No parity-sweep ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub(crate) const LEN_PREFIX_BYTES` in `ferrotorch-distributed/src/gloo_native/transport.rs`; non-test consumer: `send_msg` and `recv_msg_into` (same file) reference the const for buffer sizes; the const documents the wire-protocol shape. |
| REQ-2 | SHIPPED | impl: `pub(crate) fn send_msg` in `ferrotorch-distributed/src/gloo_native/transport.rs`; non-test consumer: `ferrotorch-distributed/src/gloo_native/mod.rs` `fn send_inner` invokes it (`send_msg(&mut stream, data)`). |
| REQ-3 | SHIPPED | impl: `pub(crate) fn recv_msg_into` in `ferrotorch-distributed/src/gloo_native/transport.rs`; non-test consumer: `ferrotorch-distributed/src/gloo_native/mod.rs` `fn recv_inner` invokes it inside the `with_read_timeout` closure. |
| REQ-4 | SHIPPED | impl: `pub(crate) fn with_read_timeout` in `ferrotorch-distributed/src/gloo_native/transport.rs`; non-test consumer: `ferrotorch-distributed/src/gloo_native/mod.rs` `fn recv_inner` wraps every recv in `with_read_timeout(&mut stream, timeout, |s| recv_msg_into(s, dst))`. |
| REQ-5 | SHIPPED | impl: the `if len > 0` guard in `send_msg` and `recv_msg_into` in `ferrotorch-distributed/src/gloo_native/transport.rs`; non-test consumer: implicit — the production code in `gloo_native/collectives.rs` `ring_barrier` sends single-byte tokens which are non-zero, but the zero-length contract is part of the framing primitive's invariant. Verified by `zero_length_frame_round_trips` test. |
| REQ-6 | SHIPPED | impl: `#[cfg(test)] pub(crate) fn recv_msg` in `ferrotorch-distributed/src/gloo_native/transport.rs`; non-test consumer: not applicable — REQ-6 is a NEGATIVE requirement (production code uses `recv_msg_into`, not `recv_msg`). `grep "recv_msg(" ferrotorch-distributed/src/gloo_native/` confirms no production caller of `recv_msg`. The `#[cfg(test)]` gating itself is the evidence. |
