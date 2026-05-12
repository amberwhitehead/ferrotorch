//! TCP framing layer for the native Rust Gloo backend.
//!
//! Wire format: every logical message is `[u64 length, little-endian][payload
//! bytes]`. The length prefix is 8 bytes so it can address payloads up to
//! `u64::MAX` (effectively unbounded for collective traffic — the underlying
//! TCP socket is the real bottleneck).
//!
//! This module is intentionally tiny: it does *not* know about ranks,
//! topology, or collective ops. Each public helper takes a single
//! [`TcpStream`] (or a mutable reference to one) and a byte buffer.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use crate::error::DistributedError;

use super::error::GlooResult;

/// Length-prefix size in bytes. `u64` little-endian.
pub(crate) const LEN_PREFIX_BYTES: usize = std::mem::size_of::<u64>();

/// Send a single length-prefixed frame over `stream`.
///
/// Writes 8 bytes of LE length followed by `payload`, then flushes.
pub(crate) fn send_msg(stream: &mut TcpStream, payload: &[u8]) -> GlooResult<()> {
    let len = payload.len() as u64;
    stream
        .write_all(&len.to_le_bytes())
        .map_err(|e| DistributedError::Io {
            message: format!("gloo_native::transport send_msg len: {e}"),
        })?;
    stream
        .write_all(payload)
        .map_err(|e| DistributedError::Io {
            message: format!("gloo_native::transport send_msg payload: {e}"),
        })?;
    stream.flush().map_err(|e| DistributedError::Io {
        message: format!("gloo_native::transport send_msg flush: {e}"),
    })?;
    Ok(())
}

/// Receive a single length-prefixed frame from `stream` into a freshly
/// allocated [`Vec<u8>`]. The returned vec has exactly `length` bytes.
///
/// Errors with [`DistributedError::Io`] on socket failure.
///
/// Currently used only by the transport-layer unit tests; production
/// collectives reach for [`recv_msg_into`] so they can re-use a
/// caller-allocated chunk buffer.
#[cfg(test)]
pub(crate) fn recv_msg(stream: &mut TcpStream) -> GlooResult<Vec<u8>> {
    let mut len_buf = [0u8; LEN_PREFIX_BYTES];
    stream
        .read_exact(&mut len_buf)
        .map_err(|e| DistributedError::Io {
            message: format!("gloo_native::transport recv_msg len: {e}"),
        })?;
    let len = u64::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    if len > 0 {
        stream
            .read_exact(&mut buf)
            .map_err(|e| DistributedError::Io {
                message: format!("gloo_native::transport recv_msg payload ({len} bytes): {e}"),
            })?;
    }
    Ok(buf)
}

/// Receive a single length-prefixed frame **into** a caller-supplied buffer
/// `dst`. The frame's payload length must equal `dst.len()`; otherwise this
/// returns [`DistributedError::SizeMismatch`].
///
/// This is the hot-path entry point used by collective steps where the
/// caller has already pre-allocated a chunk-sized buffer.
pub(crate) fn recv_msg_into(stream: &mut TcpStream, dst: &mut [u8]) -> GlooResult<()> {
    let mut len_buf = [0u8; LEN_PREFIX_BYTES];
    stream
        .read_exact(&mut len_buf)
        .map_err(|e| DistributedError::Io {
            message: format!("gloo_native::transport recv_msg_into len: {e}"),
        })?;
    let len = u64::from_le_bytes(len_buf) as usize;
    if len != dst.len() {
        return Err(DistributedError::SizeMismatch {
            expected: dst.len(),
            got: len,
        });
    }
    if len > 0 {
        stream
            .read_exact(dst)
            .map_err(|e| DistributedError::Io {
                message: format!("gloo_native::transport recv_msg_into payload: {e}"),
            })?;
    }
    Ok(())
}

/// Apply a read timeout to `stream`, run `f`, then restore blocking mode.
///
/// On Linux the kernel surfaces a per-socket read timeout as `EAGAIN`
/// (`ErrorKind::WouldBlock`); on Windows the same condition surfaces as
/// `ErrorKind::TimedOut`. We translate both into the explicit
/// [`DistributedError::Timeout`] variant. Since `f` may have already
/// converted the underlying `std::io::Error` into a string-wrapped
/// `DistributedError::Io`, we additionally fingerprint the message — the
/// POSIX `EAGAIN` formatting (`"Resource temporarily unavailable (os error
/// 11)"`) and the macOS / Windows variants (`"operation timed out"`,
/// `"would block"`) are matched verbatim.
pub(crate) fn with_read_timeout<F, R>(
    stream: &mut TcpStream,
    timeout: Duration,
    f: F,
) -> GlooResult<R>
where
    F: FnOnce(&mut TcpStream) -> GlooResult<R>,
{
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| DistributedError::Io {
            message: format!("gloo_native::transport set_read_timeout: {e}"),
        })?;
    let result = f(stream);
    // Always restore blocking mode, even on error. We deliberately ignore
    // the restore-result: the only failure mode is the socket being closed
    // out from under us, in which case the next read will surface the real
    // error.
    let _ = stream.set_read_timeout(None);

    match result {
        Ok(v) => Ok(v),
        Err(DistributedError::Io { message }) if is_timeout_message(&message) => {
            Err(DistributedError::Timeout {
                seconds: timeout.as_secs(),
            })
        }
        Err(other) => Err(other),
    }
}

/// Heuristic: does `msg` carry an OS-level read-timeout signature?
///
/// Matches the three platform spellings we care about:
/// - Linux: `EAGAIN` → `"Resource temporarily unavailable"`
/// - macOS/Windows: `ErrorKind::TimedOut` → `"timed out"` /
///   `"operation timed out"`
/// - Generic `ErrorKind::WouldBlock` Display: `"would block"`
fn is_timeout_message(msg: &str) -> bool {
    msg.contains("Resource temporarily unavailable")
        || msg.contains("timed out")
        || msg.contains("would block")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};
    use std::thread;

    /// Helper: spin up a localhost listener, return a connected `(client,
    /// server)` stream pair plus the listener's port.
    fn local_pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("local_addr");
        let server_handle = thread::spawn(move || {
            let (s, _) = listener.accept().expect("accept");
            s
        });
        let client = TcpStream::connect(addr).expect("connect");
        let server = server_handle.join().expect("server thread");
        (client, server)
    }

    #[test]
    fn round_trip_small_payload() {
        let (mut client, mut server) = local_pair();
        let payload = b"hello, gloo-native";
        let writer = thread::spawn(move || {
            send_msg(&mut client, payload).expect("send");
        });
        let got = recv_msg(&mut server).expect("recv");
        writer.join().expect("writer thread");
        assert_eq!(got, payload);
    }

    #[test]
    fn round_trip_into_dst_buffer() {
        let (mut client, mut server) = local_pair();
        let payload = vec![7u8; 1024];
        let p2 = payload.clone();
        let writer = thread::spawn(move || {
            send_msg(&mut client, &p2).expect("send");
        });
        let mut dst = vec![0u8; 1024];
        recv_msg_into(&mut server, &mut dst).expect("recv_into");
        writer.join().expect("writer thread");
        assert_eq!(dst, payload);
    }

    #[test]
    fn size_mismatch_into_dst_buffer() {
        let (mut client, mut server) = local_pair();
        let payload = vec![1u8; 32];
        let writer = thread::spawn(move || {
            send_msg(&mut client, &payload).expect("send");
        });
        let mut dst = vec![0u8; 16];
        let err = recv_msg_into(&mut server, &mut dst).expect_err("must err");
        writer.join().expect("writer thread");
        match err {
            DistributedError::SizeMismatch { expected, got } => {
                assert_eq!(expected, 16);
                assert_eq!(got, 32);
            }
            other => panic!("expected SizeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn zero_length_frame_round_trips() {
        let (mut client, mut server) = local_pair();
        let writer = thread::spawn(move || {
            send_msg(&mut client, &[]).expect("send empty");
        });
        let got = recv_msg(&mut server).expect("recv empty");
        writer.join().expect("writer thread");
        assert!(got.is_empty());
    }

    #[test]
    fn read_timeout_surfaces_as_timeout_error() {
        let (_client, mut server) = local_pair();
        // No writer — server should hit the timeout deadline.
        let err = with_read_timeout(&mut server, Duration::from_millis(50), |s| {
            recv_msg(s)?;
            Ok(())
        })
        .expect_err("must time out");
        match err {
            DistributedError::Timeout { seconds } => assert_eq!(seconds, 0),
            other => panic!("expected Timeout, got {other:?}"),
        }
    }
}
