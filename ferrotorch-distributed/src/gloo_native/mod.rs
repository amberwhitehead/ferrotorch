//! Native-Rust Gloo backend (issue #1132).
//!
//! This module implements a CPU-only collective-communication backend over
//! pure-Rust TCP — no C/C++ FFI, no `libgloo` link. The wire layer ([`transport`])
//! frames messages with an 8-byte LE length prefix; [`connect`] establishes
//! the rendezvous + full-mesh topology using PyTorch's standard env vars
//! (`MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`); [`collectives`]
//! provides the textbook ring allreduce, tree broadcast, and ring barrier.
//!
//! Gated under the `gloo-backend` cargo feature. Without the feature
//! enabled, [`super::GlooBackend::new`] returns
//! [`DistributedError::BackendUnavailable`](crate::error::DistributedError::BackendUnavailable)
//! to keep the public API contract from #459 intact.
//!
//! ## REQ status (per `.design/ferrotorch-distributed/gloo_native/mod.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (GlooBackendInner struct) | SHIPPED | `pub struct GlooBackendInner` in `gloo_native/mod.rs`; consumers: `gloo_backend.rs` (`GlooBackend.inner`), `mpi_backend.rs` (`MpiBackend.inner`), `ucc_backend.rs` (`UccBackend.cpu_inner`). |
//! | REQ-2 (new constructor) | SHIPPED | `pub fn GlooBackendInner::new` in `gloo_native/mod.rs`; consumers: `GlooBackend::new` / `from_env` in `gloo_backend.rs`, `MpiBackend::new` in `mpi_backend.rs`, `UccBackend::new` in `ucc_backend.rs`. |
//! | REQ-3 (DEFAULT_GLOO_TIMEOUT) | SHIPPED | `pub const DEFAULT_GLOO_TIMEOUT` in `gloo_native/mod.rs`; consumer: `impl Backend for GlooBackendInner::recv` / `::barrier` (same file). |
//! | REQ-4 (ring_allreduce_sum_f32 entry points) | SHIPPED | `pub fn ring_allreduce_sum_f32` and `pub fn ring_allreduce_sum_f32_with_timeout` in `gloo_native/mod.rs`; consumers: `GlooBackend::ring_allreduce_sum_f32` in `gloo_backend.rs`, `MpiBackend::allreduce_sum_f32` in `mpi_backend.rs`, `UccBackend::allreduce_sum_f32` in `ucc_backend.rs`. |
//! | REQ-5 (tree_broadcast_f32 entry points) | SHIPPED | `pub fn tree_broadcast_f32` and `pub fn tree_broadcast_f32_with_timeout` in `gloo_native/mod.rs`; consumers: `gloo_backend.rs`, `mpi_backend.rs`, `ucc_backend.rs` broadcast methods. |
//! | REQ-6 (RingTransport trait impl) | SHIPPED | `impl RingTransport for GlooBackendInner` in `gloo_native/mod.rs`; consumer: the `_with_timeout` entry points (same file) pass `self` as `&dyn RingTransport` into `collectives::ring_allreduce_sum_f32_bytes` / `tree_broadcast_f32_bytes` / `ring_barrier`. |
//! | REQ-7 (Backend trait impl) | SHIPPED | `impl Backend for GlooBackendInner` in `gloo_native/mod.rs`; consumer: `impl Backend for GlooBackend` in `gloo_backend.rs` forwards every method to the inner; same for `mpi_backend.rs` and `ucc_backend.rs`. |
//! | REQ-8 (GlooRendezvousConfig re-export) | SHIPPED | `pub use self::connect::RendezvousConfig as GlooRendezvousConfig` in `gloo_native/mod.rs`; consumers: `mpi_backend.rs` and `ucc_backend.rs` import `crate::gloo_native::GlooRendezvousConfig`. |
//! | REQ-9 (cfg-gated submodule visibility) | SHIPPED | `pub(crate) mod gloo_native` under `#[cfg(feature = "gloo-backend")]` in `lib.rs`; consumers: `mpi_backend.rs` / `ucc_backend.rs` import via `crate::gloo_native::...` paths that resolve under the feature. |

use std::time::Duration;

use ferrotorch_core::FerrotorchResult;

use crate::backend::Backend;
use crate::error::DistributedError;

pub(super) mod collectives;
pub(super) mod connect;
pub(super) mod error;
pub(super) mod transport;

use self::collectives::{
    RingTransport, ring_allreduce_sum_f32_bytes, ring_barrier, tree_broadcast_f32_bytes,
};
use self::connect::{PeerConn, PeerStreams, RendezvousConfig, rendezvous};
use self::error::GlooResult;
use self::transport::{recv_msg_into, send_msg, with_read_timeout};

pub use self::connect::RendezvousConfig as GlooRendezvousConfig;

/// Default per-recv timeout for the ring collectives when callers don't
/// supply one. 60 seconds matches [`crate::collective::DEFAULT_COLLECTIVE_TIMEOUT`].
pub const DEFAULT_GLOO_TIMEOUT: Duration = Duration::from_secs(60);

/// Native-Rust Gloo backend. CPU-only collective communication over TCP
/// with a full-mesh connection topology and ring/tree collective
/// algorithms.
///
/// # Connection topology
///
/// Every rank holds one [`std::net::TcpStream`] to every other rank
/// (`world_size - 1` connections per rank). The full-mesh shape means
/// [`Backend::send`] / [`Backend::recv`] can address any peer directly
/// without relaying through rank 0 (unlike [`crate::TcpBackend`]'s star
/// topology). The ring collectives only ever talk to `(prev, next)`
/// neighbours, so most of the mesh sits idle during allreduce — the
/// extra edges exist for compatibility with the trait-level
/// [`crate::collective`] helpers which assume any-to-any send/recv.
///
/// # Algorithms
///
/// - [`Self::ring_allreduce_sum_f32`] — ring scatter-reduce + allgather.
/// - [`Self::tree_broadcast_f32`] — binary-tree broadcast.
/// - [`Backend::barrier`] — two-wave ring barrier (arrival + release).
#[derive(Debug)]
pub struct GlooBackendInner {
    rank: usize,
    world_size: usize,
    /// One `TcpStream` per peer (self-slot is `None`). Wrapped in a `Mutex`
    /// so concurrent send/recv on disjoint peer pairs can proceed without
    /// blocking each other.
    connections: PeerStreams,
}

impl GlooBackendInner {
    /// Construct a Gloo backend by driving the rendezvous handshake.
    pub fn new(cfg: &RendezvousConfig) -> GlooResult<Self> {
        let connections = rendezvous(cfg)?;
        Ok(Self {
            rank: cfg.rank,
            world_size: cfg.world_size,
            connections,
        })
    }

    fn conn(&self, peer: usize) -> GlooResult<&PeerConn> {
        if peer == self.rank {
            return Err(DistributedError::SelfSend { rank: self.rank });
        }
        if peer >= self.world_size {
            return Err(DistributedError::InvalidRank {
                rank: peer,
                world_size: self.world_size,
            });
        }
        self.connections[peer]
            .as_ref()
            .ok_or(DistributedError::NoConnection { rank: peer })
    }

    fn send_inner(&self, data: &[u8], dst: usize) -> GlooResult<()> {
        let conn = self.conn(dst)?;
        let mut stream = conn
            .writer
            .lock()
            .map_err(|e| DistributedError::LockPoisoned {
                message: format!("gloo_native send rank {} -> {dst}: {e}", self.rank),
            })?;
        send_msg(&mut stream, data)
    }

    fn recv_inner(&self, dst: &mut [u8], src: usize, timeout: Duration) -> GlooResult<()> {
        let conn = self.conn(src)?;
        let mut stream = conn
            .reader
            .lock()
            .map_err(|e| DistributedError::LockPoisoned {
                message: format!("gloo_native recv rank {src} -> {}: {e}", self.rank),
            })?;
        with_read_timeout(&mut stream, timeout, |s| recv_msg_into(s, dst))
    }

    /// Ring-allreduce a contiguous `f32` slice **in place** with element-wise
    /// sum across all ranks. Every rank's slice must have the same length.
    pub fn ring_allreduce_sum_f32(&self, data: &mut [f32]) -> FerrotorchResult<()> {
        self.ring_allreduce_sum_f32_with_timeout(data, DEFAULT_GLOO_TIMEOUT)
    }

    /// Like [`Self::ring_allreduce_sum_f32`] with a custom per-step recv
    /// timeout.
    pub fn ring_allreduce_sum_f32_with_timeout(
        &self,
        data: &mut [f32],
        timeout: Duration,
    ) -> FerrotorchResult<()> {
        // SAFETY: `f32` is a plain-old-data type with no padding; reinterpreting
        // its slice as `&mut [u8]` is sound and the byte count is exactly
        // `data.len() * 4`. The view is held only for the duration of this
        // call, so no aliasing issues arise. We use the explicit
        // `bytemuck`-style cast inline rather than depending on `bytemuck`
        // because this crate doesn't already pull in bytemuck.
        let byte_len = std::mem::size_of_val(data);
        // Cast through a raw pointer is the canonical safe-by-construction
        // shape; `[f32]::as_mut_ptr()` returns a unique pointer.
        let bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u8>(), byte_len) };
        ring_allreduce_sum_f32_bytes(self, bytes, timeout).map_err(Into::into)
    }

    /// Tree-broadcast a contiguous `f32` slice from `root` to every other
    /// rank, **in place**. Non-root ranks' input contents are overwritten.
    pub fn tree_broadcast_f32(&self, data: &mut [f32], root: usize) -> FerrotorchResult<()> {
        self.tree_broadcast_f32_with_timeout(data, root, DEFAULT_GLOO_TIMEOUT)
    }

    /// Like [`Self::tree_broadcast_f32`] with a custom recv timeout.
    pub fn tree_broadcast_f32_with_timeout(
        &self,
        data: &mut [f32],
        root: usize,
        timeout: Duration,
    ) -> FerrotorchResult<()> {
        let byte_len = std::mem::size_of_val(data);
        // SAFETY: identical reasoning to `ring_allreduce_sum_f32_with_timeout`.
        let bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u8>(), byte_len) };
        tree_broadcast_f32_bytes(self, bytes, root, timeout).map_err(Into::into)
    }
}

impl RingTransport for GlooBackendInner {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        self.world_size
    }
    fn send(&self, data: &[u8], dst: usize) -> GlooResult<()> {
        self.send_inner(data, dst)
    }
    fn recv(&self, dst: &mut [u8], src: usize, timeout: Duration) -> GlooResult<()> {
        self.recv_inner(dst, src, timeout)
    }
}

impl Backend for GlooBackendInner {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn send(&self, data: &[u8], dst_rank: usize) -> FerrotorchResult<()> {
        self.send_inner(data, dst_rank).map_err(Into::into)
    }

    fn recv(&self, dst: &mut [u8], src_rank: usize) -> FerrotorchResult<()> {
        // Without an explicit timeout we still set the default — production
        // collectives that hang forever would mask deadlock bugs.
        self.recv_inner(dst, src_rank, DEFAULT_GLOO_TIMEOUT)
            .map_err(Into::into)
    }

    fn recv_timeout(
        &self,
        dst: &mut [u8],
        src_rank: usize,
        timeout: Duration,
    ) -> FerrotorchResult<()> {
        self.recv_inner(dst, src_rank, timeout).map_err(Into::into)
    }

    fn barrier(&self) -> FerrotorchResult<()> {
        ring_barrier(self, DEFAULT_GLOO_TIMEOUT).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener};
    use std::sync::Arc;
    use std::thread;

    /// Spin up `world_size` in-process ranks against a kernel-assigned
    /// `MASTER_PORT`. Returns `Arc<GlooBackendInner>` per rank for use in
    /// the calling test.
    fn spawn_group(world_size: usize) -> Vec<Arc<GlooBackendInner>> {
        let probe = TcpListener::bind("127.0.0.1:0").expect("probe bind");
        let master_addr = probe.local_addr().expect("local_addr").to_string();
        drop(probe);

        let handles: Vec<_> = (0..world_size)
            .map(|rank| {
                let ma = master_addr.clone();
                thread::spawn(move || {
                    let cfg = RendezvousConfig {
                        master_addr: ma,
                        rank,
                        world_size,
                        bind_addr: SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)),
                    };
                    Arc::new(GlooBackendInner::new(&cfg).expect("backend"))
                })
            })
            .collect();

        handles
            .into_iter()
            .map(|h| h.join().expect("join"))
            .collect()
    }

    #[test]
    fn full_mesh_send_recv_two_ranks() {
        let group = spawn_group(2);
        let b0 = Arc::clone(&group[0]);
        let b1 = Arc::clone(&group[1]);
        let sender = thread::spawn(move || {
            Backend::send(&*b0, &[7, 8, 9, 10], 1).expect("send 0->1");
        });
        let mut buf = [0u8; 4];
        Backend::recv(&*b1, &mut buf, 0).expect("recv 1<-0");
        sender.join().expect("join");
        assert_eq!(buf, [7, 8, 9, 10]);
    }

    #[test]
    fn ring_allreduce_over_real_tcp_two_ranks() {
        let group = spawn_group(2);
        let mut a = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut b = vec![10.0f32, 20.0, 30.0, 40.0];

        thread::scope(|s| {
            let g0 = Arc::clone(&group[0]);
            let g1 = Arc::clone(&group[1]);
            let h0 = s.spawn(move || {
                g0.ring_allreduce_sum_f32(&mut a).expect("ar 0");
                a
            });
            let h1 = s.spawn(move || {
                g1.ring_allreduce_sum_f32(&mut b).expect("ar 1");
                b
            });
            let r0 = h0.join().unwrap();
            let r1 = h1.join().unwrap();
            let expected = vec![11.0f32, 22.0, 33.0, 44.0];
            assert_eq!(r0, expected, "rank 0");
            assert_eq!(r1, expected, "rank 1");
        });
    }

    #[test]
    fn ring_allreduce_over_real_tcp_four_ranks() {
        let group = spawn_group(4);
        // 13 elements, 4 ranks -> uneven chunks (3, 3, 3, 4). Verifies the
        // tail-chunk path under real TCP framing.
        let inputs: Vec<Vec<f32>> = (0..4u32)
            .map(|r| (0..13u32).map(|i| (r * 100 + i) as f32).collect())
            .collect();
        let expected: Vec<f32> = (0..13u32)
            .map(|i| (0..4u32).map(|r| (r * 100 + i) as f32).sum())
            .collect();

        thread::scope(|s| {
            let mut handles = Vec::new();
            for (rank, input) in inputs.into_iter().enumerate() {
                let g = Arc::clone(&group[rank]);
                handles.push(s.spawn(move || {
                    let mut data = input;
                    g.ring_allreduce_sum_f32(&mut data).expect("allreduce");
                    data
                }));
            }
            for h in handles {
                let got = h.join().unwrap();
                assert_eq!(got, expected);
            }
        });
    }

    #[test]
    fn tree_broadcast_over_real_tcp_four_ranks() {
        let group = spawn_group(4);
        // Bit-exact f32 magic numbers (no approx-constant lint trip; these
        // are arbitrary identifiers, not approximations of `PI` / `E`).
        let payload = vec![100.5f32, 200.25, 300.125];

        thread::scope(|s| {
            let mut handles = Vec::new();
            for (rank, g_ref) in group.iter().enumerate() {
                let g = Arc::clone(g_ref);
                let p = payload.clone();
                handles.push(s.spawn(move || {
                    let mut data = if rank == 2 { p } else { vec![0.0f32; 3] };
                    g.tree_broadcast_f32(&mut data, 2).expect("broadcast");
                    data
                }));
            }
            for h in handles {
                let got = h.join().unwrap();
                assert_eq!(got, vec![100.5f32, 200.25, 300.125]);
            }
        });
    }

    #[test]
    fn barrier_over_real_tcp_three_ranks() {
        let group = spawn_group(3);
        thread::scope(|s| {
            let mut handles = Vec::new();
            for g_ref in group.iter().take(3) {
                let g = Arc::clone(g_ref);
                handles.push(s.spawn(move || {
                    Backend::barrier(&*g).expect("barrier");
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });
    }

    #[test]
    fn rendezvous_config_from_env_reads_pytorch_vars() {
        // Use SAFETY around env mutation (test runs single-threaded with
        // respect to this set; cargo's --test-threads default still
        // schedules tests in the same process but env mutation is process
        // wide — for this reason we use a unique prefix and clean up).
        // The test only validates parsing, not real rendezvous.
        // SAFETY: tests in the same process can race on env; we use
        // process-unique values and unset before exit.
        unsafe {
            std::env::set_var("MASTER_ADDR", "127.0.0.1");
            std::env::set_var("MASTER_PORT", "29501");
            std::env::set_var("RANK", "2");
            std::env::set_var("WORLD_SIZE", "4");
        }
        let cfg = RendezvousConfig::from_env().expect("from_env");
        assert_eq!(cfg.master_addr, "127.0.0.1:29501");
        assert_eq!(cfg.rank, 2);
        assert_eq!(cfg.world_size, 4);
        // SAFETY: we set these vars above; nothing else should depend on
        // them within the test binary.
        unsafe {
            std::env::remove_var("MASTER_ADDR");
            std::env::remove_var("MASTER_PORT");
            std::env::remove_var("RANK");
            std::env::remove_var("WORLD_SIZE");
        }
    }
}
