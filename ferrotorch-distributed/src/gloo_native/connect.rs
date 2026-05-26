//! Rendezvous + full-mesh TCP connection setup for the native Gloo backend.
//!
//! # Protocol
//!
//! The protocol mirrors PyTorch's `init_process_group(backend="gloo")`
//! env-var handshake (`MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`):
//!
//! 1. Rank 0 binds a [`TcpListener`] on `MASTER_ADDR:MASTER_PORT` and accepts
//!    `world_size - 1` connections. Each non-zero rank sends its rank index
//!    as the first 8 bytes (LE u64) on connect. Rank 0 also collects, from
//!    each connecting rank, a 6-byte advertisement `(ipv4 = 4 bytes, port =
//!    2 bytes LE u16)` for the listener that rank intends to use for its
//!    peer-to-peer connections.
//!
//! 2. Rank 0 broadcasts the assembled `(rank → addr)` table back to every
//!    non-zero rank.
//!
//! 3. Each rank then pairs up with every higher-numbered rank: rank `r`
//!    listens on its advertised port and accepts from ranks `> r`; rank `r`
//!    also connects to every rank `< r`. The resulting full mesh stores
//!    one [`TcpStream`] per peer.
//!
//! The handshake is intentionally simple (and uses the same length-prefix
//! framing as the rest of the backend, except for the fixed-size rendezvous
//! frames). It is **not** a replacement for production rendezvous services
//! (etcd / `c10d::TCPStore`) — the goal is parity with PyTorch's env-var
//! convention so users can drop ferrotorch into the same launch scripts.
//!
//! ## REQ status (per `.design/ferrotorch-distributed/gloo_native/connect.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (RendezvousConfig struct) | SHIPPED | `pub struct RendezvousConfig` in `gloo_native/connect.rs`; consumer: `gloo_native/mod.rs` re-exports as `GlooRendezvousConfig`; constructed in `gloo_backend.rs` `GlooBackend::new`, `mpi_backend.rs` `MpiBackend::new`, `ucc_backend.rs` `UccBackend::new`. |
//! | REQ-2 (from_env reads PyTorch env vars) | SHIPPED | `pub fn RendezvousConfig::from_env` in `gloo_native/connect.rs`; consumer: `gloo_backend.rs` `GlooBackend::from_env` invokes `GlooRendezvousConfig::from_env()`. |
//! | REQ-3 (PeerConn split halves) | SHIPPED | `pub(super) struct PeerConn` in `gloo_native/connect.rs`; consumer: `gloo_native/mod.rs` `fn conn` returns `&PeerConn`, used by `send_inner` / `recv_inner` to acquire the writer/reader halves. |
//! | REQ-4 (PeerStreams slot vector) | SHIPPED | `pub(super) type PeerStreams` in `gloo_native/connect.rs`; consumer: `gloo_native/mod.rs` `GlooBackendInner.connections: PeerStreams`. |
//! | REQ-5 (rendezvous entry point) | SHIPPED | `pub(super) fn rendezvous` in `gloo_native/connect.rs`; consumer: `gloo_native/mod.rs` `GlooBackendInner::new` calls `rendezvous(cfg)?`. |
//! | REQ-6 (3-step master/worker/full-mesh) | SHIPPED | `fn run_master` / `fn run_worker` / `fn form_full_mesh` in `gloo_native/connect.rs`; consumer: `pub(super) fn rendezvous` (same file) dispatches to them; tests `rendezvous_full_mesh_n2` and `_n4_all_slots_filled` exercise both protocol ends. |
//! | REQ-7 (bounded retry on connect-to-master) | SHIPPED | `RENDEZVOUS_RETRY_TIMEOUT` / `RENDEZVOUS_RETRY_INTERVAL` constants and retry loop in `fn run_worker` in `gloo_native/connect.rs`; consumer: every multi-rank `GlooBackendInner::new` path exercises it (race-tolerant by design). |
//! | REQ-8 (IPv4-only peer ad) | SHIPPED | `fn encode_peer_ad` matches `SocketAddr::V4` / rejects `V6` with `Io` error in `gloo_native/connect.rs`; consumer: `pub(super) fn rendezvous` invokes it; verified by `peer_ad_round_trip` test. |

use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener, TcpStream};
use std::sync::Mutex;
use std::time::Duration;

use crate::error::DistributedError;

use super::error::GlooResult;

/// 4 bytes IPv4 + 2 bytes LE u16 port = 6 bytes per peer advertisement.
const PEER_AD_BYTES: usize = 6;

/// Default per-connect attempt timeout. Each non-zero rank retries the
/// connect to rank 0 for up to this long before giving up — useful when
/// rank 0 hasn't quite bound its listener yet at launch time.
const RENDEZVOUS_RETRY_TIMEOUT: Duration = Duration::from_secs(30);
const RENDEZVOUS_RETRY_INTERVAL: Duration = Duration::from_millis(50);

/// One TCP socket between this rank and one peer, exposed as two separately
/// lockable halves so concurrent send + recv on the same peer don't block
/// each other.
///
/// `writer` and `reader` are two `TcpStream` views of the **same** OS
/// socket, obtained via [`TcpStream::try_clone`]. Each direction takes its
/// own [`std::sync::Mutex`], so `Backend::send(peer)` and
/// `Backend::recv(peer)` from different threads grab disjoint locks and
/// proceed concurrently. This is the standard "split read/write" pattern
/// for full-duplex TCP — without it, a 2-rank ring allreduce deadlocks
/// because `next == prev` and both directions compete for one mutex.
#[derive(Debug)]
pub(super) struct PeerConn {
    pub(super) writer: Mutex<TcpStream>,
    pub(super) reader: Mutex<TcpStream>,
}

impl PeerConn {
    fn from_stream(stream: TcpStream) -> GlooResult<Self> {
        let reader = stream.try_clone().map_err(|e| DistributedError::Io {
            message: format!("gloo_native try_clone read half: {e}"),
        })?;
        Ok(Self {
            writer: Mutex::new(stream),
            reader: Mutex::new(reader),
        })
    }
}

/// Per-peer connection slot. `None` for the self-slot.
pub(super) type PeerStreams = Vec<Option<PeerConn>>;

/// Parsed rendezvous configuration. Construct via
/// [`RendezvousConfig::from_env`] (PyTorch-compatible env vars) or by
/// supplying fields directly for testing.
#[derive(Debug, Clone)]
pub struct RendezvousConfig {
    /// Address of rank 0's rendezvous listener (`host:port`).
    pub master_addr: String,
    /// This process's rank within the world.
    pub rank: usize,
    /// Total number of ranks in the world.
    pub world_size: usize,
    /// Local address this rank's peer listener will bind to. Defaults to
    /// `127.0.0.1:0` (kernel-assigned port). The actual bound port is
    /// re-advertised to rank 0 during rendezvous.
    pub bind_addr: SocketAddr,
}

impl RendezvousConfig {
    /// Construct a rendezvous config from PyTorch's standard env vars:
    /// `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`.
    ///
    /// `bind_addr` defaults to `127.0.0.1:0`; override
    /// [`RendezvousConfig::bind_addr`] after construction for non-loopback
    /// deployments.
    pub fn from_env() -> GlooResult<Self> {
        fn env(key: &str) -> GlooResult<String> {
            std::env::var(key).map_err(|_| DistributedError::Io {
                message: format!("gloo_native rendezvous: env var `{key}` is not set"),
            })
        }
        fn parse_usize(key: &str, raw: &str) -> GlooResult<usize> {
            raw.parse::<usize>().map_err(|e| DistributedError::Io {
                message: format!("gloo_native rendezvous: env `{key}` parse: {e}"),
            })
        }

        let master_addr_host = env("MASTER_ADDR")?;
        let master_port = env("MASTER_PORT")?;
        let rank_raw = env("RANK")?;
        let world_size_raw = env("WORLD_SIZE")?;

        let rank = parse_usize("RANK", &rank_raw)?;
        let world_size = parse_usize("WORLD_SIZE", &world_size_raw)?;
        Ok(Self {
            master_addr: format!("{master_addr_host}:{master_port}"),
            rank,
            world_size,
            bind_addr: SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)),
        })
    }
}

/// Drive the rendezvous + full-mesh setup. Returns one [`TcpStream`] per
/// peer (with the self-slot held as `None`), wrapped in a `Mutex` so the
/// backend can take a per-peer lock for concurrent send/recv on disjoint
/// pairs.
///
/// Validates that `rank < world_size` and `world_size >= 2` before touching
/// the network.
pub(super) fn rendezvous(cfg: &RendezvousConfig) -> GlooResult<PeerStreams> {
    if cfg.world_size < 2 {
        return Err(DistributedError::InvalidWorldSize {
            world_size: cfg.world_size,
        });
    }
    if cfg.rank >= cfg.world_size {
        return Err(DistributedError::InvalidRank {
            rank: cfg.rank,
            world_size: cfg.world_size,
        });
    }

    // Step 1: bind this rank's peer listener (used by ranks > self to
    // connect to us in Step 3). `bind_addr` may have port=0 — capture the
    // kernel-assigned port for the rendezvous advertisement.
    let peer_listener = TcpListener::bind(cfg.bind_addr).map_err(|e| DistributedError::Io {
        message: format!("gloo_native rank {} bind peer listener: {e}", cfg.rank),
    })?;
    let my_addr = peer_listener
        .local_addr()
        .map_err(|e| DistributedError::Io {
            message: format!("gloo_native rank {} local_addr: {e}", cfg.rank),
        })?;
    let my_ad = encode_peer_ad(&my_addr)?;

    // Step 2: rendezvous via rank 0.
    let peer_table = if cfg.rank == 0 {
        run_master(cfg, my_ad)?
    } else {
        run_worker(cfg, my_ad)?
    };

    // Step 3: form the full mesh.
    form_full_mesh(cfg, &peer_listener, &peer_table)
}

/// Encode `addr` as 4 IPv4 bytes + 2 LE u16 port bytes. IPv6 is rejected
/// for simplicity (the rendezvous protocol assumes IPv4 endpoints, which
/// is what PyTorch's `MASTER_ADDR` defaults to in practice).
fn encode_peer_ad(addr: &SocketAddr) -> GlooResult<[u8; PEER_AD_BYTES]> {
    match addr {
        SocketAddr::V4(v4) => {
            let mut out = [0u8; PEER_AD_BYTES];
            out[..4].copy_from_slice(&v4.ip().octets());
            out[4..].copy_from_slice(&v4.port().to_le_bytes());
            Ok(out)
        }
        SocketAddr::V6(_) => Err(DistributedError::Io {
            message: "gloo_native rendezvous: IPv6 bind_addr is not supported".to_string(),
        }),
    }
}

/// Decode a 6-byte peer advertisement into a `SocketAddr`.
fn decode_peer_ad(buf: [u8; PEER_AD_BYTES]) -> SocketAddr {
    let ip = Ipv4Addr::new(buf[0], buf[1], buf[2], buf[3]);
    let port = u16::from_le_bytes([buf[4], buf[5]]);
    SocketAddr::V4(SocketAddrV4::new(ip, port))
}

/// Rank-0 side of rendezvous. Accepts `world_size - 1` connections, reads
/// `(rank u64, peer_ad)` from each, then writes back the full
/// `world_size * PEER_AD_BYTES` table to every connecting rank.
fn run_master(
    cfg: &RendezvousConfig,
    my_ad: [u8; PEER_AD_BYTES],
) -> GlooResult<Vec<[u8; PEER_AD_BYTES]>> {
    let listener = TcpListener::bind(&cfg.master_addr).map_err(|e| DistributedError::Io {
        message: format!("gloo_native rank 0 bind {}: {e}", cfg.master_addr),
    })?;

    let mut peer_table = vec![[0u8; PEER_AD_BYTES]; cfg.world_size];
    peer_table[0] = my_ad;
    let mut master_conns: Vec<(usize, TcpStream)> = Vec::with_capacity(cfg.world_size - 1);

    for _ in 1..cfg.world_size {
        let (mut stream, _) = listener.accept().map_err(|e| DistributedError::Io {
            message: format!("gloo_native rank 0 accept: {e}"),
        })?;
        let mut rank_buf = [0u8; 8];
        stream
            .read_exact(&mut rank_buf)
            .map_err(|e| DistributedError::Io {
                message: format!("gloo_native rank 0 read peer rank: {e}"),
            })?;
        let peer_rank = u64::from_le_bytes(rank_buf) as usize;
        if peer_rank == 0 || peer_rank >= cfg.world_size {
            return Err(DistributedError::InvalidRank {
                rank: peer_rank,
                world_size: cfg.world_size,
            });
        }
        let mut peer_ad = [0u8; PEER_AD_BYTES];
        stream
            .read_exact(&mut peer_ad)
            .map_err(|e| DistributedError::Io {
                message: format!("gloo_native rank 0 read peer ad: {e}"),
            })?;
        peer_table[peer_rank] = peer_ad;
        master_conns.push((peer_rank, stream));
    }

    // Broadcast the assembled table back to every non-zero rank.
    let mut flat = Vec::with_capacity(cfg.world_size * PEER_AD_BYTES);
    for ad in &peer_table {
        flat.extend_from_slice(ad);
    }
    for (_, mut s) in master_conns {
        s.write_all(&flat).map_err(|e| DistributedError::Io {
            message: format!("gloo_native rank 0 broadcast peer table: {e}"),
        })?;
        s.flush().map_err(|e| DistributedError::Io {
            message: format!("gloo_native rank 0 flush peer table: {e}"),
        })?;
    }

    Ok(peer_table)
}

/// Non-zero rank side of rendezvous. Connects to rank 0 with retry,
/// announces `(rank, peer_ad)`, then reads back the full peer table.
fn run_worker(
    cfg: &RendezvousConfig,
    my_ad: [u8; PEER_AD_BYTES],
) -> GlooResult<Vec<[u8; PEER_AD_BYTES]>> {
    let deadline = std::time::Instant::now() + RENDEZVOUS_RETRY_TIMEOUT;
    let mut stream = loop {
        match TcpStream::connect(&cfg.master_addr) {
            Ok(s) => break s,
            Err(_) if std::time::Instant::now() < deadline => {
                std::thread::sleep(RENDEZVOUS_RETRY_INTERVAL);
            }
            Err(e) => {
                return Err(DistributedError::Io {
                    message: format!(
                        "gloo_native rank {} connect to {} (after retries): {e}",
                        cfg.rank, cfg.master_addr,
                    ),
                });
            }
        }
    };

    stream
        .write_all(&(cfg.rank as u64).to_le_bytes())
        .map_err(|e| DistributedError::Io {
            message: format!("gloo_native rank {} announce rank: {e}", cfg.rank),
        })?;
    stream.write_all(&my_ad).map_err(|e| DistributedError::Io {
        message: format!("gloo_native rank {} announce ad: {e}", cfg.rank),
    })?;
    stream.flush().map_err(|e| DistributedError::Io {
        message: format!("gloo_native rank {} flush announce: {e}", cfg.rank),
    })?;

    let flat_len = cfg.world_size * PEER_AD_BYTES;
    let mut flat = vec![0u8; flat_len];
    stream
        .read_exact(&mut flat)
        .map_err(|e| DistributedError::Io {
            message: format!("gloo_native rank {} read peer table: {e}", cfg.rank),
        })?;

    let mut peer_table = vec![[0u8; PEER_AD_BYTES]; cfg.world_size];
    for (i, slot) in peer_table.iter_mut().enumerate() {
        slot.copy_from_slice(&flat[i * PEER_AD_BYTES..(i + 1) * PEER_AD_BYTES]);
    }
    Ok(peer_table)
}

/// Form the full mesh: for every unordered pair `(low, high)` with `low <
/// high`, the `low`-rank side accepts a connection on its peer listener
/// and the `high`-rank side initiates the connect. Rank `r` therefore
/// (a) accepts from all peers `> r` and (b) connects to all peers `< r`,
/// giving every rank one direct stream to every other rank.
fn form_full_mesh(
    cfg: &RendezvousConfig,
    peer_listener: &TcpListener,
    peer_table: &[[u8; PEER_AD_BYTES]],
) -> GlooResult<PeerStreams> {
    let mut streams: Vec<Option<TcpStream>> = (0..cfg.world_size).map(|_| None).collect();

    // (a) Accept from peers > self. We expect `world_size - 1 - rank`
    // inbound connections. Each connect sends a u64 LE peer rank as the
    // first frame so we can slot it into the right index.
    for _ in (cfg.rank + 1)..cfg.world_size {
        let (mut stream, _) = peer_listener.accept().map_err(|e| DistributedError::Io {
            message: format!("gloo_native rank {} accept full-mesh peer: {e}", cfg.rank,),
        })?;
        let mut rank_buf = [0u8; 8];
        stream
            .read_exact(&mut rank_buf)
            .map_err(|e| DistributedError::Io {
                message: format!(
                    "gloo_native rank {} read full-mesh peer rank: {e}",
                    cfg.rank,
                ),
            })?;
        let peer_rank = u64::from_le_bytes(rank_buf) as usize;
        if peer_rank <= cfg.rank || peer_rank >= cfg.world_size {
            return Err(DistributedError::InvalidRank {
                rank: peer_rank,
                world_size: cfg.world_size,
            });
        }
        streams[peer_rank] = Some(stream);
    }

    // (b) Connect to peers < self.
    for peer in 0..cfg.rank {
        let peer_addr = decode_peer_ad(peer_table[peer]);
        let mut stream = TcpStream::connect(peer_addr).map_err(|e| DistributedError::Io {
            message: format!(
                "gloo_native rank {} connect full-mesh peer {peer} at {peer_addr}: {e}",
                cfg.rank,
            ),
        })?;
        stream
            .write_all(&(cfg.rank as u64).to_le_bytes())
            .map_err(|e| DistributedError::Io {
                message: format!(
                    "gloo_native rank {} announce to full-mesh peer {peer}: {e}",
                    cfg.rank,
                ),
            })?;
        stream.flush().map_err(|e| DistributedError::Io {
            message: format!("gloo_native rank {} flush to peer {peer}: {e}", cfg.rank),
        })?;
        streams[peer] = Some(stream);
    }

    // Wrap each populated slot into a split-direction `PeerConn`. The
    // self-slot stays `None`. `try_clone` errors propagate via
    // `transpose()` on the per-slot `Option<Result<_, _>>`.
    streams
        .into_iter()
        .enumerate()
        .map(|(i, opt)| {
            if i == cfg.rank {
                Ok(None)
            } else {
                opt.map(PeerConn::from_stream).transpose()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    /// Spin up `world_size` in-process ranks against a shared
    /// `MASTER_ADDR:MASTER_PORT` rendezvous. Returns one connection map
    /// per rank.
    fn run_n_rank_rendezvous(world_size: usize) -> Vec<PeerStreams> {
        // Bind a kernel-assigned master port so parallel test runs don't
        // collide.
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
                    rendezvous(&cfg).expect("rendezvous")
                })
            })
            .collect();

        handles
            .into_iter()
            .map(|h| h.join().expect("join"))
            .collect()
    }

    #[test]
    fn rendezvous_full_mesh_n2() {
        let conns = run_n_rank_rendezvous(2);
        assert_eq!(conns.len(), 2);
        // Rank 0 has slot[1] populated and slot[0] empty (self).
        assert!(conns[0][0].is_none());
        assert!(conns[0][1].is_some());
        // Rank 1 has slot[0] populated and slot[1] empty (self).
        assert!(conns[1][0].is_some());
        assert!(conns[1][1].is_none());
    }

    #[test]
    fn rendezvous_full_mesh_n4_all_slots_filled() {
        let conns = run_n_rank_rendezvous(4);
        assert_eq!(conns.len(), 4);
        for (rank, slots) in conns.iter().enumerate() {
            assert_eq!(slots.len(), 4);
            for (peer, slot) in slots.iter().enumerate() {
                if peer == rank {
                    assert!(slot.is_none(), "rank {rank}: self-slot must be None");
                } else {
                    assert!(
                        slot.is_some(),
                        "rank {rank}: slot for peer {peer} must be Some"
                    );
                }
            }
        }
    }

    #[test]
    fn rendezvous_rejects_world_size_below_two() {
        let cfg = RendezvousConfig {
            master_addr: "127.0.0.1:0".to_string(),
            rank: 0,
            world_size: 1,
            bind_addr: SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)),
        };
        let err = rendezvous(&cfg).expect_err("must reject world_size=1");
        match err {
            DistributedError::InvalidWorldSize { world_size } => assert_eq!(world_size, 1),
            other => panic!("expected InvalidWorldSize, got {other:?}"),
        }
    }

    #[test]
    fn rendezvous_rejects_rank_out_of_range() {
        let cfg = RendezvousConfig {
            master_addr: "127.0.0.1:0".to_string(),
            rank: 5,
            world_size: 4,
            bind_addr: SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)),
        };
        let err = rendezvous(&cfg).expect_err("must reject rank >= world_size");
        match err {
            DistributedError::InvalidRank { rank, world_size } => {
                assert_eq!(rank, 5);
                assert_eq!(world_size, 4);
            }
            other => panic!("expected InvalidRank, got {other:?}"),
        }
    }

    #[test]
    fn peer_ad_round_trip() {
        let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 51234));
        let ad = encode_peer_ad(&addr).expect("encode");
        let back = decode_peer_ad(ad);
        assert_eq!(addr, back);
    }
}
