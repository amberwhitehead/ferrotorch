//! Ring collective algorithms for the native Rust Gloo backend.
//!
//! All algorithms here are textbook ring versions parameterised by a
//! `&dyn RingTransport` trait so the unit tests can drive them over a
//! `SimulatedBackend` instead of TCP. The production wiring in
//! [`super::mod`] adapts [`super::GlooBackend`] to this trait via the
//! crate's [`Backend`](crate::backend::Backend) trait — `send(&[u8], dst)`
//! and `recv(&mut [u8], src)` are the only primitives the collectives need.
//!
//! # Ring topology
//!
//! Rank `r` always sends to `(r + 1) mod world_size` and receives from
//! `(r + world_size - 1) mod world_size`. The collectives never address
//! peers other than `prev` / `next`, so even though [`super::connect`]
//! builds a full mesh for `send`/`recv` flexibility, the ring routines
//! only ever exercise two of those edges.
//!
//! # Reduction semantics
//!
//! Allreduce currently supports element-wise sum on `f32` byte buffers.
//! The byte-level interface is chosen so the implementation can be reused
//! by the trait-level [`Backend`](crate::backend::Backend)
//! collectives in [`crate::collective`] without paying a typed-tensor
//! dependency at this layer.
//!
//! ## REQ status (per `.design/ferrotorch-distributed/gloo_native/collectives.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (RingTransport trait) | SHIPPED | `pub(super) trait RingTransport: Sync` in `gloo_native/collectives.rs`; consumer: `impl RingTransport for GlooBackendInner` in `gloo_native/mod.rs`. |
//! | REQ-2 (ring_allreduce_sum_f32_bytes) | SHIPPED | `pub(super) fn ring_allreduce_sum_f32_bytes` in `gloo_native/collectives.rs`; consumer: `gloo_native/mod.rs` `pub fn ring_allreduce_sum_f32_with_timeout` invokes it; that is consumed by `GlooBackend::ring_allreduce_sum_f32`, `MpiBackend::allreduce_sum_f32`, `UccBackend::allreduce_sum_f32`. |
//! | REQ-3 (chunk_ranges cover-once) | SHIPPED | `fn chunk_ranges` and `const fn bytes_of` in `gloo_native/collectives.rs`; consumer: `pub(super) fn ring_allreduce_sum_f32_bytes` (same file) calls `chunk_ranges` and `bytes_of` at every step. |
//! | REQ-4 (scoped-thread send/recv) | SHIPPED | `fn send_recv` (scoped-thread shape) in `gloo_native/collectives.rs`; consumer: `pub(super) fn ring_allreduce_sum_f32_bytes` calls `send_recv` at every ring step. Disjoint-locks safety relies on `PeerConn`'s split halves in `gloo_native/connect.rs`. |
//! | REQ-5 (tree_broadcast_f32_bytes) | SHIPPED | `pub(super) fn tree_broadcast_f32_bytes` in `gloo_native/collectives.rs`; consumer: `gloo_native/mod.rs` `pub fn tree_broadcast_f32_with_timeout` invokes it; that is consumed by `gloo_backend.rs`, `mpi_backend.rs`, `ucc_backend.rs` broadcast methods. |
//! | REQ-6 (ring_barrier two-wave) | SHIPPED | `pub(super) fn ring_barrier` in `gloo_native/collectives.rs`; consumer: `gloo_native/mod.rs` `impl Backend for GlooBackendInner::barrier` invokes `ring_barrier(self, DEFAULT_GLOO_TIMEOUT)`. |
//! | REQ-7 (edge cases: world=1, empty, %4) | SHIPPED | early-return guards at the top of `ring_allreduce_sum_f32_bytes` in `gloo_native/collectives.rs`; consumer: every collective invocation through `GlooBackendInner` traverses these guards. |
//! | REQ-8 (accumulate_f32_inplace) | SHIPPED | `fn accumulate_f32_inplace` in `gloo_native/collectives.rs`; consumer: `pub(super) fn ring_allreduce_sum_f32_bytes` (same file) calls `accumulate_f32_inplace` at every scatter-reduce step. |

use std::time::Duration;

use crate::error::DistributedError;

use super::error::GlooResult;

/// Minimal point-to-point transport surface the ring algorithms need.
///
/// Implemented by [`super::GlooBackend`] in production and by an in-process
/// channel shim in tests so we can exercise the ring shape without binding
/// real TCP sockets in every test.
pub(super) trait RingTransport: Sync {
    fn rank(&self) -> usize;
    fn world_size(&self) -> usize;
    fn send(&self, data: &[u8], dst: usize) -> GlooResult<()>;
    fn recv(&self, dst: &mut [u8], src: usize, timeout: Duration) -> GlooResult<()>;
}

/// Ring neighbours for `rank` in a world of size `world_size`. Returns
/// `(prev, next)` where `next = (rank + 1) mod world_size` and `prev = (rank
/// + world_size - 1) mod world_size`.
fn ring_neighbours(rank: usize, world_size: usize) -> (usize, usize) {
    debug_assert!(world_size >= 2);
    let next = (rank + 1) % world_size;
    let prev = (rank + world_size - 1) % world_size;
    (prev, next)
}

// ---------------------------------------------------------------------------
// Ring allreduce (sum on f32 buffers)
// ---------------------------------------------------------------------------

/// Ring allreduce over an `f32` byte buffer.
///
/// `buf` must be a byte-view of an `f32` slice (`buf.len() % 4 == 0`). The
/// data is conceptually chunked into `world_size` shards along the linear
/// element axis; chunk boundaries are at element indices `chunk_lo(i) =
/// i * n / world_size` to keep total bytes exactly `buf.len()` even when
/// `n % world_size != 0` (the tail chunk simply absorbs the remainder).
///
/// # Phases
///
/// 1. **Scatter-reduce** (`world_size - 1` steps). At step `s` rank `r`
///    sends chunk `(r - s) mod world_size` to `next` and receives chunk
///    `(r - s - 1) mod world_size` from `prev`, accumulating element-wise
///    sum into its local copy of that chunk.
/// 2. **Allgather** (`world_size - 1` steps). At step `s` rank `r` sends
///    chunk `(r - s + 1) mod world_size` (the freshly reduced one) to
///    `next` and receives chunk `(r - s) mod world_size` from `prev`,
///    **replacing** that chunk's contents.
///
/// After the two phases every rank's `buf` holds the element-wise sum of
/// every starting `buf`.
///
/// # Communication cost
///
/// Total bytes moved per rank: `2 * (N - 1) / N * buf.len()`, which is
/// asymptotically `2 * buf.len()` for large `N` — the communication-optimal
/// allreduce volume.
pub(super) fn ring_allreduce_sum_f32_bytes(
    transport: &dyn RingTransport,
    buf: &mut [u8],
    timeout: Duration,
) -> GlooResult<()> {
    let rank = transport.rank();
    let world_size = transport.world_size();
    if world_size == 1 {
        return Ok(());
    }
    if buf.is_empty() {
        return Ok(());
    }
    if buf.len() % std::mem::size_of::<f32>() != 0 {
        return Err(DistributedError::SizeMismatch {
            expected: buf.len() - (buf.len() % std::mem::size_of::<f32>()),
            got: buf.len(),
        });
    }

    let total_elems = buf.len() / std::mem::size_of::<f32>();
    let chunk_ranges = chunk_ranges(total_elems, world_size);
    let (prev, next) = ring_neighbours(rank, world_size);

    // Phase 1: scatter-reduce.
    for step in 0..(world_size - 1) {
        let send_chunk = (rank + world_size - step) % world_size;
        let recv_chunk = (rank + world_size - step - 1) % world_size;
        let (send_lo, send_hi) = chunk_ranges[send_chunk];
        let (recv_lo, recv_hi) = chunk_ranges[recv_chunk];

        // Stage the send-chunk in its own buffer so we can interleave
        // a blocking send and recv without aliasing rules complaining.
        let send_bytes = buf[bytes_of(send_lo)..bytes_of(send_hi)].to_vec();
        let mut recv_bytes = vec![0u8; bytes_of(recv_hi) - bytes_of(recv_lo)];

        send_recv(transport, &send_bytes, next, &mut recv_bytes, prev, timeout)?;

        // Accumulate the received chunk into our local slot.
        accumulate_f32_inplace(&mut buf[bytes_of(recv_lo)..bytes_of(recv_hi)], &recv_bytes);
    }

    // Phase 2: allgather. Each rank's chunk `(rank + 1) mod N` now holds
    // the final reduction (this is the chunk it last accumulated into).
    // We rotate it around the ring so every rank gets every chunk.
    for step in 0..(world_size - 1) {
        let send_chunk = (rank + world_size + 1 - step) % world_size;
        let recv_chunk = (rank + world_size - step) % world_size;
        let (send_lo, send_hi) = chunk_ranges[send_chunk];
        let (recv_lo, recv_hi) = chunk_ranges[recv_chunk];

        let send_bytes = buf[bytes_of(send_lo)..bytes_of(send_hi)].to_vec();
        let mut recv_bytes = vec![0u8; bytes_of(recv_hi) - bytes_of(recv_lo)];

        send_recv(transport, &send_bytes, next, &mut recv_bytes, prev, timeout)?;

        // Overwrite (NOT accumulate) the local chunk with the received
        // bytes — this is the allgather, not the scatter-reduce.
        buf[bytes_of(recv_lo)..bytes_of(recv_hi)].copy_from_slice(&recv_bytes);
    }

    Ok(())
}

/// Compute `[lo, hi)` element ranges for each of `world_size` chunks over
/// `total_elems` elements. The last chunk absorbs the remainder.
fn chunk_ranges(total_elems: usize, world_size: usize) -> Vec<(usize, usize)> {
    (0..world_size)
        .map(|i| {
            let lo = i * total_elems / world_size;
            let hi = if i + 1 == world_size {
                total_elems
            } else {
                (i + 1) * total_elems / world_size
            };
            (lo, hi)
        })
        .collect()
}

const fn bytes_of(elem_idx: usize) -> usize {
    elem_idx * std::mem::size_of::<f32>()
}

/// In-place `dst[i] += src[i]` over `f32` byte slices of equal length.
fn accumulate_f32_inplace(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    debug_assert_eq!(dst.len() % std::mem::size_of::<f32>(), 0);
    for (d_chunk, s_chunk) in dst
        .chunks_exact_mut(std::mem::size_of::<f32>())
        .zip(src.chunks_exact(std::mem::size_of::<f32>()))
    {
        let d_arr: [u8; 4] = d_chunk.try_into().expect("4-byte chunk");
        let s_arr: [u8; 4] = s_chunk.try_into().expect("4-byte chunk");
        let new = f32::from_le_bytes(d_arr) + f32::from_le_bytes(s_arr);
        d_chunk.copy_from_slice(&new.to_le_bytes());
    }
}

/// One ring step: send-to-next + recv-from-prev.
///
/// # Concurrency strategy
///
/// The send runs on a scoped worker thread; the caller thread blocks on
/// the recv. For `world_size >= 3` the `next` and `prev` peers are
/// distinct so each direction takes a disjoint per-peer lock. For
/// `world_size == 2` the same TCP socket carries both directions, but
/// `super::connect::PeerConn` splits the socket into independent reader /
/// writer halves (`try_clone` + two separate `Mutex`es), so the parallel
/// shape still avoids deadlock.
fn send_recv(
    transport: &dyn RingTransport,
    send_bytes: &[u8],
    next: usize,
    recv_bytes: &mut [u8],
    prev: usize,
    timeout: Duration,
) -> GlooResult<()> {
    std::thread::scope(|scope| {
        let send_handle = scope.spawn(move || transport.send(send_bytes, next));
        let recv_result = transport.recv(recv_bytes, prev, timeout);
        let send_result = send_handle.join().map_err(|_| DistributedError::Io {
            message: "gloo_native ring send worker panicked".to_string(),
        })?;
        send_result?;
        recv_result?;
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Tree broadcast
// ---------------------------------------------------------------------------

/// Binary-tree broadcast: `root` sends to its two children; each child
/// recursively forwards to its own two children. Depth is `ceil(log2(N))`.
///
/// Tree shape: rank `r` (in tree coords) has children at `2r + 1` and
/// `2r + 2`. We use rank-rooted tree coords by remapping `tree_rank = (rank
/// + world_size - root) mod world_size`, so the protocol is symmetric in
/// `root`.
pub(super) fn tree_broadcast_f32_bytes(
    transport: &dyn RingTransport,
    buf: &mut [u8],
    root: usize,
    timeout: Duration,
) -> GlooResult<()> {
    let rank = transport.rank();
    let world_size = transport.world_size();
    if world_size == 1 {
        return Ok(());
    }
    if root >= world_size {
        return Err(DistributedError::InvalidRank {
            rank: root,
            world_size,
        });
    }

    let tree_rank = (rank + world_size - root) % world_size;

    // Non-root ranks: receive from parent before forwarding.
    if tree_rank != 0 {
        let parent_tree = (tree_rank - 1) / 2;
        let parent = (parent_tree + root) % world_size;
        transport.recv(buf, parent, timeout)?;
    }

    // Forward to up to two children.
    for child_tree in [tree_rank * 2 + 1, tree_rank * 2 + 2] {
        if child_tree < world_size {
            let child = (child_tree + root) % world_size;
            transport.send(buf, child)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Ring barrier
// ---------------------------------------------------------------------------

/// Ring barrier: a single 1-byte token is forwarded all the way around the
/// ring twice (forward to confirm all ranks reached the barrier, then a
/// second time as the release signal). This is the simplest barrier that
/// guarantees no rank exits before every rank has entered, without
/// requiring a centralised coordinator.
pub(super) fn ring_barrier(transport: &dyn RingTransport, timeout: Duration) -> GlooResult<()> {
    let rank = transport.rank();
    let world_size = transport.world_size();
    if world_size == 1 {
        return Ok(());
    }
    let (prev, next) = ring_neighbours(rank, world_size);
    let token = [0u8; 1];

    // Phase 1: arrival wave. Rank 0 starts; everyone else waits for prev,
    // then forwards to next. After this loop every rank has heard from
    // every other rank's prev, transitively, so every rank has entered.
    if rank == 0 {
        transport.send(&token, next)?;
        let mut buf = [0u8; 1];
        transport.recv(&mut buf, prev, timeout)?;
    } else {
        let mut buf = [0u8; 1];
        transport.recv(&mut buf, prev, timeout)?;
        transport.send(&token, next)?;
    }

    // Phase 2: release wave. Same shape, second token. Without this, rank
    // 0 could exit while rank 1 is still waiting for it.
    if rank == 0 {
        transport.send(&token, next)?;
        let mut buf = [0u8; 1];
        transport.recv(&mut buf, prev, timeout)?;
    } else {
        let mut buf = [0u8; 1];
        transport.recv(&mut buf, prev, timeout)?;
        transport.send(&token, next)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::mpsc::{Receiver, Sender, channel};

    /// In-process transport for testing the ring algorithms.
    ///
    /// One in-process `(Sender, Receiver)` pair per `(src, dst)` rank
    /// combination. Wrapped in `Mutex` because both `Sender` and
    /// `Receiver` from `std::sync::mpsc` are `!Sync`.
    type ChannelPair = (Mutex<Sender<Vec<u8>>>, Mutex<Receiver<Vec<u8>>>);
    type ChannelMatrix = Vec<Vec<ChannelPair>>;

    /// `channels[src][dst]` is the `(Sender, Receiver)` pair for messages
    /// flowing `src -> dst`. Each `Channels` instance is shared by every
    /// rank; per-rank views are wrapped in [`RankView`].
    struct Channels {
        inner: ChannelMatrix,
    }
    impl Channels {
        fn new(world_size: usize) -> Self {
            let inner = (0..world_size)
                .map(|_src| {
                    (0..world_size)
                        .map(|_dst| {
                            let (tx, rx) = channel();
                            (Mutex::new(tx), Mutex::new(rx))
                        })
                        .collect()
                })
                .collect();
            Self { inner }
        }
    }

    struct RankView<'a> {
        my_rank: usize,
        world_size: usize,
        ch: &'a Channels,
    }

    impl RingTransport for RankView<'_> {
        fn rank(&self) -> usize {
            self.my_rank
        }
        fn world_size(&self) -> usize {
            self.world_size
        }
        fn send(&self, data: &[u8], dst: usize) -> GlooResult<()> {
            self.ch.inner[self.my_rank][dst]
                .0
                .lock()
                .unwrap()
                .send(data.to_vec())
                .map_err(|e| DistributedError::ChannelClosed {
                    message: format!("test send {} -> {dst}: {e}", self.my_rank),
                })?;
            Ok(())
        }
        fn recv(&self, dst: &mut [u8], src: usize, _timeout: Duration) -> GlooResult<()> {
            let v = self.ch.inner[src][self.my_rank]
                .1
                .lock()
                .unwrap()
                .recv()
                .map_err(|e| DistributedError::ChannelClosed {
                    message: format!("test recv {src} -> {}: {e}", self.my_rank),
                })?;
            if v.len() != dst.len() {
                return Err(DistributedError::SizeMismatch {
                    expected: dst.len(),
                    got: v.len(),
                });
            }
            dst.copy_from_slice(&v);
            Ok(())
        }
    }

    fn floats_to_le_bytes(xs: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(xs.len() * 4);
        for &x in xs {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }
    fn le_bytes_to_floats(bs: &[u8]) -> Vec<f32> {
        bs.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn ring_allreduce_two_ranks_sum() {
        let channels = Channels::new(2);
        let inputs = [[1.0f32, 2.0, 3.0], [4.0f32, 5.0, 6.0]];

        let mut bufs: Vec<Vec<u8>> = inputs.iter().map(|x| floats_to_le_bytes(x)).collect();

        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for (rank, buf) in bufs.iter_mut().enumerate() {
                let ch = &channels;
                handles.push(s.spawn(move || {
                    let r = RankView {
                        my_rank: rank,
                        world_size: 2,
                        ch,
                    };
                    ring_allreduce_sum_f32_bytes(&r, buf, Duration::from_secs(5))
                        .expect("allreduce");
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });

        let expected = [5.0f32, 7.0, 9.0];
        for (i, buf) in bufs.iter().enumerate() {
            let got = le_bytes_to_floats(buf);
            assert_eq!(got, expected, "rank {i}");
        }
    }

    #[test]
    fn ring_allreduce_four_ranks_sum_with_uneven_chunks() {
        // 7 elements over 4 ranks → chunk sizes 1, 1, 1, 4. Exercises the
        // remainder-into-last-chunk path.
        let world_size = 4;
        let channels = Channels::new(world_size);
        // Inputs designed so the sum is easy to verify by hand:
        // rank 0: [1, 1, 1, 1, 1, 1, 1]
        // rank 1: [2, 2, 2, 2, 2, 2, 2]
        // rank 2: [4, 4, 4, 4, 4, 4, 4]
        // rank 3: [8, 8, 8, 8, 8, 8, 8]
        // sum    = [15, 15, 15, 15, 15, 15, 15]
        let inputs: Vec<Vec<f32>> = [1.0f32, 2.0, 4.0, 8.0]
            .iter()
            .map(|&v| vec![v; 7])
            .collect();

        let mut bufs: Vec<Vec<u8>> = inputs.iter().map(|x| floats_to_le_bytes(x)).collect();

        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for (rank, buf) in bufs.iter_mut().enumerate() {
                let ch = &channels;
                handles.push(s.spawn(move || {
                    let r = RankView {
                        my_rank: rank,
                        world_size,
                        ch,
                    };
                    ring_allreduce_sum_f32_bytes(&r, buf, Duration::from_secs(5))
                        .expect("allreduce");
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });

        let expected = vec![15.0f32; 7];
        for (i, buf) in bufs.iter().enumerate() {
            let got = le_bytes_to_floats(buf);
            assert_eq!(got, expected, "rank {i}");
        }
    }

    #[test]
    fn ring_allreduce_three_ranks_sum() {
        let world_size = 3;
        let channels = Channels::new(world_size);
        let inputs = [
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
            vec![100.0f32, 200.0, 300.0, 400.0, 500.0, 600.0],
        ];
        let mut bufs: Vec<Vec<u8>> = inputs.iter().map(|x| floats_to_le_bytes(x)).collect();

        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for (rank, buf) in bufs.iter_mut().enumerate() {
                let ch = &channels;
                handles.push(s.spawn(move || {
                    let r = RankView {
                        my_rank: rank,
                        world_size,
                        ch,
                    };
                    ring_allreduce_sum_f32_bytes(&r, buf, Duration::from_secs(5))
                        .expect("allreduce");
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });

        let expected = vec![111.0f32, 222.0, 333.0, 444.0, 555.0, 666.0];
        for (i, buf) in bufs.iter().enumerate() {
            let got = le_bytes_to_floats(buf);
            assert_eq!(got, expected, "rank {i}");
        }
    }

    #[test]
    fn tree_broadcast_distributes_from_root() {
        let world_size = 4;
        let channels = Channels::new(world_size);
        let payload = floats_to_le_bytes(&[42.0, 43.0, 44.0]);

        let mut bufs: Vec<Vec<u8>> = (0..world_size)
            .map(|r| {
                if r == 1 {
                    payload.clone()
                } else {
                    vec![0u8; payload.len()]
                }
            })
            .collect();

        // Broadcast from rank 1.
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for (rank, buf) in bufs.iter_mut().enumerate() {
                let ch = &channels;
                handles.push(s.spawn(move || {
                    let r = RankView {
                        my_rank: rank,
                        world_size,
                        ch,
                    };
                    tree_broadcast_f32_bytes(&r, buf, 1, Duration::from_secs(5))
                        .expect("broadcast");
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });

        for (rank, buf) in bufs.iter().enumerate() {
            let got = le_bytes_to_floats(buf);
            assert_eq!(got, vec![42.0, 43.0, 44.0], "rank {rank}");
        }
    }

    #[test]
    fn ring_barrier_serialises_all_ranks() {
        // Discriminating test: if the barrier returns early for any rank,
        // a stricter test that asserts every rank has entered before any
        // exits would catch it. We use a per-rank "entered" counter
        // protected by a Mutex; after each barrier call we assert
        // counter >= world_size.
        let world_size = 4;
        let channels = Channels::new(world_size);
        let entered = std::sync::atomic::AtomicUsize::new(0);

        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for rank in 0..world_size {
                let ch = &channels;
                let entered_ref = &entered;
                handles.push(s.spawn(move || {
                    let r = RankView {
                        my_rank: rank,
                        world_size,
                        ch,
                    };
                    // Mark entered, then barrier. After barrier returns,
                    // every rank must have marked entered.
                    entered_ref.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    ring_barrier(&r, Duration::from_secs(5)).expect("barrier");
                    let n = entered_ref.load(std::sync::atomic::Ordering::SeqCst);
                    assert_eq!(
                        n, world_size,
                        "rank {rank}: expected all {world_size} to be entered, saw {n}"
                    );
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });
    }

    #[test]
    fn chunk_ranges_balanced() {
        let r = chunk_ranges(8, 4);
        assert_eq!(r, vec![(0, 2), (2, 4), (4, 6), (6, 8)]);
    }

    #[test]
    fn chunk_ranges_unbalanced_cover_all_elements_exactly_once() {
        // 7 elements / 4 chunks: proportional partition yields chunk sizes
        // (1, 2, 2, 2) via the `i * n / world_size` formula, with the
        // last chunk pinned to `total_elems` so floor-division never
        // drops a tail element.
        let r = chunk_ranges(7, 4);
        assert_eq!(r, vec![(0, 1), (1, 3), (3, 5), (5, 7)]);
        // Strict cover-once invariant: ranges are contiguous, non-overlapping,
        // and span exactly [0, total_elems).
        assert_eq!(r.first().unwrap().0, 0);
        assert_eq!(r.last().unwrap().1, 7);
        for w in r.windows(2) {
            assert_eq!(w[0].1, w[1].0, "chunk boundaries must abut");
        }
        let total: usize = r.iter().map(|(lo, hi)| hi - lo).sum();
        assert_eq!(total, 7);
    }

    #[test]
    fn ring_neighbours_wrap_around() {
        assert_eq!(ring_neighbours(0, 4), (3, 1));
        assert_eq!(ring_neighbours(3, 4), (2, 0));
        assert_eq!(ring_neighbours(2, 4), (1, 3));
    }
}
