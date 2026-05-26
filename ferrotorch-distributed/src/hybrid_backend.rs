//! Hybrid TCP+NCCL backend for distributed training.
//!
//! [`HybridBackend`] combines a [`TcpBackend`] for point-to-point
//! communication (send/recv/barrier) with an [`NcclBackend`] for
//! GPU-native collective operations (allreduce, broadcast, etc.).
//!
//! This matches PyTorch's `ProcessGroupNCCL` behavior where NCCL handles
//! collectives and Gloo/TCP handles P2P fallback.
//!
//! # Feature gate
//!
//! Requires the `nccl` feature.
//!
//! ## REQ status (per `.design/ferrotorch-distributed/hybrid_backend.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (HybridBackend struct) | SHIPPED | `pub struct HybridBackend { tcp: TcpBackend, nccl: NcclBackend }` in `hybrid_backend.rs`; consumer `pub use hybrid_backend::HybridBackend` at `lib.rs` under `#[cfg(feature = "nccl")]`. |
//! | REQ-2 (constructor order TCP→NCCL) | SHIPPED | `pub fn new` builds `TcpBackend::new(...)` first then `NcclBackend::new(...)` in `hybrid_backend.rs`; consumer via crate-root re-export reachable from `ferrotorch/src/lib.rs`. |
//! | REQ-3 (nccl() / tcp() accessors) | SHIPPED | `pub fn nccl(&self) -> &NcclBackend` and `pub fn tcp(&self) -> &TcpBackend` in `hybrid_backend.rs`; documented production call shape `nccl_allreduce(&mut buf, hybrid.nccl(), ...)`. |
//! | REQ-4 (synchronize_nccl) | SHIPPED | `pub fn synchronize_nccl(&self)` in `hybrid_backend.rs` forwards to `NcclBackend::synchronize`; consumer via crate-root re-export. |
//! | REQ-5 (Backend trait delegation to tcp) | SHIPPED | `impl Backend for HybridBackend` in `hybrid_backend.rs` delegates all six methods to `self.tcp`; consumer is every `&dyn Backend`-accepting fn in `crate::collective::*` and `crate::p2p::*`. |

use std::time::Duration;

use ferrotorch_core::FerrotorchResult;

use crate::backend::{Backend, TcpBackend};
use crate::nccl_backend::NcclBackend;
use crate::nccl_sys::NcclUniqueId;

/// Hybrid backend combining TCP for P2P and NCCL for GPU collectives.
///
/// Use the [`Backend`] trait methods for P2P (delegated to TCP), and
/// access the inner [`NcclBackend`] via [`nccl()`](Self::nccl) for
/// GPU-native collective operations.
///
/// # Example
///
/// ```ignore
/// let hybrid = HybridBackend::new(rank, world_size, addr, unique_id)?;
///
/// // P2P via TCP
/// hybrid.send(&data, dst_rank)?;
///
/// // GPU collectives via NCCL
/// nccl_allreduce(&mut gpu_buffer, hybrid.nccl(), &ReduceOp::Sum)?;
/// ```
pub struct HybridBackend {
    tcp: TcpBackend,
    nccl: NcclBackend,
}

impl HybridBackend {
    /// Create a hybrid backend.
    ///
    /// `rank` and `world_size` define the process group. `addr` is the
    /// TCP rendezvous address (rank 0 listens, others connect).
    /// `unique_id` is the NCCL unique ID (same on all ranks).
    ///
    /// The correct CUDA device must be set before calling.
    pub fn new(
        rank: usize,
        world_size: usize,
        addr: &str,
        unique_id: NcclUniqueId,
    ) -> FerrotorchResult<Self> {
        let tcp = TcpBackend::new(rank, world_size, addr)?;
        let nccl = NcclBackend::new(rank, world_size, unique_id)?;
        Ok(Self { tcp, nccl })
    }

    /// Access the inner NCCL backend for GPU-native collectives.
    pub fn nccl(&self) -> &NcclBackend {
        &self.nccl
    }

    /// Access the inner TCP backend for direct use.
    pub fn tcp(&self) -> &TcpBackend {
        &self.tcp
    }

    /// Synchronize the NCCL stream — blocks until all enqueued NCCL
    /// operations have completed.
    pub fn synchronize_nccl(&self) -> FerrotorchResult<()> {
        self.nccl.synchronize()
    }
}

impl Backend for HybridBackend {
    fn rank(&self) -> usize {
        self.tcp.rank()
    }

    fn world_size(&self) -> usize {
        self.tcp.world_size()
    }

    fn send(&self, data: &[u8], dst_rank: usize) -> FerrotorchResult<()> {
        self.tcp.send(data, dst_rank)
    }

    fn recv(&self, dst: &mut [u8], src_rank: usize) -> FerrotorchResult<()> {
        self.tcp.recv(dst, src_rank)
    }

    fn recv_timeout(
        &self,
        dst: &mut [u8],
        src_rank: usize,
        timeout: Duration,
    ) -> FerrotorchResult<()> {
        self.tcp.recv_timeout(dst, src_rank, timeout)
    }

    fn barrier(&self) -> FerrotorchResult<()> {
        // Use TCP barrier (reliable) rather than NCCL barrier (requires GPU).
        self.tcp.barrier()
    }
}
