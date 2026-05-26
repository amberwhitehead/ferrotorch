//! Error types for distributed operations.
//!
//! ## REQ status (per `.design/ferrotorch-distributed/error.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream
//! cites) live in the design doc; this synopsis is a one-line summary
//! per REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (DistributedError enum) | SHIPPED | `pub enum DistributedError` in `error.rs` with 11 `#[non_exhaustive]` variants; consumers `use crate::error::DistributedError;` in `backend.rs`, `collective.rs`, `gloo_backend.rs`. |
//! | REQ-2 (diagnostic fields per variant) | SHIPPED | every variant carries named fields rendered in `#[error("...")]` strings; verified by `backend.rs` tests (`test_invalid_world_size`, `test_send_to_invalid_rank`). |
//! | REQ-3 (From conversion) | SHIPPED | `impl From<DistributedError> for FerrotorchError` at the bottom of `error.rs`; consumers `.into()` at every fallible site in `backend.rs` and `collective.rs`. |
//! | REQ-4 (BackendUnavailable variant) | SHIPPED | `BackendUnavailable { backend: &'static str }` variant in `error.rs`; consumers in `gloo_backend.rs`, `mpi_backend.rs`, `ucc_backend.rs` (feature-off construction paths). |

use ferrotorch_core::FerrotorchError;

/// Errors specific to the distributed training subsystem.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DistributedError {
    #[error("invalid world size: {world_size} (must be >= 1)")]
    InvalidWorldSize { world_size: usize },

    #[error("invalid rank {rank} for world size {world_size}")]
    InvalidRank { rank: usize, world_size: usize },

    #[error("cannot send to self (rank {rank})")]
    SelfSend { rank: usize },

    #[error("size mismatch: expected {expected} bytes, got {got}")]
    SizeMismatch { expected: usize, got: usize },

    #[error("I/O error: {message}")]
    Io { message: String },

    #[error("lock poisoned: {message}")]
    LockPoisoned { message: String },

    #[error("channel closed: {message}")]
    ChannelClosed { message: String },

    #[error("unsupported reduce operation: {message}")]
    UnsupportedOp { message: String },

    #[error("operation timed out after {seconds}s")]
    Timeout { seconds: u64 },

    #[error("no connection to rank {rank} (star topology: non-zero ranks only connect to rank 0)")]
    NoConnection { rank: usize },

    /// Returned when the user requested a backend whose binding layer
    /// isn't compiled into this build (e.g. `gloo-backend` / `mpi-backend`
    /// / `ucc-backend` feature off, or a CUDA-required backend on a
    /// non-CUDA system). The caller is expected to either enable the
    /// feature, install the underlying C library, or pick a different
    /// backend (`SimulatedBackend` / `TcpBackend` always work).
    /// (Replaces closed #459; live follow-ups: #1132 / #1133 / #1134.)
    #[error(
        "backend `{backend}` is not available in this build (enable the corresponding cargo feature \
         and ensure the underlying library is installed)"
    )]
    BackendUnavailable { backend: &'static str },
}

impl From<DistributedError> for FerrotorchError {
    fn from(e: DistributedError) -> Self {
        FerrotorchError::InvalidArgument {
            message: e.to_string(),
        }
    }
}
