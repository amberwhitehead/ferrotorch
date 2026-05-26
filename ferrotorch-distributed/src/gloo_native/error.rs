//! Error type aliases for the native Rust Gloo backend.
//!
//! We deliberately reuse [`DistributedError`](crate::error::DistributedError)
//! rather than introducing a fresh `GlooError` enum: every failure mode the
//! native backend can surface (I/O, size mismatch, invalid rank, lock
//! poisoning, timeout, no-connection) already has a variant on
//! `DistributedError`, and adding a parallel taxonomy would force callers
//! to write match arms twice. The `GlooResult` alias is the only new name
//! introduced here, and exists solely so the rest of `gloo_native::` reads
//! `-> GlooResult<T>` instead of the longer
//! `Result<T, DistributedError>`.
//!
//! ## REQ status (per `.design/ferrotorch-distributed/gloo_native/error.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (GlooResult type alias) | SHIPPED | `pub(super) type GlooResult<T>` in `gloo_native/error.rs`; consumers: every sibling sub-module — `gloo_native/mod.rs` (`use self::error::GlooResult;` and every internal helper returns the alias), `gloo_native/transport.rs`, `gloo_native/connect.rs`, `gloo_native/collectives.rs`. |
//! | REQ-2 (re-uses DistributedError, no GlooError enum) | SHIPPED | this module's doc comment documents the re-use decision; consumer: every `gloo_native::*` error site produces a `DistributedError` variant directly (e.g., `DistributedError::Io`, `DistributedError::SizeMismatch`, `DistributedError::LockPoisoned`) in `transport.rs`, `connect.rs`, and `mod.rs`. No `enum GlooError` exists in the crate. |

use crate::error::DistributedError;

/// Internal `Result` alias used throughout `gloo_native::`. The error type
/// is the workspace-wide [`DistributedError`].
pub(super) type GlooResult<T> = Result<T, DistributedError>;
