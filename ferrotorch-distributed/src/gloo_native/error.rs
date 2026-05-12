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

use crate::error::DistributedError;

/// Internal `Result` alias used throughout `gloo_native::`. The error type
/// is the workspace-wide [`DistributedError`].
pub(super) type GlooResult<T> = Result<T, DistributedError>;
