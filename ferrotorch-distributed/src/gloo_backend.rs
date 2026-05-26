//! Gloo backend public surface (issue #1132).
//!
//! The original #459 skeleton was a fail-fast stub that returned
//! [`DistributedError::BackendUnavailable`] from every collective method
//! while deferring the real binding work to a hypothetical `gloo-sys`
//! C++ FFI crate. #1132 replaces that skeleton with a **native-Rust**
//! implementation: pure `std::net::TcpStream` transport, length-prefixed
//! framing, and textbook ring/tree collective algorithms. No `cc` crate,
//! no `bindgen`, no `libgloo` link.
//!
//! # Public surface
//!
//! - [`GlooBackend`] — the user-facing handle. Implements
//!   [`Backend`](crate::backend::Backend); construction goes through a
//!   PyTorch-compatible rendezvous (`MASTER_ADDR`, `MASTER_PORT`, `RANK`,
//!   `WORLD_SIZE`).
//! - [`is_gloo_available`] — returns `true` iff the build was compiled
//!   with the `gloo-backend` feature.
//!
//! # Feature gate
//!
//! Default off. Under `--features=gloo-backend`, the real native
//! implementation is compiled in. Without the feature, [`GlooBackend::new`]
//! and the env-var constructor still exist (so callers can write
//! `Backend::Gloo` paths) but return
//! [`DistributedError::BackendUnavailable`].
//!
//! The feature name retains the historical `gloo-backend` spelling (rather
//! than `gloo-native`) to avoid a breaking rename of #459's published
//! surface — see the `_surface_inventory.toml` entries and the
//! `is_gloo_available_matches_fixture` conformance test.
//!
//! ## REQ status (per `.design/ferrotorch-distributed/gloo_backend.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (is_gloo_available) | SHIPPED | `pub fn is_gloo_available` in `gloo_backend.rs`; consumer: re-export in `lib.rs` mirroring upstream `torch.distributed.is_gloo_available` in `torch/distributed/distributed_c10d.py`. |
//! | REQ-2 (GlooBackend struct) | SHIPPED | `pub struct GlooBackend` in `gloo_backend.rs`; consumer: `pub use gloo_backend::{GlooBackend, is_gloo_available}` in `lib.rs`. |
//! | REQ-3 (new constructor) | SHIPPED | `pub fn GlooBackend::new` in `gloo_backend.rs`; consumer: re-export in `lib.rs` plus feature-on rendezvous reuse from `mpi_backend.rs` / `ucc_backend.rs` via shared `GlooBackendInner::new`. |
//! | REQ-4 (from_env constructor) | SHIPPED | `pub fn GlooBackend::from_env` in `gloo_backend.rs`; consumer: re-export in `lib.rs`. |
//! | REQ-5 (Backend trait impl) | SHIPPED | `impl Backend for GlooBackend` in `gloo_backend.rs`; consumer: every `&dyn Backend`-accepting function in `collective.rs` and `p2p.rs`. |
//! | REQ-6 (ring/tree direct entry points) | SHIPPED | feature-gated `pub fn ring_allreduce_sum_f32` / `pub fn tree_broadcast_f32` in `gloo_backend.rs`; consumer: feature-on builds reach the inherent methods through the `lib.rs` re-export. |
//! | REQ-7 (BackendUnavailable discrimination) | SHIPPED | `BackendUnavailable { backend: "gloo" }` raised in `new` / `from_env` / `send` / `recv` / `recv_timeout` / `barrier` in `gloo_backend.rs`; consumer: `_surface.json` conformance fixture in `ferrotorch-core/tests/conformance/`. |

use std::time::Duration;

use ferrotorch_core::FerrotorchResult;

use crate::backend::Backend;
#[cfg(not(feature = "gloo-backend"))]
use crate::error::DistributedError;

#[cfg(feature = "gloo-backend")]
mod native {
    pub use crate::gloo_native::{GlooBackendInner, GlooRendezvousConfig};
}

/// Returns `true` when this build was compiled with the `gloo-backend`
/// feature enabled (which wires in the native-Rust TCP backend from #1132).
///
/// The same predicate covered the #459 fail-fast skeleton; #1132 keeps the
/// signature stable so downstream code that switches on this function does
/// not break.
pub fn is_gloo_available() -> bool {
    cfg!(feature = "gloo-backend")
}

/// Native-Rust Gloo backend handle.
///
/// Construction:
///
/// - [`GlooBackend::new`] — explicit rank / world-size / master-addr.
/// - [`GlooBackend::from_env`] — read `MASTER_ADDR` / `MASTER_PORT` /
///   `RANK` / `WORLD_SIZE` (PyTorch-compatible).
///
/// Without the `gloo-backend` cargo feature, every constructor returns
/// [`DistributedError::BackendUnavailable`]. The struct itself is still
/// present so `dyn Backend` type erasure paths compile against it.
#[derive(Debug)]
pub struct GlooBackend {
    /// `Some` on feature-enabled builds; `None` is unreachable (constructors
    /// reject before reaching here when the feature is off, and the field
    /// is the only inhabitant otherwise). The `Option` is required so the
    /// no-feature struct has a layout — `()` would also work but `Option`
    /// lets us share one impl block across both builds.
    #[cfg(feature = "gloo-backend")]
    inner: native::GlooBackendInner,
    #[cfg(not(feature = "gloo-backend"))]
    _phantom: std::marker::PhantomData<()>,
}

impl GlooBackend {
    /// Construct a Gloo backend with explicit parameters.
    ///
    /// * `rank` — this process's rank.
    /// * `world_size` — total number of ranks. Must be `>= 2`.
    /// * `master_addr` — `host:port` of rank 0's rendezvous listener
    ///   (matches PyTorch's `MASTER_ADDR:MASTER_PORT` convention).
    ///
    /// # Errors
    ///
    /// - [`DistributedError::BackendUnavailable`] without the
    ///   `gloo-backend` feature.
    /// - [`DistributedError::InvalidWorldSize`] / [`DistributedError::InvalidRank`]
    ///   on out-of-range inputs.
    /// - [`DistributedError::Io`] on rendezvous network failures.
    #[allow(unused_variables)] // Args are real when the feature is on.
    pub fn new(rank: usize, world_size: usize, master_addr: &str) -> FerrotorchResult<Self> {
        #[cfg(feature = "gloo-backend")]
        {
            use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
            let cfg = native::GlooRendezvousConfig {
                master_addr: master_addr.to_string(),
                rank,
                world_size,
                bind_addr: SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)),
            };
            let inner = native::GlooBackendInner::new(&cfg)?;
            Ok(Self { inner })
        }
        #[cfg(not(feature = "gloo-backend"))]
        {
            Err(DistributedError::BackendUnavailable { backend: "gloo" }.into())
        }
    }

    /// Construct a Gloo backend from PyTorch's standard env vars:
    /// `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`.
    ///
    /// # Errors
    ///
    /// See [`GlooBackend::new`]. Additionally returns
    /// [`DistributedError::Io`] if any required env var is missing or
    /// fails to parse as a `usize`.
    pub fn from_env() -> FerrotorchResult<Self> {
        #[cfg(feature = "gloo-backend")]
        {
            let cfg = native::GlooRendezvousConfig::from_env()?;
            let inner = native::GlooBackendInner::new(&cfg)?;
            Ok(Self { inner })
        }
        #[cfg(not(feature = "gloo-backend"))]
        {
            Err(DistributedError::BackendUnavailable { backend: "gloo" }.into())
        }
    }

    /// Ring-allreduce a contiguous `f32` slice **in place** with element-wise
    /// sum across all ranks. Available only with the `gloo-backend` feature.
    #[cfg(feature = "gloo-backend")]
    pub fn ring_allreduce_sum_f32(&self, data: &mut [f32]) -> FerrotorchResult<()> {
        self.inner.ring_allreduce_sum_f32(data)
    }

    /// Tree-broadcast a contiguous `f32` slice from `root`. Available only
    /// with the `gloo-backend` feature.
    #[cfg(feature = "gloo-backend")]
    pub fn tree_broadcast_f32(&self, data: &mut [f32], root: usize) -> FerrotorchResult<()> {
        self.inner.tree_broadcast_f32(data, root)
    }
}

impl Backend for GlooBackend {
    fn rank(&self) -> usize {
        #[cfg(feature = "gloo-backend")]
        {
            self.inner.rank()
        }
        #[cfg(not(feature = "gloo-backend"))]
        {
            // Unreachable in practice: construction always errors without
            // the feature, so no caller can hold a `GlooBackend` instance
            // here. A panic would be confusing — return 0 to keep the
            // surface total. This matches the #459 contract.
            0
        }
    }

    fn world_size(&self) -> usize {
        #[cfg(feature = "gloo-backend")]
        {
            Backend::world_size(&self.inner)
        }
        #[cfg(not(feature = "gloo-backend"))]
        {
            0
        }
    }

    #[allow(unused_variables)]
    fn send(&self, data: &[u8], dst_rank: usize) -> FerrotorchResult<()> {
        #[cfg(feature = "gloo-backend")]
        {
            self.inner.send(data, dst_rank)
        }
        #[cfg(not(feature = "gloo-backend"))]
        {
            Err(DistributedError::BackendUnavailable { backend: "gloo" }.into())
        }
    }

    #[allow(unused_variables)]
    fn recv(&self, dst: &mut [u8], src_rank: usize) -> FerrotorchResult<()> {
        #[cfg(feature = "gloo-backend")]
        {
            self.inner.recv(dst, src_rank)
        }
        #[cfg(not(feature = "gloo-backend"))]
        {
            Err(DistributedError::BackendUnavailable { backend: "gloo" }.into())
        }
    }

    #[allow(unused_variables)]
    fn recv_timeout(
        &self,
        dst: &mut [u8],
        src_rank: usize,
        timeout: Duration,
    ) -> FerrotorchResult<()> {
        #[cfg(feature = "gloo-backend")]
        {
            self.inner.recv_timeout(dst, src_rank, timeout)
        }
        #[cfg(not(feature = "gloo-backend"))]
        {
            Err(DistributedError::BackendUnavailable { backend: "gloo" }.into())
        }
    }

    fn barrier(&self) -> FerrotorchResult<()> {
        #[cfg(feature = "gloo-backend")]
        {
            self.inner.barrier()
        }
        #[cfg(not(feature = "gloo-backend"))]
        {
            Err(DistributedError::BackendUnavailable { backend: "gloo" }.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "gloo-backend"))]
    use ferrotorch_core::FerrotorchError;

    #[cfg(not(feature = "gloo-backend"))]
    #[test]
    fn gloo_unavailable_without_feature() {
        // Non-vacuous discrimination: when the `gloo-backend` feature is
        // off (the default), construction must fail with a
        // `DistributedError::BackendUnavailable { backend: "gloo" }`,
        // which converts to `FerrotorchError::InvalidArgument { message }`
        // whose `message` carries the backend name. This test preserves
        // the #459 contract validated by `is_gloo_available_matches_fixture`.
        // Feature-on path is exercised in `gloo_native::tests` instead.
        let err = GlooBackend::new(0, 2, "127.0.0.1:0").expect_err("default build must err");
        match err {
            FerrotorchError::InvalidArgument { ref message } => {
                assert!(
                    message.contains("`gloo`"),
                    "expected message to discriminate the gloo backend by name, got: {message}"
                );
                assert!(
                    !message.contains("`mpi`") && !message.contains("`ucc`"),
                    "message must not name a different backend, got: {message}"
                );
            }
            other => panic!(
                "expected FerrotorchError::InvalidArgument from BackendUnavailable, got {other:?}"
            ),
        }
    }

    #[cfg(not(feature = "gloo-backend"))]
    #[test]
    fn gloo_from_env_unavailable_without_feature() {
        let err = GlooBackend::from_env().expect_err("default build must err");
        match err {
            FerrotorchError::InvalidArgument { message } => {
                assert!(message.contains("`gloo`"));
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn is_gloo_available_default_off() {
        // The default workspace build does not enable `gloo-backend`, so
        // this returns false. Feature-enabled builds exercise the live
        // path via the native-module tests instead.
        if !cfg!(feature = "gloo-backend") {
            assert!(!is_gloo_available());
        }
    }
}
