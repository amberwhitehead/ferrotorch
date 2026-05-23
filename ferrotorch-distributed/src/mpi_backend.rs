//! Native-Rust MPI-subset backend (#1133, replaces closed #459).
//!
//! [MPI](https://www.mpi-forum.org/) (Message Passing Interface) is the
//! HPC-standard collective library, with implementations like Open MPI,
//! MPICH, and MVAPICH. PyTorch's `torch.distributed` ships an MPI
//! backend that's the most common choice on supercomputers and InfiniBand
//! clusters.
//!
//! # Status
//!
//! Under `--features=mpi-native` (or its alias `mpi-backend`), this module
//! ships a **native-Rust** MPI-subset backend that delegates to the
//! [`crate::gloo_native`] primitives. The MPI standard surface PyTorch
//! actually drives through `init_process_group(backend="mpi")` is a small
//! subset:
//!
//! - `MPI_Allreduce` → [`crate::gloo_native::GlooBackendInner::ring_allreduce_sum_f32`]
//! - `MPI_Bcast` → [`crate::gloo_native::GlooBackendInner::tree_broadcast_f32`]
//! - `MPI_Barrier` → ring barrier in [`crate::gloo_native::collectives`]
//! - `MPI_Send` / `MPI_Recv` → [`crate::backend::Backend::send`] /
//!   [`crate::backend::Backend::recv`] over the gloo full-mesh
//!
//! These map 1:1 to the gloo_native operations, so the MPI backend is a
//! thin wrapper around `GlooBackendInner` that ships an MPI-style
//! rendezvous (env-var conventions match the typical `mpirun` /
//! `mpiexec` launchers).
//!
//! # Why native (no C FFI)
//!
//! The original #459 plan was to bind a C MPI implementation via the
//! `mpi` Rust crate, which would require `mpicc` at compile time and a
//! working `libmpi.so` at runtime — substantially heavier than a typical
//! cargo-test setup. The #1133 closure-task user directive replaces that
//! plan with a pure-Rust implementation reusing the gloo_native TCP
//! transport. No `mpi-sys`, no `mpicc`, no `libmpi.so` link — the entire
//! collective stack is portable Rust.
//!
//! # Feature gate
//!
//! Default off. Under `--features=mpi-native`, the real native
//! implementation is compiled in. Without the feature, [`MpiBackend::new`]
//! and the env-var constructor still exist (so callers can write
//! `Backend::Mpi` paths) but return
//! [`DistributedError::BackendUnavailable`].
//!
//! The historical `mpi-backend` feature name is retained as an alias for
//! `mpi-native` so downstream code keying off either spelling resolves to
//! the same code path. Both [`is_mpi_available`] and the
//! `is_mpi_available` conformance fixture continue to be discriminated by
//! the same `cfg!(any(feature = "mpi-native", feature = "mpi-backend"))`
//! predicate — they collapse to the same `cfg` because the feature graph
//! forces both flags on together.

use std::time::Duration;

use ferrotorch_core::FerrotorchResult;

use crate::backend::Backend;
#[cfg(not(feature = "mpi-native"))]
use crate::error::DistributedError;

#[cfg(feature = "mpi-native")]
mod native {
    //! Real implementation surface for the native MPI-subset backend.
    //!
    //! `MpiBackend` delegates every collective and point-to-point call to
    //! [`crate::gloo_native::GlooBackendInner`]; the only MPI-specific
    //! code path is the env-var rendezvous, which accepts both the
    //! PyTorch-style `MASTER_ADDR` / `RANK` / `WORLD_SIZE` convention and
    //! the typical MPI launcher conventions (`OMPI_COMM_WORLD_RANK` /
    //! `OMPI_COMM_WORLD_SIZE` for Open MPI, `PMI_RANK` / `PMI_SIZE` for
    //! MPICH / Hydra).

    use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};

    use crate::error::DistributedError;
    use crate::gloo_native::GlooRendezvousConfig;

    /// Build a rendezvous config from MPI-launcher env vars.
    ///
    /// Resolution order, first hit wins:
    ///
    /// 1. `OMPI_COMM_WORLD_RANK` + `OMPI_COMM_WORLD_SIZE` (Open MPI's
    ///    `mpirun`).
    /// 2. `PMI_RANK` + `PMI_SIZE` (MPICH / Hydra / SLURM `srun` with
    ///    PMI-1 / PMI-2 wire-up).
    /// 3. `RANK` + `WORLD_SIZE` (PyTorch / torchrun fallback — also what
    ///    [`crate::gloo_native::GlooRendezvousConfig::from_env`] reads).
    ///
    /// `MASTER_ADDR` and `MASTER_PORT` are always read from those names;
    /// the MPI launcher conventions do not standardise a rendezvous
    /// endpoint (MPI does its own out-of-band rendezvous over the
    /// launcher's wire protocol), so we ride PyTorch's convention here
    /// for the native TCP rendezvous that backs the gloo_native
    /// primitives.
    pub(super) fn mpi_rendezvous_from_env() -> Result<GlooRendezvousConfig, DistributedError> {
        fn try_pair(rank_key: &str, size_key: &str) -> Option<(usize, usize)> {
            let rank = std::env::var(rank_key).ok()?.parse::<usize>().ok()?;
            let size = std::env::var(size_key).ok()?.parse::<usize>().ok()?;
            Some((rank, size))
        }

        // First, decide which (rank, world_size) pair to use. The MPI-style
        // pairs win over the PyTorch fallback so a job launched under
        // `mpirun` doesn't silently misread a leftover `RANK` env var.
        let (rank, world_size) = try_pair("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE")
            .or_else(|| try_pair("PMI_RANK", "PMI_SIZE"))
            .or_else(|| try_pair("RANK", "WORLD_SIZE"))
            .ok_or_else(|| DistributedError::Io {
                message: "mpi_backend rendezvous: none of (OMPI_COMM_WORLD_RANK + \
                          OMPI_COMM_WORLD_SIZE), (PMI_RANK + PMI_SIZE), (RANK + WORLD_SIZE) \
                          are set in the environment"
                    .to_string(),
            })?;

        // MPI launchers do not standardise a rendezvous endpoint, so we
        // reuse PyTorch's `MASTER_ADDR` / `MASTER_PORT` for the native
        // TCP rendezvous handshake.
        let master_host = std::env::var("MASTER_ADDR").map_err(|_| DistributedError::Io {
            message: "mpi_backend rendezvous: env var `MASTER_ADDR` is not set".to_string(),
        })?;
        let master_port = std::env::var("MASTER_PORT").map_err(|_| DistributedError::Io {
            message: "mpi_backend rendezvous: env var `MASTER_PORT` is not set".to_string(),
        })?;

        Ok(GlooRendezvousConfig {
            master_addr: format!("{master_host}:{master_port}"),
            rank,
            world_size,
            bind_addr: SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)),
        })
    }
}

/// Returns `true` when this build was compiled with the `mpi-native`
/// feature enabled (which wires in the native-Rust TCP backend from
/// #1133, delegating to the gloo_native primitives).
///
/// The same predicate covered the #459 fail-fast skeleton; #1133 keeps
/// the signature stable so downstream code that switches on this function
/// does not break. The legacy `mpi-backend` feature name is an alias for
/// `mpi-native` (see `Cargo.toml`) so either spelling toggles the same
/// code path.
pub fn is_mpi_available() -> bool {
    cfg!(feature = "mpi-native")
}

/// Native-Rust MPI-subset backend handle.
///
/// Construction:
///
/// - [`MpiBackend::new`] — explicit rank / world-size / master-addr.
/// - [`MpiBackend::from_env`] — read MPI-launcher env vars
///   (`OMPI_COMM_WORLD_RANK` / `OMPI_COMM_WORLD_SIZE` for Open MPI,
///   `PMI_RANK` / `PMI_SIZE` for MPICH, falling back to PyTorch's
///   `RANK` / `WORLD_SIZE`), plus `MASTER_ADDR` / `MASTER_PORT` for the
///   rendezvous endpoint.
///
/// Without the `mpi-native` cargo feature, every constructor returns
/// [`DistributedError::BackendUnavailable`]. The struct itself is still
/// present so `dyn Backend` type-erasure paths compile against it.
#[derive(Debug)]
pub struct MpiBackend {
    /// `Some` on feature-enabled builds; `None` is unreachable
    /// (constructors reject before reaching here when the feature is
    /// off, and the field is the only inhabitant otherwise). Mirrors the
    /// `GlooBackend` shape so both backends present the same external
    /// API contract on feature-off builds.
    #[cfg(feature = "mpi-native")]
    inner: crate::gloo_native::GlooBackendInner,
    #[cfg(not(feature = "mpi-native"))]
    _phantom: std::marker::PhantomData<()>,
}

impl MpiBackend {
    /// Construct an MPI backend with explicit parameters.
    ///
    /// * `rank` — this process's MPI rank (`MPI_Comm_rank` on
    ///   `MPI_COMM_WORLD`).
    /// * `world_size` — total number of ranks (`MPI_Comm_size`). Must be
    ///   `>= 2`.
    /// * `master_addr` — `host:port` of rank 0's rendezvous listener.
    ///
    /// # Errors
    ///
    /// - [`DistributedError::BackendUnavailable`] without the
    ///   `mpi-native` feature.
    /// - [`DistributedError::InvalidWorldSize`] / [`DistributedError::InvalidRank`]
    ///   on out-of-range inputs.
    /// - [`DistributedError::Io`] on rendezvous network failures.
    #[allow(unused_variables)] // Args are real when the feature is on.
    pub fn new(rank: usize, world_size: usize, master_addr: &str) -> FerrotorchResult<Self> {
        #[cfg(feature = "mpi-native")]
        {
            use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
            let cfg = crate::gloo_native::GlooRendezvousConfig {
                master_addr: master_addr.to_string(),
                rank,
                world_size,
                bind_addr: SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)),
            };
            let inner = crate::gloo_native::GlooBackendInner::new(&cfg)?;
            Ok(Self { inner })
        }
        #[cfg(not(feature = "mpi-native"))]
        {
            Err(DistributedError::BackendUnavailable { backend: "mpi" }.into())
        }
    }

    /// Construct an MPI backend from MPI-launcher env vars.
    ///
    /// Resolution order, first hit wins:
    /// 1. `OMPI_COMM_WORLD_RANK` + `OMPI_COMM_WORLD_SIZE` (Open MPI).
    /// 2. `PMI_RANK` + `PMI_SIZE` (MPICH / Hydra / SLURM PMI).
    /// 3. `RANK` + `WORLD_SIZE` (PyTorch / torchrun fallback).
    ///
    /// `MASTER_ADDR` and `MASTER_PORT` are always read from those names
    /// for the rendezvous endpoint.
    ///
    /// # Errors
    ///
    /// See [`MpiBackend::new`]. Additionally returns
    /// [`DistributedError::Io`] if none of the recognised rank /
    /// world-size pairs (or `MASTER_ADDR` / `MASTER_PORT`) are set.
    pub fn from_env() -> FerrotorchResult<Self> {
        #[cfg(feature = "mpi-native")]
        {
            let cfg = native::mpi_rendezvous_from_env()?;
            let inner = crate::gloo_native::GlooBackendInner::new(&cfg)?;
            Ok(Self { inner })
        }
        #[cfg(not(feature = "mpi-native"))]
        {
            Err(DistributedError::BackendUnavailable { backend: "mpi" }.into())
        }
    }

    /// MPI_Allreduce-style ring allreduce over a contiguous `f32` slice
    /// **in place** with element-wise sum across all ranks. Available
    /// only with the `mpi-native` feature.
    #[cfg(feature = "mpi-native")]
    pub fn allreduce_sum_f32(&self, data: &mut [f32]) -> FerrotorchResult<()> {
        self.inner.ring_allreduce_sum_f32(data)
    }

    /// MPI_Bcast-style tree broadcast from `root`. Available only with
    /// the `mpi-native` feature.
    #[cfg(feature = "mpi-native")]
    pub fn broadcast_f32(&self, data: &mut [f32], root: usize) -> FerrotorchResult<()> {
        self.inner.tree_broadcast_f32(data, root)
    }
}

impl Backend for MpiBackend {
    fn rank(&self) -> usize {
        #[cfg(feature = "mpi-native")]
        {
            self.inner.rank()
        }
        #[cfg(not(feature = "mpi-native"))]
        {
            // Unreachable in practice: construction always errors without
            // the feature, so no caller can hold an `MpiBackend` instance
            // here. A panic would be confusing — return 0 to keep the
            // surface total. Matches the `GlooBackend` shape.
            0
        }
    }

    fn world_size(&self) -> usize {
        #[cfg(feature = "mpi-native")]
        {
            Backend::world_size(&self.inner)
        }
        #[cfg(not(feature = "mpi-native"))]
        {
            0
        }
    }

    #[allow(unused_variables)]
    fn send(&self, data: &[u8], dst_rank: usize) -> FerrotorchResult<()> {
        #[cfg(feature = "mpi-native")]
        {
            self.inner.send(data, dst_rank)
        }
        #[cfg(not(feature = "mpi-native"))]
        {
            Err(DistributedError::BackendUnavailable { backend: "mpi" }.into())
        }
    }

    #[allow(unused_variables)]
    fn recv(&self, dst: &mut [u8], src_rank: usize) -> FerrotorchResult<()> {
        #[cfg(feature = "mpi-native")]
        {
            self.inner.recv(dst, src_rank)
        }
        #[cfg(not(feature = "mpi-native"))]
        {
            Err(DistributedError::BackendUnavailable { backend: "mpi" }.into())
        }
    }

    #[allow(unused_variables)]
    fn recv_timeout(
        &self,
        dst: &mut [u8],
        src_rank: usize,
        timeout: Duration,
    ) -> FerrotorchResult<()> {
        #[cfg(feature = "mpi-native")]
        {
            self.inner.recv_timeout(dst, src_rank, timeout)
        }
        #[cfg(not(feature = "mpi-native"))]
        {
            Err(DistributedError::BackendUnavailable { backend: "mpi" }.into())
        }
    }

    fn barrier(&self) -> FerrotorchResult<()> {
        #[cfg(feature = "mpi-native")]
        {
            self.inner.barrier()
        }
        #[cfg(not(feature = "mpi-native"))]
        {
            Err(DistributedError::BackendUnavailable { backend: "mpi" }.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "mpi-native"))]
    use ferrotorch_core::FerrotorchError;

    #[cfg(not(feature = "mpi-native"))]
    #[test]
    fn mpi_unavailable_without_feature() {
        // Non-vacuous discrimination: when the `mpi-native` feature is
        // off (the default), construction must fail with a
        // `DistributedError::BackendUnavailable { backend: "mpi" }`,
        // which converts to `FerrotorchError::InvalidArgument { message }`
        // whose `message` carries the backend name. Preserves the #459
        // contract validated by `is_mpi_available_matches_fixture`.
        // Feature-on path is exercised in the `mpi_native_e2e` test
        // below.
        let err = MpiBackend::new(0, 2, "127.0.0.1:0").expect_err("default build must err");
        match err {
            FerrotorchError::InvalidArgument { ref message } => {
                assert!(
                    message.contains("`mpi`"),
                    "expected message to discriminate the mpi backend by name, got: {message}"
                );
                assert!(
                    !message.contains("`gloo`") && !message.contains("`ucc`"),
                    "message must not name a different backend, got: {message}"
                );
            }
            other => panic!(
                "expected FerrotorchError::InvalidArgument from BackendUnavailable, got {other:?}"
            ),
        }
    }

    #[cfg(not(feature = "mpi-native"))]
    #[test]
    fn mpi_from_env_unavailable_without_feature() {
        let err = MpiBackend::from_env().expect_err("default build must err");
        match err {
            FerrotorchError::InvalidArgument { message } => {
                assert!(message.contains("`mpi`"));
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn is_mpi_available_default_off() {
        // The default workspace build does not enable `mpi-native`, so
        // this returns false. Feature-enabled builds exercise the live
        // path via the `mpi_native_*` tests below instead.
        if !cfg!(feature = "mpi-native") {
            assert!(!is_mpi_available());
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // Feature-on path: end-to-end test that the MPI backend really
    // delegates collective operations to gloo_native primitives. Mirrors
    // the gloo_native tests' in-process spawn pattern: one thread per
    // rank, kernel-assigned `MASTER_PORT`, real TCP rendezvous, real
    // collective wire traffic.
    // ──────────────────────────────────────────────────────────────────

    #[cfg(feature = "mpi-native")]
    #[test]
    fn mpi_native_e2e_allreduce_two_ranks() {
        use std::net::TcpListener;
        use std::sync::Arc;
        use std::thread;

        // Pick an ephemeral port for the rendezvous master.
        let probe = TcpListener::bind("127.0.0.1:0").expect("probe bind");
        let master_addr = probe.local_addr().expect("local_addr").to_string();
        drop(probe);

        // Spawn two MPI ranks in-process.
        let world_size = 2usize;
        let handles: Vec<_> = (0..world_size)
            .map(|rank| {
                let ma = master_addr.clone();
                thread::spawn(move || {
                    Arc::new(MpiBackend::new(rank, world_size, &ma).expect("MpiBackend::new"))
                })
            })
            .collect();
        let backends: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().expect("join"))
            .collect();

        // Drive an allreduce: rank 0 has [1, 2, 3, 4]; rank 1 has
        // [10, 20, 30, 40]; expected sum [11, 22, 33, 44] on both.
        thread::scope(|s| {
            let b0 = Arc::clone(&backends[0]);
            let b1 = Arc::clone(&backends[1]);
            let h0 = s.spawn(move || {
                let mut a = vec![1.0f32, 2.0, 3.0, 4.0];
                b0.allreduce_sum_f32(&mut a).expect("allreduce rank 0");
                a
            });
            let h1 = s.spawn(move || {
                let mut a = vec![10.0f32, 20.0, 30.0, 40.0];
                b1.allreduce_sum_f32(&mut a).expect("allreduce rank 1");
                a
            });
            let r0 = h0.join().unwrap();
            let r1 = h1.join().unwrap();
            let expected = vec![11.0f32, 22.0, 33.0, 44.0];
            assert_eq!(r0, expected, "rank 0 allreduce result");
            assert_eq!(r1, expected, "rank 1 allreduce result");
        });
    }

    #[cfg(feature = "mpi-native")]
    #[test]
    fn mpi_native_e2e_broadcast_and_barrier_three_ranks() {
        use std::net::TcpListener;
        use std::sync::Arc;
        use std::thread;

        let probe = TcpListener::bind("127.0.0.1:0").expect("probe bind");
        let master_addr = probe.local_addr().expect("local_addr").to_string();
        drop(probe);

        let world_size = 3usize;
        let handles: Vec<_> = (0..world_size)
            .map(|rank| {
                let ma = master_addr.clone();
                thread::spawn(move || {
                    Arc::new(MpiBackend::new(rank, world_size, &ma).expect("MpiBackend::new"))
                })
            })
            .collect();
        let backends: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().expect("join"))
            .collect();

        // Broadcast from rank 1 to ranks 0 and 2.
        let payload = vec![7.5f32, 8.25, 9.125];
        let root = 1usize;
        thread::scope(|s| {
            let mut handles = Vec::new();
            for (rank, backend) in backends.iter().enumerate() {
                let b = Arc::clone(backend);
                let p = payload.clone();
                handles.push(s.spawn(move || {
                    let mut data = if rank == root { p } else { vec![0.0f32; 3] };
                    b.broadcast_f32(&mut data, root).expect("broadcast");
                    // Then a barrier — exercises the ring-barrier path.
                    Backend::barrier(&*b).expect("barrier");
                    data
                }));
            }
            for h in handles {
                let got = h.join().unwrap();
                assert_eq!(got, vec![7.5f32, 8.25, 9.125]);
            }
        });
    }

    #[cfg(feature = "mpi-native")]
    #[test]
    fn mpi_native_from_env_prefers_ompi_then_pmi_then_pytorch() {
        // Validates the resolution order in `mpi_rendezvous_from_env`
        // without actually running the rendezvous (we only check that
        // the parsed (rank, world_size) match the source pair). Env
        // mutation is shared across tests in the same process, so we
        // scope it to a single test and clean up at the end. We avoid
        // running this in parallel with the e2e tests above by relying
        // on each call to `from_env` parsing exactly once before we
        // mutate again.
        //
        // SAFETY: env mutation is process-wide; we set / unset within
        // this test scope. The other `mpi_native_*` tests do not touch
        // these vars, so even if cargo schedules them concurrently in
        // the same process they cannot observe a partial state.
        unsafe {
            std::env::remove_var("OMPI_COMM_WORLD_RANK");
            std::env::remove_var("OMPI_COMM_WORLD_SIZE");
            std::env::remove_var("PMI_RANK");
            std::env::remove_var("PMI_SIZE");
            std::env::remove_var("RANK");
            std::env::remove_var("WORLD_SIZE");

            // Set MASTER_ADDR / MASTER_PORT once for the duration.
            std::env::set_var("MASTER_ADDR", "127.0.0.1");
            std::env::set_var("MASTER_PORT", "29555");

            // Case 1: PyTorch fallback (lowest priority).
            std::env::set_var("RANK", "3");
            std::env::set_var("WORLD_SIZE", "4");
            let cfg = native::mpi_rendezvous_from_env().expect("pytorch fallback");
            assert_eq!(cfg.rank, 3, "pytorch fallback rank");
            assert_eq!(cfg.world_size, 4, "pytorch fallback world_size");
            assert_eq!(cfg.master_addr, "127.0.0.1:29555");

            // Case 2: PMI takes priority over the PyTorch fallback.
            std::env::set_var("PMI_RANK", "5");
            std::env::set_var("PMI_SIZE", "8");
            let cfg = native::mpi_rendezvous_from_env().expect("pmi over pytorch");
            assert_eq!(cfg.rank, 5, "pmi rank wins over RANK");
            assert_eq!(cfg.world_size, 8, "pmi size wins over WORLD_SIZE");

            // Case 3: Open MPI takes priority over PMI.
            std::env::set_var("OMPI_COMM_WORLD_RANK", "7");
            std::env::set_var("OMPI_COMM_WORLD_SIZE", "16");
            let cfg = native::mpi_rendezvous_from_env().expect("ompi over pmi");
            assert_eq!(cfg.rank, 7, "ompi rank wins over pmi");
            assert_eq!(cfg.world_size, 16, "ompi size wins over pmi");

            // Cleanup.
            std::env::remove_var("OMPI_COMM_WORLD_RANK");
            std::env::remove_var("OMPI_COMM_WORLD_SIZE");
            std::env::remove_var("PMI_RANK");
            std::env::remove_var("PMI_SIZE");
            std::env::remove_var("RANK");
            std::env::remove_var("WORLD_SIZE");
            std::env::remove_var("MASTER_ADDR");
            std::env::remove_var("MASTER_PORT");
        }
    }
}
