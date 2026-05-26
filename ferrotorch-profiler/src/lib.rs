#![warn(clippy::all, clippy::pedantic)]
#![warn(missing_debug_implementations, rust_2018_idioms)]
#![deny(unsafe_code)]
#![allow(clippy::module_name_repetitions)] // ProfileConfig, ProfileEvent, etc. are intentional
//! Operation profiling for ferrotorch.
//!
//! Provides [`Profiler`] for recording operation timings, memory events, and
//! input shapes during a forward/backward pass.  The resulting [`ProfileReport`]
//! can be rendered as a human-readable table or exported to Chrome trace JSON
//! (`chrome://tracing`).
//!
//! # Quick start
//!
//! ```rust
//! use ferrotorch_profiler::{with_profiler, ProfileConfig};
//!
//! let config = ProfileConfig::default();
//! let (result, report) = with_profiler(config, |profiler| {
//!     profiler.record("matmul", "tensor_op", &[&[32, 784], &[784, 256]]);
//!     profiler.record("relu", "tensor_op", &[&[32, 256]]);
//!     42
//! });
//!
//! println!("{}", report.table(10));
//! ```
//!
//! ## REQ status (per `.design/ferrotorch-profiler/lib.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | lint pin at the top of `lib.rs` with `clippy::module_name_repetitions` allowance carrying inline rationale; consumer: workspace `cargo clippy -p ferrotorch-profiler --lib -- -D warnings` is clean against this pin. |
//! | REQ-2 | SHIPPED | submodule decls in `lib.rs` (`pub mod cuda_timing` cuda-gated, `mod event`, `pub mod flops`, `mod profiler`, `mod report`, `pub mod schedule`); consumer: `ferrotorch-profiler/src/profiler.rs:5-7` imports `crate::event`, `crate::flops`, `crate::report`; `cuda_timing.rs:46` imports `crate::profiler::Profiler` under the cuda gate. |
//! | REQ-3 | SHIPPED | `pub use` block in `lib.rs` re-exporting `CudaKernelScope`, `DeviceType`, `GpuTimingPair`, `MemoryCategory`, `ProfileEvent`, `ProfileConfig`, `Profiler`, `with_profiler`, `OpSummary`, `ProfileReport`, `ProfileSchedule`, `SchedulePhase`; consumer: `ferrotorch/src/lib.rs:107` `pub use ferrotorch_profiler::*;` propagates the surface to the meta-crate prelude. |
//! | REQ-4 | SHIPPED | quick-start doctest in this very doc-comment exercising `with_profiler` + `ProfileConfig::default` + `profiler.record` + `report.table(10)`, mirroring the example pattern at `torch/profiler/profiler.py:651-712`; consumer: `cargo test --doc` runs it as part of CI. |
//! | REQ-5 | SHIPPED | `pub(crate) struct PendingCudaScope` in `cuda_timing.rs` is intentionally absent from the `pub use` block in `lib.rs`; consumer: `ferrotorch-profiler/tests/conformance_surface_coverage.rs:66-` pins every re-exported symbol and would fail if `PendingCudaScope` were added or any re-export removed. |

#[cfg(feature = "cuda")]
pub mod cuda_timing;
mod event;
pub mod flops;
mod profiler;
mod report;
pub mod schedule;

// `CudaKernelScope` is the public API for users who want to time a GPU kernel
// region. The crate's other CUDA scope type (`PendingCudaScope`, the
// queue-internal end-of-region marker) is `pub(crate)` and not re-exported.
#[cfg(feature = "cuda")]
pub use cuda_timing::CudaKernelScope;
pub use event::{DeviceType, GpuTimingPair, MemoryCategory, ProfileEvent};
pub use profiler::{ProfileConfig, Profiler, with_profiler};
pub use report::{OpSummary, ProfileReport};
pub use schedule::{ProfileSchedule, SchedulePhase};
