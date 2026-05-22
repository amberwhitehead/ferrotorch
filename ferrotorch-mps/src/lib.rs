//! Apple Silicon Metal Performance Shaders (MPS) backend for ferrotorch. (#451, #626)
//!
//! # Sprint C.7 — what landed
//!
//! [`MtlBackend`] is now a real [`ferrotorch_core::gpu_dispatch::GpuBackend`]
//! implementation backed by 10 MSL kernels compiled at runtime via
//! `objc2-metal`. The 10 kernels cover the highest-priority subset of the
//! ~80-method `GpuBackend` trait:
//!
//! | Kernel | `GpuBackend` method(s) |
//! |---|---|
//! | `matmul_f32` | `matmul_f32` |
//! | `bmm_f32` | `bmm_f32` |
//! | `add_f32` | `add_f32` |
//! | `sub_f32` | `sub_f32` |
//! | `mul_f32` | `mul_f32` |
//! | `div_f32` | `div_f32` |
//! | `relu_f32` | `relu_f32` |
//! | `sigmoid_f32` | `sigmoid_f32` |
//! | `softmax_f32` | `softmax_f32` |
//! | `sum_axis_f32` | `sum_axis_f32`, `sum_f32` |
//!
//! All MSL sources live in `src/kernels/` as embedded string constants
//! (compiled into the binary via `include_str!`). `MtlBackend::new()`
//! compiles them eagerly at startup and caches the pipeline states.
//!
//! The remaining ~70 `GpuBackend` methods return
//! `Err(FerrotorchError::InvalidArgument { message: "MSL kernel needed: …" })`
//! — no silent CPU fallback (§3 of `rust-gpu-discipline`). Each unimplemented
//! method is tracked by an individual follow-up crosslink issue.
//!
//! # Platform gating
//!
//! `MtlBackend` and `backend` module are `#[cfg(target_os = "macos")]`.
//! The lifecycle items (`is_mps_available`, `mps_device_count`, `MpsDevice`,
//! `init_mps_backend`) compile on every platform and return
//! `DeviceUnavailable` / `false` / `0` on non-Apple hosts — matching the
//! pre-C.7 contract so existing callers are unaffected.
//!
//! # Initialization
//!
//! ```no_run
//! // On macOS, call once at startup:
//! ferrotorch_mps::init_mps_backend().expect("MPS backend init");
//! ```
//!
//! On non-macOS, `init_mps_backend()` returns
//! [`FerrotorchError::DeviceUnavailable`] (unchanged from pre-C.7).
//!
//! # Tests
//!
//! Tests that require a live Metal device use the `cascade_skip!` pattern:
//! they print a diagnostic and return early rather than failing or being
//! `#[ignore]`-marked. On Apple Silicon CI they run the full kernel path.
//!
//! # Follow-up tracking
//!
//! Issue #626 is the parent. Each of the remaining ~70 unimplemented
//! `GpuBackend` methods has its own crosslink follow-up issue filed as part
//! of Sprint C.7 completion.

#![warn(clippy::all, clippy::pedantic)]
#![deny(rust_2018_idioms, missing_debug_implementations)]
// unsafe_code: the backend module uses objc2-metal which requires unsafe blocks
// for Metal API calls. All unsafe sites have SAFETY comments.
#![cfg_attr(not(target_os = "macos"), deny(unsafe_code))]
// Pedantic lints we explicitly accept across this crate. Each allow names a
// concrete reason — the alternative would be churn-for-zero-benefit or a
// worse API. Add to this list only with a one-line justification.
#![allow(
    // The crate's name is `ferrotorch-mps` and its types naturally repeat the
    // `Mps` token (`MpsDevice`, `mps_device_count`, `init_mps_backend`); the
    // repetition is the disambiguator that prevents glob-import collisions
    // with sibling backends like `ferrotorch-gpu`.
    clippy::module_name_repetitions,
    // Tensor shape components (rows, cols, batch, m, k, n) are bounded by the
    // kernel-launch contract — MSL setBytes accepts u32, and a tensor dim
    // overflowing u32 (>4G) is rejected upstream. Truncation is impossible
    // in practice and `u32::try_from(...).unwrap()` adds noise without value.
    clippy::cast_possible_truncation,
    // Metal's `setBytes_length_atIndex` takes `NonNull<c_void>` to a small
    // scalar that lives on the stack; the `&n_u32 as *const u32 as *mut _`
    // pattern is the standard way to spell this in objc2-metal.
    clippy::ref_as_ptr,
    clippy::borrow_as_ptr,
    // MSL kernel dispatchers borrow matrix dims with the canonical math names
    // (m, k, n, a, b). These are PyTorch / BLAS convention, not Rust style.
    clippy::similar_names,
    clippy::many_single_char_names,
    // The `Pipelines` struct holds compiled kernels grouped by dtype; the
    // `_f32` postfix is intentional and disambiguates from future `_bf16` /
    // `_f16` siblings (#19).
    clippy::struct_field_names,
)]
#![deny(missing_docs)]

use core::fmt;

use ferrotorch_core::{FerrotorchError, FerrotorchResult};

/// MSL kernel source constants. One `.metal` file per logical group.
/// Embedded via `include_str!` so the MSL ships inside the Rust binary
/// and is compiled at runtime by `MtlBackend::new()`.
pub mod kernels;

/// Apple Metal backend — [`MtlBackend`] + kernel dispatch.
///
/// Compiled only on macOS (`#[cfg(target_os = "macos")]`); absent on all
/// other platforms so the workspace build stays clean on Linux/WSL.
#[cfg(target_os = "macos")]
pub mod backend;

/// Apple Metal backend implementation of [`ferrotorch_core::gpu_dispatch::GpuBackend`].
///
/// Holds a Metal device, a command queue, and compiled pipeline states for
/// all 10 Sprint C.7 MSL kernels. Construct via [`MtlBackend::new()`] or
/// call [`init_mps_backend()`] which registers it globally.
#[cfg(target_os = "macos")]
pub use backend::MtlBackend;

/// Returns `true` if this build can run MPS kernels on the current host.
///
/// On macOS: delegates to `MTLCreateSystemDefaultDevice` — returns `true`
/// when a Metal device is present (Apple Silicon or Intel Mac with AMD/Intel
/// GPU).
///
/// On all other platforms: always returns `false`. The Metal API does not
/// exist outside macOS so there is no platform-conditional lie here.
#[must_use]
pub fn is_mps_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        // MTLCreateSystemDefaultDevice returns None when no Metal device is
        // available; we only test for presence, never dereference the
        // returned pointer beyond the Option check.
        objc2_metal::MTLCreateSystemDefaultDevice().is_some()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// An opaque handle for an Apple-Silicon Metal device.
///
/// `MpsDevice` is `Copy` because it wraps a single `usize`. On macOS hosts
/// with a Metal device present, [`MpsDevice::new(0)`](Self::new) returns
/// `Ok`; on every other platform construction fails with
/// [`FerrotorchError::DeviceUnavailable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MpsDevice {
    ordinal: usize,
}

impl MpsDevice {
    /// Try to construct a device handle for the given ordinal.
    ///
    /// On macOS, returns `Ok` for `ordinal == 0` when [`is_mps_available`]
    /// reports a Metal device present, [`FerrotorchError::InvalidArgument`]
    /// for any non-zero ordinal (Apple Silicon exposes a single integrated
    /// GPU; ordinals > 0 have no PyTorch-faithful meaning), and
    /// [`FerrotorchError::DeviceUnavailable`] when no Metal device is found.
    ///
    /// On every other platform, always returns
    /// [`FerrotorchError::DeviceUnavailable`] — Metal does not exist outside
    /// macOS.
    ///
    /// # Errors
    ///
    /// - [`FerrotorchError::DeviceUnavailable`]: non-macOS platform, or no
    ///   Metal device found on macOS.
    /// - [`FerrotorchError::InvalidArgument`]: macOS-only, when `ordinal != 0`
    ///   on a system that does have Metal — Apple Silicon is single-device.
    pub fn new(ordinal: usize) -> FerrotorchResult<Self> {
        #[cfg(target_os = "macos")]
        {
            if !is_mps_available() {
                return Err(FerrotorchError::DeviceUnavailable);
            }
            if ordinal != 0 {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "Apple Silicon exposes a single integrated GPU; \
                         MpsDevice ordinal {ordinal} is unsupported (only 0 is valid)"
                    ),
                });
            }
            Ok(Self { ordinal: 0 })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = ordinal;
            Err(FerrotorchError::DeviceUnavailable)
        }
    }

    /// Number of MPS devices the system reports.
    ///
    /// Delegates to the free [`mps_device_count`] for a single source of
    /// truth. Provided as an associated function in addition to the free
    /// function for callers that prefer the type-anchored spelling.
    #[must_use]
    pub fn count() -> usize {
        mps_device_count()
    }

    /// Device ordinal (0 = system default GPU).
    #[must_use]
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
}

impl fmt::Display for MpsDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mps:{}", self.ordinal)
    }
}

/// Number of MPS devices the system reports.
///
/// Renamed from `device_count` to `mps_device_count` to avoid colliding
/// with `ferrotorch_gpu::device_count` when both backends are re-exported
/// via the `ferrotorch::{gpu, mps}` namespaces. Mirrors `PyTorch`'s
/// module-scoped `torch.cuda.device_count()` / `torch.mps.device_count()`
/// idiom rather than the type-anchored Rust idiom.
///
/// On macOS: returns `1` when a Metal device is present (Apple Silicon
/// exposes a single integrated GPU), `0` otherwise.
///
/// On every other platform: always returns `0`.
#[must_use]
pub fn mps_device_count() -> usize {
    #[cfg(target_os = "macos")]
    {
        usize::from(is_mps_available())
    }
    #[cfg(not(target_os = "macos"))]
    {
        0
    }
}

/// Initialize the MPS Metal backend and register it with `ferrotorch-core`.
///
/// On macOS: compiles all 10 MSL kernels eagerly, constructs an
/// [`MtlBackend`], and registers it via
/// [`ferrotorch_core::gpu_dispatch::register_gpu_backend`]. After this
/// call succeeds, `ferrotorch_core::gpu_dispatch::gpu_backend()` returns
/// `Some(...)` and all 10 Sprint C.7 ops dispatch to Metal.
///
/// On all other platforms: returns [`FerrotorchError::DeviceUnavailable`]
/// immediately — Metal does not exist outside macOS.
///
/// # Errors
///
/// - [`FerrotorchError::DeviceUnavailable`]: no Metal device found, or
///   called on a non-macOS platform.
/// - [`FerrotorchError::InvalidArgument`]: MSL compilation failed (indicates
///   a ferrotorch bug, not a user error) or backend already registered.
pub fn init_mps_backend() -> FerrotorchResult<()> {
    #[cfg(target_os = "macos")]
    {
        backend::init_mps_backend_metal()
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err(FerrotorchError::DeviceUnavailable)
    }
}

#[cfg(test)]
mod tests {
    use super::{FerrotorchError, MpsDevice, init_mps_backend, is_mps_available, mps_device_count};

    /// On non-macOS `is_mps_available()` is always `false`.
    /// On macOS it reflects whether a Metal device is present.
    #[test]
    fn is_mps_available_false_on_non_apple() {
        #[cfg(not(target_os = "macos"))]
        assert!(!is_mps_available());
        // On macOS the result depends on hardware; we only assert it doesn't panic.
        #[cfg(target_os = "macos")]
        let _ = is_mps_available();
    }

    /// On non-macOS targets `MpsDevice::new` always returns
    /// `DeviceUnavailable` — Metal does not exist outside macOS.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn mps_device_new_non_macos_returns_unavailable() {
        assert!(matches!(
            MpsDevice::new(0),
            Err(FerrotorchError::DeviceUnavailable)
        ));
        // Non-zero ordinals follow the same contract on non-macOS.
        assert!(matches!(
            MpsDevice::new(7),
            Err(FerrotorchError::DeviceUnavailable)
        ));
    }

    /// On macOS `MpsDevice::new(0)` is `Ok` iff a Metal device is present.
    /// This is the anti-zero-stub guard for the macOS branch — a hardcoded
    /// `Err(DeviceUnavailable)` on macOS would fail this test on any host
    /// where `is_mps_available()` is `true`.
    #[cfg(target_os = "macos")]
    #[test]
    fn mps_device_new_macos_returns_ok_when_available() {
        match MpsDevice::new(0) {
            Ok(d) => {
                assert_eq!(d.ordinal(), 0);
                assert!(
                    is_mps_available(),
                    "MpsDevice::new(0) returned Ok but is_mps_available() == false"
                );
            }
            Err(FerrotorchError::DeviceUnavailable) => {
                assert!(
                    !is_mps_available(),
                    "MpsDevice::new(0) returned DeviceUnavailable but is_mps_available() == true"
                );
            }
            Err(e) => panic!("unexpected error from MpsDevice::new(0) on macOS: {e:?}"),
        }
    }

    /// On macOS, `MpsDevice::new(N)` for `N != 0` must return
    /// `Err(InvalidArgument)` whenever Metal is available — Apple Silicon
    /// exposes a single integrated GPU.
    #[cfg(target_os = "macos")]
    #[test]
    fn mps_device_new_macos_rejects_nonzero_ordinal() {
        if !is_mps_available() {
            // No Metal device → DeviceUnavailable takes precedence over
            // ordinal validation. Either is acceptable in this branch.
            assert!(matches!(
                MpsDevice::new(7),
                Err(FerrotorchError::DeviceUnavailable)
            ));
            return;
        }
        assert!(matches!(
            MpsDevice::new(7),
            Err(FerrotorchError::InvalidArgument { .. })
        ));
    }

    /// On non-macOS targets `mps_device_count()` and `MpsDevice::count()`
    /// are always `0`.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn mps_device_count_is_zero_on_non_macos() {
        assert_eq!(mps_device_count(), 0);
        assert_eq!(MpsDevice::count(), 0);
    }

    /// On macOS, `mps_device_count()` mirrors `is_mps_available()` exactly
    /// (1 when present, 0 when absent), and `MpsDevice::count()` agrees
    /// with the free function.
    #[cfg(target_os = "macos")]
    #[test]
    fn mps_device_count_macos_matches_metal_availability() {
        let expected = usize::from(is_mps_available());
        assert_eq!(mps_device_count(), expected);
        assert_eq!(MpsDevice::count(), mps_device_count());
    }

    /// On non-macOS `init_mps_backend()` always returns `DeviceUnavailable`.
    /// On macOS it either succeeds or returns `DeviceUnavailable` (no Metal
    /// device in CI) — but never panics or returns an unexpected variant.
    #[test]
    fn init_mps_backend_contract() {
        #[cfg(not(target_os = "macos"))]
        assert!(matches!(
            init_mps_backend(),
            Err(FerrotorchError::DeviceUnavailable)
        ));
        #[cfg(target_os = "macos")]
        match init_mps_backend() {
            Ok(())
            | Err(FerrotorchError::DeviceUnavailable | FerrotorchError::InvalidArgument { .. }) => {
            }
            Err(e) => panic!("unexpected error from init_mps_backend: {e:?}"),
        }
    }

    #[test]
    fn device_mps_marker_round_trips() {
        // ferrotorch-core exposes Device::Mps(_) regardless of MPS
        // availability — the variant just doesn't do anything useful
        // without the backend.
        let d = ferrotorch_core::Device::Mps(0);
        assert!(d.is_mps());
        assert_eq!(format!("{d}"), "mps:0");
    }
}
