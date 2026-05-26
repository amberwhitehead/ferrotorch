//! MSL kernel sources for ferrotorch-mps (#626).
//!
//! Each constant is a raw MSL shader string. On macOS the [`crate::backend`]
//! module compiles them at runtime via `MTLDevice::newLibraryWithSource_options_error`
//! and caches the resulting `MTLComputePipelineState` handles. On every other
//! platform these constants are never referenced (the entire `backend` module
//! is `cfg(target_os = "macos")` gated).
//!
//! # Kernel catalogue
//!
//! | Constant | Function name(s) | torch.mps analogue |
//! |---|---|---|
//! | `MATMUL_F32` | `matmul_f32` | `torch.mm` / `torch.matmul` |
//! | `BMM_F32` | `bmm_f32` | `torch.bmm` |
//! | `ELEMENTWISE_F32` | `add_f32`, `sub_f32`, `mul_f32`, `div_f32` | `torch.add/sub/mul/div` |
//! | `ACTIVATIONS_F32` | `relu_f32`, `sigmoid_f32` | `torch.relu`, `torch.sigmoid` |
//! | `SOFTMAX_F32` | `softmax_f32` | `torch.softmax` |
//! | `SUM_AXIS_F32` | `sum_axis_f32` | `torch.sum(dim=axis)` |
//!
//! ## REQ status (per `.design/ferrotorch-mps/kernels.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream
//! cites) live in the design doc; this synopsis is a one-line summary per
//! REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (six MSL source constants) | SHIPPED | `pub const MATMUL_F32 / BMM_F32 / ELEMENTWISE_F32 / ACTIVATIONS_F32 / SOFTMAX_F32 / SUM_AXIS_F32` in `kernels/mod.rs` covering Sprint C.7's 10 ops; consumer `pub fn MtlBackend::new` in `backend.rs` calls `compile_pipeline(&device, kernels::MATMUL_F32, "matmul_f32")?` through `compile_pipeline(&device, kernels::SUM_AXIS_F32, "sum_axis_f32")?` |
//! | REQ-2 (`include_str!` build-time embedding) | SHIPPED | every constant uses `include_str!("<name>.metal")` in `kernels/mod.rs`; consumer `MTLDevice::newLibraryWithSource_options_error` inside `fn compile_pipeline` in `backend.rs` consumes the `&'static str` verbatim — no filesystem lookup at runtime |
//! | REQ-3 (platform-agnostic compilation) | SHIPPED | `kernels/mod.rs` has no `cfg(target_os = "macos")` gate (only the `backend` module that consumes the constants is gated); consumer `cargo check -p ferrotorch-mps --no-default-features` on Linux/WSL compiles the module, and the 6 `kernel_source_*_present` tests in `ferrotorch-mps/tests/conformance_mps.rs` exercise the constants without a Metal device |
//! | REQ-4 (kernel function-name catalogue) | SHIPPED | the `//!` doc-comment carries the table mapping each constant to its declared kernel function name(s) and `torch.mps` analogue; consumer `fn compile_pipeline` in `backend.rs` is called with each `(MSL_CONST, fn_name_literal)` pair following the catalogue; drift is caught by `kernel_source_*_present` tests in `ferrotorch-mps/tests/conformance_mps.rs` |

/// MSL GEMM kernel: `C[m,n] = A[m,k] * B[k,n]`.
pub const MATMUL_F32: &str = include_str!("matmul_f32.metal");

/// MSL batched-GEMM kernel: `C[b,m,n] = A[b,m,k] * B[b,k,n]`.
pub const BMM_F32: &str = include_str!("bmm_f32.metal");

/// MSL elementwise binary kernels: add, sub, mul, div (f32).
pub const ELEMENTWISE_F32: &str = include_str!("elementwise_f32.metal");

/// MSL activation kernels: relu, sigmoid (f32).
pub const ACTIVATIONS_F32: &str = include_str!("activations_f32.metal");

/// MSL last-dim softmax kernel (f32).
pub const SOFTMAX_F32: &str = include_str!("softmax_f32.metal");

/// MSL axis-reduction sum kernel (f32).
pub const SUM_AXIS_F32: &str = include_str!("sum_axis_f32.metal");
