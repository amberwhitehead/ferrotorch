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
