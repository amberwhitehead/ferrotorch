//! NVRTC-based CUDA C → PTX compilation, factored for reuse across the JIT
//! crate.
//!
//! Two production callers:
//!
//! 1. [`crate::codegen_gpu::f64_transcendental_ptx`] — lowers an f64 graph
//!    that contains a transcendental op to PTX via libdevice (#748 / #749).
//! 2. [`crate::fusion_gpu::apply_fused_gpu`] — compiles the f64 path of
//!    [`crate::fusion::FusedChain`]. PTX has no `*.approx.f64` hardware
//!    instructions, so the f64 chain is emitted as a single CUDA C
//!    function (using libdevice-resolved `exp` / `log` / `tanh` / ...)
//!    and routed through NVRTC.
//!
//! Both callers previously inlined an `nvrtc::compile_ptx_with_opts`
//! invocation with the same `#include <math.h>`-strip + `extern "C"`-rewrite
//! preprocessing; this module is the single shared implementation.
//!
//! ## REQ status (per `.design/ferrotorch-jit/nvrtc.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub fn compile_cuda_source_to_ptx` in `nvrtc.rs` (both cuda and stub); consumer: `codegen_gpu.rs:1237` and `fusion_gpu.rs:197` |
//! | REQ-2 | SHIPPED | `.lines().filter(|l| !l.trim().starts_with("#include <math.h>"))` in `nvrtc.rs`; consumer: every NVRTC invocation from the two call sites |
//! | REQ-3 | SHIPPED | `if l.starts_with("__global__ void ")` rewrite in `nvrtc.rs`; consumer: same call sites — cudarc's `cuModuleGetFunction` keys on unmangled name |
//! | REQ-4 | SHIPPED | `CompileOptions { arch: Some("compute_75"), ..Default::default() }` in `nvrtc.rs`; consumer: every PTX produced for `codegen_gpu` and `fusion_gpu` |
//! | REQ-5 | SHIPPED | `#[cfg(not(feature = "cuda"))] pub fn compile_cuda_source_to_ptx -> Err(JitError::CodegenError)` stub in `nvrtc.rs`; consumer: `codegen_gpu.rs:1237` compiles under both configs |

use crate::error::JitError;

/// NVRTC-compile a CUDA C source string to a PTX module string.
///
/// NVRTC links libdevice automatically when the source uses f64 math
/// intrinsics (`exp`, `log`, `tanh`, `pow`, ...), so the resulting PTX
/// has no unresolved external symbols — every `__nv_*` call is replaced
/// with libdevice's polynomial expansion inlined into the kernel.
///
/// # Preprocessing
///
/// - Strips `#include <math.h>` lines: nvcc's host compile expects these
///   for the host overloads, but NVRTC has no host headers in its include
///   path and rejects the line. The device-math symbols are still
///   resolved without it.
/// - Rewrites `__global__ void <name>(...)` to
///   `extern "C" __global__ void <name>(...)`. Without `extern "C"`,
///   NVRTC C++-mangles the symbol (e.g. `_Z9k_f64_expPKdPdi`); cudarc's
///   `cuModuleGetFunction` keys on the unmangled name so the load would
///   fail.
///
/// # Errors
///
/// Returns [`JitError::CodegenError`] if NVRTC rejects the source — the
/// error message includes both `kernel_name` (for traceability) and the
/// underlying NVRTC compile log.
#[cfg(feature = "cuda")]
pub fn compile_cuda_source_to_ptx(
    cuda_source: &str,
    kernel_name: &str,
) -> Result<String, JitError> {
    use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

    let nvrtc_source = cuda_source
        .lines()
        .filter(|l| !l.trim().starts_with("#include <math.h>"))
        .map(|l| {
            if l.starts_with("__global__ void ") {
                format!("extern \"C\" {l}")
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    // sm_75 is the floor for non-deprecated NVRTC targets in CUDA 13.x
    // (Volta sm_70 emits a deprecation warning) and supports f64
    // hardware ops on every Turing-and-newer GPU. libdevice's polynomial
    // expansions for f64 transcendentals depend on f64 FMA, which is
    // universally available at this baseline.
    let opts = CompileOptions {
        arch: Some("compute_75"),
        // `--use_fast_math` would enable approximate intrinsics that
        // sacrifice f64 precision; we want libdevice's IEEE-correct
        // polynomial expansions instead.
        ..Default::default()
    };

    let ptx = compile_ptx_with_opts(&nvrtc_source, opts).map_err(|e| JitError::CodegenError {
        message: format!("NVRTC compile of CUDA C source for kernel '{kernel_name}' failed: {e}"),
    })?;

    Ok(ptx.to_src())
}

/// Stub for when the `cuda` feature is disabled. Always returns
/// `JitError::CodegenError` — callers must already be gating their NVRTC
/// path on `cfg(feature = "cuda")`, so this signature only exists to keep
/// the module's exported surface uniform across feature combinations.
#[cfg(not(feature = "cuda"))]
pub fn compile_cuda_source_to_ptx(
    _cuda_source: &str,
    kernel_name: &str,
) -> Result<String, JitError> {
    Err(JitError::CodegenError {
        message: format!("NVRTC compile of kernel '{kernel_name}' requires the `cuda` feature"),
    })
}
