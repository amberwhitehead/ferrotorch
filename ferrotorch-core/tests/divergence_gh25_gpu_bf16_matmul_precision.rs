//! Divergence test for forecast-bio/ferrotorch#25 (tracked locally as #1543,
//! umbrella #1542): GPU bf16 matmul on the 3D × 2D ViT linear-shape
//! `(1, 200, 4096) @ (4096, 768)` produces `max|Δ|` vs f32-reference of
//! **2.37e-2**, while the same input on CPU bf16 measures **4.77e-4** — a
//! ~50× precision gap. Reference: GH issue
//! <https://github.com/forecast-bio/ferrotorch/issues/25>.
//!
//! # Code-path audit (host-side, no CUDA on this machine)
//!
//! For `Tensor<bf16>::matmul` on a `(1, 200, 4096)` × `(4096, 768)` GPU pair
//! the dispatcher is `grad_fns::linalg::matmul_differentiable`
//! (`ferrotorch-core/src/grad_fns/linalg.rs:1513`).
//!
//! - The 2D × 2D GPU bf16 arm at `linalg.rs:1568-1601` dispatches to
//!   `backend.matmul_bf16_bf16` → `gpu_matmul_bf16_bf16` in
//!   `ferrotorch-gpu/src/blas.rs:2527-2649`, which uses `cublasGemmEx` with
//!   `CUDA_R_16BF` in/out and `CUBLAS_COMPUTE_32F` accumulator — the standard
//!   ~1.5e-3 bf16+f32-accum floor. NOT the 3D × 2D path.
//!
//! - The (3, 2) tuple falls through the `match (a.ndim(), b.ndim())` table at
//!   `linalg.rs:1614-1620` (only `(1,1)`, `(2,1)`, `(2,2)`, and `(3,3)`-same-
//!   batch are matched).
//!
//! - The GPU "broadcast-bmm" arm at `linalg.rs:1626` is guarded by
//!   `is_f32::<T>() || is_f64::<T>()` — **bf16 is excluded.** The only
//!   GPU-direct ≥3D matmul kernels for bf16 are
//!   `gpu_matmul_bf16_bf16_strided_batched` / `gpu_matmul_bf16_bf16_strided_batched_nt`
//!   in `ferrotorch-gpu/src/blas.rs:3084 / :2977`, but no dispatcher routes
//!   bf16 ≥3D matmul through them today (no `broadcast_bmm_bf16` exists in
//!   `gpu_dispatch.rs` and no `bf16` arm in the `matmul_differentiable`
//!   broadcast guard).
//!
//! - So bf16 falls through to the CPU fallback at `linalg.rs:1704`:
//!   `linalg::matmul(&a, &b)` → `ops/linalg.rs:431` → `broadcast_matmul` at
//!   `ops/linalg.rs:472`. That function copies the GPU bf16 buffer to CPU
//!   via `a.data_vec()` (`tensor.rs:709`), routes the per-batch slab through
//!   `mm_raw<bf16>` (`ops/linalg.rs:1088`), and copies the bf16 result back
//!   to GPU via `t.to(device)` (`ops/linalg.rs:572`).
//!
//! - `mm_raw<bf16>` for `max_dim = 4096 > DIRECT_MM_THRESHOLD (128)` lands in
//!   the f16/bf16 fallback at `ops/linalg.rs:1224-1244`: every bf16 element
//!   is upcast to f64, matrix multiplied via faer in f64, then downcast to
//!   bf16. f64 accumulation is **strictly better** than the cuBLAS f32
//!   accumulator, so the result of this round-trip path should match the
//!   pure-CPU bf16 measurement (4.77e-4).
//!
//! # The divergence
//!
//! If the audit above is exhaustive, GPU bf16 3D × 2D should measure the
//! **same** error as CPU bf16 (4.77e-4) because they execute the same code
//! after the GPU→CPU round-trip. The user observes 2.37e-2 — 50× worse —
//! which means **the GPU path is NOT using the broadcast_matmul + mm_raw f64
//! fallback** that my audit claims. Three possibilities:
//!
//! 1. A path I missed: some other dispatcher converts bf16 GPU 3D×2D into
//!    a direct GPU kernel (e.g. via `linear_fused` after a reshape, or via
//!    `bmm_bf16_bf16` after a broadcast-by-stride-0 expand on the 2D weight).
//!    `linear_fused` at `linalg.rs:1283-1284` ERRORS for bf16 on GPU
//!    (`NotImplementedOnCuda { op: "linear_fused" }`), so the bug isn't
//!    there. `bmm_bf16_bf16` is reachable only through `bmm_differentiable`
//!    (`linalg.rs:1494`), which requires `ndim == 3` for BOTH operands.
//!
//! 2. `data_vec()` / `t.to(device)` for bf16 silently lose precision (would
//!    contradict the byte-copy invariant in `tensor.rs:838-844`).
//!
//! 3. The reporter's measurement runs on a ferrotorch revision NEWER than the
//!    one I audited that added a direct bf16 GPU 3D path with a buggy
//!    accumulator (the issue does say "after #19, #22, #23 all landed").
//!
//! Whichever of (1)/(2)/(3) is true, the test below pins the **expected**
//! semantics so that the regression is detected once a CUDA-enabled host
//! runs it: GPU bf16 error must be within 5× of CPU bf16 error.
//!
//! # Suggested fix shape (per upstream issue)
//!
//! Wire a `broadcast_bmm_bf16` (or a dedicated 3D × 2D bf16 dispatch) into
//! `matmul_differentiable`'s broadcast guard so that 3D × 2D bf16 lands on
//! `gpu_matmul_bf16_bf16_strided_batched` (already correct: CUDA_R_16BF in/
//! out, CUBLAS_COMPUTE_32F accumulator — see `blas.rs:3142`). That kernel
//! gives the standard ~1.5e-3 cuBLAS bf16+f32-accum floor that the upstream
//! issue expects.
//!
//! Tracking: forecast-bio/ferrotorch#25 (umbrella #1542, local #1543).

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::Tensor;
use ferrotorch_core::creation::from_vec;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the bf16 precision test");
    });
}

/// Deterministic xorshift32 PRNG. The forecast-bio/decode test recipe says
/// "deterministic xorshift" with activation scale 0.05 and weight scale 0.03;
/// this is the simplest xorshift that produces a reproducible bit pattern
/// independent of the host's `rand` version. Seeds are fixed so any CUDA
/// host running this test sees the same activation/weight values.
fn xorshift_iter(mut state: u32) -> impl FnMut() -> f32 {
    move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        // Map to (-1, 1) via top 24 bits (mantissa of f32).
        let v = (state >> 8) as f32 / (1u32 << 24) as f32; // [0, 1)
        v * 2.0 - 1.0
    }
}

fn build_activations() -> Vec<f32> {
    // shape (1, 200, 4096), seed 0xCAFEBABEu32, scaled by 0.05.
    let mut next = xorshift_iter(0xCAFE_BABE);
    let n: usize = 200 * 4096;
    (0..n).map(|_| next() * 0.05).collect()
}

fn build_weights() -> Vec<f32> {
    // shape (4096, 768), seed 0xDEADBEEFu32, scaled by 0.03.
    let mut next = xorshift_iter(0xDEAD_BEEF);
    let n: usize = 4096 * 768;
    (0..n).map(|_| next() * 0.03).collect()
}

fn f32_cpu(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    from_vec::<f32>(data.to_vec(), shape).expect("f32 cpu tensor")
}

fn bf16_cpu(data: &[f32], shape: &[usize]) -> Tensor<half::bf16> {
    let bf: Vec<half::bf16> = data.iter().copied().map(half::bf16::from_f32).collect();
    from_vec::<half::bf16>(bf, shape).expect("bf16 cpu tensor")
}

fn bf16_cuda(data: &[f32], shape: &[usize]) -> Tensor<half::bf16> {
    bf16_cpu(data, shape)
        .to(Device::Cuda(0))
        .expect("Tensor<bf16>::to(Cuda) must succeed")
}

fn max_abs_diff_bf16_vs_f32(bf: &Tensor<half::bf16>, refer: &Tensor<f32>) -> f32 {
    let bf_vec = bf.data_vec().expect("bf16 result data_vec");
    let ref_vec = refer.data_vec().expect("f32 reference data_vec");
    assert_eq!(
        bf_vec.len(),
        ref_vec.len(),
        "bf16 result and f32 reference must have the same numel"
    );
    let mut max_err: f32 = 0.0;
    for (b, r) in bf_vec.iter().zip(ref_vec.iter()) {
        let d = (b.to_f32() - *r).abs();
        if d > max_err {
            max_err = d;
        }
    }
    max_err
}

/// Divergence: GPU bf16 matmul on the (1, 200, 4096) @ (4096, 768) ViT shape
/// gives max|Δ| ~2.37e-2 vs the f32 reference, while CPU bf16 on the exact
/// same inputs gives ~4.77e-4 — a 50× gap.
///
/// Upstream expectation (per GH issue #25): GPU bf16 should be ≤ ~1.5e-3
/// (standard cuBLAS bf16+f32-accumulator floor). This test asserts the
/// weaker claim "GPU bf16 ≤ 5× CPU bf16" which any sane GPU bf16 matmul
/// implementation satisfies on this shape.
///
/// # Why this matters
///
/// The decode-side end-to-end check `parity_bf16_vit7b16_gpu` shows
/// CLS Δ 6.6e-1 on GPU vs 5.9e-2 on CPU — the 11× end-to-end gap is
/// compounded from this single-op 50× gap over ~80 matmuls in a 40-layer
/// ViT. f32 on GPU is unaffected, so this is a bf16-on-GPU-only regression.
///
/// Tracking: forecast-bio/ferrotorch#25 (umbrella #1542, local #1543).
#[test]
#[ignore = "needs CUDA hardware; tracking forecast-bio/ferrotorch#25 / local #1543"]
fn divergence_gh25_gpu_bf16_matmul_precision_vit_3d_2d() {
    ensure_cuda_backend();

    // Build the exact (1, 200, 4096) @ (4096, 768) shape with deterministic
    // xorshift activations and weights at the scales given in the issue.
    let a_data = build_activations();
    let b_data = build_weights();

    // f32 oracle on CPU — this is the "correct" answer to compare both bf16
    // paths against.
    let a_f32 = f32_cpu(&a_data, &[1, 200, 4096]);
    let b_f32 = f32_cpu(&b_data, &[4096, 768]);
    let c_f32 = a_f32.matmul(&b_f32).expect("f32 CPU reference matmul");

    // CPU bf16 — expected max|Δ| ~ 4.77e-4 per the GH issue.
    let a_bf16_cpu = bf16_cpu(&a_data, &[1, 200, 4096]);
    let b_bf16_cpu = bf16_cpu(&b_data, &[4096, 768]);
    let c_bf16_cpu = a_bf16_cpu
        .matmul(&b_bf16_cpu)
        .expect("bf16 CPU matmul must succeed");
    let cpu_err = max_abs_diff_bf16_vs_f32(&c_bf16_cpu, &c_f32);

    // GPU bf16 — observed max|Δ| ~ 2.37e-2 per the GH issue (FAIL).
    let a_bf16_gpu = bf16_cuda(&a_data, &[1, 200, 4096]);
    let b_bf16_gpu = bf16_cuda(&b_data, &[4096, 768]);
    let c_bf16_gpu = a_bf16_gpu
        .matmul(&b_bf16_gpu)
        .expect("bf16 GPU matmul must succeed");
    let gpu_err = max_abs_diff_bf16_vs_f32(&c_bf16_gpu, &c_f32);

    // Sanity bound (NOT the GH#25 assertion): CPU bf16 should be ~5e-4. The GH
    // issue reports 4.77e-4; this host measures ~5.00e-4 (minor RNG/env
    // variance). Use a loose ceiling so this setup check doesn't block the
    // actual GPU precision assertions below, which ARE the #25 contract.
    assert!(
        cpu_err <= 1.0e-3,
        "CPU bf16 max|Δ| vs f32 ref = {cpu_err:.3e} (issue reports 4.77e-4, host ~5.0e-4); \
         test setup may be wrong if this is far off"
    );

    // Standard cuBLAS bf16+f32-accumulator floor on this shape.
    // The issue cites ~1.5e-3 as the torch GPU bf16 baseline; 2e-3 is a
    // little headroom for cuBLAS algo-selection / cuda-version noise.
    assert!(
        gpu_err <= 2.0e-3,
        "GPU bf16 max|Δ| vs f32 ref = {gpu_err:.3e}; \
         expected ≤ 2.0e-3 (standard cuBLAS bf16+f32-accum floor). \
         The current 3D × 2D bf16 GPU path either dispatches through a \
         non-cuBLAS kernel or uses a bf16 accumulator. See \
         grad_fns/linalg.rs:1626 — the broadcast-bmm guard is \
         `(is_f32::<T>() || is_f64::<T>())`, which excludes bf16 and falls \
         through to the CPU `broadcast_matmul` round-trip. \
         If THAT path were taken, GPU bf16 would match CPU bf16 \
         ({cpu_err:.3e}). The 50× gap means a different path is hit."
    );

    // Sanity bound: GPU bf16 should be within ~5× of CPU bf16. The CPU
    // path's f64 accumulator is genuinely better than GPU's f32, so a small
    // gap is expected, but not 50×.
    let ratio = gpu_err / cpu_err.max(f32::MIN_POSITIVE);
    assert!(
        ratio <= 5.0,
        "GPU bf16 error ({gpu_err:.3e}) is {ratio:.1}× worse than CPU bf16 \
         error ({cpu_err:.3e}); expected ≤ 5×. This is the headline \
         regression in forecast-bio/ferrotorch#25."
    );
}

/// Sister probe: verify that the GPU 2D × 2D bf16 path on a slice of the
/// same shape (200 × 4096) @ (4096 × 768) gives the correct ~1.5e-3 floor.
/// If this passes and the 3D × 2D test above fails, the divergence is
/// localised to the 3D dispatch (NOT the 2D `gpu_matmul_bf16_bf16` kernel),
/// confirming the bug shape "missing bf16 arm in matmul_differentiable
/// broadcast guard".
#[test]
#[ignore = "needs CUDA hardware; tracking forecast-bio/ferrotorch#25 / local #1543"]
fn divergence_gh25_gpu_bf16_matmul_precision_vit_2d_2d_baseline() {
    ensure_cuda_backend();

    // Same recipe but with a 2D activation slice of shape (200, 4096).
    let a_data: Vec<f32> = {
        let mut next = xorshift_iter(0xCAFE_BABE);
        let n: usize = 200 * 4096;
        (0..n).map(|_| next() * 0.05).collect()
    };
    let b_data = build_weights();

    let a_f32 = f32_cpu(&a_data, &[200, 4096]);
    let b_f32 = f32_cpu(&b_data, &[4096, 768]);
    let c_f32 = a_f32.matmul(&b_f32).expect("f32 CPU reference matmul (2D)");

    let a_bf16_gpu = bf16_cuda(&a_data, &[200, 4096]);
    let b_bf16_gpu = bf16_cuda(&b_data, &[4096, 768]);
    let c_bf16_gpu = a_bf16_gpu
        .matmul(&b_bf16_gpu)
        .expect("bf16 GPU 2D × 2D matmul must succeed");
    let gpu_err = max_abs_diff_bf16_vs_f32(&c_bf16_gpu, &c_f32);

    // 2D × 2D goes through `gpu_matmul_bf16_bf16` (cuBLAS GemmEx,
    // CUBLAS_COMPUTE_32F) — this is the known-good path.
    assert!(
        gpu_err <= 2.0e-3,
        "2D × 2D GPU bf16 max|Δ| vs f32 ref = {gpu_err:.3e}; \
         expected ≤ 2.0e-3 (cuBLAS bf16+f32-accum). If this fails the bug \
         is in `gpu_matmul_bf16_bf16` itself, not the 3D dispatcher."
    );
}
