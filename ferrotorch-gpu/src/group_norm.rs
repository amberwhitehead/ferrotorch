//! GPU GroupNorm forward (f32).
//!
//! Mirrors `torch.nn.functional.group_norm` for input shape
//! `[B, C, H, W]` with `G` groups and per-channel affine
//! parameters `γ, β` of shape `[C]`:
//!
//! ```text
//! For each (b, g):
//!     stats over c in [g*(C/G), (g+1)*(C/G)), all (h, w):
//!         mean = (1/N) * Σ x
//!         var  = (1/N) * Σ (x - mean)^2
//!         inv_std = 1 / sqrt(var + eps)
//!     For each c, (h, w) in the group:
//!         out[b, c, h, w] = γ[c] * (x[b, c, h, w] - mean) * inv_std + β[c]
//! ```
//!
//! where `N = (C/G) * H * W`.
//!
//! # Why this is its own module
//!
//! GroupNorm is shape-aware (per-group reduction over `(C/G, H, W)`)
//! and is required by the SD VAE / UNet stack but absent from
//! `kernels.rs`'s LayerNorm family. Keeping it isolated avoids
//! conflating the row-major LayerNorm semantics with the
//! channels-then-spatial GroupNorm reduction.
//!
//! # Kernel layout
//!
//! - One CUDA block per `(b, g)` pair (grid: `(groups, batch, 1)`).
//! - 256 threads per block; one block per group; per-block reductions
//!   for the mean and variance use 256 f32 of shared memory.
//! - Affine application uses the per-channel `γ[c]`, `β[c]`.
//!
//! ## REQ status (per `.design/ferrotorch-gpu/group_norm.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream
//! cites) live in the design doc; this synopsis is a one-line summary per
//! REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`gpu_group_norm_f32`) | NOT-STARTED | blocker #1356 — impl: `pub fn gpu_group_norm_f32 in group_norm.rs` mirrors upstream `GroupNormKernelImpl` at `aten/src/ATen/native/cuda/group_norm_kernel.cu:649`, BUT no non-test production consumer invokes it (only `#[cfg(test)] mod tests` callers + `pub use` re-export at `lib.rs`). Prereq: needs a `group_norm_f32` trait method on `GpuBackend` + `CudaBackendImpl` consumer wiring |
//! | REQ-2 (PTX template + ABI) | SHIPPED | `pub(crate) const GROUP_NORM_PTX in group_norm.rs` carries the three-pass mean / variance / affine PTX; ABI matches the launch site; verified by unit tests being numerically correct |
//! | REQ-3 (input validation) | SHIPPED | validation checks in `pub fn gpu_group_norm_f32 in group_norm.rs` (groups divisibility, input length, weight length, bias length, device ordinal) |
//! | REQ-4 (degenerate short-circuit) | SHIPPED | degenerate short-circuit in `gpu_group_norm_f32 in group_norm.rs` returns `alloc_zeros_f32(n, device)` for `n == 0 || channels == 0 || hw == 0` |
//! | REQ-5 (backend trait wiring) | NOT-STARTED | blocker #1357 — no `group_norm_*` trait method exists on `GpuBackend` in `ferrotorch-core/src/gpu_dispatch.rs`, and no `fn group_norm_*` exists in `backend_impl.rs`. Prereq: add `group_norm_f32` to the `GpuBackend` trait + wire `CudaBackendImpl` to call `crate::group_norm::gpu_group_norm_f32`; ferrotorch-nn's `GroupNorm` currently falls back to a CPU path or per-element kernels |

#[cfg(feature = "cuda")]
use cudarc::driver::{LaunchConfig, PushKernelArg};

#[cfg(feature = "cuda")]
use crate::buffer::CudaBuffer;
#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use crate::error::{GpuError, GpuResult};
#[cfg(feature = "cuda")]
use crate::transfer::alloc_zeros_f32;

/// PTX source for the GroupNorm forward kernel.
///
/// The kernel runs as `(groups, batch, 1)` blocks of 256 threads. Each
/// block reduces over `(C/G) * H*W` elements, computes the per-group
/// mean and variance, and writes the per-element affine-normalized
/// output back to global memory.
///
/// ABI: `(in_ptr, out_ptr, w_ptr, b_ptr, batch, channels, groups,
/// hw, eps)`.
#[cfg(feature = "cuda")]
pub(crate) const GROUP_NORM_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.shared .align 4 .f32 sdata[256];

.visible .entry group_norm_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u64 w_ptr,
    .param .u64 b_ptr,
    .param .u32 batch,
    .param .u32 channels,
    .param .u32 groups,
    .param .u32 hw,
    .param .f32 eps
) {
    .reg .u32 %r_tid, %r_bdim, %r_g, %r_b, %channels_r, %groups_r, %hw_r;
    .reg .u32 %cpg, %c_start, %c_end, %n_elem, %i, %c, %p, %half, %r_otid;
    .reg .u64 %in, %out, %w, %bv, %off, %row_off, %sbase, %saddr, %el_per_b, %el_per_g;
    .reg .f32 %val, %mean, %var, %diff, %eps_r, %inv_std, %normed, %wv, %bw, %result, %other_val, %n_f;
    .reg .pred %pe, %lp, %rp, %g_oob, %b_oob;

    ld.param.u64 %in,  [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %w,   [w_ptr];
    ld.param.u64 %bv,  [b_ptr];
    ld.param.u32 %channels_r, [channels];
    ld.param.u32 %groups_r,   [groups];
    ld.param.u32 %hw_r,       [hw];
    ld.param.f32 %eps_r,      [eps];

    mov.u64 %sbase, sdata;

    mov.u32 %r_g, %ctaid.x;       // group index
    mov.u32 %r_b, %ctaid.y;       // batch index
    mov.u32 %r_bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;

    // c_per_group = channels / groups
    div.u32 %cpg, %channels_r, %groups_r;

    // c_start = r_g * cpg, c_end = c_start + cpg
    mul.lo.u32 %c_start, %r_g, %cpg;
    add.u32 %c_end, %c_start, %cpg;

    // n_elem (per group): cpg * hw
    mul.lo.u32 %n_elem, %cpg, %hw_r;
    cvt.rn.f32.u32 %n_f, %n_elem;

    // Base byte offset for this (b, g) block:
    //   in[b, c_start, 0, 0]:  byte offset = ((b * channels + c_start) * hw) * 4
    // We use `row_off` to hold (b * channels + c_start) * hw as u64 bytes.
    cvt.u64.u32 %row_off, %r_b;
    cvt.u64.u32 %el_per_b, %channels_r;
    mul.lo.u64 %row_off, %row_off, %el_per_b;  // b * channels
    cvt.u64.u32 %el_per_g, %c_start;
    add.u64 %row_off, %row_off, %el_per_g;     // b * channels + c_start
    cvt.u64.u32 %el_per_b, %hw_r;
    mul.lo.u64 %row_off, %row_off, %el_per_b;  // (b * channels + c_start) * hw
    shl.b64 %row_off, %row_off, 2;             // bytes

    // ---- Pass 1: mean ----
    mov.f32 %mean, 0f00000000;
    mov.u32 %i, %r_tid;
SM:
    setp.ge.u32 %lp, %i, %n_elem;
    @%lp bra SMD;
    cvt.u64.u32 %off, %i;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.f32 %val, [%off];
    add.f32 %mean, %mean, %val;
    add.u32 %i, %i, %r_bdim;
    bra SM;
SMD:
    // store partial sum to shared
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %mean;
    bar.sync 0;
    mov.u32 %half, %r_bdim;
MR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra MRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra MRS;
    add.u32 %r_otid, %r_tid, %half;
    cvt.u64.u32 %off, %r_otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other_val, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %mean, [%saddr];
    add.f32 %mean, %mean, %other_val;
    st.shared.f32 [%saddr], %mean;
MRS:
    bar.sync 0;
    bra MR;
MRD:
    ld.shared.f32 %mean, [%sbase];
    div.approx.f32 %mean, %mean, %n_f;
    bar.sync 0;

    // ---- Pass 2: variance ----
    mov.f32 %var, 0f00000000;
    mov.u32 %i, %r_tid;
SV:
    setp.ge.u32 %lp, %i, %n_elem;
    @%lp bra SVD;
    cvt.u64.u32 %off, %i;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.f32 %val, [%off];
    sub.f32 %diff, %val, %mean;
    fma.rn.f32 %var, %diff, %diff, %var;
    add.u32 %i, %i, %r_bdim;
    bra SV;
SVD:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %var;
    bar.sync 0;
    mov.u32 %half, %r_bdim;
VR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra VRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra VRS;
    add.u32 %r_otid, %r_tid, %half;
    cvt.u64.u32 %off, %r_otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other_val, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %var, [%saddr];
    add.f32 %var, %var, %other_val;
    st.shared.f32 [%saddr], %var;
VRS:
    bar.sync 0;
    bra VR;
VRD:
    ld.shared.f32 %var, [%sbase];
    div.approx.f32 %var, %var, %n_f;
    add.f32 %var, %var, %eps_r;
    sqrt.approx.f32 %inv_std, %var;
    rcp.approx.f32 %inv_std, %inv_std;
    bar.sync 0;

    // ---- Pass 3: write normalized + affine ----
    // For element i in [0, n_elem): channel = c_start + i / hw.
    mov.u32 %i, %r_tid;
NM:
    setp.ge.u32 %lp, %i, %n_elem;
    @%lp bra NMD;
    cvt.u64.u32 %off, %i;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.f32 %val, [%off];
    sub.f32 %normed, %val, %mean;
    mul.f32 %normed, %normed, %inv_std;

    // Compute channel index = c_start + (i / hw)
    div.u32 %p, %i, %hw_r;
    add.u32 %c, %c_start, %p;

    cvt.u64.u32 %off, %c;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %w, %off;
    ld.global.f32 %wv, [%saddr];
    add.u64 %saddr, %bv, %off;
    ld.global.f32 %bw, [%saddr];
    fma.rn.f32 %result, %wv, %normed, %bw;

    cvt.u64.u32 %off, %i;
    shl.b64 %off, %off, 2;
    add.u64 %off, %out, %off;
    add.u64 %off, %off, %row_off;
    st.global.f32 [%off], %result;
    add.u32 %i, %i, %r_bdim;
    bra NM;
NMD:
    ret;
}
";

/// GPU forward GroupNorm over a `[B, C, H, W]`-laid-out f32 buffer.
///
/// Computes `out = γ[c] * (x - mean_g) / sqrt(var_g + eps) + β[c]`,
/// where mean / variance are taken per-`(batch, group)` over
/// `(C/G) * H*W` elements and the affine `γ, β` are per-channel.
///
/// # Arguments
///
/// - `input` — flat `[B * C * H * W]` f32 buffer in row-major layout
///   matching PyTorch `[B, C, H, W]`.
/// - `weight` — `[C]` per-channel scale.
/// - `bias` — `[C]` per-channel shift.
/// - `batch`, `channels`, `hw` — outer dims. `hw = H * W`.
/// - `groups` — number of groups; must divide `channels`.
/// - `eps` — numerical stability constant (SD uses `1e-6`).
/// - `device` — owning GPU device for all buffers.
///
/// # Errors
///
/// - [`GpuError::ShapeMismatch`] when any buffer length disagrees with
///   the declared dims or when `groups` does not divide `channels`.
/// - [`GpuError::DeviceMismatch`] when buffers live on a different
///   device.
/// - [`GpuError::PtxCompileFailed`] if the PTX module fails to compile.
/// - [`GpuError::Driver`] on launch failure.
#[cfg(feature = "cuda")]
pub fn gpu_group_norm_f32(
    input: &CudaBuffer<f32>,
    weight: &CudaBuffer<f32>,
    bias: &CudaBuffer<f32>,
    batch: usize,
    channels: usize,
    groups: usize,
    hw: usize,
    eps: f32,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    if groups == 0 || channels % groups != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "group_norm",
            expected: vec![channels],
            got: vec![groups],
        });
    }
    let n = batch * channels * hw;
    if input.len() != n {
        return Err(GpuError::ShapeMismatch {
            op: "group_norm",
            expected: vec![batch, channels, hw],
            got: vec![input.len()],
        });
    }
    if weight.len() != channels {
        return Err(GpuError::ShapeMismatch {
            op: "group_norm",
            expected: vec![channels],
            got: vec![weight.len()],
        });
    }
    if bias.len() != channels {
        return Err(GpuError::ShapeMismatch {
            op: "group_norm",
            expected: vec![channels],
            got: vec![bias.len()],
        });
    }
    if input.device_ordinal() != device.ordinal()
        || weight.device_ordinal() != device.ordinal()
        || bias.device_ordinal() != device.ordinal()
    {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: input.device_ordinal(),
        });
    }

    // Degenerate / empty shape: return zero output.
    if n == 0 || channels == 0 || hw == 0 {
        return alloc_zeros_f32(n, device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        GROUP_NORM_PTX,
        "group_norm_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "group_norm_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let batch_u32 = batch as u32;
    let channels_u32 = channels as u32;
    let groups_u32 = groups as u32;
    let hw_u32 = hw as u32;

    let cfg = LaunchConfig {
        grid_dim: (groups_u32.max(1), batch_u32.max(1), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for `group_norm_kernel`
    //   returned by `module_cache::get_or_compile`; ABI matches
    //   `(in_ptr, out_ptr, w_ptr, b_ptr, batch, channels, groups,
    //   hw, eps)` exactly per `GROUP_NORM_PTX`.
    // - `input.len() == batch * channels * hw` (validated above);
    //   `weight.len() == bias.len() == channels` (validated above).
    //   All three buffers live on `device` (validated above).
    // - `out` was just allocated with the same length and cannot
    //   alias `input/weight/bias` (Rust borrow rules; `out` is `&mut`).
    // - Grid `(groups, batch, 1)` × block `(256, 1, 1)`: each block
    //   reads `(c/G * hw)` elements from `input` starting at byte
    //   offset `((b*C + g*(C/G)) * hw) * 4` and writes the same
    //   range to `out`. The kernel's per-thread loop is bounded by
    //   the per-group element count and the channel index `c =
    //   c_start + i/hw` stays in `[g*(C/G), (g+1)*(C/G)) ⊂ [0, C)`.
    // - Shared memory: 256 * 4 = 1024 bytes (matches PTX
    //   `.shared sdata[256]`). One scalar `f32` per thread, used
    //   for the mean and variance reductions across the block.
    // - `eps: f32` passed by-ref; cudarc copies it into the launch
    //   parameter buffer.
    // - Stream sync is the caller's responsibility (matches every
    //   other kernel launch in this crate).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(weight.inner())
            .arg(bias.inner())
            .arg(&batch_u32)
            .arg(&channels_u32)
            .arg(&groups_u32)
            .arg(&hw_u32)
            .arg(&eps)
            .launch(cfg)?;
    }

    Ok(out)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::transfer::{cpu_to_gpu, gpu_to_cpu};

    /// Reference CPU implementation: PyTorch-style GroupNorm over
    /// `[B, C, H, W]` with per-channel γ, β.
    fn cpu_group_norm_ref(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        b: usize,
        c: usize,
        groups: usize,
        hw: usize,
        eps: f32,
    ) -> Vec<f32> {
        let cpg = c / groups;
        let n_per_group = (cpg * hw) as f32;
        let mut out = vec![0.0f32; b * c * hw];
        for bi in 0..b {
            for g in 0..groups {
                let c0 = g * cpg;
                // mean
                let mut sum = 0.0_f64;
                for cc in c0..c0 + cpg {
                    for p in 0..hw {
                        sum += x[((bi * c + cc) * hw) + p] as f64;
                    }
                }
                let mean = (sum / n_per_group as f64) as f32;
                // var
                let mut vsum = 0.0_f64;
                for cc in c0..c0 + cpg {
                    for p in 0..hw {
                        let d = x[((bi * c + cc) * hw) + p] - mean;
                        vsum += (d * d) as f64;
                    }
                }
                let var = (vsum / n_per_group as f64) as f32;
                let inv_std = 1.0 / (var + eps).sqrt();
                for cc in c0..c0 + cpg {
                    let w = gamma[cc];
                    let bv = beta[cc];
                    for p in 0..hw {
                        let i = ((bi * c + cc) * hw) + p;
                        out[i] = w * (x[i] - mean) * inv_std + bv;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn group_norm_matches_cpu_small() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let b = 2;
        let c = 16;
        let groups = 4;
        let hw = 5;
        let eps = 1e-6_f32;
        // deterministic data
        let n = b * c * hw;
        let x: Vec<f32> = (0..n).map(|k| ((k % 13) as f32) * 0.1 - 0.6).collect();
        let gamma: Vec<f32> = (0..c).map(|k| 1.0 + 0.05 * (k as f32)).collect();
        let beta: Vec<f32> = (0..c).map(|k| -0.1 + 0.02 * (k as f32)).collect();

        let xg = cpu_to_gpu(&x, &device).unwrap();
        let gg = cpu_to_gpu(&gamma, &device).unwrap();
        let bg = cpu_to_gpu(&beta, &device).unwrap();
        let yg = gpu_group_norm_f32(&xg, &gg, &bg, b, c, groups, hw, eps, &device).unwrap();
        let got = gpu_to_cpu(&yg, &device).unwrap();
        let expected = cpu_group_norm_ref(&x, &gamma, &beta, b, c, groups, hw, eps);
        assert_eq!(got.len(), expected.len());
        let mut max_abs = 0.0_f32;
        for (a, e) in got.iter().zip(expected.iter()) {
            let d = (a - e).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        assert!(
            max_abs < 1e-4,
            "group_norm gpu vs cpu max abs diff = {max_abs}"
        );
    }

    #[test]
    fn group_norm_sd_vae_shape() {
        // SD VAE uses G=32; smallest channel count is 128 (block_out_channels[0]).
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let b = 1;
        let c = 128;
        let groups = 32;
        let hw = 4 * 4; // 4x4 spatial as a quick test
        let eps = 1e-6_f32;
        let n = b * c * hw;
        let x: Vec<f32> = (0..n).map(|k| ((k as f32) * 0.001).sin()).collect();
        let gamma: Vec<f32> = (0..c).map(|k| 0.9 + 0.1 * ((k % 7) as f32) / 7.0).collect();
        let beta: Vec<f32> = (0..c).map(|k| 0.05 * ((k % 3) as f32) - 0.05).collect();

        let xg = cpu_to_gpu(&x, &device).unwrap();
        let gg = cpu_to_gpu(&gamma, &device).unwrap();
        let bg = cpu_to_gpu(&beta, &device).unwrap();
        let yg = gpu_group_norm_f32(&xg, &gg, &bg, b, c, groups, hw, eps, &device).unwrap();
        let got = gpu_to_cpu(&yg, &device).unwrap();
        let expected = cpu_group_norm_ref(&x, &gamma, &beta, b, c, groups, hw, eps);
        let mut max_abs = 0.0_f32;
        for (a, e) in got.iter().zip(expected.iter()) {
            let d = (a - e).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        assert!(
            max_abs < 1e-4,
            "group_norm SD-shape gpu vs cpu max abs diff = {max_abs}"
        );
    }

    #[test]
    fn group_norm_validates_groups_divisibility() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        // 10 channels with 3 groups -- doesn't divide.
        let b = 1;
        let c = 10;
        let groups = 3;
        let hw = 4;
        let x = vec![0.0f32; b * c * hw];
        let gamma = vec![1.0f32; c];
        let beta = vec![0.0f32; c];
        let xg = cpu_to_gpu(&x, &device).unwrap();
        let gg = cpu_to_gpu(&gamma, &device).unwrap();
        let bg = cpu_to_gpu(&beta, &device).unwrap();
        let res = gpu_group_norm_f32(&xg, &gg, &bg, b, c, groups, hw, 1e-6, &device);
        assert!(matches!(res, Err(GpuError::ShapeMismatch { .. })));
    }
}
