//! Pairwise-distance GPU compute kernel — `torch.cdist`
//! (crosslink #1545 / sub #1535).
//!
//! Hand-written PTX owned by Rust (no CUDA C++, no nvrtc), loaded via
//! [`crate::module_cache::get_or_compile`] exactly like [`crate::triangular`].
//!
//! # Semantics (PyTorch parity)
//!
//! `torch.cdist(x1, x2, p)` is the batched Lp pairwise distance matrix:
//! `x1` is `[B, P, M]`, `x2` is `[B, R, M]`, the result is `[B, P, R]` with
//! `out[b, i, j] = (sum_k |x1[b,i,k] - x2[b,j,k]|^p)^(1/p)`. This mirrors the
//! upstream CUDA kernel `cdist_kernel_cuda_impl` at
//! `aten/src/ATen/native/cuda/DistanceKernel.cu:195` and the per-norm
//! accumulate/finish in `dists<scalar_t>::{p,one,two,inf}`
//! (`DistanceKernel.cu:50-86`). Upstream assigns one CUDA *block* per output
//! cell and parallelises the `M`-reduction across the block's threads; we keep
//! the identical arithmetic but assign one *thread* per output cell with a
//! serial `M`-loop — the reduction order over `k` is the same (ascending) so
//! the float result matches the CPU ferrotorch `cdist` path and `torch.cdist`
//! to the usual fp tolerance.
//!
//! `diff` fed to the accumulator is `|x1 - x2|` (`std::abs(*a - *b)` in
//! `DistanceKernel.cu:210`). Per-norm accumulation:
//! - **two** (`p == 2`): `agg += diff*diff`; `finish = sqrt(agg)`.
//! - **one** (`p == 1`): `agg += diff`; `finish = agg`.
//! - **inf** (`p == inf`): `agg = max(agg, diff)`; `finish = agg`.
//! - **general** (other `p`): `agg += pow(diff, p)`; `finish = pow(agg, 1/p)`.
//!
//! `p == 0` (count of nonzero diffs) is delegated to the CPU path — see the
//! `MODE_*` table; the GPU kernel covers the four common norms.
//!
//! ## REQ status (per `.design/ferrotorch-gpu/distance.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (f32 cdist, p=1/2/inf/general) | SHIPPED | `pub fn gpu_cdist_f32 in distance.rs`; consumer `CudaBackendImpl::cdist_f32 in backend_impl.rs` dispatched from `ops::tensor_ops::cdist` |
//! | REQ-2 (f64 cdist) | SHIPPED | `pub fn gpu_cdist_f64 in distance.rs`; consumer `CudaBackendImpl::cdist_f64 in backend_impl.rs` |
//! | REQ-3 (batched [B,P,M]x[B,R,M]) | SHIPPED | per-output-cell `(b,i,j)` decode in the cdist PTX; verified by `cdist_f32_batched` unit test |
//! | REQ-4 (norm modes) | SHIPPED | `MODE_TWO`/`MODE_ONE`/`MODE_INF`/`MODE_GENERAL` branch in the PTX; verified by `cdist_f32_{l2,l1,linf,p3}` unit tests vs CPU ref |

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::module_cache::get_or_compile;
use crate::transfer::{alloc_zeros_f32, alloc_zeros_f64};

const BLOCK_SIZE: u32 = 256;

fn launch_1d(n: usize) -> LaunchConfig {
    let grid = ((n as u32).saturating_add(BLOCK_SIZE - 1)) / BLOCK_SIZE;
    LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

// Norm-mode selector pushed to the kernel.
const MODE_GENERAL: u32 = 0;
const MODE_ONE: u32 = 1;
const MODE_TWO: u32 = 2;
const MODE_INF: u32 = 3;

/// Resolve the kernel `mode` for a given `p`, mirroring the dispatch in
/// `aten/src/ATen/native/cuda/DistanceKernel.cu:230-240`. `p == 0` returns
/// `None` (the GPU kernel does not cover the count-of-nonzeros norm; the
/// caller falls back to the CPU path).
fn mode_for_p(p: f64) -> Option<u32> {
    if p == 0.0 {
        None
    } else if p == 1.0 {
        Some(MODE_ONE)
    } else if p == 2.0 {
        Some(MODE_TWO)
    } else if p.is_infinite() {
        Some(MODE_INF)
    } else {
        Some(MODE_GENERAL)
    }
}

// ===========================================================================
// cdist f32
//
// Params: (x1_ptr, x2_ptr, out_ptr, total, p_dim, r_dim, m, p, mode)
//   x1  : f32[B * p_dim * m]
//   x2  : f32[B * r_dim * m]
//   out : f32[B * p_dim * r_dim]    (total = B * p_dim * r_dim)
// Thread t in [0, total): b = t / (p_dim*r_dim); rem = t % (p_dim*r_dim);
//   i = rem / r_dim; j = rem % r_dim.
//   off1 = (b*p_dim + i)*m; off2 = (b*r_dim + j)*m.
//   agg over k in [0,m): diff = |x1[off1+k] - x2[off2+k]|; accumulate per mode.
//   out[t] = finish(agg) per mode.
// ===========================================================================
const CDIST_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry cdist_f32_kernel(
    .param .u64 x1_ptr, .param .u64 x2_ptr, .param .u64 out_ptr,
    .param .u32 total, .param .u32 p_dim, .param .u32 r_dim, .param .u32 m,
    .param .f32 p, .param .u32 mode
) {
    .reg .u32 %gtid, %bid, %bdim, %tdx, %total, %pdim, %rdim, %m, %mode;
    .reg .u32 %rsz, %b, %rem, %i, %j, %o1, %o2, %k, %idx;
    .reg .u64 %x1, %x2, %out, %off, %addr, %a1, %a2;
    .reg .f32 %p_r, %agg, %va, %vb, %diff, %t1, %invp;
    .reg .pred %pred, %is_two, %is_one, %is_inf, %is_gen;

    ld.param.u64 %x1, [x1_ptr];
    ld.param.u64 %x2, [x2_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %total, [total];
    ld.param.u32 %pdim, [p_dim];
    ld.param.u32 %rdim, [r_dim];
    ld.param.u32 %m, [m];
    ld.param.f32 %p_r, [p];
    ld.param.u32 %mode, [mode];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tdx, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %tdx;
    setp.ge.u32 %pred, %gtid, %total;
    @%pred bra DONE;

    // decode (b, i, j)
    mul.lo.u32 %rsz, %pdim, %rdim;
    div.u32 %b, %gtid, %rsz;
    rem.u32 %rem, %gtid, %rsz;
    div.u32 %i, %rem, %rdim;
    rem.u32 %j, %rem, %rdim;

    // off1 = (b*pdim + i)*m
    mad.lo.u32 %o1, %b, %pdim, %i;
    mul.lo.u32 %o1, %o1, %m;
    // off2 = (b*rdim + j)*m
    mad.lo.u32 %o2, %b, %rdim, %j;
    mul.lo.u32 %o2, %o2, %m;

    setp.eq.u32 %is_two, %mode, 2;
    setp.eq.u32 %is_one, %mode, 1;
    setp.eq.u32 %is_inf, %mode, 3;
    setp.eq.u32 %is_gen, %mode, 0;

    mov.f32 %agg, 0f00000000;
    mov.u32 %k, 0;
LOOP:
    setp.ge.u32 %pred, %k, %m;
    @%pred bra FINISH;

    add.u32 %idx, %o1, %k;
    cvt.u64.u32 %addr, %idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %a1, %x1, %addr;
    ld.global.f32 %va, [%a1];

    add.u32 %idx, %o2, %k;
    cvt.u64.u32 %addr, %idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %a2, %x2, %addr;
    ld.global.f32 %vb, [%a2];

    sub.f32 %diff, %va, %vb;
    abs.f32 %diff, %diff;

    // two: agg += diff*diff
    @%is_two fma.rn.f32 %agg, %diff, %diff, %agg;
    // one: agg += diff
    @%is_one add.f32 %agg, %agg, %diff;
    // inf: agg = max(agg, diff)
    @%is_inf max.f32 %agg, %agg, %diff;
    // general: agg += pow(diff, p) = exp(p * log(diff)); guard diff==0 -> +0
    @!%is_gen bra SKIPGEN;
    setp.eq.f32 %pred, %diff, 0f00000000;
    @%pred bra SKIPGEN;
    lg2.approx.f32 %t1, %diff;
    mul.f32 %t1, %t1, %p_r;
    // pow uses base-2: diff^p = 2^(p*log2(diff))
    ex2.approx.f32 %t1, %t1;
    add.f32 %agg, %agg, %t1;
SKIPGEN:
    add.u32 %k, %k, 1;
    bra LOOP;
FINISH:
    // two: sqrt(agg)
    @%is_two sqrt.rn.f32 %agg, %agg;
    // general: agg^(1/p) = 2^( (1/p) * log2(agg) ); guard agg==0 -> 0
    @!%is_gen bra STORE;
    setp.eq.f32 %pred, %agg, 0f00000000;
    @%pred bra STORE;
    rcp.rn.f32 %invp, %p_r;
    lg2.approx.f32 %t1, %agg;
    mul.f32 %t1, %t1, %invp;
    ex2.approx.f32 %agg, %t1;
STORE:
    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out, %off;
    st.global.f32 [%addr], %agg;
DONE:
    ret;
}
";

// ===========================================================================
// cdist f64 — same structure, 8-byte values, f64 math.
// pow via lg2/ex2 done in f64 by exp2/log2 emulation: f64 has no
// lg2.approx, so use the identity diff^p = exp(p*ln(diff)) via lg2.approx on
// the f32-narrowed mantissa is too lossy; instead we compute pow through
// the natural-log path using the device __log/__exp equivalents available as
// f64 ops is not in base PTX. To keep f64 exact for the common norms we
// support p=1, p=2, inf directly and route GENERAL f64 through the host CPU
// fallback (see mode_for_p + the f64 launcher's MODE_GENERAL guard).
// ===========================================================================
const CDIST_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry cdist_f64_kernel(
    .param .u64 x1_ptr, .param .u64 x2_ptr, .param .u64 out_ptr,
    .param .u32 total, .param .u32 p_dim, .param .u32 r_dim, .param .u32 m,
    .param .f64 p, .param .u32 mode
) {
    .reg .u32 %gtid, %bid, %bdim, %tdx, %total, %pdim, %rdim, %m, %mode;
    .reg .u32 %rsz, %b, %rem, %i, %j, %o1, %o2, %k, %off;
    .reg .u64 %x1, %x2, %out, %addr, %a1, %a2;
    .reg .f64 %p_r, %agg, %va, %vb, %diff;
    .reg .pred %pred, %is_two, %is_one, %is_inf;

    ld.param.u64 %x1, [x1_ptr];
    ld.param.u64 %x2, [x2_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %total, [total];
    ld.param.u32 %pdim, [p_dim];
    ld.param.u32 %rdim, [r_dim];
    ld.param.u32 %m, [m];
    ld.param.f64 %p_r, [p];
    ld.param.u32 %mode, [mode];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tdx, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %tdx;
    setp.ge.u32 %pred, %gtid, %total;
    @%pred bra DONE;

    mul.lo.u32 %rsz, %pdim, %rdim;
    div.u32 %b, %gtid, %rsz;
    rem.u32 %rem, %gtid, %rsz;
    div.u32 %i, %rem, %rdim;
    rem.u32 %j, %rem, %rdim;

    mad.lo.u32 %o1, %b, %pdim, %i;
    mul.lo.u32 %o1, %o1, %m;
    mad.lo.u32 %o2, %b, %rdim, %j;
    mul.lo.u32 %o2, %o2, %m;

    setp.eq.u32 %is_two, %mode, 2;
    setp.eq.u32 %is_one, %mode, 1;
    setp.eq.u32 %is_inf, %mode, 3;

    mov.f64 %agg, 0d0000000000000000;
    mov.u32 %k, 0;
LOOP:
    setp.ge.u32 %pred, %k, %m;
    @%pred bra FINISH;

    add.u32 %off, %o1, %k;
    cvt.u64.u32 %addr, %off;
    shl.b64 %addr, %addr, 3;
    add.u64 %a1, %x1, %addr;
    ld.global.f64 %va, [%a1];

    add.u32 %off, %o2, %k;
    cvt.u64.u32 %addr, %off;
    shl.b64 %addr, %addr, 3;
    add.u64 %a2, %x2, %addr;
    ld.global.f64 %vb, [%a2];

    sub.f64 %diff, %va, %vb;
    abs.f64 %diff, %diff;

    @%is_two fma.rn.f64 %agg, %diff, %diff, %agg;
    @%is_one add.f64 %agg, %agg, %diff;
    @%is_inf max.f64 %agg, %agg, %diff;

    add.u32 %k, %k, 1;
    bra LOOP;
FINISH:
    @%is_two sqrt.rn.f64 %agg, %agg;
    cvt.u64.u32 %addr, %gtid;
    shl.b64 %addr, %addr, 3;
    add.u64 %addr, %out, %addr;
    st.global.f64 [%addr], %agg;
DONE:
    ret;
}
";

/// Launch the cdist kernel over `total = b * p_dim * r_dim` output cells, one
/// thread each. `mode` selects the norm (see [`MODE_GENERAL`] etc.). Writes
/// `out_slice` (length `total`).
#[allow(clippy::too_many_arguments)]
fn launch_cdist_f32(
    x1: &CudaSlice<f32>,
    x2: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    total: usize,
    p_dim: usize,
    r_dim: usize,
    m: usize,
    p: f64,
    mode: u32,
    device: &GpuDevice,
) -> GpuResult<()> {
    if total == 0 {
        return Ok(());
    }
    if out.len() < total {
        return Err(GpuError::LengthMismatch {
            a: out.len(),
            b: total,
        });
    }
    let stream = device.stream();
    let ctx = device.context();
    let f = get_or_compile(
        ctx,
        CDIST_F32_PTX,
        "cdist_f32_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "cdist_f32_kernel",
        source: e,
    })?;
    let cfg = launch_1d(total);
    let total_u = total as u32;
    let pdim_u = p_dim as u32;
    let rdim_u = r_dim as u32;
    let m_u = m as u32;
    let p_f = p as f32;
    // SAFETY:
    // - `f` is `cdist_f32_kernel`; its 9-arg signature
    //   (x1, x2, out, total, p_dim, r_dim, m, p, mode) matches the args below.
    // - `x1`/`x2` hold >= `b*p_dim*m` / `b*r_dim*m` (caller's tensors); `out`
    //   holds >= `total` (checked), distinct allocation, only `&mut`.
    // - Each thread `t in [0,total)` (bound-checked) reads `x1[off1+k]`,
    //   `x2[off2+k]` for `k in [0,m)` (in bounds: `off1 < b*p_dim*m`,
    //   `off2 < b*r_dim*m`) and writes one `out[t]`.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(x1)
            .arg(x2)
            .arg(out)
            .arg(&total_u)
            .arg(&pdim_u)
            .arg(&rdim_u)
            .arg(&m_u)
            .arg(&p_f)
            .arg(&mode)
            .launch(cfg)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_cdist_f64(
    x1: &CudaSlice<f64>,
    x2: &CudaSlice<f64>,
    out: &mut CudaSlice<f64>,
    total: usize,
    p_dim: usize,
    r_dim: usize,
    m: usize,
    p: f64,
    mode: u32,
    device: &GpuDevice,
) -> GpuResult<()> {
    if total == 0 {
        return Ok(());
    }
    if out.len() < total {
        return Err(GpuError::LengthMismatch {
            a: out.len(),
            b: total,
        });
    }
    let stream = device.stream();
    let ctx = device.context();
    let f = get_or_compile(
        ctx,
        CDIST_F64_PTX,
        "cdist_f64_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "cdist_f64_kernel",
        source: e,
    })?;
    let cfg = launch_1d(total);
    let total_u = total as u32;
    let pdim_u = p_dim as u32;
    let rdim_u = r_dim as u32;
    let m_u = m as u32;
    // SAFETY: identical contract to `launch_cdist_f32` with 8-byte f64 values.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(x1)
            .arg(x2)
            .arg(out)
            .arg(&total_u)
            .arg(&pdim_u)
            .arg(&rdim_u)
            .arg(&m_u)
            .arg(&p)
            .arg(&mode)
            .launch(cfg)?;
    }
    Ok(())
}

/// Batched f32 cdist: `x1` is `[b, p_dim, m]`, `x2` is `[b, r_dim, m]`,
/// result is `[b, p_dim, r_dim]` (flattened). `p` must satisfy
/// [`cdist_supported_on_gpu`] (everything except `p == 0`). Returns the
/// resident output buffer of `b * p_dim * r_dim` elements.
#[allow(clippy::too_many_arguments)]
pub fn gpu_cdist_f32(
    x1: &CudaBuffer<f32>,
    x2: &CudaBuffer<f32>,
    b: usize,
    p_dim: usize,
    r_dim: usize,
    m: usize,
    p: f64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let mode = mode_for_p(p).ok_or(GpuError::Unsupported {
        op: "cdist (p=0)",
        dtype: "f32",
    })?;
    let total = b * p_dim * r_dim;
    let mut out = alloc_zeros_f32(total.max(1), device)?;
    launch_cdist_f32(
        x1.inner(),
        x2.inner(),
        out.inner_mut(),
        total,
        p_dim,
        r_dim,
        m,
        p,
        mode,
        device,
    )?;
    Ok(out)
}

/// Batched f64 cdist. The on-device f64 kernel covers `p == 1`, `p == 2`, and
/// `p == inf`; a non-`{1,2,inf}` `p` returns [`GpuError::Unsupported`] so the
/// caller falls back to the CPU path (f64 PTX lacks an accurate `pow`; see the
/// `CDIST_F64_PTX` note). Returns the resident output.
#[allow(clippy::too_many_arguments)]
pub fn gpu_cdist_f64(
    x1: &CudaBuffer<f64>,
    x2: &CudaBuffer<f64>,
    b: usize,
    p_dim: usize,
    r_dim: usize,
    m: usize,
    p: f64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let mode = mode_for_p(p).ok_or(GpuError::Unsupported {
        op: "cdist (p=0)",
        dtype: "f64",
    })?;
    if mode == MODE_GENERAL {
        return Err(GpuError::Unsupported {
            op: "cdist (general-p)",
            dtype: "f64",
        });
    }
    let total = b * p_dim * r_dim;
    let mut out = alloc_zeros_f64(total.max(1), device)?;
    launch_cdist_f64(
        x1.inner(),
        x2.inner(),
        out.inner_mut(),
        total,
        p_dim,
        r_dim,
        m,
        p,
        mode,
        device,
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transfer::{cpu_to_gpu, gpu_to_cpu};

    fn dev() -> GpuDevice {
        GpuDevice::new(0).expect("cuda device")
    }

    /// CPU reference matching `ferrotorch_core::ops::tensor_ops::cdist`.
    fn cpu_cdist(
        x1: &[f32],
        x2: &[f32],
        b: usize,
        pd: usize,
        rd: usize,
        m: usize,
        p: f64,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(b * pd * rd);
        for bb in 0..b {
            let o1 = bb * pd * m;
            let o2 = bb * rd * m;
            for i in 0..pd {
                for j in 0..rd {
                    let mut agg = 0.0f32;
                    let mut mx = 0.0f32;
                    for k in 0..m {
                        let diff = (x1[o1 + i * m + k] - x2[o2 + j * m + k]).abs();
                        if p.is_infinite() {
                            if diff > mx {
                                mx = diff;
                            }
                        } else {
                            agg += diff.powf(p as f32);
                        }
                    }
                    if p.is_infinite() {
                        out.push(mx);
                    } else {
                        out.push(agg.powf(1.0 / p as f32));
                    }
                }
            }
        }
        out
    }

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-4, "got {g} want {w}");
        }
    }

    #[test]
    fn cdist_f32_l2() {
        let d = dev();
        // x1 = [[0,0],[1,0],[0,1]], x2 = [[1,1]] ; torch.cdist(...,2)
        let x1 = vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        let x2 = vec![1.0f32, 1.0];
        let h1 = cpu_to_gpu(&x1, &d).unwrap();
        let h2 = cpu_to_gpu(&x2, &d).unwrap();
        let out = gpu_cdist_f32(&h1, &h2, 1, 3, 1, 2, 2.0, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        assert_close(&got[..3], &cpu_cdist(&x1, &x2, 1, 3, 1, 2, 2.0));
        // sqrt(2), 1, 1
        assert_close(&got[..3], &[2.0f32.sqrt(), 1.0, 1.0]);
    }

    #[test]
    fn cdist_f32_l1() {
        let d = dev();
        let x1 = vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        let x2 = vec![1.0f32, 1.0];
        let h1 = cpu_to_gpu(&x1, &d).unwrap();
        let h2 = cpu_to_gpu(&x2, &d).unwrap();
        let out = gpu_cdist_f32(&h1, &h2, 1, 3, 1, 2, 1.0, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        assert_close(&got[..3], &cpu_cdist(&x1, &x2, 1, 3, 1, 2, 1.0));
        // |0-1|+|0-1|=2, |1-1|+|0-1|=1, |0-1|+|1-1|=1
        assert_close(&got[..3], &[2.0, 1.0, 1.0]);
    }

    #[test]
    fn cdist_f32_linf() {
        let d = dev();
        let x1 = vec![0.0f32, 0.0, 3.0, 1.0];
        let x2 = vec![1.0f32, 5.0];
        let h1 = cpu_to_gpu(&x1, &d).unwrap();
        let h2 = cpu_to_gpu(&x2, &d).unwrap();
        let out = gpu_cdist_f32(&h1, &h2, 1, 2, 1, 2, f64::INFINITY, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        assert_close(&got[..2], &cpu_cdist(&x1, &x2, 1, 2, 1, 2, f64::INFINITY));
        // max(|0-1|,|0-5|)=5, max(|3-1|,|1-5|)=4
        assert_close(&got[..2], &[5.0, 4.0]);
    }

    #[test]
    fn cdist_f32_p3() {
        let d = dev();
        let x1 = vec![0.0f32, 0.0, 1.0, 2.0];
        let x2 = vec![1.0f32, 1.0];
        let h1 = cpu_to_gpu(&x1, &d).unwrap();
        let h2 = cpu_to_gpu(&x2, &d).unwrap();
        let out = gpu_cdist_f32(&h1, &h2, 1, 2, 1, 2, 3.0, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        // general-p path: compare to CPU ref (powf) with a looser fp tolerance
        let want = cpu_cdist(&x1, &x2, 1, 2, 1, 2, 3.0);
        for (g, w) in got[..2].iter().zip(&want) {
            assert!((g - w).abs() < 2e-3, "got {g} want {w}");
        }
    }

    #[test]
    fn cdist_f32_batched() {
        let d = dev();
        // b=2, p_dim=2, r_dim=2, m=2
        let x1: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let x2: Vec<f32> = (8..16).map(|i| i as f32).collect();
        let h1 = cpu_to_gpu(&x1, &d).unwrap();
        let h2 = cpu_to_gpu(&x2, &d).unwrap();
        let out = gpu_cdist_f32(&h1, &h2, 2, 2, 2, 2, 2.0, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        assert_close(&got[..8], &cpu_cdist(&x1, &x2, 2, 2, 2, 2, 2.0));
    }

    #[test]
    fn cdist_f64_l2() {
        let d = dev();
        let x1 = vec![0.0f64, 0.0, 1.0, 0.0, 0.0, 1.0];
        let x2 = vec![1.0f64, 1.0];
        let h1 = cpu_to_gpu(&x1, &d).unwrap();
        let h2 = cpu_to_gpu(&x2, &d).unwrap();
        let out = gpu_cdist_f64(&h1, &h2, 1, 3, 1, 2, 2.0, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        assert!((got[0] - 2.0f64.sqrt()).abs() < 1e-12);
        assert!((got[1] - 1.0).abs() < 1e-12);
        assert!((got[2] - 1.0).abs() < 1e-12);
    }
}
