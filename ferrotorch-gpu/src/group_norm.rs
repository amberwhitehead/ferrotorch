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
//! | REQ-1 (`gpu_group_norm_f32`) | SHIPPED | impl: `pub fn gpu_group_norm_f32 in group_norm.rs` mirrors upstream `GroupNormKernelImpl` at `aten/src/ATen/native/cuda/group_norm_kernel.cu:649`. Non-test consumer: `CudaBackendImpl::group_norm_f32 in backend_impl.rs` → `GpuBackend::group_norm_f32` → `ferrotorch-nn::GroupNorm::forward` GPU fast path (#1356) |
//! | REQ-2 (PTX template + ABI) | SHIPPED | `pub(crate) const GROUP_NORM_PTX in group_norm.rs` carries the three-pass mean / variance / affine PTX; ABI matches the launch site; verified by unit tests being numerically correct |
//! | REQ-3 (input validation) | SHIPPED | validation checks in `pub fn gpu_group_norm_f32 in group_norm.rs` (groups divisibility, input length, weight length, bias length, device ordinal) |
//! | REQ-4 (degenerate short-circuit) | SHIPPED | degenerate short-circuit in `gpu_group_norm_f32 in group_norm.rs` returns `alloc_zeros_f32(n, device)` for `n == 0 || channels == 0 || hw == 0` |
//! | REQ-5 (backend trait wiring) | SHIPPED | `fn group_norm_f32` on `GpuBackend` in `ferrotorch-core/src/gpu_dispatch.rs` (default `InvalidArgument`) overridden by `CudaBackendImpl::group_norm_f32 in backend_impl.rs` calling `crate::group_norm::gpu_group_norm_f32`. Consumer: `ferrotorch-nn::GroupNorm::forward` dispatches to `backend.group_norm_f32(...)` for CUDA input (#1357) |
//! | REQ-6 (`gpu_batch_norm_backward_f32`, #1449) | SHIPPED | impl: `pub fn gpu_batch_norm_backward_f32 in group_norm.rs` (PTX `BATCH_NORM_BACKWARD_PTX`) mirrors `aten/src/ATen/native/cuda/Normalization.cuh:388 batch_norm_backward_kernel`. Non-test consumer: `CudaBackendImpl::batch_norm_backward_f32 in backend_impl.rs` → `GpuBackend::batch_norm_backward_f32` → `ferrotorch-nn::BatchNorm{1,2,3}dBackward` + `InstanceNormBackward` GPU backward. Live-vs-torch grad parity (<1e-3) pinned by `ferrotorch-nn` `divergence_critic_batchnorm_gpu.rs`. |
//! | REQ-7 (`gpu_local_response_norm_f32` + backward, #1449) | SHIPPED | impl: `pub fn gpu_local_response_norm_f32` (PTX `LRN_FORWARD_PTX`) + `pub fn gpu_local_response_norm_backward_f32` (PTX `LRN_BACKWARD_PTX`) in `group_norm.rs` mirror `torch/nn/functional.py:3032-3046 local_response_norm`. Non-test consumer: `CudaBackendImpl::local_response_norm_f32` / `local_response_norm_backward_f32 in backend_impl.rs` → `ferrotorch-nn::LocalResponseNorm::forward` + `LocalResponseNormBackward`. Live-vs-torch fwd+grad parity (<1e-3) pinned by `divergence_local_response_norm_gpu_fwd_bwd_vs_torch`. |

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
    if groups == 0 || !channels.is_multiple_of(groups) {
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

// ---------------------------------------------------------------------------
// Softmax2d (#1451): channel-axis softmax over [N, C, H*W].
// ---------------------------------------------------------------------------

/// PTX source for the channel-axis softmax (`Softmax2d`) forward kernel.
///
/// One thread per `(n, p)` spatial position (grid is flat over `n * hw`
/// threads). Each thread walks the `c` channel values — which are strided
/// `hw` apart in the `[N, C, H*W]` row-major buffer — three times:
/// max-find, `exp` sum, then normalize. All accumulation is f32 with the
/// standard max-subtraction for numerical stability, matching
/// `torch.nn.Softmax2d` (softmax over `dim=1`).
///
/// ABI: `(in_ptr, out_ptr, total, channels, hw)` where `total = n * hw`.
#[cfg(feature = "cuda")]
pub(crate) const SOFTMAX2D_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry softmax2d_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 total,
    .param .u32 channels,
    .param .u32 hw
) {
    .reg .u32 %r_tid, %gid, %total_r, %c_r, %hw_r, %nidx, %pidx, %ci;
    .reg .u64 %in, %out, %base, %off, %el_per_n, %addr;
    .reg .f32 %val, %maxv, %sum, %e, %inv, %log2e;
    .reg .pred %oob, %lp;

    ld.param.u64 %in,  [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %total_r, [total];
    ld.param.u32 %c_r,     [channels];
    ld.param.u32 %hw_r,    [hw];

    // log2(e): exp(v) is computed as ex2.approx.f32(v * log2(e)).
    mov.f32 %log2e, 0f3FB8AA3B;

    // global thread id = ctaid.x * ntid.x + tid.x
    mov.u32 %r_tid, %tid.x;
    mov.u32 %gid, %ctaid.x;
    mov.u32 %nidx, %ntid.x;
    mul.lo.u32 %gid, %gid, %nidx;
    add.u32 %gid, %gid, %r_tid;

    setp.ge.u32 %oob, %gid, %total_r;
    @%oob bra DONE;

    // gid encodes (n, p): n = gid / hw, p = gid % hw.
    div.u32 %nidx, %gid, %hw_r;
    rem.u32 %pidx, %gid, %hw_r;

    // base element offset for (n, channel 0, p) = (n * channels) * hw + p
    cvt.u64.u32 %el_per_n, %c_r;
    cvt.u64.u32 %base, %nidx;
    mul.lo.u64 %base, %base, %el_per_n;   // n * channels
    cvt.u64.u32 %off, %hw_r;
    mul.lo.u64 %base, %base, %off;         // (n * channels) * hw
    cvt.u64.u32 %off, %pidx;
    add.u64 %base, %base, %off;            // + p
    shl.b64 %base, %base, 2;               // bytes
    add.u64 %base, %in, %base;

    // hw in bytes (stride between consecutive channels)
    cvt.u64.u32 %el_per_n, %hw_r;
    shl.b64 %el_per_n, %el_per_n, 2;

    // ---- Pass 1: max over channels ----
    mov.f32 %maxv, 0fFF800000;             // -inf
    mov.u32 %ci, 0;
    mov.u64 %addr, %base;
MX:
    setp.ge.u32 %lp, %ci, %c_r;
    @%lp bra MXD;
    ld.global.f32 %val, [%addr];
    max.f32 %maxv, %maxv, %val;
    add.u64 %addr, %addr, %el_per_n;
    add.u32 %ci, %ci, 1;
    bra MX;
    // ---- Pass 2: sum of exp(x - max) ----
MXD:
    mov.f32 %sum, 0f00000000;
    mov.u32 %ci, 0;
    mov.u64 %addr, %base;
SX:
    setp.ge.u32 %lp, %ci, %c_r;
    @%lp bra SXD;
    ld.global.f32 %val, [%addr];
    sub.f32 %val, %val, %maxv;
    mul.f32 %val, %val, %log2e;
    ex2.approx.f32 %e, %val;               // exp(x - max) = 2^((x-max)*log2 e)
    add.f32 %sum, %sum, %e;
    add.u64 %addr, %addr, %el_per_n;
    add.u32 %ci, %ci, 1;
    bra SX;
SXD:
    rcp.approx.f32 %inv, %sum;

    // ---- Pass 3: write exp(x - max) / sum ----
    // out base offset matches in base offset.
    cvt.u64.u32 %el_per_n, %c_r;
    cvt.u64.u32 %addr, %nidx;
    mul.lo.u64 %addr, %addr, %el_per_n;
    cvt.u64.u32 %off, %hw_r;
    mul.lo.u64 %addr, %addr, %off;
    cvt.u64.u32 %off, %pidx;
    add.u64 %addr, %addr, %off;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %out, %addr;
    cvt.u64.u32 %el_per_n, %hw_r;
    shl.b64 %el_per_n, %el_per_n, 2;

    mov.u32 %ci, 0;
    mov.u64 %off, %base;
WX:
    setp.ge.u32 %lp, %ci, %c_r;
    @%lp bra DONE;
    ld.global.f32 %val, [%off];
    sub.f32 %val, %val, %maxv;
    mul.f32 %val, %val, %log2e;
    ex2.approx.f32 %e, %val;
    mul.f32 %e, %e, %inv;
    st.global.f32 [%addr], %e;
    add.u64 %off, %off, %el_per_n;
    add.u64 %addr, %addr, %el_per_n;
    add.u32 %ci, %ci, 1;
    bra WX;
DONE:
    ret;
}
";

/// GPU forward `Softmax2d` over a `[N, C, H*W]`-laid-out f32 buffer.
///
/// Computes softmax over the channel axis (PyTorch `dim=1`):
/// `out[n,c,p] = exp(x[n,c,p] - m) / Σ_{c'} exp(x[n,c',p] - m)` where
/// `m = max_{c'} x[n,c',p]` and `p` ranges over the `hw = H*W` spatial
/// positions. Mirrors `torch.nn.Softmax2d`.
///
/// # Arguments
///
/// - `input` — flat `[N * C * H * W]` f32 buffer in row-major layout.
/// - `n` — batch size.
/// - `c` — channel count (softmax axis).
/// - `hw` — flattened spatial size `H * W`.
/// - `device` — owning GPU device.
///
/// # Errors
///
/// - [`GpuError::ShapeMismatch`] when `input.len() != n * c * hw`.
/// - [`GpuError::DeviceMismatch`] when `input` lives on another device.
/// - [`GpuError::PtxCompileFailed`] if the PTX module fails to compile.
/// - [`GpuError::Driver`] on launch failure.
#[cfg(feature = "cuda")]
pub fn gpu_softmax2d_f32(
    input: &CudaBuffer<f32>,
    n: usize,
    c: usize,
    hw: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let total_elems = n * c * hw;
    if input.len() != total_elems {
        return Err(GpuError::ShapeMismatch {
            op: "softmax2d",
            expected: vec![n, c, hw],
            got: vec![input.len()],
        });
    }
    if input.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: input.device_ordinal(),
        });
    }

    // Degenerate / empty shape: nothing to normalize, return zeros.
    if total_elems == 0 || c == 0 || hw == 0 {
        return alloc_zeros_f32(total_elems, device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        SOFTMAX2D_PTX,
        "softmax2d_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "softmax2d_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(total_elems, device)?;
    let total = (n * hw) as u32; // one thread per (n, p) spatial position
    let channels_u32 = c as u32;
    let hw_u32 = hw as u32;

    const BLOCK: u32 = 256;
    let grid = total.div_ceil(BLOCK).max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for `softmax2d_kernel` returned by
    //   `module_cache::get_or_compile`; ABI matches `(in_ptr, out_ptr, total,
    //   channels, hw)` exactly per `SOFTMAX2D_PTX`.
    // - `input.len() == n * c * hw` (validated above) and `input` lives on
    //   `device` (validated above). `out` was just allocated with the same
    //   length and cannot alias `input` (Rust borrow rules; `out` is `&mut`).
    // - Each thread handles one `(n, p)` position; the kernel guards
    //   `gid >= total` (= `n * hw`) and walks exactly `c` channel values that
    //   are `hw` elements apart, all in `[0, n*c*hw)`.
    // - No shared memory is used (the per-position reduction is sequential
    //   within a single thread).
    // - Stream sync is the caller's responsibility (matches every other
    //   kernel launch in this crate).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&total)
            .arg(&channels_u32)
            .arg(&hw_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// BatchNorm (#1449): per-channel normalize over (N, spatial).
// ---------------------------------------------------------------------------

/// PTX source for the BatchNorm forward kernel (f32).
///
/// One CUDA block per channel (grid: `(channels, 1, 1)`), 256 threads per
/// block. In **training** mode (`training != 0`) the block reduces over all
/// `batch * hw` elements of its channel — strided across the batch — to
/// compute the biased mean and variance, writes them back to `mean_ptr` /
/// `var_ptr` (so the host can update running stats), then normalizes and
/// applies the per-channel affine. In **eval** mode (`training == 0`) the
/// block reads the precomputed per-channel mean/var from `mean_ptr` /
/// `var_ptr` (the running statistics) and only normalizes + affines.
///
/// Element `i in [0, batch*hw)` for channel `c` maps to batch
/// `b = i / hw`, spatial position `s = i % hw`, global offset
/// `((b*channels + c)*hw + s)`. Mirrors the per-channel reduction in
/// `aten/src/ATen/native/Normalization.cpp`
/// `batch_norm_cpu_transform_input_template` (mean over `(N, *spatial)`,
/// biased variance, `y = γ*(x-μ)/sqrt(σ²+eps) + β`).
///
/// ABI: `(in_ptr, out_ptr, w_ptr, b_ptr, mean_ptr, var_ptr, batch,
/// channels, hw, eps, training)`.
#[cfg(feature = "cuda")]
pub(crate) const BATCH_NORM_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.shared .align 4 .f32 bdata[256];

.visible .entry batch_norm_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u64 w_ptr,
    .param .u64 b_ptr,
    .param .u64 mean_ptr,
    .param .u64 var_ptr,
    .param .u32 batch,
    .param .u32 channels,
    .param .u32 hw,
    .param .f32 eps,
    .param .u32 training
) {
    .reg .u32 %r_tid, %r_bdim, %r_c, %batch_r, %channels_r, %hw_r, %train_r;
    .reg .u32 %n_elem, %i, %bb, %ss, %half, %r_otid, %goff;
    .reg .u64 %in, %out, %w, %bv, %mp, %vp, %off, %sbase, %saddr;
    .reg .f32 %val, %mean, %var, %diff, %eps_r, %inv_std, %normed;
    .reg .f32 %wv, %bw, %result, %other, %n_f;
    .reg .pred %lp, %rp, %is_eval;

    ld.param.u64 %in,  [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %w,   [w_ptr];
    ld.param.u64 %bv,  [b_ptr];
    ld.param.u64 %mp,  [mean_ptr];
    ld.param.u64 %vp,  [var_ptr];
    ld.param.u32 %batch_r,    [batch];
    ld.param.u32 %channels_r, [channels];
    ld.param.u32 %hw_r,       [hw];
    ld.param.f32 %eps_r,      [eps];
    ld.param.u32 %train_r,    [training];

    mov.u64 %sbase, bdata;
    mov.u32 %r_c,    %ctaid.x;        // channel index
    mov.u32 %r_bdim, %ntid.x;
    mov.u32 %r_tid,  %tid.x;

    // n_elem = batch * hw  (elements reduced per channel)
    mul.lo.u32 %n_elem, %batch_r, %hw_r;
    cvt.rn.f32.u32 %n_f, %n_elem;

    // channel byte offset into the per-channel mean/var arrays
    cvt.u64.u32 %off, %r_c;
    shl.b64 %off, %off, 2;

    setp.eq.u32 %is_eval, %train_r, 0;
    @%is_eval bra EVAL;

    // ===== Training: reduce mean over (batch, hw) =====
    mov.f32 %mean, 0f00000000;
    mov.u32 %i, %r_tid;
BSM:
    setp.ge.u32 %lp, %i, %n_elem;
    @%lp bra BSMD;
    // global offset = ((b*channels + c)*hw + s) where b=i/hw, s=i%hw
    div.u32 %bb, %i, %hw_r;
    rem.u32 %ss, %i, %hw_r;
    mad.lo.u32 %goff, %bb, %channels_r, %r_c;   // b*channels + c
    mad.lo.u32 %goff, %goff, %hw_r, %ss;        // *hw + s   (NOTE: mul then add)
    cvt.u64.u32 %off, %goff;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    ld.global.f32 %val, [%off];
    add.f32 %mean, %mean, %val;
    add.u32 %i, %i, %r_bdim;
    bra BSM;
BSMD:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %mean;
    bar.sync 0;
    mov.u32 %half, %r_bdim;
BMR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra BMRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra BMRS;
    add.u32 %r_otid, %r_tid, %half;
    cvt.u64.u32 %off, %r_otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %mean, [%saddr];
    add.f32 %mean, %mean, %other;
    st.shared.f32 [%saddr], %mean;
BMRS:
    bar.sync 0;
    bra BMR;
BMRD:
    ld.shared.f32 %mean, [%sbase];
    div.approx.f32 %mean, %mean, %n_f;
    bar.sync 0;

    // ----- variance over (batch, hw) -----
    mov.f32 %var, 0f00000000;
    mov.u32 %i, %r_tid;
BSV:
    setp.ge.u32 %lp, %i, %n_elem;
    @%lp bra BSVD;
    div.u32 %bb, %i, %hw_r;
    rem.u32 %ss, %i, %hw_r;
    mad.lo.u32 %goff, %bb, %channels_r, %r_c;
    mad.lo.u32 %goff, %goff, %hw_r, %ss;
    cvt.u64.u32 %off, %goff;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    ld.global.f32 %val, [%off];
    sub.f32 %diff, %val, %mean;
    fma.rn.f32 %var, %diff, %diff, %var;
    add.u32 %i, %i, %r_bdim;
    bra BSV;
BSVD:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %var;
    bar.sync 0;
    mov.u32 %half, %r_bdim;
BVR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra BVRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra BVRS;
    add.u32 %r_otid, %r_tid, %half;
    cvt.u64.u32 %off, %r_otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %var, [%saddr];
    add.f32 %var, %var, %other;
    st.shared.f32 [%saddr], %var;
BVRS:
    bar.sync 0;
    bra BVR;
BVRD:
    ld.shared.f32 %var, [%sbase];
    div.approx.f32 %var, %var, %n_f;
    bar.sync 0;

    // thread 0 writes the biased mean/var back for the host running-stat update
    setp.ne.u32 %rp, %r_tid, 0;
    @%rp bra BSTATS_DONE;
    cvt.u64.u32 %off, %r_c;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %mp, %off;
    st.global.f32 [%saddr], %mean;
    add.u64 %saddr, %vp, %off;
    st.global.f32 [%saddr], %var;
BSTATS_DONE:
    bra NORM;

EVAL:
    // read running mean/var for this channel
    cvt.u64.u32 %off, %r_c;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %mp, %off;
    ld.global.f32 %mean, [%saddr];
    add.u64 %saddr, %vp, %off;
    ld.global.f32 %var, [%saddr];

NORM:
    // inv_std = 1 / sqrt(var + eps)
    add.f32 %var, %var, %eps_r;
    sqrt.approx.f32 %inv_std, %var;
    rcp.approx.f32 %inv_std, %inv_std;

    // per-channel affine
    cvt.u64.u32 %off, %r_c;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %w, %off;
    ld.global.f32 %wv, [%saddr];
    add.u64 %saddr, %bv, %off;
    ld.global.f32 %bw, [%saddr];

    // write normalized + affine for each element in the channel
    mov.u32 %i, %r_tid;
BNW:
    setp.ge.u32 %lp, %i, %n_elem;
    @%lp bra BNWD;
    div.u32 %bb, %i, %hw_r;
    rem.u32 %ss, %i, %hw_r;
    mad.lo.u32 %goff, %bb, %channels_r, %r_c;
    mad.lo.u32 %goff, %goff, %hw_r, %ss;
    cvt.u64.u32 %off, %goff;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    ld.global.f32 %val, [%off];
    sub.f32 %normed, %val, %mean;
    mul.f32 %normed, %normed, %inv_std;
    fma.rn.f32 %result, %wv, %normed, %bw;
    cvt.u64.u32 %off, %goff;
    shl.b64 %off, %off, 2;
    add.u64 %off, %out, %off;
    st.global.f32 [%off], %result;
    add.u32 %i, %i, %r_bdim;
    bra BNW;
BNWD:
    ret;
}
";

/// GPU forward BatchNorm over a `[batch, channels, hw]`-laid-out f32 buffer.
///
/// Computes `out = γ[c] * (x - μ_c) / sqrt(σ²_c + eps) + β[c]`, where the
/// per-channel mean / variance are taken over `(batch, hw)` in **training**
/// mode (`training == true`) or read from the provided running statistics in
/// **eval** mode. In training mode the computed biased mean / variance are
/// written into the returned `(mean, var)` buffers so the caller can update
/// the running statistics; in eval mode the caller-supplied `mean` / `var`
/// buffers are used unchanged.
///
/// # Arguments
///
/// - `input` — flat `[batch * channels * hw]` f32 buffer (`hw = ∏ spatial`).
/// - `weight` / `bias` — `[channels]` per-channel affine (ones / zeros when
///   the layer is non-affine, so the affine is the identity).
/// - `mean` / `var` — `[channels]` buffers: input running stats in eval mode,
///   output batch stats in training mode.
/// - `batch`, `channels`, `hw` — outer dims.
/// - `eps` — numerical-stability constant.
/// - `training` — `true` to compute batch stats, `false` to use `mean`/`var`.
/// - `device` — owning GPU device for all buffers.
///
/// # Errors
///
/// - [`GpuError::ShapeMismatch`] when any buffer length disagrees with the
///   declared dims.
/// - [`GpuError::DeviceMismatch`] when buffers live on a different device.
/// - [`GpuError::PtxCompileFailed`] if the PTX module fails to compile.
/// - [`GpuError::Driver`] on launch failure.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_batch_norm_f32(
    input: &CudaBuffer<f32>,
    weight: &CudaBuffer<f32>,
    bias: &CudaBuffer<f32>,
    mean: &mut CudaBuffer<f32>,
    var: &mut CudaBuffer<f32>,
    batch: usize,
    channels: usize,
    hw: usize,
    eps: f32,
    training: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let n = batch * channels * hw;
    if input.len() != n {
        return Err(GpuError::ShapeMismatch {
            op: "batch_norm",
            expected: vec![batch, channels, hw],
            got: vec![input.len()],
        });
    }
    for (name_len, buf_len) in [
        ("weight", weight.len()),
        ("bias", bias.len()),
        ("mean", mean.len()),
        ("var", var.len()),
    ] {
        let _ = name_len;
        if buf_len != channels {
            return Err(GpuError::ShapeMismatch {
                op: "batch_norm",
                expected: vec![channels],
                got: vec![buf_len],
            });
        }
    }
    if input.device_ordinal() != device.ordinal()
        || weight.device_ordinal() != device.ordinal()
        || bias.device_ordinal() != device.ordinal()
        || mean.device_ordinal() != device.ordinal()
        || var.device_ordinal() != device.ordinal()
    {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: input.device_ordinal(),
        });
    }

    // Degenerate / empty shape: nothing to normalize.
    if n == 0 || channels == 0 || hw == 0 || batch == 0 {
        return alloc_zeros_f32(n, device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        BATCH_NORM_PTX,
        "batch_norm_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "batch_norm_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let batch_u32 = batch as u32;
    let channels_u32 = channels as u32;
    let hw_u32 = hw as u32;
    let training_u32: u32 = if training { 1 } else { 0 };

    let cfg = LaunchConfig {
        grid_dim: (channels_u32.max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for `batch_norm_kernel` returned by
    //   `module_cache::get_or_compile`; ABI matches `(in_ptr, out_ptr, w_ptr,
    //   b_ptr, mean_ptr, var_ptr, batch, channels, hw, eps, training)` exactly
    //   per `BATCH_NORM_PTX`.
    // - `input.len() == batch * channels * hw` and
    //   `weight.len() == bias.len() == mean.len() == var.len() == channels`
    //   (all validated above). All buffers live on `device` (validated above).
    // - `out` was just allocated with `input.len()` elements and cannot alias
    //   `input`/`weight`/`bias`/`mean`/`var` (Rust borrow rules; `out` is &mut).
    //   `mean`/`var` are `&mut`: in training mode thread 0 of each block
    //   writes its channel's stat (`channels` disjoint slots, one block per
    //   channel — no data race); in eval mode they are read-only.
    // - Grid `(channels, 1, 1)` × block `(256, 1, 1)`: each block reads/writes
    //   exactly the `batch * hw` elements of channel `c` whose flat offset is
    //   `((b*channels + c)*hw + s) ∈ [0, batch*channels*hw)`.
    // - Shared memory: 256 * 4 = 1024 bytes (matches PTX `.shared bdata[256]`),
    //   one f32 per thread for the mean and variance block reductions.
    // - `eps: f32` / `training: u32` passed by-ref; cudarc copies them into the
    //   launch parameter buffer. Stream sync is the caller's responsibility.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(weight.inner())
            .arg(bias.inner())
            .arg(mean.inner_mut())
            .arg(var.inner_mut())
            .arg(&batch_u32)
            .arg(&channels_u32)
            .arg(&hw_u32)
            .arg(&eps)
            .arg(&training_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

/// PTX source for the BatchNorm **backward** kernel (#1449).
///
/// One CUDA block per channel (grid `(channels, 1, 1)`, block `(256, 1, 1)`).
/// Mirrors `aten/src/ATen/native/cuda/Normalization.cuh:388` `batch_norm_backward_kernel`:
///
/// ```text
/// N = batch * hw
/// mean, invstd:
///   train: recomputed from input over (batch, hw)  (biased var, +eps)
///   eval : mean = running_mean[c]; invstd = 1/sqrt(running_var[c] + eps)
/// reduce over (batch, hw):
///   grad_output_sum = Σ go
///   dot_p           = Σ (x - mean) * go
/// grad_mean   = grad_output_sum / N
/// proj_scale  = dot_p / N * invstd * invstd
/// grad_scale  = invstd * weight[c]
/// grad_input  = train ? (go - (x - mean)*proj_scale - grad_mean) * grad_scale
///                     : go * grad_scale
/// grad_weight[c] = dot_p * invstd          (thread 0)
/// grad_bias[c]   = grad_output_sum         (thread 0)
/// ```
///
/// ABI: `(in_ptr, go_ptr, gi_ptr, gw_ptr, gb_ptr, w_ptr, rmean_ptr, rvar_ptr,
/// batch, channels, hw, eps, training)`. `gw`/`gb` are `[channels]`; in the
/// non-affine case the caller passes an all-ones `weight` buffer (so
/// `grad_scale = invstd`) and discards the `gw`/`gb` outputs.
#[cfg(feature = "cuda")]
pub(crate) const BATCH_NORM_BACKWARD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.shared .align 4 .f32 bbdata[256];

.visible .entry batch_norm_backward_kernel(
    .param .u64 in_ptr,
    .param .u64 go_ptr,
    .param .u64 gi_ptr,
    .param .u64 gw_ptr,
    .param .u64 gb_ptr,
    .param .u64 w_ptr,
    .param .u64 rmean_ptr,
    .param .u64 rvar_ptr,
    .param .u32 batch,
    .param .u32 channels,
    .param .u32 hw,
    .param .f32 eps,
    .param .u32 training
) {
    .reg .u32 %r_tid, %r_bdim, %r_c, %batch_r, %channels_r, %hw_r, %train_r;
    .reg .u32 %n_elem, %i, %bb, %ss, %half, %r_otid, %goff;
    .reg .u64 %in, %go, %gi, %gw, %gb, %w, %rmp, %rvp, %off, %sbase, %saddr;
    .reg .f32 %val, %mean, %var, %diff, %eps_r, %inv_std, %g, %n_f;
    .reg .f32 %gsum, %dotp, %other, %wv, %grad_mean, %proj_scale, %grad_scale;
    .reg .f32 %proj, %result;
    .reg .pred %lp, %rp, %is_eval;

    ld.param.u64 %in,  [in_ptr];
    ld.param.u64 %go,  [go_ptr];
    ld.param.u64 %gi,  [gi_ptr];
    ld.param.u64 %gw,  [gw_ptr];
    ld.param.u64 %gb,  [gb_ptr];
    ld.param.u64 %w,   [w_ptr];
    ld.param.u64 %rmp, [rmean_ptr];
    ld.param.u64 %rvp, [rvar_ptr];
    ld.param.u32 %batch_r,    [batch];
    ld.param.u32 %channels_r, [channels];
    ld.param.u32 %hw_r,       [hw];
    ld.param.f32 %eps_r,      [eps];
    ld.param.u32 %train_r,    [training];

    mov.u64 %sbase, bbdata;
    mov.u32 %r_c,    %ctaid.x;
    mov.u32 %r_bdim, %ntid.x;
    mov.u32 %r_tid,  %tid.x;

    mul.lo.u32 %n_elem, %batch_r, %hw_r;
    cvt.rn.f32.u32 %n_f, %n_elem;

    setp.eq.u32 %is_eval, %train_r, 0;
    @%is_eval bra BB_EVAL;

    // ===== Training: recompute mean over (batch, hw) =====
    mov.f32 %mean, 0f00000000;
    mov.u32 %i, %r_tid;
BB_MS:
    setp.ge.u32 %lp, %i, %n_elem;
    @%lp bra BB_MSD;
    div.u32 %bb, %i, %hw_r;
    rem.u32 %ss, %i, %hw_r;
    mad.lo.u32 %goff, %bb, %channels_r, %r_c;
    mad.lo.u32 %goff, %goff, %hw_r, %ss;
    cvt.u64.u32 %off, %goff;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    ld.global.f32 %val, [%off];
    add.f32 %mean, %mean, %val;
    add.u32 %i, %i, %r_bdim;
    bra BB_MS;
BB_MSD:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %mean;
    bar.sync 0;
    mov.u32 %half, %r_bdim;
BB_MR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra BB_MRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra BB_MRS;
    add.u32 %r_otid, %r_tid, %half;
    cvt.u64.u32 %off, %r_otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %mean, [%saddr];
    add.f32 %mean, %mean, %other;
    st.shared.f32 [%saddr], %mean;
BB_MRS:
    bar.sync 0;
    bra BB_MR;
BB_MRD:
    ld.shared.f32 %mean, [%sbase];
    div.approx.f32 %mean, %mean, %n_f;
    bar.sync 0;

    // ----- variance over (batch, hw) -----
    mov.f32 %var, 0f00000000;
    mov.u32 %i, %r_tid;
BB_VS:
    setp.ge.u32 %lp, %i, %n_elem;
    @%lp bra BB_VSD;
    div.u32 %bb, %i, %hw_r;
    rem.u32 %ss, %i, %hw_r;
    mad.lo.u32 %goff, %bb, %channels_r, %r_c;
    mad.lo.u32 %goff, %goff, %hw_r, %ss;
    cvt.u64.u32 %off, %goff;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    ld.global.f32 %val, [%off];
    sub.f32 %diff, %val, %mean;
    fma.rn.f32 %var, %diff, %diff, %var;
    add.u32 %i, %i, %r_bdim;
    bra BB_VS;
BB_VSD:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %var;
    bar.sync 0;
    mov.u32 %half, %r_bdim;
BB_VR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra BB_VRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra BB_VRS;
    add.u32 %r_otid, %r_tid, %half;
    cvt.u64.u32 %off, %r_otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %var, [%saddr];
    add.f32 %var, %var, %other;
    st.shared.f32 [%saddr], %var;
BB_VRS:
    bar.sync 0;
    bra BB_VR;
BB_VRD:
    ld.shared.f32 %var, [%sbase];
    div.approx.f32 %var, %var, %n_f;
    bar.sync 0;
    add.f32 %var, %var, %eps_r;
    sqrt.approx.f32 %inv_std, %var;
    rcp.approx.f32 %inv_std, %inv_std;
    bra BB_REDUCE;

BB_EVAL:
    cvt.u64.u32 %off, %r_c;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %rmp, %off;
    ld.global.f32 %mean, [%saddr];
    add.u64 %saddr, %rvp, %off;
    ld.global.f32 %var, [%saddr];
    add.f32 %var, %var, %eps_r;
    sqrt.approx.f32 %inv_std, %var;
    rcp.approx.f32 %inv_std, %inv_std;

BB_REDUCE:
    // ----- reduce grad_output_sum (gsum) -----
    mov.f32 %gsum, 0f00000000;
    mov.u32 %i, %r_tid;
BB_GS:
    setp.ge.u32 %lp, %i, %n_elem;
    @%lp bra BB_GSD;
    div.u32 %bb, %i, %hw_r;
    rem.u32 %ss, %i, %hw_r;
    mad.lo.u32 %goff, %bb, %channels_r, %r_c;
    mad.lo.u32 %goff, %goff, %hw_r, %ss;
    cvt.u64.u32 %off, %goff;
    shl.b64 %off, %off, 2;
    add.u64 %off, %go, %off;
    ld.global.f32 %g, [%off];
    add.f32 %gsum, %gsum, %g;
    add.u32 %i, %i, %r_bdim;
    bra BB_GS;
BB_GSD:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %gsum;
    bar.sync 0;
    mov.u32 %half, %r_bdim;
BB_GR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra BB_GRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra BB_GRS;
    add.u32 %r_otid, %r_tid, %half;
    cvt.u64.u32 %off, %r_otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %gsum, [%saddr];
    add.f32 %gsum, %gsum, %other;
    st.shared.f32 [%saddr], %gsum;
BB_GRS:
    bar.sync 0;
    bra BB_GR;
BB_GRD:
    ld.shared.f32 %gsum, [%sbase];
    bar.sync 0;

    // ----- reduce dot_p = sum (x - mean) * go -----
    mov.f32 %dotp, 0f00000000;
    mov.u32 %i, %r_tid;
BB_DS:
    setp.ge.u32 %lp, %i, %n_elem;
    @%lp bra BB_DSD;
    div.u32 %bb, %i, %hw_r;
    rem.u32 %ss, %i, %hw_r;
    mad.lo.u32 %goff, %bb, %channels_r, %r_c;
    mad.lo.u32 %goff, %goff, %hw_r, %ss;
    cvt.u64.u32 %off, %goff;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %in, %off;
    ld.global.f32 %val, [%saddr];
    add.u64 %saddr, %go, %off;
    ld.global.f32 %g, [%saddr];
    sub.f32 %diff, %val, %mean;
    fma.rn.f32 %dotp, %diff, %g, %dotp;
    add.u32 %i, %i, %r_bdim;
    bra BB_DS;
BB_DSD:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %dotp;
    bar.sync 0;
    mov.u32 %half, %r_bdim;
BB_DR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra BB_DRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra BB_DRS;
    add.u32 %r_otid, %r_tid, %half;
    cvt.u64.u32 %off, %r_otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %dotp, [%saddr];
    add.f32 %dotp, %dotp, %other;
    st.shared.f32 [%saddr], %dotp;
BB_DRS:
    bar.sync 0;
    bra BB_DR;
BB_DRD:
    ld.shared.f32 %dotp, [%sbase];
    bar.sync 0;

    // ----- per-channel weight value, scales -----
    cvt.u64.u32 %off, %r_c;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %w, %off;
    ld.global.f32 %wv, [%saddr];

    div.approx.f32 %grad_mean, %gsum, %n_f;       // grad_output_sum / N
    div.approx.f32 %proj_scale, %dotp, %n_f;       // dot_p / N
    mul.f32 %proj_scale, %proj_scale, %inv_std;
    mul.f32 %proj_scale, %proj_scale, %inv_std;    // * invstd^2
    mul.f32 %grad_scale, %inv_std, %wv;            // invstd * w

    // ----- write grad_input -----
    mov.u32 %i, %r_tid;
BB_GIW:
    setp.ge.u32 %lp, %i, %n_elem;
    @%lp bra BB_GIWD;
    div.u32 %bb, %i, %hw_r;
    rem.u32 %ss, %i, %hw_r;
    mad.lo.u32 %goff, %bb, %channels_r, %r_c;
    mad.lo.u32 %goff, %goff, %hw_r, %ss;
    cvt.u64.u32 %off, %goff;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %go, %off;
    ld.global.f32 %g, [%saddr];
    @%is_eval bra BB_GIW_EVAL;
    // train: (go - (x - mean)*proj_scale - grad_mean) * grad_scale
    add.u64 %saddr, %in, %off;
    ld.global.f32 %val, [%saddr];
    sub.f32 %diff, %val, %mean;
    mul.f32 %proj, %diff, %proj_scale;
    sub.f32 %result, %g, %proj;
    sub.f32 %result, %result, %grad_mean;
    mul.f32 %result, %result, %grad_scale;
    bra BB_GIW_STORE;
BB_GIW_EVAL:
    // eval: go * grad_scale
    mul.f32 %result, %g, %grad_scale;
BB_GIW_STORE:
    add.u64 %saddr, %gi, %off;
    st.global.f32 [%saddr], %result;
    add.u32 %i, %i, %r_bdim;
    bra BB_GIW;
BB_GIWD:
    // ----- thread 0 writes grad_weight / grad_bias -----
    setp.ne.u32 %rp, %r_tid, 0;
    @%rp bra BB_DONE;
    mul.f32 %result, %dotp, %inv_std;     // grad_weight = dot_p * invstd
    cvt.u64.u32 %off, %r_c;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %gw, %off;
    st.global.f32 [%saddr], %result;
    add.u64 %saddr, %gb, %off;
    st.global.f32 [%saddr], %gsum;        // grad_bias = grad_output_sum
BB_DONE:
    ret;
}
";

/// GPU backward BatchNorm over a `[batch, channels, hw]`-laid-out f32 buffer
/// (#1449).
///
/// Computes `(grad_input, grad_weight, grad_bias)` on-device, mirroring
/// `aten/src/ATen/native/cuda/Normalization.cuh:388 batch_norm_backward_kernel`.
/// In **training** mode the per-channel mean / variance are recomputed from
/// `input` (no saved buffers required); in **eval** mode the supplied
/// `running_mean` / `running_var` are used. `weight` has length `channels`
/// (all-ones when the layer is non-affine, so `grad_scale = invstd`).
///
/// # Arguments
///
/// - `input` — flat `[batch * channels * hw]` f32 forward input.
/// - `grad_output` — flat `[batch * channels * hw]` upstream gradient.
/// - `weight` — `[channels]` affine scale (ones when non-affine).
/// - `running_mean` / `running_var` — `[channels]` running stats (read only in
///   eval mode; ignored in training mode but must be valid `[channels]`).
/// - `batch`, `channels`, `hw` — outer dims.
/// - `eps` — numerical-stability constant.
/// - `training` — `true` to recompute batch stats, `false` to use running stats.
/// - `device` — owning GPU device for all buffers.
///
/// Returns `(grad_input, grad_weight, grad_bias)`: `grad_input` has the input
/// shape, `grad_weight` / `grad_bias` have length `channels`.
///
/// # Errors
///
/// - [`GpuError::ShapeMismatch`] when any buffer length disagrees with the dims.
/// - [`GpuError::DeviceMismatch`] when buffers live on a different device.
/// - [`GpuError::PtxCompileFailed`] if the PTX module fails to compile.
/// - [`GpuError::Driver`] on launch failure.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_batch_norm_backward_f32(
    input: &CudaBuffer<f32>,
    grad_output: &CudaBuffer<f32>,
    weight: &CudaBuffer<f32>,
    running_mean: &CudaBuffer<f32>,
    running_var: &CudaBuffer<f32>,
    batch: usize,
    channels: usize,
    hw: usize,
    eps: f32,
    training: bool,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<f32>, CudaBuffer<f32>)> {
    let n = batch * channels * hw;
    for (buf_len, want) in [(input.len(), n), (grad_output.len(), n)] {
        if buf_len != want {
            return Err(GpuError::ShapeMismatch {
                op: "batch_norm_backward",
                expected: vec![batch, channels, hw],
                got: vec![buf_len],
            });
        }
    }
    for buf_len in [weight.len(), running_mean.len(), running_var.len()] {
        if buf_len != channels {
            return Err(GpuError::ShapeMismatch {
                op: "batch_norm_backward",
                expected: vec![channels],
                got: vec![buf_len],
            });
        }
    }
    if input.device_ordinal() != device.ordinal()
        || grad_output.device_ordinal() != device.ordinal()
        || weight.device_ordinal() != device.ordinal()
        || running_mean.device_ordinal() != device.ordinal()
        || running_var.device_ordinal() != device.ordinal()
    {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: input.device_ordinal(),
        });
    }

    if n == 0 || channels == 0 || hw == 0 || batch == 0 {
        return Ok((
            alloc_zeros_f32(n, device)?,
            alloc_zeros_f32(channels, device)?,
            alloc_zeros_f32(channels, device)?,
        ));
    }

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        BATCH_NORM_BACKWARD_PTX,
        "batch_norm_backward_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "batch_norm_backward_kernel",
                source: e,
            });
        }
    };

    let mut grad_input = alloc_zeros_f32(n, device)?;
    let mut grad_weight = alloc_zeros_f32(channels, device)?;
    let mut grad_bias = alloc_zeros_f32(channels, device)?;
    let batch_u32 = batch as u32;
    let channels_u32 = channels as u32;
    let hw_u32 = hw as u32;
    let training_u32: u32 = if training { 1 } else { 0 };

    let cfg = LaunchConfig {
        grid_dim: (channels_u32.max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };

    // SAFETY:
    // - `f` is the `batch_norm_backward_kernel` `CudaFunction` from
    //   `module_cache::get_or_compile`; its ABI matches the arg order
    //   `(in, go, gi, gw, gb, w, rmean, rvar, batch, channels, hw, eps,
    //   training)` declared in `BATCH_NORM_BACKWARD_PTX`.
    // - `input.len() == grad_output.len() == batch*channels*hw` and
    //   `weight.len() == running_mean.len() == running_var.len() == channels`
    //   (validated above). All buffers (incl. the freshly allocated `gi`/`gw`/
    //   `gb`) live on `device` (validated above).
    // - `gi`/`gw`/`gb` are `&mut` (disjoint, freshly allocated, cannot alias the
    //   `&` read-only inputs by Rust borrow rules). Grid `(channels,1,1)`: block
    //   `c` writes only channel `c`'s `batch*hw` `gi` slots plus the single
    //   `gw[c]`/`gb[c]` slot (thread 0), so writes are disjoint across blocks.
    // - Shared memory 256*4 = 1024 bytes matches PTX `.shared bbdata[256]` (one
    //   f32 per thread, reused across the mean/var/gsum/dotp block reductions).
    // - `eps: f32` / `training: u32` are passed by-ref; cudarc copies them into
    //   the launch parameter buffer. Stream sync is the caller's responsibility.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(grad_output.inner())
            .arg(grad_input.inner_mut())
            .arg(grad_weight.inner_mut())
            .arg(grad_bias.inner_mut())
            .arg(weight.inner())
            .arg(running_mean.inner())
            .arg(running_var.inner())
            .arg(&batch_u32)
            .arg(&channels_u32)
            .arg(&hw_u32)
            .arg(&eps)
            .arg(&training_u32)
            .launch(cfg)?;
    }

    Ok((grad_input, grad_weight, grad_bias))
}

/// PTX source for the LocalResponseNorm **forward** kernel (#1449).
///
/// One thread per output element of a `[batch, channels, spatial]`-laid-out
/// f32 buffer. Mirrors the `torch/nn/functional.py:3032-3046`
/// `local_response_norm` decomposition (square → windowed channel sum via
/// avg_pool → `* alpha + k` → `pow(beta)` → divide):
///
/// ```text
/// half = size / 2;  upper = size - half     (== (size+1)/2)
/// window for output channel c (in original-channel coords, zero-padded):
///   [c - half, c + upper)  clamped to [0, channels)
/// sq_sum   = Σ_{j in window} x[b,j,s]^2
/// denom    = (sq_sum / size) * alpha + k
/// out      = x[b,c,s] / pow(denom, beta)
/// denom_out[b,c,s] = denom    (saved for backward)
/// ```
///
/// ABI: `(in_ptr, out_ptr, denom_ptr, batch, channels, spatial, size, alpha,
/// beta, k)`. `denom_ptr` receives the per-element `denom` for the backward.
#[cfg(feature = "cuda")]
pub(crate) const LRN_FORWARD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry lrn_forward_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u64 denom_ptr,
    .param .u32 batch,
    .param .u32 channels,
    .param .u32 spatial,
    .param .u32 size,
    .param .f32 alpha,
    .param .f32 beta,
    .param .f32 k
) {
    .reg .u32 %r_tid, %r_bdim, %gid, %n, %bs_i, %c_i, %s_i, %half, %upper;
    .reg .u32 %cs, %ce, %j, %cspatial, %spatial_r, %channels_r, %jidx, %size_r, %base, %tmp;
    .reg .u64 %in, %out, %dn, %off, %addr;
    .reg .f32 %xv, %sq, %denom, %size_f, %alpha_r, %beta_r, %k_r, %powd, %lg;
    .reg .pred %p;

    ld.param.u64 %in,  [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %dn,  [denom_ptr];
    ld.param.u32 %channels_r, [channels];
    ld.param.u32 %spatial_r,  [spatial];
    ld.param.u32 %size_r,     [size];
    ld.param.f32 %alpha_r,    [alpha];
    ld.param.f32 %beta_r,     [beta];
    ld.param.f32 %k_r,        [k];

    // global thread id = ctaid.x * ntid.x + tid.x
    mov.u32 %r_tid,  %tid.x;
    mov.u32 %gid,    %ctaid.x;
    mov.u32 %r_bdim, %ntid.x;
    mad.lo.u32 %gid, %gid, %r_bdim, %r_tid;

    // cspatial = channels * spatial ; n = batch * cspatial
    mul.lo.u32 %cspatial, %channels_r, %spatial_r;
    ld.param.u32 %tmp, [batch];
    mul.lo.u32 %n, %tmp, %cspatial;
    setp.ge.u32 %p, %gid, %n;
    @%p bra LRN_F_DONE;

    // decode (bs_i, c_i, s_i) from gid
    div.u32 %bs_i, %gid, %cspatial;
    rem.u32 %tmp, %gid, %cspatial;
    div.u32 %c_i, %tmp, %spatial_r;
    rem.u32 %s_i, %tmp, %spatial_r;

    // half = size/2 ; upper = size - half
    shr.u32 %half, %size_r, 1;
    sub.u32 %upper, %size_r, %half;

    // c_start = (c_i >= half) ? c_i - half : 0
    setp.lt.u32 %p, %c_i, %half;
    sub.u32 %cs, %c_i, %half;
    @%p mov.u32 %cs, 0;
    // c_end = min(c_i + upper, channels)
    add.u32 %ce, %c_i, %upper;
    setp.lt.u32 %p, %channels_r, %ce;
    @%p mov.u32 %ce, %channels_r;

    // base of (bs_i, 0, s_i) = bs_i*cspatial + s_i
    mad.lo.u32 %base, %bs_i, %cspatial, %s_i;

    // sq_sum over window channels [cs, ce)
    mov.f32 %sq, 0f00000000;
    mov.u32 %j, %cs;
LRN_F_WIN:
    setp.ge.u32 %p, %j, %ce;
    @%p bra LRN_F_WIND;
    mad.lo.u32 %jidx, %j, %spatial_r, %base;   // bs*CS + j*S + s
    cvt.u64.u32 %off, %jidx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.f32 %xv, [%addr];
    fma.rn.f32 %sq, %xv, %xv, %sq;
    add.u32 %j, %j, 1;
    bra LRN_F_WIN;
LRN_F_WIND:
    // denom = (sq / size) * alpha + k
    cvt.rn.f32.u32 %size_f, %size_r;
    div.approx.f32 %denom, %sq, %size_f;
    fma.rn.f32 %denom, %denom, %alpha_r, %k_r;

    // out = x / pow(denom, beta) = x * exp2(beta * log2(denom))^-1 ;
    // here we compute pow(denom, beta) = exp2(beta * log2(denom)) then divide
    cvt.u64.u32 %off, %gid;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.f32 %xv, [%addr];
    lg2.approx.f32 %lg, %denom;
    mul.f32 %lg, %lg, %beta_r;
    ex2.approx.f32 %powd, %lg;        // denom^beta
    div.approx.f32 %xv, %xv, %powd;
    add.u64 %addr, %out, %off;
    st.global.f32 [%addr], %xv;
    // save denom for backward
    add.u64 %addr, %dn, %off;
    st.global.f32 [%addr], %denom;
LRN_F_DONE:
    ret;
}
";

/// PTX source for the LocalResponseNorm **backward** kernel (#1449).
///
/// One thread per input element. Mirrors the ferrotorch CPU
/// `LocalResponseNormBackward` VJP (which itself mirrors the
/// `torch/nn/functional.py` decomposition):
///
/// ```text
/// half = size/2 ; upper = size - half
/// term1 = denom[i]^(-beta) * go[i]
/// cross window for input channel i_c: c in [i_c+1-upper, i_c+half+1) clamped
/// cross_sum = Σ_c go[c] * x[c] * denom[c]^(-beta-1)
/// grad_input[i] = term1 - 2*beta*alpha/size * x[i] * cross_sum
/// ```
///
/// ABI: `(in_ptr, go_ptr, denom_ptr, gi_ptr, batch, channels, spatial, size,
/// alpha, beta)`.
#[cfg(feature = "cuda")]
pub(crate) const LRN_BACKWARD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry lrn_backward_kernel(
    .param .u64 in_ptr,
    .param .u64 go_ptr,
    .param .u64 denom_ptr,
    .param .u64 gi_ptr,
    .param .u32 batch,
    .param .u32 channels,
    .param .u32 spatial,
    .param .u32 size,
    .param .f32 alpha,
    .param .f32 beta
) {
    .reg .u32 %r_tid, %r_bdim, %gid, %n, %bs_i, %ic, %s_i, %half, %upper;
    .reg .u32 %cs, %ce, %c, %cspatial, %spatial_r, %channels_r, %cidx, %size_r, %base, %tmp;
    .reg .u64 %in, %go, %dn, %gi, %off, %addr;
    .reg .f32 %xv, %gov, %dv, %term1, %cross, %denom_i, %xi, %res;
    .reg .f32 %alpha_r, %beta_r, %size_f, %lg, %powc, %coef;
    .reg .pred %p;

    ld.param.u64 %in,  [in_ptr];
    ld.param.u64 %go,  [go_ptr];
    ld.param.u64 %dn,  [denom_ptr];
    ld.param.u64 %gi,  [gi_ptr];
    ld.param.u32 %channels_r, [channels];
    ld.param.u32 %spatial_r,  [spatial];
    ld.param.u32 %size_r,     [size];
    ld.param.f32 %alpha_r,    [alpha];
    ld.param.f32 %beta_r,     [beta];

    mov.u32 %r_tid,  %tid.x;
    mov.u32 %gid,    %ctaid.x;
    mov.u32 %r_bdim, %ntid.x;
    mad.lo.u32 %gid, %gid, %r_bdim, %r_tid;

    mul.lo.u32 %cspatial, %channels_r, %spatial_r;
    ld.param.u32 %tmp, [batch];
    mul.lo.u32 %n, %tmp, %cspatial;
    setp.ge.u32 %p, %gid, %n;
    @%p bra LRN_B_DONE;

    div.u32 %bs_i, %gid, %cspatial;
    rem.u32 %tmp, %gid, %cspatial;
    div.u32 %ic, %tmp, %spatial_r;
    rem.u32 %s_i, %tmp, %spatial_r;

    shr.u32 %half, %size_r, 1;
    sub.u32 %upper, %size_r, %half;

    // term1 = denom[gid]^(-beta) * go[gid]
    cvt.u64.u32 %off, %gid;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %dn, %off;
    ld.global.f32 %denom_i, [%addr];
    add.u64 %addr, %go, %off;
    ld.global.f32 %gov, [%addr];
    add.u64 %addr, %in, %off;
    ld.global.f32 %xi, [%addr];
    lg2.approx.f32 %lg, %denom_i;
    mul.f32 %lg, %lg, %beta_r;
    neg.f32 %lg, %lg;
    ex2.approx.f32 %term1, %lg;        // denom_i^(-beta)
    mul.f32 %term1, %term1, %gov;

    // cross window for input channel ic:  c in [ic+1-upper, ic+half+1)
    // c_start = (ic+1 >= upper) ? ic+1-upper : 0
    add.u32 %tmp, %ic, 1;
    setp.lt.u32 %p, %tmp, %upper;
    sub.u32 %cs, %tmp, %upper;
    @%p mov.u32 %cs, 0;
    // c_end = min(ic + half + 1, channels)
    add.u32 %ce, %ic, %half;
    add.u32 %ce, %ce, 1;
    setp.lt.u32 %p, %channels_r, %ce;
    @%p mov.u32 %ce, %channels_r;

    // base = bs_i*cspatial + s_i
    mad.lo.u32 %base, %bs_i, %cspatial, %s_i;

    mov.f32 %cross, 0f00000000;
    mov.u32 %c, %cs;
LRN_B_WIN:
    setp.ge.u32 %p, %c, %ce;
    @%p bra LRN_B_WIND;
    mad.lo.u32 %cidx, %c, %spatial_r, %base;     // bs*C*S + c*S + s
    cvt.u64.u32 %off, %cidx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %go, %off;
    ld.global.f32 %gov, [%addr];
    add.u64 %addr, %in, %off;
    ld.global.f32 %xv, [%addr];
    add.u64 %addr, %dn, %off;
    ld.global.f32 %dv, [%addr];
    // dv^(-beta-1) = exp2((-beta-1)*log2(dv))
    lg2.approx.f32 %lg, %dv;
    add.f32 %coef, %beta_r, 0f3F800000;      // beta + 1.0
    neg.f32 %coef, %coef;                      // -(beta+1)
    mul.f32 %lg, %lg, %coef;
    ex2.approx.f32 %powc, %lg;
    mul.f32 %res, %gov, %xv;
    fma.rn.f32 %cross, %res, %powc, %cross;
    add.u32 %c, %c, 1;
    bra LRN_B_WIN;
LRN_B_WIND:
    // coef2 = 2 * beta * alpha / size
    cvt.rn.f32.u32 %size_f, %size_r;
    mov.f32 %coef, 0f40000000;                 // 2.0
    mul.f32 %coef, %coef, %beta_r;
    mul.f32 %coef, %coef, %alpha_r;
    div.approx.f32 %coef, %coef, %size_f;
    // grad_input = term1 - coef2 * x[i] * cross
    mul.f32 %res, %coef, %xi;
    mul.f32 %res, %res, %cross;
    sub.f32 %res, %term1, %res;

    cvt.u64.u32 %off, %gid;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %gi, %off;
    st.global.f32 [%addr], %res;
LRN_B_DONE:
    ret;
}
";

/// GPU forward LocalResponseNorm over a `[batch, channels, spatial]`-laid-out
/// f32 buffer (#1449).
///
/// Mirrors `torch/nn/functional.py:3032-3046 local_response_norm`. Returns
/// `(output, denom)` where `denom[i] = (Σ_window x²/size)*alpha + k` is saved
/// for the backward pass. One thread per element.
///
/// # Errors
///
/// - [`GpuError::ShapeMismatch`] when `input.len() != batch*channels*spatial`.
/// - [`GpuError::DeviceMismatch`] when `input` lives on a different device.
/// - [`GpuError::PtxCompileFailed`] if the PTX module fails to compile.
/// - [`GpuError::Driver`] on launch failure.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_local_response_norm_f32(
    input: &CudaBuffer<f32>,
    batch: usize,
    channels: usize,
    spatial: usize,
    size: usize,
    alpha: f32,
    beta: f32,
    k: f32,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<f32>)> {
    let n = batch * channels * spatial;
    if input.len() != n {
        return Err(GpuError::ShapeMismatch {
            op: "local_response_norm",
            expected: vec![batch, channels, spatial],
            got: vec![input.len()],
        });
    }
    if input.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: input.device_ordinal(),
        });
    }
    if n == 0 || channels == 0 || size == 0 {
        return Ok((alloc_zeros_f32(n, device)?, alloc_zeros_f32(n, device)?));
    }

    let ctx = device.context();
    let stream = device.stream();
    let f = match crate::module_cache::get_or_compile(
        ctx,
        LRN_FORWARD_PTX,
        "lrn_forward_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "lrn_forward_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let mut denom = alloc_zeros_f32(n, device)?;
    let batch_u32 = batch as u32;
    let channels_u32 = channels as u32;
    let spatial_u32 = spatial as u32;
    let size_u32 = size as u32;
    let block = 256u32;
    let grid = (n as u32).div_ceil(block);
    let cfg = LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY:
    // - `f` is the `lrn_forward_kernel` `CudaFunction`; ABI matches
    //   `(in, out, denom, batch, channels, spatial, size, alpha, beta, k)` per
    //   `LRN_FORWARD_PTX`.
    // - `input.len() == n` (validated). `out`/`denom` are freshly allocated with
    //   `n` elements on `device`, cannot alias `input` (Rust borrow rules: `&mut`
    //   vs `&`). Each thread `gid < n` writes exactly `out[gid]`/`denom[gid]`
    //   (disjoint) and reads only window elements within `[0, n)`.
    // - No shared memory. Grid covers all `n` elements; threads with `gid >= n`
    //   early-return before any memory access.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(denom.inner_mut())
            .arg(&batch_u32)
            .arg(&channels_u32)
            .arg(&spatial_u32)
            .arg(&size_u32)
            .arg(&alpha)
            .arg(&beta)
            .arg(&k)
            .launch(cfg)?;
    }

    Ok((out, denom))
}

/// GPU backward LocalResponseNorm (#1449).
///
/// Mirrors the ferrotorch CPU `LocalResponseNormBackward` VJP. Consumes the
/// `denom` buffer saved by [`gpu_local_response_norm_f32`]. One thread per
/// element. Returns `grad_input` (input shape).
///
/// # Errors
///
/// - [`GpuError::ShapeMismatch`] when any buffer length disagrees with the dims.
/// - [`GpuError::DeviceMismatch`] when buffers live on a different device.
/// - [`GpuError::PtxCompileFailed`] if the PTX module fails to compile.
/// - [`GpuError::Driver`] on launch failure.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_local_response_norm_backward_f32(
    input: &CudaBuffer<f32>,
    grad_output: &CudaBuffer<f32>,
    denom: &CudaBuffer<f32>,
    batch: usize,
    channels: usize,
    spatial: usize,
    size: usize,
    alpha: f32,
    beta: f32,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let n = batch * channels * spatial;
    for buf_len in [input.len(), grad_output.len(), denom.len()] {
        if buf_len != n {
            return Err(GpuError::ShapeMismatch {
                op: "local_response_norm_backward",
                expected: vec![batch, channels, spatial],
                got: vec![buf_len],
            });
        }
    }
    if input.device_ordinal() != device.ordinal()
        || grad_output.device_ordinal() != device.ordinal()
        || denom.device_ordinal() != device.ordinal()
    {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: input.device_ordinal(),
        });
    }
    if n == 0 || channels == 0 || size == 0 {
        return alloc_zeros_f32(n, device);
    }

    let ctx = device.context();
    let stream = device.stream();
    let f = match crate::module_cache::get_or_compile(
        ctx,
        LRN_BACKWARD_PTX,
        "lrn_backward_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "lrn_backward_kernel",
                source: e,
            });
        }
    };

    let mut grad_input = alloc_zeros_f32(n, device)?;
    let batch_u32 = batch as u32;
    let channels_u32 = channels as u32;
    let spatial_u32 = spatial as u32;
    let size_u32 = size as u32;
    let block = 256u32;
    let grid = (n as u32).div_ceil(block);
    let cfg = LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY:
    // - `f` is the `lrn_backward_kernel` `CudaFunction`; ABI matches
    //   `(in, go, denom, gi, batch, channels, spatial, size, alpha, beta)` per
    //   `LRN_BACKWARD_PTX`.
    // - `input.len() == grad_output.len() == denom.len() == n` (validated).
    //   `grad_input` is freshly allocated with `n` elements on `device` and
    //   cannot alias the `&` inputs (`&mut` vs `&`). Each thread `gid < n` writes
    //   only `grad_input[gid]` (disjoint) and reads window elements in `[0, n)`.
    // - No shared memory; threads with `gid >= n` early-return before access.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(grad_output.inner())
            .arg(denom.inner())
            .arg(grad_input.inner_mut())
            .arg(&batch_u32)
            .arg(&channels_u32)
            .arg(&spatial_u32)
            .arg(&size_u32)
            .arg(&alpha)
            .arg(&beta)
            .launch(cfg)?;
    }

    Ok(grad_input)
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

    /// Reference CPU channel-axis softmax over `[N, C, H*W]` (PyTorch
    /// `Softmax2d`, softmax over `dim=1`).
    fn cpu_softmax2d_ref(x: &[f32], n: usize, c: usize, hw: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n * c * hw];
        for ni in 0..n {
            for p in 0..hw {
                let mut maxv = f32::NEG_INFINITY;
                for ci in 0..c {
                    let v = x[(ni * c + ci) * hw + p];
                    if v > maxv {
                        maxv = v;
                    }
                }
                let mut sum = 0.0f32;
                for ci in 0..c {
                    sum += (x[(ni * c + ci) * hw + p] - maxv).exp();
                }
                for ci in 0..c {
                    let idx = (ni * c + ci) * hw + p;
                    out[idx] = (x[idx] - maxv).exp() / sum;
                }
            }
        }
        out
    }

    #[test]
    fn softmax2d_matches_cpu_small() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let n = 2;
        let c = 5;
        let hw = 3 * 4; // 3x4 spatial
        let total = n * c * hw;
        let x: Vec<f32> = (0..total).map(|k| ((k % 11) as f32) * 0.3 - 1.4).collect();

        let xg = cpu_to_gpu(&x, &device).unwrap();
        let yg = gpu_softmax2d_f32(&xg, n, c, hw, &device).unwrap();
        let got = gpu_to_cpu(&yg, &device).unwrap();
        let expected = cpu_softmax2d_ref(&x, n, c, hw);
        assert_eq!(got.len(), expected.len());
        let mut max_abs = 0.0f32;
        for (a, e) in got.iter().zip(expected.iter()) {
            let d = (a - e).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        assert!(
            max_abs < 1e-4,
            "softmax2d gpu vs cpu max abs diff = {max_abs}"
        );
    }

    #[test]
    fn softmax2d_columns_sum_to_one() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let n = 1;
        let c = 8;
        let hw = 4;
        let total = n * c * hw;
        let x: Vec<f32> = (0..total)
            .map(|k| ((k as f32) * 0.05).cos() * 3.0)
            .collect();
        let xg = cpu_to_gpu(&x, &device).unwrap();
        let yg = gpu_softmax2d_f32(&xg, n, c, hw, &device).unwrap();
        let got = gpu_to_cpu(&yg, &device).unwrap();
        // Every (n, p) column over the channel axis must sum to 1.
        for ni in 0..n {
            for p in 0..hw {
                let mut s = 0.0f32;
                for ci in 0..c {
                    s += got[(ni * c + ci) * hw + p];
                }
                assert!((s - 1.0).abs() < 1e-4, "column sum = {s}, expected 1.0");
            }
        }
    }

    #[test]
    fn softmax2d_validates_length() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        // Buffer length disagrees with declared n*c*hw.
        let x = vec![0.0f32; 10];
        let xg = cpu_to_gpu(&x, &device).unwrap();
        let res = gpu_softmax2d_f32(&xg, 2, 3, 4, &device); // expects 24
        assert!(matches!(res, Err(GpuError::ShapeMismatch { .. })));
    }

    /// Reference CPU BatchNorm over `[batch, channels, hw]` with per-channel
    /// γ, β. `training` selects batch-stats (biased mean/var over (N, hw)) vs.
    /// the supplied `mean`/`var` running statistics.
    #[allow(clippy::too_many_arguments)]
    fn cpu_batch_norm_ref(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mean_in: &[f32],
        var_in: &[f32],
        batch: usize,
        channels: usize,
        hw: usize,
        eps: f32,
        training: bool,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * channels * hw];
        let n = (batch * hw) as f64;
        for c in 0..channels {
            let (mean, var) = if training {
                let mut s = 0.0f64;
                for b in 0..batch {
                    for p in 0..hw {
                        s += x[(b * channels + c) * hw + p] as f64;
                    }
                }
                let m = s / n;
                let mut vs = 0.0f64;
                for b in 0..batch {
                    for p in 0..hw {
                        let d = x[(b * channels + c) * hw + p] as f64 - m;
                        vs += d * d;
                    }
                }
                (m as f32, (vs / n) as f32)
            } else {
                (mean_in[c], var_in[c])
            };
            let inv_std = 1.0 / (var + eps).sqrt();
            for b in 0..batch {
                for p in 0..hw {
                    let i = (b * channels + c) * hw + p;
                    out[i] = gamma[c] * (x[i] - mean) * inv_std + beta[c];
                }
            }
        }
        out
    }

    #[test]
    fn batch_norm_training_matches_cpu() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let batch = 4;
        let channels = 8;
        let hw = 3 * 3;
        let eps = 1e-5_f32;
        let n = batch * channels * hw;
        let x: Vec<f32> = (0..n).map(|k| ((k % 17) as f32) * 0.13 - 0.9).collect();
        let gamma: Vec<f32> = (0..channels).map(|k| 1.0 + 0.07 * (k as f32)).collect();
        let beta: Vec<f32> = (0..channels).map(|k| -0.2 + 0.03 * (k as f32)).collect();

        let xg = cpu_to_gpu(&x, &device).unwrap();
        let gg = cpu_to_gpu(&gamma, &device).unwrap();
        let bg = cpu_to_gpu(&beta, &device).unwrap();
        let mut mean_g = cpu_to_gpu(&vec![0.0f32; channels], &device).unwrap();
        let mut var_g = cpu_to_gpu(&vec![1.0f32; channels], &device).unwrap();

        let yg = gpu_batch_norm_f32(
            &xg,
            &gg,
            &bg,
            &mut mean_g,
            &mut var_g,
            batch,
            channels,
            hw,
            eps,
            true,
            &device,
        )
        .unwrap();
        let got = gpu_to_cpu(&yg, &device).unwrap();
        let expected =
            cpu_batch_norm_ref(&x, &gamma, &beta, &[], &[], batch, channels, hw, eps, true);
        let mut max_abs = 0.0f32;
        for (a, e) in got.iter().zip(expected.iter()) {
            max_abs = max_abs.max((a - e).abs());
        }
        assert!(
            max_abs < 1e-4,
            "batch_norm train gpu vs cpu max|Δ| = {max_abs}"
        );

        // The kernel must also have written the biased batch mean/var back.
        let mean_back = gpu_to_cpu(&mean_g, &device).unwrap();
        let var_back = gpu_to_cpu(&var_g, &device).unwrap();
        for c in 0..channels {
            let mut s = 0.0f64;
            for b in 0..batch {
                for p in 0..hw {
                    s += x[(b * channels + c) * hw + p] as f64;
                }
            }
            let m = (s / (batch * hw) as f64) as f32;
            assert!(
                (mean_back[c] - m).abs() < 1e-4,
                "mean[{c}] {} vs {m}",
                mean_back[c]
            );
            assert!(var_back[c] >= 0.0);
        }
    }

    #[test]
    fn batch_norm_eval_uses_running_stats() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let batch = 2;
        let channels = 5;
        let hw = 4;
        let eps = 1e-5_f32;
        let n = batch * channels * hw;
        let x: Vec<f32> = (0..n).map(|k| ((k as f32) * 0.05).sin()).collect();
        let gamma: Vec<f32> = (0..channels).map(|k| 0.5 + 0.2 * (k as f32)).collect();
        let beta: Vec<f32> = (0..channels).map(|k| 0.1 * (k as f32)).collect();
        let running_mean: Vec<f32> = (0..channels).map(|k| 0.05 * (k as f32) - 0.1).collect();
        let running_var: Vec<f32> = (0..channels).map(|k| 0.8 + 0.1 * (k as f32)).collect();

        let xg = cpu_to_gpu(&x, &device).unwrap();
        let gg = cpu_to_gpu(&gamma, &device).unwrap();
        let bg = cpu_to_gpu(&beta, &device).unwrap();
        let mut mean_g = cpu_to_gpu(&running_mean, &device).unwrap();
        let mut var_g = cpu_to_gpu(&running_var, &device).unwrap();

        let yg = gpu_batch_norm_f32(
            &xg,
            &gg,
            &bg,
            &mut mean_g,
            &mut var_g,
            batch,
            channels,
            hw,
            eps,
            false,
            &device,
        )
        .unwrap();
        let got = gpu_to_cpu(&yg, &device).unwrap();
        let expected = cpu_batch_norm_ref(
            &x,
            &gamma,
            &beta,
            &running_mean,
            &running_var,
            batch,
            channels,
            hw,
            eps,
            false,
        );
        let mut max_abs = 0.0f32;
        for (a, e) in got.iter().zip(expected.iter()) {
            max_abs = max_abs.max((a - e).abs());
        }
        assert!(
            max_abs < 1e-4,
            "batch_norm eval gpu vs cpu max|Δ| = {max_abs}"
        );
    }

    #[test]
    fn batch_norm_validates_lengths() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let x = vec![0.0f32; 10]; // not batch*channels*hw
        let xg = cpu_to_gpu(&x, &device).unwrap();
        let gg = cpu_to_gpu(&[1.0f32; 3], &device).unwrap();
        let bg = cpu_to_gpu(&[0.0f32; 3], &device).unwrap();
        let mut mean_g = cpu_to_gpu(&[0.0f32; 3], &device).unwrap();
        let mut var_g = cpu_to_gpu(&[1.0f32; 3], &device).unwrap();
        let res = gpu_batch_norm_f32(
            &xg,
            &gg,
            &bg,
            &mut mean_g,
            &mut var_g,
            2,
            3,
            4,
            1e-5,
            true,
            &device,
        );
        assert!(matches!(res, Err(GpuError::ShapeMismatch { .. })));
    }
}
