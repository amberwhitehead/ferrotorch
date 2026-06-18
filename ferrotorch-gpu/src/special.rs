//! GPU forward kernels for the orthogonal-polynomial special functions
//! (`torch.special.{chebyshev,hermite,laguerre,legendre}_polynomial_*`).
//!
//! Each kernel evaluates the n-th degree basis polynomial pointwise at every
//! element of a contiguous input buffer, running the standard three-term
//! recurrence **on-device** (one thread per element). There is no host
//! round-trip: the input buffer stays in VRAM, the recurrence runs in the
//! thread's registers, and the output buffer stays in VRAM (R-CODE-4).
//!
//! # Math contract — mirror the ferrotorch CPU path, not torch edge cases
//!
//! These kernels reproduce the exact three-term recurrences in
//! `ferrotorch_core::special` (the `chebyshev_t`, `chebyshev_u`,
//! `chebyshev_v`, `chebyshev_w`, `hermite_h`, `hermite_he`, `laguerre_l`,
//! `legendre_p` scalar evaluators), so the GPU result is bit-for-relevant-
//! tolerance identical to the CPU result. They deliberately do NOT reproduce
//! PyTorch's CUDA edge-case shortcuts (the `|x| == 1` closed forms, the
//! `cos(n*acos(x))` / `sin((n+1)*acos(x))` shortcuts for high `n`, the
//! `n < 0 -> 0` guard, and the NaN early-exit) found in
//! `aten/src/ATen/native/Math.h` `chebyshev_polynomial_t_forward` et al.
//! Reproducing those here would make the GPU path disagree with the
//! ferrotorch CPU path — a silent CPU/GPU divergence. The ferrotorch-CPU vs.
//! torch-CUDA edge-case gap is a pre-existing CPU-side divergence tracked
//! separately; the GPU kernel's job is CPU/GPU agreement.
//!
//! Upstream recurrence reference (the core loop these mirror, minus the
//! edge-case shortcuts):
//!   - `aten/src/ATen/native/Math.h:2861-2869` (chebyshev T: `r = 2x q - p`)
//!   - `aten/src/ATen/native/Math.h:3072-3080` (hermite H: `r = 2x q - 2k p`)
//!   - `aten/src/ATen/native/Math.h:3113-3121` (hermite He: `r = x q - k p`)
//!   - `aten/src/ATen/native/Math.h:3149-3157` (laguerre: `((2k+1-x) q - k p)/(k+1)`)
//!   - `aten/src/ATen/native/Math.h:3189-3197` (legendre: `((2k+1) x q - k p)/(k+1)`)
//!
//! # Kernel layout
//!
//! - Grid: `((total + 255) / 256, 1, 1)`. One thread per element.
//! - Block: `(256, 1, 1)`. No shared memory.
//!
//! The chebyshev kernel handles all four kinds (T/U/V/W) AND their shifted
//! variants through three parameters: `(seed_a, seed_b, shift)`. The thread
//! computes `xx = shift != 0 ? (2*x - 1) : x`, seeds `q1 = seed_a*xx + seed_b`
//! (T: a=1,b=0 → q1=xx; U: a=2,b=0 → q1=2xx; V: a=2,b=-1 → q1=2xx-1;
//! W: a=2,b=1 → q1=2xx+1), then runs `r = 2*xx*q - p`. This is the exact
//! ferrotorch CPU seeding (`chebyshev_{t,u,v,w}` scalar fns) and the shifted
//! variants are `chebyshev_*(2x-1)` per `shifted_chebyshev_polynomial_*`.
//!
//! ## REQ status (per `.design/ferrotorch-gpu/special.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (chebyshev T/U/V/W + shifted, f32/f64) | SHIPPED | `pub fn gpu_chebyshev_poly_f32`/`_f64 in special.rs`; consumer `CudaBackendImpl::chebyshev_polynomial_t_f32` … `shifted_chebyshev_polynomial_w_f64 in backend_impl.rs`, dispatched from `ferrotorch_core::special::chebyshev_polynomial_t` GPU branch |
//! | REQ-2 (hermite H/He, f32/f64) | SHIPPED | `pub fn gpu_hermite_h_poly_*`/`gpu_hermite_he_poly_* in special.rs`; consumer `CudaBackendImpl::hermite_polynomial_h_f32` etc. |
//! | REQ-3 (laguerre, f32/f64) | SHIPPED | `pub fn gpu_laguerre_poly_* in special.rs`; consumer `CudaBackendImpl::laguerre_polynomial_l_f32`/`_f64` |
//! | REQ-4 (legendre, f32/f64) | SHIPPED | `pub fn gpu_legendre_poly_* in special.rs`; consumer `CudaBackendImpl::legendre_polynomial_p_f32`/`_f64` |
//! | REQ-5 (re-export + consumer wiring) | SHIPPED | `pub use special::* in lib.rs`; consumer the `CudaBackendImpl` trait-method bodies registered via `init_cuda_backend`, which `ferrotorch_core::special` GPU branches dispatch through |
//! | REQ-7 (Airy Ai + Hurwitz zeta GPU f32, #1651 GPU tail) | SHIPPED | impl `pub fn gpu_airy_ai_f32` (`AIRY_AI_F32_PTX`) + `pub fn gpu_zeta_f32` (`ZETA_F32_PTX`, via `launch_binary_elementwise_f32`) `in special.rs`, porting `airy_ai_forward`/`zeta` at upstream `cuda/Math.cuh:1280-1459, 299-383` as fixed-count unrolls (bounded loops); consumer `CudaBackendImpl::airy_ai_f32`/`zeta_f32 in backend_impl.rs`, dispatched from the `special_gpu_simple`/`special_gpu_binary` GPU branches of `ferrotorch_core::special::airy_ai`/`zeta`. f64/bf16/f16 CUDA reject `NotImplementedOnCuda` |

#[cfg(feature = "cuda")]
use cudarc::driver::{LaunchConfig, PushKernelArg};

#[cfg(feature = "cuda")]
use crate::buffer::CudaBuffer;
#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use crate::error::{GpuError, GpuResult};
#[cfg(feature = "cuda")]
use crate::transfer::{alloc_zeros_f32, alloc_zeros_f64};

// ---------------------------------------------------------------------------
// PTX — f32
// ---------------------------------------------------------------------------
//
// Every kernel shares the same prologue: compute the global thread index
// `idx`, bounds-check against `total`, load `x = in[idx]`, then evaluate the
// recurrence in registers and store `out[idx]`. f32 element offset is
// `idx << 2` (4 bytes); f64 is `idx << 3` (8 bytes).

/// Chebyshev (all four kinds + shifted) — f32.
///
/// ABI: `(in, out, n, total, seed_a, seed_b, shift)`.
///   `xx = shift ? 2*x - 1 : x`
///   `q1 = seed_a*xx + seed_b`  (T: (1,0); U: (2,0); V: (2,-1); W: (2,1))
///   `r  = 2*xx*q - p`,  iterated `n - 1` times from `q0 = 1`, `q = q1`.
/// `n == 0 -> 1`, `n == 1 -> q1`.
#[cfg(feature = "cuda")]
pub(crate) const CHEBYSHEV_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry chebyshev_poly_f32(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 total,
    .param .f32 seed_a,
    .param .f32 seed_b,
    .param .u32 shift
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r, %n_r, %shift_r, %k;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %x, %xx, %p, %q, %r, %two_xx, %sa, %sb, %one;
    .reg .pred %oob, %is0, %is1, %loop, %do_shift;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %n_r,     [n];
    ld.param.u32 %total_r, [total];
    ld.param.f32 %sa,      [seed_a];
    ld.param.f32 %sb,      [seed_b];
    ld.param.u32 %shift_r, [shift];

    mov.f32 %one, 0f3F800000;       // 1.0

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %x, [%addr];

    // xx = shift ? 2x - 1 : x
    mov.f32 %xx, %x;
    setp.ne.u32 %do_shift, %shift_r, 0;
    @!%do_shift bra AFTER_SHIFT;
    add.f32 %xx, %x, %x;
    sub.f32 %xx, %xx, %one;          // 2x - 1.0
AFTER_SHIFT:

    // q0 = 1
    mov.f32 %p, %one;
    // q1 = seed_a*xx + seed_b
    fma.rn.f32 %q, %sa, %xx, %sb;
    // two_xx = 2*xx
    add.f32 %two_xx, %xx, %xx;

    // n == 0 -> result 1
    setp.eq.u32 %is0, %n_r, 0;
    @%is0 bra STORE_P;
    // n == 1 -> result q1
    setp.eq.u32 %is1, %n_r, 1;
    mov.f32 %r, %q;
    @%is1 bra STORE_R;

    // loop k = 2 .. n  (n-1 iterations): r = 2*xx*q - p
    mov.u32 %k, 2;
LOOP:
    setp.gt.u32 %loop, %k, %n_r;
    @%loop bra STORE_R;
    mul.f32 %r, %two_xx, %q;         // r = 2xx*q
    sub.f32 %r, %r, %p;              // r = 2xx*q - p
    mov.f32 %p, %q;
    mov.f32 %q, %r;
    add.u32 %k, %k, 1;
    bra LOOP;

STORE_P:
    mov.f32 %r, %p;
STORE_R:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %r;
DONE:
    ret;
}
";

/// Chebyshev (all four kinds + shifted) — f64.
#[cfg(feature = "cuda")]
pub(crate) const CHEBYSHEV_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry chebyshev_poly_f64(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 total,
    .param .f64 seed_a,
    .param .f64 seed_b,
    .param .u32 shift
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r, %n_r, %shift_r, %k;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f64 %x, %xx, %p, %q, %r, %two_xx, %sa, %sb, %one;
    .reg .pred %oob, %is0, %is1, %loop, %do_shift;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %n_r,     [n];
    ld.param.u32 %total_r, [total];
    ld.param.f64 %sa,      [seed_a];
    ld.param.f64 %sb,      [seed_b];
    ld.param.u32 %shift_r, [shift];

    mov.f64 %one, 0d3FF0000000000000;       // 1.0

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %in_p, %off;
    ld.global.f64 %x, [%addr];

    mov.f64 %xx, %x;
    setp.ne.u32 %do_shift, %shift_r, 0;
    @!%do_shift bra AFTER_SHIFT;
    add.f64 %xx, %x, %x;
    sub.f64 %xx, %xx, %one;                  // 2x - 1.0
AFTER_SHIFT:

    mov.f64 %p, %one;                        // 1.0
    fma.rn.f64 %q, %sa, %xx, %sb;            // q1 = seed_a*xx + seed_b
    add.f64 %two_xx, %xx, %xx;

    setp.eq.u32 %is0, %n_r, 0;
    @%is0 bra STORE_P;
    setp.eq.u32 %is1, %n_r, 1;
    mov.f64 %r, %q;
    @%is1 bra STORE_R;

    mov.u32 %k, 2;
LOOP:
    setp.gt.u32 %loop, %k, %n_r;
    @%loop bra STORE_R;
    mul.f64 %r, %two_xx, %q;
    sub.f64 %r, %r, %p;
    mov.f64 %p, %q;
    mov.f64 %q, %r;
    add.u32 %k, %k, 1;
    bra LOOP;

STORE_P:
    mov.f64 %r, %p;
STORE_R:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out_p, %off;
    st.global.f64 [%addr], %r;
DONE:
    ret;
}
";

/// Hermite (physicist's) `H_n` — f32. `q0=1`, `q1=2x`,
/// `r = 2x*q - 2k*p` for `k = 1 .. n-1` (CPU: loop `k in 1..n`).
#[cfg(feature = "cuda")]
pub(crate) const HERMITE_H_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry hermite_h_poly_f32(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r, %n_r, %k;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %x, %two_x, %p, %q, %r, %twok;
    .reg .pred %oob, %is0, %is1, %loop, %over;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %n_r,     [n];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %x, [%addr];

    add.f32 %two_x, %x, %x;        // 2x
    mov.f32 %p, 0f3F800000;        // 1.0
    mov.f32 %q, %two_x;            // q1 = 2x

    setp.eq.u32 %is0, %n_r, 0;
    @%is0 bra STORE_ONE;
    setp.eq.u32 %is1, %n_r, 1;
    mov.f32 %r, %q;
    @%is1 bra STORE_R;
    // getHermitianLimit<float>() == 128 (Math.h:3044-3052); n>limit -> NaN
    // (Math.h:3068-3070), matching torch + the ferrotorch CPU f64 path.
    setp.gt.u32 %over, %n_r, 128;
    @%over bra STORE_NAN;

    mov.u32 %k, 1;
LOOP:
    setp.ge.u32 %loop, %k, %n_r;
    @%loop bra STORE_R;
    // r = 2x*q - 2k*p
    mul.f32 %r, %two_x, %q;
    cvt.rn.f32.u32 %twok, %k;
    add.f32 %twok, %twok, %twok;   // 2k
    mul.f32 %twok, %twok, %p;
    sub.f32 %r, %r, %twok;
    mov.f32 %p, %q;
    mov.f32 %q, %r;
    add.u32 %k, %k, 1;
    bra LOOP;

STORE_NAN:
    mov.f32 %r, 0f7FC00000;        // quiet NaN
    bra STORE_R;
STORE_ONE:
    mov.f32 %r, 0f3F800000;
STORE_R:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %r;
DONE:
    ret;
}
";

/// Hermite (physicist's) `H_n` — f64.
#[cfg(feature = "cuda")]
pub(crate) const HERMITE_H_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry hermite_h_poly_f64(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r, %n_r, %k;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f64 %x, %two_x, %p, %q, %r, %twok;
    .reg .pred %oob, %is0, %is1, %loop, %over;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %n_r,     [n];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %in_p, %off;
    ld.global.f64 %x, [%addr];

    add.f64 %two_x, %x, %x;
    mov.f64 %p, 0d3FF0000000000000;   // 1.0
    mov.f64 %q, %two_x;

    setp.eq.u32 %is0, %n_r, 0;
    @%is0 bra STORE_ONE;
    setp.eq.u32 %is1, %n_r, 1;
    mov.f64 %r, %q;
    @%is1 bra STORE_R;
    // getHermitianLimit<double>() == 512 (Math.h:3044-3052); n>limit -> NaN
    // (Math.h:3068-3070), matching torch + the ferrotorch CPU f64 path.
    setp.gt.u32 %over, %n_r, 512;
    @%over bra STORE_NAN;

    mov.u32 %k, 1;
LOOP:
    setp.ge.u32 %loop, %k, %n_r;
    @%loop bra STORE_R;
    mul.f64 %r, %two_x, %q;
    cvt.rn.f64.u32 %twok, %k;
    add.f64 %twok, %twok, %twok;
    mul.f64 %twok, %twok, %p;
    sub.f64 %r, %r, %twok;
    mov.f64 %p, %q;
    mov.f64 %q, %r;
    add.u32 %k, %k, 1;
    bra LOOP;

STORE_NAN:
    mov.f64 %r, 0d7FF8000000000000;   // quiet NaN
    bra STORE_R;
STORE_ONE:
    mov.f64 %r, 0d3FF0000000000000;
STORE_R:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out_p, %off;
    st.global.f64 [%addr], %r;
DONE:
    ret;
}
";

/// Hermite (probabilist's) `He_n` — f32. `q0=1`, `q1=x`,
/// `r = x*q - k*p` for `k = 1 .. n-1`.
#[cfg(feature = "cuda")]
pub(crate) const HERMITE_HE_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry hermite_he_poly_f32(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r, %n_r, %k;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %x, %p, %q, %r, %kf;
    .reg .pred %oob, %is0, %is1, %loop, %over;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %n_r,     [n];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %x, [%addr];

    mov.f32 %p, 0f3F800000;   // 1.0
    mov.f32 %q, %x;           // q1 = x

    setp.eq.u32 %is0, %n_r, 0;
    @%is0 bra STORE_ONE;
    setp.eq.u32 %is1, %n_r, 1;
    mov.f32 %r, %q;
    @%is1 bra STORE_R;
    // getHermitianLimit<float>() == 128 (Math.h:3044-3052); n>limit -> NaN
    // (Math.h:3109-3111), matching torch + the ferrotorch CPU f64 path.
    setp.gt.u32 %over, %n_r, 128;
    @%over bra STORE_NAN;

    mov.u32 %k, 1;
LOOP:
    setp.ge.u32 %loop, %k, %n_r;
    @%loop bra STORE_R;
    mul.f32 %r, %x, %q;
    cvt.rn.f32.u32 %kf, %k;
    mul.f32 %kf, %kf, %p;
    sub.f32 %r, %r, %kf;
    mov.f32 %p, %q;
    mov.f32 %q, %r;
    add.u32 %k, %k, 1;
    bra LOOP;

STORE_NAN:
    mov.f32 %r, 0f7FC00000;   // quiet NaN
    bra STORE_R;
STORE_ONE:
    mov.f32 %r, 0f3F800000;
STORE_R:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %r;
DONE:
    ret;
}
";

/// Hermite (probabilist's) `He_n` — f64.
#[cfg(feature = "cuda")]
pub(crate) const HERMITE_HE_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry hermite_he_poly_f64(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r, %n_r, %k;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f64 %x, %p, %q, %r, %kf;
    .reg .pred %oob, %is0, %is1, %loop, %over;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %n_r,     [n];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %in_p, %off;
    ld.global.f64 %x, [%addr];

    mov.f64 %p, 0d3FF0000000000000;   // 1.0
    mov.f64 %q, %x;

    setp.eq.u32 %is0, %n_r, 0;
    @%is0 bra STORE_ONE;
    setp.eq.u32 %is1, %n_r, 1;
    mov.f64 %r, %q;
    @%is1 bra STORE_R;
    // getHermitianLimit<double>() == 512 (Math.h:3044-3052); n>limit -> NaN
    // (Math.h:3109-3111), matching torch + the ferrotorch CPU f64 path.
    setp.gt.u32 %over, %n_r, 512;
    @%over bra STORE_NAN;

    mov.u32 %k, 1;
LOOP:
    setp.ge.u32 %loop, %k, %n_r;
    @%loop bra STORE_R;
    mul.f64 %r, %x, %q;
    cvt.rn.f64.u32 %kf, %k;
    mul.f64 %kf, %kf, %p;
    sub.f64 %r, %r, %kf;
    mov.f64 %p, %q;
    mov.f64 %q, %r;
    add.u32 %k, %k, 1;
    bra LOOP;

STORE_NAN:
    mov.f64 %r, 0d7FF8000000000000;   // quiet NaN
    bra STORE_R;
STORE_ONE:
    mov.f64 %r, 0d3FF0000000000000;
STORE_R:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out_p, %off;
    st.global.f64 [%addr], %r;
DONE:
    ret;
}
";

/// Laguerre `L_n` — f32. `q0=1`, `q1=1-x`,
/// `r = ((2k+1-x)*q - k*p) / (k+1)` for `k = 1 .. n-1`.
#[cfg(feature = "cuda")]
pub(crate) const LAGUERRE_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry laguerre_poly_f32(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r, %n_r, %k;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %x, %p, %q, %r, %kf, %coef, %tmp, %one;
    .reg .pred %oob, %is0, %is1, %loop;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %n_r,     [n];
    ld.param.u32 %total_r, [total];

    mov.f32 %one, 0f3F800000;      // 1.0

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %x, [%addr];

    mov.f32 %p, %one;              // 1.0
    sub.f32 %q, %one, %x;          // q1 = 1 - x

    setp.eq.u32 %is0, %n_r, 0;
    @%is0 bra STORE_ONE;
    setp.eq.u32 %is1, %n_r, 1;
    mov.f32 %r, %q;
    @%is1 bra STORE_R;

    mov.u32 %k, 1;
LOOP:
    setp.ge.u32 %loop, %k, %n_r;
    @%loop bra STORE_R;
    cvt.rn.f32.u32 %kf, %k;
    // coef = 2k + 1 - x
    add.f32 %coef, %kf, %kf;
    add.f32 %coef, %coef, %one;
    sub.f32 %coef, %coef, %x;
    // r = coef*q - k*p
    mul.f32 %r, %coef, %q;
    mul.f32 %tmp, %kf, %p;
    sub.f32 %r, %r, %tmp;
    // r /= (k + 1)
    add.f32 %tmp, %kf, %one;
    div.rn.f32 %r, %r, %tmp;
    mov.f32 %p, %q;
    mov.f32 %q, %r;
    add.u32 %k, %k, 1;
    bra LOOP;

STORE_ONE:
    mov.f32 %r, %one;
STORE_R:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %r;
DONE:
    ret;
}
";

/// Laguerre `L_n` — f64.
#[cfg(feature = "cuda")]
pub(crate) const LAGUERRE_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry laguerre_poly_f64(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r, %n_r, %k;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f64 %x, %p, %q, %r, %kf, %coef, %tmp, %one;
    .reg .pred %oob, %is0, %is1, %loop;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %n_r,     [n];
    ld.param.u32 %total_r, [total];

    mov.f64 %one, 0d3FF0000000000000;        // 1.0

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %in_p, %off;
    ld.global.f64 %x, [%addr];

    mov.f64 %p, %one;                        // 1.0
    sub.f64 %q, %one, %x;                    // q1 = 1 - x

    setp.eq.u32 %is0, %n_r, 0;
    @%is0 bra STORE_ONE;
    setp.eq.u32 %is1, %n_r, 1;
    mov.f64 %r, %q;
    @%is1 bra STORE_R;

    mov.u32 %k, 1;
LOOP:
    setp.ge.u32 %loop, %k, %n_r;
    @%loop bra STORE_R;
    cvt.rn.f64.u32 %kf, %k;
    add.f64 %coef, %kf, %kf;
    add.f64 %coef, %coef, %one;
    sub.f64 %coef, %coef, %x;
    mul.f64 %r, %coef, %q;
    mul.f64 %tmp, %kf, %p;
    sub.f64 %r, %r, %tmp;
    add.f64 %tmp, %kf, %one;
    div.rn.f64 %r, %r, %tmp;
    mov.f64 %p, %q;
    mov.f64 %q, %r;
    add.u32 %k, %k, 1;
    bra LOOP;

STORE_ONE:
    mov.f64 %r, %one;
STORE_R:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out_p, %off;
    st.global.f64 [%addr], %r;
DONE:
    ret;
}
";

/// Legendre `P_n` — f32. `q0=1`, `q1=x`,
/// `r = ((2k+1)*x*q - k*p) / (k+1)` for `k = 1 .. n-1`.
#[cfg(feature = "cuda")]
pub(crate) const LEGENDRE_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry legendre_poly_f32(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r, %n_r, %k;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %x, %p, %q, %r, %kf, %coef, %tmp, %one;
    .reg .pred %oob, %is0, %is1, %loop;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %n_r,     [n];
    ld.param.u32 %total_r, [total];

    mov.f32 %one, 0f3F800000;   // 1.0

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %x, [%addr];

    mov.f32 %p, %one;         // 1.0
    mov.f32 %q, %x;           // q1 = x

    setp.eq.u32 %is0, %n_r, 0;
    @%is0 bra STORE_ONE;
    setp.eq.u32 %is1, %n_r, 1;
    mov.f32 %r, %q;
    @%is1 bra STORE_R;

    mov.u32 %k, 1;
LOOP:
    setp.ge.u32 %loop, %k, %n_r;
    @%loop bra STORE_R;
    cvt.rn.f32.u32 %kf, %k;
    // coef = 2k + 1
    add.f32 %coef, %kf, %kf;
    add.f32 %coef, %coef, %one;
    // r = coef * x * q - k*p
    mul.f32 %r, %coef, %x;
    mul.f32 %r, %r, %q;
    mul.f32 %tmp, %kf, %p;
    sub.f32 %r, %r, %tmp;
    // r /= (k+1)
    add.f32 %tmp, %kf, %one;
    div.rn.f32 %r, %r, %tmp;
    mov.f32 %p, %q;
    mov.f32 %q, %r;
    add.u32 %k, %k, 1;
    bra LOOP;

STORE_ONE:
    mov.f32 %r, %one;
STORE_R:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %r;
DONE:
    ret;
}
";

/// Legendre `P_n` — f64.
#[cfg(feature = "cuda")]
pub(crate) const LEGENDRE_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry legendre_poly_f64(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r, %n_r, %k;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f64 %x, %p, %q, %r, %kf, %coef, %tmp, %one;
    .reg .pred %oob, %is0, %is1, %loop;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %n_r,     [n];
    ld.param.u32 %total_r, [total];

    mov.f64 %one, 0d3FF0000000000000;   // 1.0

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %in_p, %off;
    ld.global.f64 %x, [%addr];

    mov.f64 %p, %one;                 // 1.0
    mov.f64 %q, %x;

    setp.eq.u32 %is0, %n_r, 0;
    @%is0 bra STORE_ONE;
    setp.eq.u32 %is1, %n_r, 1;
    mov.f64 %r, %q;
    @%is1 bra STORE_R;

    mov.u32 %k, 1;
LOOP:
    setp.ge.u32 %loop, %k, %n_r;
    @%loop bra STORE_R;
    cvt.rn.f64.u32 %kf, %k;
    add.f64 %coef, %kf, %kf;
    add.f64 %coef, %coef, %one;
    mul.f64 %r, %coef, %x;
    mul.f64 %r, %r, %q;
    mul.f64 %tmp, %kf, %p;
    sub.f64 %r, %r, %tmp;
    add.f64 %tmp, %kf, %one;
    div.rn.f64 %r, %r, %tmp;
    mov.f64 %p, %q;
    mov.f64 %q, %r;
    add.u32 %k, %k, 1;
    bra LOOP;

STORE_ONE:
    mov.f64 %r, %one;
STORE_R:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out_p, %off;
    st.global.f64 [%addr], %r;
DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// PTX — Normal-distribution trio: entr / ndtr / ndtri (#1651 batch 1)
// ---------------------------------------------------------------------------
//
// These elementwise transcendental kernels are f32-only on the GPU. Base PTX
// (the `Ptx::from_src` path in `module_cache`, no libdevice link) has no
// `lg2.approx.f64` / `ex2.approx.f64`: the f64 logarithm/exponential required
// by these kernels cannot be evaluated at f64 precision on-device. Rather than
// silently bounce f64 CUDA tensors through the host (forbidden by R-CODE-4),
// the f64 GpuBackend methods return `NotImplementedOnCuda` (see
// `backend_impl.rs`); only f32 runs on-device. This mirrors the existing
// `cdist_f64` decision (`distance.rs:211-219`) where general f64
// transcendentals are not expressible in base PTX.
//
// The f32 math mirrors the ferrotorch CPU f32 scalar path so GPU == CPU to f32
// tolerance: `ln(x) = lg2.approx.f32(x) * ln(2)`, `exp(x) = ex2.approx.f32(x *
// log2(e))`, and the ndtr `erf` is the Abramowitz-Stegun 7.1.26 polynomial
// (`erf_scalar`'s f32 branch, special.rs). ndtri ports the Cephes rational in
// f32 (`ndtri_f64`, special.rs) — the f32-narrowed coefficients stay inside
// the f32 transcendental tolerance.

/// Entropy `entr(x)` — f32. Mirrors `entr_string`
/// (`aten/src/ATen/native/cuda/Math.cuh:463-480`):
/// NaN -> NaN; `x > 0 -> -x*ln(x)`; `x == 0 -> 0`; else `-inf`.
/// ABI: `(in, out, total)`. `ln(x) = lg2.approx.f32(x) * ln(2)`.
#[cfg(feature = "cuda")]
pub(crate) const ENTR_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry entr_f32(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %x, %r, %lg, %ln2, %zero;
    .reg .pred %oob, %isnan, %pos, %iszero;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %x, [%addr];

    mov.f32 %zero, 0f00000000;

    // NaN -> x (NaN). setp.nan true when x is NaN.
    setp.nan.f32 %isnan, %x, %x;
    @%isnan mov.f32 %r, %x;
    @%isnan bra STORE;

    // x > 0 -> -x*ln(x)
    setp.gt.f32 %pos, %x, %zero;
    @!%pos bra NOT_POS;
    mov.f32 %ln2, 0f3F317218;       // ln(2)
    lg2.approx.f32 %lg, %x;         // log2(x)
    mul.f32 %lg, %lg, %ln2;         // ln(x)
    mul.f32 %r, %x, %lg;            // x*ln(x)
    neg.f32 %r, %r;                 // -x*ln(x)
    bra STORE;

NOT_POS:
    // x == 0 -> +0.0 ; x < 0 -> -inf
    setp.eq.f32 %iszero, %x, %zero;
    @%iszero mov.f32 %r, 0f00000000;
    @%iszero bra STORE;
    mov.f32 %r, 0fFF800000;         // -inf

STORE:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %r;
DONE:
    ret;
}
";

/// Standard-normal CDF `ndtr(x)` — f32. Mirrors `calc_ndtr`
/// (`aten/src/ATen/native/UnaryOps.cpp:715-718`):
/// `ndtr(x) = (1 + erf(x/sqrt(2))) * 0.5`. `erf` is the Abramowitz-Stegun
/// 7.1.26 polynomial (matching the ferrotorch CPU f32 `erf_scalar` path):
/// `t = 1/(1 + p*|z|)`, `poly = a1 + t*(a2 + t*(a3 + t*(a4 + t*a5)))`,
/// `erf(z) = sign(z) * (1 - poly*t*exp(-z*z))`. ABI: `(in, out, total)`.
#[cfg(feature = "cuda")]
pub(crate) const NDTR_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry ndtr_f32(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %x, %z, %az, %sign, %t, %poly, %ex, %erf, %r;
    .reg .f32 %sqrt1_2, %p_c, %a1, %a2, %a3, %a4, %a5, %one, %half, %log2e, %zero;
    .reg .pred %oob, %neg;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %x, [%addr];

    mov.f32 %one,     0f3F800000;       // 1.0
    mov.f32 %half,    0f3F000000;       // 0.5
    mov.f32 %zero,    0f00000000;       // 0.0
    mov.f32 %sqrt1_2, 0f3F3504F3;       // 1/sqrt(2) = 0.70710677
    mov.f32 %log2e,   0f3FB8AA3B;       // log2(e)
    // Abramowitz-Stegun 7.1.26 constants (special.rs ERF_*).
    mov.f32 %p_c, 0f3EA7BA05;           //  0.3275911
    mov.f32 %a1,  0f3E827906;           //  0.254829592
    mov.f32 %a2,  0fBE91A98E;           // -0.284496736
    mov.f32 %a3,  0f3FB5F0E3;           //  1.421413741
    mov.f32 %a4,  0fBFBA00E3;           // -1.453152027
    mov.f32 %a5,  0f3F87DC22;           //  1.061405429

    // z = x / sqrt(2)
    mul.f32 %z, %x, %sqrt1_2;

    // erf(z): sign and |z|
    setp.lt.f32 %neg, %z, %zero;
    mov.f32 %sign, %one;
    @%neg neg.f32 %sign, %one;          // sign = -1 when z<0
    abs.f32 %az, %z;

    // t = 1 / (1 + p*|z|)
    fma.rn.f32 %t, %p_c, %az, %one;     // 1 + p*az
    rcp.rn.f32 %t, %t;                  // 1/(1+p*az)

    // poly = a1 + t*(a2 + t*(a3 + t*(a4 + t*a5)))
    mov.f32 %poly, %a5;
    fma.rn.f32 %poly, %poly, %t, %a4;
    fma.rn.f32 %poly, %poly, %t, %a3;
    fma.rn.f32 %poly, %poly, %t, %a2;
    fma.rn.f32 %poly, %poly, %t, %a1;

    // ex = exp(-z*z) = 2^((-z*z)*log2e)
    mul.f32 %ex, %az, %az;              // z*z (= |z|^2)
    neg.f32 %ex, %ex;                   // -z*z
    mul.f32 %ex, %ex, %log2e;
    ex2.approx.f32 %ex, %ex;

    // erf = sign * (1 - poly*t*ex)
    mul.f32 %erf, %poly, %t;
    mul.f32 %erf, %erf, %ex;
    sub.f32 %erf, %one, %erf;           // 1 - poly*t*ex
    mul.f32 %erf, %erf, %sign;

    // ndtr = (1 + erf) * 0.5
    add.f32 %r, %one, %erf;
    mul.f32 %r, %r, %half;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %r;
DONE:
    ret;
}
";

/// Inverse standard-normal CDF `ndtri(p)` — f32. Direct port of the Cephes
/// `ndtri_string` (`aten/src/ATen/native/cuda/Math.cuh:48-173`) in f32: the
/// three coefficient regions (central P0/Q0, tail P1/Q1, far-tail P2/Q2) and
/// the `code`-flag sign flip. `log`/`sqrt` use `lg2.approx.f32`*ln(2) /
/// `sqrt.rn.f32`. Domain `(0,1)`: `0 -> -inf`, `1 -> +inf`, outside -> NaN.
/// ABI: `(in, out, total)`.
///
/// The polevl regions are unrolled `fma.rn.f32` chains over the reverse-order
/// Cephes coefficients (`polevl`, special.rs). `code` starts true; if
/// `y > 1 - exp(-2)` we set `y = 1 - y`, `code = false`, and the final result
/// is negated unless `code` (i.e. `return code ? -x : x`).
#[cfg(feature = "cuda")]
pub(crate) const NDTRI_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

// NOTE: PTX comments must stay ASCII-only. The WSL driver (591.86) JIT parser
// rejects non-ASCII bytes (e.g. an em-dash in a comment) with
// CUDA_ERROR_INVALID_PTX, even though standalone ptxas tolerates UTF-8.
.visible .entry ndtri_f32(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %y0, %y, %y2, %x, %x0, %x1, %z, %num, %den, %r;
    .reg .f32 %one, %zero, %half, %ln2, %s2pi, %expm2, %thresh, %lg;
    .reg .pred %oob, %is0, %is1, %ood, %flip, %central, %smallx;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %y0, [%addr];

    mov.f32 %one,   0f3F800000;     // 1.0
    mov.f32 %zero,  0f00000000;     // 0.0
    mov.f32 %half,  0f3F000000;     // 0.5
    mov.f32 %ln2,   0f3F317218;     // ln(2)
    mov.f32 %s2pi,  0f40206C99;     // sqrt(2*pi) = 2.5066283
    mov.f32 %expm2, 0f3E0A9555;     // exp(-2) = 0.13533528
    mov.f32 %thresh,0f3F5D5AAB;     // 1 - exp(-2) = 0.86466472

    // y0 == 0 -> -inf
    setp.eq.f32 %is0, %y0, %zero;
    @%is0 bra NEG_INF;
    // y0 == 1 -> +inf
    setp.eq.f32 %is1, %y0, %one;
    @%is1 bra POS_INF;
    // y0 < 0 || y0 > 1 -> NaN
    setp.lt.f32 %ood, %y0, %zero;
    @%ood bra NANV;
    setp.gt.f32 %ood, %y0, %one;
    @%ood bra NANV;

    // code = true; y = y0; if (y > 1 - exp(-2)) { y = 1 - y; code = false; }
    mov.f32 %y, %y0;
    setp.gt.f32 %flip, %y0, %thresh;
    @%flip sub.f32 %y, %one, %y0;

    // central region: y > exp(-2)
    setp.gt.f32 %central, %y, %expm2;
    @!%central bra TAIL;

    // y = y - 0.5; y2 = y*y; x = y + y*(y2 * P0(y2)/Q0(y2)); return x*s2pi
    sub.f32 %y, %y, %half;
    mul.f32 %y2, %y, %y;
    // P0 (reverse order): -59.9633501, 98.0010754, -56.6762857,
    //                      13.9312609, -1.23916584
    mov.f32 %num, 0fC26FDA78;       // -59.9633501
    mov.f32 %x0,  0f42C4008D;       //  98.0010754  (scratch for coeffs)
    fma.rn.f32 %num, %num, %y2, %x0;
    mov.f32 %x0,  0fC262B484;       // -56.6762857
    fma.rn.f32 %num, %num, %y2, %x0;
    mov.f32 %x0,  0f415EE672;       //  13.9312609
    fma.rn.f32 %num, %num, %y2, %x0;
    mov.f32 %x0,  0fBF9E9CFC;       //  -1.23916584
    fma.rn.f32 %num, %num, %y2, %x0;
    // Q0 (reverse order): 1, 1.95448858, 4.67627913, 86.3602421,
    //   -225.462688, 200.260212, -82.0372256, 15.9056225, -1.18331621
    mov.f32 %den, 0f3F800000;       // 1.0
    mov.f32 %x0,  0f3FFA2CAF;       // 1.95448858
    fma.rn.f32 %den, %den, %y2, %x0;
    mov.f32 %x0,  0f4095A414;       // 4.67627913
    fma.rn.f32 %den, %den, %y2, %x0;
    mov.f32 %x0,  0f42ACB872;       // 86.3602421
    fma.rn.f32 %den, %den, %y2, %x0;
    mov.f32 %x0,  0fC3617673;       // -225.462688
    fma.rn.f32 %den, %den, %y2, %x0;
    mov.f32 %x0,  0f4348429D;       // 200.260212
    fma.rn.f32 %den, %den, %y2, %x0;
    mov.f32 %x0,  0fC2A4130F;       // -82.0372256
    fma.rn.f32 %den, %den, %y2, %x0;
    mov.f32 %x0,  0f417E7D6E;       // 15.9056225
    fma.rn.f32 %den, %den, %y2, %x0;
    mov.f32 %x0,  0fBF9776E8;       // -1.18331621
    fma.rn.f32 %den, %den, %y2, %x0;
    // x = y + y*(y2 * num/den)
    div.rn.f32 %x0, %num, %den;
    mul.f32 %x0, %x0, %y2;
    fma.rn.f32 %x, %y, %x0, %y;      // y + y*x0
    mul.f32 %r, %x, %s2pi;
    // Central region returns x*s2pi directly (Math.cuh:101): NO sign flip.
    bra STORE;

TAIL:
    // x = sqrt(-2 log y); ln y = lg2(y)*ln2
    lg2.approx.f32 %lg, %y;
    mul.f32 %lg, %lg, %ln2;          // ln(y)
    mov.f32 %x0, 0fC0000000;         // -2.0
    mul.f32 %x, %lg, %x0;            // -2 log y
    sqrt.rn.f32 %x, %x;              // x = sqrt(-2 log y)
    // x0 = x - log(x)/x
    lg2.approx.f32 %lg, %x;
    mul.f32 %lg, %lg, %ln2;          // ln(x)
    div.rn.f32 %lg, %lg, %x;         // log(x)/x
    sub.f32 %x0, %x, %lg;            // x0 = x - log(x)/x
    rcp.rn.f32 %z, %x;               // z = 1/x

    mov.f32 %lg, 0f41000000;         // 8.0 (reuse %lg as scratch)
    setp.lt.f32 %smallx, %x, %lg;    // x < 8.0
    @!%smallx bra FARTAIL;

    // P1/Q1 (x < 8). reverse-order coeffs.
    // P1: 4.05544892, 31.5251095, 57.1628192, 44.0805074, 14.6849562,
    //     2.18663307, -0.140256079, -0.0350424627, -0.000857456785
    mov.f32 %num, 0f4081C63D;        // 4.05544892
    mov.f32 %x1,  0f41FC336D;        // 31.5251095
    fma.rn.f32 %num, %num, %z, %x1;
    mov.f32 %x1,  0f4264A6BA;        // 57.1628192
    fma.rn.f32 %num, %num, %z, %x1;
    mov.f32 %x1,  0f42305271;        // 44.0805074
    fma.rn.f32 %num, %num, %z, %x1;
    mov.f32 %x1,  0f416AF595;        // 14.6849562
    fma.rn.f32 %num, %num, %z, %x1;
    mov.f32 %x1,  0f400BF1CC;        // 2.18663307
    fma.rn.f32 %num, %num, %z, %x1;
    mov.f32 %x1,  0fBE0F9F4A;        // -0.140256079
    fma.rn.f32 %num, %num, %z, %x1;
    mov.f32 %x1,  0fBD0F88AF;        // -0.0350424627
    fma.rn.f32 %num, %num, %z, %x1;
    mov.f32 %x1,  0fBA60C6F3;        // -0.000857456785
    fma.rn.f32 %num, %num, %z, %x1;
    // Q1
    mov.f32 %den, 0f3F800000;        // 1.0
    mov.f32 %x1,  0f417C7AD5;        // 15.7799883
    fma.rn.f32 %den, %den, %z, %x1;
    mov.f32 %x1,  0f42359024;        // 45.3907635
    fma.rn.f32 %den, %den, %z, %x1;
    mov.f32 %x1,  0f422544D1;        // 41.3172038
    fma.rn.f32 %den, %den, %z, %x1;
    mov.f32 %x1,  0f4170AE3D;        // 15.0425386
    fma.rn.f32 %den, %den, %z, %x1;
    mov.f32 %x1,  0f40204C2D;        // 2.50464946
    fma.rn.f32 %den, %den, %z, %x1;
    mov.f32 %x1,  0fBE119866;        // -0.142182923
    fma.rn.f32 %den, %den, %z, %x1;
    mov.f32 %x1,  0fBD1BFA72;        // -0.0380806408
    fma.rn.f32 %den, %den, %z, %x1;
    mov.f32 %x1,  0fBA74A5FC;        // -0.000933259481
    fma.rn.f32 %den, %den, %z, %x1;
    bra TAIL_FINISH;

FARTAIL:
    // P2/Q2 (x >= 8). reverse-order coeffs.
    mov.f32 %num, 0f404F3747;        // 3.23774892
    mov.f32 %x1,  0f40DD498E;        // 6.91522889
    fma.rn.f32 %num, %num, %z, %x1;
    mov.f32 %x1,  0f407C1578;        // 3.93881025
    fma.rn.f32 %num, %num, %z, %x1;
    mov.f32 %x1,  0f3FAAA0E1;        // 1.33303461
    fma.rn.f32 %num, %num, %z, %x1;
    mov.f32 %x1,  0f3E4E5230;        // 0.201485390
    fma.rn.f32 %num, %num, %z, %x1;
    mov.f32 %x1,  0f3C4AB285;        // 0.0123716635
    fma.rn.f32 %num, %num, %z, %x1;
    mov.f32 %x1,  0f399E1D97;        // 0.000301581554
    fma.rn.f32 %num, %num, %z, %x1;
    mov.f32 %x1,  0f3632614A;        // 2.65806975e-06
    fma.rn.f32 %num, %num, %z, %x1;
    mov.f32 %x1,  0f31D66562;        // 6.23974539e-09
    fma.rn.f32 %num, %num, %z, %x1;
    // Q2
    mov.f32 %den, 0f3F800000;        // 1.0
    mov.f32 %x1,  0f40C0C6D3;        // 6.02427039
    fma.rn.f32 %den, %den, %z, %x1;
    mov.f32 %x1,  0f406B826D;        // 3.67983564
    fma.rn.f32 %den, %den, %z, %x1;
    mov.f32 %x1,  0f3FB04239;        // 1.37702099
    fma.rn.f32 %den, %den, %z, %x1;
    mov.f32 %x1,  0f3E5D6D3B;        // 0.216236994
    fma.rn.f32 %den, %den, %z, %x1;
    mov.f32 %x1,  0f3C5BE13D;        // 0.0134204006
    fma.rn.f32 %den, %den, %z, %x1;
    mov.f32 %x1,  0f39ABF95B;        // 0.000328014465
    fma.rn.f32 %den, %den, %z, %x1;
    mov.f32 %x1,  0f36421C68;        // 2.89247865e-06
    fma.rn.f32 %den, %den, %z, %x1;
    mov.f32 %x1,  0f31E94F2E;        // 6.79019408e-09
    fma.rn.f32 %den, %den, %z, %x1;

TAIL_FINISH:
    // x1 = z * num/den ; x = x0 - x1
    div.rn.f32 %x1, %num, %den;
    mul.f32 %x1, %x1, %z;
    sub.f32 %r, %x0, %x1;

SIGN_FLIP:
    // return code ? -x : x. code is FALSE iff we flipped (y0 > thresh).
    // So: if (!flip) r = -r.
    @!%flip neg.f32 %r, %r;
    bra STORE;

NEG_INF:
    mov.f32 %r, 0fFF800000;          // -inf
    bra STORE;
POS_INF:
    mov.f32 %r, 0f7F800000;          // +inf
    bra STORE;
NANV:
    mov.f32 %r, 0f7FC00000;          // quiet NaN

STORE:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %r;
DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

/// Validate that `input` is on `device` and its element count fits in u32.
#[cfg(feature = "cuda")]
fn check_input<T>(input: &CudaBuffer<T>, device: &GpuDevice, op: &'static str) -> GpuResult<usize> {
    if input.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: input.device_ordinal(),
        });
    }
    let total = input.len();
    if total > u32::MAX as usize {
        return Err(GpuError::ShapeMismatch {
            op,
            expected: vec![u32::MAX as usize],
            got: vec![total],
        });
    }
    Ok(total)
}

#[cfg(feature = "cuda")]
fn launch_cfg(total: u32) -> LaunchConfig {
    let block_dim: u32 = 256;
    let grid_x = total.div_ceil(block_dim).max(1);
    LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Chebyshev polynomial (T/U/V/W + shifted) forward on an f32 buffer.
///
/// `seed_a`/`seed_b` select the kind via `q1 = seed_a*xx + seed_b`
/// (T: 1,0; U: 2,0; V: 2,-1; W: 2,1). `shift` true uses the shifted domain
/// (`xx = 2x - 1`).
///
/// # Errors
/// [`GpuError::DeviceMismatch`], [`GpuError::ShapeMismatch`] on validation
/// failure; [`GpuError::PtxCompileFailed`] / [`GpuError::Driver`] on launch.
#[cfg(feature = "cuda")]
pub fn gpu_chebyshev_poly_f32(
    input: &CudaBuffer<f32>,
    n: usize,
    seed_a: f32,
    seed_b: f32,
    shift: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let total = check_input(input, device, "chebyshev_poly_f32")?;
    let mut out = alloc_zeros_f32(total, device)?;
    if total == 0 {
        return Ok(out);
    }
    let f = crate::module_cache::get_or_compile(
        device.context(),
        CHEBYSHEV_F32_PTX,
        "chebyshev_poly_f32",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "chebyshev_poly_f32",
        source: e,
    })?;
    let n_u32 = n as u32;
    let total_u32 = total as u32;
    let shift_u32: u32 = u32::from(shift);
    let cfg = launch_cfg(total_u32);
    // SAFETY: `f` is `chebyshev_poly_f32`; the launch ABI
    // `(in, out, n, total, seed_a, seed_b, shift)` matches the PTX `.entry`
    // one-for-one. `input` is on `device` with exactly `total` f32 elements
    // (validated by `check_input`); `out` was just allocated to `total` f32
    // elements and cannot alias `input` (fresh cudarc slice + exclusive
    // `&mut`). Every thread either has `idx < total` or exits at the
    // `setp.ge.u32` guard, so all loads/stores are in-bounds; `total` is
    // range-checked against `u32::MAX`. By-ref params outlive the synchronous
    // launch on this stack frame.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .arg(&total_u32)
            .arg(&seed_a)
            .arg(&seed_b)
            .arg(&shift_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Chebyshev polynomial (T/U/V/W + shifted) forward on an f64 buffer.
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_chebyshev_poly_f64(
    input: &CudaBuffer<f64>,
    n: usize,
    seed_a: f64,
    seed_b: f64,
    shift: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let total = check_input(input, device, "chebyshev_poly_f64")?;
    let mut out = alloc_zeros_f64(total, device)?;
    if total == 0 {
        return Ok(out);
    }
    let f = crate::module_cache::get_or_compile(
        device.context(),
        CHEBYSHEV_F64_PTX,
        "chebyshev_poly_f64",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "chebyshev_poly_f64",
        source: e,
    })?;
    let n_u32 = n as u32;
    let total_u32 = total as u32;
    let shift_u32: u32 = u32::from(shift);
    let cfg = launch_cfg(total_u32);
    // SAFETY: identical contract to `gpu_chebyshev_poly_f32`, f64 element
    // width (8-byte offsets in the PTX). `f` is `chebyshev_poly_f64`; ABI
    // matches; `input`/`out` are validated/non-aliasing f64 buffers on
    // `device`; threads are bounds-guarded; `total` fits u32.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .arg(&total_u32)
            .arg(&seed_a)
            .arg(&seed_b)
            .arg(&shift_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Macro-free helper to launch a single-`n` recurrence kernel (hermite /
/// laguerre / legendre) whose ABI is `(in, out, n, total)`.
#[cfg(feature = "cuda")]
fn launch_simple_f32(
    input: &CudaBuffer<f32>,
    n: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel: &'static str,
) -> GpuResult<CudaBuffer<f32>> {
    let total = check_input(input, device, kernel)?;
    let mut out = alloc_zeros_f32(total, device)?;
    if total == 0 {
        return Ok(out);
    }
    let f =
        crate::module_cache::get_or_compile(device.context(), ptx, kernel, device.ordinal() as u32)
            .map_err(|e| GpuError::PtxCompileFailed { kernel, source: e })?;
    let n_u32 = n as u32;
    let total_u32 = total as u32;
    let cfg = launch_cfg(total_u32);
    // SAFETY: `f` is `kernel` (one of the `(in,out,n,total)`-ABI recurrence
    // entries); the four launch args match the PTX `.entry` order. `input`
    // is on `device` with `total` f32 elements (validated); `out` is a fresh
    // non-aliasing `total`-element buffer; threads are bounds-guarded by the
    // `setp.ge.u32` head; `total` fits u32. By-ref params outlive the launch.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// f64 counterpart of [`launch_simple_f32`].
#[cfg(feature = "cuda")]
fn launch_simple_f64(
    input: &CudaBuffer<f64>,
    n: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel: &'static str,
) -> GpuResult<CudaBuffer<f64>> {
    let total = check_input(input, device, kernel)?;
    let mut out = alloc_zeros_f64(total, device)?;
    if total == 0 {
        return Ok(out);
    }
    let f =
        crate::module_cache::get_or_compile(device.context(), ptx, kernel, device.ordinal() as u32)
            .map_err(|e| GpuError::PtxCompileFailed { kernel, source: e })?;
    let n_u32 = n as u32;
    let total_u32 = total as u32;
    let cfg = launch_cfg(total_u32);
    // SAFETY: as `launch_simple_f32` with f64 element width (8-byte offsets
    // in the PTX). ABI `(in,out,n,total)` matches; buffers validated and
    // non-aliasing; threads bounds-guarded; `total` fits u32.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Hermite (physicist's) `H_n` forward on an f32 buffer.
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_hermite_h_poly_f32(
    input: &CudaBuffer<f32>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    launch_simple_f32(input, n, device, HERMITE_H_F32_PTX, "hermite_h_poly_f32")
}

/// Hermite (physicist's) `H_n` forward on an f64 buffer.
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_hermite_h_poly_f64(
    input: &CudaBuffer<f64>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    launch_simple_f64(input, n, device, HERMITE_H_F64_PTX, "hermite_h_poly_f64")
}

/// Hermite (probabilist's) `He_n` forward on an f32 buffer.
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_hermite_he_poly_f32(
    input: &CudaBuffer<f32>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    launch_simple_f32(input, n, device, HERMITE_HE_F32_PTX, "hermite_he_poly_f32")
}

/// Hermite (probabilist's) `He_n` forward on an f64 buffer.
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_hermite_he_poly_f64(
    input: &CudaBuffer<f64>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    launch_simple_f64(input, n, device, HERMITE_HE_F64_PTX, "hermite_he_poly_f64")
}

/// Laguerre `L_n` forward on an f32 buffer.
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_laguerre_poly_f32(
    input: &CudaBuffer<f32>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    launch_simple_f32(input, n, device, LAGUERRE_F32_PTX, "laguerre_poly_f32")
}

/// Laguerre `L_n` forward on an f64 buffer.
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_laguerre_poly_f64(
    input: &CudaBuffer<f64>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    launch_simple_f64(input, n, device, LAGUERRE_F64_PTX, "laguerre_poly_f64")
}

/// Legendre `P_n` forward on an f32 buffer.
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_legendre_poly_f32(
    input: &CudaBuffer<f32>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    launch_simple_f32(input, n, device, LEGENDRE_F32_PTX, "legendre_poly_f32")
}

/// Legendre `P_n` forward on an f64 buffer.
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_legendre_poly_f64(
    input: &CudaBuffer<f64>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    launch_simple_f64(input, n, device, LEGENDRE_F64_PTX, "legendre_poly_f64")
}

// ---------------------------------------------------------------------------
// PTX - Modified-Bessel-I family: i0 / i0e / i1 / i1e (#1651 batch 2)
// ---------------------------------------------------------------------------
//
// f32-only on the GPU (base PTX lacks lg2/ex2.approx.f64; f64 -> Not-
// ImplementedOnCuda, same constraint as batch 1). The f32 math mirrors the
// ferrotorch CPU f64 scalar evaluators narrowed to f32 (i0_f64 .. in
// special.rs): the shared chbevl Clenshaw recurrence is unrolled over the
// Cephes A/B Chebyshev coefficient tables, exp via ex2.approx.f32(ax*log2e),
// and the B-set divides by sqrt.rn.f32(ax). i0/i0e use fabs (even); i1/i1e
// negate on x<0 (odd). The |x|<=8 split selects the A vs B coefficient set.

/// Modified Bessel i0(x), f32. Even (fabs). |x|<=8 A-set: exp(ax)*chbevl(ax/2-2, A); |x|>8 B-set: exp(ax)*chbevl(32/ax-2, B)/sqrt(ax). Mirrors i0_string (aten/src/ATen/native/cuda/Math.cuh:502-555).
/// ABI: `(in, out, total)`.
#[cfg(feature = "cuda")]
pub(crate) const I0_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

// NOTE: PTX comments must stay ASCII-only. The WSL driver (591.86) JIT parser
// rejects non-ASCII bytes with CUDA_ERROR_INVALID_PTX. Coefficient immediates
// are the f32 bit-hex of the Cephes A/B Chebyshev tables (special.rs I0E_A..).

.visible .entry i0_f32(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %x, %ax, %xx, %b0, %b1, %b2, %cb, %ex, %r, %sq;
    .reg .pred %oob, %small, %neg;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %x, [%addr];

    abs.f32 %ax, %x;
    setp.le.f32 %small, %ax, 0f41000000;   // 8.0
    @!%small bra BIG;
    // A-set: xx = ax/2 - 2
    mul.f32 %xx, %ax, 0f3F000000;          // ax*0.5
    add.f32 %xx, %xx, 0fC0000000;          // -2.0
    mov.f32 %b0, 0fA2A2E5B9;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f24199B15;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA58C275C;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f26F736C5;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA8528116;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f29ACDA32;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAB08B263;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2C4FF17F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAD97E4AC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2ED4C5F6;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB00EA7F1;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3136C81D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB25F57B4;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3381DBB5;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB48F631C;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3595F925;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB694337E;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3789FAC6;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB8715933;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3945A8DC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBA1717E9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3AD6E3AC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBB8DB2F1;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3C2CCB10;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBCC274F8;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3D49F456;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBDC25B82;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3E2FBD64;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBE9BFF5E;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3F2D4275;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %r, %cb;
    mul.f32 %ex, %ax, 0f3FB8AA3B;          // ax*log2e
    ex2.approx.f32 %ex, %ex;
    mul.f32 %r, %r, %ex;                   // *exp(ax)
    bra SIGN;

BIG:
    rcp.rn.f32 %xx, %ax;
    mul.f32 %xx, %xx, 0f42000000;          // 32/ax
    add.f32 %xx, %xx, 0fC0000000;          // -2.0
    mov.f32 %b0, 0fA3056DBB;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA2B236D3;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f244DF0C1;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f241F9EE8;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA5A3005D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA5C5773F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f26FF73ED;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2789548D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA82C1FF4;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA93AECCE;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f288AB7F8;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2AD8E463;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2B4A1A40;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fABFC8218;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAD687EBA;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAE0A88E8;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2D5127F5;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3007CE66;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f31696325;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f32C2B494;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f345C003F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3642095E;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f38907D1C;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3B5CCC65;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3F4DF315;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %r, %cb;
    mul.f32 %ex, %ax, 0f3FB8AA3B;          // ax*log2e
    ex2.approx.f32 %ex, %ex;
    mul.f32 %r, %r, %ex;                   // *exp(ax)
    sqrt.rn.f32 %sq, %ax;
    div.rn.f32 %r, %r, %sq;                // /sqrt(ax)

SIGN:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %r;
DONE:
    ret;
}
";

/// Exp-scaled i0e(x)=exp(-|x|)I0(x), f32. Even. Same A/B sets as i0 WITHOUT exp(ax). Mirrors calc_i0e (aten/src/ATen/native/Math.h:101-145).
/// ABI: `(in, out, total)`.
#[cfg(feature = "cuda")]
pub(crate) const I0E_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

// NOTE: PTX comments must stay ASCII-only. The WSL driver (591.86) JIT parser
// rejects non-ASCII bytes with CUDA_ERROR_INVALID_PTX. Coefficient immediates
// are the f32 bit-hex of the Cephes A/B Chebyshev tables (special.rs I0E_A..).

.visible .entry i0e_f32(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %x, %ax, %xx, %b0, %b1, %b2, %cb, %ex, %r, %sq;
    .reg .pred %oob, %small, %neg;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %x, [%addr];

    abs.f32 %ax, %x;
    setp.le.f32 %small, %ax, 0f41000000;   // 8.0
    @!%small bra BIG;
    // A-set: xx = ax/2 - 2
    mul.f32 %xx, %ax, 0f3F000000;          // ax*0.5
    add.f32 %xx, %xx, 0fC0000000;          // -2.0
    mov.f32 %b0, 0fA2A2E5B9;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f24199B15;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA58C275C;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f26F736C5;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA8528116;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f29ACDA32;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAB08B263;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2C4FF17F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAD97E4AC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2ED4C5F6;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB00EA7F1;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3136C81D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB25F57B4;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3381DBB5;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB48F631C;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3595F925;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB694337E;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3789FAC6;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB8715933;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3945A8DC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBA1717E9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3AD6E3AC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBB8DB2F1;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3C2CCB10;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBCC274F8;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3D49F456;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBDC25B82;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3E2FBD64;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBE9BFF5E;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3F2D4275;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %r, %cb;
    bra SIGN;

BIG:
    rcp.rn.f32 %xx, %ax;
    mul.f32 %xx, %xx, 0f42000000;          // 32/ax
    add.f32 %xx, %xx, 0fC0000000;          // -2.0
    mov.f32 %b0, 0fA3056DBB;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA2B236D3;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f244DF0C1;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f241F9EE8;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA5A3005D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA5C5773F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f26FF73ED;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2789548D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA82C1FF4;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA93AECCE;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f288AB7F8;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2AD8E463;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2B4A1A40;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fABFC8218;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAD687EBA;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAE0A88E8;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2D5127F5;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3007CE66;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f31696325;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f32C2B494;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f345C003F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3642095E;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f38907D1C;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3B5CCC65;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3F4DF315;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %r, %cb;
    sqrt.rn.f32 %sq, %ax;
    div.rn.f32 %r, %r, %sq;                // /sqrt(ax)

SIGN:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %r;
DONE:
    ret;
}
";

/// Modified Bessel i1(x), f32. Odd (sign of x). |x|<=8: exp(ax)*ax*chbevl(ax/2-2, A); |x|>8: exp(ax)*chbevl(32/ax-2, B)/sqrt(ax). Mirrors i1_string (aten/src/ATen/native/cuda/Math.cuh:575-622).
/// ABI: `(in, out, total)`.
#[cfg(feature = "cuda")]
pub(crate) const I1_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

// NOTE: PTX comments must stay ASCII-only. The WSL driver (591.86) JIT parser
// rejects non-ASCII bytes with CUDA_ERROR_INVALID_PTX. Coefficient immediates
// are the f32 bit-hex of the Cephes A/B Chebyshev tables (special.rs I0E_A..).

.visible .entry i1_f32(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %x, %ax, %xx, %b0, %b1, %b2, %cb, %ex, %r, %sq;
    .reg .pred %oob, %small, %neg;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %x, [%addr];

    abs.f32 %ax, %x;
    setp.le.f32 %small, %ax, 0f41000000;   // 8.0
    @!%small bra BIG;
    // A-set: xx = ax/2 - 2
    mul.f32 %xx, %ax, 0f3F000000;          // ax*0.5
    add.f32 %xx, %xx, 0fC0000000;          // -2.0
    mov.f32 %b0, 0f224CF950;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA3C2BE86;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f25331F1F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA69F5554;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2808EBF8;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA9631471;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2AB57BC2;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAC0B9C1B;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2D4E7716;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAE92881D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2FC751A6;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB101B0D9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f32212C70;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB33EE9F1;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f34571A26;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB56603CC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3668E277;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB75EAFCE;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f38488DAA;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB9299E57;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3A064AEE;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBAC66310;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3B88329A;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBC2D14FC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3CCA8F1F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBD58DDE3;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3DD236D7;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBE34A688;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3E81531C;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %r, %cb;
    mul.f32 %r, %r, %ax;                   // *x (i1 small branch)
    mul.f32 %ex, %ax, 0f3FB8AA3B;          // ax*log2e
    ex2.approx.f32 %ex, %ex;
    mul.f32 %r, %r, %ex;                   // *exp(ax)
    bra SIGN;

BIG:
    rcp.rn.f32 %xx, %ax;
    mul.f32 %xx, %xx, 0f42000000;          // 32/ax
    add.f32 %xx, %xx, 0fC0000000;          // -2.0
    mov.f32 %b0, 0f230AAB6E;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f22A2DC57;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA456751E;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA4140365;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f25AAC8B0;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f25BEB473;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA7077E6C;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA7896DA9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f283BB70C;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f294069E1;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA8BD4A41;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAAE5E22C;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAB4A9F08;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2C0F3EA0;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2D7880FB;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2E0F0D10;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fADA6E7CF;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB019A653;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB183C85D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB2E20A9D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB486DFE9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB68246FA;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB8E7EBFC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBC1FED03;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3F4750C6;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %r, %cb;
    mul.f32 %ex, %ax, 0f3FB8AA3B;          // ax*log2e
    ex2.approx.f32 %ex, %ex;
    mul.f32 %r, %r, %ex;                   // *exp(ax)
    sqrt.rn.f32 %sq, %ax;
    div.rn.f32 %r, %r, %sq;                // /sqrt(ax)

SIGN:
    setp.lt.f32 %neg, %x, 0f00000000;
    @%neg neg.f32 %r, %r;                  // odd: sign of x
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %r;
DONE:
    ret;
}
";

/// Exp-scaled i1e(x)=exp(-|x|)I1(x), f32. Odd. |x|<=8: ax*chbevl(ax/2-2, A); |x|>8: chbevl(32/ax-2, B)/sqrt(ax). Mirrors calc_i1e (aten/src/ATen/native/cuda/Math.cuh:647-696).
/// ABI: `(in, out, total)`.
#[cfg(feature = "cuda")]
pub(crate) const I1E_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

// NOTE: PTX comments must stay ASCII-only. The WSL driver (591.86) JIT parser
// rejects non-ASCII bytes with CUDA_ERROR_INVALID_PTX. Coefficient immediates
// are the f32 bit-hex of the Cephes A/B Chebyshev tables (special.rs I0E_A..).

.visible .entry i1e_f32(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %x, %ax, %xx, %b0, %b1, %b2, %cb, %ex, %r, %sq;
    .reg .pred %oob, %small, %neg;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %x, [%addr];

    abs.f32 %ax, %x;
    setp.le.f32 %small, %ax, 0f41000000;   // 8.0
    @!%small bra BIG;
    // A-set: xx = ax/2 - 2
    mul.f32 %xx, %ax, 0f3F000000;          // ax*0.5
    add.f32 %xx, %xx, 0fC0000000;          // -2.0
    mov.f32 %b0, 0f224CF950;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA3C2BE86;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f25331F1F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA69F5554;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2808EBF8;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA9631471;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2AB57BC2;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAC0B9C1B;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2D4E7716;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAE92881D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2FC751A6;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB101B0D9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f32212C70;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB33EE9F1;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f34571A26;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB56603CC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3668E277;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB75EAFCE;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f38488DAA;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB9299E57;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3A064AEE;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBAC66310;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3B88329A;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBC2D14FC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3CCA8F1F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBD58DDE3;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3DD236D7;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBE34A688;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3E81531C;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %r, %cb;
    mul.f32 %r, %r, %ax;                   // *x (i1 small branch)
    bra SIGN;

BIG:
    rcp.rn.f32 %xx, %ax;
    mul.f32 %xx, %xx, 0f42000000;          // 32/ax
    add.f32 %xx, %xx, 0fC0000000;          // -2.0
    mov.f32 %b0, 0f230AAB6E;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f22A2DC57;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA456751E;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA4140365;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f25AAC8B0;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f25BEB473;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA7077E6C;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA7896DA9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f283BB70C;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f294069E1;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA8BD4A41;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAAE5E22C;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAB4A9F08;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2C0F3EA0;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2D7880FB;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2E0F0D10;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fADA6E7CF;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB019A653;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB183C85D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB2E20A9D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB486DFE9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB68246FA;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB8E7EBFC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBC1FED03;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3F4750C6;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %r, %cb;
    sqrt.rn.f32 %sq, %ax;
    div.rn.f32 %r, %r, %sq;                // /sqrt(ax)

SIGN:
    setp.lt.f32 %neg, %x, 0f00000000;
    @%neg neg.f32 %r, %r;                  // odd: sign of x
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %r;
DONE:
    ret;
}
";

/// Spherical Bessel j0(x), f32. `isinf -> 0`; `|x| < 0.5` Taylor (Horner in
/// x*x over the 6 explicit Cephes terms); else `sin.approx.f32(x) / x`. Mirrors
/// spherical_bessel_j0_forward (aten/src/ATen/native/cuda/Math.cuh:3039-3052).
/// `j0(NaN) = NaN` (NaN is not `< 0.5`, falls through to `sin(NaN)/NaN = NaN`;
/// the `isinf` test is false for NaN). ABI: `(in, out, total)`.
#[cfg(feature = "cuda")]
pub(crate) const SPHERICAL_BESSEL_J0_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

// NOTE: PTX comments must stay ASCII-only. The WSL driver (591.86) JIT parser
// rejects non-ASCII bytes with CUDA_ERROR_INVALID_PTX. Taylor coefficients are
// the f32 bit-hex of the Cephes terms (-1/6, 1/120, -1/5040, 1/362880,
// -1/39916800, 1/6227020800) in cuda/Math.cuh:3047.

.visible .entry spherical_bessel_j0_f32(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %x, %ax, %x2, %p, %s, %r;
    .reg .pred %oob, %isinf, %small;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %x, [%addr];

    // isinf(x) -> 0.0
    testp.infinite.f32 %isinf, %x;
    @%isinf bra ZERO;

    abs.f32 %ax, %x;
    setp.lt.f32 %small, %ax, 0f3F000000;   // |x| < 0.5
    @!%small bra SINX;

    // Taylor: Horner in x2 = x*x.
    mul.f32 %x2, %x, %x;
    mov.f32 %p, 0f2F309231;                 // 1/6227020800
    fma.rn.f32 %p, %p, %x2, 0fB2D7322B;     // *x2 + (-1/39916800)
    fma.rn.f32 %p, %p, %x2, 0f3638EF1D;     // *x2 + (1/362880)
    fma.rn.f32 %p, %p, %x2, 0fB9500D01;     // *x2 + (-1/5040)
    fma.rn.f32 %p, %p, %x2, 0f3C088889;     // *x2 + (1/120)
    fma.rn.f32 %p, %p, %x2, 0fBE2AAAAB;     // *x2 + (-1/6)
    fma.rn.f32 %r, %p, %x2, 0f3F800000;     // *x2 + 1.0
    bra STORE;

SINX:
    sin.approx.f32 %s, %x;
    div.rn.f32 %r, %s, %x;
    bra STORE;

ZERO:
    mov.f32 %r, 0f00000000;

STORE:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %r;
DONE:
    ret;
}
";

/// modified_bessel_k0(x), f32. x==0 -> +inf; x<0 -> NaN. SMALL (x<=2): chbevl(x*x-2, K0_A) - log(0.5x)*i0(x), where i0(x)=exp(x)*chbevl(x*0.5-2, I0E_A). BIG (x>2): exp(-x)*chbevl(8/x-2, K0_B)/sqrt(x). Mirrors modified_bessel_k0_forward (aten/src/ATen/native/cuda/Math.cuh:2503-2577).
/// ABI: `(in, out, total)`.
#[cfg(feature = "cuda")]
pub(crate) const K0_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

// NOTE: PTX comments must stay ASCII-only. The WSL driver (591.86) JIT parser
// rejects non-ASCII bytes with CUDA_ERROR_INVALID_PTX. Coefficient immediates
// are the f32 bit-hex of the Cephes A/B Chebyshev tables (special.rs K0_A.. /
// I0E_A.. for the inner i0/i1 of the small-region log term).

.visible .entry modified_bessel_k0_f32(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %x, %xx, %xi, %b0, %b1, %b2, %cb, %cba, %ibe, %ex, %lg, %r, %sq;
    .reg .pred %oob, %small, %zero, %neg;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %x, [%addr];

    // x == 0 -> +inf
    setp.eq.f32 %zero, %x, 0f00000000;
    @%zero mov.f32 %r, 0f7F800000;
    @%zero bra STORE;
    // x < 0 -> NaN
    setp.lt.f32 %neg, %x, 0f00000000;
    @%neg mov.f32 %r, 0f7FC00000;
    @%neg bra STORE;

    setp.le.f32 %small, %x, 0f40000000;   // x <= 2.0
    @!%small bra BIG;
    // SMALL (x <= 2): chbevl(x*x - 2, K0_A)
    mul.f32 %xx, %x, %x;
    add.f32 %xx, %xx, 0fC0000000;          // -2.0
    mov.f32 %b0, 0f251E770F;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f293FD856;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2D3612E2;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3102E09B;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f348815E7;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f37BFC808;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3AA5C174;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3D135FB3;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3EB046C4;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBF090B37;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %cba, %cb;                    // chbevl(x*x-2, A)
    // inner i0(x) = exp(x) * chbevl(x*0.5 - 2, I0E_A)
    mul.f32 %xi, %x, 0f3F000000;          // x*0.5
    add.f32 %xi, %xi, 0fC0000000;          // -2.0
    mov.f32 %b0, 0fA2A2E5B9;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f24199B15;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA58C275C;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f26F736C5;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA8528116;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f29ACDA32;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAB08B263;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2C4FF17F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAD97E4AC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2ED4C5F6;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB00EA7F1;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3136C81D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB25F57B4;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3381DBB5;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB48F631C;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3595F925;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB694337E;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3789FAC6;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB8715933;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3945A8DC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBA1717E9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3AD6E3AC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBB8DB2F1;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3C2CCB10;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBCC274F8;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3D49F456;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBDC25B82;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3E2FBD64;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBE9BFF5E;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3F2D4275;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %ibe, %cb;
    mul.f32 %ex, %x, 0f3FB8AA3B;          // x*log2e
    ex2.approx.f32 %ex, %ex;              // exp(x)
    mul.f32 %ibe, %ibe, %ex;
    // log(0.5*x) = lg2.approx(0.5*x) * ln2
    mul.f32 %lg, %x, 0f3F000000;          // 0.5*x
    lg2.approx.f32 %lg, %lg;
    mul.f32 %lg, %lg, 0f3F317218;          // * ln2
    mul.f32 %r, %lg, %ibe;
    sub.f32 %r, %cba, %r;                 // chbevl(A) - log(0.5x)*i0
    bra STORE;

BIG:
    // BIG (x > 2): chbevl(8/x - 2, K0_B) / sqrt(x) * exp(-x)
    rcp.rn.f32 %xx, %x;
    mul.f32 %xx, %xx, 0f41000000;          // 8/x
    add.f32 %xx, %xx, 0fC0000000;          // -2.0
    mov.f32 %b0, 0f22C38D2E;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA397F665;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f24704972;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA5417CA4;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f261EDFCB;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA7053487;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f27E46CE2;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA8C8A2F6;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f29B4E276;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAAA7C3B9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2BA07D19;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAC9ED7E2;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2DA34042;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAEAEFCAF;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2FC4AB64;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB0E94EB0;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f32134D00;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB3482AA1;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f34948ACC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB5F5DAA4;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f376A09FE;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB906BCBF;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3ACDC490;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBD00CFB9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f401C2DED;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %r, %cb;
    sqrt.rn.f32 %sq, %x;
    div.rn.f32 %r, %r, %sq;                // / sqrt(x)
    // * exp(-x)
    mul.f32 %ex, %x, 0fBFB8AA3B;          // -x*log2e
    ex2.approx.f32 %ex, %ex;              // exp(-x)
    mul.f32 %r, %r, %ex;

STORE:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %r;
DONE:
    ret;
}
";

/// scaled_modified_bessel_k0(x)=exp(x)*k0(x), f32. SMALL: (chbevl(x*x-2, K0_A) - log(0.5x)*i0(x))*exp(x). BIG: chbevl(8/x-2, K0_B)/sqrt(x). Mirrors scaled_modified_bessel_k0_forward (aten/src/ATen/native/cuda/Math.cuh:2582-2656).
/// ABI: `(in, out, total)`.
#[cfg(feature = "cuda")]
pub(crate) const SCALED_K0_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

// NOTE: PTX comments must stay ASCII-only. The WSL driver (591.86) JIT parser
// rejects non-ASCII bytes with CUDA_ERROR_INVALID_PTX. Coefficient immediates
// are the f32 bit-hex of the Cephes A/B Chebyshev tables (special.rs K0_A.. /
// I0E_A.. for the inner i0/i1 of the small-region log term).

.visible .entry scaled_modified_bessel_k0_f32(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %x, %xx, %xi, %b0, %b1, %b2, %cb, %cba, %ibe, %ex, %lg, %r, %sq;
    .reg .pred %oob, %small, %zero, %neg;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %x, [%addr];

    // x == 0 -> +inf
    setp.eq.f32 %zero, %x, 0f00000000;
    @%zero mov.f32 %r, 0f7F800000;
    @%zero bra STORE;
    // x < 0 -> NaN
    setp.lt.f32 %neg, %x, 0f00000000;
    @%neg mov.f32 %r, 0f7FC00000;
    @%neg bra STORE;

    setp.le.f32 %small, %x, 0f40000000;   // x <= 2.0
    @!%small bra BIG;
    // SMALL (x <= 2): chbevl(x*x - 2, K0_A)
    mul.f32 %xx, %x, %x;
    add.f32 %xx, %xx, 0fC0000000;          // -2.0
    mov.f32 %b0, 0f251E770F;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f293FD856;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2D3612E2;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3102E09B;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f348815E7;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f37BFC808;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3AA5C174;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3D135FB3;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3EB046C4;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBF090B37;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %cba, %cb;                    // chbevl(x*x-2, A)
    // inner i0(x) = exp(x) * chbevl(x*0.5 - 2, I0E_A)
    mul.f32 %xi, %x, 0f3F000000;          // x*0.5
    add.f32 %xi, %xi, 0fC0000000;          // -2.0
    mov.f32 %b0, 0fA2A2E5B9;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f24199B15;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA58C275C;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f26F736C5;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA8528116;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f29ACDA32;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAB08B263;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2C4FF17F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAD97E4AC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2ED4C5F6;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB00EA7F1;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3136C81D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB25F57B4;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3381DBB5;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB48F631C;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3595F925;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB694337E;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3789FAC6;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB8715933;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3945A8DC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBA1717E9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3AD6E3AC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBB8DB2F1;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3C2CCB10;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBCC274F8;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3D49F456;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBDC25B82;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3E2FBD64;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBE9BFF5E;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3F2D4275;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %ibe, %cb;
    mul.f32 %ex, %x, 0f3FB8AA3B;          // x*log2e
    ex2.approx.f32 %ex, %ex;              // exp(x)
    mul.f32 %ibe, %ibe, %ex;
    // log(0.5*x) = lg2.approx(0.5*x) * ln2
    mul.f32 %lg, %x, 0f3F000000;          // 0.5*x
    lg2.approx.f32 %lg, %lg;
    mul.f32 %lg, %lg, 0f3F317218;          // * ln2
    mul.f32 %r, %lg, %ibe;
    sub.f32 %r, %cba, %r;                 // chbevl(A) - log(0.5x)*i0
    // scaled: * exp(x)
    mul.f32 %ex, %x, 0f3FB8AA3B;          // x*log2e
    ex2.approx.f32 %ex, %ex;
    mul.f32 %r, %r, %ex;
    bra STORE;

BIG:
    // BIG (x > 2): chbevl(8/x - 2, K0_B) / sqrt(x)
    rcp.rn.f32 %xx, %x;
    mul.f32 %xx, %xx, 0f41000000;          // 8/x
    add.f32 %xx, %xx, 0fC0000000;          // -2.0
    mov.f32 %b0, 0f22C38D2E;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA397F665;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f24704972;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA5417CA4;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f261EDFCB;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA7053487;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f27E46CE2;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA8C8A2F6;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f29B4E276;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAAA7C3B9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2BA07D19;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAC9ED7E2;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2DA34042;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAEAEFCAF;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2FC4AB64;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB0E94EB0;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f32134D00;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB3482AA1;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f34948ACC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB5F5DAA4;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f376A09FE;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB906BCBF;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3ACDC490;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBD00CFB9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f401C2DED;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %r, %cb;
    sqrt.rn.f32 %sq, %x;
    div.rn.f32 %r, %r, %sq;                // / sqrt(x)

STORE:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %r;
DONE:
    ret;
}
";

/// modified_bessel_k1(x), f32. x==0 -> +inf; x<0 -> NaN. SMALL (x<=2): log(0.5x)*i1(x) + chbevl(x*x-2, K1_A)/x, where i1(x)=exp(x)*x*chbevl(x*0.5-2, I1E_A). BIG (x>2): exp(-x)*chbevl(8/x-2, K1_B)/sqrt(x). Mirrors modified_bessel_k1_forward (aten/src/ATen/native/cuda/Math.cuh:2661-2736).
/// ABI: `(in, out, total)`.
#[cfg(feature = "cuda")]
pub(crate) const K1_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

// NOTE: PTX comments must stay ASCII-only. The WSL driver (591.86) JIT parser
// rejects non-ASCII bytes with CUDA_ERROR_INVALID_PTX. Coefficient immediates
// are the f32 bit-hex of the Cephes A/B Chebyshev tables (special.rs K0_A.. /
// I0E_A.. for the inner i0/i1 of the small-region log term).

.visible .entry modified_bessel_k1_f32(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %x, %xx, %xi, %b0, %b1, %b2, %cb, %cba, %ibe, %ex, %lg, %r, %sq;
    .reg .pred %oob, %small, %zero, %neg;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %x, [%addr];

    // x == 0 -> +inf
    setp.eq.f32 %zero, %x, 0f00000000;
    @%zero mov.f32 %r, 0f7F800000;
    @%zero bra STORE;
    // x < 0 -> NaN
    setp.lt.f32 %neg, %x, 0f00000000;
    @%neg mov.f32 %r, 0f7FC00000;
    @%neg bra STORE;

    setp.le.f32 %small, %x, 0f40000000;   // x <= 2.0
    @!%small bra BIG;
    // SMALL (x <= 2): chbevl(x*x - 2, K1_A)
    mul.f32 %xx, %x, %x;
    add.f32 %xx, %xx, 0fC0000000;          // -2.0
    mov.f32 %b0, 0fA3019142;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA72EEA8C;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAB3BA817;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAF1B31DE;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB2BE20E9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB6234D99;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB9356F17;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBBE4949D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBDFB1B92;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBEB4D0DC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3FC33D0A;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %cba, %cb;                    // chbevl(x*x-2, A)
    // inner i1(x) = exp(x) * chbevl(x*0.5 - 2, I1E_A) * x
    mul.f32 %xi, %x, 0f3F000000;          // x*0.5
    add.f32 %xi, %xi, 0fC0000000;          // -2.0
    mov.f32 %b0, 0f224CF950;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA3C2BE86;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f25331F1F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA69F5554;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2808EBF8;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA9631471;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2AB57BC2;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAC0B9C1B;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2D4E7716;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAE92881D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2FC751A6;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB101B0D9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f32212C70;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB33EE9F1;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f34571A26;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB56603CC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3668E277;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB75EAFCE;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f38488DAA;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB9299E57;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3A064AEE;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBAC66310;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3B88329A;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBC2D14FC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3CCA8F1F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBD58DDE3;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3DD236D7;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBE34A688;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3E81531C;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %ibe, %cb;
    mul.f32 %ex, %x, 0f3FB8AA3B;          // x*log2e
    ex2.approx.f32 %ex, %ex;              // exp(x)
    mul.f32 %ibe, %ibe, %ex;
    mul.f32 %ibe, %ibe, %x;              // i1 odd: * x
    // log(0.5*x) = lg2.approx(0.5*x) * ln2
    mul.f32 %lg, %x, 0f3F000000;          // 0.5*x
    lg2.approx.f32 %lg, %lg;
    mul.f32 %lg, %lg, 0f3F317218;          // * ln2
    mul.f32 %r, %lg, %ibe;               // log(0.5x)*i1
    div.rn.f32 %cba, %cba, %x;            // chbevl(A)/x
    add.f32 %r, %r, %cba;
    bra STORE;

BIG:
    // BIG (x > 2): chbevl(8/x - 2, K1_B) / sqrt(x) * exp(-x)
    rcp.rn.f32 %xx, %x;
    mul.f32 %xx, %xx, 0f41000000;          // 8/x
    add.f32 %xx, %xx, 0fC0000000;          // -2.0
    mov.f32 %b0, 0fA2D462EA;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f23A578D4;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA48330A9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2553EAF2;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA62E951F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2712E76F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA7FCEA92;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f28DF1F4B;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA9CA232F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2ABC7FD0;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fABB56F63;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2CB4D5C2;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fADBB5A9D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2ECAB680;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAFE65E00;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f310A7921;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB231BD25;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f337693E3;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB4BC0294;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f36217F19;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB7A26B89;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f394CB2C4;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBB3B4A36;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3DD4D5F8;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f402E1EBD;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %r, %cb;
    sqrt.rn.f32 %sq, %x;
    div.rn.f32 %r, %r, %sq;                // / sqrt(x)
    // * exp(-x)
    mul.f32 %ex, %x, 0fBFB8AA3B;          // -x*log2e
    ex2.approx.f32 %ex, %ex;              // exp(-x)
    mul.f32 %r, %r, %ex;

STORE:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %r;
DONE:
    ret;
}
";

/// scaled_modified_bessel_k1(x)=exp(x)*k1(x), f32. SMALL: (log(0.5x)*i1(x) + chbevl(x*x-2, K1_A)/x)*exp(x). BIG: chbevl(8/x-2, K1_B)/sqrt(x). Mirrors scaled_modified_bessel_k1_forward (aten/src/ATen/native/cuda/Math.cuh:2740-2815).
/// ABI: `(in, out, total)`.
#[cfg(feature = "cuda")]
pub(crate) const SCALED_K1_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

// NOTE: PTX comments must stay ASCII-only. The WSL driver (591.86) JIT parser
// rejects non-ASCII bytes with CUDA_ERROR_INVALID_PTX. Coefficient immediates
// are the f32 bit-hex of the Cephes A/B Chebyshev tables (special.rs K0_A.. /
// I0E_A.. for the inner i0/i1 of the small-region log term).

.visible .entry scaled_modified_bessel_k1_f32(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %x, %xx, %xi, %b0, %b1, %b2, %cb, %cba, %ibe, %ex, %lg, %r, %sq;
    .reg .pred %oob, %small, %zero, %neg;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %x, [%addr];

    // x == 0 -> +inf
    setp.eq.f32 %zero, %x, 0f00000000;
    @%zero mov.f32 %r, 0f7F800000;
    @%zero bra STORE;
    // x < 0 -> NaN
    setp.lt.f32 %neg, %x, 0f00000000;
    @%neg mov.f32 %r, 0f7FC00000;
    @%neg bra STORE;

    setp.le.f32 %small, %x, 0f40000000;   // x <= 2.0
    @!%small bra BIG;
    // SMALL (x <= 2): chbevl(x*x - 2, K1_A)
    mul.f32 %xx, %x, %x;
    add.f32 %xx, %xx, 0fC0000000;          // -2.0
    mov.f32 %b0, 0fA3019142;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA72EEA8C;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAB3BA817;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAF1B31DE;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB2BE20E9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB6234D99;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB9356F17;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBBE4949D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBDFB1B92;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBEB4D0DC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3FC33D0A;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %cba, %cb;                    // chbevl(x*x-2, A)
    // inner i1(x) = exp(x) * chbevl(x*0.5 - 2, I1E_A) * x
    mul.f32 %xi, %x, 0f3F000000;          // x*0.5
    add.f32 %xi, %xi, 0fC0000000;          // -2.0
    mov.f32 %b0, 0f224CF950;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA3C2BE86;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f25331F1F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA69F5554;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2808EBF8;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA9631471;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2AB57BC2;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAC0B9C1B;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2D4E7716;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAE92881D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2FC751A6;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB101B0D9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f32212C70;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB33EE9F1;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f34571A26;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB56603CC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3668E277;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB75EAFCE;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f38488DAA;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB9299E57;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3A064AEE;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBAC66310;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3B88329A;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBC2D14FC;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3CCA8F1F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBD58DDE3;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3DD236D7;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBE34A688;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xi, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3E81531C;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %ibe, %cb;
    mul.f32 %ex, %x, 0f3FB8AA3B;          // x*log2e
    ex2.approx.f32 %ex, %ex;              // exp(x)
    mul.f32 %ibe, %ibe, %ex;
    mul.f32 %ibe, %ibe, %x;              // i1 odd: * x
    // log(0.5*x) = lg2.approx(0.5*x) * ln2
    mul.f32 %lg, %x, 0f3F000000;          // 0.5*x
    lg2.approx.f32 %lg, %lg;
    mul.f32 %lg, %lg, 0f3F317218;          // * ln2
    mul.f32 %r, %lg, %ibe;               // log(0.5x)*i1
    div.rn.f32 %cba, %cba, %x;            // chbevl(A)/x
    add.f32 %r, %r, %cba;
    // scaled: * exp(x)
    mul.f32 %ex, %x, 0f3FB8AA3B;          // x*log2e
    ex2.approx.f32 %ex, %ex;
    mul.f32 %r, %r, %ex;
    bra STORE;

BIG:
    // BIG (x > 2): chbevl(8/x - 2, K1_B) / sqrt(x)
    rcp.rn.f32 %xx, %x;
    mul.f32 %xx, %xx, 0f41000000;          // 8/x
    add.f32 %xx, %xx, 0fC0000000;          // -2.0
    mov.f32 %b0, 0fA2D462EA;
    mov.f32 %b1, 0f00000000;
    mov.f32 %b2, 0f00000000;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f23A578D4;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA48330A9;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2553EAF2;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA62E951F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2712E76F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA7FCEA92;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f28DF1F4B;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fA9CA232F;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2ABC7FD0;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fABB56F63;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2CB4D5C2;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fADBB5A9D;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f2ECAB680;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fAFE65E00;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f310A7921;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB231BD25;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f337693E3;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB4BC0294;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f36217F19;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fB7A26B89;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f394CB2C4;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0fBB3B4A36;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f3DD4D5F8;
    mov.f32 %b2, %b1;
    mov.f32 %b1, %b0;
    mul.f32 %b0, %xx, %b1;
    sub.f32 %b0, %b0, %b2;
    add.f32 %b0, %b0, 0f402E1EBD;
    sub.f32 %cb, %b0, %b2;
    mul.f32 %cb, %cb, 0f3F000000;
    mov.f32 %r, %cb;
    sqrt.rn.f32 %sq, %x;
    div.rn.f32 %r, %r, %sq;                // / sqrt(x)

STORE:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %r;
DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// PTX - Airy Ai (#1651 GPU tail) + Hurwitz zeta (#1651 GPU tail)
// ---------------------------------------------------------------------------
//
// f32-only on the GPU (base PTX lacks lg2/ex2.approx.f64; f64 ->
// NotImplementedOnCuda, same constraint as the earlier #1651 batches). The f32
// math mirrors the ferrotorch CPU f64 scalar evaluators (`airy_ai_f64` /
// `zeta_f64` in ferrotorch-core/src/special.rs, themselves verbatim ports of
// the Cephes kernels at aten/src/ATen/native/cuda/Math.cuh:1280-1459 (airy)
// and :299-383 (zeta)) narrowed to f32. Both kernels are BOUNDED:
//
//   - airy_ai's central Maclaurin series `while (t > MACHEP)` (Math.cuh:1437)
//     is only reached for the bounded x in [-2.09, 8.3203353] window; over that
//     entire window it converges in <= 30 terms (f64) and far fewer in f32, so
//     a FIXED 36-iteration unrolled counted loop reproduces the unbounded
//     reference bit-for-relevant-tolerance (verified: 0 rel err vs the f64
//     reference across x in [-2.2, 10.5] step 0.01, and <=2.2e-8 rel vs live
//     torch f32). The oscillatory (x<-2.09) and decaying (x>8.32) regions are
//     already flat polynomial Horner chains in upstream.
//
//   - zeta's `while ((i < 9) || (a <= 9.0))` (Math.cuh:349) ALWAYS terminates
//     at exactly i==9 because a = q + 9 > 9 for any q > 0 by then, so it is a
//     FIXED 9-iteration counted loop with a per-iteration MACHEP-relative
//     early-exit `converged` flag (Math.cuh:354-356); the Euler-Maclaurin tail
//     (Math.cuh:364-379) is a FIXED 12-term loop over the ZETA_A table with the
//     same relative early-exit flag. Once `converged` is set the accumulation
//     is guarded off and the tail is skipped, exactly mirroring the upstream
//     early `return s`.
//
// `pow(a, -x)` (Math.cuh:347) is evaluated on-device as
// `a^(-x) = 2^(-x * log2(a)) = ex2.approx.f32(-x * lg2.approx.f32(a))`; `exp`,
// `sin`, `cos`, `sqrt` use `ex2.approx.f32`(* log2e) / `sin.approx.f32` /
// `cos.approx.f32` / `sqrt.rn.f32`. PTX comments stay ASCII-only (the WSL
// driver JIT rejects non-ASCII bytes with CUDA_ERROR_INVALID_PTX).

/// Airy function of the first kind `Ai(x)`, f32. Multi-region Cephes kernel
/// (`aten/src/ATen/native/cuda/Math.cuh:1280-1459`, `airy_ai_forward`): `isinf
/// -> NaN`; `x > 103.892 -> 0`; `x < -2.09` oscillatory asymptotic (AFN/AFD +
/// AGN/AGD Horner chains over `z2`); `x >= 2.09` decaying asymptotic (AN/AD over
/// `1/zeta`, early-return for `x > 8.3203353`); else the central Maclaurin
/// `f`/`g` series (FIXED 36-iter unroll). ABI: `(in, out, total)`.
#[cfg(feature = "cuda")]
pub(crate) const AIRY_AI_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

// NOTE: PTX comments must stay ASCII-only (WSL driver JIT rejects non-ASCII
// with CUDA_ERROR_INVALID_PTX). Coefficient immediates are the f32 bit-hex of
// the Cephes airy AN/AD/AFN/AFD/AGN/AGD tables (special.rs AIRY_*).

.visible .entry airy_ai_f32(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r, %kk;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %x, %r, %ai, %z, %z2, %iz, %sx, %t, %st, %ct;
    .reg .f32 %afn, %afd, %agn, %agd, %an, %ad, %zeta;
    .reg .f32 %ff, %g, %k, %m, %n, %zc, %tmp, %c0, %ai0, %nai0;
    .reg .pred %oob, %isinf, %big0, %osc, %dec, %df1, %fardec, %loop;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %x, [%addr];

    mov.f32 %c0,   0f3F106EBB;     // 5.64189583547756286948e-01 = 1/(2*sqrt(pi))
    mov.f32 %ai0,  0f3EB5C63D;     // 0.355028053887817239260 = Ai(0)
    mov.f32 %nai0, 0f3E8483FA;     // 0.258819403792806798405 = -Ai'(0)

    // isinf(x) -> NaN (Math.cuh:1360-1362). |x| == +inf.
    abs.f32 %tmp, %x;
    setp.eq.f32 %isinf, %tmp, 0f7F800000;
    @%isinf mov.f32 %r, 0f7FC00000;
    @%isinf bra STORE;
    // NaN input -> NaN: the region comparisons below all fail (NaN compares
    // false), so a NaN x routes through the central series, which propagates
    // NaN through the arithmetic to a NaN result, matching torch.

    // x > 103.892 -> 0 (Math.cuh:1364-1366).
    setp.gt.f32 %big0, %x, 0f42CFC8B4;
    @%big0 mov.f32 %r, 0f00000000;
    @%big0 bra STORE;

    // df1 encodes domain_flag&1: false everywhere except the decaying region
    // (domain_flag = 5, &1 == 1). Initialise FALSE via an always-false compare.
    setp.gt.f32 %df1, %x, 0f7F800000;   // x > +inf is always false -> df1 = 0
    mov.f32 %ai, 0f00000000;

    // x < -2.09 -> oscillatory asymptotic region (Math.cuh:1372-1402).
    setp.lt.f32 %osc, %x, 0fC005C28F;   // -2.09
    @!%osc bra AFTER_OSC;
    // t = -2*x*sqrt(-x)/3 ; z = 1/t ; z2 = z*z
    neg.f32 %tmp, %x;
    sqrt.rn.f32 %sx, %tmp;              // sqrt(-x)
    mul.f32 %t, %x, %sx;                // x*sqrt(-x)
    mov.f32 %tmp, 0fC0000000;           // -2.0
    mul.f32 %t, %t, %tmp;               // -2*x*sqrt(-x)
    mov.f32 %tmp, 0f3EAAAAAB;           // 1/3
    mul.f32 %t, %t, %tmp;               // -2*x*sqrt(-x)/3
    rcp.rn.f32 %z, %t;                  // z = 1/(-2*x*sqrt(-x)/3)
    mul.f32 %z2, %z, %z;
    // AFN Horner over z2
    mov.f32 %afn, 0fBE06DB67;
    fma.rn.f32 %afn, %afn, %z2, 0fBF205F75;
    fma.rn.f32 %afn, %afn, %z2, 0fBF3172CE;
    fma.rn.f32 %afn, %afn, %z2, 0fBE8F3F52;
    fma.rn.f32 %afn, %afn, %z2, 0fBD497B78;
    fma.rn.f32 %afn, %afn, %z2, 0fBB85200E;
    fma.rn.f32 %afn, %afn, %z2, 0fB9270375;
    fma.rn.f32 %afn, %afn, %z2, 0fB63A53C1;
    fma.rn.f32 %afn, %afn, %z2, 0fB29020E8;
    // AFD Horner over z2
    mov.f32 %afd, 0f4155B259;
    fma.rn.f32 %afd, %afd, %z2, 0f4202BAE2;
    fma.rn.f32 %afd, %afd, %z2, 0f41D5E4C5;
    fma.rn.f32 %afd, %afd, %z2, 0f4112FE41;
    fma.rn.f32 %afd, %afd, %z2, 0f3FBCD65A;
    fma.rn.f32 %afd, %afd, %z2, 0f3DECED66;
    fma.rn.f32 %afd, %afd, %z2, 0f3B904657;
    fma.rn.f32 %afd, %afd, %z2, 0f389E46BD;
    fma.rn.f32 %afd, %afd, %z2, 0f34F295CE;
    // AGN Horner over z2
    mov.f32 %agn, 0f3CA1A92F;
    fma.rn.f32 %agn, %agn, %z2, 0f3EC83EA8;
    fma.rn.f32 %agn, %agn, %z2, 0f3F886C1A;
    fma.rn.f32 %agn, %agn, %z2, 0f3F706D65;
    fma.rn.f32 %agn, %agn, %z2, 0f3EB3F34E;
    fma.rn.f32 %agn, %agn, %z2, 0f3D81D209;
    fma.rn.f32 %agn, %agn, %z2, 0f3BBFF4D0;
    fma.rn.f32 %agn, %agn, %z2, 0f39944BB3;
    fma.rn.f32 %agn, %agn, %z2, 0f36EA79EB;
    fma.rn.f32 %agn, %agn, %z2, 0f33AE5496;
    fma.rn.f32 %agn, %agn, %z2, 0f2FBBC524;
    // AGD Horner over z2
    mov.f32 %agd, 0f4114F160;
    fma.rn.f32 %agd, %agd, %z2, 0f419EAEAE;
    fma.rn.f32 %agd, %agd, %z2, 0f417908DC;
    fma.rn.f32 %agd, %agd, %z2, 0f40AF4271;
    fma.rn.f32 %agd, %agd, %z2, 0f3F744C96;
    fma.rn.f32 %agd, %agd, %z2, 0f3DB110EF;
    fma.rn.f32 %agd, %agd, %z2, 0f3B873823;
    fma.rn.f32 %agd, %agd, %z2, 0f38D45B0F;
    fma.rn.f32 %agd, %agd, %z2, 0f359D422F;
    fma.rn.f32 %agd, %agd, %z2, 0f31A8FE3A;
    // t_arg = -2*x*sqrt(-x)/3 + 0.25*pi  (reuse %t which currently holds the
    // -2*x*sqrt(-x)/3 value; add 0.25*pi).
    mov.f32 %tmp, 0f3E800000;           // 0.25
    mov.f32 %st,  0f40490FDB;           // pi
    fma.rn.f32 %t, %tmp, %st, %t;       // 0.25*pi + t
    sin.approx.f32 %st, %t;             // sin(t_arg)
    cos.approx.f32 %ct, %t;             // cos(t_arg)
    // result = c0 / sqrt(sqrt(-x)) * (sin*(1 + z2*afn/afd) - cos*(z*agn/agd))
    sqrt.rn.f32 %tmp, %sx;              // sqrt(sqrt(-x))  (sx = sqrt(-x))
    rcp.rn.f32 %tmp, %tmp;              // 1/sqrt(sqrt(-x))
    mul.f32 %tmp, %tmp, %c0;            // c0/sqrt(sqrt(-x))
    // term_a = 1 + z2*afn/afd
    div.rn.f32 %afn, %afn, %afd;
    mul.f32 %afn, %afn, %z2;
    add.f32 %afn, %afn, 0f3F800000;     // 1 + z2*afn/afd
    mul.f32 %st, %st, %afn;             // sin*(...)
    // term_b = z*agn/agd
    div.rn.f32 %agn, %agn, %agd;
    mul.f32 %agn, %agn, %z;
    mul.f32 %ct, %ct, %agn;             // cos*(...)
    sub.f32 %r, %st, %ct;
    mul.f32 %r, %r, %tmp;
    bra STORE;

AFTER_OSC:
    // x >= 2.09 -> decaying asymptotic region (Math.cuh:1404-1426).
    setp.lt.f32 %dec, %x, 0f4005C28F;   // x < 2.09 ?
    @%dec bra CENTRAL;                  // x < 2.09 -> skip decaying, dec=true
    // here x >= 2.09. set dec = (domain_flag&1 == 1) i.e. TRUE for this branch.
    // zeta = 2*x*sqrt(x)/3 ; iz = 1/zeta
    sqrt.rn.f32 %sx, %x;                // sqrt(x)
    mul.f32 %zeta, %x, %sx;             // x*sqrt(x)
    add.f32 %zeta, %zeta, %zeta;        // 2*x*sqrt(x)
    mov.f32 %tmp, 0f3EAAAAAB;           // 1/3
    mul.f32 %zeta, %zeta, %tmp;         // 2*x*sqrt(x)/3
    rcp.rn.f32 %iz, %zeta;
    // AN Horner over iz
    mov.f32 %an, 0f3EB16D71;
    fma.rn.f32 %an, %an, %iz, 0f41401F1C;
    fma.rn.f32 %an, %an, %iz, 0f42988F28;
    fma.rn.f32 %an, %an, %iz, 0f432816D7;
    fma.rn.f32 %an, %an, %iz, 0f431FC1A3;
    fma.rn.f32 %an, %an, %iz, 0f428D127A;
    fma.rn.f32 %an, %an, %iz, 0f41606C6B;
    fma.rn.f32 %an, %an, %iz, 0f3F800000;
    // AD Horner over iz
    mov.f32 %ad, 0f3F114DE0;
    fma.rn.f32 %ad, %ad, %iz, 0f416C19A0;
    fma.rn.f32 %ad, %ad, %iz, 0f42A9071E;
    fma.rn.f32 %ad, %ad, %iz, 0f4331516E;
    fma.rn.f32 %ad, %ad, %iz, 0f43243C15;
    fma.rn.f32 %ad, %ad, %iz, 0f428EF4A7;
    fma.rn.f32 %ad, %ad, %iz, 0f416188DD;
    fma.rn.f32 %ad, %ad, %iz, 0f3F800000;
    // ai = c0 * (an/ad) / (2 * sqrt(sqrt(x)) * exp(zeta))
    div.rn.f32 %ai, %an, %ad;
    mul.f32 %ai, %ai, %c0;
    sqrt.rn.f32 %tmp, %sx;              // sqrt(sqrt(x))
    add.f32 %tmp, %tmp, %tmp;           // 2*sqrt(sqrt(x))
    // exp(zeta) = ex2.approx.f32(zeta*log2e)
    mov.f32 %st, 0f3FB8AA3B;            // log2e
    mul.f32 %st, %zeta, %st;
    ex2.approx.f32 %st, %st;            // exp(zeta)
    mul.f32 %tmp, %tmp, %st;            // 2*sqrt(sqrt(x))*exp(zeta)
    div.rn.f32 %ai, %ai, %tmp;
    // domain_flag = 5 here -> &1 == 1 -> df1 TRUE (final result is `ai`).
    setp.eq.f32 %df1, %x, %x;          // df1 = TRUE (x finite/non-NaN here)
    // x > 8.3203353 -> return ai immediately (Math.cuh:1423-1425).
    setp.gt.f32 %fardec, %x, 0f41052018;
    @%fardec mov.f32 %r, %ai;
    @%fardec bra STORE;
    // else fall through to the central series; final result is `ai` (df1 TRUE).

CENTRAL:
    // Central Maclaurin f/g series (Math.cuh:1428-1457), FIXED 36-iter unroll.
    // f=1; g=x; k=1; m=1; n=x; zc=x*x*x. Per iter:
    //   m*=zc; k+=1; m/=k; n*=zc; k+=1; n/=k; m/=k; f+=m; k+=1; n/=k; g+=n
    mov.f32 %ff, 0f3F800000;           // f = 1
    mov.f32 %g, %x;                    // g = x
    mov.f32 %k, 0f3F800000;            // k = 1
    mov.f32 %m, 0f3F800000;            // m = 1
    mov.f32 %n, %x;                    // n = x
    mul.f32 %zc, %x, %x;
    mul.f32 %zc, %zc, %x;              // zc = x^3
    mov.u32 %kk, 0;
CLOOP:
    setp.ge.u32 %loop, %kk, 36;
    @%loop bra CDONE;
    mul.f32 %m, %m, %zc;               // m *= zc
    add.f32 %k, %k, 0f3F800000;        // k += 1
    div.rn.f32 %m, %m, %k;             // m /= k
    mul.f32 %n, %n, %zc;               // n *= zc
    add.f32 %k, %k, 0f3F800000;        // k += 1
    div.rn.f32 %n, %n, %k;             // n /= k
    div.rn.f32 %m, %m, %k;             // m /= k
    add.f32 %ff, %ff, %m;              // f += m
    add.f32 %k, %k, 0f3F800000;        // k += 1
    div.rn.f32 %n, %n, %k;             // n /= k
    add.f32 %g, %g, %n;                // g += n
    add.u32 %kk, %kk, 1;
    bra CLOOP;
CDONE:
    // if (domain_flag & 1) == 0 -> return ai0*f - nai0*g ; else return ai.
    @%df1 mov.f32 %r, %ai;
    @%df1 bra STORE;
    mul.f32 %r, %ai0, %ff;
    mul.f32 %tmp, %nai0, %g;
    sub.f32 %r, %r, %tmp;

STORE:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %r;
DONE:
    ret;
}
";

/// Hurwitz zeta `zeta(x, q) = sum_{k=0}^inf (k+q)^{-x}`, f32, binary. Direct
/// port of the Cephes `zeta_string` (`aten/src/ATen/native/cuda/Math.cuh:
/// 299-383`) in f32: the edge ladder (`x==1 -> +inf`, `x<1 -> NaN`, `q<=0`
/// integer `-> +inf`, `q<=0` non-integer with non-integer `x -> NaN`), a FIXED
/// 9-iteration first-sum loop with a relative-MACHEP `converged` early-exit
/// flag, and the FIXED 12-term Euler-Maclaurin tail over the ZETA_A table (same
/// early-exit). `pow(a,-x) = ex2.approx.f32(-x * lg2.approx.f32(a))`.
/// ABI: `(in_x, in_q, out, total)`.
#[cfg(feature = "cuda")]
pub(crate) const ZETA_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

// NOTE: PTX comments must stay ASCII-only (WSL driver JIT rejects non-ASCII
// with CUDA_ERROR_INVALID_PTX). ZETA_A immediates are the f32 bit-hex of the
// Cephes Euler-Maclaurin coefficient table (special.rs ZETA_A).

.visible .entry zeta_f32(
    .param .u64 inx_ptr,
    .param .u64 inq_ptr,
    .param .u64 out_ptr,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r, %i;
    .reg .u64 %inx_p, %inq_p, %out_p, %off, %addr;
    .reg .f32 %x, %q, %s, %a, %b, %w, %t, %k, %r, %eps, %one, %tmp, %xm1;
    .reg .pred %oob, %xeq1, %xlt1, %qle0, %qint, %xint, %conv, %term;

    ld.param.u64 %inx_p,   [inx_ptr];
    ld.param.u64 %inq_p,   [inq_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %inx_p, %off;
    ld.global.f32 %x, [%addr];
    add.u64 %addr, %inq_p, %off;
    ld.global.f32 %q, [%addr];

    mov.f32 %one, 0f3F800000;          // 1.0
    mov.f32 %eps, 0f34000000;          // f32 MACHEP 1.1920929e-07

    // x == 1 -> +inf (Math.cuh:325-327).
    setp.eq.f32 %xeq1, %x, %one;
    @%xeq1 mov.f32 %r, 0f7F800000;
    @%xeq1 bra STORE;
    // x < 1 -> NaN (Math.cuh:330-332).
    setp.lt.f32 %xlt1, %x, %one;
    @%xlt1 mov.f32 %r, 0f7FC00000;
    @%xlt1 bra STORE;
    // q <= 0 ladder (Math.cuh:336-343).
    setp.le.f32 %qle0, %q, 0f00000000;
    @!%qle0 bra MAIN;
    cvt.rmi.f32.f32 %tmp, %q;          // floor(q)
    setp.eq.f32 %qint, %tmp, %q;       // q is integer
    @%qint mov.f32 %r, 0f7F800000;     // q<=0 integer -> +inf
    @%qint bra STORE;
    cvt.rmi.f32.f32 %tmp, %x;          // floor(x)
    setp.ne.f32 %xint, %tmp, %x;       // x non-integer
    @%xint mov.f32 %r, 0f7FC00000;     // q<=0 non-int, x non-int -> NaN
    @%xint bra STORE;

MAIN:
    // s = pow(q, -x) = ex2(-x * lg2(q)); a = q; conv = false.
    lg2.approx.f32 %tmp, %q;
    mul.f32 %tmp, %tmp, %x;
    neg.f32 %tmp, %tmp;                // -x*log2(q)
    ex2.approx.f32 %s, %tmp;           // q^(-x)
    mov.f32 %a, %q;
    mov.u32 %i, 0;
    setp.ne.f32 %conv, %x, %x;         // conv = false (x==x is true -> ne false)
    mov.f32 %b, 0f00000000;
SLOOP:
    setp.ge.u32 %oob, %i, 9;
    @%oob bra AFTER_SLOOP;
    @%conv bra SLOOP_NEXT;             // once converged, stop accumulating
    add.f32 %a, %a, %one;              // a += 1
    lg2.approx.f32 %tmp, %a;
    mul.f32 %tmp, %tmp, %x;
    neg.f32 %tmp, %tmp;
    ex2.approx.f32 %b, %tmp;           // b = a^(-x)
    add.f32 %s, %s, %b;                // s += b
    // converged if |b| < eps*|s|  (Math.cuh:354-356, relative MACHEP test).
    abs.f32 %tmp, %s;
    mul.f32 %tmp, %tmp, %eps;          // eps*|s|
    abs.f32 %r, %b;                    // |b| (reuse %r as scratch)
    setp.lt.f32 %conv, %r, %tmp;
SLOOP_NEXT:
    add.u32 %i, %i, 1;
    bra SLOOP;
AFTER_SLOOP:
    @%conv mov.f32 %r, %s;             // converged -> return s (skip tail)
    @%conv bra STORE;

    // Euler-Maclaurin tail (Math.cuh:359-379).
    // w = a; s += b*w/(x-1); s -= 0.5*b; a = 1; k = 0.
    mov.f32 %w, %a;
    sub.f32 %xm1, %x, %one;            // x - 1
    mul.f32 %tmp, %b, %w;
    div.rn.f32 %tmp, %tmp, %xm1;
    add.f32 %s, %s, %tmp;              // s += b*w/(x-1)
    mov.f32 %tmp, 0f3F000000;          // 0.5
    mul.f32 %tmp, %tmp, %b;
    sub.f32 %s, %s, %tmp;              // s -= 0.5*b
    mov.f32 %a, %one;                  // a = 1
    mov.f32 %k, 0f00000000;            // k = 0
    setp.ne.f32 %conv, %x, %x;         // term-converged flag = false
    // unrolled 12-term loop over ZETA_A; per term:
    //   a *= x+k; b /= w; t = a*b/coeff; s += t; if |t/s|<eps break;
    //   k += 1; a *= x+k; b /= w; k += 1;
    mov.u32 %i, 0;
TLOOP:
    setp.ge.u32 %oob, %i, 12;
    @%oob bra AFTER_TLOOP;
    @%conv bra AFTER_TLOOP;            // term converged -> stop
    // a *= (x + k)
    add.f32 %tmp, %x, %k;
    mul.f32 %a, %a, %tmp;
    div.rn.f32 %b, %b, %w;             // b /= w
    // t = a*b/coeff  (coeff selected per-iter via the table below)
    mul.f32 %t, %a, %b;
    // load coeff[i] into %tmp via a small jump table.
    setp.eq.u32 %term, %i, 0;
    @%term mov.f32 %tmp, 0f41400000;   // 12
    setp.eq.u32 %term, %i, 1;
    @%term mov.f32 %tmp, 0fC4340000;   // -720
    setp.eq.u32 %term, %i, 2;
    @%term mov.f32 %tmp, 0f46EC4000;   // 30240
    setp.eq.u32 %term, %i, 3;
    @%term mov.f32 %tmp, 0fC993A800;   // -1209600
    setp.eq.u32 %term, %i, 4;
    @%term mov.f32 %tmp, 0f4C36B980;   // 47900160
    setp.eq.u32 %term, %i, 5;
    @%term mov.f32 %tmp, 0fCEE1989D;   // -1.8924375803183791606e9
    setp.eq.u32 %term, %i, 6;
    @%term mov.f32 %tmp, 0f518B2F4C;   // 7.47242496e10
    setp.eq.u32 %term, %i, 7;
    @%term mov.f32 %tmp, 0fD42BB860;   // -2.950130727918164224e12
    setp.eq.u32 %term, %i, 8;
    @%term mov.f32 %tmp, 0f56D3DA8F;   // 1.1646782814350067249e14
    setp.eq.u32 %term, %i, 9;
    @%term mov.f32 %tmp, 0fD982AEB3;   // -4.5979787224074726105e15
    setp.eq.u32 %term, %i, 10;
    @%term mov.f32 %tmp, 0f5C21391C;   // 1.8152105401943546773e17
    setp.eq.u32 %term, %i, 11;
    @%term mov.f32 %tmp, 0fDEC6E6AB;   // -7.1661652561756670113e18
    div.rn.f32 %t, %t, %tmp;           // t = a*b/coeff
    add.f32 %s, %s, %t;                // s += t
    // term converged if |t/s| < eps
    div.rn.f32 %tmp, %t, %s;
    abs.f32 %tmp, %tmp;
    setp.lt.f32 %conv, %tmp, %eps;
    @%conv bra AFTER_TLOOP;
    // k += 1; a *= (x + k); b /= w; k += 1
    add.f32 %k, %k, %one;
    add.f32 %tmp, %x, %k;
    mul.f32 %a, %a, %tmp;
    div.rn.f32 %b, %b, %w;
    add.f32 %k, %k, %one;
    add.u32 %i, %i, 1;
    bra TLOOP;
AFTER_TLOOP:
    mov.f32 %r, %s;

STORE:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %r;
DONE:
    ret;
}
";

/// Spherical Bessel `j0(x)` forward on an f32 buffer (on-device, no host
/// round-trip).
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_spherical_bessel_j0_f32(
    input: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    launch_elementwise_f32(
        input,
        device,
        SPHERICAL_BESSEL_J0_F32_PTX,
        "spherical_bessel_j0_f32",
    )
}

/// Launch a parameterless elementwise transcendental kernel (entr / ndtr /
/// ndtri) whose ABI is `(in, out, total)` on an f32 buffer.
#[cfg(feature = "cuda")]
fn launch_elementwise_f32(
    input: &CudaBuffer<f32>,
    device: &GpuDevice,
    ptx: &'static str,
    kernel: &'static str,
) -> GpuResult<CudaBuffer<f32>> {
    let total = check_input(input, device, kernel)?;
    let mut out = alloc_zeros_f32(total, device)?;
    if total == 0 {
        return Ok(out);
    }
    let f =
        crate::module_cache::get_or_compile(device.context(), ptx, kernel, device.ordinal() as u32)
            .map_err(|e| GpuError::PtxCompileFailed { kernel, source: e })?;
    let total_u32 = total as u32;
    let cfg = launch_cfg(total_u32);
    // SAFETY: `f` is `kernel` (one of the `(in, out, total)`-ABI elementwise
    // transcendental entries); the three launch args match the PTX `.entry`
    // order. `input` is on `device` with `total` f32 elements (validated by
    // `check_input`); `out` is a fresh non-aliasing `total`-element buffer;
    // every thread is bounds-guarded by the `setp.ge.u32` head; `total` fits
    // u32. By-ref params outlive the synchronous launch on this stack frame.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&total_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Entropy `entr(x)` forward on an f32 buffer (on-device, no host round-trip).
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_entr_f32(input: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    launch_elementwise_f32(input, device, ENTR_F32_PTX, "entr_f32")
}

/// Standard-normal CDF `ndtr(x)` forward on an f32 buffer (on-device).
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_ndtr_f32(input: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    launch_elementwise_f32(input, device, NDTR_F32_PTX, "ndtr_f32")
}

/// Inverse standard-normal CDF `ndtri(p)` forward on an f32 buffer (on-device).
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_ndtri_f32(input: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    launch_elementwise_f32(input, device, NDTRI_F32_PTX, "ndtri_f32")
}

/// Modified Bessel `i0(x)` forward on an f32 buffer (on-device, no host round-trip).
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_i0_f32(input: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    launch_elementwise_f32(input, device, I0_F32_PTX, "i0_f32")
}

/// Exp-scaled modified Bessel `i0e(x)` forward on an f32 buffer (on-device).
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_i0e_f32(input: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    launch_elementwise_f32(input, device, I0E_F32_PTX, "i0e_f32")
}

/// Modified Bessel `i1(x)` forward on an f32 buffer (on-device).
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_i1_f32(input: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    launch_elementwise_f32(input, device, I1_F32_PTX, "i1_f32")
}

/// Exp-scaled modified Bessel `i1e(x)` forward on an f32 buffer (on-device).
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_i1e_f32(input: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    launch_elementwise_f32(input, device, I1E_F32_PTX, "i1e_f32")
}

/// Modified Bessel `modified_bessel_k0(x)` forward on an f32 buffer (on-device, no host round-trip).
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_modified_bessel_k0_f32(
    input: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    launch_elementwise_f32(input, device, K0_F32_PTX, "modified_bessel_k0_f32")
}

/// Modified Bessel `scaled_modified_bessel_k0(x)` forward on an f32 buffer (on-device, no host round-trip).
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_scaled_modified_bessel_k0_f32(
    input: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    launch_elementwise_f32(
        input,
        device,
        SCALED_K0_F32_PTX,
        "scaled_modified_bessel_k0_f32",
    )
}

/// Modified Bessel `modified_bessel_k1(x)` forward on an f32 buffer (on-device, no host round-trip).
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_modified_bessel_k1_f32(
    input: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    launch_elementwise_f32(input, device, K1_F32_PTX, "modified_bessel_k1_f32")
}

/// Modified Bessel `scaled_modified_bessel_k1(x)` forward on an f32 buffer (on-device, no host round-trip).
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_scaled_modified_bessel_k1_f32(
    input: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    launch_elementwise_f32(
        input,
        device,
        SCALED_K1_F32_PTX,
        "scaled_modified_bessel_k1_f32",
    )
}

/// Airy function of the first kind `Ai(x)` forward on an f32 buffer (on-device,
/// no host round-trip).
///
/// # Errors
/// See [`gpu_chebyshev_poly_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_airy_ai_f32(input: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    launch_elementwise_f32(input, device, AIRY_AI_F32_PTX, "airy_ai_f32")
}

/// `torch.special.xlogy(x, y)` forward, f32.
///
/// Branch order mirrors PyTorch's CUDA kernel:
/// `isnan(y) -> NaN`, then `x == 0 -> 0`, otherwise `x * log(y)`.
#[cfg(feature = "cuda")]
pub(crate) const XLOGY_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry xlogy_f32(
    .param .u64 x_ptr,
    .param .u64 y_ptr,
    .param .u64 out_ptr,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r;
    .reg .u64 %x_p, %y_p, %out_p, %off, %addr_x, %addr_y, %addr_out;
    .reg .f32 %x, %y, %r, %log2_y, %ln2;
    .reg .pred %oob, %y_nan, %x_zero;

    ld.param.u64 %x_p,     [x_ptr];
    ld.param.u64 %y_p,     [y_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr_x, %x_p, %off;
    add.u64 %addr_y, %y_p, %off;
    add.u64 %addr_out, %out_p, %off;
    ld.global.f32 %x, [%addr_x];
    ld.global.f32 %y, [%addr_y];

    setp.nan.f32 %y_nan, %y, %y;
    @%y_nan bra STORE_NAN;
    setp.eq.f32 %x_zero, %x, 0f00000000;
    @%x_zero bra STORE_ZERO;

    lg2.approx.f32 %log2_y, %y;
    mov.f32 %ln2, 0f3F317218;
    mul.f32 %r, %log2_y, %ln2;
    mul.f32 %r, %x, %r;
    bra STORE;

STORE_NAN:
    mov.f32 %r, 0f7FC00000;
    bra STORE;

STORE_ZERO:
    mov.f32 %r, 0f00000000;

STORE:
    st.global.f32 [%addr_out], %r;
DONE:
    ret;
}
";

/// `torch.special.xlogy(x, y)` forward, f64.
///
/// The logarithm follows the crate's existing f64 CUDA policy in
/// `gpu_log_f64`: downcast to f32 for the PTX `lg2.approx.f32` operation,
/// then widen back to f64 for storage. This keeps `xlogy_f64` consistent with
/// the rest of the current f64 CUDA transcendental backend.
#[cfg(feature = "cuda")]
pub(crate) const XLOGY_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry xlogy_f64(
    .param .u64 x_ptr,
    .param .u64 y_ptr,
    .param .u64 out_ptr,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %total_r;
    .reg .u64 %x_p, %y_p, %out_p, %off, %addr_x, %addr_y, %addr_out;
    .reg .f64 %x, %y, %r, %log_y64;
    .reg .f32 %y32, %log2_y, %log_y32, %ln2;
    .reg .pred %oob, %y_nan, %x_zero;

    ld.param.u64 %x_p,     [x_ptr];
    ld.param.u64 %y_p,     [y_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %oob, %idx, %total_r;
    @%oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 3;
    add.u64 %addr_x, %x_p, %off;
    add.u64 %addr_y, %y_p, %off;
    add.u64 %addr_out, %out_p, %off;
    ld.global.f64 %x, [%addr_x];
    ld.global.f64 %y, [%addr_y];

    setp.nan.f64 %y_nan, %y, %y;
    @%y_nan bra STORE_NAN;
    setp.eq.f64 %x_zero, %x, 0d0000000000000000;
    @%x_zero bra STORE_ZERO;

    cvt.rn.f32.f64 %y32, %y;
    lg2.approx.f32 %log2_y, %y32;
    mov.f32 %ln2, 0f3F317218;
    mul.f32 %log_y32, %log2_y, %ln2;
    cvt.f64.f32 %log_y64, %log_y32;
    mul.rn.f64 %r, %x, %log_y64;
    bra STORE;

STORE_NAN:
    mov.f64 %r, 0d7FF8000000000000;
    bra STORE;

STORE_ZERO:
    mov.f64 %r, 0d0000000000000000;

STORE:
    st.global.f64 [%addr_out], %r;
DONE:
    ret;
}
";

/// Launch a two-input elementwise kernel whose ABI is `(in_x, in_q, out,
/// total)` on a pair of equal-length f32 buffers (Hurwitz zeta). The result
/// stays on-device (no host round-trip, R-CODE-4).
///
/// Both inputs must be on `device` with the same element count; mismatched
/// lengths return [`GpuError::ShapeMismatch`].
#[cfg(feature = "cuda")]
fn launch_binary_elementwise_f32(
    x: &CudaBuffer<f32>,
    q: &CudaBuffer<f32>,
    device: &GpuDevice,
    ptx: &'static str,
    kernel: &'static str,
) -> GpuResult<CudaBuffer<f32>> {
    let total = check_input(x, device, kernel)?;
    let total_q = check_input(q, device, kernel)?;
    if total != total_q {
        return Err(GpuError::ShapeMismatch {
            op: kernel,
            expected: vec![total],
            got: vec![total_q],
        });
    }
    let mut out = alloc_zeros_f32(total, device)?;
    if total == 0 {
        return Ok(out);
    }
    let f =
        crate::module_cache::get_or_compile(device.context(), ptx, kernel, device.ordinal() as u32)
            .map_err(|e| GpuError::PtxCompileFailed { kernel, source: e })?;
    let total_u32 = total as u32;
    let cfg = launch_cfg(total_u32);
    // SAFETY: `f` is `kernel` (a `(in_x, in_q, out, total)`-ABI two-input
    // elementwise entry); the four launch args match the PTX `.entry` order.
    // `x` and `q` are on `device` with exactly `total` f32 elements each
    // (validated by `check_input` + the equal-length check above); `out` is a
    // fresh non-aliasing `total`-element buffer; every thread is bounds-guarded
    // by the `setp.ge.u32` head, so all loads/stores are in-bounds; `total`
    // fits u32. By-ref params outlive the synchronous launch on this frame.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(x.inner())
            .arg(q.inner())
            .arg(out.inner_mut())
            .arg(&total_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// f64 counterpart of [`launch_binary_elementwise_f32`].
#[cfg(feature = "cuda")]
fn launch_binary_elementwise_f64(
    x: &CudaBuffer<f64>,
    y: &CudaBuffer<f64>,
    device: &GpuDevice,
    ptx: &'static str,
    kernel: &'static str,
) -> GpuResult<CudaBuffer<f64>> {
    let total = check_input(x, device, kernel)?;
    let total_y = check_input(y, device, kernel)?;
    if total != total_y {
        return Err(GpuError::ShapeMismatch {
            op: kernel,
            expected: vec![total],
            got: vec![total_y],
        });
    }
    let mut out = alloc_zeros_f64(total, device)?;
    if total == 0 {
        return Ok(out);
    }
    let f =
        crate::module_cache::get_or_compile(device.context(), ptx, kernel, device.ordinal() as u32)
            .map_err(|e| GpuError::PtxCompileFailed { kernel, source: e })?;
    let total_u32 = total as u32;
    let cfg = launch_cfg(total_u32);
    // SAFETY: `f` is `kernel` (a `(x,y,out,total)`-ABI two-input elementwise
    // entry); the four launch args match the PTX `.entry` order. `x` and `y`
    // are on `device` with exactly `total` f64 elements each (validated by
    // `check_input` + equal-length check); `out` is fresh and non-aliasing;
    // the PTX head bounds-checks every thread; `total` fits u32.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(x.inner())
            .arg(y.inner())
            .arg(out.inner_mut())
            .arg(&total_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Hurwitz zeta `zeta(x, q)` forward on a pair of equal-length f32 buffers
/// (on-device, no host round-trip). `x` is the exponent buffer, `q` the shift.
///
/// # Errors
/// [`GpuError::DeviceMismatch`] / [`GpuError::ShapeMismatch`] on validation
/// failure; [`GpuError::PtxCompileFailed`] / [`GpuError::Driver`] on launch.
#[cfg(feature = "cuda")]
pub fn gpu_zeta_f32(
    x: &CudaBuffer<f32>,
    q: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    launch_binary_elementwise_f32(x, q, device, ZETA_F32_PTX, "zeta_f32")
}

/// `torch.special.xlogy(x, y)` forward on equal-length f32 buffers.
#[cfg(feature = "cuda")]
pub fn gpu_xlogy_f32(
    x: &CudaBuffer<f32>,
    y: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    launch_binary_elementwise_f32(x, y, device, XLOGY_F32_PTX, "xlogy_f32")
}

/// `torch.special.xlogy(x, y)` forward on equal-length f64 buffers.
#[cfg(feature = "cuda")]
pub fn gpu_xlogy_f64(
    x: &CudaBuffer<f64>,
    y: &CudaBuffer<f64>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    launch_binary_elementwise_f64(x, y, device, XLOGY_F64_PTX, "xlogy_f64")
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::transfer::{cpu_to_gpu, gpu_to_cpu};

    // CPU reference recurrences — copied verbatim from
    // `ferrotorch_core::special` so the GPU result is asserted against the
    // exact ferrotorch CPU contract (not a re-derivation).
    fn cheb_t(n: usize, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return x;
        }
        let (mut p, mut q, mut r) = (1.0, x, 0.0);
        for _ in 2..=n {
            r = 2.0 * x * q - p;
            p = q;
            q = r;
        }
        r
    }
    fn cheb_seeded(n: usize, x: f64, q1: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return q1;
        }
        let (mut p, mut q, mut r) = (1.0, q1, 0.0);
        for _ in 2..=n {
            r = 2.0 * x * q - p;
            p = q;
            q = r;
        }
        r
    }
    fn hermite_h(n: usize, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return 2.0 * x;
        }
        let (mut p, mut q, mut r) = (1.0, 2.0 * x, 0.0);
        for k in 1..n {
            r = 2.0 * x * q - 2.0 * (k as f64) * p;
            p = q;
            q = r;
        }
        r
    }
    fn hermite_he(n: usize, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return x;
        }
        let (mut p, mut q, mut r) = (1.0, x, 0.0);
        for k in 1..n {
            r = x * q - (k as f64) * p;
            p = q;
            q = r;
        }
        r
    }
    fn laguerre(n: usize, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return 1.0 - x;
        }
        let (mut p, mut q, mut r) = (1.0, 1.0 - x, 0.0);
        for k in 1..n {
            let kf = k as f64;
            r = ((2.0 * kf + 1.0 - x) * q - kf * p) / (kf + 1.0);
            p = q;
            q = r;
        }
        r
    }
    fn legendre(n: usize, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return x;
        }
        let (mut p, mut q, mut r) = (1.0, x, 0.0);
        for k in 1..n {
            let kf = k as f64;
            r = ((2.0 * kf + 1.0) * x * q - kf * p) / (kf + 1.0);
            p = q;
            q = r;
        }
        r
    }

    fn dev() -> Option<GpuDevice> {
        GpuDevice::new(0).ok()
    }

    const XS: [f32; 7] = [-1.5, -0.7, -0.25, 0.0, 0.3, 0.8, 1.4];

    #[test]
    fn chebyshev_t_on_device_matches_cpu() {
        let Some(device) = dev() else { return };
        let xg = cpu_to_gpu(&XS, &device).unwrap();
        for n in 0..=12usize {
            let yg = gpu_chebyshev_poly_f32(&xg, n, 1.0, 0.0, false, &device).unwrap();
            // result stays on device: it is a CudaBuffer (is_cuda by type).
            assert_eq!(yg.device_ordinal(), device.ordinal());
            let got = gpu_to_cpu(&yg, &device).unwrap();
            for (i, &x) in XS.iter().enumerate() {
                let want = cheb_t(n, x as f64) as f32;
                assert!(
                    (got[i] - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "cheb_t n={n} x={x}: got {} want {want}",
                    got[i]
                );
            }
        }
    }

    #[test]
    fn chebyshev_uvw_seeds_match_cpu() {
        let Some(device) = dev() else { return };
        let xg = cpu_to_gpu(&XS, &device).unwrap();
        // U: q1 = 2x (a=2,b=0); V: 2x-1 (a=2,b=-1); W: 2x+1 (a=2,b=1)
        let cases: [(f32, f32); 3] = [(2.0, 0.0), (2.0, -1.0), (2.0, 1.0)];
        for (a, b) in cases {
            for n in 0..=10usize {
                let yg = gpu_chebyshev_poly_f32(&xg, n, a, b, false, &device).unwrap();
                let got = gpu_to_cpu(&yg, &device).unwrap();
                for (i, &x) in XS.iter().enumerate() {
                    let q1 = a as f64 * x as f64 + b as f64;
                    let want = cheb_seeded(n, x as f64, q1) as f32;
                    assert!(
                        (got[i] - want).abs() <= 1e-4 * (1.0 + want.abs()),
                        "cheb seed=({a},{b}) n={n} x={x}: got {} want {want}",
                        got[i]
                    );
                }
            }
        }
    }

    #[test]
    fn shifted_chebyshev_t_matches_cpu() {
        let Some(device) = dev() else { return };
        // shifted domain [0,1]
        let xs: [f32; 5] = [0.0, 0.2, 0.5, 0.75, 1.0];
        let xg = cpu_to_gpu(&xs, &device).unwrap();
        for n in 0..=10usize {
            let yg = gpu_chebyshev_poly_f32(&xg, n, 1.0, 0.0, true, &device).unwrap();
            let got = gpu_to_cpu(&yg, &device).unwrap();
            for (i, &x) in xs.iter().enumerate() {
                let want = cheb_t(n, 2.0 * x as f64 - 1.0) as f32;
                assert!(
                    (got[i] - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "shifted cheb_t n={n} x={x}: got {} want {want}",
                    got[i]
                );
            }
        }
    }

    #[test]
    fn hermite_h_on_device_matches_cpu() {
        let Some(device) = dev() else { return };
        let xg = cpu_to_gpu(&XS, &device).unwrap();
        for n in 0..=8usize {
            let yg = gpu_hermite_h_poly_f32(&xg, n, &device).unwrap();
            let got = gpu_to_cpu(&yg, &device).unwrap();
            for (i, &x) in XS.iter().enumerate() {
                let want = hermite_h(n, x as f64) as f32;
                assert!(
                    (got[i] - want).abs() <= 1e-3 * (1.0 + want.abs()),
                    "hermite_h n={n} x={x}: got {} want {want}",
                    got[i]
                );
            }
        }
    }

    #[test]
    fn hermite_he_on_device_matches_cpu() {
        let Some(device) = dev() else { return };
        let xg = cpu_to_gpu(&XS, &device).unwrap();
        for n in 0..=8usize {
            let yg = gpu_hermite_he_poly_f32(&xg, n, &device).unwrap();
            let got = gpu_to_cpu(&yg, &device).unwrap();
            for (i, &x) in XS.iter().enumerate() {
                let want = hermite_he(n, x as f64) as f32;
                assert!(
                    (got[i] - want).abs() <= 1e-3 * (1.0 + want.abs()),
                    "hermite_he n={n} x={x}: got {} want {want}",
                    got[i]
                );
            }
        }
    }

    #[test]
    fn laguerre_on_device_matches_cpu() {
        let Some(device) = dev() else { return };
        let xg = cpu_to_gpu(&XS, &device).unwrap();
        for n in 0..=10usize {
            let yg = gpu_laguerre_poly_f32(&xg, n, &device).unwrap();
            let got = gpu_to_cpu(&yg, &device).unwrap();
            for (i, &x) in XS.iter().enumerate() {
                let want = laguerre(n, x as f64) as f32;
                assert!(
                    (got[i] - want).abs() <= 1e-3 * (1.0 + want.abs()),
                    "laguerre n={n} x={x}: got {} want {want}",
                    got[i]
                );
            }
        }
    }

    #[test]
    fn legendre_on_device_matches_cpu() {
        let Some(device) = dev() else { return };
        let xg = cpu_to_gpu(&XS, &device).unwrap();
        for n in 0..=10usize {
            let yg = gpu_legendre_poly_f32(&xg, n, &device).unwrap();
            let got = gpu_to_cpu(&yg, &device).unwrap();
            for (i, &x) in XS.iter().enumerate() {
                let want = legendre(n, x as f64) as f32;
                assert!(
                    (got[i] - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "legendre n={n} x={x}: got {} want {want}",
                    got[i]
                );
            }
        }
    }

    // --- entr / ndtr / ndtri (#1651 batch 1) ---------------------------------
    //
    // Expected values are live `torch.special.*` (torch 2.11.0+cu130, f32)
    // outputs (R-CHAR-3: oracle-derived). The GPU result stays on-device — it
    // is a `CudaBuffer<f32>` (is_cuda by type), `device_ordinal()` is asserted
    // before the explicit `gpu_to_cpu` read-back for value comparison.

    #[test]
    fn entr_on_device_matches_torch() {
        let Some(device) = dev() else { return };
        let xs: [f32; 7] = [-1.5, -0.7, -0.25, 0.0, 0.3, 0.8, 1.4];
        // torch.special.entr f32 oracle:
        let want: [f32; 7] = [
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.361_191_9,
            0.178_514_8,
            -0.471_061_1,
        ];
        let xg = cpu_to_gpu(&xs, &device).unwrap();
        let yg = gpu_entr_f32(&xg, &device).unwrap();
        // result stays on device.
        assert_eq!(yg.device_ordinal(), device.ordinal());
        let got = gpu_to_cpu(&yg, &device).unwrap();
        for i in 0..7 {
            if want[i].is_infinite() {
                assert!(
                    got[i].is_infinite() && got[i] < 0.0,
                    "entr idx {i}: got {} want -inf",
                    got[i]
                );
            } else {
                assert!(
                    (got[i] - want[i]).abs() <= 1e-5 * (1.0 + want[i].abs()),
                    "entr idx {i} x={}: got {} want {}",
                    xs[i],
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn ndtr_on_device_matches_torch() {
        let Some(device) = dev() else { return };
        let xs: [f32; 7] = [-1.5, -0.7, -0.25, 0.0, 0.3, 0.8, 1.4];
        let want: [f32; 7] = [
            0.066_807_21,
            0.241_963_7,
            0.401_293_7,
            0.5,
            0.617_911_4,
            0.788_144_6,
            0.919_243_3,
        ];
        let xg = cpu_to_gpu(&xs, &device).unwrap();
        let yg = gpu_ndtr_f32(&xg, &device).unwrap();
        assert_eq!(yg.device_ordinal(), device.ordinal());
        let got = gpu_to_cpu(&yg, &device).unwrap();
        for i in 0..7 {
            assert!(
                (got[i] - want[i]).abs() <= 1e-5,
                "ndtr idx {i} x={}: got {} want {}",
                xs[i],
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn ndtri_on_device_matches_torch() {
        let Some(device) = dev() else { return };
        // central + flip-region interior points.
        let ps: [f32; 7] = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975];
        let want: [f32; 7] = [
            -1.959_964,
            -1.281_552,
            -0.674_489_8,
            0.0,
            0.674_489_8,
            1.281_551,
            1.959_964,
        ];
        let xg = cpu_to_gpu(&ps, &device).unwrap();
        let yg = gpu_ndtri_f32(&xg, &device).unwrap();
        assert_eq!(yg.device_ordinal(), device.ordinal());
        let got = gpu_to_cpu(&yg, &device).unwrap();
        for i in 0..7 {
            assert!(
                (got[i] - want[i]).abs() <= 2e-4 * (1.0 + want[i].abs()),
                "ndtri idx {i} p={}: got {} want {}",
                ps[i],
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn ndtri_tail_and_edges_on_device_matches_torch() {
        let Some(device) = dev() else { return };
        // 0.001/1e-6 -> tail+far-tail regions; 0.999 -> flip region;
        // 0.0/1.0 -> -inf/+inf; -0.1/1.1 -> NaN.
        let ps: [f32; 7] = [0.001, 1e-6, 0.999, 0.0, 1.0, -0.1, 1.1];
        let want_tail: [f32; 3] = [-3.090_232, -4.753_424, 3.090_236];
        let xg = cpu_to_gpu(&ps, &device).unwrap();
        let yg = gpu_ndtri_f32(&xg, &device).unwrap();
        assert_eq!(yg.device_ordinal(), device.ordinal());
        let got = gpu_to_cpu(&yg, &device).unwrap();
        for i in 0..3 {
            assert!(
                (got[i] - want_tail[i]).abs() <= 5e-3 * (1.0 + want_tail[i].abs()),
                "ndtri tail idx {i} p={}: got {} want {}",
                ps[i],
                got[i],
                want_tail[i]
            );
        }
        assert!(got[3].is_infinite() && got[3] < 0.0, "ndtri(0) == -inf");
        assert!(got[4].is_infinite() && got[4] > 0.0, "ndtri(1) == +inf");
        assert!(got[5].is_nan(), "ndtri(-0.1) == NaN");
        assert!(got[6].is_nan(), "ndtri(1.1) == NaN");
    }

    // --- i0 / i0e / i1 / i1e (#1651 batch 2) ---------------------------------
    //
    // Expected values are live `torch.special.{i0,i0e,i1,i1e}` (torch
    // 2.11.0+cu130, f32) outputs (R-CHAR-3: oracle-derived). The GPU result
    // stays on-device (`CudaBuffer<f32>`, is_cuda by type); `device_ordinal()`
    // is asserted before the explicit `gpu_to_cpu` read-back for value compare.
    // The grid [-1.5,-0.7,0,0.3,2,5,9] spans the A-set (|x|<=8) and the B-set
    // (x=9), and i1/i1e exercise the odd-function sign flip on the negatives.

    #[test]
    fn i0_on_device_matches_torch() {
        let Some(device) = dev() else { return };
        let xs: [f32; 7] = [-1.5, -0.7, 0.0, 0.3, 2.0, 5.0, 9.0];
        let want: [f32; 7] = [
            1.646_723_3,
            1.126_303_1,
            1.0,
            1.022_626_9,
            2.279_585_1,
            27.239_874,
            1_093.588_4,
        ];
        let xg = cpu_to_gpu(&xs, &device).unwrap();
        let yg = gpu_i0_f32(&xg, &device).unwrap();
        assert_eq!(yg.device_ordinal(), device.ordinal());
        let got = gpu_to_cpu(&yg, &device).unwrap();
        for i in 0..7 {
            assert!(
                (got[i] - want[i]).abs() <= 2e-4 * (1.0 + want[i].abs()),
                "i0 idx {i} x={}: got {} want {}",
                xs[i],
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn i0e_on_device_matches_torch() {
        let Some(device) = dev() else { return };
        let xs: [f32; 7] = [-1.5, -0.7, 0.0, 0.3, 2.0, 5.0, 9.0];
        let want: [f32; 7] = [
            0.367_433_64,
            0.559_305_55,
            1.0,
            0.757_580_6,
            0.308_508_3,
            0.183_540_82,
            0.134_959_53,
        ];
        let xg = cpu_to_gpu(&xs, &device).unwrap();
        let yg = gpu_i0e_f32(&xg, &device).unwrap();
        assert_eq!(yg.device_ordinal(), device.ordinal());
        let got = gpu_to_cpu(&yg, &device).unwrap();
        for i in 0..7 {
            assert!(
                (got[i] - want[i]).abs() <= 2e-4 * (1.0 + want[i].abs()),
                "i0e idx {i} x={}: got {} want {}",
                xs[i],
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn i1_on_device_matches_torch() {
        let Some(device) = dev() else { return };
        let xs: [f32; 7] = [-1.5, -0.7, 0.0, 0.3, 2.0, 5.0, 9.0];
        let want: [f32; 7] = [
            -0.981_666_45,
            -0.371_879_67,
            0.0,
            0.151_693_87,
            1.590_636_8,
            24.335_642,
            1_030.914_8,
        ];
        let xg = cpu_to_gpu(&xs, &device).unwrap();
        let yg = gpu_i1_f32(&xg, &device).unwrap();
        assert_eq!(yg.device_ordinal(), device.ordinal());
        let got = gpu_to_cpu(&yg, &device).unwrap();
        for i in 0..7 {
            assert!(
                (got[i] - want[i]).abs() <= 2e-4 * (1.0 + want[i].abs()),
                "i1 idx {i} x={}: got {} want {}",
                xs[i],
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn i1e_on_device_matches_torch() {
        let Some(device) = dev() else { return };
        let xs: [f32; 7] = [-1.5, -0.7, 0.0, 0.3, 2.0, 5.0, 9.0];
        let want: [f32; 7] = [
            -0.219_039_41,
            -0.184_669_99,
            0.0,
            0.112_377_57,
            0.215_269_28,
            0.163_972_26,
            0.127_225,
        ];
        let xg = cpu_to_gpu(&xs, &device).unwrap();
        let yg = gpu_i1e_f32(&xg, &device).unwrap();
        assert_eq!(yg.device_ordinal(), device.ordinal());
        let got = gpu_to_cpu(&yg, &device).unwrap();
        for i in 0..7 {
            assert!(
                (got[i] - want[i]).abs() <= 2e-4 * (1.0 + want[i].abs()),
                "i1e idx {i} x={}: got {} want {}",
                xs[i],
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn spherical_bessel_j0_on_device_matches_torch() {
        // Live torch.special.spherical_bessel_j0 (2.11.0+cu130, f32). Grid spans
        // the |x|<0.5 Taylor branch (0,0.25), the boundary (0.5), and the
        // sin(x)/x branch (1,2,5), plus a negative (sin/x is even -> same as +3).
        // The kernel runs ON-DEVICE (CudaBuffer<f32>, is_cuda by type); the
        // device_ordinal is asserted before the explicit gpu_to_cpu read-back.
        let Some(device) = dev() else { return };
        let xs: [f32; 7] = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, -3.0];
        let want: [f32; 7] = [
            1.0,
            0.989_615_86,
            0.958_851_1,
            0.841_470_96,
            0.454_648_7,
            -0.191_784_86,
            0.047_04,
        ];
        let xg = cpu_to_gpu(&xs, &device).unwrap();
        let yg = gpu_spherical_bessel_j0_f32(&xg, &device).unwrap();
        assert_eq!(yg.device_ordinal(), device.ordinal());
        let got = gpu_to_cpu(&yg, &device).unwrap();
        for i in 0..7 {
            assert!(
                (got[i] - want[i]).abs() <= 2e-4 * (1.0 + want[i].abs()),
                "spherical_bessel_j0 idx {i} x={}: got {} want {}",
                xs[i],
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn spherical_bessel_j0_on_device_edges_match_torch() {
        // Live torch: spherical_bessel_j0([inf,-inf,nan]) = [0, 0, nan]; j0(0)=1.
        let Some(device) = dev() else { return };
        let xs: [f32; 4] = [0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
        let xg = cpu_to_gpu(&xs, &device).unwrap();
        let yg = gpu_spherical_bessel_j0_f32(&xg, &device).unwrap();
        assert_eq!(yg.device_ordinal(), device.ordinal());
        let got = gpu_to_cpu(&yg, &device).unwrap();
        assert!((got[0] - 1.0).abs() <= 1e-6, "j0(0) == 1: got {}", got[0]);
        assert!(got[1].abs() <= 1e-6, "j0(+inf) == 0: got {}", got[1]);
        assert!(got[2].abs() <= 1e-6, "j0(-inf) == 0: got {}", got[2]);
        assert!(got[3].is_nan(), "j0(NaN) == NaN: got {}", got[3]);
    }

    #[test]
    fn legendre_f64_matches_cpu_tight() {
        let Some(device) = dev() else { return };
        let xs: [f64; 6] = [-0.9, -0.4, -0.1, 0.2, 0.6, 0.95];
        let xg = cpu_to_gpu(&xs, &device).unwrap();
        for n in 0..=14usize {
            let yg = gpu_legendre_poly_f64(&xg, n, &device).unwrap();
            let got = gpu_to_cpu(&yg, &device).unwrap();
            for (i, &x) in xs.iter().enumerate() {
                let want = legendre(n, x);
                assert!(
                    (got[i] - want).abs() <= 1e-12 * (1.0 + want.abs()),
                    "legendre_f64 n={n} x={x}: got {} want {want}",
                    got[i]
                );
            }
        }
    }
}
