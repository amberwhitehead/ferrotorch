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
