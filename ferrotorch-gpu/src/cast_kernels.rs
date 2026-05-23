//! Dtype-cast GPU compute kernels — crosslink #1185 Phase 2c.
//!
//! Hand-written PTX, loaded via [`crate::module_cache::get_or_compile`] like
//! [`crate::int_kernels`]. Elementwise 1-D, one thread per element, no host
//! round-trip — the result stays resident on the device.
//!
//! # Matrix
//!
//! - **float → int** (`f32`/`f64`/`bf16`/`f16` → `i32`/`i64`): TRUNCATE toward
//!   zero, matching PyTorch `.to(torch.int)` / `.to(torch.long)`. PTX
//!   `cvt.rzi.s{32,64}.f{32,64}` does exactly this (`rzi` = round-to-zero,
//!   integer). bf16/f16 are first widened to f32, then `cvt.rzi`.
//! - **int → float** (`i32`/`i64` → `f32`/`f64`/`bf16`/`f16`): PTX
//!   `cvt.rn.f{32,64}.s{32,64}` (round-to-nearest-even). bf16/f16 outputs go
//!   via f32 then `cvt.rn.bf16.f32` / `cvt.rn.f16.f32` (round-to-nearest-even,
//!   PyTorch parity for the narrowing step).
//! - **int → int** (`i32` ↔ `i64`): widen `cvt.s64.s32` (sign-extend) /
//!   narrow `cvt.s32.s64` (PTX truncates the high bits — this is C/PyTorch
//!   `.to(torch.int)` wrap-around on overflow, NOT a saturating or erroring
//!   cast; the CPU reference path in `IntTensor::cast` errors on out-of-range,
//!   so the GPU narrow path documents the wrapping divergence: an out-of-range
//!   i64→i32 on CUDA wraps, mirroring PyTorch's CUDA `.to(torch.int)`).
//!
//! bf16: a bf16 is the high 16 bits of an f32. Decode = splat into the high
//! half of a b32 register (`shl 16`) + reinterpret (`mov.b32`). Encode uses the
//! hardware `cvt.rn.bf16.f32` (sm_80+; the host RTX 3090 is sm_86).

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits};

use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::module_cache::get_or_compile;

const BLOCK_SIZE: u32 = 256;

fn launch_1d(n: usize) -> LaunchConfig {
    let grid = ((n as u32).saturating_add(BLOCK_SIZE - 1)) / BLOCK_SIZE;
    LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

// Each kernel has the signature (in_ptr: u64, out_ptr: u64, n: u32). The input
// element stride and output element stride are encoded in the PTX body
// (shl by 1/2/3 for 2/4/8-byte elements).

// ── float → int (truncate toward zero) ─────────────────────────────────────

const F32_TO_I32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry f32_to_i32_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %out, %off;
    .reg .f32 %v; .reg .s32 %r; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %off, %idx; shl.b64 %off, %off, 2; add.u64 %in, %in, %off;
    ld.global.f32 %v, [%in]; cvt.rzi.s32.f32 %r, %v;
    add.u64 %out, %out, %off; st.global.s32 [%out], %r;
DONE: ret;
}
";

const F32_TO_I64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry f32_to_i64_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %out, %ioff, %ooff;
    .reg .f32 %v; .reg .s64 %r; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 2; add.u64 %in, %in, %ioff;
    ld.global.f32 %v, [%in]; cvt.rzi.s64.f32 %r, %v;
    cvt.u64.u32 %ooff, %idx; shl.b64 %ooff, %ooff, 3; add.u64 %out, %out, %ooff;
    st.global.s64 [%out], %r;
DONE: ret;
}
";

const F64_TO_I32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry f64_to_i32_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %out, %ioff, %ooff;
    .reg .f64 %v; .reg .s32 %r; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 3; add.u64 %in, %in, %ioff;
    ld.global.f64 %v, [%in]; cvt.rzi.s32.f64 %r, %v;
    cvt.u64.u32 %ooff, %idx; shl.b64 %ooff, %ooff, 2; add.u64 %out, %out, %ooff;
    st.global.s32 [%out], %r;
DONE: ret;
}
";

const F64_TO_I64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry f64_to_i64_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %out, %off;
    .reg .f64 %v; .reg .s64 %r; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %off, %idx; shl.b64 %off, %off, 3; add.u64 %in, %in, %off;
    ld.global.f64 %v, [%in]; cvt.rzi.s64.f64 %r, %v;
    add.u64 %out, %out, %off; st.global.s64 [%out], %r;
DONE: ret;
}
";

// f16 → int: widen f16→f32 (cvt.f32.f16) then cvt.rzi.s*.f32.
const F16_TO_I32_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64
.visible .entry f16_to_i32_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %out, %ioff, %ooff;
    .reg .b16 %h; .reg .f32 %v; .reg .s32 %r; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 1; add.u64 %in, %in, %ioff;
    ld.global.b16 %h, [%in]; cvt.f32.f16 %v, %h; cvt.rzi.s32.f32 %r, %v;
    cvt.u64.u32 %ooff, %idx; shl.b64 %ooff, %ooff, 2; add.u64 %out, %out, %ooff;
    st.global.s32 [%out], %r;
DONE: ret;
}
";

const F16_TO_I64_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64
.visible .entry f16_to_i64_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %out, %ioff, %ooff;
    .reg .b16 %h; .reg .f32 %v; .reg .s64 %r; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 1; add.u64 %in, %in, %ioff;
    ld.global.b16 %h, [%in]; cvt.f32.f16 %v, %h; cvt.rzi.s64.f32 %r, %v;
    cvt.u64.u32 %ooff, %idx; shl.b64 %ooff, %ooff, 3; add.u64 %out, %out, %ooff;
    st.global.s64 [%out], %r;
DONE: ret;
}
";

// bf16 → int: decode (shl 16, mov.b32 → f32) then cvt.rzi.s*.f32.
const BF16_TO_I32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry bf16_to_i32_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr, %bits; .reg .u16 %h;
    .reg .u64 %in, %out, %ioff, %ooff;
    .reg .f32 %v; .reg .s32 %r; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 1; add.u64 %in, %in, %ioff;
    ld.global.u16 %h, [%in]; cvt.u32.u16 %bits, %h; shl.b32 %bits, %bits, 16; mov.b32 %v, %bits;
    cvt.rzi.s32.f32 %r, %v;
    cvt.u64.u32 %ooff, %idx; shl.b64 %ooff, %ooff, 2; add.u64 %out, %out, %ooff;
    st.global.s32 [%out], %r;
DONE: ret;
}
";

const BF16_TO_I64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry bf16_to_i64_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr, %bits; .reg .u16 %h;
    .reg .u64 %in, %out, %ioff, %ooff;
    .reg .f32 %v; .reg .s64 %r; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 1; add.u64 %in, %in, %ioff;
    ld.global.u16 %h, [%in]; cvt.u32.u16 %bits, %h; shl.b32 %bits, %bits, 16; mov.b32 %v, %bits;
    cvt.rzi.s64.f32 %r, %v;
    cvt.u64.u32 %ooff, %idx; shl.b64 %ooff, %ooff, 3; add.u64 %out, %out, %ooff;
    st.global.s64 [%out], %r;
DONE: ret;
}
";

// ── int → float (round-to-nearest-even) ────────────────────────────────────

const I32_TO_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry i32_to_f32_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %out, %off;
    .reg .s32 %iv; .reg .f32 %r; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %off, %idx; shl.b64 %off, %off, 2; add.u64 %in, %in, %off;
    ld.global.s32 %iv, [%in]; cvt.rn.f32.s32 %r, %iv;
    add.u64 %out, %out, %off; st.global.f32 [%out], %r;
DONE: ret;
}
";

const I32_TO_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry i32_to_f64_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %out, %ioff, %ooff;
    .reg .s32 %iv; .reg .f64 %r; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 2; add.u64 %in, %in, %ioff;
    ld.global.s32 %iv, [%in]; cvt.rn.f64.s32 %r, %iv;
    cvt.u64.u32 %ooff, %idx; shl.b64 %ooff, %ooff, 3; add.u64 %out, %out, %ooff;
    st.global.f64 [%out], %r;
DONE: ret;
}
";

const I64_TO_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry i64_to_f32_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %out, %ioff, %ooff;
    .reg .s64 %iv; .reg .f32 %r; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 3; add.u64 %in, %in, %ioff;
    ld.global.s64 %iv, [%in]; cvt.rn.f32.s64 %r, %iv;
    cvt.u64.u32 %ooff, %idx; shl.b64 %ooff, %ooff, 2; add.u64 %out, %out, %ooff;
    st.global.f32 [%out], %r;
DONE: ret;
}
";

const I64_TO_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry i64_to_f64_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %out, %off;
    .reg .s64 %iv; .reg .f64 %r; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %off, %idx; shl.b64 %off, %off, 3; add.u64 %in, %in, %off;
    ld.global.s64 %iv, [%in]; cvt.rn.f64.s64 %r, %iv;
    add.u64 %out, %out, %off; st.global.f64 [%out], %r;
DONE: ret;
}
";

// int → f16: int→f32 (cvt.rn.f32.s*) then narrow f32→f16 (cvt.rn.f16.f32).
const I32_TO_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64
.visible .entry i32_to_f16_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %out, %ioff, %ooff;
    .reg .s32 %iv; .reg .f32 %f; .reg .b16 %r; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 2; add.u64 %in, %in, %ioff;
    ld.global.s32 %iv, [%in]; cvt.rn.f32.s32 %f, %iv; cvt.rn.f16.f32 %r, %f;
    cvt.u64.u32 %ooff, %idx; shl.b64 %ooff, %ooff, 1; add.u64 %out, %out, %ooff;
    st.global.b16 [%out], %r;
DONE: ret;
}
";

const I64_TO_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64
.visible .entry i64_to_f16_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %out, %ioff, %ooff;
    .reg .s64 %iv; .reg .f32 %f; .reg .b16 %r; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 3; add.u64 %in, %in, %ioff;
    ld.global.s64 %iv, [%in]; cvt.rn.f32.s64 %f, %iv; cvt.rn.f16.f32 %r, %f;
    cvt.u64.u32 %ooff, %idx; shl.b64 %ooff, %ooff, 1; add.u64 %out, %out, %ooff;
    st.global.b16 [%out], %r;
DONE: ret;
}
";

// int → bf16: int→f32 then narrow f32→bf16 (cvt.rn.bf16.f32, sm_80+).
const I32_TO_BF16_PTX: &str = "\
.version 7.8
.target sm_80
.address_size 64
.visible .entry i32_to_bf16_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %out, %ioff, %ooff;
    .reg .s32 %iv; .reg .f32 %f; .reg .b16 %r; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 2; add.u64 %in, %in, %ioff;
    ld.global.s32 %iv, [%in]; cvt.rn.f32.s32 %f, %iv; cvt.rn.bf16.f32 %r, %f;
    cvt.u64.u32 %ooff, %idx; shl.b64 %ooff, %ooff, 1; add.u64 %out, %out, %ooff;
    st.global.b16 [%out], %r;
DONE: ret;
}
";

const I64_TO_BF16_PTX: &str = "\
.version 7.8
.target sm_80
.address_size 64
.visible .entry i64_to_bf16_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %out, %ioff, %ooff;
    .reg .s64 %iv; .reg .f32 %f; .reg .b16 %r; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 3; add.u64 %in, %in, %ioff;
    ld.global.s64 %iv, [%in]; cvt.rn.f32.s64 %f, %iv; cvt.rn.bf16.f32 %r, %f;
    cvt.u64.u32 %ooff, %idx; shl.b64 %ooff, %ooff, 1; add.u64 %out, %out, %ooff;
    st.global.b16 [%out], %r;
DONE: ret;
}
";

// ── int → int (widen / narrow) ─────────────────────────────────────────────

const I32_TO_I64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry i32_to_i64_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %out, %ioff, %ooff;
    .reg .s32 %iv; .reg .s64 %r; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 2; add.u64 %in, %in, %ioff;
    ld.global.s32 %iv, [%in]; cvt.s64.s32 %r, %iv;
    cvt.u64.u32 %ooff, %idx; shl.b64 %ooff, %ooff, 3; add.u64 %out, %out, %ooff;
    st.global.s64 [%out], %r;
DONE: ret;
}
";

const I64_TO_I32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry i64_to_i32_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %out, %ioff, %ooff;
    .reg .s64 %iv; .reg .s32 %r; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 3; add.u64 %in, %in, %ioff;
    ld.global.s64 %iv, [%in]; cvt.s32.s64 %r, %iv;
    cvt.u64.u32 %ooff, %idx; shl.b64 %ooff, %ooff, 2; add.u64 %out, %out, %ooff;
    st.global.s32 [%out], %r;
DONE: ret;
}
";

// Same-width identity copies (i32->i32, i64->i64). A `.cast::<I>()` to the
// SAME integer dtype must preserve the full value bit-for-bit (a narrow-then-
// widen round trip would corrupt i64 values outside the i32 range), so these
// are plain element copies, kept GPU-resident (no host round trip).
const I32_COPY_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry i32_copy_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr; .reg .u64 %in, %out, %off;
    .reg .b32 %v; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %off, %idx; shl.b64 %off, %off, 2;
    add.u64 %in, %in, %off; ld.global.b32 %v, [%in];
    add.u64 %out, %out, %off; st.global.b32 [%out], %v;
DONE: ret;
}
";

const I64_COPY_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry i64_copy_kernel(.param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr; .reg .u64 %in, %out, %off;
    .reg .b64 %v; .reg .pred %p;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %off, %idx; shl.b64 %off, %off, 3;
    add.u64 %in, %in, %off; ld.global.b64 %v, [%in];
    add.u64 %out, %out, %off; st.global.b64 [%out], %v;
DONE: ret;
}
";

/// Launch an elementwise cast kernel from `IN` to `OUT` native element types,
/// returning a fresh resident `CudaSlice<OUT>` of `n` elements.
fn launch_cast<IN: DeviceRepr + ValidAsZeroBits, OUT: DeviceRepr + ValidAsZeroBits>(
    input: &CudaSlice<IN>,
    n: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaSlice<OUT>> {
    // `n` is the LOGICAL element count supplied by the caller — NOT
    // `input.len()`, which may be a pool-rounded over-allocation when the input
    // comes from a pooled float op. The kernel reads/writes strictly `[0, n)`.
    debug_assert!(input.len() >= n, "cast input slice shorter than logical n");
    let stream = device.stream();
    if n == 0 {
        return Ok(stream.alloc_zeros::<OUT>(0)?);
    }
    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;
    let mut out = stream.alloc_zeros::<OUT>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is the PTX entry `kernel_name`; signature (in_ptr, out_ptr, n)
    //   matches the three args pushed below.
    // - `input` holds `n` `IN`-elements; `out` is a fresh `n`-element `OUT`
    //   buffer, the only `&mut`, non-aliased with `input`.
    // - Each thread reads `input[i]` / writes `out[i]` only for `i in [0,n)`
    //   (bound check `setp.ge.u32 %p, %idx, %nr`).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

// ── Public entry points ─────────────────────────────────────────────────────
// float → int (i32 result)
/// Cast f32 → i32 (truncate toward zero).
pub fn cast_f32_to_i32(x: &CudaSlice<f32>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<i32>> {
    launch_cast(x, n, d, F32_TO_I32_PTX, "f32_to_i32_kernel")
}
/// Cast f64 → i32 (truncate toward zero).
pub fn cast_f64_to_i32(x: &CudaSlice<f64>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<i32>> {
    launch_cast(x, n, d, F64_TO_I32_PTX, "f64_to_i32_kernel")
}
/// Cast f16 (u16 bits) → i32 (truncate toward zero).
pub fn cast_f16_to_i32(x: &CudaSlice<u16>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<i32>> {
    launch_cast(x, n, d, F16_TO_I32_PTX, "f16_to_i32_kernel")
}
/// Cast bf16 (u16 bits) → i32 (truncate toward zero).
pub fn cast_bf16_to_i32(x: &CudaSlice<u16>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<i32>> {
    launch_cast(x, n, d, BF16_TO_I32_PTX, "bf16_to_i32_kernel")
}
// float → int (i64 result)
/// Cast f32 → i64 (truncate toward zero).
pub fn cast_f32_to_i64(x: &CudaSlice<f32>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<i64>> {
    launch_cast(x, n, d, F32_TO_I64_PTX, "f32_to_i64_kernel")
}
/// Cast f64 → i64 (truncate toward zero).
pub fn cast_f64_to_i64(x: &CudaSlice<f64>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<i64>> {
    launch_cast(x, n, d, F64_TO_I64_PTX, "f64_to_i64_kernel")
}
/// Cast f16 (u16 bits) → i64 (truncate toward zero).
pub fn cast_f16_to_i64(x: &CudaSlice<u16>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<i64>> {
    launch_cast(x, n, d, F16_TO_I64_PTX, "f16_to_i64_kernel")
}
/// Cast bf16 (u16 bits) → i64 (truncate toward zero).
pub fn cast_bf16_to_i64(x: &CudaSlice<u16>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<i64>> {
    launch_cast(x, n, d, BF16_TO_I64_PTX, "bf16_to_i64_kernel")
}
// int (i32) → float
/// Cast i32 → f32 (round-to-nearest-even).
pub fn cast_i32_to_f32(x: &CudaSlice<i32>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<f32>> {
    launch_cast(x, n, d, I32_TO_F32_PTX, "i32_to_f32_kernel")
}
/// Cast i32 → f64.
pub fn cast_i32_to_f64(x: &CudaSlice<i32>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<f64>> {
    launch_cast(x, n, d, I32_TO_F64_PTX, "i32_to_f64_kernel")
}
/// Cast i32 → f16 (u16 bits, round-to-nearest-even).
pub fn cast_i32_to_f16(x: &CudaSlice<i32>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<u16>> {
    launch_cast(x, n, d, I32_TO_F16_PTX, "i32_to_f16_kernel")
}
/// Cast i32 → bf16 (u16 bits, round-to-nearest-even).
pub fn cast_i32_to_bf16(x: &CudaSlice<i32>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<u16>> {
    launch_cast(x, n, d, I32_TO_BF16_PTX, "i32_to_bf16_kernel")
}
// int (i64) → float
/// Cast i64 → f32 (round-to-nearest-even).
pub fn cast_i64_to_f32(x: &CudaSlice<i64>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<f32>> {
    launch_cast(x, n, d, I64_TO_F32_PTX, "i64_to_f32_kernel")
}
/// Cast i64 → f64.
pub fn cast_i64_to_f64(x: &CudaSlice<i64>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<f64>> {
    launch_cast(x, n, d, I64_TO_F64_PTX, "i64_to_f64_kernel")
}
/// Cast i64 → f16 (u16 bits, round-to-nearest-even).
pub fn cast_i64_to_f16(x: &CudaSlice<i64>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<u16>> {
    launch_cast(x, n, d, I64_TO_F16_PTX, "i64_to_f16_kernel")
}
/// Cast i64 → bf16 (u16 bits, round-to-nearest-even).
pub fn cast_i64_to_bf16(x: &CudaSlice<i64>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<u16>> {
    launch_cast(x, n, d, I64_TO_BF16_PTX, "i64_to_bf16_kernel")
}
// int ↔ int
/// Cast i32 → i64 (sign-extend).
pub fn cast_i32_to_i64(x: &CudaSlice<i32>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<i64>> {
    launch_cast(x, n, d, I32_TO_I64_PTX, "i32_to_i64_kernel")
}
/// Cast i64 → i32 (truncate high bits — wrapping, PyTorch CUDA `.to(int)`).
pub fn cast_i64_to_i32(x: &CudaSlice<i64>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<i32>> {
    launch_cast(x, n, d, I64_TO_I32_PTX, "i64_to_i32_kernel")
}
/// Same-dtype i32 identity copy (full-value preserving, GPU-resident).
pub fn cast_i32_copy(x: &CudaSlice<i32>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<i32>> {
    launch_cast(x, n, d, I32_COPY_PTX, "i32_copy_kernel")
}
/// Same-dtype i64 identity copy (full-value preserving, GPU-resident).
pub fn cast_i64_copy(x: &CudaSlice<i64>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<i64>> {
    launch_cast(x, n, d, I64_COPY_PTX, "i64_copy_kernel")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> GpuDevice {
        GpuDevice::new(0).expect("cuda device")
    }

    #[test]
    fn f32_to_i32_truncates_toward_zero() {
        let d = dev();
        let h = d.stream().clone_htod(&vec![1.9f32, -1.9, 2.0, -2.5, 0.0]).unwrap();
        let r = cast_f32_to_i32(&h, h.len(), &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&r).unwrap(), vec![1i32, -1, 2, -2, 0]);
    }

    #[test]
    fn f32_to_i64_truncates() {
        let d = dev();
        let h = d.stream().clone_htod(&vec![3.7f32, -3.7]).unwrap();
        let r = cast_f32_to_i64(&h, h.len(), &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&r).unwrap(), vec![3i64, -3]);
    }

    #[test]
    fn i32_to_f32_and_i32_to_i64() {
        let d = dev();
        let h = d.stream().clone_htod(&vec![-5i32, 7, 0]).unwrap();
        let f = cast_i32_to_f32(&h, h.len(), &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&f).unwrap(), vec![-5.0f32, 7.0, 0.0]);
        let w = cast_i32_to_i64(&h, h.len(), &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&w).unwrap(), vec![-5i64, 7, 0]);
    }

    #[test]
    fn i64_to_i32_narrow_and_f64() {
        let d = dev();
        let h = d.stream().clone_htod(&vec![5i64, -9, 123456]).unwrap();
        let narrowed = cast_i64_to_i32(&h, h.len(), &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&narrowed).unwrap(), vec![5i32, -9, 123456]);
        let f = cast_i64_to_f64(&h, h.len(), &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&f).unwrap(), vec![5.0f64, -9.0, 123456.0]);
    }

    #[test]
    fn f64_to_i64_and_i32_to_bf16_f16() {
        let d = dev();
        let h = d.stream().clone_htod(&vec![9.99f64, -9.99]).unwrap();
        let r = cast_f64_to_i64(&h, h.len(), &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&r).unwrap(), vec![9i64, -9]);
        let hi = d.stream().clone_htod(&vec![3i32, -4, 256]).unwrap();
        let bf = cast_i32_to_bf16(&hi, hi.len(), &d).unwrap();
        let bf_h: Vec<f32> = d.stream().clone_dtoh(&bf).unwrap().into_iter()
            .map(|b| half::bf16::from_bits(b).to_f32()).collect();
        assert_eq!(bf_h, vec![3.0f32, -4.0, 256.0]);
        let f16 = cast_i32_to_f16(&hi, hi.len(), &d).unwrap();
        let f16_h: Vec<f32> = d.stream().clone_dtoh(&f16).unwrap().into_iter()
            .map(|b| half::f16::from_bits(b).to_f32()).collect();
        assert_eq!(f16_h, vec![3.0f32, -4.0, 256.0]);
    }

    #[test]
    fn bf16_f16_to_int_truncate() {
        let d = dev();
        let bf: Vec<u16> = [1.9f32, -2.9].iter().map(|&v| half::bf16::from_f32(v).to_bits()).collect();
        let hb = d.stream().clone_htod(&bf).unwrap();
        let r = cast_bf16_to_i32(&hb, hb.len(), &d).unwrap();
        // bf16(1.9) rounds to ~1.898..; trunc -> 1. bf16(-2.9) -> ~-2.90; trunc -> -2
        assert_eq!(d.stream().clone_dtoh(&r).unwrap(), vec![1i32, -2]);
        let f: Vec<u16> = [4.5f32, -5.5].iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        let hf = d.stream().clone_htod(&f).unwrap();
        let r2 = cast_f16_to_i64(&hf, hf.len(), &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&r2).unwrap(), vec![4i64, -5]);
    }

    #[test]
    fn same_dtype_copy_preserves_full_value() {
        let d = dev();
        // i64 value far outside the i32 range must survive an i64->i64 "cast".
        let big = 9_000_000_000i64;
        let h = d.stream().clone_htod(&vec![big, -big, 7]).unwrap();
        let c = cast_i64_copy(&h, h.len(), &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&c).unwrap(), vec![big, -big, 7]);
        let hi = d.stream().clone_htod(&vec![i32::MIN, i32::MAX, 0]).unwrap();
        let ci = cast_i32_copy(&hi, hi.len(), &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&ci).unwrap(), vec![i32::MIN, i32::MAX, 0]);
    }
}
