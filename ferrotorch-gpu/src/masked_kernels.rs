//! Mask-driven GPU compute kernels — crosslink #1185 Phase 3c.
//!
//! Hand-written PTX owned by Rust (no CUDA C++, no nvrtc, no external toolchain
//! at load time), loaded via [`crate::module_cache::get_or_compile`] exactly
//! like [`crate::bool_kernels`] / [`crate::cast_kernels`] / [`crate::f16`] /
//! [`crate::bf16`]. The mask is a GPU-resident `CudaSlice<u8>` (one byte per
//! element, 0/1; the `DType::Bool` tag distinguishes it from an integer u8) —
//! the same storage the Phase-3b comparison kernels emit, so a comparison →
//! mask → masked-op chain never leaves the device.
//!
//! # Operations
//!
//! - **masked_fill** (`out[i] = mask[i]!=0 ? value : input[i]`): value dtype in
//!   (`f32`/`f64`/`bf16`/`f16`/`i32`/`i64`) + u8 mask, same dtype out. The fill
//!   value is a scalar passed as f32; for f64 it is passed as f64; for bf16/f16
//!   the f32 scalar is converted to the half bit pattern in-kernel.
//! - **where** (`out[i] = cond[i]!=0 ? x[i] : y[i]`): u8 cond + two same-dtype
//!   value buffers, same dtype out. For bf16/f16 the select is a pure 16-bit
//!   bit-pattern copy (no decode needed — we never inspect the value, only
//!   choose one of two), so a single `where_16` kernel serves both.
//! - **masked_select** (stream compaction → data-dependent 1-D output):
//!   `count_true` (serial OR-style sum reduction of the u8 mask → one i32) sizes
//!   the output; then a serial compaction kernel writes `input[i] -> out[j++]`
//!   for each `i` where `mask[i]!=0`. The single COUNT integer is the only host
//!   crossing — it is the result *shape*, not a data buffer round-trip, exactly
//!   what PyTorch does internally (a CUDA sync to size the data-dependent
//!   output). A parallel prefix-sum (scan) compaction is a perf follow-up; the
//!   serial walk is correct and matches the existing serial reductions in
//!   `bool_kernels` / `int_kernels`.
//!
//! # PyTorch parity (rust-gpu-discipline §3)
//!
//! Every op runs a real PTX kernel on CUDA; the result stays GPU-resident. An
//! unsupported (op, dtype) returns a structured error upstream
//! (`FerrotorchError::NotImplementedOnCuda` / `InvalidArgument`) — never a
//! silent CPU detour.

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

// ===========================================================================
// masked_fill PTX
//
// Signature: (input_ptr, mask_ptr, out_ptr, value, n)
//   input: VAL_BYTES per element ; mask: 1 byte per element (u8 0/1)
//   out:   VAL_BYTES per element
//   out[i] = (mask[i] != 0) ? value : input[i]
// The value param is `.f32` for f32/bf16/f16 (converted in-kernel for halves)
// and `.f64` for f64. For ints the value is passed as the native int.
// ===========================================================================

const MASKED_FILL_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry masked_fill_f32_kernel(
    .param .u64 in_ptr, .param .u64 mask_ptr, .param .u64 out_ptr,
    .param .f32 value, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %mk, %out, %ioff, %moff;
    .reg .f32 %v, %iv; .reg .u16 %m; .reg .pred %p, %sel;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %mk, [mask_ptr];
    ld.param.u64 %out, [out_ptr]; ld.param.f32 %v, [value]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %moff, %idx; add.u64 %mk, %mk, %moff;
    ld.global.u8 %m, [%mk]; setp.ne.u16 %sel, %m, 0;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 2;
    add.u64 %in, %in, %ioff; ld.global.f32 %iv, [%in];
    selp.f32 %iv, %v, %iv, %sel;
    add.u64 %out, %out, %ioff; st.global.f32 [%out], %iv;
DONE: ret;
}
";

const MASKED_FILL_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry masked_fill_f64_kernel(
    .param .u64 in_ptr, .param .u64 mask_ptr, .param .u64 out_ptr,
    .param .f64 value, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %mk, %out, %ioff, %moff;
    .reg .f64 %v, %iv; .reg .u16 %m; .reg .pred %p, %sel;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %mk, [mask_ptr];
    ld.param.u64 %out, [out_ptr]; ld.param.f64 %v, [value]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %moff, %idx; add.u64 %mk, %mk, %moff;
    ld.global.u8 %m, [%mk]; setp.ne.u16 %sel, %m, 0;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 3;
    add.u64 %in, %in, %ioff; ld.global.f64 %iv, [%in];
    selp.f64 %iv, %v, %iv, %sel;
    add.u64 %out, %out, %ioff; st.global.f64 [%out], %iv;
DONE: ret;
}
";

// bf16/f16 masked_fill: value comes in as f32, narrowed to the half bit pattern
// in-kernel (cvt.rn.bf16.f32 / cvt.rn.f16.f32). The selected value (the fill or
// the existing 16-bit element) is stored as the raw b16 bit pattern.
const MASKED_FILL_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64
.visible .entry masked_fill_f16_kernel(
    .param .u64 in_ptr, .param .u64 mask_ptr, .param .u64 out_ptr,
    .param .f32 value, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %mk, %out, %ioff, %moff;
    .reg .f32 %v; .reg .b16 %vh, %iv; .reg .u16 %m; .reg .pred %p, %sel;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %mk, [mask_ptr];
    ld.param.u64 %out, [out_ptr]; ld.param.f32 %v, [value]; ld.param.u32 %nr, [n];
    cvt.rn.f16.f32 %vh, %v;
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %moff, %idx; add.u64 %mk, %mk, %moff;
    ld.global.u8 %m, [%mk]; setp.ne.u16 %sel, %m, 0;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 1;
    add.u64 %in, %in, %ioff; ld.global.b16 %iv, [%in];
    selp.b16 %iv, %vh, %iv, %sel;
    add.u64 %out, %out, %ioff; st.global.b16 [%out], %iv;
DONE: ret;
}
";

const MASKED_FILL_BF16_PTX: &str = "\
.version 7.8
.target sm_80
.address_size 64
.visible .entry masked_fill_bf16_kernel(
    .param .u64 in_ptr, .param .u64 mask_ptr, .param .u64 out_ptr,
    .param .f32 value, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %mk, %out, %ioff, %moff;
    .reg .f32 %v; .reg .b16 %vh, %iv; .reg .u16 %m; .reg .pred %p, %sel;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %mk, [mask_ptr];
    ld.param.u64 %out, [out_ptr]; ld.param.f32 %v, [value]; ld.param.u32 %nr, [n];
    cvt.rn.bf16.f32 %vh, %v;
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %moff, %idx; add.u64 %mk, %mk, %moff;
    ld.global.u8 %m, [%mk]; setp.ne.u16 %sel, %m, 0;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 1;
    add.u64 %in, %in, %ioff; ld.global.b16 %iv, [%in];
    selp.b16 %iv, %vh, %iv, %sel;
    add.u64 %out, %out, %ioff; st.global.b16 [%out], %iv;
DONE: ret;
}
";

const MASKED_FILL_I32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry masked_fill_i32_kernel(
    .param .u64 in_ptr, .param .u64 mask_ptr, .param .u64 out_ptr,
    .param .u32 value, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %mk, %out, %ioff, %moff;
    .reg .b32 %v, %iv; .reg .u16 %m; .reg .pred %p, %sel;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %mk, [mask_ptr];
    ld.param.u64 %out, [out_ptr]; ld.param.b32 %v, [value]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %moff, %idx; add.u64 %mk, %mk, %moff;
    ld.global.u8 %m, [%mk]; setp.ne.u16 %sel, %m, 0;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 2;
    add.u64 %in, %in, %ioff; ld.global.b32 %iv, [%in];
    selp.b32 %iv, %v, %iv, %sel;
    add.u64 %out, %out, %ioff; st.global.b32 [%out], %iv;
DONE: ret;
}
";

const MASKED_FILL_I64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry masked_fill_i64_kernel(
    .param .u64 in_ptr, .param .u64 mask_ptr, .param .u64 out_ptr,
    .param .u64 value, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %mk, %out, %ioff, %moff;
    .reg .b64 %v, %iv; .reg .u16 %m; .reg .pred %p, %sel;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %mk, [mask_ptr];
    ld.param.u64 %out, [out_ptr]; ld.param.b64 %v, [value]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %moff, %idx; add.u64 %mk, %mk, %moff;
    ld.global.u8 %m, [%mk]; setp.ne.u16 %sel, %m, 0;
    cvt.u64.u32 %ioff, %idx; shl.b64 %ioff, %ioff, 3;
    add.u64 %in, %in, %ioff; ld.global.b64 %iv, [%in];
    selp.b64 %iv, %v, %iv, %sel;
    add.u64 %out, %out, %ioff; st.global.b64 [%out], %iv;
DONE: ret;
}
";

// ===========================================================================
// where PTX
//
// Signature: (cond_ptr, x_ptr, y_ptr, out_ptr, n)
//   cond: 1 byte per element (u8 0/1) ; x,y,out: VAL_BYTES per element
//   out[i] = (cond[i] != 0) ? x[i] : y[i]
// The select never inspects the value, only picks one of two bit patterns, so
// a single kernel per element-width (32/64/16 bit) covers every value dtype.
// ===========================================================================

const WHERE_32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry where_32_kernel(
    .param .u64 cond_ptr, .param .u64 x_ptr, .param .u64 y_ptr,
    .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %cd, %x, %y, %out, %voff, %coff;
    .reg .b32 %xv, %yv, %r; .reg .u16 %c; .reg .pred %p, %sel;
    ld.param.u64 %cd, [cond_ptr]; ld.param.u64 %x, [x_ptr]; ld.param.u64 %y, [y_ptr];
    ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %coff, %idx; add.u64 %cd, %cd, %coff;
    ld.global.u8 %c, [%cd]; setp.ne.u16 %sel, %c, 0;
    cvt.u64.u32 %voff, %idx; shl.b64 %voff, %voff, 2;
    add.u64 %x, %x, %voff; ld.global.b32 %xv, [%x];
    add.u64 %y, %y, %voff; ld.global.b32 %yv, [%y];
    selp.b32 %r, %xv, %yv, %sel;
    add.u64 %out, %out, %voff; st.global.b32 [%out], %r;
DONE: ret;
}
";

const WHERE_64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry where_64_kernel(
    .param .u64 cond_ptr, .param .u64 x_ptr, .param .u64 y_ptr,
    .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %cd, %x, %y, %out, %voff, %coff;
    .reg .b64 %xv, %yv, %r; .reg .u16 %c; .reg .pred %p, %sel;
    ld.param.u64 %cd, [cond_ptr]; ld.param.u64 %x, [x_ptr]; ld.param.u64 %y, [y_ptr];
    ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %coff, %idx; add.u64 %cd, %cd, %coff;
    ld.global.u8 %c, [%cd]; setp.ne.u16 %sel, %c, 0;
    cvt.u64.u32 %voff, %idx; shl.b64 %voff, %voff, 3;
    add.u64 %x, %x, %voff; ld.global.b64 %xv, [%x];
    add.u64 %y, %y, %voff; ld.global.b64 %yv, [%y];
    selp.b64 %r, %xv, %yv, %sel;
    add.u64 %out, %out, %voff; st.global.b64 [%out], %r;
DONE: ret;
}
";

const WHERE_16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry where_16_kernel(
    .param .u64 cond_ptr, .param .u64 x_ptr, .param .u64 y_ptr,
    .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %cd, %x, %y, %out, %voff, %coff;
    .reg .b16 %xv, %yv, %r; .reg .u16 %c; .reg .pred %p, %sel;
    ld.param.u64 %cd, [cond_ptr]; ld.param.u64 %x, [x_ptr]; ld.param.u64 %y, [y_ptr];
    ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr; @%p bra DONE;
    cvt.u64.u32 %coff, %idx; add.u64 %cd, %cd, %coff;
    ld.global.u8 %c, [%cd]; setp.ne.u16 %sel, %c, 0;
    cvt.u64.u32 %voff, %idx; shl.b64 %voff, %voff, 1;
    add.u64 %x, %x, %voff; ld.global.b16 %xv, [%x];
    add.u64 %y, %y, %voff; ld.global.b16 %yv, [%y];
    selp.b16 %r, %xv, %yv, %sel;
    add.u64 %out, %out, %voff; st.global.b16 [%out], %r;
DONE: ret;
}
";

// ===========================================================================
// masked_select: count + serial compaction
// ===========================================================================

// Count of true (nonzero) mask bytes. One launched thread folds all n bytes
// serially into a single i32 (matching the serial-reduction harness in
// bool_kernels). Output is one s32. n == 0 is guarded on the host.
const COUNT_TRUE_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry count_true_kernel(.param .u64 mask_ptr, .param .u64 out_ptr, .param .u32 n) {
    .reg .u32 %idx, %bid, %bdim, %nr, %i;
    .reg .u64 %mk, %out, %off, %cur;
    .reg .u16 %v; .reg .s32 %acc, %one;
    .reg .pred %only0, %p, %nz;
    ld.param.u64 %mk, [mask_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ne.u32 %only0, %idx, 0; @%only0 bra DONE;
    mov.s32 %acc, 0; mov.u32 %i, 0;
LOOP:
    setp.ge.u32 %p, %i, %nr; @%p bra STORE;
    cvt.u64.u32 %off, %i; add.u64 %cur, %mk, %off;
    ld.global.u8 %v, [%cur];
    setp.ne.u16 %nz, %v, 0; selp.s32 %one, 1, 0, %nz;
    add.s32 %acc, %acc, %one;
    add.u32 %i, %i, 1; bra LOOP;
STORE:
    st.global.s32 [%out], %acc;
DONE: ret;
}
";

// Serial compaction: one launched thread walks the n elements in order; for
// each i where mask[i] != 0 it copies input[i] -> out[j] and increments j.
// VAL_SHIFT is log2(VAL_BYTES) (1/2/3 for 2/4/8-byte elements). Templated over
// the load/store type so one builder serves f32/f64/i32/i64/half. The output
// buffer is sized to the count (computed by COUNT_TRUE_PTX) on the host.
fn compact_ptx(kernel_name: &str, val_shift: u32, ld_st_ty: &str, reg_decl: &str) -> String {
    format!(
        "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry {kernel_name}(.param .u64 in_ptr, .param .u64 mask_ptr, .param .u64 out_ptr, .param .u32 n) {{
    .reg .u32 %idx, %bid, %bdim, %nr, %i, %j;
    .reg .u64 %in, %mk, %out, %ioff, %ooff, %icur, %mcur, %ocur;
    .reg .u16 %m; {reg_decl}
    .reg .pred %only0, %p, %nz;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %mk, [mask_ptr];
    ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ne.u32 %only0, %idx, 0; @%only0 bra DONE;
    mov.u32 %i, 0; mov.u32 %j, 0;
LOOP:
    setp.ge.u32 %p, %i, %nr; @%p bra DONE;
    cvt.u64.u32 %ioff, %i; add.u64 %mcur, %mk, %ioff;
    ld.global.u8 %m, [%mcur]; setp.ne.u16 %nz, %m, 0;
    @!%nz bra NEXT;
    // mask[i] true: out[j] = input[i]
    shl.b64 %ioff, %ioff, {val_shift}; add.u64 %icur, %in, %ioff;
    ld.global.{ld_st_ty} %val, [%icur];
    cvt.u64.u32 %ooff, %j; shl.b64 %ooff, %ooff, {val_shift}; add.u64 %ocur, %out, %ooff;
    st.global.{ld_st_ty} [%ocur], %val;
    add.u32 %j, %j, 1;
NEXT:
    add.u32 %i, %i, 1; bra LOOP;
DONE: ret;
}}
"
    )
}

// ===========================================================================
// masked_scatter: serial scatter (the inverse of the compaction above)
//
// Signature: (grad_ptr, mask_ptr, out_ptr, n)
//   grad: VAL_BYTES per element, length = #true (the compacted gradient)
//   mask: 1 byte per element (u8 0/1), length n
//   out:  VAL_BYTES per element, length n, PRE-ZEROED by the host (alloc_zeros)
//   for each i in [0,n): if mask[i]!=0 { out[i] = grad[j++] }  (else left 0)
// This is the VJP of masked_select: the compaction wrote input[i] -> out[j++]
// for each true i, so the backward scatters grad[j++] -> out[i] at those same
// positions and zeros everywhere else. One launched thread walks serially in
// order (matching COMPACT/COUNT_TRUE above); a parallel prefix-sum scatter is a
// perf follow-up. VAL_SHIFT is log2(VAL_BYTES). Out is left untouched (already
// 0) where mask[i]==0, so no else-store is needed.
// ===========================================================================
fn scatter_ptx(kernel_name: &str, val_shift: u32, ld_st_ty: &str, reg_decl: &str) -> String {
    format!(
        "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry {kernel_name}(.param .u64 grad_ptr, .param .u64 mask_ptr, .param .u64 out_ptr, .param .u32 n) {{
    .reg .u32 %idx, %bid, %bdim, %nr, %i, %j;
    .reg .u64 %gr, %mk, %out, %goff, %ooff, %gcur, %mcur, %ocur;
    .reg .u16 %m; {reg_decl}
    .reg .pred %only0, %p, %nz;
    ld.param.u64 %gr, [grad_ptr]; ld.param.u64 %mk, [mask_ptr];
    ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ne.u32 %only0, %idx, 0; @%only0 bra DONE;
    mov.u32 %i, 0; mov.u32 %j, 0;
LOOP:
    setp.ge.u32 %p, %i, %nr; @%p bra DONE;
    cvt.u64.u32 %ooff, %i; add.u64 %mcur, %mk, %ooff;
    ld.global.u8 %m, [%mcur]; setp.ne.u16 %nz, %m, 0;
    @!%nz bra NEXT;
    // mask[i] true: out[i] = grad[j]
    cvt.u64.u32 %goff, %j; shl.b64 %goff, %goff, {val_shift}; add.u64 %gcur, %gr, %goff;
    ld.global.{ld_st_ty} %val, [%gcur];
    shl.b64 %ooff, %ooff, {val_shift}; add.u64 %ocur, %out, %ooff;
    st.global.{ld_st_ty} [%ocur], %val;
    add.u32 %j, %j, 1;
NEXT:
    add.u32 %i, %i, 1; bra LOOP;
DONE: ret;
}}
"
    )
}

// ===========================================================================
// Launch harness
// ===========================================================================

/// Launch a masked_fill kernel over a value buffer of native element type `T`,
/// returning a fresh resident `CudaSlice<T>` of `n` elements.
/// `value` is pushed as the scalar argument `S` (f32 / f64 / i-as-bits) the PTX
/// declares.
#[allow(clippy::too_many_arguments)]
fn launch_masked_fill<T, S>(
    input: &CudaSlice<T>,
    mask: &CudaSlice<u8>,
    value: S,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaSlice<T>>
where
    T: DeviceRepr + ValidAsZeroBits,
    S: DeviceRepr,
{
    if input.len() != mask.len() {
        return Err(GpuError::LengthMismatch {
            a: input.len(),
            b: mask.len(),
        });
    }
    let n = input.len();
    let stream = device.stream();
    if n == 0 {
        return Ok(stream.alloc_zeros::<T>(0)?);
    }
    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;
    let mut out = stream.alloc_zeros::<T>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is the PTX entry `kernel_name`; signature
    //   (in_ptr, mask_ptr, out_ptr, value, n) matches the five args below.
    // - `input` (n `T`-elements) and `mask` (n u8) are length-equal (checked
    //   above); `out` is a fresh n-element `T` buffer, the only `&mut`, not
    //   aliased with the inputs.
    // - Each thread reads input[i]/mask[i] and writes out[i] only for i in
    //   [0,n) (PTX bound check `setp.ge.u32 %p, %idx, %nr`).
    // - `value` is a scalar passed by reference, living for the launch.
    // - `n_u32` is non-truncating for any host-allocatable contiguous buffer.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(mask)
            .arg(&mut out)
            .arg(&value)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Launch a `where` (ternary select) kernel over two value buffers of native
/// element type `T` + a u8 cond buffer, returning a fresh resident
/// `CudaSlice<T>` of `n` elements.
fn launch_where<T: DeviceRepr + ValidAsZeroBits>(
    cond: &CudaSlice<u8>,
    x: &CudaSlice<T>,
    y: &CudaSlice<T>,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaSlice<T>> {
    if x.len() != y.len() || x.len() != cond.len() {
        return Err(GpuError::LengthMismatch {
            a: x.len(),
            b: y.len(),
        });
    }
    let n = x.len();
    let stream = device.stream();
    if n == 0 {
        return Ok(stream.alloc_zeros::<T>(0)?);
    }
    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;
    let mut out = stream.alloc_zeros::<T>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is the PTX entry `kernel_name`; signature
    //   (cond_ptr, x_ptr, y_ptr, out_ptr, n) matches the five args below.
    // - `cond` (n u8), `x`, `y` (n `T` each) are length-equal (checked above);
    //   `out` is a fresh n-element `T` buffer, the only `&mut`, not aliased.
    // - Each thread reads cond[i]/x[i]/y[i] and writes out[i] only for i in
    //   [0,n) (PTX bound check).
    // - `n_u32` is non-truncating.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(cond)
            .arg(x)
            .arg(y)
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Count the number of true (nonzero) bytes in `mask`, returning the count to
/// the host. This single integer is the data-dependent OUTPUT SHAPE for
/// `masked_select` — NOT a data buffer round-trip. PyTorch performs the same
/// device→host sync internally to size `torch.masked_select`'s output. The
/// count reduction itself runs on the device (one i32 written by the kernel);
/// only that one scalar crosses to the host.
pub fn count_true(mask: &CudaSlice<u8>, device: &GpuDevice) -> GpuResult<usize> {
    let n = mask.len();
    let stream = device.stream();
    if n == 0 {
        return Ok(0);
    }
    let ctx = device.context();
    let f = get_or_compile(ctx, COUNT_TRUE_PTX, "count_true_kernel", device.ordinal() as u32)
        .map_err(|e| GpuError::PtxCompileFailed {
            kernel: "count_true_kernel",
            source: e,
        })?;
    let mut out = stream.alloc_zeros::<i32>(1)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is the PTX entry `count_true_kernel`; signature
    //   (mask_ptr, out_ptr, n) matches the three args below.
    // - `mask` is the caller's n-byte buffer; thread 0 reads mask[0..n) and
    //   writes the single out[0] (other threads gated off via `%only0`).
    // - `out` is a fresh 1-element i32 buffer, exclusively borrowed.
    // - `n_u32` is non-truncating.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(mask)
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    // Single-scalar device→host read: the output SHAPE, not the data. (See the
    // doc comment above — PyTorch parity.)
    let host = stream.clone_dtoh(&out)?;
    Ok(host[0].max(0) as usize)
}

/// Serial compaction: write `input[i] -> out[j++]` for each `i` where
/// `mask[i] != 0`. `out_len` is the true count from [`count_true`]. Returns a
/// fresh resident `CudaSlice<T>` of length `out_len`.
fn launch_compact<T: DeviceRepr + ValidAsZeroBits>(
    input: &CudaSlice<T>,
    mask: &CudaSlice<u8>,
    out_len: usize,
    device: &GpuDevice,
    ptx: String,
    kernel_name: String,
) -> GpuResult<CudaSlice<T>> {
    let n = input.len();
    let stream = device.stream();
    if out_len == 0 {
        return Ok(stream.alloc_zeros::<T>(0)?);
    }
    let ctx = device.context();
    let f = crate::module_cache::get_or_compile_owned(ctx, ptx, kernel_name, device.ordinal() as u32)
        .map_err(|e| GpuError::PtxCompileFailed {
            kernel: "masked_select_compact",
            source: e,
        })?;
    let mut out = stream.alloc_zeros::<T>(out_len)?;
    // Single block, single active thread (thread 0 compacts serially).
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is the compaction PTX entry; signature (in_ptr, mask_ptr, out_ptr,
    //   n) matches the four args below.
    // - `input` and `mask` are n-element buffers; thread 0 reads input[0..n) /
    //   mask[0..n) and writes out[0..out_len). out_len equals the true count of
    //   `mask` (from `count_true`), so the `j` counter never exceeds `out_len`;
    //   `out` is a fresh out_len-element buffer, exclusively borrowed.
    // - `n_u32` is non-truncating.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(mask)
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Serial scatter (inverse of [`launch_compact`]): write `grad[j++] -> out[i]`
/// for each `i` where `mask[i] != 0`, leaving every other position 0. `out` is
/// a fresh `out_numel`-element zeroed buffer; `grad` holds the compacted
/// gradient (length = #true). Returns the resident `CudaSlice<T>` of length
/// `out_numel`.
fn launch_scatter<T: DeviceRepr + ValidAsZeroBits>(
    grad: &CudaSlice<T>,
    mask: &CudaSlice<u8>,
    out_numel: usize,
    device: &GpuDevice,
    ptx: String,
    kernel_name: String,
) -> GpuResult<CudaSlice<T>> {
    if mask.len() != out_numel {
        return Err(GpuError::LengthMismatch {
            a: mask.len(),
            b: out_numel,
        });
    }
    let stream = device.stream();
    if out_numel == 0 {
        return Ok(stream.alloc_zeros::<T>(0)?);
    }
    let ctx = device.context();
    let f =
        crate::module_cache::get_or_compile_owned(ctx, ptx, kernel_name, device.ordinal() as u32)
            .map_err(|e| GpuError::PtxCompileFailed {
                kernel: "masked_scatter",
                source: e,
            })?;
    // out_numel-element zeroed buffer: positions where mask is false stay 0.
    let mut out = stream.alloc_zeros::<T>(out_numel)?;
    // Single block, single active thread (thread 0 scatters serially).
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u32 = out_numel as u32;
    // SAFETY:
    // - `f` is the scatter PTX entry; signature (grad_ptr, mask_ptr, out_ptr, n)
    //   matches the four args below.
    // - `mask` is an out_numel-byte buffer; `grad` holds the #true compacted
    //   elements. Thread 0 walks mask[0..out_numel) and reads grad[j] only as it
    //   increments `j` once per true byte, so `j` never exceeds grad.len() (the
    //   true count). `out` is a fresh out_numel-element buffer, exclusively
    //   borrowed, not aliased with `grad`/`mask`.
    // - `n_u32` is non-truncating for any host-allocatable contiguous buffer.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(grad)
            .arg(mask)
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// Public entry points
// ===========================================================================

/// masked_fill for f32: `out[i] = mask[i]!=0 ? value : input[i]`.
pub fn masked_fill_f32(
    input: &CudaSlice<f32>,
    mask: &CudaSlice<u8>,
    value: f32,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<f32>> {
    launch_masked_fill(input, mask, value, d, MASKED_FILL_F32_PTX, "masked_fill_f32_kernel")
}

/// masked_fill for f64.
pub fn masked_fill_f64(
    input: &CudaSlice<f64>,
    mask: &CudaSlice<u8>,
    value: f64,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<f64>> {
    launch_masked_fill(input, mask, value, d, MASKED_FILL_F64_PTX, "masked_fill_f64_kernel")
}

/// masked_fill for f16 (u16 bits). `value` is the f32 fill, narrowed in-kernel.
pub fn masked_fill_f16(
    input: &CudaSlice<u16>,
    mask: &CudaSlice<u8>,
    value: f32,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_masked_fill(input, mask, value, d, MASKED_FILL_F16_PTX, "masked_fill_f16_kernel")
}

/// masked_fill for bf16 (u16 bits). `value` is the f32 fill, narrowed in-kernel.
pub fn masked_fill_bf16(
    input: &CudaSlice<u16>,
    mask: &CudaSlice<u8>,
    value: f32,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_masked_fill(input, mask, value, d, MASKED_FILL_BF16_PTX, "masked_fill_bf16_kernel")
}

/// masked_fill for i32. `value` is the fill, passed as raw 32-bit.
pub fn masked_fill_i32(
    input: &CudaSlice<i32>,
    mask: &CudaSlice<u8>,
    value: i32,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i32>> {
    launch_masked_fill(input, mask, value, d, MASKED_FILL_I32_PTX, "masked_fill_i32_kernel")
}

/// masked_fill for i64. `value` is the fill, passed as raw 64-bit.
pub fn masked_fill_i64(
    input: &CudaSlice<i64>,
    mask: &CudaSlice<u8>,
    value: i64,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_masked_fill(input, mask, value, d, MASKED_FILL_I64_PTX, "masked_fill_i64_kernel")
}

/// where (ternary select) for a 32-bit value dtype (f32 / i32).
pub fn where_32<T: DeviceRepr + ValidAsZeroBits>(
    cond: &CudaSlice<u8>,
    x: &CudaSlice<T>,
    y: &CudaSlice<T>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<T>> {
    debug_assert_eq!(std::mem::size_of::<T>(), 4, "where_32 requires a 4-byte element");
    launch_where(cond, x, y, d, WHERE_32_PTX, "where_32_kernel")
}

/// where (ternary select) for a 64-bit value dtype (f64 / i64).
pub fn where_64<T: DeviceRepr + ValidAsZeroBits>(
    cond: &CudaSlice<u8>,
    x: &CudaSlice<T>,
    y: &CudaSlice<T>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<T>> {
    debug_assert_eq!(std::mem::size_of::<T>(), 8, "where_64 requires an 8-byte element");
    launch_where(cond, x, y, d, WHERE_64_PTX, "where_64_kernel")
}

/// where (ternary select) for a 16-bit value dtype (f16 / bf16; pure bit
/// select, no decode).
pub fn where_16(
    cond: &CudaSlice<u8>,
    x: &CudaSlice<u16>,
    y: &CudaSlice<u16>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_where(cond, x, y, d, WHERE_16_PTX, "where_16_kernel")
}

/// masked_select for a 32-bit value dtype (f32 / i32): returns `(out, len)`
/// where `out` holds the `len` elements of `input` where `mask` is true, 1-D
/// and GPU-resident. `len` is the on-device true count read once to the host to
/// size the output (the data-dependent shape — see [`count_true`]).
pub fn masked_select_32<T: DeviceRepr + ValidAsZeroBits>(
    input: &CudaSlice<T>,
    mask: &CudaSlice<u8>,
    d: &GpuDevice,
) -> GpuResult<(CudaSlice<T>, usize)> {
    let len = count_true(mask, d)?;
    let ptx = compact_ptx("masked_select_compact_32", 2, "b32", ".reg .b32 %val;");
    let out = launch_compact(input, mask, len, d, ptx, "masked_select_compact_32".to_string())?;
    Ok((out, len))
}

/// masked_select for a 64-bit value dtype (f64 / i64).
pub fn masked_select_64<T: DeviceRepr + ValidAsZeroBits>(
    input: &CudaSlice<T>,
    mask: &CudaSlice<u8>,
    d: &GpuDevice,
) -> GpuResult<(CudaSlice<T>, usize)> {
    let len = count_true(mask, d)?;
    let ptx = compact_ptx("masked_select_compact_64", 3, "b64", ".reg .b64 %val;");
    let out = launch_compact(input, mask, len, d, ptx, "masked_select_compact_64".to_string())?;
    Ok((out, len))
}

/// masked_select for a 16-bit value dtype (f16 / bf16).
pub fn masked_select_16(
    input: &CudaSlice<u16>,
    mask: &CudaSlice<u8>,
    d: &GpuDevice,
) -> GpuResult<(CudaSlice<u16>, usize)> {
    let len = count_true(mask, d)?;
    let ptx = compact_ptx("masked_select_compact_16", 1, "b16", ".reg .b16 %val;");
    let out = launch_compact(input, mask, len, d, ptx, "masked_select_compact_16".to_string())?;
    Ok((out, len))
}

/// masked_scatter for a 32-bit value dtype (f32 / i32): scatter the compacted
/// `grad` back into a zeros buffer of length `out_numel` at the positions where
/// `mask` is true (the VJP of [`masked_select_32`]).
pub fn masked_scatter_32<T: DeviceRepr + ValidAsZeroBits>(
    grad: &CudaSlice<T>,
    mask: &CudaSlice<u8>,
    out_numel: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<T>> {
    debug_assert_eq!(std::mem::size_of::<T>(), 4, "masked_scatter_32 requires a 4-byte element");
    let ptx = scatter_ptx("masked_scatter_32", 2, "b32", ".reg .b32 %val;");
    launch_scatter(grad, mask, out_numel, d, ptx, "masked_scatter_32".to_string())
}

/// masked_scatter for a 64-bit value dtype (f64 / i64).
pub fn masked_scatter_64<T: DeviceRepr + ValidAsZeroBits>(
    grad: &CudaSlice<T>,
    mask: &CudaSlice<u8>,
    out_numel: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<T>> {
    debug_assert_eq!(std::mem::size_of::<T>(), 8, "masked_scatter_64 requires an 8-byte element");
    let ptx = scatter_ptx("masked_scatter_64", 3, "b64", ".reg .b64 %val;");
    launch_scatter(grad, mask, out_numel, d, ptx, "masked_scatter_64".to_string())
}

/// masked_scatter for a 16-bit value dtype (f16 / bf16; pure 16-bit bit copy,
/// no decode — we only move bit patterns into their original slots).
pub fn masked_scatter_16(
    grad: &CudaSlice<u16>,
    mask: &CudaSlice<u8>,
    out_numel: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    let ptx = scatter_ptx("masked_scatter_16", 1, "b16", ".reg .b16 %val;");
    launch_scatter(grad, mask, out_numel, d, ptx, "masked_scatter_16".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> GpuDevice {
        GpuDevice::new(0).expect("cuda device")
    }

    #[test]
    fn masked_fill_f32_replaces_true_positions() {
        let d = dev();
        let input = d.stream().clone_htod(&vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let mask = d.stream().clone_htod(&vec![0u8, 1, 0, 1]).unwrap();
        let r = masked_fill_f32(&input, &mask, -9.0, &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&r).unwrap(), vec![1.0f32, -9.0, 3.0, -9.0]);
    }

    #[test]
    fn where_32_selects() {
        let d = dev();
        let cond = d.stream().clone_htod(&vec![1u8, 0, 1, 0]).unwrap();
        let x = d.stream().clone_htod(&vec![10.0f32, 20.0, 30.0, 40.0]).unwrap();
        let y = d.stream().clone_htod(&vec![-1.0f32, -2.0, -3.0, -4.0]).unwrap();
        let r = where_32::<f32>(&cond, &x, &y, &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&r).unwrap(), vec![10.0f32, -2.0, 30.0, -4.0]);
    }

    #[test]
    fn masked_select_32_compacts() {
        let d = dev();
        let input = d.stream().clone_htod(&vec![1.0f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let mask = d.stream().clone_htod(&vec![1u8, 0, 1, 1, 0]).unwrap();
        let (out, len) = masked_select_32::<f32>(&input, &mask, &d).unwrap();
        assert_eq!(len, 3);
        assert_eq!(d.stream().clone_dtoh(&out).unwrap(), vec![1.0f32, 3.0, 4.0]);
    }

    #[test]
    fn masked_scatter_32_is_inverse_of_compact() {
        let d = dev();
        // mask [1,0,1,1,0]; compacted grad [g0,g2,g3] scatters back to
        // out = [g0, 0, g2, g3, 0].
        let mask = d.stream().clone_htod(&vec![1u8, 0, 1, 1, 0]).unwrap();
        let grad = d.stream().clone_htod(&vec![10.0f32, 30.0, 40.0]).unwrap();
        let out = masked_scatter_32::<f32>(&grad, &mask, 5, &d).unwrap();
        assert_eq!(
            d.stream().clone_dtoh(&out).unwrap(),
            vec![10.0f32, 0.0, 30.0, 40.0, 0.0]
        );
    }

    #[test]
    fn masked_scatter_16_bf16_bits_roundtrip() {
        let d = dev();
        // bf16 bit patterns for 1.0 (0x3F80) and 2.0 (0x4000).
        let one = half::bf16::from_f32(1.0).to_bits();
        let two = half::bf16::from_f32(2.0).to_bits();
        let mask = d.stream().clone_htod(&vec![0u8, 1, 1]).unwrap();
        let grad = d.stream().clone_htod(&vec![one, two]).unwrap();
        let out = masked_scatter_16(&grad, &mask, 3, &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&out).unwrap(), vec![0u16, one, two]);
    }

    #[test]
    fn count_true_counts() {
        let d = dev();
        let mask = d.stream().clone_htod(&vec![1u8, 0, 1, 1, 0, 1]).unwrap();
        assert_eq!(count_true(&mask, &d).unwrap(), 4);
        let empty: Vec<u8> = vec![];
        let m0 = d.stream().clone_htod(&empty).unwrap();
        assert_eq!(count_true(&m0, &d).unwrap(), 0);
    }
}
