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
//!
//! ## REQ status (per `.design/ferrotorch-gpu/masked_kernels.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream
//! cites) live in the design doc; this synopsis is a one-line summary per
//! REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`masked_fill` per-dtype) | SHIPPED | six `pub fn masked_fill_*` symbols in `masked_kernels.rs`; consumer `use crate::masked_kernels as mk` site in `backend_impl.rs` dispatches per-dtype calls |
//! | REQ-2 (`where`) | SHIPPED | `pub fn where_32 / where_64 / where_16 in masked_kernels.rs`; consumer `CudaBackendImpl::where_cond_* in backend_impl.rs` |
//! | REQ-3 (`masked_select`) | SHIPPED | `pub fn count_true in masked_kernels.rs` + `masked_select_32 / masked_select_64 / masked_select_16`; consumer `CudaBackendImpl::masked_select_* in backend_impl.rs` |
//! | REQ-4 (`masked_scatter`) | SHIPPED | backward `pub fn masked_scatter_32 / masked_scatter_64 / masked_scatter_16` (VJP) + forward `pub fn masked_scatter_forward_32 / masked_scatter_forward_64 / masked_scatter_forward_16` (#1662, `out[i] = mask[i] ? source[j++] : input[i]`) in `masked_kernels.rs`; consumers `CudaBackendImpl::masked_scatter` / `masked_scatter_forward in backend_impl.rs` (the latter wired by `grad_fns::indexing::masked_scatter`'s all-CUDA branch so a fully-resident forward keeps `is_cuda()`) |
//! | REQ-5 (single PTX load path) | SHIPPED | `use crate::module_cache::get_or_compile` in `masked_kernels.rs` binds the single PTX load path; no `cudarc::nvrtc` import — every launch routes through `module_cache` |
//! | REQ-6 ((op, dtype) coverage matrix) | SHIPPED | per-dtype `pub fn` entries in `masked_kernels.rs` mean (op, dtype) coverage is structurally surfaced — a missing combination is a missing function symbol that the `backend_impl` dispatcher converts to `FerrotorchError::NotImplementedOnCuda` |
//! | REQ-7 (workspace consumer wiring) | SHIPPED | four `use crate::masked_kernels as mk` sites in `backend_impl.rs` (each is the body of a `CudaBackendImpl` trait method); ferrotorch-core dispatches `Tensor::masked_fill / etc.` through the `GpuBackend` trait when input is CUDA-resident |

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
// masked_scatter FORWARD (#1662): out = input.clone(); for each i where
// mask[i]!=0, out[i] = source[j++] (source consumed serially in flat order),
// else out[i] = input[i].
//
// Signature: (in_ptr, src_ptr, mask_ptr, out_ptr, n)
//   in:   VAL_BYTES per element, length n (the base tensor)
//   src:  VAL_BYTES per element, length >= #true (the scatter source)
//   mask: 1 byte per element (u8 0/1), length n
//   out:  VAL_BYTES per element, length n (fresh)
//   for i in [0,n): out[i] = mask[i]!=0 ? src[j++] : in[i]
//
// This differs from the backward `scatter_ptx` above in the FALSE branch: the
// backward leaves a pre-zeroed buffer untouched (VJP semantic), whereas the
// forward copies `in[i]` through. The serial source cursor `j` is exactly the
// EXCLUSIVE PREFIX-SUM of the mask at position `i` (the offset upstream computes
// via `at::cuda::cub::mask_exclusive_sum` in
// `aten/src/ATen/native/cuda/IndexKernel.cu:416` then gathers
// `source[maskPrefixSum]` at `:450`); a single in-order thread realises the
// same offset without a separate scan buffer (matching COMPACT/COUNT_TRUE).
// VAL_SHIFT is log2(VAL_BYTES).
// ===========================================================================
fn scatter_forward_ptx(
    kernel_name: &str,
    val_shift: u32,
    ld_st_ty: &str,
    reg_decl: &str,
) -> String {
    format!(
        "\
.version 7.0
.target sm_52
.address_size 64
.visible .entry {kernel_name}(.param .u64 in_ptr, .param .u64 src_ptr, .param .u64 mask_ptr, .param .u64 out_ptr, .param .u32 n) {{
    .reg .u32 %idx, %bid, %bdim, %nr, %i, %j;
    .reg .u64 %in, %sr, %mk, %out, %eoff, %soff, %icur, %scur, %mcur, %ocur;
    .reg .u16 %m; {reg_decl}
    .reg .pred %only0, %p, %nz;
    ld.param.u64 %in, [in_ptr]; ld.param.u64 %sr, [src_ptr];
    ld.param.u64 %mk, [mask_ptr]; ld.param.u64 %out, [out_ptr]; ld.param.u32 %nr, [n];
    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ne.u32 %only0, %idx, 0; @%only0 bra DONE;
    mov.u32 %i, 0; mov.u32 %j, 0;
LOOP:
    setp.ge.u32 %p, %i, %nr; @%p bra DONE;
    cvt.u64.u32 %eoff, %i; add.u64 %mcur, %mk, %eoff;
    ld.global.u8 %m, [%mcur]; setp.ne.u16 %nz, %m, 0;
    shl.b64 %eoff, %eoff, {val_shift};
    @!%nz bra COPYIN;
    // mask[i] true: out[i] = src[j]; j++
    cvt.u64.u32 %soff, %j; shl.b64 %soff, %soff, {val_shift}; add.u64 %scur, %sr, %soff;
    ld.global.{ld_st_ty} %val, [%scur];
    add.u64 %ocur, %out, %eoff; st.global.{ld_st_ty} [%ocur], %val;
    add.u32 %j, %j, 1;
    bra NEXT;
COPYIN:
    // mask[i] false: out[i] = in[i]
    add.u64 %icur, %in, %eoff; ld.global.{ld_st_ty} %val, [%icur];
    add.u64 %ocur, %out, %eoff; st.global.{ld_st_ty} [%ocur], %val;
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
///
/// `n` is the LOGICAL element count of the operands (`CudaBuffer::len()`), not
/// the raw `CudaSlice::len()`. The raw `input`/`mask` slices may be
/// OVER-ALLOCATED past `n`: a `.contiguous()`-materialised row-narrowed view is
/// backed by a pooled buffer rounded up to a multiple of `ROUND_ELEMENTS`
/// (#1661), while the mask is exact-length. We therefore validate and launch on
/// the logical `n`, treating each raw slice as a backing store that need only be
/// `>= n`; comparing raw lens would spuriously reject `256 vs 6`. The caller
/// (dispatch site) supplies `n` from the logical buffer len and owns the
/// input/mask numel equality check.
#[allow(clippy::too_many_arguments)]
fn launch_masked_fill<T, S>(
    input: &CudaSlice<T>,
    mask: &CudaSlice<u8>,
    value: S,
    n: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaSlice<T>>
where
    T: DeviceRepr + ValidAsZeroBits,
    S: DeviceRepr,
{
    if input.len() < n || mask.len() < n {
        return Err(GpuError::LengthMismatch {
            a: input.len().min(mask.len()),
            b: n,
        });
    }
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
    // - `input` (`T`) and `mask` (u8) each back AT LEAST `n` elements
    //   (`*.len() >= n` checked above; the input may be a pooled, over-allocated
    //   `.contiguous()` materialisation of a row-narrowed view, #1661); the
    //   kernel reads only `[0, n)`. `out` is a fresh n-element `T` buffer, the
    //   only `&mut`, not aliased with the inputs.
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
///
/// `n` is the LOGICAL element count of the operands (`CudaBuffer::len()`), not
/// the raw `CudaSlice::len()`. The raw `cond`/`x`/`y` slices may be
/// OVER-ALLOCATED past `n`: a `.contiguous()`-materialised view is backed by a
/// pooled buffer rounded up to a multiple of `ROUND_ELEMENTS` (#1660), while a
/// `clone_htod` operand is exact-length. We therefore validate and launch on
/// the logical `n`, treating each raw slice as a backing store that need only be
/// `>= n`; comparing raw lens would spuriously reject `256 vs 6`. The caller
/// (dispatch site) supplies `n` from the logical buffer len and owns the
/// operand-shape equality check.
fn launch_where<T: DeviceRepr + ValidAsZeroBits>(
    cond: &CudaSlice<u8>,
    x: &CudaSlice<T>,
    y: &CudaSlice<T>,
    n: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaSlice<T>> {
    if x.len() < n || y.len() < n || cond.len() < n {
        return Err(GpuError::LengthMismatch {
            a: x.len().min(y.len()).min(cond.len()),
            b: n,
        });
    }
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
    // - `cond` (u8), `x`, `y` (`T`) each back AT LEAST `n` elements
    //   (`*.len() >= n` checked above; any may be a pooled, over-allocated
    //   `.contiguous()` materialisation, #1660); the kernel reads only `[0, n)`.
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
    let f = get_or_compile(
        ctx,
        COUNT_TRUE_PTX,
        "count_true_kernel",
        device.ordinal() as u32,
    )
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
    let f =
        crate::module_cache::get_or_compile_owned(ctx, ptx, kernel_name, device.ordinal() as u32)
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

/// Serial masked_scatter FORWARD (#1662): write `out[i] = mask[i]!=0 ?
/// source[j++] : input[i]`, walking `[0,n)` in order with a serial source
/// cursor `j`. `out` is a fresh `n`-element buffer (NOT pre-zeroed — every
/// position is written, either from `source` or passed through from `input`).
///
/// `n` is the LOGICAL element count of `input`/`mask` (their numel). The raw
/// slices may be over-allocated past `n` (a pooled `.contiguous()`
/// materialisation, #1661 class) so we validate `>= n`, matching
/// `launch_masked_fill`/`launch_where`. `source` need only hold the #true
/// elements; the caller (core layer) has already checked
/// `source.numel() >= count_nonzero(mask)`, so the in-order cursor `j` never
/// exceeds `source.len()`.
fn launch_scatter_forward<T: DeviceRepr + ValidAsZeroBits>(
    input: &CudaSlice<T>,
    source: &CudaSlice<T>,
    mask: &CudaSlice<u8>,
    n: usize,
    device: &GpuDevice,
    ptx: String,
    kernel_name: String,
) -> GpuResult<CudaSlice<T>> {
    if input.len() < n || mask.len() < n {
        return Err(GpuError::LengthMismatch {
            a: input.len().min(mask.len()),
            b: n,
        });
    }
    let stream = device.stream();
    if n == 0 {
        return Ok(stream.alloc_zeros::<T>(0)?);
    }
    let ctx = device.context();
    let f =
        crate::module_cache::get_or_compile_owned(ctx, ptx, kernel_name, device.ordinal() as u32)
            .map_err(|e| GpuError::PtxCompileFailed {
            kernel: "masked_scatter_forward",
            source: e,
        })?;
    let mut out = stream.alloc_zeros::<T>(n)?;
    // Single block, single active thread (thread 0 scatters serially).
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is the forward-scatter PTX entry; signature
    //   (in_ptr, src_ptr, mask_ptr, out_ptr, n) matches the five args below.
    // - `input` (`T`) and `mask` (u8) each back AT LEAST `n` elements
    //   (`*.len() >= n` checked above; either may be a pooled, over-allocated
    //   `.contiguous()` materialisation). Thread 0 reads only `[0, n)` of each.
    // - `source` holds the #true compacted scatter values; thread 0 reads
    //   `source[j]` only as it increments `j` once per true byte. The caller
    //   guarantees `source.len() >= count_nonzero(mask)`, so `j` stays in bounds.
    // - `out` is a fresh n-element `T` buffer, the only `&mut`, not aliased with
    //   `input`/`source`/`mask`; thread 0 writes each `out[i]` exactly once.
    // - `n_u32` is non-truncating for any host-allocatable contiguous buffer.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(source)
            .arg(mask)
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// Predicate-mask kernels (#1545 / #1534): compute a Bool (u8 0/1) mask
// directly on-device from a float value buffer, so the masked-tensor
// constructors `masked_invalid` / `masked_equal` never download the data
// tensor to host to compute the predicate. The result is the host-resident
// mask of `MaskedTensor` (which is fundamentally a `Vec<bool>`), so the only
// host crossing is the one-way readback of the freshly-computed predicate
// bytes — the value data never leaves and returns to the device (no R-CODE-4
// round trip).
//
// - isfinite: `out[i] = (v==v) && (|v| != +inf)` — exact PyTorch parity with
//   `aten/src/ATen/native/TensorCompare.cpp:484` `isfinite = (self == self) *
//   (self.abs() != inf)`. f16/bf16 first widen each stored 16-bit value to f32,
//   matching PyTorch half/bfloat predicate math without staging the data on
//   host. `setp.eq.f{32,64}` is unordered-false so NaN gives `v==v -> false`;
//   `|v|!=inf` is ordered, and ordered/unordered agree once `v==v` held.
// - ne_scalar: `out[i] = (v != value)` — the VALID mask for `masked_equal`
//   under the torch convention (positions equal to `value` are masked OUT, so
//   `mask = (v != value)`). The kernel uses `setp.neu.f{32,64}` (the UNORDERED
//   not-equal: NaN != x is true), matching the CPU `v != value` walk where Rust
//   `NaN != x` is true. Plain `setp.ne` is the ordered form (NaN -> false) and
//   would diverge.
// ===========================================================================

/// PTX for `isfinite`: reads `v` (f{32,64}), writes 1 if `v` is finite else 0.
/// `ty` ∈ {"f32","f64"}; `in_shift` = log2(elem bytes) (2 for f32, 3 for f64).
/// `inf_lit` is the PTX float literal for +inf in that width.
fn isfinite_ptx(kernel_name: &str, ty: &str, in_shift: u32, inf_lit: &str) -> String {
    format!(
        "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n
) {{
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %out, %ioff, %ooff;
    .reg .{ty} %v, %av, %inf;
    .reg .u16 %res;
    .reg .pred %p, %notnan, %notinf, %fin;

    ld.param.u64 %a, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nr, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    cvt.u64.u32 %ioff, %idx;
    shl.b64 %ioff, %ioff, {in_shift};
    add.u64 %a, %a, %ioff;
    cvt.u64.u32 %ooff, %idx;
    add.u64 %out, %out, %ooff;

    ld.global.{ty} %v, [%a];
    // not-NaN: v == v  (setp.eq is unordered-false, so NaN -> false)
    setp.eq.{ty} %notnan, %v, %v;
    // |v| != inf
    abs.{ty} %av, %v;
    mov.{ty} %inf, {inf_lit};
    setp.ne.{ty} %notinf, %av, %inf;
    and.pred %fin, %notnan, %notinf;
    selp.u16 %res, 1, 0, %fin;
    st.global.u8 [%out], %res;
DONE:
    ret;
}}
"
    )
}

/// PTX for `ne_scalar`: reads `v` (f{32,64}), writes 1 if `v != value` else 0.
/// `value` is a kernel scalar param of the same width as the elements.
fn ne_scalar_ptx(kernel_name: &str, ty: &str, in_shift: u32) -> String {
    format!(
        "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 in_ptr, .param .u64 out_ptr, .param .{ty} value, .param .u32 n
) {{
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %out, %ioff, %ooff;
    .reg .{ty} %v, %val;
    .reg .u16 %res;
    .reg .pred %p, %c;

    ld.param.u64 %a, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.{ty} %val, [value];
    ld.param.u32 %nr, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    cvt.u64.u32 %ioff, %idx;
    shl.b64 %ioff, %ioff, {in_shift};
    add.u64 %a, %a, %ioff;
    cvt.u64.u32 %ooff, %idx;
    add.u64 %out, %out, %ooff;

    ld.global.{ty} %v, [%a];
    // setp.neu.f is the UNORDERED not-equal: NaN != value -> true (matches the
    // CPU `v != value` walk where Rust `NaN != x` is true). Plain `setp.ne.f`
    // is the *ordered* form (NaN -> false), which would diverge from the CPU
    // reference, so `.neu` is required here.
    setp.neu.{ty} %c, %v, %val;
    selp.u16 %res, 1, 0, %c;
    st.global.u8 [%out], %res;
DONE:
    ret;
}}
"
    )
}

/// PTX for 16-bit float `isfinite`. The element is widened into f32 first,
/// then compared with the same predicate used by torch's float path.
fn isfinite_f16_ptx(kernel_name: &str) -> String {
    format!(
        "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n
) {{
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %out, %ioff, %ooff;
    .reg .b16 %h;
    .reg .f32 %v, %av, %inf;
    .reg .u16 %res;
    .reg .pred %p, %notnan, %notinf, %fin;

    ld.param.u64 %a, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nr, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    cvt.u64.u32 %ioff, %idx;
    shl.b64 %ioff, %ioff, 1;
    add.u64 %a, %a, %ioff;
    cvt.u64.u32 %ooff, %idx;
    add.u64 %out, %out, %ooff;

    ld.global.b16 %h, [%a];
    cvt.f32.f16 %v, %h;
    setp.eq.f32 %notnan, %v, %v;
    abs.f32 %av, %v;
    mov.f32 %inf, 0f7F800000;
    setp.ne.f32 %notinf, %av, %inf;
    and.pred %fin, %notnan, %notinf;
    selp.u16 %res, 1, 0, %fin;
    st.global.u8 [%out], %res;
DONE:
    ret;
}}
"
    )
}

/// PTX for bf16 `isfinite`. bf16 is stored as the high 16 bits of an f32, so
/// decode by shifting the u16 payload into the high half of a b32 register.
fn isfinite_bf16_ptx(kernel_name: &str) -> String {
    format!(
        "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n
) {{
    .reg .u32 %idx, %bid, %bdim, %nr, %bits;
    .reg .u64 %a, %out, %ioff, %ooff;
    .reg .u16 %h;
    .reg .f32 %v, %av, %inf;
    .reg .u16 %res;
    .reg .pred %p, %notnan, %notinf, %fin;

    ld.param.u64 %a, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nr, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    cvt.u64.u32 %ioff, %idx;
    shl.b64 %ioff, %ioff, 1;
    add.u64 %a, %a, %ioff;
    cvt.u64.u32 %ooff, %idx;
    add.u64 %out, %out, %ooff;

    ld.global.u16 %h, [%a];
    cvt.u32.u16 %bits, %h;
    shl.b32 %bits, %bits, 16;
    mov.b32 %v, %bits;
    setp.eq.f32 %notnan, %v, %v;
    abs.f32 %av, %v;
    mov.f32 %inf, 0f7F800000;
    setp.ne.f32 %notinf, %av, %inf;
    and.pred %fin, %notnan, %notinf;
    selp.u16 %res, 1, 0, %fin;
    st.global.u8 [%out], %res;
DONE:
    ret;
}}
"
    )
}

/// PTX for 16-bit float `v != value`. `value` is supplied as f32 after the
/// core layer converts the caller's half scalar through `T`, so equality is
/// exact for representable half values. `setp.neu` preserves NaN != value.
fn ne_scalar_f16_ptx(kernel_name: &str) -> String {
    format!(
        "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 in_ptr, .param .u64 out_ptr, .param .f32 value, .param .u32 n
) {{
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %out, %ioff, %ooff;
    .reg .b16 %h;
    .reg .f32 %v, %val;
    .reg .u16 %res;
    .reg .pred %p, %c;

    ld.param.u64 %a, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.f32 %val, [value];
    ld.param.u32 %nr, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    cvt.u64.u32 %ioff, %idx;
    shl.b64 %ioff, %ioff, 1;
    add.u64 %a, %a, %ioff;
    cvt.u64.u32 %ooff, %idx;
    add.u64 %out, %out, %ooff;

    ld.global.b16 %h, [%a];
    cvt.f32.f16 %v, %h;
    setp.neu.f32 %c, %v, %val;
    selp.u16 %res, 1, 0, %c;
    st.global.u8 [%out], %res;
DONE:
    ret;
}}
"
    )
}

/// PTX for bf16 `v != value`, decoding bf16 to f32 before the unordered compare.
fn ne_scalar_bf16_ptx(kernel_name: &str) -> String {
    format!(
        "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 in_ptr, .param .u64 out_ptr, .param .f32 value, .param .u32 n
) {{
    .reg .u32 %idx, %bid, %bdim, %nr, %bits;
    .reg .u64 %a, %out, %ioff, %ooff;
    .reg .u16 %h;
    .reg .f32 %v, %val;
    .reg .u16 %res;
    .reg .pred %p, %c;

    ld.param.u64 %a, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.f32 %val, [value];
    ld.param.u32 %nr, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    cvt.u64.u32 %ioff, %idx;
    shl.b64 %ioff, %ioff, 1;
    add.u64 %a, %a, %ioff;
    cvt.u64.u32 %ooff, %idx;
    add.u64 %out, %out, %ooff;

    ld.global.u16 %h, [%a];
    cvt.u32.u16 %bits, %h;
    shl.b32 %bits, %bits, 16;
    mov.b32 %v, %bits;
    setp.neu.f32 %c, %v, %val;
    selp.u16 %res, 1, 0, %c;
    st.global.u8 [%out], %res;
DONE:
    ret;
}}
"
    )
}

/// Launch a unary predicate kernel `(in_ptr, out_ptr, n)` over a value buffer
/// of native element type `T`, producing a fresh `CudaSlice<u8>` of `n` 0/1
/// bytes resident on `device`.
fn launch_predicate<T: DeviceRepr + ValidAsZeroBits>(
    input: &CudaSlice<T>,
    device: &GpuDevice,
    ptx: String,
    kernel_name: String,
) -> GpuResult<CudaSlice<u8>> {
    let n = input.len();
    let stream = device.stream();
    if n == 0 {
        return Ok(stream.alloc_zeros::<u8>(0)?);
    }
    let ctx = device.context();
    let f =
        crate::module_cache::get_or_compile_owned(ctx, ptx, kernel_name, device.ordinal() as u32)
            .map_err(|e| GpuError::PtxCompileFailed {
            kernel: "masked_predicate",
            source: e,
        })?;
    let mut out = stream.alloc_zeros::<u8>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is the predicate PTX entry; signature (in_ptr, out_ptr, n) matches
    //   the three args below.
    // - `input` is the caller's n-element `T` buffer; `out` is a fresh n-element
    //   u8 buffer, the only `&mut`, not aliased with `input`.
    // - Each thread reads input[i] and writes out[i] only for i in [0,n) (PTX
    //   bound check `setp.ge.u32 %p, %idx, %nr`).
    // - `n_u32` is non-truncating for any host-allocatable contiguous buffer.
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

/// Launch a unary predicate-with-scalar kernel `(in_ptr, out_ptr, value, n)`
/// over a value buffer of native element type `T`, producing a fresh
/// `CudaSlice<u8>` of `n` 0/1 bytes resident on `device`.
fn launch_predicate_scalar<T: DeviceRepr + ValidAsZeroBits, S: DeviceRepr>(
    input: &CudaSlice<T>,
    value: S,
    device: &GpuDevice,
    ptx: String,
    kernel_name: String,
) -> GpuResult<CudaSlice<u8>> {
    let n = input.len();
    let stream = device.stream();
    if n == 0 {
        return Ok(stream.alloc_zeros::<u8>(0)?);
    }
    let ctx = device.context();
    let f =
        crate::module_cache::get_or_compile_owned(ctx, ptx, kernel_name, device.ordinal() as u32)
            .map_err(|e| GpuError::PtxCompileFailed {
            kernel: "masked_predicate_scalar",
            source: e,
        })?;
    let mut out = stream.alloc_zeros::<u8>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is the predicate-scalar PTX entry; signature
    //   (in_ptr, out_ptr, value, n) matches the four args below.
    // - `input` is the caller's n-element `T` buffer; `out` is a fresh
    //   n-element u8 buffer, the only `&mut`, not aliased with `input`.
    // - `value` is a scalar passed by reference, living for the launch.
    // - Each thread reads input[i] and writes out[i] only for i in [0,n).
    // - `n_u32` is non-truncating.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&value)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// Public entry points
// ===========================================================================

/// `isfinite` mask for f32: `out[i] = (v==v) && (|v| != +inf)` (1 if finite).
///
/// Mirrors `aten/src/ATen/native/TensorCompare.cpp:484`
/// (`isfinite = (self == self) * (self.abs() != inf)`). The result is a
/// `DType::Bool` (u8 0/1) buffer the caller reads back to populate the
/// host-resident `MaskedTensor` mask.
pub fn isfinite_mask_f32(input: &CudaSlice<f32>, d: &GpuDevice) -> GpuResult<CudaSlice<u8>> {
    let ptx = isfinite_ptx("isfinite_mask_f32_kernel", "f32", 2, "0f7F800000");
    launch_predicate(input, d, ptx, "isfinite_mask_f32_kernel".to_string())
}

/// `isfinite` mask for f64. `+inf` (f64) literal is `0d7FF0000000000000`.
pub fn isfinite_mask_f64(input: &CudaSlice<f64>, d: &GpuDevice) -> GpuResult<CudaSlice<u8>> {
    let ptx = isfinite_ptx("isfinite_mask_f64_kernel", "f64", 3, "0d7FF0000000000000");
    launch_predicate(input, d, ptx, "isfinite_mask_f64_kernel".to_string())
}

/// `isfinite` mask for f16, decoded to f32 on-device before the predicate.
pub fn isfinite_mask_f16(input: &CudaSlice<u16>, d: &GpuDevice) -> GpuResult<CudaSlice<u8>> {
    let ptx = isfinite_f16_ptx("isfinite_mask_f16_kernel");
    launch_predicate(input, d, ptx, "isfinite_mask_f16_kernel".to_string())
}

/// `isfinite` mask for bf16, decoded to f32 on-device before the predicate.
pub fn isfinite_mask_bf16(input: &CudaSlice<u16>, d: &GpuDevice) -> GpuResult<CudaSlice<u8>> {
    let ptx = isfinite_bf16_ptx("isfinite_mask_bf16_kernel");
    launch_predicate(input, d, ptx, "isfinite_mask_bf16_kernel".to_string())
}

/// `ne_scalar` mask for f32: `out[i] = (v != value)` (1 where not equal).
///
/// This is the VALID mask for `numpy.ma.masked_equal` under the torch
/// convention (entries equal to `value` are masked OUT, so the valid mask is
/// `v != value`). `setp.ne.f32` is unordered-true (NaN gives true), matching
/// the CPU `v != value` walk.
pub fn ne_scalar_mask_f32(
    input: &CudaSlice<f32>,
    value: f32,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let ptx = ne_scalar_ptx("ne_scalar_mask_f32_kernel", "f32", 2);
    launch_predicate_scalar(
        input,
        value,
        d,
        ptx,
        "ne_scalar_mask_f32_kernel".to_string(),
    )
}

/// `ne_scalar` mask for f64.
pub fn ne_scalar_mask_f64(
    input: &CudaSlice<f64>,
    value: f64,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let ptx = ne_scalar_ptx("ne_scalar_mask_f64_kernel", "f64", 3);
    launch_predicate_scalar(
        input,
        value,
        d,
        ptx,
        "ne_scalar_mask_f64_kernel".to_string(),
    )
}

/// `ne_scalar` mask for f16, decoded to f32 on-device before unordered compare.
pub fn ne_scalar_mask_f16(
    input: &CudaSlice<u16>,
    value: f32,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let ptx = ne_scalar_f16_ptx("ne_scalar_mask_f16_kernel");
    launch_predicate_scalar(
        input,
        value,
        d,
        ptx,
        "ne_scalar_mask_f16_kernel".to_string(),
    )
}

/// `ne_scalar` mask for bf16, decoded to f32 on-device before unordered compare.
pub fn ne_scalar_mask_bf16(
    input: &CudaSlice<u16>,
    value: f32,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let ptx = ne_scalar_bf16_ptx("ne_scalar_mask_bf16_kernel");
    launch_predicate_scalar(
        input,
        value,
        d,
        ptx,
        "ne_scalar_mask_bf16_kernel".to_string(),
    )
}

#[derive(Clone, Copy)]
enum Float16Kind {
    F16,
    Bf16,
}

#[derive(Clone, Copy)]
enum ExtremeKind {
    Min,
    Max,
}

impl ExtremeKind {
    fn sentinel_f32(self) -> &'static str {
        match self {
            Self::Min => "0f7F800000",
            Self::Max => "0fFF800000",
        }
    }

    fn compare_ptx(self) -> &'static str {
        match self {
            Self::Min => "setp.lt.f32 %cmp, %val, %acc;",
            Self::Max => "setp.gt.f32 %cmp, %val, %acc;",
        }
    }
}

fn load_16_to_f32(kind: Float16Kind, ptr: &str, out: &str) -> String {
    match kind {
        Float16Kind::F16 => format!(
            "\
    ld.global.b16 %h, [{ptr}];
    cvt.f32.f16 {out}, %h;"
        ),
        Float16Kind::Bf16 => format!(
            "\
    ld.global.u16 %h, [{ptr}];
    cvt.u32.u16 %bits, %h;
    shl.b32 %bits, %bits, 16;
    mov.b32 {out}, %bits;"
        ),
    }
}

fn store_f32_to_16(kind: Float16Kind, ptr: &str, value: &str) -> String {
    match kind {
        Float16Kind::F16 => format!(
            "\
    cvt.rn.f16.f32 %out_h, {value};
    st.global.b16 [{ptr}], %out_h;"
        ),
        Float16Kind::Bf16 => format!(
            "\
    cvt.rn.bf16.f32 %out_h, {value};
    st.global.b16 [{ptr}], %out_h;"
        ),
    }
}

fn half_reg_decl(kind: Float16Kind) -> &'static str {
    match kind {
        Float16Kind::F16 => ".reg .b16 %h, %out_h;",
        Float16Kind::Bf16 => ".reg .u16 %h; .reg .b16 %out_h;",
    }
}

fn ptx_header(kind: Float16Kind) -> &'static str {
    match kind {
        Float16Kind::F16 => ".version 7.0\n.target sm_53",
        Float16Kind::Bf16 => ".version 7.8\n.target sm_80",
    }
}

/// PTX for f16/bf16 masked or unmasked extrema. Values are decoded to f32,
/// reduced with the f32 NaN-poisoning combiner used by the f32 path, then
/// rounded back to the storage dtype for the partial/result buffer.
fn reduce_extreme_16_ptx(
    kernel_name: &str,
    kind: Float16Kind,
    extreme: ExtremeKind,
    with_mask: bool,
) -> String {
    let params = if with_mask {
        ".param .u64 data_ptr, .param .u64 mask_ptr, .param .u64 out_ptr, .param .u32 n"
    } else {
        ".param .u64 data_ptr, .param .u64 out_ptr, .param .u32 n"
    };
    let load_params = if with_mask {
        "    ld.param.u64 %dat, [data_ptr];
    ld.param.u64 %msk, [mask_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];"
    } else {
        "    ld.param.u64 %dat, [data_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];"
    };
    let mask_load = if with_mask {
        let decode_mask = load_16_to_f32(kind, "%maddr", "%m");
        format!(
            "\
    cvt.u64.u32 %moff, %idx;
    shl.b64 %moff, %moff, 1;
    add.u64 %maddr, %msk, %moff;
{decode_mask}
    setp.ne.f32 %p_valid, %m, 0f00000000;"
        )
    } else {
        "\
    setp.ne.f32 %p_valid, 0f3F800000, 0f00000000;"
            .to_string()
    };
    let load_data = load_16_to_f32(kind, "%daddr", "%d");
    let store_out = store_f32_to_16(kind, "%out", "%acc");
    let header = ptx_header(kind);
    let sentinel = extreme.sentinel_f32();
    let compare = extreme.compare_ptx();
    let half_reg_decl = half_reg_decl(kind);

    format!(
        "\
{header}
.address_size 64

.shared .align 4 .f32 sdata[256];

.visible .entry {kernel_name}({params}) {{
    .reg .u32 %my_tid, %bid, %bdim, %gdim, %n_reg, %idx, %stride, %half, %bits;
    .reg .u64 %dat, %msk, %out, %off, %moff, %daddr, %maddr, %saddr;
    .reg .f32 %acc, %d, %m, %sentinel, %val;
    {half_reg_decl}
    .reg .pred %p, %ptid, %p_valid, %val_nan, %acc_nan, %acc_ok, %cmp, %take;

{load_params}

    mov.u32 %my_tid, %tid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gdim, %nctaid.x;

    mad.lo.u32 %idx, %bid, %bdim, %my_tid;
    mul.lo.u32 %stride, %bdim, %gdim;
    mov.f32 %acc, {sentinel};
    mov.f32 %sentinel, {sentinel};

GRID_LOOP:
    setp.ge.u32 %p, %idx, %n_reg;
    @%p bra GRID_DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 1;
    add.u64 %daddr, %dat, %off;
{load_data}

{mask_load}
    selp.f32 %val, %d, %sentinel, %p_valid;

    setp.nan.f32 %val_nan, %val, %val;
    setp.nan.f32 %acc_nan, %acc, %acc;
    not.pred %acc_ok, %acc_nan;
    {compare}
    and.pred %cmp, %acc_ok, %cmp;
    or.pred %take, %val_nan, %cmp;
    @%take mov.f32 %acc, %val;
    add.u32 %idx, %idx, %stride;
    bra GRID_LOOP;

GRID_DONE:
    cvt.u64.u32 %off, %my_tid;
    shl.b64 %off, %off, 2;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    st.shared.f32 [%saddr], %acc;
    bar.sync 0;

    mov.u32 %half, 128;
TREE_LOOP:
    setp.lt.u32 %p, %half, 1;
    @%p bra TREE_DONE;

    setp.ge.u32 %ptid, %my_tid, %half;
    @%ptid bra TREE_SKIP;

    add.u32 %idx, %my_tid, %half;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    ld.shared.f32 %val, [%saddr];
    cvt.u64.u32 %off, %my_tid;
    shl.b64 %off, %off, 2;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    ld.shared.f32 %acc, [%saddr];
    setp.nan.f32 %val_nan, %val, %val;
    setp.nan.f32 %acc_nan, %acc, %acc;
    not.pred %acc_ok, %acc_nan;
    {compare}
    and.pred %cmp, %acc_ok, %cmp;
    or.pred %take, %val_nan, %cmp;
    @%take mov.f32 %acc, %val;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    st.shared.f32 [%saddr], %acc;

TREE_SKIP:
    bar.sync 0;
    shr.u32 %half, %half, 1;
    bra TREE_LOOP;

TREE_DONE:
    setp.ne.u32 %ptid, %my_tid, 0;
    @%ptid bra END;

    mov.u64 %saddr, sdata;
    ld.shared.f32 %acc, [%saddr];
    cvt.u64.u32 %off, %bid;
    shl.b64 %off, %off, 1;
    add.u64 %out, %out, %off;
{store_out}

END:
    ret;
}}
"
    )
}

fn masked_extreme_backward_16_ptx(
    count_kernel: &str,
    fill_kernel: &str,
    kind: Float16Kind,
) -> String {
    let header = ptx_header(kind);
    let load_target = load_16_to_f32(kind, "%ext", "%target");
    let load_count_val = load_16_to_f32(kind, "%addr_in", "%val");
    let load_count_mask = load_16_to_f32(kind, "%addr_mask", "%mask_v");
    let load_fill_target = load_16_to_f32(kind, "%ext", "%target");
    let load_fill_grad = load_16_to_f32(kind, "%grad", "%go");
    let load_fill_val = load_16_to_f32(kind, "%addr_in", "%val");
    let load_fill_mask = load_16_to_f32(kind, "%addr_mask", "%mask_v");
    let store_fill = store_f32_to_16(kind, "%addr_out", "%res");
    let half_reg_decl = half_reg_decl(kind);

    format!(
        "\
{header}
.address_size 64

.visible .entry {count_kernel}(
    .param .u64 input_ptr,
    .param .u64 mask_ptr,
    .param .u64 extreme_ptr,
    .param .u64 count_ptr,
    .param .u32 n
) {{
    .reg .u32 %idx, %stride, %bid, %bdim, %tid_r, %gdim, %n_reg, %old, %bits;
    .reg .u64 %in, %mask, %ext, %count, %off, %addr_in, %addr_mask;
    .reg .f32 %val, %mask_v, %target, %zero;
    {half_reg_decl}
    .reg .pred %p, %valid, %match, %take;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %mask, [mask_ptr];
    ld.param.u64 %ext, [extreme_ptr];
    ld.param.u64 %count, [count_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %tid_r, %tid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gdim, %nctaid.x;
    mad.lo.u32 %idx, %bid, %bdim, %tid_r;
    mul.lo.u32 %stride, %bdim, %gdim;
{load_target}
    mov.f32 %zero, 0f00000000;

COUNT_LOOP:
    setp.ge.u32 %p, %idx, %n_reg;
    @%p bra COUNT_DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 1;
    add.u64 %addr_in, %in, %off;
    add.u64 %addr_mask, %mask, %off;
{load_count_val}
{load_count_mask}
    setp.ne.f32 %valid, %mask_v, %zero;
    setp.eq.f32 %match, %val, %target;
    and.pred %take, %valid, %match;
    @%take atom.global.add.u32 %old, [%count], 1;

    add.u32 %idx, %idx, %stride;
    bra COUNT_LOOP;

COUNT_DONE:
    ret;
}}

.visible .entry {fill_kernel}(
    .param .u64 input_ptr,
    .param .u64 mask_ptr,
    .param .u64 extreme_ptr,
    .param .u64 grad_ptr,
    .param .u64 count_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {{
    .reg .u32 %idx, %stride, %bid, %bdim, %tid_r, %gdim, %n_reg, %count_u, %bits;
    .reg .u64 %in, %mask, %ext, %grad, %count, %out, %off, %addr_in, %addr_mask, %addr_out;
    .reg .f32 %val, %mask_v, %target, %go, %count_f, %zero, %scale, %res;
    {half_reg_decl}
    .reg .pred %p, %valid, %match, %take, %zero_count;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %mask, [mask_ptr];
    ld.param.u64 %ext, [extreme_ptr];
    ld.param.u64 %grad, [grad_ptr];
    ld.param.u64 %count, [count_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %tid_r, %tid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gdim, %nctaid.x;
    mad.lo.u32 %idx, %bid, %bdim, %tid_r;
    mul.lo.u32 %stride, %bdim, %gdim;

{load_fill_target}
{load_fill_grad}
    ld.global.u32 %count_u, [%count];
    mov.f32 %zero, 0f00000000;
    setp.eq.u32 %zero_count, %count_u, 0;
    cvt.rn.f32.u32 %count_f, %count_u;
    div.rn.f32 %scale, %go, %count_f;
    @%zero_count div.rn.f32 %scale, %zero, %zero;

FILL_LOOP:
    setp.ge.u32 %p, %idx, %n_reg;
    @%p bra FILL_DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 1;
    add.u64 %addr_in, %in, %off;
    add.u64 %addr_mask, %mask, %off;
    add.u64 %addr_out, %out, %off;
{load_fill_val}
{load_fill_mask}
    setp.ne.f32 %valid, %mask_v, %zero;
    setp.eq.f32 %match, %val, %target;
    and.pred %take, %valid, %match;
    mov.f32 %res, %zero;
    @%take mov.f32 %res, %scale;
    and.pred %take, %valid, %zero_count;
    @%take mov.f32 %res, %scale;
{store_fill}

    add.u32 %idx, %idx, %stride;
    bra FILL_LOOP;

FILL_DONE:
    ret;
}}
"
    )
}

fn launch_reduce_extreme_16(
    data: &CudaSlice<u16>,
    mask: Option<&CudaSlice<u16>>,
    device: &GpuDevice,
    kind: Float16Kind,
    extreme: ExtremeKind,
) -> GpuResult<CudaSlice<u16>> {
    if let Some(mask) = mask
        && data.len() != mask.len()
    {
        return Err(GpuError::LengthMismatch {
            a: data.len(),
            b: mask.len(),
        });
    }
    let n = data.len();
    if n > u32::MAX as usize {
        return Err(GpuError::ShapeMismatch {
            op: "masked_reduce_extreme_16",
            expected: vec![u32::MAX as usize],
            got: vec![n],
        });
    }
    if n == 0 {
        let bits = match (kind, extreme) {
            (Float16Kind::F16, ExtremeKind::Min) => half::f16::INFINITY.to_bits(),
            (Float16Kind::F16, ExtremeKind::Max) => half::f16::NEG_INFINITY.to_bits(),
            (Float16Kind::Bf16, ExtremeKind::Min) => half::bf16::INFINITY.to_bits(),
            (Float16Kind::Bf16, ExtremeKind::Max) => half::bf16::NEG_INFINITY.to_bits(),
        };
        return device.stream().clone_htod(&vec![bits]).map_err(Into::into);
    }

    let stream = device.stream();
    let ctx = device.context();
    let num_blocks = ((n as u32).saturating_add(BLOCK_SIZE - 1)) / BLOCK_SIZE;
    let num_blocks = num_blocks.min(1024);
    let kernel_name = match (kind, extreme, mask.is_some()) {
        (Float16Kind::F16, ExtremeKind::Min, true) => "masked_reduce_min_f16_kernel",
        (Float16Kind::F16, ExtremeKind::Max, true) => "masked_reduce_max_f16_kernel",
        (Float16Kind::Bf16, ExtremeKind::Min, true) => "masked_reduce_min_bf16_kernel",
        (Float16Kind::Bf16, ExtremeKind::Max, true) => "masked_reduce_max_bf16_kernel",
        (Float16Kind::F16, ExtremeKind::Min, false) => "reduce_min_f16_kernel",
        (Float16Kind::F16, ExtremeKind::Max, false) => "reduce_max_f16_kernel",
        (Float16Kind::Bf16, ExtremeKind::Min, false) => "reduce_min_bf16_kernel",
        (Float16Kind::Bf16, ExtremeKind::Max, false) => "reduce_max_bf16_kernel",
    };
    let ptx = reduce_extreme_16_ptx(kernel_name, kind, extreme, mask.is_some());
    let f = crate::module_cache::get_or_compile_owned(
        ctx,
        ptx,
        kernel_name.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: kernel_name,
        source: e,
    })?;
    let mut partials = stream.alloc_zeros::<u16>(num_blocks as usize)?;
    let cfg = LaunchConfig {
        grid_dim: (num_blocks.max(1), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u32 = n as u32;

    unsafe {
        let mut launch = stream.launch_builder(&f);
        launch.arg(data);
        if let Some(mask) = mask {
            launch.arg(mask);
        }
        launch.arg(&mut partials);
        launch.arg(&n_u32);
        launch.launch(cfg)?;
    }

    if num_blocks <= 1 {
        return Ok(partials);
    }
    launch_reduce_extreme_16(&partials, None, device, kind, extreme)
}

fn launch_masked_extreme_backward_16(
    input: &CudaSlice<u16>,
    mask: &CudaSlice<u16>,
    extreme: &CudaSlice<u16>,
    grad_output: &CudaSlice<u16>,
    device: &GpuDevice,
    kind: Float16Kind,
) -> GpuResult<CudaSlice<u16>> {
    let n = input.len();
    if mask.len() != n {
        return Err(GpuError::LengthMismatch {
            a: n,
            b: mask.len(),
        });
    }
    if extreme.len() != 1 {
        return Err(GpuError::ShapeMismatch {
            op: "masked_extreme_backward_16",
            expected: vec![1],
            got: vec![extreme.len()],
        });
    }
    if grad_output.len() != 1 {
        return Err(GpuError::ShapeMismatch {
            op: "masked_extreme_backward_16",
            expected: vec![1],
            got: vec![grad_output.len()],
        });
    }
    if n == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }

    let (count_kernel, fill_kernel) = match kind {
        Float16Kind::F16 => (
            "masked_extreme_backward_count_f16_kernel",
            "masked_extreme_backward_fill_f16_kernel",
        ),
        Float16Kind::Bf16 => (
            "masked_extreme_backward_count_bf16_kernel",
            "masked_extreme_backward_fill_bf16_kernel",
        ),
    };
    let ptx = masked_extreme_backward_16_ptx(count_kernel, fill_kernel, kind);
    let ctx = device.context();
    let stream = device.stream();
    let count_f = crate::module_cache::get_or_compile_owned(
        ctx,
        ptx.clone(),
        count_kernel.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: count_kernel,
        source: e,
    })?;
    let fill_f = crate::module_cache::get_or_compile_owned(
        ctx,
        ptx,
        fill_kernel.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: fill_kernel,
        source: e,
    })?;

    let mut count = stream.alloc_zeros::<u32>(1)?;
    let mut out = stream.alloc_zeros::<u16>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    unsafe {
        stream
            .launch_builder(&count_f)
            .arg(input)
            .arg(mask)
            .arg(extreme)
            .arg(&mut count)
            .arg(&n_u32)
            .launch(cfg)?;
        stream
            .launch_builder(&fill_f)
            .arg(input)
            .arg(mask)
            .arg(extreme)
            .arg(grad_output)
            .arg(&count)
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Masked f16 minimum reduction.
///
/// `data` and `mask` are resident f16 bit-pattern buffers. Nonzero mask
/// entries participate in the reduction; values are widened to f32 for
/// comparison/NaN poisoning and the scalar result is rounded back to f16.
pub fn masked_reduce_min_f16(
    data: &CudaSlice<u16>,
    mask: &CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_reduce_extreme_16(data, Some(mask), device, Float16Kind::F16, ExtremeKind::Min)
}

/// Masked f16 maximum reduction.
///
/// See [`masked_reduce_min_f16`] for the resident mask/value contract.
pub fn masked_reduce_max_f16(
    data: &CudaSlice<u16>,
    mask: &CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_reduce_extreme_16(data, Some(mask), device, Float16Kind::F16, ExtremeKind::Max)
}

/// Masked bf16 minimum reduction.
///
/// `data` and `mask` are resident bf16 bit-pattern buffers. Values are decoded
/// to f32 for comparison/NaN poisoning and the scalar result is rounded back to
/// bf16 on-device.
pub fn masked_reduce_min_bf16(
    data: &CudaSlice<u16>,
    mask: &CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_reduce_extreme_16(
        data,
        Some(mask),
        device,
        Float16Kind::Bf16,
        ExtremeKind::Min,
    )
}

/// Masked bf16 maximum reduction.
///
/// See [`masked_reduce_min_bf16`] for the resident mask/value contract.
pub fn masked_reduce_max_bf16(
    data: &CudaSlice<u16>,
    mask: &CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_reduce_extreme_16(
        data,
        Some(mask),
        device,
        Float16Kind::Bf16,
        ExtremeKind::Max,
    )
}

/// f16 masked-extremum VJP.
///
/// Counts valid f16 ties against `extreme` on-device, then writes
/// `grad_output / tie_count` at valid tie positions and zero elsewhere. If the
/// saved extreme is NaN and no equality ties exist, PyTorch's NaN-gradient edge
/// is reproduced by writing NaN to valid mask positions while masked-out slots
/// stay zero.
pub fn masked_extreme_backward_f16(
    input: &CudaSlice<u16>,
    mask: &CudaSlice<u16>,
    extreme: &CudaSlice<u16>,
    grad_output: &CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_masked_extreme_backward_16(input, mask, extreme, grad_output, device, Float16Kind::F16)
}

/// bf16 masked-extremum VJP.
///
/// Same contract as [`masked_extreme_backward_f16`], with bf16 decode/store and
/// f32 tie-count/scale math inside PTX.
pub fn masked_extreme_backward_bf16(
    input: &CudaSlice<u16>,
    mask: &CudaSlice<u16>,
    extreme: &CudaSlice<u16>,
    grad_output: &CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_masked_extreme_backward_16(input, mask, extreme, grad_output, device, Float16Kind::Bf16)
}

/// masked_fill for f32: `out[i] = mask[i]!=0 ? value : input[i]`.
pub fn masked_fill_f32(
    input: &CudaSlice<f32>,
    mask: &CudaSlice<u8>,
    value: f32,
    n: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<f32>> {
    launch_masked_fill(
        input,
        mask,
        value,
        n,
        d,
        MASKED_FILL_F32_PTX,
        "masked_fill_f32_kernel",
    )
}

/// masked_fill for f64.
pub fn masked_fill_f64(
    input: &CudaSlice<f64>,
    mask: &CudaSlice<u8>,
    value: f64,
    n: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<f64>> {
    launch_masked_fill(
        input,
        mask,
        value,
        n,
        d,
        MASKED_FILL_F64_PTX,
        "masked_fill_f64_kernel",
    )
}

/// masked_fill for f16 (u16 bits). `value` is the f32 fill, narrowed in-kernel.
pub fn masked_fill_f16(
    input: &CudaSlice<u16>,
    mask: &CudaSlice<u8>,
    value: f32,
    n: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_masked_fill(
        input,
        mask,
        value,
        n,
        d,
        MASKED_FILL_F16_PTX,
        "masked_fill_f16_kernel",
    )
}

/// masked_fill for bf16 (u16 bits). `value` is the f32 fill, narrowed in-kernel.
pub fn masked_fill_bf16(
    input: &CudaSlice<u16>,
    mask: &CudaSlice<u8>,
    value: f32,
    n: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_masked_fill(
        input,
        mask,
        value,
        n,
        d,
        MASKED_FILL_BF16_PTX,
        "masked_fill_bf16_kernel",
    )
}

/// masked_fill for i32. `value` is the fill, passed as raw 32-bit.
pub fn masked_fill_i32(
    input: &CudaSlice<i32>,
    mask: &CudaSlice<u8>,
    value: i32,
    n: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i32>> {
    launch_masked_fill(
        input,
        mask,
        value,
        n,
        d,
        MASKED_FILL_I32_PTX,
        "masked_fill_i32_kernel",
    )
}

/// masked_fill for i64. `value` is the fill, passed as raw 64-bit.
pub fn masked_fill_i64(
    input: &CudaSlice<i64>,
    mask: &CudaSlice<u8>,
    value: i64,
    n: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_masked_fill(
        input,
        mask,
        value,
        n,
        d,
        MASKED_FILL_I64_PTX,
        "masked_fill_i64_kernel",
    )
}

/// where (ternary select) for a 32-bit value dtype (f32 / i32).
pub fn where_32<T: DeviceRepr + ValidAsZeroBits>(
    cond: &CudaSlice<u8>,
    x: &CudaSlice<T>,
    y: &CudaSlice<T>,
    n: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<T>> {
    debug_assert_eq!(
        std::mem::size_of::<T>(),
        4,
        "where_32 requires a 4-byte element"
    );
    launch_where(cond, x, y, n, d, WHERE_32_PTX, "where_32_kernel")
}

/// where (ternary select) for a 64-bit value dtype (f64 / i64).
pub fn where_64<T: DeviceRepr + ValidAsZeroBits>(
    cond: &CudaSlice<u8>,
    x: &CudaSlice<T>,
    y: &CudaSlice<T>,
    n: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<T>> {
    debug_assert_eq!(
        std::mem::size_of::<T>(),
        8,
        "where_64 requires an 8-byte element"
    );
    launch_where(cond, x, y, n, d, WHERE_64_PTX, "where_64_kernel")
}

/// where (ternary select) for a 16-bit value dtype (f16 / bf16; pure bit
/// select, no decode).
pub fn where_16(
    cond: &CudaSlice<u8>,
    x: &CudaSlice<u16>,
    y: &CudaSlice<u16>,
    n: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_where(cond, x, y, n, d, WHERE_16_PTX, "where_16_kernel")
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
    let out = launch_compact(
        input,
        mask,
        len,
        d,
        ptx,
        "masked_select_compact_32".to_string(),
    )?;
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
    let out = launch_compact(
        input,
        mask,
        len,
        d,
        ptx,
        "masked_select_compact_64".to_string(),
    )?;
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
    let out = launch_compact(
        input,
        mask,
        len,
        d,
        ptx,
        "masked_select_compact_16".to_string(),
    )?;
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
    debug_assert_eq!(
        std::mem::size_of::<T>(),
        4,
        "masked_scatter_32 requires a 4-byte element"
    );
    let ptx = scatter_ptx("masked_scatter_32", 2, "b32", ".reg .b32 %val;");
    launch_scatter(
        grad,
        mask,
        out_numel,
        d,
        ptx,
        "masked_scatter_32".to_string(),
    )
}

/// masked_scatter for a 64-bit value dtype (f64 / i64).
pub fn masked_scatter_64<T: DeviceRepr + ValidAsZeroBits>(
    grad: &CudaSlice<T>,
    mask: &CudaSlice<u8>,
    out_numel: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<T>> {
    debug_assert_eq!(
        std::mem::size_of::<T>(),
        8,
        "masked_scatter_64 requires an 8-byte element"
    );
    let ptx = scatter_ptx("masked_scatter_64", 3, "b64", ".reg .b64 %val;");
    launch_scatter(
        grad,
        mask,
        out_numel,
        d,
        ptx,
        "masked_scatter_64".to_string(),
    )
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
    launch_scatter(
        grad,
        mask,
        out_numel,
        d,
        ptx,
        "masked_scatter_16".to_string(),
    )
}

/// masked_scatter FORWARD (#1662) for a 32-bit value dtype (f32 / i32):
/// `out[i] = mask[i]!=0 ? source[j++] : input[i]`. `n` is the logical numel of
/// `input`/`mask`; `source` holds at least the #true elements (caller-checked).
/// Result is a fresh `n`-element GPU-resident buffer.
pub fn masked_scatter_forward_32<T: DeviceRepr + ValidAsZeroBits>(
    input: &CudaSlice<T>,
    source: &CudaSlice<T>,
    mask: &CudaSlice<u8>,
    n: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<T>> {
    debug_assert_eq!(
        std::mem::size_of::<T>(),
        4,
        "masked_scatter_forward_32 requires a 4-byte element"
    );
    let ptx = scatter_forward_ptx("masked_scatter_forward_32", 2, "b32", ".reg .b32 %val;");
    launch_scatter_forward(
        input,
        source,
        mask,
        n,
        d,
        ptx,
        "masked_scatter_forward_32".to_string(),
    )
}

/// masked_scatter FORWARD for a 64-bit value dtype (f64 / i64).
pub fn masked_scatter_forward_64<T: DeviceRepr + ValidAsZeroBits>(
    input: &CudaSlice<T>,
    source: &CudaSlice<T>,
    mask: &CudaSlice<u8>,
    n: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<T>> {
    debug_assert_eq!(
        std::mem::size_of::<T>(),
        8,
        "masked_scatter_forward_64 requires an 8-byte element"
    );
    let ptx = scatter_forward_ptx("masked_scatter_forward_64", 3, "b64", ".reg .b64 %val;");
    launch_scatter_forward(
        input,
        source,
        mask,
        n,
        d,
        ptx,
        "masked_scatter_forward_64".to_string(),
    )
}

/// masked_scatter FORWARD for a 16-bit value dtype (f16 / bf16; pure 16-bit
/// bit copy, no decode or arithmetic).
pub fn masked_scatter_forward_16(
    input: &CudaSlice<u16>,
    source: &CudaSlice<u16>,
    mask: &CudaSlice<u8>,
    n: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    let ptx = scatter_forward_ptx("masked_scatter_forward_16", 1, "b16", ".reg .b16 %val;");
    launch_scatter_forward(
        input,
        source,
        mask,
        n,
        d,
        ptx,
        "masked_scatter_forward_16".to_string(),
    )
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
        let r = masked_fill_f32(&input, &mask, -9.0, 4, &d).unwrap();
        assert_eq!(
            d.stream().clone_dtoh(&r).unwrap(),
            vec![1.0f32, -9.0, 3.0, -9.0]
        );
    }

    #[test]
    fn where_32_selects() {
        let d = dev();
        let cond = d.stream().clone_htod(&vec![1u8, 0, 1, 0]).unwrap();
        let x = d
            .stream()
            .clone_htod(&vec![10.0f32, 20.0, 30.0, 40.0])
            .unwrap();
        let y = d
            .stream()
            .clone_htod(&vec![-1.0f32, -2.0, -3.0, -4.0])
            .unwrap();
        let r = where_32::<f32>(&cond, &x, &y, 4, &d).unwrap();
        assert_eq!(
            d.stream().clone_dtoh(&r).unwrap(),
            vec![10.0f32, -2.0, 30.0, -4.0]
        );
    }

    #[test]
    fn masked_select_32_compacts() {
        let d = dev();
        let input = d
            .stream()
            .clone_htod(&vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
            .unwrap();
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
    fn masked_scatter_forward_32_keeps_input_where_false() {
        let d = dev();
        // input [1,2,3,4]; mask [F,T,T,F]; source [-1,-2] ->
        // out = [1, -1, -2, 4]  (source consumed in flat order; false keeps in).
        let input = d.stream().clone_htod(&vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let mask = d.stream().clone_htod(&vec![0u8, 1, 1, 0]).unwrap();
        let source = d.stream().clone_htod(&vec![-1.0f32, -2.0]).unwrap();
        let out = masked_scatter_forward_32::<f32>(&input, &source, &mask, 4, &d).unwrap();
        assert_eq!(
            d.stream().clone_dtoh(&out).unwrap(),
            vec![1.0f32, -1.0, -2.0, 4.0]
        );
    }

    #[test]
    fn masked_scatter_forward_64_all_false_and_all_true() {
        let d = dev();
        let input = d.stream().clone_htod(&vec![1.0f64, 2.0, 3.0]).unwrap();
        // all-false: out == input (source unconsumed).
        let mask_f = d.stream().clone_htod(&vec![0u8, 0, 0]).unwrap();
        let src_f = d.stream().clone_htod(&vec![9.0f64]).unwrap();
        let out_f = masked_scatter_forward_64::<f64>(&input, &src_f, &mask_f, 3, &d).unwrap();
        assert_eq!(
            d.stream().clone_dtoh(&out_f).unwrap(),
            vec![1.0f64, 2.0, 3.0]
        );
        // all-true: out == source (full copy).
        let mask_t = d.stream().clone_htod(&vec![1u8, 1, 1]).unwrap();
        let src_t = d.stream().clone_htod(&vec![-7.0f64, -8.0, -9.0]).unwrap();
        let out_t = masked_scatter_forward_64::<f64>(&input, &src_t, &mask_t, 3, &d).unwrap();
        assert_eq!(
            d.stream().clone_dtoh(&out_t).unwrap(),
            vec![-7.0f64, -8.0, -9.0]
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
    fn masked_scatter_forward_16_keeps_input_and_source_bits() {
        let d = dev();
        let one = half::f16::from_f32(1.0).to_bits();
        let two = half::f16::from_f32(2.0).to_bits();
        let three = half::f16::from_f32(3.0).to_bits();
        let four = half::f16::from_f32(4.0).to_bits();
        let neg_one = half::f16::from_f32(-1.0).to_bits();
        let neg_two = half::f16::from_f32(-2.0).to_bits();
        let input = d.stream().clone_htod(&vec![one, two, three, four]).unwrap();
        let mask = d.stream().clone_htod(&vec![0u8, 1, 1, 0]).unwrap();
        let source = d.stream().clone_htod(&vec![neg_one, neg_two]).unwrap();

        let out = masked_scatter_forward_16(&input, &source, &mask, 4, &d).unwrap();

        assert_eq!(
            d.stream().clone_dtoh(&out).unwrap(),
            vec![one, neg_one, neg_two, four]
        );
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

    // ── #1545: predicate-mask kernels ────────────────────────────────────────

    #[test]
    fn isfinite_mask_f32_matches_ieee() {
        let d = dev();
        // [finite, NaN, finite, +inf, -inf, finite]
        let input = d
            .stream()
            .clone_htod(&vec![
                1.0f32,
                f32::NAN,
                3.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                -2.5,
            ])
            .unwrap();
        let mask = isfinite_mask_f32(&input, &d).unwrap();
        // finite -> 1, non-finite -> 0 (matches f32::is_finite)
        assert_eq!(
            d.stream().clone_dtoh(&mask).unwrap(),
            vec![1u8, 0, 1, 0, 0, 1]
        );
    }

    #[test]
    fn isfinite_mask_f64_matches_ieee() {
        let d = dev();
        let host = vec![1.0f64, f64::NAN, f64::INFINITY, 0.0, f64::NEG_INFINITY];
        let input = d.stream().clone_htod(&host).unwrap();
        let mask = isfinite_mask_f64(&input, &d).unwrap();
        let expected: Vec<u8> = host.iter().map(|v| u8::from(v.is_finite())).collect();
        assert_eq!(d.stream().clone_dtoh(&mask).unwrap(), expected);
    }

    #[test]
    fn isfinite_mask_f16_matches_ieee() {
        let d = dev();
        let host = vec![
            half::f16::from_f32(1.0).to_bits(),
            half::f16::NAN.to_bits(),
            half::f16::INFINITY.to_bits(),
            half::f16::NEG_INFINITY.to_bits(),
            half::f16::from_f32(-0.0).to_bits(),
            half::f16::from_f32(65504.0).to_bits(),
        ];
        let input = d.stream().clone_htod(&host).unwrap();
        let mask = isfinite_mask_f16(&input, &d).unwrap();
        assert_eq!(
            d.stream().clone_dtoh(&mask).unwrap(),
            vec![1u8, 0, 0, 0, 1, 1]
        );
    }

    #[test]
    fn isfinite_mask_bf16_matches_ieee() {
        let d = dev();
        let host = vec![
            half::bf16::from_f32(1.0).to_bits(),
            half::bf16::NAN.to_bits(),
            half::bf16::INFINITY.to_bits(),
            half::bf16::NEG_INFINITY.to_bits(),
            half::bf16::from_f32(-0.0).to_bits(),
            half::bf16::from_f32(3.3895314e38).to_bits(),
        ];
        let input = d.stream().clone_htod(&host).unwrap();
        let mask = isfinite_mask_bf16(&input, &d).unwrap();
        assert_eq!(
            d.stream().clone_dtoh(&mask).unwrap(),
            vec![1u8, 0, 0, 0, 1, 1]
        );
    }

    #[test]
    fn ne_scalar_mask_f32_marks_unequal() {
        let d = dev();
        // value 5.0; mask = (v != 5.0): equal -> 0, unequal -> 1.
        let input = d.stream().clone_htod(&vec![1.0f32, 5.0, 5.0, 2.0]).unwrap();
        let mask = ne_scalar_mask_f32(&input, 5.0, &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&mask).unwrap(), vec![1u8, 0, 0, 1]);
    }

    #[test]
    fn ne_scalar_mask_f64_nan_is_unequal() {
        let d = dev();
        // NaN != value is true (unordered) -> mask 1, matching the CPU walk.
        let input = d.stream().clone_htod(&vec![5.0f64, f64::NAN, 5.0]).unwrap();
        let mask = ne_scalar_mask_f64(&input, 5.0, &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&mask).unwrap(), vec![0u8, 1, 0]);
    }

    #[test]
    fn ne_scalar_mask_f16_nan_is_unequal() {
        let d = dev();
        let host = vec![
            half::f16::from_f32(5.0).to_bits(),
            half::f16::NAN.to_bits(),
            half::f16::from_f32(6.0).to_bits(),
            half::f16::from_f32(5.0).to_bits(),
        ];
        let input = d.stream().clone_htod(&host).unwrap();
        let mask = ne_scalar_mask_f16(&input, 5.0, &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&mask).unwrap(), vec![0u8, 1, 1, 0]);
    }

    #[test]
    fn ne_scalar_mask_bf16_nan_is_unequal() {
        let d = dev();
        let host = vec![
            half::bf16::from_f32(5.0).to_bits(),
            half::bf16::NAN.to_bits(),
            half::bf16::from_f32(6.0).to_bits(),
            half::bf16::from_f32(5.0).to_bits(),
        ];
        let input = d.stream().clone_htod(&host).unwrap();
        let mask = ne_scalar_mask_bf16(&input, 5.0, &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&mask).unwrap(), vec![0u8, 1, 1, 0]);
    }

    fn f16_bits(values: &[f32]) -> Vec<u16> {
        values
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect()
    }

    fn bf16_bits(values: &[f32]) -> Vec<u16> {
        values
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_bits())
            .collect()
    }

    fn f16_values(bits: &[u16]) -> Vec<f32> {
        bits.iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .collect()
    }

    fn bf16_values(bits: &[u16]) -> Vec<f32> {
        bits.iter()
            .map(|&bits| half::bf16::from_bits(bits).to_f32())
            .collect()
    }

    #[test]
    fn masked_reduce_extrema_f16_matches_mask_and_identities() {
        let d = dev();
        let data = d
            .stream()
            .clone_htod(&f16_bits(&[1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        let mask = d
            .stream()
            .clone_htod(&f16_bits(&[1.0, 0.0, 1.0, 0.0]))
            .unwrap();

        let min = masked_reduce_min_f16(&data, &mask, &d).unwrap();
        let max = masked_reduce_max_f16(&data, &mask, &d).unwrap();
        assert_eq!(f16_values(&d.stream().clone_dtoh(&min).unwrap()), vec![1.0]);
        assert_eq!(f16_values(&d.stream().clone_dtoh(&max).unwrap()), vec![3.0]);

        let all_false = d
            .stream()
            .clone_htod(&f16_bits(&[0.0, 0.0, 0.0, 0.0]))
            .unwrap();
        let min = masked_reduce_min_f16(&data, &all_false, &d).unwrap();
        let max = masked_reduce_max_f16(&data, &all_false, &d).unwrap();
        assert_eq!(
            f16_values(&d.stream().clone_dtoh(&min).unwrap()),
            vec![f32::INFINITY]
        );
        assert_eq!(
            f16_values(&d.stream().clone_dtoh(&max).unwrap()),
            vec![f32::NEG_INFINITY]
        );
    }

    #[test]
    fn masked_reduce_extrema_bf16_matches_mask_and_identities() {
        let d = dev();
        let data = d
            .stream()
            .clone_htod(&bf16_bits(&[1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        let mask = d
            .stream()
            .clone_htod(&bf16_bits(&[1.0, 0.0, 1.0, 0.0]))
            .unwrap();

        let min = masked_reduce_min_bf16(&data, &mask, &d).unwrap();
        let max = masked_reduce_max_bf16(&data, &mask, &d).unwrap();
        assert_eq!(
            bf16_values(&d.stream().clone_dtoh(&min).unwrap()),
            vec![1.0]
        );
        assert_eq!(
            bf16_values(&d.stream().clone_dtoh(&max).unwrap()),
            vec![3.0]
        );

        let all_false = d
            .stream()
            .clone_htod(&bf16_bits(&[0.0, 0.0, 0.0, 0.0]))
            .unwrap();
        let min = masked_reduce_min_bf16(&data, &all_false, &d).unwrap();
        let max = masked_reduce_max_bf16(&data, &all_false, &d).unwrap();
        assert_eq!(
            bf16_values(&d.stream().clone_dtoh(&min).unwrap()),
            vec![f32::INFINITY]
        );
        assert_eq!(
            bf16_values(&d.stream().clone_dtoh(&max).unwrap()),
            vec![f32::NEG_INFINITY]
        );
    }

    #[test]
    fn masked_reduce_extrema_f16_bf16_valid_nan_poison() {
        let d = dev();
        let f16_data = d
            .stream()
            .clone_htod(&vec![
                half::f16::from_f32(1.0).to_bits(),
                half::f16::NAN.to_bits(),
                half::f16::from_f32(2.0).to_bits(),
            ])
            .unwrap();
        let f16_mask = d.stream().clone_htod(&f16_bits(&[1.0, 1.0, 0.0])).unwrap();
        let f16_min = masked_reduce_min_f16(&f16_data, &f16_mask, &d).unwrap();
        assert!(f16_values(&d.stream().clone_dtoh(&f16_min).unwrap())[0].is_nan());

        let bf16_data = d
            .stream()
            .clone_htod(&vec![
                half::bf16::from_f32(1.0).to_bits(),
                half::bf16::NAN.to_bits(),
                half::bf16::from_f32(2.0).to_bits(),
            ])
            .unwrap();
        let bf16_mask = d.stream().clone_htod(&bf16_bits(&[1.0, 1.0, 0.0])).unwrap();
        let bf16_max = masked_reduce_max_bf16(&bf16_data, &bf16_mask, &d).unwrap();
        assert!(bf16_values(&d.stream().clone_dtoh(&bf16_max).unwrap())[0].is_nan());
    }

    #[test]
    fn masked_extreme_backward_f16_bf16_splits_valid_ties() {
        let d = dev();
        let f16_input = d
            .stream()
            .clone_htod(&f16_bits(&[5.0, 5.0, 1.0, 5.0]))
            .unwrap();
        let f16_mask = d
            .stream()
            .clone_htod(&f16_bits(&[1.0, 1.0, 1.0, 0.0]))
            .unwrap();
        let f16_extreme = d.stream().clone_htod(&f16_bits(&[5.0])).unwrap();
        let f16_go = d.stream().clone_htod(&f16_bits(&[1.0])).unwrap();
        let f16_grad =
            masked_extreme_backward_f16(&f16_input, &f16_mask, &f16_extreme, &f16_go, &d).unwrap();
        assert_eq!(
            f16_values(&d.stream().clone_dtoh(&f16_grad).unwrap()),
            vec![0.5, 0.5, 0.0, 0.0]
        );

        let bf16_input = d
            .stream()
            .clone_htod(&bf16_bits(&[5.0, 5.0, 1.0, 5.0]))
            .unwrap();
        let bf16_mask = d
            .stream()
            .clone_htod(&bf16_bits(&[1.0, 1.0, 1.0, 0.0]))
            .unwrap();
        let bf16_extreme = d.stream().clone_htod(&bf16_bits(&[5.0])).unwrap();
        let bf16_go = d.stream().clone_htod(&bf16_bits(&[1.0])).unwrap();
        let bf16_grad =
            masked_extreme_backward_bf16(&bf16_input, &bf16_mask, &bf16_extreme, &bf16_go, &d)
                .unwrap();
        assert_eq!(
            bf16_values(&d.stream().clone_dtoh(&bf16_grad).unwrap()),
            vec![0.5, 0.5, 0.0, 0.0]
        );
    }

    #[test]
    fn masked_extreme_backward_f16_bf16_valid_nan_matches_mask() {
        let d = dev();
        let f16_input = d
            .stream()
            .clone_htod(&vec![
                half::f16::from_f32(1.0).to_bits(),
                half::f16::NAN.to_bits(),
                half::f16::from_f32(2.0).to_bits(),
            ])
            .unwrap();
        let f16_mask = d.stream().clone_htod(&f16_bits(&[1.0, 1.0, 0.0])).unwrap();
        let f16_extreme = d
            .stream()
            .clone_htod(&vec![half::f16::NAN.to_bits()])
            .unwrap();
        let f16_go = d.stream().clone_htod(&f16_bits(&[3.0])).unwrap();
        let f16_grad =
            masked_extreme_backward_f16(&f16_input, &f16_mask, &f16_extreme, &f16_go, &d).unwrap();
        let f16_grad = f16_values(&d.stream().clone_dtoh(&f16_grad).unwrap());
        assert!(f16_grad[0].is_nan());
        assert!(f16_grad[1].is_nan());
        assert_eq!(f16_grad[2], 0.0);

        let bf16_input = d
            .stream()
            .clone_htod(&vec![
                half::bf16::from_f32(1.0).to_bits(),
                half::bf16::NAN.to_bits(),
                half::bf16::from_f32(2.0).to_bits(),
            ])
            .unwrap();
        let bf16_mask = d.stream().clone_htod(&bf16_bits(&[1.0, 1.0, 0.0])).unwrap();
        let bf16_extreme = d
            .stream()
            .clone_htod(&vec![half::bf16::NAN.to_bits()])
            .unwrap();
        let bf16_go = d.stream().clone_htod(&bf16_bits(&[3.0])).unwrap();
        let bf16_grad =
            masked_extreme_backward_bf16(&bf16_input, &bf16_mask, &bf16_extreme, &bf16_go, &d)
                .unwrap();
        let bf16_grad = bf16_values(&d.stream().clone_dtoh(&bf16_grad).unwrap());
        assert!(bf16_grad[0].is_nan());
        assert!(bf16_grad[1].is_nan());
        assert_eq!(bf16_grad[2], 0.0);
    }
}
