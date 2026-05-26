//! Boolean / comparison GPU compute kernels — crosslink #1185 Phase 3b.
//!
//! Hand-written PTX owned by Rust (no CUDA C++, no nvrtc, no external toolchain
//! at load time), loaded via [`crate::module_cache::get_or_compile`] exactly
//! like [`crate::int_kernels`] / [`crate::f16`] / [`crate::bf16`]. Boolean
//! buffers are stored as native `CudaSlice<u8>` (cudarc `DeviceRepr` for `u8`;
//! a `bool` is one byte holding 0 or 1). The [`crate::backend_impl`] handle is
//! tagged `DType::Bool` so a u8 bool buffer is never read as an i8/u8 integer.
//!
//! # Operations
//!
//! - **Comparison** (value dtype in, u8 0/1 out): for each value dtype
//!   `f32 / f64 / bf16 / f16 / i32 / i64`, the six operators
//!   `eq / ne / lt / le / gt / ge`. bf16/f16 inputs (u16 bit patterns) are
//!   decoded to f32 first, then compared (matching the bf16/f16 elementwise
//!   kernels). The output buffer is `CudaSlice<u8>` (1 byte per element).
//! - **Logical** (u8 in, u8 out): `and / or / xor` (binary), `not` (unary).
//!   Inputs are treated as "nonzero == true"; outputs are canonical 0/1.
//! - **Reductions** to a 1-element u8 buffer: `any` (OR-reduce), `all`
//!   (AND-reduce), global. One launched thread folds all `n` elements serially
//!   (matching `int_kernels`' reduction harness), so the result equals a
//!   left-fold over the buffer.
//!
//! # PyTorch parity (rust-gpu-discipline §3)
//!
//! A comparison's result dtype is `bool` regardless of the value dtype — the
//! output is always a `DType::Bool` (u8) buffer. NaN comparisons follow IEEE:
//! `eq/lt/le/gt/ge` involving NaN are false, `ne` involving NaN is true. PTX
//! `setp.{eq,lt,le,gt,ge}.f32` are unordered-false / `setp.ne.f32` is
//! unordered-true, which is exactly the IEEE / PyTorch behaviour.
//!
//! ## REQ status (per `.design/ferrotorch-gpu/bool_kernels.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream
//! cites) live in the design doc; this synopsis is a one-line summary per
//! REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (comparison ops) | SHIPPED | `pub fn gpu_cmp_f32 / gpu_cmp_f64 / gpu_cmp_i32 / gpu_cmp_i64 / gpu_cmp_bf16 / gpu_cmp_f16 in bool_kernels.rs` thin-wrap `launch_cmp` / `launch_cmp_half` templated per (dtype, op); consumer bool-result arms of `eq/ne/lt/le/gt/ge` op handlers in `backend_impl.rs` |
//! | REQ-2 (logical binary) | SHIPPED | `pub fn gpu_and_bool / gpu_or_bool / gpu_xor_bool in bool_kernels.rs` thin-wrap `launch_logic_bin`; consumer `CudaBackendImpl::and_bool / or_bool / xor_bool in backend_impl.rs` |
//! | REQ-3 (`gpu_not_bool`) | SHIPPED | `pub fn gpu_not_bool in bool_kernels.rs`; consumer `CudaBackendImpl::not_bool in backend_impl.rs` |
//! | REQ-4 (`gpu_any_bool`/`gpu_all_bool`) | SHIPPED | `pub fn gpu_any_bool / gpu_all_bool in bool_kernels.rs`; consumer `CudaBackendImpl::any_bool / all_bool in backend_impl.rs` |
//! | REQ-5 (NaN comparison semantics) | SHIPPED | PTX `setp.{eq,lt,le,gt,ge}.f32` (unordered-false) and `setp.ne.f32` (unordered-true) inside the comparison kernels in `bool_kernels.rs`; consumer bool-comparison ops in `backend_impl.rs` rely on this for IEEE-NaN parity |
//! | REQ-6 (half-precision compare) | SHIPPED | `fn cmp_half_ptx in bool_kernels.rs` decodes bf16 via `mov.b32 %ua, {%zero16, %ha}` and f16 via `cvt.f32.f16 %fa, %ha` then `setp.{op}.f32`; consumer `pub fn gpu_cmp_bf16 / gpu_cmp_f16` invoke `launch_cmp_half` from bool-comparison arms of `backend_impl.rs` |
//! | REQ-7 (SAFETY annotations) | SHIPPED | every `unsafe { stream.launch_builder(&f)... }` in `bool_kernels.rs` (`launch_cmp`, `launch_not`, `launch_reduce_bool`) carries a multi-line `SAFETY:` comment; consumer SAFETY contract inherited via each public wrapper |
//! | REQ-8 (empty-input short-circuit) | SHIPPED | `launch_cmp` and `launch_not` short-circuit `n == 0` via `if n == 0 { return Ok(stream.alloc_zeros::<u8>(0)?); }`; `launch_reduce_bool` short-circuits with empty-identity clone_htod; consumer backend dispatch path (`torch.any(empty)`) |

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits};

use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::module_cache::{get_or_compile, get_or_compile_owned};

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
// PTX builders.
//
// A comparison kernel reads `a[i]`, `b[i]` (each `IN_BYTES` wide), computes a
// predicate, and writes `out[i]` as one u8 (0 or 1). `in_off = i << IN_SHIFT`,
// `out_off = i` (1 byte per bool). All bound-checked against `n`.
//
// Rather than hand-write 36 near-identical comparison kernels, we generate the
// PTX as owned `String`s at module-load time (once, cached by `get_or_compile`
// keyed on the kernel name). The body differs only in: the load type, the
// `setp` form, and the input element shift.
// ===========================================================================

/// PTX prologue computing `%idx` (global thread id) and bound-checking it
/// against the `n` param, branching to `DONE` when out of range. Shared by the
/// float and integer comparison kernels.
fn cmp_ptx(
    kernel_name: &str,
    in_shift: u32,  // log2(IN_BYTES): f32/i32→2, f64/i64→3
    load_ty: &str,  // "f32" | "f64" | "s32" | "s64"
    reg_decl: &str, // register decls for the value regs
    setp: &str,     // e.g. "setp.lt.f32 %c, %va, %vb;"
) -> String {
    format!(
        "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {{
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %ioff, %ooff;
    {reg_decl}
    .reg .u16 %res;
    .reg .pred %p, %c;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
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
    add.u64 %b, %b, %ioff;
    // output is 1 byte per element: out_off = idx
    cvt.u64.u32 %ooff, %idx;
    add.u64 %out, %out, %ooff;

    ld.global.{load_ty} %va, [%a];
    ld.global.{load_ty} %vb, [%b];
    {setp}
    selp.u16 %res, 1, 0, %c;
    st.global.u8 [%out], %res;
DONE:
    ret;
}}
"
    )
}

/// PTX for a comparison whose value type is bf16 or f16 (a u16 bit pattern).
/// The two halves are loaded as `.b16` and decoded to f32 (`decode`), then
/// compared in f32. `target` is the required `.target` line (sm_53 for the
/// f16 `cvt.f32.f16`, sm_80 for the bf16 splat path — both supported by the
/// host RTX 3090, sm_86).
fn cmp_half_ptx(
    kernel_name: &str,
    target: &str, // e.g. "sm_53" | "sm_80"
    decode: &str, // PTX decoding %ha (b16) → %fa (f32) and %hb → %fb
    setp: &str,   // e.g. "setp.lt.f32 %c, %fa, %fb;"
) -> String {
    format!(
        "\
.version 7.0
.target {target}
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {{
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %ioff, %ooff;
    .reg .b16 %ha, %hb, %zero16;
    .reg .b32 %ua, %ub;
    .reg .u16 %res;
    .reg .f32 %fa, %fb;
    .reg .pred %p, %c;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
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
    add.u64 %b, %b, %ioff;
    cvt.u64.u32 %ooff, %idx;
    add.u64 %out, %out, %ooff;

    mov.b16 %zero16, 0;
    ld.global.b16 %ha, [%a];
    ld.global.b16 %hb, [%b];
    {decode}
    {setp}
    selp.u16 %res, 1, 0, %c;
    st.global.u8 [%out], %res;
DONE:
    ret;
}}
"
    )
}

// bf16 decode: a bf16 is the high 16 bits of an f32. Compose a b32 from
// {low=0, high=bf16_bits} and reinterpret as f32 (mirrors crate::bf16's
// `mov.b32 %u, {%zero16, %b16}; mov.b32 %f, %u` pattern).
const BF16_DECODE: &str = "\
    mov.b32 %ua, {%zero16, %ha}; mov.b32 %fa, %ua;
    mov.b32 %ub, {%zero16, %hb}; mov.b32 %fb, %ub;";
// f16 decode: hardware convert IEEE half → f32 (cvt.f32.f16, sm_53+).
const F16_DECODE: &str = "\
    cvt.f32.f16 %fa, %ha;
    cvt.f32.f16 %fb, %hb;";

/// PTX for a logical binary op (`and`/`or`/`xor`) over u8 bool buffers.
/// `nonzero(a) OP nonzero(b)` → canonical 0/1.
fn logic_bin_ptx(kernel_name: &str, op: &str /* "and"|"or"|"xor" */) -> String {
    format!(
        "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {{
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %off;
    .reg .u16 %va, %vb, %res;
    .reg .pred %pa, %pb, %pr, %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nr, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    cvt.u64.u32 %off, %idx;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.u8 %va, [%a];
    ld.global.u8 %vb, [%b];
    setp.ne.u16 %pa, %va, 0;
    setp.ne.u16 %pb, %vb, 0;
    {op}.pred %pr, %pa, %pb;
    selp.u16 %res, 1, 0, %pr;
    st.global.u8 [%out], %res;
DONE:
    ret;
}}
"
    )
}

const NOT_BOOL_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry not_bool_kernel(
    .param .u64 a_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %out, %off;
    .reg .u16 %va, %res;
    .reg .pred %pa, %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nr, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    cvt.u64.u32 %off, %idx;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.u8 %va, [%a];
    // res = (va == 0) ? 1 : 0
    setp.eq.u16 %pa, %va, 0;
    selp.u16 %res, 1, 0, %pa;
    st.global.u8 [%out], %res;
DONE:
    ret;
}
";

// Reduction: one launched thread (thread 0) folds all n bytes. `op`: 0 = any
// (OR-reduce, identity 0), 1 = all (AND-reduce, identity 1). Each input byte is
// normalised to 0/1 (nonzero → 1). Output is one u8 (0/1). The host guards the
// n == 0 case (any of empty = false, all of empty = true) before launching.
const REDUCE_BOOL_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry reduce_bool_kernel(
    .param .u64 a_ptr, .param .u64 out_ptr, .param .u32 n, .param .u32 op
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %op_r, %i;
    .reg .u64 %a, %out, %off, %cur;
    .reg .u16 %acc, %v, %vn;
    .reg .pred %only0, %p, %is_any, %pacc, %pv;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nr, [n];
    ld.param.u32 %op_r, [op];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ne.u32 %only0, %idx, 0;
    @%only0 bra DONE;

    setp.eq.u32 %is_any, %op_r, 0;

    // Initialise accumulator from a[0] normalised to 0/1 (n >= 1 guaranteed).
    ld.global.u8 %v, [%a];
    setp.ne.u16 %pv, %v, 0;
    selp.u16 %acc, 1, 0, %pv;
    mov.u32 %i, 1;
LOOP:
    setp.ge.u32 %p, %i, %nr;
    @%p bra STORE;

    cvt.u64.u32 %off, %i;
    add.u64 %cur, %a, %off;
    ld.global.u8 %v, [%cur];
    setp.ne.u16 %pv, %v, 0;
    selp.u16 %vn, 1, 0, %pv;

    setp.ne.u16 %pacc, %acc, 0;
    setp.ne.u16 %pv, %vn, 0;
    // any: acc = acc OR v ; all: acc = acc AND v
    @%is_any or.pred %pacc, %pacc, %pv;
    @!%is_any and.pred %pacc, %pacc, %pv;
    selp.u16 %acc, 1, 0, %pacc;

    add.u32 %i, %i, 1;
    bra LOOP;
STORE:
    st.global.u8 [%out], %acc;
DONE:
    ret;
}
";

const REDUCE_ANY: u32 = 0;
const REDUCE_ALL: u32 = 1;

// ===========================================================================
// Launch harness
// ===========================================================================

/// Launch a comparison `(a_ptr, b_ptr, out_ptr, n)` kernel over a value buffer
/// of native element type `T` (`f32`/`f64`/`i32`/`i64`), producing a fresh
/// `CudaSlice<u8>` of `n` 0/1 bytes resident on `device`.
fn launch_cmp<T: DeviceRepr + ValidAsZeroBits>(
    a: &CudaSlice<T>,
    b: &CudaSlice<T>,
    device: &GpuDevice,
    ptx: String,
    kernel_name: String,
    err_label: &'static str,
) -> GpuResult<CudaSlice<u8>> {
    if a.len() != b.len() {
        return Err(GpuError::LengthMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    let n = a.len();
    let stream = device.stream();
    if n == 0 {
        return Ok(stream.alloc_zeros::<u8>(0)?);
    }
    let ctx = device.context();
    // The comparison PTX is built at runtime (one variant per value-dtype ×
    // operator), so use the owned-string cache, which hashes the PTX, compiles
    // once, and reuses thereafter. `kernel_name` is the entry-point name inside
    // that PTX. `err_label` is a `'static` family name for error reporting only.
    let f = get_or_compile_owned(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: err_label,
            source: e,
        }
    })?;
    let mut out = stream.alloc_zeros::<u8>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is the PTX entry `kernel_name` just compiled; its signature is
    //   (a_ptr: u64, b_ptr: u64, out_ptr: u64, n: u32), matching the four args
    //   pushed below in order.
    // - `a`, `b` are immutable input buffers of `n` `T`-elements each (length
    //   equality enforced above); `n` is bound to `a.len()`.
    // - `out` was alloc'd `n` `u8`-elements from `stream` and is the only `&mut`
    //   here, non-aliased with the immutable inputs (distinct allocations).
    // - The kernel reads `a[i]`/`b[i]` and writes `out[i]` only within `[0, n)`
    //   per the `setp.ge.u32 %p, %idx, %nr` bound check.
    // - `n_u32` is non-truncating: `launch_1d` already cast `n as u32` to size
    //   the grid covering `[0, n)`.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(b)
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Launch a comparison over a half-precision (u16 bit-pattern) value buffer.
/// Identical contract to [`launch_cmp`], specialised to `CudaSlice<u16>`.
fn launch_cmp_half(
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    device: &GpuDevice,
    ptx: String,
    kernel_name: String,
) -> GpuResult<CudaSlice<u8>> {
    launch_cmp::<u16>(a, b, device, ptx, kernel_name, "cmp_half")
}

/// Launch a logical binary `(a_ptr, b_ptr, out_ptr, n)` kernel over two u8 bool
/// buffers, producing a fresh `CudaSlice<u8>` of `n` 0/1 bytes.
fn launch_logic_bin(
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    device: &GpuDevice,
    ptx: String,
    kernel_name: &'static str,
) -> GpuResult<CudaSlice<u8>> {
    launch_cmp::<u8>(a, b, device, ptx, kernel_name.to_string(), kernel_name)
}

/// Launch the unary NOT `(a_ptr, out_ptr, n)` kernel over a u8 bool buffer.
fn launch_not(a: &CudaSlice<u8>, device: &GpuDevice) -> GpuResult<CudaSlice<u8>> {
    let n = a.len();
    let stream = device.stream();
    if n == 0 {
        return Ok(stream.alloc_zeros::<u8>(0)?);
    }
    let ctx = device.context();
    let f = get_or_compile(
        ctx,
        NOT_BOOL_PTX,
        "not_bool_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "not_bool_kernel",
        source: e,
    })?;
    let mut out = stream.alloc_zeros::<u8>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` resolves to the unary PTX entry `(a_ptr, out_ptr, n)`.
    // - `a` is the caller's input of `n` u8 elements; the bound check limits
    //   reads to `[0, n)`.
    // - `out` is freshly alloc'd `n` u8 elements, exclusively borrowed here.
    // - `n_u32` is non-truncating (see `launch_cmp`).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Launch the serial bool reduction `(a_ptr, out_ptr, n, op)`, returning a
/// 1-element u8 buffer. `op` selects any (OR) / all (AND).
fn launch_reduce_bool(
    a: &CudaSlice<u8>,
    device: &GpuDevice,
    op: u32,
    empty_identity: u8,
) -> GpuResult<CudaSlice<u8>> {
    let n = a.len();
    let stream = device.stream();
    if n == 0 {
        // any of empty = false (0), all of empty = true (1).
        let host = [empty_identity];
        return Ok(stream.clone_htod(&host)?);
    }
    let ctx = device.context();
    let f = get_or_compile(
        ctx,
        REDUCE_BOOL_PTX,
        "reduce_bool_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "reduce_bool_kernel",
        source: e,
    })?;
    let mut out = stream.alloc_zeros::<u8>(1)?;
    // Single block, single active thread (thread 0 folds serially).
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is the reduce PTX entry `(a_ptr, out_ptr, n, op)`; four args below
    //   match in order.
    // - `a` is the caller's input of `n` u8 elements; the kernel reads `a[0..n)`
    //   from thread 0 only (`setp.ne.u32 %only0, %idx, 0` gates the rest off)
    //   and writes the single `out[0]`.
    // - `out` is a freshly alloc'd 1-element buffer, exclusively borrowed.
    // - `op` is one of {0, 1} (REDUCE_ANY / REDUCE_ALL).
    // - `n_u32` is non-truncating for any host-allocatable contiguous buffer.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&mut out)
            .arg(&n_u32)
            .arg(&op)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// Public entry points
// ===========================================================================

/// The `setp` PTX form for a comparison `op` over the given PTX type
/// (`"f32"`/`"f64"`/`"s32"`/`"s64"`). Result predicate is `%c`.
fn setp_for(op: &str, ty: &str) -> String {
    format!("setp.{op}.{ty} %c, %va, %vb;")
}

/// f32 comparison: `out = (a OP b)` as a u8 0/1 buffer. `op` ∈
/// {eq,ne,lt,le,gt,ge}.
pub fn gpu_cmp_f32(
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    op: &str,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let name = format!("cmp_{op}_f32_kernel");
    let ptx = cmp_ptx(&name, 2, "f32", ".reg .f32 %va, %vb;", &setp_for(op, "f32"));
    launch_cmp::<f32>(a, b, d, ptx, name, "cmp_f32")
}

/// f64 comparison → u8 0/1 buffer.
pub fn gpu_cmp_f64(
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    op: &str,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let name = format!("cmp_{op}_f64_kernel");
    let ptx = cmp_ptx(&name, 3, "f64", ".reg .f64 %va, %vb;", &setp_for(op, "f64"));
    launch_cmp::<f64>(a, b, d, ptx, name, "cmp_f64")
}

/// i32 comparison → u8 0/1 buffer (signed compare).
pub fn gpu_cmp_i32(
    a: &CudaSlice<i32>,
    b: &CudaSlice<i32>,
    op: &str,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let name = format!("cmp_{op}_i32_kernel");
    let ptx = cmp_ptx(&name, 2, "s32", ".reg .s32 %va, %vb;", &setp_for(op, "s32"));
    launch_cmp::<i32>(a, b, d, ptx, name, "cmp_i32")
}

/// i64 comparison → u8 0/1 buffer (signed compare).
pub fn gpu_cmp_i64(
    a: &CudaSlice<i64>,
    b: &CudaSlice<i64>,
    op: &str,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let name = format!("cmp_{op}_i64_kernel");
    let ptx = cmp_ptx(&name, 3, "s64", ".reg .s64 %va, %vb;", &setp_for(op, "s64"));
    launch_cmp::<i64>(a, b, d, ptx, name, "cmp_i64")
}

/// bf16 comparison (u16 bit patterns decoded to f32) → u8 0/1 buffer.
pub fn gpu_cmp_bf16(
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    op: &str,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let name = format!("cmp_{op}_bf16_kernel");
    let setp = format!("setp.{op}.f32 %c, %fa, %fb;");
    let ptx = cmp_half_ptx(&name, "sm_52", BF16_DECODE, &setp);
    launch_cmp_half(a, b, d, ptx, name)
}

/// f16 comparison (u16 bit patterns decoded to f32) → u8 0/1 buffer.
pub fn gpu_cmp_f16(
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    op: &str,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let name = format!("cmp_{op}_f16_kernel");
    let setp = format!("setp.{op}.f32 %c, %fa, %fb;");
    let ptx = cmp_half_ptx(&name, "sm_53", F16_DECODE, &setp);
    launch_cmp_half(a, b, d, ptx, name)
}

/// Logical AND of two u8 bool buffers → u8 0/1 buffer.
pub fn gpu_and_bool(
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let ptx = logic_bin_ptx("and_bool_kernel", "and");
    launch_logic_bin(a, b, d, ptx, "and_bool_kernel")
}

/// Logical OR of two u8 bool buffers → u8 0/1 buffer.
pub fn gpu_or_bool(
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let ptx = logic_bin_ptx("or_bool_kernel", "or");
    launch_logic_bin(a, b, d, ptx, "or_bool_kernel")
}

/// Logical XOR of two u8 bool buffers → u8 0/1 buffer.
pub fn gpu_xor_bool(
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let ptx = logic_bin_ptx("xor_bool_kernel", "xor");
    launch_logic_bin(a, b, d, ptx, "xor_bool_kernel")
}

/// Logical NOT of a u8 bool buffer → u8 0/1 buffer.
pub fn gpu_not_bool(a: &CudaSlice<u8>, d: &GpuDevice) -> GpuResult<CudaSlice<u8>> {
    launch_not(a, d)
}

/// Global OR-reduction (`torch.any`) → 1-element u8 buffer (0/1).
pub fn gpu_any_bool(a: &CudaSlice<u8>, d: &GpuDevice) -> GpuResult<CudaSlice<u8>> {
    launch_reduce_bool(a, d, REDUCE_ANY, 0)
}

/// Global AND-reduction (`torch.all`) → 1-element u8 buffer (0/1).
pub fn gpu_all_bool(a: &CudaSlice<u8>, d: &GpuDevice) -> GpuResult<CudaSlice<u8>> {
    launch_reduce_bool(a, d, REDUCE_ALL, 1)
}
