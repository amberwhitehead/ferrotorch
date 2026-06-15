//! CUDA cumulative-scan kernels that do not fit the legacy f32/f64 block in
//! [`crate::kernels`].
//!
//! The important contracts here mirror PyTorch CUDA behavior for half-family
//! tensors: outputs stay in the input dtype, extrema indices are real int64
//! tensors, and backward kernels never stage through the CPU.

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::transfer::{alloc_zeros_bf16, alloc_zeros_f32, alloc_zeros_f64};

const BLOCK_SIZE: u32 = 256;

#[derive(Clone, Copy)]
enum HalfKind {
    F16,
    BF16,
}

impl HalfKind {
    fn ptx_version(self) -> &'static str {
        match self {
            Self::F16 => "7.0",
            Self::BF16 => "7.8",
        }
    }

    fn target(self) -> &'static str {
        match self {
            Self::F16 => "sm_53",
            Self::BF16 => "sm_80",
        }
    }

    fn load_raw_to_val(self) -> &'static str {
        match self {
            Self::F16 => "    ld.global.b16 %raw, [%addr];\n    cvt.f32.f16 %val, %raw;",
            Self::BF16 => {
                "    ld.global.b16 %raw, [%addr];\n    mov.b16 %zero16, 0;\n    mov.b32 %bits, {%zero16, %raw};\n    mov.b32 %val, %bits;"
            }
        }
    }

    fn raw_to_named(self, raw: &str, out: &str) -> String {
        match self {
            Self::F16 => format!("    cvt.f32.f16 {out}, {raw};"),
            Self::BF16 => format!(
                "    mov.b16 %zero16, 0;\n    mov.b32 %bits, {{%zero16, {raw}}};\n    mov.b32 {out}, %bits;"
            ),
        }
    }

    fn narrow_acc_store_and_reload(self) -> &'static str {
        match self {
            Self::F16 => {
                "    cvt.rn.f16.f32 %acc_h, %acc;\n    st.global.b16 [%addr], %acc_h;\n    cvt.f32.f16 %acc, %acc_h;"
            }
            Self::BF16 => {
                "    cvt.rn.bf16.f32 %acc_h, %acc;\n    st.global.b16 [%addr], %acc_h;\n    mov.b16 %zero16, 0;\n    mov.b32 %bits, {%zero16, %acc_h};\n    mov.b32 %acc, %bits;"
            }
        }
    }

    fn narrow_result_store(self) -> &'static str {
        match self {
            Self::F16 => "    cvt.rn.f16.f32 %acc_h, %acc;\n    st.global.b16 [%addr], %acc_h;",
            Self::BF16 => "    cvt.rn.bf16.f32 %acc_h, %acc;\n    st.global.b16 [%addr], %acc_h;",
        }
    }
}

fn replace_all(mut template: String, replacements: &[(&str, String)]) -> String {
    for (needle, value) in replacements {
        template = template.replace(needle, value);
    }
    template
}

fn launch_cfg(n: usize) -> GpuResult<LaunchConfig> {
    let n_u32 = u32::try_from(n).map_err(|_| GpuError::InvalidState {
        message: format!("cumulative launch has {n} threads, exceeds u32::MAX"),
    })?;
    let grid = n_u32.saturating_add(BLOCK_SIZE - 1) / BLOCK_SIZE;
    Ok(LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn checked_dims(
    op: &str,
    outer: usize,
    dim_size: usize,
    inner: usize,
) -> GpuResult<(usize, usize)> {
    let threads = outer
        .checked_mul(inner)
        .ok_or_else(|| GpuError::InvalidState {
            message: format!("{op}: outer * inner overflow"),
        })?;
    let total = threads
        .checked_mul(dim_size)
        .ok_or_else(|| GpuError::InvalidState {
            message: format!("{op}: outer * dim_size * inner overflow"),
        })?;
    for (name, value) in [
        ("outer", outer),
        ("dim_size", dim_size),
        ("inner", inner),
        ("threads", threads),
        ("total", total),
    ] {
        if value > u32::MAX as usize {
            return Err(GpuError::InvalidState {
                message: format!("{op}: {name}={value} exceeds CUDA u32 indexing limit"),
            });
        }
    }
    Ok((threads, total))
}

fn validate_len(op: &str, have: usize, want: usize) -> GpuResult<()> {
    if have < want {
        return Err(GpuError::LengthMismatch { a: want, b: have });
    }
    let _ = op;
    Ok(())
}

fn scan16_ptx(entry: &str, kind: HalfKind, op: &'static str) -> String {
    let init = if op == "sum" {
        "    mov.f32 %acc, 0f00000000;"
    } else {
        "    mov.f32 %acc, 0f3F800000;"
    };
    let combine = if op == "sum" {
        "    add.f32 %acc, %acc, %val;"
    } else {
        "    mul.f32 %acc, %acc, %val;"
    };
    let template = r"
.version $VERSION
.target $TARGET
.address_size 64

.visible .entry $ENTRY(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 outer_size,
    .param .u32 dim_size,
    .param .u32 inner_size,
    .param .u32 total_threads
) {
    .reg .u32 %r_tid, %bid, %bdim, %outer_sz, %dim_sz, %inner_sz;
    .reg .u32 %outer_idx, %inner_idx, %k, %base, %idx, %tmp;
    .reg .u64 %in, %out, %off, %addr;
    .reg .b16 %raw, %acc_h, %zero16;
    .reg .b32 %bits;
    .reg .f32 %val, %acc;
    .reg .pred %p, %lp;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %outer_sz, [outer_size];
    ld.param.u32 %dim_sz, [dim_size];
    ld.param.u32 %inner_sz, [inner_size];
    ld.param.u32 %tmp, [total_threads];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;
    setp.ge.u32 %p, %r_tid, %tmp;
    @%p bra DONE;

    div.u32 %outer_idx, %r_tid, %inner_sz;
    rem.u32 %inner_idx, %r_tid, %inner_sz;
    mul.lo.u32 %base, %outer_idx, %dim_sz;
    mul.lo.u32 %base, %base, %inner_sz;
    add.u32 %base, %base, %inner_idx;

$INIT
    mov.u32 %k, 0;
SCAN_LOOP:
    setp.ge.u32 %lp, %k, %dim_sz;
    @%lp bra DONE;
    mul.lo.u32 %idx, %k, %inner_sz;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
$LOAD
$COMBINE
    add.u64 %addr, %out, %off;
$STORE
    add.u32 %k, %k, 1;
    bra SCAN_LOOP;

DONE:
    ret;
}
"
    .to_string();
    replace_all(
        template,
        &[
            ("$VERSION", kind.ptx_version().to_string()),
            ("$TARGET", kind.target().to_string()),
            ("$ENTRY", entry.to_string()),
            ("$INIT", init.to_string()),
            ("$LOAD", kind.load_raw_to_val().to_string()),
            ("$COMBINE", combine.to_string()),
            ("$STORE", kind.narrow_acc_store_and_reload().to_string()),
        ],
    )
}

fn reverse_cumsum16_ptx(entry: &str, kind: HalfKind) -> String {
    let template = r"
.version $VERSION
.target $TARGET
.address_size 64

.visible .entry $ENTRY(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 outer_size,
    .param .u32 dim_size,
    .param .u32 inner_size,
    .param .u32 total_threads
) {
    .reg .u32 %r_tid, %bid, %bdim, %outer_sz, %dim_sz, %inner_sz;
    .reg .u32 %outer_idx, %inner_idx, %k, %base, %idx, %tmp;
    .reg .u64 %in, %out, %off, %addr;
    .reg .b16 %raw, %acc_h, %zero16;
    .reg .b32 %bits;
    .reg .f32 %val, %acc;
    .reg .pred %p, %done;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %outer_sz, [outer_size];
    ld.param.u32 %dim_sz, [dim_size];
    ld.param.u32 %inner_sz, [inner_size];
    ld.param.u32 %tmp, [total_threads];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;
    setp.ge.u32 %p, %r_tid, %tmp;
    @%p bra DONE;

    div.u32 %outer_idx, %r_tid, %inner_sz;
    rem.u32 %inner_idx, %r_tid, %inner_sz;
    mul.lo.u32 %base, %outer_idx, %dim_sz;
    mul.lo.u32 %base, %base, %inner_sz;
    add.u32 %base, %base, %inner_idx;

    mov.f32 %acc, 0f00000000;
    mov.u32 %k, %dim_sz;
SCAN_LOOP:
    setp.eq.u32 %done, %k, 0;
    @%done bra DONE;
    sub.u32 %k, %k, 1;
    mul.lo.u32 %idx, %k, %inner_sz;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
$LOAD
    add.f32 %acc, %acc, %val;
    add.u64 %addr, %out, %off;
$STORE
    bra SCAN_LOOP;

DONE:
    ret;
}
"
    .to_string();
    replace_all(
        template,
        &[
            ("$VERSION", kind.ptx_version().to_string()),
            ("$TARGET", kind.target().to_string()),
            ("$ENTRY", entry.to_string()),
            ("$LOAD", kind.load_raw_to_val().to_string()),
            ("$STORE", kind.narrow_acc_store_and_reload().to_string()),
        ],
    )
}

fn cumextreme16_ptx(entry: &str, kind: HalfKind, is_max: bool) -> String {
    let init = if is_max {
        "    mov.b32 %acc, 0xFF800000;"
    } else {
        "    mov.b32 %acc, 0x7F800000;"
    };
    let cmp = if is_max {
        "    setp.ge.f32 %cmp, %val, %acc;"
    } else {
        "    setp.le.f32 %cmp, %val, %acc;"
    };
    let template = r"
.version $VERSION
.target $TARGET
.address_size 64

.visible .entry $ENTRY(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u64 indices_ptr,
    .param .u32 outer_size,
    .param .u32 dim_size,
    .param .u32 inner_size,
    .param .u32 total_threads
) {
    .reg .u32 %r_tid, %bid, %bdim, %outer_sz, %dim_sz, %inner_sz;
    .reg .u32 %outer_idx, %inner_idx, %k, %base, %idx, %tmp, %best_k;
    .reg .u64 %in, %out, %ind, %off_val, %off_idx, %addr;
    .reg .s64 %best_k_s64;
    .reg .b16 %raw, %best_raw, %zero16;
    .reg .b32 %bits;
    .reg .f32 %val, %acc;
    .reg .pred %p, %lp, %take, %curr_nan, %acc_nan, %acc_ok, %cmp;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u64 %ind, [indices_ptr];
    ld.param.u32 %outer_sz, [outer_size];
    ld.param.u32 %dim_sz, [dim_size];
    ld.param.u32 %inner_sz, [inner_size];
    ld.param.u32 %tmp, [total_threads];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;
    setp.ge.u32 %p, %r_tid, %tmp;
    @%p bra DONE;

    div.u32 %outer_idx, %r_tid, %inner_sz;
    rem.u32 %inner_idx, %r_tid, %inner_sz;
    mul.lo.u32 %base, %outer_idx, %dim_sz;
    mul.lo.u32 %base, %base, %inner_sz;
    add.u32 %base, %base, %inner_idx;

$INIT
    mov.b16 %best_raw, 0;
    mov.u32 %best_k, 0;
    mov.u32 %k, 0;
SCAN_LOOP:
    setp.ge.u32 %lp, %k, %dim_sz;
    @%lp bra DONE;
    mul.lo.u32 %idx, %k, %inner_sz;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off_val, %idx;
    shl.b64 %off_val, %off_val, 1;
    add.u64 %addr, %in, %off_val;
$LOAD

    setp.nan.f32 %curr_nan, %val, %val;
    setp.nan.f32 %acc_nan, %acc, %acc;
    not.pred %acc_ok, %acc_nan;
$CMP
    and.pred %cmp, %acc_ok, %cmp;
    or.pred %take, %curr_nan, %cmp;
    @%take mov.u32 %best_k, %k;
    @%take mov.f32 %acc, %val;
    @%take mov.b16 %best_raw, %raw;

    add.u64 %addr, %out, %off_val;
    st.global.b16 [%addr], %best_raw;

    cvt.s64.u32 %best_k_s64, %best_k;
    cvt.u64.u32 %off_idx, %idx;
    shl.b64 %off_idx, %off_idx, 3;
    add.u64 %addr, %ind, %off_idx;
    st.global.s64 [%addr], %best_k_s64;

    add.u32 %k, %k, 1;
    bra SCAN_LOOP;

DONE:
    ret;
}
"
    .to_string();
    replace_all(
        template,
        &[
            ("$VERSION", kind.ptx_version().to_string()),
            ("$TARGET", kind.target().to_string()),
            ("$ENTRY", entry.to_string()),
            ("$INIT", init.to_string()),
            ("$LOAD", kind.load_raw_to_val().to_string()),
            ("$CMP", cmp.to_string()),
        ],
    )
}

fn logcumsumexp16_ptx(entry: &str, kind: HalfKind) -> String {
    let raw_to_acc = kind.raw_to_named("%raw", "%acc");
    let template = r"
.version $VERSION
.target $TARGET
.address_size 64

.visible .entry $ENTRY(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 outer_size,
    .param .u32 dim_size,
    .param .u32 inner_size,
    .param .u32 total_threads
) {
    .reg .u32 %r_tid, %bid, %bdim, %outer_sz, %dim_sz, %inner_sz;
    .reg .u32 %outer_idx, %inner_idx, %k, %base, %idx, %tmp;
    .reg .u64 %in, %out, %off, %addr;
    .reg .b16 %raw, %acc_h, %zero16;
    .reg .b32 %bits;
    .reg .f32 %val, %acc, %m, %ea, %ev, %s, %ls, %log2e, %ln2, %abs_m;
    .reg .pred %p, %lp, %inf, %acc_neg_inf, %val_neg_inf, %neg_inf;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %outer_sz, [outer_size];
    ld.param.u32 %dim_sz, [dim_size];
    ld.param.u32 %inner_sz, [inner_size];
    ld.param.u32 %tmp, [total_threads];

    mov.b32 %log2e, 0x3FB8AA3B;
    mov.b32 %ln2, 0x3F317218;

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;
    setp.ge.u32 %p, %r_tid, %tmp;
    @%p bra DONE;

    div.u32 %outer_idx, %r_tid, %inner_sz;
    rem.u32 %inner_idx, %r_tid, %inner_sz;
    mul.lo.u32 %base, %outer_idx, %dim_sz;
    mul.lo.u32 %base, %base, %inner_sz;
    add.u32 %base, %base, %inner_idx;

    cvt.u64.u32 %off, %base;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
    ld.global.b16 %raw, [%addr];
$RAW_TO_ACC
    add.u64 %addr, %out, %off;
    st.global.b16 [%addr], %raw;
    mov.u32 %k, 1;
SCAN_LOOP:
    setp.ge.u32 %lp, %k, %dim_sz;
    @%lp bra DONE;
    mul.lo.u32 %idx, %k, %inner_sz;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
$LOAD

    max.f32 %m, %acc, %val;
    abs.f32 %abs_m, %m;
    setp.eq.f32 %inf, %abs_m, 0f7F800000;
    @%inf bra STORE_MAX;
    setp.eq.f32 %acc_neg_inf, %acc, 0fFF800000;
    setp.eq.f32 %val_neg_inf, %val, 0fFF800000;
    or.pred %neg_inf, %acc_neg_inf, %val_neg_inf;
    @%neg_inf bra STORE_MAX;

    sub.f32 %ea, %acc, %m;
    mul.f32 %ea, %ea, %log2e;
    ex2.approx.f32 %ea, %ea;
    sub.f32 %ev, %val, %m;
    mul.f32 %ev, %ev, %log2e;
    ex2.approx.f32 %ev, %ev;
    add.f32 %s, %ea, %ev;
    lg2.approx.f32 %ls, %s;
    mul.f32 %ls, %ls, %ln2;
    add.f32 %acc, %m, %ls;
    bra STORE_ACC;

STORE_MAX:
    mov.f32 %acc, %m;
STORE_ACC:
    add.u64 %addr, %out, %off;
$STORE
    add.u32 %k, %k, 1;
    bra SCAN_LOOP;

DONE:
    ret;
}
"
    .to_string();
    replace_all(
        template,
        &[
            ("$VERSION", kind.ptx_version().to_string()),
            ("$TARGET", kind.target().to_string()),
            ("$ENTRY", entry.to_string()),
            ("$RAW_TO_ACC", raw_to_acc),
            ("$LOAD", kind.load_raw_to_val().to_string()),
            ("$STORE", kind.narrow_acc_store_and_reload().to_string()),
        ],
    )
}

fn cumprod_backward16_ptx(entry: &str, kind: HalfKind) -> String {
    let load_x = kind.load_raw_to_val().replace("%val", "%x");
    let load_g = kind.load_raw_to_val().replace("%val", "%g");
    let load_y = kind.load_raw_to_val().replace("%val", "%y");
    let template = r"
.version $VERSION
.target $TARGET
.address_size 64

.visible .entry $ENTRY(
    .param .u64 input_ptr,
    .param .u64 grad_ptr,
    .param .u64 output_ptr,
    .param .u32 outer_size,
    .param .u32 dim_size,
    .param .u32 inner_size,
    .param .u32 total_threads
) {
    .reg .u32 %r_tid, %bid, %bdim, %outer_sz, %dim_sz, %inner_sz;
    .reg .u32 %outer_idx, %inner_idx, %i, %j, %base, %idx, %idx_i, %tmp, %first_zero;
    .reg .u64 %in, %grad, %out, %off, %addr;
    .reg .b16 %raw, %acc_h, %zero16;
    .reg .b32 %bits;
    .reg .f32 %x, %g, %y, %prefix, %prod, %tail, %partial, %acc, %rev, %grad_i;
    .reg .pred %p, %done_i, %done_j, %is_zero, %take_zero, %found_zero, %not_found_zero, %not_zero, %before_zero, %no_zero;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %grad, [grad_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %outer_sz, [outer_size];
    ld.param.u32 %dim_sz, [dim_size];
    ld.param.u32 %inner_sz, [inner_size];
    ld.param.u32 %tmp, [total_threads];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;
    setp.ge.u32 %p, %r_tid, %tmp;
    @%p bra DONE;

    div.u32 %outer_idx, %r_tid, %inner_sz;
    rem.u32 %inner_idx, %r_tid, %inner_sz;
    mul.lo.u32 %base, %outer_idx, %dim_sz;
    mul.lo.u32 %base, %base, %inner_sz;
    add.u32 %base, %base, %inner_idx;

    // Find the first zero while retaining the product before it. The prefix
    // product is needed because only that first zero can receive nonzero
    // gradients from positions at and after the zero.
    mov.f32 %prefix, 0f3F800000;
    mov.u32 %first_zero, %dim_sz;
    mov.u32 %i, 0;
FIND_ZERO:
    setp.ge.u32 %done_i, %i, %dim_sz;
    @%done_i bra DISPATCH_ZERO_CASE;
    mul.lo.u32 %idx, %i, %inner_sz;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
$LOAD_X
    setp.ne.u32 %found_zero, %first_zero, %dim_sz;
    not.pred %not_found_zero, %found_zero;
    setp.eq.f32 %is_zero, %x, 0f00000000;
    not.pred %not_zero, %is_zero;
    and.pred %before_zero, %not_found_zero, %not_zero;
    @%before_zero mul.f32 %prefix, %prefix, %x;
    and.pred %take_zero, %not_found_zero, %is_zero;
    @%take_zero mov.u32 %first_zero, %i;
    add.u32 %i, %i, 1;
    bra FIND_ZERO;

DISPATCH_ZERO_CASE:
    setp.eq.u32 %no_zero, %first_zero, %dim_sz;
    @%no_zero bra NO_ZERO_FORWARD;

    // Prefix before the first zero: no zeros, so use the standard saved
    // cumprod + reverse accumulation identity on that segment.
    mov.f32 %prod, 0f3F800000;
    mov.u32 %i, 0;
ZERO_PREFIX_FORWARD:
    setp.ge.u32 %done_i, %i, %first_zero;
    @%done_i bra ZERO_PREFIX_BACKWARD_INIT;
    mul.lo.u32 %idx, %i, %inner_sz;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
$LOAD_X
    mul.f32 %prod, %prod, %x;
    add.u64 %addr, %out, %off;
    mov.f32 %acc, %prod;
$STORE
    add.u32 %i, %i, 1;
    bra ZERO_PREFIX_FORWARD;

ZERO_PREFIX_BACKWARD_INIT:
    mov.f32 %rev, 0f00000000;
    mov.u32 %i, %first_zero;
ZERO_PREFIX_BACKWARD:
    setp.eq.u32 %done_i, %i, 0;
    @%done_i bra FIRST_ZERO_GRAD_INIT;
    sub.u32 %i, %i, 1;
    mul.lo.u32 %idx, %i, %inner_sz;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %grad, %off;
$LOAD_G
    add.u64 %addr, %out, %off;
$LOAD_Y
    fma.rn.f32 %rev, %g, %y, %rev;
    add.u64 %addr, %in, %off;
$LOAD_X
    div.rn.f32 %grad_i, %rev, %x;
    add.u64 %addr, %out, %off;
    mov.f32 %acc, %grad_i;
$STORE
    bra ZERO_PREFIX_BACKWARD;

FIRST_ZERO_GRAD_INIT:
    mov.f32 %acc, 0f00000000;
    mov.f32 %tail, 0f3F800000;
    mov.u32 %j, %first_zero;
FIRST_ZERO_GRAD_LOOP:
    setp.ge.u32 %done_j, %j, %dim_sz;
    @%done_j bra WRITE_FIRST_ZERO_GRAD;
    mul.lo.u32 %idx, %j, %inner_sz;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 1;
    mul.f32 %partial, %prefix, %tail;
    add.u64 %addr, %grad, %off;
$LOAD_G
    fma.rn.f32 %acc, %g, %partial, %acc;
    add.u32 %j, %j, 1;
    setp.ge.u32 %done_j, %j, %dim_sz;
    @%done_j bra WRITE_FIRST_ZERO_GRAD;
    mul.lo.u32 %idx, %j, %inner_sz;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
$LOAD_X
    setp.eq.f32 %is_zero, %x, 0f00000000;
    @%is_zero bra WRITE_FIRST_ZERO_GRAD;
    mul.f32 %tail, %tail, %x;
    bra FIRST_ZERO_GRAD_LOOP;

WRITE_FIRST_ZERO_GRAD:
    mul.lo.u32 %idx_i, %first_zero, %inner_sz;
    add.u32 %idx_i, %base, %idx_i;
    cvt.u64.u32 %off, %idx_i;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %out, %off;
$STORE
    bra DONE;

NO_ZERO_FORWARD:
    mov.f32 %prod, 0f3F800000;
    mov.u32 %i, 0;
NO_ZERO_FORWARD_LOOP:
    setp.ge.u32 %done_i, %i, %dim_sz;
    @%done_i bra NO_ZERO_BACKWARD_INIT;
    mul.lo.u32 %idx, %i, %inner_sz;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
$LOAD_X
    mul.f32 %prod, %prod, %x;
    add.u64 %addr, %out, %off;
    mov.f32 %acc, %prod;
$STORE
    add.u32 %i, %i, 1;
    bra NO_ZERO_FORWARD_LOOP;

NO_ZERO_BACKWARD_INIT:
    mov.f32 %rev, 0f00000000;
    mov.u32 %i, %dim_sz;
NO_ZERO_BACKWARD:
    setp.eq.u32 %done_i, %i, 0;
    @%done_i bra DONE;
    sub.u32 %i, %i, 1;
    mul.lo.u32 %idx_i, %i, %inner_sz;
    add.u32 %idx_i, %base, %idx_i;
    cvt.u64.u32 %off, %idx_i;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %grad, %off;
$LOAD_G
    add.u64 %addr, %out, %off;
$LOAD_Y
    fma.rn.f32 %rev, %g, %y, %rev;
    add.u64 %addr, %in, %off;
$LOAD_X
    div.rn.f32 %grad_i, %rev, %x;
    add.u64 %addr, %out, %off;
    mov.f32 %acc, %grad_i;
$STORE
    bra NO_ZERO_BACKWARD;

DONE:
    ret;
}
"
    .to_string();
    replace_all(
        template,
        &[
            ("$VERSION", kind.ptx_version().to_string()),
            ("$TARGET", kind.target().to_string()),
            ("$ENTRY", entry.to_string()),
            ("$LOAD_X", load_x),
            ("$LOAD_G", load_g),
            ("$LOAD_Y", load_y),
            ("$STORE", kind.narrow_result_store().to_string()),
        ],
    )
}

fn launch16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
    ptx: String,
    entry: &str,
) -> GpuResult<CudaSlice<u16>> {
    let (threads, total) = checked_dims(entry, outer, dim_size, inner)?;
    validate_len(entry, input.len(), total)?;
    let mut out = alloc_zeros_bf16(total, device)?;
    if total == 0 {
        return Ok(out);
    }
    let f = crate::module_cache::get_or_compile_owned(
        device.context(),
        ptx,
        entry.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "cumulative16_kernel",
        source: e,
    })?;
    let cfg = launch_cfg(threads)?;
    let (o, d, i, t) = (outer as u32, dim_size as u32, inner as u32, threads as u32);
    // SAFETY: dimensions were checked to fit u32 and `input.len() >= total`.
    // The kernel maps one thread to one `(outer, inner)` line and only reads
    // or writes indices `base + k * inner` for `k < dim_size`, bounded by
    // `outer * dim_size * inner`.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&o)
            .arg(&d)
            .arg(&i)
            .arg(&t)
            .launch(cfg)?;
    }
    Ok(out)
}

fn launch16_extreme(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
    ptx: String,
    entry: &str,
) -> GpuResult<(CudaSlice<u16>, CudaBuffer<i64>)> {
    let (threads, total) = checked_dims(entry, outer, dim_size, inner)?;
    validate_len(entry, input.len(), total)?;
    let mut values = alloc_zeros_bf16(total, device)?;
    let mut indices = crate::transfer::alloc_zeros::<i64>(total, device)?;
    if total == 0 {
        return Ok((values, indices));
    }
    let f = crate::module_cache::get_or_compile_owned(
        device.context(),
        ptx,
        entry.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "cumextreme16_kernel",
        source: e,
    })?;
    let cfg = launch_cfg(threads)?;
    let (o, d, i, t) = (outer as u32, dim_size as u32, inner as u32, threads as u32);
    // SAFETY: same bounds as `launch16`; `indices` is a fresh i64 buffer with
    // one output slot per input element.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input)
            .arg(&mut values)
            .arg(indices.inner_mut())
            .arg(&o)
            .arg(&d)
            .arg(&i)
            .arg(&t)
            .launch(cfg)?;
    }
    Ok((values, indices))
}

fn launch16_binary_backward(
    input: &CudaSlice<u16>,
    grad: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
    ptx: String,
    entry: &str,
) -> GpuResult<CudaSlice<u16>> {
    let (threads, total) = checked_dims(entry, outer, dim_size, inner)?;
    validate_len(entry, input.len(), total)?;
    validate_len(entry, grad.len(), total)?;
    let mut out = alloc_zeros_bf16(total, device)?;
    if total == 0 {
        return Ok(out);
    }
    let f = crate::module_cache::get_or_compile_owned(
        device.context(),
        ptx,
        entry.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "cumprod_backward16_kernel",
        source: e,
    })?;
    let cfg = launch_cfg(threads)?;
    let (o, d, i, t) = (outer as u32, dim_size as u32, inner as u32, threads as u32);
    // SAFETY: input/grad/output buffers all have `total` half elements, and
    // the kernel only addresses within the validated scan-line domain.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input)
            .arg(grad)
            .arg(&mut out)
            .arg(&o)
            .arg(&d)
            .arg(&i)
            .arg(&t)
            .launch(cfg)?;
    }
    Ok(out)
}

fn reverse_float_ptx(entry: &str, ty: &str, shift: u32, zero: &str) -> String {
    let template = r"
.version 7.0
.target sm_52
.address_size 64

.visible .entry $ENTRY(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 outer_size,
    .param .u32 dim_size,
    .param .u32 inner_size,
    .param .u32 total_threads
) {
    .reg .u32 %r_tid, %bid, %bdim, %outer_sz, %dim_sz, %inner_sz;
    .reg .u32 %outer_idx, %inner_idx, %k, %base, %idx, %tmp;
    .reg .u64 %in, %out, %off, %addr;
    .reg .$TY %val, %acc;
    .reg .pred %p, %done;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %outer_sz, [outer_size];
    ld.param.u32 %dim_sz, [dim_size];
    ld.param.u32 %inner_sz, [inner_size];
    ld.param.u32 %tmp, [total_threads];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;
    setp.ge.u32 %p, %r_tid, %tmp;
    @%p bra DONE;

    div.u32 %outer_idx, %r_tid, %inner_sz;
    rem.u32 %inner_idx, %r_tid, %inner_sz;
    mul.lo.u32 %base, %outer_idx, %dim_sz;
    mul.lo.u32 %base, %base, %inner_sz;
    add.u32 %base, %base, %inner_idx;

    mov.$TY %acc, $ZERO;
    mov.u32 %k, %dim_sz;
SCAN_LOOP:
    setp.eq.u32 %done, %k, 0;
    @%done bra DONE;
    sub.u32 %k, %k, 1;
    mul.lo.u32 %idx, %k, %inner_sz;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, $SHIFT;
    add.u64 %addr, %in, %off;
    ld.global.$TY %val, [%addr];
    add.$TY %acc, %acc, %val;
    add.u64 %addr, %out, %off;
    st.global.$TY [%addr], %acc;
    bra SCAN_LOOP;

DONE:
    ret;
}
"
    .to_string();
    replace_all(
        template,
        &[
            ("$ENTRY", entry.to_string()),
            ("$TY", ty.to_string()),
            ("$ZERO", zero.to_string()),
            ("$SHIFT", shift.to_string()),
        ],
    )
}

fn cumprod_backward_float_ptx(entry: &str, ty: &str, shift: u32, zero: &str, one: &str) -> String {
    let template = r"
.version 7.0
.target sm_52
.address_size 64

.visible .entry $ENTRY(
    .param .u64 input_ptr,
    .param .u64 grad_ptr,
    .param .u64 output_ptr,
    .param .u32 outer_size,
    .param .u32 dim_size,
    .param .u32 inner_size,
    .param .u32 total_threads
) {
    .reg .u32 %r_tid, %bid, %bdim, %outer_sz, %dim_sz, %inner_sz;
    .reg .u32 %outer_idx, %inner_idx, %i, %j, %base, %idx, %idx_i, %tmp, %first_zero;
    .reg .u64 %in, %grad, %out, %off, %addr;
    .reg .$TY %x, %g, %y, %prefix, %prod, %tail, %partial, %acc, %grad_i;
    .reg .pred %p, %done_i, %done_j, %is_zero, %take_zero, %found_zero, %not_found_zero, %not_zero, %before_zero, %no_zero;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %grad, [grad_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %outer_sz, [outer_size];
    ld.param.u32 %dim_sz, [dim_size];
    ld.param.u32 %inner_sz, [inner_size];
    ld.param.u32 %tmp, [total_threads];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;
    setp.ge.u32 %p, %r_tid, %tmp;
    @%p bra DONE;

    div.u32 %outer_idx, %r_tid, %inner_sz;
    rem.u32 %inner_idx, %r_tid, %inner_sz;
    mul.lo.u32 %base, %outer_idx, %dim_sz;
    mul.lo.u32 %base, %base, %inner_sz;
    add.u32 %base, %base, %inner_idx;

    mov.$TY %prefix, $ONE;
    mov.u32 %first_zero, %dim_sz;
    mov.u32 %i, 0;
FIND_ZERO:
    setp.ge.u32 %done_i, %i, %dim_sz;
    @%done_i bra DISPATCH_ZERO_CASE;
    mul.lo.u32 %idx, %i, %inner_sz;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, $SHIFT;
    add.u64 %addr, %in, %off;
    ld.global.$TY %x, [%addr];
    setp.ne.u32 %found_zero, %first_zero, %dim_sz;
    not.pred %not_found_zero, %found_zero;
    setp.eq.$TY %is_zero, %x, $ZERO;
    not.pred %not_zero, %is_zero;
    and.pred %before_zero, %not_found_zero, %not_zero;
    @%before_zero mul.$TY %prefix, %prefix, %x;
    and.pred %take_zero, %not_found_zero, %is_zero;
    @%take_zero mov.u32 %first_zero, %i;
    add.u32 %i, %i, 1;
    bra FIND_ZERO;

DISPATCH_ZERO_CASE:
    setp.eq.u32 %no_zero, %first_zero, %dim_sz;
    @%no_zero bra NO_ZERO_FORWARD;

    mov.$TY %prod, $ONE;
    mov.u32 %i, 0;
ZERO_PREFIX_FORWARD:
    setp.ge.u32 %done_i, %i, %first_zero;
    @%done_i bra ZERO_PREFIX_BACKWARD_INIT;
    mul.lo.u32 %idx, %i, %inner_sz;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, $SHIFT;
    add.u64 %addr, %in, %off;
    ld.global.$TY %x, [%addr];
    mul.$TY %prod, %prod, %x;
    add.u64 %addr, %out, %off;
    st.global.$TY [%addr], %prod;
    add.u32 %i, %i, 1;
    bra ZERO_PREFIX_FORWARD;

ZERO_PREFIX_BACKWARD_INIT:
    mov.$TY %acc, $ZERO;
    mov.u32 %i, %first_zero;
ZERO_PREFIX_BACKWARD:
    setp.eq.u32 %done_i, %i, 0;
    @%done_i bra FIRST_ZERO_GRAD_INIT;
    sub.u32 %i, %i, 1;
    mul.lo.u32 %idx, %i, %inner_sz;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, $SHIFT;
    add.u64 %addr, %grad, %off;
    ld.global.$TY %g, [%addr];
    add.u64 %addr, %out, %off;
    ld.global.$TY %y, [%addr];
    fma.rn.$TY %acc, %g, %y, %acc;
    add.u64 %addr, %in, %off;
    ld.global.$TY %x, [%addr];
    div.rn.$TY %grad_i, %acc, %x;
    add.u64 %addr, %out, %off;
    st.global.$TY [%addr], %grad_i;
    bra ZERO_PREFIX_BACKWARD;

FIRST_ZERO_GRAD_INIT:
    mov.$TY %acc, $ZERO;
    mov.$TY %tail, $ONE;
    mov.u32 %j, %first_zero;
FIRST_ZERO_GRAD_LOOP:
    setp.ge.u32 %done_j, %j, %dim_sz;
    @%done_j bra WRITE_FIRST_ZERO_GRAD;
    mul.lo.u32 %idx, %j, %inner_sz;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, $SHIFT;
    mul.$TY %partial, %prefix, %tail;
    add.u64 %addr, %grad, %off;
    ld.global.$TY %g, [%addr];
    fma.rn.$TY %acc, %g, %partial, %acc;
    add.u32 %j, %j, 1;
    setp.ge.u32 %done_j, %j, %dim_sz;
    @%done_j bra WRITE_FIRST_ZERO_GRAD;
    mul.lo.u32 %idx, %j, %inner_sz;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, $SHIFT;
    add.u64 %addr, %in, %off;
    ld.global.$TY %x, [%addr];
    setp.eq.$TY %is_zero, %x, $ZERO;
    @%is_zero bra WRITE_FIRST_ZERO_GRAD;
    mul.$TY %tail, %tail, %x;
    bra FIRST_ZERO_GRAD_LOOP;

WRITE_FIRST_ZERO_GRAD:
    mul.lo.u32 %idx_i, %first_zero, %inner_sz;
    add.u32 %idx_i, %base, %idx_i;
    cvt.u64.u32 %off, %idx_i;
    shl.b64 %off, %off, $SHIFT;
    add.u64 %addr, %out, %off;
    st.global.$TY [%addr], %acc;
    bra DONE;

NO_ZERO_FORWARD:
    mov.$TY %prod, $ONE;
    mov.u32 %i, 0;
NO_ZERO_FORWARD_LOOP:
    setp.ge.u32 %done_i, %i, %dim_sz;
    @%done_i bra NO_ZERO_BACKWARD_INIT;
    mul.lo.u32 %idx, %i, %inner_sz;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, $SHIFT;
    add.u64 %addr, %in, %off;
    ld.global.$TY %x, [%addr];
    mul.$TY %prod, %prod, %x;
    add.u64 %addr, %out, %off;
    st.global.$TY [%addr], %prod;
    add.u32 %i, %i, 1;
    bra NO_ZERO_FORWARD_LOOP;

NO_ZERO_BACKWARD_INIT:
    mov.$TY %acc, $ZERO;
    mov.u32 %i, %dim_sz;
NO_ZERO_BACKWARD:
    setp.eq.u32 %done_i, %i, 0;
    @%done_i bra DONE;
    sub.u32 %i, %i, 1;
    mul.lo.u32 %idx_i, %i, %inner_sz;
    add.u32 %idx_i, %base, %idx_i;
    cvt.u64.u32 %off, %idx_i;
    shl.b64 %off, %off, $SHIFT;
    add.u64 %addr, %grad, %off;
    ld.global.$TY %g, [%addr];
    add.u64 %addr, %out, %off;
    ld.global.$TY %y, [%addr];
    fma.rn.$TY %acc, %g, %y, %acc;
    add.u64 %addr, %in, %off;
    ld.global.$TY %x, [%addr];
    div.rn.$TY %grad_i, %acc, %x;
    add.u64 %addr, %out, %off;
    st.global.$TY [%addr], %grad_i;
    bra NO_ZERO_BACKWARD;

DONE:
    ret;
}
"
    .to_string();
    replace_all(
        template,
        &[
            ("$ENTRY", entry.to_string()),
            ("$TY", ty.to_string()),
            ("$ZERO", zero.to_string()),
            ("$ONE", one.to_string()),
            ("$SHIFT", shift.to_string()),
        ],
    )
}

fn launch_float_unary(
    input: &CudaBuffer<f32>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
    ptx: String,
    entry: &str,
) -> GpuResult<CudaBuffer<f32>> {
    if input.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: input.device_ordinal(),
            got: device.ordinal(),
        });
    }
    let (threads, total) = checked_dims(entry, outer, dim_size, inner)?;
    validate_len(entry, input.len(), total)?;
    let mut out = alloc_zeros_f32(total, device)?;
    if total == 0 {
        return Ok(out);
    }
    let f = crate::module_cache::get_or_compile_owned(
        device.context(),
        ptx,
        entry.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "cumulative_f32_kernel",
        source: e,
    })?;
    let cfg = launch_cfg(threads)?;
    let (o, d, i, t) = (outer as u32, dim_size as u32, inner as u32, threads as u32);
    // SAFETY: dimensions and input length are validated above.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&o)
            .arg(&d)
            .arg(&i)
            .arg(&t)
            .launch(cfg)?;
    }
    Ok(out)
}

fn launch_double_unary(
    input: &CudaBuffer<f64>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
    ptx: String,
    entry: &str,
) -> GpuResult<CudaBuffer<f64>> {
    if input.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: input.device_ordinal(),
            got: device.ordinal(),
        });
    }
    let (threads, total) = checked_dims(entry, outer, dim_size, inner)?;
    validate_len(entry, input.len(), total)?;
    let mut out = alloc_zeros_f64(total, device)?;
    if total == 0 {
        return Ok(out);
    }
    let f = crate::module_cache::get_or_compile_owned(
        device.context(),
        ptx,
        entry.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "cumulative_f64_kernel",
        source: e,
    })?;
    let cfg = launch_cfg(threads)?;
    let (o, d, i, t) = (outer as u32, dim_size as u32, inner as u32, threads as u32);
    // SAFETY: dimensions and input length are validated above.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&o)
            .arg(&d)
            .arg(&i)
            .arg(&t)
            .launch(cfg)?;
    }
    Ok(out)
}

fn launch_float_binary_backward(
    input: &CudaBuffer<f32>,
    grad: &CudaBuffer<f32>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
    ptx: String,
    entry: &str,
) -> GpuResult<CudaBuffer<f32>> {
    if input.device_ordinal() != device.ordinal() || grad.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: input.device_ordinal(),
            got: device.ordinal(),
        });
    }
    let (threads, total) = checked_dims(entry, outer, dim_size, inner)?;
    validate_len(entry, input.len(), total)?;
    validate_len(entry, grad.len(), total)?;
    let mut out = alloc_zeros_f32(total, device)?;
    if total == 0 {
        return Ok(out);
    }
    let f = crate::module_cache::get_or_compile_owned(
        device.context(),
        ptx,
        entry.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "cumprod_backward_f32_kernel",
        source: e,
    })?;
    let cfg = launch_cfg(threads)?;
    let (o, d, i, t) = (outer as u32, dim_size as u32, inner as u32, threads as u32);
    // SAFETY: input/grad/output lengths are all validated against the same
    // scan domain.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(grad.inner())
            .arg(out.inner_mut())
            .arg(&o)
            .arg(&d)
            .arg(&i)
            .arg(&t)
            .launch(cfg)?;
    }
    Ok(out)
}

fn launch_double_binary_backward(
    input: &CudaBuffer<f64>,
    grad: &CudaBuffer<f64>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
    ptx: String,
    entry: &str,
) -> GpuResult<CudaBuffer<f64>> {
    if input.device_ordinal() != device.ordinal() || grad.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: input.device_ordinal(),
            got: device.ordinal(),
        });
    }
    let (threads, total) = checked_dims(entry, outer, dim_size, inner)?;
    validate_len(entry, input.len(), total)?;
    validate_len(entry, grad.len(), total)?;
    let mut out = alloc_zeros_f64(total, device)?;
    if total == 0 {
        return Ok(out);
    }
    let f = crate::module_cache::get_or_compile_owned(
        device.context(),
        ptx,
        entry.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "cumprod_backward_f64_kernel",
        source: e,
    })?;
    let cfg = launch_cfg(threads)?;
    let (o, d, i, t) = (outer as u32, dim_size as u32, inner as u32, threads as u32);
    // SAFETY: input/grad/output lengths are all validated against the same
    // scan domain.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(grad.inner())
            .arg(out.inner_mut())
            .arg(&o)
            .arg(&d)
            .arg(&i)
            .arg(&t)
            .launch(cfg)?;
    }
    Ok(out)
}

/// f32 reverse cumulative sum.
pub fn gpu_reverse_cumsum_f32(
    input: &CudaBuffer<f32>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    launch_float_unary(
        input,
        outer,
        dim_size,
        inner,
        device,
        reverse_float_ptx("reverse_cumsum_f32_kernel", "f32", 2, "0f00000000"),
        "reverse_cumsum_f32_kernel",
    )
}

/// f64 reverse cumulative sum.
pub fn gpu_reverse_cumsum_f64(
    input: &CudaBuffer<f64>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    launch_double_unary(
        input,
        outer,
        dim_size,
        inner,
        device,
        reverse_float_ptx("reverse_cumsum_f64_kernel", "f64", 3, "0d0000000000000000"),
        "reverse_cumsum_f64_kernel",
    )
}

/// f32 cumprod backward, resident and zero-safe.
pub fn gpu_cumprod_backward_f32(
    input: &CudaBuffer<f32>,
    grad: &CudaBuffer<f32>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    launch_float_binary_backward(
        input,
        grad,
        outer,
        dim_size,
        inner,
        device,
        cumprod_backward_float_ptx(
            "cumprod_backward_f32_kernel",
            "f32",
            2,
            "0f00000000",
            "0f3F800000",
        ),
        "cumprod_backward_f32_kernel",
    )
}

/// f64 cumprod backward, resident and zero-safe.
pub fn gpu_cumprod_backward_f64(
    input: &CudaBuffer<f64>,
    grad: &CudaBuffer<f64>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    launch_double_binary_backward(
        input,
        grad,
        outer,
        dim_size,
        inner,
        device,
        cumprod_backward_float_ptx(
            "cumprod_backward_f64_kernel",
            "f64",
            3,
            "0d0000000000000000",
            "0d3FF0000000000000",
        ),
        "cumprod_backward_f64_kernel",
    )
}

/// f16 cumulative sum with resident half output.
pub fn gpu_cumsum_f16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch16(
        input,
        outer,
        dim_size,
        inner,
        device,
        scan16_ptx("cumsum_f16_kernel", HalfKind::F16, "sum"),
        "cumsum_f16_kernel",
    )
}

/// bf16 cumulative sum with resident bf16 output.
pub fn gpu_cumsum_bf16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch16(
        input,
        outer,
        dim_size,
        inner,
        device,
        scan16_ptx("cumsum_bf16_kernel", HalfKind::BF16, "sum"),
        "cumsum_bf16_kernel",
    )
}

/// f16 cumulative product with resident half output.
pub fn gpu_cumprod_f16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch16(
        input,
        outer,
        dim_size,
        inner,
        device,
        scan16_ptx("cumprod_f16_kernel", HalfKind::F16, "prod"),
        "cumprod_f16_kernel",
    )
}

/// bf16 cumulative product with resident bf16 output.
pub fn gpu_cumprod_bf16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch16(
        input,
        outer,
        dim_size,
        inner,
        device,
        scan16_ptx("cumprod_bf16_kernel", HalfKind::BF16, "prod"),
        "cumprod_bf16_kernel",
    )
}

/// f16 cumulative maximum with int64 indices.
pub fn gpu_cummax_f16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaSlice<u16>, CudaBuffer<i64>)> {
    launch16_extreme(
        input,
        outer,
        dim_size,
        inner,
        device,
        cumextreme16_ptx("cummax_f16_kernel", HalfKind::F16, true),
        "cummax_f16_kernel",
    )
}

/// bf16 cumulative maximum with int64 indices.
pub fn gpu_cummax_bf16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaSlice<u16>, CudaBuffer<i64>)> {
    launch16_extreme(
        input,
        outer,
        dim_size,
        inner,
        device,
        cumextreme16_ptx("cummax_bf16_kernel", HalfKind::BF16, true),
        "cummax_bf16_kernel",
    )
}

/// f16 cumulative minimum with int64 indices.
pub fn gpu_cummin_f16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaSlice<u16>, CudaBuffer<i64>)> {
    launch16_extreme(
        input,
        outer,
        dim_size,
        inner,
        device,
        cumextreme16_ptx("cummin_f16_kernel", HalfKind::F16, false),
        "cummin_f16_kernel",
    )
}

/// bf16 cumulative minimum with int64 indices.
pub fn gpu_cummin_bf16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaSlice<u16>, CudaBuffer<i64>)> {
    launch16_extreme(
        input,
        outer,
        dim_size,
        inner,
        device,
        cumextreme16_ptx("cummin_bf16_kernel", HalfKind::BF16, false),
        "cummin_bf16_kernel",
    )
}

/// f16 logcumsumexp with PyTorch equal-infinity guard.
pub fn gpu_logcumsumexp_f16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch16(
        input,
        outer,
        dim_size,
        inner,
        device,
        logcumsumexp16_ptx("logcumsumexp_f16_kernel", HalfKind::F16),
        "logcumsumexp_f16_kernel",
    )
}

/// bf16 logcumsumexp with PyTorch equal-infinity guard.
pub fn gpu_logcumsumexp_bf16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch16(
        input,
        outer,
        dim_size,
        inner,
        device,
        logcumsumexp16_ptx("logcumsumexp_bf16_kernel", HalfKind::BF16),
        "logcumsumexp_bf16_kernel",
    )
}

/// f16 reverse cumulative sum, used by cumsum/logcumsumexp backward.
pub fn gpu_reverse_cumsum_f16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch16(
        input,
        outer,
        dim_size,
        inner,
        device,
        reverse_cumsum16_ptx("reverse_cumsum_f16_kernel", HalfKind::F16),
        "reverse_cumsum_f16_kernel",
    )
}

/// bf16 reverse cumulative sum, used by cumsum/logcumsumexp backward.
pub fn gpu_reverse_cumsum_bf16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch16(
        input,
        outer,
        dim_size,
        inner,
        device,
        reverse_cumsum16_ptx("reverse_cumsum_bf16_kernel", HalfKind::BF16),
        "reverse_cumsum_bf16_kernel",
    )
}

/// f16 cumprod backward, resident and zero-safe.
pub fn gpu_cumprod_backward_f16(
    input: &CudaSlice<u16>,
    grad: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch16_binary_backward(
        input,
        grad,
        outer,
        dim_size,
        inner,
        device,
        cumprod_backward16_ptx("cumprod_backward_f16_kernel", HalfKind::F16),
        "cumprod_backward_f16_kernel",
    )
}

/// bf16 cumprod backward, resident and zero-safe.
pub fn gpu_cumprod_backward_bf16(
    input: &CudaSlice<u16>,
    grad: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch16_binary_backward(
        input,
        grad,
        outer,
        dim_size,
        inner,
        device,
        cumprod_backward16_ptx("cumprod_backward_bf16_kernel", HalfKind::BF16),
        "cumprod_backward_bf16_kernel",
    )
}
