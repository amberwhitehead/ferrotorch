//! `index_select` / `gather` GPU kernels driven by a GPU-resident **integer**
//! index buffer (`i32` / `i64`) — crosslink #1185 Phase 2c.
//!
//! The pre-existing `index_select_1d` / `index_select_dim` kernels in
//! [`crate::kernels`] take an **f32-encoded** index slice (the legacy
//! `&[u32]`-uploaded-as-f32 contract). Phase 2c's new capability is supplying
//! the indices as a real GPU-resident `IntTensor<i32|i64>` (the Llama token-id
//! path), so these kernels read the index with `ld.global.s32` / `ld.global.s64`
//! directly — no f32 detour, no host round-trip.
//!
//! # Element movement is dtype-generic (raw bytes)
//!
//! `index_select` / `gather` only *copy* elements; they never do arithmetic on
//! the value. So the value kernels are parameterised by **byte width** (2 / 4 /
//! 8), exactly like PyTorch's `gather` which dispatches on `itemsize` for the
//! plain copy. A 4-byte copy kernel serves both f32 and i32; an 8-byte copy
//! serves both f64 and i64; a 2-byte copy serves f16 and bf16. Combined with
//! the two index widths and the two ops, that is 3×2×2 = 12 hand-written PTX
//! entries.
//!
//! # Layouts (C-order, contiguous)
//!
//! `index_select(dim)`: input `[outer, in_dim, inner]`, index `[out_dim]`
//! (1-D), output `[outer, out_dim, inner]`. Thread `t in [0, outer*out_dim*
//! inner)` decomposes to `(o, i, k)` and writes
//! `out[t] = input[o*in_dim*inner + index[i]*inner + k]`.
//!
//! `gather(dim)`: input `[outer, in_dim, inner]`, index AND output both
//! `[outer, out_dim, inner]`. Thread `t` writes
//! `out[t] = input[o*in_dim*inner + index[t]*inner + k]` (the index varies per
//! output element, read at the SAME flat position `t`).
//!
//! # Out-of-range indices
//!
//! Match PyTorch CUDA: an out-of-range index is undefined behaviour on the
//! device (no host round-trip to validate — that would defeat the no-CPU
//! contract). The kernels compute the address from `index[..]` without a bound
//! check, exactly as PyTorch's CUDA `index_select` / `gather` do. Documented;
//! not silently clamped.
//!
//! ## REQ status (per `.design/ferrotorch-gpu/gather_int.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream
//! cites) live in the design doc; this synopsis is a one-line summary per
//! REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (20 isel/gather entries) | SHIPPED | 20 `pub fn isel_* / gather_*` entries in `gather_int.rs` (via `select_entry!` macro invocations); consumer `CudaBackendImpl::gather_or_select in backend_impl.rs` dispatches all 20 cells through the `run!` macro |
//! | REQ-2 (12 PTX macro-expansions) | SHIPPED | `index_select_ptx!` and `gather_ptx!` macros in `gather_int.rs` expand 12 PTX entries (6 select × {W2,W4,W8} × {I32,I64} + 6 gather × ditto), resolved by `isel_ptx / gathr_ptx` in the same file |
//! | REQ-3 (C-order layout contract) | SHIPPED | layout contract documented in `gather_int.rs` module `//!` block and reflected in the PTX address math; verified by unit tests' expected-output construction |
//! | REQ-4 (out-of-range UB contract) | SHIPPED | out-of-range UB contract documented in `gather_int.rs` module `//!` block; PTX templates omit any bounds check on the loaded index, matching upstream `at::native::index_select_cuda` in `aten/src/ATen/native/cuda/Indexing.cu` |
//! | REQ-5 (consumer wiring) | SHIPPED | `CudaBackendImpl::gather_or_select in backend_impl.rs` is the production consumer; ferrotorch-core's `Tensor::index_select / Tensor::gather` dispatch through it via the `GpuBackend::gather_or_select` trait method when source is CUDA-resident |

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
// Macro-built PTX. Parameters:
//   $kname  : entry name (str)
//   $wsh    : value byte-width shift ("1"=2B, "2"=4B, "3"=8B)
//   $ish    : index byte-width shift ("2"=i32, "3"=i64)
//   $ldi    : index load op ("ld.global.s32" / "ld.global.s64")
//   $icvt   : cvt source suffix for sel widening ("s32" / "s64")
//   $ldv    : value load op ("ld.global.u16" / ".u32" / ".u64")
//   $stv    : value store op (matching width)
//   $vreg   : value register type (".u16" / ".u32" / ".u64")
//   $ireg   : index register type (".s32" / ".s64")
//
// index_select: sel = idx[i]  (i = which output row along the dim)
// ===========================================================================
macro_rules! index_select_ptx {
    ($kname:literal, $wsh:literal, $ish:literal, $ldi:literal, $icvt:literal,
     $ldv:literal, $stv:literal, $vreg:literal, $ireg:literal) => {
        concat!(
            ".version 7.0\n.target sm_52\n.address_size 64\n",
            ".visible .entry ",
            $kname,
            "(
    .param .u64 in_ptr, .param .u64 idx_ptr, .param .u64 out_ptr,
    .param .u32 outer, .param .u32 in_dim, .param .u32 out_dim,
    .param .u32 inner, .param .u32 total
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %indim, %outdim, %inn;
    .reg .u32 %o, %rem, %i, %k, %slab, %sel, %srcelem;
    .reg .u64 %in, %idx, %out, %off, %addr;
    .reg ",
            $ireg,
            " %selv;
    .reg ",
            $vreg,
            " %v;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %idx, [idx_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %tot, [total];
    ld.param.u32 %indim, [in_dim];
    ld.param.u32 %outdim, [out_dim];
    ld.param.u32 %inn, [inner];

    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot; @%p bra DONE;

    mul.lo.u32 %slab, %outdim, %inn;
    div.u32 %o, %gtid, %slab;
    rem.u32 %rem, %gtid, %slab;
    div.u32 %i, %rem, %inn;
    rem.u32 %k, %rem, %inn;

    cvt.u64.u32 %off, %i; shl.b64 %off, %off, ",
            $ish,
            "; add.u64 %addr, %idx, %off;
    ",
            $ldi,
            " %selv, [%addr];
    cvt.u32.",
            $icvt,
            " %sel, %selv;

    mul.lo.u32 %srcelem, %o, %indim;
    add.u32 %srcelem, %srcelem, %sel;
    mul.lo.u32 %srcelem, %srcelem, %inn;
    add.u32 %srcelem, %srcelem, %k;

    cvt.u64.u32 %off, %srcelem; shl.b64 %off, %off, ",
            $wsh,
            "; add.u64 %addr, %in, %off;
    ",
            $ldv,
            " %v, [%addr];

    cvt.u64.u32 %off, %gtid; shl.b64 %off, %off, ",
            $wsh,
            "; add.u64 %addr, %out, %off;
    ",
            $stv,
            " [%addr], %v;
DONE:
    ret;
}
"
        )
    };
}

// gather: sel = idx[t]  (index buffer is parallel to output)
macro_rules! gather_ptx {
    ($kname:literal, $wsh:literal, $ish:literal, $ldi:literal, $icvt:literal,
     $ldv:literal, $stv:literal, $vreg:literal, $ireg:literal) => {
        concat!(
            ".version 7.0\n.target sm_52\n.address_size 64\n",
            ".visible .entry ",
            $kname,
            "(
    .param .u64 in_ptr, .param .u64 idx_ptr, .param .u64 out_ptr,
    .param .u32 outer, .param .u32 in_dim, .param .u32 out_dim,
    .param .u32 inner, .param .u32 total
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %indim, %outdim, %inn;
    .reg .u32 %o, %rem, %k, %slab, %sel, %srcelem;
    .reg .u64 %in, %idx, %out, %off, %addr;
    .reg ",
            $ireg,
            " %selv;
    .reg ",
            $vreg,
            " %v;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %idx, [idx_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %tot, [total];
    ld.param.u32 %indim, [in_dim];
    ld.param.u32 %outdim, [out_dim];
    ld.param.u32 %inn, [inner];

    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot; @%p bra DONE;

    mul.lo.u32 %slab, %outdim, %inn;
    div.u32 %o, %gtid, %slab;
    rem.u32 %rem, %gtid, %slab;
    rem.u32 %k, %rem, %inn;

    cvt.u64.u32 %off, %gtid; shl.b64 %off, %off, ",
            $ish,
            "; add.u64 %addr, %idx, %off;
    ",
            $ldi,
            " %selv, [%addr];
    cvt.u32.",
            $icvt,
            " %sel, %selv;

    mul.lo.u32 %srcelem, %o, %indim;
    add.u32 %srcelem, %srcelem, %sel;
    mul.lo.u32 %srcelem, %srcelem, %inn;
    add.u32 %srcelem, %srcelem, %k;

    cvt.u64.u32 %off, %srcelem; shl.b64 %off, %off, ",
            $wsh,
            "; add.u64 %addr, %in, %off;
    ",
            $ldv,
            " %v, [%addr];

    cvt.u64.u32 %off, %gtid; shl.b64 %off, %off, ",
            $wsh,
            "; add.u64 %addr, %out, %off;
    ",
            $stv,
            " [%addr], %v;
DONE:
    ret;
}
"
        )
    };
}

// ── index_select PTX constants (value width × index width) ──────────────────
const ISEL_W2_I32_PTX: &str = index_select_ptx!(
    "isel_w2_i32_kernel",
    "1",
    "2",
    "ld.global.s32",
    "s32",
    "ld.global.u16",
    "st.global.u16",
    ".u16",
    ".s32"
);
const ISEL_W2_I64_PTX: &str = index_select_ptx!(
    "isel_w2_i64_kernel",
    "1",
    "3",
    "ld.global.s64",
    "s64",
    "ld.global.u16",
    "st.global.u16",
    ".u16",
    ".s64"
);
const ISEL_W4_I32_PTX: &str = index_select_ptx!(
    "isel_w4_i32_kernel",
    "2",
    "2",
    "ld.global.s32",
    "s32",
    "ld.global.u32",
    "st.global.u32",
    ".u32",
    ".s32"
);
const ISEL_W4_I64_PTX: &str = index_select_ptx!(
    "isel_w4_i64_kernel",
    "2",
    "3",
    "ld.global.s64",
    "s64",
    "ld.global.u32",
    "st.global.u32",
    ".u32",
    ".s64"
);
const ISEL_W8_I32_PTX: &str = index_select_ptx!(
    "isel_w8_i32_kernel",
    "3",
    "2",
    "ld.global.s32",
    "s32",
    "ld.global.u64",
    "st.global.u64",
    ".u64",
    ".s32"
);
const ISEL_W8_I64_PTX: &str = index_select_ptx!(
    "isel_w8_i64_kernel",
    "3",
    "3",
    "ld.global.s64",
    "s64",
    "ld.global.u64",
    "st.global.u64",
    ".u64",
    ".s64"
);

// ── gather PTX constants ────────────────────────────────────────────────────
const GATHER_W2_I32_PTX: &str = gather_ptx!(
    "gather_w2_i32_kernel",
    "1",
    "2",
    "ld.global.s32",
    "s32",
    "ld.global.u16",
    "st.global.u16",
    ".u16",
    ".s32"
);
const GATHER_W2_I64_PTX: &str = gather_ptx!(
    "gather_w2_i64_kernel",
    "1",
    "3",
    "ld.global.s64",
    "s64",
    "ld.global.u16",
    "st.global.u16",
    ".u16",
    ".s64"
);
const GATHER_W4_I32_PTX: &str = gather_ptx!(
    "gather_w4_i32_kernel",
    "2",
    "2",
    "ld.global.s32",
    "s32",
    "ld.global.u32",
    "st.global.u32",
    ".u32",
    ".s32"
);
const GATHER_W4_I64_PTX: &str = gather_ptx!(
    "gather_w4_i64_kernel",
    "2",
    "3",
    "ld.global.s64",
    "s64",
    "ld.global.u32",
    "st.global.u32",
    ".u32",
    ".s64"
);
const GATHER_W8_I32_PTX: &str = gather_ptx!(
    "gather_w8_i32_kernel",
    "3",
    "2",
    "ld.global.s32",
    "s32",
    "ld.global.u64",
    "st.global.u64",
    ".u64",
    ".s32"
);
const GATHER_W8_I64_PTX: &str = gather_ptx!(
    "gather_w8_i64_kernel",
    "3",
    "3",
    "ld.global.s64",
    "s64",
    "ld.global.u64",
    "st.global.u64",
    ".u64",
    ".s64"
);

/// Byte width of a value element, used to pick the copy kernel.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ValWidth {
    /// 2-byte element (f16 / bf16).
    W2,
    /// 4-byte element (f32 / i32).
    W4,
    /// 8-byte element (f64 / i64).
    W8,
}

/// Index element width.
#[derive(Clone, Copy, PartialEq, Eq)]
enum IdxWidth {
    /// 32-bit signed index.
    I32,
    /// 64-bit signed index.
    I64,
}

#[allow(clippy::too_many_arguments)]
fn launch_select<V: DeviceRepr + ValidAsZeroBits, I: DeviceRepr + ValidAsZeroBits>(
    input: &CudaSlice<V>,
    idx: &CudaSlice<I>,
    outer: usize,
    in_dim: usize,
    out_dim: usize,
    inner: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaSlice<V>> {
    let total = outer
        .checked_mul(out_dim)
        .and_then(|x| x.checked_mul(inner))
        .ok_or(GpuError::LengthMismatch {
            a: outer,
            b: out_dim,
        })?;
    let stream = device.stream();
    if total == 0 {
        return Ok(stream.alloc_zeros::<V>(0)?);
    }
    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;
    let mut out = stream.alloc_zeros::<V>(total)?;
    let cfg = launch_1d(total);
    let (outer_u, indim_u, outdim_u, inner_u, total_u) = (
        outer as u32,
        in_dim as u32,
        out_dim as u32,
        inner as u32,
        total as u32,
    );
    // SAFETY:
    // - `f` is the PTX entry `kernel_name`; its 8-arg signature
    //   (in, idx, out, outer, in_dim, out_dim, inner, total) matches the args
    //   pushed below in order.
    // - `input` (V-elements) and `idx` (I-elements) are immutable inputs; `out`
    //   is the fresh `total`-element V buffer, the only `&mut`, non-aliased.
    // - Each thread writes one `out[t]` for `t in [0,total)` (bound-checked).
    //   The source element is computed from `idx[..]`; an out-of-range index is
    //   documented UB matching PyTorch CUDA (module note), not a memory-safety
    //   bug of this harness — the buffers passed are exactly those sized by the
    //   caller and `total` bounds the writes.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(idx)
            .arg(&mut out)
            .arg(&outer_u)
            .arg(&indim_u)
            .arg(&outdim_u)
            .arg(&inner_u)
            .arg(&total_u)
            .launch(cfg)?;
    }
    Ok(out)
}

fn isel_ptx(vw: ValWidth, iw: IdxWidth) -> (&'static str, &'static str) {
    match (vw, iw) {
        (ValWidth::W2, IdxWidth::I32) => (ISEL_W2_I32_PTX, "isel_w2_i32_kernel"),
        (ValWidth::W2, IdxWidth::I64) => (ISEL_W2_I64_PTX, "isel_w2_i64_kernel"),
        (ValWidth::W4, IdxWidth::I32) => (ISEL_W4_I32_PTX, "isel_w4_i32_kernel"),
        (ValWidth::W4, IdxWidth::I64) => (ISEL_W4_I64_PTX, "isel_w4_i64_kernel"),
        (ValWidth::W8, IdxWidth::I32) => (ISEL_W8_I32_PTX, "isel_w8_i32_kernel"),
        (ValWidth::W8, IdxWidth::I64) => (ISEL_W8_I64_PTX, "isel_w8_i64_kernel"),
    }
}

fn gathr_ptx(vw: ValWidth, iw: IdxWidth) -> (&'static str, &'static str) {
    match (vw, iw) {
        (ValWidth::W2, IdxWidth::I32) => (GATHER_W2_I32_PTX, "gather_w2_i32_kernel"),
        (ValWidth::W2, IdxWidth::I64) => (GATHER_W2_I64_PTX, "gather_w2_i64_kernel"),
        (ValWidth::W4, IdxWidth::I32) => (GATHER_W4_I32_PTX, "gather_w4_i32_kernel"),
        (ValWidth::W4, IdxWidth::I64) => (GATHER_W4_I64_PTX, "gather_w4_i64_kernel"),
        (ValWidth::W8, IdxWidth::I32) => (GATHER_W8_I32_PTX, "gather_w8_i32_kernel"),
        (ValWidth::W8, IdxWidth::I64) => (GATHER_W8_I64_PTX, "gather_w8_i64_kernel"),
    }
}

macro_rules! select_entry {
    ($name:ident, $vty:ty, $vw:expr, $idxty:ty, $iw:expr, $sel:ident) => {
        #[doc = concat!("`", stringify!($sel), "` on a ", stringify!($vty), " value buffer with a ", stringify!($idxty), " index buffer.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            input: &CudaSlice<$vty>,
            idx: &CudaSlice<$idxty>,
            outer: usize,
            in_dim: usize,
            out_dim: usize,
            inner: usize,
            d: &GpuDevice,
        ) -> GpuResult<CudaSlice<$vty>> {
            let (ptx, name) = $sel($vw, $iw);
            launch_select(input, idx, outer, in_dim, out_dim, inner, d, ptx, name)
        }
    };
}

// index_select: value f32/f64/i32/i64/u16(f16,bf16) × index i32/i64
select_entry!(
    isel_f32_i32,
    f32,
    ValWidth::W4,
    i32,
    IdxWidth::I32,
    isel_ptx
);
select_entry!(
    isel_f32_i64,
    f32,
    ValWidth::W4,
    i64,
    IdxWidth::I64,
    isel_ptx
);
select_entry!(
    isel_f64_i32,
    f64,
    ValWidth::W8,
    i32,
    IdxWidth::I32,
    isel_ptx
);
select_entry!(
    isel_f64_i64,
    f64,
    ValWidth::W8,
    i64,
    IdxWidth::I64,
    isel_ptx
);
select_entry!(
    isel_i32_i32,
    i32,
    ValWidth::W4,
    i32,
    IdxWidth::I32,
    isel_ptx
);
select_entry!(
    isel_i32_i64,
    i32,
    ValWidth::W4,
    i64,
    IdxWidth::I64,
    isel_ptx
);
select_entry!(
    isel_i64_i32,
    i64,
    ValWidth::W8,
    i32,
    IdxWidth::I32,
    isel_ptx
);
select_entry!(
    isel_i64_i64,
    i64,
    ValWidth::W8,
    i64,
    IdxWidth::I64,
    isel_ptx
);
select_entry!(
    isel_u16_i32,
    u16,
    ValWidth::W2,
    i32,
    IdxWidth::I32,
    isel_ptx
);
select_entry!(
    isel_u16_i64,
    u16,
    ValWidth::W2,
    i64,
    IdxWidth::I64,
    isel_ptx
);

// gather: same matrix
select_entry!(
    gather_f32_i32,
    f32,
    ValWidth::W4,
    i32,
    IdxWidth::I32,
    gathr_ptx
);
select_entry!(
    gather_f32_i64,
    f32,
    ValWidth::W4,
    i64,
    IdxWidth::I64,
    gathr_ptx
);
select_entry!(
    gather_f64_i32,
    f64,
    ValWidth::W8,
    i32,
    IdxWidth::I32,
    gathr_ptx
);
select_entry!(
    gather_f64_i64,
    f64,
    ValWidth::W8,
    i64,
    IdxWidth::I64,
    gathr_ptx
);
select_entry!(
    gather_i32_i32,
    i32,
    ValWidth::W4,
    i32,
    IdxWidth::I32,
    gathr_ptx
);
select_entry!(
    gather_i32_i64,
    i32,
    ValWidth::W4,
    i64,
    IdxWidth::I64,
    gathr_ptx
);
select_entry!(
    gather_i64_i32,
    i64,
    ValWidth::W8,
    i32,
    IdxWidth::I32,
    gathr_ptx
);
select_entry!(
    gather_i64_i64,
    i64,
    ValWidth::W8,
    i64,
    IdxWidth::I64,
    gathr_ptx
);
select_entry!(
    gather_u16_i32,
    u16,
    ValWidth::W2,
    i32,
    IdxWidth::I32,
    gathr_ptx
);
select_entry!(
    gather_u16_i64,
    u16,
    ValWidth::W2,
    i64,
    IdxWidth::I64,
    gathr_ptx
);

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> GpuDevice {
        GpuDevice::new(0).expect("cuda device")
    }

    #[test]
    fn index_select_dim0_f32_i64() {
        let d = dev();
        // input [4,2] rows; select rows [2,0,2] -> output [3,2]
        let inp = d
            .stream()
            .clone_htod(&vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .unwrap();
        let idx = d.stream().clone_htod(&vec![2i64, 0, 2]).unwrap();
        // outer=1 in_dim=4 out_dim=3 inner=2
        let out = isel_f32_i64(&inp, &idx, 1, 4, 3, 2, &d).unwrap();
        assert_eq!(
            d.stream().clone_dtoh(&out).unwrap(),
            vec![4.0f32, 5.0, 0.0, 1.0, 4.0, 5.0]
        );
    }

    #[test]
    fn index_select_dim1_f32_i32() {
        let d = dev();
        // input [2,3], select along dim=1 indices [2,0] -> outer=2 in_dim=3 out_dim=2 inner=1
        let inp = d
            .stream()
            .clone_htod(&vec![10.0f32, 11.0, 12.0, 20.0, 21.0, 22.0])
            .unwrap();
        let idx = d.stream().clone_htod(&vec![2i32, 0]).unwrap();
        let out = isel_f32_i32(&inp, &idx, 2, 3, 2, 1, &d).unwrap();
        assert_eq!(
            d.stream().clone_dtoh(&out).unwrap(),
            vec![12.0f32, 10.0, 22.0, 20.0]
        );
    }

    #[test]
    fn gather_dim1_i32_values() {
        let d = dev();
        // gather along dim=1: input [2,3], index [2,2] -> outer=2 in_dim=3 out_dim=2 inner=1
        let inp = d.stream().clone_htod(&vec![5i32, 6, 7, 8, 9, 10]).unwrap();
        let idx = d.stream().clone_htod(&vec![0i64, 2, 2, 1]).unwrap();
        let out = gather_i32_i64(&inp, &idx, 2, 3, 2, 1, &d).unwrap();
        // row0: in[0,0]=5 in[0,2]=7 ; row1: in[1,2]=10 in[1,1]=9
        assert_eq!(d.stream().clone_dtoh(&out).unwrap(), vec![5i32, 7, 10, 9]);
    }

    #[test]
    fn index_select_f64_and_i64_values() {
        let d = dev();
        let inp = d.stream().clone_htod(&vec![1.5f64, 2.5, 3.5, 4.5]).unwrap();
        let idx = d.stream().clone_htod(&vec![1i32, 0]).unwrap();
        let out = isel_f64_i32(&inp, &idx, 1, 2, 2, 2, &d).unwrap();
        assert_eq!(
            d.stream().clone_dtoh(&out).unwrap(),
            vec![3.5f64, 4.5, 1.5, 2.5]
        );
    }
}
