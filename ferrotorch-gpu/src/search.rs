//! GPU `searchsorted` / `bucketize` binary-search + `topk` selection kernels
//! (f32 / f64).
//!
//! Mirrors `aten/src/ATen/native/cuda/Bucketization.cu` for the 1-D
//! `boundaries` case (`is_1d_boundaries == true`, so `start_bd == 0` and the
//! whole boundary array is searched for every value). Each output element is
//! an int64 insertion index — PyTorch returns `ScalarType::Long` when
//! `out_int32 == false`, which is the ferrotorch default.
//!
//! # Boundary / tie semantics (the bug-prone part)
//!
//! Matched byte-for-byte to upstream `lower_bound` / `upper_bound`:
//!
//! - `right == false` (PyTorch `side="left"`): first index `i` with
//!   `boundaries[i] >= val`. Upstream `lower_bound` advances `start` while
//!   `!(mid_val >= val)`. A value equal to a boundary lands ON that
//!   boundary's index (`seq[i-1] < v <= seq[i]`).
//! - `right == true` (PyTorch `side="right"`): first index `i` with
//!   `boundaries[i] > val`. Upstream `upper_bound` advances `start` while
//!   `!(mid_val > val)`. A value equal to a boundary lands AFTER it
//!   (`seq[i-1] <= v < seq[i]`).
//!
//! This is the exact pair of half-open comparisons the CPU `partition_point`
//! path in `ferrotorch_core::ops::search::searchsorted` uses
//! (`partition_point(|b| *b < *v)` for left, `partition_point(|b| *b <= *v)`
//! for right), so GPU and CPU agree bit-for-bit, including the tie case where
//! a value equals a boundary.
//!
//! # Kernel layout
//!
//! - Grid: `((n_vals + 255) / 256, 1, 1)`. One thread per value.
//! - Block: `(256, 1, 1)`. No shared memory.
//!
//! Each thread runs a serial `[lo=0, hi=n_bounds)` binary search and writes
//! the converged `lo` as an `s64`.
//!
//! ## REQ status (per `.design/ferrotorch-gpu/search.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream
//! cites) live in the design doc; this synopsis is a one-line summary per
//! REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`gpu_searchsorted_f32`/`_f64`) | SHIPPED | `pub fn gpu_searchsorted_f32`/`_f64 in search.rs` mirror `lower_bound`/`upper_bound` at `aten/src/ATen/native/cuda/Bucketization.cu:26,44`; consumer `CudaBackendImpl::searchsorted_1d in backend_impl.rs` |
//! | REQ-2 (PTX template + ABI) | SHIPPED | `SEARCHSORTED_F32_PTX`/`SEARCHSORTED_F64_PTX in search.rs` carry the 6-arg ABI; launch site binds args in matching order |
//! | REQ-3 (trait surface) | SHIPPED | `fn searchsorted_1d in gpu_dispatch.rs`; consumer `ops::search::searchsorted` GPU branch |
//! | REQ-4 (dispatch wiring) | SHIPPED | `CudaBackendImpl::searchsorted_1d in backend_impl.rs` dispatches `match dtype { F32, F64 }` |
//! | REQ-5 (re-export + consumer) | SHIPPED | `pub use search::{gpu_searchsorted_f32, gpu_searchsorted_f64} in lib.rs`; consumer `ferrotorch_core::ops::search::searchsorted` CUDA branch |
//! | REQ-6 (`gpu_topk_f32`/`_f64`) | SHIPPED | `pub fn gpu_topk_f32`/`_f64 in search.rs` mirror `topk_out_cuda` gather+`sortKeyValueInplace` at `aten/src/ATen/native/cuda/TensorTopK.cpp:97,106`; consumer `CudaBackendImpl::topk_1d in backend_impl.rs` |
//! | REQ-7 (topk PTX + ABI) | SHIPPED | `TOPK_F32_PTX`/`TOPK_F64_PTX in search.rs` (via `topk_ptx!`) carry the 7-arg ABI; launch site binds args in matching order |
//! | REQ-8 (topk trait surface) | SHIPPED | `fn topk_1d in gpu_dispatch.rs`; consumer `ops::search::topk` GPU branch |
//! | REQ-9 (topk dispatch wiring) | SHIPPED | `CudaBackendImpl::topk_1d in backend_impl.rs` dispatches `match dtype { F32, F64 }` |
//! | REQ-10 (topk re-export + consumer) | SHIPPED | `pub use search::{gpu_topk_f32, gpu_topk_f64} in lib.rs`; consumer `ferrotorch_core::ops::search::topk` CUDA branch (values stay GPU-resident) |
//! | REQ-11 (`gpu_histc_f32`/`_f64`) | SHIPPED | `pub fn gpu_histc_f32`/`_f64 in search.rs` mirror getBin + last-bin clamp + range guard at `aten/src/ATen/native/cuda/SummaryOps.cu:41,47,92`; consumer `CudaBackendImpl::histc_1d in backend_impl.rs` |
//! | REQ-12 (histc PTX + ABI) | SHIPPED | `HISTC_F32_PTX`/`HISTC_F64_PTX in search.rs` carry the 6-arg ABI `(in,out,n,nbins,minv,maxv)`; f32 uses `red.global.add.f32`, f64 `red.global.add.f64` (sm_60) |
//! | REQ-13 (histc trait surface) | SHIPPED | `fn histc_1d in gpu_dispatch.rs`; consumer `ops::search::histc` GPU branch |
//! | REQ-14 (histc dispatch + consumer) | SHIPPED | `CudaBackendImpl::histc_1d in backend_impl.rs` dispatches `match dtype { F32, F64 }`; non-test consumer `ferrotorch_core::ops::search::histc` CUDA branch keeps counts GPU-resident (`TensorStorage::gpu`) |
//! | REQ-15 (`gpu_meshgrid_f32`/`_f64`) | SHIPPED | `pub fn gpu_meshgrid_f32`/`_f64 in search.rs` mirror `view(view_shape).expand(shape)` at `aten/src/ATen/native/TensorShape.cpp:4462`; consumer `CudaBackendImpl::meshgrid_grid in backend_impl.rs` |
//! | REQ-16 (meshgrid PTX + ABI) | SHIPPED | `MESHGRID_F32_PTX`/`MESHGRID_F64_PTX in search.rs` carry the 5-arg ABI `(in,out,total,inner,axis_len)`; one thread per output element gathers `in[(flat/inner)%axis_len]` |
//! | REQ-17 (meshgrid trait surface) | SHIPPED | `fn meshgrid_grid in gpu_dispatch.rs`; consumer `ops::search::meshgrid` GPU branch |
//! | REQ-18 (meshgrid dispatch + consumer) | SHIPPED | `CudaBackendImpl::meshgrid_grid in backend_impl.rs` dispatches `match dtype { F32, F64 }`; non-test consumer `ferrotorch_core::ops::search::meshgrid` CUDA branch keeps each grid GPU-resident (`TensorStorage::gpu`) |

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::kernels::gpu_cumsum;
use crate::module_cache::get_or_compile;
use crate::transfer::{alloc_zeros_f32, alloc_zeros_f64, gpu_to_cpu};

const BLOCK_SIZE: u32 = 256;

/// `right` flag values pushed to the kernel.
const SIDE_LEFT: u32 = 0; // lower_bound: first i with boundaries[i] >= val
const SIDE_RIGHT: u32 = 1; // upper_bound: first i with boundaries[i] >  val

fn launch_1d(n: usize) -> LaunchConfig {
    let grid = ((n as u32).saturating_add(BLOCK_SIZE - 1)) / BLOCK_SIZE;
    LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ===========================================================================
// f32
//
// Params: (vals_ptr, bounds_ptr, out_ptr, n_vals, n_bounds, right)
//   vals   : f32[n_vals]
//   bounds : f32[n_bounds]   (sorted ascending, 1-D)
//   out    : i64[n_vals]
//   right  : u32 (0 = lower_bound, 1 = upper_bound)
//
// Thread t in [0, n_vals):
//   v = vals[t]
//   lo = 0; hi = n_bounds
//   while (lo < hi):
//     mid = lo + ((hi - lo) >> 1)
//     bv = bounds[mid]
//     // advance lo while the half-open predicate holds:
//     //   left  (lower_bound): advance while !(bv >= v)  <=>  bv <  v
//     //   right (upper_bound): advance while !(bv >  v)  <=>  bv <= v
//     if (advance) lo = mid + 1 else hi = mid
//   out[t] = (s64) lo
// ===========================================================================
const SEARCHSORTED_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry searchsorted_f32_kernel(
    .param .u64 vals_ptr,
    .param .u64 bounds_ptr,
    .param .u64 out_ptr,
    .param .u32 n_vals,
    .param .u32 n_bounds,
    .param .u32 right
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %t, %nv, %nb, %rt;
    .reg .u32 %lo, %hi, %mid, %half, %mid1;
    .reg .u64 %vals_p, %bnd_p, %out_p, %off, %addr;
    .reg .f32 %v, %bv;
    .reg .s64 %res;
    .reg .pred %p_oob, %p_loop, %p_is_right, %p_not_right, %p_adv;
    .reg .pred %p_ge, %p_gt, %p_nge, %p_ngt, %p_a, %p_b;

    ld.param.u64 %vals_p, [vals_ptr];
    ld.param.u64 %bnd_p,  [bounds_ptr];
    ld.param.u64 %out_p,  [out_ptr];
    ld.param.u32 %nv,     [n_vals];
    ld.param.u32 %nb,     [n_bounds];
    ld.param.u32 %rt,     [right];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %t, %bid_r, %bdim_r, %tid_r;

    setp.ge.u32 %p_oob, %t, %nv;
    @%p_oob bra DONE;

    // v = vals[t]
    cvt.u64.u32 %off, %t;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %vals_p, %off;
    ld.global.f32 %v, [%addr];

    setp.ne.u32 %p_is_right, %rt, 0;   // p_is_right  = (right != 0)
    setp.eq.u32 %p_not_right, %rt, 0;  // p_not_right = (right == 0)

    mov.u32 %lo, 0;
    mov.u32 %hi, %nb;

LOOP:
    setp.ge.u32 %p_loop, %lo, %hi;     // exit when lo >= hi
    @%p_loop bra STORE;

    // mid = lo + ((hi - lo) >> 1)
    sub.u32 %half, %hi, %lo;
    shr.u32 %half, %half, 1;
    add.u32 %mid, %lo, %half;

    // bv = bounds[mid]
    cvt.u64.u32 %off, %mid;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %bnd_p, %off;
    ld.global.f32 %bv, [%addr];

    // advance predicate (no `selp.pred`; build it with predicate logic),
    // mirroring upstream aten/src/ATen/native/cuda/Bucketization.cu:33,51:
    //   left  (lower_bound): advance while `!(bv >= v)`  (Bucketization.cu:33)
    //   right (upper_bound): advance while `!(bv >  v)`  (Bucketization.cu:51)
    //   p_adv = (right & !(bv > v)) | (!right & !(bv >= v))
    // `setp.ge`/`setp.gt` are ORDERED (false for NaN), so the negation is TRUE
    // for a NaN value -> always advance -> lo = len, matching torch. For finite
    // operands `!(bv >= v) == (bv < v)` and `!(bv > v) == (bv <= v)`, so the
    // finite tie/dup/oob cases are byte-identical to the prior setp.lt/le form.
    setp.ge.f32 %p_ge, %bv, %v;        // p_ge = (bv >= v), ordered (false for NaN)
    setp.gt.f32 %p_gt, %bv, %v;        // p_gt = (bv >  v), ordered (false for NaN)
    not.pred %p_nge, %p_ge;            // p_nge = !(bv >= v)  (true for NaN)
    not.pred %p_ngt, %p_gt;            // p_ngt = !(bv >  v)  (true for NaN)
    and.pred %p_a, %p_is_right, %p_ngt;
    and.pred %p_b, %p_not_right, %p_nge;
    or.pred %p_adv, %p_a, %p_b;

    // if advance: lo = mid + 1 ; else: hi = mid
    add.u32 %mid1, %mid, 1;
    @%p_adv mov.u32 %lo, %mid1;
    @!%p_adv mov.u32 %hi, %mid;
    bra LOOP;

STORE:
    cvt.s64.u32 %res, %lo;
    cvt.u64.u32 %off, %t;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out_p, %off;
    st.global.s64 [%addr], %res;

DONE:
    ret;
}
";

// ===========================================================================
// f64 — identical structure, 8-byte value stride, .f64 compares.
// ===========================================================================
const SEARCHSORTED_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry searchsorted_f64_kernel(
    .param .u64 vals_ptr,
    .param .u64 bounds_ptr,
    .param .u64 out_ptr,
    .param .u32 n_vals,
    .param .u32 n_bounds,
    .param .u32 right
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %t, %nv, %nb, %rt;
    .reg .u32 %lo, %hi, %mid, %half, %mid1;
    .reg .u64 %vals_p, %bnd_p, %out_p, %off, %addr;
    .reg .f64 %v, %bv;
    .reg .s64 %res;
    .reg .pred %p_oob, %p_loop, %p_is_right, %p_not_right, %p_adv;
    .reg .pred %p_ge, %p_gt, %p_nge, %p_ngt, %p_a, %p_b;

    ld.param.u64 %vals_p, [vals_ptr];
    ld.param.u64 %bnd_p,  [bounds_ptr];
    ld.param.u64 %out_p,  [out_ptr];
    ld.param.u32 %nv,     [n_vals];
    ld.param.u32 %nb,     [n_bounds];
    ld.param.u32 %rt,     [right];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %t, %bid_r, %bdim_r, %tid_r;

    setp.ge.u32 %p_oob, %t, %nv;
    @%p_oob bra DONE;

    cvt.u64.u32 %off, %t;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %vals_p, %off;
    ld.global.f64 %v, [%addr];

    setp.ne.u32 %p_is_right, %rt, 0;
    setp.eq.u32 %p_not_right, %rt, 0;

    mov.u32 %lo, 0;
    mov.u32 %hi, %nb;

LOOP:
    setp.ge.u32 %p_loop, %lo, %hi;
    @%p_loop bra STORE;

    sub.u32 %half, %hi, %lo;
    shr.u32 %half, %half, 1;
    add.u32 %mid, %lo, %half;

    cvt.u64.u32 %off, %mid;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %bnd_p, %off;
    ld.global.f64 %bv, [%addr];

    // advance predicate mirroring aten/src/ATen/native/cuda/Bucketization.cu:33,51:
    //   left  (lower_bound): advance while `!(bv >= v)`  (Bucketization.cu:33)
    //   right (upper_bound): advance while `!(bv >  v)`  (Bucketization.cu:51)
    // `setp.ge`/`setp.gt` are ORDERED (false for NaN) -> negation TRUE for NaN ->
    // always advance -> lo = len, matching torch. Finite operands unchanged.
    setp.ge.f64 %p_ge, %bv, %v;        // p_ge = (bv >= v), ordered (false for NaN)
    setp.gt.f64 %p_gt, %bv, %v;        // p_gt = (bv >  v), ordered (false for NaN)
    not.pred %p_nge, %p_ge;            // p_nge = !(bv >= v)  (true for NaN)
    not.pred %p_ngt, %p_gt;            // p_ngt = !(bv >  v)  (true for NaN)
    and.pred %p_a, %p_is_right, %p_ngt;
    and.pred %p_b, %p_not_right, %p_nge;
    or.pred %p_adv, %p_a, %p_b;

    add.u32 %mid1, %mid, 1;
    @%p_adv mov.u32 %lo, %mid1;
    @!%p_adv mov.u32 %hi, %mid;
    bra LOOP;

STORE:
    cvt.s64.u32 %res, %lo;
    cvt.u64.u32 %off, %t;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out_p, %off;
    st.global.s64 [%addr], %res;

DONE:
    ret;
}
";

/// Launch one of the searchsorted kernels over device-resident value /
/// boundary slices, returning a fresh `CudaSlice<i64>` of insertion indices.
///
/// `n_vals` / `n_bounds` are the LOGICAL element counts; the slices may be
/// pool-oversized (their `len()` can exceed the logical count), so we require
/// only that each holds AT LEAST its logical count.
fn launch_searchsorted<V>(
    values: &CudaSlice<V>,
    boundaries: &CudaSlice<V>,
    n_vals: usize,
    n_bounds: usize,
    right: bool,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaSlice<i64>>
where
    V: cudarc::driver::DeviceRepr,
{
    if values.len() < n_vals {
        return Err(GpuError::LengthMismatch {
            a: values.len(),
            b: n_vals,
        });
    }
    if boundaries.len() < n_bounds {
        return Err(GpuError::LengthMismatch {
            a: boundaries.len(),
            b: n_bounds,
        });
    }
    if n_vals > u32::MAX as usize || n_bounds > u32::MAX as usize {
        return Err(GpuError::LengthMismatch {
            a: n_vals,
            b: u32::MAX as usize,
        });
    }

    let stream = device.stream();
    if n_vals == 0 {
        // No values to place — an empty 0-length i64 buffer is the answer.
        return Ok(stream.alloc_zeros::<i64>(0)?);
    }

    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;

    let mut out = stream.alloc_zeros::<i64>(n_vals)?;
    let cfg = launch_1d(n_vals);
    let n_vals_u = n_vals as u32;
    let n_bounds_u = n_bounds as u32;
    let right_u = if right { SIDE_RIGHT } else { SIDE_LEFT };

    // SAFETY:
    // - `f` is the PTX entry `kernel_name`; its 6-arg signature
    //   (vals_ptr, bounds_ptr, out_ptr, n_vals, n_bounds, right) matches the
    //   args pushed below in order.
    // - `values` holds at least `n_vals` `V`-elements and `boundaries` at
    //   least `n_bounds` `V`-elements (both checked above); the kernel reads
    //   `vals[t]` for `t in [0, n_vals)` and `bounds[mid]` for
    //   `mid in [lo, hi) ⊆ [0, n_bounds)`, all strictly in range.
    // - `out` is a fresh `n_vals`-element i64 buffer (just allocated), the
    //   only `&mut` arg, and cannot alias `values`/`boundaries` because
    //   cudarc returns a distinct `CudaSlice` and Rust's borrow checker keeps
    //   the `&mut out` borrow exclusive.
    // - Every thread either has `t < n_vals` or exits early via the leading
    //   `setp.ge.u32 %p_oob` predicate, so only `out[t]` for `t in [0,n_vals)`
    //   is written.
    // - `n_vals`/`n_bounds` are range-checked against `u32::MAX`, so the
    //   kernel's u32 index arithmetic cannot overflow.
    // - cudarc copies the by-reference `u32` params into the launch parameter
    //   buffer; their lifetime is tied to this stack frame which outlives the
    //   synchronous `launch`. Stream sync is the caller's responsibility
    //   (matches the rest of the kernel modules, e.g. `reduce_arg`).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(values)
            .arg(boundaries)
            .arg(&mut out)
            .arg(&n_vals_u)
            .arg(&n_bounds_u)
            .arg(&right_u)
            .launch(cfg)?;
    }

    Ok(out)
}

/// On-device `searchsorted` over an f32 sorted 1-D boundary buffer.
///
/// Returns a device `CudaSlice<i64>` of length `n_vals`; `out[t]` is the
/// insertion index of `values[t]` into the sorted `boundaries`, using
/// `right=false` → lower_bound (first `i` with `boundaries[i] >= v`) or
/// `right=true` → upper_bound (first `i` with `boundaries[i] > v`).
///
/// Mirrors `searchsorted_cuda_kernel` (`is_1d_boundaries == true`) in
/// `aten/src/ATen/native/cuda/Bucketization.cu`.
///
/// # Errors
///
/// - [`GpuError::LengthMismatch`] when a slice is shorter than its logical
///   count or a count exceeds `u32::MAX`.
/// - [`GpuError::PtxCompileFailed`] if the PTX module fails to compile.
/// - [`GpuError::Driver`] on launch failure.
pub fn gpu_searchsorted_f32(
    values: &CudaSlice<f32>,
    boundaries: &CudaSlice<f32>,
    n_vals: usize,
    n_bounds: usize,
    right: bool,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_searchsorted(
        values,
        boundaries,
        n_vals,
        n_bounds,
        right,
        device,
        SEARCHSORTED_F32_PTX,
        "searchsorted_f32_kernel",
    )
}

/// On-device `searchsorted` over an f64 sorted 1-D boundary buffer. f64
/// counterpart of [`gpu_searchsorted_f32`].
///
/// # Errors
///
/// See [`gpu_searchsorted_f32`].
pub fn gpu_searchsorted_f64(
    values: &CudaSlice<f64>,
    boundaries: &CudaSlice<f64>,
    n_vals: usize,
    n_bounds: usize,
    right: bool,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_searchsorted(
        values,
        boundaries,
        n_vals,
        n_bounds,
        right,
        device,
        SEARCHSORTED_F64_PTX,
        "searchsorted_f64_kernel",
    )
}

// ===========================================================================
// topk — k extrema along the last dim, with int64 indices
//
// Layout: input is logically `[outer, dim]` (row-major, contiguous). One CUDA
// thread per output slice (`outer` threads). Each thread does a serial
// selection of `k` extrema over its `dim`-length slice and writes `k` values
// (same dtype) + `k` int64 indices in *sorted* order:
//
//   - largest == 1: descending value; ties broken by ascending original index.
//   - largest == 0: ascending value;  ties broken by ascending original index.
//
// This is byte-identical to the CPU `topk` path
// (`ferrotorch_core::ops::search::topk`, which uses Rust's stable `sort_by`)
// AND is a valid `torch.topk(sorted=True)` result: upstream
// `topk_out_cuda` (`aten/src/ATen/native/cuda/TensorTopK.cpp:101`) gathers the
// top-k then calls `sortKeyValueInplace(.., stable=false)`, so torch leaves the
// per-tie index order unspecified — any permutation within a tie group is a
// correct topk. We pin the deterministic ascending-index choice so GPU and CPU
// agree exactly and the divergence-prone tie case is reproducible.
//
// Selection-sort-by-rank (no data mutation): for output position j we find the
// element that ranks best among those strictly *after* the previously-selected
// element in the sort order. "outranks" for largest:
//     a beats b  <=>  (a > b) || (a == b && idx_a < idx_b)
// "after previous" for largest:
//     elem is eligible  <=>  (prev > val) || (prev == val && prev_idx < idx)
// For largest == 0 the value `>` becomes `<` (index tie-break stays ascending).
// `setp.gt`/`setp.lt`/`setp.eq` are ORDERED (false for NaN). NaN ordering
// mirrors torch's sort/topk comparator (`GTOp`/`LTOp` with `handleNaN=true` in
// `aten/src/ATen/native/cuda/SortingCommon.cuh:47-60`): NaN compares GREATER
// than every finite/inf value. The kernel adds `testp.notanumber` terms so a
// NaN value OUTRANKS any finite for largest=true (selected first) and is ranked
// LAST for largest=false (only picked once finite values are exhausted), and
// two NaNs compare equal so the ascending-index tie-break applies. This is
// byte-identical to the CPU path (`ops::search::topk` `nan_is_max_cmp`) and to
// `torch.topk` — verified live on torch 2.11.0+cu130 (RTX 3090):
//   topk([3,NaN,1,5,NaN,2], k=4, largest=True) -> [NaN,NaN,5,3] idx [1,4,3,0].
//
// ABI: (in_ptr, vals_ptr, idx_ptr, outer, dim, k, largest)
//   in    : V[outer * dim]
//   vals  : V[outer * k]    (output values)
//   idx   : i64[outer * k]  (output indices into [0, dim))
//   largest : u32 (1 = largest, 0 = smallest)
// ===========================================================================

/// Generate a topk PTX kernel for the given element type.
///
/// `tyld` is the PTX value register/instruction type (`f32`/`f64`), `vbytes`
/// is the element stride shift (`2` for f32, `3` for f64). The structure is
/// identical across dtypes — only the load/compare type and stride differ.
macro_rules! topk_ptx {
    ($entry:literal, $tyld:literal, $shift:literal) => {
        concat!(
            ".version 7.0\n.target sm_52\n.address_size 64\n\n.visible .entry ",
            $entry,
            "(\n",
            "    .param .u64 in_ptr,\n",
            "    .param .u64 vals_ptr,\n",
            "    .param .u64 idx_ptr,\n",
            "    .param .u32 outer,\n",
            "    .param .u32 dim,\n",
            "    .param .u32 k,\n",
            "    .param .u32 largest\n",
            ") {\n",
            "    .reg .u32 %tid_r, %bid_r, %bdim_r, %s, %no, %nd, %nk, %lg;\n",
            "    .reg .u32 %j, %i, %prev_idx, %best_idx, %cur_idx;\n",
            "    .reg .u64 %in_p, %vp, %ip, %slice_off, %off, %addr, %tmp64;\n",
            "    .reg .",
            $tyld,
            " %prev_val, %best_val, %cur_val;\n",
            "    .reg .s64 %ridx;\n",
            "    .reg .pred %p_oob, %p_jloop, %p_iloop, %p_lg, %p_have, %p_first;\n",
            "    .reg .pred %p_elig, %p_beat, %p_vgt, %p_vlt, %p_veq, %p_vsel, %p_idx;\n",
            "    .reg .pred %p_pgt, %p_plt, %p_peq, %p_psel, %p_pidx, %p_upd;\n",
            "    .reg .pred %p_cnan, %p_pnan, %p_bnan, %p_na, %p_nb;\n",
            "\n",
            "    ld.param.u64 %in_p, [in_ptr];\n",
            "    ld.param.u64 %vp,   [vals_ptr];\n",
            "    ld.param.u64 %ip,   [idx_ptr];\n",
            "    ld.param.u32 %no,   [outer];\n",
            "    ld.param.u32 %nd,   [dim];\n",
            "    ld.param.u32 %nk,   [k];\n",
            "    ld.param.u32 %lg,   [largest];\n",
            "\n",
            "    mov.u32 %tid_r,  %tid.x;\n",
            "    mov.u32 %bid_r,  %ctaid.x;\n",
            "    mov.u32 %bdim_r, %ntid.x;\n",
            "    mad.lo.u32 %s, %bid_r, %bdim_r, %tid_r;\n",
            "    setp.ge.u32 %p_oob, %s, %no;\n",
            "    @%p_oob bra DONE;\n",
            "\n",
            "    setp.ne.u32 %p_lg, %lg, 0;          // p_lg = largest\n",
            "    // slice_off = s * dim (in elements)\n",
            "    mul.lo.u32 %i, %s, %nd;\n",
            "    cvt.u64.u32 %slice_off, %i;\n",
            "    shl.b64 %slice_off, %slice_off, ",
            $shift,
            ";\n",
            "\n",
            "    mov.u32 %j, 0;\n",
            "JLOOP:\n",
            "    setp.ge.u32 %p_jloop, %j, %nk;\n",
            "    @%p_jloop bra DONE;\n",
            "\n",
            "    setp.eq.u32 %p_first, %j, 0;        // j == 0 -> no previous pick\n",
            "    mov.pred %p_have, 0;                // have_best = false\n",
            "    mov.u32 %i, 0;\n",
            "ILOOP:\n",
            "    setp.ge.u32 %p_iloop, %i, %nd;\n",
            "    @%p_iloop bra ISTORE;\n",
            "\n",
            "    // cur_val = in[slice + i]\n",
            "    cvt.u64.u32 %off, %i;\n",
            "    shl.b64 %off, %off, ",
            $shift,
            ";\n",
            "    add.u64 %addr, %in_p, %slice_off;\n",
            "    add.u64 %addr, %addr, %off;\n",
            "    ld.global.",
            $tyld,
            " %cur_val, [%addr];\n",
            "    mov.u32 %cur_idx, %i;\n",
            "    testp.notanumber.",
            $tyld,
            " %p_cnan, %cur_val;          // p_cnan = isnan(cur)\n",
            "\n",
            "    // eligibility: for j==0 every element is eligible. Otherwise eligible iff\n",
            "    // `prev` ranks strictly before `cur` in selection order. NaN ordering\n",
            "    // mirrors torch's GTOp/LTOp comparator with handleNaN=true\n",
            "    // (aten/src/ATen/native/cuda/SortingCommon.cuh:47-60): NaN compares\n",
            "    // GREATER than every finite/inf value. So `prev outranks cur`:\n",
            "    //   largest:  (isnan(prev) && !isnan(cur)) || (prev > cur)\n",
            "    //   smallest: (isnan(cur)  && !isnan(prev)) || (prev < cur)\n",
            "    // equal-rank (so the ascending-index tie-break applies, incl. NaN==NaN):\n",
            "    //   (isnan(prev) && isnan(cur)) || (prev == cur)\n",
            "    // `setp.gt/lt/eq` are ORDERED (false if either operand is NaN), so the\n",
            "    // finite terms need no extra masking; the NaN terms add the ordering.\n",
            "    testp.notanumber.",
            $tyld,
            " %p_pnan, %prev_val;          // p_pnan = isnan(prev)\n",
            "    setp.gt.",
            $tyld,
            " %p_pgt, %prev_val, %cur_val;\n",
            "    setp.lt.",
            $tyld,
            " %p_plt, %prev_val, %cur_val;\n",
            "    setp.eq.",
            $tyld,
            " %p_peq, %prev_val, %cur_val;\n",
            "    // NaN-greater terms\n",
            "    not.pred %p_na, %p_cnan;            // !isnan(cur)\n",
            "    and.pred %p_na, %p_pnan, %p_na;     // isnan(prev) && !isnan(cur)\n",
            "    or.pred  %p_pgt, %p_pgt, %p_na;     // largest:  prev outranks cur\n",
            "    not.pred %p_nb, %p_pnan;            // !isnan(prev)\n",
            "    and.pred %p_nb, %p_cnan, %p_nb;     // isnan(cur) && !isnan(prev)\n",
            "    or.pred  %p_plt, %p_plt, %p_nb;     // smallest: prev outranks cur\n",
            "    and.pred %p_na, %p_pnan, %p_cnan;   // isnan(prev) && isnan(cur)\n",
            "    or.pred  %p_peq, %p_peq, %p_na;     // equal-rank (incl. NaN==NaN)\n",
            "    // p_psel = largest ? p_pgt : p_plt\n",
            "    and.pred %p_psel, %p_lg, %p_pgt;\n",
            "    not.pred %p_idx, %p_lg;\n",
            "    and.pred %p_pidx, %p_idx, %p_plt;\n",
            "    or.pred  %p_psel, %p_psel, %p_pidx;\n",
            "    setp.lt.u32 %p_pidx, %prev_idx, %cur_idx;\n",
            "    and.pred %p_pidx, %p_peq, %p_pidx;  // equal-rank && prev_idx<cur_idx\n",
            "    or.pred  %p_elig, %p_psel, %p_pidx;\n",
            "    or.pred  %p_elig, %p_elig, %p_first; // j==0 -> always eligible\n",
            "    @!%p_elig bra INEXT;\n",
            "\n",
            "    // candidate beats current best? Same NaN-as-maximum comparator:\n",
            "    //   if !have_best -> yes\n",
            "    //   else largest:  (isnan(cur) && !isnan(best)) || (cur > best)\n",
            "    //                  || (equal-rank && cur_idx < best_idx)\n",
            "    //        smallest: (isnan(best) && !isnan(cur)) || (cur < best)\n",
            "    //                  || (equal-rank && cur_idx < best_idx)\n",
            "    not.pred %p_upd, %p_have;           // !have_best\n",
            "    testp.notanumber.",
            $tyld,
            " %p_bnan, %best_val;          // p_bnan = isnan(best)\n",
            "    setp.gt.",
            $tyld,
            " %p_vgt, %cur_val, %best_val;\n",
            "    setp.lt.",
            $tyld,
            " %p_vlt, %cur_val, %best_val;\n",
            "    setp.eq.",
            $tyld,
            " %p_veq, %cur_val, %best_val;\n",
            "    not.pred %p_na, %p_bnan;            // !isnan(best)\n",
            "    and.pred %p_na, %p_cnan, %p_na;     // isnan(cur) && !isnan(best)\n",
            "    or.pred  %p_vgt, %p_vgt, %p_na;     // largest:  cur outranks best\n",
            "    not.pred %p_nb, %p_cnan;            // !isnan(cur)\n",
            "    and.pred %p_nb, %p_bnan, %p_nb;     // isnan(best) && !isnan(cur)\n",
            "    or.pred  %p_vlt, %p_vlt, %p_nb;     // smallest: cur outranks best\n",
            "    and.pred %p_na, %p_cnan, %p_bnan;   // isnan(cur) && isnan(best)\n",
            "    or.pred  %p_veq, %p_veq, %p_na;     // equal-rank (incl. NaN==NaN)\n",
            "    and.pred %p_vsel, %p_lg, %p_vgt;\n",
            "    not.pred %p_idx, %p_lg;\n",
            "    and.pred %p_idx, %p_idx, %p_vlt;\n",
            "    or.pred  %p_vsel, %p_vsel, %p_idx;\n",
            "    setp.lt.u32 %p_idx, %cur_idx, %best_idx;\n",
            "    and.pred %p_idx, %p_veq, %p_idx;\n",
            "    or.pred  %p_beat, %p_vsel, %p_idx;\n",
            "    and.pred %p_beat, %p_beat, %p_have; // only meaningful when have_best\n",
            "    or.pred  %p_upd, %p_upd, %p_beat;\n",
            "    @!%p_upd bra INEXT;\n",
            "\n",
            "    mov.",
            $tyld,
            " %best_val, %cur_val;\n",
            "    mov.u32 %best_idx, %cur_idx;\n",
            "    mov.pred %p_have, 1;\n",
            "\n",
            "INEXT:\n",
            "    add.u32 %i, %i, 1;\n",
            "    bra ILOOP;\n",
            "\n",
            "ISTORE:\n",
            "    // out position = s * k + j\n",
            "    mul.lo.u32 %cur_idx, %s, %nk;\n",
            "    add.u32 %cur_idx, %cur_idx, %j;\n",
            "    cvt.u64.u32 %off, %cur_idx;\n",
            "    // store value\n",
            "    shl.b64 %addr, %off, ",
            $shift,
            ";\n",
            "    add.u64 %addr, %vp, %addr;\n",
            "    st.global.",
            $tyld,
            " [%addr], %best_val;\n",
            "    // store index (i64)\n",
            "    shl.b64 %tmp64, %off, 3;\n",
            "    add.u64 %addr, %ip, %tmp64;\n",
            "    cvt.s64.u32 %ridx, %best_idx;\n",
            "    st.global.s64 [%addr], %ridx;\n",
            "    // prev = best (for next j)\n",
            "    mov.",
            $tyld,
            " %prev_val, %best_val;\n",
            "    mov.u32 %prev_idx, %best_idx;\n",
            "\n",
            "    add.u32 %j, %j, 1;\n",
            "    bra JLOOP;\n",
            "\n",
            "DONE:\n",
            "    ret;\n",
            "}\n"
        )
    };
}

const TOPK_F32_PTX: &str = topk_ptx!("topk_f32_kernel", "f32", "2");
const TOPK_F64_PTX: &str = topk_ptx!("topk_f64_kernel", "f64", "3");

fn launch_topk_config(outer: usize) -> LaunchConfig {
    let grid = ((outer as u32).saturating_add(BLOCK_SIZE - 1)) / BLOCK_SIZE;
    LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch a topk kernel over a device-resident `[outer, dim]` value buffer,
/// returning `(values, indices)`: a fresh `CudaSlice<V>` of `outer * k` extrema
/// (same dtype as the input) and a `CudaSlice<i64>` of the matching original
/// indices into `[0, dim)`. Both outputs stay GPU-resident.
///
/// One thread per output slice; each thread serially selects `k` extrema in
/// sorted order with an ascending-index tie-break (see the module-level note).
#[allow(clippy::too_many_arguments)]
fn launch_topk<V>(
    input: &CudaSlice<V>,
    outer: usize,
    dim: usize,
    k: usize,
    largest: bool,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<(CudaSlice<V>, CudaSlice<i64>)>
where
    V: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
{
    if k > dim {
        return Err(GpuError::LengthMismatch { a: k, b: dim });
    }
    if input.len() < outer.saturating_mul(dim) {
        return Err(GpuError::LengthMismatch {
            a: input.len(),
            b: outer.saturating_mul(dim),
        });
    }
    if outer > u32::MAX as usize || dim > u32::MAX as usize || k > u32::MAX as usize {
        return Err(GpuError::LengthMismatch {
            a: outer.max(dim).max(k),
            b: u32::MAX as usize,
        });
    }

    let stream = device.stream();
    let n_out = outer.saturating_mul(k);
    if n_out == 0 {
        return Ok((stream.alloc_zeros::<V>(0)?, stream.alloc_zeros::<i64>(0)?));
    }

    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;

    let mut out_vals = stream.alloc_zeros::<V>(n_out)?;
    let mut out_idx = stream.alloc_zeros::<i64>(n_out)?;
    let cfg = launch_topk_config(outer);
    let outer_u = outer as u32;
    let dim_u = dim as u32;
    let k_u = k as u32;
    let largest_u: u32 = u32::from(largest);

    // SAFETY:
    // - `f` is the PTX entry `kernel_name`; its 7-arg signature
    //   (in_ptr, vals_ptr, idx_ptr, outer, dim, k, largest) matches the args
    //   pushed below in order.
    // - `input` holds at least `outer * dim` `V`-elements (checked above); each
    //   thread `s in [0, outer)` reads `in[s*dim + i]` for `i in [0, dim)`,
    //   strictly in range.
    // - `out_vals` (V) and `out_idx` (i64) are fresh `outer*k`-element buffers
    //   (just allocated), the only `&mut` args; the kernel writes
    //   `out[s*k + j]` for `j in [0, k)`, all in range. They are distinct
    //   cudarc allocations and cannot alias `input` or each other.
    // - Threads with `s >= outer` exit via the leading `setp.ge.u32 %p_oob`.
    // - `outer`/`dim`/`k` are range-checked against `u32::MAX`, so the kernel's
    //   u32 index arithmetic cannot overflow.
    // - cudarc copies the by-reference `u32` params into the launch parameter
    //   buffer; their lifetime spans this synchronous frame. Stream sync is the
    //   caller's responsibility (matches the other kernel modules).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out_vals)
            .arg(&mut out_idx)
            .arg(&outer_u)
            .arg(&dim_u)
            .arg(&k_u)
            .arg(&largest_u)
            .launch(cfg)?;
    }

    Ok((out_vals, out_idx))
}

/// On-device `topk` over an f32 `[outer, dim]` buffer (last-dim selection).
///
/// Returns `(values, indices)` — a `CudaSlice<f32>` of `outer * k` extrema and
/// a `CudaSlice<i64>` of the matching original indices into `[0, dim)`, both
/// in sorted order (`largest` → descending, else ascending; ties broken by
/// ascending index). Mirrors the gather+`sortKeyValueInplace` contract of
/// `topk_out_cuda` in `aten/src/ATen/native/cuda/TensorTopK.cpp` for the
/// last-dim, sorted case.
///
/// # Errors
///
/// - [`GpuError::LengthMismatch`] when `k > dim`, the slice is shorter than
///   `outer * dim`, or a count exceeds `u32::MAX`.
/// - [`GpuError::PtxCompileFailed`] if the PTX module fails to compile.
/// - [`GpuError::Driver`] on launch failure.
pub fn gpu_topk_f32(
    input: &CudaSlice<f32>,
    outer: usize,
    dim: usize,
    k: usize,
    largest: bool,
    device: &GpuDevice,
) -> GpuResult<(CudaSlice<f32>, CudaSlice<i64>)> {
    launch_topk(
        input,
        outer,
        dim,
        k,
        largest,
        device,
        TOPK_F32_PTX,
        "topk_f32_kernel",
    )
}

/// On-device `topk` over an f64 `[outer, dim]` buffer. f64 counterpart of
/// [`gpu_topk_f32`].
///
/// # Errors
///
/// See [`gpu_topk_f32`].
pub fn gpu_topk_f64(
    input: &CudaSlice<f64>,
    outer: usize,
    dim: usize,
    k: usize,
    largest: bool,
    device: &GpuDevice,
) -> GpuResult<(CudaSlice<f64>, CudaSlice<i64>)> {
    launch_topk(
        input,
        outer,
        dim,
        k,
        largest,
        device,
        TOPK_F64_PTX,
        "topk_f64_kernel",
    )
}

// ===========================================================================
// histc — fixed-bin histogram with parallel atomic-add (#1545)
//
// One CUDA thread per input element. Each thread reads its value `v`, computes
// the destination bin, and `atom.global.add`s `1` into the bin counter. The
// output buffer is pre-zeroed by the launcher (`alloc_zeros`).
//
// Bin / range semantics are byte-for-byte from upstream
// `aten/src/ATen/native/cuda/SummaryOps.cu`:
//   getBin (SummaryOps.cu:41): bin = (int)((v - min) * nbins / (max - min))
//   SummaryOps.cu:47-48:       if (bin == nbins) bin -= 1;  // last bin [min,max]
//   kernelHistogram1D guard (SummaryOps.cu:92,118):
//                              only count when (v >= min && v <= max)
// The `(int)` cast truncates toward zero (matches C++ `(int)`); since the guard
// already forces `v >= min`, `(v - min) >= 0` so truncation == floor here. NaN
// values fail BOTH `v >= min` and `v <= max` (ordered compares are false for
// NaN), so they are skipped — matching torch (NaN is not counted).
//
// The counts are accumulated in the SAME float dtype as the input (PyTorch's
// `_histc_cuda` allocates the output with `self.scalar_type()`), so the f32
// kernel uses `atom.global.add.f32` (sm_20+) and the f64 kernel uses
// `atom.global.add.f64` (sm_60+ — the RTX 3090 is sm_86). Integer counts up to
// 2^24 (f32) / 2^53 (f64) are represented exactly, matching the CPU path which
// accumulates `T::one()` per element.
//
// ABI: (in_ptr, out_ptr, n, nbins, minv, maxv)
//   in   : V[n]
//   out  : V[nbins]   (pre-zeroed by the launcher)
//   n    : u32        (number of input elements)
//   nbins: u32
//   minv : V          (range lower bound, inclusive)
//   maxv : V          (range upper bound, inclusive)
// ===========================================================================

// f32 histogram. One thread per input value; `red.global.add.f32` (sm_20+)
// bumps the destination bin. `0f3F800000` is the f32 bit pattern for `1.0`.
const HISTC_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry histc_f32_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 nbins,
    .param .f32 minv,
    .param .f32 maxv
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %t, %nn, %nb, %bin, %bin1;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %v, %minv, %maxv, %range, %rel, %scaled, %nbf, %binf, %one;
    .reg .pred %p_oob, %p_lo, %p_hi, %p_in, %p_last;

    ld.param.u64 %in_p,  [in_ptr];
    ld.param.u64 %out_p, [out_ptr];
    ld.param.u32 %nn,    [n];
    ld.param.u32 %nb,    [nbins];
    ld.param.f32 %minv,  [minv];
    ld.param.f32 %maxv,  [maxv];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %t, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %p_oob, %t, %nn;
    @%p_oob bra DONE;

    // v = in[t]
    cvt.u64.u32 %off, %t;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %v, [%addr];

    // guard (SummaryOps.cu:92): count only when (v >= min && v <= max).
    // setp.ge/le are ORDERED (false for NaN) -> NaN skipped, matching torch.
    setp.ge.f32 %p_lo, %v, %minv;
    setp.le.f32 %p_hi, %v, %maxv;
    and.pred %p_in, %p_lo, %p_hi;
    @!%p_in bra DONE;

    // bin = (int)((v - min) * nbins / (max - min))   (SummaryOps.cu:41)
    sub.f32 %rel, %v, %minv;
    sub.f32 %range, %maxv, %minv;
    cvt.rn.f32.u32 %nbf, %nb;
    mul.f32 %scaled, %rel, %nbf;
    div.rn.f32 %binf, %scaled, %range;
    // truncate toward zero -> u32 bin. rel >= 0 here so trunc == floor.
    cvt.rzi.u32.f32 %bin, %binf;
    // if (bin == nbins) bin -= 1;  (SummaryOps.cu:47-48, last bin [min,max])
    setp.eq.u32 %p_last, %bin, %nb;
    sub.u32 %bin1, %bin, 1;
    @%p_last mov.u32 %bin, %bin1;

    // atomicAdd(&out[bin], 1.0f)
    cvt.u64.u32 %off, %bin;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    mov.f32 %one, 0f3F800000;
    red.global.add.f32 [%addr], %one;

DONE:
    ret;
}
";

// f64 histogram. Identical structure; 8-byte stride, .f64 compares, sm_60
// `red.global.add.f64`. `0d3FF0000000000000` is the f64 bit pattern for `1.0`.
const HISTC_F64_PTX: &str = "\
.version 7.0
.target sm_60
.address_size 64

.visible .entry histc_f64_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 nbins,
    .param .f64 minv,
    .param .f64 maxv
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %t, %nn, %nb, %bin, %bin1;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f64 %v, %minv, %maxv, %range, %rel, %scaled, %nbf, %binf, %one;
    .reg .pred %p_oob, %p_lo, %p_hi, %p_in, %p_last;

    ld.param.u64 %in_p,  [in_ptr];
    ld.param.u64 %out_p, [out_ptr];
    ld.param.u32 %nn,    [n];
    ld.param.u32 %nb,    [nbins];
    ld.param.f64 %minv,  [minv];
    ld.param.f64 %maxv,  [maxv];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %t, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %p_oob, %t, %nn;
    @%p_oob bra DONE;

    cvt.u64.u32 %off, %t;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %in_p, %off;
    ld.global.f64 %v, [%addr];

    setp.ge.f64 %p_lo, %v, %minv;
    setp.le.f64 %p_hi, %v, %maxv;
    and.pred %p_in, %p_lo, %p_hi;
    @!%p_in bra DONE;

    sub.f64 %rel, %v, %minv;
    sub.f64 %range, %maxv, %minv;
    cvt.rn.f64.u32 %nbf, %nb;
    mul.f64 %scaled, %rel, %nbf;
    div.rn.f64 %binf, %scaled, %range;
    cvt.rzi.u32.f64 %bin, %binf;
    setp.eq.u32 %p_last, %bin, %nb;
    sub.u32 %bin1, %bin, 1;
    @%p_last mov.u32 %bin, %bin1;

    cvt.u64.u32 %off, %bin;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out_p, %off;
    mov.f64 %one, 0d3FF0000000000000;
    red.global.add.f64 [%addr], %one;

DONE:
    ret;
}
";

fn launch_histc_config(n: usize) -> LaunchConfig {
    launch_1d(n)
}

/// Launch a histc kernel over a device-resident value buffer of `n` elements,
/// returning a fresh pre-zeroed `CudaSlice<V>` of `bins` counts. Each thread
/// `atom.global.add`s `1` into its element's bin.
///
/// `min_val`/`max_val` are the (inclusive) range bounds in the value dtype.
/// The caller guarantees `bins > 0` and `min_val < max_val` (the production
/// consumer rejects the degenerate cases before lowering to the GPU).
fn launch_histc<V>(
    input: &CudaSlice<V>,
    n: usize,
    bins: usize,
    min_val: V,
    max_val: V,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaSlice<V>>
where
    V: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Copy,
{
    if input.len() < n {
        return Err(GpuError::LengthMismatch {
            a: input.len(),
            b: n,
        });
    }
    if n > u32::MAX as usize || bins > u32::MAX as usize {
        return Err(GpuError::LengthMismatch {
            a: n.max(bins),
            b: u32::MAX as usize,
        });
    }

    let stream = device.stream();
    // Output is always `bins` long, pre-zeroed (the kernel only ever adds).
    let mut out = stream.alloc_zeros::<V>(bins)?;
    if n == 0 {
        // No values to bin — an all-zero `bins`-length buffer is the answer.
        return Ok(out);
    }

    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;

    let cfg = launch_histc_config(n);
    let n_u = n as u32;
    let bins_u = bins as u32;

    // SAFETY:
    // - `f` is the PTX entry `kernel_name`; its 6-arg signature
    //   (in_ptr, out_ptr, n, nbins, minv, maxv) matches the args pushed below
    //   in order.
    // - `input` holds at least `n` `V`-elements (checked above); thread
    //   `t in [0, n)` reads `in[t]`, strictly in range.
    // - `out` is a fresh pre-zeroed `bins`-element buffer (just allocated), the
    //   only `&mut` arg; the kernel writes only `out[bin]` for
    //   `bin in [0, bins)` (the `bin == nbins -> nbins-1` clamp keeps it in
    //   range), via `red.global.add` which needs no read-back. It cannot alias
    //   `input` (distinct cudarc allocation).
    // - Threads with `t >= n` exit via the leading `setp.ge.u32 %p_oob`.
    // - `n`/`bins` are range-checked against `u32::MAX`, so the kernel's u32
    //   index arithmetic cannot overflow.
    // - cudarc copies the by-reference scalar params into the launch parameter
    //   buffer; their lifetime spans this synchronous frame. Stream sync is the
    //   caller's responsibility (matches the other kernel modules).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&n_u)
            .arg(&bins_u)
            .arg(&min_val)
            .arg(&max_val)
            .launch(cfg)?;
    }

    Ok(out)
}

/// On-device `histc` over an f32 value buffer (#1545).
///
/// Returns a fresh device `CudaSlice<f32>` of `bins` counts. `out[b]` is the
/// number of `input` elements falling in bin `b`, where the half-open bins
/// `[min + b·w, min + (b+1)·w)` partition `[min, max]` and the LAST bin is
/// closed at both ends (so a value exactly `== max` lands in `bins-1`). Values
/// outside `[min, max]` (and NaN) are not counted. Mirrors `getBin` +
/// `kernelHistogram1D` in `aten/src/ATen/native/cuda/SummaryOps.cu:41-48,92`.
///
/// # Errors
///
/// - [`GpuError::LengthMismatch`] when the slice is shorter than `n` or a count
///   exceeds `u32::MAX`.
/// - [`GpuError::PtxCompileFailed`] if the PTX module fails to compile.
/// - [`GpuError::Driver`] on launch failure.
pub fn gpu_histc_f32(
    input: &CudaSlice<f32>,
    n: usize,
    bins: usize,
    min_val: f32,
    max_val: f32,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<f32>> {
    launch_histc(
        input,
        n,
        bins,
        min_val,
        max_val,
        device,
        HISTC_F32_PTX,
        "histc_f32_kernel",
    )
}

/// On-device `histc` over an f64 value buffer. f64 counterpart of
/// [`gpu_histc_f32`]; uses the sm_60 `red.global.add.f64` atomic.
///
/// # Errors
///
/// See [`gpu_histc_f32`].
pub fn gpu_histc_f64(
    input: &CudaSlice<f64>,
    n: usize,
    bins: usize,
    min_val: f64,
    max_val: f64,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<f64>> {
    launch_histc(
        input,
        n,
        bins,
        min_val,
        max_val,
        device,
        HISTC_F64_PTX,
        "histc_f64_kernel",
    )
}

// ===========================================================================
// meshgrid — pure index broadcast (`indexing='ij'`) (#1545)
//
// For N input 1-D coordinate vectors of lengths `shapes[0..N]`, output grid `d`
// has shape `shapes` (total = product) and `out[flat] = input_d[coord]` where
//   coord = (flat / inner_d) % shapes[d],   inner_d = product(shapes[d+1..N])
// This is exactly the `view(view_shape).expand(shape)` decomposition that
// upstream `meshgrid` uses (`aten/src/ATen/native/TensorShape.cpp:4462-4467`):
// axis `d`'s vector is reshaped to put its length at position `d` and broadcast
// (stride 0) along every other axis. One CUDA thread per output element does
// the index arithmetic and a single gather load — no `expand` materialisation
// of an intermediate strided tensor.
//
// ABI: (in_ptr, out_ptr, total, inner, axis_len)
//   in       : V[axis_len]        (the d-th coordinate vector)
//   out      : V[total]           (grid for axis d)
//   total    : u32                (product of all shapes)
//   inner    : u32                (product of shapes[d+1..N])
//   axis_len : u32                (shapes[d])
// ===========================================================================

// f32 meshgrid gather. One thread per output element.
const MESHGRID_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry meshgrid_f32_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 total,
    .param .u32 inner,
    .param .u32 axis_len
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %t, %tot, %inr, %al, %q, %coord;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %v;
    .reg .pred %p_oob;

    ld.param.u64 %in_p,  [in_ptr];
    ld.param.u64 %out_p, [out_ptr];
    ld.param.u32 %tot,   [total];
    ld.param.u32 %inr,   [inner];
    ld.param.u32 %al,    [axis_len];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %t, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %p_oob, %t, %tot;
    @%p_oob bra DONE;

    // coord = (flat / inner) % axis_len
    div.u32 %q, %t, %inr;
    rem.u32 %coord, %q, %al;

    // v = in[coord]; out[flat] = v
    cvt.u64.u32 %off, %coord;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %v, [%addr];

    cvt.u64.u32 %off, %t;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %v;

DONE:
    ret;
}
";

// f64 meshgrid gather. Identical structure; 8-byte stride.
const MESHGRID_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry meshgrid_f64_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 total,
    .param .u32 inner,
    .param .u32 axis_len
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %t, %tot, %inr, %al, %q, %coord;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f64 %v;
    .reg .pred %p_oob;

    ld.param.u64 %in_p,  [in_ptr];
    ld.param.u64 %out_p, [out_ptr];
    ld.param.u32 %tot,   [total];
    ld.param.u32 %inr,   [inner];
    ld.param.u32 %al,    [axis_len];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %t, %bid_r, %bdim_r, %tid_r;
    setp.ge.u32 %p_oob, %t, %tot;
    @%p_oob bra DONE;

    div.u32 %q, %t, %inr;
    rem.u32 %coord, %q, %al;

    cvt.u64.u32 %off, %coord;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %in_p, %off;
    ld.global.f64 %v, [%addr];

    cvt.u64.u32 %off, %t;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out_p, %off;
    st.global.f64 [%addr], %v;

DONE:
    ret;
}
";

/// Launch a meshgrid gather kernel producing the grid for ONE axis.
///
/// `input` is the axis's 1-D coordinate vector (length `axis_len`); the output
/// is a fresh `CudaSlice<V>` of `total` elements where
/// `out[flat] = input[(flat / inner) % axis_len]`. One thread per output
/// element.
fn launch_meshgrid<V>(
    input: &CudaSlice<V>,
    total: usize,
    inner: usize,
    axis_len: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaSlice<V>>
where
    V: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
{
    if input.len() < axis_len {
        return Err(GpuError::LengthMismatch {
            a: input.len(),
            b: axis_len,
        });
    }
    if total > u32::MAX as usize || inner > u32::MAX as usize || axis_len > u32::MAX as usize {
        return Err(GpuError::LengthMismatch {
            a: total.max(inner).max(axis_len),
            b: u32::MAX as usize,
        });
    }

    let stream = device.stream();
    let mut out = stream.alloc_zeros::<V>(total)?;
    if total == 0 {
        return Ok(out);
    }

    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;

    let cfg = launch_1d(total);
    let total_u = total as u32;
    let inner_u = inner.max(1) as u32;
    let axis_u = axis_len as u32;

    // SAFETY:
    // - `f` is the PTX entry `kernel_name`; its 5-arg signature
    //   (in_ptr, out_ptr, total, inner, axis_len) matches the args pushed below.
    // - `input` holds at least `axis_len` `V`-elements (checked above); the
    //   kernel reads `in[coord]` where `coord = (flat / inner) % axis_len`, so
    //   `coord in [0, axis_len)`, strictly in range.
    // - `out` is a fresh `total`-element buffer (just allocated), the only `&mut`
    //   arg; the kernel writes `out[flat]` for `flat in [0, total)`, in range,
    //   and cannot alias `input` (distinct cudarc allocation).
    // - Threads with `flat >= total` exit via the leading `setp.ge.u32 %p_oob`.
    // - `total`/`inner`/`axis_len` are range-checked against `u32::MAX`. `inner`
    //   is forced `>= 1` so the `div.u32` divisor is never zero.
    // - cudarc copies the by-reference `u32` params into the launch parameter
    //   buffer; their lifetime spans this synchronous frame. Stream sync is the
    //   caller's responsibility.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&total_u)
            .arg(&inner_u)
            .arg(&axis_u)
            .launch(cfg)?;
    }

    Ok(out)
}

/// On-device `meshgrid` grid for one axis over an f32 coordinate vector (#1545).
///
/// `input` is the axis's 1-D coordinate vector (length `axis_len`). Returns a
/// fresh device `CudaSlice<f32>` of `total` elements forming the broadcast grid
/// for this axis (`indexing='ij'`): `out[flat] = input[(flat / inner) %
/// axis_len]`, where `inner = product(shapes[axis+1..])` and `total =
/// product(shapes)`. Mirrors the `view(view_shape).expand(shape)` decomposition
/// of upstream `meshgrid` in `aten/src/ATen/native/TensorShape.cpp:4462-4467`.
///
/// # Errors
///
/// - [`GpuError::LengthMismatch`] when `input` is shorter than `axis_len` or a
///   count exceeds `u32::MAX`.
/// - [`GpuError::PtxCompileFailed`] if the PTX module fails to compile.
/// - [`GpuError::Driver`] on launch failure.
pub fn gpu_meshgrid_f32(
    input: &CudaSlice<f32>,
    total: usize,
    inner: usize,
    axis_len: usize,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<f32>> {
    launch_meshgrid(
        input,
        total,
        inner,
        axis_len,
        device,
        MESHGRID_F32_PTX,
        "meshgrid_f32_kernel",
    )
}

/// On-device `meshgrid` grid for one axis over an f64 coordinate vector. f64
/// counterpart of [`gpu_meshgrid_f32`].
///
/// # Errors
///
/// See [`gpu_meshgrid_f32`].
pub fn gpu_meshgrid_f64(
    input: &CudaSlice<f64>,
    total: usize,
    inner: usize,
    axis_len: usize,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<f64>> {
    launch_meshgrid(
        input,
        total,
        inner,
        axis_len,
        device,
        MESHGRID_F64_PTX,
        "meshgrid_f64_kernel",
    )
}

// ===========================================================================
// unique_consecutive — data-dependent run compaction (#1545)
//
// `torch.unique_consecutive` collapses each maximal RUN of equal adjacent
// elements into a single output element. The output length is DATA-DEPENDENT
// (unknown until the adjacency scan runs), which is the hard part on GPU. The
// pipeline keeps the VALUE DATA on-device end-to-end:
//
//   1. RUN_FLAG kernel: flag[i] = (i == 0 || in[i] != in[i-1]) ? 1 : 0 — a
//      device f32 buffer marking each run-start. `!=` uses `setp.neu` (the
//      UNORDERED not-equal) so NaN (where NaN != NaN) starts its own run,
//      matching the CPU path's `data[i] == data[i-1]` (PartialEq, false for
//      NaN). The ordered `setp.ne` returns FALSE for NaN operands and would
//      wrongly collapse consecutive NaNs into one run.
//   2. INCLUSIVE PREFIX SUM over the flags via the existing `gpu_cumsum`
//      primitive (one flat axis: outer=1, dim_size=n, inner=1). `incl[i]` is
//      the number of run-starts in `[0, i]`, so a run-start at `i` writes to
//      output slot `incl[i] - 1` and the total output length is `incl[n-1]`.
//   3. COMPACT kernel: each run-start `i` scatters `in[i]` to
//      `out[(u32)incl[i] - 1]` — the compacted values stay on-device.
//
// The launcher reads back ONLY the `incl` array (a length-`n` f32 buffer of
// derived run-position INDICES) to (a) size the output allocation, (b) build
// the host `inverse` vector (`inverse[i] = incl[i] - 1`), and (c) build the
// host `counts` vector (run lengths). This is NOT an R-CODE-4 value round trip:
// the VALUE data never leaves the device and returns — only the freshly
// computed integer run-position metadata is copied to host, exactly as
// `searchsorted_1d` reads back its i64 indices while the value/boundary data
// stays device-resident. The `inverse` / `counts` outputs are host `Vec<usize>`
// BY the CPU `ops::search::unique_consecutive` signature (they were never
// device tensors); the deduplicated VALUE tensor is the only result that stays
// GPU-resident.
//
// RUN_FLAG ABI: (in_ptr, flag_ptr, n)
//   in   : V[n]    (input values)
//   flag : f32[n]  (1.0 at run-starts, else 0.0)
//   n    : u32
// COMPACT ABI: (in_ptr, incl_ptr, out_ptr, n)
//   in   : V[n]      (input values)
//   incl : f32[n]    (inclusive prefix sum of the run-start flags)
//   out  : V[out_len](compacted run-start values)
//   n    : u32
// ===========================================================================

/// f32 run-start flag kernel. One thread per element.
const RUN_FLAG_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry run_flag_f32_kernel(
    .param .u64 in_ptr,
    .param .u64 flag_ptr,
    .param .u32 n
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %n_r;
    .reg .u64 %in_p, %flag_p, %off, %addr, %prev_addr;
    .reg .f32 %cur, %prev, %one, %zero;
    .reg .pred %p_oob, %p_first, %p_ne;

    ld.param.u64 %in_p,   [in_ptr];
    ld.param.u64 %flag_p, [flag_ptr];
    ld.param.u32 %n_r,    [n];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;

    setp.ge.u32 %p_oob, %idx, %n_r;
    @%p_oob bra DONE;

    mov.f32 %one,  0f3F800000;
    mov.f32 %zero, 0f00000000;

    // off = idx * 4 (f32 element stride)
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %cur, [%addr];

    // idx == 0 is always a run-start.
    setp.eq.u32 %p_first, %idx, 0;
    @%p_first bra WRITE_ONE;

    // prev = in[idx-1]
    sub.u64 %prev_addr, %addr, 4;
    ld.global.f32 %prev, [%prev_addr];

    // run-start iff cur != prev. setp.neu is the UNORDERED not-equal: NaN vs
    // NaN -> true and NaN vs finite -> true, so every NaN starts its own run
    // (matching the CPU `data[i] == data[i-1]` negation and torch). The ordered
    // setp.ne returns FALSE for NaN operands and would collapse consecutive NaNs.
    setp.neu.f32 %p_ne, %cur, %prev;
    @%p_ne bra WRITE_ONE;

    // not a run-start
    add.u64 %addr, %flag_p, %off;
    st.global.f32 [%addr], %zero;
    bra DONE;

WRITE_ONE:
    add.u64 %addr, %flag_p, %off;
    st.global.f32 [%addr], %one;

DONE:
    ret;
}
";

/// f64 run-start flag kernel. One thread per element. The input is f64 (8-byte
/// stride) but the flag output is f32 (4-byte), identical to the f32 variant.
const RUN_FLAG_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry run_flag_f64_kernel(
    .param .u64 in_ptr,
    .param .u64 flag_ptr,
    .param .u32 n
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %n_r;
    .reg .u64 %in_p, %flag_p, %ioff, %foff, %addr, %prev_addr;
    .reg .f64 %cur, %prev;
    .reg .f32 %one, %zero;
    .reg .pred %p_oob, %p_first, %p_ne;

    ld.param.u64 %in_p,   [in_ptr];
    ld.param.u64 %flag_p, [flag_ptr];
    ld.param.u32 %n_r,    [n];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;

    setp.ge.u32 %p_oob, %idx, %n_r;
    @%p_oob bra DONE;

    mov.f32 %one,  0f3F800000;
    mov.f32 %zero, 0f00000000;

    // ioff = idx * 8 (f64 input stride); foff = idx * 4 (f32 flag stride)
    cvt.u64.u32 %ioff, %idx;
    shl.b64 %ioff, %ioff, 3;
    cvt.u64.u32 %foff, %idx;
    shl.b64 %foff, %foff, 2;
    add.u64 %addr, %in_p, %ioff;
    ld.global.f64 %cur, [%addr];

    setp.eq.u32 %p_first, %idx, 0;
    @%p_first bra WRITE_ONE;

    sub.u64 %prev_addr, %addr, 8;
    ld.global.f64 %prev, [%prev_addr];

    // run-start iff cur != prev. setp.neu (unordered not-equal) makes every NaN
    // its own run (NaN vs NaN -> true), matching the CPU path and torch; the
    // ordered setp.ne returns FALSE for NaN and would collapse consecutive NaNs.
    setp.neu.f64 %p_ne, %cur, %prev;
    @%p_ne bra WRITE_ONE;

    add.u64 %addr, %flag_p, %foff;
    st.global.f32 [%addr], %zero;
    bra DONE;

WRITE_ONE:
    add.u64 %addr, %flag_p, %foff;
    st.global.f32 [%addr], %one;

DONE:
    ret;
}
";

/// f32 compaction scatter kernel. One thread per element; only run-starts
/// store. `out_pos = (u32)incl[idx] - 1`.
const COMPACT_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry compact_f32_kernel(
    .param .u64 in_ptr,
    .param .u64 incl_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %n_r, %pos;
    .reg .u64 %in_p, %incl_p, %out_p, %off, %addr, %prev_addr, %ooff;
    .reg .f32 %cur, %prev, %incl;
    .reg .pred %p_oob, %p_first, %p_ne;

    ld.param.u64 %in_p,   [in_ptr];
    ld.param.u64 %incl_p, [incl_ptr];
    ld.param.u64 %out_p,  [out_ptr];
    ld.param.u32 %n_r,    [n];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;

    setp.ge.u32 %p_oob, %idx, %n_r;
    @%p_oob bra DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %cur, [%addr];

    // Re-derive run-start from the data (same predicate as RUN_FLAG).
    setp.eq.u32 %p_first, %idx, 0;
    @%p_first bra DO_STORE;
    sub.u64 %prev_addr, %addr, 4;
    ld.global.f32 %prev, [%prev_addr];
    // setp.neu (unordered) must match RUN_FLAG exactly: NaN -> own run, else
    // out_len and the scatter positions disagree.
    setp.neu.f32 %p_ne, %cur, %prev;
    @%p_ne bra DO_STORE;
    bra DONE;

DO_STORE:
    // pos = (u32)incl[idx] - 1
    add.u64 %addr, %incl_p, %off;
    ld.global.f32 %incl, [%addr];
    cvt.rzi.u32.f32 %pos, %incl;
    sub.u32 %pos, %pos, 1;
    // out[pos] = cur
    cvt.u64.u32 %ooff, %pos;
    shl.b64 %ooff, %ooff, 2;
    add.u64 %addr, %out_p, %ooff;
    st.global.f32 [%addr], %cur;

DONE:
    ret;
}
";

/// f64 compaction scatter kernel. f64 input/output (8-byte stride); `incl` is
/// f32 (4-byte stride).
const COMPACT_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry compact_f64_kernel(
    .param .u64 in_ptr,
    .param .u64 incl_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %idx, %n_r, %pos;
    .reg .u64 %in_p, %incl_p, %out_p, %ioff, %foff, %addr, %prev_addr, %ooff;
    .reg .f64 %cur, %prev;
    .reg .f32 %incl;
    .reg .pred %p_oob, %p_first, %p_ne;

    ld.param.u64 %in_p,   [in_ptr];
    ld.param.u64 %incl_p, [incl_ptr];
    ld.param.u64 %out_p,  [out_ptr];
    ld.param.u32 %n_r,    [n];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %idx, %bid_r, %bdim_r, %tid_r;

    setp.ge.u32 %p_oob, %idx, %n_r;
    @%p_oob bra DONE;

    cvt.u64.u32 %ioff, %idx;
    shl.b64 %ioff, %ioff, 3;
    add.u64 %addr, %in_p, %ioff;
    ld.global.f64 %cur, [%addr];

    setp.eq.u32 %p_first, %idx, 0;
    @%p_first bra DO_STORE;
    sub.u64 %prev_addr, %addr, 8;
    ld.global.f64 %prev, [%prev_addr];
    // setp.neu (unordered) must match RUN_FLAG_F64 exactly: NaN -> own run, else
    // out_len and the scatter positions disagree.
    setp.neu.f64 %p_ne, %cur, %prev;
    @%p_ne bra DO_STORE;
    bra DONE;

DO_STORE:
    cvt.u64.u32 %foff, %idx;
    shl.b64 %foff, %foff, 2;
    add.u64 %addr, %incl_p, %foff;
    ld.global.f32 %incl, [%addr];
    cvt.rzi.u32.f32 %pos, %incl;
    sub.u32 %pos, %pos, 1;
    cvt.u64.u32 %ooff, %pos;
    shl.b64 %ooff, %ooff, 3;
    add.u64 %addr, %out_p, %ooff;
    st.global.f64 [%addr], %cur;

DONE:
    ret;
}
";

/// Launch a run-flag kernel over `input` (`n` elements) into a fresh f32 flag
/// buffer, then inclusive-prefix-sum the flags on-device.
///
/// Returns `(incl, out_len)` where `incl` is the device inclusive-scan buffer
/// (`incl[i]` = number of run-starts in `[0, i]`) and `out_len = incl[n-1]`
/// (the data-dependent number of unique consecutive runs). Caller guarantees
/// `n > 0`.
#[cfg(feature = "cuda")]
fn run_flags_and_scan(
    in_slice: &CudaSlice<impl cudarc::driver::DeviceRepr>,
    n: usize,
    device: &GpuDevice,
    flag_ptx: &'static str,
    flag_kernel: &'static str,
) -> GpuResult<CudaBuffer<f32>> {
    let stream = device.stream();
    let ctx = device.context();

    let mut flags = alloc_zeros_f32(n, device)?;
    let n_u = n as u32;

    let f = get_or_compile(ctx, flag_ptx, flag_kernel, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: flag_kernel,
            source: e,
        }
    })?;
    let block: u32 = 256;
    let grid = (n as u32).div_ceil(block).max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY:
    // - `f` is the run-flag PTX entry `flag_kernel`; its 3-arg ABI
    //   `(in_ptr, flag_ptr, n)` matches the args pushed below in order.
    // - `in_slice` holds `n` value elements; thread `idx in [0, n)` reads
    //   `in[idx]` (and, when `idx > 0`, `in[idx-1]`), strictly in range.
    // - `flags` is a fresh `n`-element f32 buffer (just allocated), the only
    //   `&mut` arg; the kernel writes exactly `flag[idx]` for `idx in [0, n)`.
    //   It cannot alias `in_slice` (distinct cudarc allocation).
    // - Threads with `idx >= n` exit via the leading `setp.ge.u32 %p_oob`.
    // - `n` is bounded below (`n <= incl.len()`); the launcher's caller
    //   range-checks `n` against `u32::MAX` before calling.
    // - cudarc copies the by-reference `n_u` into the launch parameter buffer;
    //   its lifetime spans this synchronous frame.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(in_slice)
            .arg(flags.inner_mut())
            .arg(&n_u)
            .launch(cfg)?;
    }

    // Inclusive prefix-sum the run-start flags on-device (flat axis).
    let incl = gpu_cumsum(&flags, 1, n, 1, device)?;
    Ok(incl)
}

/// Build the host `inverse` and `counts` vectors from an inclusive run-start
/// scan, and return the data-dependent output length.
///
/// `incl[i]` is the number of run-starts in `[0, i]`. The output index of
/// element `i` is `inverse[i] = incl[i] - 1`. `counts[j]` is the run length of
/// output run `j`, recovered by counting how many inputs map to each output
/// slot. `out_len = incl[n-1]`.
fn decode_runs(incl_host: &[f32]) -> (Vec<usize>, Vec<usize>, usize) {
    let n = incl_host.len();
    if n == 0 {
        return (vec![], vec![], 0);
    }
    let out_len = incl_host[n - 1] as usize;
    let mut inverse = vec![0usize; n];
    let mut counts = vec![0usize; out_len];
    for (i, &incl) in incl_host.iter().enumerate() {
        // `incl >= 1` for every i (idx 0 is always a run-start, so the scan is
        // monotone and starts at 1); `inv` is therefore in `[0, out_len)`.
        let inv = (incl as usize) - 1;
        inverse[i] = inv;
        counts[inv] += 1;
    }
    (inverse, counts, out_len)
}

/// On-device `unique_consecutive` over an f32 value buffer (#1545).
///
/// Returns `(values, inverse, counts)`:
/// - `values` — a fresh device `CudaBuffer<f32>` of `out_len` run-start values
///   (the deduplicated output, GPU-resident).
/// - `inverse` — host `Vec<usize>` of length `n`: each input's index in the
///   output (`torch.unique_consecutive(return_inverse=True)`).
/// - `counts` — host `Vec<usize>` of length `out_len`: the run length of each
///   output element (`return_counts=True`).
///
/// Mirrors the CPU `ferrotorch_core::ops::search::unique_consecutive` run
/// semantics exactly (NaN starts its own run because `NaN != NaN`). Only the
/// derived run-position metadata is read back to host; the VALUE data stays
/// on-device (no R-CODE-4 round trip).
///
/// # Errors
///
/// - [`GpuError::LengthMismatch`] when `n > u32::MAX`.
/// - [`GpuError::PtxCompileFailed`] if a PTX module fails to compile.
/// - [`GpuError::Driver`] on launch failure.
#[cfg(feature = "cuda")]
pub fn gpu_unique_consecutive_f32(
    input: &CudaBuffer<f32>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, Vec<usize>, Vec<usize>)> {
    if n == 0 {
        return Ok((alloc_zeros_f32(0, device)?, vec![], vec![]));
    }
    if n > u32::MAX as usize {
        return Err(GpuError::LengthMismatch {
            a: n,
            b: u32::MAX as usize,
        });
    }

    let incl = run_flags_and_scan(
        input.inner(),
        n,
        device,
        RUN_FLAG_F32_PTX,
        "run_flag_f32_kernel",
    )?;
    let incl_host = gpu_to_cpu(&incl, device)?;
    let (inverse, counts, out_len) = decode_runs(&incl_host);

    let mut out = alloc_zeros_f32(out_len, device)?;
    launch_compact_f32(input, &incl, &mut out, n, device)?;
    Ok((out, inverse, counts))
}

/// On-device `unique_consecutive` over an f64 value buffer. f64 counterpart of
/// [`gpu_unique_consecutive_f32`]; the run-start scan still runs in f32 (the
/// flags are 0/1, exact), only the value load/store width differs.
///
/// # Errors
///
/// See [`gpu_unique_consecutive_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_unique_consecutive_f64(
    input: &CudaBuffer<f64>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f64>, Vec<usize>, Vec<usize>)> {
    if n == 0 {
        return Ok((alloc_zeros_f64(0, device)?, vec![], vec![]));
    }
    if n > u32::MAX as usize {
        return Err(GpuError::LengthMismatch {
            a: n,
            b: u32::MAX as usize,
        });
    }

    // The run-start scan runs in f32 (the flags are exact 0/1); only the value
    // load/store width differs from the f32 path.
    let incl = run_flags_and_scan(
        input.inner(),
        n,
        device,
        RUN_FLAG_F64_PTX,
        "run_flag_f64_kernel",
    )?;
    let incl_host = gpu_to_cpu(&incl, device)?;
    let (inverse, counts, out_len) = decode_runs(&incl_host);

    let mut out = alloc_zeros_f64(out_len, device)?;
    launch_compact_f64(input, &incl, &mut out, n, device)?;
    Ok((out, inverse, counts))
}

/// Launch the f32 compaction scatter kernel.
#[cfg(feature = "cuda")]
fn launch_compact_f32(
    input: &CudaBuffer<f32>,
    incl: &CudaBuffer<f32>,
    out: &mut CudaBuffer<f32>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<()> {
    let stream = device.stream();
    let ctx = device.context();
    let f = get_or_compile(
        ctx,
        COMPACT_F32_PTX,
        "compact_f32_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "compact_f32_kernel",
        source: e,
    })?;
    let block: u32 = 256;
    let grid = (n as u32).div_ceil(block).max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u = n as u32;
    // SAFETY:
    // - `f` is the `compact_f32_kernel` PTX entry; its 4-arg ABI
    //   `(in_ptr, incl_ptr, out_ptr, n)` matches the args pushed below.
    // - `input` and `incl` each hold `n` f32 elements; thread `idx in [0, n)`
    //   reads `in[idx]` / `in[idx-1]` / `incl[idx]`, all in range.
    // - `out` is the only `&mut` arg, freshly allocated with `out_len`
    //   elements. Each run-start thread writes `out[(u32)incl[idx]-1]`; because
    //   `incl` is the monotone inclusive scan of the run-start flags (idx 0 is
    //   a run-start so `incl >= 1`), the write index lies in `[0, out_len)` and
    //   each output slot is written by exactly one run-start thread (no data
    //   race). `out` cannot alias `input`/`incl` (distinct allocations).
    // - Threads with `idx >= n` exit via the leading `setp.ge.u32 %p_oob`.
    // - cudarc copies `n_u` by reference into the launch buffer for this frame.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(incl.inner())
            .arg(out.inner_mut())
            .arg(&n_u)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch the f64 compaction scatter kernel.
#[cfg(feature = "cuda")]
fn launch_compact_f64(
    input: &CudaBuffer<f64>,
    incl: &CudaBuffer<f32>,
    out: &mut CudaBuffer<f64>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<()> {
    let stream = device.stream();
    let ctx = device.context();
    let f = get_or_compile(
        ctx,
        COMPACT_F64_PTX,
        "compact_f64_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "compact_f64_kernel",
        source: e,
    })?;
    let block: u32 = 256;
    let grid = (n as u32).div_ceil(block).max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u = n as u32;
    // SAFETY: identical contract to `launch_compact_f32`; the only differences
    // are the f64 value load/store width (8 bytes) and the f64 `compact_f64`
    // entry. `incl` is still an f32 inclusive scan of `n` run-start flags;
    // `out` is a fresh `out_len`-element f64 buffer, each slot written once.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(incl.inner())
            .arg(out.inner_mut())
            .arg(&n_u)
            .launch(cfg)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transfer::cpu_to_gpu;

    /// Read a device-resident `CudaSlice<i64>` back to a host `Vec<i64>`,
    /// truncated to its logical length. Only the result indices are read; the
    /// value/boundary data never leaves the device.
    fn read_i64(slice: &CudaSlice<i64>, device: &GpuDevice) -> Vec<i64> {
        let n = slice.len();
        let mut v = device.stream().clone_dtoh(slice).unwrap();
        v.truncate(n);
        v
    }

    /// CPU reference matching the GPU half-open comparisons exactly.
    fn cpu_searchsorted_ref(bounds: &[f64], vals: &[f64], right: bool) -> Vec<i64> {
        vals.iter()
            .map(|&v| {
                if right {
                    // upper_bound: first i with bounds[i] > v
                    bounds.partition_point(|&b| b <= v) as i64
                } else {
                    // lower_bound: first i with bounds[i] >= v
                    bounds.partition_point(|&b| b < v) as i64
                }
            })
            .collect()
    }

    #[test]
    fn searchsorted_f32_left_and_right_match_cpu() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let bounds = [1.0f32, 3.0, 5.0, 7.0];
        let vals = [0.0f32, 2.0, 3.0, 6.0, 8.0];
        let bg = cpu_to_gpu(&bounds, &device).unwrap();
        let vg = cpu_to_gpu(&vals, &device).unwrap();

        // right=true (upper_bound): [0,2,3,6,8] -> [0,1,2,3,4]
        let og = gpu_searchsorted_f32(
            vg.inner(),
            bg.inner(),
            vals.len(),
            bounds.len(),
            true,
            &device,
        )
        .unwrap();
        // The result lives on device — this IS the GPU buffer.
        assert_eq!(og.len(), vals.len());
        let got = read_i64(&og, &device);
        let bounds64: Vec<f64> = bounds.iter().map(|&x| x as f64).collect();
        let vals64: Vec<f64> = vals.iter().map(|&x| x as f64).collect();
        assert_eq!(got, cpu_searchsorted_ref(&bounds64, &vals64, true));
        assert_eq!(got, vec![0, 1, 2, 3, 4]);

        // right=false (lower_bound) on the same data.
        let og2 = gpu_searchsorted_f32(
            vg.inner(),
            bg.inner(),
            vals.len(),
            bounds.len(),
            false,
            &device,
        )
        .unwrap();
        let got2 = read_i64(&og2, &device);
        assert_eq!(got2, cpu_searchsorted_ref(&bounds64, &vals64, false));
    }

    #[test]
    fn searchsorted_f32_boundary_tie_left_vs_right() {
        // The bug-prone case: every value lands exactly ON a boundary.
        // left  -> that boundary's own index; right -> one past it.
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let bounds = [1.0f32, 3.0, 5.0, 7.0];
        let vals = [1.0f32, 3.0, 5.0, 7.0];
        let bg = cpu_to_gpu(&bounds, &device).unwrap();
        let vg = cpu_to_gpu(&vals, &device).unwrap();

        let left = gpu_searchsorted_f32(
            vg.inner(),
            bg.inner(),
            vals.len(),
            bounds.len(),
            false,
            &device,
        )
        .unwrap();
        let left_h = read_i64(&left, &device);
        assert_eq!(left_h, vec![0, 1, 2, 3]); // value on boundary -> its index

        let right = gpu_searchsorted_f32(
            vg.inner(),
            bg.inner(),
            vals.len(),
            bounds.len(),
            true,
            &device,
        )
        .unwrap();
        let right_h = read_i64(&right, &device);
        assert_eq!(right_h, vec![1, 2, 3, 4]); // value on boundary -> after it
    }

    #[test]
    fn searchsorted_f32_empty_boundaries_all_zero() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let bounds: [f32; 0] = [];
        let vals = [1.0f32, 2.0];
        let bg = cpu_to_gpu(&bounds, &device).unwrap();
        let vg = cpu_to_gpu(&vals, &device).unwrap();
        let og =
            gpu_searchsorted_f32(vg.inner(), bg.inner(), vals.len(), 0, true, &device).unwrap();
        let got = read_i64(&og, &device);
        assert_eq!(got, vec![0, 0]);
    }

    #[test]
    fn searchsorted_f64_matches_cpu() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let bounds = [-2.5f64, 0.0, 0.0, 4.25, 9.0];
        let vals = [-3.0f64, -2.5, 0.0, 1.0, 9.0, 100.0];
        let bg = cpu_to_gpu(&bounds, &device).unwrap();
        let vg = cpu_to_gpu(&vals, &device).unwrap();

        for right in [false, true] {
            let og = gpu_searchsorted_f64(
                vg.inner(),
                bg.inner(),
                vals.len(),
                bounds.len(),
                right,
                &device,
            )
            .unwrap();
            let got = read_i64(&og, &device);
            assert_eq!(got, cpu_searchsorted_ref(&bounds, &vals, right));
        }
    }

    // --- topk ---

    /// Read a device-resident `CudaSlice<f32>` back to a host `Vec<f32>`.
    fn read_f32(slice: &CudaSlice<f32>, device: &GpuDevice) -> Vec<f32> {
        let n = slice.len();
        let mut v = device.stream().clone_dtoh(slice).unwrap();
        v.truncate(n);
        v
    }

    fn read_f64(slice: &CudaSlice<f64>, device: &GpuDevice) -> Vec<f64> {
        let n = slice.len();
        let mut v = device.stream().clone_dtoh(slice).unwrap();
        v.truncate(n);
        v
    }

    /// CPU reference identical to `ferrotorch_core::ops::search::topk` — a
    /// stable sort by value (descending for `largest`, else ascending) with the
    /// resulting ascending-index tie-break. This is a valid `torch.topk`
    /// result (upstream sorts the gathered top-k with `stable=false`, leaving
    /// the per-tie index order unspecified) and is exactly what the production
    /// CPU path produces, so the GPU kernel must reproduce it bit-for-bit.
    fn cpu_topk_ref(
        data: &[f64],
        outer: usize,
        dim: usize,
        k: usize,
        largest: bool,
    ) -> (Vec<f64>, Vec<i64>) {
        let mut vals = Vec::with_capacity(outer * k);
        let mut idxs = Vec::with_capacity(outer * k);
        for o in 0..outer {
            let slice = &data[o * dim..(o + 1) * dim];
            let mut idx: Vec<usize> = (0..dim).collect();
            // Stable sort by value; equal values keep ascending index order.
            if largest {
                idx.sort_by(|&a, &b| slice[b].partial_cmp(&slice[a]).unwrap());
            } else {
                idx.sort_by(|&a, &b| slice[a].partial_cmp(&slice[b]).unwrap());
            }
            for &i in &idx[..k] {
                vals.push(slice[i]);
                idxs.push(i as i64);
            }
        }
        (vals, idxs)
    }

    #[test]
    fn topk_f32_largest_matches_cpu_ref() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let data = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0];
        let g = cpu_to_gpu(&data, &device).unwrap();
        let (vals, idx) = gpu_topk_f32(g.inner(), 1, 6, 3, true, &device).unwrap();
        // Result buffers live on device.
        assert_eq!(vals.len(), 3);
        assert_eq!(idx.len(), 3);
        let gv = read_f32(&vals, &device);
        let gi = read_i64(&idx, &device);
        assert_eq!(gv, vec![9.0, 5.0, 4.0]);
        assert_eq!(gi, vec![5, 4, 2]);
        let data64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
        let (rv, ri) = cpu_topk_ref(&data64, 1, 6, 3, true);
        assert_eq!(gv.iter().map(|&x| x as f64).collect::<Vec<_>>(), rv);
        assert_eq!(gi, ri);
    }

    #[test]
    fn topk_f32_smallest_matches_cpu_ref() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let data = [3.0f32, 1.0, 4.0, 1.0, 5.0];
        let g = cpu_to_gpu(&data, &device).unwrap();
        let (vals, idx) = gpu_topk_f32(g.inner(), 1, 5, 2, false, &device).unwrap();
        let gv = read_f32(&vals, &device);
        let gi = read_i64(&idx, &device);
        // Two ties at value 1.0 (indices 1 and 3) -> ascending index tie-break.
        assert_eq!(gv, vec![1.0, 1.0]);
        assert_eq!(gi, vec![1, 3]);
    }

    #[test]
    fn topk_f32_ties_ascending_index() {
        // The bug-prone case: many equal values. Both CPU and GPU must pick
        // them in ascending original-index order (a valid torch topk result).
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let data = [2.0f32, 2.0, 2.0, 2.0, 1.0];
        let g = cpu_to_gpu(&data, &device).unwrap();
        let (vals, idx) = gpu_topk_f32(g.inner(), 1, 5, 3, true, &device).unwrap();
        let gv = read_f32(&vals, &device);
        let gi = read_i64(&idx, &device);
        assert_eq!(gv, vec![2.0, 2.0, 2.0]);
        assert_eq!(gi, vec![0, 1, 2]); // ascending index among ties
        let data64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
        let (_, ri) = cpu_topk_ref(&data64, 1, 5, 3, true);
        assert_eq!(gi, ri);
    }

    #[test]
    fn topk_f32_multi_row() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        // [2, 4] -> per-row top-2 largest
        let data = [1.0f32, 5.0, 3.0, 2.0, 8.0, 0.0, 7.0, 6.0];
        let g = cpu_to_gpu(&data, &device).unwrap();
        let (vals, idx) = gpu_topk_f32(g.inner(), 2, 4, 2, true, &device).unwrap();
        let gv = read_f32(&vals, &device);
        let gi = read_i64(&idx, &device);
        assert_eq!(gv, vec![5.0, 3.0, 8.0, 7.0]);
        assert_eq!(gi, vec![1, 2, 0, 2]);
        let data64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
        let (rv, ri) = cpu_topk_ref(&data64, 2, 4, 2, true);
        assert_eq!(gv.iter().map(|&x| x as f64).collect::<Vec<_>>(), rv);
        assert_eq!(gi, ri);
    }

    #[test]
    fn topk_f32_k_equals_dim() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let data = [3.0f32, 1.0, 2.0];
        let g = cpu_to_gpu(&data, &device).unwrap();
        let (vals, idx) = gpu_topk_f32(g.inner(), 1, 3, 3, true, &device).unwrap();
        let gv = read_f32(&vals, &device);
        let gi = read_i64(&idx, &device);
        assert_eq!(gv, vec![3.0, 2.0, 1.0]);
        assert_eq!(gi, vec![0, 2, 1]);
    }

    #[test]
    fn topk_f64_matches_cpu_ref() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let data = [3.0f64, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0];
        let g = cpu_to_gpu(&data, &device).unwrap();
        for largest in [true, false] {
            let (vals, idx) = gpu_topk_f64(g.inner(), 1, 8, 4, largest, &device).unwrap();
            let gv = read_f64(&vals, &device);
            let gi = read_i64(&idx, &device);
            let (rv, ri) = cpu_topk_ref(&data, 1, 8, 4, largest);
            assert_eq!(gv, rv);
            assert_eq!(gi, ri);
        }
    }

    // --- histc ---

    /// CPU reference for `torch.histc` bin assignment, byte-for-byte from
    /// `aten/src/ATen/native/cuda/SummaryOps.cu` getBin (`SummaryOps.cu:41`) +
    /// the last-bin clamp (`:47-48`) + the `[min,max]` guard (`:92`). Counts are
    /// `f64` so the comparison is exact for the integer counts in these tests.
    fn cpu_histc_ref(data: &[f64], bins: usize, min: f64, max: f64) -> Vec<f64> {
        let mut counts = vec![0.0f64; bins];
        let range = max - min;
        for &v in data {
            if !(v >= min && v <= max) {
                continue; // out-of-range / NaN -> skipped (torch)
            }
            let mut bin = ((v - min) * bins as f64 / range) as i64;
            if bin == bins as i64 {
                bin -= 1;
            }
            counts[bin as usize] += 1.0;
        }
        counts
    }

    #[test]
    fn histc_f32_matches_torch_bins() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        // torch.histc(tensor([0,1,2,3,4,5,6,7,8,9,10.]), bins=5, min=0, max=10)
        //   live torch 2.x: [2., 2., 2., 2., 3.]  (10 lands in the last bin,
        //   which is closed at both ends -> bins-1).
        let data = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let g = cpu_to_gpu(&data, &device).unwrap();
        let out = gpu_histc_f32(g.inner(), data.len(), 5, 0.0, 10.0, &device).unwrap();
        // result lives on device — this IS the GPU buffer.
        assert_eq!(out.len(), 5);
        let got = read_f32(&out, &device);
        assert_eq!(got, vec![2.0, 2.0, 2.0, 2.0, 3.0]);
        // and matches the upstream-getBin CPU reference bit-for-bit.
        let data64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
        let want = cpu_histc_ref(&data64, 5, 0.0, 10.0);
        assert_eq!(got.iter().map(|&x| x as f64).collect::<Vec<_>>(), want);
    }

    #[test]
    fn histc_f32_skips_out_of_range_and_nan() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        // Values below min, above max, and NaN are all dropped (torch).
        // In-range [0,4]: 0.5->bin0, 1.5->bin1, 2.5->bin2, 4.0->bin3 (last,
        // closed). torch.histc(tensor([-1,0.5,1.5,2.5,4.0,5.0,nan]),4,0,4)
        //   = [1., 1., 1., 1.].
        let data = [-1.0f32, 0.5, 1.5, 2.5, 4.0, 5.0, f32::NAN];
        let g = cpu_to_gpu(&data, &device).unwrap();
        let out = gpu_histc_f32(g.inner(), data.len(), 4, 0.0, 4.0, &device).unwrap();
        let got = read_f32(&out, &device);
        assert_eq!(got, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn histc_f64_matches_torch_bins() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let data = [0.0f64, 0.25, 0.5, 0.75, 1.0];
        let g = cpu_to_gpu(&data, &device).unwrap();
        // torch.histc(tensor([0,.25,.5,.75,1.],dtype=float64),bins=4,min=0,max=1)
        //   = [1., 1., 1., 2.]  (1.0 in the closed last bin).
        let out = gpu_histc_f64(g.inner(), data.len(), 4, 0.0, 1.0, &device).unwrap();
        let got = read_f64(&out, &device);
        assert_eq!(got, vec![1.0, 1.0, 1.0, 2.0]);
        assert_eq!(got, cpu_histc_ref(&data, 4, 0.0, 1.0));
    }

    // --- meshgrid ---

    /// CPU reference for the per-axis `meshgrid` grid (`indexing='ij'`):
    /// `out[flat] = vec[(flat / inner) % axis_len]`, matching the
    /// `view(view_shape).expand(shape)` decomposition of upstream `meshgrid`
    /// (`aten/src/ATen/native/TensorShape.cpp:4462-4467`).
    fn cpu_meshgrid_axis(vec: &[f64], shapes: &[usize], axis: usize) -> Vec<f64> {
        let total: usize = shapes.iter().product();
        let inner: usize = shapes[axis + 1..].iter().product();
        (0..total)
            .map(|flat| vec[(flat / inner.max(1)) % shapes[axis]])
            .collect()
    }

    #[test]
    fn meshgrid_f32_two_axis_ij() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        // meshgrid([1,2,3],[4,5], indexing='ij'):
        //   grid0 (shape [3,2]) = [[1,1],[2,2],[3,3]] flat [1,1,2,2,3,3]
        //   grid1 (shape [3,2]) = [[4,5],[4,5],[4,5]] flat [4,5,4,5,4,5]
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0];
        let shapes = [3usize, 2];
        let ga = cpu_to_gpu(&a, &device).unwrap();
        let gb = cpu_to_gpu(&b, &device).unwrap();
        let total = 6;

        // axis 0: inner = shapes[1] = 2
        let g0 = gpu_meshgrid_f32(ga.inner(), total, 2, 3, &device).unwrap();
        assert_eq!(g0.len(), total);
        let h0 = read_f32(&g0, &device);
        assert_eq!(h0, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);

        // axis 1: inner = 1 (last axis)
        let g1 = gpu_meshgrid_f32(gb.inner(), total, 1, 2, &device).unwrap();
        let h1 = read_f32(&g1, &device);
        assert_eq!(h1, vec![4.0, 5.0, 4.0, 5.0, 4.0, 5.0]);

        // matches the upstream-decomposition CPU reference bit-for-bit.
        let a64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
        let b64: Vec<f64> = b.iter().map(|&x| x as f64).collect();
        assert_eq!(
            h0.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            cpu_meshgrid_axis(&a64, &shapes, 0)
        );
        assert_eq!(
            h1.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            cpu_meshgrid_axis(&b64, &shapes, 1)
        );
    }

    #[test]
    fn meshgrid_f64_three_axis_ij() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        // shapes [2,3,2], total = 12.
        let a = [10.0f64, 20.0];
        let shapes = [2usize, 3, 2];
        let total = 12;
        let ga = cpu_to_gpu(&a, &device).unwrap();
        // axis 0: inner = shapes[1]*shapes[2] = 6.
        let g0 = gpu_meshgrid_f64(ga.inner(), total, 6, 2, &device).unwrap();
        let h0 = read_f64(&g0, &device);
        assert_eq!(h0, cpu_meshgrid_axis(&a, &shapes, 0));
        // first 6 elements come from a[0]=10, next 6 from a[1]=20.
        assert_eq!(
            h0,
            vec![
                10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0
            ]
        );
    }
}
