//! GPU `searchsorted` / `bucketize` binary-search kernel (f32 / f64).
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

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::module_cache::get_or_compile;

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
    .reg .pred %p_lt, %p_le, %p_a, %p_b;

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

    // advance predicate (no `selp.pred`; build it with predicate logic):
    //   left : bv <  v   (lower_bound advances while !(bv >= v))
    //   right: bv <= v   (upper_bound advances while !(bv >  v))
    //   p_adv = (right & (bv <= v)) | (!right & (bv < v))
    setp.lt.f32 %p_lt, %bv, %v;
    setp.le.f32 %p_le, %bv, %v;
    and.pred %p_a, %p_is_right, %p_le;
    and.pred %p_b, %p_not_right, %p_lt;
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
    .reg .pred %p_lt, %p_le, %p_a, %p_b;

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

    setp.lt.f64 %p_lt, %bv, %v;
    setp.le.f64 %p_le, %bv, %v;
    and.pred %p_a, %p_is_right, %p_le;
    and.pred %p_b, %p_not_right, %p_lt;
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
}
