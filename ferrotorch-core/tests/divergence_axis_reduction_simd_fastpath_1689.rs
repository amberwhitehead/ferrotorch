//! #1689: CORRECTNESS guard for the `[outer, axis, inner]` SIMD fast path that
//! replaced the per-element odometer scan in the CPU `sum_dim` / `mean_dim`
//! forward (`ferrotorch-core/src/grad_fns/reduction.rs`,
//! `fn reduce_axis_sum_contiguous`). The fast path has two regimes:
//!   * `inner > 1`  — lane-add down the reduced axis into contiguous output
//!     lanes (regime 1, torch's `vectorized_outer_reduction`).
//!   * `inner == 1` — horizontal SIMD sum of each contiguous last-dim slice via
//!     `simd_reduce::sum_f32` / `sum_f64` (regime 2, torch's
//!     `vectorized_inner_reduction`).
//!
//! The risk a perf rewrite introduces is a WRONG slot mapping that the square
//! `[1000,1000]` benchmark cannot detect. This file pins:
//!   * the large `[1000,1000]` dim=0 (inner>1) and dim=1 (inner==1) cases — the
//!     two perf-target shapes,
//!   * a 3-D `[8,16,32]` reducing EACH of dim=0 (outer=1, inner=512),
//!     dim=1 (outer=8, inner=32 — the middle-dim probe), dim=2 (inner=1) —
//!     exercising outer>1, inner>1, and inner==1 plus the axis-in-the-middle
//!     indexing,
//!     for both f32 and f64, keepdim true and false.
//!
//! Every expected value is the CLOSED-FORM of `torch.sum` / `torch.mean` over a
//! `torch.arange(numel).reshape(shape)` input: torch's reduction of a
//! row-major arange along an axis is an arithmetic-series sum, so the reference
//! is the upstream torch value computed analytically (NOT copied from the
//! ferrotorch side — R-CHAR-3). For `x[o,a,i] = (o*axis + a)*inner + i` the sum
//! over `a in 0..axis` is
//!   `axis*(o*axis + i)*1  ... ` — derived per-case below in `expected_sum`.
//! Using integer-valued arange means the f64 reference is bit-exact and the f32
//! reference is exact for the magnitudes used here.

use ferrotorch_core::grad_fns::reduction::{mean_dim, sum_dim};
use ferrotorch_core::{Tensor, TensorStorage};

/// Closed-form `torch.sum(arange(shape), dim)` value at output coordinate
/// `(o, i)` of the `[outer, axis, inner]` decomposition. Input element
/// `(o, a, i)` is `(o*axis + a)*inner + i`; summing over `a in 0..axis`:
///   `Σ_a [(o*axis + a)*inner + i]`
///   `= inner * Σ_a (o*axis + a) + axis*i`
///   `= inner * (axis*o*axis + axis*(axis-1)/2) + axis*i`.
fn expected_sum_at(o: usize, i: usize, outer: usize, axis: usize, inner: usize) -> f64 {
    let _ = outer;
    let o = o as f64;
    let a = axis as f64;
    let inn = inner as f64;
    let i = i as f64;
    inn * (a * o * a + a * (a - 1.0) / 2.0) + a * i
}

fn decompose(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let outer: usize = shape[..dim].iter().product();
    let axis: usize = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    (outer, axis, inner)
}

fn arange_f32(shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let v: Vec<f32> = (0..numel).map(|i| i as f32).collect();
    Tensor::from_storage(TensorStorage::cpu(v), shape.to_vec(), false).expect("arange f32")
}

fn arange_f64(shape: &[usize]) -> Tensor<f64> {
    let numel: usize = shape.iter().product();
    let v: Vec<f64> = (0..numel).map(|i| i as f64).collect();
    Tensor::from_storage(TensorStorage::cpu(v), shape.to_vec(), false).expect("arange f64")
}

/// Build the full expected `sum` buffer (row-major over `[outer, inner]`).
fn expected_sum_buffer(shape: &[usize], dim: usize) -> Vec<f64> {
    let (outer, axis, inner) = decompose(shape, dim);
    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            out.push(expected_sum_at(o, i, outer, axis, inner));
        }
    }
    out
}

fn expected_out_shape(shape: &[usize], dim: usize, keepdim: bool) -> Vec<usize> {
    let mut s = shape.to_vec();
    if keepdim {
        s[dim] = 1;
    } else {
        s.remove(dim);
    }
    s
}

/// Verify `sum_dim` (f64) over `shape` reducing `dim`, both keepdim toggles,
/// against the closed-form torch reference. f64 arange sums are bit-exact.
fn check_sum_f64(shape: &[usize], dim: usize) {
    let exp = expected_sum_buffer(shape, dim);
    for &keepdim in &[false, true] {
        let x = arange_f64(shape);
        let out = sum_dim(&x, dim as i64, keepdim).expect("sum_dim f64");
        assert_eq!(
            out.shape(),
            expected_out_shape(shape, dim, keepdim).as_slice(),
            "sum_dim f64 shape: shape={shape:?} dim={dim} keepdim={keepdim}"
        );
        let got = out.data().expect("read");
        for (idx, (&g, &e)) in got.iter().zip(exp.iter()).enumerate() {
            assert_eq!(
                g, e,
                "sum_dim f64 VALUE: shape={shape:?} dim={dim} keepdim={keepdim} idx={idx}"
            );
        }
    }
}

/// Verify `sum_dim` (f32) over `shape` reducing `dim` against the closed-form
/// torch reference. Tolerance 1e-5 relative (the SIMD multi-accumulator order
/// is torch-adjacent, not byte-exact; the task's f32 envelope is 1e-5).
fn check_sum_f32(shape: &[usize], dim: usize) {
    let exp = expected_sum_buffer(shape, dim);
    for &keepdim in &[false, true] {
        let x = arange_f32(shape);
        let out = sum_dim(&x, dim as i64, keepdim).expect("sum_dim f32");
        assert_eq!(
            out.shape(),
            expected_out_shape(shape, dim, keepdim).as_slice(),
            "sum_dim f32 shape: shape={shape:?} dim={dim} keepdim={keepdim}"
        );
        let got = out.data().expect("read");
        for (idx, (&g, &e)) in got.iter().zip(exp.iter()).enumerate() {
            let e = e as f32;
            let tol = 1e-5 * e.abs().max(1.0);
            assert!(
                (g - e).abs() <= tol,
                "sum_dim f32 VALUE: shape={shape:?} dim={dim} keepdim={keepdim} idx={idx} torch={e} ferro={g}"
            );
        }
    }
}

/// Verify `mean_dim` (f64) — `sum / axis` — over `shape` reducing `dim`.
fn check_mean_f64(shape: &[usize], dim: usize) {
    let (_outer, axis, _inner) = decompose(shape, dim);
    let exp = expected_sum_buffer(shape, dim);
    for &keepdim in &[false, true] {
        let x = arange_f64(shape);
        let out = mean_dim(&x, dim as i64, keepdim).expect("mean_dim f64");
        assert_eq!(
            out.shape(),
            expected_out_shape(shape, dim, keepdim).as_slice(),
            "mean_dim f64 shape: shape={shape:?} dim={dim} keepdim={keepdim}"
        );
        let got = out.data().expect("read");
        for (idx, (&g, &e)) in got.iter().zip(exp.iter()).enumerate() {
            let mean_ref = e / axis as f64;
            assert!(
                (g - mean_ref).abs() < 1e-10,
                "mean_dim f64 VALUE: shape={shape:?} dim={dim} keepdim={keepdim} idx={idx} torch={mean_ref} ferro={g}"
            );
        }
    }
}

/// Verify `mean_dim` (f32) — `sum / axis` — over `shape` reducing `dim`.
fn check_mean_f32(shape: &[usize], dim: usize) {
    let (_outer, axis, _inner) = decompose(shape, dim);
    let exp = expected_sum_buffer(shape, dim);
    for &keepdim in &[false, true] {
        let x = arange_f32(shape);
        let out = mean_dim(&x, dim as i64, keepdim).expect("mean_dim f32");
        assert_eq!(
            out.shape(),
            expected_out_shape(shape, dim, keepdim).as_slice(),
            "mean_dim f32 shape: shape={shape:?} dim={dim} keepdim={keepdim}"
        );
        let got = out.data().expect("read");
        for (idx, (&g, &e)) in got.iter().zip(exp.iter()).enumerate() {
            let mean_ref = (e / axis as f64) as f32;
            let tol = 1e-5 * mean_ref.abs().max(1.0);
            assert!(
                (g - mean_ref).abs() <= tol,
                "mean_dim f32 VALUE: shape={shape:?} dim={dim} keepdim={keepdim} idx={idx} torch={mean_ref} ferro={g}"
            );
        }
    }
}

/// The large `[1000,1000]` perf-target shape, dim=0 (inner=1000>1, regime 1)
/// and dim=1 (inner=1, regime 2 SIMD horizontal sum), f64 — bit-exact.
#[test]
fn sum_dim_1000x1000_f64_both_dims() {
    check_sum_f64(&[1000, 1000], 0);
    check_sum_f64(&[1000, 1000], 1);
}

/// Same `[1000,1000]` shape, f32 (1e-5 envelope).
#[test]
fn sum_dim_1000x1000_f32_both_dims() {
    check_sum_f32(&[1000, 1000], 0);
    check_sum_f32(&[1000, 1000], 1);
}

/// `mean_dim` on the `[1000,1000]` perf shape, both dims, f64 + f32.
#[test]
fn mean_dim_1000x1000_both_dims() {
    check_mean_f64(&[1000, 1000], 0);
    check_mean_f64(&[1000, 1000], 1);
    check_mean_f32(&[1000, 1000], 0);
    check_mean_f32(&[1000, 1000], 1);
}

/// The 3-D `[8,16,32]` probe: dim=0 (outer=1, inner=512), dim=1 (outer=8,
/// inner=32 — the MIDDLE-dim, key [outer,axis,inner] indexing probe), dim=2
/// (inner=1, regime 2). f64 bit-exact for sum and mean.
#[test]
fn reduce_8x16x32_each_dim_f64() {
    for dim in 0..3 {
        check_sum_f64(&[8, 16, 32], dim);
        check_mean_f64(&[8, 16, 32], dim);
    }
}

/// The same 3-D `[8,16,32]` probe in f32 (1e-5 envelope), every dim.
#[test]
fn reduce_8x16x32_each_dim_f32() {
    for dim in 0..3 {
        check_sum_f32(&[8, 16, 32], dim);
        check_mean_f32(&[8, 16, 32], dim);
    }
}
