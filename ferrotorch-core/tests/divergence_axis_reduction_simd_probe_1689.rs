//! #1689 RE-AUDIT (discriminator probe): the CPU `sum_dim` / `mean_dim` forward
//! was rewritten from a per-element odometer scan to an `[outer, axis, inner]`
//! SIMD fast path (`reduce_axis_sum_contiguous` +
//! `simd_reduce::sum_f32`/`sum_f64`) in commit dcd035bfd.
//!
//! The builder's own guard (`divergence_axis_reduction_simd_fastpath_1689.rs`)
//! uses an `arange` input on near-power-of-two shapes. `arange`'s symmetry can
//! MASK an `[outer, axis, inner]` slot-mapping transposition (a wrong split can
//! preserve the sum-of-all while permuting individual outputs). This probe
//! instead uses a NON-UNIFORM, non-monotone fill `(i*7 % 13) - 6` over
//! NON-SQUARE, NON-power-of-two shapes ([3,5,7], [7,11], [2,3,4,5]) reducing
//! EVERY axis, plus the SIMD horizontal-sum tail lengths (1,2,3,7,8,9,15,16,17,
//! 1000,1001), degenerate extent-1 dims, single-element, negative-dim indexing,
//! and non-contiguous transposed/permuted views.
//!
//! Every expected value is from LIVE torch 2.11.0+cu130 (`torch.sum(dim=)` /
//! `torch.mean(dim=)`) baked into the fixtures
//! `tests/fixtures/reduction_simd_torch_ref_1689.json` and
//! `..._ref_t_1689.json` by the oracle (R-CHAR-3: torch-sourced, NOT copied
//! from ferrotorch). Each output element is compared individually so an index
//! transposition that preserves the total is still caught.

use ferrotorch_core::grad_fns::reduction::{mean_dim, sum_dim};
use ferrotorch_core::{Tensor, TensorStorage};
use serde::Deserialize;

const REF_CONTIG: &str = include_str!("fixtures/reduction_simd_torch_ref_1689.json");
const REF_T: &str = include_str!("fixtures/reduction_simd_torch_ref_t_1689.json");

/// The non-uniform fill used by the oracle: `(i*7) % 13 - 6`. Reproduced here
/// so the ferrotorch input matches the torch input element-for-element.
fn fill(numel: usize) -> Vec<f64> {
    (0..numel).map(|i| ((i * 7) % 13) as f64 - 6.0).collect()
}

#[derive(Deserialize)]
struct ContigCase {
    shape: Vec<usize>,
    dim: i64,
    keepdim: bool,
    dtype: String,
    sum_shape: Vec<usize>,
    sum: Vec<f64>,
    #[serde(default)]
    mean_shape: Vec<usize>,
    #[serde(default)]
    mean: Vec<f64>,
}

#[derive(Deserialize)]
struct TCase {
    base: Vec<usize>,
    perm: Vec<usize>,
    tdim: i64,
    dtype: String,
    sum: Vec<f64>,
    sum_shape: Vec<usize>,
    tshape: Vec<usize>,
}

fn tol(dtype: &str, e: f64) -> f64 {
    if dtype == "f32" {
        1e-4 * e.abs().max(1.0)
    } else {
        1e-9 * e.abs().max(1.0)
    }
}

fn build_f32(shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let v: Vec<f32> = fill(numel).into_iter().map(|x| x as f32).collect();
    Tensor::from_storage(TensorStorage::cpu(v), shape.to_vec(), false).unwrap()
}
fn build_f64(shape: &[usize]) -> Tensor<f64> {
    let numel: usize = shape.iter().product();
    Tensor::from_storage(TensorStorage::cpu(fill(numel)), shape.to_vec(), false).unwrap()
}

/// Contiguous probe: every shape/dim/keepdim/dtype, element-by-element vs torch.
#[test]
fn probe_contiguous_nonuniform_vs_torch() {
    let cases: Vec<ContigCase> = serde_json::from_str(REF_CONTIG).unwrap();
    let mut failures: Vec<String> = Vec::new();

    for c in &cases {
        // ---- sum ----
        let (got_sum_shape, got_sum): (Vec<usize>, Vec<f64>) = if c.dtype == "f32" {
            let out = sum_dim(&build_f32(&c.shape), c.dim, c.keepdim).unwrap();
            (
                out.shape().to_vec(),
                out.data().unwrap().iter().map(|&x| x as f64).collect(),
            )
        } else {
            let out = sum_dim(&build_f64(&c.shape), c.dim, c.keepdim).unwrap();
            (out.shape().to_vec(), out.data().unwrap().to_vec())
        };
        if got_sum_shape != c.sum_shape {
            failures.push(format!(
                "SUM SHAPE shape={:?} dim={} keepdim={} dtype={}: torch={:?} ferro={:?}",
                c.shape, c.dim, c.keepdim, c.dtype, c.sum_shape, got_sum_shape
            ));
        }
        for (idx, (&g, &e)) in got_sum.iter().zip(c.sum.iter()).enumerate() {
            if (g - e).abs() > tol(&c.dtype, e) {
                failures.push(format!(
                    "SUM VALUE shape={:?} dim={} keepdim={} dtype={} idx={}: torch={} ferro={}",
                    c.shape, c.dim, c.keepdim, c.dtype, idx, e, g
                ));
            }
        }

        // ---- mean ---- (skip the synthetic neg cases that carry no mean fields populated? they do)
        if c.mean.is_empty() {
            continue;
        }
        let (got_mean_shape, got_mean): (Vec<usize>, Vec<f64>) = if c.dtype == "f32" {
            let out = mean_dim(&build_f32(&c.shape), c.dim, c.keepdim).unwrap();
            (
                out.shape().to_vec(),
                out.data().unwrap().iter().map(|&x| x as f64).collect(),
            )
        } else {
            let out = mean_dim(&build_f64(&c.shape), c.dim, c.keepdim).unwrap();
            (out.shape().to_vec(), out.data().unwrap().to_vec())
        };
        if got_mean_shape != c.mean_shape {
            failures.push(format!(
                "MEAN SHAPE shape={:?} dim={} keepdim={} dtype={}: torch={:?} ferro={:?}",
                c.shape, c.dim, c.keepdim, c.dtype, c.mean_shape, got_mean_shape
            ));
        }
        for (idx, (&g, &e)) in got_mean.iter().zip(c.mean.iter()).enumerate() {
            if (g - e).abs() > tol(&c.dtype, e) {
                failures.push(format!(
                    "MEAN VALUE shape={:?} dim={} keepdim={} dtype={} idx={}: torch={} ferro={}",
                    c.shape, c.dim, c.keepdim, c.dtype, idx, e, g
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} divergences vs live torch:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Non-contiguous probe: build a contiguous base, transpose/permute it (zero-copy
/// strided view), reduce each axis. The fast path requires contiguity, so the
/// `contiguous()` materialisation must fire and still match torch.
#[test]
fn probe_noncontiguous_views_vs_torch() {
    let cases: Vec<TCase> = serde_json::from_str(REF_T).unwrap();
    let mut failures: Vec<String> = Vec::new();

    for c in &cases {
        let (got_shape, got): (Vec<usize>, Vec<f64>) = if c.dtype == "f32" {
            let base = build_f32(&c.base);
            let view = base.permute(&c.perm).unwrap();
            assert_eq!(
                view.shape(),
                c.tshape.as_slice(),
                "view shape mismatch base={:?} perm={:?}",
                c.base,
                c.perm
            );
            let out = sum_dim(&view, c.tdim, false).unwrap();
            (
                out.shape().to_vec(),
                out.data().unwrap().iter().map(|&x| x as f64).collect(),
            )
        } else {
            let base = build_f64(&c.base);
            let view = base.permute(&c.perm).unwrap();
            let out = sum_dim(&view, c.tdim, false).unwrap();
            (out.shape().to_vec(), out.data().unwrap().to_vec())
        };
        if got_shape != c.sum_shape {
            failures.push(format!(
                "T SUM SHAPE base={:?} perm={:?} tdim={} dtype={}: torch={:?} ferro={:?}",
                c.base, c.perm, c.tdim, c.dtype, c.sum_shape, got_shape
            ));
        }
        for (idx, (&g, &e)) in got.iter().zip(c.sum.iter()).enumerate() {
            if (g - e).abs() > tol(&c.dtype, e) {
                failures.push(format!(
                    "T SUM VALUE base={:?} perm={:?} tdim={} dtype={} idx={}: torch={} ferro={}",
                    c.base, c.perm, c.tdim, c.dtype, idx, e, g
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} non-contiguous divergences vs live torch:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Negative-dim indexing must map to the same positive-index result.
#[test]
fn probe_negative_dim_matches_positive() {
    let shape = [3usize, 5, 7];
    let x = build_f64(&shape);
    for (neg, pos) in [(-1i64, 2i64), (-2, 1), (-3, 0)] {
        let a = sum_dim(&x, neg, false).unwrap();
        let b = sum_dim(&x, pos, false).unwrap();
        assert_eq!(a.shape(), b.shape(), "neg {neg} vs pos {pos} shape");
        assert_eq!(
            a.data().unwrap().to_vec(),
            b.data().unwrap().to_vec(),
            "neg-dim {neg} must equal pos-dim {pos}"
        );
    }
}

/// SIMD horizontal-sum tail integrity (inner==1): a high-precision f64 Kahan
/// reference over the exact same fill must match ferrotorch within a few ULP.
/// A DROPPED-TAIL bug (axis not a multiple of the lane count) shows as a large
/// error here, independent of the torch fixture.
#[test]
fn probe_simd_tail_kahan_reference() {
    let mut failures = Vec::new();
    for &ax in &[1usize, 2, 3, 7, 8, 9, 15, 16, 17, 1000, 1001] {
        let shape = [3usize, ax];
        // f64 path
        let xf = build_f64(&shape);
        let out = sum_dim(&xf, 1, false).unwrap();
        let got = out.data().unwrap().to_vec();
        let data = fill(3 * ax);
        for o in 0..3 {
            // Kahan sum of row o.
            let mut s = 0.0f64;
            let mut comp = 0.0f64;
            for k in 0..ax {
                let y = data[o * ax + k] - comp;
                let t = s + y;
                comp = (t - s) - y;
                s = t;
            }
            if (got[o] - s).abs() > 1e-9 * s.abs().max(1.0) {
                failures.push(format!(
                    "TAIL f64 axis={ax} row={o}: kahan={s} ferro={} (dropped tail?)",
                    got[o]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
