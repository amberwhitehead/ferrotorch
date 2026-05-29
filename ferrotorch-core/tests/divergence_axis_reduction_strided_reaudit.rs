//! RE-AUDIT regression guard for commit 22c844cb3 (#1678):
//! the CPU `sum_dim_inner` / `mean_dim_inner` accumulation loops in
//! `ferrotorch-core/src/grad_fns/reduction.rs` were rewritten from a
//! per-element `coords` Vec + div/mod flat-index decomposition into a hoisted
//! reusable `coords` buffer + precomputed `out_strides` + an odometer
//! coordinate increment (last dim fastest). This is a PERF rewrite of a
//! CORRECTNESS-sensitive index-mapping loop.
//!
//! The risk: the odometer increment, the `oi` (output index) computation, or
//! the `norm_dim` skip could map an input element to the WRONG accumulator
//! slot on higher-rank / non-square / specific-dim cases that the
//! `[1000,1000]` benchmark never exercises. A wrong slot mapping corrupts the
//! reduction silently.
//!
//! Every expected value below was produced by LIVE PyTorch 2.11.0 via
//! `torch.sum(x, dim, keepdim)` / `torch.mean(x, dim, keepdim)` on
//! `x = torch.arange(numel, dtype=torch.float64).reshape(shape)`. Inputs use
//! integer-valued `arange` so a wrong slot mapping shows up bit-exactly (no
//! float-eps masking). The expected values are therefore traceable to the
//! upstream torch reduction kernel
//! (`aten/src/ATen/native/ReduceOps.cpp::sum`/`mean`), NOT copied from the
//! ferrotorch side (R-CHAR-3).
//!
//! Verdict at audit time: all cases PASS — this file is a PASSING regression
//! guard pinning the rank x dim x keepdim matrix against torch.

use ferrotorch_core::grad_fns::reduction::{mean_dim, sum_dim};
use ferrotorch_core::{Tensor, TensorStorage};

/// `x = torch.arange(numel, dtype=f64).reshape(shape)` as a CPU f64 tensor.
fn arange(shape: &[usize]) -> Tensor<f64> {
    let numel: usize = shape.iter().product();
    let v: Vec<f64> = (0..numel).map(|i| i as f64).collect();
    Tensor::from_storage(TensorStorage::cpu(v), shape.to_vec(), false).expect("arange tensor")
}

/// One row of the torch reference matrix.
struct Case {
    shape: &'static [usize],
    dim: i64,
    keepdim: bool,
    /// `list(torch.sum(x, dim, keepdim).shape)`
    sum_shape: &'static [usize],
    /// `torch.sum(x, dim, keepdim).flatten().tolist()`
    sum: &'static [f64],
    /// `list(torch.mean(x, dim, keepdim).shape)`
    mean_shape: &'static [usize],
    /// `torch.mean(x, dim, keepdim).flatten().tolist()`
    mean: &'static [f64],
}

// ---------------------------------------------------------------------------
// The torch reference matrix (live torch 2.11.0).
// ---------------------------------------------------------------------------
const MATRIX: &[Case] = &[
    // --- 2-D non-square [3,5] ---
    Case {
        shape: &[3, 5],
        dim: 0,
        keepdim: true,
        sum_shape: &[1, 5],
        sum: &[15.0, 18.0, 21.0, 24.0, 27.0],
        mean_shape: &[1, 5],
        mean: &[5.0, 6.0, 7.0, 8.0, 9.0],
    },
    Case {
        shape: &[3, 5],
        dim: 0,
        keepdim: false,
        sum_shape: &[5],
        sum: &[15.0, 18.0, 21.0, 24.0, 27.0],
        mean_shape: &[5],
        mean: &[5.0, 6.0, 7.0, 8.0, 9.0],
    },
    Case {
        shape: &[3, 5],
        dim: 1,
        keepdim: true,
        sum_shape: &[3, 1],
        sum: &[10.0, 35.0, 60.0],
        mean_shape: &[3, 1],
        mean: &[2.0, 7.0, 12.0],
    },
    Case {
        shape: &[3, 5],
        dim: 1,
        keepdim: false,
        sum_shape: &[3],
        sum: &[10.0, 35.0, 60.0],
        mean_shape: &[3],
        mean: &[2.0, 7.0, 12.0],
    },
    // --- 3-D [2,3,4] — the critical multi-dim-output / odometer case ---
    Case {
        shape: &[2, 3, 4],
        dim: 0,
        keepdim: true,
        sum_shape: &[1, 3, 4],
        sum: &[
            12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0,
        ],
        mean_shape: &[1, 3, 4],
        mean: &[
            6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ],
    },
    Case {
        shape: &[2, 3, 4],
        dim: 0,
        keepdim: false,
        sum_shape: &[3, 4],
        sum: &[
            12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0,
        ],
        mean_shape: &[3, 4],
        mean: &[
            6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ],
    },
    Case {
        shape: &[2, 3, 4],
        dim: 1,
        keepdim: true,
        sum_shape: &[2, 1, 4],
        sum: &[12.0, 15.0, 18.0, 21.0, 48.0, 51.0, 54.0, 57.0],
        mean_shape: &[2, 1, 4],
        mean: &[4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0],
    },
    Case {
        shape: &[2, 3, 4],
        dim: 1,
        keepdim: false,
        sum_shape: &[2, 4],
        sum: &[12.0, 15.0, 18.0, 21.0, 48.0, 51.0, 54.0, 57.0],
        mean_shape: &[2, 4],
        mean: &[4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0],
    },
    Case {
        shape: &[2, 3, 4],
        dim: 2,
        keepdim: true,
        sum_shape: &[2, 3, 1],
        sum: &[6.0, 22.0, 38.0, 54.0, 70.0, 86.0],
        mean_shape: &[2, 3, 1],
        mean: &[1.5, 5.5, 9.5, 13.5, 17.5, 21.5],
    },
    Case {
        shape: &[2, 3, 4],
        dim: 2,
        keepdim: false,
        sum_shape: &[2, 3],
        sum: &[6.0, 22.0, 38.0, 54.0, 70.0, 86.0],
        mean_shape: &[2, 3],
        mean: &[1.5, 5.5, 9.5, 13.5, 17.5, 21.5],
    },
    // --- 4-D [2,3,2,3] — reduce a MIDDLE dim ---
    Case {
        shape: &[2, 3, 2, 3],
        dim: 1,
        keepdim: true,
        sum_shape: &[2, 1, 2, 3],
        sum: &[
            18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 72.0, 75.0, 78.0, 81.0, 84.0, 87.0,
        ],
        mean_shape: &[2, 1, 2, 3],
        mean: &[
            6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
        ],
    },
    Case {
        shape: &[2, 3, 2, 3],
        dim: 1,
        keepdim: false,
        sum_shape: &[2, 2, 3],
        sum: &[
            18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 72.0, 75.0, 78.0, 81.0, 84.0, 87.0,
        ],
        mean_shape: &[2, 2, 3],
        mean: &[
            6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
        ],
    },
    Case {
        shape: &[2, 3, 2, 3],
        dim: 2,
        keepdim: true,
        sum_shape: &[2, 3, 1, 3],
        sum: &[
            3.0, 5.0, 7.0, 15.0, 17.0, 19.0, 27.0, 29.0, 31.0, 39.0, 41.0, 43.0, 51.0, 53.0, 55.0,
            63.0, 65.0, 67.0,
        ],
        mean_shape: &[2, 3, 1, 3],
        mean: &[
            1.5, 2.5, 3.5, 7.5, 8.5, 9.5, 13.5, 14.5, 15.5, 19.5, 20.5, 21.5, 25.5, 26.5, 27.5,
            31.5, 32.5, 33.5,
        ],
    },
    Case {
        shape: &[2, 3, 2, 3],
        dim: 2,
        keepdim: false,
        sum_shape: &[2, 3, 3],
        sum: &[
            3.0, 5.0, 7.0, 15.0, 17.0, 19.0, 27.0, 29.0, 31.0, 39.0, 41.0, 43.0, 51.0, 53.0, 55.0,
            63.0, 65.0, 67.0,
        ],
        mean_shape: &[2, 3, 3],
        mean: &[
            1.5, 2.5, 3.5, 7.5, 8.5, 9.5, 13.5, 14.5, 15.5, 19.5, 20.5, 21.5, 25.5, 26.5, 27.5,
            31.5, 32.5, 33.5,
        ],
    },
    // --- negative dim ---
    Case {
        shape: &[2, 3, 4],
        dim: -1,
        keepdim: false,
        sum_shape: &[2, 3],
        sum: &[6.0, 22.0, 38.0, 54.0, 70.0, 86.0],
        mean_shape: &[2, 3],
        mean: &[1.5, 5.5, 9.5, 13.5, 17.5, 21.5],
    },
    Case {
        shape: &[2, 3, 4],
        dim: -2,
        keepdim: true,
        sum_shape: &[2, 1, 4],
        sum: &[12.0, 15.0, 18.0, 21.0, 48.0, 51.0, 54.0, 57.0],
        mean_shape: &[2, 1, 4],
        mean: &[4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0],
    },
    // --- extent-1 reduced dim ---
    Case {
        shape: &[2, 1, 4],
        dim: 1,
        keepdim: false,
        sum_shape: &[2, 4],
        sum: &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        mean_shape: &[2, 4],
        mean: &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    },
    Case {
        shape: &[2, 1, 4],
        dim: 1,
        keepdim: true,
        sum_shape: &[2, 1, 4],
        sum: &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        mean_shape: &[2, 1, 4],
        mean: &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    },
    // --- 1-D reduced to scalar ---
    Case {
        shape: &[5],
        dim: 0,
        keepdim: false,
        sum_shape: &[],
        sum: &[10.0],
        mean_shape: &[],
        mean: &[2.0],
    },
    Case {
        shape: &[5],
        dim: 0,
        keepdim: true,
        sum_shape: &[1],
        sum: &[10.0],
        mean_shape: &[1],
        mean: &[2.0],
    },
];

/// RE-AUDIT (#1678): full rank x dim x keepdim correctness matrix for
/// `sum_dim` against live torch 2.11 `torch.sum(x, dim, keepdim)`.
///
/// Mirrors the contiguous reduction in
/// `aten/src/ATen/native/ReduceOps.cpp::sum`. Bit-exact (integer-valued f64
/// arange inputs).
#[test]
fn divergence_sum_dim_rank_dim_keepdim_matrix_vs_torch() {
    for c in MATRIX {
        let x = arange(c.shape);
        let out = sum_dim(&x, c.dim, c.keepdim).unwrap_or_else(|e| {
            panic!(
                "sum_dim {:?} dim={} kd={} err: {e:?}",
                c.shape, c.dim, c.keepdim
            )
        });
        assert_eq!(
            out.shape(),
            c.sum_shape,
            "sum_dim SHAPE mismatch: shape={:?} dim={} keepdim={}",
            c.shape,
            c.dim,
            c.keepdim
        );
        let got = out.data().expect("read sum_dim output");
        assert_eq!(
            got, c.sum,
            "sum_dim VALUE mismatch (wrong slot mapping?): shape={:?} dim={} keepdim={}\n  torch={:?}\n  ferro={:?}",
            c.shape, c.dim, c.keepdim, c.sum, got
        );
    }
}

/// RE-AUDIT (#1678): full rank x dim x keepdim correctness matrix for
/// `mean_dim` against live torch 2.11 `torch.mean(x, dim, keepdim)`.
///
/// Mirrors `aten/src/ATen/native/ReduceOps.cpp::mean`. Verifies the divisor is
/// `in_shape[dim]` (the reduced-axis size), not `numel` — the `[2,1,4]`
/// extent-1 and `[2,3,4]` dim=2 (divisor 4) cases pin this. Bit-exact where
/// the torch result is integer-valued, half-integer otherwise (still exact in
/// f64).
#[test]
fn divergence_mean_dim_rank_dim_keepdim_matrix_vs_torch() {
    for c in MATRIX {
        let x = arange(c.shape);
        let out = mean_dim(&x, c.dim, c.keepdim).unwrap_or_else(|e| {
            panic!(
                "mean_dim {:?} dim={} kd={} err: {e:?}",
                c.shape, c.dim, c.keepdim
            )
        });
        assert_eq!(
            out.shape(),
            c.mean_shape,
            "mean_dim SHAPE mismatch: shape={:?} dim={} keepdim={}",
            c.shape,
            c.dim,
            c.keepdim
        );
        let got = out.data().expect("read mean_dim output");
        assert_eq!(
            got.len(),
            c.mean.len(),
            "mean_dim numel mismatch: shape={:?} dim={}",
            c.shape,
            c.dim
        );
        for (i, (&g, &e)) in got.iter().zip(c.mean.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-12,
                "mean_dim VALUE mismatch (wrong slot or wrong divisor?): shape={:?} dim={} keepdim={} idx={}\n  torch={e}\n  ferro={g}",
                c.shape,
                c.dim,
                c.keepdim,
                i
            );
        }
    }
}

/// RE-AUDIT (#1678): odometer-order verdict. `sum_dim` materializes via
/// `input.contiguous()` before reading `in_data`, so a transposed
/// (non-contiguous) input must still reduce correctly. A wrong odometer order
/// (e.g. first-dim-fastest instead of last-dim-fastest) would scramble this.
///
/// `x = torch.arange(15, f64).reshape(3,5).t()` -> shape [5,3], non-contiguous.
/// `torch.sum(xt, dim=1)` -> `[15, 18, 21, 24, 27]` (shape [5]);
/// `torch.sum(xt, dim=0)` -> `[10, 35, 60]` (shape [3]).
#[test]
fn divergence_sum_dim_transposed_noncontiguous_vs_torch() {
    let base = arange(&[3, 5]);
    let xt = base.transpose(0, 1).expect("transpose"); // [5,3], non-contig
    assert_eq!(xt.shape(), &[5, 3]);

    let s1 = sum_dim(&xt, 1, false).expect("sum_dim transposed dim=1");
    assert_eq!(s1.shape(), &[5]);
    // torch.sum(xt, dim=1, keepdim=False).tolist()
    assert_eq!(s1.data().expect("read"), &[15.0, 18.0, 21.0, 24.0, 27.0]);

    let s0 = sum_dim(&xt, 0, false).expect("sum_dim transposed dim=0");
    assert_eq!(s0.shape(), &[3]);
    // torch.sum(xt, dim=0, keepdim=False).tolist()
    assert_eq!(s0.data().expect("read"), &[10.0, 35.0, 60.0]);
}
