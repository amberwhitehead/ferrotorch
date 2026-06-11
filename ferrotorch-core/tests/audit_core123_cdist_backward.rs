//! CORE-123 (#1817, CLASS-S High) regression battery: `cdist`
//! (`ferrotorch-core/src/ops/tensor_ops.rs`) must differentiate BOTH input
//! point sets — pre-fix every CPU and CUDA result was created with
//! `requires_grad = false` and no grad_fn, silently turning any training
//! loss through `cdist` into a detached constant.
//!
//! Upstream backward semantics (`tools/autograd/derivatives.yaml`:
//! `_cdist_forward -> _cdist_backward` for x1 and the negated/transposed
//! scatter for x2; per-norm weights per
//! `aten/src/ATen/native/cuda/DistanceKernel.cu` `dists::{zero,one,two,inf,p}::backward`):
//!   w(d, dist) = 0                                   (p = 0)
//!              | sign(d)                             (p = 1, sign(0) = 0)
//!              | d / dist          (0 at dist == 0)  (p = 2)
//!              | sign(d)*(|d| == dist)               (p = inf; EVERY tied max)
//!              | sign(d)*(|d|/dist)^(p-1)  (0 at d == 0 or dist == 0)
//!   grad_x1[i,k] = sum_j g[i,j] * w;  grad_x2[j,k] = -sum_i g[i,j] * w.
//!
//! Every expectation below is quoted from a LIVE torch 2.11.0+cu130 session
//! (R-ORACLE-1(b)); each test asserts gradient FLOW — values reaching the
//! original leaves — per R-ORACLE-3, never bare `requires_grad` flags.

use ferrotorch_core::ops::tensor_ops::cdist;
use ferrotorch_core::{Tensor, TensorStorage};

fn leaf(data: &[f32], shape: &[usize], rg: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), rg).expect("leaf")
}

/// Tolerance: oracle gradients are O(1); f32 eps ~1.2e-7 with one powf/sqrt
/// round-trip and a 2-term reduction — 1e-5 absolute bounds that with
/// margin (pre-fix divergence is "no gradient at all").
const TOL: f32 = 1e-5;

#[track_caller]
fn assert_close(name: &str, got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len(), "{name} length");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= TOL,
            "{name}[{i}]: got {g}, expected {e} (tol {TOL})"
        );
    }
}

/// Run one CPU case: forward, cotangent backward, both leaf grads.
#[track_caller]
fn check_grads(p: f64, g: &[f32], exp_g1: &[f32], exp_g2: &[f32]) {
    let x1 = leaf(&[0.0, 0.5, 2.0, 1.0, 1.0, 1.0], &[2, 3], true);
    let x2 = leaf(&[0.0, 1.5, 0.5, 1.0, 1.0, 1.0], &[2, 3], true);
    let d = cdist(&x1, &x2, p).unwrap_or_else(|e| panic!("cdist p={p}: {e:?}"));
    assert!(
        d.requires_grad(),
        "cdist p={p}: torch output carries grad_fn; detached output cannot train"
    );
    let cot = leaf(g, &[2, 2], false);
    d.backward_with_gradient(&cot)
        .unwrap_or_else(|e| panic!("backward p={p}: {e:?}"));
    let g1 = x1.grad().unwrap().expect("x1.grad present");
    let g2 = x2.grad().unwrap().expect("x2.grad present");
    assert_close(&format!("p={p} grad_x1"), g1.data().expect("g1"), exp_g1);
    assert_close(&format!("p={p} grad_x2"), g2.data().expect("g2"), exp_g2);
}

// torch oracle (one session, x1/x2 as in `check_grads`, cot = [[1,2],[3,4]]):
//   >>> d = torch.cdist(a, b, p=P); d.backward(torch.tensor([[1.,2.],[3.,4.]]))
//   p=1:   a.grad [[-2,-3,3],[3,-3,3]]          b.grad [[-3,4,-4],[2,2,-2]]
//   p=2:   a.grad [[-1.3333334,-1.2213669,2.1653838],[2.4494896,-1.2247448,1.2247448]]
//          b.grad [[-2.4494896,1.7794449,-2.0567951],[1.3333334,0.6666667,-1.3333334]]
//   p=inf: a.grad [[-2,0,3],[3,0,0]]            b.grad [[-3,0,-1],[2,0,-2]]
//   p=3:   a.grad [[-1.2100148,-0.6763398,2.051146],[2.5853217,-0.6463304,0.6463304]]
//          b.grad [[-2.5853217,1.0201665,-1.4874617],[1.2100148,0.3025037,-1.2100148]]

#[test]
fn cdist_backward_p1_matches_torch() {
    check_grads(
        1.0,
        &[1.0, 2.0, 3.0, 4.0],
        &[-2.0, -3.0, 3.0, 3.0, -3.0, 3.0],
        &[-3.0, 4.0, -4.0, 2.0, 2.0, -2.0],
    );
}

#[test]
fn cdist_backward_p2_matches_torch() {
    check_grads(
        2.0,
        &[1.0, 2.0, 3.0, 4.0],
        &[
            -1.333_333_4,
            -1.221_366_9,
            2.165_383_8,
            2.449_489_6,
            -1.224_744_8,
            1.224_744_8,
        ],
        &[
            -2.449_489_6,
            1.779_444_9,
            -2.056_795_1,
            1.333_333_4,
            0.666_666_7,
            -1.333_333_4,
        ],
    );
}

/// p = inf assigns sign-grad through the max coordinate(s).
#[test]
fn cdist_backward_p_inf_matches_torch() {
    check_grads(
        f64::INFINITY,
        &[1.0, 2.0, 3.0, 4.0],
        &[-2.0, 0.0, 3.0, 3.0, 0.0, 0.0],
        &[-3.0, 0.0, -1.0, 2.0, 0.0, -2.0],
    );
}

/// General p (= 3).
#[test]
fn cdist_backward_p3_matches_torch() {
    check_grads(
        3.0,
        &[1.0, 2.0, 3.0, 4.0],
        &[
            -1.210_014_8,
            -0.676_339_8,
            2.051_146,
            2.585_321_7,
            -0.646_330_4,
            0.646_330_4,
        ],
        &[
            -2.585_321_7,
            1.020_166_5,
            -1.487_461_7,
            1.210_014_8,
            0.302_503_7,
            -1.210_014_8,
        ],
    );
}

/// p = 0: the count-"norm" is piecewise constant — torch backward yields
/// EXACT zeros for both inputs (live oracle: `a.grad == b.grad == zeros`),
/// not an error and not a detached output.
#[test]
fn cdist_backward_p0_is_zeros() {
    check_grads(0.0, &[1.0, 2.0, 3.0, 4.0], &[0.0; 6], &[0.0; 6]);
}

/// Zero-distance guard at p = 2 (torch `_euclidean_dist` backward zeroes the
/// 0/0 direction): identical points produce zero grads, not NaN.
/// Oracle: `torch.cdist(ones(1,3), ones(1,3), 2).backward(ones)` →
/// `a.grad == b.grad == [[0,0,0]]`.
#[test]
fn cdist_backward_p2_zero_distance_yields_zeros_not_nan() {
    let a = leaf(&[1.0, 1.0, 1.0], &[1, 3], true);
    let b = leaf(&[1.0, 1.0, 1.0], &[1, 3], true);
    let d = cdist(&a, &b, 2.0).expect("cdist");
    d.backward_with_gradient(&leaf(&[1.0], &[1, 1], false))
        .expect("backward");
    let ga = a.grad().unwrap().expect("a.grad");
    let gb = b.grad().unwrap().expect("b.grad");
    assert_close("zero-dist a.grad", ga.data().expect("ga"), &[0.0, 0.0, 0.0]);
    assert_close("zero-dist b.grad", gb.data().expect("gb"), &[0.0, 0.0, 0.0]);
}

/// p = inf with TWO coordinates tied at the max |diff|: torch gives BOTH the
/// sign-grad (live oracle: a=[[2,0,1]], b=[[0,2,1]] → d=[[2]],
/// a.grad=[[1,-1,0]], b.grad=[[-1,1,0]]).
#[test]
fn cdist_backward_p_inf_ties_grad_every_max_coordinate() {
    let a = leaf(&[2.0, 0.0, 1.0], &[1, 3], true);
    let b = leaf(&[0.0, 2.0, 1.0], &[1, 3], true);
    let d = cdist(&a, &b, f64::INFINITY).expect("cdist");
    d.backward_with_gradient(&leaf(&[1.0], &[1, 1], false))
        .expect("backward");
    let ga = a.grad().unwrap().expect("a.grad");
    let gb = b.grad().unwrap().expect("b.grad");
    assert_close("inf-tie a.grad", ga.data().expect("ga"), &[1.0, -1.0, 0.0]);
    assert_close("inf-tie b.grad", gb.data().expect("gb"), &[-1.0, 1.0, 0.0]);
}

/// Sub-quadratic norms guard the zero-diff coordinate (|d|^(p-1) would blow
/// up for p < 1): torch yields 0 there. Live oracle (a=[[0,.5,2]],
/// b=[[0,1.5,.5]]): p=0.5 → a.grad [[0,-2.2247448,1.8164966]];
/// p=1.5 → a.grad [[0,-0.7063841,0.8651404]]; b.grad = -a.grad.
#[test]
fn cdist_backward_fractional_p_guards_zero_diff() {
    for (p, exp) in [
        (0.5, [0.0, -2.224_744_8, 1.816_496_6]),
        (1.5, [0.0, -0.706_384_1, 0.865_140_4]),
    ] {
        let a = leaf(&[0.0, 0.5, 2.0], &[1, 3], true);
        let b = leaf(&[0.0, 1.5, 0.5], &[1, 3], true);
        let d = cdist(&a, &b, p).unwrap_or_else(|e| panic!("cdist p={p}: {e:?}"));
        d.backward_with_gradient(&leaf(&[1.0], &[1, 1], false))
            .unwrap_or_else(|e| panic!("backward p={p}: {e:?}"));
        let ga = a.grad().unwrap().expect("a.grad");
        let gb = b.grad().unwrap().expect("b.grad");
        assert_close(&format!("p={p} a.grad"), ga.data().expect("ga"), &exp);
        let neg: Vec<f32> = exp.iter().map(|v| -v).collect();
        assert_close(&format!("p={p} b.grad"), gb.data().expect("gb"), &neg);
    }
}

/// Only one input tracking: the other side gets NO gradient (None), the
/// tracking side still flows.
#[test]
fn cdist_backward_single_tracking_input() {
    let a = leaf(&[0.0, 0.5, 2.0], &[1, 3], true);
    let b = leaf(&[0.0, 1.5, 0.5], &[1, 3], false);
    let d = cdist(&a, &b, 2.0).expect("cdist");
    assert!(d.requires_grad());
    d.backward_with_gradient(&leaf(&[1.0], &[1, 1], false))
        .expect("backward");
    assert!(a.grad().unwrap().is_some(), "tracking input gets grad");
    assert!(b.grad().unwrap().is_none(), "non-tracking input stays None");
}

/// Batched 3-D backward at p = 2. Live oracle:
/// ```text
/// >>> a = torch.tensor([[[0.,1.],[2.,3.]],[[1.,0.],[0.,1.]]], requires_grad=True)
/// >>> b = torch.tensor([[[1.,1.]],[[2.,2.]]], requires_grad=True)
/// >>> d = torch.cdist(a, b, p=2.); d.backward(torch.ones_like(d))
/// a.grad [[[-1,0],[0.4472136,0.8944272]],[[-0.4472136,-0.8944272],[-0.8944272,-0.4472136]]]
/// b.grad [[[0.5527864,-0.8944272]],[[1.3416407,1.3416407]]]
/// ```
#[test]
fn cdist_backward_batched_3d_p2_matches_torch() {
    let a = leaf(&[0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2], true);
    let b = leaf(&[1.0, 1.0, 2.0, 2.0], &[2, 1, 2], true);
    let d = cdist(&a, &b, 2.0).expect("cdist");
    d.backward_with_gradient(&leaf(&[1.0, 1.0, 1.0, 1.0], &[2, 2, 1], false))
        .expect("backward");
    let ga = a.grad().unwrap().expect("a.grad");
    let gb = b.grad().unwrap().expect("b.grad");
    assert_close(
        "batched a.grad",
        ga.data().expect("ga"),
        &[
            -1.0,
            0.0,
            0.447_213_6,
            0.894_427_2, //
            -0.447_213_6,
            -0.894_427_2,
            -0.894_427_2,
            -0.447_213_6,
        ],
    );
    assert_close(
        "batched b.grad",
        gb.data().expect("gb"),
        &[0.552_786_4, -0.894_427_2, 1.341_640_7, 1.341_640_7],
    );
}

// ─────────────────────────────────────────────────────────────────────────
// CUDA lane: the GPU forward must also carry the backward edge; gradients
// must reach the CUDA leaves and live on Device::Cuda(0) (R-ORACLE-3).
// ─────────────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the CORE-123 GPU pins");
        });
    }

    /// Same p=2 case as the CPU lane, leaves on CUDA. Oracle identical
    /// (torch cuda backward matches cpu for these values).
    #[test]
    fn cdist_backward_p2_cuda_grads_flow_and_stay_resident() {
        ensure_cuda_backend();
        let x1 = leaf(&[0.0, 0.5, 2.0, 1.0, 1.0, 1.0], &[2, 3], true)
            .to(Device::Cuda(0))
            .expect("upload x1");
        let x2 = leaf(&[0.0, 1.5, 0.5, 1.0, 1.0, 1.0], &[2, 3], true)
            .to(Device::Cuda(0))
            .expect("upload x2");
        assert!(x1.requires_grad(), "to(cuda) keeps requires_grad");
        let d = cdist(&x1, &x2, 2.0).expect("cdist on cuda");
        assert_eq!(d.device(), Device::Cuda(0), "forward output device");
        assert!(d.requires_grad(), "cuda cdist output carries grad_fn");
        let cot = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false)
            .to(Device::Cuda(0))
            .expect("upload cot");
        d.backward_with_gradient(&cot).expect("cuda backward");
        let g1 = x1.grad().unwrap().expect("x1.grad present");
        let g2 = x2.grad().unwrap().expect("x2.grad present");
        assert_eq!(g1.device(), Device::Cuda(0), "x1.grad device");
        assert_eq!(g2.device(), Device::Cuda(0), "x2.grad device");
        let g1h = g1.cpu().expect("g1 D2H");
        let g2h = g2.cpu().expect("g2 D2H");
        assert_close(
            "cuda p=2 grad_x1",
            g1h.data().expect("g1"),
            &[
                -1.333_333_4,
                -1.221_366_9,
                2.165_383_8,
                2.449_489_6,
                -1.224_744_8,
                1.224_744_8,
            ],
        );
        assert_close(
            "cuda p=2 grad_x2",
            g2h.data().expect("g2"),
            &[
                -2.449_489_6,
                1.779_444_9,
                -2.056_795_1,
                1.333_333_4,
                0.666_666_7,
                -1.333_333_4,
            ],
        );
    }
}
