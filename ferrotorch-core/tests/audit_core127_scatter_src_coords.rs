//! CORE-127 (#1821, CLASS-V High) regression battery: scatter / scatter_add
//! must consume `src` by COORDINATE, not as a flat prefix, when `src` is
//! larger than `index`; per-axis `src` validation must mirror the upstream
//! rule; and the backward contract for a larger `src` must match live torch.
//!
//! Upstream contract (verified LIVE against `torch==2.11.0+cu130`, all
//! expectations below are pasted from that session — R-ORACLE-1(b)):
//!
//!   - Validation (`scatter_shape_check`,
//!     `aten/src/ATen/native/ScatterGatherChecks.h:67-124`):
//!       * "Index tensor must have the same number of dimensions as src
//!         tensor" (rank equality index vs src);
//!       * "Expected index [..] to be no larger than self [..] apart from
//!         dimension <dim> and to be no larger size than src [..]" — i.e.
//!         `index.size(d) <= src.size(d)` for ALL d (the
//!         `index.size(d) <= self.size(d), d != dim` half is CORE-126/#1820,
//!         not pinned here).
//!   - Forward consumption is coordinate-mapped: with `index` shape [2,1]
//!     and `src` shape [2,3], the consumed values are `src[0,0]` and
//!     `src[1,0]` — NOT flat offsets 0 and 1.
//!   - Backward: when `src.requires_grad` and `src.shape != index.shape`,
//!     live torch RAISES at backward time: "RuntimeError: Function
//!     ScatterBackward0 returned an invalid gradient at index 1 - got
//!     [2, 4] but expected shape compatible with [2, 5]"
//!     (same for ScatterAddBackward0). ferrotorch matches that oracle with a
//!     structured error (R-ORACLE-4: one contract, no dual-accept), while
//!     preserving PyTorch's scalar-src reduction and empty-index broadcast-zero
//!     cases. The audit recommendation of a full-src-shaped zero-padded gradient was
//!     checked against live torch and REJECTED: torch never produces such a
//!     gradient.
//!   - Backward grad for `input` alone is well-defined for a larger `src`
//!     (live torch computes it) and must keep working.
//!
//! Pre-fix observed behavior (R-AHON-1 probe, pasted in #1821): CPU forward
//! consumed `src` as a flat prefix (`out[1,0]=1.0` for the audit case, torch
//! says `30.0` with the src used here); per-axis-invalid `src` ([4,1] for
//! index [2,2]) silently accepted; backward silently returned an
//! index-shaped grad_src for a larger src.

use ferrotorch_core::autograd::graph::backward;
use ferrotorch_core::{FerrotorchError, Tensor, TensorStorage, scatter, scatter_add};

fn cpu_f32(data: &[f32], shape: &[usize], rg: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), rg).unwrap()
}

/// `arange(12).reshape(3,4)` — matches the live-torch session's `inp`.
fn input_3x4(rg: bool) -> Tensor<f32> {
    let d: Vec<f32> = (0..12).map(|v| v as f32).collect();
    cpu_f32(&d, &[3, 4], rg)
}

/// `(arange(6)*10).reshape(2,3)` — the audit case's larger src (trailing
/// axis 3 > index trailing axis 1).
fn src_2x3(rg: bool) -> Tensor<f32> {
    let d: Vec<f32> = (0..6).map(|v| (v * 10) as f32).collect();
    cpu_f32(&d, &[2, 3], rg)
}

/// `(arange(10)*10).reshape(2,5)` — trailing axis 5 > index trailing axis 4.
fn src_2x5(rg: bool) -> Tensor<f32> {
    let d: Vec<f32> = (0..10).map(|v| (v * 10) as f32).collect();
    cpu_f32(&d, &[2, 5], rg)
}

const IDX_A: [usize; 2] = [2, 1]; // shape [2,1]
const IDX_B: [usize; 8] = [1, 0, 2, 1, 2, 1, 0, 0]; // shape [2,4]

/// Live torch: `inp.scatter(0, [[2],[1]], (arange(6)*10).reshape(2,3))`
/// -> [0,1,2,3, 30,5,6,7, 0,9,10,11] (src[0,0]=0 to row 2, src[1,0]=30 to
/// row 1 — coordinate-mapped).
#[test]
fn core127_cpu_scatter_src_trailing_axis_larger_coordinate_mapped() {
    let out = scatter(&input_3x4(false), 0, &IDX_A, &[2, 1], &src_2x3(false)).unwrap();
    assert_eq!(
        out.data_vec().unwrap(),
        vec![
            0.0, 1.0, 2.0, 3.0, 30.0, 5.0, 6.0, 7.0, 0.0, 9.0, 10.0, 11.0
        ],
        "scatter must consume src by coordinate (src[1,0]=30), not flat prefix (src[0,1]=10)"
    );
}

/// Live torch: `inp.scatter_add(0, [[2],[1]], src_2x3)` ->
/// [0,1,2,3, 34,5,6,7, 8,9,10,11].
#[test]
fn core127_cpu_scatter_add_src_trailing_axis_larger_coordinate_mapped() {
    let out = scatter_add(&input_3x4(false), 0, &IDX_A, &[2, 1], &src_2x3(false)).unwrap();
    assert_eq!(
        out.data_vec().unwrap(),
        vec![
            0.0, 1.0, 2.0, 3.0, 34.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0
        ],
    );
}

/// Live torch: `inp.scatter(0, [[1,0,2,1],[2,1,0,0]], (arange(10)*10).reshape(2,5))`
/// -> [0,10,70,80, 0,60,6,30, 50,9,20,11]. The flat-prefix bug consumes
/// src flat 0..8 (0..70) instead of the coordinate slab {0,10,20,30,50,60,70,80}.
#[test]
fn core127_cpu_scatter_src_2x5_index_2x4_coordinate_mapped() {
    let out = scatter(&input_3x4(false), 0, &IDX_B, &[2, 4], &src_2x5(false)).unwrap();
    assert_eq!(
        out.data_vec().unwrap(),
        vec![
            0.0, 10.0, 70.0, 80.0, 0.0, 60.0, 6.0, 30.0, 50.0, 9.0, 20.0, 11.0
        ],
    );
}

/// Live torch: `inp.scatter_add(0, idxB, src_2x5)` ->
/// [0,11,72,83, 4,65,6,37, 58,9,30,11].
#[test]
fn core127_cpu_scatter_add_src_2x5_index_2x4_coordinate_mapped() {
    let out = scatter_add(&input_3x4(false), 0, &IDX_B, &[2, 4], &src_2x5(false)).unwrap();
    assert_eq!(
        out.data_vec().unwrap(),
        vec![
            0.0, 11.0, 72.0, 83.0, 4.0, 65.0, 6.0, 37.0, 58.0, 9.0, 30.0, 11.0
        ],
    );
}

// ── Per-axis src validation (torch rejects; pre-fix ferrotorch accepted) ──

#[track_caller]
fn assert_metadata_err<T: std::fmt::Debug>(r: Result<T, FerrotorchError>, what: &str) {
    match r {
        Err(FerrotorchError::ShapeMismatch { .. } | FerrotorchError::InvalidArgument { .. }) => {}
        Err(other) => panic!("{what}: expected ShapeMismatch/InvalidArgument, got Err({other:?})"),
        Ok(v) => panic!("{what}: expected structured Err, got Ok({v:?})"),
    }
}

/// Live torch: index [2,2], src [4,1] -> "Expected index [2, 2] to be no
/// larger than self [3, 4] apart from dimension 0 and to be no larger size
/// than src [4, 1]" (index.size(1)=2 > src.size(1)=1; equal numel ties the
/// pre-fix numel-only gate).
#[test]
fn core127_cpu_scatter_add_src_axis_smaller_than_index_rejected() {
    let src = cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[4, 1], false);
    let r = scatter_add(&input_3x4(false), 0, &[2, 1, 0, 1], &[2, 2], &src);
    assert_metadata_err(
        r.map(|t| t.data_vec()),
        "scatter_add index [2,2] src [4,1] (index axis 1 exceeds src)",
    );
}

/// Live torch: index [3,1] with src [2,3] -> "...no larger size than
/// src [2, 3]" (index.size(0)=3 > src.size(0)=2; pre-fix the numel gate
/// passed because 3 <= 6).
#[test]
fn core127_cpu_scatter_index_dim_axis_larger_than_src_rejected() {
    let r = scatter(&input_3x4(false), 0, &[2, 1, 0], &[3, 1], &src_2x3(false));
    assert_metadata_err(
        r.map(|t| t.data_vec()),
        "scatter index [3,1] src [2,3] (index dim axis exceeds src)",
    );
}

/// Live torch: rank-1 src for rank-2 index -> "Index tensor must have the
/// same number of dimensions as src tensor".
#[test]
fn core127_cpu_scatter_src_rank_mismatch_rejected() {
    let src = cpu_f32(&[0.0, 10.0, 20.0, 30.0, 40.0, 50.0], &[6], false);
    let r = scatter(&input_3x4(false), 0, &IDX_A, &[2, 1], &src);
    assert_metadata_err(
        r.map(|t| t.data_vec()),
        "scatter rank-1 src for rank-2 index",
    );
}

// ── Backward contract ──

/// Live torch (input grad only, src larger): grads flow;
/// `i.scatter(0, [[2],[1]], src_2x3).sum().backward()` ->
/// grad_input = [1,1,1,1, 0,1,1,1, 0,1,1,1].
#[test]
fn core127_cpu_scatter_larger_src_input_only_grad_matches_torch() {
    let input = input_3x4(true);
    let out = scatter(&input, 0, &IDX_A, &[2, 1], &src_2x3(false)).unwrap();
    backward(&out.sum_all().unwrap()).unwrap();
    let g = input.grad().unwrap().expect("grad_input must exist");
    assert_eq!(g.shape(), &[3, 4]);
    assert_eq!(
        g.data_vec().unwrap(),
        vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    );
}

/// Live torch (src requires grad, src larger): backward RAISES
/// ("ScatterBackward0 returned an invalid gradient at index 1 - got [2, 4]
/// but expected shape compatible with [2, 5]"). Pre-fix ferrotorch silently
/// returned an index-shaped grad_src. Matching contract: structured Err.
#[test]
fn core127_cpu_scatter_larger_src_grad_src_errs_like_torch() {
    let input = input_3x4(true);
    let src = src_2x5(true);
    let out = scatter(&input, 0, &IDX_B, &[2, 4], &src).unwrap();
    let r = backward(&out.sum_all().unwrap());
    assert!(
        matches!(r, Err(FerrotorchError::InvalidArgument { .. })),
        "backward with src [2,5] != index [2,4] must be a structured error \
         (live torch RuntimeError, ScatterBackward0 invalid gradient), got {r:?}"
    );
}

/// Same oracle for scatter_add (ScatterAddBackward0 raises identically).
#[test]
fn core127_cpu_scatter_add_larger_src_grad_src_errs_like_torch() {
    let input = input_3x4(true);
    let src = src_2x5(true);
    let out = scatter_add(&input, 0, &IDX_B, &[2, 4], &src).unwrap();
    let r = backward(&out.sum_all().unwrap());
    assert!(
        matches!(r, Err(FerrotorchError::InvalidArgument { .. })),
        "scatter_add backward with src [2,5] != index [2,4] must be a \
         structured error (live torch oracle), got {r:?}"
    );
}

/// Equal-shape backward stays intact (live torch: grad_input =
/// [1,0,0,0, 0,0,1,0, 0,1,0,1], grad_src = ones(2,4)).
#[test]
fn core127_cpu_scatter_equal_shape_grads_match_torch() {
    let input = input_3x4(true);
    let d: Vec<f32> = (0..8).map(|v| (v * 10) as f32).collect();
    let src = cpu_f32(&d, &[2, 4], true);
    let out = scatter(&input, 0, &IDX_B, &[2, 4], &src).unwrap();
    backward(&out.sum_all().unwrap()).unwrap();
    assert_eq!(
        input.grad().unwrap().unwrap().data_vec().unwrap(),
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    );
    let gs = src.grad().unwrap().unwrap();
    assert_eq!(gs.shape(), &[2, 4]);
    assert_eq!(gs.data_vec().unwrap(), vec![1.0; 8]);
}

/// Live torch 2.11.0+cu130:
/// `torch.tensor(1., requires_grad=True).scatter(0, tensor([0]), tensor(2., requires_grad=True))`
/// is legal and returns a scalar; backward gives `input.grad=0`, `src.grad=1`.
#[test]
fn core127_cpu_scatter_scalar_input_scalar_src_grad_matches_torch() {
    let input = cpu_f32(&[1.0], &[], true);
    let src = cpu_f32(&[2.0], &[], true);
    let out = scatter(&input, 0, &[0], &[1], &src).unwrap();
    assert!(
        out.shape().is_empty(),
        "scalar input must produce scalar output"
    );
    assert_eq!(out.data_vec().unwrap(), vec![2.0]);

    backward(&out.sum_all().unwrap()).unwrap();
    let gi = input.grad().unwrap().unwrap();
    let gs = src.grad().unwrap().unwrap();
    assert!(gs.shape().is_empty(), "scalar src grad must stay scalar");
    assert_eq!(gi.data_vec().unwrap(), vec![0.0]);
    assert_eq!(gs.data_vec().unwrap(), vec![1.0]);
}

/// Live torch accepts the forward (`index` is 0-D, `src` is shape [1]) but the
/// derivative is index-shaped `[]`, so autograd raises an invalid-gradient
/// error for `src`. ferrotorch should surface the same structured failure.
#[test]
fn core127_cpu_scatter_index0d_src_len1_grad_src_errs_like_torch() {
    let input = cpu_f32(&[1.0], &[], true);
    let src = cpu_f32(&[2.0], &[1], true);
    let out = scatter(&input, 0, &[0], &[], &src).unwrap();
    let r = backward(&out.sum_all().unwrap());
    assert!(
        matches!(r, Err(FerrotorchError::InvalidArgument { .. })),
        "index-shaped [] grad is not compatible with src [1], got {r:?}"
    );
}

/// Live torch: scalar tensor `src` is valid for `scatter_add` with one effective
/// index element; `grad.gather(dim, index)` is reduced back to scalar `src.grad`.
#[test]
fn core127_cpu_scatter_add_scalar_src_grad_reduces_like_torch() {
    let input = cpu_f32(&[1.0], &[1], true);
    let src = cpu_f32(&[9.0], &[], true);
    let out = scatter_add(&input, 0, &[0], &[1], &src).unwrap();
    assert_eq!(out.data_vec().unwrap(), vec![10.0]);

    backward(&out.sum_all().unwrap()).unwrap();
    let gi = input.grad().unwrap().unwrap();
    let gs = src.grad().unwrap().unwrap();
    assert!(gs.shape().is_empty(), "scalar src grad must stay scalar");
    assert_eq!(gi.data_vec().unwrap(), vec![1.0]);
    assert_eq!(gs.data_vec().unwrap(), vec![1.0]);
}

/// `scatter_shape_check` returns immediately for an empty index. Backward still
/// follows PyTorch's `src: grad.gather(dim, index)` contract: a grad of shape
/// [0,3] reduces to zeros for broadcast-compatible src [1,3].
#[test]
fn core127_cpu_scatter_empty_index_broadcast_src_zero_grad_matches_torch() {
    let input = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let src = cpu_f32(&[9.0, 10.0, 11.0], &[1, 3], true);
    let out = scatter(&input, 0, &[], &[0, 3], &src).unwrap();
    assert_eq!(out.data_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    backward(&out.sum_all().unwrap()).unwrap();
    assert_eq!(
        input.grad().unwrap().unwrap().data_vec().unwrap(),
        vec![1.0; 6]
    );
    let gs = src.grad().unwrap().unwrap();
    assert_eq!(gs.shape(), &[1, 3]);
    assert_eq!(gs.data_vec().unwrap(), vec![0.0, 0.0, 0.0]);
}

/// Scalar tensor src is not a blanket escape hatch: live torch rejects a
/// non-empty rank-1 index with size 2 because `index.size(0) > ensure_nonempty_size(src, 0)`.
#[test]
fn core127_cpu_scalar_src_larger_nonempty_index_rejected_like_torch() {
    let input = cpu_f32(&[1.0, 2.0, 3.0], &[3], false);
    let src = cpu_f32(&[9.0], &[], false);
    assert_metadata_err(
        scatter(&input, 0, &[0, 1], &[2], &src).map(|t| t.data_vec()),
        "scatter scalar src index [2]",
    );
    assert_metadata_err(
        scatter_add(&input, 0, &[0, 1], &[2], &src).map(|t| t.data_vec()),
        "scatter_add scalar src index [2]",
    );
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();
    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the CORE-127 GPU pins");
        });
    }

    fn cuda(t: Tensor<f32>, rg: bool) -> Tensor<f32> {
        t.to(Device::Cuda(0)).unwrap().requires_grad_(rg)
    }

    /// Live torch CUDA (same session): `inp.cuda().scatter(0, idxB.cuda(),
    /// srcB.cuda())` -> [0,10,70,80, 0,60,6,30, 50,9,20,11] — identical to
    /// CPU. Pre-fix the CUDA kernel consumed `src[t]` as a flat prefix.
    #[test]
    fn core127_cuda_scatter_src_2x5_index_2x4_coordinate_mapped() {
        ensure_cuda_backend();
        let input = cuda(input_3x4(false), false);
        let src = cuda(src_2x5(false), false);
        let out = scatter(&input, 0, &IDX_B, &[2, 4], &src).unwrap();
        assert!(out.is_cuda(), "result must stay CUDA-resident");
        assert_eq!(
            out.cpu().unwrap().data_vec().unwrap(),
            vec![
                0.0, 10.0, 70.0, 80.0, 0.0, 60.0, 6.0, 30.0, 50.0, 9.0, 20.0, 11.0
            ],
        );
    }

    /// Live torch CUDA: scatter_add -> [0,11,72,83, 4,65,6,37, 58,9,30,11].
    #[test]
    fn core127_cuda_scatter_add_src_2x5_index_2x4_coordinate_mapped() {
        ensure_cuda_backend();
        let input = cuda(input_3x4(false), false);
        let src = cuda(src_2x5(false), false);
        let out = scatter_add(&input, 0, &IDX_B, &[2, 4], &src).unwrap();
        assert!(out.is_cuda(), "result must stay CUDA-resident");
        assert_eq!(
            out.cpu().unwrap().data_vec().unwrap(),
            vec![
                0.0, 11.0, 72.0, 83.0, 4.0, 65.0, 6.0, 37.0, 58.0, 9.0, 30.0, 11.0
            ],
        );
    }

    /// Per-axis src validation runs before the CUDA dispatch too.
    #[test]
    fn core127_cuda_scatter_src_axis_smaller_than_index_rejected() {
        ensure_cuda_backend();
        let input = cuda(input_3x4(false), false);
        // index [2,4] needs src.size(1) >= 4; src [4,2] ties the numel gate
        // (8 == 8) but violates the per-axis rule on BOTH axes' mapping.
        let src = cuda(cpu_f32(&[1.0; 8], &[4, 2], false), false);
        let r = scatter(&input, 0, &IDX_B, &[2, 4], &src);
        assert_metadata_err(
            r.map(|t| t.cpu().and_then(|c| c.data_vec())),
            "CUDA scatter index [2,4] src [4,2]",
        );
    }

    /// Live torch CUDA backward (src requires grad, larger src) raises the
    /// same ScatterBackward0 invalid-gradient error as CPU; ferrotorch must
    /// return the structured error on the CUDA lane as well.
    #[test]
    fn core127_cuda_scatter_larger_src_grad_src_errs_like_torch() {
        ensure_cuda_backend();
        let input = cuda(input_3x4(false), true);
        let src = cuda(src_2x5(false), true);
        let out = scatter(&input, 0, &IDX_B, &[2, 4], &src).unwrap();
        let r = backward(&out.sum_all().unwrap());
        assert!(
            matches!(r, Err(FerrotorchError::InvalidArgument { .. })),
            "CUDA backward with src [2,5] != index [2,4] must be a structured \
             error (live torch oracle), got {r:?}"
        );
    }

    /// Live torch CUDA (input grad only, larger src): grad_input =
    /// [1,0,0,0, 0,0,1,0, 0,1,0,1] (same as CPU oracle for idxB).
    #[test]
    fn core127_cuda_scatter_larger_src_input_only_grad_matches_torch() {
        ensure_cuda_backend();
        let input = cuda(input_3x4(false), true);
        let src = cuda(src_2x5(false), false);
        let out = scatter(&input, 0, &IDX_B, &[2, 4], &src).unwrap();
        backward(&out.sum_all().unwrap()).unwrap();
        let g = input.grad().unwrap().expect("grad_input must exist");
        assert!(
            g.is_cuda(),
            "grad_input must stay CUDA-resident (R-ORACLE-3)"
        );
        assert_eq!(
            g.cpu().unwrap().data_vec().unwrap(),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        );
    }

    #[test]
    fn core127_cuda_scatter_scalar_input_scalar_src_grad_stays_resident() {
        ensure_cuda_backend();
        let input = cuda(cpu_f32(&[1.0], &[], false), true);
        let src = cuda(cpu_f32(&[2.0], &[], false), true);
        let out = scatter(&input, 0, &[0], &[1], &src).unwrap();
        assert!(out.is_cuda(), "forward must stay CUDA-resident");
        assert!(out.shape().is_empty());
        assert_eq!(out.cpu().unwrap().data_vec().unwrap(), vec![2.0]);

        backward(&out.sum_all().unwrap()).unwrap();
        let gi = input.grad().unwrap().unwrap();
        let gs = src.grad().unwrap().unwrap();
        assert!(gi.is_cuda(), "input grad must stay CUDA-resident");
        assert!(gs.is_cuda(), "scalar src grad must stay CUDA-resident");
        assert!(gs.shape().is_empty());
        assert_eq!(gi.cpu().unwrap().data_vec().unwrap(), vec![0.0]);
        assert_eq!(gs.cpu().unwrap().data_vec().unwrap(), vec![1.0]);
    }

    #[test]
    fn core127_cuda_scatter_add_scalar_src_grad_stays_resident() {
        ensure_cuda_backend();
        let input = cuda(cpu_f32(&[1.0], &[1], false), true);
        let src = cuda(cpu_f32(&[9.0], &[], false), true);
        let out = scatter_add(&input, 0, &[0], &[1], &src).unwrap();
        assert!(out.is_cuda(), "forward must stay CUDA-resident");
        assert_eq!(out.cpu().unwrap().data_vec().unwrap(), vec![10.0]);

        backward(&out.sum_all().unwrap()).unwrap();
        let gi = input.grad().unwrap().unwrap();
        let gs = src.grad().unwrap().unwrap();
        assert!(gi.is_cuda(), "input grad must stay CUDA-resident");
        assert!(gs.is_cuda(), "scalar src grad must stay CUDA-resident");
        assert!(gs.shape().is_empty());
        assert_eq!(gi.cpu().unwrap().data_vec().unwrap(), vec![1.0]);
        assert_eq!(gs.cpu().unwrap().data_vec().unwrap(), vec![1.0]);
    }

    #[test]
    fn core127_cuda_scatter_empty_index_broadcast_src_zero_grad_stays_resident() {
        ensure_cuda_backend();
        let input = cuda(
            cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false),
            true,
        );
        let src = cuda(cpu_f32(&[9.0, 10.0, 11.0], &[1, 3], false), true);
        let out = scatter(&input, 0, &[], &[0, 3], &src).unwrap();
        assert!(out.is_cuda(), "empty-index forward must stay CUDA-resident");
        assert_eq!(
            out.cpu().unwrap().data_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        backward(&out.sum_all().unwrap()).unwrap();
        let gi = input.grad().unwrap().unwrap();
        let gs = src.grad().unwrap().unwrap();
        assert!(gi.is_cuda(), "input grad must stay CUDA-resident");
        assert!(
            gs.is_cuda(),
            "broadcast src zero-grad must stay CUDA-resident"
        );
        assert_eq!(gi.cpu().unwrap().data_vec().unwrap(), vec![1.0; 6]);
        assert_eq!(gs.shape(), &[1, 3]);
        assert_eq!(gs.cpu().unwrap().data_vec().unwrap(), vec![0.0, 0.0, 0.0]);
    }
}
