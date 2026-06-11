//! CORE-125 (#1819, CLASS-U Critical) regression battery: the safe indexing
//! APIs (`gather` / `scatter` / `scatter_value` / `scatter_add` in
//! `ferrotorch-core/src/ops/indexing.rs`) must treat the host index slice and
//! its claimed shape as ONE checked logical tensor BEFORE any CPU loop or
//! CUDA upload/dispatch.
//!
//! Violation classes pinned here (per the #1819 dispatch scope):
//!   (a) metadata coherence — `index.len() == product(index_shape)` with a
//!       CHECKED product (CORE-007 class overflow);
//!   (b) index-VALUE bounds vs the indexed axis, before any device dispatch;
//!   (c) `src` length sufficiency for the consumed element count
//!       (scatter family);
//!   (d) all of the above run BEFORE the CUDA fast paths (pre-fix the CUDA
//!       branches dispatched kernels for the shape-derived element count and
//!       read past short index/src device buffers; host `index_shape[dim]`
//!       also panicked on rank-mismatched metadata inside a fallible API).
//!
//! NOT in scope here (separate filed Highs): per-axis `index.shape[d] <=
//! input.shape[d]` constraints (CORE-126 → #1820) and coordinate-addressed
//! `src` consumption (CORE-127 → #1821).
//!
//! Upstream contract: PyTorch validates rank + shape before ANY kernel
//! dispatch — `gather_shape_check` / `scatter_shape_check` in
//! `aten/src/ATen/native/ScatterGatherChecks.h:41-124`, invoked from the meta
//! functions at `aten/src/ATen/native/TensorAdvancedIndexing.cpp:179,192`
//! (i.e. before the device kernel is selected). In PyTorch the index is a
//! real `Tensor`, so "data length == shape product" holds by construction;
//! ferrotorch's flat-slice API must enforce it explicitly. Out-of-bounds
//! index VALUES raise `RuntimeError` on the CPU device in torch; ferrotorch
//! validates values on the host for BOTH devices (the resident kernels do
//! not re-check, per `ferrotorch-gpu/src/scatter_gather_kernels.rs` module
//! note — which is exactly why the host validator must run first).
//!
//! NOTE on "negative index": the public API takes `index: &[usize]`, so a
//! negative index value is unrepresentable at this boundary; the adversarial
//! analog is a huge positive value (tested below as the out-of-bounds class).
//!
//! Pre-fix observed behavior (R-AHON-1 probe, pasted in #1819's result
//! comment): CPU short-index → `index out of bounds: the len is N but the
//! index is N` panic inside a `Result` API; CUDA short-index / OOB-value →
//! kernels launched and returned plausible garbage (pool-rounded buffers hid
//! the device OOB read); CUDA rank-mismatch → host panic at
//! `index_shape[dim]`.

use ferrotorch_core::ops::indexing::scatter_value;
use ferrotorch_core::{FerrotorchError, Tensor, TensorStorage, gather, scatter, scatter_add};

fn cpu_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f32 tensor")
}

/// Assert the result is a structured metadata error (ShapeMismatch or
/// InvalidArgument) — NOT a panic, NOT Ok-with-garbage.
#[track_caller]
fn assert_metadata_err<T: std::fmt::Debug>(r: Result<T, FerrotorchError>, what: &str) {
    match r {
        Err(FerrotorchError::ShapeMismatch { .. } | FerrotorchError::InvalidArgument { .. }) => {}
        Err(other) => panic!("{what}: expected ShapeMismatch/InvalidArgument, got Err({other:?})"),
        Ok(v) => panic!("{what}: expected structured Err, got Ok({v:?})"),
    }
}

/// Assert the result is the structured out-of-bounds index-VALUE error.
#[track_caller]
fn assert_oob_err<T: std::fmt::Debug>(r: Result<T, FerrotorchError>, what: &str) {
    match r {
        Err(FerrotorchError::IndexOutOfBounds { .. }) => {}
        Err(other) => panic!("{what}: expected IndexOutOfBounds, got Err({other:?})"),
        Ok(v) => panic!("{what}: expected IndexOutOfBounds Err, got Ok({v:?})"),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// CPU lane — class (a): index.len() != product(index_shape)
// Pre-fix: short slice panics inside the fallible API (`index[out_flat]`
// past the slice end); long slice silently computes from a prefix.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn core125_cpu_gather_short_index_slice_errs() {
    let input = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    // Claimed shape [2, 3] => 6 elements; only 4 supplied.
    let r = gather(&input, 1, &[0, 1, 2, 0], &[2, 3]);
    assert_metadata_err(r, "gather CPU short index");
}

#[test]
fn core125_cpu_gather_long_index_slice_errs() {
    let input = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    // Claimed shape [2, 1] => 2 elements; 6 supplied (pre-fix: Ok from prefix).
    let r = gather(&input, 1, &[0, 1, 2, 0, 1, 2], &[2, 1]);
    assert_metadata_err(r, "gather CPU long index");
}

#[test]
fn core125_cpu_gather_index_shape_product_overflow_errs() {
    let input = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    // product(index_shape) overflows usize (CORE-007 class). Pre-fix the
    // release-mode wrapping product drove a capacity-overflow panic / OOB
    // slice panic; post-fix this is a structured error from the checked
    // product.
    let r = gather(&input, 0, &[], &[usize::MAX, usize::MAX]);
    assert_metadata_err(r, "gather CPU index_shape product overflow");
}

#[test]
fn core125_cpu_scatter_short_index_slice_errs() {
    let input = cpu_f32(&[0.0; 6], &[2, 3]);
    let src = cpu_f32(&[9.0; 6], &[2, 3]);
    // Claimed [2, 3] => 6; only 4 index values supplied (src is full-size so
    // the pre-existing src-sufficiency check cannot save us).
    let r = scatter(&input, 1, &[0, 1, 2, 0], &[2, 3], &src);
    assert_metadata_err(r, "scatter CPU short index");
}

#[test]
fn core125_cpu_scatter_long_index_slice_errs() {
    let input = cpu_f32(&[0.0; 6], &[2, 3]);
    let src = cpu_f32(&[9.0; 6], &[2, 3]);
    let r = scatter(&input, 1, &[0, 1, 2, 0, 1, 2], &[2, 1], &src);
    assert_metadata_err(r, "scatter CPU long index");
}

#[test]
fn core125_cpu_scatter_value_short_index_slice_errs() {
    let input = cpu_f32(&[0.0; 6], &[2, 3]);
    let r = scatter_value(&input, 1, &[0, 1, 2, 0], &[2, 3], 9.0_f32);
    assert_metadata_err(r, "scatter_value CPU short index");
}

#[test]
fn core125_cpu_scatter_value_long_index_slice_errs() {
    let input = cpu_f32(&[0.0; 6], &[2, 3]);
    let r = scatter_value(&input, 1, &[0, 1, 2, 0, 1, 2], &[2, 1], 9.0_f32);
    assert_metadata_err(r, "scatter_value CPU long index");
}

#[test]
fn core125_cpu_scatter_add_short_index_slice_errs() {
    let input = cpu_f32(&[0.0; 6], &[2, 3]);
    let src = cpu_f32(&[9.0; 6], &[2, 3]);
    let r = scatter_add(&input, 1, &[0, 1, 2, 0], &[2, 3], &src);
    assert_metadata_err(r, "scatter_add CPU short index");
}

#[test]
fn core125_cpu_scatter_add_long_index_slice_errs() {
    let input = cpu_f32(&[0.0; 6], &[2, 3]);
    let src = cpu_f32(&[9.0; 6], &[2, 3]);
    let r = scatter_add(&input, 1, &[0, 1, 2, 0, 1, 2], &[2, 1], &src);
    assert_metadata_err(r, "scatter_add CPU long index");
}

// ─────────────────────────────────────────────────────────────────────────
// CPU lane — class (b): index VALUE out of bounds for the indexed axis.
// gather already had lib coverage; the scatter family is pinned here so the
// adversarial inputs from the finding stay permanent tests on every op.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn core125_cpu_gather_oob_index_value_errs() {
    let input = cpu_f32(&[1.0, 2.0, 3.0], &[3]);
    assert_oob_err(gather(&input, 0, &[99], &[1]), "gather CPU oob value");
}

#[test]
fn core125_cpu_scatter_oob_index_value_errs() {
    let input = cpu_f32(&[0.0; 3], &[3]);
    let src = cpu_f32(&[9.0], &[1]);
    assert_oob_err(
        scatter(&input, 0, &[99], &[1], &src),
        "scatter CPU oob value",
    );
}

#[test]
fn core125_cpu_scatter_value_oob_index_value_errs() {
    let input = cpu_f32(&[0.0; 3], &[3]);
    assert_oob_err(
        scatter_value(&input, 0, &[99], &[1], 9.0_f32),
        "scatter_value CPU oob value",
    );
}

#[test]
fn core125_cpu_scatter_add_oob_index_value_errs() {
    let input = cpu_f32(&[0.0; 3], &[3]);
    let src = cpu_f32(&[9.0], &[1]);
    assert_oob_err(
        scatter_add(&input, 0, &[99], &[1], &src),
        "scatter_add CPU oob value",
    );
}

// ─────────────────────────────────────────────────────────────────────────
// CPU lane — class (c): src shorter than the consumed element count.
// (Pre-existing checks on CPU; pinned permanently per the finding.)
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn core125_cpu_scatter_short_src_errs() {
    let input = cpu_f32(&[0.0; 6], &[2, 3]);
    let src = cpu_f32(&[9.0, 9.0], &[2, 1]);
    // index consumes 6 elements; src has 2.
    let r = scatter(&input, 1, &[0, 1, 2, 0, 1, 2], &[2, 3], &src);
    assert_metadata_err(r, "scatter CPU short src");
}

#[test]
fn core125_cpu_scatter_add_short_src_errs() {
    let input = cpu_f32(&[0.0; 6], &[2, 3]);
    let src = cpu_f32(&[9.0, 9.0], &[2, 1]);
    let r = scatter_add(&input, 1, &[0, 1, 2, 0, 1, 2], &[2, 3], &src);
    assert_metadata_err(r, "scatter_add CPU short src");
}

// ─────────────────────────────────────────────────────────────────────────
// CPU lane — rank-mismatched metadata stays a structured error (the same
// validator must also guard the CUDA lane, below).
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn core125_cpu_scatter_rank_mismatch_errs() {
    let input = cpu_f32(&[0.0; 6], &[2, 3]);
    let src = cpu_f32(&[9.0; 6], &[2, 3]);
    let r = scatter(&input, 1, &[0, 1, 2, 0, 1, 2], &[6], &src);
    assert_metadata_err(r, "scatter CPU rank mismatch");
}

// ─────────────────────────────────────────────────────────────────────────
// CUDA lane — classes (a)/(b)/(c)/(d). Pre-fix EVERY one of these reached
// the kernel launch (or a host `index_shape[dim]` panic) because the fast
// paths ran before `validate_gather_shapes`.
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
                .expect("CUDA backend must initialize for the CORE-125 GPU pins");
        });
    }

    fn cuda_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        cpu_f32(data, shape)
            .to(Device::Cuda(0))
            .expect("upload to cuda")
    }

    // ── gather ──

    #[test]
    fn core125_cuda_gather_short_index_slice_errs() {
        ensure_cuda_backend();
        let input = cuda_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        // Claimed [2, 3] => kernel total 6; only 4 index values uploaded.
        // Pre-fix: kernel read past the uploaded i64 buffer (device OOB,
        // masked by pool rounding) and returned plausible garbage.
        let r = gather(&input, 1, &[0, 1, 2, 0], &[2, 3]);
        assert_metadata_err(
            r.map(|t| t.cpu().and_then(|c| c.data_vec())),
            "gather CUDA short index",
        );
    }

    #[test]
    fn core125_cuda_gather_oob_index_value_errs() {
        ensure_cuda_backend();
        let input = cuda_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        // 99 >= axis size 3: pre-fix this became a raw device offset (OOB
        // device read, garbage result). Post-fix: host IndexOutOfBounds
        // BEFORE the upload.
        let r = gather(&input, 1, &[0, 1, 2, 0, 1, 99], &[2, 3]);
        assert_oob_err(
            r.map(|t| t.cpu().and_then(|c| c.data_vec())),
            "gather CUDA oob value",
        );
    }

    #[test]
    fn core125_cuda_gather_rank_mismatch_errs() {
        ensure_cuda_backend();
        let input = cuda_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        // 1-D index_shape for a 2-D input, dim=1: pre-fix the CUDA branch
        // panicked at `index_shape[dim]` (panic inside a Result API).
        let r = gather(&input, 1, &[0, 1, 2, 0, 1, 2], &[6]);
        assert_metadata_err(
            r.map(|t| t.cpu().and_then(|c| c.data_vec())),
            "gather CUDA rank mismatch",
        );
    }

    // ── scatter ──

    #[test]
    fn core125_cuda_scatter_short_index_slice_errs() {
        ensure_cuda_backend();
        let input = cuda_f32(&[0.0; 6], &[2, 3]);
        let src = cuda_f32(&[9.0; 6], &[2, 3]);
        let r = scatter(&input, 1, &[0, 1, 2, 0], &[2, 3], &src);
        assert_metadata_err(
            r.map(|t| t.cpu().and_then(|c| c.data_vec())),
            "scatter CUDA short index",
        );
    }

    #[test]
    fn core125_cuda_scatter_oob_index_value_errs() {
        ensure_cuda_backend();
        let input = cuda_f32(&[0.0; 6], &[2, 3]);
        let src = cuda_f32(&[9.0; 6], &[2, 3]);
        // Pre-fix: 99 became a raw device WRITE offset (OOB device write).
        let r = scatter(&input, 1, &[0, 1, 2, 0, 1, 99], &[2, 3], &src);
        assert_oob_err(
            r.map(|t| t.cpu().and_then(|c| c.data_vec())),
            "scatter CUDA oob value",
        );
    }

    #[test]
    fn core125_cuda_scatter_short_src_errs() {
        ensure_cuda_backend();
        let input = cuda_f32(&[0.0; 6], &[2, 3]);
        // Kernel consumes 6 src elements; src has 2 (pre-fix: device OOB read
        // past the src buffer).
        let src = cuda_f32(&[9.0, 9.0], &[2, 1]);
        let r = scatter(&input, 1, &[0, 1, 2, 0, 1, 2], &[2, 3], &src);
        assert_metadata_err(
            r.map(|t| t.cpu().and_then(|c| c.data_vec())),
            "scatter CUDA short src",
        );
    }

    #[test]
    fn core125_cuda_scatter_rank_mismatch_errs() {
        ensure_cuda_backend();
        let input = cuda_f32(&[0.0; 6], &[2, 3]);
        let src = cuda_f32(&[9.0; 6], &[2, 3]);
        // Pre-fix: host panic at `index_shape[dim]` inside the CUDA branch.
        let r = scatter(&input, 1, &[0, 1, 2, 0, 1, 2], &[6], &src);
        assert_metadata_err(
            r.map(|t| t.cpu().and_then(|c| c.data_vec())),
            "scatter CUDA rank mismatch",
        );
    }

    // ── scatter_value ──

    #[test]
    fn core125_cuda_scatter_value_short_index_slice_errs() {
        ensure_cuda_backend();
        let input = cuda_f32(&[0.0; 6], &[2, 3]);
        let r = scatter_value(&input, 1, &[0, 1, 2, 0], &[2, 3], 9.0_f32);
        assert_metadata_err(
            r.map(|t| t.cpu().and_then(|c| c.data_vec())),
            "scatter_value CUDA short index",
        );
    }

    #[test]
    fn core125_cuda_scatter_value_oob_index_value_errs() {
        ensure_cuda_backend();
        let input = cuda_f32(&[0.0; 6], &[2, 3]);
        let r = scatter_value(&input, 1, &[0, 1, 2, 0, 1, 99], &[2, 3], 9.0_f32);
        assert_oob_err(
            r.map(|t| t.cpu().and_then(|c| c.data_vec())),
            "scatter_value CUDA oob value",
        );
    }

    #[test]
    fn core125_cuda_scatter_value_rank_mismatch_errs() {
        ensure_cuda_backend();
        let input = cuda_f32(&[0.0; 6], &[2, 3]);
        let r = scatter_value(&input, 1, &[0, 1, 2, 0, 1, 2], &[6], 9.0_f32);
        assert_metadata_err(
            r.map(|t| t.cpu().and_then(|c| c.data_vec())),
            "scatter_value CUDA rank mismatch",
        );
    }

    // ── scatter_add ──

    #[test]
    fn core125_cuda_scatter_add_short_index_slice_errs() {
        ensure_cuda_backend();
        let input = cuda_f32(&[0.0; 6], &[2, 3]);
        let src = cuda_f32(&[9.0; 6], &[2, 3]);
        let r = scatter_add(&input, 1, &[0, 1, 2, 0], &[2, 3], &src);
        assert_metadata_err(
            r.map(|t| t.cpu().and_then(|c| c.data_vec())),
            "scatter_add CUDA short index",
        );
    }

    #[test]
    fn core125_cuda_scatter_add_oob_index_value_errs() {
        ensure_cuda_backend();
        let input = cuda_f32(&[0.0; 6], &[2, 3]);
        let src = cuda_f32(&[9.0; 6], &[2, 3]);
        // Pre-fix: 99 became a raw device atomic-add offset (OOB device
        // write).
        let r = scatter_add(&input, 1, &[0, 1, 2, 0, 1, 99], &[2, 3], &src);
        assert_oob_err(
            r.map(|t| t.cpu().and_then(|c| c.data_vec())),
            "scatter_add CUDA oob value",
        );
    }

    #[test]
    fn core125_cuda_scatter_add_short_src_errs() {
        ensure_cuda_backend();
        let input = cuda_f32(&[0.0; 6], &[2, 3]);
        let src = cuda_f32(&[9.0, 9.0], &[2, 1]);
        let r = scatter_add(&input, 1, &[0, 1, 2, 0, 1, 2], &[2, 3], &src);
        assert_metadata_err(
            r.map(|t| t.cpu().and_then(|c| c.data_vec())),
            "scatter_add CUDA short src",
        );
    }

    #[test]
    fn core125_cuda_scatter_add_rank_mismatch_errs() {
        ensure_cuda_backend();
        let input = cuda_f32(&[0.0; 6], &[2, 3]);
        let src = cuda_f32(&[9.0; 6], &[2, 3]);
        let r = scatter_add(&input, 1, &[0, 1, 2, 0, 1, 2], &[6], &src);
        assert_metadata_err(
            r.map(|t| t.cpu().and_then(|c| c.data_vec())),
            "scatter_add CUDA rank mismatch",
        );
    }

    // ── valid-input byte-identity smoke: the validation hoist must NOT
    //    change results for well-formed CUDA inputs (oracle: the live torch
    //    parity pins in ferrotorch-gpu/tests/divergence_scatter_gather_gpu.rs
    //    cover values; this pins the unchanged Ok path through the same
    //    public API used above). torch oracle for the gather case:
    //      torch.gather(torch.tensor([[1.,2.],[3.,4.],[5.,6.]],device='cuda'),
    //                   0, torch.tensor([[2,0],[1,1]],device='cuda'))
    //      -> tensor([[5., 2.], [3., 4.]], device='cuda:0')
    //    (same fixture as divergence_scatter_gather_gpu.rs gather 2D dim=0).

    #[test]
    fn core125_cuda_gather_valid_input_still_ok() {
        ensure_cuda_backend();
        let input = cuda_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let out = gather(&input, 0, &[2, 0, 1, 1], &[2, 2]).expect("valid gather must stay Ok");
        assert!(out.is_cuda(), "result must stay CUDA-resident");
        let host = out.cpu().expect("readback").data().unwrap().to_vec();
        assert_eq!(host, vec![5.0, 2.0, 3.0, 4.0]);
    }
}
