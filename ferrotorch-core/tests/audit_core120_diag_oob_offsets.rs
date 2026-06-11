//! CORE-120 (#1814, CLASS-U High) regression battery: 2-D `diag` extraction
//! (`ferrotorch-core/src/ops/tensor_ops.rs`) must return an EMPTY 1-D tensor
//! for valid diagonal offsets at/beyond the matrix edges, on every path.
//!
//! Pre-fix, the CPU extraction path computed `rows - start_r` and
//! `cols - start_c` with UNCHECKED subtraction: an offset beyond the matrix
//! bounds underflowed (wrapping in release) and drove out-of-bounds slice
//! indexing — a panic inside a fallible (`FerrotorchResult`) API. The CUDA
//! path already used `saturating_sub` for the same length calculation, so the
//! two devices disagreed (CUDA: empty result; CPU: panic).
//!
//! Upstream contract (LIVE torch 2.11.0+cu130 oracle, quoted per case below):
//! `torch.diag` clamps the diagonal length to zero outside the matrix —
//! the length is `min(rows - start_r, cols - start_c)` with non-negative
//! clamping, per `aten/src/ATen/native/TensorShape.cpp` `diag` sizing
//! (`sz = std::max<int64_t>(0, ...)` in `apply_diag`).
//!
//! ```text
//! >>> a = torch.arange(12.).reshape(3,4)
//! >>> [list(torch.diag(a, d).shape) for d in (3, 4, 5, 100, -2, -3, -4, -100)]
//! [[1], [0], [0], [0], [1], [0], [0], [0]]
//! >>> torch.diag(a, 3).tolist(), torch.diag(a, -2).tolist()
//! ([3.0], [8.0])
//! ```

use ferrotorch_core::ops::tensor_ops::diag;
use ferrotorch_core::{Tensor, TensorStorage};

/// `[3, 4]` test matrix `arange(12)` — rows `[0..4)`, `[4..8)`, `[8..12)`.
fn a_3x4() -> Tensor<f32> {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    Tensor::from_storage(TensorStorage::cpu(data), vec![3, 4], false).expect("cpu 3x4")
}

/// Assert `diag(a, d)` is `Ok` with exactly the oracle's 1-D shape + data.
#[track_caller]
fn assert_diag_cpu(d: i64, expected: &[f32]) {
    let r = diag(&a_3x4(), d).unwrap_or_else(|e| panic!("diag(3x4, {d}): expected Ok, got {e:?}"));
    assert_eq!(r.shape(), &[expected.len()], "diag(3x4, {d}) shape");
    // Pure gather — bit-identical to the source elements, so exact compare.
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(
            r.data().expect("cpu data").to_vec(),
            expected,
            "diag(3x4, {d}) data"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// CPU lane. Pre-fix observed (R-AHON-1 probe): offsets beyond either edge
// panic with `index out of bounds` (release: the unchecked `cols - start_c`
// wraps, then the gather loop reads past the data slice; debug: the
// subtraction itself panics with `attempt to subtract with overflow`).
// ─────────────────────────────────────────────────────────────────────────

/// Just inside the upper edge: oracle `torch.diag(a, 3).tolist() == [3.0]`.
#[test]
fn diag_cpu_just_inside_upper_edge() {
    assert_diag_cpu(3, &[3.0]);
}

/// Exactly at the upper edge (`d == cols`): oracle shape `[0]`.
#[test]
fn diag_cpu_at_upper_edge_is_empty() {
    assert_diag_cpu(4, &[]);
}

/// Beyond the upper edge: oracle shape `[0]`. Pre-fix: CPU panic.
#[test]
fn diag_cpu_beyond_upper_edge_is_empty() {
    assert_diag_cpu(5, &[]);
    assert_diag_cpu(100, &[]);
}

/// Just inside the lower edge: oracle `torch.diag(a, -2).tolist() == [8.0]`.
#[test]
fn diag_cpu_just_inside_lower_edge() {
    assert_diag_cpu(-2, &[8.0]);
}

/// Exactly at the lower edge (`-d == rows`): oracle shape `[0]`.
#[test]
fn diag_cpu_at_lower_edge_is_empty() {
    assert_diag_cpu(-3, &[]);
}

/// Beyond the lower edge: oracle shape `[0]`. Pre-fix: CPU panic.
#[test]
fn diag_cpu_beyond_lower_edge_is_empty() {
    assert_diag_cpu(-4, &[]);
    assert_diag_cpu(-100, &[]);
}

// ─────────────────────────────────────────────────────────────────────────
// CUDA lane: the same offset battery must agree with CPU and torch
// (`torch.diag` on cuda returns the same empty/1-element results), and the
// result must stay CUDA-resident (R-ORACLE-3 device assertion).
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
                .expect("CUDA backend must initialize for the CORE-120 GPU pins");
        });
    }

    #[track_caller]
    fn assert_diag_cuda(d: i64, expected: &[f32]) {
        let a = a_3x4().to(Device::Cuda(0)).expect("upload 3x4");
        let r =
            diag(&a, d).unwrap_or_else(|e| panic!("diag(3x4, {d}) on cuda: expected Ok, {e:?}"));
        assert_eq!(r.device(), Device::Cuda(0), "diag(3x4, {d}) output device");
        assert_eq!(r.shape(), &[expected.len()], "diag(3x4, {d}) cuda shape");
        let back = r.cpu().expect("D2H readback");
        // Pure gather — bit-identical, exact compare.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(
                back.data().expect("cpu data").to_vec(),
                expected,
                "diag(3x4, {d}) cuda data"
            );
        }
    }

    /// Full edge battery on CUDA — same oracles as the CPU lane.
    #[test]
    fn diag_cuda_edge_offsets_match_torch() {
        ensure_cuda_backend();
        assert_diag_cuda(3, &[3.0]);
        assert_diag_cuda(4, &[]);
        assert_diag_cuda(5, &[]);
        assert_diag_cuda(100, &[]);
        assert_diag_cuda(-2, &[8.0]);
        assert_diag_cuda(-3, &[]);
        assert_diag_cuda(-4, &[]);
        assert_diag_cuda(-100, &[]);
    }
}
