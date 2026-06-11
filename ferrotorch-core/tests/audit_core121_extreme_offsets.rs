//! CORE-121 (#1815, CLASS-U High) regression battery: extreme `i64` diagonal
//! offsets through `triu` / `tril` / `diag`
//! (`ferrotorch-core/src/ops/tensor_ops.rs`) must mask/size WITHOUT signed
//! overflow — degenerate offsets select everything or nothing per torch, and
//! unrepresentable 1-D embed sizes are a checked structured `Err`, never a
//! panic or a wrapped allocation.
//!
//! Pre-fix mechanisms:
//!   (a) `triu`/`tril` CPU masks evaluate `(r as i64) + diagonal`, which
//!       overflows for offsets within `rows` of `i64::MAX`/`i64::MIN`
//!       (release: wraps and keeps/zeros the WRONG rows; debug: panics);
//!   (b) 2-D `diag` negates the offset (`(-diagonal) as usize`) — `i64::MIN`
//!       is unnegatable (debug panic, release wrap);
//!   (c) 1-D `diag` computes `size = n + |k|` and `size * size` UNCHECKED —
//!       huge offsets wrap the element count and then scatter out of bounds
//!       (CPU) or dispatch a CUDA kernel sized from the wrapped count.
//!
//! LIVE torch 2.11.0+cu130 oracle (quoted; `m = torch.arange(9.).reshape(3,3)`):
//! ```text
//! >>> torch.triu(m, 2**63 - 1).flatten().tolist()
//! [0.0]*9                       # degenerate: select nothing
//! >>> torch.tril(m, 2**63 - 1).flatten().tolist()
//! [0.0, 1.0, ..., 8.0]          # degenerate: keep everything
//! >>> torch.triu(m, -2**63).flatten().tolist()
//! [0.0, 1.0, ..., 8.0]          # keep everything
//! >>> torch.tril(m, -2**63).flatten().tolist()
//! [0.0]*9                       # select nothing
//! >>> [list(torch.diag(m, d).shape) for d in (2**63-1, -2**63, 2**62, -2**62)]
//! [[0], [0], [0], [0]]          # 2-D extract: empty diagonal
//! >>> torch.diag(torch.ones(2), 2**62)
//! RuntimeError: Storage size calculation overflowed with sizes=[4611686018427387906, 4611686018427387906]
//! >>> torch.diag(torch.ones(2), 2**63 - 1)
//! RuntimeError: IntArrayRef contains an int that cannot be represented as a SymInt
//! ```
//! (torch errors on every unrepresentable 1-D embed size; ferrotorch's
//! contract is the structured `InvalidArgument` analog.)

use ferrotorch_core::ops::tensor_ops::{diag, tril, triu};
use ferrotorch_core::{FerrotorchError, Tensor, TensorStorage};

/// `[3, 3]` test matrix `arange(9)`.
fn m_3x3() -> Tensor<f32> {
    let data: Vec<f32> = (0..9).map(|i| i as f32).collect();
    Tensor::from_storage(TensorStorage::cpu(data), vec![3, 3], false).expect("cpu 3x3")
}

fn v_arange9() -> Vec<f32> {
    (0..9).map(|i| i as f32).collect()
}

fn v_zeros9() -> Vec<f32> {
    vec![0.0; 9]
}

/// Assert a structured `InvalidArgument` — NOT a panic, NOT Ok-with-garbage.
#[track_caller]
fn assert_invalid_arg<T: std::fmt::Debug>(r: Result<T, FerrotorchError>, what: &str) {
    match r {
        Err(FerrotorchError::InvalidArgument { .. }) => {}
        Err(other) => panic!("{what}: expected InvalidArgument, got Err({other:?})"),
        Ok(v) => panic!("{what}: expected InvalidArgument Err, got Ok({v:?})"),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// (a) triu/tril CPU masks at extreme offsets.
// Pre-fix observed (release): `(r as i64) + diagonal` wraps —
// triu(3x3, i64::MAX) keeps rows 1..3 (torch: all zeros), tril(3x3, i64::MAX)
// zeros rows 1..3 (torch: identity copy).
// ─────────────────────────────────────────────────────────────────────────

// reason: triu/tril are pure copy-or-zero masks — no arithmetic on the kept
// elements, so exact compare is correct.
#[allow(clippy::float_cmp)]
#[test]
fn triu_tril_cpu_extreme_offsets_degenerate() {
    let m = m_3x3();
    for (name, r, expected) in [
        ("triu(i64::MAX)", triu(&m, i64::MAX), v_zeros9()),
        ("triu(i64::MAX-1)", triu(&m, i64::MAX - 1), v_zeros9()),
        ("tril(i64::MAX)", tril(&m, i64::MAX), v_arange9()),
        ("tril(i64::MAX-1)", tril(&m, i64::MAX - 1), v_arange9()),
        ("triu(i64::MIN)", triu(&m, i64::MIN), v_arange9()),
        ("triu(i64::MIN+1)", triu(&m, i64::MIN + 1), v_arange9()),
        ("tril(i64::MIN)", tril(&m, i64::MIN), v_zeros9()),
        ("tril(i64::MIN+1)", tril(&m, i64::MIN + 1), v_zeros9()),
        // large finite offsets (no overflow at HEAD — pinned for permanence)
        ("triu(2^62)", triu(&m, 1 << 62), v_zeros9()),
        ("tril(2^62)", tril(&m, 1 << 62), v_arange9()),
        ("triu(-2^62)", triu(&m, -(1 << 62)), v_arange9()),
        ("tril(-2^62)", tril(&m, -(1 << 62)), v_zeros9()),
    ] {
        let t = r.unwrap_or_else(|e| panic!("{name}: expected Ok, got {e:?}"));
        assert_eq!(t.shape(), &[3, 3], "{name} shape");
        assert_eq!(t.data().expect("cpu data").to_vec(), expected, "{name}");
    }
}

// ─────────────────────────────────────────────────────────────────────────
// (b) 2-D diag extraction at extreme offsets → empty (oracle shape [0]).
// Pre-fix: `(-diagonal)` is a debug-panic / release-wrap for i64::MIN.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn diag_cpu_2d_extreme_offsets_empty() {
    let m = m_3x3();
    for d in [i64::MAX, i64::MIN, 1 << 62, -(1 << 62)] {
        let r = diag(&m, d).unwrap_or_else(|e| panic!("diag(3x3, {d}): expected Ok, got {e:?}"));
        assert_eq!(r.shape(), &[0], "diag(3x3, {d}) shape");
    }
}

// ─────────────────────────────────────────────────────────────────────────
// (c) 1-D diag embed with unrepresentable sizes → checked structured Err.
// Pre-fix observed (release): `size * size` wraps the element count, the
// scatter then indexes out of bounds → panic inside a fallible API.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn diag_cpu_1d_huge_offsets_checked_err() {
    let v = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0]), vec![2], false)
        .expect("cpu 1d");
    for d in [i64::MAX, i64::MIN, 1 << 62, -(1 << 62)] {
        assert_invalid_arg(diag(&v, d), &format!("diag(1d, {d})"));
    }
}

/// Moderate offsets must keep working: oracle
/// `torch.diag(torch.tensor([1.,2.]), 3)` → `[5,5]` with `1.` at `(0,3)`,
/// `2.` at `(1,4)`; offset `-3` mirrors to `(3,0)`, `(4,1)`.
// reason: diag embed is a pure scatter — bit-identical elements, exact compare.
#[allow(clippy::float_cmp)]
#[test]
fn diag_cpu_1d_moderate_offsets_still_work() {
    let v = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0]), vec![2], false)
        .expect("cpu 1d");

    let r = diag(&v, 3).expect("diag(1d, 3)");
    assert_eq!(r.shape(), &[5, 5]);
    let d = r.data().expect("cpu data");
    assert_eq!(d[3], 1.0);
    assert_eq!(d[5 + 4], 2.0);
    assert_eq!(d.iter().filter(|&&x| x != 0.0).count(), 2);

    let r = diag(&v, -3).expect("diag(1d, -3)");
    assert_eq!(r.shape(), &[5, 5]);
    let d = r.data().expect("cpu data");
    assert_eq!(d[3 * 5], 1.0);
    assert_eq!(d[4 * 5 + 1], 2.0);
    assert_eq!(d.iter().filter(|&&x| x != 0.0).count(), 2);
}

// ─────────────────────────────────────────────────────────────────────────
// CUDA lane: same extreme-offset battery. The PTX kernels clamp the offset
// into i32 with degenerate-preserving semantics
// (`ferrotorch-gpu/src/triangular.rs` / `diag.rs` module notes), so the
// kernels are sound — the core-side size math before dispatch is what must
// not overflow. 1-D embed with an unrepresentable size must Err BEFORE any
// kernel launch (pre-fix it dispatched a kernel sized from the wrapped
// count — an out-of-bounds device write).
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
                .expect("CUDA backend must initialize for the CORE-121 GPU pins");
        });
    }

    fn cuda_3x3() -> Tensor<f32> {
        m_3x3().to(Device::Cuda(0)).expect("upload 3x3")
    }

    // reason: pure copy-or-zero masks — exact compare (see CPU note).
    #[allow(clippy::float_cmp)]
    #[test]
    fn triu_tril_cuda_extreme_offsets_degenerate() {
        ensure_cuda_backend();
        let m = cuda_3x3();
        for (name, r, expected) in [
            ("triu(i64::MAX)", triu(&m, i64::MAX), v_zeros9()),
            ("tril(i64::MAX)", tril(&m, i64::MAX), v_arange9()),
            ("triu(i64::MIN)", triu(&m, i64::MIN), v_arange9()),
            ("tril(i64::MIN)", tril(&m, i64::MIN), v_zeros9()),
            ("triu(2^62)", triu(&m, 1 << 62), v_zeros9()),
            ("tril(-2^62)", tril(&m, -(1 << 62)), v_zeros9()),
        ] {
            let t = r.unwrap_or_else(|e| panic!("{name} on cuda: expected Ok, got {e:?}"));
            assert_eq!(t.device(), Device::Cuda(0), "{name} output device");
            let back = t.cpu().expect("D2H readback");
            assert_eq!(
                back.data().expect("cpu data").to_vec(),
                expected,
                "{name} cuda"
            );
        }
    }

    #[test]
    fn diag_cuda_2d_extreme_offsets_empty() {
        ensure_cuda_backend();
        let m = cuda_3x3();
        for d in [i64::MAX, i64::MIN, 1 << 62, -(1 << 62)] {
            let r = diag(&m, d)
                .unwrap_or_else(|e| panic!("diag(3x3, {d}) on cuda: expected Ok, got {e:?}"));
            assert_eq!(r.device(), Device::Cuda(0), "diag(3x3, {d}) output device");
            assert_eq!(r.shape(), &[0], "diag(3x3, {d}) cuda shape");
        }
    }

    #[test]
    fn diag_cuda_1d_huge_offsets_checked_err_before_dispatch() {
        ensure_cuda_backend();
        let v = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0]), vec![2], false)
            .expect("cpu 1d")
            .to(Device::Cuda(0))
            .expect("upload 1d");
        for d in [i64::MAX, i64::MIN, 1 << 62, -(1 << 62)] {
            assert_invalid_arg(diag(&v, d), &format!("diag(cuda 1d, {d})"));
        }
    }
}
