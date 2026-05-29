//! Re-audit (#1659): VALUE-correctness + multi-size guard for the GPU
//! `masked_invalid` / `masked_equal` pooled-buffer truncation fix
//! (`ferrotorch-core/src/masked.rs:564` `predicate_mask_gpu` now takes `numel`
//! and returns `bytes.iter().take(numel)`).
//!
//! The original pin (`divergence_masked_invalid_equal_gpu.rs`) proved the
//! constructor no longer ERRORS (length 256 != 6). This guard proves the
//! returned mask carries the CORRECT predicate VALUES in the correct flat
//! order — not merely the right length — across:
//!
//!   - mixed NaN / +inf / -inf / finite (IEEE isfinite semantics),
//!   - both a NON-contiguous (transposed) CUDA view AND a contiguous offset-0
//!     CUDA input (the two distinct buffer-layout paths into
//!     `predicate_mask_gpu`),
//!   - `masked_equal` scalar edge cases: -0.0 vs +0.0 (IEEE `-0.0 == +0.0`)
//!     and NaN-never-equal,
//!   - sizes straddling `ROUND_ELEMENTS = 256` (`pool.rs:114`): 6, 255, 256,
//!     257, 300 — the `.take(numel)` must keep exactly the first `numel`
//!     predicates and discard the pooled tail at every boundary, including the
//!     raw==numel case (256) where the truncation is a no-op.
//!
//! # Oracle (R-CHAR-3 — no tautological assertions)
//!
//! Every expected mask is produced by the CPU `masked_invalid` /
//! `masked_equal` reference walk (`masked.rs:503-511` `f.is_finite()`;
//! `masked.rs:544-546` `v != value`) on the equivalent CPU tensor — the
//! torch-convention reference, NOT copied from the GPU result. `isfinite`
//! mirrors `aten/src/ATen/native/TensorCompare.cpp:484`
//! (`(self == self) * (self.abs() != inf)`). For the small fixed cases the CPU
//! reference is additionally cross-checked against a hand-derived literal so a
//! silent CPU regression cannot hide a GPU regression.
//!
//! Tracking: #1659. LEFT UNMARKED (not `#[ignore]`): a wrong predicate value or
//! a wrong-size mask on a valid CUDA input is a release block, and the test
//! failing IS the block.

#![cfg(feature = "gpu")]

use ferrotorch_core::masked::{masked_equal, masked_invalid};
use ferrotorch_core::{Device, Tensor, TensorStorage};
use std::sync::Once;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the #1659 re-audit");
    });
}

fn cpu_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f32 tensor")
}

fn cpu_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f64 tensor")
}

/// CPU isfinite reference, independent of `masked.rs`, used only to sanity-pin
/// the small fixed cases so a CPU-path regression can't mask a GPU regression.
fn ref_isfinite(data: &[f32]) -> Vec<bool> {
    data.iter().map(|v| v.is_finite()).collect()
}

// ─────────────────────────────────────────────────────────────────────────
// (1) masked_invalid VALUE correctness — non-contiguous (transposed) view
// ─────────────────────────────────────────────────────────────────────────

/// Mixed NaN / +inf / -inf / finite, on a transposed [2,3]->[3,2] CUDA view
/// (the original repro). The GPU mask must equal the CPU reference in LOGICAL
/// (transposed) flat order — proving `.contiguous()` materialised the logical
/// order into the first `numel` flat positions and `.take(numel)` kept them.
#[test]
fn reaudit_1659_masked_invalid_gpu_transposed_values() {
    ensure_cuda_backend();

    // [2,3] row-major: [ [n0, +inf, n2], [-inf, nan, n5] ]
    let base = [1.0_f32, f32::INFINITY, 3.0, f32::NEG_INFINITY, f32::NAN, 6.0];

    let cpu_t = cpu_f32(&base, &[2, 3]).transpose(0, 1).expect("cpu transpose");
    assert!(!cpu_t.is_contiguous());
    assert_eq!(cpu_t.numel(), 6);
    let expected: Vec<bool> = masked_invalid(cpu_t).expect("cpu ref").mask().to_vec();
    // Transposed logical order is [1, -inf, +inf, nan, 3, 6];
    // isfinite -> [T, F, F, F, T, T].
    assert_eq!(
        expected,
        vec![true, false, false, false, true, true],
        "CPU isfinite reference (transposed order) sanity"
    );

    let gpu_t = cpu_f32(&base, &[2, 3])
        .to(Device::Cuda(0))
        .expect("upload")
        .transpose(0, 1)
        .expect("gpu transpose");
    assert!(gpu_t.is_cuda() && !gpu_t.is_contiguous());

    let mt = masked_invalid(gpu_t).expect("gpu masked_invalid must not error (#1659)");
    assert_eq!(
        mt.mask(),
        expected.as_slice(),
        "GPU transposed isfinite VALUES must match the CPU reference, not the \
         base-buffer order or a pooled-tail readback (#1659)"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// (2) masked_invalid VALUE correctness — contiguous offset-0 CUDA input
// ─────────────────────────────────────────────────────────────────────────

/// The OTHER layout path: a plain contiguous offset-0 CUDA tensor (the
/// `.contiguous()` no-op branch where the raw slice may still be pool-rounded).
/// `.take(numel)` must still yield exactly the `numel` correct predicates.
#[test]
fn reaudit_1659_masked_invalid_gpu_contiguous_values() {
    ensure_cuda_backend();

    let base = [
        f32::NAN,
        2.0,
        f32::INFINITY,
        4.0,
        f32::NEG_INFINITY,
        6.0,
        7.0,
    ];
    let expected = ref_isfinite(&base);
    assert_eq!(
        expected,
        vec![false, true, false, true, false, true, true],
        "hand-derived isfinite sanity (contiguous order)"
    );
    // Cross-pin against the CPU masked.rs path too.
    let cpu_mask = masked_invalid(cpu_f32(&base, &[7]))
        .expect("cpu ref")
        .mask()
        .to_vec();
    assert_eq!(cpu_mask, expected, "CPU masked.rs path agrees with reference");

    let gpu = cpu_f32(&base, &[7]).to(Device::Cuda(0)).expect("upload");
    assert!(gpu.is_cuda() && gpu.is_contiguous());
    let mt = masked_invalid(gpu).expect("gpu masked_invalid contiguous must not error");
    assert_eq!(
        mt.mask(),
        expected.as_slice(),
        "GPU contiguous isfinite VALUES must match the CPU reference (#1659)"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// (3) masked_equal VALUE correctness — scalar edges (-0.0/+0.0, NaN), both layouts
// ─────────────────────────────────────────────────────────────────────────

/// `masked_equal(0.0)` over a buffer mixing -0.0, +0.0, NaN and other values,
/// on BOTH a transposed view and a contiguous input. IEEE: `-0.0 == +0.0` so
/// BOTH zero variants are masked (valid=false); `NaN != 0.0` so NaN is valid.
/// The GPU `setp.neu.f32` (`masked_kernels.rs`) must match the CPU `v != value`
/// walk for every one of these edges.
#[test]
fn reaudit_1659_masked_equal_gpu_signed_zero_and_nan() {
    ensure_cuda_backend();

    // [2,3]: [ [-0.0, +0.0, nan], [1.0, 0.0, 2.0] ]
    let base = [-0.0_f32, 0.0, f32::NAN, 1.0, 0.0, 2.0];
    let value = 0.0_f32;

    // --- transposed layout ---
    let cpu_t = cpu_f32(&base, &[2, 3]).transpose(0, 1).expect("cpu transpose");
    let expected_t: Vec<bool> = masked_equal(cpu_t, value).expect("cpu ref").mask().to_vec();
    // Transposed logical order [-0.0, 1.0, +0.0, 0.0, nan, 2.0];
    // valid = (v != 0.0): [-0.0->F, 1.0->T, +0.0->F, 0.0->F, nan->T, 2.0->T].
    assert_eq!(
        expected_t,
        vec![false, true, false, false, true, true],
        "CPU masked_equal(0.0) sanity: -0.0 and +0.0 both masked; NaN valid"
    );
    let gpu_t = cpu_f32(&base, &[2, 3])
        .to(Device::Cuda(0))
        .expect("upload")
        .transpose(0, 1)
        .expect("gpu transpose");
    let mt_t = masked_equal(gpu_t, value).expect("gpu masked_equal transposed");
    assert_eq!(
        mt_t.mask(),
        expected_t.as_slice(),
        "GPU masked_equal(0.0) transposed: signed-zero/NaN edges must match CPU (#1659)"
    );

    // --- contiguous layout ---
    let cpu_c = cpu_f32(&base, &[6]);
    let expected_c: Vec<bool> = masked_equal(cpu_c, value).expect("cpu ref").mask().to_vec();
    assert_eq!(
        expected_c,
        vec![false, false, true, true, false, true],
        "CPU masked_equal(0.0) contiguous order sanity"
    );
    let gpu_c = cpu_f32(&base, &[6]).to(Device::Cuda(0)).expect("upload");
    let mt_c = masked_equal(gpu_c, value).expect("gpu masked_equal contiguous");
    assert_eq!(
        mt_c.mask(),
        expected_c.as_slice(),
        "GPU masked_equal(0.0) contiguous: signed-zero/NaN edges must match CPU (#1659)"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// (4) Multi-size straddle of ROUND_ELEMENTS = 256
// ─────────────────────────────────────────────────────────────────────────

/// For each size straddling the 256-element pool granularity, build a tensor
/// with deterministic NaN/inf positions, run GPU `masked_invalid`, and assert
/// the returned mask is EXACTLY `numel` long AND value-equal to the CPU
/// reference at every position. This catches an off-by-one at a boundary
/// (e.g. keeping 256 when numel is 255/257) or keeping the wrong slice.
#[test]
fn reaudit_1659_masked_invalid_gpu_size_straddle_256() {
    ensure_cuda_backend();

    for &numel in &[6usize, 255, 256, 257, 300] {
        // Deterministic: every 5th element NaN, every 7th +inf, every 11th -inf,
        // otherwise a finite value. Distinct prime strides so positions vary.
        let data: Vec<f32> = (0..numel)
            .map(|i| {
                if i % 5 == 0 {
                    f32::NAN
                } else if i % 7 == 0 {
                    f32::INFINITY
                } else if i % 11 == 0 {
                    f32::NEG_INFINITY
                } else {
                    (i as f32) + 0.5
                }
            })
            .collect();

        let expected = ref_isfinite(&data);
        // Cross-pin CPU masked.rs path.
        let cpu_mask = masked_invalid(cpu_f32(&data, &[numel]))
            .expect("cpu ref")
            .mask()
            .to_vec();
        assert_eq!(
            cpu_mask, expected,
            "CPU masked.rs path matches reference at numel={numel}"
        );

        let gpu = cpu_f32(&data, &[numel]).to(Device::Cuda(0)).expect("upload");
        let mt = masked_invalid(gpu).unwrap_or_else(|e| {
            panic!("gpu masked_invalid must not error at numel={numel}: {e:?}")
        });
        assert_eq!(
            mt.mask().len(),
            numel,
            "GPU mask length must equal numel={numel}, not the pooled rounded len (#1659)"
        );
        assert_eq!(
            mt.mask(),
            expected.as_slice(),
            "GPU isfinite VALUES at numel={numel} must match the CPU reference; \
             a boundary off-by-one or wrong-slice retention fails here (#1659)"
        );
    }
}

/// Same size straddle for `masked_equal` to cover the scalar predicate launcher
/// (`launch_predicate_scalar`) at the 256 boundaries.
#[test]
fn reaudit_1659_masked_equal_gpu_size_straddle_256() {
    ensure_cuda_backend();

    let value = 42.0_f32;
    for &numel in &[6usize, 255, 256, 257, 300] {
        // Plant the target value at deterministic positions so masking is
        // non-trivial at every size.
        let data: Vec<f32> = (0..numel)
            .map(|i| if i % 13 == 0 { 42.0 } else { (i as f32) - 3.0 })
            .collect();

        let expected: Vec<bool> = data.iter().map(|&v| v != value).collect();
        let cpu_mask = masked_equal(cpu_f32(&data, &[numel]), value)
            .expect("cpu ref")
            .mask()
            .to_vec();
        assert_eq!(
            cpu_mask, expected,
            "CPU masked_equal path matches reference at numel={numel}"
        );

        let gpu = cpu_f32(&data, &[numel]).to(Device::Cuda(0)).expect("upload");
        let mt = masked_equal(gpu, value)
            .unwrap_or_else(|e| panic!("gpu masked_equal must not error at numel={numel}: {e:?}"));
        assert_eq!(
            mt.mask().len(),
            numel,
            "GPU masked_equal mask length must equal numel={numel} (#1659)"
        );
        assert_eq!(
            mt.mask(),
            expected.as_slice(),
            "GPU masked_equal VALUES at numel={numel} must match the CPU reference (#1659)"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// (5) f64 multi-size — confirm the width-independent path is also value-correct
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn reaudit_1659_masked_invalid_f64_gpu_size_straddle_256() {
    ensure_cuda_backend();

    for &numel in &[255usize, 256, 257] {
        let data: Vec<f64> = (0..numel)
            .map(|i| {
                if i % 5 == 0 {
                    f64::NAN
                } else if i % 7 == 0 {
                    f64::NEG_INFINITY
                } else {
                    (i as f64) + 0.25
                }
            })
            .collect();
        let expected: Vec<bool> = data.iter().map(|v| v.is_finite()).collect();

        let gpu = cpu_f64(&data, &[numel]).to(Device::Cuda(0)).expect("upload f64");
        let mt = masked_invalid(gpu)
            .unwrap_or_else(|e| panic!("gpu f64 masked_invalid at numel={numel}: {e:?}"));
        assert_eq!(mt.mask().len(), numel, "f64 mask length numel={numel}");
        assert_eq!(
            mt.mask(),
            expected.as_slice(),
            "GPU f64 isfinite VALUES at numel={numel} must match the IEEE reference (#1659)"
        );
    }
}
