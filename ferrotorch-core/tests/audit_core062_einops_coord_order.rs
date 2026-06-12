//! Regression tests for audit finding CORE-062 (#1756): reordered-axis einops
//! `repeat` and `reduce` used coordinate vectors in the wrong axis order.
//!
//! Pre-fix, `einops::repeat` collected source coordinates while walking axes
//! in *right*-pattern order and then interpreted that vector with the *left*
//! (input) elementary shape; the `einops::reduce` fallback had the inverse
//! mismatch (kept coordinates in left order flattened against the right-order
//! output shape). Patterns that reorder kept axes therefore read/wrote wrong
//! elements and could index past the buffer (panic from a fallible API).
//!
//! ## Oracle (R-ORACLE-1b)
//!
//! Every expectation below was produced by the live einops library on top of
//! PyTorch — einops 0.8.2, torch 2.11.0+cu130 — via:
//!
//! ```python
//! import torch
//! from einops import repeat, reduce
//! x = torch.arange(n, dtype=torch.float32).reshape(shape)
//! repeat(x, pattern, **axes_lengths).flatten().tolist()
//! reduce(x, pattern, op).flatten().tolist()
//! ```
//!
//! The exact session output is quoted per case. All inputs are small integer
//! ramps, so f32 holds every value and every sum/mean here exactly (largest
//! magnitude 86 << 2^24); comparisons are bit-exact.
//!
//! GPU lanes (`gpu` feature) assert result device before readback per
//! R-ORACLE-3 / CORE-196.

use ferrotorch_core::einops::{EinopsReduction, reduce, repeat};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

/// CPU leaf with an integer ramp 0..n.
fn ramp(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false).unwrap()
}

// Pure data-movement (repeat) and small-integer arithmetic (reduce) — every
// oracle value is exactly representable in f32, so equality is bit-exact.
#[allow(clippy::float_cmp)]
fn assert_flat_eq(actual: &Tensor<f32>, shape: &[usize], expected: &[f32], label: &str) {
    assert_eq!(actual.shape(), shape, "{label}: shape");
    let data = actual.data().unwrap();
    assert_eq!(data, expected, "{label}: values vs live einops oracle");
}

// ---------------------------------------------------------------------------
// repeat — reorder + new axis combinations
// ---------------------------------------------------------------------------

/// Pre-fix this PANICKED: src_flat = b*3 + a reaches 7 > numel-1 (5).
/// Oracle: `repeat(arange(6).reshape(2,3), 'a b -> b a c', c=2)`
/// -> `[0, 0, 3, 3, 1, 1, 4, 4, 2, 2, 5, 5]`, shape `[3, 2, 2]`.
#[test]
fn repeat_reorder_plus_new_trailing_axis() {
    let t = ramp(&[2, 3]);
    let r = repeat(&t, "a b -> b a c", &[("c", 2)]).unwrap();
    assert_flat_eq(
        &r,
        &[3, 2, 2],
        &[0., 0., 3., 3., 1., 1., 4., 4., 2., 2., 5., 5.],
        "repeat 'a b -> b a c' [2,3]",
    );
}

/// Square input: no OOB pre-fix, but silently TRANSPOSED values.
/// Oracle: `repeat(arange(4).reshape(2,2), 'a b -> b a c', c=2)`
/// -> `[0, 0, 2, 2, 1, 1, 3, 3]`, shape `[2, 2, 2]`.
#[test]
fn repeat_reorder_square_silent_wrong_values() {
    let t = ramp(&[2, 2]);
    let r = repeat(&t, "a b -> b a c", &[("c", 2)]).unwrap();
    assert_flat_eq(
        &r,
        &[2, 2, 2],
        &[0., 0., 2., 2., 1., 1., 3., 3.],
        "repeat 'a b -> b a c' [2,2]",
    );
}

/// New axis LEADING the reordered kept axes.
/// Oracle: `repeat(arange(6).reshape(2,3), 'a b -> c b a', c=2)`
/// -> `[0, 3, 1, 4, 2, 5, 0, 3, 1, 4, 2, 5]`, shape `[2, 3, 2]`.
#[test]
fn repeat_reorder_plus_new_leading_axis() {
    let t = ramp(&[2, 3]);
    let r = repeat(&t, "a b -> c b a", &[("c", 2)]).unwrap();
    assert_flat_eq(
        &r,
        &[2, 3, 2],
        &[0., 3., 1., 4., 2., 5., 0., 3., 1., 4., 2., 5.],
        "repeat 'a b -> c b a' [2,3]",
    );
}

/// New axis inside a merge group next to a reordered kept axis.
/// Oracle: `repeat(arange(6).reshape(2,3), 'a b -> b (a c)', c=2)`
/// -> `[0, 0, 3, 3, 1, 1, 4, 4, 2, 2, 5, 5]`, shape `[3, 4]`.
#[test]
fn repeat_reorder_new_axis_in_merge_group() {
    let t = ramp(&[2, 3]);
    let r = repeat(&t, "a b -> b (a c)", &[("c", 2)]).unwrap();
    assert_flat_eq(
        &r,
        &[3, 4],
        &[0., 0., 3., 3., 1., 1., 4., 4., 2., 2., 5., 5.],
        "repeat 'a b -> b (a c)' [2,3]",
    );
}

/// Merge group whose FIRST member is a reordered kept axis, second is new.
/// Oracle: `repeat(arange(6).reshape(2,3), 'a b -> (b c) a', c=3)`
/// -> `[0, 3, 0, 3, 0, 3, 1, 4, 1, 4, 1, 4, 2, 5, 2, 5, 2, 5]`, shape `[9, 2]`.
#[test]
fn repeat_reorder_merged_kept_and_new_axis() {
    let t = ramp(&[2, 3]);
    let r = repeat(&t, "a b -> (b c) a", &[("c", 3)]).unwrap();
    assert_flat_eq(
        &r,
        &[9, 2],
        &[
            0., 3., 0., 3., 0., 3., 1., 4., 1., 4., 1., 4., 2., 5., 2., 5., 2., 5.,
        ],
        "repeat 'a b -> (b c) a' [2,3]",
    );
}

/// Split on the left, then reorder the split axes around a new axis.
/// Oracle: `repeat(arange(4), '(a b) -> b c a', b=2, c=2)`
/// -> `[0, 2, 0, 2, 1, 3, 1, 3]`, shape `[2, 2, 2]`.
#[test]
fn repeat_split_reorder_plus_new_axis() {
    let t = ramp(&[4]);
    let r = repeat(&t, "(a b) -> b c a", &[("b", 2), ("c", 2)]).unwrap();
    assert_flat_eq(
        &r,
        &[2, 2, 2],
        &[0., 2., 0., 2., 1., 3., 1., 3.],
        "repeat '(a b) -> b c a' [4]",
    );
}

/// Pure reorder through the `repeat` API (no new axes — einops allows this).
/// Oracle: `repeat(arange(6).reshape(2,3), 'a b -> b a')`
/// -> `[0, 3, 1, 4, 2, 5]`, shape `[3, 2]`.
#[test]
fn repeat_pure_reorder_no_new_axes() {
    let t = ramp(&[2, 3]);
    let r = repeat(&t, "a b -> b a", &[]).unwrap();
    assert_flat_eq(
        &r,
        &[3, 2],
        &[0., 3., 1., 4., 2., 5.],
        "repeat 'a b -> b a' [2,3]",
    );
}

/// No-reorder guard: the pre-fix code was correct for order-preserving
/// patterns; this pins that the fix does not regress them.
/// Oracle: `repeat(arange(4).reshape(2,2), 'h w -> b h w', b=3)`
/// -> `[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]`, shape `[3, 2, 2]`.
#[test]
fn repeat_no_reorder_guard() {
    let t = ramp(&[2, 2]);
    let r = repeat(&t, "h w -> b h w", &[("b", 3)]).unwrap();
    assert_flat_eq(
        &r,
        &[3, 2, 2],
        &[0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3.],
        "repeat 'h w -> b h w' [2,2]",
    );
}

// ---------------------------------------------------------------------------
// reduce — kept-axis reorder combinations (the fallback path pre-fix)
// ---------------------------------------------------------------------------

/// Kept axes reordered (c a vs a..c). Pre-fix: silent wrong positions.
/// Oracle: `reduce(arange(24).reshape(2,3,4), 'a b c -> c a', 'sum')`
/// -> `[12, 48, 15, 51, 18, 54, 21, 57]`, shape `[4, 2]`.
#[test]
fn reduce_sum_kept_reorder() {
    let t = ramp(&[2, 3, 4]);
    let r = reduce(&t, "a b c -> c a", EinopsReduction::Sum).unwrap();
    assert_flat_eq(
        &r,
        &[4, 2],
        &[12., 48., 15., 51., 18., 54., 21., 57.],
        "reduce sum 'a b c -> c a' [2,3,4]",
    );
}

/// Wide leading axis: pre-fix this PANICKED (accum write at flat index up to
/// 21 in a 10-element output buffer).
/// Oracle: `reduce(arange(30).reshape(5,3,2), 'a b c -> c a', 'sum')`
/// -> `[6, 24, 42, 60, 78, 9, 27, 45, 63, 81]`, shape `[2, 5]`.
#[test]
fn reduce_sum_kept_reorder_wide_axis() {
    let t = ramp(&[5, 3, 2]);
    let r = reduce(&t, "a b c -> c a", EinopsReduction::Sum).unwrap();
    assert_flat_eq(
        &r,
        &[2, 5],
        &[6., 24., 42., 60., 78., 9., 27., 45., 63., 81.],
        "reduce sum 'a b c -> c a' [5,3,2]",
    );
}

/// Trailing axis reduced, kept axes swapped.
/// Oracle: `reduce(arange(24).reshape(2,3,4), 'a b c -> b a', 'sum')`
/// -> `[6, 54, 22, 70, 38, 86]`, shape `[3, 2]`.
#[test]
fn reduce_sum_trailing_reduced_kept_swapped() {
    let t = ramp(&[2, 3, 4]);
    let r = reduce(&t, "a b c -> b a", EinopsReduction::Sum).unwrap();
    assert_flat_eq(
        &r,
        &[3, 2],
        &[6., 54., 22., 70., 38., 86.],
        "reduce sum 'a b c -> b a' [2,3,4]",
    );
}

/// Kept reorder + merge group on the right.
/// Oracle: `reduce(arange(24).reshape(2,3,4), 'a b c -> (c a)', 'sum')`
/// -> `[12, 48, 15, 51, 18, 54, 21, 57]`, shape `[8]`.
#[test]
fn reduce_sum_kept_reorder_merged() {
    let t = ramp(&[2, 3, 4]);
    let r = reduce(&t, "a b c -> (c a)", EinopsReduction::Sum).unwrap();
    assert_flat_eq(
        &r,
        &[8],
        &[12., 48., 15., 51., 18., 54., 21., 57.],
        "reduce sum 'a b c -> (c a)' [2,3,4]",
    );
}

/// Mean over the reordered-kept pattern. Exact in f32: every cell mean is an
/// integer (sum of 12 consecutive-step ints divided by 12).
/// Oracle: `reduce(arange(24).reshape(2,3,4), 'a b c -> c a', 'mean')`
/// -> `[4, 16, 5, 17, 6, 18, 7, 19]`, shape `[4, 2]`.
#[test]
fn reduce_mean_kept_reorder() {
    let t = ramp(&[2, 3, 4]);
    let r = reduce(&t, "a b c -> c a", EinopsReduction::Mean).unwrap();
    assert_flat_eq(
        &r,
        &[4, 2],
        &[4., 16., 5., 17., 6., 18., 7., 19.],
        "reduce mean 'a b c -> c a' [2,3,4]",
    );
}

/// Max over the reordered-kept pattern.
/// Oracle: `reduce(arange(24).reshape(2,3,4), 'a b c -> c a', 'max')`
/// -> `[8, 20, 9, 21, 10, 22, 11, 23]`, shape `[4, 2]`.
#[test]
fn reduce_max_kept_reorder() {
    let t = ramp(&[2, 3, 4]);
    let r = reduce(&t, "a b c -> c a", EinopsReduction::Max).unwrap();
    assert_flat_eq(
        &r,
        &[4, 2],
        &[8., 20., 9., 21., 10., 22., 11., 23.],
        "reduce max 'a b c -> c a' [2,3,4]",
    );
}

/// Min over the reordered-kept pattern.
/// Oracle: `reduce(arange(24).reshape(2,3,4), 'a b c -> c a', 'min')`
/// -> `[0, 12, 1, 13, 2, 14, 3, 15]`, shape `[4, 2]`.
#[test]
fn reduce_min_kept_reorder() {
    let t = ramp(&[2, 3, 4]);
    let r = reduce(&t, "a b c -> c a", EinopsReduction::Min).unwrap();
    assert_flat_eq(
        &r,
        &[4, 2],
        &[0., 12., 1., 13., 2., 14., 3., 15.],
        "reduce min 'a b c -> c a' [2,3,4]",
    );
}

/// No-reorder guard for the reduce fallback family.
/// Oracle: `reduce(arange(6).reshape(3,2), 'a b -> b', 'sum')`
/// -> `[6, 9]`, shape `[2]`.
#[test]
fn reduce_sum_no_reorder_guard() {
    let t = ramp(&[3, 2]);
    let r = reduce(&t, "a b -> b", EinopsReduction::Sum).unwrap();
    assert_flat_eq(&r, &[2], &[6., 9.], "reduce sum 'a b -> b' [3,2]");
}

// ---------------------------------------------------------------------------
// GPU lanes — same oracle, device-checked results (R-ORACLE-3 / CORE-196)
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::device::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the gpu einops audit tests");
        });
    }

    /// Device-checked readback: result must be CUDA-resident (no silent CPU
    /// fallback per CORE-196), then D2H for value comparison.
    fn cuda_flat(t: &Tensor<f32>) -> Vec<f32> {
        assert_eq!(
            t.device(),
            Device::Cuda(0),
            "einops result expected on cuda:0 but resides on {:?}",
            t.device()
        );
        t.cpu().expect("D2H readback").data().unwrap().to_vec()
    }

    // Pure data movement / exact small-int arithmetic — bit-exact (see module doc).
    #[allow(clippy::float_cmp)]
    fn check(label: &str, t: &Tensor<f32>, shape: &[usize], expected: &[f32]) {
        assert_eq!(t.shape(), shape, "{label}: shape");
        assert_eq!(cuda_flat(t), expected, "{label}: values vs einops oracle");
    }

    /// Oracle values identical to the CPU lanes above (same quoted session).
    #[test]
    fn gpu_repeat_reorder_plus_new_trailing_axis() {
        ensure_cuda_backend();
        let t = ramp(&[2, 3]).to(Device::Cuda(0)).unwrap();
        let r = repeat(&t, "a b -> b a c", &[("c", 2)]).unwrap();
        check(
            "gpu repeat 'a b -> b a c' [2,3]",
            &r,
            &[3, 2, 2],
            &[0., 0., 3., 3., 1., 1., 4., 4., 2., 2., 5., 5.],
        );
    }

    #[test]
    fn gpu_reduce_sum_kept_reorder() {
        ensure_cuda_backend();
        let t = ramp(&[2, 3, 4]).to(Device::Cuda(0)).unwrap();
        let r = reduce(&t, "a b c -> c a", EinopsReduction::Sum).unwrap();
        check(
            "gpu reduce sum 'a b c -> c a' [2,3,4]",
            &r,
            &[4, 2],
            &[12., 48., 15., 51., 18., 54., 21., 57.],
        );
    }

    #[test]
    fn gpu_reduce_max_kept_reorder() {
        ensure_cuda_backend();
        let t = ramp(&[2, 3, 4]).to(Device::Cuda(0)).unwrap();
        let r = reduce(&t, "a b c -> c a", EinopsReduction::Max).unwrap();
        check(
            "gpu reduce max 'a b c -> c a' [2,3,4]",
            &r,
            &[4, 2],
            &[8., 20., 9., 21., 10., 22., 11., 23.],
        );
    }
}
