//! Regression tests for CORE-155 / crosslink #1849 (CLASS-U).
//!
//! `irfft_norm` computed `n.unwrap_or(2 * (half_n - 1))` eagerly after rank
//! checks: a `[0, 2]` input (zero-length frequency axis) underflowed
//! `half_n - 1`, panicking inside the fallible API in debug builds (even with
//! an explicit `n`, because `unwrap_or` evaluates its argument eagerly) and
//! feeding a wrapped allocation size in release builds for `n=None`. The same
//! expression sits in `hfft_norm`'s CUDA gate and is duplicated inside
//! ferray-fft (`ferray-fft-0.4.1/src/hermitian.rs:72`, `src/real.rs:139`), so
//! the wrapper must guard before delegating.
//!
//! Torch oracle (R-ORACLE-1, live torch 2.11.0+cu130, 2026-06-11):
//!
//! ```text
//! >>> torch.fft.irfft(torch.zeros(0, dtype=torch.complex64))
//! RuntimeError: Invalid number of data points (-2) specified
//! >>> torch.fft.irfft(torch.zeros(0, dtype=torch.complex64), n=8)
//! tensor([0., 0., 0., 0., 0., 0., 0., 0.])
//! >>> torch.fft.irfft(torch.zeros(0, dtype=torch.complex64), n=0)
//! RuntimeError: Invalid number of data points (0) specified
//! >>> torch.fft.irfft(torch.zeros(3, 0, dtype=torch.complex64), n=5)
//! tensor of zeros, shape [3, 5]
//! >>> torch.fft.irfft(torch.ones(1, dtype=torch.complex64))
//! RuntimeError: Invalid number of data points (0) specified
//! >>> torch.fft.hfft(torch.zeros(0, dtype=torch.complex64))
//! RuntimeError: Invalid number of data points (-2) specified
//! >>> torch.fft.hfft(torch.zeros(0, dtype=torch.complex64), n=8)
//! tensor([0., 0., 0., 0., 0., 0., 0., 0.])
//! >>> z = torch.zeros(3, 0, dtype=torch.complex64)
//! >>> torch.fft.irfftn(z)
//! RuntimeError: Invalid number of data points (0) specified
//! >>> torch.fft.irfftn(z, s=(3, 4)).shape
//! torch.Size([3, 4])
//! >>> torch.fft.irfft2(z, s=(4,))
//! RuntimeError: When given, dim and shape arguments must have the same length
//! ```
//!
//! Contract semantics: upstream `fft_c2r`
//! (`aten/src/ATen/native/SpectralOps.cpp:207-208`) computes
//! `n = n_opt.value_or(2*(input.sym_sizes()[dim] - 1))` in *signed* SymInt
//! arithmetic and rejects `n < 1` with
//! `TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified")`;
//! with an explicit valid `n` it zero-pads the empty spectrum
//! (`resize_fft_input`, `SpectralOps.cpp:209-211`) and the all-zero spectrum
//! inverts to all zeros.

use ferrotorch_core::fft::{hfft, hfft2, hfftn, irfft, irfft2, irfftn};
use ferrotorch_core::{FerrotorchError, Tensor, TensorStorage};

fn cpu_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
}

/// Assert a structured `InvalidArgument` whose message mirrors torch's
/// `Invalid number of data points (N) specified` (R-ORACLE-4: one contract).
fn assert_invalid_data_points<T: std::fmt::Debug>(r: Result<T, FerrotorchError>, op: &str, n: i64) {
    match r {
        Err(FerrotorchError::InvalidArgument { message }) => {
            let expected = format!("Invalid number of data points ({n}) specified");
            assert!(
                message.contains(&expected),
                "{op}: error message must mirror torch's {expected:?}, got {message:?}"
            );
        }
        other => panic!(
            "{op}: expected InvalidArgument mirroring torch's 'Invalid number of \
             data points ({n}) specified', got {other:?}"
        ),
    }
}

#[test]
fn irfft_empty_freq_axis_default_n_errors_like_torch() {
    // torch: RuntimeError: Invalid number of data points (-2) specified
    let empty = cpu_f64(vec![], vec![0, 2]);
    assert_invalid_data_points(irfft(&empty, None), "irfft", -2);
}

#[test]
fn irfft_empty_freq_axis_explicit_n_zero_pads_like_torch() {
    // torch: torch.fft.irfft(empty, n=8) == zeros(8)
    let empty = cpu_f64(vec![], vec![0, 2]);
    let out = irfft(&empty, Some(8)).expect("irfft with explicit n on empty input");
    assert_eq!(out.shape(), &[8]);
    assert_eq!(out.data().expect("cpu data"), &[0.0; 8]);
}

#[test]
fn irfft_empty_freq_axis_batched_explicit_n_zero_pads_like_torch() {
    // torch: torch.fft.irfft(torch.zeros(3, 0, dtype=torch.complex64), n=5)
    //        == zeros([3, 5])
    let empty = cpu_f64(vec![], vec![3, 0, 2]);
    let out = irfft(&empty, Some(5)).expect("irfft batched empty with explicit n");
    assert_eq!(out.shape(), &[3, 5]);
    assert_eq!(out.data().expect("cpu data"), &[0.0; 15]);
}

#[test]
fn irfft_empty_freq_axis_explicit_n_zero_errors_like_torch() {
    // torch: RuntimeError: Invalid number of data points (0) specified
    let empty = cpu_f64(vec![], vec![0, 2]);
    assert_invalid_data_points(irfft(&empty, Some(0)), "irfft", 0);
}

#[test]
fn irfft_singleton_freq_axis_default_n_errors_like_torch() {
    // half_n == 1 makes the default n = 2*(1-1) = 0;
    // torch: RuntimeError: Invalid number of data points (0) specified
    let one = cpu_f64(vec![1.0, 0.0], vec![1, 2]);
    assert_invalid_data_points(irfft(&one, None), "irfft", 0);
}

#[test]
fn hfft_empty_freq_axis_default_n_errors_like_torch() {
    // torch: RuntimeError: Invalid number of data points (-2) specified
    let empty = cpu_f64(vec![], vec![0, 2]);
    assert_invalid_data_points(hfft(&empty, None), "hfft", -2);
}

#[test]
fn hfft_empty_freq_axis_explicit_n_zero_pads_like_torch() {
    // torch: torch.fft.hfft(empty, n=8) == zeros(8)
    let empty = cpu_f64(vec![], vec![0, 2]);
    let out = hfft(&empty, Some(8)).expect("hfft with explicit n on empty input");
    assert_eq!(out.shape(), &[8]);
    assert_eq!(out.data().expect("cpu data"), &[0.0; 8]);
}

#[test]
fn hfft_singleton_freq_axis_default_n_errors_like_torch() {
    // torch: RuntimeError: Invalid number of data points (0) specified
    let one = cpu_f64(vec![1.0, 0.0], vec![1, 2]);
    assert_invalid_data_points(hfft(&one, None), "hfft", 0);
}

#[test]
fn nd_c2r_empty_last_axis_default_s_errors_like_torch() {
    // live torch 2.11.0+cu130:
    // torch.fft.{irfftn,irfft2,hfftn,hfft2}(zeros([3, 0], complex64))
    // -> RuntimeError: Invalid number of data points (0) specified
    let empty = cpu_f64(vec![], vec![3, 0, 2]);
    assert_invalid_data_points(irfftn(&empty, None, None), "irfftn", 0);
    assert_invalid_data_points(irfft2(&empty, None, None), "irfft2", 0);
    assert_invalid_data_points(hfftn(&empty, None, None), "hfftn", 0);
    assert_invalid_data_points(hfft2(&empty, None, None), "hfft2", 0);
}

#[test]
fn nd_c2r_singleton_last_axis_default_s_errors_like_torch() {
    // The N-D canonicalized transform shape is valid for [3, 1], then c2r
    // derives the real last dimension as 2 * (1 - 1) = 0 and rejects it.
    let one = cpu_f64(vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0], vec![3, 1, 2]);
    assert_invalid_data_points(irfftn(&one, None, None), "irfftn", 0);
    assert_invalid_data_points(irfft2(&one, None, None), "irfft2", 0);
    assert_invalid_data_points(hfftn(&one, None, None), "hfftn", 0);
    assert_invalid_data_points(hfft2(&one, None, None), "hfft2", 0);
}

fn assert_zero_output(out: Tensor<f64>, expected_shape: &[usize], op: &str) {
    assert_eq!(out.shape(), expected_shape, "{op}: output shape");
    assert!(
        out.data()
            .expect("cpu data")
            .iter()
            .all(|&value| value == 0.0),
        "{op}: explicit-size empty spectrum must produce all zeros"
    );
}

#[test]
fn nd_c2r_empty_last_axis_explicit_s_zero_pads_like_torch() {
    // live torch 2.11.0+cu130:
    // each op below on zeros([3, 0], complex64), s=(3, 4) returns zeros [3, 4].
    let empty = cpu_f64(vec![], vec![3, 0, 2]);
    let s = [3, 4];
    assert_zero_output(
        irfftn(&empty, Some(&s), None).expect("irfftn explicit s"),
        &[3, 4],
        "irfftn",
    );
    assert_zero_output(
        irfft2(&empty, Some(&s), None).expect("irfft2 explicit s"),
        &[3, 4],
        "irfft2",
    );
    assert_zero_output(
        hfftn(&empty, Some(&s), None).expect("hfftn explicit s"),
        &[3, 4],
        "hfftn",
    );
    assert_zero_output(
        hfft2(&empty, Some(&s), None).expect("hfft2 explicit s"),
        &[3, 4],
        "hfft2",
    );
}

#[test]
fn nd_c2r_empty_last_axis_explicit_one_axis_zero_pads_like_torch() {
    // torch.fft.irfftn(torch.zeros(2, 3, 0, complex64), s=(5,), dim=(-1,))
    // and hfftn with the same arguments return zeros [2, 3, 5].
    let empty = cpu_f64(vec![], vec![2, 3, 0, 2]);
    let s = [5];
    let axes = [-1];
    assert_zero_output(
        irfftn(&empty, Some(&s), Some(&axes)).expect("irfftn one-axis explicit s"),
        &[2, 3, 5],
        "irfftn one-axis",
    );
    assert_zero_output(
        hfftn(&empty, Some(&s), Some(&axes)).expect("hfftn one-axis explicit s"),
        &[2, 3, 5],
        "hfftn one-axis",
    );
}

#[test]
fn fft2_c2r_default_dim_requires_two_shape_values_like_torch() {
    // torch.fft.irfft2(z, s=(4,)) uses default dim=(-2,-1) and rejects the
    // length mismatch before it can treat s=(4,) as a one-axis transform.
    let empty = cpu_f64(vec![], vec![3, 0, 2]);
    let s = [4];
    for (op, result) in [
        ("irfft2", irfft2(&empty, Some(&s), None)),
        ("hfft2", hfft2(&empty, Some(&s), None)),
    ] {
        match result {
            Err(FerrotorchError::InvalidArgument { message }) => assert!(
                message.contains("When given, dim and shape arguments must have the same length"),
                "{op}: expected torch length-mismatch error, got {message:?}"
            ),
            other => panic!("{op}: expected length-mismatch InvalidArgument, got {other:?}"),
        }
    }
}
