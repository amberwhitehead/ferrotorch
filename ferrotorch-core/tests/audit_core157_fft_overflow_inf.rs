//! Regression tests for CORE-157 / crosslink #1851 (CLASS-V).
//!
//! The CPU FFT path computes in f64 and cast back through
//! `numeric_cast::cast`, whose saturation guard (#815) returns `Err` when a
//! finite f64 overflows f32. Any spectrum bin exceeding `f32::MAX` turned the
//! whole transform into `InvalidArgument`, where PyTorch computes natively in
//! f32 and returns `inf` bins. The array->tensor bridges must use direct
//! `as`-cast float semantics (finite overflow saturates to ±inf).
//!
//! Torch oracle (R-ORACLE-1, live torch 2.11.0+cu130, 2026-06-11):
//!
//! ```text
//! >>> torch.fft.rfft(torch.tensor([3e38, 3e38]))
//! tensor([inf+0.j, 0.+0.j])      # interleaved [inf, 0.0, 0.0, 0.0]
//! >>> torch.fft.rfft(torch.tensor([-3e38, -3e38]))
//! tensor([-inf+0.j, 0.+0.j])     # interleaved [-inf, 0.0, 0.0, 0.0]
//! >>> z = torch.complex(torch.tensor([3e38, 3e38]), torch.tensor([-3e38, -3e38]))
//! >>> torch.fft.fft(z)
//! # interleaved [inf, -inf, 0.0, 0.0]
//! >>> zz = torch.complex(torch.tensor([3e38, 3e38]), torch.zeros(2))
//! >>> torch.fft.hfft(zz, n=2)
//! tensor([inf, 0.])
//! ```
//!
//! All four pinned bins are exact in f64 (sums of two ±3e38 terms), so the
//! only rounding step is the final f64→f32 cast; comparisons are exact
//! (`==`, with `inf == inf`).
//!
//! Known residual divergence, NOT this mechanism: ferrotorch's CPU FFT
//! computes in f64 (documented module contract), so bins where torch's
//! native-f32 butterfly overflows an *intermediate* (e.g.
//! `torch.fft.rfft(torch.tensor([3e38]*4))[2]` is `nan` in torch from
//! `inf - inf`) come out finite here. Only finitely-representable-in-f64
//! oracles are pinned below.

use ferrotorch_core::fft::{fft, hfft, rfft};
use ferrotorch_core::{Tensor, TensorStorage};

fn cpu_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
}

#[test]
fn rfft_f32_overflow_bin_saturates_to_inf_like_torch() {
    let x = cpu_f32(vec![3e38, 3e38], vec![2]);
    let r = rfft(&x, None).expect("rfft must compute, not error, on f32 overflow");
    assert_eq!(r.shape(), &[2, 2]);
    assert_eq!(
        r.data().expect("cpu data"),
        &[f32::INFINITY, 0.0, 0.0, 0.0],
        "torch.fft.rfft([3e38, 3e38]) == [inf+0j, 0+0j]"
    );
}

#[test]
fn rfft_f32_negative_overflow_bin_saturates_to_neg_inf_like_torch() {
    let x = cpu_f32(vec![-3e38, -3e38], vec![2]);
    let r = rfft(&x, None).expect("rfft must compute on negative f32 overflow");
    assert_eq!(
        r.data().expect("cpu data"),
        &[f32::NEG_INFINITY, 0.0, 0.0, 0.0],
        "torch.fft.rfft([-3e38, -3e38]) == [-inf+0j, 0+0j]"
    );
}

#[test]
fn fft_f32_complex_overflow_bins_saturate_like_torch() {
    // z = [3e38 - 3e38j, 3e38 - 3e38j], interleaved [2, 2].
    let z = cpu_f32(vec![3e38, -3e38, 3e38, -3e38], vec![2, 2]);
    let r = fft(&z, None).expect("fft must compute on complex f32 overflow");
    assert_eq!(
        r.data().expect("cpu data"),
        &[f32::INFINITY, f32::NEG_INFINITY, 0.0, 0.0],
        "torch.fft.fft([3e38-3e38j, 3e38-3e38j]) == [inf-infj, 0+0j]"
    );
}

#[test]
fn hfft_f32_overflow_bin_saturates_to_inf_like_torch() {
    // Exercises the REAL array->tensor bridge (hfft output is real).
    let z = cpu_f32(vec![3e38, 0.0, 3e38, 0.0], vec![2, 2]);
    let r = hfft(&z, Some(2)).expect("hfft must compute on f32 overflow");
    assert_eq!(r.shape(), &[2]);
    assert_eq!(
        r.data().expect("cpu data"),
        &[f32::INFINITY, 0.0],
        "torch.fft.hfft([3e38+0j, 3e38+0j], n=2) == [inf, 0]"
    );
}
