//! Regression tests for CORE-156 / crosslink #1850 (CLASS-V).
//!
//! `fftshift`/`ifftshift` delegated the FULL tensor shape (including the
//! trailing interleaved complex pair axis) to ferray's all-axes roll, so
//! `axes=None` rolled the re/im axis by one — swapping real and imaginary
//! parts — and negative axes resolved against the wrong layout. The canonical
//! `fftshift(fft(x))` silently returned corrupted spectra with the right
//! shape.
//!
//! Contract (per upstream `fft_fftshift` / `fft_ifftshift`,
//! `aten/src/ATen/native/SpectralOps.cpp:767-789`): the shift rolls the
//! tensor's DIMS — for a complex tensor the re/im pair is dtype payload, not
//! a dim, and is never rolled. In ferrotorch's interleaved `[..., 2]`
//! convention an input with `ndim >= 2` and trailing dim 2 is
//! complex-encoded: axes resolve against the signal layout (trailing 2
//! stripped) and `axes=None` shifts every SIGNAL axis only.
//!
//! Torch oracle (R-ORACLE-1, live torch 2.11.0+cu130, 2026-06-11), values as
//! `torch.view_as_real(...).flatten().tolist()`:
//!
//! ```text
//! >>> z = torch.fft.fft(torch.tensor([1.0, 2.0, 3.0, 4.0]))
//! >>> torch.view_as_real(z).flatten().tolist()
//! [10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0]
//! >>> torch.view_as_real(torch.fft.fftshift(z)).flatten().tolist()
//! [-2.0, 0.0, -2.0, -2.0, 10.0, 0.0, -2.0, 2.0]
//! >>> torch.view_as_real(torch.fft.ifftshift(z)).flatten().tolist()
//! [-2.0, 0.0, -2.0, -2.0, 10.0, 0.0, -2.0, 2.0]
//! >>> torch.view_as_real(torch.fft.fftshift(z, dim=-1)).flatten().tolist()
//! [-2.0, 0.0, -2.0, -2.0, 10.0, 0.0, -2.0, 2.0]
//!
//! >>> z5 = torch.fft.fft(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
//! >>> torch.view_as_real(z5).flatten().tolist()
//! [15.0, 0.0, -2.5, 3.4409549236297607, -2.499999761581421,
//!  0.8122991919517517, -2.499999761581421, -0.8122991919517517,
//!  -2.5, -3.4409549236297607]
//! >>> torch.view_as_real(torch.fft.fftshift(z5)).flatten().tolist()
//! [-2.499999761581421, -0.8122991919517517, -2.5, -3.4409549236297607,
//!  15.0, 0.0, -2.5, 3.4409549236297607, -2.499999761581421,
//!  0.8122991919517517]
//! >>> torch.view_as_real(torch.fft.ifftshift(z5)).flatten().tolist()
//! [-2.499999761581421, 0.8122991919517517, -2.499999761581421,
//!  -0.8122991919517517, -2.5, -3.4409549236297607, 15.0, 0.0, -2.5,
//!  3.4409549236297607]
//!
//! >>> z2 = torch.fft.fft2(torch.arange(6, dtype=torch.float64).reshape(2, 3))
//! >>> torch.view_as_real(z2).flatten().tolist()
//! [15.0, 0.0, -3.0, 1.7320508075688772, -3.0, -1.7320508075688772,
//!  -9.0, 0.0, 0.0, 0.0, 0.0, 0.0]
//! >>> torch.view_as_real(torch.fft.fftshift(z2)).flatten().tolist()
//! [0.0, 0.0, -9.0, 0.0, 0.0, 0.0, -3.0, -1.7320508075688772, 15.0, 0.0,
//!  -3.0, 1.7320508075688772]
//! >>> torch.view_as_real(torch.fft.fftshift(z2, dim=0)).flatten().tolist()
//! [-9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 0.0, -3.0, 1.7320508075688772,
//!  -3.0, -1.7320508075688772]
//! >>> torch.view_as_real(torch.fft.ifftshift(z2)).flatten().tolist()
//! [0.0, 0.0, 0.0, 0.0, -9.0, 0.0, -3.0, 1.7320508075688772, -3.0,
//!  -1.7320508075688772, 15.0, 0.0]
//! ```
//!
//! The shifts are pure data movement, so every comparison below is
//! bit-exact (`assert_eq!` on the f64 buffers).

use ferrotorch_core::fft::{fftshift, ifftshift};
use ferrotorch_core::{Tensor, TensorStorage};

fn cpu_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
}

/// fft([1, 2, 3, 4]) as interleaved `[4, 2]` (torch oracle above).
fn z_even4() -> Tensor<f64> {
    cpu_f64(
        vec![10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0],
        vec![4, 2],
    )
}

/// fft([1, 2, 3, 4, 5]) as interleaved `[5, 2]` (torch f32 oracle above).
fn z_odd5() -> Tensor<f64> {
    cpu_f64(
        vec![
            15.0,
            0.0,
            -2.5,
            3.440_954_923_629_760_7,
            -2.499_999_761_581_421,
            0.812_299_191_951_751_7,
            -2.499_999_761_581_421,
            -0.812_299_191_951_751_7,
            -2.5,
            -3.440_954_923_629_760_7,
        ],
        vec![5, 2],
    )
}

/// fft2(arange(6).reshape(2, 3)) as interleaved `[2, 3, 2]` (torch oracle).
fn z_2d() -> Tensor<f64> {
    cpu_f64(
        vec![
            15.0,
            0.0,
            -3.0,
            1.732_050_807_568_877_2,
            -3.0,
            -1.732_050_807_568_877_2,
            -9.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        vec![2, 3, 2],
    )
}

#[test]
fn fftshift_complex_even_axes_none_matches_torch() {
    let s = fftshift(&z_even4(), None).expect("fftshift");
    assert_eq!(s.shape(), &[4, 2]);
    assert_eq!(
        s.data().expect("cpu data"),
        &[-2.0, 0.0, -2.0, -2.0, 10.0, 0.0, -2.0, 2.0],
        "fftshift(fft([1,2,3,4])) must match torch (re/im pair axis is \
         metadata, never shifted)"
    );
}

#[test]
fn ifftshift_complex_even_axes_none_matches_torch() {
    let s = ifftshift(&z_even4(), None).expect("ifftshift");
    assert_eq!(
        s.data().expect("cpu data"),
        &[-2.0, 0.0, -2.0, -2.0, 10.0, 0.0, -2.0, 2.0],
        "ifftshift on an even-length complex spectrum must match torch"
    );
}

#[test]
fn fftshift_complex_negative_axis_resolves_against_signal_layout() {
    // torch.fft.fftshift(z, dim=-1): -1 is the LAST SIGNAL axis (length 4),
    // never the interleaved pair axis.
    let s = fftshift(&z_even4(), Some(&[-1])).expect("fftshift dim=-1");
    assert_eq!(
        s.data().expect("cpu data"),
        &[-2.0, 0.0, -2.0, -2.0, 10.0, 0.0, -2.0, 2.0],
        "fftshift(z, dim=-1) must resolve -1 against the signal layout"
    );
}

#[test]
fn fftshift_complex_odd_axes_none_matches_torch() {
    let s = fftshift(&z_odd5(), None).expect("fftshift odd");
    assert_eq!(
        s.data().expect("cpu data"),
        &[
            -2.499_999_761_581_421,
            -0.812_299_191_951_751_7,
            -2.5,
            -3.440_954_923_629_760_7,
            15.0,
            0.0,
            -2.5,
            3.440_954_923_629_760_7,
            -2.499_999_761_581_421,
            0.812_299_191_951_751_7,
        ]
    );
}

#[test]
fn ifftshift_complex_odd_axes_none_matches_torch() {
    let s = ifftshift(&z_odd5(), None).expect("ifftshift odd");
    assert_eq!(
        s.data().expect("cpu data"),
        &[
            -2.499_999_761_581_421,
            0.812_299_191_951_751_7,
            -2.499_999_761_581_421,
            -0.812_299_191_951_751_7,
            -2.5,
            -3.440_954_923_629_760_7,
            15.0,
            0.0,
            -2.5,
            3.440_954_923_629_760_7,
        ]
    );
}

#[test]
fn fftshift_complex_2d_axes_none_shifts_both_signal_axes() {
    let s = fftshift(&z_2d(), None).expect("fftshift 2d");
    assert_eq!(s.shape(), &[2, 3, 2]);
    assert_eq!(
        s.data().expect("cpu data"),
        &[
            0.0,
            0.0,
            -9.0,
            0.0,
            0.0,
            0.0,
            -3.0,
            -1.732_050_807_568_877_2,
            15.0,
            0.0,
            -3.0,
            1.732_050_807_568_877_2,
        ]
    );
}

#[test]
fn fftshift_complex_2d_explicit_axis0_matches_torch() {
    let s = fftshift(&z_2d(), Some(&[0])).expect("fftshift dim=0");
    assert_eq!(
        s.data().expect("cpu data"),
        &[
            -9.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            15.0,
            0.0,
            -3.0,
            1.732_050_807_568_877_2,
            -3.0,
            -1.732_050_807_568_877_2,
        ]
    );
}

#[test]
fn ifftshift_complex_2d_axes_none_matches_torch() {
    let s = ifftshift(&z_2d(), None).expect("ifftshift 2d");
    assert_eq!(
        s.data().expect("cpu data"),
        &[
            0.0,
            0.0,
            0.0,
            0.0,
            -9.0,
            0.0,
            -3.0,
            1.732_050_807_568_877_2,
            -3.0,
            -1.732_050_807_568_877_2,
            15.0,
            0.0,
        ]
    );
}

#[test]
fn fftshift_real_1d_keeps_all_axes_semantics() {
    // Real tensors without a trailing pair axis keep torch's all-dims
    // semantics. Oracle (live torch 2.11.0+cu130):
    //   >>> torch.fft.fftshift(torch.fft.fftfreq(4)).tolist()
    //   [-0.5, -0.25, 0.0, 0.25]
    // (fftfreq(4) == [0.0, 0.25, -0.5, -0.25])
    let f = cpu_f64(vec![0.0, 0.25, -0.5, -0.25], vec![4]);
    let s = fftshift(&f, None).expect("fftshift real 1d");
    assert_eq!(s.data().expect("cpu data"), &[-0.5, -0.25, 0.0, 0.25]);
}
