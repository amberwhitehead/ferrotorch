//! Regression tests for CORE-160 / #1854.
//!
//! PyTorch implements `torch.fft.fftfreq` and `torch.fft.rfftfreq` by
//! materializing the integer frequency bins and multiplying by
//! `1.0 / (n * d)`. It does not reject `n == 0` or `d == 0`; the IEEE
//! `NaN`/`±Inf` values from that multiply are the public contract.

use ferrotorch_core::fft::{fftfreq, rfftfreq};

fn assert_nan(value: f64, label: &str) {
    assert!(value.is_nan(), "{label}: expected NaN, got {value:?}");
}

fn assert_pos_inf(value: f64, label: &str) {
    assert!(
        value.is_infinite() && value.is_sign_positive(),
        "{label}: expected +inf, got {value:?}"
    );
}

fn assert_neg_inf(value: f64, label: &str) {
    assert!(
        value.is_infinite() && value.is_sign_negative(),
        "{label}: expected -inf, got {value:?}"
    );
}

fn assert_neg_zero(value: f64, label: &str) {
    assert_eq!(value, 0.0, "{label}: expected zero, got {value:?}");
    assert!(
        value.is_sign_negative(),
        "{label}: expected negative zero, got {value:?}"
    );
}

#[test]
fn fftfreq_zero_length_is_empty_for_any_spacing() {
    for d in [1.0, 0.0, -0.0, -2.0, f64::INFINITY, f64::NAN] {
        let out = fftfreq(0, d).expect("fftfreq n=0");
        assert_eq!(out.shape(), &[0]);
        assert!(out.data().unwrap().is_empty());
    }
}

#[test]
fn rfftfreq_zero_length_is_single_nan() {
    for d in [1.0, 0.0, -0.0, -2.0, f64::INFINITY, f64::NAN] {
        let out = rfftfreq(0, d).expect("rfftfreq n=0");
        assert_eq!(out.shape(), &[1]);
        let data = out.data().unwrap();
        assert_eq!(data.len(), 1);
        assert_nan(data[0], "rfftfreq(0, d)[0]");
    }
}

#[test]
fn fftfreq_zero_spacing_matches_torch_ieee_payloads() {
    let out = fftfreq(5, 0.0).expect("fftfreq d=0");
    let data = out.data().unwrap();
    assert_eq!(data.len(), 5);
    assert_nan(data[0], "bin 0");
    assert_pos_inf(data[1], "bin 1");
    assert_pos_inf(data[2], "bin 2");
    assert_neg_inf(data[3], "bin 3");
    assert_neg_inf(data[4], "bin 4");
}

#[test]
fn rfftfreq_zero_spacing_matches_torch_ieee_payloads() {
    let out = rfftfreq(4, 0.0).expect("rfftfreq d=0");
    let data = out.data().unwrap();
    assert_eq!(data.len(), 3);
    assert_nan(data[0], "bin 0");
    assert_pos_inf(data[1], "bin 1");
    assert_pos_inf(data[2], "bin 2");
}

#[test]
fn negative_zero_spacing_preserves_infinity_signs() {
    let full = fftfreq(4, -0.0).expect("fftfreq d=-0");
    let full = full.data().unwrap();
    assert_nan(full[0], "fftfreq bin 0");
    assert_neg_inf(full[1], "fftfreq bin 1");
    assert_pos_inf(full[2], "fftfreq bin 2");
    assert_pos_inf(full[3], "fftfreq bin 3");

    let real = rfftfreq(4, -0.0).expect("rfftfreq d=-0");
    let real = real.data().unwrap();
    assert_nan(real[0], "rfftfreq bin 0");
    assert_neg_inf(real[1], "rfftfreq bin 1");
    assert_neg_inf(real[2], "rfftfreq bin 2");
}

#[test]
fn negative_spacing_scales_bins_like_torch() {
    let full = fftfreq(5, -2.0).expect("fftfreq d=-2");
    let full = full.data().unwrap();
    assert_neg_zero(full[0], "fftfreq bin 0");
    assert_eq!(full[1..], [-0.1, -0.2, 0.2, 0.1]);

    let real = rfftfreq(5, -2.0).expect("rfftfreq d=-2");
    let real = real.data().unwrap();
    assert_neg_zero(real[0], "rfftfreq bin 0");
    assert_eq!(real[1..], [-0.1, -0.2]);
}
