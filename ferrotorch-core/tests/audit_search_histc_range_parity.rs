use ferrotorch_core::ops::search::histc;
use ferrotorch_core::{FerrotorchError, Tensor, TensorStorage};

fn tensor(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("tensor")
}

#[test]
fn histc_default_range_nan_errors_like_torch() {
    let x = tensor(&[f32::NAN, 1.0, 2.0]);

    let err = histc(&x, 4, 0.0, 0.0).expect_err("NaN inferred range must error");

    assert!(
        matches!(err, FerrotorchError::InvalidArgument { .. }),
        "expected InvalidArgument for NaN inferred range, got {err:?}"
    );
    assert!(
        err.to_string().contains("not finite"),
        "error should name the non-finite inferred range, got {err}"
    );
}

#[test]
fn histc_default_range_infinite_errors_like_torch() {
    let x = tensor(&[f32::INFINITY, 1.0, 2.0]);

    let err = histc(&x, 4, 0.0, 0.0).expect_err("infinite inferred range must error");

    assert!(
        matches!(err, FerrotorchError::InvalidArgument { .. }),
        "expected InvalidArgument for infinite inferred range, got {err:?}"
    );
    assert!(
        err.to_string().contains("not finite"),
        "error should name the non-finite inferred range, got {err}"
    );
}

#[test]
fn histc_empty_equal_finite_range_returns_zero_histogram_like_torch() {
    let x = tensor(&[]);

    let out = histc(&x, 4, 0.0, 0.0).expect("empty finite equal range");

    assert_eq!(out.data().expect("hist data"), &[0.0, 0.0, 0.0, 0.0]);
}
