//! CORE-132 (#1826): CPU kernels must consume logical tensor values from
//! non-contiguous views, matching PyTorch's arbitrary-stride operator surface.

use ferrotorch_core::ops::cumulative::{
    cummax_forward, cummin_forward, cumprod_forward, cumsum_forward, logcumsumexp_forward,
};
use ferrotorch_core::ops::elementwise::{
    fast_add, fast_cos, fast_div, fast_mul, fast_sigmoid, fast_sin, fast_sub, fast_tanh, logsumexp,
    logsumexp_dim, mean, nanmean, nansum, scalar_map, simd_add_f32, simd_exp_f32, simd_mul_f32,
    sum, sum_axis, unary_map,
};
use ferrotorch_core::{Tensor, TensorStorage};

fn cpu_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "index {i}: expected NaN, got {a}");
        } else {
            assert!((a - e).abs() <= tol, "index {i}: expected {e}, got {a}");
        }
    }
}

fn log_add_exp(x: f32, y: f32) -> f32 {
    let m = x.max(y);
    m + ((x - m).exp() + (y - m).exp()).ln()
}

#[test]
fn core132_binary_unary_and_scalar_cpu_paths_accept_transpose_views() {
    let base = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let view = base.transpose(0, 1).unwrap();
    assert_eq!(view.shape(), &[3, 2]);
    assert!(!view.is_contiguous());
    assert_eq!(view.data_vec().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    assert_eq!(
        simd_add_f32(&view, &view).unwrap().data_vec().unwrap(),
        vec![2.0, 8.0, 4.0, 10.0, 6.0, 12.0]
    );
    assert_eq!(
        simd_mul_f32(&view, &view).unwrap().data_vec().unwrap(),
        vec![1.0, 16.0, 4.0, 25.0, 9.0, 36.0]
    );
    assert_close(
        &simd_exp_f32(&view).unwrap().data_vec().unwrap(),
        &[
            1.0_f32.exp(),
            4.0_f32.exp(),
            2.0_f32.exp(),
            5.0_f32.exp(),
            3.0_f32.exp(),
            6.0_f32.exp(),
        ],
        1e-5,
    );

    assert_eq!(
        fast_add(&view, &view).unwrap().data_vec().unwrap(),
        vec![2.0, 8.0, 4.0, 10.0, 6.0, 12.0]
    );
    assert_eq!(
        fast_sub(&view, &view).unwrap().data_vec().unwrap(),
        vec![0.0; 6]
    );
    assert_eq!(
        fast_mul(&view, &view).unwrap().data_vec().unwrap(),
        vec![1.0, 16.0, 4.0, 25.0, 9.0, 36.0]
    );
    assert_eq!(
        fast_div(&view, &view).unwrap().data_vec().unwrap(),
        vec![1.0; 6]
    );

    let logical = view.data_vec().unwrap();
    let expected_sigmoid: Vec<f32> = logical.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    assert_close(
        &fast_sigmoid(&view).unwrap().data_vec().unwrap(),
        &expected_sigmoid,
        1e-6,
    );
    let expected_tanh: Vec<f32> = logical.iter().map(|&x| x.tanh()).collect();
    assert_close(
        &fast_tanh(&view).unwrap().data_vec().unwrap(),
        &expected_tanh,
        1e-6,
    );
    let expected_sin: Vec<f32> = logical.iter().map(|&x| x.sin()).collect();
    assert_close(
        &fast_sin(&view).unwrap().data_vec().unwrap(),
        &expected_sin,
        1e-5,
    );
    let expected_cos: Vec<f32> = logical.iter().map(|&x| x.cos()).collect();
    assert_close(
        &fast_cos(&view).unwrap().data_vec().unwrap(),
        &expected_cos,
        1e-5,
    );

    assert_eq!(
        unary_map(&view, |x| x.sqrt()).unwrap().data_vec().unwrap(),
        logical.iter().map(|&x| x.sqrt()).collect::<Vec<_>>()
    );
    assert_eq!(
        scalar_map(&view, 10.0, |x, y| x + y)
            .unwrap()
            .data_vec()
            .unwrap(),
        vec![11.0, 14.0, 12.0, 15.0, 13.0, 16.0]
    );
}

#[test]
fn core132_reductions_accept_transpose_views_in_logical_order() {
    let base = cpu_f32(&[1.0, f32::NAN, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let view = base.transpose(0, 1).unwrap();
    assert!(!view.is_contiguous());

    assert!(sum(&view).unwrap().data_vec().unwrap()[0].is_nan());
    assert!(mean(&view).unwrap().data_vec().unwrap()[0].is_nan());
    assert_eq!(nansum(&view).unwrap().data_vec().unwrap(), vec![19.0]);
    assert_eq!(nanmean(&view).unwrap().data_vec().unwrap(), vec![3.8]);

    let finite = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .transpose(0, 1)
        .unwrap();
    assert_eq!(sum(&finite).unwrap().data_vec().unwrap(), vec![21.0]);
    assert_eq!(mean(&finite).unwrap().data_vec().unwrap(), vec![3.5]);
    assert_eq!(
        sum_axis(&finite, 0).unwrap().data_vec().unwrap(),
        vec![6.0, 15.0]
    );
    assert_eq!(
        sum_axis(&finite, 1).unwrap().data_vec().unwrap(),
        vec![5.0, 7.0, 9.0]
    );

    let logical = finite.data_vec().unwrap();
    let expected_global = logical.iter().map(|&x| x.exp()).sum::<f32>().ln();
    assert_close(
        &logsumexp(&finite).unwrap().data_vec().unwrap(),
        &[expected_global],
        1e-5,
    );
    let expected_dim1 = vec![
        log_add_exp(1.0, 4.0),
        log_add_exp(2.0, 5.0),
        log_add_exp(3.0, 6.0),
    ];
    assert_close(
        &logsumexp_dim(&finite, 1, false)
            .unwrap()
            .data_vec()
            .unwrap(),
        &expected_dim1,
        1e-5,
    );
}

#[test]
fn core132_cumulative_cpu_paths_accept_transpose_views_in_logical_order() {
    let view = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .transpose(0, 1)
        .unwrap();
    assert!(!view.is_contiguous());

    assert_eq!(
        cumsum_forward(&view, 0).unwrap().data_vec().unwrap(),
        vec![1.0, 4.0, 3.0, 9.0, 6.0, 15.0]
    );
    assert_eq!(
        cumsum_forward(&view, 1).unwrap().data_vec().unwrap(),
        vec![1.0, 5.0, 2.0, 7.0, 3.0, 9.0]
    );
    assert_eq!(
        cumprod_forward(&view, 0).unwrap().data_vec().unwrap(),
        vec![1.0, 4.0, 2.0, 20.0, 6.0, 120.0]
    );

    let max = cummax_forward(&view, 1).unwrap();
    assert_eq!(
        max.values.data_vec().unwrap(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
    assert_eq!(max.indices, vec![0, 1, 0, 1, 0, 1]);

    let min = cummin_forward(&view, 1).unwrap();
    assert_eq!(
        min.values.data_vec().unwrap(),
        vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    );
    assert_eq!(min.indices, vec![0, 0, 0, 0, 0, 0]);

    let expected_lce_dim1 = vec![
        1.0,
        log_add_exp(4.0, 1.0),
        2.0,
        log_add_exp(5.0, 2.0),
        3.0,
        log_add_exp(6.0, 3.0),
    ];
    assert_close(
        &logcumsumexp_forward(&view, 1).unwrap().data_vec().unwrap(),
        &expected_lce_dim1,
        1e-5,
    );
}
