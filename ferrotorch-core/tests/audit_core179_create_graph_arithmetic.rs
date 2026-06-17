use ferrotorch_core::grad_fns::arithmetic::{
    abs, add_scaled, addcdiv, addcmul, fmod, mul, neg, reciprocal, remainder, rsqrt, sqrt,
};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::{Device, Tensor, TensorStorage, grad};

fn cpu_f64(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: actual={actual:?} expected={expected:?}"
    );
    for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "idx {idx}: actual {a:?} != expected {e:?} within {tol}; actual={actual:?} expected={expected:?}"
        );
    }
}

fn only_grad<T: ferrotorch_core::Float>(grads: Vec<Option<Tensor<T>>>, index: usize) -> Tensor<T> {
    grads
        .into_iter()
        .nth(index)
        .unwrap_or_else(|| panic!("missing grad slot {index}"))
        .unwrap_or_else(|| panic!("gradient slot {index} was None"))
}

#[test]
fn broadcast_reduction_create_graph_sums_like_pytorch() {
    let x = cpu_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let b = cpu_f64(&[0.5, -0.25, 2.0], &[3], true);

    let y = sum(&mul(&x, &b).unwrap()).unwrap();
    let gb = only_grad(grad(&y, &[&b], true, true).unwrap(), 0);
    assert_close(&gb.data_vec().unwrap(), &[5.0, 7.0, 9.0], 1e-12);
    assert!(
        gb.requires_grad(),
        "broadcast-reduced first gradient must preserve its dependence on x"
    );

    let gb_sum = sum(&gb).unwrap();
    let gx2 = only_grad(grad(&gb_sum, &[&x], false, false).unwrap(), 0);
    assert_close(
        &gx2.data_vec().unwrap(),
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        1e-12,
    );
}

#[test]
fn add_scaled_and_neg_create_graph_preserve_upstream_gradient() {
    let x = cpu_f64(&[1.0, 2.0, 3.0], &[3], true);
    let b = cpu_f64(&[0.4, 0.5, 0.6], &[3], true);
    let zero = cpu_f64(&[0.0, 0.0, 0.0], &[3], false);

    let scaled = add_scaled(&zero, &b, -2.0).unwrap();
    let y = sum(&mul(&x, &scaled).unwrap()).unwrap();
    let gb = only_grad(grad(&y, &[&b], true, true).unwrap(), 0);
    assert_close(&gb.data_vec().unwrap(), &[-2.0, -4.0, -6.0], 1e-12);
    assert!(
        gb.requires_grad(),
        "sub/add_scaled gradient should depend on x"
    );

    let gb_sum = sum(&gb).unwrap();
    let gx2 = only_grad(grad(&gb_sum, &[&x], false, false).unwrap(), 0);
    assert_close(&gx2.data_vec().unwrap(), &[-2.0, -2.0, -2.0], 1e-12);

    let neg_b = neg(&b).unwrap();
    let z = sum(&mul(&x, &neg_b).unwrap()).unwrap();
    let gb_neg = only_grad(grad(&z, &[&b], true, true).unwrap(), 0);
    assert_close(&gb_neg.data_vec().unwrap(), &[-1.0, -2.0, -3.0], 1e-12);
    assert!(gb_neg.requires_grad(), "neg gradient should depend on x");

    let gb_neg_sum = sum(&gb_neg).unwrap();
    let gx2_neg = only_grad(grad(&gb_neg_sum, &[&x], false, false).unwrap(), 0);
    assert_close(&gx2_neg.data_vec().unwrap(), &[-1.0, -1.0, -1.0], 1e-12);
}

#[test]
fn saved_output_unary_create_graph_matches_pytorch_second_derivatives() {
    let input = [0.5, 2.0, 5.0];

    let x = cpu_f64(&input, &[3], true);
    let gx = only_grad(
        grad(&sum(&sqrt(&x).unwrap()).unwrap(), &[&x], true, true).unwrap(),
        0,
    );
    assert!(
        gx.requires_grad(),
        "sqrt first gradient must be differentiable"
    );
    let g2 = only_grad(grad(&sum(&gx).unwrap(), &[&x], false, false).unwrap(), 0);
    assert_close(
        &g2.data_vec().unwrap(),
        &[
            -0.7071067811865474,
            -0.08838834764831842,
            -0.022360679774997894,
        ],
        1e-12,
    );

    let x = cpu_f64(&input, &[3], true);
    let gx = only_grad(
        grad(&sum(&rsqrt(&x).unwrap()).unwrap(), &[&x], true, true).unwrap(),
        0,
    );
    assert!(
        gx.requires_grad(),
        "rsqrt first gradient must be differentiable"
    );
    let g2 = only_grad(grad(&sum(&gx).unwrap(), &[&x], false, false).unwrap(), 0);
    assert_close(
        &g2.data_vec().unwrap(),
        &[4.242640687119283, 0.1325825214724776, 0.013416407864998736],
        1e-12,
    );

    let x = cpu_f64(&input, &[3], true);
    let gx = only_grad(
        grad(&sum(&reciprocal(&x).unwrap()).unwrap(), &[&x], true, true).unwrap(),
        0,
    );
    assert!(
        gx.requires_grad(),
        "reciprocal first gradient must be differentiable"
    );
    let g2 = only_grad(grad(&sum(&gx).unwrap(), &[&x], false, false).unwrap(), 0);
    assert_close(&g2.data_vec().unwrap(), &[16.0, 0.25, 0.016], 1e-12);
}

#[test]
fn abs_create_graph_matches_pytorch_zero_second_derivative() {
    let x = cpu_f64(&[-2.0, 0.0, 3.0], &[3], true);

    let gx = only_grad(
        grad(&sum(&abs(&x).unwrap()).unwrap(), &[&x], true, true).unwrap(),
        0,
    );
    assert_close(&gx.data_vec().unwrap(), &[-1.0, 0.0, 1.0], 1e-12);
    assert!(
        gx.requires_grad(),
        "abs first gradient should be SignBackward-backed"
    );

    let g2 = only_grad(grad(&sum(&gx).unwrap(), &[&x], false, false).unwrap(), 0);
    assert_close(&g2.data_vec().unwrap(), &[0.0, 0.0, 0.0], 1e-12);
}

#[test]
fn remainder_and_fmod_divisor_create_graph_match_pytorch_zero_second_derivatives() {
    let a = cpu_f64(&[5.5, -5.5, 7.0], &[3], true);
    let b = cpu_f64(&[2.0, 2.0, 3.0], &[3], true);
    let grads = grad(
        &sum(&remainder(&a, &b).unwrap()).unwrap(),
        &[&a, &b],
        true,
        true,
    )
    .unwrap();
    let ga = grads[0].as_ref().unwrap();
    let gb = grads[1].as_ref().unwrap();
    assert_close(&ga.data_vec().unwrap(), &[1.0, 1.0, 1.0], 1e-12);
    assert_close(&gb.data_vec().unwrap(), &[-2.0, 3.0, -2.0], 1e-12);
    assert!(
        gb.requires_grad(),
        "remainder divisor gradient should be backed by rounded-div zero VJP"
    );

    let gb_sum = sum(gb).unwrap();
    let g2 = grad(&gb_sum, &[&a, &b], false, false).unwrap();
    assert_close(
        &g2[0].as_ref().unwrap().data_vec().unwrap(),
        &[0.0, 0.0, 0.0],
        1e-12,
    );
    assert_close(
        &g2[1].as_ref().unwrap().data_vec().unwrap(),
        &[0.0, 0.0, 0.0],
        1e-12,
    );

    let a = cpu_f64(&[5.5, -5.5, 7.0], &[3], true);
    let b = cpu_f64(&[2.0, 2.0, 3.0], &[3], true);
    let grads = grad(&sum(&fmod(&a, &b).unwrap()).unwrap(), &[&a, &b], true, true).unwrap();
    let ga = grads[0].as_ref().unwrap();
    let gb = grads[1].as_ref().unwrap();
    assert_close(&ga.data_vec().unwrap(), &[1.0, 1.0, 1.0], 1e-12);
    assert_close(&gb.data_vec().unwrap(), &[-2.0, 2.0, -2.0], 1e-12);
    assert!(
        gb.requires_grad(),
        "fmod divisor gradient should be backed by rounded-div zero VJP"
    );

    let gb_sum = sum(gb).unwrap();
    let g2 = grad(&gb_sum, &[&a, &b], false, false).unwrap();
    assert_close(
        &g2[0].as_ref().unwrap().data_vec().unwrap(),
        &[0.0, 0.0, 0.0],
        1e-12,
    );
    assert_close(
        &g2[1].as_ref().unwrap().data_vec().unwrap(),
        &[0.0, 0.0, 0.0],
        1e-12,
    );
}

#[test]
fn addcmul_and_addcdiv_create_graph_mixed_partials_match_pytorch() {
    let input = cpu_f64(&[0.1, 0.2, 0.3], &[3], true);
    let t1 = cpu_f64(&[1.0, 2.0, 3.0], &[3], true);
    let t2 = cpu_f64(&[4.0, 5.0, 6.0], &[3], true);
    let value = 1.7;

    let g_t1 = only_grad(
        grad(
            &sum(&addcmul(&input, &t1, &t2, value).unwrap()).unwrap(),
            &[&t1],
            true,
            true,
        )
        .unwrap(),
        0,
    );
    assert_close(&g_t1.data_vec().unwrap(), &[6.8, 8.5, 10.2], 1e-12);
    assert!(
        g_t1.requires_grad(),
        "addcmul tensor1 gradient should depend on tensor2"
    );
    let mixed = only_grad(grad(&sum(&g_t1).unwrap(), &[&t2], false, false).unwrap(), 0);
    assert_close(&mixed.data_vec().unwrap(), &[value, value, value], 1e-12);

    let g_t1 = only_grad(
        grad(
            &sum(&addcdiv(&input, &t1, &t2, value).unwrap()).unwrap(),
            &[&t1],
            true,
            true,
        )
        .unwrap(),
        0,
    );
    assert_close(
        &g_t1.data_vec().unwrap(),
        &[0.425, 0.34, 0.2833333333333333],
        1e-12,
    );
    assert!(
        g_t1.requires_grad(),
        "addcdiv tensor1 gradient should depend on tensor2"
    );
    let mixed = only_grad(grad(&sum(&g_t1).unwrap(), &[&t2], false, false).unwrap(), 0);
    assert_close(
        &mixed.data_vec().unwrap(),
        &[-0.10625, -0.068, -0.04722222222222222],
        1e-12,
    );

    let g_t2 = only_grad(
        grad(
            &sum(&addcdiv(&input, &t1, &t2, value).unwrap()).unwrap(),
            &[&t2],
            true,
            true,
        )
        .unwrap(),
        0,
    );
    assert_close(
        &g_t2.data_vec().unwrap(),
        &[-0.10625, -0.136, -0.14166666666666666],
        1e-12,
    );
    assert!(
        g_t2.requires_grad(),
        "addcdiv tensor2 gradient should be differentiable"
    );
    let second = grad(&sum(&g_t2).unwrap(), &[&t1, &t2], false, false).unwrap();
    assert_close(
        &second[0].as_ref().unwrap().data_vec().unwrap(),
        &[-0.10625, -0.068, -0.04722222222222222],
        1e-12,
    );
    assert_close(
        &second[1].as_ref().unwrap().data_vec().unwrap(),
        &[0.053125, 0.0544, 0.04722222222222222],
        1e-12,
    );
}

#[test]
fn cuda_abs_create_graph_stays_resident_and_has_zero_second_derivative() {
    ferrotorch_gpu::init_cuda_backend()
        .expect("CUDA backend must initialize for CORE-179 GPU probe");
    let x = cpu_f64(&[-2.0, 0.0, 3.0], &[3], true)
        .to(Device::Cuda(0))
        .unwrap();

    let gx = only_grad(
        grad(&sum(&abs(&x).unwrap()).unwrap(), &[&x], true, true).unwrap(),
        0,
    );
    assert_eq!(
        gx.device(),
        Device::Cuda(0),
        "first gradient must remain CUDA-resident"
    );
    assert_close(
        &gx.to(Device::Cpu).unwrap().data_vec().unwrap(),
        &[-1.0, 0.0, 1.0],
        1e-12,
    );
    assert!(
        gx.requires_grad(),
        "CUDA abs first gradient should be graph-backed"
    );

    let g2 = only_grad(grad(&sum(&gx).unwrap(), &[&x], false, false).unwrap(), 0);
    assert_eq!(
        g2.device(),
        Device::Cuda(0),
        "second gradient must remain CUDA-resident"
    );
    assert_close(
        &g2.to(Device::Cpu).unwrap().data_vec().unwrap(),
        &[0.0, 0.0, 0.0],
        1e-12,
    );
}
