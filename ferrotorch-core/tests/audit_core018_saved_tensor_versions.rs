use ferrotorch_core::grad_fns::arithmetic::{add, div, mul, pow, reciprocal, rsqrt, sqrt};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::grad_fns::transcendental::{
    atan2, copysign, cos, exp, hypot, log, nextafter, sin, sinc, tan,
};
use ferrotorch_core::{FerrotorchResult, Tensor, TensorStorage, view_t};

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn constant(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn assert_version_error(err: ferrotorch_core::FerrotorchError) {
    let msg = err.to_string();
    assert!(
        msg.contains("modified by an inplace operation")
            && msg.contains("version")
            && msg.contains("expected"),
        "unexpected error: {msg}"
    );
}

fn backward_sum(t: &Tensor<f32>) -> Result<(), ferrotorch_core::FerrotorchError> {
    sum(t)?.backward()
}

fn expect_input_alias_mutation_error(
    make: fn(&Tensor<f32>) -> FerrotorchResult<Tensor<f32>>,
    data: &[f32],
) {
    let x = leaf(data, &[data.len()]);
    let y = make(&x).unwrap();

    x.detach().fill_(0.75).unwrap();

    let err = backward_sum(&y).expect_err("input alias mutation must invalidate saved tensor");
    assert_version_error(err);
}

fn expect_output_alias_mutation_error(make: fn(&Tensor<f32>) -> FerrotorchResult<Tensor<f32>>) {
    let x = leaf(&[0.25, 0.5], &[2]);
    let y = make(&x).unwrap();
    let loss = sum(&y).unwrap();

    y.detach().fill_(100.0).unwrap();

    let err = loss
        .backward()
        .expect_err("output alias mutation must invalidate saved tensor");
    assert_version_error(err);
}

fn expect_binary_input_alias_mutation_error(
    make: fn(&Tensor<f32>, &Tensor<f32>) -> FerrotorchResult<Tensor<f32>>,
    mutate_first: bool,
) {
    let a = leaf(&[0.5, 1.0], &[2]);
    let b = leaf(&[2.0, 3.0], &[2]);
    let y = make(&a, &b).unwrap();

    if mutate_first {
        a.detach().fill_(0.75).unwrap();
    } else {
        b.detach().fill_(0.75).unwrap();
    }

    let err = backward_sum(&y).expect_err("input alias mutation must invalidate saved tensor");
    assert_version_error(err);
}

fn expect_binary_output_alias_mutation_error(
    make: fn(&Tensor<f32>, &Tensor<f32>) -> FerrotorchResult<Tensor<f32>>,
) {
    let a = leaf(&[0.5, 1.0], &[2]);
    let b = leaf(&[2.0, 3.0], &[2]);
    let y = make(&a, &b).unwrap();
    let loss = sum(&y).unwrap();

    y.detach().fill_(100.0).unwrap();

    let err = loss
        .backward()
        .expect_err("output alias mutation must invalidate saved tensor");
    assert_version_error(err);
}

#[test]
fn detached_alias_mutating_mul_saved_input_errors() {
    let x = leaf(&[2.0, 3.0], &[2]);
    let y = mul(&x, &x).unwrap();

    x.detach().fill_(10.0).unwrap();

    let err = backward_sum(&y).expect_err("mul backward must reject mutated saved input");
    assert_version_error(err);
}

#[test]
fn view_alias_mutating_mul_saved_input_errors() {
    let x = leaf(&[2.0, 3.0], &[2]);
    let view = view_t(&x, &[2]).unwrap();
    let y = mul(&view, &view).unwrap();

    view.detach().fill_(10.0).unwrap();

    let err = backward_sum(&y).expect_err("view alias mutation must invalidate saved input");
    assert_version_error(err);
}

#[test]
fn add_routing_inputs_are_not_overchecked() {
    let x = leaf(&[2.0, 3.0], &[2]);
    let y = add(&x, &x).unwrap();

    x.detach().fill_(10.0).unwrap();
    backward_sum(&y).expect("add backward does not save input values");

    let grad = x.grad().unwrap().unwrap();
    assert_eq!(grad.data().unwrap(), &[2.0, 2.0]);
}

#[test]
fn pow_saved_input_rejects_detached_alias_mutation() {
    let x = leaf(&[2.0, 3.0], &[2]);
    let y = pow(&x, 2.0).unwrap();

    x.detach().fill_(10.0).unwrap();

    let err = backward_sum(&y).expect_err("pow backward must reject mutated saved input");
    assert_version_error(err);
}

#[test]
fn div_saved_tensor_operand_rejects_detached_alias_mutation() {
    let a = constant(&[2.0, 2.0], &[2]);
    let b = leaf(&[2.0, 3.0], &[2]);
    let y = div(&a, &b).unwrap();

    b.detach().fill_(10.0).unwrap();

    let err = backward_sum(&y).expect_err("tensor/tensor div must reject mutated denominator");
    assert_version_error(err);
}

#[test]
fn sqrt_uses_saved_output_not_mutated_input() {
    let x = leaf(&[4.0, 9.0], &[2]);
    let y = sqrt(&x).unwrap();

    x.detach().fill_(100.0).unwrap();
    backward_sum(&y).expect("sqrt backward uses saved result, not mutated input");

    let grad = x.grad().unwrap().unwrap();
    let got = grad.data().unwrap();
    assert!((got[0] - 0.25).abs() < 1e-6, "grad[0]={}", got[0]);
    assert!((got[1] - (1.0 / 6.0)).abs() < 1e-6, "grad[1]={}", got[1]);
}

#[test]
fn sqrt_saved_output_rejects_detached_alias_mutation() {
    let x = leaf(&[4.0, 9.0], &[2]);
    let y = sqrt(&x).unwrap();
    let loss = sum(&y).unwrap();

    y.detach().fill_(100.0).unwrap();

    let err = loss
        .backward()
        .expect_err("sqrt backward must reject mutated saved output");
    assert_version_error(err);
}

#[test]
fn rsqrt_saved_output_rejects_detached_alias_mutation() {
    let x = leaf(&[4.0, 9.0], &[2]);
    let y = rsqrt(&x).unwrap();
    let loss = sum(&y).unwrap();

    y.detach().fill_(100.0).unwrap();

    let err = loss
        .backward()
        .expect_err("rsqrt backward must reject mutated saved result");
    assert_version_error(err);
}

#[test]
fn reciprocal_saved_output_rejects_detached_alias_mutation() {
    let x = leaf(&[4.0, 9.0], &[2]);
    let y = reciprocal(&x).unwrap();
    let loss = sum(&y).unwrap();

    y.detach().fill_(100.0).unwrap();

    let err = loss
        .backward()
        .expect_err("reciprocal backward must reject mutated saved result");
    assert_version_error(err);
}

#[test]
fn transcendental_input_saved_ops_reject_detached_alias_mutation() {
    expect_input_alias_mutation_error(ferrotorch_core::grad_fns::arithmetic::abs, &[-2.0, 3.0]);
    expect_input_alias_mutation_error(log, &[2.0, 3.0]);
    expect_input_alias_mutation_error(sin, &[0.25, 0.5]);
    expect_input_alias_mutation_error(cos, &[0.25, 0.5]);
    expect_input_alias_mutation_error(sinc, &[0.25, 0.5]);
}

#[test]
fn output_saved_transcendentals_reject_output_alias_mutation() {
    expect_output_alias_mutation_error(exp);
    expect_output_alias_mutation_error(tan);
}

#[test]
fn output_saved_transcendentals_do_not_overcheck_input_alias_mutation() {
    for make in [
        exp as fn(&Tensor<f32>) -> FerrotorchResult<Tensor<f32>>,
        tan,
    ] {
        let x = leaf(&[0.25, 0.5], &[2]);
        let y = make(&x).unwrap();

        x.detach().fill_(0.75).unwrap();
        backward_sum(&y).expect("output-saved backward does not save input values");
    }
}

#[test]
fn atan2_saved_operands_reject_detached_alias_mutation() {
    expect_binary_input_alias_mutation_error(atan2, true);
    expect_binary_input_alias_mutation_error(atan2, false);
}

#[test]
fn copysign_checks_saved_magnitude_and_output_but_not_sign_operand() {
    expect_binary_input_alias_mutation_error(copysign, true);
    expect_binary_output_alias_mutation_error(copysign);

    let magnitude = leaf(&[0.5, 1.0], &[2]);
    let sign = leaf(&[-2.0, 3.0], &[2]);
    let y = copysign(&magnitude, &sign).unwrap();

    sign.detach().fill_(-100.0).unwrap();
    backward_sum(&y).expect("copysign backward does not save the sign operand values");

    let grad_magnitude = magnitude.grad().unwrap().unwrap();
    assert_eq!(grad_magnitude.data().unwrap(), &[-1.0, 1.0]);
    let grad_sign = sign.grad().unwrap().unwrap();
    assert_eq!(grad_sign.data().unwrap(), &[0.0, 0.0]);
}

#[test]
fn hypot_saved_operands_and_output_reject_detached_alias_mutation() {
    expect_binary_input_alias_mutation_error(hypot, true);
    expect_binary_input_alias_mutation_error(hypot, false);
    expect_binary_output_alias_mutation_error(hypot);
}

#[test]
fn hypot_origin_backward_preserves_pytorch_nan() {
    let a = leaf(&[0.0, 1.0], &[2]);
    let b = leaf(&[0.0, 0.0], &[2]);
    let y = hypot(&a, &b).unwrap();

    backward_sum(&y).unwrap();

    let grad_a = a.grad().unwrap().unwrap();
    let grad_b = b.grad().unwrap().unwrap();
    let grad_a_data = grad_a.data().unwrap();
    let grad_b_data = grad_b.data().unwrap();
    assert!(grad_a_data[0].is_nan(), "grad_a[0]={}", grad_a_data[0]);
    assert_eq!(grad_a_data[1], 1.0);
    assert!(grad_b_data[0].is_nan(), "grad_b[0]={}", grad_b_data[0]);
    assert_eq!(grad_b_data[1], 0.0);
}

#[test]
fn nextafter_saved_operands_reject_detached_alias_mutation() {
    expect_binary_input_alias_mutation_error(nextafter, true);
    expect_binary_input_alias_mutation_error(nextafter, false);
}

#[cfg(feature = "gpu")]
#[test]
fn cuda_detached_alias_mutating_mul_saved_input_errors_before_kernel_read() {
    use ferrotorch_core::Device;
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend().expect("CUDA backend must initialise");
    });

    let x = leaf(&[2.0, 3.0], &[2]).to(Device::Cuda(0)).unwrap();
    let y = mul(&x, &x).unwrap();

    x.detach().fill_(10.0).unwrap();

    let err = backward_sum(&y).expect_err("CUDA mul backward must reject mutated saved input");
    assert_version_error(err);
}
