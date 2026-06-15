#![cfg(feature = "gpu")]

//! CUDA integer comparison broadcasting parity.
//!
//! PyTorch 2.11.0+cu130 routes broadcasted integer comparisons through CUDA
//! TensorIterator and returns a CUDA bool tensor. These tests pin i32/i64
//! broadcasting, scalar operands, and zero-size outputs without allowing a CPU
//! value round trip in the implementation path.

use std::sync::Once;

use ferrotorch_core::bool_tensor::BoolTensor;
use ferrotorch_core::device::Device;
use ferrotorch_core::int_tensor::{IntElement, IntTensor};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for integer comparison tests");
    });
}

fn cuda_int<I: IntElement>(values: Vec<I>, shape: &[usize]) -> IntTensor<I> {
    IntTensor::<I>::from_vec(values, shape.to_vec())
        .expect("CPU IntTensor")
        .to(Device::Cuda(0))
        .expect("upload IntTensor")
}

fn cuda_scalar<I: IntElement>(value: I) -> IntTensor<I> {
    IntTensor::<I>::scalar(value)
        .to(Device::Cuda(0))
        .expect("upload scalar IntTensor")
}

fn int_vec<I: IntElement>(values: &[i64]) -> Vec<I> {
    values
        .iter()
        .map(|&v| I::try_from_i64(v).expect("test value fits integer dtype"))
        .collect()
}

fn bool_values(mask: &BoolTensor) -> Vec<bool> {
    assert_eq!(mask.device(), Device::Cuda(0), "mask must stay on CUDA");
    mask.to(Device::Cpu)
        .expect("assertion readback")
        .data()
        .expect("BoolTensor data")
        .to_vec()
}

fn assert_matrix_ops<I: IntElement>() {
    let a = cuda_int(
        vec![I::try_from_i64(1).unwrap(), I::try_from_i64(3).unwrap()],
        &[2, 1],
    );
    let b = cuda_int(
        vec![
            I::try_from_i64(0).unwrap(),
            I::try_from_i64(1).unwrap(),
            I::try_from_i64(4).unwrap(),
        ],
        &[1, 3],
    );

    let gt = BoolTensor::gt_int(&a, &b).expect("broadcast gt");
    assert_eq!(gt.shape(), &[2, 3]);
    assert_eq!(
        bool_values(&gt),
        vec![true, false, false, true, true, false]
    );

    let le = BoolTensor::le_int(&a, &b).expect("broadcast le");
    assert_eq!(le.shape(), &[2, 3]);
    assert_eq!(
        bool_values(&le),
        vec![false, true, true, false, false, true]
    );

    let eq = BoolTensor::eq_int(&a, &b).expect("broadcast eq");
    assert_eq!(eq.shape(), &[2, 3]);
    assert_eq!(
        bool_values(&eq),
        vec![false, true, false, false, false, false]
    );

    let ne = BoolTensor::ne_int(&a, &b).expect("broadcast ne");
    assert_eq!(ne.shape(), &[2, 3]);
    assert_eq!(bool_values(&ne), vec![true, false, true, true, true, true]);
}

fn assert_high_rank_broadcast<I: IntElement>() {
    let a_host = int_vec::<I>(&[1, 4, 7, 2, 5, 8]);
    let b_host = int_vec::<I>(&[0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 6, 7, 8, 9, 10]);
    let a = cuda_int(a_host.clone(), &[2, 1, 3, 1]);
    let b = cuda_int(b_host.clone(), &[3, 5]);

    let le = BoolTensor::le_int(&a, &b).expect("high-rank broadcast le");
    assert_eq!(le.shape(), &[2, 1, 3, 5]);
    assert_eq!(le.device(), Device::Cuda(0));

    let mut expected = Vec::with_capacity(2 * 3 * 5);
    for batch in 0..2 {
        for row in 0..3 {
            for col in 0..5 {
                let a_value = a_host[batch * 3 + row].to_i64();
                let b_value = b_host[row * 5 + col].to_i64();
                expected.push(a_value <= b_value);
            }
        }
    }
    assert_eq!(bool_values(&le), expected);
}

fn assert_scalar_and_empty<I: IntElement>() {
    let scalar = cuda_scalar(I::try_from_i64(3).unwrap());
    let vector = cuda_int(
        vec![
            I::try_from_i64(2).unwrap(),
            I::try_from_i64(3).unwrap(),
            I::try_from_i64(4).unwrap(),
        ],
        &[3],
    );
    let ge = BoolTensor::ge_int(&scalar, &vector).expect("scalar ge broadcast");
    assert_eq!(ge.shape(), &[3]);
    assert_eq!(bool_values(&ge), vec![true, true, false]);

    let empty = cuda_int(Vec::<I>::new(), &[0, 1]);
    let row = cuda_int(
        vec![
            I::try_from_i64(10).unwrap(),
            I::try_from_i64(20).unwrap(),
            I::try_from_i64(30).unwrap(),
        ],
        &[1, 3],
    );
    let lt = BoolTensor::lt_int(&empty, &row).expect("empty lt broadcast");
    assert_eq!(lt.shape(), &[0, 3]);
    assert_eq!(lt.device(), Device::Cuda(0));
    assert_eq!(lt.numel(), 0);
    assert!(bool_values(&lt).is_empty());
}

#[test]
fn int_compare_cuda_broadcast_i32_matches_torch() {
    ensure_cuda_backend();
    assert_matrix_ops::<i32>();
    assert_high_rank_broadcast::<i32>();
    assert_scalar_and_empty::<i32>();
}

#[test]
fn int_compare_cuda_broadcast_i64_matches_torch() {
    ensure_cuda_backend();
    assert_matrix_ops::<i64>();
    assert_high_rank_broadcast::<i64>();
    assert_scalar_and_empty::<i64>();
}
