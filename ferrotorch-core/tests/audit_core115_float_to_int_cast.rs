//! CORE-115 (#1809): float-to-integer casts must use PyTorch's target-width
//! conversion result, not Rust's float -> i64 saturation followed by a narrow
//! range check.
//!
//! Live oracle, torch 2.11.0+cu130 on this host:
//! - CPU copy casts flow through `c10::convert` / C++ `static_cast`; invalid
//!   float-to-int conversions produce the target-width integer-indefinite
//!   sentinel (`INT_MIN` / `LONG_MIN`) and do not raise range errors.
//! - CUDA copy casts use device static-cast/PTX behavior, which differs from
//!   CPU for some invalid conversions. The CUDA tests below pin PyTorch CUDA
//!   outputs and assert Ferrotorch keeps the result resident on CUDA.
//!
//! Upstream source: `/home/doll/pytorch/c10/util/TypeCast.h` and
//! `aten/src/ATen/native/{cpu,cuda}/Copy*`.

use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

#[cfg(feature = "gpu")]
use ferrotorch_core::device::Device;

const I32_MIN: i32 = i32::MIN;
const I32_MAX: i32 = i32::MAX;
const I64_MIN: i64 = i64::MIN;
#[cfg(feature = "gpu")]
const I64_MAX: i64 = i64::MAX;

fn tensor<T: ferrotorch_core::dtype::Float>(data: Vec<T>) -> Tensor<T> {
    let len = data.len();
    Tensor::from_storage(TensorStorage::cpu(data), vec![len], false).unwrap()
}

fn read_i32(t: &IntTensor<i32>) -> Vec<i32> {
    t.data().unwrap().to_vec()
}

fn read_i64(t: &IntTensor<i64>) -> Vec<i64> {
    t.data().unwrap().to_vec()
}

#[cfg(feature = "gpu")]
fn ensure_cuda_backend() {
    ferrotorch_gpu::init_cuda_backend().expect("CUDA backend init for CORE-115 suite");
}

#[cfg(feature = "gpu")]
fn read_cuda_i32(t: &IntTensor<i32>) -> Vec<i32> {
    assert!(t.is_cuda(), "cast result must stay CUDA-resident");
    t.to(Device::Cpu).unwrap().data().unwrap().to_vec()
}

#[cfg(feature = "gpu")]
fn read_cuda_i64(t: &IntTensor<i64>) -> Vec<i64> {
    assert!(t.is_cuda(), "cast result must stay CUDA-resident");
    t.to(Device::Cpu).unwrap().data().unwrap().to_vec()
}

#[test]
fn cpu_f32_to_int_exceptional_values_match_torch_cpu() {
    let x = tensor(vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        1.9,
        -1.9,
        2_147_483_647.0,
        2_147_483_648.0,
        -2_147_483_648.0,
        -2_147_483_904.0,
        9_223_372_036_854_775_808.0,
        -9_223_372_036_854_775_808.0,
    ]);

    assert_eq!(
        read_i32(&x.to_int::<i32>().unwrap()),
        vec![
            I32_MIN, I32_MIN, I32_MIN, 0, 0, 1, -1, I32_MIN, I32_MIN, I32_MIN, I32_MIN, I32_MIN,
            I32_MIN,
        ]
    );
    assert_eq!(
        read_i64(&x.to_int::<i64>().unwrap()),
        vec![
            I64_MIN,
            I64_MIN,
            I64_MIN,
            0,
            0,
            1,
            -1,
            2_147_483_648,
            2_147_483_648,
            -2_147_483_648,
            -2_147_483_904,
            I64_MIN,
            I64_MIN,
        ]
    );
}

#[test]
fn cpu_f64_to_int_exceptional_values_match_torch_cpu() {
    let x = tensor(vec![
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        1.9,
        -1.9,
        2_147_483_647.0,
        2_147_483_648.0,
        -2_147_483_648.0,
        -2_147_483_649.0,
        9_223_372_036_854_774_784.0,
        9_223_372_036_854_775_808.0,
        -9_223_372_036_854_775_808.0,
        -9_223_372_036_854_777_856.0,
        1.0e20,
        -1.0e20,
    ]);

    assert_eq!(
        read_i32(&x.to_int::<i32>().unwrap()),
        vec![
            I32_MIN, I32_MIN, I32_MIN, 0, 0, 1, -1, I32_MAX, I32_MIN, I32_MIN, I32_MIN, I32_MIN,
            I32_MIN, I32_MIN, I32_MIN, I32_MIN, I32_MIN,
        ]
    );
    assert_eq!(
        read_i64(&x.to_int::<i64>().unwrap()),
        vec![
            I64_MIN,
            I64_MIN,
            I64_MIN,
            0,
            0,
            1,
            -1,
            2_147_483_647,
            2_147_483_648,
            -2_147_483_648,
            -2_147_483_649,
            9_223_372_036_854_774_784,
            I64_MIN,
            I64_MIN,
            I64_MIN,
            I64_MIN,
            I64_MIN,
        ]
    );
}

#[test]
fn cpu_f16_to_int_exceptional_values_match_torch_cpu() {
    let x = tensor(
        [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.9,
            -1.9,
            65_504.0,
            -65_504.0,
            2_147_483_648.0,
            -2_147_483_904.0,
            1.0e20,
            -1.0e20,
        ]
        .into_iter()
        .map(half::f16::from_f32)
        .collect(),
    );

    assert_eq!(
        read_i32(&x.to_int::<i32>().unwrap()),
        vec![
            I32_MIN, I32_MIN, I32_MIN, 1, -1, 65_504, -65_504, I32_MIN, I32_MIN, I32_MIN, I32_MIN,
        ]
    );
    assert_eq!(
        read_i64(&x.to_int::<i64>().unwrap()),
        vec![
            I64_MIN, I64_MIN, I64_MIN, 1, -1, 65_504, -65_504, I64_MIN, I64_MIN, I64_MIN, I64_MIN,
        ]
    );
}

#[test]
fn cpu_bf16_to_int_exceptional_values_match_torch_cpu() {
    let x = tensor(
        [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.9,
            -1.9,
            65_504.0,
            -65_504.0,
            2_147_483_648.0,
            -2_147_483_904.0,
            1.0e20,
            -1.0e20,
        ]
        .into_iter()
        .map(half::bf16::from_f32)
        .collect(),
    );

    assert_eq!(
        read_i32(&x.to_int::<i32>().unwrap()),
        vec![
            I32_MIN, I32_MIN, I32_MIN, 1, -1, 65_536, -65_536, I32_MIN, I32_MIN, I32_MIN, I32_MIN,
        ]
    );
    assert_eq!(
        read_i64(&x.to_int::<i64>().unwrap()),
        vec![
            I64_MIN,
            I64_MIN,
            I64_MIN,
            1,
            -1,
            65_536,
            -65_536,
            2_147_483_648,
            -2_147_483_648,
            I64_MIN,
            I64_MIN,
        ]
    );
}

#[cfg(feature = "gpu")]
#[test]
fn cuda_f32_to_int_exceptional_values_match_torch_cuda() {
    ensure_cuda_backend();
    let x = tensor(vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        1.9,
        -1.9,
        2_147_483_647.0,
        2_147_483_648.0,
        -2_147_483_648.0,
        -2_147_483_904.0,
        9_223_372_036_854_775_808.0,
        -9_223_372_036_854_775_808.0,
    ])
    .to(Device::Cuda(0))
    .unwrap();

    assert_eq!(
        read_cuda_i32(&x.to_int::<i32>().unwrap()),
        vec![
            0, I32_MAX, I32_MIN, 0, 0, 1, -1, I32_MAX, I32_MAX, I32_MIN, I32_MIN, I32_MAX, I32_MIN,
        ]
    );
    assert_eq!(
        read_cuda_i64(&x.to_int::<i64>().unwrap()),
        vec![
            I64_MIN,
            I64_MAX,
            I64_MIN,
            0,
            0,
            1,
            -1,
            2_147_483_648,
            2_147_483_648,
            -2_147_483_648,
            -2_147_483_904,
            I64_MAX,
            I64_MIN,
        ]
    );
}

#[cfg(feature = "gpu")]
#[test]
fn cuda_f64_to_int_exceptional_values_match_torch_cuda() {
    ensure_cuda_backend();
    let x = tensor(vec![
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        1.9,
        -1.9,
        2_147_483_647.0,
        2_147_483_648.0,
        -2_147_483_648.0,
        -2_147_483_649.0,
        9_223_372_036_854_774_784.0,
        9_223_372_036_854_775_808.0,
        -9_223_372_036_854_775_808.0,
        -9_223_372_036_854_777_856.0,
        1.0e20,
        -1.0e20,
    ])
    .to(Device::Cuda(0))
    .unwrap();

    assert_eq!(
        read_cuda_i32(&x.to_int::<i32>().unwrap()),
        vec![
            I32_MIN, I32_MAX, I32_MIN, 0, 0, 1, -1, I32_MAX, I32_MAX, I32_MIN, I32_MIN, I32_MAX,
            I32_MAX, I32_MIN, I32_MIN, I32_MAX, I32_MIN,
        ]
    );
    assert_eq!(
        read_cuda_i64(&x.to_int::<i64>().unwrap()),
        vec![
            I64_MIN,
            I64_MAX,
            I64_MIN,
            0,
            0,
            1,
            -1,
            2_147_483_647,
            2_147_483_648,
            -2_147_483_648,
            -2_147_483_649,
            9_223_372_036_854_774_784,
            I64_MAX,
            I64_MIN,
            I64_MIN,
            I64_MAX,
            I64_MIN,
        ]
    );
}

#[cfg(feature = "gpu")]
#[test]
fn cuda_reduced_float_to_int_exceptional_values_match_torch_cuda() {
    ensure_cuda_backend();
    let values = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.9,
        -1.9,
        65_504.0,
        -65_504.0,
        2_147_483_648.0,
        -2_147_483_904.0,
        1.0e20,
        -1.0e20,
    ];

    let f16 = tensor(values.into_iter().map(half::f16::from_f32).collect())
        .to(Device::Cuda(0))
        .unwrap();
    assert_eq!(
        read_cuda_i32(&f16.to_int::<i32>().unwrap()),
        vec![
            0, I32_MAX, I32_MIN, 1, -1, 65_504, -65_504, I32_MAX, I32_MIN, I32_MAX, I32_MIN,
        ]
    );
    assert_eq!(
        read_cuda_i64(&f16.to_int::<i64>().unwrap()),
        vec![
            I64_MIN, I64_MAX, I64_MIN, 1, -1, 65_504, -65_504, I64_MAX, I64_MIN, I64_MAX, I64_MIN,
        ]
    );

    let bf16 = tensor(values.into_iter().map(half::bf16::from_f32).collect())
        .to(Device::Cuda(0))
        .unwrap();
    assert_eq!(
        read_cuda_i32(&bf16.to_int::<i32>().unwrap()),
        vec![
            0, I32_MAX, I32_MIN, 1, -1, 65_536, -65_536, I32_MAX, I32_MIN, I32_MAX, I32_MIN,
        ]
    );
    assert_eq!(
        read_cuda_i64(&bf16.to_int::<i64>().unwrap()),
        vec![
            I64_MIN,
            I64_MAX,
            I64_MIN,
            1,
            -1,
            65_536,
            -65_536,
            2_147_483_648,
            -2_147_483_648,
            I64_MAX,
            I64_MIN,
        ]
    );
}
