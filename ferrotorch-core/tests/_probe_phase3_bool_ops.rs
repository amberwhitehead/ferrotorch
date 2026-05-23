//! Phase 3a/3b sentinel (GPU dtype-parity epic, crosslink #1185): `BoolTensor`
//! is GPU-resident and its comparison / logical / reduction machinery runs on
//! CUDA (real PTX kernels; results stay resident — NO CPU round trip), matching
//! a CPU reference exactly.
//!
//! What this probe asserts:
//!   1. BoolTensor round-trips CPU→CUDA→CPU bit-exact; `data()` on a GPU tensor
//!      returns `Err(GpuTensorNotAccessible)` (no silent host readback).
//!   2. Comparison over f32 AND i32 GPU tensors, all six operators, produces a
//!      GPU-resident `BoolTensor` (`is_cuda()`); values equal the CPU reference.
//!   3. Logical and/or/xor/not on GPU bool tensors are GPU-resident and correct.
//!   4. any/all run the reduction on-device (input is GPU-resident) and return
//!      the correct scalar bool.
//!   5. to_float on a GPU bool tensor stays GPU-resident and casts correctly.
//!
//! Prints a PASS/FAIL table ending `PASS: N, FAIL: 0`. Requires the `gpu`
//! feature + a real CUDA device (run on the host RTX 3090).

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::bool_tensor::BoolTensor;
use ferrotorch_core::device::Device;
use ferrotorch_core::error::FerrotorchError;
use ferrotorch_core::int_tensor::IntTensor;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialise for Phase 3 bool-ops probe");
    });
}

fn record(label: &str, ok: bool, detail: &str, pass: &mut usize, fail: &mut usize) {
    if ok {
        *pass += 1;
        println!("PASS  {label:<34} {detail}");
    } else {
        *fail += 1;
        println!("FAIL  {label:<34} {detail}");
    }
}

/// 3a: round-trip + GpuTensorNotAccessible.
fn check_round_trip(pass: &mut usize, fail: &mut usize) {
    let host = vec![true, false, true, true, false, false, true, false];
    let cpu = BoolTensor::from_vec(host.clone(), vec![8]).expect("from_vec");

    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let resident = gpu.is_cuda();

    // data() on a GPU tensor must error (no silent readback).
    let data_errs = matches!(gpu.data(), Err(FerrotorchError::GpuTensorNotAccessible));

    let back = gpu.to(Device::Cpu).expect("to cpu");
    let bit_exact = back.data().expect("cpu data") == host.as_slice();

    record(
        "round_trip CPU->CUDA->CPU",
        resident && data_errs && bit_exact,
        &format!("resident={resident} data_errs={data_errs} bit_exact={bit_exact}"),
        pass,
        fail,
    );
}

/// 3b: comparison over f32 GPU tensors, all six ops.
fn check_compare_f32(pass: &mut usize, fail: &mut usize) {
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = [5.0f32, 2.0, 1.0, 4.0, 0.0];
    let a_cpu = ferrotorch_core::creation::from_slice::<f32>(&a, &[5]).unwrap();
    let b_cpu = ferrotorch_core::creation::from_slice::<f32>(&b, &[5]).unwrap();
    let a_g = a_cpu.to(Device::Cuda(0)).unwrap();
    let b_g = b_cpu.to(Device::Cuda(0)).unwrap();

    for name in ["eq", "ne", "lt", "le", "gt", "ge"] {
        let expected: Vec<bool> = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| match name {
                "eq" => x == y,
                "ne" => x != y,
                "lt" => x < y,
                "le" => x <= y,
                "gt" => x > y,
                _ => x >= y,
            })
            .collect();
        let m = match name {
            "eq" => BoolTensor::eq_t(&a_g, &b_g),
            "ne" => BoolTensor::ne(&a_g, &b_g),
            "lt" => BoolTensor::lt(&a_g, &b_g),
            "le" => BoolTensor::le(&a_g, &b_g),
            "gt" => BoolTensor::gt(&a_g, &b_g),
            _ => BoolTensor::ge(&a_g, &b_g),
        }
        .expect("compare f32");
        let resident = m.is_cuda();
        let vals = m.to(Device::Cpu).unwrap();
        let ok = resident && vals.data().unwrap() == expected.as_slice();
        record(
            &format!("compare f32 {name}"),
            ok,
            &format!("resident={resident}"),
            pass,
            fail,
        );
    }
}

/// 3b: comparison over integer GPU tensors, all six ops, for i32 AND i64.
fn check_compare_int(pass: &mut usize, fail: &mut usize) {
    run_six_int::<i32>("i32", pass, fail);
    run_six_int::<i64>("i64", pass, fail);
}

fn run_six_int<I: ferrotorch_core::int_tensor::IntElement>(
    dtype: &str,
    pass: &mut usize,
    fail: &mut usize,
) {
    let a = [1i64, 2, 3, 4, 5];
    let b = [5i64, 2, 1, 4, 0];
    let to_i = |v: &[i64]| -> Vec<I> { v.iter().map(|&x| I::try_from_i64(x).unwrap()).collect() };
    let a_g = IntTensor::<I>::from_vec(to_i(&a), vec![5])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let b_g = IntTensor::<I>::from_vec(to_i(&b), vec![5])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    for name in ["eq", "ne", "lt", "le", "gt", "ge"] {
        let expected: Vec<bool> = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| match name {
                "eq" => x == y,
                "ne" => x != y,
                "lt" => x < y,
                "le" => x <= y,
                "gt" => x > y,
                _ => x >= y,
            })
            .collect();
        let m = match name {
            "eq" => BoolTensor::eq_int(&a_g, &b_g),
            "ne" => BoolTensor::ne_int(&a_g, &b_g),
            "lt" => BoolTensor::lt_int(&a_g, &b_g),
            "le" => BoolTensor::le_int(&a_g, &b_g),
            "gt" => BoolTensor::gt_int(&a_g, &b_g),
            _ => BoolTensor::ge_int(&a_g, &b_g),
        }
        .expect("compare int");
        let resident = m.is_cuda();
        let vals = m.to(Device::Cpu).unwrap();
        let ok = resident && vals.data().unwrap() == expected.as_slice();
        record(
            &format!("compare {dtype} {name}"),
            ok,
            &format!("resident={resident}"),
            pass,
            fail,
        );
    }
}

/// 3b: comparison over the remaining value dtypes (f64, f16, bf16), all six
/// ops — exercises every comparison-kernel family so the full 6×6 matrix is
/// covered on hardware, not just f32/i32.
fn check_compare_other_dtypes(pass: &mut usize, fail: &mut usize) {
    let af = [1.0f64, 2.0, 3.0, 4.0, 5.0];
    let bf = [5.0f64, 2.0, 1.0, 4.0, 0.0];
    // f64
    {
        let a_g = ferrotorch_core::creation::from_slice::<f64>(&af, &[5])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let b_g = ferrotorch_core::creation::from_slice::<f64>(&bf, &[5])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        run_six_float(&a_g, &b_g, &af, &bf, "f64", pass, fail);
    }
    // f16
    {
        let a16: Vec<half::f16> = af.iter().map(|&v| half::f16::from_f64(v)).collect();
        let b16: Vec<half::f16> = bf.iter().map(|&v| half::f16::from_f64(v)).collect();
        let a_g = ferrotorch_core::creation::from_slice::<half::f16>(&a16, &[5])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let b_g = ferrotorch_core::creation::from_slice::<half::f16>(&b16, &[5])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        run_six_float(&a_g, &b_g, &af, &bf, "f16", pass, fail);
    }
    // bf16
    {
        let a16: Vec<half::bf16> = af.iter().map(|&v| half::bf16::from_f64(v)).collect();
        let b16: Vec<half::bf16> = bf.iter().map(|&v| half::bf16::from_f64(v)).collect();
        let a_g = ferrotorch_core::creation::from_slice::<half::bf16>(&a16, &[5])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let b_g = ferrotorch_core::creation::from_slice::<half::bf16>(&b16, &[5])
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        run_six_float(&a_g, &b_g, &af, &bf, "bf16", pass, fail);
    }
}

/// Run all six comparison ops on a float GPU tensor pair against an f64
/// reference (the small integer values used are exactly representable in every
/// tested float dtype, so the reference is dtype-independent).
fn run_six_float<T: ferrotorch_core::dtype::Float>(
    a_g: &ferrotorch_core::tensor::Tensor<T>,
    b_g: &ferrotorch_core::tensor::Tensor<T>,
    ar: &[f64],
    br: &[f64],
    dtype: &str,
    pass: &mut usize,
    fail: &mut usize,
) {
    for name in ["eq", "ne", "lt", "le", "gt", "ge"] {
        let expected: Vec<bool> = ar
            .iter()
            .zip(br.iter())
            .map(|(&x, &y)| match name {
                "eq" => x == y,
                "ne" => x != y,
                "lt" => x < y,
                "le" => x <= y,
                "gt" => x > y,
                _ => x >= y,
            })
            .collect();
        let m = match name {
            "eq" => BoolTensor::eq_t(a_g, b_g),
            "ne" => BoolTensor::ne(a_g, b_g),
            "lt" => BoolTensor::lt(a_g, b_g),
            "le" => BoolTensor::le(a_g, b_g),
            "gt" => BoolTensor::gt(a_g, b_g),
            _ => BoolTensor::ge(a_g, b_g),
        }
        .expect("compare");
        let resident = m.is_cuda();
        let vals = m.to(Device::Cpu).unwrap();
        let ok = resident && vals.data().unwrap() == expected.as_slice();
        record(
            &format!("compare {dtype} {name}"),
            ok,
            &format!("resident={resident}"),
            pass,
            fail,
        );
    }
}

/// 3b: logical and/or/xor/not on GPU bool tensors.
fn check_logical(pass: &mut usize, fail: &mut usize) {
    let a = vec![true, false, true, false];
    let b = vec![true, true, false, false];
    let a_g = BoolTensor::from_vec(a.clone(), vec![4])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let b_g = BoolTensor::from_vec(b.clone(), vec![4])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let cases: [(&str, Vec<bool>); 4] = [
        ("and", a.iter().zip(&b).map(|(&x, &y)| x && y).collect()),
        ("or", a.iter().zip(&b).map(|(&x, &y)| x || y).collect()),
        ("xor", a.iter().zip(&b).map(|(&x, &y)| x ^ y).collect()),
        ("not(a)", a.iter().map(|&x| !x).collect()),
    ];
    for (name, expected) in cases {
        let r = match name {
            "and" => a_g.and(&b_g).unwrap(),
            "or" => a_g.or(&b_g).unwrap(),
            "xor" => a_g.xor(&b_g).unwrap(),
            "not(a)" => a_g.not(),
            _ => unreachable!(),
        };
        let resident = r.is_cuda();
        let vals = r.to(Device::Cpu).unwrap();
        let ok = resident && vals.data().unwrap() == expected.as_slice();
        record(
            &format!("logical {name}"),
            ok,
            &format!("resident={resident}"),
            pass,
            fail,
        );
    }
}

/// 3b: any/all run the reduction on-device.
fn check_reductions(pass: &mut usize, fail: &mut usize) {
    // mixed -> any=true, all=false
    let mixed = BoolTensor::from_vec(vec![false, false, true, false], vec![4])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    // all false -> any=false
    let allfalse = BoolTensor::from_vec(vec![false, false, false], vec![3])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    // all true -> all=true
    let alltrue = BoolTensor::from_vec(vec![true, true, true, true], vec![4])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let any_mixed = mixed.any().expect("any");
    let all_mixed = mixed.all().expect("all");
    let any_false = allfalse.any().expect("any");
    let all_true = alltrue.all().expect("all");

    record(
        "any/all on GPU (reduction on device)",
        mixed.is_cuda() && any_mixed && !all_mixed && !any_false && all_true,
        &format!(
            "any(mixed)={any_mixed} all(mixed)={all_mixed} any(allfalse)={any_false} all(alltrue)={all_true}"
        ),
        pass,
        fail,
    );
}

/// 3b: to_float on GPU bool tensor stays resident and casts correctly.
fn check_to_float(pass: &mut usize, fail: &mut usize) {
    let m = BoolTensor::from_vec(vec![true, false, true, true], vec![4])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let f = m.to_float::<f32>().expect("to_float");
    let resident = f.is_cuda();
    let back = f.to(Device::Cpu).unwrap();
    let vals = back.data_vec().unwrap();
    let ok = resident && vals == vec![1.0f32, 0.0, 1.0, 1.0];
    record(
        "to_float bool->f32 (resident)",
        ok,
        &format!("resident={resident} vals={vals:?}"),
        pass,
        fail,
    );
}

#[test]
fn probe_phase3_bool_ops() {
    ensure_cuda_backend();

    let mut pass = 0usize;
    let mut fail = 0usize;

    println!("── 3a device-aware BoolTensor ────────────────────────");
    check_round_trip(&mut pass, &mut fail);
    println!("── 3b comparison (f32) ───────────────────────────────");
    check_compare_f32(&mut pass, &mut fail);
    println!("── 3b comparison (i32 / i64) ─────────────────────────");
    check_compare_int(&mut pass, &mut fail);
    println!("── 3b comparison (f64 / f16 / bf16) ──────────────────");
    check_compare_other_dtypes(&mut pass, &mut fail);
    println!("── 3b logical ────────────────────────────────────────");
    check_logical(&mut pass, &mut fail);
    println!("── 3b reductions any/all ─────────────────────────────");
    check_reductions(&mut pass, &mut fail);
    println!("── 3b to_float ───────────────────────────────────────");
    check_to_float(&mut pass, &mut fail);

    println!("──────────────────────────────────────────────────────");
    println!("PASS: {pass}, FAIL: {fail}");
    assert_eq!(fail, 0, "Phase 3 bool-ops probe had failures");
}
