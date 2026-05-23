//! Phase 3c sentinel (GPU dtype-parity epic, crosslink #1185): the mask-driven
//! ops run on CUDA with a GPU-resident `BoolTensor` mask — real PTX kernels,
//! results stay resident (NO CPU round trip for the data), matching a CPU
//! reference exactly.
//!
//! What this probe asserts:
//!   1. `masked_fill` on a GPU f32 (and bf16) tensor with a GPU `BoolTensor`
//!      mask → result `is_cuda()` + values vs CPU reference.
//!   2. `where_cond_bt` on GPU f32 tensors + GPU bool cond → `is_cuda()` +
//!      values.
//!   3. `masked_select` on GPU f32 + GPU mask → correct 1-D output (length +
//!      values) vs CPU reference; output `is_cuda()`.
//!   4. Attention-mask-style `where_cond(causal_mask, scores, -inf)` stays
//!      GPU-resident.
//!
//! The only host crossing in the whole probe's op paths is the single
//! masked_select output-length integer (the data-dependent SHAPE, PyTorch
//! parity); masked_fill / where_cond are fully resident. The probe pulls
//! results back to host ONLY for value assertions (explicit `.to(Cpu)`), which
//! is the test reading the answer, not the op detouring through host.
//!
//! Prints a PASS/FAIL table ending `PASS: N, FAIL: 0`. Requires the `gpu`
//! feature + a real CUDA device (run on the host RTX 3090).

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::bool_tensor::BoolTensor;
use ferrotorch_core::device::Device;
use ferrotorch_core::ops::indexing::{masked_select, where_cond_bt};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialise for Phase 3c masked-ops probe");
    });
}

fn record(label: &str, ok: bool, detail: &str, pass: &mut usize, fail: &mut usize) {
    if ok {
        *pass += 1;
        println!("PASS  {label:<40} {detail}");
    } else {
        *fail += 1;
        println!("FAIL  {label:<40} {detail}");
    }
}

/// masked_fill on a GPU f32 tensor + GPU bool mask, vs CPU reference.
fn check_masked_fill_f32(pass: &mut usize, fail: &mut usize) {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let mask_h = vec![false, true, false, true, true];
    let value = -9.0f32;

    let expected: Vec<f32> = input
        .iter()
        .zip(&mask_h)
        .map(|(&v, &m)| if m { value } else { v })
        .collect();

    let t = ferrotorch_core::creation::from_slice::<f32>(&input, &[5])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let mask = BoolTensor::from_vec(mask_h.clone(), vec![5])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let r = t.masked_fill(&mask, value).expect("masked_fill f32");
    let resident = r.is_cuda();
    let vals = r.to(Device::Cpu).unwrap().data_vec().unwrap();
    let ok = resident && vals == expected;
    record(
        "masked_fill f32 (resident)",
        ok,
        &format!("resident={resident} vals={vals:?}"),
        pass,
        fail,
    );
}

/// masked_fill on a GPU bf16 tensor + GPU bool mask, vs CPU reference.
fn check_masked_fill_bf16(pass: &mut usize, fail: &mut usize) {
    let input_f = [1.0f32, 2.0, 3.0, 4.0];
    let mask_h = vec![true, false, false, true];
    let value = 7.0f32;

    let expected: Vec<f32> = input_f
        .iter()
        .zip(&mask_h)
        .map(|(&v, &m)| if m { value } else { v })
        .collect();

    let input16: Vec<half::bf16> = input_f.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let t = ferrotorch_core::creation::from_slice::<half::bf16>(&input16, &[4])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let mask = BoolTensor::from_vec(mask_h.clone(), vec![4])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let r = t
        .masked_fill(&mask, half::bf16::from_f32(value))
        .expect("masked_fill bf16");
    let resident = r.is_cuda();
    let vals: Vec<f32> = r
        .to(Device::Cpu)
        .unwrap()
        .data_vec()
        .unwrap()
        .into_iter()
        .map(|b| b.to_f32())
        .collect();
    // bf16(small int) is exact; expected values are exact in bf16.
    let ok = resident && vals == expected;
    record(
        "masked_fill bf16 (resident)",
        ok,
        &format!("resident={resident} vals={vals:?}"),
        pass,
        fail,
    );
}

/// where_cond on GPU f32 tensors + GPU bool cond, vs CPU reference.
fn check_where_cond_f32(pass: &mut usize, fail: &mut usize) {
    let x = [10.0f32, 20.0, 30.0, 40.0, 50.0];
    let y = [-1.0f32, -2.0, -3.0, -4.0, -5.0];
    let cond_h = vec![true, false, true, false, true];

    let expected: Vec<f32> = cond_h
        .iter()
        .zip(x.iter().zip(y.iter()))
        .map(|(&c, (&xv, &yv))| if c { xv } else { yv })
        .collect();

    let x_g = ferrotorch_core::creation::from_slice::<f32>(&x, &[5])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let y_g = ferrotorch_core::creation::from_slice::<f32>(&y, &[5])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let cond = BoolTensor::from_vec(cond_h.clone(), vec![5])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let r = where_cond_bt(&cond, &x_g, &y_g).expect("where_cond f32");
    let resident = r.is_cuda();
    let vals = r.to(Device::Cpu).unwrap().data_vec().unwrap();
    let ok = resident && vals == expected;
    record(
        "where_cond f32 (resident)",
        ok,
        &format!("resident={resident} vals={vals:?}"),
        pass,
        fail,
    );
}

/// masked_select on GPU f32 + GPU mask → correct 1-D output + resident.
fn check_masked_select_f32(pass: &mut usize, fail: &mut usize) {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let mask_h = vec![true, false, true, true, false, false, true];

    let expected: Vec<f32> = input
        .iter()
        .zip(&mask_h)
        .filter_map(|(&v, &m)| if m { Some(v) } else { None })
        .collect();

    let t = ferrotorch_core::creation::from_slice::<f32>(&input, &[7])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let mask = BoolTensor::from_vec(mask_h.clone(), vec![7])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let r = masked_select(&t, &mask).expect("masked_select f32");
    let resident = r.is_cuda();
    let shape_ok = r.shape() == [expected.len()];
    let vals = r.to(Device::Cpu).unwrap().data_vec().unwrap();
    let ok = resident && shape_ok && vals == expected;
    record(
        "masked_select f32 (len + values, resident)",
        ok,
        &format!(
            "resident={resident} len={} (expected {}) vals={vals:?}",
            r.shape().first().copied().unwrap_or(0),
            expected.len()
        ),
        pass,
        fail,
    );

    // Edge cases: all-true and all-false masks.
    let all_true = BoolTensor::ones(&[7]).to(Device::Cuda(0)).unwrap();
    let r2 = masked_select(&t, &all_true).expect("masked_select all-true");
    let v2 = r2.to(Device::Cpu).unwrap().data_vec().unwrap();
    record(
        "masked_select all-true (full copy)",
        r2.is_cuda() && v2 == input.to_vec(),
        &format!("len={}", r2.shape().first().copied().unwrap_or(0)),
        pass,
        fail,
    );

    let all_false = BoolTensor::zeros(&[7]).to(Device::Cuda(0)).unwrap();
    let r3 = masked_select(&t, &all_false).expect("masked_select all-false");
    let v3 = r3.to(Device::Cpu).unwrap().data_vec().unwrap();
    record(
        "masked_select all-false (empty)",
        r3.is_cuda() && r3.shape() == [0] && v3.is_empty(),
        &format!("len={}", r3.shape().first().copied().unwrap_or(0)),
        pass,
        fail,
    );
}

/// Attention-mask-style `where_cond(causal_mask, scores, -inf)` stays resident.
/// scores is a [3,3] tensor; causal_mask is the lower-triangular boolean mask
/// (keep where true, else -inf). Mirrors the additive-mask step in attention.
fn check_attention_mask(pass: &mut usize, fail: &mut usize) {
    let n = 3usize;
    let scores: Vec<f32> = (0..n * n).map(|i| i as f32 + 1.0).collect();
    // Causal: keep scores[i][j] where j <= i, else -inf.
    let mask_h: Vec<bool> = (0..n)
        .flat_map(|i| (0..n).map(move |j| j <= i))
        .collect();
    let neg_inf = f32::NEG_INFINITY;

    let expected: Vec<f32> = scores
        .iter()
        .zip(&mask_h)
        .map(|(&s, &keep)| if keep { s } else { neg_inf })
        .collect();

    let scores_g = ferrotorch_core::creation::from_slice::<f32>(&scores, &[n, n])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    // y = -inf tensor on device.
    let neginf_g = ferrotorch_core::creation::from_slice::<f32>(&vec![neg_inf; n * n], &[n, n])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let mask = BoolTensor::from_vec(mask_h.clone(), vec![n, n])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let masked = where_cond_bt(&mask, &scores_g, &neginf_g).expect("attention where_cond");
    let resident = masked.is_cuda();
    let vals = masked.to(Device::Cpu).unwrap().data_vec().unwrap();
    // Compare with -inf-aware equality.
    let vals_ok = vals
        .iter()
        .zip(&expected)
        .all(|(&a, &b)| a == b || (a.is_infinite() && b.is_infinite() && a.is_sign_negative() == b.is_sign_negative()));
    let ok = resident && masked.shape() == [n, n] && vals_ok;
    record(
        "attention where_cond(causal, scores, -inf)",
        ok,
        &format!("resident={resident} vals={vals:?}"),
        pass,
        fail,
    );
}

#[test]
fn probe_phase3c_masked() {
    ensure_cuda_backend();

    let mut pass = 0usize;
    let mut fail = 0usize;

    println!("── 3c masked_fill (resident bool mask) ───────────────");
    check_masked_fill_f32(&mut pass, &mut fail);
    check_masked_fill_bf16(&mut pass, &mut fail);
    println!("── 3c where_cond (resident bool cond) ────────────────");
    check_where_cond_f32(&mut pass, &mut fail);
    println!("── 3c masked_select (GPU stream compaction) ──────────");
    check_masked_select_f32(&mut pass, &mut fail);
    println!("── 3c attention-mask-style where_cond ────────────────");
    check_attention_mask(&mut pass, &mut fail);

    println!("──────────────────────────────────────────────────────");
    println!("PASS: {pass}, FAIL: {fail}");
    assert_eq!(fail, 0, "Phase 3c masked-ops probe had failures");
}
