//! Audit probes for commit 36b245151 — verify ferrotorch matches torch
//! for various scale=0 / scale<0 / NaN-scale / Inf-scale / NaN-input cases
//! beyond the two #[test] cases already de-#[ignore]'d.
//!
//! Per R-CHAR-3 every expected value below comes from live torch oracle
//! 2026-05-25 (`/tmp/oracle_probes.py`), NOT from ferrotorch's output.

use ferrotorch_core::{IntTensor, from_vec, grad_fns};

#[allow(
    clippy::too_many_arguments,
    reason = "test probe helper mirrors fake_quantize_per_channel_affine's full (input, shape, scale, zero_point, axis, qmin, qmax) parameter surface; bundling into a struct adds no test clarity"
)]
fn run(
    name: &str,
    input: Vec<f32>,
    ishape: Vec<usize>,
    scale: Vec<f32>,
    zp: Vec<i64>,
    axis: i64,
    qmin: i64,
    qmax: i64,
) -> (Vec<f32>, Vec<u32>) {
    let i = from_vec(input, &ishape).unwrap();
    let s = from_vec(scale, &[zp.len()]).unwrap();
    let z = IntTensor::<i64>::from_vec(zp.clone(), vec![zp.len()]).unwrap();
    let out =
        grad_fns::quantize_grad::fake_quantize_per_channel_affine(&i, &s, &z, axis, qmin, qmax)
            .expect(name);
    let data = out.data().unwrap().to_vec();
    let bits: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
    (data, bits)
}

#[test]
fn s0_1_scale0_zp0_qmin_neg128_minus_zero() {
    // Torch live 2026-05-25:
    //   input=[[5.0]], scale=[0.0], zp=[0], axis=0, qmin=-128, qmax=127
    //   -> tensor([[-0.]])
    let (_d, b) = run(
        "S0.1",
        vec![5.0],
        vec![1, 1],
        vec![0.0],
        vec![0],
        0,
        -128,
        127,
    );
    assert_eq!(
        b[0], 0x80000000,
        "S0.1 expected -0.0 (0x80000000), got 0x{:08x}",
        b[0]
    );
}

#[test]
fn s0_2_scale0_zp64() {
    // Torch live: tensor([[-0.]])
    let (_d, b) = run(
        "S0.2",
        vec![5.0],
        vec![1, 1],
        vec![0.0],
        vec![64],
        0,
        -128,
        127,
    );
    assert_eq!(b[0], 0x80000000, "S0.2 expected -0.0, got 0x{:08x}", b[0]);
}

#[test]
fn s0_3_scale0_zp0_qmin_zero_anchored_plus_zero() {
    // qmin=0, zp=0 → (qmin-zp)=0, 0*0.0 = +0.0
    // Torch live: tensor([[0.]]) signbit False
    let (_d, b) = run("S0.3", vec![5.0], vec![1, 1], vec![0.0], vec![0], 0, 0, 127);
    assert_eq!(b[0], 0x00000000, "S0.3 expected +0.0, got 0x{:08x}", b[0]);
}

#[test]
fn s0_4_scale0_zp_above_qmin_minus_zero() {
    // qmin=0, zp=10 → (qmin-zp)=-10, -10*0.0 = -0.0
    // Torch live: tensor([[-0.]])
    let (_d, b) = run(
        "S0.4",
        vec![5.0],
        vec![1, 1],
        vec![0.0],
        vec![10],
        0,
        0,
        127,
    );
    assert_eq!(
        b[0], 0x80000000,
        "S0.4 expected -0.0 (0x80000000), got 0x{:08x}",
        b[0]
    );
}

#[test]
fn s0_5_scale0_neg_input() {
    let (_d, b) = run(
        "S0.5",
        vec![-5.0],
        vec![1, 1],
        vec![0.0],
        vec![0],
        0,
        -128,
        127,
    );
    assert_eq!(b[0], 0x80000000, "S0.5 expected -0.0, got 0x{:08x}", b[0]);
}

#[test]
fn s0_6_scale0_zero_input() {
    let (_d, b) = run(
        "S0.6",
        vec![0.0],
        vec![1, 1],
        vec![0.0],
        vec![0],
        0,
        -128,
        127,
    );
    assert_eq!(b[0], 0x80000000, "S0.6 expected -0.0, got 0x{:08x}", b[0]);
}

#[test]
fn s0_7_scale0_zp_at_qmin_plus_zero() {
    // qmin=10, zp=10 → (qmin-zp)=0 → +0.0
    // Torch live: tensor([[0.]])
    let (_d, b) = run(
        "S0.7",
        vec![5.0],
        vec![1, 1],
        vec![0.0],
        vec![10],
        0,
        10,
        200,
    );
    assert_eq!(b[0], 0x00000000, "S0.7 expected +0.0, got 0x{:08x}", b[0]);
}

#[test]
fn sn_1_neg_scale_saturate() {
    // input=300 with scale=-2: 300/-2=-150 → clamps to qmin=-128 → (-128-0)*-2 = 256.
    // Torch live: [[100., 200., 256., -100.]]
    let (d, _b) = run(
        "Sn.1",
        vec![100.0, 200.0, 300.0, -100.0],
        vec![1, 4],
        vec![-2.0],
        vec![0],
        0,
        -128,
        127,
    );
    assert!(
        (d[0] - 100.0).abs() < 1e-5,
        "Sn.1[0] expected 100, got {}",
        d[0]
    );
    assert!(
        (d[1] - 200.0).abs() < 1e-5,
        "Sn.1[1] expected 200, got {}",
        d[1]
    );
    assert!(
        (d[2] - 256.0).abs() < 1e-5,
        "Sn.1[2] expected 256 (clamp under neg scale), got {}",
        d[2]
    );
    assert!(
        (d[3] - -100.0).abs() < 1e-5,
        "Sn.1[3] expected -100, got {}",
        d[3]
    );
}

#[test]
fn nan_scale_propagates_only_in_nan_channel() {
    // Torch live: input=[[1,2,3]] scale=[NaN,1,1] zp=[0,0,0] axis=1
    //   -> tensor([[nan, 2., 3.]])
    let (d, _b) = run(
        "N.1",
        vec![1.0, 2.0, 3.0],
        vec![1, 3],
        vec![f32::NAN, 1.0, 1.0],
        vec![0, 0, 0],
        1,
        -128,
        127,
    );
    assert!(d[0].is_nan(), "N.1[0] expected NaN, got {}", d[0]);
    assert!(
        (d[1] - 2.0).abs() < 1e-5,
        "N.1[1] expected 2.0, got {}",
        d[1]
    );
    assert!(
        (d[2] - 3.0).abs() < 1e-5,
        "N.1[2] expected 3.0, got {}",
        d[2]
    );
}

#[test]
fn pos_inf_scale_yields_nan() {
    // scale=+Inf → inv_scale=0, qval = zp + nearbyint(x*0) = zp+0=zp,
    // clamp passes, dequant (zp-zp)*+Inf = 0*Inf = NaN.
    // Torch live: tensor([[nan, nan, nan]])
    let (d, _b) = run(
        "I.1",
        vec![1.0, -1.0, 0.0],
        vec![1, 3],
        vec![f32::INFINITY],
        vec![0],
        0,
        -128,
        127,
    );
    for (i, v) in d.iter().enumerate() {
        assert!(v.is_nan(), "I.1[{i}] expected NaN, got {v}");
    }
}

#[test]
fn nan_input_clamps_to_qmin() {
    // NaN input → x*inv_scale = NaN → cast i64 = INT64_MIN → clamps to qmin
    // → (qmin-zp)*scale = (-128-0)*1.0 = -128.
    // Torch live: input=[[NaN]] scale=[1.0] zp=[0] -> tensor([[-128.]])
    let (d, _b) = run(
        "NI.1",
        vec![f32::NAN],
        vec![1, 1],
        vec![1.0],
        vec![0],
        0,
        -128,
        127,
    );
    assert!(
        (d[0] - -128.0).abs() < 1e-5,
        "NI.1 expected -128 (NaN input clamps to qmin), got {}",
        d[0]
    );
}
