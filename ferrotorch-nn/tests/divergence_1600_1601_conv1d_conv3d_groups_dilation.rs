//! ACToR discriminator re-audit of commit `7acfe4bf5` (#1600 Conv1d +
//! #1601 Conv3d groups+dilation, forward + backward).
//!
//! Every EXPECTED value below is the LIVE torch 2.11.0+cu130 output of the
//! matching `torch.nn.functional.conv{1,3}d` call, computed with the torch
//! driver reproduced inline above each constant block, NOT copied from the
//! ferrotorch side (R-CHAR-3).
//!
//! WHY THIS FILE EXISTS — the builder's own conv.rs unit tests for #1600/#1601
//! only exercise SYMMETRIC-magnitude per-group weights and SYMMETRIC kernels /
//! dilation. Those configurations do NOT discriminate the two classic silent
//! grouped-conv bugs:
//!   1. cross-group channel LEAK: when both groups' weights share the same
//!      `arange(..)*c` pattern, an output channel reading the WRONG group's
//!      input channels produces plausible-but-wrong numbers that are still in
//!      the same ballpark — masked by a loose tol.
//!   2. D/H/W axis SWAP in the 5-D col2im/im2col bookkeeping: a symmetric
//!      kernel `(2,2,2)` with symmetric dilation `(2,2,2)` over a cubic volume
//!      is invariant under D<->H<->W transposition, so a swapped axis passes.
//! This file pins both with DISTINCT-magnitude per-group weights (a leak blows
//! up by orders of magnitude) and an ASYMMETRIC kernel `(1,2,3)` + asymmetric
//! dilation `(1,2,1)` over a non-cubic `3x5x4` volume (an axis swap changes
//! the output shape AND values).
//!
//! Production path driven: `Conv1d::new_full` / `Conv3d::new_full` +
//! `Module::parameters_mut` + `Parameter::set_data` + `Module::forward` then
//! `grad_fn().backward(grad_output)` — exactly the production consumer chain
//! (lazy_conv.rs materialize -> new_full -> forward -> autograd engine).

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::{Conv1d, Conv3d};

/// Leaf tensor that does NOT require grad (weights/bias get requires_grad via
/// `Parameter::set_data`; inputs needing grad use `leaf`).
fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// Leaf tensor that requires grad (so `grad_input` is computed in backward).
fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

/// torch-envelope close check (rtol 1e-4, atol 1e-5). A cross-group leak or a
/// D/H/W axis swap diverges by O(1)..O(1000) >> this tolerance.
fn assert_close(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = 1e-4_f32 + 1e-3_f32 * w.abs();
        assert!(
            (g - w).abs() <= tol,
            "{ctx}: element {i} ferrotorch={g} torch={w} |diff|={} > tol={tol}",
            (g - w).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// TEST A — Conv1d groups=2 with DISTINCT per-group weight magnitudes.
//
// Catches a cross-group channel LEAK. Output channels 2,3 (group 1) read ONLY
// input channels 2,3; output channels 0,1 (group 0) read ONLY input channels
// 0,1 (`aten/src/ATen/native/Convolution.cpp:1723-1729`: subtensor(input,1,g)
// / subtensor(weight,0,g)). Group-1 weights are ~100x group-0 weights, so a
// leak (group-1 output reading group-0 input or vice versa) diverges by a huge
// margin instead of staying in the same ballpark.
//
// torch driver:
//   w = torch.tensor([0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,
//                     10.,11.,12.,13., 14.,15.,16.,17.]).reshape(4,2,2)
//   b = torch.tensor([0.5,-0.5,0.25,-0.25])
//   x = torch.arange(1,21).float().reshape(1,4,5).requires_grad_(True)
//   y = F.conv1d(x, w, b, stride=1, padding=0, dilation=1, groups=2)
//   y.sum().backward()
// ---------------------------------------------------------------------------
#[test]
fn divergence_1600_conv1d_groups2_distinct_weights_no_cross_group_leak() {
    let weight = [
        0.1f32, 0.2, 0.3, 0.4, // out0 (group0)
        0.5, 0.6, 0.7, 0.8, // out1 (group0)
        10.0, 11.0, 12.0, 13.0, // out2 (group1)
        14.0, 15.0, 16.0, 17.0, // out3 (group1)
    ];
    let bias = [0.5f32, -0.5, 0.25, -0.25];

    let mut conv = Conv1d::<f32>::new_full(4, 4, 2, 1, 0, 1, 2, true).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&weight, &[4, 2, 2]));
        params[1].set_data(t(&bias, &[4]));
    }

    let x_data: Vec<f32> = (1..=20).map(|i| i as f32).collect();
    let x = leaf(&x_data, &[1, 4, 5]);
    let y = Module::<f32>::forward(&conv, &x).unwrap();
    assert_eq!(y.shape(), &[1, 4, 4]);

    // torch A_fwd: note group-1 channels (>=655) dwarf group-0 (<=18.8). A leak
    // would mix these magnitudes and fail the close check.
    let a_fwd = [
        5.6f32, 6.6, 7.6, 8.6, 11.0, 13.6, 16.2, 18.8, 655.25, 701.25, 747.25, 793.25, 878.75,
        940.75, 1002.75, 1064.75,
    ];
    assert_close(y.data().unwrap(), &a_fwd, "A_fwd conv1d groups=2 distinct weights");

    // out.sum().backward() => grad_output = ones.
    let grad_output = t(&[1.0f32; 16], &[1, 4, 4]);
    let grads = Module::<f32>::forward(&conv, &x)
        .unwrap()
        .grad_fn()
        .unwrap()
        .backward(&grad_output)
        .unwrap();

    // torch A_gx: group-1 input gradients (>=24) >> group-0 (<=2.2). A
    // mis-routed grad_input would cross these magnitudes.
    let a_gx = [
        0.6f32, 1.4, 1.4, 1.4, 0.8, 1.0, 2.2, 2.2, 2.2, 1.2, 24.0, 50.0, 50.0, 50.0, 26.0, 28.0,
        58.0, 58.0, 58.0, 30.0,
    ];
    assert_close(
        grads[0].as_ref().unwrap().data().unwrap(),
        &a_gx,
        "A_gx conv1d groups=2 grad_input",
    );

    // torch A_gw [4,2,2].
    assert_eq!(grads[1].as_ref().unwrap().shape(), &[4, 2, 2]);
    let a_gw = [
        10.0f32, 14.0, 30.0, 34.0, 10.0, 14.0, 30.0, 34.0, 50.0, 54.0, 70.0, 74.0, 50.0, 54.0,
        70.0, 74.0,
    ];
    assert_close(
        grads[1].as_ref().unwrap().data().unwrap(),
        &a_gw,
        "A_gw conv1d groups=2 grad_weight",
    );

    // torch A_gb.
    assert_close(
        grads[2].as_ref().unwrap().data().unwrap(),
        &[4.0, 4.0, 4.0, 4.0],
        "A_gb conv1d groups=2 grad_bias",
    );
}

// ---------------------------------------------------------------------------
// TEST B — Conv1d dilation=2 COMBINED WITH padding=1.
//
// The builder's dilation test used padding=0 and stride=2. This pins dilation
// interacting with the padded boundary (where dilated taps reach into the pad
// region). `ConvUtils.h:255`: eff_k = dilation*(k-1)+1 = 3; with padding=1 the
// padded length is 9, L_out=(9-3)/1+1=7.
//
// torch driver:
//   w = (torch.arange(1,9).float()*0.1).reshape(2,2,2)
//   x = torch.arange(1,15).float().reshape(1,2,7).requires_grad_(True)
//   y = F.conv1d(x, w, None, stride=1, padding=1, dilation=2, groups=1)
//   y.sum().backward()
// ---------------------------------------------------------------------------
#[test]
fn divergence_1600_conv1d_dilation2_with_padding_matches_torch() {
    let weight: Vec<f32> = (1..=8).map(|i| i as f32 * 0.1).collect();
    let mut conv = Conv1d::<f32>::new_full(2, 2, 2, 1, 1, 2, 1, false).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&weight, &[2, 2, 2]));
    }

    let x_data: Vec<f32> = (1..=14).map(|i| i as f32).collect();
    let x = leaf(&x_data, &[1, 2, 7]);
    let y = Module::<f32>::forward(&conv, &x).unwrap();
    assert_eq!(y.shape(), &[1, 2, 7]);

    // torch B_fwd.
    let b_fwd = [
        4.0f32, 7.1, 8.1, 9.1, 10.1, 11.1, 4.5, 8.4, 15.9, 18.5, 21.1, 23.7, 26.3, 12.1,
    ];
    assert_close(y.data().unwrap(), &b_fwd, "B_fwd conv1d dilation2+padding1");

    let grad_output = t(&[1.0f32; 14], &[1, 2, 7]);
    let grads = Module::<f32>::forward(&conv, &x)
        .unwrap()
        .grad_fn()
        .unwrap()
        .backward(&grad_output)
        .unwrap();

    // torch B_gx.
    let b_gx = [
        0.6f32, 1.4, 1.4, 1.4, 1.4, 1.4, 0.8, 1.0, 2.2, 2.2, 2.2, 2.2, 2.2, 1.2,
    ];
    assert_close(
        grads[0].as_ref().unwrap().data().unwrap(),
        &b_gx,
        "B_gx conv1d dilation2+padding1 grad_input",
    );

    // torch B_gw [2,2,2].
    assert_eq!(grads[1].as_ref().unwrap().shape(), &[2, 2, 2]);
    let b_gw = [21.0f32, 27.0, 63.0, 69.0, 21.0, 27.0, 63.0, 69.0];
    assert_close(
        grads[1].as_ref().unwrap().data().unwrap(),
        &b_gw,
        "B_gw conv1d dilation2+padding1 grad_weight",
    );
}

// ---------------------------------------------------------------------------
// TEST F — Conv3d groups=2, ASYMMETRIC kernel (kD,kH,kW)=(1,2,3), ASYMMETRIC
// dilation (dD,dH,dW)=(1,2,1), non-cubic volume D=3,H=5,W=4.
//
// This is the hardest #1601 case the builder flagged (5-D col2im row-stride
// bookkeeping). The asymmetry makes a D<->H<->W axis swap detectable: a swap
// would change the OUTPUT SHAPE [1,4,3,3,2] AND the values. groups=2 makes
// in_per_group=1, out_per_group=2 — a per-group channel mis-route changes the
// output too. eff: kD=1, eff_kH=2*(2-1)+1=3, eff_kW=1*(3-1)+1=3, so
// D_out=(3-1)/1+1=3, H_out=(5-3)/1+1=3, W_out=(4-3)/1+1=2.
//
// torch driver:
//   w = (torch.arange(1, 4*1*1*2*3+1).float()*0.01).reshape(4,1,1,2,3)
//   b = torch.tensor([0.1,-0.1,0.2,-0.2])
//   x = torch.arange(1, 2*3*5*4+1).float().reshape(1,2,3,5,4).requires_grad_(True)
//   y = F.conv3d(x, w, b, stride=(1,1,1), padding=(0,0,0),
//                dilation=(1,2,1), groups=2)
//   y.sum().backward()
// ---------------------------------------------------------------------------
#[test]
fn divergence_1601_conv3d_groups2_asymmetric_kernel_dilation_matches_torch() {
    // weight [4,1,1,2,3] = arange(1..=24)*0.01.
    let weight: Vec<f32> = (1..=24).map(|i| i as f32 * 0.01).collect();
    let bias = [0.1f32, -0.1, 0.2, -0.2];

    let mut conv =
        Conv3d::<f32>::new_full(2, 4, (1, 2, 3), (1, 1, 1), (0, 0, 0), (1, 2, 1), 2, true).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&weight, &[4, 1, 1, 2, 3]));
        params[1].set_data(t(&bias, &[4]));
    }

    let x_data: Vec<f32> = (1..=120).map(|i| i as f32).collect();
    let x = leaf(&x_data, &[1, 2, 3, 5, 4]);
    let y = Module::<f32>::forward(&conv, &x).unwrap();
    // An axis swap would not even produce this shape.
    assert_eq!(y.shape(), &[1, 4, 3, 3, 2]);

    // torch F_fwd (72 elements).
    #[rustfmt::skip]
    let f_fwd: [f32; 72] = [
        1.76, 1.97, 2.6, 2.81, 3.44, 3.65, 5.96, 6.17, 6.8, 7.01, 7.64, 7.85, 10.16, 10.37, 11.0,
        11.21, 11.84, 12.05, 3.72, 4.29, 6.0, 6.57, 8.28, 8.85, 15.12, 15.69, 17.4, 17.97, 19.68,
        20.25, 26.52, 27.09, 28.8, 29.37, 31.08, 31.65, 61.98, 62.91, 65.7, 66.63, 69.42, 70.35,
        80.58, 81.51, 84.3, 85.23, 88.02, 88.95, 99.18, 100.11, 102.9, 103.83, 106.62, 107.55,
        85.34, 86.63, 90.5, 91.79, 95.66, 96.95, 111.14, 112.43, 116.3, 117.59, 121.46, 122.75,
        136.94, 138.23, 142.1, 143.39, 147.26, 148.55,
    ];
    assert_close(y.data().unwrap(), &f_fwd, "F_fwd conv3d groups2 asym kernel+dilation");

    let grad_output = t(&[1.0f32; 72], &[1, 4, 3, 3, 2]);
    let grads = Module::<f32>::forward(&conv, &x)
        .unwrap()
        .grad_fn()
        .unwrap()
        .backward(&grad_output)
        .unwrap();

    // torch F_gx (120 elements). An asymmetric col2im D/H/W swap or a per-group
    // channel mis-route changes these.
    #[rustfmt::skip]
    let f_gx: [f32; 120] = [
        0.08, 0.18, 0.22, 0.12, 0.08, 0.18, 0.22, 0.12, 0.22, 0.48, 0.56, 0.3, 0.14, 0.3, 0.34,
        0.18, 0.14, 0.3, 0.34, 0.18, 0.08, 0.18, 0.22, 0.12, 0.08, 0.18, 0.22, 0.12, 0.22, 0.48,
        0.56, 0.3, 0.14, 0.3, 0.34, 0.18, 0.14, 0.3, 0.34, 0.18, 0.08, 0.18, 0.22, 0.12, 0.08,
        0.18, 0.22, 0.12, 0.22, 0.48, 0.56, 0.3, 0.14, 0.3, 0.34, 0.18, 0.14, 0.3, 0.34, 0.18,
        0.32, 0.66, 0.7, 0.36, 0.32, 0.66, 0.7, 0.36, 0.7, 1.44, 1.52, 0.78, 0.38, 0.78, 0.82,
        0.42, 0.38, 0.78, 0.82, 0.42, 0.32, 0.66, 0.7, 0.36, 0.32, 0.66, 0.7, 0.36, 0.7, 1.44,
        1.52, 0.78, 0.38, 0.78, 0.82, 0.42, 0.38, 0.78, 0.82, 0.42, 0.32, 0.66, 0.7, 0.36, 0.32,
        0.66, 0.7, 0.36, 0.7, 1.44, 1.52, 0.78, 0.38, 0.78, 0.82, 0.42, 0.38, 0.78, 0.82, 0.42,
    ];
    assert_close(
        grads[0].as_ref().unwrap().data().unwrap(),
        &f_gx,
        "F_gx conv3d groups2 asym grad_input",
    );

    // torch F_gw [4,1,1,2,3].
    assert_eq!(grads[1].as_ref().unwrap().shape(), &[4, 1, 1, 2, 3]);
    #[rustfmt::skip]
    let f_gw: [f32; 24] = [
        459.0, 477.0, 495.0, 603.0, 621.0, 639.0, 459.0, 477.0, 495.0, 603.0, 621.0, 639.0,
        1539.0, 1557.0, 1575.0, 1683.0, 1701.0, 1719.0, 1539.0, 1557.0, 1575.0, 1683.0, 1701.0,
        1719.0,
    ];
    assert_close(
        grads[1].as_ref().unwrap().data().unwrap(),
        &f_gw,
        "F_gw conv3d groups2 asym grad_weight",
    );

    // torch F_gb.
    assert_close(
        grads[2].as_ref().unwrap().data().unwrap(),
        &[18.0, 18.0, 18.0, 18.0],
        "F_gb conv3d groups2 asym grad_bias",
    );
}
