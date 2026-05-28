//! ACToR discriminator re-audit of commit `7acfe4bf5` (#1600/#1601),
//! part 2 — dense-path regression + the trickiest bookkeeping combo
//! (non-uniform stride AND non-uniform dilation AND groups, with distinct
//! per-group weight magnitudes).
//!
//! Every EXPECTED value is the LIVE torch 2.11.0+cu130 output of the matching
//! `torch.nn.functional.conv{1,3}d` call (driver inline above each block),
//! NOT copied from ferrotorch (R-CHAR-3).
//!
//! Goals beyond part 1:
//!   - REGRESSION: after the groups/dilation refactor, the dense path
//!     (groups=1, dilation=1) must still match torch — Conv1d with stride=2
//!     and Conv3d with an asymmetric kernel both exercise the common case the
//!     refactor could silently break.
//!   - HARDEST col2im case: Conv3d groups=2 with stride=(1,2,1) AND
//!     dilation=(2,1,1) (both non-uniform on DIFFERENT axes) + per-group
//!     weights differing by ~1000x. If stride/dilation get conflated, or the
//!     per-group channel routing leaks, grad_input diverges by orders of
//!     magnitude.

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::{Conv1d, Conv3d};

fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

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
// TEST G — Conv1d DENSE regression (groups=1, dilation=1), stride=2, with bias.
//
// torch driver:
//   w = (torch.arange(1,13).float()*0.1).reshape(2,2,3)
//   b = torch.tensor([0.3,-0.3])
//   x = torch.arange(1,17).float().reshape(1,2,8).requires_grad_(True)
//   y = F.conv1d(x, w, b, stride=2, padding=0, dilation=1, groups=1)
//   y.sum().backward()
// ---------------------------------------------------------------------------
#[test]
fn divergence_1600_conv1d_dense_stride2_regression_matches_torch() {
    let weight: Vec<f32> = (1..=12).map(|i| i as f32 * 0.1).collect();
    let bias = [0.3f32, -0.3];
    let mut conv = Conv1d::<f32>::new_full(2, 2, 3, 2, 0, 1, 1, true).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&weight, &[2, 2, 3]));
        params[1].set_data(t(&bias, &[2]));
    }

    let x_data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let x = leaf(&x_data, &[1, 2, 8]);
    let y = Module::<f32>::forward(&conv, &x).unwrap();
    assert_eq!(y.shape(), &[1, 2, 3]);

    assert_close(
        y.data().unwrap(),
        &[16.9, 21.1, 25.3, 37.9, 49.3, 60.7],
        "G_fwd conv1d dense stride2",
    );

    let grad_output = t(&[1.0f32; 6], &[1, 2, 3]);
    let grads = Module::<f32>::forward(&conv, &x)
        .unwrap()
        .grad_fn()
        .unwrap()
        .backward(&grad_output)
        .unwrap();

    assert_close(
        grads[0].as_ref().unwrap().data().unwrap(),
        &[
            0.8, 1.0, 2.0, 1.0, 2.0, 1.0, 1.2, 0.0, 1.4, 1.6, 3.2, 1.6, 3.2, 1.6, 1.8, 0.0,
        ],
        "G_gx conv1d dense stride2 grad_input",
    );
    assert_eq!(grads[1].as_ref().unwrap().shape(), &[2, 2, 3]);
    assert_close(
        grads[1].as_ref().unwrap().data().unwrap(),
        &[
            9.0, 12.0, 15.0, 33.0, 36.0, 39.0, 9.0, 12.0, 15.0, 33.0, 36.0, 39.0,
        ],
        "G_gw conv1d dense stride2 grad_weight",
    );
    assert_close(
        grads[2].as_ref().unwrap().data().unwrap(),
        &[3.0, 3.0],
        "G_gb conv1d dense stride2 grad_bias",
    );
}

// ---------------------------------------------------------------------------
// TEST H — Conv3d DENSE regression (groups=1, dilation=1), asymmetric kernel
// (kD,kH,kW)=(2,1,2), with bias. Verifies the refactor didn't break the dense
// 3-D path for a non-cubic kernel.
//
// torch driver:
//   w = (torch.arange(1,9).float()*0.05).reshape(2,1,2,1,2)
//   b = torch.tensor([1.0,-1.0])
//   x = torch.arange(1,19).float().reshape(1,1,3,2,3).requires_grad_(True)
//   y = F.conv3d(x, w, b, stride=(1,1,1), padding=(0,0,0),
//                dilation=(1,1,1), groups=1)
//   y.sum().backward()
// ---------------------------------------------------------------------------
#[test]
fn divergence_1601_conv3d_dense_asymmetric_kernel_regression_matches_torch() {
    let weight: Vec<f32> = (1..=8).map(|i| i as f32 * 0.05).collect();
    let bias = [1.0f32, -1.0];
    let mut conv =
        Conv3d::<f32>::new_full(1, 2, (2, 1, 2), (1, 1, 1), (0, 0, 0), (1, 1, 1), 1, true).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&weight, &[2, 1, 2, 1, 2]));
        params[1].set_data(t(&bias, &[2]));
    }

    let x_data: Vec<f32> = (1..=18).map(|i| i as f32).collect();
    let x = leaf(&x_data, &[1, 1, 3, 2, 3]);
    let y = Module::<f32>::forward(&conv, &x).unwrap();
    assert_eq!(y.shape(), &[1, 2, 2, 2, 2]);

    #[rustfmt::skip]
    let h_fwd: [f32; 16] = [
        3.9, 4.4, 5.4, 5.9, 6.9, 7.4, 8.4, 8.9, 5.5, 6.8, 9.4, 10.7, 13.3, 14.6, 17.2, 18.5,
    ];
    assert_close(y.data().unwrap(), &h_fwd, "H_fwd conv3d dense asym kernel");

    let grad_output = t(&[1.0f32; 16], &[1, 2, 2, 2, 2]);
    let grads = Module::<f32>::forward(&conv, &x)
        .unwrap()
        .grad_fn()
        .unwrap()
        .backward(&grad_output)
        .unwrap();

    #[rustfmt::skip]
    let h_gx: [f32; 18] = [
        0.3, 0.7, 0.4, 0.3, 0.7, 0.4, 0.8, 1.8, 1.0, 0.8, 1.8, 1.0, 0.5, 1.1, 0.6, 0.5, 1.1, 0.6,
    ];
    assert_close(
        grads[0].as_ref().unwrap().data().unwrap(),
        &h_gx,
        "H_gx conv3d dense asym grad_input",
    );
    assert_eq!(grads[1].as_ref().unwrap().shape(), &[2, 1, 2, 1, 2]);
    assert_close(
        grads[1].as_ref().unwrap().data().unwrap(),
        &[48.0, 56.0, 96.0, 104.0, 48.0, 56.0, 96.0, 104.0],
        "H_gw conv3d dense asym grad_weight",
    );
    assert_close(
        grads[2].as_ref().unwrap().data().unwrap(),
        &[8.0, 8.0],
        "H_gb conv3d dense asym grad_bias",
    );
}

// ---------------------------------------------------------------------------
// TEST I — Conv3d groups=2 with NON-UNIFORM stride (1,2,1) AND NON-UNIFORM
// dilation (2,1,1) (each non-uniform on a DIFFERENT axis) + DISTINCT per-group
// weight magnitudes (group1 ~1000x group0). The hardest col2im bookkeeping:
// stride and dilation must NOT be conflated, AND the per-group channel routing
// must hold. eff_kD=2*(2-1)+1=3 -> D_out=(5-3)/1+1=3; H stride=2 ->
// H_out=(5-2)/2+1=2; W_out=(4-2)/1+1=3 -> output [1,2,3,2,3].
//
// torch driver:
//   w0 = torch.arange(1,9).float()*0.01
//   w1 = torch.arange(1,9).float()*10.0
//   w  = torch.cat([w0,w1]).reshape(2,1,2,2,2)
//   x  = torch.arange(1,201).float().reshape(1,2,5,5,4).requires_grad_(True)
//   y  = F.conv3d(x, w, None, stride=(1,2,1), padding=(0,0,0),
//                 dilation=(2,1,1), groups=2)
//   y.sum().backward()
// ---------------------------------------------------------------------------
#[test]
fn divergence_1601_conv3d_groups2_nonuniform_stride_dilation_matches_torch() {
    let mut weight: Vec<f32> = (1..=8).map(|i| i as f32 * 0.01).collect();
    weight.extend((1..=8).map(|i| i as f32 * 10.0));
    let mut conv =
        Conv3d::<f32>::new_full(2, 2, (2, 2, 2), (1, 2, 1), (0, 0, 0), (2, 1, 1), 2, false)
            .unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&weight, &[2, 1, 2, 2, 2]));
    }

    // Input is the full [1,2,5,5,4] = 200-element volume: arange(1..=200).
    let x_data: Vec<f32> = (1..=200).map(|i| i as f32).collect();
    let x = leaf(&x_data, &[1, 2, 5, 5, 4]);
    let y = Module::<f32>::forward(&conv, &x).unwrap();
    assert_eq!(y.shape(), &[1, 2, 3, 2, 3]);

    // torch I_fwd (36 elements): group1 channels (>=47840) >> group0 (<=30).
    #[rustfmt::skip]
    let i_fwd: [f32; 36] = [
        11.84, 12.2, 12.56, 14.72, 15.08, 15.44, 19.04, 19.4, 19.76, 21.92, 22.28, 22.64, 26.24,
        26.6, 26.96, 29.12, 29.48, 29.84, 47840.0, 48200.0, 48560.0, 50720.0, 51080.0, 51440.0,
        55040.0, 55400.0, 55760.0, 57920.0, 58280.0, 58640.0, 62240.0, 62600.0, 62960.0, 65120.0,
        65480.0, 65840.0,
    ];
    assert_close(
        y.data().unwrap(),
        &i_fwd,
        "I_fwd conv3d groups2 nonuniform stride+dilation",
    );

    let grad_output = t(&[1.0f32; 36], &[1, 2, 3, 2, 3]);
    let grads = Module::<f32>::forward(&conv, &x)
        .unwrap()
        .grad_fn()
        .unwrap()
        .backward(&grad_output)
        .unwrap();

    // torch I_gx (200 elements = full [1,2,5,5,4] volume). group0 small
    // (<=0.22), group1 huge (>=10). The 0.0 entries are the stride/dilation
    // gaps the kernel never visits.
    #[rustfmt::skip]
    let i_gx: [f32; 200] = [
        0.01, 0.03, 0.03, 0.02, 0.03, 0.07, 0.07, 0.04, 0.01, 0.03, 0.03, 0.02, 0.03, 0.07, 0.07,
        0.04, 0.0, 0.0, 0.0, 0.0, 0.01, 0.03, 0.03, 0.02, 0.03, 0.07, 0.07, 0.04, 0.01, 0.03, 0.03,
        0.02, 0.03, 0.07, 0.07, 0.04, 0.0, 0.0, 0.0, 0.0, 0.06, 0.14, 0.14, 0.08, 0.1, 0.22, 0.22,
        0.12, 0.06, 0.14, 0.14, 0.08, 0.1, 0.22, 0.22, 0.12, 0.0, 0.0, 0.0, 0.0, 0.05, 0.11, 0.11,
        0.06, 0.07, 0.15, 0.15, 0.08, 0.05, 0.11, 0.11, 0.06, 0.07, 0.15, 0.15, 0.08, 0.0, 0.0,
        0.0, 0.0, 0.05, 0.11, 0.11, 0.06, 0.07, 0.15, 0.15, 0.08, 0.05, 0.11, 0.11, 0.06, 0.07,
        0.15, 0.15, 0.08, 0.0, 0.0, 0.0, 0.0, 10.0, 30.0, 30.0, 20.0, 30.0, 70.0, 70.0, 40.0, 10.0,
        30.0, 30.0, 20.0, 30.0, 70.0, 70.0, 40.0, 0.0, 0.0, 0.0, 0.0, 10.0, 30.0, 30.0, 20.0, 30.0,
        70.0, 70.0, 40.0, 10.0, 30.0, 30.0, 20.0, 30.0, 70.0, 70.0, 40.0, 0.0, 0.0, 0.0, 0.0, 60.0,
        140.0, 140.0, 80.0, 100.0, 220.0, 220.0, 120.0, 60.0, 140.0, 140.0, 80.0, 100.0, 220.0,
        220.0, 120.0, 0.0, 0.0, 0.0, 0.0, 50.0, 110.0, 110.0, 60.0, 70.0, 150.0, 150.0, 80.0, 50.0,
        110.0, 110.0, 60.0, 70.0, 150.0, 150.0, 80.0, 0.0, 0.0, 0.0, 0.0, 50.0, 110.0, 110.0, 60.0,
        70.0, 150.0, 150.0, 80.0, 50.0, 110.0, 110.0, 60.0, 70.0, 150.0, 150.0, 80.0, 0.0, 0.0,
        0.0, 0.0,
    ];
    assert_eq!(grads[0].as_ref().unwrap().data().unwrap().len(), 200);
    assert_close(
        grads[0].as_ref().unwrap().data().unwrap(),
        &i_gx,
        "I_gx conv3d groups2 nonuniform stride+dilation grad_input",
    );
    assert_eq!(grads[1].as_ref().unwrap().shape(), &[2, 1, 2, 2, 2]);
    #[rustfmt::skip]
    let i_gw: [f32; 16] = [
        468.0, 486.0, 540.0, 558.0, 1188.0, 1206.0, 1260.0, 1278.0, 2268.0, 2286.0, 2340.0, 2358.0,
        2988.0, 3006.0, 3060.0, 3078.0,
    ];
    assert_close(
        grads[1].as_ref().unwrap().data().unwrap(),
        &i_gw,
        "I_gw conv3d groups2 nonuniform grad_weight",
    );
}
