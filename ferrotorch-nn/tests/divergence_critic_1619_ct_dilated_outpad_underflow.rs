//! Adversarial discriminator re-audit of commit 00aff68a9d
//! (ConvTranspose{1,2,3}d dilated-forward usize-underflow fix; refs #1619).
//!
//! The #1619 bug: the transposed-conv forward helpers reformulate the dilated
//! transposed conv as `stride-insert -> internal stride-1 dilated conv` with
//! `internal_pad = dilation*(k-1) - padding = eff_k - 1 - padding` computed in
//! `usize`. When `padding > dilation*(k-1)` the subtraction WRAPPED to
//! `usize::MAX`, the bounds check rejected every scatter position, and the
//! output collapsed to bias-only in the trailing region. The fix recomputes the
//! pad as `isize`: negative -> CROP the upsampled signal (new `crop_volume_3d` /
//! `crop_plane_2d`); non-negative -> zero-pad as before.
//!
//! The prior #1608 critic tested `dilation=(2,2,2)` but NEVER combined
//! `output_padding>0` with dilation in the underflow regime. This file does:
//!   - the underflow regime `eff_k-1 < padding` for ct1d/ct2d/ct3d, forward AND
//!     backward, INCLUDING kernel=2/dilation=1/padding=2 (eff_k-1=1 < 2);
//!   - the #1619 sample (ct3d g=2, dilation=(2,3,2), output_padding=1, stride=2,
//!     padding=1) AND a distinct-per-group-weight variant where any cross-group
//!     leak is value-detectable;
//!   - the crop path itself with asymmetric per-axis (one axis crops, another
//!     zero-pads) for ct2d/ct3d — a wrong crop offset shifts the output;
//!   - ct1d/ct2d direct (the fixer claims they share the latent bug);
//!   - a dense regression (dilation=1, output_padding=0, groups=1) so the signed
//!     refactor did not perturb the common path.
//!
//! EVERY `expected` value below was produced by the LIVE PyTorch 2.11.0+cu130
//! oracle (R-CHAR-3): `torch.nn.functional.conv_transpose{1,2,3}d(...)` forward
//! output plus `x.grad / weight.grad / bias.grad` after backward with an
//! all-ones grad_output, on the identical deterministic inputs constructed
//! below. Nothing is copied from ferrotorch. grad_weight is asserted in torch's
//! transposed `[in, out/groups, *k]` layout.
//!
//! Driving the grouped/dilated transposed path with controlled weights requires
//! `new_full` + `Module::parameters_mut()` + `Parameter::set_data` to overwrite
//! the Kaiming-random weights. params[0]=weight, params[1]=bias.
//!
//! Upstream: pytorch aten/src/ATen/native/vol2col.h:80-106 (col2vol scatter
//! `t_pad = t*dT - pT + t_offset*dilationT`, bounded by output extent),
//! aten/src/ATen/native/NaiveConvolutionTranspose3d.cpp:131-139 (output extent).

use ferrotorch_core::Tensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_nn::{ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, Module};

fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn assert_close(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{ctx}: length mismatch got={} want={}",
        got.len(),
        want.len()
    );
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = 1e-3_f32 + 1e-3_f32 * w.abs();
        assert!(
            (g - w).abs() <= tol,
            "{ctx}: element {i} ferrotorch={g} torch={w} |diff|={} > tol={tol}\n full ferro={got:?}",
            (g - w).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// CASE A — ct1d UNDERFLOW REGIME: kernel=1 (eff_k=1), dilation=2, padding=1.
// internal_pad = eff_k-1-padding = 0-1 = -1 (NEGATIVE) -> crop path.
//
// torch:
//   x = arange(1,5).float().reshape(1,1,4)
//   w = (arange(1,2).float()*0.1).reshape(1,1,1)  # [0.1]
//   b = tensor([0.5])
//   y = F.conv_transpose1d(x,w,b,stride=1,padding=1,output_padding=0,
//                          groups=1,dilation=2)
//   -> y_shape [1,1,2], y=[0.7,0.8]
// ---------------------------------------------------------------------------
#[test]
fn divergence_ct1d_k1_d2_p1_underflow_matches_torch() {
    let weight = [0.1f32];
    let bias = [0.5f32];
    let mut ct = ConvTranspose1d::<f32>::new_full(1, 1, 1, 1, 1, 0, 2, 1, true).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut ct);
        params[0].set_data(t(&weight, &[1, 1, 1]));
        params[1].set_data(t(&bias, &[1]));
    }
    let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let y = Module::<f32>::forward(&ct, &x).unwrap();
    assert_eq!(y.shape(), &[1, 1, 2], "A ct1d k1 d2 p1 shape");
    assert_close(y.data().unwrap(), &[0.7, 0.8], "A ct1d k1 d2 p1 forward");

    let grads = Module::<f32>::forward(&ct, &x)
        .unwrap()
        .grad_fn()
        .unwrap()
        .backward(&t(&[1.0; 2], &[1, 1, 2]))
        .unwrap();
    assert_close(
        grads[0].as_ref().unwrap().data().unwrap(),
        &[0.0, 0.1, 0.1, 0.0],
        "A ct1d grad_input",
    );
    assert_close(
        grads[1].as_ref().unwrap().data().unwrap(),
        &[5.0],
        "A ct1d grad_weight",
    );
    assert_close(
        grads[2].as_ref().unwrap().data().unwrap(),
        &[2.0],
        "A ct1d grad_bias",
    );
}

// ---------------------------------------------------------------------------
// CASE B — ct1d kernel=2, dilation=1, padding=2 (eff_k-1 = 1 < 2 -> -1 crop).
//
//   x = arange(1,6).float().reshape(1,1,5)
//   w = (arange(1,3).float()*0.1).reshape(1,1,2)  # [0.1,0.2]
//   b = tensor([0.5])
//   y = F.conv_transpose1d(x,w,b,stride=1,padding=2,dilation=1)
//   -> y_shape [1,1,2], y=[1.2,1.5]
// ---------------------------------------------------------------------------
#[test]
fn divergence_ct1d_k2_d1_p2_underflow_matches_torch() {
    let weight = [0.1f32, 0.2];
    let bias = [0.5f32];
    let mut ct = ConvTranspose1d::<f32>::new_full(1, 1, 2, 1, 2, 0, 1, 1, true).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut ct);
        params[0].set_data(t(&weight, &[1, 1, 2]));
        params[1].set_data(t(&bias, &[1]));
    }
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
    let y = Module::<f32>::forward(&ct, &x).unwrap();
    assert_eq!(y.shape(), &[1, 1, 2], "B ct1d k2 d1 p2 shape");
    assert_close(y.data().unwrap(), &[1.2, 1.5], "B ct1d k2 d1 p2 forward");

    let grads = Module::<f32>::forward(&ct, &x)
        .unwrap()
        .grad_fn()
        .unwrap()
        .backward(&t(&[1.0; 2], &[1, 1, 2]))
        .unwrap();
    assert_close(
        grads[0].as_ref().unwrap().data().unwrap(),
        &[0.0, 0.2, 0.3, 0.1, 0.0],
        "B ct1d grad_input",
    );
    assert_close(
        grads[1].as_ref().unwrap().data().unwrap(),
        &[7.0, 5.0],
        "B ct1d grad_weight",
    );
    assert_close(
        grads[2].as_ref().unwrap().data().unwrap(),
        &[2.0],
        "B ct1d grad_bias",
    );
}

// ---------------------------------------------------------------------------
// CASE C — ct1d underflow + output_padding + stride: kernel=1, dilation=2,
// padding=1, stride=2, output_padding=1 (the crop path must coexist with the
// stride-insert AND the output_padding-extended trailing region).
//
//   y = F.conv_transpose1d(x,w,b,stride=2,padding=1,output_padding=1,dilation=2)
//   -> y_shape [1,1,6], y=[0.5,0.7,0.5,0.8,0.5,0.9]  (bias 0.5 between taps)
// ---------------------------------------------------------------------------
#[test]
fn divergence_ct1d_k1_d2_p1_outpad_stride_matches_torch() {
    let weight = [0.1f32];
    let bias = [0.5f32];
    let mut ct = ConvTranspose1d::<f32>::new_full(1, 1, 1, 2, 1, 1, 2, 1, true).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut ct);
        params[0].set_data(t(&weight, &[1, 1, 1]));
        params[1].set_data(t(&bias, &[1]));
    }
    let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let y = Module::<f32>::forward(&ct, &x).unwrap();
    assert_eq!(y.shape(), &[1, 1, 6], "C ct1d outpad shape");
    assert_close(
        y.data().unwrap(),
        &[0.5, 0.7, 0.5, 0.8, 0.5, 0.9],
        "C ct1d k1 d2 p1 outpad stride forward",
    );

    let grads = Module::<f32>::forward(&ct, &x)
        .unwrap()
        .grad_fn()
        .unwrap()
        .backward(&t(&[1.0; 6], &[1, 1, 6]))
        .unwrap();
    assert_close(
        grads[0].as_ref().unwrap().data().unwrap(),
        &[0.0, 0.1, 0.1, 0.1],
        "C ct1d grad_input",
    );
    assert_close(
        grads[1].as_ref().unwrap().data().unwrap(),
        &[9.0],
        "C ct1d grad_weight",
    );
    assert_close(
        grads[2].as_ref().unwrap().data().unwrap(),
        &[6.0],
        "C ct1d grad_bias",
    );
}

// ---------------------------------------------------------------------------
// CASE D — ct2d underflow analog: kernel=(1,1), dilation=(2,2), padding=(1,1).
// Both axes crop (internal pad -1 each).
//
//   x = arange(1,10).float().reshape(1,1,3,3)
//   w = tensor([0.1]).reshape(1,1,1,1) ; b = tensor([0.5])
//   y = F.conv_transpose2d(x,w,b,stride=1,padding=1,dilation=2)
//   -> y_shape [1,1,1,1], y=[1.0]   (only the center tap survives)
// ---------------------------------------------------------------------------
#[test]
fn divergence_ct2d_k1_d2_p1_underflow_matches_torch() {
    let weight = [0.1f32];
    let bias = [0.5f32];
    let mut ct =
        ConvTranspose2d::<f32>::new_full(1, 1, (1, 1), (1, 1), (1, 1), (0, 0), (2, 2), 1, true)
            .unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut ct);
        params[0].set_data(t(&weight, &[1, 1, 1, 1]));
        params[1].set_data(t(&bias, &[1]));
    }
    let x = leaf(
        &(1..=9).map(|i| i as f32).collect::<Vec<_>>(),
        &[1, 1, 3, 3],
    );
    let y = Module::<f32>::forward(&ct, &x).unwrap();
    assert_eq!(y.shape(), &[1, 1, 1, 1], "D ct2d k1 d2 p1 shape");
    assert_close(y.data().unwrap(), &[1.0], "D ct2d k1 d2 p1 forward");

    let grads = Module::<f32>::forward(&ct, &x)
        .unwrap()
        .grad_fn()
        .unwrap()
        .backward(&t(&[1.0; 1], &[1, 1, 1, 1]))
        .unwrap();
    assert_close(
        grads[0].as_ref().unwrap().data().unwrap(),
        &[0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
        "D ct2d grad_input (only center survives)",
    );
    assert_close(
        grads[1].as_ref().unwrap().data().unwrap(),
        &[5.0],
        "D ct2d grad_weight",
    );
    assert_close(
        grads[2].as_ref().unwrap().data().unwrap(),
        &[1.0],
        "D ct2d grad_bias",
    );
}

// ---------------------------------------------------------------------------
// CASE E — ct2d ASYMMETRIC crop/pad: kw=1,dw=2,pw=1 -> CROP w; kh=3,dh=1,ph=0
// -> eff_kh=3 internal pad 2 ZERO-PAD h. A wrong crop offset (cropping the
// wrong end of the width axis) shifts the output; this is the crop-offset
// correctness probe.
//
//   x = arange(1,13).float().reshape(1,1,3,4)
//   w = (arange(1,4).float()*0.1).reshape(1,1,3,1)  # [0.1,0.2,0.3]
//   b = tensor([0.5])
//   y = F.conv_transpose2d(x,w,b,stride=1,padding=(0,1),dilation=(1,2))
//   -> y_shape [1,1,5,2]
// ---------------------------------------------------------------------------
#[test]
fn divergence_ct2d_asym_crop_w_pad_h_matches_torch() {
    let weight = [0.1f32, 0.2, 0.3];
    let bias = [0.5f32];
    let mut ct =
        ConvTranspose2d::<f32>::new_full(1, 1, (3, 1), (1, 1), (0, 1), (0, 0), (1, 2), 1, true)
            .unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut ct);
        params[0].set_data(t(&weight, &[1, 1, 3, 1]));
        params[1].set_data(t(&bias, &[1]));
    }
    let x = leaf(
        &(1..=12).map(|i| i as f32).collect::<Vec<_>>(),
        &[1, 1, 3, 4],
    );
    let y = Module::<f32>::forward(&ct, &x).unwrap();
    assert_eq!(y.shape(), &[1, 1, 5, 2], "E ct2d asym shape");
    assert_close(
        y.data().unwrap(),
        &[0.7, 0.8, 1.5, 1.8, 3.3, 3.9, 4.3, 4.8, 3.5, 3.8],
        "E ct2d asym crop-w pad-h forward",
    );

    let grads = Module::<f32>::forward(&ct, &x)
        .unwrap()
        .grad_fn()
        .unwrap()
        .backward(&t(&[1.0; 10], &[1, 1, 5, 2]))
        .unwrap();
    assert_close(
        grads[0].as_ref().unwrap().data().unwrap(),
        &[0.0, 0.6, 0.6, 0.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.6, 0.6, 0.0],
        "E ct2d grad_input",
    );
    assert_close(
        grads[1].as_ref().unwrap().data().unwrap(),
        &[39.0, 39.0, 39.0],
        "E ct2d grad_weight",
    );
    assert_close(
        grads[2].as_ref().unwrap().data().unwrap(),
        &[10.0],
        "E ct2d grad_bias",
    );
}

// ---------------------------------------------------------------------------
// CASE F — ct2d combo: groups=2 (distinct weights), output_padding=(1,0),
// dilation=(2,2), stride=(2,2), padding=(1,1), kernel=(1,2).
//   kh=1,dh=2,ph=1 -> eff=1 internal -1 CROP h ; kw=2,dw=2,pw=1 -> eff=3 pad1 w.
// output_padding+dilation+groups TOGETHER with one axis cropping.
//
//   x = arange(1,19).float().reshape(1,2,3,3)
//   w = (arange(1,5).float()*0.1).reshape(2,1,1,2)  # distinct per group
//   b = tensor([0.3,-0.3])
//   y = F.conv_transpose2d(x,w,b,stride=2,padding=1,output_padding=(1,0),
//                          groups=2,dilation=2)
//   -> y_shape [1,2,4,5]
// ---------------------------------------------------------------------------
#[test]
fn divergence_ct2d_g2_outpad_dilation_crop_matches_torch() {
    let weight: Vec<f32> = (1..=4).map(|i| i as f32 * 0.1).collect();
    let bias = [0.3f32, -0.3];
    let mut ct =
        ConvTranspose2d::<f32>::new_full(2, 2, (1, 2), (2, 2), (1, 1), (1, 0), (2, 2), 2, true)
            .unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut ct);
        params[0].set_data(t(&weight, &[2, 1, 1, 2]));
        params[1].set_data(t(&bias, &[2]));
    }
    let x = leaf(
        &(1..=18).map(|i| i as f32).collect::<Vec<_>>(),
        &[1, 2, 3, 3],
    );
    let y = Module::<f32>::forward(&ct, &x).unwrap();
    assert_eq!(y.shape(), &[1, 2, 4, 5], "F ct2d combo shape");
    assert_close(
        y.data().unwrap(),
        &[
            0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 1.6, 0.3, 1.9, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 2.5,
            0.3, 2.8, 0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, 9.1, -0.3, 9.8, -0.3, -0.3, -0.3,
            -0.3, -0.3, -0.3, -0.3, 11.2, -0.3, 11.9, -0.3,
        ],
        "F ct2d g2 outpad dilation crop forward",
    );

    let grads = Module::<f32>::forward(&ct, &x)
        .unwrap()
        .grad_fn()
        .unwrap()
        .backward(&t(&[1.0; 40], &[1, 2, 4, 5]))
        .unwrap();
    assert_close(
        grads[0].as_ref().unwrap().data().unwrap(),
        &[
            0.0, 0.0, 0.0, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 0.4, 0.7, 0.3, 0.4, 0.7,
            0.3,
        ],
        "F ct2d grad_input",
    );
    assert_eq!(
        grads[1].as_ref().unwrap().shape(),
        &[2, 1, 1, 2],
        "F grad_weight transposed [in,out/g,kH,kW] layout"
    );
    assert_close(
        grads[1].as_ref().unwrap().data().unwrap(),
        &[28.0, 24.0, 64.0, 60.0],
        "F ct2d grad_weight",
    );
    assert_close(
        grads[2].as_ref().unwrap().data().unwrap(),
        &[20.0, 20.0],
        "F ct2d grad_bias",
    );
}

// ---------------------------------------------------------------------------
// CASE G — the #1619 sample exactly, but with DISTINCT per-group weights and a
// per-group bias: ct3d groups=2, dilation=(2,3,2), output_padding=1, stride=2,
// padding=1, kernel=(2,2,1). kw=1,dw=2,pw=1 -> eff_kw=1 internal -1 CROP w; the
// d and h axes zero-pad. Distinct per-group weights make any cross-group leak
// or [in,out/g] layout swap value-detectable — the strongest form of the #1619
// op_db sample.
//
//   x = arange(1,17).float().reshape(1,2,2,2,2)
//   w = (arange(1,9).float()*0.1).reshape(2,1,2,2,1)  # distinct per group
//   b = tensor([0.3,-0.3])
//   y = F.conv_transpose3d(x,w,b,stride=2,padding=1,output_padding=1,
//                          groups=2,dilation=(2,2,2))
//   -> y_shape [1,2,4,4,2]
// ---------------------------------------------------------------------------
#[test]
fn divergence_ct3d_g2_distinct_outpad_dilation_crop_matches_torch() {
    let weight: Vec<f32> = (1..=8).map(|i| i as f32 * 0.1).collect();
    let bias = [0.3f32, -0.3];
    let mut ct = ConvTranspose3d::<f32>::new_full(
        2,
        2,
        (2, 2, 1),
        (2, 2, 2),
        (1, 1, 1),
        (1, 1, 1),
        (2, 2, 2),
        2,
        true,
    )
    .unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut ct);
        params[0].set_data(t(&weight, &[2, 1, 2, 2, 1]));
        params[1].set_data(t(&bias, &[2]));
    }
    let x = leaf(
        &(1..=16).map(|i| i as f32).collect::<Vec<_>>(),
        &[1, 2, 2, 2, 2],
    );
    let y = Module::<f32>::forward(&ct, &x).unwrap();
    assert_eq!(y.shape(), &[1, 2, 4, 4, 2], "G ct3d g2 distinct shape");
    assert_close(
        y.data().unwrap(),
        &[
            0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 4.3, 0.3, 0.3, 0.3, 3.5, 0.3,
            0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 5.1, 0.3, 0.3, 0.3, 3.5, -0.3, -0.3,
            -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, 32.5, -0.3, -0.3, -0.3, 18.9,
            -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, 22.1, -0.3, -0.3,
            -0.3, 12.5,
        ],
        "G ct3d g2 distinct outpad dilation crop forward",
    );

    let grads = Module::<f32>::forward(&ct, &x)
        .unwrap()
        .grad_fn()
        .unwrap()
        .backward(&t(&[1.0; 64], &[1, 2, 4, 4, 2]))
        .unwrap();
    assert_close(
        grads[0].as_ref().unwrap().data().unwrap(),
        &[
            0.0, 0.4, 0.0, 0.7, 0.0, 0.6, 0.0, 1.0, 0.0, 0.8, 0.0, 1.5, 0.0, 1.4, 0.0, 2.6,
        ],
        "G ct3d grad_input",
    );
    assert_eq!(
        grads[1].as_ref().unwrap().shape(),
        &[2, 1, 2, 2, 1],
        "G grad_weight transposed [in,out/g,kD,kH,kW] layout"
    );
    assert_close(
        grads[1].as_ref().unwrap().data().unwrap(),
        &[8.0, 14.0, 12.0, 20.0, 16.0, 30.0, 28.0, 52.0],
        "G ct3d grad_weight",
    );
    assert_close(
        grads[2].as_ref().unwrap().data().unwrap(),
        &[32.0, 32.0],
        "G ct3d grad_bias",
    );
}

// ---------------------------------------------------------------------------
// CASE H — DENSE REGRESSION: dilation=1, output_padding=0, groups=1. The signed
// refactor must NOT have perturbed the common path.
//
//   x = arange(1,9).float().reshape(1,1,2,2,2)
//   w = (arange(1,9).float()*0.1).reshape(1,1,2,2,2)
//   b = tensor([0.5])
//   y = F.conv_transpose3d(x,w,b,stride=1,padding=0,dilation=1)
//   -> y_shape [1,1,3,3,3]
// ---------------------------------------------------------------------------
#[test]
fn divergence_ct3d_dense_regression_matches_torch() {
    let weight: Vec<f32> = (1..=8).map(|i| i as f32 * 0.1).collect();
    let bias = [0.5f32];
    let mut ct = ConvTranspose3d::<f32>::new_full(
        1,
        1,
        (2, 2, 2),
        (1, 1, 1),
        (0, 0, 0),
        (0, 0, 0),
        (1, 1, 1),
        1,
        true,
    )
    .unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut ct);
        params[0].set_data(t(&weight, &[1, 1, 2, 2, 2]));
        params[1].set_data(t(&bias, &[1]));
    }
    let x = leaf(
        &(1..=8).map(|i| i as f32).collect::<Vec<_>>(),
        &[1, 1, 2, 2, 2],
    );
    let y = Module::<f32>::forward(&ct, &x).unwrap();
    assert_eq!(y.shape(), &[1, 1, 3, 3, 3], "H dense regression shape");
    assert_close(
        y.data().unwrap(),
        &[
            0.6, 0.9, 0.9, 1.1, 2.5, 2.1, 1.4, 2.9, 2.1, 1.5, 3.7, 2.9, 4.9, 12.5, 8.5, 4.7, 10.9,
            6.9, 3.0, 6.5, 4.1, 7.5, 16.9, 10.1, 5.4, 11.7, 6.9,
        ],
        "H ct3d dense regression forward",
    );

    let grads = Module::<f32>::forward(&ct, &x)
        .unwrap()
        .grad_fn()
        .unwrap()
        .backward(&t(&[1.0; 27], &[1, 1, 3, 3, 3]))
        .unwrap();
    assert_close(
        grads[0].as_ref().unwrap().data().unwrap(),
        &[3.6, 3.6, 3.6, 3.6, 3.6, 3.6, 3.6, 3.6],
        "H ct3d grad_input",
    );
    assert_close(
        grads[1].as_ref().unwrap().data().unwrap(),
        &[36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0],
        "H ct3d grad_weight",
    );
    assert_close(
        grads[2].as_ref().unwrap().data().unwrap(),
        &[27.0],
        "H ct3d grad_bias",
    );
}
