//! ACToR discriminator re-audit of commit `3dd7d2cca` (#1441: conv1d/2d/3d +
//! bilinear runner arms converged to 0-skip).
//!
//! The runner's `dispatch_conv` (`tools/parity-sweep/runner/src/main.rs`) now
//! decodes the op_db sample's `groups` / `dilation` / `stride` / `padding`
//! (incl. `'same'`/`'valid'`) and the unbatched `(C,*spatial)` form, then drives
//! the SAME production constructors as this file (`Conv2d::new_full` +
//! `parameters_mut`/`set_data` + `Module::forward`; `Bilinear::new` +
//! `forward_pair`). The 4 sweeps re-run on `3dd7d2cca` at 0-skip/0-failed.
//!
//! The pre-existing `divergence_1441_conv_linear_bilinear_arms.rs` pins only a
//! SYMMETRIC synthetic grouped(=2)/dilated(=2) conv2d and a 2-D bilinear. The
//! op_db samples that the 0-skip jump newly RUNS are the ADVERSARIAL combos a
//! mis-decode or a per-dim (H/W) ordering bug would silently mistest:
//!   * conv2d op_db i=2: groups=2 AND dilation=(4,4) AND stride=(3,2) AND
//!     asymmetric padding=(2,1) simultaneously — a non-square spatial output
//!     (2x1) that a H/W swap would reshape-mismatch.
//!   * its unbatched twin (op_db i=3, input rank D+1) through the implicit-batch
//!     unsqueeze/squeeze path (#1604).
//!   * conv2d op_db i=12: padding='same' with dilation=3 — output spatial MUST
//!     equal input spatial (5x5); a decode that treated 'same' as padding=0 (the
//!     pre-#1602 SameSkip behavior) would shrink the output and shape-mismatch.
//!   * bilinear op_db i=3: 3-D input (#1603 N-D flatten path) the arm formerly
//!     SKIPPED.
//!
//! Every EXPECTED value here is the LIVE torch 2.11.0+cu130 output of the
//! matching `torch.nn.functional.*` call (the SAME backend the parity-sweep
//! oracle routes to), NOT copied from the ferrotorch side (R-CHAR-3). The torch
//! driver script is reproduced inline above each constant block so the value is
//! regenerable. The inputs are deterministic `arange`-derived (NOT op_db's
//! random fill) so the test is self-contained, but the GEOMETRY (shapes,
//! groups, dilation, stride, padding, batched-ness) exactly mirrors the named
//! op_db samples the runner arm now decodes.
//!
//! These tests PASS on `3dd7d2cca` (confirming the 0-skip comparisons test the
//! RIGHT config, not a mis-grouped/mis-dilated stand-in). They are committed as
//! permanent regression coverage: any future regression in the production conv
//! grouped/dilated/'same'/unbatched path, or in the bilinear N-D path, fails
//! here independently of the parity-sweep harness. A mis-grouped conv on the
//! i=2 case diverges by O(1)-O(10) absolute (output absMAX ~20.7), 6 orders of
//! magnitude above the runner's atol=1e-5 / rtol=1e-4 conv envelope, so this is
//! NOT tolerance-masked.

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_nn::Bilinear;
use ferrotorch_nn::Conv2d;
use ferrotorch_nn::StringPadding;
use ferrotorch_nn::module::Module;

fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// Element-wise close check with the runner's conv envelope (rtol=1e-4,
/// atol=1e-5) — the SAME tolerance `tolerance_for("nn.functional.conv2d")`
/// applies in the sweep. A correct grouped/dilated/'same' conv matches torch to
/// ~6 sig figs; a mis-decoded conv diverges by O(1) >> this envelope.
fn assert_close(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = 1e-5_f32 + 1e-4_f32 * w.abs();
        assert!(
            (g - w).abs() <= tol,
            "{ctx}: element {i} ferrotorch={g} torch={w} |diff|={} > tol={tol}",
            (g - w).abs()
        );
    }
}

/// op_db conv2d i=2 GEOMETRY: groups=2, dilation=(4,4), stride=(3,2),
/// padding=(2,1), input (2,4,8,8), weight (2,2,3,3), bias[2]. The arm decodes
/// all four kwargs and drives `Conv2d::new_full(in=4,out=2,(3,3),(3,2),(2,1),
/// (4,4),2,true)`. Pins that the per-dim H/W ordering, the channel partition,
/// and the dilated im2col all match torch on the HARDEST simultaneous combo.
///
/// torch driver:
///   inp = (torch.arange(2*4*8*8).float()*0.01 - 2.0).reshape(2,4,8,8)
///   w   = (torch.arange(2*2*3*3).float()*0.05 - 0.4).reshape(2,2,3,3)
///   b   = torch.tensor([0.3, -0.7])
///   F.conv2d(inp, w, b, stride=(3,2), padding=(2,1), dilation=(4,4), groups=2)
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "oracle-derived f32 expected values copied verbatim from live torch 2.11"
)]
fn divergence_1441_conv2d_grouped_dilated_asym_matches_torch() {
    let input: Vec<f32> = (0..(2 * 4 * 8 * 8))
        .map(|i| i as f32 * 0.01 - 2.0)
        .collect();
    let weight: Vec<f32> = (0..(2 * 2 * 3 * 3))
        .map(|i| i as f32 * 0.05 - 0.4)
        .collect();
    let bias = vec![0.3f32, -0.7];

    // EXPECTED: live torch.nn.functional.conv2d output, [2,2,2,1] flattened.
    let torch_out = [
        -0.3340000510215759_f32,
        1.254000186920166_f32,
        -0.270000696182251_f32,
        -0.7940003871917725_f32,
        2.2259998321533203_f32,
        0.7419999837875366_f32,
        20.72199821472168_f32,
        17.125999450683594_f32,
    ];

    let mut conv = Conv2d::<f32>::new_full(4, 2, (3, 3), (3, 2), (2, 1), (4, 4), 2, true).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&weight, &[2, 2, 3, 3]));
        params[1].set_data(t(&bias, &[2]));
    }
    let y = Module::<f32>::forward(&conv, &t(&input, &[2, 4, 8, 8])).unwrap();
    // A H/W swap or wrong stride/dilation ordering changes this asymmetric shape.
    assert_eq!(y.shape(), &[2, 2, 2, 1]);
    assert_close(
        y.data().unwrap(),
        &torch_out,
        "conv2d groups=2 dilation=(4,4) stride=(3,2) pad=(2,1) vs torch",
    );
}

/// op_db conv2d i=3 GEOMETRY: the UNBATCHED twin of i=2 (input rank D+1 =
/// (4,8,8)). The arm now decodes rank-(D+1) and routes through the
/// `Module::forward` implicit-batch unsqueeze/squeeze (#1604), so the output is
/// rank-(D+1) = (2,2,1). Pins that the unbatched path matches torch with the
/// SAME grouped+dilated geometry.
///
/// torch driver:
///   inp = (torch.arange(4*8*8).float()*0.01 - 1.0).reshape(4,8,8)
///   w   = (torch.arange(2*2*3*3).float()*0.05 - 0.4).reshape(2,2,3,3)
///   b   = torch.tensor([0.3, -0.7])
///   F.conv2d(inp, w, b, stride=(3,2), padding=(2,1), dilation=(4,4), groups=2)
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "oracle-derived f32 expected values copied verbatim from live torch 2.11"
)]
fn divergence_1441_conv2d_unbatched_grouped_dilated_matches_torch() {
    let input: Vec<f32> = (0..(4 * 8 * 8)).map(|i| i as f32 * 0.01 - 1.0).collect();
    let weight: Vec<f32> = (0..(2 * 2 * 3 * 3))
        .map(|i| i as f32 * 0.05 - 0.4)
        .collect();
    let bias = vec![0.3f32, -0.7];

    // EXPECTED: live torch.nn.functional.conv2d output, [2,2,1] flattened
    // (rank D+1 = 3, no batch axis — the unbatched contract).
    let torch_out = [
        0.6660000085830688_f32,
        1.0540001392364502_f32,
        7.930000305175781_f32,
        6.205999851226807_f32,
    ];

    let mut conv = Conv2d::<f32>::new_full(4, 2, (3, 3), (3, 2), (2, 1), (4, 4), 2, true).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&weight, &[2, 2, 3, 3]));
        params[1].set_data(t(&bias, &[2]));
    }
    let y = Module::<f32>::forward(&conv, &t(&input, &[4, 8, 8])).unwrap();
    // Unbatched in -> unbatched out (rank D+1), NOT [1,2,2,1].
    assert_eq!(y.shape(), &[2, 2, 1]);
    assert_close(
        y.data().unwrap(),
        &torch_out,
        "conv2d UNBATCHED groups=2 dilation=(4,4) vs torch",
    );
}

/// op_db conv2d i=12 GEOMETRY: padding='same', dilation=3, stride=1, input
/// (1,4,5,5), weight (1,4,2,3). 'same' must preserve the input spatial size
/// (5x5) via the asymmetric `same_pad_lr` split (#1602). A decode that treated
/// 'same' as padding=0 (the pre-#1602 SameSkip) would yield a SMALLER output
/// and fail the shape assertion below.
///
/// torch driver:
///   inp = (torch.arange(1*4*5*5).float()*0.02 - 0.5).reshape(1,4,5,5)
///   w   = (torch.arange(1*4*2*3).float()*0.03 - 0.2).reshape(1,4,2,3)
///   F.conv2d(inp, w, None, stride=1, padding='same', dilation=3, groups=1)
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "oracle-derived f32 expected values copied verbatim from live torch 2.11"
)]
fn divergence_1441_conv2d_same_dilated_matches_torch() {
    let input: Vec<f32> = (0..(4 * 5 * 5)).map(|i| i as f32 * 0.02 - 0.5).collect();
    let weight: Vec<f32> = (0..(4 * 2 * 3)).map(|i| i as f32 * 0.03 - 0.2).collect();

    // EXPECTED: live torch.nn.functional.conv2d output, [1,1,5,5] flattened.
    let torch_out = [
        1.6907999515533447_f32,
        1.723599910736084_f32,
        0.8223999738693237_f32,
        1.5755999088287354_f32,
        1.6035999059677124_f32,
        3.0159997940063477_f32,
        3.06719970703125_f32,
        1.464399814605713_f32,
        2.8095996379852295_f32,
        2.8511998653411865_f32,
        3.2719998359680176_f32,
        3.32319974899292_f32,
        1.5803998708724976_f32,
        3.017599582672119_f32,
        3.059199810028076_f32,
        1.3451999425888062_f32,
        1.3635998964309692_f32,
        0.6460000276565552_f32,
        1.2299998998641968_f32,
        1.2435998916625977_f32,
        1.4371999502182007_f32,
        1.4555999040603638_f32,
        0.6859999299049377_f32,
        1.2979999780654907_f32,
        1.311599850654602_f32,
    ];

    // The arm builds new_full with padding=0 then applies StringPadding::Same.
    let conv = Conv2d::<f32>::new_full(4, 1, (2, 3), (1, 1), (0, 0), (3, 3), 1, false).unwrap();
    let mut conv = conv.with_string_padding(StringPadding::Same).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&weight, &[1, 4, 2, 3]));
    }
    let y = Module::<f32>::forward(&conv, &t(&input, &[1, 4, 5, 5])).unwrap();
    // 'same' MUST preserve input spatial size (5x5), not shrink to padding=0.
    assert_eq!(y.shape(), &[1, 1, 5, 5]);
    assert_close(
        y.data().unwrap(),
        &torch_out,
        "conv2d padding='same' dilation=3 vs torch",
    );
}

/// op_db bilinear i=3 GEOMETRY: 3-D input (x1=(2,3,3), x2=(2,3,4), w=(5,3,4),
/// bias[5]) — the #1603 N-D flatten path the runner arm FORMERLY SKIPPED (the
/// pre-3dd7d2cca `x1.ndim() > 2 -> Ok(None)` guard). Pins that the leading dims
/// (2,3) are flattened to N=6, contracted, and reshaped back to (2,3,5).
///
/// torch driver:
///   x1 = (torch.arange(2*3*3).float()*0.1 - 0.4).reshape(2,3,3)
///   x2 = (torch.arange(2*3*4).float()*0.05 - 0.3).reshape(2,3,4)
///   w  = (torch.arange(5*3*4).float()*0.02 - 0.3).reshape(5,3,4)
///   b  = torch.tensor([0.1,-0.2,0.3,-0.4,0.5])
///   F.bilinear(x1, x2, w, b)
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "oracle-derived f32 expected values copied verbatim from live torch 2.11"
)]
fn divergence_1441_bilinear_3d_matches_torch() {
    let x1: Vec<f32> = (0..(2 * 3 * 3)).map(|i| i as f32 * 0.1 - 0.4).collect();
    let x2: Vec<f32> = (0..(2 * 3 * 4)).map(|i| i as f32 * 0.05 - 0.3).collect();
    let weight: Vec<f32> = (0..(5 * 3 * 4)).map(|i| i as f32 * 0.02 - 0.3).collect();
    let bias = vec![0.1f32, -0.2, 0.3, -0.4, 0.5];

    // EXPECTED: live torch.nn.functional.bilinear output, [2,3,5] flattened.
    let torch_out = [
        -0.07280001789331436_f32,
        -0.17840002477169037_f32,
        0.5160000324249268_f32,
        0.01040002703666687_f32,
        1.1047999858856201_f32,
        0.09840000420808792_f32,
        -0.20160000026226044_f32,
        0.29840001463890076_f32,
        -0.4016000032424927_f32,
        0.4984000027179718_f32,
        -0.0040000081062316895_f32,
        -0.15280002355575562_f32,
        0.4984000027179718_f32,
        -0.05040004849433899_f32,
        1.0007998943328857_f32,
        -0.3800000548362732_f32,
        -0.03200004994869232_f32,
        1.1159999370574951_f32,
        1.0640000104904175_f32,
        2.611999988555908_f32,
        -1.0296001434326172_f32,
        0.16079984605312347_f32,
        2.1511998176574707_f32,
        2.9415998458862305_f32,
        5.332000255584717_f32,
        -1.9528003931045532_f32,
        0.4255998730659485_f32,
        3.6040000915527344_f32,
        5.582399845123291_f32,
        9.160799026489258_f32,
    ];

    let mut bl = Bilinear::<f32>::new(3, 4, 5, true).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut bl);
        params[0].set_data(t(&weight, &[5, 3, 4]));
        params[1].set_data(t(&bias, &[5]));
    }
    let y = bl
        .forward_pair(&t(&x1, &[2, 3, 3]), &t(&x2, &[2, 3, 4]))
        .unwrap();
    // N-D: leading (2,3) preserved, last dim -> out_features 5.
    assert_eq!(y.shape(), &[2, 3, 5]);
    assert_close(
        y.data().unwrap(),
        &torch_out,
        "bilinear 3-D N-D flatten vs torch",
    );
}
