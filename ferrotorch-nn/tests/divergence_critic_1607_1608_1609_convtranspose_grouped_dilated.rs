//! Adversarial discriminator re-audit of commit c5f9f102e
//! (ConvTranspose1d/2d/3d groups + dilation + unbatched, closes #1607/#1608/#1609).
//!
//! The builder's in-crate `mod tests` covered groups=2, depthwise, dilation=2,
//! and a g2/d2/s2/p1/op1 combo with a SYMMETRIC kernel. This file pushes past
//! that surface with the cases the mandate flags as the classic silent bugs:
//!
//!   - groups=3 ConvTranspose2d with DISTINCT per-group weights — the strongest
//!     cross-group-leak / `[in,out/g]`<->`[out,in/g]` layout-mixup detector,
//!     because every group's weight slab is numerically distinct so any leak or
//!     transpose is value-detectable.
//!   - ConvTranspose2d g2/d2/s2/p1/op1 combo with an ASYMMETRIC kernel (3,2) —
//!     the builder used a symmetric (2,2); an asymmetric kernel exposes a
//!     transposed kH<->kW or flip-axis error the symmetric case cannot see.
//!   - ConvTranspose3d g2/d2/s2/p1/op1 combo (the builder's 3D combo used
//!     dilation=(1,1,1)); this is the hardest 3D path — dilated internal-pad
//!     arithmetic interacting with grouped channels + output_padding.
//!
//! Every weight slab `[in, out/groups, *k]` and every `expected` value below was
//! produced by the LIVE PyTorch 2.11.0+cu130 oracle (R-CHAR-3): the exact
//! `torch.nn.functional.conv_transpose{2,3}d(...)` forward output and
//! `x.grad / weight.grad / bias.grad` after `y.sum().backward()`, reproduced
//! with the same deterministic inputs constructed below. Nothing is copied from
//! ferrotorch. grad_weight is asserted in torch's transposed `[in, out/groups,
//! *k]` layout — the layout the commit message claims to mirror.
//!
//! Driving the grouped/dilated transposed path with controlled weights requires
//! `new_full` (the public dense `from_parts` / `functional::conv_transpose*` are
//! groups=1/dilation=1 only) + `Module::parameters_mut()` + `Parameter::set_data`
//! to overwrite the Kaiming-random weights. params[0]=weight, params[1]=bias.

use ferrotorch_core::Tensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_nn::{ConvTranspose2d, ConvTranspose3d, Module};

fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn assert_close(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
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
// CASE A — ConvTranspose2d groups=3, DISTINCT per-group weights.
//
// in=6 out=3 groups=3 (in_pg=2, out_pg=1), weight [6,1,2,2], stride=1, pad=0,
// output_padding=0, dilation=1. With six distinct 2x2 weight slabs and a
// distinct input, ANY cross-group channel leak or [in,out/g] layout swap shifts
// the output, so a PASS is strong evidence the grouped transposed forward AND
// backward are correct.
//
// torch driver:
//   w = (torch.arange(1,25).float()*0.1).reshape(6,1,2,2)
//   b = torch.tensor([0.5,-0.5,0.25])
//   x = (torch.arange(1,55).float()*0.5).reshape(1,6,3,3).requires_grad_(True)
//   y = F.conv_transpose2d(x, w, b, stride=(1,1), padding=(0,0),
//                          output_padding=(0,0), groups=3, dilation=(1,1))
//   y.sum().backward()
// ---------------------------------------------------------------------------
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "oracle-derived f32 expected values copied verbatim from live torch 2.11 — full precision intentional"
)]
fn divergence_ct2d_groups3_distinct_weights_matches_torch() {
    let weight: Vec<f32> = (1..=24).map(|i| i as f32 * 0.1).collect();
    let bias = [0.5f32, -0.5, 0.25];
    let mut ct =
        ConvTranspose2d::<f32>::new_full(6, 3, (2, 2), (1, 1), (0, 0), (0, 0), (1, 1), 3, true)
            .unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut ct);
        params[0].set_data(t(&weight, &[6, 1, 2, 2]));
        params[1].set_data(t(&bias, &[3]));
    }

    let x_data: Vec<f32> = (1..=54).map(|i| i as f32 * 0.5).collect();
    let x = leaf(&x_data, &[1, 6, 3, 3]);
    let y = Module::<f32>::forward(&ct, &x).unwrap();
    assert_eq!(y.shape(), &[1, 3, 4, 4]);

    assert_close(
        y.data().unwrap(),
        &[
            3.05, 6.45, 7.150001, 4.4, 7.6, 16.900002, 18.700001, 11.0, 10.0, 22.299999, 24.1,
            14.0, 7.15, 15.450001, 16.550001, 9.5, 26.25, 56.449997, 58.75, 31.0, 61.0, 129.899994,
            134.899994, 71.199997, 68.199997, 144.899994, 149.899994, 79.0, 38.75, 82.25,
            84.949997, 44.5, 80.0, 165.799988, 169.700012, 88.150002, 173.75, 359.850006,
            368.049988, 190.75, 185.75, 384.450012, 392.649994, 203.350006, 100.899994, 208.399994,
            212.700012, 110.050003,
        ],
        "A_fwd ct2d groups=3 distinct",
    );

    let grad_output = t(&[1.0f32; 48], &[1, 3, 4, 4]);
    let grads = Module::<f32>::forward(&ct, &x)
        .unwrap()
        .grad_fn()
        .unwrap()
        .backward(&grad_output)
        .unwrap();

    assert_close(
        grads[0].as_ref().unwrap().data().unwrap(),
        &[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6,
            2.6, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.8,
            5.8, 5.8, 7.4, 7.4, 7.4, 7.4, 7.4, 7.4, 7.4, 7.4, 7.4, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
            9.0, 9.0, 9.0,
        ],
        "A_gx ct2d groups=3 distinct grad_input",
    );

    assert_eq!(
        grads[1].as_ref().unwrap().shape(),
        &[6, 1, 2, 2],
        "A grad_weight must be in transposed [in, out/groups, kH, kW] layout"
    );
    assert_close(
        grads[1].as_ref().unwrap().data().unwrap(),
        &[
            22.5, 22.5, 22.5, 22.5, 63.0, 63.0, 63.0, 63.0, 103.5, 103.5, 103.5, 103.5, 144.0,
            144.0, 144.0, 144.0, 184.5, 184.5, 184.5, 184.5, 225.0, 225.0, 225.0, 225.0,
        ],
        "A_gw ct2d groups=3 distinct grad_weight [in,out/g,k] layout",
    );
    assert_close(
        grads[2].as_ref().unwrap().data().unwrap(),
        &[16.0, 16.0, 16.0],
        "A_gb ct2d groups=3 distinct grad_bias",
    );
}

// ---------------------------------------------------------------------------
// CASE B — ConvTranspose2d combo with ASYMMETRIC kernel.
// groups=2, dilation=(2,2), stride=(2,2), padding=(1,1), output_padding=(1,1),
// kernel (kH,kW)=(3,2). in=4 out=2 groups=2 (in_pg=2, out_pg=1) weight [4,1,3,2].
//
// torch driver:
//   w = (torch.arange(1,25).float()*0.1).reshape(4,1,3,2)
//   b = torch.tensor([0.3,-0.3])
//   x = torch.arange(1,37).float().reshape(1,4,3,3).requires_grad_(True)
//   y = F.conv_transpose2d(x, w, b, stride=(2,2), padding=(1,1),
//                          output_padding=(1,1), groups=2, dilation=(2,2))
//   y.sum().backward()
// ---------------------------------------------------------------------------
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "oracle-derived f32 expected values copied verbatim from live torch 2.11 — full precision intentional"
)]
fn divergence_ct2d_combo_asymmetric_kernel_matches_torch() {
    let weight: Vec<f32> = (1..=24).map(|i| i as f32 * 0.1).collect();
    let bias = [0.3f32, -0.3];
    let mut ct =
        ConvTranspose2d::<f32>::new_full(4, 2, (3, 2), (2, 2), (1, 1), (1, 1), (2, 2), 2, true)
            .unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut ct);
        params[0].set_data(t(&weight, &[4, 1, 3, 2]));
        params[1].set_data(t(&bias, &[2]));
    }

    let x_data: Vec<f32> = (1..=36).map(|i| i as f32).collect();
    let x = leaf(&x_data, &[1, 4, 3, 3]);
    let y = Module::<f32>::forward(&ct, &x).unwrap();
    assert_eq!(y.shape(), &[1, 2, 8, 6]);

    assert_close(
        y.data().unwrap(),
        &[
            0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 42.700001, 0.3, 47.099998, 0.3, 26.699999, 0.3, 0.3,
            0.3, 0.3, 0.3, 0.3, 0.3, 81.599998, 0.3, 89.400002, 0.3, 50.099998, 0.3, 0.3, 0.3, 0.3,
            0.3, 0.3, 0.3, 72.700005, 0.3, 78.700005, 0.3, 43.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
            46.400002, 0.3, 49.799999, 0.3, 27.299999, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3,
            366.100006, -0.3, 380.100037, -0.3, 198.900009, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3,
            -0.3, 610.200012, -0.3, 632.399963, -0.3, 330.300018, -0.3, -0.3, -0.3, -0.3, -0.3,
            -0.3, -0.3, 453.700012, -0.3, 469.300049, -0.3, 244.5, -0.3, -0.3, -0.3, -0.3, -0.3,
            -0.3, -0.3, 251.0, -0.3, 259.200012, -0.3, 134.699997,
        ],
        "B_fwd ct2d combo asymmetric kernel",
    );

    let grad_output = t(&[1.0f32; 96], &[1, 2, 8, 6]);
    let grads = Module::<f32>::forward(&ct, &x)
        .unwrap()
        .grad_fn()
        .unwrap()
        .backward(&grad_output)
        .unwrap();

    assert_close(
        grads[0].as_ref().unwrap().data().unwrap(),
        &[
            1.0, 1.8, 1.8, 1.2, 2.1, 2.1, 1.2, 2.1, 2.1, 2.2, 4.2, 4.2, 3.0, 5.7, 5.7, 3.0, 5.7,
            5.7, 3.4, 6.6, 6.6, 4.8, 9.3, 9.3, 4.8, 9.3, 9.3, 4.6, 9.0, 9.0, 6.6, 12.9, 12.9, 6.6,
            12.9, 12.9,
        ],
        "B_gx ct2d combo asymmetric grad_input",
    );

    assert_eq!(
        grads[1].as_ref().unwrap().shape(),
        &[4, 1, 3, 2],
        "B grad_weight must be in transposed [in, out/groups, kH, kW] layout"
    );
    assert_close(
        grads[1].as_ref().unwrap().data().unwrap(),
        &[
            28.0, 39.0, 33.0, 45.0, 33.0, 45.0, 64.0, 93.0, 87.0, 126.0, 87.0, 126.0, 100.0, 147.0,
            141.0, 207.0, 141.0, 207.0, 136.0, 201.0, 195.0, 288.0, 195.0, 288.0,
        ],
        "B_gw ct2d combo asymmetric grad_weight [in,out/g,kH,kW]",
    );
    assert_close(
        grads[2].as_ref().unwrap().data().unwrap(),
        &[48.0, 48.0],
        "B_gb ct2d combo asymmetric grad_bias",
    );
}

// ---------------------------------------------------------------------------
// CASE C — ConvTranspose3d combo WITH dilation (the builder's 3D combo had
// dilation=(1,1,1)). groups=2, dilation=(2,2,2), stride=(2,2,2),
// padding=(1,1,1), output_padding=(1,1,1), k=(2,2,2).
// in=4 out=2 groups=2 (in_pg=2, out_pg=1) weight [4,1,2,2,2].
//
// torch driver:
//   w = (torch.arange(1,33).float()*0.05).reshape(4,1,2,2,2)
//   b = torch.tensor([0.1,-0.1])
//   x = torch.arange(1,33).float().reshape(1,4,2,2,2).requires_grad_(True)
//   y = F.conv_transpose3d(x, w, b, stride=(2,2,2), padding=(1,1,1),
//                          output_padding=(1,1,1), groups=2, dilation=(2,2,2))
//   y.sum().backward()
// ---------------------------------------------------------------------------
#[test]
#[allow(
    clippy::excessive_precision,
    reason = "oracle-derived f32 expected values copied verbatim from live torch 2.11 — full precision intentional"
)]
fn divergence_ct3d_combo_dilated_grouped_matches_torch() {
    let weight: Vec<f32> = (1..=32).map(|i| i as f32 * 0.05).collect();
    let bias = [0.1f32, -0.1];
    let mut ct = ConvTranspose3d::<f32>::new_full(
        4,
        2,
        (2, 2, 2),
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
        params[0].set_data(t(&weight, &[4, 1, 2, 2, 2]));
        params[1].set_data(t(&bias, &[2]));
    }

    let x_data: Vec<f32> = (1..=32).map(|i| i as f32).collect();
    let x = leaf(&x_data, &[1, 4, 2, 2, 2]);
    let y = Module::<f32>::forward(&ct, &x).unwrap();
    assert_eq!(y.shape(), &[1, 2, 4, 4, 4]);

    assert_close(
        y.data().unwrap(),
        &[
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 66.5, 0.1, 36.899998, 0.1, 0.1, 0.1, 0.1, 0.1, 40.899998, 0.1,
            22.500002, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 50.099998, 0.1, 27.300001, 0.1, 0.1, 0.1, 0.1, 0.1,
            29.700001, 0.1, 16.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
            -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 488.699982, -0.1,
            254.300003, -0.1, -0.1, -0.1, -0.1, -0.1, 264.699982, -0.1, 137.5, -0.1, -0.1, -0.1,
            -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
            -0.1, -0.1, -0.1, -0.1, 286.700012, -0.1, 148.699997, -0.1, -0.1, -0.1, -0.1, -0.1,
            154.299988, -0.1, 79.900002,
        ],
        "C_fwd ct3d combo dilated grouped",
    );

    let grad_output = t(&[1.0f32; 128], &[1, 2, 4, 4, 4]);
    let grads = Module::<f32>::forward(&ct, &x)
        .unwrap()
        .grad_fn()
        .unwrap()
        .backward(&grad_output)
        .unwrap();

    assert_close(
        grads[0].as_ref().unwrap().data().unwrap(),
        &[
            0.4, 0.75, 0.7, 1.3, 0.6, 1.1, 1.0, 1.8, 0.8, 1.55, 1.5, 2.9, 1.4, 2.7, 2.6, 5.0, 1.2,
            2.35, 2.3, 4.5, 2.2, 4.3, 4.2, 8.2, 1.6, 3.15, 3.1, 6.1, 3.0, 5.9, 5.8, 11.4,
        ],
        "C_gx ct3d combo dilated grouped grad_input",
    );

    assert_eq!(
        grads[1].as_ref().unwrap().shape(),
        &[4, 1, 2, 2, 2],
        "C grad_weight must be in transposed [in, out/groups, kD, kH, kW] layout"
    );
    assert_close(
        grads[1].as_ref().unwrap().data().unwrap(),
        &[
            8.0, 15.0, 14.0, 26.0, 12.0, 22.0, 20.0, 36.0, 16.0, 31.0, 30.0, 58.0, 28.0, 54.0,
            52.0, 100.0, 24.0, 47.0, 46.0, 90.0, 44.0, 86.0, 84.0, 164.0, 32.0, 63.0, 62.0, 122.0,
            60.0, 118.0, 116.0, 228.0,
        ],
        "C_gw ct3d combo dilated grouped grad_weight [in,out/g,kD,kH,kW]",
    );
    assert_close(
        grads[2].as_ref().unwrap().data().unwrap(),
        &[64.0, 64.0],
        "C_gb ct3d combo dilated grouped grad_bias",
    );
}
