//! ACToR discriminator re-audit of commit `adc619d83` (#1602 conv2d
//! `padding='same'`), targeting the PER-DIMENSION asymmetric split that the
//! builder's own unit tests never disambiguate.
//!
//! WHY THIS FILE EXISTS — the builder's conv2d 'same' tests use only SQUARE
//! kernels ((3,3) odd, (2,2) even). A square kernel applies the IDENTICAL split
//! to H and W, so the test is invariant under an H<->W split swap or an
//! implementation that computes one split and reuses it for both dims. This
//! file uses:
//!   1. a MIXED-parity kernel (2,3): H is EVEN (total=1 -> top0/bottom1,
//!      ASYMMETRIC) while W is ODD (total=2 -> left1/right1, SYMMETRIC). If the
//!      code swapped the per-dim split or applied W's symmetric split to H, the
//!      output diverges.
//!   2. a DILATED (3,3) kernel with dilation (2,2): total per dim = 4
//!      SYMMETRIC, but the pad MAGNITUDE (2 each side) scales with dilation; a
//!      `total = kernel-1` bug pads only 1 each side and diverges.
//!
//! EXPECTED values are LIVE torch 2.11.0+cu130 `F.conv2d(.., padding="same")`
//! outputs from the inline drivers; NOT copied from ferrotorch (R-CHAR-3).
//! Backward uses the full-graph consumer path (`sum` -> `backward` -> `grad`).

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_nn::conv::StringPadding;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::Conv2d;

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
        "{ctx}: length mismatch (got {} want {})",
        got.len(),
        want.len()
    );
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
// MIXED-parity kernel (2,3): H EVEN (asym top0/bot1), W ODD (sym left1/right1).
//
// torch driver:
//   x = torch.arange(1,17).float().view(1,1,4,4).requires_grad_(True)
//   w = torch.arange(1,7).float().view(1,1,2,3)
//   y = F.conv2d(x, w, padding="same"); y.sum().backward()
// ---------------------------------------------------------------------------
#[test]
fn divergence_1602_conv2d_same_mixed_parity_kernel_fwd() {
    let mut conv = Conv2d::<f32>::new_full(1, 1, (2, 3), (1, 1), (0, 0), (1, 1), 1, false).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 2, 3]));
    }
    let conv = conv.with_string_padding(StringPadding::Same).unwrap();
    let x = t(
        &(1..=16).map(|v| v as f32).collect::<Vec<_>>(),
        &[1, 1, 4, 4],
    );
    let y = Module::<f32>::forward(&conv, &x).unwrap();
    assert_eq!(
        y.shape(),
        &[1, 1, 4, 4],
        "mixed-parity 'same' preserves spatial dims"
    );
    let expected = [
        69.0, 106.0, 127.0, 79.0, 133.0, 190.0, 211.0, 127.0, 197.0, 274.0, 295.0, 175.0, 68.0,
        86.0, 92.0, 47.0,
    ];
    assert_close(y.data().unwrap(), &expected, "conv2d same kernel (2,3) fwd");
}

#[test]
fn divergence_1602_conv2d_same_mixed_parity_kernel_bwd() {
    let mut conv = Conv2d::<f32>::new_full(1, 1, (2, 3), (1, 1), (0, 0), (1, 1), 1, false).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 2, 3]));
    }
    let conv = conv.with_string_padding(StringPadding::Same).unwrap();
    let x = leaf(
        &(1..=16).map(|v| v as f32).collect::<Vec<_>>(),
        &[1, 1, 4, 4],
    );
    let y = Module::<f32>::forward(&conv, &x).unwrap();
    let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
    ferrotorch_core::backward(&sum).unwrap();
    let gx = x.grad().unwrap().expect("input grad must be populated");
    assert_eq!(gx.shape(), &[1, 1, 4, 4]);
    let expected = [
        3.0, 6.0, 6.0, 5.0, 12.0, 21.0, 21.0, 16.0, 12.0, 21.0, 21.0, 16.0, 12.0, 21.0, 21.0, 16.0,
    ];
    assert_close(
        gx.data().unwrap(),
        &expected,
        "conv2d same kernel (2,3) grad_input",
    );
}

// ---------------------------------------------------------------------------
// DILATED (3,3) kernel, dilation (2,2): total per dim = 2*(3-1) = 4 SYMMETRIC,
// pad MAGNITUDE 2 each side (scales with dilation).
//
// torch driver:
//   x = torch.arange(1,26).float().view(1,1,5,5).requires_grad_(True)
//   w = torch.arange(1,10).float().view(1,1,3,3)
//   y = F.conv2d(x, w, padding="same", dilation=(2,2)); y.sum().backward()
// ---------------------------------------------------------------------------
#[test]
fn divergence_1602_conv2d_same_dilated_kernel_fwd() {
    let mut conv = Conv2d::<f32>::new_full(1, 1, (3, 3), (1, 1), (0, 0), (2, 2), 1, false).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 1, 3, 3],
        ));
    }
    let conv = conv.with_string_padding(StringPadding::Same).unwrap();
    let x = t(
        &(1..=25).map(|v| v as f32).collect::<Vec<_>>(),
        &[1, 1, 5, 5],
    );
    let y = Module::<f32>::forward(&conv, &x).unwrap();
    assert_eq!(
        y.shape(),
        &[1, 1, 5, 5],
        "dilated 'same' preserves spatial dims"
    );
    let expected = [
        228.0, 256.0, 365.0, 224.0, 248.0, 368.0, 396.0, 560.0, 344.0, 368.0, 519.0, 552.0, 777.0,
        474.0, 501.0, 224.0, 240.0, 326.0, 188.0, 200.0, 304.0, 320.0, 431.0, 248.0, 260.0,
    ];
    assert_close(
        y.data().unwrap(),
        &expected,
        "conv2d same (3,3) dil(2,2) fwd",
    );
}

#[test]
fn divergence_1602_conv2d_same_dilated_kernel_bwd() {
    let mut conv = Conv2d::<f32>::new_full(1, 1, (3, 3), (1, 1), (0, 0), (2, 2), 1, false).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 1, 3, 3],
        ));
    }
    let conv = conv.with_string_padding(StringPadding::Same).unwrap();
    let x = leaf(
        &(1..=25).map(|v| v as f32).collect::<Vec<_>>(),
        &[1, 1, 5, 5],
    );
    let y = Module::<f32>::forward(&conv, &x).unwrap();
    let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
    ferrotorch_core::backward(&sum).unwrap();
    let gx = x.grad().unwrap().expect("input grad must be populated");
    assert_eq!(gx.shape(), &[1, 1, 5, 5]);
    let expected = [
        12.0, 12.0, 21.0, 16.0, 16.0, 12.0, 12.0, 21.0, 16.0, 16.0, 27.0, 27.0, 45.0, 33.0, 33.0,
        24.0, 24.0, 39.0, 28.0, 28.0, 24.0, 24.0, 39.0, 28.0, 28.0,
    ];
    assert_close(
        gx.data().unwrap(),
        &expected,
        "conv2d same (3,3) dil(2,2) grad_input",
    );
}
