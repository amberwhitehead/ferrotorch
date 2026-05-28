//! ACToR discriminator re-audit of commit `adc619d83` (#1602 conv
//! `padding='same'/'valid'` + #1604 unbatched input).
//!
//! WHY THIS FILE EXISTS — the builder's own conv.rs unit tests for #1602 cover
//! `'same'` for ODD and EVEN kernels at DILATION=1 and GROUPS=1 only. They do
//! NOT exercise the configurations where the asymmetric `'same'` split
//! interacts with the rest of the conv geometry and is most likely to diverge:
//!
//!   1. DILATION>1 `'same'`: `total = dilation*(kernel-1)` so the pad amount
//!      and its asymmetric split both scale with dilation. An implementation
//!      that hard-codes `total = kernel-1` (ignoring dilation) or that pads the
//!      WRONG side passes the dilation=1 unit tests but fails here. Two sub-
//!      cases pin the split parity: k=2 dil=3 -> total=3 (ASYMMETRIC, END+1)
//!      and k=3 dil=2 -> total=4 (SYMMETRIC). torch warns this path may need a
//!      zero-padded copy (`Convolution.cpp:1024`), so it is the highest-risk
//!      code path.
//!   2. GROUPS>1 `'same'`: the `'same'` pre-pad must compose with the grouped
//!      im2col/col2im (the #1600/#1601 feature).
//!   3. The DILATION>1 + GROUPS>1 + `'same'` combination (case D), the full
//!      cross-product the builder never touched.
//!
//! Every EXPECTED value below is the LIVE torch 2.11.0+cu130 output of the
//! matching `torch.nn.functional.conv{1,2}d(.., padding="same", ..)` call,
//! computed by the inline torch driver quoted above each constant block and
//! NOT copied from the ferrotorch side (R-CHAR-3).
//!
//! BACKWARD methodology mirrors the builder's own passing unit tests AND the
//! production consumer: `out.sum()` -> `ferrotorch_core::backward(&sum)` ->
//! read `x.grad()` on the original (unpadded) leaf. This traverses the full
//! autograd graph including the `Pad*Backward` slice-back that must un-pad the
//! asymmetric `'same'` pre-pad. (A direct `grad_fn().backward()` on the conv
//! output only returns the grad of the *padded* tensor and would NOT exercise
//! the un-pad — that is not the consumer path.)
//!
//! NOTE: `StringPadding` is reached via the `ferrotorch_nn::conv::` module path
//! because the builder did NOT re-export it from the crate root `lib.rs:213`
//! `pub use conv::{...}` list (separate API-surface gap).
//!
//! Production path driven: `Conv{1,2}d::new_full` + `Module::parameters_mut` +
//! `Parameter::set_data` + `Conv*d::with_string_padding(StringPadding::Same)` +
//! `Module::forward` then full-graph `backward`.

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_nn::conv::StringPadding;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::{Conv1d, Conv2d};

/// Leaf tensor that does NOT require grad.
fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// Leaf tensor that requires grad (so `grad_input` is computed in backward).
fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

/// torch-envelope close check (rtol 1e-3, atol 1e-4). A wrong-side / wrong-
/// magnitude `'same'` pad diverges by O(1)..O(100) >> this tolerance.
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
// CASE A — conv1d 'same' DILATION=3, EVEN kernel k=2.
//
// total = dilation*(kernel-1) = 3*(2-1) = 3 -> ASYMMETRIC split left=1 right=2
// (the END side gets the extra unit, `Pool.h:91-107`). A symmetric split, or a
// split that ignores dilation (total = k-1 = 1), gives a different sequence.
//
// torch driver:
//   x = torch.arange(1,8).float().view(1,1,7).requires_grad_(True)
//   w = torch.tensor([1.,2.]).view(1,1,2)
//   y = F.conv1d(x, w, padding="same", dilation=3); y -> [6,9,12,15,18,5,6]
//   y.sum().backward(); x.grad -> [1,1,3,3,3,3,2]
// ---------------------------------------------------------------------------
#[test]
fn divergence_1602_conv1d_same_dilation3_even_kernel_fwd() {
    let mut conv = Conv1d::<f32>::new_full(1, 1, 2, 1, 0, 3, 1, false).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&[1.0, 2.0], &[1, 1, 2]));
    }
    let conv = conv.with_string_padding(StringPadding::Same).unwrap();
    let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[1, 1, 7]);
    let y = Module::<f32>::forward(&conv, &x).unwrap();
    assert_eq!(y.shape(), &[1, 1, 7], "dilated 'same' must preserve length");
    assert_close(
        y.data().unwrap(),
        &[6.0, 9.0, 12.0, 15.0, 18.0, 5.0, 6.0],
        "A_fwd conv1d same k=2 dil=3",
    );
}

#[test]
fn divergence_1602_conv1d_same_dilation3_even_kernel_bwd() {
    let mut conv = Conv1d::<f32>::new_full(1, 1, 2, 1, 0, 3, 1, false).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&[1.0, 2.0], &[1, 1, 2]));
    }
    let conv = conv.with_string_padding(StringPadding::Same).unwrap();
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[1, 1, 7]);
    let y = Module::<f32>::forward(&conv, &x).unwrap();
    let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
    ferrotorch_core::backward(&sum).unwrap();
    let gx = x.grad().unwrap().expect("input grad must be populated");
    assert_eq!(
        gx.shape(),
        &[1, 1, 7],
        "grad_input must match the ORIGINAL (unpadded) input length"
    );
    assert_close(
        gx.data().unwrap(),
        &[1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 2.0],
        "A_gx conv1d same k=2 dil=3 grad_input",
    );
}

// ---------------------------------------------------------------------------
// CASE B — conv1d 'same' DILATION=2, ODD kernel k=3.
//
// total = 2*(3-1) = 4 -> SYMMETRIC split left=2 right=2. Pins that the dilated
// SYMMETRIC case is also right (a fixed total=k-1=2 split would give left=1
// right=1 and the wrong sequence/length).
//
// torch driver:
//   x = torch.arange(1,8).float().view(1,1,7).requires_grad_(True)
//   w = torch.tensor([1.,2.,3.]).view(1,1,3)
//   y = F.conv1d(x, w, padding="same", dilation=2); y -> [11,16,22,28,34,16,19]
//   y.sum().backward(); x.grad -> [3,3,6,6,6,5,5]
// ---------------------------------------------------------------------------
#[test]
fn divergence_1602_conv1d_same_dilation2_odd_kernel_fwd() {
    let mut conv = Conv1d::<f32>::new_full(1, 1, 3, 1, 0, 2, 1, false).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&[1.0, 2.0, 3.0], &[1, 1, 3]));
    }
    let conv = conv.with_string_padding(StringPadding::Same).unwrap();
    let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[1, 1, 7]);
    let y = Module::<f32>::forward(&conv, &x).unwrap();
    assert_eq!(y.shape(), &[1, 1, 7]);
    assert_close(
        y.data().unwrap(),
        &[11.0, 16.0, 22.0, 28.0, 34.0, 16.0, 19.0],
        "B_fwd conv1d same k=3 dil=2",
    );
}

#[test]
fn divergence_1602_conv1d_same_dilation2_odd_kernel_bwd() {
    let mut conv = Conv1d::<f32>::new_full(1, 1, 3, 1, 0, 2, 1, false).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&[1.0, 2.0, 3.0], &[1, 1, 3]));
    }
    let conv = conv.with_string_padding(StringPadding::Same).unwrap();
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[1, 1, 7]);
    let y = Module::<f32>::forward(&conv, &x).unwrap();
    let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
    ferrotorch_core::backward(&sum).unwrap();
    let gx = x.grad().unwrap().expect("input grad must be populated");
    assert_eq!(gx.shape(), &[1, 1, 7]);
    assert_close(
        gx.data().unwrap(),
        &[3.0, 3.0, 6.0, 6.0, 6.0, 5.0, 5.0],
        "B_gx conv1d same k=3 dil=2 grad_input",
    );
}

// ---------------------------------------------------------------------------
// CASE C — conv2d 'same' GROUPS=2, EVEN kernel (2,2).
//
// total per dim = 1 -> ASYMMETRIC (0,1). Group-1 weights ([5,6,7,8]) >> group-0
// ([1,2,3,4]); a cross-group leak in the grouped+same path diverges by a large
// margin (group-1 outputs are ~10x group-0).
//
// torch driver:
//   x = torch.arange(1,33).float().view(1,2,4,4).requires_grad_(True)
//   w = torch.arange(1,9).float().view(2,1,2,2)
//   y = F.conv2d(x, w, padding="same", groups=2)
//   y.sum().backward()
// ---------------------------------------------------------------------------
#[test]
fn divergence_1602_conv2d_same_grouped2_even_kernel_fwd() {
    let mut conv = Conv2d::<f32>::new_full(2, 2, (2, 2), (1, 1), (0, 0), (1, 1), 2, false).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 1, 2, 2]));
    }
    let conv = conv.with_string_padding(StringPadding::Same).unwrap();
    let x = t(
        &(1..=32).map(|v| v as f32).collect::<Vec<_>>(),
        &[1, 2, 4, 4],
    );
    let y = Module::<f32>::forward(&conv, &x).unwrap();
    assert_eq!(
        y.shape(),
        &[1, 2, 4, 4],
        "grouped 'same' must preserve spatial dims"
    );
    let expected = [
        44.0, 54.0, 64.0, 28.0, 84.0, 94.0, 104.0, 44.0, 124.0, 134.0, 144.0, 60.0, 41.0, 44.0,
        47.0, 16.0, 516.0, 542.0, 568.0, 268.0, 620.0, 646.0, 672.0, 316.0, 724.0, 750.0, 776.0,
        364.0, 325.0, 336.0, 347.0, 160.0,
    ];
    assert_close(
        y.data().unwrap(),
        &expected,
        "C_fwd conv2d same grouped(2) k=(2,2)",
    );
}

#[test]
fn divergence_1602_conv2d_same_grouped2_even_kernel_bwd() {
    let mut conv = Conv2d::<f32>::new_full(2, 2, (2, 2), (1, 1), (0, 0), (1, 1), 2, false).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 1, 2, 2]));
    }
    let conv = conv.with_string_padding(StringPadding::Same).unwrap();
    let x = leaf(
        &(1..=32).map(|v| v as f32).collect::<Vec<_>>(),
        &[1, 2, 4, 4],
    );
    let y = Module::<f32>::forward(&conv, &x).unwrap();
    let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
    ferrotorch_core::backward(&sum).unwrap();
    let gx = x.grad().unwrap().expect("input grad must be populated");
    assert_eq!(
        gx.shape(),
        &[1, 2, 4, 4],
        "grad_input must match the ORIGINAL (unpadded) spatial dims"
    );
    let expected = [
        1.0, 3.0, 3.0, 3.0, 4.0, 10.0, 10.0, 10.0, 4.0, 10.0, 10.0, 10.0, 4.0, 10.0, 10.0, 10.0,
        5.0, 11.0, 11.0, 11.0, 12.0, 26.0, 26.0, 26.0, 12.0, 26.0, 26.0, 26.0, 12.0, 26.0, 26.0,
        26.0,
    ];
    assert_close(
        gx.data().unwrap(),
        &expected,
        "C_gx conv2d same grouped(2) grad_input",
    );
}

// ---------------------------------------------------------------------------
// CASE D — conv1d 'same' GROUPS=2 + DILATION=2 + ODD kernel k=3 (full combo).
//
// The cross-product the builder never exercised: total=2*(3-1)=4 SYMMETRIC pad,
// split across two channel groups with distinct weights ([1,2,3] vs [4,5,6]).
//
// torch driver:
//   x = torch.arange(1,17).float().view(1,2,8).requires_grad_(True)
//   w = torch.tensor([1.,2.,3., 4.,5.,6.]).view(2,1,3)
//   y = F.conv1d(x, w, padding="same", groups=2, dilation=2)
//   y.sum().backward()
// ---------------------------------------------------------------------------
#[test]
fn divergence_1602_conv1d_same_grouped2_dilation2_combo_fwd() {
    let mut conv = Conv1d::<f32>::new_full(2, 2, 3, 1, 0, 2, 2, false).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]));
    }
    let conv = conv.with_string_padding(StringPadding::Same).unwrap();
    let x = t(&(1..=16).map(|v| v as f32).collect::<Vec<_>>(), &[1, 2, 8]);
    let y = Module::<f32>::forward(&conv, &x).unwrap();
    assert_eq!(y.shape(), &[1, 2, 8]);
    let expected = [
        11.0, 16.0, 22.0, 28.0, 34.0, 40.0, 19.0, 22.0, 111.0, 122.0, 169.0, 184.0, 199.0, 214.0,
        127.0, 136.0,
    ];
    assert_close(
        y.data().unwrap(),
        &expected,
        "D_fwd conv1d same grouped(2) dil=2",
    );
}

#[test]
fn divergence_1602_conv1d_same_grouped2_dilation2_combo_bwd() {
    let mut conv = Conv1d::<f32>::new_full(2, 2, 3, 1, 0, 2, 2, false).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut conv);
        params[0].set_data(t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]));
    }
    let conv = conv.with_string_padding(StringPadding::Same).unwrap();
    let x = leaf(&(1..=16).map(|v| v as f32).collect::<Vec<_>>(), &[1, 2, 8]);
    let y = Module::<f32>::forward(&conv, &x).unwrap();
    let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
    ferrotorch_core::backward(&sum).unwrap();
    let gx = x.grad().unwrap().expect("input grad must be populated");
    assert_eq!(gx.shape(), &[1, 2, 8]);
    let expected = [
        3.0, 3.0, 6.0, 6.0, 6.0, 6.0, 5.0, 5.0, 9.0, 9.0, 15.0, 15.0, 15.0, 15.0, 11.0, 11.0,
    ];
    assert_close(
        gx.data().unwrap(),
        &expected,
        "D_gx conv1d same grouped(2) dil=2 grad_input",
    );
}
