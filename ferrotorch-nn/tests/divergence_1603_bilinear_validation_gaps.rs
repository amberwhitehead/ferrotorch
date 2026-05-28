//! ACToR discriminator re-audit of commit `09ffca9c0` (#1603) — validation
//! parity gaps in `Bilinear::forward_pair` vs `aten/src/ATen/native/Linear.cpp`.
//!
//! The generator's in-crate error tests assert only `.is_err()` for three
//! cases (mismatched ndim, mismatched leading dim, wrong feature dim). Two
//! upstream-checked conditions are NOT exercised:
//!
//!   (A) Same-batch-PRODUCT leading mismatch. Generator's leading-mismatch
//!       test uses (2,3,3) vs (2,4,2): products 6 vs 8 differ, so even a
//!       buggy product-only flatten would still fail downstream. A flatten
//!       that only compares the batch PRODUCT (not per-dim) would WRONGLY
//!       accept (2,3,3) vs (3,2,2) since 2*3 == 3*2 == 6. Upstream
//!       `Linear.cpp:778-781` checks per-dim equality and raises. This test
//!       pins the per-dim check with a product-equal mismatch.
//!
//!   (B) Bias-size validation. `Linear.cpp:788-790`:
//!       `TORCH_CHECK(!bias.defined() || bias.sym_size(0) == weight.sym_size(0))`.
//!       torch raises for ANY bias whose size != out_features (incl. size 1 —
//!       no broadcast leniency). ferrotorch's `forward_pair` has NO explicit
//!       bias-size guard; it reshapes the bias to `[1, out]` then add-broadcasts.
//!       This test pins that a wrong-size bias is rejected (matching torch),
//!       NOT silently broadcast or panicked.
//!
//! EXPECTED behaviors are the LIVE torch 2.11.0+cu130 outcomes (each driver
//! reproduced inline), NOT copied from ferrotorch (R-CHAR-3).

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_nn::Bilinear;
use ferrotorch_nn::module::Module;

fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// (A) Per-dim leading-shape mismatch with EQUAL batch product (6 == 6).
///
/// torch driver:
///   torch.nn.functional.bilinear(zeros(2,3,3), zeros(3,2,2), zeros(2,3,2), zeros(2))
///   -> RuntimeError: bilinear(): input batch dimensions do not match at dim 0
///
/// A product-only flatten (`N1 = 2*3 == 6 == 3*2 = N2`) would not catch this;
/// only a per-dim check does. The contraction would otherwise run on
/// mismatched [6,3] vs [6,2] and silently produce a [6,2]/[2,3,2] result that
/// torch never returns.
#[test]
fn divergence_1603_bilinear_leading_mismatch_equal_product_rejected() {
    let bl = Bilinear::<f32>::new(3, 2, 2, true).unwrap();
    // x1 leading (2,3) prod 6; x2 leading (3,2) prod 6. Same product, diff dims.
    let x1 = t(&[0.0; 2 * 3 * 3], &[2, 3, 3]);
    let x2 = t(&[0.0; 3 * 2 * 2], &[3, 2, 2]);
    let r = bl.forward_pair(&x1, &x2);
    assert!(
        r.is_err(),
        "torch raises 'input batch dimensions do not match at dim 0' for \
         leading (2,3) vs (3,2) even though both flatten to N=6; ferrotorch \
         must also Err, not silently contract"
    );
}

/// (B) Bias-size mismatch: bias.size(0) = 3 != out_features = 2.
///
/// torch driver:
///   torch.nn.functional.bilinear(zeros(2,3), zeros(2,2), zeros(2,3,2), zeros(3))
///   -> RuntimeError: bilinear(): bias size does not match weight size: got 3 but expected 2
///
/// Mirrors `Linear.cpp:788-790`. ferrotorch has no explicit bias-size guard in
/// `forward_pair`; this test pins that a wrong-size bias is rejected (Err),
/// matching torch — not a panic and not a silently mis-shaped output.
#[test]
fn divergence_1603_bilinear_oversized_bias_rejected() {
    let mut bl = Bilinear::<f32>::new(3, 2, 2, true).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut bl);
        params[0].set_data(t(&[0.0; 12], &[2, 3, 2]));
        // Inject a wrong-size bias: 3 elems, out_features is 2.
        params[1].set_data(t(&[0.1, 0.2, 0.3], &[3]));
    }
    let x1 = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let x2 = t(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| bl.forward_pair(&x1, &x2)));
    match r {
        Ok(Ok(out)) => panic!(
            "torch raises for bias size 3 != out 2; ferrotorch silently \
             returned shape {:?} instead of erroring",
            out.shape()
        ),
        Ok(Err(_)) => { /* matches torch: rejected */ }
        Err(_) => panic!(
            "torch raises a clean RuntimeError for bias size 3 != out 2; \
             ferrotorch PANICKED instead of returning Err"
        ),
    }
}

/// (B') Undersized bias (size 1): torch ALSO raises (no broadcast leniency).
///
/// torch driver:
///   torch.nn.functional.bilinear(zeros(2,3), zeros(2,2), zeros(2,3,2), zeros(1))
///   -> RuntimeError: bilinear(): bias size does not match weight size: got 1 but expected 2
#[test]
fn divergence_1603_bilinear_undersized_bias_rejected() {
    let mut bl = Bilinear::<f32>::new(3, 2, 2, true).unwrap();
    {
        let mut params = Module::<f32>::parameters_mut(&mut bl);
        params[0].set_data(t(&[0.0; 12], &[2, 3, 2]));
        params[1].set_data(t(&[0.5], &[1])); // size 1, out is 2
    }
    let x1 = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let x2 = t(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| bl.forward_pair(&x1, &x2)));
    match r {
        Ok(Ok(out)) => panic!(
            "torch raises for bias size 1 != out 2 (no broadcast); ferrotorch \
             silently returned shape {:?}",
            out.shape()
        ),
        Ok(Err(_)) => { /* matches torch */ }
        Err(_) => panic!(
            "torch raises a clean RuntimeError for bias size 1 != out 2; \
             ferrotorch PANICKED instead of returning Err"
        ),
    }
}
