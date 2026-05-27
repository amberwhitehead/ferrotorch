//! Divergence audit for commit cffd5117d (#1294): rfft non-default-axis
//! backward routing.
//!
//! The builder routes 1-D `rfft`/`irfft` backward for `dim != -1` through the
//! axis-general `RfftnBackward`/`IrfftnBackward` with `axes=[dim]`
//! (`ferrotorch-core/src/grad_fns/fft.rs:512-541`). The default last-axis
//! backward-norm case takes the cheaper `RfftBackward` fast path
//! (`:521-523`); the non-default path is a DISTINCT code path the f32 parity
//! sweep only covers at rtol=1e-4. This pins the gradient at float64 against
//! torch autograd.
//!
//! Method: drive `rfft_differentiable_norm` along dim=0 of a real [4,3]
//! tensor, attach the grad_fn, call `backward(&cotangent)` with the
//! real-ones cotangent (≡ torch loss `y.real.sum()`), and compare to torch's
//! `x.grad` at float64.
//!
//! Expected values are torch float64 `x.grad` outputs (torch 2.11.0,
//! `torch.fft.rfft(x, dim=0, norm=...).real.sum().backward()`), NOT copied
//! from ferrotorch (R-CHAR-3).

use ferrotorch_core::grad_fns::fft::rfft_differentiable_norm;
use ferrotorch_core::{FftNorm, from_vec};

const TOL: f64 = 1e-9;

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    let mut worst = 0.0f64;
    let mut wi = 0usize;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let d = (a - e).abs();
        if d > worst {
            worst = d;
            wi = i;
        }
    }
    assert!(
        worst < tol,
        "{label}: worst abs diff {worst:.3e} at index {wi} \
         (ferrotorch={}, torch_float64={}) exceeds tol {tol:.1e}",
        actual[wi],
        expected[wi],
    );
}

/// Real-ones cotangent for a complex output of `cells` complex elements,
/// laid out interleaved `[re, im, ...]`. (1,0) per element ≡ loss = y.real.sum().
fn real_ones_cotangent(cells: usize) -> ferrotorch_core::Tensor<f64> {
    let mut data = Vec::with_capacity(cells * 2);
    for _ in 0..cells {
        data.push(1.0);
        data.push(0.0);
    }
    // Caller reshapes; rfft output here is [3, 3, 2] (9 complex cells).
    from_vec(data, &[3, 3, 2]).unwrap()
}

/// Divergence probe: rfft along dim=0 (non-default axis), backward norm.
/// torch oracle: `torch.fft.rfft(arange(1..13).reshape(4,3), dim=0).real.sum()
/// .backward()` → x.grad = [3,3,3, 0,0,0, 1,1,1, 0,0,0].
#[test]
fn divergence_rfft_dim0_backward_grad_matches_torch() {
    let original: Vec<f64> = (1..=12).map(|x| x as f64).collect();
    let x = from_vec(original, &[4, 3]).unwrap().requires_grad_(true);
    let y = rfft_differentiable_norm(&x, None, Some(0), FftNorm::Backward).unwrap();
    assert_eq!(y.shape(), &[3, 3, 2]);
    let gf = y
        .grad_fn()
        .expect("rfft_norm dim=0 should attach a grad_fn");
    let cot = real_ones_cotangent(9);
    let grads = gf.backward(&cot).unwrap();
    let g = grads[0].as_ref().unwrap();
    let expected = [3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    assert_close(g.data().unwrap(), &expected, TOL, "rfft dim0 backward grad");
}

/// Divergence probe: rfft along dim=0, ortho norm. The non-default-axis
/// ortho backward folds 1/sqrt(n) into the adjoint; a wrong scale shows here.
/// torch oracle: same loss, norm="ortho" → x.grad = [1.5,1.5,1.5, 0,0,0,
/// 0.5,0.5,0.5, 0,0,0].
#[test]
fn divergence_rfft_dim0_ortho_grad_matches_torch() {
    let original: Vec<f64> = (1..=12).map(|x| x as f64).collect();
    let x = from_vec(original, &[4, 3]).unwrap().requires_grad_(true);
    let y = rfft_differentiable_norm(&x, None, Some(0), FftNorm::Ortho).unwrap();
    assert_eq!(y.shape(), &[3, 3, 2]);
    let gf = y.grad_fn().unwrap();
    let cot = real_ones_cotangent(9);
    let grads = gf.backward(&cot).unwrap();
    let g = grads[0].as_ref().unwrap();
    let expected = [1.5, 1.5, 1.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0];
    assert_close(g.data().unwrap(), &expected, TOL, "rfft dim0 ortho grad");
}
