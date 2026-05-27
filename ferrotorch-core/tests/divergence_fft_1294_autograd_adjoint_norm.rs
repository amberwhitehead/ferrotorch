//! Divergence audit for commit cffd5117d (FFT norm/dim/s autograd, #1294).
//!
//! Verifies the `adjoint_norm` backward identity
//! (`ferrotorch-core/src/grad_fns/fft.rs:69,127-130,188-189`) against torch
//! autograd at float64/complex128. The builder claims:
//!   fft'  = ifft(adjoint),  ifft' = fft(adjoint),
//!   adjoint maps Backward<->Forward, Ortho->Ortho
//! (derivatives.yaml:2960-2961). A wrong SIGN or SCALE in the adjoint would
//! land within the f32 rtol=1e-4 sweep for small n but is pinned here at
//! float64 against torch's exact gradient.
//!
//! Method: drive ferrotorch's `*_differentiable_norm` to attach the grad_fn,
//! then call `grad_fn().backward(&cotangent)` with an explicit cotangent and
//! compare to torch's `x.grad` for the matching scalar loss. The cotangent
//! `(1,0)` at every output ≡ torch loss `y.real.sum()`; `(0,1)` ≡
//! `y.imag.sum()`.
//!
//! Expected values are torch float64/complex128 `x.grad` outputs (torch
//! 2.11.0), NOT copied from ferrotorch (R-CHAR-3).

use ferrotorch_core::grad_fns::fft::{fft_differentiable_norm, ifft_differentiable_norm};
use ferrotorch_core::{FftNorm, from_vec};

const TOL: f64 = 1e-9;

fn complex_tensor(re: &[f64], im: &[f64]) -> ferrotorch_core::Tensor<f64> {
    let mut data = Vec::with_capacity(re.len() * 2);
    for i in 0..re.len() {
        data.push(re[i]);
        data.push(im[i]);
    }
    from_vec(data, &[re.len(), 2]).unwrap()
}

fn cotangent(n: usize, re: f64, im: f64) -> ferrotorch_core::Tensor<f64> {
    let mut data = Vec::with_capacity(n * 2);
    for _ in 0..n {
        data.push(re);
        data.push(im);
    }
    from_vec(data, &[n, 2]).unwrap()
}

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

const RE: [f64; 6] = [1.0, -1.0, 3.0, 0.0, -2.5, 0.7];
const IM: [f64; 6] = [2.0, 0.5, -1.0, 0.0, 1.5, -0.3];

/// Divergence probe: backward of `fft(norm="ortho")` with real-ones cotangent.
/// torch oracle: `torch.fft.fft(x, norm="ortho").real.sum().backward()` →
/// x.grad = [2.4494897427831783, 0, 0...].
#[test]
fn divergence_fft_backward_ortho_grad_matches_torch() {
    let x = complex_tensor(&RE, &IM).requires_grad_(true);
    let y = fft_differentiable_norm(&x, None, None, FftNorm::Ortho).unwrap();
    let gf = y.grad_fn().expect("fft_norm should attach grad_fn");
    let cot = cotangent(6, 1.0, 0.0); // ≡ loss = y.real.sum()
    let grads = gf.backward(&cot).unwrap();
    let g = grads[0].as_ref().unwrap();
    let expected = [
        2.4494897427831783,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    assert_close(g.data().unwrap(), &expected, TOL, "fft ortho real-cot grad");
}

/// Divergence probe: backward of `fft(norm="ortho")` with imag-ones cotangent.
/// torch oracle: `torch.fft.fft(x, norm="ortho").imag.sum().backward()` →
/// x.grad = [0, 2.4494897427831783, 0...].
#[test]
fn divergence_fft_backward_ortho_imag_cot_grad_matches_torch() {
    let x = complex_tensor(&RE, &IM).requires_grad_(true);
    let y = fft_differentiable_norm(&x, None, None, FftNorm::Ortho).unwrap();
    let gf = y.grad_fn().unwrap();
    let cot = cotangent(6, 0.0, 1.0); // ≡ loss = y.imag.sum()
    let grads = gf.backward(&cot).unwrap();
    let g = grads[0].as_ref().unwrap();
    let expected = [
        0.0,
        2.4494897427831783,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    assert_close(g.data().unwrap(), &expected, TOL, "fft ortho imag-cot grad");
}

/// Divergence probe: backward of `fft(norm="backward")` real cotangent.
/// torch oracle: x.grad = [6, 0, 0...] (adjoint of fft sums to n at index 0).
#[test]
fn divergence_fft_backward_default_grad_matches_torch() {
    let x = complex_tensor(&RE, &IM).requires_grad_(true);
    let y = fft_differentiable_norm(&x, None, None, FftNorm::Backward).unwrap();
    let gf = y.grad_fn().unwrap();
    let cot = cotangent(6, 1.0, 0.0);
    let grads = gf.backward(&cot).unwrap();
    let g = grads[0].as_ref().unwrap();
    let expected = [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_close(
        g.data().unwrap(),
        &expected,
        TOL,
        "fft backward real-cot grad",
    );
}

/// Divergence probe: backward of `fft(norm="forward")` real cotangent.
/// torch oracle: x.grad = [1, 0, 0...] (forward norm divides adjoint by n).
#[test]
fn divergence_fft_backward_forward_grad_matches_torch() {
    let x = complex_tensor(&RE, &IM).requires_grad_(true);
    let y = fft_differentiable_norm(&x, None, None, FftNorm::Forward).unwrap();
    let gf = y.grad_fn().unwrap();
    let cot = cotangent(6, 1.0, 0.0);
    let grads = gf.backward(&cot).unwrap();
    let g = grads[0].as_ref().unwrap();
    let expected = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_close(
        g.data().unwrap(),
        &expected,
        TOL,
        "fft forward real-cot grad",
    );
}

/// Divergence probe: backward of `ifft(norm="ortho")` real cotangent.
/// torch oracle: `torch.fft.ifft(x, norm="ortho").real.sum().backward()` →
/// x.grad = [2.4494897427831783, 0...]. ifft' = fft(adjoint=Ortho).
#[test]
fn divergence_ifft_backward_ortho_grad_matches_torch() {
    let x = complex_tensor(&RE, &IM).requires_grad_(true);
    let y = ifft_differentiable_norm(&x, None, None, FftNorm::Ortho).unwrap();
    let gf = y.grad_fn().unwrap();
    let cot = cotangent(6, 1.0, 0.0);
    let grads = gf.backward(&cot).unwrap();
    let g = grads[0].as_ref().unwrap();
    let expected = [
        2.4494897427831783,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    assert_close(
        g.data().unwrap(),
        &expected,
        TOL,
        "ifft ortho real-cot grad",
    );
}

/// Divergence probe: backward of `ifft(norm="backward")` real cotangent.
/// torch oracle: x.grad = [1, 0...] (ifft' = fft(adjoint=Forward), 1/n scale).
#[test]
fn divergence_ifft_backward_default_grad_matches_torch() {
    let x = complex_tensor(&RE, &IM).requires_grad_(true);
    let y = ifft_differentiable_norm(&x, None, None, FftNorm::Backward).unwrap();
    let gf = y.grad_fn().unwrap();
    let cot = cotangent(6, 1.0, 0.0);
    let grads = gf.backward(&cot).unwrap();
    let g = grads[0].as_ref().unwrap();
    let expected = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_close(
        g.data().unwrap(),
        &expected,
        TOL,
        "ifft backward real-cot grad",
    );
}
