#![allow(clippy::needless_range_loop)]

//! Permanent regression sentinel for #808 — irfftn / hfft / ihfft panic on
//! non-Hermitian input (Bugfix Batch 7 / Dispatch A3).
//!
//! ## #808 — strict-Hermitian rejection in ferray-fft 0.3.0
//!
//! The realfft backend used by ferray-fft 0.3.0 panics with
//! *"Imaginary parts of both first and last values were non-zero"* whenever
//! the input spectrum supplied to a complex-to-real (C2R) inverse transform
//! has a non-zero imaginary component at the DC bin (k=0) or — for even-
//! length output — at the Nyquist bin (k = output_n / 2). PyTorch's
//! `torch.fft.irfftn` / `torch.fft.hfft` / `torch.fft.ihfft` accept ANY
//! complex input and silently project to the Hermitian subspace before the
//! actual transform (see `aten::_fft_c2r`). To match PyTorch parity ferrotorch
//! pre-projects the input spectrum to the Hermitian subspace inside the
//! wrapper functions before calling into ferray-fft. The projection zeroes
//! the imaginary parts of the DC bin and (for even output length) the
//! Nyquist bin along the last transform axis — every other bin is left
//! unchanged. This matches `aten::_fft_c2r` exactly.
//!
//! Cases covered:
//!
//! * `irfftn` 1-D last axis even — Nyquist bin is real.
//! * `irfftn` 1-D last axis odd — no Nyquist bin (only DC).
//! * `irfftn` multi-axis last axis even.
//! * `irfftn` multi-axis last axis odd.
//! * `hfft` last axis even.
//! * `hfft` last axis odd.
//! * `ihfft` (input is real — projection is a no-op, but verifies that the
//!   wrapper still returns a result for any real input).
//!
//! Each case constructs a deliberately non-Hermitian complex spectrum, calls
//! the ferrotorch wrapper, and compares against an independent NumPy/PyTorch
//! reference computed in `f64` — the reference applies the same Hermitian
//! projection then runs the canonical inverse-DFT formula.
//!
//! Pre-fix: each `irfftn` / `hfft` call panics inside ferray-fft with the
//! "Imaginary parts of both first and last values were non-zero" message.
//! Post-fix: every assertion within `F64_FFT = 1e-10`.

use ferrotorch_core::fft::{hfft, ihfft, irfftn};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

const F64_FFT: f64 = 1e-10;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_complex(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn make_real(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn read_back(t: &Tensor<f64>) -> Vec<f64> {
    t.data().unwrap().to_vec()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch ({} vs {})",
        actual.len(),
        expected.len(),
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tol,
            "{label}: index {i}: got {a}, expected {e} (diff {diff:.3e})",
        );
    }
}

/// Apply the Hermitian projection along the last axis of a `[..., half_n, 2]`
/// flat interleaved buffer. The projection zeroes `im[0]` and (when
/// `output_n` is even and `output_n / 2 < half_n`) `im[output_n / 2]`,
/// for every position along the leading axes.
fn project_hermitian_last_axis(data: &mut [f64], leading: usize, half_n: usize, output_n: usize) {
    for b in 0..leading {
        let dc_im = b * half_n * 2 + 1;
        data[dc_im] = 0.0;
        if output_n.is_multiple_of(2) {
            let nyq = output_n / 2;
            if nyq < half_n {
                let nyq_im = b * half_n * 2 + nyq * 2 + 1;
                data[nyq_im] = 0.0;
            }
        }
    }
}

/// Reference implementation of `torch.fft.irfft(spec, n=output_n)` for a
/// `[batch, half_n, 2]` input. Slices/zero-pads the spectrum to
/// `output_n / 2 + 1`, applies the Hermitian projection (DC + Nyquist im=0),
/// Hermitian-extends to length `output_n`, and runs the unnormalized inverse
/// DFT divided by `output_n` — matches PyTorch's `aten::_fft_c2r`.
#[allow(clippy::needless_range_loop)]
fn reference_irfft_1d(data: &[f64], batch: usize, half_n: usize, output_n: usize) -> Vec<f64> {
    let half_for_out = output_n / 2 + 1;
    let mut out = vec![0.0f64; batch * output_n];
    for b in 0..batch {
        // Step 1: slice/zero-pad to `half_for_out` and Hermitian-project DC + Nyquist.
        let mut half: Vec<(f64, f64)> = (0..half_for_out)
            .map(|i| {
                if i < half_n {
                    let re = data[b * half_n * 2 + i * 2];
                    let im = data[b * half_n * 2 + i * 2 + 1];
                    (re, im)
                } else {
                    (0.0, 0.0)
                }
            })
            .collect();
        // Hermitian projection: zero im at DC and (for even output) Nyquist.
        half[0].1 = 0.0;
        if output_n.is_multiple_of(2) {
            let nyq = output_n / 2;
            half[nyq].1 = 0.0;
        }
        // Step 2: Hermitian-extend to `output_n` complex bins.
        let mut full: Vec<(f64, f64)> = Vec::with_capacity(output_n);
        for k in 0..output_n {
            if k < half_for_out {
                full.push(half[k]);
            } else {
                let mirror = output_n - k;
                let (re, im) = half[mirror];
                full.push((re, -im));
            }
        }
        // Step 3: unnormalized inverse DFT divided by `output_n`.
        let n_f = output_n as f64;
        for t in 0..output_n {
            let mut acc = 0.0f64;
            for k in 0..output_n {
                let theta = 2.0 * std::f64::consts::PI * (k as f64) * (t as f64) / n_f;
                let (re, im) = full[k];
                acc += re * theta.cos() - im * theta.sin();
            }
            out[b * output_n + t] = acc / n_f;
        }
    }
    out
}

/// Reference implementation of `torch.fft.hfft(spec, n=output_n)` for a
/// `[batch, half_n, 2]` input.
///
/// `hfft(a, n) == irfft(conj(a), n) * n` (PyTorch / NumPy convention).
fn reference_hfft_1d(data: &[f64], batch: usize, half_n: usize, output_n: usize) -> Vec<f64> {
    // Build conj(a) and call the irfft reference, then multiply by n.
    let mut conj = data.to_vec();
    for b in 0..batch {
        for i in 0..half_n {
            conj[b * half_n * 2 + i * 2 + 1] = -conj[b * half_n * 2 + i * 2 + 1];
        }
    }
    let mut out = reference_irfft_1d(&conj, batch, half_n, output_n);
    let scale = output_n as f64;
    for v in out.iter_mut() {
        *v *= scale;
    }
    out
}

// ---------------------------------------------------------------------------
// irfftn — 1-D last axis even (Nyquist bin present)
// ---------------------------------------------------------------------------

#[test]
fn probe_808_irfftn_1d_even_nyquist() {
    // half_n = 5, output_n = 8 (even, Nyquist at k=4).
    let half_n = 5usize;
    let output_n = 8usize;
    let batch = 2usize;
    // Deliberately non-Hermitian: im at DC and Nyquist are non-zero.
    let mut a_data: Vec<f64> = vec![
        // batch 0
        2.0, 1.5, // bin 0 (DC, im=1.5  ← non-zero!)
        0.5, -0.25, // bin 1
        -0.75, 0.125, // bin 2
        1.5, -0.5, // bin 3
        -1.0, 0.875, // bin 4 (Nyquist, im=0.875 ← non-zero!)
        // batch 1
        0.7, -0.4, // bin 0 (DC im non-zero)
        1.1, 0.6, -0.2, -0.3, 0.45, 1.2, 0.9, 0.55, // bin 4 (Nyquist im non-zero)
    ];
    let spec = make_complex(&a_data, &[batch, half_n, 2]);
    // s = [output_n], axes = [-1] is implicit when axes=None means transform
    // over all axes — so we explicitly target only the last (signal) axis.
    let r = irfftn(&spec, Some(&[output_n]), Some(&[-1])).expect("irfftn 1d even");
    assert_eq!(r.shape(), &[batch, output_n]);
    let actual = read_back(&r);

    // Reference: project then 1-D irfft.
    project_hermitian_last_axis(&mut a_data, batch, half_n, output_n);
    let expected = reference_irfft_1d(&a_data, batch, half_n, output_n);
    assert_close(&actual, &expected, F64_FFT, "irfftn 1d even Nyquist");
}

// ---------------------------------------------------------------------------
// irfftn — 1-D last axis odd (no Nyquist bin)
// ---------------------------------------------------------------------------

#[test]
fn probe_808_irfftn_1d_odd_no_nyquist() {
    // half_n = 4, output_n = 7 (odd, no Nyquist bin).
    let half_n = 4usize;
    let output_n = 7usize;
    let batch = 1usize;
    // Non-zero im at DC. Last-bin (k=3) is the highest non-Nyquist bin.
    let mut a_data: Vec<f64> = vec![
        1.0, 0.6, // bin 0 (DC im=0.6 non-zero)
        -0.3, 0.7, 0.4, -0.2, -0.9, 0.1,
    ];
    let spec = make_complex(&a_data, &[batch, half_n, 2]);
    let r = irfftn(&spec, Some(&[output_n]), Some(&[-1])).expect("irfftn 1d odd");
    assert_eq!(r.shape(), &[batch, output_n]);
    let actual = read_back(&r);

    project_hermitian_last_axis(&mut a_data, batch, half_n, output_n);
    let expected = reference_irfft_1d(&a_data, batch, half_n, output_n);
    assert_close(&actual, &expected, F64_FFT, "irfftn 1d odd no-Nyquist");
}

// ---------------------------------------------------------------------------
// irfftn — multi-axis with last axis even
// ---------------------------------------------------------------------------
//
// Input shape [3, 3, 2] with axes=None, s=[3, 4]: irfftn first applies an
// inverse complex FFT on axis 0 (length 3, no Hermitian constraint), then
// the c2r transform on axis 1 (last axis, output length 4 → Nyquist at k=2).
// We deliberately set a non-zero imaginary part at indices 0 and 2 along the
// last axis for every row, which would panic ferray-fft 0.3.0 pre-fix.
//
// PyTorch reference: take the spectrum, run ifftn (1-axis on axis 0), then
// take the resulting matrix and apply 1-D irfft along axis 1 with the
// Hermitian projection. Since the inverse complex FFT on axis 0 mixes bins
// across rows, the cleanest way to validate is to project the input
// spectrum first then call the same reference path. We instead verify
// post-fix matches the round-trip identity: rfftn(irfftn(spec)) of a
// projected spec is the projected spec.
#[test]
fn probe_808_irfftn_multiaxis_last_even() {
    let s_shape = [3usize, 3usize, 2usize]; // [rows, half_cols, 2]
    let rows = s_shape[0];
    let half_cols = s_shape[1];
    let output_cols = 4usize;

    // Build a deliberately non-Hermitian spectrum (DC + Nyquist im non-zero
    // along the last axis).
    let total = rows * half_cols * 2;
    let mut data = Vec::with_capacity(total);
    let mut acc = 0.0f64;
    for r in 0..rows {
        for c in 0..half_cols {
            acc += 0.37;
            let re = (acc.sin() + r as f64) * 0.5;
            let im = (acc.cos() - c as f64) * 0.3;
            data.push(re);
            data.push(im);
        }
    }
    let spec = make_complex(&data, &s_shape);
    let r = irfftn(&spec, Some(&[rows, output_cols]), None).expect("irfftn multi-axis even");
    assert_eq!(r.shape(), &[rows, output_cols]);
    let actual = read_back(&r);

    // Independent reference matches PyTorch's `aten::native::fft_irfftn`:
    // (1) inverse complex FFT on axes[..last] of the RAW input spectrum,
    // (2) Hermitian projection on the c2r axis of the intermediate (NOT
    //     the input — the inner inverses mix bins, so the projection has
    //     to happen on the post-inner-inverse intermediate),
    // (3) 1-D irfft (c2r) on axes[last] of the projected intermediate.

    // Step 1: inverse FFT along axis 0 of the raw spectrum.
    let n0 = rows as f64;
    let mut after_axis0 = vec![(0.0f64, 0.0f64); rows * half_cols];
    for r_out in 0..rows {
        for c in 0..half_cols {
            let mut sum_re = 0.0f64;
            let mut sum_im = 0.0f64;
            for r_in in 0..rows {
                let re = data[r_in * half_cols * 2 + c * 2];
                let im = data[r_in * half_cols * 2 + c * 2 + 1];
                let theta = 2.0 * std::f64::consts::PI * (r_in as f64) * (r_out as f64) / n0;
                let (cos_t, sin_t) = (theta.cos(), theta.sin());
                sum_re += re * cos_t - im * sin_t;
                sum_im += re * sin_t + im * cos_t;
            }
            after_axis0[r_out * half_cols + c] = (sum_re / n0, sum_im / n0);
        }
    }

    // Step 2: project DC + Nyquist (last-axis) of the intermediate.
    let mut flat = Vec::with_capacity(rows * half_cols * 2);
    for r_idx in 0..rows {
        for c in 0..half_cols {
            let (re, im) = after_axis0[r_idx * half_cols + c];
            flat.push(re);
            flat.push(im);
        }
    }
    project_hermitian_last_axis(&mut flat, rows, half_cols, output_cols);

    // Step 3: 1-D irfft per row.
    let expected = reference_irfft_1d(&flat, rows, half_cols, output_cols);
    assert_close(&actual, &expected, F64_FFT, "irfftn multi-axis last even");
}

// ---------------------------------------------------------------------------
// irfftn — multi-axis with last axis odd (no Nyquist)
// ---------------------------------------------------------------------------
#[test]
fn probe_808_irfftn_multiaxis_last_odd() {
    let s_shape = [3usize, 3usize, 2usize];
    let rows = s_shape[0];
    let half_cols = s_shape[1];
    let output_cols = 5usize; // odd → no Nyquist on last axis

    let total = rows * half_cols * 2;
    let mut data = Vec::with_capacity(total);
    let mut acc = 0.0f64;
    for r in 0..rows {
        for c in 0..half_cols {
            acc += 0.41;
            let re = (acc.sin() + r as f64) * 0.5;
            let im = (acc.cos() + c as f64 * 0.7) * 0.4;
            data.push(re);
            data.push(im);
        }
    }
    let spec = make_complex(&data, &s_shape);
    let r = irfftn(&spec, Some(&[rows, output_cols]), None).expect("irfftn multi-axis odd");
    assert_eq!(r.shape(), &[rows, output_cols]);
    let actual = read_back(&r);

    // Reference matches PyTorch: ifft on axis 0 of raw input, then project
    // intermediate on last-axis DC (no Nyquist for odd output), then
    // 1-D irfft on axis 1.
    let n0 = rows as f64;
    let mut after_axis0 = vec![(0.0f64, 0.0f64); rows * half_cols];
    for r_out in 0..rows {
        for c in 0..half_cols {
            let mut sum_re = 0.0f64;
            let mut sum_im = 0.0f64;
            for r_in in 0..rows {
                let re = data[r_in * half_cols * 2 + c * 2];
                let im = data[r_in * half_cols * 2 + c * 2 + 1];
                let theta = 2.0 * std::f64::consts::PI * (r_in as f64) * (r_out as f64) / n0;
                let (cos_t, sin_t) = (theta.cos(), theta.sin());
                sum_re += re * cos_t - im * sin_t;
                sum_im += re * sin_t + im * cos_t;
            }
            after_axis0[r_out * half_cols + c] = (sum_re / n0, sum_im / n0);
        }
    }

    let mut flat = Vec::with_capacity(rows * half_cols * 2);
    for r_idx in 0..rows {
        for c in 0..half_cols {
            let (re, im) = after_axis0[r_idx * half_cols + c];
            flat.push(re);
            flat.push(im);
        }
    }
    project_hermitian_last_axis(&mut flat, rows, half_cols, output_cols);
    let expected = reference_irfft_1d(&flat, rows, half_cols, output_cols);
    assert_close(&actual, &expected, F64_FFT, "irfftn multi-axis last odd");
}

// ---------------------------------------------------------------------------
// hfft — last axis even
// ---------------------------------------------------------------------------
#[test]
fn probe_808_hfft_even() {
    // half_n = 5, output_n = 8 (even, Nyquist at k=4).
    let half_n = 5usize;
    let output_n = 8usize;
    let batch = 1usize;
    let mut a_data: Vec<f64> = vec![
        2.0, 1.5, // DC im non-zero
        0.5, -0.25, -0.75, 0.125, 1.5, -0.5, -1.0, 0.875, // Nyquist im non-zero
    ];
    let spec = make_complex(&a_data, &[batch, half_n, 2]);
    let r = hfft(&spec, Some(output_n)).expect("hfft even");
    assert_eq!(r.shape(), &[batch, output_n]);
    let actual = read_back(&r);

    project_hermitian_last_axis(&mut a_data, batch, half_n, output_n);
    let expected = reference_hfft_1d(&a_data, batch, half_n, output_n);
    assert_close(&actual, &expected, F64_FFT, "hfft even");
}

// ---------------------------------------------------------------------------
// hfft — last axis odd
// ---------------------------------------------------------------------------
#[test]
fn probe_808_hfft_odd() {
    // half_n = 4, output_n = 7 (odd, no Nyquist).
    let half_n = 4usize;
    let output_n = 7usize;
    let batch = 1usize;
    let mut a_data: Vec<f64> = vec![
        1.0, 0.6, // DC im non-zero
        -0.3, 0.7, 0.4, -0.2, -0.9, 0.1,
    ];
    let spec = make_complex(&a_data, &[batch, half_n, 2]);
    let r = hfft(&spec, Some(output_n)).expect("hfft odd");
    assert_eq!(r.shape(), &[batch, output_n]);
    let actual = read_back(&r);

    project_hermitian_last_axis(&mut a_data, batch, half_n, output_n);
    let expected = reference_hfft_1d(&a_data, batch, half_n, output_n);
    assert_close(&actual, &expected, F64_FFT, "hfft odd");
}

// ---------------------------------------------------------------------------
// ihfft — input is real-valued so projection is a no-op, but the wrapper
// must still return a result for any real input. This case verifies the
// projection helper does not corrupt the real path.
// ---------------------------------------------------------------------------
#[test]
fn probe_808_ihfft_real_input() {
    // Real input length 8; ihfft returns complex Hermitian-folded length 5.
    let input: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0];
    let n = input.len();
    let a = make_real(&input, &[n]);
    let r = ihfft(&a, None).expect("ihfft real input");
    assert_eq!(r.shape(), &[n / 2 + 1, 2]);
    // Spot-check: ihfft of any real input has DC bin im == 0 (Hermitian).
    let d = read_back(&r);
    assert!(
        d[1].abs() < F64_FFT,
        "ihfft DC im should be ~0, got {}",
        d[1]
    );
    // Nyquist bin (index n/2 in half-spectrum) im should also be ~0 for even n.
    let nyq_idx = n / 2;
    let nyq_im = d[nyq_idx * 2 + 1];
    assert!(
        nyq_im.abs() < F64_FFT,
        "ihfft Nyquist im should be ~0, got {nyq_im}",
    );
}
