//! Permanent regression sentinel for #807 — irfft CPU pad/truncate semantics
//! (Bugfix Batch 7 / Dispatch A2).
//!
//! ## #807 — irfft CPU path mishandles `n` mismatch with `half_n`
//!
//! PyTorch's `torch.fft.irfft(spec, n=output_n)` always slices / zero-pads the
//! input spectrum to the canonical Hermitian half-size for the requested
//! output length, namely `output_n / 2 + 1`, *before* Hermitian extension and
//! the inverse FFT. Concretely:
//!
//! * If `spec.shape[-2] > output_n / 2 + 1` — truncate to the first
//!   `output_n / 2 + 1` bins.
//! * If `spec.shape[-2] < output_n / 2 + 1` — zero-pad to
//!   `output_n / 2 + 1` bins.
//! * If `spec.shape[-2] == output_n / 2 + 1` — identity (no slicing).
//!
//! Pre-fix, `ferrotorch-core/src/fft.rs:419-444` instead copied
//! `min(half_n, output_n)` raw bins from the source spectrum and then
//! Hermitian-mirrored only the `output_n - half_n` (potentially negative)
//! tail. For the canonical fixture `len_8_pad_to_16` (input shape
//! `[2, 9, 2]`, `output_n = 8`) this copied 8 of the 9 spectrum bins
//! verbatim — the result is not the IFFT of any Hermitian spectrum and
//! diverges from PyTorch by O(1) magnitude.
//!
//! Post-fix, the slicing logic computes `half_n_for_output = output_n / 2 + 1`,
//! copies up to that many bins (zero-padding the rest), then Hermitian-extends
//! the resulting `half_n_for_output` half-spectrum to length `output_n` and
//! inverse-FFTs. This matches PyTorch's `aten::_fft_c2r` semantics.
//!
//! Cases covered:
//!
//! 1. `n == 2 * (half_n - 1)` — identity case (no slicing). `half_n = 5`,
//!    `output_n = 8`.
//! 2. `n` smaller than `2 * (half_n - 1)` — truncate spectrum. `half_n = 9`,
//!    `output_n = 8`. **This is the cited fixture** `len_8_pad_to_16`.
//! 3. `n` larger than `2 * (half_n - 1)` — zero-pad spectrum. `half_n = 5`,
//!    `output_n = 16`.
//! 4. Odd `n` — no Nyquist bin. `half_n = 6`, `output_n = 7` (truncate from
//!    `output_n / 2 + 1 = 4`).
//!
//! All cases are checked against a NumPy / PyTorch reference computed by
//! taking the first `output_n / 2 + 1` bins of the spectrum, Hermitian-
//! extending to `output_n` complex bins, and running an unnormalized inverse
//! DFT divided by `output_n` — exactly the PyTorch / SciPy convention. The
//! reference uses `f64` arithmetic and the assertion uses `F64_FFT = 1e-10`.

use ferrotorch_core::fft::irfft;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

const F64_FFT: f64 = 1e-10;

/// Build a `[batch, half_n, 2]` interleaved-complex tensor from a flat
/// `(re, im, re, im, …)` slice.
fn make_spec_f64(data: &[f64], batch: usize, half_n: usize) -> Tensor<f64> {
    assert_eq!(data.len(), batch * half_n * 2);
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        vec![batch, half_n, 2],
        false,
    )
    .unwrap()
}

/// Reference implementation of `torch.fft.irfft(spec, n=output_n)` for a
/// `[batch, half_n, 2]` input. Uses the unnormalized inverse-DFT-of-the-
/// Hermitian-extension formula; matches PyTorch's `aten::_fft_c2r`.
#[allow(clippy::needless_range_loop)]
fn reference_irfft(
    data: &[f64],
    batch: usize,
    half_n: usize,
    output_n: usize,
) -> Vec<f64> {
    let half_for_out = output_n / 2 + 1;
    let mut out = vec![0.0f64; batch * output_n];
    for b in 0..batch {
        // Step 1: slice / zero-pad spectrum to `half_for_out` bins.
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
        // PyTorch zeroes the imaginary part of DC (and Nyquist when `output_n`
        // is even) before the c2r IFFT; we don't impose that here because the
        // probe inputs already have those imaginary parts equal to zero.
        // (The fixture `len_8_pad_to_16` has `im[0] == im[Nyquist] == 0`.)
        // We still mirror the bins straightforwardly.
        let _ = &mut half;
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
        // Step 3: unnormalized inverse DFT, then divide by `output_n` (the
        // PyTorch / NumPy normalization for `norm="backward"`, the default).
        let n_f = output_n as f64;
        for t in 0..output_n {
            let mut acc = 0.0f64;
            for k in 0..output_n {
                let theta =
                    2.0 * std::f64::consts::PI * (k as f64) * (t as f64) / n_f;
                let (re, im) = full[k];
                // x[t] = (1/N) Σ_k X[k] exp(+i 2π k t / N).
                // (re + i im) * (cos θ + i sin θ) = (re cos θ - im sin θ) + i(...)
                acc += re * theta.cos() - im * theta.sin();
            }
            out[b * output_n + t] = acc / n_f;
        }
    }
    out
}

fn read_back_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.data().unwrap().to_vec()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}: index {i}: got {a}, expected {e} (diff {})",
            (a - e).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// The cited fixture: `len_8_pad_to_16`. Pre-fix this assert fires.
// ---------------------------------------------------------------------------

#[test]
fn probe_807_irfft_len_8_pad_to_16_truncate() {
    // `a_data` lifted verbatim from
    // `ferrotorch-core/tests/conformance/fixtures/fft.json` (op="irfft",
    // tag="len_8_pad_to_16", dtype="float64", device="cpu"). `out_values`
    // there is the PyTorch reference; we re-derive it independently below to
    // make this probe self-contained against fixture drift.
    let a_data: [f64; 36] = [
        -2.2791919605244897, 0.0,
        2.710971387839778, -0.3985263039298929,
        -4.941980720701352, -2.206674262762721,
        1.2688113543971502, 5.8095557246415375,
        2.6519951430991187, -3.444751895364187,
        -0.6703435547949778, 0.38770013469285347,
        0.6697259766842323, -4.487188402382817,
        -5.337637613671956, 0.6959291679256558,
        1.4633143099004813, 0.0,
        1.858235109863668, 0.0,
        -2.488430956080993, -2.224062319779218,
        -2.0831563290657047, -0.15278921884620034,
        -4.458411766657262, 0.9169045686293953,
        1.1216939191945214, 4.157918087168396,
        -0.18285250383635665, -3.119408805299979,
        -3.046947299467702, 3.176234769421063,
        2.555117559260311, -0.5747461039560797,
        -2.990571025814498, 0.0,
    ];
    let batch = 2;
    let half_n = 9;
    let output_n = 8;

    let spec = make_spec_f64(&a_data, batch, half_n);
    let r = irfft(&spec, Some(output_n)).expect("irfft");
    assert_eq!(r.shape(), &[batch, output_n]);
    let actual = read_back_f64(&r);
    let expected = reference_irfft(&a_data, batch, half_n, output_n);
    assert_close(&actual, &expected, F64_FFT, "len_8_pad_to_16 (#807)");
}

// ---------------------------------------------------------------------------
// Edge case 1: identity — `output_n == 2 * (half_n - 1)`.
// ---------------------------------------------------------------------------

#[test]
fn probe_807_irfft_identity_no_slice() {
    // half_n = 5, output_n = 8 → `output_n / 2 + 1 == half_n`, no slicing.
    let half_n = 5usize;
    let output_n = 8usize;
    let batch = 1usize;
    let a_data: Vec<f64> = vec![
        // bin 0 (DC, im=0)
        2.0, 0.0,
        // bin 1
        0.5, -0.25,
        // bin 2
        -0.75, 0.125,
        // bin 3
        1.5, -0.5,
        // bin 4 (Nyquist, im=0)
        -1.0, 0.0,
    ];
    let spec = make_spec_f64(&a_data, batch, half_n);
    let r = irfft(&spec, Some(output_n)).expect("irfft");
    let actual = read_back_f64(&r);
    let expected = reference_irfft(&a_data, batch, half_n, output_n);
    assert_close(&actual, &expected, F64_FFT, "identity n=2*(half_n-1)");
}

// ---------------------------------------------------------------------------
// Edge case 2: zero-pad — `output_n > 2 * (half_n - 1)`.
// ---------------------------------------------------------------------------

#[test]
fn probe_807_irfft_zero_pad_spectrum() {
    // half_n = 5, output_n = 16 → `output_n / 2 + 1 = 9`, must zero-pad
    // bins 5..9 before Hermitian extension.
    let half_n = 5usize;
    let output_n = 16usize;
    let batch = 1usize;
    let a_data: Vec<f64> = vec![
        1.0, 0.0,
        -0.3, 0.7,
        0.4, -0.2,
        -0.9, 0.1,
        0.5, 0.0,
    ];
    let spec = make_spec_f64(&a_data, batch, half_n);
    let r = irfft(&spec, Some(output_n)).expect("irfft");
    let actual = read_back_f64(&r);
    let expected = reference_irfft(&a_data, batch, half_n, output_n);
    assert_close(&actual, &expected, F64_FFT, "zero-pad to n=16");
}

// ---------------------------------------------------------------------------
// Edge case 3: odd `output_n` — no Nyquist bin.
// ---------------------------------------------------------------------------

#[test]
fn probe_807_irfft_odd_n() {
    // half_n = 6, output_n = 7 → `output_n / 2 + 1 = 4`, truncate from 6 → 4.
    let half_n = 6usize;
    let output_n = 7usize;
    let batch = 1usize;
    let a_data: Vec<f64> = vec![
        0.5, 0.0,
        1.2, -0.3,
        -0.4, 0.8,
        0.9, -0.7,
        0.2, 0.4,
        -0.6, 0.0,
    ];
    let spec = make_spec_f64(&a_data, batch, half_n);
    let r = irfft(&spec, Some(output_n)).expect("irfft");
    let actual = read_back_f64(&r);
    let expected = reference_irfft(&a_data, batch, half_n, output_n);
    assert_close(&actual, &expected, F64_FFT, "odd n=7");
}

// ---------------------------------------------------------------------------
// Cross-check: the cited fixture's `out_values` field matches our reference,
// guarding against a mistake in the manual reference implementation above.
// ---------------------------------------------------------------------------

#[test]
fn probe_807_reference_matches_fixture_out_values() {
    let a_data: [f64; 36] = [
        -2.2791919605244897, 0.0,
        2.710971387839778, -0.3985263039298929,
        -4.941980720701352, -2.206674262762721,
        1.2688113543971502, 5.8095557246415375,
        2.6519951430991187, -3.444751895364187,
        -0.6703435547949778, 0.38770013469285347,
        0.6697259766842323, -4.487188402382817,
        -5.337637613671956, 0.6959291679256558,
        1.4633143099004813, 0.0,
        1.858235109863668, 0.0,
        -2.488430956080993, -2.224062319779218,
        -2.0831563290657047, -0.15278921884620034,
        -4.458411766657262, 0.9169045686293953,
        1.1216939191945214, 4.157918087168396,
        -0.18285250383635665, -3.119408805299979,
        -3.046947299467702, 3.176234769421063,
        2.555117559260311, -0.5747461039560797,
        -2.990571025814498, 0.0,
    ];
    // PyTorch out_values from fft.json (op=irfft, tag=len_8_pad_to_16,
    // dtype=float64, device=cpu).
    let pytorch_out: [f64; 16] = [
        -0.1939490967942774,
        -0.7663334366076758,
        2.834116085140024,
        -2.379551137590786,
        -2.1838404679127414,
        0.6368737920831341,
        -0.2699249291456911,
        0.043417230303523846,
        -1.8850086343187162,
        0.7095866785163056,
        1.6785219330008534,
        -0.06330132588972025,
        1.5884127270504114,
        -0.4490567714259188,
        0.10803848879654654,
        0.17104201413390674,
    ];
    let our_ref = reference_irfft(&a_data, 2, 9, 8);
    assert_close(
        &our_ref,
        &pytorch_out,
        F64_FFT,
        "reference vs fixture out_values",
    );
}
