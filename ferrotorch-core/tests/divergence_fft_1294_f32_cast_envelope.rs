//! Tolerance-justification audit for commit cffd5117d (#1294).
//!
//! The builder widened the FFT-family `tolerance_for` to rtol=1e-4
//! (`tools/parity-sweep/runner/src/main.rs:6396-6400`), citing
//! `fft2 seed=6 i=5 diff=3.4e-6 at |e|=0.034 → relative 1e-4`. The user has
//! previously rejected gratuitous tolerance-loosening, so the audit must
//! confirm the widening is forced by genuine f32 cast noise, NOT masking an
//! algorithmic error.
//!
//! This test pins the structural fact that justifies rtol=1e-4: ferrotorch's
//! **f32** fft2 (which computes the butterfly in f64 via ferray_fft, then
//! casts to f32) matches torch's **float64** fft2 reference to a tight
//! ABSOLUTE envelope (a few e-6 — pure f32 round-trip noise). The relative
//! error only blows up at near-zero bins, which is exactly why an absolute
//! framing (or the widened rtol) is correct. torch's own f32-vs-f64 fft2 on
//! this signal differs by up to ~7.5e-7 abs (measured via the oracle), so an
//! abs tolerance of 5e-5 is comfortably above the inherent f32 floor while
//! being ~9 orders of magnitude tighter than what would hide a real norm/dim
//! scale bug.
//!
//! Reference output is torch float64/complex128 (torch 2.11.0,
//! `torch.fft.fft2(x.double())`), NOT copied from ferrotorch (R-CHAR-3).
//! This test PASSES under the current implementation — it documents that the
//! rtol=1e-4 widening is justified cast noise, not a masked divergence.

use ferrotorch_core::{FftNorm, fft2_norm, from_vec};

/// Absolute envelope: torch's own f32 fft2 on this signal departs from its
/// f64 fft2 by <= 7.6e-7 (oracle-measured). ferrotorch's f64-butterfly +
/// f32-cast path adds one rounding step. 5e-5 abs is the f32 cast envelope;
/// a wrong norm/dim scale (a constant multiplicative factor) would blow this
/// by orders of magnitude.
const ABS_ENVELOPE: f64 = 5e-5;

#[test]
fn fft2_f32_path_within_cast_envelope_of_torch_float64() {
    // torch.manual_seed(6); x = torch.randn(5,6, float32). Exact f32 bits.
    let input_f32: Vec<f32> = vec![
        -1.211_304_5,
        0.630_358_6,
        -1.471_303_9,
        -1.335_199,
        -0.489_666_88,
        0.131_742_13,
        0.329_497_07,
        0.326_429_3,
        -0.480_550_77,
        1.103_160_1,
        2.548_506,
        0.300_635_37,
        -0.543_218_2,
        -1.084_129_5,
        0.867_176_2,
        -0.073_806_45,
        1.953_842_9,
        -0.446_028_95,
        1.710_205_8,
        0.894_446,
        -0.545_832_34,
        -0.641_804_2,
        -0.789_924_3,
        0.252_545_06,
        -0.696_875,
        -0.004_699_554_3,
        -0.313_625_8,
        -1.260_157_3,
        0.697_658_3,
        0.372_038_04,
    ];
    // Build interleaved complex f32 [5, 6, 2] with imag = 0.
    let mut data = Vec::with_capacity(input_f32.len() * 2);
    for &v in &input_f32 {
        data.push(v);
        data.push(0.0f32);
    }
    let input = from_vec(data, &[5, 6, 2]).unwrap();
    let out = fft2_norm(&input, None, None, FftNorm::Backward).unwrap();
    assert_eq!(out.shape(), &[5, 6, 2]);

    // torch.fft.fft2(x.double()) — float64 reference, interleaved re/im.
    let ref64: [f64; 60] = [
        0.7301141358911991,
        0.0,
        1.494640627875924,
        4.947671923585598,
        -4.294309655204415,
        -5.21003127608424,
        2.399054791778326,
        0.0,
        -4.294309655204415,
        5.21003127608424,
        1.494640627875924,
        -4.947671923585598,
        -4.099206360641652,
        -4.951340135291354,
        3.0873006894721757,
        6.621405537523197,
        -4.580265268873216,
        -2.5502921485498136,
        -5.2484152695615665,
        -2.4414710972803824,
        -0.5416762855228305,
        0.8022793510357263,
        -2.2313976708443994,
        4.609767888470289,
        -5.6292848357672165,
        -3.3305851269354894,
        1.2101970132675297,
        -7.162542872228951,
        -1.3497642775840974,
        -2.6512665542248177,
        -2.44905481985911,
        3.7648326062446102,
        1.0306730367446297,
        -4.804183642884126,
        3.8664104316255776,
        -2.2947739307046486,
        -5.6292848357672165,
        3.3305851269354894,
        3.8664104316255776,
        2.2947739307046486,
        1.0306730367446297,
        4.804183642884126,
        -2.44905481985911,
        -3.7648326062446102,
        -1.3497642775840974,
        2.6512665542248177,
        1.2101970132675297,
        7.162542872228951,
        -4.099206360641652,
        4.951340135291354,
        -2.2313976708443994,
        -4.609767888470289,
        -0.5416762855228305,
        -0.8022793510357263,
        -5.2484152695615665,
        2.4414710972803824,
        -4.580265268873216,
        2.5502921485498136,
        3.0873006894721757,
        -6.621405537523197,
    ];

    let got = out.data().unwrap();
    assert_eq!(got.len(), ref64.len());
    let mut worst = 0.0f64;
    let mut wi = 0usize;
    for (i, (&a, &e)) in got.iter().zip(ref64.iter()).enumerate() {
        let d = (a as f64 - e).abs();
        if d > worst {
            worst = d;
            wi = i;
        }
    }
    assert!(
        worst < ABS_ENVELOPE,
        "fft2 f32 path abs diff {worst:.3e} at index {wi} \
         (ferrotorch_f32={}, torch_f64={}) exceeds the f32 cast envelope \
         {ABS_ENVELOPE:.1e} — a norm/dim scale bug, not cast noise",
        got[wi],
        ref64[wi],
    );
}
