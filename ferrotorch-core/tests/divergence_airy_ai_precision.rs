//! Precision audit of `ferrotorch_core::airy_ai` (CPU, f32) at the op_db
//! `special.airy_ai` sample — the parity-sweep surfaced an apparent divergence
//! at index 4 of the `[20]` sample (x = -7.950586795806885f32, oscillatory
//! region x < -2.09).
//!
//! ## Verdict: B — ferrotorch is MORE precise than torch; ACCEPTED per contract.
//!
//! The parity-sweep flagged: ferrotorch returns `-0.006113018`, live torch 2.11
//! returns `-0.0061132023`, |diff| ≈ 1.84e-7, just past the transcendental
//! envelope (atol 1e-7 + rtol 1e-5). Per the precision contract, ferrotorch
//! being *more* precise than torch is NOT a bug — so the FIRST step is deciding
//! which side is correct against a high-precision oracle independent of both.
//!
//! ### Gold reference: mpmath at 50 digits
//! ```text
//! $ python3
//! >>> import struct
//! >>> from mpmath import airyai, mp, mpf
//! >>> mp.dps = 50
//! >>> x = struct.unpack('<f', struct.pack('<I', 0xc0fe6b35))[0]   # f32 idx-4
//! >>> x
//! -7.950586795806885
//! >>> airyai(mpf(x))
//! mpf('-0.0061130178225984497843270619915794439430476139028158')
//! ```
//! Recovering the exact op_db input (torch 2.11):
//! ```text
//! >>> from torch.testing._internal.common_methods_invocations import op_db
//! >>> op = next(o for o in op_db if o.name == "special.airy_ai")
//! >>> s = next(s for s in op.sample_inputs("cpu", torch.float32)
//! ...          if torch.is_tensor(s.input) and s.input.numel() == 20)
//! >>> s.input.reshape(-1)[4].item()
//! -7.950586795806885                 # bits 0xc0fe6b35
//! >>> torch.special.airy_ai(s.input).reshape(-1)[4].item()
//! -0.006113202311098576
//! ```
//!
//! ### The three values at x = -7.950586795806885f32
//! | source       | value (f32)        | |value − truth| (as f64) |
//! |--------------|--------------------|--------------------------|
//! | mpmath truth | -0.006113017822... | 0 (reference)            |
//! | ferrotorch   | -0.006113018       | 8.66e-11                 |
//! | torch 2.11   | -0.0061132023      | 1.844e-07                |
//!
//! ferrotorch is ~2130x CLOSER to the mathematically-true value than torch.
//! torch's CUDA-jiterator-derived f32 `airy_ai` carries the error; ferrotorch's
//! f64-then-narrow Cephes evaluator (`special.rs::airy_ai_f64`, the oscillatory
//! `x < -2.09` AFN/AFD + AGN/AGD asymptotic branch at `special.rs:1640-1666`,
//! mirroring `aten/src/ATen/native/cuda/Math.cuh:1372-1402`) is correct.
//!
//! This is therefore a PASSING regression-guard, NOT a failing divergence test.
//! It pins ferrotorch to the mpmath truth across the oscillatory, central, and
//! decaying regions, and asserts the "more precise than torch" property at the
//! offending op_db point so any future regression that drags ferrotorch toward
//! torch's larger error fails here.
//!
//! Test-infra follow-up (for the orchestrator, NOT done here): the parity-sweep
//! per-op tolerance for `special.airy_ai` should be widened — the current
//! envelope flags ferrotorch for being more accurate than the torch oracle it
//! is compared against. The right oracle for this op's f32 path is mpmath, not
//! torch's f32 output.

use ferrotorch_core::TensorStorage;
use ferrotorch_core::airy_ai;
use ferrotorch_core::tensor::Tensor;

/// (x_f32, Ai(x) rounded to f32 from the mpmath 50-digit oracle).
/// Every `true` value below is `f32(float(airyai(mpf(x))))` with `mp.dps = 50`
/// (R-CHAR-3: oracle-derived, never copied from the ferrotorch side). The grid
/// spans the oscillatory region (x < -2.09), the central Maclaurin region, and
/// the decaying-asymptotic region (x >= 2.09).
const GRID: &[(f32, f32)] = &[
    // The op_db special.airy_ai sample[20] index-4 input (the surfaced point).
    (-7.950586795806885, -0.006113017909228802),
    (-7.0, 0.18428084254264832),
    (-5.0, 0.3507609963417053),
    (-3.0, -0.3788142800331116),
    (-2.5, -0.11232506483793259),
    (-1.0, 0.5355609059333801),
    (0.0, 0.35502806305885315),
    (1.0, 0.1352924108505249),
    (3.0, 0.0065911393612623215),
    (6.0, 9.94769470707979e-06),
];

/// ferrotorch's `airy_ai` matches the mpmath 50-digit truth across all three
/// regions to within a tight f32 envelope. The chosen tolerance (atol 5e-7 +
/// rtol 5e-6) is tighter than the parity-sweep transcendental gate yet still
/// honors the f32 representable limit; ferrotorch's worst observed |error| on
/// this grid is 2.34e-7 (at x = -2.5), inside the band. A regression that
/// degrades the oscillatory branch toward torch's error would fail here.
#[test]
fn airy_ai_f32_matches_mpmath_truth() {
    let xs: Vec<f32> = GRID.iter().map(|&(x, _)| x).collect();
    let input =
        Tensor::from_storage(TensorStorage::cpu(xs.clone()), vec![xs.len()], false).unwrap();
    let out = airy_ai(&input).unwrap();
    let got = out.data().unwrap();

    for (i, &(x, truth)) in GRID.iter().enumerate() {
        let g = got[i];
        let err = (g - truth).abs();
        let tol = 5e-7 + 5e-6 * truth.abs();
        assert!(
            err <= tol,
            "airy_ai f32 idx {i} x={x}: ferrotorch={g} mpmath_truth={truth} \
             |err|={err:.3e} tol={tol:.3e}",
        );
    }
}

/// Pin the "more precise than torch" property at the op_db sample[20] index-4
/// point. ferrotorch must be at least as close to the mpmath truth as torch is.
/// The torch witness is the live torch 2.11 output recorded in the module doc
/// (R-CHAR-3: it is torch's value, NOT ferrotorch's — we assert ferrotorch
/// strictly beats it). Guards against any future regression that would make
/// ferrotorch inherit torch's ~1.84e-7 oscillatory-region error.
#[test]
fn airy_ai_more_precise_than_torch_at_opdb_index4() {
    // x = op_db special.airy_ai sample[20] index 4, exact f32 (bits 0xc0fe6b35).
    let x = f32::from_bits(0xc0fe_6b35);
    assert_eq!(x, -7.950586795806885, "input reconstruction");

    // mpmath airyai(mpf(x)) @ mp.dps=50, as f64 (the gold reference).
    let truth: f64 = -0.0061130178225984497843;
    // Live torch 2.11 `torch.special.airy_ai` f32 output at this x (witness).
    let torch_val: f64 = -0.006113202311098576;

    let input = Tensor::from_storage(TensorStorage::cpu(vec![x]), vec![1], false).unwrap();
    let ferro = airy_ai(&input).unwrap().data().unwrap()[0] as f64;

    let ferro_err = (ferro - truth).abs();
    let torch_err = (torch_val - truth).abs();

    assert!(
        ferro_err <= torch_err,
        "ferrotorch must be at least as close to mpmath truth as torch: \
         ferro={ferro} (|err|={ferro_err:.3e}) torch={torch_val} (|err|={torch_err:.3e}) \
         truth={truth}",
    );
    // Sanity: ferrotorch is comfortably inside the f32 transcendental envelope
    // around the truth (~8.7e-11), proving it is the correct side.
    assert!(
        ferro_err < 1e-8,
        "ferrotorch within tight f32 envelope of truth: |err|={ferro_err:.3e}",
    );
}
