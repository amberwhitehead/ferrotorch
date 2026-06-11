//! Discrimination audit for commit `381fad746` — the FINAL 7 linalg
//! gauge-invariant / unique runner arms (#1344): `solve` / `qr` / `cholesky` /
//! `inv` / `det` / `slogdet` / `cross`, plus the `addr` beta==0 skip removal.
//!
//! These tests do NOT re-run the `parity-sweep` binary's sweep loop; they
//! re-implement, in a host test, the EXACT comparison the runner performs
//! (ferrotorch's own derived quantity vs. the live-torch oracle's derived
//! quantity, under the runner's linalg tolerance `rtol=1e-4, atol=1e-6`). The
//! point is to verify the comparison is REAL and TIGHT, NOT too-loose /
//! tautological:
//!
//!   1. NON-TAUTOLOGY (R-CHAR-3 (a)): the "expected" value is fetched from a
//!      live torch subprocess (`tools/parity-sweep/oracle.py`, `execute` cmd),
//!      NEVER derived from ferrotorch's own output.
//!
//!   2. DISCRIMINATION: a deliberately-WRONG ferrotorch result (perturbed
//!      solution / wrong L entry / sign-flipped slogdet / negated determinant
//!      / wrong cross component / non-reconstructing Q@R) must FAIL the same
//!      gate, proving the comparison would catch a genuinely-wrong forward
//!      rather than rubber-stamp it.
//!
//! Oracle availability (CORE-206 / #1900, fail-closed gate in
//! `tests/common/mod.rs`): if the oracle (python3 + torch) is unavailable the
//! tests print a single-line `VACUOUS-PASS:` marker and soft-skip — UNLESS
//! `PARITY_ORACLE_REQUIRED=1` is set (as in the nightly parity-smoke step),
//! in which case they PANIC with diagnostics. Run explicitly with:
//!   LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH" \
//!     PARITY_ORACLE_REQUIRED=1 \
//!     cargo test -p parity-sweep-runner --test divergence_linalg_final_gauge \
//!     -- --nocapture
//!
//! Upstream sites mirrored (the runner-arm doc-comments + design doc rows):
//!   - `torch.linalg.solve`    : aten/src/ATen/native/BatchLinearAlgebra.cpp
//!   - `torch.linalg.qr`       : torch/linalg/__init__.py (qr = _add_docstr(...))
//!   - `torch.linalg.cholesky` : aten/src/ATen/native/BatchLinearAlgebra.cpp
//!   - `torch.linalg.inv`      : aten/src/ATen/native/BatchLinearAlgebra.cpp
//!   - `torch.linalg.det`      : aten/src/ATen/native/LinearAlgebra.cpp
//!   - `torch.linalg.slogdet`  : torch/linalg/__init__.py (slogdet = _add_docstr(...))
//!   - `torch.linalg.cross`    : torch/linalg/__init__.py (cross = _add_docstr(...))

mod common;

use common::OracleProc;
use ferrotorch_core::from_vec;

// Runner's linalg tolerance (`tolerance_for` in main.rs) for this subset:
// (rtol=1e-4, atol=1e-6) — the SAME tuple solve/qr/cholesky/inv/det/slogdet/
// cross are wired to.
const RTOL: f32 = 1e-4;
const ATOL: f32 = 1e-6;

/// The SAME element-wise close gate the runner's `assert_close_f32_with_tol`
/// applies (shape gate + `|a-e| <= atol + rtol*|e|`, NaN-position match).
fn gate(
    actual: &[f32],
    actual_shape: &[usize],
    expected: &[f32],
    expected_shape: &[usize],
) -> Result<(), String> {
    if actual_shape != expected_shape {
        return Err(format!(
            "shape mismatch: ferrotorch {actual_shape:?} vs torch {expected_shape:?}"
        ));
    }
    if actual.len() != expected.len() {
        return Err(format!(
            "len mismatch: ferrotorch {} vs torch {}",
            actual.len(),
            expected.len()
        ));
    }
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if a.is_nan() || e.is_nan() {
            if a.is_nan() != e.is_nan() {
                return Err(format!("NaN mismatch at {i}: {a} vs {e}"));
            }
            continue;
        }
        let diff = (a - e).abs();
        let bound = ATOL + RTOL * e.abs();
        if diff > bound {
            return Err(format!(
                "value mismatch at index {i}: ferrotorch={a} vs torch={e} (diff={diff} > bound={bound})"
            ));
        }
    }
    Ok(())
}

// ----- ferrotorch-side probes (mirror of main.rs dispatch arms) -------------

fn ft_recon_mm(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    ferrotorch_core::ops::linalg::mm_raw::<f32>(a, b, m, k, n)
}
fn ft_transpose(m: &[f32], r: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = m[i * c + j];
        }
    }
    out
}

/// A well-conditioned diagonal-dominant SQUARE matrix (mirrors the oracle's
/// `_linalg_square_matrices` construction so torch agrees to f32 ULP). 3x3.
fn sample_square_3x3() -> (Vec<f32>, Vec<usize>) {
    let data = vec![
        4.0f32, -0.3, 0.5, //
        0.2, 5.0, -0.4, //
        -0.6, 0.1, 6.0,
    ];
    (data, vec![3, 3])
}

/// An SPD matrix `M = G Gᵀ + 3·I` (mirrors `_linalg_spd_matrices`). 3x3.
fn sample_spd_3x3() -> (Vec<f32>, Vec<usize>) {
    // G chosen so M is comfortably SPD and L is well separated from singular.
    let g = [
        0.8f32, -0.2, 0.3, //
        0.1, 0.9, -0.4, //
        -0.3, 0.2, 1.1,
    ];
    let gt = ft_transpose(&g, 3, 3);
    let mut m = ft_recon_mm(&g, 3, 3, &gt, 3);
    for i in 0..3 {
        m[i * 3 + i] += 3.0;
    }
    (m, vec![3, 3])
}

// ============================================================================
// solve — UNIQUE solution X, compared directly.
// ============================================================================
#[test]
fn divergence_solve_gate_is_real_and_tight() {
    // Soft-skip marker / fail-closed panic handled in `common` (#1900).
    let Some(mut oracle) = OracleProc::spawn() else {
        return;
    };
    let (a_data, a_shape) = sample_square_3x3();
    let b_data = vec![1.0f32, -2.0, 3.0, 0.5, -1.5, 2.5]; // [3, 2]
    let b_shape = vec![3usize, 2];
    let a = from_vec(a_data.clone(), &a_shape).unwrap();
    let b = from_vec(b_data.clone(), &b_shape).unwrap();

    let (torch_out, torch_shape) = oracle
        .execute("solve", &[(&a_data, &a_shape), (&b_data, &b_shape)])
        .expect("torch solve execute");

    let x = ferrotorch_core::linalg::solve(&a, &b).expect("solve");
    let ft_out = x.data_vec().unwrap();
    let ft_shape = x.shape().to_vec();

    gate(&ft_out, &ft_shape, &torch_out, &torch_shape)
        .expect("solve parity should hold vs live torch");

    // DISCRIMINATION: perturb X[0] by 1e-2 (>> tol). A wrong solution must fail.
    let mut corrupt = ft_out.clone();
    corrupt[0] += 1e-2;
    assert!(
        gate(&corrupt, &ft_shape, &torch_out, &torch_shape).is_err(),
        "gate FAILED to catch a perturbed solve solution — comparison is TOO LOOSE"
    );
}

// ============================================================================
// inv — UNIQUE inverse, compared directly. Sample is non-singular.
// ============================================================================
#[test]
fn divergence_inv_gate_is_real_and_tight() {
    // Soft-skip marker / fail-closed panic handled in `common` (#1900).
    let Some(mut oracle) = OracleProc::spawn() else {
        return;
    };
    let (a_data, a_shape) = sample_square_3x3();
    let a = from_vec(a_data.clone(), &a_shape).unwrap();

    // det != 0 (the diagonal-dominant sample is non-singular).
    let dv = ferrotorch_core::linalg::det(&a)
        .unwrap()
        .data_vec()
        .unwrap();
    assert!(
        dv[0].abs() > 1.0,
        "inv sample must be non-singular (det={})",
        dv[0]
    );

    let (torch_out, torch_shape) = oracle
        .execute("inv", &[(&a_data, &a_shape)])
        .expect("torch inv execute");

    let y = ferrotorch_core::linalg::inv(&a).expect("inv");
    let ft_out = y.data_vec().unwrap();
    let ft_shape = y.shape().to_vec();

    gate(&ft_out, &ft_shape, &torch_out, &torch_shape)
        .expect("inv parity should hold vs live torch");

    let mut corrupt = ft_out.clone();
    corrupt[0] += 1e-2;
    assert!(
        gate(&corrupt, &ft_shape, &torch_out, &torch_shape).is_err(),
        "gate FAILED to catch a perturbed inverse — comparison is TOO LOOSE"
    );
}

// ============================================================================
// det — UNIQUE 0-D scalar. Confirm shape `[]` matches torch `()` AND a wrong
// determinant fails (the perturbation must exceed rtol*|det|).
// ============================================================================
#[test]
fn divergence_det_gate_is_real_and_tight() {
    // Soft-skip marker / fail-closed panic handled in `common` (#1900).
    let Some(mut oracle) = OracleProc::spawn() else {
        return;
    };
    let (a_data, a_shape) = sample_square_3x3();
    let a = from_vec(a_data.clone(), &a_shape).unwrap();

    let (torch_out, torch_shape) = oracle
        .execute("det", &[(&a_data, &a_shape)])
        .expect("torch det execute");

    let d = ferrotorch_core::linalg::det(&a).expect("det");
    let ft_out = d.data_vec().unwrap();
    let ft_shape = d.shape().to_vec();

    // SHAPE: ferrotorch det emits a 0-D `[]`; torch emits `()` — both are the
    // empty shape on the wire. A mismatch here would be a real divergence.
    assert_eq!(
        ft_shape, torch_shape,
        "det shape divergence: ferrotorch {ft_shape:?} vs torch {torch_shape:?}"
    );

    gate(&ft_out, &ft_shape, &torch_out, &torch_shape)
        .expect("det parity should hold vs live torch");

    // DISCRIMINATION: a wrong determinant (scaled by 1.01, ~1% > rtol) fails.
    let corrupt = vec![ft_out[0] * 1.01];
    assert!(
        gate(&corrupt, &ft_shape, &torch_out, &torch_shape).is_err(),
        "gate FAILED to catch a wrong determinant — comparison is TOO LOOSE"
    );
}

// ============================================================================
// qr — gauge-invariant reconstruction Q@R ≈ A. The discrimination must prove
// that a Q@R that does NOT reconstruct A is caught (i.e. the reconstruction is
// genuinely compared against torch's A, not self-compared).
// ============================================================================
#[test]
fn divergence_qr_gate_is_real_and_tight() {
    // Soft-skip marker / fail-closed panic handled in `common` (#1900).
    let Some(mut oracle) = OracleProc::spawn() else {
        return;
    };
    // Tall m>=n sample (4x3), matching the oracle's m>=n-only QR samples.
    let a_data = vec![
        4.0f32, -0.3, 0.5, //
        0.2, 5.0, -0.4, //
        -0.6, 0.1, 6.0, //
        0.3, -0.2, 0.4,
    ];
    let a_shape = vec![4usize, 3];
    let a = from_vec(a_data.clone(), &a_shape).unwrap();

    let (torch_out, torch_shape) = oracle
        .execute("qr", &[(&a_data, &a_shape)])
        .expect("torch qr execute");

    // ferrotorch probe: Q@R reconstruction (mirror of `ferrotorch_qr_probe`).
    let (q, r) = ferrotorch_core::linalg::qr(&a).expect("qr");
    let m = q.shape()[0];
    let k = q.shape()[1];
    let n = r.shape()[1];
    let q_d = q.data_vec().unwrap();
    let r_d = r.data_vec().unwrap();
    let recon = ft_recon_mm(&q_d, m, k, &r_d, n);
    let ft_shape = vec![recon.len()];

    // (a) The reconstruction must equal torch's A flattened (== the input).
    gate(&recon, &ft_shape, &torch_out, &torch_shape)
        .expect("qr reconstruction parity should hold vs live torch");

    // The torch "expected" IS a flattening of A (gauge-invariant), so confirm
    // it equals the input A flattened — proving torch_out is the real input
    // matrix, NOT a ferrotorch-derived value (R-CHAR-3). `torch_shape` is the
    // flattened `[m*n]` shape; compare the flattened input A against it.
    gate(&a_data, &ft_shape, &torch_out, &torch_shape)
        .expect("torch qr output should reconstruct the original A");

    // DISCRIMINATION: a Q@R that fails to reconstruct A (one entry off by 1e-2)
    // must be CAUGHT. This proves R's structure isn't silently ignored: an
    // upper-triangular R that produces the wrong product is rejected.
    let mut corrupt = recon.clone();
    corrupt[0] += 1e-2;
    assert!(
        gate(&corrupt, &ft_shape, &torch_out, &torch_shape).is_err(),
        "gate FAILED to catch a non-reconstructing Q@R — comparison is TOO LOOSE"
    );

    // DISCRIMINATION B: confirm ferrotorch's R is genuinely UPPER-TRIANGULAR
    // (strictly-lower entries == 0). The reconstruction probe would still pass
    // for a non-triangular factorization that happens to multiply to A, so we
    // separately assert R's structure here — a non-upper-triangular "R" is a
    // real divergence from torch's reduced-QR contract.
    for i in 0..k {
        for j in 0..i.min(n) {
            assert!(
                r_d[i * n + j].abs() < 1e-5,
                "ferrotorch QR R must be upper-triangular: R[{i}][{j}]={}",
                r_d[i * n + j]
            );
        }
    }
}

// ============================================================================
// cholesky — L compared DIRECTLY (unique for SPD) AND L@Lᵀ ≈ A reconstruction.
// Discrimination: a wrong L entry must fail the L block; the sample is SPD.
// ============================================================================
#[test]
fn divergence_cholesky_gate_is_real_and_tight() {
    // Soft-skip marker / fail-closed panic handled in `common` (#1900).
    let Some(mut oracle) = OracleProc::spawn() else {
        return;
    };
    let (a_data, a_shape) = sample_spd_3x3();
    let a = from_vec(a_data.clone(), &a_shape).unwrap();

    // Confirm the sample is genuinely SPD: symmetric + cholesky succeeds.
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (a_data[i * 3 + j] - a_data[j * 3 + i]).abs() < 1e-5,
                "cholesky sample must be symmetric"
            );
        }
    }

    let (torch_out, torch_shape) = oracle
        .execute("cholesky", &[(&a_data, &a_shape)])
        .expect("torch cholesky execute");

    // ferrotorch probe: concat([L.flatten(), (L Lᵀ).flatten()]) (mirror).
    let l = ferrotorch_core::linalg::cholesky(&a).expect("cholesky");
    let n = l.shape()[0];
    let l_d = l.data_vec().unwrap();
    let lt = ft_transpose(&l_d, n, n);
    let recon = ft_recon_mm(&l_d, n, n, &lt, n);
    let mut ft_out = l_d.clone();
    ft_out.extend_from_slice(&recon);
    let ft_shape = vec![ft_out.len()];

    gate(&ft_out, &ft_shape, &torch_out, &torch_shape)
        .expect("cholesky L + reconstruction parity should hold vs live torch");

    // The L block (first n*n entries) of torch_out must equal torch's UNIQUE L
    // (lower-triangular: upper entries strictly 0). Confirm torch's L is
    // genuinely lower-triangular so the direct-L comparison is meaningful.
    for i in 0..n {
        for j in (i + 1)..n {
            assert!(
                torch_out[i * n + j].abs() < 1e-6,
                "torch cholesky L must be lower-triangular: L[{i}][{j}]={}",
                torch_out[i * n + j]
            );
        }
    }

    // DISCRIMINATION A: a WRONG L diagonal entry (L[0][0] off by 1e-2) must be
    // caught by the DIRECT L block — proving L is compared against torch's L,
    // not just self-consistent with the reconstruction.
    let mut corrupt_l = ft_out.clone();
    corrupt_l[0] += 1e-2; // L[0][0]
    assert!(
        gate(&corrupt_l, &ft_shape, &torch_out, &torch_shape).is_err(),
        "gate FAILED to catch a wrong L entry — L is NOT compared directly (TAUTOLOGICAL)"
    );

    // DISCRIMINATION B: a reconstruction that does not equal A must be caught.
    let mut corrupt_recon = ft_out.clone();
    corrupt_recon[n * n] += 1e-2; // first reconstruction element
    assert!(
        gate(&corrupt_recon, &ft_shape, &torch_out, &torch_shape).is_err(),
        "gate FAILED to catch a non-reconstructing L@Lᵀ — comparison is TOO LOOSE"
    );
}

// ============================================================================
// slogdet — [sign, logabsdet]. Sign EXACT, logabsdet rtol. The sample is the
// NEGATIVE-determinant odd-permutation case (sign must be -1). Discrimination:
// a sign FLIP (+1 vs -1) and a wrong logabsdet must both be caught.
// ============================================================================
#[test]
fn divergence_slogdet_gate_is_real_and_tight_negative_det() {
    // Soft-skip marker / fail-closed panic handled in `common` (#1900).
    let Some(mut oracle) = OracleProc::spawn() else {
        return;
    };
    // A diagonal-dominant 3x3 with rows 0 and 1 swapped (odd permutation =>
    // NEGATIVE determinant => sign = -1). Mirrors the oracle's negative-det
    // slogdet sample construction.
    let base = vec![
        4.0f32, -0.3, 0.5, //
        0.2, 5.0, -0.4, //
        -0.6, 0.1, 6.0,
    ];
    // swap rows 0 and 1.
    let a_data = vec![
        base[3], base[4], base[5], // old row 1
        base[0], base[1], base[2], // old row 0
        base[6], base[7], base[8], // row 2
    ];
    let a_shape = vec![3usize, 3];
    let a = from_vec(a_data.clone(), &a_shape).unwrap();

    let (torch_out, torch_shape) = oracle
        .execute("slogdet", &[(&a_data, &a_shape)])
        .expect("torch slogdet execute");
    assert_eq!(torch_shape, vec![2], "slogdet probe must be a [2] tensor");

    // CONFIRM the live torch sign is genuinely -1 (negative-det case present).
    assert!(
        (torch_out[0] - (-1.0)).abs() < 1e-6,
        "expected torch slogdet sign = -1 for the odd-permutation sample, got {}",
        torch_out[0]
    );

    // ferrotorch probe: [sign, logabsdet] (mirror).
    let (sign, logabs) = ferrotorch_core::linalg::slogdet(&a).expect("slogdet");
    let s = sign.data_vec().unwrap();
    let lg = logabs.data_vec().unwrap();
    let ft_out = vec![s[0], lg[0]];
    let ft_shape = vec![2usize];

    gate(&ft_out, &ft_shape, &torch_out, &torch_shape)
        .expect("slogdet [sign, logabsdet] parity should hold vs live torch");

    // DISCRIMINATION A: FLIP the sign (+1 instead of -1). |1-(-1)|=2 >> bound,
    // so the sign IS compared exactly — a sign-confused forward is caught.
    let sign_flipped = vec![-ft_out[0], ft_out[1]];
    assert!(
        gate(&sign_flipped, &ft_shape, &torch_out, &torch_shape).is_err(),
        "gate FAILED to catch a flipped slogdet sign — sign is NOT compared exactly"
    );

    // DISCRIMINATION B: a wrong logabsdet (off by 1e-2 absolute) must be caught.
    let logabs_wrong = vec![ft_out[0], ft_out[1] + 1e-2];
    assert!(
        gate(&logabs_wrong, &ft_shape, &torch_out, &torch_shape).is_err(),
        "gate FAILED to catch a wrong logabsdet — comparison is TOO LOOSE"
    );
}

// ============================================================================
// cross — UNIQUE bilinear product along dim=-1. Both the 1-D [3] case and the
// batched [n,3] case. Discrimination: a wrong component must be caught.
// ============================================================================
#[test]
fn divergence_cross_gate_is_real_and_tight() {
    // Soft-skip marker / fail-closed panic handled in `common` (#1900).
    let Some(mut oracle) = OracleProc::spawn() else {
        return;
    };

    // ----- 1-D [3] case (the default dim=-1 is the only axis) -----
    let a1 = vec![1.0f32, 2.0, 3.0];
    let b1 = vec![-0.5f32, 0.25, 1.5];
    let s1 = vec![3usize];
    let a1t = from_vec(a1.clone(), &s1).unwrap();
    let b1t = from_vec(b1.clone(), &s1).unwrap();

    let (torch1, ts1) = oracle
        .execute("cross", &[(&a1, &s1), (&b1, &s1)])
        .expect("torch cross [3] execute");
    let c1 = ferrotorch_core::linalg::cross(&a1t, &b1t, -1).expect("cross [3]");
    let ft1 = c1.data_vec().unwrap();
    let fs1 = c1.shape().to_vec();
    gate(&ft1, &fs1, &torch1, &ts1).expect("cross [3] parity should hold vs live torch");

    // The cross product of [1,2,3] x [-0.5,0.25,1.5] has a known closed form;
    // confirm torch's value matches the hand-computed bilinear product (a
    // SECOND non-tautological anchor independent of ferrotorch):
    //   c = (a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1)
    let expect = [
        a1[1] * b1[2] - a1[2] * b1[1],
        a1[2] * b1[0] - a1[0] * b1[2],
        a1[0] * b1[1] - a1[1] * b1[0],
    ];
    for i in 0..3 {
        assert!(
            (torch1[i] - expect[i]).abs() < 1e-5,
            "torch cross[{i}]={} != hand-computed {}",
            torch1[i],
            expect[i]
        );
    }

    // DISCRIMINATION: a wrong cross component (1e-2 off) must be caught.
    let mut corrupt = ft1.clone();
    corrupt[0] += 1e-2;
    assert!(
        gate(&corrupt, &fs1, &torch1, &ts1).is_err(),
        "gate FAILED to catch a wrong cross component — comparison is TOO LOOSE"
    );

    // ----- batched [2, 3] case (dim=-1 must be the size-3 axis) -----
    let a2 = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
    let b2 = vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 1.0];
    let s2 = vec![2usize, 3];
    let a2t = from_vec(a2.clone(), &s2).unwrap();
    let b2t = from_vec(b2.clone(), &s2).unwrap();
    let (torch2, ts2) = oracle
        .execute("cross", &[(&a2, &s2), (&b2, &s2)])
        .expect("torch cross [2,3] execute");
    let c2 = ferrotorch_core::linalg::cross(&a2t, &b2t, -1).expect("cross [2,3]");
    let ft2 = c2.data_vec().unwrap();
    let fs2 = c2.shape().to_vec();
    // x̂ × ŷ = ẑ, ŷ × ẑ = x̂ — confirms dim handling on the batched axis.
    assert_eq!(ts2, vec![2, 3], "batched cross output shape must be [2,3]");
    gate(&ft2, &fs2, &torch2, &ts2).expect("cross [2,3] parity should hold vs live torch");
}

// ============================================================================
// addr — the beta==0 skip removal (#1598 fix). The runner now runs the
// degenerate {beta:0, alpha:0, self=NaN} sample with NO skip. Confirm
// ferrotorch's addr forward DROPS the NaN self term (returns finite 0) so the
// gate passes — and that a forward which DID propagate the NaN would be caught.
// ============================================================================
#[test]
fn divergence_addr_beta0_nan_self_dropped() {
    // No oracle needed: torch's `addr_kernel` beta==0 contract is documented at
    // `aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp:53-55,60` ("when beta
    // == 0 ... nans and infs in self should not propagate"; returns
    // `alpha_val * vec1_val * vec2_val`). With beta=0 AND alpha=0, torch's
    // output is the all-finite ZERO matrix regardless of a NaN `self`.
    let nan = f32::NAN;
    let bias = from_vec(vec![nan], &[1, 1]).unwrap(); // [1,1] self carrying NaN
    let vec1 = from_vec(vec![1.0f32, 2.0], &[2]).unwrap();
    let vec2 = from_vec(vec![3.0f32, 4.0, 5.0], &[3]).unwrap();

    let out =
        ferrotorch_core::grad_fns::linalg::addr_differentiable(&bias, &vec1, &vec2, 0.0f32, 0.0f32)
            .expect("addr beta=0 alpha=0");
    let data = out.data_vec().unwrap();

    // EXPECTED (named symbolic constant traceable to the upstream kernel):
    // beta==0 drops self; alpha==0 zeroes the outer product => all-zero,
    // all-finite. A forward that computed `beta*self` literally would yield
    // `0.0 * NaN = NaN` (IEEE-754) and FAIL this assertion.
    const TORCH_BETA0_ALPHA0_NAN_SELF: f32 = 0.0; // LinearAlgebraKernel.cpp:53-55,60
    assert_eq!(out.shape(), &[2, 3], "addr output shape must be [n=2, m=3]");
    for (i, &v) in data.iter().enumerate() {
        assert!(
            v.is_finite() && v == TORCH_BETA0_ALPHA0_NAN_SELF,
            "addr[{i}]={v}: beta==0 must DROP the NaN self term (torch returns finite 0); \
             a literal `0*NaN` would propagate NaN here"
        );
    }
}
