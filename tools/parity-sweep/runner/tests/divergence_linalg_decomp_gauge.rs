//! Discrimination audit for commit `a1d24b878` — the 12 linalg DECOMPOSITION
//! gauge-invariant runner arms (#1344).
//!
//! These tests do NOT re-run the `parity-sweep` binary; they re-implement, in
//! a host test, the EXACT comparison the runner performs (ferrotorch's own
//! gauge-invariant derived quantity vs. the live-torch oracle's derived
//! quantity, under the runner's decomposition tolerance `rtol=1e-4,
//! atol=1e-6`). The point is to verify the comparison is REAL and TIGHT:
//!
//!   1. NON-TAUTOLOGY: the "expected" value is fetched from a live torch
//!      subprocess (`tools/parity-sweep/oracle.py`, `execute` cmd), NOT
//!      derived from ferrotorch's own output. (R-CHAR-3 (a).)
//!
//!   2. DISCRIMINATION: a deliberately-WRONG ferrotorch result (perturbed
//!      reconstruction / permuted singular values / imaginary part dropped)
//!      must FAIL the same gate, proving the comparison would catch a
//!      genuinely-wrong decomposition rather than rubber-stamp it.
//!
//! Upstream sites mirrored:
//!   - `torch.linalg.svd`  : aten/src/ATen/native/BatchLinearAlgebra.cpp
//!   - `torch.linalg.eigh` : aten/src/ATen/native/BatchLinearAlgebra.cpp
//!   - `torch.linalg.eig`  : aten/src/ATen/native/BatchLinearAlgebra.cpp:3075
//!
//! Oracle availability (CORE-206 / #1900, fail-closed gate in
//! `tests/common/mod.rs`): if the oracle (python3 + torch) is unavailable the
//! tests print a single-line `VACUOUS-PASS:` marker and soft-skip — UNLESS
//! `PARITY_ORACLE_REQUIRED=1` is set (as in the nightly parity-smoke step),
//! in which case they PANIC with diagnostics. Run explicitly with:
//!   LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH" \
//!     PARITY_ORACLE_REQUIRED=1 \
//!     cargo test -p parity-sweep-runner --test divergence_linalg_decomp_gauge \
//!     -- --nocapture

mod common;

use common::OracleProc;
use ferrotorch_core::{Tensor, from_vec};

// Runner's decomposition tolerance (`tolerance_for` in main.rs): (1e-4, 1e-6).
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

// ----- ferrotorch-side gauge-invariant probes (mirror of main.rs helpers) ---

fn ft_recon_mm(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    ferrotorch_core::ops::linalg::mm_raw::<f32>(a, b, m, k, n)
}
fn ft_scale_cols(m: &[f32], rows: usize, cols: usize, s: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = m[i * cols + j] * s[j];
        }
    }
    out
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

/// ferrotorch SVD probe: `concat([S, U diag(S) Vh])` — same as `_svd_torch_call`.
fn ft_svd_probe(a: &Tensor<f32>) -> (Vec<f32>, usize) {
    let (u, s, vh) = ferrotorch_core::linalg::svd(a).expect("svd");
    let m = u.shape()[0];
    let k = u.shape()[1];
    let n = vh.shape()[1];
    let u_d = u.data_vec().unwrap();
    let s_d = s.data_vec().unwrap();
    let vh_d = vh.data_vec().unwrap();
    let us = ft_scale_cols(&u_d, m, k, &s_d);
    let recon = ft_recon_mm(&us, m, k, &vh_d, n);
    let mut out = s_d;
    out.extend_from_slice(&recon);
    (out, k)
}

/// A SQUARE matrix with well-separated singular values (mirrors the oracle's
/// `_linalg_decomp_matrices` diagonal-dominant construction so torch agrees to
/// f32 ULP). 3x3.
fn sample_3x3() -> (Vec<f32>, Vec<usize>) {
    let data = vec![
        4.0f32, -0.3, 0.5, //
        0.2, 5.0, -0.4, //
        -0.6, 0.1, 6.0,
    ];
    (data, vec![3, 3])
}

#[test]
fn divergence_svd_gate_is_real_and_tight() {
    // `common::OracleProc::spawn` prints the `VACUOUS-PASS:` marker on soft
    // skip and PANICS under PARITY_ORACLE_REQUIRED=1 (CORE-206 / #1900).
    let Some(mut oracle) = OracleProc::spawn() else {
        return;
    };
    let (data, shape) = sample_3x3();
    let a = from_vec(data.clone(), &shape).unwrap();

    // Live torch gauge-invariant SVD output (NON-TAUTOLOGICAL expected).
    let (torch_out, torch_shape) = oracle
        .execute_mat("svd", &data, &shape)
        .expect("torch svd execute");

    // ferrotorch's own probe.
    let (ft_out, s_len) = ft_svd_probe(&a);
    let ft_shape = vec![ft_out.len()];

    // (1) Parity must hold — the gauge-invariant probe genuinely matches torch.
    gate(&ft_out, &ft_shape, &torch_out, &torch_shape)
        .expect("svd gauge-invariant parity should hold vs live torch");

    // (2) DISCRIMINATION A: perturb the RECONSTRUCTION block by 1e-2 (>> tol).
    // A wrong ferrotorch U/S/Vh that fails to reconstruct A must be CAUGHT.
    let mut corrupt = ft_out.clone();
    corrupt[s_len] += 1e-2; // first reconstruction element
    assert!(
        gate(&corrupt, &ft_shape, &torch_out, &torch_shape).is_err(),
        "gate FAILED to catch a perturbed reconstruction — comparison is TOO LOOSE \
         (a wrong svd would pass)"
    );

    // (3) DISCRIMINATION B: permute the singular VALUES (swap S[0], S[1]).
    // torch returns S DESCENDING; a ferrotorch that returned S in the wrong
    // order must be CAUGHT by the S block (not hidden by reconstruction).
    if s_len >= 2 && (ft_out[0] - ft_out[1]).abs() > 1e-3 {
        let mut sperm = ft_out.clone();
        sperm.swap(0, 1);
        assert!(
            gate(&sperm, &ft_shape, &torch_out, &torch_shape).is_err(),
            "gate FAILED to catch permuted singular values — S is not compared \
             against torch's descending S (reconstruction alone is insufficient)"
        );
    }
}

#[test]
fn divergence_eigh_gate_is_real_and_tight() {
    // Soft-skip marker / fail-closed panic handled in `common` (#1900).
    let Some(mut oracle) = OracleProc::spawn() else {
        return;
    };
    // Symmetric, well-separated eigenvalues (mirror `_linalg_symmetric_matrices`).
    let data = vec![
        4.0f32, 0.3, -0.2, //
        0.3, 5.5, 0.4, //
        -0.2, 0.4, 7.0,
    ];
    let shape = vec![3usize, 3];
    let a = from_vec(data.clone(), &shape).unwrap();

    let (torch_out, torch_shape) = oracle
        .execute_mat("eigh", &data, &shape)
        .expect("torch eigh execute");

    // ferrotorch eigh probe: concat([w, Q diag(w) Q^T]).
    let (w, q) = ferrotorch_core::linalg::eigh(&a).expect("eigh");
    let n = q.shape()[0];
    let w_d = w.data_vec().unwrap();
    let q_d = q.data_vec().unwrap();
    let qw = ft_scale_cols(&q_d, n, n, &w_d);
    let qt = ft_transpose(&q_d, n, n);
    let recon = ft_recon_mm(&qw, n, n, &qt, n);
    let mut ft_out = w_d.clone();
    ft_out.extend_from_slice(&recon);
    let ft_shape = vec![ft_out.len()];

    gate(&ft_out, &ft_shape, &torch_out, &torch_shape)
        .expect("eigh gauge-invariant parity should hold vs live torch");

    // DISCRIMINATION: the eigenVALUES `w` are compared against TORCH's `w`
    // (ascending), NOT just self-consistency. Perturb w[0] by 1e-2 and confirm
    // the gate fails on the eigenvalue block (catches a wrong-but-self-
    // consistent reconstruction where ferrotorch's own Q diag(w) Q^T == A but
    // w disagrees with torch — i.e. a tautology would hide this).
    let mut corrupt_w = ft_out.clone();
    corrupt_w[0] += 1e-2;
    assert!(
        gate(&corrupt_w, &ft_shape, &torch_out, &torch_shape).is_err(),
        "gate FAILED to catch a perturbed eigenvalue — eigenvalues are NOT compared \
         against torch's w (TAUTOLOGICAL: w only self-consistent with reconstruction)"
    );
}

#[test]
fn divergence_eig_complex_imaginary_part_is_compared() {
    // Soft-skip marker / fail-closed panic handled in `common` (#1900).
    let Some(mut oracle) = OracleProc::spawn() else {
        return;
    };
    // A 2x2 rotation-like matrix with a genuine complex-conjugate eigenvalue
    // pair: [[a, -b],[b, a]] has eigenvalues a ± b i. Pick a=1.5, b=2.0 so the
    // imaginary part is large and unambiguous.
    let data = vec![1.5f32, -2.0, 2.0, 1.5];
    let shape = vec![2usize, 2];
    let a = from_vec(data.clone(), &shape).unwrap();

    let (torch_out, torch_shape) = oracle
        .execute_mat("eig", &data, &shape)
        .expect("torch eig execute");

    // torch_out is the sorted [n,2] interleaved (re, im) eigenvalue set,
    // flattened. For a=1.5, b=2.0 the eigenvalues are 1.5 ± 2.0 i, so the
    // IMAGINARY parts are non-zero. Confirm the oracle did NOT real-project.
    let max_abs_im = torch_out
        .iter()
        .skip(1)
        .step_by(2)
        .fold(0.0f32, |m, &x| m.max(x.abs()));
    assert!(
        max_abs_im > 1.0,
        "oracle eig output dropped the imaginary part (max|im|={max_abs_im}) — \
         the complex eigenvalue set is being real-projected, NOT compared as complex"
    );

    // ferrotorch's eig probe: sort the [n,2] interleaved eigenvalues by (re,im)
    // with the SAME 5-decimal key as main.rs `sort_complex_eigvals`.
    let (w, _v) = ferrotorch_core::linalg::eig(&a).expect("eig");
    let n = a.shape()[0];
    let w_d = w.data_vec().unwrap();
    let mut idx: Vec<usize> = (0..n).collect();
    let key = |i: usize| -> (i64, i64) {
        let re = (w_d[2 * i] as f64 * 1e5).round() as i64;
        let im = (w_d[2 * i + 1] as f64 * 1e5).round() as i64;
        (re, im)
    };
    idx.sort_by_key(|&x| key(x));
    let mut ft_out = Vec::with_capacity(2 * n);
    for i in idx {
        ft_out.push(w_d[2 * i]);
        ft_out.push(w_d[2 * i + 1]);
    }
    let ft_shape = vec![ft_out.len()];

    gate(&ft_out, &ft_shape, &torch_out, &torch_shape)
        .expect("eig complex eigenvalue-set parity should hold vs live torch");

    // DISCRIMINATION: drop the imaginary part on the ferrotorch side (a real-
    // only forward bug) and confirm the gate CATCHES it.
    let mut real_only = ft_out.clone();
    for v in real_only.iter_mut().skip(1).step_by(2) {
        *v = 0.0;
    }
    assert!(
        gate(&real_only, &ft_shape, &torch_out, &torch_shape).is_err(),
        "gate FAILED to catch a real-only eig result — the imaginary part is NOT \
         being compared (complex set collapses to a real projection)"
    );
}
