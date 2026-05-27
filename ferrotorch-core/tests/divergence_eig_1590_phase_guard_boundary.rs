//! R-FIX-4 re-audit of commit `bd3387455` — eig backward phase-invariance guard
//! (#1590, gates closing #1345).
//!
//! The fixer added a guard in `EigBackwardV::grad_a_from_gv` mirroring torch's
//! `linalg_eig_backward` phase-invariance check:
//!
//!   torch/csrc/autograd/FunctionsManual.cpp:3865-3879
//!     auto VhgV = at::matmul(V.mH(), gV);
//!     const auto diag_VhgV = VhgV.diagonal(0, -2, -1);
//!     if (V.is_complex() && !at::isTensorSubclassLike(diag_VhgV)) {
//!       const auto imdiag_VhgV = at::imag(diag_VhgV);
//!       TORCH_CHECK(
//!           at::allclose(imdiag_VhgV, at::zeros_like(imdiag_VhgV),
//!                        /*rtol=*/1e-2, /*atol=*/1e-2), ... ill-defined);
//!     }
//!
//! ferrotorch implements this as
//!   `(0..n).any(|i| vhgv[i*n+i].1.abs() > atol)` with `atol = 1e-2`
//! (ferrotorch-core/src/grad_fns/linalg.rs:6078-6088).
//!
//! These tests pin the BOUNDARY behavior against LIVE torch 2.11.0+cu130
//! float64. All expected values below are quoted from a live torch run
//! (R-CHAR-3 (a)); none are copied from the ferrotorch side.
//!
//! KEY RISK (gauge): ferrotorch's V comes from faer, torch's from LAPACK. The
//! per-column phase may differ, so for the SAME loss the value of
//! `imag(diag(V^H gV))` — and thus whether the guard fires — can differ between
//! the two libraries even though the guard CODE is identical.
//!
//! #1591 RESOLUTION (mirrors eigh #1584): the EXACT guard boundary fundamentally
//! cannot match torch's LAPACK gauge (it would require replicating LAPACK geev's
//! phases). Complex eigenvectors are a genuine gauge freedom up to `e^{i phi}`
//! (`FunctionsManual.cpp:3867-3879`). ferrotorch instead canonicalizes the eig
//! eigenvector PHASE deterministically (`canonicalize_complex_eigenvector_phase`
//! in linalg.rs) for a reproducible gauge, and these tests pin the WELL-POSED,
//! gauge-robust quantities: (1) phase-INVARIANT losses match torch's gradient
//! exactly (gauge-free); (2) grossly phase-DEPENDENT losses ERROR (ill-defined,
//! torch errors too); (3) eig eigenvectors are now DETERMINISTIC (call twice,
//! identical). The original exact-boundary assertion was unmatchable and has
//! been replaced — see the #1591 RESOLUTION block near the end of this file.

use ferrotorch_core::Tensor;
use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::linalg as linalg_fwd;
use ferrotorch_core::storage::TensorStorage;

fn leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}
fn no_grad_leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// `loss = sum( re_w[i,j]*V.re[i,j] + im_w[i,j]*V.im[i,j] )` — a linear loss on
/// the complex `[n,n,2]` eigenvectors with arbitrary per-slot weights.
fn eigvec_linear_loss(v: &Tensor<f64>, re_w: &[f64], im_w: &[f64], n: usize) -> Tensor<f64> {
    let mut wt = vec![0.0; n * n * 2];
    for idx in 0..n * n {
        wt[2 * idx] = re_w[idx];
        wt[2 * idx + 1] = im_w[idx];
    }
    let wts = no_grad_leaf(&wt, &[n, n, 2]);
    reduce_sum(&mul(v, &wts).unwrap()).unwrap()
}

/// Phase-invariant `sum( (re^2+im^2) * M[i,j] )`.
fn eigvec_phase_invariant_loss(v: &Tensor<f64>, m: &[f64], n: usize) -> Tensor<f64> {
    let mut wt = vec![0.0; n * n * 2];
    for idx in 0..n * n {
        wt[2 * idx] = m[idx];
        wt[2 * idx + 1] = m[idx];
    }
    let wts = no_grad_leaf(&wt, &[n, n, 2]);
    let vsq = mul(v, v).unwrap();
    reduce_sum(&mul(&vsq, &wts).unwrap()).unwrap()
}

fn assert_close(got: &[f64], want: &[f64], tol: f64, ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{ctx}: element {i}: got {g}, want {w} (torch), diff {}",
            (g - w).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// (1) Fires for a phase-DEPENDENT loss: A=[[1,-1],[1,1]], loss = sum(V.real).
//     LIVE torch RAISES; ferrotorch must Err.
// ---------------------------------------------------------------------------
#[test]
fn phase_dependent_sum_real_errors_like_torch() {
    let a = leaf(&[1.0, -1.0, 1.0, 1.0], &[2, 2]);
    let (_w, v) = linalg_fwd::eig(&a).unwrap();
    // loss = sum(V.real): re weights all 1, im weights all 0.
    let loss = eigvec_linear_loss(&v, &[1.0; 4], &[0.0; 4], 2);
    let r = loss.backward();
    assert!(
        r.is_err(),
        "torch RAISES on sum(V.real) for complex eig (FunctionsManual.cpp:3867); \
         ferrotorch backward returned Ok with A.grad={:?}",
        a.grad()
            .ok()
            .flatten()
            .and_then(|g| g.data().ok().map(|d| d.to_vec()))
    );
}

// ---------------------------------------------------------------------------
// (2) Does NOT fire for phase-INVARIANT losses; grad matches torch.
//     LIVE torch float64 for A=[[1,-1],[1,1]]:
//       sum(|V|^2 * M), M=[[0.5,-0.3],[0.2,0.8]]  -> grad [[~0, 0.2],[0.2, ~0]]
//       sum(|V|^2)  (M=all 1)                      -> grad [[~0, 0],[0, ~0]]
// ---------------------------------------------------------------------------
#[test]
fn phase_invariant_weighted_sq_matches_torch() {
    let a = leaf(&[1.0, -1.0, 1.0, 1.0], &[2, 2]);
    let (_w, v) = linalg_fwd::eig(&a).unwrap();
    let m = [0.5, -0.3, 0.2, 0.8];
    let loss = eigvec_phase_invariant_loss(&v, &m, 2);
    loss.backward()
        .expect("phase-invariant loss must NOT error (torch computes a grad)");
    let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    // LIVE torch.autograd.grad((|V|^2*M).sum(), A): [[1.11e-16, 0.2],[0.2, 1.11e-16]]
    assert_close(&g, &[0.0, 0.2, 0.2, 0.0], 1e-6, "phase-invariant |V|^2*M");
}

#[test]
fn phase_invariant_uniform_sq_matches_torch() {
    let a = leaf(&[1.0, -1.0, 1.0, 1.0], &[2, 2]);
    let (_w, v) = linalg_fwd::eig(&a).unwrap();
    let loss = eigvec_phase_invariant_loss(&v, &[1.0; 4], 2);
    loss.backward()
        .expect("phase-invariant sum(|V|^2) must NOT error");
    let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    // LIVE torch.autograd.grad((V.abs()**2).sum(), A): all ~0 (2.22e-16).
    assert_close(
        &g,
        &[0.0, 0.0, 0.0, 0.0],
        1e-6,
        "phase-invariant sum(|V|^2)",
    );
}

// ---------------------------------------------------------------------------
// (3a) Tolerance boundary BELOW threshold: loss = c*sum(V.real) with
//      c = 1/(100*sqrt(2)) = 0.0070710678... makes |imag(diag(V^H gV))| = 0.005
//      < atol=1e-2 in torch's gauge. LIVE torch does NOT error and returns
//      grad [[0.0025, -0.0075],[0.0025, -0.0025]]. ferrotorch must match (NOT
//      error AND value-match). A too-tight tolerance over-fires here.
// ---------------------------------------------------------------------------
#[test]
fn tolerance_below_threshold_does_not_error_and_matches_torch() {
    let c = 1.0 / (100.0 * std::f64::consts::SQRT_2); // 0.0070710678118654755
    let a = leaf(&[1.0, -1.0, 1.0, 1.0], &[2, 2]);
    let (_w, v) = linalg_fwd::eig(&a).unwrap();
    let loss = eigvec_linear_loss(&v, &[c; 4], &[0.0; 4], 2);
    let r = loss.backward();
    assert!(
        r.is_ok(),
        "torch does NOT error at |imag(diag)|=0.005 < atol=1e-2 (allclose true) \
         but ferrotorch errored: {:?}. Guard over-fires (tolerance too tight or \
         gauge mismatch).",
        r.err()
    );
    let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    // LIVE torch.autograd.grad((c*V.real).sum(), A), c=0.0070710678118654755:
    //   [[ 0.0025, -0.0075],[ 0.0025, -0.0025]]
    assert_close(
        &g,
        &[0.0025, -0.0075, 0.0025, -0.0025],
        1e-6,
        "below-threshold scaled-real loss grad vs torch",
    );
}

// ---------------------------------------------------------------------------
// (3b) Tolerance boundary ABOVE threshold: c = 1/(10*sqrt(2)) = 0.0707106781...
//      makes |imag(diag(V^H gV))| = 0.05 > atol=1e-2. LIVE torch RAISES.
//      ferrotorch must Err. A too-loose tolerance misses this.
// ---------------------------------------------------------------------------
#[test]
fn tolerance_above_threshold_errors_like_torch() {
    let c = 1.0 / (10.0 * std::f64::consts::SQRT_2); // 0.07071067811865475
    let a = leaf(&[1.0, -1.0, 1.0, 1.0], &[2, 2]);
    let (_w, v) = linalg_fwd::eig(&a).unwrap();
    let loss = eigvec_linear_loss(&v, &[c; 4], &[0.0; 4], 2);
    let r = loss.backward();
    assert!(
        r.is_err(),
        "torch RAISES at |imag(diag)|=0.05 > atol=1e-2; ferrotorch returned Ok \
         A.grad={:?}. Guard under-fires (tolerance too loose).",
        a.grad()
            .ok()
            .flatten()
            .and_then(|g| g.data().ok().map(|d| d.to_vec()))
    );
}

// ---------------------------------------------------------------------------
// (4) Real-V case NEVER triggers the guard: A upper-triangular with distinct
//     REAL eigenvalues -> V real -> imag(diag(V^H gV))=0 for ANY loss, even a
//     "phase-dependent-looking" sum(V.real). LIVE torch does NOT error and
//     returns the finite grad below.
// ---------------------------------------------------------------------------
#[test]
fn real_v_sum_real_does_not_error_and_matches_torch() {
    let a = leaf(&[2.0, 0.5, 0.3, 0.0, 3.0, 0.4, 0.0, 0.0, 5.0], &[3, 3]);
    let (_w, v) = linalg_fwd::eig(&a).unwrap();
    // loss = sum(V.real). For real V this is gauge-free up to sign (no phase).
    let loss = eigvec_linear_loss(&v, &[1.0; 9], &[0.0; 9], 3);
    let r = loss.backward();
    assert!(
        r.is_ok(),
        "real-V eig has imag(diag(V^H gV))=0 so torch does NOT error on \
         sum(V.real); ferrotorch errored: {:?}",
        r.err()
    );
    // We assert only NO-ERROR + finiteness here (the sign gauge of a real
    // eigenvector can flip between faer and LAPACK, so the per-element grad is
    // gauge-dependent; the guard-firing behavior is what (4) pins). Finiteness
    // guards against a NaN/Inf gauge-divide bug.
    let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert!(
        g.iter().all(|x| x.is_finite()),
        "real-V sum(V.real) grad must be finite, got {g:?}"
    );
}

// ---------------------------------------------------------------------------
// (5) eigvals (lambda-only) backward is GAUGE-FREE and must NOT carry the
//     phase guard: even the "phase-dependent" linear eigenvalue loss works.
//     LIVE torch.linalg.eigvals A=[[1,-1],[1,1]], loss=sum(re(L)*1.3 - im... ):
//     reuse the in-lib pinned value to confirm eigvals still computes a grad.
// ---------------------------------------------------------------------------
#[test]
fn eigvals_backward_complex_still_works_no_guard() {
    let a = leaf(&[1.0, -1.0, 1.0, 1.0], &[2, 2]);
    let w = linalg_fwd::eigvals(&a).unwrap();
    // loss = sum(re(L)*1.3 + re? ) ; build linear loss on [n,2] eigenvalues:
    //   re weights [1.3, -0.7], im weights [0.4, 0.6].
    let mut wt = vec![0.0; 2 * 2];
    wt[0] = 1.3;
    wt[1] = 0.4;
    wt[2] = -0.7;
    wt[3] = 0.6;
    let wts = no_grad_leaf(&wt, &[2, 2]);
    let loss = reduce_sum(&mul(&w, &wts).unwrap()).unwrap();
    loss.backward()
        .expect("eigvals backward is gauge-free, must NOT error");
    let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    // LIVE torch.linalg.eigvals A.grad (matches in-lib eigvals_backward_complex_pair_2x2).
    assert_close(
        &g,
        &[
            0.30000000000000004,
            0.09999999999999996,
            -0.09999999999999996,
            0.30000000000000004,
        ],
        1e-6,
        "eigvals complex backward (no phase guard)",
    );
}

// ---------------------------------------------------------------------------
// (gauge probe) Print ferrotorch's V to compare against torch's LAPACK gauge.
// Not an assertion — diagnostic to confirm the boundary tests above are not
// passing by gauge coincidence.
// ---------------------------------------------------------------------------
#[test]
fn gauge_diagnostic_print_v() {
    let a = no_grad_leaf(&[1.0, -1.0, 1.0, 1.0], &[2, 2]);
    let (w, v) = linalg_fwd::eig(&a).unwrap();
    eprintln!("ferrotorch W = {:?}", w.data().unwrap().to_vec());
    eprintln!("ferrotorch V = {:?}", v.data().unwrap().to_vec());
}

// ---------------------------------------------------------------------------
// (gauge probe 3x3) A 3x3 with one real eigenvalue + a complex-conjugate pair.
// The complex columns are genuinely phase-ambiguous. Confirms (a) phase-
// invariant loss matches torch even if faer's gauge != LAPACK's, and (b) the
// guard still fires for sum(V.real). LIVE torch 2.11.0 float64 values pinned.
// ---------------------------------------------------------------------------
const A3: [f64; 9] = [0.0, -1.0, 0.5, 1.0, 0.0, 0.3, 0.2, 0.1, 2.0];

#[test]
fn mixed_3x3_phase_invariant_matches_torch() {
    let a = leaf(&A3, &[3, 3]);
    let (_w, v) = linalg_fwd::eig(&a).unwrap();
    let m = [0.5, -0.3, 0.1, 0.2, 0.8, -0.4, 0.6, 0.1, 0.7];
    let loss = eigvec_phase_invariant_loss(&v, &m, 3);
    loss.backward()
        .expect("phase-invariant loss must NOT error for mixed 3x3");
    let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    // LIVE torch.autograd.grad(((V.real^2+V.imag^2)*M).sum(), A):
    let torch = [
        -0.018451, 0.165583, -0.131945, 0.185531, -0.029623, -0.158005, 0.005740, -0.038701,
        0.048074,
    ];
    assert_close(&g, &torch, 1e-5, "mixed 3x3 phase-invariant grad vs torch");
}

#[test]
fn mixed_3x3_phase_dependent_errors_like_torch() {
    let a = leaf(&A3, &[3, 3]);
    let (_w, v) = linalg_fwd::eig(&a).unwrap();
    let loss = eigvec_linear_loss(&v, &[1.0; 9], &[0.0; 9], 3);
    let r = loss.backward();
    assert!(
        r.is_err(),
        "torch RAISES on sum(V.real) for mixed 3x3 (complex columns); \
         ferrotorch returned Ok A.grad={:?}",
        a.grad()
            .ok()
            .flatten()
            .and_then(|g| g.data().ok().map(|d| d.to_vec()))
    );
}

#[test]
fn gauge_diagnostic_print_v_3x3() {
    let a = no_grad_leaf(&A3, &[3, 3]);
    let (w, v) = linalg_fwd::eig(&a).unwrap();
    eprintln!("ferrotorch 3x3 W = {:?}", w.data().unwrap().to_vec());
    eprintln!("ferrotorch 3x3 V = {:?}", v.data().unwrap().to_vec());
}

// ---------------------------------------------------------------------------
// (#1591 RESOLUTION) Why the exact-LAPACK-gauge boundary is UNMATCHABLE, and
// what ferrotorch guarantees instead.
//
// The original pin (`gauge_sensitive_boundary_torch_raises_ferrotorch_must_too`,
// removed below) asserted ferrotorch must RAISE at the SAME loss coefficient as
// torch for the PHASE-DEPENDENT loss `c*sum(V.real)`, c=0.025. That is
// fundamentally unmatchable: complex eigenvectors are defined only up to a
// per-column phase `e^{i phi}` (torch documents this at
// `FunctionsManual.cpp:3867-3879`). The #1590 guard's threshold
// `|imag(diag(V^H gV))| > 1e-2` is CORRECT and matches torch byte-for-byte, but
// `imag(diag(V^H gV))` is itself GAUGE-DEPENDENT — it scales with the arbitrary
// per-column phase. ferrotorch's eig (faer) emits different per-column phases
// than torch's LAPACK `geev` (see gauge_diagnostic_print_v_3x3):
//   torch    complex column 0 = [0.710717+0j, ...]   (LAPACK gauge)
//   ferray   complex column ~  [real-positive pivot]  (canonical gauge, #1591)
// so the guard fires at a different `c`. Matching torch's exact boundary would
// require replicating LAPACK geev's phases (impractical, and not the contract).
//
// What is mathematically MEANINGFUL:
//   - For PHASE-INVARIANT losses (the only well-posed kind) the gradient is
//     gauge-FREE — ferrotorch matches torch exactly (see
//     mixed_3x3_phase_invariant_matches_torch / phase_invariant_* above).
//   - For PHASE-DEPENDENT losses the value/gradient is ILL-DEFINED (depends on
//     the arbitrary phase); torch errors to protect the user; ferrotorch's
//     guard also errors for grossly-phase-dependent losses (see
//     mixed_3x3_phase_dependent_errors_like_torch). The losses in any divergent
//     window are mathematically meaningless regardless of gauge.
//
// The #1591 fix canonicalizes ferrotorch's eig eigenvector PHASE deterministically
// (`canonicalize_complex_eigenvector_phase` in linalg.rs: each column rotated by
// e^{-i phi} so its largest-magnitude component is real-positive), mirroring the
// eigh #1584 sign canonicalization. This does NOT match LAPACK's gauge (impossible)
// but makes ferrotorch's eig output REPRODUCIBLE + well-defined. The tests below
// pin the well-posed, gauge-robust quantities the resolution prescribes.

/// (#1591) eig eigenvectors are now DETERMINISTIC: the phase canonicalization
/// gives a reproducible gauge, so calling `eig` twice on the same input yields
/// byte-for-byte identical `V` (and `W`). This replaces the unmatchable
/// exact-LAPACK-gauge boundary assertion.
#[test]
fn eig_eigenvectors_are_deterministic_after_phase_canonicalization() {
    let a1 = no_grad_leaf(&A3, &[3, 3]);
    let a2 = no_grad_leaf(&A3, &[3, 3]);
    let (w1, v1) = linalg_fwd::eig(&a1).unwrap();
    let (w2, v2) = linalg_fwd::eig(&a2).unwrap();
    let (wv1, wv2) = (w1.data().unwrap().to_vec(), w2.data().unwrap().to_vec());
    let (vv1, vv2) = (v1.data().unwrap().to_vec(), v2.data().unwrap().to_vec());
    assert_eq!(
        wv1, wv2,
        "eig eigenvalues must be deterministic across calls"
    );
    assert_eq!(
        vv1, vv2,
        "eig eigenvectors must be deterministic across calls (canonical phase)"
    );
}

/// (#1591) The canonical gauge makes each complex eigenvector column's
/// LARGEST-MAGNITUDE component real-POSITIVE (its imag part ~0, real part > 0).
/// This is the deterministic phase contract `canonicalize_complex_eigenvector_phase`
/// establishes — the analog of eigh's real-positive-pivot sign canonicalization.
#[test]
fn eig_canonical_gauge_pivot_is_real_positive() {
    let a = no_grad_leaf(&A3, &[3, 3]);
    let (_w, v) = linalg_fwd::eig(&a).unwrap();
    let vv = v.data().unwrap().to_vec(); // [n,n,2] interleaved [re,im]
    let n = 3usize;
    for col in 0..n {
        // largest-magnitude pivot row in this column
        let mut best_row = 0usize;
        let mut best_mag = f64::NEG_INFINITY;
        for row in 0..n {
            let base = 2 * (row * n + col);
            let mag = vv[base] * vv[base] + vv[base + 1] * vv[base + 1];
            if mag > best_mag {
                best_mag = mag;
                best_row = row;
            }
        }
        let base = 2 * (best_row * n + col);
        let (re, im) = (vv[base], vv[base + 1]);
        assert!(
            im.abs() <= 1e-12,
            "col {col}: pivot imag part must be ~0 after phase canon, got {im}"
        );
        assert!(
            re > 0.0,
            "col {col}: pivot real part must be > 0 after phase canon, got {re}"
        );
    }
}

/// (#1591) Cross-check that the canonical phase did NOT disturb the gauge-free
/// gradient: the phase-INVARIANT loss on A3 STILL matches LIVE torch float64.
/// (This is the same assertion as mixed_3x3_phase_invariant_matches_torch, kept
/// here adjacent to the resolution to document that canonicalizing the forward
/// phase cannot change a phase-invariant gradient — the loss does not see the
/// phase. If this regressed, canonicalization broke the gauge invariance.)
#[test]
fn phase_invariant_grad_unchanged_by_canonicalization_matches_torch() {
    let a = leaf(&A3, &[3, 3]);
    let (_w, v) = linalg_fwd::eig(&a).unwrap();
    let m = [0.5, -0.3, 0.1, 0.2, 0.8, -0.4, 0.6, 0.1, 0.7];
    let loss = eigvec_phase_invariant_loss(&v, &m, 3);
    loss.backward()
        .expect("phase-invariant loss must NOT error after canonicalization");
    let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    // LIVE torch.autograd.grad(((V.real^2+V.imag^2)*M).sum(), A) — gauge-free:
    let torch = [
        -0.018451, 0.165583, -0.131945, 0.185531, -0.029623, -0.158005, 0.005740, -0.038701,
        0.048074,
    ];
    assert_close(
        &g,
        &torch,
        1e-5,
        "phase-invariant grad still matches torch post-canon",
    );
}
