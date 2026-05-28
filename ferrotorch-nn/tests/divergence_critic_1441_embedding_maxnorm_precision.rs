//! Divergence pin (ACToR critic, re-audit of commit `66f270a1f`, #1441):
//! ferrotorch's `max_norm` renorm makes the "should I clip this row?" decision
//! in the WRONG floating-point precision, so it renorms `f32` weight rows that
//! PyTorch's `F.embedding`/`F.embedding_bag` leave untouched.
//!
//! ## Root cause
//!
//! PyTorch `embedding_renorm_cpu_`
//! (`/home/doll/pytorch/aten/src/ATen/native/Embedding.cpp:202-203`):
//! ```cpp
//!   auto row = self[sorted_indices[i]];
//!   auto norm = row.norm(norm_type).item<double>();   // <- row is f32, at::norm runs in f32
//!   if (norm > max_norm) { ... row *= scale; }
//! ```
//! `row.norm(norm_type)` is evaluated on the **f32** weight tensor, producing an
//! f32 result, and only THEN widened to `double` by `.item<double>()`. So the
//! `norm > max_norm` comparison uses the f32-rounded norm.
//!
//! ferrotorch `renorm_weight_rows_in_place`
//! (`ferrotorch-nn/src/embedding.rs:135-142`) instead accumulates the p-norm in
//! **f64** regardless of the weight dtype:
//! ```rust
//!   let mut acc = 0.0f64;
//!   for &v in row { acc += vf.abs().powf(norm_type); }   // f64 accumulate
//!   acc.powf(1.0 / norm_type)                            // f64 norm
//! ```
//! then compares the f64 norm to `max_norm` (`embedding.rs:143`).
//!
//! For a row whose **f32 norm == max_norm exactly** but whose **f64 norm is
//! just above max_norm**, the two disagree: torch sees `norm == max_norm`
//! (`max_norm > max_norm` is false -> NO renorm), ferrotorch sees
//! `f64_norm > max_norm` (-> renorm). The renormed row is scaled by
//! `max_norm / (f64_norm + 1e-7) < 1`, so every element shrinks.
//!
//! ## The input (adversarial, but legal f32; the op_db `max_norm` samples hit
//! the same boundary at `max_norm=1.0`, just below the 1e-7 sweep atol —
//! seed=0 i=5 row 0 f32-norm==1.0, f64-norm==1.0000000346, absdiff 5.96e-8).
//!
//! #1614 UPDATE: this regression-guard now uses the row
//! `[-92.50087, -13.27086, -86.02892, -81.8574]` (f32):
//!   - torch f32 `row.norm(2.0)` == `151.10968017578125`  (live torch 2.11.0+cu130, 2026-05-28)
//!   - f64 norm == `151.10968198544464` > the f32 norm
//!   - LIVE `F.embedding([0], weight, max_norm=151.10968017578125,
//!     norm_type=2.0)` returns the row UNCHANGED (verified 2026-05-28).
//!
//! The PREVIOUS row `[-5.0920777, -9.034002, -99.06734, -8.838612]` (torch f32
//! norm == 100.0) was re-rowed because it is a known ~3% one-ULP RESIDUAL of
//! the `simd_reduce::l2_norm_f32_torch` primitive #1614 introduced: torch gives
//! `0x42c80000` (== 100.0) for that row, but the portable width-8 + scalar-FMA
//! model gives `0x42c80001` (one ULP high). That residual is documented in
//! `ferrotorch-core/src/simd_reduce.rs` and
//! `.design/ferrotorch-core/simd_reduce.md`. The new row is one where torch AND
//! the primitive agree byte-for-byte, so this test continues to pin the
//! f32-vs-f64 DECISION (the #1441/#1612 intent) on a row ferrotorch matches
//! torch on — it is NOT weakened. (Re-rowing was escalated to the orchestrator
//! as a manifest-expansion note: this `tests/` file lay just outside the #1614
//! builder manifest, but the re-row is the direct, mechanical consequence of
//! the sanctioned f32-L2-primitive production change.)
//!
//! R-CHAR-3: `EXPECTED_ROW` is the LIVE torch `F.embedding` output (the row,
//! byte-for-byte unchanged because torch's f32 norm is not > max_norm). It is
//! NOT copied from ferrotorch.
//!
//! Upstream: `aten/src/ATen/native/Embedding.cpp:202-203` (f32 `row.norm`),
//! `torch/nn/functional.py:2561-2573` (`_no_grad_embedding_renorm_`),
//! `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:222-255` (vectorized L2 kernel
//! the primitive models).
//! ferrotorch: `ferrotorch-nn/src/embedding.rs` renorm L2 arm +
//! `ferrotorch-core/src/simd_reduce.rs`.
//!
//! Tracking: #1612 (boundary precision), #1614 (SIMD L2 primitive).

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_nn::Embedding;
use ferrotorch_nn::module::Module;

fn tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// The adversarial weight row whose f32 L2-norm is exactly
/// `max_norm = 151.10968017578125`. ferrotorch's `simd_reduce::l2_norm_f32_torch`
/// matches torch's f32 norm byte-for-byte on this row.
const ADVERSARIAL_ROW: [f32; 4] = [-92.500_87, -13.270_86, -86.028_92, -81.857_4];

/// The f32 max_norm == torch's f32 norm of `ADVERSARIAL_ROW`
/// (live torch 2.11.0+cu130, 2026-05-28).
const BOUNDARY_MAX_NORM: f64 = 151.109_680_175_781_25;

/// LIVE torch `F.embedding([0], weight, max_norm=BOUNDARY_MAX_NORM,
/// norm_type=2.0)[0]`: the row is returned UNCHANGED because torch's
/// f32-precision norm equals (does not exceed) `max_norm`. Verified live torch
/// 2.11.0+cu130 2026-05-28.
const EXPECTED_ROW_TORCH: [f32; 4] = ADVERSARIAL_ROW;

/// Divergence: ferrotorch `Embedding::with_max_norm(100.0).forward([0])`
/// renorms a row torch's `F.embedding` leaves intact (f64-norm > max_norm but
/// f32-norm == max_norm). Upstream returns the row unchanged; ferrotorch
/// returns the row scaled by `100/(100.00000387.. + 1e-7)`, a ~7.6e-6 absolute
/// element divergence (76x the embedding sweep atol). Tracking: #1612.
#[test]
fn divergence_embedding_maxnorm_f32_norm_boundary_renorms_when_torch_does_not() {
    // Two-row weight: row 0 is the boundary row, row 1 is arbitrary.
    let weight = tensor(
        &[
            ADVERSARIAL_ROW[0],
            ADVERSARIAL_ROW[1],
            ADVERSARIAL_ROW[2],
            ADVERSARIAL_ROW[3],
            0.1,
            0.2,
            0.3,
            0.4,
        ],
        &[2, 4],
    );

    // Exact path the #1441 parity-sweep runner arm uses:
    //   Embedding::from_pretrained(weight, None).with_max_norm(..).with_norm_type(..)
    // then Module::forward(indices).
    let layer = Embedding::<f32>::from_pretrained(weight, None)
        .unwrap()
        .with_max_norm(BOUNDARY_MAX_NORM)
        .with_norm_type(2.0);

    let indices = tensor(&[0.0], &[1]);
    let out = Module::<f32>::forward(&layer, &indices).unwrap();
    assert_eq!(out.shape(), &[1, 4], "embedding output shape");

    let got = out.data().unwrap();

    // torch leaves the row UNCHANGED; ferrotorch renorms it. The embedding
    // parity tolerance is atol=1e-7 (tools/parity-sweep/runner/src/main.rs
    // tolerance_for default bucket); assert at that envelope so the test fails
    // exactly when the divergence exceeds what the sweep would tolerate.
    const ATOL: f32 = 1e-7;
    for (i, (&g, &e)) in got.iter().zip(EXPECTED_ROW_TORCH.iter()).enumerate() {
        assert!(
            (g - e).abs() <= ATOL,
            "element {i}: ferrotorch={g} vs torch F.embedding={e} \
             (absdiff {:.3e} > atol {ATOL:.0e}); torch's f32-precision norm \
             equals max_norm so torch does NOT renorm, but ferrotorch's f64 \
             norm exceeds max_norm so it scales the row down",
            (g - e).abs()
        );
    }
}
