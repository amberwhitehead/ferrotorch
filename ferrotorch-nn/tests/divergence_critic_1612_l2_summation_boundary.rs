//! Divergence pin (ACToR critic, re-audit of commit `9587ff1bb`, #1612):
//! the #1612 fix moved the `max_norm` renorm DECISION from f64 into the
//! weight's native dtype `T` (f32) — but for the DEFAULT `norm_type == 2.0`
//! it computes the L2 norm with a SCALAR `Σ v.abs().powf(2.0f32)` accumulation
//! rooted by `.powf(0.5f32)`, which still does NOT match torch's `at::norm(2.0)`
//! at the f32-norm boundary. So the boundary decision the commit claims to fix
//! is STILL wrong for a different (and far more common) class of f32 rows.
//!
//! ## Root cause — torch's L2 path is `v*v` lane-summed, ferrotorch's is `powf`
//!
//! For the default `norm_type == 2.0`, a contiguous last-dim f32 reduction goes
//! through the VECTORIZED L2 kernel
//! (`/home/doll/pytorch/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:221-256`):
//! ```cpp
//!   acc_fvec += data_fvec0 * data_fvec0;   // squares, in opmath_type<f32> == float
//!   ...
//!   result_data[0] = scalar_t(std::sqrt(buffer[0]));   // sqrt, stored back as f32
//! ```
//! It SQUARES (`v*v`) and lane-sums in f32, then `sqrt`. ferrotorch
//! (`ferrotorch-nn/src/embedding.rs:153-162`, the NormOps finite-p arm that
//! `norm_type == 2.0` falls into) instead does:
//! ```rust
//!   for &v in row { acc += v.abs().powf(p_t); }   // p_t == 2.0f32, powf not v*v
//!   acc.powf(inv_p)                               // inv_p == 0.5f32, powf not sqrt
//! ```
//! `v.abs().powf(2.0f32)` is NOT bit-identical to `v*v` in f32, and
//! `acc.powf(0.5f32)` is NOT bit-identical to `acc.sqrt()`. Across random f32
//! rows the two L2 results differ on ~5.6% of rows (20k-row sweep), and on a
//! fraction of those ferrotorch's norm lands ONE ULP ABOVE torch's f32 norm.
//!
//! ## The input (legal f32, reproducible from f32 literals)
//!
//! Row `[3.6006885, 18.799816, 0.4159323, -2.6984854, -4.786058, 25.550726]`:
//!   - torch `at::norm(2.0)` (f32) == `32.39751052856445`  (bits `0x4201970d`)
//!   - ferrotorch `Σ powf(|v|,2f32) then powf(.,0.5f32)` == `32.39751434326172`
//!     (bits `0x4201970e`) — exactly ONE ULP higher (verified with native Rust
//!     `f32::powf`, not numpy).
//!
//! With `max_norm == 32.39751052856445` (== torch's f32 norm), torch sees
//! `norm == max_norm` so `norm > max_norm` is FALSE and the row is UNCHANGED
//! (verified live torch 2.11.0+cu130, `F.embedding([0], w, max_norm, 2.0)`
//! leaves the weight byte-identical). ferrotorch's norm `32.39751434326172 >
//! 32.39751052856445`, so it RENORMS, scaling the row by
//! `32.39751052856445 / (32.39751434326172 + 1e-7) ≈ 0.99999988`. The largest
//! element shrinks by `3.81e-6` — 38x the embedding sweep atol of 1e-7.
//!
//! This is the SAME divergence class #1612 claimed to fix; the f32-accumulate
//! change closed the f64-vs-f32 boundary but opened (left open) the
//! powf-vs-`v*v` summation-method boundary for the default L2 path.
//!
//! R-CHAR-3: `TORCH_UNCHANGED_ROW` is the LIVE torch `F.embedding` output (the
//! input row, byte-for-byte unchanged because torch's f32 norm is not >
//! max_norm). `TORCH_F32_L2_NORM` is torch's `at::norm(2.0)` evaluated live.
//! Neither is copied from ferrotorch.
//!
//! Upstream: `aten/src/ATen/native/Embedding.cpp:202-203` (`row.norm(norm_type)
//! .item<double>()`), `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:221-256`
//! (vectorized L2 = `v*v` + `sqrt`, NOT powf).
//! ferrotorch: `ferrotorch-nn/src/embedding.rs:153-162` (`powf(.,2)` + `powf(.,0.5)`).
//!
//! Tracking: #1614.

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::{Embedding, EmbeddingBag, EmbeddingBagMode};

fn tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// The adversarial weight row. Its torch f32 L2-norm is `TORCH_F32_L2_NORM`,
/// but ferrotorch's `Σ powf(|v|,2)` accumulation gives one ULP more.
const ROW: [f32; 6] = [
    3.6006885, 18.799816, 0.4159323, -2.6984854, -4.786058, 25.550726,
];

/// LIVE torch `at::norm(row, 2.0)` in f32 (verified torch 2.11.0+cu130,
/// 2026-05-28). Bits `0x4201970d`.
const TORCH_F32_L2_NORM: f32 = 32.39751052856445;

/// LIVE torch `F.embedding([0], w, max_norm=TORCH_F32_L2_NORM, norm_type=2.0)`:
/// the row is returned UNCHANGED (torch's f32 norm == max_norm, not greater).
const TORCH_UNCHANGED_ROW: [f32; 6] = ROW;

/// The embedding parity-sweep tolerance (atol=1e-7, the default bucket in
/// `tools/parity-sweep/runner/src/main.rs tolerance_for`). Assert at this
/// envelope so the test fails exactly when the divergence exceeds what the
/// sweep tolerates — the observed divergence is ~3.81e-6, 38x this atol.
const ATOL: f32 = 1e-7;

/// Divergence: `Embedding::with_max_norm(TORCH_F32_L2_NORM).with_norm_type(2.0)`
/// renorms a row torch's `F.embedding` leaves intact. torch's L2 path squares
/// (`v*v`) and lane-sums in f32 then `sqrt`s; ferrotorch's NormOps arm uses
/// `Σ powf(|v|,2f32)` then `powf(.,0.5f32)`, landing one ULP higher, so the
/// `norm > max_norm` decision flips. Upstream: Embedding.cpp:202-203 +
/// ReduceOpsKernel.cpp:221-256. ferrotorch: embedding.rs:153-162. Tracking: #1614.
#[test]
fn divergence_embedding_l2_powf_vs_vv_summation_boundary_renorms() {
    // Two-row weight: row 0 is the boundary row, row 1 is arbitrary.
    let mut data = ROW.to_vec();
    data.extend_from_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    let weight = tensor(&data, &[2, 6]);

    let layer = Embedding::<f32>::from_pretrained(weight, None)
        .unwrap()
        .with_max_norm(TORCH_F32_L2_NORM as f64)
        .with_norm_type(2.0);

    let indices = tensor(&[0.0], &[1]);
    let out = Module::<f32>::forward(&layer, &indices).unwrap();
    assert_eq!(out.shape(), &[1, 6], "embedding output shape");
    let got = out.data().unwrap();

    for (i, (&g, &e)) in got.iter().zip(TORCH_UNCHANGED_ROW.iter()).enumerate() {
        assert!(
            (g - e).abs() <= ATOL,
            "element {i}: ferrotorch={g} vs torch F.embedding={e} \
             (absdiff {:.3e} > atol {ATOL:.0e}); torch's f32 L2 norm \
             ({TORCH_F32_L2_NORM}) equals max_norm so torch does NOT renorm, \
             but ferrotorch's powf-summed L2 norm exceeds max_norm by one ULP \
             so it scales the row down (#1612 fix incomplete for default L2)",
            (g - e).abs()
        );
    }

    // The persisted weight must also be byte-for-byte unchanged (torch mutates
    // the weight in place; here it does not, because no row exceeds max_norm).
    let w = layer.weight.data().unwrap();
    for (i, &orig) in ROW.iter().enumerate() {
        assert_eq!(
            w[i], orig,
            "persisted weight row 0 elt {i} must stay byte-identical (torch \
             leaves the weight untouched at the f32-norm boundary)"
        );
    }
}

/// Divergence: the SAME shared `renorm_weight_rows_in_place` backs EmbeddingBag,
/// so `EmbeddingBag::with_max_norm(TORCH_F32_L2_NORM)` over a bag touching the
/// boundary row renorms it where torch's `F.embedding_bag` would not. Confirms
/// the #1612-incomplete L2 boundary is not Embedding-specific. Tracking: #1614.
#[test]
fn divergence_embedding_bag_l2_powf_vs_vv_summation_boundary_renorms() {
    let mut bag = EmbeddingBag::<f32>::new(2, 6, EmbeddingBagMode::Sum).unwrap();
    let mut data = ROW.to_vec();
    data.extend_from_slice(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    // Set the pretrained weight via the public `Module::parameters_mut` +
    // `Parameter::set_data` path (the same path the #1441 parity-sweep runner
    // arm uses to install pretrained bag weights).
    {
        let mut params = Module::<f32>::parameters_mut(&mut bag);
        params[0].set_data(Tensor::from_storage(
            TensorStorage::cpu(data),
            vec![2, 6],
            true,
        ).unwrap());
    }
    let bag = bag
        .with_max_norm(TORCH_F32_L2_NORM as f64)
        .with_norm_type(2.0);

    // Single bag over index 0 only: the bag output equals the (renormed-or-not)
    // row 0. torch leaves row 0 intact, so the Sum-bag output == ROW.
    let input = tensor(&[0.0], &[1]);
    let offsets = [0usize];
    let out = bag.forward_bag(&input, &offsets).unwrap();
    assert_eq!(out.shape(), &[1, 6], "embedding_bag output shape");
    let got = out.data().unwrap();

    for (i, (&g, &e)) in got.iter().zip(TORCH_UNCHANGED_ROW.iter()).enumerate() {
        assert!(
            (g - e).abs() <= ATOL,
            "bag elt {i}: ferrotorch={g} vs torch F.embedding_bag={e} \
             (absdiff {:.3e} > atol {ATOL:.0e}); shared renorm fn renorms a row \
             torch's f32 L2 norm leaves at the boundary (#1612 incomplete)",
            (g - e).abs()
        );
    }

    // Persisted weight row 0 unchanged.
    let w = Module::<f32>::parameters(&bag)[0].data().unwrap();
    for (i, &orig) in ROW.iter().enumerate() {
        assert_eq!(
            w[i], orig,
            "EmbeddingBag persisted weight row 0 elt {i} must stay byte-identical"
        );
    }
}
