//! Divergence-coverage test for #1238 (fake_quantize_per_tensor_affine REQ-1)
//! audit (commit `77781d844`).
//!
//! ## Pin
//!
//! The commit ships `grad_fns::quantize_grad::fake_quantize_per_tensor_affine`
//! at `ferrotorch-core/src/grad_fns/quantize_grad.rs:84` with the docstring
//! claim:
//!
//! > matches the upstream CPU kernel at
//! > `aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp:2655-2715
//! > _fake_quantize_tensor_helper` byte-for-byte
//!
//! and the per-element body at `:252-254`:
//!
//! ```text
//! let qval_f = zp_f64 + (x_f64 * inv_scale).round_ties_even();
//! let qval_clamped = qmax_f64.min(qmin_f64.max(qval_f));
//! let dq_f64 = (qval_clamped - zp_f64) * scale;
//! ```
//!
//! ## Divergence
//!
//! Upstream's kernel runs the rounding arithmetic at **f32** precision (the
//! stub at `aten/src/ATen/native/quantized/FakeQuantAffine.h:13-20` takes
//! `float sc`; the kernel at `QuantizedOpKernels.cpp:2665` computes
//! `float inv_scale = 1.0f / sc;` and at `:2683`/`:2703`
//! `auto qval_f = z_point + std::nearbyint(*input_val * inv_scale);`
//! evaluates as `scalar_t * float` which for `scalar_t = float` is
//! `float * float -> float`).
//!
//! Ferrotorch promotes `input` and `scale` to `f64` and computes the entire
//! rounding chain at `f64`. For inputs that are exact-half boundaries in f32
//! but NOT in f64 (the f32 cast adds a sub-ULP perturbation that survives
//! into the f64 product), the two paths disagree by one quantization step.
//!
//! ## Repro (live torch 2.11.0+cu130, 2026-05-25)
//!
//! ```python
//! import torch
//! input_t = torch.tensor([0.025], dtype=torch.float32)
//! out = torch.fake_quantize_per_tensor_affine(input_t, 0.05, 64, -128, 127)
//! # torch returns: 0.0
//! #   reason: x_f32 = 0.025_f32 (bit pattern 0x3CCCCCCD = 0.02500000037252903)
//! #           but f32 arithmetic: 0.025_f32 * (1/0.05)_f32 = 0.5_f32 EXACTLY
//! #           (because the rounding in the f32 multiply absorbs the input's
//! #           sub-ULP offset). nearbyint(0.5) under FE_TONEAREST is 0 (banker).
//! #           qval_clamped = 64 + 0 = 64; dq = (64-64) * 0.05 = 0.0.
//! ```
//!
//! Ferrotorch's f64 path:
//!
//! ```text
//! x_f64 = 0.025_f32 -> 0.02500000037252903_f64  (exact bit-extension of f32)
//! inv_scale_f64 = 1.0 / 0.05 = 20.000000000000004
//! prod_f64 = x_f64 * inv_scale_f64 = 0.5000000074505806   (> 0.5)
//! round_ties_even(0.5000000074505806) = 1   (not a tie — strictly above .5)
//! qval_f = 64 + 1 = 65
//! qval_clamped = 65
//! dq = (65 - 64) * 0.05 = 0.05
//! ```
//!
//! Independently confirmed via the Python repro above; `assert
//! out.item() == 0.0` against torch, while the ferrotorch surface returns
//! `0.05` (one quantization step off).
//!
//! ## Why the builder's own banker-rounding test missed this
//!
//! `fake_quantize_uses_banker_rounding_on_half_boundaries` at
//! `quantize_grad.rs:807` uses `scale = 1.0` and integer-plus-half inputs
//! (`0.5, 1.5, 2.5, ...`). With `scale = 1.0` the f32 vs f64 inv_scale are
//! both exactly `1.0` and `x / 1.0` is bit-identical at either precision, so
//! the f64 path coincidentally matches the f32 path on those samples. The
//! test passes — but it only checks one corner of the precision contract,
//! not the precision contract itself.
//!
//! The oracle's hand-crafted samples
//! (`tools/parity-sweep/oracle.py:172-244`) likewise avoid this regime: they
//! use `torch.randn * 0.5` with scales `{0.01, 0.05, 0.1, 1.0, 100.0}`, none
//! of which land on the precise f32 half-boundary boundary the divergence
//! requires. That is why "72/72 passed" in the parity sweep — the oracle is
//! not adversarial here.
//!
//! ## R-CHAR-3 compliance
//!
//! Expected values are taken from live `torch.fake_quantize_per_tensor_affine`
//! against `torch 2.11.0+cu130` on 2026-05-25 (full repro snippet in this
//! doc comment). They are NOT copied from any ferrotorch source.
//!
//! ## Tracking
//!
//! Tracking: #1259 (blocker, high). The test is INTENTIONALLY un-`#[ignore]`d
//! so that the failing assertion blocks until the precision contract is
//! either (a) reconciled to f32 arithmetic byte-for-byte with upstream, or
//! (b) the design doc is updated to declare ferrotorch's f64 path as the
//! contract with explicit deviation from upstream documented at REQ-1.

use ferrotorch_core::grad_fns::quantize_grad::fake_quantize_per_tensor_affine;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn t(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
}

#[test]
fn fake_quantize_matches_torch_at_f32_half_boundary_x025_scale005() {
    // Live torch reference (torch 2.11.0+cu130, 2026-05-25):
    //   >>> torch.fake_quantize_per_tensor_affine(
    //   ...     torch.tensor([0.025], dtype=torch.float32),
    //   ...     0.05, 64, -128, 127,
    //   ... ).item()
    //   0.0
    let input = t(vec![0.025_f32], vec![1]);
    let out = fake_quantize_per_tensor_affine(&input, 0.05, 64, -128, 127).unwrap();
    let actual = out.data().unwrap();
    // Upstream produces 0.0 because at f32 precision:
    //   inv_scale_f32 = 1.0_f32 / 0.05_f32      = 20.000_f32  (rounding hides ULP)
    //   prod_f32      = 0.025_f32 * inv_scale_f32 = 0.5_f32 EXACTLY
    //   nearbyint(0.5) under FE_TONEAREST       = 0  (banker)
    //   qval_clamped  = 64; dq = (64-64) * 0.05  = 0.0
    let expected_torch: f32 = 0.0;
    assert_eq!(
        actual[0],
        expected_torch,
        "fake_quantize_per_tensor_affine(0.025_f32, scale=0.05, zp=64, qmin=-128, qmax=127) \
         must match torch's f32-precision kernel: torch returns {expected_torch}, \
         ferrotorch returned {got}. Upstream stub takes `float sc` per \
         FakeQuantAffine.h:13-20; the f32 product `0.025_f32 * (1/0.05)_f32` is \
         exactly 0.5_f32 (banker-rounds to 0). Ferrotorch's f64 promotion produces \
         prod = 0.5000000074505806 which rounds up to 1, yielding dq = 0.05.",
        got = actual[0],
    );
}

#[test]
fn fake_quantize_matches_torch_at_f32_half_boundary_x175_scale005() {
    // Live torch reference (torch 2.11.0+cu130, 2026-05-25):
    //   >>> torch.fake_quantize_per_tensor_affine(
    //   ...     torch.tensor([0.175], dtype=torch.float32),
    //   ...     0.05, 64, -128, 127,
    //   ... ).item()
    //   0.2
    //   (3.5 rounded banker = 4, dq = (4) * 0.05 = 0.2)
    let input = t(vec![0.175_f32], vec![1]);
    let out = fake_quantize_per_tensor_affine(&input, 0.05, 64, -128, 127).unwrap();
    let actual = out.data().unwrap();
    let expected_torch: f32 = 0.2;
    assert!(
        (actual[0] - expected_torch).abs() < 1e-6,
        "fake_quantize_per_tensor_affine(0.175_f32, scale=0.05, zp=64, qmin=-128, qmax=127) \
         must match torch's f32-precision kernel: torch returns {expected_torch}, \
         ferrotorch returned {got}. The f32 product `0.175_f32 * (1/0.05)_f32` lands \
         at a half boundary that rounds to 4 under torch's banker rounding (giving \
         dq = 0.2); ferrotorch's f64 path produces a product that rounds to 3, \
         yielding dq = 0.15.",
        got = actual[0],
    );
}

#[test]
fn fake_quantize_batched_f32_half_boundary_matches_torch() {
    // Live torch reference (torch 2.11.0+cu130, 2026-05-25):
    //   >>> torch.fake_quantize_per_tensor_affine(
    //   ...     torch.tensor([0.025, 0.075, 0.125, 0.175, 0.225], dtype=torch.float32),
    //   ...     0.05, 64, -128, 127,
    //   ... ).tolist()
    //   [0.0, 0.1, 0.1, 0.2, 0.2]
    //
    // The two divergent cells are index 0 (0.025 -> torch 0.0, ferro 0.05) and
    // index 3 (0.175 -> torch 0.2, ferro 0.15). The other three values
    // (0.075, 0.125, 0.225) coincidentally have f32 and f64 paths agreeing
    // because their fractional-part products land cleanly on the same side of
    // the .5 boundary.
    let input = t(vec![0.025_f32, 0.075, 0.125, 0.175, 0.225], vec![5]);
    let out = fake_quantize_per_tensor_affine(&input, 0.05, 64, -128, 127).unwrap();
    let actual = out.data().unwrap();
    let expected_torch: [f32; 5] = [0.0, 0.1, 0.1, 0.2, 0.2];
    for (i, (&a, &e)) in actual.iter().zip(expected_torch.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-6,
            "fake_quantize_per_tensor_affine batched at idx={i}: torch expects {e}, \
             ferrotorch returned {a}; divergence source is f64-vs-f32 arithmetic in \
             the inv_scale * input product. Upstream stub at FakeQuantAffine.h:13-20 \
             uses `float sc`; ferrotorch hoists to f64 at quantize_grad.rs:252.",
        );
    }
}
