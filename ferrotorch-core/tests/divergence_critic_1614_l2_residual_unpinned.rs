//! Divergence pin (ACToR critic, re-audit of commit `3c422f8fd`, #1614):
//! `ferrotorch_core::simd_reduce::l2_norm_f32_torch` is a fixed-FMA-tail model
//! of torch's vectorized last-dim f32 L2 kernel. It matches live torch
//! `Tensor::norm(2.0)` (== `at::norm(2.0)` == `torch.linalg.vector_norm(.,2)`)
//! on ~97.8% of random f32 rows — INDEPENDENTLY re-measured by this critic over
//! an 8800-row corpus emitted by the REAL compiled Rust function and compared
//! bit-for-bit to live torch 2.11.0+cu130 on this AVX2 host (8607/8800 =
//! 97.81%). The match is genuine BIT-equality, not a tolerance. The residual
//! ~2.2% diverge from torch by exactly one ULP, and the divergence is
//! TWO-SIDED (model one ULP HIGH on some rows, one ULP LOW on others),
//! corroborating that torch's compiled scalar remainder contracts FMA
//! value-dependently: no single deterministic FMA-on/FMA-off tail reaches 100%.
//!
//! ## Why this file exists (the residual was UN-pinned by the re-row)
//!
//! The builder RE-ROWED the #1441 regression guard (and the in-lib
//! `embedding.rs` test) off the `b4` residual row onto a row both torch and the
//! model agree on, AND added a unit test
//! `simd_reduce::tests::known_residual_one_ulp_below_torch` that pins the
//! MODEL'S (wrong) bits and only asserts the *direction* (`abs(model-torch) <=
//! 1`). That unit test PASSES, so NOTHING in the suite fails when the model
//! disagrees with torch — the residual is *documented*, not *caught*. This file
//! pins the residual AS A DIVERGENCE FROM TORCH: every assertion's expected
//! value is the LIVE TORCH bit pattern, so each test FAILS against the current
//! model. That failure keeps the residual on the books under the tracking issue
//! instead of swept under the rug. The tests are `#[ignore]`'d (the issue tracks
//! it; this is a documented value-dependent-FMA hard limit, not a CI block).
//!
//! ## Residual rows (live torch 2.11.0+cu130, 2026-05-28, this AVX2 host)
//!
//! Expected value = bit pattern live torch `t.norm(2.0)` produced for an f32
//! tensor of the row (R-CHAR-3: torch is the oracle; the expected values
//! DISAGREE with ferrotorch's output — they are NOT copied from it). Row
//! elements are pinned by their EXACT f32 bit patterns via `f32::from_bits`, so
//! there is zero literal-rounding ambiguity: the bytes the Rust function sees
//! are byte-identical to the bytes the torch oracle saw.
//!
//! | row (elem f32 bits) | torch norm bits | model norm bits | dir |
//! |---|---|---|---|
//! | `c0a2f24d c1108b46 c2c6227a c10d6af4` (old #1612 b4, norm == 100.0) | `0x42c80000` | `0x42c80001` | model 1 ULP HIGH |
//! | `41aa264e 41e32348 c1a01b51 c1acee2a` (len4) | `0x42387237` | `0x42387238` | model 1 ULP HIGH |
//! | `4198ca60 bf9417d0 c11052b4 c175cc08` (len4) | `0x41d12550` | `0x41d1254f` | model 1 ULP LOW |
//!
//! The old #1612 b4 row is the SAME row the #1441/in-lib regression guards were
//! re-rowed AWAY from: torch's f32 norm of it is EXACTLY `100.0` (`0x42c80000`),
//! the renorm boundary those tests were designed to pin. The model gives
//! `0x42c80001` (one ULP HIGH), which is `> max_norm=100.0`, so a `max_norm`
//! renorm using this model would CLIP a row torch leaves intact — exactly the
//! #1441/#1612 over-renorm divergence. Re-rowing the regression guards off this
//! row removed the coverage that would catch this residual; this test restores
//! it. (The `known_residual_one_ulp_below_torch` unit test in `simd_reduce.rs`
//! pins a DIFFERENT len-5 residual and asserts the MODEL'S bits, so even that
//! row is not pinned against torch anywhere.)
//!
//! Upstream: `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:222-255` (vectorized
//! last-dim L2 kernel torch actually runs); `aten/src/ATen/native/Embedding.cpp:202-203`
//! (the renorm decision the #1612/#1441 boundary feeds).
//! ferrotorch: `ferrotorch-core/src/simd_reduce.rs::l2_norm_f32_torch`.
//!
//! Tracking: #1617 (residual unpinned after re-row); refs #1614 #1441 #1612.

use ferrotorch_core::simd_reduce::l2_norm_f32_torch;

/// Assert the model reproduces the LIVE torch `at::norm(2.0)` f32 bits.
/// `torch_bits` is the live oracle value (R-CHAR-3); the model currently
/// DISAGREES by one ULP, so this FAILS — that failure IS the pinned divergence.
#[track_caller]
fn assert_matches_torch(elem_bits: &[u32], torch_bits: u32) {
    let row: Vec<f32> = elem_bits.iter().map(|&b| f32::from_bits(b)).collect();
    let got = l2_norm_f32_torch(&row).to_bits();
    assert_eq!(
        got,
        torch_bits,
        "l2_norm_f32_torch({row:?}) = {:#010x} ({}); live torch at::norm(2.0) \
         f32 = {torch_bits:#010x} ({}); residual one-ULP divergence (#1617)",
        got,
        f32::from_bits(got),
        f32::from_bits(torch_bits),
    );
}

/// Divergence: the old #1612 `b4` row whose torch f32 L2-norm is EXACTLY
/// `100.0` (`0x42c80000`). The model gives `0x42c80001` (one ULP HIGH), so a
/// `max_norm=100.0` renorm using this model would clip a row torch leaves
/// intact — the precise #1441/#1612 over-renorm bug. The regression guards were
/// re-rowed off this row; this re-pins it against TORCH so the divergence stays
/// on the books. Row `[-5.0920777, -9.034002, -99.06734, -8.838612]`.
#[test]
#[ignore = "divergence: l2_norm_f32_torch gives 0x42c80001 for the norm==100.0 b4 row, torch gives 0x42c80000 (1 ULP HIGH -> over-renorm); value-dependent-FMA residual re-rowed away from by #1441 guard; tracking #1617"]
fn divergence_residual_old_1612_b4_norm_100() {
    let elem_bits = [0xc0a2_f24d, 0xc110_8b46, 0xc2c6_227a, 0xc10d_6af4];
    // live torch 2.11.0+cu130: row.norm(2.0) f32 == 100.0 == bits 0x42c80000
    assert_matches_torch(&elem_bits, 0x42c8_0000);
}

/// Divergence: a len-4 residual row, model one ULP HIGH vs torch (taken
/// directly from the real-Rust-vs-live-torch corpus diff). Row f32 values
/// `[21.268703, 28.392227, -20.013338, -21.616291]`.
#[test]
#[ignore = "divergence: l2_norm_f32_torch gives 0x42387238, torch gives 0x42387237 (1 ULP HIGH); value-dependent-FMA residual; tracking #1617"]
fn divergence_residual_len4_one_ulp_high() {
    let elem_bits = [0x41aa_264e, 0x41e3_2348, 0xc1a0_1b51, 0xc1ac_ee2a];
    // live torch 2.11.0+cu130: row.norm(2.0) f32 == bits 0x42387237
    assert_matches_torch(&elem_bits, 0x4238_7237);
}

/// Divergence: a len-4 residual row, model one ULP LOW vs torch. Confirms the
/// residual is TWO-SIDED — it cannot be corrected by a fixed nudge, consistent
/// with torch's value-dependent FMA contraction (the builder's "100% is
/// impossible portably" claim). Row f32 values
/// `[19.098816, -1.1569767, -9.020191, -15.362312]`.
#[test]
#[ignore = "divergence: l2_norm_f32_torch gives 0x41d1254f, torch gives 0x41d12550 (1 ULP LOW); two-sided value-dependent-FMA residual; tracking #1617"]
fn divergence_residual_len4_one_ulp_low() {
    let elem_bits = [0x4198_ca60, 0xbf94_17d0, 0xc110_52b4, 0xc175_cc08];
    // live torch 2.11.0+cu130: row.norm(2.0) f32 == bits 0x41d12550
    assert_matches_torch(&elem_bits, 0x41d1_2550);
}
