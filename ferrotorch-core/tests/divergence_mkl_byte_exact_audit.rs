//! Divergence-coverage tests for commit `8b90369c1` (closes #1348)
//!
//! The commit claims byte-for-byte matmul/bmm/linalg.matmul parity with
//! `torch 2.11.0+cu130` (MKL CPU build) when ferrotorch-core is built with
//! `--features mkl`. The mechanism:
//!
//!   1. `intel-mkl-src` vendors MKL 2020.1, `cblas-sys` provides the CBLAS
//!      FFI signatures, and `ops::linalg::mm_raw` / `mm_raw_bt` / `mm_raw_at`
//!      gain `#[cfg(feature = "mkl")]` branches that call `cblas_sgemm` /
//!      `cblas_dgemm` directly.
//!   2. A `.init_array` constructor `_ferrotorch_mkl_cbwr_init` exports
//!      `MKL_CBWR=COMPATIBLE` via POSIX `setenv` at library-load time so
//!      MKL's first-dispatch reads the SSE2-only reproducible branch (=3).
//!   3. An `OnceLock`-guarded `MKL_CBWR_Set(3)` FFI call provides
//!      defense-in-depth fallback.
//!   4. `tools/parity-sweep/runner/src/main.rs::tolerance_for` reads
//!      `ferrotorch_core::ops::linalg::MKL_ENABLED` at runtime to flip the
//!      matmul-family envelope from `rtol=1e-4` (faer fallback) to the
//!      default `(1e-5, 1e-7)` envelope.
//!
//! op_db's matmul SampleInputs enumerate shapes
//! `[20]@[20]`, `[5,10]@[10]`, `[10]@[10,5]`, `[5,10]@[10,5]`, `[5,0]@[0,10]`,
//! `[5,5,10]@[10]`, `[5,5,10]@[10,5]`, `[5,5,0]@[0,5]`, `[10]@[5,10,5]`,
//! `[5,10]@[5,10,5]`, `[0,0]@[5,0,0]`, `[5,5,10,10]@[5,5,10,5]`,
//! `[5,5,10,10]@[10]`, `[10]@[5,5,10,5]`, `[5,5,5]@[1,5,5]`. NONE of these
//! exercise:
//!   * thin matrices with extreme K (M=1, K=256, N=1)
//!   * the standalone `mm_raw_bt` / `mm_raw_at` transpose-fused paths
//!     (the matmul dispatcher never routes through them — only
//!     `MmBackward::backward` does, which the parity sweep does not
//!     exercise at all because the sweep is forward-only)
//!   * verification that the `.init_array` MKL_CBWR=COMPATIBLE trick
//!     actually engaged at runtime (parity-sweep success can't
//!     distinguish "CBWR=COMPATIBLE engaged" from "MKL 2020.1 default
//!     branch happens to match torch's MKL 2024.2 on op_db's small
//!     samples")
//!
//! Each probe below uses reference outputs computed directly via
//! `python3 -c 'import torch; ...'` on the same `torch 2.11.0+cu130`
//! build the commit claims parity against, embedded as IEEE-754 bit
//! patterns (R-CHAR-3: tautological literal-copy from the ferrotorch
//! side is forbidden; these bits trace to live torch invocations).
//!
//! Tracking: #1538 (mm_raw_at MKL transa=Trans path 1-ULP drift vs
//! torch — pinned by `divergence_mkl_mm_raw_at_direct`; the #1348
//! byte-exact claim holds only for the forward 2D mm/bmm/matmul paths
//! exercised by the forward-only parity sweep).

#![cfg(feature = "mkl")]

use ferrotorch_core::ops::linalg::{mm_raw, mm_raw_at, mm_raw_bt};

/// Reinterpret a `&[u32]` of IEEE-754 bit patterns as `&[f32]`.
fn bits_to_f32(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

/// Assert that two f32 slices are byte-identical (every bit pattern
/// matches). Distinguishes `+0.0` from `-0.0` and propagates NaN bit
/// patterns. This is the "byte-for-byte parity" the commit claims.
fn assert_byte_exact(actual: &[f32], expected_bits: &[u32], label: &str) {
    assert_eq!(
        actual.len(),
        expected_bits.len(),
        "{label}: length mismatch"
    );
    let mut mismatches = Vec::new();
    for (i, (&a, &eb)) in actual.iter().zip(expected_bits.iter()).enumerate() {
        let ab = a.to_bits();
        if ab != eb {
            mismatches.push((i, ab, eb, a, f32::from_bits(eb)));
        }
    }
    if !mismatches.is_empty() {
        let mut msg = format!(
            "{label}: byte-exact parity FAILED on {}/{} elements",
            mismatches.len(),
            actual.len()
        );
        for (i, ab, eb, av, ev) in mismatches.iter().take(8) {
            msg.push_str(&format!(
                "\n  [{i}] ferrotorch=0x{ab:08x} ({av}) torch=0x{eb:08x} ({ev}) ulp_diff={}",
                ab.abs_diff(*eb)
            ));
        }
        if mismatches.len() > 8 {
            msg.push_str(&format!("\n  ... and {} more", mismatches.len() - 8));
        }
        panic!("{msg}");
    }
}

// ---------------------------------------------------------------------------
// Probe A — thin matmul (M=1, K=256, N=1) NOT in op_db's matmul samples.
// ---------------------------------------------------------------------------
//
// op_db only emits k in {0, 10}; with k=256 the cumulative summation order
// effects are 25x larger than at k=10. The build is `cblas_sgemm` so the
// SSE2-reproducible branch (CBWR=COMPATIBLE, =3) should be selected and
// match torch's MKL 2024.2 SSE2 branch byte-for-byte.
//
// Torch reference computed via:
//   python3 -c '
//     import torch; torch.manual_seed(7)
//     a = torch.randn(1, 256, dtype=torch.float32)
//     b = torch.randn(256, 1, dtype=torch.float32)
//     c = torch.matmul(a, b)
//     # c = -3.560312032699585; bits = 0xc063dc27
//   '
//
// Inputs are pinned as raw IEEE-754 bit patterns so the test does not
// depend on the host's `torch.randn` PRNG (input bytes are the same
// regardless of where the test runs).

#[test]
#[ignore = "divergence probe: byte-exact thin-matmul (1,256)@(256,1) vs torch; tracking #1538 (mm_raw_at MKL transa=Trans 1-ULP drift vs torch; #1348 closed prematurely)"]
fn divergence_mkl_thin_1x256_byte_exact() {
    let a_bits: &[u32] = &[
        0xbf51f456, 0x3eca902b, 0x3f661ede, 0xbfb1b738, 0xbe2b0101, 0x3e91ff2d, 0xbf241e93,
        0xbf64c69a, 0x3f6d31ec, 0xbf091754, 0xbf9471b7, 0xbeeb99b8, 0x3f3562d1, 0x3f81a1f8,
        0x3e6bed28, 0x3f8b8a88, 0xbfca94c1, 0xbea62d99, 0x3ff6933b, 0xbea8f74c, 0x3e4b352e,
        0x3f4835ea, 0x3f85017b, 0xbf397989, 0xbe565d56, 0xbe5c827d, 0xbfe869cc, 0xbeb0c38e,
        0xc003ef41, 0x3f2c91dd, 0xbfa96365, 0xbfae0ce9, 0xbdab10d9, 0xbcc054cb, 0x3e328f74,
        0x4013180c, 0x3f7504a6, 0xbf297027, 0xbf54156f, 0xbf1b0ddd, 0xbfb35c30, 0x3fa60f1a,
        0x3fd20a1a, 0xbf87416d, 0xbe85eeb2, 0xbe801162, 0x3f00497f, 0x3e852377, 0xbe367820,
        0xbe84dd09, 0xbc6d5ed4, 0xbec48d5c, 0xc03dd5ba, 0xbf87c044, 0xbe9e34c3, 0x3f6f2d83,
        0x3fcfe99a, 0x3acd6ce1, 0xbee00556, 0xc006f236, 0x3f928f9b, 0xbec3adaa, 0xbeb5e56e,
        0x3f4112c3, 0x3e087004, 0x3e3aed2f, 0xbf03bef2, 0x3f4ceecd, 0xbf9f20d7, 0xbfdbdf4b,
        0xbf15176c, 0xbf1d873b, 0xbf62c39b, 0x3f9937e6, 0x3eef5730, 0xbe1364ff, 0xbf4fdf88,
        0xbec5ed86, 0xbf84bbf3, 0x3f178f43, 0xc0107f81, 0x3ecf0034, 0x3f127ba2, 0x3e9d99b4,
        0xbe00eb11, 0xbf753138, 0x3fe03bd0, 0x3f7ac4d2, 0xbecc02cc, 0xbe3152da, 0x3f41c2f3,
        0x3f7c7953, 0xbf53491d, 0x3e27348b, 0x3f005347, 0x3fb5d49d, 0xbf096d32, 0x3f5431a7,
        0x3f874e10, 0xbf8d669b, 0x3e02808f, 0xbc9b2e3c, 0xbed085cd, 0x3f081d43, 0x3ec46c21,
        0x3ff4cf0f, 0xbf9b74db, 0x3fe2b4a6, 0x3f01cfc3, 0xbe215e54, 0x3eebae9d, 0xbf939c53,
        0xbf8ff62d, 0xbea0fe40, 0x3f203f0b, 0x3db9f8b1, 0xbf160141, 0xbefaab1f, 0x3f23c9c7,
        0x3f1c0276, 0xbf50e37b, 0x3f9406a4, 0xbf9af750, 0x3f01d452, 0xbedcb45f, 0x3f1cfe62,
        0x3e90e8c5, 0x3f97bce5, 0x3f11a332, 0x3e088138, 0x3f0f5cbf, 0xbf3c3240, 0xbe655d5e,
        0xbf4ba3dd, 0xc0089250, 0xbf3fd027, 0x3fba0d72, 0xbfc54d4f, 0xbea9b50f, 0x3f0ceb2a,
        0x3e01cd70, 0xbfc1f31c, 0xbf69a665, 0x3fa2c3fa, 0x3e85e128, 0xbe1901f6, 0x3f938014,
        0xbf90c07b, 0x40059928, 0xbf07dc62, 0x3f4de79e, 0x3d975667, 0x3d8a85f6, 0xbf0c854c,
        0xbed2f1cb, 0x3d5cf65f, 0x3f3936f0, 0xbf8ab4f7, 0xbf07bd43, 0x3e07c336, 0x3f2ea0fc,
        0xbf836f60, 0xbda1e2d2, 0x3e13463d, 0x3fb38663, 0xbed55944, 0x3e296cb8, 0xbf3737a3,
        0xbf2c67ad, 0xbf36640f, 0xbf30e758, 0x3e6cb1bc, 0xbf416d7b, 0xbdf551f2, 0x3f74586b,
        0x3e18c583, 0xbef0c865, 0xbf5924bd, 0x3f51894b, 0x3e5306c1, 0x3f337b8a, 0xbf30e284,
        0x3e95997e, 0x3fbb0ae1, 0x3e9d53bc, 0x3f4262fe, 0x3fc44410, 0xbf46841f, 0xc009aa7a,
        0xbfa3cb7e, 0xbf7d38a9, 0xbb8fa15a, 0xbeb710cf, 0x3e3eda5c, 0xbf952b7a, 0x3f628809,
        0x3fa46509, 0xbf563909, 0xbfb6a6bd, 0xbe8952b4, 0xc02898d2, 0xbcbd4ffc, 0x3f3d1efb,
        0x3f299c4e, 0x3f40ad1d, 0x3f1366d4, 0x3d8afb84, 0xbdc8beca, 0x3f1c3b19, 0xbf67e479,
        0xbf617790, 0xbf5c552e, 0xbf2ca3e4, 0xc0415171, 0x3ecd87a5, 0xbf87b86f, 0xbf90eda4,
        0xbfeac7c4, 0xbf06c6e3, 0x3f92ec56, 0xbe8ff6e7, 0xbf30a99c, 0x3fbd9118, 0xbd435fc3,
        0x3ef14296, 0x3f25dc2f, 0x3faa7221, 0xbe9715be, 0xbc6afe91, 0xbf13493c, 0x3ecc622c,
        0xbeffde41, 0xbfa4c47c, 0xbe42b84f, 0xbda73fb2, 0xbf5a1ed0, 0xbed66056, 0x3eaee13d,
        0xbe778920, 0x3fa510c2, 0xbf377fc1, 0x3f5cb05c, 0xbedf37df, 0xbef7b472, 0xbf20bb67,
        0x40008994, 0x3f412153, 0x3d82bc8a, 0x3f2039f8, 0xbe860ee3, 0xbed33267, 0x3dc762f8,
        0x3f49be1a, 0xbf4f3d69, 0x3dbeb7e6, 0xbe99d22a,
    ];
    let b_bits: &[u32] = &[
        0x3bd00db9, 0xbfc0d398, 0xbf1fddd7, 0x3ee7ed31, 0x3f4fa786, 0x4031b419, 0x3f437768,
        0xbe379a9a, 0xbfb5ea2c, 0xbf858bfa, 0xbe929825, 0x3f489606, 0x3e2731a7, 0xbf57ef0d,
        0x3f3fa4bb, 0x3df997e4, 0xbe1849cf, 0x3fbc3b53, 0x3e4d33d1, 0xbf318f38, 0x3f199e2c,
        0xbf413a94, 0x3fd00802, 0x3fe64868, 0xbf81c10a, 0xbf751ecd, 0x3eb3c94f, 0x3c0dc5cf,
        0x3ed1da42, 0x3b981f17, 0x3f4c5655, 0x3e562431, 0xbeda6e27, 0x3f4cad1c, 0xbe03d14f,
        0x3ecb6d07, 0x3edc96ed, 0xbf84ab09, 0xbf1de789, 0x3fbba903, 0x3fe4b0df, 0xbdf4aaba,
        0x3f8d2c03, 0xbf57fb08, 0xbe393cc5, 0xbeb2b66d, 0x3f935e34, 0xbfc83063, 0x3f25769b,
        0xbf2aa8b7, 0x3ea5834d, 0x3edf1893, 0x3f67c23e, 0x3ebfd153, 0x3f8d75a6, 0x3dba0422,
        0x3f2cfd60, 0xbef51daf, 0xbf53aadb, 0xbd75c718, 0x3f1d0625, 0xbe70f935, 0xbfc1d0a7,
        0x3e2e4fc6, 0xbfc882ad, 0x3e87e1e3, 0xbce15125, 0x3c0d686e, 0x3f24216c, 0x3f1e7f84,
        0xbf4ad259, 0xbf3c67a5, 0x3dd936ac, 0x3f67d555, 0xbfb7e9dd, 0xbeafee19, 0x3e854b6e,
        0x3e142b84, 0x3f225e89, 0xbf611add, 0x3ea116fa, 0xbf8275a7, 0x3ea57616, 0xbc8b4d46,
        0x3f977140, 0x3fc6e129, 0xbe2c2773, 0xbf36fb31, 0x3ec971a8, 0xbed3eaab, 0xbf9761d6,
        0xbf002f06, 0x40160f72, 0xbf9f4b19, 0x4003d976, 0x3edffc02, 0x3f695d0f, 0xbfd00e41,
        0x3e0a9603, 0xbf220e57, 0xbebc1bc9, 0xbfc199ef, 0x3f7975f8, 0x3f0a32ce, 0xbe7ce79d,
        0xbe456ca9, 0x3e648248, 0xbc3549f3, 0xbf81747f, 0x3f9915f2, 0x3d9474e8, 0xbfe04162,
        0x3e9a87ea, 0x3f0dddd9, 0x3fd659ef, 0x3f5cb9fb, 0xbfa39e36, 0xbe4d6574, 0xbfc5f501,
        0x3c2ca855, 0x3f41930e, 0xbe0bbd9e, 0xbdad85cf, 0xbf8bd7d7, 0x3fc4a860, 0x3f7d3116,
        0xbf75b004, 0xbf7ad6bc, 0x3f004818, 0xbb925ab5, 0x3f80c814, 0xbfdd261e, 0xbfbee912,
        0xbf17db66, 0xbf8d676a, 0x3e652d6e, 0x3f97e51f, 0x3fc98c7c, 0xbf07a48d, 0x3e8d4f05,
        0x3f2b61e8, 0x3fd31f6b, 0x3e161aba, 0xbebc47f4, 0xbf965b47, 0x3e15045f, 0xbfa37aca,
        0x3deb70c6, 0x3f865f7d, 0x40628085, 0xbe25b4cc, 0x3f9cd061, 0xbdaa4086, 0xbfceed3c,
        0xbf9a6b4d, 0x3de98a8c, 0x3f88f674, 0xbfdff18b, 0x3f3c294f, 0x3efad80d, 0xbdbcd756,
        0x3d345bc4, 0x3fadc273, 0x3fa008c5, 0x3ed777c7, 0x3f76a3d6, 0x3f8dc664, 0x3f3abc1d,
        0xbe9f45c6, 0x3eba5be6, 0xbe7eb8ea, 0xbe865180, 0x3f0abc9f, 0xbe669e42, 0x3f34ebe5,
        0x3fa9d1f5, 0x3da66ba5, 0xbef8b48d, 0x4008626c, 0x3ecb6993, 0xbfcf4217, 0x3f64146f,
        0xbf0827cc, 0x3f66135d, 0xbfaca943, 0x3fadd7e9, 0x3f8ed4fd, 0x3f8e7387, 0xbf80f082,
        0xbe061804, 0xbdbf199f, 0x3f207c60, 0xbed27fca, 0xbfc072c6, 0x3f25f4be, 0x3f8839e2,
        0x3f845405, 0x3fbfe4ae, 0x3fdfbcdd, 0xbff20b4a, 0x3f30de26, 0xbf3c86fe, 0x3f25ab65,
        0xbca4df0f, 0x3fdf412f, 0xbe7083b4, 0xbef31787, 0xbfc51c68, 0x3ea853c4, 0xbf309890,
        0xbf24d674, 0xbe1882fe, 0x3fdecb60, 0xbf81817a, 0x3eed79a8, 0xbe9d3c64, 0x3e37891f,
        0xbf6b9a1f, 0x3f07d2b5, 0xbfb9533c, 0xbe1da7c5, 0xbfb497f0, 0x3fb28dab, 0x3ec91791,
        0xbfd43e6f, 0x3f9d10f0, 0xbfe4cddb, 0x3e2daab7, 0xbf8902a9, 0xbf4347f0, 0x40001dbd,
        0x3fa1c151, 0x3f3e2d6d, 0x3e90f055, 0x3f31b0ef, 0xbdeaf56a, 0x3db8e350, 0x3dcc6d40,
        0x3ef6ddaa, 0xbf782a84, 0xbf834da8, 0xbe13951c, 0x3f070d55, 0x3eb5e28b, 0x3f2e77b8,
        0x3eb73731, 0x3f97fb28, 0x3f5b4d0b, 0xbffb13f0, 0xbedcdb55, 0xbe418ffa, 0xbf953db3,
        0x3e050304, 0x3e800025, 0xbf5aaf94, 0x3ee01064,
    ];
    let c_expected_bits: &[u32] = &[0xc063dc27]; // torch = -3.560312032699585

    let a = bits_to_f32(a_bits);
    let b = bits_to_f32(b_bits);
    let c = mm_raw::<f32>(&a, &b, 1, 256, 1);
    assert_byte_exact(&c, c_expected_bits, "mm_raw (1,256)@(256,1) f32");
}

// ---------------------------------------------------------------------------
// Probe B — direct `mm_raw_bt` smoke (A=(3,4), B=(5,4), C=A@B^T = (3,5)).
// ---------------------------------------------------------------------------
//
// The matmul dispatcher never routes through `mm_raw_bt`; only
// `MmBackward::backward` does (`dA = grad_C @ B^T`). The parity sweep is
// forward-only, so under `--features mkl` the transpose-fused MKL path
// is wholly untested by the sweep. Probe it directly with hand-computed
// expected values (small integers, exact in f32, so byte-equality holds
// trivially when the call is correct).
//
// Torch reference:
//   python3 -c '
//     import torch
//     a = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]], dtype=torch.float32)
//     b = torch.tensor([[0.5,-1,2,0.25],[1.5,-2,1,0.75],[2.5,-3,0,1.25],
//                       [3.5,-4,-1,1.75],[4.5,-5,-2,2.25]], dtype=torch.float32)
//     c = a @ b.T
//     # c = [[5.5,3.5,1.5,-0.5,-2.5],
//            [12.5,8.5,4.5,0.5,-3.5],
//            [19.5,13.5,7.5,1.5,-4.5]]
//   '

#[test]
#[ignore = "divergence probe: direct mm_raw_bt MKL transpose-fused path; tracking #1538 (mm_raw_at MKL transa=Trans 1-ULP drift vs torch; #1348 closed prematurely)"]
fn divergence_mkl_mm_raw_bt_direct() {
    let a_bits: &[u32] = &[
        0x3f800000, 0x40000000, 0x40400000, 0x40800000, // [1,2,3,4]
        0x40a00000, 0x40c00000, 0x40e00000, 0x41000000, // [5,6,7,8]
        0x41100000, 0x41200000, 0x41300000, 0x41400000, // [9,10,11,12]
    ];
    let b_bits: &[u32] = &[
        0x3f000000, 0xbf800000, 0x40000000, 0x3e800000, // [0.5,-1,2,0.25]
        0x3fc00000, 0xc0000000, 0x3f800000, 0x3f400000, // [1.5,-2,1,0.75]
        0x40200000, 0xc0400000, 0x00000000, 0x3fa00000, // [2.5,-3,0,1.25]
        0x40600000, 0xc0800000, 0xbf800000, 0x3fe00000, // [3.5,-4,-1,1.75]
        0x40900000, 0xc0a00000, 0xc0000000, 0x40100000, // [4.5,-5,-2,2.25]
    ];
    let c_expected_bits: &[u32] = &[
        0x40b00000, 0x40600000, 0x3fc00000, 0xbf000000, 0xc0200000, // 5.5,3.5,1.5,-0.5,-2.5
        0x41480000, 0x41080000, 0x40900000, 0x3f000000, 0xc0600000, // 12.5,8.5,4.5,0.5,-3.5
        0x419c0000, 0x41580000, 0x40f00000, 0x3fc00000, 0xc0900000, // 19.5,13.5,7.5,1.5,-4.5
    ];

    let a = bits_to_f32(a_bits);
    let b = bits_to_f32(b_bits);
    // mm_raw_bt(A=(M,K), B=(N,K)) -> C=(M,N) = A @ B^T
    let c = mm_raw_bt::<f32>(&a, &b, 3, 4, 5);
    assert_byte_exact(&c, c_expected_bits, "mm_raw_bt (3,4)@(5,4)^T f32");
}

// ---------------------------------------------------------------------------
// Probe C — direct `mm_raw_at` smoke (A=(4,3), B=(4,5), C=A^T@B = (3,5)).
// ---------------------------------------------------------------------------
//
// Same coverage rationale as Probe B: the matmul dispatcher never reaches
// `mm_raw_at`; only `MmBackward::backward` does (`dB = A^T @ grad_C`),
// which is not exercised by the forward-only parity sweep.
//
// Torch reference:
//   python3 -c '
//     import torch
//     a = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=torch.float32)
//     b = torch.tensor([[0.5,-1,2,0.25,0.0],[1.5,-2,1,0.75,0.1],
//                       [2.5,-3,0,1.25,0.2],[3.5,-4,-1,1.75,0.3]],
//                      dtype=torch.float32)
//     c = a.T @ b
//   '

// NOT #[ignore]d: this is the release-blocking failing test for #1538.
// Anyone running `cargo test -p ferrotorch-core --features mkl` will see
// this FAIL until the mm_raw_at MKL transa=Trans 1-ULP drift is resolved.
#[test]
fn divergence_mkl_mm_raw_at_direct() {
    let a_bits: &[u32] = &[
        0x3f800000, 0x40000000, 0x40400000, // [1,2,3]
        0x40800000, 0x40a00000, 0x40c00000, // [4,5,6]
        0x40e00000, 0x41000000, 0x41100000, // [7,8,9]
        0x41200000, 0x41300000, 0x41400000, // [10,11,12]
    ];
    let b_bits: &[u32] = &[
        0x3f000000, 0xbf800000, 0x40000000, 0x3e800000, 0x00000000, // [0.5,-1,2,0.25,0]
        0x3fc00000, 0xc0000000, 0x3f800000, 0x3f400000, 0x3dcccccd, // [1.5,-2,1,0.75,0.1]
        0x40200000, 0xc0400000, 0x00000000, 0x3fa00000, 0x3e4ccccd, // [2.5,-3,0,1.25,0.2]
        0x40600000, 0xc0800000, 0xbf800000, 0x3fe00000, 0x3e99999a, // [3.5,-4,-1,1.75,0.3]
    ];
    let c_expected_bits: &[u32] = &[
        0x426c0000, 0xc28c0000, 0xc0800000, 0x41ec0000, 0x4099999a, // 59,-70,-4,29.5,4.80000019
        0x42860000, 0xc2a00000, 0xc0000000, 0x42060000, 0x40accccd, // 67,-80,-2,33.5,5.40000010
        0x42960000, 0xc2b40000, 0x00000000, 0x42160000, 0x40c00001, // 75,-90, 0,37.5,6.00000048
    ];

    let a = bits_to_f32(a_bits);
    let b = bits_to_f32(b_bits);
    // mm_raw_at(A=(K,M), B=(K,N)) -> C=(M,N) = A^T @ B
    let c = mm_raw_at::<f32>(&a, &b, 3, 4, 5);
    assert_byte_exact(&c, c_expected_bits, "mm_raw_at (4,3)^T@(4,5) f32");
}

// ---------------------------------------------------------------------------
// Probe D — verify CBWR=COMPATIBLE is actually engaged at runtime.
// ---------------------------------------------------------------------------
//
// The commit relies on a `.init_array` POSIX `setenv("MKL_CBWR",
// "COMPATIBLE", 1)` constructor running BEFORE MKL's own static
// constructors initialise the dispatch table. If MKL's ctors run first
// (link-order-dependent — `.init_array` does not guarantee ordering
// between TUs from different static archives), the env-var trick is a
// no-op and MKL selects its default AVX2 / AVX-512 branch instead of
// the SSE2 (=3) reproducible branch.
//
// MKL exposes `MKL_CBWR_Get(int input)` (Intel docs:
// https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/support-functions/cbwr/support-functions-for-cbwr/mkl-cbwr-get.html);
// passing `MKL_CBWR_BRANCH = 1` returns the currently selected branch.
// We touch `mm_raw` first to ensure MKL has dispatched at least once
// (so `Get` returns the locked-in value, not the configured-but-not-yet-
// applied value), then assert the branch equals 3 (`MKL_CBWR_COMPATIBLE`).
//
// If this probe FAILS with e.g. 5 (`MKL_CBWR_AVX2`) or 7 (`MKL_CBWR_AVX512`),
// the byte-exact-parity claim is contingent on MKL 2020.1 (vendored) and
// MKL 2024.2 (torch's) happening to produce identical sgemm output on
// that specific dispatch — which is FALSE per the commit's own
// architectural notes: "without CBWR=COMPATIBLE these versions diverge
// by ~3e-6 on f32 dot products with k>=10". So a non-3 result here
// invalidates the cross-version byte-exact claim.
//
// MKL CBWR branch constants (per Intel header `mkl_service.h`):
//   MKL_CBWR_BRANCH        = 1   (input mode)
//   MKL_CBWR_OFF           = 0
//   MKL_CBWR_AUTO          = 2
//   MKL_CBWR_COMPATIBLE    = 3
//   MKL_CBWR_SSE2          = 4
//   MKL_CBWR_SSE4_2        = 8
//   MKL_CBWR_AVX           = 9
//   MKL_CBWR_AVX2          = 10
//   MKL_CBWR_AVX512_MIC    = 11
//   MKL_CBWR_AVX512        = 12

#[allow(non_snake_case)]
unsafe extern "C" {
    fn MKL_CBWR_Get(input: i32) -> i32;
}

const MKL_CBWR_BRANCH: i32 = 1;
const MKL_CBWR_COMPATIBLE_VAL: i32 = 3;

#[test]
#[ignore = "divergence probe: assert .init_array MKL_CBWR=COMPATIBLE engaged; tracking #1538 (mm_raw_at MKL transa=Trans 1-ULP drift vs torch; #1348 closed prematurely)"]
fn divergence_mkl_cbwr_branch_is_compatible() {
    // Force MKL to dispatch its first GEMM so the branch is locked in
    // and `MKL_CBWR_Get` returns the actual engaged branch (not just the
    // configured-but-not-applied value).
    let a = vec![1.0f32; 16];
    let b = vec![1.0f32; 16];
    let _c = mm_raw::<f32>(&a, &b, 4, 4, 4);

    // SAFETY: leaf FFI shim to MKL's documented `MKL_CBWR_Get(int)` —
    // takes one i32, returns i32, no pointers, no aliasing.
    let branch = unsafe { MKL_CBWR_Get(MKL_CBWR_BRANCH) };

    // Branch decoder for the panic message — readable for the
    // failure-mode log.
    let branch_name = match branch {
        0 => "MKL_CBWR_OFF",
        2 => "MKL_CBWR_AUTO",
        3 => "MKL_CBWR_COMPATIBLE",
        4 => "MKL_CBWR_SSE2",
        8 => "MKL_CBWR_SSE4_2",
        9 => "MKL_CBWR_AVX",
        10 => "MKL_CBWR_AVX2",
        11 => "MKL_CBWR_AVX512_MIC",
        12 => "MKL_CBWR_AVX512",
        n if n < 0 => "ERROR (negative)",
        _ => "UNKNOWN",
    };
    assert_eq!(
        branch, MKL_CBWR_COMPATIBLE_VAL,
        ".init_array MKL_CBWR=COMPATIBLE NOT engaged: branch={branch} ({branch_name}); \
         the byte-exact parity claim depends on CBWR=COMPATIBLE so MKL 2020.1 \
         (vendored by intel-mkl-src 0.8) matches torch's MKL 2024.2 sgemm bit-for-bit. \
         If branch != 3, parity is incidental to op_db's small samples and will \
         drift on shapes/values not in the sweep's enumeration."
    );
}

// ---------------------------------------------------------------------------
// Probe E — tolerance regression: confirm MKL_ENABLED is read by runner.
// ---------------------------------------------------------------------------
//
// Sanity-check that the compile-time const `MKL_ENABLED` actually
// surfaces `true` under `--features mkl`. If the parity-sweep runner's
// `tolerance_for` reads this as false, the matmul-family envelope stays
// at the loose `rtol=1e-4` and the "byte-exact parity" claim is
// silently downgraded to the same widened envelope as the faer path.
//
// This is a metadata-check; it shouldn't fail under the committed build,
// but documents the contract the runner relies on so a future build-
// system change that breaks the cfg propagation FAILS this test rather
// than silently widening the envelope.

#[test]
fn divergence_mkl_enabled_const_true_under_feature() {
    assert!(
        ferrotorch_core::ops::linalg::MKL_ENABLED,
        "MKL_ENABLED const is false under --features mkl; the parity-sweep \
         runner's tolerance_for will fall through to the faer-widened \
         rtol=1e-4 envelope, silently downgrading 'byte-exact parity' to \
         the same loose envelope as the no-mkl build (closes #1348 claim invalidated)"
    );
}
