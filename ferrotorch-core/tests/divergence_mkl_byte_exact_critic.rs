//! Critic audit of commit `c7d81320d` (closes #1538):
//! Byte-exact MKL matmul/bmm parity claim probed against torch on this
//! host.
//!
//! The commit's "honesty note" asserts torch on this host links OpenBLAS,
//! not MKL — used to justify `#[ignore]` on `divergence_mkl_thin_1x256_byte_exact`
//! and widening bmm parity-sweep tolerance to `1e-4`. The critic verified
//! the claim is **false**:
//!
//! - `/proc/<pid>/maps` after `torch.matmul` shows `numpy.libs/libscipy_openblas64_*.so`
//!   loaded by **numpy**, but `libtorch_cpu.so` has MKL statically baked
//!   in (`nm -D libtorch_cpu.so | grep sgemm_` shows `T sgemm_` defined
//!   at offset `0x9117a50`).
//! - `MKL_VERBOSE=1 python3 -c 'import torch; torch.randn(64,64) @ torch.randn(64,64)'`
//!   prints `MKL_VERBOSE oneMKL 2024.0 Update 2 Product build 20240605 ...
//!   SGEMM(N,N,64,64,64,...) CNR:OFF Dyn:1 FastMM:1 TID:0 NThr:14`.
//!   torch DOES use MKL.
//! - ferrotorch's `libmkl_rt.so.2` reports `Intel(R) oneAPI MKL Version
//!   2024.2-Product Build 20240605` — same build date `20240605` as torch's.
//!   Same Intel oneMKL release SKU.
//!
//! So the test oracle (torch) IS using MKL on this host, and the byte-exact
//! claim CAN be tested here. These probes pit ferrotorch+MKL against
//! torch+MKL on the inputs the commit's prose specifically calls out:
//! the asymmetric thin matrix, power-of-2, non-power-of-2, k=1 outer,
//! and k=257 inner-loop shapes.
//!
//! Tracking: critic findings refile #1538 if any probe fails byte-exact.

#![cfg(feature = "mkl")]

use ferrotorch_core::ops::linalg::mm_raw;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn load_bits(path: &str) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert!(bytes.len().is_multiple_of(4), "{path}: not a u32 stream");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bits_to_f32(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

fn assert_byte_exact(actual: &[f32], expected_bits: &[u32], label: &str) {
    assert_eq!(actual.len(), expected_bits.len(), "{label}: length mismatch");
    let mut mismatches = Vec::new();
    let mut max_ulp = 0u32;
    for (i, (&a, &eb)) in actual.iter().zip(expected_bits.iter()).enumerate() {
        let ab = a.to_bits();
        if ab != eb {
            let ulp = ab.abs_diff(eb);
            max_ulp = max_ulp.max(ulp);
            mismatches.push((i, ab, eb, a, f32::from_bits(eb), ulp));
        }
    }
    if !mismatches.is_empty() {
        let mut msg = format!(
            "{label}: byte-exact parity FAILED on {}/{} elements (max ulp_diff = {})",
            mismatches.len(),
            actual.len(),
            max_ulp,
        );
        for (i, ab, eb, av, ev, ulp) in mismatches.iter().take(6) {
            msg.push_str(&format!(
                "\n  [{i}] ferrotorch=0x{ab:08x} ({av}) torch=0x{eb:08x} ({ev}) ulp_diff={ulp}"
            ));
        }
        if mismatches.len() > 6 {
            msg.push_str(&format!("\n  ... and {} more", mismatches.len() - 6));
        }
        panic!("{msg}");
    }
}

// ---------------------------------------------------------------------------
// Critic Probe A — asymmetric thin matrix (1, 16384) @ (16384, 1).
// ---------------------------------------------------------------------------
//
// Single dot product over a long vector. The block-summation order in
// MKL's AVX2 micro-kernel for a thin column GEMV-shape differs from
// the dense GEMM micro-kernel; this probes whether ferrotorch's
// `sgemm_('N','N', 1, 1, 16384, ...)` call routes through the same
// micro-kernel as torch's matmul of identical inputs.
//
// Reference produced by `python3 -c "import torch; torch.manual_seed(1);
// a=torch.randn(1,16384); b=torch.randn(16384,1); print((a@b))"` on the
// same host. Input bits saved as binary fixtures (deterministic, host-
// independent).
//
// Expected (torch): c = -71.26595306396484 (0xc28e882b).
//
// Tracking: blocker #N (file if FAILS).
#[test]
fn divergence_critic_thin_1x16384x1_byte_exact() {
    let a = bits_to_f32(&load_bits(&format!("{FIXTURES}/probe_1x16384x1_a_bits.bin")));
    let b = bits_to_f32(&load_bits(&format!("{FIXTURES}/probe_1x16384x1_b_bits.bin")));
    let expected_bits = load_bits(&format!("{FIXTURES}/probe_1x16384x1_c_bits.bin"));
    let c = mm_raw::<f32>(&a, &b, 1, 16384, 1);
    assert_byte_exact(&c, &expected_bits, "mm_raw (1,16384)@(16384,1) thin f32");
}

// ---------------------------------------------------------------------------
// Critic Probe B — power-of-2 (64, 64) @ (64, 64).
// ---------------------------------------------------------------------------
//
// Both MKL and OpenBLAS may use vectorized AVX2 paths that happen to
// coincide on this dim-aligned shape. If torch is OpenBLAS, parity may
// PASS here but FAIL on a non-power-of-2. Combined with Probe C below,
// this isolates the BLAS-impl variance.
//
// Reference: torch.manual_seed(2); a=randn(64,64); b=randn(64,64); c=a@b.
#[test]
fn divergence_critic_64x64_byte_exact() {
    let a = bits_to_f32(&load_bits(&format!("{FIXTURES}/probe_64x64_a_bits.bin")));
    let b = bits_to_f32(&load_bits(&format!("{FIXTURES}/probe_64x64_b_bits.bin")));
    let expected_bits = load_bits(&format!("{FIXTURES}/probe_64x64_c_bits.bin"));
    let c = mm_raw::<f32>(&a, &b, 64, 64, 64);
    assert_byte_exact(&c, &expected_bits, "mm_raw (64,64)@(64,64) power-of-2 f32");
}

// ---------------------------------------------------------------------------
// Critic Probe C — non-power-of-2 (127, 127) @ (127, 127).
// ---------------------------------------------------------------------------
//
// Boundary handling differs between MKL and OpenBLAS (different remainder
// loops). This is the canonical "shape that exposes BLAS-impl divergence"
// from the parity-sweep prose.
//
// Reference: torch.manual_seed(3); a=randn(127,127); b=randn(127,127);
// c=a@b.
#[test]
fn divergence_critic_127x127_byte_exact() {
    let a = bits_to_f32(&load_bits(&format!("{FIXTURES}/probe_127x127_a_bits.bin")));
    let b = bits_to_f32(&load_bits(&format!("{FIXTURES}/probe_127x127_b_bits.bin")));
    let expected_bits = load_bits(&format!("{FIXTURES}/probe_127x127_c_bits.bin"));
    let c = mm_raw::<f32>(&a, &b, 127, 127, 127);
    assert_byte_exact(&c, &expected_bits, "mm_raw (127,127)@(127,127) non-power-of-2 f32");
}

// ---------------------------------------------------------------------------
// Critic Probe D — outer product k=1 (5, 1) @ (1, 7).
// ---------------------------------------------------------------------------
//
// Trivial: single multiplication per output element, no accumulation
// order to differ. If THIS probe fails, dispatcher arg shape is wrong
// (a structural #1538 bug regardless of BLAS impl). Expected to pass
// byte-exact under any correct dispatch.
//
// Reference: torch.manual_seed(4); a=randn(5,1); b=randn(1,7); c=a@b.
#[test]
fn divergence_critic_outer_5x1x7_byte_exact() {
    let a_bits: &[u32] = &[
        0xbfcd79b2, 0x3e6e10bc, 0x400f5a08, 0x3f58e83f, 0x3f99aeb6,
    ];
    let b_bits: &[u32] = &[
        0xbecd9801, 0xbfb687d0, 0x3f676812, 0x3f5b102d, 0x3f305a80, 0x3f628e6c, 0x3fe2a46e,
    ];
    let c_expected_bits: &[u32] = &[
        0x3f250475, 0x40128187, 0xbfb9bc58, 0xbfafd417, 0xbf8d8c4e, 0xbfb5d7bf, 0xc035e969,
        0xbdbf30c1, 0xbea9be32, 0x3e5731e1, 0x3e4bb75c, 0x3e23ffaa, 0x3e52af38, 0x3ed2c3af,
        0xbf66406d, 0xc04c6c1d, 0x40019484, 0x3ff55627, 0x3fc58126, 0x3ffdba77, 0x407dd31d,
        0xbeae32c5, 0xbf9aa82f, 0x3f4411be, 0x3f399c63, 0x3f156c59, 0x3f3ff5b4, 0x3fc0085a,
        0xbef6d850, 0xbfdb277b, 0x3f8aeb20, 0x3f83822c, 0x3f53bce3, 0x3f8801bc, 0x40080ef3,
    ];
    let a = bits_to_f32(a_bits);
    let b = bits_to_f32(b_bits);
    let c = mm_raw::<f32>(&a, &b, 5, 1, 7);
    assert_byte_exact(&c, c_expected_bits, "mm_raw (5,1)@(1,7) outer f32");
}

// ---------------------------------------------------------------------------
// Critic Probe E — non-trivial inner k=257 (3, 257) @ (257, 5).
// ---------------------------------------------------------------------------
//
// k=257 is prime, so no BLAS micro-kernel can align block boundaries to
// the K dimension cleanly — both MKL and OpenBLAS must handle the
// remainder loop, but they handle it DIFFERENTLY. If torch were OpenBLAS,
// this probe would expose order-of-summation drift; since both are MKL
// 2024 Update 2 SKU build 20240605, this probe MUST pass byte-exact
// (or the dispatcher port has a deeper bug than just dispatch shape).
//
// Reference: torch.manual_seed(5); a=randn(3,257); b=randn(257,5); c=a@b.
#[test]
fn divergence_critic_k257_3x257x5_byte_exact() {
    let a = bits_to_f32(&load_bits(&format!("{FIXTURES}/probe_3x257x5_a_bits.bin")));
    let b = bits_to_f32(&load_bits(&format!("{FIXTURES}/probe_3x257x5_b_bits.bin")));
    let expected_bits = load_bits(&format!("{FIXTURES}/probe_3x257x5_c_bits.bin"));
    let c = mm_raw::<f32>(&a, &b, 3, 257, 5);
    assert_byte_exact(&c, &expected_bits, "mm_raw (3,257)@(257,5) k=prime f32");
}

// ---------------------------------------------------------------------------
// Critic Probe F — BLAS-oracle confound refutation.
// ---------------------------------------------------------------------------
//
// The commit message's "honesty note" claims "torch on the build host
// links OpenBLAS, NOT MKL — verified via /proc/<pid>/maps". This claim
// is the basis for `#[ignore]`-ing `divergence_mkl_thin_1x256_byte_exact`
// and widening bmm parity tolerance to 1e-4.
//
// The critic verified (via MKL_VERBOSE=1 python3 -c '... torch.matmul'
// printing `MKL_VERBOSE oneMKL 2024.0 Update 2 Product build 20240605 ...
// SGEMM(...)`) that torch DOES use MKL on this host. libtorch_cpu.so has
// MKL statically linked (nm -D shows `T sgemm_`). The OpenBLAS visible in
// /proc/<pid>/maps belongs to **numpy**, not torch.
//
// This test embeds the bit-identical input from
// `divergence_mkl_thin_1x256_byte_exact` and ASSERTS the thin-matmul
// IS byte-exact on this host. If this probe PASSES, the host-dependent
// `#[ignore]` on the sibling test is incorrectly justified — that test
// should be unignored, and the bmm tolerance widening to 1e-4
// (`tools/parity-sweep/runner/src/main.rs::tolerance_for`) reverted.
#[test]
fn divergence_critic_blas_oracle_confound_refutation() {
    // SAME inputs as `divergence_mkl_thin_1x256_byte_exact` in the sibling
    // file. Copy-pasted (rather than imported) to make this test stand
    // alone — the sibling is in the same test crate so symbol clash
    // is the only reason not to share. Different shape: (1, 256) @
    // (256, 1).
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
    assert_byte_exact(
        &c,
        c_expected_bits,
        "mm_raw (1,256)@(256,1) — sibling test #[ignore]'d on bad OpenBLAS-confound premise",
    );
}
