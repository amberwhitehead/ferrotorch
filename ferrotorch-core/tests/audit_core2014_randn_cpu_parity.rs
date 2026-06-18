//! CORE-2014: CPU `randn` must mirror PyTorch's scalar/normal_fill split.
//!
//! Fixtures below were captured from the local PyTorch 2.11.0+cu130 CPU build
//! on 2026-06-18 with `torch.backends.cpu.get_cpu_capability() == "AVX2"`.
//! They pin exact IEEE bit patterns, not tolerances.

use ferrotorch_core::{Generator, manual_seed, rand, randn};
use half::{bf16, f16};
use std::sync::{Mutex, MutexGuard};

fn default_rng_test_lock() -> MutexGuard<'static, ()> {
    static TEST_LOCK: Mutex<()> = Mutex::new(());
    TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn assert_f32_bits(label: &str, got: &[f32], expected: &[u32]) {
    assert!(
        got.len() >= expected.len(),
        "{label}: got {} values, expected at least {}",
        got.len(),
        expected.len()
    );
    for (i, (&value, &bits)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            value.to_bits(),
            bits,
            "{label}[{i}]: got 0x{:08x} ({value:.9}), expected 0x{bits:08x}",
            value.to_bits()
        );
    }
}

fn assert_f64_bits(label: &str, got: &[f64], expected: &[u64]) {
    assert!(
        got.len() >= expected.len(),
        "{label}: got {} values, expected at least {}",
        got.len(),
        expected.len()
    );
    for (i, (&value, &bits)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            value.to_bits(),
            bits,
            "{label}[{i}]: got 0x{:016x} ({value:.17}), expected 0x{bits:016x}",
            value.to_bits()
        );
    }
}

fn assert_f16_bits(label: &str, got: &[f16], expected: &[u16]) {
    assert!(
        got.len() >= expected.len(),
        "{label}: got {} values, expected at least {}",
        got.len(),
        expected.len()
    );
    for (i, (&value, &bits)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            value.to_bits(),
            bits,
            "{label}[{i}]: got 0x{:04x} ({:.9}), expected 0x{bits:04x}",
            value.to_bits(),
            value.to_f32()
        );
    }
}

fn assert_bf16_bits(label: &str, got: &[bf16], expected: &[u16]) {
    assert!(
        got.len() >= expected.len(),
        "{label}: got {} values, expected at least {}",
        got.len(),
        expected.len()
    );
    for (i, (&value, &bits)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            value.to_bits(),
            bits,
            "{label}[{i}]: got 0x{:04x} ({:.9}), expected 0x{bits:04x}",
            value.to_bits(),
            value.to_f32()
        );
    }
}

const TORCH_RAND_42_F16_10: [u16; 10] = [
    0x3866, 0x39b3, 0x36b8, 0x390e, 0x386a, 0x3847, 0x3abc, 0x3814, 0x3a66, 0x2b90,
];

const TORCH_RAND_42_BF16_10: [u16; 10] = [
    0x3ecc, 0x3f33, 0x3eb8, 0x3d60, 0x3ed4, 0x3e8e, 0x3f3c, 0x3da0, 0x3ecc, 0x3ef2,
];

const TORCH_RANDN_42_F32_10: [u32; 10] = [
    0x3eac_62ae,
    0x3e03_e69d,
    0x3e70_16e7,
    0x3e6b_dc6c,
    0xbf8f_b9c2,
    0xbe3e_ccd8,
    0x400d_532c,
    0xbf23_53c6,
    0x3eec_5e56,
    0x3e88_e237,
];

const TORCH_RANDN_42_F32_16: [u32; 16] = [
    0x3ff6_a529,
    0x3fbe_5f53,
    0x3f66_9567,
    0xc006_c0db,
    0x3f2d_acd5,
    0xbf9e_0591,
    0xbd30_6786,
    0xbfcd_65ba,
    0xbf40_8bf0,
    0x3fd3_095b,
    0xbec8_f2f6,
    0xbfb3_a966,
    0xbf3a_566e,
    0xbf0f_36d1,
    0xbf44_d2a0,
    0x3f43_2f9f,
];

const TORCH_RANDN_42_F32_17: [u32; 17] = [
    0x3ff6_a529,
    0xbe23_6d85,
    0xbefe_aae4,
    0x3ee1_11d8,
    0xbf42_14e2,
    0x3f8a_0650,
    0x3f4d_0144,
    0x3fd7_1e93,
    0x3eb6_3345,
    0x3fa5_f12f,
    0x3f1c_4788,
    0x3faa_d8b0,
    0xbe6d_2eed,
    0x3d2b_0c00,
    0xbe80_ce7a,
    0x3f5c_1fb0,
    0xbe9e_9482,
];

const TORCH_RANDN_42_F32_33_PREFIX: [u32; 20] = [
    0x3ff6_a529,
    0x3fbe_5f53,
    0x3f66_9567,
    0xc006_c0db,
    0x3f2d_acd5,
    0xbf9e_0591,
    0xbd30_6786,
    0xbfcd_65ba,
    0xbf40_8bf0,
    0x3fd3_095b,
    0xbec8_f2f6,
    0xbfb3_a966,
    0xbf3a_566e,
    0xbf0f_36d1,
    0xbf44_d2a0,
    0x3f43_2f9f,
    0x3fd2_3771,
    0xbf5f_0955,
    0xbe64_ba09,
    0x3fdb_d280,
];

const TORCH_RANDN_42_F64_10: [u64; 10] = [
    0x3fd5_8c55_b33d_44cc,
    0x3fc0_7cd3_9b3d_4b85,
    0x3fce_02dc_db63_ec41,
    0x3fcd_7b8d_7b0a_7c22,
    0xbff1_f738_3f83_20c1,
    0xbfc7_d99b_0fe3_842e,
    0x4001_aa65_763a_b83b,
    0xbfe4_6a78_cdb8_4d7b,
    0x3fdd_8bca_bb37_ba27,
    0x3fd1_1c46_d07e_388e,
];

const TORCH_RANDN_42_F64_16: [u64; 16] = [
    0x3fd3_2bee_22cc_b503,
    0x3fcf_063c_4666_8b78,
    0x3fd2_2011_e840_abbd,
    0xbfcd_d10c_e0b8_c5a4,
    0x3fe5_7a16_51f4_080f,
    0x3fe9_043d_0330_1e76,
    0xbffc_a51a_01b5_635e,
    0xbffb_f975_374a_1832,
    0x3fc6_3479_875e_ff86,
    0x3fd1_13f5_4563_0654,
    0x3fdb_6cd4_189e_d584,
    0x3fcd_adec_f5c7_9cd4,
    0xbff0_57e3_7304_749c,
    0xbfea_78f0_bc58_5a1c,
    0x3ffb_bf53_77e0_5516,
    0xbff7_df87_a738_4c38,
];

const TORCH_RANDN_42_F64_17: [u64; 17] = [
    0x3fd3_2bee_22cc_b503,
    0x3ff8_12ba_60e8_23bd,
    0x3ff4_586f_c503_25b4,
    0x3fe3_cc25_03d8_2af8,
    0xbfa3_0c45_c714_5abe,
    0x3ff0_905f_784f_7792,
    0xbf84_a7b7_ceff_41c7,
    0x3fb6_e82b_fad7_3187,
    0x3fa3_adc8_96b5_cc6a,
    0x3fd8_f5d1_2bd9_d16a,
    0xbfe2_8e0f_ab04_0e8c,
    0x3fec_0752_9caa_8c80,
    0x3fd9_1cd5_51d1_ca00,
    0x3ff1_bc41_cab3_82c5,
    0x3feb_af80_4309_bfb6,
    0x3feb_5ded_dd31_4d28,
    0x3fcf_ecba_7d41_2db7,
];

const TORCH_RANDN_42_F16_10: [u16; 10] = [
    0x3563, 0x301f, 0x3381, 0x335f, 0xbc7e, 0xb1f6, 0x406b, 0xb91b, 0x3763, 0x3447,
];

const TORCH_RANDN_42_F16_16: [u16; 16] = [
    0x3631, 0x3de2, 0x30b3, 0xbc79, 0x395e, 0xb286, 0x3f69, 0xb7fd, 0xbcd0, 0x3895, 0x3c23, 0xbae8,
    0x3c4c, 0xbce2, 0x380d, 0xbc57,
];

const TORCH_RANDN_42_F16_17: [u16; 17] = [
    0x3631, 0xb503, 0xbdd4, 0x3065, 0xbe32, 0xbdab, 0x3925, 0xb845, 0x3440, 0xbc05, 0x3ca5, 0x355c,
    0xbc1f, 0x34e1, 0x39d0, 0xb492, 0x3e7a,
];

const TORCH_RANDN_42_F16_33_PREFIX: [u16; 20] = [
    0x3631, 0x3de2, 0x30b3, 0xbc79, 0x395e, 0xb286, 0x3f69, 0xb7fd, 0xbcd0, 0x3895, 0x3c23, 0xbae8,
    0x3c4c, 0xbce2, 0x380d, 0xbc57, 0xad28, 0x20b9, 0x2c8d, 0x381a,
];

const TORCH_RANDN_42_BF16_10: [u16; 10] = [
    0x3eac, 0x3e04, 0x3e70, 0x3e6c, 0xbf90, 0xbe3f, 0x400d, 0xbf23, 0x3eec, 0x3e89,
];

const TORCH_RANDN_42_BF16_16: [u16; 16] = [
    0xbf4f, 0xbfc4, 0x3ed0, 0x3e30, 0xbe7d, 0x3e51, 0xbf61, 0xbec6, 0x3f1a, 0x3e89, 0xbf5a, 0xbe94,
    0x3f80, 0xbf48, 0x3fb0, 0x3df3,
];

const TORCH_RANDN_42_BF16_17: [u16; 17] = [
    0xbf4f, 0xbf44, 0x3f53, 0xbf98, 0xbd07, 0xbda4, 0x3da0, 0xbf26, 0x3f11, 0xbf2b, 0x3f87, 0xbd13,
    0xbfa9, 0xbf2c, 0x3d2a, 0xbf25, 0xc00b,
];

const TORCH_RANDN_42_BF16_33_PREFIX: [u16; 20] = [
    0xbf4f, 0xbfc4, 0x3ed0, 0x3e30, 0xbe7d, 0x3e51, 0xbf61, 0xbec6, 0x3f1a, 0x3e89, 0xbf5a, 0xbe94,
    0x3f80, 0xbf48, 0x3fb0, 0x3df3, 0x3f5d, 0x3f35, 0xbe60, 0x4039,
];

#[test]
fn rand_reduced_precision_matches_torch_uniform_real_distribution() {
    let _guard = default_rng_test_lock();
    manual_seed(42).unwrap();
    let t = rand::<f16>(&[10]).unwrap();
    assert_f16_bits("rand_f16_10", t.data().unwrap(), &TORCH_RAND_42_F16_10);

    manual_seed(42).unwrap();
    let t = rand::<bf16>(&[10]).unwrap();
    assert_bf16_bits("rand_bf16_10", t.data().unwrap(), &TORCH_RAND_42_BF16_10);
}

#[test]
fn generator_reduced_precision_uniform_helpers_match_torch_distribution() {
    let mut generator = Generator::new(42);
    let got: Vec<f16> = (0..10).map(|_| generator.next_uniform_f16()).collect();
    assert_f16_bits("generator_rand_f16_10", &got, &TORCH_RAND_42_F16_10);

    let mut generator = Generator::new(42);
    let got: Vec<bf16> = (0..10).map(|_| generator.next_uniform_bf16()).collect();
    assert_bf16_bits("generator_rand_bf16_10", &got, &TORCH_RAND_42_BF16_10);
}

#[test]
fn randn_f32_scalar_path_matches_torch() {
    let _guard = default_rng_test_lock();
    manual_seed(42).unwrap();
    let t = randn::<f32>(&[10]).unwrap();
    assert_f32_bits("randn_f32_10", t.data().unwrap(), &TORCH_RANDN_42_F32_10);
}

#[test]
fn randn_f32_normal_fill_path_matches_torch() {
    let _guard = default_rng_test_lock();
    manual_seed(42).unwrap();
    let t = randn::<f32>(&[16]).unwrap();
    assert_f32_bits("randn_f32_16", t.data().unwrap(), &TORCH_RANDN_42_F32_16);
}

#[test]
fn randn_f32_normal_fill_tail_recompute_matches_torch() {
    let _guard = default_rng_test_lock();
    manual_seed(42).unwrap();
    let t = randn::<f32>(&[17]).unwrap();
    assert_f32_bits("randn_f32_17", t.data().unwrap(), &TORCH_RANDN_42_F32_17);

    manual_seed(42).unwrap();
    let t = randn::<f32>(&[33]).unwrap();
    assert_f32_bits(
        "randn_f32_33_prefix",
        t.data().unwrap(),
        &TORCH_RANDN_42_F32_33_PREFIX,
    );
}

#[test]
fn randn_f64_scalar_and_normal_fill_paths_match_torch() {
    let _guard = default_rng_test_lock();
    manual_seed(42).unwrap();
    let t = randn::<f64>(&[10]).unwrap();
    assert_f64_bits("randn_f64_10", t.data().unwrap(), &TORCH_RANDN_42_F64_10);

    manual_seed(42).unwrap();
    let t = randn::<f64>(&[16]).unwrap();
    assert_f64_bits("randn_f64_16", t.data().unwrap(), &TORCH_RANDN_42_F64_16);

    manual_seed(42).unwrap();
    let t = randn::<f64>(&[17]).unwrap();
    assert_f64_bits("randn_f64_17", t.data().unwrap(), &TORCH_RANDN_42_F64_17);
}

#[test]
fn randn_f16_scalar_and_normal_fill_paths_match_torch() {
    let _guard = default_rng_test_lock();
    manual_seed(42).unwrap();
    let t = randn::<f16>(&[10]).unwrap();
    assert_f16_bits("randn_f16_10", t.data().unwrap(), &TORCH_RANDN_42_F16_10);

    manual_seed(42).unwrap();
    let t = randn::<f16>(&[16]).unwrap();
    assert_f16_bits("randn_f16_16", t.data().unwrap(), &TORCH_RANDN_42_F16_16);

    manual_seed(42).unwrap();
    let t = randn::<f16>(&[17]).unwrap();
    assert_f16_bits("randn_f16_17", t.data().unwrap(), &TORCH_RANDN_42_F16_17);

    manual_seed(42).unwrap();
    let t = randn::<f16>(&[33]).unwrap();
    assert_f16_bits(
        "randn_f16_33_prefix",
        t.data().unwrap(),
        &TORCH_RANDN_42_F16_33_PREFIX,
    );
}

#[test]
fn randn_bf16_scalar_and_normal_fill_paths_match_torch() {
    let _guard = default_rng_test_lock();
    manual_seed(42).unwrap();
    let t = randn::<bf16>(&[10]).unwrap();
    assert_bf16_bits("randn_bf16_10", t.data().unwrap(), &TORCH_RANDN_42_BF16_10);

    manual_seed(42).unwrap();
    let t = randn::<bf16>(&[16]).unwrap();
    assert_bf16_bits("randn_bf16_16", t.data().unwrap(), &TORCH_RANDN_42_BF16_16);

    manual_seed(42).unwrap();
    let t = randn::<bf16>(&[17]).unwrap();
    assert_bf16_bits("randn_bf16_17", t.data().unwrap(), &TORCH_RANDN_42_BF16_17);

    manual_seed(42).unwrap();
    let t = randn::<bf16>(&[33]).unwrap();
    assert_bf16_bits(
        "randn_bf16_33_prefix",
        t.data().unwrap(),
        &TORCH_RANDN_42_BF16_33_PREFIX,
    );
}

#[test]
fn normal_fill_preserves_scalar_normal_cache_like_torch() {
    let _guard = default_rng_test_lock();

    manual_seed(42).unwrap();
    let first = randn::<f32>(&[1]).unwrap();
    let _large = randn::<f32>(&[16]).unwrap();
    let cached = randn::<f32>(&[1]).unwrap();
    let next_uniform = rand::<f32>(&[1]).unwrap();
    assert_f32_bits("cache_f32_first", first.data().unwrap(), &[0x3eac_62ae]);
    assert_f32_bits("cache_f32_cached", cached.data().unwrap(), &[0x3e03_e69d]);
    assert_f32_bits(
        "cache_f32_rand",
        next_uniform.data().unwrap(),
        &[0x3e8a_0d2a],
    );

    manual_seed(42).unwrap();
    let first = randn::<f64>(&[1]).unwrap();
    let _large = randn::<f64>(&[16]).unwrap();
    let cached = randn::<f64>(&[1]).unwrap();
    let next_uniform = rand::<f64>(&[1]).unwrap();
    assert_f64_bits(
        "cache_f64_first",
        first.data().unwrap(),
        &[0x3fd5_8c55_b33d_44cc],
    );
    assert_f64_bits(
        "cache_f64_cached",
        cached.data().unwrap(),
        &[0x3fc0_7cd3_9b3d_4b85],
    );
    assert_f64_bits(
        "cache_f64_rand",
        next_uniform.data().unwrap(),
        &[0x3fe3_f2eb_05e7_6b58],
    );

    manual_seed(42).unwrap();
    let first = randn::<f16>(&[1]).unwrap();
    let _large = randn::<f16>(&[16]).unwrap();
    let cached = randn::<f16>(&[1]).unwrap();
    let next_uniform = rand::<f16>(&[1]).unwrap();
    assert_f16_bits("cache_f16_first", first.data().unwrap(), &[0x3563]);
    assert_f16_bits("cache_f16_cached", cached.data().unwrap(), &[0x301f]);
    assert_f16_bits("cache_f16_rand", next_uniform.data().unwrap(), &[0x3a95]);

    manual_seed(42).unwrap();
    let first = randn::<bf16>(&[1]).unwrap();
    let _large = randn::<bf16>(&[16]).unwrap();
    let cached = randn::<bf16>(&[1]).unwrap();
    let next_uniform = rand::<bf16>(&[1]).unwrap();
    assert_bf16_bits("cache_bf16_first", first.data().unwrap(), &[0x3eac]);
    assert_bf16_bits("cache_bf16_cached", cached.data().unwrap(), &[0x3e04]);
    assert_bf16_bits("cache_bf16_rand", next_uniform.data().unwrap(), &[0x3f15]);
}
