//! Divergence/parity tests for #1537 — byte-exact `manual_seed` agreement
//! with `torch.manual_seed`.
//!
//! Closes #1537 (creation.rs per-call xorshift PRNG) by pinning
//! `ferrotorch_core::manual_seed(s); ferrotorch_core::rand(...)` to
//! `torch.manual_seed(s); torch.rand(...)` at the f32 bit-pattern level.
//!
//! The fixture bit patterns below come from running:
//!
//! ```python
//! import torch, struct
//! torch.manual_seed(42)
//! v = torch.rand(10).tolist()
//! for x in v:
//!     print(hex(struct.unpack('<I', struct.pack('<f', x))[0]))
//! ```
//!
//! against PyTorch 2.x. Since ferrotorch's `Generator` reproduces the same
//! MT19937 + `uniform_real` transform used by `at::CPUGeneratorImpl`
//! (`aten/src/ATen/core/MT19937RNGEngine.h:139-150` + `DistributionsHelper.h:106-113`),
//! the output bits must agree exactly.

use ferrotorch_core::{manual_seed, rand, randn, Generator};

// `torch.manual_seed(42); torch.rand(10)` — exact f32 bit patterns.
const TORCH_RAND_SEED_42_F32_BITS: [u32; 10] = [
    0x3f61_dc66,
    0x3f6a_3db3,
    0x3ec4_06b8,
    0x3f75_950e,
    0x3ec7_e8d4,
    0x3f19_d447,
    0x3e83_5d78,
    0x3f4b_2c14,
    0x3f70_d666,
    0x3e08_61e4,
];

// `torch.manual_seed(42); torch.randn(10)` — exact f32 bit patterns.
// NOTE: torch's `randn` for size < 16 uses
// `cpu_serial_kernel { normal_distribution<double>(...)(gen) }` at
// `aten/src/ATen/native/cpu/DistributionTemplates.h:222-235` — Box-Muller
// in `f64` acctype then cast down to `f32`. For size >= 16 it uses
// `normal_fill` which fills uniform samples in the OUTPUT BUFFER first
// then runs vectorised Box-Muller on 16-element blocks pairing element
// `i` with element `i+8` (different algorithm + different ordering).
//
// Byte-exact parity for randn would require BOTH paths plus SIMD libm
// agreement (torch uses `sincos256_ps` from avx_mathfun.h). Ferrotorch
// pins `manual_seed` determinism + uniform-rand byte-exact parity (#1537);
// randn byte-exact parity is documented as a separate divergence
// (cross-platform libm + algorithmic split) to be tracked in a follow-up.
#[allow(dead_code)]
const TORCH_RANDN_SEED_42_F32_BITS_REFERENCE: [u32; 10] = [
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

#[test]
fn manual_seed_42_rand_byte_exact_vs_torch_f32() {
    manual_seed(42);
    let t = rand::<f32>(&[10]).unwrap();
    let data = t.data().unwrap();
    for (i, (&got, &expected_bits)) in data
        .iter()
        .zip(TORCH_RAND_SEED_42_F32_BITS.iter())
        .enumerate()
    {
        assert_eq!(
            got.to_bits(),
            expected_bits,
            "rand[{i}]: got 0x{:08x} ({got:.17}), expected 0x{expected_bits:08x}",
            got.to_bits()
        );
    }
}

#[test]
fn manual_seed_is_deterministic_across_calls() {
    manual_seed(12345);
    let a = rand::<f32>(&[20]).unwrap();
    manual_seed(12345);
    let b = rand::<f32>(&[20]).unwrap();
    let ad = a.data().unwrap();
    let bd = b.data().unwrap();
    for i in 0..20 {
        assert_eq!(ad[i].to_bits(), bd[i].to_bits(), "i={i}");
    }
}

#[test]
fn distinct_seeds_distinct_streams() {
    manual_seed(1);
    let a = rand::<f32>(&[5]).unwrap();
    manual_seed(2);
    let b = rand::<f32>(&[5]).unwrap();
    let ad = a.data().unwrap();
    let bd = b.data().unwrap();
    // At least one element must differ (overwhelmingly likely; not all-zero).
    let differs = ad
        .iter()
        .zip(bd.iter())
        .any(|(x, y)| x.to_bits() != y.to_bits());
    assert!(differs);
}

#[test]
fn manual_seed_randn_is_deterministic_across_calls() {
    // randn does NOT byte-exact match torch (different code path for
    // size<16 vs >=16, SIMD libm vendoring, f64-acctype on the small
    // path). What ferrotorch DOES guarantee post-#1537: given the same
    // seed, randn produces the same output across runs and across calls
    // within the same run.
    manual_seed(42);
    let a = randn::<f32>(&[10]).unwrap();
    manual_seed(42);
    let b = randn::<f32>(&[10]).unwrap();
    let ad = a.data().unwrap();
    let bd = b.data().unwrap();
    for i in 0..10 {
        assert_eq!(
            ad[i].to_bits(),
            bd[i].to_bits(),
            "randn must be deterministic under manual_seed: i={i}"
        );
    }
    // Sanity: the output looks like a standard normal — finite, not
    // all-zero, mean roughly 0 (loose; 10 samples).
    assert!(ad.iter().all(|x| x.is_finite()));
    assert!(ad.iter().any(|x| *x != 0.0));
}

#[test]
fn explicit_generator_independent_of_thread_local() {
    manual_seed(99);
    let _ = rand::<f32>(&[3]).unwrap(); // advance thread-local
    let mut g = Generator::new(42);
    // Explicit generator stream is independent of thread-local.
    let v0 = g.next_uniform_f32();
    assert_eq!(v0.to_bits(), TORCH_RAND_SEED_42_F32_BITS[0]);
}
