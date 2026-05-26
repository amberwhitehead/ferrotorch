//! Extended #1454 follow-up probes: pins that the structured-initializer
//! `*_with_generator` variants — `trunc_normal_with_generator`,
//! `orthogonal_with_generator`, `sparse_with_generator` — ACTUALLY consume
//! bits from the passed-in `&mut Generator` rather than silently falling
//! back to the thread-local. Companion to
//! `divergence_manual_seed_init_threading_audit.rs` which covers the
//! uniform/normal/xavier/kaiming family.
//!
//! Note: `dirac_` is intentionally NOT covered here — upstream
//! `torch.nn.init.dirac_` at `torch/nn/init.py:402-455` takes no
//! `generator` kwarg because the function is deterministic (places `1.0`
//! at the kernel center along the channel diagonal, consumes 0 random
//! bits). Adding `dirac_with_generator` would be a vacuous API that
//! ignores its `&mut Generator` argument.

use ferrotorch_core::Generator;
use ferrotorch_nn::init::{
    orthogonal_with_generator, sparse_with_generator, trunc_normal_with_generator,
};
use ferrotorch_nn::parameter::Parameter;

/// Two identically-seeded generators must produce bit-identical truncated-
/// normal samples. If the helper ignored the explicit generator and
/// sampled from the thread-local, the two calls would still differ
/// because the thread-local would advance between them.
#[test]
fn trunc_normal_with_generator_uses_explicit_stream() {
    let mut p_a = Parameter::<f32>::zeros(&[256]).unwrap();
    let mut p_b = Parameter::<f32>::zeros(&[256]).unwrap();
    let mut g_a = Generator::new(42);
    let mut g_b = Generator::new(42);
    trunc_normal_with_generator(&mut p_a, 0.0, 1.0, -2.0, 2.0, &mut g_a).unwrap();
    trunc_normal_with_generator(&mut p_b, 0.0, 1.0, -2.0, 2.0, &mut g_b).unwrap();
    let a = p_a.data().unwrap();
    let b = p_b.data().unwrap();
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "i={i}: trunc_normal_with_generator ignored explicit Generator(42)"
        );
    }
}

#[test]
fn trunc_normal_with_generator_respects_bounds() {
    let mut p = Parameter::<f32>::zeros(&[2048]).unwrap();
    let mut g = Generator::new(11);
    trunc_normal_with_generator(&mut p, 0.0, 1.0, -2.0, 2.0, &mut g).unwrap();
    let data = p.data().unwrap();
    assert!(
        data.iter().all(|&x| (-2.0..=2.0).contains(&x)),
        "trunc_normal_with_generator violated [-2,2] bound"
    );
}

#[test]
fn orthogonal_with_generator_uses_explicit_stream() {
    let mut p_a = Parameter::<f64>::zeros(&[32, 16]).unwrap();
    let mut p_b = Parameter::<f64>::zeros(&[32, 16]).unwrap();
    let mut g_a = Generator::new(7);
    let mut g_b = Generator::new(7);
    orthogonal_with_generator(&mut p_a, 1.0, &mut g_a).unwrap();
    orthogonal_with_generator(&mut p_b, 1.0, &mut g_b).unwrap();
    let a = p_a.data().unwrap();
    let b = p_b.data().unwrap();
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "i={i}: orthogonal_with_generator ignored explicit Generator(7)"
        );
    }
}

#[test]
fn orthogonal_with_generator_preserves_orthonormality() {
    // Verify the explicit-generator path still produces an orthonormal result.
    let mut p = Parameter::<f64>::zeros(&[16, 16]).unwrap();
    let mut g = Generator::new(99);
    orthogonal_with_generator(&mut p, 1.0, &mut g).unwrap();
    let data = p.data().unwrap();
    let n = 16;
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0;
            for k in 0..n {
                dot += data[k * n + i] * data[k * n + j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-6,
                "Q^T Q [{i},{j}] = {dot}, expected {expected}"
            );
        }
    }
}

#[test]
fn sparse_with_generator_uses_explicit_stream() {
    let mut p_a = Parameter::<f32>::zeros(&[64, 32]).unwrap();
    let mut p_b = Parameter::<f32>::zeros(&[64, 32]).unwrap();
    let mut g_a = Generator::new(123);
    let mut g_b = Generator::new(123);
    sparse_with_generator(&mut p_a, 0.5, 1.0, &mut g_a).unwrap();
    sparse_with_generator(&mut p_b, 0.5, 1.0, &mut g_b).unwrap();
    let a = p_a.data().unwrap();
    let b = p_b.data().unwrap();
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "i={i}: sparse_with_generator ignored explicit Generator(123) (covers BOTH N(0,std) sampling AND Fisher-Yates index draws)"
        );
    }
}

#[test]
fn sparse_with_generator_preserves_sparsity_ratio() {
    let mut p = Parameter::<f32>::zeros(&[100, 50]).unwrap();
    let mut g = Generator::new(321);
    sparse_with_generator(&mut p, 0.9, 0.01, &mut g).unwrap();
    let data = p.data().unwrap();
    let num_zeros = data.iter().filter(|&&x| x == 0.0).count();
    let total = data.len();
    let actual_sparsity = num_zeros as f64 / total as f64;
    assert!(
        (actual_sparsity - 0.9).abs() < 0.05,
        "sparse_with_generator sparsity = {actual_sparsity}, expected ~0.9"
    );
}

/// Distinct seeds for `trunc_normal_with_generator` must produce distinct
/// outputs (combined with the same-seed equality probe above, the pair
/// proves the explicit Generator is the actual entropy source).
#[test]
fn trunc_normal_distinct_seeds_distinct_init() {
    let mut p_a = Parameter::<f32>::zeros(&[128]).unwrap();
    let mut p_b = Parameter::<f32>::zeros(&[128]).unwrap();
    let mut g_a = Generator::new(1);
    let mut g_b = Generator::new(2);
    trunc_normal_with_generator(&mut p_a, 0.0, 1.0, -2.0, 2.0, &mut g_a).unwrap();
    trunc_normal_with_generator(&mut p_b, 0.0, 1.0, -2.0, 2.0, &mut g_b).unwrap();
    let a = p_a.data().unwrap();
    let b = p_b.data().unwrap();
    let differs = a
        .iter()
        .zip(b.iter())
        .any(|(x, y)| x.to_bits() != y.to_bits());
    assert!(
        differs,
        "trunc_normal_with_generator: two different seeds produced identical streams"
    );
}

#[test]
fn orthogonal_distinct_seeds_distinct_init() {
    let mut p_a = Parameter::<f64>::zeros(&[16, 16]).unwrap();
    let mut p_b = Parameter::<f64>::zeros(&[16, 16]).unwrap();
    let mut g_a = Generator::new(101);
    let mut g_b = Generator::new(202);
    orthogonal_with_generator(&mut p_a, 1.0, &mut g_a).unwrap();
    orthogonal_with_generator(&mut p_b, 1.0, &mut g_b).unwrap();
    let a = p_a.data().unwrap();
    let b = p_b.data().unwrap();
    let differs = a
        .iter()
        .zip(b.iter())
        .any(|(x, y)| x.to_bits() != y.to_bits());
    assert!(
        differs,
        "orthogonal_with_generator: two different seeds produced identical streams"
    );
}

#[test]
fn sparse_distinct_seeds_distinct_init() {
    let mut p_a = Parameter::<f32>::zeros(&[64, 32]).unwrap();
    let mut p_b = Parameter::<f32>::zeros(&[64, 32]).unwrap();
    let mut g_a = Generator::new(11);
    let mut g_b = Generator::new(22);
    sparse_with_generator(&mut p_a, 0.5, 1.0, &mut g_a).unwrap();
    sparse_with_generator(&mut p_b, 0.5, 1.0, &mut g_b).unwrap();
    let a = p_a.data().unwrap();
    let b = p_b.data().unwrap();
    let differs = a
        .iter()
        .zip(b.iter())
        .any(|(x, y)| x.to_bits() != y.to_bits());
    assert!(
        differs,
        "sparse_with_generator: two different seeds produced identical streams"
    );
}
