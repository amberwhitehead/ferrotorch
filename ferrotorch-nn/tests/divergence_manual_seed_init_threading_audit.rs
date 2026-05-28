//! Companion audit probe for #1542 / #1537: pins that
//! `*_with_generator` init helpers ACTUALLY consume bits from the
//! passed-in `&mut Generator` rather than silently falling back to the
//! thread-local. Catches the `_generator: Option<&mut Generator>`
//! underscored-out red flag the dispatch prompt called out.

use ferrotorch_core::Generator;
use ferrotorch_nn::init::{
    NonLinearity, kaiming_normal_with_generator, kaiming_uniform_with_generator,
    normal_with_generator, uniform_with_generator, xavier_normal_with_generator,
    xavier_uniform_with_generator,
};
use ferrotorch_nn::parameter::Parameter;

/// Two identically-seeded generators must produce identical parameter
/// data through `uniform_with_generator`. If the helper ignored the
/// explicit generator and sampled from the thread-local, the two calls
/// would still differ because the thread-local would advance between
/// them.
#[test]
fn uniform_with_generator_uses_explicit_stream() {
    let mut p_a = Parameter::<f32>::zeros(&[64]).unwrap();
    let mut p_b = Parameter::<f32>::zeros(&[64]).unwrap();

    let mut g_a = Generator::new(42);
    let mut g_b = Generator::new(42);
    uniform_with_generator(&mut p_a, -1.0, 1.0, &mut g_a).unwrap();
    uniform_with_generator(&mut p_b, -1.0, 1.0, &mut g_b).unwrap();

    let a = p_a.data().unwrap();
    let b = p_b.data().unwrap();
    for i in 0..64 {
        assert_eq!(
            a[i].to_bits(),
            b[i].to_bits(),
            "i={i}: uniform_with_generator ignored explicit Generator(42)"
        );
    }
}

#[test]
fn normal_with_generator_uses_explicit_stream() {
    let mut p_a = Parameter::<f32>::zeros(&[64]).unwrap();
    let mut p_b = Parameter::<f32>::zeros(&[64]).unwrap();
    let mut g_a = Generator::new(7);
    let mut g_b = Generator::new(7);
    normal_with_generator(&mut p_a, 0.0, 1.0, &mut g_a).unwrap();
    normal_with_generator(&mut p_b, 0.0, 1.0, &mut g_b).unwrap();
    let a = p_a.data().unwrap();
    let b = p_b.data().unwrap();
    for i in 0..64 {
        assert_eq!(
            a[i].to_bits(),
            b[i].to_bits(),
            "i={i}: normal_with_generator ignored explicit Generator(7)"
        );
    }
}

#[test]
fn kaiming_with_generator_uses_explicit_stream() {
    let mut p_a = Parameter::<f32>::zeros(&[32, 64]).unwrap();
    let mut p_b = Parameter::<f32>::zeros(&[32, 64]).unwrap();
    let mut g_a = Generator::new(123);
    let mut g_b = Generator::new(123);
    kaiming_uniform_with_generator(&mut p_a, NonLinearity::ReLU, &mut g_a).unwrap();
    kaiming_uniform_with_generator(&mut p_b, NonLinearity::ReLU, &mut g_b).unwrap();
    let a = p_a.data().unwrap();
    let b = p_b.data().unwrap();
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "i={i}: kaiming_uniform_with_generator ignored explicit Generator(123)"
        );
    }

    let mut p_c = Parameter::<f32>::zeros(&[32, 64]).unwrap();
    let mut p_d = Parameter::<f32>::zeros(&[32, 64]).unwrap();
    let mut g_c = Generator::new(456);
    let mut g_d = Generator::new(456);
    kaiming_normal_with_generator(&mut p_c, NonLinearity::ReLU, &mut g_c).unwrap();
    kaiming_normal_with_generator(&mut p_d, NonLinearity::ReLU, &mut g_d).unwrap();
    let c = p_c.data().unwrap();
    let d = p_d.data().unwrap();
    for (i, (x, y)) in c.iter().zip(d.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "i={i}: kaiming_normal_with_generator ignored explicit Generator(456)"
        );
    }
}

#[test]
fn xavier_with_generator_uses_explicit_stream() {
    let mut p_a = Parameter::<f32>::zeros(&[16, 32]).unwrap();
    let mut p_b = Parameter::<f32>::zeros(&[16, 32]).unwrap();
    let mut g_a = Generator::new(99);
    let mut g_b = Generator::new(99);
    xavier_uniform_with_generator(&mut p_a, &mut g_a).unwrap();
    xavier_uniform_with_generator(&mut p_b, &mut g_b).unwrap();
    let a = p_a.data().unwrap();
    let b = p_b.data().unwrap();
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "i={i}: xavier_uniform_with_generator ignored explicit Generator(99)"
        );
    }

    let mut p_c = Parameter::<f32>::zeros(&[16, 32]).unwrap();
    let mut p_d = Parameter::<f32>::zeros(&[16, 32]).unwrap();
    let mut g_c = Generator::new(100);
    let mut g_d = Generator::new(100);
    xavier_normal_with_generator(&mut p_c, &mut g_c).unwrap();
    xavier_normal_with_generator(&mut p_d, &mut g_d).unwrap();
    let c = p_c.data().unwrap();
    let d = p_d.data().unwrap();
    for (i, (x, y)) in c.iter().zip(d.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "i={i}: xavier_normal_with_generator ignored explicit Generator(100)"
        );
    }
}

/// Different explicit generator seeds MUST produce different parameter
/// data. If the helper ignored the explicit Generator and used the
/// thread-local, this would still differ (because the thread-local
/// advances between calls), so this test on its own does NOT prove the
/// explicit generator is consumed. But combined with the
/// "same-seed-same-output" probes above, the conjunction does prove it.
#[test]
fn distinct_generator_seeds_distinct_init() {
    let mut p_a = Parameter::<f32>::zeros(&[64]).unwrap();
    let mut p_b = Parameter::<f32>::zeros(&[64]).unwrap();
    let mut g_a = Generator::new(1);
    let mut g_b = Generator::new(2);
    uniform_with_generator(&mut p_a, -1.0, 1.0, &mut g_a).unwrap();
    uniform_with_generator(&mut p_b, -1.0, 1.0, &mut g_b).unwrap();
    let a = p_a.data().unwrap();
    let b = p_b.data().unwrap();
    let differs = a
        .iter()
        .zip(b.iter())
        .any(|(x, y)| x.to_bits() != y.to_bits());
    assert!(
        differs,
        "two different explicit-Generator seeds produced identical streams"
    );
}
