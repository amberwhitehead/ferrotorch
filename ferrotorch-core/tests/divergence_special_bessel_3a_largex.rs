//! Discriminator probes for #1651 batch 3a (commit c9b9ffe6c):
//! spherical_bessel_j0 + modified_bessel_k0/k1 (+scaled).
//!
//! The generator's in-crate tests cover spherical_bessel_j0 on a grid that
//! tops out at |x|=10 and a K-family grid topping out at x=50/700. These
//! probes extend the audit to the inputs the generator's grid did NOT touch:
//!   - spherical_bessel_j0 at large argument (100..1e8) where sin(x)/x argument
//!     reduction is the divergence-prone path,
//!   - subnormal / tiny x (the Taylor branch limit),
//!   - the |x| == 0.5 region boundary (and just below) including negative x,
//!   - K-family at the x==2.0 region boundary and just-above/just-below.
//!
//! Every `want` below is a LIVE torch.special.* f64 value (torch 2.11.0+cu130),
//! captured via the oracle, NOT copied from the ferrotorch side (R-CHAR-3).

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_core::{
    modified_bessel_k0, modified_bessel_k1, scaled_modified_bessel_k0, scaled_modified_bessel_k1,
    spherical_bessel_j0,
};

fn tf64(data: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
}

/// Probe: spherical_bessel_j0 at large argument.
/// Live torch.special.spherical_bessel_j0(f64):
///   100      -> -0.005063656411097588
///   1000     ->  0.0008268795405320025
///   1e6      -> -3.4999350217129296e-07
///   1e8      ->  9.31639027109726e-09
/// Upstream sin(x)/x at large x (`aten/.../cuda/Math.cuh:3051`).
#[test]
fn spherical_bessel_j0_large_x_vs_torch() {
    let xs = [100.0, 1000.0, 1e6, 1e8];
    let want = [
        -0.005_063_656_411_097_588,
        0.000_826_879_540_532_002_5,
        -3.499_935_021_712_929_6e-7,
        9.316_390_271_097_26e-9,
    ];
    let r = spherical_bessel_j0(&tf64(&xs)).unwrap();
    let d = r.data().unwrap();
    for i in 0..xs.len() {
        assert!(
            (d[i] - want[i]).abs() <= 1e-12 * (1.0 + want[i].abs()),
            "spherical_bessel_j0({}) got {} want {} (torch)",
            xs[i],
            d[i],
            want[i]
        );
    }
}

/// Probe: spherical_bessel_j0 tiny / subnormal x and the |x|==0.5 boundary
/// (incl. negative). Live torch:
///   1e-8   -> 1.0
///   1e-300 -> 1.0
///   -0.49  -> 0.9604609962676695
///   -0.5   -> 0.958851077208406   (boundary -> sin(x)/x, |x|<0.5 is strict)
///    0.5   -> 0.958851077208406
#[test]
fn spherical_bessel_j0_tiny_and_boundary_vs_torch() {
    let xs = [1e-8, 1e-300, -0.49, -0.5, 0.5];
    let want = [
        1.0,
        1.0,
        0.960_460_996_267_669_5,
        0.958_851_077_208_406,
        0.958_851_077_208_406,
    ];
    let r = spherical_bessel_j0(&tf64(&xs)).unwrap();
    let d = r.data().unwrap();
    for i in 0..xs.len() {
        assert!(
            (d[i] - want[i]).abs() <= 1e-12 * (1.0 + want[i].abs()),
            "spherical_bessel_j0({}) got {} want {} (torch)",
            xs[i],
            d[i],
            want[i]
        );
    }
}

/// Probe: K-family right at the x==2.0 small/large region boundary and the two
/// sides of it (1.9999999, 2.0, 2.0000001). The split is `x <= 2.0` in both
/// upstream and ferrotorch; this confirms continuity across the seam.
/// Live torch.special f64:
///   k0:  1.9999999 -> 0.11389388673612258 | 2.0 -> 0.1138938727495334 | 2.0000001 -> 0.11389385876294619
///   k1:  1.9999999 -> 0.13986590019920514 | 2.0 -> 0.13986588181652246 | 2.0000001 -> 0.13986586343384244
///   sk0: 1.9999999 -> 0.8415682342616356  | 2.0 -> 0.8415682150707712  | 2.0000001 -> 0.8415681958799089
///   sk1: 1.9999999 -> 1.0334768795516693  | 2.0 -> 1.0334768470686888  | 2.0000001 -> 1.033476814585711
#[test]
fn k_family_region_boundary_vs_torch() {
    let xs = [1.9999999, 2.0, 2.0000001];
    let cases: [(
        &str,
        fn(&Tensor<f64>) -> ferrotorch_core::FerrotorchResult<Tensor<f64>>,
        [f64; 3],
    ); 4] = [
        (
            "k0",
            modified_bessel_k0,
            [
                0.113_893_886_736_122_58,
                0.113_893_872_749_533_4,
                0.113_893_858_762_946_19,
            ],
        ),
        (
            "k1",
            modified_bessel_k1,
            [
                0.139_865_900_199_205_14,
                0.139_865_881_816_522_46,
                0.139_865_863_433_842_44,
            ],
        ),
        (
            "scaled_k0",
            scaled_modified_bessel_k0,
            [
                0.841_568_234_261_635_6,
                0.841_568_215_070_771_2,
                0.841_568_195_879_908_9,
            ],
        ),
        (
            "scaled_k1",
            scaled_modified_bessel_k1,
            [
                1.033_476_879_551_669_3,
                1.033_476_847_068_688_8,
                1.033_476_814_585_711,
            ],
        ),
    ];
    for (name, f, want) in cases {
        let r = f(&tf64(&xs)).unwrap();
        let d = r.data().unwrap();
        for i in 0..xs.len() {
            assert!(
                (d[i] - want[i]).abs() <= 1e-12 * (1.0 + want[i].abs()),
                "{name}({}) got {} want {} (torch)",
                xs[i],
                d[i],
                want[i]
            );
        }
    }
}

/// Internal-consistency cross-check requested by the audit brief:
/// scaled_k(x) == k(x) * exp(x) for moderate x, evaluated INDEPENDENTLY
/// (the scaled path drops exp(-x) in the large region rather than multiplying,
/// so this catches a region-split mismatch between the two pub fns).
/// This is NOT tautological: lhs and rhs are produced by different code paths
/// (scaled fn vs unscaled fn * libm exp), and the relation is the mathematical
/// definition `scaled_modified_bessel_k(x) = exp(x) * modified_bessel_k(x)`.
#[test]
fn scaled_equals_unscaled_times_exp_x() {
    let xs = [0.3, 1.0, 2.0, 2.5, 5.0, 20.0];
    let r_k0 = modified_bessel_k0(&tf64(&xs)).unwrap();
    let r_sk0 = scaled_modified_bessel_k0(&tf64(&xs)).unwrap();
    let r_k1 = modified_bessel_k1(&tf64(&xs)).unwrap();
    let r_sk1 = scaled_modified_bessel_k1(&tf64(&xs)).unwrap();
    let (k0, sk0, k1, sk1) = (
        r_k0.data().unwrap(),
        r_sk0.data().unwrap(),
        r_k1.data().unwrap(),
        r_sk1.data().unwrap(),
    );
    for i in 0..xs.len() {
        let e = xs[i].exp();
        assert!(
            (sk0[i] - k0[i] * e).abs() <= 1e-10 * (1.0 + (k0[i] * e).abs()),
            "scaled_k0({}) {} != k0*exp {}",
            xs[i],
            sk0[i],
            k0[i] * e
        );
        assert!(
            (sk1[i] - k1[i] * e).abs() <= 1e-10 * (1.0 + (k1[i] * e).abs()),
            "scaled_k1({}) {} != k1*exp {}",
            xs[i],
            sk1[i],
            k1[i] * e
        );
    }
}
