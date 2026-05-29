//! Adversarial RE-AUDIT of commit 66be142e9 (#1675): fused single-launch GPU
//! `add_scaled` kernel (FMA) — `out[i] = a[i] + alpha*b[i]`.
//!
//! Background: the NEW fused kernel (`ADD_SCALED_PTX` scalar +
//! `ADD_SCALED_VEC4_PTX` 4-lane, kernels.rs) now serves
//! `sub` / `sub_scaled` / `rsub` / `add_scaled` on same-shape same-device
//! CUDA f32/f64. The launcher `gpu_add_scaled_f32` dispatches the vec4
//! kernel when `n >= 16 && n % 4 == 0` and the scalar kernel otherwise
//! (incl. the `n % 4 != 0` tail and `n < 16`); `gpu_add_scaled_f64` is
//! scalar-only (f64 PTX auto-derived by `ptx_f32_to_f64`).
//!
//! These tests call the launchers DIRECTLY (`ferrotorch_gpu::kernels::
//! gpu_add_scaled_f{32,64}`) so the dispatch boundary and per-lane mapping
//! are exercised without `add_scaled`'s `alpha==1.0` short-circuit masking
//! anything.
//!
//! Reference oracles (R-CHAR-3 — never literal-copy the ferrotorch side):
//!   * alpha=-1 (sub): `fma(-1, b, a) = a - b` is provably a single
//!     rounding (the `-1*b` product is exact), so host IEEE-754 `a - b` is
//!     the correctly-rounded reference. This is the load-bearing claim for
//!     GPU `sub`. Also cross-checked vs the torch values dumped in the
//!     `EXPECT_*` symbolic tables below (computed offline with PyTorch
//!     `torch.add(a, b, alpha=...)`).
//!   * general finite alpha: CUDA `fma.rn.f{32,64}` and Rust
//!     `f{32,64}::mul_add` both compute the correctly-rounded FMA, so the
//!     host `mul_add` is the single-rounding reference and matches torch's
//!     fused `add_stub` to <=0.5 ULP.
//!   * alpha=0 with inf/nan b: torch `add(a, b, alpha=0) = a + 0*b`, and
//!     `0*inf = 0*nan = NaN` in IEEE-754, so torch returns NaN wherever b is
//!     non-finite (verified live: `torch.add([..], [inf,nan,-inf,1], 0) =
//!     [nan,nan,nan,1]`). The fused `fma(0, b, a)` must reproduce this.
//!
//! All tests require a live CUDA device (RTX 3090 in the audit env).

#![cfg(feature = "cuda")]

use std::sync::Once;

use ferrotorch_gpu::device::GpuDevice;
use ferrotorch_gpu::kernels::{gpu_add_scaled_f32, gpu_add_scaled_f64};
use ferrotorch_gpu::transfer::{cpu_to_gpu, gpu_to_cpu};

fn ensure_cuda() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend().expect("CUDA backend init for #1675 reaudit");
    });
}

fn device() -> GpuDevice {
    GpuDevice::new(0).expect("GpuDevice::new(0)")
}

/// Distinct per-element, per-lane inputs. Distinctness is load-bearing: an
/// all-same buffer would pass even with a swapped vec4 lane offset (every
/// lane would read the same value). Here a[i]/b[i] are unique across i and
/// across lanes within each 4-group.
fn make_inputs_f32(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for i in 0..n {
        a.push((i as f32) * 0.5 - (n as f32) * 0.25 + 0.125);
        b.push(((i % 89) as f32) * 1.1875 - 11.0 + (i as f32) * 7e-4);
    }
    (a, b)
}

fn make_inputs_f64(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for i in 0..n {
        a.push((i as f64) * 0.5 - (n as f64) * 0.25 + 0.125);
        b.push(((i % 89) as f64) * 1.1875 - 11.0 + (i as f64) * 7e-4);
    }
    (a, b)
}

fn run_f32(a: &[f32], b: &[f32], alpha: f32) -> Vec<f32> {
    let dev = device();
    let ga = cpu_to_gpu(a, &dev).expect("h2d a");
    let gb = cpu_to_gpu(b, &dev).expect("h2d b");
    let out = gpu_add_scaled_f32(&ga, &gb, alpha, &dev).expect("gpu_add_scaled_f32");
    gpu_to_cpu(&out, &dev).expect("d2h")
}

fn run_f64(a: &[f64], b: &[f64], alpha: f64) -> Vec<f64> {
    let dev = device();
    let ga = cpu_to_gpu(a, &dev).expect("h2d a");
    let gb = cpu_to_gpu(b, &dev).expect("h2d b");
    let out = gpu_add_scaled_f64(&ga, &gb, alpha, &dev).expect("gpu_add_scaled_f64");
    gpu_to_cpu(&out, &dev).expect("d2h")
}

fn assert_bits_f32(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        if want.is_nan() {
            assert!(got.is_nan(), "{label}[{i}]: want NaN got {got}");
            continue;
        }
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{label}[{i}]: GPU={got} (bits {:#010x}) != ref={want} (bits {:#010x})",
            got.to_bits(),
            want.to_bits()
        );
    }
}

fn assert_bits_f64(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        if want.is_nan() {
            assert!(got.is_nan(), "{label}[{i}]: want NaN got {got}");
            continue;
        }
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{label}[{i}]: GPU={got} (bits {:#018x}) != ref={want} (bits {:#018x})",
            got.to_bits(),
            want.to_bits()
        );
    }
}

// ===========================================================================
// 1. sub (alpha=-1) BIT-EXACT a - b across vec4 / tail / scalar / large sizes.
// ===========================================================================

#[test]
fn sub_alpha_neg1_bit_exact_f32_all_sizes() {
    ensure_cuda();
    // vec4 sizes (n>=16, n%4==0): 16, 1024, 4096, 1_000_000.
    // tail sizes (n%4!=0): 17, 1001, 999_999.
    // sub-vec4 scalar sizes (n<16): 1, 3, 8, 15.
    for &n in &[
        1usize, 3, 8, 15, 16, 17, 1001, 1024, 4096, 999_999, 1_000_000,
    ] {
        let (a, b) = make_inputs_f32(n);
        let got = run_f32(&a, &b, -1.0);
        // fma(-1, b, a) = a - b, single rounding == host IEEE a - b.
        let want: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x - y).collect();
        assert_bits_f32(&got, &want, &format!("sub f32 n={n}"));
    }
}

#[test]
fn sub_alpha_neg1_bit_exact_f64_all_sizes() {
    ensure_cuda();
    for &n in &[1usize, 3, 8, 15, 16, 17, 1001, 1024, 4096, 99_999] {
        let (a, b) = make_inputs_f64(n);
        let got = run_f64(&a, &b, -1.0);
        let want: Vec<f64> = a.iter().zip(&b).map(|(&x, &y)| x - y).collect();
        assert_bits_f64(&got, &want, &format!("sub f64 n={n}"));
    }
}

// ===========================================================================
// 2. add_scaled general finite alpha vs torch's correctly-rounded fused FMA.
//    Host mul_add == CUDA fma.rn == torch add_stub (single rounding).
// ===========================================================================

#[test]
fn add_scaled_general_alpha_matches_fma_f32() {
    ensure_cuda();
    // 1000 -> vec4 (1000%4==0); 999 -> scalar tail.
    for &n in &[999usize, 1000, 4096] {
        let (a, b) = make_inputs_f32(n);
        for &alpha in &[-1.0f32, 2.5, -0.5, 0.0, 3.0] {
            let got = run_f32(&a, &b, alpha);
            let want: Vec<f32> = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| y.mul_add(alpha, x))
                .collect();
            assert_bits_f32(&got, &want, &format!("add_scaled f32 n={n} alpha={alpha}"));
        }
    }
}

#[test]
fn add_scaled_general_alpha_matches_fma_f64() {
    ensure_cuda();
    for &n in &[999usize, 1000, 4096] {
        let (a, b) = make_inputs_f64(n);
        for &alpha in &[-1.0f64, 2.5, -0.5, 0.0, 3.0] {
            let got = run_f64(&a, &b, alpha);
            let want: Vec<f64> = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| y.mul_add(alpha, x))
                .collect();
            assert_bits_f64(&got, &want, &format!("add_scaled f64 n={n} alpha={alpha}"));
        }
    }
}

// ===========================================================================
// 3. *** alpha=0 with inf/nan b *** — the suspect edge the shipped test
//    explicitly skips (it claims this is "covered by the fall-through, not
//    this fused path", but alpha=0.0 IS finite, so SAME-shape CUDA f32/f64
//    add_scaled takes the FUSED path: fma(0, b, a)). Torch returns NaN
//    wherever b is non-finite (a + 0*b, 0*inf = 0*nan = NaN), finite-b
//    positions return a exactly. Verified live:
//      torch.add([1..16], [inf,nan,-inf,1]*4, alpha=0)
//        = [nan,nan,nan,1, nan,nan,nan,5, ...]
//    The fused fma must reproduce this on BOTH the vec4 (n=16, n%4==0) and
//    the scalar tail (n=17) launch paths.
// ===========================================================================

/// Build a/b where b has inf/nan in lanes 0,1,2 of each 4-group and a finite
/// value in lane 3, so torch's `a + 0*b` is NaN in lanes 0..2 and `a` in
/// lane 3 of every group.
fn make_alpha0_edge(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for i in 0..n {
        a.push((i as f32) + 1.0);
        let bi = match i % 4 {
            0 => f32::INFINITY,
            1 => f32::NAN,
            2 => f32::NEG_INFINITY,
            _ => 2.0 + i as f32,
        };
        b.push(bi);
    }
    (a, b)
}

#[test]
fn add_scaled_alpha0_inf_nan_b_matches_torch_f32_vec4() {
    ensure_cuda();
    // n=16 -> vec4 launch.
    let (a, b) = make_alpha0_edge(16);
    let got = run_f32(&a, &b, 0.0);
    // torch: a + 0*b ; 0*inf = 0*nan = NaN. Reproduce the torch reference.
    let want: Vec<f32> = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| if y.is_finite() { x } else { f32::NAN })
        .collect();
    assert_bits_f32(&got, &want, "add_scaled f32 alpha=0 inf/nan b (vec4 n=16)");
}

#[test]
fn add_scaled_alpha0_inf_nan_b_matches_torch_f32_scalar_tail() {
    ensure_cuda();
    // n=17 -> scalar launch (n%4!=0).
    let (a, b) = make_alpha0_edge(17);
    let got = run_f32(&a, &b, 0.0);
    let want: Vec<f32> = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| if y.is_finite() { x } else { f32::NAN })
        .collect();
    assert_bits_f32(
        &got,
        &want,
        "add_scaled f32 alpha=0 inf/nan b (scalar n=17)",
    );
}

#[test]
fn add_scaled_alpha0_inf_nan_b_matches_torch_f64() {
    ensure_cuda();
    let n = 16;
    let (af, bf) = make_alpha0_edge(n);
    let a: Vec<f64> = af.iter().map(|&x| x as f64).collect();
    let b: Vec<f64> = bf.iter().map(|&x| x as f64).collect();
    let got = run_f64(&a, &b, 0.0);
    let want: Vec<f64> = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| if y.is_finite() { x } else { f64::NAN })
        .collect();
    assert_bits_f64(&got, &want, "add_scaled f64 alpha=0 inf/nan b");
}

// ===========================================================================
// 4. VEC4 LANE CORRECTNESS — distinct per-lane values, size a multiple of 4
//    spanning many blocks, so a wrong lane->index mapping visibly corrupts.
//    n=8192 spans 16 blocks at 128 threads/block / 4 elems each (2048 threads
//    -> 16 blocks). Each lane i must compute a[i] + alpha*b[i] for the RIGHT i.
// ===========================================================================

#[test]
fn vec4_lane_mapping_correct_f32() {
    ensure_cuda();
    let n = 8192;
    let (a, b) = make_inputs_f32(n);
    // alpha=2.5 (not ±1 or 0) so a wrong lane is not accidentally masked.
    let alpha = 2.5f32;
    let got = run_f32(&a, &b, alpha);
    let want: Vec<f32> = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| y.mul_add(alpha, x))
        .collect();
    assert_bits_f32(&got, &want, "vec4 lane mapping f32 n=8192 alpha=2.5");
}

// ===========================================================================
// 5. Symbolic torch cross-check table (R-CHAR-3 (b)): values computed offline
//    with PyTorch `torch.add(a, b, alpha=alpha)` for a tiny hand-picked input,
//    pinned as named constants. These are NOT copied from ferrotorch.
// ===========================================================================

#[test]
fn add_scaled_matches_torch_reference_table_f32() {
    ensure_cuda();
    // a = [0.0, 0.5, 1.0, ..., 7.5] (i*0.5), b = [-3.0, -1.75, ..., 15.0]
    // (i*1.25 - 3.0), n=16 (vec4). torch.add(a, b, alpha):
    let a: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..16).map(|i| i as f32 * 1.25 - 3.0).collect();

    // alpha=-1: torch.add(a,b,alpha=-1) = a - b.
    const EXPECT_NEG1: [f32; 16] = [
        3.0, 2.25, 1.5, 0.75, 0.0, -0.75, -1.5, -2.25, -3.0, -3.75, -4.5, -5.25, -6.0, -6.75, -7.5,
        -8.25,
    ];
    let got = run_f32(&a, &b, -1.0);
    assert_bits_f32(&got, &EXPECT_NEG1, "torch table alpha=-1");

    // alpha=2.5: torch.add(a,b,alpha=2.5) = a + 2.5*b.
    const EXPECT_2_5: [f32; 16] = [
        -7.5, -3.875, -0.25, 3.375, 7.0, 10.625, 14.25, 17.875, 21.5, 25.125, 28.75, 32.375, 36.0,
        39.625, 43.25, 46.875,
    ];
    let got = run_f32(&a, &b, 2.5);
    assert_bits_f32(&got, &EXPECT_2_5, "torch table alpha=2.5");
}
