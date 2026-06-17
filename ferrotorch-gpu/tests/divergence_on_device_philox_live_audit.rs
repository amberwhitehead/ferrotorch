//! Adversarial re-audit of the LIVE on-device Philox PTX kernels (#1684).
//!
//! Commit 2240fa9e2 fixed PHILOX_UNIFORM_PTX / PHILOX_NORMAL_PTX so they JIT
//! and execute on the GPU for the first time (the `%tid` register shadow bug).
//! The prior audit (310a0c545) only verified the CPU fallback. These probes
//! exercise the *on-device* kernel output directly on a live RTX 3090.
//!
//! Reference discipline (R-CHAR-3): the GPU uniform kernel is written to mirror
//! the CPU `PhiloxGenerator` EXACTLY (output `i` = word `i%4` of Philox counter
//! `base + i/4`, rng.rs:570-571 + next_f32 at rng.rs:236-241). The CPU
//! PhiloxGenerator is the byte-exact-with-torch reference (#1683: "first value
//! matches torch"). So a correct on-device uniform kernel MUST be bit-identical
//! to `PhiloxGenerator::generate_uniform` for the same seed/counter. The expected
//! values come from the CPU PhiloxGenerator (NOT from the GPU side, and NOT from
//! `rand_on_device(Cpu)` which is a DIFFERENT algorithm — MT19937).
//!
//! The normal kernel uses .approx PTX transcendentals so it is NOT bit-identical
//! to the libm CPU Box-Muller; for it we assert distribution moments against the
//! analytic standard-normal values plus exact element count/finiteness.
//!
//! SERIALIZATION: the CUDA RNG manager is a process-global singleton. Tests that
//! seed-then-sample (and especially the multi-call stream-continuity probe) hold
//! `SEED_LOCK` across their seed+sample window so a concurrent test's
//! `manual_seed` cannot perturb the shared per-device Philox counter between
//! calls. Without this guard the stream-continuity probe is flaky under the
//! default multi-threaded test runner (the failure is a harness race, not a
//! kernel divergence — verified: it passes in isolation / single-threaded).

use ferrotorch_core::{Device, manual_seed, rand_on_device, randn_on_device};
use ferrotorch_gpu::init_cuda_backend;
use ferrotorch_gpu::rng::PhiloxGenerator;
use std::sync::Mutex;

static SEED_LOCK: Mutex<()> = Mutex::new(());

fn ensure_init() {
    if !ferrotorch_core::gpu_dispatch::has_gpu_backend() {
        init_cuda_backend().expect("init_cuda_backend");
    }
}

fn to_host(t: &ferrotorch_core::Tensor<f32>) -> Vec<f32> {
    let cpu = t.to(Device::Cpu).expect("tensor.to(Cpu)");
    cpu.data().expect("cpu data").to_vec()
}

fn to_host_f64(t: &ferrotorch_core::Tensor<f64>) -> Vec<f64> {
    let cpu = t.to(Device::Cpu).expect("tensor.to(Cpu)");
    cpu.data().expect("cpu data").to_vec()
}

/// PROBE 1+3 — UNIFORM on-device, boundary lengths (n not divisible by 4).
/// Reference: CPU PhiloxGenerator (byte-exact-with-torch). A correct kernel is
/// bit-identical at the awkward 4-lane boundaries n = 5, 7, 4097.
#[test]
fn uniform_gpu_bit_exact_with_philox_reference_boundaries() {
    ensure_init();
    for &n in &[1usize, 4, 5, 7, 8, 4096, 4097] {
        let seed = 2024u64;
        let _g = SEED_LOCK.lock().unwrap();
        manual_seed(seed);
        let gpu = to_host(&rand_on_device::<f32>(&[n], Device::Cuda(0)).expect("gpu uniform"));
        let cpu = PhiloxGenerator::new(seed).generate_uniform(n);
        assert_eq!(gpu.len(), n, "gpu uniform wrong length at n={n}");
        assert_eq!(
            gpu, cpu,
            "on-device uniform kernel diverges from CPU Philox reference at n={n}"
        );
    }
}

/// PROBE 1 — UNIFORM range invariant: every on-device value strictly in [0,1).
#[test]
fn uniform_gpu_strict_unit_interval() {
    ensure_init();
    let v = {
        let _g = SEED_LOCK.lock().unwrap();
        manual_seed(11);
        to_host(&rand_on_device::<f32>(&[1_000_000], Device::Cuda(0)).expect("gpu uniform"))
    };
    for &x in &v {
        assert!(
            (0.0..1.0).contains(&x),
            "on-device uniform produced {x} outside [0,1)"
        );
    }
}

/// PROBE 6 — consecutive on-device UNIFORM calls continue the stream. After a
/// call of n, the manager advances ceil(n/4). The reference is a single CPU
/// PhiloxGenerator stream of 2n: the first call must equal prefix [0..n), the
/// second call must equal [n..2n) i.e. the continued Philox stream (n%4==0 so
/// advance is exactly n/4 counters = n values, blocks are disjoint).
#[test]
fn uniform_gpu_consecutive_calls_continue_philox_stream() {
    ensure_init();
    let n = 4096usize;
    let seed = 77u64;
    let (a, b) = {
        let _g = SEED_LOCK.lock().unwrap();
        manual_seed(seed);
        let a = to_host(&rand_on_device::<f32>(&[n], Device::Cuda(0)).expect("a"));
        let b = to_host(&rand_on_device::<f32>(&[n], Device::Cuda(0)).expect("b"));
        (a, b)
    };
    let full = PhiloxGenerator::new(seed).generate_uniform(2 * n);
    assert_eq!(a, full[..n].to_vec(), "first call != Philox prefix");
    assert_eq!(
        b,
        full[n..].to_vec(),
        "second on-device call does not continue the Philox stream (counter-advance mismatch)"
    );
}

/// PROBE 4 — NORMAL on-device, ODD n: each thread writes idx0=2*gid,
/// idx1=2*gid+1 guarded idx1<n (rng.rs:1199-1205). For odd n exactly n finite
/// values must be written.
#[test]
fn normal_gpu_odd_length_finite_count() {
    ensure_init();
    for &n in &[1usize, 7, 4097] {
        let gpu = {
            let _g = SEED_LOCK.lock().unwrap();
            manual_seed(5);
            to_host(&randn_on_device::<f32>(&[n], Device::Cuda(0)).expect("gpu normal"))
        };
        assert_eq!(gpu.len(), n, "gpu normal wrong length at n={n}");
        for (i, &x) in gpu.iter().enumerate() {
            assert!(
                x.is_finite(),
                "on-device normal value[{i}]={x} not finite at n={n}"
            );
        }
    }
}

/// PROBE 2 — NORMAL distribution moments from the on-device kernel. 1M samples:
/// mean~0, std~1, |skew|~0, kurtosis~3. The .approx transcendentals must not
/// skew the distribution. Targets are the analytic standard-normal moments.
#[test]
fn normal_gpu_moments_standard_normal() {
    ensure_init();
    let n = 1_000_000usize;
    let v = {
        let _g = SEED_LOCK.lock().unwrap();
        manual_seed(13);
        to_host(&randn_on_device::<f32>(&[n], Device::Cuda(0)).expect("gpu normal"))
    };
    assert_eq!(v.len(), n);

    let nf = n as f64;
    let mut s1 = 0.0f64;
    for &x in &v {
        assert!(x.is_finite(), "normal value not finite: {x}");
        s1 += x as f64;
    }
    let mean = s1 / nf;
    let (mut m2, mut m3, mut m4) = (0.0f64, 0.0f64, 0.0f64);
    for &x in &v {
        let d = x as f64 - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= nf;
    m3 /= nf;
    m4 /= nf;
    let std = m2.sqrt();
    let skew = m3 / m2.powf(1.5);
    let kurt = m4 / (m2 * m2); // standard normal == 3.0

    assert!((mean).abs() < 0.01, "on-device normal mean {mean} != ~0");
    assert!((std - 1.0).abs() < 0.01, "on-device normal std {std} != ~1");
    assert!(skew.abs() < 0.05, "on-device normal skew {skew} != ~0");
    assert!(
        (kurt - 3.0).abs() < 0.1,
        "on-device normal kurtosis {kurt} != ~3 (approx-transcendental tail distortion)"
    );
}

/// F64 UNIFORM uses Rust-generated PTX, not CUDA C/NVRTC/libdevice. Exercise
/// odd lengths so both lanes of the two-u32 -> f64 packing and the last-lane
/// guard run through the public backend API.
#[test]
fn uniform_f64_gpu_stays_resident_and_in_unit_interval() {
    ensure_init();
    for &n in &[1usize, 2, 3, 7, 4097] {
        let t = {
            let _g = SEED_LOCK.lock().unwrap();
            manual_seed(2026);
            rand_on_device::<f64>(&[n], Device::Cuda(0)).expect("gpu f64 uniform")
        };
        assert_eq!(t.device(), Device::Cuda(0), "f64 uniform must stay CUDA");
        let v = to_host_f64(&t);
        assert_eq!(v.len(), n, "f64 uniform wrong length at n={n}");
        for (i, &x) in v.iter().enumerate() {
            assert!(
                x.is_finite() && (0.0..1.0).contains(&x),
                "f64 uniform value[{i}]={x} outside [0, 1) at n={n}"
            );
        }
    }
}

/// F64 NORMAL mirrors PyTorch's double Box-Muller structure: two 32-bit Philox
/// lanes per uniform, double log/sqrt/sincos math, two outputs per thread with
/// a guarded odd tail. The math is emitted as Rust-owned PTX.
#[test]
fn normal_f64_gpu_odd_tail_and_moments() {
    ensure_init();
    for &n in &[1usize, 7, 4097] {
        let t = {
            let _g = SEED_LOCK.lock().unwrap();
            manual_seed(41);
            randn_on_device::<f64>(&[n], Device::Cuda(0)).expect("gpu f64 normal")
        };
        assert_eq!(t.device(), Device::Cuda(0), "f64 normal must stay CUDA");
        let v = to_host_f64(&t);
        assert_eq!(v.len(), n, "f64 normal wrong length at n={n}");
        for (i, &x) in v.iter().enumerate() {
            assert!(
                x.is_finite(),
                "f64 normal value[{i}]={x} not finite at n={n}"
            );
        }
    }

    let n = 200_000usize;
    let t = {
        let _g = SEED_LOCK.lock().unwrap();
        manual_seed(43);
        randn_on_device::<f64>(&[n], Device::Cuda(0)).expect("gpu f64 normal moments")
    };
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "f64 normal moments must stay CUDA"
    );
    let v = to_host_f64(&t);
    assert_eq!(v.len(), n);

    let nf = n as f64;
    let mean = v
        .iter()
        .inspect(|&&x| assert!(x.is_finite(), "f64 normal value not finite: {x}"))
        .sum::<f64>()
        / nf;
    let variance = v
        .iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / nf;
    let std = variance.sqrt();

    assert!(mean.abs() < 0.015, "f64 normal mean {mean} != ~0");
    assert!((std - 1.0).abs() < 0.015, "f64 normal std {std} != ~1");
}
