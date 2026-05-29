//! Fused single-launch `add_scaled` on CUDA — correctness re-audit (#1675).
//!
//! Background: `add_scaled(a, b, alpha)` previously staged two GPU kernels
//! (`scale_tensor(b, alpha)` then `add_inner(a, b_scaled)`) plus a temporary
//! device buffer. Since `sub` -> `sub_scaled(a,b,1)` -> `add_scaled(a,b,-1)`
//! and `rsub` -> `sub_scaled(b,a,alpha)` -> `add_scaled(b,a,-alpha)`, GPU
//! `sub` paid ~2x the launch cost of GPU `add`. The fix routes the
//! SAME-shape, f32/f64, finite-alpha CUDA case through a single fused
//! `out[i] = fma(alpha, b[i], a[i])` kernel (kernels.rs
//! `gpu_add_scaled_f32` / `gpu_add_scaled_f64`).
//!
//! Reference oracle (R-CHAR-3): the host-side IEEE-754 fused multiply-add
//! `a[i].mul_add(?, ?)` is NOT used as the reference here because the prior
//! production path was scale-then-add (two roundings). Instead we anchor on
//! two independently-derived references:
//!
//!   1. For `alpha == -1.0`: torch's `sub_out` delegates to
//!      `add_stub(device_type(), *this, -alpha)` — a fused add-with-negated-
//!      alpha — per `aten/src/ATen/native/BinaryOps.cpp:434` (quoted in
//!      `arithmetic.rs::sub_scaled`). The mathematically exact result is
//!      `a[i] - b[i]`, and `fma(-1, b, a)` is provably exact (no rounding:
//!      `-1 * b` is exact, the single sum is the only rounding, identical to
//!      a plain subtract). We assert BIT-EXACT against host `a[i] - b[i]`.
//!      This is the strongest discriminator and the load-bearing parity
//!      claim for `sub` / `sub_scaled` / `rsub`.
//!
//!   2. For general finite `alpha` (2.5, -0.5, 0.0): the fused FMA does ONE
//!      rounding and is therefore at least as accurate as scale-then-add's
//!      two roundings; the exact real value is `a + alpha*b`. We assert the
//!      GPU fused result matches the host single-rounding `mul_add`
//!      reference BIT-EXACT (CUDA `fma.rn.f32` and Rust `f32::mul_add` both
//!      compute the correctly-rounded FMA), which simultaneously bounds the
//!      error vs the exact value to <= 0.5 ULP.
//!
//! `alpha == 0.0`: `a + 0*b`. With finite `b`, `0*b == 0` so the result is
//! `a` exactly; `fma(0, b, a) == a`. (The `0*inf -> NaN` torch edge is
//! covered by the broadcast/scalar fall-through, not this fused path.)
//!
//! Requires a live CUDA device (RTX 3090 in the audit env).

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::grad_fns::arithmetic;
use ferrotorch_core::tensor::Tensor;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the #1675 fused add_scaled audit");
    });
}

/// Non-trivial, distinct per-element inputs. Distinctness is load-bearing: an
/// all-same buffer would pass even with a swapped vec4 lane offset.
fn make_inputs_f32(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for i in 0..n {
        a.push((i as f32) * 0.5 - (n as f32) * 0.25 + 0.125);
        b.push(((i % 97) as f32) * 1.0625 - 13.0 + (i as f32) * 1e-3);
    }
    (a, b)
}

fn make_inputs_f64(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for i in 0..n {
        a.push((i as f64) * 0.5 - (n as f64) * 0.25 + 0.125);
        b.push(((i % 97) as f64) * 1.0625 - 13.0 + (i as f64) * 1e-3);
    }
    (a, b)
}

fn cuda_f32(v: &[f32]) -> Tensor<f32> {
    from_vec::<f32>(v.to_vec(), &[v.len()])
        .expect("from_vec f32")
        .to(Device::Cuda(0))
        .expect("to cuda")
}

fn cuda_f64(v: &[f64]) -> Tensor<f64> {
    from_vec::<f64>(v.to_vec(), &[v.len()])
        .expect("from_vec f64")
        .to(Device::Cuda(0))
        .expect("to cuda")
}

fn read_f32(t: &Tensor<f32>) -> Vec<f32> {
    let cpu = if t.is_cuda() {
        t.cpu().expect("d2h")
    } else {
        t.clone()
    };
    cpu.data().expect("read").to_vec()
}

fn read_f64(t: &Tensor<f64>) -> Vec<f64> {
    let cpu = if t.is_cuda() {
        t.cpu().expect("d2h")
    } else {
        t.clone()
    };
    cpu.data().expect("read").to_vec()
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

// ---------------------------------------------------------------------------
// 1. sub / sub_scaled / rsub: alpha=-1 fused path is BIT-EXACT a - b.
//    Cover both the vec4 path (n>=16 && n%4==0) and the scalar/tail path.
// ---------------------------------------------------------------------------

#[test]
fn sub_fused_bit_exact_a_minus_b_f32_all_sizes() {
    ensure_cuda_backend();
    // 4096 exercises vec4; 17/19 exercise the scalar tail; 8 is scalar.
    for &n in &[8usize, 16, 17, 19, 256, 4096] {
        let (a, b) = make_inputs_f32(n);
        let ga = cuda_f32(&a);
        let gb = cuda_f32(&b);

        let out = arithmetic::sub(&ga, &gb).expect("gpu sub");
        assert!(out.is_cuda(), "sub n={n}: result must stay is_cuda()");

        // Reference: host IEEE-754 subtraction (== exact for the fused
        // fma(-1,b,a) which has the single sum as its only rounding).
        let want: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x - y).collect();
        assert_bits_f32(&read_f32(&out), &want, &format!("sub f32 n={n}"));
    }
}

#[test]
fn sub_scaled_and_rsub_fused_match_cpu_f32() {
    ensure_cuda_backend();
    let n = 1024;
    let (a, b) = make_inputs_f32(n);

    // sub_scaled(a, b, alpha) == a - alpha*b ; rsub(a, b, alpha) == b - alpha*a.
    for &alpha in &[1.0f64, 2.5, -0.5] {
        // GPU
        let g_sub = arithmetic::sub_scaled(&cuda_f32(&a), &cuda_f32(&b), alpha).expect("gpu subs");
        let g_rsub = arithmetic::rsub(&cuda_f32(&a), &cuda_f32(&b), alpha).expect("gpu rsub");
        assert!(g_sub.is_cuda() && g_rsub.is_cuda());

        // CPU reference via the SAME production function (torch-parity-verified
        // on CPU; serves as the independent oracle here).
        let c_sub = arithmetic::sub_scaled(
            &from_vec::<f32>(a.clone(), &[n]).unwrap(),
            &from_vec::<f32>(b.clone(), &[n]).unwrap(),
            alpha,
        )
        .expect("cpu subs");
        let c_rsub = arithmetic::rsub(
            &from_vec::<f32>(a.clone(), &[n]).unwrap(),
            &from_vec::<f32>(b.clone(), &[n]).unwrap(),
            alpha,
        )
        .expect("cpu rsub");

        // alpha=1 (sub_scaled -> add_scaled(-1)) is exact on both; 2.5/-0.5
        // fused is ONE rounding while CPU scale-then-add is TWO. The fused
        // result is >= as accurate, so compare with a tight ULP tolerance
        // rather than bit-exact for the inexact-alpha cases.
        let (gs, cs) = (read_f32(&g_sub), read_f32(&c_sub));
        let (gr, cr) = (read_f32(&g_rsub), read_f32(&c_rsub));
        for i in 0..n {
            let tol = (cs[i].abs() * 4.0 * f32::EPSILON).max(1e-5);
            assert!(
                (gs[i] - cs[i]).abs() <= tol,
                "sub_scaled alpha={alpha} [{i}]: gpu={} cpu={} (tol {tol})",
                gs[i],
                cs[i]
            );
            let tolr = (cr[i].abs() * 4.0 * f32::EPSILON).max(1e-5);
            assert!(
                (gr[i] - cr[i]).abs() <= tolr,
                "rsub alpha={alpha} [{i}]: gpu={} cpu={} (tol {tolr})",
                gr[i],
                cr[i]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. add_scaled fused: f32 across alphas {-1, 1, 2.5, 0, -0.5}, bit-exact vs
//    host correctly-rounded FMA (CUDA fma.rn.f32 == Rust f32::mul_add).
//    alpha=1 short-circuits to plain add (not the fused kernel) — still
//    checked for parity.
// ---------------------------------------------------------------------------

#[test]
fn add_scaled_fused_matches_host_fma_f32() {
    ensure_cuda_backend();
    let n = 1000; // [1000] like the issue's [1000,1000] flattened; non-vec4 tail (1000%4==0 → vec4 here)
    let (a, b) = make_inputs_f32(n);

    for &alpha in &[-1.0f64, 1.0, 2.5, 0.0, -0.5] {
        let out =
            arithmetic::add_scaled(&cuda_f32(&a), &cuda_f32(&b), alpha).expect("gpu add_scaled");
        assert!(
            out.is_cuda(),
            "add_scaled alpha={alpha}: must stay is_cuda()"
        );
        let got = read_f32(&out);

        let alpha_f = alpha as f32;
        // Host correctly-rounded FMA reference: out = a + alpha*b in one
        // rounding. For alpha=1 the production fast path uses plain add
        // (a + 1*b == a.mul_add(1,? ) bit-identical for finite values).
        let want: Vec<f32> = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| {
                if alpha == 1.0 {
                    x + y
                } else {
                    y.mul_add(alpha_f, x) // fma(alpha, b, a)
                }
            })
            .collect();
        assert_bits_f32(&got, &want, &format!("add_scaled f32 alpha={alpha}"));
    }
}

// ---------------------------------------------------------------------------
// 3. add_scaled / sub fused: f64 across alphas, bit-exact vs host FMA.
// ---------------------------------------------------------------------------

#[test]
fn add_scaled_and_sub_fused_f64() {
    ensure_cuda_backend();
    let n = 512;
    let (a, b) = make_inputs_f64(n);

    // sub: bit-exact a - b.
    let s = arithmetic::sub(&cuda_f64(&a), &cuda_f64(&b)).expect("gpu sub f64");
    assert!(s.is_cuda());
    let want_sub: Vec<f64> = a.iter().zip(&b).map(|(&x, &y)| x - y).collect();
    assert_bits_f64(&read_f64(&s), &want_sub, "sub f64");

    for &alpha in &[-1.0f64, 1.0, 2.5, 0.0, -0.5] {
        let out = arithmetic::add_scaled(&cuda_f64(&a), &cuda_f64(&b), alpha)
            .expect("gpu add_scaled f64");
        assert!(out.is_cuda(), "add_scaled f64 alpha={alpha}: is_cuda()");
        let want: Vec<f64> = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| {
                if alpha == 1.0 {
                    x + y
                } else {
                    y.mul_add(alpha, x)
                }
            })
            .collect();
        assert_bits_f64(
            &read_f64(&out),
            &want,
            &format!("add_scaled f64 alpha={alpha}"),
        );
    }
}

// ---------------------------------------------------------------------------
// 4. NaN / inf alpha: must FALL BACK to scale-then-add (fused path excludes
//    non-finite alpha) and still match the host scale-then-add semantics.
//    For finite b: alpha=inf -> a + inf*b = +-inf (sign of b); alpha=NaN ->
//    NaN. We compare against the host scale-then-add reference (TWO
//    roundings) since that is the fall-through path the fused branch defers
//    to for non-finite alpha.
// ---------------------------------------------------------------------------

#[test]
fn add_scaled_nonfinite_alpha_falls_back_f32() {
    ensure_cuda_backend();
    let n = 64;
    let (a, b) = make_inputs_f32(n);

    for &alpha in &[f32::INFINITY as f64, f64::NAN, f32::NEG_INFINITY as f64] {
        let out = arithmetic::add_scaled(&cuda_f32(&a), &cuda_f32(&b), alpha)
            .expect("gpu add_scaled inf/nan");
        assert!(out.is_cuda(), "non-finite alpha result must stay is_cuda()");
        let got = read_f32(&out);
        let af = alpha as f32;
        // scale-then-add reference (the fall-through path): (alpha*b) + a.
        let want: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| (af * y) + x).collect();
        assert_bits_f32(&got, &want, &format!("add_scaled nonfinite alpha={alpha}"));
    }
}

// ---------------------------------------------------------------------------
// 5. Broadcast add_scaled (different shapes) still works — must NOT take the
//    fused same-shape path; falls back to broadcast scale-then-add.
// ---------------------------------------------------------------------------

#[test]
fn add_scaled_broadcast_falls_back_f32() {
    ensure_cuda_backend();
    // a: [4, 3], b: [3] -> broadcast to [4, 3].
    let a_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.25 - 1.0).collect();
    let b_data: Vec<f32> = vec![0.5, -2.0, 3.25];

    let a = from_vec::<f32>(a_data.clone(), &[4, 3])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let b = from_vec::<f32>(b_data.clone(), &[3])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let alpha = 2.5f64;
    let out = arithmetic::add_scaled(&a, &b, alpha).expect("gpu broadcast add_scaled");
    assert!(out.is_cuda(), "broadcast result must stay is_cuda()");
    assert_eq!(out.shape(), &[4, 3]);

    let got = read_f32(&out);
    let af = alpha as f32;
    // Host broadcast reference (scale-then-add, the fall-through path).
    let mut want = vec![0.0f32; 12];
    for r in 0..4 {
        for c in 0..3 {
            want[r * 3 + c] = (af * b_data[c]) + a_data[r * 3 + c];
        }
    }
    assert_bits_f32(&got, &want, "add_scaled broadcast");
}
