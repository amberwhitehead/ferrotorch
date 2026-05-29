//! Adversarial re-audit of commit 0ead3af79 (#1671).
//!
//! Background: `ADD_VEC4_PTX` (ferrotorch-gpu/src/kernels.rs) contained a
//! non-ASCII multiplication sign (U+00D7) inside a PTX `//` comment line
//! (`4 floats × 4 bytes`). On the WSL CUDA JIT this produced
//! `CUDA_ERROR_INVALID_PTX`, so the vec4 add kernel failed to compile on
//! EVERY call and silently fell back to the scalar `add_kernel`. The commit
//! replaced the `×` with ASCII `x`, so for the FIRST time the
//! `add_vec4_kernel` actually executes on inputs where `n >= 16 && n % 4 == 0`
//! (the dispatch condition in `gpu_add`, kernels.rs:13152).
//!
//! THE KEY RISK this file pins: the vec4 kernel was never numerically
//! exercised before. A wrong v4 load/store offset or wrong per-lane mapping
//! (kernels.rs:297-313 — `shl.b64 %off, %off, 4` byte offset, the two
//! `ld.global.v4.f32` 128-bit loads, the four per-lane `add.f32`, and the
//! `st.global.v4.f32`) would now silently produce wrong sums that the old
//! scalar fallback masked. We verify element-wise correctness against an
//! independently computed host reference, using NON-TRIVIAL distinct values
//! so a misaligned v4 load/store or wrong per-lane offset is observable.
//!
//! Reference oracle: f32 IEEE-754 add. CUDA PTX `add.f32` defaults to
//! round-to-nearest-even, identical to Rust's `f32 + f32`, so finite results
//! are BIT-EXACT — the strongest possible discriminator for a lane bug.
//!
//! All tests require a live CUDA device (RTX 3090 in the audit env).

#![cfg(feature = "cuda")]

use std::sync::Once;

use ferrotorch_gpu::device::GpuDevice;
use ferrotorch_gpu::transfer::{cpu_to_gpu, gpu_to_cpu};

fn ensure_cuda() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend().expect("CUDA backend init");
    });
}

fn device() -> GpuDevice {
    GpuDevice::new(0).expect("GpuDevice::new(0)")
}

/// Non-trivial, distinct per-element inputs. Distinctness is load-bearing:
/// an all-same buffer would pass even with a wrong per-lane offset, because
/// every lane would read the same value. Here `a[i]` and `b[i]` are unique
/// across `i` (and across lanes within a v4 group), so a swapped/misaligned
/// lane produces a detectably wrong sum.
fn make_inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for i in 0..n {
        // Spread magnitudes and signs; avoid all values landing on integers.
        let ai = (i as f32) * 0.5 - (n as f32) * 0.25 + 0.125;
        let bi = ((i % 97) as f32) * 1.0625 - 13.0 + (i as f32) * 1e-3;
        a.push(ai);
        b.push(bi);
    }
    (a, b)
}

/// Independent host reference: plain IEEE-754 f32 add, element-wise.
fn host_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

fn host_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

fn assert_bit_exact(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch got={} want={}",
        actual.len(),
        expected.len()
    );
    for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        if want.is_nan() {
            assert!(got.is_nan(), "{label}[{i}]: want NaN, got {got}");
            continue;
        }
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{label}[{i}]: GPU={got} (bits {:#010x}) != host-ref={want} (bits {:#010x}); \
             vec4 lane-offset / load-store divergence",
            got.to_bits(),
            want.to_bits()
        );
    }
}

fn gpu_add_roundtrip(a: &[f32], b: &[f32], dev: &GpuDevice) -> Vec<f32> {
    let ga = cpu_to_gpu(a, dev).expect("upload a");
    let gb = cpu_to_gpu(b, dev).expect("upload b");
    let out = ferrotorch_gpu::kernels::gpu_add(&ga, &gb, dev).expect("gpu_add");
    gpu_to_cpu(&out, dev).expect("download")
}

fn gpu_sub_roundtrip(a: &[f32], b: &[f32], dev: &GpuDevice) -> Vec<f32> {
    let ga = cpu_to_gpu(a, dev).expect("upload a");
    let gb = cpu_to_gpu(b, dev).expect("upload b");
    let out = ferrotorch_gpu::kernels::gpu_sub(&ga, &gb, dev).expect("gpu_sub");
    gpu_to_cpu(&out, dev).expect("download")
}

// ---------------------------------------------------------------------------
// 1. VEC4 ADD CORRECTNESS — sizes that EXERCISE the vec4 path
//    (n >= 16 && n % 4 == 0; gpu_add dispatch at kernels.rs:13152)
// ---------------------------------------------------------------------------

#[test]
fn vec4_add_correct_across_sizes() {
    ensure_cuda();
    let dev = device();
    // Exactly 16 (boundary of n>=16), small multi-of-4, multi-block sizes,
    // and the 1_000_000 bench size. All satisfy n>=16 && n%4==0 → vec4 path.
    let sizes = [16, 20, 32, 100, 256, 1024, 4096, 65_536, 999_996, 1_000_000];
    for &n in &sizes {
        assert!(n >= 16 && n % 4 == 0, "size {n} must hit vec4 path");
        let (a, b) = make_inputs(n);
        let got = gpu_add_roundtrip(&a, &b, &dev);
        let want = host_add(&a, &b);
        assert_bit_exact(&got, &want, &format!("vec4 add n={n}"));
    }
}

// ---------------------------------------------------------------------------
// 2. TAIL / NON-VEC4 sizes — scalar fallback path (regression)
//    n not divisible by 4, and n < 16.
// ---------------------------------------------------------------------------

#[test]
fn scalar_add_correct_non_vec4_sizes() {
    ensure_cuda();
    let dev = device();
    // n % 4 != 0  → scalar path; n < 16 → scalar path.
    let sizes = [1, 2, 3, 7, 15, 17, 19, 1001, 4095, 999_999];
    for &n in &sizes {
        let takes_vec4 = n >= 16 && n % 4 == 0;
        assert!(!takes_vec4, "size {n} should NOT hit vec4 path");
        let (a, b) = make_inputs(n);
        let got = gpu_add_roundtrip(&a, &b, &dev);
        let want = host_add(&a, &b);
        assert_bit_exact(&got, &want, &format!("scalar add n={n}"));
    }
}

// ---------------------------------------------------------------------------
// 3. SUB correctness for vec4-sized AND tail inputs.
//    (gpu_sub uses SUB_PTX scalar kernel — kernels.rs:13182 — but the audit
//    brief flags sub as historically routed through the add path; we pin
//    element-wise correctness regardless of which kernel runs.)
// ---------------------------------------------------------------------------

#[test]
fn sub_correct_vec4_and_tail_sizes() {
    ensure_cuda();
    let dev = device();
    let sizes = [16, 20, 1024, 4096, 1_000_000, 17, 1001, 999_999, 7];
    for &n in &sizes {
        let (a, b) = make_inputs(n);
        let got = gpu_sub_roundtrip(&a, &b, &dev);
        let want = host_sub(&a, &b);
        assert_bit_exact(&got, &want, &format!("sub n={n}"));
    }
}

// ---------------------------------------------------------------------------
// 4. NO-RECOMPILE / repeated-call stability.
//    The fix's premise: add_vec4_kernel now compiles ONCE (cached). Pin
//    correctness across many repeated calls (a per-call recompile-and-fail
//    regression would either error or silently fall back — both produce the
//    same correct sum, so correctness alone can't catch it, but a coarse
//    timing guard can: the second call must not be ~100x slower than steady
//    state. We assert correctness on every iteration and a loose timing
//    bound on the amortized per-call cost as a coarse class guard.)
// ---------------------------------------------------------------------------

#[test]
fn repeated_add_stable_and_no_percall_recompile() {
    ensure_cuda();
    let dev = device();
    let n = 1_000_000;
    let (a, b) = make_inputs(n);
    let want = host_add(&a, &b);

    let ga = cpu_to_gpu(&a, &dev).expect("upload a");
    let gb = cpu_to_gpu(&b, &dev).expect("upload b");

    // Warm-up (first call may JIT-compile + cache).
    let warm = ferrotorch_gpu::kernels::gpu_add(&ga, &gb, &dev).expect("warm add");
    let warm_h = gpu_to_cpu(&warm, &dev).expect("download warm");
    assert_bit_exact(&warm_h, &want, "repeated add warm");

    // Steady-state: 200 calls. If the kernel recompiled per-call (the #1671
    // bug), this loop would be dominated by ~1.8ms JIT-fail overhead each.
    let iters = 200u32;
    let t0 = std::time::Instant::now();
    for k in 0..iters {
        let out = ferrotorch_gpu::kernels::gpu_add(&ga, &gb, &dev).expect("steady add");
        if k == iters - 1 {
            let h = gpu_to_cpu(&out, &dev).expect("download steady");
            assert_bit_exact(&h, &want, "repeated add steady last");
        }
    }
    let elapsed = t0.elapsed();
    let per_call_us = elapsed.as_secs_f64() * 1e6 / f64::from(iters);
    // Coarse class guard: pre-fix per-call cost was ~1812us. Steady-state
    // (cached) is single-digit us; even with download/launch overhead it is
    // well under 1ms. A 1ms ceiling cannot be hit unless per-call recompile
    // returned. Generous to absorb host scheduling jitter on WSL.
    assert!(
        per_call_us < 1000.0,
        "per-call gpu_add amortized {per_call_us:.1}us exceeds 1000us ceiling — \
         suggests per-call PTX recompile (the #1671 regression) returned"
    );
}

// ---------------------------------------------------------------------------
// 5. NON-ASCII PTX REGRESSION GUARD (the bug-CLASS guard).
//    Scan every `*_PTX` string-literal const body in ferrotorch-gpu/src/*.rs
//    and FAIL if any byte is non-ASCII. The #1671 bug was a non-ASCII char
//    inside a PTX `//` comment line; this guard prevents the whole class
//    (×, →, ±, em-dash, etc. in any future kernel literal) from recurring.
//
//    Only the LITERAL BODY is scanned (the chars CUDA's JIT sees), not the
//    surrounding Rust doc/prose comments which legitimately use Unicode.
// ---------------------------------------------------------------------------

/// Extract the body of every `const <NAME>_PTX: &str = "<body>";` literal in
/// `src`. Handles the multi-line `= "\ ... ";` form used by hand-written PTX
/// kernels. Macro-generated PTX (`= some_macro!(...)`) takes ASCII macro
/// inputs and is covered by scanning those macro-input string literals via
/// the same `"..."` walk below.
fn ptx_literal_bodies(src: &str) -> Vec<(usize, String)> {
    let bytes = src.as_bytes();
    let mut out = Vec::new();
    // Find each `_PTX` token, then locate the next `"` and capture until the
    // matching closing `"` (respecting `\"` escapes). This catches both
    //   const X_PTX: &str = "....";
    // and macro-call string args on the same/next lines after `X_PTX`.
    let mut search_from = 0usize;
    while let Some(rel) = src[search_from..].find("_PTX") {
        let tok = search_from + rel;
        // Bound the literal search to the next ~; or 4000 chars after the
        // token so we don't run away into unrelated code; PTX literals are
        // large but bounded.
        let scan_end = (tok + 200_000).min(src.len());
        // Find opening quote after the token.
        if let Some(qrel) = src[tok..scan_end].find('"') {
            let qstart = tok + qrel + 1;
            // Walk to closing quote, honoring backslash escapes.
            let mut i = qstart;
            let mut line = src[..qstart].bytes().filter(|&c| c == b'\n').count() + 1;
            let mut body = String::new();
            while i < bytes.len() {
                let c = bytes[i];
                if c == b'\\' {
                    // Skip the escaped byte.
                    if i + 1 < bytes.len() {
                        body.push('\\');
                        body.push(bytes[i + 1] as char);
                        i += 2;
                        continue;
                    }
                }
                if c == b'"' {
                    break;
                }
                if c == b'\n' {
                    line += 1;
                }
                // Push raw byte position content as-is (may be multi-byte UTF-8).
                body.push(c as char);
                i += 1;
            }
            // Re-extract the exact UTF-8 slice (the byte-cast above mangles
            // multi-byte chars, but we only need byte-level ASCII detection,
            // so scan the raw byte slice directly).
            let raw = &src[qstart..i.min(src.len())];
            out.push((line, raw.to_string()));
            search_from = i.max(tok + 4);
        } else {
            search_from = tok + 4;
        }
    }
    out
}

#[test]
fn all_ptx_literals_are_pure_ascii() {
    let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut violations: Vec<String> = Vec::new();
    let mut files_scanned = 0usize;
    let mut literals_scanned = 0usize;

    let entries = std::fs::read_dir(&src_dir).expect("read src dir");
    for entry in entries {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("read src file");
        files_scanned += 1;
        for (line, body) in ptx_literal_bodies(&src) {
            literals_scanned += 1;
            for (off, b) in body.bytes().enumerate() {
                if !b.is_ascii() {
                    violations.push(format!(
                        "{}: PTX literal starting ~line {} has non-ASCII byte {:#04x} at offset {} \
                         in literal body (CUDA JIT sees this → CUDA_ERROR_INVALID_PTX, the #1671 bug class)",
                        path.file_name().unwrap().to_string_lossy(),
                        line,
                        b,
                        off
                    ));
                    break;
                }
            }
        }
    }

    assert!(files_scanned > 0, "scanned zero src files — path wrong?");
    assert!(
        literals_scanned > 0,
        "found zero PTX literals to scan — extractor broken?"
    );
    assert!(
        violations.is_empty(),
        "non-ASCII bytes inside PTX string literal(s):\n{}",
        violations.join("\n")
    );
}

/// Self-check the extractor against a synthetic source containing the exact
/// #1671 defect, so the guard above cannot silently pass due to a broken
/// extractor. (R-CHAR-3: expected value is the U+00D7 byte sequence, named.)
#[test]
fn ptx_extractor_catches_seeded_non_ascii() {
    // 0xC3 0x97 is the UTF-8 encoding of U+00D7 MULTIPLICATION SIGN — the
    // exact byte pair the #1671 bug introduced in ADD_VEC4_PTX.
    let seeded =
        "pub const BAD_PTX: &str = \"\\\n.version 7.0\n// 4 floats \u{00D7} 4 bytes\nret;\n\";\n";
    let bodies = ptx_literal_bodies(seeded);
    assert!(
        !bodies.is_empty(),
        "extractor found no literal in seeded src"
    );
    let any_non_ascii = bodies.iter().any(|(_, b)| b.bytes().any(|c| !c.is_ascii()));
    assert!(
        any_non_ascii,
        "extractor failed to surface the seeded U+00D7 — guard would give false PASS"
    );
}

// ---------------------------------------------------------------------------
// 6. dtype coverage: f64 add unaffected by the f32 vec4 fix.
// ---------------------------------------------------------------------------

#[test]
fn f64_add_unaffected() {
    ensure_cuda();
    let dev = device();
    let sizes = [16usize, 17, 1024, 1_000_000];
    for &n in &sizes {
        let a: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 7.0 + 1e-9).collect();
        let b: Vec<f64> = (0..n)
            .map(|i| ((i % 53) as f64) * 1.25 - 3.0 + (i as f64) * 1e-6)
            .collect();
        let ga = cpu_to_gpu(&a, &dev).expect("upload a f64");
        let gb = cpu_to_gpu(&b, &dev).expect("upload b f64");
        let out = ferrotorch_gpu::kernels::gpu_add_f64(&ga, &gb, &dev).expect("gpu_add_f64");
        let got = gpu_to_cpu(&out, &dev).expect("download f64");
        for (i, (&g, (&x, &y))) in got.iter().zip(a.iter().zip(b.iter())).enumerate() {
            let want = x + y;
            assert_eq!(
                g.to_bits(),
                want.to_bits(),
                "f64 add n={n}[{i}]: GPU={g} != host-ref={want}"
            );
        }
    }
}
