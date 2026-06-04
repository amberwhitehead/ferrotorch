//! ADVERSARIAL RE-AUDIT of commit ef58b851d (#1536/#1545): fft f16/bf16 reject.
//!
//! VERDICT: NO DIVERGENCE FOUND. This is a PASSING regression-guard that pins
//! the verified-correct behavior of `ferrotorch-core/src/fft.rs` against LIVE
//! torch 2.11.0+cu130 (run with `LD_LIBRARY_PATH="$HOME/.local/lib:..."`).
//!
//! ## Oracle (R-CHAR-3) — exact python run, recorded behavior
//!
//! torch source: `aten/src/ATen/native/SpectralOps.cpp:88-90`
//!   `promote_type_fft`: on non-CUDA/XPU/meta devices,
//!   `TORCH_CHECK(type == kFloat || type == kDouble, "Unsupported dtype ", type)`.
//!   `half` (kHalf) is accepted ONLY under `maybe_support_half` (CUDA/XPU/meta);
//!   `bfloat16` (kBFloat16) is absent from the accepted set on ALL devices.
//! torch source: `torch/fft/__init__.py:49` — half FFT is CUDA-only, native chalf.
//!
//! LIVE torch 2.11 sweep over ALL 18 spectral transform entry points
//! (fft ifft rfft irfft hfft ihfft fft2 ifft2 rfft2 irfft2 fftn ifftn rfftn
//!  irfftn hfft2 ihfft2 hfftn ihfftn), real input cast to float16 / bfloat16
//! on CPU — EVERY one raised, uniformly:
//!   * float16  -> `RuntimeError: Unsupported dtype Half`
//!   * bfloat16 -> `RuntimeError: Unsupported dtype BFloat16`
//!
//! There is NO CPU half upcast path. (Verified 2026-05-29.)
//!
//! LIVE torch 2.11, the two NON-transform helpers (pure axis rolls) ACCEPT half:
//!   >>> torch.fft.fftshift(torch.arange(8).to(torch.float16))
//! > > > tensor([4,5,6,7,0,1,2,3], dtype=torch.float16)        # OK, same dtype
//!   >>> torch.fft.ifftshift(...float16/bfloat16...)        # OK, same dtype
//!
//! LIVE torch 2.11, f32 forward FFT is unregressed by the reject:
//!   >>> torch.fft.fft(torch.tensor([1.,2.,3.,4.]))
//! > > > [(10+0j), (-2+2j), (-2+0j), (-2-2j)]
//!
//! ferrotorch represents complex tensors as interleaved real (trailing dim == 2)
//! over the SAME element type (complex64 == Tensor<f32>, complex128 == Tensor<f64>),
//! so the `reject_half_cpu_fft::<T>` guard (which inspects `<T as Element>::dtype()`)
//! can never accidentally fire on a complex64/complex128 input — confirmed by the
//! f32 round-trip below (a complex c2c transform that succeeds).
//!
//! ## Coverage of the fix
//!
//! All 18 public `*_norm` entry points call `reject_half_cpu_fft::<T>(op)` at the
//! top (verified by reading ef58b851d), and every short-form public wrapper
//! (`fft`, `fft2`, `hfft2`, ...) delegates to its `*_norm` sibling, so the guard
//! is reached on every public surface. fftshift/ifftshift deliberately do NOT
//! call the guard. This test exercises all 18 transforms for BOTH half dtypes
//! plus the two permissive helpers and the f32 happy path.

use ferrotorch_core::fft::{
    fft, fft2, fftn, fftshift, hfft, hfft2, hfftn, ifft, ifft2, ifftn, ifftshift, ihfft, ihfft2,
    ihfftn, irfft, irfft2, irfftn, rfft, rfft2, rfftn,
};
use ferrotorch_core::{Tensor, TensorStorage};
use half::{bf16, f16};

/// Build a CPU f16 tensor from f32 values.
fn t_f16(vals: &[f32], shape: &[usize]) -> Tensor<f16> {
    let data: Vec<f16> = vals.iter().map(|v| f16::from_f32(*v)).collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false).unwrap()
}

/// Build a CPU bf16 tensor from f32 values.
fn t_bf16(vals: &[f32], shape: &[usize]) -> Tensor<bf16> {
    let data: Vec<bf16> = vals.iter().map(|v| bf16::from_f32(*v)).collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false).unwrap()
}

/// Build a CPU f32 tensor.
fn t_f32(vals: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(vals.to_vec()), shape.to_vec(), false).unwrap()
}

fn assert_unsupported(label: &str, err: &ferrotorch_core::FerrotorchError) {
    let msg = format!("{err}");
    assert!(
        msg.contains("Unsupported dtype"),
        "{label}: torch raises `RuntimeError: Unsupported dtype ...` on CPU half FFT; \
         ferrotorch error must mirror that (got: {msg})"
    );
}

/// All 18 spectral transforms reject f16 on CPU — uniform with live torch 2.11.
#[test]
fn all_transforms_reject_f16_cpu() {
    // complex (trailing dim == 2) inputs for c2c / c2r transforms:
    //   [4, 2] => 4 complex samples; [4, 4, 2] grid for the 2-D/N-D variants.
    let c1 = || t_f16(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0], &[4, 2]);
    let cgrid: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let c2 = || t_f16(&cgrid, &[4, 4, 2]);
    // real inputs for r2c transforms:
    let r1 = || t_f16(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let rvals: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let r2 = || t_f16(&rvals, &[4, 4]);

    // c2c / c2r (complex input)
    assert_unsupported("fft", &fft(&c1(), None).unwrap_err());
    assert_unsupported("ifft", &ifft(&c1(), None).unwrap_err());
    assert_unsupported("irfft", &irfft(&c1(), None).unwrap_err());
    assert_unsupported("hfft", &hfft(&c1(), None).unwrap_err());
    assert_unsupported("fft2", &fft2(&c2()).unwrap_err());
    assert_unsupported("ifft2", &ifft2(&c2()).unwrap_err());
    assert_unsupported("fftn", &fftn(&c2(), None, None).unwrap_err());
    assert_unsupported("ifftn", &ifftn(&c2(), None, None).unwrap_err());
    assert_unsupported("irfft2", &irfft2(&c2(), None, None).unwrap_err());
    assert_unsupported("irfftn", &irfftn(&c2(), None, None).unwrap_err());
    assert_unsupported("hfft2", &hfft2(&c2(), None, None).unwrap_err());
    assert_unsupported("hfftn", &hfftn(&c2(), None, None).unwrap_err());

    // r2c (real input)
    assert_unsupported("rfft", &rfft(&r1(), None).unwrap_err());
    assert_unsupported("ihfft", &ihfft(&r1(), None).unwrap_err());
    assert_unsupported("rfft2", &rfft2(&r2(), None, None).unwrap_err());
    assert_unsupported("rfftn", &rfftn(&r2(), None, None).unwrap_err());
    assert_unsupported("ihfft2", &ihfft2(&r2(), None, None).unwrap_err());
    assert_unsupported("ihfftn", &ihfftn(&r2(), None, None).unwrap_err());
}

/// All 18 spectral transforms reject bf16 on CPU — uniform with live torch 2.11.
#[test]
fn all_transforms_reject_bf16_cpu() {
    let c1 = || t_bf16(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0], &[4, 2]);
    let cgrid: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let c2 = || t_bf16(&cgrid, &[4, 4, 2]);
    let r1 = || t_bf16(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let rvals: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let r2 = || t_bf16(&rvals, &[4, 4]);

    assert_unsupported("fft", &fft(&c1(), None).unwrap_err());
    assert_unsupported("ifft", &ifft(&c1(), None).unwrap_err());
    assert_unsupported("irfft", &irfft(&c1(), None).unwrap_err());
    assert_unsupported("hfft", &hfft(&c1(), None).unwrap_err());
    assert_unsupported("fft2", &fft2(&c2()).unwrap_err());
    assert_unsupported("ifft2", &ifft2(&c2()).unwrap_err());
    assert_unsupported("fftn", &fftn(&c2(), None, None).unwrap_err());
    assert_unsupported("ifftn", &ifftn(&c2(), None, None).unwrap_err());
    assert_unsupported("irfft2", &irfft2(&c2(), None, None).unwrap_err());
    assert_unsupported("irfftn", &irfftn(&c2(), None, None).unwrap_err());
    assert_unsupported("hfft2", &hfft2(&c2(), None, None).unwrap_err());
    assert_unsupported("hfftn", &hfftn(&c2(), None, None).unwrap_err());

    assert_unsupported("rfft", &rfft(&r1(), None).unwrap_err());
    assert_unsupported("ihfft", &ihfft(&r1(), None).unwrap_err());
    assert_unsupported("rfft2", &rfft2(&r2(), None, None).unwrap_err());
    assert_unsupported("rfftn", &rfftn(&r2(), None, None).unwrap_err());
    assert_unsupported("ihfft2", &ihfft2(&r2(), None, None).unwrap_err());
    assert_unsupported("ihfftn", &ihfftn(&r2(), None, None).unwrap_err());
}

/// fftshift / ifftshift stay dtype-permissive for half — matches live torch
/// (a pure axis roll, NOT a spectral transform). The reject guard must NOT
/// catch them.
#[test]
fn fftshift_ifftshift_stay_permissive_for_half() {
    // torch.fft.fftshift(arange(8).half()) == [4,5,6,7,0,1,2,3] (same dtype).
    let xf16 = t_f16(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[8]);
    let shifted = fftshift(&xf16, None).expect("fftshift(f16) must succeed like torch");
    let got: Vec<f32> = shifted.data().unwrap().iter().map(|v| v.to_f32()).collect();
    assert_eq!(got, vec![4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0]);

    // ifftshift is the inverse roll; round-trip must restore the original.
    let back = ifftshift(&shifted, None).expect("ifftshift(f16) must succeed like torch");
    let got_back: Vec<f32> = back.data().unwrap().iter().map(|v| v.to_f32()).collect();
    assert_eq!(got_back, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

    // Same for bf16.
    let xbf16 = t_bf16(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[8]);
    let s2 = fftshift(&xbf16, None).expect("fftshift(bf16) must succeed like torch");
    let g2: Vec<f32> = s2.data().unwrap().iter().map(|v| v.to_f32()).collect();
    assert_eq!(g2, vec![4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0]);
}

/// The reject did NOT break the normal f32 path. complex64 (== interleaved f32,
/// trailing dim == 2) input runs the c2c transform and matches live torch:
///   torch.fft.fft([1,2,3,4]) == [(10+0j),(-2+2j),(-2+0j),(-2-2j)]
/// This also proves the guard never fires on a complex64 input (it is Tensor<f32>).
#[test]
fn f32_fft_roundtrip_unregressed_matches_torch() {
    // Real signal [1,2,3,4] expressed as interleaved complex [4, 2] (imag = 0).
    let input = t_f32(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0], &[4, 2]);
    let out = fft(&input, None).expect("f32 fft must still succeed");
    let d = out.data().unwrap();
    // Layout: [re0, im0, re1, im1, ...].
    let pairs: Vec<(f32, f32)> = d.chunks(2).map(|c| (c[0], c[1])).collect();
    let expect = [(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)];
    for (i, (e_re, e_im)) in expect.iter().enumerate() {
        assert!(
            (pairs[i].0 - e_re).abs() < 1e-4 && (pairs[i].1 - e_im).abs() < 1e-4,
            "fft(f32) bin {i}: torch={:?} ferrotorch={:?}",
            (e_re, e_im),
            pairs[i]
        );
    }

    // ifft round-trip recovers the original real signal (c2c, also f32 path).
    let rt = ifft(&out, None).expect("f32 ifft must still succeed");
    let rd = rt.data().unwrap();
    let re: Vec<f32> = rd.chunks(2).map(|c| c[0]).collect();
    let expect_re = [1.0, 2.0, 3.0, 4.0];
    for (i, e) in expect_re.iter().enumerate() {
        assert!(
            (re[i] - e).abs() < 1e-4,
            "ifft round-trip bin {i}: {} != {e}",
            re[i]
        );
    }
}
