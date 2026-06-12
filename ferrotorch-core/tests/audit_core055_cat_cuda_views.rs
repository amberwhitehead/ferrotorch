//! Red regression tests for CORE-055 (#1749), CUDA lane — CLASS-V, High.
//!
//! The CUDA `cat` fast path (`src/grad_fns/shape.rs`) passes each raw
//! underlying GPU handle (`gpu_handle()`, which points at storage element 0)
//! plus logical `numel` to `strided_cat`, whose trait signature
//! (`src/gpu_dispatch.rs`) carries NO source strides and NO storage offset —
//! the backend kernel reads contiguous `input[i]` from the start of the
//! underlying allocation. A transposed / narrowed / permuted / offset CUDA
//! view therefore concatenates the WRONG VALUES while returning a successful
//! result. `CatBackward` repeats the raw-handle assumption for a
//! non-contiguous or offset upstream gradient (`strided_split_*` on the raw
//! `grad_output` handle). `cat` also never validates that all inputs share
//! the first tensor's device. Same `gpu_handle()`-drops-view-geometry class
//! as #1657 / #1845 (CORE-151).
//!
//! Oracle (R-ORACLE-1b): live torch 2.11.0+cu130 on cuda:0 (RTX 3090),
//! 2026-06-11:
//!
//! ```python
//! x = torch.arange(24., device='cuda:0').reshape(4,6)
//! torch.cat([x.t(), x.t()], 0).flatten()
//! # [0,6,12,18, 1,7,13,19, 2,8,14,20, 3,9,15,21, 4,10,16,22, 5,11,17,23] * 2
//! v = x.narrow(0,1,2)                          # offset 6
//! torch.cat([v,v],1).flatten()                 # [6..11, 6..11, 12..17, 12..17]
//! n = x.narrow(1,2,3)                          # offset 2, non-contiguous
//! torch.cat([n,n],0).flatten()                 # [2,3,4, 8,9,10, 14,15,16, 20,21,22] * 2
//! y = torch.arange(24., device='cuda:0').reshape(2,3,4); p = y.permute(2,0,1)
//! torch.cat([p,p],0).flatten()
//! # [0,4,8,12,16,20, 1,5,9,13,17,21, 2,6,10,14,18,22, 3,7,11,15,19,23] * 2
//! torch.cat([x, x.cpu()], 0)
//! # RuntimeError: Expected all tensors to be on the same device, ...
//!
//! a = torch.arange(24., device='cuda:0').reshape(4,6).detach().requires_grad_(True)
//! b = (torch.arange(24., device='cuda:0')*0.5).reshape(6,4).detach().requires_grad_(True)
//! out = torch.cat([a.t(), b], 0)
//! out.backward(torch.arange(48., device='cuda:0').reshape(12,4))
//! a.grad.flatten()  # [0,4,8,12,16,20, 1,5,9,13,17,21, 2,6,10,14,18,22, 3,7,11,15,19,23]
//! b.grad.flatten()  # [24..47]   (both on cuda:0)
//!
//! d = torch.arange(8., device='cuda:0').reshape(2,4).detach().requires_grad_(True)
//! e = (torch.arange(8., device='cuda:0')+10.).reshape(2,4).detach().requires_grad_(True)
//! out3 = torch.cat([d, e], 0)
//! out3.backward(torch.arange(16., device='cuda:0').reshape(4,4).t())  # non-contig go
//! d.grad.flatten()  # [0,4,8,12, 1,5,9,13]
//! e.grad.flatten()  # [2,6,10,14, 3,7,11,15]
//! ```

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::cat;
use ferrotorch_core::error::FerrotorchError;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_core::{Device, creation::from_vec};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the CORE-055 regression suite");
    });
}

fn arange(n: usize) -> Vec<f32> {
    (0..n).map(|v| v as f32).collect()
}

fn cuda(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    from_vec(data.to_vec(), shape)
        .expect("construct cpu tensor")
        .to(Device::Cuda(0))
        .expect("cpu->gpu upload")
}

/// CORE-012 (#1706) idiom: a real CUDA leaf is
/// `x.to('cuda').detach().requires_grad_(True)` — `.requires_grad_` AFTER
/// the upload.
fn cuda_leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    cuda(data, shape).requires_grad_(true)
}

/// Device-asserting readback (post-#1890 pattern): the result must actually
/// reside on Cuda(0) — a silent CPU fallback fails here, not in the values.
// cat is pure data movement (no arithmetic): every element round-trips
// bit-exactly, so float equality is the right check throughout this file.
fn assert_cuda_exact(got: &Tensor<f32>, want_shape: &[usize], want: &[f32], label: &str) {
    assert_eq!(
        got.device(),
        Device::Cuda(0),
        "{label}: expected on Cuda(0) but resides on {:?} — silent CPU fallback",
        got.device()
    );
    assert_eq!(got.shape(), want_shape, "{label}: shape");
    let host = got.cpu().expect("D2H readback").data_vec().expect("read");
    assert_eq!(host.len(), want.len(), "{label}: numel");
    for (i, (a, e)) in host.iter().zip(want).enumerate() {
        assert!(
            a == e,
            "{label}[{i}]: got {a}, torch oracle {e} — CUDA cat read the wrong \
             storage elements (CORE-055). full: {host:?}"
        );
    }
}

/// Transpose views: pre-fix `strided_cat` copies the base buffer in
/// ROW-MAJOR order (`[0,1,2,...]`), not the transposed logical order —
/// WRONG VALUES with a successful return.
#[test]
fn cat_of_cuda_transpose_views_matches_torch() {
    ensure_cuda_backend();
    let x = cuda(&arange(24), &[4, 6]);
    let w = x.transpose(0, 1).expect("transpose view"); // (6,4)
    assert!(!w.is_contiguous(), "transpose view must be non-contiguous");
    let out = cat(&[w.clone(), w], 0).expect("cat forward");
    let half: Vec<f32> = vec![
        0.0, 6.0, 12.0, 18.0, 1.0, 7.0, 13.0, 19.0, 2.0, 8.0, 14.0, 20.0, 3.0, 9.0, 15.0, 21.0,
        4.0, 10.0, 16.0, 22.0, 5.0, 11.0, 17.0, 23.0,
    ];
    let want: Vec<f32> = half.iter().chain(half.iter()).copied().collect();
    assert_cuda_exact(&out, &[12, 4], &want, "cuda cat([t,t],0)");
}

/// Row-narrow views (contiguous strides, storage_offset 6): pre-fix the
/// kernel reads from storage element 0 — rows 0..2 instead of rows 1..3.
#[test]
fn cat_of_cuda_narrow_offset_views_matches_torch() {
    ensure_cuda_backend();
    let x = cuda(&arange(24), &[4, 6]);
    let v = x.narrow(0, 1, 2).expect("narrow view"); // (2,6), offset 6
    assert_eq!(v.storage_offset(), 6, "narrow view must carry its offset");
    assert!(v.is_contiguous(), "row-narrow keeps contiguous strides");
    let out = cat(&[v.clone(), v], 1).expect("cat forward");
    let want: Vec<f32> = vec![
        6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
    ];
    assert_cuda_exact(&out, &[2, 12], &want, "cuda cat([narrow,narrow],1)");
}

/// Column-narrow views (offset 2, non-contiguous strides).
#[test]
fn cat_of_cuda_column_narrow_views_matches_torch() {
    ensure_cuda_backend();
    let x = cuda(&arange(24), &[4, 6]);
    let n = x.narrow(1, 2, 3).expect("column narrow"); // (4,3), offset 2
    assert!(!n.is_contiguous(), "column narrow must be non-contiguous");
    let out = cat(&[n.clone(), n], 0).expect("cat forward");
    let half: Vec<f32> = vec![
        2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 14.0, 15.0, 16.0, 20.0, 21.0, 22.0,
    ];
    let want: Vec<f32> = half.iter().chain(half.iter()).copied().collect();
    assert_cuda_exact(&out, &[8, 3], &want, "cuda cat([ncol,ncol],0)");
}

/// 3-D permute views.
#[test]
fn cat_of_cuda_permute_views_matches_torch() {
    ensure_cuda_backend();
    let y = cuda(&arange(24), &[2, 3, 4]);
    let p = y.permute(&[2, 0, 1]).expect("permute view"); // (4,2,3)
    assert!(!p.is_contiguous(), "permute view must be non-contiguous");
    let out = cat(&[p.clone(), p], 0).expect("cat forward");
    let half: Vec<f32> = vec![
        0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 2.0, 6.0, 10.0, 14.0,
        18.0, 22.0, 3.0, 7.0, 11.0, 15.0, 19.0, 23.0,
    ];
    let want: Vec<f32> = half.iter().chain(half.iter()).copied().collect();
    assert_cuda_exact(&out, &[8, 2, 3], &want, "cuda cat([perm,perm],0)");
}

/// Mixed devices must be rejected up front with a structured
/// `DeviceMismatch` — torch: "Expected all tensors to be on the same
/// device". Pre-fix: no validation; the failure mode depends on which
/// path happens to choke on the foreign tensor.
#[test]
fn cat_rejects_mixed_devices_both_orders() {
    ensure_cuda_backend();
    let g = cuda(&arange(8), &[2, 4]);
    let c = from_vec(arange(8), &[2, 4]).expect("cpu tensor");

    for (label, pair) in [
        ("cat([cuda, cpu])", [g.clone(), c.clone()]),
        ("cat([cpu, cuda])", [c, g]),
    ] {
        let err = cat(&pair, 0).expect_err(&format!(
            "{label} must be rejected (torch: same-device requirement)"
        ));
        assert!(
            matches!(err, FerrotorchError::DeviceMismatch { .. }),
            "{label}: expected DeviceMismatch, got {err:?} (CORE-055: no up-front \
             device validation)"
        );
    }
}

/// Backward over CUDA transpose views of tracking leaves: grads must reach
/// the ORIGINAL leaves with torch's values, ON Cuda(0) (R-ORACLE-3 grad
/// flow + device assertion). Pre-fix the forward already concatenates the
/// wrong values, so the whole chain is wrong.
#[test]
fn cat_cuda_backward_over_transpose_view_reaches_original_leaf() {
    ensure_cuda_backend();
    let a = cuda_leaf(&arange(24), &[4, 6]);
    let b_data: Vec<f32> = (0..24).map(|v| v as f32 * 0.5).collect();
    let b = cuda_leaf(&b_data, &[6, 4]);
    let at = a.transpose(0, 1).expect("transpose view of CUDA leaf"); // (6,4)
    let out = cat(&[at, b.clone()], 0).expect("cat forward");
    assert_eq!(out.shape(), &[12, 4]);

    let go = cuda(&arange(48), &[12, 4]);
    out.backward_with_gradient(&go).expect("backward");

    let ga = a
        .grad()
        .unwrap()
        .expect("grad must reach the ORIGINAL transposed CUDA leaf");
    assert_cuda_exact(
        &ga,
        &[4, 6],
        &[
            0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 2.0, 6.0, 10.0, 14.0,
            18.0, 22.0, 3.0, 7.0, 11.0, 15.0, 19.0, 23.0,
        ],
        "cuda a.grad",
    );
    let gb = b.grad().unwrap().expect("grad must reach CUDA leaf b");
    let want_b: Vec<f32> = (24..48).map(|v| v as f32).collect();
    assert_cuda_exact(&gb, &[6, 4], &want_b, "cuda b.grad");
}

/// `CatBackward` fed a NON-contiguous CUDA `grad_output` (transpose view):
/// pre-fix `strided_split_*` reads the raw base handle in row-major order —
/// silently WRONG grad values.
#[test]
fn cat_cuda_backward_with_noncontiguous_grad_output_matches_torch() {
    ensure_cuda_backend();
    let d = cuda_leaf(&arange(8), &[2, 4]);
    let e_data: Vec<f32> = (0..8).map(|v| v as f32 + 10.0).collect();
    let e = cuda_leaf(&e_data, &[2, 4]);
    let out = cat(&[d.clone(), e.clone()], 0).expect("cat forward");
    assert_eq!(out.shape(), &[4, 4]);

    let go_base = cuda(&arange(16), &[4, 4]);
    let go_t = go_base.transpose(0, 1).expect("transposed grad_output");
    assert!(!go_t.is_contiguous());
    out.backward_with_gradient(&go_t)
        .expect("backward with a non-contiguous CUDA grad_output");

    let gd = d.grad().unwrap().expect("grad must reach CUDA leaf d");
    assert_cuda_exact(
        &gd,
        &[2, 4],
        &[0.0, 4.0, 8.0, 12.0, 1.0, 5.0, 9.0, 13.0],
        "cuda d.grad",
    );
    let ge = e.grad().unwrap().expect("grad must reach CUDA leaf e");
    assert_cuda_exact(
        &ge,
        &[2, 4],
        &[2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0],
        "cuda e.grad",
    );
}
