#![cfg(feature = "cuda")]
//! RE-AUDIT regression guard for the #1661 fix (masked_fill storage_offset,
//! two-layer: core `.contiguous()` in grad_fns/indexing.rs + logical-`n`
//! `launch_masked_fill` in ferrotorch-gpu/src/masked_kernels.rs).
//!
//! This file PINS the post-fix contract that the #1661 commit (8be4e4b63)
//! claims, across the matrix the re-audit brief enumerates:
//!
//!   (2) NORMAL offset-0 contiguous CUDA masked_fill at sizes that are NOT a
//!       multiple of the launcher's ROUND_ELEMENTS=256 grid quantum AND that
//!       straddle it (5, 100, 257, 300, 1000) for f32 AND f64 — every element
//!       correct vs torch, no tail under-launch (logical-`n` thread count), no
//!       over-read past logical `n` (the pooled `.contiguous()` backing store is
//!       only required `>= n`).
//!   (3) MASKED_FILL on a narrowed-offset CUDA view for f64 (the pinned suite
//!       already covers f32) -> matches torch.
//!   (4) THE masked_scatter SPILLOVER VERDICT (the key open question).
//!       `launch_scatter` (masked_kernels.rs:711) still validates the RAW
//!       `mask.len() != out_numel`. Empirical findings:
//!         - masked_scatter FORWARD rejects mixed CPU-mask/CUDA-input residency
//!           like live torch. Older ferrotorch builds accidentally round-tripped
//!           through host data and returned a CPU tensor; that path is not
//!           production GPU semantics.
//!         - The ONLY user-facing path into `backend.masked_scatter` ->
//!           `launch_scatter` is `MaskedSelectBackward::backward` (indexing.rs
//!           :979). The saved bool mask there is an EXACT-length resident buffer
//!           (`GpuBufferHandle::len() == input_numel == out_numel`), so the
//!           dispatch check (backend_impl.rs:8315) AND the launcher raw-len check
//!           both pass. Exercised on a narrowed-offset (storage_offset=2) CUDA
//!           input below -> matches torch grad.
//!       VERDICT: masked_scatter SAFE w.r.t. the `launch_scatter` raw-len
//!       spillover — the offset-view case never reaches it (forward host path)
//!       and the masked_select-backward path that does carries an exact-length
//!       mask. `launch_scatter`'s raw-len check is latent dead-ish code for the
//!       offset-view case, NOT a user-facing divergence. (#1660 critic claim
//!       confirmed.)
//!
//!       SEPARATE divergence uncovered during this re-audit (pre-existing,
//!       independent of #1661): masked_scatter forward REJECTS a GPU-resident
//!       mask with `GpuTensorNotAccessible` (`mask_b.data()` at indexing.rs:3620),
//!       whereas torch accepts it. Pinned (ignored, tracking #1662) below.
//!
//! # R-CHAR-3 provenance
//! masked_fill expectations are computed IN-TEST from the upstream elementwise
//! kernel semantic `out[i] = mask[i] ? value : self[i]` — NOT literal-copied
//! from ferrotorch. The narrowed-view / masked_scatter / masked_select-backward
//! expectations are live-torch constants (2.11.0+cu130, RTX 3090, this env)
//! quoted at each use.

use ferrotorch_core::{BoolTensor, Device, FerrotorchError, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = init_cuda_backend();
    });
}

fn cpu_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f32 tensor")
}
fn cpu_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f64 tensor")
}
fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("cpu()").data().unwrap().to_vec()
}
fn host_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.cpu().expect("cpu()").data().unwrap().to_vec()
}

/// Upstream elementwise semantic, applied on the host to build the expectation
/// independently of the ferrotorch GPU kernel (R-CHAR-3 route (b)).
fn torch_masked_fill_f32(input: &[f32], mask: &[bool], value: f32) -> Vec<f32> {
    input
        .iter()
        .zip(mask)
        .map(|(&x, &m)| if m { value } else { x })
        .collect()
}
fn torch_masked_fill_f64(input: &[f64], mask: &[bool], value: f64) -> Vec<f64> {
    input
        .iter()
        .zip(mask)
        .map(|(&x, &m)| if m { value } else { x })
        .collect()
}

// ===========================================================================
// (2) NORMAL offset-0 contiguous masked_fill across sizes + dtypes.
// Sizes straddle ROUND_ELEMENTS=256: below (5,100), exactly+1 (257), above
// non-multiple (300), well above (1000). A logical-`n` launch must touch EVERY
// element (no tail under-launch) and read NOTHING past `n`.
// ===========================================================================

const FILL_SIZES: [usize; 5] = [5, 100, 257, 300, 1000];

#[test]
fn masked_fill_normal_offset0_matrix_f32_matches_torch() {
    ensure_cuda();
    for &n in &FILL_SIZES {
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mask_vec: Vec<bool> = (0..n).map(|i| (i % 3 == 0) || i == n - 1).collect();
        let value = -7.5f32;
        let expect = torch_masked_fill_f32(&data, &mask_vec, value);

        let x = cpu_f32(&data, &[n]).to(Device::Cuda(0)).expect("to cuda");
        assert_eq!(x.storage_offset(), 0, "n={n}");
        let mask = BoolTensor::from_vec(mask_vec, vec![n])
            .expect("mask")
            .to(Device::Cuda(0))
            .expect("mask cuda");
        let got = x.masked_fill(&mask, value).expect("masked_fill");
        assert!(got.is_cuda(), "n={n}: result must stay CUDA");
        assert_eq!(
            host_f32(&got),
            expect,
            "f32 masked_fill n={n}: GPU diverged from torch elementwise semantic"
        );
    }
}

#[test]
fn masked_fill_normal_offset0_matrix_f64_matches_torch() {
    ensure_cuda();
    for &n in &FILL_SIZES {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mask_vec: Vec<bool> = (0..n).map(|i| (i % 3 == 0) || i == n - 1).collect();
        let value = -7.5f64;
        let expect = torch_masked_fill_f64(&data, &mask_vec, value);

        let x = cpu_f64(&data, &[n]).to(Device::Cuda(0)).expect("to cuda");
        assert_eq!(x.storage_offset(), 0, "n={n}");
        let mask = BoolTensor::from_vec(mask_vec, vec![n])
            .expect("mask")
            .to(Device::Cuda(0))
            .expect("mask cuda");
        let got = x.masked_fill(&mask, value).expect("masked_fill");
        assert!(got.is_cuda(), "n={n}: result must stay CUDA");
        assert_eq!(
            host_f64(&got),
            expect,
            "f64 masked_fill n={n}: GPU diverged from torch elementwise semantic"
        );
    }
}

// ===========================================================================
// (3) MASKED_FILL on a narrowed-offset CUDA view — f64 (f32 in pinned suite).
// CUDA [4,2]=[[1..8]], narrow rows 1..4 -> logical [3,2] storage_offset 2.
// live torch 2.11.0+cu130 (RTX 3090):
//   view.masked_fill([[F,F],[T,T],[T,T]], -1.0).flatten() == [3,4,-1,-1,-1,-1]
// ===========================================================================

const TORCH_F64_NARROWED_FILL: [f64; 6] = [3.0, 4.0, -1.0, -1.0, -1.0, -1.0];

#[test]
fn masked_fill_f64_narrowed_offset_view_gpu_matches_torch() {
    ensure_cuda();
    let full = cpu_f64(&[1., 2., 3., 4., 5., 6., 7., 8.], &[4, 2])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let view = full.narrow(0, 1, 3).expect("narrow rows 1..4");
    assert_eq!(view.shape(), &[3, 2]);
    assert!(view.is_contiguous());
    assert_ne!(view.storage_offset(), 0);
    let mask = BoolTensor::from_vec(vec![false, false, true, true, true, true], vec![3, 2])
        .expect("mask")
        .to(Device::Cuda(0))
        .expect("mask cuda");
    let got = view
        .masked_fill(&mask, -1.0)
        .expect("masked_fill f64 narrowed");
    assert!(got.is_cuda());
    assert_eq!(
        host_f64(&got),
        TORCH_F64_NARROWED_FILL.to_vec(),
        "f64 masked_fill narrowed-offset view diverged from live torch"
    );
}

// ===========================================================================
// (4) masked_scatter SPILLOVER verdict.
// ===========================================================================

#[test]
fn masked_scatter_forward_cpumask_cuda_input_rejected_like_torch() {
    ensure_cuda();
    let full = cpu_f32(&[1., 2., 3., 4., 5., 6., 7., 8.], &[4, 2])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let view = full.narrow(0, 1, 3).expect("narrow");
    assert_ne!(view.storage_offset(), 0);
    // Live torch rejects CPU masks with CUDA inputs for masked_scatter.
    let mask =
        BoolTensor::from_vec(vec![false, true, true, false, true, true], vec![3, 2]).expect("mask");
    let src = cpu_f32(&[-1., -2., -3., -4.], &[4])
        .to(Device::Cuda(0))
        .expect("src cuda");
    let err = view
        .masked_scatter_t(&mask, &src)
        .expect_err("CPU mask + CUDA input must be rejected like torch");
    assert!(
        matches!(err, FerrotorchError::DeviceMismatch { .. }),
        "expected DeviceMismatch for CPU mask + CUDA input, got {err:?}"
    );
}

/// The ONLY user-facing path into `backend.masked_scatter` -> `launch_scatter`:
/// `masked_select(narrowed_view).backward()`. The saved mask is exact-length, so
/// the raw-len check in launch_scatter is correct here. live torch 2.11.0+cu130
/// (RTX 3090): base=arange(1..9) leaf, b2=base.reshape(4,2), v=b2[1:4] (offset 2),
///   sel = v.masked_select([[F,F],[T,T],[T,T]]) == [5,6,7,8];
///   (sel * [10,20,30,40]).sum().backward()
///   => base.grad == [0,0,0,0,10,20,30,40]
const TORCH_MS_BACKWARD_GRAD: [f32; 8] = [0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0];

#[test]
fn masked_select_backward_narrowed_offset_view_reaches_scatter_and_matches_torch() {
    ensure_cuda();
    let base = cpu_f32(&[1., 2., 3., 4., 5., 6., 7., 8.], &[8])
        .to(Device::Cuda(0))
        .expect("cuda")
        .requires_grad_(true);
    let b2 = base.reshape_t(&[4, 2]).expect("reshape");
    let v = b2.narrow(0, 1, 3).expect("narrow"); // offset 2
    assert_ne!(v.storage_offset(), 0);
    let mask = BoolTensor::from_vec(vec![false, false, true, true, true, true], vec![3, 2])
        .expect("mask")
        .to(Device::Cuda(0))
        .expect("mask cuda");
    let sel = v.masked_select(&mask).expect("masked_select");
    assert!(sel.is_cuda());
    assert_eq!(host_f32(&sel), vec![5.0, 6.0, 7.0, 8.0]);
    let w = cpu_f32(&[10., 20., 30., 40.], &[4])
        .to(Device::Cuda(0))
        .expect("w cuda");
    let loss = sel.mul_t(&w).expect("mul").sum_all().expect("sum");
    loss.backward().expect(
        "masked_select backward (reaches backend.masked_scatter -> launch_scatter) \
         must not error on a narrowed-offset CUDA input",
    );
    let g = base.grad().expect("grad() ok").expect("base.grad present");
    assert_eq!(
        host_f32(&g),
        TORCH_MS_BACKWARD_GRAD.to_vec(),
        "masked_select backward grad (via launch_scatter) diverged from live torch"
    );
}

// ---------------------------------------------------------------------------
// SEPARATE DIVERGENCE (pre-existing, NOT #1661): masked_scatter forward rejects
// a GPU-resident mask. Upstream `aten/src/ATen/native/TensorAdvancedIndexing.cpp`
// masked_scatter accepts a CUDA mask; ferrotorch errors at indexing.rs:3620
// (`mask_b.data()` -> BoolTensor::data -> try_as_slice -> GpuTensorNotAccessible).
// live torch 2.11.0+cu130 (RTX 3090):
//   torch.tensor([1,2,3,4],cuda).masked_scatter(
//       torch.tensor([F,T,T,F],cuda), torch.tensor([-1,-2],cuda)) == [1,-1,-2,4]
// ferrotorch: Err(GpuTensorNotAccessible). Tracking #1662 (blocker).
// ---------------------------------------------------------------------------

const TORCH_MS_GPUMASK: [f32; 4] = [1.0, -1.0, -2.0, 4.0];

#[test]
fn masked_scatter_forward_gpu_mask_rejected_divergence() {
    ensure_cuda();
    let x = cpu_f32(&[1., 2., 3., 4.], &[4])
        .to(Device::Cuda(0))
        .expect("cuda");
    let mask = BoolTensor::from_vec(vec![false, true, true, false], vec![4])
        .expect("mask")
        .to(Device::Cuda(0))
        .expect("mask cuda");
    let src = cpu_f32(&[-1., -2.], &[2])
        .to(Device::Cuda(0))
        .expect("src cuda");
    let got = x.masked_scatter_t(&mask, &src).expect(
        "masked_scatter with a GPU-resident mask must succeed (torch accepts it); \
         ferrotorch currently errors GpuTensorNotAccessible at indexing.rs:3620",
    );
    assert_eq!(
        host_f32(&got),
        TORCH_MS_GPUMASK.to_vec(),
        "masked_scatter (GPU mask) diverged from live torch"
    );
}

// ---------------------------------------------------------------------------
// #1662 NEW COVERAGE: on-device masked_scatter forward (input + mask + source
// all CUDA) across mask patterns + dtypes. Result must stay is_cuda() (NO host
// round trip) and match live torch / the upstream
// `out[i] = mask[i] ? source[exclusive_prefix_sum(mask)[i]] : input[i]`
// semantic (`aten/src/ATen/native/cuda/IndexKernel.cu:447-453`). R-CHAR-3:
// expectations built in-test from that elementwise semantic (route (b)).
// ---------------------------------------------------------------------------

/// Build the torch masked_scatter expectation from the upstream semantic.
fn torch_masked_scatter_f32(input: &[f32], mask: &[bool], source: &[f32]) -> Vec<f32> {
    let mut out = input.to_vec();
    let mut j = 0usize;
    for (i, &m) in mask.iter().enumerate() {
        if m {
            out[i] = source[j];
            j += 1;
        }
    }
    out
}
fn torch_masked_scatter_f64(input: &[f64], mask: &[bool], source: &[f64]) -> Vec<f64> {
    let mut out = input.to_vec();
    let mut j = 0usize;
    for (i, &m) in mask.iter().enumerate() {
        if m {
            out[i] = source[j];
            j += 1;
        }
    }
    out
}

#[test]
fn masked_scatter_forward_all_cuda_patterns_f32_matches_torch() {
    ensure_cuda();
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    // (mask, source) cases: some-trues, all-false (input unchanged),
    // all-true (full source copy).
    let cases: &[(Vec<bool>, Vec<f32>)] = &[
        (
            vec![false, true, true, false, true, false],
            vec![-1.0, -2.0, -3.0],
        ),
        (vec![false; 6], vec![]),
        (
            vec![true; 6],
            vec![-10.0, -20.0, -30.0, -40.0, -50.0, -60.0],
        ),
    ];
    for (mask_vec, source) in cases {
        let expect = torch_masked_scatter_f32(&input, mask_vec, source);
        let x = cpu_f32(&input, &[6]).to(Device::Cuda(0)).expect("cuda");
        let mask = BoolTensor::from_vec(mask_vec.clone(), vec![6])
            .expect("mask")
            .to(Device::Cuda(0))
            .expect("mask cuda");
        let src = cpu_f32(source, &[source.len()])
            .to(Device::Cuda(0))
            .expect("src cuda");
        let got = x
            .masked_scatter_t(&mask, &src)
            .expect("masked_scatter all-cuda f32");
        assert!(got.is_cuda(), "result must stay CUDA (mask={mask_vec:?})");
        assert_eq!(
            host_f32(&got),
            expect,
            "masked_scatter all-cuda f32 diverged (mask={mask_vec:?})"
        );
    }
}

#[test]
fn masked_scatter_forward_all_cuda_patterns_f64_matches_torch() {
    ensure_cuda();
    let input = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let cases: &[(Vec<bool>, Vec<f64>)] = &[
        (vec![true, false, true, false, true], vec![-7.0, -8.0, -9.0]),
        (vec![false; 5], vec![]),
        (vec![true; 5], vec![-1.0, -2.0, -3.0, -4.0, -5.0]),
    ];
    for (mask_vec, source) in cases {
        let expect = torch_masked_scatter_f64(&input, mask_vec, source);
        let x = cpu_f64(&input, &[5]).to(Device::Cuda(0)).expect("cuda");
        let mask = BoolTensor::from_vec(mask_vec.clone(), vec![5])
            .expect("mask")
            .to(Device::Cuda(0))
            .expect("mask cuda");
        let src = cpu_f64(source, &[source.len()])
            .to(Device::Cuda(0))
            .expect("src cuda");
        let got = x
            .masked_scatter_t(&mask, &src)
            .expect("masked_scatter all-cuda f64");
        assert!(got.is_cuda(), "result must stay CUDA (mask={mask_vec:?})");
        assert_eq!(
            host_f64(&got),
            expect,
            "masked_scatter all-cuda f64 diverged (mask={mask_vec:?})"
        );
    }
}

/// All-CUDA masked_scatter FORWARD must still attach a correct backward.
/// live torch 2.11.0+cu130 (RTX 3090): input=[1,2,3,4] (requires_grad),
/// mask=[F,T,T,F]cuda, source=[10,20]cuda (requires_grad);
///   out = input.masked_scatter(mask, source) = [1,10,20,4];
///   (out * [1,2,3,4]).sum().backward()
///   => input.grad = grad_out.masked_fill(mask,0) = [1,0,0,4]
///   => source.grad = grad_out[mask] = [2,3]
const TORCH_MS_FWD_INPUT_GRAD: [f32; 4] = [1.0, 0.0, 0.0, 4.0];
const TORCH_MS_FWD_SOURCE_GRAD: [f32; 2] = [2.0, 3.0];

#[test]
fn masked_scatter_forward_all_cuda_backward_matches_torch() {
    ensure_cuda();
    let input = cpu_f32(&[1., 2., 3., 4.], &[4])
        .to(Device::Cuda(0))
        .expect("cuda")
        .requires_grad_(true);
    let mask = BoolTensor::from_vec(vec![false, true, true, false], vec![4])
        .expect("mask")
        .to(Device::Cuda(0))
        .expect("mask cuda");
    let source = cpu_f32(&[10., 20.], &[2])
        .to(Device::Cuda(0))
        .expect("src cuda")
        .requires_grad_(true);
    let out = input
        .masked_scatter_t(&mask, &source)
        .expect("masked_scatter all-cuda forward");
    assert!(out.is_cuda());
    assert_eq!(host_f32(&out), vec![1.0, 10.0, 20.0, 4.0]);
    let w = cpu_f32(&[1., 2., 3., 4.], &[4])
        .to(Device::Cuda(0))
        .expect("w cuda");
    let loss = out.mul_t(&w).expect("mul").sum_all().expect("sum");
    loss.backward()
        .expect("backward of all-cuda masked_scatter forward");
    let gi = input.grad().expect("ok").expect("input grad");
    assert_eq!(
        host_f32(&gi),
        TORCH_MS_FWD_INPUT_GRAD.to_vec(),
        "masked_scatter forward input grad diverged from live torch"
    );
    let gs = source.grad().expect("ok").expect("source grad");
    assert_eq!(
        host_f32(&gs),
        TORCH_MS_FWD_SOURCE_GRAD.to_vec(),
        "masked_scatter forward source grad diverged from live torch"
    );
}
