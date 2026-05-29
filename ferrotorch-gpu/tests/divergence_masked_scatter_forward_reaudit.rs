//! ADVERSARIAL RE-AUDIT of commit a7d6e5008 (#1662): the new ALL-CUDA
//! `masked_scatter` forward kernel (`masked_scatter_forward_{32,64}` /
//! `launch_scatter_forward` in `ferrotorch-gpu/src/masked_kernels.rs`, wired
//! from `grad_fns::indexing::masked_scatter`'s all-CUDA branch).
//!
//! Hunt targets (per the re-audit brief):
//!   1. SERIAL-THREAD CORRECTNESS AT SCALE — a single in-order thread with a
//!      serial source cursor `j` (exclusive prefix-sum of the mask). Verify a
//!      large (5000-elem) scattered mask reads source[j] (0-indexed, exclusive)
//!      at every true position with no off-by-one and no timeout. PASSES.
//!   2. OFF-BY-ONE source indexing — mask [T,F,T,T,F,T] with distinct source
//!      values; the k-th true slot must read source[k]. PASSES.
//!   3. STORAGE_OFFSET — narrowed-offset CUDA *input* routed through the ALL-CUDA
//!      kernel (CUDA mask + CUDA source), exercising `input_c = input_b
//!      .contiguous()` in the new branch (the prior reaudit only tested the
//!      narrowed input with a CPU mask, i.e. the host path). PASSES — the
//!      .contiguous() normalisation honours the offset (priority concern: CLEAN).
//!   4. SOURCE-COUNT — source LONGER than #true (extra ignored, torch allows).
//!      PASSES. (Source TOO SHORT: torch raises a device-side assert; ferrotorch
//!      returns a clean `ShapeMismatch` Err — acceptable, not pinned here.)
//!   5. MASK PATTERNS — all-false / all-true / single-true; f32 AND f64;
//!      `is_cuda()` preserved (no host round trip). PASSES.
//!   6. BACKWARD on an all-CUDA masked_scatter with a non-trivial grad_output.
//!      PASSES.
//!   7. BROADCAST MASK — input [2,3] + CUDA mask [3] (torch broadcasts the mask
//!      on-device). FIXED in #1663: `broadcast_bool_tensor`'s CUDA path now
//!      dispatches to the on-device `GpuBackend::broadcast_bool` kernel
//!      (`ferrotorch-gpu/src/bool_kernels.rs::gpu_broadcast_bool`), so the
//!      broadcast mask reaches the all-CUDA forward kernel. PASSES (test (7)).
//!
//! R-CHAR-3 provenance: every expected value is the live-torch result captured
//! in this env (torch 2.11.0+cu130, RTX 3090) — recorded inline as a symbolic
//! constant traceable to the `LD_LIBRARY_PATH=$HOME/.local/lib python3` run.
//!
//! VERDICT: CLEAN — tests (1)-(7) are permanent regression guards (NOT
//! `#[ignore]`d; #1663 closed the broadcast divergence on the all-CUDA path).

#![cfg(feature = "cuda")]

use ferrotorch_core::{BoolTensor, Device, Tensor, TensorStorage};
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

fn cuda_mask(bits: &[bool], shape: &[usize]) -> BoolTensor {
    BoolTensor::from_vec(bits.to_vec(), shape.to_vec())
        .expect("mask")
        .to(Device::Cuda(0))
        .expect("mask to cuda")
}

// ───────────────────────────────────────────────────────────────────────────
// (2) OFF-BY-ONE: distinct sources, scattered mask [T,F,T,T,F,T].
// live torch: inp [10,20,30,40,50,60], mask [T,F,T,T,F,T], src [-1,-2,-3,-4]
//   -> [-1, 20, -2, -3, 50, -4]   (k-th true reads src[k], exclusive cursor)
// ───────────────────────────────────────────────────────────────────────────
const TORCH_OFFBYONE: [f32; 6] = [-1.0, 20.0, -2.0, -3.0, 50.0, -4.0];

#[test]
fn masked_scatter_forward_all_cuda_offbyone_source_index_matches_torch() {
    ensure_cuda();
    let inp = cpu_f32(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[6])
        .to(Device::Cuda(0))
        .expect("inp cuda");
    let mask = cuda_mask(&[true, false, true, true, false, true], &[6]);
    let src = cpu_f32(&[-1.0, -2.0, -3.0, -4.0], &[4])
        .to(Device::Cuda(0))
        .expect("src cuda");
    let out = inp
        .masked_scatter_t(&mask, &src)
        .expect("masked_scatter all-cuda");
    assert!(out.is_cuda(), "result must stay CUDA (no host round trip)");
    assert_eq!(
        host_f32(&out),
        TORCH_OFFBYONE.to_vec(),
        "off-by-one: k-th true position must read source[k] (exclusive cursor)"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// (1) SCALE: N=5000, mask = (i % 3 == 0), source[k] = -(k+1).
// At a true position i (i%3==0) the source index is i//3 -> value -(i//3)-1;
// false positions keep input[i] = i. (live torch sum == 6941389.0.)
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn masked_scatter_forward_all_cuda_scale_5000_no_offbyone_matches_torch() {
    ensure_cuda();
    const N: usize = 5000;
    let inp_h: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let mask_bits: Vec<bool> = (0..N).map(|i| i % 3 == 0).collect();
    let ntrue = mask_bits.iter().filter(|&&b| b).count();
    let src_h: Vec<f32> = (0..ntrue).map(|k| -(k as f32) - 1.0).collect();

    let inp = cpu_f32(&inp_h, &[N]).to(Device::Cuda(0)).expect("inp cuda");
    let mask = cuda_mask(&mask_bits, &[N]);
    let src = cpu_f32(&src_h, &[ntrue])
        .to(Device::Cuda(0))
        .expect("src cuda");
    let out = inp
        .masked_scatter_t(&mask, &src)
        .expect("masked_scatter scale");
    assert!(out.is_cuda());
    let got = host_f32(&out);

    // Reconstruct the live-torch expectation elementwise (route (b) symbolic).
    let exp: Vec<f32> = (0..N)
        .map(|i| {
            if i % 3 == 0 {
                -((i / 3) as f32) - 1.0
            } else {
                i as f32
            }
        })
        .collect();
    assert_eq!(got, exp, "serial cursor diverged at scale");
    let sum: f32 = got.iter().sum();
    assert_eq!(sum, 6941389.0, "scale checksum (live torch)");
}

// ───────────────────────────────────────────────────────────────────────────
// (3) STORAGE_OFFSET via the ALL-CUDA path: narrowed-offset CUDA input + CUDA
// mask + CUDA source. This routes through the new branch's
// `input_c = input_b.contiguous()`, NOT the host path the prior reaudit covered.
// live torch:
//   full = arange(1..9).cuda().reshape(4,2); view = full[1:4] -> [[3,4],[5,6],[7,8]]
//   mask(cuda) [[F,T],[T,F],[F,T]]; src(cuda) [-1,-2,-3]
//   view.masked_scatter(mask, src).flatten() -> [3, -1, -2, 6, 7, -3]
// ───────────────────────────────────────────────────────────────────────────
const TORCH_OFFSET_INPUT: [f32; 6] = [3.0, -1.0, -2.0, 6.0, 7.0, -3.0];

#[test]
fn masked_scatter_forward_all_cuda_narrowed_offset_input_matches_torch() {
    ensure_cuda();
    let full = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2])
        .to(Device::Cuda(0))
        .expect("full cuda");
    let view = full.narrow(0, 1, 3).expect("narrow rows 1..4");
    assert_eq!(view.shape(), &[3, 2]);
    assert_ne!(view.storage_offset(), 0, "view must carry a storage_offset");
    // CUDA mask + CUDA source -> exercises the all-CUDA fast path with offset.
    let mask = cuda_mask(&[false, true, true, false, false, true], &[3, 2]);
    let src = cpu_f32(&[-1.0, -2.0, -3.0], &[3])
        .to(Device::Cuda(0))
        .expect("src cuda");
    let out = view
        .masked_scatter_t(&mask, &src)
        .expect("masked_scatter offset");
    assert!(out.is_cuda());
    assert_eq!(
        host_f32(&out),
        TORCH_OFFSET_INPUT.to_vec(),
        "all-CUDA path must honour the narrowed input's storage_offset"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// (4) SOURCE LONGER than #true: extra source elements ignored.
// live torch: inp [1,2,3,4], mask [F,T,T,F], src [-1,-2,-3,-4,-5]
//   -> [1, -1, -2, 4]
// ───────────────────────────────────────────────────────────────────────────
const TORCH_LONGER_SRC: [f32; 4] = [1.0, -1.0, -2.0, 4.0];

#[test]
fn masked_scatter_forward_all_cuda_source_longer_than_trues_matches_torch() {
    ensure_cuda();
    let inp = cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[4])
        .to(Device::Cuda(0))
        .expect("inp cuda");
    let mask = cuda_mask(&[false, true, true, false], &[4]);
    let src = cpu_f32(&[-1.0, -2.0, -3.0, -4.0, -5.0], &[5])
        .to(Device::Cuda(0))
        .expect("src cuda");
    let out = inp
        .masked_scatter_t(&mask, &src)
        .expect("masked_scatter long src");
    assert!(out.is_cuda());
    assert_eq!(
        host_f32(&out),
        TORCH_LONGER_SRC.to_vec(),
        "extra source elements past #true must be ignored (torch takes first #true)"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// (5) MASK PATTERNS f32: all-false / all-true / single-true; is_cuda preserved.
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn masked_scatter_forward_all_cuda_patterns_f32_extra_matches_torch() {
    ensure_cuda();
    // all-false -> out == input
    {
        let inp = cpu_f32(&[1.0, 2.0, 3.0], &[3]).to(Device::Cuda(0)).unwrap();
        let mask = cuda_mask(&[false, false, false], &[3]);
        let src = cpu_f32(&[9.0], &[1]).to(Device::Cuda(0)).unwrap();
        let out = inp.masked_scatter_t(&mask, &src).unwrap();
        assert!(out.is_cuda());
        assert_eq!(host_f32(&out), vec![1.0, 2.0, 3.0]);
    }
    // all-true -> out == source (reshaped)
    {
        let inp = cpu_f32(&[1.0, 2.0, 3.0], &[3]).to(Device::Cuda(0)).unwrap();
        let mask = cuda_mask(&[true, true, true], &[3]);
        let src = cpu_f32(&[-7.0, -8.0, -9.0], &[3])
            .to(Device::Cuda(0))
            .unwrap();
        let out = inp.masked_scatter_t(&mask, &src).unwrap();
        assert_eq!(host_f32(&out), vec![-7.0, -8.0, -9.0]);
    }
    // single-true at the last position
    {
        let inp = cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[4])
            .to(Device::Cuda(0))
            .unwrap();
        let mask = cuda_mask(&[false, false, false, true], &[4]);
        let src = cpu_f32(&[-5.0, -6.0], &[2]).to(Device::Cuda(0)).unwrap();
        let out = inp.masked_scatter_t(&mask, &src).unwrap();
        assert_eq!(host_f32(&out), vec![1.0, 2.0, 3.0, -5.0]);
    }
}

#[test]
fn masked_scatter_forward_all_cuda_patterns_f64_extra_matches_torch() {
    ensure_cuda();
    let inp = cpu_f64(&[1.0, 2.0, 3.0, 4.0], &[4])
        .to(Device::Cuda(0))
        .unwrap();
    let mask = cuda_mask(&[true, false, true, false], &[4]);
    let src = cpu_f64(&[-1.0, -2.0], &[2]).to(Device::Cuda(0)).unwrap();
    let out = inp.masked_scatter_t(&mask, &src).unwrap();
    assert!(out.is_cuda());
    // live torch: [-1, 2, -2, 4]
    assert_eq!(host_f64(&out), vec![-1.0, 2.0, -2.0, 4.0]);
}

// ───────────────────────────────────────────────────────────────────────────
// (6) BACKWARD on an all-CUDA masked_scatter, non-trivial grad_output.
// live torch: inp(req grad) [1,2,3,4], src(req grad) [10,20], mask [F,T,T,F];
//   out.backward([5,6,7,8]) -> input.grad [5,0,0,8]; source.grad [6,7].
// ───────────────────────────────────────────────────────────────────────────
const TORCH_BWD_INPUT_GRAD: [f32; 4] = [5.0, 0.0, 0.0, 8.0];
const TORCH_BWD_SOURCE_GRAD: [f32; 2] = [6.0, 7.0];

#[test]
fn masked_scatter_forward_all_cuda_backward_nontrivial_matches_torch() {
    ensure_cuda();
    let inp = cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[4])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let src = cpu_f32(&[10.0, 20.0], &[2])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let mask = cuda_mask(&[false, true, true, false], &[4]);
    let out = inp
        .masked_scatter_t(&mask, &src)
        .expect("masked_scatter bwd");
    assert!(out.is_cuda());
    let go = cpu_f32(&[5.0, 6.0, 7.0, 8.0], &[4])
        .to(Device::Cuda(0))
        .unwrap();
    out.backward_with_gradient(&go).expect("backward");
    let ig = inp
        .grad()
        .expect("input grad query")
        .expect("input has grad");
    let sg = src
        .grad()
        .expect("source grad query")
        .expect("source has grad");
    assert_eq!(
        host_f32(&ig),
        TORCH_BWD_INPUT_GRAD.to_vec(),
        "input.grad = grad.masked_fill(mask, 0)"
    );
    assert_eq!(
        host_f32(&sg),
        TORCH_BWD_SOURCE_GRAD.to_vec(),
        "source.grad = grad compacted at true positions"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// (7) BROADCAST MASK on the ALL-CUDA path — DIVERGENCE.
//
/// Divergence: ferrotorch's `grad_fns::indexing::masked_scatter` diverges from
/// `pytorch aten/src/ATen/native/TensorAdvancedIndexing.cpp:2406`
/// (`expand_outplace(mask, self)`) for a CUDA-resident mask that needs
/// broadcasting. `masked_scatter` runs `mask_b = broadcast_bool_tensor(mask,
/// &common)?` at `ferrotorch-core/src/grad_fns/indexing.rs:3630` BEFORE the
/// #1662 all-CUDA forward branch (`indexing.rs:3644`); `broadcast_bool_tensor`
/// (`indexing.rs:1777-1780`) returns `NotImplementedOnCuda { op:
/// "broadcast_bool_tensor" }` for any CUDA mask whose shape != out_shape.
/// Upstream (live torch 2.11.0+cu130, RTX 3090) broadcasts the 1-D CUDA mask
/// [T,F,T] over input [[1,2,3],[4,5,6]] and returns [[-1,2,-2],[-3,5,-4]] with
/// source [-1,-2,-3,-4], all on device. ferrotorch returns an `Err` instead.
/// Tracking: #1663
// ───────────────────────────────────────────────────────────────────────────
const TORCH_BCAST_MASK: [f32; 6] = [-1.0, 2.0, -2.0, -3.0, 5.0, -4.0];

#[test]
fn masked_scatter_forward_all_cuda_broadcast_mask_matches_torch() {
    ensure_cuda();
    let inp = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .to(Device::Cuda(0))
        .expect("inp cuda");
    // 1-D CUDA mask that torch broadcasts to [2,3].
    let mask = cuda_mask(&[true, false, true], &[3]);
    let src = cpu_f32(&[-1.0, -2.0, -3.0, -4.0], &[4])
        .to(Device::Cuda(0))
        .expect("src cuda");
    let out = inp
        .masked_scatter_t(&mask, &src)
        .expect("masked_scatter broadcast CUDA mask (torch supports this on-device)");
    assert!(out.is_cuda());
    assert_eq!(
        host_f32(&out),
        TORCH_BCAST_MASK.to_vec(),
        "torch broadcasts a CUDA mask on-device; ferrotorch must not reject it"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// (8) DIRECT broadcast_bool kernel matrix (#1663). Exercise the on-device
// `GpuBackend::broadcast_bool` slot for the broadcast patterns the consumer
// relies on, comparing the expanded CUDA bool buffer to the CPU broadcast of
// the same mask. is_cuda() preserved (no host round trip in the kernel; we read
// host-side only to assert). The "expected" is the host broadcast computed by
// the same NumPy/torch rule (right-align dims; size-1/absent dim replicates) —
// route (b) symbolic, the standard broadcast contract.
// ───────────────────────────────────────────────────────────────────────────
fn cpu_broadcast_ref(bits: &[bool], in_shape: &[usize], out_shape: &[usize]) -> Vec<bool> {
    let out_numel: usize = out_shape.iter().product();
    let out_ndim = out_shape.len();
    let in_ndim = in_shape.len();
    let mut in_strides = vec![0usize; in_ndim];
    if in_ndim > 0 {
        in_strides[in_ndim - 1] = 1;
        for d in (0..in_ndim - 1).rev() {
            in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
        }
    }
    let mut out = Vec::with_capacity(out_numel);
    for flat in 0..out_numel {
        let mut rem = flat;
        let mut in_idx = 0usize;
        for d_out in (0..out_ndim).rev() {
            let coord = rem % out_shape[d_out];
            rem /= out_shape[d_out];
            let d_off = out_ndim - 1 - d_out;
            if d_off < in_ndim {
                let d_in = in_ndim - 1 - d_off;
                if in_shape[d_in] != 1 {
                    in_idx += coord * in_strides[d_in];
                }
            }
        }
        out.push(bits[in_idx]);
    }
    out
}

fn broadcast_bool_on_device(bits: &[bool], in_shape: &[usize], out_shape: &[usize]) -> Vec<bool> {
    use ferrotorch_core::gpu_dispatch::gpu_backend;
    let mask = cuda_mask(bits, in_shape);
    let backend = gpu_backend().expect("cuda backend registered");
    let handle = backend
        .broadcast_bool(mask.gpu_handle().unwrap(), in_shape, out_shape)
        .expect("broadcast_bool on device");
    let bt = BoolTensor::from_gpu_handle(handle, out_shape.to_vec());
    assert!(bt.is_cuda(), "broadcast_bool result must stay CUDA");
    bt.to(Device::Cpu).unwrap().data().unwrap().to_vec()
}

#[test]
fn broadcast_bool_cuda_matrix_matches_cpu_broadcast() {
    ensure_cuda();
    // [3] -> [2,3]
    let bits = [true, false, true];
    assert_eq!(
        broadcast_bool_on_device(&bits, &[3], &[2, 3]),
        cpu_broadcast_ref(&bits, &[3], &[2, 3]),
        "[3] -> [2,3]"
    );
    // [1,3] -> [2,3]
    assert_eq!(
        broadcast_bool_on_device(&bits, &[1, 3], &[2, 3]),
        cpu_broadcast_ref(&bits, &[1, 3], &[2, 3]),
        "[1,3] -> [2,3]"
    );
    // [2,1] -> [2,3]
    let col = [true, false];
    assert_eq!(
        broadcast_bool_on_device(&col, &[2, 1], &[2, 3]),
        cpu_broadcast_ref(&col, &[2, 1], &[2, 3]),
        "[2,1] -> [2,3]"
    );
    // scalar [1] -> [2,3]
    let scalar = [true];
    assert_eq!(
        broadcast_bool_on_device(&scalar, &[1], &[2, 3]),
        vec![true; 6],
        "scalar [1] -> [2,3]"
    );
    // already-equal shape: kernel returns the same bits (no-op broadcast).
    let same = [true, false, true, false, true, false];
    assert_eq!(
        broadcast_bool_on_device(&same, &[2, 3], &[2, 3]),
        same.to_vec(),
        "[2,3] -> [2,3] no-op"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// (9) masked_scatter with a broadcast CUDA mask, additional shapes (#1663).
// live torch (torch 2.11.0+cu130, RTX 3090):
//   inp [[1,2,3],[4,5,6]], col mask [[T],[F]] (shape [2,1]) broadcast to [2,3],
//   src [-1,-2,-3,-4] -> row 0 all true (3 trues take -1,-2,-3), row 1 all false
//   -> [[-1,-2,-3],[4,5,6]]
// ───────────────────────────────────────────────────────────────────────────
const TORCH_BCAST_COL_MASK: [f32; 6] = [-1.0, -2.0, -3.0, 4.0, 5.0, 6.0];

#[test]
fn masked_scatter_forward_all_cuda_broadcast_col_mask_matches_torch() {
    ensure_cuda();
    let inp = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .to(Device::Cuda(0))
        .expect("inp cuda");
    // [2,1] CUDA mask broadcast to [2,3].
    let mask = cuda_mask(&[true, false], &[2, 1]);
    let src = cpu_f32(&[-1.0, -2.0, -3.0, -4.0], &[4])
        .to(Device::Cuda(0))
        .expect("src cuda");
    let out = inp
        .masked_scatter_t(&mask, &src)
        .expect("masked_scatter broadcast [2,1] CUDA mask");
    assert!(out.is_cuda());
    assert_eq!(
        host_f32(&out),
        TORCH_BCAST_COL_MASK.to_vec(),
        "torch broadcasts a [2,1] CUDA mask over [2,3] on-device"
    );
}
