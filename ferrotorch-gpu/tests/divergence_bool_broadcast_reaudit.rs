//! ADVERSARIAL RE-AUDIT of commit 58e754a9b (#1663): on-device CUDA bool
//! broadcast — `gpu_broadcast_bool` (`ferrotorch-gpu/src/bool_kernels.rs`,
//! 8-dim unrolled `BOOL_BROADCAST_PTX`), `CudaBackendImpl::broadcast_bool`
//! (`backend_impl.rs`), wired into the CUDA branch of `broadcast_bool_tensor`
//! (`ferrotorch-core/src/grad_fns/indexing.rs:1781`), so masked ops can
//! broadcast a CUDA bool mask on device. Mirrors `expand_outplace(mask, self)`
//! at `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2406`.
//!
//! The builder's tests only exercised 1-D and 2-D broadcasts. This re-audit
//! attacks the parts they did NOT cover, per the re-audit brief:
//!   - HIGHER RANK: 3-D and 4-D broadcasts ([1,3,1]->[2,3,4], [4]->[2,3,4]).
//!   - LEADING-DIM INSERTION: [3]->[4,5,3] (mask aligns trailing, 2 new leading
//!     axes; replication must be along the leading axes, not the trailing one).
//!   - 8-DIM BOUNDARY: a full ndim==8 broadcast (the unrolled PTX limit) and
//!     ndim==9 (must be a CLEAN ERROR, never silent corruption).
//!   - OTHER CONSUMERS: masked_select / where with a broadcast CUDA mask now
//!     route through broadcast_bool_tensor's CUDA path.
//!
//! Reference (R-CHAR-3 route (b), symbolic): the broadcast contract is the
//! standard NumPy / torch rule — right-align dims; a size-1 or absent input dim
//! replicates. `cpu_broadcast_ref` encodes exactly that rule (it is the within-
//! framework reference the builder's own test (8) uses, and matches the CPU
//! `broadcast_in_flat` index map at `indexing.rs:1730`). The true/false patterns
//! are deliberately ASYMMETRIC so a stride-0-on-the-wrong-dim bug produces a
//! visibly wrong bool vector rather than an accidentally-correct one.

#![cfg(feature = "cuda")]

use ferrotorch_core::gpu_dispatch::gpu_backend;
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
fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("cpu()").data().unwrap().to_vec()
}

fn cuda_mask(bits: &[bool], shape: &[usize]) -> BoolTensor {
    BoolTensor::from_vec(bits.to_vec(), shape.to_vec())
        .expect("mask")
        .to(Device::Cuda(0))
        .expect("mask to cuda")
}

/// The standard NumPy / torch broadcast index map (route (b) symbolic). Mirrors
/// the CPU `grad_fns::indexing::broadcast_in_flat` at `indexing.rs:1730`.
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

/// Run the on-device broadcast kernel and bring the result host-side.
/// Asserts the result stays CUDA-resident (R-CODE-4: no host round trip).
fn gpu_broadcast(bits: &[bool], in_shape: &[usize], out_shape: &[usize]) -> Vec<bool> {
    let mask = cuda_mask(bits, in_shape);
    let backend = gpu_backend().expect("cuda backend");
    let handle = backend
        .broadcast_bool(mask.gpu_handle().unwrap(), in_shape, out_shape)
        .expect("broadcast_bool on device");
    let bt = BoolTensor::from_gpu_handle(handle, out_shape.to_vec());
    assert!(bt.is_cuda(), "broadcast_bool result must stay CUDA-resident");
    bt.to(Device::Cpu).unwrap().data().unwrap().to_vec()
}

// ───────────────────────────────────────────────────────────────────────────
// HIGHER RANK — 3-D and 4-D broadcasts with asymmetric true/false patterns.
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn broadcast_bool_cuda_3d_inner_axes_broadcast_matches_ref() {
    ensure_cuda();
    // [1,3,1] -> [2,3,4]: replicate along outer (axis 0) AND inner (axis 2);
    // middle axis (size 3) is the only carrier. Asymmetric: [T,F,T].
    let bits = [true, false, true];
    let got = gpu_broadcast(&bits, &[1, 3, 1], &[2, 3, 4]);
    let exp = cpu_broadcast_ref(&bits, &[1, 3, 1], &[2, 3, 4]);
    assert_eq!(got, exp, "[1,3,1] -> [2,3,4]");
}

#[test]
fn broadcast_bool_cuda_4d_leading_insert_matches_ref() {
    ensure_cuda();
    // [4] -> [2,3,4]: 1-D mask aligns to the trailing axis (size 4), two new
    // leading axes inserted. Asymmetric: [T,T,F,T].
    let bits = [true, true, false, true];
    let got = gpu_broadcast(&bits, &[4], &[2, 3, 4]);
    let exp = cpu_broadcast_ref(&bits, &[4], &[2, 3, 4]);
    assert_eq!(got, exp, "[4] -> [2,3,4]");
}

// ───────────────────────────────────────────────────────────────────────────
// LEADING-DIM INSERTION — [3] -> [4,5,3]. The mask aligns to the trailing axis;
// the value must replicate along the two NEW leading axes, never along the
// trailing one. A stride-0 placed on the wrong axis here yields a visibly wrong
// vector (the asymmetric [T,F,F] pattern would smear).
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn broadcast_bool_cuda_leading_dim_insertion_matches_ref() {
    ensure_cuda();
    let bits = [true, false, false];
    let got = gpu_broadcast(&bits, &[3], &[4, 5, 3]);
    let exp = cpu_broadcast_ref(&bits, &[3], &[4, 5, 3]);
    assert_eq!(got, exp, "[3] -> [4,5,3] leading-dim insertion");
    // Pin the exact replication shape: every length-3 trailing run must equal
    // the source pattern [T,F,F]; 20 such runs.
    assert_eq!(got.len(), 60);
    for chunk in got.chunks(3) {
        assert_eq!(chunk, &[true, false, false], "trailing run must mirror src");
    }
}

// ───────────────────────────────────────────────────────────────────────────
// 8-DIM BOUNDARY — a full ndim==8 broadcast (the unrolled PTX limit). Mix
// carrier and broadcast axes so the per-dim stride map is fully exercised:
// in [2,1,2,1,2,1,2,1] -> out [2,2,2,2,2,2,2,2] (256 elements).
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn broadcast_bool_cuda_full_8dim_matches_ref() {
    ensure_cuda();
    let in_shape = [2, 1, 2, 1, 2, 1, 2, 1];
    let out_shape = [2, 2, 2, 2, 2, 2, 2, 2];
    let in_numel: usize = in_shape.iter().product(); // 16
    // Asymmetric pattern so any wrong stride visibly corrupts.
    let bits: Vec<bool> = (0..in_numel).map(|i| i % 3 == 0).collect();
    let got = gpu_broadcast(&bits, &in_shape, &out_shape);
    let exp = cpu_broadcast_ref(&bits, &in_shape, &out_shape);
    assert_eq!(got, exp, "[2,1,2,1,2,1,2,1] -> [2;8] full 8-dim");
}

// ───────────────────────────────────────────────────────────────────────────
// 8-DIM BOUNDARY+1 — ndim==9 must be a CLEAN ERROR, never silent corruption.
// pad_bool_broadcast_params rejects rank > BOOL_BROADCAST_MAX_DIMS (8).
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn broadcast_bool_cuda_9dim_is_clean_error_not_corruption() {
    ensure_cuda();
    let in_shape = [1usize; 9];
    let out_shape = [2usize; 9];
    let mask = cuda_mask(&[true], &in_shape);
    let backend = gpu_backend().expect("cuda backend");
    let res = backend.broadcast_bool(mask.gpu_handle().unwrap(), &in_shape, &out_shape);
    assert!(
        res.is_err(),
        "ndim=9 exceeds the 8-dim PTX limit; must error cleanly, not corrupt"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// OTHER CONSUMER — masked_select with a broadcast CUDA mask routes through
// broadcast_bool_tensor's CUDA path. Reference: the CPU masked_select_bcast of
// the SAME inputs (within-framework reference, verified by the CPU unit tests at
// indexing.rs:4172). Both must agree; the GPU result must stay CUDA-resident.
// inp [[1,2,3],[4,5,6]] ([2,3]); 1-D mask [T,F,T] broadcast to [2,3].
// torch: input[broadcast_mask] -> [1,3,4,6] (row-major over true positions).
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn masked_select_bcast_cuda_mask_matches_cpu() {
    ensure_cuda();
    use ferrotorch_core::grad_fns::indexing::masked_select_bcast;

    let inp_cpu = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let mask_cpu = BoolTensor::from_vec(vec![true, false, true], vec![3]).unwrap();
    let cpu_out = masked_select_bcast(&inp_cpu, &mask_cpu).expect("cpu masked_select_bcast");
    let cpu_vec = host_f32(&cpu_out);

    let inp_cuda = inp_cpu.to(Device::Cuda(0)).expect("inp cuda");
    let mask_cuda = cuda_mask(&[true, false, true], &[3]);
    let gpu_out = masked_select_bcast(&inp_cuda, &mask_cuda)
        .expect("masked_select_bcast with broadcast CUDA mask (torch supports this)");
    assert!(
        gpu_out.is_cuda(),
        "masked_select_bcast on CUDA mask must keep result on device"
    );
    assert_eq!(
        host_f32(&gpu_out),
        cpu_vec,
        "GPU broadcast masked_select must match CPU broadcast masked_select"
    );
    // Pin the torch-contract value: [1,3,4,6].
    assert_eq!(cpu_vec, vec![1.0, 3.0, 4.0, 6.0], "torch contract");
}

// ───────────────────────────────────────────────────────────────────────────
// OTHER CONSUMER — where (cond ? input : other) with a broadcast CUDA cond.
// cond [T,F,T] ([3]) broadcast over input/other [2,3]. Reference: the CPU
// where_cond_bcast of the same inputs (within-framework, verified at
// indexing.rs:4200+). torch.where: pick input where cond, else other.
// input rows 1..6, other = 100..600 -> [1,200,3, 4,500,6].
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn where_cond_bcast_cuda_cond_matches_cpu() {
    ensure_cuda();
    use ferrotorch_core::grad_fns::indexing::where_cond_bcast;

    let inp_cpu = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let oth_cpu = cpu_f32(&[100.0, 200.0, 300.0, 400.0, 500.0, 600.0], &[2, 3]);
    let cond_cpu = BoolTensor::from_vec(vec![true, false, true], vec![3]).unwrap();
    let cpu_out = where_cond_bcast(&cond_cpu, &inp_cpu, &oth_cpu).expect("cpu where_cond_bcast");
    let cpu_vec = host_f32(&cpu_out);

    let inp_cuda = inp_cpu.to(Device::Cuda(0)).expect("inp cuda");
    let oth_cuda = oth_cpu.to(Device::Cuda(0)).expect("oth cuda");
    let cond_cuda = cuda_mask(&[true, false, true], &[3]);
    let gpu_out = where_cond_bcast(&cond_cuda, &inp_cuda, &oth_cuda)
        .expect("where_cond_bcast with broadcast CUDA cond (torch supports this)");
    assert!(gpu_out.is_cuda(), "where on CUDA cond must keep result on device");
    assert_eq!(
        host_f32(&gpu_out),
        cpu_vec,
        "GPU broadcast where must match CPU broadcast where"
    );
    assert_eq!(
        cpu_vec,
        vec![1.0, 200.0, 3.0, 4.0, 500.0, 6.0],
        "torch.where contract"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// REGRESSION GUARD — masked_scatter end-to-end with a broadcast CUDA mask, 3-D
// case the builder did not cover. inp [2,2,2]; mask [2,1,2] -> [2,2,2].
// Reference: CPU masked_scatter of the same inputs (within-framework). torch
// broadcasts the [2,1,2] mask over [2,2,2].
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn masked_scatter_broadcast_3d_cuda_mask_matches_cpu() {
    ensure_cuda();
    let inp_h: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let mask_bits = [true, false, false, true]; // [2,1,2]
    let src_h: Vec<f32> = (0..8).map(|k| -(k as f32) - 1.0).collect();

    // CPU reference.
    let inp_cpu = cpu_f32(&inp_h, &[2, 2, 2]);
    let mask_cpu = BoolTensor::from_vec(mask_bits.to_vec(), vec![2, 1, 2]).unwrap();
    let src_cpu = cpu_f32(&src_h, &[8]);
    let cpu_out = inp_cpu
        .masked_scatter_t(&mask_cpu, &src_cpu)
        .expect("cpu masked_scatter broadcast");
    let cpu_vec = host_f32(&cpu_out);

    // GPU all-CUDA path.
    let inp_cuda = cpu_f32(&inp_h, &[2, 2, 2]).to(Device::Cuda(0)).unwrap();
    let mask_cuda = cuda_mask(&mask_bits, &[2, 1, 2]);
    let src_cuda = cpu_f32(&src_h, &[8]).to(Device::Cuda(0)).unwrap();
    let gpu_out = inp_cuda
        .masked_scatter_t(&mask_cuda, &src_cuda)
        .expect("gpu masked_scatter broadcast 3d");
    assert!(gpu_out.is_cuda());
    assert_eq!(
        host_f32(&gpu_out),
        cpu_vec,
        "3-D broadcast masked_scatter: GPU must match CPU (source consumed in \
         flat order over the BROADCAST mask)"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// DEGENERATE — empty target and scalar (rank-0) broadcast.
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn broadcast_bool_cuda_degenerate_shapes() {
    ensure_cuda();
    // [1] -> [0,3]: empty target. Result must be empty.
    let got = gpu_broadcast(&[true], &[1], &[0, 3]);
    assert_eq!(got, Vec::<bool>::new(), "[1] -> [0,3] empty target");
    // scalar [1] -> [5]: all replicate.
    let got2 = gpu_broadcast(&[true], &[1], &[5]);
    assert_eq!(got2, vec![true; 5], "scalar [1] -> [5]");
    // [1] (false) -> [4]: all false.
    let got3 = gpu_broadcast(&[false], &[1], &[4]);
    assert_eq!(got3, vec![false; 4], "scalar false [1] -> [4]");
}
