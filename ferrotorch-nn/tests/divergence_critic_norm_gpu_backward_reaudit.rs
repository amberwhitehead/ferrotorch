//! Critic RE-AUDIT of commit `cbf8db6be` (#1449): BatchNorm / InstanceNorm /
//! LocalResponseNorm GPU BACKWARD on-device.
//!
//! The builder's `divergence_critic_batchnorm_gpu.rs` pinned the GPU backward
//! against torch — but only for ONE shape per op, and (for LRN) only 3 sampled
//! grad_input indices. The user's failure class is "a GPU kernel that compiles
//! and produces plausible-but-wrong gradients" — a kernel that is right for the
//! builder's specific shape but wrong for the GENERAL reduction (B>1, C>1,
//! spatial>1) or at WINDOW BOUNDARIES (LRN cross-channel edges, even `size`).
//!
//! These probes therefore (a) use SHAPES THE BUILDER DID NOT TEST, (b) assert
//! the FULL grad_input array element-by-element (not 2-3 sampled indices), and
//! (c) for LRN exercise EVEN `size` (size=4, size=2) where `half != upper`, the
//! classic place LRN backward gets the asymmetric window wrong.
//!
//! R-CHAR-3 compliance: every expected value is captured LIVE from torch
//! 2.11.0+cu130 autograd on this host's RTX 3090 via `/tmp/oracle_norm.py` /
//! `/tmp/oracle_bn_in.py` (torch.nn.{BatchNorm1d,BatchNorm2d,InstanceNorm2d} /
//! torch.nn.functional.local_response_norm), NOT copied from the ferrotorch
//! side. The capture scripts construct the input deterministically and run
//! `y.backward(go)`; the constants below are torch's `.grad` tensors.
//!
//! Build/run (mold broken on this host):
//!   RUSTFLAGS="-C link-arg=-fuse-ld=lld" cargo test -p ferrotorch-nn \
//!     --features cuda --test divergence_critic_norm_gpu_backward_reaudit

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, Tensor, TensorStorage, backward_with_grad};
use ferrotorch_nn::module::Module as _;
use ferrotorch_nn::norm::{
    BatchNorm1d, BatchNorm2d, BatchNorm3d, InstanceNorm2d, LocalResponseNorm,
};

fn cuda_ready() -> bool {
    ferrotorch_gpu::init_cuda_backend().is_ok()
}

fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// Assert two flat arrays agree to <2e-3 (loose enough for f32 .approx PTX
/// transcendentals; tight enough that a wrong-reduction kernel fails).
fn assert_close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            worst_i = i;
        }
    }
    assert!(
        worst < 2e-3,
        "{label}: worst |Δ|={worst:.6e} at idx {worst_i} (ferrotorch={}, torch={})",
        got[worst_i],
        want[worst_i]
    );
}

// ===========================================================================
// torch oracle constants (live, torch 2.11.0+cu130, RTX 3090)
// ===========================================================================
include!("norm_gpu_reaudit_oracle.rs");

// ---------------------------------------------------------------------------
// BatchNorm2d TRAIN backward, GENERAL reduction: B=4, C=5, H=3, W=3.
// (Builder tested B=2,C=6,H=4,W=5 only.) The cross-element reduction over
// batch*spatial=36 per channel is exercised here with B=4 so a kernel that
// folds the batch dim wrong drifts.
// torch: bn(B=4,C=5,H=3,W=3); see /tmp/oracle_bn_in.py::bn_train.
// Mirrors aten/src/ATen/native/cuda/Normalization.cuh:437
//   grad_input[b][c][x] = (go - (inp-mean)*proj_scale - grad_mean) * grad_scale
// ---------------------------------------------------------------------------
#[test]
fn divergence_bn2d_gpu_train_backward_general_reduction_vs_torch() {
    if !cuda_ready() {
        return;
    }
    let (b, c, h, w) = (4usize, 5, 3, 3);
    let n = b * c * h * w;
    let data: Vec<f32> = (0..n).map(|k| ((k % 19) as f32) * 0.11 - 1.0).collect();
    let gamma: Vec<f32> = (0..c).map(|k| 1.0 + 0.07 * (k as f32)).collect();
    let beta: Vec<f32> = (0..c).map(|k| -0.2 + 0.03 * (k as f32)).collect();
    let go: Vec<f32> = (0..n).map(|k| ((k % 7) as f32) * 0.05 - 0.1).collect();

    let mut bn = BatchNorm2d::<f32>::new(c, 1e-5, 0.1, true).unwrap();
    bn.weight
        .as_mut()
        .unwrap()
        .set_data(cpu_tensor(&gamma, &[c]));
    bn.bias.as_mut().unwrap().set_data(cpu_tensor(&beta, &[c]));
    bn.to_device(Device::Cuda(0)).unwrap();

    let x = cpu_tensor(&data, &[b, c, h, w])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let y = bn.forward(&x).unwrap();
    let go_t = cpu_tensor(&go, &[b, c, h, w]).to(Device::Cuda(0)).unwrap();
    backward_with_grad(&y, Some(&go_t)).unwrap();

    let gi = x.grad().unwrap().expect("grad_input populated");
    assert!(gi.is_cuda(), "grad_input must stay on CUDA (R-CODE-4)");
    assert_close(
        "BN2d train grad_input (full)",
        &gi.data_vec().unwrap(),
        &BNG_GI,
    );

    let gw = bn
        .weight
        .as_ref()
        .unwrap()
        .tensor()
        .grad()
        .unwrap()
        .unwrap();
    assert!(gw.is_cuda());
    assert_close("BN2d train grad_weight", &gw.data_vec().unwrap(), &BNG_GW);
    let gb = bn
        .bias
        .as_ref()
        .unwrap()
        .tensor()
        .grad()
        .unwrap()
        .expect("grad_bias populated");
    assert_close("BN2d train grad_bias", &gb.data_vec().unwrap(), &BNG_GB);
}

// ---------------------------------------------------------------------------
// BatchNorm1d TRAIN backward, GENERAL reduction: B=4, C=3, L=5.
// (Builder never tested BN1d backward at all.) Confirms the shared
// batch_norm_gpu_backward helper handles the 1d (spatial=L) layout.
// ---------------------------------------------------------------------------
#[test]
fn divergence_bn1d_gpu_train_backward_vs_torch() {
    if !cuda_ready() {
        return;
    }
    let (b, c, l) = (4usize, 3, 5);
    let n = b * c * l;
    let data: Vec<f32> = (0..n).map(|k| ((k % 19) as f32) * 0.11 - 1.0).collect();
    let gamma: Vec<f32> = (0..c).map(|k| 1.0 + 0.07 * (k as f32)).collect();
    let beta: Vec<f32> = (0..c).map(|k| -0.2 + 0.03 * (k as f32)).collect();
    let go: Vec<f32> = (0..n).map(|k| ((k % 7) as f32) * 0.05 - 0.1).collect();

    let mut bn = BatchNorm1d::<f32>::new(c, 1e-5, 0.1, true).unwrap();
    bn.weight
        .as_mut()
        .unwrap()
        .set_data(cpu_tensor(&gamma, &[c]));
    bn.bias.as_mut().unwrap().set_data(cpu_tensor(&beta, &[c]));
    bn.to_device(Device::Cuda(0)).unwrap();

    let x = cpu_tensor(&data, &[b, c, l])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let y = bn.forward(&x).unwrap();
    let go_t = cpu_tensor(&go, &[b, c, l]).to(Device::Cuda(0)).unwrap();
    backward_with_grad(&y, Some(&go_t)).unwrap();

    let gi = x.grad().unwrap().expect("grad_input populated");
    assert!(gi.is_cuda(), "grad_input must stay on CUDA (R-CODE-4)");
    assert_close(
        "BN1d train grad_input (full)",
        &gi.data_vec().unwrap(),
        &BN1DG_GI,
    );
    let gw = bn
        .weight
        .as_ref()
        .unwrap()
        .tensor()
        .grad()
        .unwrap()
        .unwrap();
    assert_close("BN1d train grad_weight", &gw.data_vec().unwrap(), &BN1DG_GW);
}

// ---------------------------------------------------------------------------
// InstanceNorm2d affine backward, NEW shape B=2, C=3, H=2, W=4.
// (Builder tested B=3,C=4,H=5,W=6.) The [B,C,S]->[1,B*C,S] reshape + batch-axis
// sum reduction for grad_weight/grad_bias is the user-flagged risk; assert the
// FULL grad_input plus grad_weight/grad_bias.
// torch: InstanceNorm2d(affine=True); see /tmp/oracle_bn_in.py::in_bwd.
// ---------------------------------------------------------------------------
#[test]
fn divergence_instancenorm2d_gpu_backward_newshape_vs_torch() {
    if !cuda_ready() {
        return;
    }
    let (b, c, h, w) = (2usize, 3, 2, 4);
    let n = b * c * h * w;
    let data: Vec<f32> = (0..n).map(|k| ((k % 23) as f32) * 0.09 - 0.8).collect();
    let gamma: Vec<f32> = (0..c).map(|k| 0.9 + 0.05 * (k as f32)).collect();
    let beta: Vec<f32> = (0..c).map(|k| -0.05 + 0.04 * (k as f32)).collect();
    let go: Vec<f32> = (0..n).map(|k| ((k % 11) as f32) * 0.04 - 0.15).collect();

    let mut inorm = InstanceNorm2d::<f32>::new(c, 1e-5, true).unwrap();
    {
        let mut params = inorm.parameters_mut();
        params[0].set_data(cpu_tensor(&gamma, &[c]));
        params[1].set_data(cpu_tensor(&beta, &[c]));
    }
    inorm.to_device(Device::Cuda(0)).unwrap();

    let x = cpu_tensor(&data, &[b, c, h, w])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let y = inorm.forward(&x).unwrap();
    let go_t = cpu_tensor(&go, &[b, c, h, w]).to(Device::Cuda(0)).unwrap();
    backward_with_grad(&y, Some(&go_t)).unwrap();

    let gi = x.grad().unwrap().expect("grad_input populated");
    assert!(gi.is_cuda(), "InstanceNorm grad_input must stay on CUDA");
    assert_close(
        "InstanceNorm grad_input (full)",
        &gi.data_vec().unwrap(),
        &ING_GI,
    );

    let params = inorm.named_parameters();
    let gw = params
        .iter()
        .find(|(n, _)| n == "weight")
        .unwrap()
        .1
        .tensor()
        .grad()
        .unwrap()
        .unwrap();
    assert!(gw.is_cuda());
    assert_close("InstanceNorm grad_weight", &gw.data_vec().unwrap(), &ING_GW);
    let gb = params
        .iter()
        .find(|(n, _)| n == "bias")
        .unwrap()
        .1
        .tensor()
        .grad()
        .unwrap()
        .unwrap();
    assert_close("InstanceNorm grad_bias", &gb.data_vec().unwrap(), &ING_GB);
}

// ---------------------------------------------------------------------------
// LRN forward+backward, ODD size=3, c=5, B=2: FULL out + FULL grad_input.
// (Builder only sampled 3 grad indices at size=5.) Asserting all 60 grad
// elements catches a per-element window-bound off-by-one anywhere in the array.
// ---------------------------------------------------------------------------
#[test]
fn divergence_lrn_gpu_size3_full_array_vs_torch() {
    if !cuda_ready() {
        return;
    }
    let (b, c, h, w) = (2usize, 5, 2, 3);
    let n = b * c * h * w;
    let data: Vec<f32> = (0..n).map(|k| ((k % 13) as f32) * 0.17 - 1.0).collect();
    let go: Vec<f32> = (0..n).map(|k| ((k % 5) as f32) * 0.07 - 0.1).collect();

    let lrn = LocalResponseNorm::new(3, 1e-4, 0.75, 1.0).unwrap();
    let x = cpu_tensor(&data, &[b, c, h, w])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let y: Tensor<f32> = lrn.forward(&x).unwrap();
    assert!(y.is_cuda());
    assert_close(
        "LRN size=3 forward (full)",
        &y.data_vec().unwrap(),
        &LRNA_OUT,
    );

    let go_t = cpu_tensor(&go, &[b, c, h, w]).to(Device::Cuda(0)).unwrap();
    backward_with_grad(&y, Some(&go_t)).unwrap();
    let gi = x.grad().unwrap().expect("grad_input populated");
    assert!(gi.is_cuda(), "LRN grad_input must stay on CUDA (R-CODE-4)");
    assert_close(
        "LRN size=3 grad_input (full)",
        &gi.data_vec().unwrap(),
        &LRNA_GI,
    );
}

// ---------------------------------------------------------------------------
// LRN EVEN size=4, c=6, B=1: half=2, upper=2 — but PyTorch pads asymmetrically
// (size//2 left, (size-1)//2 right => 2 left, 1 right), so the forward window
// for channel c is [c-2, c+2) and the backward cross window is asymmetric.
// This is THE classic place an LRN kernel that assumes a symmetric window
// drifts. FULL forward + FULL grad_input.
// torch: F.local_response_norm(size=4, alpha=2e-4, beta=0.5, k=1.5).
// ---------------------------------------------------------------------------
#[test]
fn divergence_lrn_gpu_even_size4_window_vs_torch() {
    if !cuda_ready() {
        return;
    }
    let (b, c, h, w) = (1usize, 6, 2, 2);
    let n = b * c * h * w;
    let data: Vec<f32> = (0..n).map(|k| ((k % 11) as f32) * 0.21 - 0.9).collect();
    let go: Vec<f32> = (0..n).map(|k| ((k % 7) as f32) * 0.05 - 0.12).collect();

    let lrn = LocalResponseNorm::new(4, 2e-4, 0.5, 1.5).unwrap();
    let x = cpu_tensor(&data, &[b, c, h, w])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let y: Tensor<f32> = lrn.forward(&x).unwrap();
    assert!(y.is_cuda());
    assert_close(
        "LRN even size=4 forward (full)",
        &y.data_vec().unwrap(),
        &LRNB_OUT,
    );

    let go_t = cpu_tensor(&go, &[b, c, h, w]).to(Device::Cuda(0)).unwrap();
    backward_with_grad(&y, Some(&go_t)).unwrap();
    let gi = x.grad().unwrap().expect("grad_input populated");
    assert!(gi.is_cuda(), "LRN grad_input must stay on CUDA");
    assert_close(
        "LRN even size=4 grad_input (full)",
        &gi.data_vec().unwrap(),
        &LRNB_GI,
    );
}

// ---------------------------------------------------------------------------
// LRN EVEN size=2, c=4: half=1, upper=1 — minimal even window. PyTorch pad is
// (size//2=1 left, (size-1)//2=0 right) => window [c-1, c+1). Forward window
// covers {c-1, c}; an implementation using a symmetric [c-1, c+2) window drifts.
// FULL forward + FULL grad_input.
// torch: F.local_response_norm(size=2, alpha=5e-4, beta=0.6, k=2.0).
// ---------------------------------------------------------------------------
#[test]
fn divergence_lrn_gpu_even_size2_window_vs_torch() {
    if !cuda_ready() {
        return;
    }
    let (b, c, h, w) = (2usize, 4, 1, 3);
    let n = b * c * h * w;
    let data: Vec<f32> = (0..n).map(|k| ((k % 9) as f32) * 0.3 - 1.2).collect();
    let go: Vec<f32> = (0..n).map(|k| ((k % 4) as f32) * 0.1 - 0.15).collect();

    let lrn = LocalResponseNorm::new(2, 5e-4, 0.6, 2.0).unwrap();
    let x = cpu_tensor(&data, &[b, c, h, w])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let y: Tensor<f32> = lrn.forward(&x).unwrap();
    assert!(y.is_cuda());
    assert_close(
        "LRN even size=2 forward (full)",
        &y.data_vec().unwrap(),
        &LRNC_OUT,
    );

    let go_t = cpu_tensor(&go, &[b, c, h, w]).to(Device::Cuda(0)).unwrap();
    backward_with_grad(&y, Some(&go_t)).unwrap();
    let gi = x.grad().unwrap().expect("grad_input populated");
    assert!(gi.is_cuda(), "LRN grad_input must stay on CUDA");
    assert_close(
        "LRN even size=2 grad_input (full)",
        &gi.data_vec().unwrap(),
        &LRNC_GI,
    );
}

// ---------------------------------------------------------------------------
// DIVERGENCE (#1567): NON-AFFINE BatchNorm2d train backward.
//
// `BatchNorm{1,2,3}dBackward::inputs()` returns 1 element when affine=false
// (just `input`), but the new GPU helper `batch_norm_gpu_backward`
// (ferrotorch-nn/src/norm.rs:262) — and the CPU path (norm.rs:2253) — always
// return a 3-element grad vec `[Some(grad_input), grad_weight, grad_bias]`.
// For affine=false the autograd engine then errors
//   `backward returned 3 gradients but expected 1`,
// so a non-affine GPU BatchNorm backward CANNOT run at all. torch handles
// `nn.BatchNorm2d(affine=False)` backward fine (oracle BNNA_GI captured live).
//
// Pre-existing (CPU has the identical bug) but reachable through the audited
// commit's GPU path, so the commit's "BatchNorm backward all run on-device and
// match torch" claim is false for affine=false. Fix is the generator's: make
// the returned grad vec length match `inputs()` (drop weight/bias when
// non-affine) in BOTH the GPU helper and the CPU return.
//
// torch: BatchNorm2d(affine=False); see /tmp/oracle_extra.py::bn_noaffine.
// Tracking: #1567
// ---------------------------------------------------------------------------
// FIXED #1567: BatchNorm{1,2,3}dBackward now return a grad vec whose length
// matches `inputs()` (length 1 when affine=false), so the non-affine GPU
// backward runs and matches torch. Permanent regression coverage.
#[test]
fn divergence_bn2d_gpu_nonaffine_train_backward_vs_torch() {
    if !cuda_ready() {
        return;
    }
    let (b, c, h, w) = (3usize, 4, 2, 3);
    let n = b * c * h * w;
    let data: Vec<f32> = (0..n).map(|k| ((k % 19) as f32) * 0.11 - 1.0).collect();
    let go: Vec<f32> = (0..n).map(|k| ((k % 7) as f32) * 0.05 - 0.1).collect();

    let mut bn = BatchNorm2d::<f32>::new(c, 1e-5, 0.1, false).unwrap();
    bn.to_device(Device::Cuda(0)).unwrap();

    let x = cpu_tensor(&data, &[b, c, h, w])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let y = bn.forward(&x).unwrap();
    let go_t = cpu_tensor(&go, &[b, c, h, w]).to(Device::Cuda(0)).unwrap();
    backward_with_grad(&y, Some(&go_t)).unwrap();

    let gi = x.grad().unwrap().expect("grad_input populated");
    assert!(gi.is_cuda(), "grad_input must stay on CUDA (R-CODE-4)");
    assert_close(
        "BN2d non-affine grad_input (full)",
        &gi.data_vec().unwrap(),
        &BNNA_GI,
    );
}

// ---------------------------------------------------------------------------
// BatchNorm3d train backward: B=2, C=3, D=2, H=2, W=2 (spatial=8).
// (Builder never tested BN3d backward.) Confirms the shared helper flattens
// the 3 trailing spatial dims into hw correctly.
// torch: BatchNorm3d(affine=True); see /tmp/oracle_extra.py::bn3d.
// ---------------------------------------------------------------------------
#[test]
fn divergence_bn3d_gpu_train_backward_vs_torch() {
    if !cuda_ready() {
        return;
    }
    let (b, c, d, h, w) = (2usize, 3, 2, 2, 2);
    let n = b * c * d * h * w;
    let data: Vec<f32> = (0..n).map(|k| ((k % 19) as f32) * 0.11 - 1.0).collect();
    let gamma: Vec<f32> = (0..c).map(|k| 1.0 + 0.07 * (k as f32)).collect();
    let beta: Vec<f32> = (0..c).map(|k| -0.2 + 0.03 * (k as f32)).collect();
    let go: Vec<f32> = (0..n).map(|k| ((k % 7) as f32) * 0.05 - 0.1).collect();

    let mut bn = BatchNorm3d::<f32>::new(c, 1e-5, 0.1, true).unwrap();
    bn.weight
        .as_mut()
        .unwrap()
        .set_data(cpu_tensor(&gamma, &[c]));
    bn.bias.as_mut().unwrap().set_data(cpu_tensor(&beta, &[c]));
    bn.to_device(Device::Cuda(0)).unwrap();

    let x = cpu_tensor(&data, &[b, c, d, h, w])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let y = bn.forward(&x).unwrap();
    let go_t = cpu_tensor(&go, &[b, c, d, h, w])
        .to(Device::Cuda(0))
        .unwrap();
    backward_with_grad(&y, Some(&go_t)).unwrap();

    let gi = x.grad().unwrap().expect("grad_input populated");
    assert!(gi.is_cuda(), "grad_input must stay on CUDA (R-CODE-4)");
    assert_close(
        "BN3d train grad_input (full)",
        &gi.data_vec().unwrap(),
        &BN3D_GI,
    );
    let gw = bn
        .weight
        .as_ref()
        .unwrap()
        .tensor()
        .grad()
        .unwrap()
        .unwrap();
    assert_close("BN3d train grad_weight", &gw.data_vec().unwrap(), &BN3D_GW);
    let gb = bn
        .bias
        .as_ref()
        .unwrap()
        .tensor()
        .grad()
        .unwrap()
        .expect("grad_bias populated");
    assert_close("BN3d train grad_bias", &gb.data_vec().unwrap(), &BN3D_GB);
}
