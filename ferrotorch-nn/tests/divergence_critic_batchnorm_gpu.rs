//! Critic re-audit of #1449 (commit e09ae21d8): BatchNorm/InstanceNorm GPU
//! forward. The builder's own tests only compared GPU output against
//! ferrotorch's CPU path (GPU == CPU), which is tautological w.r.t. PyTorch:
//! if the CPU path is wrong, GPU == CPU still passes while both diverge from
//! torch. These probes instead pin the LIVE CUDA GPU output (and running-stat
//! updates) directly against `torch.nn.BatchNorm2d` / `torch.nn.InstanceNorm2d`
//! oracle values captured from torch 2.11.0+cu130 on this host.
//!
//! Oracle capture (R-CHAR-3 compliant — expected values come from live torch,
//! NOT copied from the ferrotorch side):
//!
//! ```text
//! import torch
//! b,c,h,w = 2,6,4,5
//! data  = [[k%19]*0.11 - 1.0 for k in range(b*c*h*w)] reshaped
//! gamma = [1.0+0.07*k], beta = [-0.2+0.03*k]
//! rmean = [0.05*k-0.1],  rvar = [0.8+0.05*k]
//! bn = torch.nn.BatchNorm2d(c, eps=1e-5, momentum=0.1, affine=True)
//! ```
//!
//! These tests run live (host HAS CUDA, RTX 3090). They build via lld:
//! `RUSTFLAGS="-C link-arg=-fuse-ld=lld" cargo test -p ferrotorch-nn \
//!   --features cuda --test divergence_critic_batchnorm_gpu`
//! (the host's `mold` is currently broken: TBB `get_thread_reference_vertex`
//! symbol-lookup error — orthogonal to this audit).

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, Tensor, TensorStorage, backward_with_grad};
use ferrotorch_nn::module::Module as _;
use ferrotorch_nn::norm::{BatchNorm2d, InstanceNorm2d, LocalResponseNorm};

/// Skip-guard: returns false when no CUDA device initializes (so the file is a
/// no-op on non-CUDA hosts). On this host it always inits.
fn cuda_ready() -> bool {
    ferrotorch_gpu::init_cuda_backend().is_ok()
}

fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// Trap 1 + 4: train-mode batch statistics + affine.
///
/// Divergence target: ferrotorch's `BatchNorm2d::forward` GPU path (train) vs.
/// `torch.nn.BatchNorm2d(...).train()` forward on identical input.
/// Upstream torch value (live, torch 2.11.0+cu130):
///   out[0,0,0,0] = -1.7711849212646484
///   out[1,5,3,4] =  0.4639267027378082
/// `aten/src/ATen/native/Normalization.cpp:135`
/// (`batch_norm_cpu_transform_input_template`): biased batch mean/var over
/// (N, *spatial), then y = γ·(x−μ)/sqrt(σ²+eps)+β.
/// Tracking: #1449 re-audit.
#[test]
fn divergence_batchnorm2d_gpu_train_output_vs_torch() {
    if !cuda_ready() {
        return;
    }
    let (b, c, h, w) = (2usize, 6, 4, 5);
    let n = b * c * h * w;
    let data: Vec<f32> = (0..n).map(|k| ((k % 19) as f32) * 0.11 - 1.0).collect();
    let gamma: Vec<f32> = (0..c).map(|k| 1.0 + 0.07 * (k as f32)).collect();
    let beta: Vec<f32> = (0..c).map(|k| -0.2 + 0.03 * (k as f32)).collect();
    let rmean: Vec<f32> = (0..c).map(|k| 0.05 * (k as f32) - 0.1).collect();
    let rvar: Vec<f32> = (0..c).map(|k| 0.8 + 0.05 * (k as f32)).collect();

    let mut bn = BatchNorm2d::<f32>::new(c, 1e-5, 0.1, true).unwrap();
    bn.weight
        .as_mut()
        .unwrap()
        .set_data(cpu_tensor(&gamma, &[c]));
    bn.bias.as_mut().unwrap().set_data(cpu_tensor(&beta, &[c]));
    bn.set_running_mean(&rmean).unwrap();
    bn.set_running_var(&rvar).unwrap();
    // default: training = true

    bn.to_device(Device::Cuda(0)).unwrap();
    let x = cpu_tensor(&data, &[b, c, h, w])
        .to(Device::Cuda(0))
        .unwrap();
    let y = bn.forward(&x).unwrap();
    assert!(y.is_cuda(), "GPU BatchNorm output must stay on CUDA");
    let out = y.data_vec().unwrap();

    // index [0,0,0,0] -> 0 ; index [1,5,3,4] -> ((1*c+5)*h+3)*w+4
    let idx_1_5_3_4 = ((1 * c + 5) * h + 3) * w + 4;
    let torch_0 = -1.771_184_9_f32; // torch live
    let torch_1 = 0.463_926_7_f32; // torch live
    assert!(
        (out[0] - torch_0).abs() < 1e-3,
        "train out[0,0,0,0]: torch={torch_0} ferrotorch={}",
        out[0]
    );
    assert!(
        (out[idx_1_5_3_4] - torch_1).abs() < 1e-3,
        "train out[1,5,3,4]: torch={torch_1} ferrotorch={}",
        out[idx_1_5_3_4]
    );
}

/// Trap 2: running-stats update — the classic biased-for-normalize /
/// UNBIASED (Bessel) running_var trap, plus correct momentum direction.
///
/// Divergence target: after one train forward, ferrotorch GPU
/// running_mean/running_var vs torch (live):
///   running_mean = [-0.09429999, -0.04875, -0.0032, 0.04235, 0.0879, 0.13345]
///   running_var  = [0.75804985, 0.80240142, 0.84687084, 0.89145821,
///                   0.93616349, 0.9809866]   (Bessel-corrected: n/(n-1))
/// torch updates running_var with the UNBIASED batch variance (n=b*h*w=40,
/// bessel=40/39) while normalizing with the BIASED variance. A kernel that
/// stores biased var into running_var (no Bessel) or flips momentum drifts.
/// `torch/nn/modules/batchnorm.py:191` (momentum) +
/// `aten/src/ATen/native/Normalization.cpp` (unbiased running-var save).
/// Tracking: #1449 re-audit.
#[test]
fn divergence_batchnorm2d_gpu_running_stats_vs_torch() {
    if !cuda_ready() {
        return;
    }
    let (b, c, h, w) = (2usize, 6, 4, 5);
    let n = b * c * h * w;
    let data: Vec<f32> = (0..n).map(|k| ((k % 19) as f32) * 0.11 - 1.0).collect();
    let gamma: Vec<f32> = (0..c).map(|k| 1.0 + 0.07 * (k as f32)).collect();
    let beta: Vec<f32> = (0..c).map(|k| -0.2 + 0.03 * (k as f32)).collect();
    let rmean: Vec<f32> = (0..c).map(|k| 0.05 * (k as f32) - 0.1).collect();
    let rvar: Vec<f32> = (0..c).map(|k| 0.8 + 0.05 * (k as f32)).collect();

    let mut bn = BatchNorm2d::<f32>::new(c, 1e-5, 0.1, true).unwrap();
    bn.weight
        .as_mut()
        .unwrap()
        .set_data(cpu_tensor(&gamma, &[c]));
    bn.bias.as_mut().unwrap().set_data(cpu_tensor(&beta, &[c]));
    bn.set_running_mean(&rmean).unwrap();
    bn.set_running_var(&rvar).unwrap();

    bn.to_device(Device::Cuda(0)).unwrap();
    let x = cpu_tensor(&data, &[b, c, h, w])
        .to(Device::Cuda(0))
        .unwrap();
    let _ = bn.forward(&x).unwrap();

    // torch live oracle (Bessel-corrected running_var):
    let torch_rmean: [f64; 6] = [-0.09429999, -0.04875, -0.0032, 0.04235, 0.0879, 0.13345];
    let torch_rvar: [f64; 6] = [
        0.75804985, 0.80240142, 0.84687084, 0.89145821, 0.93616349, 0.9809866,
    ];
    let rm = bn.running_mean();
    let rv = bn.running_var();
    assert_eq!(bn.num_batches_tracked(), 1, "nbt must increment");
    for k in 0..c {
        assert!(
            (rm[k] - torch_rmean[k]).abs() < 1e-4,
            "running_mean[{k}]: torch={} ferrotorch={}",
            torch_rmean[k],
            rm[k]
        );
        assert!(
            (rv[k] - torch_rvar[k]).abs() < 1e-4,
            "running_var[{k}] (UNBIASED/Bessel): torch={} ferrotorch={}",
            torch_rvar[k],
            rv[k]
        );
    }
}

/// Trap 3: eval-mode uses running stats only.
///
/// Divergence target: ferrotorch GPU eval output vs torch eval (live):
///   eval out[0,0,0,0] = -1.2062243223190308
///   eval out[1,5,3,4] =  0.029047515243291855
/// `torch/nn/functional.py:2817` selects running_mean/running_var when not
/// training.
/// Tracking: #1449 re-audit.
#[test]
fn divergence_batchnorm2d_gpu_eval_output_vs_torch() {
    if !cuda_ready() {
        return;
    }
    let (b, c, h, w) = (2usize, 6, 4, 5);
    let n = b * c * h * w;
    let data: Vec<f32> = (0..n).map(|k| ((k % 19) as f32) * 0.11 - 1.0).collect();
    let gamma: Vec<f32> = (0..c).map(|k| 1.0 + 0.07 * (k as f32)).collect();
    let beta: Vec<f32> = (0..c).map(|k| -0.2 + 0.03 * (k as f32)).collect();
    let rmean: Vec<f32> = (0..c).map(|k| 0.05 * (k as f32) - 0.1).collect();
    let rvar: Vec<f32> = (0..c).map(|k| 0.8 + 0.05 * (k as f32)).collect();

    let mut bn = BatchNorm2d::<f32>::new(c, 1e-5, 0.1, true).unwrap();
    bn.weight
        .as_mut()
        .unwrap()
        .set_data(cpu_tensor(&gamma, &[c]));
    bn.bias.as_mut().unwrap().set_data(cpu_tensor(&beta, &[c]));
    bn.set_running_mean(&rmean).unwrap();
    bn.set_running_var(&rvar).unwrap();
    bn.eval();

    bn.to_device(Device::Cuda(0)).unwrap();
    let x = cpu_tensor(&data, &[b, c, h, w])
        .to(Device::Cuda(0))
        .unwrap();
    let y = bn.forward(&x).unwrap();
    let out = y.data_vec().unwrap();

    let idx_1_5_3_4 = ((1 * c + 5) * h + 3) * w + 4;
    let torch_0 = -1.206_224_3_f32;
    let torch_1 = 0.029_047_515_f32;
    assert!(
        (out[0] - torch_0).abs() < 1e-3,
        "eval out[0,0,0,0]: torch={torch_0} ferrotorch={}",
        out[0]
    );
    assert!(
        (out[idx_1_5_3_4] - torch_1).abs() < 1e-3,
        "eval out[1,5,3,4]: torch={torch_1} ferrotorch={}",
        out[idx_1_5_3_4]
    );
}

/// Trap 2 (compounding): running-stat drift across MULTIPLE train forwards.
///
/// A single-step running-stat match can mask a subtle bug (e.g. tiny Bessel or
/// momentum error) that only diverges after accumulation. This runs 3 train
/// forwards and pins the accumulated running stats to torch (live):
///   running_mean(3) = [0.0012403130531311035, 0.003442188259214163,
///                      0.005644062999635935]
///   running_var(3)  = [0.8420900106430054, 0.844513475894928,
///                      0.8463091254234314]
/// `torch/nn/modules/batchnorm.py:189-194`.
/// Tracking: #1449 re-audit.
#[test]
fn divergence_batchnorm2d_gpu_multistep_running_stats_vs_torch() {
    if !cuda_ready() {
        return;
    }
    let c = 3usize;
    let (b, h, w) = (2usize, 4, 4);
    let mut bn = BatchNorm2d::<f32>::new(c, 1e-5, 0.1, true).unwrap();
    bn.to_device(Device::Cuda(0)).unwrap();

    for step in 0..3usize {
        let n = b * c * h * w;
        let data: Vec<f32> = (0..n)
            .map(|k| (((step * 100 + k) % 17) as f32) * 0.13 - 1.0)
            .collect();
        let x = cpu_tensor(&data, &[b, c, h, w])
            .to(Device::Cuda(0))
            .unwrap();
        let _ = bn.forward(&x).unwrap();
    }

    let torch_rmean: [f64; 3] = [
        0.001_240_313_053_131_103_5,
        0.003_442_188_259_214_163,
        0.005_644_062_999_635_935,
    ];
    let torch_rvar: [f64; 3] = [
        0.842_090_010_643_005_4,
        0.844_513_475_894_928,
        0.846_309_125_423_431_4,
    ];
    let rm = bn.running_mean();
    let rv = bn.running_var();
    assert_eq!(bn.num_batches_tracked(), 3);
    for k in 0..c {
        assert!(
            (rm[k] - torch_rmean[k]).abs() < 1e-5,
            "3-step running_mean[{k}]: torch={} ferrotorch={}",
            torch_rmean[k],
            rm[k]
        );
        assert!(
            (rv[k] - torch_rvar[k]).abs() < 1e-5,
            "3-step running_var[{k}]: torch={} ferrotorch={}",
            torch_rvar[k],
            rv[k]
        );
    }
}

/// Trap 5: InstanceNorm2d per-(N,C) normalization over spatial dims.
///
/// Divergence target: ferrotorch GPU InstanceNorm2d vs
/// `torch.nn.InstanceNorm2d(affine=True)` (live):
///   out[0,0,0,0] = -1.6974784135818481
///   out[2,3,4,3] = -0.2620939612388611
/// `torch/nn/modules/instancenorm.py` — per-instance per-channel norm.
/// Sets affine params via the public `parameters_mut()` (order: [weight, bias]
/// for affine InstanceNorm2d, per `InstanceNorm2d::parameters_mut`).
/// Tracking: #1449 re-audit.
#[test]
fn divergence_instancenorm2d_gpu_output_vs_torch() {
    if !cuda_ready() {
        return;
    }
    let (b, c, h, w) = (3usize, 4, 5, 4);
    let n = b * c * h * w;
    let data: Vec<f32> = (0..n).map(|k| ((k % 23) as f32) * 0.09 - 0.8).collect();
    let gamma: Vec<f32> = (0..c).map(|k| 1.0 + 0.06 * (k as f32)).collect();
    let beta: Vec<f32> = (0..c).map(|k| -0.05 + 0.04 * (k as f32)).collect();

    let mut inorm = InstanceNorm2d::<f32>::new(c, 1e-5, true).unwrap();
    {
        let mut params = inorm.parameters_mut();
        // affine InstanceNorm2d::parameters_mut yields [weight, bias].
        assert_eq!(params.len(), 2, "affine InstanceNorm2d has weight+bias");
        params[0].set_data(cpu_tensor(&gamma, &[c]));
        params[1].set_data(cpu_tensor(&beta, &[c]));
    }

    inorm.to_device(Device::Cuda(0)).unwrap();
    let x = cpu_tensor(&data, &[b, c, h, w])
        .to(Device::Cuda(0))
        .unwrap();
    let y = inorm.forward(&x).unwrap();
    assert!(y.is_cuda(), "InstanceNorm GPU output must stay on CUDA");
    let out = y.data_vec().unwrap();

    let idx_0_0_0_0 = 0usize;
    let idx_2_3_4_3 = ((2 * c + 3) * h + 4) * w + 3;
    let torch_0 = -1.697_478_4_f32;
    let torch_1 = -0.262_093_96_f32;
    assert!(
        (out[idx_0_0_0_0] - torch_0).abs() < 1e-3,
        "InstanceNorm out[0,0,0,0]: torch={torch_0} ferrotorch={}",
        out[idx_0_0_0_0]
    );
    assert!(
        (out[idx_2_3_4_3] - torch_1).abs() < 1e-3,
        "InstanceNorm out[2,3,4,3]: torch={torch_1} ferrotorch={}",
        out[idx_2_3_4_3]
    );
}

// ===========================================================================
// GPU BACKWARD probes (#1449). The forward critic above only pinned the
// forward; these pin the on-GPU gradient VALUES (grad_input / grad_weight /
// grad_bias) directly against torch's autograd oracle (live, torch
// 2.11.0+cu130 on this host's RTX 3090). The key anti-pattern the user warns
// about is a GPU backward that compiles but produces wrong gradients — these
// FD-equivalent oracle checks catch exactly that. They also assert the input
// gradient stays GPU-resident (`is_cuda`), pinning the NO-CPU-round-trip
// contract (R-CODE-4): a backward that silently round-tripped through `.cpu()`
// would surface here as a non-CUDA grad tensor.
//
// Oracle capture (R-CHAR-3 — values come from torch autograd, NOT ferrotorch):
// ```text
// bn = torch.nn.BatchNorm2d(c, eps=1e-5, momentum=0.1, affine=True).cuda()
// x.requires_grad_(); y = bn(x); y.backward(go)  # go is a fixed tensor
// ```

/// Build the fixed upstream-gradient tensor `go[k] = (k%7)*0.05 - 0.1`.
fn make_go(n: usize) -> Vec<f32> {
    (0..n).map(|k| ((k % 7) as f32) * 0.05 - 0.1).collect()
}

/// BatchNorm2d TRAIN-mode backward: grad_input / grad_weight / grad_bias on GPU
/// vs. torch autograd. Upstream torch values (live, torch 2.11.0+cu130):
///   grad_input[0]   = -0.19677015   grad_input[239] = -0.22319010
///   grad_weight[0]  =  0.76753289   grad_weight[5]  = -0.61141950
///   grad_bias[0]    =  2.00000000   grad_bias[5]    =  2.15000010
/// `aten/src/ATen/native/cuda/Normalization.cuh:388 batch_norm_backward_kernel`
/// (train branch: grad_input = (go - (x-mean)*proj_scale - grad_mean)*grad_scale).
#[test]
fn divergence_batchnorm2d_gpu_train_backward_vs_torch() {
    if !cuda_ready() {
        return;
    }
    let (b, c, h, w) = (2usize, 6, 4, 5);
    let n = b * c * h * w;
    let data: Vec<f32> = (0..n).map(|k| ((k % 19) as f32) * 0.11 - 1.0).collect();
    let gamma: Vec<f32> = (0..c).map(|k| 1.0 + 0.07 * (k as f32)).collect();
    let beta: Vec<f32> = (0..c).map(|k| -0.2 + 0.03 * (k as f32)).collect();

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
    let go = cpu_tensor(&make_go(n), &[b, c, h, w])
        .to(Device::Cuda(0))
        .unwrap();
    backward_with_grad(&y, Some(&go)).unwrap();

    let gi = x.grad().unwrap().expect("grad_input populated");
    assert!(
        gi.is_cuda(),
        "grad_input must stay on CUDA (no .cpu() round trip)"
    );
    let gid = gi.data_vec().unwrap();
    let idx_239 = ((1 * c + 5) * h + 3) * w + 4;
    for (idx, torch) in [(0usize, -0.196_770_15_f32), (idx_239, -0.223_190_10_f32)] {
        assert!(
            (gid[idx] - torch).abs() < 1e-3,
            "BN2d train grad_input[{idx}]: torch={torch} ferrotorch={}",
            gid[idx]
        );
    }

    let gw = bn
        .weight
        .as_ref()
        .unwrap()
        .tensor()
        .grad()
        .unwrap()
        .expect("grad_weight populated");
    assert!(gw.is_cuda(), "grad_weight must stay on CUDA");
    let gwd = gw.data_vec().unwrap();
    for (idx, torch) in [(0usize, 0.767_532_89_f32), (5usize, -0.611_419_50_f32)] {
        assert!(
            (gwd[idx] - torch).abs() < 1e-3,
            "BN2d train grad_weight[{idx}]: torch={torch} ferrotorch={}",
            gwd[idx]
        );
    }

    let gb = bn
        .bias
        .as_ref()
        .unwrap()
        .tensor()
        .grad()
        .unwrap()
        .expect("grad_bias populated");
    assert!(gb.is_cuda(), "grad_bias must stay on CUDA");
    let gbd = gb.data_vec().unwrap();
    for (idx, torch) in [(0usize, 2.0_f32), (5usize, 2.150_000_1_f32)] {
        assert!(
            (gbd[idx] - torch).abs() < 1e-3,
            "BN2d train grad_bias[{idx}]: torch={torch} ferrotorch={}",
            gbd[idx]
        );
    }
}

/// BatchNorm2d EVAL-mode backward (running stats) vs. torch. Upstream torch:
///   grad_input[0]  = -0.11180270   grad_input[239] = -0.06587294
///   grad_weight[0] =  0.65013272   grad_weight[5]  = -0.70069295
/// `Normalization.cuh:388` eval branch: grad_input = go * invstd * weight,
/// invstd = 1/sqrt(running_var + eps).
#[test]
fn divergence_batchnorm2d_gpu_eval_backward_vs_torch() {
    if !cuda_ready() {
        return;
    }
    let (b, c, h, w) = (2usize, 6, 4, 5);
    let n = b * c * h * w;
    let data: Vec<f32> = (0..n).map(|k| ((k % 19) as f32) * 0.11 - 1.0).collect();
    let gamma: Vec<f32> = (0..c).map(|k| 1.0 + 0.07 * (k as f32)).collect();
    let beta: Vec<f32> = (0..c).map(|k| -0.2 + 0.03 * (k as f32)).collect();
    let rmean: Vec<f32> = (0..c).map(|k| 0.05 * (k as f32) - 0.1).collect();
    let rvar: Vec<f32> = (0..c).map(|k| 0.8 + 0.05 * (k as f32)).collect();

    let mut bn = BatchNorm2d::<f32>::new(c, 1e-5, 0.1, true).unwrap();
    bn.weight
        .as_mut()
        .unwrap()
        .set_data(cpu_tensor(&gamma, &[c]));
    bn.bias.as_mut().unwrap().set_data(cpu_tensor(&beta, &[c]));
    bn.set_running_mean(&rmean).unwrap();
    bn.set_running_var(&rvar).unwrap();
    bn.eval();
    bn.to_device(Device::Cuda(0)).unwrap();

    let x = cpu_tensor(&data, &[b, c, h, w])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let y = bn.forward(&x).unwrap();
    let go = cpu_tensor(&make_go(n), &[b, c, h, w])
        .to(Device::Cuda(0))
        .unwrap();
    backward_with_grad(&y, Some(&go)).unwrap();

    let gi = x.grad().unwrap().expect("grad_input populated");
    assert!(gi.is_cuda(), "grad_input must stay on CUDA");
    let gid = gi.data_vec().unwrap();
    let idx_239 = ((1 * c + 5) * h + 3) * w + 4;
    for (idx, torch) in [(0usize, -0.111_802_70_f32), (idx_239, -0.065_872_94_f32)] {
        assert!(
            (gid[idx] - torch).abs() < 1e-3,
            "BN2d eval grad_input[{idx}]: torch={torch} ferrotorch={}",
            gid[idx]
        );
    }

    let gw = bn
        .weight
        .as_ref()
        .unwrap()
        .tensor()
        .grad()
        .unwrap()
        .expect("grad_weight populated");
    let gwd = gw.data_vec().unwrap();
    for (idx, torch) in [(0usize, 0.650_132_72_f32), (5usize, -0.700_692_95_f32)] {
        assert!(
            (gwd[idx] - torch).abs() < 1e-3,
            "BN2d eval grad_weight[{idx}]: torch={torch} ferrotorch={}",
            gwd[idx]
        );
    }
}

/// InstanceNorm2d (affine) backward vs. torch. InstanceNorm reuses the
/// BatchNorm backward kernel via the `[1, B*C, S]` reshape (instance stats).
/// Upstream torch values:
///   grad_input[0]   = -0.18283656   grad_input[357] = 0.03581836
///   grad_weight     = [ 2.06521964, -0.00224167, -0.89685678, -0.52081096 ]
///   grad_bias       = [ 3.41999984,  4.50000000,  5.57999992,  4.01999998 ]
/// `torch/nn/functional.py instance_norm` (lowers to the per-instance
/// batch-norm reduction).
#[test]
fn divergence_instancenorm2d_gpu_backward_vs_torch() {
    if !cuda_ready() {
        return;
    }
    let (b, c, h, w) = (3usize, 4, 5, 6);
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
    let gid = gi.data_vec().unwrap();
    let idx_357 = ((2 * c + 3) * h + 4) * w + 3;
    for (idx, torch) in [(0usize, -0.182_836_56_f32), (idx_357, 0.035_818_36_f32)] {
        assert!(
            (gid[idx] - torch).abs() < 1e-3,
            "InstanceNorm grad_input[{idx}]: torch={torch} ferrotorch={}",
            gid[idx]
        );
    }

    let params = inorm.named_parameters();
    let weight = params.iter().find(|(n, _)| n == "weight").unwrap().1;
    let gw = weight
        .tensor()
        .grad()
        .unwrap()
        .expect("grad_weight populated");
    assert!(gw.is_cuda(), "InstanceNorm grad_weight must stay on CUDA");
    let gwd = gw.data_vec().unwrap();
    let torch_gw = [2.065_219_6_f32, -0.002_241_67, -0.896_856_78, -0.520_810_96];
    for (idx, &torch) in torch_gw.iter().enumerate() {
        assert!(
            (gwd[idx] - torch).abs() < 1e-3,
            "InstanceNorm grad_weight[{idx}]: torch={torch} ferrotorch={}",
            gwd[idx]
        );
    }
    let bias = params.iter().find(|(n, _)| n == "bias").unwrap().1;
    let gb = bias.tensor().grad().unwrap().expect("grad_bias populated");
    let gbd = gb.data_vec().unwrap();
    let torch_gb = [3.419_999_8_f32, 4.5, 5.579_999_9, 4.02];
    for (idx, &torch) in torch_gb.iter().enumerate() {
        assert!(
            (gbd[idx] - torch).abs() < 1e-3,
            "InstanceNorm grad_bias[{idx}]: torch={torch} ferrotorch={}",
            gbd[idx]
        );
    }
}

/// LocalResponseNorm forward + backward on GPU vs. torch. Upstream torch
/// (size=5, alpha=1e-4, beta=0.75, k=2.0):
///   out[0]   = -0.59459764   out[191] = -0.28540772
///   grad_input[0]   = -0.11891798   grad_input[191] = -0.04756723
///   grad_input[53]  =  0.16648498
/// `torch/nn/functional.py:3032-3046 local_response_norm` (square →
/// avg_pool-over-channels → `*alpha + k` → pow(beta) → divide).
#[test]
fn divergence_local_response_norm_gpu_fwd_bwd_vs_torch() {
    if !cuda_ready() {
        return;
    }
    let (b, c, h, w) = (2usize, 8, 3, 4);
    let n = b * c * h * w;
    let data: Vec<f32> = (0..n).map(|k| ((k % 17) as f32) * 0.13 - 1.0).collect();
    let go: Vec<f32> = (0..n).map(|k| ((k % 9) as f32) * 0.06 - 0.2).collect();

    let lrn = LocalResponseNorm::new(5, 1e-4, 0.75, 2.0).unwrap();

    let x = cpu_tensor(&data, &[b, c, h, w])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let y: Tensor<f32> = lrn.forward(&x).unwrap();
    assert!(y.is_cuda(), "LRN GPU forward output must stay on CUDA");
    let out = y.data_vec().unwrap();
    let idx_191 = ((1 * c + 7) * h + 2) * w + 3;
    let idx_53 = ((0 * c + 4) * h + 1) * w + 1;
    for (idx, torch) in [(0usize, -0.594_597_64_f32), (idx_191, -0.285_407_72_f32)] {
        assert!(
            (out[idx] - torch).abs() < 1e-3,
            "LRN out[{idx}]: torch={torch} ferrotorch={}",
            out[idx]
        );
    }

    let go_t = cpu_tensor(&go, &[b, c, h, w]).to(Device::Cuda(0)).unwrap();
    backward_with_grad(&y, Some(&go_t)).unwrap();
    let gi = x.grad().unwrap().expect("grad_input populated");
    assert!(
        gi.is_cuda(),
        "LRN grad_input must stay on CUDA (no .cpu() round trip)"
    );
    let gid = gi.data_vec().unwrap();
    for (idx, torch) in [
        (0usize, -0.118_917_98_f32),
        (idx_191, -0.047_567_23_f32),
        (idx_53, 0.166_484_98_f32),
    ] {
        assert!(
            (gid[idx] - torch).abs() < 1e-3,
            "LRN grad_input[{idx}]: torch={torch} ferrotorch={}",
            gid[idx]
        );
    }
}
