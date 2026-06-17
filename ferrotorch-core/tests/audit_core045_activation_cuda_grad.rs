//! Regression tests for audit finding CORE-045 (crosslink #1739).
//!
//! `hardtanh`/`relu6`, `hardsigmoid`, `hardswish`, `selu`, `softsign`, and
//! `prelu` used to depend on CPU-only closure paths for CUDA forwards or
//! backwards. Once `unary_map` correctly rejected CUDA tensors, those paths
//! failed with `NotImplementedOnCuda`/`GpuTensorNotAccessible`.
//!
//! Post-fix contract: f32/f64 CUDA forwards and VJPs below are resident CUDA
//! dispatches. No derivative mask is built through `unary_map`, and PReLU's
//! alpha VJP is a device-side map-reduce.
//!
//! Every expectation below is from a live torch session (R-ORACLE-1(b)):
//!
//! ```python
//! # torch 2.11.0+cu130
//! >>> x = [-2.0, -0.5, 0.0, 0.5, 2.0, 4.0]
//! >>> t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
//! >>> o = OP(t); o.sum().backward()   # fwd/grad pairs quoted per test
//! ```
//!
//! Device assertions follow R-ORACLE-3: the result AND the gradient must
//! reside on `Cuda(0)` — readback transparency is not accepted.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::grad_fns::activation::{
    hardsigmoid, hardswish, hardtanh, prelu, relu6, selu, softsign,
};
use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::{Device, FerrotorchError, FerrotorchResult, Tensor, TensorStorage};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the CORE-045 regression suite");
    });
}

const X: [f64; 6] = [-2.0, -0.5, 0.0, 0.5, 2.0, 4.0];

/// Build a true CUDA LEAF.
///
/// CORE-012 (#1706): `.to(device)` of a requires-grad leaf is a
/// differentiable copy — a NON-leaf whose gradients accumulate on the
/// ORIGINAL CPU leaf (torch: `is_leaf=False`, grad_fn `ToCopyBackward0`).
/// These tests assert CUDA-resident `.grad()` on the uploaded tensor, so
/// they need a real CUDA leaf — torch's
/// `x.to('cuda').detach().requires_grad_(True)` idiom.
fn cuda_f64(data: &[f64], rg: bool) -> Tensor<f64> {
    cuda_f64_shape(data, &[data.len()], rg)
}

fn cuda_f64_shape(data: &[f64], shape: &[usize], rg: bool) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .unwrap()
        .to(Device::Cuda(0))
        .expect("upload to cuda:0")
        .requires_grad_(rg)
}

fn read_back(t: &Tensor<f64>, what: &str) -> Vec<f64> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "{what} expected on Cuda(0) but resides on {:?} — silent CPU fallback",
        t.device()
    );
    t.cpu().expect("D2H readback").data_vec().expect("read")
}

fn assert_close(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        // Tolerance rationale (R-ORACLE-5): most ops here are branch/rational
        // arithmetic; SELU uses one `exp`; Hardsigmoid/Hardswish CUDA f64
        // intentionally mirror PyTorch's f32-rounded one-sixth constant.
        assert!(
            (a - e).abs() < 1e-12,
            "{label}: index {i} diverges (actual={a:.17}, expected={e:.17})"
        );
    }
}

type OpFn = fn(&Tensor<f64>) -> FerrotorchResult<Tensor<f64>>;

/// Shared lane: CUDA forward with requires_grad=true must succeed (this is
/// the exact call that returned `GpuTensorNotAccessible` pre-fix), match the
/// torch oracle, and the gradient must flow back to the CUDA leaf with
/// oracle values — result and grad both device-asserted.
fn run_cuda_grad_lane(name: &str, op: OpFn, fwd_exp: &[f64], grad_exp: &[f64]) {
    ensure_cuda_backend();
    let x = cuda_f64(&X, true);
    let out = op(&x).unwrap_or_else(|e| {
        panic!("{name}: CUDA forward with requires_grad=true failed (CORE-045): {e:?}")
    });
    assert_close(&read_back(&out, &format!("{name} fwd")), fwd_exp, name);

    let loss = reduce_sum(&out).expect("sum-to-scalar");
    loss.backward().expect("backward");
    let g = x.grad().unwrap().expect("grad must reach the CUDA leaf");
    assert_close(
        &read_back(&g, &format!("{name} grad")),
        grad_exp,
        &format!("{name} grad"),
    );
}

/// ```python
/// >>> o = F.hardtanh(t)  # fwd, then grad
/// [-1.0, -0.5, 0.0, 0.5, 1.0, 1.0]
/// [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
/// ```
#[test]
fn cuda_grad_hardtanh() {
    run_cuda_grad_lane(
        "hardtanh",
        hardtanh,
        &[-1.0, -0.5, 0.0, 0.5, 1.0, 1.0],
        &[0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    );
}

/// ```python
/// >>> o = F.relu6(t)
/// [0.0, 0.0, 0.0, 0.5, 2.0, 4.0]
/// [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
/// ```
#[test]
fn cuda_grad_relu6() {
    run_cuda_grad_lane(
        "relu6",
        relu6,
        &[0.0, 0.0, 0.0, 0.5, 2.0, 4.0],
        &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    );
}

/// ```python
/// >>> o = F.hardsigmoid(t)
/// [0.1666666716337204, 0.416666679084301, 0.5000000149011612,
///  0.5833333507180214, 0.833333358168602, 1.0000000298023224]
/// >>> t.grad
/// [0.1666666716337204, 0.1666666716337204, 0.1666666716337204,
///  0.1666666716337204, 0.1666666716337204, 0.0]
/// ```
#[test]
fn cuda_grad_hardsigmoid() {
    let sixth = 0.166_666_671_633_720_4;
    run_cuda_grad_lane(
        "hardsigmoid",
        hardsigmoid,
        &[
            0.166_666_671_633_720_4,
            0.416_666_679_084_301,
            0.500_000_014_901_161_2,
            0.583_333_350_718_021_4,
            0.833_333_358_168_602,
            1.000_000_029_802_322_4,
        ],
        &[sixth, sixth, sixth, sixth, sixth, 0.0],
    );
}

/// ```python
/// >>> o = F.hardswish(t)
/// [-0.3333333432674408, -0.2083333395421505, 0.0, 0.2916666753590107,
///  1.666666716337204, 4.0000001192092896]
/// [-0.16666666666666663, 0.33333333333333337, 0.5, 0.6666666666666666,
///  1.1666666666666665, 1.0]
/// ```
#[test]
fn cuda_grad_hardswish() {
    run_cuda_grad_lane(
        "hardswish",
        hardswish,
        &[
            -0.333_333_343_267_440_8,
            -0.208_333_339_542_150_5,
            0.0,
            0.291_666_675_359_010_7,
            1.666_666_716_337_204,
            4.000_000_119_209_289_6,
        ],
        &[
            -0.166_666_666_666_666_63,
            0.333_333_333_333_333_37,
            0.5,
            0.666_666_666_666_666_6,
            1.166_666_666_666_666_5,
            1.0,
        ],
    );
}

/// ```python
/// >>> o = F.selu(t)
/// [-1.5201664685956948, -0.6917581878028713, 0.0, 0.5253504936777402,
///  2.101401974710961, 4.202803949421922]
/// [0.2379328722516818, 1.0663411530445053, 1.7580993408473766,
///  1.0507009873554805, 1.0507009873554805, 1.0507009873554805]
/// ```
#[test]
fn cuda_grad_selu() {
    run_cuda_grad_lane(
        "selu",
        selu,
        &[
            -1.520_166_468_595_694_8,
            -0.691_758_187_802_871_3,
            0.0,
            0.525_350_493_677_740_2,
            2.101_401_974_710_961,
            4.202_803_949_421_922,
        ],
        &[
            0.237_932_872_251_681_8,
            1.066_341_153_044_505_3,
            1.758_099_340_847_376_6,
            1.050_700_987_355_480_5,
            1.050_700_987_355_480_5,
            1.050_700_987_355_480_5,
        ],
    );
}

/// ```python
/// >>> o = F.softsign(t)
/// [-0.6666666666666666, -0.3333333333333333, 0.0, 0.3333333333333333,
///  0.6666666666666666, 0.8]
/// [0.1111111111111111, 0.4444444444444444, 1.0, 0.4444444444444444,
///  0.1111111111111111, 0.04000000000000001]
/// ```
#[test]
fn cuda_grad_softsign() {
    run_cuda_grad_lane(
        "softsign",
        softsign,
        &[
            -0.666_666_666_666_666_6,
            -0.333_333_333_333_333_3,
            0.0,
            0.333_333_333_333_333_3,
            0.666_666_666_666_666_6,
            0.8,
        ],
        &[
            0.111_111_111_111_111_1,
            0.444_444_444_444_444_4,
            1.0,
            0.444_444_444_444_444_4,
            0.111_111_111_111_111_1,
            0.040_000_000_000_000_01,
        ],
    );
}

/// prelu with both input AND scalar alpha resident on CUDA (the torch-
/// realistic configuration — torch requires weight on the input's device):
/// ```python
/// >>> alpha = torch.tensor([0.25], dtype=torch.float64, requires_grad=True)
/// >>> o = F.prelu(t, alpha); o.sum().backward()
/// fwd        = [-0.5, -0.125, 0.0, 0.5, 2.0, 4.0]
/// grad_x     = [0.25, 0.25, 0.25, 1.0, 1.0, 1.0]
/// grad_alpha = [-2.5]
/// ```
#[test]
fn cuda_grad_prelu_dual_vjp() {
    ensure_cuda_backend();
    let x = cuda_f64(&X, true);
    let alpha = cuda_f64(&[0.25], true);
    let out = prelu(&x, &alpha).unwrap_or_else(|e| {
        panic!("prelu: CUDA forward with requires_grad=true failed (CORE-045): {e:?}")
    });
    assert_close(
        &read_back(&out, "prelu fwd"),
        &[-0.5, -0.125, 0.0, 0.5, 2.0, 4.0],
        "prelu fwd",
    );
    reduce_sum(&out).expect("sum").backward().expect("backward");
    let gx = x.grad().unwrap().expect("grad_x must reach the CUDA leaf");
    assert_close(
        &read_back(&gx, "prelu grad_x"),
        &[0.25, 0.25, 0.25, 1.0, 1.0, 1.0],
        "prelu grad_x",
    );
    let ga = alpha
        .grad()
        .unwrap()
        .expect("grad_alpha must reach the CUDA alpha leaf");
    assert_close(
        &read_back(&ga, "prelu grad_alpha"),
        &[-2.5],
        "prelu grad_alpha",
    );
}

/// PyTorch per-channel PReLU accepts a 1D weight matching channel dim 1:
/// ```python
/// >>> x = torch.tensor([...], device='cuda', dtype=torch.float64,
/// ...                  requires_grad=True).reshape(2, 3, 2)
/// >>> w = torch.tensor([0.1, 0.2, 0.3], device='cuda', dtype=torch.float64,
/// ...                  requires_grad=True)
/// >>> y = F.prelu(x, w); y.sum().backward()
/// fwd    = [-0.2, 1.0, -0.6, 4.0, 0.0, -1.5,
///           6.0, -0.7, -1.6, 9.0, 10.0, -3.3]
/// grad_x = [0.1, 1.0, 0.2, 1.0, 0.3, 0.3,
///           1.0, 0.1, 0.2, 1.0, 1.0, 0.3]
/// grad_w = [-9.0, -11.0, -16.0]
/// ```
#[test]
fn cuda_grad_prelu_channel_dual_vjp() {
    ensure_cuda_backend();
    let x = cuda_f64_shape(
        &[
            -2.0, 1.0, -3.0, 4.0, 0.0, -5.0, 6.0, -7.0, -8.0, 9.0, 10.0, -11.0,
        ],
        &[2, 3, 2],
        true,
    );
    let alpha = cuda_f64(&[0.1, 0.2, 0.3], true);
    let out = prelu(&x, &alpha).unwrap_or_else(|e| {
        panic!("prelu channel: CUDA forward with requires_grad=true failed: {e:?}")
    });
    assert_close(
        &read_back(&out, "prelu channel fwd"),
        &[
            -0.2, 1.0, -0.6, 4.0, 0.0, -1.5, 6.0, -0.7, -1.6, 9.0, 10.0, -3.3,
        ],
        "prelu channel fwd",
    );

    reduce_sum(&out).expect("sum").backward().expect("backward");
    let gx = x.grad().unwrap().expect("grad_x must reach the CUDA leaf");
    assert_close(
        &read_back(&gx, "prelu channel grad_x"),
        &[0.1, 1.0, 0.2, 1.0, 0.3, 0.3, 1.0, 0.1, 0.2, 1.0, 1.0, 0.3],
        "prelu channel grad_x",
    );
    let ga = alpha
        .grad()
        .unwrap()
        .expect("grad_alpha must reach the CUDA alpha leaf");
    assert_close(
        &read_back(&ga, "prelu channel grad_alpha"),
        &[-9.0, -11.0, -16.0],
        "prelu channel grad_alpha",
    );
}

#[test]
fn cuda_grad_prelu_scalar_alpha_nan_matches_torch_branch() {
    ensure_cuda_backend();
    let x = cuda_f64(&[f64::NAN, -2.0, 1.0, 0.0], true);
    let alpha = cuda_f64(&[0.5], true);
    let out = prelu(&x, &alpha).expect("prelu scalar nan forward");
    reduce_sum(&out).expect("sum").backward().expect("backward");
    let ga = alpha
        .grad()
        .unwrap()
        .expect("grad_alpha must reach the CUDA alpha leaf");
    let got = read_back(&ga, "prelu scalar nan grad_alpha");
    assert!(
        got[0].is_nan(),
        "torch prelu backward treats NaN input as the false x>0 branch"
    );
}

/// PyTorch requires PReLU input and weight on the same device. A CPU-resident
/// scalar alpha with CUDA input must be rejected rather than handled through a
/// host readback.
#[test]
fn cuda_grad_prelu_cpu_alpha() {
    ensure_cuda_backend();
    let x = cuda_f64(&X, true);
    let alpha = Tensor::from_storage(TensorStorage::cpu(vec![0.25f64]), vec![1], true).unwrap();
    let err = prelu(&x, &alpha).expect_err("cpu alpha with cuda input must fail");
    assert!(
        matches!(err, FerrotorchError::DeviceMismatch { .. }),
        "expected DeviceMismatch for mixed-device PReLU, got {err:?}"
    );
}
