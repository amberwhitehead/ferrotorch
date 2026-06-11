//! Regression tests for audit finding CORE-045 (crosslink #1739).
//!
//! `hardtanh`/`relu6`, `hardsigmoid`, `hardswish`, `selu`, `softsign`, and
//! `prelu` computed their CUDA forward through `unary_map` (host round trip,
//! output restored to CUDA) but, when the input required gradients, rebuilt
//! the output storage via the CPU-only `output.data()` — so the SAME valid
//! forward failed with `GpuTensorNotAccessible` solely because
//! `requires_grad` was enabled. Their backwards likewise called `data()`
//! unconditionally.
//!
//! Post-fix contract (the `leaky_relu` pattern at `activation.rs`): the
//! grad-enabled forward consumes `unary_map`'s device-resident storage, and
//! the backward builds the derivative mask via `unary_map` (documented host
//! round trip per R-LOUD-2) multiplied by `grad_output` with the CUDA-aware
//! `arithmetic::mul`.
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
use ferrotorch_core::{Device, FerrotorchResult, Tensor, TensorStorage};

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
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
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
        // Tolerance rationale (R-ORACLE-5): hardtanh/relu6/hardsigmoid/
        // hardswish/softsign/prelu are pure rational arithmetic — but the
        // values transit one f64 D2H/H2D round trip (lossless) and selu's
        // negative branch evaluates `exp` once (CPU libm on the host round
        // trip, ≤ 1 ulp ≈ 2.3e-16 relative at |x| ≤ 4.3). 1e-12 absolute
        // gives > 3 orders of margin on these O(1) magnitudes.
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
/// [0.16666666666666666, 0.4166666666666667, 0.5, 0.5833333333333334,
///  0.8333333333333334, 1.0]
/// >>> t.grad  # f64 grads are exactly 1/6 inside (-3, 3)
/// [1/6, 1/6, 1/6, 1/6, 1/6, 0.0]
/// ```
/// (torch's printed f64 grad is the f32-rounded 0.1666666716337204 — a
/// documented upstream artifact, see closed #795; ferrotorch computes the
/// exact 1/6, inside the 1e-12 band vs 1.0/6.0.)
#[test]
fn cuda_grad_hardsigmoid() {
    let sixth = 1.0 / 6.0;
    run_cuda_grad_lane(
        "hardsigmoid",
        hardsigmoid,
        &[
            0.166_666_666_666_666_66,
            0.416_666_666_666_666_7,
            0.5,
            0.583_333_333_333_333_4,
            0.833_333_333_333_333_4,
            1.0,
        ],
        &[sixth, sixth, sixth, sixth, sixth, 0.0],
    );
}

/// ```python
/// >>> o = F.hardswish(t)
/// [-0.3333333333333333, -0.20833333333333334, 0.0, 0.2916666666666667,
///  1.6666666666666667, 4.0]
/// [-0.16666666666666663, 0.33333333333333337, 0.5, 0.6666666666666666,
///  1.1666666666666665, 1.0]
/// ```
#[test]
fn cuda_grad_hardswish() {
    run_cuda_grad_lane(
        "hardswish",
        hardswish,
        &[
            -0.333_333_333_333_333_3,
            -0.208_333_333_333_333_34,
            0.0,
            0.291_666_666_666_666_7,
            1.666_666_666_666_666_7,
            4.0,
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

/// prelu with a CPU-resident scalar alpha and CUDA input: grad_x stays on
/// CUDA, grad_alpha lands on alpha's own (CPU) device.
#[test]
fn cuda_grad_prelu_cpu_alpha() {
    ensure_cuda_backend();
    let x = cuda_f64(&X, true);
    let alpha = Tensor::from_storage(TensorStorage::cpu(vec![0.25f64]), vec![1], true).unwrap();
    let out = prelu(&x, &alpha).unwrap_or_else(|e| {
        panic!("prelu (cpu alpha): CUDA forward with requires_grad=true failed: {e:?}")
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
    let ga = alpha.grad().unwrap().expect("grad_alpha must reach alpha");
    assert_eq!(
        ga.device(),
        Device::Cpu,
        "grad_alpha must live on alpha's device (CPU)"
    );
    assert_close(&ga.data_vec().unwrap(), &[-2.5], "prelu grad_alpha");
}
