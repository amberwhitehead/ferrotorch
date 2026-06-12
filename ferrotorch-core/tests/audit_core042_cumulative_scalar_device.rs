//! CORE-042 (#1736, CLASS-S High) regression battery: the differentiable
//! cumulative wrappers' 0-D (scalar) fast paths (`cumsum`, `cumprod`,
//! `cummax`, `cummin`, `logcumsumexp`) must
//!
//!   1. preserve device residency — a CUDA scalar in yields a CUDA scalar
//!      out (the identity value is read via a documented host round trip
//!      and re-uploaded, R-LOUD-2); non-scalar CUDA inputs already ran
//!      device-resident kernels;
//!   2. deliver the identity backward gradient on the leaf's device
//!      (R-ORACLE-3);
//!   3. keep the 0-D dim contract (`dim ∈ {-1, 0}`, else error) intact on
//!      CUDA exactly as on CPU.
//!
//! Pre-fix observed behavior (R-AHON-1 probe at HEAD, pasted in #1736):
//! the shared scalar-identity helpers read the scalar via `Tensor::item()`
//! → `Tensor::data()`, which refuses GPU storage — every CUDA scalar call
//! failed with `GpuTensorNotAccessible` (an internal accessor error, not a
//! structured public contract) instead of returning the identity the
//! audited revision's `TensorStorage::cpu` path silently demoted to.
//! Either symptom violates the torch contract below.
//!
//! All numerical expectations are pasted from a LIVE `torch==2.11.0+cu130`
//! session on the same device class (RTX 3090) — R-ORACLE-1(b):
//!
//! ```text
//! >>> for name in ["cumsum", "cumprod", "logcumsumexp"]:
//! ...     x = torch.tensor(5.0, device='cuda', requires_grad=True)
//! ...     out = getattr(torch, name)(x, 0)
//! ...     out.backward(torch.tensor(2.5, device='cuda'))
//! cumsum(tensor(5., cuda), 0)       -> 5.0 dev=cuda:0 shape=(); grad=2.5 dev=cuda:0
//! cumprod(tensor(5., cuda), 0)      -> 5.0 dev=cuda:0 shape=(); grad=2.5 dev=cuda:0
//! logcumsumexp(tensor(5., cuda), 0) -> 5.0 dev=cuda:0 shape=(); grad=2.5 dev=cuda:0
//! cummax: values=5.0 dev=cuda:0, indices=0 (int64, cuda:0); grad=2.5 dev=cuda:0
//! cummin: values=5.0 dev=cuda:0, indices=0 (int64, cuda:0); grad=2.5 dev=cuda:0
//! >>> torch.cumsum(torch.tensor(-3.5, device='cuda'), -1)
//! tensor(-3.5000, device='cuda:0')
//! >>> torch.cumsum(torch.tensor(-3.5, device='cuda'), 1)
//! IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
//! >>> torch.cummax(torch.tensor(-3.5, device='cuda'), 1)
//! IndexError: cummax(): Expected reduction dim -1 or 0 for scalar but got 1
//! ```
//!
//! Comparisons are exact (`==`): 5.0 / -3.5 / 2.5 are dyadic rationals and
//! the 0-D paths are pure identity copies (no arithmetic).

#![cfg(feature = "gpu")]

use ferrotorch_core::autograd::graph::backward_with_grad;
use ferrotorch_core::grad_fns::cumulative::{cummax, cummin, cumprod, cumsum, logcumsumexp};
use ferrotorch_core::{Device, FerrotorchError, Float, Tensor, TensorStorage};
use std::sync::Once;

static GPU_INIT: Once = Once::new();
fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the CORE-042 GPU pins");
    });
}

/// Build a CUDA-resident 0-D scalar tensor.
fn scalar_cuda<T: Float>(v: f64, rg: bool) -> Tensor<T> {
    let cast = <T as num_traits::NumCast>::from(v).unwrap();
    Tensor::from_storage(TensorStorage::cpu(vec![cast]), Vec::new(), false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(rg)
}

/// Read any-device tensor back as exact f64 values for oracle comparison.
fn host_f64<T: Float>(t: &Tensor<T>) -> Vec<f64> {
    t.data_vec()
        .unwrap()
        .into_iter()
        .map(|v| <f64 as num_traits::NumCast>::from(v).unwrap())
        .collect()
}

/// Assert a 0-D CUDA-resident scalar with the exact oracle value.
fn assert_scalar_cuda<T: Float>(t: &Tensor<T>, what: &str, expected: f64) {
    assert!(
        t.is_cuda(),
        "{what} must be CUDA-resident (CORE-042: scalar path demotion), got {:?}",
        t.device()
    );
    assert_eq!(t.shape(), &[] as &[usize], "{what} must stay 0-D");
    assert_eq!(host_f64(t), vec![expected], "{what} value vs live torch");
}

fn grad_of<T: Float>(leaf: &Tensor<T>) -> Tensor<T> {
    leaf.grad()
        .unwrap()
        .expect("grad must reach the leaf (R-ORACLE-3)")
}

/// Forward + backward residency for the three single-output identity ops.
fn identity_op_scalar_cuda<T: Float>(
    name: &str,
    op: impl Fn(&Tensor<T>, i64) -> ferrotorch_core::FerrotorchResult<Tensor<T>>,
) {
    ensure_cuda_backend();
    // dim = 0, requires_grad: forward value/device + backward grad device.
    let x = scalar_cuda::<T>(5.0, true);
    let out = op(&x, 0).unwrap_or_else(|e| panic!("{name}(cuda scalar, 0) must succeed: {e}"));
    assert_scalar_cuda(&out, &format!("{name} output"), 5.0);

    let seed = scalar_cuda::<T>(2.5, false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    let gx = grad_of(&x);
    assert_scalar_cuda(&gx, &format!("{name} grad_input"), 2.5);

    // dim = -1, no grad: identity, CUDA-resident, no grad_fn.
    let x2 = scalar_cuda::<T>(-3.5, false);
    let out2 = op(&x2, -1).unwrap();
    assert_scalar_cuda(&out2, &format!("{name} dim=-1 output"), -3.5);
    assert!(out2.grad_fn().is_none());

    // dim = 1: same structured error as on CPU (torch: IndexError).
    let err = op(&x2, 1).unwrap_err();
    assert!(
        matches!(err, FerrotorchError::InvalidArgument { .. }),
        "{name}(cuda scalar, 1) must reject with InvalidArgument, got {err:?}"
    );
    assert!(
        err.to_string().contains("Dimension out of range"),
        "{name} dim error message: {err}"
    );
}

#[test]
fn cumsum_scalar_cuda_resident_f32() {
    identity_op_scalar_cuda::<f32>("cumsum", cumsum);
}

#[test]
fn cumsum_scalar_cuda_resident_f64() {
    identity_op_scalar_cuda::<f64>("cumsum", cumsum);
}

#[test]
fn cumprod_scalar_cuda_resident_f32() {
    identity_op_scalar_cuda::<f32>("cumprod", cumprod);
}

#[test]
fn cumprod_scalar_cuda_resident_f64() {
    identity_op_scalar_cuda::<f64>("cumprod", cumprod);
}

#[test]
fn logcumsumexp_scalar_cuda_resident_f32() {
    identity_op_scalar_cuda::<f32>("logcumsumexp", logcumsumexp);
}

#[test]
fn logcumsumexp_scalar_cuda_resident_f64() {
    identity_op_scalar_cuda::<f64>("logcumsumexp", logcumsumexp);
}

/// Forward + backward residency for the (values, indices) extremum ops.
/// ferrotorch's `indices` carrier is `Vec<usize>` (documented R-DEV
/// deviation — no device dimension); torch returns `tensor(0)` on cuda:0.
fn extremum_op_scalar_cuda<T: Float>(
    name: &str,
    op: impl Fn(
        &Tensor<T>,
        i64,
    ) -> ferrotorch_core::FerrotorchResult<ferrotorch_core::CumExtremeResult<T>>,
) {
    ensure_cuda_backend();
    let x = scalar_cuda::<T>(5.0, true);
    let res = op(&x, 0).unwrap_or_else(|e| panic!("{name}(cuda scalar, 0) must succeed: {e}"));
    assert_scalar_cuda(&res.values, &format!("{name} values"), 5.0);
    assert_eq!(res.indices, vec![0], "{name} indices vs torch tensor(0)");

    let seed = scalar_cuda::<T>(2.5, false);
    backward_with_grad(&res.values, Some(&seed)).unwrap();
    let gx = grad_of(&x);
    assert_scalar_cuda(&gx, &format!("{name} grad_input"), 2.5);

    // dim = -1, no grad.
    let x2 = scalar_cuda::<T>(-3.5, false);
    let res2 = op(&x2, -1).unwrap();
    assert_scalar_cuda(&res2.values, &format!("{name} dim=-1 values"), -3.5);
    assert_eq!(res2.indices, vec![0]);

    // dim = 1: structured error, op-specific message (torch: IndexError:
    // "<op>(): Expected reduction dim -1 or 0 for scalar but got 1").
    let err = op(&x2, 1).unwrap_err();
    assert!(
        matches!(err, FerrotorchError::InvalidArgument { .. }),
        "{name}(cuda scalar, 1) must reject with InvalidArgument, got {err:?}"
    );
    assert!(
        err.to_string()
            .contains("Expected reduction dim -1 or 0 for scalar"),
        "{name} dim error message: {err}"
    );
}

#[test]
fn cummax_scalar_cuda_resident_f32() {
    extremum_op_scalar_cuda::<f32>("cummax", cummax);
}

#[test]
fn cummax_scalar_cuda_resident_f64() {
    extremum_op_scalar_cuda::<f64>("cummax", cummax);
}

#[test]
fn cummin_scalar_cuda_resident_f32() {
    extremum_op_scalar_cuda::<f32>("cummin", cummin);
}

#[test]
fn cummin_scalar_cuda_resident_f64() {
    extremum_op_scalar_cuda::<f64>("cummin", cummin);
}
