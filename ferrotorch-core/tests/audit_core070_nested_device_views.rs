//! Red-then-green regression tests for audit finding CORE-070 (crosslink
//! #1764, CLASS-S): `NestedTensor::new` accepts mixed-device components
//! (which later make every op fail confusingly), and the nested CPU paths
//! reject valid non-contiguous logical views because they materialize via
//! the raw, layout-sensitive `data()`.
//!
//! Observed at HEAD (probe, 2026-06-11):
//! - `NestedTensor::new(vec![cpu, cuda], 0)` returns `Ok`; the subsequent
//!   `to_padded` fails with "cannot access GPU tensor data as CPU slice".
//! - CPU `to_padded`, `NestedTensor::from_padded`, packed `from_padded`,
//!   `PackedNestedTensor::from_nested`, and the CPU attention fallback all
//!   fail with "tensor is not contiguous; call .contiguous() or use
//!   .data_vec()" on valid transpose/narrow views.
//!
//! Post-fix contract:
//! - single-device invariant enforced at construction (`DeviceMismatch`);
//! - CPU paths materialize the LOGICAL view (`data_vec` semantics), so
//!   non-contiguous components and sources round-trip correctly. CUDA
//!   tensors that reach a CPU-only path still error loudly (no silent
//!   host demotion, R-LOUD-1).
//!
//! Expectations are pure data movement (transpose view + scatter into a
//! padded buffer — no float arithmetic), so values are hand-derived from
//! the input literals and asserted bit-exactly; the attention test quotes
//! a live torch SDPA oracle (R-ORACLE-1(b)).

use ferrotorch_core::nested::{
    NestedTensor, PackedNestedTensor, nested_scaled_dot_product_attention,
};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn t_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
}

fn t_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
}

/// `to_padded` must accept a non-contiguous CPU component view.
/// base = [[1,2,3],[4,5,6]]; transpose view = [[1,4],[2,5],[3,6]] (logical).
/// Pure data movement — bit-exact assertion.
#[test]
// reason: to_padded copies component values verbatim and writes the literal
// pad value; no float arithmetic, so bitwise equality is correct.
#[allow(clippy::float_cmp)]
fn core070_to_padded_accepts_noncontiguous_component_view() {
    let base = t_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = base.transpose(0, 1).unwrap(); // [3, 2], non-contiguous
    assert!(!view.is_contiguous(), "precondition: transpose view");
    let other = t_f32(vec![7.0, 8.0], vec![1, 2]);
    let nt = NestedTensor::new(vec![view, other], 0).unwrap();

    let padded = nt.to_padded(0.0).expect("valid logical view must pad");
    assert!(padded.is_cpu(), "CPU inputs produce a CPU padded tensor");
    assert_eq!(padded.shape(), &[2, 3, 2]);
    assert_eq!(
        padded.data().unwrap(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0]
    );
}

/// `NestedTensor::from_padded` must accept a non-contiguous padded source.
#[test]
#[allow(clippy::float_cmp)] // reason: pure data movement, bit-exact.
fn core070_from_padded_accepts_noncontiguous_source() {
    // base [2, 2, 3]; transpose(1, 2) -> logical [2, 3, 2] padded tensor.
    let base = t_f32((0..12).map(|x| x as f32).collect(), vec![2, 2, 3]);
    let src = base.transpose(1, 2).unwrap();
    assert!(!src.is_contiguous(), "precondition: transpose view");
    // logical src: batch 0 = [[0,3],[1,4],[2,5]], batch 1 = [[6,9],[7,10],[8,11]]
    let nt = NestedTensor::from_padded(&src, &[3, 2], 0)
        .expect("valid non-contiguous padded source must slice");
    assert_eq!(nt.tensors()[0].shape(), &[3, 2]);
    assert_eq!(nt.tensors()[1].shape(), &[2, 2]);
    assert!(nt.tensors()[0].is_cpu());
    assert_eq!(
        nt.tensors()[0].data_vec().unwrap(),
        vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
    );
    assert_eq!(
        nt.tensors()[1].data_vec().unwrap(),
        vec![6.0, 9.0, 7.0, 10.0]
    );
}

/// `PackedNestedTensor::from_nested` must accept non-contiguous components.
#[test]
#[allow(clippy::float_cmp)] // reason: pure data movement, bit-exact.
fn core070_packed_from_nested_accepts_noncontiguous_component() {
    let base = t_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let view = base.transpose(0, 1).unwrap(); // logical [[1,4],[2,5],[3,6]]
    let nt = NestedTensor::new(vec![view], 0).unwrap();
    let packed =
        PackedNestedTensor::from_nested(&nt).expect("valid logical component view must pack");
    assert_eq!(packed.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(packed.length(0), 3);
    assert_eq!(packed.tail_shape(), &[2]);
}

/// Packed `from_padded` must accept a non-contiguous padded source.
#[test]
#[allow(clippy::float_cmp)] // reason: pure data movement, bit-exact.
fn core070_packed_from_padded_accepts_noncontiguous_source() {
    let base = t_f32((0..12).map(|x| x as f32).collect(), vec![2, 2, 3]);
    let src = base.transpose(1, 2).unwrap(); // logical [2, 3, 2]
    let packed = PackedNestedTensor::from_padded(&src, &[3, 1])
        .expect("valid non-contiguous padded source must pack");
    assert_eq!(packed.length(0), 3);
    assert_eq!(packed.length(1), 1);
    assert_eq!(packed.data(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0, 6.0, 9.0]);
}

/// The CPU attention fallback must accept non-contiguous q/k/v views.
///
/// torch oracle (live session, torch 2.11.0+cu130, float64):
///
/// ```python
/// >>> q = torch.tensor([[1.,0.,1.,0.],[0.,1.,0.,1.]], dtype=torch.float64)
/// >>> k = torch.tensor([[1.,1.,0.,0.],[0.,0.,1.,1.],[1.,0.,1.,0.]], dtype=torch.float64)
/// >>> v = torch.tensor([[1.,2.],[3.,4.],[5.,6.]], dtype=torch.float64)
/// >>> torch.nn.functional.scaled_dot_product_attention(q, k, v).flatten().tolist()
/// [3.3555882856328187, 4.35558828563282, 2.698089612856696, 3.698089612856696]
/// ```
#[test]
fn core070_cpu_attention_accepts_noncontiguous_query_view() {
    // q is the transpose view of qT = [[1,0],[0,1],[1,0],[0,1]] ([4, 2]).
    let qt = t_f64(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], vec![4, 2]);
    let q = qt.transpose(0, 1).unwrap(); // logical [[1,0,1,0],[0,1,0,1]]
    assert!(!q.is_contiguous(), "precondition: transpose view");
    let k = t_f64(
        vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        vec![3, 4],
    );
    let v = t_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

    let qn = NestedTensor::new(vec![q], 0).unwrap();
    let kn = NestedTensor::new(vec![k], 0).unwrap();
    let vn = NestedTensor::new(vec![v], 0).unwrap();
    let out = nested_scaled_dot_product_attention(&qn, &kn, &vn)
        .expect("valid logical q view must run attention");
    assert!(out.tensors()[0].is_cpu());
    let got = out.tensors()[0].data_vec().unwrap();
    let want = [
        3.3555882856328187,
        4.35558828563282,
        2.698089612856696,
        3.698089612856696,
    ];
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        // Tolerance: f64 eps (2.2e-16) over 4-term dot products, a 3-way
        // softmax, and a 3-term weighted sum — comfortably within 1e-12.
        assert!((g - w).abs() < 1e-12, "out[{i}]: got {g}, torch oracle {w}");
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::{Device, FerrotorchError};
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the CORE-070 regression suite");
        });
    }

    /// Mixed CPU/CUDA construction must fail AT CONSTRUCTION with a
    /// structured `DeviceMismatch`, not later with a confusing
    /// GPU-data-access error.
    #[test]
    fn core070_mixed_device_construction_rejected() {
        ensure_cuda_backend();
        let cpu = t_f32(vec![1.0, 2.0], vec![1, 2]);
        let gpu = t_f32(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2])
            .to(Device::Cuda(0))
            .unwrap();
        let r = NestedTensor::new(vec![cpu, gpu], 0);
        match r {
            Err(FerrotorchError::DeviceMismatch { .. }) => {}
            other => panic!("mixed-device construction must return DeviceMismatch, got {other:?}"),
        }
    }

    /// Guard: a uniform all-CUDA component list still constructs, and
    /// to_padded stays on-device (R-ORACLE-3).
    #[test]
    #[allow(clippy::float_cmp)] // reason: pure data movement, bit-exact.
    fn core070_uniform_cuda_construction_still_works() {
        ensure_cuda_backend();
        let a = t_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
            .to(Device::Cuda(0))
            .unwrap();
        let b = t_f32(vec![5.0, 6.0], vec![1, 2])
            .to(Device::Cuda(0))
            .unwrap();
        let nt = NestedTensor::new(vec![a, b], 0).expect("uniform CUDA components");
        let padded = nt.to_padded(0.0).unwrap();
        assert_eq!(
            padded.device(),
            Device::Cuda(0),
            "R-ORACLE-3: padded result must stay on CUDA"
        );
        assert_eq!(padded.shape(), &[2, 2, 2]);
        assert_eq!(
            padded.cpu().unwrap().data().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0]
        );
    }
}
