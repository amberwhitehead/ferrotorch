//! CUDA parity probes for tensor_ops view packing and `cdist` batch broadcast.
//!
//! Live torch 2.11.0+cu130 oracles:
//! - `torch.arange(1,13).reshape(4,3)[1:4]`:
//!   - `triu(...,0).flatten()` -> `[4,5,6, 0,8,9, 0,0,12]`
//!   - `tril(...,0).flatten()` -> `[4,0,0, 7,8,0, 10,11,12]`
//!   - `diag(...,0)` -> `[4,8,12]`
//!   - `roll(...,1,0).flatten()` -> `[10,11,12, 4,5,6, 7,8,9]`
//! - `torch.cdist(arange(24).reshape(2,3,4), arange(20).reshape(1,5,4), p=2)`
//!   has shape `[2,3,5]` and the values asserted below.

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, Tensor, TensorStorage, cdist, diag, roll, tril, triu};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu tensor")
}

fn read_cuda(t: &Tensor<f32>) -> Vec<f32> {
    assert!(t.is_cuda(), "result must stay CUDA-resident");
    t.cpu().expect("cpu").data_vec().expect("data")
}

fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-5,
            "idx {i}: got {g}, want {w}; got={got:?}, want={want:?}"
        );
    }
}

#[test]
fn cuda_tensor_ops_pack_storage_offset_views_before_kernel_reads() {
    ensure_cuda();
    let base_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let base = cpu_t(&base_data, &[4, 3])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let view = base.narrow(0, 1, 3).expect("narrow");
    assert!(view.is_cuda());
    assert_eq!(view.storage_offset(), 3);
    assert!(view.is_contiguous());

    assert_eq!(
        read_cuda(&triu(&view, 0).expect("triu")),
        vec![4.0, 5.0, 6.0, 0.0, 8.0, 9.0, 0.0, 0.0, 12.0]
    );
    assert_eq!(
        read_cuda(&tril(&view, 0).expect("tril")),
        vec![4.0, 0.0, 0.0, 7.0, 8.0, 0.0, 10.0, 11.0, 12.0]
    );
    assert_eq!(
        read_cuda(&diag(&view, 0).expect("diag")),
        vec![4.0, 8.0, 12.0]
    );
    assert_eq!(
        read_cuda(&roll(&view, 1, 0).expect("roll")),
        vec![10.0, 11.0, 12.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );
}

#[test]
fn cuda_cdist_broadcasts_leading_batch_dims_and_stays_resident() {
    ensure_cuda();
    let x1 = cpu_t(&(0..24).map(|x| x as f32).collect::<Vec<_>>(), &[2, 3, 4])
        .to(Device::Cuda(0))
        .expect("x1 cuda");
    let x2 = cpu_t(&(0..20).map(|x| x as f32).collect::<Vec<_>>(), &[1, 5, 4])
        .to(Device::Cuda(0))
        .expect("x2 cuda");

    let out = cdist(&x1, &x2, 2.0).expect("cdist");
    assert!(out.is_cuda());
    assert_eq!(out.shape(), &[2, 3, 5]);
    assert_close(
        &read_cuda(&out),
        &[
            0.0, 8.0, 16.0, 24.0, 32.0, 8.0, 0.0, 8.0, 16.0, 24.0, 16.0, 8.0, 0.0, 8.0, 16.0, 24.0,
            16.0, 8.0, 0.0, 8.0, 32.0, 24.0, 16.0, 8.0, 0.0, 40.0, 32.0, 24.0, 16.0, 8.0,
        ],
    );
}
