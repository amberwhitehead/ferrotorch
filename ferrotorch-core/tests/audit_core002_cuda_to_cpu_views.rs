//! Regression tests for CORE-002 / crosslink #1696.
//!
//! CUDA-to-CPU transfer must materialize the logical tensor view, not the
//! underlying storage prefix. Live torch 2.11.0+cu130 oracle:
//!
//! ```text
//! >>> x = torch.tensor([1., 2., 3., 4.], device='cuda')
//! >>> v = torch.as_strided(x, (1,1,1,1,1,1,1,1,2),
//! ...                      (2,2,2,2,2,2,2,2,1), 2)
//! >>> v.cpu().flatten()
//! tensor([3., 4.])
//! ```
//!
//! Tolerance justification: the values are small exactly-representable f32
//! integers moved by copy only, so exact equality is the correct assertion.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::creation::from_vec;
use ferrotorch_core::{Device, Tensor};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the CORE-002 regression suite");
    });
}

fn cuda_f32(data: &[f32]) -> Tensor<f32> {
    from_vec(data.to_vec(), &[data.len()])
        .expect("construct cpu tensor")
        .to(Device::Cuda(0))
        .expect("upload to cuda")
}

#[test]
#[allow(clippy::float_cmp)]
fn cuda_rank9_offset_view_to_cpu_materializes_logical_values() {
    ensure_cuda_backend();
    let x = cuda_f32(&[1.0, 2.0, 3.0, 4.0]);
    let v = x
        .try_stride_view(
            vec![1, 1, 1, 1, 1, 1, 1, 1, 2],
            vec![2, 2, 2, 2, 2, 2, 2, 2, 1],
            2,
        )
        .expect("rank-9 CUDA view");
    assert_eq!(v.shape(), &[1, 1, 1, 1, 1, 1, 1, 1, 2]);
    assert_eq!(v.storage_offset(), 2);
    assert!(v.is_cuda(), "precondition: view must stay CUDA-resident");

    let host = v.to(Device::Cpu).expect("rank-9 CUDA view .cpu()");
    assert_eq!(host.device(), Device::Cpu);
    assert_eq!(host.shape(), v.shape());
    assert_eq!(host.data().expect("host data"), &[3.0, 4.0]);
}
