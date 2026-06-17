//! CORE-168 / #1862: the torch-colliding free `ferrotorch_core::einsum`
//! must attach autograd like `torch.einsum`, not return a detached forward
//! value.

use ferrotorch_core::Tensor;
use ferrotorch_core::storage::TensorStorage;

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu tensor")
        .requires_grad_(true)
}

fn plain(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu tensor")
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (idx, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-5,
            "{label}[{idx}]: got {a}, expected {e}, abs diff {}",
            (a - e).abs()
        );
    }
}

#[test]
fn crate_root_free_einsum_tracks_autograd_like_torch() {
    let a = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = leaf(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);

    let out = ferrotorch_core::einsum("ij,jk->ik", &[&a, &b]).expect("free einsum");
    assert!(
        out.requires_grad(),
        "crate-root free einsum must preserve autograd flow"
    );
    assert_eq!(out.shape(), &[2, 2]);

    // PyTorch 2026-06-17:
    // a = torch.tensor([[1,2,3],[4,5,6]], requires_grad=True)
    // b = torch.tensor([[7,8],[9,10],[11,12]], requires_grad=True)
    // torch.einsum("ij,jk->ik", a, b).backward([[1,-2],[3,-4]])
    let seed = plain(&[1.0, -2.0, 3.0, -4.0], &[2, 2]);
    out.backward_with_gradient(&seed).expect("backward");

    let ga = a.grad().expect("a grad result").expect("a grad");
    let gb = b.grad().expect("b grad result").expect("b grad");
    assert_close(
        ga.data().expect("a grad data"),
        &[-9.0, -11.0, -13.0, -11.0, -13.0, -15.0],
        "a grad",
    );
    assert_close(
        gb.data().expect("b grad data"),
        &[13.0, -18.0, 17.0, -24.0, 21.0, -30.0],
        "b grad",
    );
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::{assert_close, plain};
    use ferrotorch_core::Tensor;
    use ferrotorch_core::device::Device;
    use ferrotorch_core::storage::TensorStorage;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for CORE-168 CUDA probes");
        });
    }

    fn cuda_leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        ensure_cuda_backend();
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
            .expect("cpu tensor")
            .to(Device::Cuda(0))
            .expect("upload")
            .detach()
            .requires_grad_(true)
    }

    fn cuda_seed(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        ensure_cuda_backend();
        plain(data, shape).to(Device::Cuda(0)).expect("upload")
    }

    #[test]
    fn crate_root_free_einsum_cuda_tracks_autograd_like_torch() {
        let a = cuda_leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = cuda_leaf(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);

        let out = ferrotorch_core::einsum("ij,jk->ik", &[&a, &b]).expect("free cuda einsum");
        assert_eq!(out.device(), Device::Cuda(0));
        assert!(
            out.requires_grad(),
            "crate-root free CUDA einsum must preserve autograd flow"
        );

        let seed = cuda_seed(&[1.0, -2.0, 3.0, -4.0], &[2, 2]);
        out.backward_with_gradient(&seed).expect("backward");

        let ga = a.grad().expect("a grad result").expect("a grad");
        let gb = b.grad().expect("b grad result").expect("b grad");
        assert_eq!(ga.device(), Device::Cuda(0));
        assert_eq!(gb.device(), Device::Cuda(0));
        assert_close(
            &ga.data_vec().expect("a grad readback"),
            &[-9.0, -11.0, -13.0, -11.0, -13.0, -15.0],
            "cuda a grad",
        );
        assert_close(
            &gb.data_vec().expect("b grad readback"),
            &[13.0, -18.0, 17.0, -24.0, 21.0, -30.0],
            "cuda b grad",
        );
    }
}
