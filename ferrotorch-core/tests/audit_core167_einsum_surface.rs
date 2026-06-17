//! CORE-167 / #1861: torch-legal einsum surface that ferrotorch used to reject.

use ferrotorch_core::Tensor;
use ferrotorch_core::einsum::{einsum, einsum_differentiable};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::storage::TensorStorage;

fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).expect("tensor")
}

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).expect("leaf")
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (idx, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-5,
            "idx {idx}: got {a}, expected {e}, abs diff {}",
            (a - e).abs()
        );
    }
}

#[test]
fn uppercase_subscripts_match_torch() {
    // PyTorch 2026-06-17:
    // torch.einsum("AB,BC->AC", [[1,2],[3,4]], [[5,6],[7,8]])
    // -> [[19, 22], [43, 50]]
    let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let out = einsum("AB,BC->AC", &[&a, &b]).expect("uppercase einsum");
    assert_eq!(out.shape(), &[2, 2]);
    assert_close(out.data().expect("data"), &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn shared_label_size_one_broadcast_matches_torch() {
    // PyTorch 2026-06-17:
    // torch.einsum("ij,jk->ik", [[2],[3]], [[1,10],[2,20],[3,30]])
    // -> [[12, 120], [18, 180]]
    let a = t(&[2.0, 3.0], &[2, 1]);
    let b = t(&[1.0, 10.0, 2.0, 20.0, 3.0, 30.0], &[3, 2]);
    let out = einsum("ij,jk->ik", &[&a, &b]).expect("broadcast einsum");
    assert_eq!(out.shape(), &[2, 2]);
    assert_close(out.data().expect("data"), &[12.0, 120.0, 18.0, 180.0]);
}

#[test]
fn ellipsis_batch_permute_matches_torch() {
    // PyTorch 2026-06-17:
    // torch.einsum("...ij->...ji", torch.arange(24.).reshape(2,3,2,2))
    // has shape [2,3,2,2] and flattens as below.
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let x = t(&data, &[2, 3, 2, 2]);
    let out = einsum("...ij->...ji", &[&x]).expect("ellipsis einsum");
    assert_eq!(out.shape(), &[2, 3, 2, 2]);
    assert_close(
        out.data().expect("data"),
        &[
            0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0, 8.0, 10.0, 9.0, 11.0, 12.0, 14.0, 13.0, 15.0,
            16.0, 18.0, 17.0, 19.0, 20.0, 22.0, 21.0, 23.0,
        ],
    );
}

#[test]
fn ellipsis_reduction_matches_torch() {
    // PyTorch 2026-06-17:
    // torch.einsum("...ij->ij", torch.arange(24.).reshape(2,3,2,2))
    // -> [[60, 66], [72, 78]]
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let x = t(&data, &[2, 3, 2, 2]);
    let out = einsum("...ij->ij", &[&x]).expect("ellipsis reduction einsum");
    assert_eq!(out.shape(), &[2, 2]);
    assert_close(out.data().expect("data"), &[60.0, 66.0, 72.0, 78.0]);
}

#[test]
fn nary_forward_and_backward_match_torch() {
    // PyTorch 2026-06-17:
    // l = [[1,2],[3,4]]
    // a = torch.arange(12.).reshape(3,2,2)
    // r = [[5,6],[7,8]]
    // torch.einsum("bn,anm,bm->ba", l, a, r)
    // -> [[62,194,326], [176,596,1016]]
    // out.sum().backward() gives:
    // dl = [[150,216], [204,294]]
    // da = [[[26,30],[38,44]], repeated for a=0..2]
    // dr = [[48,57], [108,129]]
    let l = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let a = leaf(&(0..12).map(|x| x as f32).collect::<Vec<_>>(), &[3, 2, 2]);
    let r = leaf(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let out = einsum_differentiable("bn,anm,bm->ba", &[&l, &a, &r]).expect("nary einsum");
    assert_eq!(out.shape(), &[2, 3]);
    assert_close(
        out.data().expect("data"),
        &[62.0, 194.0, 326.0, 176.0, 596.0, 1016.0],
    );

    let loss = sum(&out).expect("sum loss");
    loss.backward().expect("backward");

    assert_close(
        l.grad()
            .expect("l grad result")
            .expect("l grad")
            .data()
            .expect("l grad data"),
        &[150.0, 216.0, 204.0, 294.0],
    );
    assert_close(
        a.grad()
            .expect("a grad result")
            .expect("a grad")
            .data()
            .expect("a grad data"),
        &[
            26.0, 30.0, 38.0, 44.0, 26.0, 30.0, 38.0, 44.0, 26.0, 30.0, 38.0, 44.0,
        ],
    );
    assert_close(
        r.grad()
            .expect("r grad result")
            .expect("r grad")
            .data()
            .expect("r grad data"),
        &[48.0, 57.0, 108.0, 129.0],
    );
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::{assert_close, einsum, einsum_differentiable, sum, t};
    use ferrotorch_core::Tensor;
    use ferrotorch_core::device::Device;
    use ferrotorch_core::storage::TensorStorage;
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn ensure_cuda() {
        INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for CORE-167 CUDA probes")
        });
    }

    fn cuda_t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        ensure_cuda();
        t(data, shape).to(Device::Cuda(0)).expect("upload")
    }

    fn cuda_leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        ensure_cuda();
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
            .expect("cpu tensor")
            .to(Device::Cuda(0))
            .expect("upload")
            .requires_grad_(true)
    }

    #[test]
    fn cuda_shared_label_size_one_broadcast_stays_resident() {
        let a = cuda_t(&[2.0, 3.0], &[2, 1]);
        let b = cuda_t(&[1.0, 10.0, 2.0, 20.0, 3.0, 30.0], &[3, 2]);
        let out = einsum("ij,jk->ik", &[&a, &b]).expect("cuda broadcast einsum");
        assert_eq!(out.device(), Device::Cuda(0));
        assert_eq!(out.shape(), &[2, 2]);
        assert_close(
            &out.data_vec().expect("readback"),
            &[12.0, 120.0, 18.0, 180.0],
        );
    }

    #[test]
    fn cuda_nary_forward_and_backward_stay_resident() {
        let l = cuda_leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let a = cuda_leaf(&(0..12).map(|x| x as f32).collect::<Vec<_>>(), &[3, 2, 2]);
        let r = cuda_leaf(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        let out = einsum_differentiable("bn,anm,bm->ba", &[&l, &a, &r]).expect("cuda nary einsum");
        assert_eq!(out.device(), Device::Cuda(0));
        assert_eq!(out.shape(), &[2, 3]);
        assert_close(
            &out.data_vec().expect("readback"),
            &[62.0, 194.0, 326.0, 176.0, 596.0, 1016.0],
        );

        sum(&out).expect("sum").backward().expect("backward");
        let dl = l.grad().expect("l grad result").expect("l grad");
        let da = a.grad().expect("a grad result").expect("a grad");
        let dr = r.grad().expect("r grad result").expect("r grad");
        assert_eq!(dl.device(), Device::Cuda(0));
        assert_eq!(da.device(), Device::Cuda(0));
        assert_eq!(dr.device(), Device::Cuda(0));
        assert_close(
            &dl.data_vec().expect("dl readback"),
            &[150.0, 216.0, 204.0, 294.0],
        );
        assert_close(
            &da.data_vec().expect("da readback"),
            &[
                26.0, 30.0, 38.0, 44.0, 26.0, 30.0, 38.0, 44.0, 26.0, 30.0, 38.0, 44.0,
            ],
        );
        assert_close(
            &dr.data_vec().expect("dr readback"),
            &[48.0, 57.0, 108.0, 129.0],
        );
    }
}
