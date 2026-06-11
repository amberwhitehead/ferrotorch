//! Red-then-green regression tests for audit finding CORE-109 (crosslink
//! #1803): every `topk` path constructs the values tensor with
//! `requires_grad = false` and no backward function, silently detaching a
//! gradient-tracking input (CLASS-S). torch's topk values are differentiable:
//! backward scatters the value gradients to the SELECTED input indices
//! (`derivatives.yaml`: `value_selecting_reduction_backward`, i.e.
//! `zeros_like(self).scatter(dim, indices, grad)`).
//!
//! Every numerical expectation below is quoted from a LIVE torch session
//! (torch 2.11.0+cu130, R-ORACLE-1 path (b)); the generating snippet is
//! pasted next to each test. All assertions are exact: the backward is a
//! pure scatter of the cotangent values (no arithmetic), so the gradients
//! are bit-identical to the weights.

use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_core::topk;

#[cfg(feature = "gpu")]
use ferrotorch_core::Device;

fn leaf_f32(data: &[f32], shape: &[usize], rg: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), rg).unwrap()
}

// torch oracle (torch 2.11.0+cu130):
//   >>> x = torch.tensor([3.,1.,4.,1.,5.,9.], requires_grad=True)
//   >>> v, i = torch.topk(x, 3)
//   >>> v, i  # tensor([9., 5., 4.], grad_fn=<TopkBackward0>), tensor([5, 4, 2])
//   >>> (v * torch.tensor([10.,20.,30.])).sum().backward()
//   >>> x.grad  # tensor([ 0.,  0., 30.,  0., 20., 10.])
#[test]
fn core109_topk_backward_1d_weighted_cotangent_cpu() {
    let x = leaf_f32(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[6], true);
    let (v, i) = topk(&x, 3, true).expect("topk forward");
    assert_eq!(v.data().unwrap(), &[9.0, 5.0, 4.0], "forward values");
    assert_eq!(i, vec![5, 4, 2], "forward indices");
    assert!(
        v.requires_grad(),
        "torch topk values carry TopkBackward0; detached values cannot train top-k losses"
    );
    let w = leaf_f32(&[10.0, 20.0, 30.0], &[3], false);
    let loss = sum(&mul(&v, &w).expect("weight")).expect("sum");
    loss.backward().expect("backward");
    // R-ORACLE-3: assert gradient FLOW — values reaching the leaf.
    let g = x.grad().unwrap().expect("x.grad present");
    assert_eq!(
        g.data().unwrap(),
        &[0.0, 0.0, 30.0, 0.0, 20.0, 10.0],
        "torch oracle x.grad"
    );
}

// torch oracle:
//   >>> x2 = torch.tensor([[1.,5.,2.],[7.,0.,7.]], requires_grad=True)
//   >>> v, i = torch.topk(x2, 2)
//   >>> v, i  # tensor([[5., 2.], [7., 7.]]), tensor([[1, 2], [0, 2]])
//   >>> (v * torch.tensor([[1.,2.],[3.,4.]])).sum().backward()
//   >>> x2.grad  # tensor([[0., 1., 2.], [3., 0., 4.]])
#[test]
fn core109_topk_backward_2d_cpu() {
    let x = leaf_f32(&[1.0, 5.0, 2.0, 7.0, 0.0, 7.0], &[2, 3], true);
    let (v, i) = topk(&x, 2, true).expect("topk forward");
    assert_eq!(v.data().unwrap(), &[5.0, 2.0, 7.0, 7.0], "forward values");
    assert_eq!(i, vec![1, 2, 0, 2], "forward indices (per-row)");
    let w = leaf_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let loss = sum(&mul(&v, &w).expect("weight")).expect("sum");
    loss.backward().expect("backward");
    let g = x.grad().unwrap().expect("x.grad present");
    assert_eq!(
        g.data().unwrap(),
        &[0.0, 1.0, 2.0, 3.0, 0.0, 4.0],
        "torch oracle x2.grad"
    );
}

// Tie case — torch propagates the gradient to the RETURNED index only:
//   >>> xt = torch.tensor([2.,2.,1.], requires_grad=True)
//   >>> v, i = torch.topk(xt, 1)
//   >>> i  # tensor([0])
//   >>> v.sum().backward()
//   >>> xt.grad  # tensor([1., 0., 0.])  (NOT split between the tied 2.0s)
#[test]
fn core109_topk_backward_tie_goes_to_returned_index_cpu() {
    let x = leaf_f32(&[2.0, 2.0, 1.0], &[3], true);
    let (v, i) = topk(&x, 1, true).expect("topk forward");
    assert_eq!(i, vec![0], "torch returns the first of the tied maxima");
    assert_eq!(v.data().unwrap(), &[2.0]);
    let loss = sum(&v).expect("sum");
    loss.backward().expect("backward");
    let g = x.grad().unwrap().expect("x.grad present");
    assert_eq!(
        g.data().unwrap(),
        &[1.0, 0.0, 0.0],
        "grad must land on the SELECTED index only (torch oracle [1., 0., 0.])"
    );
}

// largest=False — torch oracle:
//   >>> xs = torch.tensor([3.,1.,4.], requires_grad=True)
//   >>> v, i = torch.topk(xs, 2, largest=False)
//   >>> v, i  # tensor([1., 3.]), tensor([1, 0])
//   >>> (v * torch.tensor([5.,7.])).sum().backward()
//   >>> xs.grad  # tensor([7., 5., 0.])
#[test]
fn core109_topk_backward_smallest_cpu() {
    let x = leaf_f32(&[3.0, 1.0, 4.0], &[3], true);
    let (v, i) = topk(&x, 2, false).expect("topk forward");
    assert_eq!(v.data().unwrap(), &[1.0, 3.0], "forward values");
    assert_eq!(i, vec![1, 0], "forward indices");
    let w = leaf_f32(&[5.0, 7.0], &[2], false);
    let loss = sum(&mul(&v, &w).expect("weight")).expect("sum");
    loss.backward().expect("backward");
    let g = x.grad().unwrap().expect("x.grad present");
    assert_eq!(g.data().unwrap(), &[7.0, 5.0, 0.0], "torch oracle xs.grad");
}

// k=0 edge — torch still connects the (empty) values to the graph:
//   >>> xk = torch.tensor([1.,2.], requires_grad=True)
//   >>> v, i = torch.topk(xk, 0)
//   >>> v.shape, v.requires_grad  # (torch.Size([0]), True)
//   >>> v.sum().backward()
//   >>> xk.grad  # tensor([0., 0.])
#[test]
fn core109_topk_backward_k_zero_cpu() {
    let x = leaf_f32(&[1.0, 2.0], &[2], true);
    let (v, _i) = topk(&x, 0, true).expect("topk forward k=0");
    assert_eq!(v.shape(), &[0]);
    assert!(
        v.requires_grad(),
        "torch topk(k=0) values still carry TopkBackward0"
    );
    let loss = sum(&v).expect("sum of empty");
    loss.backward().expect("backward");
    let g = x.grad().unwrap().expect("x.grad present");
    assert_eq!(g.data().unwrap(), &[0.0, 0.0], "torch oracle xk.grad");
}

/// Non-tracking input stays detached (no spurious graph nodes).
#[test]
fn core109_topk_no_grad_input_stays_detached() {
    let x = leaf_f32(&[3.0, 1.0, 4.0], &[3], false);
    let (v, _i) = topk(&x, 2, true).expect("topk forward");
    assert!(!v.requires_grad());
    assert!(v.grad_fn().is_none());
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the GPU lane of this suite");
        });
    }

    // torch oracle (cuda):
    //   >>> xc = torch.tensor([3.,1.,4.,1.,5.,9.], device='cuda', requires_grad=True)
    //   >>> v, i = torch.topk(xc, 3)
    //   >>> (v * torch.tensor([10.,20.,30.], device='cuda')).sum().backward()
    //   >>> xc.grad         # tensor([ 0.,  0., 30.,  0., 20., 10.], device='cuda:0')
    //   >>> xc.grad.device  # device(type='cuda', index=0)
    #[test]
    fn core109_gpu_topk_backward_f32() {
        ensure_cuda_backend();
        let dev = Device::Cuda(0);
        let x = leaf_f32(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[6], true)
            .to(dev)
            .expect("upload x")
            .detach()
            .requires_grad_(true);
        let (v, i) = topk(&x, 3, true).expect("topk forward cuda");
        // R-ORACLE-3 / post-#1890: values stay CUDA-resident.
        assert_eq!(v.device(), dev, "values must be CUDA-resident");
        assert_eq!(i, vec![5, 4, 2], "forward indices");
        assert!(v.requires_grad(), "cuda topk values must track");
        let w = leaf_f32(&[10.0, 20.0, 30.0], &[3], false)
            .to(dev)
            .expect("upload w");
        let loss = sum(&mul(&v, &w).expect("weight")).expect("sum");
        loss.backward().expect("backward cuda");
        let g = x.grad().unwrap().expect("x.grad present");
        assert_eq!(g.device(), dev, "grad must be CUDA-resident");
        let g_cpu = g.cpu().expect("D2H readback");
        assert_eq!(
            g_cpu.data().unwrap(),
            &[0.0, 0.0, 30.0, 0.0, 20.0, 10.0],
            "torch oracle xc.grad"
        );
    }
}
