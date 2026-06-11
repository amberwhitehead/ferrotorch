//! Red-then-green regression tests for audit finding CORE-139 (crosslink
//! #1833): `broadcast_matmul` panics (remainder-by-zero inside
//! `batch_linear_index`) on zero-sized batch dimensions, where PyTorch
//! returns a correctly-shaped empty tensor.
//!
//! Every shape expectation below is quoted from a LIVE torch session
//! (torch 2.11.0+cu130, R-ORACLE-1 path (b)); the generating snippet is
//! pasted in a comment next to each expected value.
//!
//! torch oracle:
//!
//! ```text
//! >>> (torch.zeros(0,2,3) @ torch.zeros(0,3,2)).shape
//! torch.Size([0, 2, 2])
//! >>> (torch.ones(1,2,3) @ torch.zeros(0,3,2)).shape   # broadcast-produced zero
//! torch.Size([0, 2, 2])
//! >>> (torch.ones(2,3) @ torch.zeros(0,3,4)).shape
//! torch.Size([0, 2, 4])
//! >>> (torch.ones(3) @ torch.zeros(0,3,2)).shape       # 1-D promotion
//! torch.Size([0, 2])
//! >>> (torch.zeros(0,3,2) @ torch.ones(2)).shape
//! torch.Size([0, 3])
//! >>> torch.ones(2,2,0) @ torch.ones(2,0,3)            # k=0, non-zero batch
//! tensor of zeros, shape [2, 2, 3]
//! ```

use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::ops::linalg::matmul as ops_matmul;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn t_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}
fn t_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}
fn leaf_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

// ===========================================================================
// ops::linalg::matmul — the public Result-returning API named by the finding
// ===========================================================================

// torch: (torch.zeros(0,2,3) @ torch.zeros(0,3,2)).shape == [0, 2, 2]
#[test]
fn core139_zero_batch_both_operands_f32() {
    let a = t_f32(&[], &[0, 2, 3]);
    let b = t_f32(&[], &[0, 3, 2]);
    let c = ops_matmul(&a, &b).expect("matmul on zero-sized batch must not panic");
    assert_eq!(c.shape(), &[0, 2, 2], "torch: (0,2,3)@(0,3,2) -> (0,2,2)");
    assert_eq!(c.numel(), 0);
}

#[test]
fn core139_zero_batch_both_operands_f64() {
    let a = t_f64(&[], &[0, 2, 3]);
    let b = t_f64(&[], &[0, 3, 2]);
    let c = ops_matmul(&a, &b).expect("matmul on zero-sized batch must not panic");
    assert_eq!(c.shape(), &[0, 2, 2], "torch: (0,2,3)@(0,3,2) -> (0,2,2)");
    assert_eq!(c.numel(), 0);
}

// torch: (torch.ones(1,2,3) @ torch.zeros(0,3,2)).shape == [0, 2, 2]
// The zero is PRODUCED by broadcasting 1 against 0.
#[test]
fn core139_broadcast_produced_zero_batch_f32() {
    let a = t_f32(&[1.0; 6], &[1, 2, 3]);
    let b = t_f32(&[], &[0, 3, 2]);
    let c = ops_matmul(&a, &b).expect("matmul broadcasting 1 vs 0 must not panic");
    assert_eq!(c.shape(), &[0, 2, 2], "torch: (1,2,3)@(0,3,2) -> (0,2,2)");
    assert_eq!(c.numel(), 0);
}

// torch: (torch.ones(2,3) @ torch.zeros(0,3,4)).shape == [0, 2, 4]
#[test]
fn core139_2d_lhs_zero_batch_rhs_f32() {
    let a = t_f32(&[1.0; 6], &[2, 3]);
    let b = t_f32(&[], &[0, 3, 4]);
    let c = ops_matmul(&a, &b).expect("matmul (2,3)@(0,3,4) must not panic");
    assert_eq!(c.shape(), &[0, 2, 4], "torch: (2,3)@(0,3,4) -> (0,2,4)");
    assert_eq!(c.numel(), 0);
}

// torch: (torch.ones(3) @ torch.zeros(0,3,2)).shape == [0, 2]
#[test]
fn core139_1d_lhs_zero_batch_rhs_f32() {
    let a = t_f32(&[1.0; 3], &[3]);
    let b = t_f32(&[], &[0, 3, 2]);
    let c = ops_matmul(&a, &b).expect("matmul (3,)@(0,3,2) must not panic");
    assert_eq!(c.shape(), &[0, 2], "torch: (3,)@(0,3,2) -> (0,2)");
    assert_eq!(c.numel(), 0);
}

// torch: (torch.zeros(0,3,2) @ torch.ones(2)).shape == [0, 3]
#[test]
fn core139_zero_batch_lhs_1d_rhs_f32() {
    let a = t_f32(&[], &[0, 3, 2]);
    let b = t_f32(&[1.0; 2], &[2]);
    let c = ops_matmul(&a, &b).expect("matmul (0,3,2)@(2,) must not panic");
    assert_eq!(c.shape(), &[0, 3], "torch: (0,3,2)@(2,) -> (0,3)");
    assert_eq!(c.numel(), 0);
}

// Guard (already-correct neighbour case): k=0 with NON-zero batch dims.
// torch: torch.ones(2,2,0) @ torch.ones(2,0,3) == zeros(2,2,3)
#[test]
fn core139_zero_inner_dim_nonzero_batch_f32() {
    let a = t_f32(&[], &[2, 2, 0]);
    let b = t_f32(&[], &[2, 0, 3]);
    let c = ops_matmul(&a, &b).expect("matmul with k=0 must not panic");
    assert_eq!(c.shape(), &[2, 2, 3], "torch: (2,2,0)@(2,0,3) -> (2,2,3)");
    let data = c.data().unwrap();
    assert!(
        data.iter().all(|&v| v == 0.0),
        "torch fills the empty-contraction result with zeros, got {data:?}"
    );
}

// ===========================================================================
// Tensor::matmul — the grad_fns dispatcher route named by the finding
// ===========================================================================

#[test]
fn core139_tensor_matmul_zero_batch_f32() {
    let a = t_f32(&[], &[0, 2, 3]);
    let b = t_f32(&[], &[0, 3, 2]);
    let c = a
        .matmul(&b)
        .expect("Tensor::matmul on zero-sized batch must not panic");
    assert_eq!(c.shape(), &[0, 2, 2], "torch: (0,2,3)@(0,3,2) -> (0,2,2)");
    assert_eq!(c.numel(), 0);
}

// GPU lane: torch returns the same empty (0,2,2) on cuda:0.
//   >>> (torch.zeros(0,2,3,device='cuda') @ torch.zeros(0,3,2,device='cuda')).shape
//   torch.Size([0, 2, 2])
#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the GPU lane of this suite");
        });
    }

    #[test]
    fn core139_gpu_zero_batch_both_operands_f32() {
        ensure_cuda_backend();
        let dev = Device::Cuda(0);
        let a = t_f32(&[], &[0, 2, 3]).to(dev).expect("upload a");
        let b = t_f32(&[], &[0, 3, 2]).to(dev).expect("upload b");
        let c = a
            .matmul(&b)
            .expect("CUDA matmul on zero-sized batch must not panic");
        assert_eq!(c.shape(), &[0, 2, 2], "torch: (0,2,3)@(0,3,2) -> (0,2,2)");
        assert_eq!(c.numel(), 0);
    }
}

// Autograd through the empty matmul: torch supports backward through
// zero-sized batches; the gradients are empty tensors of the input shapes.
//   >>> a = torch.zeros(0,2,3, requires_grad=True)
//   >>> b = torch.zeros(0,3,2, requires_grad=True)
//   >>> (a @ b).sum().backward()
//   >>> a.grad.shape, b.grad.shape
//   (torch.Size([0, 2, 3]), torch.Size([0, 3, 2]))
#[test]
fn core139_backward_through_zero_batch_f32() {
    let a = leaf_f32(&[], &[0, 2, 3]);
    let b = leaf_f32(&[], &[0, 3, 2]);
    let c = a
        .matmul(&b)
        .expect("Tensor::matmul on zero-sized batch must not panic");
    let loss = sum(&c).expect("sum of empty tensor");
    loss.backward().expect("backward through zero-sized batch");
    let ga = a.grad().unwrap().expect("grad_a present");
    let gb = b.grad().unwrap().expect("grad_b present");
    assert_eq!(ga.shape(), &[0, 2, 3], "torch: a.grad.shape == (0,2,3)");
    assert_eq!(gb.shape(), &[0, 3, 2], "torch: b.grad.shape == (0,3,2)");
}
