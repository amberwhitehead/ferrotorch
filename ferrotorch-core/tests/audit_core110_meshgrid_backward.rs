//! Red-then-green regression tests for audit finding CORE-110 (crosslink
//! #1804): both the CPU and CUDA `meshgrid` paths build fresh tensors with
//! `requires_grad = false` regardless of whether the coordinate inputs track
//! gradients — a silent autograd detach (CLASS-S). In torch, grid `i` stays
//! connected to coordinate tensor `i` (upstream `meshgrid` is
//! `view(view_shape).expand(shape)` per input,
//! `aten/src/ATen/native/TensorShape.cpp:4462-4467`, so each grid carries
//! `ExpandBackward0`); the backward reduces the grid gradient over every
//! axis except the grid's own.
//!
//! Every numerical expectation below is quoted from a LIVE torch session
//! (torch 2.11.0+cu130, R-ORACLE-1 path (b)); the generating snippet is
//! pasted next to each test. All assertions are exact: the backward is a
//! plain sum of small integer-valued weights (<= 4 f32 addends per output,
//! all exactly representable), so no tolerance is needed.

use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_core::{MeshIndexing, meshgrid, meshgrid_indexing};

#[cfg(feature = "gpu")]
use ferrotorch_core::Device;

fn leaf_f32(data: &[f32], shape: &[usize], rg: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), rg).unwrap()
}

// torch oracle (torch 2.11.0+cu130):
//   >>> a = torch.tensor([1.,2.,3.], requires_grad=True)
//   >>> b = torch.tensor([4.,5.], requires_grad=True)
//   >>> ga, gb = torch.meshgrid(a, b, indexing='ij')
//   >>> ga.grad_fn, gb.grad_fn  # <ExpandBackward0>, <ExpandBackward0>
//   >>> wa = torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
//   >>> wb = torch.tensor([[10.,20.],[30.,40.],[50.,60.]])
//   >>> ((ga*wa).sum() + (gb*wb).sum()).backward()
//   >>> a.grad  # tensor([ 3.,  7., 11.])   (row sums of wa)
//   >>> b.grad  # tensor([ 90., 120.])      (column sums of wb)
#[test]
fn core110_meshgrid_ij_backward_cpu() {
    let a = leaf_f32(&[1.0, 2.0, 3.0], &[3], true);
    let b = leaf_f32(&[4.0, 5.0], &[2], true);
    let grids = meshgrid(&[a.clone(), b.clone()]).expect("meshgrid forward");
    assert!(
        grids[0].requires_grad() && grids[1].requires_grad(),
        "torch meshgrid grids stay connected to their coordinate tensors"
    );
    let wa = leaf_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], false);
    let wb = leaf_f32(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[3, 2], false);
    let la = sum(&mul(&grids[0], &wa).expect("ga*wa")).expect("sum");
    let lb = sum(&mul(&grids[1], &wb).expect("gb*wb")).expect("sum");
    let loss = add(&la, &lb).expect("loss");
    loss.backward().expect("backward");
    // R-ORACLE-3: gradient FLOW to the leaves, not requires_grad flags.
    let ga = a.grad().unwrap().expect("a.grad present");
    let gb = b.grad().unwrap().expect("b.grad present");
    assert_eq!(ga.data().unwrap(), &[3.0, 7.0, 11.0], "torch oracle a.grad");
    assert_eq!(gb.data().unwrap(), &[90.0, 120.0], "torch oracle b.grad");
}

// torch oracle ('xy' swaps the first two grids; backward must still route
// each grid's gradient to ITS coordinate tensor):
//   >>> a2 = torch.tensor([1.,2.,3.], requires_grad=True)
//   >>> b2 = torch.tensor([4.,5.], requires_grad=True)
//   >>> gx, gy = torch.meshgrid(a2, b2, indexing='xy')   # shapes [2, 3]
//   >>> wx = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
//   >>> wy = torch.tensor([[10.,20.,30.],[40.,50.,60.]])
//   >>> ((gx*wx).sum() + (gy*wy).sum()).backward()
//   >>> a2.grad  # tensor([5., 7., 9.])     (column sums of wx)
//   >>> b2.grad  # tensor([ 60., 150.])     (row sums of wy)
#[test]
fn core110_meshgrid_xy_backward_cpu() {
    let a = leaf_f32(&[1.0, 2.0, 3.0], &[3], true);
    let b = leaf_f32(&[4.0, 5.0], &[2], true);
    let grids =
        meshgrid_indexing(&[a.clone(), b.clone()], MeshIndexing::Xy).expect("meshgrid xy forward");
    assert_eq!(grids[0].shape(), &[2, 3], "xy grid shape");
    let wx = leaf_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let wy = leaf_f32(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3], false);
    let lx = sum(&mul(&grids[0], &wx).expect("gx*wx")).expect("sum");
    let ly = sum(&mul(&grids[1], &wy).expect("gy*wy")).expect("sum");
    let loss = add(&lx, &ly).expect("loss");
    loss.backward().expect("backward");
    let ga = a.grad().unwrap().expect("a2.grad present");
    let gb = b.grad().unwrap().expect("b2.grad present");
    assert_eq!(ga.data().unwrap(), &[5.0, 7.0, 9.0], "torch oracle a2.grad");
    assert_eq!(gb.data().unwrap(), &[60.0, 150.0], "torch oracle b2.grad");
}

// Mixed tracking — torch oracle:
//   >>> a3 = torch.tensor([1.,2.], requires_grad=True)
//   >>> b3 = torch.tensor([7.])
//   >>> g3a, g3b = torch.meshgrid(a3, b3, indexing='ij')
//   >>> g3a.requires_grad, g3b.requires_grad  # (True, False)
#[test]
fn core110_meshgrid_mixed_tracking_cpu() {
    let a = leaf_f32(&[1.0, 2.0], &[2], true);
    let b = leaf_f32(&[7.0], &[1], false);
    let grids = meshgrid(&[a, b]).expect("meshgrid forward");
    assert!(
        grids[0].requires_grad(),
        "grid of the tracking input must track"
    );
    assert!(
        !grids[1].requires_grad() && grids[1].grad_fn().is_none(),
        "grid of the non-tracking input must stay honestly detached"
    );
}

// Singleton + EMPTY axis — torch oracle:
//   >>> s = torch.tensor([5.], requires_grad=True)
//   >>> e = torch.tensor([], requires_grad=True)
//   >>> gs, ge = torch.meshgrid(s, e, indexing='ij')   # shapes [1, 0]
//   >>> (gs.sum() + ge.sum()).backward()
//   >>> s.grad  # tensor([0.])   (sum over an empty axis)
//   >>> e.grad  # tensor([])     (shape [0])
#[test]
fn core110_meshgrid_singleton_and_empty_axis_backward_cpu() {
    let s = leaf_f32(&[5.0], &[1], true);
    let e = leaf_f32(&[], &[0], true);
    let grids = meshgrid(&[s.clone(), e.clone()]).expect("meshgrid forward");
    assert_eq!(grids[0].shape(), &[1, 0], "grid shape");
    let ls = sum(&grids[0]).expect("sum gs");
    let le = sum(&grids[1]).expect("sum ge");
    let loss = add(&ls, &le).expect("loss");
    loss.backward().expect("backward");
    let gs = s.grad().unwrap().expect("s.grad present");
    assert_eq!(gs.data().unwrap(), &[0.0], "torch oracle s.grad");
    let ge = e.grad().unwrap().expect("e.grad present");
    assert_eq!(ge.numel(), 0, "torch oracle e.grad shape [0]");
}

// Three tensors — torch oracle:
//   >>> t1 = torch.tensor([1.,2.], requires_grad=True)
//   >>> t2 = torch.tensor([3.,4.,5.], requires_grad=True)
//   >>> t3 = torch.tensor([6.,7.], requires_grad=True)
//   >>> g1, g2, g3 = torch.meshgrid(t1, t2, t3, indexing='ij')
//   >>> (g1.sum() + 2*g2.sum() + 3*g3.sum()).backward()
//   >>> t1.grad  # tensor([6., 6.])
//   >>> t2.grad  # tensor([8., 8., 8.])
//   >>> t3.grad  # tensor([18., 18.])
#[test]
fn core110_meshgrid_three_tensors_backward_cpu() {
    let t1 = leaf_f32(&[1.0, 2.0], &[2], true);
    let t2 = leaf_f32(&[3.0, 4.0, 5.0], &[3], true);
    let t3 = leaf_f32(&[6.0, 7.0], &[2], true);
    let grids = meshgrid(&[t1.clone(), t2.clone(), t3.clone()]).expect("meshgrid forward");
    let two = leaf_f32(&[2.0; 12], &[2, 3, 2], false);
    let three = leaf_f32(&[3.0; 12], &[2, 3, 2], false);
    let l1 = sum(&grids[0]).expect("sum g1");
    let l2 = sum(&mul(&grids[1], &two).expect("2*g2")).expect("sum");
    let l3 = sum(&mul(&grids[2], &three).expect("3*g3")).expect("sum");
    let loss = add(&add(&l1, &l2).expect("l1+l2"), &l3).expect("loss");
    loss.backward().expect("backward");
    assert_eq!(
        t1.grad().unwrap().expect("t1.grad").data().unwrap(),
        &[6.0, 6.0],
        "torch oracle t1.grad"
    );
    assert_eq!(
        t2.grad().unwrap().expect("t2.grad").data().unwrap(),
        &[8.0, 8.0, 8.0],
        "torch oracle t2.grad"
    );
    assert_eq!(
        t3.grad().unwrap().expect("t3.grad").data().unwrap(),
        &[18.0, 18.0],
        "torch oracle t3.grad"
    );
}

// Single tensor with 'xy' (no swap below 2 inputs) — torch oracle:
//   >>> u = torch.tensor([1.,2.,3.], requires_grad=True)
//   >>> gu, = torch.meshgrid(u, indexing='xy')
//   >>> (gu * torch.tensor([2.,3.,4.])).sum().backward()
//   >>> u.grad  # tensor([2., 3., 4.])
#[test]
fn core110_meshgrid_single_tensor_xy_backward_cpu() {
    let u = leaf_f32(&[1.0, 2.0, 3.0], &[3], true);
    let grids =
        meshgrid_indexing(std::slice::from_ref(&u), MeshIndexing::Xy).expect("meshgrid forward");
    let w = leaf_f32(&[2.0, 3.0, 4.0], &[3], false);
    let loss = sum(&mul(&grids[0], &w).expect("gu*w")).expect("sum");
    loss.backward().expect("backward");
    assert_eq!(
        u.grad().unwrap().expect("u.grad").data().unwrap(),
        &[2.0, 3.0, 4.0],
        "torch oracle u.grad"
    );
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
    //   >>> ac = torch.tensor([1.,2.,3.], device='cuda', requires_grad=True)
    //   >>> bc = torch.tensor([4.,5.], device='cuda', requires_grad=True)
    //   >>> gac, gbc = torch.meshgrid(ac, bc, indexing='ij')
    //   >>> ((gac*wa.cuda()).sum() + (gbc*wb.cuda()).sum()).backward()
    //   >>> ac.grad  # tensor([ 3.,  7., 11.], device='cuda:0')
    //   >>> bc.grad  # tensor([ 90., 120.], device='cuda:0')
    #[test]
    fn core110_gpu_meshgrid_ij_backward_f32() {
        ensure_cuda_backend();
        let dev = Device::Cuda(0);
        let a = leaf_f32(&[1.0, 2.0, 3.0], &[3], true)
            .to(dev)
            .expect("upload a");
        let b = leaf_f32(&[4.0, 5.0], &[2], true).to(dev).expect("upload b");
        let grids = meshgrid(&[a.clone(), b.clone()]).expect("meshgrid forward cuda");
        // R-ORACLE-3 / post-#1890: forward grids stay CUDA-resident.
        assert_eq!(grids[0].device(), dev, "grid0 must be CUDA-resident");
        assert_eq!(grids[1].device(), dev, "grid1 must be CUDA-resident");
        assert!(
            grids[0].requires_grad() && grids[1].requires_grad(),
            "cuda meshgrid grids must track"
        );
        let wa = leaf_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], false)
            .to(dev)
            .expect("upload wa");
        let wb = leaf_f32(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[3, 2], false)
            .to(dev)
            .expect("upload wb");
        let la = sum(&mul(&grids[0], &wa).expect("ga*wa")).expect("sum");
        let lb = sum(&mul(&grids[1], &wb).expect("gb*wb")).expect("sum");
        let loss = add(&la, &lb).expect("loss");
        loss.backward().expect("backward cuda");
        let ga = a.grad().unwrap().expect("a.grad present");
        let gb = b.grad().unwrap().expect("b.grad present");
        assert_eq!(ga.device(), dev, "a.grad must be CUDA-resident");
        assert_eq!(gb.device(), dev, "b.grad must be CUDA-resident");
        assert_eq!(
            ga.cpu().expect("D2H").data().unwrap(),
            &[3.0, 7.0, 11.0],
            "torch oracle ac.grad"
        );
        assert_eq!(
            gb.cpu().expect("D2H").data().unwrap(),
            &[90.0, 120.0],
            "torch oracle bc.grad"
        );
    }
}
