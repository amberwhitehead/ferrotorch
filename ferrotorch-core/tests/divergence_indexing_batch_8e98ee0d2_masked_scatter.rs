//! Divergences/regression-pins in commit `8e98ee0d2` (#1252 masked_scatter).
//!
//! Forward + backward pin: live torch oracle for the broadcast-mask case
//! (mask 1-D + input 2-D → broadcast). Per upstream
//! `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2402-2409
//! Tensor masked_scatter(...) { auto [_mask, _self] = expand_outplace(...);
//!   return _self->clone(...).masked_scatter_(*_mask, source); }`
//! and VJP per `tools/autograd/derivatives.yaml:1105-1108
//!   self:   grad.masked_fill(mask, 0)
//!   source: masked_scatter_backward_symint(grad, mask, source.sym_sizes())`.
//!
//! Live oracle for the broadcast case:
//!   inp = t([[1,2,3],[4,5,6]], rg=T)
//!   mask = t([T, F, T])
//!   src = t([10,20,30,40], rg=T)
//!   out = masked_scatter(inp, mask, src)
//!   -> tensor([[10., 2., 20.], [30., 5., 40.]])
//!   out.sum().backward()
//!   inp.grad = t([[0,1,0],[0,1,0]])
//!   src.grad = t([1,1,1,1])
//!
//! These tests pin the existing behavior. If they fail at HEAD `8e98ee0d2`,
//! that is a divergence; if they pass, they guard against a future
//! regression of the broadcast path.

use ferrotorch_core::bool_tensor::BoolTensor;
use ferrotorch_core::grad_fns::indexing::masked_scatter;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

/// Forward parity pin: mask broadcasts to input shape, source consumed
/// in C-order at every mask-true position.
#[test]
fn masked_scatter_broadcast_mask_forward_pin() {
    let inp = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![2, 3],
        false,
    )
    .unwrap();
    let mask = BoolTensor::from_vec(vec![true, false, true], vec![3]).unwrap();
    let src = Tensor::from_storage(
        TensorStorage::cpu(vec![10.0_f32, 20.0, 30.0, 40.0]),
        vec![4],
        false,
    )
    .unwrap();
    let out = masked_scatter(&inp, &mask, &src).expect("forward must succeed");
    assert_eq!(
        out.data().unwrap(),
        &[10.0_f32, 2.0, 20.0, 30.0, 5.0, 40.0],
        "masked_scatter broadcast + C-order src consumption per upstream oracle"
    );
}

/// Backward parity pin: VJP for input zeros at mask-true positions;
/// VJP for source receives 1 grad per mask-true position (in C-order).
#[test]
fn masked_scatter_broadcast_mask_backward_pin() {
    let inp = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![2, 3],
        true,
    )
    .unwrap();
    let mask = BoolTensor::from_vec(vec![true, false, true], vec![3]).unwrap();
    let src = Tensor::from_storage(
        TensorStorage::cpu(vec![10.0_f32, 20.0, 30.0, 40.0]),
        vec![4],
        true,
    )
    .unwrap();
    let out = masked_scatter(&inp, &mask, &src).expect("forward must succeed");
    let gf = out
        .grad_fn()
        .expect("masked_scatter attaches MaskedScatterBackward when inputs require grad");
    let go = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0]),
        vec![2, 3],
        false,
    )
    .unwrap();
    let grads = gf.backward(&go).expect("backward must succeed");
    let g_input = grads[0].as_ref().expect("Some(grad_input)");
    assert_eq!(
        g_input.data().unwrap(),
        &[0.0_f32, 1.0, 0.0, 0.0, 1.0, 0.0],
        "input VJP zeros at mask-true positions per upstream \
         derivatives.yaml:1106 `grad.masked_fill(mask, 0)`"
    );
    let g_src = grads[1].as_ref().expect("Some(grad_src)");
    assert_eq!(
        g_src.data().unwrap(),
        &[1.0_f32, 1.0, 1.0, 1.0],
        "source VJP collects grad at mask-true positions in C-order, \
         padding tail with zeros per upstream \
         derivatives.yaml:1107 `masked_scatter_backward_symint(...)`"
    );
}

/// Simple non-broadcast pin: live oracle
///   inp=t([1,2,3,4],rg=T); mask=t([T,F,T,F]); src=t([100,200],rg=T)
///   out=masked_scatter(inp,mask,src) -> t([100,2,200,4])
///   out.sum().backward()
///   inp.grad = t([0,1,0,1])
///   src.grad = t([1,1])
#[test]
fn masked_scatter_non_broadcast_backward_pin() {
    let inp = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0]),
        vec![4],
        true,
    )
    .unwrap();
    let mask = BoolTensor::from_vec(vec![true, false, true, false], vec![4]).unwrap();
    let src =
        Tensor::from_storage(TensorStorage::cpu(vec![100.0_f32, 200.0]), vec![2], true).unwrap();
    let out = masked_scatter(&inp, &mask, &src).unwrap();
    assert_eq!(out.data().unwrap(), &[100.0_f32, 2.0, 200.0, 4.0]);
    let gf = out.grad_fn().expect("MaskedScatterBackward");
    let go = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 1.0, 1.0, 1.0]),
        vec![4],
        false,
    )
    .unwrap();
    let grads = gf.backward(&go).unwrap();
    let g_inp = grads[0].as_ref().expect("Some(g_inp)");
    assert_eq!(g_inp.data().unwrap(), &[0.0_f32, 1.0, 0.0, 1.0]);
    let g_src = grads[1].as_ref().expect("Some(g_src)");
    assert_eq!(g_src.data().unwrap(), &[1.0_f32, 1.0]);
}
