use ferrotorch_core::autograd::checkpoint::{checkpoint, checkpoint_multi};
use ferrotorch_core::creation::{from_slice, rand, randn};
use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{Tensor, manual_seed};

fn leaf_grad(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

#[test]
fn checkpoint_cpu_rng_recompute_and_restore_are_deterministic() {
    manual_seed(123).unwrap();
    let x = leaf_grad(&[1.0; 6], &[6]);
    let y = checkpoint(
        |t: &Tensor<f32>| {
            let mask = rand::<f32>(t.shape())?;
            mul(t, &mask)
        },
        &x,
    )
    .unwrap();
    let forward_mask = y.data().unwrap().to_vec();
    sum(&y).unwrap().backward().unwrap();
    let grad = x.grad().unwrap().expect("x should have a gradient");
    assert_eq!(
        grad.data().unwrap(),
        forward_mask.as_slice(),
        "checkpoint recompute must reuse the exact CPU uniform RNG stream"
    );

    manual_seed(456).unwrap();
    let _cached_sibling = randn::<f32>(&[1]).unwrap();
    let x = leaf_grad(&[1.0; 5], &[5]);
    let y = checkpoint(
        |t: &Tensor<f32>| {
            let noise = randn::<f32>(t.shape())?;
            mul(t, &noise)
        },
        &x,
    )
    .unwrap();
    let forward_noise = y.data().unwrap().to_vec();
    sum(&y).unwrap().backward().unwrap();
    let grad = x.grad().unwrap().expect("x should have a gradient");
    assert_eq!(
        grad.data().unwrap(),
        forward_noise.as_slice(),
        "checkpoint recompute must preserve cached CPU normal samples"
    );

    manual_seed(789).unwrap();
    let x = leaf_grad(&[1.0; 6], &[6]);
    let y = checkpoint(
        |t: &Tensor<f32>| {
            let mask = rand::<f32>(t.shape())?;
            mul(t, &mask)
        },
        &x,
    )
    .unwrap();
    let after_forward = rand::<f32>(&[4]).unwrap().data().unwrap().to_vec();
    sum(&y).unwrap().backward().unwrap();
    let after_backward = rand::<f32>(&[4]).unwrap().data().unwrap().to_vec();

    manual_seed(789).unwrap();
    let _forward_mask = rand::<f32>(&[6]).unwrap();
    let expected_after_forward = rand::<f32>(&[4]).unwrap().data().unwrap().to_vec();
    let expected_after_backward = rand::<f32>(&[4]).unwrap().data().unwrap().to_vec();
    assert_eq!(after_forward, expected_after_forward);
    assert_eq!(
        after_backward, expected_after_backward,
        "checkpoint recompute should fork RNG and restore the caller stream"
    );

    manual_seed(321).unwrap();
    let a = leaf_grad(&[1.0; 4], &[4]);
    let b = from_slice(&[0.0f32; 4], &[4]).unwrap();
    let y = checkpoint_multi(
        |ts: &[Tensor<f32>]| {
            let mask = rand::<f32>(ts[0].shape())?;
            let masked = mul(&ts[0], &mask)?;
            add(&masked, &ts[1])
        },
        &[a.clone(), b],
    )
    .unwrap();
    let forward_mask = y.data().unwrap().to_vec();
    sum(&y).unwrap().backward().unwrap();
    let grad = a.grad().unwrap().expect("a should have a gradient");
    assert_eq!(
        grad.data().unwrap(),
        forward_mask.as_slice(),
        "checkpoint_multi recompute must reuse the exact CPU RNG stream"
    );
}
