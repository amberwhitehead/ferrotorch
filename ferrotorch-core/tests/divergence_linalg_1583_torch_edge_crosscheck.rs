//! R-BUILD-4 re-audit of `211b0af56` (#1583) — edge cases the builder's
//! single-square-input FD tests did not cover: NON-square `diagonal`/`diag`
//! (where row/col strides differ) and a SCALAR `addmm` bias broadcast to the
//! whole output (the most aggressive sum-reduce). Per R-CHAR-3 all reference
//! values are from LIVE `torch` float64 (torch 2.11.0+cu130), reproduction
//! pinned in each block.

use ferrotorch_core::Tensor;
use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::linalg as linalg_fwd;
use ferrotorch_core::ops::tensor_ops;
use ferrotorch_core::storage::TensorStorage;

fn leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}
fn no_grad_leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}
fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{label}[{i}]: ferrotorch={a}, torch={e}, diff={}",
            (a - e).abs()
        );
    }
}
fn weighted_backward(out: &Tensor<f64>, weight: &[f64], wshape: &[usize]) {
    let w = no_grad_leaf(weight, wshape);
    let prod = mul(out, &w).unwrap();
    reduce_sum(&prod).unwrap().backward().unwrap();
}

// torch float64:
//   A = arange(1..9).reshape(2,4); d = diagonal(A,1) = [2,7]; w=[10,100]
//   dA = [0,10,0,0, 0,0,100,0]
#[test]
fn diagonal_nonsquare_2x4_offset1_matches_torch() {
    let a = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    let d = linalg_fwd::diagonal(&a, 1).unwrap();
    assert_close(
        &d.data().unwrap().to_vec(),
        &[2.0, 7.0],
        1e-9,
        "diagonal([2,4],1) fwd",
    );
    weighted_backward(&d, &[10.0, 100.0], &[2]);
    let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(
        &g,
        &[0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0],
        1e-9,
        "diagonal([2,4],1) dA vs torch",
    );
}

// torch float64:
//   A = arange(1..9).reshape(2,4); d = diag(A,0) = [1,6]; w=[10,100]
//   dA = [10,0,0,0, 0,100,0,0]
#[test]
fn diag_extract_nonsquare_2x4_matches_torch() {
    let a = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    let d = tensor_ops::diag(&a, 0).unwrap();
    assert_close(
        &d.data().unwrap().to_vec(),
        &[1.0, 6.0],
        1e-9,
        "diag([2,4],0) fwd",
    );
    weighted_backward(&d, &[10.0, 100.0], &[2]);
    let g = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(
        &g,
        &[10.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0],
        1e-9,
        "diag([2,4],0) dA vs torch",
    );
}

// torch float64:
//   bias = scalar 3.0 (0-dim) -> broadcast to [2,2]; m1=[[1,2],[3,4]];
//   m2=I; out=addmm(3.0,m1,m2,beta=0.5,alpha=2.0) = [3.5,5.5,7.5,9.5]
//   dbias = [2.0]  (= beta * sum over all 4 entries of grad=1 => 0.5*4)
#[test]
fn addmm_scalar_bias_full_broadcast_grad_matches_torch() {
    let bias = leaf(&[3.0], &[]); // 0-dim scalar
    let m1 = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let m2 = leaf(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let c = linalg_fwd::addmm(&bias, &m1, &m2, 0.5, 2.0).unwrap();
    assert_close(
        &c.data().unwrap().to_vec(),
        &[3.5, 5.5, 7.5, 9.5],
        1e-9,
        "addmm scalar-bias fwd",
    );
    reduce_sum(&c).unwrap().backward().unwrap();
    let gb = bias.grad().unwrap().unwrap();
    assert_close(
        &gb.data().unwrap().to_vec(),
        &[2.0],
        1e-9,
        "addmm scalar dbias vs torch",
    );
}

// torch float64:
//   addr with beta=2.0: dbias = beta*grad = all 2.0
#[test]
fn addr_beta2_bias_grad_matches_torch() {
    let bias = leaf(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[2, 3]);
    let v1 = leaf(&[1.0, 2.0], &[2]);
    let v2 = leaf(&[1.0, 1.0, 1.0], &[3]);
    let c = linalg_fwd::addr(&bias, &v1, &v2, 2.0, 1.0).unwrap();
    reduce_sum(&c).unwrap().backward().unwrap();
    let gb = bias.grad().unwrap().unwrap().data().unwrap().to_vec();
    assert_close(
        &gb,
        &[2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        1e-9,
        "addr beta=2 dbias vs torch",
    );
}
